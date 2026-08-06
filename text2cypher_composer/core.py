from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .cypher_utils import CypherExecutionError, execute_cypher, normalize_generated_cypher
from .graph_db import DatabaseLike, resolve_database
from .llm import ModelLike, resolve_model, resolve_pruning_model
from .prompts import messages_for
from .rag import RAGDataset
from .rescue import build_error_message, needs_rescue, rescue_messages
from .schema_modes import resolve_schema_text
from .techniques import (
    RAG_TECHNIQUES,
    SCHEMA_TECHNIQUES,
    OUTPUT_AUGMENTED_TECHNIQUES,
    SchemaMode,
    SchemaModeLike,
    Technique,
)
from .validation import CypherValidationReport, validate_cypher

DatasetLike = Union[RAGDataset, str]


@dataclass
class Text2CypherResult:
    """The outcome of translating and executing one natural-language question.

    `prompt` is the exact list of messages sent to the model, with all
    placeholders (schema, examples, question) already substituted in — for
    the *initial* generation attempt; if rescued, `cypher` reflects the
    fix-up attempt but `prompt` does not. `validation` (a CyVer report) is
    always populated, for every query, whether it executed successfully or
    not, and always reflects the final (possibly rescued) attempt. `result`
    holds the query rows if it executed, and is None otherwise.
    """

    question: str
    technique: str
    model: str
    cypher: Optional[str]
    initial_cypher: Optional[str]
    prompt: List[Dict[str, str]]
    executed: bool
    dry_run: bool = False
    rescued: bool = False
    rescue_attempts: int = 0
    result: Optional[List[Dict[str, Any]]] = None
    validation: Optional[CypherValidationReport] = None
    schema: Optional[str] = None
    retrieved_examples: Optional[Dict[str, Any]] = None


def _resolve_dataset(dataset: DatasetLike) -> RAGDataset:
    if isinstance(dataset, RAGDataset):
        return dataset
    if isinstance(dataset, str):
        return RAGDataset.from_root(dataset)
    raise TypeError(
        "`dataset` must be a RAGDataset instance or a path to a bio2C-style "
        "benchmark root (containing chroma_db/, CypherQueries/, Neo4jOutputs/)."
    )


def run(
    input_NL: str,
    model: ModelLike,
    database: DatabaseLike,
    technique: str,
    dataset: Optional[DatasetLike] = None,
    schema_mode: Optional[SchemaModeLike] = None,
    nlp: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    rescue_prompt: bool = False,
    max_retries: int = 1,
    dry_run: bool = False,
) -> Text2CypherResult:
    """Translate a natural-language question into Cypher and run it.

    Args:
        input_NL: the natural language question.
        model: an OpenAI model id (e.g. "gpt-4o", "gpt-4o-mini", or a
            fine-tuned "ft:..." id), or a LangChain-compatible chat model /
            Runnable (e.g. a HuggingFacePipeline wrapping a local LLaMA
            checkpoint) for anything else.
        database: a Neo4jGraph instance, or a dict with
            uri/username/password/database keys (an empty dict `{}` falls
            back to the NEO4J_* environment variables).
        technique: one of "vanilla", "Schema", "RAG", "RAG+O",
            "Schema+RAG", "Schema+RAG+O".
        dataset: the vector store used for retrieval — a RAGDataset, or a
            path to a bio2C-style benchmark root. Required when `technique`
            uses RAG, and disallowed otherwise.
        schema_mode: how the schema is derived/pruned — one of "schema"
            (default, plain), "enhanced" (with per-property stats),
            "exact_match", "ner_exact_match", "similarity", or
            "llm_pruning" (see `SchemaMode`). Only meaningful — and only
            allowed — when `technique` uses the schema.
        nlp: a loaded NLP pipeline (e.g. a spaCy `Language`), required by
            `schema_mode="ner_exact_match"` (needs named-entity recognition,
            e.g. spaCy's `en_ner_bionlp13cg_md`) and `"similarity"` (needs
            word vectors, e.g. spaCy's `en_core_web_md`). Not bundled by
            this library — bring your own, already loaded.
        similarity_threshold: word-vector similarity cutoff used by
            `schema_mode="similarity"`. Ignored otherwise.
        rescue_prompt: if True, a query that fails to execute or comes back
            empty is retried with a second "fix this query" prompt — reusing
            the same schema/examples context as `technique`, plus the bad
            query and an error message built from CyVer's validation report
            (both its warning-level notifications and hard errors).
        max_retries: how many rescue attempts to make (stopping early once
            one succeeds) before giving up. Only relevant when
            `rescue_prompt` is True.
        dry_run: if True, build and return the fully-instantiated `prompt`
            (schema resolved, RAG examples retrieved, exactly as it would be
            for a real call) but stop there — no generation call, no Cypher
            execution, no CyVer validation, no rescue. `cypher`,
            `initial_cypher`, `result`, and `validation` are all `None`, and
            `executed` is `False`. Incompatible with `rescue_prompt=True`
            (there's nothing generated to rescue).

    Returns:
        A Text2CypherResult. `prompt` holds the fully-instantiated messages
        sent to the model for the initial attempt; `initial_cypher` is what
        it generated, and `cypher` is the final one — identical to
        `initial_cypher` unless `rescued` is True. `validation` always holds
        a CyVer report (syntax validity, schema-alignment score,
        property-access score, plus per-issue metadata) for the final
        Cypher, regardless of whether it executed. `result` holds the query
        rows if it did. If `dry_run` is True, only `prompt` (and `schema`/
        `retrieved_examples`, if applicable) are populated.
    """
    technique = Technique(technique)
    uses_rag = technique in RAG_TECHNIQUES
    uses_schema = technique in SCHEMA_TECHNIQUES

    if uses_rag and dataset is None:
        raise ValueError(f"technique='{technique.value}' requires a `dataset` (vector store) for RAG retrieval.")
    if not uses_rag and dataset is not None:
        raise ValueError(f"`dataset` was provided but technique='{technique.value}' does not use RAG.")
    if not uses_schema and (schema_mode is not None or nlp is not None):
        raise ValueError(
            f"`schema_mode`/`nlp` were provided but technique='{technique.value}' does not use the schema."
        )
    if max_retries < 1:
        raise ValueError("`max_retries` must be >= 1.")
    if dry_run and rescue_prompt:
        raise ValueError("`dry_run` and `rescue_prompt` are incompatible: dry_run generates nothing to rescue.")

    graph = resolve_database(database)
    llm = resolve_model(model)

    schema_text = None
    if uses_schema:
        resolved_mode = SchemaMode(schema_mode) if schema_mode is not None else SchemaMode.SCHEMA
        pruning_llm = resolve_pruning_model(model) if resolved_mode == SchemaMode.LLM_PRUNING else None
        schema_text = resolve_schema_text(
            graph,
            resolved_mode,
            input_NL,
            llm=pruning_llm,
            nlp=nlp,
            similarity_threshold=similarity_threshold,
        )

    retrieved = None
    format_kwargs: Dict[str, Any] = {"question": input_NL}
    if uses_schema:
        format_kwargs["enhanced_schema"] = schema_text
    if uses_rag:
        rag_dataset = _resolve_dataset(dataset)
        with_output = technique in OUTPUT_AUGMENTED_TECHNIQUES
        retrieved = rag_dataset.retrieve_examples(input_NL, with_output=with_output)
        format_kwargs["examples"] = retrieved["examples_text"]

    prompt_template = ChatPromptTemplate.from_messages(messages_for(technique))
    formatted_messages = prompt_template.format_messages(**format_kwargs)
    prompt_messages = [{"role": m.type, "content": m.content} for m in formatted_messages]

    if dry_run:
        return Text2CypherResult(
            question=input_NL,
            technique=technique.value,
            model=model if isinstance(model, str) else type(model).__name__,
            cypher=None,
            initial_cypher=None,
            prompt=prompt_messages,
            executed=False,
            dry_run=True,
            schema=schema_text,
            retrieved_examples=retrieved,
        )

    chain = prompt_template | llm | StrOutputParser()
    cypher = normalize_generated_cypher(chain.invoke(format_kwargs))
    initial_cypher = cypher

    validation = validate_cypher(graph, cypher)
    try:
        result = execute_cypher(graph, cypher)
        executed = True
    except CypherExecutionError:
        result = None
        executed = False

    rescue_attempts = 0
    if rescue_prompt and needs_rescue(executed, result, validation):
        rescue_chain = ChatPromptTemplate.from_messages(rescue_messages(uses_schema, uses_rag)) | llm | StrOutputParser()
        for _ in range(max_retries):
            rescue_attempts += 1
            rescue_kwargs = {
                **format_kwargs,
                "query": cypher,
                "error_message": build_error_message(executed, result, validation),
            }
            cypher = normalize_generated_cypher(rescue_chain.invoke(rescue_kwargs))
            validation = validate_cypher(graph, cypher)
            try:
                result = execute_cypher(graph, cypher)
                executed = True
            except CypherExecutionError:
                result = None
                executed = False
            if not needs_rescue(executed, result, validation):
                break

    return Text2CypherResult(
        question=input_NL,
        technique=technique.value,
        model=model if isinstance(model, str) else type(model).__name__,
        cypher=cypher,
        initial_cypher=initial_cypher,
        prompt=prompt_messages,
        executed=executed,
        rescued=rescue_attempts > 0,
        rescue_attempts=rescue_attempts,
        result=result,
        validation=validation,
        schema=schema_text,
        retrieved_examples=retrieved,
    )
