from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .cypher_utils import CypherExecutionError, execute_cypher_with_warnings, normalize_generated_cypher
from .graph_db import DatabaseLike, resolve_database
from .llm import ModelLike, resolve_model, resolve_pruning_model
from .prompts import messages_for
from .rag import RAGDataset
from .rescue import build_error_message, needs_rescue, rescue_messages
from .schema_modes import resolve_schema_text
from .tokens import count_message_tokens
from .techniques import (
    DEFAULT_SCHEMA_COMPONENTS,
    RAG_TECHNIQUES,
    SCHEMA_TECHNIQUES,
    OUTPUT_AUGMENTED_TECHNIQUES,
    SchemaComponentLike,
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
    fix-up attempt but `prompt` does not (see `rescue_prompts` for those).
    `validation` (a CyVer report) is always populated, for every query,
    whether it executed successfully or not, and always reflects the final
    (possibly rescued) attempt. `result` holds the query rows if it
    executed, and is None otherwise.

    `execution_error`/`execution_warnings` reflect the *final* (possibly
    rescued) attempt's actual run against Neo4j: `execution_error` is the
    native Neo4j error (code + message) if it failed to execute, `None`
    otherwise; `execution_warnings` are any Neo4j notifications observed
    during execution (deprecated syntax, unknown labels/relationship-types/
    properties, cartesian products, ...) — populated whether the query
    succeeded or not, and independently of `rescue_prompt`.

    `rescue_error_messages` holds, in order, the `error_message` fed to each
    rescue attempt — the native Neo4j error, any execution warnings, and
    CyVer's validation report, concatenated (see `rescue.build_error_message`)
    — empty if `rescued` is False. `rescue_prompts` holds, in the same
    order, the exact fully-instantiated messages sent to the model for each
    rescue attempt (`rescue_prompts[i]` produced `cypher` from
    `error_message` `rescue_error_messages[i]`) — also empty if `rescued` is
    False.

    `prompt_tokens` is `prompt`'s token count (via `tiktoken`, `None` if it
    isn't installed) — handy for comparing prompt size across `technique`/
    `schema_mode` (e.g. how much schema filtering actually saves).
    `rescue_prompt_tokens` is the parallel per-attempt token count for
    `rescue_prompts` — always a list of `rescue_attempts` numbers when
    rescued (handy for tallying how many extra tokens `rescue_prompt` costs),
    empty otherwise.
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
    rescue_error_messages: List[str] = field(default_factory=list)
    rescue_prompts: List[List[Dict[str, str]]] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    rescue_prompt_tokens: List[Optional[int]] = field(default_factory=list)
    execution_error: Optional[str] = None
    execution_warnings: List[str] = field(default_factory=list)
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
    schema_components: Iterable[SchemaComponentLike] = DEFAULT_SCHEMA_COMPONENTS,
    nlp: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    ie_engine: Optional[Any] = None,
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
            "exact_match", "ner_exact_match", "similarity", "llm_pruning",
            or "ie_extraction" (see `SchemaMode`). Only meaningful — and
            only allowed — when `technique` uses the schema.
        schema_components: which schema element kinds `schema_mode=
            "exact_match"`/`"ner_exact_match"`/`"ie_extraction"` match
            against the question — any of "entity_types",
            "relationship_types", "node_properties",
            "relationship_properties" (see `SchemaComponent`). Defaults to
            entity types only. Ignored by every other `schema_mode`.
        nlp: a loaded NLP pipeline (e.g. a spaCy `Language`), required by
            `schema_mode="ner_exact_match"` (needs named-entity recognition,
            e.g. spaCy's `en_ner_bionlp13cg_md`) and `"similarity"` (needs
            word vectors, e.g. spaCy's `en_core_web_md`). Not bundled by
            this library — bring your own, already loaded.
        similarity_threshold: word-vector similarity cutoff used by
            `schema_mode="similarity"`. Ignored otherwise.
        ie_engine: a callable `ie_engine(schema_yaml, question) -> dict`
            performing schema-grounded information extraction, required by
            `schema_mode="ie_extraction"` (see `ie_prune`) — pass
            `schemalink_ie_engine()` for a ready-made one backed by the real
            `schemalink-engine` package (`pip install schemalink-engine`, or
            `pip install "text2cypher-composer[schemalink]"`), or bring your
            own. Ignored otherwise.
        rescue_prompt: if True, a query that fails to execute or comes back
            empty is retried with a second "fix this query" prompt — reusing
            the same schema/examples context as `technique`, plus the bad
            query and an error message concatenating the native Neo4j error
            (if it didn't execute), any Neo4j notifications observed during
            execution, an "Empty result set." note if it executed but
            returned nothing, and CyVer's validation report (both its
            warning-level notifications and hard errors) — see
            `rescue.build_error_message`, `result.rescue_error_messages`, and
            `result.rescue_prompts` (the fully-instantiated messages sent for
            each rescue attempt).
        max_retries: how many rescue attempts to make (stopping early once
            one succeeds) before giving up. Only relevant when
            `rescue_prompt` is True.
        dry_run: if True, build and return the fully-instantiated `prompt`
            (schema resolved, RAG examples retrieved, exactly as it would be
            for a real call) but stop there — no generation call, no Cypher
            execution, no CyVer validation, no rescue. `cypher`,
            `initial_cypher`, `result`, and `validation` are all `None`, and
            `executed` is `False`. Incompatible with `rescue_prompt=True`
            (there's nothing generated to rescue). `prompt_tokens` is still
            computed, so `dry_run` is enough to compare prompt token counts
            across `technique`/`schema_mode` without spending a generation call.

    Returns:
        A Text2CypherResult. `prompt` holds the fully-instantiated messages
        sent to the model for the initial attempt; `initial_cypher` is what
        it generated, and `cypher` is the final one — identical to
        `initial_cypher` unless `rescued` is True. `validation` always holds
        a CyVer report (syntax validity, schema-alignment score,
        property-access score, plus per-issue metadata) for the final
        Cypher, regardless of whether it executed. `result` holds the query
        rows if it did. `execution_error`/`execution_warnings` are the native
        Neo4j error/notifications from the final attempt's actual execution —
        populated regardless of `rescue_prompt`. `rescue_error_messages` and
        `rescue_prompts` list, in order, the `error_message`/fully-
        instantiated messages for each rescue attempt — both empty if
        `rescued` is False. `prompt_tokens`/`rescue_prompt_tokens` are their
        `tiktoken` token counts (`None` if `tiktoken` isn't installed) —
        `rescue_prompt_tokens` is a list of `rescue_attempts` numbers,
        parallel to `rescue_prompts`. If `dry_run` is True, only `prompt`
        (`prompt_tokens`, and `schema`/`retrieved_examples`, if applicable)
        are populated.
    """
    technique = Technique(technique)
    uses_rag = technique in RAG_TECHNIQUES
    uses_schema = technique in SCHEMA_TECHNIQUES

    if uses_rag and dataset is None:
        raise ValueError(f"technique='{technique.value}' requires a `dataset` (vector store) for RAG retrieval.")
    if not uses_rag and dataset is not None:
        raise ValueError(f"`dataset` was provided but technique='{technique.value}' does not use RAG.")
    if not uses_schema and (schema_mode is not None or nlp is not None or ie_engine is not None):
        raise ValueError(
            f"`schema_mode`/`nlp`/`ie_engine` were provided but technique='{technique.value}' "
            "does not use the schema."
        )
    if max_retries < 1:
        raise ValueError("`max_retries` must be >= 1.")
    if dry_run and rescue_prompt:
        raise ValueError("`dry_run` and `rescue_prompt` are incompatible: dry_run generates nothing to rescue.")

    graph = resolve_database(database)
    llm = resolve_model(model)
    model_name = model if isinstance(model, str) else type(model).__name__

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
            schema_components=schema_components,
            ie_engine=ie_engine,
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
    prompt_tokens = count_message_tokens(prompt_messages, model_name)

    if dry_run:
        return Text2CypherResult(
            question=input_NL,
            technique=technique.value,
            model=model_name,
            cypher=None,
            initial_cypher=None,
            prompt=prompt_messages,
            executed=False,
            dry_run=True,
            prompt_tokens=prompt_tokens,
            schema=schema_text,
            retrieved_examples=retrieved,
        )

    chain = prompt_template | llm | StrOutputParser()
    cypher = normalize_generated_cypher(chain.invoke(format_kwargs))
    initial_cypher = cypher

    validation = validate_cypher(graph, cypher)
    execution_error: Optional[str] = None
    execution_warnings: List[str] = []
    try:
        result, execution_warnings = execute_cypher_with_warnings(graph, cypher)
        executed = True
    except CypherExecutionError as e:
        result = None
        executed = False
        execution_error = f"{e.code}: {e.message}"
        execution_warnings = e.warnings

    rescue_attempts = 0
    rescue_error_messages: List[str] = []
    rescue_prompts: List[List[Dict[str, str]]] = []
    rescue_prompt_tokens: List[Optional[int]] = []
    if rescue_prompt and needs_rescue(executed, result, validation):
        rescue_prompt_template = ChatPromptTemplate.from_messages(rescue_messages(uses_schema, uses_rag))
        rescue_generation = llm | StrOutputParser()
        for _ in range(max_retries):
            rescue_attempts += 1
            error_message = build_error_message(executed, result, validation, execution_error, execution_warnings)
            rescue_error_messages.append(error_message)
            rescue_kwargs = {
                **format_kwargs,
                "query": cypher,
                "error_message": error_message,
            }
            formatted_rescue_messages = rescue_prompt_template.format_messages(**rescue_kwargs)
            rescue_prompt_dicts = [{"role": m.type, "content": m.content} for m in formatted_rescue_messages]
            rescue_prompts.append(rescue_prompt_dicts)
            rescue_prompt_tokens.append(count_message_tokens(rescue_prompt_dicts, model_name))
            cypher = normalize_generated_cypher(rescue_generation.invoke(formatted_rescue_messages))
            validation = validate_cypher(graph, cypher)
            try:
                result, execution_warnings = execute_cypher_with_warnings(graph, cypher)
                executed = True
                execution_error = None
            except CypherExecutionError as e:
                result = None
                executed = False
                execution_error = f"{e.code}: {e.message}"
                execution_warnings = e.warnings
            if not needs_rescue(executed, result, validation):
                break

    return Text2CypherResult(
        question=input_NL,
        technique=technique.value,
        model=model_name,
        cypher=cypher,
        initial_cypher=initial_cypher,
        prompt=prompt_messages,
        executed=executed,
        rescued=rescue_attempts > 0,
        rescue_attempts=rescue_attempts,
        rescue_error_messages=rescue_error_messages,
        rescue_prompts=rescue_prompts,
        prompt_tokens=prompt_tokens,
        rescue_prompt_tokens=rescue_prompt_tokens,
        execution_error=execution_error,
        execution_warnings=execution_warnings,
        result=result,
        validation=validation,
        schema=schema_text,
        retrieved_examples=retrieved,
    )
