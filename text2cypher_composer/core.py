from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .cypher_utils import CypherExecutionError, execute_cypher_with_warnings, normalize_generated_cypher
from .graph_db import DatabaseLike, resolve_database
from .llm import ModelLike, resolve_model, resolve_pruning_model
from .prompts import messages_for
from .rag import RAGDataset, resolve_adaptive_rag_levels
from .rescue import build_error_message, needs_rescue, rescue_messages
from .schema_modes import resolve_cascade_mode_levels, resolve_schema_text
from .tokens import count_message_tokens
from .techniques import (
    CascadeStrategy,
    CascadeStrategyLike,
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
from .verification import SemanticVerification, verify_semantics

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
    rescue attempt — the native Neo4j error, an "Empty result set." note,
    and CyVer's validation report, concatenated (see
    `rescue.build_error_message`) — empty if `rescued` is False.
    `rescue_prompts` holds, in the same order, the exact fully-instantiated
    messages sent to the model for each rescue attempt (`rescue_prompts[i]`
    produced `cypher` from `error_message` `rescue_error_messages[i]`) —
    also empty if `rescued` is False. `cascade_mode` and `rescue_prompt` are
    mutually exclusive (see `run()`), so these are always empty when
    `cascade_mode` was used.

    `prompt_tokens` is `prompt`'s token count (via `tiktoken`, `None` if it
    isn't installed) — handy for comparing prompt size across `technique`/
    `schema_mode` (e.g. how much schema filtering actually saves).
    `rescue_prompt_tokens` is the parallel per-attempt token count for
    `rescue_prompts` — always a list of `rescue_attempts` numbers when
    rescued (handy for tallying how many extra tokens `rescue_prompt` costs),
    empty otherwise.

    `cascade_mode_level` is which rung of the `cascade_mode` cascade
    (see `run()`) the returned `cypher`/`result`/etc. came from — `"narrow"`,
    `"nodes_only"`, or `"full"` — `None` if `cascade_mode` wasn't used.
    `cascade_mode_attempts` is how many rungs were tried (1 if the first
    one tried already succeeded, up to 3). `cascade_mode_prompts`/
    `cascade_mode_prompt_tokens` hold, one entry per rung tried (in the
    same order, from most to least pruned), that rung's initial
    fully-instantiated prompt and its token count — both empty if
    `cascade_mode` wasn't used.

    `adaptive_rag_level`/`adaptive_rag_attempts`/`adaptive_rag_prompts`/
    `adaptive_rag_prompt_tokens` are the RAG-side siblings of the
    `cascade_mode_*` fields above, populated instead when `adaptive_rag=True`
    was used (`cascade_mode` and `adaptive_rag` are mutually exclusive, so
    only one set is ever non-empty): `adaptive_rag_level` is which rung of
    the `adaptive_rag` cascade — `"minimal"`, `"moderate"`, or `"full"` —
    the returned `cypher`/`result`/etc. came from; `adaptive_rag_attempts`
    is how many rungs were tried (1..3); `adaptive_rag_prompts`/
    `adaptive_rag_prompt_tokens` hold, one entry per rung tried (from fewest
    to most retrieved examples), that rung's initial fully-instantiated
    prompt and its token count.

    `self_verification_passed`/`self_verification_reasoning` hold, for the
    final returned attempt only, whether `self_verification` (see `run()`)
    judged `cypher` to actually answer `question`, and why — both `None` if
    `self_verification` wasn't used, or if the final attempt was already
    mechanically broken (`rescue.needs_rescue`), since there's no point
    asking for a semantic verdict on a query that doesn't even execute.
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
    cascade_mode_level: Optional[str] = None
    cascade_mode_attempts: int = 0
    cascade_mode_prompts: List[List[Dict[str, str]]] = field(default_factory=list)
    cascade_mode_prompt_tokens: List[Optional[int]] = field(default_factory=list)
    adaptive_rag_level: Optional[str] = None
    adaptive_rag_attempts: int = 0
    adaptive_rag_prompts: List[List[Dict[str, str]]] = field(default_factory=list)
    adaptive_rag_prompt_tokens: List[Optional[int]] = field(default_factory=list)
    self_verification_passed: Optional[bool] = None
    self_verification_reasoning: Optional[str] = None


def _resolve_dataset(dataset: DatasetLike) -> RAGDataset:
    if isinstance(dataset, RAGDataset):
        return dataset
    if isinstance(dataset, str):
        return RAGDataset.from_root(dataset)
    raise TypeError(
        "`dataset` must be a RAGDataset instance or a path to a bio2C-style "
        "benchmark root (containing chroma_db/, CypherQueries/, Neo4jOutputs/)."
    )


@dataclass
class _AttemptResult:
    """One full generation attempt: initial prompt, optionally followed by `rescue_prompt` retries.

    Everything `run()` needs from a single (schema-level, in the
    `cascade_mode` case) attempt to populate `Text2CypherResult` — factored
    out so the cascade loop and the non-cascade path share the exact same
    generate/execute/validate/rescue logic instead of duplicating it.
    """

    cypher: str
    initial_cypher: str
    prompt_messages: List[Dict[str, str]]
    prompt_tokens: Optional[int]
    executed: bool
    result: Optional[List[Dict[str, Any]]]
    validation: CypherValidationReport
    execution_error: Optional[str]
    execution_warnings: List[str]
    rescue_attempts: int
    rescue_error_messages: List[str]
    rescue_prompts: List[List[Dict[str, str]]]
    rescue_prompt_tokens: List[Optional[int]]
    needs_retry: bool
    self_verification_passed: Optional[bool]
    self_verification_reasoning: Optional[str]


def _needs_retry(
    executed: bool,
    result: Optional[List[Dict[str, Any]]],
    validation: CypherValidationReport,
    question: str,
    cypher: str,
    self_verification: bool,
    verification_llm: Optional[Any],
    verification_criteria: Optional[str],
) -> "tuple[bool, Optional[SemanticVerification]]":
    """Whether an attempt still needs a retry — mechanical checks first, semantic ones only if those pass.

    `rescue.needs_rescue` alone decides this when `self_verification` is
    off. When it's on, a query that already needs_rescue mechanically
    short-circuits (no point spending a verification call on a query that
    doesn't even execute); otherwise `verify_semantics` is asked whether the
    query actually answers `question`, and its verdict becomes the retry
    decision instead.
    """
    if needs_rescue(executed, result, validation):
        return True, None
    if not self_verification:
        return False, None
    verification = verify_semantics(verification_llm, question, cypher, result, criteria=verification_criteria)
    return not verification.answers_question, verification


@dataclass
class _OneAttempt:
    """One single generation+execute+validate+retry-check shot from a given prompt/kwargs pair.

    The building block both `_generate_execute_and_rescue` (its initial attempt, and each
    `rescue_prompt` retry) and the `cascade_mode` `strategy="delta"` loop in `run()` are built
    from — factored out so both share the exact same generate/execute/validate/`_needs_retry`
    logic instead of duplicating it.
    """

    cypher: str
    prompt_messages: List[Dict[str, str]]
    prompt_tokens: Optional[int]
    executed: bool
    result: Optional[List[Dict[str, Any]]]
    validation: CypherValidationReport
    execution_error: Optional[str]
    execution_warnings: List[str]
    retry_needed: bool
    verification: Optional[SemanticVerification]


def _generate_once(
    prompt_template: ChatPromptTemplate,
    format_kwargs: Dict[str, Any],
    llm: Any,
    model_name: str,
    graph: Any,
    question: str,
    self_verification: bool,
    verification_llm: Optional[Any],
    verification_criteria: Optional[str],
) -> _OneAttempt:
    """Format `prompt_template` with `format_kwargs`, generate, execute, validate, and check retry."""
    formatted_messages = prompt_template.format_messages(**format_kwargs)
    prompt_messages = [{"role": m.type, "content": m.content} for m in formatted_messages]
    prompt_tokens = count_message_tokens(prompt_messages, model_name)

    chain = prompt_template | llm | StrOutputParser()
    cypher = normalize_generated_cypher(chain.invoke(format_kwargs))

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

    retry_needed, verification = _needs_retry(
        executed, result, validation, question, cypher, self_verification, verification_llm, verification_criteria
    )

    return _OneAttempt(
        cypher=cypher,
        prompt_messages=prompt_messages,
        prompt_tokens=prompt_tokens,
        executed=executed,
        result=result,
        validation=validation,
        execution_error=execution_error,
        execution_warnings=execution_warnings,
        retry_needed=retry_needed,
        verification=verification,
    )


def _generate_execute_and_rescue(
    llm: Any,
    prompt_template: ChatPromptTemplate,
    format_kwargs: Dict[str, Any],
    model_name: str,
    graph: Any,
    uses_schema: bool,
    uses_rag: bool,
    rescue_prompt: bool,
    max_retries: int,
    self_verification: bool = False,
    verification_llm: Optional[Any] = None,
    verification_criteria: Optional[str] = None,
) -> _AttemptResult:
    """Generate a Cypher query from `format_kwargs`, run it, and rescue it if asked to."""
    question = format_kwargs["question"]
    first = _generate_once(
        prompt_template, format_kwargs, llm, model_name, graph, question,
        self_verification, verification_llm, verification_criteria,
    )
    initial_cypher = first.cypher
    cypher = first.cypher
    prompt_messages = first.prompt_messages
    prompt_tokens = first.prompt_tokens
    executed, result, validation = first.executed, first.result, first.validation
    execution_error, execution_warnings = first.execution_error, first.execution_warnings
    retry_needed, verification = first.retry_needed, first.verification

    rescue_attempts = 0
    rescue_error_messages: List[str] = []
    rescue_prompts: List[List[Dict[str, str]]] = []
    rescue_prompt_tokens: List[Optional[int]] = []
    if rescue_prompt and retry_needed:
        rescue_prompt_template = ChatPromptTemplate.from_messages(rescue_messages(uses_schema, uses_rag))
        for _ in range(max_retries):
            rescue_attempts += 1
            semantic_feedback = (
                verification.reasoning if verification and not verification.answers_question else None
            )
            error_message = build_error_message(
                executed, result, validation, execution_error, semantic_feedback=semantic_feedback
            )
            rescue_error_messages.append(error_message)
            rescue_kwargs = {
                **format_kwargs,
                "query": cypher,
                "error_message": error_message,
            }
            attempt = _generate_once(
                rescue_prompt_template, rescue_kwargs, llm, model_name, graph, question,
                self_verification, verification_llm, verification_criteria,
            )
            rescue_prompts.append(attempt.prompt_messages)
            rescue_prompt_tokens.append(attempt.prompt_tokens)
            cypher = attempt.cypher
            executed, result, validation = attempt.executed, attempt.result, attempt.validation
            execution_error, execution_warnings = attempt.execution_error, attempt.execution_warnings
            retry_needed, verification = attempt.retry_needed, attempt.verification
            if not retry_needed:
                break

    return _AttemptResult(
        cypher=cypher,
        initial_cypher=initial_cypher,
        prompt_messages=prompt_messages,
        prompt_tokens=prompt_tokens,
        executed=executed,
        result=result,
        validation=validation,
        execution_error=execution_error,
        execution_warnings=execution_warnings,
        rescue_attempts=rescue_attempts,
        rescue_error_messages=rescue_error_messages,
        rescue_prompts=rescue_prompts,
        rescue_prompt_tokens=rescue_prompt_tokens,
        needs_retry=retry_needed,
        self_verification_passed=verification.answers_question if verification else None,
        self_verification_reasoning=verification.reasoning if verification else None,
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
    cascade_mode: bool = False,
    skip_narrow_schema_filter: bool = False,
    cascade_strategy: CascadeStrategyLike = CascadeStrategy.STANDARD,
    adaptive_rag: bool = False,
    cache_schema: bool = True,
    self_verification: bool = False,
    verification_model: Optional[ModelLike] = None,
    verification_criteria: Optional[str] = None,
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
            (if it didn't execute), an "Empty result set." note if it
            executed but returned nothing, and CyVer's validation report
            (both its warning-level notifications and hard errors) — see
            `rescue.build_error_message`, `result.rescue_error_messages`, and
            `result.rescue_prompts` (the fully-instantiated messages sent for
            each rescue attempt).
        max_retries: how many rescue attempts to make (stopping early once
            one succeeds) before giving up. Only relevant when
            `rescue_prompt` is True.
        cascade_mode: if True, an attempt that fails to execute or comes
            back empty is retried from scratch (a fresh prompt, not
            `rescue_prompt`'s error-aware fix-up) with progressively less
            aggressive schema pruning: first the most aggressively pruned
            schema ("narrow" — node labels, relationship types, and
            properties all narrowed to the question), then a less aggressive
            fallback ("nodes_only" — only node labels matched; relationships
            kept via shared endpoints, every property of a selected label/
            type kept), then finally the unpruned schema ("full"). Stops
            early once one rung succeeds. Mutually exclusive with
            `rescue_prompt`/`max_retries` — `run()` raises `ValueError` if
            `cascade_mode=True` is combined with `rescue_prompt=True` or a
            non-default `max_retries`; pick one retry strategy or the other,
            not both. Only meaningful — and only allowed — for a pruning
            `schema_mode` ("exact_match", "ner_exact_match", "similarity",
            "llm_pruning", or "ie_extraction"); "schema"/"enhanced" have
            nothing to prune from. See `result.cascade_mode_level`,
            `result.cascade_mode_attempts`, `result.cascade_mode_prompts`,
            and `result.cascade_mode_prompt_tokens`.
        skip_narrow_schema_filter: if True, skip the "narrow" rung and start
            the `cascade_mode` cascade directly at "nodes_only". Requires
            `cascade_mode=True`.
        cascade_strategy: how `cascade_mode`'s rungs are prompted — `"standard"`
            (default) sends each rung as a fresh, self-contained prompt
            carrying that rung's *entire* schema, same as if `cascade_mode`
            were used alone. `"delta"` — the "Incremental delta cascade" —
            keeps this fresh-prompt behavior only for the first rung
            ("narrow"); every rung after that instead reuses
            `rescue_prompt`'s error-aware fix-up mechanics (the previous
            rung's generated query, plus why it needed to move on) but shows
            only the schema elements newly introduced at this rung (see
            `schema_modes.schema_delta`), not the ones already shown at a
            previous rung — cutting redundant schema tokens repeated across
            rungs, at the cost of each rung after the first depending on the
            previous rung's output rather than being independent. Requires
            `cascade_mode=True` — `run()` raises `ValueError` if
            `cascade_strategy="delta"` is passed with `cascade_mode=False`.
        adaptive_rag: if True, an attempt that fails to execute or comes
            back empty is retried from scratch (a fresh prompt, not
            `rescue_prompt`'s error-aware fix-up) with progressively more
            retrieved RAG examples — the RAG-side sibling of `cascade_mode`,
            but expanding retrieved context instead of un-pruning the
            schema: first a single example ("minimal"), then the dataset's
            configured `n_results` ("moderate"), then finally every example
            in the collection ("full"). Stops early once one rung succeeds.
            Mutually exclusive with `cascade_mode`, `rescue_prompt`, and a
            non-default `max_retries` — `run()` raises `ValueError` if
            `adaptive_rag=True` is combined with any of those; pick one
            retry strategy, not several. Only meaningful — and only
            allowed — for a RAG-using `technique` ("RAG", "RAG+O",
            "Schema+RAG", "Schema+RAG+O"). See `result.adaptive_rag_level`,
            `result.adaptive_rag_attempts`, `result.adaptive_rag_prompts`,
            and `result.adaptive_rag_prompt_tokens`.
        cache_schema: if True (the default), the graph schema extracted from
            Neo4j for `technique`s that use the schema is cached per
            `(database, schema_mode's is_enhanced, sample)` and reused across
            `run()` calls against the same graph — see
            `schema.get_structured_schema`. Extracting a schema is a fixed
            cost that doesn't change across the many questions of a
            benchmark run against the same database, so re-extracting it on
            every call (multiplied by every `cascade_mode` rung) is pure
            overhead; the cache only covers this extraction step, not the
            per-question filtering/pruning, and it never reduces LLM call
            cost. Pass `False` to always re-extract (e.g. if the schema
            legitimately changes mid-experiment), or see
            `clear_schema_cache` to invalidate an already-cached entry
            instead. Ignored (harmlessly) for a `technique` that doesn't use
            the schema.
        self_verification: if True, an attempt that mechanically looks fine
            (executed, non-empty, syntactically valid — i.e. `rescue.
            needs_rescue` says no rescue is needed) is additionally reviewed
            by a model: given the question, the generated `cypher`, and the
            rows it returned, is this actually the right query? A query can
            run cleanly and still not answer what was asked (wrong
            direction on a relationship, an aggregate over the wrong
            property, too broad/narrow a filter, ...) — this is a different,
            orthogonal signal from CyVer's mechanical checks, not a
            replacement for them: a mechanically-broken attempt is retried
            without spending a verification call on it. A failed verdict is
            folded into the same retry decision `rescue_prompt`/
            `cascade_mode` already make — under `rescue_prompt`, the
            verdict's reasoning is also fed into the fix-up prompt's
            `error_message` (see `rescue.build_error_message`'s
            `semantic_feedback`); under `cascade_mode`, a failed verdict at
            one rung falls through to the next exactly like a mechanical
            failure would, with no error context (consistent with
            `cascade_mode`'s "fresh prompt, not a fix-up" design). Requires
            `rescue_prompt=True` or `cascade_mode=True` — `run()` raises
            `ValueError` otherwise, since there would be no retry to inform.
            Costs one extra LLM call per mechanically-valid attempt/rung.
            See `result.self_verification_passed`/
            `result.self_verification_reasoning`.
        verification_model: which model judges `self_verification` — an
            OpenAI/Anthropic/Google/DeepSeek model id or a LangChain-
            compatible chat model / Runnable, same as `model`. Defaults to
            reusing `model` itself if omitted. Ignored (and rejected with
            `ValueError` if passed) when `self_verification` is False.
        verification_criteria: free-text extra evaluation guidance appended
            to the verification prompt (e.g. "the answer must include
            units"), on top of "does this query answer the question".
            Ignored (and rejected with `ValueError` if passed) when
            `self_verification` is False.
        dry_run: if True, build and return the fully-instantiated `prompt`
            (schema resolved, RAG examples retrieved, exactly as it would be
            for a real call) but stop there — no generation call, no Cypher
            execution, no CyVer validation, no rescue. `cypher`,
            `initial_cypher`, `result`, and `validation` are all `None`, and
            `executed` is `False`. Incompatible with `rescue_prompt=True`,
            `cascade_mode=True`, or `adaptive_rag=True` (there's nothing
            generated to rescue/fall back from). `prompt_tokens` is still
            computed, so `dry_run` is enough to compare prompt token counts
            across `technique`/`schema_mode` without spending a generation
            call.

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
        parallel to `rescue_prompts`. If `cascade_mode` was used,
        `cascade_mode_level` says which rung ("narrow"/"nodes_only"/
        "full") the rest of the result reflects, `cascade_mode_attempts`
        how many rungs were tried, and `cascade_mode_prompts`/
        `cascade_mode_prompt_tokens` list, one per rung tried, that
        rung's initial prompt/token count — all default to `None`/`0`/`[]`
        if `cascade_mode` wasn't used. Under `cascade_strategy="delta"`,
        `schema` holds only the *delta* text actually shown at the winning
        rung (not the cumulative schema up to it) — the rest of what that
        rung's model call saw (the previous rung's query, why it needed to
        move on) is visible in `prompt` instead. If `adaptive_rag` was used instead,
        `adaptive_rag_level`/`adaptive_rag_attempts`/`adaptive_rag_prompts`/
        `adaptive_rag_prompt_tokens` are the same shape, one rung per
        retrieved-example count tried ("minimal"/"moderate"/"full") instead
        of one per schema pruning level. If `self_verification` was used,
        `self_verification_passed`/`self_verification_reasoning` report the
        final attempt's semantic verdict — both `None` if it wasn't used, or
        if the final attempt was already mechanically broken. If `dry_run`
        is True, only `prompt` (`prompt_tokens`, and `schema`/
        `retrieved_examples`, if applicable) are populated.
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
    if cascade_mode and not uses_schema:
        raise ValueError(f"`cascade_mode` requires technique='{technique.value}' to use the schema.")
    if skip_narrow_schema_filter and not cascade_mode:
        raise ValueError("`skip_narrow_schema_filter` requires `cascade_mode=True`.")
    if CascadeStrategy(cascade_strategy) != CascadeStrategy.STANDARD and not cascade_mode:
        raise ValueError("`cascade_strategy` other than 'standard' requires `cascade_mode=True`.")
    if dry_run and cascade_mode:
        raise ValueError(
            "`dry_run` and `cascade_mode` are incompatible: dry_run generates nothing to fall back from."
        )
    if cascade_mode and rescue_prompt:
        raise ValueError(
            "`cascade_mode` and `rescue_prompt` are mutually exclusive: pick one retry strategy, not both."
        )
    if cascade_mode and max_retries != 1:
        raise ValueError(
            "`cascade_mode` and `max_retries` are mutually exclusive (max_retries only applies to "
            "rescue_prompt): pick one retry strategy, not both."
        )
    if cascade_mode and SchemaMode(schema_mode if schema_mode is not None else SchemaMode.SCHEMA) in (
        SchemaMode.SCHEMA,
        SchemaMode.ENHANCED,
    ):
        raise ValueError(
            "`cascade_mode` requires a pruning `schema_mode` (exact_match, ner_exact_match, "
            "similarity, llm_pruning, or ie_extraction) — 'schema'/'enhanced' have nothing to "
            "prune from."
        )
    if adaptive_rag and not uses_rag:
        raise ValueError(f"`adaptive_rag` requires technique='{technique.value}' to use RAG.")
    if dry_run and adaptive_rag:
        raise ValueError(
            "`dry_run` and `adaptive_rag` are incompatible: dry_run generates nothing to fall back from."
        )
    if adaptive_rag and rescue_prompt:
        raise ValueError(
            "`adaptive_rag` and `rescue_prompt` are mutually exclusive: pick one retry strategy, not both."
        )
    if adaptive_rag and max_retries != 1:
        raise ValueError(
            "`adaptive_rag` and `max_retries` are mutually exclusive (max_retries only applies to "
            "rescue_prompt): pick one retry strategy, not both."
        )
    if adaptive_rag and cascade_mode:
        raise ValueError(
            "`adaptive_rag` and `cascade_mode` are mutually exclusive: pick one retry strategy, not both."
        )
    if self_verification and not (rescue_prompt or cascade_mode):
        raise ValueError(
            "`self_verification` requires `rescue_prompt=True` or `cascade_mode=True` — there "
            "would otherwise be no retry for its verdict to inform."
        )
    if not self_verification and (verification_model is not None or verification_criteria is not None):
        raise ValueError(
            "`verification_model`/`verification_criteria` were provided but `self_verification` "
            "is False."
        )
    cascade_strategy = CascadeStrategy(cascade_strategy)

    graph = resolve_database(database)
    llm = resolve_model(model)
    model_name = model if isinstance(model, str) else type(model).__name__
    verification_llm = None
    if self_verification:
        verification_llm = resolve_pruning_model(verification_model if verification_model is not None else model)

    schema_text = None
    schema_levels = None
    if uses_schema:
        resolved_mode = SchemaMode(schema_mode) if schema_mode is not None else SchemaMode.SCHEMA
        pruning_llm = resolve_pruning_model(model) if resolved_mode == SchemaMode.LLM_PRUNING else None
        if cascade_mode:
            schema_levels = resolve_cascade_mode_levels(
                graph,
                resolved_mode,
                input_NL,
                llm=pruning_llm,
                nlp=nlp,
                similarity_threshold=similarity_threshold,
                ie_engine=ie_engine,
                skip_narrow=skip_narrow_schema_filter,
                cache_schema=cache_schema,
                strategy=cascade_strategy,
            )
            schema_text = schema_levels[0][1]
        else:
            schema_text = resolve_schema_text(
                graph,
                resolved_mode,
                input_NL,
                llm=pruning_llm,
                nlp=nlp,
                similarity_threshold=similarity_threshold,
                schema_components=schema_components,
                ie_engine=ie_engine,
                cache_schema=cache_schema,
            )

    retrieved = None
    rag_levels = None
    format_kwargs: Dict[str, Any] = {"question": input_NL}
    if uses_schema and not cascade_mode:
        format_kwargs["enhanced_schema"] = schema_text
    if uses_rag:
        rag_dataset = _resolve_dataset(dataset)
        with_output = technique in OUTPUT_AUGMENTED_TECHNIQUES
        if adaptive_rag:
            rag_levels = resolve_adaptive_rag_levels(rag_dataset, input_NL, with_output)
            retrieved = rag_levels[0][1]
        else:
            retrieved = rag_dataset.retrieve_examples(input_NL, with_output=with_output)
            format_kwargs["examples"] = retrieved["examples_text"]

    prompt_template = ChatPromptTemplate.from_messages(messages_for(technique))

    if dry_run:
        formatted_messages = prompt_template.format_messages(**format_kwargs)
        prompt_messages = [{"role": m.type, "content": m.content} for m in formatted_messages]
        prompt_tokens = count_message_tokens(prompt_messages, model_name)
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

    cascade_mode_level: Optional[str] = None
    cascade_mode_prompts: List[List[Dict[str, str]]] = []
    cascade_mode_prompt_tokens: List[Optional[int]] = []
    adaptive_rag_level: Optional[str] = None
    adaptive_rag_prompts: List[List[Dict[str, str]]] = []
    adaptive_rag_prompt_tokens: List[Optional[int]] = []

    if cascade_mode and cascade_strategy == CascadeStrategy.DELTA:
        # cascade_mode and rescue_prompt are mutually exclusive (validated above), but the
        # "delta" strategy reuses rescue_prompt's fix-up mechanics for every rung after the
        # first — only that first rung is a fresh, self-contained attempt.
        rescue_prompt_template = ChatPromptTemplate.from_messages(rescue_messages(uses_schema, uses_rag))
        previous: Optional[_OneAttempt] = None
        for i, (level, level_schema_text) in enumerate(schema_levels):
            if previous is None:
                level_kwargs = {**format_kwargs, "enhanced_schema": level_schema_text}
                one = _generate_once(
                    prompt_template, level_kwargs, llm, model_name, graph, input_NL,
                    self_verification, verification_llm, verification_criteria,
                )
            else:
                semantic_feedback = (
                    previous.verification.reasoning
                    if previous.verification and not previous.verification.answers_question
                    else None
                )
                error_message = build_error_message(
                    previous.executed, previous.result, previous.validation, previous.execution_error,
                    semantic_feedback=semantic_feedback,
                )
                rescue_kwargs = {
                    **format_kwargs,
                    "enhanced_schema": level_schema_text,
                    "query": previous.cypher,
                    "error_message": error_message,
                }
                one = _generate_once(
                    rescue_prompt_template, rescue_kwargs, llm, model_name, graph, input_NL,
                    self_verification, verification_llm, verification_criteria,
                )
            cascade_mode_prompts.append(one.prompt_messages)
            cascade_mode_prompt_tokens.append(one.prompt_tokens)
            cascade_mode_level = level.value
            schema_text = level_schema_text
            previous = one
            is_last = i == len(schema_levels) - 1
            if is_last or not one.retry_needed:
                break
        attempt = _AttemptResult(
            cypher=previous.cypher,
            initial_cypher=previous.cypher,
            prompt_messages=previous.prompt_messages,
            prompt_tokens=previous.prompt_tokens,
            executed=previous.executed,
            result=previous.result,
            validation=previous.validation,
            execution_error=previous.execution_error,
            execution_warnings=previous.execution_warnings,
            rescue_attempts=0,
            rescue_error_messages=[],
            rescue_prompts=[],
            rescue_prompt_tokens=[],
            needs_retry=previous.retry_needed,
            self_verification_passed=previous.verification.answers_question if previous.verification else None,
            self_verification_reasoning=previous.verification.reasoning if previous.verification else None,
        )
    elif cascade_mode:
        # cascade_mode and rescue_prompt are mutually exclusive (validated above), so every
        # rung here is a single clean attempt, never followed by an error-aware fix-up retry.
        for i, (level, level_schema_text) in enumerate(schema_levels):
            level_kwargs = {**format_kwargs, "enhanced_schema": level_schema_text}
            attempt = _generate_execute_and_rescue(
                llm, prompt_template, level_kwargs, model_name, graph, uses_schema, uses_rag,
                rescue_prompt=False, max_retries=1,
                self_verification=self_verification, verification_llm=verification_llm,
                verification_criteria=verification_criteria,
            )
            cascade_mode_prompts.append(attempt.prompt_messages)
            cascade_mode_prompt_tokens.append(attempt.prompt_tokens)
            cascade_mode_level = level.value
            schema_text = level_schema_text
            is_last = i == len(schema_levels) - 1
            if is_last or not attempt.needs_retry:
                break
    elif adaptive_rag:
        # adaptive_rag and rescue_prompt are mutually exclusive (validated above), so every
        # rung here is a single clean attempt, never followed by an error-aware fix-up retry.
        for i, (level, retrieved_level) in enumerate(rag_levels):
            level_kwargs = {**format_kwargs, "examples": retrieved_level["examples_text"]}
            attempt = _generate_execute_and_rescue(
                llm, prompt_template, level_kwargs, model_name, graph, uses_schema, uses_rag,
                rescue_prompt=False, max_retries=1,
            )
            adaptive_rag_prompts.append(attempt.prompt_messages)
            adaptive_rag_prompt_tokens.append(attempt.prompt_tokens)
            adaptive_rag_level = level.value
            retrieved = retrieved_level
            is_last = i == len(rag_levels) - 1
            if is_last or not needs_rescue(attempt.executed, attempt.result, attempt.validation):
                break
    else:
        attempt = _generate_execute_and_rescue(
            llm, prompt_template, format_kwargs, model_name, graph, uses_schema, uses_rag, rescue_prompt, max_retries,
            self_verification=self_verification, verification_llm=verification_llm,
            verification_criteria=verification_criteria,
        )

    return Text2CypherResult(
        question=input_NL,
        technique=technique.value,
        model=model_name,
        cypher=attempt.cypher,
        initial_cypher=attempt.initial_cypher,
        prompt=attempt.prompt_messages,
        executed=attempt.executed,
        rescued=attempt.rescue_attempts > 0,
        rescue_attempts=attempt.rescue_attempts,
        rescue_error_messages=attempt.rescue_error_messages,
        rescue_prompts=attempt.rescue_prompts,
        prompt_tokens=attempt.prompt_tokens,
        rescue_prompt_tokens=attempt.rescue_prompt_tokens,
        execution_error=attempt.execution_error,
        execution_warnings=attempt.execution_warnings,
        result=attempt.result,
        validation=attempt.validation,
        schema=schema_text,
        retrieved_examples=retrieved,
        cascade_mode_level=cascade_mode_level,
        cascade_mode_attempts=len(cascade_mode_prompts),
        cascade_mode_prompts=cascade_mode_prompts,
        cascade_mode_prompt_tokens=cascade_mode_prompt_tokens,
        adaptive_rag_level=adaptive_rag_level,
        adaptive_rag_attempts=len(adaptive_rag_prompts),
        adaptive_rag_prompts=adaptive_rag_prompts,
        adaptive_rag_prompt_tokens=adaptive_rag_prompt_tokens,
        self_verification_passed=attempt.self_verification_passed,
        self_verification_reasoning=attempt.self_verification_reasoning,
    )
