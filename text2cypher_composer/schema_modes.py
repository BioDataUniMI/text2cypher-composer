"""Schema representation modes: how the graph schema is derived/pruned for the prompt.

Ported from the miRNAKG schema-representation notebook (itself following
"The Impact of Schema Representation in the Text2Cypher Task",
https://doi.org/10.48550/arXiv.2505.05118), plus a new LLM-driven pruning mode.

`exact_match`/`ner_exact_match`/`similarity` need no NLP dependency from this
package itself — the caller supplies an already-loaded NLP pipeline (e.g. a
spaCy `Language`) via `nlp`; this module only calls it (`nlp(text)`,
`nlp.vocab[...]`, token `.similarity()`/`.is_alpha`/`.has_vector`), the same
duck-typed "bring your own model" pattern `resolve_model` uses for `model`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .schema import format_schema, get_schema, get_structured_schema
from .techniques import SchemaMode, SchemaModeLike


def _normalize_tokens(text: str) -> List[str]:
    """lowercase, drop punctuation/underscores/hyphens, split, light de-pluralize."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[_\-]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens]


def _mentioned(term: str, question_tokens: List[str], min_len: int = 2) -> bool:
    term_norm = " ".join(_normalize_tokens(term))
    if not term_norm:
        return False
    return any(tok in term_norm for tok in question_tokens if len(tok) >= min_len)


def _apply_selection(
    structured_schema: Dict[str, Any],
    node_labels: List[str],
    relationship_types: List[str],
) -> Dict[str, Any]:
    """Build a pruned structured schema from a selected set of labels/rel types.

    Falls back to the full, unpruned schema if no known node label was
    selected. Relationships are kept if their type was explicitly selected;
    otherwise (no relationship types selected), a relationship is kept when
    both its endpoints are among the selected node labels.
    """
    node_props = structured_schema.get("node_props", {}) or {}
    selected_nodes = [n for n in node_labels if n in node_props]
    if not selected_nodes:
        return structured_schema
    selected_node_set = set(selected_nodes)

    relationships = structured_schema.get("relationships", []) or []
    rel_props = structured_schema.get("rel_props", {}) or {}
    selected_rel_types = set(relationship_types) & set(rel_props.keys())

    if selected_rel_types:
        pruned_relationships = [r for r in relationships if r.get("type") in selected_rel_types]
    else:
        pruned_relationships = [
            r
            for r in relationships
            if r.get("start") in selected_node_set and r.get("end") in selected_node_set
        ]

    kept_rel_types = {r.get("type") for r in pruned_relationships}

    return {
        "node_props": {n: node_props[n] for n in selected_nodes},
        "relationships": pruned_relationships,
        "rel_props": {t: rel_props[t] for t in kept_rel_types if t in rel_props},
        "metadata": structured_schema.get("metadata", {}),
    }


def exact_match_prune(structured_schema: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Keep node labels mentioned (substring match) in `question`, plus their relationships."""
    q_tokens = _normalize_tokens(question)
    node_props = structured_schema.get("node_props", {}) or {}
    mentioned = [n for n in node_props if _mentioned(n, q_tokens)]
    return _apply_selection(structured_schema, mentioned, [])


def mask_entities(question: str, nlp: Any) -> str:
    """Replace each named entity span in `question` with its entity type label."""
    doc = nlp(question)
    masked = question
    for ent in reversed(list(doc.ents)):
        masked = masked[: ent.start_char] + ent.label_ + masked[ent.end_char :]
    return masked


def ner_exact_match_prune(structured_schema: Dict[str, Any], question: str, nlp: Any) -> Dict[str, Any]:
    """`exact_match_prune`, but named entities are masked first to reduce value/field-name confusion."""
    return exact_match_prune(structured_schema, mask_entities(question, nlp))


def similarity_prune(
    structured_schema: Dict[str, Any], question: str, nlp: Any, threshold: float = 0.5
) -> Dict[str, Any]:
    """Keep labels/types/properties whose word-vector similarity to the question exceeds `threshold`.

    Unlike `exact_match_prune`, there is no full-schema fallback: if nothing
    is similar enough, the pruned schema is empty.
    """
    qdoc = nlp(question.lower())
    q_tokens = [t for t in qdoc if t.is_alpha and t.has_vector]

    def _similar(term: str) -> bool:
        term = (term or "").lower().strip()
        if not term:
            return False
        lex = nlp.vocab[term]
        if not lex.has_vector:
            return False
        return any(qt.similarity(lex) >= threshold for qt in q_tokens)

    node_props = structured_schema.get("node_props", {}) or {}
    pruned_node_props = {}
    for node_type, props in node_props.items():
        filtered = [p for p in props if _similar(node_type) or _similar(p.get("property", ""))]
        if filtered:
            pruned_node_props[node_type] = filtered

    relationships = structured_schema.get("relationships", []) or []
    pruned_relationships = [
        r
        for r in relationships
        if _similar(r.get("type", "")) or _similar(r.get("start", "")) or _similar(r.get("end", ""))
    ]

    rel_props = structured_schema.get("rel_props", {}) or {}
    pruned_rel_props = {}
    for rel_type, props in rel_props.items():
        filtered = [p for p in props if _similar(rel_type) or _similar(p.get("property", ""))]
        if filtered:
            pruned_rel_props[rel_type] = filtered

    return {
        "node_props": pruned_node_props,
        "relationships": pruned_relationships,
        "rel_props": pruned_rel_props,
        "metadata": structured_schema.get("metadata", {}),
    }


class SchemaSelection(BaseModel):
    """Node labels and relationship types from a Neo4j schema relevant to a question."""

    node_labels: List[str] = Field(
        default_factory=list,
        description="Node labels — must exactly match labels in the given schema — relevant to answering the question.",
    )
    relationship_types: List[str] = Field(
        default_factory=list,
        description="Relationship types — must exactly match types in the given schema — relevant to answering the question.",
    )


_LLM_PRUNING_SYSTEM = (
    "You select the minimal subset of a Neo4j graph schema needed to write a "
    "Cypher query answering a natural-language question. Only choose node "
    "labels and relationship types that literally appear in the schema below "
    "— never invent new ones. If unsure whether a label/type is needed, "
    "prefer including it over omitting it."
)
_LLM_PRUNING_TEMPLATE = "Schema:\n{schema}\n\nQuestion: {question}"


def llm_prune(structured_schema: Dict[str, Any], question: str, llm: Any) -> Dict[str, Any]:
    """Ask `llm` (via structured/JSON-schema output) which labels/relationship types are relevant."""
    full_schema_text = format_schema(structured_schema, is_enhanced=True)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _LLM_PRUNING_SYSTEM), ("human", _LLM_PRUNING_TEMPLATE)]
    )
    chain = prompt | llm.with_structured_output(SchemaSelection, method="json_schema")
    selection = chain.invoke({"schema": full_schema_text, "question": question})
    return _apply_selection(structured_schema, selection.node_labels, selection.relationship_types)


def resolve_schema_text(
    graph: Any,
    mode: SchemaModeLike,
    question: str,
    llm: Optional[Any] = None,
    nlp: Optional[Any] = None,
    similarity_threshold: float = 0.5,
) -> str:
    """Compute the schema text to place in the prompt, per `mode` (see `SchemaMode`)."""
    mode = SchemaMode(mode)

    if mode == SchemaMode.SCHEMA:
        return get_schema(graph, is_enhanced=False)

    if mode == SchemaMode.ENHANCED:
        return get_schema(graph, is_enhanced=True)

    if mode == SchemaMode.EXACT_MATCH:
        structured = get_structured_schema(graph, is_enhanced=True)
        return format_schema(exact_match_prune(structured, question), is_enhanced=True)

    if mode == SchemaMode.NER_EXACT_MATCH:
        if nlp is None:
            raise ValueError(
                "schema_mode='ner_exact_match' requires an `nlp` argument: a loaded NLP "
                "pipeline with named-entity recognition (e.g. spaCy's en_ner_bionlp13cg_md)."
            )
        structured = get_structured_schema(graph, is_enhanced=True)
        return format_schema(ner_exact_match_prune(structured, question, nlp), is_enhanced=True)

    if mode == SchemaMode.SIMILARITY:
        if nlp is None:
            raise ValueError(
                "schema_mode='similarity' requires an `nlp` argument: a loaded NLP pipeline "
                "with word vectors (e.g. spaCy's en_core_web_md)."
            )
        structured = get_structured_schema(graph, is_enhanced=False)
        return format_schema(
            similarity_prune(structured, question, nlp, threshold=similarity_threshold), is_enhanced=False
        )

    if mode == SchemaMode.LLM_PRUNING:
        if llm is None:
            raise ValueError("schema_mode='llm_pruning' requires a structured-output-capable model.")
        structured = get_structured_schema(graph, is_enhanced=True)
        return format_schema(llm_prune(structured, question, llm), is_enhanced=True)

    raise ValueError(f"Unknown schema_mode: {mode}")
