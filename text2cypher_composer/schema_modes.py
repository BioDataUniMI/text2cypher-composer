"""Schema representation modes: how the graph schema is derived/pruned for the prompt.

Ported from the miRNAKG schema-representation notebook (itself following
"The Impact of Schema Representation in the Text2Cypher Task",
https://doi.org/10.48550/arXiv.2505.05118), plus a new LLM-driven pruning mode
and a schema-grounded information-extraction mode (`ie_extraction`).

`exact_match`/`ner_exact_match`/`similarity` need no NLP dependency from this
package itself — the caller supplies an already-loaded NLP pipeline (e.g. a
spaCy `Language`) via `nlp`; this module only calls it (`nlp(text)`,
`nlp.vocab[...]`, token `.similarity()`/`.is_alpha`/`.has_vector`), the same
duck-typed "bring your own model" pattern `resolve_model` uses for `model`.
`ie_extraction` follows the same pattern via `ie_engine` — see `ie_prune` —
though `schemalink_adapter.schemalink_ie_engine()` ships a ready-made one
backed by the real `schemalink-engine` PyPI package.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .schema import format_schema, get_schema, get_structured_schema
from .techniques import (
    ALL_SCHEMA_COMPONENTS,
    CascadeModeLevel,
    DEFAULT_SCHEMA_COMPONENTS,
    SchemaComponent,
    SchemaComponentLike,
    SchemaMode,
    SchemaModeLike,
)


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


def _normalize_components(components: Iterable[SchemaComponentLike]) -> "set[SchemaComponent]":
    return {SchemaComponent(c) for c in components}


def _filter_props(
    all_props: Dict[str, List[Dict[str, Any]]],
    keep: Optional[Dict[str, List[str]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Narrow each entry's property list to `keep[key]`, if given.

    `keep=None` leaves every property in place. When `keep` is given but a
    key's kept-property list is empty (nothing matched), that key's
    properties are left unfiltered too — an entity/relationship type with no
    properties left isn't useful, so this favors over- over under-inclusion,
    mirroring `_apply_selection`'s fallback to the full schema when no label
    is selected at all.
    """
    if keep is None:
        return all_props
    result = {}
    for key, props in all_props.items():
        kept_names = set(keep.get(key, []))
        filtered = [p for p in props if p.get("property") in kept_names]
        result[key] = filtered or props
    return result


def _apply_selection(
    structured_schema: Dict[str, Any],
    node_labels: List[str],
    relationship_types: List[str],
    node_properties: Optional[Dict[str, List[str]]] = None,
    relationship_properties: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Build a pruned structured schema from a selected set of labels/rel types.

    Falls back to the full, unpruned schema if no known node label was
    selected. Relationships are kept if their type was explicitly selected;
    otherwise (no relationship types selected), a relationship is kept when
    both its endpoints are among the selected node labels. `node_properties`/
    `relationship_properties` (label/type -> property names to keep), if
    given, additionally narrow each selected label's/type's property list
    (see `_filter_props`); by default every property of a selected label/type
    is kept, as before.
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

    pruned_node_props = _filter_props({n: node_props[n] for n in selected_nodes}, node_properties)
    pruned_rel_props = _filter_props(
        {t: rel_props[t] for t in kept_rel_types if t in rel_props}, relationship_properties
    )

    return {
        "node_props": pruned_node_props,
        "relationships": pruned_relationships,
        "rel_props": pruned_rel_props,
        "metadata": structured_schema.get("metadata", {}),
    }


def exact_match_prune(
    structured_schema: Dict[str, Any],
    question: str,
    components: Iterable[SchemaComponentLike] = DEFAULT_SCHEMA_COMPONENTS,
) -> Dict[str, Any]:
    """Keep schema elements mentioned (substring match) in `question`.

    `components` selects which schema element kinds are matched against the
    question (see `SchemaComponent`); by default only entity types (node
    labels) are matched, exactly as before. Node labels always anchor the
    selection — their matches decide which node types, and (via shared
    endpoints) which relationships, survive. Enabling `RELATIONSHIP_TYPES`
    additionally matches relationship type names directly, instead of only
    inferring kept relationships from selected node labels' endpoints.
    Enabling `NODE_PROPERTIES`/`RELATIONSHIP_PROPERTIES` additionally narrows
    each selected label's/type's properties to the ones mentioned.
    """
    components = _normalize_components(components)
    q_tokens = _normalize_tokens(question)
    node_props = structured_schema.get("node_props", {}) or {}
    rel_props = structured_schema.get("rel_props", {}) or {}

    mentioned_nodes = [n for n in node_props if _mentioned(n, q_tokens)]

    mentioned_rel_types = (
        [t for t in rel_props if _mentioned(t, q_tokens)]
        if SchemaComponent.RELATIONSHIP_TYPES in components
        else []
    )

    node_properties = None
    if SchemaComponent.NODE_PROPERTIES in components:
        node_properties = {
            n: [p.get("property") for p in props if _mentioned(p.get("property", ""), q_tokens)]
            for n, props in node_props.items()
        }

    relationship_properties = None
    if SchemaComponent.RELATIONSHIP_PROPERTIES in components:
        relationship_properties = {
            t: [p.get("property") for p in props if _mentioned(p.get("property", ""), q_tokens)]
            for t, props in rel_props.items()
        }

    return _apply_selection(
        structured_schema,
        mentioned_nodes,
        mentioned_rel_types,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
    )


def mask_entities(question: str, nlp: Any) -> str:
    """Replace each named entity span in `question` with its entity type label."""
    doc = nlp(question)
    masked = question
    for ent in reversed(list(doc.ents)):
        masked = masked[: ent.start_char] + ent.label_ + masked[ent.end_char :]
    return masked


def ner_exact_match_prune(
    structured_schema: Dict[str, Any],
    question: str,
    nlp: Any,
    components: Iterable[SchemaComponentLike] = DEFAULT_SCHEMA_COMPONENTS,
) -> Dict[str, Any]:
    """`exact_match_prune`, but named entities are masked first to reduce value/field-name confusion."""
    return exact_match_prune(structured_schema, mask_entities(question, nlp), components=components)


def _similarity_matcher(nlp: Any, question: str, threshold: float):
    """Build the `_similar(term) -> bool` closure `similarity_prune`/`similarity_prune_nodes_only` share."""
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

    return _similar


def similarity_prune(
    structured_schema: Dict[str, Any], question: str, nlp: Any, threshold: float = 0.5
) -> Dict[str, Any]:
    """Keep labels/types/properties whose word-vector similarity to the question exceeds `threshold`.

    Unlike `exact_match_prune`, there is no full-schema fallback: if nothing
    is similar enough, the pruned schema is empty.
    """
    _similar = _similarity_matcher(nlp, question, threshold)

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


def similarity_prune_nodes_only(
    structured_schema: Dict[str, Any], question: str, nlp: Any, threshold: float = 0.5
) -> Dict[str, Any]:
    """`similarity_prune`, but only node-label similarity anchors the selection.

    A less aggressive fallback than `similarity_prune`: node labels are
    matched by word-vector similarity same as before, but relationship
    types/properties are not independently matched — instead, relationships
    are kept when both endpoints are among the selected labels, and every
    property of a selected label is kept (see `_apply_selection`), the same
    "nodes only" semantics `exact_match_prune`/`ner_exact_match_prune` get
    from `components=DEFAULT_SCHEMA_COMPONENTS`. Falls back to the full
    schema if no label is similar enough.
    """
    _similar = _similarity_matcher(nlp, question, threshold)
    node_props = structured_schema.get("node_props", {}) or {}
    matched_nodes = [n for n in node_props if _similar(n)]
    return _apply_selection(structured_schema, matched_nodes, [])


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


def _llm_select(structured_schema: Dict[str, Any], question: str, llm: Any) -> SchemaSelection:
    """The `llm.with_structured_output` call `llm_prune`/`llm_prune_nodes_only` share."""
    full_schema_text = format_schema(structured_schema, is_enhanced=True)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _LLM_PRUNING_SYSTEM), ("human", _LLM_PRUNING_TEMPLATE)]
    )
    chain = prompt | llm.with_structured_output(SchemaSelection, method="json_schema")
    return chain.invoke({"schema": full_schema_text, "question": question})


def llm_prune(structured_schema: Dict[str, Any], question: str, llm: Any) -> Dict[str, Any]:
    """Ask `llm` (via structured/JSON-schema output) which labels/relationship types are relevant."""
    selection = _llm_select(structured_schema, question, llm)
    return _apply_selection(structured_schema, selection.node_labels, selection.relationship_types)


def llm_prune_nodes_only(structured_schema: Dict[str, Any], question: str, llm: Any) -> Dict[str, Any]:
    """`llm_prune`, but only the model's node-label selection is used.

    A less aggressive fallback than `llm_prune`: the model is still asked
    for both node labels and relationship types (same call), but only
    `node_labels` is used — relationships are instead kept via shared
    endpoints among the selected labels (see `_apply_selection`), the same
    "nodes only" semantics `exact_match_prune`/`ner_exact_match_prune` get
    from `components=DEFAULT_SCHEMA_COMPONENTS`.
    """
    selection = _llm_select(structured_schema, question, llm)
    return _apply_selection(structured_schema, selection.node_labels, [])


def structured_schema_to_linkml(
    structured_schema: Dict[str, Any],
    components: Iterable[SchemaComponentLike] = DEFAULT_SCHEMA_COMPONENTS,
) -> str:
    """Convert a structured Neo4j schema into a LinkML YAML schema for `ie_prune`.

    Generated for schema-grounded information extraction (e.g. SchemaLink,
    https://github.com/BioDataUniMI/schemalink-engine), not for ontology
    grounding — no `id_prefixes`/`annotations` are emitted, since `ie_prune`
    only needs entity/relation/attribute *presence*, not links to external
    ontology IDs.

    Each node label becomes a class (`is_a: NamedEntity`) named *exactly*
    like the label. If `RELATIONSHIP_TYPES` is in `components`, each
    relationship type additionally becomes a `Triple` class (named exactly
    like the type) plus a companion `"{type}__Predicate"` class — using the
    type's first start/end label pair as the subject/object range (a
    relationship type connecting more than one label pair only gets its
    first pair modeled; a documented simplification, not a limitation of the
    underlying schema). `NODE_PROPERTIES`/`RELATIONSHIP_PROPERTIES`, if in
    `components`, add each property as an untyped LinkML attribute on the
    corresponding class.

    This verbatim naming convention (class name == schema label/type) is
    exactly what `ie_prune` relies on to map the extraction engine's output
    keys back onto the original schema — don't rename classes downstream
    without updating `ie_prune` to match.
    """
    components = _normalize_components(components)
    node_props = structured_schema.get("node_props", {}) or {}
    rel_props = structured_schema.get("rel_props", {}) or {}
    relationships = structured_schema.get("relationships", []) or []

    classes: Dict[str, Any] = {}

    for label, props in node_props.items():
        node_class: Dict[str, Any] = {"is_a": "NamedEntity"}
        if SchemaComponent.NODE_PROPERTIES in components:
            node_class["attributes"] = {p.get("property"): {} for p in props if p.get("property")}
        classes[label] = node_class

    if SchemaComponent.RELATIONSHIP_TYPES in components:
        first_endpoints: Dict[str, Any] = {}
        for r in relationships:
            rel_type = r.get("type")
            if rel_type and rel_type not in first_endpoints:
                first_endpoints[rel_type] = (r.get("start"), r.get("end"))

        for rel_type, props in rel_props.items():
            start, end = first_endpoints.get(rel_type, (None, None))
            predicate_class = f"{rel_type}__Predicate"
            rel_class: Dict[str, Any] = {
                "is_a": "Triple",
                "slot_usage": {
                    "subject": {"range": start} if start else {},
                    "object": {"range": end} if end else {},
                    "predicate": {"range": predicate_class},
                },
            }
            if SchemaComponent.RELATIONSHIP_PROPERTIES in components:
                rel_class["attributes"] = {p.get("property"): {} for p in props if p.get("property")}
            classes[rel_type] = rel_class
            classes[predicate_class] = {
                "is_a": "RelationshipType",
                "attributes": {"id": {"pattern": rel_type}},
            }

    schema = {
        "id": "https://text2cypher-composer.local/generated-schema",
        "name": "generated_schema",
        "imports": ["ontogpt:core", "linkml:types"],
        "classes": classes,
    }
    return yaml.safe_dump(schema, sort_keys=False)


def _extraction_mentions(extraction: Dict[str, Any], class_name: str) -> List[Dict[str, Any]]:
    return (extraction.get(class_name) or {}).get("mentions", []) or []


def ie_prune(
    structured_schema: Dict[str, Any],
    question: str,
    ie_engine: Any,
    components: Iterable[SchemaComponentLike] = DEFAULT_SCHEMA_COMPONENTS,
) -> Dict[str, Any]:
    """Keep schema elements a schema-grounded IE engine actually found in `question`.

    Unlike `exact_match_prune`/`ner_exact_match_prune`, this does no substring
    matching itself: `structured_schema_to_linkml` turns `structured_schema`
    into a LinkML schema (restricted to what `components` asks for — see
    `SchemaComponent`), and `ie_engine` is called as `ie_engine(schema_yaml,
    question)`, the same "bring your own object" duck-typed pattern `nlp`/
    `llm` use elsewhere in this module. It must return a dict shaped like
    SchemaLink's (https://github.com/BioDataUniMI/schemalink-engine) output:
    `{class_name: {"mentions": [{...}, ...]}}`, one entry per class that was
    actually asked for, with an empty (or absent) `"mentions"` list for a
    class found nowhere in the text — the ontology-grounding fields
    SchemaLink can add to each mention (`"id"`, etc.) are ignored here.

    Node labels/relationship types are kept when their (verbatim-named)
    class has at least one mention. `NODE_PROPERTIES`/`RELATIONSHIP_PROPERTIES`
    narrow a kept label's/type's properties to whichever property names
    appear as keys in its mentions — as with `_apply_selection` generally, a
    label/type with no property keys found keeps every property, rather than
    being left with none.

    `schemalink_ie_engine()` (see `schemalink_adapter`) is a ready-made
    `ie_engine` backed by the real `schemalink-engine` PyPI package
    (`pip install schemalink-engine`) satisfying this exact contract — pass
    its result straight through as `ie_engine`.
    """
    components = _normalize_components(components)
    schema_yaml = structured_schema_to_linkml(structured_schema, components)
    extraction = ie_engine(schema_yaml, question)

    node_props = structured_schema.get("node_props", {}) or {}
    rel_props = structured_schema.get("rel_props", {}) or {}

    node_labels = [n for n in node_props if _extraction_mentions(extraction, n)]

    relationship_types = (
        [t for t in rel_props if _extraction_mentions(extraction, t)]
        if SchemaComponent.RELATIONSHIP_TYPES in components
        else []
    )

    def _mentioned_properties(class_name: str, props: List[Dict[str, Any]]) -> List[str]:
        keys_seen: set = set()
        for mention in _extraction_mentions(extraction, class_name):
            keys_seen.update(mention.keys())
        return [p.get("property") for p in props if p.get("property") in keys_seen]

    node_properties = None
    if SchemaComponent.NODE_PROPERTIES in components:
        node_properties = {n: _mentioned_properties(n, props) for n, props in node_props.items()}

    relationship_properties = None
    if SchemaComponent.RELATIONSHIP_PROPERTIES in components:
        relationship_properties = {t: _mentioned_properties(t, props) for t, props in rel_props.items()}

    return _apply_selection(
        structured_schema,
        node_labels,
        relationship_types,
        node_properties=node_properties,
        relationship_properties=relationship_properties,
    )


def resolve_schema_text(
    graph: Any,
    mode: SchemaModeLike,
    question: str,
    llm: Optional[Any] = None,
    nlp: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    schema_components: Iterable[SchemaComponentLike] = DEFAULT_SCHEMA_COMPONENTS,
    ie_engine: Optional[Any] = None,
) -> str:
    """Compute the schema text to place in the prompt, per `mode` (see `SchemaMode`).

    `schema_components` is only used by `mode="exact_match"`/`"ner_exact_match"`/
    `"ie_extraction"` (see `SchemaComponent`); ignored otherwise.
    """
    mode = SchemaMode(mode)

    if mode == SchemaMode.SCHEMA:
        return get_schema(graph, is_enhanced=False)

    if mode == SchemaMode.ENHANCED:
        return get_schema(graph, is_enhanced=True)

    if mode == SchemaMode.EXACT_MATCH:
        structured = get_structured_schema(graph, is_enhanced=True)
        return format_schema(
            exact_match_prune(structured, question, components=schema_components), is_enhanced=True
        )

    if mode == SchemaMode.NER_EXACT_MATCH:
        if nlp is None:
            raise ValueError(
                "schema_mode='ner_exact_match' requires an `nlp` argument: a loaded NLP "
                "pipeline with named-entity recognition (e.g. spaCy's en_ner_bionlp13cg_md)."
            )
        structured = get_structured_schema(graph, is_enhanced=True)
        return format_schema(
            ner_exact_match_prune(structured, question, nlp, components=schema_components), is_enhanced=True
        )

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

    if mode == SchemaMode.IE_EXTRACTION:
        if ie_engine is None:
            raise ValueError(
                "schema_mode='ie_extraction' requires an `ie_engine` argument: a callable "
                "ie_engine(schema_yaml, question) -> dict performing schema-grounded "
                "information extraction — pass schemalink_ie_engine() for a ready-made one "
                "backed by the real schemalink-engine package (pip install schemalink-engine), "
                "or see `ie_prune` for the contract to implement your own."
            )
        structured = get_structured_schema(graph, is_enhanced=True)
        return format_schema(
            ie_prune(structured, question, ie_engine, components=schema_components), is_enhanced=True
        )

    raise ValueError(f"Unknown schema_mode: {mode}")


def resolve_cascade_mode_levels(
    graph: Any,
    mode: SchemaModeLike,
    question: str,
    llm: Optional[Any] = None,
    nlp: Optional[Any] = None,
    similarity_threshold: float = 0.5,
    ie_engine: Optional[Any] = None,
    skip_narrow: bool = False,
) -> List[Tuple[CascadeModeLevel, str]]:
    """Resolve the schema text for each rung of the `cascade_mode` cascade, in order.

    Returns `[(level, schema_text), ...]`, trying progressively less
    aggressive pruning: `CascadeModeLevel.NARROW` (node labels,
    relationship types, and properties all narrowed to the question —
    `ALL_SCHEMA_COMPONENTS`; skipped entirely if `skip_narrow`),
    `CascadeModeLevel.NODES_ONLY` (only node labels matched —
    `DEFAULT_SCHEMA_COMPONENTS`; relationships kept via shared endpoints,
    every property of a selected label/type kept), then always
    `CascadeModeLevel.FULL` last (the unpruned schema).

    `mode` must be a pruning schema_mode — `"exact_match"`,
    `"ner_exact_match"`, `"similarity"`, `"llm_pruning"`, or
    `"ie_extraction"`. `"schema"`/`"enhanced"` have nothing to prune from and
    raise `ValueError`, same as passing `cascade_mode=True` for a
    non-schema technique would in `run()`. `llm`/`nlp`/`ie_engine` are
    required by the same modes `resolve_schema_text` requires them for.
    """
    mode = SchemaMode(mode)
    if mode in (SchemaMode.SCHEMA, SchemaMode.ENHANCED):
        raise ValueError(
            f"schema_mode='{mode.value}' has nothing to prune — cascade_mode requires a "
            "pruning schema_mode (exact_match, ner_exact_match, similarity, llm_pruning, or "
            "ie_extraction)."
        )
    if mode in (SchemaMode.NER_EXACT_MATCH, SchemaMode.SIMILARITY) and nlp is None:
        raise ValueError(f"schema_mode='{mode.value}' requires an `nlp` argument.")
    if mode == SchemaMode.LLM_PRUNING and llm is None:
        raise ValueError("schema_mode='llm_pruning' requires a structured-output-capable model.")
    if mode == SchemaMode.IE_EXTRACTION and ie_engine is None:
        raise ValueError("schema_mode='ie_extraction' requires an `ie_engine` argument.")

    is_enhanced = mode != SchemaMode.SIMILARITY
    structured = get_structured_schema(graph, is_enhanced=is_enhanced)

    levels: List[Tuple[CascadeModeLevel, str]] = []

    if not skip_narrow:
        if mode == SchemaMode.EXACT_MATCH:
            narrow = exact_match_prune(structured, question, components=ALL_SCHEMA_COMPONENTS)
        elif mode == SchemaMode.NER_EXACT_MATCH:
            narrow = ner_exact_match_prune(structured, question, nlp, components=ALL_SCHEMA_COMPONENTS)
        elif mode == SchemaMode.SIMILARITY:
            narrow = similarity_prune(structured, question, nlp, threshold=similarity_threshold)
        elif mode == SchemaMode.LLM_PRUNING:
            narrow = llm_prune(structured, question, llm)
        else:  # IE_EXTRACTION
            narrow = ie_prune(structured, question, ie_engine, components=ALL_SCHEMA_COMPONENTS)
        levels.append((CascadeModeLevel.NARROW, format_schema(narrow, is_enhanced=is_enhanced)))

    if mode == SchemaMode.EXACT_MATCH:
        nodes_only = exact_match_prune(structured, question, components=DEFAULT_SCHEMA_COMPONENTS)
    elif mode == SchemaMode.NER_EXACT_MATCH:
        nodes_only = ner_exact_match_prune(structured, question, nlp, components=DEFAULT_SCHEMA_COMPONENTS)
    elif mode == SchemaMode.SIMILARITY:
        nodes_only = similarity_prune_nodes_only(structured, question, nlp, threshold=similarity_threshold)
    elif mode == SchemaMode.LLM_PRUNING:
        nodes_only = llm_prune_nodes_only(structured, question, llm)
    else:  # IE_EXTRACTION
        nodes_only = ie_prune(structured, question, ie_engine, components=DEFAULT_SCHEMA_COMPONENTS)
    levels.append((CascadeModeLevel.NODES_ONLY, format_schema(nodes_only, is_enhanced=is_enhanced)))

    levels.append((CascadeModeLevel.FULL, get_schema(graph, is_enhanced=is_enhanced)))

    return levels
