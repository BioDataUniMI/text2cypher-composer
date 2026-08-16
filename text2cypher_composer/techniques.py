from dataclasses import dataclass
from enum import Enum
from typing import List, Union


class Technique(str, Enum):
    """The six Text2Cypher prompting strategies from the bio2C notebooks."""

    VANILLA = "vanilla"
    SCHEMA = "Schema"
    RAG = "RAG"
    RAG_O = "RAG+O"
    SCHEMA_RAG = "Schema+RAG"
    SCHEMA_RAG_O = "Schema+RAG+O"


SCHEMA_TECHNIQUES = {Technique.SCHEMA, Technique.SCHEMA_RAG, Technique.SCHEMA_RAG_O}
RAG_TECHNIQUES = {Technique.RAG, Technique.RAG_O, Technique.SCHEMA_RAG, Technique.SCHEMA_RAG_O}
OUTPUT_AUGMENTED_TECHNIQUES = {Technique.RAG_O, Technique.SCHEMA_RAG_O}

TechniqueLike = Union[str, Technique]


@dataclass
class TechniqueInfo:
    """What a technique needs from `run()`'s `database`/`dataset` arguments."""

    technique: str
    uses_schema: bool
    uses_rag: bool
    uses_output: bool


def list_techniques() -> List[str]:
    """Return the string values accepted by `run()`'s `technique` argument."""
    return [t.value for t in Technique]


def describe_technique(technique: TechniqueLike) -> TechniqueInfo:
    """Return whether `technique` uses the enhanced schema, RAG, and/or output-augmented RAG."""
    t = Technique(technique)
    return TechniqueInfo(
        technique=t.value,
        uses_schema=t in SCHEMA_TECHNIQUES,
        uses_rag=t in RAG_TECHNIQUES,
        uses_output=t in OUTPUT_AUGMENTED_TECHNIQUES,
    )


def list_technique_info() -> List[TechniqueInfo]:
    """Return `describe_technique` for every available technique."""
    return [describe_technique(t) for t in Technique]


class SchemaMode(str, Enum):
    """How the graph schema is derived/pruned before being placed in the prompt.

    Only meaningful when `technique` uses the schema (see `TechniqueInfo.uses_schema`).

    - "schema" (default): the plain schema as returned by Neo4j, no per-property stats.
    - "enhanced": adds per-property examples/min-max stats.
    - "exact_match": prunes the enhanced schema to node labels mentioned (as a
      substring match) in the question, plus relationships between two
      mentioned labels.
    - "ner_exact_match": same as "exact_match", but named entities in the
      question are first masked with their entity type (via a user-supplied
      NLP pipeline) to avoid entity *values* being confused with schema
      field names.
    - "similarity": prunes the base schema to labels/types/properties whose
      word-vector similarity to the question exceeds a threshold (via a
      user-supplied NLP pipeline with word vectors).
    - "llm_pruning": asks the model itself (via structured/JSON-schema
      output) which labels and relationship types are relevant.
    - "ie_extraction": runs schema-grounded information extraction (NER +
      relation extraction) over the question via a user-supplied `ie_engine`,
      and keeps exactly the entity types/relationship types/properties it
      found — no substring matching involved, unlike "exact_match"/
      "ner_exact_match". See `ie_prune`.
    """

    SCHEMA = "schema"
    ENHANCED = "enhanced"
    EXACT_MATCH = "exact_match"
    NER_EXACT_MATCH = "ner_exact_match"
    SIMILARITY = "similarity"
    LLM_PRUNING = "llm_pruning"
    IE_EXTRACTION = "ie_extraction"


SchemaModeLike = Union[str, SchemaMode]


def list_schema_modes() -> List[str]:
    """Return the string values accepted by `run()`'s `schema_mode` argument."""
    return [m.value for m in SchemaMode]


class SchemaComponent(str, Enum):
    """Schema element kinds the filtering modes can match against.

    Meaningful for `schema_mode="exact_match"`/`"ner_exact_match"` (literal
    substring matching against the question) and `"ie_extraction"`
    (schema-grounded information extraction) — see `resolve_schema_text`'s
    `schema_components` argument. For "exact_match"/"ner_exact_match", node/
    entity types always anchor the selection; the other components only
    narrow further what's kept once an entity type is selected. For
    "ie_extraction", `components` instead controls what's *asked of* the
    extraction engine in the first place (see `structured_schema_to_linkml`).

    - "entity_types" (default): node labels mentioned in the question.
    - "relationship_types": relationship types mentioned in the question.
    - "node_properties": node property names mentioned in the question.
    - "relationship_properties": relationship property names mentioned in
      the question.
    """

    ENTITY_TYPES = "entity_types"
    RELATIONSHIP_TYPES = "relationship_types"
    NODE_PROPERTIES = "node_properties"
    RELATIONSHIP_PROPERTIES = "relationship_properties"


SchemaComponentLike = Union[str, SchemaComponent]

DEFAULT_SCHEMA_COMPONENTS = frozenset({SchemaComponent.ENTITY_TYPES})
ALL_SCHEMA_COMPONENTS = frozenset(SchemaComponent)


def list_schema_components() -> List[str]:
    """Return the string values accepted by `run()`'s `schema_components` argument."""
    return [c.value for c in SchemaComponent]


class CascadeModeLevel(str, Enum):
    """A rung of the `cascade_mode` cascade (see `run()`'s `cascade_mode` argument).

    - "narrow": the most aggressively pruned schema — node labels,
      relationship types, and node/relationship properties all narrowed to
      what's relevant to the question (`ALL_SCHEMA_COMPONENTS`).
    - "nodes_only": a less aggressive fallback — only node labels are
      matched (`DEFAULT_SCHEMA_COMPONENTS`); relationships are kept via
      shared endpoints among the selected labels, and every property of a
      selected label/type is kept.
    - "full": the unpruned schema — the final fallback, always tried last.
    """

    NARROW = "narrow"
    NODES_ONLY = "nodes_only"
    FULL = "full"
