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
    """

    SCHEMA = "schema"
    ENHANCED = "enhanced"
    EXACT_MATCH = "exact_match"
    NER_EXACT_MATCH = "ner_exact_match"
    SIMILARITY = "similarity"
    LLM_PRUNING = "llm_pruning"


SchemaModeLike = Union[str, SchemaMode]


def list_schema_modes() -> List[str]:
    """Return the string values accepted by `run()`'s `schema_mode` argument."""
    return [m.value for m in SchemaMode]
