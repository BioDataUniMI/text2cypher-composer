from text2cypher_composer import Technique, describe_technique, list_technique_info, list_techniques
from text2cypher_composer.techniques import (
    ALL_SCHEMA_COMPONENTS,
    DEFAULT_SCHEMA_COMPONENTS,
    OUTPUT_AUGMENTED_TECHNIQUES,
    RAG_TECHNIQUES,
    SCHEMA_TECHNIQUES,
    SchemaComponent,
)


def test_list_techniques_matches_enum():
    assert list_techniques() == [t.value for t in Technique]


def test_describe_technique_flags():
    info = describe_technique("Schema+RAG+O")
    assert info.uses_schema is True
    assert info.uses_rag is True
    assert info.uses_output is True

    info = describe_technique("vanilla")
    assert info.uses_schema is False
    assert info.uses_rag is False
    assert info.uses_output is False


def test_output_augmented_implies_rag():
    # every "+O" technique must also be a RAG technique (it augments RAG examples).
    assert OUTPUT_AUGMENTED_TECHNIQUES <= RAG_TECHNIQUES


def test_list_technique_info_covers_every_technique():
    infos = list_technique_info()
    assert {info.technique for info in infos} == {t.value for t in Technique}


def test_every_schema_or_rag_technique_is_a_real_technique():
    # guards against a stale entry left behind after removing/renaming a Technique
    assert SCHEMA_TECHNIQUES <= set(Technique)
    assert RAG_TECHNIQUES <= set(Technique)
    assert OUTPUT_AUGMENTED_TECHNIQUES <= set(Technique)


def test_all_schema_components_covers_every_component():
    # the cascade_mode "narrow" rung's aggressiveness comes from this being everything
    assert ALL_SCHEMA_COMPONENTS == set(SchemaComponent)
    assert DEFAULT_SCHEMA_COMPONENTS < ALL_SCHEMA_COMPONENTS
