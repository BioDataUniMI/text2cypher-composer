import json
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import yaml

from text2cypher_composer.schemalink_adapter import _filter_schema_yaml, schemalink_ie_engine

_SCHEMA_YAML = yaml.safe_dump(
    {
        "id": "https://text2cypher-composer.local/generated-schema",
        "name": "generated_schema",
        "imports": ["ontogpt:core", "linkml:types"],
        "classes": {
            "Gene": {"is_a": "NamedEntity", "attributes": {"Label": {}}},
            "miRNA": {"is_a": "NamedEntity", "attributes": {"Label": {}, "sequence_size": {}}},
            "transcribed_to": {
                "is_a": "Triple",
                "slot_usage": {
                    "subject": {"range": "Gene"},
                    "object": {"range": "miRNA"},
                    "predicate": {"range": "transcribed_to__Predicate"},
                },
                "attributes": {"source": {}},
            },
            "transcribed_to__Predicate": {
                "is_a": "RelationshipType",
                "attributes": {"id": {"pattern": "transcribed_to"}},
            },
        },
    },
    sort_keys=False,
)


def _classes(filtered_yaml):
    return yaml.safe_load(filtered_yaml)["classes"]


def test_filter_schema_yaml_keeps_everything_by_default():
    classes = _classes(_filter_schema_yaml(_SCHEMA_YAML, True, True, True))
    assert set(classes) == {"Gene", "miRNA", "transcribed_to", "transcribed_to__Predicate"}
    assert "attributes" in classes["Gene"]


def test_filter_schema_yaml_drops_relationship_types():
    classes = _classes(_filter_schema_yaml(_SCHEMA_YAML, True, False, True))
    assert set(classes) == {"Gene", "miRNA"}


def test_filter_schema_yaml_drops_properties_but_keeps_classes():
    classes = _classes(_filter_schema_yaml(_SCHEMA_YAML, True, True, False))
    assert set(classes) == {"Gene", "miRNA", "transcribed_to", "transcribed_to__Predicate"}
    assert "attributes" not in classes["Gene"]
    assert "attributes" not in classes["transcribed_to"]
    assert classes["transcribed_to"]["slot_usage"]["subject"]["range"] == "Gene"  # non-attribute keys survive


def test_filter_schema_yaml_drops_node_types_not_referenced_by_kept_relationships():
    # both node types are referenced by the (kept) relationship, so both survive
    classes = _classes(_filter_schema_yaml(_SCHEMA_YAML, False, True, True))
    assert {"Gene", "miRNA"} <= set(classes)


def test_filter_schema_yaml_node_types_false_and_relationships_false_drops_all_entities():
    # relationships dropped first, so nothing references Gene/miRNA anymore -> both dropped
    classes = _classes(_filter_schema_yaml(_SCHEMA_YAML, False, False, True))
    assert classes == {}


def test_schemalink_ie_engine_raises_helpful_import_error_when_package_missing():
    with pytest.raises(ImportError, match="schemalink-engine"):
        schemalink_ie_engine()


def _install_fake_schemalink_engine(monkeypatch, fake_run_extraction_pipeline):
    pipeline_module = types.ModuleType("schemalink_engine.pipeline")
    pipeline_module.run_extraction_pipeline = fake_run_extraction_pipeline
    package_module = types.ModuleType("schemalink_engine")
    package_module.pipeline = pipeline_module
    monkeypatch.setitem(sys.modules, "schemalink_engine", package_module)
    monkeypatch.setitem(sys.modules, "schemalink_engine.pipeline", pipeline_module)


def test_schemalink_ie_engine_writes_files_calls_pipeline_and_reads_output(monkeypatch, tmp_path):
    captured = {}

    def fake_run_extraction_pipeline(**kwargs):
        captured.update(kwargs)
        # the schema/text files must exist while the pipeline runs, before the scratch dir is cleaned up
        captured["schema_path_existed"] = os.path.exists(kwargs["schema_path"])
        captured["text_path_existed"] = os.path.exists(kwargs["text_path"])
        with open(kwargs["text_path"], encoding="utf-8") as f:
            captured["text_content"] = f.read()
        # simulate the real pipeline: writes its output relative to the current cwd
        os.makedirs("output", exist_ok=True)
        with open("output/generated_responses.json", "w", encoding="utf-8") as f:
            json.dump({"miRNA": {"mentions": [{"Label": "precursor"}]}}, f)

    _install_fake_schemalink_engine(monkeypatch, fake_run_extraction_pipeline)

    engine = schemalink_ie_engine()
    previous_cwd = os.getcwd()
    try:
        result = engine(_SCHEMA_YAML, "How many miRNAs mention precursor?")
    finally:
        os.chdir(previous_cwd)

    assert result == {"miRNA": {"mentions": [{"Label": "precursor"}]}}
    assert captured["with_dependencies"] is True
    assert captured["ground_entities"] is None
    assert captured["schema_path_existed"] is True
    assert captured["text_path_existed"] is True
    assert captured["text_content"] == "How many miRNAs mention precursor?"
    # cwd is restored after the call, even though the pipeline chdir'd into a scratch dir
    assert os.getcwd() == previous_cwd


def test_schemalink_ie_engine_returns_empty_dict_when_no_output_file(monkeypatch):
    def fake_run_extraction_pipeline(**kwargs):
        pass  # doesn't write anything

    _install_fake_schemalink_engine(monkeypatch, fake_run_extraction_pipeline)

    engine = schemalink_ie_engine()
    assert engine(_SCHEMA_YAML, "some question") == {}


def test_schemalink_ie_engine_forwards_with_dependencies_and_ground_entities(monkeypatch):
    captured = {}

    def fake_run_extraction_pipeline(**kwargs):
        captured.update(kwargs)
        os.makedirs("output", exist_ok=True)
        with open("output/generated_responses_without_dependencies.json", "w", encoding="utf-8") as f:
            json.dump({}, f)

    _install_fake_schemalink_engine(monkeypatch, fake_run_extraction_pipeline)

    engine = schemalink_ie_engine(with_dependencies=False, ground_entities={"mode": "auto"})
    engine(_SCHEMA_YAML, "q")

    assert captured["with_dependencies"] is False
    assert captured["ground_entities"] == {"mode": "auto"}


def test_schemalink_ie_engine_applies_component_filtering_before_pipeline_call(monkeypatch):
    captured = {}

    def fake_run_extraction_pipeline(**kwargs):
        with open(kwargs["schema_path"], encoding="utf-8") as f:
            captured["schema_yaml"] = f.read()
        os.makedirs("output", exist_ok=True)
        with open("output/generated_responses.json", "w", encoding="utf-8") as f:
            json.dump({}, f)

    _install_fake_schemalink_engine(monkeypatch, fake_run_extraction_pipeline)

    engine = schemalink_ie_engine(include_relationship_types=False)
    engine(_SCHEMA_YAML, "q")

    classes = yaml.safe_load(captured["schema_yaml"])["classes"]
    assert "transcribed_to" not in classes
    assert "Gene" in classes and "miRNA" in classes
