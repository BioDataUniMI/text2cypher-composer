"""A ready-made `ie_engine` adapter around the real `schemalink-engine` package.

Install with `pip install schemalink-engine` (or
`pip install "text2cypher-composer[schemalink]"`), and set an OpenAI API key
for it (`schemalink api-key set sk-...`) — see
https://github.com/BioDataUniMI/schemalink-engine.

Bridges `ie_prune`'s `ie_engine(schema_yaml, question) -> dict` contract
(see `schema_modes.ie_prune`) to `schemalink_engine.pipeline.run_extraction_pipeline`,
which works over schema/text *file paths* and writes its extraction output to
a JSON file rather than returning it: `schemalink_ie_engine()` writes both to
a scratch temp directory (so it doesn't litter the caller's cwd with the
pipeline's `generated/`/`output/` working directories) and reads the output
back into the shape `ie_prune` expects — including unwrapping the
`schemaResponse` nesting the real pipeline's output uses (see
`_normalize_extraction`).
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Callable, Dict, Optional

import yaml

_RELATIONSHIP_IS_A = ("Triple", "RelationshipType")


def _filter_schema_yaml(
    schema_yaml: str,
    include_node_types: bool,
    include_relationship_types: bool,
    include_properties: bool,
) -> str:
    """Restrict a LinkML schema (as produced by `structured_schema_to_linkml`)
    to the class/attribute kinds the caller actually wants SchemaLink to look
    for, on top of whatever `schema_components` already produced upstream.

    Dropping a `NamedEntity` class that a surviving `Triple` class still
    references as its subject/object range would leave that `Triple`
    dangling, so `include_node_types=False` only drops entity classes no
    kept relationship class still points to.
    """
    schema = yaml.safe_load(schema_yaml) or {}
    classes: Dict[str, Any] = dict(schema.get("classes") or {})

    if not include_relationship_types:
        classes = {name: cls for name, cls in classes.items() if cls.get("is_a") not in _RELATIONSHIP_IS_A}

    if not include_node_types:
        referenced = set()
        for cls in classes.values():
            if cls.get("is_a") == "Triple":
                slot_usage = cls.get("slot_usage") or {}
                for slot in ("subject", "object"):
                    rng = (slot_usage.get(slot) or {}).get("range")
                    if rng:
                        referenced.add(rng)
        classes = {
            name: cls
            for name, cls in classes.items()
            if not (cls.get("is_a") == "NamedEntity" and name not in referenced)
        }

    if not include_properties:
        classes = {name: {k: v for k, v in cls.items() if k != "attributes"} for name, cls in classes.items()}

    schema["classes"] = classes
    return yaml.safe_dump(schema, sort_keys=False)


def _normalize_extraction(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap schemalink-engine's raw per-class output to `ie_prune`'s expected shape.

    The real pipeline nests each class's payload under a `schemaResponse` key
    (`{class_name: {"schemaResponse": {"mentions": [...]}}}`) rather than
    exposing `mentions` at the top level of each class's entry, which is what
    `ie_prune`/`_extraction_mentions` (see `schema_modes.py`) expect — without
    this, every class looks mention-free and `ie_prune` silently falls back
    to the (near-)full schema. Entries without a `schemaResponse` key are
    passed through unchanged, so an already-flat shape still works too.
    """
    normalized: Dict[str, Any] = {}
    for class_name, entry in raw.items():
        if isinstance(entry, dict) and "schemaResponse" in entry:
            entry = entry["schemaResponse"] or {}
        normalized[class_name] = entry
    return normalized


def schemalink_ie_engine(
    *,
    include_node_types: bool = True,
    include_relationship_types: bool = True,
    include_properties: bool = True,
    with_dependencies: bool = True,
    ground_entities: Optional[Dict[str, Any]] = None,
) -> Callable[[str, str], Dict[str, Any]]:
    """Build an `ie_engine` callable backed by the real `schemalink-engine` package.

    Pass the result as `run(..., schema_mode="ie_extraction", ie_engine=schemalink_ie_engine())`.

    Args:
        include_node_types/include_relationship_types/include_properties:
            like `schema_components` elsewhere in this library (see
            `SchemaComponent`), but coarser (node/relationship *properties*
            aren't split) and independent of it — these filter the LinkML
            schema actually sent to SchemaLink on this call, regardless of
            what `schema_components` already restricted it to upstream. All
            `True` by default (ask SchemaLink about everything the schema
            given to it already contains).
        with_dependencies: forwarded to `run_extraction_pipeline` — dependency
            -aware extraction (each class's GPT call conditioned on its
            dependencies' results) if True (the default, and SchemaLink's
            own default), flat/independent extraction per class if False.
        ground_entities: forwarded to `run_extraction_pipeline` — e.g.
            `{"mode": "auto"}` to ground extracted entities to biomedical
            ontology IDs via OAK (requires the schema's classes to declare
            `annotators:`, and downloads ontology databases on first use).
            `None` (the default) leaves entities ungrounded.

    Returns:
        A callable `(schema_yaml, question) -> dict`, matching `ie_prune`'s
        `ie_engine` contract exactly, so `ie_prune`/`run()`'s schema
        filtering downstream needs no changes to work with it.

    Not thread-safe: `run_extraction_pipeline` reads/writes several files via
    relative paths, so each call temporarily `chdir`s into a scratch
    directory for the duration of the extraction — don't call the returned
    engine from multiple threads concurrently.
    """
    try:
        from schemalink_engine.pipeline import run_extraction_pipeline
    except ImportError as e:
        raise ImportError(
            "schemalink_ie_engine requires the `schemalink-engine` package. Install it with "
            '`pip install schemalink-engine` (or `pip install "text2cypher-composer[schemalink]"`).'
        ) from e

    def _engine(schema_yaml: str, question: str) -> Dict[str, Any]:
        filtered_yaml = _filter_schema_yaml(
            schema_yaml, include_node_types, include_relationship_types, include_properties
        )
        with tempfile.TemporaryDirectory(prefix="schemalink_") as tmp_dir:
            schema_path = os.path.join(tmp_dir, "schema.yaml")
            text_path = os.path.join(tmp_dir, "text.txt")
            with open(schema_path, "w", encoding="utf-8") as f:
                f.write(filtered_yaml)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(question)

            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                run_extraction_pipeline(
                    schema_path=schema_path,
                    text_path=text_path,
                    with_dependencies=with_dependencies,
                    ground_entities=ground_entities,
                    show_prompts=False,
                    show_results=False,
                    generate_prompts_only=False,
                    json_schema=False,
                )
                output_path = (
                    "output/generated_responses.json"
                    if with_dependencies
                    else "output/generated_responses_without_dependencies.json"
                )
                if not os.path.exists(output_path):
                    return {}
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                return _normalize_extraction(json.loads(content)) if content else {}
            finally:
                os.chdir(previous_cwd)

    return _engine
