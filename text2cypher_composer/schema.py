#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Enhanced Neo4j schema extraction.

Adapted from langchain's Neo4jGraph enhanced-schema utilities, with a
project-specific extension: LIST properties are summarized via a sample
item (``item_examples``) rather than a raw min/max size range.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

BASE_KG_BUILDER_LABEL = "__KGBuilder__"
BASE_ENTITY_LABEL = "__Entity__"
EXCLUDED_LABELS = ["_Bloom_Perspective_", "_Bloom_Scene_"]
EXCLUDED_RELS = ["_Bloom_HAS_SCENE_"]
EXHAUSTIVE_SEARCH_LIMIT = 1_700_000
LIST_LIMIT = 128
DISTINCT_VALUE_LIMIT = 10

NODE_PROPERTIES_QUERY = (
    "CALL apoc.meta.data({sample: $SAMPLE}) "
    "YIELD label, other, elementType, type, property "
    "WHERE NOT type = 'RELATIONSHIP' AND elementType = 'node' "
    "AND NOT label IN $EXCLUDED_LABELS "
    "WITH label AS nodeLabel, collect({property:property, type:type}) AS properties "
    "RETURN {label: nodeLabel, properties: properties} AS output"
)

REL_PROPERTIES_QUERY = (
    "CALL apoc.meta.data({sample: $SAMPLE}) "
    "YIELD label, other, elementType, type, property "
    "WHERE NOT type = 'RELATIONSHIP' AND elementType = 'relationship' "
    "AND NOT label in $EXCLUDED_RELS "
    "WITH label AS relType, collect({property:property, type:type}) AS properties "
    "RETURN {type: relType, properties: properties} AS output"
)

REL_QUERY = (
    "CALL apoc.meta.data({sample: $SAMPLE}) "
    "YIELD label, other, elementType, type, property "
    "WHERE type = 'RELATIONSHIP' AND elementType = 'node' "
    "UNWIND other AS other_node "
    "WITH * WHERE NOT label IN $EXCLUDED_LABELS "
    "AND NOT other_node IN $EXCLUDED_LABELS "
    "RETURN {start: label, type: property, end: toString(other_node)} AS output"
)

INDEX_QUERY = (
    "CALL apoc.schema.nodes() YIELD label, properties, type, size, valuesSelectivity "
    "WHERE type = 'RANGE' RETURN *, "
    "size * valuesSelectivity as distinctValues"
)

SCHEMA_COUNTS_QUERY = (
    "CALL apoc.meta.graph({sample: $SAMPLE, maxRels: $MAX_RELS}) "
    "YIELD nodes, relationships "
    "RETURN nodes, [rel in relationships | {name: apoc.any.property(rel, 'type'), "
    "count: apoc.any.property(rel, 'count')}] AS relationships"
)


def _clean_string_values(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


def get_schema(graph: Any, is_enhanced: bool = False, sample: int = 1000) -> str:
    """Return the graph schema as a formatted string (see `format_schema`)."""
    structured_schema = get_structured_schema(graph, is_enhanced=is_enhanced, sample=sample)
    return format_schema(structured_schema, is_enhanced)


def get_structured_schema(graph: Any, is_enhanced: bool = False, sample: int = 1000) -> Dict[str, Any]:
    """Return {node_props, rel_props, relationships, metadata} for `graph`."""
    node_properties = [
        data["output"]
        for data in graph.query(
            NODE_PROPERTIES_QUERY,
            params={
                "EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL],
                "SAMPLE": sample,
            },
        )
    ]

    rel_properties = [
        data["output"]
        for data in graph.query(
            REL_PROPERTIES_QUERY,
            params={"EXCLUDED_RELS": EXCLUDED_RELS, "SAMPLE": sample},
        )
    ]

    relationships = [
        data["output"]
        for data in graph.query(
            REL_QUERY,
            params={
                "EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL],
                "SAMPLE": sample,
            },
        )
    ]

    try:
        constraint = graph.query("SHOW CONSTRAINTS")
        index = graph.query(INDEX_QUERY)
    except Exception:
        constraint, index = [], []

    structured_schema: Dict[str, Any] = {
        "node_props": {el["label"]: el["properties"] for el in node_properties},
        "rel_props": {el["type"]: el["properties"] for el in rel_properties},
        "relationships": relationships,
        "metadata": {"constraint": constraint, "index": index},
    }
    if is_enhanced:
        enhance_schema(graph, structured_schema)
    return structured_schema


def _format_property(prop: Dict[str, Any]) -> Optional[str]:
    if prop["type"] == "STRING" and prop.get("values"):
        return f'Example: "{_clean_string_values(prop["values"][0])}"'
    elif prop["type"] in ["INTEGER", "FLOAT", "DATE", "DATE_TIME", "LOCAL_DATE_TIME"]:
        if prop.get("min") and prop.get("max"):
            return f"Min: {prop['min']}, Max: {prop['max']}"
        return f'Example: "{prop["values"][0]}"' if prop.get("values") else ""
    elif prop["type"] == "LIST":
        examples = prop.get("item_examples") or prop.get("values")
        if examples and isinstance(examples, list) and examples:
            return f'Item Example: "{_clean_string_values(str(examples[0]))}"'
        return ""
    return None


def _format_properties(property_dict: Dict[str, Any], is_enhanced: bool) -> List[str]:
    formatted_props = []
    if is_enhanced:
        for label, props in property_dict.items():
            formatted_props.append(f"- **{label}**")
            for prop in props:
                example = _format_property(prop)
                if example is not None:
                    formatted_props.append(f"  - `{prop['property']}`: {prop['type']} {example}")
    else:
        for label, props in property_dict.items():
            props_str = ", ".join(f"{prop['property']}: {prop['type']}" for prop in props)
            formatted_props.append(f"{label} {{{props_str}}}")
    return formatted_props


def _format_relationships(rels: List[Dict[str, Any]]) -> List[str]:
    return [f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in rels]


def format_schema(schema: Dict[str, Any], is_enhanced: bool) -> str:
    formatted_node_props = _format_properties(schema["node_props"], is_enhanced)
    formatted_rel_props = _format_properties(schema["rel_props"], is_enhanced)
    formatted_rels = _format_relationships(schema["relationships"])
    return "\n".join(
        [
            "Node properties:",
            "\n".join(formatted_node_props),
            "The relationships:",
            "\n".join(formatted_rels),
            "Relationship properties:",
            "\n".join(formatted_rel_props),
        ]
    )


def _build_str_clauses(
    prop_name: str,
    graph: Any,
    label_or_type: str,
    exhaustive: bool,
    prop_index: Optional[List[Any]] = None,
) -> Tuple[List[str], List[str]]:
    with_clauses: List[str] = []
    return_clauses: List[str] = []
    if (
        not exhaustive
        and prop_index
        and prop_index[0].get("size") > 0
        and prop_index[0].get("distinctValues") <= DISTINCT_VALUE_LIMIT
    ):
        distinct_values = graph.query(
            f"CALL apoc.schema.properties.distinct('{label_or_type}', '{prop_name}') YIELD value"
        )[0]["value"]
        return_clauses.append(f"values: {distinct_values}, distinct_count: {len(distinct_values)}")
    else:
        with_clauses.append(
            f"collect(distinct substring(toString(n.`{prop_name}`), 0, 50)) AS `{prop_name}_values`"
        )
        if not exhaustive:
            return_clauses.append(f"values: `{prop_name}_values`")
        else:
            return_clauses.append(
                f"values: `{prop_name}_values`[..{DISTINCT_VALUE_LIMIT}],"
                f" distinct_count: size(`{prop_name}_values`)"
            )
    return with_clauses, return_clauses


def _build_list_clauses(prop_name: str) -> Tuple[str, str]:
    with_clause_size = (
        f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
        f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
    )
    with_clause_items = (
        f"collect(distinct substring(toString(head(n.`{prop_name}`)), 0, 50)) AS `{prop_name}_item_examples`"
    )
    with_clause = with_clause_size + ", " + with_clause_items
    return_clause = (
        f"item_examples: `{prop_name}_item_examples`, "
        f"min_size: `{prop_name}_size_min`, max_size: `{prop_name}_size_max`"
    )
    return with_clause, return_clause


def _build_num_date_clauses(
    prop_name: str, exhaustive: bool, prop_index: Optional[List[Any]] = None
) -> Tuple[List[str], List[str]]:
    with_clauses: List[str] = []
    return_clauses: List[str] = []
    if not prop_index and not exhaustive:
        with_clauses.append(f"collect(distinct toString(n.`{prop_name}`)) AS `{prop_name}_values`")
        return_clauses.append(f"values: `{prop_name}_values`")
    else:
        with_clauses.append(f"min(n.`{prop_name}`) AS `{prop_name}_min`")
        with_clauses.append(f"max(n.`{prop_name}`) AS `{prop_name}_max`")
        with_clauses.append(f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`")
        return_clauses.append(
            f"min: toString(`{prop_name}_min`), "
            f"max: toString(`{prop_name}_max`), "
            f"distinct_count: `{prop_name}_distinct`"
        )
    return with_clauses, return_clauses


def get_enhanced_schema_cypher(
    graph: Any,
    structured_schema: Dict[str, Any],
    label_or_type: str,
    properties: List[Dict[str, Any]],
    exhaustive: bool,
    sample_size: int = 5,
    is_relationship: bool = False,
) -> str:
    if is_relationship:
        match_clause = f"MATCH ()-[n:`{label_or_type}`]->()"
    else:
        match_clause = f"MATCH (n:`{label_or_type}`)"

    with_clauses: List[str] = []
    output_dict: Dict[str, str] = {}
    if not exhaustive:
        match_clause += f" WITH n LIMIT {sample_size}"

    for prop in properties:
        prop_name = prop["property"]
        prop_type = prop["type"]
        prop_index = (
            [
                el
                for el in structured_schema["metadata"]["index"]
                if el["label"] == label_or_type
                and el["properties"] == [prop_name]
                and el["type"] == "RANGE"
            ]
            if not exhaustive
            else None
        )
        return_clauses: List[str] = []
        if prop_type == "STRING":
            str_w_clauses, str_r_clauses = _build_str_clauses(
                prop_name=prop_name,
                graph=graph,
                label_or_type=label_or_type,
                exhaustive=exhaustive,
                prop_index=prop_index,
            )
            with_clauses += str_w_clauses
            return_clauses += str_r_clauses
        elif prop_type in ["INTEGER", "FLOAT", "DATE", "DATE_TIME", "LOCAL_DATE_TIME"]:
            num_date_w_clauses, num_date_r_clauses = _build_num_date_clauses(
                prop_name=prop_name, exhaustive=exhaustive, prop_index=prop_index
            )
            with_clauses += num_date_w_clauses
            return_clauses += num_date_r_clauses
        elif prop_type == "LIST":
            list_w_clause, list_r_clause = _build_list_clauses(prop_name=prop_name)
            with_clauses.append(list_w_clause)
            return_clauses.append(list_r_clause)
        elif prop_type in ["BOOLEAN", "POINT", "DURATION"]:
            continue
        if not return_clauses:
            continue
        output_dict[prop_name] = "{" + return_clauses.pop() + "}"

    if not output_dict:
        return f"{match_clause}\nRETURN {{}} AS output"

    with_clause = "WITH " + ",\n     ".join(with_clauses) if with_clauses else ""
    return_clause = "RETURN {" + ", ".join(f"`{k}`: {v}" for k, v in output_dict.items()) + "} AS output"
    return "\n".join([match_clause, with_clause, return_clause])


def enhance_properties(
    graph: Any,
    structured_schema: Dict[str, Any],
    prop_dict: Dict[str, Any],
    is_relationship: bool,
) -> None:
    name = prop_dict["name"]
    count = prop_dict["count"]
    excluded = EXCLUDED_RELS if is_relationship else EXCLUDED_LABELS
    if name in excluded:
        return
    props = structured_schema["rel_props" if is_relationship else "node_props"].get(name)
    if not props:
        return
    enhanced_cypher = get_enhanced_schema_cypher(
        graph=graph,
        structured_schema=structured_schema,
        label_or_type=name,
        properties=props,
        exhaustive=count < EXHAUSTIVE_SEARCH_LIMIT,
        is_relationship=is_relationship,
    )
    try:
        enhanced_info = graph.query(enhanced_cypher)[0]["output"]
        for prop in props:
            if prop["property"] in enhanced_info:
                prop.update(enhanced_info[prop["property"]])
    except Exception:
        return


def enhance_schema(graph: Any, structured_schema: Dict[str, Any]) -> None:
    """Enrich `structured_schema` in place with per-property statistics."""
    schema_counts = graph.query(
        SCHEMA_COUNTS_QUERY, params={"SAMPLE": -1, "MAX_RELS": 2_000_000}
    )
    for node in schema_counts[0]["nodes"]:
        enhance_properties(graph, structured_schema, node, is_relationship=False)
    for rel in schema_counts[0]["relationships"]:
        enhance_properties(graph, structured_schema, rel, is_relationship=True)
