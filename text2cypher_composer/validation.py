from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from CyVer import PropertiesValidator, SchemaValidator, SyntaxValidator


@dataclass
class CypherValidationReport:
    """CyVer's verdict on a Cypher query that failed to execute.

    `schema_score` and `properties_score` are weighted-average scores in
    [0, 1] (1 = fully aligned with the graph schema / property usage);
    `properties_score` is None if the query accesses no properties.
    """

    syntax_valid: bool
    syntax_metadata: List[Dict[str, Any]]
    schema_score: float
    schema_metadata: List[Dict[str, Any]]
    properties_score: Optional[float]
    properties_metadata: List[Dict[str, Any]]


def _driver_and_database(graph: Any):
    driver = getattr(graph, "_driver", None)
    if driver is None:
        raise AttributeError(
            "Could not access the underlying neo4j.Driver from the given "
            "Neo4jGraph (no `_driver` attribute found); CyVer validation "
            "requires direct driver access."
        )
    return driver, getattr(graph, "_database", None)


def _safe_validate(validator, cypher: str, database_name: Optional[str], **kwargs):
    try:
        return validator.validate(cypher, database_name=database_name, **kwargs)
    except Exception as e:
        return None, [{"code": "CyVerValidatorError", "description": str(e)}]


def validate_cypher(graph: Any, cypher: str) -> CypherValidationReport:
    """Run CyVer's syntax/schema/property validators over a failing Cypher query."""
    driver, database = _driver_and_database(graph)

    syntax_valid, syntax_metadata = _safe_validate(SyntaxValidator(driver), cypher, database)
    schema_score, schema_metadata = _safe_validate(SchemaValidator(driver), cypher, database)
    properties_score, properties_metadata = _safe_validate(PropertiesValidator(driver), cypher, database)

    return CypherValidationReport(
        syntax_valid=bool(syntax_valid),
        syntax_metadata=syntax_metadata,
        schema_score=schema_score if schema_score is not None else 0.0,
        schema_metadata=schema_metadata,
        properties_score=properties_score,
        properties_metadata=properties_metadata,
    )
