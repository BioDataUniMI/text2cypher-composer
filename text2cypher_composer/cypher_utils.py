import re
from typing import Any, List, Optional, Tuple

from langchain_community.graphs.neo4j_graph import value_sanitize
from neo4j import Query
from neo4j.exceptions import ClientError, CypherSyntaxError, Neo4jError, SessionExpired


class CypherExecutionError(Exception):
    """Raised when a generated Cypher query fails to execute against the graph.

    `warnings` carries any Neo4j notifications observed before the failure
    (rare, since most errors abort the query before any are emitted, but
    kept for parity with the success path).
    """

    def __init__(self, code: str, message: str, warnings: Optional[List[str]] = None):
        self.code = code
        self.message = message
        self.warnings = warnings or []
        super().__init__(message)


def normalize_generated_cypher(cypher: Any) -> str:
    """Strip code fences/whitespace artifacts an LLM may wrap the query in."""
    if cypher is None:
        return ""
    if not isinstance(cypher, str):
        cypher = str(cypher)

    cypher = cypher.strip()
    cypher = re.sub(r"^```(?:cypher)?\s*", "", cypher, flags=re.IGNORECASE)
    cypher = re.sub(r"\s*```$", "", cypher)
    cypher = cypher.replace("\r\n", "\n").replace("\r", "\n").strip()
    cypher = cypher.replace("\x08", r"\b")
    cypher = re.sub(r"(?<!\\)\\b", r"\\\\b", cypher)
    return cypher


def _format_notifications(notifications: Any) -> List[str]:
    """Format Neo4j notifications (deprecations, unknown labels/rel-types/
    properties, cartesian products, ...) the way the miRNAKG rescue-prompt
    notebook's `execute_query_with_warnings` did."""
    warnings = []
    for notification in notifications or []:
        description = notification.get("description", "")
        position = notification.get("position", "")
        warnings.append(f"Warning: {description} - Position: {position}")
    return warnings


def execute_cypher_with_warnings(
    graph: Any, cypher: str, _retry_on_session_expired: bool = True
) -> Tuple[List[dict], List[str]]:
    """Run `cypher` directly against the underlying driver, returning `(records, warnings)`.

    Ported from the miRNAKG rescue-prompt notebook's `execute_query_with_warnings`:
    runs through the raw `neo4j.Driver` (like `Neo4jGraph.query` does internally)
    rather than `graph.query()`, so Neo4j's own notifications are captured
    alongside the records, not just hard errors. Raises `CypherExecutionError`
    on failure — mirrors `Neo4jGraph.query`'s handling of syntax errors and
    session-expiry retries, plus `sanitize`/`timeout` if set on `graph`.
    """
    driver = getattr(graph, "_driver", None)
    if driver is None:
        raise AttributeError(
            "Could not access the underlying neo4j.Driver from the given "
            "Neo4jGraph (no `_driver` attribute found)."
        )
    database = getattr(graph, "_database", None)
    timeout = getattr(graph, "timeout", None)
    sanitize = getattr(graph, "sanitize", False)

    try:
        with driver.session(database=database) as session:
            result = session.run(Query(text=cypher, timeout=timeout))
            records = [r.data() for r in result]
            if sanitize:
                records = [value_sanitize(r) for r in records]
            summary = result.consume()
            warnings = _format_notifications(getattr(summary, "notifications", None))
        return records, warnings
    except CypherSyntaxError as e:
        raise CypherExecutionError("Cypher syntax error", str(e)) from e
    except SessionExpired as e:
        if _retry_on_session_expired:
            return execute_cypher_with_warnings(graph, cypher, _retry_on_session_expired=False)
        raise CypherExecutionError("Neo4j session expired", str(e)) from e
    except ClientError as e:
        if e.code == "Neo.ClientError.Statement.SemanticError":
            raise CypherExecutionError("Cypher semantic error", str(e)) from e
        raise CypherExecutionError(f"Cypher client error ({e.code})", str(e)) from e
    except Neo4jError as e:
        raise CypherExecutionError("Neo4j runtime error", str(e)) from e
    except Exception as e:
        raise CypherExecutionError("Other error", str(e)) from e


def execute_cypher(graph: Any, cypher: str) -> List[dict]:
    """Run `cypher` against `graph`, raising CypherExecutionError on failure."""
    records, _warnings = execute_cypher_with_warnings(graph, cypher)
    return records
