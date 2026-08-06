import re
from typing import Any, List

from neo4j.exceptions import ClientError, Neo4jError


class CypherExecutionError(Exception):
    """Raised when a generated Cypher query fails to execute against the graph."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
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


def execute_cypher(graph: Any, cypher: str) -> List[dict]:
    """Run `cypher` against `graph`, raising CypherExecutionError on failure."""
    try:
        return graph.query(cypher)
    except ClientError as e:
        if e.code == "Neo.ClientError.Statement.SyntaxError":
            raise CypherExecutionError("Cypher syntax error", str(e)) from e
        if e.code == "Neo.ClientError.Statement.SemanticError":
            raise CypherExecutionError("Cypher semantic error", str(e)) from e
        raise CypherExecutionError(f"Cypher client error ({e.code})", str(e)) from e
    except Neo4jError as e:
        raise CypherExecutionError("Neo4j runtime error", str(e)) from e
    except Exception as e:
        raise CypherExecutionError("Other error", str(e)) from e
