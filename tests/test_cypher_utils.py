from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import ClientError

from text2cypher_composer.cypher_utils import (
    CypherExecutionError,
    execute_cypher,
    execute_cypher_with_warnings,
)


def _mock_graph(session):
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    graph = MagicMock()
    graph._driver = driver
    graph._database = "neo4j"
    graph.timeout = None
    graph.sanitize = False
    return graph


def _mock_result(records, notifications=None):
    result = MagicMock()
    result.__iter__.return_value = iter(records)
    summary = MagicMock()
    summary.notifications = notifications or []
    result.consume.return_value = summary
    return result


def test_execute_cypher_with_warnings_returns_records_and_notifications():
    record = MagicMock()
    record.data.return_value = {"c": 1}
    notifications = [{"description": "unknown label", "position": "line 1, column 8"}]
    session = MagicMock()
    session.run.return_value = _mock_result([record], notifications)
    graph = _mock_graph(session)

    records, warnings = execute_cypher_with_warnings(graph, "MATCH (n) RETURN n")

    assert records == [{"c": 1}]
    assert warnings == ["Warning: unknown label - Position: line 1, column 8"]


def test_execute_cypher_with_warnings_no_notifications_is_empty_list():
    session = MagicMock()
    session.run.return_value = _mock_result([])
    graph = _mock_graph(session)

    records, warnings = execute_cypher_with_warnings(graph, "MATCH (n) RETURN n")

    assert records == []
    assert warnings == []


def test_execute_cypher_with_warnings_raises_on_syntax_error():
    session = MagicMock()
    session.run.side_effect = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Statement.SyntaxError", message="Invalid input 'x'"
    )
    graph = _mock_graph(session)

    with pytest.raises(CypherExecutionError) as exc_info:
        execute_cypher_with_warnings(graph, "MATCXH (n) RETURN n")

    assert exc_info.value.code == "Cypher syntax error"
    assert "Invalid input 'x'" in exc_info.value.message
    assert exc_info.value.warnings == []


def test_execute_cypher_with_warnings_raises_on_semantic_error():
    session = MagicMock()
    session.run.side_effect = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Statement.SemanticError", message="Variable not defined"
    )
    graph = _mock_graph(session)

    with pytest.raises(CypherExecutionError) as exc_info:
        execute_cypher_with_warnings(graph, "MATCH (n) RETURN m")

    assert exc_info.value.code == "Cypher semantic error"


def test_execute_cypher_backward_compatible_returns_only_records():
    record = MagicMock()
    record.data.return_value = {"c": 1}
    session = MagicMock()
    session.run.return_value = _mock_result([record])
    graph = _mock_graph(session)

    assert execute_cypher(graph, "MATCH (n) RETURN n") == [{"c": 1}]


def test_execute_cypher_raises_cypher_execution_error_on_failure():
    session = MagicMock()
    session.run.side_effect = RuntimeError("boom")
    graph = _mock_graph(session)

    with pytest.raises(CypherExecutionError):
        execute_cypher(graph, "MATCH (n) RETURN n")
