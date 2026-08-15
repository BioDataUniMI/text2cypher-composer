from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda

from text2cypher_composer import run
from text2cypher_composer.cypher_utils import CypherExecutionError
from text2cypher_composer.validation import CypherValidationReport


def _validation(syntax_valid=True):
    return CypherValidationReport(
        syntax_valid=syntax_valid,
        syntax_metadata=[] if syntax_valid else [{"code": "SYN001", "description": "bad syntax"}],
        schema_score=1.0,
        schema_metadata=[],
        properties_score=1.0,
        properties_metadata=[],
    )


def test_rescue_error_messages_concatenates_native_error_and_notifications_then_succeeds():
    call_count = {"n": 0}

    def flaky_then_fixed(_):
        call_count["n"] += 1
        return "MATCH (n:Broken) RETURN n" if call_count["n"] == 1 else "MATCH (n:Fixed) RETURN n"

    model = RunnableLambda(flaky_then_fixed)

    def fake_execute(_graph, cypher):
        if "Broken" in cypher:
            raise CypherExecutionError("Cypher semantic error", "Variable `x` not defined", warnings=["Warning: deprecated - Position: line 1, column 1"])
        return [{"n": 1}], []

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
            max_retries=2,
        )

    assert result.rescued is True
    assert result.rescue_attempts == 1
    assert len(result.rescue_error_messages) == 1
    msg = result.rescue_error_messages[0]
    assert "Cypher semantic error: Variable `x` not defined" in msg
    assert "Warning: deprecated - Position: line 1, column 1" in msg
    assert result.cypher == "MATCH (n:Fixed) RETURN n"
    assert result.initial_cypher == "MATCH (n:Broken) RETURN n"

    # execution_error/execution_warnings reflect the FINAL (successful, rescued) attempt
    assert result.execution_error is None
    assert result.execution_warnings == []

    # one fully-instantiated rescue prompt, containing the bad query and the error message
    assert len(result.rescue_prompts) == 1
    rescue_prompt_text = " ".join(m["content"] for m in result.rescue_prompts[0])
    assert "MATCH (n:Broken) RETURN n" in rescue_prompt_text
    assert "Cypher semantic error: Variable `x` not defined" in rescue_prompt_text

    # a token count is available (via tiktoken) for the initial prompt and each rescue attempt
    assert result.prompt_tokens is None or result.prompt_tokens > 0
    assert len(result.rescue_prompt_tokens) == 1
    assert result.rescue_prompt_tokens[0] is None or result.rescue_prompt_tokens[0] > 0


def test_rescue_error_messages_reports_empty_result_set():
    call_count = {"n": 0}

    def flaky_then_fixed(_):
        call_count["n"] += 1
        return "MATCH (n:Empty) RETURN n" if call_count["n"] == 1 else "MATCH (n:Fixed) RETURN n"

    model = RunnableLambda(flaky_then_fixed)

    def fake_execute(_graph, cypher):
        if "Empty" in cypher:
            return [], []
        return [{"n": 1}], []

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
        )

    assert result.rescued is True
    assert "Empty result set." in result.rescue_error_messages[0]


def test_no_rescue_needed_leaves_rescue_error_messages_empty():
    model = RunnableLambda(lambda _: "MATCH (n:Fixed) RETURN n")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
        )

    assert result.rescued is False
    assert result.rescue_attempts == 0
    assert result.rescue_error_messages == []
    assert result.rescue_prompts == []
    assert result.rescue_prompt_tokens == []


def test_execution_error_and_warnings_populated_even_without_rescue_prompt():
    """A query that executes successfully but with Neo4j notifications (and a
    non-empty result, so no rescue is needed) should still surface those
    warnings on the result — independently of `rescue_prompt`."""
    model = RunnableLambda(lambda _: "MATCH (n:Deprecated) RETURN n")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch(
             "text2cypher_composer.core.execute_cypher_with_warnings",
             return_value=([{"n": 1}], ["Warning: deprecated syntax - Position: line 1, column 1"]),
         ), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=False,
        )

    assert result.executed is True
    assert result.execution_error is None
    assert result.execution_warnings == ["Warning: deprecated syntax - Position: line 1, column 1"]


def test_execution_error_reflects_final_failed_attempt_when_rescue_exhausted():
    model = RunnableLambda(lambda _: "MATCH (n:StillBroken) RETURN n")

    def fake_execute(_graph, _cypher):
        raise CypherExecutionError("Cypher syntax error", "Invalid input 'x'", warnings=[])

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation(syntax_valid=False)):
        result = run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
            max_retries=2,
        )

    assert result.executed is False
    assert result.rescue_attempts == 2
    assert len(result.rescue_error_messages) == 2
    assert len(result.rescue_prompts) == 2
    assert len(result.rescue_prompt_tokens) == 2  # a list of k=2 token counts, one per attempt
    assert result.execution_error == "Cypher syntax error: Invalid input 'x'"


def test_second_rescue_attempt_sees_the_first_attempts_proposed_query_and_new_error():
    """If the first rescue's proposed fix still fails, but with a *different*
    error than the original one, the second rescue prompt must be built from
    that new query/error — not from the original attempt's."""
    proposals = iter([
        "MATCH (n:Broken) RETURN n",       # initial (bad label)
        "MATCH (n:Fixed) RETURN n.oops",   # 1st rescue proposal (bad property this time)
        "MATCH (n:Fixed) RETURN n.ok",     # 2nd rescue proposal (finally works)
    ])
    model = RunnableLambda(lambda _: next(proposals))

    def fake_execute(_graph, cypher):
        if "Broken" in cypher:
            raise CypherExecutionError("Cypher semantic error", "Unknown label 'Broken'")
        if "oops" in cypher:
            raise CypherExecutionError("Cypher semantic error", "Unknown property 'oops'")
        return [{"n": 1}], []

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation(syntax_valid=False)):
        result = run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
            max_retries=2,
        )

    assert result.executed is True
    assert result.cypher == "MATCH (n:Fixed) RETURN n.ok"
    assert result.rescue_attempts == 2

    # 1st rescue prompt: built from the *initial* query and its "Unknown label" error
    first_error, second_error = result.rescue_error_messages
    assert "Unknown label 'Broken'" in first_error
    assert "Unknown property" not in first_error
    first_prompt_text = " ".join(m["content"] for m in result.rescue_prompts[0])
    assert "MATCH (n:Broken) RETURN n" in first_prompt_text

    # 2nd rescue prompt: built from the 1st rescue's *proposed* query and its NEW "Unknown property" error
    assert "Unknown property 'oops'" in second_error
    assert "Unknown label" not in second_error
    second_prompt_text = " ".join(m["content"] for m in result.rescue_prompts[1])
    assert "MATCH (n:Fixed) RETURN n.oops" in second_prompt_text
    assert "Broken" not in second_prompt_text


def test_max_retries_zero_raises_value_error():
    import pytest

    model = RunnableLambda(lambda _: "MATCH (n) RETURN n")
    with pytest.raises(ValueError, match="max_retries"):
        run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
            max_retries=0,
        )
