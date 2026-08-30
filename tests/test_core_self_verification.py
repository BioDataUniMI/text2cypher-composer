from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from text2cypher_composer import run
from text2cypher_composer.cypher_utils import CypherExecutionError
from text2cypher_composer.techniques import CascadeModeLevel
from text2cypher_composer.validation import CypherValidationReport
from text2cypher_composer.verification import SemanticVerification


def _validation(syntax_valid=True):
    return CypherValidationReport(
        syntax_valid=syntax_valid,
        syntax_metadata=[] if syntax_valid else [{"code": "SYN001", "description": "bad syntax"}],
        schema_score=1.0,
        schema_metadata=[],
        properties_score=1.0,
        properties_metadata=[],
    )


def test_self_verification_triggers_rescue_on_failed_semantic_verdict():
    call_count = {"n": 0}

    def flaky_then_fixed(_):
        call_count["n"] += 1
        return "MATCH (n:WrongAnswer) RETURN n" if call_count["n"] == 1 else "MATCH (n:RightAnswer) RETURN n"

    def fake_verify(_llm, _question, cypher, _result, criteria=None):
        if "WrongAnswer" in cypher:
            return SemanticVerification(answers_question=False, reasoning="wrong node label for the question")
        return SemanticVerification(answers_question=True, reasoning="looks right")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()), \
         patch("text2cypher_composer.core.verify_semantics", side_effect=fake_verify):
        result = run(
            input_NL="q",
            model=RunnableLambda(flaky_then_fixed),
            database={},
            technique="vanilla",
            rescue_prompt=True,
            self_verification=True,
        )

    assert result.rescued is True
    assert result.rescue_attempts == 1
    assert result.cypher == "MATCH (n:RightAnswer) RETURN n"
    assert result.self_verification_passed is True
    assert result.self_verification_reasoning == "looks right"
    assert "Semantic review: wrong node label for the question" in result.rescue_error_messages[0]


def test_self_verification_not_invoked_when_mechanically_broken():
    """No point spending a verification call on a query that doesn't even execute."""
    def fake_execute(_graph, _cypher):
        raise CypherExecutionError("Cypher semantic error", "Unknown label 'Broken'")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation(syntax_valid=False)), \
         patch("text2cypher_composer.core.verify_semantics") as verify_mock:
        result = run(
            input_NL="q",
            model=RunnableLambda(lambda _: "MATCH (n:Broken) RETURN n"),
            database={},
            technique="vanilla",
            rescue_prompt=True,
            max_retries=1,
            self_verification=True,
        )

    verify_mock.assert_not_called()
    assert result.self_verification_passed is None
    assert result.self_verification_reasoning is None


_FAKE_LEVELS = [
    (CascadeModeLevel.NARROW, "NARROW_SCHEMA"),
    (CascadeModeLevel.NODES_ONLY, "NODES_ONLY_SCHEMA"),
    (CascadeModeLevel.FULL, "FULL_SCHEMA"),
]


def _fake_cascade_model(prompt_value):
    text = prompt_value.to_string()
    if "NARROW_SCHEMA" in text:
        return "MATCH (n:Narrow) RETURN n"
    if "NODES_ONLY_SCHEMA" in text:
        return "MATCH (n:NodesOnly) RETURN n"
    return "MATCH (n:Full) RETURN n"


def test_self_verification_falls_through_cascade_rungs_on_failed_semantic_verdict():
    def fake_verify(_llm, _question, cypher, _result, criteria=None):
        if "Narrow" in cypher:
            return SemanticVerification(answers_question=False, reasoning="over-pruned schema, wrong answer")
        return SemanticVerification(answers_question=True, reasoning="looks right")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=_FAKE_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()), \
         patch("text2cypher_composer.core.verify_semantics", side_effect=fake_verify):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_cascade_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            self_verification=True,
        )

    assert result.cypher == "MATCH (n:NodesOnly) RETURN n"
    assert result.cascade_mode_level == "nodes_only"
    assert result.cascade_mode_attempts == 2
    assert result.self_verification_passed is True
    assert result.self_verification_reasoning == "looks right"


def test_self_verification_requires_rescue_prompt_or_cascade_mode():
    with pytest.raises(ValueError, match="self_verification"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="vanilla",
            self_verification=True,
        )


def test_verification_model_rejected_without_self_verification():
    with pytest.raises(ValueError, match="self_verification"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="vanilla",
            rescue_prompt=True,
            verification_model="gpt-4o",
        )


def test_verification_criteria_rejected_without_self_verification():
    with pytest.raises(ValueError, match="self_verification"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="vanilla",
            rescue_prompt=True,
            verification_criteria="must include units",
        )


def test_verification_model_defaults_to_the_main_model():
    model = RunnableLambda(lambda _: "MATCH (n) RETURN n")
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()), \
         patch(
             "text2cypher_composer.core.verify_semantics",
             return_value=SemanticVerification(answers_question=True, reasoning="ok"),
         ), \
         patch("text2cypher_composer.core.resolve_pruning_model") as resolve_pruning:
        run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
            self_verification=True,
        )

    resolve_pruning.assert_called_once_with(model)


def test_verification_model_overrides_the_main_model_when_given():
    model = RunnableLambda(lambda _: "MATCH (n) RETURN n")
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()), \
         patch(
             "text2cypher_composer.core.verify_semantics",
             return_value=SemanticVerification(answers_question=True, reasoning="ok"),
         ), \
         patch("text2cypher_composer.core.resolve_pruning_model") as resolve_pruning:
        run(
            input_NL="q",
            model=model,
            database={},
            technique="vanilla",
            rescue_prompt=True,
            self_verification=True,
            verification_model="gpt-4o",
        )

    resolve_pruning.assert_called_once_with("gpt-4o")
