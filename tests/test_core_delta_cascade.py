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


FAKE_DELTA_LEVELS = [
    (CascadeModeLevel.NARROW, "NARROW_DELTA_SCHEMA"),
    (CascadeModeLevel.NODES_ONLY, "NODES_ONLY_DELTA_SCHEMA"),
    (CascadeModeLevel.FULL, "FULL_DELTA_SCHEMA"),
]


def _fake_model(prompt_value):
    text = prompt_value.to_string()
    if "NARROW_DELTA_SCHEMA" in text:
        return "MATCH (n:Narrow) RETURN n"
    if "NODES_ONLY_DELTA_SCHEMA" in text:
        return "MATCH (n:NodesOnly) RETURN n"
    return "MATCH (n:Full) RETURN n"


def test_delta_cascade_first_rung_uses_a_fresh_self_contained_prompt():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_DELTA_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            cascade_strategy="delta",
        )

    assert result.cascade_mode_level == "narrow"
    assert result.cascade_mode_attempts == 1
    first_prompt_text = " ".join(m["content"] for m in result.cascade_mode_prompts[0])
    assert "NARROW_DELTA_SCHEMA" in first_prompt_text
    # a fresh, self-contained prompt -- no reference to a previous attempt
    assert "error message" not in first_prompt_text.lower()


def test_delta_cascade_second_rung_is_a_rescue_style_continuation():
    def fake_execute(_graph, cypher):
        if "Narrow" in cypher:
            raise CypherExecutionError("Cypher semantic error", "Unknown label 'Narrow'")
        return [{"n": 1}], []

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_DELTA_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            cascade_strategy="delta",
        )

    assert result.executed is True
    assert result.cypher == "MATCH (n:NodesOnly) RETURN n"
    assert result.cascade_mode_level == "nodes_only"
    assert result.cascade_mode_attempts == 2

    second_prompt_text = " ".join(m["content"] for m in result.cascade_mode_prompts[1])
    # rescue-style: references the previous rung's failed query and why it failed
    assert "MATCH (n:Narrow) RETURN n" in second_prompt_text
    assert "Unknown label 'Narrow'" in second_prompt_text
    # only the NEW schema at this rung is shown -- not the narrow rung's, already seen
    assert "NODES_ONLY_DELTA_SCHEMA" in second_prompt_text
    assert "NARROW_DELTA_SCHEMA" not in second_prompt_text


def test_delta_cascade_falls_through_to_full_when_second_rung_also_fails():
    def fake_execute(_graph, cypher):
        if "Full" in cypher:
            return [{"n": 1}], []
        raise CypherExecutionError("Cypher semantic error", f"still broken: {cypher}")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_DELTA_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            cascade_strategy="delta",
        )

    assert result.executed is True
    assert result.cypher == "MATCH (n:Full) RETURN n"
    assert result.cascade_mode_level == "full"
    assert result.cascade_mode_attempts == 3

    third_prompt_text = " ".join(m["content"] for m in result.cascade_mode_prompts[2])
    assert "MATCH (n:NodesOnly) RETURN n" in third_prompt_text  # references rung 2's attempt, not rung 1's
    assert "FULL_DELTA_SCHEMA" in third_prompt_text


def test_delta_cascade_composes_with_self_verification():
    def fake_verify(_llm, _question, cypher, _result, criteria=None):
        if "Narrow" in cypher:
            return SemanticVerification(answers_question=False, reasoning="wrong node label for the question")
        return SemanticVerification(answers_question=True, reasoning="looks right")

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_DELTA_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()), \
         patch("text2cypher_composer.core.verify_semantics", side_effect=fake_verify):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            cascade_strategy="delta",
            self_verification=True,
        )

    assert result.cascade_mode_level == "nodes_only"
    assert result.self_verification_passed is True
    second_prompt_text = " ".join(m["content"] for m in result.cascade_mode_prompts[1])
    assert "Semantic review: wrong node label for the question" in second_prompt_text


def test_cascade_strategy_delta_requires_cascade_mode():
    with pytest.raises(ValueError, match="cascade_strategy"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_strategy="delta",
        )
