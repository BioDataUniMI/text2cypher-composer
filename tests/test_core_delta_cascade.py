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


def test_delta_cascade_second_rung_is_also_a_fresh_self_contained_prompt():
    """No rescue-style coupling: the schema-expansion effect must stay isolated from any
    correction/rescue dynamic -- rung 2 gets no reference to rung 1's query or failure reason."""
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
    # no reference at all to the previous rung's query or why it failed
    assert "MATCH (n:Narrow) RETURN n" not in second_prompt_text
    assert "Unknown label 'Narrow'" not in second_prompt_text
    assert "error message" not in second_prompt_text.lower()
    # only the NEW schema at this rung is shown -- not the narrow rung's, already seen
    assert "NODES_ONLY_DELTA_SCHEMA" in second_prompt_text
    assert "NARROW_DELTA_SCHEMA" not in second_prompt_text


def test_delta_cascade_falls_through_to_full_with_no_rescue_coupling_either():
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
    assert "MATCH (n:NodesOnly) RETURN n" not in third_prompt_text  # no reference to rung 2's attempt
    assert "still broken" not in third_prompt_text
    assert "FULL_DELTA_SCHEMA" in third_prompt_text


def test_delta_cascade_composes_with_self_verification_without_leaking_into_the_next_prompt():
    def fake_verify(_llm, _question, cypher, _result, schema=None, examples=None, criteria=None):
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

    # self_verification still gates the per-rung retry decision (a failed verdict on rung 1
    # falls through to rung 2, same as a mechanical failure would) ...
    assert result.cascade_mode_level == "nodes_only"
    assert result.self_verification_passed is True
    # ... but its reasoning is never fed into the next rung's prompt -- there is no "next rung's
    # prompt" concept of a fix-up here, each rung stays independent of the one before it.
    second_prompt_text = " ".join(m["content"] for m in result.cascade_mode_prompts[1])
    assert "wrong node label for the question" not in second_prompt_text
    assert "Semantic review" not in second_prompt_text


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
