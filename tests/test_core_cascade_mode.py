from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from text2cypher_composer import run
from text2cypher_composer.cypher_utils import CypherExecutionError
from text2cypher_composer.techniques import CascadeModeLevel
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


FAKE_LEVELS = [
    (CascadeModeLevel.NARROW, "NARROW_SCHEMA"),
    (CascadeModeLevel.NODES_ONLY, "NODES_ONLY_SCHEMA"),
    (CascadeModeLevel.FULL, "FULL_SCHEMA"),
]


def _fake_model(prompt_value):
    text = prompt_value.to_string()
    if "NARROW_SCHEMA" in text:
        return "MATCH (n:Narrow) RETURN n"
    if "NODES_ONLY_SCHEMA" in text:
        return "MATCH (n:NodesOnly) RETURN n"
    return "MATCH (n:Full) RETURN n"


def test_cascade_mode_cascades_through_levels_until_success():
    def fake_execute(_graph, cypher):
        if "Narrow" in cypher:
            raise CypherExecutionError("Cypher semantic error", "Unknown label 'Narrow'")
        if "NodesOnly" in cypher:
            return [], []  # executes but empty -> still needs_rescue
        return [{"n": 1}], []

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
        )

    assert result.executed is True
    assert result.cypher == "MATCH (n:Full) RETURN n"
    assert result.schema == "FULL_SCHEMA"
    assert result.cascade_mode_level == "full"
    assert result.cascade_mode_attempts == 3
    assert len(result.cascade_mode_prompts) == 3
    assert len(result.cascade_mode_prompt_tokens) == 3
    # each rung's prompt actually carries that rung's schema text
    assert "NARROW_SCHEMA" in result.cascade_mode_prompts[0][-1]["content"]
    assert "NODES_ONLY_SCHEMA" in result.cascade_mode_prompts[1][-1]["content"]
    assert "FULL_SCHEMA" in result.cascade_mode_prompts[2][-1]["content"]


def test_cascade_mode_stops_early_once_a_rung_succeeds():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
        )

    assert result.cypher == "MATCH (n:Narrow) RETURN n"
    assert result.cascade_mode_level == "narrow"
    assert result.cascade_mode_attempts == 1
    assert len(result.cascade_mode_prompts) == 1


def test_skip_narrow_schema_filter_is_forwarded_to_the_resolver():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch(
             "text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_LEVELS[1:]
         ) as resolve_levels, \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            skip_narrow_schema_filter=True,
        )

    assert resolve_levels.call_args.kwargs["skip_narrow"] is True
    assert result.cascade_mode_level == "nodes_only"


def test_cache_schema_is_forwarded_to_the_resolver():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch(
             "text2cypher_composer.core.resolve_cascade_mode_levels", return_value=FAKE_LEVELS
         ) as resolve_levels, \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            cache_schema=False,
        )

    assert resolve_levels.call_args.kwargs["cache_schema"] is False


def test_cascade_mode_and_rescue_prompt_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            rescue_prompt=True,
        )


def test_cascade_mode_and_non_default_max_retries_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            max_retries=2,
        )


def test_cascade_mode_requires_a_schema_technique():
    with pytest.raises(ValueError, match="cascade_mode"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="vanilla",
            cascade_mode=True,
        )


def test_skip_narrow_schema_filter_requires_cascade_mode():
    with pytest.raises(ValueError, match="skip_narrow_schema_filter"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            skip_narrow_schema_filter=True,
        )


def test_cascade_mode_incompatible_with_dry_run():
    with pytest.raises(ValueError, match="dry_run"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            dry_run=True,
        )


@pytest.mark.parametrize("schema_mode", [None, "schema", "enhanced"])
def test_cascade_mode_requires_a_pruning_schema_mode(schema_mode):
    with pytest.raises(ValueError, match="pruning"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema",
            schema_mode=schema_mode,
            cascade_mode=True,
        )
