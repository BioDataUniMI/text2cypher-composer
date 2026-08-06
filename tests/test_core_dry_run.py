from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from text2cypher_composer import run


def _boom(_):
    raise AssertionError("the model should never be invoked during a dry_run")


FAKE_MODEL = RunnableLambda(_boom)


def test_dry_run_builds_prompt_without_generating_or_executing():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()) as resolve_db, \
         patch("text2cypher_composer.core.execute_cypher") as exec_cypher, \
         patch("text2cypher_composer.core.validate_cypher") as validate:
        result = run(
            input_NL="How many genes are there?",
            model=FAKE_MODEL,
            database={},
            technique="vanilla",
            dry_run=True,
        )

    resolve_db.assert_called_once()
    exec_cypher.assert_not_called()
    validate.assert_not_called()

    assert result.dry_run is True
    assert result.cypher is None
    assert result.initial_cypher is None
    assert result.executed is False
    assert result.validation is None
    assert result.result is None
    assert "How many genes are there?" in result.prompt[-1]["content"]


def test_dry_run_still_resolves_schema_for_schema_techniques():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_schema_text", return_value="Node labels: Gene") as resolve_schema, \
         patch("text2cypher_composer.core.execute_cypher") as exec_cypher, \
         patch("text2cypher_composer.core.validate_cypher") as validate:
        result = run(
            input_NL="How many genes?",
            model=FAKE_MODEL,
            database={},
            technique="Schema",
            dry_run=True,
        )

    resolve_schema.assert_called_once()
    exec_cypher.assert_not_called()
    validate.assert_not_called()
    assert result.schema == "Node labels: Gene"
    assert "Node labels: Gene" in result.prompt[-1]["content"]


def test_dry_run_incompatible_with_rescue_prompt():
    with pytest.raises(ValueError, match="dry_run"):
        run(
            input_NL="q",
            model=FAKE_MODEL,
            database={},
            technique="vanilla",
            dry_run=True,
            rescue_prompt=True,
        )
