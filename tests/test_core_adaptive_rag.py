from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from text2cypher_composer import run
from text2cypher_composer.cypher_utils import CypherExecutionError
from text2cypher_composer.techniques import RAGExpansionLevel
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
    (RAGExpansionLevel.MINIMAL, {"examples_text": "MINIMAL_EXAMPLES", "example_ids": [], "example_distances": []}),
    (RAGExpansionLevel.MODERATE, {"examples_text": "MODERATE_EXAMPLES", "example_ids": [], "example_distances": []}),
    (RAGExpansionLevel.FULL, {"examples_text": "FULL_EXAMPLES", "example_ids": [], "example_distances": []}),
]


def _fake_model(prompt_value):
    text = prompt_value.to_string()
    if "MINIMAL_EXAMPLES" in text:
        return "MATCH (n:Minimal) RETURN n"
    if "MODERATE_EXAMPLES" in text:
        return "MATCH (n:Moderate) RETURN n"
    return "MATCH (n:Full) RETURN n"


def test_adaptive_rag_cascades_through_levels_until_success():
    def fake_execute(_graph, cypher):
        if "Minimal" in cypher:
            raise CypherExecutionError("Cypher semantic error", "Unknown label 'Minimal'")
        if "Moderate" in cypher:
            return [], []  # executes but empty -> still needs_rescue
        return [{"n": 1}], []

    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_adaptive_rag_levels", return_value=FAKE_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", side_effect=fake_execute), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="RAG",
            dataset="fake_dataset_root",
            adaptive_rag=True,
        )

    assert result.executed is True
    assert result.cypher == "MATCH (n:Full) RETURN n"
    assert result.retrieved_examples["examples_text"] == "FULL_EXAMPLES"
    assert result.adaptive_rag_level == "full"
    assert result.adaptive_rag_attempts == 3
    assert len(result.adaptive_rag_prompts) == 3
    assert len(result.adaptive_rag_prompt_tokens) == 3
    # each rung's prompt actually carries that rung's retrieved examples
    assert "MINIMAL_EXAMPLES" in result.adaptive_rag_prompts[0][-1]["content"]
    assert "MODERATE_EXAMPLES" in result.adaptive_rag_prompts[1][-1]["content"]
    assert "FULL_EXAMPLES" in result.adaptive_rag_prompts[2][-1]["content"]


def test_adaptive_rag_stops_early_once_a_rung_succeeds():
    with patch("text2cypher_composer.core.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.core.resolve_adaptive_rag_levels", return_value=FAKE_LEVELS), \
         patch("text2cypher_composer.core.execute_cypher_with_warnings", return_value=([{"n": 1}], [])), \
         patch("text2cypher_composer.core.validate_cypher", return_value=_validation()):
        result = run(
            input_NL="q",
            model=RunnableLambda(_fake_model),
            database={},
            technique="RAG",
            dataset="fake_dataset_root",
            adaptive_rag=True,
        )

    assert result.cypher == "MATCH (n:Minimal) RETURN n"
    assert result.adaptive_rag_level == "minimal"
    assert result.adaptive_rag_attempts == 1
    assert len(result.adaptive_rag_prompts) == 1


def test_adaptive_rag_and_rescue_prompt_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="RAG",
            dataset="fake_dataset_root",
            adaptive_rag=True,
            rescue_prompt=True,
        )


def test_adaptive_rag_and_non_default_max_retries_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="RAG",
            dataset="fake_dataset_root",
            adaptive_rag=True,
            max_retries=2,
        )


def test_adaptive_rag_and_cascade_mode_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="Schema+RAG",
            dataset="fake_dataset_root",
            schema_mode="exact_match",
            adaptive_rag=True,
            cascade_mode=True,
        )


def test_adaptive_rag_requires_a_rag_technique():
    with pytest.raises(ValueError, match="adaptive_rag"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="vanilla",
            adaptive_rag=True,
        )


def test_adaptive_rag_incompatible_with_dry_run():
    with pytest.raises(ValueError, match="dry_run"):
        run(
            input_NL="q",
            model=RunnableLambda(lambda _: "x"),
            database={},
            technique="RAG",
            dataset="fake_dataset_root",
            adaptive_rag=True,
            dry_run=True,
        )
