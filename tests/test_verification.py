from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from text2cypher_composer.verification import SemanticVerification, verify_semantics


def _fake_llm(respond):
    """A fake structured-output-capable llm: `respond` is either a fixed SemanticVerification,
    or a callable(prompt_value) -> SemanticVerification to inspect the formatted prompt."""
    llm = MagicMock()
    chain = respond if callable(respond) else (lambda _: respond)
    llm.with_structured_output.return_value = RunnableLambda(chain)
    return llm


def test_verify_semantics_returns_the_structured_verdict():
    response = SemanticVerification(answers_question=True, reasoning="looks right")
    llm = _fake_llm(response)

    result = verify_semantics(llm, "How many genes?", "MATCH (g:Gene) RETURN count(g)", [{"count(g)": 3}])

    assert result == response
    llm.with_structured_output.assert_called_once_with(SemanticVerification, method="json_schema")


def test_verify_semantics_prompt_includes_question_cypher_and_result():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=False, reasoning="wrong direction")

    llm = _fake_llm(_capture)
    verify_semantics(llm, "How many genes?", "MATCH (g:Gene) RETURN count(g)", [{"count(g)": 3}])

    text = captured["text"]
    assert "How many genes?" in text
    assert "MATCH (g:Gene) RETURN count(g)" in text
    assert "count(g)" in text  # the result row is rendered into the prompt


def test_verify_semantics_reports_no_rows_when_result_is_empty_or_none():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=False, reasoning="empty")

    llm = _fake_llm(_capture)
    verify_semantics(llm, "q", "MATCH (n) RETURN n", None)

    assert "(no rows)" in captured["text"]


def test_verify_semantics_appends_criteria_when_given():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=True, reasoning="ok")

    llm = _fake_llm(_capture)
    verify_semantics(llm, "q", "MATCH (n) RETURN n", [{"n": 1}], criteria="must include units")

    assert "must include units" in captured["text"]


def test_verify_semantics_omits_criteria_block_when_not_given():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=True, reasoning="ok")

    llm = _fake_llm(_capture)
    verify_semantics(llm, "q", "MATCH (n) RETURN n", [{"n": 1}])

    assert "Additional evaluation criteria" not in captured["text"]


def test_verify_semantics_includes_the_schema_when_given():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=True, reasoning="ok")

    llm = _fake_llm(_capture)
    verify_semantics(
        llm, "How many genes?", "MATCH (g:Gene) RETURN count(g)", [{"count(g)": 3}],
        schema="Node properties:\n- **Gene**\n  - `Label`: STRING",
    )

    assert "Graph schema:" in captured["text"]
    assert "- **Gene**" in captured["text"]


def test_verify_semantics_includes_examples_when_given():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=True, reasoning="ok")

    llm = _fake_llm(_capture)
    verify_semantics(
        llm, "q", "MATCH (n) RETURN n", [{"n": 1}],
        examples="[Query natural language] How many genes?\n[Query Cypher] MATCH (g:Gene) RETURN count(g)",
    )

    assert "Examples:" in captured["text"]
    assert "MATCH (g:Gene) RETURN count(g)" in captured["text"]


def test_verify_semantics_omits_schema_and_examples_blocks_when_not_given():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=True, reasoning="ok")

    llm = _fake_llm(_capture)
    verify_semantics(llm, "q", "MATCH (n) RETURN n", [{"n": 1}])

    assert "Graph schema:" not in captured["text"]
    assert "Examples:" not in captured["text"]


def test_verify_semantics_truncates_a_large_result_preview():
    captured = {}

    def _capture(prompt_value):
        captured["text"] = prompt_value.to_string()
        return SemanticVerification(answers_question=True, reasoning="ok")

    llm = _fake_llm(_capture)
    big_result = [{"n": i} for i in range(20)]
    verify_semantics(llm, "q", "MATCH (n) RETURN n", big_result)

    assert "{'n': 4}" in captured["text"]
    assert "{'n': 5}" not in captured["text"]
