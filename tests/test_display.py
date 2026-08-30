from text2cypher_composer.core import Text2CypherResult
from text2cypher_composer.display import show
from text2cypher_composer.validation import CypherValidationReport


def _validation(**overrides):
    defaults = dict(
        syntax_valid=True,
        syntax_metadata=[],
        schema_score=1.0,
        schema_metadata=[],
        properties_score=1.0,
        properties_metadata=[],
    )
    defaults.update(overrides)
    return CypherValidationReport(**defaults)


def _result(**overrides) -> Text2CypherResult:
    defaults = dict(
        question="How many genes are there?",
        technique="vanilla",
        model="gpt-4o",
        cypher="MATCH (g:Gene) RETURN count(g) AS c",
        initial_cypher="MATCH (g:Gene) RETURN count(g) AS c",
        prompt=[{"role": "human", "content": "How many genes are there?"}],
        executed=True,
        result=[{"c": 42}],
        validation=_validation(),
    )
    defaults.update(overrides)
    return Text2CypherResult(**defaults)


def test_show_dry_run_prints_placeholder_and_returns_early(capsys):
    show(_result(dry_run=True, cypher=None, result=None, validation=None))
    out = capsys.readouterr().out
    assert "[dry_run]" in out
    assert "Generated Cypher" not in out


def test_show_prints_basic_fields_and_cyver_report(capsys):
    show(_result())
    out = capsys.readouterr().out
    assert "Technique:        vanilla" in out
    assert "MATCH (g:Gene) RETURN count(g) AS c" in out
    assert "Result (1 rows):" in out
    assert "Syntax valid:      True" in out


def test_show_reports_execution_failure(capsys):
    show(_result(executed=False, result=None, execution_error="Cypher syntax error: bad"))
    out = capsys.readouterr().out
    assert "Execution FAILED" in out
    assert "Cypher syntax error: bad" in out


def test_show_prints_rescue_info_when_rescued(capsys):
    result = _result(
        rescued=True,
        rescue_attempts=1,
        rescue_error_messages=["Empty result set."],
        rescue_prompts=[[{"role": "human", "content": "fix it"}]],
        rescue_prompt_tokens=[10],
    )
    show(result)
    out = capsys.readouterr().out
    assert "Rescue attempts: 1" in out
    assert "Empty result set." in out


def test_show_prints_cascade_mode_info_when_present(capsys):
    result = _result(cascade_mode_level="nodes_only", cascade_mode_attempts=2)
    show(result)
    out = capsys.readouterr().out
    assert "Schema fallback rung used: nodes_only  (of 2 tried)" in out


def test_show_prints_adaptive_rag_info_when_present(capsys):
    result = _result(adaptive_rag_level="full", adaptive_rag_attempts=3)
    show(result)
    out = capsys.readouterr().out
    assert "RAG expansion rung used: full  (of 3 tried)" in out


def test_show_prints_self_verification_info_when_present(capsys):
    result = _result(self_verification_passed=False, self_verification_reasoning="wrong direction")
    show(result)
    out = capsys.readouterr().out
    assert "Self-verification: failed" in out
    assert "wrong direction" in out


def test_show_omits_self_verification_block_when_unused(capsys):
    show(_result())
    out = capsys.readouterr().out
    assert "Self-verification" not in out


def test_show_prompt_true_prints_every_rescue_prompt_too(capsys):
    result = _result(
        rescued=True,
        rescue_prompts=[
            [{"role": "human", "content": "fix attempt 1"}],
            [{"role": "human", "content": "fix attempt 2"}],
        ],
    )
    show(result, show_prompt=True)
    out = capsys.readouterr().out
    assert "1 initial + 2 rescue" in out
    assert "fix attempt 1" in out
    assert "fix attempt 2" in out
