from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from text2cypher_composer.core import Text2CypherResult
from text2cypher_composer.evaluation import (
    EvaluationReport,
    EvaluationSummary,
    QuestionEvaluation,
    evaluate_technique,
    save_evaluation_report,
)


def _fake_result(**overrides) -> Text2CypherResult:
    defaults = dict(
        question="How many genes are there?",
        technique="Schema+RAG",
        model="gpt-4o-mini",
        cypher="MATCH (g:Gene) RETURN count(g) AS c",
        initial_cypher="MATCH (g:Gene) RETURN count(g) AS c",
        prompt=[{"role": "human", "content": "How many genes are there?"}],
        executed=True,
        result=[{"c": 42}],
        retrieved_examples={"example_ids": ["ex_1", "ex_2"], "example_distances": [0.1, 0.2]},
    )
    defaults.update(overrides)
    return Text2CypherResult(**defaults)


def _fake_question_evaluation(**overrides) -> QuestionEvaluation:
    defaults = dict(
        question="How many genes are there?",
        gold_cypher="MATCH (g:Gene) RETURN count(g) AS c",
        gold_data=[{"c": 42}],
        attempts=[_fake_result()],
        jaro_winkler=1.0,
        levenshtein=1.0,
        jaccard=1.0,
        coverage=1.0,
        passes=[True],
        extra={"ID": "nodeLevel_1", "level": "nodeLevel"},
    )
    defaults.update(overrides)
    return QuestionEvaluation(**defaults)


def _fake_report(**summary_overrides) -> EvaluationReport:
    details = [_fake_question_evaluation()]
    summary_defaults = dict(
        technique="Schema+RAG",
        model="gpt-4o-mini",
        n_questions=1,
        k=1,
        mean_jaro_winkler=1.0,
        mean_levenshtein=1.0,
        mean_jaccard=1.0,
        mean_coverage=1.0,
        pass_at_k={1: 1.0},
    )
    summary_defaults.update(summary_overrides)
    return EvaluationReport(summary=EvaluationSummary(**summary_defaults), details=details)


def test_to_dataframe_includes_extra_prompt_and_data_columns():
    df = _fake_report().to_dataframe()
    row = df.iloc[0]

    assert row["ID"] == "nodeLevel_1"
    assert row["level"] == "nodeLevel"
    assert row["prompt"] == [{"role": "human", "content": "How many genes are there?"}]
    assert row["gold_data"] == [{"c": 42}]
    assert row["predicted_data"] == [{"c": 42}]
    assert row["retrieved_example_ids"] == ["ex_1", "ex_2"]
    assert row["retrieved_example_distances"] == [0.1, 0.2]


def test_to_dataframe_includes_rescue_columns():
    rescued_result = _fake_result(
        rescued=True,
        rescue_attempts=2,
        rescue_error_messages=["Cypher syntax error: bad", "Empty result set."],
        rescue_prompts=[
            [{"role": "human", "content": "fix attempt 1"}],
            [{"role": "human", "content": "fix attempt 2"}],
        ],
        rescue_prompt_tokens=[12, 15],
        prompt_tokens=8,
        execution_error=None,
        execution_warnings=["Warning: deprecated - Position: line 1, column 1"],
    )
    details = [_fake_question_evaluation(attempts=[rescued_result])]
    report = EvaluationReport(
        summary=EvaluationSummary(
            technique="Schema+RAG",
            model="gpt-4o-mini",
            n_questions=1,
            k=1,
            mean_jaro_winkler=1.0,
            mean_levenshtein=1.0,
            mean_jaccard=1.0,
            mean_coverage=1.0,
            pass_at_k={1: 1.0},
        ),
        details=details,
    )

    row = report.to_dataframe().iloc[0]
    assert bool(row["rescued"]) is True
    assert row["rescue_attempts"] == 2
    assert row["rescue_error_messages"] == ["Cypher syntax error: bad", "Empty result set."]
    assert row["rescue_prompts"] == [
        [{"role": "human", "content": "fix attempt 1"}],
        [{"role": "human", "content": "fix attempt 2"}],
    ]
    assert row["rescue_prompt_tokens"] == [12, 15]
    assert row["prompt_tokens"] == 8
    assert row["execution_error"] is None
    assert row["execution_warnings"] == ["Warning: deprecated - Position: line 1, column 1"]


def test_to_dataframe_rescue_columns_default_when_not_rescued():
    row = _fake_report().to_dataframe().iloc[0]
    assert bool(row["rescued"]) is False
    assert row["rescue_attempts"] == 0
    assert row["rescue_error_messages"] == []
    assert row["rescue_prompts"] == []
    assert row["rescue_prompt_tokens"] == []
    assert row["execution_error"] is None
    assert row["execution_warnings"] == []


def test_to_dataframe_includes_self_verification_columns():
    verified_result = _fake_result(
        self_verification_passed=False,
        self_verification_reasoning="wrong relationship direction",
    )
    details = [_fake_question_evaluation(attempts=[verified_result])]
    report = EvaluationReport(
        summary=EvaluationSummary(
            technique="Schema+RAG",
            model="gpt-4o-mini",
            n_questions=1,
            k=1,
            mean_jaro_winkler=1.0,
            mean_levenshtein=1.0,
            mean_jaccard=1.0,
            mean_coverage=1.0,
            pass_at_k={1: 1.0},
        ),
        details=details,
    )

    row = report.to_dataframe().iloc[0]
    assert bool(row["self_verification_passed"]) is False
    assert row["self_verification_reasoning"] == "wrong relationship direction"


def test_to_dataframe_self_verification_columns_default_when_not_used():
    row = _fake_report().to_dataframe().iloc[0]
    assert row["self_verification_passed"] is None
    assert row["self_verification_reasoning"] is None


def test_to_dataframe_includes_cascade_mode_columns():
    cascade_result = _fake_result(
        cascade_mode_level="nodes_only",
        cascade_mode_attempts=2,
        cascade_mode_prompts=[
            [{"role": "human", "content": "narrow attempt"}],
            [{"role": "human", "content": "nodes_only attempt"}],
        ],
        cascade_mode_prompt_tokens=[10, 25],
    )
    details = [_fake_question_evaluation(attempts=[cascade_result])]
    report = EvaluationReport(
        summary=EvaluationSummary(
            technique="Schema+RAG",
            model="gpt-4o-mini",
            n_questions=1,
            k=1,
            mean_jaro_winkler=1.0,
            mean_levenshtein=1.0,
            mean_jaccard=1.0,
            mean_coverage=1.0,
            pass_at_k={1: 1.0},
        ),
        details=details,
    )

    row = report.to_dataframe().iloc[0]
    assert row["cascade_mode_level"] == "nodes_only"
    assert row["cascade_mode_attempts"] == 2
    assert row["cascade_mode_prompts"] == [
        [{"role": "human", "content": "narrow attempt"}],
        [{"role": "human", "content": "nodes_only attempt"}],
    ]
    assert row["cascade_mode_prompt_tokens"] == [10, 25]


def test_to_dataframe_cascade_mode_columns_default_when_not_used():
    row = _fake_report().to_dataframe().iloc[0]
    assert row["cascade_mode_level"] is None
    assert row["cascade_mode_attempts"] == 0
    assert row["cascade_mode_prompts"] == []
    assert row["cascade_mode_prompt_tokens"] == []


def test_to_dataframe_includes_adaptive_rag_columns():
    adaptive_result = _fake_result(
        adaptive_rag_level="moderate",
        adaptive_rag_attempts=2,
        adaptive_rag_prompts=[
            [{"role": "human", "content": "minimal attempt"}],
            [{"role": "human", "content": "moderate attempt"}],
        ],
        adaptive_rag_prompt_tokens=[18, 40],
    )
    details = [_fake_question_evaluation(attempts=[adaptive_result])]
    report = EvaluationReport(
        summary=EvaluationSummary(
            technique="Schema+RAG",
            model="gpt-4o-mini",
            n_questions=1,
            k=1,
            mean_jaro_winkler=1.0,
            mean_levenshtein=1.0,
            mean_jaccard=1.0,
            mean_coverage=1.0,
            pass_at_k={1: 1.0},
        ),
        details=details,
    )

    row = report.to_dataframe().iloc[0]
    assert row["adaptive_rag_level"] == "moderate"
    assert row["adaptive_rag_attempts"] == 2
    assert row["adaptive_rag_prompts"] == [
        [{"role": "human", "content": "minimal attempt"}],
        [{"role": "human", "content": "moderate attempt"}],
    ]
    assert row["adaptive_rag_prompt_tokens"] == [18, 40]


def test_to_dataframe_adaptive_rag_columns_default_when_not_used():
    row = _fake_report().to_dataframe().iloc[0]
    assert row["adaptive_rag_level"] is None
    assert row["adaptive_rag_attempts"] == 0
    assert row["adaptive_rag_prompts"] == []
    assert row["adaptive_rag_prompt_tokens"] == []


def test_to_dataframe_retrieved_examples_none_for_non_rag():
    non_rag_result = _fake_result(technique="vanilla", retrieved_examples=None)
    details = [_fake_question_evaluation(attempts=[non_rag_result])]
    report = EvaluationReport(
        summary=EvaluationSummary(
            technique="vanilla",
            model="gpt-4o-mini",
            n_questions=1,
            k=1,
            mean_jaro_winkler=1.0,
            mean_levenshtein=1.0,
            mean_jaccard=1.0,
            mean_coverage=1.0,
            pass_at_k={1: 1.0},
        ),
        details=details,
    )

    row = report.to_dataframe().iloc[0]
    assert row["retrieved_example_ids"] is None
    assert row["retrieved_example_distances"] is None


def test_evaluate_technique_forwards_rescue_prompt_and_max_retries_to_run():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(
            df,
            model="gpt-4o-mini",
            database={},
            technique="vanilla",
            rescue_prompt=True,
            max_retries=3,
        )

    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["rescue_prompt"] is True
    assert mock_run.call_args.kwargs["max_retries"] == 3


def test_evaluate_technique_defaults_rescue_prompt_to_false():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(df, model="gpt-4o-mini", database={}, technique="vanilla")

    assert mock_run.call_args.kwargs["rescue_prompt"] is False
    assert mock_run.call_args.kwargs["max_retries"] == 1


def test_evaluate_technique_forwards_self_verification_params_to_run():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(
            df,
            model="gpt-4o-mini",
            database={},
            technique="vanilla",
            rescue_prompt=True,
            self_verification=True,
            verification_model="gpt-4o",
            verification_criteria="must include units",
        )

    assert mock_run.call_args.kwargs["self_verification"] is True
    assert mock_run.call_args.kwargs["verification_model"] == "gpt-4o"
    assert mock_run.call_args.kwargs["verification_criteria"] == "must include units"


def test_evaluate_technique_defaults_self_verification_to_false():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(df, model="gpt-4o-mini", database={}, technique="vanilla")

    assert mock_run.call_args.kwargs["self_verification"] is False
    assert mock_run.call_args.kwargs["verification_model"] is None
    assert mock_run.call_args.kwargs["verification_criteria"] is None


def test_evaluate_technique_forwards_schema_params_to_run():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])
    fake_nlp = object()
    fake_ie_engine = object()

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(
            df,
            model="gpt-4o-mini",
            database={},
            technique="Schema",
            schema_mode="exact_match",
            schema_components=["entity_types", "relationship_types"],
            nlp=fake_nlp,
            similarity_threshold=0.7,
            ie_engine=fake_ie_engine,
        )

    assert mock_run.call_args.kwargs["schema_mode"] == "exact_match"
    assert mock_run.call_args.kwargs["schema_components"] == ["entity_types", "relationship_types"]
    assert mock_run.call_args.kwargs["nlp"] is fake_nlp
    assert mock_run.call_args.kwargs["similarity_threshold"] == 0.7
    assert mock_run.call_args.kwargs["ie_engine"] is fake_ie_engine


def test_evaluate_technique_defaults_schema_params():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(df, model="gpt-4o-mini", database={}, technique="vanilla")

    assert mock_run.call_args.kwargs["schema_mode"] is None
    assert mock_run.call_args.kwargs["nlp"] is None
    assert mock_run.call_args.kwargs["similarity_threshold"] == 0.5
    assert mock_run.call_args.kwargs["ie_engine"] is None


def test_evaluate_technique_forwards_cascade_mode_params_to_run():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(
            df,
            model="gpt-4o-mini",
            database={},
            technique="Schema",
            schema_mode="exact_match",
            cascade_mode=True,
            skip_narrow_schema_filter=True,
            cascade_strategy="delta",
        )

    assert mock_run.call_args.kwargs["cascade_mode"] is True
    assert mock_run.call_args.kwargs["skip_narrow_schema_filter"] is True
    assert mock_run.call_args.kwargs["cascade_strategy"] == "delta"


def test_evaluate_technique_forwards_adaptive_rag_to_run():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(df, model="gpt-4o-mini", database={}, technique="RAG", dataset="fake/root", adaptive_rag=True)

    assert mock_run.call_args.kwargs["adaptive_rag"] is True


def test_evaluate_technique_defaults_cascade_and_adaptive_rag_to_false():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with patch("text2cypher_composer.evaluation.resolve_database", return_value=MagicMock()), \
         patch("text2cypher_composer.evaluation.execute_cypher", return_value=[{"c": 42}]), \
         patch("text2cypher_composer.evaluation.run", return_value=_fake_result()) as mock_run:
        evaluate_technique(df, model="gpt-4o-mini", database={}, technique="vanilla")

    assert mock_run.call_args.kwargs["cascade_mode"] is False
    assert mock_run.call_args.kwargs["skip_narrow_schema_filter"] is False
    assert mock_run.call_args.kwargs["cascade_strategy"] == "standard"
    assert mock_run.call_args.kwargs["adaptive_rag"] is False


def test_save_evaluation_report_writes_pkl_and_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    report = _fake_report()

    paths = save_evaluation_report(report, tmp_path)

    assert paths["pkl"] == tmp_path / "evaluating_text2cypher_gpt-4o-mini_Schema+RAG.pkl"
    assert paths["xlsx"] == tmp_path / "evaluating_text2cypher_gpt-4o-mini_Schema+RAG.xlsx"
    assert paths["pkl"].exists()
    assert paths["xlsx"].exists()

    pkl_df = pd.read_pickle(paths["pkl"])
    assert pkl_df.iloc[0]["gold_data"] == [{"c": 42}]

    xlsx_df = pd.read_excel(paths["xlsx"])
    assert xlsx_df.iloc[0]["gold_data"] == str([{"c": 42}])


def test_save_evaluation_report_sanitizes_unsafe_model_chars(tmp_path):
    pytest.importorskip("openpyxl")
    report = _fake_report(model="ft:gpt-4o-mini:acme::abc123")

    paths = save_evaluation_report(report, tmp_path)

    assert ":" not in paths["pkl"].name
    assert ":" not in paths["xlsx"].name


def test_save_evaluation_report_xlsx_maps_booleans_to_1_0(tmp_path):
    """Excel renders Python bool in the *display* locale (e.g. VERO/FALSO in Italian Excel) --
    the .xlsx export should carry 1/0 instead, so the value reads the same everywhere. The .pkl
    keeps real bools (readable back into pandas/Python as-is)."""
    pytest.importorskip("openpyxl")
    rescued_result = _fake_result(rescued=True, self_verification_passed=False)
    details = [_fake_question_evaluation(attempts=[rescued_result])]
    report = EvaluationReport(
        summary=EvaluationSummary(
            technique="Schema+RAG",
            model="gpt-4o-mini",
            n_questions=1,
            k=1,
            mean_jaro_winkler=1.0,
            mean_levenshtein=1.0,
            mean_jaccard=1.0,
            mean_coverage=1.0,
            pass_at_k={1: 1.0},
        ),
        details=details,
    )

    paths = save_evaluation_report(report, tmp_path)

    pkl_df = pd.read_pickle(paths["pkl"])
    assert bool(pkl_df.iloc[0]["executed"]) is True
    assert bool(pkl_df.iloc[0]["rescued"]) is True
    assert bool(pkl_df.iloc[0]["pass@1"]) is True

    xlsx_df = pd.read_excel(paths["xlsx"])
    assert xlsx_df.iloc[0]["executed"] == 1
    assert xlsx_df.iloc[0]["rescued"] == 1
    assert xlsx_df.iloc[0]["pass@1"] == 1
    assert xlsx_df.iloc[0]["self_verification_passed"] == 0


def test_evaluate_technique_requires_cypher_column():
    df = pd.DataFrame([{"question": "How many genes?", "query": "MATCH (g:Gene) RETURN count(g) AS c"}])

    with pytest.raises(ValueError, match="cypher"):
        evaluate_technique(df, model="gpt-4o-mini", database={}, technique="vanilla")
