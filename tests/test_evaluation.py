import pandas as pd
import pytest

from text2cypher_composer.core import Text2CypherResult
from text2cypher_composer.evaluation import (
    EvaluationReport,
    EvaluationSummary,
    QuestionEvaluation,
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
