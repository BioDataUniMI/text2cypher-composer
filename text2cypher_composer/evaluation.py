from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .core import DatabaseLike, DatasetLike, Text2CypherResult, run
from .cypher_utils import CypherExecutionError, execute_cypher, normalize_generated_cypher
from .graph_db import resolve_database
from .llm import ModelLike
from .metrics import (
    coverage_similarity,
    jaccard_similarity,
    jaro_winkler_similarity,
    normalized_levenshtein_similarity,
)


@dataclass
class QuestionEvaluation:
    """The `k`-attempt evaluation of a single (question, gold query) pair.

    `jaro_winkler`/`levenshtein`/`jaccard`/`coverage` are computed on the
    first attempt only. `passes[i]` is whether attempt `i` alone achieved
    `coverage == 1.0`; `pass_at(j)` folds `passes[:j]` into "at least one of
    the first j attempts passed". `extra` is any `df` columns other than
    "question"/"query" (e.g. bio2C's "ID"/"level"), carried through unchanged
    for traceability in `to_dataframe()`.
    """

    question: str
    gold_cypher: str
    gold_data: List[Dict[str, Any]]
    attempts: List[Text2CypherResult]
    jaro_winkler: float
    levenshtein: float
    jaccard: float
    coverage: float
    passes: List[bool]
    extra: Dict[str, Any] = field(default_factory=dict)

    def pass_at(self, j: int) -> bool:
        return any(self.passes[:j])


@dataclass
class EvaluationSummary:
    """Dataset-level averages over all evaluated questions."""

    technique: str
    model: str
    n_questions: int
    k: int
    mean_jaro_winkler: float
    mean_levenshtein: float
    mean_jaccard: float
    mean_coverage: float
    pass_at_k: Dict[int, float]


@dataclass
class EvaluationReport:
    summary: EvaluationSummary
    details: List[QuestionEvaluation]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten `details` into one row per question, with a pass@j column for every j in 1..k.

        Beyond the core metrics, each row also carries: any `extra` columns
        from the input `df` (e.g. bio2C's "ID"/"level"); `prompt`, the exact
        messages sent for the first attempt; `gold_data`/`predicted_data`,
        the gold/generated query's result rows; and
        `retrieved_example_ids`/`retrieved_example_distances` for RAG
        techniques (`None` otherwise) — see `Text2CypherResult.retrieved_examples`.
        """
        k = self.summary.k
        rows = []
        for d in self.details:
            first = d.attempts[0]
            predicted_data = first.result if first.executed else []
            retrieved = first.retrieved_examples or {}
            row = {
                "question": d.question,
                **d.extra,
                "gold_cypher": d.gold_cypher,
                "generated_cypher": first.cypher,
                "executed": first.executed,
                "jaro_winkler": d.jaro_winkler,
                "levenshtein": d.levenshtein,
                "jaccard": d.jaccard,
                "coverage": d.coverage,
            }
            for j in range(1, k + 1):
                row[f"pass@{j}"] = d.pass_at(j)
            row["prompt"] = first.prompt
            row["gold_data"] = d.gold_data
            row["predicted_data"] = predicted_data
            row["retrieved_example_ids"] = retrieved.get("example_ids")
            row["retrieved_example_distances"] = retrieved.get("example_distances")
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_technique(
    df: pd.DataFrame,
    model: ModelLike,
    database: DatabaseLike,
    technique: str,
    dataset: Optional[DatasetLike] = None,
    k: int = 1,
) -> EvaluationReport:
    """Evaluate `technique` in bulk over a gold (question, query) test set.

    For each row, generates `k` independent Cypher completions for the
    question (via `run()`, so every attempt goes through the full
    schema/RAG/execution/CyVer pipeline for `technique`) and scores them
    against the gold query's actual result rows (obtained by executing
    `row["query"]` against `database`):

    - `jaro_winkler` / `levenshtein`: text similarity between the gold and
      generated Cypher (computed on the first attempt only).
    - `jaccard` / `coverage`: structural similarity between the gold and
      generated result *rows* (computed on the first attempt only) — see
      `text2cypher_composer.metrics` for their definitions.
    - `pass@j` for every `j` in `1..k`: whether at least one of the first
      `j` attempts achieved `coverage == 1.0`.

    Args:
        df: DataFrame with "question" and "query" columns (the gold set).
        model, database, technique, dataset: forwarded to `run()` for every
            attempt — see `run()` for their meaning and constraints (e.g.
            `dataset` required iff `technique` uses RAG).
        k: number of independent generation attempts per question. Defaults
            to 1 (only `pass@1` is reported); use a larger `k` to also get
            `pass@2`, ..., `pass@k`.

    Returns:
        An EvaluationReport: a dataset-level `summary` (mean metrics plus
        `pass_at_k`), and per-question `details` (`.to_dataframe()` for a
        flat table).
    """
    if "question" not in df.columns or "query" not in df.columns:
        raise ValueError("`df` must have 'question' and 'query' columns.")
    if k < 1:
        raise ValueError("`k` must be >= 1.")

    graph = resolve_database(database)
    extra_cols = [c for c in df.columns if c not in ("question", "query")]

    details: List[QuestionEvaluation] = []
    for _, row in df.iterrows():
        question = str(row["question"])
        gold_cypher = normalize_generated_cypher(str(row["query"]))

        try:
            gold_data = execute_cypher(graph, gold_cypher)
        except CypherExecutionError:
            gold_data = []

        attempts: List[Text2CypherResult] = []
        attempt_coverages: List[float] = []
        for _ in range(k):
            result = run(question, model, database=graph, technique=technique, dataset=dataset)
            attempts.append(result)
            pred_data = result.result if result.executed else []
            attempt_coverages.append(coverage_similarity(gold_cypher, gold_data, result.cypher, pred_data))

        first = attempts[0]
        first_pred_data = first.result if first.executed else []

        details.append(
            QuestionEvaluation(
                question=question,
                gold_cypher=gold_cypher,
                gold_data=gold_data,
                attempts=attempts,
                jaro_winkler=jaro_winkler_similarity(gold_cypher, first.cypher),
                levenshtein=normalized_levenshtein_similarity(gold_cypher, first.cypher),
                jaccard=jaccard_similarity(gold_cypher, gold_data, first.cypher, first_pred_data),
                coverage=attempt_coverages[0],
                passes=[c >= 1.0 for c in attempt_coverages],
                extra={c: row[c] for c in extra_cols},
            )
        )

    n = len(details)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / n if n else 0.0

    summary = EvaluationSummary(
        technique=str(technique),
        model=model if isinstance(model, str) else type(model).__name__,
        n_questions=n,
        k=k,
        mean_jaro_winkler=_mean([d.jaro_winkler for d in details]),
        mean_levenshtein=_mean([d.levenshtein for d in details]),
        mean_jaccard=_mean([d.jaccard for d in details]),
        mean_coverage=_mean([d.coverage for d in details]),
        pass_at_k={j: _mean([1.0 if d.pass_at(j) else 0.0 for d in details]) for j in range(1, k + 1)},
    )

    return EvaluationReport(summary=summary, details=details)


_UNSAFE_FILENAME_CHARS = re.compile(r"[\\/:]")


def save_evaluation_report(report: EvaluationReport, output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Persist `report` to disk as a `.pkl` and an `.xlsx`, one pair per (model, technique).

    Named `evaluating_text2cypher_{model}_{technique}.{pkl,xlsx}` under
    `output_dir` (created if missing), taking `model`/`technique` from
    `report.summary` — mirroring the naming convention of the bio2C
    evaluation notebooks `evaluate_technique` was ported from. Characters
    unsafe in a filename (`/`, `\\`, `:` — e.g. a HuggingFace model id or a
    `"ft:..."` id) are replaced with `_`.

    The `.pkl` holds `report.to_dataframe()` as-is (lists/dicts intact —
    `prompt`, `gold_data`, `predicted_data`, `retrieved_example_*`, no extra
    dependency needed). The `.xlsx` is the same table with any list/dict
    column stringified (Excel has no native list/dict type), and needs the
    optional `excel` dependency (`openpyxl`) — install with
    `pip install "text2cypher-composer[excel]"`.

    Returns `{"pkl": <path>, "xlsx": <path>}`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _UNSAFE_FILENAME_CHARS.sub("_", report.summary.model)
    technique = _UNSAFE_FILENAME_CHARS.sub("_", report.summary.technique)
    stem = f"evaluating_text2cypher_{model}_{technique}"
    pkl_path = output_dir / f"{stem}.pkl"
    xlsx_path = output_dir / f"{stem}.xlsx"

    df = report.to_dataframe()
    df.to_pickle(pkl_path)

    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "save_evaluation_report requires the optional excel dependency (openpyxl) to "
            'write the .xlsx file. Install it with `pip install "text2cypher-composer[excel]"`.'
        ) from e

    excel_df = df.copy()
    for col in excel_df.columns:
        if excel_df[col].map(lambda v: isinstance(v, (list, dict))).any():
            excel_df[col] = excel_df[col].map(str)
    excel_df.to_excel(xlsx_path, index=False)

    return {"pkl": pkl_path, "xlsx": xlsx_path}
