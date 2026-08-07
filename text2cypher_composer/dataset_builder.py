from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import pandas as pd

from .graph_db import DatabaseLike, resolve_database


@dataclass
class RAGExampleFiles:
    """Paths to the example files written by `build_rag_example_files`."""

    root: Path
    name: str
    question_dir: Path
    cypher_dir: Path
    output_dir: Path
    count: int


def _truncate_values(obj: Any, max_len: int) -> Any:
    if isinstance(obj, dict):
        return {k: _truncate_values(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_values(v, max_len) for v in obj]
    if isinstance(obj, str):
        return (obj[: max_len - 3] + "...") if len(obj) > max_len else obj
    return obj


def _with_limit(cypher: str, limit: int) -> str:
    cypher = cypher.strip()
    if "limit" not in cypher.lower():
        cypher += f" LIMIT {limit}"
    return cypher


def build_rag_example_files(
    df: pd.DataFrame,
    name: str,
    database: DatabaseLike,
    root: Union[str, Path] = ".",
    limit: int = 5,
    truncate_len: int = 100,
) -> RAGExampleFiles:
    """Materialize a bio2C-style RAG example set from a (question, cypher) DataFrame.

    For each row (1-indexed, in row order) writes:
    - `NLquestions/{name}/question_i.txt` — the question;
    - `CypherQueries/{name}/cypher_i.txt` — the Cypher query;
    - `Neo4jOutputs/{name}/output_i.txt` — the query's output, obtained by
      running it against `database` (a `LIMIT` is appended if none is
      present); "No results" if it returns nothing, or the error message if
      it fails to execute.

    This produces the directory layout `RAGDataset.from_root(root)` expects.
    Building the Chroma index itself over `NLquestions/` is a separate step,
    not done here.

    Args:
        df: DataFrame with "question" and "cypher" columns.
        name: subfolder name for this example group (e.g. "1hop", "custom").
        database: a Neo4jGraph instance, or a dict with
            uri/username/password/database keys, used to execute each query
            and capture its output.
        root: directory under which NLquestions/, CypherQueries/, and
            Neo4jOutputs/ are created (may already hold other `name` groups).
        limit: row cap appended to queries that don't already have one.
        truncate_len: string values longer than this in a query's output are
            truncated (with a trailing "...") to keep output files compact.
    """
    if "question" not in df.columns or "cypher" not in df.columns:
        raise ValueError("`df` must have 'question' and 'cypher' columns.")

    graph = resolve_database(database)
    root = Path(root)

    question_dir = root / "NLquestions" / name
    cypher_dir = root / "CypherQueries" / name
    output_dir = root / "Neo4jOutputs" / name
    for directory in (question_dir, cypher_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        question = str(row["question"]).strip()
        cypher = str(row["cypher"]).strip()

        (question_dir / f"question_{i}.txt").write_text(question, encoding="utf-8")
        (cypher_dir / f"cypher_{i}.txt").write_text(cypher, encoding="utf-8")

        output_path = output_dir / f"output_{i}.txt"
        try:
            records = graph.query(_with_limit(cypher, limit))
        except Exception as e:
            output_path.write_text(f"Error during query execution:\n{e}", encoding="utf-8")
        else:
            if not records:
                output_path.write_text("No results", encoding="utf-8")
            else:
                blocks = [
                    json.dumps(_truncate_values(record, truncate_len), ensure_ascii=False, indent=2)
                    for record in records
                ]
                output_path.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")

        count += 1

    return RAGExampleFiles(
        root=root,
        name=name,
        question_dir=question_dir,
        cypher_dir=cypher_dir,
        output_dir=output_dir,
        count=count,
    )
