"""Preparing a fine-tuning dataset: load leveled gold files, split off a test
set, and export it either for `finetuning.finetune_lora` (a local JSON file)
or for OpenAI's fine-tuning GUI (a ready-to-upload chat-format `.jsonl`).

Ported from the miRNAKG evaluation notebooks' `FTdataset` preparation cells
(e.g. `bio2C/evaluating_text2cypher/evaluating_text2cypher_gpt.ipynb`).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd

from .prompts import SYSTEM_MESSAGE


def load_finetune_levels(level_paths: Dict[str, str]) -> pd.DataFrame:
    """Load and concatenate several per-level gold JSON files into one DataFrame.

    Mirrors the notebooks' `load_level`/concat cells: each
    `level_paths[level_name]` is a JSON file (a list of records, e.g. with
    "question"/"cypher" keys); loaded rows get an `"ID"`
    (`"{level_name}/question_{i}"`, 1-indexed) and a `"level"` column, then
    all levels are concatenated (index reset) in `level_paths` insertion
    order.
    """
    frames = []
    for level_name, path in level_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data).reset_index(drop=True)
        df["ID"] = f"{level_name}/question_" + (df.index + 1).astype(str)
        df["level"] = level_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def max_cypher_tokens(
    df: pd.DataFrame, cypher_col: str = "cypher", model: str = "gpt-4o", round_to: int = 100
) -> Dict[str, object]:
    """Token count of the longest Cypher query in `df`, plus a rounded-up token budget.

    Useful for sizing a generation `max_tokens` limit so it isn't needlessly
    large (which slows down bulk evaluation). Requires `tiktoken`
    (`pip install tiktoken`).
    """
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError("max_cypher_tokens requires `tiktoken` (`pip install tiktoken`).") from e

    longest = df.loc[df[cypher_col].str.len().idxmax(), cypher_col]
    encoding = tiktoken.encoding_for_model(model)
    n_tokens = len(encoding.encode(longest))
    return {
        "longest_cypher": longest,
        "n_tokens": n_tokens,
        "max_tokens": math.ceil(n_tokens / round_to) * round_to,
    }


def split_finetune_dataset(
    df: pd.DataFrame, level_col: str = "level", test_frac: float = 0.10, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-`level_col` stratified train/test split, matching the notebooks' `groupby(...).sample(...)`."""
    test_df = df.groupby(level_col, group_keys=False).sample(frac=test_frac, random_state=random_state)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def write_local_finetune_dataset(
    df: pd.DataFrame, path: Union[str, Path], question_col: str = "question", cypher_col: str = "cypher"
) -> Path:
    """Write `df[[question_col, cypher_col]]` as pretty-printed JSON records — the format `finetune_lora` reads."""
    if question_col not in df.columns or cypher_col not in df.columns:
        raise ValueError(f"`df` must have '{question_col}' and '{cypher_col}' columns.")
    path = Path(path)
    df[[question_col, cypher_col]].to_json(path, orient="records", indent=2)
    return path


@dataclass
class GPTFinetuneJSONL:
    """Where `build_gpt_finetune_jsonl` wrote its output, and how many examples it holds."""

    path: Path
    n_examples: int


def build_gpt_finetune_jsonl(
    df: pd.DataFrame,
    path: Union[str, Path],
    question_col: str = "question",
    cypher_col: str = "cypher",
    system_prompt: str = SYSTEM_MESSAGE,
    add_system: bool = True,
) -> GPTFinetuneJSONL:
    """Write a chat-format `.jsonl` ready to upload to the OpenAI fine-tuning GUI.

    Each row becomes one `{"messages": [...]}` line: an optional system
    message, the question as the user turn, and the gold Cypher as the
    assistant turn — the format OpenAI's fine-tuning UI expects for chat
    models.
    """
    if question_col not in df.columns or cypher_col not in df.columns:
        raise ValueError(f"`df` must have '{question_col}' and '{cypher_col}' columns.")

    path = Path(path)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            messages = []
            if add_system:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": row[question_col]})
            messages.append({"role": "assistant", "content": row[cypher_col]})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    return GPTFinetuneJSONL(path=path, n_examples=count)
