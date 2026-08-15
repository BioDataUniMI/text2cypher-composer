"""Prompt token counting via `tiktoken` (optional `dataset-tools` extra)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def count_message_tokens(messages: List[Dict[str, str]], model: Any = "gpt-4o") -> Optional[int]:
    """Token count of a fully-instantiated prompt (list of `{"role", "content"}` messages).

    Returns `None` if `tiktoken` isn't installed, rather than raising —
    token counting is ancillary to `run()`, so its absence shouldn't break
    generation for everyone; install it with
    `pip install "text2cypher-composer[dataset-tools]"` to get real counts.

    `model` is used to pick tiktoken's encoding via `encoding_for_model`;
    for anything it doesn't recognize by name (non-OpenAI models, `"ft:..."`
    ids, a LangChain `Runnable`'s class name, ...) this falls back to
    `cl100k_base` — a reasonable approximation for comparing prompt sizes
    across techniques/schema modes, not an exact count for that model.
    """
    try:
        import tiktoken
    except ImportError:
        return None

    model_name = model if isinstance(model, str) else type(model).__name__
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return sum(len(encoding.encode(m.get("content", ""))) for m in messages)
