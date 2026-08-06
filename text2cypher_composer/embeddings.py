"""Pluggable embedding backends for RAG (`RAGDataset`), plus persisting which
one was used at index time so retrieval automatically reuses it — the user
never has to pass `embedding_model` again after `RAGDataset.index_from_root`.

A string `embedding_model` is resolved to a backend by a simple, deterministic
rule: HuggingFace/sentence-transformers model ids conventionally look like
"<namespace>/<model>" (e.g. "sentence-transformers/all-mpnet-base-v2"), while
OpenAI's don't (e.g. "text-embedding-3-large") — so a "/" in the string means
HuggingFace, anything else means OpenAI. An already-built LangChain
`Embeddings` instance (the "custom" backend) is used as-is, the same "bring
your own object" pattern `model`/`nlp` use elsewhere in this library — but it
can't be persisted/reconstructed from a string, so it must be supplied again
(the same instance, or an equivalent one) at retrieval time.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from langchain_core.embeddings import Embeddings

EmbeddingModelLike = Union[str, Embeddings]

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


def embedding_backend_for(embedding_model: EmbeddingModelLike) -> str:
    """"openai", "huggingface", or "custom" (an already-built `Embeddings` instance)."""
    if isinstance(embedding_model, Embeddings):
        return "custom"
    if isinstance(embedding_model, str):
        return "huggingface" if "/" in embedding_model else "openai"
    raise TypeError("`embedding_model` must be a model id (str) or a LangChain Embeddings instance.")


def _require_openai_api_key(embedding_model: str) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            f"embedding_model='{embedding_model}' is an OpenAI embedding model, but "
            "OPENAI_API_KEY is not set. Set it before indexing/retrieving, or use a "
            "local model instead (e.g. "
            'embedding_model="sentence-transformers/all-mpnet-base-v2").'
        )


def resolve_embedder(embedding_model: EmbeddingModelLike) -> Embeddings:
    """Build (or pass through) the `Embeddings` instance for `embedding_model`."""
    if isinstance(embedding_model, Embeddings):
        return embedding_model

    backend = embedding_backend_for(embedding_model)
    if backend == "openai":
        _require_openai_api_key(embedding_model)
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=embedding_model)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        raise ImportError(
            "Using a HuggingFace/sentence-transformers embedding model "
            f"('{embedding_model}') requires the optional local-embeddings dependency. "
            'Install it with `pip install "text2cypher-composer[local-embeddings]"`.'
        ) from e
    return HuggingFaceEmbeddings(model_name=embedding_model)


@dataclass
class EmbeddingMeta:
    """What `embedding_model` a Chroma collection was indexed with — see `read_embedding_meta`."""

    backend: str
    model: Optional[str]


def embedding_meta_path(chroma_path: Union[str, Path], collection_name: str) -> Path:
    return Path(chroma_path) / f"{collection_name}.embedding.json"


def write_embedding_meta(
    chroma_path: Union[str, Path], collection_name: str, embedding_model: EmbeddingModelLike
) -> None:
    """Record which embedding backend/model a collection was indexed with.

    So that a later `RAGDataset.from_root`/`index_from_root` call can reuse it
    automatically instead of the caller having to remember and re-specify it —
    which matters, since embedding a retrieval query with a different model
    than the one used to index the documents silently produces meaningless
    similarity scores.
    """
    backend = embedding_backend_for(embedding_model)
    model = embedding_model if isinstance(embedding_model, str) else None
    path = embedding_meta_path(chroma_path, collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"backend": backend, "model": model}), encoding="utf-8")


def read_embedding_meta(chroma_path: Union[str, Path], collection_name: str) -> Optional[EmbeddingMeta]:
    path = embedding_meta_path(chroma_path, collection_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EmbeddingMeta(backend=data.get("backend"), model=data.get("model"))


def reconcile_embedding_model(
    requested: Optional[EmbeddingModelLike], saved: Optional[EmbeddingMeta], collection_name: str
) -> EmbeddingModelLike:
    """The embedding model to actually use for an existing collection.

    `requested` is what the caller passed (`None` if they didn't specify
    anything). If nothing was recorded for this collection (a legacy
    collection, indexed before this feature existed), falls back to
    `requested` (or the OpenAI default). Otherwise the recorded model is
    authoritative: if `requested` is `None`, it's used as-is (no need to
    re-specify); if `requested` was given, it must match — a mismatch almost
    always means the caller forgot what this collection was actually indexed
    with, so it's treated as an error rather than silently corrupting
    retrieval.
    """
    if saved is None:
        return requested if requested is not None else DEFAULT_OPENAI_EMBEDDING_MODEL

    if saved.backend == "custom" and requested is None:
        raise ValueError(
            f"Collection '{collection_name}' was indexed with a custom (non-string) "
            "Embeddings object, which can't be reconstructed automatically. Pass the "
            "same (or an equivalent) Embeddings instance as `embedding_model=` to use it."
        )

    if requested is None:
        return saved.model

    requested_backend = embedding_backend_for(requested)
    mismatched_backend = requested_backend != saved.backend
    mismatched_model = isinstance(requested, str) and requested != saved.model
    if mismatched_backend or mismatched_model:
        recorded = saved.model if saved.model is not None else f"<custom {saved.backend} object>"
        raise ValueError(
            f"Collection '{collection_name}' was indexed with embedding_model={recorded!r} "
            f"({saved.backend}), but embedding_model={requested!r} ({requested_backend}) was "
            "requested now. Retrieval must use the exact same embedding model as indexing, or "
            "results are meaningless. Omit `embedding_model` to reuse the recorded one "
            "automatically, or pass rebuild=True to re-embed with the new one."
        )
    return requested
