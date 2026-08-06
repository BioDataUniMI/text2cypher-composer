import pytest
from langchain_core.embeddings import Embeddings

from text2cypher_composer.embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingMeta,
    embedding_backend_for,
    read_embedding_meta,
    reconcile_embedding_model,
    resolve_embedder,
    write_embedding_meta,
)


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


def test_backend_detection():
    assert embedding_backend_for("text-embedding-3-large") == "openai"
    assert embedding_backend_for("sentence-transformers/all-mpnet-base-v2") == "huggingface"
    assert embedding_backend_for(_FakeEmbeddings()) == "custom"


def test_resolve_embedder_passes_through_custom_instance():
    fake = _FakeEmbeddings()
    assert resolve_embedder(fake) is fake


def test_resolve_embedder_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        resolve_embedder("text-embedding-3-large")


def test_embedding_meta_roundtrip(tmp_path):
    assert read_embedding_meta(tmp_path, "my_collection") is None
    write_embedding_meta(tmp_path, "my_collection", "sentence-transformers/all-mpnet-base-v2")
    meta = read_embedding_meta(tmp_path, "my_collection")
    assert meta == EmbeddingMeta(backend="huggingface", model="sentence-transformers/all-mpnet-base-v2")


def test_reconcile_no_saved_meta_falls_back_to_default():
    assert reconcile_embedding_model(None, None, "c") == DEFAULT_OPENAI_EMBEDDING_MODEL
    assert reconcile_embedding_model("foo", None, "c") == "foo"


def test_reconcile_auto_loads_saved_model_when_none_requested():
    saved = EmbeddingMeta(backend="huggingface", model="sentence-transformers/all-mpnet-base-v2")
    assert reconcile_embedding_model(None, saved, "c") == "sentence-transformers/all-mpnet-base-v2"


def test_reconcile_raises_on_mismatch():
    saved = EmbeddingMeta(backend="huggingface", model="sentence-transformers/all-mpnet-base-v2")
    with pytest.raises(ValueError, match="was indexed with"):
        reconcile_embedding_model("text-embedding-3-large", saved, "c")


def test_reconcile_raises_when_custom_backend_and_nothing_requested():
    saved = EmbeddingMeta(backend="custom", model=None)
    with pytest.raises(ValueError, match="custom"):
        reconcile_embedding_model(None, saved, "c")


def test_reconcile_accepts_matching_custom_instance():
    fake = _FakeEmbeddings()
    saved = EmbeddingMeta(backend="custom", model=None)
    assert reconcile_embedding_model(fake, saved, "c") is fake
