"""chromadb is an optional dependency (pip install "text2cypher-composer[rag]"), imported lazily
only where RAGDataset actually needs it -- these tests simulate it being absent to confirm
importing the package, and using non-RAG components, never requires it."""
import builtins
import sys

import pytest


def _hide_chromadb(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("simulated: chromadb not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in list(sys.modules):
        if mod == "chromadb" or mod.startswith("chromadb."):
            monkeypatch.delitem(sys.modules, mod)


def test_package_imports_without_chromadb(monkeypatch):
    _hide_chromadb(monkeypatch)
    for mod in list(sys.modules):
        if mod.startswith("text2cypher_composer"):
            monkeypatch.delitem(sys.modules, mod)

    import text2cypher_composer as t2c

    assert t2c.run is not None
    assert t2c.RAGDataset is not None
    assert t2c.CascadeStrategy is not None


def test_rag_dataset_collection_raises_a_helpful_error_without_chromadb(monkeypatch):
    _hide_chromadb(monkeypatch)
    for mod in list(sys.modules):
        if mod.startswith("text2cypher_composer"):
            monkeypatch.delitem(sys.modules, mod)

    from text2cypher_composer import RAGDataset

    dataset = RAGDataset(chroma_path="/tmp/does-not-matter", cypher_dir="/tmp/does-not-matter")

    with pytest.raises(ImportError, match="rag"):
        dataset.collection
