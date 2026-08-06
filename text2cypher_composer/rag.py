from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from langchain_core.embeddings import Embeddings

from .embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingMeta,
    EmbeddingModelLike,
    read_embedding_meta,
    reconcile_embedding_model,
    resolve_embedder,
    write_embedding_meta,
)


@dataclass
class RAGDataset:
    """The vector store (and paired golden Cypher/output files) used for RAG retrieval.

    Mirrors the bio2C benchmark layout: a Chroma collection embedding each
    NL question, plus sibling directories holding the corresponding golden
    Cypher query (`cypher_dir`) and, for the "+O" techniques, the Neo4j
    output that query produced (`output_dir`). Each retrieved document's
    `source` metadata (e.g. "3hop/question_3.txt") is mapped to
    "3hop/cypher_3.txt" / "3hop/output_3.txt" respectively.
    """

    chroma_path: Union[str, Path]
    cypher_dir: Union[str, Path]
    output_dir: Optional[Union[str, Path]] = None
    collection_name: str = "my_collection"
    embedding_model: Optional[EmbeddingModelLike] = None
    n_results: int = 3

    def __post_init__(self) -> None:
        self.chroma_path = Path(self.chroma_path)
        self.cypher_dir = Path(self.cypher_dir)
        self.output_dir = Path(self.output_dir) if self.output_dir else None
        self._collection = None
        self._embedder: Optional[Embeddings] = None

    @classmethod
    def from_root(cls, root: Union[str, Path], **overrides: Any) -> "RAGDataset":
        """Build a RAGDataset from a bio2C-style benchmark root directory.

        Assumes `root` contains `chroma_db/`, `CypherQueries/`, and
        `Neo4jOutputs/` subdirectories, as in `bio2C/evaluating_text2cypher`.
        """
        root = Path(root)
        defaults: Dict[str, Any] = {
            "chroma_path": root / "chroma_db",
            "cypher_dir": root / "CypherQueries",
            "output_dir": root / "Neo4jOutputs",
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def index_from_root(
        cls,
        root: Union[str, Path],
        embedding_model: Optional[EmbeddingModelLike] = None,
        collection_name: str = "my_collection",
        batch_size: int = 512,
        rebuild: bool = False,
        **overrides: Any,
    ) -> "RAGDataset":
        """Load an existing Chroma collection, or embed `root/NLquestions/` into a new one.

        Mirrors the indexing step from the bio2C notebooks: every `.txt` file
        under `root/NLquestions/<group>/` is embedded with `embedding_model`
        and stored in a persistent Chroma collection at `root/chroma_db`,
        with `source` metadata `"<group>/<file>.txt"` — the same convention
        `retrieve_examples` uses to map a retrieved example back to its
        golden Cypher/output files.

        `embedding_model` accepts either an OpenAI embedding model id (e.g.
        `"text-embedding-3-large"`, the default), a HuggingFace/
        sentence-transformers model id (e.g.
        `"sentence-transformers/all-mpnet-base-v2"` — a `"/"` in the string
        is what selects this backend), or an already-built LangChain
        `Embeddings` instance for anything else.

        Whichever one is used to build the collection is recorded alongside
        it (see `text2cypher_composer.embeddings.write_embedding_meta`), so a
        later call — to retrieve, or just to reuse the collection — never
        needs `embedding_model` passed again: omit it and the recorded one is
        used automatically. Passing a *different* `embedding_model` than what
        a collection was actually indexed with raises `ValueError` rather
        than silently corrupting retrieval (a query embedded with a
        different model than its documents produces meaningless similarity
        scores).

        If a collection named `collection_name` already exists at
        `root/chroma_db`, it is reused as-is by default — no re-embedding, no
        API calls — so calling this again (e.g. in a later session) is cheap.
        Pass `rebuild=True` to delete it and re-embed from scratch instead,
        e.g. after adding new examples under `root/NLquestions/`.

        Typically run once after `build_rag_example_files()` has populated
        `root/NLquestions/`.
        """
        root = Path(root)
        chroma_path = root / "chroma_db"
        client = chromadb.PersistentClient(path=str(chroma_path))

        collection = None
        if not rebuild:
            try:
                collection = client.get_collection(name=collection_name)
            except Exception:
                collection = None

        if collection is not None:
            saved_meta = read_embedding_meta(chroma_path, collection_name)
            resolved_embedding_model = reconcile_embedding_model(embedding_model, saved_meta, collection_name)
            if saved_meta is None:
                # A legacy collection, indexed before this metadata existed — start
                # recording from now on so future reuse doesn't need re-specifying it.
                write_embedding_meta(chroma_path, collection_name, resolved_embedding_model)
        else:
            resolved_embedding_model = embedding_model if embedding_model is not None else DEFAULT_OPENAI_EMBEDDING_MODEL

            question_root = root / "NLquestions"
            if not question_root.exists():
                raise FileNotFoundError(
                    f"{question_root} does not exist. Build it first, e.g. with "
                    "build_rag_example_files()."
                )

            sources: List[str] = []
            texts: List[str] = []
            for group_dir in sorted(p for p in question_root.iterdir() if p.is_dir()):
                for txt_file in sorted(group_dir.glob("*.txt")):
                    sources.append(f"{group_dir.name}/{txt_file.name}")
                    texts.append(txt_file.read_text(encoding="utf-8"))

            if not texts:
                raise ValueError(f"No .txt files found under {question_root}.")

            embedder = resolve_embedder(resolved_embedding_model)

            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            collection = client.create_collection(name=collection_name)

            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_sources = sources[start : start + batch_size]
                batch_embeddings = embedder.embed_documents(batch_texts)
                collection.add(
                    documents=batch_texts,
                    ids=batch_sources,
                    metadatas=[{"source": s} for s in batch_sources],
                    embeddings=batch_embeddings,
                )

            write_embedding_meta(chroma_path, collection_name, resolved_embedding_model)

        dataset = cls.from_root(
            root,
            collection_name=collection_name,
            embedding_model=resolved_embedding_model,
            **overrides,
        )
        dataset._collection = collection
        return dataset

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = client.get_collection(name=self.collection_name)
        return self._collection

    @property
    def embedder(self) -> Embeddings:
        if self._embedder is None:
            embedding_model = self.embedding_model
            if embedding_model is None:
                saved_meta = read_embedding_meta(self.chroma_path, self.collection_name)
                embedding_model = reconcile_embedding_model(None, saved_meta, self.collection_name)
                self.embedding_model = embedding_model  # cache what was actually resolved
            self._embedder = resolve_embedder(embedding_model)
        return self._embedder

    def recorded_embedding(self) -> Optional[EmbeddingMeta]:
        """What `embedding_model` this dataset's collection was actually indexed with, if recorded."""
        return read_embedding_meta(self.chroma_path, self.collection_name)

    def _cypher_file(self, source: str) -> Path:
        rel = Path(source)
        return self.cypher_dir / rel.parent / rel.name.replace("question", "cypher")

    def _output_file(self, source: str) -> Path:
        if self.output_dir is None:
            raise ValueError(
                "RAGDataset.output_dir is required for output-augmented (+O) techniques."
            )
        rel = Path(source)
        return self.output_dir / rel.parent / rel.name.replace("question", "output")

    def retrieve_examples(self, question: str, with_output: bool) -> Dict[str, Any]:
        """Retrieve the top-`n_results` nearest examples for `question`.

        Returns a dict with `examples_text` (ready to interpolate into a
        prompt), plus `example_ids`/`example_distances` for traceability.
        """
        query_embedding = self.embedder.embed_query(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results.get("distances", [[None] * len(docs)])[0]

        parts: List[str] = []
        example_ids: List[str] = []
        example_distances: List[float] = []

        for doc, meta, dist in zip(docs, metas, dists):
            source = meta["source"]
            example_ids.append(source)
            example_distances.append(dist)

            cypher_file = self._cypher_file(source)
            if not cypher_file.exists():
                raise FileNotFoundError(f"Cypher file not found for source={source}: {cypher_file}")
            cypher_query = cypher_file.read_text(encoding="utf-8").strip()

            parts.append(f"[Query natural language] {doc}\n")
            parts.append(f"[Query Cypher] {cypher_query}\n")

            if with_output:
                output_file = self._output_file(source)
                if not output_file.exists():
                    raise FileNotFoundError(
                        f"Neo4j output file not found for source={source}: {output_file}"
                    )
                parts.append(f"[Output Neo4j] {output_file.read_text(encoding='utf-8').strip()}\n")

            parts.append("\n")

        return {
            "examples_text": "".join(parts),
            "example_ids": example_ids,
            "example_distances": example_distances,
        }
