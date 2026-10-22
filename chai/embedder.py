"""Embedders: components that index text by its embedding and retrieve by similarity.

``Embedder`` is the role: it turns text into vectors with ``embed`` and keeps them in a
``VectorStore`` (defined in ``chai.persistence.vectors``), a SQLite-backed store with cosine search.
``VectorIndexer`` writes its input texts (or a configured document list) into a collection, then
passes its input through unchanged; ``VectorRetriever`` treats its input as a query and returns the
top-k most similar stored texts. Together they cover the retrieval side of RAG -- wire a retriever's
output into an AI component's prompt for the generation side.

The plain components embed with ``hash_embed``, a deterministic local feature-hash embedding -- no
model, no keys, fine for tests and small corpora. Model-backed variants (``GeminiVectorIndexer``,
``OllamaVectorRetriever``, ``OpenAIVectorIndexer``, ...) are generated from the backends in
``chai.ai``, which supply ``embed``; the backend's settings (``model``, ``api_host``, ``api_key``,
...) choose the embedding model and where it runs. A collection must be indexed and searched with
the same embedding.
"""

import hashlib
import logging

from .ai import create_all_components
from .core import Component
from .persistence.vectors import VectorStore
from .result import ItemResult, ListResult, Result
from .utils import text_from_input

logger = logging.getLogger("chai")

HASH_DIM = 512


def hash_embed(texts):
    """Deterministic local embedding: L2-normalized feature hashing over word 3-grams."""
    import numpy as np

    out = []
    for text in texts:
        vec = np.zeros(HASH_DIM, dtype="float32")
        words = str(text).lower().split()
        grams = words + [" ".join(words[i : i + 3]) for i in range(max(0, len(words) - 2))]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % HASH_DIM] += 1.0 if (h >> 64) % 2 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        out.append(vec.tolist())
    return out


class Embedder(Component):
    """Abstract base for the embedding role: turn text into vectors and use them.

    Subclasses implement ``_process`` and call ``embed``, which returns ``(vectors, metadata)`` --
    one vector per text, and the call's metadata (``token_usage``, ``cost``, ``engine``, ``model``
    ... for a model; just ``engine`` for the local hash). Here ``embed`` is ``hash_embed``; an AI
    backend mixed in by ``create_all_components`` replaces it with its own, and ``MODEL_KIND`` tells
    that backend its ``model`` setting names an embedding model.

    Settings:
        - database: vector store SQLite path (default 'vectors.db')
        - collection: collection name within the store (default 'default')
    """

    MODEL_KIND = "embedding"

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.expects = "data"
        if self.settings.get("service"):
            raise ValueError(
                f"{self}: the `service` setting is gone; use a backend's component instead, "
                "e.g. embedder.GeminiVectorIndexer, with its `model` setting"
            )

    def embed(self, texts):
        return hash_embed(texts), {"engine": "hash"}

    def _store(self):
        return VectorStore(self.settings.get("database", "vectors.db"))

    def _collection(self):
        return self.settings.get("collection", "default")

    def _process(self, input):
        raise NotImplementedError()


class VectorIndexer(Embedder):
    """Embeds text into a vector store collection, then passes its input through.

    Indexes either the configured ``documents`` (once, lazily) or the input
    itself: a list-shaped input indexes every text entry, a single text input
    indexes one row. Because the input flows through unchanged, an indexer can
    sit in front of a ``VectorRetriever`` -- documents get indexed, then the
    run input continues on as the query.

    Settings:
        - documents: optional list of texts to index instead of the input
        - database / collection: see Embedder
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self._documents_indexed = False

    def _texts_from(self, input):
        value = input.value if isinstance(input, Result) else input
        if type(value) is list:
            texts = []
            for v in value:
                if isinstance(v, Result):
                    v = v.value
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", "replace")
                if isinstance(v, str) and v.strip():
                    texts.append(v)
            return texts
        text = text_from_input(input)
        return [text] if text.strip() else []

    def _index(self, texts):
        vectors, call = self.embed(texts)
        count = self._store().add(self._collection(), texts, vectors)
        # the input passes on unchanged, so no result of this step carries the embedding call's
        # usage: report it to the run's ledger here
        if self.workflow is not None and self.workflow is not self:
            self.workflow.record_usage(self, call)
        return count

    def _process(self, input):
        docs = self.settings.get("documents")
        if docs:
            if not self._documents_indexed:
                count = self._index(list(docs))
                self._documents_indexed = True
                logger.info(f"{self} indexed {count} configured documents")
        else:
            texts = self._texts_from(input)
            if texts:
                self._index(texts)
        # pass-through: indexing is a side effect
        if isinstance(input, Result):
            return input
        return ItemResult(input, metadata={"type": "TEXT"}, processor=self)


class VectorRetriever(Embedder):
    """Treats its input text as a query and returns the most similar stored texts.

    Output is a ListResult of ItemResults (the matched texts), each with
    ``score``, ``match_id`` and the stored metadata in its metadata -- ready to
    iterate, gate on score, or feed into an AI component's prompt as context.
    The list's own metadata has the query and the embedding call's metadata.

    Settings:
        - top_k: how many matches to return (default 5)
        - min_score: drop matches below this cosine similarity (default none)
        - database / collection: see Embedder
    """

    def _process(self, input):
        query = text_from_input(input)
        vectors, call = self.embed([query])
        hits = self._store().search(self._collection(), vectors[0], top_k=int(self.settings.get("top_k", 5)))
        min_score = self.settings.get("min_score")
        if min_score is not None:
            hits = [h for h in hits if h["score"] >= float(min_score)]
        out = ListResult([], input=input, processor=self, metadata={"query": query, **call})
        for h in hits:
            out.append(
                ItemResult(
                    h["text"],
                    metadata={"type": "TEXT", "score": h["score"], "match_id": h["id"], **(h["metadata"] or {})},
                    input=input,
                    processor=self,
                )
            )
        return out


globals().update(create_all_components(VectorIndexer, delegate="embed"))
globals().update(create_all_components(VectorRetriever, delegate="embed"))
