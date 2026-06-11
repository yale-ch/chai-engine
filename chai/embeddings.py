"""Embeddings and vector search: index text, retrieve by similarity.

``embed_texts`` turns text into vectors via a pluggable service: ``hash`` (a
deterministic local feature-hash embedding -- no model, no keys, fine for tests
and small corpora), ``gemini``, ``ollama``, or any OpenAI-compatible endpoint
(``openai`` / ``custom`` / ``lmstudio`` / ``vllm`` with ``api_host``).

``VectorStore`` is a SQLite-backed store with cosine search. The workflow
components: ``VectorIndexer`` writes its input texts (or a configured document
list) into a collection, then passes its input through unchanged;
``VectorRetriever`` treats its input as a query and returns the top-k most
similar stored texts. Together they cover the retrieval side of RAG -- wire a
retriever's output into an AI component's prompt for the generation side.
"""

import logging
import os
import sqlite3

import ujson as json

from .core import Component
from .result import ItemResult, ListResult, Result
from .utils import text_from_input

logger = logging.getLogger("chai")

HASH_DIM = 512


def _hash_embed(texts):
    """Deterministic local embedding: L2-normalized feature hashing over word 3-grams."""
    import hashlib

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


def embed_texts(texts, service="hash", model=None, api_host=None):
    """Embed *texts* (list of str) -> list of float vectors, via *service*."""
    service = (service or "hash").lower()
    texts = [str(t) for t in texts]
    if not texts:
        return []
    if service == "hash":
        return _hash_embed(texts)
    if service == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")))
        resp = client.models.embed_content(model=model or "gemini-embedding-001", contents=texts)
        return [e.values for e in resp.embeddings]
    if service == "ollama":
        import ollama

        client = ollama.Client(host=api_host) if api_host else ollama
        resp = client.embed(model=model or "nomic-embed-text", input=texts)
        return list(resp["embeddings"])
    # everything else speaks the OpenAI embeddings API
    from openai import OpenAI

    host = api_host or "api.openai.com"
    base_url = host if host.startswith("http") else f"http://{host}"
    if "api.openai.com" in base_url:
        base_url = "https://api.openai.com/v1"
    elif not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    resp = client.embeddings.create(model=model or "text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


class VectorStore:
    """SQLite-backed vector collection with brute-force cosine search.

    Right-sized for workflow corpora (thousands to low hundreds of thousands of
    rows); swap in a dedicated vector database beyond that.
    """

    def __init__(self, database):
        parent = os.path.dirname(database)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.database = database
        with sqlite3.connect(self.database) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    collection TEXT,
                    text TEXT,
                    vector_json TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection)")

    def add(self, collection, texts, vectors, metadatas=None, ids=None):
        import hashlib

        rows = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            rid = (ids[i] if ids else None) or hashlib.sha256(f"{collection}:{text}".encode()).hexdigest()[:32]
            md = (metadatas[i] if metadatas else None) or {}
            rows.append((rid, collection, text, json.dumps(list(vec)), json.dumps(md)))
        with sqlite3.connect(self.database) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (id, collection, text, vector_json, metadata_json) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def search(self, collection, query_vector, top_k=5):
        import numpy as np

        with sqlite3.connect(self.database) as conn:
            rows = conn.execute(
                "SELECT id, text, vector_json, metadata_json FROM vectors WHERE collection = ?", (collection,)
            ).fetchall()
        if not rows:
            return []
        matrix = np.array([json.loads(r[2]) for r in rows], dtype="float32")
        q = np.array(query_vector, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1.0)
        norms[norms == 0] = 1.0
        scores = matrix @ q / norms
        order = scores.argsort()[::-1][:top_k]
        return [
            {"id": rows[i][0], "text": rows[i][1], "score": float(scores[i]), "metadata": json.loads(rows[i][3])}
            for i in order
        ]

    def count(self, collection):
        with sqlite3.connect(self.database) as conn:
            return conn.execute("SELECT COUNT(*) FROM vectors WHERE collection = ?", (collection,)).fetchone()[0]


class Embedder(Component):
    """Abstract base for the embedding role: turn text into vectors or use them.

    Concrete components: ``VectorIndexer`` (write to a collection),
    ``VectorRetriever`` (similarity search). All share the embedding settings:

    Settings:
        - service: hash (local, default) | gemini | ollama | openai | custom (default hash)
        - model: embedding model id for the chosen service
        - api_host: server host:port for local/custom services
        - database: vector store SQLite path (default 'vectors.db')
        - collection: collection name within the store (default 'default')
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.expects = "data"

    def _embed(self, texts):
        return embed_texts(
            texts,
            service=self.settings.get("service", "hash"),
            model=self.settings.get("model"),
            api_host=self.settings.get("api_host"),
        )

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
        - service / model / api_host / database / collection: see Embedder
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

    def _process(self, input):
        docs = self.settings.get("documents")
        if docs:
            if not self._documents_indexed:
                count = self._store().add(self._collection(), list(docs), self._embed(list(docs)))
                self._documents_indexed = True
                logger.info(f"{self} indexed {count} configured documents")
        else:
            texts = self._texts_from(input)
            if texts:
                self._store().add(self._collection(), texts, self._embed(texts))
        # pass-through: indexing is a side effect
        if isinstance(input, Result):
            return input
        return ItemResult(input, metadata={"type": "TEXT"}, processor=self)


class VectorRetriever(Embedder):
    """Treats its input text as a query and returns the most similar stored texts.

    Output is a ListResult of ItemResults (the matched texts), each with
    ``score``, ``match_id`` and the stored metadata in its metadata -- ready to
    iterate, gate on score, or feed into an AI component's prompt as context.

    Settings:
        - top_k: how many matches to return (default 5)
        - min_score: drop matches below this cosine similarity (default none)
        - service / model / api_host / database / collection: see Embedder
    """

    def _process(self, input):
        query = text_from_input(input)
        vec = self._embed([query])[0]
        hits = self._store().search(self._collection(), vec, top_k=int(self.settings.get("top_k", 5)))
        min_score = self.settings.get("min_score")
        if min_score is not None:
            hits = [h for h in hits if h["score"] >= float(min_score)]
        out = ListResult([], input=input, processor=self, metadata={"query": query})
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
