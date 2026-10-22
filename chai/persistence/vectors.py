"""``VectorStore``: embeddings in SQLite with cosine search, for ``VectorIndexer``/``VectorRetriever``."""

import os
import sqlite3

import ujson as json


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
