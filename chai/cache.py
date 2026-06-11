"""Run cache: skip re-processing items that already completed in an earlier run.

A small SQLite table keyed by (config_hash, item_key): the ``Iterator`` (with a
``cache`` setting) stores each item's serialized output after processing and
replays it on the next run, so a long corpus run that died at item 7,000
resumes instead of restarting. The config hash covers the iterator's child-step
configuration, so editing a prompt or model invalidates the cache naturally.
"""

import hashlib
import os
import sqlite3

import ujson as json


def config_hash(config) -> str:
    """Stable hash of a component configuration (dict/list tree)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]


def item_key(item) -> str:
    """Stable identity for one work item: a file's path, or a hash of the value."""
    file_name = getattr(item, "file_name", "")
    if file_name:
        return f"file:{file_name}"
    value = getattr(item, "value", item)
    try:
        blob = json.dumps(value, sort_keys=True)
    except Exception:
        blob = repr(value)
    return "sha:" + hashlib.sha256(blob.encode()).hexdigest()[:32]


class RunCache:
    """SQLite-backed (config, item) -> serialized output store. Thread-safe via
    per-operation connections."""

    def __init__(self, database):
        parent = os.path.dirname(database)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.database = database
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_cache (
                    config_hash TEXT,
                    item_key TEXT,
                    payload_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (config_hash, item_key)
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.database)

    def get(self, cfg_hash, key):
        """Return the cached payload (parsed JSON) or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM run_cache WHERE config_hash = ? AND item_key = ?",
                (cfg_hash, key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, cfg_hash, key, payload):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_cache (config_hash, item_key, payload_json) VALUES (?, ?, ?)",
                (cfg_hash, key, json.dumps(payload)),
            )

    def clear(self, cfg_hash=None):
        with self._connect() as conn:
            if cfg_hash:
                conn.execute("DELETE FROM run_cache WHERE config_hash = ?", (cfg_hash,))
            else:
                conn.execute("DELETE FROM run_cache")
