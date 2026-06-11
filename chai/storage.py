"""Storage components: persist Results to the filesystem or a database.

Storage steps are pass-through: they write the input Result somewhere and return it unchanged, so
they can be inserted anywhere in a pipeline (typically as ``next_steps``) without affecting the data
flow.

Besides the components, this module exposes plain helper functions (``list_results``, ``get_result``,
``save_correction``, ``list_processors``) that a viewer app can import to browse a ``SqliteStorage``
database and record human corrections alongside the original values. The helpers open a fresh
connection per call, so they are safe to use from multi-threaded servers (e.g. Flask).
"""

import os
import sqlite3

import ujson as json

from .core import Component, Result
from .result import FileItemResult


def _json_safe(value):
    """Recursively replace bytes with a ``{"__bytes__": <len>}`` placeholder so *value* JSON-encodes."""
    if isinstance(value, bytes):
        return {"__bytes__": len(value)}
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def result_to_json(result: Result):
    """Bytes-safe, JSON-serializable form of *result* (non-recursive ``Result.to_json``).

    ``FileItemResult`` values are stored as their ``file_name`` instead of the file content -- this
    also avoids triggering the lazy on-disk read. Any bytes left in values or metadata are replaced
    with a ``{"__bytes__": <len>}`` placeholder.
    """
    if isinstance(result, FileItemResult):
        # Mirror Result.to_json(recurse=False) but never touch the lazy ``value`` property
        js = {
            "id": result.id,
            "type": result.__class__.__name__,
            "workflowId": result.workflow.id
            if result.workflow
            else (result.processor.workflow.id if result.processor else None),
            "processorId": result.processor.id if result.processor else None,
            "metadata": result.metadata,
            "extraInfo": result.extra,
            "input": result.input.id if isinstance(result.input, Result) else result.input,
            "value": result.file_name,
        }
    else:
        js = result.to_json(recurse=False)
    return _json_safe(js)


class Storage(Component):
    """Take the input and store it somewhere.

    Base class for persistence components: ``build_json`` renders the input Result via
    ``result_to_json`` (non-recursive, bytes-safe, child Results referenced by id), and subclasses
    override ``_process`` to write that representation out. ``_process`` always returns the input
    unchanged so downstream steps see the original Result.
    """

    def build_json(self, input: Result):
        """Serialize *input* for persistence (non-recursive, bytes-safe ``Result.to_json``)."""
        return result_to_json(input)

    def _process(self, input: Result) -> Result:
        # Do persistence here
        return input


class FileSystemStorage(Storage):
    """Writes each Result as a JSON file in a pairtree under a per-processor directory.

    The path is ``<directory>/<processor id>/<id[0:2]>/<id[2:4]>/<result id>.json``; an existing file
    gets a ``.1`` version suffix. ``FileItemResult`` values are stored as their ``file_name`` and any
    bytes are replaced with a placeholder, so the output is always valid JSON. Returns the input
    unchanged.

    Settings:
        - directory: root directory for stored results (default 'results')
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "directory" not in self.settings:
            self.settings["directory"] = "results"
        dn = self.settings["directory"]
        if not os.path.exists(dn):
            os.makedirs(dn, exist_ok=True)

    def _process(self, input: Result) -> Result:
        """Write the result to the filesystem according to the processor id"""
        if input.processor is not None:
            # Store in directory per processor
            base = os.path.join(self.settings["directory"], input.processor.id)
        else:
            base = os.path.join(self.settings["directory"], "base")
        if not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
        # Now make a pair-tree
        pair = os.path.join(base, input.id[0:2], input.id[2:4])
        if not os.path.exists(pair):
            os.makedirs(pair, exist_ok=True)
        fn = os.path.join(pair, f"{input.id}.json")
        if os.path.exists(fn):
            # make a new version
            vn = 1  # FIXME: Make this the count of files with this name
            fn = f"{fn}.{vn}"
        js = self.build_json(input)
        with open(fn, "w") as fh:
            json.dump(js, fh)

        return input


class PostgresStorage(Storage):
    """Placeholder for PostgreSQL persistence -- not implemented yet.

    Currently behaves like the base ``Storage`` (a pass-through that stores nothing).
    """

    pass


def _ensure_schema(conn):
    """Create the ``results``/``derivatives`` tables and indexes if missing; add new columns to old DBs."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            processor_id TEXT,
            workflow_id TEXT,
            value_json TEXT,
            metadata_json TEXT,
            extra_json TEXT,
            corrected_value_json TEXT,
            corrected_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS derivatives (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            component_id TEXT,
            result_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Upgrade pre-correction databases in place; the ALTERs fail harmlessly once the columns exist
    for ddl in (
        "ALTER TABLE results ADD COLUMN corrected_value_json TEXT",
        "ALTER TABLE results ADD COLUMN corrected_at TIMESTAMP",
    ):
        try:
            cursor.execute(ddl)
        except sqlite3.OperationalError:
            pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_processor ON results(processor_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_workflow ON results(workflow_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_derivatives_source ON derivatives(source_id)")
    conn.commit()


def _connect(database):
    """Open a fresh connection to *database* with the schema ensured and dict-style row access."""
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _loads(text):
    """Parse a JSON column value, passing ``None`` through and tolerating non-JSON legacy content."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def _row_to_dict(row):
    """Convert a ``results`` row into the dict shape returned by the viewer helpers."""
    return {
        "id": row["id"],
        "processor_id": row["processor_id"],
        "workflow_id": row["workflow_id"],
        "value": _loads(row["value_json"]),
        "metadata": _loads(row["metadata_json"]),
        "extra": _loads(row["extra_json"]),
        "corrected": row["corrected_value_json"] is not None,
        "corrected_value": _loads(row["corrected_value_json"]),
        "corrected_at": row["corrected_at"],
        "created_at": row["created_at"],
    }


def ensure_database(database):
    """Create *database* (and its parent directory) with the chai schema if missing; returns the path.

    Lets an app pre-build its results database so storage viewers work before the first run.
    """
    parent = os.path.dirname(database)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _connect(database).close()
    return database


def store_json_result(database, result_id, value, processor_id=None, workflow_id=None, metadata=None):
    """Insert/refresh one result row from already-serialized JSON values.

    The viewer-side counterpart of ``SqliteStorage._process`` for callers that hold a run's
    serialized output (dicts) rather than live ``Result`` objects -- e.g. a front-end persisting
    run results into its app-local database. Re-storing keeps any human correction and the
    original ``created_at``.
    """
    conn = _connect(database)
    try:
        conn.execute(
            """
            INSERT INTO results (id, processor_id, workflow_id, value_json, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                processor_id = excluded.processor_id,
                workflow_id = excluded.workflow_id,
                value_json = excluded.value_json,
                metadata_json = excluded.metadata_json
            """,
            (
                result_id,
                processor_id,
                workflow_id,
                json.dumps(_json_safe(value)),
                json.dumps(_json_safe(metadata)) if metadata else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return True


def list_results(database, processor_id=None, workflow_id=None, limit=100, offset=0):
    """Return stored results as a list of dicts, newest first.

    Each dict has ``id``, ``processor_id``, ``workflow_id``, ``value`` (parsed JSON), ``metadata``,
    ``extra``, ``corrected`` (bool), ``corrected_value``, ``corrected_at`` and ``created_at``.
    Optionally filter by *processor_id* and/or *workflow_id*; page with *limit*/*offset*.
    """
    sql = "SELECT * FROM results"
    clauses, params = [], []
    if processor_id is not None:
        clauses.append("processor_id = ?")
        params.append(processor_id)
    if workflow_id is not None:
        clauses.append("workflow_id = ?")
        params.append(workflow_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    conn = _connect(database)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_result(database, result_id):
    """Return a single stored result as a dict (see ``list_results``), or ``None`` if not found."""
    conn = _connect(database)
    try:
        row = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None
    finally:
        conn.close()


def save_correction(database, result_id, corrected_value):
    """Store a human-corrected value for *result_id* alongside the original.

    The value is JSON-encoded into ``corrected_value_json`` (bytes-safe) and ``corrected_at`` is set
    to the current time. Returns ``True`` if a row was updated, ``False`` if the id is unknown.
    """
    conn = _connect(database)
    try:
        cursor = conn.execute(
            "UPDATE results SET corrected_value_json = ?, corrected_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(_json_safe(corrected_value)), result_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def list_processors(database):
    """Return the distinct ``processor_id`` values with their row counts.

    Each entry is ``{"processor_id": ..., "count": ...}``, ordered by processor id.
    """
    conn = _connect(database)
    try:
        rows = conn.execute(
            "SELECT processor_id, COUNT(*) AS count FROM results GROUP BY processor_id ORDER BY processor_id"
        ).fetchall()
        return [{"processor_id": row["processor_id"], "count": row["count"]} for row in rows]
    finally:
        conn.close()


class SqliteStorage(Storage):
    """Store results in a SQLite database.

    Writes the input Result into a ``results`` table and each of its ``derivative_results`` into a
    ``derivatives`` table keyed by source result and component. ``value_json`` holds the full
    bytes-safe ``to_json(recurse=False)`` representation (id, type, value, metadata, provenance), so a
    viewer app can reconstruct what was produced; ``metadata_json``/``extra_json`` are kept as
    dedicated columns for querying. The nullable ``corrected_value_json``/``corrected_at`` columns
    hold human corrections written via ``save_correction``. The schema is created (and old databases
    upgraded) lazily on first use, a fresh connection is opened per operation (thread-safe for use
    from e.g. Flask), and the input is returned unchanged.

    Settings:
        - database: path of the SQLite database file (default 'results.db')
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "database" not in self.settings:
            self.settings["database"] = "results.db"

    def _process(self, input: Result) -> Result:
        """Store the result in SQLite"""
        conn = _connect(self.settings["database"])
        try:
            cursor = conn.cursor()

            # Build JSON representations (bytes-safe; FileItemResults store their file_name)
            value_json = json.dumps(self.build_json(input))
            metadata_json = json.dumps(_json_safe(input.metadata)) if input.metadata else None
            extra_json = json.dumps(_json_safe(input.extra)) if input.extra else None

            # Insert result -- an upsert (not OR REPLACE) so re-storing a result keeps any
            # human correction and the original created_at
            cursor.execute(
                """
                INSERT INTO results
                (id, processor_id, workflow_id, value_json, metadata_json, extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    processor_id = excluded.processor_id,
                    workflow_id = excluded.workflow_id,
                    value_json = excluded.value_json,
                    metadata_json = excluded.metadata_json,
                    extra_json = excluded.extra_json
            """,
                (
                    input.id,
                    input.processor.id if input.processor else None,
                    input.workflow.id if input.workflow else None,
                    value_json,
                    metadata_json,
                    extra_json,
                ),
            )

            # Store derivatives
            for component, results in input.derivative_results.items():
                for result in results:
                    result_json = json.dumps(result_to_json(result))
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO derivatives
                        (id, source_id, component_id, result_json)
                        VALUES (?, ?, ?, ?)
                    """,
                        (result.id, input.id, component.id, result_json),
                    )

            conn.commit()
        finally:
            conn.close()
        return input


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
