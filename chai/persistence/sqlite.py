"""SQLite: the CHAI-TEA tables in a local database file, and the helpers a viewer app browses them with.

The helpers (``list_results``, ``get_result``, ``save_correction``, ``list_processors``) open a fresh
connection per call, so they are safe to use from multi-threaded servers (e.g. Flask).
"""

import os
import sqlite3

from .records import as_uuid
from .rows import _correction_metadata, correction_row
from .schema import column_types, sqlite_statements, table_names
from .sql import write_rows
from .values import _loads


def _ensure_schema(conn, tables=None):
    """Create the CHAI-TEA tables in a SQLite database if they are missing; returns their names.

    The same tables as in PostgreSQL (``chai.persistence.schema.sqlite_statements``). A table of one of the names
    with other columns -- the pre-CHAI-TEA ``results`` table, say -- raises rather than being written
    into: there is no migration from earlier layouts, so delete the database and re-run.
    """
    names = table_names(tables)
    for entity, table in names.items():
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        expected = set(column_types(entity))
        if columns and not expected <= columns:
            raise RuntimeError(
                f"Table {table} already exists and is not the CHAI-TEA {entity} table "
                f"(no {', '.join(sorted(expected - columns)[:3])} column). Delete the database and re-run."
            )
    for statement in sqlite_statements(tables):
        conn.execute(statement)
    conn.commit()
    return names


def _connect(database, tables=None):
    """A fresh connection to *database*: schema ensured, foreign keys enforced, dict-style rows."""
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn, tables)
    return conn


def _row_to_dict(row, entity="RESULT"):
    """A stored row as a dict of its columns, JSON parsed and booleans as booleans.

    A RESULT row from the viewer helpers' query also has ``correction_count`` and ``corrected``,
    saying how many rows correct it.
    """
    js = {}
    for column, sql_type in column_types(entity).items():
        value = row[column]
        if sql_type == "JSONB":
            value = _loads(value)
        elif sql_type == "BOOLEAN" and value is not None:
            value = bool(value)
        js[column] = value
    if "correction_count" in row.keys():
        js["correction_count"] = row["correction_count"]
        js["corrected"] = row["correction_count"] > 0
    return js


def ensure_database(database, tables=None):
    """Create *database* (and its parent directory) with the CHAI-TEA tables if missing; returns the path.

    Lets an app pre-build its results database so storage viewers work before the first run.
    """
    parent = os.path.dirname(database)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _connect(database, tables).close()
    return database


def store_rows(database, records, tables=None):
    """Upsert ``{entity: row or [rows]}`` into the CHAI-TEA tables of a SQLite database; returns the count.

    The viewer-side counterpart of ``SqliteStorage`` for callers that hold a run's rows (from
    ``workflow.records()``, ``result_row``, a JSON-Lines file) rather than live ``Result`` objects --
    e.g. a front-end persisting run results into its app-local database. Storing a row again keeps its
    ``created`` and leaves any correction rows pointing at it untouched.
    """
    conn = _connect(database, tables)
    try:
        written = write_rows(conn.cursor(), "sqlite", table_names(tables), records)
        conn.commit()
        return written
    finally:
        conn.close()


def _results_select(names):
    """Every RESULT column plus the number of rows correcting each one, for the viewer helpers."""
    table = names["RESULT"]
    return f"SELECT r.*, (SELECT COUNT(*) FROM {table} c WHERE c.previous_id = r.id) AS correction_count FROM {table} r"


def list_results(
    database, process_id=None, workflow_run_id=None, input_hash=None, previous_id=None, limit=100, offset=0
):
    """Return stored RESULT rows as a list of dicts, newest first.

    Each dict has the RESULT columns (``value``, ``metadata``, ``extra_data`` parsed from JSON) and
    the ``correction_count``/``corrected`` pair saying how many rows correct this one. Corrections are
    rows like any other, so they are listed too; pass *previous_id* to list only the corrections of
    one result, or *input_hash* for every entry generated from the same input content. Optionally
    filter by *process_id* and/or *workflow_run_id*; page with *limit*/*offset*.
    """
    names = table_names()
    sql = _results_select(names)
    clauses, params = [], []
    for column, value in (
        ("process_id", process_id),
        ("workflow_run_id", workflow_run_id),
        ("input_hash", input_hash),
        ("previous_id", previous_id),
    ):
        if value is not None:
            clauses.append(f"r.{column} = ?")
            params.append(value)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY r.created DESC, r.id LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    conn = _connect(database)
    try:
        return [_row_to_dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def get_result(database, result_id, with_corrections=True):
    """Return one stored RESULT row as a dict (see ``list_results``), or ``None`` if not found.

    Unless *with_corrections* is false the dict also has ``corrections``: the rows that correct this
    one, oldest first, each carrying the metadata of whoever made it.
    """
    names = table_names()
    conn = _connect(database)
    try:
        row = conn.execute(f"{_results_select(names)} WHERE r.id = ?", (as_uuid(result_id),)).fetchone()
        if row is None:
            return None
        js = _row_to_dict(row)
        if with_corrections:
            corrections = conn.execute(
                f"{_results_select(names)} WHERE r.previous_id = ? ORDER BY r.created, r.id", (js["id"],)
            ).fetchall()
            js["corrections"] = [_row_to_dict(correction) for correction in corrections]
        return js
    finally:
        conn.close()


def save_correction(
    database, result_id, corrected_value, agent=None, metadata=None, process_id=None, editor_user_id=None
):
    """Record a correction of *result_id* as a new RESULT row pointing back at it; returns the new row's id.

    The original is never touched: the correction is stored as its own entry whose ``previous_id`` is
    the corrected row, so a result can be corrected more than once and each correction keeps its own
    metadata -- *agent* (the person or model that made it, recorded as ``metadata['agent']``) and
    anything else in *metadata*, e.g. who reviewed it, why, or which model and prompt produced it.
    See ``correction_row``. Returns ``None`` if *result_id* is not in the database.
    """
    original = get_result(database, result_id, with_corrections=False)
    if original is None:
        return None
    row = correction_row(original, corrected_value, _correction_metadata(agent, metadata), process_id, editor_user_id)
    store_rows(database, {"RESULT": row})
    return row["id"]


def list_processors(database):
    """Return the distinct ``process_id`` values with their row counts.

    Each entry is ``{"process_id": ..., "count": ...}``, ordered by process id.
    """
    table = table_names()["RESULT"]
    conn = _connect(database)
    try:
        rows = conn.execute(
            f"SELECT process_id, COUNT(*) AS count FROM {table} GROUP BY process_id ORDER BY process_id"
        ).fetchall()
        return [{"process_id": row["process_id"], "count": row["count"]} for row in rows]
    finally:
        conn.close()
