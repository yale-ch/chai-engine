"""Writing CHAI-TEA rows into a database -- PostgreSQL or SQLite, the same tables either way.

``write_rows`` is the one upsert writer for both dialects.
"""

from datetime import timezone

import ujson as json

from .records import as_datetime, as_uuid
from .schema import TABLE_ORDER, column_types
from .values import _json_safe


#: Nullable references a row may make to a row the database does not hold -- a correction of a result
#: that was never stored here, an editor with no USER row yet, a workflow version whose predecessor
#: lives elsewhere. They are written as NULL rather than failing the foreign key and the whole write.
_SOFT_REFERENCES = {
    ("RESULT", "previous_id"): "RESULT",
    ("RESULT", "editor_user_id"): "USER",
    ("WORKFLOW", "previous_id"): "WORKFLOW",
}

#: PostgreSQL casts for parameters whose column type text would not convert to implicitly.
_PG_CASTS = {"UUID": "::uuid", "JSONB": "::jsonb", "TIMESTAMPTZ": "::timestamptz"}


def _db_value(value, sql_type, dialect):
    """*value* as the parameter a *sql_type* column takes in *dialect* ('postgres' or 'sqlite')."""
    if value is None:
        return None
    if sql_type == "JSONB":
        return json.dumps(_json_safe(value), ensure_ascii=False)
    if sql_type == "TIMESTAMPTZ":
        value = as_datetime(value)
        if dialect == "sqlite" and value is not None:
            return value.astimezone(timezone.utc).isoformat()
        return value
    if sql_type == "BOOLEAN":
        return bool(value) if dialect == "postgres" else int(bool(value))
    if sql_type == "UUID":
        return as_uuid(value)
    return value


def _upsert_statement(entity, names, dialect):
    """``INSERT ... ON CONFLICT (id) DO UPDATE`` for every column of *entity*'s table.

    An update keeps the row's ``created`` (and, in PostgreSQL, the schema's trigger sets
    ``modified``), so storing a row again refreshes it without rewriting its history.
    """
    types = column_types(entity)
    holder = "%s" if dialect == "postgres" else "?"
    values = []
    for column, sql_type in types.items():
        expr = holder + (_PG_CASTS.get(sql_type, "") if dialect == "postgres" else "")
        target = _SOFT_REFERENCES.get((entity, column))
        if target:
            expr = f"(SELECT id FROM {names[target]} WHERE id = {expr})"
        values.append(expr)
    updates = ", ".join(f"{column} = excluded.{column}" for column in types if column not in ("id", "created"))
    return (
        f"INSERT INTO {names[entity]} ({', '.join(types)}) VALUES ({', '.join(values)}) "
        f"ON CONFLICT (id) DO UPDATE SET {updates}"
    )


def write_rows(cursor, dialect, names, records):
    """Upsert CHAI-TEA rows: *records* is ``{entity: row or [rows]}``, written parents first.

    Each row must have exactly its table's columns (``Record.to_row``, ``Workflow.to_row``,
    ``result_row``). Returns the number of rows written. The caller commits.
    """
    written = 0
    for entity in TABLE_ORDER:
        rows = records.get(entity)
        if not rows:
            continue
        if isinstance(rows, dict):
            rows = [rows]
        types = column_types(entity)
        sql = _upsert_statement(entity, names, dialect)
        for row in rows:
            if set(row) != set(types):
                raise ValueError(
                    f"A {entity} row must have exactly the columns {', '.join(types)}; got {', '.join(row)}"
                )
            cursor.execute(sql, [_db_value(row[column], sql_type, dialect) for column, sql_type in types.items()])
            written += 1
    return written
