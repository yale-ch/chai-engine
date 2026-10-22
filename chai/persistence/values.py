"""Value helpers every persistence format shares: JSON-safe values, JSON text, ISO-8601 timestamps."""

from datetime import datetime

import ujson as json

from ..result import Result


def _json_safe(value):
    """Recursively make *value* JSON-encodable: bytes become a placeholder, Results become their id.

    A Result found in a value, in metadata or in ``extra`` (``extra['corrects']``, say, naming the
    result this one corrects) is stored as its id, the way ``result_to_json`` stores an input -- so
    one result referring to another never drags the whole object graph into the row, or fails to
    encode at all.
    """
    if isinstance(value, bytes):
        return {"__bytes__": len(value)}
    if isinstance(value, Result):
        return value.id
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _isoformat_rows(records):
    """``{entity: row or [rows]}`` with its datetimes as ISO-8601 text, for JSON."""

    def isoformat(row):
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in row.items()}

    return {
        entity: isoformat(rows) if isinstance(rows, dict) else [isoformat(row) for row in rows]
        for entity, rows in records.items()
    }


def _loads(text):
    """Parse a JSON column value, passing ``None`` through and tolerating non-JSON content."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text
