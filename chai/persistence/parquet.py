"""Parquet files: ``ParquetRecordWriter``, packing a JSON-Lines run into Parquet, and reading one back.

Needs ``pyarrow``, imported when a file is first written or read.
"""

import logging
import os
import threading
from datetime import datetime

import ujson as json

from .jsonl import read_run_records, run_records_file
from .records import as_datetime
from .rows import RESULT_COLUMNS, is_result_row
from .values import _isoformat_rows, _json_safe

logger = logging.getLogger("chai")


def _arrow_type(pa, kind):
    """The pyarrow type for one of the ``ParquetRecordWriter`` column kinds."""
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    if kind == "bool":
        return pa.bool_()
    if kind == "timestamp":
        return pa.timestamp("us", tz="UTC")
    # 'string' and 'json' are both text columns; 'json' encodes its non-text values on the way in
    return pa.string()


def _value_kind(value):
    """The column kind *value* alone would need, or ``None`` for a null (which fits any column)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, datetime):
        return "timestamp"
    return "json"


def _merge_kinds(current, new):
    """The kind a column needs to hold both kinds: ints widen to float, anything else mixed to json."""
    if current is None or current == new:
        return new
    if new is None:
        return current
    if {current, new} == {"int", "float"}:
        return "float"
    return "json"


def _coerce(value, kind):
    """*value* as the Python type a *kind* column stores; unconvertible values become null.

    A ``json`` column encodes every value, strings included, so each cell of the loaded column parses
    the same way. A ``string`` column keeps text as it is and renders anything else readably.
    """
    if value is None:
        return None
    if kind == "json":
        return json.dumps(_json_safe(value), ensure_ascii=False)
    if kind == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(_json_safe(value), ensure_ascii=False)
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "t", "yes", "y", "1")
        try:
            return bool(value)
        except (TypeError, ValueError):
            return None
    if kind == "timestamp":
        return as_datetime(value)
    return value


class ParquetRecordWriter:
    """Write flat dict records to a single Parquet file, inferring the column types from the data.

    Used by ``ParquetStorage`` and ``jsonl_to_parquet``; also usable on its own to turn any sequence
    of dicts into a Parquet file. Records are buffered, then written as Parquet row groups:

    * ``batch_size`` 0 (the default) buffers every record and writes one row group on ``close``, so
      the column types are inferred from the whole dataset.
    * a positive ``batch_size`` writes a row group each time that many records have accumulated. The
      schema is fixed by the first batch; later values are coerced to it, and a column that only
      appears in a later batch cannot be added and is logged and dropped.

    Column types (the values of the *schema* dict, and what inference produces):
        - string: text, as-is
        - json: the value JSON-encoded into a text column -- what dicts, lists and columns of mixed
          types become, so the loader on the far side can parse one column consistently
        - int / float / bool: numeric and boolean columns
        - timestamp: microsecond UTC timestamps, from ``datetime`` objects, ISO-8601 strings or epoch
          seconds

    Anything declared in *schema* keeps the declared type (and its column order comes first), which
    is how a run whose data happens to be uniform can still produce the same table as every other
    run. Missing keys become nulls, so records need not all have the same shape.

    ``close`` finalizes the file and returns ``{"path", "rows", "columns"}``; the writer is
    thread-safe, so an Iterator running with ``workers`` can add from several threads.
    """

    KINDS = ("string", "json", "int", "float", "bool", "timestamp")

    def __init__(self, path, schema=None, batch_size=0, compression="snappy", metadata=None):
        self.path = path
        self.batch_size = int(batch_size or 0)
        self.compression = compression or "snappy"
        self.metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
        self.declared = {}
        for column, kind in (schema or {}).items():
            kind = str(kind).lower()
            if kind not in self.KINDS:
                raise ValueError(
                    f"Unknown Parquet column type {kind!r} for column {column!r}; use one of {', '.join(self.KINDS)}"
                )
            self.declared[column] = kind
        self.kinds = None
        self.rows = 0
        self._buffer = []
        self._lock = threading.Lock()
        self._writer = None
        self._schema = None
        self._modules_cache = None
        self._dropped = set()

    def _modules(self):
        """The pyarrow modules, imported on first write so chai works without pyarrow installed."""
        if self._modules_cache is None:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as e:  # pragma: no cover - depends on the install
                raise ImportError("Writing Parquet files requires pyarrow: pip install pyarrow") from e
            self._modules_cache = (pa, pq)
        return self._modules_cache

    def add(self, record):
        """Buffer one record (a flat dict of column name to value)."""
        with self._lock:
            self._buffer.append(record)
            if self.batch_size and len(self._buffer) >= self.batch_size:
                self._write_buffer()

    def add_all(self, records):
        """Buffer each record of an iterable."""
        for record in records:
            self.add(record)

    def close(self):
        """Write whatever is buffered, finalize the file and return ``{"path", "rows", "columns"}``."""
        with self._lock:
            self._write_buffer(final=True)
            writer, self._writer = self._writer, None
            if writer is not None:
                writer.close()
            return {"path": self.path, "rows": self.rows, "columns": dict(self.kinds or {})}

    def _write_buffer(self, final=False):
        """Write the buffered records as one row group, opening the file (and schema) if needed."""
        rows, self._buffer = self._buffer, []
        if not rows:
            # A run that produced nothing still leaves an empty, correctly typed file when the
            # schema was declared -- there is nothing to infer one from otherwise.
            if final and self._writer is None and self.declared:
                self._open(dict(self.declared))
            return
        if self._writer is None:
            self._open(self._infer(rows))
        pa, _pq = self._modules()
        arrays = [
            pa.array([_coerce(row.get(column), kind) for row in rows], type=_arrow_type(pa, kind))
            for column, kind in self.kinds.items()
        ]
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self._schema))
        self.rows += len(rows)
        for row in rows:
            for column in row:
                if column not in self.kinds and column not in self._dropped:
                    self._dropped.add(column)
                    logger.warning(
                        f"Parquet column {column!r} first seen after the schema was fixed; "
                        f"not written to {self.path}"
                    )

    def _infer(self, rows):
        """Column name to column kind for *rows*: declared kinds first, then the inferred ones."""
        kinds = dict(self.declared)
        for row in rows:
            for column, value in row.items():
                if column in self.declared:
                    continue
                kinds[column] = _merge_kinds(kinds.get(column), _value_kind(value))
        # A column that was null all the way through has no kind to infer; text holds nulls fine
        return {column: kind or "string" for column, kind in kinds.items()}

    def _open(self, kinds):
        """Create the Parquet file with a schema built from *kinds* (plus the file-level metadata)."""
        pa, pq = self._modules()
        self.kinds = kinds
        metadata = dict(self.metadata)
        metadata["chai_columns"] = json.dumps(kinds)
        self._schema = pa.schema(
            [pa.field(column, _arrow_type(pa, kind)) for column, kind in kinds.items()],
            metadata=metadata,
        )
        parent_dir = os.path.dirname(self.path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._writer = pq.ParquetWriter(self.path, self._schema, compression=self.compression)


def jsonl_to_parquet(
    jsonl_file, parquet_file, schema=None, batch_size=50000, compression="snappy", metadata=None, records=None
):
    """Convert a JSON-Lines file (see ``JsonLinesStorage``) into one Parquet file.

    The counterpart to ``ParquetStorage`` for runs too large to hold in memory: stream the results to
    JSON-Lines as they are produced, then pack the finished file into Parquet for the bulk load.
    Reads and writes in batches of *batch_size* lines, so neither file is ever fully in memory; as
    with ``ParquetRecordWriter``, the column types come from the first batch unless declared in
    *schema*. A file of CHAI-TEA RESULT rows (what the storages write by default) is typed by
    ``RESULT_COLUMNS`` instead, so its JSON columns stay JSON even where every value in them happens to
    be text. Blank lines are skipped and an unparseable line raises. Returns
    ``{"path", "rows", "columns"}``.

    *records* (``{entity: row or [rows]}``) are the PROJECT/WORKFLOW/WORKFLOW_RUN rows the file's
    RESULT rows hang under, carried in the Parquet file's ``chai_tea_records`` metadata as
    ``ParquetStorage`` carries them; by default those ``JsonLinesStorage`` recorded beside the file
    (``run_records_file``), if it did.
    """
    if records is None:
        records = read_run_records(run_records_file(jsonl_file))
    if records:
        metadata = dict(metadata or {})
        metadata.setdefault("chai_tea_records", json.dumps(_json_safe(_isoformat_rows(records))))
    writer = None
    with open(jsonl_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if writer is None:
                if schema is None and is_result_row(record):
                    schema = RESULT_COLUMNS
                writer = ParquetRecordWriter(
                    parquet_file, schema=schema, batch_size=batch_size, compression=compression, metadata=metadata
                )
            writer.add(record)
    if writer is None:
        writer = ParquetRecordWriter(
            parquet_file, schema=schema, batch_size=batch_size, compression=compression, metadata=metadata
        )
    return writer.close()


def parquet_run_records(parquet_file):
    """The ``chai_tea_records`` a Parquet file of RESULT rows carries -- ``{entity: row or [rows]}`` -- or ``{}``."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(parquet_file).metadata or {}
    text = metadata.get(b"chai_tea_records")
    return json.loads(text.decode()) if text else {}


def parquet_columns(parquet_file):
    """The ``{column: kind}`` a Parquet file holds -- from the ``chai_columns`` metadata if it is there.

    Files this module wrote record their column kinds when they are created; for any other Parquet
    file the kinds are read off the Arrow schema instead (a text column is ``string``, since nothing
    says whether it holds JSON).
    """
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(parquet_file).schema_arrow
    declared = {}
    if schema.metadata and b"chai_columns" in schema.metadata:
        declared = json.loads(schema.metadata[b"chai_columns"].decode())
    kinds = {}
    for field in schema:
        kind = declared.get(field.name)
        if kind not in ParquetRecordWriter.KINDS:
            arrow = str(field.type)
            if arrow.startswith("timestamp"):
                kind = "timestamp"
            elif arrow in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                kind = "int"
            elif arrow in ("float", "double", "halffloat"):
                kind = "float"
            elif arrow == "bool":
                kind = "bool"
            else:
                kind = "string"
        kinds[field.name] = kind
    return kinds
