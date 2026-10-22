"""Older files: records stored before the storages wrote RESULT rows, reshaped into them."""

import logging
import os
from datetime import datetime

import ujson as json

from .jsonl import read_jsonl
from .provenance import _file_md5, _sequence_of
from .records import as_datetime, as_uuid, now
from .rows import RESULT_COLUMNS, recorded_cost
from .values import _loads

logger = logging.getLogger("chai")


def _maybe_json(value):
    """Parse *value* if it is JSON text, otherwise return it as it is.

    Older tabular layouts kept JSON in text columns, so a field can arrive either way.
    """
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")) or text in ("null", "true", "false"):
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def _first(record, *keys):
    """The first of *keys* present in *record* with a value that is not None."""
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def record_to_result_row(record, workflow_run_id=None, fields=None, at=None):
    """One stored *record* -- in any layout a storage has written -- as a RESULT row, or ``None``.

    For files from before the storages wrote RESULT rows: accepts the RESULT columns themselves
    (``process_id``, ``extra_data``, ``input_segment``, ``previous_id``), the ``result_to_json`` keys
    (``processorId``, ``extraInfo``) and the old tabular columns (``processor_id``, ``value_json``,
    ``metadata_json``, ``extra_json``, ``input_uri``, ``input_locator``, ``corrects_id``,
    ``created_at``). *fields* first renames a workflow's own keys to those (``{"text": "value",
    "page": "input_uri"}``). *workflow_run_id* overrides the run the record names. ``md_cost`` is
    the cost the component recorded, else the record's own ``md_cost``; ``created``/``modified``, when the record has none, and
    ``md_timestamp`` as a last resort, are *at* (default now). A record with no id is no row.
    """
    if fields:
        record = {fields.get(key, key): value for key, value in record.items()}
    row_id = as_uuid(_first(record, "id", "@id"))
    if row_id is None:
        return None
    at = at or now()
    metadata = _maybe_json(_first(record, "metadata", "metadata_json")) or {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}
    extra = _maybe_json(_first(record, "extra_data", "extraInfo", "extra", "extra_json")) or None
    locator = _maybe_json(_first(record, "input_segment", "input_locator"))
    if isinstance(locator, str):
        locator = _maybe_json(locator)
    # input_uri is what the storage recorded the row as coming from; without one, the plain "input"
    # field is the id of the result this was made from, which is still the best answer available
    source_input = _first(record, "input_uri", "@input_uri", "input")
    successful = not (metadata.get("type") == "ERROR" or "error" in metadata)
    if isinstance(record.get("was_successful"), bool):
        successful = record["was_successful"]
    value = _maybe_json(_first(record, "value", "@record"))
    if value is None:
        value = _loads(record.get("value_json"))  # always JSON text, a quoted string included
    sequence = record.get("input_sequence")
    duration = metadata.get("duration")
    cost = recorded_cost(metadata)
    if cost is None and isinstance(record.get("md_cost"), (int, float)):
        cost = float(record["md_cost"])
    values = {
        "id": row_id,
        "workflow_run_id": as_uuid(workflow_run_id or record.get("workflow_run_id")),
        "process_id": _first(record, "process_id", "processorId", "processor_id", "step"),
        "input": None if source_input is None else str(source_input),
        "input_hash": _first(record, "input_hash", "@input_hash"),
        "input_segment": json.dumps(locator) if locator else None,
        "input_sequence": sequence if isinstance(sequence, int) else _sequence_of(locator),
        "was_successful": successful,
        "value": value,
        "metadata": metadata or None,
        "extra_data": extra,
        "previous_id": as_uuid(_first(record, "previous_id", "corrects_id", "@corrects")),
        "editor_user_id": as_uuid(record.get("editor_user_id")),
        "md_cost": cost,
        "md_duration": float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None,
        "md_timestamp": as_datetime(_first(record, "md_timestamp", "created_at"))
        or as_datetime(metadata.get("timestamp"))
        or at,
        "created": as_datetime(record.get("created")) or at,
        "modified": as_datetime(record.get("modified")) or at,
    }
    return {column: values[column] for column in RESULT_COLUMNS}


def _file_run_id(path):
    """A workflow_run_id for the run stored in *path*, stable for as long as its content is."""
    return as_uuid(f"workflow_run:{_file_md5(path)}")


def jsonl_to_result_rows(source, target, workflow_run_id=None, fields=None):
    """Rewrite the JSON-Lines file *source*, in whatever layout it was stored, as RESULT rows in *target*.

    Each record goes through ``record_to_result_row``. A record that names no run gets
    *workflow_run_id*, else one derived from the file's content (so converting the same file twice
    gives the same id); *workflow_run_id* also overrides the run records name. The loader has to
    create that WORKFLOW_RUN row first -- pass its rows to ``jsonl_to_parquet`` as ``records``. A
    row whose record says nothing of its order gets ``input_sequence`` from its ``input``: that
    input's position among the file's inputs, sorted -- the order ``DirFileProvider`` reads a
    directory of pages in. Records with no id, and lines that are not JSON, are skipped and counted.
    Returns ``{"path", "rows", "skipped", "workflow_run_id"}``, the last being the first row's run.
    """
    at = now()
    inputs = set()
    for record in read_jsonl(source):
        row = record_to_result_row(record, workflow_run_id, fields=fields, at=at)
        if row and row["input_sequence"] is None and row["input"] is not None:
            inputs.add(row["input"])
    positions = {value: position for position, value in enumerate(sorted(inputs))}
    derived = None
    first_run = None
    rows = skipped = 0
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as out:
        for record in read_jsonl(source):
            run = workflow_run_id or record.get("workflow_run_id")
            if not run:
                derived = derived or _file_run_id(source)
                run = derived
            row = record_to_result_row(record, run, fields, at)
            if row is None:
                logger.warning("Record with no id skipped: %r", record)
                skipped += 1
                continue
            if row["input_sequence"] is None and row["input"] is not None:
                row["input_sequence"] = positions.get(row["input"])
            row = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in row.items()}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            first_run = first_run or row["workflow_run_id"]
    first_run = first_run or as_uuid(workflow_run_id) or derived or _file_run_id(source)
    return {"path": target, "rows": rows, "skipped": skipped, "workflow_run_id": first_run}
