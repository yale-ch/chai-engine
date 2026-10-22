"""A result as a row of the CHAI-TEA RESULT table, and the other records the storages write.

``result_row`` is what every storage writes by default -- exactly the RESULT columns
(``RESULT_COLUMNS``), in order -- and ``build_record`` the record of a file storage, which can instead
select fields of ``result_to_json`` for a file meant for something else. ``correction_row`` is a
RESULT row correcting another.
"""

import uuid

import ujson as json

from ..result import FileItemResult, Result
from .provenance import correction_target, input_provenance, result_sequence, source_value
from .records import as_datetime, as_uuid, now
from .schema import column_kinds, column_types
from .values import _json_safe


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


#: Keys usable in a record's ``fields`` that are computed from the result rather than read off its
#: JSON -- the columns the tabular storages fill in beside the result itself.
COMPUTED_FIELDS = ("@record", "@input_uri", "@input_hash", "@input_locator", "@corrects")

#: The columns of the CHAI-TEA RESULT table, in order, as ``ParquetRecordWriter`` column kinds: the
#: shape every record-writing storage produces by default (defined in ``chai.persistence.schema``).
RESULT_COLUMNS = column_kinds("RESULT")


#: RESULT columns a Parquet file of RESULT rows leaves out. The central loader refuses a file that
#: has them, and the table fills them in itself (``CURRENT_TIMESTAMP``) when the rows are loaded; when
#: a result was made is still in the file, as ``md_timestamp``.
PARQUET_OMITTED_COLUMNS = ("created", "modified")

#: The columns of a Parquet file of RESULT rows -- ``RESULT_COLUMNS`` less ``PARQUET_OMITTED_COLUMNS``
#: -- as ``ParquetStorage`` and ``jsonl_to_parquet`` write them.
PARQUET_RESULT_COLUMNS = {
    column: kind for column, kind in RESULT_COLUMNS.items() if column not in PARQUET_OMITTED_COLUMNS
}


def is_result_row(record):
    """Whether *record* has exactly the CHAI-TEA RESULT columns -- what the storages write by default."""
    return isinstance(record, dict) and list(record) == list(RESULT_COLUMNS)


def parquet_result_row(row):
    """A RESULT *row* as a Parquet file holds it: without the ``PARQUET_OMITTED_COLUMNS``."""
    return {column: value for column, value in row.items() if column not in PARQUET_OMITTED_COLUMNS}


def result_row(
    result: Result, run_id=None, sources=None, input_source=None, constants=None, with_hash=True
):
    """*result* as a row of the CHAI-TEA RESULT table: exactly its columns, in its order.

    The default record of ``JsonLinesStorage`` and ``ParquetStorage``, so a run's output loads into
    the central database as it stands. *run_id* becomes ``workflow_run_id`` (a non-uuid id is hashed
    into one). ``input`` is the file or URI the result was generated from (see ``input_provenance``),
    else the id of the result it was made from; ``input_segment`` is the input locator as JSON text,
    ``input_sequence`` the result's position among its siblings (``result_sequence``: the position
    the nearest iterator gave the entry it was made from), for ordering rows. ``previous_id`` is the
    result this one corrects, and ``md_duration``/``md_timestamp`` come from the result's metadata;
    ``md_cost`` is the cost the component recorded for its call (``recorded_cost``). The columns a
    ``Result`` holds itself -- ``was_successful``, ``editor_user_id``, ``created``, ``modified`` --
    are taken from it. *sources* and *constants* -- which would otherwise be extra columns -- are
    recorded in ``extra_data`` under their keys. *with_hash* false leaves ``input_hash`` null, and
    skips any file read the digest would need.
    """
    whole = result_to_json(result)
    metadata = whole.get("metadata") or {}
    provenance = input_provenance(result, input_source, with_hash=with_hash)
    source_input = provenance["uri"]
    if source_input is None and whole.get("input") is not None:
        source_input = str(whole["input"])
    locator = provenance["locator"]
    extra = dict(whole.get("extraInfo") or {})
    extra.pop("corrects", None)  # recorded as previous_id
    extra.pop("sequence", None)  # recorded as input_sequence
    for key, cid in (sources or {}).items():
        extra[key] = _json_safe(source_value(result, cid))
    for key, value in (constants or {}).items():
        extra[key] = _json_safe(value)
    duration = metadata.get("duration")
    created = getattr(result, "created", None) or now()
    return {
        "id": as_uuid(whole["id"]),
        "workflow_run_id": as_uuid(run_id),
        "process_id": whole.get("processorId"),
        "input": source_input,
        "input_hash": provenance["hash"],
        "input_segment": json.dumps(locator) if locator else None,
        "input_sequence": result_sequence(result),
        "was_successful": not (metadata.get("type") == "ERROR" or "error" in metadata),
        "value": whole.get("value"),
        "metadata": metadata or None,
        "extra_data": extra or None,
        "previous_id": as_uuid(correction_target(result)),
        "editor_user_id": as_uuid(getattr(result, "editor_user_id", None)),
        "md_cost": recorded_cost(metadata),
        "md_duration": float(duration) if isinstance(duration, (int, float)) else None,
        "md_timestamp": as_datetime(metadata.get("timestamp")) or created,
        "created": created,
        "modified": getattr(result, "modified", None) or created,
    }


def recorded_cost(metadata):
    """The USD cost a component recorded for its call (``metadata.cost``), or ``None``.

    The LiteLLM-backed components price every call when they make it (LiteLLM's figure, or their
    ``pricing`` setting); nothing is priced afterwards.
    """
    cost = (metadata or {}).get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
        return None
    return float(cost)


def build_record(
    result: Result, fields=None, sources=None, input_source=None, run_id=None, constants=None
):
    """The flat, JSON-safe record for *result*: a CHAI-TEA RESULT row, or the selected fields.

    Shared by the storage components that write one record per result (``JsonLinesStorage``,
    ``ParquetStorage``) so a run can be streamed to JSON-Lines or packed into Parquet with the same
    columns. With no *fields* -- the default, and what a workflow's output should be -- the record is
    ``result_row``: exactly the columns of the CHAI-TEA RESULT table, with *sources* and *constants*
    kept in its ``extra_data``. *fields* instead selects keys of the ``result_to_json``
    representation -- a list to keep them as they are, a ``{key: name}`` dict to keep and rename
    them -- for a record meant for something other than the CHAI-TEA database.

    In the dict form a key may also be one of the computed ``COMPUTED_FIELDS``:

    * ``@record`` -- the whole result JSON
    * ``@input_uri`` / ``@input_hash`` / ``@input_locator`` -- the file or URI the entry was generated
      from, the md5 of that input's content, and where in it this entry is (see ``input_provenance``;
      *input_source* names the component whose result counts as the input, overriding the component
      marked ``source: true``)
    * ``@corrects`` -- the id of the result this one corrects (see ``correction_target``)

    *sources* is a ``{key: component id}`` dict, each recording the value of the nearest ancestor
    result that component produced (see ``source_value``).
    """
    if not fields:
        return result_row(result, run_id, sources, input_source, constants)
    whole = result_to_json(result)
    js = whole
    if fields:
        if isinstance(fields, dict):
            computed = {"@record": whole}
            if any(key in fields for key in ("@input_uri", "@input_hash", "@input_locator")):
                provenance = input_provenance(result, input_source, with_hash="@input_hash" in fields)
                computed["@input_uri"] = provenance["uri"]
                computed["@input_hash"] = provenance["hash"]
                computed["@input_locator"] = provenance["locator"]
            if "@corrects" in fields:
                computed["@corrects"] = correction_target(result)
            js = {}
            for key, name in fields.items():
                if key.startswith("@"):
                    if key not in COMPUTED_FIELDS:
                        raise ValueError(f"Unknown computed field {key!r}; use one of {', '.join(COMPUTED_FIELDS)}")
                    js[name] = computed[key]
                elif key in whole:
                    js[name] = whole[key]
        else:
            js = {key: whole[key] for key in fields if key in whole}
    for key, cid in (sources or {}).items():
        js[key] = _json_safe(source_value(result, cid))
    return js


def correction_row(original, corrected_value, metadata=None, process_id=None, editor_user_id=None):
    """A RESULT row correcting the stored row *original*: a new entry pointing back at it.

    The correction belongs to the same run and was made from the same input (``input``,
    ``input_hash``, ``input_segment``, ``input_sequence`` are the original's), its ``value`` is what
    the original should have said, its ``previous_id`` is the original and its ``metadata`` says who
    made it. *process_id* names the component that made it, if one did; *editor_user_id* the USER who
    did, if a person did (it is stored only if that USER row exists).
    """
    stamp = now()
    row = {
        "id": str(uuid.uuid4()),
        "workflow_run_id": original["workflow_run_id"],
        "process_id": process_id,
        "input": original.get("input"),
        "input_hash": original.get("input_hash"),
        "input_segment": original.get("input_segment"),
        "input_sequence": original.get("input_sequence"),
        "was_successful": True,
        "value": _json_safe(corrected_value),
        "metadata": _json_safe(metadata) if metadata else None,
        "extra_data": None,
        "previous_id": original["id"],
        "editor_user_id": as_uuid(editor_user_id),
        "md_cost": None,
        "md_duration": None,
        "md_timestamp": stamp,
        "created": stamp,
        "modified": stamp,
    }
    return {column: row[column] for column in column_types("RESULT")}


def _correction_metadata(agent, metadata):
    metadata = dict(metadata or {})
    if agent is not None:
        metadata.setdefault("agent", agent)
    return metadata
