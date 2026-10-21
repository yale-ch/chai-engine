"""Storage components: persist Results to the filesystem or a database.

Storage steps are pass-through: they write the input Result somewhere and return it unchanged, so
they can be inserted anywhere in a pipeline (typically as ``next_steps``) without affecting the data
flow.

Everything a storage records is in the CHAI-TEA schema (``chai.schema``): ``SqliteStorage`` and
``PostgresStorage`` write its tables -- each result a RESULT row, under the run's PROJECT, WORKFLOW
and WORKFLOW_RUN rows -- and ``JsonLinesStorage``/``ParquetStorage`` write the same RESULT rows to
files. ``write_rows`` is the one writer for both databases; ``store_rows``/``store_postgres_rows``
put rows made elsewhere in.

Besides the components, this module exposes plain helper functions (``list_results``, ``get_result``,
``save_correction``, ``list_processors``) that a viewer app can import to browse a ``SqliteStorage``
database and record human corrections alongside the original values. The helpers open a fresh
connection per call, so they are safe to use from multi-threaded servers (e.g. Flask).

``jsonl_to_parquet`` (and the ``ParquetRecordWriter`` behind it and ``ParquetStorage``) packs a run's
results into a single Parquet file for bulk loading into a remote database; ``parquet_to_postgres``
is the load step for PostgreSQL, putting a run's file into a table there, and ``postgres_params`` /
``ensure_postgres_database`` / ``ensure_postgres_schema`` are the connection and setup helpers it and
``PostgresStorage`` share.
"""

import atexit
import hashlib
import logging
import os
import re
import sqlite3
import threading
import uuid
import weakref
from datetime import datetime, timezone

import ujson as json

from .core import Component, Result
from .records import UUID_NAMESPACE, as_uuid, now  # noqa: F401 -- as_uuid/UUID_NAMESPACE re-exported
from .result import FileItemResult
from .schema import (
    TABLE_ORDER,
    check_for_clashes,
    column_kinds,
    column_types,
    identifier as _pg_identifier,
    schema_statements,
    sqlite_statements,
    table_names,
)

logger = logging.getLogger("chai")


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


_line_locks = {}
_line_locks_guard = threading.Lock()
#: path -> the run that last emptied it, so 'truncate' empties a file once per run, not once per step
_truncated_runs = {}


def _line_lock(path):
    """The write lock shared by everything appending to *path* (one per file, not per component)."""
    with _line_locks_guard:
        lock = _line_locks.get(path)
        if lock is None:
            lock = _line_locks[path] = threading.Lock()
        return lock


def source_value(result: Result, component_id: str):
    """Value of the nearest result in *result*'s provenance chain that component *component_id* made.

    Walks up the ``Result.input`` chain, so a result can record what it was computed from -- the CSV
    row, the file, the segment of text -- rather than just the id of a result that lives elsewhere.
    Returns ``None`` when no ancestor came from that component. A ``FileItemResult`` contributes its
    ``file_name``, never the file's content.

    Note that an ``Iterator`` stamps itself onto each entry it processes, so the entry a step ran on
    is found under the *iterator's* id rather than that of the component that first made it.
    """
    found = _source_result(result, component_id)
    if found is None:
        return None
    if isinstance(found, FileItemResult):
        return found.file_name
    return found.value


_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

_FILE_HASHES = {}
_file_hash_guard = threading.Lock()


def _file_md5(path):
    """The md5 of a file's content, remembered while the file's size and mtime stay the same.

    A run stores one result per page, per row or per crop of the same file, so the file is read and
    hashed once rather than once per result.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _file_hash_guard:
        digest = _FILE_HASHES.get(key)
    if digest is not None:
        return digest
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    with _file_hash_guard:
        if len(_FILE_HASHES) >= 4096:  # a long run over many files should not grow without bound
            _FILE_HASHES.clear()
        _FILE_HASHES[key] = digest
    return digest


def content_md5(value):
    """The md5 of *value*'s content: bytes as they are, text as UTF-8, anything else as sorted JSON.

    Sorting the keys makes the digest of a dict (a CSV row, a parsed record) independent of the order
    its keys happen to be in, so the same content always hashes the same way.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return hashlib.md5(value).hexdigest()
    if not isinstance(value, str):
        value = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def result_md5(value):
    """The md5 of the content *value* holds -- a ``Result``, a file path or a plain value.

    A ``FileItemResult`` hashes the file (or the bytes it is carrying), a path names its file's
    content, and any other Result hashes its value. The digest is remembered on the Result, so a
    result derived from the same input by several steps is only hashed once.
    """
    if isinstance(value, Result):
        cached = getattr(value, "_content_md5", None)
        if cached:
            return cached
        if isinstance(value, FileItemResult):
            digest = content_md5(value.file_bytes) if value.file_bytes else _file_md5(value.file_name)
        else:
            digest = content_md5(value.value)
        try:
            value._content_md5 = digest
        except AttributeError:  # pragma: no cover - Results are ordinary objects
            pass
        return digest
    if isinstance(value, str) and os.path.isfile(value):
        return _file_md5(value)
    return content_md5(value)


def _source_result(result: Result, component_id: str):
    """The nearest result in *result*'s provenance chain that component *component_id* produced."""
    for node in _provenance(result):
        processor = node.processor
        if processor is not None and getattr(processor, "id", None) == component_id:
            return node
    return None


def _provenance(result):
    """Yield *result* and each Result up its ``input`` chain, stopping if the chain loops back."""
    seen = set()
    while isinstance(result, Result) and id(result) not in seen:
        seen.add(id(result))
        yield result
        result = result.input


def _uri_for(node):
    """Walk *node*'s provenance chain for the file or URI it came from, or ``None`` if there is none."""
    last = None
    for last in _provenance(node):
        if isinstance(last, FileItemResult) and last.file_name:
            return last.file_name
    # The end of the chain is the raw input a provider was given: a path or a URI, but not text
    raw = last.input if isinstance(last, Result) else node
    if isinstance(raw, str) and (_URI_SCHEME.match(raw) or os.path.exists(raw)):
        return raw
    return None


def recorded_source(result: Result, source_id=None):
    """The result a stored row should name as what it was generated from, or ``None`` if unmarked.

    A run's real source is rarely the raw input at the bottom of the chain -- that may be a directory
    of scans, iterated into pages, segmented into regions and iterated again. The component whose
    results are the source is marked ``source: true`` in the config (see ``Component``), and this
    returns the nearest such result up *result*'s provenance chain, so every row a page gives rise to,
    however many steps later, records that page. *source_id* names a component explicitly instead.
    """
    if source_id:
        return _source_result(result, source_id)
    for node in _provenance(result):
        if getattr(node.processor, "is_source", False):
            return node
    return None


def input_locator(result: Result, source=None):
    """How to find *result* inside its source: the locators between it and *source*, outermost first.

    Each component that carves a piece out of something records where that piece is (``Result.locator``
    -- a bounding box, a character range), and this collects those frames along the chain, so a stored
    row says where in the source it came from without any of the pieces being materialized. A region
    of a page whose transcript was split into sentences yields two frames,
    ``[{"bbox": [...]}, {"start": 120, "end": 168}]``: the region within the page, then the sentence
    within the region's text. ``None`` when nothing on the way asserted a position.
    """
    frames = []
    for node in _provenance(result):
        if node is source:
            break
        locator = node.extra.get("locator", None) if isinstance(node.extra, dict) else None
        if locator:
            frames.append(_json_safe(locator))
    frames.reverse()  # outermost (nearest the source) first, so they read source -> result
    return frames or None


def input_provenance(result: Result, source_id=None, with_hash=True):
    """What a stored row records about the input *result* was generated from.

    Returns ``{"uri", "hash", "locator"}``: the file or URI of the recorded source (see
    ``recorded_source``), the md5 of that source's content, so rows made from the same input can be
    found together whatever the run or the file name, and the locators saying where in it this result
    is (see ``input_locator``). With no component marked as the source, the input is the result's
    immediate ``input`` and the URI is the nearest file or URI up the chain; a *source_id* naming a
    component that made nothing in this chain records nothing at all. *with_hash* false skips the
    digest, and any file read it would need.
    """
    source = recorded_source(result, source_id)
    start = source
    if source is None and not source_id:
        # Nothing marked, and no component named: the input is what the result was made from, and the
        # nearest file to the result itself is the best URI there is. A *named* source that is not in
        # the chain records nothing rather than quietly standing in something else.
        source = result.input if isinstance(result, Result) else None
        start = result
    return {
        "uri": _uri_for(start),
        "hash": result_md5(source) if with_hash else None,
        "locator": input_locator(result, source),
    }


def correction_target(result: Result):
    """The id of the result *result* corrects, or ``None`` when it is not a correction.

    A component that produces a corrected version of an earlier result says so by putting that
    result (or its id) in the new result's ``extra['corrects']`` -- ``metadata['corrects']`` is read
    too -- and the storages record it as the row's ``previous_id``, which is what makes the
    correction an entry in its own right rather than an overwrite of the original.
    """
    for holder in (getattr(result, "extra", None), getattr(result, "metadata", None)):
        if isinstance(holder, dict) and holder.get("corrects"):
            target = holder["corrects"]
            return target.id if isinstance(target, Result) else str(target)
    return None


#: Keys usable in a record's ``fields`` that are computed from the result rather than read off its
#: JSON -- the columns the tabular storages fill in beside the result itself.
COMPUTED_FIELDS = ("@record", "@input_uri", "@input_hash", "@input_locator", "@corrects")

#: The columns of the CHAI-TEA RESULT table, in order, as ``ParquetRecordWriter`` column kinds: the
#: shape every record-writing storage produces by default (defined in ``chai.schema``).
RESULT_COLUMNS = column_kinds("RESULT")


def _sequence_of(locator):
    """The last integer frame of an input locator (an index, a page number), if it has one."""
    if not isinstance(locator, list):
        return None
    for frame in reversed(locator):
        if isinstance(frame, int) and not isinstance(frame, bool):
            return frame
        if isinstance(frame, dict):
            for key in ("sequence", "index", "page", "n"):
                if isinstance(frame.get(key), int):
                    return frame[key]
    return None


def result_sequence(result: Result):
    """*result*'s position among its siblings: the nearest ``sequence`` up its chain, or ``None``.

    An ``Iterator`` stamps each entry with its position, so this is the position of the entry the
    result was made from -- the page a transcription and its translation came from, the line within
    an image. Failing that, an integer frame of the input locator counts (see ``_sequence_of``).
    """
    for node in _provenance(result):
        sequence = node.extra.get("sequence") if isinstance(node.extra, dict) else None
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            return sequence
    return _sequence_of(input_locator(result))


def is_result_row(record):
    """Whether *record* has exactly the CHAI-TEA RESULT columns -- what the storages write by default."""
    return isinstance(record, dict) and list(record) == list(RESULT_COLUMNS)


def result_row(result: Result, run_id=None, sources=None, input_source=None, constants=None, with_hash=True):
    """*result* as a row of the CHAI-TEA RESULT table: exactly its columns, in its order.

    The default record of ``JsonLinesStorage`` and ``ParquetStorage``, so a run's output loads into
    the central database as it stands. *run_id* becomes ``workflow_run_id`` (a non-uuid id is hashed
    into one). ``input`` is the file or URI the result was generated from (see ``input_provenance``),
    else the id of the result it was made from; ``input_segment`` is the input locator as JSON text,
    ``input_sequence`` the result's position among its siblings (``result_sequence``: the position
    the nearest iterator gave the entry it was made from), for ordering rows. ``previous_id`` is the
    result this one corrects, and
    ``md_cost``/``md_duration``/``md_timestamp`` come from the result's metadata. The columns a
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
    cost = metadata.get("cost")
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
        "md_cost": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        "md_duration": float(duration) if isinstance(duration, (int, float)) else None,
        "md_timestamp": _as_datetime(metadata.get("timestamp")) or created,
        "created": created,
        "modified": getattr(result, "modified", None) or created,
    }


def _run_id_of(component):
    """The run a storage's rows belong to: its ``run_id`` setting, else its workflow's current run."""
    return component.settings.get("run_id") or getattr(component.workflow, "run_id", None)


def build_record(result: Result, fields=None, sources=None, input_source=None, run_id=None, constants=None):
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


def _as_datetime(value):
    """*value* as a UTC-aware datetime -- from a datetime, an ISO-8601 string or epoch seconds."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return _as_datetime(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


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
        return _as_datetime(value)
    return value


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


class TextFileStorage(Storage):
    """Writes each Result's text to a file of its own, named after the input it was made from.

    Where ``FileSystemStorage`` keeps the whole result as JSON in a pairtree, this keeps just the
    text, under a name a person can find: a run over a directory of page images that transcribes and
    translates each page writes ``f0007r.txt`` per page in each of its two output directories, one
    file per page rather than one file per run. The name comes from the nearest file up the result's
    provenance chain (or the result named by ``name_from``), so it survives however many steps sit
    between the page and the text. Values that are not text are written as JSON. Returns the input
    unchanged.

    Settings:
        - directory: where to write the files (default 'results'); created if it does not exist
        - name_from: id of the component whose result in the chain names the file; its value (or file
          name) is used (default: the nearest file up the chain)
        - suffix: appended to the name, e.g. '-en' for 'f0007r-en.txt' (default '')
        - extension: file extension including the dot (default '.txt')
        - encoding: text encoding to write with (default 'utf-8')
        - overwrite: overwrite an existing file (default true); false adds a '.1', '.2', ... suffix
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.directory = self.settings.get("directory", "results")
        self.name_from = self.settings.get("name_from", "") or ""
        self.suffix = self.settings.get("suffix", "")
        self.extension = self.settings.get("extension", ".txt")
        self.encoding = self.settings.get("encoding", "utf-8")
        self.overwrite = bool(self.settings.get("overwrite", True))
        os.makedirs(self.directory, exist_ok=True)

    def result_text(self, input: Result):
        """The text to write: the value itself, decoded if it is bytes, or JSON if it is neither."""
        value = input.value
        if isinstance(value, (bytes, bytearray)):
            return value.decode(self.encoding, "replace")
        if isinstance(value, str):
            return value
        return json.dumps(_json_safe(value), indent=2)

    def result_name(self, input: Result):
        """The base name (no extension) for *input*'s file, from the input it was generated from."""
        source = None
        if self.name_from:
            source = _source_result(input, self.name_from)
            if source is None:
                logger.warning(f"{self} found no result from '{self.name_from}' to name the file after")
        name = ""
        if source is not None:
            name = source.file_name if isinstance(source, FileItemResult) else str(source.value)
        else:
            name = _uri_for(input) or ""
        if not name:
            # Nothing in the chain says where this came from; the result's own id at least is unique
            return input.id
        return os.path.splitext(os.path.basename(name.rstrip("/")))[0]

    def _process(self, input: Result) -> Result:
        if isinstance(input, FileItemResult):
            # Writing a file result's text would mean reading the file back off disk to copy it
            logger.warning(f"{self} was given a file result and wrote nothing: {input!r}")
            return input
        fn = os.path.join(self.directory, f"{self.result_name(input)}{self.suffix}{self.extension}")
        if not self.overwrite and os.path.exists(fn):
            version = 1
            while os.path.exists(f"{fn}.{version}"):
                version += 1
            fn = f"{fn}.{version}"
        with open(fn, "w", encoding=self.encoding) as fh:
            fh.write(self.result_text(input))
        return input


class JsonLinesStorage(Storage):
    """Append each Result to one JSON-Lines file as soon as it is produced.

    One line of JSON per result, written and flushed the moment the result passes through, so a run
    over a large input never holds its results in memory waiting to write them at the end, and
    whatever has been processed so far survives a crash or a kill part way through. Pair it with an
    Iterator configured with ``retain_results: false`` -- otherwise the results are streamed to disk
    but still kept in the iterator's output.

    The file is re-opened in append mode for each line and closed again, so it is a complete, valid
    JSON-Lines file at every moment; writes to the same path are serialized by a shared lock, making
    it safe for an Iterator running with ``workers`` and for several steps (a value branch and an
    error branch, say) writing to one file. Separate *processes*, e.g. the parallel runs of a
    ``SliceIterator``, must each be given their own file.

    Values are rendered by ``result_to_json``: bytes-safe, non-recursive, and a ``FileItemResult``
    stores its ``file_name`` rather than its content. Returns the input unchanged.

    Settings:
        - file: path of the .jsonl file to write (default 'results.jsonl'); it and its parent
          directories are created when the workflow is built, so a run that produces nothing still
          leaves an empty file rather than none
        - mode: 'append' (default) adds to whatever the file already holds; 'truncate' empties it
          when a run starts, so re-running replaces the previous run's lines. Building a workflow
          never empties anything -- a tool that only loads or inspects a workflow leaves its output
          alone. The file is emptied once per run however many steps write to it, and results
          pushed through outside a ``Workflow.run`` empty it before the first is written
        - fields: leave unset. Each line is then a row of the CHAI-TEA RESULT table -- exactly its
          columns, in its order (see ``result_row``), with ``workflow_run_id`` the workflow's
          ``run_id`` -- which is what a run's output must be for the central database to load it.
          Set only for a file meant for something else: a list of result-JSON keys to keep, or a
          ``{key: name}`` dict that keeps and renames them, in which a key from ``COMPUTED_FIELDS``
          (``@record``, ``@input_uri``, ``@input_hash``, ``@corrects``) names a column computed from
          the result rather than read off its JSON
        - sources: ``{key: component id}`` dict; for each entry, the value of the nearest ancestor
          result that component produced is recorded under *key* (see ``source_value``) -- in the
          row's ``extra_data``, or as a key of its own when ``fields`` is set. This is how a line can
          carry the input it was derived from next to the value derived from it
        - input_source: component id whose result counts as the input for ``input``/``input_hash``
          (default: the component marked ``source: true``, else the result's immediate input)
        - run_id: the ``workflow_run_id`` to record (default: the workflow's ``run_id``, fresh for
          each run); an id that is not a uuid is hashed into one
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "file" not in self.settings:
            self.settings["file"] = "results.jsonl"
        self.file_name = os.path.abspath(self.settings["file"])
        parent_dir = os.path.dirname(self.file_name)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        # Created (never emptied) when the workflow is built, so a run that produces nothing still
        # leaves a file; 'truncate' empties it when a run starts, not before
        self.truncate = str(self.settings.get("mode", "append")).lower() == "truncate"
        with _line_lock(self.file_name):
            with open(self.file_name, "a"):
                pass
        if self.truncate and hasattr(self.workflow, "add_listener"):
            self.workflow.add_listener(self._on_workflow_event)

    def _on_workflow_event(self, payload):
        """Empty the file when the workflow starts a run."""
        if payload.get("component_id") == self.workflow.id and payload.get("event") == "component_start":
            self._truncate_for_run()

    def _truncate_for_run(self):
        """Empty the file, once per run: whichever step writing to it gets there first does it."""
        run_id = getattr(self.workflow, "run_id", None) or "standalone"
        with _line_lock(self.file_name):
            if _truncated_runs.get(self.file_name) != run_id:
                with open(self.file_name, "w"):
                    pass
                _truncated_runs[self.file_name] = run_id

    def build_json(self, input: Result):
        """The record for one line: the selected result fields plus any configured source values."""
        return build_record(
            input,
            self.settings.get("fields", None),
            self.settings.get("sources", None),
            self.settings.get("input_source", None),
            run_id=_run_id_of(self),
        )

    def _process(self, input: Result) -> Result:
        """Append the result to the JSON-Lines file, then pass it through unchanged."""
        # A RESULT row's timestamps are datetimes; in JSON they are ISO-8601 text
        record = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.build_json(input).items()}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        if self.truncate:
            self._truncate_for_run()  # a no-op once this run has emptied the file
        with _line_lock(self.file_name):
            with open(self.file_name, "a", encoding="utf-8") as fh:
                fh.write(line)
        return input


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


def _isoformat_rows(records):
    """``{entity: row}`` with its datetimes as ISO-8601 text, for JSON."""
    return {
        entity: {k: v.isoformat() if isinstance(v, datetime) else v for k, v in row.items()}
        for entity, row in records.items()
    }


def _flush_parquet_at_exit(ref):
    """atexit hook for a ``ParquetStorage``, held weakly so the component can still be collected."""
    storage = ref()
    if storage is not None:
        storage._close_quietly()


class ParquetStorage(Storage):
    """Collect every Result of one run into a single Parquet file, ready to bulk load elsewhere.

    Each result that passes through becomes one row; the file is written when the run finishes (the
    component subscribes to the workflow's ``component_end``/``component_error`` event, so a run that
    dies part way through still leaves the rows it got to). By default each row is a row of the
    CHAI-TEA RESULT table (see ``result_row``) and the file is typed by that table's columns
    (``RESULT_COLUMNS``), so it loads into the central database as it stands -- a run that produced
    nothing still leaves an empty file with those columns. With ``fields`` set, columns and their
    types are inferred from the data unless declared in ``schema``; dicts, lists and columns of mixed
    types become JSON-encoded text columns, which is what a loader on the far side can parse
    consistently. See ``ParquetRecordWriter`` for the column kinds and how batching interacts with
    type inference.

    Unlike ``SqliteStorage`` this stores one row per result and no derivative results: it is a table
    of a run's output, not a browsable object graph. Rows are held in memory until the file is
    written, so for a run too large for that use ``JsonLinesStorage`` (which writes as it goes) and
    convert its output afterwards with ``jsonl_to_parquet``, or set ``batch_size`` to write row
    groups during the run.

    Returns the input unchanged. Writing needs ``pyarrow``, imported when the first row is written.

    Settings:
        - file: path of the .parquet file (default 'results.parquet'); ``{run_id}`` and
          ``{timestamp}`` in the path are substituted per run, which is how parallel runs (e.g. the
          slices of a ``SliceIterator``) each get their own file. Without a placeholder, a later run
          of the same workflow overwrites the file.
        - run_id: identifier for this run, recorded in every row (as ``workflow_run_id``, hashed into
          a uuid if it is not one) and in the file's metadata (default: the workflow's ``run_id``,
          fresh for each run)
        - fields: leave unset for CHAI-TEA RESULT rows, which is what a run's output must be. Set only
          for a file meant for something else: a list of result-JSON keys to keep, or a
          ``{key: name}`` dict that keeps and renames them, in which a key from ``COMPUTED_FIELDS``
          (``@record``, ``@input_uri``, ``@input_hash``, ``@corrects``) names a column computed from
          the result rather than read off its JSON. The settings below marked *fields only* apply
          only then
        - constants: ``{key: value}`` recorded in every row -- a batch name, a source system, the slice
          number -- in its ``extra_data`` (with ``fields`` set: as columns of their own)
        - sources: ``{key: component id}`` dict; for each entry, the value of the nearest ancestor
          result that component produced is recorded under *key* (see ``source_value``), in
          ``extra_data`` (with ``fields`` set: as a column of its own)
        - input_source: component id whose result counts as the input for ``input``/``input_hash``
          (default: the component marked ``source: true``, else the result's immediate input)
        - run_columns (*fields only*): add the ``run_id`` and ``stored_at`` columns to every row
          (default true)
        - schema (*fields only*): ``{column: type}`` declaring the column types and their order, with
          type one of string, json, int, float, bool, timestamp. Columns not listed are inferred;
          listed columns missing from the data become null columns, so every run yields the same table
        - null_if_empty (*fields only*): write an empty dict, list or string as null rather than as
          ``{}``/``[]``/``""`` (default false) -- which is what a table that treats "nothing here" as
          NULL wants, and what ``SqliteStorage`` and ``PostgresStorage`` do with an empty ``metadata``
        - batch_size: rows to buffer before writing a row group; 0 (the default) writes one row group
          at the end of the run, having inferred the column types from all of it
        - compression: Parquet codec, e.g. 'snappy' (default), 'zstd', 'gzip', 'none'
        - flush_on_exit: also write the file at interpreter exit if the run never ended cleanly
          (default true)

    With RESULT rows the file's metadata also carries ``chai_tea_records``: the PROJECT, WORKFLOW and
    WORKFLOW_RUN rows of the run as JSON, so a loader can create them before loading the rows that
    reference them.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "file" not in self.settings:
            self.settings["file"] = "results.parquet"
        self.file_template = os.path.abspath(self.settings["file"])
        parent_dir = os.path.dirname(self.file_template)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        # Without fields the rows are RESULT rows, whose workflow_run_id already says which run
        self.result_rows = not self.settings.get("fields")
        self.run_columns = self.settings.get("run_columns", True) and not self.result_rows
        self._lock = threading.Lock()
        self._writer = None
        self.run_id = None
        self.file_name = None
        self._run_started = False
        self._start_run()
        if hasattr(self.workflow, "add_listener"):  # a Workflow; the file is written when its run ends
            self.workflow.add_listener(self._on_workflow_event)
        if self.settings.get("flush_on_exit", True):
            # A weak reference, so a workflow that is built and dropped is not kept alive until exit
            atexit.register(_flush_parquet_at_exit, weakref.ref(self))

    def _start_run(self):
        """Pick the run id and resolve the file name for the run that is about to write."""
        self.run_id = str(_run_id_of(self) or uuid.uuid4())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.file_name = self.file_template.replace("{run_id}", self.run_id).replace("{timestamp}", stamp)
        self._run_started = False

    def _on_workflow_event(self, payload):
        """Note that a run began, and close the file when the workflow finishes -- successfully or not."""
        if payload.get("component_id") != self.workflow.id:
            return
        event = payload.get("event")
        if event == "component_start":
            if self._writer is None:
                # The workflow has just picked this run's id; the rows and the file name follow it
                self._start_run()
            self._run_started = True
        elif event in ("component_end", "component_error"):
            self.close()

    def _close_quietly(self):
        """``close`` for the atexit hook: a failure there must not mask what ended the process."""
        try:
            self.close()
        except Exception as e:  # pragma: no cover - interpreter shutdown
            logger.warning(f"Could not write Parquet file {self.file_name}: {e}")

    def build_json(self, input: Result):
        """The row for one result: a RESULT row, or the selected fields, constants and run columns."""
        if self.result_rows:
            return build_record(
                input,
                sources=self.settings.get("sources"),
                input_source=self.settings.get("input_source"),
                run_id=self.run_id,
                constants=self.settings.get("constants"),
            )
        js = build_record(
            input, self.settings.get("fields"), self.settings.get("sources"), self.settings.get("input_source")
        )
        if self.settings.get("null_if_empty"):
            js = {column: (None if value in ({}, [], "") else value) for column, value in js.items()}
        for column, value in (self.settings.get("constants") or {}).items():
            js[column] = value
        if self.run_columns:
            js["run_id"] = self.run_id
            js["stored_at"] = datetime.now(timezone.utc)
        return js

    def _declared_schema(self):
        """The configured ``schema``, extended with the columns this component adds to every row.

        Only when a schema was declared at all: it is then the contract for the table on the far
        side, so the constants and run columns belong in it -- otherwise a run that produced no rows
        would leave a file missing columns that every other run has.
        """
        if self.result_rows:
            return dict(RESULT_COLUMNS)
        schema = self.settings.get("schema")
        if not schema:
            return None
        schema = dict(schema)
        for column, value in (self.settings.get("constants") or {}).items():
            schema.setdefault(column, _value_kind(value) or "string")
        if self.run_columns:
            schema.setdefault("run_id", "string")
            schema.setdefault("stored_at", "timestamp")
        return schema

    def _new_writer(self):
        """A writer for the current run, tagging the file with the run's provenance."""
        return ParquetRecordWriter(
            self.file_name,
            schema=self._declared_schema(),
            batch_size=self.settings.get("batch_size", 0),
            compression=self.settings.get("compression", "snappy"),
            metadata={
                "chai_run_id": self.run_id,
                "chai_workflow_id": self.workflow.id if self.workflow else "",
                "chai_component_id": self.id,
                "chai_started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def close(self):
        """Write the Parquet file for the current run; returns ``{"path", "rows", "columns"}``.

        Called automatically when the workflow finishes. Call it directly when results are pushed
        through the component outside a ``Workflow.run``. Afterwards the component starts a new run,
        so re-running the same workflow object writes a fresh file rather than appending to a closed
        one. A run that produced no rows writes no file unless its columns are known -- RESULT rows, or a
        ``schema`` saying what the file holds --
        in which case it leaves an empty one with those columns.
        """
        with self._lock:
            writer, self._writer = self._writer, None
            if writer is None:
                # A run that produced nothing still leaves an empty, correctly typed file when the
                # schema says what the file holds -- but only for a run that actually happened, so a
                # second close (the atexit hook after the workflow's own) writes nothing.
                if not (self._run_started and self._declared_schema()):
                    return None
                writer = self._new_writer()
            if self.result_rows and hasattr(self.workflow, "records"):
                # The rows the file's RESULT rows hang under, for whoever loads it: a loader creates
                # them first (``store_postgres_rows``). With the default single row group the file is
                # written now, so they include the finished run.
                writer.metadata["chai_tea_records"] = json.dumps(_json_safe(_isoformat_rows(self.workflow.records())))
            info = writer.close()
            logger.info(f"Wrote {info['rows']} rows to {info['path']}")
            self._start_run()
            return info

    def _process(self, input: Result) -> Result:
        """Add the result as a row of the run's Parquet file, then pass it through unchanged."""
        with self._lock:
            if self._writer is None:
                self._writer = self._new_writer()
            writer = self._writer
        writer.add(self.build_json(input))
        return input


def jsonl_to_parquet(jsonl_file, parquet_file, schema=None, batch_size=50000, compression="snappy", metadata=None):
    """Convert a JSON-Lines file (see ``JsonLinesStorage``) into one Parquet file.

    The counterpart to ``ParquetStorage`` for runs too large to hold in memory: stream the results to
    JSON-Lines as they are produced, then pack the finished file into Parquet for the bulk load.
    Reads and writes in batches of *batch_size* lines, so neither file is ever fully in memory; as
    with ``ParquetRecordWriter``, the column types come from the first batch unless declared in
    *schema*. A file of CHAI-TEA RESULT rows (what the storages write by default) is typed by
    ``RESULT_COLUMNS`` instead, so its JSON columns stay JSON even where every value in them happens to
    be text. Blank lines are skipped and an unparseable line raises. Returns
    ``{"path", "rows", "columns"}``.
    """
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


#: The zero value of a usage total, and the order the counts are reported in.
_USAGE_FIELDS = ("calls", "input", "text", "images", "thinking", "output", "total", "duration")


def _usage_of(record, metadata_key="metadata"):
    """The ``token_usage`` dict and duration recorded on one stored record, or ``None``.

    Reads the metadata an AI component puts on its result (see ``GeminiComponent.get_usage``),
    wherever the storage put it: the record's own ``metadata``, or the whole-result ``@record``
    column the tabular storages write.
    """
    for holder in (record.get(metadata_key), record.get("@record"), record.get("value_json")):
        if isinstance(holder, str):
            try:
                holder = json.loads(holder)
            except ValueError:
                continue
        if isinstance(holder, dict):
            meta = holder.get("metadata", holder) if "token_usage" not in holder else holder
            if isinstance(meta, dict) and isinstance(meta.get("token_usage"), dict):
                return meta["token_usage"], meta.get("duration", 0) or 0
    return None


def _add_usage(totals, usage, duration):
    """Add one call's counts to a running total, filling in what the API did not break down."""
    def count(key):
        value = usage.get(key, 0)
        return value if isinstance(value, (int, float)) and value > 0 else 0

    text, images = count("prompt"), count("images")
    thinking, output, total = count("thinking"), count("result"), count("total")
    sent = text + images
    if not sent and total:
        # No per-modality breakdown in the response: what is left of the total is what went in
        sent = max(total - thinking - output, 0)
    if not total:
        total = sent + thinking + output
    totals["calls"] += 1
    totals["input"] += sent
    totals["text"] += text
    totals["images"] += images
    totals["thinking"] += thinking
    totals["output"] += output
    totals["total"] += total
    totals["duration"] += duration


def token_usage_summary(
    jsonl_file, step_key=None, metadata_key="metadata", input_price=None, output_price=None
):
    """Total the tokens (and optionally the cost) an AI run recorded, per component and overall.

    Every AI component records what a call used in its result's ``token_usage`` metadata, so a run
    that persisted its results -- ``JsonLinesStorage`` is the cheapest way, one line per result as it
    is produced -- can be totalled afterwards without re-reading anything from the API. Lines with no
    token usage (deterministic steps, and calls whose response carried no usage metadata) are counted
    in ``without_usage`` rather than silently ignored, so a suspiciously cheap-looking run is visible.

    Returns ``{"components": {id: counts}, "total": counts, "records", "without_usage"}``, where each
    counts dict holds ``calls``, ``input`` (everything sent: ``text`` + ``images``), ``thinking``,
    ``output``, ``total`` and ``duration`` in seconds. *input_price* and *output_price*, in dollars
    per million tokens, add a ``cost`` to each: thinking tokens are billed as output. Take the prices
    from the provider's own pricing page for the exact model -- they are not guessed here.
    """
    components = {}
    records = 0
    without_usage = 0
    with open(jsonl_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records += 1
            record = json.loads(line)
            found = _usage_of(record, metadata_key)
            if found is None:
                without_usage += 1
                continue
            step = record.get(step_key) if step_key else None
            if step is None:
                step = record.get("process_id", record.get("processorId", record.get("step", "unknown")))
            totals = components.setdefault(str(step), dict.fromkeys(_USAGE_FIELDS, 0))
            _add_usage(totals, *found)

    total = dict.fromkeys(_USAGE_FIELDS, 0)
    for counts in components.values():
        for key in _USAGE_FIELDS:
            total[key] += counts[key]
    if input_price is not None or output_price is not None:
        for counts in list(components.values()) + [total]:
            counts["cost"] = (counts["input"] * (input_price or 0) / 1_000_000) + (
                (counts["output"] + counts["thinking"]) * (output_price or 0) / 1_000_000
            )
    return {
        "components": components,
        "total": total,
        "records": records,
        "without_usage": without_usage,
    }


def _pg_driver():
    """The installed psycopg module -- version 3 if present, else psycopg2."""
    try:
        import psycopg

        return psycopg
    except ImportError:
        pass
    try:
        import psycopg2

        return psycopg2
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("Talking to PostgreSQL requires psycopg: pip install psycopg[binary]") from e


def postgres_params(settings=None, **overrides):
    """Connection keywords for the chai database, from *settings*, the environment and the defaults.

    A ``dsn`` setting (a libpq connection string or ``postgresql://`` URL) is passed through as-is and
    used instead of the individual keywords. Otherwise ``host``, ``port``, ``database``, ``user`` and
    ``password`` are taken from the settings, falling back to the standard ``PGHOST``/``PGPORT``/
    ``PGDATABASE``/``PGUSER``/``PGPASSWORD`` environment variables and then to localhost:5432 and the
    ``chai`` database. Its own output is accepted too (``dbname``, ``conninfo``), so normalizing
    settings twice gives the same connection rather than quietly falling back to the default database.
    """
    settings = dict(settings or {})
    settings.update(overrides)
    dsn = settings.get("dsn") or settings.get("conninfo")
    if dsn:
        return {"conninfo": dsn}
    params = {
        "host": settings.get("host") or os.environ.get("PGHOST") or "localhost",
        "port": int(settings.get("port") or os.environ.get("PGPORT") or 5432),
        "dbname": settings.get("database") or settings.get("dbname") or os.environ.get("PGDATABASE") or "chai",
    }
    user = settings.get("user") or os.environ.get("PGUSER")
    if user:
        params["user"] = user
    password = settings.get("password") or os.environ.get("PGPASSWORD")
    if password:
        params["password"] = password
    return params


def _pg_connect(params, autocommit=False):
    """Open a connection from the keywords ``postgres_params`` produced (psycopg 3 or psycopg2)."""
    driver = _pg_driver()
    params = dict(params)
    conninfo = params.pop("conninfo", None)
    if driver.__name__ == "psycopg2":
        # psycopg2 takes the connection string positionally and calls the database 'dbname' too
        conn = driver.connect(conninfo) if conninfo else driver.connect(**params)
        conn.autocommit = autocommit
        return conn
    if conninfo:
        return driver.connect(conninfo, autocommit=autocommit)
    return driver.connect(autocommit=autocommit, **params)


def ensure_postgres_database(params=None, **overrides):
    """Create the configured database if it does not exist yet; returns its name.

    Connects to the server's ``postgres`` maintenance database to do it, so it needs an account
    allowed to create databases -- the counterpart of SQLite's file appearing on first use. A server
    that cannot be reached, or an account that may not create databases, raises.
    """
    params = postgres_params(params, **overrides)
    if "conninfo" in params:
        conn = _pg_connect(params)  # a dsn names its own database; nothing to create
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT current_database()")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    database = params["dbname"]
    try:
        _pg_connect(params).close()
        return database
    except Exception as e:
        if "does not exist" not in str(e):
            raise
    admin = dict(params, dbname="postgres")
    conn = _pg_connect(admin, autocommit=True)  # CREATE DATABASE cannot run inside a transaction
    try:
        conn.cursor().execute(f'CREATE DATABASE "{database}"')
        logger.info(f"Created PostgreSQL database {database}")
    finally:
        conn.close()
    return database


# --------------------------------------------------------------------------------------------------
# Writing CHAI-TEA rows into a database -- PostgreSQL or SQLite, the same tables either way
# --------------------------------------------------------------------------------------------------

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
        value = _as_datetime(value)
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


def _pg_ensure_chai_tea(conn, tables=None):
    """Create the CHAI-TEA tables in *conn*'s database unless they are all there; returns their names.

    A table of one of the names that has other columns -- the pre-CHAI-TEA ``results`` table, say --
    raises rather than being written into (see ``chai.schema.check_for_clashes``).
    """
    names = table_names(tables)
    check_for_clashes(conn, names)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = ANY(%s)",
        (list(names.values()),),
    )
    if cursor.fetchone()[0] < len(names):
        for statement in schema_statements(tables):
            cursor.execute(statement)
    conn.commit()
    return names


def ensure_postgres_schema(params=None, tables=None, create_database=True, **overrides):
    """Create the CHAI-TEA tables (and, unless told not to, the database) in PostgreSQL; returns the names.

    Lets an app -- or the loader that follows a Parquet upload -- prepare the tables before anything
    has run, the PostgreSQL counterpart of ``ensure_database``. *tables* overrides table names.
    """
    params = postgres_params(params, **overrides)
    if create_database:
        ensure_postgres_database(params)
    conn = _pg_connect(params)
    try:
        return _pg_ensure_chai_tea(conn, tables)
    finally:
        conn.close()


def store_postgres_rows(records, params=None, tables=None, create_schema=True, **overrides):
    """Upsert ``{entity: row or [rows]}`` into the CHAI-TEA tables in PostgreSQL; returns the count.

    How rows made elsewhere get in: ``workflow.records()`` before ``parquet_to_postgres`` loads a run's
    RESULT rows, say, so the rows they reference are there first.
    """
    params = postgres_params(params, **overrides)
    if create_schema:
        ensure_postgres_database(params)
    conn = _pg_connect(params)
    try:
        names = _pg_ensure_chai_tea(conn, tables) if create_schema else table_names(tables)
        written = write_rows(conn.cursor(), "postgres", names, records)
        conn.commit()
        return written
    finally:
        conn.close()


class _ChaiTeaStorage(Storage):
    """What ``SqliteStorage`` and ``PostgresStorage`` share: writing a run into the CHAI-TEA tables.

    Each result becomes a RESULT row (``result_row``), and so does each of its ``derivative_results``
    -- a derivative is a result like any other. The first time a run writes, the rows its results
    hang under go first: the workflow's PROJECT, WORKFLOW and WORKFLOW_RUN (``Workflow.records``).
    When the run ends the WORKFLOW_RUN row is written again, now with its outcome and duration.
    """

    dialect = ""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.tables = self.settings.get("tables") or None
        self.names = table_names(self.tables)
        self._lock = threading.Lock()
        self._run_written = None
        if hasattr(self.workflow, "add_listener"):  # a Workflow; record the run's outcome when it ends
            self.workflow.add_listener(self._on_workflow_event)

    def _on_workflow_event(self, payload):
        if payload.get("component_id") == self.workflow.id and payload.get("event") in (
            "component_end",
            "component_error",
        ):
            self._end_run()

    def records_for(self, input: Result):
        """The rows storing *input* writes: its RESULT row and its derivatives', and the run's if new."""
        run_id = getattr(self.workflow, "run_id", None)
        if run_id is None or not hasattr(self.workflow, "records"):
            raise ValueError(f"{self} must be part of a Workflow: every RESULT row belongs to a WORKFLOW_RUN")
        with_hash = self.settings.get("hash_input", True)
        rows = [result_row(input, run_id, input_source=self.settings.get("input_source"), with_hash=with_hash)]
        for results in input.derivative_results.values():
            rows.extend(result_row(result, run_id, with_hash=with_hash) for result in results)
        records = {"RESULT": rows}
        if self._run_written != run_id:
            records.update(self.workflow.records())
        return records

    def _write(self, records):
        raise NotImplementedError

    def _end_run(self):
        """Write the finished WORKFLOW_RUN (and WORKFLOW) rows for a run this storage wrote into."""
        with self._lock:
            if self._run_written is not None and self._run_written == self.workflow.run_id:
                records = self.workflow.records()
                self._write({"WORKFLOW": records["WORKFLOW"], "WORKFLOW_RUN": records["WORKFLOW_RUN"]})

    def _process(self, input: Result) -> Result:
        with self._lock:
            records = self.records_for(input)
            self._write(records)
            self._run_written = self.workflow.run_id
        return input


class PostgresStorage(_ChaiTeaStorage):
    """Store results in PostgreSQL, in the CHAI-TEA schema (``chai.schema``).

    Each result is a row of the RESULT table and each of its derivatives another; the run's PROJECT,
    WORKFLOW and WORKFLOW_RUN rows are written before its first result and the WORKFLOW_RUN row again
    when it ends (see ``_ChaiTeaStorage``). Every value comes from the objects themselves -- the
    workflow's name, description and project, the run's timing and outcome, each result's
    timestamps -- so the rows are the same ones ``JsonLinesStorage``/``ParquetStorage`` write to
    files. Storing a result again is an upsert that keeps its ``created``; a correction is a row of its
    own whose ``previous_id`` is the row it corrects (see ``save_postgres_correction``).

    The database and the tables are created on first use if they are missing, so a workflow can be
    pointed at a fresh server; a database holding a table of the same name in another layout is
    refused rather than written into. One connection is held per component and shared by the run's
    threads under a lock (an ``Iterator`` with ``workers`` serializes on it); a broken connection is
    reopened on the next result rather than failing the run, and the connection is given back when the
    run ends (``close`` does it by hand for results pushed through outside a ``Workflow.run``).

    Needs psycopg (version 3, or psycopg2), imported when the first result is written.

    Settings:
        - dsn: full connection string or ``postgresql://`` URL; overrides the keywords below
        - host / port / database / user / password: connection keywords, defaulting to the
          ``PG*`` environment variables and then to localhost:5432 and the ``chai`` database
        - tables: ``{entity: table name}`` overriding the CHAI-TEA table names (``chai.schema.TABLES``)
        - create_database: create the database if the server does not have it yet (default true)
        - create_schema: create the CHAI-TEA tables if they are missing (default true)
        - input_source: component id whose result counts as the input the row was generated from,
          overriding the component marked ``source: true`` (default: the marked component's result,
          or the result's immediate input when nothing is marked)
        - hash_input: fill in ``input_hash`` (default true); false skips the digest, and any file
          read it would need
    """

    dialect = "postgres"

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.params = postgres_params(self.settings)
        self._conn = None

    def _connection(self):
        """The component's connection, opened (and the schema ensured) on first use."""
        if self._conn is None:
            if self.settings.get("create_database", True):
                ensure_postgres_database(self.params)
            self._conn = _pg_connect(self.params)
            if self.settings.get("create_schema", True):
                _pg_ensure_chai_tea(self._conn, self.tables)
        return self._conn

    def close(self):
        """Close the connection; the next result opens a new one."""
        with self._lock:
            self._close_connection()

    def _close_connection(self):
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - already broken
                pass

    def _write_once(self, records):
        conn = self._connection()
        try:
            write_rows(conn.cursor(), self.dialect, self.names, records)
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:  # pragma: no cover - the connection is gone
                pass
            raise

    def _write(self, records):
        """Write *records*; assumes the caller holds the lock."""
        try:
            self._write_once(records)
        except Exception:
            # A connection that died between results (server restart, idle timeout) should cost one
            # retry, not the rest of the run; a fresh connection fails the same way on a real error
            if self._conn is None or not getattr(self._conn, "closed", False):
                raise
            self._close_connection()
            self._write_once(records)

    def _end_run(self):
        try:
            super()._end_run()
        finally:
            self.close()


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


def save_postgres_correction(
    result_id,
    corrected_value,
    agent=None,
    metadata=None,
    process_id=None,
    editor_user_id=None,
    params=None,
    tables=None,
    **overrides,
):
    """Record a correction of *result_id* in PostgreSQL as a new RESULT row pointing back at it.

    The PostgreSQL counterpart of ``save_correction``: the original row is never touched, the
    correction is an entry of its own whose ``previous_id`` is the corrected row and whose metadata
    says who made it -- *agent* (the person or model, recorded as ``metadata['agent']``) plus anything
    else in *metadata*. See ``correction_row``. Returns the new row's id, or ``None`` if *result_id*
    is not in the table.
    """
    names = table_names(tables)
    conn = _pg_connect(postgres_params(params, **overrides))
    try:
        cursor = conn.cursor()
        columns = list(column_types("RESULT"))
        cursor.execute(f"SELECT {', '.join(columns)} FROM {names['RESULT']} WHERE id = %s::uuid", (as_uuid(result_id),))
        found = cursor.fetchone()
        if found is None:
            return None
        original = {column: (str(value) if column.endswith("id") and value is not None else value)
                    for column, value in zip(columns, found)}
        row = correction_row(original, corrected_value, _correction_metadata(agent, metadata), process_id, editor_user_id)
        write_rows(cursor, "postgres", names, {"RESULT": row})
        conn.commit()
        return row["id"]
    finally:
        conn.close()


_PG_TYPES = {
    "json": "JSONB",
    "string": "TEXT",
    "int": "BIGINT",
    "float": "DOUBLE PRECISION",
    "bool": "BOOLEAN",
    "timestamp": "TIMESTAMPTZ",
}


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
        if kind not in _PG_TYPES:
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


def parquet_to_postgres(
    parquet_file,
    table,
    params=None,
    create=True,
    truncate=False,
    batch_size=10000,
    create_database=True,
    **overrides,
):
    """Load a Parquet file written by ``ParquetStorage`` into a PostgreSQL table.

    The step after a run: the file holding everything one run produced is bulk loaded into the remote
    database in row-group-sized batches, with ``COPY`` where the driver supports it. The table is
    created from the file's own columns if it does not exist yet -- ``json`` columns become ``jsonb``,
    so the loaded values are queryable rather than text -- and a table that already exists is loaded
    into as it stands, which is how a Parquet run can land in the same shape ``PostgresStorage``
    writes (see ``ensure_postgres_schema``). A column the table does not have raises rather than
    being silently dropped.

    Returns ``{"table", "rows", "columns"}``. Arguments:
        - parquet_file / table: what to load and where
        - params / dsn / host / port / database / user / password: where the server is (see
          ``postgres_params``)
        - create: create the table from the file's columns when it is missing (default true)
        - truncate: empty the table before loading, so a re-load replaces rather than adds
        - batch_size: rows per batch read from the file and sent to the server
        - create_database: create the database if the server does not have it yet (default true)
    """
    import pyarrow.parquet as pq

    table = _pg_identifier(table)
    kinds = parquet_columns(parquet_file)
    params = postgres_params(params, **overrides)
    if create_database:
        ensure_postgres_database(params)
    conn = _pg_connect(params)
    try:
        cursor = conn.cursor()
        if create:
            columns = ", ".join(f"{_pg_identifier(c)} {_PG_TYPES[k]}" for c, k in kinds.items())
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")
            conn.commit()
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,))
        existing = {row[0] for row in cursor.fetchall()}
        missing = [c for c in kinds if c not in existing]
        if missing:
            raise ValueError(f"Table {table} has no column(s) {', '.join(missing)} for this Parquet file")
        if truncate:
            cursor.execute(f"TRUNCATE TABLE {table}")
        names = list(kinds)
        quoted = ", ".join(_pg_identifier(c) for c in names)
        rows = 0
        for batch in pq.ParquetFile(parquet_file).iter_batches(batch_size=batch_size, columns=names):
            records = batch.to_pylist()
            if not records:
                continue
            rows += _pg_copy_rows(cursor, table, quoted, names, records, kinds)
        conn.commit()
    finally:
        conn.close()
    logger.info(f"Loaded {rows} rows from {parquet_file} into {table}")
    return {"table": table, "rows": rows, "columns": kinds}


def _pg_copy_rows(cursor, table, quoted, names, records, kinds):
    """Send one batch of records to *table*; ``COPY`` on psycopg 3, batched INSERTs on psycopg2."""
    values = [[record.get(name) for name in names] for record in records]
    if hasattr(cursor, "copy"):
        # Text-format COPY: each value is written as text and parsed by the column's own type, so a
        # json column's text lands in jsonb without a round trip through Python objects
        with cursor.copy(f"COPY {table} ({quoted}) FROM STDIN") as copy:
            for row in values:
                copy.write_row(row)
    else:  # psycopg2
        casts = ", ".join("%s::jsonb" if kinds[name] == "json" else "%s" for name in names)
        cursor.executemany(f"INSERT INTO {table} ({quoted}) VALUES ({casts})", values)
    return len(values)


def _ensure_schema(conn, tables=None):
    """Create the CHAI-TEA tables in a SQLite database if they are missing; returns their names.

    The same tables as in PostgreSQL (``chai.schema.sqlite_statements``). A table of one of the names
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


def _loads(text):
    """Parse a JSON column value, passing ``None`` through and tolerating non-JSON content."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


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


class SqliteStorage(_ChaiTeaStorage):
    """Store results in a SQLite database, in the CHAI-TEA schema (``chai.schema``).

    The same tables and rows as ``PostgresStorage``, in SQLite's dialect: each result and each of its
    derivatives is a RESULT row, under the run's PROJECT, WORKFLOW and WORKFLOW_RUN rows (see
    ``_ChaiTeaStorage``), and corrections are rows of their own (``save_correction``). The viewer
    helpers in this module (``list_results``, ``get_result``, ``list_processors``) read it back. The
    schema is created on first use, a fresh connection is opened per operation (thread-safe for use
    from e.g. Flask), foreign keys are enforced, and the input is returned unchanged.

    Settings:
        - database: path of the SQLite database file (default 'results.db')
        - tables: ``{entity: table name}`` overriding the CHAI-TEA table names
        - input_source: component id whose result counts as the input the row was generated from,
          overriding the component marked ``source: true`` (default: the marked component's result,
          or the result's immediate input when nothing is marked)
        - hash_input: fill in ``input_hash`` (default true); false skips the digest, and any file
          read it would need
    """

    dialect = "sqlite"

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "database" not in self.settings:
            self.settings["database"] = "results.db"

    def _write(self, records):
        conn = _connect(self.settings["database"], self.tables)
        try:
            write_rows(conn.cursor(), self.dialect, self.names, records)
            conn.commit()
        finally:
            conn.close()


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
