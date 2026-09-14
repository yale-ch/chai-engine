"""Storage components: persist Results to the filesystem or a database.

Storage steps are pass-through: they write the input Result somewhere and return it unchanged, so
they can be inserted anywhere in a pipeline (typically as ``next_steps``) without affecting the data
flow.

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
from .result import FileItemResult

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
    too -- and the storages record it in the ``corrects_id`` column, which is what makes the
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


def build_record(result: Result, fields=None, sources=None, input_source=None):
    """The flat, JSON-safe record for *result*: its selected fields plus any configured sources.

    Shared by the storage components that write one record per result (``JsonLinesStorage``,
    ``ParquetStorage``) so a run can be streamed to JSON-Lines or packed into Parquet with the same
    columns. *fields* selects keys of the ``result_to_json`` representation -- a list to keep them as
    they are, a ``{key: name}`` dict to keep and rename them; ``None`` keeps all of them.

    In the dict form a key may also be one of the computed ``COMPUTED_FIELDS``, which is how a record
    can carry the same columns ``SqliteStorage`` and ``PostgresStorage`` write:

    * ``@record`` -- the whole result JSON (their ``value_json``)
    * ``@input_uri`` / ``@input_hash`` / ``@input_locator`` -- the file or URI the entry was generated
      from, the md5 of that input's content, and where in it this entry is (see ``input_provenance``;
      *input_source* names the component whose result counts as the input, overriding the component
      marked ``source: true``)
    * ``@corrects`` -- the id of the result this one corrects (see ``correction_target``)

    *sources* is a ``{key: component id}`` dict, each recording the value of the nearest ancestor
    result that component produced (see ``source_value``).
    """
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
          when the component is built, so re-running replaces the previous run's lines
        - fields: which parts of the result JSON to record -- a list of keys to keep, or a
          ``{key: name}`` dict that keeps and renames them, in which a key from ``COMPUTED_FIELDS``
          (``@record``, ``@input_uri``, ``@input_hash``, ``@corrects``) names a column computed from
          the result rather than read off its JSON (default: the whole result JSON split into its own
          columns, i.e. id, type, workflowId, processorId, metadata, extraInfo, input and value)
        - sources: ``{key: component id}`` dict; for each entry, the value of the nearest ancestor
          result that component produced is recorded under *key* (see ``source_value``). This is how
          a line can carry the input it was derived from next to the value derived from it
        - input_source: component id whose result counts as the input for ``@input_uri``/
          ``@input_hash`` (default: the result's immediate input)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "file" not in self.settings:
            self.settings["file"] = "results.jsonl"
        self.file_name = os.path.abspath(self.settings["file"])
        parent_dir = os.path.dirname(self.file_name)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        # Opened once, when the workflow is built, rather than on the first write -- so a run that
        # produces nothing still leaves an (empty) file, and 'truncate' still clears the last run's
        mode = "w" if str(self.settings.get("mode", "append")).lower() == "truncate" else "a"
        with _line_lock(self.file_name):
            with open(self.file_name, mode):
                pass

    def build_json(self, input: Result):
        """The record for one line: the selected result fields plus any configured source values."""
        return build_record(
            input,
            self.settings.get("fields", None),
            self.settings.get("sources", None),
            self.settings.get("input_source", None),
        )

    def _process(self, input: Result) -> Result:
        """Append the result to the JSON-Lines file, then pass it through unchanged."""
        line = json.dumps(self.build_json(input), ensure_ascii=False) + "\n"
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


def _flush_parquet_at_exit(ref):
    """atexit hook for a ``ParquetStorage``, held weakly so the component can still be collected."""
    storage = ref()
    if storage is not None:
        storage._close_quietly()


class ParquetStorage(Storage):
    """Collect every Result of one run into a single Parquet file, ready to bulk load elsewhere.

    Each result that passes through becomes one row; the file is written when the run finishes (the
    component subscribes to the workflow's ``component_end``/``component_error`` event, so a run that
    dies part way through still leaves the rows it got to). Columns and their types are inferred from
    the data unless declared in ``schema``; dicts, lists and columns of mixed types become
    JSON-encoded text columns, which is what a loader on the far side can parse consistently. See
    ``ParquetRecordWriter`` for the column kinds and how batching interacts with type inference.

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
        - run_id: identifier for this run, recorded in every row and in the file's metadata
          (default: a generated hex id, fresh for each run)
        - run_columns: add the ``run_id`` and ``stored_at`` columns to every row (default true)
        - constants: ``{column: value}`` written unchanged into every row -- a batch name, a source
          system, the slice number -- so the loaded table can be filtered by them
        - schema: ``{column: type}`` declaring the column types and their order, with type one of
          string, json, int, float, bool, timestamp. Columns not listed are inferred; listed columns
          missing from the data become null columns, so every run yields the same table
        - fields: which parts of the result JSON to record -- a list of keys to keep, or a
          ``{key: name}`` dict that keeps and renames them, in which a key from ``COMPUTED_FIELDS``
          (``@record``, ``@input_uri``, ``@input_hash``, ``@corrects``) names a column computed from
          the result rather than read off its JSON (default: the whole result JSON split into its own
          columns, i.e. id, type, workflowId, processorId, metadata, extraInfo, input and value)
        - sources: ``{column: component id}`` dict; for each entry, the value of the nearest ancestor
          result that component produced is recorded in that column (see ``source_value``)
        - input_source: component id whose result counts as the input for ``@input_uri``/
          ``@input_hash`` (default: the result's immediate input)
        - null_if_empty: write an empty dict, list or string as null rather than as ``{}``/``[]``/``""``
          (default false) -- which is what a table that treats "nothing here" as NULL wants, and what
          ``SqliteStorage`` and ``PostgresStorage`` do with an empty ``metadata``
        - batch_size: rows to buffer before writing a row group; 0 (the default) writes one row group
          at the end of the run, having inferred the column types from all of it
        - compression: Parquet codec, e.g. 'snappy' (default), 'zstd', 'gzip', 'none'
        - flush_on_exit: also write the file at interpreter exit if the run never ended cleanly
          (default true)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "file" not in self.settings:
            self.settings["file"] = "results.parquet"
        self.file_template = os.path.abspath(self.settings["file"])
        parent_dir = os.path.dirname(self.file_template)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.run_columns = self.settings.get("run_columns", True)
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
        self.run_id = str(self.settings.get("run_id") or uuid.uuid4().hex)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.file_name = self.file_template.replace("{run_id}", self.run_id).replace("{timestamp}", stamp)
        self._run_started = False

    def _on_workflow_event(self, payload):
        """Note that a run began, and close the file when the workflow finishes -- successfully or not."""
        if payload.get("component_id") != self.workflow.id:
            return
        event = payload.get("event")
        if event == "component_start":
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
        """The row for one result: the selected result fields, the constants and the run columns."""
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
        one. A run that produced no rows writes no file unless ``schema`` says what the file holds,
        in which case it leaves an empty one with those columns.
        """
        with self._lock:
            writer, self._writer = self._writer, None
            if writer is None:
                # A run that produced nothing still leaves an empty, correctly typed file when the
                # schema says what the file holds -- but only for a run that actually happened, so a
                # second close (the atexit hook after the workflow's own) writes nothing.
                if not (self._run_started and self.settings.get("schema")):
                    return None
                writer = self._new_writer()
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
    *schema*. Blank lines are skipped and an unparseable line raises. Returns
    ``{"path", "rows", "columns"}``.
    """
    writer = ParquetRecordWriter(
        parquet_file, schema=schema, batch_size=batch_size, compression=compression, metadata=metadata
    )
    with open(jsonl_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                writer.add(json.loads(line))
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
                step = record.get("processorId", record.get("step", "unknown"))
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


_PG_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _pg_identifier(name):
    """Validate a table or column name coming from configuration before it is put into SQL."""
    name = str(name)
    if not _PG_IDENTIFIER.match(name):
        raise ValueError(f"{name!r} is not a usable PostgreSQL identifier")
    return name


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
    ``chai`` database.
    """
    settings = dict(settings or {})
    settings.update(overrides)
    if settings.get("dsn"):
        return {"conninfo": settings["dsn"]}
    params = {
        "host": settings.get("host") or os.environ.get("PGHOST") or "localhost",
        "port": int(settings.get("port") or os.environ.get("PGPORT") or 5432),
        "dbname": settings.get("database") or os.environ.get("PGDATABASE") or "chai",
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


def _pg_ensure_schema(conn, table="results", derivatives_table=None):
    """Create the results/derivatives tables and their indexes in PostgreSQL if they are missing.

    The same tabular shape ``_ensure_schema`` builds for SQLite, with ``jsonb`` for the JSON columns.
    There is no migration from earlier layouts -- a database from before a schema change is dropped
    and rebuilt.
    """
    table = _pg_identifier(table)
    derivatives = _pg_identifier(derivatives_table or f"{table}_derivatives")
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            processor_id TEXT,
            workflow_id TEXT,
            value_json JSONB,
            metadata_json JSONB,
            extra_json JSONB,
            input_uri TEXT,
            input_hash TEXT,
            input_locator JSONB,
            corrects_id TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {derivatives} (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            component_id TEXT,
            result_json JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_processor ON {table}(processor_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_workflow ON {table}(workflow_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_corrects ON {table}(corrects_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_input_hash ON {table}(input_hash)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{derivatives}_source ON {derivatives}(source_id)")
    conn.commit()
    return table, derivatives


def ensure_postgres_schema(params=None, table="results", derivatives_table=None, create_database=True, **overrides):
    """Create the chai tables (and, unless told not to, the database) in PostgreSQL; returns the names.

    Lets an app -- or the loader that follows a Parquet upload -- prepare the tables before anything
    has run, the PostgreSQL counterpart of ``ensure_database``.
    """
    params = postgres_params(params, **overrides)
    if create_database:
        ensure_postgres_database(params)
    conn = _pg_connect(params)
    try:
        return _pg_ensure_schema(conn, table, derivatives_table)
    finally:
        conn.close()


class PostgresStorage(Storage):
    """Store results in PostgreSQL, in the same shape ``SqliteStorage`` uses for SQLite.

    Writes the input Result into a ``results`` table and each of its ``derivative_results`` into a
    ``results_derivatives`` table keyed by source result and component. ``value_json`` holds the full
    bytes-safe ``to_json(recurse=False)`` representation (id, type, value, metadata, provenance), with
    ``metadata_json``/``extra_json`` as dedicated columns for querying -- the JSON columns are
    ``jsonb``, so they can be indexed and queried in place. ``input_uri`` and ``input_hash`` record
    the file or URI the result was generated from and the md5 of that input's content, and
    ``corrects_id`` points at the row this result corrects, if it is a correction of one (see
    ``correction_target`` and ``save_correction``) -- corrections are entries in the table like any
    other, so the agent that made one, human or model, is described by that entry's own metadata. The
    ``processor_id``/``workflow_id`` columns are the ones in ``value_json``, so a result made by a
    component of a workflow counts as that workflow's even when nothing stamped the workflow onto the
    result itself. Storing a result again is an upsert that keeps the original ``created_at`` and the
    rows correcting it. The input is returned unchanged.

    The database and the tables are created on first use if they are missing, so a workflow can be
    pointed at a fresh server. One connection is held per component and shared by the run's threads
    under a lock (an ``Iterator`` with ``workers`` serializes on it); a broken connection is reopened
    on the next result rather than failing the run, and the connection is given back when the run
    ends (``close`` does it by hand for results pushed through outside a ``Workflow.run``).

    Needs psycopg (version 3, or psycopg2), imported when the first result is written.

    Settings:
        - dsn: full connection string or ``postgresql://`` URL; overrides the keywords below
        - host / port / database / user / password: connection keywords, defaulting to the
          ``PG*`` environment variables and then to localhost:5432 and the ``chai`` database
        - table: name of the results table (default 'results')
        - derivatives_table: name of the derivatives table (default '<table>_derivatives')
        - create_database: create the database if the server does not have it yet (default true)
        - input_source: component id whose result counts as the input the row was generated from,
          overriding the component marked ``source: true`` (default: the marked component's result,
          or the result's immediate input when nothing is marked)
        - hash_input: fill in ``input_hash`` (default true); false skips the digest, and any file
          read it would need
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.params = postgres_params(self.settings)
        self.table = _pg_identifier(self.settings.get("table", "results"))
        self.derivatives_table = _pg_identifier(
            self.settings.get("derivatives_table") or f"{self.table}_derivatives"
        )
        self._lock = threading.Lock()
        self._conn = None
        if hasattr(self.workflow, "add_listener"):  # a Workflow; give the connection back when it ends
            self.workflow.add_listener(self._on_workflow_event)

    def _on_workflow_event(self, payload):
        """Close the connection when the workflow finishes, rather than holding it until collection."""
        if payload.get("component_id") == self.workflow.id and payload.get("event") in (
            "component_end",
            "component_error",
        ):
            self.close()

    def _connection(self):
        """The component's connection, opened (and the schema ensured) on first use."""
        if self._conn is None:
            if self.settings.get("create_database", True):
                ensure_postgres_database(self.params)
            self._conn = _pg_connect(self.params)
            _pg_ensure_schema(self._conn, self.table, self.derivatives_table)
        return self._conn

    def close(self):
        """Close the connection; the next result opens a new one."""
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # pragma: no cover - already broken
                    pass

    def _store(self, input: Result):
        """Upsert the result and its derivatives; assumes the caller holds the lock."""
        conn = self._connection()
        cursor = conn.cursor()
        # The dedicated columns come out of the stored JSON, so a row cannot say one thing in
        # value_json and another in the column a query filters on
        js = self.build_json(input)
        provenance = input_provenance(
            input, self.settings.get("input_source"), self.settings.get("hash_input", True)
        )
        cursor.execute(
            f"""
            INSERT INTO {self.table}
            (id, processor_id, workflow_id, value_json, metadata_json, extra_json,
             input_uri, input_hash, input_locator, corrects_id)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, %s)
            ON CONFLICT (id) DO UPDATE SET
                processor_id = excluded.processor_id,
                workflow_id = excluded.workflow_id,
                value_json = excluded.value_json,
                metadata_json = excluded.metadata_json,
                extra_json = excluded.extra_json,
                input_uri = excluded.input_uri,
                input_hash = excluded.input_hash,
                input_locator = excluded.input_locator,
                corrects_id = excluded.corrects_id
            """,
            (
                input.id,
                js.get("processorId"),
                js.get("workflowId"),
                json.dumps(js),
                json.dumps(_json_safe(input.metadata)) if input.metadata else None,
                json.dumps(_json_safe(input.extra)) if input.extra else None,
                provenance["uri"],
                provenance["hash"],
                json.dumps(provenance["locator"]) if provenance["locator"] else None,
                correction_target(input),
            ),
        )
        for component, results in input.derivative_results.items():
            for result in results:
                cursor.execute(
                    f"""
                    INSERT INTO {self.derivatives_table} (id, source_id, component_id, result_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        source_id = excluded.source_id,
                        component_id = excluded.component_id,
                        result_json = excluded.result_json
                    """,
                    (result.id, input.id, component.id, json.dumps(result_to_json(result))),
                )
        conn.commit()

    def _process(self, input: Result) -> Result:
        """Store the result in PostgreSQL"""
        with self._lock:
            try:
                self._store(input)
            except Exception:
                # A connection that died between results (server restart, idle timeout) should cost
                # one result at most, not the rest of the run
                conn, self._conn = self._conn, None
                if conn is None:
                    raise
                try:
                    conn.close()
                except Exception:  # pragma: no cover - already broken
                    pass
                self._store(input)
        return input


def save_postgres_correction(
    result_id,
    corrected_value,
    agent=None,
    metadata=None,
    processor_id=None,
    params=None,
    table="results",
    **overrides,
):
    """Record a correction of *result_id* in PostgreSQL as a new row pointing back at it.

    The PostgreSQL counterpart of ``save_correction``: the original row is never touched, the
    correction is an entry of its own whose ``corrects_id`` is the corrected row and whose metadata
    says who made it -- *agent* (the person or model, recorded as ``metadata['agent']``) plus
    anything else in *metadata*. The correction inherits the original's ``input_uri``/``input_hash``.
    Returns the new row's id, or ``None`` if *result_id* is not in the table.
    """
    table = _pg_identifier(table)
    conn = _pg_connect(postgres_params(params, **overrides))
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT workflow_id, input_uri, input_hash, input_locator FROM {table} WHERE id = %s",
            (result_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        workflow_id, input_uri, input_hash, locator = row
        metadata = dict(metadata or {})
        if agent is not None:
            metadata.setdefault("agent", agent)
        correction_id, value_json = correction_json(
            result_id, corrected_value, metadata=metadata, processor_id=processor_id, workflow_id=workflow_id
        )
        cursor.execute(
            f"""
            INSERT INTO {table}
            (id, processor_id, workflow_id, value_json, metadata_json,
             input_uri, input_hash, input_locator, corrects_id)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, %s)
            """,
            (
                correction_id,
                processor_id,
                workflow_id,
                json.dumps(value_json),
                json.dumps(_json_safe(metadata)) if metadata else None,
                input_uri,
                input_hash,
                json.dumps(locator) if locator else None,
                result_id,
            ),
        )
        conn.commit()
        return correction_id
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


def _ensure_schema(conn):
    """Create the ``results``/``derivatives`` tables and their indexes if they are missing.

    The results table is the tabular shape both SQLite and PostgreSQL storage use: one row per
    result, ``input_uri``/``input_hash``/``input_locator`` saying what it was generated from and
    where in that input it is, and ``corrects_id`` pointing at the row this one corrects (see
    ``save_correction``). There is no migration from
    earlier layouts -- a database from before a schema change is deleted and rebuilt.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            processor_id TEXT,
            workflow_id TEXT,
            value_json TEXT,
            metadata_json TEXT,
            extra_json TEXT,
            input_uri TEXT,
            input_hash TEXT,
            input_locator TEXT,
            corrects_id TEXT,
            -- millisecond resolution, so results (and the corrections of one) sort in the order
            -- they were made rather than tying on the second
            created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_processor ON results(processor_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_workflow ON results(workflow_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_corrects ON results(corrects_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_input_hash ON results(input_hash)")
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
    keys = row.keys()
    js = {
        "id": row["id"],
        "processor_id": row["processor_id"],
        "workflow_id": row["workflow_id"],
        "value": _loads(row["value_json"]),
        "metadata": _loads(row["metadata_json"]),
        "extra": _loads(row["extra_json"]),
        "input_uri": row["input_uri"],
        "input_hash": row["input_hash"],
        "input_locator": _loads(row["input_locator"]),
        "corrects_id": row["corrects_id"],
        "created_at": row["created_at"],
    }
    if "correction_count" in keys:
        js["correction_count"] = row["correction_count"]
        js["corrected"] = row["correction_count"] > 0
    return js


def ensure_database(database):
    """Create *database* (and its parent directory) with the chai schema if missing; returns the path.

    Lets an app pre-build its results database so storage viewers work before the first run.
    """
    parent = os.path.dirname(database)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _connect(database).close()
    return database


def store_json_result(
    database,
    result_id,
    value,
    processor_id=None,
    workflow_id=None,
    metadata=None,
    input_uri=None,
    input_hash=None,
    input_locator=None,
    corrects_id=None,
):
    """Insert/refresh one result row from already-serialized JSON values.

    The viewer-side counterpart of ``SqliteStorage._process`` for callers that hold a run's
    serialized output (dicts) rather than live ``Result`` objects -- e.g. a front-end persisting
    run results into its app-local database. Re-storing a result keeps its original ``created_at``
    and leaves any correction rows pointing at it untouched.
    """
    conn = _connect(database)
    try:
        conn.execute(
            """
            INSERT INTO results
            (id, processor_id, workflow_id, value_json, metadata_json,
             input_uri, input_hash, input_locator, corrects_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                processor_id = excluded.processor_id,
                workflow_id = excluded.workflow_id,
                value_json = excluded.value_json,
                metadata_json = excluded.metadata_json,
                input_uri = excluded.input_uri,
                input_hash = excluded.input_hash,
                input_locator = excluded.input_locator,
                corrects_id = excluded.corrects_id
            """,
            (
                result_id,
                processor_id,
                workflow_id,
                json.dumps(_json_safe(value)),
                json.dumps(_json_safe(metadata)) if metadata else None,
                input_uri,
                input_hash,
                json.dumps(_json_safe(input_locator)) if input_locator else None,
                corrects_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return True


#: Every results column plus the number of rows correcting each one, for the viewer helpers.
_RESULTS_SELECT = (
    "SELECT r.*, (SELECT COUNT(*) FROM results c WHERE c.corrects_id = r.id) AS correction_count FROM results r"
)


def list_results(
    database, processor_id=None, workflow_id=None, input_hash=None, corrects_id=None, limit=100, offset=0
):
    """Return stored results as a list of dicts, newest first.

    Each dict has ``id``, ``processor_id``, ``workflow_id``, ``value`` (parsed JSON), ``metadata``,
    ``extra``, ``input_uri``, ``input_hash``, ``corrects_id``, ``created_at``, and the
    ``correction_count``/``corrected`` pair saying how many rows correct this one. Corrections are
    rows like any other, so they are listed too; pass *corrects_id* to list only the corrections of
    one result, or *input_hash* for every entry generated from the same input content. Optionally
    filter by *processor_id* and/or *workflow_id*; page with *limit*/*offset*.
    """
    sql = _RESULTS_SELECT
    clauses, params = [], []
    for column, value in (
        ("processor_id", processor_id),
        ("workflow_id", workflow_id),
        ("input_hash", input_hash),
        ("corrects_id", corrects_id),
    ):
        if value is not None:
            clauses.append(f"r.{column} = ?")
            params.append(value)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY r.created_at DESC, r.id LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    conn = _connect(database)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_result(database, result_id, with_corrections=True):
    """Return a single stored result as a dict (see ``list_results``), or ``None`` if not found.

    Unless *with_corrections* is false the dict also has ``corrections``: the rows that correct this
    one, oldest first, each carrying the metadata of whoever made it.
    """
    conn = _connect(database)
    try:
        row = conn.execute(f"{_RESULTS_SELECT} WHERE r.id = ?", (result_id,)).fetchone()
        if row is None:
            return None
        js = _row_to_dict(row)
        if with_corrections:
            corrections = conn.execute(
                f"{_RESULTS_SELECT} WHERE r.corrects_id = ? ORDER BY r.created_at, r.id", (result_id,)
            ).fetchall()
            js["corrections"] = [_row_to_dict(correction) for correction in corrections]
        return js
    finally:
        conn.close()


def correction_json(result_id, corrected_value, metadata=None, processor_id=None, workflow_id=None):
    """The ``(id, value_json)`` pair for a correction of *result_id*, shaped like a stored result.

    A correction is an entry in its own right: its ``input`` is the result it corrects, its ``value``
    is what that result should have said, and its ``metadata`` says who made it -- so the shape a
    viewer reads for a result works unchanged for a correction of one.
    """
    correction_id = str(uuid.uuid4())
    return correction_id, {
        "id": correction_id,
        "type": "Correction",
        "workflowId": workflow_id,
        "processorId": processor_id,
        "metadata": _json_safe(metadata or {}),
        "extraInfo": {"corrects": result_id},
        "input": result_id,
        "value": _json_safe(corrected_value),
    }


def save_correction(database, result_id, corrected_value, agent=None, metadata=None, processor_id=None):
    """Record a correction of *result_id* as a new row pointing back at it; returns the new row's id.

    The original is never touched: the correction is stored as its own entry whose ``corrects_id`` is
    the corrected row, so a result can be corrected more than once and each correction keeps its own
    metadata -- *agent* (the person or model that made it, recorded as ``metadata['agent']``) and
    anything else in *metadata*, e.g. who reviewed it, why, or which model and prompt produced it.
    The correction inherits the original's ``input_uri``/``input_hash``, since it is a correction of
    what was made from that same input. Returns ``None`` if *result_id* is not in the database.
    """
    conn = _connect(database)
    try:
        row = conn.execute(
            "SELECT workflow_id, input_uri, input_hash, input_locator FROM results WHERE id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        metadata = dict(metadata or {})
        if agent is not None:
            metadata.setdefault("agent", agent)
        correction_id, value_json = correction_json(
            result_id, corrected_value, metadata=metadata, processor_id=processor_id, workflow_id=row["workflow_id"]
        )
        conn.execute(
            """
            INSERT INTO results
            (id, processor_id, workflow_id, value_json, metadata_json,
             input_uri, input_hash, input_locator, corrects_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                correction_id,
                processor_id,
                row["workflow_id"],
                json.dumps(value_json),
                json.dumps(_json_safe(metadata)) if metadata else None,
                row["input_uri"],
                row["input_hash"],
                row["input_locator"],
                result_id,
            ),
        )
        conn.commit()
        return correction_id
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
    dedicated columns for querying. ``input_uri`` and ``input_hash`` record the file or URI the
    result was generated from and the md5 of that input's content, and ``corrects_id`` points at the
    row this result corrects, if it is a correction of one (see ``correction_target`` and
    ``save_correction``) -- corrections are entries in the table like any other, so the agent that
    made one, human or model, is described by that entry's own metadata. The schema is created lazily
    on first use, a fresh connection is opened per operation (thread-safe for use from e.g. Flask),
    and the input is returned unchanged.

    Settings:
        - database: path of the SQLite database file (default 'results.db')
        - input_source: component id whose result counts as the input the row was generated from,
          overriding the component marked ``source: true`` (default: the marked component's result,
          or the result's immediate input when nothing is marked)
        - hash_input: fill in ``input_hash`` (default true); false skips the digest, and any file
          read it would need
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
            js = self.build_json(input)
            metadata_json = json.dumps(_json_safe(input.metadata)) if input.metadata else None
            extra_json = json.dumps(_json_safe(input.extra)) if input.extra else None
            provenance = input_provenance(
                input, self.settings.get("input_source"), self.settings.get("hash_input", True)
            )

            # Insert result -- an upsert (not OR REPLACE) so re-storing a result keeps the original
            # created_at, and the rows correcting it go on pointing at it
            cursor.execute(
                """
                INSERT INTO results
                (id, processor_id, workflow_id, value_json, metadata_json, extra_json,
                 input_uri, input_hash, input_locator, corrects_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    processor_id = excluded.processor_id,
                    workflow_id = excluded.workflow_id,
                    value_json = excluded.value_json,
                    metadata_json = excluded.metadata_json,
                    extra_json = excluded.extra_json,
                    input_uri = excluded.input_uri,
                    input_hash = excluded.input_hash,
                    input_locator = excluded.input_locator,
                    corrects_id = excluded.corrects_id
            """,
                (
                    input.id,
                    js.get("processorId"),
                    js.get("workflowId"),
                    json.dumps(js),
                    metadata_json,
                    extra_json,
                    provenance["uri"],
                    provenance["hash"],
                    json.dumps(provenance["locator"]) if provenance["locator"] else None,
                    correction_target(input),
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
