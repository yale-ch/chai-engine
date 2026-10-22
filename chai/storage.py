"""Storage components: persist Results to the filesystem or a database.

Storage steps are pass-through: they write the input Result somewhere and return it unchanged, so
they can be inserted anywhere in a pipeline (typically as ``next_steps``) without affecting the data
flow.

Everything a storage records is in the CHAI-TEA schema (``chai.persistence.schema``):
``SqliteStorage`` and ``PostgresStorage`` write its tables -- each result a RESULT row, under the
run's PROJECT, WORKFLOW and WORKFLOW_RUN rows -- and ``JsonLinesStorage``/``ParquetStorage`` write
the same RESULT rows to files. The components only decide what to write and when; the writing
itself -- rows, files, databases, and moving them between one another -- is ``chai.persistence``.
"""

import atexit
import logging
import os
import threading
import uuid
import weakref
from datetime import datetime, timezone

import ujson as json

from .core import Component, Result
from .persistence.jsonl import _line_lock, add_run_records, run_records_file
from .persistence.parquet import ParquetRecordWriter, _value_kind
from .persistence.postgres import (
    _pg_connect,
    _pg_ensure_chai_tea,
    ensure_postgres_database,
    postgres_params,
)
from .persistence.provenance import _source_result, _uri_for
from .persistence.rows import RESULT_COLUMNS, build_record, result_row, result_to_json
from .persistence.schema import table_names
from .persistence.sql import write_rows
from .persistence.sqlite import _connect
from .persistence.values import _isoformat_rows, _json_safe
from .result import FileItemResult

logger = logging.getLogger("chai")

#: path -> the run that last emptied it, so 'truncate' empties a file once per run, not once per step
_truncated_runs = {}
#: path -> the run whose PROJECT/WORKFLOW/WORKFLOW_RUN rows were last recorded beside it
_recorded_runs = {}


def _run_id_of(component):
    """The run a storage's rows belong to: its ``run_id`` setting, else its workflow's current run."""
    return component.settings.get("run_id") or getattr(
        component.workflow, "run_id", None
    )


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
                logger.warning(
                    f"{self} found no result from '{self.name_from}' to name the file after"
                )
        name = ""
        if source is not None:
            name = (
                source.file_name
                if isinstance(source, FileItemResult)
                else str(source.value)
            )
        else:
            name = _uri_for(input) or ""
        if not name:
            # Nothing in the chain says where this came from; the result's own id at least is unique
            return input.id
        return os.path.splitext(os.path.basename(name.rstrip("/")))[0]

    def _process(self, input: Result) -> Result:
        if isinstance(input, FileItemResult):
            # Writing a file result's text would mean reading the file back off disk to copy it
            logger.warning(
                f"{self} was given a file result and wrote nothing: {input!r}"
            )
            return input
        fn = os.path.join(
            self.directory, f"{self.result_name(input)}{self.suffix}{self.extension}"
        )
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

    With RESULT rows the rows they hang under -- the run's PROJECT, WORKFLOW and WORKFLOW_RUN -- are
    kept beside the file, in ``run.records.json`` for ``run.jsonl`` (``run_records_file``): written
    before the run's first line, so every line's run is on disk before the line is, and again with
    the run's outcome when it ends. A file appended to by several runs records all of them;
    ``jsonl_to_parquet`` carries them into the Parquet file's ``chai_tea_records`` metadata, as
    ``ParquetStorage`` does, so a loader can create them before the rows that reference them. With
    a ``run_id`` setting there are none to record: that run is the caller's to create.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "file" not in self.settings:
            self.settings["file"] = "results.jsonl"
        self.file_name = os.path.abspath(self.settings["file"])
        self.records_file = run_records_file(self.file_name)
        parent_dir = os.path.dirname(self.file_name)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        # Created (never emptied) when the workflow is built, so a run that produces nothing still
        # leaves a file; 'truncate' empties it when a run starts, not before
        self.truncate = str(self.settings.get("mode", "append")).lower() == "truncate"
        # RESULT rows reference the workflow's run -- unless fields make the lines something else, or
        # a configured run_id names a run the workflow knows nothing of
        self.result_rows = (
            not self.settings.get("fields")
            and not self.settings.get("run_id")
            and hasattr(self.workflow, "records")
        )
        with _line_lock(self.file_name):
            with open(self.file_name, "a"):
                pass
        if (self.truncate or self.result_rows) and hasattr(
            self.workflow, "add_listener"
        ):
            self.workflow.add_listener(self._on_workflow_event)

    def _on_workflow_event(self, payload):
        """Empty the file when the workflow starts a run; record the run's outcome when it ends."""
        if payload.get("component_id") != self.workflow.id:
            return
        event = payload.get("event")
        if event == "component_start" and self.truncate:
            self._truncate_for_run()
        elif event in ("component_end", "component_error") and self.result_rows:
            with _line_lock(self.file_name):
                if _recorded_runs.get(self.file_name) == self.workflow.run_id:
                    add_run_records(self.records_file, self.workflow.records())

    def _truncate_for_run(self):
        """Empty the file, once per run: whichever step writing to it gets there first does it."""
        run_id = getattr(self.workflow, "run_id", None) or "standalone"
        with _line_lock(self.file_name):
            if _truncated_runs.get(self.file_name) != run_id:
                with open(self.file_name, "w"):
                    pass
                if os.path.exists(
                    self.records_file
                ):  # the runs it recorded went with the lines
                    os.remove(self.records_file)
                _recorded_runs.pop(self.file_name, None)
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
        record = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in self.build_json(input).items()
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        if self.truncate:
            self._truncate_for_run()  # a no-op once this run has emptied the file
        with _line_lock(self.file_name):
            if (
                self.result_rows
                and _recorded_runs.get(self.file_name) != self.workflow.run_id
            ):
                add_run_records(self.records_file, self.workflow.records())
                _recorded_runs[self.file_name] = self.workflow.run_id
            with open(self.file_name, "a", encoding="utf-8") as fh:
                fh.write(line)
        return input


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
        self.run_columns = (
            self.settings.get("run_columns", True) and not self.result_rows
        )
        self._lock = threading.Lock()
        self._writer = None
        self.run_id = None
        self.file_name = None
        self._run_started = False
        self._start_run()
        if hasattr(
            self.workflow, "add_listener"
        ):  # a Workflow; the file is written when its run ends
            self.workflow.add_listener(self._on_workflow_event)
        if self.settings.get("flush_on_exit", True):
            # A weak reference, so a workflow that is built and dropped is not kept alive until exit
            atexit.register(_flush_parquet_at_exit, weakref.ref(self))

    def _start_run(self):
        """Pick the run id and resolve the file name for the run that is about to write."""
        self.run_id = str(_run_id_of(self) or uuid.uuid4())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.file_name = self.file_template.replace("{run_id}", self.run_id).replace(
            "{timestamp}", stamp
        )
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
            input,
            self.settings.get("fields"),
            self.settings.get("sources"),
            self.settings.get("input_source"),
        )
        if self.settings.get("null_if_empty"):
            js = {
                column: (None if value in ({}, [], "") else value)
                for column, value in js.items()
            }
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
                writer.metadata["chai_tea_records"] = json.dumps(
                    _json_safe(_isoformat_rows(self.workflow.records()))
                )
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
        if hasattr(
            self.workflow, "add_listener"
        ):  # a Workflow; record the run's outcome when it ends
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
            raise ValueError(
                f"{self} must be part of a Workflow: every RESULT row belongs to a WORKFLOW_RUN"
            )
        with_hash = self.settings.get("hash_input", True)
        rows = [
            result_row(
                input,
                run_id,
                input_source=self.settings.get("input_source"),
                with_hash=with_hash,
            )
        ]
        for results in input.derivative_results.values():
            rows.extend(
                result_row(result, run_id, with_hash=with_hash) for result in results
            )
        records = {"RESULT": rows}
        if self._run_written != run_id:
            records.update(self.workflow.records())
        return records

    def _write(self, records):
        raise NotImplementedError

    def _end_run(self):
        """Write the finished WORKFLOW_RUN (and WORKFLOW) rows for a run this storage wrote into."""
        with self._lock:
            if (
                self._run_written is not None
                and self._run_written == self.workflow.run_id
            ):
                records = self.workflow.records()
                self._write(
                    {
                        "WORKFLOW": records["WORKFLOW"],
                        "WORKFLOW_RUN": records["WORKFLOW_RUN"],
                    }
                )

    def _process(self, input: Result) -> Result:
        with self._lock:
            records = self.records_for(input)
            self._write(records)
            self._run_written = self.workflow.run_id
        return input


class PostgresStorage(_ChaiTeaStorage):
    """Store results in PostgreSQL, in the CHAI-TEA schema (``chai.persistence.schema``).

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
        - tables: ``{entity: table name}`` overriding the CHAI-TEA table names (``chai.persistence.schema.TABLES``)
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


class SqliteStorage(_ChaiTeaStorage):
    """Store results in a SQLite database, in the CHAI-TEA schema (``chai.persistence.schema``).

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
