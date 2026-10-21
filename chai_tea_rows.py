"""Reshape what a chai run stored into rows with exactly the columns of the CHAI-TEA RESULT table.

``chai_tea_import.py`` turns a run into RESULT rows by inserting them into a database it can reach.
This produces the same rows as a file instead, for a central database that ingests files rather than
being written to: one JSON-Lines record per stored result whose keys are exactly the RESULT columns
of ``chai_tea_schema.py`` -- in the same order, no more and no fewer -- then a Parquet file typed to
match, ready to upload (see ``chai.uploader.S3Uploader``) and load with ``parquet_to_postgres``:

    python chai_tea_rows.py results/m804/run.jsonl --field text=value --field page=input_uri
    python chai_tea_rows.py results/m804/run.jsonl --field text=value --field page=input_uri \\
        --profile chai-tea-pilot --bucket yale-chai-tea-pilot-tst --prefix m804/

Each record goes through ``chai_tea_import.normalize`` and ``cost_of``, so the rows are the ones an
import would have inserted. What the import leaves to the database is filled in here, because a
file loaded into a NOT NULL column has to carry a value for it:

* ``workflow_run_id`` -- given with ``--workflow-run-id``; else the one the record already carries
  (the storages write RESULT rows by default, with the workflow's run id); else derived from the md5
  of the source file, so transforming the same file twice gives the same id. The loader must create
  the WORKFLOW_RUN row with that id (and the WORKFLOW and PROJECT above it) before loading the rows;
  the Parquet file's metadata records the id.
* ``md_timestamp`` -- the storage's ``created_at``, else the ``timestamp`` the result's metadata
  recorded when it was made, else the time of the transform
* ``created``/``modified`` -- the time of the transform
* ``editor_user_id`` -- as the record has it; NULL for a run's own results, made by a component

Workflows that name their stored fields differently (the m804 workflow writes ``value`` as ``text``
and ``@input_uri`` as ``page``) are mapped back with ``--field NAME=STANDARD``, where STANDARD is a
key ``normalize`` reads (``value``, ``input_uri``, ``processorId``, ``metadata``, ``extraInfo``,
``input_hash``, ``input_locator``, ``corrects_id``, ...).

``previous_id`` is kept as stored. A correction that points at a result the central database does
not hold will fail its foreign key there, so load the corrected run first.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import uuid

import chai_tea_import
from chai.schema import column_kinds
from chai.storage import jsonl_to_parquet, read_run_records, run_records_file

logger = logging.getLogger(__name__)

#: ``{column: Parquet column kind}`` for the RESULT table, in the table's column order.
RESULT_KINDS = column_kinds("RESULT")


def run_id_for(path):
    """A workflow_run_id for the run stored in *path*, stable for as long as its content is."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return str(uuid.uuid5(chai_tea_import.ID_NAMESPACE, f"workflow_run:{digest.hexdigest()}"))


def parse_fields(specs):
    """``["text=value", ...]`` -> ``{"text": "value"}``: stored field names to the standard ones."""
    fields = {}
    for spec in specs or []:
        name, _, standard = str(spec).partition("=")
        if not name or not standard:
            raise ValueError(f"Field mapping {spec!r} is not NAME=STANDARD")
        fields[name.strip()] = standard.strip()
    return fields


def result_row(record, workflow_run_id, prices=None, fields=None, now=None):
    """One stored *record* as a RESULT row: exactly the table's columns, in its order.

    Returns ``None`` for a record with no id, which cannot be a row. *fields* renames stored keys to
    the ones ``normalize`` reads before anything else happens; *now* is the transform time used for
    ``created``/``modified`` and as the last resort for ``md_timestamp``.
    """
    if fields:
        record = {fields.get(key, key): value for key, value in record.items()}
    row = chai_tea_import.normalize(record)
    if row["id"] is None:
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    metadata = row["metadata"]
    stamp = row["md_timestamp"] or chai_tea_import._timestamp_of(metadata.get("timestamp")) or now
    created = chai_tea_import._timestamp_of(record.get("created")) or now
    modified = chai_tea_import._timestamp_of(record.get("modified")) or now
    values = {
        "id": row["id"],
        "workflow_run_id": str(workflow_run_id),
        "process_id": row["process_id"],
        "input": row["input"],
        "input_hash": row["input_hash"],
        "input_segment": row["input_segment"],
        "input_sequence": row["input_sequence"],
        "was_successful": row["was_successful"],
        "value": row["value"],
        "metadata": metadata,
        "extra_data": row["extra_data"],
        "previous_id": row["previous_id"],
        "editor_user_id": chai_tea_import.as_uuid(record.get("editor_user_id")),
        "md_cost": _first_not_none(chai_tea_import.cost_of(metadata, prices or {}), record.get("md_cost")),
        "md_duration": row["md_duration"],
        "md_timestamp": stamp,
        "created": created,
        "modified": modified,
    }
    # The schema is the contract: a column added to RESULT must be filled in here, not left out
    missing = [column for column in RESULT_KINDS if column not in values]
    if missing:
        raise ValueError(f"RESULT has column(s) {', '.join(missing)} that result_row does not fill in")
    return {column: values[column] for column in RESULT_KINDS}


def _first_not_none(*values):
    return next((value for value in values if value is not None), None)


def _jsonable(value):
    """Timestamps as ISO-8601 text, so a row survives the trip through JSON-Lines."""
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def input_positions(source, fields=None):
    """``{input: position}`` for the inputs of the records in *source* that carry no order of their own.

    The inputs are numbered in sorted order, which for a directory of page images is the order
    ``DirFileProvider`` reads them in.
    """
    inputs = set()
    for record in chai_tea_import.read_jsonl(source):
        if fields:
            record = {fields.get(key, key): value for key, value in record.items()}
        row = chai_tea_import.normalize(record)
        if row["input_sequence"] is None and row["input"] is not None:
            inputs.add(row["input"])
    return {value: position for position, value in enumerate(sorted(inputs))}


def transform_jsonl(source, target, workflow_run_id=None, prices=None, fields=None):
    """Write the RESULT rows for every record of the JSON-Lines file *source* into *target*.

    *workflow_run_id* overrides the run each record names; a record that names none gets one derived
    from the file. Returns ``{"path", "rows", "skipped", "workflow_run_id"}``, the last being the run
    of the first row. Records with no id, and lines that are not JSON (a run killed mid-write), are
    skipped and counted.

    A row whose record says nothing of its order gets ``input_sequence`` from its ``input``: the
    position of that input among the file's inputs, sorted -- the order ``DirFileProvider`` reads a
    directory of pages in -- so a page's transcription and translation share the page's number.
    Files written since the storages record ``input_sequence`` themselves keep theirs.
    """
    positions = input_positions(source, fields)
    derived = None
    first_run = workflow_run_id
    now = datetime.datetime.now(datetime.timezone.utc)
    rows = skipped = 0
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as out:
        for record in chai_tea_import.read_jsonl(source):
            run = workflow_run_id or record.get("workflow_run_id")
            if not run:
                derived = derived or run_id_for(source)
                run = derived
            row = result_row(record, run, prices, fields, now)
            if row is None:
                logger.warning("Record with no id skipped: %r", record)
                skipped += 1
                continue
            if row["input_sequence"] is None and row["input"] is not None:
                row["input_sequence"] = positions.get(row["input"])
            out.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")
            rows += 1
            first_run = first_run or row["workflow_run_id"]
    first_run = first_run or derived or run_id_for(source)
    return {"path": target, "rows": rows, "skipped": skipped, "workflow_run_id": str(first_run)}


def rows_to_parquet(rows_jsonl, parquet_file, workflow_run_id, compression="snappy", records=None):
    """Pack a file of RESULT rows into Parquet, typed by the schema rather than inferred from the data.

    Declaring every column means the file has the RESULT columns and types even where the data is
    all null, and an empty run still produces a file the loader can read. *records* are the run rows
    to carry in its ``chai_tea_records`` metadata (see ``jsonl_to_parquet``) -- for a file
    ``JsonLinesStorage`` wrote, those it recorded beside the source (``read_run_records``).
    """
    return jsonl_to_parquet(
        rows_jsonl,
        parquet_file,
        schema=RESULT_KINDS,
        compression=compression,
        metadata={"chai_tea_entity": "RESULT", "chai_tea_workflow_run_id": str(workflow_run_id)},
        records=records or {},
    )


def upload(parquet_file, bucket, prefix="", profile=None, region=None, overwrite=False):
    """Upload *parquet_file* with an ``S3Uploader``; returns its ``{uri, key, size, md5, source}``."""
    from chai.result import FileItemResult
    from chai.workflow import Workflow

    settings = {"bucket": bucket, "prefix": prefix, "overwrite": overwrite}
    if profile:
        settings["profile"] = profile
    if region:
        settings["region"] = region
    wf = Workflow({"id": "chai_tea_rows", "type": "workflow.Workflow"})
    uploader = wf._make_step({"id": "to_s3", "type": "uploader.S3Uploader", "settings": settings}, wf)
    return uploader.process(FileItemResult(parquet_file)).value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the .jsonl file a JsonLinesStorage wrote")
    parser.add_argument("--rows", help="where to write the RESULT rows (default: <source>.result.jsonl)")
    parser.add_argument("--parquet", help="where to write the Parquet file (default: <source>.result.parquet)")
    parser.add_argument("--workflow-run-id", help="the WORKFLOW_RUN the rows belong to (default: from the file's md5)")
    parser.add_argument(
        "--field",
        action="append",
        metavar="NAME=STANDARD",
        help="a stored field to read as a standard one, e.g. text=value; repeatable",
    )
    parser.add_argument(
        "--price",
        action="append",
        metavar="MODEL=IN/OUT",
        help="dollars per million tokens for a model, as for chai_tea_import.py; repeatable",
    )
    parser.add_argument("--bucket", help="upload the Parquet file to this S3 bucket")
    parser.add_argument("--prefix", default="", help="key prefix for the upload, e.g. m804/")
    parser.add_argument("--profile", help="AWS profile for the upload")
    parser.add_argument("--region", help="AWS region for the upload")
    parser.add_argument("--overwrite", action="store_true", help="replace an object already at the key")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    try:
        fields = parse_fields(args.field)
        prices = chai_tea_import.parse_prices(args.price)
    except ValueError as e:
        parser.error(str(e))

    stem = os.path.splitext(args.source)[0]
    info = transform_jsonl(
        args.source, args.rows or f"{stem}.result.jsonl", args.workflow_run_id, prices, fields
    )
    print(f"{info['rows']} RESULT rows ({info['skipped']} skipped) -> {info['path']}")
    print(f"workflow_run_id {info['workflow_run_id']}")
    packed = rows_to_parquet(
        info["path"],
        args.parquet or f"{stem}.result.parquet",
        info["workflow_run_id"],
        # an overriding run id leaves the recorded runs naming rows the file no longer has
        records=None if args.workflow_run_id else read_run_records(run_records_file(args.source)),
    )
    print(f"{packed['rows']} rows -> {packed['path']}")
    if args.bucket:
        sent = upload(packed["path"], args.bucket, args.prefix, args.profile, args.region, args.overwrite)
        print(f"uploaded to {sent['uri']} ({sent['size']} bytes, md5 {sent['md5']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
