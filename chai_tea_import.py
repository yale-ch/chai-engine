"""Load the results a chai run persisted into the CHAI-TEA schema ``chai_tea_schema.py`` builds.

The missing half of the CHAI-TEA story: ``chai_tea_parquet.py`` moves CHAI-TEA rows between
PostgreSQL instances, but nothing turned a *run* into CHAI-TEA rows in the first place. This reads
what any of the storage components wrote and inserts one RESULT row per stored result, under a
WORKFLOW_RUN row standing for the run as a whole.

Every storage layer is a source, because they all persist the same ``result_to_json`` shape:

    python chai_tea_import.py run.jsonl          --project M804 --database chai_tea
    python chai_tea_import.py run.parquet        --project M804 --database chai_tea
    python chai_tea_import.py results.db         --project M804 --database chai_tea   # SqliteStorage
    python chai_tea_import.py postgres://.../chai --project M804 --database chai_tea  # PostgresStorage

The type is taken from the extension (``.jsonl``/``.json``, ``.parquet``, ``.db``/``.sqlite``, a
``postgres://`` URL), or named outright with ``--source``.

AI metadata
-----------
The whole metadata dict an AI component recorded goes into RESULT.metadata verbatim -- for the
Gemini/Ollama/LM Studio/OpenAI/MLX/Transformers components that is ``token_usage`` (``total``,
``prompt``, ``images``, ``thinking``, ``result``), ``duration``, ``type``, ``engine`` and ``model``.
Nothing is dropped or flattened, so a column added to a provider's usage metadata later needs no
change here. Two values are *also* promoted to columns of their own so they can be queried and
indexed without reaching into the JSON:

* ``md_duration`` -- ``metadata.duration``, the wall-clock seconds the call took
* ``md_timestamp`` -- when the storage recorded the row

``md_cost`` is filled in only when prices are supplied, since nothing in a provider's response says
what a call cost. Give them per million tokens as ``--price MODEL=INPUT/OUTPUT``, taken from the
provider's own pricing page:

    python chai_tea_import.py run.jsonl --database chai_tea --project M804 \\
        --price gemini-3.1-flash-lite-preview=0.10/0.40 --price 'qwen/*=0/0'

The model is read from each result's own ``metadata.model``, so a run that used several models costs
each row at its own rate; ``--price '*=IN/OUT'`` sets a fallback for models not named. Thinking
tokens are billed as output, matching ``storage.token_usage_summary``. A row whose model has no
price leaves ``md_cost`` NULL rather than guessing at zero.

One thing to know before writing rollup SQL over the stored counts: a provider that did not break a
modality down records ``-1`` for it, not 0 (Ollama and LM Studio do this for ``images`` and
``thinking``), and those sentinels are kept as the provider wrote them. So a plain
``sum((metadata->'token_usage'->>'thinking')::int)`` returns a negative number across such rows.
Filter them out with ``GREATEST(..., 0)``, as the costing here does.

Mapping
-------
A stored result becomes a RESULT row like this, accepting both the ``result_to_json`` key spelling
(``processorId``, ``extraInfo``) that ``JsonLinesStorage``/``ParquetStorage`` write and the column
spelling (``processor_id``, ``extra_json``) the SQLite and PostgreSQL tables use:

    id              <- the result's own uuid (a non-uuid id is hashed into one, deterministically)
    process_id      <- processorId / processor_id
    input           <- input_uri if the storage recorded one, else the input result's id
    input_hash      <- input_hash
    input_segment   <- input_locator, as JSON text
    input_sequence  <- the last integer frame of the locator, if it has one
    was_successful  <- false when the metadata says the step errored, true otherwise
    value/metadata/extra_data  <- value, metadata, extraInfo/extra_json, unchanged
    previous_id     <- corrects_id: in CHAI-TEA the corrected row is the one this supersedes
    md_timestamp    <- the storage's created_at, or the import time when it recorded none
                       (never NULL: chai_tea_parquet.py's --since export filters on it)

Rows are inserted in one transaction with ``ON CONFLICT (id) DO NOTHING``, so re-importing a run
that was already loaded adds only what is new -- which is what makes it safe to point this at a
JSON-Lines file a long run is still appending to.
"""

import argparse
import datetime
import fnmatch
import json
import logging
import os
import sys
import uuid

import chai_tea_schema
from chai.storage import _pg_connect, _pg_identifier, postgres_params

logger = logging.getLogger(__name__)

# Namespace for turning a non-uuid result id into a uuid, stable across imports so a re-import of
# the same run collides on the primary key (and is skipped) instead of duplicating the row.
ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Metadata "type" values a component uses to mark a result it failed to produce.
ERROR_TYPES = {"ERROR"}


# --------------------------------------------------------------------------------------------------
# Reading what the storage layers wrote
# --------------------------------------------------------------------------------------------------


def detect_source(path):
    """Which storage layer *path* came from, by extension or URL scheme."""
    if str(path).startswith(("postgres://", "postgresql://")):
        return "postgres"
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".jsonl", ".ndjson", ".json"):
        return "jsonl"
    if ext == ".parquet":
        return "parquet"
    if ext in (".db", ".sqlite", ".sqlite3"):
        return "sqlite"
    raise ValueError(
        f"Cannot tell what kind of storage {path!r} is; name it with "
        f"--source jsonl|parquet|sqlite|postgres"
    )


def read_jsonl(path):
    """Yield each record of a ``JsonLinesStorage`` file, skipping blank and unreadable lines."""
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as e:
                # A run killed mid-write can leave a partial last line; the rest of the file is good
                logger.warning("%s line %d is not valid JSON, skipped (%s)", path, number, e)


def read_parquet(path):
    """Yield each row of a ``ParquetStorage`` file as a dict."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("Reading Parquet requires pyarrow: pip install pyarrow") from e
    table = pq.read_table(path)
    for batch in table.to_batches():
        for row in batch.to_pylist():
            yield row


def read_sqlite(path):
    """Yield each row of a ``SqliteStorage`` results table as a dict."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute("SELECT * FROM results"):
            yield dict(row)
    finally:
        conn.close()


def read_postgres(dsn, table="results"):
    """Yield each row of a ``PostgresStorage`` results table as a dict."""
    conn = _pg_connect({"conninfo": dsn})
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {_pg_identifier(table)}")
        columns = [c[0] for c in cursor.description]
        for row in cursor:
            yield dict(zip(columns, row))
    finally:
        conn.close()


def read_records(path, source=None, table="results"):
    """Yield the stored records of *path*, whichever storage layer wrote them."""
    source = source or detect_source(path)
    if source == "jsonl":
        return read_jsonl(path)
    if source == "parquet":
        return read_parquet(path)
    if source == "sqlite":
        return read_sqlite(path)
    if source == "postgres":
        return read_postgres(path, table)
    raise ValueError(f"Unknown source {source!r}")


# --------------------------------------------------------------------------------------------------
# Normalizing a record
# --------------------------------------------------------------------------------------------------


def _maybe_json(value):
    """Parse *value* if it is JSON text, otherwise return it as it is.

    The tabular storages keep JSON in text columns (SQLite) or jsonb (PostgreSQL, already parsed by
    the driver), and a Parquet file keeps whatever kind the writer inferred -- so a field can arrive
    either way and both have to work.
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


def as_uuid(value):
    """*value* as a uuid: used directly if it is one, else hashed into one deterministically."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(ID_NAMESPACE, str(value)))


def _sequence_of(locator):
    """The last integer frame of an input locator, if it has one.

    A locator is a list of frames saying where a result sits inside its source (see
    ``storage.input_locator``); an entry carved out by index records that index, which is what
    ``input_sequence`` is for. Anything else (a bbox, a character range) has no sequence.
    """
    if not isinstance(locator, list):
        return None
    for frame in reversed(locator):
        if isinstance(frame, int):
            return frame
        if isinstance(frame, dict):
            for key in ("sequence", "index", "page", "n"):
                if isinstance(frame.get(key), int):
                    return frame[key]
    return None


def _timestamp_of(value):
    """*value* as a datetime, accepting what each storage layer records."""
    if value is None or isinstance(value, datetime.datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Unparseable timestamp %r, left NULL", value)
        return None


def normalize(record):
    """One stored record as the fields a CHAI-TEA RESULT row needs.

    Accepts both spellings a storage layer might have used -- the ``result_to_json`` keys
    (``processorId``, ``extraInfo``, ``workflowId``) and the tabular columns (``processor_id``,
    ``extra_json``, ``workflow_id``) -- so every storage component is a source without the caller
    having to say which shape it wrote.
    """
    metadata = _maybe_json(_first(record, "metadata", "metadata_json")) or {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}
    extra = _maybe_json(_first(record, "extraInfo", "extra", "extra_json")) or None
    locator = _maybe_json(record.get("input_locator"))

    # input_uri is what the storage recorded the row as coming from; without one, the plain "input"
    # field is the id of the result this was made from, which is still the best answer available.
    source_input = _first(record, "input_uri", "@input_uri", "input")

    error = metadata.get("type") in ERROR_TYPES or "error" in metadata
    return {
        "id": as_uuid(_first(record, "id", "@id")),
        "process_id": _first(record, "processorId", "processor_id", "step"),
        "workflow_id": _first(record, "workflowId", "workflow_id"),
        "input": None if source_input is None else str(source_input),
        "input_hash": _first(record, "input_hash", "@input_hash"),
        "input_segment": json.dumps(locator) if locator else None,
        "input_sequence": _sequence_of(locator),
        "was_successful": not error,
        "value": _maybe_json(_first(record, "value", "value_json", "@record")),
        "metadata": metadata,
        "extra_data": extra,
        "previous_id": as_uuid(_first(record, "corrects_id", "@corrects")),
        "md_duration": metadata.get("duration"),
        "md_timestamp": _timestamp_of(_first(record, "created_at", "md_timestamp")),
    }


# --------------------------------------------------------------------------------------------------
# Costing
# --------------------------------------------------------------------------------------------------


def parse_prices(specs):
    """``["model=IN/OUT", ...]`` -> ``{model glob: (input, output)}``, in dollars per million tokens."""
    prices = {}
    for spec in specs or []:
        model, _, rates = str(spec).partition("=")
        if not model or not rates:
            raise ValueError(f"Price {spec!r} is not MODEL=INPUT/OUTPUT")
        inp, _, out = rates.partition("/")
        try:
            prices[model.strip()] = (float(inp), float(out if out else inp))
        except ValueError as e:
            raise ValueError(f"Price {spec!r} has non-numeric rates") from e
    return prices


def price_for(model, prices):
    """The (input, output) rates for *model*: an exact name first, then a glob, then ``*``."""
    if not prices:
        return None
    if model and model in prices:
        return prices[model]
    if model:
        for pattern, rates in prices.items():
            if pattern not in ("*", model) and fnmatch.fnmatch(model, pattern):
                return rates
    return prices.get("*")


def cost_of(metadata, prices):
    """What the call *metadata* describes cost, or ``None`` if its model has no price.

    Thinking tokens bill as output, as ``storage.token_usage_summary`` does. Counts the providers
    record as -1 ("this response did not break that down") are read as zero rather than negative.
    """
    usage = metadata.get("token_usage")
    if not isinstance(usage, dict) or not usage:
        return None
    rates = price_for(metadata.get("model"), prices)
    if rates is None:
        return None

    def count(key):
        value = usage.get(key, 0)
        return value if isinstance(value, (int, float)) and value > 0 else 0

    text, images = count("prompt"), count("images")
    thinking, output, total = count("thinking"), count("result"), count("total")
    sent = text + images
    if not sent and total:
        # No per-modality breakdown: whatever is left of the total is what went in
        sent = max(total - thinking - output, 0)
    input_rate, output_rate = rates
    return (sent * input_rate / 1_000_000) + ((output + thinking) * output_rate / 1_000_000)


# --------------------------------------------------------------------------------------------------
# Writing the CHAI-TEA rows
# --------------------------------------------------------------------------------------------------


def _get_or_create(cursor, table, match, values):
    """The id of the row of *table* matching *match*, inserting it with *values* if there is none."""
    where = " AND ".join(f"{k} = %s" for k in match)
    cursor.execute(f"SELECT id FROM {table} WHERE {where} LIMIT 1", list(match.values()))
    row = cursor.fetchone()
    if row:
        return row[0]
    columns = {**match, **values}
    names = ", ".join(columns)
    holders = ", ".join(["%s"] * len(columns))
    cursor.execute(
        f"INSERT INTO {table} ({names}) VALUES ({holders}) RETURNING id", list(columns.values())
    )
    return cursor.fetchone()[0]


def import_records(
    records,
    params=None,
    project="imported",
    workflow=None,
    prices=None,
    tables=None,
    run_metadata=None,
    batch_size=1000,
    **overrides,
):
    """Insert *records* into the CHAI-TEA schema as one workflow run; returns a summary dict.

    *project* and *workflow* name the PROJECT and WORKFLOW rows the run hangs under, reused if rows
    of those names are already there and created if not. *workflow* defaults to the ``workflowId``
    the first record carries. Everything happens in one transaction: either the run and all its
    results land, or nothing does.
    """
    names = chai_tea_schema.table_names(tables)
    settings = chai_tea_schema.connection_settings(params, **overrides)
    prices = prices or {}

    conn = _pg_connect(postgres_params(settings))
    summary = {
        "records": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
        "with_usage": 0,
        "costed": 0,
        "cost": 0.0,
        "duration": 0.0,
        "models": {},
    }
    try:
        cursor = conn.cursor()
        batch = []
        links = []
        run_id = None
        last_run = None

        for record in records:
            row = normalize(record)
            if row["id"] is None:
                logger.warning("Record with no id skipped: %r", record)
                summary["skipped"] += 1
                continue
            summary["records"] += 1

            if run_id is None:
                # Deferred until the first record so the workflow can take its name from it
                project_id = _get_or_create(
                    cursor,
                    names["PROJECT"],
                    {"name": project},
                    {"description": "Created by chai_tea_import"},
                )
                workflow_name = workflow or row["workflow_id"] or "imported"
                workflow_id = _get_or_create(
                    cursor,
                    names["WORKFLOW"],
                    {"name": str(workflow_name), "project_id": project_id},
                    {"description": "Created by chai_tea_import"},
                )
                cursor.execute(
                    f"INSERT INTO {names['WORKFLOW_RUN']} (workflow_id, metadata) "
                    f"VALUES (%s, %s) RETURNING id",
                    [workflow_id, json.dumps(run_metadata or {})],
                )
                run_id = cursor.fetchone()[0]

            metadata = row["metadata"]
            if isinstance(metadata.get("token_usage"), dict) and metadata["token_usage"]:
                summary["with_usage"] += 1
                model = metadata.get("model") or "unknown"
                summary["models"][model] = summary["models"].get(model, 0) + 1
            if not row["was_successful"]:
                summary["errors"] += 1
            if isinstance(row["md_duration"], (int, float)):
                summary["duration"] += row["md_duration"]

            cost = cost_of(metadata, prices)
            if cost is not None:
                summary["costed"] += 1
                summary["cost"] += cost
            if row["md_timestamp"] and (last_run is None or row["md_timestamp"] > last_run):
                last_run = row["md_timestamp"]

            batch.append(
                [
                    row["id"],
                    run_id,
                    row["process_id"],
                    row["input"],
                    row["input_hash"],
                    row["input_segment"],
                    row["input_sequence"],
                    row["was_successful"],
                    json.dumps(row["value"]),
                    json.dumps(metadata),
                    json.dumps(row["extra_data"]) if row["extra_data"] is not None else None,
                    row["previous_id"],
                    cost,
                    row["md_duration"],
                    row["md_timestamp"],
                ]
            )
            if row["previous_id"]:
                links.append((row["id"], row["previous_id"]))
            if len(batch) >= batch_size:
                summary["inserted"] += _insert_results(cursor, names["RESULT"], batch)
                batch = []

        if batch:
            summary["inserted"] += _insert_results(cursor, names["RESULT"], batch)
        summary["linked"] = _link_previous(cursor, names["RESULT"], links)
        summary["dangling"] = len(links) - summary["linked"]

        if run_id is not None:
            cursor.execute(
                f"UPDATE {names['WORKFLOW_RUN']} SET was_successful = %s, duration = %s, "
                f"last_run = COALESCE(%s, CURRENT_TIMESTAMP) WHERE id = %s",
                [summary["errors"] == 0, summary["duration"], last_run, run_id],
            )
            summary["workflow_run_id"] = str(run_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    summary["duplicates"] = summary["records"] - summary["inserted"]
    return summary


RESULT_COLUMNS = (
    "id, workflow_run_id, process_id, input, input_hash, input_segment, input_sequence, "
    "was_successful, value, metadata, extra_data, previous_id, md_cost, md_duration, md_timestamp"
)


def _insert_results(cursor, table, batch):
    """Insert one batch of RESULT rows, skipping ids already there; returns how many landed.

    ``previous_id`` is left NULL here and linked up afterwards by ``_link_previous``: a result often
    precedes the row it points at within the same file, and the self foreign key would reject it on
    the way in. Inserting the rows first and pointing them at each other second makes the order the
    records happen to arrive in irrelevant.
    """
    holders = "(" + ", ".join(["%s"] * 14) + ", COALESCE(%s, CURRENT_TIMESTAMP))"
    sql = (
        f"INSERT INTO {table} ({RESULT_COLUMNS}) VALUES {holders} "
        f"ON CONFLICT (id) DO NOTHING RETURNING id"
    )
    inserted = 0
    for row in batch:
        cursor.execute(sql, [*row[:11], None, *row[12:]])
        if cursor.fetchone():
            inserted += 1
    return inserted


def _link_previous(cursor, table, links):
    """Set ``previous_id`` on the imported rows, ignoring targets that were never imported.

    The join against the table itself is what drops a dangling pointer -- a correction whose target
    is in neither this import nor the database already -- instead of failing the foreign key and
    losing the whole run.
    """
    if not links:
        return 0
    cursor.execute(
        f"UPDATE {table} AS r SET previous_id = v.previous_id "
        f"FROM (SELECT unnest(%s::uuid[]) AS id, unnest(%s::uuid[]) AS previous_id) AS v "
        f"JOIN {table} AS target ON target.id = v.previous_id "
        f"WHERE r.id = v.id AND r.previous_id IS DISTINCT FROM v.previous_id",
        [[a for a, _ in links], [b for _, b in links]],
    )
    return cursor.rowcount


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="the .jsonl/.parquet/.db file or postgres:// URL to read")
    parser.add_argument(
        "--source",
        dest="source_type",
        choices=["jsonl", "parquet", "sqlite", "postgres"],
        help="what kind of storage it is (default: from the extension)",
    )
    parser.add_argument("--source-table", default="results", help="table to read for a database source")
    parser.add_argument("--project", default="imported", help="CHAI-TEA project to import under")
    parser.add_argument("--workflow", default=None, help="workflow name (default: the run's workflowId)")
    parser.add_argument(
        "--price",
        action="append",
        metavar="MODEL=IN/OUT",
        help="dollars per million tokens for a model, e.g. gemini-3.1-flash-lite-preview=0.10/0.40; "
        "repeatable, and '*=IN/OUT' sets a fallback. Without it md_cost stays NULL.",
    )
    parser.add_argument("--dsn", default=None, help="target connection string; overrides the options below")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--database", default=None, help="target database (default: chai)")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    try:
        prices = parse_prices(args.price)
    except ValueError as e:
        parser.error(str(e))

    connection = {
        k: v
        for k, v in (
            ("dsn", args.dsn),
            ("host", args.host),
            ("port", args.port),
            ("database", args.database),
            ("user", args.user),
            ("password", args.password),
        )
        if v
    }
    records = read_records(args.source, args.source_type, args.source_table)
    summary = import_records(
        records,
        connection,
        project=args.project,
        workflow=args.workflow,
        prices=prices,
        run_metadata={"source": str(args.source), "imported": datetime.datetime.now().isoformat()},
    )

    print(
        f"{summary['records']} records read, {summary['inserted']} inserted"
        + (f", {summary['duplicates']} already present" if summary.get("duplicates") else "")
        + (f", {summary['errors']} failed steps" if summary["errors"] else "")
    )
    print(f"{summary['with_usage']} with AI token usage, {summary['duration']:.1f}s total duration")
    for model, count in sorted(summary["models"].items()):
        print(f"  {model}: {count}")
    if summary["costed"]:
        print(f"${summary['cost']:.4f} over {summary['costed']} costed calls")
    elif prices:
        print("No rows costed: no result carried a model with a matching price")
    return 0


if __name__ == "__main__":
    sys.exit(main())
