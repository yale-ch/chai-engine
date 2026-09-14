"""Move CHAI-TEA results and annotations out of a local PostgreSQL instance and into a central one.

The Parquet route ``ParquetStorage``/``parquet_to_postgres`` take for a single run's ``results``
table, applied to the CHAI-TEA schema ``chai_tea_schema.py`` builds: ``export_parquet`` reads the
local database and writes one Parquet file per table into a directory, and ``load_parquet`` loads
that directory into the central database. The export is a bundle rather than one file because the
tables have different shapes; a directory of typed files keeps each table's columns its own.

    # on the local instance
    python chai_tea_parquet.py export ./upload --database chai_tea
    python chai_tea_parquet.py export ./upload --database chai_tea --since 2026-09-01

    # against the central database
    python chai_tea_parquet.py load ./upload --host master.example.edu --database chai_tea

RESULT and ANNOTATION are the payload; the rows they point at come with them, so the bundle can be
loaded into an empty central database without a foreign key failing:

* the WORKFLOW of every exported result, and that workflow's ``previous_id`` chain
* the PROJECT of every exported workflow
* the USER named by an exported result's ``editor_user_id`` or an annotation's ``user_id``
* every result in an exported result's ``previous_id`` chain, and every annotation in an exported
  annotation's ``target_annotation_id`` chain -- in both directions, so replies travel too
* ROLE and the two permission tables only with ``--include-permissions``: access control is usually
  the central database's own business, not something a local instance should overwrite

That closure is what makes ``--since`` safe to use for a top-up: the new results come with their
ancestors, so nothing dangles even if the central database has never seen this project. It also
means ``--since`` re-sends those ancestors and their annotations, which a load already holding them
skips -- the timestamp bounds what the export is *about*, not every row it carries.

Loading is one transaction -- either the whole bundle lands or none of it does -- and is idempotent:
rows already in the central database are left alone (``--on-conflict skip``, the default) or
overwritten from the file (``--on-conflict update``). Each table is COPYed into a temporary table
and then inserted from there in a single statement, which is both faster than row-by-row inserts and
what lets ``results`` and ``annotations`` carry references within their own table regardless of the
order the rows happen to be in.

Needs ``pyarrow`` (as the other Parquet paths do) and ``psycopg``.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import uuid

import chai_tea_schema
from chai.storage import (
    ParquetRecordWriter,
    _pg_connect,
    _pg_copy_rows,
    _pg_identifier,
    ensure_postgres_database,
    parquet_columns,
    postgres_params,
)

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
MANIFEST_VERSION = 1

# The tables a results-and-annotations export carries, parents before children.
EXPORT_ORDER = ["PROJECT", "USER", "WORKFLOW", "RESULT", "ANNOTATION"]

# Added to it by --include-permissions, keeping the same parents-first order.
PERMISSION_ORDER = ["ROLE", "WF_PERMISSION", "PJ_PERMISSION"]

# The CHAI-TEA schema's SQL types as ParquetRecordWriter column kinds. Declaring them (rather than
# letting the writer infer from the data) means every export of a table has the same columns and
# types, including an export where the table came out empty.
KINDS = {
    "UUID": "string",
    "TEXT": "string",
    "INTEGER": "int",
    "DOUBLE PRECISION": "float",
    "JSONB": "json",
    "TIMESTAMPTZ": "timestamp",
}


def column_kinds(entity):
    """``{column: Parquet column kind}`` for one entity of the CHAI-TEA schema."""
    types = chai_tea_schema.column_types(entity)
    unknown = {t for t in types.values() if t not in KINDS}
    if unknown:
        raise ValueError(f"No Parquet column kind for SQL type(s) {', '.join(sorted(unknown))}")
    return {column: KINDS[sql_type] for column, sql_type in types.items()}


def export_order(include_permissions=False):
    """The entities an export writes, in the order they have to be loaded."""
    if not include_permissions:
        return list(EXPORT_ORDER)
    order = []
    for entity in chai_tea_schema.TABLE_ORDER:
        if entity in EXPORT_ORDER or entity in PERMISSION_ORDER:
            order.append(entity)
    return order


# --------------------------------------------------------------------------------------------------
# Selecting what to export
#
# Each step puts the ids of one table's selected rows into a temporary table, so the queries that
# follow can join against a small set instead of repeating the filters. Ordered so that every step
# only needs sets that are already built.
# --------------------------------------------------------------------------------------------------

# Workflows the filters name; with no filters, all of them, so an empty workflow travels too.
SELECT_WORKFLOW_SEED = """
CREATE TEMP TABLE sel_wf_seed AS
SELECT w.id FROM {WORKFLOW} w JOIN {PROJECT} p ON p.id = w.project_id
WHERE (%(project)s::text IS NULL OR p.id::text = %(project)s::text OR p.name = %(project)s::text)
  AND (%(workflow)s::text IS NULL OR w.id::text = %(workflow)s::text OR w.name = %(workflow)s::text)
"""

# The results of those workflows made at or after --since, plus each one's previous_id ancestors.
SELECT_RESULTS = """
CREATE TEMP TABLE sel_results AS
WITH RECURSIVE chain AS (
    SELECT r.id, r.previous_id FROM {RESULT} r
    WHERE r.workflow_id IN (SELECT id FROM sel_wf_seed)
      AND (%(since)s::timestamptz IS NULL OR r.md_timestamp >= %(since)s::timestamptz)
  UNION
    SELECT r.id, r.previous_id FROM {RESULT} r JOIN chain c ON r.id = c.previous_id
)
SELECT DISTINCT id FROM chain
"""

# The annotations on those results, walking target_annotation_id both ways: the annotation an
# exported one annotates has to come (the foreign key needs it), and the ones annotating it should.
SELECT_ANNOTATIONS = """
CREATE TEMP TABLE sel_annotations AS
WITH RECURSIVE chain AS (
    SELECT a.id, a.target_annotation_id FROM {ANNOTATION} a
    WHERE a.target_result_id IN (SELECT id FROM sel_results)
  UNION
    SELECT a.id, a.target_annotation_id FROM {ANNOTATION} a JOIN chain c
      ON a.id = c.target_annotation_id OR a.target_annotation_id = c.id
)
SELECT DISTINCT id FROM chain
"""

# An annotation pulled in by that chain can point at a result outside the selection; it comes too.
EXPAND_RESULTS = """
INSERT INTO sel_results
WITH RECURSIVE chain AS (
    SELECT r.id, r.previous_id FROM {RESULT} r
    WHERE r.id IN (
        SELECT target_result_id FROM {ANNOTATION}
        WHERE id IN (SELECT id FROM sel_annotations) AND target_result_id IS NOT NULL
    )
  UNION
    SELECT r.id, r.previous_id FROM {RESULT} r JOIN chain c ON r.id = c.previous_id
)
SELECT DISTINCT id FROM chain WHERE id NOT IN (SELECT id FROM sel_results)
"""

SELECT_WORKFLOWS = """
CREATE TEMP TABLE sel_workflows AS
WITH RECURSIVE chain AS (
    SELECT w.id, w.previous_id FROM {WORKFLOW} w
    WHERE w.id IN (SELECT id FROM sel_wf_seed)
       OR w.id IN (SELECT workflow_id FROM {RESULT} WHERE id IN (SELECT id FROM sel_results))
  UNION
    SELECT w.id, w.previous_id FROM {WORKFLOW} w JOIN chain c ON w.id = c.previous_id
)
SELECT DISTINCT id FROM chain
"""

# Projects the exported workflows belong to; with no --workflow filter, the ones --project names as
# well, so a project with no workflows in it still reaches the central database.
SELECT_PROJECTS = """
CREATE TEMP TABLE sel_projects AS
SELECT DISTINCT project_id AS id FROM {WORKFLOW} WHERE id IN (SELECT id FROM sel_workflows)
UNION
SELECT p.id FROM {PROJECT} p
WHERE %(workflow)s::text IS NULL
  AND (%(project)s::text IS NULL OR p.id::text = %(project)s::text OR p.name = %(project)s::text)
"""

SELECT_PERMISSIONS = [
    """
    CREATE TEMP TABLE sel_wf_permissions AS
    SELECT id FROM {WF_PERMISSION} WHERE workflow_id IN (SELECT id FROM sel_workflows)
    """,
    """
    CREATE TEMP TABLE sel_pj_permissions AS
    SELECT id FROM {PJ_PERMISSION} WHERE project_id IN (SELECT id FROM sel_projects)
    """,
    """
    CREATE TEMP TABLE sel_roles AS
    SELECT id FROM {ROLE} WHERE id IN (
        SELECT role_id FROM {WF_PERMISSION} WHERE id IN (SELECT id FROM sel_wf_permissions)
        UNION
        SELECT role_id FROM {PJ_PERMISSION} WHERE id IN (SELECT id FROM sel_pj_permissions)
    )
    """,
]

# Last, because which users are needed depends on everything selected above.
SELECT_USERS = """
CREATE TEMP TABLE sel_users AS
SELECT DISTINCT editor_user_id AS id FROM {RESULT}
WHERE id IN (SELECT id FROM sel_results) AND editor_user_id IS NOT NULL
UNION
SELECT DISTINCT user_id FROM {ANNOTATION}
WHERE id IN (SELECT id FROM sel_annotations) AND user_id IS NOT NULL
"""

SELECT_PERMISSION_USERS = """
INSERT INTO sel_users
SELECT id FROM (
    SELECT user_id AS id FROM {WF_PERMISSION} WHERE id IN (SELECT id FROM sel_wf_permissions)
    UNION
    SELECT user_id FROM {PJ_PERMISSION} WHERE id IN (SELECT id FROM sel_pj_permissions)
) u
WHERE id NOT IN (SELECT id FROM sel_users)
"""

# Entity -> the temporary table holding its selected ids.
SELECTIONS = {
    "PROJECT": "sel_projects",
    "USER": "sel_users",
    "ROLE": "sel_roles",
    "WORKFLOW": "sel_workflows",
    "WF_PERMISSION": "sel_wf_permissions",
    "PJ_PERMISSION": "sel_pj_permissions",
    "RESULT": "sel_results",
    "ANNOTATION": "sel_annotations",
}


def _select(cursor, names, since=None, project=None, workflow=None, include_permissions=False):
    """Build the temporary id tables for one export; returns ``{entity: rows selected}``."""
    filters = {"since": since, "project": project, "workflow": workflow}
    statements = [SELECT_WORKFLOW_SEED, SELECT_RESULTS, SELECT_ANNOTATIONS, EXPAND_RESULTS]
    statements += [SELECT_WORKFLOWS, SELECT_PROJECTS]
    if include_permissions:
        statements += SELECT_PERMISSIONS
    statements.append(SELECT_USERS)
    if include_permissions:
        statements.append(SELECT_PERMISSION_USERS)
    for statement in statements:
        cursor.execute(statement.format(**names), filters)
    counts = {}
    for entity in export_order(include_permissions):
        cursor.execute(f"SELECT count(*) FROM {SELECTIONS[entity]}")
        counts[entity] = cursor.fetchone()[0]
    return counts


def _stream(conn, sql, batch_size):
    """Yield batches of rows for *sql*, through a server-side cursor where the driver has one.

    Keeps a table too big to hold in memory from having to be, which a local instance's ``results``
    can well be. Falls back to an ordinary cursor for a driver that will not name one.
    """
    cursor = None
    try:
        try:
            cursor = conn.cursor(name=f"chai_tea_export_{uuid.uuid4().hex[:8]}")
        except Exception:
            cursor = conn.cursor()
        cursor.execute(sql)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                return
            yield rows
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:  # pragma: no cover - a server cursor already gone
                pass


def export_parquet(
    directory,
    params=None,
    since=None,
    project=None,
    workflow=None,
    include_permissions=False,
    tables=None,
    batch_size=50000,
    compression="snappy",
    **overrides,
):
    """Write the local CHAI-TEA results and annotations to a Parquet bundle in *directory*.

    Returns the manifest dict, which is also written to ``manifest.json`` beside the files.
    Arguments:
        - directory: where to put the bundle (created if it is not there; existing table files of
          the same names are overwritten)
        - params / dsn / host / port / database / user / password: the local database (see
          ``postgres_params``)
        - since: export results and annotations from this timestamp on (anything PostgreSQL reads as
          a ``timestamptz``), for topping up a central database rather than sending everything
        - project / workflow: limit the export to one project or workflow, by name or by id
        - include_permissions: also export ROLE and the two permission tables
        - tables: table name overrides, as ``chai_tea_schema.TABLES``
        - batch_size: rows per batch read from the server and per Parquet row group
        - compression: Parquet codec ('snappy', 'zstd', 'gzip', 'none')
    """
    names = chai_tea_schema.table_names(tables)
    entities = export_order(include_permissions)
    settings = chai_tea_schema.connection_settings(params, **overrides)
    connection = postgres_params(settings)
    os.makedirs(directory, exist_ok=True)

    conn = _pg_connect(connection)
    try:
        counts = _select(
            conn.cursor(),
            names,
            since=since,
            project=project,
            workflow=workflow,
            include_permissions=include_permissions,
        )
        exported = []
        for entity in entities:
            kinds = column_kinds(entity)
            table = names[entity]
            path = os.path.join(directory, f"{table}.parquet")
            writer = ParquetRecordWriter(
                path,
                schema=kinds,
                batch_size=batch_size,
                compression=compression,
                metadata={"chai_tea_entity": entity, "chai_tea_table": table},
            )
            columns = ", ".join(_pg_identifier(c) for c in kinds)
            sql = f"SELECT {columns} FROM {table} WHERE id IN (SELECT id FROM {SELECTIONS[entity]})"
            try:
                for rows in _stream(conn, sql, batch_size):
                    writer.add_all(dict(zip(kinds, row)) for row in rows)
            finally:
                written = writer.close()  # closed even half-written, so nothing holds the file open
            if written["rows"] != counts[entity]:  # a concurrent delete, or a filter gone wrong
                logger.warning(
                    f"Selected {counts[entity]} {entity} rows but wrote {written['rows']} to {path}"
                )
            exported.append(
                {
                    "entity": entity,
                    "table": table,
                    "file": os.path.basename(path),
                    "rows": written["rows"],
                    "columns": kinds,
                }
            )
        conn.rollback()  # nothing was written to the database; drop the temporary tables
    finally:
        conn.close()

    manifest = {
        "chai_tea_export": MANIFEST_VERSION,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            key: connection[key] for key in ("host", "port", "dbname", "conninfo") if key in connection
        },
        "filters": {"since": str(since) if since else None, "project": project, "workflow": workflow},
        "include_permissions": bool(include_permissions),
        "tables": exported,
    }
    with open(os.path.join(directory, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(t["rows"] for t in exported)
    logger.info(f"Exported {total} rows from {len(exported)} tables to {directory}")
    return manifest


def read_manifest(directory):
    """The manifest of a bundle *directory*, or one built from the files there if it has none."""
    path = os.path.join(directory, MANIFEST)
    if os.path.exists(path):
        with open(path) as fh:
            manifest = json.load(fh)
        version = manifest.get("chai_tea_export")
        if version != MANIFEST_VERSION:
            raise ValueError(f"{path} is a version {version} export; this is version {MANIFEST_VERSION}")
        return manifest
    names = chai_tea_schema.table_names()
    tables = []
    for entity in export_order(include_permissions=True):
        file_name = f"{names[entity]}.parquet"
        if os.path.exists(os.path.join(directory, file_name)):
            tables.append({"entity": entity, "table": names[entity], "file": file_name, "rows": None})
    if not tables:
        raise ValueError(f"No CHAI-TEA Parquet files in {directory}")
    return {"chai_tea_export": MANIFEST_VERSION, "tables": tables}


def load_parquet(
    directory,
    params=None,
    on_conflict="skip",
    create=True,
    batch_size=10000,
    create_database=True,
    tables=None,
    **overrides,
):
    """Load a bundle written by ``export_parquet`` into the central PostgreSQL database.

    The whole bundle goes in one transaction, so a foreign key that cannot be satisfied -- a
    ``--since`` export whose parents are somehow missing, say -- leaves the central database exactly
    as it was. Each table is COPYed into a temporary table and inserted from there in a single
    statement, so rows referring to other rows of the same table do not have to be in any order.

    Returns ``{entity: {"table", "staged", "written", "skipped"}}``. Arguments:
        - directory: the bundle to load
        - params / dsn / host / port / database / user / password: the central database
        - on_conflict: 'skip' (default) leaves rows the central database already has alone;
          'update' overwrites them from the file. Either way the load is safe to repeat.
        - create: create the CHAI-TEA tables there if they are missing (default true)
        - batch_size: rows per batch read from each Parquet file
        - create_database: create the database if the server does not have it yet (default true)
        - tables: table name overrides, as ``chai_tea_schema.TABLES``
    """
    import pyarrow.parquet as pq

    if on_conflict not in ("skip", "update"):
        raise ValueError(f"on_conflict is 'skip' or 'update', not {on_conflict!r}")
    manifest = read_manifest(directory)
    names = chai_tea_schema.table_names(tables)
    settings = chai_tea_schema.connection_settings(params, **overrides)
    if create:
        chai_tea_schema.create_schema(settings, tables=tables, create_database=create_database)
    elif create_database:
        ensure_postgres_database(settings)

    loaded = {}
    conn = _pg_connect(postgres_params(settings))
    try:
        cursor = conn.cursor()
        wanted = [names.get(e["entity"], e["table"]) for e in manifest["tables"]]
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
            (wanted,),
        )
        present = {row[0] for row in cursor.fetchall()}
        absent = [t for t in wanted if t not in present]
        if absent:  # clearer than the raw 'relation does not exist' the first COPY would raise
            raise RuntimeError(
                f"No {', '.join(absent)} table in this database. Run chai_tea_schema.py against it "
                f"first, or load without --no-create."
            )
        for entry in manifest["tables"]:
            entity = entry["entity"]
            table = _pg_identifier(names.get(entity, entry["table"]))
            path = os.path.join(directory, entry["file"])
            kinds = parquet_columns(path)
            columns = list(kinds)
            quoted = ", ".join(_pg_identifier(c) for c in columns)
            stage = f"stage_{table}"
            cursor.execute(f"CREATE TEMP TABLE {stage} AS SELECT {quoted} FROM {table} WITH NO DATA")
            staged = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
                records = batch.to_pylist()
                if records:
                    staged += _pg_copy_rows(cursor, stage, quoted, columns, records, kinds)
            if on_conflict == "update":
                # DISTINCT ON keeps ON CONFLICT DO UPDATE from being handed the same id twice
                assignments = ", ".join(
                    f"{_pg_identifier(c)} = EXCLUDED.{_pg_identifier(c)}" for c in columns if c != "id"
                )
                cursor.execute(
                    f"INSERT INTO {table} ({quoted}) "
                    f"SELECT {quoted} FROM (SELECT DISTINCT ON (id) {quoted} FROM {stage} ORDER BY id) s "
                    f"ON CONFLICT (id) DO UPDATE SET {assignments}"
                )
            else:
                # No conflict target, so a row already there is left alone whichever constraint says so
                cursor.execute(
                    f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM {stage} ON CONFLICT DO NOTHING"
                )
            written = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else staged
            cursor.execute(f"DROP TABLE {stage}")
            loaded[entity] = {
                "table": table,
                "staged": staged,
                "written": written,
                "skipped": max(staged - written, 0),
            }
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    total = sum(t["written"] for t in loaded.values())
    logger.info(f"Loaded {total} rows from {directory} into {len(loaded)} tables")
    return loaded


def _connection_args(parser):
    """The connection options both subcommands take."""
    parser.add_argument("--dsn", default=None, help="connection string; overrides the options below")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--database", default=None, help="default: chai")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)


def _connection(args):
    return {
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write the local results and annotations to a bundle")
    export.add_argument("directory", help="where to write the Parquet files and manifest.json")
    export.add_argument("--since", default=None, help="only results and annotations from this time on")
    export.add_argument("--project", default=None, help="limit to one project, by name or id")
    export.add_argument("--workflow", default=None, help="limit to one workflow, by name or id")
    export.add_argument(
        "--include-permissions", action="store_true", help="also export roles and permissions"
    )
    export.add_argument("--batch-size", type=int, default=50000)
    export.add_argument("--compression", default="snappy")
    _connection_args(export)

    load = sub.add_parser("load", help="load a bundle into the central database")
    load.add_argument("directory", help="the bundle to load")
    load.add_argument(
        "--on-conflict",
        choices=("skip", "update"),
        default="skip",
        help="rows the central database already has: leave them (default) or overwrite them",
    )
    load.add_argument(
        "--no-create", action="store_true", help="fail instead of creating missing tables there"
    )
    load.add_argument("--batch-size", type=int, default=10000)
    _connection_args(load)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "export":
        manifest = export_parquet(
            args.directory,
            _connection(args),
            since=args.since,
            project=args.project,
            workflow=args.workflow,
            include_permissions=args.include_permissions,
            batch_size=args.batch_size,
            compression=args.compression,
        )
        for entry in manifest["tables"]:
            print(f"  {entry['rows']:>9,} {entry['table']} -> {entry['file']}")
        print(f"Wrote {args.directory}/{MANIFEST}")
    else:
        loaded = load_parquet(
            args.directory,
            _connection(args),
            on_conflict=args.on_conflict,
            create=not args.no_create,
            batch_size=args.batch_size,
        )
        for entity, counts in loaded.items():
            print(
                f"  {counts['written']:>9,} written, {counts['skipped']:>9,} already there"
                f"  {counts['table']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
