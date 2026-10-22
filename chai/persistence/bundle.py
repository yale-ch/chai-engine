"""Moving CHAI-TEA rows between PostgreSQL databases as a bundle of Parquet files, one per table.

``postgres_to_parquet`` exports the results and annotations of a local instance -- with every row
they reference, so nothing dangles -- and ``parquet_bundle_to_postgres`` loads that into a central
one. Selecting what to export: each step puts the ids of one table's selected rows into a temporary
table, so the queries that follow can join against a small set instead of repeating the filters,
ordered so that every step only needs sets that are already built.
"""

import contextlib
import logging
import os
import uuid

import ujson as json

from .parquet import ParquetRecordWriter, parquet_columns
from .postgres import _pg_connect, _pg_copy_rows, ensure_postgres_database, ensure_postgres_schema, postgres_params
from .records import now
from .schema import TABLE_ORDER, column_kinds, table_names
from .schema import identifier as _pg_identifier

logger = logging.getLogger("chai")


#: The bundle's own description of what it holds, beside the table files.
BUNDLE_MANIFEST = "manifest.json"
BUNDLE_VERSION = 1

#: The tables a results-and-annotations export carries, parents before children.
BUNDLE_ORDER = ["PROJECT", "USER", "WORKFLOW", "WORKFLOW_RUN", "RESULT", "ANNOTATION"]

#: Added to it by ``include_permissions``, keeping the same parents-first order.
PERMISSION_ORDER = ["ROLE", "WF_PERMISSION", "PJ_PERMISSION"]


def bundle_order(include_permissions=False):
    """The entities a bundle holds, in the order they have to be loaded."""
    if not include_permissions:
        return list(BUNDLE_ORDER)
    return [entity for entity in TABLE_ORDER if entity in BUNDLE_ORDER or entity in PERMISSION_ORDER]


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
    JOIN {WORKFLOW_RUN} wr ON wr.id = r.workflow_run_id
    WHERE wr.workflow_id IN (SELECT id FROM sel_wf_seed)
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

# Every run an exported result belongs to; RESULT.workflow_run_id is NOT NULL, so the run has to
# travel with it. Runs of a named workflow come too, so an empty run reaches the central database.
SELECT_WORKFLOW_RUNS = """
CREATE TEMP TABLE sel_workflow_runs AS
SELECT DISTINCT wr.id FROM {WORKFLOW_RUN} wr
WHERE wr.id IN (SELECT workflow_run_id FROM {RESULT} WHERE id IN (SELECT id FROM sel_results))
   OR wr.workflow_id IN (SELECT id FROM sel_wf_seed)
"""

SELECT_WORKFLOWS = """
CREATE TEMP TABLE sel_workflows AS
WITH RECURSIVE chain AS (
    SELECT w.id, w.previous_id FROM {WORKFLOW} w
    WHERE w.id IN (SELECT id FROM sel_wf_seed)
       OR w.id IN (SELECT workflow_id FROM {WORKFLOW_RUN} WHERE id IN (SELECT id FROM sel_workflow_runs))
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
    "WORKFLOW_RUN": "sel_workflow_runs",
    "WF_PERMISSION": "sel_wf_permissions",
    "PJ_PERMISSION": "sel_pj_permissions",
    "RESULT": "sel_results",
    "ANNOTATION": "sel_annotations",
}


def _select(cursor, names, since=None, project=None, workflow=None, include_permissions=False):
    """Build the temporary id tables for one export; returns ``{entity: rows selected}``."""
    filters = {"since": since, "project": project, "workflow": workflow}
    statements = [SELECT_WORKFLOW_SEED, SELECT_RESULTS, SELECT_ANNOTATIONS, EXPAND_RESULTS]
    statements += [SELECT_WORKFLOW_RUNS, SELECT_WORKFLOWS, SELECT_PROJECTS]
    if include_permissions:
        statements += SELECT_PERMISSIONS
    statements.append(SELECT_USERS)
    if include_permissions:
        statements.append(SELECT_PERMISSION_USERS)
    for statement in statements:
        cursor.execute(statement.format(**names), filters)
    counts = {}
    for entity in bundle_order(include_permissions):
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
            cursor = conn.cursor(name=f"chai_tea_bundle_{uuid.uuid4().hex[:8]}")
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


def postgres_to_parquet(
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
    """Write the CHAI-TEA results and annotations of a PostgreSQL database to a Parquet bundle in *directory*.

    One file per table, typed by the schema (``column_kinds``), and a ``manifest.json``. RESULT and
    ANNOTATION are the payload; the rows they point at come with them, so the bundle loads into an
    empty central database without a foreign key failing: the WORKFLOW_RUN of every exported result,
    its WORKFLOW and that workflow's ``previous_id`` chain, their PROJECTs, the USERs named by
    ``editor_user_id``/``user_id``, every result in a ``previous_id`` chain and every annotation in a
    ``target_annotation_id`` chain (both ways, so replies travel too). ROLE and the permission tables
    only with *include_permissions* -- access control is usually the central database's own business.

    Returns the manifest dict. Arguments:
        - directory: where to put the bundle (created if missing; table files of the same names are
          overwritten)
        - params / dsn / host / port / database / user / password: the database to export from (see
          ``postgres_params``)
        - since: only results and annotations from this timestamp on (anything PostgreSQL reads as a
          ``timestamptz``), for topping up a central database. Their ancestors still come, so a
          ``since`` export re-sends rows a load already holding them skips
        - project / workflow: limit the export to one project or workflow, by name or by id
        - include_permissions: also export ROLE and the two permission tables
        - tables: table name overrides, as ``chai.persistence.schema.TABLES``
        - batch_size: rows per batch read from the server and per Parquet row group
        - compression: Parquet codec ('snappy', 'zstd', 'gzip', 'none')
    """
    names = table_names(tables)
    entities = bundle_order(include_permissions)
    connection = postgres_params(params, **overrides)
    os.makedirs(directory, exist_ok=True)

    conn = _pg_connect(connection)
    try:
        counts = _select(
            conn.cursor(), names, since=since, project=project, workflow=workflow, include_permissions=include_permissions
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
                logger.warning(f"Selected {counts[entity]} {entity} rows but wrote {written['rows']} to {path}")
            exported.append(
                {"entity": entity, "table": table, "file": os.path.basename(path), "rows": written["rows"], "columns": kinds}
            )
        conn.rollback()  # nothing was written to the database; drop the temporary tables
    finally:
        conn.close()

    manifest = {
        "chai_tea_export": BUNDLE_VERSION,
        "exported_at": now().isoformat(),
        "source": {key: connection[key] for key in ("host", "port", "dbname", "conninfo") if key in connection},
        "filters": {"since": str(since) if since else None, "project": project, "workflow": workflow},
        "include_permissions": bool(include_permissions),
        "tables": exported,
    }
    with open(os.path.join(directory, BUNDLE_MANIFEST), "w") as fh:
        fh.write(json.dumps(manifest, indent=2))
    total = sum(t["rows"] for t in exported)
    logger.info(f"Exported {total} rows from {len(exported)} tables to {directory}")
    return manifest


def read_bundle_manifest(directory):
    """The manifest of a bundle *directory*, or one built from the files there if it has none."""
    path = os.path.join(directory, BUNDLE_MANIFEST)
    if os.path.exists(path):
        with open(path) as fh:
            manifest = json.load(fh)
        version = manifest.get("chai_tea_export")
        if version != BUNDLE_VERSION:
            raise ValueError(f"{path} is a version {version} export; this is version {BUNDLE_VERSION}")
        return manifest
    names = table_names()
    tables = []
    for entity in bundle_order(include_permissions=True):
        file_name = f"{names[entity]}.parquet"
        if os.path.exists(os.path.join(directory, file_name)):
            tables.append({"entity": entity, "table": names[entity], "file": file_name, "rows": None})
    if not tables:
        raise ValueError(f"No CHAI-TEA Parquet files in {directory}")
    return {"chai_tea_export": BUNDLE_VERSION, "tables": tables}


@contextlib.contextmanager
def _triggers_suspended(cursor, table):
    """Turn off *table*'s own triggers for the duration of a load, if this account may.

    A transfer copies rows verbatim, ``modified`` included -- but the schema's ``modified`` trigger
    would stamp every updated row with the load time instead, so a row that crossed unchanged would
    come out looking edited. Disabling the trigger is transactional, so a failed load leaves it
    enabled either way. An account that does not own the table cannot disable them; the load still
    runs, and only ``modified`` is affected.
    """
    disabled = False
    try:
        cursor.execute(f"ALTER TABLE {table} DISABLE TRIGGER USER")
        disabled = True
    except Exception as e:
        logger.debug("Could not disable triggers on %s (%s); modified will be restamped", table, e)
    try:
        yield
    finally:
        if disabled:
            cursor.execute(f"ALTER TABLE {table} ENABLE TRIGGER USER")


def parquet_bundle_to_postgres(
    directory,
    params=None,
    on_conflict="skip",
    create=True,
    batch_size=10000,
    create_database=True,
    tables=None,
    **overrides,
):
    """Load a bundle written by ``postgres_to_parquet`` into a (central) PostgreSQL database.

    The whole bundle goes in one transaction, so a foreign key that cannot be satisfied leaves the
    database exactly as it was. Each table is COPYed into a temporary table and inserted from there
    in a single statement, so rows referring to other rows of the same table need not be in order.

    Returns ``{entity: {"table", "staged", "written", "skipped"}}``. Arguments:
        - directory: the bundle to load
        - params / dsn / host / port / database / user / password: the database to load into
        - on_conflict: 'skip' (default) leaves rows the database already has alone; 'update'
          overwrites them from the file. Either way the load is safe to repeat.
        - create: create the CHAI-TEA tables if they are missing (default true)
        - batch_size: rows per batch read from each Parquet file
        - create_database: create the database if the server does not have it yet (default true)
        - tables: table name overrides, as ``chai.persistence.schema.TABLES``
    """
    import pyarrow.parquet as pq

    if on_conflict not in ("skip", "update"):
        raise ValueError(f"on_conflict is 'skip' or 'update', not {on_conflict!r}")
    manifest = read_bundle_manifest(directory)
    names = table_names(tables)
    params = postgres_params(params, **overrides)
    if create:
        ensure_postgres_schema(params, tables, create_database=create_database)
    elif create_database:
        ensure_postgres_database(params)

    loaded = {}
    conn = _pg_connect(params)
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
                f"No {', '.join(absent)} table in this database. Create the CHAI-TEA tables first "
                f"(ensure_postgres_schema, chai_tea_schema.py), or load with create=True."
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
                assignments = ", ".join(f"{_pg_identifier(c)} = EXCLUDED.{_pg_identifier(c)}" for c in columns if c != "id")
                with _triggers_suspended(cursor, table):
                    cursor.execute(
                        f"INSERT INTO {table} ({quoted}) "
                        f"SELECT {quoted} FROM (SELECT DISTINCT ON (id) {quoted} FROM {stage} ORDER BY id) s "
                        f"ON CONFLICT (id) DO UPDATE SET {assignments}"
                    )
            else:
                # No conflict target, so a row already there is left alone whichever constraint says so
                cursor.execute(f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM {stage} ON CONFLICT DO NOTHING")
            written = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else staged
            cursor.execute(f"DROP TABLE {stage}")
            loaded[entity] = {"table": table, "staged": staged, "written": written, "skipped": max(staged - written, 0)}
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    total = sum(t["written"] for t in loaded.values())
    logger.info(f"Loaded {total} rows from {directory} into {len(loaded)} tables")
    return loaded
