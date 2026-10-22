"""PostgreSQL: connecting, creating the CHAI-TEA tables, writing rows, and loading Parquet files.

``postgres_params`` reads the ``dsn``/``host``/``port``/``database``/``user``/``password`` settings and
the ``PG*`` environment variables; ``store_postgres_rows`` upserts rows made elsewhere, and
``parquet_to_postgres``/``load_run_parquet`` bulk load a run's Parquet file.
"""

import logging
import os

from .parquet import parquet_columns, parquet_run_records
from .records import as_uuid
from .rows import _correction_metadata, correction_row
from .schema import check_for_clashes, column_types, schema_statements, table_names
from .schema import identifier as _pg_identifier
from .sql import write_rows

logger = logging.getLogger("chai")


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


def _pg_ensure_chai_tea(conn, tables=None):
    """Create the CHAI-TEA tables in *conn*'s database unless they are all there; returns their names.

    A table of one of the names that has other columns -- the pre-CHAI-TEA ``results`` table, say --
    raises rather than being written into (see ``chai.persistence.schema.check_for_clashes``).
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


def load_run_parquet(parquet_file, params=None, tables=None, batch_size=10000, **overrides):
    """Load a run's Parquet file of RESULT rows into the CHAI-TEA tables in PostgreSQL.

    The two steps a loader takes, in order: the PROJECT, WORKFLOW and WORKFLOW_RUN rows the file
    carries (``parquet_run_records``) are upserted first, then the RESULT rows are bulk loaded with
    ``parquet_to_postgres``. Creates the database and tables if missing. A file carrying no run rows
    loads only if the database already holds the runs its rows name. Returns ``{"records", "rows"}``:
    run rows written and RESULT rows loaded.
    """
    params = postgres_params(params, **overrides)
    names = ensure_postgres_schema(params, tables)
    written = store_postgres_rows(parquet_run_records(parquet_file), params, tables, create_schema=False)
    loaded = parquet_to_postgres(
        parquet_file, names["RESULT"], params, create=False, batch_size=batch_size, create_database=False
    )
    return {"records": written, "rows": loaded["rows"]}


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
