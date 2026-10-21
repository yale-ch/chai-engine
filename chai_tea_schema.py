"""Create the CHAI-TEA tables in PostgreSQL from the ER diagram in ``mermaid-er-database.md``.

One table per entity in the diagram (PROJECT, WORKFLOW, RESULT, ANNOTATION, USER, ROLE, and the two
permission tables), with the columns, types and keys the diagram declares. Connection settings come
from ``chai.storage.postgres_params`` -- the same ``dsn``/``host``/``port``/``database``/``user``
settings and ``PG*`` environment variables ``PostgresStorage`` uses -- and the database itself is
created if it is missing, like the storage components do.

    python chai_tea_schema.py --database chai_tea            # create (or top up) the tables
    python chai_tea_schema.py --database chai_tea --drop     # drop them first, then recreate
    python chai_tea_schema.py --sql                          # just print the DDL

Choices the diagram does not make, all of them in one place at the top of this module so they are
easy to change:

* ``USER`` becomes the table ``users``: ``user`` is a reserved word in PostgreSQL and would have to
  be quoted in every query. Every table name lives in ``TABLES``.
* Mermaid types map to PostgreSQL as uuid -> ``UUID`` (defaulting to ``gen_random_uuid()``), string
  -> ``TEXT``, int -> ``INTEGER``, float -> ``DOUBLE PRECISION``, JSON -> ``JSONB``, datetime ->
  ``TIMESTAMPTZ``.
* ``ON DELETE`` actions are not expressible in a mermaid ER diagram. Identifying relationships (the
  solid ``--`` lines) cascade, optional links (the dashed ``..`` lines, and the nullable self
  references) null out, and a ROLE cannot be deleted while permissions still point at it.
* ``RESULT.process_id`` and ``ANNOTATION.user_id`` are marked FK in the diagram but have no
  relationship line. ``user_id`` is wired to ``users`` (the only entity it can mean);
  ``process_id`` has no entity to point at, so it stays a plain column.
* The relationship ``ANNOTATION |o..o| ANNOTATION`` is one-to-one, so ``target_annotation_id`` is
  UNIQUE: an annotation can be annotated at most once. Drop that constraint from
  ``ANNOTATION_TABLE`` if annotations are meant to accept several replies.
* Each permission row is unique on (user, workflow/project, role), and every FK column is indexed.

The tables themselves are defined in ``chai/schema.py`` (re-exported here), so the storages and this
tool build exactly the same schema; ``SqliteStorage`` and ``PostgresStorage`` write it directly. A
table of one of these names left over from an older layout (the pre-CHAI-TEA ``results`` table
``PostgresStorage`` used to write, say) is detected and reported rather than silently left alone.
"""

import argparse
import sys

from chai.schema import (  # noqa: F401 -- the schema lives in the package; re-exported here
    CREATE_TABLES,
    INDEXED_COLUMNS,
    KINDS,
    MODIFIED_TABLES,
    MODIFIED_TRIGGER_FUNCTION,
    TABLE_ORDER,
    TABLES,
    check_for_clashes as _check_for_clashes,
    column_kinds,
    column_types,
    schema_statements,
    sqlite_statements,
    table_names,
)
from chai.storage import _pg_connect, ensure_postgres_database, postgres_params

def connection_settings(params=None, **overrides):
    """Connection settings, tolerating a dict ``postgres_params`` has already been through.

    ``postgres_params`` reads ``database``/``dsn`` but writes ``dbname``/``conninfo``, so feeding its
    own output back into it (or into anything that normalizes again, like
    ``ensure_postgres_database``) silently loses the database and falls back to the default one.
    Renaming the keys back makes either form safe to pass around.
    """
    settings = dict(params or {})
    if "dbname" in settings:
        settings.setdefault("database", settings.pop("dbname"))
    if "conninfo" in settings:
        settings.setdefault("dsn", settings.pop("conninfo"))
    settings.update(overrides)
    return settings


def create_schema(params=None, tables=None, drop_existing=False, create_database=True, **overrides):
    """Create the CHAI-TEA tables in PostgreSQL; returns the entity -> table name mapping.

    *params* and *overrides* are the ``postgres_params`` settings (``dsn`` or ``host``/``port``/
    ``database``/``user``/``password``). Unless *create_database* is false the database is created
    first if it does not exist, which needs an account allowed to create databases. Everything runs
    in one transaction, so a failure leaves the database as it was.
    """
    settings = connection_settings(params, **overrides)
    if create_database:
        ensure_postgres_database(settings)
    names = table_names(tables)
    conn = _pg_connect(postgres_params(settings))
    try:
        if not drop_existing:
            _check_for_clashes(conn, names)
        cursor = conn.cursor()
        for statement in schema_statements(tables, drop_existing):
            cursor.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=None, help="connection string; overrides the options below")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--database", default=None, help="database to build in (default: chai)")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--drop", action="store_true", help="drop the tables first, losing their data")
    parser.add_argument(
        "--no-create-database", action="store_true", help="fail instead of creating a missing database"
    )
    parser.add_argument("--sql", action="store_true", help="print the DDL instead of running it")
    args = parser.parse_args(argv)

    if args.sql:
        print(";\n\n".join(schema_statements(drop_existing=args.drop)) + ";")
        return 0

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
    names = create_schema(
        connection, drop_existing=args.drop, create_database=not args.no_create_database
    )
    params = postgres_params(connection)
    where = params.get("conninfo") or f"{params['host']}:{params['port']}/{params['dbname']}"
    print(f"Created {len(names)} tables in {where}: {', '.join(sorted(names.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
