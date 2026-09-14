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

``RESULT`` becomes a table called ``results``, which is also the default table ``PostgresStorage``
writes to -- a different shape entirely. Build this schema in a database of its own (or rename the
table in ``TABLES``); a clashing table is detected and reported rather than silently left alone.
"""

import argparse
import sys

from chai.storage import _pg_connect, _pg_identifier, ensure_postgres_database, postgres_params

# Diagram entity -> table name. USER is renamed because it is a PostgreSQL reserved word.
TABLES = {
    "PROJECT": "projects",
    "USER": "users",
    "ROLE": "roles",
    "WORKFLOW": "workflows",
    "WORKFLOW_RUN": "workflow_runs",
    "WF_PERMISSION": "wf_permissions",
    "PJ_PERMISSION": "pj_permissions",
    "RESULT": "results",
    "ANNOTATION": "annotations",
}

# Creation order: a table is created after everything it references (self references aside).
TABLE_ORDER = [
    "PROJECT",
    "USER",
    "ROLE",
    "WORKFLOW",
    "WF_PERMISSION",
    "PJ_PERMISSION",
    "WORKFLOW_RUN",
    "RESULT",
    "ANNOTATION",
]

PROJECT_TABLE = """
CREATE TABLE IF NOT EXISTS {PROJECT} (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    description     TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    auth_key        TEXT,
    created         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    planned_enddate TIMESTAMPTZ
)
"""

USER_TABLE = """
CREATE TABLE IF NOT EXISTS {USER} (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    netid         TEXT UNIQUE,
    email_address TEXT,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    auth_key      TEXT,
    created       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

ROLE_TABLE = """
CREATE TABLE IF NOT EXISTS {ROLE} (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

# WORKFLOW }|--|| PROJECT (exactly one project) and WORKFLOW |o--o{ WORKFLOW (optional predecessor).
WORKFLOW_TABLE = """
CREATE TABLE IF NOT EXISTS {WORKFLOW} (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id  UUID NOT NULL REFERENCES {PROJECT}(id) ON DELETE CASCADE,
    previous_id UUID REFERENCES {WORKFLOW}(id) ON DELETE SET NULL,
    name        TEXT NOT NULL,
    description TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

# WORKFLOW ||--o{ WF_PERMISSION }o--|| ROLE / USER: all three sides are required.
WF_PERMISSION_TABLE = """
CREATE TABLE IF NOT EXISTS {WF_PERMISSION} (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES {USER}(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL REFERENCES {WORKFLOW}(id) ON DELETE CASCADE,
    role_id     UUID NOT NULL REFERENCES {ROLE}(id) ON DELETE RESTRICT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, workflow_id, role_id)
)
"""

PJ_PERMISSION_TABLE = """
CREATE TABLE IF NOT EXISTS {PJ_PERMISSION} (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES {USER}(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES {PROJECT}(id) ON DELETE CASCADE,
    role_id    UUID NOT NULL REFERENCES {ROLE}(id) ON DELETE RESTRICT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, project_id, role_id)
)
"""

# WORKFLOW_RUN }|--|| WORKFLOW: one execution of a workflow. duration/last_run live here rather
# than on WORKFLOW, so a workflow run repeatedly keeps a row per run instead of only its latest.
WORKFLOW_RUN_TABLE = """
CREATE TABLE IF NOT EXISTS {WORKFLOW_RUN} (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id    UUID NOT NULL REFERENCES {WORKFLOW}(id) ON DELETE CASCADE,
    was_successful BOOLEAN,
    metadata       JSONB,
    extra_data     JSONB,
    duration       DOUBLE PRECISION,
    last_run       TIMESTAMPTZ,
    created        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

# RESULT }|--|| WORKFLOW_RUN (exactly one), RESULT |o--o{ RESULT (optional predecessor),
# RESULT }o..o| USER (optional editor). process_id is an FK in the diagram with no entity behind it.
#
# The metadata an AI call records (token_usage, duration, type, engine, model) is stored whole in
# the ``metadata`` column; ``md_duration``/``md_cost``/``md_timestamp`` are promoted out of it for
# querying without reaching into the JSON. See ``chai_tea_import.py``.
RESULT_TABLE = """
CREATE TABLE IF NOT EXISTS {RESULT} (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_run_id UUID NOT NULL REFERENCES {WORKFLOW_RUN}(id) ON DELETE CASCADE,
    process_id      TEXT,
    input           TEXT,
    input_hash      TEXT,
    input_segment   TEXT,
    input_sequence  INTEGER,
    was_successful  BOOLEAN,
    value           JSONB,
    metadata        JSONB,
    extra_data      JSONB,
    previous_id     UUID REFERENCES {RESULT}(id) ON DELETE SET NULL,
    editor_user_id  UUID REFERENCES {USER}(id) ON DELETE SET NULL,
    md_cost         DOUBLE PRECISION,
    md_duration     DOUBLE PRECISION,
    md_timestamp    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

# RESULT |o..o{ ANNOTATION (a result has many, each annotation targets at most one result) and
# ANNOTATION |o..o| ANNOTATION (one-to-one, hence the UNIQUE on target_annotation_id).
ANNOTATION_TABLE = """
CREATE TABLE IF NOT EXISTS {ANNOTATION} (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_result_id     UUID REFERENCES {RESULT}(id) ON DELETE CASCADE,
    target_annotation_id UUID UNIQUE REFERENCES {ANNOTATION}(id) ON DELETE CASCADE,
    user_id              UUID REFERENCES {USER}(id) ON DELETE SET NULL,
    flag                 TEXT,
    comment              TEXT,
    metadata             JSONB,
    extra_data           JSONB,
    md_timestamp         TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_TABLES = {
    "PROJECT": PROJECT_TABLE,
    "USER": USER_TABLE,
    "ROLE": ROLE_TABLE,
    "WORKFLOW": WORKFLOW_TABLE,
    "WORKFLOW_RUN": WORKFLOW_RUN_TABLE,
    "WF_PERMISSION": WF_PERMISSION_TABLE,
    "PJ_PERMISSION": PJ_PERMISSION_TABLE,
    "RESULT": RESULT_TABLE,
    "ANNOTATION": ANNOTATION_TABLE,
}

# One index per foreign key column, so the joins the diagram implies do not sequentially scan.
# (entity, column) -- the unique constraints above already cover annotations.target_annotation_id
# and the permission tables' leading columns.
INDEXED_COLUMNS = [
    ("WORKFLOW", "project_id"),
    ("WORKFLOW", "previous_id"),
    ("WF_PERMISSION", "workflow_id"),
    ("WF_PERMISSION", "role_id"),
    ("PJ_PERMISSION", "project_id"),
    ("PJ_PERMISSION", "role_id"),
    ("WORKFLOW_RUN", "workflow_id"),
    ("RESULT", "workflow_run_id"),
    ("RESULT", "previous_id"),
    ("RESULT", "editor_user_id"),
    ("RESULT", "process_id"),
    ("RESULT", "input_hash"),
    ("ANNOTATION", "target_result_id"),
    ("ANNOTATION", "user_id"),
]


# Every table except ANNOTATION has a ``modified`` column the diagram expects to track updates.
# clock_timestamp(), not CURRENT_TIMESTAMP: the latter is the transaction's start time, so a row
# inserted and then updated inside one transaction would come out with modified equal to created.
MODIFIED_TRIGGER_FUNCTION = """
CREATE OR REPLACE FUNCTION chai_tea_set_modified() RETURNS trigger AS $$
BEGIN
    NEW.modified := clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""

MODIFIED_TABLES = [
    "PROJECT",
    "USER",
    "ROLE",
    "WORKFLOW",
    "WF_PERMISSION",
    "PJ_PERMISSION",
    "WORKFLOW_RUN",
    "RESULT",
]


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


def table_names(tables=None):
    """The entity -> table name mapping, with *tables* overriding the defaults; names are checked."""
    names = dict(TABLES)
    names.update(tables or {})
    missing = set(TABLES) - set(names)
    if missing:
        raise ValueError(f"No table name for {', '.join(sorted(missing))}")
    return {entity: _pg_identifier(name) for entity, name in names.items()}


def schema_statements(tables=None, drop_existing=False):
    """Every statement needed to build the schema, in the order it has to run.

    With *drop_existing* the ``DROP TABLE`` statements come first, in reverse dependency order and
    with ``CASCADE`` so the self references and views go too.
    """
    names = table_names(tables)
    statements = []
    if drop_existing:
        for entity in reversed(TABLE_ORDER):
            statements.append(f"DROP TABLE IF EXISTS {names[entity]} CASCADE")
    # gen_random_uuid() is built in from PostgreSQL 13; pgcrypto supplies it on older servers.
    statements.append("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    for entity in TABLE_ORDER:
        statements.append(CREATE_TABLES[entity].format(**names).strip())
    for entity, column in INDEXED_COLUMNS:
        table = names[entity]
        statements.append(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})")
    statements.append(MODIFIED_TRIGGER_FUNCTION.strip())
    for entity in MODIFIED_TABLES:
        table = names[entity]
        statements.append(f"DROP TRIGGER IF EXISTS trg_{table}_modified ON {table}")
        statements.append(
            f"CREATE TRIGGER trg_{table}_modified BEFORE UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION chai_tea_set_modified()"
        )
    return statements


def _check_for_clashes(conn, names):
    """Raise if a table of one of our names exists already with columns that are not ours.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so an unrelated ``results``
    table (the one ``PostgresStorage`` writes, say) would otherwise be quietly accepted and then
    break on the first insert.
    """
    expected = {entity: set(column_types(entity)) for entity in TABLE_ORDER}
    cursor = conn.cursor()
    cursor.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
        (list(names.values()),),
    )
    found = {}
    for table, column in cursor.fetchall():
        found.setdefault(table, set()).add(column)
    for entity, table in names.items():
        columns = found.get(table)
        if columns and not expected[entity] <= columns:
            absent = sorted(expected[entity] - columns)
            missing = ", ".join(absent[:3]) + (f" and {len(absent) - 3} more" if len(absent) > 3 else "")
            raise RuntimeError(
                f"Table {table} already exists and is not the CHAI-TEA {entity} table "
                f"(no {missing} column). Use another database, rename it in TABLES, or --drop."
            )


# The words that end a column's type and begin its constraints, in the templates above.
_CONSTRAINT_WORDS = ("PRIMARY", "NOT", "NULL", "DEFAULT", "REFERENCES", "UNIQUE", "CHECK", "GENERATED")


def column_types(entity):
    """``{column: SQL type}`` for one diagram entity, in order, read off its CREATE TABLE template.

    The single description of each table's columns, so anything that has to agree with the schema
    (the Parquet export, the clash check) reads it from here instead of repeating the list.
    """
    ddl = CREATE_TABLES[entity]
    body = ddl[ddl.index("(") + 1 : ddl.rindex(")")]
    types = {}
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("UNIQUE", "PRIMARY KEY", "FOREIGN KEY", "CHECK", "CONSTRAINT")):
            continue
        name, _, rest = line.partition(" ")
        words = []
        for word in rest.split():
            if word.upper() in _CONSTRAINT_WORDS:
                break
            words.append(word)
        types[name] = " ".join(words)
    return types


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
