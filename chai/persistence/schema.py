"""The CHAI-TEA database schema: the tables, their columns and their DDL, for everything in chai.

The single definition of the shared PostgreSQL schema drawn in ``mermaid-er-database.md`` --
PROJECT -> WORKFLOW -> WORKFLOW_RUN -> RESULT, with ANNOTATION, USER, ROLE and the two permission
tables. Everything that has to agree with it reads it from here: the storages that write it
(``SqliteStorage``, ``PostgresStorage``, and the RESULT rows of ``JsonLinesStorage``/
``ParquetStorage``), the objects that carry its values (``chai.persistence.records``, ``Workflow``, ``Result``),
and the top-level ``chai_tea_*`` tools, which re-export it. ``column_types(entity)`` describes each
table's columns, ``column_kinds(entity)`` the same as Parquet column kinds, and
``schema_statements``/``sqlite_statements`` build the tables in PostgreSQL and SQLite.

Choices the diagram does not make are recorded at the top of ``chai_tea_schema.py`` and in
``chai_tea_schema.sql``, which must stay in step with the DDL here.
"""

import re

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def identifier(name):
    """Validate a table or column name coming from configuration before it is put into SQL."""
    name = str(name)
    if not _IDENTIFIER.match(name):
        raise ValueError(f"{name!r} is not a usable PostgreSQL identifier")
    return name


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
# querying without reaching into the JSON; ``md_cost`` is the ``cost`` the component recorded for its
# call (LiteLLM's figure, or the component's ``pricing`` setting).
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


def table_names(tables=None):
    """The entity -> table name mapping, with *tables* overriding the defaults; names are checked."""
    names = dict(TABLES)
    names.update(tables or {})
    missing = set(TABLES) - set(names)
    if missing:
        raise ValueError(f"No table name for {', '.join(sorted(missing))}")
    return {entity: identifier(name) for entity, name in names.items()}


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


def check_for_clashes(conn, names):
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


# The CHAI-TEA schema's SQL types as ParquetRecordWriter column kinds. Declaring them (rather than
# letting the writer infer from the data) means every file of a table has the same columns and
# types, including one where the table came out empty.
KINDS = {
    "UUID": "string",
    "TEXT": "string",
    "BOOLEAN": "bool",
    "INTEGER": "int",
    "DOUBLE PRECISION": "float",
    "JSONB": "json",
    "TIMESTAMPTZ": "timestamp",
}


def column_kinds(entity):
    """``{column: Parquet column kind}`` for one entity of the CHAI-TEA schema."""
    types = column_types(entity)
    unknown = {t for t in types.values() if t not in KINDS}
    if unknown:
        raise ValueError(f"No Parquet column kind for SQL type(s) {', '.join(sorted(unknown))}")
    return {column: KINDS[sql_type] for column, sql_type in types.items()}


# PostgreSQL types as SQLite storage classes. SQLite keeps uuids, JSON and timestamps as text
# (timestamps ISO-8601, so they sort), booleans as 0/1.
_SQLITE_TYPES = {
    "UUID": "TEXT",
    "TEXT": "TEXT",
    "BOOLEAN": "INTEGER",
    "INTEGER": "INTEGER",
    "DOUBLE PRECISION": "REAL",
    "JSONB": "TEXT",
    "TIMESTAMPTZ": "TEXT",
}


def sqlite_statements(tables=None):
    """The same tables, keys and indexes as ``schema_statements``, in SQLite's dialect.

    Translated from the PostgreSQL DDL rather than written out again, so the two cannot drift: the
    column types become SQLite's, the server-side defaults (``gen_random_uuid()``, the timestamps)
    are dropped -- the storages always write those columns -- and the ``modified`` trigger is left
    out, because the storages set ``modified`` themselves. Foreign keys are kept; SQLite enforces
    them when ``PRAGMA foreign_keys = ON``, which the storages turn on.
    """
    names = table_names(tables)
    statements = []
    for entity in TABLE_ORDER:
        ddl = CREATE_TABLES[entity].format(**names).strip()
        for pg_type in sorted(_SQLITE_TYPES, key=len, reverse=True):
            ddl = re.sub(rf"(?<=\s){pg_type}(?=[\s,]|$)", _SQLITE_TYPES[pg_type], ddl, flags=re.M)
        ddl = re.sub(r"\s+DEFAULT\s+(gen_random_uuid\(\)|CURRENT_TIMESTAMP|TRUE|FALSE)", "", ddl)
        statements.append(ddl)
    for entity, column in INDEXED_COLUMNS:
        table = names[entity]
        statements.append(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})")
    return statements
