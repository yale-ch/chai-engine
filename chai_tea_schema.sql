-- CHAI-TEA database schema for Amazon RDS for PostgreSQL.
--
-- Generated from the ER diagram in mermaid-er-database.md: one table per entity
-- (PROJECT, USER, ROLE, WORKFLOW, WORKFLOW_RUN, RESULT, ANNOTATION and the two
-- permission tables), with the columns, types and keys the diagram declares.
--
--   psql "host=<endpoint>.rds.amazonaws.com port=5432 dbname=chai_tea user=<master> sslmode=require" \
--        -v ON_ERROR_STOP=1 -f chai_tea_schema.sql
--
-- The database itself is not created here: create it first (CREATE DATABASE chai_tea;)
-- or use the one the RDS instance was launched with, then run this against it.
--
-- Choices the diagram does not make:
--
--   * USER becomes the table "users" -- user is a reserved word in PostgreSQL and
--     would otherwise need quoting in every query.
--   * Type mapping: uuid -> UUID (default gen_random_uuid()), string -> TEXT,
--     int -> INTEGER, float -> DOUBLE PRECISION, JSON -> JSONB,
--     datetime -> TIMESTAMPTZ.
--   * ON DELETE actions are not expressible in mermaid. Identifying relationships
--     (solid -- lines) cascade, optional links (dashed .. lines and the nullable
--     self references) null out, and a ROLE cannot be deleted while permissions
--     still point at it.
--   * ANNOTATION |o..o| ANNOTATION is one-to-one, so target_annotation_id is
--     UNIQUE: an annotation can be annotated at most once. Drop that constraint if
--     annotations should accept several replies.
--   * RESULT.process_id is marked FK in the diagram but has no entity behind it, so
--     it stays a plain (indexed) column. ANNOTATION.user_id, likewise lineless, is
--     wired to users -- the only entity it can mean.
--   * created defaults to now(); modified is maintained by the trigger below.
--   * Every foreign key column is indexed, and each permission row is unique on
--     (user, workflow/project, role).
--
-- RESULT becomes a table called "results", which is also the default table
-- PostgresStorage writes to -- a different shape entirely. Build this schema in a
-- database of its own.

BEGIN;

-- gen_random_uuid() is built in from PostgreSQL 13; pgcrypto supplies it on older
-- RDS engine versions. Requires the rds_superuser role (the RDS master user has it).
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- ---------------------------------------------------------------------------
-- modified timestamps
-- ---------------------------------------------------------------------------

-- clock_timestamp(), not CURRENT_TIMESTAMP: the latter is the transaction's start
-- time, so a row inserted and then updated inside one transaction would come out
-- with modified exactly equal to created. clock_timestamp() is real wall time, so
-- modified always advances.
CREATE OR REPLACE FUNCTION chai_tea_set_modified() RETURNS trigger AS $$
BEGIN
    NEW.modified := clock_timestamp();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ---------------------------------------------------------------------------
-- PROJECT, USER, ROLE -- referenced by everything else, so created first
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS projects (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             TEXT NOT NULL,
    description      TEXT,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    auth_key         TEXT,
    created          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    planned_enddate  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    netid         TEXT UNIQUE,
    email_address TEXT,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    auth_key      TEXT,
    created       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS roles (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- WORKFLOW }|--|| PROJECT, WORKFLOW |o--o{ WORKFLOW (optional predecessor)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS workflows (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id  UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    previous_id UUID REFERENCES workflows(id) ON DELETE SET NULL,
    name        TEXT NOT NULL,
    description TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- Permissions: WORKFLOW/PROJECT ||--o{ PERMISSION }o--|| ROLE / USER.
-- All three sides are required on each row.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS wf_permissions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    role_id     UUID NOT NULL REFERENCES roles(id) ON DELETE RESTRICT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, workflow_id, role_id)
);

CREATE TABLE IF NOT EXISTS pj_permissions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    role_id    UUID NOT NULL REFERENCES roles(id) ON DELETE RESTRICT,
    is_active  BOOLEAN NOT NULL DEFAULT TRUE,
    created    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, project_id, role_id)
);


-- ---------------------------------------------------------------------------
-- WORKFLOW_RUN }|--|| WORKFLOW: one execution of a workflow
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS workflow_runs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id    UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    was_successful BOOLEAN,
    metadata       JSONB,
    extra_data     JSONB,
    duration       DOUBLE PRECISION,
    last_run       TIMESTAMPTZ,
    created        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- RESULT }|--|| WORKFLOW_RUN, RESULT |o--o{ RESULT (optional predecessor),
-- RESULT }o..o| USER (optional editor)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_run_id UUID NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    process_id      TEXT,
    input           TEXT,
    input_hash      TEXT,
    input_segment   TEXT,
    input_sequence  INTEGER,
    was_successful  BOOLEAN,
    value           JSONB,
    metadata        JSONB,
    extra_data      JSONB,
    previous_id     UUID REFERENCES results(id) ON DELETE SET NULL,
    editor_user_id  UUID REFERENCES users(id) ON DELETE SET NULL,
    md_cost         DOUBLE PRECISION,
    md_duration     DOUBLE PRECISION,
    md_timestamp    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    modified        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- RESULT |o..o{ ANNOTATION and ANNOTATION |o..o| ANNOTATION (one-to-one, hence
-- the UNIQUE on target_annotation_id)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS annotations (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_result_id     UUID REFERENCES results(id) ON DELETE CASCADE,
    target_annotation_id UUID UNIQUE REFERENCES annotations(id) ON DELETE CASCADE,
    user_id              UUID REFERENCES users(id) ON DELETE SET NULL,
    flag                 TEXT,
    comment              TEXT,
    metadata             JSONB,
    extra_data           JSONB,
    md_timestamp         TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);


-- ---------------------------------------------------------------------------
-- Indexes on the foreign key columns, so the joins the diagram implies do not
-- sequentially scan. The UNIQUE constraints above already cover
-- annotations.target_annotation_id and the permission tables' leading columns.
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_workflows_project_id        ON workflows(project_id);
CREATE INDEX IF NOT EXISTS idx_workflows_previous_id       ON workflows(previous_id);
CREATE INDEX IF NOT EXISTS idx_wf_permissions_workflow_id  ON wf_permissions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wf_permissions_role_id      ON wf_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_pj_permissions_project_id   ON pj_permissions(project_id);
CREATE INDEX IF NOT EXISTS idx_pj_permissions_role_id      ON pj_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_workflow_id   ON workflow_runs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_results_workflow_run_id     ON results(workflow_run_id);
CREATE INDEX IF NOT EXISTS idx_results_previous_id         ON results(previous_id);
CREATE INDEX IF NOT EXISTS idx_results_editor_user_id      ON results(editor_user_id);
CREATE INDEX IF NOT EXISTS idx_results_process_id          ON results(process_id);
CREATE INDEX IF NOT EXISTS idx_results_input_hash          ON results(input_hash);
CREATE INDEX IF NOT EXISTS idx_annotations_target_result_id ON annotations(target_result_id);
CREATE INDEX IF NOT EXISTS idx_annotations_user_id         ON annotations(user_id);


-- ---------------------------------------------------------------------------
-- Keep modified current on every UPDATE. annotations has no modified column in
-- the diagram, so it gets no trigger.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'projects', 'users', 'roles', 'workflows',
        'wf_permissions', 'pj_permissions', 'workflow_runs', 'results'
    ] LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_%1$s_modified ON %1$I', t);
        EXECUTE format(
            'CREATE TRIGGER trg_%1$s_modified BEFORE UPDATE ON %1$I '
            'FOR EACH ROW EXECUTE FUNCTION chai_tea_set_modified()', t);
    END LOOP;
END;
$$;

COMMIT;
