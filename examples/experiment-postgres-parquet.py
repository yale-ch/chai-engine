"""Run one workflow into PostgreSQL two ways -- directly, and via Parquet -- and diff the tables.

The workflow reads a CSV of people, pulls the name out of each row, and stores every result twice:

* ``PostgresStorage`` writes each result into the ``people_direct`` table as it is produced -- one
  INSERT per result, the way a run that talks to the database as it goes does it.
* ``ParquetStorage`` collects the whole run into one Parquet file, written when the workflow ends.
  ``parquet_to_postgres`` then bulk loads that file into ``people_parquet`` -- the way a run that
  cannot reach the database (or should not hold a connection open for hours) hands its results over.

The Parquet steps are configured to produce exactly the columns ``PostgresStorage`` writes:
``@record`` puts the whole result JSON in ``value_json``, ``@input_uri``/``@input_hash`` record the
CSV each row came from and the md5 of the row itself, ``@corrects`` carries the pointer a correction
entry has back to the entry it corrects, and ``null_if_empty`` matches the storage convention of
writing an absent ``metadata``/``extraInfo`` as NULL rather than as ``{}``. The target table is
created by ``ensure_postgres_schema``, so both tables have the same DDL.

The script then diffs the two tables with EXCEPT in both directions and reports whether the rows the
two routes produced are identical. Exits non-zero if they are not.

Usage (from the repository root, with a PostgreSQL server on localhost:5432):

    python examples/experiment-postgres-parquet.py
    python examples/experiment-postgres-parquet.py --limit 500 --database chai
    python examples/experiment-postgres-parquet.py --dsn postgresql://user@host:5432/chai
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

from chai.storage import (  # noqa: E402
    _pg_connect,
    ensure_postgres_schema,
    parquet_to_postgres,
    postgres_params,
)
from chai.workflow import Workflow  # noqa: E402

# The columns both routes fill in: PostgresStorage's own, minus created_at, which the database sets.
SHARED_COLUMNS = (
    "id, processor_id, workflow_id, value_json, metadata_json, extra_json, "
    "input_uri, input_hash, corrects_id"
)

# The Parquet row, shaped to match those columns: the keys are the result JSON's, the values the
# column names, and the @ keys are computed -- @record is the whole result JSON (what PostgresStorage
# puts in value_json), @input_uri/@input_hash say what the result was generated from, and @corrects
# is the row this one corrects, if it is a correction of one.
PARQUET_FIELDS = {
    "id": "id",
    "processorId": "processor_id",
    "workflowId": "workflow_id",
    "@record": "value_json",
    "metadata": "metadata_json",
    "extraInfo": "extra_json",
    "@input_uri": "input_uri",
    "@input_hash": "input_hash",
    "@corrects": "corrects_id",
}
PARQUET_SCHEMA = {
    "id": "string",
    "processor_id": "string",
    "workflow_id": "string",
    "value_json": "json",
    "metadata_json": "json",
    "extra_json": "json",
    "input_uri": "string",
    "input_hash": "string",
    "corrects_id": "string",
}


def build_workflow(csv_path, limit, parquet_file, direct_table, connection):
    """The workflow config: read rows, extract each name, store the result both ways."""
    return {
        "id": "pg_parquet_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "id": "people_csv",
                "type": "provider.CsvFileProvider",
                "input": csv_path,
                "settings": {"limit": limit},
                "steps": [
                    {
                        "id": "people_rows",
                        "type": "iterator.Iterator",
                        # The results are persisted as they are produced, so nothing needs keeping
                        "settings": {"retain_results": False, "workers": 4},
                        "steps": [
                            {
                                "id": "person_name",
                                "type": "extractor.JsonXpathExtractor",
                                "settings": {"xpath": "/name"},
                                "next_steps": [
                                    {
                                        "id": "to_postgres",
                                        "type": "storage.PostgresStorage",
                                        "settings": dict(connection, table=direct_table),
                                    },
                                    {
                                        "id": "to_parquet",
                                        "type": "storage.ParquetStorage",
                                        "settings": {
                                            "file": parquet_file,
                                            "fields": PARQUET_FIELDS,
                                            "schema": PARQUET_SCHEMA,
                                            "null_if_empty": True,
                                            "run_columns": False,
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def reset_tables(params, tables):
    """Drop the demo's tables so a re-run compares this run's rows and no others."""
    conn = _pg_connect(params, autocommit=True)
    try:
        for table in tables:
            conn.cursor().execute(f"DROP TABLE IF EXISTS {table}")
            conn.cursor().execute(f"DROP TABLE IF EXISTS {table}_derivatives")
    finally:
        conn.close()


def compare_tables(params, left, right):
    """Row counts for both tables and the rows each has that the other does not."""
    conn = _pg_connect(params)
    try:
        cursor = conn.cursor()
        counts = {}
        for table in (left, right):
            cursor.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        diffs = {}
        for a, b in ((left, right), (right, left)):
            cursor.execute(
                f"SELECT {SHARED_COLUMNS} FROM {a} EXCEPT SELECT {SHARED_COLUMNS} FROM {b} LIMIT 5"
            )
            diffs[f"{a} - {b}"] = cursor.fetchall()
        return counts, diffs
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=os.path.join(ROOT, "inputs", "people.csv"))
    parser.add_argument("--limit", type=int, default=100, help="rows to read (0 for all of them)")
    parser.add_argument("--parquet", default=os.path.join(ROOT, "results", "people-run.parquet"))
    parser.add_argument("--direct-table", default="people_direct")
    parser.add_argument("--parquet-table", default="people_parquet")
    parser.add_argument("--dsn", default=None, help="connection string; overrides the options below")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--database", default="chai")
    parser.add_argument("--user", default=None)
    parser.add_argument("--keep", action="store_true", help="add to the tables instead of dropping them")
    args = parser.parse_args()

    connection = {
        k: v
        for k, v in (
            ("dsn", args.dsn),
            ("host", args.host),
            ("port", args.port),
            ("database", args.database),
            ("user", args.user),
        )
        if v
    }
    params = postgres_params(connection)
    if not args.keep:
        reset_tables(params, (args.direct_table, args.parquet_table))

    workflow = Workflow(build_workflow(args.csv, args.limit, args.parquet, args.direct_table, connection))
    workflow.run()  # the Parquet file is written when the run ends
    storage = workflow.get_component_by_id("to_parquet")
    print(f"Wrote the run to {args.direct_table} and to {storage.file_name}")

    # The Parquet table is created with the same DDL as the one PostgresStorage writes, so the two
    # can be compared column for column; the loader then fills in the columns the file carries.
    ensure_postgres_schema(params, table=args.parquet_table)
    loaded = parquet_to_postgres(storage.file_name, args.parquet_table, params)
    print(f"Loaded {loaded['rows']} rows from the Parquet file into {loaded['table']}")

    counts, diffs = compare_tables(params, args.direct_table, args.parquet_table)
    for table, count in counts.items():
        print(f"  {table}: {count} rows")
    identical = len(set(counts.values())) == 1 and not any(diffs.values())
    for label, rows in diffs.items():
        for row in rows:
            print(f"  only in {label}: {row}")
    print("The two tables are identical" if identical else "The two tables DIFFER")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
