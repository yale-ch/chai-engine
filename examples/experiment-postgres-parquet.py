"""Run one workflow into PostgreSQL two ways -- directly, and via Parquet -- and diff the tables.

The workflow reads a CSV of people, pulls the name out of each row, and stores every result twice,
both times in the CHAI-TEA schema:

* ``PostgresStorage`` writes each result into the CHAI-TEA tables as it is produced -- the run's
  PROJECT, WORKFLOW and WORKFLOW_RUN rows first, then one RESULT row per result -- the way a run
  that talks to the database as it goes does it.
* ``ParquetStorage`` collects the whole run into one Parquet file of RESULT rows, written when the
  workflow ends, with the run's PROJECT/WORKFLOW/WORKFLOW_RUN rows in the file's metadata.
  ``store_postgres_rows`` creates those, then ``parquet_to_postgres`` bulk loads the RESULT rows --
  the way a run that cannot reach the database (or should not hold a connection open for hours)
  hands its results over, e.g. by uploading the file for a central loader.

Neither storage is configured beyond where to write: RESULT rows are what both produce by default,
with every value taken from the objects -- the workflow's name, description and project, the run's
timing and outcome, each result's own timestamps. The two routes write two sets of tables (suffixed
``_direct`` and ``_parquet``) in one database, and the script diffs their RESULT tables, every
column, with EXCEPT in both directions. Exits non-zero if they are not identical.

Usage (from the repository root, with a PostgreSQL server on localhost:5432):

    python examples/experiment-postgres-parquet.py
    python examples/experiment-postgres-parquet.py --limit 500 --database chai_demo
    python examples/experiment-postgres-parquet.py --dsn postgresql://user@host:5432/chai_demo
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

import pyarrow.parquet as pq  # noqa: E402

from chai.persistence.schema import TABLE_ORDER, TABLES  # noqa: E402
from chai.persistence import (  # noqa: E402
    RESULT_COLUMNS,
    ensure_postgres_database,
    parquet_to_postgres,
    postgres_params,
    store_postgres_rows,
)
from chai.persistence.postgres import _pg_connect  # noqa: E402
from chai.workflow import Workflow  # noqa: E402


def suffixed(suffix):
    """A set of CHAI-TEA table names of their own: ``results_direct``, ``workflows_direct``, ..."""
    return {entity: f"{name}_{suffix}" for entity, name in TABLES.items()}


def build_workflow(csv_path, limit, parquet_file, direct_tables, connection):
    """The workflow config: read rows, extract each name, store the result both ways."""
    return {
        "id": "pg_parquet_wf",
        "type": "workflow.Workflow",
        "name": "People: names, stored two ways",
        "description": "Extract each person's name from the CSV; store it in PostgreSQL and in Parquet",
        "project": {"name": "chai demo", "description": "examples/experiment-postgres-parquet.py"},
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
                        # Each row this iterator hands out is a recorded source: every result made
                        # from it, however many steps later, records that row as its input
                        "source": True,
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
                                        "settings": dict(connection, tables=direct_tables),
                                    },
                                    {
                                        "id": "to_parquet",
                                        "type": "storage.ParquetStorage",
                                        "settings": {"file": parquet_file},
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def reset_tables(params, *table_sets):
    """Drop the demo's tables so a re-run compares this run's rows and no others."""
    conn = _pg_connect(params, autocommit=True)
    try:
        for tables in table_sets:
            for entity in reversed(TABLE_ORDER):
                conn.cursor().execute(f"DROP TABLE IF EXISTS {tables[entity]} CASCADE")
    finally:
        conn.close()


def compare_tables(params, left, right):
    """Row counts for both tables and the rows each has that the other does not, every column."""
    columns = ", ".join(RESULT_COLUMNS)
    conn = _pg_connect(params)
    try:
        cursor = conn.cursor()
        counts = {}
        for table in (left, right):
            cursor.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        diffs = {}
        for a, b in ((left, right), (right, left)):
            cursor.execute(f"SELECT {columns} FROM {a} EXCEPT SELECT {columns} FROM {b} LIMIT 5")
            diffs[f"{a} - {b}"] = cursor.fetchall()
        return counts, diffs
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=os.path.join(ROOT, "inputs", "people.csv"))
    parser.add_argument("--limit", type=int, default=100, help="rows to read (0 for all of them)")
    parser.add_argument("--parquet", default=os.path.join(ROOT, "results", "people-run.parquet"))
    parser.add_argument("--dsn", default=None, help="connection string; overrides the options below")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--database", default="chai_demo", help="created if missing (default chai_demo)")
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
    direct, via_parquet = suffixed("direct"), suffixed("parquet")
    ensure_postgres_database(params)
    if not args.keep:
        reset_tables(params, direct, via_parquet)

    workflow = Workflow(build_workflow(args.csv, args.limit, args.parquet, direct, connection))
    workflow.run()  # the Parquet file is written when the run ends
    storage = workflow.get_component_by_id("to_parquet")
    print(f"Wrote run {workflow.run_id} to {direct['RESULT']} and to {storage.file_name}")

    # The loader's side: the rows the RESULT rows hang under (carried in the file), then the rows
    records = json.loads(pq.read_schema(storage.file_name).metadata[b"chai_tea_records"])
    store_postgres_rows(records, params, tables=via_parquet)
    loaded = parquet_to_postgres(storage.file_name, via_parquet["RESULT"], params, create=False)
    print(f"Loaded {loaded['rows']} rows from the Parquet file into {loaded['table']}")

    counts, diffs = compare_tables(params, direct["RESULT"], via_parquet["RESULT"])
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
