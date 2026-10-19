"""The CHAI-TEA schema, and the Parquet route from a local PostgreSQL instance to a central one.

Needs a PostgreSQL server on localhost:5432 (the whole module skips without one) and pyarrow. Each
test builds its own pair of databases and drops them again, so the suite can run beside a real one.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chai_tea_parquet  # noqa: E402
import chai_tea_schema  # noqa: E402
from chai.storage import _pg_connect, ensure_postgres_database, postgres_params  # noqa: E402


def postgres_available():
    try:
        ensure_postgres_database(postgres_params({}))
        _pg_connect(postgres_params({})).close()
        return True
    except Exception:
        return False


def pyarrow_available():
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


HAS_POSTGRES = postgres_available()
HAS_PYARROW = pyarrow_available()


class TestSchemaColumns(unittest.TestCase):
    """The column list every other piece reads off the DDL templates."""

    def test_types_come_off_the_ddl_without_the_constraints(self):
        types = chai_tea_schema.column_types("RESULT")
        self.assertEqual(types["id"], "UUID")  # not 'UUID PRIMARY KEY DEFAULT gen_random_uuid()'
        self.assertEqual(types["md_cost"], "DOUBLE PRECISION")  # a two-word type survives
        self.assertEqual(types["value"], "JSONB")
        self.assertEqual(types["input_sequence"], "INTEGER")
        self.assertEqual(types["md_timestamp"], "TIMESTAMPTZ")
        self.assertNotIn("UNIQUE", types)  # a table-level constraint is not a column

    def test_every_entity_maps_to_parquet_kinds(self):
        for entity in chai_tea_schema.TABLE_ORDER:
            kinds = chai_tea_parquet.column_kinds(entity)
            self.assertEqual(list(kinds), list(chai_tea_schema.column_types(entity)))
            self.assertEqual(kinds["id"], "string")

    def test_export_order_puts_parents_first(self):
        for order in (chai_tea_parquet.export_order(), chai_tea_parquet.export_order(True)):
            self.assertLess(order.index("PROJECT"), order.index("WORKFLOW"))
            self.assertLess(order.index("WORKFLOW"), order.index("WORKFLOW_RUN"))
            self.assertLess(order.index("WORKFLOW_RUN"), order.index("RESULT"))
            self.assertLess(order.index("RESULT"), order.index("ANNOTATION"))
            self.assertLess(order.index("USER"), order.index("RESULT"))
        self.assertNotIn("WF_PERMISSION", chai_tea_parquet.export_order())
        self.assertIn("WF_PERMISSION", chai_tea_parquet.export_order(True))


@unittest.skipUnless(HAS_POSTGRES, "no PostgreSQL server on localhost:5432")
@unittest.skipUnless(HAS_PYARROW, "pyarrow is not installed")
class TestChaiTeaParquet(unittest.TestCase):
    """A seeded local database, exported to Parquet and loaded into a central one."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.bundle = os.path.join(self.dir, "bundle")
        suffix = uuid.uuid4().hex[:8]
        self.local = f"chai_tea_local_{suffix}"
        self.central = f"chai_tea_central_{suffix}"
        chai_tea_schema.create_schema({"database": self.local})
        self.ids = self.seed()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        conn = _pg_connect(postgres_params({"database": "postgres"}), autocommit=True)
        try:
            for database in (self.local, self.central):
                conn.cursor().execute(f'DROP DATABASE IF EXISTS "{database}"')
        finally:
            conn.close()

    def connect(self, database):
        return _pg_connect(postgres_params({"database": database}))

    def query(self, database, sql, args=()):
        conn = self.connect(database)
        try:
            cursor = conn.cursor()
            cursor.execute(sql, args)
            return cursor.fetchall()
        finally:
            conn.close()

    def seed(self):
        """Two projects, a workflow superseding another, a chain of results, and annotations on them."""
        conn = self.connect(self.local)
        cursor = conn.cursor()

        def one(sql, args=()):
            cursor.execute(sql, args)
            return cursor.fetchone()[0]

        ids = {}
        ids["project"] = one("INSERT INTO projects (name) VALUES ('Ledgers') RETURNING id")
        ids["other_project"] = one("INSERT INTO projects (name) VALUES ('Photos') RETURNING id")
        ids["editor"] = one("INSERT INTO users (name, netid) VALUES ('Ada','ada1') RETURNING id")
        ids["reviewer"] = one("INSERT INTO users (name, netid) VALUES ('Bob','bob1') RETURNING id")
        # only ever referenced by a permission, so only an --include-permissions export carries them
        ids["admin"] = one("INSERT INTO users (name, netid) VALUES ('Carol','carol1') RETURNING id")
        ids["role"] = one("INSERT INTO roles (name) VALUES ('editor') RETURNING id")
        ids["wf1"] = one(
            "INSERT INTO workflows (project_id, name) VALUES (%s,'v1') RETURNING id", (ids["project"],)
        )
        ids["wf2"] = one(
            "INSERT INTO workflows (project_id, name, previous_id) VALUES (%s,'v2',%s) RETURNING id",
            (ids["project"], ids["wf1"]),
        )
        ids["wf3"] = one(
            "INSERT INTO workflows (project_id, name) VALUES (%s,'detect') RETURNING id",
            (ids["other_project"],),
        )
        cursor.execute(
            "INSERT INTO wf_permissions (user_id, workflow_id, role_id) VALUES (%s,%s,%s)",
            (ids["admin"], ids["wf1"], ids["role"]),
        )
        cursor.execute(
            "INSERT INTO pj_permissions (user_id, project_id, role_id) VALUES (%s,%s,%s)",
            (ids["admin"], ids["project"], ids["role"]),
        )
        for key in ("wf1", "wf2", "wf3"):
            ids[f"run_{key}"] = one(
                "INSERT INTO workflow_runs (workflow_id, was_successful) VALUES (%s, true) RETURNING id",
                (ids[key],),
            )
        ids["old"] = one(
            """INSERT INTO results (workflow_run_id, process_id, input, input_hash, input_segment,
                                    input_sequence, value, metadata, extra_data, md_cost, md_duration,
                                    md_timestamp)
               VALUES (%s,'transcriber','page1.png','abc','[{"bbox": [1, 2, 3, 4]}]',1,%s,%s,%s,
                       0.01,2.5,'2026-01-15T10:00:00Z') RETURNING id""",
            (
                ids["run_wf1"],
                json.dumps({"text": "olde ledger"}),
                json.dumps({"model": "gemini"}),
                json.dumps({"note": "x"}),
            ),
        )
        # the correction of it, in the workflow that superseded the one that made it
        ids["new"] = one(
            """INSERT INTO results (workflow_run_id, value, previous_id, editor_user_id, md_timestamp)
               VALUES (%s,%s,%s,%s,'2026-09-05T09:00:00Z') RETURNING id""",
            (ids["run_wf2"], json.dumps({"text": "old ledger"}), ids["old"], ids["editor"]),
        )
        ids["recent"] = one(
            """INSERT INTO results (workflow_run_id, value, previous_id, md_timestamp)
               VALUES (%s,%s,%s,'2026-09-06T09:00:00Z') RETURNING id""",
            (ids["run_wf2"], json.dumps({"text": "old ledger!"}), ids["new"]),
        )
        ids["elsewhere"] = one(
            "INSERT INTO results (workflow_run_id, value, md_timestamp) VALUES (%s,%s,'2026-09-07T09:00:00Z') RETURNING id",
            (ids["run_wf3"], json.dumps({"boxes": [[1, 2, 3, 4]]})),
        )
        ids["flag"] = one(
            """INSERT INTO annotations (target_result_id, user_id, flag, comment, metadata, md_timestamp)
               VALUES (%s,%s,'wrong','not olde',%s,'2026-09-05T10:00:00Z') RETURNING id""",
            (ids["old"], ids["editor"], json.dumps({"src": "review"})),
        )
        ids["reply"] = one(
            """INSERT INTO annotations (target_annotation_id, user_id, comment, md_timestamp)
               VALUES (%s,%s,'agreed','2026-09-05T11:00:00Z') RETURNING id""",
            (ids["flag"], ids["reviewer"]),
        )
        ids["elsewhere_flag"] = one(
            "INSERT INTO annotations (target_result_id, user_id, flag) VALUES (%s,%s,'ok') RETURNING id",
            (ids["elsewhere"], ids["reviewer"]),
        )
        conn.commit()
        conn.close()
        return ids

    def rows(self, database, entity):
        """One table's rows as comparable text, over the columns the export carries."""
        columns = ", ".join(chai_tea_schema.column_types(entity))
        table = chai_tea_schema.TABLES[entity]
        return [
            tuple(str(value) for value in row)
            for row in self.query(database, f"SELECT {columns} FROM {table} ORDER BY id")
        ]

    def assert_same_rows(self, entities=None):
        for entity in entities or chai_tea_parquet.EXPORT_ORDER:
            self.assertEqual(
                self.rows(self.local, entity), self.rows(self.central, entity), f"{entity} differs"
            )

    def export(self, directory=None, **kwargs):
        return chai_tea_parquet.export_parquet(directory or self.bundle, {"database": self.local}, **kwargs)

    def load(self, directory=None, **kwargs):
        return chai_tea_parquet.load_parquet(directory or self.bundle, {"database": self.central}, **kwargs)

    def test_a_full_export_reproduces_the_local_rows_centrally(self):
        manifest = self.export()
        self.assertEqual({t["file"] for t in manifest["tables"]} | {"manifest.json"}, set(os.listdir(self.bundle)))
        counts = {t["entity"]: t["rows"] for t in manifest["tables"]}
        self.assertEqual(counts, {"PROJECT": 2, "USER": 2, "WORKFLOW": 3, "WORKFLOW_RUN": 3, "RESULT": 4, "ANNOTATION": 3})

        loaded = self.load()
        self.assertEqual(sum(t["written"] for t in loaded.values()), 17)
        # the values, the jsonb, the timestamps and the links all come across unchanged
        self.assert_same_rows(["PROJECT", "WORKFLOW", "RESULT", "ANNOTATION"])
        self.assertEqual(
            self.query(self.central, "SELECT value->>'text' FROM results WHERE id = %s", (self.ids["old"],)),
            [("olde ledger",)],
        )
        self.assertEqual(
            self.query(self.central, "SELECT input_segment FROM results WHERE id = %s", (self.ids["old"],)),
            [('[{"bbox": [1, 2, 3, 4]}]',)],
        )

    def test_permissions_and_their_users_only_come_when_asked_for(self):
        self.export()
        self.load()
        # Carol has no results and no annotations, so a plain export has no reason to carry her
        self.assertEqual(self.query(self.central, "SELECT count(*) FROM users"), [(2,)])
        self.assertEqual(self.query(self.central, "SELECT count(*) FROM wf_permissions"), [(0,)])

        self.export(include_permissions=True)
        self.load()
        self.assertEqual(self.query(self.central, "SELECT count(*) FROM users"), [(3,)])
        self.assert_same_rows(chai_tea_parquet.export_order(include_permissions=True))

    def test_loading_twice_changes_nothing_the_second_time(self):
        self.export()
        first = self.load()
        second = self.load()
        self.assertEqual(sum(t["written"] for t in second.values()), 0)
        self.assertEqual(
            [t["skipped"] for t in second.values()], [t["written"] for t in first.values()]
        )
        self.assert_same_rows(["PROJECT", "WORKFLOW", "RESULT", "ANNOTATION"])

    def test_on_conflict_decides_whether_a_local_edit_overwrites_the_central_row(self):
        self.export()
        self.load()
        conn = self.connect(self.local)
        conn.cursor().execute(
            "UPDATE results SET value = %s WHERE id = %s",
            (json.dumps({"text": "corrected"}), self.ids["recent"]),
        )
        conn.commit()
        conn.close()
        self.export()

        self.load(on_conflict="skip")
        self.assertEqual(
            self.query(self.central, "SELECT value->>'text' FROM results WHERE id = %s", (self.ids["recent"],)),
            [("old ledger!",)],
        )
        self.load(on_conflict="update")
        self.assertEqual(
            self.query(self.central, "SELECT value->>'text' FROM results WHERE id = %s", (self.ids["recent"],)),
            [("corrected",)],
        )
        self.assert_same_rows(["RESULT"])

    def test_since_carries_the_ancestors_of_the_results_it_selects(self):
        # only 'elsewhere' was made this late, but its whole previous_id chain has to travel with it
        manifest = self.export(since="2026-09-06")
        counts = {t["entity"]: t["rows"] for t in manifest["tables"]}
        self.assertEqual(counts["RESULT"], 4)
        self.load()  # into an empty central database: nothing may dangle
        self.assertEqual(
            self.query(self.central, "SELECT count(*) FROM results WHERE previous_id IS NOT NULL"), [(2,)]
        )

    def test_a_project_filter_leaves_the_other_project_behind(self):
        manifest = self.export(project="Photos")
        counts = {t["entity"]: t["rows"] for t in manifest["tables"]}
        self.assertEqual(counts, {"PROJECT": 1, "USER": 1, "WORKFLOW": 1, "WORKFLOW_RUN": 1, "RESULT": 1, "ANNOTATION": 1})
        self.load()
        self.assertEqual(self.query(self.central, "SELECT name FROM projects"), [("Photos",)])

    def test_a_workflow_filter_carries_the_workflow_it_superseded(self):
        manifest = self.export(workflow="v2")
        counts = {t["entity"]: t["rows"] for t in manifest["tables"]}
        self.assertEqual(counts["WORKFLOW"], 2)  # v2 and the v1 its previous_id points at
        self.assertEqual(counts["RESULT"], 3)  # its two results and the one they correct
        self.assertEqual(counts["PROJECT"], 1)  # not the project the filter passed over
        self.load()
        self.assertEqual(
            sorted(name for (name,) in self.query(self.central, "SELECT name FROM workflows")),
            ["v1", "v2"],  # 'detect', in the other project, stayed at home
        )
        self.assertEqual(
            self.query(self.central, "SELECT count(*) FROM workflows WHERE previous_id IS NOT NULL"),
            [(1,)],  # and v2 still points at the v1 that came with it
        )

    def test_rows_load_whatever_order_they_are_in(self):
        import pyarrow.parquet as pq

        self.export()
        for entity in ("RESULT", "ANNOTATION"):
            path = os.path.join(self.bundle, f"{chai_tea_schema.TABLES[entity]}.parquet")
            handle = pq.ParquetFile(path)
            table = handle.read()
            reversed_rows = table.take(list(reversed(range(table.num_rows))))
            pq.write_table(reversed_rows.replace_schema_metadata(handle.schema_arrow.metadata), path)
        self.load()  # a result ahead of the one its previous_id points at must still go in
        self.assert_same_rows(["RESULT", "ANNOTATION"])

    def test_a_bundle_missing_its_parents_leaves_the_central_database_alone(self):
        self.export()
        path = os.path.join(self.bundle, chai_tea_parquet.MANIFEST)
        with open(path) as fh:
            manifest = json.load(fh)
        manifest["tables"] = [t for t in manifest["tables"] if t["entity"] in ("RESULT", "ANNOTATION")]
        with open(path, "w") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(Exception) as caught:
            self.load()
        self.assertIn("foreign key", str(caught.exception).lower())
        self.assertEqual(self.query(self.central, "SELECT count(*) FROM results"), [(0,)])
        self.assertEqual(self.query(self.central, "SELECT count(*) FROM annotations"), [(0,)])

    def test_loading_into_a_database_with_no_schema_says_so(self):
        self.export()
        ensure_postgres_database({"database": self.central})
        with self.assertRaises(RuntimeError) as caught:
            self.load(create=False)
        self.assertIn("chai_tea_schema.py", str(caught.exception))

    def test_an_empty_table_still_gets_a_typed_file(self):
        conn = self.connect(self.local)
        conn.cursor().execute("DELETE FROM annotations")
        conn.commit()
        conn.close()
        manifest = self.export()
        annotations = [t for t in manifest["tables"] if t["entity"] == "ANNOTATION"][0]
        self.assertEqual(annotations["rows"], 0)
        self.assertEqual(
            chai_tea_parquet.parquet_columns(os.path.join(self.bundle, annotations["file"])),
            chai_tea_parquet.column_kinds("ANNOTATION"),
        )
        self.load()  # and an empty file is nothing to trip over on the way in
        self.assertEqual(self.query(self.central, "SELECT count(*) FROM annotations"), [(0,)])

    def test_the_manifest_records_where_the_export_came_from(self):
        manifest = self.export(since="2026-09-06", project="Ledgers")
        self.assertEqual(manifest["chai_tea_export"], chai_tea_parquet.MANIFEST_VERSION)
        self.assertEqual(manifest["source"]["dbname"], self.local)
        self.assertEqual(manifest["filters"]["since"], "2026-09-06")
        self.assertEqual(manifest["filters"]["project"], "Ledgers")
        self.assertNotIn("password", manifest["source"])

    def test_a_bundle_from_a_later_version_is_refused(self):
        self.export()
        path = os.path.join(self.bundle, chai_tea_parquet.MANIFEST)
        with open(path) as fh:
            manifest = json.load(fh)
        manifest["chai_tea_export"] = chai_tea_parquet.MANIFEST_VERSION + 1
        with open(path, "w") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(ValueError):
            self.load()

    def test_a_bundle_with_no_manifest_is_read_from_its_files(self):
        self.export()
        os.remove(os.path.join(self.bundle, chai_tea_parquet.MANIFEST))
        manifest = chai_tea_parquet.read_manifest(self.bundle)
        self.assertEqual(
            [t["entity"] for t in manifest["tables"]], chai_tea_parquet.export_order()
        )
        self.load()
        self.assert_same_rows(["PROJECT", "WORKFLOW", "RESULT", "ANNOTATION"])

    def test_on_conflict_takes_skip_or_update_only(self):
        self.export()
        with self.assertRaises(ValueError):
            self.load(on_conflict="replace")


if __name__ == "__main__":
    unittest.main()
