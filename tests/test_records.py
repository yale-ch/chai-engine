"""The objects a run is made of carry the columns of their CHAI-TEA rows (``chai.records``)."""

import time
import unittest
import uuid
from datetime import datetime, timezone

from chai.records import Project, WorkflowRun, as_uuid
from chai.result import ItemResult
from chai.schema import column_types
from chai.workflow import Workflow


def doubling_workflow(**config):
    tree = {
        "id": "records_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "type": "provider.StaticProvider",
                "settings": {"values": ["a", "b"]},
                "input": "x",
                "steps": [{"type": "iterator.Iterator", "steps": [{"id": "double", "type": "extractor.DoubleExtractor"}]}],
            }
        ],
    }
    tree.update(config)
    return Workflow(tree)


class TestWorkflowRow(unittest.TestCase):
    def test_the_workflow_columns_come_from_its_config(self):
        wf = doubling_workflow(
            name="M804",
            description="Transcribe and translate each page",
            project={"name": "M804", "description": "Froissart, Chroniques"},
            created="2026-09-01T12:00:00+00:00",
            modified="2026-09-02T08:30:00Z",
            is_active=False,
        )
        row = wf.to_row()
        self.assertEqual(list(row), list(column_types("WORKFLOW")))
        self.assertEqual(row["name"], "M804")
        self.assertEqual(row["description"], "Transcribe and translate each page")
        self.assertEqual(row["project_id"], wf.project.id)
        self.assertFalse(row["is_active"])
        self.assertEqual(row["created"], datetime(2026, 9, 1, 12, tzinfo=timezone.utc))
        self.assertEqual(row["modified"], datetime(2026, 9, 2, 8, 30, tzinfo=timezone.utc))
        self.assertEqual(wf.project.description, "Froissart, Chroniques")

    def test_defaults(self):
        before = datetime.now(timezone.utc)
        wf = doubling_workflow()
        self.assertTrue(wf.is_active)
        self.assertIsNone(wf.description)
        self.assertEqual(wf.project.name, "default")
        self.assertGreaterEqual(wf.created, before)
        self.assertEqual(wf.modified, wf.created)
        self.assertEqual(wf.uuid, as_uuid("records_wf"))  # derived from the config id, so stable

    def test_to_config_keeps_every_column(self):
        wf = doubling_workflow(name="M804", description="pages", project="M804", previous_id="an-earlier-version")
        wf.touch()
        again = Workflow(wf.to_config())
        self.assertEqual(again.to_row(), wf.to_row())
        self.assertEqual(again.project.to_row(), wf.project.to_row())
        self.assertEqual(again.previous_id, as_uuid("an-earlier-version"))

    def test_touch_moves_modified(self):
        wf = doubling_workflow(created="2026-01-01T00:00:00Z")
        wf.touch()
        self.assertGreater(wf.modified, wf.created)


class TestProject(unittest.TestCase):
    def test_a_named_project_has_a_stable_id(self):
        self.assertEqual(Project("M804").id, Project.from_config("M804").id)
        self.assertNotEqual(Project("M804").id, Project("M805").id)
        given = str(uuid.uuid4())
        self.assertEqual(Project.from_config({"name": "M804", "id": given}).id, given)

    def test_the_row_is_the_project_table(self):
        project = Project.from_config({"name": "M804", "planned_enddate": "2027-06-30", "auth_key": "k"})
        row = project.to_row()
        self.assertEqual(list(row), list(column_types("PROJECT")))
        self.assertEqual(row["planned_enddate"], datetime(2027, 6, 30, tzinfo=timezone.utc))
        self.assertEqual(row["auth_key"], "k")


class TestWorkflowRun(unittest.TestCase):
    def test_each_run_is_started_and_finished(self):
        wf = doubling_workflow()
        before_run = wf.run_id  # there is a run id before the first run, for results made outside one
        wf.run()
        run = wf.current_run
        self.assertNotEqual(run.id, before_run)
        self.assertEqual(run.workflow_id, wf.uuid)
        self.assertTrue(run.was_successful)
        self.assertGreater(run.duration, 0)
        self.assertIsNotNone(run.last_run)
        self.assertIn("usage", run.metadata)
        row = run.to_row()
        self.assertEqual(list(row), list(column_types("WORKFLOW_RUN")))
        wf.run()
        self.assertNotEqual(wf.run_id, run.id)

    def test_a_failed_run_says_so(self):
        wf = doubling_workflow()
        wf.get_component_by_id("double")._process = lambda input: 1 / 0
        with self.assertRaises(ZeroDivisionError):
            wf.run()
        self.assertFalse(wf.current_run.was_successful)
        self.assertIn("division", wf.current_run.extra_data["error"])

    def test_records_are_the_rows_a_run_hangs_under(self):
        wf = doubling_workflow(project="M804")
        records = wf.records()
        self.assertEqual(set(records), {"PROJECT", "WORKFLOW", "WORKFLOW_RUN"})
        self.assertEqual(records["WORKFLOW"]["project_id"], records["PROJECT"]["id"])
        self.assertEqual(records["WORKFLOW_RUN"]["workflow_id"], records["WORKFLOW"]["id"])

    def test_finish_without_start_has_no_duration(self):
        run = WorkflowRun()
        run.finish(True)
        self.assertIsNone(run.duration)
        self.assertTrue(run.was_successful)


class TestResultColumns(unittest.TestCase):
    def test_a_result_has_its_own_timestamps_and_editor(self):
        before = datetime.now(timezone.utc)
        res = ItemResult("x")
        self.assertGreaterEqual(res.created, before)
        self.assertEqual(res.modified, res.created)
        self.assertIsNone(res.editor_user_id)
        self.assertTrue(res.was_successful)

    def test_changing_the_value_moves_modified(self):
        res = ItemResult("x")
        time.sleep(0.001)
        res.value = "y"
        self.assertGreater(res.modified, res.created)

    def test_an_error_result_was_not_successful(self):
        self.assertFalse(ItemResult(None, metadata={"type": "ERROR", "error": "boom"}).was_successful)


if __name__ == "__main__":
    unittest.main()
