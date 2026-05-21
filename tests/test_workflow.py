import os
import tempfile
import unittest
from unittest.mock import patch

import ujson as json

from chai.result import ItemResult, ListResult
from chai.workflow import Workflow


class TestWorkflow(unittest.TestCase):
    def test_init_basic(self):
        tree = {"id": "test_wf", "type": "workflow.Workflow"}
        wf = Workflow(tree)
        self.assertEqual(wf.id, "test_wf")
        self.assertEqual(wf.workflow, wf)
        self.assertEqual(wf.id_counter, -1)
        self.assertIn("test_wf", wf.registry_ids)
        self.assertEqual(wf.registry_ids["test_wf"], wf)

    def test_init_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = os.path.join(tmpdir, "prompts.json")
            library_path = os.path.join(tmpdir, "library.json")

            with open(prompts_path, "w") as f:
                json.dump({"test_prompt": "Hello"}, f)
            with open(library_path, "w") as f:
                json.dump({"lib_item": {"type": "core.Component"}}, f)

            tree = {
                "id": "wf_paths",
                "type": "workflow.Workflow",
                "settings": {"defaults_path": prompts_path, "library_path": library_path},
            }
            wf = Workflow(tree)
            self.assertEqual(wf.default_prompts.get("test_prompt"), "Hello")
            self.assertIn("lib_item", wf.library)

    def test_init_with_directory_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data directory structure within tmpdir
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            prompts_path = os.path.join(data_dir, "prompts.json")
            library_path = os.path.join(data_dir, "library.json")

            with open(prompts_path, "w") as f:
                json.dump({"test_prompt": "HelloDir"}, f)
            with open(library_path, "w") as f:
                json.dump({"lib_item": {"type": "core.Component"}}, f)

            tree = {
                "id": "wf_paths_dir",
                "type": "workflow.Workflow",
                "settings": {"defaults_path": data_dir, "library_path": data_dir},
            }
            wf = Workflow(tree)
            self.assertEqual(wf.default_prompts.get("test_prompt"), "HelloDir")
            self.assertIn("lib_item", wf.library)

    def test_get_component_by_id(self):
        wf = Workflow({"id": "wf1", "type": "workflow.Workflow"})
        self.assertEqual(wf.get_component_by_id("wf1"), wf)
        self.assertIsNone(wf.get_component_by_id("nonexistent"))

    def test_register_component(self):
        wf = Workflow({"id": "wf1", "type": "workflow.Workflow"})

        class DummyComponent:
            def __init__(self, cid):
                self.id = cid

        comp = DummyComponent("comp1")
        wf.register_component(comp)
        self.assertEqual(wf.get_component_by_id("comp1"), comp)

        # Test duplicate registration raises error
        with self.assertRaises(ValueError):
            wf.register_component(comp)

    def test_get_new_id(self):
        wf = Workflow({"id": "wf1", "type": "workflow.Workflow"})
        new_id1 = wf.get_new_id()
        self.assertEqual(new_id1, "wf1_0")

        new_id2 = wf.get_new_id()
        self.assertEqual(new_id2, "wf1_1")

        class DummyComponent:
            def __init__(self, cid):
                self.id = cid

        # Register next ID manually
        wf.register_component(DummyComponent("wf1_2"))
        # get_new_id should now conflict and raise ValueError
        with self.assertRaises(ValueError):
            wf.get_new_id()

    def test_nested_workflow(self):
        parent_wf = Workflow({"id": "parent", "type": "workflow.Workflow"})
        child_tree = {"id": "child", "type": "workflow.Workflow"}
        child_wf = Workflow(child_tree, workflow=parent_wf)

        self.assertEqual(child_wf.workflow, parent_wf)
        self.assertIn("child", parent_wf.registry_ids)
        self.assertEqual(parent_wf.registry_ids["child"], child_wf)

    @patch("chai.core.Component.process")
    def test_run_with_input(self, mock_process):
        tree = {
            "id": "wf1",
            "type": "workflow.Workflow",
            "steps": [{"id": "step1", "type": "core.Component"}],
        }
        wf = Workflow(tree)

        # When process is called, return an ItemResult
        mock_process.return_value = ItemResult("step1_res")

        input_res = ItemResult("start")
        res = wf.run(input=input_res)

        self.assertIsInstance(res, ListResult)
        self.assertEqual(len(res.value), 1)
        self.assertEqual(res.value[0].value, "step1_res")
        mock_process.assert_called_once_with(input_res)

    def test_run_no_input_value_error(self):
        tree = {
            "id": "wf1",
            "type": "workflow.Workflow",
            "steps": [{"id": "step1", "type": "core.Component"}],
        }
        wf = Workflow(tree)
        with self.assertRaises(ValueError) as ctx:
            wf.run()
        self.assertEqual(str(ctx.exception), "No input value provided")

    @patch("chai.provider.Provider.run")
    def test_run_with_step_input(self, mock_run):
        tree = {
            "id": "wf1",
            "type": "workflow.Workflow",
            "steps": [{"id": "step1", "type": "provider.Provider", "input": "some_input"}],
        }
        wf = Workflow(tree)
        mock_run.return_value = ItemResult("provider_res")

        res = wf.run()

        self.assertIsInstance(res, ListResult)
        self.assertEqual(len(res.value), 1)
        self.assertEqual(res.value[0].value, "provider_res")
        mock_run.assert_called_once()

    def test_run_chained_steps(self):
        tree = {
            "id": "wf1",
            "type": "workflow.Workflow",
            "steps": [
                {"id": "step1", "type": "core.Component"},
                {"id": "step2", "type": "core.Component"},
            ],
        }
        wf = Workflow(tree)

        with (
            patch.object(wf.steps[0], "process") as mock_process1,
            patch.object(wf.steps[1], "process") as mock_process2,
        ):
            mock_process1.return_value = ItemResult("res1")
            mock_process2.return_value = ItemResult("res2")

            input_res = ItemResult("start")
            res = wf.run(input=input_res)

            # Since res.append(input) happens in the loop:
            # First loop: s.process(input) -> input = res1. res.append(res1)
            # Second loop: s.process(input) -> input = res2. res.append(res2)
            self.assertEqual(len(res.value), 2)
            self.assertEqual(res.value[0].value, "res1")
            self.assertEqual(res.value[1].value, "res2")

            mock_process1.assert_called_once_with(input_res)
            mock_process2.assert_called_once_with(mock_process1.return_value)


if __name__ == "__main__":
    unittest.main()
