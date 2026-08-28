"""End-to-end workflow runs over real files.

These tests drive full JSON workflow trees through ``Workflow.run`` -- provider, iterator,
classifier, label gate and storage working together -- rather than exercising components in
isolation. They pin the canonical wiring documented in ``chai/data/library.json``:
a classifier ``register_on`` the iterator, and a ``LabelTestGate`` testing its labels.
"""

import json
import os
import tempfile
import unittest

from chai.workflow import Workflow


def write_docs(docs_dir):
    """A tiny corpus: two files mention cats, one does not."""
    corpus = {
        "a.txt": "the cat sat on the mat",
        "b.txt": "dogs bark at the moon",
        "c.txt": "another cat appears here",
    }
    for name, text in corpus.items():
        with open(os.path.join(docs_dir, name), "w") as f:
            f.write(text)
    return corpus


class TestProviderIteratorGateStorage(unittest.TestCase):
    def test_cat_files_routed_to_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = os.path.join(tmp, "docs")
            store_dir = os.path.join(tmp, "store")
            os.makedirs(docs_dir)
            write_docs(docs_dir)

            tree = {
                "type": "workflow.Workflow",
                "id": "e2e_wf",
                "steps": [
                    {
                        "type": "provider.DirFileProvider",
                        "id": "prov",
                        "input": docs_dir,
                        "steps": [
                            {
                                "type": "iterator.Iterator",
                                "id": "file_iter",
                                "steps": [
                                    {
                                        "type": "classifier.KeywordClassifier",
                                        "id": "cat_classifier",
                                        "settings": {"labels": {"cat": ["cat"]}},
                                        "register_on": ["file_iter"],
                                    },
                                    {
                                        "type": "gate.LabelTestGate",
                                        "id": "cat_gate",
                                        "settings": {"component": "cat_classifier", "label": ["cat"]},
                                        "true_steps": [
                                            {
                                                "type": "storage.FileSystemStorage",
                                                "id": "store",
                                                "settings": {"directory": store_dir},
                                            }
                                        ],
                                        "false_steps": [],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }

            wf = Workflow(tree)
            wf.run()

            stored = []
            for root, _dirs, files in os.walk(store_dir):
                for f in files:
                    if f.endswith(".json"):
                        with open(os.path.join(root, f)) as fh:
                            stored.append(json.load(fh))
            # only the two cat files should reach storage; the stored value is the file path
            self.assertEqual(
                sorted(os.path.basename(r["value"]) for r in stored), ["a.txt", "c.txt"]
            )


class TestBaseTemplateWorkflowRun(unittest.TestCase):
    def test_base_step_expands_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_path = os.path.join(tmp, "library.json")
            with open(library_path, "w") as f:
                json.dump(
                    {
                        "CatClassifier": {
                            "type": "classifier.KeywordClassifier",
                            "settings": {"labels": {"cat": ["cat"], "dog": ["dog"]}},
                        }
                    },
                    f,
                )

            tree = {
                "type": "workflow.Workflow",
                "id": "base_wf",
                "settings": {"library_path": library_path},
                "steps": [
                    {
                        "type": "provider.StaticProvider",
                        "id": "static",
                        # StaticProvider ignores its input, but Workflow._run only starts
                        # input-less steps when handed a runtime input -- so configure one.
                        "input": "",
                        "settings": {"values": ["a cat and a dog", "just a dog", "nothing here"]},
                        "steps": [
                            {
                                "type": "iterator.Iterator",
                                "id": "it",
                                "steps": [{"base": "CatClassifier", "id": "cls"}],
                            }
                        ],
                    }
                ],
            }

            wf = Workflow(tree)
            result = wf.run()

            # workflow -> provider result -> iterator output: one wrapped label list per value
            iterator_out = result.value[0].value[0]
            labels = [list(entry.value[0].value) for entry in iterator_out.value]
            self.assertEqual(labels, [["cat", "dog"], ["dog"], []])

    def test_base_settings_merge_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_path = os.path.join(tmp, "library.json")
            with open(library_path, "w") as f:
                json.dump(
                    {
                        "Splitter": {
                            "type": "segmenter.TextSegmenter",
                            "settings": {"mode": "line"},
                        }
                    },
                    f,
                )
            tree = {
                "type": "workflow.Workflow",
                "id": "merge_wf",
                "settings": {"library_path": library_path},
                "steps": [{"base": "Splitter", "id": "seg", "settings": {"mode": "word"}}],
            }
            wf = Workflow(tree)
            seg = wf.get_component_by_id("seg")
            # the step's own settings win over the template's
            self.assertEqual(seg.settings["mode"], "word")


if __name__ == "__main__":
    unittest.main()
