import unittest

from chai.result import FileItemResult, ItemResult
from chai.workflow import Workflow


def build_gate(gate_tree):
    """Wrap a gate config in a bare workflow and return the gate instance."""
    wf = Workflow({"id": "gate_test_wf", "type": "workflow.Workflow"})
    return wf._make_step(gate_tree, wf)


def routed_values(result):
    return [r.value for r in result.value] if result is not None else None


TRUE_FALSE = {
    "true_steps": [{"type": "describer.FileInfoDescriber", "name": "T"}],
    "false_steps": [],
}


class TestConditionGate(unittest.TestCase):
    def make(self, condition, **extra):
        tree = {"type": "gate.ConditionGate", "settings": {"condition": condition}, **TRUE_FALSE}
        tree.update(extra)
        return build_gate(tree)

    def test_routes_true_and_false(self):
        gate = self.make({"source": "metadata.yolo_class", "op": "eq", "value": "barcode"})
        barcode = FileItemResult("c.png", metadata={"type": "IMAGE", "yolo_class": "barcode"})
        stamp = FileItemResult("c.png", metadata={"type": "IMAGE", "yolo_class": "stamp"})
        self.assertIsNotNone(gate.process(barcode))  # true branch has a step
        self.assertIsNone(gate.process(stamp))  # false branch is empty -> no-op

    def test_requires_condition(self):
        with self.assertRaises(ValueError):
            build_gate({"type": "gate.ConditionGate", "settings": {}})

    def test_raw_input(self):
        gate = self.make({"source": "value", "op": "contains", "value": "cat"})
        self.assertTrue(gate._test("the cat sat"))
        self.assertFalse(gate._test("only dogs here"))


class TestSugarGates(unittest.TestCase):
    def test_value_test_gate(self):
        gate = build_gate(
            {"type": "gate.ValueTestGate", "settings": {"op": "matches", "value": r"\d{4}"}, **TRUE_FALSE}
        )
        self.assertTrue(gate._test(ItemResult("collected in 1922")))
        self.assertFalse(gate._test(ItemResult("no year here")))

    def test_metadata_test_gate(self):
        gate = build_gate(
            {
                "type": "gate.MetadataTestGate",
                "settings": {"key": "yolo_class", "op": "in", "value": ["label", "barcode"]},
                **TRUE_FALSE,
            }
        )
        self.assertTrue(gate._test(ItemResult("x", metadata={"yolo_class": "label"})))
        self.assertFalse(gate._test(ItemResult("x", metadata={"yolo_class": "stamp"})))
        with self.assertRaises(ValueError):
            build_gate({"type": "gate.MetadataTestGate", "settings": {}})

    def test_threshold_gate(self):
        gate = build_gate({"type": "gate.ThresholdGate", "settings": {"threshold": 0.8}, **TRUE_FALSE})
        self.assertTrue(gate._test(ItemResult("x", metadata={"confidence": 0.92})))
        self.assertFalse(gate._test(ItemResult("x", metadata={"confidence": 0.5})))
        self.assertFalse(gate._test(ItemResult("x")))  # no confidence at all

    def test_file_type_gate(self):
        gate = build_gate({"type": "gate.FileTypeGate", "settings": {"types": "image, text"}, **TRUE_FALSE})
        self.assertTrue(gate._test(ItemResult("x", metadata={"type": "IMAGE"})))
        self.assertFalse(gate._test(ItemResult("x", metadata={"type": "AUDIO"})))


class TestRealComponents(unittest.TestCase):
    """The deterministic components that replaced the Mock* classes."""

    def setUp(self):
        self.wf = Workflow({"id": "real_test_wf", "type": "workflow.Workflow"})

    def step(self, tree):
        return self.wf._make_step(tree, self.wf)

    def test_keyword_classifier(self):
        c = self.step(
            {
                "type": "classifier.KeywordClassifier",
                "settings": {"labels": {"pii": ["ssn", "passport"], "ok": ["public"]}},
            }
        )
        res = c.process(ItemResult("Public record, no SSN given"))
        self.assertEqual(set(res.value), {"pii", "ok"})  # case-insensitive by default

    def test_text_segmenter(self):
        s = self.step({"type": "segmenter.TextSegmenter", "settings": {"mode": "sentence"}})
        res = s.process(ItemResult("One. Two! Three?"))
        self.assertEqual([r for r in res.value], ["One.", "Two!", "Three?"])

    def test_static_provider(self):
        p = self.step({"type": "provider.StaticProvider", "settings": {"values": [1, 2, 3]}})
        self.assertEqual(p.run().value, [1, 2, 3])

    def test_glossary_translator(self):
        t = self.step(
            {
                "type": "translator.GlossaryTranslator",
                "settings": {"glossary": {"cat": "Katze", "the": "die"}, "language": "de"},
            }
        )
        res = t.process(ItemResult("The cat"))
        self.assertEqual(res.value, "die Katze")
        self.assertEqual(res.metadata["language"], "de")

    def test_file_info_describer(self):
        d = self.step({"type": "describer.FileInfoDescriber"})
        res = d.process(ItemResult([1, 2, 3]))
        self.assertEqual(res.metadata["length"], 3)


class TestGateInsideIterator(unittest.TestCase):
    """Regression: ResultIter must not re-wrap Results (hiding their metadata),
    or gates inside an Iterator can never see e.g. YOLO crop metadata."""

    def test_threshold_gate_per_item(self):
        wf = Workflow({"id": "iter_gate_wf", "type": "workflow.Workflow"})
        it = wf._make_step(
            {
                "type": "iterator.Iterator",
                "steps": [
                    {
                        "type": "gate.ThresholdGate",
                        "settings": {"threshold": 0.8},
                        "true_steps": [{"type": "describer.FileInfoDescriber"}],
                    }
                ],
            },
            wf,
        )
        items = [
            ItemResult("hi", metadata={"confidence": 0.92}),
            ItemResult("lo", metadata={"confidence": 0.4}),
        ]
        from chai.result import ListResult

        merged = it.process(ListResult(items))
        routed = [entry.value[0] for entry in merged.value]
        self.assertIsNotNone(routed[0])  # confident -> described
        self.assertIsNone(routed[1])  # not confident -> empty false branch


if __name__ == "__main__":
    unittest.main()
