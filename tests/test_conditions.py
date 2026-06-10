import unittest

from chai.conditions import MISSING, apply_op, evaluate, resolve_source
from chai.result import FileItemResult, ItemResult, LabelListResult


class TestResolveSource(unittest.TestCase):
    def test_value(self):
        self.assertEqual(resolve_source(ItemResult("hello"), "value"), "hello")

    def test_raw_input_value(self):
        # First-step gates receive the raw workflow input, not a Result
        self.assertEqual(resolve_source("raw", "value"), "raw")
        self.assertIs(resolve_source("raw", "metadata.x"), MISSING)

    def test_metadata_dotted(self):
        r = ItemResult("x", metadata={"a": {"b": 3}})
        self.assertEqual(resolve_source(r, "metadata.a.b"), 3)
        self.assertIs(resolve_source(r, "metadata.a.zzz"), MISSING)

    def test_type_shorthand(self):
        r = ItemResult("x", metadata={"type": "IMAGE"})
        self.assertEqual(resolve_source(r, "type"), "IMAGE")

    def test_input_chain(self):
        parent = ItemResult("the source", metadata={"type": "TEXT"})
        child = ItemResult("derived", input=parent)
        self.assertEqual(resolve_source(child, "input.value"), "the source")
        self.assertEqual(resolve_source(child, "input.metadata.type"), "TEXT")

    def test_labels(self):
        r = LabelListResult(["cat", "dog"])
        self.assertEqual(resolve_source(r, "labels"), ["cat", "dog"])

    def test_unknown_source(self):
        with self.assertRaises(ValueError):
            resolve_source(ItemResult("x"), "bogus")


class TestApplyOp(unittest.TestCase):
    def test_comparisons(self):
        self.assertTrue(apply_op("gte", 0.9, 0.5))
        self.assertFalse(apply_op("gte", 0.4, 0.5))
        self.assertTrue(apply_op("lt", "3", 4))  # numeric coercion from strings
        self.assertFalse(apply_op("gt", "abc", 4))  # non-numeric never passes

    def test_contains_in_intersects(self):
        self.assertTrue(apply_op("contains", "hello world", "world"))
        self.assertTrue(apply_op("contains", ["a", "b"], "a"))
        self.assertTrue(apply_op("in", "IMAGE", ["IMAGE", "TEXT"]))
        self.assertTrue(apply_op("intersects", ["x", "y"], ["y", "z"]))
        self.assertFalse(apply_op("intersects", ["x"], ["y"]))

    def test_matches(self):
        self.assertTrue(apply_op("matches", "barcode-123", r"barcode-\d+"))

    def test_missing(self):
        self.assertFalse(apply_op("eq", MISSING, "x"))
        self.assertTrue(apply_op("ne", MISSING, "x"))
        self.assertFalse(apply_op("exists", MISSING, None))
        self.assertFalse(apply_op("truthy", MISSING, None))

    def test_unknown_op(self):
        with self.assertRaises(ValueError):
            apply_op("zorp", 1, 1)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.crop = FileItemResult(
            "crop.png",
            metadata={"type": "IMAGE", "yolo_class": "barcode", "confidence": 0.92},
        )

    def test_leaf(self):
        self.assertTrue(evaluate({"source": "metadata.yolo_class", "op": "eq", "value": "barcode"}, self.crop))
        self.assertFalse(evaluate({"source": "metadata.confidence", "op": "gte", "value": 0.95}, self.crop))

    def test_combinators(self):
        cond = {
            "all": [
                {"source": "type", "op": "eq", "value": "IMAGE"},
                {
                    "any": [
                        {"source": "metadata.yolo_class", "op": "in", "value": ["label", "barcode"]},
                        {"source": "metadata.confidence", "op": "gte", "value": 0.99},
                    ]
                },
            ]
        }
        self.assertTrue(evaluate(cond, self.crop))
        self.assertFalse(evaluate({"not": cond}, self.crop))

    def test_default_source_and_op(self):
        self.assertTrue(evaluate({}, ItemResult("non-empty")))
        self.assertFalse(evaluate({}, ItemResult("")))


if __name__ == "__main__":
    unittest.main()
