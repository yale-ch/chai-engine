import unittest

from chai.result import ItemResult, LabelListResult, ListResult
from chai.workflow import Workflow


class TestReducers(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "reducer_test_wf", "type": "workflow.Workflow"})

    def step(self, tree):
        return self.wf._make_step(tree, self.wf)

    # ---- the built-in convergence pattern: steps fan out, next_steps merge ----

    def test_fan_out_then_merge_dict(self):
        fan = self.step(
            {
                "type": "core.Component",
                "name": "fan out",
                "steps": [
                    {"type": "extractor.WordCountExtractor", "id": "wordcount"},
                    {
                        "type": "classifier.KeywordClassifier",
                        "id": "keywords",
                        "settings": {"labels": {"money": ["$"], "question": ["?"]}},
                    },
                ],
                "next_steps": [{"type": "reducer.MergeDictReducer", "id": "merge", "settings": {"namespaced": True}}],
            }
        )
        out = fan.process(ItemResult("Is $5 enough? $5 it is."))
        self.assertEqual(out.value["wordcount"]["$5"], 2)
        self.assertEqual(set(out.value["keywords"]), {"money", "question"})

    def test_merge_dict_flat_later_wins(self):
        r = self.step({"type": "reducer.MergeDictReducer"})
        merged = r.process(ListResult([ItemResult({"a": 1, "b": 1}), ItemResult({"b": 2})]))
        self.assertEqual(merged.value, {"a": 1, "b": 2})

    # ---- flatten + join over nested gate/iterator trees ----

    def _nested(self):
        inner = ListResult([ItemResult("one"), ListResult([ItemResult("two"), None, ItemResult("")])])
        return ListResult([inner, ItemResult("three")])

    def test_flatten(self):
        r = self.step({"type": "reducer.FlattenReducer"})
        flat = r.process(self._nested())
        self.assertEqual([x.value for x in flat.value], ["one", "two", "three"])

    def test_text_join(self):
        r = self.step({"type": "reducer.TextJoinReducer", "settings": {"separator": " | "}})
        self.assertEqual(r.process(self._nested()).value, "one | two | three")

    # ---- collect: gather branch outputs by producing component ----

    def test_collect_across_gate_branches(self):
        iterator = self.step(
            {
                "type": "iterator.Iterator",
                "id": "iter",
                "steps": [
                    {
                        "type": "gate.ValueTestGate",
                        "settings": {"op": "contains", "value": "?"},
                        "true_steps": [{"type": "describer.FileInfoDescriber", "id": "questions"}],
                        "false_steps": [
                            {
                                "type": "translator.GlossaryTranslator",
                                "id": "statements",
                                "settings": {"glossary": {"cat": "Katze"}},
                            }
                        ],
                    }
                ],
                "next_steps": [
                    {"type": "reducer.CollectReducer", "id": "collect", "settings": {"components": "questions, statements"}}
                ],
            }
        )
        out = iterator.process(ListResult([ItemResult("Is this a cat?"), ItemResult("The cat sat.")]))
        producers = sorted(r.processor.id for r in out.value)
        self.assertEqual(producers, ["questions", "statements"])
        translated = [r.value for r in out.value if r.processor.id == "statements"]
        self.assertEqual(translated, ["The Katze sat."])

    def test_collect_includes_registered_derivatives(self):
        # a classifier registers labels on the iterated item; collect finds them
        iterator = self.step(
            {
                "type": "iterator.Iterator",
                "id": "iter2",
                "steps": [
                    {
                        "type": "classifier.KeywordClassifier",
                        "id": "tagger",
                        "register_on": ["iter2"],
                        "settings": {"labels": {"q": ["?"]}},
                    }
                ],
                "next_steps": [
                    {"type": "reducer.CollectReducer", "id": "collect2", "settings": {"components": ["tagger"]}}
                ],
            }
        )
        out = iterator.process(ListResult([ItemResult("really?")]))
        self.assertTrue(any(isinstance(r, LabelListResult) for r in out.value))

    def test_collect_requires_components(self):
        with self.assertRaises(ValueError):
            self.step({"type": "reducer.CollectReducer", "settings": {}})


if __name__ == "__main__":
    unittest.main()
