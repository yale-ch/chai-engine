"""How steps compose: ``steps`` vs ``next_steps``, with and without an Iterator.

``IntListProvider`` returns [1, 2, 3] and ``DoubleExtractor`` multiplies its input's value by 2, so
each pattern's output says which values each step was given.
"""

import unittest

from chai.result import Result
from chai.workflow import Workflow


def flatten(res):
    """Recursively pull the plain (non-Result) values out of a result tree."""
    vals = []
    if isinstance(res.value, list):
        for v in res.value:
            if isinstance(v, Result):
                vals.extend(flatten(v))
            else:
                vals.append(v)
    elif isinstance(res.value, Result):
        vals.extend(flatten(res.value))
    else:
        vals.append(res.value)
    return vals


def ints_workflow(provider_steps):
    """A workflow of an IntListProvider whose ``steps`` are *provider_steps*."""
    return Workflow(
        {
            "id": "ints_wf",
            "type": "workflow.Workflow",
            # IntListProvider ignores its input, but a provider only starts a run with one configured
            "steps": [{"type": "provider.IntListProvider", "input": "ints", "steps": provider_steps}],
        }
    )


class TestComposition(unittest.TestCase):
    def test_iterator_doubles_each_entry(self):
        wf = ints_workflow([{"type": "iterator.Iterator", "steps": [{"type": "extractor.DoubleExtractor"}]}])
        self.assertEqual(flatten(wf.run()), [2, 4, 6])

    def test_without_iterator_the_step_gets_the_whole_list(self):
        # The extractor is given the ListResult itself: Python semantics repeat the list
        wf = ints_workflow([{"type": "extractor.DoubleExtractor"}])
        self.assertEqual(flatten(wf.run()), [1, 2, 3, 1, 2, 3])

    def test_next_steps_chain(self):
        # next_steps receive the step's output, so two doublings compose: each entry * 4
        double_twice = {"type": "extractor.DoubleExtractor", "next_steps": [{"type": "extractor.DoubleExtractor"}]}
        wf = ints_workflow([{"type": "iterator.Iterator", "steps": [double_twice]}])
        self.assertEqual(flatten(wf.run()), [4, 8, 12])

    def test_sibling_steps_do_not_chain(self):
        # Sibling steps each receive the ORIGINAL entry, not the previous sibling's output
        wf = ints_workflow(
            [
                {
                    "id": "int_iter",
                    "type": "iterator.Iterator",
                    "steps": [
                        {"id": "double_a", "type": "extractor.DoubleExtractor"},
                        {"id": "double_b", "type": "extractor.DoubleExtractor"},
                    ],
                }
            ]
        )
        res = wf.run()
        self.assertEqual(flatten(res), [2, 2, 4, 4, 6, 6])
        # Both siblings' outputs are kept per entry; the last one is the entry's final answer
        entries = res.value[0].value[0].value
        self.assertEqual([entry.value[-1].value for entry in entries], [2, 4, 6])
        self.assertEqual([[step.processor.id for step in entry.value] for entry in entries], [["double_a", "double_b"]] * 3)


if __name__ == "__main__":
    unittest.main()
