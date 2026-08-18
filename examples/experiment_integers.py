"""Test workflows for integer doubling use cases.

Uses IntListProvider (returns [1, 2, 3]) and DoubleExtractor (value * 2) to
test which composition patterns the engine supports:

1. Iterate and double each integer            -> [2, 4, 6]
2. Double the whole list (Python semantics)   -> [1, 2, 3, 1, 2, 3]
3. Iterate and double twice, chained          -> [4, 8, 12]
4. Iterate and double twice, independently    -> [2, 4, 6] (middle result not composed)
"""

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


# Use case 1: iterate through each integer and multiply by 2
uc1 = {
    "id": "uc1",
    "type": "Workflow",
    "name": "Iterate and double each item",
    "steps": [
        {
            "type": "provider.IntListProvider",
            "steps": [
                {
                    "type": "iterator.Iterator",
                    "steps": [{"type": "extractor.DoubleExtractor"}],
                }
            ],
        }
    ],
}

# Use case 2: multiply the entire list by 2 ([1,2,3] -> [1,2,3,1,2,3])
# The extractor is applied to the ListResult itself, no Iterator
uc2 = {
    "id": "uc2",
    "type": "Workflow",
    "name": "Double the whole list",
    "steps": [
        {
            "type": "provider.IntListProvider",
            "steps": [{"type": "extractor.DoubleExtractor"}],
        }
    ],
}

# Use case 3: iterate and double twice, chained, so each item is *4.
# Sequential composition uses next_steps: the second extractor receives
# the first extractor's result.
uc3 = {
    "id": "uc3",
    "type": "Workflow",
    "name": "Iterate and double twice, chained (*4)",
    "steps": [
        {
            "type": "provider.IntListProvider",
            "steps": [
                {
                    "type": "iterator.Iterator",
                    "steps": [
                        {
                            "type": "extractor.DoubleExtractor",
                            "next_steps": [{"type": "extractor.DoubleExtractor"}],
                        }
                    ],
                }
            ],
        }
    ],
}

# Use case 4: iterate and double twice, independently, so the final answer
# is *2. Sibling steps inside an Iterator each receive the ORIGINAL item,
# not the previous step's result, so the second doubling supersedes the
# first rather than composing with it.
uc4 = {
    "id": "uc4",
    "type": "Workflow",
    "name": "Iterate and double twice, independently (*2)",
    "steps": [
        {
            "type": "provider.IntListProvider",
            "steps": [
                {
                    "id": "int_iter",
                    "type": "iterator.Iterator",
                    "steps": [
                        {"id": "double_a", "type": "extractor.DoubleExtractor"},
                        {"id": "double_b", "type": "extractor.DoubleExtractor"},
                    ],
                }
            ],
        }
    ],
}


def run(tree, expected):
    wf = Workflow(tree)
    res = wf.run()
    vals = flatten(res)
    status = "PASS" if vals == expected else "FAIL"
    print(f"\n=== {tree['name']} [{status}] ===")
    print(f"expected: {expected}")
    print(f"got:      {vals}")
    res.view()
    return res


if __name__ == "__main__":
    run(uc1, [2, 4, 6])
    run(uc2, [1, 2, 3, 1, 2, 3])
    run(uc3, [4, 8, 12])

    res4 = run(uc4, [2, 2, 4, 4, 6, 6])
    # Per item, the last sibling's result is the final answer (*2); the
    # earlier sibling's result is retained in the tree but never composed.
    iter_result = res4.value[0].value[0]
    finals = [step_value.value[-1].value for step_value in iter_result.value]
    print(f"final answer per item (last sibling wins): {finals}")
