"""Run workflow-ppl.json over a CSV of people and write the results as JSON-Lines.

One line per CSV row, written as soon as that row has been parsed, pairing the row it came from with
the parse of its name:

    {"input": {"name": "Ford, Brinsley [Sir, 1908-1999] (introduction by)", "title": ...},
     "parsed": {"first_name": "Brinsley", "last_name": "Ford", "titles": ["Sir"], ...},
     "metadata": {"type": "DATA", "timestamp": 1755000000.0, ...}}

The writing is done by the JsonLinesStorage steps in the workflow itself, which append each result to
the file as it is produced; the row iterator runs with ``retain_results: false``, so nothing is held
in memory until the end of the run and the output file is complete and valid at every moment. Rows
the workflow could not parse still get a line, with "error" in place of "parsed" (the workflow gives
both extraction steps an error branch that writes to the same file), so the output always has one
line per row read. This script only points those steps at the output file and reports what happened.

Usage (from the repository root):

    python examples/experiment-ppl-jsonl.py
    python examples/experiment-ppl-jsonl.py --csv inputs/people.csv --limit 0 --out parsed.jsonl

--slice and --max-slices divide the rows between runs, so the work can be done by several processes
at once -- each takes every max_slices'th row and writes its own file:

    for s in $(seq 0 23); do
        python examples/experiment-ppl-jsonl.py --limit 0 --slice $s --max-slices 24 &
    done; wait
    cat results/people-parsed.*-of-24.jsonl > results/people-parsed.jsonl

Note that --limit caps the rows *read* from the CSV before they are divided up, so a run with
--limit 3 --max-slices 24 has only rows 0-2 to slice between its 24 runs. Two runs must never share
an output file: the file is truncated when the run starts, and separate processes do not share the
lock that keeps concurrent lines apart.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# examples/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, ROOT)

import ujson as json  # noqa: E402

from chai.result import Result  # noqa: E402
from chai.workflow import Workflow  # noqa: E402

BRANCHES = ("steps", "next_steps", "error_steps", "true_steps", "false_steps")

# Component ids from workflow-ppl.json
PROVIDER_ID = "ppl_csv"
ITERATOR_ID = "ppl_rows"


def find_step(tree, cid):
    """Find the config dict for the step with id *cid* anywhere in a workflow config *tree*."""
    if tree.get("id", None) == cid:
        return tree
    for key in BRANCHES:
        for step in tree.get(key, []):
            found = find_step(step, cid)
            if found is not None:
                return found
    return None


def set_jsonl_output(tree, path, count=0):
    """Point every JsonLinesStorage step in *tree* at *path*, truncating it at the start of the run.

    The workflow has one such step per outcome -- the parse, and the error branch of each extraction
    step -- all writing to the same file, so they are found by type rather than by id. Only the first
    one truncates: 'append' for the rest, or each would empty what the others had written.
    """
    if "JsonLinesStorage" in f"{tree.get('type', '')}{tree.get('base', '')}":
        settings = tree.setdefault("settings", {})
        settings["file"] = path
        settings["mode"] = "truncate" if count == 0 else "append"
        count += 1
    for key in BRANCHES:
        for step in tree.get(key, []):
            count = set_jsonl_output(step, path, count)
    return count


def find_result(result, component):
    """Find the result *component* produced, searching *result* and its nested values.

    Top-down, so an Iterator's merged output is found before its per-entry lists (both carry the
    Iterator as their processor).
    """
    if not isinstance(result, Result):
        return None
    if result.processor is component:
        return result
    # Never touch a FileItemResult's value; reading it would pull the file off disk
    if not getattr(result, "file_name", "") and type(result.value) is list:
        for child in result.value:
            found = find_result(child, component)
            if found is not None:
                return found
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workflow", default=os.path.join(ROOT, "workflow-ppl.json"))
    ap.add_argument("--csv", default=None, help="CSV to read (default: the workflow's own input)")
    ap.add_argument("--limit", type=int, default=None, help="rows to read; 0 for all of them")
    ap.add_argument("--slice", type=int, default=None, help="zero-based slice of the rows to process")
    ap.add_argument(
        "--max-slices", type=int, default=None, help="how many slices to divide the rows into"
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output .jsonl (default: results/people-parsed[.<slice>-of-<max_slices>].jsonl)",
    )
    args = ap.parse_args()

    with open(args.workflow) as fh:
        tree = json.load(fh)

    provider = find_step(tree, PROVIDER_ID)
    if provider is None:
        raise SystemExit(f"No '{PROVIDER_ID}' step in {args.workflow}")
    if args.csv is not None:
        provider["input"] = args.csv
    if args.limit is not None:
        provider.setdefault("settings", {})["limit"] = args.limit
    csv_path = provider.get("input", "")
    if csv_path and not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        # Paths in the config are relative to the repository root, not the current directory
        provider["input"] = os.path.join(ROOT, csv_path)

    iterator = find_step(tree, ITERATOR_ID)
    if iterator is None:
        raise SystemExit(f"No '{ITERATOR_ID}' step in {args.workflow}")
    settings = iterator.setdefault("settings", {})
    if args.max_slices is not None:
        settings["max_slices"] = args.max_slices
    if args.slice is not None:
        settings["slice"] = args.slice
    max_slices = int(settings.get("max_slices", 1) or 1)
    offset = int(settings.get("slice", 0) or 0)
    # Check here as well as in the iterator, so a bad slice fails before the model is loaded
    if max_slices < 1 or not 0 <= offset < max_slices:
        raise SystemExit(f"--slice must be between 0 and {max(max_slices - 1, 0)}, got {offset}")

    # Each slice needs its own file, so parallel runs don't overwrite each other's output
    out = args.out
    if out is None:
        name = "people-parsed" if max_slices == 1 else f"people-parsed.{offset}-of-{max_slices}"
        out = os.path.join(ROOT, "results", f"{name}.jsonl")
    out = os.path.abspath(out)
    if not set_jsonl_output(tree, out):
        raise SystemExit(f"No JsonLinesStorage step to write to in {args.workflow}")

    wf = Workflow(tree)
    res = wf.run()

    iter_result = find_result(res, wf.get_component_by_id(ITERATOR_ID))
    if iter_result is None:
        raise SystemExit(f"Workflow produced no output from '{ITERATOR_ID}'")

    rows = iter_result.metadata.get("processed", 0)
    # Rows that failed outside the two error branches, and so have no line of their own
    skipped = iter_result.metadata.get("errors", 0)
    written = sum(1 for _ in open(out))  # the parsed lines and the error branches' together
    where = "" if max_slices == 1 else f" of slice {offset} of {max_slices}"
    print(f"Processed {rows} rows{where}, wrote {written} lines to {out} ({skipped} rows skipped)")


if __name__ == "__main__":
    main()
