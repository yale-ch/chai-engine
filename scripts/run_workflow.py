#!/usr/bin/env python3
"""Run a chai workflow from its JSON definition.

    python scripts/run_workflow.py examples/workflow-m804-lines.json
    python scripts/run_workflow.py examples/workflow-m804.json --input inputs/m804/split
    python scripts/run_workflow.py wf.json --input a.jpg --input b.jpg --set transcribe.model=gemini-3.8-flash
    python scripts/run_workflow.py wf.json --set each_page.workers=4 --json
    python scripts/run_workflow.py examples/workflow-experiment-ppl.json --text "Sanderson, Robert D 1976-"
    python scripts/run_workflow.py wf.json --text - < page.txt

The run happens in the repository root by default (``--root`` to change it), because the paths in
the workflow files -- ``inputs/...``, ``results/...`` -- are relative to it; the workflow path and
``--input`` paths given on the command line are taken relative to where you are.

``--input`` replaces the ``input`` of the first top-level step (normally the provider): once for a
single value (a directory, a file), repeated for a list (a ``FileListProvider``'s files).
``--text`` is for a workflow with no provider of its own: the text is handed to the run as a TEXT
``ItemResult``, or, repeated, as a ``ListResult`` of them (one per ``--text``); ``-`` reads the text
from standard input. It cannot be combined with ``--input``, and is refused when the first
top-level step has an ``input`` of its own, which would ignore it.
``--set ID.KEY=VALUE`` sets one setting of the component with that id before the workflow is built;
the value is read as JSON when it parses (``4``, ``true``, ``["a","b"]``) and as a string otherwise.

When the run ends, its id, outcome, duration and model usage are printed (``--json`` prints the
WORKFLOW_RUN row, whose metadata holds the usage summary, as JSON instead); the exit status is 1 if the run failed.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# scripts/ sits outside the package, so put the repository root on the path to import chai
sys.path.insert(0, str(ROOT))

BRANCHES = ("steps", "next_steps", "error_steps", "true_steps", "false_steps", "default_steps",
            "file_steps")


def walk(tree):
    """Yield every component config in *tree*, depth first."""
    yield tree
    for key in BRANCHES:
        for child in tree.get(key) or []:
            yield from walk(child)
    for children in (tree.get("case_steps") or {}).values():
        for child in children:
            yield from walk(child)


def find_step(tree, cid):
    return next((node for node in walk(tree) if node.get("id") == cid), None)


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_settings(tree, assignments):
    """Apply ``ID.KEY=VALUE`` *assignments* to the component configs in *tree*."""
    for item in assignments:
        target, sep, value = item.partition("=")
        cid, dot, key = target.rpartition(".")
        if not sep or not dot or not cid or not key:
            raise SystemExit(f"--set expects ID.KEY=VALUE, not {item!r}")
        step = find_step(tree, cid)
        if step is None:
            raise SystemExit(f"--set: no component with id {cid!r} in the workflow")
        step.setdefault("settings", {})[key] = parse_value(value)


def apply_input(tree, inputs):
    """Make *inputs* the ``input`` of the workflow's first top-level step."""
    steps = tree.get("steps") or []
    if not steps:
        raise SystemExit("--input: the workflow has no top-level steps")
    paths = [str(Path(p).resolve()) if Path(p).exists() else p for p in inputs]
    steps[0]["input"] = paths[0] if len(paths) == 1 else paths


def text_input(texts, tree):
    """The run's input for *texts*: one TEXT ItemResult, or a ListResult of them."""
    from chai.result import ItemResult, ListResult

    steps = tree.get("steps") or []
    if steps and steps[0].get("input") is not None:
        raise SystemExit(f"--text: the first step ({steps[0].get('id') or steps[0].get('type')}) has an "
                         "input of its own and would ignore it; use --input")
    stdin = None
    items = []
    for text in texts:
        if text == "-":
            stdin = sys.stdin.read() if stdin is None else stdin
            text = stdin
        items.append(ItemResult(text, metadata={"type": "TEXT"}))
    return items[0] if len(items) == 1 else ListResult(items)


def usage_lines(usage):
    totals = (f"{usage.get('calls', 0)} model calls, {usage.get('total_tokens', 0)} tokens, "
              f"${usage.get('cost', 0.0):.4f}")
    if usage.get("unpriced_calls"):
        totals += f" ({usage['unpriced_calls']} calls unpriced)"
    lines = [f"usage:    {totals}"]
    for model, row in (usage.get("by_model") or {}).items():
        line = (f"  {model}: {row.get('calls', 0)} calls, {row.get('total_tokens', 0)} tokens, "
                f"${row.get('cost', 0.0):.4f}")
        if row.get("local_calls"):
            line += f", {row.get('local_hours', 0.0):.4f} h local"
        lines.append(line)
    for machine, row in (usage.get("by_hardware") or {}).items():
        devices = ", ".join(row.get("devices") or []) or "local server"
        lines.append(f"local:    {row.get('local_calls', 0)} calls, {row.get('local_hours', 0.0):.4f} h "
                     f"on {machine} ({devices})")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow", help="workflow definition (JSON)")
    given = parser.add_mutually_exclusive_group()
    given.add_argument("-i", "--input", action="append", default=[],
                       help="input for the first top-level step; repeat for a list")
    given.add_argument("-t", "--text", action="append", default=[],
                       help="text to run the workflow on ('-' for standard input); repeat for a list")
    parser.add_argument("-s", "--set", action="append", default=[], metavar="ID.KEY=VALUE",
                        help="set a component's setting before the run; repeatable")
    parser.add_argument("--root", default=str(ROOT),
                        help="directory to run in, which the workflow's paths are relative to "
                             "(default: the repository root)")
    parser.add_argument("--json", action="store_true", help="print the run's record as JSON")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    path = Path(args.workflow).resolve()
    tree = json.loads(path.read_text())
    if args.input:
        apply_input(tree, args.input)
    apply_settings(tree, args.set)

    os.chdir(args.root)
    from chai.core import result_preview
    from chai.workflow import Workflow

    run_input = text_input(args.text, tree) if args.text else None
    try:
        wf = Workflow(tree)
    except Exception as e:  # a bad config: say what, without the traceback of building it
        raise SystemExit(f"Could not build the workflow in {path}: {e}")
    error = None
    res = None
    try:
        res = wf.run(run_input)
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as e:  # the run is finished as failed by Workflow.run; report it below
        logging.getLogger("chai").exception("workflow failed")
        error = str(e)

    run = wf.current_run
    usage = (run.metadata or {}).get("usage") or wf.usage.summary()
    if args.json:
        print(json.dumps({"run": run.to_row(), "error": error}, indent=2, default=str))
    else:
        outcome = "succeeded" if run.was_successful else f"failed: {error}"
        print(f"workflow: {wf.id} ({path.name})")
        print(f"run:      {run.id}")
        print(f"outcome:  {outcome}")
        if run.duration is not None:
            print(f"duration: {run.duration:.1f}s")
        print("\n".join(usage_lines(usage)))
        if res is not None:
            print(f"result:   {result_preview(res)}")
    return 0 if run.was_successful else 1


if __name__ == "__main__":
    sys.exit(main())
