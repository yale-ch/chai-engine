"""Run examples/workflow-lux-routing.json on a directory of images and report where each one went.

Gemini 3.1 Flash-Lite sorts each image into one of six kinds; photographs get alt-text (Gemini 3.1
Flash-Lite), typed documents, newspapers and easy handwriting are read by Chandra OCR 2 on this
machine, and hard handwriting goes to Gemini 3.8 Flash. Every result is a CHAI-TEA RESULT row in
``results/lux_routing/run.jsonl``; this script reads the rows back and prints, per image, the label,
the reader it went to, how long that took, and what it cost -- dollars for a hosted model, hours on
named hardware for a local one -- then the run's totals.

    python examples/lux_routing.py                       # inputs/lux_routing
    python examples/lux_routing.py --images path/to/images --expected examples/lux-routing-images.csv

The twenty images used to try it are Yale LUX pages listed in ``examples/lux-routing-images.csv``
(LUX record, IIIF manifest, canvas and image URL, and the kind each was picked as); fetch them from
the ``image`` column into a directory under the names in ``file``. ``--expected`` compares the
classifier's labels with that column. Chandra needs its weights (``datalab-to/chandra-ocr-2``, about
10 GB) and runs on CUDA, Apple Silicon or the CPU, whichever is there.
"""

import argparse
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from chai.hardware import hardware_label  # noqa: E402
from chai.workflow import Workflow  # noqa: E402

WORKFLOW = ROOT / "examples" / "workflow-lux-routing.json"
ROUTES = {"alt_text": "alt-text", "chandra": "Chandra OCR 2 (local)", "gemini_htr": "Gemini 3.8 Flash"}


def find_step(tree, cid):
    if tree.get("id") == cid:
        return tree
    for key in ("steps", "next_steps", "true_steps", "false_steps"):
        for child in tree.get(key, []):
            found = find_step(child, cid)
            if found:
                return found
    return None


def build(images, log):
    tree = json.loads(WORKFLOW.read_text())
    find_step(tree, "images")["input"] = str(images)
    for cid in ("log_classify", "log_alt_text", "log_gemini_htr", "log_chandra"):
        find_step(tree, cid)["settings"]["file"] = str(log)
    return Workflow(tree)


def report(log, expected=None):
    rows = [json.loads(line) for line in open(log)]
    by_image = {}
    for row in rows:
        by_image.setdefault(os.path.basename(row["input"] or "?"), {})[row["process_id"]] = row
    agree = 0
    print(f"\n{'image':44} {'label':30} {'read by':24} {'seconds':>8}  cost / hardware")
    for name in sorted(by_image):
        steps = by_image[name]
        label = ", ".join(steps["classify"]["value"]) if "classify" in steps else "(not classified)"
        if expected and expected.get(name):
            hit = expected[name] == label
            agree += hit
            label += "" if hit else f" (picked as {expected[name]})"
        route = next((r for r in ROUTES if r in steps), None)
        if route is None:
            print(f"{name[:44]:44} {label:30} {'(failed)':24}")
            continue
        row = steps[route]
        md = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        compute = md.get("local_compute")
        if compute:
            cost = f"{compute['hours']:.4f} h on {hardware_label(compute['hardware'])} ({compute['device']})"
        else:
            cost = f"${md.get('cost') or 0:.5f}"
        ok = "" if row["was_successful"] else " FAILED"
        print(f"{name[:44]:44} {label[:30]:30} {ROUTES[route]:24} {md.get('duration', 0):8.1f}  {cost}{ok}")
    if expected:
        print(f"\nclassifier agreed with the picked kind for {agree} of {len(by_image)} images")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", default=str(ROOT / "inputs" / "lux_routing"), help="directory of images")
    ap.add_argument("--log", default=str(ROOT / "results" / "lux_routing" / "run.jsonl"), help="the run's JSON-Lines file")
    ap.add_argument("--expected", help="CSV with `file` and `expected` columns to compare the labels with")
    ap.add_argument("--report-only", action="store_true", help="report on an existing run without running")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    if not args.report_only:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        wf = build(args.images, args.log)
        wf.run()
        usage = wf.usage.summary()
        print(f"\nhosted: ${usage['cost']:.4f} for {usage['calls'] - usage['local_calls']} calls")
        for machine, row in usage["by_hardware"].items():
            print(f"local:  {row['local_hours']:.4f} h for {row['local_calls']} calls on {machine} "
                  f"({', '.join(row['devices']) or 'local server'})")
    expected = None
    if args.expected:
        with open(args.expected, newline="") as fh:
            expected = {r["file"]: r["expected"] for r in csv.DictReader(fh)}
    report(args.log, expected)


if __name__ == "__main__":
    main()
