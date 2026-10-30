"""Run examples/workflow-shadow-puppets.json on photographs of shadow puppets and print their sizes.

Each photograph shows a puppet on grey card with a 92 cm yardstick standing up one side and a 61 cm
ruler along the bottom. A YOLO11 detector trained on the collection
(``yale-cultural-heritage/shadow-puppets-yolov11`` on Hugging Face, five sizes, run on this machine)
finds the puppet with and without its rods, its shoulders and the two rulers; ``ScaleMeasurer``
reads the rulers as references of known size, so every other box gets a width and height in cm.
This is the shadow-puppets project's Roboflow detector and ``MeasurementCalculator`` as chai
components, with no hosted model.

Every region is a CHAI-TEA RESULT row in ``results/shadow_puppets/run.jsonl``: its box is its
``input_segment`` (``{"bbox": [x1, y1, x2, y2], "size": [w, h]}`` in the upright photograph's
pixels), its size and the detector's confidence are in its ``value``, and the scale the rulers gave
is in its ``metadata``. Nothing is drawn; this script reads the rows back and prints, per
photograph, the scale and each region's size.

    python examples/shadow_puppets.py                                  # inputs/shadow_puppets
    python examples/shadow_puppets.py photo1.JPG photo2.JPG --size x
"""

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chai.workflow import Workflow  # noqa: E402

WORKFLOW = ROOT / "examples" / "workflow-shadow-puppets.json"
MODEL = "hf://yale-cultural-heritage/shadow-puppets-yolov11/shadow-puppets-yolo11{size}.pt"


def find_step(tree, cid):
    if tree.get("id") == cid:
        return tree
    for key in ("steps", "next_steps", "true_steps", "false_steps"):
        for child in tree.get(key, []):
            found = find_step(child, cid)
            if found:
                return found
    return None


def gather(images, directory):
    """A directory of the photographs: *images* itself, or links to the files named in it."""
    if len(images) == 1 and os.path.isdir(images[0]):
        return images[0]
    os.makedirs(directory, exist_ok=True)
    for old in os.listdir(directory):
        os.remove(os.path.join(directory, old))
    for path in images:
        os.symlink(os.path.abspath(path), os.path.join(directory, os.path.basename(path)))
    return directory


def build(images, log, size=None):
    tree = json.loads(WORKFLOW.read_text())
    find_step(tree, "photographs")["input"] = str(images)
    find_step(tree, "log_region")["settings"]["file"] = str(log)
    if size:
        find_step(tree, "detect")["settings"]["model"] = MODEL.format(size=size)
    return Workflow(tree)


def loads(value):
    return json.loads(value) if isinstance(value, str) else value


def report(log):
    with open(log) as fh:
        rows = [json.loads(line) for line in fh]
    by_photo = {}
    for row in rows:
        by_photo.setdefault(os.path.basename(row["input"] or "?"), []).append(row)
    unmeasured = 0
    for name in sorted(by_photo):
        regions = sorted(by_photo[name], key=lambda r: r["input_sequence"] or 0)
        print(f"\n{name}")
        scale = (loads(regions[0]["metadata"]) or {}).get("scale") or {}
        if scale.get("x") and scale.get("y"):
            print(f"  scale: {scale['x']:.2f} px/cm across, {scale['y']:.2f} px/cm down "
                  f"(rulers disagree by {abs(scale['x'] - scale['y']) / max(scale.values()):.1%})")
        for r in regions:
            value, md = loads(r["value"]), loads(r["metadata"]) or {}
            box = loads(r["input_segment"])[0]["bbox"]
            if md.get("measured"):
                size = f"{value['width']:6.1f} x {value['height']:6.1f} cm"
            else:
                size, unmeasured = "   (no ruler: not measured)", unmeasured + 1
            print(f"  {value['label']:18} {size}   conf {value.get('confidence', 0):.2f}   box {[round(v) for v in box]}")
    print(f"\n{len(by_photo)} photographs, {len(rows)} regions, {unmeasured} not measured")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="*", default=[str(ROOT / "inputs" / "shadow_puppets")],
                        help="a directory of photographs, or photographs")
    parser.add_argument("--log", default=str(ROOT / "results" / "shadow_puppets" / "run.jsonl"))
    parser.add_argument("--size", choices=list("nsmlx"), help="detector size (default: the workflow's, s)")
    parser.add_argument("--report-only", action="store_true", help="print the rows of an earlier run")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    if not args.report_only:
        images = gather(args.images, str(pathlib.Path(args.log).parent / "photographs"))
        start = time.time()
        build(images, args.log, args.size).run()
        print(f"ran in {time.time() - start:.1f}s")
    report(args.log)


if __name__ == "__main__":
    main()
