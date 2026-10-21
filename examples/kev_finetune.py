"""Run a kev fine-tune through a chai workflow, on decisions in system1-finetune-bench's format.

A kev fine-tune exported by system1-finetune-bench (``s1bench.export``: ``adapter_config.json``,
``adapter_model.safetensors``, ``head.pt``) runs in process through chai's ``pointer`` System One family
(``chai.system_one``), from the Hugging Face Hub or from a local folder:

    python examples/kev_finetune.py DATA.jsonl
    python examples/kev_finetune.py DATA.jsonl --model pointer//path/to/export --limit 50
    python examples/kev_finetune.py test.jsonl --compare runs/<run>/predictions-tuned@<dataset>.jsonl

DATA.jsonl holds one choice decision per line, as the bench writes them::

    {"id": ..., "type": "choice", "state": "...", "question": "Which record is ...?",
     "options": [{"key": "wikidata:Q42", "text": "..."}, {"key": "none", "text": "None of ..."}],
     "answer": "wikidata:Q42"}

The workflow is a provider of those records, an Iterator, and one ``DecisionClassifier`` whose state,
question and options are read from each record -- the same wiring a reconciliation workflow uses with
candidates it retrieved itself. It prints how often the model's choice matches ``answer`` and, with
``--compare``, how closely chai's probabilities match the ones the bench recorded for the same
checkpoint. The base model (Qwen3.5-4B-Base, ~9 GB) is downloaded on first use unless cached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

DEFAULT_MODEL = "kev-4b-europeana-persons"  # a preset; "pointer/<hub repo or folder>" names any kev-layout export


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("type", "choice") == "choice":
                    rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def build_workflow(model: str, **settings) -> dict:
    return {
        "type": "Workflow",
        "id": "kev_wf",
        "steps": [
            {
                "type": "iterator.Iterator",
                "id": "decisions",
                "settings": {"workers": 1},  # one local model: forward passes run one at a time anyway
                "steps": [
                    {
                        "type": "decision.DecisionClassifier",
                        "id": "decide",
                        "settings": {
                            "model": model,
                            "question": "{question}",
                            "state": "{state}",
                            "options_from": "options",
                            "option_key": "{key}",
                            "option_text": "{text}",
                            "layout": "described",  # how the bench trains kev: key: description
                            **settings,
                        },
                    }
                ],
            }
        ],
    }


def run(rows, model=DEFAULT_MODEL, **settings):
    """``(decisions, usage)``: per row its ``choice`` and ``probabilities``, and the run's usage."""
    from chai.result import ItemResult, ListResult
    from chai.workflow import Workflow

    wf = Workflow(build_workflow(model, **settings))
    items = ListResult([ItemResult(r, metadata={"type": "DATA"}) for r in rows])
    entries = wf.run(items).value[0].value
    decisions = []
    for row, entry in zip(rows, entries):
        label = entry.value[0]
        decisions.append({"id": row.get("id"), "answer": row.get("answer"), "choice": label.metadata["choice"],
                          "label": label.value[0], "probabilities": label.metadata["probabilities"]})
    return decisions, wf.usage_summary()


def compare(decisions, predictions_path):
    """Agreement and largest probability difference against predictions the bench recorded."""
    recorded = {}
    with open(predictions_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                p = json.loads(line)
                recorded[p["id"]] = p["probs"]
    same, diffs, matched = 0, [], 0
    for d in decisions:
        ref = recorded.get(d["id"])
        if not ref:
            continue
        matched += 1
        same += max(ref, key=ref.get) == d["choice"]
        diffs.extend(abs(ref[k] - d["probabilities"].get(k, 0.0)) for k in ref)
    return {"matched": matched, "same_choice": same, "max_diff": max(diffs) if diffs else None,
            "mean_diff": sum(diffs) / len(diffs) if diffs else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="JSONL of choice decisions in the bench's format")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"kev model id (default {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, default=0, help="only the first N decisions (default all)")
    ap.add_argument("--compare", default=None, help="the bench's predictions JSONL for the same checkpoint")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the first available)")
    args = ap.parse_args(argv)

    rows = read_jsonl(args.data, args.limit)
    settings = {"device": args.device} if args.device else {}
    t = time.time()
    decisions, usage = run(rows, args.model, **settings)
    seconds = time.time() - t

    agree = sum(d["choice"] == d["answer"] for d in decisions)
    print(f"{len(decisions)} decisions by {args.model} in {seconds:.0f}s (including loading the model)")
    print(f"choice matches the answer: {agree}/{len(decisions)} ({100 * agree / max(len(decisions), 1):.1f}%)")
    nones = [d for d in decisions if d["answer"] == "none"]
    if nones:
        print(f"  of which 'none' answers recovered: {sum(d['choice'] == 'none' for d in nones)}/{len(nones)}")
    print(f"usage: {usage['calls']} calls, {usage['total_tokens']} tokens read, cost ${usage['cost']:.2f}")
    if args.compare:
        c = compare(decisions, args.compare)
        print(f"vs the bench's recorded predictions: same choice {c['same_choice']}/{c['matched']}, "
              f"probability difference mean {c['mean_diff']:.4f}, max {c['max_diff']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
