"""Compare System One decision models side by side in one chai workflow.

Two tasks, every available model on both:

1. **Reconciliation (choice).** Five mentions from one sentence, each with the candidate records
   Wikidata's search returns for it (fetched once and stored below, so this runs offline) plus a
   "none" option. The right answer for "Theo" is none: his brother Theo van Gogh is not among the
   hits, but Theo van Doesburg, a Dutch painter, is.
2. **Triage (noul + score).** Two short texts, each asked "does it mention a painter?" (noul) and
   "how much is it about art?" (score, 1-3), in one ``DecisionExtractor`` per model.

Each model gets the layout it does best with -- one setting -- and the workflow's usage ledger
reports calls, time and cost per model.

    python examples/system1_models.py                      # every model available here
    python examples/system1_models.py --models laya,gliner

Models: ``laya`` and ``gliner`` (in process; ``pip install laya`` and ``gliner2[local]``), ``von``,
``bekko`` and ``decider-2b`` (in process, used when their weights are already cached), the Yale
fine-tunes ``kev-ft`` (kev-4b on Europeana persons) and ``gliner-ft`` (GLiNER2.5-Decide on Europeana
persons), ``jev-omni`` (in process, ~48 GB of weights, ~25 GB of accelerator memory; used when
cached), ``jev`` (TypeSafe API; needs ``TYPESAFE_API_KEY``) and ``gemini`` (a generative baseline;
needs ``GOOGLE_CLOUD_PROJECT`` or ``GEMINI_API_KEY``). A model that is not available is skipped with
the reason. ``--offline`` loads local weights from the Hugging Face cache only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

TEXT = (
    "The Night Café was painted by Vincent van Gogh in Arles in 1888. Van Gogh wrote to his brother "
    "Theo about it. It is now at the Yale University Art Gallery."
)

NONE_TEXT = "None of the listed records is the entity referred to in the text"
QUESTION = "Which candidate record is the mention '{name}' ({type}) referring to in the text? Choose none if no listed record fits."

# wbsearchentities hits for each mention (limit 5), fetched 2026-10-19; "gold" is the right record.
MENTIONS = [
    {
        "name": "Van Gogh", "type": "person", "gold": "wikidata:Q5582",
        "candidates": [
            {"key": "wikidata:Q5582", "summary": "Vincent van Gogh — Dutch painter (1853–1890) (aka Van Gogh)"},
            {"key": "wikidata:Q42865365", "summary": "van Gogh — family name"},
            {"key": "wikidata:Q17029510", "summary": "Van Gogh — music album by the band Van Gogh"},
            {"key": "wikidata:Q84161251", "summary": "Van Gogh National Park — national park in the Netherlands"},
            {"key": "wikidata:Q119625610", "summary": "Van Gogh — family name"},
        ],
    },
    {
        "name": "Arles", "type": "place", "gold": "wikidata:Q48292",
        "candidates": [
            {"key": "wikidata:Q48292", "summary": "Arles — commune in Bouches-du-Rhône, France"},
            {"key": "wikidata:Q7897", "summary": "Arles-sur-Tech — commune in Pyrénées-Orientales, France"},
            {"key": "wikidata:Q75135296", "summary": "Arles — family name"},
            {"key": "wikidata:Q581647", "summary": "Arlesheim — municipality in the canton of Basel-Land in Switzerland"},
            {"key": "wikidata:Q3271735", "summary": "Kingdom of Arles — Kingdom in Europe active during the Early Middle Ages (aka Arles)"},
        ],
    },
    {
        "name": "Theo", "type": "person", "gold": "none",
        "candidates": [
            {"key": "wikidata:Q7062902", "summary": "Theo — male given name"},
            {"key": "wikidata:Q1165126", "summary": "Theodor — male given name"},
            {"key": "wikidata:Q19914934", "summary": "Théo — male given name"},
            {"key": "wikidata:Q115670193", "summary": "THEO — German musician"},
            {"key": "wikidata:Q160422", "summary": "Theo van Doesburg — Dutch architect, painter, draughtsman and writer (1883–1931)"},
        ],
    },
    {
        "name": "The Night Café", "type": "work", "gold": "wikidata:Q674846",
        "candidates": [
            {"key": "wikidata:Q674846", "summary": "The Night Café — painting by Vincent van Gogh (Yale University Art Gallery)"},
            {"key": "wikidata:Q20489268", "summary": "The night café — painting by Brett Whiteley"},
            {"key": "wikidata:Q85808007", "summary": "The Night Café — British indie pop band"},
            {"key": "wikidata:Q106603505", "summary": "The Night Café — several paintings by Vincent Van Gogh"},
            {"key": "wikidata:Q780986", "summary": "The Night Café in Arles — painting by Paul Gauguin"},
        ],
    },
    {
        "name": "Yale University Art Gallery", "type": "organization", "gold": "wikidata:Q1568434",
        "candidates": [
            {"key": "wikidata:Q1568434", "summary": "Yale University Art Gallery — art museum in New Haven, Connecticut"},
            {"key": "wikidata:Q115045393", "summary": "Yale University Art Gallery — 2003–2005 Exhibition at Metropolitan Museum"},
            {"key": "wikidata:Q99005775", "summary": "Yale University Art Gallery Bulletin — annual compilation of themed essays and Gallery news"},
            {"key": "wikidata:Q99082689", "summary": "Yale University Art Gallery Selections — academic work"},
            {"key": "wikidata:Q131342348", "summary": "Yale University Art Gallery Swartwout Building — museum building in New Haven"},
        ],
    },
]

TRIAGE_TEXTS = [
    {"id": "art", "text": TEXT},
    {"id": "sport", "text": "Liverpool beat Everton 2-1 at Anfield on Saturday; Salah scored the winner in the 89th minute."},
]
TRIAGE_QUESTIONS = {
    "painter": {"type": "noul", "question": "The text mentions a painter."},
    "about_art": {"type": "score", "question": "How much is the text about art?", "options": {"1": "not at all", "2": "partly", "3": "entirely"}},
}

# Each model's best-measured input layout (see docs/system1.md). GLiNER answers without the
# question's instructions, so the mention has to be in the state; Laya does best with the
# candidates listed in the state and no extra header lines. The Yale fine-tunes expect the layout
# they were trained on: a Mention / Entity type / Text header.
HEADER = "Mention: {name}\nEntity type: {type}\nText: {context}"
MODELS = {
    "laya": {"model": "laya/convaiinnovations/laya", "layout": "in_state", "state": "Text: {context}"},
    "gliner": {"model": "gliner/fastino/GLiNER2.5-Decide", "layout": "described", "state": HEADER},
    "von": {"model": "von", "layout": "described", "state": "Text: {context}"},
    "bekko": {"model": "bekko-400m", "layout": "described", "state": "Text: {context}"},
    "decider-2b": {"model": "decider-2b", "layout": "described", "state": "Text: {context}"},
    "kev-ft": {"model": "kev-4b-europeana-persons", "layout": "described", "state": HEADER},
    "gliner-ft": {"model": "gliner-decide-europeana-persons", "layout": "described", "state": HEADER},
    "jev-omni": {"model": "jev-omni/akhilaaa3/Jev-Omni", "layout": "described", "state": "Text: {context}"},
    "jev": {"model": "jev/jev-latest", "layout": "in_state", "state": "Text: {context}"},
    "gemini": {"model": "gemini-3.1-flash-lite", "layout": "in_state", "state": "Text: {context}"},
}

#: What a local model needs in the Hugging Face cache before it can run here.
CACHED_FILES = {
    "von": ("wfzyx/von", "option_marker.pt"),
    "bekko": ("hotchpotch/bekko-system-one-v0-400m", "inference_v0.py"),
    "decider-2b": ("Mapika/decider-2b", "config.json"),
    "kev-ft": ("yale-cultural-heritage/kev-4b-europeana-persons", "head.pt"),
    "gliner-ft": ("yale-cultural-heritage/GLiNER2.5-Decide-europeana-persons", "model.safetensors"),
}


def _cached(repo: str, filename: str, revision: str = None) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo, filename, revision=revision)
    return isinstance(path, str) and os.path.exists(path)


def _preset_cached(name: str) -> bool:
    """Whether the weights a preset pins (its revision, when it has one) are in the cache."""
    from chai.system_one import PRESETS

    repo, filename = CACHED_FILES[name]
    model = MODELS[name]["model"]
    revision = PRESETS.get(model, {}).get("revision") if PRESETS.get(model, {}).get("checkpoint") == repo else None
    return _cached(repo, filename, revision)


def availability(name: str):
    """``(True, "")`` when *name* can run here, else ``(False, reason)``."""
    if name == "laya":
        if not importlib.util.find_spec("laya"):
            return False, "pip install laya"
        return True, ""
    if name in ("gliner", "gliner-ft"):
        if not importlib.util.find_spec("gliner2"):
            return False, "pip install 'gliner2[local]'"
        if name in CACHED_FILES and not _preset_cached(name):
            return False, "weights not in the Hugging Face cache"
        return True, ""
    if name == "bekko" and not importlib.util.find_spec("sentence_transformers"):
        return False, "pip install sentence-transformers (bekko's own inference code needs it)"
    if name in CACHED_FILES:
        if not _preset_cached(name):
            return False, "weights not in the Hugging Face cache (use --offline only once they are)"
        return True, ""
    if name == "jev-omni":
        if not _cached("akhilaaa3/Jev-Omni", "head.pt"):
            return False, "weights not in the Hugging Face cache (~48 GB download)"
        return True, ""
    if name == "jev":
        if not (os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")):
            return False, "set TYPESAFE_API_KEY"
        return True, ""
    if name == "gemini":
        key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not os.getenv("GOOGLE_CLOUD_PROJECT") and (not key or key == "api_key_here"):
            return False, "set GOOGLE_CLOUD_PROJECT or GEMINI_API_KEY"
        return True, ""
    return False, "unknown model"


def selector(name: str) -> dict:
    preset = MODELS[name]
    return {
        "type": "decision.DecisionClassifier",
        "id": f"select_{name}",
        "name": f"Select record ({name})",
        "settings": {
            "model": preset["model"],
            "question": QUESTION,
            "state": preset["state"],
            "options_from": "candidates",
            "option_text": "{summary}",
            "none_option": NONE_TEXT,
            "layout": preset["layout"],
            "options_heading": "Candidate records for the mention '{name}' ({type}):",
        },
    }


def triage(name: str) -> dict:
    return {
        "type": "decision.DecisionExtractor",
        "id": f"triage_{name}",
        "name": f"Triage ({name})",
        "settings": {"model": MODELS[name]["model"], "state": "{text}", "questions": TRIAGE_QUESTIONS},
    }


def build_workflow(names) -> dict:
    """Two providers, each feeding an Iterator whose steps run every model on the same entry."""
    records = [{**m, "context": TEXT} for m in MENTIONS]
    return {
        "type": "Workflow",
        "id": "system1_compare",
        "name": "System One models side by side",
        "steps": [
            {
                "type": "provider.StaticProvider",
                "id": "mentions",
                "input": "",
                "settings": {"values": records},
                "steps": [{"type": "iterator.Iterator", "id": "each_mention", "steps": [selector(n) for n in names]}],
            },
            {
                "type": "provider.StaticProvider",
                "id": "texts",
                "input": "",
                "settings": {"values": TRIAGE_TEXTS},
                "steps": [{"type": "iterator.Iterator", "id": "each_text", "steps": [triage(n) for n in names]}],
            },
        ],
    }


def run(names):
    """Run the comparison; returns ``(rows, triage_rows, usage)`` for printing or testing."""
    from chai.workflow import Workflow

    wf = Workflow(build_workflow(names))
    t0 = time.time()
    res = wf.run()
    seconds = time.time() - t0
    mention_entries = res.value[0].value[0].value  # provider -> its iterator -> one list per mention
    text_entries = res.value[1].value[0].value
    rows = []
    for mention, entry in zip(MENTIONS, mention_entries):
        picks = {}
        for name, label in zip(names, entry.value):
            choice = label.value[0]
            picks[name] = {"choice": choice, "p": label.metadata["probabilities"].get(label.metadata.get("choice"), 0.0), "correct": choice == mention["gold"]}
        rows.append({"name": mention["name"], "gold": mention["gold"], "picks": picks})
    triage_rows = []
    for text, entry in zip(TRIAGE_TEXTS, text_entries):
        triage_rows.append({"id": text["id"], "answers": {name: out.value for name, out in zip(names, entry.value)}})
    usage = wf.usage_summary()
    usage["wall_seconds"] = seconds
    return rows, triage_rows, usage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated: " + ", ".join(MODELS))
    ap.add_argument("--offline", action="store_true", help="load local weights from the Hugging Face cache only")
    args = ap.parse_args(argv)
    if args.offline:
        import huggingface_hub.constants

        huggingface_hub.constants.HF_HUB_OFFLINE = True

    import chai.core  # noqa: F401 -- loads .env before the availability checks read it

    names = []
    for name in [n.strip() for n in args.models.split(",") if n.strip()]:
        ok, why = availability(name)
        if ok:
            names.append(name)
        else:
            print(f"skipping {name}: {why}")
    if not names:
        print("no model available")
        return 1

    rows, triage_rows, usage = run(names)

    width = 13
    short = lambda key: "none" if key == "none" else key.replace("wikidata:", "")  # noqa: E731
    print(f"\n{TEXT}\n")
    print(f"{'mention':28s}{'gold':12s}" + "".join(f"{n:>{width}s}" for n in names))
    for row in rows:
        cells = "".join(f"{(short(p['choice']) + (' ✓' if p['correct'] else ' ✗')):>{width}s}" for p in row["picks"].values())
        print(f"{row['name']:28s}{short(row['gold']):12s}{cells}")
    totals = {n: sum(r["picks"][n]["correct"] for r in rows) for n in names}
    print(f"{'correct':40s}" + "".join(f"{f'{totals[n]}/{len(rows)}':>{width}s}" for n in names))

    print("\nP(mentions a painter) / expected 'about art' (1-3):")
    print(f"{'text':28s}" + "".join(f"{n:>{width + 2}s}" for n in names))
    for row in triage_rows:
        print(f"{row['id']:28s}" + "".join(f"{c:>{width + 2}s}" for c in [f"{a['painter']['probability']:.2f} / {a['about_art']['expected']:.1f}" for a in row['answers'].values()]))

    print("\nper model: calls, seconds, cost")
    for name in names:
        calls = secs = 0
        cost, unpriced = 0.0, 0
        for cid in (f"select_{name}", f"triage_{name}"):
            c = usage["by_component"].get(cid, {})
            calls += c.get("calls", 0)
            secs += c.get("seconds", 0.0)
            cost += c.get("cost", 0.0)
            unpriced += c.get("unpriced_calls", 0)
        price = "unpriced" if unpriced else f"${cost:.5f}"
        print(f"  {name:10s} {calls:3d} calls  {secs:6.1f} s  {price}")
    print(f"\nwall time {usage['wall_seconds']:.1f} s (includes loading local models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
