"""Reconcile the named entities in a text against Wikidata.

Three stages -- two chai workflows with plain Python between them:

1. **NER** (chai): an LLM extractor finds every named entity, grouping the mentions of one
   referent, and returns JSON.
2. **Candidates** (plain Python, outside chai): each entity's name is searched on Wikidata with
   ``wbsearchentities`` and every hit becomes an option record ``{key: "wikidata:Q...", summary}``.
3. **Selection** (chai): a ``DecisionClassifier`` asks the LLM which candidate the mention refers
   to -- the candidates listed in the state, a "none of these" option added -- and labels each
   entity with the chosen record key. The answer's probabilities and confidence ride in the
   result metadata, and every model call is priced into the workflow's usage ledger.

Run it::

    python examples/reconcile_wikidata.py                     # the sample text
    python examples/reconcile_wikidata.py "Ada Lovelace met Charles Babbage in London."
    CHAI_MODEL=openai/gpt-4o-mini python examples/reconcile_wikidata.py

Needs the model's credentials in ``.env`` (``GOOGLE_CLOUD_PROJECT`` for Vertex AI, or
``GEMINI_API_KEY``; any LiteLLM model id works through ``CHAI_MODEL``) and network access to
wikidata.org. ``tests/test_example_reconcile.py`` runs the same functions offline.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

import requests

from chai.result import ItemResult, ListResult
from chai.workflow import Workflow

DEFAULT_MODEL = os.environ.get("CHAI_MODEL", "gemini-3.1-flash-lite")
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "chai-engine example (https://github.com/yale-ch/chai-engine)"

SAMPLE_TEXT = (
    "The Night Café was painted by Vincent van Gogh in Arles in 1888. "
    "Van Gogh wrote to his brother Theo about it. It is now at the Yale University Art Gallery."
)

NER_PROMPT = """Find every named person, place, organization and work in the text below. Group all mentions that
refer to the same real-world entity into ONE entity, copy each mention exactly as it is written, and
give a one-phrase gloss of who or what the entity is, using only what the text says.

Return JSON only, of this shape:
{"entities": [{"name": "<canonical name as written in the text>", "type": "<person|place|organization|work>",
"mentions": ["<exact surface string>", ...], "gloss": "<one short phrase from the text>"}]}

TEXT:
{text_input_0}"""

SELECT_QUESTION = "Which Wikidata record is the {type} '{name}' ({gloss}) referring to in the text?"
NONE_OPTION = "None of the listed records is the entity referred to in the text"


# ----------------------------------------------------------------------------- 1. NER (chai)


def extractor_type(model: str) -> str:
    """A bare Gemini id goes through the Gemini preset (credentials, Vertex); anything with a
    LiteLLM provider prefix goes through the generic LLM extractor."""
    return "extractor.LLMExtractor" if "/" in model else "extractor.GeminiExtractor"


def ner_workflow(model: str = DEFAULT_MODEL, **settings) -> Workflow:
    base = {"model": model, "prompt": NER_PROMPT, "expected_output": "json", "retries": 1}
    if "/" in model:
        base["json_mode"] = True
    else:
        base.update({"tools": [], "response_mime_type": "application/json"})
    return Workflow(
        {
            "type": "Workflow",
            "id": "ner_wf",
            "steps": [{"type": extractor_type(model), "id": "ner", "name": "Named entities", "settings": {**base, **settings}}],
        }
    )


def extract_entities(text: str, model: str = DEFAULT_MODEL, **settings):
    """``(entities, usage)``: the LLM's entity records and the NER workflow's usage summary."""
    wf = ner_workflow(model, **settings)
    out = wf.run(text).value[0].value
    entities = out.get("entities", []) if isinstance(out, dict) else []
    cleaned = []
    for ent in entities:
        if not isinstance(ent, dict) or not str(ent.get("name", "")).strip():
            continue
        cleaned.append(
            {
                "name": str(ent["name"]).strip(),
                "type": str(ent.get("type") or "entity").strip().lower(),
                "gloss": str(ent.get("gloss") or "").strip(),
                "mentions": [str(m) for m in (ent.get("mentions") or [ent["name"]])],
            }
        )
    return cleaned, wf.usage_summary()


# ----------------------------------------------------------------------------- 2. Wikidata (plain Python)


def wikidata_candidates(name: str, limit: int = 8, language: str = "en", session=None) -> List[dict]:
    """Search Wikidata for *name*; one option record per hit.

    Each record has ``key`` (``wikidata:Q...``), ``label``, ``description``, ``url`` and ``summary``
    -- the one line the model reads: ``label — description (aka alias, ...)``.
    """
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": int(limit),
        "format": "json",
    }
    resp = (session or requests).get(WIKIDATA_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    records = []
    for hit in resp.json().get("search", []):
        qid = hit.get("id")
        if not qid:
            continue
        label = hit.get("label") or qid
        description = hit.get("description") or ""
        summary = f"{label} — {description}" if description else label
        aliases = [a for a in (hit.get("aliases") or []) if a and a != label]
        if aliases:
            summary += " (aka " + ", ".join(aliases[:3]) + ")"
        records.append(
            {
                "key": f"wikidata:{qid}",
                "label": label,
                "description": description,
                "url": hit.get("concepturi") or f"https://www.wikidata.org/wiki/{qid}",
                "summary": summary,
            }
        )
    return records


# ----------------------------------------------------------------------------- 3. Selection (chai)


def select_workflow(model: str = DEFAULT_MODEL, workers: int = 4, **settings) -> Workflow:
    return Workflow(
        {
            "type": "Workflow",
            "id": "select_wf",
            "steps": [
                {
                    "type": "iterator.Iterator",
                    "id": "entities",
                    "name": "For each entity",
                    "settings": {"workers": workers, "continue_on_error": True},
                    "steps": [
                        {
                            "type": "decision.DecisionClassifier",
                            "id": "select",
                            "name": "Select Wikidata record",
                            "register_on": ["entities"],
                            "settings": {
                                "model": model,
                                "question": SELECT_QUESTION,
                                "state": "Text: {context}",
                                "options_from": "candidates",
                                "option_key": "{key}",
                                "option_text": "{summary}",
                                "none_option": NONE_OPTION,
                                "layout": "in_state",
                                "options_heading": "Candidate Wikidata records for '{name}':",
                                "retries": 1,
                                **settings,
                            },
                        }
                    ],
                }
            ],
        }
    )


def select_records(records: List[dict], model: str = DEFAULT_MODEL, **settings):
    """``(selections, usage)``: each record with the model's ``choice`` (a candidate key or
    ``none``), its ``probabilities`` and ``confidence`` -- or an ``error`` when the entry failed."""
    wf = select_workflow(model, **settings)
    items = ListResult([ItemResult(rec, metadata={"type": "DATA"}) for rec in records], metadata={"type": "DATA"})
    entries = wf.run(items).value[0].value
    selections = []
    by_key = lambda rec: {c["key"]: c for c in rec.get("candidates", [])}  # noqa: E731
    for rec, entry in zip(records, entries):
        if isinstance(entry, ItemResult) and entry.metadata.get("type") == "ERROR":
            selections.append({**rec, "choice": None, "error": entry.metadata.get("error", entry.value)})
            continue
        label = entry.value[0]  # the DecisionClassifier's LabelListResult
        choice = label.value[0]
        selections.append(
            {
                **rec,
                "choice": choice,
                "record": by_key(rec).get(choice),
                "probabilities": label.metadata["probabilities"],
                "confidence": label.metadata["confidence"],
                "cost": label.metadata.get("cost"),
            }
        )
    return selections, wf.usage_summary()


# ----------------------------------------------------------------------------- all three


def reconcile(
    text: str,
    model: str = DEFAULT_MODEL,
    lookup: Callable[..., List[dict]] = wikidata_candidates,
    limit: int = 8,
    ner_settings: Optional[dict] = None,
    select_settings: Optional[dict] = None,
) -> dict:
    """NER, then candidates for every entity, then one decision per entity. Returns the entities
    with their selections plus the usage of both workflows."""
    entities, ner_usage = extract_entities(text, model, **(ner_settings or {}))
    records = []
    for ent in entities:
        records.append({**ent, "context": text, "candidates": lookup(ent["name"], limit)})
    selections, select_usage = select_records(records, model, **(select_settings or {}))
    cost = (ner_usage.get("cost") or 0.0) + (select_usage.get("cost") or 0.0)
    return {"text": text, "entities": selections, "usage": {"ner": ner_usage, "selection": select_usage, "cost": cost}}


def main(argv: List[str]) -> int:
    text = " ".join(argv[1:]).strip() or SAMPLE_TEXT
    out = reconcile(text)
    print(f"\n{text}\n")
    for ent in out["entities"]:
        head = f"{ent['name']} ({ent['type']})"
        if ent.get("error"):
            print(f"  {head:40s} ERROR {ent['error']}")
        elif ent["choice"] == "none" or ent["record"] is None:
            print(f"  {head:40s} -> none of {len(ent['candidates'])} candidates  (p={ent['confidence']:.2f})")
        else:
            rec = ent["record"]
            print(f"  {head:40s} -> {rec['key']}  {rec['summary'][:70]}  (p={ent['confidence']:.2f})")
    usage = out["usage"]
    calls = usage["ner"]["calls"] + usage["selection"]["calls"]
    tokens = usage["ner"]["total_tokens"] + usage["selection"]["total_tokens"]
    print(f"\n{calls} model calls, {tokens} tokens, ${usage['cost']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
