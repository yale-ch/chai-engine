# System One decision models in chai

A **System One** model answers a typed question about a *state* (text) with a probability per
option, in one forward pass -- no generation, no JSON to parse, calibrated by construction. Chai
supports them natively: the data model and providers live in `chai/ai/system1.py`, and three roles
in `chai/decision.py` put them into workflows.

## Question types

| Type | Asks | `options` | Answer |
|---|---|---|---|
| `choice` | pick one option | `{key: description}` (or a list) | `choice`, `probabilities`, `confidence` |
| `score` | rate on an ordinal scale | the levels, low to high | `score` (level key), `expected` (probability-weighted level), `probabilities`, `confidence` |
| `noul` | is this statement true? | none | `truth`, `probability` (P(true)), `probabilities`, `confidence` |

These are the same three types, in the same `{key, text}` option shape, as the
`system1-finetune-bench` training format, so what a workflow asks can be exported as training data.

## Models

The `model` setting picks the provider by prefix, the way LiteLLM ids do for chat models:

| `model` | Runs | Needs |
|---|---|---|
| `jev/jev-latest` | TypeSafe's Jev, `POST /v1/systemone`, all questions in one call | `TYPESAFE_API_KEY` (or `api_key`), optional `api_base` |
| `laya/convaiinnovations/laya` | Laya, in process (about 850 MB on first use) | `pip install laya`; settings `subfolder`, `device` |
| `gliner/fastino/GLiNER2.5-Decide` | GLiNER2.5-Decide, 340M encoder classifier, in process | `pip install gliner2`; settings `device`, `use_instruction` |
| any LiteLLM chat id, e.g. `gemini-3.1-flash-lite`, `openai/gpt-4o-mini`, `ollama_chat/llama3.2` | a generative baseline asked for the same JSON | that provider's credentials |

The generative baseline exists so a decision model can be compared like for like; its
probabilities are self-reported, not calibrated. Local models cost 0; Jev's price is unknown to
chai (`cost: null`, counted as unpriced in the ledger) unless you set `pricing`.

Models that only classify (GLiNER) answer `score` as a choice over the levels and `noul` as a
choice over true/false, as the bench's `as_choice` approaches do.

## Layouts

`layout` is the bench's inference-time prompt approach: `described` (default -- option
descriptions on the options), `in_state` (descriptions listed in the state under `options_heading`,
options become bare keys: Laya's best layout on reconciliation, GLiNER's worst), `bare` (keys
only). The layout a model scored best with is one setting away, and the state and question
actually sent are stored on the result (`metadata.decision`).

## The roles

### DecisionClassifier -- one question, the answer as a label

Reconciliation: an entity record with retrieved `candidates` comes in; the label is the chosen
record key, or `none`.

```json
{"type": "decision.DecisionClassifier", "id": "select",
 "register_on": ["entities"],
 "settings": {
   "model": "laya/convaiinnovations/laya",
   "question": "Which candidate record is the mention '{name}' referring to in the text?",
   "state": "Text: {context_text}",
   "options_from": "candidates", "option_key": "{key}", "option_text": "{summary}",
   "none_option": "None of the listed records is the entity referred to in the text",
   "layout": "in_state", "options_heading": "Candidate records for the mention '{name}':",
   "max_options": 20}}
```

Output: `LabelListResult(["wikidata:Q5582"])` with `metadata.probabilities` (every option),
`confidence`, `choice`, and `decision` (state, question, canonical answer). `threshold` makes the
label `none` when the top probability is below it (the reconciler's `null_mode: threshold`);
`none_option` adds an option the model may pick (`null_mode: explicit`). For `question_type:
noul` the label is `true`/`false` (threshold 0.5 on P(true)); for `score` it is the level key, with
`metadata.expected` as the probability-weighted level.

Wire a `LabelTestGate` (`label: ["none"]`) or `SwitchGate` on the label, or a `ThresholdGate` on
`metadata.confidence`, to route what happens next.

### DecisionGate -- a noul question as a branch

```json
{"type": "decision.DecisionGate", "id": "is_person",
 "settings": {"model": "jev/jev-latest", "question": "The mention '{name}' refers to a person.", "threshold": 0.7},
 "true_steps": [...], "false_steps": [...]}
```

Gates are guarantees: the true branch runs only when the model puts at least `threshold` on the
statement. The answer is appended to the input result's `extra["decisions"]` and the probability is
on the gate's `component_end` event.

### DecisionExtractor -- several questions, one record

```json
{"type": "decision.DecisionExtractor", "id": "triage",
 "settings": {"model": "jev/jev-latest",
   "questions": {
     "team":    {"type": "choice", "question": "Which team should handle this ticket?", "options": {"billing": "payments, invoices, refunds", "tech": "bugs, outages", "none": "none of these"}},
     "urgent":  {"type": "noul",   "question": "The customer is blocked right now."},
     "tone":    {"type": "score",  "question": "How angry is the customer?", "options": ["1", "2", "3", "4", "5"]}}}}
```

Output: `ItemResult({"team": {...}, "urgent": {...}, "tone": {...}})`. Jev answers all questions in
one call; the other providers loop.

## A complete example

`examples/reconcile_wikidata.py` runs the whole reconciliation shape: an LLM extractor finds the
entities, plain Python (outside chai) searches Wikidata for each name, and a `DecisionClassifier`
asks the model which candidate the mention refers to -- candidates listed in the state, a "none"
option added -- then prints each entity's chosen record, its probability and what the run cost.
`tests/test_example_reconcile.py` runs the same functions offline.

## Cost and usage

Every decision result carries `model`, `token_usage` and `cost`, so the workflow's usage ledger
(`workflow.usage_summary()`) counts decisions alongside LLM calls.

## Moving from chai-reconciler's selectors

| reconciler setting | chai |
|---|---|
| `LayaSelector` / `JevSelector` / `GlinerDecideSelector` / `GeminiSelector` | `decision.DecisionClassifier` with `model: laya/...`, `jev/...`, `gliner/...`, `gemini-...` |
| `question_format: state` / `criteria` | `layout: in_state` / `described` |
| `null_mode: explicit` / `threshold`, `null_threshold` | `none_option: "..."` / `threshold: 0.5` |
| `max_candidates` | `max_options` |
| `selector_name` | the component `id` |
| `candidates[].summary` as option text | `options_from: candidates`, `option_text: "{summary}"` |

Jev-Omni (`jev_omni_runtime.py`) has no provider yet; see `docs/roadmap.md`.
