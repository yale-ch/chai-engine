# System One decision models in chai

A **System One** model answers a typed question about a *state* (text) with a probability per
option, in one forward pass -- no generation, no JSON to parse, calibrated by construction. Chai
supports them natively: the models live in the `chai/system_one/` package, one provider per
architecture family, and three roles in `chai/decision.py` put them into workflows.

## Question types

| Type | Asks | `options` | Answer |
|---|---|---|---|
| `choice` | pick one option | `{key: description}` (or a list) | `choice`, `probabilities`, `confidence` |
| `score` | rate on an ordinal scale | the levels, low to high | `score` (level key), `expected` (probability-weighted level), `probabilities`, `confidence` |
| `noul` | is this statement true? | none | `truth`, `probability` (P(true)), `probabilities`, `confidence` |

These are the same three types, in the same `{key, text}` option shape, as the
`system1-finetune-bench` training format, so what a workflow asks can be exported as training data.

## The models

Every architecture the bench covers is a family, with the prompt layout and readout copied from the
model's own code, so a model is asked exactly as its authors serve it:

| family (id prefix) | models | how it decides | package |
|---|---|---|---|
| `pointer` (alias `kev`) | kev-4b / 9b / 27b; Yale's `kev-4b-europeana-persons` | Qwen3.5 base + LoRA; a pointer head over the options' closing tokens | `peft` |
| `letters` | plumb-4b, JevK5-4b/9b, decider-2b/4b/35b, Tev1-4b | one prompt listing lettered options; the LM head's letter rows | |
| `candidate` | Open-Jev-9B / 27B | one chat prompt per option, a scalar head; softmax across options | `peft` |
| `nli` | openjev-nli-4b | a 3-way NLI cross-encoder, log P(entailment) per option | |
| `slots` | JEV-9B / 27B, GEV-26B | a bare template and a 24-slot head on the last token | `peft` |
| `laya` | convaiinnovations/laya | ModernBERT, one `[MASK]` per option; through the `laya` package | `laya` |
| `von` | wfzyx/von | ModernBERT, options packed and marked, order-invariant; model code vendored (Apache-2.0) | |
| `bekko` | hotchpotch/bekko-system-one-v0-400m | ModernBERT, each candidate a branch over the query; runs the repo's own inference code | `sentence-transformers` |
| `gliner` | fastino/GLiNER2.5-Decide; Yale's Europeana and Linked Art fine-tunes | DeBERTa classifier over described labels | `gliner2` |
| `jev-omni` | akhilaaa3/Jev-Omni | Gemma 4 12B with a 256-way option head (~48 GB of weights, ~25 GB of memory) | |
| `jev` | TypeSafe's hosted Jev | `POST /v1/systemone`, all questions in one call | `TYPESAFE_API_KEY` |
| *(any LiteLLM id)* | e.g. `gemini-3.1-flash-lite`, `openai/gpt-4o-mini` | a chat model asked for the same JSON: a baseline, not calibrated | that provider's key |

Models that only classify (GLiNER, Jev-Omni) answer `score` as a choice over the levels and `noul`
as a choice over true/false, as the bench's `as_choice` approaches do. Models that read at most N
options in one pass (the letters styles: 16, 10 or 24; slots: 16) score a longer choice by a
two-round knockout. Local models load once per process and share one device lock (Apple's MPS
backend aborts the process when two threads submit work at once).

### Naming a model

The `model` setting takes three forms:

- **A preset name** -- the bench's model configurations (`configs/models/*.yaml` there), pinned
  revisions, released adapters, heads and shipped temperatures included: `kev-4b`, `kev-9b`,
  `kev-27b`, `plumb-4b`, `jevk5-4b`, `jevk5-9b`, `decider-2b`, `decider-4b`, `decider-35b-a3b`,
  `tev1-4b`, `open-jev-9b`, `open-jev-27b`, `openjev-nli-4b`, `jev-9b`, `jev-27b`, `gev-26b`, `laya`
  (the typed-decisions checkpoint, as the bench runs it), `von`, `bekko-400m`, `gliner-decide`,
  `jev-omni`, and the Yale fine-tunes `kev-4b-europeana-persons`, `gliner-decide-europeana-persons`,
  `gliner-decide-linkedart-recon`. Any key may be overridden in the settings (`temperature`,
  `device`, ...).
- **`<family>/<checkpoint>`** -- a Hub repo or a local folder: `laya/convaiinnovations/laya` (the
  English root checkpoint), `gliner//path/to/checkpoint`, `pointer/yale-cultural-heritage/kev-4b-europeana-persons`,
  `letters/crh225/plumb-4b` with `style: jevk5`. Families that need a released adapter or head take
  them as `adapter` and `head` settings (see the presets for the shape).
- **A LiteLLM chat model id** for the generative baseline.

Two kinds of folder are understood as fine-tunes. A **kev-layout** folder or repo
(`adapter_config.json`, `adapter_model.safetensors`, `head.pt`) names its base model and revision in
`head.pt`, so `pointer/<it>` is enough. A **bench export** (`s1bench.json` + `trainable.safetensors`,
what `system1-finetune-bench` saves as `best/`) names its family, base, LoRA shape and fitted
temperatures; `<family>/<folder>` loads the base, attaches the LoRA and restores every trained
tensor. The kev layout is exercised on `kev-4b-europeana-persons`; the bench-export path on a
fabricated export (`tests/test_system_one_export.py`), since no real export is on this machine.

### Yale's fine-tunes

The three reconciliation fine-tunes expect the layout they were trained on: a `Mention: <name>`,
`Entity type: <type>`, `Text: <record>` header, candidates as `<source>:<id>` keys described as
`<label> — <description>`, a `none` option (that is how they abstain), and no task instruction.
In a workflow that is the `state` template plus `options_from`:

```json
{"type": "decision.DecisionClassifier", "id": "select",
 "settings": {"model": "kev-4b-europeana-persons",
              "state": "Mention: {name}\nEntity type: {type}\nText: {context_text}",
              "question": "Which record is the {type} “{name}” referred to in the text?",
              "options_from": "candidates", "option_key": "{key}", "option_text": "{summary}",
              "none_option": "None of the listed records is the entity referred to in the text"}}
```

On the Europeana persons test split (364 decisions), chai's run of `kev-4b-europeana-persons`
agreed with the labels on 353 and with the bench's own recorded predictions on 363, probabilities
within 0.0004 on average (`examples/kev_finetune.py --compare`).

## Layouts

`layout` is the bench's inference-time prompt approach: `described` (default -- option
descriptions on the options), `in_state` (descriptions listed in the state under `options_heading`,
options become bare keys: Laya's best layout on reconciliation, GLiNER's worst), `bare` (keys
only). The state and question actually sent are stored on the result (`metadata.decision`).

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
label `none` when the top probability is below it; `none_option` adds an option the model may pick.
For `question_type: noul` the label is `true`/`false` (threshold 0.5 on P(true)); for `score` it is
the level key, with `metadata.expected` as the probability-weighted level. Wire a `LabelTestGate`
(`label: ["none"]`) or `SwitchGate` on the label, or a `ThresholdGate` on `metadata.confidence`.

### DecisionGate -- a noul question as a branch

```json
{"type": "decision.DecisionGate", "id": "is_person",
 "settings": {"model": "von", "question": "The mention '{name}' refers to a person.", "threshold": 0.7},
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

Output: `ItemResult({"team": {...}, "urgent": {...}, "tone": {...}})`. Jev and Laya answer all
questions in one call; the other providers loop.

## Installing the local models

`peft` (adapters for the decoder families) is a chai dependency. The rest are not; install the
ones you use:

```bash
pip install laya                       # laya
pip install sentence-transformers      # bekko (its own inference code imports it)
```

```bash
uv pip install "gliner2[local]" --override <(echo "transformers>=5.5")   # gliner (declares transformers<5 but runs on 5)
```

Licences: the kev, plumb, JevK5, decider, Open-Jev, JEV, Laya, von and GLiNER2.5-Decide weights are
Apache-2.0, openjev MIT, Yale's GLiNER fine-tunes CC-BY-NC-4.0; bekko declares none and Tev1's is
"being finalized" (fine for internal experiments, not for redistributing fine-tuned weights).

## Measured

`examples/system1_models.py --offline` runs every available model on five reconciliation decisions
(real Wikidata search hits, a none option, "Theo" has no right candidate) and on a noul and a score
question about two texts. On an Apple M4 Max, each in its best layout:

| Model | Correct | Time for 7 decisions |
|---|---|---|
| Laya (English root, `in_state`) | 2/5 | ~1 s |
| GLiNER2.5-Decide (described, mention in the state) | 3/5 | ~6 s |
| von | 2/5 | ~1 s |
| bekko-400m | 5/5 | ~21 s |
| decider-2b | 4/5 | ~3 s |
| kev-4b-europeana-persons (Yale fine-tune) | 5/5 | ~6 s |
| gliner-decide-europeana-persons (Yale fine-tune) | 4/5 | ~5 s |
| Jev-Omni | 4/5 | ~31 s after a ~20 s load |
| Gemini 3.1 Flash Lite, generative baseline | 5/5 | ~25-30 s, ~$0.004 |

Only bekko, the two Yale fine-tunes and Gemini answer "none" for Theo. The Europeana persons GLiNER
fine-tune abstains on Arles, a place: it was trained on people. Local models cost nothing. The hosted Jev API was not measured: no key was available.

## A complete example

`examples/reconcile_wikidata.py` runs the whole reconciliation shape: an LLM extractor finds the
entities, plain Python (outside chai) searches Wikidata for each name, and a `DecisionClassifier`
asks the model which candidate the mention refers to, then prints each entity's chosen record, its
probability and what the run cost. `tests/test_example_reconcile.py` runs the same functions offline.

## Cost and usage

Every decision result carries `model`, `token_usage` and `cost`, so the workflow's usage ledger
(`workflow.usage_summary()`) counts decisions alongside LLM calls.

## Moving from chai-reconciler's selectors

| reconciler setting | chai |
|---|---|
| `LayaSelector` / `JevSelector` / `GlinerDecideSelector` / `JevOmniSelector` / `GeminiSelector` | `decision.DecisionClassifier` with `model: laya/convaiinnovations/laya`, `jev/jev-latest`, `gliner/fastino/GLiNER2.5-Decide`, `jev-omni`, `gemini-3.1-flash-lite` |
| `GlinerDecideTunedSelector` | `model: gliner-decide-linkedart-recon` (or `gliner/<checkpoint folder>`), `state: "Mention: {name}\nEntity type: {type}\nText: {context_text}"` |
| `question_format: state` / `criteria` | `layout: in_state` / `described` |
| `null_mode: explicit` / `threshold`, `null_threshold` | `none_option: "..."` / `threshold: 0.5` |
| `max_candidates` | `max_options` |
| `selector_name` | the component `id` |
| `candidates[].summary` as option text | `options_from: candidates`, `option_text: "{summary}"` |

## Tests

`tests/test_system_one.py` covers the registry and presets, the logit readout and knockout, and the
exact prompt every family builds (against the bench's layouts), without weights.
`tests/test_decision.py` covers the data model, the roles and the Jev request shape (a throwaway HTTP
server stands in for the API). `tests/test_system_one_models.py` runs the real models where they are
cached and skips otherwise: Laya, von, bekko, decider-2b, GLiNER and the Yale GLiNER fine-tunes
(including a real Europeana decision), the kev fine-tune with `CHAI_TEST_KEV=1`, Jev-Omni with
`CHAI_TEST_JEV_OMNI=1`, the hosted Jev API with `TYPESAFE_API_KEY`. `tests/test_system_one_export.py`
loads a fabricated bench export with `CHAI_TEST_EXPORT=1`.

```bash
CHAI_TEST_KEV=1 CHAI_TEST_JEV_OMNI=1 CHAI_TEST_EXPORT=1 python -m unittest tests.test_system_one_models tests.test_system_one_export -v
```
