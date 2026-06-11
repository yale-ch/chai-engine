# chai-engine

A component-based processing engine for AI workflows. Chai abstracts complex
AI builds — OCR, detection, extraction, classification, retrieval — behind a
single JSON configuration: you describe a tree of components, chai runs it,
and every intermediate result stays traceable, reviewable, and correctable.

Chai is one of three sibling projects:

| Repo | What it is |
|---|---|
| **chai-engine** (this repo) | The engine: components, results, routing, AI backends |
| **chai-workflow-builder** | n8n-style drag-and-drop designer for building and test-running workflows |
| **chai-ui-builder** | Drag-and-drop front-end builder; exports standalone Flask apps for a workflow |

## Install & run

```bash
pip install -r requirements.txt
cp env.template .env        # add GEMINI_API_KEY (or GOOGLE_CLOUD_PROJECT for Vertex)

python -m unittest discover -s tests    # 78 tests, no API keys needed
```

A workflow is a JSON tree handed to `Workflow`:

```python
from chai.workflow import Workflow

wf = Workflow({
    "type": "Workflow", "id": "wf",
    "steps": [
        {"type": "extractor.AIExtractor", "id": "ner",
         "settings": {"service": "gemini", "model": "gemini-3.1-flash-lite",
                      "prompt": "Extract entities as a JSON list of {\"type\":..., \"text\":...}"},
         "next_steps": [{"type": "annotator.TextHighlightAnnotator", "id": "highlight"}]}
    ],
})
result = wf.run("Tom visited D.C. with Ada Lovelace in 1922.")
result.view()
```

## Core concepts

### Components and Results

A **`Component`** receives a **`Result`**, computes, and returns a new
`Result`. Components implement `_process(input)`; the engine's
`process(input)` wrapper adds lifecycle events, error policy, provenance
bookkeeping, and forwarding to downstream steps.

A `Result` is self-describing:

- `value` — the payload (text, dict, list of child Results, file bytes…)
- `processor` — the component that produced it
- `input` — the Result it was computed **from**; following `.input` repeatedly
  walks the provenance chain back to the original run input
- `metadata` — `type` (`TEXT`/`IMAGE`/`AUDIO`/`DATA`), timestamps, and
  component facts (`bbox`, `confidence`, `yolo_class`, `token_usage`, …)
- `derivative_results` — results other components registered against this one
  (via `register_on`), e.g. "the labels a classifier assigned to this file"

Shapes: `ItemResult` (one value), `ListResult` (many), `FileItemResult`
(mirrors a file on disk; bytes load lazily), `DirectoryListResult`,
`LabelListResult`.

### Routing: how results reach downstream operations

Which result a downstream component runs on is determined entirely by **which
port it hangs off of** in the config tree:

| Wiring | What the child receives |
|---|---|
| `steps` of a component | The parent's **input** — every child gets the *same* input (fan-out) |
| `steps` of the **Workflow root** | The **previous step's output** (top-level steps chain — the one exception) |
| `next_steps` | The parent's **output**; one follower gets it unwrapped, several each get it and their outputs merge into a `ListResult` |
| `true_steps` / `false_steps` (gates) | The gate's **input**, only when the test passed/failed |
| `case:<label>` (SwitchGate) | Each **individual matching item**, wrapped with `case` metadata |
| `error_steps` | An **ERROR result** describing a failure, only when the component failed after retries |
| children of an `Iterator` | Each **entry** of the list input, one at a time |

Two consequences worth internalizing:

1. **Gates are guarantees.** "This transcriber only ever runs on confident
   barcode crops" is structural — wire it behind
   `MetadataTestGate(yolo_class = barcode)` + `ThresholdGate(0.8)`.
2. **Reducers are the joins.** A parent's `steps` fan out; its `next_steps`
   receive the merged list — so *run A and B, then merge* is
   `FanOut{steps: [A, B], next_steps: [MergeDictReducer]}`. When results are
   scattered deeper (inside gates/iterators), `CollectReducer(components:
   [ids])` gathers everything those components produced, wherever it landed.

## Component roles

| Role | Job | Examples |
|---|---|---|
| **Provider** | Generate a Result from raw input | `DirFileProvider`, `FileListProvider`, `IIIFDirFileProvider`, `StaticProvider` |
| **Iterator** | Run children once per entry | `Iterator` (with `workers`, `continue_on_error`, `cache`) |
| **Classifier** | Assign labels | `KeywordClassifier`, `FileTypeClassifier`, `YoloClassifier`, `AIClassifier` |
| **Gate** | Conditional branch (`true_steps`/`false_steps`) | `ConditionGate`, `ValueTestGate`, `MetadataTestGate`, `ThresholdGate`, `FileTypeGate`, `LabelTestGate` |
| **SwitchGate** | Per-label branch (`case_steps`), for-each over lists | `SwitchGate` |
| **Segmenter** | Break content into parts | `TextSegmenter`, `WordSegmenter`, `YoloSegmenter` (detection crops), `AISegmenter` |
| **Transcriber** | Image/audio → text | `TextFileTranscriber`, `AITranscriber` (vision OCR) |
| **Describer** | Content → descriptive text | `FileInfoDescriber`, `AIDescriber` |
| **Extractor** | Content → structured data | `WordCountExtractor`, `JsonXpathExtractor`, `AIExtractor` (with `schema` validation) |
| **Translator** | Language transforms | `GlossaryTranslator`, `AITranslator` |
| **Reducer** | Merge branched runs into one result | `FlattenReducer`, `CollectReducer`, `MergeDictReducer`, `TextJoinReducer`, `AIReducer` |
| **Annotator** | Human-reviewable artifacts | `ImageBoxAnnotator` (supervision boxes), `TextHighlightAnnotator` (labelled `<mark>`s) |
| **Embedder** | Vectors + similarity | `VectorIndexer`, `VectorRetriever` (SQLite store; hash/gemini/ollama/openai services) |
| **Evaluator** | Score vs ground truth | `TextMetricsEvaluator` (exact/CER/WER), `RecordFieldEvaluator` (field P/R/F1) |
| **Storage** | Persist results | `FileSystemStorage`, `SqliteStorage` (+ human corrections) |
| **Utils** | Plumbing | `FanOut` (explicit parallel fan-out), `DebugStep` (print & pass) |

Deterministic components (`KeywordClassifier`, `TextSegmenter`,
`StaticProvider`, `TextFileTranscriber`, `FileInfoDescriber`,
`GlossaryTranslator`, all reducers/gates/evaluators, hash embeddings) run
without models or API keys — the test suite and example workflows rely on
this.

### AI services: one node, any backend

`AITranscriber`, `AIExtractor`, etc. pick their backend with a `service`
setting — constructing one returns the matching composite class:

```json
{"type": "transcriber.AITranscriber",
 "settings": {"service": "gemini", "model": "gemini-3.1-flash-lite", "prompt": "Transcribe this."}}
```

Services: `gemini` (default; Vertex AI via `GOOGLE_CLOUD_PROJECT`),
`lmstudio`, `ollama`, `openai`, `custom` (any OpenAI-compatible endpoint —
set `api_host`), `vllm`, `sglang`, `mlx_vlm` (Apple Silicon).

### Conditions: one test language for all gates

Gates evaluate JSON conditions (`chai/conditions.py`) against any Result:

```json
{"all": [
  {"source": "metadata.yolo_class", "op": "in", "value": ["label", "barcode"]},
  {"source": "metadata.confidence", "op": "gte", "value": 0.8}
]}
```

Sources: `value`, `metadata.<dotted.path>`, `type`, `labels`, `file_name`,
`input.<source>` (walk up provenance). Ops: `eq ne gt gte lt lte contains in
intersects matches exists truthy`. Combinators: `all`, `any`, `not`.

## Reliability

Every component accepts an error policy in `settings`:

```json
{"retries": 2, "retry_delay": 1.0, "on_error": "skip"}
```

- `retries` — re-run on failure with exponential backoff
  (`component_retry` events stream to listeners)
- `error_steps` — a config branch that receives an ERROR ItemResult when
  retries are exhausted, instead of the run dying
- `on_error: skip` — drop this component's output and continue

`Iterator` adds corpus-scale controls:

- `workers: 8` — process entries through a thread pool (input order preserved)
- `continue_on_error: true` — failed entries become ERROR records, the rest proceed
- `cache: run_cache.db` — completed entries replay on re-runs (keyed by item
  identity + a hash of the child-step config, so editing a prompt invalidates
  the cache); interrupted corpus runs resume where they stopped

`Extractor` adds `schema` — a JSON-Schema subset (`type`, `properties`,
`required`, `items`, `enum`) checked on every output; invalid model output
flows into the retry loop, so `retries: 2` + a schema re-asks the model until
the record validates.

## Observability & human review

- **Lifecycle events** — `Workflow.add_listener(fn)` receives
  `component_start` / `component_end` (with a value preview, the branch a gate
  took, and the live result) / `component_retry` / `component_error` for every
  component. The workflow-builder streams these as NDJSON for live per-node
  progress.
- **Provenance** — any result's `.input` chain answers "where did this come
  from"; `processor` answers "who made it".
- **Storage + corrections** — `SqliteStorage` keeps the full JSON of every
  result; `chai.storage` viewer helpers (`list_results`, `save_correction`,
  …) let an app show stored rows and save human corrections alongside the
  originals (originals are never overwritten).
- **Annotators** — burn detection boxes into images (`ImageBoxAnnotator`) or
  highlight extracted entities in their source text, one visualization per
  task, labels colored per entity type (`TextHighlightAnnotator`).

## Workflow JSON reference

```json
{
  "type": "Workflow", "id": "wf",
  "settings": {"defaults_path": "...", "library_path": "..."},
  "steps": [
    {
      "type": "segmenter.YoloSegmenter",       // module.Class under chai/
      "id": "detect",                           // stable id (referenced by gates/reducers)
      "name": "Find regions",                  // display name
      "settings": {"model": "yolo11n.pt"},     // component-specific config
      "register_on": ["some_id" | "parent"],   // register output on an ancestor result
      "base": "LibraryEntry",                   // start from a library.json template
      "steps": [...], "next_steps": [...], "error_steps": [...],
      "true_steps": [...], "false_steps": [...],          // gates only
      "case_steps": {"Label": [...]}, "default_steps": [...]  // SwitchGate only
    }
  ]
}
```

Default prompts live in `chai/data/prompts.json`; reusable step templates in
`chai/data/library.json` (referenced via `base`).

## The builders

- **chai-workflow-builder** (`uv run app.py`, port 5057): drag components
  onto a canvas, wire ports (steps / next / err / true / false / case),
  run with live per-node progress and outputs, preview annotated
  images/text, run-up-to-any-node. Custom components drop into its
  `custom_components/` folder and appear under *Custom*.
- **chai-ui-builder** (`uv run app.py`, port 5067): design a front-end for a
  workflow (inputs, run button, live progress, output viewers, storage
  tables with human correction), bind viewers to specific workflow steps,
  theme it (Yale palette by default), and **Export** a standalone Flask app
  runnable with `uv run app.py`.

## Extending chai

Subclass a role, implement `_process`, document settings in the docstring
(the builders parse the `Settings:` block into their inspectors):

```python
from chai.classifier import Classifier
from chai.result import LabelListResult

class SentimentClassifier(Classifier):
    """Labels text positive/negative by a wordlist.

    Settings:
        - threshold: minimum hits to emit a label (default 1)
    """
    def _process(self, input):
        ...
        return LabelListResult(labels, input=input, processor=self)
```

Drop the file in the workflow-builder's `custom_components/` to use it
visually, or reference it by dotted path in workflow JSON.
