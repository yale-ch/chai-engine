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

python -m unittest discover -s tests    # no API keys, no network needed
```

Hosted and served models are reached through [LiteLLM](https://docs.litellm.ai), so any
provider it knows -- Gemini and Vertex AI, OpenAI, Anthropic, Ollama, LM Studio, vLLM, sglang,
OpenRouter, Groq, ... -- is one `model` string away, and every call is priced.

A workflow is a JSON tree handed to `Workflow`:

```python
from chai.workflow import Workflow

wf = Workflow({
    "type": "Workflow", "id": "wf",
    "steps": [
        {"type": "extractor.GeminiExtractor", "id": "ner",
         "settings": {"model": "gemini-3.1-flash-lite",
                      "prompt": "Extract entities as a JSON list of {\"type\":..., \"text\":...}"},
         "next_steps": [{"type": "segmenter.TextSpanLocator", "id": "locate"}]}
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
| **Provider** | Generate a Result from raw input | `DirFileProvider`, `FileListProvider`, `CsvFileProvider`, `IIIFDirFileProvider`, `StaticProvider` |
| **Iterator** | Run children once per entry | `Iterator` (with `workers`, `continue_on_error`), `SliceIterator` (every nth entry, for parallel runs) |
| **Classifier** | Assign labels | `KeywordClassifier`, `FileTypeClassifier`, `YoloClassifier`, `GeminiClassifier`, ... |
| **Gate** | Conditional branch (`true_steps`/`false_steps`) | `ConditionGate`, `ValueTestGate`, `MetadataTestGate`, `ThresholdGate`, `FileTypeGate`, `LabelTestGate` |
| **SwitchGate** | Per-label branch (`case_steps`), for-each over lists | `SwitchGate` |
| **Segmenter** | Break content into parts | `TextSegmenter`, `WordSegmenter`, `TextSpanLocator` (where extracted values are in their text), `YoloSegmenter` (detection crops), `GeminiSegmenter`, ... |
| **Transcriber** | Image/audio → text | `TextFileTranscriber`, `GeminiTranscriber` and other vision backends |
| **Describer** | Content → descriptive text | `FileInfoDescriber`, `GeminiDescriber`, ... |
| **Extractor** | Content → structured data | `KeywordExtractor`, `WordCountExtractor`, `JsonXpathExtractor`, `GeminiExtractor`, ... |
| **Translator** | Language transforms | `GlossaryTranslator`, `GeminiTranslator`, ... |
| **Reducer** | Merge branched runs into one result | `FlattenReducer`, `CollectReducer`, `MergeDictReducer`, `TextJoinReducer`, `GeminiReducer`, ... |
| **Embedder** | Vectors + similarity | `VectorIndexer`, `VectorRetriever` (SQLite store; local hash, or `GeminiVectorIndexer`, `OllamaVectorRetriever`, ... per AI backend) |
| **Decision** | System One questions: choice / score / noul answered with probabilities | `DecisionClassifier` (one question, the answer as a label), `DecisionGate` (a true/false question as a branch), `DecisionExtractor` (several questions, one record) -- see [docs/system1.md](docs/system1.md) |
| **Storage** | Persist results (CHAI-TEA rows) | `JsonLinesStorage`, `ParquetStorage`, `SqliteStorage`, `PostgresStorage` (+ human corrections), `FileSystemStorage` |
| **Uploader** | Hand a file to another system | `S3Uploader`, `FileSystemUploader` |
| **FanOut** | Explicit parallel split | `FanOut` (run all children on the same input; optional concurrent `workers`) |
| **Utils** | Plumbing | `DebugStep` (print & pass) |

Deterministic components (`KeywordClassifier`, `TextSegmenter`,
`StaticProvider`, `CsvFileProvider`, `TextFileTranscriber`, `FileInfoDescriber`,
`GlossaryTranslator`, all reducers/gates, hash embeddings) run
without models or API keys — the test suite and example workflows rely on
this. Tests of the model-backed components use LiteLLM's local mock replies
(`"litellm_params": {"mock_response": "..."}`), so they run the real component
code without a network.

### AI backends

Every AI-capable role has a generated composite per backend: the role contributes prompts and
defaults, the backend contributes the model call. There is one backend for hosted and served chat
models, **`LLMComponent`** (`chai/ai/llm.py`), built on LiteLLM: the provider is part of the model
id, so `extractor.LLMExtractor` with `"model": "gemini/gemini-3.1-flash-lite"`,
`"vertex_ai/gemini-3.7-flash"`, `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-5-5"`,
`"ollama_chat/llama3.2"`, `"lm_studio/qwen/qwen3.5-9b"` or `"hosted_vllm/Qwen/Qwen3-8B"` is the
same component talking to a different model. The familiar names still work and keep each
provider's defaults -- `GeminiTranscriber` (Vertex AI via `GOOGLE_CLOUD_PROJECT`, Google Search
grounding on by default, `tools: []` disables; `thinking_level` on Gemini 3+), `OllamaExtractor`,
`LMStudioDescriber`, `OpenAIClassifier`, `VLLMSegmenter`, `SGLangTranslator` -- as thin subclasses.
Settings that are not set are not sent (the provider's defaults apply); a setting that is set is
always sent, so a model that rejects it fails loudly instead of being silently ignored. In-process
backends: `TransformersComponent` (text, or vision-language models such as Chandra OCR 2, on CUDA,
Apple Silicon or the CPU -- `transcriber.TransformersTranscriber`) and `MLXVLMComponent` (vision,
Apple Silicon). An AI classifier (`GeminiClassifier`, `LLMClassifier`, ...) returns the model's
reply as labels matched to its `labels` setting, so a `LabelTestGate` can route on them.

**System One decision models** are a different kind of backend -- they answer typed questions with
calibrated probabilities instead of text -- and live in their own package, `chai/system_one/`: one
provider per architecture family (kev, the letter-readout models, Open-Jev, openjev-NLI, JEV, Laya,
von, bekko, GLiNER2.5-Decide, Jev-Omni, the hosted Jev API, or any chat model as a baseline), named
by preset (`kev-4b`, `laya`, `gliner-decide-europeana-persons`) or as `<family>/<checkpoint>`, with
their roles in `chai/decision.py`; see [docs/system1.md](docs/system1.md).

### Usage and cost

Every model-backed result carries `token_usage` (prompt / result / thinking / images / total),
`cost` (USD from LiteLLM's price table, or your own `pricing` setting for self-hosted models; `null`
when nobody can price the call) and `duration`. The workflow keeps the run-wide tally:

```python
wf.run(input)
wf.usage_summary()
# {'calls': 42, 'total_tokens': 310_221, 'cost': 0.0834, 'unpriced_calls': 0, 'seconds': 97.2,
#  'by_component': {'ner': {...}, 'select': {...}}, 'by_model': {'vertex_ai/gemini-3.1-flash-lite': {...}}}
```

The same summary rides on the workflow's final `component_end` event as `usage`, so a harness can
show what a run cost as it finishes.

A local model's cost is the machine and the time, so every call that ran on this machine (an
in-process model, or a server at localhost such as Ollama or LM Studio) records `local_compute`:
the device, the seconds and hours, and the hardware -- OS, CPU model and cores, RAM and GPUs, the
same on Windows, macOS and Linux (`chai/hardware.py`). The summary adds `local_hours` and a
`by_hardware` breakdown, so a run that mixes hosted and local models reports dollars for the one
and machine-hours for the other.

### Conditions: one test language for all gates

Gates evaluate JSON conditions (defined alongside the gates in `chai/gate.py`) against any Result:

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
- `continue_on_error: true` — failed entries become ERROR records, the rest
  proceed (pair with `SqliteStorage` to persist what succeeded)


## Observability & human review

- **Lifecycle events** — `Workflow.add_listener(fn)` receives
  `component_start` / `component_end` (with a value preview, the branch a gate
  took, and the live result) / `component_retry` / `component_error` for every
  component. The workflow-builder streams these as NDJSON for live per-node
  progress.
- **Provenance** — any result's `.input` chain answers "where did this come
  from"; `processor` answers "who made it".
- **Storage + corrections** — every storage writes the CHAI-TEA schema
  (`chai/persistence/schema.py`): `SqliteStorage` and `PostgresStorage` write its tables —
  the run's PROJECT, WORKFLOW and WORKFLOW_RUN rows, taken from the workflow
  object, and a RESULT row per result — and `JsonLinesStorage`/`ParquetStorage`
  write the same RESULT rows to files. Each row keeps the result's value and
  metadata, with the file or URI it was generated from (the
  component marked `source: true`, e.g. the page, not the directory), the md5
  of that input, and a locator saying where in it the row is — a bounding box
  or a character range — so millions of sentences or regions need no files of
  their own; `chai.persistence` viewer helpers (`list_results`,
  `save_correction`, …) let an app show stored rows and record corrections.
  A correction is an entry of its own pointing back at the row it corrects,
  so originals are never overwritten and each correction carries the metadata
  of the agent — human or model — that made it.
- **Regions and spans are data, not pictures** — the engine records what was
  found and where, and a front end draws it. A `YoloSegmenter` region is a row
  whose `input_segment` is its bounding box; `TextSpanLocator` turns an
  extractor's values into one row per occurrence, `{"label", "text"}` with the
  character range it occupies in the text row it names
  (`chai.persistence.locators` defines the frames). `viewer/` draws them: boxes
  over the page image, highlights in the text, a colour per label.

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
`chai/data/library.json` (or the file named by the workflow's `library_path` setting).

A step with `base` starts from the library entry of that name: its own keys replace the
entry's, except `settings`, which are merged key by key with the step's winning. So a workflow
can name a storage once in the library and reuse it with only what differs:

```json
{"base": "JsonLinesStorage", "id": "log_pages", "settings": {"file": "results/pages.jsonl"}}
```

is the library's `storage.JsonLinesStorage` step with its `mode` setting kept and its `file` replaced.

## The builders

- **chai-workflow-builder** (`uv run app.py`, port 5057): drag components
  onto a canvas, wire ports (steps / next / err / true / false / case),
  run with live per-node progress and outputs, run-up-to-any-node. Custom components drop into its
  `custom_components/` folder and appear under *Custom*.
- **chai-ui-builder** (`uv run app.py`, port 5067): design a front-end for a
  workflow (inputs, run button, live progress, output viewers, storage
  tables with human correction), bind viewers to specific workflow steps,
  theme it (Yale palette by default), and **Export** a standalone Flask app
  runnable with `uv run app.py`.

## Examples

`examples/` holds runnable scripts and workflow JSON. `examples/reconcile_wikidata.py` is the
end-to-end shape chai is built for: NER with an LLM, candidate records from Wikidata (plain
Python), and a System One style selection -- which candidate does this mention refer to, or none
-- with every model call priced. `examples/system1_models.py` runs every available System One
model side by side on the same decisions and reports accuracy, time and cost per model.
`examples/workflow-system1.json` is the selection step alone as workflow JSON.

The other examples, each showing one thing:

| Example | Shows |
|---|---|
| `experiment-locators.py` | Rows that point back into their document (`source`, character-range locators) -- no models needed |
| `workflow-m804-lines.json` | Layout detection, line transcription with neighbouring lines as context, Parquet, S3 upload |
| `experiment-m804.py` | Page-to-page context while transcribing and translating a manuscript |
| `experiment-ppl-jsonl.py` | A CSV streamed to JSON-Lines, error branches, dividing a run between processes |
| `experiment-postgres-parquet.py` | PostgreSQL directly vs. via Parquet, producing identical rows |
| `experiment-jsonl-parquet-s3.py` | Handing a finished run to a central database through S3 |
| `workflow-pii-routing.json` | Classifier and gate: pages with personal data stay local, the rest go to the cloud |
| `workflow-iiif.json` | Transcribing the images of a IIIF manifest |
| `workflow-alt-text.json` | Alt-text from a catalog record and a manifest's first image, all JSON: fetch, XPath, fan-out, describe |
| `workflow-lux-routing.json`, `lux_routing.py` | Classify each image, then route it: alt-text, Chandra OCR 2 locally, or Gemini 3.8 Flash for hard handwriting; dollars and machine-hours per row |
| `workflow-alt-text-yuag.json` | The same from IIIF manifests alone (classification and medium from their metadata, images scaled by the image service), with cost per row |
| `kev_finetune.py` | Running a fine-tuned System One model on a decision benchmark |

Workflow JSON runs as it is with `scripts/run_workflow.py` (`--input` for the provider's input,
`--text` for text, `--set ID.KEY=VALUE` for a setting); `scripts/workflow_diagram.py` draws one as a
Mermaid flowchart (`.mmd` and `.png`); `scripts/smoke_backend.py` checks a vLLM, sglang or MLX-VLM
backend.

## Where chai is going

[docs/roadmap.md](docs/roadmap.md) lists what the engine still needs to reach its goals, drawn
from the projects that lean on it hardest (entity reconciliation, decision-model fine-tuning).

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
