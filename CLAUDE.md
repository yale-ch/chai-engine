# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The chai-engine is a processing engine for LLM-based workflows. It defines a component-based architecture where `Component`s process `Result`s through configurable pipelines.

## Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp env.template .env
# Edit .env to add API keys (GEMINI_API_KEY, GOOGLE_CLOUD_PROJECT)

# Run an experiment
python experiment.py
```

## Architecture

### Core Classes

- **`chai.core.Component`**: Base class for all processing components. Receives `Result` input, performs computation, returns `Result`.
- **`chai.workflow.Workflow`**: Extends `Component`; manages a registry of components and orchestrates multiple components in a tree structure.
- **`chai.result.Result`**: Base result class with `value`, `input`, `processor`, and `metadata`.
- **`chai.result.ListResult`**: A `Result` containing a list of values.
- **`chai.result.ItemResult`**: A `Result` with a single value.
- **`chai.result.FileItemResult`**: A `Result` that mirrors an on-disk file.
- **`chai.result.DirectoryListResult`**: A `Result` containing a list of file paths.

### Component Types

- **`Provider`**: Generates a `Result` from raw input (e.g., `DirFileProvider` reads files from a directory; `CsvFileProvider` reads a CSV file into one dict-valued `ItemResult` per row).
- **`Iterator`**: Calls further components for each entry in a `Result` to make a new result; `retain_results: false` drops each entry's results once its steps are done (for runs whose results are persisted as they go) and reports only the `processed`/`errors` counts in its output's metadata. `SliceIterator` processes only every nth entry (`slice`/`max_slices`), so one input can be divided between parallel runs.
- **`Classifier`**: Assigns one or more labels to input (e.g., `KeywordClassifier`, `FileTypeClassifier`, `YoloClassifier`).
- **`Gate`**: Acts as a gating mechanism with `true_steps` and `false_steps` based on a test. `ConditionGate` evaluates a component-agnostic JSON condition (see `chai/gate.py`); `ValueTestGate`, `MetadataTestGate`, `ThresholdGate`, and `FileTypeGate` are convenience subclasses; `LabelTestGate` tests labels registered by a classifier.
- **`Transcriber`**: Extracts text from images or audio.
- **`Describer`**: Generates text to describe content.
- **`Extractor`**: Extracts structured data from content.
- **`Reducer`**: Combines multiple results into one. Branches converge two ways: a parent's `steps` fan out and its `next_steps` (e.g. `MergeDictReducer`, `TextJoinReducer`) receive the merged list; or `CollectReducer` gathers everything specific components produced anywhere in the input subtree (gate branches, iterator entries). `FlattenReducer` collapses nested lists; `fanout.FanOut` is the explicit fan-out node.
- **`Annotator`**: Renders results as human-reviewable artifacts (e.g., `ImageBoxAnnotator` burns detection boxes into the source image via supervision; `TextHighlightAnnotator` highlights extracted values in their source text).
- **`Translator`**: Translates linguistic content into different languages.
- **`Storage`**: Persists input somewhere (e.g., `FileSystemStorage`, `PostgresStorage`, `SqliteStorage`). `JsonLinesStorage` appends each result to one `.jsonl` file the moment it is produced, so a long run writes nothing at the end and holds nothing in memory; pair it with an `Iterator` set to `retain_results: false`. `ParquetStorage` collects a whole run into one `.parquet` file (written when the workflow finishes) for bulk loading into a remote database; `jsonl_to_parquet` converts a streamed `.jsonl` run into the same thing afterwards. `PostgresStorage` writes the same `results`/`derivatives` shape as `SqliteStorage` into PostgreSQL (jsonb columns, database and tables created on first use), and `parquet_to_postgres` loads a Parquet run into a table there -- the two routes produce identical rows (see `examples/experiment-postgres-parquet.py`). Both tabular schemas record where a row came from -- `input_uri`/`input_hash` (the file or URI it was generated from and the md5 of that input's content) and `input_locator` (where in that input it is: `[{"bbox": [...]}, {"start": 120, "end": 168}]`, outermost frame first, so a sentence of a region of a page needs no file of its own) -- and hold corrections as entries of their own: a correction row's `corrects_id` points at the row it corrects and its own metadata says which agent, human or model, made it (`save_correction`, `save_postgres_correction`, or a component putting `corrects` in a result's `extra`). Schema changes are not migrated -- delete the database and re-run.
- **`Embedder`**: Embeddings + vector search (`VectorIndexer`, `VectorRetriever` over a SQLite `VectorStore` that lives in `chai/storage.py`; services: hash/gemini/ollama/openai-compatible).

`source: true` on a component marks its results as the run's recorded source, so rows made further downstream record *that* (the page, the CSV row) rather than the raw input at the bottom of the chain; a component that carves a piece out of something sets `Result.locator` (`TextSegmenter` with `locate: true` records character ranges, `YoloSegmenter` its bounding boxes) and the storages stack those into `input_locator`. See `examples/experiment-locators.py`.

Every component supports an error policy via settings (`retries`, `retry_delay`, `on_error: skip`) and an `error_steps` config branch. `Iterator` adds `workers` (thread-pool concurrency), `continue_on_error` and `retain_results`.

### AI Components (`chai/ai/`)

- **`GeminiComponent`**: Uses Google's Gemini API (supports Vertex AI via `GOOGLE_CLOUD_PROJECT`).
- **`LMStudioComponent`**: Uses LM Studio local server (`localhost:1234` by default).
- **`OllamaComponent`**: Uses Ollama local server (`localhost:11434` by default).

AI components are typically mixed with base components (e.g., `GeminiTranscriber` extends both `Transcriber` and `GeminiComponent`).

### Workflow Definition

Workflows are defined as JSON trees with `steps` and `next_steps`:

```json
{
  "type": "Workflow",
  "id": "wf1",
  "steps": [
    {
      "type": "provider.DirFileProvider",
      "steps": [
        {
          "type": "iterator.Iterator",
          "steps": [
            {"type": "classifier.Classifier", "id": "classifier1"}
          ]
        }
      ]
    }
  ]
}
```

### Key Files

- `chai/core.py`: Core `Component` and `BaseThing` classes.
- `chai/workflow.py`: `Workflow` class for managing component registries.
- `chai/result.py`: Result class hierarchy (`Result`, `ItemResult`, `ListResult`, `FileItemResult`, `DirectoryListResult`).
- `chai/provider.py`: Provider components for generating results from raw input.
- `chai/gate.py`: `Gate`, `ConditionGate` (+ convenience gates), and `LabelTestGate` for conditional branching.
- `chai/annotator.py`: `Annotator` components plus `annotate_image_bytes`/`collect_detections` helpers (also used by chai-workflow-builder for run previews).
- `chai/ai/gemini.py`, `chai/ai/lm_studio.py`, `chai/ai/ollama.py`: AI component implementations.
- `chai/transcriber.py`: Transcriber components with AI mixins.
- `chai/ai_utils.py`: JSON extraction utilities for LLM responses.

### Common Patterns

1. **Component composition**: `Component._process()` defines the core logic; `Component.process()` wraps it with result tracking and registration.
2. **Mixin pattern**: AI components use multiple inheritance (e.g., `GeminiTranscriber(Transcriber, GeminiComponent)`).
3. **Result registration**: Results can be registered against components via `register_on` to track derivatives.
4. **Prompt loading**: Components can load default prompts from `data/prompts.json` via `workflow.default_prompts`.

### Running Tests

```bash
# Run the unit tests (the PostgreSQL ones skip unless a server is reachable)
python -m unittest discover -s tests

# Run experiment.py as a live test case
python experiment.py
```

The `PostgresStorage` tests need a PostgreSQL server on localhost:5432 and use (creating it if
necessary) a database called `chai`; they skip when there is no server to talk to.

All components are real (no mocks): deterministic ones (`KeywordClassifier`,
`TextSegmenter`, `StaticProvider`, `TextFileTranscriber`, `FileInfoDescriber`,
`GlossaryTranslator`) run without models or API keys.
