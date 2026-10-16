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
- **`Iterator`**: Calls further components for each entry in a `Result` to make a new result. `SliceIterator` processes only every nth entry (`slice`/`max_slices`), so one input can be divided between parallel runs.
- **`Classifier`**: Assigns one or more labels to input (e.g., `KeywordClassifier`, `FileTypeClassifier`, `YoloClassifier`).
- **`Gate`**: Acts as a gating mechanism with `true_steps` and `false_steps` based on a test. `ConditionGate` evaluates a component-agnostic JSON condition (see `chai/gate.py`); `ValueTestGate`, `MetadataTestGate`, `ThresholdGate`, and `FileTypeGate` are convenience subclasses; `LabelTestGate` tests labels registered by a classifier.
- **`Transcriber`**: Extracts text from images or audio.
- **`Describer`**: Generates text to describe content.
- **`Extractor`**: Extracts structured data from content.
- **`Reducer`**: Combines multiple results into one. Branches converge two ways: a parent's `steps` fan out and its `next_steps` (e.g. `MergeDictReducer`, `TextJoinReducer`) receive the merged list; or `CollectReducer` gathers everything specific components produced anywhere in the input subtree (gate branches, iterator entries). `FlattenReducer` collapses nested lists; `fanout.FanOut` is the explicit fan-out node.
- **`Annotator`**: Renders results as human-reviewable artifacts (e.g., `ImageBoxAnnotator` burns detection boxes into the source image via supervision; `TextHighlightAnnotator` highlights extracted values in their source text).
- **`Translator`**: Translates linguistic content into different languages.
- **`Storage`**: Persists input somewhere (e.g., `FileSystemStorage`, `PostgresStorage`, `SqliteStorage`).
- **`Embedder`**: Embeddings + vector search (`VectorIndexer`, `VectorRetriever` over a SQLite `VectorStore` that lives in `chai/storage.py`; services: hash, or gemini/ollama/openai-compatible through LiteLLM).
- **Decision roles** (`chai/decision.py`): System One questions -- `choice`, `score`, `noul` -- answered with probabilities by a decision model (`chai/ai/system1.py`: Jev API, Laya and GLiNER2.5-Decide in process, or any LiteLLM chat model as a generative baseline). `DecisionClassifier` asks one question and labels the input with the answer; `DecisionGate` branches on a noul question; `DecisionExtractor` asks several questions and returns one record. Options can come from a list in the input (`options_from`), descriptions can be laid out in the options or in the state (`layout`). See `docs/system1.md`.

Every component supports an error policy via settings (`retries`, `retry_delay`, `on_error: skip`) and an `error_steps` config branch. `Iterator` adds `workers` (thread-pool concurrency) and `continue_on_error`.

Every model-backed result carries `token_usage`, `cost` (USD; `None` when unpriceable) and `duration` metadata; `Component.process` hands them to `workflow.usage` (`chai/usage.py`, a `UsageLedger`), which is reset per `run` and summarized on the workflow's `component_end` event and by `workflow.usage_summary()`.

### AI Components (`chai/ai/`)

- **`LLMComponent`** (`chai/ai/llm.py`): the one backend for hosted and served chat models, built on LiteLLM. The provider is part of the model id (`gemini/...`, `vertex_ai/...`, `openai/...`, `anthropic/...`, `ollama_chat/...`, `lm_studio/...`, `hosted_vllm/...`). Builds the `litellm.completion` call once (`base_kwargs`; `request_kwargs(input)` shows the whole call), normalizes usage and prices the call. Settings that are not set are not sent; a set one is always sent (fail loudly, never silently drop).
- **`GeminiComponent`**, **`OllamaComponent`**, **`LMStudioComponent`**, **`OpenAIComponent`**, **`VLLMComponent`**, **`SGLangComponent`**: thin subclasses that keep each provider's prefix, host and defaults so existing workflow JSON keeps working. Gemini: Vertex AI when `GOOGLE_CLOUD_PROJECT` is set, else the Gemini API; grounding on by default; `thinking_level` (3+) / `thinking_budget` (2.5) rules live in `build_base_kwargs`.
- **`TransformersComponent`**, **`MLXVLMComponent`**: in-process models, unchanged.
- **`system1.System1Model`** (`chai/ai/system1.py`): decision models, provider by prefix (`jev/`, `laya/`, `gliner/`, else generative via `LLMComponent`); canonical questions/answers, layouts, `normalize_answer`.

AI components are typically mixed with role components by `chai.ai.create_all_components` (e.g. `GeminiTranscriber`, `LLMExtractor`). `litellm` is imported lazily, when a component is built.

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
- `chai/ai/llm.py`: `LLMComponent`, the LiteLLM backend; `chai/ai/gemini.py`, `chai/ai/ollama.py`, `chai/ai/lm_studio.py`, `chai/ai/openai.py`, `chai/ai/vllm.py`, `chai/ai/sglang.py`: provider presets on top of it.
- `chai/ai/system1.py`: System One decision models (providers, question/answer model); `chai/decision.py`: the decision roles.
- `chai/usage.py`: `UsageLedger`, run-wide tokens and cost.
- `chai/transcriber.py`: Transcriber components with AI mixins.
- `chai/ai_utils.py`: JSON extraction utilities for LLM responses.

### Common Patterns

1. **Component composition**: `Component._process()` defines the core logic; `Component.process()` wraps it with result tracking and registration.
2. **Mixin pattern**: AI components use multiple inheritance (e.g., `GeminiTranscriber(Transcriber, GeminiComponent)`).
3. **Result registration**: Results can be registered against components via `register_on` to track derivatives.
4. **Prompt loading**: Components can load default prompts from `data/prompts.json` via `workflow.default_prompts`.

### Running Tests

```bash
# Run the unit tests (no API keys, no network)
python -m unittest discover -s tests

# Run an example as a live test case (needs credentials in .env)
python examples/experiment.py
```

All components are real (no mock components): deterministic ones (`KeywordClassifier`,
`TextSegmenter`, `StaticProvider`, `TextFileTranscriber`, `FileInfoDescriber`,
`GlossaryTranslator`) run without models or API keys, and the model-backed ones are tested with
LiteLLM's own local mock replies -- `"litellm_params": {"mock_response": "..."}` in a component's
settings makes LiteLLM answer with that text, with real usage and pricing, without a network. The
Jev System One provider is tested against a throwaway HTTP server in the test process.
