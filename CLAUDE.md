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

- **`Provider`**: Generates a `Result` from raw input (e.g., `DirFileProvider` reads files from a directory).
- **`Iterator`**: Calls further components for each entry in a `Result` to make a new result.
- **`Classifier`**: Assigns one or more labels to input (e.g., `SampleClassifier`, `HumanClassifier`).
- **`Gate`**: Acts as a gating mechanism with `true_steps` and `false_steps` based on a test (e.g., `LabelTestGate` tests for specific labels).
- **`Transcriber`**: Extracts text from images or audio.
- **`Describer`**: Generates text to describe content.
- **`Extractor`**: Extracts structured data from content.
- **`Reducer`**: Combines multiple results into one.
- **`Translator`**: Translates linguistic content into different languages.
- **`Storage`**: Persists input somewhere (e.g., `FileSystemStorage`, `PostgresStorage`, `SqliteStorage`).

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
- `chai/gate.py`: `Gate` and `LabelTestGate` for conditional branching.
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
# Run experiment.py as a test case
python experiment.py

# For individual components, create a test script that imports and instantiates them
```
