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

# Run a workflow (--input for the provider's input, --text for text, --set ID.KEY=VALUE for a setting)
python scripts/run_workflow.py examples/workflow-pii-routing.json --input path/to/images

# Draw a workflow as a Mermaid flowchart (.mmd and .png beside it, or -o)
python scripts/workflow_diagram.py examples/workflow-m804-lines.json
```

## Architecture

### Core Classes

- **`chai.core.Component`**: Base class for all processing components. Receives `Result` input, performs computation, returns `Result`.
- **`chai.workflow.Workflow`**: Extends `Component`; manages a registry of components and orchestrates multiple components in a tree structure. It is also the WORKFLOW row of the CHAI-TEA schema and carries its columns from its config: `name`, `description`, `project` (a name or a dict of PROJECT columns), `previous_id`, `is_active`, `created`/`modified` (ISO-8601, default build time); `uuid` is its row id (config `uuid`, else derived from `id`). `to_row()` renders the row, `to_config()` a config that keeps every column, `touch()` moves `modified`. Each `run()` is a `WorkflowRun` (`current_run`, started and finished -- success, duration, usage -- before the final event); `run_id` is its id, and `records()` gives the PROJECT, WORKFLOW and WORKFLOW_RUN rows a run's results hang under.
- **`chai.persistence.records`**: `Project` and `WorkflowRun`, the PROJECT and WORKFLOW_RUN rows as objects (`Record.to_row()` gives exactly the table's columns). USER, ROLE, ANNOTATION and the permission tables have no chai object; they are the database's own.
- **`chai.persistence.schema`**: the CHAI-TEA schema itself -- `TABLES`, the DDL, `column_types(entity)`, `column_kinds(entity)` (as Parquet kinds), `schema_statements()`/`sqlite_statements()`. Everything that has to agree with the schema reads it from here; `chai_tea_schema.py` re-exports it.
- **`chai.result.Result`**: Base result class with `value`, `input`, `processor`, and `metadata`, plus the RESULT columns a result holds itself: `created`/`modified` (assigning `value` moves `modified`), `editor_user_id`, and `was_successful` (false for an ERROR result).
- **`chai.result.ListResult`**: A `Result` containing a list of values.
- **`chai.result.ItemResult`**: A `Result` with a single value.
- **`chai.result.FileItemResult`**: A `Result` that mirrors an on-disk file.
- **`chai.result.DirectoryListResult`**: A `Result` containing a list of file paths.

### Component Types

- **`Provider`**: Generates a `Result` from raw input (e.g., `DirFileProvider` reads files from a directory; `CsvFileProvider` reads a CSV file into one dict-valued `ItemResult` per row).
- **`Iterator`**: Calls further components for each entry in a `Result` to make a new result; `retain_results: false` drops each entry's results once its steps are done (for runs whose results are persisted as they go) and reports only the `processed`/`errors` counts in its output's metadata. `SliceIterator` processes only every nth entry (`slice`/`max_slices`), so one input can be divided between parallel runs.
- **`ContextWindow`**: Carries text from one entry of an `Iterator` into the next (`chai/context.py`). It runs its `steps` with what it kept from the previous entry injected as a TEXT result alongside the input -- so a prompt can address the context as `{text_input_0}` and the input as `{text_input_1}` -- then captures a slice of what they produced (`capture`: `last_line`, `last_lines`, `first_line`, `all`, `tail_chars`; `capture_from` names the component to capture from) for the next entry. Used for work that is not entry-independent: transcribing a page with the last line of the page before it, translating a page with the previous page's text. The carry-over makes the run sequential -- the surrounding iterator must keep `workers: 1` and the input cannot be split with a `SliceIterator`.
- **`Classifier`**: Assigns one or more labels to input (e.g., `KeywordClassifier`, `FileTypeClassifier`, `YoloClassifier`).
- **`Gate`**: Acts as a gating mechanism with `true_steps` and `false_steps` based on a test. `ConditionGate` evaluates a component-agnostic JSON condition (see `chai/gate.py`); `ValueTestGate`, `MetadataTestGate`, `ThresholdGate`, and `FileTypeGate` are convenience subclasses; `LabelTestGate` tests labels registered by a classifier.
- **`Transcriber`**: Extracts text from images or audio.
- **`Describer`**: Generates text to describe content.
- **`Extractor`**: Extracts structured data from content (`KeywordExtractor` is the deterministic one for entities: `[{"label", "text"}]`).
- **`Reducer`**: Combines multiple results into one. Branches converge two ways: a parent's `steps` fan out and its `next_steps` (e.g. `MergeDictReducer`, `TextJoinReducer`) receive the merged list; or `CollectReducer` gathers everything specific components produced anywhere in the input subtree (gate branches, iterator entries). `FlattenReducer` collapses nested lists; `fanout.FanOut` is the explicit fan-out node.
- **`Measurer`**: Sizes regions of an image in real units (`chai/measurer.py`). `ScaleMeasurer` takes the regions found in one image (anything with a label and a `bbox` locator: `YoloSegmenter` crops, `BoxLocator` regions), treats those named in `references` as objects of known size (`{"vertical-ruler": {"height": 92}, "scale_bar": {"length": 5}, "color_chart": {"width": .., "height": ..}}`, in `unit`), uses the most confident detection of each reference (`duplicates: best`; `mean` uses them all), averages pixels-per-unit per axis across references, and emits one result per region with `width`/`height`/`unit` in its value and the `scale`/`calibrated` references in its metadata, the region's locator carried on it. One axis with a reference sizes both (`single_axis`); none leaves `measured: false`. `segmenter.BoxLocator` is its partner for AI detectors: it maps a model's boxes (Gemini `box_2d`, `[ymin, xmin, ymax, xmax]` out of 1000, by default) onto the upright (EXIF-applied) image's pixels as `bbox` locators -- give the model a `max_image_size` so it is sent the upright image too. `examples/workflow-shadow-puppets.json` sizes puppets from the rulers photographed beside them, with `YoloSegmenter` running `hf://yale-cultural-heritage/shadow-puppets-yolov11/shadow-puppets-yolo11s.pt` (`crop: false`: the measurer needs only the boxes).
- **`Translator`**: Translates linguistic content into different languages.
- **`Storage`**: Persists input somewhere (e.g., `FileSystemStorage`, `PostgresStorage`, `SqliteStorage`). `TextFileStorage` writes each result's text to a file of its own, named after the input it was made from (the nearest file up the chain, or `name_from`), so a run over page images leaves one `.txt` per page rather than one file per run. **Record shape:** `JsonLinesStorage` and `ParquetStorage` write rows of the CHAI-TEA RESULT table by default -- exactly its columns, in order (`persistence.RESULT_COLUMNS`, read from `persistence.schema`; built by `result_row`). `workflow_run_id` is the workflow's `run_id`, fresh per `run`, so every storage in a run agrees; `sources`/`constants` go into `extra_data`. Workflows must not set `fields` -- it exists only for files meant for something other than the CHAI-TEA database. `JsonLinesStorage` appends each result to one `.jsonl` file the moment it is produced (`mode: truncate` empties the file when a run starts, once per run however many steps share it -- never when a workflow is merely built), so a long run writes nothing at the end and holds nothing in memory; pair it with an `Iterator` set to `retain_results: false`. The run rows its RESULT rows hang under (PROJECT, WORKFLOW, WORKFLOW_RUN) are kept beside the file in `run.records.json` for `run.jsonl` (`run_records_file`/`read_run_records`): written before a run's first line and again with its outcome when it ends, one WORKFLOW_RUN per run the file holds; none when `fields` or a `run_id` setting is given. `token_usage_summary` totals the `token_usage` metadata an AI run recorded in such a file, per component and overall, optionally costed at prices given per million tokens. `ParquetStorage` collects a whole run into one `.parquet` file (written when the workflow finishes) for bulk loading into a remote database, with the run's PROJECT/WORKFLOW/WORKFLOW_RUN rows in its `chai_tea_records` metadata; several `ParquetStorage` steps of one workflow naming the same `file` share it (the first writes it), and its `file_steps` branch runs on the written file once the run has ended -- an `S3Uploader` there hands the finished run to the central loader from inside the workflow (`examples/workflow-m804-lines.json`; a failing file step is logged as an error and leaves the file); `jsonl_to_parquet` converts a streamed `.jsonl` run into the same thing afterwards, carrying the recorded run rows into `chai_tea_records` too. `parquet_run_records` reads them back, and `load_run_parquet` does a loader's two steps in one call: the run rows, then the RESULT rows. Files from before the storages wrote RESULT rows are reshaped by `jsonl_to_result_rows` (`record_to_result_row` per record, any old key spelling; `fields={"text": "value"}` maps a workflow's renamed keys back; a run id is derived from the file's md5 unless given, and its rows go to `jsonl_to_parquet` as `records`). `SqliteStorage` and `PostgresStorage` write the CHAI-TEA tables themselves (`persistence.schema`; created on first use, foreign keys enforced, a table of the same name in another layout refused): before a run's first result the run's PROJECT, WORKFLOW and WORKFLOW_RUN rows from the objects, then one RESULT row per result and per derivative, and the WORKFLOW_RUN row again with its outcome when the run ends. `write_rows` is the one upsert writer for both dialects; `store_rows`/`store_postgres_rows` put rows made elsewhere in (e.g. a Parquet file's `chai_tea_records` metadata before `parquet_to_postgres` loads its RESULT rows -- the two routes produce identical rows, see `examples/experiment-postgres-parquet.py`). `postgres_to_parquet` exports a PostgreSQL instance's results and annotations to a bundle -- one Parquet file per table plus `manifest.json`, carrying every row they reference (run, workflow and its `previous_id` chain, project, users, correction and reply chains; `since`/`project`/`workflow` filters; permissions only with `include_permissions`) -- and `parquet_bundle_to_postgres` loads one into another database in one transaction (`on_conflict: skip|update`). Every RESULT row records where it came from -- `input`/`input_hash` (the file or URI it was generated from and the md5 of that input's content) and `input_segment` (JSON text: where in that input it is, `[{"bbox": [...]}, {"start": 120, "end": 168}]`, outermost frame first, so a sentence of a region of a page needs no file of its own) -- and corrections are entries of their own: a correction row's `previous_id` points at the row it corrects, its `editor_user_id` at the USER who made it (when that row exists) and its metadata says which agent, human or model, made it (`save_correction`, `save_postgres_correction`, or a component putting `corrects` in a result's `extra`). The SQLite viewer helpers (`list_results`, `get_result`, `list_processors`) return RESULT columns. Schema changes are not migrated -- delete the database and re-run.
- **`Uploader`**: Copies a local file to where another system collects it and returns `{uri, key, size, md5, source}` (`chai/uploader.py`); `key`/`prefix` name the destination (`{name}`, `{stem}`, `{date}`, ... placeholders) and `overwrite: false` refuses to replace one. `S3Uploader` puts it in a bucket via boto3's credential chain (`bucket`, defaulting to `S3_BUCKET`; `profile`, defaulting to `S3_PROFILE`; `region`, `endpoint_url`, `extra_args`), tagging the object with `chai-md5`/`chai-workflow-id`; `FileSystemUploader` copies into a directory. The hand-off a central database ingests from -- `jsonl_to_parquet` packs a run's `.jsonl`, then a workflow uploads the Parquet file -- is `examples/experiment-jsonl-parquet-s3.py`. The S3 tests skip unless `CHAI_TEST_S3_BUCKET` (and `AWS_ENDPOINT_URL` for MinIO etc.) is set.
- **`Embedder`**: Embeddings + vector search (`chai/embedder.py`: `VectorIndexer`, `VectorRetriever` over a SQLite `VectorStore` that lives in `chai/persistence/vectors.py`). The role calls `embed(texts)`; the plain components use the local deterministic `hash_embed`, and the model-backed ones are generated from the `chai/ai/` backends like the other roles (`GeminiVectorIndexer`, `OllamaVectorRetriever`, `OpenAIVectorIndexer`, ...), whose `LLMComponent.embed` makes the LiteLLM `embedding` call with the backend's `DEFAULT_EMBEDDING_MODEL`. No service switching in the component.

**Order (`input_sequence`):** every `Iterator` stamps each entry with its zero-based position in the whole input (`Result.sequence`; a `SliceIterator` keeps the global position, `workers` do not change it), and a RESULT row's `input_sequence` is the nearest position up its chain (`persistence.result_sequence`). So a page's transcription and translation both carry the page's position, and a line handed out by an inner iterator carries its position within its page -- a front end reads a run with `ORDER BY input_sequence`, or `ORDER BY input, input_sequence` for lines. `YoloSegmenter` emits regions in reading order first (`order`: `reading` -- columns left to right, each top to bottom -- `top_to_bottom`, or `detected`; see `segmenter.reading_order`). `persistence.jsonl_to_result_rows` numbers older files, which carry no positions, by the sorted order of their inputs.

`source: true` on a component marks its results as the run's recorded source, so rows made further downstream record *that* (the page, the CSV row) rather than the raw input at the bottom of the chain; a component that carves a piece out of something sets `Result.locator` (`TextSegmenter` with `locate: true` records character ranges, `YoloSegmenter` its bounding boxes) and the storages stack those into the RESULT row's `input_segment`. See `examples/experiment-locators.py`.

**Rendering is the UI's job, not the engine's.** There are no components that draw: the engine stores what was found and where, and a front end decides how to show it. The frames are defined in `chai/persistence/locators.py` -- `{"bbox": [x1, y1, x2, y2]}` in the image's pixels; `{"start", "end"}` characters, plus `result` (the id of the result whose text the offsets index, when that is not a file's content -- e.g. a transcription of a page image) and `exact` (the characters, when the row's value is not them); `{"index"}`/`{"page"}`. A detection is a `YoloSegmenter` crop row (`input_segment` its bbox, `yolo_class`/`confidence` in its metadata); `segmenter.TextSpanLocator`, a `next_step` of an extractor, emits one result per occurrence of each extracted value -- value `{"label", "text"}`, a text frame naming the text it was found in -- and keeps values it cannot find with `located: false`. Hand either list to an `Iterator` with a storage in its steps for a row each. `viewer/` draws them (boxes as an SVG over the image, spans as `<mark>`s in the panel of the row they name, a colour per label). Tying text frames to a result id is provisional and may change with how storage and the segmenters come to handle images and texts.

Every component supports an error policy via settings (`retries`, `retry_delay`, `on_error: skip`) and an `error_steps` config branch. `Iterator` adds `workers` (thread-pool concurrency), `continue_on_error` and `retain_results`.

### AI Components (`chai/ai/`)

- **`GeminiComponent`**: Uses Google's Gemini API (supports Vertex AI via `GOOGLE_CLOUD_PROJECT`).
- **`LMStudioComponent`**: Uses LM Studio local server (`localhost:1234` by default).
- **`OllamaComponent`**: Uses Ollama local server (`localhost:11434` by default).

AI components are typically mixed with base components (e.g., `GeminiTranscriber` extends both `Transcriber` and `GeminiComponent`).

A prompt's `{text_input_<i>}` slot takes the TEXT/DATA item at position `i` of the input; a slot the input did not fill raises `ValueError` (so `retries`/`on_error`/`error_steps` apply) instead of reaching the model as literal braces. Variable slots belong at the end of the prompt, after the fixed rules (or with the rules in `system_instruction`), so the provider can cache the fixed prefix: more than 200 characters of fixed text after the first variable slot is logged as a warning once per component (`ai_utils.prompt_order_problem`).

Every AI component records the same metadata on the result of a call: `token_usage` (`total`, `prompt`, `images`, `thinking`, `result` -- a provider that did not break a modality down writes `-1` for it, not 0), `duration` in seconds, `type`, plus `engine` (the component's `ENGINE_NAME`) and `model` (the model actually used). Storage keeps that dict whole, so a run can be totalled by model afterwards without re-reading anything from the API -- see `token_usage_summary`. The LiteLLM-backed components also record the call's `cost` in USD (LiteLLM's figure, or the component's `pricing` setting for a model LiteLLM cannot price; `None` when neither can); a RESULT row's `md_cost` is that recorded cost, in every storage, and nothing is priced afterwards.

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
- `chai/storage.py`: the `Storage` components only -- what to write and when.
- `chai/persistence/`: everything that writes or transforms what the storages record, with no components: `schema.py` (the CHAI-TEA tables), `records.py` (`Project`/`WorkflowRun`), `provenance.py` (where a result came from), `locators.py` (the frames of `input_segment`), `rows.py` (`result_row`, `build_record`, `correction_row`), `jsonl.py` (JSON-Lines files, run records, `token_usage_summary`), `parquet.py` (`ParquetRecordWriter`, `jsonl_to_parquet`), `sql.py` (`write_rows`), `sqlite.py` and `postgres.py` (each database, its viewer/loader helpers), `bundle.py` (PostgreSQL-to-PostgreSQL via Parquet), `legacy.py` (older files), `vectors.py` (`VectorStore`), `values.py` (shared JSON/timestamp helpers). The public functions are re-exported from `chai.persistence`.
- `chai/uploader.py`: `Uploader`, `S3Uploader`, `FileSystemUploader`.
- `chai/context.py`: `ContextWindow`, which carries text from one iterator entry into the next.
- `chai/measurer.py`: `ScaleMeasurer`, real-world sizes from references of known size.
- `scripts/`: `run_workflow.py` (run a workflow JSON), `workflow_diagram.py` (Mermaid flowchart of one), `smoke_backend.py` (check a vLLM, sglang or MLX-VLM backend).
- `examples/`: runnable scripts and workflow JSON, one idea each (the README's Examples table).
- `viewer/`: a read-only Flask viewer of a stored run -- page images with their text, regions drawn as boxes and spans as highlights from the stored rows.
- `chai/ai/gemini.py`, `chai/ai/lm_studio.py`, `chai/ai/ollama.py`: AI component implementations.
- `chai/transcriber.py`: Transcriber components with AI mixins.
- `chai/ai_utils.py`: JSON extraction utilities for LLM responses.

### CHAI-TEA database (top level, not part of the `chai` package)

The shared PostgreSQL schema a run's results are published into, defined by the ER diagram in `mermaid-er-database.md`: PROJECT -> WORKFLOW -> WORKFLOW_RUN -> RESULT, with ANNOTATION, USER, ROLE and the two permission tables.

The tables are defined once, in `chai/persistence/schema.py`; the storages write them directly (see `Storage` above), and these tools work on the same definition.

- `chai_tea_schema.sql`: the DDL as a plain file, for `psql -f` against Amazon RDS or any PostgreSQL server. Verified to build the same schema as the Python builder.
- `chai_tea_schema.py`: builds the schema from Python (`--sql` prints the DDL, `--drop` rebuilds), re-exporting `chai.persistence.schema` (`column_types(entity)` etc.).
- `chai_tea_parquet.py`: the command line (`export`/`load`) for `persistence.postgres_to_parquet` and `parquet_bundle_to_postgres`, which move CHAI-TEA rows from a local instance to a central one via a Parquet bundle.

Editing a table's columns in `chai/persistence/schema.py` means editing `chai_tea_schema.sql` to match; `tests/test_chai_tea_parquet.py` (the bundle export/load) and the PostgreSQL tests in `tests/test_storage.py` cover both.

### Common Patterns

1. **Component composition**: `Component._process()` defines the core logic; `Component.process()` wraps it with result tracking and registration.
2. **Mixin pattern**: AI components use multiple inheritance (e.g., `GeminiTranscriber(Transcriber, GeminiComponent)`). `create_all_components(role)` generates them, delegating `_process` to the backend; a role that keeps its own `_process` names the backend method it needs instead (`create_all_components(VectorIndexer, delegate="embed")`).
3. **Result registration**: Results can be registered against components via `register_on` to track derivatives.
4. **Prompt loading**: Components can load default prompts from `data/prompts.json` via `workflow.default_prompts`.

### Running Tests

```bash
# Run the unit tests (the PostgreSQL ones skip unless a server is reachable)
python -m unittest discover -s tests

# Smoke-test a self-hosted model backend
python scripts/smoke_backend.py vllm path/to/images/
```

The PostgreSQL tests need a server (the `PG*` environment variables, default localhost:5432) with an
account that may create databases: each test class creates a throwaway database and drops it
afterwards. They skip when there is no server to talk to. `.env` may point `PG*` at a remote server;
override them on the command line to test locally, e.g.
`PGHOST=localhost PGUSER=$USER PGPASSWORD= python -m unittest discover -s tests`.

All components are real (no mocks): deterministic ones (`KeywordClassifier`, `KeywordExtractor`,
`TextSegmenter`, `TextSpanLocator`, `BoxLocator`, `ScaleMeasurer`, `StaticProvider`, `TextFileTranscriber`, `FileInfoDescriber`,
`GlossaryTranslator`) run without models or API keys.
