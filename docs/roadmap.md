# What chai still needs to reach its goals

Chai's promise (README): describe an AI build as one JSON tree, run it, and keep every
intermediate result traceable, reviewable and correctable. This list comes from reading the two
projects that lean on chai hardest today -- **chai-reconciler** (entity reconciliation with Gemini
NER, authority lookups and System One selectors) and **system1-finetune-bench** (fine-tuning the
same decision models) -- and noting where they had to work around the engine, re-implement
something, or leave the engine for a script. Items are roughly in priority order. Where the
`chai-engine-v2` branch already delivers a piece, it says so.

## 1. Project-local components without monkeypatching

`importClass` only resolves `type` strings under the `chai.` package. chai-reconciler
(`reconciler/__init__.py`) and chai-workflow-builder (`builder/custom_loader.py`) both patch
`chai.core.importClass` to load their own components. The engine should resolve any importable
dotted path (`reconciler.components.selector.LayaSelector`) and take a `component_paths` list in
the Workflow settings for a project's `custom_components/` folder.

## 2. Decisions as training data (the fine-tuning loop)

system1-finetune-bench trains on one JSONL schema: `{id, group, type, state, question,
options: [{key, text}], answer, split, meta}`. The v2 `DecisionClassifier`/`DecisionExtractor`
already record exactly the `state`, `question` and `options` they sent (in `metadata.decision`), so
a `DecisionDatasetStorage` that writes those records -- with the model's answer, or a human
correction from `save_correction` as the gold `answer` -- would close the loop from workflow to
fine-tuning set with no scripts in between (today `finetune/gliner_decide/recon_data.py` in the
reconciler rebuilds it by hand).

## 3. Evaluation against gold

The reconciler's selector comparison table (12/12, 11/12 ...) is counted by hand. Chai once had an
`Evaluator` and removed it as premature; with decisions carrying probabilities, the bench's metrics
are the right spec: accuracy, log loss and ECE for choice (plus linking accuracy, abstention recall
and false-link rate when a `none` option exists), exact/within-one accuracy and MAE for score,
balanced accuracy, Brier and AUROC for noul. One component that reads gold from the input record
(or a CSV) and emits those per decision component would replace every project's scoring script.

## 4. Budgets and rate limits on top of the ledger

v2 adds `Workflow.usage` (calls, tokens, cost per component and model). Next: `max_cost` /
`max_calls` on the Workflow that stop a run (fail loudly, or hand an ERROR result to a branch) once
exceeded, so a corpus run cannot silently overspend; LiteLLM's `num_retries`/rate-limit handling
surfaced as settings on every LLM component (the Jev provider hand-rolls 429 + Retry-After); and a
`serialize: true` component setting for in-process models that cannot run concurrently (System One
providers in v2 share one `DEVICE_LOCK` because Apple MPS aborts on concurrent submits -- that
should be a general engine knob).

## 5. The exact request, kept for review

Human review needs what the model saw. Decision components store the rendered state and question;
text LLM components do not yet keep the rendered prompt/messages (`LLMComponent.request_kwargs`
computes it, but it is not written to the result). Add `metadata.request` (messages minus image
bytes) so `SqliteStorage` and the builders can show it, and so item 2 can export prompts too.

## 6. Side-by-side model comparison as a reducer

The reconciler runs several selectors on the same candidates and folds their outputs into a
`selections` dict keyed by selector. Chai has the fan-out (`next_steps`) but no join that aligns N
decision results per item: a `CompareReducer` that emits `{component_id: {label, probabilities,
confidence}}` per item (and agreement/majority) would make "which model is right more often" a
workflow, not an `assemble()` function.

## 7. Human-in-the-loop for low-confidence decisions

`HumanClassifier` is a placeholder. The pieces exist separately -- `ThresholdGate` on
`metadata.confidence`, `SqliteStorage`, `save_correction` -- but there is no review queue (rows a
person must settle), no resume after correction, and corrections do not flow back into item 2.

## 8. Batch and asynchronous model calls

Every call is one synchronous request per item. Laya has `predict_batch` (many states in one
forward pass), Gemini and OpenAI have batch APIs at half price, and LiteLLM has
`abatch_completion`. An `Iterator` that hands a whole batch to components declaring
`supports_batch` would cut both cost and wall-clock on corpus runs.

## 9. Caching identical calls

The run cache was removed because persistence belongs to Storage. LiteLLM ships a pluggable cache
(`litellm.cache`: in-memory, disk, Redis) that can be switched on from settings without chai
owning any persistence; re-running a workflow during development would then cost nothing.

## 10. Structured output that is actually validated

v2 sends a `schema` setting as a `json_schema` response format so capable APIs enforce it. For
providers that ignore it, validate with the `jsonschema` package and count a mismatch as a failure
so `retries` re-asks -- the `Extractor` docstring has promised this since the hand-rolled validator
was removed.

## 11. Audio, video and PDF inputs

`FileItemResult` types `.mp3`/`.wav` as AUDIO but no transcriber could send them; v2's
`LLMComponent` sends audio as a file part (shape tested, not yet exercised against a provider).
PDFs, videos and multi-page TIFFs need Segmenters that split them into page images.

## 12. A lighter install

Importing any role module imports `chai.ai`, and the package depends on `ultralytics`,
`transformers` and therefore `torch`, so a purely deterministic pipeline installs gigabytes (the
chai skill warns about this). Make YOLO, transformers and MLX extras (`chai-engine[yolo,local]`)
and import them lazily, as `litellm` now is.

## 13. Workflow-level model defaults

With LiteLLM a pipeline switches providers by editing one string -- per step. A
`settings.ai: {model, pricing, timeout}` block on the Workflow, inherited by every LLM/decision
component that does not set its own, would make "run this whole pipeline on the local Ollama
instead of Vertex" one edit.

## 14. More in-process decision models

v2 ships Jev (API), Laya and GLiNER2.5-Decide (in process) and a generative baseline. The
reconciler's Jev-Omni runtime (Gemma 4 12B decision classifier, custom loader) and the bench's
other adapter families (pointer, letter-slot, per-candidate yes/no, NLI, slot heads) are the
remaining System One models; a `System1Model` provider that loads a bench checkpoint (`best/` +
`s1bench.json`) directly would let a fine-tuned model go straight into a workflow.

## 15. Dataset providers

`CsvFileProvider` exists; the bench reads JSONL, Parquet and Hugging Face Hub datasets. Providers
for those (one dict-valued `ItemResult` per row, like the CSV one) let a workflow run a test split
as input, which item 3 needs.

## 16. Streaming partial outputs

The builders stream lifecycle events, not tokens. Long transcriptions and descriptions would feel
alive with a `component_progress` event carrying LiteLLM's streamed chunks.
