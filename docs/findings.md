# Findings: running vLLM, sglang, and MLX-VLM with chai

A running log of platform constraints and surprises encountered while
integrating these three backends into `chai/ai/`. Recorded May 2026.

## TL;DR support matrix

| Backend | Linux + CUDA | Apple Silicon (M-series) |
|---|---|---|
| **vLLM** (`chai.ai.VLLMComponent`) | ✅ Full support, text + VL models | ⚠️ `vllm-metal` plugin only, **text-only** |
| **sglang** (`chai.ai.SGLangComponent`) | ✅ Full support, text + VL models | ⚠️ MLX backend (`SGLANG_USE_MLX=1`), **text-only verified** |
| **MLX-VLM** (`chai.ai.MLXVLMComponent`) | ❌ Apple-only by design | ✅ **The only native Mac path for VL models** |
| **LM Studio** (`chai.ai.LMStudioComponent`) | ✅ | ✅ |
| **Ollama** (`chai.ai.OllamaComponent`) | ✅ | ✅ |

Pragmatic rule of thumb on a Mac: use `MLXVLMComponent` for vision,
`VLLMComponent`/`SGLangComponent` for text (if you specifically want to
benchmark those engines).

## Discovery 1 -- vLLM-Metal is text-only

The community [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)
plugin adds an MLX-backed `Platform` to vLLM. As of v0.2.0, the
[Supported Models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/)
page is explicit:

> vllm-metal currently focuses on text-only language models on Apple
> Silicon. Multi-modal (vision / audio input) models are not yet supported.

This means `Qwen/Qwen3-VL-4B-Instruct` will not load via `vllm serve` on a
Mac. Use `MLXVLMComponent` instead.

## Discovery 2 -- sglang's MLX backend is brand new and text-validated

The [MLX execution backend PR](https://github.com/sgl-project/sglang/pull/20342)
was merged in March 2026. The PR description and CI benchmarks only
exercise text models (`Qwen3-0.6B`). No documented support for multimodal
models on the MLX path yet.

## Discovery 3 -- the chai Iterator wraps each yielded value

`chai.result.ListResult` iterates via `ResultIter`, which re-wraps each
element in the list's `valueClass` (default `ItemResult`):

```python
val = self.valueClass(self.result.value[self.count])
```

So inside a step that lives under `iterator.Iterator`, an `ItemResult(value="hello", metadata={"type": "TEXT"})` arrives as
`ItemResult(value=ItemResult(value="hello", metadata={"type": "TEXT"}))` --
the outer wrapper has no metadata. `FileItemResult` sidesteps this because
its `value` setter auto-infers `type` from the file extension.

We handle this in `OpenAIComponent._unwrap_typed` and
`MLXVLMComponent._unwrap_typed`: look one level into `item.value` for type
metadata, and infer `TEXT`/`IMAGE` from `str`/`bytes` as a last resort.

## Discovery 4 -- circular import if you import `chai.result` first

`chai.core` imports `chai.result` at module-bottom (`# noqa -- Prevent
circular import`) and `chai.result` imports `chai.core` at top. If a user
script does:

```python
from chai.result import ItemResult, ListResult  # triggers chai.core load
from chai.workflow import Workflow              # too late
```

...the second-pass `from .result import ListResult, Result` inside
`chai.core` hits a partially initialized `chai.result`. The fix is purely
ordering: import `chai.workflow` (or anything else that pulls in
`chai.core` from the top) **before** `chai.result`.

The shipped experiment scripts do this already.

## Discovery 5 -- sglang on macOS needs the `pyproject_other.toml` swap

sglang's repo ships multiple `pyproject_*.toml` variants. The default
`pyproject.toml` is the CUDA one. The macOS/MPS variant (which exposes the
`all_mps` / `srt_mps` extras with torch 2.11.0 + MLX + MLX-LM) lives in
`pyproject_other.toml`. Install on a Mac with:

```bash
cd ~/sglang-src/python
cp pyproject.toml pyproject.toml.cuda_backup
cp pyproject_other.toml pyproject.toml
cd ..
uv venv ~/.venv-sglang --python 3.11 --seed
VIRTUAL_ENV=$HOME/.venv-sglang uv pip install -e "python[all_mps]"
```

This is documented inside the [Apple Device Support roadmap issue](https://github.com/sgl-project/sglang/issues/19137)
but it's easy to miss.

## Discovery 6 -- Metal device is sandboxed away

If you spawn a vLLM/sglang server from a sandboxed shell, the Metal device
is unreachable and you get:

```
RuntimeError: [metal::load_device] No Metal device available. This typically
occurs in headless, sandboxed, or virtualized macOS sessions where the GPU
is not accessible.
```

Solution: start the server from a non-sandboxed terminal session.

## Discovery 7 -- MLX-VLM caches its weight load globally

`MLXVLMComponent` caches the loaded `(model, processor, config)` on the
class itself (`_model`, `_processor`, `_config`, `_loaded_model_name`).
Multiple component instances within one process reuse the load, but if you
switch `settings.model` mid-process the cache is rebuilt. This matches how
typical mlx-vlm scripts pattern around `mlx_vlm.load()`.

## Discovery 8 -- VL outputs sometimes loop

On dense archival pages, Qwen3-VL-4B (4-bit) occasionally repeats the same
sub-token sequence (e.g. `"Mangana, Ben 2/11\n"` × N). Mitigations:

- Bump `temperature` from 0.2 to 0.4-0.6.
- Set `max_output_tokens` tightly so a runaway doesn't waste minutes.
- For structured output, pin the model to JSON via `expected_output: "json"` and a JSON-shaped prompt; the schema constraint discourages loops.
- Use the 6-bit/8-bit quants for higher-density pages.

## Discovery 9 -- one client venv, multiple server venvs

vLLM and sglang on Apple Silicon pin conflicting versions of `torch`,
`transformers`, and `mlx-lm`. We separate environments by role:

| Venv | Python | Purpose |
|---|---|---|
| `~/.venv-chai` | 3.12 | Drives workflows. Has `openai`, `mlx-vlm`, chai deps. |
| `~/.venv-vllm-metal` | 3.12 | vLLM-Metal server only. Managed by official `install.sh`. |
| `~/.venv-sglang` | 3.11 | sglang MLX server only. Built from source. |

`chai/ai/__init__.py` imports each optional backend defensively so users
who only install some of these still get a working `chai.ai` namespace.
