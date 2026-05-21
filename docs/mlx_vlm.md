# MLX-VLM backend

`chai.ai.MLXVLMComponent` (and the auto-generated `MLXVLMTranscriber`,
`MLXVLMClassifier`, ...) runs vision-language models **in-process** on
Apple Silicon via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) -- no HTTP
server.

This is the only chai backend today that can run multimodal models like
`Qwen/Qwen3-VL-4B-Instruct` natively on a Mac. vllm-metal and sglang+MLX
are both text-only on Apple Silicon (see [findings.md](./findings.md)).

## Component reference

```python
{
    "type": "transcriber.MLXVLMTranscriber",
    "settings": {
        "model": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "max_image_size": 1536,       # downscale long edge before VLM
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "expected_output": "text",
        "verbose": false               # mlx-vlm streaming verbose
    }
}
```

Optional extras:
- `prompt` -- overrides chai's default transcription prompt.
- `max_image_size` -- if set, the long edge is resized before encoding.

The component caches the loaded `(model, processor, config)` on the class so
multiple `MLXVLMComponent` instances created within one workflow reuse a
single weight load.

## Model selection

`mlx-vlm` accepts any model from the [mlx-community](https://huggingface.co/mlx-community)
org that's been converted to the MLX format. For Qwen3-VL-4B the verified
options are:

| HF repo | Bits | Disk | Notes |
|---|---|---|---|
| `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 4 | ~3 GB | Default. Fastest on M-series, mild quality loss on dense OCR. |
| `mlx-community/Qwen3-VL-4B-Instruct-6bit` | 6 | ~4 GB | Better quality. |
| `mlx-community/Qwen3-VL-4B-Instruct-8bit` | 8 | ~5 GB | Best quality, slowest. |

For other Qwen3-VL sizes browse https://huggingface.co/mlx-community?search=Qwen3-VL .

## Install

```bash
# In the chai client venv (Python 3.10+):
pip install mlx-vlm
# or via uv
VIRTUAL_ENV=$HOME/.venv-chai uv pip install mlx-vlm
```

`mlx-vlm` only installs on Apple Silicon (`darwin` + `arm64`). `chai.ai`
imports it defensively; on Linux/Windows you'll see an info-level message
that `MLXVLMComponent` is unavailable, and the rest of chai keeps working.

## Run

```bash
~/.venv-chai/bin/python experiment_mlx_vlm.py
```

Override the model via env var:

```bash
MLX_VLM_MODEL=mlx-community/Qwen3-VL-4B-Instruct-6bit \
  ~/.venv-chai/bin/python experiment_mlx_vlm.py
```

## Tested combinations

| Environment | Model | Outcome |
|---|---|---|
| Apple Silicon M-series + mlx-vlm 0.5.0 + chai client venv | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | ✅ 10/10 images transcribed in ~3.5 min including a one-time ~67 s model download (~3 GB); real archival OCR output. Occasional decode loops on dense pages (mitigatable with higher temperature or a JSON-constrained prompt). |
| Apple Silicon + mlx-vlm | `mlx-community/Qwen3-VL-4B-Instruct-6bit` / `-8bit` | ✅ Same wiring; higher quality, ~30-50% slower. |

## Notes on prompts

When using `MLXVLMTranscriber` you inherit chai's default transcription
prompt (`data/prompts.json::transcription`). It explicitly steers the model
toward archival-faithful OCR. You can override it per-step via
`settings.prompt` if you need a different task (captioning, table
extraction, structured JSON, ...).

For JSON output, set `"expected_output": "json"` and shape your prompt to
emit JSON. `MLXVLMComponent._process` will route the model output through
`chai.ai.ai_utils.extract_json` exactly like the other backends.
