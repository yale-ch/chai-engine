# vLLM backend

`chai.ai.VLLMComponent` (and the auto-generated `VLLMTranscriber`,
`VLLMClassifier`, ...) talks to a [vLLM](https://github.com/vllm-project/vllm)
server through its OpenAI-compatible HTTP API.

## Component reference

```python
{
    "type": "transcriber.VLLMTranscriber",
    "settings": {
        "api_host": "localhost:8000",           # vLLM default
        "model":    "Qwen/Qwen3-VL-4B-Instruct",
        "max_image_size": 1536,                  # downscale long edge before send
        "max_output_tokens": 2048,
        "temperature": 0.2,
        "expected_output": "text"                # or "json"
    }
}
```

Settings inherited from `OpenAIComponent`: `api_host`, `api_key`, `model`,
`temperature`, `top_p`, `max_output_tokens`, `max_image_size`,
`image_mime_type`, `image_detail`, `timeout`, `prompt`, `expected_output`,
`extra_body`. The only thing `VLLMComponent` itself does is preset
`DEFAULT_API_HOST = "localhost:8000"`.

## Server setup

### NVIDIA + CUDA (recommended for vision-language models)

This is the standard, fully-supported path. Vision models like
`Qwen/Qwen3-VL-4B-Instruct` work here.

```bash
# Once
python -m venv ~/.venv-vllm
source ~/.venv-vllm/bin/activate
pip install --upgrade pip
pip install vllm

# Each session
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```

A 24 GB GPU (e.g. RTX 3090, 4090, A10) is comfortable for the 4B VL model.

Then from your dev box:

```bash
VLLM_API_HOST=my-gpu-host:8000 \
VLLM_MODEL=Qwen/Qwen3-VL-4B-Instruct \
python experiment_vllm.py
```

### Apple Silicon Mac (text-only, via vllm-metal plugin)

`vllm-metal` is a community plugin that adds an MLX-backed platform to vLLM
on Apple Silicon. **It is text-only** -- multimodal models do not load
(see [findings.md](./findings.md)).

```bash
# Once -- builds vllm 0.21.0 from source, installs vllm-metal 0.2.0 wheel,
# creates ~/.venv-vllm-metal (uv-based)
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

# Each session
~/.venv-vllm-metal/bin/vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --max-model-len 4096

# Smoke-test the chai integration
~/.venv-chai/bin/python experiment_vllm_text.py
```

Requires Python 3.12 in the server venv. The installer manages this for you.

For vision models on a Mac, use [`MLXVLMComponent`](./mlx_vlm.md) instead.

### Docker (CPU only, slow, but works for VL on Mac)

```bash
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest-cpu \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --trust-remote-code
```

Useful as a fallback if you need a VL model on a Mac and don't want to use
MLX-VLM. Expect single-digit tokens/s.

## Tested combinations

| Environment | Model | Outcome |
|---|---|---|
| Apple Silicon (M-series) + vllm-metal 0.2.0 | `Qwen/Qwen3-0.6B` | ✅ 10/10 chat completions in ~25 s, ~64 tok/s decode on Metal |
| Apple Silicon + vllm-metal 0.2.0 | `Qwen/Qwen3-VL-4B-Instruct` | ❌ refused -- multimodal not supported by vllm-metal |
| NVIDIA + vLLM (mainline) | `Qwen/Qwen3-VL-4B-Instruct` | ✅ Standard supported path |
