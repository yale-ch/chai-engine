# sglang backend

`chai.ai.SGLangComponent` (and the auto-generated `SGLangTranscriber`,
`SGLangClassifier`, ...) talks to an
[sglang](https://github.com/sgl-project/sglang) server through its
OpenAI-compatible HTTP API.

## Component reference

```python
{
    "type": "transcriber.SGLangTranscriber",
    "settings": {
        "api_host": "localhost:30000",          # sglang default
        "model":    "Qwen/Qwen3-VL-4B-Instruct",
        "max_image_size": 1536,
        "max_output_tokens": 2048,
        "temperature": 0.2,
        "expected_output": "text"
    }
}
```

Same setting catalogue as `OpenAIComponent`; the only thing
`SGLangComponent` itself does is preset `DEFAULT_API_HOST = "localhost:30000"`.

## Server setup

### NVIDIA + CUDA (recommended for vision-language models)

The standard supported path. Vision models like
`Qwen/Qwen3-VL-4B-Instruct` work here.

```bash
# Once
python -m venv ~/.venv-sglang
source ~/.venv-sglang/bin/activate
pip install --upgrade pip
pip install "sglang[all]>=0.5.0"

# Each session
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code
```

Driving the workflow from a dev box:

```bash
SGLANG_API_HOST=my-gpu-host:30000 \
SGLANG_MODEL=Qwen/Qwen3-VL-4B-Instruct \
python experiment_sglang.py
```

### Apple Silicon Mac (text-only, via MLX backend)

sglang added a native MLX execution backend in March 2026 (see PR
[#20342](https://github.com/sgl-project/sglang/pull/20342)). The
roadmap-tracking issue is [#19137](https://github.com/sgl-project/sglang/issues/19137).
**It is currently validated only against text models like
`Qwen/Qwen3-0.6B`.** Vision-language models are not on the MLX path yet --
see [findings.md](./findings.md).

```bash
# Once
brew install ffmpeg
git clone --depth 1 https://github.com/sgl-project/sglang.git ~/sglang-src
# Swap to the macOS/MPS pyproject (the default targets CUDA)
( cd ~/sglang-src/python \
    && cp pyproject.toml pyproject.toml.cuda_backup \
    && cp pyproject_other.toml pyproject.toml )

uv venv ~/.venv-sglang --python 3.11 --seed
( cd ~/sglang-src && VIRTUAL_ENV=$HOME/.venv-sglang uv pip install -e "python[all_mps]" )

# Each session
SGLANG_USE_MLX=1 ~/.venv-sglang/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code

# Smoke-test the chai integration
~/.venv-chai/bin/python experiment_sglang_text.py
```

Notes:
- **Python 3.11 is required.** sglang's MLX path is only verified there.
- The `pyproject_other.toml` swap is what unlocks the `all_mps` extra
  (`srt_mps` + `diffusion_mps` -- torch 2.11.0, MLX, MLX-LM).
- Pass `--trust-remote-code` for Qwen3 models (the chat template lives in
  the model repo).

For vision models on a Mac, use [`MLXVLMComponent`](./mlx_vlm.md) instead.

## Tested combinations

| Environment | Model | Outcome |
|---|---|---|
| Apple Silicon (M-series) + sglang `0.0.0.dev1+g81d686d9f` (MLX backend, May 2026) | `Qwen/Qwen3-0.6B` | ✅ 10/10 chat completions; MLX model loaded in 0.69 s; ~220-270 tok/s decode |
| Apple Silicon + sglang MLX backend | `Qwen/Qwen3-VL-4B-Instruct` | ❌ not on the verified MLX path; expected to fail or fall back |
| NVIDIA + sglang (mainline) | `Qwen/Qwen3-VL-4B-Instruct` | ✅ Standard supported path; see PR notes about `mrope_section`/`yarn` if you hit decoder-config errors |
