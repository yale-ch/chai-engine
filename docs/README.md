# chai-engine docs

Backend guides for running open inference servers locally and pointing chai
workflows at them.

| Backend | chai component | LiteLLM id | Best on | Vision? |
|---|---|---|---|---|
| Any provider LiteLLM knows | `chai.ai.LLMComponent` | `<provider>/<model>` | -- | per model |
| Gemini / Vertex AI | `chai.ai.GeminiComponent` | `gemini/…` or `vertex_ai/…` (chosen from the environment) | -- | yes |
| [vLLM](./vllm.md) | `chai.ai.VLLMComponent` | `hosted_vllm/<model>` on `localhost:8000/v1` | NVIDIA + CUDA | yes on CUDA, no on Apple Silicon (vllm-metal is text-only) |
| [sglang](./sglang.md) | `chai.ai.SGLangComponent` | `openai/<model>` on `localhost:30000/v1` | NVIDIA + CUDA | yes on CUDA, not validated on MLX backend yet |
| [MLX-VLM](./mlx_vlm.md) | `chai.ai.MLXVLMComponent` | in-process (no server, no LiteLLM) | Apple Silicon | **yes -- the only native Mac option for VL models** |
| [LM Studio](https://lmstudio.ai/) | `chai.ai.LMStudioComponent` | `lm_studio/<model>` on `localhost:1234/v1` | Any | yes |
| [Ollama](https://ollama.com/) | `chai.ai.OllamaComponent` | `ollama_chat/<model>` on `localhost:11434` | Any | yes |
| Generic OpenAI-compatible | `chai.ai.OpenAIComponent` | `openai/<model>` with `api_host` | Any | yes |

System One decision models (Jev, Laya, GLiNER2.5-Decide) are covered in [system1.md](./system1.md);
what the engine still needs is in [roadmap.md](./roadmap.md).

Every component above is auto-mixed with the chai role base classes
(`Transcriber`, `Classifier`, `Segmenter`, ...) by
`chai.ai.create_all_components`, so in a workflow JSON you can write any of
these freely:

```json
{"type": "transcriber.VLLMTranscriber"}
{"type": "transcriber.SGLangTranscriber"}
{"type": "transcriber.MLXVLMTranscriber"}
{"type": "transcriber.LMStudioTranscriber"}
{"type": "transcriber.OpenAITranscriber"}
```

`{"type": "transcriber.LLMTranscriber", "settings": {"model": "anthropic/claude-sonnet-5-5"}}`
reaches any other provider. Vision backends need the server itself to serve a
vision model; the Mac notes in `mlx_vlm.md` explain why MLX-VLM is in-process.
