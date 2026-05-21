# chai-engine docs

Backend guides for running open inference servers locally and pointing chai
workflows at them.

| Backend | chai component | Wire protocol | Best on | Vision? |
|---|---|---|---|---|
| [vLLM](./vllm.md) | `chai.ai.VLLMComponent` | OpenAI `/v1` | NVIDIA + CUDA | yes on CUDA, no on Apple Silicon (vllm-metal is text-only) |
| [sglang](./sglang.md) | `chai.ai.SGLangComponent` | OpenAI `/v1` | NVIDIA + CUDA | yes on CUDA, not validated on MLX backend yet |
| [MLX-VLM](./mlx_vlm.md) | `chai.ai.MLXVLMComponent` | In-process (no server) | Apple Silicon | **yes -- the only native Mac option for VL models** |
| [LM Studio](https://lmstudio.ai/) | `chai.ai.LMStudioComponent` | LM Studio native SDK (WebSocket on 1234) | Any | yes |
| [Ollama](https://ollama.com/) | `chai.ai.OllamaComponent` | Ollama native API on 11434 | Any | yes |
| Generic OpenAI | `chai.ai.OpenAIComponent` | OpenAI `/v1` | Any | yes |

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

Read [`findings.md`](./findings.md) first if you're on a Mac -- there are
non-obvious platform constraints that drove the design.
