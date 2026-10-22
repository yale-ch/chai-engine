"""Ollama backend for chai.

Talks to a local `Ollama <https://ollama.com>`_ server (default ``localhost:11434``) through LiteLLM's
``ollama_chat`` provider. Role mixins such as ``OllamaClassifier`` are generated from
``OllamaComponent`` by ``chai.ai.create_all_components``.
"""

from .llm import LLMComponent


class OllamaComponent(LLMComponent):
    """A local Ollama model, with the defaults chai has used for it, on top of ``LLMComponent``.

    Settings:
        - api_host: host:port (or URL) of the Ollama server (default 'localhost:11434')
        - model: Ollama model name, e.g. 'llama3.2' or 'qwen2.5vl' (default 'llama3.2';
          'nomic-embed-text' in an embedding role)
        - temperature: sampling temperature (default 0.4)
        - top_p: nucleus sampling threshold (default 0.9)
        - max_output_tokens: response token cap (default 20000)
        - plus every LLMComponent setting (prompt, expected_output, system_instruction, timeout, pricing, ...)
    """

    PROVIDER = "ollama_chat"
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_API_BASE = "http://localhost:11434"
    DEFAULT_TEMPERATURE = 0.4
    DEFAULT_TOP_P = 0.9
    DEFAULT_MAX_OUTPUT_TOKENS = 20000
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_PROVIDER = "ollama"  # LiteLLM embeds through "ollama", not "ollama_chat"
    ENGINE_NAME = "ollama"
