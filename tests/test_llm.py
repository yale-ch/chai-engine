"""Tests for the LiteLLM backend: the request each setting produces, and the result each reply
becomes. No network: ``litellm_params: {"mock_response": ...}`` makes LiteLLM answer locally, with
real ``ModelResponse`` objects, usage figures and prices from its own table."""

import base64
import io
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from chai.ai.llm import LLMComponent
from chai.ai.lm_studio import LMStudioComponent
from chai.ai.ollama import OllamaComponent
from chai.ai.openai import OpenAIComponent
from chai.ai.sglang import SGLangComponent
from chai.ai.vllm import VLLMComponent
from chai.result import FileItemResult, ItemResult, ListResult
from chai.workflow import Workflow


def build(cls, settings, cid="llm1"):
    wf = Workflow({"id": "llm_test_wf", "type": "workflow.Workflow"})
    return cls({"type": "ai.llm.LLMComponent", "id": cid, "settings": settings}, wf), wf


def text(value):
    return ItemResult(value, metadata={"type": "TEXT"})


class TestRequestShape(unittest.TestCase):
    def test_prompt_slots_and_literal_braces(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "name": "n", "prompt": 'Return {"k": 1} for {text_input_0} ({step_name})'})
        kwargs = c.request_kwargs(text("hello"))
        self.assertEqual(len(kwargs["messages"]), 1)
        content = kwargs["messages"][0]["content"]
        self.assertEqual(content, [{"type": "text", "text": f'Return {{"k": 1}} for hello ({c.name})'}])
        self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertNotIn("temperature", kwargs)  # not set -> not sent

    def test_unslotted_texts_are_attached_before_the_prompt(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "Compare the two texts."})
        kwargs = c.request_kwargs(ListResult([text("one"), text("two")]))
        content = kwargs["messages"][0]["content"]
        self.assertEqual([p["text"] for p in content], ["one", "two", "Compare the two texts."])

    def test_raw_string_input(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "Echo {text_input_0}"})
        self.assertEqual(c.build_messages("typed in")[0]["content"][0]["text"], "Echo typed in")

    def test_system_instruction_and_json_mode(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "p", "system_instruction": "Be terse.", "json_mode": True})
        kwargs = c.request_kwargs(text("x"))
        self.assertEqual(kwargs["messages"][0], {"role": "system", "content": "Be terse."})
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_schema_becomes_json_schema_response_format(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "p", "schema": schema})
        fmt = c.base_kwargs["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["schema"], schema)

    def test_reasoning_and_thinking_settings(self):
        c, _ = build(LLMComponent, {"model": "openai/o3-mini", "prompt": "p", "reasoning_effort": "high"})
        self.assertEqual(c.base_kwargs["reasoning_effort"], "high")
        c, _ = build(LLMComponent, {"model": "anthropic/claude-sonnet-5-5", "prompt": "p", "thinking_budget": 512})
        self.assertEqual(c.base_kwargs["thinking"], {"type": "enabled", "budget_tokens": 512})
        c, _ = build(LLMComponent, {"model": "anthropic/claude-sonnet-5-5", "prompt": "p", "thinking_budget": 0})
        self.assertEqual(c.base_kwargs["thinking"]["type"], "disabled")

    def test_timeout_api_key_extra_body_and_passthrough(self):
        c, _ = build(
            LLMComponent,
            {"model": "openai/gpt-4o-mini", "prompt": "p", "timeout": 12, "api_key": "k", "extra_body": {"top_k": 5}, "litellm_params": {"drop_params": True}},
        )
        self.assertEqual(c.base_kwargs["timeout"], 12.0)
        self.assertEqual(c.base_kwargs["api_key"], "k")
        self.assertEqual(c.base_kwargs["extra_body"], {"top_k": 5})
        self.assertTrue(c.base_kwargs["drop_params"])

    def test_unknown_tool_fails_loudly(self):
        with self.assertRaises(ValueError):
            build(LLMComponent, {"model": "gemini/gemini-3.1-flash-lite", "prompt": "p", "tools": ["maps"]})

    def test_unknown_model_fails_at_build(self):
        with self.assertRaises(ValueError):
            build(LLMComponent, {"model": "no-such-model-xyz", "prompt": "p"})

    def test_missing_model_fails_at_build(self):
        with self.assertRaises(ValueError):
            build(LLMComponent, {"prompt": "p"})

    def test_empty_prompt_fails_at_run(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini"})
        with self.assertRaises(ValueError):
            c.build_messages(text("x"))


class TestMediaParts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "pic.jpg")
        Image.new("RGB", (64, 32), "red").save(self.path, format="JPEG")

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_image_becomes_data_url_with_its_mime(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "Transcribe"})
        kwargs = c.request_kwargs(FileItemResult(self.path))
        parts = kwargs["messages"][0]["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(parts[1], {"type": "text", "text": "Transcribe"})

    def test_downscale_and_detail(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "p", "max_image_size": 16, "image_detail": "low"})
        part = c.image_part(FileItemResult(self.path))
        self.assertEqual(part["image_url"]["detail"], "low")
        header, b64 = part["image_url"]["url"].split(",", 1)
        self.assertEqual(header, "data:image/png;base64")
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        self.assertLessEqual(max(img.size), 16)

    def test_remote_and_gs_references_pass_through(self):
        c, _ = build(LLMComponent, {"model": "gemini/gemini-3.1-flash-lite", "prompt": "p"})
        part = c.image_part(ItemResult("gs://bucket/scan.png", metadata={"type": "IMAGE"}))
        self.assertEqual(part["image_url"]["url"], "gs://bucket/scan.png")

    def test_iterator_rewrapped_item_keeps_its_type(self):
        inner = ItemResult(b"\x89PNG", metadata={"type": "IMAGE"})
        typ, value = LLMComponent.unwrap_typed(ItemResult(inner))
        self.assertEqual((typ, value), ("IMAGE", b"\x89PNG"))
        self.assertEqual(LLMComponent.unwrap_typed(ItemResult("plain"))[0], "TEXT")

    def test_prompt_counters(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "{input_length} images, first {first_input}, last {last_input}"})
        prompt, parts = c.build_contents(ListResult([FileItemResult(self.path), FileItemResult(self.path)]))
        self.assertEqual(prompt, "2 images, first pic.jpg, last pic.jpg")
        self.assertEqual(len(parts), 2)


class TestBackendDefaults(unittest.TestCase):
    def test_ollama(self):
        c, _ = build(OllamaComponent, {"prompt": "p"})
        self.assertEqual(c.model, "ollama_chat/llama3.2")
        self.assertEqual(c.base_kwargs["api_base"], "http://localhost:11434")
        self.assertEqual(c.base_kwargs["temperature"], 0.4)
        self.assertEqual(c.base_kwargs["max_tokens"], 20000)
        c, _ = build(OllamaComponent, {"prompt": "p", "model": "ollama/llava", "api_host": "gpu-box:11434"})
        self.assertEqual(c.model, "ollama/llava")  # an explicit provider prefix is kept
        self.assertEqual(c.base_kwargs["api_base"], "http://gpu-box:11434")

    def test_lm_studio(self):
        c, _ = build(LMStudioComponent, {"prompt": "p"})
        self.assertEqual(c.model, "lm_studio/qwen/qwen3.5-9b")  # 'qwen' is an org, not a provider
        self.assertEqual(c.base_kwargs["api_base"], "http://localhost:1234/v1")
        self.assertEqual(c.base_kwargs["api_key"], "EMPTY")
        self.assertEqual(c.base_kwargs["max_tokens"], 32000)

    def test_openai_and_compatible_servers(self):
        c, _ = build(OpenAIComponent, {"prompt": "p", "model": "gpt-4o-mini"})
        self.assertEqual(c.model, "openai/gpt-4o-mini")
        self.assertNotIn("api_base", c.base_kwargs)  # the real API; LiteLLM reads OPENAI_API_KEY itself
        c, _ = build(OpenAIComponent, {"prompt": "p", "model": "my-model", "api_host": "myhost:8080"})
        self.assertEqual(c.base_kwargs["api_base"], "http://myhost:8080/v1")
        self.assertEqual(c.base_kwargs["api_key"], "EMPTY")

    def test_vllm_and_sglang(self):
        c, _ = build(VLLMComponent, {"prompt": "p", "model": "Qwen/Qwen3-8B"})
        self.assertEqual(c.model, "hosted_vllm/Qwen/Qwen3-8B")
        self.assertEqual(c.base_kwargs["api_base"], "http://localhost:8000/v1")
        c, _ = build(SGLangComponent, {"prompt": "p", "model": "Qwen/Qwen3-8B"})
        self.assertEqual(c.model, "openai/Qwen/Qwen3-8B")
        self.assertEqual(c.base_kwargs["api_base"], "http://localhost:30000/v1")
        self.assertEqual(c.ENGINE_NAME, "sglang")


class TestProcessWithMockResponses(unittest.TestCase):
    def test_json_reply_usage_and_price(self):
        c, wf = build(LLMComponent, {"model": "gemini/gemini-3.1-flash-lite", "prompt": "p", "litellm_params": {"mock_response": '{"a": 2}'}})
        out = c.process(text("x"))
        self.assertEqual(out.value, {"a": 2})
        self.assertEqual(out.metadata["type"], "DATA")
        usage = out.metadata["token_usage"]
        self.assertGreater(usage["prompt"], 0)
        self.assertGreater(usage["result"], 0)
        self.assertEqual(usage["total"], usage["prompt"] + usage["result"])
        self.assertGreater(out.metadata["cost"], 0)  # Gemini is in LiteLLM's price table
        self.assertEqual(out.metadata["provider"], "gemini")
        self.assertEqual(out.metadata["model"], "gemini/gemini-3.1-flash-lite")
        self.assertEqual(wf.usage.summary()["calls"], 1)

    def test_text_and_yaml_outputs(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "p", "expected_output": "text", "litellm_params": {"mock_response": "just words"}})
        out = c.process(text("x"))
        self.assertEqual((out.value, out.metadata["type"]), ("just words", "TEXT"))
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "p", "expected_output": "yaml", "litellm_params": {"mock_response": "a: 1\nb: [x]"}})
        self.assertEqual(c.process(text("x")).value, {"a": 1, "b": ["x"]})

    def test_think_block_and_code_fence_are_stripped(self):
        c, _ = build(LLMComponent, {"model": "openai/gpt-4o-mini", "prompt": "p", "litellm_params": {"mock_response": '<think>hmm</think>\n```json\n{"a": 1}\n```'}})
        self.assertEqual(c.process(text("x")).value, {"a": 1})

    def test_local_models_are_free_and_pricing_overrides(self):
        c, _ = build(OllamaComponent, {"prompt": "p", "litellm_params": {"mock_response": "{}"}})
        self.assertEqual(c.process(text("x")).metadata["cost"], 0.0)
        c, _ = build(OllamaComponent, {"prompt": "p", "pricing": {"input_per_million": 1.0, "output_per_million": 2.0}, "litellm_params": {"mock_response": "{}"}})
        out = c.process(text("x"))
        usage = out.metadata["token_usage"]
        self.assertAlmostEqual(out.metadata["cost"], (usage["prompt"] * 1.0 + usage["result"] * 2.0) / 1_000_000)

    def test_role_composite_in_a_workflow(self):
        wf = Workflow(
            {
                "type": "Workflow",
                "id": "wf",
                "steps": [
                    {
                        "type": "extractor.LLMExtractor",
                        "id": "ner",
                        "settings": {"model": "openai/gpt-4o-mini", "prompt": "Entities in {text_input_0}", "litellm_params": {"mock_response": '{"entities": ["Ada"]}'}},
                    }
                ],
            }
        )
        res = wf.run("Ada Lovelace wrote notes.")
        self.assertEqual(res.value[0].value, {"entities": ["Ada"]})
        self.assertEqual(res.value[0].processor.id, "ner")
        self.assertEqual(wf.usage_summary()["by_component"]["ner"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
