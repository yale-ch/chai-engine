"""Request-config tests for GeminiComponent across model generations.

Gemini 2.5 takes a thinking token budget; Gemini 3+ replaces it with `thinking_level`, which
LiteLLM carries as `reasoning_effort`. For 3.x models no sampling defaults are sent (Google
recommends the API default), but an explicitly configured temperature/top_p is always honored --
verified live against gemini-3.7-flash on Vertex, which accepts them. These tests pin the
``litellm.completion`` call the component builds for each generation without talking to the API.
"""

import os
import unittest
from unittest import mock

from chai.ai.gemini import GeminiComponent
from chai.result import ItemResult
from chai.workflow import Workflow

API_ENV = {"GEMINI_API_KEY": "test-key", "GOOGLE_API_KEY": "", "GOOGLE_CLOUD_PROJECT": "", "VERTEXAI_PROJECT": ""}
VERTEX_ENV = {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "GOOGLE_CLOUD_PROJECT": "my-project", "VERTEXAI_PROJECT": ""}
NO_ENV = {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "GOOGLE_CLOUD_PROJECT": "", "VERTEXAI_PROJECT": ""}


def build_gemini(settings, env=API_ENV):
    """Construct a GeminiComponent with fake credentials; nothing is sent anywhere."""
    wf = Workflow({"id": "gemini_test_wf", "type": "workflow.Workflow"})
    tree = {"type": "ai.gemini.GeminiComponent", "id": "g1", "settings": {"prompt": "Describe {text_input_0}", **settings}}
    with mock.patch.dict(os.environ, env):
        return GeminiComponent(tree, wf)


class TestModelVersion(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(GeminiComponent._model_version("gemini-3.7-flash"), (3, 7))
        self.assertEqual(GeminiComponent._model_version("vertex_ai/gemini-3.7-flash"), (3, 7))
        self.assertEqual(GeminiComponent._model_version("gemini-3-flash-preview"), (3, 0))
        self.assertEqual(GeminiComponent._model_version("gemini/gemini-3.1-flash-lite-preview"), (3, 1))
        self.assertEqual(GeminiComponent._model_version("gemini-2.5-flash"), (2, 5))
        self.assertEqual(GeminiComponent._model_version("my-tuned-endpoint"), (0, 0))


class TestGemini37(unittest.TestCase):
    def test_thinking_level_default_low(self):
        c = build_gemini({"model": "gemini-3.7-flash"})
        self.assertEqual(c.base_kwargs["reasoning_effort"], "low")
        self.assertNotIn("thinking", c.base_kwargs)

    def test_thinking_level_setting(self):
        c = build_gemini({"model": "gemini-3.7-flash", "thinking_level": "high"})
        self.assertEqual(c.base_kwargs["reasoning_effort"], "high")

    def test_no_sampling_knobs_sent(self):
        c = build_gemini({"model": "gemini-3.7-flash"})
        self.assertNotIn("temperature", c.base_kwargs)
        self.assertNotIn("top_p", c.base_kwargs)

    def test_explicit_sampling_knobs_honored(self):
        # temperature=0.0 is the reproducible-extraction case: it must survive
        # both the "only when explicit" rule and any falsy-value check.
        c = build_gemini({"model": "gemini-3.7-flash", "temperature": 0.0, "top_p": 0.9})
        self.assertEqual(c.base_kwargs["temperature"], 0.0)
        self.assertEqual(c.base_kwargs["top_p"], 0.9)

    def test_thinking_budget_ignored_with_warning(self):
        with self.assertLogs("chai", level="WARNING") as logs:
            c = build_gemini({"model": "gemini-3.7-flash", "thinking_budget": 512})
        self.assertEqual(c.base_kwargs["reasoning_effort"], "low")
        self.assertNotIn("thinking", c.base_kwargs)
        self.assertTrue(any("thinking_level" in m for m in logs.output))


class TestGemini30to35(unittest.TestCase):
    def test_sampling_only_when_explicit(self):
        c = build_gemini({"model": "gemini-3.1-flash-lite-preview"})
        self.assertNotIn("temperature", c.base_kwargs)
        self.assertNotIn("top_p", c.base_kwargs)
        c = build_gemini({"model": "gemini-3.1-flash-lite-preview", "temperature": 0.7})
        self.assertEqual(c.base_kwargs["temperature"], 0.7)
        self.assertNotIn("top_p", c.base_kwargs)

    def test_thinking_level_applies(self):
        c = build_gemini({"model": "gemini-3-flash-preview", "thinking_level": "medium"})
        self.assertEqual(c.base_kwargs["reasoning_effort"], "medium")


class TestGemini25(unittest.TestCase):
    def test_thinking_budget_default_off(self):
        c = build_gemini({"model": "gemini-2.5-flash"})
        self.assertEqual(c.base_kwargs["thinking"], {"type": "disabled", "budget_tokens": 0})
        self.assertNotIn("reasoning_effort", c.base_kwargs)

    def test_thinking_budget_setting(self):
        c = build_gemini({"model": "gemini-2.5-flash", "thinking_budget": 1024})
        self.assertEqual(c.base_kwargs["thinking"], {"type": "enabled", "budget_tokens": 1024})

    def test_sampling_defaults_still_sent(self):
        c = build_gemini({"model": "gemini-2.5-flash"})
        self.assertEqual(c.base_kwargs["temperature"], 0.4)
        self.assertEqual(c.base_kwargs["top_p"], 0.9)


class TestUnversionedModel(unittest.TestCase):
    def test_legacy_request_shape(self):
        c = build_gemini({"model": "my-tuned-endpoint"})
        self.assertEqual(c.base_kwargs["temperature"], 0.4)
        self.assertNotIn("thinking", c.base_kwargs)
        self.assertNotIn("reasoning_effort", c.base_kwargs)


class TestSystemInstructionAndMime(unittest.TestCase):
    def test_pass_through(self):
        c = build_gemini(
            {
                "model": "gemini-3.7-flash",
                "system_instruction": "You are a cataloguer.",
                "response_mime_type": "application/json",
            }
        )
        kwargs = c.request_kwargs(ItemResult("a letter", metadata={"type": "TEXT"}))
        self.assertEqual(kwargs["messages"][0], {"role": "system", "content": "You are a cataloguer."})
        self.assertEqual(kwargs["messages"][1]["content"][-1]["text"], "Describe a letter")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


class TestCredentialsAndProvider(unittest.TestCase):
    def test_api_key_selects_gemini_api(self):
        c = build_gemini({})
        self.assertEqual(c.model, "gemini/gemini-3.1-flash-lite-preview")
        self.assertEqual(c.base_kwargs["api_key"], "test-key")
        self.assertNotIn("vertex_project", c.base_kwargs)

    def test_project_selects_vertex(self):
        c = build_gemini({"model": "gemini-3.7-flash", "location": "us-central1"}, env=VERTEX_ENV)
        self.assertEqual(c.model, "vertex_ai/gemini-3.7-flash")
        self.assertEqual(c.base_kwargs["vertex_project"], "my-project")
        self.assertEqual(c.base_kwargs["vertex_location"], "us-central1")
        self.assertNotIn("api_key", c.base_kwargs)

    def test_vertex_location_defaults_to_global(self):
        c = build_gemini({}, env=VERTEX_ENV)
        self.assertEqual(c.base_kwargs["vertex_location"], "global")

    def test_prefixed_model_id_is_kept(self):
        c = build_gemini({"model": "gemini/gemini-3.1-flash-lite"}, env=VERTEX_ENV)
        self.assertEqual(c.model, "gemini/gemini-3.1-flash-lite")

    def test_no_credentials_fails_at_build(self):
        with self.assertRaises(ValueError):
            build_gemini({}, env=NO_ENV)


class TestGeminiDefaults(unittest.TestCase):
    def test_grounding_on_by_default_and_off_with_empty_tools(self):
        c = build_gemini({"model": "gemini-3.7-flash"})
        self.assertEqual(c.base_kwargs["tools"], [{"googleSearch": {}}])
        c = build_gemini({"model": "gemini-3.7-flash", "tools": []})
        self.assertNotIn("tools", c.base_kwargs)
        c = build_gemini({"model": "gemini-3.7-flash", "tools": ["search", "url"]})
        self.assertEqual(c.base_kwargs["tools"], [{"googleSearch": {}}, {"urlContext": {}}])

    def test_safety_off_with_per_category_override(self):
        c = build_gemini({"model": "gemini-3.7-flash", "harassment_safety": "BLOCK_ONLY_HIGH"})
        thresholds = {s["category"]: s["threshold"] for s in c.base_kwargs["safety_settings"]}
        self.assertEqual(len(thresholds), 4)
        self.assertEqual(thresholds["HARM_CATEGORY_HARASSMENT"], "BLOCK_ONLY_HIGH")
        self.assertEqual(thresholds["HARM_CATEGORY_HATE_SPEECH"], "OFF")

    def test_output_cap(self):
        c = build_gemini({"model": "gemini-3.7-flash"})
        self.assertEqual(c.base_kwargs["max_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
