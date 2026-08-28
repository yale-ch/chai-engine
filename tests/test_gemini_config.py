"""Request-config tests for GeminiComponent across model generations.

Gemini 2.5 takes a thinking token budget; Gemini 3+ replaces it with `thinking_level`. For 3.x
models no sampling defaults are sent (Google recommends the API default), but an explicitly
configured temperature/top_p is always honored -- verified live against gemini-3.7-flash on
Vertex, which accepts them. These tests pin the config the component builds for each
generation without talking to the API.
"""

import os
import unittest
from unittest import mock

from chai.ai.gemini import GeminiComponent
from chai.workflow import Workflow

FAKE_ENV = {"GEMINI_API_KEY": "test-key", "GOOGLE_CLOUD_PROJECT": ""}


def level_of(component):
    """The configured thinking level as a lowercase string (the SDK stores an enum)."""
    lvl = component.base_config.thinking_config.thinking_level
    return str(getattr(lvl, "value", lvl)).lower()


def build_gemini(settings):
    """Construct a GeminiComponent with fake credentials and no client connection."""
    wf = Workflow({"id": "gemini_test_wf", "type": "workflow.Workflow"})
    tree = {"type": "ai.gemini.GeminiComponent", "id": "g1", "settings": settings}
    with mock.patch.dict(os.environ, FAKE_ENV):
        with mock.patch.object(GeminiComponent, "connect_to_client", lambda self: None):
            return GeminiComponent(tree, wf)


class TestModelVersion(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(GeminiComponent._model_version("gemini-3.7-flash"), (3, 7))
        self.assertEqual(GeminiComponent._model_version("gemini-3-flash-preview"), (3, 0))
        self.assertEqual(GeminiComponent._model_version("gemini-3.1-flash-lite-preview"), (3, 1))
        self.assertEqual(GeminiComponent._model_version("gemini-2.5-flash"), (2, 5))
        self.assertEqual(GeminiComponent._model_version("my-tuned-endpoint"), (0, 0))


class TestGemini37(unittest.TestCase):
    def test_thinking_level_default_low(self):
        c = build_gemini({"model": "gemini-3.7-flash"})
        self.assertEqual(level_of(c), "low")
        self.assertIsNone(c.base_config.thinking_config.thinking_budget)

    def test_thinking_level_setting(self):
        c = build_gemini({"model": "gemini-3.7-flash", "thinking_level": "high"})
        self.assertEqual(level_of(c), "high")

    def test_no_sampling_knobs_sent(self):
        c = build_gemini({"model": "gemini-3.7-flash"})
        self.assertIsNone(c.base_config.temperature)
        self.assertIsNone(c.base_config.top_p)

    def test_explicit_sampling_knobs_honored(self):
        # temperature=0.0 is the reproducible-extraction case: it must survive
        # both the "only when explicit" rule and any falsy-value check.
        c = build_gemini({"model": "gemini-3.7-flash", "temperature": 0.0, "top_p": 0.9})
        self.assertEqual(c.base_config.temperature, 0.0)
        self.assertEqual(c.base_config.top_p, 0.9)

    def test_thinking_budget_ignored_with_warning(self):
        with self.assertLogs("chai", level="WARNING") as logs:
            c = build_gemini({"model": "gemini-3.7-flash", "thinking_budget": 512})
        self.assertEqual(level_of(c), "low")
        self.assertTrue(any("thinking_level" in m for m in logs.output))


class TestGemini30to35(unittest.TestCase):
    def test_sampling_only_when_explicit(self):
        c = build_gemini({"model": "gemini-3.1-flash-lite-preview"})
        self.assertIsNone(c.base_config.temperature)
        self.assertIsNone(c.base_config.top_p)
        c = build_gemini({"model": "gemini-3.1-flash-lite-preview", "temperature": 0.7})
        self.assertEqual(c.base_config.temperature, 0.7)
        self.assertIsNone(c.base_config.top_p)

    def test_thinking_level_applies(self):
        c = build_gemini({"model": "gemini-3-flash-preview", "thinking_level": "medium"})
        self.assertEqual(level_of(c), "medium")


class TestGemini25(unittest.TestCase):
    def test_thinking_budget_default_off(self):
        c = build_gemini({"model": "gemini-2.5-flash"})
        self.assertEqual(c.base_config.thinking_config.thinking_budget, 0)
        self.assertIsNone(c.base_config.thinking_config.thinking_level)

    def test_thinking_budget_setting(self):
        c = build_gemini({"model": "gemini-2.5-flash", "thinking_budget": 1024})
        self.assertEqual(c.base_config.thinking_config.thinking_budget, 1024)

    def test_sampling_defaults_still_sent(self):
        c = build_gemini({"model": "gemini-2.5-flash"})
        self.assertEqual(c.base_config.temperature, 0.4)
        self.assertEqual(c.base_config.top_p, 0.9)


class TestUnversionedModel(unittest.TestCase):
    def test_legacy_request_shape(self):
        c = build_gemini({"model": "my-tuned-endpoint"})
        self.assertEqual(c.base_config.temperature, 0.4)
        self.assertIsNone(c.base_config.thinking_config)


class TestSystemInstructionAndMime(unittest.TestCase):
    def test_pass_through(self):
        c = build_gemini(
            {
                "model": "gemini-3.7-flash",
                "system_instruction": "You are a cataloguer.",
                "response_mime_type": "application/json",
            }
        )
        self.assertEqual(c.base_config.system_instruction, "You are a cataloguer.")
        self.assertEqual(c.base_config.response_mime_type, "application/json")

    def test_request_config_per_call_override(self):
        c = build_gemini({"model": "gemini-3.7-flash", "system_instruction": "standing"})
        override = c.request_config(system_instruction="per-call")
        self.assertEqual(override.system_instruction, "per-call")
        # the configured request is untouched
        self.assertEqual(c.base_config.system_instruction, "standing")
        self.assertIs(c.request_config(None), c.base_config)


if __name__ == "__main__":
    unittest.main()
