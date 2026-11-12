"""The device an in-process model loads on is the machine's: ``CHAI_DEVICE``, never a setting.

A workflow file is meant to run unchanged on a laptop and on a cluster, so it cannot say ``mps`` or
``cuda:1``. These tests cover the environment variable, the refusal of a ``device`` setting wherever
one used to be read, and -- where the weights are in the Hugging Face cache -- a real model loading
on the device the environment names.
"""

import importlib.util
import os
import unittest
from unittest import mock

from chai.core import importClass
from chai.hardware import DEVICE_ENV, requested_device, torch_device
from chai.system_one import (
    GlinerProvider,
    JevOmniProvider,
    LayaProvider,
    LogitProvider,
    System1Model,
    make_provider,
    make_question,
    pick_device,
)
from chai.workflow import Workflow

NOUL = make_question("noul", "The text mentions a painter.")
STATE = "The Night Café was painted by Vincent van Gogh in Arles in 1888."


def device_env(value=None):
    """The environment with ``CHAI_DEVICE`` set to *value*, or without it."""
    env = {k: v for k, v in os.environ.items() if k != DEVICE_ENV}
    if value is not None:
        env[DEVICE_ENV] = value
    return mock.patch.dict(os.environ, env, clear=True)


def cached(repo, filename, revision=None):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo, filename, revision=revision)
    return isinstance(path, str) and os.path.exists(path)


VON_REVISION = "498ceba33390b32cfefaab6422ec380318ba9b99"
VON_READY = cached("wfzyx/von", "option_marker.pt", VON_REVISION)
GLINER_READY = importlib.util.find_spec("gliner2") is not None and cached("fastino/GLiNER2.5-Decide", "model.safetensors")
LAYA_READY = importlib.util.find_spec("laya") is not None and cached("convaiinnovations/laya", "typed-decisions/model.safetensors")

_saved_offline = None


def setUpModule():
    """Local weights come from the Hugging Face cache only; never download during tests."""
    global _saved_offline
    try:
        import huggingface_hub.constants as hf
    except ImportError:
        return
    _saved_offline = hf.HF_HUB_OFFLINE
    hf.HF_HUB_OFFLINE = True


def tearDownModule():
    if _saved_offline is not None:
        import huggingface_hub.constants as hf

        hf.HF_HUB_OFFLINE = _saved_offline


class TestTheEnvironmentNamesTheDevice(unittest.TestCase):
    def test_chai_device_is_read_from_the_environment(self):
        with device_env("cuda:1"):
            self.assertEqual(requested_device(), "cuda:1")
            self.assertEqual(torch_device(), "cuda:1")  # named, not probed: the machine's owner knows it
            self.assertEqual(pick_device(), "cuda:1")
        with device_env("  mps "):
            self.assertEqual(requested_device(), "mps")

    def test_unset_or_empty_leaves_the_choice_to_the_machine(self):
        for value in (None, "", "   "):  # `CHAI_DEVICE=` in .env is the same as leaving it out
            with device_env(value):
                self.assertIsNone(requested_device())
                self.assertIn(torch_device(), ("cuda", "mps", "cpu"))
                self.assertEqual(pick_device(), torch_device())

    def test_a_value_that_is_not_a_device_fails_by_name(self):
        with device_env("gpu"):
            with self.assertRaises(ValueError) as caught:
                torch_device()
            self.assertIn("CHAI_DEVICE='gpu'", str(caught.exception))

    def test_providers_take_the_device_from_the_environment(self):
        with device_env("cpu"):
            self.assertEqual(make_provider("kev-4b", {}).device, "cpu")
            self.assertEqual(make_provider("von", {}).device, "cpu")
            self.assertEqual(make_provider("openjev-nli-4b", {}).device, "cpu")
            self.assertEqual(LayaProvider("", {}).device, "cpu")
            self.assertEqual(GlinerProvider("", {}).device, "cpu")
            self.assertEqual(JevOmniProvider("", {}).device, "cpu")
        with device_env():
            for provider in (make_provider("kev-4b", {}), LayaProvider("", {}), GlinerProvider("", {}), JevOmniProvider("", {})):
                self.assertIsNone(provider.device)  # decided when the model loads

    def test_weights_are_kept_per_device(self):
        with device_env("cpu"):
            on_cpu = make_provider("kev-4b", {}).cache_key()
        with device_env():
            unset = make_provider("kev-4b", {}).cache_key()
        self.assertNotEqual(on_cpu, unset)  # a second device means a second copy, not the first one reused


class TestADeviceSettingIsRefused(unittest.TestCase):
    def refused(self, build, value="cpu"):
        with self.assertRaises(ValueError) as caught:
            build()
        message = str(caught.exception)
        self.assertIn("`device` is not a setting", message)
        self.assertIn(f"CHAI_DEVICE={value}", message)  # says what to set instead
        return message

    def test_helper(self):
        self.assertIn("decide: ", self.refused(lambda: requested_device({"device": "cpu"}, "decide")))
        self.refused(lambda: requested_device({"device": "cuda:1"}), "cuda:1")
        self.refused(lambda: requested_device({"device": None}), "mps")  # present at all is refused
        with device_env("cpu"):  # the environment does not excuse the setting
            self.refused(lambda: requested_device({"device": "mps"}), "mps")
        with device_env():
            self.assertIsNone(requested_device({"model": "x"}))  # other settings are none of its business

    def test_every_system_one_model_id(self):
        for model in ("kev-4b", "laya", "gliner-decide", "jev-omni", "von", "bekko-400m", "openjev-nli-4b",
                      "pointer/yale-cultural-heritage/kev-4b-europeana-persons", "openai/gpt-4o-mini"):
            with self.subTest(model=model):
                message = self.refused(lambda: make_provider(model, {"device": "cpu"}))
                self.assertIn(repr(model), message)
                self.refused(lambda: System1Model({"model": model, "device": "cpu"}))

    def test_providers_built_directly(self):
        class Fixed(LogitProvider):
            name = "fixed"

        for build in (
            lambda: Fixed("fixed", {"device": "cpu"}),
            lambda: LayaProvider("", {"device": "cpu"}),
            lambda: GlinerProvider("", {"device": "cpu"}),
            lambda: JevOmniProvider("", {"device": "cpu"}),
        ):
            self.refused(build)

    def test_a_decision_component_fails_at_build(self):
        tree = {"type": "Workflow", "id": "w", "steps": [{
            "type": "decision.DecisionClassifier", "id": "decide",
            "settings": {"model": "pointer/yale-cultural-heritage/kev-4b-europeana-persons", "device": "mps",
                         "question_type": "noul", "question": "It is so."}}]}
        self.refused(lambda: Workflow(tree), "mps")
        del tree["steps"][0]["settings"]["device"]
        Workflow(tree)  # the same workflow without it builds, and loads nothing yet

    def test_a_transformers_component_fails_at_build_before_it_looks_for_the_model(self):
        tree = {"type": "transcriber.TransformersTranscriber", "id": "read",
                "settings": {"model": "no-such-org/no-such-model", "device": "cuda"}}
        wf = Workflow({"type": "Workflow", "id": "w"})
        message = self.refused(lambda: importClass(tree["type"])(tree, wf), "cuda")
        self.assertIn("read", message)  # names the component


class TestRealModelsLoadOnTheNamedDevice(unittest.TestCase):
    """With ``CHAI_DEVICE=cpu`` a model loads on the CPU, whatever accelerator the machine has."""

    def decide(self, model, layout="described"):
        with device_env("cpu"):
            _, meta = System1Model({"model": model}).decide(STATE, {"q": NOUL}, layout)
        return meta

    @unittest.skipUnless(VON_READY, "wfzyx/von is not cached")
    def test_von(self):
        meta = self.decide("von")
        self.assertEqual(meta["device"], "cpu")
        self.assertEqual(meta["local_compute"]["device"], "cpu")

    @unittest.skipUnless(GLINER_READY, "GLiNER2 not installed or GLiNER2.5-Decide is not cached")
    def test_gliner(self):
        self.assertEqual(self.decide("gliner-decide")["device"], "cpu")

    @unittest.skipUnless(LAYA_READY, "laya not installed or its typed-decisions checkpoint is not cached")
    def test_laya(self):
        self.assertEqual(self.decide("laya", "in_state")["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
