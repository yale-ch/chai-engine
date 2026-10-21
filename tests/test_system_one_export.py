"""Loading a bench export (``s1bench.json`` + ``trainable.safetensors``) into a decoder family.

system1-finetune-bench saves what it trained -- the LoRA tensors and the readout head -- as
``trainable.safetensors`` next to an ``s1bench.json`` naming the base checkpoint, the LoRA shape and
the fitted temperatures. ``chai.system_one`` loads such an export as ``<family>/<folder>``. No real
export is on this machine (the bench keeps its weights on the cluster), so the test fabricates one
around Qwen3.5-4B-Base: it attaches a fresh LoRA, writes random values for every trainable tensor,
then loads the folder as a model and checks every tensor came back. Opt in with
``CHAI_TEST_EXPORT=1``: it loads the 9 GB base twice (once to shape the export, once as the model).
"""

import json
import os
import tempfile
import unittest

from chai.system_one import base, decoders
from chai.system_one.decoders import LetterProvider

QWEN = "Qwen/Qwen3.5-4B-Base"
QWEN_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def cached(repo, filename, revision=None):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo, filename, revision=revision)
    return isinstance(path, str) and os.path.exists(path)


READY = os.getenv("CHAI_TEST_EXPORT") == "1" and cached(QWEN, "model.safetensors.index.json", QWEN_REVISION)


@unittest.skipUnless(READY, "set CHAI_TEST_EXPORT=1 with Qwen3.5-4B-Base cached (~9 GB, loaded twice)")
class TestBenchExport(unittest.TestCase):
    def test_a_fabricated_letters_export_loads_and_decides(self):
        import torch
        from safetensors.torch import save_file

        lora = {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["q_proj"]}
        model_cfg = {"family": "letters", "style": "decider", "checkpoint": QWEN, "revision": QWEN_REVISION, "lora": lora}
        with tempfile.TemporaryDirectory() as export:
            # Shape the export: the LoRA parameter names come from a PEFT model on the same base
            device = base.pick_device(None)
            backbone, lm_head, tok, full = decoders.load_text_lm(QWEN, revision=QWEN_REVISION, device=device)
            peft = decoders.add_lora(backbone, lora)
            tensors = {f"model.{n}": torch.randn(tuple(p.shape)) for n, p in peft.named_parameters() if p.requires_grad}
            self.assertTrue(tensors, "the LoRA added no trainable parameters")
            readout = LetterProvider(QWEN, {"style": "decider"}).letters
            tensors["readout.rows"] = torch.randn(len(readout), backbone.config.hidden_size)
            save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(export, "trainable.safetensors"))
            with open(os.path.join(export, "s1bench.json"), "w") as fh:
                json.dump({"model": model_cfg, "temperature": {"choice": 1.5, "score": 1.25, "noul": 2.0}}, fh)
            del peft, backbone, lm_head, full
            base._LOADED.clear()

            provider = LetterProvider(export, {})
            provider.load()
            self.assertEqual(provider.checkpoint, QWEN)
            self.assertEqual(provider.style, "decider")
            self.assertEqual(provider.temperature, {"choice": 1.5, "score": 1.25, "noul": 2.0})
            loaded = dict(provider.rt.model.named_parameters())
            restored = 0
            for key, value in tensors.items():
                prefix, _, name = key.partition(".")
                target = loaded[name] if prefix == "model" else dict(provider.rt.heads[prefix].named_parameters())[name]
                self.assertTrue(torch.allclose(target.detach().float().cpu(), value, atol=1e-2), key)
                restored += 1
            self.assertEqual(restored, len(tensors))
            # and it still decides: a distribution over the asked keys
            from chai.system_one import make_question

            answers, meta = provider.decide("Some text about a painting.", {"q": make_question("choice", "Which?", {"a": "A", "b": "B"})})
            self.assertAlmostEqual(sum(answers["q"]["probabilities"].values()), 1.0, places=4)
            self.assertEqual(meta["cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
