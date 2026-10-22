"""Encoder System One models, all ModernBERT-based: Laya, von and bekko.

* ``laya`` -- ``convaiinnovations/laya``: ``[CLS] "<type> question: q" [SEP] [MASK] opt0 [MASK] opt1 .. [SEP] state
  [SEP]``, a type embedding and a small head scoring each ``[MASK]``; run through the ``laya`` package,
  which answers every question about a state in one forward pass with its own calibrated temperatures.
* ``von`` -- ``wfzyx/von``: the state, question and options packed into one sequence marked by ``[MASK]``
  tokens, each option attending only to the prefix and itself (``independent_options``); the model code
  is carried in ``vendor/von_option_marker.py`` (Apache-2.0).
* ``bekko`` -- ``hotchpotch/bekko-system-one-v0-400m``: the query (instruction and state) encoded once and
  each candidate as a branch over it; runs the model repo's own ``inference_v0.py`` (no licence declared
  on the Hub: fine for internal experiments, not for redistributing fine-tuned weights).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

from .base import DEVICE_LOCK, DecisionProvider, LogitProvider, load_once, pick_device, register, temperatures, usage_from
from .questions import wire_question

logger = logging.getLogger("chai")


def import_file(path, name: str):
    """Import a Python file (a model repo's own inference code) as a module."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(repo: str, revision=None, allow_patterns=None) -> str:
    """A local folder holding *repo*: the folder itself, or the Hub snapshot."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=allow_patterns)


# --------------------------------------------------------------------------- Laya

_LAYA_AGENTS: dict = {}


_LAYA_LOCK = threading.Lock()


@register("laya")
class LayaProvider(DecisionProvider):
    """Laya (``convaiinnovations/laya``), in process. Loaded once per process (about 850 MB on
    first use); answers every question about a state in one forward pass. Settings: ``subfolder``
    (checkpoint variant such as 'multilingual' or 'typed-decisions') and ``device`` (torch device)."""

    name = "laya"

    def __init__(self, model: str, settings: dict):
        self.model = model or "convaiinnovations/laya"
        self.settings = dict(settings or {})
        self.subfolder = self.settings.get("subfolder") or None
        self.device = self.settings.get("device") or None
        self.lengths = {k: int(self.settings[k]) for k in ("max_len", "head_max_len") if self.settings.get(k)}

    def agent(self):
        key = (self.model, self.subfolder or "", self.device or "")
        with _LAYA_LOCK, DEVICE_LOCK:
            if key not in _LAYA_AGENTS:
                import laya  # heavy import, deferred until first use

                logger.info(f"Loading Laya {self.model} subfolder={self.subfolder!r} device={self.device!r}")
                _LAYA_AGENTS[key] = laya.load(self.model, subfolder=self.subfolder, device=self.device)
            return _LAYA_AGENTS[key]

    def decide(self, state, questions):
        agent = self.agent()
        wire = {n: wire_question(q) for n, q in questions.items()}
        with DEVICE_LOCK:
            res = agent.predict(state, wire, **self.lengths)
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            raise RuntimeError(f"Laya returned no answers: {str(res)[:300]}")
        return answers, {
            "model": (res.get("model") if isinstance(res, dict) else None) or self.model,
            "token_usage": usage_from(res.get("usage")),
            "cost": 0.0,
            "device": str(getattr(agent, "device", "") or "") or None,
        }



# --------------------------------------------------------------------------- von


@register("von")
class VonProvider(LogitProvider):
    """von (``wfzyx/von``), in process: ``von/<hub repo or folder>``.

    Settings: ``revision``, ``independent_options`` (default true: each option attends only to the
    prefix and itself, so the result is option-order invariant), ``digit_split`` (default false, as
    the SDK serves it), ``max_len`` (default 4096), ``max_state_tokens`` (default 2048), ``device``,
    ``temperature`` (default: the checkpoint's ``calibration.json``).
    """

    name = "von"
    noul_slots = ("true", "false")

    def __init__(self, model: str, settings: dict):
        super().__init__(model or "wfzyx/von", settings)
        self.revision = self.settings.get("revision")
        self.independent = bool(self.settings.get("independent_options", True))
        self.digit_split = bool(self.settings.get("digit_split", False))
        self.max_len = int(self.settings.get("max_len", 4096))
        self.max_state = int(self.settings.get("max_state_tokens", 2048))
        self.rt = None

    def load(self):
        if self.rt is None:
            self.rt = load_once(("von", self.model, self.revision or "", self.device or "", self.digit_split), self._load_runtime)
            self.device = self.rt.device
            if "temperature" not in self.settings:
                self.temperature = temperatures(self.rt.temperature)

    def _load_runtime(self):
        import torch

        from .vendor.von_option_marker import OptionMarkerModel

        device = pick_device(self.device)
        root = snapshot(self.model, self.revision, ["config.json", "model.safetensors", "tokenizer*", "calibration.json", "marker_calibration.json", "option_marker.pt"])
        logger.info(f"Loading von {self.model} on {device}")
        model = OptionMarkerModel(base_model_id=root, digit_split=self.digit_split)
        state = torch.load(os.path.join(root, "option_marker.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        temperature = 1.0
        calibration = Path(root) / "calibration.json"
        if calibration.exists():
            cal = json.loads(calibration.read_text())
            temperature = float(cal.get("temperature", 1.0)) if isinstance(cal, dict) else float(cal)
        model = model.float().to(device).eval()
        return SimpleNamespace(model=model, tok=model.tokenizer, device=device, temperature=temperature)

    @staticmethod
    def option_texts(question: dict) -> List[str]:
        if question["type"] == "noul":
            return ["Yes, condition holds true.", "No, condition is false."]
        return [o["text"] or o["key"] for o in question["options"]]

    def logits(self, state, question):
        import torch

        tok, model = self.rt.tok, self.rt.model
        ids = tok.encode(state, add_special_tokens=False)
        if len(ids) > self.max_state:
            state = tok.decode(ids[: self.max_state])
        text = model.pack_sequence(state, question["instructions"], self.option_texts(question))
        enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.rt.device)
        positions = [(enc.input_ids[0] == model.mask_token_id).nonzero().flatten().tolist()]
        wanted = len(self.slots(question))
        if len(positions[0]) != wanted:
            raise ValueError(f"{len(positions[0])} of {wanted} option markers survived truncation to {self.max_len} tokens")
        self.last_tokens = int(enc.attention_mask.sum())
        with DEVICE_LOCK, torch.inference_mode():
            zs = model(enc.input_ids, enc.attention_mask, positions, independent_options=self.independent)
        return zs[0].float().cpu().tolist()


# --------------------------------------------------------------------------- bekko


@register("bekko")
class BekkoProvider(LogitProvider):
    """bekko (``hotchpotch/bekko-system-one-v0-400m``), in process: ``bekko/<hub repo>``.

    Runs the model repo's own ``inference_v0.py`` (downloaded and imported). Settings: ``revision``,
    ``prefix_layout`` (default 'instruction_state'), ``query_length`` (default 4096),
    ``document_length`` (default 512), ``device``, ``temperature`` (default 1).
    """

    name = "bekko"
    noul_slots = ("true", "false")

    def __init__(self, model: str, settings: dict):
        super().__init__(model or "hotchpotch/bekko-system-one-v0-400m", settings)
        self.revision = self.settings.get("revision")
        self.layout = self.settings.get("prefix_layout", "instruction_state")
        self.query_length = int(self.settings.get("query_length", 4096))
        self.document_length = int(self.settings.get("document_length", 512))
        self.rt = None

    def load(self):
        if self.rt is None:
            self.rt = load_once(("bekko", self.model, self.revision or "", self.device or ""), self._load_runtime)
            self.device = self.rt.device

    def _load_runtime(self):
        from huggingface_hub import hf_hub_download

        device = pick_device(self.device)
        code = hf_hub_download(self.model, "inference_v0.py", revision=self.revision)
        lib = import_file(code, "bekko_inference_v0")
        logger.info(f"Loading bekko {self.model} on {device}")
        model = lib.BekkoInference.load(self.model, subfolder="0_BekkoInference", revision=self.revision)
        return SimpleNamespace(lib=lib, model=model.to(device).eval(), device=device)

    @staticmethod
    def native(question: dict, state: str) -> dict:
        """The decision in bekko's own input format."""
        if question["type"] == "noul":
            criteria = [
                {"id": "true", "description_json": json.dumps("The statement is true."), "value": None},
                {"id": "false", "description_json": json.dumps("The statement is false."), "value": None},
            ]
        else:
            criteria = [
                {"id": o["key"], "description_json": json.dumps(o["text"] or o["key"]), "value": float(i) if question["type"] == "score" else None}
                for i, o in enumerate(question["options"])
            ]
        return {
            "state_json": json.dumps(state, ensure_ascii=False),
            "decisions": [
                {"id": "q", "kind": "judgment", "type": question["type"], "instructions_json": json.dumps(question["instructions"]),
                 "system_prompt": "", "criteria": criteria, "documents": [], "scoring": None}
            ],
        }

    def logits(self, state, question):
        import torch

        lib, model = self.rt.lib, self.rt.model
        groups = list(lib.input_groups(self.native(question, state), prefix_layout=self.layout))
        prepared = model.prepare_groups(groups, query_length=self.query_length, document_length=self.document_length)
        feats = lib.to_device(lib.collate_groups(prepared, model), self.rt.device)
        with DEVICE_LOCK, torch.inference_mode():
            scores = model(feats)["scores"].flatten()
        n = len(prepared[0].documents)
        return scores[:n].float().cpu().tolist()
