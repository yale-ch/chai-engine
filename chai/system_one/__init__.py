"""System One decision models for chai: typed questions answered with probabilities, not text.

A System One model reads a *state* (text) and answers one or more typed questions about it with a
calibrated probability per option, in a single forward pass -- no generation, no parsing. The three
question types, their options and the canonical answers are in ``questions``; the roles that put
them into workflows are ``chai.decision`` (``DecisionClassifier``, ``DecisionGate``,
``DecisionExtractor``). This package holds the models, one provider per architecture family, the
way ``system1-finetune-bench`` groups them:

| family (id prefix) | models | how it decides |
|---|---|---|
| ``pointer`` / ``kev`` | kev-4b/9b/27b, Yale's ``kev-4b-europeana-persons`` | Qwen3.5 base + LoRA; a pointer head over the options' closing tokens |
| ``letters`` | plumb-4b, JevK5, decider, Tev1 | one prompt listing lettered options; the LM head's letter rows |
| ``candidate`` | Open-Jev-9B/27B | one chat prompt per option, a scalar head; softmax across options |
| ``nli`` | openjev-nli-4b | a 3-way NLI cross-encoder, log P(entailment) per option |
| ``slots`` | JEV-9B/27B, GEV-26B | a bare template and a 24-slot head on the last token |
| ``laya`` | convaiinnovations/laya | ModernBERT, one ``[MASK]`` per option, through the ``laya`` package |
| ``von`` | wfzyx/von | ModernBERT, options packed and marked, order-invariant |
| ``bekko`` | hotchpotch/bekko-system-one-v0-400m | ModernBERT, each candidate a branch over the query |
| ``gliner`` | fastino/GLiNER2.5-Decide and Yale's fine-tunes | DeBERTa classifier over described labels |
| ``jev-omni`` | akhilaaa3/Jev-Omni | Gemma 4 12B with a 256-way option head |
| ``jev`` | TypeSafe's hosted Jev | ``POST /v1/systemone`` |
| *(anything else)* | a LiteLLM chat model | asked for the same JSON: a baseline, not calibrated |

The ``model`` setting is a preset name (``"kev-4b"``, ``"plumb-4b"``, ``"laya"``, ... -- ``PRESETS``
mirrors the bench's model configs, pinned revisions included), a ``<family>/<checkpoint>`` id where
the checkpoint is a Hub repo or a local folder (a kev-layout export or an ``s1bench`` export), or a
LiteLLM chat model id. Models that only answer choice questions natively get score as a choice over
the levels and noul as a choice over true/false, as the bench's ``as_choice`` approaches do. Local
models load once per process and share ``DEVICE_LOCK``.
"""

from . import decoders, encoders, generative, gliner, jev  # noqa: F401 -- registers the families
from .base import (  # noqa: F401
    DEVICE_LOCK,
    FAMILIES,
    GENERATIVE,
    PRESETS,
    DecisionProvider,
    LogitProvider,
    System1Model,
    load_once,
    make_provider,
    pick_device,
    register,
    temperatures,
    usage_from,
)
from .decoders import (  # noqa: F401
    KEV_SPECIAL,
    CandidateProvider,
    JevOmniProvider,
    LetterProvider,
    NLIProvider,
    PointerProvider,
    SlotHeadProvider,
    jev_omni_prompt,
    kev_encode,
    kev_options,
)
from .encoders import BekkoProvider, LayaProvider, VonProvider  # noqa: F401
from .generative import SYSTEM1_PROMPT, GenerativeProvider  # noqa: F401
from .gliner import GlinerProvider, gliner_sanitize  # noqa: F401
from .jev import JevProvider  # noqa: F401
from .questions import (  # noqa: F401
    FALSE_KEY,
    LAYOUTS,
    NONE_KEY,
    NOUL_OPTIONS,
    NULL_KEYS,
    QUESTION_TYPES,
    TRUE_KEY,
    apply_layout,
    level_value,
    make_question,
    normalize_answer,
    normalize_options,
    question_keys,
    wire_question,
)
