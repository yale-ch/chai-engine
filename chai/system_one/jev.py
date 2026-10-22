"""TypeSafe's hosted Jev: all of a state's questions in one ``POST /v1/systemone`` call."""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Tuple

from .base import DecisionProvider, register, usage_from
from .questions import wire_question


@register("jev", "typesafe")
class JevProvider(DecisionProvider):
    """TypeSafe's Jev through ``POST {base}/v1/systemone``: all questions in one call.

    Authenticates with the ``api_key`` setting, else ``TYPESAFE_API_KEY`` / ``JEV_API_KEY``. A 429 or
    529 is retried a few times with backoff, honouring ``Retry-After``; other errors raise.
    """

    name = "jev"

    def __init__(self, model: str, settings: dict):
        self.model = model or "jev-latest"
        self.api_key = settings.get("api_key") or os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY", "")
        if not self.api_key:
            raise ValueError("the Jev System One model needs TYPESAFE_API_KEY or JEV_API_KEY (env) or an `api_key` setting")
        base = settings.get("api_base") or settings.get("base_url") or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
        self.base_url = str(base).rstrip("/")
        self.timeout = float(settings.get("timeout", 30))
        self.attempts = int(settings.get("rate_limit_retries", 3)) + 1

    def decide(self, state, questions):
        import requests

        body = {"state": state, "model": self.model, "questions": {n: wire_question(q) for n, q in questions.items()}}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 1.0
        for attempt in range(self.attempts):
            resp = requests.post(f"{self.base_url}/v1/systemone", json=body, headers=headers, timeout=self.timeout)
            if resp.status_code in (429, 529) and attempt < self.attempts - 1:
                retry_after = resp.headers.get("retry-after", "")
                wait = float(retry_after) if re.fullmatch(r"\d+(\.\d+)?", retry_after) else delay
                time.sleep(wait)
                delay *= 2
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Jev API {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            answers = data.get("answers")
            if not isinstance(answers, dict):
                raise RuntimeError(f"Jev API returned no answers: {str(data)[:300]}")
            return answers, {"model": data.get("model") or self.model, "token_usage": usage_from(data.get("usage")), "cost": None}
        raise RuntimeError("Jev API: retries exhausted")
