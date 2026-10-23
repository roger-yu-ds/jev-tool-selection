"""A minimal client for TypeSafe's Jev (System One): one ``choice`` question, up to 255 options.

    POST https://api.typesafe.ai/v1/systemone
    {"state": <the customer's message>, "model": "jev-latest",
     "questions": {"tool": {"type": "choice", "instructions": ..., "criteria": {name: description}}}}

The answer carries a probability for every option plus a confidence. Input tokens cost
US$0.042 per million; output is free (https://docs.typesafe.ai/models).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_MTOK = 0.042


class Jev:
    def __init__(self, model: str = "jev-latest", timeout_s: float = 20.0) -> None:
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set (put it in .env)")
        self.model = model
        self._client = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {key}"})

    def choose(self, text: str, instructions: str, criteria: dict[str, str], question_id: str = "tool") -> dict[str, Any]:
        """Returns the raw response body plus ``latency_s`` and ``cost_usd``."""
        payload = {"state": text, "model": self.model,
                   "questions": {question_id: {"type": "choice", "instructions": instructions, "criteria": criteria}}}  # fmt: skip
        started = time.perf_counter()
        body = self._post(payload)
        latency = time.perf_counter() - started
        tokens = body["usage"]["input_tokens"]
        return {**body, "latency_s": latency, "cost_usd": tokens * PRICE_PER_MTOK / 1e6}

    def _post(self, payload: dict, attempts: int = 5) -> dict:
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(URL, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{resp.status_code} from Jev", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                retryable = isinstance(exc, httpx.TransportError) or exc.response.status_code in (429,) or exc.response.status_code >= 500
                if attempt == attempts or not retryable:
                    raise
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random()))
        raise AssertionError("unreachable")
