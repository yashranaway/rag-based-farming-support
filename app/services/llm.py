from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    tokens_prompt: int
    tokens_output: int
    model: str


class LLMAdapter(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResponse:  # pragma: no cover (interface)
        ...


class GraniteAdapter:
    """
    Placeholder adapter for IBM Granite models.
    - Reads API key/config from environment but does NOT make network calls in this scaffold.
    - Produces a deterministic stubbed response for tests and local dev.
    """

    def __init__(self, *, model: str = "granite-13b-chat", api_key_env: str = "GRANITE_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)  # Not required for stubbed mode

    def _estimate_tokens(self, text: str) -> int:
        # Rough approximation: 1 token ~= 0.75 words
        words = max(1, len(text.strip().split()))
        return int(words / 0.75)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResponse:
        stub_text = (
            f"[granite-stub:{self.model}] "
            f"{prompt[:120]}" + ("…" if len(prompt) > 120 else "")
        )
        out_text = stub_text[:max_tokens * 4]  # very rough cap
        tp = self._estimate_tokens(prompt)
        to = self._estimate_tokens(out_text)
        return LLMResponse(text=out_text, tokens_prompt=tp, tokens_output=to, model=self.model)


class FakeAdapter:
    """Deterministic test adapter."""

    def __init__(self, response: str = "ok", model: str = "fake-llm") -> None:
        self._response = response
        self.model = model

    def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResponse:
        text = self._response[: max_tokens * 4]
        tokens_prompt = max(1, len(prompt.split()))
        tokens_output = max(1, len(text.split()))
        return LLMResponse(text=text, tokens_prompt=tokens_prompt, tokens_output=tokens_output, model=self.model)
