from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Iterable, List, Optional


@dataclass
class LLMResponse:
    text: str
    tokens_prompt: int
    tokens_output: int
    model: str


class LLMAdapter(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:  # pragma: no cover (interface)
        ...
    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> Iterable[str]:  # pragma: no cover (interface)
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
        self._fail_mode = os.getenv("LLM_SIMULATE_ERROR", "").lower()  # e.g., "quota" | "credit"

    def _estimate_tokens(self, text: str) -> int:
        # Rough approximation: 1 token ~= 0.75 words
        words = max(1, len(text.strip().split()))
        return int(words / 0.75)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        if self._fail_mode == "quota":
            raise RuntimeError("llm_error:quota_exceeded")
        if self._fail_mode == "credit":
            raise RuntimeError("llm_error:insufficient_credit")
        stub_text = (
            f"[granite-stub:{self.model}] "
            f"{prompt[:120]}" + ("…" if len(prompt) > 120 else "")
        )
        out_text = stub_text[:max_tokens * 4]  # very rough cap
        # Apply naive stop sequence truncation for stub
        if stop:
            for s in stop:
                if s and s in out_text:
                    out_text = out_text.split(s, 1)[0]
                    break
        tp = self._estimate_tokens(prompt)
        to = self._estimate_tokens(out_text)
        return LLMResponse(text=out_text, tokens_prompt=tp, tokens_output=to, model=self.model)

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> Iterable[str]:
        if self._fail_mode == "quota":
            raise RuntimeError("llm_error:quota_exceeded")
        if self._fail_mode == "credit":
            raise RuntimeError("llm_error:insufficient_credit")
        # Simple 3-chunk stream stub
        full = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop).text
        n = max(1, len(full) // 3)
        for i in range(0, len(full), n):
            yield full[i : i + n]


class GraniteWatsonXAdapter(GraniteAdapter):
    def __init__(self, *, model: str = "granite-13b-chat-wx", api_key_env: str = "GRANITE_API_KEY") -> None:
        super().__init__(model=model, api_key_env=api_key_env)


class GraniteReplicateAdapter(GraniteAdapter):
    def __init__(self, *, model: str = "granite-13b-chat-replicate", api_key_env: str = "REPLICATE_API_TOKEN") -> None:
        super().__init__(model=model, api_key_env=api_key_env)


class FakeAdapter:
    """Deterministic test adapter."""

    def __init__(self, response: str = "ok", model: str = "fake-llm") -> None:
        self._response = response
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        text = self._response[: max_tokens * 4]
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s, 1)[0]
                    break
        tokens_prompt = max(1, len(prompt.split()))
        tokens_output = max(1, len(text.split()))
        return LLMResponse(text=text, tokens_prompt=tokens_prompt, tokens_output=tokens_output, model=self.model)

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> Iterable[str]:
        text = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop).text
        split = max(1, len(text) // 2)
        yield text[:split]
        yield text[split:]
