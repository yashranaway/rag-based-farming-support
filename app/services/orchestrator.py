from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterable

from app.services.prompting import PromptBuilder
from app.services.retrieval import Retriever
from app.services.llm import LLMAdapter
from app.services.connectors import WeatherClient, MandiClient


@dataclass
class OrchestratorResult:
    answer: str
    citations: List[Dict[str, str]]
    prompt: str
    language: str
    tokens_prompt: Optional[int] = None
    tokens_output: Optional[int] = None


class QueryOrchestrator:
    """
    Minimal RAG orchestrator stub:
    - Uses provided Retriever to fetch chunks
    - Builds a prompt via PromptBuilder
    - Calls LLMAdapter to generate an answer
    """

    def __init__(self, retriever: Retriever, llm: LLMAdapter):
        self.retriever = retriever
        self.llm = llm
        self._weather = WeatherClient()
        self._mandi = MandiClient()

    def _classify_intent(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["price", "rate", "mandi", "market"]):
            return "mandi_prices"
        if any(k in q for k in ["weather", "rain", "temperature", "irrigation"]):
            return "weather_advice"
        return "general_agri"

    def _fetch_signals(self, intent: str, *, crop: Optional[str], region: Optional[str]) -> Dict[str, Any]:
        signals: Dict[str, Any] = {}
        if intent == "mandi_prices" and crop and region:
            signals["mandi_prices"] = self._mandi.latest_prices(crop, region)
        if intent in {"weather_advice", "general_agri"} and region:
            signals["weather"] = self._weather.current_and_forecast({"region": region})
        return signals

    def _safety_intercept(self, question: str, draft_answer: str) -> Optional[str]:
        q = question.lower()
        unsafe_keywords = ["kerosene", "bleach", "acid", "mix pesticide", "ingest"]
        if any(k in q for k in unsafe_keywords):
            return (
                "WARNING: The requested action may be unsafe. Avoid harmful mixtures and follow IPM and label guidance.\n"
                + draft_answer
            )
        return None

    def run(
        self,
        question: str,
        *,
        language: str = "auto",
        filters: Optional[Dict[str, str]] = None,
        k: int = 4,
        max_context_tokens: Optional[int] = None,
        max_generate_tokens: Optional[int] = None,
        external_signals: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        filters = filters or {}
        intent = self._classify_intent(question)
        # derive region/crop from filters if available
        region = filters.get("region")
        crop = filters.get("crop")
        signals = dict(external_signals or {})
        signals.update(self._fetch_signals(intent, crop=crop, region=region))

        results = self.retriever.retrieve(question, filters=filters, k=k)
        chunks = [r.chunk for r in results]
        pb = PromptBuilder(language=language)
        built = pb.build(question, chunks, max_context_tokens=max_context_tokens, external_signals=signals)
        # Enforce generation token cap
        gen_cap = max_generate_tokens if max_generate_tokens is not None else int(os.getenv("MAX_GENERATE_TOKENS", "256"))
        llm_out = self.llm.generate(built.prompt, max_tokens=gen_cap)
        intercepted = self._safety_intercept(question, llm_out.text)
        return OrchestratorResult(
            answer=intercepted or llm_out.text,
            citations=built.citations,
            prompt=built.prompt,
            language=language,
            tokens_prompt=getattr(llm_out, "tokens_prompt", None),
            tokens_output=getattr(llm_out, "tokens_output", None),
        )

    def run_stream(
        self,
        question: str,
        *,
        language: str = "auto",
        filters: Optional[Dict[str, str]] = None,
        k: int = 4,
        max_context_tokens: Optional[int] = None,
        max_generate_tokens: Optional[int] = None,
        external_signals: Optional[Dict[str, Any]] = None,
    ) -> Iterable[str]:
        """Yield answer tokens in a streaming fashion from the LLM."""
        filters = filters or {}
        intent = self._classify_intent(question)
        region = filters.get("region")
        crop = filters.get("crop")
        signals = dict(external_signals or {})
        signals.update(self._fetch_signals(intent, crop=crop, region=region))

        results = self.retriever.retrieve(question, filters=filters, k=k)
        chunks = [r.chunk for r in results]
        pb = PromptBuilder(language=language)
        built = pb.build(question, chunks, max_context_tokens=max_context_tokens, external_signals=signals)
        # Safety intercept preface if needed
        preface = self._safety_intercept(question, "")
        if preface:
            yield preface
        gen_cap = max_generate_tokens if max_generate_tokens is not None else int(os.getenv("MAX_GENERATE_TOKENS", "256"))
        for part in self.llm.stream_generate(built.prompt, max_tokens=gen_cap):
            yield part
