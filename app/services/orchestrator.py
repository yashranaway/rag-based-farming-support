from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from app.services.prompting import PromptBuilder
from app.services.retrieval import Retriever
from app.services.llm import LLMAdapter


@dataclass
class OrchestratorResult:
    answer: str
    citations: List[Dict[str, str]]
    prompt: str
    language: str


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

    def run(
        self,
        question: str,
        *,
        language: str = "auto",
        filters: Optional[Dict[str, str]] = None,
        k: int = 4,
        max_context_tokens: Optional[int] = None,
        external_signals: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        results = self.retriever.retrieve(question, filters=filters or {}, k=k)
        chunks = [r.chunk for r in results]
        pb = PromptBuilder(language=language)
        built = pb.build(question, chunks, max_context_tokens=max_context_tokens, external_signals=external_signals)
        llm_out = self.llm.generate(built.prompt)
        return OrchestratorResult(
            answer=llm_out.text,
            citations=built.citations,
            prompt=built.prompt,
            language=language,
        )
