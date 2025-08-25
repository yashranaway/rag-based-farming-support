import os
import time
import uuid
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict

from app.api.models import QueryRequest, AnswerResponse
from app.services.lang import detect_language, choose_response_language
from app.services.ingestion import UpsertStore
from app.services.retrieval import (
    InMemoryRetriever,
    EmbeddingRetriever,
    index_store_chunks,
    FreshnessWeightedRetriever,
    RerankerWrapper,
)
from app.services.orchestrator import QueryOrchestrator
from app.services.llm import GraniteAdapter, FakeAdapter
from app.services.embeddings import SimpleTokenizerEmbeddings
from app.services.vectorstore import vector_store_from_env

api_router = APIRouter()

# Feature flag: enable orchestrated RAG path
FEATURE_ORCHESTRATOR = os.getenv("FEATURE_ORCHESTRATOR", "0").lower() in {"1", "true", "yes"}

# Lightweight singletons for dev
_STORE = UpsertStore()
_LLM = GraniteAdapter() if os.getenv("GRANITE_API_KEY") else FakeAdapter(response="RAG stub answer")

# Retrieval provider selection
RETRIEVAL_PROVIDER = os.getenv("RETRIEVAL_PROVIDER", "keyword").lower()  # keyword | embedding
VECTOR_PROVIDER = os.getenv("VECTOR_PROVIDER", "memory").lower()  # memory | opensearch
RETRIEVAL_FRESHNESS = os.getenv("RETRIEVAL_FRESHNESS", "0").lower() in {"1", "true", "yes"}
RETRIEVAL_RERANKER = os.getenv("RETRIEVAL_RERANKER", "0").lower() in {"1", "true", "yes"}

_EMB = SimpleTokenizerEmbeddings(dim=256)
_VS = vector_store_from_env()  # memory by default; OpenSearch requires injected client in app wiring
_INDEXED_ONCE = False

def _get_retriever():
    global _INDEXED_ONCE
    if RETRIEVAL_PROVIDER == "embedding":
        # Lazy one-time indexing of current store contents
        if not _INDEXED_ONCE:
            index_store_chunks(_STORE, _EMB, _VS)
            _INDEXED_ONCE = True
        base = EmbeddingRetriever(_STORE, _EMB, _VS)
    else:
        base = InMemoryRetriever(_STORE)

    if RETRIEVAL_FRESHNESS:
        base = FreshnessWeightedRetriever(base)
    if RETRIEVAL_RERANKER:
        base = RerankerWrapper(base)
    return base


@api_router.post("/query", response_model=AnswerResponse)
async def query(req: QueryRequest):
    """Stub query endpoint; will be wired to RAG orchestration in later tasks."""
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # Determine response language using prefs > locale > detected
    detected = detect_language(req.text)
    prefs_lang = req.preferences.language if req.preferences else None
    language = choose_response_language(detected, prefs_language=prefs_lang, locale=req.locale)

    citations = []
    answer_text = f"[{language}] This is a placeholder response. The RAG pipeline will be implemented in later tasks."
    tokens_prompt = 0
    tokens_output = 0

    if FEATURE_ORCHESTRATOR:
        orch = QueryOrchestrator(_RETRIEVER, _LLM)
        out = orch.run(req.text, language=language, filters={})
        citations = out.citations
        answer_text = out.answer
        # Best-effort tokens from LLM stub if present
        try:
            tokens_prompt = _LLM.generate.__self__._estimate_tokens(out.prompt)  # type: ignore[attr-defined]
            tokens_output = _LLM.generate.__self__._estimate_tokens(out.answer)  # type: ignore[attr-defined]
        except Exception:
            tokens_prompt = 0
            tokens_output = 0

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    resp = AnswerResponse(
        answer=answer_text,
        language=language,
        citations=citations,
        warnings=[],
        diagnostics={
            "latency_ms": elapsed_ms,
            "tokens_prompt": tokens_prompt,
            "tokens_output": tokens_output,
            "retrieval_k": 0,
        },
    )
    headers = {"X-Trace-Id": trace_id, "X-Elapsed-Ms": str(elapsed_ms)}
    return JSONResponse(resp.model_dump(), headers=headers)


@api_router.get("/sources")
async def list_sources() -> Dict[str, str]:
    """List indexed sources (placeholder)."""
    return {"status": "ok", "sources": []}


@api_router.post("/admin/reindex")
async def admin_reindex():
    """Trigger reindex (placeholder)."""
    return {"status": "accepted", "message": "Reindex triggered (stub)."}
