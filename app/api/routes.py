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
from app.services.llm import (
    GraniteAdapter,
    GraniteWatsonXAdapter,
    GraniteReplicateAdapter,
    FakeAdapter,
)
from app.services.embeddings import SimpleTokenizerEmbeddings
from app.services.vectorstore import vector_store_from_env

api_router = APIRouter()

# Feature flag: enable orchestrated RAG path
FEATURE_ORCHESTRATOR = os.getenv("FEATURE_ORCHESTRATOR", "0").lower() in {"1", "true", "yes"}

# Lightweight singletons for dev
_STORE = UpsertStore()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "granite-wx").lower()  # granite-wx | granite-replicate | fake
if LLM_PROVIDER == "granite-replicate":
    _LLM = GraniteReplicateAdapter()
elif LLM_PROVIDER == "fake":
    _LLM = FakeAdapter(response="RAG stub answer")
else:
    _LLM = GraniteWatsonXAdapter()

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
    """Query endpoint using feature-flagged orchestrator pipeline."""
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # Determine response language using prefs > locale > detected
    detected = detect_language(req.text)
    prefs_lang = req.preferences.language if req.preferences else None
    language = choose_response_language(detected, prefs_language=prefs_lang, locale=req.locale)

    citations = []
    answer_text = f"[{language}] This is a placeholder response. Enable FEATURE_ORCHESTRATOR=1 for RAG."
    tokens_prompt = None
    tokens_output = None

    if FEATURE_ORCHESTRATOR:
        retriever = _get_retriever()
        orch = QueryOrchestrator(retriever, _LLM)
        out = orch.run(req.text, language=language, filters={})
        citations = out.citations
        answer_text = out.answer
        tokens_prompt = None
        tokens_output = None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    resp = AnswerResponse(
        answer=answer_text,
        language=language,
        citations=[],
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


@api_router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming answer chunks from orchestrator."""
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    detected = detect_language(req.text)
    prefs_lang = req.preferences.language if req.preferences else None
    language = choose_response_language(detected, prefs_language=prefs_lang, locale=req.locale)

    if not FEATURE_ORCHESTRATOR:
        def _placeholder():
            yield f"[{language}] Streaming not enabled. Set FEATURE_ORCHESTRATOR=1."
        return StreamingResponse(_placeholder(), media_type="text/plain", headers={"X-Trace-Id": trace_id})

    retriever = _get_retriever()
    orch = QueryOrchestrator(retriever, _LLM)
    gen = orch.run_stream(req.text, language=language, filters={})
    headers = {"X-Trace-Id": trace_id, "X-Elapsed-Ms": str(int((time.perf_counter()-t0)*1000))}
    return StreamingResponse(gen, media_type="text/plain", headers=headers)
