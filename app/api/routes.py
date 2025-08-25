import os
import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict

from app.api.models import (
    QueryRequest,
    AnswerResponse,
    ReindexRequest,
    TemplateSetRequest,
    TemplateRollbackRequest,
)
from app.services.lang import detect_language, choose_response_language
from app.services.ingestion import UpsertStore, ingest_text
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
from app.services.observability import set_trace_id, get_logger, redact_payload
from app.services.templates import TemplateRegistry

api_router = APIRouter()

# Feature flag: evaluate at request-time
def is_orchestrator_enabled() -> bool:
    return os.getenv("FEATURE_ORCHESTRATOR", "0").lower() in {"1", "true", "yes"}

# Lightweight singletons for dev
_STORE = UpsertStore()
_TPL = TemplateRegistry()
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
    set_trace_id(trace_id)
    t0 = time.perf_counter()
    log = get_logger("api.query")

    # Determine response language using prefs > locale > detected
    detected = detect_language(req.text)
    prefs_lang = req.preferences.language if req.preferences else None
    language = choose_response_language(detected, prefs_language=prefs_lang, locale=req.locale)

    citations = []
    answer_text = f"[{language}] This is a placeholder response. Enable FEATURE_ORCHESTRATOR=1 for RAG."
    tokens_prompt = None
    tokens_output = None

    if is_orchestrator_enabled():
        retriever = _get_retriever()
        orch = QueryOrchestrator(retriever, _LLM)
        out = orch.run(req.text, language=language, filters={})
        # Map internal citations (doc_id/chunk_index/source_url) to API model shape
        citations = []
        for c in out.citations:
            title = f"{c.get('doc_id', '')}#{c.get('chunk_index', '')}".strip('#') or "source"
            url = c.get("source_url") or None
            citations.append({"title": title, "url": url})
        answer_text = out.answer
        tokens_prompt = None
        tokens_output = None

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
            "retrieval_k": len(citations) if citations else 0,
        },
    )
    headers = {"X-Trace-Id": trace_id, "X-Elapsed-Ms": str(elapsed_ms)}
    try:
        log.info(
            "handled query",
            extra={
                "extra": {
                    "elapsed_ms": elapsed_ms,
                    "feature_orchestrator": is_orchestrator_enabled(),
                    "language": language,
                    "request": redact_payload(req.model_dump() if hasattr(req, "model_dump") else {}),
                }
            },
        )
    except Exception:
        pass
    return JSONResponse(resp.model_dump(), headers=headers)


@api_router.get("/sources")
async def list_sources() -> Dict[str, str]:
    """List indexed sources (placeholder)."""
    return {"status": "ok", "sources": []}


@api_router.post("/admin/reindex")
async def admin_reindex(req: ReindexRequest):
    """Ingest provided text into the store and refresh embedding index if enabled."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    ingest_text(
        _STORE,
        text,
        region=req.region,
        crop=req.crop,
        source_url=req.source_url,
        max_chars=req.max_chars or 800,
        overlap=req.overlap or 100,
    )
    # If embedding retriever is active, refresh index
    if RETRIEVAL_PROVIDER == "embedding":
        index_store_chunks(_STORE, _EMB, _VS)
    get_logger("api.admin").info("reindex", extra={"extra": {"has_text": True, "region": req.region or "", "crop": req.crop or ""}})
    return {"status": "ok", "message": "ingested"}


@api_router.post("/admin/templates/{name}")
async def admin_template_set(name: str, req: TemplateSetRequest):
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must be non-empty")
    tv = _TPL.set(name, content)
    get_logger("api.admin").info("template_set", extra={"extra": {"name": name, "version": tv.version}})
    return {"status": "ok", "name": name, "version": tv.version, "created_at": tv.created_at}


@api_router.get("/admin/templates/{name}")
async def admin_template_current(name: str):
    tv = _TPL.current(name)
    if tv is None:
        raise HTTPException(status_code=404, detail="template not found")
    return {"name": name, "version": tv.version, "content": tv.content, "created_at": tv.created_at}


@api_router.get("/admin/templates/{name}/versions")
async def admin_template_versions(name: str):
    versions = _TPL.list_versions(name)
    return {
        "name": name,
        "versions": [
            {"version": tv.version, "created_at": tv.created_at, "content": tv.content} for tv in versions
        ],
    }


@api_router.post("/admin/templates/{name}/rollback")
async def admin_template_rollback(name: str, req: TemplateRollbackRequest):
    try:
        tv = _TPL.rollback(name, req.version)
    except KeyError:
        raise HTTPException(status_code=404, detail="template not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="version not found")
    return {"status": "ok", "name": name, "version": tv.version, "created_at": tv.created_at}


@api_router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming answer chunks from orchestrator."""
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    t0 = time.perf_counter()

    detected = detect_language(req.text)
    prefs_lang = req.preferences.language if req.preferences else None
    language = choose_response_language(detected, prefs_language=prefs_lang, locale=req.locale)

    if not is_orchestrator_enabled():
        def _placeholder():
            yield f"[{language}] Streaming not enabled. Set FEATURE_ORCHESTRATOR=1."
        return StreamingResponse(_placeholder(), media_type="text/plain", headers={"X-Trace-Id": trace_id})

    retriever = _get_retriever()
    orch = QueryOrchestrator(retriever, _LLM)
    gen = orch.run_stream(req.text, language=language, filters={})
    headers = {"X-Trace-Id": trace_id, "X-Elapsed-Ms": str(int((time.perf_counter()-t0)*1000))}
    get_logger("api.query").info("handled query stream", extra={"extra": {"language": language}})
    return StreamingResponse(gen, media_type="text/plain", headers=headers)
