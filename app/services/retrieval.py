from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence, Tuple, Dict
from datetime import datetime

from app.services.ingestion import UpsertStore, Chunk
from app.services.embeddings import Embeddings
from app.services.vectorstore import InMemoryVectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, *, filters: Dict[str, str] | None = None, k: int = 5) -> List[RetrievalResult]:  # pragma: no cover (interface)
        ...


class InMemoryRetriever:
    """
    Very simple keyword retriever over UpsertStore.
    - Scores by term frequency of query tokens present in chunk text.
    - Applies filters on chunk.metadata (exact match) and region tags if filter key 'region_tag' provided.
    """

    def __init__(self, store: UpsertStore) -> None:
        self.store = store

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return [t.lower() for t in s.split() if t.strip()]

    def _score(self, query_tokens: Sequence[str], text: str) -> float:
        if not text:
            return 0.0
        tl = text.lower()
        return sum(tl.count(tok) for tok in query_tokens)

    def _passes_filters(self, ch: Chunk, filters: Dict[str, str] | None) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if k == "region_tag":
                # region_tag filter checks presence in tags derived into metadata (if any)
                # We copy region into metadata in ingestion; tags may not be present, so fallback to region
                tag_val = ch.metadata.get("region", "")
                if tag_val.lower() != v.lower():
                    return False
            else:
                if ch.metadata.get(k) != v:
                    return False
        return True

    def retrieve(self, query: str, *, filters: Dict[str, str] | None = None, k: int = 5) -> List[RetrievalResult]:
        qtokens = self._tokenize(query)
        scored: List[RetrievalResult] = []
        for ch in self.store.chunks.values():
            if not self._passes_filters(ch, filters):
                continue
            s = self._score(qtokens, ch.text)
            if s > 0:
                scored.append(RetrievalResult(chunk=ch, score=float(s)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


class EmbeddingRetriever:
    """
    Vector-store based retriever using provided Embeddings and VectorStore.
    Requires chunks to be indexed into the vector store ahead of time.
    """

    def __init__(self, store: UpsertStore, embeddings: Embeddings, vector_store: InMemoryVectorStore):
        self.store = store
        self.embeddings = embeddings
        self.vs = vector_store

    def retrieve(self, query: str, *, filters: Dict[str, str] | None = None, k: int = 5) -> List[RetrievalResult]:
        qv = self.embeddings.embed([query])[0]
        results = self.vs.similarity_search(qv, k=k, filter=filters or {})
        out: List[RetrievalResult] = []
        for itm, score in results:
            ch = self.store.chunks.get(itm.id)
            if ch is None:
                # Fall back: try to find by chunk_id in metadata
                cid = itm.metadata.get("chunk_id")
                if cid:
                    ch = self.store.chunks.get(cid)
            if ch is not None:
                out.append(RetrievalResult(chunk=ch, score=float(score)))
        return out


def index_store_chunks(store: UpsertStore, embeddings: Embeddings, vector_store: InMemoryVectorStore, *, dim: int | None = None) -> int:
    """
    Embed all chunks in UpsertStore and upsert into vector store. Returns count indexed.
    """
    chunks = list(store.chunks.values())
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    vecs = embeddings.embed(texts)
    ids = [c.id for c in chunks]
    metas = [dict(c.metadata) | {"chunk_id": c.id} for c in chunks]
    vector_store.upsert(ids, vecs, metas)
    return len(chunks)


class FreshnessWeightedRetriever:
    """
    Wrapper that applies exponential decay on scores based on chunk ingested_at.
    score' = score * exp(-lambda * age_days).
    """

    def __init__(self, base: Retriever, *, decay_lambda_per_day: float = 0.05) -> None:
        self.base = base
        self.lmbda = decay_lambda_per_day

    @staticmethod
    def _age_days(ch: Chunk) -> float:
        ts = ch.metadata.get("ingested_at")
        if not ts:
            return 0.0
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            return 0.0
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
        return max(delta.days + delta.seconds / 86400.0, 0.0)

    def retrieve(self, query: str, *, filters: Dict[str, str] | None = None, k: int = 5) -> List[RetrievalResult]:
        base_res = self.base.retrieve(query, filters=filters, k=k * 4)
        rescored: List[RetrievalResult] = []
        for r in base_res:
            age = self._age_days(r.chunk)
            factor = pow(2.718281828, -self.lmbda * age)
            rescored.append(RetrievalResult(chunk=r.chunk, score=r.score * factor))
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:k]


class RerankerWrapper:
    """
    Simple reranker stub that boosts chunks with 'authority' metadata.
    score'' = score' + boost if authority present.
    """

    def __init__(self, base: Retriever, *, authority_boost: float = 0.1) -> None:
        self.base = base
        self.boost = authority_boost

    def retrieve(self, query: str, *, filters: Dict[str, str] | None = None, k: int = 5) -> List[RetrievalResult]:
        res = self.base.retrieve(query, filters=filters, k=k * 2)
        reranked: List[RetrievalResult] = []
        for r in res:
            b = self.boost if r.chunk.metadata.get("authority") else 0.0
            reranked.append(RetrievalResult(chunk=r.chunk, score=r.score + b))
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:k]
