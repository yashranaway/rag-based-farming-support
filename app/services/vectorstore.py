from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import os


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class VSItem:
    id: str
    vector: List[float]
    metadata: Dict[str, str]


class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity."""

    def __init__(self) -> None:
        self.items: Dict[str, VSItem] = {}

    def upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, str]]) -> None:
        for _id, vec, meta in zip(ids, vectors, metadatas):
            self.items[_id] = VSItem(id=_id, vector=vec, metadata=meta)

    def similarity_search(self, query: List[float], k: int = 5, filter: Dict[str, str] | None = None) -> List[Tuple[VSItem, float]]:
        results: List[Tuple[VSItem, float]] = []
        for it in self.items.values():
            if filter:
                ok = True
                for fk, fv in filter.items():
                    if it.metadata.get(fk) != fv:
                        ok = False
                        break
                if not ok:
                    continue
            s = _cosine(query, it.vector)
            results.append((it, s))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:k]


class OpenSearchVectorStore:
    """
    OpenSearch adapter (mock/injected client).
    - Does not require opensearch-py import; client is injected (mockable in tests).
    - Stores vectors and metadata under a single document per ID.
    - similarity_search uses a generic `search` call; tests will mock the client.
    """

    def __init__(self, client: Any, index_name: str, dim: int) -> None:
        self.client = client
        self.index = index_name
        self.dim = dim

    def upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, str]]) -> None:
        for _id, vec, meta in zip(ids, vectors, metadatas):
            doc = {"vector": vec, "metadata": meta}
            self.client.index(index=self.index, id=_id, document=doc)

    def similarity_search(self, query: List[float], k: int = 5, filter: Dict[str, str] | None = None) -> List[Tuple[VSItem, float]]:
        # Build a mockable body. Real KNN would use kNN query, but we keep generic for tests.
        body = {
            "size": k,
            "query": {"knn": {"field": "vector", "query_vector": query, "k": k, "num_candidates": max(k, 50)}},
            "post_filter": {"term": filter} if filter else None,
        }
        # Remove None fields
        body = {k2: v2 for k2, v2 in body.items() if v2 is not None}
        res = self.client.search(index=self.index, body=body)
        hits = res.get("hits", {}).get("hits", [])
        out: List[Tuple[VSItem, float]] = []
        for h in hits:
            _id = h.get("_id")
            _score = float(h.get("_score", 0.0))
            src = h.get("_source", {})
            meta = src.get("metadata", {})
            vec = src.get("vector", [0.0] * self.dim)
            out.append((VSItem(id=_id, vector=vec, metadata=meta), _score))
        # Apply client-side filter for deterministic behavior in tests/fake client
        if filter:
            def _ok(m: Dict[str, str]) -> bool:
                for fk, fv in filter.items():
                    if m.get(fk) != fv:
                        return False
                return True
            out = [t for t in out if _ok(t[0].metadata)]
        return out[:k]


def vector_store_from_env(client: Optional[Any] = None, *, index: str = "rag-chunks", dim: int = 256):
    provider = os.getenv("VECTOR_PROVIDER", "memory").lower()
    if provider == "opensearch":
        if client is None:
            raise ValueError("OpenSearch client must be provided when VECTOR_PROVIDER=opensearch")
        return OpenSearchVectorStore(client=client, index_name=index, dim=dim)
    return InMemoryVectorStore()
