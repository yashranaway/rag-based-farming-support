import os
import importlib
from contextlib import contextmanager

import pytest

from app.services import vectorstore as vs_mod


@contextmanager
def env(**kwargs):
    old = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class FakeMilvusClient:
    def __init__(self):
        self.upserts = []
        self.searches = []

    def upsert(self, collection, ids, vectors, metadatas):
        self.upserts.append((collection, list(ids), list(vectors), list(metadatas)))

    def search(self, collection, query, *, k=5, filter=None):
        self.searches.append((collection, list(query), k, dict(filter or {})))
        # Return a trivial single hit result
        return [
            {
                "id": "doc-1",
                "score": 0.9,
                "metadata": {"region": "MH"},
                "vector": [0.0] * 4,
            }
        ]


def test_vector_store_factory_milvus_returns_adapter():
    with env(VECTOR_PROVIDER="milvus"):
        client = FakeMilvusClient()
        store = vs_mod.vector_store_from_env(client=client, index="test-coll", dim=4)
        assert isinstance(store, vs_mod.MilvusVectorStore)
        # upsert and search go through client
        store.upsert(["a"], [[0.1, 0.2, 0.3, 0.4]], [{"region": "MH"}])
        out = store.similarity_search([0.1, 0.2, 0.3, 0.4], k=1)
        assert out and out[0][0].id == "doc-1"
        assert client.upserts and client.searches


def test_vector_store_factory_opensearch_requires_client():
    with env(VECTOR_PROVIDER="opensearch"):
        with pytest.raises(ValueError):
            vs_mod.vector_store_from_env(client=None)
