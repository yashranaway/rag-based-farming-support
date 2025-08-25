import os
import types
import uuid

from app.services.vectorstore import OpenSearchVectorStore, vector_store_from_env


class FakeOSClient:
    def __init__(self):
        self.indexed = {}
        self.last_search = None

    def index(self, index, id, document):  # noqa: A003 (shadow builtin)
        self.indexed[id] = {"_index": index, "_source": document}
        return {"result": "created", "_id": id}

    def search(self, index, body):
        self.last_search = {"index": index, "body": body}
        # very simple scorer: dot product between query_vector and stored vector
        qv = body["query"]["knn"]["query_vector"]
        hits = []
        for _id, rec in self.indexed.items():
            vec = rec["_source"]["vector"]
            score = sum(a * b for a, b in zip(qv, vec))
            hits.append({
                "_id": _id,
                "_score": score,
                "_source": rec["_source"],
            })
        hits.sort(key=lambda h: h["_score"], reverse=True)
        size = body.get("size", 5)
        return {"hits": {"hits": hits[:size]}}


def test_opensearch_vectorstore_basic_similarity_and_filter(monkeypatch):
    client = FakeOSClient()
    vs = OpenSearchVectorStore(client=client, index_name="rag", dim=4)

    ids = ["a", "b"]
    vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    metas = [{"crop": "tomato"}, {"crop": "wheat"}]
    vs.upsert(ids, vecs, metas)

    q = [0.9, 0.1, 0.0, 0.0]
    results = vs.similarity_search(q, k=1)
    assert results[0][0].id == "a" and results[0][0].metadata["crop"] == "tomato"

    results_f = vs.similarity_search(q, k=2, filter={"crop": "wheat"})
    assert results_f and results_f[0][0].metadata["crop"] == "wheat"


def test_vector_store_from_env_memory_default(monkeypatch):
    monkeypatch.delenv("VECTOR_PROVIDER", raising=False)
    vs = vector_store_from_env()
    # Should be InMemoryVectorStore with upsert/similarity_search methods
    assert hasattr(vs, "upsert") and hasattr(vs, "similarity_search")


def test_vector_store_from_env_opensearch_requires_client(monkeypatch):
    monkeypatch.setenv("VECTOR_PROVIDER", "opensearch")
    try:
        vector_store_from_env()
        assert False, "Expected ValueError when client missing"
    except ValueError:
        pass

    client = FakeOSClient()
    vs = vector_store_from_env(client=client, index="idx", dim=4)
    assert isinstance(vs, OpenSearchVectorStore)
