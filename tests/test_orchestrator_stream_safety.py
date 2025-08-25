import os
from contextlib import contextmanager

from app.services.ingestion import UpsertStore, ingest_text
from app.services.retrieval import InMemoryRetriever
from app.services.llm import FakeAdapter
from app.services.orchestrator import QueryOrchestrator


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


def _setup_retriever():
    store = UpsertStore()
    ingest_text(
        store,
        "Tomato irrigation: mulch retains moisture. Avoid overwatering during rains.",
        region="maharashtra",
        crop="tomato",
        source_url="http://advisory",
        max_chars=200,
    )
    return InMemoryRetriever(store)


def test_run_stream_yields_chunks():
    retriever = _setup_retriever()
    resp_text = "Use drip irrigation and mulch to retain moisture."
    llm = FakeAdapter(response=resp_text, model="fake")
    orch = QueryOrchestrator(retriever, llm)
    gen = orch.run_stream("Best irrigation for tomato?", language="en", filters={"region": "maharashtra", "crop": "tomato"})
    chunks = list(gen)
    assert len(chunks) >= 1
    joined = "".join(chunks)
    assert "mulch" in joined or "drip" in joined


def test_safety_intercept_prefixes_warning():
    retriever = _setup_retriever()
    llm = FakeAdapter(response="Do not actually do this.", model="fake")
    orch = QueryOrchestrator(retriever, llm)
    out = orch.run("Can I mix pesticide with bleach for better results?", language="en")
    assert out.answer.startswith("WARNING:")
