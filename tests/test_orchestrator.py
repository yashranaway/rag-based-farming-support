from app.services.ingestion import UpsertStore, ingest_text
from app.services.retrieval import InMemoryRetriever
from app.services.llm import FakeAdapter
from app.services.orchestrator import QueryOrchestrator


def test_orchestrator_flow_builds_prompt_and_returns_answer():
    store = UpsertStore()
    # Ingest domain context
    ingest_text(
        store,
        "Tomato irrigation: water deeply but infrequently. Mulch helps retain soil moisture.",
        region="maharashtra",
        crop="tomato",
        source_url="http://advisory",
        max_chars=200,
    )

    retriever = InMemoryRetriever(store)
    llm = FakeAdapter(response="use drip irrigation; mulch to retain moisture", model="fake")
    orch = QueryOrchestrator(retriever, llm)

    out = orch.run("Best irrigation for tomato in hot weather", language="en", filters={"region": "maharashtra"}, k=3)

    assert out.language == "en"
    assert "mulch" in out.answer
    assert len(out.citations) >= 1
    assert "doc_id" in out.citations[0]
    assert "chunk_index" in out.citations[0]
    assert "User Question:" in out.prompt
