from app.services.ingestion import UpsertStore, ingest_text
from app.services.retrieval import InMemoryRetriever


def test_inmemory_retriever_keyword_scoring_and_filters():
    store = UpsertStore()
    # Ingest two docs with different regions/crops
    ingest_text(store, "Tomato needs regular irrigation. Tomato pests include hornworm.", region="maharashtra", crop="tomato", source_url="http://a", max_chars=200)
    ingest_text(store, "Wheat sowing time varies. Wheat thrives in cool weather.", region="punjab", crop="wheat", source_url="http://b", max_chars=200)

    r = InMemoryRetriever(store)
    # Query about tomato
    res = r.retrieve("Best irrigation for tomato", filters={"region": "maharashtra"}, k=3)
    assert len(res) >= 1
    # Top result should reference maharashtra tomato doc
    assert all(rr.chunk.metadata.get("region") == "maharashtra" for rr in res)

    # Query with non-matching region should produce none
    res2 = r.retrieve("tomato pests", filters={"region": "punjab"}, k=3)
    assert len(res2) == 0 or all("tomato" not in rr.chunk.text.lower() for rr in res2)
