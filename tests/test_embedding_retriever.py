from app.services.ingestion import UpsertStore, ingest_text
from app.services.embeddings import SimpleTokenizerEmbeddings
from app.services.vectorstore import InMemoryVectorStore
from app.services.retrieval import EmbeddingRetriever, index_store_chunks


def test_embedding_retriever_end_to_end():
    store = UpsertStore()
    # Ingest two docs with different crops
    ingest_text(store, "Tomato needs regular irrigation and mulching.", region="mh", crop="tomato")
    ingest_text(store, "Wheat prefers cool weather and timely sowing.", region="pb", crop="wheat")

    emb = SimpleTokenizerEmbeddings(dim=64)
    vs = InMemoryVectorStore()
    # Index all chunks
    n = index_store_chunks(store, emb, vs)
    assert n >= 2

    retriever = EmbeddingRetriever(store, emb, vs)
    res = retriever.retrieve("best irrigation for tomato mulch", filters=None, k=2)
    assert res and res[0].chunk.metadata.get("crop") == "tomato"

    res_f = retriever.retrieve("harvesting", filters={"crop": "wheat"}, k=2)
    # With filter, even weak matches must obey metadata
    assert all(r.chunk.metadata.get("crop") == "wheat" for r in res_f)
