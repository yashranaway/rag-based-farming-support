from app.services.embeddings import SimpleTokenizerEmbeddings
from app.services.vectorstore import InMemoryVectorStore


def test_embeddings_and_vectorstore_similarity():
    emb = SimpleTokenizerEmbeddings(dim=64)
    vs = InMemoryVectorStore()

    texts = [
        "tomato irrigation mulch",
        "wheat sowing cool weather",
        "onion storage post-harvest",
    ]
    vecs = emb.embed(texts)
    ids = [f"id{i}" for i in range(len(texts))]
    metas = [{"crop": "tomato"}, {"crop": "wheat"}, {"crop": "onion"}]
    vs.upsert(ids, vecs, metas)

    qv = emb.embed(["best irrigation tomato mulch"])[0]
    results = vs.similarity_search(qv, k=2)
    assert results and results[0][0].metadata["crop"] == "tomato"

    # Filter should narrow to a specific crop
    results_filtered = vs.similarity_search(qv, k=2, filter={"crop": "wheat"})
    assert results_filtered and results_filtered[0][0].metadata["crop"] == "wheat"
