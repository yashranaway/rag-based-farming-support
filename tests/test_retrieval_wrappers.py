from datetime import datetime, UTC, timedelta

from app.services.ingestion import UpsertStore, ingest_text
from app.services.retrieval import InMemoryRetriever, FreshnessWeightedRetriever, RerankerWrapper


def test_freshness_weighting_prefers_recent():
    store = UpsertStore()
    # Two chunks same content but different ingested_at
    text = "Advisory: use drip irrigation for tomatoes."
    # Old (10 days ago)
    old_time = datetime.now(UTC) - timedelta(days=10)
    ingest_text(store, text, region="mh", crop="tomato", authority="ICAR", effective_date=old_time)
    # Recent (now)
    ingest_text(store, text, region="mh", crop="tomato", authority="ICAR")

    base = InMemoryRetriever(store)
    fr = FreshnessWeightedRetriever(base, decay_lambda_per_day=0.5)
    res = fr.retrieve("drip irrigation tomatoes", k=2)
    assert len(res) >= 1
    # The most recent chunk should outrank the older one
    assert res[0].chunk.metadata["ingested_at"] >= res[-1].chunk.metadata["ingested_at"]


def test_reranker_boosts_authority():
    store = UpsertStore()
    ingest_text(store, "Use certified seeds.", region="pb", crop="wheat")
    ingest_text(store, "Use certified seeds.", region="pb", crop="wheat", authority="ICAR")

    base = InMemoryRetriever(store)
    rr = RerankerWrapper(base, authority_boost=100.0)
    res = rr.retrieve("certified seeds", k=1)
    assert res and res[0].chunk.metadata.get("authority") == "ICAR"
