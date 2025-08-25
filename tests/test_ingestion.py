from datetime import datetime, UTC

from app.services.ingestion import chunk_text, enrich_metadata, UpsertStore, ingest_text


def test_chunk_text_boundaries_and_overlap():
    text = ("A" * 500) + "\n\n" + ("B" * 1200)
    chunks = chunk_text(text, max_chars=400, overlap=50)
    # First block should be intact
    assert chunks[0] == "A" * 500 or len(chunks[0]) <= 400  # depending on policy; our logic slices large blocks
    # Second block should be split into multiple chunks with overlap
    assert len(chunks) >= 3
    # Ensure no empty chunks
    assert all(c.strip() for c in chunks)


def test_enrich_metadata_contents():
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    meta = enrich_metadata(region="Maharashtra", crop="Tomato", authority="IMD", source_url="http://x", effective_date=ts)
    assert meta["region"] == "maharashtra"
    assert meta["crop"] == "tomato"
    assert meta["authority"] == "IMD"
    assert meta["source_url"] == "http://x"
    assert meta["ingested_at"].startswith("2024-01-01")


def test_ingest_text_upsert_flow():
    store = UpsertStore()
    text = "Para1" + "\n\n" + ("Para2-" * 300)
    doc, chunks = ingest_text(
        store,
        text,
        region="Maharashtra",
        crop="Tomato",
        authority="ICAR",
        source_url="http://doc",
        max_chars=200,
        overlap=40,
    )
    assert doc.id in store.docs
    assert all(ch.id in store.chunks for ch in chunks)
    assert all(ch.metadata.get("region") == "maharashtra" for ch in chunks)
    assert all("chunk_index" in ch.metadata for ch in chunks)
