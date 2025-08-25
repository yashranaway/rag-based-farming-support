from app.services.prompting import PromptBuilder
from app.services.ingestion import Chunk


def make_chunk(text: str, doc_id: str = "d1", idx: int = 0):
    return Chunk(id=f"c{idx}", doc_id=doc_id, text=text, metadata={"chunk_index": str(idx), "region": "mh", "crop": "tomato"})


def test_token_budget_truncates_context():
    pb = PromptBuilder(language="en")
    # Create a long chunk
    text = "A" * 1000  # ~250 tokens
    ch = make_chunk(text)
    built = pb.build("Q", [ch], max_context_chars=5000, max_context_tokens=100)
    # Expect truncation to around 400 chars (100 tokens * 4 chars)
    assert "Context:\n" in built.prompt
    ctx = built.prompt.split("Context:\n", 1)[1]
    ctx = ctx.split("\n\nUser Question:", 1)[0]
    assert len(ctx) <= 420
    assert built.citations and built.citations[0]["chunk_index"] == "0"


def test_external_signals_included():
    pb = PromptBuilder(language="en")
    ch = make_chunk("Advice snippet.")
    signals = {"weather": {"temp_c": 30.5}, "prices": [
        {"market": "Vashi", "price": 1800}
    ]}
    built = pb.build("Q2", [ch], external_signals=signals)
    assert "External Signals:" in built.prompt
    assert "weather" in built.prompt
    assert "prices" in built.prompt
