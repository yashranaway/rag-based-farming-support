from typing import List

from app.services.llm import (
    GraniteAdapter,
    GraniteWatsonXAdapter,
    GraniteReplicateAdapter,
    FakeAdapter,
)


def _collect(gen) -> str:
    parts: List[str] = []
    for chunk in gen:
        assert isinstance(chunk, str)
        assert chunk != ""
        parts.append(chunk)
    return "".join(parts)


def test_granite_adapter_streams_chunks():
    llm = GraniteAdapter(model="granite-13b-chat")
    prompt = "tomato irrigation advice"
    chunks = list(llm.stream_generate(prompt, max_tokens=16))
    assert 1 <= len(chunks) <= 4
    text = "".join(chunks)
    assert text.startswith("[granite-stub:")


def test_fake_adapter_streaming_reassembles():
    llm = FakeAdapter(response="abcdef", model="fake")
    merged = _collect(llm.stream_generate("ignored", max_tokens=10))
    assert merged == "abcdef"


def test_provider_variants_streaming_behave():
    wx = GraniteWatsonXAdapter()
    rep = GraniteReplicateAdapter()
    out_wx = _collect(wx.stream_generate("hello", max_tokens=8))
    out_rep = _collect(rep.stream_generate("hello", max_tokens=8))
    assert out_wx.startswith("[granite-stub:")
    assert out_rep.startswith("[granite-stub:")
