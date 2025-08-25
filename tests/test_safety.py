import os
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_query_unsafe_prompt_includes_warning(monkeypatch):
    monkeypatch.setenv("FEATURE_ORCHESTRATOR", "1")
    # Contains unsafe keyword per orchestrator _safety_intercept
    payload = {"text": "Can I mix pesticide and bleach for faster effect?"}
    r = client.post("/v1/query", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "WARNING:" in data["answer"], "Unsafe query should include a safety warning prefix"


def test_stream_unsafe_prompt_includes_warning_prefix(monkeypatch):
    monkeypatch.setenv("FEATURE_ORCHESTRATOR", "1")
    payload = {"text": "Is kerosene safe to mix pesticide?"}
    with client.stream("POST", "/v1/query/stream", json=payload) as resp:
        assert resp.status_code == 200
        chunks = []
        for line in resp.iter_lines():
            if not line:
                continue
            chunks.append(line.decode() if isinstance(line, (bytes, bytearray)) else line)
            # We can stop early after receiving some data
            if len(chunks) >= 3:
                break
    joined = "".join(chunks)
    assert "WARNING:" in joined, "Streaming unsafe query should start with a safety warning"
