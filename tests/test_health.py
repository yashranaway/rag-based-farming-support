from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz():
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_readyz():
    r = client.get("/v1/readyz")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") in {"ok", "degraded"}
    assert "checks" in data


def test_query_stub():
    r = client.post("/v1/query", json={"text": "Hello", "locale": "en-IN"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert data["language"] in {"en-IN", "auto"}
