from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_reindex_validation():
    # Empty text should 400
    resp = client.post("/v1/admin/reindex", json={"text": "  "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "text must be non-empty"

    # Valid ingest
    resp = client.post(
        "/v1/admin/reindex",
        json={
            "text": "Irrigation guidance for tomato",
            "region": "maharashtra",
            "crop": "tomato",
            "source_url": "http://doc",
            "max_chars": 100,
            "overlap": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_templates_crud_and_rollback():
    # Not found current
    resp = client.get("/v1/admin/templates/prompt")
    assert resp.status_code == 404

    # Set v1
    resp = client.post("/v1/admin/templates/prompt", json={"content": "hello"})
    assert resp.status_code == 200
    v1 = resp.json()["version"]
    assert v1 == 1

    # Current
    resp = client.get("/v1/admin/templates/prompt")
    assert resp.status_code == 200
    assert resp.json()["content"] == "hello"

    # Set v2
    resp = client.post("/v1/admin/templates/prompt", json={"content": "hello world"})
    assert resp.status_code == 200
    v2 = resp.json()["version"]
    assert v2 == 2

    # List versions
    resp = client.get("/v1/admin/templates/prompt/versions")
    assert resp.status_code == 200
    versions = resp.json()["versions"]
    assert len(versions) >= 2

    # Rollback to v1 -> creates v3 with v1 content
    resp = client.post("/v1/admin/templates/prompt/rollback", json={"version": 1})
    assert resp.status_code == 200
    v3 = resp.json()["version"]
    assert v3 == 3

    # Current should now be v3 with content "hello"
    resp = client.get("/v1/admin/templates/prompt")
    assert resp.status_code == 200
    assert resp.json()["version"] == 3
    assert resp.json()["content"] == "hello"

    # Bad rollback version
    resp = client.post("/v1/admin/templates/prompt/rollback", json={"version": 999})
    assert resp.status_code == 400

    # Bad set content
    resp = client.post("/v1/admin/templates/prompt", json={"content": "  "})
    assert resp.status_code == 400
