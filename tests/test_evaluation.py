import os

from fastapi.testclient import TestClient

from app.main import app
from app.services.eval import evaluate_with_client


client = TestClient(app)


def test_evaluation_harness_basic(tmp_path, monkeypatch):
    # Ensure orchestrator is enabled so citations may be returned
    monkeypatch.setenv("FEATURE_ORCHESTRATOR", "1")

    # Use bundled small dataset
    ds_path = os.path.join(os.path.dirname(__file__), "data", "eval_qa.jsonl")
    metrics = evaluate_with_client(client, ds_path)

    # Basic shape and ranges
    assert set(metrics.keys()) == {"count", "relevance", "citation_accuracy", "groundedness"}
    assert metrics["count"] >= 1
    for k in ("relevance", "citation_accuracy", "groundedness"):
        assert 0.0 <= float(metrics[k]) <= 1.0
