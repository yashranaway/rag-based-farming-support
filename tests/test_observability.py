import json
import os
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.main import app
from app.services.observability import JsonFormatter, get_logger, set_trace_id, redact_payload


client = TestClient(app)


@contextmanager
def env(**kwargs):
    old = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_json_formatter_includes_trace_id(capsys):
    set_trace_id("tid-123")
    log = get_logger("test.logger")
    log.info("hello", extra={"extra": {"foo": "bar"}})
    captured = capsys.readouterr()
    # Our JsonFormatter writes one JSON line to stderr
    line = captured.err.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["trace_id"] == "tid-123"
    assert data["message"] == "hello"
    assert data["foo"] == "bar"


def test_redact_payload():
    payload = {"gps": (19.1, 72.8), "pincode": "400001", "token": "secret", "x": 1}
    red = redact_payload(payload)
    assert red["gps"] == "[REDACTED]"
    assert red["pincode"] == "[REDACTED]"
    assert red["token"] == "[REDACTED]"
    assert red["x"] == 1


def test_query_includes_trace_headers():
    with env(FEATURE_ORCHESTRATOR="0"):
        r = client.post("/v1/query", json={"text": "hello"})
        assert r.status_code == 200
        assert "X-Trace-Id" in r.headers
        assert "X-Elapsed-Ms" in r.headers
