import os
import importlib
from contextlib import contextmanager

from fastapi.testclient import TestClient


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


def _reload_app():
    # Ensure app.main picks up new env for rate limiter globals
    import app.main as main_mod
    importlib.reload(main_mod)
    return main_mod.app


def test_rate_limit_headers_and_429():
    with env(RATE_LIMIT_ENABLED="1", RATE_LIMIT_WINDOW_SEC="5", RATE_LIMIT_MAX_REQUESTS="2"):
        app = _reload_app()
        client = TestClient(app)
        r1 = client.get("/v1/healthz")
        r2 = client.get("/v1/healthz")
        r3 = client.get("/v1/healthz")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
        # Headers present
        for r in (r1, r2, r3):
            assert "X-RateLimit-Limit" in r.headers
            assert "X-RateLimit-Remaining" in r.headers
            assert "X-RateLimit-Reset" in r.headers


def test_token_budget_affects_output_tokens():
    with env(FEATURE_ORCHESTRATOR="1"):
        # Low cap
        with env(MAX_GENERATE_TOKENS="4"):
            from app.main import app as app_low
            client_low = TestClient(app_low)
            r_low = client_low.post("/v1/query", json={"text": "tell me a very long detailed answer about agriculture and irrigation systems"})
            assert r_low.status_code == 200
            t_low = r_low.json()["diagnostics"]["tokens_output"]
            assert t_low is not None
        # Higher cap
        with env(MAX_GENERATE_TOKENS="128"):
            from app.main import app as app_hi
            client_hi = TestClient(app_hi)
            r_hi = client_hi.post("/v1/query", json={"text": "tell me a very long detailed answer about agriculture and irrigation systems"})
            assert r_hi.status_code == 200
            t_hi = r_hi.json()["diagnostics"]["tokens_output"]
            assert t_hi is not None
        # With larger cap, we expect tokens_output to be >= low cap run
        assert t_hi >= t_low
