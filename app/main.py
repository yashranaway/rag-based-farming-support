import time
import os
from collections import deque, defaultdict
from typing import Deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import api_router

app = FastAPI(title="Smart Farming Advice API", version="0.1.0")


# Simple in-memory sliding-window rate limiter (per-IP)
_REQS: dict[str, Deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Enabled check (default off to avoid interfering with tests)
    enabled = os.getenv("RATE_LIMIT_ENABLED", "0").lower() in {"1", "true", "yes"}
    if not enabled:
        return await call_next(request)

    window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
    max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    dq = _REQS[client_ip]
    # purge old
    cutoff = now - window_seconds
    while dq and dq[0] < cutoff:
        dq.popleft()
    remaining = max_requests - len(dq)
    reset_in = 0 if not dq else max(0, int(dq[0] + window_seconds - now))
    if remaining <= 0:
        headers = {
            "X-RateLimit-Limit": str(max_requests),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(reset_in),
        }
        return JSONResponse({"detail": "rate_limited"}, status_code=429, headers=headers)
    dq.append(now)
    response = await call_next(request)
    # update headers
    headers = {
        "X-RateLimit-Limit": str(max_requests),
        "X-RateLimit-Remaining": str(max(0, remaining - 1)),
        "X-RateLimit-Reset": str(reset_in),
    }
    for k, v in headers.items():
        response.headers[k] = v
    return response

# Mount versioned API routes
app.include_router(api_router, prefix="/v1")


@app.get("/v1/healthz")
def healthz():
    """Basic liveness probe."""
    return JSONResponse({"status": "ok"})


@app.get("/v1/readyz")
def readyz():
    """Readiness probe; expand checks as components wire up."""
    checks = {
        "vector_store": "not_configured",
        "llm_provider": "not_configured",
        "embeddings": "not_configured",
    }
    return JSONResponse({"status": "degraded", "checks": checks})


# Optional: local dev entrypoint
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
