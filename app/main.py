from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes import api_router

app = FastAPI(title="Smart Farming Advice API", version="0.1.0")

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
