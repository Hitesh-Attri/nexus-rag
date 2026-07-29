"""nexus-rag FastAPI app.

Lifespan warms the two expensive singletons (embedding model + DB pool) at
startup, and closes the pool cleanly on shutdown — no per-request cold start,
and no dangling pool threads at exit.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.health import router as health_router
from api.ingest import router as ingest_router
from core.db import get_pool
from core.embeddings import get_embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_embedder()      # load the ONNX model once, up front
    get_pool()          # open the connection pool
    yield
    get_pool().close()  # release pool connections/threads on shutdown


app = FastAPI(title="nexus-rag", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(ingest_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)