"""AI Memory Engine — production REST API.

Run:   python server.py          (or: uvicorn server:app)
Docs:  http://127.0.0.1:8000/docs

Features: optional API-key auth, structured logging, consistent JSON errors,
pagination, tag-filtered search, update, stats, JSON backup/restore.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from memory_engine import config
from memory_engine.store import (
    MemoryStore,
    MemoryStoreError,
    NotFoundError,
    ValidationError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("memory_engine.api")

store: MemoryStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    config.validate_deployment_config()
    log.info("Starting AI Memory Engine (model warm-up on first request)...")
    store = MemoryStore()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="AI Memory Engine",
    version="2.0.0",
    lifespan=lifespan,
    description="Local semantic memory: store facts, recall them by meaning.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── auth (enabled only when MEMORY_API_KEY is set in .env) ───────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(api_key_header)):
    if config.API_KEY and key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# ── consistent error responses ───────────────────────────────────────────
@app.exception_handler(ValidationError)
async def on_validation(_: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(NotFoundError)
async def on_not_found(_: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(MemoryStoreError)
async def on_store_error(_: Request, exc: MemoryStoreError):
    log.error("Storage error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info(
        "%s %s -> %d (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
    )
    return response


# ── schemas ──────────────────────────────────────────────────────────────
class MemoryIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class MemoryPatch(BaseModel):
    text: str | None = Field(None, min_length=1, max_length=10_000)
    tags: list[str] | None = Field(None, max_length=20)


class MemoryOut(BaseModel):
    id: str
    text: str
    tags: list[str]
    created_at: str
    updated_at: str
    score: float | None = None
    duplicate: bool | None = None
    similar_to: str | None = None
    similarity: float | None = None


class PageOut(BaseModel):
    total: int
    items: list[MemoryOut]


class ImportIn(BaseModel):
    records: list[dict]


# ── routes ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "memories": store.count(), "auth": bool(config.API_KEY)}


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return store.stats()


@app.post(
    "/memories",
    response_model=MemoryOut,
    response_model_exclude_none=True,
    status_code=201,
    dependencies=[Depends(require_key)],
)
def add_memory(body: MemoryIn):
    return store.add(body.text, body.tags)


@app.get(
    "/memories/search",
    response_model=list[MemoryOut],
    response_model_exclude_none=True,
    dependencies=[Depends(require_key)],
)
def search_memories(
    q: str = Query(..., min_length=1, max_length=10_000),
    k: int = Query(5, ge=1, le=50),
    tag: str | None = Query(None, max_length=50),
    min_score: float = Query(-1.0, ge=-1.0, le=1.0),
):
    return store.search(q, k=k, tag=tag, min_score=min_score)


@app.get(
    "/memories",
    response_model=PageOut,
    response_model_exclude_none=True,
    dependencies=[Depends(require_key)],
)
def list_memories(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return store.list_all(limit=limit, offset=offset)


@app.get(
    "/memories/{memory_id}",
    response_model=MemoryOut,
    response_model_exclude_none=True,
    dependencies=[Depends(require_key)],
)
def get_memory(memory_id: str):
    return store.get(memory_id)


@app.patch(
    "/memories/{memory_id}",
    response_model=MemoryOut,
    response_model_exclude_none=True,
    dependencies=[Depends(require_key)],
)
def update_memory(memory_id: str, body: MemoryPatch):
    return store.update(memory_id, text=body.text, tags=body.tags)


@app.delete("/memories/{memory_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_memory(memory_id: str):
    if not store.delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")


@app.get("/backup/export", dependencies=[Depends(require_key)])
def export_backup():
    records = store.export_json()
    return {"count": len(records), "records": records}


@app.post("/backup/import", dependencies=[Depends(require_key)])
def import_backup(body: ImportIn):
    imported = store.import_json(body.records)
    return {"imported": imported, "skipped": len(body.records) - imported}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=config.API_HOST, port=config.API_PORT)
