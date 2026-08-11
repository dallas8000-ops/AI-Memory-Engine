"""Deep Lake-backed semantic memory store — production hardened.

Guarantees:
  - Thread-safe: all dataset access serialized through an RLock
    (FastAPI runs sync endpoints in a threadpool; without this,
    concurrent writes can corrupt a local Deep Lake dataset).
  - Validated input: length limits, empty rejection, tag sanitizing.
  - Duplicate protection: exact re-adds return the existing record;
    near-duplicates are flagged with the similar record's id.
  - Never raises raw backend errors to callers: wraps them in MemoryError-
    family exceptions with actionable messages.

Schema (one row per memory):
    id          Text       uuid4 hex
    text        Text       the memory content
    tags        Text       comma-separated tags ("" if none)
    created_at  Text       ISO-8601 UTC
    updated_at  Text       ISO-8601 UTC (== created_at until edited)
    embedding   Embedding  float32, cosine-normalized
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deeplake
import numpy as np

from . import config
from .embedder import Embedder

log = logging.getLogger("memory_engine.store")

MAX_TEXT_LEN = 10_000
MAX_TAGS = 20
MAX_TAG_LEN = 50
NEAR_DUP_THRESHOLD = 0.97


class MemoryStoreError(Exception):
    """Base error with a user-presentable message."""


class ValidationError(MemoryStoreError):
    pass


class NotFoundError(MemoryStoreError):
    pass


class StorageError(MemoryStoreError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).absolute().as_uri()


def _clean_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    out: list[str] = []
    for t in tags:
        t = str(t).strip().lower().replace(",", " ")[:MAX_TAG_LEN]
        if t and t not in out:
            out.append(t)
    if len(out) > MAX_TAGS:
        raise ValidationError(f"Too many tags (max {MAX_TAGS}).")
    return out


class MemoryStore:
    def __init__(self, data_path: str = config.DATA_PATH, embedder: Embedder | None = None):
        self.uri = _to_uri(data_path)
        self.embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self.ds = self._open_or_create()
        log.info("Memory store ready at %s (%d memories)", self.uri, len(self.ds))

    # ── dataset lifecycle ────────────────────────────────────────────────
    def _open_or_create(self):
        try:
            ds = deeplake.open(self.uri)
            # sanity: required columns present?
            cols = {c.name for c in ds.schema.columns}
            required = {"id", "text", "tags", "created_at", "updated_at", "embedding"}
            if not required.issubset(cols):
                raise StorageError(
                    f"Dataset at {self.uri} is missing columns {required - cols}. "
                    "Run doctor.py to repair/migrate it."
                )
            return ds
        except StorageError:
            raise
        except Exception:
            log.info("Creating new dataset at %s", self.uri)
            try:
                ds = deeplake.create(self.uri)
                ds.add_column("id", deeplake.types.Text())
                ds.add_column("text", deeplake.types.Text())
                ds.add_column("tags", deeplake.types.Text())
                ds.add_column("created_at", deeplake.types.Text())
                ds.add_column("updated_at", deeplake.types.Text())
                ds.add_column("embedding", deeplake.types.Embedding(self.embedder.dim))
                ds.commit()
                return ds
            except Exception as e:
                raise StorageError(
                    f"Could not open or create the dataset at {self.uri}: {e}"
                ) from e

    # ── validation ───────────────────────────────────────────────────────
    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("Memory text cannot be empty.")
        text = text.strip()
        if len(text) > MAX_TEXT_LEN:
            raise ValidationError(f"Memory text too long ({len(text)} chars, max {MAX_TEXT_LEN}).")
        return text

    # ── operations ───────────────────────────────────────────────────────
    def add(self, text: str, tags: list[str] | None = None) -> dict[str, Any]:
        """Store one memory. Duplicate-safe:
        - identical text already stored -> returns existing record, duplicate=True
        - very similar memory exists    -> stores anyway, sets similar_to=<id>
        """
        text = self._validate_text(text)
        tags = _clean_tags(tags)
        vec = self.embedder.embed(text)

        with self._lock:
            dup = self._find_near_duplicate(text, vec)
            if dup and dup["exact"]:
                log.info("Duplicate add ignored (id=%s)", dup["record"]["id"])
                return {**dup["record"], "duplicate": True}

            now = _now()
            record = {
                "id": uuid.uuid4().hex,
                "text": text,
                "tags": ",".join(tags),
                "created_at": now,
                "updated_at": now,
            }
            try:
                self.ds.append({**{k: [v] for k, v in record.items()}, "embedding": [vec]})
                self.ds.commit()
            except Exception as e:
                raise StorageError(f"Failed to write memory: {e}") from e

            out = {**record, "tags": tags}
            if dup:
                out["similar_to"] = dup["record"]["id"]
                out["similarity"] = dup["score"]
            log.info("Stored memory %s (%d chars, tags=%s)", record["id"], len(text), tags)
            return out

    def _find_near_duplicate(self, text: str, vec: np.ndarray) -> dict | None:
        if len(self.ds) == 0:
            return None
        hits = self._raw_search(vec, k=5)
        if not hits:
            return None
        norm = text.strip().lower()
        for h in hits:  # exact text match anywhere in the top hits wins
            if h["text"].strip().lower() == norm:
                return {"record": h, "score": h["score"], "exact": True}
        top = hits[0]
        if top["score"] >= NEAR_DUP_THRESHOLD:
            return {"record": top, "score": top["score"], "exact": False}
        return None

    def search(
        self,
        query: str,
        k: int = 5,
        tag: str | None = None,
        min_score: float = -1.0,
    ) -> list[dict[str, Any]]:
        """Semantic top-k search with optional tag filter and score floor."""
        query = self._validate_text(query)
        qvec = self.embedder.embed(query)
        with self._lock:
            if len(self.ds) == 0:
                return []
            # over-fetch when filtering by tag so k survives the filter
            fetch = min(len(self.ds), k * 5 if tag else k)
            hits = self._raw_search(qvec, k=fetch)
        if tag:
            t = tag.strip().lower()
            hits = [h for h in hits if t in h["tags"]]
        hits = [h for h in hits if h["score"] >= min_score]
        return hits[:k]

    def _raw_search(self, qvec: np.ndarray, k: int) -> list[dict[str, Any]]:
        arr = ",".join(str(float(x)) for x in qvec)
        k = max(1, min(int(k), len(self.ds)))
        try:
            view = self.ds.query(
                f"SELECT * ORDER BY COSINE_SIMILARITY(embedding, ARRAY[{arr}]) DESC LIMIT {k}"
            )
        except Exception as e:
            raise StorageError(f"Search failed: {e}") from e
        out = []
        for row in view:
            emb = np.asarray(row["embedding"], dtype=np.float32)
            out.append(self._row_to_dict(row, score=self._cosine(qvec, emb)))
        return out

    def update(
        self, memory_id: str, text: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Edit a memory's text and/or tags in place. Re-embeds when text changes."""
        if text is None and tags is None:
            raise ValidationError("Provide new text, new tags, or both.")
        with self._lock:
            idx = self._index_of(memory_id)
            if idx is None:
                raise NotFoundError(f"No memory with id {memory_id}.")
            try:
                if text is not None:
                    text = self._validate_text(text)
                    self.ds["text"][idx] = text
                    self.ds["embedding"][idx] = self.embedder.embed(text)
                if tags is not None:
                    self.ds["tags"][idx] = ",".join(_clean_tags(tags))
                self.ds["updated_at"][idx] = _now()
                self.ds.commit()
                return self._record_at(idx)
            except MemoryStoreError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to update memory: {e}") from e

    def list_all(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Paginated, newest first. Returns {'total', 'items'}."""
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self._lock:
            n = len(self.ds)
            start = n - 1 - offset
            items = [
                self._record_at(i)
                for i in range(start, max(-1, start - limit), -1)
            ]
        return {"total": n, "items": items}

    def get(self, memory_id: str) -> dict[str, Any]:
        with self._lock:
            idx = self._index_of(memory_id)
            if idx is None:
                raise NotFoundError(f"No memory with id {memory_id}.")
            return self._record_at(idx)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            idx = self._index_of(memory_id)
            if idx is None:
                return False
            try:
                self.ds.delete(idx)
                self.ds.commit()
            except Exception as e:
                raise StorageError(f"Failed to delete memory: {e}") from e
            log.info("Deleted memory %s", memory_id)
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = len(self.ds)
            tag_counts: dict[str, int] = {}
            newest = oldest = None
            if n:
                for t in self.ds["tags"][:]:
                    for tag in (t or "").split(","):
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                stamps = list(self.ds["created_at"][:])
                newest, oldest = max(stamps), min(stamps)
        return {
            "total_memories": n,
            "tags": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
            "newest": newest,
            "oldest": oldest,
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
        }

    # ── backup / restore ─────────────────────────────────────────────────
    def export_json(self) -> list[dict[str, Any]]:
        """All memories as plain JSON (no vectors — they're re-derived on import)."""
        with self._lock:
            return [self._record_at(i) for i in range(len(self.ds))]

    def import_json(self, records: list[dict[str, Any]]) -> int:
        """Re-add records (skips duplicates). Returns number actually imported."""
        count = 0
        for r in records:
            try:
                res = self.add(r.get("text", ""), r.get("tags") or [])
                if not res.get("duplicate"):
                    count += 1
            except ValidationError:
                log.warning("Skipping invalid record on import: %r", r)
        return count

    def count(self) -> int:
        with self._lock:
            return len(self.ds)

    # ── helpers ──────────────────────────────────────────────────────────
    def _index_of(self, memory_id: str) -> int | None:
        ids = list(self.ds["id"][:])
        try:
            return ids.index(memory_id)
        except ValueError:
            return None

    def _record_at(self, idx: int) -> dict[str, Any]:
        return {
            "id": self.ds["id"][idx],
            "text": self.ds["text"][idx],
            "tags": self._split_tags(self.ds["tags"][idx]),
            "created_at": self.ds["created_at"][idx],
            "updated_at": self.ds["updated_at"][idx],
        }

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return round(float(np.dot(a, b) / denom), 4)

    @staticmethod
    def _split_tags(tags: str) -> list[str]:
        return [t for t in (tags or "").split(",") if t]

    def _row_to_dict(self, row, score: float | None = None) -> dict[str, Any]:
        d = {
            "id": row["id"],
            "text": row["text"],
            "tags": self._split_tags(row["tags"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if score is not None:
            d["score"] = score
        return d
