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
import shutil
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
    # Deep Lake 3.x expects local filesystem paths, not file:// URIs.
    return str(Path(path).absolute())


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
            ds = deeplake.load(self.uri, verbose=False)
            # sanity: required columns present?
            cols = set(ds.tensors.keys())
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
                return self._create_dataset(self.uri)
            except Exception as e:
                raise StorageError(
                    f"Could not open or create the dataset at {self.uri}: {e}"
                ) from e

    def _create_dataset(self, path: str):
        ds = deeplake.empty(path, verbose=False)
        for name in ("id", "text", "tags", "created_at", "updated_at"):
            ds.create_tensor(name, htype="text", dtype="str", verbose=False)
        ds.create_tensor("embedding", htype="embedding", dtype="float32", verbose=False)
        ds.commit(message="create memory schema")
        return ds

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
                self.ds.append({**record, "embedding": vec})
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
            # Deep Lake 3.x SQL queries require libdeeplake, unavailable on Windows.
            rows = []
            for idx in range(len(self.ds)):
                row = self._row_at(idx)
                score = self._cosine(qvec, self._tensor_value("embedding", idx))
                rows.append(self._row_to_dict(row, score=score))
            rows.sort(key=lambda row: row["score"], reverse=True)
            return rows[:k]
        except Exception as e:
            raise StorageError(f"Search failed: {e}") from e

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
                record = self._record_at(idx)
                embedding = self._tensor_value("embedding", idx)
                if text is not None:
                    text = self._validate_text(text)
                    record["text"] = text
                    embedding = self.embedder.embed(text)
                if tags is not None:
                    record["tags"] = _clean_tags(tags)
                record["updated_at"] = _now()
                rows = []
                for row_idx in range(len(self.ds)):
                    row = self._row_at(row_idx)
                    row["tags"] = self._split_tags(row["tags"])
                    row["embedding"] = self._tensor_value("embedding", row_idx)
                    rows.append(record if row_idx == idx else row)
                rows[idx]["embedding"] = embedding
                self._rewrite_dataset(rows)
                return {**record, "tags": record["tags"]}
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
                rows = []
                for row_idx in range(len(self.ds)):
                    if row_idx == idx:
                        continue
                    row = self._row_at(row_idx)
                    row["tags"] = self._split_tags(row["tags"])
                    row["embedding"] = self._tensor_value("embedding", row_idx)
                    rows.append(row)
                self._rewrite_dataset(rows)
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
                for t in self._tensor_values("tags"):
                    for tag in (t or "").split(","):
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                stamps = self._tensor_values("created_at")
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
        ids = self._tensor_values("id")
        try:
            return ids.index(memory_id)
        except ValueError:
            return None

    def _record_at(self, idx: int) -> dict[str, Any]:
        return {
            "id": self._tensor_value("id", idx),
            "text": self._tensor_value("text", idx),
            "tags": self._split_tags(self._tensor_value("tags", idx)),
            "created_at": self._tensor_value("created_at", idx),
            "updated_at": self._tensor_value("updated_at", idx),
        }

    def _row_at(self, idx: int) -> dict[str, Any]:
        return {
            "id": self._tensor_value("id", idx),
            "text": self._tensor_value("text", idx),
            "tags": self._tensor_value("tags", idx),
            "created_at": self._tensor_value("created_at", idx),
            "updated_at": self._tensor_value("updated_at", idx),
        }

    def _tensor_values(self, name: str) -> list[Any]:
        values = self.ds[name][:].numpy()
        return np.asarray(values).reshape(-1).tolist()

    def _tensor_value(self, name: str, idx: int) -> Any:
        value = self.ds[name][idx].numpy()
        array = np.asarray(value)
        return array.reshape(-1)[0].item() if array.size == 1 else value

    def _rewrite_dataset(self, rows: list[dict[str, Any]]) -> None:
        path = Path(self.uri)
        temp_path = path.with_name(f"{path.name}.rewrite-{uuid.uuid4().hex}")
        temp_ds = None
        try:
            temp_ds = self._create_dataset(str(temp_path))
            for row in rows:
                temp_ds.append({
                    "id": row["id"],
                    "text": row["text"],
                    "tags": ",".join(row["tags"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "embedding": row["embedding"],
                })
            temp_ds.commit(message="rewrite updated memory")
            close = getattr(self.ds, "close", None)
            if close:
                close()
            shutil.rmtree(path)
            shutil.move(str(temp_path), str(path))
            self.ds = deeplake.load(self.uri, verbose=False)
        except Exception:
            if temp_ds is not None:
                close = getattr(temp_ds, "close", None)
                if close:
                    close()
            shutil.rmtree(temp_path, ignore_errors=True)
            raise

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
