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

Storage model — append-only revision log:
  Every write appends a row; nothing is ever mutated or rewritten, so the
  Deep Lake commit history stays intact and rollback remains possible.
  An edit appends a new revision carrying the same id; a delete appends a
  tombstone. The newest row for an id is its current state. compact() is
  the only operation that discards rows.

  Rows are flushed to storage immediately and committed as version
  checkpoints every CHECKPOINT_EVERY writes. Because a hard kill can leave
  Deep Lake's uncommitted HEAD unreadable, each row is first recorded in a
  fsync'd write-ahead log and replayed on the next open.

  (Deep Lake 3.9 cannot update samples in place once appends are committed
  individually — its chunk encoder underflows under numpy 2.x — so
  append-only is also the only safe write path on this stack.)

The whole scalar dataset is held in memory (see _build_cache); reads never
touch the backend, which matters because per-cell reads load a chunk each.

Schema (one row per revision):
    id            Text       uuid4 hex, shared by every revision of a memory
    text          Text       the memory content
    tags          Text       comma-separated tags ("" if none)
    created_at    Text       ISO-8601 UTC, carried forward across revisions
    updated_at    Text       ISO-8601 UTC (== created_at until edited)
    source_agent  Text       which agent wrote this revision ("unknown")
    session_id    Text       caller-supplied conversation id ("" if none)
    embedding     Embedding  float32, cosine-normalized
    deleted       int32      1 = tombstone row

Because rows are immutable and ordered, the log doubles as a change feed:
feed() is a slice of it, which is how several agents stay in sync.
"""
from __future__ import annotations

import json
import logging
import os
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
MAX_AGENT_LEN = 50
MAX_SESSION_LEN = 100
NEAR_DUP_THRESHOLD = 0.97

DEFAULT_AGENT = "unknown"

_COLUMNS = (
    "id", "text", "tags", "created_at", "updated_at", "source_agent", "session_id",
)

# Committing on every write closes a chunk per row, which makes reopening the
# dataset pathologically slow (~1s per 1-2 rows). Flushing persists the row
# without closing the chunk, so commits are taken periodically as checkpoints.
CHECKPOINT_EVERY = 50


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


def _clean_agent(agent: str | None) -> str:
    """Lowercased so feed filtering by agent is predictable."""
    name = str(agent or "").strip().lower()[:MAX_AGENT_LEN]
    return name or DEFAULT_AGENT


def _clean_session(session_id: str | None) -> str:
    return str(session_id or "").strip()[:MAX_SESSION_LEN]


class MemoryStore:
    def __init__(self, data_path: str = config.DATA_PATH, embedder: Embedder | None = None):
        self.uri = _to_uri(data_path)
        self.embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self._wal_path = Path(self.uri).with_name(Path(self.uri).name + ".wal.jsonl")
        self._wal_replaying = False
        self.ds = self._open_or_create()
        self._writes_since_checkpoint = 0
        self._build_cache()
        self._recover_from_wal()
        log.info("Memory store ready at %s (%d memories)", self.uri, self.count())

    # ── dataset lifecycle ────────────────────────────────────────────────
    def _open_or_create(self):
        try:
            ds = self._load_dataset()
            # sanity: required columns present?
            cols = set(ds.tensors.keys())
            required = {"id", "text", "tags", "created_at", "updated_at", "embedding"}
            if not required.issubset(cols):
                raise StorageError(
                    f"Dataset at {self.uri} is missing columns {required - cols}. "
                    "Run doctor.py to repair/migrate it."
                )
            self._ensure_deleted_tensor(ds)
            self._ensure_text_column(ds, "source_agent", DEFAULT_AGENT)
            self._ensure_text_column(ds, "session_id", "")
            return ds
        except StorageError:
            raise
        except Exception:
            log.info("Creating new dataset at %s", self.uri)
            # A WAL left over from a deleted dataset would resurrect its rows.
            self._wal_clear()
            try:
                return self._create_dataset(self.uri)
            except Exception as e:
                raise StorageError(
                    f"Could not open or create the dataset at {self.uri}: {e}"
                ) from e

    def _create_dataset(self, path: str):
        ds = deeplake.empty(path, verbose=False)
        for name in _COLUMNS:
            ds.create_tensor(name, htype="text", dtype="str", verbose=False)
        ds.create_tensor("embedding", htype="embedding", dtype="float32", verbose=False)
        ds.create_tensor("deleted", dtype="int32", verbose=False)
        ds.commit(message="create memory schema")
        return ds

    @staticmethod
    def _ensure_deleted_tensor(ds) -> None:
        """Datasets written before soft delete existed have no flag column."""
        if "deleted" in ds.tensors:
            return
        ds.create_tensor("deleted", dtype="int32", verbose=False)
        n = len(ds)
        if n:
            ds.deleted.extend(np.zeros((n, 1), dtype=np.int32))
        ds.commit(message="migrate: add soft-delete flag")
        log.info("Migrated dataset: added 'deleted' column for %d rows", n)

    @staticmethod
    def _ensure_text_column(ds, name: str, default: str) -> None:
        """Backfill a text column added after the dataset was created."""
        if name in ds.tensors:
            return
        ds.create_tensor(name, htype="text", dtype="str", verbose=False)
        n = len(ds)
        if n:
            ds[name].extend([default] * n)
        ds.commit(message=f"migrate: add {name}")
        log.info("Migrated dataset: added '%s' column for %d rows", name, n)

    def _load_dataset(self):
        """Load the dataset, recovering from what a hard kill leaves behind.

        Rows lost by a HEAD rollback are restored from the write-ahead log.
        """
        try:
            ds = deeplake.load(self.uri, verbose=False)
        except Exception as e:
            if "Corrupt" not in type(e).__name__:
                raise
            log.warning("Dataset HEAD unreadable (%s); resetting to last checkpoint", e)
            # A reset needs write access, so any stale lock has to go first.
            self._clear_locks()
            ds = deeplake.load(self.uri, verbose=False, reset=True)
        if getattr(ds, "read_only", False):
            ds = self._break_stale_lock(ds)
        return ds

    def _clear_locks(self) -> bool:
        """Remove lock files left behind by a killed writer."""
        if "://" in self.uri:
            return False
        removed = False
        for lock in Path(self.uri).rglob("dataset_lock.lock"):
            try:
                lock.unlink()
                removed = True
            except OSError:
                log.warning("Could not remove stale lock %s", lock)
        return removed

    def _break_stale_lock(self, ds):
        """A killed writer leaves dataset_lock.lock behind, forcing read-only.

        The store is single-writer by design, so a lock we did not take is
        stale; leaving it would make the store permanently unwritable.
        """
        if not self._clear_locks():
            log.error("Dataset opened read-only and no stale lock was found; writes will fail.")
            return ds
        log.warning("Cleared a stale dataset lock left by a previous process")
        return deeplake.load(self.uri, verbose=False)

    # ── write-ahead log ──────────────────────────────────────────────────
    def _wal_append(
        self, record: dict[str, Any], tags_str: str, agent: str, session: str,
        deleted: int, seq: int,
    ) -> None:
        entry = {
            "seq": seq,
            "id": record["id"],
            "text": record["text"],
            "tags": tags_str,
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "source_agent": agent,
            "session_id": session,
            "deleted": deleted,
        }
        try:
            with open(self._wal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            raise StorageError(f"Failed to write the write-ahead log: {e}") from e

    def _wal_clear(self) -> None:
        try:
            self._wal_path.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove the write-ahead log at %s", self._wal_path)

    def _recover_from_wal(self) -> None:
        """Replay rows the dataset lost to a crash. Caller must hold no lock."""
        if not self._wal_path.exists():
            return
        try:
            lines = self._wal_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            log.warning("Could not read the write-ahead log: %s", e)
            return

        pending = []
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a torn final line is expected after a hard kill
            if entry.get("seq", -1) >= len(self._cols["id"]):
                pending.append(entry)
        if not pending:
            return

        log.warning("Recovering %d row(s) from the write-ahead log", len(pending))
        self._wal_replaying = True
        try:
            for entry in sorted(pending, key=lambda e: e["seq"]):
                record = {
                    "id": entry["id"],
                    "text": entry["text"],
                    "tags": entry["tags"],
                    "created_at": entry["created_at"],
                    "updated_at": entry["updated_at"],
                    "source_agent": entry.get("source_agent", DEFAULT_AGENT),
                    "session_id": entry.get("session_id", ""),
                }
                vec = self.embedder.embed(entry["text"])
                self._append_row(record, vec, int(entry["deleted"]), "wal recovery")
        finally:
            self._wal_replaying = False

    # ── in-memory index ──────────────────────────────────────────────────
    def _build_cache(self) -> None:
        """Load every column into RAM, recovering from a torn chunk if needed.

        A kill can leave a chunk half-written: the dataset still opens, but
        reading a sample out of it fails. Salvage the rows that do read and
        rewrite from them; only if that yields nothing fall back to rolling
        back to the last checkpoint, which the WAL replay then tops up.
        """
        try:
            self._load_cache()
            return
        except Exception as e:
            log.warning("Dataset rows unreadable (%s); attempting salvage", e)

        if self._try_salvage():
            return

        try:
            self._clear_locks()
            self.ds = deeplake.load(self.uri, verbose=False, reset=True)
            self._load_cache()
            return
        except Exception as e:
            log.warning("Reset did not clear the damage: %s", e)

        self._quarantine_and_recreate()

    def _quarantine_and_recreate(self) -> None:
        """Last resort: set the unreadable dataset aside so the service boots.

        Rows still in the write-ahead log are replayed into the new dataset;
        anything older is preserved in the quarantined copy for doctor.py.
        """
        if "://" in self.uri:
            raise StorageError(f"Dataset at {self.uri} is unreadable and cannot be repaired.")
        path = Path(self.uri)
        quarantine = path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%d_%H%M%S}")
        close = getattr(self.ds, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass
        try:
            shutil.move(str(path), str(quarantine))
            self.ds = self._create_dataset(self.uri)
            self._load_cache()
        except Exception as e:
            raise StorageError(
                f"Dataset at {self.uri} is unreadable and could not be replaced: {e}"
            ) from e
        log.error(
            "Dataset was unreadable; moved to %s and started a new one. "
            "Recent writes will be restored from the write-ahead log; "
            "run doctor.py to recover the rest.",
            quarantine.name,
        )

    def _try_salvage(self) -> bool:
        """Rewrite the dataset from the rows that still read cleanly."""
        try:
            self._clear_locks()
            # Never reset first: a reset empties the chunk encoders, after
            # which the per-row reads below fail for every row.
            self.ds = deeplake.load(self.uri, verbose=False)
            rows, dropped = self._salvage_rows()
        except Exception as e:
            log.warning("Salvage could not read the dataset: %s", e)
            return False

        if not rows:
            log.warning("Salvage recovered no rows; falling back to a reset")
            return False

        log.warning("Salvaged %d row(s); %d unreadable and dropped", len(rows), dropped)
        try:
            self._rewrite_dataset(rows, keep_deleted=True)
            self._load_cache()
            return True
        except Exception as e:
            log.warning("Rewriting salvaged rows failed: %s", e)
            return False

    def _salvage_rows(self) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        dropped = 0
        for i in range(len(self.ds)):
            try:
                row = {name: self._cell(name, i) for name in _COLUMNS}
                row["deleted"] = int(self._cell("deleted", i))
            except Exception:
                dropped += 1
                continue
            row["embedding"] = self._cell_embedding(i, row["text"])
            rows.append(row)
        return rows, dropped

    def _cell(self, name: str, idx: int) -> Any:
        value = np.asarray(self.ds[name][idx].numpy()).reshape(-1)
        return value[0].item() if value.size else ""

    def _cell_embedding(self, idx: int, text: str) -> np.ndarray:
        """Re-embed from text when the stored vector is unreadable."""
        try:
            vec = np.asarray(
                self.ds["embedding"][idx].numpy(), dtype=np.float32
            ).reshape(-1)
            if vec.shape[0] == self.embedder.dim:
                return vec
        except Exception:
            pass
        return self.embedder.embed(text)

    def _load_cache(self) -> None:
        n = len(self.ds)
        if n == 0:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._live = np.zeros(0, dtype=bool)
            self._tombstone = np.zeros(0, dtype=bool)
            self._cols = {name: [] for name in _COLUMNS}
            self._id_index = {}
            return
        self._matrix = np.asarray(
            self.ds["embedding"][:].numpy(), dtype=np.float32
        ).reshape(n, -1)
        self._tombstone = np.asarray(self.ds["deleted"][:].numpy()).reshape(-1) != 0
        self._cols = {name: self._tensor_values(name) for name in _COLUMNS}

        newest: dict[str, int] = {}
        for i, mid in enumerate(self._cols["id"]):
            newest[mid] = i
        self._live = np.zeros(n, dtype=bool)
        self._id_index = {}
        for mid, i in newest.items():
            if not self._tombstone[i]:
                self._live[i] = True
                self._id_index[mid] = i

    def _live_indices(self) -> np.ndarray:
        return np.flatnonzero(self._live)

    def _append_row(
        self, record: dict[str, Any], vec: np.ndarray, deleted: int, message: str
    ) -> int:
        """Append one revision and return its row index. Caller holds the lock."""
        tags = record["tags"]
        tags_str = tags if isinstance(tags, str) else ",".join(tags)
        agent = _clean_agent(record.get("source_agent"))
        session = _clean_session(record.get("session_id"))
        seq = len(self._cols["id"])
        if not self._wal_replaying:
            self._wal_append(record, tags_str, agent, session, deleted, seq)
        try:
            self.ds.append({
                "id": record["id"],
                "text": record["text"],
                "tags": tags_str,
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
                "source_agent": agent,
                "session_id": session,
                "embedding": vec,
                "deleted": np.int32(deleted),
            })
            self.ds.flush()
            self._writes_since_checkpoint += 1
            if self._writes_since_checkpoint >= CHECKPOINT_EVERY:
                self.ds.commit(message=message)
                self._writes_since_checkpoint = 0
                self._wal_clear()
        except Exception as e:
            raise StorageError(f"Failed to write memory: {e}") from e

        row = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        self._matrix = np.vstack([self._matrix, row]) if len(self._matrix) else row
        self._cols["id"].append(record["id"])
        self._cols["text"].append(record["text"])
        self._cols["tags"].append(tags_str)
        self._cols["created_at"].append(record["created_at"])
        self._cols["updated_at"].append(record["updated_at"])
        self._cols["source_agent"].append(agent)
        self._cols["session_id"].append(session)
        self._tombstone = np.append(self._tombstone, bool(deleted))

        idx = len(self._live)
        self._live = np.append(self._live, False)
        prev = self._id_index.pop(record["id"], None)
        if prev is not None:
            self._live[prev] = False
        if not deleted:
            self._live[idx] = True
            self._id_index[record["id"]] = idx
        return idx

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
    def add(
        self,
        text: str,
        tags: list[str] | None = None,
        source_agent: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Store one memory. Duplicate-safe:
        - identical text already stored -> returns existing record, duplicate=True
        - very similar memory exists    -> stores anyway, sets similar_to=<id>
        """
        text = self._validate_text(text)
        tags = _clean_tags(tags)

        with self._lock:
            vec = self.embedder.embed(text)
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
                "source_agent": _clean_agent(source_agent),
                "session_id": _clean_session(session_id),
            }
            self._append_row(record, vec, deleted=0, message=f"add {record['id']}")

            out = {**record, "tags": tags}
            if dup:
                out["similar_to"] = dup["record"]["id"]
                out["similarity"] = dup["score"]
            log.info("Stored memory %s (%d chars, tags=%s)", record["id"], len(text), tags)
            return out

    def _find_near_duplicate(self, text: str, vec: np.ndarray) -> dict | None:
        if not self._id_index:
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
        with self._lock:
            qvec = self.embedder.embed(query)
            if not self._id_index:
                return []
            # over-fetch when filtering by tag so k survives the filter
            fetch = min(self.count(), k * 5 if tag else k)
            hits = self._raw_search(qvec, k=fetch)
        if tag:
            t = tag.strip().lower()
            hits = [h for h in hits if t in h["tags"]]
        hits = [h for h in hits if h["score"] >= min_score]
        return hits[:k]

    def _raw_search(self, qvec: np.ndarray, k: int) -> list[dict[str, Any]]:
        live = len(self._id_index)
        if live == 0:
            return []
        k = max(1, min(int(k), live))
        try:
            scores = self._matrix @ qvec
            scores = np.where(self._live, scores, -np.inf)
            order = np.argsort(scores)[::-1][:k]
            rows = []
            for idx in order:
                row = self._row_at(int(idx))
                rows.append(self._row_to_dict(row, score=round(float(scores[idx]), 4)))
            return rows
        except Exception as e:
            raise StorageError(f"Search failed: {e}") from e

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        tags: list[str] | None = None,
        source_agent: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Edit a memory by appending a new revision. Re-embeds when text changes."""
        if text is None and tags is None:
            raise ValidationError("Provide new text, new tags, or both.")
        with self._lock:
            idx = self._index_of(memory_id)
            if idx is None:
                raise NotFoundError(f"No memory with id {memory_id}.")
            try:
                record = self._record_at(idx)
                vec = self._matrix[idx]
                if text is not None:
                    text = self._validate_text(text)
                    record["text"] = text
                    vec = self.embedder.embed(text)
                if tags is not None:
                    record["tags"] = _clean_tags(tags)
                record["updated_at"] = _now()
                # the revision is attributed to whoever made the edit
                record["source_agent"] = _clean_agent(source_agent)
                record["session_id"] = _clean_session(session_id)
                self._append_row(record, vec, deleted=0, message=f"update {memory_id}")
                return record
            except MemoryStoreError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to update memory: {e}") from e

    def list_all(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Paginated, newest first. Returns {'total', 'items'}."""
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self._lock:
            live = self._live_indices()[::-1]
            page = live[offset:offset + limit]
            items = [self._record_at(int(i)) for i in page]
            return {"total": int(live.size), "items": items}

    def get(self, memory_id: str) -> dict[str, Any]:
        with self._lock:
            idx = self._index_of(memory_id)
            if idx is None:
                raise NotFoundError(f"No memory with id {memory_id}.")
            return self._record_at(idx)

    def delete(
        self,
        memory_id: str,
        source_agent: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Append a tombstone. History survives; see compact() to reclaim space."""
        with self._lock:
            idx = self._index_of(memory_id)
            if idx is None:
                return False
            try:
                record = self._record_at(idx)
                record["updated_at"] = _now()
                record["source_agent"] = _clean_agent(source_agent)
                record["session_id"] = _clean_session(session_id)
                self._append_row(
                    record, self._matrix[idx], deleted=1, message=f"delete {memory_id}"
                )
            except MemoryStoreError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to delete memory: {e}") from e
            log.info("Deleted memory %s", memory_id)
            return True

    def revisions(self, memory_id: str) -> list[dict[str, Any]]:
        """Every stored revision of one memory, oldest first."""
        with self._lock:
            out = []
            for i, mid in enumerate(self._cols["id"]):
                if mid == memory_id:
                    rec = self._record_at(i)
                    rec["deleted"] = bool(self._tombstone[i])
                    rec["revision"] = len(out) + 1
                    out.append(rec)
            if not out:
                raise NotFoundError(f"No memory with id {memory_id}.")
            return out

    def feed(
        self,
        cursor: int = 0,
        agent: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Revisions in log order — the cross-agent change feed.

        Poll with the returned cursor. Timestamps are only second-resolution,
        so `since` can tie across rows; the cursor cannot.
        """
        limit = max(1, min(int(limit), 1000))
        cursor = max(0, int(cursor))
        want_agent = _clean_agent(agent) if agent else None

        with self._lock:
            ids = self._cols["id"]
            total_rows = len(ids)
            next_cursor = total_rows
            items: list[dict[str, Any]] = []
            seen: set[str] = set()

            for i, mid in enumerate(ids):
                first_time = mid not in seen
                seen.add(mid)
                if self._feed_skip(i, cursor, want_agent, since):
                    continue
                if len(items) >= limit:
                    next_cursor = i  # this row has not been delivered yet
                    break
                items.append(self._feed_entry(i, first_time))

        return {"cursor": next_cursor, "count": len(items), "items": items}

    def _feed_skip(
        self, i: int, cursor: int, want_agent: str | None, since: str | None
    ) -> bool:
        if i < cursor:
            return True
        if want_agent and self._cols["source_agent"][i] != want_agent:
            return True
        return bool(since and self._cols["updated_at"][i] < since)

    def _feed_entry(self, i: int, first_time: bool) -> dict[str, Any]:
        deleted = bool(self._tombstone[i])
        if deleted:
            op = "delete"
        else:
            op = "add" if first_time else "edit"
        entry = self._record_at(i)
        entry["seq"] = i
        entry["op"] = op
        entry["deleted"] = deleted
        entry["current"] = bool(self._live[i])
        return entry

    def checkpoint(self, message: str = "checkpoint") -> bool:
        """Force a version commit of any writes made since the last one."""
        with self._lock:
            if self._writes_since_checkpoint == 0:
                return False
            try:
                self.ds.commit(message=message)
            except Exception as e:
                raise StorageError(f"Checkpoint failed: {e}") from e
            self._writes_since_checkpoint = 0
            self._wal_clear()
            return True

    def compact(self) -> dict[str, Any]:
        """Physically drop superseded revisions and tombstones.

        This is the one destructive operation: it rebuilds the dataset, so
        prior commits are discarded. Everything else preserves history.
        """
        with self._lock:
            removed = int((~self._live).sum())
            if removed == 0:
                return {"removed": 0, "remaining": self.count()}
            rows = []
            for i in self._live_indices():
                row = self._row_at(int(i))
                row["tags"] = self._split_tags(row["tags"])
                row["embedding"] = self._matrix[int(i)]
                rows.append(row)
            try:
                self._rewrite_dataset(rows)
            except Exception as e:
                raise StorageError(f"Compaction failed: {e}") from e
            self._writes_since_checkpoint = 0
            self._wal_clear()
            self._build_cache()
            log.info("Compacted dataset: dropped %d rows", removed)
            return {"removed": removed, "remaining": self.count()}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            live = self._live_indices()
            n = int(live.size)
            tag_counts: dict[str, int] = {}
            agent_counts: dict[str, int] = {}
            newest = oldest = None
            if n:
                all_tags = self._cols["tags"]
                all_stamps = self._cols["created_at"]
                all_agents = self._cols["source_agent"]
                stamps = []
                for i in live:
                    for tag in (all_tags[i] or "").split(","):
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    a = all_agents[i] or DEFAULT_AGENT
                    agent_counts[a] = agent_counts.get(a, 0) + 1
                    stamps.append(all_stamps[i])
                newest, oldest = max(stamps), min(stamps)
        return {
            "total_memories": n,
            "history_rows": int(self._live.size) - n,
            "tags": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
            "agents": dict(sorted(agent_counts.items(), key=lambda kv: -kv[1])),
            "newest": newest,
            "oldest": oldest,
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
        }

    # ── backup / restore ─────────────────────────────────────────────────
    def export_json(self) -> list[dict[str, Any]]:
        """All memories as plain JSON (no vectors — they're re-derived on import)."""
        with self._lock:
            return [self._record_at(int(i)) for i in self._live_indices()]

    def import_json(self, records: list[dict[str, Any]]) -> int:
        """Re-add records (skips duplicates). Returns number actually imported."""
        count = 0
        for r in records:
            try:
                res = self.add(
                    r.get("text", ""),
                    r.get("tags") or [],
                    source_agent=r.get("source_agent"),
                    session_id=r.get("session_id"),
                )
                if not res.get("duplicate"):
                    count += 1
            except ValidationError:
                log.warning("Skipping invalid record on import: %r", r)
        return count

    def count(self) -> int:
        with self._lock:
            return len(self._id_index)

    # ── helpers ──────────────────────────────────────────────────────────
    def _index_of(self, memory_id: str) -> int | None:
        return self._id_index.get(memory_id)

    def _record_at(self, idx: int) -> dict[str, Any]:
        return {
            "id": self._cols["id"][idx],
            "text": self._cols["text"][idx],
            "tags": self._split_tags(self._cols["tags"][idx]),
            "created_at": self._cols["created_at"][idx],
            "updated_at": self._cols["updated_at"][idx],
            "source_agent": self._cols["source_agent"][idx],
            "session_id": self._cols["session_id"][idx],
        }

    def _row_at(self, idx: int) -> dict[str, Any]:
        return {name: self._cols[name][idx] for name in _COLUMNS}

    def _tensor_values(self, name: str) -> list[Any]:
        values = self.ds[name][:].numpy()
        return np.asarray(values).reshape(-1).tolist()

    def _rewrite_dataset(
        self, rows: list[dict[str, Any]], keep_deleted: bool = False
    ) -> None:
        path = Path(self.uri)
        temp_path = path.with_name(f"{path.name}.rewrite-{uuid.uuid4().hex}")
        temp_ds = None
        try:
            temp_ds = self._create_dataset(str(temp_path))
            for row in rows:
                tags = row["tags"]
                temp_ds.append({
                    "id": row["id"],
                    "text": row["text"],
                    "tags": tags if isinstance(tags, str) else ",".join(tags),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source_agent": row.get("source_agent", DEFAULT_AGENT),
                    "session_id": row.get("session_id", ""),
                    "embedding": row["embedding"],
                    "deleted": np.int32(row.get("deleted", 0) if keep_deleted else 0),
                })
            temp_ds.commit(message="compact", allow_empty=True)
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
            "source_agent": row.get("source_agent", DEFAULT_AGENT),
            "session_id": row.get("session_id", ""),
        }
        if score is not None:
            d["score"] = score
        return d
