# AI Memory Engine

A local semantic memory system. Store facts and notes, then recall them by *meaning*, not keywords. Built with Deep Lake (vector storage), sentence-transformers (embeddings), FastAPI (REST API), and MCP (so Claude Desktop can use it as long-term memory).

Two things make it more than a notes database:

- Storage is an **append-only revision log**, so edits and deletes never overwrite data and every version stays recoverable.
- Every write is **attributed to the agent that made it**, and `GET /feed` replays the log as a change feed — so Claude, Cursor, and ChatGPT can share findings against one store instead of each re-deriving them.

## Setup

```powershell
cd "C:\Software Projects\AI Memory Engine"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

The first run downloads the embedding model (~90 MB) automatically.

## Run the REST API

```powershell
python server.py
```

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

| Method | Route | Purpose |
|---|---|---|
| POST | `/memories` | Store a memory (duplicate-safe) |
| GET | `/memories/search?q=...&k=5&tag=...&min_score=...` | Semantic search with filters |
| GET | `/memories?limit=50&offset=0` | Paginated list, newest first |
| GET | `/memories/{id}` | Fetch one memory |
| GET | `/memories/{id}/revisions` | Full edit history, oldest first |
| PATCH | `/memories/{id}` | Edit text and/or tags (auto re-embeds) |
| DELETE | `/memories/{id}` | Delete by id (tombstone; history kept) |
| GET | `/stats` | Totals, history rows, tag/agent breakdown, model info |
| GET | `/feed?cursor=0&agent=...&limit=100` | Change feed: every revision in log order |
| POST | `/admin/compact` | Drop superseded revisions and reclaim space |
| GET | `/backup/export` | Full JSON backup |
| POST | `/backup/import` | Restore from backup (skips duplicates) |
| GET | `/health` | Status + memory count |

Set `MEMORY_API_KEY=yoursecret` in `.env` to require an `X-API-Key` header on every route except `/health`. Leave it empty for local-only use.

Quick test:

```powershell
curl -X POST http://127.0.0.1:8000/memories -H "Content-Type: application/json" -d "{\"text\": \"My cat is named Momo\", \"tags\": [\"pets\"]}"
curl "http://127.0.0.1:8000/memories/search?q=what+is+my+cat+called"
```

## How storage works

Every write **appends a row**; nothing is ever mutated or rewritten.

- **Add** appends a new row with a fresh id.
- **Edit** appends a new revision carrying the *same* id. The newest row wins; older revisions stay on disk and are readable via `/memories/{id}/revisions`.
- **Delete** appends a tombstone. The memory disappears from search, listing, `GET /memories/{id}` and exports, but its history remains.
- **Compact** (`POST /admin/compact`) is the only destructive operation: it rebuilds the dataset with just the live rows, permanently discarding superseded revisions, tombstones, and the Deep Lake commit history.

Why it matters:

- **History survives.** The previous design rebuilt the whole dataset on every edit and delete, which wiped the Deep Lake commit history each time. Append-only keeps it, which is what makes Git-style versioning possible.
- **Writes are O(1).** Edits and deletes no longer rewrite every row.
- **It's the only safe path on this stack.** Deep Lake 3.9 cannot update samples in place once appends are committed individually — its chunk encoder underflows under numpy 2.x — so appends are the sole reliable write primitive here.

The trade-off: the dataset grows with edit volume. `GET /stats` reports `history_rows`; run compaction when that number gets large and you no longer need the history.

On startup the store reads every column once into memory (embedding matrix, scalar columns, id → newest row) and serves all reads from there. Per-cell reads load and decompress a chunk each, which is ruinous on a freshly opened dataset — an export of 113 rows took over 30 seconds that way, and is now instant.

### Durability

Rows are flushed to storage on every write and committed as version checkpoints every 50 writes. Committing on *every* write instead would close a chunk per row, which made reopening the dataset take 77 seconds for 113 rows; flushing is both faster to write (38ms vs 154ms) and ~90x faster to reopen.

Because a checkpoint can lag behind the last write, each row is also appended to a write-ahead log (`<dataset>.wal.jsonl`) before it is written. On startup the store:

1. resets the dataset to its last good checkpoint if a kill left the HEAD unreadable,
2. removes lock files a killed writer never released (a stale lock otherwise forces read-only mode, which would also block the reset), and
3. replays any WAL rows missing from the dataset.

The WAL is cleared once its rows are durable. `_crashprobe.py` verifies all three recovery paths.

## Run as an MCP server (Claude Desktop)

Test it first with the MCP inspector:

```powershell
mcp dev mcp_server.py
```

Then add it to Claude Desktop's config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ai-memory-engine": {
      "command": "C:\\Software Projects\\AI Memory Engine\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Software Projects\\AI Memory Engine\\mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop and it gains seven tools: `store_memory`, `search_memories`, `update_memory`, `list_memories`, `delete_memory`, `memory_stats`, `recent_changes`. Tools never crash the server - failures come back as structured errors the model can react to.

The MCP server does not open the dataset directly; it talks to the REST API and starts one if none is running. That keeps a single writer while any number of assistants read and write through it.

## Sharing memory between assistants

Every write records **who made it**. Set it per request with an `X-Agent` header (or `source_agent` in the body); MCP clients send it automatically from `MEMORY_AGENT_NAME`. Unattributed writes are stored as `unknown`.

```powershell
curl -X POST http://127.0.0.1:8000/memories -H "Content-Type: application/json" -H "X-Agent: claude" -H "X-Session: sess-1" -d "{\"text\": \"The retry loop races on cancel\"}"
```

Because the log is append-only and ordered, `GET /feed` is just a slice of it — no separate event system. Another assistant polls it to pick up what the others learned:

```powershell
curl "http://127.0.0.1:8000/feed?cursor=0"
```

```jsonc
{
  "cursor": 42,            // pass back next time
  "count": 2,
  "items": [
    {"seq": 40, "op": "add",  "source_agent": "claude", "text": "The retry loop races on cancel", "current": true},
    {"seq": 41, "op": "edit", "source_agent": "cursor", "text": "Race fixed by cancelling before await", "current": true}
  ]
}
```

- `op` is `add`, `edit`, or `delete`; edits and deletes are attributed to whoever made them, not the original author.
- `current` marks whether the row is still the live state.
- Filter with `agent=cursor` to watch one assistant.
- **Poll with `cursor`, not `since`.** Timestamps are second-resolution, so several rows can share one; the cursor is exact.

In Claude or Cursor this is the `recent_changes` tool, so the model can ask what the others have already worked out instead of re-deriving it.

## Burn test (release gate)

Before trusting a new version, run the full live burn test — it boots the real server as a process and attacks it over actual HTTP:

```powershell
python burn_test.py          # real embeddings
python burn_test.py --stub   # fast smoke run
```

Ten phases: cold boot, functional sweep of every endpoint, 300-op / 16-thread load with latency report, **SIGKILL mid-write + recovery**, backup round-trip, live auth checks, a real MCP stdio client session, restart persistence, **append-only versioning** (revision history, tombstones, compaction, and log replay across restarts), and **agent attribution + change feed**. Exit code 0 only when everything passes.

Phase D kills the server at a random instant, so it hits the different corruption modes only some of the time. `_crashprobe.py` reproduces all three deterministically — torn HEAD, stale lock, and both together:

```powershell
python _crashprobe.py
```

## AI-powered bug triage

`doctor.py` auto-fixes environment problems. For actual **logic bugs**, `triage.py` uses Claude to read the failing tests and your source, then points you at the exact spot:

```powershell
pip install anthropic
# add to .env:   ANTHROPIC_API_KEY=sk-ant-...   (from console.anthropic.com)
python triage.py
```

Output per finding: file, function, line range, root cause, a suggested patch, and how to verify the fix. It **never edits your code** — you review and apply. When `ANTHROPIC_API_KEY` is set, `doctor.py` automatically hands off to triage whenever it hits a failure it can't fix mechanically. Use `python triage.py --paste` to triage a traceback you copied from anywhere (e.g. the API server log).

## Project layout

```
server.py                REST API (FastAPI + uvicorn)
doctor.py                auto-fixes environment issues, runs tests
burn_test.py             live burn test: boots real server, load, crash-kill, MCP stdio
_crashprobe.py           deterministic crash-recovery test (torn HEAD, stale lock)
triage.py                AI triage: pinpoints logic bugs in source
mcp_server.py            MCP server (stdio) for Claude Desktop; REST API client
memory_engine/
  config.py              settings from .env
  embedder.py            sentence-transformers wrapper
  store.py               append-only store (add/search/revisions/feed/compact)
data/memories/           the vector database (created on first run)
data/memories.wal.jsonl  write-ahead log, replayed after a crash
```

## Configuration (.env)

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_DATA_PATH` | `data/memories` | Where the dataset lives |
| `MEMORY_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `MEMORY_EMBED_DIM` | `384` | Must match the model's output size |
| `MEMORY_API_HOST` / `MEMORY_API_PORT` | `127.0.0.1:8000` | API bind address |
| `MEMORY_API_KEY` | *(empty)* | When set, API requires `X-API-Key` header |
| `MEMORY_AGENT_NAME` | `mcp` | Name this process reports when it writes memories |
| `MEMORY_CORS_ORIGINS` | `http://localhost:3000` | Allowed browser origins |

## Railway deployment

The memory database must be on persistent storage. Railway containers are replaceable, so never deploy with the default `data/memories` path.

1. Create a Railway Volume and mount it at `/data`.
2. In the Railway service Variables, set:

  ```text
  MEMORY_DATA_PATH=/data/memories
  MEMORY_API_KEY=<a-long-random-secret>
  MEMORY_CORS_ORIGINS=https://<your-web-app-domain>
  ```

3. Deploy with the repository's `railway.toml`. Railway supplies `PORT`; the service binds to it automatically and Railway probes `GET /health`.

The application intentionally refuses to start on Railway unless `MEMORY_DATA_PATH` and `MEMORY_API_KEY` are configured. This prevents both memory loss after a container replacement and unauthenticated public access to stored memories.
