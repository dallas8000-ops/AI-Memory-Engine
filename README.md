# AI Memory Engine

A local semantic memory system. Store facts and notes, then recall them by *meaning*, not keywords. Built with Deep Lake (vector storage), sentence-transformers (embeddings), FastAPI (REST API), and MCP (so Claude Desktop can use it as long-term memory).

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
| PATCH | `/memories/{id}` | Edit text and/or tags (auto re-embeds) |
| DELETE | `/memories/{id}` | Delete by id |
| GET | `/stats` | Totals, tag breakdown, model info |
| GET | `/backup/export` | Full JSON backup |
| POST | `/backup/import` | Restore from backup (skips duplicates) |
| GET | `/health` | Status + memory count |

Set `MEMORY_API_KEY=yoursecret` in `.env` to require an `X-API-Key` header on every route except `/health`. Leave it empty for local-only use.

Quick test:

```powershell
curl -X POST http://127.0.0.1:8000/memories -H "Content-Type: application/json" -d "{\"text\": \"My cat is named Momo\", \"tags\": [\"pets\"]}"
curl "http://127.0.0.1:8000/memories/search?q=what+is+my+cat+called"
```

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

Restart Claude Desktop and it gains six tools: `store_memory`, `search_memories`, `update_memory`, `list_memories`, `delete_memory`, `memory_stats`. Tools never crash the server - failures come back as structured errors the model can react to.

Note: run either the REST API or the MCP server against the dataset at one time — the local Deep Lake dataset isn't designed for two writer processes at once.

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
triage.py                AI triage: pinpoints logic bugs in source
mcp_server.py            MCP server (stdio) for Claude Desktop
memory_engine/
  config.py              settings from .env
  embedder.py            sentence-transformers wrapper
  store.py               Deep Lake vector store (add/search/list/delete)
data/memories/           the vector database (created on first run)
```

## Configuration (.env)

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_DATA_PATH` | `data/memories` | Where the dataset lives |
| `MEMORY_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `MEMORY_EMBED_DIM` | `384` | Must match the model's output size |
| `MEMORY_API_HOST` / `MEMORY_API_PORT` | `127.0.0.1:8000` | API bind address |
| `MEMORY_API_KEY` | *(empty)* | When set, API requires `X-API-Key` header |
| `MEMORY_CORS_ORIGINS` | `http://localhost:3000` | Allowed browser origins |
