"""AI Memory Engine — MCP server (production).

Exposes the memory store as tools for MCP clients (Claude Desktop,
Cursor, VS Code). Tools never raise: they return structured
{"error": ...} payloads so the client model can react gracefully.

Dev/test:  mcp dev mcp_server.py
Direct:    python mcp_server.py   (stdio transport)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

try:  # MCP SDK >= 2.0
    from mcp.server import MCPServer as _Server
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from memory_engine import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memory_engine.mcp")

mcp = _Server("ai-memory-engine")

_api_process: subprocess.Popen | None = None

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
_scheme = config.API_SCHEME or (
    "http" if config.API_HOST in _LOOPBACK_HOSTS else "https"
)
_api_url = urllib.parse.urlunsplit(
    (_scheme, f"{config.API_HOST}:{config.API_PORT}", "", "", "")
)
# One id per server process, so a store shared with other agents can group
# everything this session wrote.
_session_id = uuid.uuid4().hex


def _ensure_api() -> None:
    """Start one local REST writer when no API process is already running."""
    global _api_process
    try:
        with urllib.request.urlopen(f"{_api_url}/health", timeout=1):
            return
    except Exception:
        pass

    if _api_process is None or _api_process.poll() is not None:
        root = Path(__file__).resolve().parent
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        _api_process = subprocess.Popen(
            [sys.executable, str(root / "server.py")],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{_api_url}/health", timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("AI Memory Engine REST API did not become ready")


def _request(method: str, path: str, body: dict | None = None):
    _ensure_api()
    request = urllib.request.Request(f"{_api_url}{path}", method=method)
    request.add_header("Content-Type", "application/json")
    request.add_header("X-Agent", config.AGENT_NAME)
    request.add_header("X-Session", _session_id)
    if config.API_KEY:
        request.add_header("X-API-Key", config.API_KEY)
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(request, data=data, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            detail = json.loads(raw).get("detail", raw.decode())
        except Exception:
            detail = raw.decode(errors="replace")
        raise RuntimeError(f"API {error.code}: {detail}") from error


def _safe(fn, *args, **kwargs):
    """Run a store operation; convert failures into structured errors."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.warning("Tool error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def store_memory(text: str, tags: list[str] | None = None) -> dict:
    """Save a memory (fact, preference, decision, or note) for later semantic recall.

    Safe to call with the same fact twice: exact duplicates are detected and
    the existing record is returned instead of storing a copy.

    Args:
        text: The content to remember, e.g. "The user's cat is named Momo".
        tags: Optional lowercase labels for grouping, e.g. ["personal", "pets"].
    """
    result = _safe(_request, "POST", "/memories", {"text": text, "tags": tags or []})
    if "error" in result:
        return result
    return {"stored": not result.get("duplicate", False), **result}


@mcp.tool()
def search_memories(query: str, k: int = 5, tag: str | None = None) -> dict:
    """Semantically search stored memories; returns the top matches with scores.

    Args:
        query: Natural-language description of what to recall.
        k: Maximum number of results (default 5).
        tag: Optional tag to restrict results to, e.g. "project".
    """
    params = f"q={urllib.parse.quote(query)}&k={k}"
    if tag:
        params += f"&tag={urllib.parse.quote(tag)}"
    result = _safe(_request, "GET", f"/memories/search?{params}")
    if isinstance(result, dict) and "error" in result:
        return result
    return {"count": len(result), "results": result}


@mcp.tool()
def update_memory(memory_id: str, text: str | None = None, tags: list[str] | None = None) -> dict:
    """Edit an existing memory's text and/or tags (get ids from search/list).

    Args:
        memory_id: The id of the memory to change.
        text: New text (re-embedded automatically), or None to keep current.
        tags: New full tag list, or None to keep current.
    """
    return _safe(_request, "PATCH", f"/memories/{memory_id}", {"text": text, "tags": tags})


@mcp.tool()
def list_memories(limit: int = 20, offset: int = 0) -> dict:
    """List stored memories, newest first, with pagination."""
    return _safe(_request, "GET", f"/memories?limit={limit}&offset={offset}")


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Permanently delete a memory by its id."""
    result = _safe(_request, "DELETE", f"/memories/{memory_id}")
    if isinstance(result, dict) and "error" in result:
        return result
    return {"deleted": True, "id": memory_id}


@mcp.tool()
def memory_stats() -> dict:
    """Overview of the memory store: total count, tag breakdown, model info."""
    return _safe(_request, "GET", "/stats")


@mcp.tool()
def recent_changes(cursor: int = 0, agent: str | None = None, limit: int = 50) -> dict:
    """See what changed in the shared store, including what other assistants wrote.

    Use this to pick up context from another agent (ChatGPT, Cursor, Claude)
    working on the same problem, instead of re-deriving it.

    Args:
        cursor: Resume point from a previous call; 0 starts at the beginning.
        agent: Only show changes written by this agent, e.g. "cursor".
        limit: Maximum number of changes to return (default 50).

    Returns each change with `op` (add/edit/delete), `source_agent`, and a
    `cursor` to pass back next time.
    """
    params = f"cursor={cursor}&limit={limit}"
    if agent:
        params += f"&agent={urllib.parse.quote(agent)}"
    return _safe(_request, "GET", f"/feed?{params}")


if __name__ == "__main__":
    mcp.run()
