"""AI Memory Engine — MCP server (production).

Exposes the memory store as tools for MCP clients (Claude Desktop,
Cursor, VS Code). Tools never raise: they return structured
{"error": ...} payloads so the client model can react gracefully.

Dev/test:  mcp dev mcp_server.py
Direct:    python mcp_server.py   (stdio transport)
"""
from __future__ import annotations

import logging

try:  # MCP SDK >= 2.0
    from mcp.server import MCPServer as _Server
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from memory_engine.store import MemoryStore, MemoryStoreError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memory_engine.mcp")

mcp = _Server("ai-memory-engine")

_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def _safe(fn, *args, **kwargs):
    """Run a store operation; convert failures into structured errors."""
    try:
        return fn(*args, **kwargs)
    except MemoryStoreError as e:
        log.warning("Tool error: %s", e)
        return {"error": str(e)}
    except Exception as e:  # never crash the server over one bad call
        log.exception("Unexpected tool error")
        return {"error": f"Unexpected error: {e}"}


@mcp.tool()
def store_memory(text: str, tags: list[str] | None = None) -> dict:
    """Save a memory (fact, preference, decision, or note) for later semantic recall.

    Safe to call with the same fact twice: exact duplicates are detected and
    the existing record is returned instead of storing a copy.

    Args:
        text: The content to remember, e.g. "The user's cat is named Momo".
        tags: Optional lowercase labels for grouping, e.g. ["personal", "pets"].
    """
    result = _safe(get_store().add, text, tags)
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
    result = _safe(get_store().search, query, k=k, tag=tag)
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
    return _safe(get_store().update, memory_id, text=text, tags=tags)


@mcp.tool()
def list_memories(limit: int = 20, offset: int = 0) -> dict:
    """List stored memories, newest first, with pagination."""
    return _safe(get_store().list_all, limit=limit, offset=offset)


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Permanently delete a memory by its id."""
    removed = _safe(get_store().delete, memory_id)
    if isinstance(removed, dict):
        return removed
    return {"deleted": removed, "id": memory_id}


@mcp.tool()
def memory_stats() -> dict:
    """Overview of the memory store: total count, tag breakdown, model info."""
    return _safe(get_store().stats)


if __name__ == "__main__":
    mcp.run()
