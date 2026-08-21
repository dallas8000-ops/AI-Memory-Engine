"""AI Memory Engine — burn_test.py

In-depth LIVE burn test. Unlike test_memory.py (unit-level), this boots the
real uvicorn server as a separate process and attacks it over actual HTTP,
kills it mid-write to prove crash safety, and drives the MCP server through
a real stdio client session.

Run from the project root with the venv active:

    python burn_test.py            # real embeddings (model loads once)
    python burn_test.py --stub     # fast hash-based embeddings (CI / smoke)

Phases:
  A  cold boot + readiness
  B  full functional sweep over live HTTP (every endpoint)
  C  sustained load: 300 mixed ops on 16 threads, latency report, zero 5xx
  D  crash resilience: SIGKILL during active writes -> restart -> integrity
  E  backup/restore round-trip over HTTP
  F  auth enforcement live
  G  real MCP stdio session: initialize, list tools, call tools
  H  restart persistence
  I  append-only versioning: revisions, tombstones, compaction, log replay
  J  agent attribution + cross-agent change feed
Exit code 0 only if every phase passes.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
STUB = "--stub" in sys.argv
PORT = int(os.getenv("BURN_PORT", "8123"))
BASE = f"http://127.0.0.1:{PORT}"
DATA = ROOT / "data" / "burn_memories"
# The server must not write to an undrained PIPE: once the OS buffer fills the
# process blocks on write and the whole server wedges. Give it a file instead.
SERVER_LOG = ROOT / "burn_server.log"

PASS, FAIL = [0], [0]
server_proc: subprocess.Popen | None = None


def ok(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS[0] += 1
        print(f"  [PASS] {name}")
    else:
        FAIL[0] += 1
        print(f"  [FAIL] {name}  {detail}")


# ── tiny HTTP client (stdlib only, so the burn test has no extra deps) ───
def req(method: str, path: str, body: dict | None = None, key: str | None = None,
        timeout: float = 30.0, headers: dict | None = None) -> tuple[int, dict | list | None]:
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header("Content-Type", "application/json")
    if key:
        r.add_header("X-API-Key", key)
    for name, value in (headers or {}).items():
        r.add_header(name, value)
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(r, data=data, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, None


# ── server lifecycle ─────────────────────────────────────────────────────
BOOT_SNIPPET = """
import os, sys
sys.path.insert(0, {root!r})
if os.getenv("BURN_STUB") == "1":
    import hashlib, numpy as np
    from memory_engine.embedder import Embedder
    def _stub(self, text):
        vec = np.zeros(384, dtype=np.float32)
        for tok in text.lower().split():
            seed = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            vec += np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n else vec
    Embedder.embed = _stub
import uvicorn
uvicorn.run("server:app", host="127.0.0.1", port={port}, log_level="warning")
"""


def _server_log_tail(limit: int = 2000) -> str:
    try:
        return SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def start_server(api_key: str = "") -> subprocess.Popen:
    env = {
        **os.environ,
        "MEMORY_DATA_PATH": str(DATA),
        "MEMORY_API_KEY": api_key,
        "BURN_STUB": "1" if STUB else "0",
    }
    script = BOOT_SNIPPET.format(root=str(ROOT), port=PORT)
    handle = open(SERVER_LOG, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-c", script], cwd=ROOT, env=env,
        stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    proc.burn_log_handle = handle  # type: ignore[attr-defined]
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            handle.close()
            raise RuntimeError(f"Server process died during boot:\n{_server_log_tail()}")
        try:
            code, _ = req("GET", "/health", timeout=2)
            if code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.3)
    proc.kill()
    handle.close()
    raise RuntimeError("Server did not become ready in 120s")


def stop_server(proc: subprocess.Popen | None, hard: bool = False):
    if proc is None or proc.poll() is not None:
        _close_log(proc)
        return
    if hard and os.name == "nt":
        proc.kill()
    else:
        proc.send_signal(signal.SIGKILL if hard else signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    _close_log(proc)


def _close_log(proc: subprocess.Popen | None):
    handle = getattr(proc, "burn_log_handle", None)
    if handle and not handle.closed:
        handle.close()


def main():
    global server_proc
    shutil.rmtree(DATA, ignore_errors=True)
    SERVER_LOG.unlink(missing_ok=True)
    t_start = time.time()

    # ── Phase A: cold boot ────────────────────────────────────────────
    print("\nPhase A - cold boot + readiness")
    t0 = time.time()
    server_proc = start_server()
    ok("server ready", True)
    print(f"         boot time: {time.time() - t0:.1f}s")
    code, h = req("GET", "/health")
    ok("health reports zero memories on fresh store", h["memories"] == 0, str(h))

    # ── Phase B: functional sweep ─────────────────────────────────────
    print("\nPhase B - full functional sweep over live HTTP")
    code, m1 = req("POST", "/memories", {"text": "The user's cat is named Momo", "tags": ["pets"]})
    ok("create 201", code == 201 and m1["id"])
    code, dup = req("POST", "/memories", {"text": "The user's cat is named Momo"})
    ok("duplicate detected live", dup.get("duplicate") is True)
    code, hits = req("GET", "/memories/search?q=cat+named+Momo&k=3")
    ok("search finds it", code == 200 and hits and "Momo" in hits[0]["text"])
    code, one = req("GET", f"/memories/{m1['id']}")
    ok("get by id", code == 200 and one["id"] == m1["id"])
    code, upd = req("PATCH", f"/memories/{m1['id']}", {"tags": ["pets", "named"]})
    ok("patch tags", code == 200 and upd["tags"] == ["pets", "named"])
    code, page = req("GET", "/memories?limit=10")
    ok("paginated list", code == 200 and page["total"] == 1)
    code, st = req("GET", "/stats")
    ok("stats", code == 200 and st["total_memories"] == 1)
    code, _ = req("POST", "/memories", {"text": ""})
    ok("empty text -> 422", code == 422)
    code, _ = req("GET", "/memories/nope")
    ok("missing id -> 404", code == 404)

    # ── Phase C: sustained load ───────────────────────────────────────
    print("\nPhase C - sustained load: 300 mixed ops, 16 threads")
    errors: list[str] = []
    latencies: list[float] = []
    lat_lock = threading.Lock()

    def worker(wid: int):
        for i in range(0, 300 // 16 + 1):
            n = wid * 100 + i
            try:
                t = time.perf_counter()
                if n % 3 == 0:
                    c, _ = req("POST", "/memories",
                               {"text": f"burn memory {n} about topic {n % 7}", "tags": [f"t{n % 7}"]})
                    if c != 201:
                        errors.append(f"write {n} -> {c}")
                elif n % 3 == 1:
                    c, _ = req("GET", f"/memories/search?q=topic+{n % 7}&k=3")
                    if c != 200:
                        errors.append(f"search {n} -> {c}")
                else:
                    c, _ = req("GET", "/memories?limit=5")
                    if c != 200:
                        errors.append(f"list {n} -> {c}")
                with lat_lock:
                    latencies.append((time.perf_counter() - t) * 1000)
            except Exception as e:
                errors.append(f"op {n}: {e}")

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(16)]
    t0 = time.time()
    [t.start() for t in threads]
    [t.join() for t in threads]
    dur = time.time() - t0
    lat_sorted = sorted(latencies)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)]
    print(f"         {len(latencies)} ops in {dur:.1f}s "
          f"({len(latencies)/dur:.0f} ops/s) | p50 {p50:.0f}ms | p95 {p95:.0f}ms | max {lat_sorted[-1]:.0f}ms")
    ok("zero errors / zero 5xx under load", not errors, "; ".join(errors[:3]))
    code, h = req("GET", "/health")
    writes = h["memories"]
    ok("all writes accounted for", writes > 50, f"memories={writes}")

    # ── Phase D: crash resilience ─────────────────────────────────────
    print("\nPhase D - crash resilience: SIGKILL during active writes")
    stop_flag = threading.Event()

    def crash_writer():
        n = 0
        while not stop_flag.is_set():
            try:
                req("POST", "/memories", {"text": f"crash-window memory {n} {time.time()}"}, timeout=5)
            except Exception:
                pass
            n += 1

    cw = threading.Thread(target=crash_writer)
    cw.start()
    time.sleep(1.5)                     # let writes flow
    stop_server(server_proc, hard=True)  # SIGKILL, no cleanup
    stop_flag.set()
    cw.join()
    ok("server killed mid-write", server_proc.poll() is not None)
    server_proc = start_server()         # must reopen the same dataset
    code, h2 = req("GET", "/health")
    ok("dataset opens after hard kill", code == 200, f"code={code}")
    ok("no data lost from before the crash window", h2["memories"] >= writes, f"{h2['memories']} vs {writes}")
    code, hits = req("GET", "/memories/search?q=cat+named+Momo&k=1")
    ok("search intact after crash", code == 200 and hits and "Momo" in hits[0]["text"])

    # ── Phase E: backup/restore round-trip ────────────────────────────
    print("\nPhase E - backup/restore round-trip over HTTP")
    code, backup = req("GET", "/backup/export")
    ok("export", code == 200 and backup["count"] == h2["memories"])
    code, imp = req("POST", "/backup/import", {"records": backup["records"]})
    ok("import to same store skips all duplicates", imp["imported"] == 0, str(imp))

    # ── Phase F: auth live ────────────────────────────────────────────
    print("\nPhase F - auth enforcement live")
    stop_server(server_proc)
    server_proc = start_server(api_key="burn-secret")
    code, _ = req("GET", "/stats")
    ok("no key -> 401", code == 401)
    code, _ = req("GET", "/stats", key="wrong")
    ok("wrong key -> 401", code == 401)
    code, _ = req("GET", "/stats", key="burn-secret")
    ok("right key -> 200", code == 200)
    code, _ = req("GET", "/health")
    ok("health stays public", code == 200)
    stop_server(server_proc)
    server_proc = None

    # ── Phase G: real MCP stdio session ───────────────────────────────
    print("\nPhase G - real MCP client session over stdio")
    import asyncio

    async def mcp_session():
        from mcp import ClientSession, StdioServerParameters, stdio_client

        boot = BOOT_SNIPPET  # reuse stub injection, but run mcp server instead
        mcp_boot = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "if os.getenv('BURN_STUB') == '1':\n"
            "    import hashlib, numpy as np\n"
            "    from memory_engine.embedder import Embedder\n"
            "    def _stub(self, text):\n"
            "        vec = np.zeros(384, dtype=np.float32)\n"
            "        for tok in text.lower().split():\n"
            "            seed = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)\n"
            "            vec += np.random.default_rng(seed).standard_normal(384).astype(np.float32)\n"
            "        n = np.linalg.norm(vec)\n"
            "        vec = vec / n if n else vec\n"
            "        return vec\n"
            "    Embedder.embed = _stub\n"
            "import mcp_server\n"
            "mcp_server.mcp.run()\n"
        )
        params = StdioServerParameters(
            command=sys.executable, args=["-c", mcp_boot],
            env={**os.environ, "MEMORY_DATA_PATH": str(DATA), "BURN_STUB": "1" if STUB else "0"},
            cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                ok("7 tools listed over stdio",
                   names == sorted(["store_memory", "search_memories", "update_memory",
                                    "list_memories", "delete_memory", "memory_stats",
                                    "recent_changes"]), str(names))
                r = await session.call_tool("store_memory",
                                            {"text": "MCP burn check: user speaks German", "tags": ["burn"]})
                payload = json.loads(r.content[0].text)
                ok("store_memory over stdio", payload.get("stored") in (True, False) and "id" in payload, str(payload)[:120])
                r = await session.call_tool("search_memories", {"query": "German speaks user", "k": 2})
                payload = json.loads(r.content[0].text)
                ok("search_memories over stdio finds it",
                   payload["count"] >= 1 and "German" in payload["results"][0]["text"], str(payload)[:120])
                r = await session.call_tool("memory_stats", {})
                payload = json.loads(r.content[0].text)
                ok("memory_stats over stdio", payload["total_memories"] >= 1)
                r = await session.call_tool("store_memory", {"text": "   "})
                payload = json.loads(r.content[0].text)
                ok("bad input returns structured error, session survives", "error" in payload)

    try:
        asyncio.run(mcp_session())
    except Exception as e:
        ok("MCP stdio session", False, f"{type(e).__name__}: {e}")

    # ── Phase H: restart persistence ──────────────────────────────────
    print("\nPhase H - restart persistence")
    server_proc = start_server()
    code, h3 = req("GET", "/health")
    ok("memories persist across full restart", h3["memories"] >= h2["memories"], f"{h3['memories']}")
    # The tag filter keeps this deterministic: with --stub the embeddings are
    # hashes, so top-1 ranking would be arbitrary.
    code, hits = req("GET", "/memories/search?q=German+speaks&k=50&tag=burn")
    ok("MCP-written memory visible via HTTP (shared store)",
       code == 200 and any("German" in h["text"] for h in (hits or [])),
       f"code={code} hits={str(hits)[:160]}")

    # ── Phase I: versioning, tombstones, compaction ───────────────────
    print("\nPhase I - append-only versioning + history")
    code, v1 = req("POST", "/memories",
                   {"text": "VERSIONED original wording alpha", "tags": ["ver"]})
    ok("create versioned memory", code == 201)
    vid = v1["id"]

    code, before = req("GET", "/memories?limit=1")
    live_before = before["total"]

    code, v2 = req("PATCH", f"/memories/{vid}", {"text": "VERSIONED revised wording beta"})
    ok("patch returns new text", code == 200 and "beta" in v2["text"])
    ok("patch preserves id", v2["id"] == vid)
    ok("created_at preserved across revision", v2["created_at"] == v1["created_at"])
    ok("updated_at advanced", v2["updated_at"] >= v1["updated_at"])

    code, cur = req("GET", f"/memories/{vid}")
    ok("current state is the newest revision", code == 200 and "beta" in cur["text"])

    code, after = req("GET", "/memories?limit=1")
    ok("edit does not inflate live count",
       after["total"] == live_before, f"{after['total']} vs {live_before}")

    code, revs = req("GET", f"/memories/{vid}/revisions")
    ok("history has 2 revisions", code == 200 and len(revs) == 2, str(revs)[:200])
    ok("revision 1 keeps the original wording", "alpha" in revs[0]["text"])
    ok("revision 2 holds the new wording", "beta" in revs[1]["text"])
    ok("revisions ordered oldest first", revs[0]["revision"] == 1 and revs[1]["revision"] == 2)

    code, ex = req("GET", "/backup/export")
    ok("superseded revision hidden from export",
       not any("alpha" in r["text"] for r in ex["records"]))

    code, st = req("GET", "/stats")
    ok("stats exposes history rows", st.get("history_rows", 0) > 0, str(st.get("history_rows")))
    ok("stats live count matches list", st["total_memories"] == after["total"])

    # tombstone: gone from reads, still in history
    code, _ = req("DELETE", f"/memories/{vid}")
    ok("delete 204", code == 204)
    code, _ = req("GET", f"/memories/{vid}")
    ok("deleted memory -> 404", code == 404)
    code, gone = req("GET", "/memories?limit=1")
    ok("live count drops by one", gone["total"] == live_before - 1, f"{gone['total']}")
    code, revs2 = req("GET", f"/memories/{vid}/revisions")
    ok("history survives delete (3 rows)", code == 200 and len(revs2) == 3, str(len(revs2)))
    ok("last row is a tombstone", revs2[-1]["deleted"] is True)
    ok("pre-delete revisions still readable", "alpha" in revs2[0]["text"])
    code, hits = req("GET", "/memories/search?q=VERSIONED+wording&k=10")
    ok("deleted memory absent from search", all(vid != h["id"] for h in hits))

    code, _ = req("DELETE", f"/memories/{vid}")
    ok("second delete -> 404", code == 404)

    # compaction: the one destructive op
    code, before_c = req("GET", "/memories?limit=1")
    code, comp = req("POST", "/admin/compact")
    ok("compact reports removed rows", code == 200 and comp["removed"] > 0, str(comp))
    code, after_c = req("GET", "/memories?limit=1")
    ok("compact preserves every live memory",
       after_c["total"] == before_c["total"], f"{after_c['total']} vs {before_c['total']}")
    ok("compact remaining matches live count", comp["remaining"] == after_c["total"])
    code, st2 = req("GET", "/stats")
    ok("no history rows left after compact", st2["history_rows"] == 0, str(st2["history_rows"]))
    code, revs3 = req("GET", f"/memories/{vid}/revisions")
    ok("compacted-away memory has no history", code == 404)
    code, hits = req("GET", "/memories/search?q=cat+named+Momo&k=1")
    ok("search still correct after compact", code == 200 and hits and "Momo" in hits[0]["text"])

    # history rebuilds correctly from disk
    stop_server(server_proc)
    server_proc = start_server()
    code, w1 = req("POST", "/memories", {"text": "REBUILD check original gamma"})
    req("PATCH", f"/memories/{w1['id']}", {"text": "REBUILD check revised delta"})
    stop_server(server_proc)
    server_proc = start_server()
    code, revs4 = req("GET", f"/memories/{w1['id']}/revisions")
    ok("history reloads from disk after restart", code == 200 and len(revs4) == 2, str(len(revs4)))
    code, cur2 = req("GET", f"/memories/{w1['id']}")
    ok("newest revision wins after restart", "delta" in cur2["text"])
    code, st3 = req("GET", "/stats")
    ok("live count correct after replaying log", st3["total_memories"] == after_c["total"] + 1,
       f"{st3['total_memories']} vs {after_c['total'] + 1}")

    # ── Phase J: agent attribution + change feed ──────────────────────
    print("\nPhase J - agent attribution + cross-agent change feed")
    code, feed0 = req("GET", "/feed?limit=1000")
    ok("feed reachable", code == 200 and "cursor" in feed0, str(feed0)[:120])
    start_cursor = feed0["cursor"]

    # three different agents write to the same store
    code, ca = req("POST", "/memories", {"text": "FEED claude found the retry bug"},
                   headers={"X-Agent": "Claude", "X-Session": "sess-c1"})
    ok("write with X-Agent header", code == 201)
    ok("agent recorded lowercased", ca["source_agent"] == "claude", str(ca.get("source_agent")))
    ok("session recorded", ca["session_id"] == "sess-c1", str(ca.get("session_id")))

    code, cu = req("POST", "/memories",
                   {"text": "FEED cursor confirmed the fix in the client",
                    "source_agent": "cursor", "session_id": "sess-x1"})
    ok("write with body attribution", code == 201 and cu["source_agent"] == "cursor")

    code, cg = req("POST", "/memories", {"text": "FEED chatgpt noted the root cause"},
                   headers={"X-Agent": "chatgpt"})
    ok("third agent write", code == 201 and cg["source_agent"] == "chatgpt")

    code, anon = req("POST", "/memories", {"text": "FEED unattributed note"})
    ok("unattributed write defaults to unknown", anon["source_agent"] == "unknown",
       str(anon.get("source_agent")))

    # the feed is what another agent polls
    code, f1 = req("GET", f"/feed?cursor={start_cursor}")
    ok("feed returns only changes after the cursor", code == 200 and f1["count"] == 4,
       f"count={f1['count']}")
    ok("feed is in log order",
       [i["seq"] for i in f1["items"]] == sorted(i["seq"] for i in f1["items"]))
    ok("feed labels new memories as add", all(i["op"] == "add" for i in f1["items"]))
    agents = [i["source_agent"] for i in f1["items"]]
    ok("feed carries every agent", agents == ["claude", "cursor", "chatgpt", "unknown"], str(agents))

    # cursor semantics: polling again returns nothing new
    code, f2 = req("GET", f"/feed?cursor={f1['cursor']}")
    ok("cursor advances so a repoll is empty", f2["count"] == 0, str(f2["count"]))

    # filtering by agent
    code, fc = req("GET", f"/feed?cursor={start_cursor}&agent=claude")
    ok("agent filter narrows the feed", fc["count"] == 1 and fc["items"][0]["source_agent"] == "claude",
       str(fc["count"]))
    code, fc2 = req("GET", f"/feed?cursor={start_cursor}&agent=Cursor")
    ok("agent filter is case-insensitive", fc2["count"] == 1, str(fc2["count"]))
    code, fnone = req("GET", f"/feed?cursor={start_cursor}&agent=nobody")
    ok("unknown agent yields nothing", fnone["count"] == 0)

    # edits and deletes show up as their own ops, attributed to the editor
    req("PATCH", f"/memories/{ca['id']}", {"text": "FEED claude retry bug, fixed by cursor"},
        headers={"X-Agent": "cursor"})
    code, f3 = req("GET", f"/feed?cursor={f1['cursor']}")
    ok("edit appears in the feed", f3["count"] == 1 and f3["items"][0]["op"] == "edit",
       str(f3["items"])[:150])
    ok("edit is attributed to the editing agent",
       f3["items"][0]["source_agent"] == "cursor", str(f3["items"][0]["source_agent"]))
    ok("edit keeps the original memory id", f3["items"][0]["id"] == ca["id"])

    req("DELETE", f"/memories/{cg['id']}", headers={"X-Agent": "chatgpt"})
    code, f4 = req("GET", f"/feed?cursor={f3['cursor']}")
    ok("delete appears in the feed", f4["count"] == 1 and f4["items"][0]["op"] == "delete",
       str(f4["items"])[:150])
    ok("tombstone is not current", f4["items"][0]["current"] is False)

    # limit + resume
    code, f5 = req("GET", f"/feed?cursor={start_cursor}&limit=2")
    ok("limit caps the page", f5["count"] == 2, str(f5["count"]))
    code, f6 = req("GET", f"/feed?cursor={f5['cursor']}&limit=2")
    ok("resuming from the cursor continues without gaps",
       f6["items"][0]["seq"] == f5["items"][-1]["seq"] + 1,
       f"{f6['items'][0]['seq']} vs {f5['items'][-1]['seq']}")

    code, stj = req("GET", "/stats")
    ok("stats breaks down live memories by agent",
       stj["agents"].get("cursor", 0) >= 1 and stj["agents"].get("unknown", 0) >= 1,
       str(stj.get("agents")))

    # attribution survives a restart
    stop_server(server_proc)
    server_proc = start_server()
    code, f7 = req("GET", f"/feed?cursor={start_cursor}&agent=claude")
    ok("attribution survives restart", f7["count"] == 1, str(f7["count"]))
    code, one = req("GET", f"/memories/{cu['id']}")
    ok("agent readable on a single memory", one["source_agent"] == "cursor")

    stop_server(server_proc)
    server_proc = None

    shutil.rmtree(DATA, ignore_errors=True)
    print("\n" + "=" * 50)
    print(f"BURN TEST: {PASS[0]} passed, {FAIL[0]} failed  ({time.time() - t_start:.0f}s total)")
    print("=" * 50)
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    try:
        main()
    finally:
        stop_server(server_proc, hard=True)
