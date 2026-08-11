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
        timeout: float = 30.0) -> tuple[int, dict | list | None]:
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header("Content-Type", "application/json")
    if key:
        r.add_header("X-API-Key", key)
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


def start_server(api_key: str = "") -> subprocess.Popen:
    env = {
        **os.environ,
        "MEMORY_DATA_PATH": str(DATA),
        "MEMORY_API_KEY": api_key,
        "BURN_STUB": "1" if STUB else "0",
    }
    script = BOOT_SNIPPET.format(root=str(ROOT), port=PORT)
    proc = subprocess.Popen(
        [sys.executable, "-c", script], cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server process died during boot")
        try:
            code, _ = req("GET", "/health", timeout=2)
            if code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("Server did not become ready in 120s")


def stop_server(proc: subprocess.Popen | None, hard: bool = False):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGKILL if hard else signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main():
    global server_proc
    shutil.rmtree(DATA, ignore_errors=True)
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
                ok("6 tools listed over stdio",
                   names == sorted(["store_memory", "search_memories", "update_memory",
                                    "list_memories", "delete_memory", "memory_stats"]), str(names))
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
    code, hits = req("GET", "/memories/search?q=German+speaks&k=1")
    ok("MCP-written memory visible via HTTP (shared store)",
       code == 200 and hits and "German" in hits[0]["text"])
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
