"""PHASE A: live HTTP burn — real uvicorn process, concurrent clients, kill/restart."""
import os, signal, subprocess, sys, threading, time, random, json
import urllib.request, urllib.error

BASE = "http://127.0.0.1:8123"
KEY = "burn-test-key-8123"
VENV_PY = sys.executable

def req(method, path, body=None, key=KEY, timeout=15):
    r = urllib.request.Request(BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            data = resp.read()
            return resp.status, json.loads(data) if data else None
    except urllib.error.HTTPError as e:
        data = e.read()
        return e.code, json.loads(data) if data else None

def start_server(env_extra=None):
    env = {**os.environ, "MEMORY_DATA_PATH": "/home/claude/burn_data/memories",
           "MEMORY_API_KEY": KEY, **(env_extra or {})}
    p = subprocess.Popen([VENV_PY, "_burn/run_server.py"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            code, _ = req("GET", "/health", key=None)
            if code == 200:
                return p
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("server never came up")

P=[0]
def ok(name, cond, detail=""):
    P[0]+=1; assert cond, f"BURN FAIL: {name} {detail}"; print(f"  ok {P[0]:2d}. {name}")

import shutil; shutil.rmtree("/home/claude/burn_data", ignore_errors=True)
print("A1. boot + auth over live HTTP")
server = start_server()
code, h = req("GET", "/health", key=None)
ok("health open, auth reported on", code == 200 and h["auth"] is True)
code, _ = req("GET", "/stats", key=None)
ok("live 401 without key", code == 401)
code, _ = req("GET", "/stats", key="wrong")
ok("live 401 with wrong key", code == 401)

print("A2. live CRUD walkthrough")
code, m = req("POST", "/memories", {"text": "Burn test memory alpha", "tags": ["burn"]})
ok("create over http", code == 201 and m["id"])
code, hits = req("GET", "/memories/search?q=alpha%20burn%20memory&k=3")
ok("search over http", code == 200 and any("alpha" in x["text"] for x in hits))
code, u = req("PATCH", f"/memories/{m['id']}", {"tags": ["burn", "edited"]})
ok("patch over http", code == 200 and u["tags"] == ["burn", "edited"])
code, _ = req("DELETE", f"/memories/{m['id']}")
ok("delete over http", code == 204)

print("A3. concurrent load: 16 threads x 25 mixed requests (400 total)")
errors, latencies, created = [], [], []
lock = threading.Lock()
def worker(wid):
    rng = random.Random(wid)
    for i in range(25):
        t0 = time.perf_counter()
        try:
            op = rng.random()
            if op < 0.5:
                code, m = req("POST", "/memories", {"text": f"load memory worker {wid} item {i} topic {rng.randint(0,5)}"})
                if code == 201:
                    with lock: created.append(m["id"])
                elif code != 201:
                    errors.append(("create", code))
            elif op < 0.75:
                code, _ = req("GET", f"/memories/search?q=topic%20{rng.randint(0,5)}&k=3")
                if code != 200: errors.append(("search", code))
            elif op < 0.9:
                code, _ = req("GET", "/memories?limit=10")
                if code != 200: errors.append(("list", code))
            else:
                with lock:
                    victim = created.pop() if created else None
                if victim:
                    code, _ = req("DELETE", f"/memories/{victim}")
                    if code not in (204, 404): errors.append(("delete", code))
        except Exception as e:
            errors.append(("exception", repr(e)))
        latencies.append((time.perf_counter() - t0) * 1000)

threads = [threading.Thread(target=worker, args=(w,)) for w in range(16)]
t0 = time.time()
[t.start() for t in threads]; [t.join() for t in threads]
dur = time.time() - t0
lat_sorted = sorted(latencies)
p50 = lat_sorted[len(lat_sorted)//2]; p99 = lat_sorted[int(len(lat_sorted)*0.99)]
print(f"      400 reqs in {dur:.1f}s ({400/dur:.0f} req/s), p50={p50:.0f}ms p99={p99:.0f}ms")
ok("zero errors under concurrent load", not errors, str(errors[:3]))
code, h = req("GET", "/health", key=None)
survivors = h["memories"]
code, page = req("GET", "/memories?limit=1000")
ok("count consistent after load", page["total"] == survivors, f"{page['total']} vs {survivors}")

print("A4. hard-kill crash safety (SIGKILL mid-life, then restart)")
server.send_signal(signal.SIGKILL); server.wait()
server = start_server()
code, h2 = req("GET", "/health", key=None)
ok("dataset survives SIGKILL + restart intact", code == 200 and h2["memories"] == survivors,
   f"{h2['memories']} vs {survivors}")
code, hits = req("GET", "/memories/search?q=load%20memory%20topic%203&k=3")
ok("search functional after crash recovery", code == 200 and len(hits) > 0)

print("A5. backup round-trip over live HTTP")
code, backup = req("GET", "/backup/export")
ok("export", code == 200 and backup["count"] == survivors)
code, imp = req("POST", "/backup/import", {"records": backup["records"]})
ok("re-import all skipped as dups", code == 200 and imp["imported"] == 0, str(imp))

server.terminate(); server.wait()
print(f"\nPHASE A PASSED ({P[0]} checks) — {400/dur:.0f} req/s sustained, p99 {p99:.0f}ms")
