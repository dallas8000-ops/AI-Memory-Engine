"""PHASE C: soak — 600 sequential ops, growth sanity, integrity audit, scale search."""
import shutil, sys, time, random, os
sys.path.insert(0, os.getcwd())
import _burn.stub_patch  # noqa
from memory_engine.store import MemoryStore

shutil.rmtree("/home/claude/burn_soak", ignore_errors=True)
s = MemoryStore(data_path="/home/claude/burn_soak/memories")
import logging; logging.getLogger("memory_engine.store").setLevel(logging.ERROR)

P=[0]
def ok(name, cond, detail=""):
    P[0]+=1; assert cond, f"BURN FAIL: {name} {detail}"; print(f"  ok {P[0]:2d}. {name}")

TOPICS = ["work", "family", "code", "travel", "food", "health"]
rng = random.Random(42)
expected_ids = set()
t0 = time.time()
adds = dups = deletes = updates = 0
add_times = []

print("C1. 600 mixed sequential operations")
for i in range(600):
    op = rng.random()
    if op < 0.60:
        text = f"soak fact {i}: note about {rng.choice(TOPICS)} number {rng.randint(0, 200)}"
        ta = time.perf_counter()
        r = s.add(text, tags=[rng.choice(TOPICS)])
        add_times.append(time.perf_counter() - ta)
        if r.get("duplicate"): dups += 1
        else: adds += 1; expected_ids.add(r["id"])
    elif op < 0.75:
        s.search(f"note about {rng.choice(TOPICS)}", k=5)
    elif op < 0.85 and expected_ids:
        victim = rng.choice(list(expected_ids))
        if s.delete(victim):
            deletes += 1; expected_ids.discard(victim)
    elif expected_ids:
        target = rng.choice(list(expected_ids))
        s.update(target, tags=[rng.choice(TOPICS), "updated"])
        updates += 1
dur = time.time() - t0
print(f"      {adds} adds, {dups} dups suppressed, {updates} updates, {deletes} deletes in {dur:.1f}s")
ok("bookkeeping matches store exactly", s.count() == len(expected_ids), f"{s.count()} vs {len(expected_ids)}")

print("C2. latency stability at scale (no degradation cliff)")
first50 = sum(add_times[:50]) / 50 * 1000
last50 = sum(add_times[-50:]) / 50 * 1000
print(f"      add latency: first-50 avg {first50:.0f}ms, last-50 avg {last50:.0f}ms at n={s.count()}")
ok("no pathological slowdown (<5x)", last50 < first50 * 5 + 50, f"{first50:.0f} -> {last50:.0f}ms")

print("C3. integrity audit of every surviving record")
page = s.list_all(limit=1000)
bad = [m for m in page["items"] if not (m["id"] and m["text"] and m["created_at"] and m["updated_at"])]
ok("every record complete", not bad and page["total"] == len(expected_ids), f"{len(bad)} bad")
ids = [m["id"] for m in page["items"]]
ok("no duplicate ids anywhere", len(ids) == len(set(ids)))

print("C4. search correctness at scale")
r = s.add("the soak canary memory: purple elephant zanzibar")
hits = s.search("purple elephant zanzibar canary", k=3)
ok("needle found in haystack of hundreds", hits and "zanzibar" in hits[0]["text"], str(hits[:1]))
ok("scores properly ordered at scale", all(hits[i]["score"] >= hits[i+1]["score"] for i in range(len(hits)-1)))

print("C5. full backup integrity at scale")
backup = s.export_json()
s2 = MemoryStore(data_path="/home/claude/burn_soak/restore")
imported = s2.import_json(backup)
ok("full-store restore", s2.count() == imported and imported > 0, f"{imported}")
h = s2.search("purple elephant zanzibar", k=1)
ok("restored store searchable", h and "zanzibar" in h[0]["text"])

print("C6. reopen after soak")
s3 = MemoryStore(data_path="/home/claude/burn_soak/memories")
ok("persistence after 600-op soak", s3.count() == s.count())

print(f"\nPHASE C PASSED ({P[0]} checks) — {600/dur:.0f} ops/s sequential, {s.count()} records intact")
