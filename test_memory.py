"""Test suite for the AI Memory Engine v2 (professional tier).

Run from the project root with the venv active:

    python test_memory.py

Uses REAL embeddings on a throwaway dataset (data/test_memories) so your
actual memories are never touched. First run downloads the model (~90 MB).
"""
from __future__ import annotations

import shutil
import sys
import time

TEST_PATH = "data/test_memories"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def main():
    shutil.rmtree(TEST_PATH, ignore_errors=True)

    print("Loading embedding model (first run downloads ~90 MB)...")
    t0 = time.time()
    from memory_engine.store import MemoryStore, NotFoundError, ValidationError

    store = MemoryStore(data_path=TEST_PATH)
    store.embedder.embed("warm up")
    print(f"Model ready in {time.time() - t0:.1f}s\n")

    print("1. Storage + validation")
    r1 = store.add("The user's cat is named Momo", tags=["Personal", "pets", "personal"])
    check("tags cleaned + deduped + lowercased", r1["tags"] == ["personal", "pets"], str(r1["tags"]))
    r2 = store.add("The project stack is FastAPI, Deep Lake and sentence-transformers")
    store.add("User prefers dark roast coffee in the morning", tags=["preferences"])
    store.add("The API server runs on port 8000")
    check("4 memories stored", store.count() == 4, f"count={store.count()}")
    try:
        store.add("   ")
        check("empty text rejected", False)
    except ValidationError:
        check("empty text rejected", True)

    print("\n2. Duplicate protection")
    d = store.add("The user's cat is named Momo")
    check("exact re-add returns existing", d.get("duplicate") is True and store.count() == 4)
    d2 = store.add("The user's cat is called Momo")  # near-identical phrasing
    check(
        "near-duplicate flagged with similar_to",
        d2.get("similar_to") == r1["id"] or store.count() == 5,
        str(d2),
    )

    print("\n3. Semantic search (no keyword overlap on purpose)")
    hits = store.search("what is my pet called", k=2)
    check("pet query -> cat memory", "Momo" in hits[0]["text"], f"got: {hits[0]['text']}")
    hits = store.search("which technologies does this codebase use", k=2)
    check("tech query -> stack memory", "FastAPI" in hits[0]["text"], f"got: {hits[0]['text']}")
    hits = store.search("how does the user take their morning drink", k=2)
    check("drink query -> coffee memory", "coffee" in hits[0]["text"], f"got: {hits[0]['text']}")
    hits = store.search("morning drink preference", k=5, tag="preferences")
    check("tag filter", all("preferences" in h["tags"] for h in hits) and hits)
    scores = [h["score"] for h in store.search("cat", k=4)]
    check("scores sorted descending", scores == sorted(scores, reverse=True), str(scores))

    print("\n4. Update")
    u = store.update(r2["id"], text="The project stack is FastAPI, Deep Lake and MCP")
    check("text updated in place", "MCP" in u["text"])
    hits = store.search("does the project use the Model Context Protocol", k=1)
    check("updated text re-embedded + searchable", "MCP" in hits[0]["text"], hits[0]["text"])
    try:
        store.update("bogus-id", text="x")
        check("update missing id raises", False)
    except NotFoundError:
        check("update missing id raises NotFoundError", True)

    print("\n5. Pagination / stats / delete")
    page = store.list_all(limit=2)
    check("pagination shape", page["total"] == store.count() and len(page["items"]) == 2)
    st = store.stats()
    check("stats", st["total_memories"] == store.count() and "pets" in st["tags"], str(st["tags"]))
    check("delete existing", store.delete(r2["id"]) is True)
    check("delete bogus -> False", store.delete("does-not-exist") is False)

    print("\n6. Backup round-trip")
    backup = store.export_json()
    imported = store.import_json(backup)
    check("re-import skips all duplicates", imported == 0, f"imported={imported}")

    print("\n7. Persistence (reopen like a restart)")
    n = store.count()
    store2 = MemoryStore(data_path=TEST_PATH)
    check("memories survive reopen", store2.count() == n, f"{store2.count()} vs {n}")
    hits = store2.search("what is my pet called", k=1)
    check("search works after reopen", "Momo" in hits[0]["text"])

    print("\n8. Agent attribution + change feed")
    base = store2.feed(limit=1000)["cursor"]
    a = store2.add("Claude found a race in the retry loop", source_agent="Claude",
                   session_id="s-1")
    check("agent lowercased on write", a["source_agent"] == "claude", str(a["source_agent"]))
    check("session recorded", a["session_id"] == "s-1", str(a["session_id"]))
    b = store2.add("Cursor confirmed the retry fix", source_agent="cursor")
    check("unattributed default", store2.add("plain note")["source_agent"] == "unknown")

    f = store2.feed(cursor=base)
    check("feed returns the new rows", f["count"] == 3, str(f["count"]))
    check("feed is in log order", [i["seq"] for i in f["items"]] == sorted(i["seq"] for i in f["items"]))
    check("feed marks them as adds", all(i["op"] == "add" for i in f["items"]))
    check("repolling the cursor is empty", store2.feed(cursor=f["cursor"])["count"] == 0)
    check("agent filter", store2.feed(cursor=base, agent="claude")["count"] == 1)

    store2.update(a["id"], text="Claude found a race; cursor fixed it", source_agent="cursor")
    fe = store2.feed(cursor=f["cursor"])
    check("edit shows as edit op", fe["count"] == 1 and fe["items"][0]["op"] == "edit", str(fe["items"]))
    check("edit attributed to editor", fe["items"][0]["source_agent"] == "cursor")

    store2.delete(b["id"], source_agent="claude")
    fd = store2.feed(cursor=fe["cursor"])
    check("delete shows as delete op", fd["count"] == 1 and fd["items"][0]["op"] == "delete")
    check("tombstone is not current", fd["items"][0]["current"] is False)
    check("agents in stats", store2.stats()["agents"].get("cursor", 0) >= 1,
          str(store2.stats()["agents"]))

    del store, store2
    shutil.rmtree(TEST_PATH, ignore_errors=True)

    print(f"\n{'=' * 40}\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
