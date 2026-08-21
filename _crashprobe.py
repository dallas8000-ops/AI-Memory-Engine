"""Deterministic crash-recovery probe.

Phase D of the burn test kills the server at a random instant, so it only
sometimes produces each failure mode. This reproduces all three on purpose:

  1. torn HEAD          - killed mid-flush, tensor meta left zero-length
  2. stale lock         - killed while holding the dataset lock
  3. torn HEAD + lock   - both, which needs the lock cleared before reset
  4. torn chunk         - chunk half-written, so the dataset opens but rows
                          cannot be read out of it

Run:  python _crashprobe.py
"""
import glob
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
P = "data/_crash"
ROWS = 20

PASS = [0]
FAIL = [0]


def ok(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS[0] += 1
        print(f"  [PASS] {name}")
    else:
        FAIL[0] += 1
        print(f"  [FAIL] {name}  {detail}")


def stub(self, text):
    v = np.zeros(384, dtype=np.float32)
    for tok in text.lower().split():
        s = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        v += np.random.default_rng(s).standard_normal(384).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


CHILD = f"""
import sys, os
sys.path.insert(0, {str(ROOT)!r})
from _crashprobe import stub
from memory_engine.embedder import Embedder
Embedder.embed = stub
from memory_engine.store import MemoryStore
s = MemoryStore(data_path={P!r})
for i in range({ROWS}):
    s.add("crash row %d" % i)
print("child wrote", s.count(), flush=True)
os._exit(9)          # no cleanup, no unlock, no commit
"""


def _clear_wal():
    for stray in Path(P).parent.glob("_crash.wal*"):
        stray.unlink()


def write_rows_then_die() -> int:
    """Fresh dataset, ROWS writes, then a hard kill with no cleanup."""
    shutil.rmtree(P, ignore_errors=True)
    Path(P).parent.mkdir(parents=True, exist_ok=True)
    _clear_wal()
    r = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "child wrote" in line:
            return int(line.split()[-1])
    raise RuntimeError(f"child failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def tear_head():
    """Zero a tensor meta file, exactly what a kill mid-flush leaves."""
    metas = [f for f in glob.glob(P + "/**/*", recursive=True)
             if os.path.isfile(f) and "tensor_meta" in os.path.basename(f)]
    open(metas[0], "wb").close()


def plant_lock():
    """Recreate the lock a killed writer never released."""
    (Path(P) / "dataset_lock.lock").write_bytes(b"\x00" * 18)


def truncate_chunk():
    """Half-write a chunk: the dataset still opens, reading a sample fails."""
    chunks = [f for f in glob.glob(P + "/**/chunks/*", recursive=True)
              if os.path.isfile(f) and os.path.getsize(f) > 64]
    if not chunks:
        raise RuntimeError("no chunk file found to truncate")
    target = max(chunks, key=os.path.getsize)
    with open(target, "r+b") as f:
        f.truncate(os.path.getsize(target) // 3)
    return target


def reopen():
    from memory_engine.store import MemoryStore
    return MemoryStore(data_path=P)


def case(name: str, torn: bool, locked: bool, chunk: bool = False):
    print(f"\n{name}")
    written = write_rows_then_die()
    ok("child wrote all rows before dying", written == ROWS, f"{written}")
    if torn:
        tear_head()
    if chunk:
        truncate_chunk()
    if locked:
        plant_lock()

    try:
        s = reopen()
    except Exception as e:
        ok("dataset reopens", False, f"{type(e).__name__}: {e}")
        return
    ok("dataset reopens", True)
    ok("no rows lost", s.count() == ROWS, f"count={s.count()}")
    ok("dataset is writable", not getattr(s.ds, "read_only", False))
    ok("search works", bool(s.search("crash row 7", k=1)))
    n = s.count()
    s.add("post recovery write")
    ok("accepts new writes", s.count() == n + 1, f"count={s.count()}")

    del s
    s2 = reopen()  # a clean restart must see the recovered state
    ok("recovered state survives a clean restart", s2.count() == n + 1, f"count={s2.count()}")
    del s2


if __name__ == "__main__":
    from memory_engine.embedder import Embedder

    Embedder.embed = stub

    case("1. torn HEAD (killed mid-flush)", torn=True, locked=False)
    case("2. stale lock (killed holding the lock)", torn=False, locked=True)
    case("3. torn HEAD + stale lock", torn=True, locked=True)
    case("4. torn chunk (opens, but rows unreadable)", torn=False, locked=True, chunk=True)

    shutil.rmtree(P, ignore_errors=True)
    _clear_wal()

    print(f"\n{'=' * 44}")
    print(f"CRASH RECOVERY: {PASS[0]} passed, {FAIL[0]} failed")
    print("=" * 44)
    sys.exit(1 if FAIL[0] else 0)
