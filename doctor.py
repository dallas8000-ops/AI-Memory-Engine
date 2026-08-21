"""AI Memory Engine — doctor.py

Diagnoses the environment, auto-fixes the common causes of test failures,
then runs test_memory.py and loops until tests pass or no fix applies.

Run from the project root with the venv active:

    python doctor.py            # diagnose + fix + run tests
    python doctor.py --check    # diagnose only, change nothing

Auto-fixes it can apply:
  1. Missing dependencies            -> pip install
  2. Embedding dim mismatch in .env  -> rewrite MEMORY_EMBED_DIM to the model's real dim
  3. Stale test dataset / lock files -> delete data/test_memories
  4. Corrupted main dataset          -> back it up, recreate empty
  5. Model/dataset dim mismatch      -> back up + re-embed every memory (migration)

Anything else is reported with the failing output so you can debug the logic.
"""
from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CHECK_ONLY = "--check" in sys.argv
MAX_ROUNDS = 3
ROOT = Path(__file__).parent

REQUIRED = {
    "deeplake": "deeplake",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sentence_transformers": "sentence-transformers",
    "mcp": "mcp[cli]",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "numpy": "numpy",
}

fixed: list[str] = []
manual: list[str] = []


def say(msg: str):
    print(msg, flush=True)


def fix(msg: str):
    fixed.append(msg)
    say(f"  [FIXED] {msg}")


def needs_human(msg: str):
    manual.append(msg)
    say(f"  [MANUAL] {msg}")


# ── check 0: python version ──────────────────────────────────────────────
def check_python():
    say("\n[0] Python version")
    v = sys.version_info
    say(f"  Python {v.major}.{v.minor}.{v.micro}")
    if not (3, 10) <= (v.major, v.minor) <= (3, 12):
        needs_human(
            f"Python {v.major}.{v.minor} is outside the well-supported 3.10-3.12 range; "
            "if installs fail, recreate the venv with 3.12."
        )


# ── check 1: dependencies ────────────────────────────────────────────────
def check_deps() -> bool:
    say("\n[1] Dependencies")
    missing = []
    for module, pkg in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pkg)
    if not missing:
        say("  all present")
        return True
    say(f"  missing: {', '.join(missing)}")
    if CHECK_ONLY:
        needs_human(f"Install missing packages: pip install {' '.join(missing)}")
        return False
    r = subprocess.run([sys.executable, "-m", "pip", "install", *missing])
    if r.returncode == 0:
        fix(f"installed {', '.join(missing)}")
        importlib.invalidate_caches()
        return True
    needs_human(f"pip install failed for: {', '.join(missing)} - see output above")
    return False


# ── check 2: embedding model vs configured dim ───────────────────────────
def real_model_dim() -> int | None:
    try:
        from sentence_transformers import SentenceTransformer
        from memory_engine import config

        model = SentenceTransformer(config.EMBED_MODEL)
        return int(model.get_sentence_embedding_dimension())
    except Exception as e:
        needs_human(f"Could not load embedding model: {e}")
        return None


def check_dim(model_dim: int | None):
    say("\n[2] Embedding dimension config")
    if model_dim is None:
        return
    from memory_engine import config

    say(f"  model={config.EMBED_MODEL} real dim={model_dim}, configured={config.EMBED_DIM}")
    if model_dim == config.EMBED_DIM:
        say("  ok")
        return
    if CHECK_ONLY:
        needs_human(f"Set MEMORY_EMBED_DIM={model_dim} in .env")
        return
    env = ROOT / ".env"
    lines = env.read_text().splitlines() if env.exists() else []
    lines = [l for l in lines if not l.startswith("MEMORY_EMBED_DIM=")]
    lines.append(f"MEMORY_EMBED_DIM={model_dim}")
    env.write_text("\n".join(lines) + "\n")
    config.EMBED_DIM = model_dim  # patch the running process too
    fix(f"rewrote MEMORY_EMBED_DIM={model_dim} in .env")


# ── check 3: stale test dataset ──────────────────────────────────────────
def check_test_dataset():
    say("\n[3] Stale test dataset")
    p = ROOT / "data" / "test_memories"
    if not p.exists():
        say("  none (good)")
        return
    if CHECK_ONLY:
        needs_human(f"Delete stale test dataset: {p}")
        return
    shutil.rmtree(p, ignore_errors=True)
    fix("removed stale data/test_memories")


# ── check 4/5: main dataset health + dim migration ───────────────────────
def check_main_dataset(model_dim: int | None):
    say("\n[4] Main dataset health")
    from memory_engine import config

    path = ROOT / config.DATA_PATH
    if not path.exists():
        say("  no dataset yet (will be created on first use)")
        return
    import deeplake

    try:
        ds = deeplake.load(str(path), verbose=False)
        n = len(ds)
        say(f"  opens fine, {n} memories")
    except Exception as e:
        if CHECK_ONLY:
            needs_human(f"Dataset won't open ({e}); back it up and recreate")
            return
        backup = path.with_name(f"memories_corrupt_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.move(str(path), str(backup))
        fix(f"dataset was corrupt; moved to {backup.name}, a fresh one will be created")
        return

    # dim mismatch between stored embeddings and current model -> re-embed
    if model_dim is None or n == 0:
        return
    try:
        import numpy as np

        stored_dim = int(np.asarray(ds["embedding"][0].numpy()).shape[-1])
    except Exception:
        return
    if stored_dim == model_dim:
        return
    say(f"  stored embeddings are {stored_dim}-dim but model outputs {model_dim}-dim")
    if CHECK_ONLY:
        needs_human("Re-embed the dataset (run doctor.py without --check)")
        return
    texts, tags = _live_rows(ds)
    del ds
    backup = path.with_name(f"memories_backup_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.move(str(path), str(backup))
    from memory_engine.store import MemoryStore

    store = MemoryStore()  # creates fresh dataset at the configured path
    for t, tg in zip(texts, tags):
        if t.strip():
            store.add(t, [x for x in (tg or "").split(",") if x])
    fix(f"re-embedded {len(texts)} memories at {model_dim}-dim (backup: {backup.name})")


def _live_rows(ds) -> tuple[list[str], list[str]]:
    """(texts, tags) for current memories only.

    The store is an append-only log, so replaying every row would resurrect
    deleted memories and duplicate superseded revisions.
    """
    import numpy as np

    texts = ds["text"][:].numpy().reshape(-1).tolist()
    tags = ds["tags"][:].numpy().reshape(-1).tolist()
    if "deleted" not in ds.tensors or "id" not in ds.tensors:
        return texts, tags
    ids = ds["id"][:].numpy().reshape(-1).tolist()
    flags = np.asarray(ds["deleted"][:].numpy()).reshape(-1)
    newest = {mid: i for i, mid in enumerate(ids)}
    live = sorted(i for i in newest.values() if flags[i] == 0)
    return [texts[i] for i in live], [tags[i] for i in live]


# ── run the test suite ───────────────────────────────────────────────────
def run_tests() -> tuple[bool, str]:
    say("\n[5] Running test_memory.py")
    if not (ROOT / "test_memory.py").exists():
        needs_human("test_memory.py not found in project root")
        return False, ""
    r = subprocess.run(
        [sys.executable, "test_memory.py"], capture_output=True, text=True, cwd=ROOT
    )
    out = r.stdout + r.stderr
    print(out)
    return r.returncode == 0, out


# known failure signatures in test output -> automated remedy
def try_targeted_fix(output: str) -> bool:
    """Return True if we applied a fix worth re-running for."""
    rules = [
        (r"ModuleNotFoundError: No module named '(\w+)'", "missing module"),
        (r"(lock|Lock).*(held|exists|timeout)", "lock file"),
        (r"(dimension|shape).*(mismatch|expected)", "dim mismatch"),
    ]
    for pattern, kind in rules:
        m = re.search(pattern, output)
        if not m:
            continue
        if kind == "missing module":
            mod = m.group(1)
            pkg = REQUIRED.get(mod, mod)
            subprocess.run([sys.executable, "-m", "pip", "install", pkg])
            fix(f"installed missing module {pkg}")
            return True
        if kind == "lock file":
            shutil.rmtree(ROOT / "data" / "test_memories", ignore_errors=True)
            fix("cleared locked test dataset")
            return True
        if kind == "dim mismatch":
            check_dim(real_model_dim())
            return True
    return False


# ── main ─────────────────────────────────────────────────────────────────
def main():
    say("=" * 50)
    say("AI Memory Engine doctor" + (" (check-only)" if CHECK_ONLY else ""))
    say("=" * 50)

    check_python()
    deps_ok = check_deps()
    model_dim = real_model_dim() if deps_ok else None
    check_dim(model_dim)
    check_test_dataset()
    check_main_dataset(model_dim)

    if CHECK_ONLY:
        report(tests_ran=False, passed=False)
        return

    passed = False
    for round_no in range(1, MAX_ROUNDS + 1):
        passed, output = run_tests()
        if passed:
            break
        say(f"\nTests failed (round {round_no}). Looking for a known cause...")
        if not try_targeted_fix(output):
            import os
            if os.getenv("ANTHROPIC_API_KEY"):
                say("\nHanding off to AI triage (triage.py)...")
                subprocess.run([sys.executable, "triage.py", "--from-doctor"], cwd=ROOT)
                needs_human("Review the AI triage findings above, apply the fix, re-run doctor.py")
            else:
                needs_human(
                    "Test failures don't match a known auto-fixable cause - "
                    "this is likely a logic bug. Set ANTHROPIC_API_KEY in .env and run "
                    "'python triage.py' to have AI pinpoint the source, or see the [FAIL] lines above."
                )
            break
    report(tests_ran=True, passed=passed)


def report(tests_ran: bool, passed: bool):
    say("\n" + "=" * 50)
    say("SUMMARY")
    say("=" * 50)
    say(f"Auto-fixes applied: {len(fixed)}")
    for f in fixed:
        say(f"  - {f}")
    if manual:
        say(f"Needs your attention: {len(manual)}")
        for m in manual:
            say(f"  - {m}")
    if tests_ran:
        say(f"\nFinal test result: {'ALL PASSING' if passed else 'STILL FAILING'}")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
