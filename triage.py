"""AI Memory Engine — triage.py

AI-powered logic-bug triage. Runs the test suite; when tests fail, sends
the failure output + your source code (with line numbers) to Claude via
the Anthropic API and prints a diagnosis that points at the exact
file / function / line range, with root cause and a suggested patch.

It never edits your code — you review and apply fixes yourself.

Setup:
    pip install anthropic
    # add to .env:  ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python triage.py                 # run tests, triage failures
    python triage.py --from-doctor   # quieter output (used by doctor.py)
    python triage.py --paste         # skip running tests; paste failure output yourself
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
MODEL = os.getenv("TRIAGE_MODEL", "claude-sonnet-4-6")

# Source files the AI is allowed to inspect, in priority order.
SOURCES = [
    "memory_engine/store.py",
    "memory_engine/embedder.py",
    "memory_engine/config.py",
    "server.py",
    "mcp_server.py",
    "test_memory.py",
]

SYSTEM_PROMPT = """You are a senior Python engineer doing failure triage on a \
semantic memory engine (Deep Lake vector store + sentence-transformers + FastAPI + MCP).

You receive: (1) failing test output, (2) the project source with line numbers.

Identify the most likely root cause of EACH failure. Distinguish logic bugs in \
the implementation from bugs in the test itself and from environment issues.

Respond with ONLY a JSON array, no prose, no markdown fences. One object per \
distinct root cause:
{
  "failure": "which [FAIL] line(s) this explains",
  "file": "path/to/file.py",
  "function": "function or method name",
  "lines": "start-end line numbers of the buggy code",
  "root_cause": "one or two sentences, concrete and specific",
  "confidence": "high|medium|low",
  "suggested_patch": "the corrected code snippet for those lines, or null if unsure",
  "verify_with": "how to confirm the fix (a command or a specific test)"
}
Order by confidence. If the failure looks environmental (imports, locks, \
dimensions), say so in root_cause and set file to null."""


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "test_memory.py"], capture_output=True, text=True, cwd=ROOT
    )
    return r.returncode == 0, r.stdout + r.stderr


def numbered(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(f"{i + 1:4d}| {line}" for i, line in enumerate(lines))


def build_context(failure_output: str) -> str:
    parts = [f"## FAILING TEST OUTPUT\n{failure_output.strip()}"]
    for rel in SOURCES:
        p = ROOT / rel
        if p.exists():
            parts.append(f"## SOURCE: {rel}\n{numbered(p)}")
    return "\n\n".join(parts)


def ask_claude(context: str) -> list[dict]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit(
            "No ANTHROPIC_API_KEY found.\n"
            "Get one at https://console.anthropic.com -> API keys, then add to .env:\n"
            "    ANTHROPIC_API_KEY=sk-ant-..."
        )
    try:
        import anthropic
    except ImportError:
        sys.exit("The 'anthropic' package is missing. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return parse_diagnoses(text)


def parse_diagnoses(text: str) -> list[dict]:
    """Tolerant JSON extraction: strips fences, finds the array."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return a JSON array. Raw output:\n{text}")
    return json.loads(cleaned[start : end + 1])


def render(diagnoses: list[dict]):
    if not diagnoses:
        print("The AI found no explainable root cause. Re-run with more context or ask in chat.")
        return
    print("\n" + "=" * 60)
    print(f"AI TRIAGE - {len(diagnoses)} finding(s)")
    print("=" * 60)
    for i, d in enumerate(diagnoses, 1):
        loc = d.get("file") or "environment (not a code bug)"
        fn = f" :: {d['function']}" if d.get("function") else ""
        ln = f" (lines {d['lines']})" if d.get("lines") else ""
        print(f"\n[{i}] {d.get('confidence', '?').upper()} - {loc}{fn}{ln}")
        print(f"    Explains: {d.get('failure', '?')}")
        print(f"    Root cause: {d.get('root_cause', '?')}")
        if d.get("suggested_patch"):
            print("    Suggested patch (review before applying!):")
            for line in str(d["suggested_patch"]).splitlines():
                print(f"        {line}")
        if d.get("verify_with"):
            print(f"    Verify with: {d['verify_with']}")
    print(
        "\nNothing was changed automatically. Apply the patch you agree with, "
        "then re-run: python test_memory.py"
    )


def main():
    if "--paste" in sys.argv:
        print("Paste the failing test output, then Ctrl-Z + Enter (Windows) / Ctrl-D:")
        failure_output = sys.stdin.read()
        if not failure_output.strip():
            sys.exit("No output pasted.")
    else:
        print("Running test suite...")
        passed, output = run_tests()
        if passed:
            print("All tests pass - nothing to triage.")
            return
        print(output)
        failure_output = output

    print("\nAsking Claude to triage the failure(s)...")
    diagnoses = ask_claude(build_context(failure_output))
    render(diagnoses)


if __name__ == "__main__":
    main()
