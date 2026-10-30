"""Central configuration, loaded from environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ON_RAILWAY = bool(
    os.getenv("RAILWAY_ENVIRONMENT")
    or os.getenv("RAILWAY_PROJECT_ID")
    or os.getenv("RAILWAY_SERVICE_ID")
)


def _resolve_data_path(raw_path: str) -> str:
    # Keep explicit URI-style paths untouched (e.g. hub://...).
    if "://" in raw_path:
        return raw_path
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())

# Where the Deep Lake dataset lives (local folder by default).
_data_path_env = os.getenv("MEMORY_DATA_PATH", "").strip()
DATA_PATH = _resolve_data_path(_data_path_env or str(Path("data") / "memories"))

# Sentence-transformers model used for embeddings.
# all-MiniLM-L6-v2 -> 384-dim, small, fast, good general-purpose quality.
EMBED_MODEL = os.getenv("MEMORY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("MEMORY_EMBED_DIM", "384"))

# API server settings — Railway injects PORT; MEMORY_API_PORT still wins if set explicitly.
API_HOST = os.getenv("MEMORY_API_HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
API_PORT = int(os.getenv("MEMORY_API_PORT", os.getenv("PORT", "8000")))

# Optional API key. When set, every route except /health requires the
# X-API-Key header. Leave empty for local-only development.
API_KEY = os.getenv("MEMORY_API_KEY", "")

# Scheme MCP clients use to reach the API. Empty means: plain HTTP for a
# loopback address, HTTPS for anything remote.
API_SCHEME = os.getenv("MEMORY_API_SCHEME", "").strip().lower()

# Name this process reports as when it writes memories, so a shared store
# shows which assistant contributed what. Any string; lowercased on write.
AGENT_NAME = os.getenv("MEMORY_AGENT_NAME", "mcp")

# Comma-separated list of allowed CORS origins (for a future web UI)
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("MEMORY_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]


# Hosts that only accept connections from this machine. "0.0.0.0" is
# deliberately excluded: it means "bind every interface", i.e. reachable
# from outside — the opposite of loopback-only.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_deployment_config() -> None:
    """Prevent insecure or non-persistent deployments.

    Railway gets its own explicit checks (volume + key) because it's the
    documented deployment target. But any deployment — Railway or not —
    that binds to a non-loopback host without an API key is silently open
    to the network it's reachable on, so that case is refused generically
    rather than only being caught when Railway's own env vars are present.
    """
    if ON_RAILWAY and not _data_path_env:
        raise RuntimeError(
            "MEMORY_DATA_PATH must be set to a Railway Volume path, such as /data/memories."
        )
    if ON_RAILWAY and not API_KEY:
        raise RuntimeError("MEMORY_API_KEY must be set for a Railway deployment.")
    if API_HOST not in _LOOPBACK_HOSTS and not API_KEY:
        raise RuntimeError(
            f"MEMORY_API_KEY must be set when binding to a non-loopback host "
            f"({API_HOST!r}) — otherwise every memory is readable and writable "
            "by anyone who can reach this port. Set MEMORY_API_KEY, or bind to "
            "127.0.0.1 for local-only use."
        )
