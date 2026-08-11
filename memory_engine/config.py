"""Central configuration, loaded from environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Where the Deep Lake dataset lives (local folder by default)
DATA_PATH = os.getenv("MEMORY_DATA_PATH", str(Path("data") / "memories"))

# Sentence-transformers model used for embeddings.
# all-MiniLM-L6-v2 -> 384-dim, small, fast, good general-purpose quality.
EMBED_MODEL = os.getenv("MEMORY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("MEMORY_EMBED_DIM", "384"))

# API server settings
API_HOST = os.getenv("MEMORY_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("MEMORY_API_PORT", "8000"))

# Optional API key. When set, every route except /health requires the
# X-API-Key header. Leave empty for local-only development.
API_KEY = os.getenv("MEMORY_API_KEY", "")

# Comma-separated list of allowed CORS origins (for a future web UI)
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("MEMORY_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
