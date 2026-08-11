"""Launch the real API server with the stub embedder, for burn testing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _burn.stub_patch  # noqa: patch before server import
import uvicorn
uvicorn.run("server:app", host="127.0.0.1", port=8123, log_level="warning")
