"""Launch the real MCP server (stdio) with the stub embedder."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _burn.stub_patch  # noqa
import mcp_server
mcp_server.mcp.run()
