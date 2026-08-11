"""PHASE B: MCP protocol burn — raw JSON-RPC over stdio against the real server process,
exactly as Claude Desktop / Cursor speak it."""
import json, os, shutil, subprocess, sys

shutil.rmtree("/home/claude/burn_mcp", ignore_errors=True)
env = {**os.environ, "MEMORY_DATA_PATH": "/home/claude/burn_mcp/memories"}
proc = subprocess.Popen([sys.executable, "_burn/run_mcp.py"], env=env,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

_id = [0]
def rpc(method, params=None, notify=False):
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None: msg["params"] = params
    if not notify:
        _id[0] += 1; msg["id"] = _id[0]
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()
    if notify: return None
    while True:
        line = proc.stdout.readline()
        if not line: raise RuntimeError("server closed stdout")
        resp = json.loads(line)
        if resp.get("id") == _id[0]:
            if "error" in resp: raise RuntimeError(f"{method} -> {resp['error']}")
            return resp["result"]

P=[0]
def ok(name, cond, detail=""):
    P[0]+=1; assert cond, f"BURN FAIL: {name} {detail}"; print(f"  ok {P[0]:2d}. {name}")

def tool_payload(result):
    """Extract the structured payload from a tools/call result."""
    if "structuredContent" in result and result["structuredContent"] is not None:
        sc = result["structuredContent"]
        return sc.get("result", sc)
    for c in result.get("content", []):
        if c.get("type") == "text":
            try: return json.loads(c["text"])
            except Exception: return c["text"]
    return result

print("B1. protocol handshake")
init = rpc("initialize", {"protocolVersion": "2025-06-18",
    "capabilities": {}, "clientInfo": {"name": "burn-test", "version": "1.0"}})
ok("initialize returns server info", init["serverInfo"]["name"] == "ai-memory-engine", str(init.get("serverInfo")))
rpc("notifications/initialized", {}, notify=True)

print("B2. tool discovery")
tools = rpc("tools/list")["tools"]
names = sorted(t["name"] for t in tools)
ok("all 6 tools advertised", names == sorted(["store_memory","search_memories","update_memory","list_memories","delete_memory","memory_stats"]), str(names))
ok("tools carry descriptions + schemas", all(t.get("description") and t.get("inputSchema") for t in tools))

print("B3. tool calls over the wire")
r = tool_payload(rpc("tools/call", {"name": "store_memory",
    "arguments": {"text": "MCP burn test: user timezone is EAT", "tags": ["burn"]}}))
ok("store_memory over protocol", r.get("stored") is True and r.get("id"), str(r)[:120])
mem_id = r["id"]
r2 = tool_payload(rpc("tools/call", {"name": "store_memory",
    "arguments": {"text": "MCP burn test: user timezone is EAT"}}))
ok("duplicate suppressed over protocol", r2.get("duplicate") is True and r2.get("id") == mem_id)
s = tool_payload(rpc("tools/call", {"name": "search_memories",
    "arguments": {"query": "timezone EAT burn", "k": 3}}))
ok("search over protocol", s.get("count", 0) >= 1 and "EAT" in s["results"][0]["text"])
u = tool_payload(rpc("tools/call", {"name": "update_memory",
    "arguments": {"memory_id": mem_id, "tags": ["burn", "tz"]}}))
ok("update over protocol", u.get("tags") == ["burn", "tz"], str(u)[:120])
st = tool_payload(rpc("tools/call", {"name": "memory_stats", "arguments": {}}))
ok("stats over protocol", st.get("total_memories") == 1)

print("B4. error shielding over the wire (bad input must not kill the server)")
e = tool_payload(rpc("tools/call", {"name": "store_memory", "arguments": {"text": "   "}}))
ok("empty text -> structured error, server alive", isinstance(e, dict) and "error" in e, str(e)[:120])
e2 = tool_payload(rpc("tools/call", {"name": "update_memory", "arguments": {"memory_id": "bogus", "text": "x"}}))
ok("bogus id -> structured error, server alive", isinstance(e2, dict) and "error" in e2)
st = tool_payload(rpc("tools/call", {"name": "memory_stats", "arguments": {}}))
ok("server still answering after errors", st.get("total_memories") == 1)
d = tool_payload(rpc("tools/call", {"name": "delete_memory", "arguments": {"memory_id": mem_id}}))
ok("delete over protocol", d.get("deleted") is True)

proc.stdin.close(); proc.terminate(); proc.wait(timeout=5)
print(f"\nPHASE B PASSED ({P[0]} checks) — full MCP protocol verified over stdio")
