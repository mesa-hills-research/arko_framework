"""A minimal MCP server over stdio for arko_agents tests (Python standard library).

It answers `initialize`, `tools/list` and `tools/call` with newline-delimited
JSON-RPC, writes a line to stderr at start, and exits with status 7 when a
call asks it to (`echo` with text "exit").
"""
import json
import sys

sys.stderr.write("stub server starting\n")
sys.stderr.flush()

TOOLS = [
    {
        "name": "echo",
        "description": "Echoes text.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "write_note",
        "description": "Writes a note.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
]


def reply(message_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message:
        continue
    if method == "initialize":
        reply(
            message["id"],
            {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stub", "version": "1"},
            },
        )
    elif method == "tools/list":
        reply(message["id"], {"tools": TOOLS})
    elif method == "tools/call":
        text = message["params"].get("arguments", {}).get("text", "")
        if text == "exit":
            sys.stderr.write("stub server: asked to exit\n")
            sys.stderr.flush()
            sys.exit(7)
        reply(message["id"], {"content": [{"type": "text", "text": "echo: " + text}]})
    else:
        sys.stdout.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "unknown"}}
            )
            + "\n"
        )
        sys.stdout.flush()
