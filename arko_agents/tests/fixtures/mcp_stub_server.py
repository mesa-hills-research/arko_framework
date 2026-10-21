"""A minimal MCP server over stdio for arko_agents tests (Python standard library).

It answers `initialize`, `tools/list` and `tools/call` with newline-delimited
JSON-RPC, writes a line to stderr at start, and exits with status 7 when a
call asks it to (`echo` with text "exit"). A call with text "hang" gets no
answer until the client cancels it (`notifications/cancelled`), and a call
with text "cancelled" answers with the IDs of the requests it cancelled.
"""
import json
import sys

# Distinct base64 payloads make byte preservation and image order observable.
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJb8AAAAASUVORK5CYII="
GIF = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

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


def tool_result(text):
    image = {"type": "image", "data": PNG, "mimeType": "image/png"}
    if text == "mixed-images":
        return {
            "content": [
                {"type": "text", "text": "before images"},
                image,
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///tiny.gif",
                        "mimeType": "image/gif",
                        "blob": GIF,
                    },
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///note.txt",
                        "mimeType": "text/plain",
                        "text": "resource text",
                    },
                },
                {"type": "image", "data": "/9j/", "mimeType": " IMAGE/JPG "},
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///tiny.webp",
                        "mimeType": "IMAGE/WEBP",
                        "blob": "UklGRgAAAABXRUJQ",
                    },
                },
                {"type": "text", "text": "after images"},
            ]
        }
    if text == "unsupported-content":
        return {
            "content": [
                {"type": "image", "data": "c3Zn", "mimeType": "image/svg+xml"},
                {"type": "audio", "data": "d2F2", "mimeType": "audio/wav"},
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///report.pdf",
                        "mimeType": "application/pdf",
                        "blob": "cGRm",
                    },
                },
                {
                    "type": "resource",
                    "resource": {"uri": "file:///unknown.bin", "blob": "Ymlu"},
                },
                {
                    "type": "resource_link",
                    "uri": "file:///linked.png",
                    "name": "linked image",
                    "mimeType": "image/png",
                },
            ]
        }
    if text == "image-only":
        return {"content": [image]}
    if text == "image-structured":
        return {"content": [image], "structuredContent": {"width": 1, "height": 1}}
    if text == "structured-only":
        return {"content": [], "structuredContent": {"answer": 42}}
    if text == "unsupported-structured":
        result = tool_result("unsupported-content")
        result["structuredContent"] = {"answer": 42}
        return result
    if text == "error-with-image":
        return {
            "content": [{"type": "text", "text": "render failed"}, image],
            "isError": True,
        }
    if text == "error-image-only":
        return {"content": [image], "isError": True}
    return {"content": [{"type": "text", "text": "echo: " + text}]}


hanging = set()
cancelled = []

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "notifications/cancelled":
        request_id = message["params"]["requestId"]
        if request_id in hanging:
            hanging.discard(request_id)
            cancelled.append(request_id)
        continue
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
        if text == "hang":
            hanging.add(message["id"])
            continue
        if text == "cancelled":
            reply(message["id"], {"content": [{"type": "text", "text": json.dumps(cancelled)}]})
            continue
        reply(message["id"], tool_result(text))
    else:
        sys.stdout.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "unknown"}}
            )
            + "\n"
        )
        sys.stdout.flush()
