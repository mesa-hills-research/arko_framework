"""A stand-in for the `claude` CLI that speaks its stream-json control protocol.

Each user message is a command: "permission", "hook", "tools", "slow" (waits
for an interrupt), "exit" (crashes); anything else is echoed back.
"""
import json
import sys

ARGS = sys.argv[1:]
SESSION = "sess-1"
state = {"model": None, "counter": 0}


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def reply(request_id, response):
    send({"type": "control_response",
          "response": {"subtype": "success", "request_id": request_id, "response": response}})


def ask(request):
    """Sends a control request to the host and waits for its answer."""
    state["counter"] += 1
    request_id = "cli_%d" % state["counter"]
    send({"type": "control_request", "request_id": request_id, "request": request})
    while True:
        message = read()
        if message.get("type") == "control_response" and message["response"]["request_id"] == request_id:
            return message["response"]
        handle(message)


def result(text, subtype="success", is_error=False):
    send({"type": "result", "subtype": subtype, "is_error": is_error, "num_turns": 1,
          "session_id": SESSION, "duration_ms": 5, "result": text, "total_cost_usd": 0.0})


def turn(text):
    send({"type": "system", "subtype": "init", "session_id": SESSION, "argv": ARGS})
    if text == "permission":
        out = json.dumps(ask({"subtype": "can_use_tool", "tool_name": "Bash",
                              "input": {"command": "ls"}, "tool_use_id": "t1",
                              "permission_suggestions": []}))
    elif text == "hook":
        out = json.dumps(ask({"subtype": "hook_callback", "callback_id": "hook_0",
                              "input": {"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                        "tool_input": {"command": "rm -rf ~"}},
                              "tool_use_id": "t2"}))
    elif text == "tools":
        def mcp(message):
            return ask({"subtype": "mcp_message", "server_name": "math", "message": message})
        out = json.dumps([
            mcp({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18"}}),
            mcp({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            mcp({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            mcp({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "add", "arguments": {"a": 2, "b": 3}}}),
        ])
    elif text == "slow":
        while True:
            message = read()
            if message.get("type") == "control_request" and message["request"]["subtype"] == "interrupt":
                reply(message["request_id"], {})
                break
            handle(message)
        result(None, "error_during_execution", True)
        return
    elif text == "exit":
        sys.stderr.write("boom\n")
        sys.stderr.flush()
        sys.exit(3)
    else:
        out = "echo:%s:model=%s" % (text, state["model"])
    send({"type": "assistant", "parent_tool_use_id": None, "session_id": SESSION,
          "message": {"role": "assistant", "model": state["model"] or "default",
                      "content": [{"type": "text", "text": out}]}})
    result(out)


def handle(message):
    if message.get("type") == "control_request":
        request = message["request"]
        subtype = request["subtype"]
        if subtype == "initialize":
            reply(message["request_id"], {"commands": [], "argv": ARGS, "hooks": request.get("hooks")})
        elif subtype == "set_model":
            state["model"] = request.get("model")
            reply(message["request_id"], {})
        elif subtype in ("set_permission_mode", "interrupt"):
            reply(message["request_id"], {})
        else:
            send({"type": "control_response",
                  "response": {"subtype": "error", "request_id": message["request_id"],
                               "error": "unsupported " + subtype}})
    elif message.get("type") == "user":
        content = message["message"]["content"]
        turn(content if isinstance(content, str) else json.dumps(content))


while True:
    handle(read())
