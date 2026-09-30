"""A scripted stand-in for the `claude` CLI, for the Claude Code agent runtime.

It speaks the stream-json control protocol like the real CLI: it calls the
host's PreToolUse and PostToolUse hooks for every tool call, asks the host's
permission handler when a hook answers "ask", and calls the host's
in-process MCP server ("arko") for `mcp__arko__*` tools.

FAKE_CLAUDE_SCRIPT names a JSON file: a list of turns, each a list of steps,
played one turn per user message across processes (FAKE_CLAUDE_STATE keeps
the next turn's index). Steps:

  {"text": "...", "input_tokens": 500}    an assistant text message
  {"tool": "mcp__arko__probe", "input": {...}, "input_tokens": 500}
                                         a tool call
  {"exit": 3}                            the CLI crashes

A turn's result is its last text. FAKE_CLAUDE_LOG receives one JSON line per
event: argv, tools, user messages, hook and permission answers, tool results.
"""
import json
import os
import sys

ARGS = sys.argv[1:]
state = {"counter": 0, "hooks": {}, "interrupted": False, "message": 0}


def session_id():
    for arg in ARGS:
        for flag in ("--session-id=", "--resume="):
            if arg.startswith(flag):
                return arg[len(flag):]
    return "fake-session"


SESSION = session_id()


def log(event, **fields):
    path = os.environ.get("FAKE_CLAUDE_LOG")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps(dict(event=event, **fields)) + "\n")


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
    state["counter"] += 1
    request_id = "cli_%d" % state["counter"]
    send({"type": "control_request", "request_id": request_id, "request": request})
    while True:
        message = read()
        if message.get("type") == "control_response" and message["response"]["request_id"] == request_id:
            return message["response"].get("response")
        handle(message)


def next_turn():
    script = json.load(open(os.environ["FAKE_CLAUDE_SCRIPT"]))
    state_path = os.environ["FAKE_CLAUDE_STATE"]
    index = int(open(state_path).read()) if os.path.exists(state_path) else 0
    with open(state_path, "w") as out:
        out.write(str(index + 1))
    return script[index] if index < len(script) else [{"text": "script exhausted"}]


def usage(step):
    tokens = step.get("input_tokens", 10)
    return {"input_tokens": 2, "cache_read_input_tokens": tokens - 2,
            "cache_creation_input_tokens": 0, "output_tokens": 5}


def assistant(content, step):
    state["message"] += 1
    send({"type": "assistant", "parent_tool_use_id": None, "session_id": SESSION,
          "message": {"id": "msg_%d" % state["message"], "role": "assistant",
                      "model": "claude-opus-5-5", "content": content,
                      "usage": usage(step)}})


def hook(event, fields, tool_use_id):
    callback = state["hooks"].get(event)
    if not callback:
        return {}
    return ask({"subtype": "hook_callback", "callback_id": callback,
                "input": dict(hook_event_name=event, session_id=SESSION, **fields),
                "tool_use_id": tool_use_id}) or {}


def call_tool(step, index):
    name, tool_input = step["tool"], step.get("input", {})
    tool_use_id = step.get("id", "toolu_%d_%d" % (state["message"] + 1, index))
    assistant([{"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}], step)
    pre = hook("PreToolUse", {"tool_name": name, "tool_input": tool_input}, tool_use_id)
    log("pre", tool=name, answer=pre)
    decision = pre.get("hookSpecificOutput", {})
    verdict = decision.get("permissionDecision")
    output, is_error = None, False
    if verdict == "deny":
        output, is_error = decision.get("permissionDecisionReason", ""), True
    elif verdict == "ask":
        answer = ask({"subtype": "can_use_tool", "tool_name": name, "input": tool_input,
                      "tool_use_id": tool_use_id, "permission_suggestions": []})
        log("permission", tool=name, answer=answer)
        if answer.get("behavior") != "allow":
            output, is_error = answer.get("message", ""), True
    if output is None:
        if name.startswith("mcp__arko__"):
            response = ask({"subtype": "mcp_message", "server_name": "arko",
                            "message": {"jsonrpc": "2.0", "id": 100 + index, "method": "tools/call",
                                        "params": {"name": name[len("mcp__arko__"):],
                                                   "arguments": tool_input}}})
            result = response["mcp_response"].get("result", {})
            output = "".join(part.get("text", "") for part in result.get("content", []))
            is_error = bool(result.get("isError"))
        else:
            output = "builtin %s ran" % name
    post = hook("PostToolUse", {"tool_name": name, "tool_input": tool_input,
                                "tool_response": output}, tool_use_id)
    log("post", tool=name, answer=post, output=output, is_error=is_error)
    send({"type": "user", "parent_tool_use_id": None, "session_id": SESSION,
          "message": {"role": "user", "content": [
              {"type": "tool_result", "tool_use_id": tool_use_id, "content": output,
               "is_error": is_error}]}})


def turn(text):
    log("user", text=text)
    if not state.get("listed"):
        state["listed"] = True
        listed = ask({"subtype": "mcp_message", "server_name": "arko",
                      "message": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}})
        log("tools", tools=listed["mcp_response"]["result"]["tools"])
    send({"type": "system", "subtype": "init", "session_id": SESSION})
    state["interrupted"] = False
    final = ""
    for index, step in enumerate(next_turn()):
        if state["interrupted"]:
            send({"type": "result", "subtype": "error_during_execution", "is_error": True,
                  "num_turns": 1, "session_id": SESSION, "duration_ms": 5, "result": None})
            return
        if "exit" in step:
            sys.exit(step["exit"])
        if "text" in step:
            assistant([{"type": "text", "text": step["text"]}], step)
            final = step["text"]
        if "tool" in step:
            call_tool(step, index)
    send({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
          "session_id": SESSION, "duration_ms": 5, "result": final})


def handle(message):
    if message.get("type") == "control_request":
        request = message["request"]
        subtype = request["subtype"]
        if subtype == "initialize":
            for event, matchers in (request.get("hooks") or {}).items():
                state["hooks"][event] = matchers[0]["hookCallbackIds"][0]
            reply(message["request_id"], {"commands": []})
            log("argv", argv=ARGS)
        elif subtype == "interrupt":
            state["interrupted"] = True
            reply(message["request_id"], {})
        else:
            reply(message["request_id"], {})
    elif message.get("type") == "user":
        content = message["message"]["content"]
        turn(content if isinstance(content, str) else json.dumps(content))


while True:
    handle(read())
