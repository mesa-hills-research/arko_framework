"""A scripted stand-in for the `claude` CLI, for the Claude Code agent runtime.

It speaks the stream-json control protocol like the real CLI: it calls the
host's PreToolUse and PostToolUse hooks for every tool call, asks the host's
permission handler when a hook answers "ask", and calls the host's
in-process MCP server ("arko") for `mcp__arko__*` tools.

FAKE_CLAUDE_SCRIPT names a JSON file: a list of turns, each a list of steps,
played one turn per user message across processes (FAKE_CLAUDE_STATE keeps
the next turn's index). Steps:

  {"text": "...", "input_tokens": 500}    an assistant text message
  {"thinking": "..."}                    an assistant thinking block
  a text or thinking step with "stream": true streams it first, a few
                                         words per content_block_delta (with
                                         --include-partial-messages), as
                                         Claude Code does
  {"retry": {"attempt": 1, ...}}         an api_retry system message with
                                         these fields
  {"tool": "mcp__arko__probe", "input": {...}, "input_tokens": 500}
                                         a tool call
  {"exit": 3}                            the CLI crashes
  a tool step with "crash": "pre"        the CLI crashes once the call is
                                         allowed, before it runs; "post":
                                         after its PostToolUse hook, before
                                         its result is reported
  {"rate_limit": {...}}                  a rate_limit_event with this info
  {"fail": "..."}                        the turn ends in this error, with
                                         "terminal_reason" (such as
                                         "prompt_too_long") when given
  {"api_error": "authentication_failed", "text": "...", "status": 401}
                                         a request fails as Claude Code
                                         2.1 reports it (not signed in:
                                         status null): its own message
                                         with the error's kind, then an
                                         error result ("terminal_reason":
                                         "api_error" unless given)
  {"system": {"subtype": ...}}           a system message
  {"refused": {"category": ...}}         a response the API refused: its
                                         message_delta stops with
                                         "refusal" and these stop_details

Text and tool steps take "model", the model that served them. A built-in
tool step takes "output", its result (otherwise "builtin <tool> ran").

Like Claude Code, the fake keeps each session's conversation (the user
messages, under the "uuid" the host gave them, and its own messages, each
with a "uuid"), in FAKE_CLAUDE_STATE plus ".sessions" across processes.
--resume-session-at=<uuid> cuts it back to that message as the session
resumes, and fails as Claude Code does when no message has the UUID.
--fork-session with --session-id=<new> starts the new session as a copy of
the one --resume names (cut back as --resume-session-at says), which stays
as it was.

A "/compact" message plays the next entry of the script when it is
{"compact": {"pre_tokens": 1500, "post_tokens": 300}}: the compaction as
Claude Code 2.1.300 reports it (status messages, a new init, the
compact_boundary, the summary and the command's output as user messages,
and a result with no turns). {"compact": {"error": "..."}} reports a
compaction that failed, without a compact_boundary (the shape is assumed,
not taken from a real failure). Otherwise "/compact" is an ordinary message
that plays a turn, as a CLI without the command would pass it to the model.

A turn's result is its last text. FAKE_CLAUDE_LOG receives one JSON line per
event: argv, tools, user messages (each with the conversation before it, as
"history"), hook and permission answers, tool results.
"""
import json
import os
import sys
import uuid

ARGS = sys.argv[1:]
state = {"counter": 0, "hooks": {}, "interrupted": False, "message": 0}


def flag(*names):
    for arg in ARGS:
        for name in names:
            if arg.startswith(name):
                return arg[len(name):]
    return None


FORK = "--fork-session" in ARGS
PARENT = flag("--resume=")
# A fork takes the ID --session-id gives it; a resume goes on in the
# conversation it names.
SESSION = (flag("--session-id=") if FORK else None) or flag("--session-id=", "--resume=") \
    or "fake-session"
SESSIONS = os.environ.get("FAKE_CLAUDE_STATE", "state") + ".sessions"


def sessions():
    return json.load(open(SESSIONS)) if os.path.exists(SESSIONS) else {}


def conversation():
    return sessions().get(SESSION, [])


def keep(conversation):
    saved = sessions()
    saved[SESSION] = conversation
    with open(SESSIONS, "w") as out:
        json.dump(saved, out)


def record(text, message_uuid=None):
    """Adds a message to the session's conversation and returns its UUID."""
    message_uuid = message_uuid or str(uuid.uuid4())
    keep(conversation() + [{"uuid": message_uuid, "text": text}])
    return message_uuid


if FORK and PARENT:
    # The fork starts as a copy of the parent's conversation, which stays.
    keep(sessions().get(PARENT, []))

RESUME_AT = flag("--resume-session-at=")
if RESUME_AT:
    entries = conversation()
    found = [index for index, entry in enumerate(entries) if entry["uuid"] == RESUME_AT]
    if not found:
        sys.stderr.write("No message found with message.uuid of: %s\n" % RESUME_AT)
        sys.exit(1)
    keep(entries[:found[0] + 1])


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


def next_turn(advance=True):
    script = json.load(open(os.environ["FAKE_CLAUDE_SCRIPT"]))
    state_path = os.environ["FAKE_CLAUDE_STATE"]
    index = int(open(state_path).read()) if os.path.exists(state_path) else 0
    if advance:
        with open(state_path, "w") as out:
            out.write(str(index + 1))
    return script[index] if index < len(script) else [{"text": "script exhausted"}]


def compact(spec):
    send({"type": "system", "subtype": "status", "status": "compacting", "session_id": SESSION})
    if "error" in spec:
        send({"type": "system", "subtype": "status", "status": None, "compact_result": "failed",
              "session_id": SESSION})
        send({"type": "user", "parent_tool_use_id": None, "session_id": SESSION, "isReplay": True,
              "message": {"role": "user", "content":
                          "<local-command-stderr>Error: %s</local-command-stderr>" % spec["error"]}})
    else:
        send({"type": "system", "subtype": "status", "status": None, "compact_result": "success",
              "session_id": SESSION})
        send({"type": "system", "subtype": "init", "session_id": SESSION})
        send({"type": "system", "subtype": "compact_boundary", "session_id": SESSION,
              "compact_metadata": {"trigger": "manual", "pre_tokens": spec.get("pre_tokens", 0),
                                   "post_tokens": spec.get("post_tokens", 0),
                                   "cumulative_dropped_tokens": 0, "duration_ms": 5}})
        send({"type": "user", "parent_tool_use_id": None, "session_id": SESSION,
              "isSynthetic": True, "message": {"role": "user", "content":
                  "This session is being continued from a previous conversation that ran out of "
                  "context. The summary below covers the earlier portion of the conversation.\n\n"
                  "Summary:\n" + spec.get("summary", "The work so far.")}})
        send({"type": "user", "parent_tool_use_id": None, "session_id": SESSION, "isReplay": True,
              "message": {"role": "user",
                          "content": "<local-command-stdout>Compacted </local-command-stdout>"}})
    send({"type": "result", "subtype": "success", "is_error": False, "num_turns": 0,
          "session_id": SESSION, "duration_ms": 5, "result": "", "local_command": "compact"})


def usage(step, output):
    # Anthropic's input counts are exclusive: these add up to input_tokens.
    tokens = step.get("input_tokens", 10)
    cached = step.get("cached", tokens - 2)
    written_1h = step.get("written_1h", 0)
    return {"input_tokens": tokens - cached - written_1h, "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": written_1h, "output_tokens": output,
            "cache_creation": {"ephemeral_5m_input_tokens": 0,
                               "ephemeral_1h_input_tokens": written_1h}}


def stream_event(event):
    if "--include-partial-messages" in ARGS:
        send({"type": "stream_event", "event": event, "parent_tool_use_id": None,
              "session_id": SESSION})


def delta(step):
    stop = {"stop_reason": "end_turn"}
    if "refused" in step:
        stop = {"stop_reason": "refusal", "stop_details": dict(type="refusal", **step["refused"])}
    stream_event({"type": "message_delta", "delta": stop,
                  "usage": usage(step, step.get("output_tokens", 5))})


def assistant(content, step):
    # "continues": part of the previous message's response (the same id),
    # as Claude Code sends a response's text and its call. Like Claude
    # Code, the assistant messages carry the output count as it stood when
    # the stream started; the request's message_delta has the final one.
    if not step.get("continues"):
        state["message"] += 1
        stream_event({"type": "message_start", "message": {
            "id": "msg_%d" % state["message"], "model": step.get("model", "claude-opus-5-5"),
            "usage": usage(step, 1)}})
    text = " ".join(block["text"] if block["type"] == "text"
                    else block["thinking"] if block["type"] == "thinking"
                    else "%s %s" % (block["name"], json.dumps(block["input"]))
                    for block in content)
    if step.get("stream"):
        for index, block in enumerate(content):
            kind = block["type"]
            if kind not in ("text", "thinking"):
                continue
            words = block[kind].split(" ")
            pieces = [" ".join(words[i:i + 2]) for i in range(0, len(words), 2)]
            for number, piece in enumerate(pieces):
                piece = piece if number == len(pieces) - 1 else piece + " "
                stream_event({"type": "content_block_delta", "index": index,
                              "delta": {"type": kind + "_delta", kind: piece}})
    send({"type": "assistant", "parent_tool_use_id": None, "session_id": SESSION,
          "uuid": record(text),
          "message": {"id": "msg_%d" % state["message"], "role": "assistant",
                      "model": step.get("model", "claude-opus-5-5"), "content": content,
                      "usage": usage(step, 1)}})
    if step.get("closes", True):
        delta(step)


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
    if output is None and step.get("crash") == "pre":
        sys.exit(3)
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
            output = step.get("output", "builtin %s ran" % name)
    post = hook("PostToolUse", {"tool_name": name, "tool_input": tool_input,
                                "tool_response": output}, tool_use_id)
    log("post", tool=name, answer=post, output=output, is_error=is_error)
    if step.get("crash") == "post":
        sys.exit(3)
    send({"type": "user", "parent_tool_use_id": None, "session_id": SESSION,
          "uuid": record(output),
          "message": {"role": "user", "content": [
              {"type": "tool_result", "tool_use_id": tool_use_id, "content": output,
               "is_error": is_error}]}})


def turn(text, message_uuid=None):
    log("user", text=text, history=[entry["text"] for entry in conversation()])
    record(text, message_uuid)
    if not state.get("listed"):
        state["listed"] = True
        listed = ask({"subtype": "mcp_message", "server_name": "arko",
                      "message": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}})
        log("tools", tools=listed["mcp_response"]["result"]["tools"])
    if text.startswith("/compact"):
        entry = next_turn(advance=False)
        if isinstance(entry, dict) and "compact" in entry:
            next_turn()
            compact(entry["compact"])
            return
    send({"type": "system", "subtype": "init", "session_id": SESSION})
    state["interrupted"] = False
    final = ""
    steps = next_turn()
    for index, step in enumerate(steps):
        # A step followed by one that continues its response doesn't close it.
        following = steps[index + 1] if index + 1 < len(steps) else {}
        step = dict(step, closes=not following.get("continues"))
        if state["interrupted"]:
            send({"type": "result", "subtype": "error_during_execution", "is_error": True,
                  "num_turns": 1, "session_id": SESSION, "duration_ms": 5, "result": None})
            return
        if "exit" in step:
            sys.exit(step["exit"])
        if "rate_limit" in step:
            send({"type": "rate_limit_event", "rate_limit_info": step["rate_limit"],
                  "session_id": SESSION})
            continue
        if "system" in step:
            send(dict({"type": "system", "session_id": SESSION}, **step["system"]))
            continue
        if "retry" in step:
            send(dict({"type": "system", "subtype": "api_retry", "session_id": SESSION},
                      **step["retry"]))
            continue
        if "refused" in step:
            # Claude Code keeps no content from a refused response.
            state["message"] += 1
            stream_event({"type": "message_start", "message": {
                "id": "msg_%d" % state["message"], "model": step.get("model", "claude-opus-5-5"),
                "usage": usage(step, 1)}})
            delta(step)
            continue
        if "api_error" in step:
            text = step.get("text", "API Error")
            send({"type": "assistant", "parent_tool_use_id": None, "session_id": SESSION,
                  "uuid": record(text),
                  "error": step["api_error"], "is_api_error_message": True,
                  "message": {"id": "synthetic_%d" % index, "role": "assistant",
                              "model": "<synthetic>", "stop_reason": "stop_sequence",
                              "content": [{"type": "text", "text": text}],
                              "usage": {"input_tokens": 0, "output_tokens": 0,
                                        "cache_read_input_tokens": 0,
                                        "cache_creation_input_tokens": 0}}})
            send({"type": "result", "subtype": "success", "is_error": True, "num_turns": 1,
                  "session_id": SESSION, "duration_ms": 5, "result": text,
                  "api_error_status": step.get("status"),
                  "terminal_reason": step.get("terminal_reason", "api_error")})
            return
        if "fail" in step:
            result = {"type": "result", "subtype": "error_during_execution", "is_error": True,
                      "num_turns": 1, "session_id": SESSION, "duration_ms": 5,
                      "result": step["fail"]}
            if "terminal_reason" in step:
                result["terminal_reason"] = step["terminal_reason"]
            send(result)
            return
        if "thinking" in step:
            # A thinking block, as Opus on Claude Code sends one before its
            # text and calls.
            assistant([{"type": "thinking", "thinking": step["thinking"], "signature": "sig"}],
                      step)
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
            log("argv", argv=ARGS, env={name: os.environ.get(name) for name in (
                "MCP_TOOL_TIMEOUT", "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT",
                "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR", "DISABLE_AUTOUPDATER")})
        elif subtype == "set_model":
            log("set_model", model=request.get("model"))
            reply(message["request_id"], {})
        elif subtype == "interrupt":
            state["interrupted"] = True
            reply(message["request_id"], {})
        else:
            reply(message["request_id"], {})
    elif message.get("type") == "user":
        content = message["message"]["content"]
        turn(content if isinstance(content, str) else json.dumps(content), message.get("uuid"))


while True:
    handle(read())
