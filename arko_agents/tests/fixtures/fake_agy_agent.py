#!/usr/bin/env python3
"""A scripted stand-in for the Antigravity CLI (`agy`), for the agy agent runtime.

It answers `agy --version`, `agy -p /usage --output-format json` and
`agy acp`. In `acp` mode it speaks the Agent Client Protocol (JSON-RPC 2.0,
one object per line) on stdio as agy 1.5.14 does, checks the profile it's
given, starts the MCP server that profile names and plays the model's turns
from a script.

Launch

    agy acp --gemini-dir DIR --agent NAME --log-file PATH
            --include-default-customizations=false --disable-telemetry

with AGY_CLI_DISABLE_AUTO_UPDATE=true in the environment. `--stdio` may be
added. An unknown flag, a missing required one or another value of that
variable makes it exit with code 2 and the reason on stderr. `--agent` and
`--log-file` may be left out, as for a sign-in check, and then it runs as
agy's default agent. It writes a line to the log file as it starts (creating
it, or appending to one that's there) and one for each new session.

`agy -p /usage --output-format json` (with `--gemini-dir DIR` and
`--disable-telemetry` allowed) prints FAKE_AGY_USAGE's file, the script's
"usage", or a real 1.5.14 answer. It needs the same environment variable.
`agy --version` prints the version and a newline.

Environment

    FAKE_AGY_SCRIPT   a JSON file with the script below, or the JSON itself
    FAKE_AGY_STATE    the file that records which turns were played, shared by
                      every process (default FAKE_AGY_SCRIPT plus ".state")
    FAKE_AGY_LOG      a file that gets one JSON line per event
    FAKE_AGY_USAGE    a JSON file for `-p /usage` to print
    FAKE_AGY_VERSION  the version it reports (default 1.5.14)
    FAKE_AGY_LOG_ENV  more environment variables, comma-separated, for the
                      argv event to record (it records AGY_* and ARKO_*, and
                      "<set>" for a name with TOKEN, KEY, SECRET or PASSWORD)

The MCP server inherits this environment, as agy's does.

Profile

As it starts, it reads DIR/config/mcp_config.json, starts the "arko" server's
command with its args, and lists its tools as agy does (server/discover, which
may fail, initialize, notifications/initialized, tools/list, then
notifications/roots/list_changed). The model sees `mcp_arko_<tool>` for each
tool, plus list_resources, read_resource and manage_task.

At the first session/new it reads DIR/config/agents/NAME.md. A missing file, a
file without frontmatter, frontmatter it can't parse or a `name` other than
NAME makes agy fall back to its default agent. The line
`Agent "NAME" not found, falling back to default` goes to the log file while
session/new is handled, and DIR/config/hooks.json loads. Any other departure
from the setup below answers session/new with the error
"fake agy: profile: ..." and logs a "profile" event with the problems.

  - The frontmatter has exactly name (NAME), description (not empty),
    excludeDefaultComponents: true, inheritCustomizations: false,
    mainAgent: true and subagent: false, and the body isn't empty.
  - mcp_config.json has only server "arko" with command, optional args,
    timeoutSeconds and tools. The timeout is 604800 seconds. "tools" names
    every tool the server lists, each {"eager": true}, and no other.

Script

FAKE_AGY_SCRIPT holds a list of turns, or an object with "turns" and any of
these keys.

    "fallback": true       act as though the agent file were broken, with the
                           log line ("silent" leaves the line out)
    "signed_out": true     session/new fails with -32000 "Authentication
                           required"
    "version": "1.5.14"    the version, unless FAKE_AGY_VERSION is set
    "version_output": "…"  the whole text `--version` prints
    "tier": "Google AI Pro" the plan session/new reports (null for none)
    "context_size": 1048576 the size every usage_update reports
    "models": [{"value": "gemini-4-neon", "name": "Gemini 4 Neon",
                "levels": ["low", "medium", "high", "xhigh"]}, ...]
    "usage": {...}         what `-p /usage` prints
    "usage_fail": {"exit": 1, "stderr": "..."}  how `-p /usage` fails

A turn is a list of steps, or {"match": "text", "steps": [...]}. Each prompt
that starts a turn plays the first turn not played yet (by any process) whose
"match", when it has one, is in the prompt's text. Without a script every
turn replies "fake agy reply", and past the script's end "script exhausted".

Steps

    {"text": "Hello"}             an agent_message_chunk ({"text": [..]} sends
                                  each piece as a chunk)
    {"thought": "..."}            an agent_thought_chunk (a list works too)
    {"tool": "read_file", "input": {...}}
                                  the model calls mcp_arko_read_file. agy adds
                                  "toolSummary" ("summary" sets it) to the
                                  arguments the client sees, asks the client's
                                  permission, then calls the MCP tool with the
                                  arguments as given. "id" sets the call's ID
                                  and "permission": false skips the request.
                                  Rejecting the permission ends the turn.
    {"parallel": [tool steps]}    calls asked for one by one, then run side by
                                  side (real agy wasn't seen doing this)
    {"builtin": "manage_task", "input": {...}}
                                  one of agy's own tools. It runs without a
                                  request unless "permission": true, and
                                  "kind", "title" and "output" set its fields.
                                  As the default agent, the PreToolUse hooks
                                  run first, and a deny fails the call.
    {"sleep": 0.5}                waits, unless the turn is cancelled
    {"wait_for": "note"}          waits for a note that no earlier wait_for
                                  took, so one sent before the step counts
                                  ("cancel" waits for a cancel), "timeout"
                                  seconds at most (30)
    {"wait_file": "PATH"}         waits until the file exists ("timeout")
    {"fail": "failed to connect"} the turn fails as agy reports it, with one
                                  chunk "\\n⚠ failed to connect\\n" and a result
                                  with no usage. {"fail": {"status":
                                  "INVALID_ARGUMENT", "code": 400, "message":
                                  "..."}} sends "\\n⚠ INVALID_ARGUMENT (code
                                  400): ...\\n". Every later prompt in the
                                  session fails the same way.
    {"end": "refusal"}            the turn ends with this stopReason
    {"error": {"code": 429, "message": "..."}}
                                  session/prompt answers with this JSON-RPC
                                  error (not seen from real agy)
    {"exit": 3}                   the process exits with this code
    {"update": {...}}             a session/update with this update
    {"request": "fs/read_text_file", "params": {...}}
                                  a request to the client, its answer logged
    {"raw": "text"}               this line as it is

Any step may carry "usage", the model call that produced it, in the names of
the prompt's result ({"inputTokens": 900, "cachedReadTokens": 100,
"outputTokens": 20, "thoughtTokens": 5, "cachedWriteTokens": 0}, with
"totalTokens" the sum of all but thoughtTokens unless given). It's sent as a
usage_update first. A result carries the usage counted since the previous
result. A turn that counts none ends with 100 input and 10 output tokens.

Turns

The first prompt of a session gets config_option_update,
available_commands_update, usage_update and session_info_update first. A
prompt sent while a turn runs joins it, as in agy. The running prompt's result
comes back at once with stopReason "end_turn" and the usage so far, and the
new prompt's result ends the turn. Its text is logged as a "note", and as
"note_taken" before the next step. A session/cancel ends the turn with
stopReason "cancelled" and its usage. A running MCP call gets
notifications/cancelled and its tool_call_update reports "failed".

Log events (FAKE_AGY_LOG, each line has "event" and "pid")

    argv, version, usage_command    a launch and its arguments
    initialize, session_new, config, prompt   the client's requests
    mcp_started, mcp_discover, mcp_initialize, tools, mcp_message, mcp_exited
    profile           the agent file, system_prompt, mcp_config, tools,
                      visible, default_agent, fallback, hooks and problems
    turn              which script turn a prompt plays
    note, note_taken, cancel, permission, tool_call, tool_result,
    tool_failed, hook, builtin, client_response, turn_end, wait_timeout

`fake_agy_agent.py --self-check` plays each kind of scenario against the fake,
with a small MCP server of its own (`--self-check-mcp`).
"""

import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DEFAULT_VERSION = "1.5.14"
CONTEXT_SIZE = 1048576
AGENT_FLAGS = {
    "excludeDefaultComponents": True,
    "inheritCustomizations": False,
    "mainAgent": True,
    "subagent": False,
}
AGENT_KEYS = {"name", "description"} | set(AGENT_FLAGS)
# The tools agy keeps with a custom agent that excludes its default components.
KEPT_BUILTINS = ["list_resources", "read_resource", "manage_task"]
# A few of the tools agy 1.5.14's default agent offers.
DEFAULT_BUILTINS = ["run_command", "read_file", "search_web", "call_mcp_tool"]

MODELS = [
    {"value": "gemini-4-neon", "name": "Gemini 4 Neon",
     "levels": ["low", "medium", "high", "xhigh"]},
    {"value": "gemini-4-argon", "name": "Gemini 4 Argon", "levels": ["low", "medium", "high"]},
    {"value": "gemini-4-hydrogen", "name": "Gemini 4 Hydrogen",
     "levels": ["low", "medium", "high", "xhigh"]},
    {"value": "gemini-pro-orchestrator-neon", "name": "Orchestrator (Neon)", "levels": []},
    {"value": "gemini-pro-orchestrator-g1-pro", "name": "Orchestrator (Argon)", "levels": []},
    {"value": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "levels": ["low", "medium", "high"]},
    {"value": "claude-opus-5-5", "name": "Claude Opus 5.5", "levels": ["low", "medium", "high"]},
    {"value": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5",
     "levels": ["low", "medium", "high"]},
    {"value": "gpt-oss-120b-medium", "name": "GPT-OSS 120B (Medium)", "levels": []},
]
LEVEL_NAMES = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra"}
MODEL_LEVEL_NAMES = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "XHigh"}
MODES = [
    {"id": "default", "name": "Default",
     "description": "Ask before edits and commands, per your permission settings."},
    {"id": "accept-edits", "name": "Accept edits",
     "description": "Approve file edits in the workspace without asking."},
    {"id": "plan", "name": "Plan", "description": "Plan carefully before executing a task."},
]
COMMANDS = [
    {"name": "plan", "description": "Plan carefully before executing a task."},
    {"name": "grill-me", "description": "Interview me to align on a plan."},
    {"name": "learn",
     "description": "Reflect on recent successes or corrections to capture reusable skills or "
                    "rules."},
]
AUTH_METHODS = [
    {"id": "oauth-personal", "name": "Log in with Google",
     "description": "Log in with your Google account"},
    {"id": "oauth-business", "name": "Log in with Gemini Enterprise",
     "description": "Log in with your Gemini Enterprise account"},
    {"id": "gemini-api-key", "name": "Gemini API key",
     "description": "Use an API key with Gemini Developer API"},
    {"id": "agent-platform", "name": "Gemini Enterprise Agent Platform",
     "description": "Use Gemini Enterprise Agent Platform (formerly Vertex AI) with Application "
                    "Default Credentials"},
]
PERMISSION_OPTIONS = [
    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
    {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
]
# What agy 1.5.14 printed for `agy -p /usage --output-format json`.
USAGE_SAMPLE = {
    "conversation_id": "", "status": "SUCCESS",
    "response": "Gemini Models\tWeekly Limit Remaining\t97%\t2026-11-28T19:38:35Z\n"
                "Gemini Models\tFive Hour Limit Remaining\t91%\t2026-11-22T00:38:35Z\n"
                "Claude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-28T21:21:32Z\n"
                "Claude and GPT models\tFive Hour Limit Remaining\t100%\t2026-11-22T02:21:32Z\n"
                "Gemini 4 Argon\tGemini 4 Argon Weekly Limit\t100%\t2026-11-28T21:21:32Z\n",
    "duration_seconds": 0, "num_turns": 0,
    "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
              "cache_read_tokens": 0, "total_tokens": 0},
    "command": {"name": "usage", "data": {
        "description": "Within each group, models share a weekly limit and a 5-hour limit. Quota "
                       "is consumed proportionally to the cost of the tokens. Thus, limits will "
                       "last longer with shorter tasks or using more cost-effective models. The "
                       "5-hour limit smooths out aggregate demand to fairly distribute global "
                       "capacity across all users, while your weekly limit is tied directly to "
                       "your individual tier.",
        "groups": [
            {"name": "Gemini Models",
             "description": "Models within this group: Gemini 4 Neon, Gemini 4 Hydrogen, "
                            "Orchestrator (Neon), Gemini 3.8 Flash",
             "buckets": [
                 {"id": "gemini-weekly", "name": "Weekly Limit Remaining",
                  "description": "You have used some of your weekly limit, it will fully "
                                 "refresh in 6 days, 22 hours.",
                  "window": "weekly", "remaining_fraction": 0.9698070883750916,
                  "reset_time": "2026-11-28T19:38:35Z"},
                 {"id": "gemini-5h", "name": "Five Hour Limit Remaining",
                  "description": "You have used some of your 5-hour limit, it will fully "
                                 "refresh in 3 hours, 17 minutes.",
                  "window": "5h", "remaining_fraction": 0.9143246412277222,
                  "reset_time": "2026-11-22T00:38:35Z"}]},
            {"name": "Claude and GPT models",
             "description": "Models within this group: Claude Opus 5.5, Claude Sonnet 5.5, "
                            "GPT-OSS 120B",
             "buckets": [
                 {"id": "3p-weekly", "name": "Weekly Limit Remaining", "window": "weekly",
                  "remaining_fraction": 1, "reset_time": "2026-11-28T21:21:32Z"},
                 {"id": "3p-5h", "name": "Five Hour Limit Remaining", "window": "5h",
                  "remaining_fraction": 1, "reset_time": "2026-11-22T02:21:32Z"}]},
            {"name": "Gemini 4 Argon",
             "description": "For paid Pro plans, Argon quota is separate. Higher Argon quota "
                            "(included as part of the larger Gemini quota bucket) is available "
                            "on Ultra plans.",
             "buckets": [
                 {"id": "gemini-argon-wtu-weekly", "name": "Gemini 4 Argon Weekly Limit",
                  "window": "weekly", "remaining_fraction": 1,
                  "reset_time": "2026-11-28T21:21:32Z"}]}]}}}

ACP_FLAGS = {
    "--gemini-dir": "value", "--agent": "value", "--log-file": "value",
    "--include-default-customizations": "bool", "--disable-telemetry": "bool", "--stdio": "bool",
}
USAGE_FLAGS = {"-p": "value", "--output-format": "value", "--gemini-dir": "value",
               "--disable-telemetry": "bool"}


def log(event, **fields):
    path = os.environ.get("FAKE_AGY_LOG")
    if not path:
        return
    line = json.dumps(dict(event=event, pid=os.getpid(), **fields), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as out:
        out.write(line + "\n")


def environment():
    """The environment variables the argv event records. A token, key, secret
    or password (such as ARKO_AGY_RELAY_TOKEN) is recorded as "<set>"."""
    extra = [name for name in os.environ.get("FAKE_AGY_LOG_ENV", "").split(",") if name]
    picked = {}
    for name, value in os.environ.items():
        if name.startswith(("AGY_", "ARKO_")) or name in extra:
            secret = re.search("TOKEN|KEY|SECRET|PASSWORD", name.upper())
            picked[name] = "<set>" if secret else value
    for name in extra:
        picked.setdefault(name, None)
    return picked


def load_script():
    raw = os.environ.get("FAKE_AGY_SCRIPT", "")
    if not raw:
        return {}, os.environ.get("FAKE_AGY_STATE")
    inline = raw.lstrip()[:1] in ("[", "{")
    script = json.loads(raw if inline else Path(raw).read_text(encoding="utf-8"))
    if isinstance(script, list):
        script = {"turns": script}
    state = os.environ.get("FAKE_AGY_STATE") or (None if inline else raw + ".state")
    return script, state


def version(script):
    return os.environ.get("FAKE_AGY_VERSION") or script.get("version") or DEFAULT_VERSION


def parse_flags(args, spec):
    """Flags as Go's flag package reads them: `--name value`, `--name=value`,
    and booleans as `--name` or `--name=false`."""
    values, problems = {}, []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        name, equals, value = arg.partition("=")
        if name not in spec:
            problems.append("unknown argument %s" % arg)
            continue
        if spec[name] == "value" and not equals:
            if index >= len(args):
                problems.append("%s needs a value" % name)
                continue
            value = args[index]
            index += 1
        elif spec[name] == "bool":
            value = value if equals else "true"
            if value not in ("true", "false"):
                problems.append("%s takes true or false, not %s" % (name, value))
        if name in values:
            problems.append("%s is given twice" % name)
        values[name] = value
    if os.environ.get("AGY_CLI_DISABLE_AUTO_UPDATE") != "true":
        problems.append("AGY_CLI_DISABLE_AUTO_UPDATE is %r, not 'true'"
                        % os.environ.get("AGY_CLI_DISABLE_AUTO_UPDATE"))
    return values, problems


def refuse(problems):
    sys.stderr.write("fake agy: %s\n" % "; ".join(problems))
    sys.stderr.flush()
    sys.exit(2)


def usage_command(args, script):
    values, problems = parse_flags(args, USAGE_FLAGS)
    if values.get("-p") != "/usage":
        problems.append("-p is %r, not '/usage'" % values.get("-p"))
    if values.get("--output-format") != "json":
        problems.append("--output-format is %r, not 'json'" % values.get("--output-format"))
    log("usage_command", argv=args, cwd=os.getcwd(), env=environment(), problems=problems)
    if problems:
        refuse(problems)
    failure = script.get("usage_fail")
    if failure:
        sys.stderr.write(failure.get("stderr", ""))
        sys.exit(failure.get("exit", 1))
    if os.environ.get("FAKE_AGY_USAGE"):
        answer = json.loads(Path(os.environ["FAKE_AGY_USAGE"]).read_text(encoding="utf-8"))
    else:
        answer = script.get("usage", USAGE_SAMPLE)
    sys.stdout.write(json.dumps(answer) + "\n")
    sys.stdout.flush()


class Turns:
    """The script's turns, and which ones any process has played."""

    def __init__(self, script, state_path):
        self.turns = script.get("turns")
        self.state_path = state_path
        self.played = []

    @contextmanager
    def locked(self):
        if not self.state_path:
            yield
            return
        lock = self.state_path + ".lock"
        deadline = time.time() + 10
        handle = None
        while handle is None:
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() > deadline:
                    break
                time.sleep(0.01)
        try:
            yield
        finally:
            if handle is not None:
                os.close(handle)
                os.remove(lock)

    def pick(self, text):
        """The index and steps of the turn this prompt plays."""
        if self.turns is None:
            return None, [{"text": "fake agy reply"}]
        with self.locked():
            played = self.played
            if self.state_path and os.path.exists(self.state_path):
                played = json.loads(Path(self.state_path).read_text() or "[]")
            for index, turn in enumerate(self.turns):
                if index in played:
                    continue
                match = turn.get("match") if isinstance(turn, dict) else None
                if match is None or match in text:
                    played = played + [index]
                    self.played = played
                    if self.state_path:
                        Path(self.state_path).write_text(json.dumps(played))
                    return index, turn["steps"] if isinstance(turn, dict) else turn
        return None, [{"text": "script exhausted"}]


def parse_scalar(value):
    """A YAML scalar as agy's frontmatter parser would read it, or raises
    ValueError for one that YAML rejects or reads differently."""
    value = value.strip()
    if value in ("true", "false"):
        return value == "true"
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise ValueError("unclosed quote")
        return value[1:-1].replace("''", "'")
    if value[:1] in tuple("[]{}&*!|>%@`#,?:-") or ": " in value or " #" in value \
            or value.endswith(":"):
        raise ValueError("a plain value YAML reads differently: %r" % value)
    return value


def read_agent(path, name):
    """The agent file's frontmatter and body, or the reason agy wouldn't load it."""
    if not path.is_file():
        return None, "%s is missing" % path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    if lines[0] != "---":
        return None, "%s has no frontmatter" % path
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None, "%s has no end to its frontmatter" % path
    frontmatter = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        found = re.match(r"([A-Za-z_][A-Za-z0-9_]*):(?:\s(.*))?$", line)
        if not found:
            return None, "can't parse the frontmatter line %r" % line
        key = found.group(1)
        if key in frontmatter:
            return None, "%s is given twice in the frontmatter" % key
        try:
            frontmatter[key] = parse_scalar(found.group(2) or "")
        except ValueError as error:
            return None, "can't parse %s in the frontmatter (%s)" % (key, error)
    if frontmatter.get("name") != name:
        return None, "the agent file's name is %r, not %r" % (frontmatter.get("name"), name)
    return {"frontmatter": frontmatter, "body": "\n".join(lines[end + 1:])}, None


def agent_problems(agent):
    problems = []
    frontmatter = agent["frontmatter"]
    for key in sorted(set(frontmatter) - AGENT_KEYS):
        problems.append("the agent file has an unexpected key %s" % key)
    for key in sorted(AGENT_KEYS - set(frontmatter)):
        problems.append("the agent file has no %s" % key)
    description = frontmatter.get("description")
    if "description" in frontmatter and not (isinstance(description, str) and description):
        problems.append("the agent file's description is empty")
    for key, wanted in AGENT_FLAGS.items():
        if key in frontmatter and frontmatter[key] is not wanted:
            problems.append("the agent file's %s is %r, not %s"
                            % (key, frontmatter[key], json.dumps(wanted)))
    if not agent["body"].strip():
        problems.append("the agent file has no system prompt")
    return problems


def mcp_config_problems(config):
    if not isinstance(config, dict) or set(config) != {"mcpServers"}:
        return ["mcp_config.json should have only mcpServers"]
    servers = config["mcpServers"]
    if not isinstance(servers, dict) or set(servers) != {"arko"}:
        return ["mcp_config.json should name only the server arko, not %s"
                % json.dumps(sorted(servers) if isinstance(servers, dict) else servers)]
    server = servers["arko"]
    problems = []
    if not isinstance(server, dict):
        return ["the arko server isn't an object"]
    for key in sorted(set(server) - {"command", "args", "tools", "timeoutSeconds"}):
        problems.append("the arko server has an unexpected key %s" % key)
    if server.get("timeoutSeconds") != 604800:
        problems.append("MCP timeoutSeconds must allow seven days")
    if not isinstance(server.get("command"), str) or not server.get("command"):
        problems.append("the arko server has no command")
    args = server.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        problems.append("the arko server's args aren't a list of strings")
    tools = server.get("tools")
    if not isinstance(tools, dict):
        problems.append("the arko server lists no tools")
    else:
        for name, entry in sorted(tools.items()):
            if entry != {"eager": True}:
                problems.append("tool %s is %s, not {\"eager\": true}" % (name, json.dumps(entry)))
    return problems


def read_hooks(path):
    """Hooks by event, from agy's hooks.json ({group: {event: [entries]}})."""
    if not path.is_file():
        return {}
    hooks = {}
    for group in json.loads(path.read_text(encoding="utf-8")).values():
        for event, entries in group.items():
            for entry in entries:
                matcher = entry.get("matcher", "*")
                for hook in entry.get("hooks", [entry]):
                    hooks.setdefault(event, []).append(dict(hook, matcher=matcher))
    return hooks


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def call_id():
    return "call_%d" % random.randrange(10 ** 17, 10 ** 19)


class Cancelled(Exception):
    """The client cancelled the turn."""


class TurnOver(Exception):
    """The turn's result was sent already."""


class Session:
    def __init__(self, session_id, cwd):
        self.id = session_id
        self.cwd = cwd
        self.model = "gemini-4-neon"
        self.level = "high"
        self.mode = "default"
        self.prompts = 0
        self.used = 0
        self.poisoned = None


class Turn:
    def __init__(self, session, prompt_id, text):
        self.session = session
        self.prompt_id = prompt_id
        self.text = text
        self.usage = {}
        self.counted = False
        self.cancelled = False
        self.notes = []
        self.note_count = 0
        self.notes_waited = 0
        self.step = 0


class Agy:
    def __init__(self, args, script, state_path):
        self.script = script
        self.version = version(script)
        self.context_size = script.get("context_size", CONTEXT_SIZE)
        self.turns = Turns(script, state_path)
        self.flags, problems = parse_flags(args, ACP_FLAGS)
        if "--gemini-dir" not in self.flags:
            problems.append("--gemini-dir is missing")
        if self.flags.get("--include-default-customizations") != "false":
            problems.append("--include-default-customizations=false is missing")
        if self.flags.get("--disable-telemetry") != "true":
            problems.append("--disable-telemetry is missing")
        log("argv", argv=["acp"] + args, cwd=os.getcwd(), env=environment(), problems=problems)
        if problems:
            refuse(problems)
        self.gemini_dir = Path(self.flags["--gemini-dir"])
        self.agent = self.flags.get("--agent")
        self.log_file = self.flags.get("--log-file")
        self.log_line("I fake agy %s acp started" % self.version)
        self.events = queue.Queue()
        self.backlog = []
        self.out = sys.stdout.buffer
        self.sessions = {}
        self.turn = None
        self.responses = {}
        self.requests = 0
        self.progress = str(uuid.uuid4())
        self.calls = 0
        self.mcp = None
        self.mcp_eof = True
        self.mcp_id = 0
        self.mcp_responses = {}
        self.abandoned = set()
        self.mcp_config = None
        self.offered = []
        self.setup_problems = []
        self.profile = None
        self.default_agent = False
        self.fallback = None
        self.hooks = {}
        threading.Thread(target=self.read_client, daemon=True).start()
        self.start_mcp()

    def log_line(self, text):
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as out:
                out.write("%s %s\n" % (stamp(), text))

    # Transport

    def read_client(self):
        for line in sys.stdin.buffer:
            if not line.strip():
                continue
            try:
                self.events.put(("client", json.loads(line)))
            except ValueError:
                self.events.put(("client_bad", line.decode("utf-8", "replace")))
        self.events.put(("client_eof", None))

    def read_mcp(self, stdout):
        for line in stdout:
            if not line.strip():
                continue
            try:
                self.events.put(("mcp", json.loads(line)))
            except ValueError:
                self.events.put(("mcp_bad", line.decode("utf-8", "replace")))
        self.events.put(("mcp_eof", None))

    def send(self, message):
        data = json.dumps(dict(jsonrpc="2.0", **message), ensure_ascii=False)
        self.out.write(data.encode("utf-8") + b"\n")
        self.out.flush()

    def reply(self, request_id, result):
        self.send({"id": request_id, "result": result})

    def error(self, request_id, code, message):
        self.send({"id": request_id, "error": {"code": code, "message": message}})

    def update(self, session, update):
        self.send({"method": "session/update",
                   "params": {"sessionId": session.id, "update": update}})

    def mcp_send(self, message):
        if self.mcp is None or self.mcp.poll() is not None:
            return
        try:
            self.mcp.stdin.write(json.dumps(dict(jsonrpc="2.0", **message)).encode() + b"\n")
            self.mcp.stdin.flush()
        except OSError:
            pass

    def pump(self, timeout=None, client=True):
        """Handles the next event, or returns False when none comes within
        `timeout` seconds. With `client` false, the client's messages wait."""
        if client and self.backlog:
            kind, message = self.backlog.pop(0)
        else:
            try:
                kind, message = self.events.get(timeout=timeout)
            except queue.Empty:
                return False
        if not client and kind.startswith("client"):
            self.backlog.append((kind, message))
            return True
        if kind == "client_eof":
            self.shutdown(0)
        elif kind == "client_bad":
            log("bad_frame", line=message[:2000])
            self.error(None, -32700, "Parse error")
        elif kind == "client":
            self.on_client(message)
        elif kind == "mcp":
            self.on_mcp(message)
        elif kind == "mcp_bad":
            log("mcp_bad_frame", line=message[:2000])
        elif kind == "mcp_eof":
            log("mcp_exited", code=self.mcp.wait() if self.mcp else None)
            self.mcp_eof = True
        return True

    def wait(self, ready, turn=None, timeout=None, client=True):
        """Pumps events until `ready()`. Raises Cancelled when the turn is
        cancelled meanwhile, and returns False when `timeout` runs out."""
        deadline = None if timeout is None else time.time() + timeout
        while not ready():
            if turn is not None and turn.cancelled:
                raise Cancelled()
            left = None if deadline is None else deadline - time.time()
            if left is not None and left <= 0:
                return False
            self.pump(left, client=client)
        return True

    def shutdown(self, code):
        if self.mcp is not None and self.mcp.poll() is None:
            try:
                self.mcp.stdin.close()
            except OSError:
                pass
            try:
                self.mcp.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.mcp.kill()
        log("exit", code=code)
        sys.stdout.flush()
        os._exit(code)

    def run(self):
        while True:
            self.pump()

    # MCP

    def start_mcp(self):
        path = self.gemini_dir / "config" / "mcp_config.json"
        if not path.is_file():
            if self.agent:
                self.setup_problems.append("%s is missing" % path)
            return
        try:
            self.mcp_config = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            self.setup_problems.append("can't parse mcp_config.json (%s)" % error)
            return
        problems = mcp_config_problems(self.mcp_config)
        if problems:
            self.setup_problems += problems
            return
        server = self.mcp_config["mcpServers"]["arko"]
        command = [server["command"]] + server.get("args", [])
        stderr = open(self.log_file, "ab") if self.log_file else None
        try:
            self.mcp = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr)
        except OSError as error:
            self.setup_problems.append("can't start the arko server %s (%s)" % (command, error))
            return
        self.mcp_eof = False
        log("mcp_started", command=command)
        threading.Thread(target=self.read_mcp, args=(self.mcp.stdout,), daemon=True).start()
        meta = {"io.modelcontextprotocol/clientCapabilities":
                {"elicitation": {"form": {}, "url": {}}, "roots": {"listChanged": True}},
                "io.modelcontextprotocol/clientInfo":
                {"name": "antigravity-client", "version": "v1.0.0"},
                "io.modelcontextprotocol/protocolVersion": "2026-07-28"}
        discover = self.mcp_request("server/discover", {"_meta": meta}, client=False)
        log("mcp_discover", answer=discover)
        initialized = self.mcp_request("initialize", {
            "clientInfo": {"name": "antigravity-client", "version": "v1.0.0"},
            "protocolVersion": "2025-11-25",
            "capabilities": {"elicitation": {"form": {}, "url": {}},
                             "roots": {"listChanged": True}}}, client=False)
        log("mcp_initialize", answer=initialized)
        if "result" not in initialized:
            self.setup_problems.append("the arko server didn't initialize: %s"
                                       % json.dumps(initialized))
            return
        self.mcp_send({"method": "notifications/initialized", "params": {}})
        listed = self.mcp_request("tools/list", {}, client=False)
        if "result" not in listed:
            self.setup_problems.append("the arko server didn't list its tools: %s"
                                       % json.dumps(listed))
            return
        self.offered = [tool["name"] for tool in listed["result"].get("tools", [])]
        log("tools", tools=listed["result"].get("tools", []))
        self.mcp_send({"method": "notifications/roots/list_changed", "params": {}})
        eager = server.get("tools", {})
        for name in self.offered:
            if name not in eager:
                self.setup_problems.append("tool %s isn't marked eager" % name)
        for name in eager:
            if name not in self.offered:
                self.setup_problems.append("the arko server doesn't offer the eager tool %s"
                                           % name)

    def mcp_request(self, method, params, client=True, timeout=30):
        """Sends an MCP request and waits for its answer (an "error" stands
        for a server that exits or doesn't answer)."""
        self.mcp_id += 1
        request_id = self.mcp_id
        self.mcp_send({"id": request_id, "method": method, "params": params})
        done = self.wait(lambda: request_id in self.mcp_responses or self.mcp_eof,
                         timeout=timeout, client=client)
        if request_id in self.mcp_responses:
            return self.mcp_responses.pop(request_id)
        return {"error": {"code": -1, "message": "the server exited" if done else "no answer"}}

    def on_mcp(self, message):
        if "method" in message and "id" in message:
            if message["method"] == "roots/list":
                cwd = Path(os.getcwd())
                result = {"roots": [{"uri": cwd.as_uri(), "name": cwd.name}]}
                self.mcp_send({"id": message["id"], "result": result})
            elif message["method"] == "ping":
                self.mcp_send({"id": message["id"], "result": {}})
            else:
                self.mcp_send({"id": message["id"], "error": {
                    "code": -32601, "message": "Method not found: %s" % message["method"]}})
            log("mcp_message", message=message)
        elif "method" in message:
            log("mcp_message", message=message)
        elif message.get("id") in self.abandoned:
            self.abandoned.discard(message.get("id"))
            log("mcp_late_answer", message=message)
        else:
            self.mcp_responses[message.get("id")] = message

    # The client's messages

    def on_client(self, message):
        if "method" in message and "id" in message:
            method = message["method"]
            params = message.get("params") or {}
            handler = {
                "initialize": self.initialize,
                "session/new": self.new_session,
                "session/set_config_option": self.set_config_option,
                "session/set_mode": self.set_mode,
                "session/prompt": self.prompt,
            }.get(method)
            if handler is None:
                log("unknown_request", method=method, params=params)
                self.error(message["id"], -32601, "Method not found: %s" % method)
            else:
                handler(message["id"], params)
        elif "method" in message:
            if message["method"] == "session/cancel":
                self.cancel(message.get("params") or {})
            else:
                log("notification", message=message)
        else:
            self.responses[message.get("id")] = message

    def initialize(self, request_id, params):
        log("initialize", params=params)
        capabilities = {"resume": {}, "close": {}, "load": {}, "list": {},
                        "additionalDirectories": {}}
        self.reply(request_id, {
            "protocolVersion": 1,
            "agentCapabilities": {
                "loadSession": True, "load_session": True,
                "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
                "sessionCapabilities": capabilities, "session_capabilities": capabilities,
                "auth": {"logout": {}},
                "_meta": {"antigravity": {"tier": True, "systemPrompt": {"append": True}}}},
            "agentInfo": {"name": "Antigravity ACP Server", "version": self.version},
            "authMethods": AUTH_METHODS})

    def check_profile(self):
        problems = list(self.setup_problems)
        agent = None
        if self.agent:
            path = self.gemini_dir / "config" / "agents" / (self.agent + ".md")
            agent, self.fallback = read_agent(path, self.agent)
            if agent is not None:
                problems += agent_problems(agent)
        if self.script.get("fallback"):
            self.fallback = self.fallback or "the script asks for it"
        self.default_agent = not self.agent or self.fallback is not None
        if self.default_agent:
            self.hooks = read_hooks(self.gemini_dir / "config" / "hooks.json")
        builtins = KEPT_BUILTINS + (DEFAULT_BUILTINS if self.default_agent else [])
        self.profile = {
            "agent": self.agent,
            "frontmatter": agent["frontmatter"] if agent else None,
            "system_prompt": agent["body"] if agent else None,
            "mcp_config": self.mcp_config,
            "tools": self.offered,
            "visible": ["mcp_arko_" + name for name in self.offered] + builtins,
            "default_agent": self.default_agent,
            "fallback": self.fallback,
            "hooks": self.hooks,
            "problems": problems,
        }
        log("profile", **self.profile)

    def model(self, value):
        models = self.script.get("models", MODELS)
        return next((model for model in models if model["value"] == value), None)

    def config_options(self, session):
        models = self.script.get("models", MODELS)
        options = [{"id": "model", "name": "Model", "category": "model", "type": "select",
                    "currentValue": session.model,
                    "options": [{"value": model["value"], "name": model["name"]}
                                for model in models]}]
        levels = self.model(session.model)["levels"]
        if levels:
            options.append({"id": "thought_level", "name": "Thinking Level",
                            "category": "thought_level", "type": "select",
                            "currentValue": session.level,
                            "options": [{"value": level, "name": LEVEL_NAMES.get(level, level)}
                                        for level in levels]})
        options.append({"id": "mode", "name": "Mode", "category": "mode", "type": "select",
                        "currentValue": session.mode,
                        "options": [{"value": mode["id"], "name": mode["name"],
                                     "description": mode["description"]} for mode in MODES]})
        return options

    def new_session(self, request_id, params):
        log("session_new", params=params)
        if self.script.get("signed_out"):
            self.error(request_id, -32000, "Authentication required")
            return
        if self.profile is None:
            self.check_profile()
        if self.profile["problems"]:
            self.error(request_id, -32603,
                       "fake agy: profile: %s" % "; ".join(self.profile["problems"]))
            return
        if self.fallback and self.script.get("fallback") != "silent":
            self.log_line('W Agent "%s" not found, falling back to default' % self.agent)
        session = Session(str(uuid.uuid4()), params.get("cwd"))
        self.log_line("I session %s created" % session.id)
        models = self.script.get("models", MODELS)
        if not self.model(session.model):
            session.model = models[0]["value"]
        self.fit_level(session)
        self.sessions[session.id] = session
        available = []
        for model in models:
            if model["levels"]:
                available += [{"modelId": "%s-%s" % (model["value"], level),
                               "name": "%s (%s)" % (model["name"],
                                                    MODEL_LEVEL_NAMES.get(level, level))}
                              for level in model["levels"]]
            else:
                available.append({"modelId": model["value"], "name": model["name"]})
        result = {"sessionId": session.id,
                  "models": {"availableModels": available,
                             "currentModelId": self.model_id(session)},
                  "modes": {"currentModeId": session.mode, "availableModes": MODES},
                  "configOptions": self.config_options(session)}
        tier = self.script.get("tier", "Google AI Pro")
        if tier is not None:
            result["_meta"] = {"antigravity": {"tier": {
                "name": tier, "id": "g1-pro-tier",
                "upgradeUri": "https://antigravity.google/g1-upgrade"}}}
        self.reply(request_id, result)

    def model_id(self, session):
        return session.model + ("-" + session.level if session.level else "")

    def fit_level(self, session):
        levels = self.model(session.model)["levels"]
        if not levels:
            session.level = None
        elif session.level not in levels:
            session.level = "high" if "high" in levels else levels[-1]

    def session(self, request_id, params):
        session = self.sessions.get(params.get("sessionId"))
        if session is None:
            self.error(request_id, -32602, "Session not found: %s" % params.get("sessionId"))
        return session

    def set_config_option(self, request_id, params):
        log("config", params=params)
        session = self.session(request_id, params)
        if session is None:
            return
        config, value = params.get("configId"), params.get("value")
        if config == "model":
            if not self.model(value):
                self.error(request_id, -32602, "Unknown model: %s" % value)
                return
            session.model = value
            self.fit_level(session)
        elif config == "thought_level":
            if value not in self.model(session.model)["levels"]:
                self.error(request_id, -32602, "Unknown thought level: %s" % value)
                return
            session.level = value
            self.update(session, {"sessionUpdate": "config_option_update",
                                  "configOptions": self.config_options(session)})
        elif config == "mode":
            if value not in [mode["id"] for mode in MODES]:
                self.error(request_id, -32602, "Unknown mode: %s" % value)
                return
            session.mode = value
        else:
            self.error(request_id, -32602, "Unknown config option: %s" % config)
            return
        self.reply(request_id, {"configOptions": self.config_options(session)})

    def set_mode(self, request_id, params):
        self.set_config_option(request_id, {"sessionId": params.get("sessionId"),
                                            "configId": "mode", "value": params.get("modeId")})

    def cancel(self, params):
        turn = self.turn
        if turn is not None and turn.session.id == params.get("sessionId"):
            log("cancel", sessionId=params.get("sessionId"), step=turn.step)
            turn.cancelled = True
        else:
            log("cancel", sessionId=params.get("sessionId"), idle=True)

    # Turns

    def prompt(self, request_id, params):
        session = self.session(request_id, params)
        if session is None:
            return
        blocks = params.get("prompt") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        turn = self.turn
        if turn is not None:
            if turn.session is not session:
                self.error(request_id, -32603, "fake agy runs one turn at a time")
                return
            # As in agy, the running prompt's result comes back at once, and
            # the new one's result ends the turn.
            self.reply(turn.prompt_id, {"stopReason": "end_turn", "usage": self.take_usage(turn)})
            log("note", sessionId=session.id, text=text, blocks=blocks, step=turn.step,
                ended=turn.prompt_id, id=request_id)
            turn.prompt_id = request_id
            turn.notes.append(text)
            turn.note_count += 1
            return
        log("prompt", sessionId=session.id, id=request_id, text=text, blocks=blocks,
            model=self.model_id(session), meta=params.get("_meta"))
        session.prompts += 1
        if session.prompts == 1:
            self.update(session, {"sessionUpdate": "config_option_update",
                                  "configOptions": self.config_options(session)})
            self.update(session, {"sessionUpdate": "available_commands_update",
                                  "availableCommands": COMMANDS})
            self.update(session, {"sessionUpdate": "usage_update", "used": session.used,
                                  "size": self.context_size})
            self.run_hooks("SessionStart", session, text)
        if session.poisoned:
            self.update(session, {"sessionUpdate": "agent_message_chunk",
                                  "content": {"type": "text", "text": session.poisoned}})
            self.reply(request_id, {"stopReason": "end_turn"})
            log("turn_end", sessionId=session.id, id=request_id, stopReason="end_turn",
                failed=session.poisoned)
            return
        index, steps = self.turns.pick(text)
        log("turn", sessionId=session.id, index=index, steps=steps)
        if session.prompts == 1:
            self.update(session, {"sessionUpdate": "session_info_update",
                                  "title": "Fake agy session"})
        turn = Turn(session, request_id, text)
        self.turn = turn
        try:
            self.play(turn, steps)
        finally:
            self.turn = None

    def play(self, turn, steps):
        try:
            for index, step in enumerate(steps):
                turn.step = index
                self.take_notes(turn)
                if turn.cancelled:
                    raise Cancelled()
                if step.get("usage") is not None:
                    self.count(turn, step["usage"])
                self.play_step(turn, step)
            turn.step = len(steps)
            self.take_notes(turn)
            if turn.cancelled:
                raise Cancelled()
            self.finish(turn, "end_turn")
        except Cancelled:
            self.finish(turn, "cancelled")
        except TurnOver:
            pass

    def play_step(self, turn, step):
        session = turn.session
        if "text" in step or "thought" in step:
            kind = "agent_message_chunk" if "text" in step else "agent_thought_chunk"
            pieces = step.get("text", step.get("thought"))
            for piece in pieces if isinstance(pieces, list) else [pieces]:
                self.update(session, {"sessionUpdate": kind,
                                      "content": {"type": "text", "text": piece}})
        elif "tool" in step:
            self.tool_calls(turn, [step])
        elif "parallel" in step:
            self.tool_calls(turn, step["parallel"])
        elif "builtin" in step:
            self.builtin(turn, step)
        elif "sleep" in step:
            self.wait(lambda: False, turn, timeout=step["sleep"])
        elif "wait_for" in step:
            if step["wait_for"] == "cancel":
                ready = lambda: turn.cancelled
            else:
                ready = lambda: turn.note_count > turn.notes_waited
            if self.wait(ready, turn, timeout=step.get("timeout", 30)):
                if step["wait_for"] != "cancel":
                    turn.notes_waited += 1
            else:
                log("wait_timeout", step=step)
        elif "wait_file" in step:
            path = step["wait_file"]
            deadline = time.time() + step.get("timeout", 30)
            while not os.path.exists(path) and time.time() < deadline:
                self.wait(lambda: False, turn, timeout=0.02)
            if not os.path.exists(path):
                log("wait_timeout", step=step)
        elif "fail" in step:
            failure = step["fail"]
            if isinstance(failure, dict):
                text = "\n⚠ %s (code %s): %s\n" % (
                    failure.get("status", "INTERNAL"), failure.get("code", 500),
                    failure.get("message", ""))
            else:
                text = "\n⚠ %s\n" % failure
            self.update(session, {"sessionUpdate": "agent_message_chunk",
                                  "content": {"type": "text", "text": text}})
            session.poisoned = text
            self.reply(turn.prompt_id, {"stopReason": "end_turn"})
            log("turn_end", sessionId=session.id, id=turn.prompt_id, stopReason="end_turn",
                failed=text)
            raise TurnOver()
        elif "end" in step:
            self.finish(turn, step["end"])
            raise TurnOver()
        elif "error" in step:
            self.send({"id": turn.prompt_id, "error": step["error"]})
            log("turn_end", sessionId=session.id, id=turn.prompt_id, error=step["error"])
            raise TurnOver()
        elif "exit" in step:
            self.shutdown(step["exit"])
        elif "update" in step:
            self.update(session, step["update"])
        elif "request" in step:
            self.requests += 1
            request_id = str(self.requests)
            self.send({"id": request_id, "method": step["request"],
                       "params": step.get("params", {})})
            self.wait(lambda: request_id in self.responses, turn)
            log("client_response", method=step["request"],
                response=self.responses.pop(request_id))
        elif "raw" in step:
            self.out.write(step["raw"].encode("utf-8") + b"\n")
            self.out.flush()
        elif "usage" not in step:
            raise SystemExit("fake agy: unknown step %s" % json.dumps(step))

    def take_notes(self, turn):
        for text in turn.notes:
            log("note_taken", sessionId=turn.session.id, text=text, step=turn.step)
        turn.notes = []

    def count(self, turn, usage):
        usage = dict(usage)
        usage.setdefault("totalTokens", sum(usage.get(name, 0) for name in (
            "inputTokens", "cachedReadTokens", "cachedWriteTokens", "outputTokens")))
        for name, value in usage.items():
            turn.usage[name] = turn.usage.get(name, 0) + value
        turn.counted = True
        turn.session.used = usage["totalTokens"]
        self.update(turn.session, {"sessionUpdate": "usage_update", "used": usage["totalTokens"],
                                   "size": self.context_size})

    def take_usage(self, turn):
        usage = {name: turn.usage.get(name, 0) for name in (
            "inputTokens", "outputTokens", "cachedReadTokens", "cachedWriteTokens",
            "thoughtTokens", "totalTokens")}
        turn.usage = {}
        return usage

    def finish(self, turn, reason):
        if not turn.counted and reason != "cancelled":
            self.count(turn, {"inputTokens": 100, "outputTokens": 10})
        usage = self.take_usage(turn)
        self.reply(turn.prompt_id, {"stopReason": reason, "usage": usage})
        log("turn_end", sessionId=turn.session.id, id=turn.prompt_id, stopReason=reason,
            usage=usage)

    def call_fields(self, name, arguments, step, kind="other", title=None):
        return {"toolCallId": step.get("id") or call_id(), "name": name, "args": arguments,
                "kind": kind, "title": title or "Run %s" % name, "rawInput": arguments}

    def call_update(self, session, fields, status, first=False, **extra):
        update = {"sessionUpdate": "tool_call" if first else "tool_call_update"}
        update.update(fields, status=status, **extra)
        self.update(session, update)

    def ask_permission(self, turn, fields):
        """Asks the client's permission for a call, and returns whether it
        allowed it. A rejection fails the call and ends the turn."""
        self.requests += 1
        request_id = str(self.requests)
        self.send({"id": request_id, "method": "session/request_permission", "params": {
            "sessionId": turn.session.id,
            "toolCall": dict({"sessionUpdate": "tool_call"}, **dict(fields, status="pending")),
            "options": PERMISSION_OPTIONS}})
        try:
            self.wait(lambda: request_id in self.responses, turn)
        except Cancelled:
            log("permission", tool=fields["name"], toolCallId=fields["toolCallId"],
                withdrawn=True)
            raise
        answer = self.responses.pop(request_id)
        outcome = (answer.get("result") or {}).get("outcome")
        allowed = outcome == {"outcome": "selected", "optionId": "allow"}
        log("permission", tool=fields["name"], toolCallId=fields["toolCallId"], answer=answer,
            allowed=allowed)
        return allowed

    def reject(self, turn, fields):
        # A rejected permission ends the whole turn, as in agy.
        self.call_update(turn.session, fields, "in_progress")
        self.call_update(turn.session, fields, "failed")
        self.finish(turn, "end_turn")
        raise TurnOver()

    def gate(self, turn, fields):
        """The default agent's PreToolUse hooks. Returns whether they allow the call."""
        allowed = self.run_hooks("PreToolUse", turn.session, turn.text, fields)
        if not allowed:
            self.call_update(turn.session, fields, "failed", rawOutput={"output": ""},
                             content=[{"type": "content",
                                       "content": {"type": "text", "text": ""}}])
        return allowed

    def tool_calls(self, turn, steps):
        session = turn.session
        calls = []
        for step in steps:
            name = step["tool"]
            name = name[len("mcp_arko_"):] if name.startswith("mcp_arko_") else name
            arguments = step.get("input", {})
            shown = dict(arguments, toolSummary=step.get("summary", "Fake call"))
            fields = self.call_fields("mcp_arko_" + name, shown, step)
            self.call_update(session, fields, "in_progress", first=True)
            if self.default_agent and not self.gate(turn, fields):
                continue
            if name not in self.offered:
                log("tool_failed", tool=fields["name"], reason="not offered")
                self.call_update(session, fields, "failed")
                continue
            asked = step.get("permission", True)
            if asked:
                self.call_update(session, fields, "pending")
                if not self.ask_permission(turn, fields):
                    self.reject(turn, fields)
            calls.append((name, arguments, fields, asked))
        running = {}
        for name, arguments, fields, asked in calls:
            if asked:
                self.call_update(session, fields, "in_progress")
            self.calls += 1
            self.mcp_id += 1
            conversation = session.id
            params = {"_meta": {
                "antigravity.google/artifacts_dir":
                    str(self.gemini_dir / "antigravity-cli" / "brain" / conversation),
                "antigravity.google/conversation_id": conversation,
                "antigravity.google/root_conversation_id": conversation,
                "progressToken": "%s:%d" % (self.progress, self.calls)},
                "name": name, "arguments": arguments}
            self.mcp_send({"id": self.mcp_id, "method": "tools/call", "params": params})
            log("tool_call", tool=name, arguments=arguments, toolCallId=fields["toolCallId"],
                id=self.mcp_id, meta=params["_meta"])
            running[self.mcp_id] = fields
        try:
            while running:
                self.wait(lambda: self.mcp_eof or any(key in self.mcp_responses
                                                      for key in running), turn)
                for key in [key for key in running if key in self.mcp_responses]:
                    answer = self.mcp_responses.pop(key)
                    result = answer.get("result") or {}
                    content = result.get("content", [])
                    log("tool_result", tool=running[key]["name"][len("mcp_arko_"):],
                        toolCallId=running[key]["toolCallId"], answer=answer,
                        text="".join(part.get("text", "") for part in content
                                     if part.get("type") == "text"),
                        images=[part.get("mimeType") for part in content
                                if part.get("type") == "image"],
                        isError=bool(result.get("isError")) or "error" in answer)
                    self.call_update(session, running.pop(key), "completed")
                if running and self.mcp_eof:
                    for key, fields in running.items():
                        log("tool_failed", tool=fields["name"], reason="the server exited")
                        self.call_update(session, fields, "failed")
                    running = {}
        except Cancelled:
            for key, fields in running.items():
                self.mcp_send({"method": "notifications/cancelled",
                               "params": {"reason": "context canceled", "requestId": key}})
                self.abandoned.add(key)
                self.call_update(session, fields, "failed")
            raise

    def builtin(self, turn, step):
        name = step["builtin"]
        arguments = step.get("input", {})
        fields = self.call_fields(name, arguments, step, kind=step.get("kind", "other"),
                                  title=step.get("title"))
        self.call_update(turn.session, fields, "in_progress", first=True)
        if self.default_agent and not self.gate(turn, fields):
            return
        if step.get("permission"):
            self.call_update(turn.session, fields, "pending")
            if not self.ask_permission(turn, fields):
                self.reject(turn, fields)
            self.call_update(turn.session, fields, "in_progress")
        log("builtin", tool=name, arguments=arguments, toolCallId=fields["toolCallId"])
        self.call_update(turn.session, fields, "completed",
                         rawOutput={"output": step.get("output", "")})

    def run_hooks(self, event, session, text, fields=None):
        """Runs the default agent's hooks for an event, and returns whether
        they allow it to go on. A hook that fails or times out blocks it."""
        if not self.default_agent:
            return True
        allowed = True
        for hook in self.hooks.get(event, []):
            matcher = hook.get("matcher", "*")
            if fields and matcher != "*" and not re.fullmatch(matcher, fields["name"]):
                continue
            conversation = session.id
            artifacts = self.gemini_dir / "antigravity-cli" / "brain" / conversation
            payload = {"artifactDirectoryPath": str(artifacts), "conversationId": conversation,
                       "cwd": session.cwd, "hookEventName": event, "lastUserInput": text,
                       "modelName": self.model_id(session),
                       "transcriptPath": str(artifacts / ".system_generated" / "logs"
                                             / "transcript_full.jsonl"),
                       "workspacePaths": [session.cwd]}
            if fields:
                payload["stepIdx"] = self.turn.step if self.turn else 0
                payload["toolCall"] = {"args": fields["args"], "id": fields["toolCallId"],
                                       "name": fields["name"]}
            try:
                cwd = session.cwd if session.cwd and os.path.isdir(session.cwd) else None
                done = subprocess.run(hook["command"], shell=True, input=json.dumps(payload),
                                      capture_output=True, text=True,
                                      timeout=hook.get("timeout", 60), cwd=cwd)
                output, code = done.stdout, done.returncode
            except subprocess.TimeoutExpired:
                output, code = "", "timeout"
            decision = None
            try:
                decision = json.loads(output).get("decision") if output.strip() else None
            except (ValueError, AttributeError):
                pass
            blocked = code != 0 or decision == "deny"
            log("hook", hook=event, command=hook["command"], input=payload, output=output,
                code=code, blocked=blocked)
            allowed = allowed and not blocked
        return allowed


# The self-check

def stub_mcp_server():
    """A small MCP server for the self-check, with echo(text) and
    sleep(seconds). It logs what it gets to FAKE_AGY_CHECK_MCP_LOG."""
    path = os.environ.get("FAKE_AGY_CHECK_MCP_LOG")
    lock = threading.Lock()

    def note(entry):
        if path:
            with lock, open(path, "a") as out:
                out.write(json.dumps(entry) + "\n")

    def send(message):
        with lock:
            sys.stdout.write(json.dumps(dict(jsonrpc="2.0", **message)) + "\n")
            sys.stdout.flush()

    def slow(request_id, seconds):
        time.sleep(seconds)
        send({"id": request_id, "result": {"content": [{"type": "text", "text": "slept"}]}})

    tools = [{"name": "echo", "inputSchema": {"type": "object"}},
             {"name": "sleep", "inputSchema": {"type": "object"}}]
    note({"started": True, "marker": os.environ.get("FAKE_AGY_CHECK_MARKER")})
    for line in sys.stdin:
        message = json.loads(line)
        note(message)
        method = message.get("method")
        if "id" not in message:
            continue
        if method == "initialize":
            send({"id": message["id"], "result": {
                "protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                "serverInfo": {"name": "check", "version": "1"}}})
        elif method == "tools/list":
            send({"id": message["id"], "result": {"tools": tools}})
        elif method == "tools/call" and message["params"]["name"] == "sleep":
            threading.Thread(target=slow, daemon=True, args=(
                message["id"], message["params"]["arguments"]["seconds"])).start()
        elif method == "tools/call":
            text = "echo: %s" % message["params"]["arguments"].get("text", "")
            send({"id": message["id"], "result": {"content": [{"type": "text", "text": text}]}})
        else:
            send({"id": message["id"], "error": {"code": -32601, "message": "unsupported"}})


class CheckClient:
    """A minimal ACP client that drives the fake for the self-check."""

    def __init__(self, args, env, cwd):
        self.child = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, env=env, cwd=cwd)
        self.messages = queue.Queue()
        self.next_id = 0
        self.permission = "allow"
        self.seen = []
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.child.stdout:
            self.messages.put(json.loads(line))
        self.messages.put(None)

    def send(self, message):
        self.child.stdin.write(json.dumps(dict(jsonrpc="2.0", **message)).encode() + b"\n")
        self.child.stdin.flush()

    def request(self, method, params):
        self.next_id += 1
        self.send({"id": self.next_id, "method": method, "params": params})
        return self.next_id

    def until(self, wanted, timeout=20):
        """Reads messages until `wanted(message)`, answering permission
        requests as `self.permission` says, and returns that message."""
        deadline = time.time() + timeout
        while True:
            message = self.messages.get(timeout=max(0.01, deadline - time.time()))
            if message is None:
                raise AssertionError("the fake exited: %s" % self.child.stderr.read().decode())
            self.seen.append(message)
            if message.get("method") == "session/request_permission":
                self.send({"id": message["id"], "result": {"outcome": {
                    "outcome": "selected", "optionId": self.permission}}})
            elif "method" in message and "id" in message:
                self.send({"id": message["id"], "error": {
                    "code": -32601, "message": "Method not found"}})
            if wanted(message):
                return message

    def call(self, method, params):
        request_id = self.request(method, params)
        return self.until(lambda message: message.get("id") == request_id
                          and "method" not in message)

    def result(self, request_id):
        return self.until(lambda message: message.get("id") == request_id
                          and "method" not in message)

    def chunks(self):
        return "".join(message["params"]["update"]["content"]["text"] for message in self.seen
                       if message.get("method") == "session/update"
                       and message["params"]["update"]["sessionUpdate"] == "agent_message_chunk")

    def close(self):
        self.child.stdin.close()
        self.child.wait(timeout=10)


def self_check():
    """Plays each scenario against this fake with a stub MCP server, and
    exits with an error at the first thing that differs."""
    root = Path(tempfile.mkdtemp(prefix="fake-agy-check-"))
    try:
        checks = Checks(root)
        for name in [name for name in dir(checks) if name.startswith("check_")]:
            getattr(checks, name)()
            print("ok %s" % name[len("check_"):])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("fake agy self-check passed")


class Checks:
    def __init__(self, root):
        self.root = root
        self.count = 0

    def setup(self, script, agent=True, eager=("echo", "sleep"), frontmatter=None,
              config_extra=None, hooks=None, env_extra=None, base=None):
        self.count += 1
        shared = base
        base = self.root / str(self.count)
        profile = base / "profile"
        (profile / "config" / "agents").mkdir(parents=True)
        (base / "work").mkdir()
        if agent:
            keys = frontmatter or ("name: arko\ndescription: Example agent.\n"
                                   "excludeDefaultComponents: true\n"
                                   "inheritCustomizations: false\nmainAgent: true\n"
                                   "subagent: false\n")
            (profile / "config" / "agents" / "arko.md").write_text(
                "---\n%s---\n\nYou are a test agent.\n" % keys)
        server = {"command": sys.executable, "timeoutSeconds": 604800,
                  "args": [os.path.abspath(__file__), "--self-check-mcp"],
                  "tools": {name: {"eager": True} for name in eager}}
        server.update(config_extra or {})
        (profile / "config" / "mcp_config.json").write_text(
            json.dumps({"mcpServers": {"arko": server}}))
        if hooks:
            (profile / "config" / "hooks.json").write_text(json.dumps(hooks))
        # A run that shares another's script and log, as workers would.
        files = shared or base
        if not shared:
            (base / "script.json").write_text(json.dumps(script))
        env = dict(os.environ, AGY_CLI_DISABLE_AUTO_UPDATE="true",
                   FAKE_AGY_SCRIPT=str(files / "script.json"),
                   FAKE_AGY_LOG=str(files / "log.jsonl"),
                   FAKE_AGY_CHECK_MCP_LOG=str(base / "mcp.jsonl"),
                   FAKE_AGY_CHECK_MARKER="inherited")
        env.update(env_extra or {})
        args = ["acp", "--gemini-dir", str(profile), "--agent", "arko",
                "--log-file", str(profile / "acp.log"),
                "--include-default-customizations=false", "--disable-telemetry"]
        self.base, self.profile, self.files = base, profile, files
        return CheckClient(args, env, str(base / "work"))

    def events(self, name=None):
        path = self.files / "log.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        events = [json.loads(line) for line in lines]
        return [event for event in events if name is None or event["event"] == name]

    def mcp_log(self):
        return [json.loads(line) for line in (self.base / "mcp.jsonl").read_text().splitlines()]

    def start(self, client):
        client.call("initialize", {"protocolVersion": 1, "clientCapabilities": {
            "fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False}})
        reply = client.call("session/new", {"cwd": str(self.base / "work"), "mcpServers": []})
        assert "result" in reply, reply
        return reply["result"]["sessionId"]

    def prompt(self, client, session, text):
        return client.request("session/prompt", {"sessionId": session,
                                                 "prompt": [{"type": "text", "text": text}]})

    def check_reply(self):
        client = self.setup([[{"thought": "Thinking."}, {"text": ["Hel", "lo"], "usage": {
            "inputTokens": 900, "cachedReadTokens": 100, "outputTokens": 20}}]],
            env_extra={"ARKO_AGY_RELAY_TOKEN": "relay-token-value",
                       "FAKE_AGY_LOG_ENV": "ARKO_AGY_RELAY_TOKEN"})
        session = self.start(client)
        level = client.call("session/set_config_option", {
            "sessionId": session, "configId": "thought_level", "value": "xhigh"})
        assert level["result"]["configOptions"][1]["currentValue"] == "xhigh", level
        flash = client.call("session/set_config_option", {
            "sessionId": session, "configId": "model", "value": "gemini-3.8-flash"})
        assert flash["result"]["configOptions"][1]["currentValue"] == "high", flash
        result = client.result(self.prompt(client, session, "Say hello."))["result"]
        assert result == {"stopReason": "end_turn", "usage": {
            "inputTokens": 900, "outputTokens": 20, "cachedReadTokens": 100,
            "cachedWriteTokens": 0, "thoughtTokens": 0, "totalTokens": 1020}}, result
        assert client.chunks() == "Hello", client.chunks()
        client.close()
        profile = self.events("profile")[0]
        assert profile["system_prompt"].strip() == "You are a test agent.", profile
        assert profile["visible"][:2] == ["mcp_arko_echo", "mcp_arko_sleep"], profile
        assert not profile["fallback"] and not profile["problems"], profile
        assert self.events("prompt")[0]["model"] == "gemini-3.8-flash-high"
        assert self.events("argv")[0]["env"]["ARKO_AGY_RELAY_TOKEN"] == "<set>"
        assert "relay-token-value" not in (self.files / "log.jsonl").read_text()
        acp_log = (self.profile / "acp.log").read_text()
        assert "acp started" in acp_log and "falling back" not in acp_log, acp_log
        assert self.mcp_log()[0]["marker"] == "inherited"

    def check_tool_call(self):
        client = self.setup([[{"tool": "echo", "input": {"text": "hi"}}, {"text": "done"}]])
        session = self.start(client)
        client.result(self.prompt(client, session, "Call echo."))
        client.close()
        asked = [message for message in client.seen
                 if message.get("method") == "session/request_permission"]
        assert len(asked) == 1 and isinstance(asked[0]["id"], str), asked
        assert asked[0]["params"]["toolCall"]["rawInput"]["toolSummary"], asked
        call = [entry for entry in self.mcp_log() if entry.get("method") == "tools/call"][0]
        assert call["params"]["arguments"] == {"text": "hi"}, call
        assert call["params"]["_meta"]["antigravity.google/conversation_id"] == session
        assert self.events("tool_result")[0]["text"] == "echo: hi"
        methods = [entry.get("method") for entry in self.mcp_log()]
        assert methods[1:5] == ["server/discover", "initialize", "notifications/initialized",
                                "tools/list"], methods

    def check_rejected_permission(self):
        client = self.setup([[{"tool": "echo", "input": {"text": "hi"}}, {"text": "after"}]])
        client.permission = "deny"
        session = self.start(client)
        result = client.result(self.prompt(client, session, "Call echo."))["result"]
        client.close()
        assert result["stopReason"] == "end_turn" and "after" not in client.chunks(), result
        assert not [entry for entry in self.mcp_log() if entry.get("method") == "tools/call"]

    def check_note(self):
        client = self.setup([[{"wait_for": "note"}, {"text": "noted"}]])
        session = self.start(client)
        first = self.prompt(client, session, "Start.")
        second = self.prompt(client, session, "A note.")
        assert client.result(first)["result"]["stopReason"] == "end_turn"
        assert "noted" not in client.chunks()
        assert client.result(second)["result"]["stopReason"] == "end_turn"
        assert client.chunks() == "noted"
        client.close()
        assert [event["text"] for event in self.events("note_taken")] == ["A note."]

    def check_note_during_tool(self):
        client = self.setup([[{"tool": "sleep", "input": {"seconds": 0.5}},
                              {"wait_for": "note", "timeout": 2}, {"text": "noted"}]])
        session = self.start(client)
        first = self.prompt(client, session, "Start.")
        client.until(lambda message: message.get("method") == "session/update"
                     and message["params"]["update"].get("status") == "in_progress"
                     and message["params"]["update"]["sessionUpdate"] == "tool_call_update")
        second = self.prompt(client, session, "A note.")
        usage = client.result(first)["result"]["usage"]
        assert usage["totalTokens"] == 0, usage
        assert client.result(second)["result"]["usage"]["totalTokens"] == 110
        client.close()
        assert [event["step"] for event in self.events("note_taken")] == [1]
        assert not self.events("wait_timeout")

    def check_turn_matching(self):
        script = [{"match": "beta", "steps": [{"text": "B"}]},
                  {"match": "alpha", "steps": [{"text": "A"}]}, [{"text": "rest"}]]
        first = self.setup(script)
        files = self.base
        second = self.setup(script, base=files)
        replies = []
        for client, text in ((first, "the alpha task"), (second, "the beta task"),
                             (first, "a follow-up"), (second, "another")):
            session = self.start(client) if not client.seen else client.session
            client.session = session
            before = len(client.chunks())
            client.result(self.prompt(client, session, text))
            replies.append(client.chunks()[before:])
        first.close()
        second.close()
        assert replies == ["A", "B", "rest", "script exhausted"], replies

    def check_cancel(self):
        client = self.setup([[{"tool": "sleep", "input": {"seconds": 5}}, {"text": "late"}],
                             [{"text": "OK"}]])
        session = self.start(client)
        first = self.prompt(client, session, "Sleep.")
        client.until(lambda message: message.get("method") == "session/update"
                     and message["params"]["update"].get("status") == "in_progress"
                     and message["params"]["update"]["sessionUpdate"] == "tool_call_update")
        client.send({"method": "session/cancel", "params": {"sessionId": session}})
        result = client.result(first)["result"]
        assert result["stopReason"] == "cancelled" and "usage" in result, result
        second = client.result(self.prompt(client, session, "Again."))["result"]
        assert second["stopReason"] == "end_turn" and client.chunks() == "OK"
        client.close()
        cancelled = [entry for entry in self.mcp_log()
                     if entry.get("method") == "notifications/cancelled"]
        assert cancelled and cancelled[0]["params"]["reason"] == "context canceled", cancelled

    def check_failed_turn(self):
        failure = {"status": "INVALID_ARGUMENT", "code": 400, "message": "Bad image."}
        client = self.setup([[{"fail": failure}], [{"text": "fresh"}]])
        session = self.start(client)
        result = client.result(self.prompt(client, session, "One."))["result"]
        assert result == {"stopReason": "end_turn"}, result
        text = "\n⚠ INVALID_ARGUMENT (code 400): Bad image.\n"
        assert client.chunks() == text
        client.result(self.prompt(client, session, "Two."))
        assert client.chunks() == text * 2
        fresh = self.start(client)
        client.result(self.prompt(client, fresh, "Three."))
        assert client.chunks().endswith("fresh")
        client.close()

    def check_fallback(self):
        hook = "%s -c \"print('{\\\"decision\\\": \\\"deny\\\"}')\"" % sys.executable
        client = self.setup([[{"builtin": "run_command", "input": {"CommandLine": "ls"}}]],
                            agent=False, hooks={"gate": {"PreToolUse": [
                                {"matcher": "*", "hooks": [{"type": "command", "command": hook,
                                                            "timeout": 10}]}]}})
        session = self.start(client)
        line = 'Agent "arko" not found, falling back to default'
        assert line in (self.profile / "acp.log").read_text()
        client.result(self.prompt(client, session, "Run ls."))
        client.close()
        assert self.events("hook")[0]["blocked"] is True
        assert not self.events("builtin")

    def check_profile_problems(self):
        client = self.setup([], eager=("echo",))
        client.call("initialize", {"protocolVersion": 1})
        reply = client.call("session/new", {"cwd": str(self.base / "work"), "mcpServers": []})
        assert "tool sleep isn't marked eager" in reply["error"]["message"], reply
        client.close()
        client = self.setup([], frontmatter="name: arko\ndescription: Example.\n"
                                            "excludeDefaultComponents: false\n")
        client.call("initialize", {"protocolVersion": 1})
        reply = client.call("session/new", {"cwd": str(self.base / "work"), "mcpServers": []})
        message = reply["error"]["message"]
        assert "excludeDefaultComponents is False" in message and "no mainAgent" in message
        client.close()

    def check_launch(self):
        env = dict(os.environ, AGY_CLI_DISABLE_AUTO_UPDATE="1")
        env.pop("FAKE_AGY_SCRIPT", None)
        done = subprocess.run([sys.executable, os.path.abspath(__file__), "acp",
                               "--gemini-dir", str(self.root), "--bogus"],
                              env=env, capture_output=True, text=True)
        assert done.returncode == 2 and "--bogus" in done.stderr, done
        assert "AGY_CLI_DISABLE_AUTO_UPDATE" in done.stderr, done
        env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
        done = subprocess.run([sys.executable, os.path.abspath(__file__), "--version"],
                              env=env, capture_output=True, text=True)
        assert done.stdout == DEFAULT_VERSION + "\n", done
        done = subprocess.run([sys.executable, os.path.abspath(__file__), "-p", "/usage",
                               "--output-format", "json"], env=env, capture_output=True,
                              text=True)
        groups = json.loads(done.stdout)["command"]["data"]["groups"]
        assert [group["name"] for group in groups][0] == "Gemini Models", done


def main():
    args = sys.argv[1:]
    if args == ["--self-check"]:
        self_check()
        return
    if args == ["--self-check-mcp"]:
        stub_mcp_server()
        return
    script, state_path = load_script()
    if args == ["--version"]:
        log("version", argv=args, env=environment())
        sys.stdout.write(script.get("version_output", version(script) + "\n"))
        sys.stdout.flush()
    elif args[:1] == ["-p"]:
        usage_command(args, script)
    elif args[:1] == ["acp"]:
        Agy(args[1:], script, state_path).run()
    else:
        refuse(["unknown command %s" % " ".join(args)])


if __name__ == "__main__":
    main()
