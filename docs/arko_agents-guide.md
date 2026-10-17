# arko_agents guide

This guide covers arko_agents in depth. An `AgentConfig` describes an agent with its role
and instructions, model, tools, and approval and error policies. A `Task` holds one run's
conversation and state. `AgentRuntime` runs a task as a Tokio task, streams `AgentEvent`s,
and accepts `UserCommand`s for approvals, answers, error recovery, context updates,
cancellation, and suspension. [arko_workflows](arko_workflows-guide.md) builds multi-agent
orchestration on this crate. The [README](../arko_agents/README.md) shows how to add the
crate and runs a first task, and the [reference map](reference-map.md#arko_agents) lists
every setting's default.

## Events and commands

`run_task` returns the event receiver, the command sender, and a join handle that
resolves to the task in its final state. `TaskStreamEnd` is the last event of a run. Keep
receiving events until then. The channel holds 100 events and the runtime waits while it
is full, and if the receiver is dropped, the run ends with `ArkoAgentsError::ChannelSend`
instead of returning the task. Keep the command sender alive too, because when every sender has
been dropped, the runtime returns the task as `Interrupted`. Tools only hold weak senders.

The task's working directory is the agent's workspace. File tools resolve relative paths
against it, and paths outside it are rejected by default (see
[Approvals and workspace paths](#approvals-and-workspace-paths)). A relative working
directory is resolved against the process's current directory.

The `arko_agents_cli` example runs one agent from a JSON configuration, with interactive
approvals, answers, and error recovery (the README shows how to run it). Its options include `--tool-calling-mode native|output_based`,
`--simplified-tool-events false` (output-based calls also reported parameter by
parameter, `true` by default), `--mcp-config PATH`,
`--openai-auth-json PATH` (ChatGPT OAuth from a Codex `auth.json`, and the `OPENAI_OAUTH_*`
variables are then not read), `--save-task-state PATH` (the final task as JSON), and the
opt-in `--user-command-timeout SECONDS`, after
which a task waiting for a command is returned `Interrupted`. Without it the task waits
until a command arrives. The example loads a `.env` file and then reads other credentials
from the environment with `LLMClient::from_env()`.

## Agent configuration

`AgentConfig` is built with `AgentConfig::builder(id, name, llm_config)` or loaded from
JSON with `AgentConfig::load_from_path`. The builder requires a role description and
instructions and validates the retry, monitoring, and refusal-fallback settings. The
runtime validates them again when a task starts. The README example's agent looks like this
as JSON:

```json
{
  "id": "writer",
  "name": "Writer",
  "role_description": "You are a careful software engineer.",
  "instructions": "Make the requested change, then call complete_task with a summary.",
  "llm_config": { "provider": "Anthropic", "model_id": "claude-opus-5-5" },
  "tool_config": {
    "available_tool_names": ["list_files", "read_file", "write_file", "complete_task"],
    "tool_calling_mode": "Native"
  }
}
```

| Field | Default | Purpose |
| --- | --- | --- |
| `id`, `name` | Required | Identifier and display name. The name heads the system prompt and is the `agent_name` tools receive. |
| `role_description`, `instructions` | Required | The system prompt's role and instructions sections. |
| `llm_config` | Required | Model and `GenerationConfig`. See [Saved tasks and configurations](#saved-tasks-and-configurations). |
| `tool_config` | `complete_task` only, native tool calling, 20-minute tool time limit | Available tools, tool calling mode, approvals, parallel calls, tool time limits, and workspace path policy. |
| `environment_details_config` | Nothing included | Working directory, file tree, date and time, and operating system, each placed in the system prompt, the task prompt, or every user message. |
| `error_recovery_config` | `InformAgentAndContinue` | What a failed tool execution does. |
| `llm_error_policy`, `llm_retry_config` | 10 attempts over up to 5 minutes, then pause | What a failed model request does. See [Model-call recovery](#model-call-recovery). |
| `user_command_timeout_seconds` | None (waits indefinitely) | An opt-in limit on how long an awaiting task waits for a decision before it is returned `Interrupted`, counted from when the wait starts. `InjectContext` and commands the wait ignores don't restart it, and a wait the host is deciding (`HostDeciding`, or every wait of a run with `HostDecisions` set, as in a workflow) isn't timed. At least 1. |
| `user_reminder` | None | Text appended as `# Reminder` to the task prompt, every user message, or every Nth message (N at least 1). |
| `skills_config` | None | Skill sources. See [Skills](#skills). |
| `monitoring` | Nothing monitored | Plan-usage and prompt-cache monitors. See [Monitoring](#monitoring). |
| `refusal_fallback` | None | A model that answers a refused turn once. See [Refusals](#refusals). |
| `handover` | None | Opt-in. At a context size, the agent writes a report (with native tool calling, through the `handover_report` tool) and continues from it in a fresh conversation. See [Conversation handover](#conversation-handover). |
| `claude_code` | None (the defaults) | How the Claude Code CLI runs, for a Claude Code model. See [Claude Code models](#claude-code-models). |
| `complete_task_summary_prompt_override` | None | Replaces the description of `complete_task`'s `summary` parameter. |
| `parsing_config_override` | Default tags | Tag syntax for output-based tool calls. |

In `role_description` and `instructions`, text between `{{#output_based_only}}` and
`{{/output_based_only}}` is kept, without the markers, in output-based mode and removed
in native mode, so one configuration can carry tool-call examples for text-parsed calls.

Details placed in every user message (`EachUserMessage`) and reminders for every user
message or every Nth go with each message the model receives, tool results included. In
native mode they are text after the turn's results, in the same user message, which every
provider accepts. Every such message counts toward `EveryNUserMessages`. A context update
that follows another user message of the same turn (tool results, say) doesn't repeat the
environment details that message carried.

The system prompt is written once, when a task starts, so a date and time placed there
is labelled `Task Start Datetime (UTC)`. Placed in a user message it is labelled
`Current Datetime (UTC)`.

Developer-defined environment details are passed to `Task::new`. Each has static content
or a `Fn(&Task)` provider, which is not serialized. A task loaded from JSON needs its
dynamic providers set again. A task that starts with a detail it will show (every
`EachUserMessage` detail, and the others until the system prompt is written) that has no
content fails at once with `ArkoAgentsError::Configuration` naming the detail and its
provider. File-tree details default to a full
first snapshot followed by changes at model-call boundaries. See
[context management](context-efficiency.md#file-tree-context).

## Tools

`ToolRegistry::new()` registers the built-in tools `complete_task`, `read_file`,
`write_file`, `edit_file`, `apply_diff`, `list_files`, `ask_user`, and `message_user`.
`edit_file` replaces one exact piece of text in a file (or every occurrence, with
`replace_all`) and changes nothing when the text isn't there exactly once. `apply_diff`
applies SEARCH/REPLACE blocks. An agent is
offered the registered tools named in `tool_config.available_tool_names`, and always
`complete_task`, which ends the task as `Completed { summary }`. `ask_user` puts the task
in `AwaitingUserResponse` until `UserCommand::SubmitUserResponse` answers it.

`message_user` shows the user a message while the agent works, with nothing to wait for,
such as an answer to a question the user asked meanwhile, or progress. The runtime sends its
`message` parameter to the host as `MessageToUser { message, tool_call_id }`, between the
call's `ToolCallFormatted` and its `ToolResult`, and the task carries on in the state it
was in. The model's result says the message was sent. Like any tool, it follows
`default_auto_approve_tool_use`. A host tool registered under the same name replaces the
built-in, as for any built-in other than `complete_task`. It then runs as the host wrote
it, and the runtime sends no `MessageToUser` for its calls.

With `tool_config.text_reply_as_message` on (off by default), a native response with text
and no tool call is a message too. The runtime sends the text as `MessageToUser` without a
`tool_call_id`, adds a note that the user has it and the task is still open, and asks
again, outside error recovery. Only the first such reply since the agent's last tool call
or new message counts as a message. Another one right after it gets the usual note (see
[Model-call recovery](#model-call-recovery)), so a model that keeps answering in text
still reaches the attempt limit. The system prompt tells the model that a plain-text reply
reaches the user. Output-based mode ignores the setting.

With `tool_config.text_reply_ends_task` on (off by default), a native response with text and
no tool call is the agent's final reply instead. The task completes with the text as its
summary, exactly as a `complete_task` call with it would, and the agent isn't offered
`complete_task`. Text written beside tool calls stays progress. A response with neither
text nor a tool call still gets the note and counts as a recovery attempt, and a handover
report that is due comes first. The system prompt tells the model that a response without a
tool call ends the task. The reply needs no approval and passes no execution middleware, so
arko_workflows refuses the setting on an agent a completion verifier reviews or runs on. It
is refused together with `text_reply_as_message`, and output-based mode ignores it.

The runtime runs only the tools the agent is offered, in either tool calling mode. In
native mode, a call to any other tool, registered or not, gets an error result and emits
`UnavailableToolAttempted`. Output-based parsing recognizes only the offered tools' tags,
so other markup stays response text. `run_task` binds the offered tools to the task with
`Tool::bind_runtime`. A saved task is offered the tools its own configuration names.

### Tool calling modes

`tool_config.tool_calling_mode` selects how calls are exchanged with the model:

- `Native` (the default) sends tools through the provider's tool-calling API. A
  response can call several different tools. See
  [Several calls in one response](#several-calls-in-one-response).
- `OutputBased` (legacy) describes each tool in the system prompt with an XML-like example,
  and the runtime parses calls from the response text. A response holds one call or one
  batch of calls to the same parallel-safe tool, which run in order. Every parameter value
  arrives as a JSON string, except that `true` or `false` (in any case) for a `bool`
  parameter becomes a boolean.

A configuration that does not name a mode gets `Native`. A saved task records the mode it
ran with. A task saved by an older version without it ran output-based and loads that way.

In native mode, a `tool_config` in the model's `GenerationConfig` is honoured. Its
`tool_choice` and `parallel_tool_calls` are sent with the agent's tools (its own tool
definitions aren't, since the agent's tools come from the registry). Without one, no tool choice
is sent and the provider's `auto` applies, so tool use is optional and a plain answer is a
valid response. The system prompt tells the model that the task ends only when it calls
`complete_task`, and a response without a tool call gets a note asking it to continue with
its tools or call `complete_task` (see [Model-call recovery](#model-call-recovery)).
`ToolChoice::Required` forces a call on the models that accept it. A forced choice the
configured model (or the refusal fallback) doesn't accept
(`Model::supports_forced_tool_choice`) fails validation, at build and when a task starts.

An agent offered no tools at all (an empty `available_tool_names` with `text_reply_ends_task`
on, so it isn't offered `complete_task` either) makes requests with no tool configuration,
and its system prompt has no section about tools. On a Claude Code model it runs with Claude
Code's built-in tools off. A request without tools sends earlier tool calls and results as
text (`arko_ai::Conversation::tool_history_as_text`), so a conversation that used tools goes
on without them.

A chat is an agent set as one with `AgentToolConfig::chat()` (the `chat` field, off by default).
It is offered no tools, its reply in text ends its task, a host gives it none of its own
tools (a workflow adds no messaging, memory, subagent or registered tools, and lists no
agents to message), and a handover asks it for a summary in text and answers the user's
latest message from it (see [Conversation handover](#conversation-handover)). An agent that
only lists no tools isn't a chat, and it keeps what its host gives every agent.

### Several calls in one response

Every call of a response runs, and each gets exactly one result. No response is rejected
for holding several. Calls run in call order as under a fair read/write lock, as Codex
runs them. A parallel-safe call starts once every earlier call that isn't parallel-safe has
finished, and any other call once every earlier call has finished. So `[read, read]` run
together, and `[read, write, read]` runs the first read, then the write, then the second
read. A tool is parallel-safe when its definition sets
`parallel_safe`, unless its `tool_specific_configs` entry sets `parallel_safe` otherwise.
The built-in `read_file`, `list_files`, `message_user`, and `load_skill` are parallel-safe. `write_file`,
`edit_file`, `apply_diff`, `ask_user`, and `complete_task` are not. An MCP tool is parallel-safe only
when its server marks it `readOnlyHint: true`. Calls after one that ends the turn
(`complete_task`, `ask_user`, a tool error under `FailTask`, or a middleware's decision)
are not run, and their results say so. Parallel-safe calls already running beside it
finish. A call awaiting approval counts as not finished. See
[Approvals and workspace paths](#approvals-and-workspace-paths).

`complete_task` after an earlier call of its turn was rejected or failed does not complete
the task, since the model chose to finish before it saw those results. Its result says the
task wasn't completed and asks the model to review the results and call `complete_task`
again if the task is done. This holds even when `complete_task` needs approval. When the
failure is known before the approval would be asked, it isn't asked, and an approval given
before a failure became known doesn't complete the task either. `complete_task` always
runs alone, after the calls before it, whatever a per-tool setting says. (An output-based
response can't call `complete_task` after another tool, because it holds one call or one batch of
calls to the same tool.)

`tool_config.one_tool_call_per_turn: true` (off by default) asks for one call per
response. Native requests then disable parallel tool calls where the provider can (OpenAI
and OpenRouter `parallel_tool_calls: false`, Anthropic `disable_parallel_tool_use`), the
system prompt asks for one call per response, and a response that still holds several
(Gemini can't turn parallel calls off) runs them one at a time, in order.
`parallel_tool_calls: Some(false)` in the model's `tool_config` means the same. In
output-based mode the switch turns batching off.

Configurations saved with the former batching settings still load, and none of them makes
calls run at the same time. Those settings let a response hold several calls, which ran
one after another. `allow_batched_tool_calls` is ignored (`false` logs a warning once). A
per-tool `allow_batched_calls: false` loads as `parallel_safe: Some(false)` and `true` is
ignored. A saved tool definition's `supports_batched_calls` is ignored, so it loads as
not parallel-safe, and in output-based mode it can no longer be batched until its
definition or a per-tool setting makes it parallel-safe.

### Tool time limit

Every tool call runs under a host-side time limit, 20 minutes by default
(`tool_config.tool_timeout_seconds`, `DEFAULT_TOOL_TIMEOUT_SECONDS`). The limit is never
shown to the model and is not a tool parameter. A call that runs longer is stopped (an
end-user command it started is killed), and its result tells the model that the tool did
not finish within the host's time limit and was stopped. The tool-error policy then
applies as to any failed call (by default `InformAgentAndContinue`, which tells the model and
goes on with the task). Set `tool_timeout_seconds: None` (JSON `null`) to turn the limit off.
A tool's `tool_specific_configs` entry sets its own, with `tool_timeout_seconds:
Some(Some(seconds))` (a JSON number) or `Some(None)` (JSON `null`) for none. Tools with
their own job and wait handling are exempt (`Tool::uses_host_timeout` returns `false`).
Those are the terminal tools, whose jobs run in the background with their own deadlines, and a
workflow's `run_workflow_agent`, which runs another agent to completion, and
`wait_for_message`, which has its own limit. A per-tool setting applies to them too.
Output supplied with an approval runs nothing and has no limit.

Response text is streamed as it arrives in native mode. In output-based mode, a turn's
text and tool events are emitted once the response has been parsed.

### Tool result size

A tool returns all of its output, `read_file` the whole file, `list_files` the whole
tree, and command and MCP tools everything they print. One 8 MB file is about two million
tokens, more than any model's input limit, so a request carrying it would fail. The
**overflow guard** (`tool_config.result_limits.overflow_guard`, on by default) prevents
that, and changes nothing while the request may fit:

- Once per turn, as the turn's results (parallel calls included) go into the
  conversation, after result interceptors and distillers, it checks whether the next
  request would certainly be over the model's input limit (`LLMClient::max_input_tokens`,
  272,000 tokens on ChatGPT OAuth). No tokenizer is used. It counts the latest request's
  reported input, plus every byte added since (the reply and the results) at 10 bytes per
  token, more than any content packs into a token (code takes about 3.6 to 4.8 bytes per
  token, padded tables 8.6). When the request may fit, nothing changes. It is sent, and an
  overflow the provider reports is recovered from (below).
- When the overflow is certain, the largest results are shortened first, each to a
  head-and-tail preview of about 2% of the limit (4,000 to 16,000 tokens), until it no
  longer is. The others stay as they were. In a nearly full context, where even the
  previews leave it certain, the room left (by a bytes-per-token ratio calibrated from the
  task's own requests, less `margin_tokens`, by default 2% of the limit and at least 4,000
  tokens) is shared between the results above about 500 tokens. A smaller result is never
  shortened, and none is replaced by a notice larger than it.
- A shortened result keeps its first and last lines, a marker for what was left out, and
  a notice. The notice is the only part the model sees:

  ```text
  [Shortened: this result is about 2,100,000 tokens (8.2 MB, 81,431 lines), too large for
  your context. Shown: the first 410 and the last 80 lines. Look at the parts you need with
  the terminal, for example: wc -l data/big.csv; grep -n 'pattern' data/big.csv; head -n
  100 data/big.csv; sed -n '5000,5400p' data/big.csv; or a short script.]
  ```

  With the terminal tools offered it names commands (on the file, for `read_file`).
  Without them, with `read_file_line_ranges` on, it names `read_file`'s ranges.
  Otherwise it says the output was too large and to ask for less.
- If the provider still answers that the request is over the model's input limit (an
  `ErrorKind::ContextOverflow` error), the largest result added since the last answered
  request is shortened and the request is sent again, at most twice. The result becomes a preview
  when it is larger than one, and otherwise (or when it was shortened before) its notice
  alone.
  After that, or at once when that result is under about 500 tokens (it is then left
  alone), an agent that hands over (with `handover`, or a host's `HandoverThreshold`)
  leaves out the step that request added, the model's reply and its results, and hands
  over from the conversation as it was at the last answered request, which fit. Its
  report request starts with `STEP_LEFT_OUT`, which says the step was left out and asks
  the report to say what it was for, so the fresh conversation can do it in smaller
  parts. Context updates added since stay. Without a handover, the error ends recovery
  like any error that retrying can't fix, recorded as `ContextOverflow`.
- `ToolResultShortened` reports each shortened result, with its estimated size and the
  reason. The `ToolResult` event before it has the whole output.

Tighter limits are opt-in. `result_limits.max_result_tokens` caps every result, and a tool's
`tool_specific_configs` entry sets its own (`max_result_tokens: Some(Some(tokens))`, or
`Some(None)` for none). A result over its cap keeps its head and tail up to the cap, with
the same notice. Turn the guard off with
`ToolResultLimits::default().without_overflow_guard()`. Results then go in whole.

```rust
use arko_agents::{PerToolConfigOverride, ToolResultLimits};

// A researcher that should never pull more than about 20,000 tokens per call.
tool_config.result_limits = ToolResultLimits::default().max_result_tokens(20_000);
// One MCP tool that returns huge JSON gets a tighter cap.
tool_config.tool_specific_configs.insert(
    "mcp__db__query".into(),
    PerToolConfigOverride { max_result_tokens: Some(Some(8_000)), ..Default::default() },
);
```

When an agent is offered both `read_file` and the terminal tools, `read_file`'s
description gains one sentence (`read_file::TERMINAL_HINT`) pointing to the terminal for
large or generated files. Likewise, when it is offered both `write_file` and `edit_file`,
`write_file`'s description gains one sentence (`write_file::EDIT_FILE_HINT`) pointing to
`edit_file` for changes to part of a file. Each is part of the tool's description, not of the agent's
instructions, so the same configuration always shows the same definitions, and a
workflow's resume check, which compares configurations, not tool definitions, isn't
affected. (In output-based mode the definitions are in the system prompt written when a
task starts, so a saved task keeps its own.) `tool_config.read_file_line_ranges` (off by default) adds the optional
`start_line` and `end_line` (1-based, inclusive) to `read_file`. Off, they are neither in
its schema nor mentioned, and a call that passes them anyway reads the whole file. Agents
given ranges tend to read in short pieces, so prefer the terminal where an agent has one.

The guard covers native tool calling in arko's own runtime. Output-based (legacy) tool
calling is not guarded, and neither are Claude Code agents, because Claude Code manages its
own context, and its `Read` and `Bash` have their own output limits.

### Approvals and workspace paths

Tool calls wait for approval unless `default_auto_approve_tool_use` or a per-tool
`auto_approve_tool_use` override is set. `ask_user` runs without approval, since its
question already waits for an answer, unless its own override sets `auto_approve_tool_use:
Some(false)`. When a response holds one call that needs
approval, the runtime emits `ToolCallAwaitingApproval`. When it holds several calls, it
emits `BatchToolCallAwaitingApproval` listing those that need approval. Each call carries
an `approval_reason` (`ApprovalReason`, also kept in the awaiting state), which says whether the
agent's configuration doesn't auto-approve the tool (`tool_not_auto_approved`), and which
path parameter is outside the workspace under `RequireApproval`
(`path_outside_workspace`). `describe()` puts it in words. Answer with
`ApproveToolUse`, `RejectToolUse`, or `SubmitBatchToolApprovals`, where a call without a
decision is rejected. An approval can carry `custom_tool_output`, which is used instead
of running the tool, and a `message_to_agent`. Approvals keep call order. A call awaiting
approval counts as not finished, and so does a call held behind one. For
`[apply_diff awaiting approval, read_file]`, the read waits until the diff is decided and
has run, or has been rejected. A call that must wait is held, saved with the task, until
the approvals are decided, and then runs with the approved calls in call order. The calls
after a rejected call run all the same. If the wait ends without a decision (the opt-in
command timeout or a closed channel), a held call's result says an earlier call of the turn
was still waiting for approval. Among the calls that run once approvals are decided, only
a call that ends the task (failing or cancelling it, or `complete_task`) leaves the rest
unrun. Every call gets exactly one result either way, although the results of calls that
waited follow the others in the message. Output-based mode answers its calls the same way.
When a call of a batch ends the task, makes it wait, or is stopped by a suspension, each
call of the batch that didn't run gets `ToolRejected` and a line in the turn's message
saying why.

Parameters marked `is_path` are checked against the task's working directory. A path
outside it is handled by `default_out_of_workspace_path_policy` or a per-tool override.
`AutoReject` (the default) rejects the call, `RequireApproval` asks for approval even for
auto-approved tools, and `AutoApprove` allows it. A native call is rejected with an error
result and `ToolRejected`. An output-based call is rejected as a tool error, under
`on_tool_execution_error`. A path that cannot be resolved, such as one holding a NUL
byte, is rejected this way under every policy.

The check and the built-in file tools resolve a path the same way, with
`arko_agents::utils::resolve_path`. Surrounding whitespace is trimmed, `~` and `$VAR` are
not expanded (`~/notes` names a directory called `~`), a relative path is joined to the
working directory, and the result is made absolute and its `.` and `..` resolved. On
Windows this follows Win32, so `\` and `/` both separate components, a drive-relative
`C:dir` uses that drive's current directory, trailing dots and spaces are dropped, and a
verbatim `\\?\` path is kept as written. A custom tool with an `is_path` parameter should
open the path `resolve_path` returns, so the check judges the path the tool opens.
End-user command tools receive that path. A non-string value for an `is_path` parameter
is rejected.

The check compares the resolved path with the working directory component by component.
On Windows it ignores ASCII case (`c:\work\x` is inside `C:\Work`) and matches a verbatim
prefix with its ordinary form (`\\?\C:\` with `C:\`, `\\?\UNC\server\share` with
`\\server\share`), also in a directory with case sensitivity turned on. Elsewhere the
comparison is exact, so on a macOS file system that ignores case, a path spelled in
another case is judged outside. The check isn't a sandbox. Symbolic links aren't
followed, so a link inside the workspace can lead outside it, and a path spelled through
a link is compared as written. On macOS, for example, `/var` links to `/private/var`,
where temporary directories live. Give the working directory in canonical form
(`std::fs::canonicalize`, as the README's example does), since programs such as `pwd` report
canonical paths.

### Tool errors

`error_recovery_config.on_tool_execution_error` handles a tool that returns an error:

- `InformAgentAndContinue` (the default) reports the error as that call's result (a
  native error result, or a text result in output-based mode) and emits
  `ErrorEncountered`. The other calls of the response run as usual.
- `PauseAndAwaitApproval` (opt-in) reports each error as its call's result too, lets the
  other calls of the response run first, including calls that wait for approval, and
  then pauses once, whatever number of calls failed, for `ApproveErrorRecovery` or
  `RejectErrorRecovery`. The pause's `ErrorEncountered` lists every failure and names the
  call when one failed. A rejected recovery fails the task with the turn's results
  recorded.
- `FailTask` (opt-in) fails the task at the first error. The calls of the response it
  did not run get a result saying so ("The task failed on an earlier tool error, so this
  call was not run."), so every call keeps its result.

### Custom tools

Implement `Tool` (with the `async-trait` and `serde_json` crates) and register it:

```rust
use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolRegistry, ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};

#[derive(Debug)]
struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "word_count".into(),
            description: "Counts the words in a text.".into(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: "text".into(),
                description: "The text to count.".into(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let text = params.get("text").and_then(Value::as_str).unwrap_or_default();
        Ok(text.split_whitespace().count().to_string())
    }
}

let mut registry = ToolRegistry::new();
registry.register(Arc::new(WordCount));
// Then add "word_count" to the agent's available_tool_names.
```

The runtime does not check arguments against the definition before calling `execute`.
`ToolDefinition::validate_params` is available for tools that want the check. Set
`parallel_safe` only for tools whose calls don't affect each other, such as reads (see
[Several calls in one response](#several-calls-in-one-response)). A tool can opt out of
the agent's time limit with `uses_host_timeout`, and out of being stopped mid-call by
`CancelTask` or `SuspendTask` with `interruptible`, when it winds its own work down (see
[Stopping tool calls](#stopping-tool-calls)). It
can opt out of result interception with `allow_result_interception`, and track what
reached the conversation with `reset_context` and `observe_context`. `bind_runtime` gives
a background tool a weak command sender for the task, through which it can send
`UserCommand::InjectContext`. `TaskContext::tool_call_id` is the runtime's id of the call
(as its events and the execution middleware see it), and `recovering` is set when the call
runs again after its task stopped while it ran. `recovery_mode` says whether it may (see
[Task snapshots and recovery](#task-snapshots-and-recovery)).

A tool can return images with `execute_with_output`, each with a `description`. In native
mode they go inside the call's own result, after its text, each after its description.
arko_ai sends them there where the model accepts images in tool results, and otherwise
after all of the turn's results, each after "Image from tool NAME, call ID: DESCRIPTION".
In output-based mode they follow the results' text the same way. Images aren't passed to
result interceptors and never appear in events. A host that shows them gets each call's
images from a `ToolImageSink` (`AgentRuntime::with_tool_image_sink`), keyed by the call's ID
in the events. `read_file` and MCP tools return PNG,
JPEG, GIF and WebP images this way too. See [tool images](reference-map.md#tool-images-arko_agents)
for size limits, embedded resources and Claude Code forwarding.

End-user tools wrap a command without Rust code. Each is a `ToolDefinition` whose
`tool_type_details` is `EndUserDefined { command_template }`, loaded with
`ToolRegistry::load_end_user_tool`. The template's `~` and environment variables are
expanded and it is split on whitespace, without interpreting quotes. String parameters
without a `flag` follow as positional arguments, in order, and then flagged string
parameters as `flag value` and each true boolean parameter as its flag. An `is_path`
parameter passes the path the workspace check judged, with `..` resolved so a symbolic
link can't redirect it. It is relative to the working directory inside it (`./src/main.rs`,
never a verbatim `\\?\` path on Windows) and absolute outside it, so the program can't
read it as an option. The command runs in the task's working directory without a shell,
with an empty standard input, and its standard output is the result. A nonzero exit
status is a tool error. The process
is killed when its call is dropped, for example when the task's runtime is aborted
mid-call. A boolean parameter takes a JSON boolean or the text `true` or `false`, so it
works in both tool calling modes.

A host that calls a tool itself (outside a runtime, or in a test) builds its context with
`TaskContext::new(task_id, agent_name, initial_task_prompt, working_directory)`, which
leaves the call fields unset and its stop signal one that never fires. Set the public
fields afterwards, or the stop signal with `with_stop`. `TaskContext` is `#[non_exhaustive]`,
so fields can be added without breaking callers. Outside this crate it can't be built with
a struct literal, so use `new`.

### Stopping tool calls

A long tool call can learn that its task is being stopped and wind its work down instead
of being dropped mid-work. `TaskContext::stop` is a `StopSignal` (`arko_agents::stop`) with
`is_stopping()`, `reason()`, and `stopping().await`, which resolves when it fires. It
fires when:

- the runtime receives `CancelTask` or `SuspendTask` while the call runs (reason `Task
  cancelled.` or `Task suspended.`). An interruptible call (the default) is dropped at that
  moment anyway, so this matters to a tool that isn't interruptible, which the runtime
  waits for.
- the host fires the signal it gave the runtime with `AgentRuntime::with_stop_signal`
  (`StopSource::new()`, `.signal()`, `.stop(reason)`). An arko_workflows graceful stop
  fires it at the start of its wrap-up, while its agents still run. Firing it changes
  nothing else, and the task goes on until the host cancels or suspends it.

A stop has no time limit of its own. The host bounds it (a workflow aborts an agent that
hasn't stopped within its grace period), and so does the tool's own grace. Make a tool's
stop safe to repeat, because a call that stops part way is settled on recovery like any other (see
[Task snapshots and recovery](#task-snapshots-and-recovery)).

`StopAwareCommand` runs a child process that stops cooperatively. The child gets
`ARKO_STOP_FILE`, the path of a file that exists once the task is stopping. A script checks
it between units of work, saves its state and exits. When the signal fires, the file is
created and the child has `grace` (10 seconds by default) to exit. Then, on Unix, its
process group gets `SIGTERM` and `terminate_grace` (5 seconds) more. Then it is killed.
Without a stop the child runs as long as it takes, since nothing is timed until a stop is
under way.

```rust
use arko_agents::{StopAwareCommand, TaskContext, ToolError};
use std::time::Duration;

async fn search(context: &TaskContext<'_>) -> Result<String, ToolError> {
    let mut command = tokio::process::Command::new("python3");
    command.arg("search.py").current_dir(context.current_working_directory);
    let output = StopAwareCommand::new(command)
        .grace(Duration::from_secs(20))
        .output(&context.stop)
        .await
        .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
    let stopped = match output.stopped {
        Some(how) => format!(" (stopped: {how:?})"),
        None => String::new(),
    };
    Ok(format!("{}{stopped}", String::from_utf8_lossy(&output.stdout)))
}
```

```python
import os
while work_left():
    do_one_unit()
    if os.path.exists(os.environ["ARKO_STOP_FILE"]):
        save_state()
        break
```

`output` captures stdout and stderr with a null stdin. `status` keeps the command's own
standard streams. `StopAwareOutput::stopped` says how a stop ended the child
(`ChildStop::Exited` within the grace, `Terminated` by `SIGTERM`, or `Killed`), and is
`None` when it ended without one. `stop_file(path)` names the file (for a child in a
sandbox that sees only some directories). Otherwise it is a fresh path in the system's
temporary directory, removed afterwards. On Unix the child leads a new process group, the
signals reach the processes it starts (unless one leaves the group with `setsid`), and
after a stop the group is killed once the child has ended. Elsewhere there is no `SIGTERM`
step, and only the child itself is terminated, so processes it started may outlive it.
Dropping the call (an interruptible call cancelled, or over its time limit) kills the child
at once, with its group on Unix. A tool that wants its child to have the grace when its
task is cancelled outright, not only in a host's graceful stop, returns `false` from
`interruptible`.

On Unix, being in its own process group, the child is out of the terminal's reach. The
terminal's Ctrl-C doesn't reach it (the host's stop does), and it can't read the terminal.
A read stops it (`SIGTTIN`), and the call waits until a stop kills it. Give a child that may
read its standard input a null one (`command_mut().stdin(Stdio::null())`) or a file when it
runs with `status`. `output` always does. A host that exits without unwinding
(`std::process::exit` runs no destructors, so no call kills its child) calls
`kill_stop_aware_commands()` first, which kills the process group of every
`StopAwareCommand` child running in the process. arko_workflows' `stop_on_signals` does at
its third signal.

### Result interception and execution middleware

The last two `run_task` arguments wrap tool execution:

- An `InterceptorConfig` holds a `ToolResultInterceptor`, which can replace a result
  before it enters the conversation (for example with a summary), and an
  `InterceptionFailurePolicy` for interceptor errors. `UseRawResult` (the default) and
  `RetryThenUseRawResult { max_attempts }` fall back to the original result, and
  `InformAgent` and `RetryThenInformAgent { max_attempts }` give the original result with
  a note telling the agent the interceptor failed. `ToolResult` carries the original
  output. `ToolResultIntercepted` follows with the replacement. Results of `complete_task`
  and `ask_user` are never intercepted, and `load_skill` results only when the agent's
  `SkillsConfig::load_skill_interception` opts in (see [Skills](#skills)).
  The event pair is emitted even when an approval supplies a message to the agent, so the
  audit stream still contains both the raw and effective result.
- A `ToolExecutionMiddleware` runs `before` and `after` every execution, whether
  approved or auto-approved, native or output-based. `before` returns `Continue`,
  `Reject { message_to_agent }`, `FailTask { reason }`, `CancelTask { reason }`,
  which ends the task as `Cancelled { reason }` (arko_workflows uses it when a run is
  cancelled), or `PauseTask { message_to_agent, reason }`. The rejected call gets
  `reason` as its result, and the calls of the response the stopped task did not run
  get a result saying so ("This call was not run: Workflow cancelled."). `PauseTask`
  leaves a decision to the host. The call doesn't run and gets `message_to_agent` as
  its result, and once the other calls of its batch have run the task awaits
  error-recovery approval with the cause `Middleware` and `reason` as its error
  (arko_workflows uses it for a completion its verifier could not review). A call
  answered with `custom_tool_output` skips `before`, and `after` receives
  `CallerProvided`. If a middleware hook returns an error, the runtime treats it as
  the same kind of host decision. A `before` error leaves the call unrun and pauses
  with the hook error. An `after` error keeps the tool's result and pauses after
  recording it, so continuing does not execute the tool again. In a batch, the other
  calls finish before the pause is reported.

### MCP servers

`McpManager::from_config` starts each server in an `McpConfig` as a child process and
discovers its tools, named `mcp__<server>__<tool>`:

```json
{
  "mcpServers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
      "env": {}
    }
  ]
}
```

An MCP tool is parallel-safe only when its server marks it `readOnlyHint: true`. Its
calls otherwise run one at a time. `register_tools` adds them to a registry and
`tool_names` returns the names to add to
`available_tool_names`. `shutdown` closes each connection and gives the server three
seconds to exit before killing it, and dropping the manager kills the servers. MCP tools
require native tool calling, and the workspace path policy cannot apply to them.

A server's stderr goes to the log (the `log` crate) at info level, a line at a time,
tagged with the server's name (`MCP server 'filesystem' stderr: ...`). It is not printed
to the host's stderr. A server that exits while the manager still uses it is reported
once, with a warning giving its exit status and last stderr lines. Its tools then fail
with a tool error, which the tool-error policy handles.

### Skills

`SkillManager::load(&skills_config, workspace_root)` reads the skills an agent's
`SkillsConfig` names from a `workspace_source` (relative to `workspace_root`) and any
`global_sources`, each with
optional `include` and `exclude` name lists. A skill is a directory with a `SKILL.md`
(other Markdown files in it, outside `scripts/`, become loadable resources), a `.skill` zip
bundle of the same layout, or a single Markdown file other than `README.md`. Its YAML
front matter sets `name`, `description`, and `auto_load`. Identical skills are loaded
once, and other name collisions get a numeric suffix. Invalid skills
are skipped with a warning unless `on_invalid_skill` is `Error`.

When there are on-demand skills, `SkillManager::setup`, given the tool registry and the
`AgentConfig`, registers the `load_skill` tool and adds it to the agent's tools with
auto-approval. Call it before `Task::new`, which keeps its own copy of the
configuration. It returns the `SkillPromptContext` to pass to `run_task`. Auto-loaded
skills are placed in the system prompt. The others are listed there by name and
description and loaded on demand.

A skill carries exact instructions, so `load_skill` results reach the agent as written,
and a result interceptor, such as a workflow's distiller, doesn't see them. The opt-in
`SkillsConfig::load_skill_interception: Some(policy)` lets it replace them, with `policy`
(an `InterceptionFailurePolicy`) deciding what a failure the interceptor reports does
(retry, fall back to the full result, or give the full result with a note). (When
arko_workflows chains several interceptors, the chain applies this policy to each
interceptor in it and reports no failures itself.)

### Interactive terminals

The opt-in `terminal` module provides PTY and pipe terminal tools that run commands in the
background and notify the agent of input requests and completion. See
[interactive terminals](interactive-terminals.md).

## Running tasks

`TaskState` is one of `Pending`, `InProgress`, `AwaitingToolApproval`,
`AwaitingBatchToolApproval`, `AwaitingUserResponse`, `AwaitingErrorRecoveryApproval`,
`Interrupted`, `Completed`, `Failed`, or `Cancelled { reason }`. `Completed`, `Failed`,
and `Cancelled` are terminal. `AwaitingErrorRecoveryApproval { error_message, cause }`
says why the task paused, as an `ErrorRecoveryCause`. It is `RetriesExhausted { retry_after_ms }`
(a request that may succeed later kept failing until recovery ended, and `retry_after_ms` is
the delay the provider asked for, or the wait the runtime had scheduled, when known), `NonRetryable` (the request, a
credential, or the configuration must change), `UnusableResponses` (responses the runtime
couldn't use, up to the attempt limit), `UsageLimit { resets_at_unix_ms }`, `Refusal`
(after the opt-in refusal fallback, if configured, also refused), `ToolError` (under
`PauseAndAwaitApproval`), `Monitor` (an opt-in monitor paused it), `Middleware` (the
execution middleware paused it with `PauseTask`), or `BudgetExceeded` (the host held its
next model request, see `TurnContextSource::hold_request` below). A task saved before
causes were recorded loads with `Unknown`. Both runtimes report every pause as the
`TaskStateChanged` that gives its cause, then `ErrorEncountered` with
`requires_approval_to_retry`. While the task waits at the pause, a command the runtime
can't apply (a `ResolveRefusal` whose changes are invalid, or any for a Claude Code agent)
is answered with another `ErrorEncountered` needing a decision, without a state change,
and the pause stands.

In `Failed { reason, cause }`, `cause` is the pause's cause when the task failed at an
error-recovery pause the host rejected (`RejectErrorRecovery`, as a workflow's error
recovery policy does for `NonRetryable`, `Refusal` and `UnusableResponses`), so a host can
tell, for example, an expired login or a bad key (`NonRetryable`, where starting the agent again
only helps once the credential or configuration is fixed) from other failures without
reading the reason. It is `None` for other failures and in a task saved by an older
version, and left out of the saved form when `None`.

`AwaitingErrorRecoveryApproval` and `Failed` also record `error_kind`, the
`arko_ai::ErrorKind` of the model-request error that ended recovery, when it was
classified, such as `Authentication` for a lost login, `Billing` for a credit wall,
`ContextOverflow` for a request over the model's input limit, and so on (see
[Error kinds](arko_ai-guide.md#error-kinds)). `ErrorEncountered` carries it too. A
rejected recovery keeps it with the failure. It is `None` for other pauses and errors,
unclassified ones, and in a task saved by an older version, and left out when `None`.

| Events | Meaning |
| --- | --- |
| `TaskStateChanged` | The task's state changed. |
| `AssistantResponseChunk`, `ReasoningChunk` | Response text and reasoning. `event_throttle_ms` coalesces chunks. |
| `ToolCallInitiated`, `ToolCallFormatted` | A tool call was announced, then parsed. Output-based mode with `simplified_tool_events` off also emits `ToolParamName`, `ToolParamValueChunk`, and `ToolParamEnd`. |
| `ToolCallAwaitingApproval`, `BatchToolCallAwaitingApproval`, `UserQuestionsAsked` | The task waits for a command. |
| `MessageToUser` | The agent sent the user a message and carries on, from a `message_user` call (`tool_call_id` names it) or, with `text_reply_as_message`, a reply in plain text (`tool_call_id` is `None`, and the text has already streamed as `AssistantResponseChunk`). See [Tools](#tools). |
| `ToolResultShortened` | A result was shortened before it went into the conversation (`reason` names the overflow guard, a cap, or overflow recovery). See [Tool result size](#tool-result-size). |
| `ToolResult`, `ToolResultIntercepted`, `ToolRejected`, `UnavailableToolAttempted` | Tool outcomes. `ToolRejected` also reports calls that didn't run (stopped, held, not reached, announced in an output-based turn that can't run them, or a `complete_task` refused after an earlier call of its turn was rejected or failed). |
| `ErrorEncountered`, `RetryScheduled`, `UsageLimitReached` | Errors and recovery. `requires_approval_to_retry` marks a pause. `error_kind` is a failed model request's `arko_ai::ErrorKind`, when classified. `RetryScheduled` carries the attempt, delay, and elapsed milliseconds, and `rate_limit` when the retry waits out a provider rate limit. |
| `OutputTokenLimitReached` | A response ended at its output-token limit, so its text may be cut short and a call still being written was dropped. The response is handled as usual. A host can tell the user, who can ask the agent to continue. |
| `RefusalReceived`, `RefusedInput`, `RefusalFallback` | Refusals. |
| `ModelRequestStarted { model, subagent, sizes, bytes_per_token }` | A model request is going out, reported just before each request (each attempt, a retry and a refusal fallback's included), or for a Claude Code model when its response starts. `sizes` is its `arko_ai::RequestSizes` (`None` for a Claude Code subagent's request, named by `subagent`), and `bytes_per_token` the ratio calibrated from the conversation's earlier requests (`calibrated_bytes_per_token`, `DEFAULT_BYTES_PER_TOKEN`, 3, before one was answered), which turns the sizes into tokens. A host times a prompt cache's lifetime from the latest one. |
| `StreamedOutputDiscarded { text_bytes, reasoning_bytes }` | The response the latest `ModelRequestStarted` (of the conversation's own requests) began failed after streaming part of it. A host drops the `AssistantResponseChunk` and `ReasoningChunk` text it showed since then (`text_bytes` and `reasoning_bytes` of it). It comes before the failure's `ErrorEncountered` and `RetryScheduled`, and a retry streams anew after its own `ModelRequestStarted`. A response that streams to its end is followed by its `UsageReported` instead. |
| `ImagesUnavailable { reasons }` | Images of the task's conversation are gone (their saved files were deleted or damaged), each now a line `(image no longer available: REASON)` in its place (`Task::settle_unavailable_images`). Reported as the run starts. |
| `UsageReported`, `ModelCallUsageReported`, `PlanUsageReported`, `ModelRerouted` | Usage after each model call. A Claude Code subagent's calls carry `ModelCallUsage::subagent`. |
| `CacheEfficiencyReported`, `CacheEfficiencyLow`, `PlanUsageThresholdReached` | Opt-in monitors. |
| `HandoverRequested`, `ConversationHandedOver` | A handover started (the agent was asked for its report), then the conversation was replaced. See [Conversation handover](#conversation-handover). |
| `ConversationCompacted { input_tokens, compacted_tokens }` | Claude Code compacted a Claude Code agent's conversation, with its size before and after. See [Claude Code models](#claude-code-models). |
| `ConversationCutBack { message_index, removed_messages }`, `Reconfigured { chat, tools }` | The host cut the conversation back (`Task::cut_back`) or continued the task under another configuration (`Task::reconfigure`). The runtime doesn't emit them. A host that changes a task this way reports them, as a workflow does before the run's own events. |
| `ContextInjected { ids, bytes }` | A `# Workflow context update` went into the conversation. `bytes` is its size, and `ids` names what it held, as the turn-context source identifies it (below). |
| `TaskStreamEnd` | The run ended. |

Besides answers to the waiting states, a running task accepts:

- `InjectContext { content }` is queued and added at the next turn boundary as a user
  message headed `# Workflow context update`. A stream in progress is not changed. Every
  injection sent before the boundary (including while a tool call ran, approved or not)
  joins that one message, in the order sent. A `TurnContextSource` given to
  `AgentRuntime::with_turn_context` is also asked for context at each such boundary,
  before each model request, and its answer joins that message. Its `hold_request`
  (by default never) can hold the request. The request isn't made, and the task pauses
  with `AwaitingErrorRecoveryApproval { cause: BudgetExceeded }` whatever its
  `llm_error_policy`. `ContinueAfterError` asks again, and approving adds no error note.
  A workflow holds agents over their budget this way. A Claude Code agent is asked before
  each turn and at each message of a turn, and a turn it holds is interrupted.
  Each update that goes into the conversation is reported with `ContextInjected`. Its
  `ids` come from the source. `context_update` returns a `TurnContext { content, ids }`
  (by default `turn_context` with no ids), and `identify` gives an id for each queued
  `InjectContext` (by default none). A workflow names notes `note:N`, subagent mail
  `mail:N` and bus messages `message:<uuid>`, so its host can tell what each run has
  read. `TurnContext::images` (each a `ContextImage { label, image }`) follow the
  update's text in the same message, each after its label, and wait with it in
  `Task::pending_context_images` when it can't go yet. During a turn, a Claude Code agent
  gets an update's text at once, in a hook (a hook's context is text), with a line saying
  its images come with the next input, and the images go with that input, at the turn's
  end, after `The images an earlier update said would follow.`
- `HostDeciding` says that, while the task awaits a command, the host is deciding that wait itself
  (an approval handler, or a wait before it continues after a usage limit), so
  `user_command_timeout_seconds` doesn't end it. The next wait is timed again. Ignored
  outside a wait. A host that decides every wait of a run gives the runtime a
  `HostDecisions` set to `true` instead (`AgentRuntime::with_host_decisions`), and then no
  wait of the run is timed. It is a run-level setting, not saved with the task, and a workflow sets
  it for the runs it drives.
- `RequestHandover` makes the agent hand its conversation over at the next turn boundary
  where a user message can go, whether or not `AgentConfig::handover` is set. See
  [Conversation handover](#conversation-handover). It doesn't end a wait, and is ignored
  while a handover is under way. An agent that compacts in Claude Code compacts instead.
- `CancelTask` ends the task as `Cancelled { reason }` with `CANCELLED_BY_USER` ("Task
  cancelled by user."). Tool calls running when it arrives are stopped, and calls it
  waited on or that had not run yet get a result saying they were not run. A task saved
  by an older version as `Failed { reason: "Task cancelled by user." }` loads as
  `Cancelled`.
- `SuspendTask` stops the runtime, which returns the task in its current state, for example
  to save it. A task suspended while it awaits a command, in any awaiting state, keeps
  its pending calls and results and waits for that command when it runs again. Tool calls
  running when it arrives are stopped, their results and those of the turn's calls not
  yet run say so ("Call it again if it is still needed."), and the task, in progress,
  makes its next request when it runs again.

A stopped tool call is dropped, which kills an end-user command it started. A call to a
tool that isn't interruptible (`Tool::interruptible` returns `false`) finishes first, and
the cancellation or suspension then takes effect. Its stop signal (`TaskContext::stop`)
fires so that it can wind down (see [Stopping tool calls](#stopping-tool-calls)).

`CancelTask` and `SuspendTask` stop the task without another model request, also when
queued behind context. Context the runtime has taken from the channel is never lost when a
task stops, for any reason. It is added to the conversation before the task is returned,
unless a user message can't go there yet, which is while the task awaits a command (a native
call's result must come right after the call), or while a refusal is unresolved (a pending
fallback answers exactly the refused input, and `Task::resolve_refusal` needs it to be the
last input). Then it is kept in `Task::pending_context_injections`, saved with the task,
and added at the first turn boundary where it can go once the task runs again, before
context sent later. A completed task also keeps the context taken during its last turn
there, since the model has not read it, and a continued task gets it.

`prepare_for_resumption_from_completed` applies the same `EachUserMessage` environment
details and reminder schedule as an ordinary user message. Pending context is delivered
once at the next safe turn boundary, including file-tree changes, without duplication.

### User images

A user's images go into the same message as the text they come with,
after it. `Task::initial_task_images` holds the task prompt's until the first message is
written (set them before the task first runs, the runtime then moves them into that message,
and `Task::initial_images` gives them either way), `prepare_for_resumption_from_completed_with_images` and
`prepare_for_resumption_with_user_response_and_images` take a follow-up's or an answer's,
and `TurnContext::images` a context update's (below). Both runtimes send them, and a
Claude Code agent receives them as image blocks of its input. A host fits a user's image to
the model first with `arko_ai::Model::sendable_image`, which shrinks one over the provider's
limits and says why one can't go (a workflow does this for the images it is given). The
images a tool returns are fitted when the runtime admits them (`ToolOutput::admit_images`),
and those past the number of images a request takes are left out, with a line saying so (see
[Tool images and Gemini 3.x](reference-map.md#tool-images-and-gemini-3x-arko_ai)).

### A new configuration

`Task::reconfigure(config)` continues a task under another
configuration, between turns (pending, completed, or awaiting the user's answer) and with
the same model. The next run offers `config`'s tools and writes a system message from it,
in place of the current one, and the conversation, its prompt cache key and the task's
usage stay. That is how a chat becomes an agent and back. The new tools and system message
are a new prefix, so the next request writes the prompt cache afresh. The task records it
(`Task::reconfigured`). A Claude Code task's session resumes with the new flags, and with
`--system-prompt-snapshot off` from then on (`Task::claude_code_prompt_changed`), since
Claude Code otherwise sends the prompt it recorded at the session's first request. A
handover's fresh session records the new one.

### Cutting back and branching

`Task::user_turns()` lists the messages of the current
conversation the host sent (`UserTurn { index, kind, text, images }`, `kind` one of
`Prompt`, `Message` and `ContextUpdate`. The runtime's own messages and tool results are
not among them, and `text` leaves out the environment details and reminder the runtime
added). `Task::cut_back(index)` cuts the conversation back to just before one of them, to
edit it or send it again. The later messages go, with what waited with them (calls and
results, context taken, a refusal or handover under way), the usage stays, and the task is
`Completed`, ready for a follow-up. Cut back to `0` it is `Pending` and runs from its prompt
again. `Task::branch(index)` is a copy cut back the same way (or whole, at the
conversation's length) as a new task with its own ID and no usage, and the original stays.
A Claude Code task records, for each message, its last entry in the session's transcript
(`Task::claude_code_entries`). A cut resumes the session at the entry before the cut
(`--resume-session-at`), sending the host's messages after it again, and a branch forks the
session into its own (`Task::claude_code_fork_of`, `--fork-session --session-id`). Claude
Code saves a cut, and creates a fork, only with the first new message, so the task keeps the
resume point until the session answers that message. A launch that fails first (the CLI
exits, or the host crashes) resumes at the same entry again, and a fork tries again under a
new ID. A cut that would leave the model's work without a known entry (a task recorded
before entries were, or before a compaction) is refused.

Tasks serialize to JSON (`Task::save_to_path`, `Task::load_from_path`). Running a
returned task again continues it. A task that is pending, in progress, or interrupted
makes its next model request, and a task awaiting a command waits for it.
`prepare_for_resumption_with_user_response` answers a saved `AwaitingUserResponse` task
before it runs, and `prepare_for_resumption_from_completed` adds a follow-up prompt to a
completed one. A wait that ends in `Interrupted` (the opt-in command timeout or a closed
command channel) does not keep the awaiting state. Calls still awaiting approval are
answered with the reason they did not run, which `ToolRejected` also carries ("Human
approval timed out after 60 seconds, so this call was not run. You can call it again if
it is still needed.", or "The approval wait ended because the host stopped responding,
so this call was not run."), and the turn's results are added to the conversation with
them, so every call keeps its result and the task can run again. A pause on a tool
error that waited for those decisions is dropped. `SuspendTask` keeps the awaiting state
instead.

In output-based mode, the results of the calls that ran in a turn that then waits, for
the approval of its other calls or on a pause, wait with the task, including when it is
suspended and saved (`Task::pending_text_tool_results`), and go to the model with the
outcome of the wait.

`Task::total_usage()` sums the usage of every model call. `Task::total_usage_cost()` is
what the account is billed. ChatGPT OAuth calls are covered by the subscription and add
zero, while their `UsageInfo::total_cost` keeps the list-price estimate.

## Task snapshots and recovery

A task is returned when its run ends. A host that must survive its own crash gives the
runtime a `TaskSnapshotSink` (`AgentRuntime::with_task_snapshots`), which receives the
running task at its safe points and saves it. arko_workflows does this for its agents.
After a crash, the saved task runs again from where it was saved.

The runtime reports the task after the model's turn is added (before its calls run),
before each group of tool calls starts, after each group's results are recorded, at every
turn boundary (which covers pauses and waits), after a handover, and when the run ends. A
safe point where nothing changed reports nothing. Before a group of calls starts, the calls
are recorded in `Task::running_tool_calls` and the report is *durable*.
`TaskSnapshotSink::task_changed(task, true)` returns only once the task is saved, and the
calls start afterwards. A group whose calls recovery simply runs again (every tool
`ToolRecovery::Rerun`, without the host's own output) is reported without waiting, and so
is a Claude Code agent's call to a read-only built-in
(`claude_code::READ_ONLY_BUILT_IN_TOOLS`, which are `Read`, `Glob`, `Grep`, `WebFetch`,
`WebSearch`, `TaskOutput`, `LSP` and `ToolSearch`), since a crash before the save only has
them run, or asked for, again. The other reports may be saved later, the latest one replacing any
not yet saved. A sink error is logged and the run goes on. Results are recorded in the
task as their calls end (`pending_native_tool_results`, `pending_text_tool_results`), so
a snapshot holds every result produced so far. The conversation the task ends up with is
the same with or without a sink. An approval decision puts the task back in progress
before the approved calls run, so a snapshot taken meanwhile no longer waits for it. A
task saved while it waited for an approval, an answer or a recovery decision waits for it
again.

When a task with running calls runs again, each is settled first, by its tool's
`Tool::recovery_mode`:

| `ToolRecovery` | The call |
| --- | --- |
| `InDoubt` (default) | Is not run again. Its result says the task was restarted while it ran, that it may have run partly or fully, and to check its effects before calling it again. |
| `Rerun` | Runs again as it was made, with `TaskContext::recovering` set. `read_file`, `list_files`, `load_skill`, `ask_user`, `message_user` (its message is sent again), `complete_task` (the host's completion checks run again) and MCP tools the server marks read-only. |
| `Resume` | Runs again with `recovering` set and its original `TaskContext::tool_call_id`, so the tool can find the work its first run started and return its result (a workflow's `run_workflow_agent` recovers the agent run it started). |

A call the host answered with its own output gives that output again, and a call to a tool
the run no longer offers is in doubt. The calls of the turn that never started are
answered as not run, a pause on the turn's tool errors still happens, and the results go
to the model as at the end of the turn (in output-based mode, with a note that calls
without a result were not run). A task saved by a version that did not record running
calls, with a native turn whose calls have no result, has them answered as not run
before any other message.

In a Claude Code session, a call is recorded before the gate allows it (or an approval
does), and its result as Claude Code reports it, until the model's next message shows
Claude Code has it. When the session resumes after a crash, Claude Code settles the calls
its transcript holds as interrupted without running them, and its transcript can lack
calls arko already ran, so the resumed session's first message lists each open call's
outcome as arko recorded it, as completed (with its result), run again now (an arko tool
that is `Rerun` or `Resume`) or unknown (the others, and for a built-in, commands it
started may still be running). The same note follows a Claude Code crash the host survives.

A host that runs a task inside another task's tool call records it in
`Task::parent` (`TaskParent { task_id, tool_call_id }`, absent in tasks saved before it
existed), so that a `Resume` tool can find the task its call started. The runtime doesn't
read it. A host that resumes the task in another call points it there. arko_workflows saves its agents this way and recovers them with
`Workflow::recover_agent`. See its [task journal](../arko_workflows/INDEX.md#task-journal-and-recovery).

## Model-call recovery

`llm_error_policy` decides what a failed model request does, and `llm_retry_config` sets
the timing:

| Policy | Behavior |
| --- | --- |
| `RetryThenPauseAndAwaitApproval { max_attempts }` | The default, with 10 attempts. Retries with backoff. When recovery ends, the task waits in `AwaitingErrorRecoveryApproval`. |
| `RetryThenInformAgent { max_attempts }` | Retries the same way. When recovery ends, the task is returned `Interrupted`. |
| `InformAgentAndContinue` | Retries failed requests without an attempt limit until the elapsed budget ends, and responses it can't use up to 10 times, then returns the task `Interrupted`. |
| `PauseAndAwaitApproval` | Waits in `AwaitingErrorRecoveryApproval` after the first failure. |

Recovery ends when the attempts or the elapsed budget run out or the error cannot be
retried. The elapsed budget, counted from the first failure of an episode, decides
whether another attempt is made and bounds the waits between attempts. It never cuts off
a request that has been sent. A request runs until it completes or fails, or the host
cancels or suspends the task, however long the model reasons. While a response the
provider accepted runs, the budget stops counting. It resumes if the response fails, so a
failure late in a long response is retried, while the waits between attempts still count
and failures that keep coming stay bounded. Attempt limits count the first request, and zero counts as one. With the
defaults, retries back off from about 2 seconds to the 64-second cap, so an outage is
retried for up to the 5-minute elapsed budget before the task pauses.

A failed request (a connection failure, an HTTP 408, 409 or 5xx response, a truncated
stream, including one cut by the opt-in stream idle timeout, or an error reported after a
stream started) is sent again unchanged, without a note in the conversation, since the model
had no part in it. A response the runtime can't use is retried with a note that tells
the model what to do. Such responses are a native response without a tool call (unless
`text_reply_as_message` makes it a message to the user, or `text_reply_ends_task` the
final reply, see [Tools](#tools)), an
output-based response without one, and output-based calls that can't be read. Such a retry isn't an outage. It
is made at once, without backoff, and bounded by the attempt limit (10 under
`InformAgentAndContinue`, which has none), not by the elapsed budget. Any completed
response ends the elapsed budget a failed request started. From a pause,
`ApproveErrorRecovery { message }` adds a note that the previous attempt resulted in the
error and that the user approved continuing (with the message) and starts a new recovery
episode, and `RejectErrorRecovery { message }` fails the task, with the message as its
reason when given. `ContinueAfterError { note }` is the
host's own decision to go on, for an unattended host. Without a note the failed request
is sent again unchanged (with any tool results the pause left waiting, which report their
errors), and a note is added as the runtime's text instead of the user-approval note. It
also starts a new episode. An interrupted task starts a new episode when it is run again.

Errors are classified by arko_ai's `Error::retry_class()`. HTTP 408, 409, 429, and 5xx
responses, connection failures and timeouts, truncated streams, errors reported after a
stream started without a permanent code, and responses without a tool call are retried.
Invalid requests, missing provider configuration, authentication failures, and permanent
provider codes (such as quota exhaustion and invalid refresh grants) end recovery at once.
The pause (or the `ErrorEncountered` of an interrupted task) records the error's
`Error::kind()` as `error_kind` when it is classified, so a host can tell a lost login
(`Authentication`) or a credit wall (`Billing`), which every request on the credential
meets, from a request one task got wrong (`InvalidRequest`, `ContextOverflow`). A request
over the model's input limit is first made smaller and sent again, at most twice. See
[Tool result size](#tool-result-size).

| `LLMRetryConfig` field | Default | Meaning |
| --- | --- | --- |
| `initial_delay_ms` | 2,000 | First backoff delay. |
| `max_delay_ms` | 64,000 | Cap of the exponential backoff. |
| `max_elapsed_ms` | 300,000 | Recovery budget from the first failure. It limits the waits between attempts and the decision to try again, never a request in flight. |
| `jitter` | `true` | Each backoff delay is drawn between half and all of its capped value. |
| `rate_limit_retry` | On | Waits for short rate limits and capacity errors. See below. |
| `capacity_retry` | `None` | Opt-in indefinite waits for capacity errors. |
| `usage_limit_policy` | `Pause` | Handling of plan usage limits. |
| `stream_idle_timeout_ms` | `None` | Opt-in. The longest gap between data on a streamed response before it is cut off. See below. |

For example, twenty attempts with indefinite capacity waits:

```rust
use arko_agents::{LLMCapacityRetryConfig, LLMErrorPolicy, LLMRetryConfig};

let builder = builder
    .llm_error_policy(LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 20 })
    .llm_retry_config(LLMRetryConfig {
        capacity_retry: Some(LLMCapacityRetryConfig::default()),
        ..LLMRetryConfig::default()
    });
```

Zero delays, a zero budget, and a `max_delay_ms` below `initial_delay_ms` are rejected.
A delay the provider asks for (`Retry-After`, rate-limit reset headers, or a delay in the
error body, as `Error::retry_after()` reports it) is a minimum, and jitter is added above
it so that agents sharing a reset do not retry together. It may exceed `max_delay_ms` but
must fit in the remaining budget. Otherwise recovery ends rather than retrying early. A
turn with valid tool calls ends the recovery episode. A token refresh or an injected
context message does not.

### Rate limits and capacity errors

Under either `RetryThen*` policy, short rate limits
(HTTP 429, `rate_limit_exceeded`, `slow_down`, and other providers' 429 codes) and,
unless `capacity_retry` is set, capacity errors (explicit overloaded or unavailable
signals) are retried after the provider's delay plus jitter, or with backoff from 1 s to
60 s when it gives none, until they succeed or the task is cancelled or suspended. These
retries resend the request unchanged, add nothing to the conversation, and neither use nor
reset the attempt limit and elapsed budget. The budget's clock stops while they wait. Rate
limits are waited out for at most `max_wait_ms` (15 minutes) in total per recovery
episode. After that, or when a rate limit asks for a longer wait, the error is handled as
a plan usage limit. A capacity error asking for a longer wait gets ordinary recovery.
`LLMRateLimitRetryConfig` sets the backoff (`initial_delay_ms`, `max_delay_ms`) and
`max_wait_ms`. `rate_limit_retry: None` (JSON `null`) gives both kinds of error ordinary
recovery.

### Indefinite capacity waits

`capacity_retry: Some(LLMCapacityRetryConfig::default())`
retries capacity errors under either `RetryThen*` policy until the service recovers, with
backoff from 30 seconds to 5 minutes and the same provider minimums and jitter, outside
the ordinary limits. A success ends the capacity episode. Another kind of error continues
ordinary recovery. `PauseAndAwaitApproval` still pauses at the first failure.

### Plan usage limits

A ChatGPT plan usage limit (`usage_limit_reached`) emits
`UsageLimitReached` with the reset time. The default `UsageLimitPolicy::Pause` ends
recovery under every error policy. The opt-in `WaitUntilReset { max_wait_ms }` waits,
under every error policy and cancellably, until the reported reset plus one second and,
with jitter, up to 30 seconds more, then resends the request unchanged. If no reset time
is reported, or the whole wait would not fit in `max_wait_ms`, recovery still ends.

### Stream idle timeout

Model requests have no timeout by default, so a response is
waited for as long as its connection stays open. This is deliberate. Models can reason
silently for a long time, GPT Pro models for over an hour, and a stream may carry nothing
until the reasoning ends. The opt-in `stream_idle_timeout_ms` (arko_ai's
`LLMClient::with_stream_idle_timeout`, applied to this agent's requests) cuts a streamed
response off once no data has arrived for that long. The cut is a truncated stream, which
recovery sends again, and the provider may bill the work that was cut off. A value
shorter than the longest silent stretch of the models you use cuts healthy responses off,
so make it generous (for example two hours, 7,200,000). Only the gaps between data count.
There is no total deadline. It is never sent to the model.

### Streams

Output a failed stream delivered is not kept, since the request is sent
again unchanged. A request whose conversation ends on assistant text, such as a task
resumed after `SuspendTask` mid-stream, first gets a user message asking the model to
continue, because some providers reject a conversation that ends on an assistant turn
(Claude 4.6 and later reject prefill). Native calls announced with `ToolCallInitiated` that never run (the
stream failed, was interrupted, or was refused, or the final response does not hold
them) get a `ToolRejected` event. Model recovery never runs a tool call again. A stream
that fails may have used provider tokens without reporting its usage.

### ChatGPT OAuth

Before each OpenAI request, the runtime refreshes ChatGPT OAuth
credentials that expire within five minutes. The refresh makes at most one token request,
under the same attempt and elapsed limits as model requests, and a failed refresh is
recovered like a failed request. A request rejected with HTTP 401 `token_expired` is
refreshed and resent once by arko_ai. With credentials from a Codex `auth.json`, build the
client with `LLMClient::with_codex_auth_file`, so that every refresh re-reads the file
first and rotated tokens are saved to it at once. See
[ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials).
The example CLI's `--openai-auth-json` does this.

`CancelTask`, `SuspendTask`, and closing the command channel interrupt recovery waits and
requests in flight.

## Monitoring

Every ChatGPT OAuth call reports the plan's usage windows (from the backend's `x-codex-*`
headers) in `ModelCallUsage::plan_usage` and a `PlanUsageReported` event. A call served by
another model than the one requested (an OpenRouter fallback, router, or alias, or a
ChatGPT backend reroute) emits `ModelRerouted` with both models.

`AgentConfig::monitoring` adds opt-in monitors:

- `plan_usage: Some(PlanUsageThreshold { percent, action })` is checked before each model
  request against the latest report. When the most-used window is at or above `percent`,
  `PlanUsageThresholdReached` is emitted once per window and reset, and
  `PlanUsageAction::Pause` pauses once, or `SlowDown { delay_ms }` waits before every
  request while usage stays that high. Context sent during the wait joins that request,
  in a second context update. The reported window is saved in
  `Task::plan_usage_threshold_reported`, so a resumed task does not report it again, and
  a report whose window has since reset is ignored.
- `cache: Some(CacheMonitorConfig::default())` emits `CacheEfficiencyReported` after each
  call, with cache reads divided by the most that could have been read (the smaller of the
  previous and the current input), for the call and over the last `window_calls` (5)
  calls. The first `warmup_calls` (2) calls, calls without a cache figure, and calls to
  the refusal fallback model are not judged. Once `window_calls` calls have been judged,
  a rolling efficiency below `warn_below` (0.5) emits `CacheEfficiencyLow` once until it
  recovers. The opt-in `pause_below` also pauses before the next request.

A monitor pause waits in `AwaitingErrorRecoveryApproval` under the pausing error policies
and returns the task `Interrupted` under the others. Approving it continues without an
error note. A message given with the approval is added as a user message.

### Choosing a plan-usage action

Neither action saves usage, because a task's requests count the
same against the plan whenever they're made. The choice is whether the task keeps going as the limit
nears, and it turns on two times, when the window resets (`PlanUsageThresholdReached`
carries it) and how long the prompt cache lasts. A request made after the cache has
expired reads the whole context again as uncached input, which uses more of the plan than
cached input, more than any wait saves. `arko_ai::prompt_cache::prompt_cache_lifetime` (or `LLMClient::prompt_cache_lifetime`)
gives a model's lifetime, which is 5 minutes for Anthropic by default (1 hour with
`AnthropicCacheTtl::OneHour`), a conservative 5 minutes for OpenAI with an API key, and
30 minutes over ChatGPT OAuth. For a Claude Code
model it is `anthropic_cache_ttl` when set, otherwise Claude Code's choice, 1 hour on a
Claude subscription within its usage limits and 5 minutes on an API key. Claude Code's
subagents (its `Agent` tool) and helper requests cache for 5 minutes whatever the main
conversation's TTL, so a stop of more than about 4½ minutes loses their caches.

### Keeping a cache warm

`AgentRuntime::refresh_prompt_cache(task, registry,
skill_prompt_context)` keeps a waiting task's cache from expiring. It sends the task's next
request (system prompt, tools, conversation) with a short ping after it, the output capped
where the provider allows (one token on Anthropic, OpenAI's minimum of 16, none on the
ChatGPT backend, and the ping asks for one word), discards the reply, and returns
`CacheRefresh` with the request's usage and output limit. A task waiting with calls that have
no results yet (paused for an approval) can't take a message after them, so its last request
goes again unchanged (`resent_last_request`). A provider that refuses the limit gets the
request once more with 1,024, and `refused_cap` holds its error. Thinking settings stay as the
task's requests have them, so the refresh reads the cached conversation. A Claude Code task
pings a fork of its session that Claude Code doesn't save
(`--fork-session --no-session-persistence`). Nothing is added to the task. Gemini, OpenRouter
models other than Anthropic's, Anthropic's on OpenRouter with reasoning on (OpenRouter derives
the thinking budget from the output limit), a manual Anthropic thinking budget, and a task
that has made no request with its current prompt return `arko_ai::Error::UnsupportedFeature`.
Opt-in live tests check each route (`cargo test -p arko_agents keep_warm_live -- --ignored`,
with the route's API key set).

| Situation | Action | Cache effect |
| --- | --- | --- |
| The reset is far off | `Pause`, so the rest of the window isn't spent before someone decides | The request that resumes after the cache expired reads the context uncached, once |
| The reset is close | Keep going, with `UsageLimitPolicy::WaitUntilReset` to wait out the limit if it's reached first | One uncached read once a long wait ends |
| The reset is close and the task shouldn't stop | `SlowDown { delay_ms }`, to stretch what is left until the reset | None while `delay_ms` stays well under the cache lifetime |

`SlowDown` spends the same usage over more time, so it only helps when the reset comes
before the rest of the window runs out. A delay that reaches the cache lifetime makes every
request after it miss the cache, which uses the window up faster than no delay at all. The
runtime logs a warning, once per run, for a delay within 30 seconds of the lifetime (for a
Claude Code agent offered the `Agent` or `Task` tool, of its subagents' 5 minutes).

## Claude Code models

An agent whose model is a Claude Code model runs in a session of the user's installed
Claude Code CLI, under its login, instead of through `LLMClient` requests (see
[Claude Code](arko_ai-guide.md#claude-code) in the arko_ai guide). Claude Code runs the agent loop. The runtime drives the session
and keeps the task, its events, commands, approvals and handovers as for any agent.
Nothing else changes for a host, because `AgentRuntime::run_task` picks the Claude Code runtime
from the model's provider. The reply and the model's thinking stream as Claude Code streams
them (`AssistantResponseChunk` and `ReasoningChunk` from its partial messages, a few tokens at
a time, while a subagent's stay its own), and Claude Code's retries of a failed request are
reported as the runtime reports its own, with `ErrorEncountered` needing no decision, then
`RetryScheduled` with the attempt, Claude Code's delay and whether it waits out a rate limit.

```rust
use arko_agents::{ClaudeCodeSettings, ClaudeCodeTools};

let config = AgentConfig::builder(id, name, AgentLLMConfig {
    model: Model::claude_code(Model::anthropic_claude_opus_5_5()),
    generation_config: GenerationConfig::default()
        .with_reasoning(ReasoningConfig::AnthropicAdaptive {
            effort: Some(AnthropicReasoningEffort::High),
        }),
})
// ...
.tool_config(AgentToolConfig {
    // Claude Code's own tools, next to the agent's arko tools.
    available_tool_names: ClaudeCodeTools::editing(),
    ..AgentToolConfig::default()
})
.claude_code(ClaudeCodeSettings::default())
.build()?;
```

### Tools

The agent's arko tools (`complete_task`, `ask_user`, `message_user`, a
workflow's `send_message`, custom tools) are served to Claude Code in-process as the MCP
server `arko`, so the model sees them as `mcp__arko__<name>` (the system prompt says so).
They run as for any agent, with the execution middleware, result interception and time
limits. With `text_reply_as_message` on, a turn that ends without `complete_task` sends
its final text as `MessageToUser` before the runtime's note. With `text_reply_ends_task`
on, a turn that ends with text completes the task with it, and `complete_task` isn't
served. The turns Claude Code is told to end (after `ask_user`, for a handover report or a
`/compact`) and the turns arko interrupts aren't replies. Claude Code's built-in tools (`claude_code::BUILT_IN_TOOLS`, such as `Read`,
`Edit`, `Bash`, `Grep` and `WebFetch`) are offered when `available_tool_names` lists them, and
only then. `ClaudeCodeTools::{read_only, editing, shell, web, all}` list common sets. A
name that isn't an arko tool is offered as a built-in, so tools a newer CLI adds can be
listed too.

### The gate

Every call, built-in or arko, first passes a `PreToolUse` hook the runtime
answers:

| Setting | Applies to built-ins | Notes |
| --- | --- | --- |
| Offered tools | Yes | A tool not listed is refused with a result saying it's not available |
| `auto_approve_tool_use` and approvals | Yes | A call that needs approval goes to a permission request that waits for `ApproveToolUse` or `RejectToolUse` (`ToolCallAwaitingApproval`, `AwaitingToolApproval`), with `user_command_timeout_seconds` and `HostDecisions` as usual. `custom_tool_output` becomes the call's result instead of running it, and `message_to_agent` comes with the result |
| Workspace path policy | Yes | On the path inputs of `Read`, `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, `Glob` and `Grep` (`claude_code::built_in_path_parameters`) |
| Execution middleware | Yes | `before` in the gate (a rejection, pause, failure or cancellation applies as for any call) and `after` once the call ran |
| Terminal command guard | `Bash`, `PowerShell` | The command must pass [`CommandGuard`](interactive-terminals.md) for the working directory, where each `Bash` call starts (arko starts the CLI with `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1` unless `ClaudeCodeSettings::env` sets it). `ClaudeCodeSettings::{allow_commands_outside_workspace, allowed_command_paths, dangerously_allow_all_commands}` are its opt-ins, and `ClaudeCodeSettings::command_rules` its [command rules](interactive-terminals.md#command-rules) (none by default), such as read-only git |
| Result interception | No | Claude Code produces a built-in's output. A workflow rejects a distiller aimed at one |
| `tool_timeout_seconds`, `parallel_safe` | No | Claude Code runs its built-ins. `validate` rejects these settings for them |
| `ToolErrorPolicy` | Arko tools only | A built-in's failure (a command that exits non-zero, say) is its result |

`AskUserQuestion`, `ExitPlanMode` and `EnterPlanMode` can't be offered (they would bypass
the host, so ask with `ask_user`). Claude Code shows a refused call to the model as
`PreToolUse:<tool> hook error: <reason>`, so the gate's reasons say who refused it and why
(`arko did not run this Read call: …`). A call held for the handover report says it isn't
an error.

### Sessions

`ClaudeCodeSettings` sets the CLI (`cli_path`, and `cli_args` for one run
through an interpreter), `env`, whether the agent's system prompt replaces Claude Code's
(`replace_system_prompt`, appended by default), and the user's settings files the
session loads (`setting_sources`, none by default, with `--strict-mcp-config`, so the
agent behaves the same on any machine). The reasoning effort of an `Anthropic` or
`AnthropicAdaptive` reasoning config becomes `--effort`, `anthropic_fast_mode` Claude
Code's fast mode, and `anthropic_cache_ttl` the main conversation's prompt-cache TTL
(Claude Code's `promptCacheTtl` setting). Without it, Claude Code caches for 1 hour on a
Claude subscription within its usage limits and 5 minutes on an API key. Its subagents
(the `Agent` tool) and helper requests cache for 5 minutes regardless, so any stop longer
than about 4½ minutes (an `ask_user` answer, an approval, an error-recovery decision, a
`SlowDown` hold, a paused workflow) loses the subagents' caches, even when the main
conversation's lasts an hour. 1-hour cache writes are recorded
(`UsageInfo::cache_write_1h_tokens`) and counted at the 1-hour write rate in the list-price
estimate. Under a subscription nothing is billed, and they count against the plan's
usage. When the CLI is billed per token instead (an API key, cloud credits, a host-managed
provider), set `ClaudeCodeSettings::metered_billing`, so each call's list-price estimate is
its billed cost and counts against workflow budgets. The session lives while the task is in progress. A wait for an
answer (`ask_user`) or an error-recovery decision closes it, and the task resumes it
(`Task::claude_code_session`) when it continues. A turn that ends without `complete_task`
gets a note asking the model to continue or complete. After 10 in a row the task is
interrupted. If the CLI exits, or a turn ends in an error Claude Code's own retries
didn't recover from, the error policy decides. The pausing policies pause (`RetriesExhausted`, or
`NonRetryable` when the CLI can't start), and the others return `Interrupted`. `ContinueAfterError` and `ApproveErrorRecovery` resume the session.

### Context and usage

`InjectContext` and the turn-context source's context reach the
model with the next tool result (`PostToolUse`), or with the next message. The task's
conversation records them after that result, as a `# Workflow context update`. The
conversation mirrors the session's (the task's messages, the model's text and calls, and
their results) as a record for hosts. Subagents' messages aren't included. Usage is
recorded per request from Claude Code's stream events (its final `message_delta`), with the
whole input, adding Anthropic's exclusive uncached, cache-read and cache-write counts, and
the output with thinking. It is billed as a subscription (nothing per call, and the cost is
the estimate at list prices).

### Monitors and limits

Claude Code's `rate_limit_event` is reported as
`PlanUsageReported` (its five-hour window as the primary one, its seven-day window as the
secondary) and attached to the requests after it, and the plan-usage threshold applies.
`Pause` interrupts the turn and pauses as a monitor. `SlowDown` holds each tool call (and
so the request after it) for its delay while usage stays over the threshold, and a
cancellation or suspension ends the hold at once. A turn that fails while Claude Code
reports the plan limit reached pauses with `ErrorRecoveryCause::UsageLimit` and its reset
(`UsageLimitReached`), so a workflow's automatic recovery waits for it. Under
`UsageLimitPolicy::WaitUntilReset` the runtime waits until the reset itself (as for other
agents, when the wait fits `max_wait_ms`) and resumes the session. The cache monitor judges the main conversation's
requests.

### Errors

A session that can't start (the CLI is missing or exits at once) pauses with
`ErrorRecoveryCause::NonRetryable`, and so does a turn that ends in a failure retrying
doesn't fix, by the error's kind (`arko_ai::ErrorKind::from_claude_code`):
- Claude Code can't authenticate (`Authentication`, when it isn't signed in or its login
  expired or was revoked). Claude Code starts without a login and reports it on the first turn,
  as an API error message whose `error` is `authentication_failed` (or a result whose
  `api_error_status` is 401).
- Its account can't be billed (`Billing`, from `billing_error`, `account_on_hold` or HTTP 402).
- Its organization or plan may not make the request (`Permission`, from
  `oauth_org_not_allowed` or HTTP 403).

The runtime reads those fields, and the pause records the kind. A session that fails otherwise (the CLI exits, or a turn
ends in an error Claude Code's own retries didn't recover from) pauses with
`RetriesExhausted` (`UsageLimit` at a plan limit), under the pausing error policies.
Every pause, also one taken after a turn (a tool error under `PauseAndAwaitApproval`, a
completion a verifier couldn't review, the execution middleware's `PauseTask`), is
reported as in arko's own runtime, with the pause's state change and then the
`ErrorEncountered` a host answers. A workflow's error recovery policy decides it by its
cause. Under `ErrorRecoveryPolicy::Automatic` a session that can't start, sign in, bill
its account or make the request (or a standing refusal) fails the run, with the cause and
kind in `TaskState::Failed`, and a failed session is resumed after the policy's wait.

### Refusals

By default Claude Code retries a refused request on a fallback model it
chooses. The runtime reports it as `RefusalReceived` then `RefusalFallback`, and a model
switch for availability as `ModelRerouted`. Claude Code keeps the rest of that session on
the fallback model (for example from Opus 5.5 to Opus 4.8 after a `cyber` classifier refusal).
Each call's `ModelCallUsage` names the model that served it, and the list-price estimate
uses that model's prices. A handover starts a new session, on the configured model. The
refused call's `ModelCallUsage` has `stop_reason` `"refusal"` and the API's `stop_details`,
and `RefusalReceived` takes its explanation from them when Claude Code's message leaves
it out. A refusal that stands (no retry ran) is
`RefusalReceived`. With `refusal_fallback` (a Claude Code model), the session switches to
that model (`set_model`) for one turn, asks it to continue, and switches back, and a
refusal on that turn stands. Without it, the task pauses with `ErrorRecoveryCause::Refusal`
under the pausing policies. The refused request stays in Claude Code's conversation, so
`ResolveRefusal` doesn't apply. It is answered with an `ErrorEncountered` needing a
decision, and the task keeps waiting.

### Handovers

With `handover` set, the threshold is checked against each request of the
main conversation as Claude Code reports it. Once reached, the model's next tool call is
refused with the report request as its result (or, if its turn ends first, the request is
sent as a message). The report is the turn's text, and a fresh Claude Code session starts
from the seed and the report. Claude Code's own auto-compaction is off for such an
agent (`autoCompactEnabled: false`), so it never replaces the conversation first. A turn
that goes over the model's input limit hands over without its last step, as in arko's own
runtime. The session resumes at the message that ended the last request Claude Code
answered (the CLI's `--resume-session-at`), and the report request there starts with
`STEP_LEFT_OUT`. This happens once per handover, so a report request that goes over the
limit as well ends the run on the error.

### Compaction

With `HandoverConfig::compact_in_claude_code`, the agent compacts instead
of handing over, at the same threshold (and on `RequestHandover`). Once the threshold is
reached, the model's next tool call is held and its turn interrupted, and the runtime
then sends Claude Code's own `/compact` into the session. The agent goes on in the same
session from Claude Code's summary, with a message that names the calls held and asks it
to continue. A conversation whose last turn ended over the threshold compacts as its next
session starts, before the new input, and so does a request sent while no session runs
(commands sent then are handled as the next session starts). A turn that fails over the
model's input limit compacts at once. When Claude Code doesn't compact (no
`compact_boundary` before the `/compact` turn ends), the agent hands over instead, and
leaves out its last step as above when the report request goes over the limit too. Each
compaction, the runtime's or Claude Code's own, is reported as `ConversationCompacted`
with the size before and after, and the task's conversation keeps every message. Tools
observing the conversation start over, the cache monitor judges the compacted
conversation alone, and `HandoverHook::on_compaction` can give the agent context again
(arko_workflows gives it the standing notes), sent with its next input.

`examples/claude_code_agent.rs` runs an agent with Claude Code's read-only tools and a
handover threshold, printing its calls, usage and handovers (with `--compact`, Claude Code
compacts at the threshold instead). It uses the CLI's login and counts against its plan
usage. The tests run a scripted fake CLI
(`tests/fixtures/fake_claude_agent.py`) and need Python 3 (`ARKO_TEST_PYTHON` selects the
interpreter).

## Conversation handover

A long task's conversation grows until its cost and quality suffer. A handover replaces
the conversation, in the same task, with a fresh one that starts from the task and a
report the agent writes. The task keeps its ID, state, budget and usage history, so a
host that addresses the agent by name or task ID sees no restart.

```rust
use arko_agents::HandoverConfig;

let config = AgentConfig::builder(id, name, llm_config)
    // ...
    // Hand over at 40% of the model's input limit, or 200K tokens, whichever is lower.
    .handover(HandoverConfig {
        at_input_tokens: Some(200_000),
        at_context_fraction: Some(0.4),
        report_request: None,
    })
    .build()?;
```

### The request

Before a model request, once that request is estimated to have at
   least the threshold's input tokens (the lower of `at_input_tokens` and
   `at_context_fraction` of the model's input limit, and the fraction is ignored for a model
   whose limit is unknown), or after a host sent
   `UserCommand::RequestHandover`, the runtime adds a user message asking for the report
   (`DEFAULT_REPORT_REQUEST` for an agent that gives it with the `handover_report` tool,
   `DEFAULT_TEXT_REPORT_REQUEST` for one that replies with it, or `report_request` for
   either) and emits `HandoverRequested`. It waits
   for a boundary where a user message can go, so not while calls wait for results, and not
   for a refusal fallback's request. A conversation whose first request was already over
   the threshold doesn't hand over (the fresh one would start over it too). A warning is
   logged once. The estimate is the latest request's reported input plus what the
   conversation gained since, the reply (at least its reported output tokens) and the
   turn's tool results, at the bytes-per-token ratio calibrated from the task's own
   requests (the [overflow guard](#tool-result-size)'s calibrated estimate, not the
   10-bytes-per-token lower bound it decides by). So a turn whose results take the
   conversation over the threshold hands over before the request that would carry them,
   not one request later. `HandoverRequested` and the record keep the latest request's
   reported input. The Claude Code runtime checks the latest request's input, as Claude
   Code reports it.
### The report turn

With native tool calling, an agent with `handover` set is
   offered the `handover_report` tool from its first request and gives the report as the
   tool's `report` argument. An agent told that every response must call a tool can keep
   to that while it writes the report. Other agents (output-based tool calling, Claude
   Code, and a host's request to an agent without `handover`) reply with the report as
   plain text. The request asks for the task, what was done and the evidence, decisions,
   where it stopped, the next steps, what it is still waiting for and from whom (such as
   a reply to a question it sent), and every message it hasn't fully answered, with what
   it meant to reply. Context that arrives from the moment the report is requested (injected
   context, the turn-context source's, file-tree changes) is held in
   `Task::pending_context_injections` and added to the fresh conversation, not the one
   being replaced, so a message that needs a reply is read in full after the handover.
   Other tool calls in this turn are rejected with a result saying the report is due,
   except `complete_task`, so a task that is done completes instead. A reply without the
   report gets a note, and the request stands. For an agent offered the tool that is a
   reply without a tool call, and for the others a reply without text. A `handover_report` call
   made when no report is due, or without a report, is rejected. The tools offered don't
   change, so the request keeps its cache prefix. A report given with the tool waits in
   `HandoverRequest::report` until the next turn boundary, once the turn's results are in,
   so a task stopped in between still hands over when it runs again.
### The fresh conversation

`Task::start_fresh_conversation` clears the messages (and
   what refers to them, such as runtime-message indices, a refused input, the ChatGPT turn
   state and the user-message count) and keeps the system prompt, the pending context injections,
   and the task's usage. Its first message is the seed's text, framed as a task prompt
   (with the initial environment details and reminder), by default the task's
   `initial_task_prompt`, then `# Handover report`, a sentence saying the conversation
   was replaced, and the report, followed by the task's images (`Task::initial_images`). The
   runtime then resets what it tracks about the old
   conversation. Tools observing the conversation start over
   (`Tool::reset_context`), the file tree is sent in full again, and the cache monitor
   judges only the fresh conversation's calls (`Task::conversation_usage_start`). The
   first call of the fresh conversation has no input delta. `ConversationHandedOver`
   carries the report, and `Task::handovers` records each handover
   (`HandoverRecord { at, input_tokens, replaced_messages, report }`).

`Task::initial_task_prompt` is not changed. A task saved during the report turn keeps
`Task::handover_requested` and hands over when it runs again. Continuing a completed task
clears it.

### Chats

A chat (`AgentToolConfig::chat`, see [Tool calling modes](#tool-calling-modes))
hands over without a tool, so its requests stay without tools. It is never offered
`handover_report`, even with `handover` set, and replies with the report in text. The
report request is `DEFAULT_CHAT_REPORT_REQUEST` (or `report_request`, read through
`HandoverConfig::chat_report_request`), which asks for a summary of the conversation that
leaves the user's latest message unanswered. The fresh conversation starts from that message
instead of the task's first one. `HandoverSeed::for_chat` restates the user's messages since the
model's last reply (`handover::unanswered_user_messages`), followed by the summary, and
their images go into the same first message. The model then answers that message from the
summary, and its reply ends the task as usual. The workflow engine's hook does the same for
these agents. A host turns it on as for any agent, with `handover` and a threshold, a
`HandoverThreshold`, or `UserCommand::RequestHandover`. A chat on a Claude Code model writes
its summary at the start of its next run, before the host's new message, because its turns end
with a reply and no tool call, so the report request can't reach it during one. The new message
waits for the fresh session, which answers it.

### Hosts

`AgentRuntime::with_handover_hook` takes a `HandoverHook`, whose
`on_handover(task, report)` is called while the task still holds the old conversation and
returns the `HandoverSeed { task_prompt, context }` that starts the fresh one
(`HandoverSeed::for_task` is the default). A host that tracks what a conversation holds
(messages delivered into it, files it read) updates its records there, and can restate
the task without material the report covers and add context after the report. Its
`on_compaction(task)` is called after Claude Code compacted a Claude Code agent's
conversation, and returns context the agent gets again with its next input. The
arko_workflows engine does both for every agent it runs.

`AgentRuntime::with_handover_threshold` takes a `HandoverThreshold`, a threshold the host
keeps a clone of and may change while the task runs, larger or smaller. Set, it replaces
the configured threshold from the next check on, and `set(None)` goes back to it. The
task, its configuration and its conversation stay as they are, so the prompt cache does
too, and the threshold isn't saved with the task. The host keeps it under the model's
input limit, with room for the report.

## Refusals

A refusal is never retried unchanged. The runtime emits `RefusalReceived` with the
`arko_ai::Refusal`, then `RefusedInput`, and handles the refusal as an error that can't
be retried, so the task pauses or is returned `Interrupted`, as the error policy decides.

A model's own refusal (`RefusalSource::Model`, such as an OpenAI structured refusal) stays
in history, so the model can see what it declined. Its tool calls are not run (announced
calls get `ToolRejected` with the reason `turn refused`) and are not kept. In native mode
the refusal explanation becomes the turn's text when the turn has none. In output-based
mode the explanation replaces the text, or without one, the text is kept without its
tool-call markup. A turn left without text is not kept. Classifier and content-filter
refusals are removed from history.

### Changing the refused input

`RefusedInput` lists the user messages after the last
assistant turn, each a `RefusedMessage` with its `message_index`, `runtime` and
`runtime_text` flags, `tool_results` (by provider call ID), and image count (images inside
its tool results included). It is also
kept in `Task::refused_input` until the refusal is resolved, recovery is approved, or a
later turn completes. Messages the runtime added (file-tree updates, error and recovery
notes, requests to continue) are marked `runtime` and cannot be changed, and neither can
a recovery note beside native tool results (`runtime_text`). While the task awaits
recovery approval, `UserCommand::ResolveRefusal { changes }` applies a list of changes.
For a task returned interrupted, call `Task::resolve_refusal(changes)` before running it
again. The changes apply together or not at all. An invalid one emits `ErrorEncountered`,
and the task keeps waiting.

| `RefusalResolution` | Effect |
| --- | --- |
| `RephraseUserMessage { message_index, text }` | Replaces the message's text and keeps the sections the runtime appended to it (`# Initial Environment Information`, `# Current Environment Information (updated this turn)`, `# Reminder`). In a message with tool results, it replaces the text beside them, such as a native `ask_user` answer. The text must not be empty. Rephrasing the task prompt applies its `output_based_only` markers and updates `Task::initial_task_prompt`. |
| `RemoveUserMessage { message_index }` | Removes a message without tool results. At least one user message not added by the runtime must remain. |
| `RemoveImages { message_index }` | Removes a message's images, such as tool screenshots, both those inside its tool results (which keep their text) and its own, with the descriptions the runtime added before them. |
| `RedactToolResults { call_ids, replacement }` | Replaces tool outputs' text. Each call stays paired with its result. Images inside the results stay. Remove them with `RemoveImages`. |

The refused input must still be the conversation's last input, followed at most by the
kept refused turn. The refused turn is removed, the changed input is sent as it is, and
tools that track the conversation observe it from the start again. Change a task's input
this way rather than with arko_ai's `Conversation` methods (`rephrase_last_user_turn`,
`remove_last_user_turn`, `redact_tool_result`), which do not update the task's record of
runtime messages or its refused input. `ApproveErrorRecovery` continues without a change.
After a model's own refusal, which stays in history, the next input is then usually only
the runtime's recovery note, which cannot be changed if the model refuses again.

### Fallback model

The opt-in `AgentConfig::refusal_fallback` is an `AgentLLMConfig` on
the configured model's provider. A fallback on another provider is rejected, since
providers cannot replay each other's turns. A refusal from the configured model is
reported (with its `RefusedInput`), and the same input goes once to the fallback model
(`RefusalFallback`). The refused turn is not kept. The fallback's response continues the
task, and later turns use the configured model again. The pending fallback is saved with
the task (`Task::fallback_for`), survives a monitor pause, and holds back context updates
(saved with the task too) and file-tree changes until it has answered. If the fallback
also refuses, the refusal stands and can be resolved as above. A fallback request that
fails otherwise is recovered from like any request, by its own error. A dropped
connection, server error or rate limit is retried to the fallback model, and when
recovery ends the task pauses with that error's cause (`RetriesExhausted`,
`NonRetryable`, ...) with the fallback still pending, so continuing
(`ContinueAfterError` or `ApproveErrorRecovery`) sends its request again, unchanged. No
error note is added, and a note or message given with the command waits as context
until the fallback has answered. Resolving the refusal instead sends the changed input
to the configured model.

Across the two models, the ChatGPT turn-state token of one is not sent with the other's
requests. OpenAI reasoning is replayed between them only when both are `Provider::OpenAI`
models of one Codex compaction family (the GPT-5.6 and GPT-6 models). Otherwise it is
dropped, together with the IDs of that response's other items, which the Responses API
would reject without their reasoning. Anthropic thinking is kept. Gemini thought
signatures and OpenRouter `reasoning_details`, including those of `openai/*` models, are
replayed unchanged, and a different model of the same provider may not accept them.

## Saved tasks and configurations

`AgentLLMConfig` saves the provider, the model ID, and a copy of the model's data. When a
configuration or task is loaded, a model the registry knows takes the registry's current
data (pricing and limits). An unknown ID keeps its saved copy, and an OpenRouter slug
without a saved copy loads without catalog metadata. A hand-written configuration can
therefore give only `provider` and `model_id` for a registry model or an OpenRouter slug.
Optional fields of `AgentConfig` and `Task` have serde defaults, so files saved by older
versions still load (a task saved without `running_tool_calls` has none).

## Testing

The tests use scripted model responses, a paused clock, and local HTTP fixtures, so they
make no provider calls and need no credentials:

```bash
cargo test -p arko_agents --locked
```

The terminal tests run Python fixtures. See
[interactive terminals](interactive-terminals.md#testing).
