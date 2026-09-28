# Arko Agents

Tool-using agents built on [arko_ai](../arko_ai/README.md). An `AgentConfig` describes an
agent: its role and instructions, model, tools, and approval and error policies. A `Task`
holds one run's conversation and state. `AgentRuntime` runs a task as a Tokio task,
streams `AgentEvent`s, and accepts `UserCommand`s for approvals, answers, error recovery,
context updates, cancellation, and suspension. [arko_workflows](../arko_workflows/README.md)
builds multi-agent orchestration on this crate.

## Installation

```toml
[dependencies]
arko_agents = { path = "../arko_agents" }
arko_ai = { path = "../arko_ai" }
tokio = { version = "1.37", features = ["full"] }
```

## Quick start

```rust
use arko_agents::{
    AgentConfig, AgentEvent, AgentLLMConfig, AgentRuntime, AgentToolConfig, Task, TaskState,
    ToolCallingMode, ToolRegistry, UserCommand,
};
use arko_ai::{GenerationConfig, LLMClient, Model};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm_config = AgentLLMConfig {
        model: Model::anthropic_claude_sonnet_5(),
        generation_config: GenerationConfig::default(),
    };
    let config = AgentConfig::builder("writer".into(), "Writer".into(), llm_config)
        .role_description("You are a careful software engineer.".into())
        .instructions("Make the requested change, then call complete_task with a summary.".into())
        .tool_config(AgentToolConfig {
            available_tool_names: vec![
                "list_files".into(),
                "read_file".into(),
                "write_file".into(),
                "complete_task".into(),
            ],
            tool_calling_mode: ToolCallingMode::Native,
            ..AgentToolConfig::default()
        })
        .build()?;

    // Configures every provider whose credentials are in the environment.
    let client = Arc::new(LLMClient::from_env()?);
    let workspace = std::fs::canonicalize("./project")?;
    let task = Task::new(config, "Add a README that explains how to build the project.".into(), workspace, None);

    let (mut events, commands, join) = AgentRuntime::new(client).run_task(
        task,
        Arc::new(ToolRegistry::new()),
        None, // event throttle
        true, // simplified tool events
        None, // skills
        None, // tool-result interceptor
        None, // tool execution middleware
    );

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::AssistantResponseChunk(text) => print!("{text}"),
            AgentEvent::ToolCallAwaitingApproval { tool_call_id, tool_name, parameters, .. } => {
                println!("\nApproving {tool_name}: {parameters:?}");
                commands
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await?;
            }
            AgentEvent::ErrorEncountered { error_message, requires_approval_to_retry: true, .. } => {
                eprintln!("\n{error_message}");
                commands.send(UserCommand::RejectErrorRecovery { message: None }).await?;
            }
            AgentEvent::TaskStreamEnd => break,
            _ => {}
        }
    }

    let task = join.await??;
    match &task.current_state {
        TaskState::Completed { summary } => println!("\nCompleted: {summary}"),
        state => println!("\nStopped: {state:?}"),
    }
    Ok(())
}
```

`run_task` returns the event receiver, the command sender, and a join handle that
resolves to the task in its final state; `TaskStreamEnd` is the last event of a run. Keep
receiving events until then: the channel holds 100 events and the runtime waits while it
is full, and if the receiver is dropped, the run ends with `ArkoAgentsError::ChannelSend`
instead of returning the task. Keep the command sender alive too: when every sender has
been dropped, the runtime returns the task as `Interrupted`. Tools only hold weak senders.

The task's working directory is the agent's workspace. File tools resolve relative paths
against it, and paths outside it are rejected by default (see
[Approvals and workspace paths](#approvals-and-workspace-paths)). A relative working
directory is resolved against the process's current directory.

The `arko_agents_cli` example runs one agent from a JSON configuration, with interactive
approvals, answers, and error recovery:

```bash
cargo run -p arko_agents --example arko_agents_cli -- \
  --agent-config /path/to/agent.json \
  --workspace-dir /path/to/workspace --prompt "Add a README"
```

Its options include `--tool-calling-mode native|output_based`,
`--simplified-tool-events false` (output-based calls also reported parameter by
parameter; `true` by default), `--mcp-config PATH`,
`--openai-auth-json PATH` (ChatGPT OAuth from a Codex `auth.json`; the `OPENAI_OAUTH_*`
variables are then not read), `--save-task-state PATH` (the final task as JSON), and the
opt-in `--user-command-timeout SECONDS`, after
which a task waiting for a command is returned `Interrupted`; without it the task waits
until a command arrives. The example loads a `.env` file and then reads other credentials
from the environment with `LLMClient::from_env()`.

## Agent configuration

`AgentConfig` is built with `AgentConfig::builder(id, name, llm_config)` or loaded from
JSON with `AgentConfig::load_from_path`. The builder requires a role description and
instructions and validates the retry, monitoring, and refusal-fallback settings; the
runtime validates them again when a task starts. The quick start's agent as JSON:

```json
{
  "id": "writer",
  "name": "Writer",
  "role_description": "You are a careful software engineer.",
  "instructions": "Make the requested change, then call complete_task with a summary.",
  "llm_config": { "provider": "Anthropic", "model_id": "claude-sonnet-5" },
  "tool_config": {
    "available_tool_names": ["list_files", "read_file", "write_file", "complete_task"],
    "tool_calling_mode": "Native"
  }
}
```

| Field | Default | Purpose |
| --- | --- | --- |
| `id`, `name` | Required | Identifier and display name; the name heads the system prompt and is the `agent_name` tools receive. |
| `role_description`, `instructions` | Required | The system prompt's role and instructions sections. |
| `llm_config` | Required | Model and `GenerationConfig`; see [Saved tasks and configurations](#saved-tasks-and-configurations). |
| `tool_config` | `complete_task` only | Available tools, tool calling mode, approvals, batching, and workspace path policy. |
| `environment_details_config` | Nothing included | Working directory, file tree, date and time, and operating system, each placed in the system prompt, the task prompt, or every user message. |
| `error_recovery_config` | `InformAgentAndContinue` | What a failed tool execution does. |
| `llm_error_policy`, `llm_retry_config` | 3 attempts, then pause | What a failed model request does; see [Model-call recovery](#model-call-recovery). |
| `user_command_timeout_seconds` | None (waits indefinitely) | Opt-in: how long an awaiting task waits for a command before it is returned `Interrupted`. |
| `user_reminder` | None | Text appended as `# Reminder` to the task prompt, every user message, or every Nth message. |
| `skills_config` | None | Skill sources; see [Skills](#skills). |
| `monitoring` | Nothing monitored | Plan-usage and prompt-cache monitors; see [Monitoring](#monitoring). |
| `refusal_fallback` | None | A model that answers a refused turn once; see [Refusals](#refusals). |
| `complete_task_summary_prompt_override` | None | Replaces the description of `complete_task`'s `summary` parameter. |
| `parsing_config_override` | Default tags | Tag syntax for output-based tool calls. |

In `role_description` and `instructions`, text between `{{#output_based_only}}` and
`{{/output_based_only}}` is kept, without the markers, in output-based mode and removed
in native mode, so one configuration can carry tool-call examples for text-parsed calls.

Developer-defined environment details are passed to `Task::new`; each has static content
or a `Fn(&Task)` provider, which is not serialized. File-tree details default to a full
first snapshot followed by changes at model-call boundaries; see
[context management](../docs/context-efficiency.md#file-tree-context).

## Tools

`ToolRegistry::new()` registers the built-in tools: `complete_task`, `read_file`,
`write_file`, `apply_diff`, `list_files`, and `ask_user`. An agent is offered the
registered tools named in `tool_config.available_tool_names`, and always
`complete_task`, which ends the task as `Completed { summary }`. `ask_user` puts the task
in `AwaitingUserResponse` until `UserCommand::SubmitUserResponse` answers it.

The runtime runs only the tools the agent is offered, in either tool calling mode. In
native mode, a call to any other tool, registered or not, gets an error result and emits
`UnavailableToolAttempted`; output-based parsing recognizes only the offered tools' tags,
so other markup stays response text. `run_task` binds the offered tools to the task with
`Tool::bind_runtime`. A saved task is offered the tools its own configuration names.

### Tool calling modes

`tool_config.tool_calling_mode` selects how calls are exchanged with the model:

- `OutputBased` (the default): the system prompt describes each tool with an XML-like
  example, and the runtime parses calls from the response text. A response holds one call
  or one batch of calls to the same tool. Every parameter value arrives as a JSON string,
  except that `true` or `false` (in any case) for a `bool` parameter becomes a boolean.
- `Native`: tools are sent through the provider's tool-calling API. A response can call
  several different tools, but only when every tool in it allows batching; otherwise none
  of the calls runs and each gets an error result explaining why.

A tool allows batching when its definition sets `supports_batched_calls` and the
configuration enables it: a per-tool `allow_batched_calls` override, or else
`allow_batched_tool_calls` (default `false`). With `allow_batched_tool_calls` off, native
requests also disable parallel tool calls on models that support that setting. Every
response must call a tool; a response without one is handled as a failed model request.

Response text is streamed as it arrives in native mode. In output-based mode, a turn's
text and tool events are emitted once the response has been parsed.

### Approvals and workspace paths

Tool calls wait for approval unless `default_auto_approve_tool_use` or a per-tool
`auto_approve_tool_use` override is set. `ask_user` runs without approval, since its
question already waits for an answer, unless its own override sets `auto_approve_tool_use:
Some(false)`. When a response holds one call that needs
approval, the runtime emits `ToolCallAwaitingApproval`; when it holds several calls, it
emits `BatchToolCallAwaitingApproval` listing those that need approval. Answer with
`ApproveToolUse`, `RejectToolUse`, or `SubmitBatchToolApprovals`, where a call without a
decision is rejected. An approval can carry `custom_tool_output`, which is used instead
of running the tool, and a `message_to_agent`.

Parameters marked `is_path` are checked against the task's working directory. A path
outside it is handled by `default_out_of_workspace_path_policy` or a per-tool override:
`AutoReject` (the default) rejects the call, `RequireApproval` asks for approval even for
auto-approved tools, and `AutoApprove` allows it. A native call is rejected with an error
result and `ToolRejected`; an output-based call is rejected as a tool error, under
`on_tool_execution_error`. A path that cannot be resolved, such as one holding a NUL
byte, is rejected this way under every policy.

The check and the built-in file tools resolve a path the same way, with
`arko_agents::utils::resolve_path`: surrounding whitespace is trimmed, `~` and `$VAR` are
not expanded (`~/notes` names a directory called `~`), a relative path is joined to the
working directory, and the result is made absolute and its `.` and `..` resolved. On
Windows this follows Win32: `\` and `/` both separate components, a drive-relative
`C:dir` uses that drive's current directory, trailing dots and spaces are dropped, and a
verbatim `\\?\` path is kept as written. A custom tool with an `is_path` parameter should
open the path `resolve_path` returns, so the check judges the path the tool opens;
end-user command tools receive that path. A non-string value for an `is_path` parameter
is rejected.

The check compares the resolved path with the working directory component by component.
On Windows it ignores ASCII case (`c:\work\x` is inside `C:\Work`) and matches a verbatim
prefix with its ordinary form (`\\?\C:\` with `C:\`, `\\?\UNC\server\share` with
`\\server\share`), also in a directory with case sensitivity turned on. Elsewhere the
comparison is exact, so on a macOS file system that ignores case, a path spelled in
another case is judged outside. The check is not a sandbox: symbolic links are not
followed, so a link inside the workspace can lead outside it, and a path spelled through
a link is compared as written. On macOS, for example, `/var` links to `/private/var`,
where temporary directories live. Give the working directory in canonical form
(`std::fs::canonicalize`, as the quick start does), since programs such as `pwd` report
canonical paths.

### Tool errors

`error_recovery_config.on_tool_execution_error` handles a tool that returns an error:

- `InformAgentAndContinue` (the default) reports the error as that call's result (a
  native error result, or a text result in output-based mode) and emits
  `ErrorEncountered`; the other calls of the response run as usual.
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
            supports_batched_calls: true,
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

The runtime does not check arguments against the definition before calling `execute`;
`ToolDefinition::validate_params` is available for tools that want the check. A tool can
return images with `execute_with_output`, opt out of result interception with
`allow_result_interception`, and track what reached the conversation with
`reset_context` and `observe_context`. `bind_runtime` gives a background tool a weak
command sender for the task, through which it can send `UserCommand::InjectContext`.

End-user tools wrap a command without Rust code: a `ToolDefinition` whose
`tool_type_details` is `EndUserDefined { command_template }`, loaded with
`ToolRegistry::load_end_user_tool`. The template's `~` and environment variables are
expanded and it is split on whitespace, without interpreting quotes. String parameters
without a `flag` follow as positional arguments, in order, and then flagged string
parameters as `flag value` and each true boolean parameter as its flag. An `is_path`
parameter passes the path the workspace check judged, with `..` resolved so a symbolic
link cannot redirect it: relative to the working directory inside it (`./src/main.rs`,
never a verbatim `\\?\` path on Windows) and absolute outside it, so the program cannot
read it as an option. The command runs in the task's working directory without a shell,
and its standard output is the result; a nonzero exit status is a tool error. The process
is killed when its call is dropped, for example when the task's runtime is aborted
mid-call. A boolean parameter takes a JSON boolean or the text `true` or `false`, so it
works in both tool calling modes.

### Result interception and execution middleware

The last two `run_task` arguments wrap tool execution:

- An `InterceptorConfig` holds a `ToolResultInterceptor`, which can replace a result
  before it enters the conversation (for example with a summary), and an
  `InterceptionFailurePolicy` for interceptor errors, which falls back to the original
  result. `ToolResult` carries the original output; `ToolResultIntercepted` follows with
  the replacement. Results of `complete_task` and `ask_user` are never intercepted.
- A `ToolExecutionMiddleware` runs `before` and `after` every execution, whether
  approved or auto-approved, native or output-based. `before` returns `Continue`,
  `Reject { message_to_agent }`, `FailTask { reason }`, or `CancelTask { reason }`,
  which ends the task as `Cancelled { reason }` (arko_workflows uses it when a run is
  cancelled). The rejected call gets `reason` as its result, and the calls of the
  response the stopped task did not run get a result saying so ("This call was not run:
  Workflow cancelled."). A call answered with `custom_tool_output` skips `before`, and
  `after` receives `CallerProvided`.

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

`register_tools` adds them to a registry and `tool_names` returns the names to add to
`available_tool_names`; `shutdown` closes each connection and gives the server three
seconds to exit before killing it, and dropping the manager kills the servers. MCP tools
require native tool calling, and the workspace path policy cannot apply to them.

A server's stderr goes to the log (the `log` crate) at info level, a line at a time,
tagged with the server's name (`MCP server 'filesystem' stderr: ...`); it is not printed
to the host's stderr. A server that exits while the manager still uses it is reported
once, with a warning giving its exit status and last stderr lines; its tools then fail
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
auto-approval; call it before `Task::new`, which keeps its own copy of the
configuration. It returns the `SkillPromptContext` to pass to `run_task`. Auto-loaded
skills are placed in the system prompt; the others are listed there by name and
description and loaded on demand.

### Interactive terminals

The opt-in `terminal` module provides PTY and pipe terminal tools that run commands in the
background and notify the agent of input requests and completion. See
[interactive terminals](../docs/interactive-terminals.md).

## Running tasks

`TaskState` is one of `Pending`, `InProgress`, `AwaitingToolApproval`,
`AwaitingBatchToolApproval`, `AwaitingUserResponse`, `AwaitingErrorRecoveryApproval`,
`Interrupted`, `Completed`, `Failed`, or `Cancelled { reason }`; `Completed`, `Failed`,
and `Cancelled` are terminal.

| Events | Meaning |
| --- | --- |
| `TaskStateChanged` | The task's state changed. |
| `AssistantResponseChunk`, `ReasoningChunk` | Response text and reasoning; `event_throttle_ms` coalesces chunks. |
| `ToolCallInitiated`, `ToolCallFormatted` | A tool call was announced, then parsed. Output-based mode with `simplified_tool_events` off also emits `ToolParamName`, `ToolParamValueChunk`, and `ToolParamEnd`. |
| `ToolCallAwaitingApproval`, `BatchToolCallAwaitingApproval`, `UserQuestionsAsked` | The task waits for a command. |
| `ToolResult`, `ToolResultIntercepted`, `ToolRejected`, `UnavailableToolAttempted` | Tool outcomes. |
| `ErrorEncountered`, `RetryScheduled`, `UsageLimitReached` | Errors and recovery. `requires_approval_to_retry` marks a pause; `RetryScheduled` carries the attempt, delay, and elapsed milliseconds, and `rate_limit` when the retry waits out a provider rate limit. |
| `RefusalReceived`, `RefusedInput`, `RefusalFallback` | Refusals. |
| `UsageReported`, `ModelCallUsageReported`, `PlanUsageReported`, `ModelRerouted` | Usage after each model call. |
| `CacheEfficiencyReported`, `CacheEfficiencyLow`, `PlanUsageThresholdReached` | Opt-in monitors. |
| `TaskStreamEnd` | The run ended. |

Besides answers to the waiting states, a running task accepts:

- `InjectContext { content }`: queued and added at the next turn boundary as a user
  message headed `# Workflow context update`; a stream in progress is not changed. Every
  injection sent before the boundary (including while a tool call ran, approved or not)
  joins that one message, in the order sent. A `TurnContextSource` given to
  `AgentRuntime::with_turn_context` is also asked for context at each such boundary,
  before each model request, and its answer joins that message.
- `CancelTask`: the task ends as `Cancelled { reason }` with `CANCELLED_BY_USER` ("Task
  cancelled by user."). Calls it waited on get a result saying they were not run. A task
  saved by an older version as `Failed { reason: "Task cancelled by user." }` loads as
  `Cancelled`.
- `SuspendTask`: the runtime stops and returns the task in its current state, for example
  to save it. A task suspended while it awaits a command, in any awaiting state, keeps
  its pending calls and results and waits for that command when it runs again.

`CancelTask` and `SuspendTask` stop the task without another model request, also when
queued behind context. Context the runtime has taken from the channel is never lost when a
task stops, for any reason. It is added to the conversation before the task is returned,
unless a user message cannot go there yet: while the task awaits a command (a native
call's result must come right after the call), or while a refusal is unresolved (a pending
fallback answers exactly the refused input, and `Task::resolve_refusal` needs it to be the
last input). Then it is kept in `Task::pending_context_injections`, saved with the task,
and added at the first turn boundary where it can go once the task runs again, before
context sent later. A completed task also keeps the context taken during its last turn
there, since the model has not read it, and a continued task gets it.

Tasks serialize to JSON (`Task::save_to_path`, `Task::load_from_path`). Running a
returned task again continues it: a task that is pending, in progress, or interrupted
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
what the account is billed: ChatGPT OAuth calls are covered by the subscription and add
zero, while their `UsageInfo::total_cost` keeps the list-price estimate.

## Model-call recovery

`llm_error_policy` decides what a failed model request does, and `llm_retry_config` sets
the timing:

| Policy | Behavior |
| --- | --- |
| `RetryThenPauseAndAwaitApproval { max_attempts }` | The default, with 3 attempts. Retries with backoff; when recovery ends, the task waits in `AwaitingErrorRecoveryApproval`. |
| `RetryThenInformAgent { max_attempts }` | Retries the same way; when recovery ends, the task is returned `Interrupted`. |
| `InformAgentAndContinue` | Retries without an attempt limit until the elapsed budget ends, then returns the task `Interrupted`. |
| `PauseAndAwaitApproval` | Waits in `AwaitingErrorRecoveryApproval` after the first failure. |

Recovery ends when the attempts or the elapsed budget run out or the error cannot be
retried. Attempt limits count the first request, and zero counts as one. Each ordinary
retry adds a note about the error to the conversation. From a pause,
`ApproveErrorRecovery { message }` adds a note about the error (with the message) and
starts a new recovery episode, and `RejectErrorRecovery` fails the task. An interrupted
task starts a new episode when it is run again.

Errors are classified by arko_ai's `Error::retry_class()`. HTTP 408, 409, 429, and 5xx
responses, connection failures and timeouts, truncated streams, errors reported after a
stream started without a permanent code, and responses without a tool call are retried.
Invalid requests, missing provider configuration, authentication failures, and permanent
provider codes (such as quota exhaustion and invalid refresh grants) end recovery at once.

| `LLMRetryConfig` field | Default | Meaning |
| --- | --- | --- |
| `initial_delay_ms` | 250 | First backoff delay. |
| `max_delay_ms` | 64,000 | Cap of the exponential backoff. |
| `max_elapsed_ms` | 300,000 | Recovery budget from the first failure, including waits, token refreshes, requests, and streams. |
| `jitter` | `true` | Each backoff delay is drawn between half and all of its capped value. |
| `rate_limit_retry` | On | Waits for short rate limits and capacity errors; see below. |
| `capacity_retry` | `None` | Opt-in indefinite waits for capacity errors. |
| `usage_limit_policy` | `Pause` | Handling of plan usage limits. |

For example, ten attempts with indefinite capacity waits:

```rust
use arko_agents::{LLMCapacityRetryConfig, LLMErrorPolicy, LLMRetryConfig};

let builder = builder
    .llm_error_policy(LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 10 })
    .llm_retry_config(LLMRetryConfig {
        capacity_retry: Some(LLMCapacityRetryConfig::default()),
        ..LLMRetryConfig::default()
    });
```

Zero delays, a zero budget, and a `max_delay_ms` below `initial_delay_ms` are rejected.
A delay the provider asks for (`Retry-After`, rate-limit reset headers, or a delay in the
error body, as `Error::retry_after()` reports it) is a minimum, and jitter is added above
it so that agents sharing a reset do not retry together. It may exceed `max_delay_ms` but
must fit in the remaining budget; otherwise recovery ends rather than retrying early. A
turn with valid tool calls ends the recovery episode; a token refresh or an injected
context message does not.

**Rate limits and capacity errors.** Under either `RetryThen*` policy, short rate limits
(HTTP 429, `rate_limit_exceeded`, `slow_down`, and other providers' 429 codes) and,
unless `capacity_retry` is set, capacity errors (explicit overloaded or unavailable
signals) are retried after the provider's delay plus jitter, or with backoff from 1 s to
60 s when it gives none, until they succeed or the task is cancelled or suspended. These
retries resend the request unchanged, add nothing to the conversation, and neither use nor
reset the attempt limit and elapsed budget; the budget's clock stops while they wait. Rate
limits are waited out for at most `max_wait_ms` (15 minutes) in total per recovery
episode. After that, or when a rate limit asks for a longer wait, the error is handled as
a plan usage limit; a capacity error asking for a longer wait gets ordinary recovery.
`LLMRateLimitRetryConfig` sets the backoff (`initial_delay_ms`, `max_delay_ms`) and
`max_wait_ms`; `rate_limit_retry: None` (JSON `null`) gives both kinds of error ordinary
recovery.

**Indefinite capacity waits.** `capacity_retry: Some(LLMCapacityRetryConfig::default())`
retries capacity errors under either `RetryThen*` policy until the service recovers, with
backoff from 30 seconds to 5 minutes and the same provider minimums and jitter, outside
the ordinary limits. A success ends the capacity episode; another kind of error continues
ordinary recovery. `PauseAndAwaitApproval` still pauses at the first failure.

**Plan usage limits.** A ChatGPT plan usage limit (`usage_limit_reached`) emits
`UsageLimitReached` with the reset time. The default `UsageLimitPolicy::Pause` ends
recovery under every error policy. The opt-in `WaitUntilReset { max_wait_ms }` waits,
under every error policy and cancellably, until the reported reset plus one second and,
with jitter, up to 30 seconds more, then resends the request unchanged. If no reset time
is reported, or the whole wait would not fit in `max_wait_ms`, recovery still ends.

**Streams.** Output a failed stream delivered is kept, followed by the error note, when
recovery adds a message, and dropped when the request is resent unchanged. A request
whose conversation ends on assistant text, such as a task resumed after `SuspendTask`
mid-stream, first gets a user message asking the model to continue, because some
providers reject a conversation that ends on an assistant turn (Claude 4.6 and later
reject prefill). Native calls announced with `ToolCallInitiated` that never run (the
stream failed, was interrupted, or was refused, or the final response does not hold
them) get a `ToolRejected` event. Model recovery never runs a tool call again. A stream
that fails may have used provider tokens without reporting its usage.

**ChatGPT OAuth.** Before each OpenAI request, the runtime refreshes ChatGPT OAuth
credentials that expire within five minutes. The refresh makes at most one token request,
under the same attempt and elapsed limits as model requests, and a failed refresh is
recovered like a failed request. A request rejected with HTTP 401 `token_expired` is
refreshed and resent once by arko_ai. With credentials from a Codex `auth.json`, build the
client with `LLMClient::with_codex_auth_file`, so that every refresh re-reads the file
first and rotated tokens are saved to it at once; see
[ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials).
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
  request while usage stays that high; context sent during the wait joins that request,
  in a second context update. The reported window is saved in
  `Task::plan_usage_threshold_reported`, so a resumed task does not report it again, and
  a report whose window has since reset is ignored.
- `cache: Some(CacheMonitorConfig::default())` emits `CacheEfficiencyReported` after each
  call: cache reads divided by the most that could have been read (the smaller of the
  previous and the current input), for the call and over the last `window_calls` (5)
  calls. The first `warmup_calls` (2) calls, calls without a cache figure, and calls to
  the refusal fallback model are not judged. Once `window_calls` calls have been judged,
  a rolling efficiency below `warn_below` (0.5) emits `CacheEfficiencyLow` once until it
  recovers; the opt-in `pause_below` also pauses before the next request.

A monitor pause waits in `AwaitingErrorRecoveryApproval` under the pausing error policies
and returns the task `Interrupted` under the others. Approving it continues without an
error note; a message given with the approval is added as a user message.

## Refusals

A refusal is never retried unchanged. The runtime emits `RefusalReceived` with the
`arko_ai::Refusal`, then `RefusedInput`, and handles the refusal as an error that cannot
be retried: the task pauses or is returned `Interrupted`, as the error policy decides.

A model's own refusal (`RefusalSource::Model`, such as an OpenAI structured refusal) stays
in history, so the model can see what it declined. Its tool calls are not run (announced
calls get `ToolRejected` with the reason `turn refused`) and are not kept. In native mode
the refusal explanation becomes the turn's text when the turn has none; in output-based
mode the explanation replaces the text, or without one, the text is kept without its
tool-call markup. A turn left without text is not kept. Classifier and content-filter
refusals are removed from history.

**Changing the refused input.** `RefusedInput` lists the user messages after the last
assistant turn, each a `RefusedMessage` with its `message_index`, `runtime` and
`runtime_text` flags, `tool_results` (by provider call ID), and image count. It is also
kept in `Task::refused_input` until the refusal is resolved, recovery is approved, or a
later turn completes. Messages the runtime added (file-tree updates, error and recovery
notes, requests to continue) are marked `runtime` and cannot be changed, and neither can
a recovery note beside native tool results (`runtime_text`). While the task awaits
recovery approval, `UserCommand::ResolveRefusal { changes }` applies a list of changes;
for a task returned interrupted, call `Task::resolve_refusal(changes)` before running it
again. The changes apply together or not at all: an invalid one emits `ErrorEncountered`
and the task keeps waiting.

| `RefusalResolution` | Effect |
| --- | --- |
| `RephraseUserMessage { message_index, text }` | Replaces the message's text and keeps the sections the runtime appended to it (`# Initial Environment Information`, `# Current Environment Information (updated this turn)`, `# Reminder`). In a message with tool results, it replaces the text beside them, such as a native `ask_user` answer. The text must not be empty. Rephrasing the task prompt applies its `output_based_only` markers and updates `Task::initial_task_prompt`. |
| `RemoveUserMessage { message_index }` | Removes a message without tool results. At least one user message not added by the runtime must remain. |
| `RemoveImages { message_index }` | Removes a message's images, such as tool screenshots, with their descriptions. |
| `RedactToolResults { call_ids, replacement }` | Replaces tool outputs; each call stays paired with its result. |

The refused input must still be the conversation's last input, followed at most by the
kept refused turn. The refused turn is removed, the changed input is sent as it is, and
tools that track the conversation observe it from the start again. Change a task's input
this way rather than with arko_ai's `Conversation` methods (`rephrase_last_user_turn`,
`remove_last_user_turn`, `redact_tool_result`), which do not update the task's record of
runtime messages or its refused input. `ApproveErrorRecovery` continues without a change.
After a model's own refusal, which stays in history, the next input is then usually only
the runtime's recovery note, which cannot be changed if the model refuses again.

**Fallback model.** The opt-in `AgentConfig::refusal_fallback` is an `AgentLLMConfig` on
the configured model's provider; a fallback on another provider is rejected, since
providers cannot replay each other's turns. A refusal from the configured model is
reported (with its `RefusedInput`), and the same input goes once to the fallback model
(`RefusalFallback`). The refused turn is not kept; the fallback's response continues the
task, and later turns use the configured model again. The pending fallback is saved with
the task (`Task::fallback_for`), survives a monitor pause, and holds back context updates
(saved with the task too) and file-tree changes until it has answered. If the fallback
also refuses or its request fails, the refusal stands and can be resolved as above;
approving recovery resends to the configured model.

Across the two models, the ChatGPT turn-state token of one is not sent with the other's
requests. OpenAI reasoning is replayed between them only when both are `Provider::OpenAI`
models of one Codex compaction family (the GPT-5.6 and GPT-6 models); otherwise it is
dropped, together with the IDs of that response's other items, which the Responses API
would reject without their reasoning. Anthropic thinking is kept. Gemini thought
signatures and OpenRouter `reasoning_details`, including those of `openai/*` models, are
replayed unchanged, and a different model of the same provider may not accept them.

## Saved tasks and configurations

`AgentLLMConfig` saves the provider, the model ID, and a copy of the model's data. When a
configuration or task is loaded, a model the registry knows takes the registry's current
data (pricing and limits); an unknown ID keeps its saved copy, and an OpenRouter slug
without a saved copy loads without catalog metadata. A hand-written configuration can
therefore give only `provider` and `model_id` for a registry model or an OpenRouter slug.
Optional fields of `AgentConfig` and `Task` have serde defaults, so files saved by older
versions still load.

## Testing

The tests use scripted model responses, a paused clock, and local HTTP fixtures, so they
make no provider calls and need no credentials:

```bash
cargo test -p arko_agents --locked
```

The terminal tests run Python fixtures; see
[interactive terminals](../docs/interactive-terminals.md#testing).

## License

MIT OR Apache-2.0
