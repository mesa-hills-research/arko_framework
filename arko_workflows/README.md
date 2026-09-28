# Arko Workflows

Multi-agent workflow orchestration built on [arko_agents](../arko_agents/README.md)
and [arko_ai](../arko_ai/README.md). Workflows are ordinary Rust or Python code: the
library provides agent execution, shared memory, messaging, budgets, hooks,
context-freshness tracking, completion verification, durable steps, and
persistence, and your code decides which agents run and in what order. There is no
graph or pipeline definition to write.

The [API index](INDEX.md) lists every public type, signature, default, and
agent-tool contract.

## Installation

```toml
[dependencies]
arko_workflows = { path = "../arko_workflows" }
arko_agents = { path = "../arko_agents" }
arko_ai = { path = "../arko_ai" }
tokio = { version = "1.37", features = ["full"] }
```

The optional `python` feature compiles PyO3 bindings
(`cargo check -p arko_workflows --features python`). The crate is not yet packaged
as a Python extension module: it builds only a Rust library (no `cdylib` crate
type) and has no `pyproject.toml`, so tools such as `maturin develop` need both
added first.

## Quick start (Rust)

```rust
use arko_workflows::prelude::*;
use arko_agents::AgentConfig;
use arko_ai::LLMClient;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), WorkflowError> {
    // Configures every provider whose credentials are in the environment.
    let client = Arc::new(LLMClient::from_env()?);

    let workflow = Workflow::builder()
        .name("my-workflow")
        .workspace("./project")
        .llm_client(client)
        .agent("planner", AgentConfig::load_from_path("configs/planner.json".as_ref()).unwrap())
        .agent("coder", AgentConfig::load_from_path("configs/coder.json".as_ref()).unwrap())
        .budget(5.0)
        .build()?;

    let plan = workflow.run_agent("planner", "Create an implementation plan").await?;
    let _code = workflow.run_agent("coder", &format!("Implement:\n{}", plan.summary)).await?;

    println!("Done! Cost: ${:.4}", workflow.result().total_cost);
    Ok(())
}
```

`LLMClient::from_env()` reads these variables and ignores blank values. Setting only some
of the `OPENAI_OAUTH_*` variables is an error:

| Provider | Variables |
| --- | --- |
| OpenAI | ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (optional `OPENAI_OAUTH_REFRESH_TOKEN`); otherwise `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY`, otherwise `GEMINI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

For other credential sources, such as a Codex `auth.json` (`arko_ai::CodexAuthFile`),
build the client yourself.

## Quick start (Python)

```python
import asyncio
import arko_workflows as aw

async def main():
    builder = aw.PyWorkflowBuilder()
    builder.name("my-workflow")
    builder.workspace("./project")
    builder.agent("planner", "configs/planner.json")
    builder.agent("coder", "configs/coder.json")
    builder.budget(5.0)
    workflow = builder.build()

    plan = await workflow.run_agent("planner", "Create an implementation plan")
    print(f"Plan cost: ${plan.total_cost:.4f}")

    code = await workflow.run_agent("coder", f"Implement:\n{plan.summary}")
    print(f"Total cost: ${plan.total_cost + code.total_cost:.4f}")

asyncio.run(main())
```

The Python builder calls `LLMClient::from_env()` during `build`, so it reads the
same environment variables. The Python API covers a subset of the Rust API; see
[Python bindings](INDEX.md#python-bindings).

For a complete orchestrator, coder, and verifier workflow, see the
[DeepCoder example](examples/deepcoder/README.md).

## Running agents

Your code drives progression. Run agents in sequence:

```rust
let a = workflow.run_agent("architect", prompt).await?;
let c = workflow.run_agent("coder", &a.summary).await?;
let v = workflow.run_agent("verifier", &c.summary).await?;
```

In parallel, including several runs of one configured agent:

```rust
let results = workflow.run_agents_parallel(vec![
    ("coder", "Implement auth module".into()),
    ("coder", "Implement database module".into()),
]).await?;
```

In the background:

```rust
let handle = workflow.spawn_agent("researcher", "Find relevant files").await?;
// ... do other work ...
let result = handle.wait().await?;
```

Or from a loop that decides the next step:

```rust
loop {
    let next = my_planning_function(&workflow.context())?;
    match next {
        Action::RunAgent { name, prompt } => {
            workflow.run_agent(&name, &prompt).await?;
        }
        Action::Done => break,
    }
}
```

The managed paths (`run_agent`, `continue_agent`, `resume_agent`,
`run_agents_parallel`, and `AgentHandle::wait`) answer tool approvals and
error-recovery pauses under the workflow's [policies](#approvals-and-error-recovery)
(by default: approve every call, and recover from errors automatically), record costs
and results, and apply budgets, hooks, verification, and persistence. In a parallel
run, an agent that pauses for input or a decision is not a failure: its result comes
back with the others, to be continued. `max_parallel_agents(n)` caps how many agents
of a parallel call run at once (no cap by default), and `fail_fast(true)` (off by
default) stops the others at the first failure. To handle events yourself, split a spawned
handle into its channels. The caller then takes over event handling, approvals, usage
accounting, and result recording:

```rust
use arko_agents::{AgentEvent, UserCommand};

let handle = workflow.spawn_agent("agent", "task").await?;
let (mut events, commands, join, _name) = handle.into_parts();

while let Some(event) = events.recv().await {
    match event {
        AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => {
            commands.send(UserCommand::ApproveToolUse {
                tool_call_id,
                custom_tool_output: None,
                message_to_agent: None,
            }).await.ok();
        }
        AgentEvent::TaskStreamEnd => break,
        _ => {}
    }
}
let _task = join.await.expect("agent task panicked")?;
```

To let an agent decide which other agents run, enable
[workflow services](#workflow-services-and-voting).

## Cancelling a run

A `Workflow` is one run. `cancel_handle()` returns a `WorkflowCancelHandle` that
can be cloned and used from any task or thread; `cancel()` on it (or on the
workflow) stops the run:

```rust
let cancel = workflow.cancel_handle();
tokio::spawn(async move {
    if tokio::signal::ctrl_c().await.is_ok() {
        cancel.cancel();
    }
});

match workflow.run_agent("coder", "Implement the parser").await {
    Ok(result) => println!("{}", result.summary),
    Err(WorkflowError::Cancelled) => {
        // The agents have stopped, and their results are recorded.
        for agent in workflow.result().agent_results {
            println!("{}: {:?}", agent.agent_name, agent.status);
        }
    }
    Err(error) => return Err(error),
}
```

Every running agent of the run is told to stop, including agents started with
`spawn_agent`. Each pending call waits for its agents to end, records their
results (status `AgentRunStatus::Cancelled` for a stopped agent), and returns
`WorkflowError::Cancelled`; every later call returns it at once without starting
anything. An agent stops at its next step, or when a long tool call it is running
returns (`wait_for_message` returns at once). One that has not stopped within the grace period (30 seconds by default,
set with `cancel_grace_period`, counted from the stop) is aborted; an agent running
inside another agent's tool call, such as a completion verifier, gets half its
parent's time. `cancel()` returns `false` if the run was already stopped. The
`OnWorkflowCancelled` hook fires once, and `PostAgentCancellation` fires for each
cancelled agent.

A hook that returns `Abort(reason)` stops the run the same way, and the calls
return `WorkflowError::AbortedByHook` instead. Whichever stop comes first stands.

With persistence, the checkpoints written after the stop record it in
`cancellation`, and `finish()` saves one and returns the stop's error. Resuming
from such a checkpoint starts the run again; to keep a cancelled run stopped,
check the checkpoint's `cancellation` before calling `resume_from`. From Python,
`workflow.cancel()` stops the run, and pending `run_agent` and
`run_agents_parallel` calls raise `arko_workflows.WorkflowCancelled`. See
[stopping a run](INDEX.md#stopping-a-run) for the details, including durable steps.

## Shared memory

Memory tiers are named, access-controlled text stores shared by the agents:

```rust
let workflow = Workflow::builder()
    .memory(
        MemoryConfig::new()
            .global_tier("decisions")
            .broadcast_tier("architecture", ["architect"])
            .tier(MemoryTier {
                name: "coder-notes".into(),
                content: String::new(),
                readers: AccessScope::Only(["coder", "verifier"].iter().map(|s| s.to_string()).collect()),
                writers: AccessScope::Only(["coder"].iter().map(|s| s.to_string()).collect()),
                write_mode: WriteMode::AppendOnly,
                entries: Vec::new(),
                max_bytes: Some(256 * 1024),
            })
    )
    .build()?;
```

Agents use the injected memory tools, each offered by access: `read_memory` to an
agent that can read at least one tier, `write_memory` to one that can write one, and
`list_memories` (which lists only the tiers the agent can access) to either. `build`
rejects an access set naming an unregistered agent.
Each write records its author, task, timestamp, and entry ID, and its text is stored
once. Reads can request a trailing window, and a tier can enforce a byte limit
(opt-in; there is none by default). On `resume_from`, the memory configuration in
code decides access and limits and the checkpoint supplies the content by tier name;
a checkpoint tier that isn't configured stops the build unless it is renamed
(`rename_memory_tier_on_resume`) or dropped (`drop_memory_tier_on_resume`). After the first full read,
`read_memory` returns exact versioned changes; see
[context management](../docs/context-efficiency.md#memory-reads).

## Hooks

Hooks are synchronous callbacks at lifecycle points, for logging, monitoring, or
control flow:

```rust
use arko_workflows::prelude::*;

let workflow = Workflow::builder()
    .hook(HookPoint::PostAgentCompletion, Arc::new(|ctx| {
        println!(
            "Agent '{}' done. Spent ${:.4}",
            ctx.agent_name.as_deref().unwrap_or("?"),
            ctx.budget_spent
        );
        HookResult::Continue
    }))
    .hook(HookPoint::PreToolExecution, Arc::new(|ctx| {
        if ctx.tool_name.as_deref() == Some("dangerous_tool") {
            return HookResult::Skip;
        }
        HookResult::Continue
    }))
    .hook(HookPoint::OnBudgetWarning, Arc::new(|ctx| {
        eprintln!("WARNING: Budget at ${:.4}", ctx.budget_spent);
        HookResult::Continue
    }))
    .build()?;
```

Hook points:

- `PreWorkflow`, `PostWorkflow`
- `PreAgentSpawn`, `PostAgentCompletion`, `PostAgentFailure`,
  `PostAgentCancellation`, `PostAgentInterruption`
- `PreToolExecution`, `PostToolExecution`, `PostToolResultInterception`
- `OnUserInputRequired`, `OnAgentPaused`
- `OnBudgetWarning`, `OnBudgetExceeded`
- `OnWorkflowFailed` (once per failed run call; not for a pause or a cancellation),
  `OnWorkflowCancelled`, `OnWorkflowPause`, `OnWorkflowResume`
- `OnAgentMessage`, `OnFreshnessConflict`

A hook returns `Continue`, `Skip`, or `Abort(reason)`. `Skip` rejects the current
tool call only at `PreToolExecution`. `Abort` at any point stops the whole run:
running agents are cancelled, and the run calls return
`WorkflowError::AbortedByHook` (see [cancelling a run](#cancelling-a-run)). At
`PreToolExecution` the call does not run; at `PostToolExecution` its result
stands. Hooks cannot change tool output: use a `ToolResultInterceptor`, which runs
before the result enters the agent's conversation. See
[hooks and human input](INDEX.md#hooks-and-human-input) for when each point fires
and what its context contains.

## Context freshness

Freshness tracking records versioned reads and writes of files, directory trees,
memory tiers, and workflow-defined resources. When a write succeeds, active agents
that read the resource earlier receive a notice that it changed. A stale writer
can also be blocked until it reads the resource again:

```rust
let workflow = Workflow::builder()
    .context_freshness(ContextFreshnessConfig {
        block_on_conflict: false,
        ..ContextFreshnessConfig::default()
    })
    .build()?;
```

`ContextFreshnessConfig::default()` tracks reads through `read_file` and
`list_files` and writes through `write_file` and `apply_diff`, and blocks stale
writes. The mapping from tool names to resources is configurable. Notices state
that a resource changed; they do not include the change. A write that makes
another agent's read stale also fires the `OnFreshnessConflict` hook once per
stale reader. A reader whose task has finished is neither notified nor reported
to the hook; its reads are kept while the task can be continued, and
`continue_agent` (or `resume_agent`) then tells it, in one message, which resources
it read have changed since, advising a reread before editing them. See
[context freshness](INDEX.md#context-freshness).

## Budgets

Track cost across all agents, with optional limits and a warning threshold:

```rust
let workflow = Workflow::builder()
    .budget(10.0)
    .budget_warning(8.0)
    .build()?;

let result = workflow.result();
println!("Total spent: ${:.6}", result.total_cost);
```

Budgets count what the account is billed (`UsageInfo::billed()`). Calls billed to a
subscription, such as ChatGPT OAuth, add zero to the budget; their per-token
estimate remains in the ledger entry's `UsageInfo`. `agent_budget` sets a
cumulative limit for one agent, and `add_manual_cost` records charges from outside
the model calls. See [budget and cost accounting](INDEX.md#budget-and-cost-accounting).

## Messaging

Agents send messages with the injected `send_message` tool, to agents that are
running or waiting (for example on a rate limit or for approval); a recipient
receives the message at its next turn boundary. Messages to agents that have not
started or have finished are refused with their status, not queued. An accepted
message stays queued until a run of the recipient that received it completes
successfully, so a later run receives it again otherwise. Messages carry stable
IDs, sender task IDs, typed attachments, and timestamps, all of which survive
checkpoints.

Agents learn whom they can reach from three sources: the initial prompt of each
new run lists the agents it may message with their roles and statuses, a
"Workflow context update" tells a running agent when one of them starts, and the
`list_recipients` tool reports current statuses. Recipient information never goes
into tool definitions or system prompts, which stay the same during a task. (An agent
allowed to delegate through workflow services finds the agents it can run, with
their roles, in the `run_workflow_agent` tool's definition; that list is fixed by
the configuration.)
`workflow.agent_status(name)` gives the same status to your code: not started,
running, waiting (with the reason), or finished (with the outcome). Agents started
together with `run_agents_parallel` see each other as running from the start.

A `MessagingPolicy` sets whom each agent may message. By default every agent may
message every other registered agent. An agent is offered `send_message` and
`list_recipients` only when it has at least one permitted recipient, so a
single-agent workflow has neither tool. A message to anyone else is rejected
without naming other agents. Restrict an agent with `can_message`, or turn
messaging off with `disable_messaging`:

```rust
use arko_workflows::messaging::MessagingPolicy;

let workflow = Workflow::builder()
    // ...
    .agent("lead", lead_config)
    .agent("worker_a", worker_config.clone())
    .agent("worker_b", worker_config)
    // Workers report to the lead; the lead may message both workers.
    .messaging(
        MessagingPolicy::all()
            .can_message("worker_a", ["lead"])
            .can_message("worker_b", ["lead"]),
    )
    .build()?;
```

Names in the policy are registration names; an unknown one fails the build. The
policy applies to every run of an agent, including runs started through workflow
services. With messaging disabled, or for agents that are not offered the tools,
the names `send_message`, `list_recipients` and `wait_for_message` are free for
tools of your own. See [messaging](INDEX.md#messaging) for the statuses, notices,
and tool results.

### Waiting for a message

An agent in native tool mode must call a tool every turn, so an agent that another
agent may message is offered `wait_for_message` to wait for a reply. It returns when a
message arrives (the message itself comes, once, with the agent's next turn), when an
agent it works with starts or finishes, or at its time limit, with a short result such
as ``Message from `lead` waiting.`` or `No messages after 270 s.` The model may pass
`timeout_seconds` to wait less than the limit, never more. Meanwhile the agent's status
is `waiting (awaiting_message)`. Cancelling the workflow, or the workflow
stopping the agent, ends the wait at once.

A wait lasts up to about 4.5 minutes by default. Each wake-up is a cheap turn that keeps
the provider's prompt cache warm, so a long wait is simply several short waits. With
Anthropic's 1-hour cache TTL (`anthropic_cache_ttl` set to `AnthropicCacheTtl::OneHour`)
the default stretches to about 54 minutes. You can set a longer wait, for every agent or
one:

```rust
use std::time::Duration;

let workflow = Workflow::builder()
    // ...
    .wait_for_message_timeout(Duration::from_secs(10 * 60))
    .agent_wait_for_message_timeout("lead", WaitForMessageTimeout::Auto)
    .build()?;
```

Provider caches often last longer (OpenAI documents about 30 minutes, and a test over
ChatGPT OAuth still hit after 40), but that isn't guaranteed, and a miss re-reads the
whole context at full price.

The optional `attachments_json` parameter is a JSON array of typed references.
They are described to the recipient; files, memory, and transcripts are not loaded
or shared automatically:

```json
[
  {"type":"file","path":"src/lib.rs"},
  {"type":"memory_tier","tier":"architecture"},
  {"type":"transcript","task_id":"550e8400-e29b-41d4-a716-446655440000"},
  {"type":"text","name":"constraint","content":"Keep the API compatible."}
]
```

## Human input

Register a `UserInputHandler` to route an agent's `ask_user` questions to an
application, chat service, or UI. Without a handler, the run is suspended:
`run_agent` returns `WorkflowError::UserInputRequired { agent, task_id, questions }`
and records an `AgentRunStatus::WaitingForInput` result in
`workflow.result().agent_results`. With persistence configured, the suspended task
is also saved. To resume, pass that result and the answer to
`continue_agent(&waiting_result, answer)`.

Suspension stops the live runtime without marking the task failed or cancelled.
`OnUserInputRequired` fires, and `OnWorkflowFailed` doesn't: a pause is not a
failure.

## Approvals and error recovery

**By default a workflow runs without human input.** Two builder settings decide what
happens when an agent would otherwise stop for a person:

- `approval_policy`: an agent asks for approval for a tool call its configuration
  doesn't auto-approve (`default_auto_approve_tool_use` is off by default in
  arko_agents) and for a path outside the workspace under
  `PathSafetyPolicy::RequireApproval`. **The default, `ApprovalPolicy::ApproveAll`,
  approves every such call**, and logs the first one per agent, tool, and reason as a
  warning. `RespectAgentConfig` pauses the agent for the caller instead; `Handler`
  gives each call to your `ToolApprovalHandler`, which approves, rejects, or pauses.
  The tool hooks receive why a call needed approval (`HookContext::approval_reason`).
- `error_recovery_policy`: the agent runtime first retries a failed model request on
  its own (with backoff, a rate limit after the provider's delay, without ever cutting
  off a request in flight); when those retries run out, the agent pauses. **The
  default, `ErrorRecoveryPolicy::Automatic`, recovers without an attempt limit**: after
  an outage it waits (30 seconds, growing to 15 minutes while failures repeat) and
  sends the request again unchanged; at a plan usage limit it waits until the limit
  resets; after tool errors it continues. It fails the run with a clear reason on
  errors that retrying can't fix (authentication, invalid requests), on refusals that
  stand after the agent's refusal fallback, and on responses the runtime can't use.
  The model is never told that a user approved anything. `Limited { max }` caps the
  automatic recoveries per run call; `ReturnToCaller` pauses at every error-recovery
  pause.

A paused agent is recorded with `AgentRunStatus::Paused`, the call returns
`WorkflowError::AgentPaused { agent, task_id, pause }`, and `OnAgentPaused` fires.
Decide and continue with `resume_agent`:

```rust
use arko_agents::runtime::command::ToolApprovalDecision;
use arko_workflows::prelude::*;

match workflow.run_agent("coder", "Deploy the fix").await {
    Err(WorkflowError::AgentPaused { task_id, pause, .. }) => {
        let paused = workflow.agent_result(task_id).unwrap();
        let resume = match pause {
            AgentPause::ToolApproval { calls } => AgentResume::ToolApprovals(
                calls.iter().map(|call| (call.tool_call_id, ToolApprovalDecision::Approve {
                    custom_tool_output: None,
                    message_to_agent: None,
                })).collect(),
            ),
            AgentPause::ErrorRecovery { .. } => AgentResume::ContinueAfterError { note: None },
        };
        workflow.resume_agent(&paused, resume).await?;
    }
    other => { other?; }
}
```

The paused task is saved with checkpoints, so it can be resumed after a restart. A
pause by an opt-in agent monitor (the plan-usage threshold's `Pause` action or the
cache monitor's `pause_below`) always returns to the caller. An agent running inside
another agent's tool call (a completion verifier or a delegated run) can't pause:
there a call that needs approval is rejected and the agent is told, and an
error-recovery pause left to the caller fails that run. Give such agents
configurations that auto-approve the tools they need, or use a `Handler`. See
[Approvals and error recovery](INDEX.md#approvals-and-error-recovery) for every cause
and option.

## Durable steps

Wrap side effects outside the workflow in a durable step with a stable key, input
hash, version, and operation ID:

```rust
let receipt: Receipt = workflow
    .durable_step(DurableStepSpec::new("pay-bill"), &bill, |step| {
        let bill = &bill;
        async move { provider.pay(bill, step.operation_id()).await }
    })
    .await?;
```

The journal syncs a `Started` record to disk before running the effect and a
`Completed` record after serializing its output. A completed step replays its
output without running again. If the process stops between the two records, the
step is `InDoubt` on replay and is not retried blindly. A `ResolveInDoubt`
implementation can check an idempotency key, provider receipt, database, or other
evidence and report the step as completed, definitely not completed, still in
doubt, or permanently failed. Operator resolutions are appended to the journal
with the actor and reason.

Durable steps make replay safe; they are not an exactly-once guarantee. Automatic
resolution needs an external system that supports idempotency or can be queried
for the outcome. Durable steps require persistence. See
[durable steps](INDEX.md#durable-steps).

## Completion verification

Completion verifiers gate the `complete_task` call of the agents they name with
`target(..)`; a verifier without a target fails to build. A correctness verifier
approves, requests a minor repair, or rejects with a major finding, within
configurable repair limits. An efficiency verifier is advisory: a major rejection
from it counts as a minor repair request, so efficiency alone cannot fail a task.
A verifier can continue its previous conversation on re-review and, with context
freshness enabled, receive a bounded summary of the tracked file and memory changes
since its last verdict, so a localized repair does not require a full
re-inspection. Without conversation reuse, a new review of a task is given the
prior verdict and findings (`include_prior_findings(false)` opts out). A review that
fails (the verifier's run errors, or its verdict stays unusable after
`format_retries`) doesn't fail the target: the agent is asked to call
`complete_task` again, up to `review_retries` times, and then the run call returns
`WorkflowError::VerifierFailed`. See
[completion verification](INDEX.md#completion-verification).

## Tool-result distillation

Distillers are opt-in interceptors that replace large tool outputs with a
model-written summary. Each names the agents (by the names they are registered
under) and the tools it distills, at least one of each, `read_file` included when
wanted; a distiller without targets fails to build. It sets the size threshold,
model, and reasoning. Their calls count against the budget and are recorded. A summary
replaces the output only when the model finished it; a refusal or a summary cut
off by its output limit keeps the raw output.
`complete_task` and `ask_user` results are never distilled, and `load_skill` results
only when the agent's `SkillsConfig::load_skill_interception` opts in. See
[context management](../docs/context-efficiency.md#tool-result-distillation) for a
configuration example.

## Workflow services and voting

Workflow services let named agents run other configured agents through the
`run_workflow_agent` tool, with separate caller and target lists and a limit on
concurrent calls. Callers and targets are named explicitly (`allow_caller`,
`allow_target`); with none named, no agent is offered the tool. The opt-ins
`allow_all_callers` and `allow_all_targets` admit every agent except
completion-verifier agents and, as a target, the caller itself. Delegated runs share
the workflow's memory, budget, hooks, verification, and persistence. Voting
(`enable_voting(voters)`, which offers `cast_vote` to those agents only) provides
named ballots with eligibility, quorum, consensus thresholds, weights, and tie
handling. See
[workflow services](INDEX.md#workflow-services-and-custom-tools) and
[structured voting](INDEX.md#structured-voting).

## Persistence

Checkpoint workflow state at agent boundaries:

```rust
use arko_workflows::persistence::PersistenceConfig;

let workflow = Workflow::builder()
    .persistence(PersistenceConfig::new("./checkpoints"))
    .build()?;

workflow.checkpoint().await?;
```

A run directory holds the checkpoints, one task file per agent run
(`agents/<task id>.json`, with its full conversation), the event log
(`events.jsonl`), and the durable-step journal. A checkpoint contains the workflow
and run identity, metadata, memory, pending messages, the cost ledger, freshness
versions, each agent result with a reference to its task file (a conversation is
stored once, in its task file), verification and distillation records, and, once the
run is stopped, its cancellation. Files are pretty-printed JSON, written to a
temporary file, synced, and renamed into place.

By default a checkpoint is saved after every finalized agent run, and every
checkpoint is kept: `retain_checkpoints(n)` opts in to keeping only the newest `n`. A
task file always holds its task's latest state, so an older checkpoint reads a task
that has since run on as it is now (a warning is logged when its recorded status and
the task disagree).
The event log records every event, including each streamed text and reasoning
fragment; `event_log(EventLog::LifecycleOnly)` leaves the fragments out (the
finished responses stay in the task files). Events are buffered and written at least
once a second (`event_flush_interval`), and at once for state changes, run ends, and
error reports. A failed event-log write is logged as a warning and retried; it never
stops an agent. `WorkflowBuilder::resume_from(path)` restores this state into a new
workflow; see [persistence](INDEX.md#persistence-and-checkpoint-state) for what is
restored and what must be configured again.

## Custom tools

Register tools for every agent or for one agent:

```rust
use arko_agents::tool::*;
use async_trait::async_trait;

#[derive(Debug)]
struct MyTool;

#[async_trait]
impl Tool for MyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "my_tool".to_string(),
            description: "Does something useful".to_string(),
            output_based_example: None,
            parameters: vec![],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _ctx: &TaskContext<'_>,
        _params: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<String, ToolError> {
        Ok("result".to_string())
    }
}

let workflow = Workflow::builder()
    .global_tool(Arc::new(MyTool))
    .agent_tool("special-agent", Arc::new(MyTool))
    .build()?;
```

For interactive shell work, register the arko_agents terminal tools with
`WorkflowBuilder::agent_tool`; see [interactive terminals](../docs/interactive-terminals.md).

## Architecture

```text
┌───────────────────────────────────────────────────────┐
│ Application code (Rust binary or Python script)       │
├───────────────────────────────────────────────────────┤
│ arko_workflows                                        │
│   workflow builder and engine · memory store, tools   │
│   freshness tracker · hooks · message bus · budgets   │
│   verification · distillation · persistence           │
├───────────────────────────────────────────────────────┤
│ arko_agents                                           │
│   AgentRuntime · agent loop (event stream)            │
│   tool registry (built-in and custom tools)           │
├───────────────────────────────────────────────────────┤
│ arko_ai                                               │
│   LLMClient: OpenAI, Anthropic, Google Gemini,        │
│   OpenRouter                                          │
└───────────────────────────────────────────────────────┘
```

## License

MIT OR Apache-2.0
