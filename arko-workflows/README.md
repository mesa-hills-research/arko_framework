# Arko Workflows

Multi-agent workflow orchestration built on [arko-agents](../arko-agents/README.md)
and [arko-ai](../arko-ai/README.md). Workflows are ordinary Rust or Python code: the
library provides agent execution, shared memory, messaging, budgets, hooks,
context-freshness tracking, completion verification, durable steps, and
persistence, and your code decides which agents run and in what order. There is no
graph or pipeline definition to write.

The [API index](INDEX.md) lists every public type, signature, default, and
agent-tool contract.

## Installation

```toml
[dependencies]
arko-workflows = { path = "../arko-workflows" }
arko-agents = { path = "../arko-agents" }
arko-ai = { path = "../arko-ai" }
tokio = { version = "1.37", features = ["full"] }
```

The optional `python` feature compiles PyO3 bindings
(`cargo check -p arko-workflows --features python`). The crate is not yet packaged
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
    let client = Arc::new(LLMClient::from_env());

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

`LLMClient::from_env()` reads these variables and ignores blank values:

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

The managed paths (`run_agent`, `continue_agent`, `run_agents_parallel`, and
`AgentHandle::wait`) approve tool calls automatically, record costs and results,
and apply budgets, hooks, verification, and persistence. To handle events yourself,
split a spawned handle into its channels. The caller then takes over event handling,
approvals, usage accounting, and result recording:

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
returns. One that has not stopped within the grace period (30 seconds by default,
set with `cancel_grace_period`) is aborted. `cancel()` returns `false` if the run
was already stopped. The `OnWorkflowCancelled` hook fires once, and
`PostAgentCancellation` fires for each cancelled agent.

A hook that returns `Abort(reason)` stops the run the same way, and the calls
return `WorkflowError::AbortedByHook` instead.

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

Agents use the injected `read_memory`, `write_memory`, and `list_memories` tools.
Each write records its author, task, timestamp, and entry ID. Reads can request a
trailing window, and a tier can enforce a byte limit. After the first full read,
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
- `OnUserInputRequired`
- `OnBudgetWarning`, `OnBudgetExceeded`
- `OnWorkflowFailed`, `OnWorkflowCancelled`, `OnWorkflowPause`, `OnWorkflowResume`
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
stale reader. See [context freshness](INDEX.md#context-freshness).

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

Agents send messages with the injected `send_message` tool. A message stays queued
until a run of the recipient that received it completes successfully. A recipient
that is already running receives the message at its next turn boundary.
Recipients are validated against the workflow's agent names, and messages carry
stable IDs, sender task IDs, typed attachments, and timestamps, all of which
survive checkpoints.

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

Completion verifiers gate an agent's `complete_task` call. A correctness verifier
approves, requests a minor repair, or rejects with a major finding, within
configurable repair limits. An efficiency verifier is advisory: a major rejection
from it counts as a minor repair request, so efficiency alone cannot fail a task.
A verifier can continue its previous conversation on re-review and, with context
freshness enabled, receive a bounded summary of the tracked file and memory changes
since its last verdict, so a localized repair does not require a full
re-inspection. See
[completion verification](INDEX.md#completion-verification).

## Tool-result distillation

Distillers are opt-in interceptors that replace large tool outputs with a
model-written summary. They are scoped to selected agents (by the names they are
registered under) and tools and configure the size threshold, model, and
reasoning. Their calls count against the budget and are recorded. A summary
replaces the output only when the model finished it; a refusal or a summary cut
off by its output limit keeps the raw output.
`complete_task` results are never distilled. See
[context management](../docs/context-efficiency.md#tool-result-distillation) for a
configuration example.

## Workflow services and voting

Workflow services let selected agents run other configured agents through the
`run_workflow_agent` tool, with separate caller and target allowlists and a limit
on concurrent calls. Delegated runs share the workflow's memory, budget, hooks,
verification, and persistence. Voting provides named ballots with eligibility,
quorum, consensus thresholds, weights, and tie handling. See
[workflow services](INDEX.md#workflow-services-and-custom-tools) and
[structured voting](INDEX.md#structured-voting).

## Persistence

Checkpoint workflow state at agent boundaries:

```rust
use arko_workflows::persistence::PersistenceConfig;

let workflow = Workflow::builder()
    .persistence(PersistenceConfig {
        checkpoint_dir: "./checkpoints".into(),
        auto_checkpoint: true,
    })
    .build()?;

workflow.checkpoint().await?;
```

A checkpoint contains the workflow and run identity, metadata, memory, pending
messages, the cost ledger, freshness versions, complete agent tasks and results,
verification and distillation records, and, once the run is stopped, its
cancellation. Agent tasks and a JSONL event log are also stored separately. Files
are written to a temporary file, synced, and renamed into place. `WorkflowBuilder::resume_from(path)` restores this state into a new
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
            supports_batched_calls: false,
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

For interactive shell work, register the arko-agents terminal tools with
`WorkflowBuilder::agent_tool`; see [interactive terminals](../docs/interactive-terminals.md).

## Architecture

```text
┌───────────────────────────────────────────────────────┐
│ Application code (Rust binary or Python script)       │
├───────────────────────────────────────────────────────┤
│ arko-workflows                                        │
│   workflow builder and engine · memory store, tools   │
│   freshness tracker · hooks · message bus · budgets   │
│   verification · distillation · persistence           │
├───────────────────────────────────────────────────────┤
│ arko-agents                                           │
│   AgentRuntime · agent loop (event stream)            │
│   tool registry (built-in and custom tools)           │
├───────────────────────────────────────────────────────┤
│ arko-ai                                               │
│   LLMClient: OpenAI, Anthropic, Google Gemini,        │
│   OpenRouter                                          │
└───────────────────────────────────────────────────────┘
```

## License

MIT OR Apache-2.0
