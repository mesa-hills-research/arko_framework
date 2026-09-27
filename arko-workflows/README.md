# Arko Workflows

Multi-agent workflow orchestration built on [arko-agents](../arko-agents) and [arko-ai](../arko-ai).

See the [workflow-author Index](INDEX.md) for the complete API, signatures, parameters, defaults, and agent-tool contracts.

Arko Workflows enables binaries and scripts to compose multiple LLM agent runs into cohesive workflows with shared memory, durable messaging, budget management, lifecycle hooks, context-freshness tracking, completion verification, replay-safe steps, and persistence — via an intuitive builder API in both Rust and Python.

## Background

As tasks scale in volume and complexity, single-agent harnesses break down. They skip steps, miss subtle requirements, or implement them poorly. Arko Workflows is the culmination of over a year of work on `arko_ai` and `arko_agents`, providing a batteries-included library for building multi-agent harnesses across domains — coding, research, creative writing, computer use.

The design philosophy is **maximum moldability balanced with maximum intuitiveness**. The library does not constrain you to DAGs or rigid pipeline shapes. Your Rust or Python code *is* the workflow. The library provides the primitives — agent execution, shared state, hooks, budget, persistence — and you compose them however your problem demands.

## Installation

Memory reads return exact versioned changes after the first full read. Distillers
can select tools/agents and configure size thresholds, models, and reasoning.
See [context management](../docs/context-efficiency.md) for the response contract,
delivery baselines, and a Luna High configuration example.

For interactive shell work, register the `arko-agents` terminal tools with
`WorkflowBuilder::agent_tool`. See [terminal integration](../docs/interactive-terminals.md)
for input requests, background notifications, and host lifecycle handling.

Add to your `Cargo.toml`:

```toml
[dependencies]
arko-workflows = { path = "../arko-workflows" }
arko-agents = { path = "../arko-agents" }
arko-ai = { path = "../arko-ai" }
tokio = { version = "1.37", features = ["full"] }
```

For Python bindings (requires [Maturin](https://github.com/PyO3/maturin)):

```bash
cd arko-workflows
maturin develop --features python
```

## Quick Start (Rust)

```rust
use arko_workflows::prelude::*;
use arko_agents::AgentConfig;
use arko_ai::LLMClient;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), WorkflowError> {
    let client = Arc::new(
        LLMClient::new().with_anthropic_api_key(&std::env::var("ANTHROPIC_API_KEY").unwrap())
    );

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

## Quick Start (Python)

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

For a complete sequential orchestrator/coder/correctness-verifier/efficiency-verifier workflow, see
the [DeepCoder example](examples/deepcoder/README.md).

The UI proposal generator and private client gallery have moved to the separate
`arko-ui-studio` repository. That application consumes this library as a dependency.

## Core Concepts

### Workflow Progression Patterns

Arko Workflows does not enforce a single execution model. Your code drives progression:

**Code-Driven (Imperative)**
```rust
let a = workflow.run_agent("architect", prompt).await?;
let c = workflow.run_agent("coder", &a.summary).await?;
let v = workflow.run_agent("verifier", &c.summary).await?;
```

**Parallel Execution**
```rust
let results = workflow.run_agents_parallel(vec![
    ("coder", "Implement auth module".into()),
    ("coder", "Implement database module".into()),
]).await?;
```

**Non-Blocking Spawn**
```rust
let handle = workflow.spawn_agent("researcher", "Find relevant files").await?;
// ... do other work ...
let result = handle.wait().await?;
```

**Custom Event Handling**
```rust
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
let _task = join.await??;
```

**Agent-Driven (via custom tools)**

Register a workflow-aware spawn tool that lets one agent decide which other agents to run. The tool can hold shared workflow state and invoke [`Workflow::run_agent`](src/workflow/mod.rs) internally.

**Loop-Driven / External Decision**
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

### Memory System

Tiered, access-controlled shared memory across agents:

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

Agents interact with memory via injected tools: `read_memory`, `write_memory`, `list_memories`. Writes retain author, task, timestamp, and entry identity. Reads can request a bounded window, and tiers can enforce a byte limit.

### Hooks

Register callbacks at lifecycle points for logging, monitoring, or control flow:

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

Available hook points:
- `PreWorkflow`, `PostWorkflow`
- `PreAgentSpawn`, `PostAgentCompletion`, `PostAgentFailure`, `PostAgentInterruption`
- `PreToolExecution`, `PostToolExecution`
- `PostToolResultInterception`, `OnUserInputRequired`
- `OnBudgetWarning`, `OnBudgetExceeded`
- `OnWorkflowFailed`, `OnWorkflowPause`, `OnWorkflowResume`
- `OnAgentMessage`, `OnMesiConflict`

Hook results are `Continue`, `Skip`, and `Abort(reason)`. Transform tool output with a
`ToolResultInterceptor`, which runs before the result enters the agent conversation.

### Context Freshness

Freshness tracking records versioned reads and writes for files, directory trees, memory tiers, and workflow-defined resources. A successful write proactively tells active stale readers that their context changed; a stale writer can also be blocked until it rereads:

```rust
let workflow = Workflow::builder()
    .context_freshness(ContextFreshnessConfig {
        block_on_conflict: false,
        ..ContextFreshnessConfig::default()
    })
    .build()?;
```

Tool-to-resource mappings are configurable. The legacy MESI builder is retained as a compatibility adapter. Diff-bearing invalidations are not currently implemented.

### Budget Management

Track cost across all agents with optional limits and warnings:

```rust
let workflow = Workflow::builder()
    .budget(10.0)
    .budget_warning(8.0)
    .build()?;

let result = workflow.result();
println!("Total spent: ${:.6}", result.total_cost);
```

Agent costs count what the account is billed (`UsageInfo::billed()`): calls on a subscription such as ChatGPT OAuth record zero against the budget while their per-token estimate stays in the ledger entry's `UsageInfo`. No per-agent exemption is needed.

> **Changed:** `BudgetManager::mark_oauth_exempt`, `is_oauth_exempt` and `oauth_exempt_snapshot` were removed, and `BudgetManager::from_ledger` no longer takes an exemption set (`from_ledger(limit, warning, ledger)`). The `oauth_exempt_agents` field of checkpoints written by older versions is ignored when they are loaded; ledger entries already in those checkpoints keep their recorded amounts.

### Inter-Agent Messaging

Agents use the injected `send_message` tool. Messages are queued durably and acknowledged only after a successful recipient run. An active recipient receives the same message through a safe turn-boundary context injection.

The optional `attachments_json` string is a JSON array of typed references. References are described to the receiving agent; they are not silently loaded or granted access:

```json
[
  {"type":"file","path":"src/lib.rs"},
  {"type":"memory_tier","tier":"architecture"},
  {"type":"transcript","task_id":"550e8400-e29b-41d4-a716-446655440000"},
  {"type":"text","name":"constraint","content":"Keep the API compatible."}
]
```

Recipients are validated against the workflow roster. Messages carry stable IDs, sender task IDs, typed attachments, and timestamps, all of which survive checkpoints.

### Human Input and Suspension

Register a `UserInputHandler` to route an agent's `ask_user` request to an application, chat service, or UI. If no handler is configured, the runtime issues internal `SuspendTask`, preserves `AwaitingUserResponse`, returns `AgentRunStatus::WaitingForInput`, and checkpoints the task. Resume it with `continue_agent(agent_run_id, response)`.

`SuspendTask` is intentionally different from cancellation: it stops the live runtime without turning the task into failed or cancelled state. It is a runtime control command, not an agent-facing tool.

### Durable Steps

Wrap non-workflow side effects in a durable step with a stable key, input hash, version, and operation ID:

```rust
let receipt: Receipt = workflow
    .durable_step(DurableStepSpec::new("pay-bill"), &bill, |step| async move {
        provider.pay(&bill, step.operation_id()).await
    })
    .await?;
```

The journal fsyncs `Started` before running the effect and `Completed` after serializing the output. If execution crashes between those records, replay yields `InDoubt` and does not blindly retry. A `ResolveInDoubt` implementation can query an idempotency key, provider receipt, database, or other domain evidence and return completed, definitely-not-completed, still-in-doubt, or permanently-failed. Operator resolutions are append-only and include actor/reason provenance.

This is replay safety, not a universal exactly-once guarantee. An external system must expose idempotency or observation for automatic uncertainty resolution.

### Completion Verification and Distillation

Completion verifiers gate the actual `complete_task` tool call. A correctness verifier can approve, request a bounded minor repair, or reject major. An efficiency verifier is advisory: even if it emits a major rejection, the workflow coerces that to a capped minor repair, so efficiency alone cannot hard-fail completion. Verifier configurations can opt into same-conversation rechecks and bounded tracked file/memory repair deltas, retaining a fresh first review while avoiding repeated whole-project inspection after localized repairs.

Tool-result distillers are opt-in, agent/tool scoped, LLM-backed interceptors for large outputs. Their calls are budget-accounted, recorded, and composed with application interceptors; `complete_task` is never distilled.

### Code-Defined Services and Voting

Workflows remain ordinary Rust/Python code rather than graphs. Opt-in workflow services allow selected agents to invoke configured workflow-agent targets through `run_workflow_agent`, with independent caller and target allowlists plus inherited budget, verification, and persistence. The voting primitive provides named ballots, eligibility, quorum, thresholds, weights, and explicit or deterministic tie handling without imposing a scheduler.

### Persistence

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

Checkpoints serialize the workflow/run identity, metadata, complete memory state, pending messages, cost ledger, freshness versions, full agent tasks/results, verification records, and distillation records. Agent tasks and a JSONL event log are also stored separately. Writes use temporary files, sync, and rename; `.resume_from(path)` restores operational state rather than only displaying summaries.

### BYOT (Bring Your Own Tool)

Register custom tools for all agents or specific agents:

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

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    User Code                        │
│  (Rust binary / Python script / Tauri app)         │
├─────────────────────────────────────────────────────┤
│                  arko_workflows                     │
│  ┌──────────┐ ┌────────┐ ┌──────┐ ┌────────────┐   │
│  │ Workflow │ │ Memory │ │Fresh.│ │   Hooks    │   │
│  │ Builder  │ │ Store  │ │Track │ │  Registry  │   │
│  │ & Engine │ │& Tools │ │  er  │ │            │   │
│  └─────┬────┘ └────────┘ └──────┘ └────────────┘   │
│        │        ┌────────┐ ┌──────┐ ┌──────────┐   │
│        │        │Message │ │Budget│ │Persistence│  │
│        │        │  Bus   │ │ Mgr  │ │ Manager   │  │
│        │        └────────┘ └──────┘ └──────────┘   │
├────────┼────────────────────────────────────────────┤
│        │                  arko_agents               │
│        ├──→ AgentRuntime.run_task()                 │
│        │    ┌──────────┐  ┌──────────┐             │
│        │    │AgentLoop │  │ToolReg.  │             │
│        │    │(event    │  │(builtins │             │
│        │    │ stream)  │  │+ custom) │             │
│        │    └──────────┘  └──────────┘             │
├─────────────────────────────────────────────────────┤
│                    arko_ai                          │
│           LLMClient (OpenAI / Anthropic / Google)   │
└─────────────────────────────────────────────────────┘
```

## License

MIT OR Apache-2.0
