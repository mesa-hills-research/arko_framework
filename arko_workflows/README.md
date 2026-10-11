# Arko Workflows

Multi-agent workflow orchestration built on [arko_agents](../arko_agents/README.md)
and [arko_ai](../arko_ai/README.md). Workflows are ordinary Rust or Python code: the
library provides agent execution, shared memory, messaging, budgets, hooks,
context-freshness tracking, completion verification, durable steps, and
persistence, and your code decides which agents run and in what order. There is no
graph or pipeline definition to write.

The [API index](INDEX.md) lists every public type, signature, default, and
agent-tool contract; the
[defaults reference](../docs/reference-map.md#arko_workflows) summarizes the
defaults of all four crates.

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

To have a workflow built for you, describe your use case to the
[forge](examples/forge/README.md): an architect agent, helped by explorer agents, designs and
writes an `arko_workflows` crate for it, checks that it builds and passes its tests, and
documents it.

For a compact end-to-end example that exercises the workflow primitives together, see
[`complete_workflow.rs`](examples/complete_workflow.rs). It supports a credential-free
`--dry-run`, checkpoint resume, `continue_agent`, and `resume_agent`.

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
run, an agent that pauses for input or a decision (or because a completion verifier
could not review its completion) is not a failure: its result comes back with the
others, to be continued. `max_parallel_agents(n)` caps how many agents
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

## Stopping a run

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

### Graceful stop

`stop(StopRequest)` stops the run for a stated reason (`StopKind::Operator`,
`Drain` before a redeploy, `Limit`, `Failure`), optionally after a wrap-up in
which running agents finish their work. No wrap-up is the default: the run is
cancelled at once, as with `cancel()`. With one, no new agent run starts, the
tools of running agents see their stop signal (a `StopAwareCommand` child finds
its `ARKO_STOP_FILE`; `wait_for_message` returns), and each agent is told at its
next turn boundary to save its progress and call `complete_task` (completion
verifiers don't review that completion: it reports progress). The wrap-up
ends when no agent runs any more or its time is up; then the run is cancelled as
above, with `CancelCause::Stopped`, and checkpointed. `stop_on_signals` maps
Ctrl-C and `SIGTERM` to a stop:

```rust
use arko_workflows::{StopRequest, stop_on_signals};

let _signals = stop_on_signals(
    workflow.cancel_handle(),
    StopRequest::drain("stopped by a signal").wrap_up(Duration::from_secs(90)),
)?;
```

A second signal ends the wrap-up at once, and a third kills the children of
running `StopAwareCommand`s (on Unix) and exits the process. Under
systemd, give the unit `KillMode=mixed` and a `TimeoutStopSec` longer than the
wrap-up, the grace period and the longest tool stop grace together.

A stopped run is blameless: `AgentResult::stopped_by()` marks the runs a stop
ended (cancelled, or completed in the wrap-up), so a host doesn't count them as
failures. After resuming from the checkpoint, `stopped_runs()` lists them, with
the runs that were interrupted, and `restart(&run, note)` relaunches each (a run
an agent delegated is left to that agent, which is told its ID to resume it):

```rust
let resumed = builder.resume_from(&checkpoint).build()?;
for run in resumed.stopped_runs() {
    resumed.restart(&run, "").await?;
}
```

See [graceful stop](INDEX.md#graceful-stop) for the wrap-up notice and the
rules, and the [workflow design guide](../docs/guides/workflow-design.md#make-side-effects-durable)
for what a resume after a redeploy allows to change.

`stop_run` and `stop_agent` stop one agent run, or one agent's runs, the same
way while the rest of the workflow runs on: to retire an agent, or to replace it
with another model once it has saved its work and reported where it stands. The
host states what it means to do with the run, and the intent is saved:
`RunStopIntent::Retire` keeps it out of `stopped_runs()` for good, and
`Relaunch` lists it there for a later `restart`:

```rust
use arko_workflows::{RunStopIntent, StopRequest};

let request = StopRequest::operator("moving it to Opus 5.5").wrap_up(Duration::from_secs(600));
workflow.stop_agent("worker-7", request, RunStopIntent::Retire)?;
```

See [stopping one agent run](INDEX.md#stopping-one-agent-run).

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
`WorkflowError::AbortedByHook` (see [stopping a run](#stopping-a-run)). At
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
the model calls.

Reaching a budget cancels nothing by default (`BudgetExceededPolicy::Pause`): each
agent over it is held before its next model request, its tool calls finishing
first. Raise the budget while the run is live with `set_budget` (or
`set_agent_budget`) and the held agents go on. A run inside another agent's tool
call waits for the raise; a run the host started returns `AgentPaused` (cause
`BudgetExceeded`) and continues with `resume_agent` once the budget is raised, or
waits too with `wait_for_budget_raise(true)`. A host that can't raise the budget
live stops once `budget_blocked()` resolves (or the call returns paused), saving a
checkpoint; resumed with a higher `.budget(...)`, `recover_agent` and `resume_agent`
continue the agents from their pauses. Checkpoints keep a limit raised live, and a
resume keeps the higher of it and the builder's.

A plan budget limits subscription usage instead of dollars: how many percentage
points of a plan's five-hour and seven-day windows the run may use, counted from
each window's level when the run started (from the plan usage Claude Code and
ChatGPT OAuth report). The ChatGPT plan and the Claude Code plan are counted
separately; `five_hour` and `seven_day` limit each plan's windows, and a plan's
own limits replace them for that plan. Plan usage is account-wide, so other use
of the same plan during the run counts too, and the points are only known after a
request completes. A window over its limit holds agents as the dollar budget
does; `set_plan_budget` raises it, and checkpoints keep the counting, so a
resumed run counts from where it started. The dollar budget counts only what is
billed, so subscription calls count as $0 against it:

```rust
let workflow = Workflow::builder()
    .budget(35.0)
    .plan_budget(
        PlanBudget::none()
            .five_hour(25.0)
            .chatgpt(PlanLimits::none().seven_day(25.0))
            .claude_code(PlanLimits::none().five_hour(85.0)),
    )
    .build()?;
println!("{:?}", workflow.plan_usage_spent_of(arko_ai::SubscriptionPlan::ClaudeCode));
```

```rust
let workflow = Workflow::builder()
    .budget(10.0)
    .wait_for_budget_raise(true)
    .build()?;
// Later, from another task: the held agents continue.
workflow.set_budget(Some(20.0))?;
```

`budget_exceeded_policy(BudgetExceededPolicy::Cancel)` cancels agents over the
budget instead; `reopen_agent` continues a cancelled run from where it stopped. See
[budget holds](INDEX.md#budget-holds) and
[budget and cost accounting](INDEX.md#budget-and-cost-accounting).

## Messaging

Agents send messages with the injected `send_message` tool, to agents that are
running or waiting (for example on a rate limit or for approval); a recipient
receives the message at its next turn boundary. Messages to agents that have not
started or have finished are refused with their status, not queued; with
`MessagingPolicy::keep_messages_for_finished()`, a message to a finished agent is
queued instead for its next run (a continuation or a new run). An accepted
message stays queued until a run of the recipient that received it completes
successfully, so a later run receives it again otherwise. Messages carry stable
IDs, sender task IDs, typed attachments, and timestamps, all of which survive
checkpoints.

Agents learn whom they can reach from three sources: the initial prompt of each
new run lists the agents it may message with their roles and statuses, a
"Workflow context update" tells a running agent when one of them starts or
finishes (saying whether messages to it are still read), and the
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

### Long-running agents

An agent that runs long (a lead coordinating workers, say) can hand its
conversation over instead of letting it grow: with `AgentConfig::handover` set, it
writes a report once its context reaches a threshold and continues from that
report in a fresh conversation, in the same run. Its name, task ID and status don't
change, so messages and replies still reach it. The workflow keeps the messages the
old conversation read acknowledged, lets freshness tracking start over, and gives the
fresh conversation the task and a current list of the agents it may message. See
[Conversation handover](INDEX.md#conversation-handover).

```rust
let mut lead = lead_config;
lead.handover = Some(arko_agents::HandoverConfig::at_context_fraction(0.4));
```

### Claude Code agents

An agent whose model is a Claude Code model (`Model::claude_code(...)`) runs in the
user's installed Claude Code CLI, under its login, and takes part in a workflow like any
other agent: messaging, approvals, error recovery, context freshness and handovers all
apply. Claude Code's own tools (`Read`, `Edit`, `Bash`, …) are offered when the agent's
`available_tool_names` lists them, and they pass the workflow's approval policy and
hooks too. See [Claude Code agents](INDEX.md#claude-code-agents), and
`examples/claude_code_team.rs` for two such agents working together.

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
the provider's prompt cache warm, so a long wait is simply several short waits. The
default follows the agent's cache lifetime: about 27 minutes for OpenAI over ChatGPT
OAuth, whose cache lasts 30, and about 54 minutes with Anthropic's 1-hour cache TTL
(`anthropic_cache_ttl` set to `AnthropicCacheTtl::OneHour`). You can set a longer wait,
for every agent or one:

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

## Operator notes

A note steers the agents while the workflow runs: a correction, a workaround found
meanwhile, a change of priority. Running agents get it at their next turn boundary,
in a context update labelled with its source, and every later run of its audience
gets it in its initial prompt (newest first, capped). Agents are told that a note
refines their task and doesn't change their role's rules. A note never touches an
agent's `AgentConfig`, so a resume still accepts the agents' saved tasks: notes are
the supported way to change instructions in a running or resumed workflow.

```rust
use arko_workflows::notes::{Note, NoteSource};

// To every agent, standing: also shown to every later run.
workflow.post_note(Note::new("The helper's RNG bug is fixed; counts after 14:00 are real."))?;
// To two agents, once: only the runs that have started get it.
workflow.post_note(Note::new("Re-read the spec; section 3 changed.").to(["searcher", "lead"]).once())?;
// From an observer agent instead of the operator.
workflow.post_note(Note::new("The last three runs repeated the same search.")
    .source(NoteSource::Agent("observer".into())))?;

// From another task while agents run.
let notes = workflow.notes_handle();
tokio::spawn(async move { notes.post(Note::new("Wrap up by 18:00.")) });
```

Completion verifiers' reviews get a note only when it names them, and so do the agents set with
`WorkflowBuilder::notes_by_name_only([...])`: keep independent roles, such as a checker,
independent of your steering (and restrict whom they can hear from, since an agent can
pass a note on in a message). A verifier judges without the target's notes, so address a
note to it too when the note changes what counts as done. Notes are saved in checkpoints, logged once, and carried
into the fresh conversation after a handover.

To post from another process (a `note` subcommand of your binary, a bot, an
observer), turn on the control folder with `WorkflowBuilder::control_folder()`; it needs
persistence. Another process appends notes to `<run dir>/control/notes.jsonl` with
`arko_workflows::notes::control::post_note`, under the folder's own lock, whether the run
is live or stopped, and the workflow picks them up at its agents' next turn boundaries.
The `runctl` example does it for any workflow:

```bash
cargo run -p arko_workflows --example runctl -- note /path/to/run "Skip d = 4; it is settled." --to searcher
cargo run -p arko_workflows --example runctl -- status /path/to/run
```

Whoever can write the control folder can post notes, so keep the run directory out of
the agents' reach. See [Operator notes](INDEX.md#operator-notes).

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
  approves every such call**, and logs the first one per agent as a warning (naming
  the tool, and that the agent's other such calls are approved without further
  warnings). `RespectAgentConfig` pauses the agent for the caller instead; `Handler`
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
  A `Middleware` pause, including an execution-middleware hook error, always returns
  to the caller under every policy. The model is never told that a user approved
  anything. `Limited { max }` caps the automatic recoveries per run call;
  `ReturnToCaller` pauses at every other error-recovery pause.

A failed run says what kind of error failed it: `AgentResult::failure_kind()` is the
model request's `arko_ai::ErrorKind`, when it was classified (`Authentication`,
`Billing`, `ContextOverflow`, `InvalidRequest`, ...), and `AgentPause::ErrorRecovery`
and the `PostAgentFailure` hook's `error_kind` metadata carry it too. Decide by it
instead of matching error text: a lost login fails every agent on that credential,
while a request too large for the model fails only one run.

```rust
use arko_ai::ErrorKind;
use arko_workflows::StopRequest;

let result = workflow.agent_result(task_id).unwrap();
if result.failure_kind() == Some(ErrorKind::Authentication) {
    // Every run on this login would fail: stop, save, and ask to sign in again.
    workflow.stop(StopRequest::failure("the login expired"));
}
```

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
another agent's tool call (a completion verifier or a delegated run) can also pause
for a `Middleware` cause, including an execution-middleware hook error; other pauses
are stopped because the parent must receive a bounded result. Under
`RespectAgentConfig` its `complete_task` is approved (its parent waits for that
result), and where it would pause for anything else it is stopped with a reason (a
verifier's review then fails). Under `RespectAgentConfig`, give such agents
configurations that auto-approve the other tools they use, or use a `Handler`. See
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
`WorkflowError::VerifierFailed` with the task paused, neither accepted nor rejected:
`continue_agent` on its result (`Workflow::agent_result(task_id)`) goes on. See
[completion verification](INDEX.md#completion-verification).

## Tool-result distillation

Distillers are opt-in interceptors that replace large tool outputs with a
model-written summary. Each names the agents (by the names they are registered
under) and the tools it distills, at least one of each, `read_file` included when
wanted; a distiller without targets fails to build. It sets the size threshold,
model, and reasoning. Their calls count against the budget and are recorded. A summary
replaces the output only when the model finished it; a refusal or a summary cut
off by its output limit keeps the raw output. An output too large for the distiller's
own model is not sent: it is kept, and the agent's overflow guard (arko_agents'
[tool result size](../arko_agents/README.md#tool-result-size)) shortens it if the
agent's model can't take it either.
`complete_task` and `ask_user` results are never distilled, and `load_skill` results
only when the agent's `SkillsConfig::load_skill_interception` opts in. See
[context management](../docs/context-efficiency.md#tool-result-distillation) for a
configuration example.

## Workflow services and voting

Workflow services let named agents run other configured agents through the
`run_workflow_agent` tool, with separate caller and target lists and a limit on
concurrent calls. With `resume_run`, a call resumes an unfinished run it was told
about instead of starting a new one. Callers and targets are named explicitly (`allow_caller`,
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

Checkpoint workflow state at agent boundaries, and save running agents as they run:

```rust
use arko_workflows::persistence::PersistenceConfig;

let workflow = Workflow::builder()
    .persistence(PersistenceConfig::new("./checkpoints"))
    .build()?;

workflow.checkpoint().await?;
```

A run directory holds the checkpoints, the agent runs' task versions
(`agents/<task id>/<content hash>.json`, each with its full conversation), the event
log (`events.jsonl`), and the durable-step journal. A checkpoint contains the workflow
and run identity, metadata, memory, pending messages, the cost ledger, freshness
versions, each agent result with a reference to the version of its task the
checkpoint saw, verification and distillation records, the operator notes, and, once
the run is stopped, its cancellation. A task version is written only when the task changed, so
checkpoints that saw the same task share its file, and an older checkpoint still
resumes a task that has since run on as it was then. Checkpoints and task versions
are pretty-printed JSON. Every file except the append-only logs and the run identity is written to a
temporary file, synced, and renamed into place (with the directory synced on Unix,
and the rename retried on Windows while another program holds the file), so a crash
leaves the old content or the new.

Running agents are saved as they run (the task journal, on by default; turn it off
with `task_journal(false)`): each running task in `agents/<task id>/live.json`, and
the workflow's state in `live-checkpoint.json`, rewritten at the task's safe points:
after each model turn, before its tool calls run (both files are on disk before a
call with effects starts; read-only calls, which recovery simply runs again, don't
wait), after their results, and at pauses. The live checkpoint is rewritten only when
a resume would read something new from it. `latest-checkpoint` names the newest
checkpoint, live or versioned, and every checkpoint records the runs still running as
`Interrupted`. These files are compact JSON unless
`pretty_json(true)`. If the workflow stops (a crash, a reboot), build it again with
`resume_from` the latest checkpoint and continue the runs that were running:

```rust
for run in workflow.interrupted_runs() {
    workflow.recover_agent(&run).await?;
}
```

A recovered run continues where it was saved, with no new message. A tool call it
was running is answered that it may have run (check its effects) unless its tool
declares it safe to run again (`Tool::recovery_mode`; read-only built-ins do), and a
`run_workflow_agent` call recovers the agent it started, so delegated agents resume
with their callers. Usage reported before the crash is charged once. No delegated run
is orphaned silently: when a caller's call can't recover the run it started (a
budget refuses it, say), the call's result names the run, and the caller can resume
it with the tool's `resume_run` parameter and a message saying what happened. A run
its caller can no longer resume (the caller ended) is listed by `interrupted_runs`
too, with `task.parent` set, and `recover_agent` recovers it for the host.

By default a checkpoint is saved after every finalized agent run, and every
checkpoint and task version is kept: `keep_last_checkpoints(n)` opts in to keeping only
the newest `n` checkpoints, and the task versions they refer to (`n` of 0 fails the build). Saving never fails
an agent's run: a task or checkpoint that can't be written is logged as a warning
and the next checkpoint tries again. A checkpoint written meanwhile refers to the
task's last saved version and lists the run in `missing_task_states`; resuming from it
restores the task as last saved, with a warning that its latest state is missing.
When a checkpoint contains a task that can be resumed (including an interrupted one),
`build` rejects a changed agent configuration, and the error names the changed fields
(`agent 'coder' configuration changed since its checkpoint (instructions,
tool_config.available_tool_names changed); ...`). Workflow-injected tools and prepared prompt state are excluded from this
comparison, as are saved model registry details; provider, model ID, generation settings,
and other user configuration must remain compatible.
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

A tool that works for a long time can watch `TaskContext::stop`, which fires when
the workflow starts a graceful stop (or its agent is cancelled), save its state
and return early; `arko_agents::StopAwareCommand` does this for a child process.
See [stopping tool calls](../arko_agents/README.md#stopping-tool-calls).

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
