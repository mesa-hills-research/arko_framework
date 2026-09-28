# Arko Workflows API index

Reference for `arko_workflows` 0.1.0: the public Rust API, the optional Python API, the agent tools a workflow injects, configuration defaults, result contracts, and execution behavior. Workflow control flow is ordinary Rust or Python code; the crate supplies the orchestration primitives.

Signatures are declarations with their receivers, not complete implementations. Standard imports are abbreviated: `Arc`/`Weak`/`OnceLock` from `std::sync`, `HashMap`/`HashSet` from `std::collections`, `Path`/`PathBuf` from `std::path`, `Uuid` from `uuid`, `Value` from `serde_json`, and `Mutex`/`RwLock` from `parking_lot` unless qualified. `AgentConfig`, `Task`, `Tool`, `ToolError`, `ToolResultInterceptor`, and `InterceptionFailurePolicy` come from `arko_agents`; `LLMClient`, `Model`, and `UsageInfo` come from `arko_ai`. Builder setters consume and return `Self` unless shown otherwise. Derived standard traits are noted only where they matter to callers.

- [Imports and feature map](#imports-and-feature-map)
- [Workflow construction](#workflow-construction)
- [Workflow execution and lifecycle](#workflow-execution-and-lifecycle)
- [Result and event types](#result-and-event-types)
- [Hooks and human input](#hooks-and-human-input)
- [Workflow services and custom tools](#workflow-services-and-custom-tools)
- [Shared context](#shared-context)
- [Memory](#memory)
- [Messaging](#messaging)
- [Context freshness](#context-freshness)
- [Completion verification](#completion-verification)
- [Tool-result distillation](#tool-result-distillation)
- [Budget and cost accounting](#budget-and-cost-accounting)
- [Durable steps](#durable-steps)
- [Persistence and checkpoint state](#persistence-and-checkpoint-state)
- [Structured voting](#structured-voting)
- [Python bindings](#python-bindings)
- [Workflow errors](#workflow-errors)
- [Minimal composition and examples](#minimal-composition-and-examples)

## Imports and feature map

Source: [Cargo.toml](Cargo.toml), [crate exports](src/lib.rs).

No Cargo features are enabled by default. The `python` feature enables the PyO3 bindings and their Tokio bridge; see [Python bindings](#python-bindings).

Root re-exports: `Workflow`, `WorkflowBuilder`, `AgentHandle`, `WorkflowError`, `WorkflowCancelHandle`, `CancelCause`, `Cancellation`, `SupportedPlatform`, `WorkflowDependency`, `WorkflowMetadata`, `AgentResult`, `AgentRunStatus`, `WorkflowEvent`, `WorkflowResult`.

`use arko_workflows::prelude::*;` re-exports the groups below. Other public types are reached through their module paths.

| Module | Prelude exports |
| --- | --- |
| `workflow`, `workflow::cancel`, `workflow::config`, `workflow::result`, `error` | All root re-exports above **except `AgentHandle`** |
| `budget` | `AddCostResult`, `BudgetManager`, `CostEntry`, `CostLedger`, `CostSource` |
| `distillation` | `DistillationConfig`, `DistillationRecord` |
| `durable` | `DurableRetryPolicy`, `DurableStepContext`, `DurableStepSpec`, `DurableStepStatus`, `InDoubtContext`, `InDoubtResolution`, `ResolveInDoubt` |
| `freshness` | `ActorId`, `ContextFreshnessConfig`, `FreshnessConflict`, `ResourceId`, `ResourceTracker`, `ToolResourceEffect` |
| `hooks` | `HookCallback`, `HookPoint`, `HookResult` |
| `memory` | `AccessScope`, `MemoryConfig`, `MemoryEntry`, `MemoryTier`, `MemoryWrite`, `WriteMode` |
| `messaging`, `messaging::registry`, `messaging::wait` | `AgentMessage`, `MessageAttachment`, `MessagingPolicy`, `AgentStatus`, `WaitReason`, `WaitForMessageTimeout` |
| `persistence` | `EventLog`, `PersistenceConfig` |
| `policy` | `AgentPause`, `AgentResume`, `ApprovalPolicy`, `ErrorRecoveryPolicy`, `ToolApprovalHandler`, `ToolApprovalOutcome`, `ToolApprovalRequest` |
| `services` | `WorkflowServices`, `WorkflowServicesConfig` |
| `user_input` | `UserInputHandler`, `UserInputRequest` |
| `verification` | `CompletionVerifierConfig`, `RepairContextConfig`, `VERIFIER_OUTPUT_CONTRACT`, `VerificationDecision`, `VerificationKind`, `VerificationRecord` |
| `voting` | `BallotConfig`, `BallotResult`, `CastVote`, `TieBreak`, `VoteOption` |

Other public modules: `context`, `hooks::context`, `memory::tools`, `messaging::tools`, `messaging::wait`, `persistence::state`, and `python` (with the `python` feature). `workflow::engine` is public but exports no items; execution goes through `Workflow` and `AgentHandle`.

| Need | Entry point |
| --- | --- |
| Build and configure a workflow | `Workflow::builder()` / `WorkflowBuilder` |
| Sequential, parallel, or background agent execution | `run_agent`, `continue_agent`, `run_agents_parallel`, `spawn_agent` |
| Monitor progress or control the lifecycle | `subscribe_events`, hooks, `pause`, `resume`, `finish` |
| Stop a run from any task, or from a hook | `cancel_handle` / `WorkflowCancelHandle`, `HookResult::Abort` |
| Share structured text and deliver messages | `MemoryConfig`, `MemoryStore`, `MessagingPolicy`, `MessageBus` |
| See which agents are running, waiting, or finished | `agent_status`, `AgentStatus` |
| Track stale files, directories, memory, or custom resources | `ContextFreshnessConfig`, `ResourceTracker` |
| Gate completion and retain repair context | `CompletionVerifierConfig` |
| Condense selected tool outputs | `DistillationConfig`, custom tool result interceptors |
| Enforce and audit spending | Workflow and agent budgets, manual costs, `CostLedger` |
| Checkpoint, restore, or reconcile uncertain side effects | `PersistenceConfig`, durable steps |
| Let agents delegate or vote | `WorkflowServicesConfig`, `enable_voting` |
| Route human input | `UserInputHandler`, or handle `UserInputRequired` and continue |
| Decide tool approvals and error recovery | `approval_policy`, `error_recovery_policy`, `ToolApprovalHandler`, or handle `AgentPaused` and `resume_agent` |
| Add domain tools | `global_tool`, `agent_tool` |

## Workflow construction

Source: [workflow/mod.rs](src/workflow/mod.rs). `Workflow` and `WorkflowBuilder` are defined in `workflow` and re-exported at the root.

### Builder signatures and defaults

`Workflow::builder() -> WorkflowBuilder`, `WorkflowBuilder::new() -> Self`, and `WorkflowBuilder::default() -> Self` return an empty builder. Required: a name, an existing workspace directory, an `Arc<LLMClient>`, and at least one uniquely named agent. By default a workflow has no memory tiers, spending limits, hooks, persistence, freshness tracking, custom tools, interceptors, verifiers, distillers, workflow services, input handler, or voting; metadata is empty, the interception failure policy is `InterceptionFailurePolicy::UseRawResult`, the cancel grace period is 30 seconds, and `wait_for_message` waits up to an automatic limit inside each agent's prompt-cache lifetime. **Every tool call is approved** (`ApprovalPolicy::ApproveAll`) and **errors are recovered from automatically without an attempt limit** (`ErrorRecoveryPolicy::Automatic`), so a workflow runs without human input; see [Approvals and error recovery](#approvals-and-error-recovery).

| Signature | Parameters, purpose, and default |
| --- | --- |
| `name(self, name: impl Into<String>) -> Self` | Workflow identity used in results, hooks, and checkpoint name checks. Required. |
| `workspace(self, path: impl Into<PathBuf>) -> Self` | Working directory for every agent. Required; must already exist (the builder does not create it). `build` stores it absolute, with `.` and `..` resolved lexically (`std::path::absolute`, then `arko_agents::utils::normalize_path`); symbolic links are kept, and on Windows no verbatim `\\?\` form is produced (`std::fs::canonicalize` is not used). |
| `llm_client(self, client: Arc<LLMClient>) -> Self` | Model client shared by agents, verifiers, and distillers. Required. The builder reads no credentials itself; `LLMClient::from_env()` configures every provider found in the environment. |
| `agent(self, name: impl Into<String>, config: AgentConfig) -> Self` | Register an agent under a unique workflow registration name. At least one is required. |
| `metadata(self, metadata: WorkflowMetadata) -> Self` | Replace discovery metadata, including supported-host restrictions. |
| `memory(self, config: MemoryConfig) -> Self` | Shared memory tiers and access policies. Default: none. |
| `budget(self, limit: f64) -> Self` | Workflow cost limit in dollars; finite and nonnegative. Default: unlimited. |
| `budget_warning(self, threshold: f64) -> Self` | One-time warning threshold in dollars; finite and nonnegative. Default: none. |
| `agent_budget(self, agent: impl Into<String>, limit: f64) -> Self` | Limit on the named agent's cumulative LLM spend across all its runs; finite and nonnegative. Default: unlimited. |
| `hook(self, point: HookPoint, callback: HookCallback) -> Self` | Append a synchronous callback at a lifecycle point. Callbacks run in registration order. |
| `persistence(self, config: PersistenceConfig) -> Self` | Enable checkpoints, event and task files, a persistent run identity, and durable steps. |
| `resume_from(self, checkpoint: impl Into<PathBuf>) -> Self` | Restore state from a checkpoint JSON file (not a run directory) during `build`. Requires `persistence` whose run directory holds the checkpoint's `workflow_run_id`; a nil ID in the checkpoint is accepted. Memory access and limits come from the builder; see [Persistence and checkpoint state](#persistence-and-checkpoint-state). |
| `rename_memory_tier_on_resume(self, from: impl Into<String>, to: impl Into<String>) -> Self` | On resume, restore the checkpoint's memory tier `from` into the configured tier `to`. A `from` the checkpoint lacks is logged as a warning. |
| `drop_memory_tier_on_resume(self, name: impl Into<String>) -> Self` | On resume, discard the checkpoint's memory tier `name` instead of failing because it isn't configured. |
| `ephemeral(self) -> Self` | Clear the persistence and resume settings made so far. Later setters can enable them again. |
| `context_freshness(self, config: ContextFreshnessConfig) -> Self` | Enable versioned resource tracking. `ContextFreshnessConfig::default()` maps the built-in file tools `read_file`, `list_files`, `write_file`, and `apply_diff`. See [Context freshness](#context-freshness). |
| `global_tool(self, tool: Arc<dyn arko_agents::tool::Tool>) -> Self` | Register a tool for every agent and add its name to each agent's available tools. |
| `agent_tool(self, agent_name: impl Into<String>, tool: Arc<dyn arko_agents::tool::Tool>) -> Self` | Register a tool for one agent and add its name to that agent's available tools. |
| `tool_result_interceptor(self, interceptor: Arc<dyn ToolResultInterceptor>) -> Self` | Append an interceptor for every agent's tool results. |
| `agent_tool_result_interceptor(self, agent_name: impl Into<String>, interceptor: Arc<dyn ToolResultInterceptor>) -> Self` | Append an interceptor for one agent; it runs after the global interceptors. |
| `interception_failure_policy(self, policy: InterceptionFailurePolicy) -> Self` | Error handling for interceptors and distillers, applied by the agent runtime to each agent's chain as a whole. Default `UseRawResult`. |
| `completion_verifier(self, config: CompletionVerifierConfig) -> Self` | Append a gate on matching agents' `complete_task` calls. See [Completion verification](#completion-verification). |
| `distiller(self, config: DistillationConfig) -> Self` | Append an LLM distiller; distillers run after all custom interceptors. See [Tool-result distillation](#tool-result-distillation). |
| `workflow_services(self, config: WorkflowServicesConfig) -> Self` | Enable bounded agent-to-agent delegation through the `run_workflow_agent` tool. |
| `messaging(self, policy: MessagingPolicy) -> Self` | Set whom each agent may message with `send_message`. Default `MessagingPolicy::all()`: every agent may message every other agent. See [Messaging policy](#messaging-policy). |
| `disable_messaging(self) -> Self` | Offer no agent `send_message`, `list_recipients`, or `wait_for_message`. |
| `wait_for_message_timeout(self, timeout: impl Into<WaitForMessageTimeout>) -> Self` | How long `wait_for_message` waits at most, for agents without their own setting. Default `WaitForMessageTimeout::Auto`, just inside each agent's prompt-cache lifetime (4 min 30 s, or 54 min with Anthropic's 1-hour TTL); a `std::time::Duration` converts to a fixed limit, more than zero and at most 24 hours. See [Waiting for a message](#waiting-for-a-message). |
| `agent_wait_for_message_timeout(self, agent: impl Into<String>, timeout: impl Into<WaitForMessageTimeout>) -> Self` | The same for one agent, in place of the workflow's setting. |
| `user_input_handler(self, handler: Arc<dyn UserInputHandler>) -> Self` | Answer agents' `ask_user` questions asynchronously. Without a handler, the run suspends and returns `UserInputRequired`. |
| `approval_policy(self, policy: ApprovalPolicy) -> Self` | How tool-approval requests are answered. **Default `ApprovalPolicy::ApproveAll`: every call is approved**, including calls an agent's configuration marks for approval. See [Approvals and error recovery](#approvals-and-error-recovery). |
| `error_recovery_policy(self, policy: ErrorRecoveryPolicy) -> Self` | How error-recovery pauses are answered. **Default `ErrorRecoveryPolicy::Automatic`: recover without an attempt limit** from errors that may pass, and fail clearly on those that won't. |
| `enable_voting<I, S>(self, voters: I) -> Self` (`I: IntoIterator<Item = S>`, `S: Into<String>`) | Create an in-memory ballot store and offer the `cast_vote` tool to `voters` (registration names) only. |
| `max_parallel_agents(self, max: usize) -> Self` | Run at most `max` agents of a `run_agents_parallel` call at once; the others start in input order as places free up. Default: no cap. `build` rejects zero. Workflow services have their own limit (`max_active_calls`). |
| `fail_fast(self, enabled: bool) -> Self` | Opt-in, default off: in `run_agents_parallel`, the first agent that fails stops the others. |
| `cancel_grace_period(self, grace_period: std::time::Duration) -> Self` | How long a stopped run waits for each running agent to stop before aborting it. Default `workflow::cancel::DEFAULT_CANCEL_GRACE_PERIOD`, 30 seconds. See [Stopping a run](#stopping-a-run). |
| `build(self) -> Result<Workflow, WorkflowError>` | Validate, initialize or restore state, and fire `PreWorkflow`. May create persistence files; runs no agent. |

### Build validation and tool composition

`build` returns `BuilderMissing` when the name, workspace, client, or agents are missing, and `BuilderInvalid` for:

- a workspace that does not exist or is not a directory;
- a host that matches none of a nonempty `metadata.supported_platforms` list;
- duplicate agent names;
- a negative or non-finite workflow budget, warning threshold, or agent budget;
- an unknown agent name in `agent_tool`, `agent_tool_result_interceptor`, `agent_budget`, a completion verifier's verifier or targets, a distiller's targets (`distiller '<name>' targets unknown agent '<target>'`), the messaging policy (`messaging policy references unknown agent '<name>'`), `agent_wait_for_message_timeout` (`wait_for_message timeout references unknown agent '<name>'`), a memory tier's `AccessScope::Only` readers or writers (`memory tier '<tier>' readers name unknown agent '<name>'`), workflow services callers or targets (`workflow services caller '<name>' is not a registered agent`), or the voters (`voting names unknown agent '<name>'`);
- a completion verifier without a target (`completion verifier '<name>' has no target`), or a distiller without a target agent or a target tool;
- two memory tiers with the same name;
- a fixed `wait_for_message` limit of zero or more than 24 hours, for the workflow or an agent;
- `max_parallel_agents(0)`;
- a distiller whose `max_output_tokens` is zero or whose `reasoning` configuration is for a different provider than its model;
- `resume_from` without `persistence`, a checkpoint for a different workflow name, or a checkpoint run ID that differs from the persistence directory's.

Checkpoint read and parse failures return `Io` or `Json`. A warning threshold may exceed the budget limit. Tool names are only warned about, since MCP servers and skills add tools whose names aren't known at build: a tool the context freshness mapping or a distiller's targets name that no agent is configured with (`available_tool_names`), no workflow or custom tool has, and no `arko_agents` built-in has is logged as a warning. Workflow services that name no caller are logged as a warning too: no agent is offered `run_workflow_agent`. Metadata dependencies are descriptive: the builder neither installs nor checks executables.

Tool names: duplicate global tool names, duplicate names within one agent's tools, and a name shared by a global and an agent-local tool are rejected. Global tools cannot use the reserved names `complete_task`, `ask_user`, `read_memory`, `write_memory`, `list_memories`, `run_workflow_agent`, or `cast_vote`, nor `send_message` or `list_recipients` while any agent is offered the workflow's messaging tools, nor `wait_for_message` while any agent is offered it; an agent-local tool with one of these names is rejected for an agent that is offered the workflow's tool. Agent-local tools are not checked against reserved names: an agent-local tool with a reserved name replaces that workflow or built-in tool for its agent, except `complete_task`, which `arko_agents` never lets a non-built-in tool replace. Custom tools are registered after the workflow tools, so a custom tool named like an `arko_agents` built-in (for example `read_file`) replaces it for the agents that receive it.

Tools added automatically: `send_message` and `list_recipients` for each agent with at least one permitted recipient under the messaging policy; `wait_for_message` for each agent that another agent may message; `read_memory` for each agent that can read at least one memory tier, `write_memory` for each that can write at least one, and `list_memories` for each that can do either; `run_workflow_agent` for permitted callers when services are enabled; `cast_vote` for the voters named in `enable_voting`. The agent's pending messages are appended to the prompt of each new run and continuation, under `# Messages from other agents:`. A new run's initial prompt also lists the agents it may message; see [What agents are told about their recipients](#what-agents-are-told-about-their-recipients). Skills configured on an agent are initialized against the workspace. There is no separate graph or scheduler configuration.

Interceptors run in this order: global interceptors, the agent's own interceptors, then the distillers that target the agent, each group in registration order. Each stage receives the previous stage's result. The interception failure policy applies to each stage: a stage that fails is retried up to the policy's attempts (the stages before it don't run, or get paid for, again) and, if it still fails, skipped, so the next stage and then the agent get the output of the last stage that succeeded. A failure never undoes an earlier stage's work: an interceptor that redacts content keeps it redacted whatever fails after it. Under `InformAgent` and `RetryThenInformAgent` the result ends with a note that an interceptor failed. A stage fails open for its own work: an interceptor that fails can't transform what it was given, so one that must withhold content (for example to redact secrets) should replace the output with a placeholder rather than return an error when it can't process it, and should run before any interceptor that might fail. With a single interceptor, a failure after the last attempt delivers the original tool output. `load_skill` results are intercepted only when the agent's `SkillsConfig::load_skill_interception` opts in, under the policy it sets. Every agent's tool calls pass through the workflow's execution middleware, which enforces a stopped run and runs tool hooks, freshness checks, and completion verification at tool execution, so these also apply to automatically approved and batched calls.

### Workflow metadata

Source: [workflow/config.rs](src/workflow/config.rs). All four types implement Serde `Serialize` and `Deserialize`; only `WorkflowMetadata` implements `Default`. `WorkflowConfig` is not re-exported.

| Type | Public fields and meaning |
| --- | --- |
| `SupportedPlatform` | `os: String`, a Rust host OS name (`std::env::consts::OS`); `arch: Option<String>`, an optional architecture restriction (`None` when omitted in JSON). |
| `WorkflowDependency` | `name: String`, `description: String` (required); `executable`, `install_hint`, `os`, `arch`: `Option<String>`, each `None` when omitted in JSON. Descriptive only; nothing enforces it. |
| `WorkflowMetadata` | `description: String` and `version: String`, default `""`; `supported_platforms: Vec<SupportedPlatform>`, `required_dependencies: Vec<WorkflowDependency>`, and `tags: Vec<String>`, default empty. Every field may be omitted in JSON. An empty platform list means any host. |
| `WorkflowConfig` | `name: String`, `workspace: PathBuf`, `metadata: WorkflowMetadata` (default when omitted in JSON). A serializable descriptor; workflows are constructed with the builder. Unknown JSON fields are ignored. |

`SupportedPlatform` methods:

```rust
fn new(os: impl Into<String>) -> Self;
fn arch(self, arch: impl Into<String>) -> Self;
fn matches_host(&self) -> bool;
```

`new` sets no architecture restriction. `matches_host` compares the OS and, when set, the architecture case-insensitively with `std::env::consts::{OS, ARCH}`. Multiple supported platforms are alternatives.

## Workflow execution and lifecycle

Source: [workflow/mod.rs](src/workflow/mod.rs), [engine](src/workflow/engine.rs). All methods below are on `Workflow`.

### Run, continue, and spawn

| Signature | Parameters, result, and behavior |
| --- | --- |
| `async run_agent(&self, agent_name: &str, prompt: &str) -> Result<AgentResult, WorkflowError>` | Start a new invocation of a registered agent and await it. Adds pending messages and the list of agents it may message to the prompt, runs the managed driver, and records costs and the result. An unknown name returns `AgentNotFound`. |
| `async continue_agent(&self, previous: &AgentResult, prompt: &str) -> Result<AgentResult, WorkflowError>` | Resume a `Completed` or `WaitingForInput` result with its task, conversation, and run ID. `prompt` is the follow-up task or the user's answer; the messages queued for the agent that the task doesn't already hold are appended (see [Message bus](#message-bus)), and no recipient list is added. Context the task took from its command channel but did not add before it stopped (`Task::pending_context_injections`) is added at its first turn boundary, not presented again. Other statuses, or a task whose state does not match its status, return `BuilderInvalid`. The new result replaces the recorded result with the same run ID; token and cost totals are cumulative. |
| `async resume_agent(&self, previous: &AgentResult, resume: AgentResume) -> Result<AgentResult, WorkflowError>` | Continue a `Paused` result (returned as `AgentPaused`) with a decision: `ToolApprovals` for calls awaiting approval, `ContinueAfterError`, `FailAfterError`, or `ResolveRefusal` for an error-recovery pause. A result that isn't paused, or a decision for the other kind of pause (or naming a call that isn't awaiting approval), returns `BuilderInvalid`. Fires `PreAgentSpawn` and runs the managed driver like a continuation; messages queued while it was paused are delivered before the decision takes effect, and none it already holds is delivered again. |
| `agent_result(&self, agent_run_id: Uuid) -> Option<AgentResult>` | The recorded result of a run, such as the one `UserInputRequired` or `AgentPaused` names by `task_id`. |
| `async run_agents_parallel(&self, agents: Vec<(&str, String)>) -> Result<Vec<AgentResult>, WorkflowError>` | Run agent-name/prompt pairs concurrently; one agent may appear several times. All the agents are marked running before any is prompted, so each one's recipient list shows the others as running; under `max_parallel_agents`, at most that many run at once, and each is marked running as it starts (a waiting one is `not_started`). Waits for every invocation, then returns the results in input order, or the first failure in input order. An agent waiting for a place was admitted (`PreAgentSpawn`) when the call began; when it starts, the run's stop and the workflow budget are checked again, and either ends it before it starts (its error in its place). With `fail_fast`, the first failure instead stops the others (each is cancelled, winding down within the cancel grace period, and recorded `Cancelled` with a reason naming the failed agent; an agent inside one's tool call, such as a verifier, is aborted with it after the grace period), agents not yet started don't start and have no result, and the call returns that failure. An agent that pauses (waiting for input, or paused for a decision) is not a failure: its result comes back in its place with status `WaitingForInput` or `Paused`, and the others run on. Finalized outcomes are recorded either way. |
| `async spawn_agent(&self, agent_name: &str, prompt: &str) -> Result<AgentHandle, WorkflowError>` | Set up and start the agent and return its handle. Nothing drives its events until `wait` is called or the raw channels are used. |

All four return `BudgetExceeded` without starting an agent when the workflow budget is already exceeded. Once the run is stopped, all four return the stop's error (`Cancelled` or `AbortedByHook`) at once, without firing `PreAgentSpawn` or starting an agent; see [Stopping a run](#stopping-a-run).

The managed driver (used by `run_agent`, `continue_agent`, `resume_agent`, `run_agents_parallel`, and `AgentHandle::wait`) answers tool approvals under the workflow's `ApprovalPolicy` and error-recovery pauses under its `ErrorRecoveryPolicy` (see [Approvals and error recovery](#approvals-and-error-recovery)), and cancels the task when the workflow or agent budget is exceeded or the task is interrupted. A task it cancels ends `Cancelled`, and the call returns `BudgetExceeded`, `AgentBudgetExceeded`, or `AgentInterrupted`. The driver waits for a stopped agent at most the run's grace period, as described under [Stopping a run](#stopping-a-run).

A managed call returns the `AgentResult` of a completed task. Otherwise it returns the driver's error, or an error for the finalized status: `UserInputRequired` for `WaitingForInput`, `AgentPaused` for `Paused`, `AgentInterrupted` for `Interrupted`, `AgentCancelled` for `Cancelled`, and `Agent(ArkoAgentsError::TaskExecution)` with the failure reason for `Failed`. Once the run is stopped, the stop's error replaces any of these outcomes. An error can be returned after the invocation's `AgentResult` has been recorded; `result().agent_results` and `agent_result` hold finalized failures, cancellations, and pauses.

`run_agent`, `continue_agent`, `resume_agent`, and `run_agents_parallel` fire `OnWorkflowFailed` once before returning a failure: an error other than `Cancelled`, `UserInputRequired`, or `AgentPaused` (a pause is not a failure), including `AbortedByHook`. An `Abort` from that hook stops the run, and the call returns the stop's error; when the run was already stopped, its first cause's error is kept. Calls refused because the run is already stopped do not fire the hook, and neither do `spawn_agent` and `AgentHandle::wait`.

### Approvals and error recovery

Source: [policy.rs](src/policy.rs), [engine](src/workflow/engine.rs). The types are in the prelude.

**Tool approvals.** An agent asks for approval for a call its configuration doesn't auto-approve (`default_auto_approve_tool_use` and per-tool `auto_approve_tool_use`, which default to asking) and for a path outside the workspace under `PathSafetyPolicy::RequireApproval`; each request carries an `arko_agents::ApprovalReason` saying which.

| `ApprovalPolicy` | Behavior |
| --- | --- |
| `ApproveAll` (**default**) | Approves every call, so the workflow runs without human input. The first call each agent makes to each tool for each reason is logged as a warning naming the reason. |
| `RespectAgentConfig` | A call the agent's configuration marks for approval pauses the agent (it is suspended and recorded `Paused`); the call returns `AgentPaused` with `AgentPause::ToolApproval { calls }`, and `OnAgentPaused` fires. Decide with `resume_agent(&result, AgentResume::ToolApprovals(decisions))`; a call without a decision is rejected. |
| `Handler(Arc<dyn ToolApprovalHandler>)` | The handler decides each such call as it is asked: `ToolApprovalOutcome::Approve`, `Reject { reason }` (the agent is told), or `Pause` (as under `RespectAgentConfig`). It may wait on a person; a stopped run doesn't wait for it. |

```rust
#[async_trait::async_trait]
pub trait ToolApprovalHandler: Send + Sync {
    async fn decide(&self, request: ToolApprovalRequest) -> ToolApprovalOutcome;
}
// ToolApprovalRequest: workflow_run_id, agent_run_id, agent_name (registration name),
// tool_call_id, tool_name, parameters, tool_description, reason: ApprovalReason.
```

The tool hooks receive the reason: `HookContext::approval_reason` is set at `PreToolExecution` and `PostToolExecution` for a call that awaited approval.

**Error recovery.** The agent runtime recovers on its own first, under its `llm_error_policy` and `llm_retry_config` (the default retries a failed request with backoff, a rate limit after the provider's delay, until 10 attempts or the 5-minute elapsed budget run out, and never cuts off a request in flight). Only then, under the pausing error policies (the default), does the agent pause for error-recovery approval, with an `arko_agents::ErrorRecoveryCause`. The workflow's policy then decides; the driver never sends a model request itself, so nothing is retried twice.

| `ErrorRecoveryPolicy` | Behavior |
| --- | --- |
| `Automatic` (**default**) | Recovers without an attempt limit. `RetriesExhausted` (an outage, repeated rate limits): after a wait growing from 30 seconds to 15 minutes while pauses repeat without a model response in between, and at least the delay the provider asked for (`retry_after_ms`), the failed request is sent again unchanged, starting a new round of the agent's retries. `UsageLimit`: waits until the reported reset plus 5 seconds (or, with no reset reported, the growing wait; and when the limit is reported again right after a reset, at least the growing wait, so a reset time already past is not retried every few seconds) and sends the request again. Each wait gets up to 30 seconds (at most a quarter of it) added at random, so agents sharing a plan don't all resend at once. `ToolError` (the agent's opt-in `ToolErrorPolicy::PauseAndAwaitApproval`): continues at once; the calls' results report the errors. `NonRetryable`, `Refusal` (after the agent's opt-in `refusal_fallback`, if configured, could not answer; the same input would be refused again), and `UnusableResponses`: the task fails with a clear reason. `Monitor` (an opt-in plan-usage or cache-efficiency monitor's pause), `Middleware` (a completion its verifier could not review; the run call returns `VerifierFailed`), and an unrecorded cause: the agent pauses for the caller, as configured. Continuing adds nothing to the conversation (`UserCommand::ContinueAfterError` without a note), so the model is never told a user approved. While it waits, the agent's status is `waiting (retrying until …)` or `waiting (usage_limit until …)`. |
| `Limited { max }` | As `Automatic`, for at most `max` automatic recoveries in each call that drives the agent (a run, continuation or resume); the next such pause fails the task with a reason naming the limit. `max: 0` recovers nothing automatically. |
| `ReturnToCaller` | Every error-recovery pause pauses the agent: the call returns `AgentPaused` with `AgentPause::ErrorRecovery { error, cause }`. Continue with `resume_agent` and `AgentResume::ContinueAfterError { note }` (without a note the request is sent again unchanged; a note is added for the model first), `FailAfterError { reason }`, or, for a refusal, `ResolveRefusal(changes)`. |

An agent whose `llm_error_policy` is `RetryThenInformAgent` or `InformAgentAndContinue` doesn't pause: when its recovery ends it is interrupted, and the call returns `AgentInterrupted`.

**Nested runs can't pause.** An agent running inside another agent's tool call (a completion verifier, or an agent run with `run_workflow_agent`) can't be paused for the caller, since its parent is waiting on the call. Under `RespectAgentConfig` its `complete_task` is approved even when its configuration marks it for approval (as a default agent configuration does), since its parent waits for that result. Where it would otherwise pause (another call that needs approval under `RespectAgentConfig`, a handler's `Pause`, an error-recovery pause left to the caller, or an `ask_user` question without a user input handler), the workflow stops it instead: it ends `Cancelled`, with a reason saying what it needed. A verifier's run counts as a failed review (see [Completion verification](#completion-verification)); `run_workflow_agent` returns a tool error to the delegating agent. Under a `Handler` the handler decides its calls, `complete_task` included. Give such agents configurations that auto-approve the tools they use, or decide them with a `Handler`. (A delegated agent whose own completion its verifier could not review is the exception: it is left paused, and the delegating agent gets the `VerifierFailed` error at once.)

**Pauses.** A paused agent's task keeps its awaiting state (`TaskState::AwaitingToolApproval`, `AwaitingBatchToolApproval`, or `AwaitingErrorRecoveryApproval`), is recorded with status `AgentRunStatus::Paused` (`error` describes the pause), and is saved with the checkpoint, so it can be resumed after a restart. `OnAgentPaused` fires once per pause, and `OnWorkflowFailed` doesn't fire. `AgentPause::of(&task.current_state)` reads the pause from a task.

### AgentHandle

`AgentHandle` has no public constructor or fields; `spawn_agent` returns it.

```rust
fn agent_name(&self) -> &str;
async fn wait(self) -> Result<AgentResult, WorkflowError>;
fn into_parts(self) -> (
    tokio::sync::mpsc::Receiver<arko_agents::runtime::event::AgentEvent>,
    tokio::sync::mpsc::Sender<arko_agents::runtime::command::UserCommand>,
    tokio::task::JoinHandle<Result<arko_agents::Task, arko_agents::ArkoAgentsError>>,
    String,
);
```

`agent_name` returns the registration name. `wait` consumes the handle, runs the managed driver, finalizes and persists the task, and returns the completed result or the same errors as `run_agent`. Messages delivered in the agent's prompt are acknowledged when it completes, and the agent is unregistered from the message bus whatever the outcome. `into_parts` consumes the handle and returns the event receiver, command sender, task join handle, and agent name. A `wait_for_message` call such an agent is running ends at a message, a start or finish, its time limit, a workflow stop, or when the task is aborted, but not at a `CancelTask` the caller sends: the runtime reads commands only after the call returns. With the raw parts, the caller is responsible for event handling, approvals, usage accounting, event broadcasting and persistence, result recording, and message acknowledgment and unregistration (through `context().messages`). Execution middleware and interceptors are installed at spawn, so tool hooks, freshness checks, verification, and interception still run. A stopped run still reaches the agent: it is sent `CancelTask`, and the middleware cancels it at its next tool call. The caller must keep receiving its events until it ends. The workflow cannot observe the end of an agent driven through `into_parts`, so it counts as running for the rest of the run.

Dropping a handle without calling `wait` or `into_parts` aborts the agent's runtime task, unregisters it, and marks it finished (cancelled); no result is recorded.

### Results, state, costs, events, and pause

| Signature | Purpose and behavior |
| --- | --- |
| `context(&self) -> &Arc<SharedContext>` | Shared stores, recorded results, and extension state. See [Shared context](#shared-context). |
| `result(&self) -> WorkflowResult` | Snapshot of recorded results, metadata, the cost ledger, and the run's `cancellation`. Does not finish or checkpoint; also available once the run is stopped. |
| `agent_status(&self, agent_name: &str) -> Option<AgentStatus>` | A registered agent's status in this run, as `list_recipients` reports it; `None` for an unknown name. See [Agent statuses](#agent-statuses). |
| `cost_ledger(&self) -> CostLedger` | Clone of the chronological ledger of automatic and manual costs. |
| `add_manual_cost(&self, amount: f64, description: &str, metadata: Option<String>) -> Result<AddCostResult, WorkflowError>` | Record a finite, nonnegative dollar cost with a description and optional free-form metadata. Fires `OnBudgetWarning` and `OnBudgetExceeded` as applicable; a hook `Abort` stops the run and returns the stop's error. Returns `BudgetExceeded` when spending exceeds the limit; the cost stays recorded. An invalid amount returns `InvalidCost` and records nothing. |
| `subscribe_events(&self) -> tokio::sync::broadcast::Receiver<WorkflowEvent>` | Receive events that managed drivers emit after subscription; there is no replay. The buffer holds 1,024 events; slow receivers can lag. |
| `async checkpoint(&self) -> Result<(), WorkflowError>` | Save a checkpoint. Succeeds without writing when persistence is not configured. |
| `async finish(&self) -> Result<WorkflowResult, WorkflowError>` | Fire `PostWorkflow`, save a checkpoint, and return the result snapshot. Does not wait for outstanding agents or prevent later runs; each call repeats the hook and checkpoint. On a stopped run, saves a checkpoint without firing `PostWorkflow` and returns the stop's error. |
| `pause(&self) -> Result<(), WorkflowError>` | Set the pause flag, firing `OnWorkflowPause` only when the flag changes. Managed drivers stop before processing their next event; model and tool work already in progress continues. A pause does not delay stopping the run. |
| `resume(&self) -> Result<(), WorkflowError>` | Clear the flag, fire `OnWorkflowResume` when it changes, and wake paused drivers. |
| `is_paused(&self) -> bool` | Current pause flag. `false` after `build`, including after restoring a checkpoint. |

### Workflow-facing resource and ballot operations

```rust
fn register_resource_read(&self, actor: ActorId, resource: ResourceId)
    -> Result<(), WorkflowError>;
async fn mark_resource_dirty(&self, actor: ActorId, resource: ResourceId)
    -> Result<Vec<ActorId>, WorkflowError>;
fn begin_ballot(&self, config: BallotConfig) -> Result<Uuid, WorkflowError>;
fn finish_ballot(&self, ballot_id: Uuid) -> Result<BallotResult, WorkflowError>;
```

The resource methods require freshness tracking and return `BuilderInvalid` otherwise. `actor` is the observing or writing actor; `resource` identifies the context. `register_resource_read` records the actor's observation of the current version. `mark_resource_dirty` increments the version, returns the actors whose reads became stale, queues a notice for each active stale reader, and fires `OnFreshnessConflict` once per stale reader; a hook `Abort` stops the run and returns the stop's error. The version change is recorded before any notification or hook error is returned. See [Resource tracker](#resource-tracker) for identity and overlap rules.

The ballot methods require `enable_voting` and return `BuilderInvalid` otherwise. `begin_ballot` validates and stores the configuration and returns the new ballot's UUID; its eligible voters must be among the workflow's voters (`Deliberation` otherwise), and an empty set becomes all of them. `finish_ballot` tallies the current votes without closing the ballot. Validation and tally errors are returned as `WorkflowError::Deliberation`.

### Workflow-facing durable operations

These methods require configured persistence and return `DurableUnavailable` otherwise. See [Durable steps](#durable-steps) for identity, retry, replay, and resolution contracts.

```rust
async fn durable_step<I, O, F, Fut>(
    &self, spec: DurableStepSpec, input: &I, execute: F,
) -> Result<O, WorkflowError>
where
    I: serde::Serialize,
    O: serde::Serialize + serde::de::DeserializeOwned,
    F: FnOnce(DurableStepContext) -> Fut,
    Fut: std::future::Future<Output = Result<O, WorkflowError>>;

async fn durable_step_with_resolver<I, O, F, Fut, R>(
    &self, spec: DurableStepSpec, input: &I, execute: F, resolver: &R,
) -> Result<O, WorkflowError>
where
    I: serde::Serialize + Sync,
    O: serde::Serialize + serde::de::DeserializeOwned,
    F: FnOnce(DurableStepContext) -> Fut,
    Fut: std::future::Future<Output = Result<O, WorkflowError>>,
    R: ResolveInDoubt<I, O>;

async fn in_doubt_steps(&self) -> Result<Vec<InDoubtContext>, WorkflowError>;

async fn resolve_in_doubt_completed<I, O>(
    &self, spec: &DurableStepSpec, input: &I, output: &O,
    actor: &str, reason: &str,
) -> Result<(), WorkflowError>
where I: serde::Serialize, O: serde::Serialize;

async fn resolve_in_doubt_failed<I: serde::Serialize>(
    &self, spec: &DurableStepSpec, input: &I, actor: &str, reason: &str,
) -> Result<(), WorkflowError>;

async fn authorize_in_doubt_retry<I: serde::Serialize>(
    &self, spec: &DurableStepSpec, input: &I, actor: &str, reason: &str,
) -> Result<(), WorkflowError>;
```

`spec` names and versions the operation, `input` establishes its persisted input identity, and `execute` performs one attempt with the supplied context. A completed step returns its recorded `O` without executing again. `resolver` reconciles an uncertain earlier outcome where the retry policy requires it. `in_doubt_steps` lists uncertain attempts. `resolve_in_doubt_completed`, `resolve_in_doubt_failed`, and `authorize_in_doubt_retry` record an externally established output, a permanent failure, or permission for the next call to retry; `actor` and `reason` are audit strings, not authorization checks.

Once the run is stopped, `durable_step` and `durable_step_with_resolver` return the stop's error without journaling anything. A step already executing when the run stops is journaled as any attempt is: when its closure fails because of the stop (for example, its `run_agent` returned `Cancelled`), the attempt is recorded in doubt with the closure's error as the reason, and when the closure still succeeds, the step is recorded `Completed`. Either way the call returns the stop's error, not `DurableInDoubt` or the output. After `resume_from`, the step replays under its retry policy: `RetrySafe` runs it again, and otherwise it is resolved through a resolver or the operator methods above, which remain available on a stopped run.

### Stopping a run

Source: [workflow/cancel.rs](src/workflow/cancel.rs). A `Workflow` is one run. The run is stopped when a `WorkflowCancelHandle` cancels it or when any hook returns `HookResult::Abort` (see [Hook registration and results](#hook-registration-and-results)). The first cause stands, and a stopped run stays stopped: a hook's `Abort` on a run that is already stopped returns the first cause's error wherever it surfaces.

```rust
// Workflow
fn cancel_handle(&self) -> WorkflowCancelHandle;
fn cancel(&self) -> bool; // cancel_handle().cancel()
fn is_cancelled(&self) -> bool;
fn cancellation(&self) -> Option<Cancellation>;

// WorkflowCancelHandle: Clone, Debug, Send, Sync
fn cancel(&self) -> bool;
fn is_cancelled(&self) -> bool;
fn cancellation(&self) -> Option<Cancellation>;
async fn cancelled(&self);

// workflow::cancel
pub const DEFAULT_CANCEL_GRACE_PERIOD: std::time::Duration; // 30 seconds
```

`cancel` stops the run with `CancelCause::Requested` and fires `OnWorkflowCancelled` once, from the calling task, ignoring its result. It returns `false` and changes nothing when the run is already stopped, whether cancelled or aborted by a hook. `is_cancelled` is `true` once the run is stopped for either cause, `cancellation` returns the cause and time, and `cancelled` resolves once the run is stopped. A handle can be cloned and used from any task or thread; clones control the same run. It holds only a weak reference to the workflow's shared context.

| Type | Fields or variants and methods |
| --- | --- |
| `CancelCause` | `Requested` (a handle cancelled the run) or `AbortedByHook { hook_point: String, reason: String }`, where `hook_point` is the `HookPoint` variant name. `fn to_error(&self) -> WorkflowError` returns `Cancelled` or `AbortedByHook` with the same fields. `fn agent_reason(&self) -> String` returns `Workflow cancelled.` or `Workflow aborted by {hook_point} hook: {reason}`. Implements `Clone`, `PartialEq`, `Eq`, and Serde. |
| `Cancellation` | `cause: CancelCause` and `timestamp: chrono::DateTime<chrono::Utc>`, the time the run stopped. Implements `Clone`, `PartialEq`, and Serde. Recorded in `WorkflowResult::cancellation` and `WorkflowState::cancellation`. |

Stopping the run has these effects:

- Every running agent of the run is sent `CancelTask` at once, queued before its next step. This includes agents started with `spawn_agent`, whether driven by `AgentHandle::wait` or through `into_parts` (see [AgentHandle](#agenthandle)).
- Every run call in progress (`run_agent`, `continue_agent`, `run_agents_parallel`, `AgentHandle::wait`) waits for its agents to end while recording their remaining events (event journal and broadcast) and usage (budget), then returns the stop's error: `Cancelled` or `AbortedByHook`. It does so even when an agent completed during the wind-down; that agent's `Completed` result is still recorded. A `durable_step` or `durable_step_with_resolver` call in progress returns the stop's error once its closure returns. Every later call returns the error at once without starting anything: no `PreAgentSpawn` and no journal record.
- An agent stops at its next step; model streams and waits are interrupted at once, `wait_for_message` returns at once, and an agent running another long tool call stops when the call returns. The grace period counts from the moment the run stopped. An agent that has not ended within it is aborted: its Tokio task is dropped with any tool call in progress (an `arko_agents` command-template tool's child process is killed), and it is recorded as `Cancelled` with the task as the run started it, so the record does not contain this invocation's conversation; the event journal does. Its reason is the stop reason described below followed by `The agent did not stop within <grace period> and was aborted.`, with the configured period in `Duration` debug form (for example `30s`). The same bounded wait applies whenever the managed driver stops an agent, for example over budget, after an interruption, or after a hook's abort; when the run itself is not stopped, it counts from that moment. A grace period too long to represent waits as long as possible.
- An agent running inside another agent's tool call, such as a completion verifier or an agent delegated through the `run_workflow_agent` tool, gets half its parent's grace period at each level of nesting, so when the run stops it is stopped and recorded before its parent can be aborted. When only the parent is stopped (for example over its budget) and it doesn't end within the grace period, it is aborted with the agent running inside its tool call, which is not recorded. Dropping the driver of a running agent, for example a `run_agent` or `run_agents_parallel` future, aborts the agent's task instead of leaving it running unsupervised, and the agent counts as finished (cancelled); no result is recorded.
- A paused workflow does not delay the stop, and a `UserInputHandler` still waiting for an answer is abandoned.
- Once the run is stopped, the execution middleware answers every tool call with `ToolExecutionDecision::CancelTask`: the call does not run (a `cast_vote` is not recorded, for example), and the agent ends `Cancelled` with the cause's agent reason. A completion verifier stopped with the run decides nothing, so its target agent is cancelled, not failed. Distillers make no further requests and leave tool output unchanged.
- An agent the managed driver stops records why in its `TaskState::Cancelled` reason (and `AgentResult::error`), in place of the runtime's generic `Task cancelled by user.` (`arko_agents::task::CANCELLED_BY_USER`): the cause's agent reason when the run is stopped, and otherwise `Stopped by the workflow: <error>`, with the error the driver stopped it for (for example `BudgetExceeded`). The cancellation's `TaskStateChanged` event, as recorded in the event journal and broadcast during the wind-down, carries the same reason. An agent that ends because of the stop has status `AgentRunStatus::Cancelled`, and `PostAgentCancellation` fires for it.
- Budgets and limits are unchanged; usage reported while agents wind down is charged.
- Votes cast before the stop stand, and `finish_ballot` works as usual, although the quorum may be unmet.
- `finish` saves a checkpoint and returns the stop's error without firing `PostWorkflow`; `result` still returns the recorded results, with `cancellation` set. `OnWorkflowFailed` does not fire for `Cancelled`.

With persistence, every checkpoint written after the stop records it in `WorkflowState::cancellation`. With `auto_checkpoint`, one is saved as each stopped agent's result is finalized and once more when the first run call returns the stop's error; `checkpoint` and `finish` save one regardless of `auto_checkpoint`. The pause flag is not persisted. A stopped run's checkpoint is recognized by its `cancellation` field and the `Cancelled` status of its stopped agents.

Resuming with `resume_from` from a checkpoint that records a cancellation starts the run again: the new workflow is not stopped, and its checkpoints do not carry the earlier record. To keep a cancelled run stopped, check `WorkflowState::cancellation`, for example with `PersistenceManager::load_checkpoint`, before resuming. Durable steps left in doubt by the stop are handled as described under [Workflow-facing durable operations](#workflow-facing-durable-operations).

## Result and event types

Source: [workflow/result.rs](src/workflow/result.rs). These types are re-exported at the root and in the prelude, and implement Serde `Serialize` and `Deserialize`.

| Type | Public fields or variants |
| --- | --- |
| `AgentRunStatus` | `Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`, `Paused`. `is_paused()` is true for the last two: the run waits for input or a decision and can go on. `Paused` is a task suspended awaiting a tool approval or error-recovery decision the workflow's policies leave to the caller. `Cancelled` is the status of a task that ended in `arko_agents::TaskState::Cancelled`: stopped with the workflow run, stopped by the managed driver (over budget or after an interruption), or cancelled by any other `CancelTask` command or middleware decision. The status follows the task state, never the wording of a failure reason. |
| `AgentResult` | `agent_run_id: Uuid`, the stable task identity; `agent_name: String`, the registration name; `status: AgentRunStatus`; `summary: String`, the completion summary for `Completed` and empty otherwise; `error: Option<String>`, the failure, interruption, or cancellation reason, the unanswered questions, or a description of the pause; `task: arko_agents::Task`, with conversation, usage, and runtime state. |
| `WorkflowResult` | `workflow_run_id: Uuid`, `workflow_name: String`, `metadata: WorkflowMetadata`, `total_cost: f64` (running workflow total in dollars), `cost_ledger: CostLedger`, `agent_results: Vec<AgentResult>` (finalized outcomes; a continuation replaces the entry with its run ID), `cancellation: Option<Cancellation>` (set once the run is stopped; omitted from JSON when `None` and `None` when absent). |
| `WorkflowEvent` | `timestamp: chrono::DateTime<chrono::Utc>`, `workflow_run_id: Uuid`, `agent_name: String`, `agent_run_id: Uuid`, `event: arko_agents::runtime::event::AgentEvent`. Wraps each broadcast and persisted runtime event. |

`AgentResult` methods:

```rust
fn total_cost(&self) -> f64;
fn total_input_tokens(&self) -> usize;
fn total_output_tokens(&self) -> usize;
```

These return task-wide totals, including earlier conversation segments after a continuation. `total_cost` is the billed amount (`UsageInfo::billed()`), which is zero for subscription calls such as ChatGPT OAuth. The workflow total also includes manual and distillation charges, so the sum of task costs need not match the workflow ledger.

## Hooks and human input

Source: [hooks/mod.rs](src/hooks/mod.rs), [hooks/context.rs](src/hooks/context.rs), [user_input.rs](src/user_input.rs), [runtime integration](src/workflow/engine.rs).

### Hook registration and results

```rust
type HookCallback = Arc<
    dyn Fn(&hooks::context::HookContext) -> HookResult + Send + Sync + 'static
>;
enum HookResult { Continue, Abort(String), Skip }

// hooks::HookRegistry (also Default)
fn new() -> Self;
fn register(&mut self, point: HookPoint, callback: HookCallback);
fn get(&self, point: &HookPoint) -> Option<&Vec<HookCallback>>;
fn has_hooks(&self, point: &HookPoint) -> bool;
```

Callbacks are synchronous and receive an immutable context; capture shared state to act on it. `register` appends to a point's list, `get` borrows the ordered list, and `has_hooks` reports whether the list is nonempty. `Continue` proceeds to the next callback; the first `Skip` or `Abort` ends the chain for that point. `Skip` has an effect only at `PreToolExecution`, where it rejects the tool call with the message `Skipped by PreToolExecution hook`; at other points it only ends the chain. Hooks cannot replace tool output; use a tool result interceptor for that.

`Abort(reason)` at any point stops the whole run with `CancelCause::AbortedByHook { hook_point, reason }`, where `hook_point` is the point's variant name: running agents are cancelled, and run calls in progress or made later return `AbortedByHook`. On a run that is already stopped, the first cause stands, and its error is returned wherever the abort surfaces. See [Stopping a run](#stopping-a-run). Where the abort surfaces depends on the point:

- At lifecycle points (`PreAgentSpawn`, the `PostAgent*` points, `OnWorkflowPause`, `OnWorkflowResume`, `PostWorkflow`, `OnBudgetWarning` and `OnBudgetExceeded` from `add_manual_cost`, and others), the operation that fired the hook also returns the stop's error. The managed driver ignores the return value of budget hooks fired for automatic usage reports, including usage reported while agents wind down, but an `Abort` from them still stops the run. Budget hooks fired by a distillation charge also turn an `Abort` into an interceptor error.
- At `PreToolExecution`, the call does not run, and the agent ends `Cancelled` with the reason `Workflow aborted by PreToolExecution hook: <reason>`; the run call driving the agent returns `AbortedByHook` with `hook_point` `"PreToolExecution"`. `Skip` still rejects only that call.
- At `PostToolExecution`, and at `OnFreshnessConflict` fired from the execution middleware, the call has run and its result stands; the agent is sent `CancelTask` and ends before its next model request, and the run call returns `AbortedByHook`. The middleware never returns an error for a hook result.
- Hooks fired inside workflow tools (`OnAgentMessage` from `send_message`, `OnFreshnessConflict` from `write_memory`) return a tool error to the agent after the message or write has been made, and the run stops.

The result of `OnWorkflowCancelled` is ignored. Hooks do not roll back work already performed.

### HookPoint variants

| Variant | Trigger and populated context |
| --- | --- |
| `PreWorkflow` | End of a successful `build`, before the workflow is returned. |
| `PostWorkflow` | `finish`, before its checkpoint. |
| `PreAgentSpawn` | Before a run, spawn, or continuation is set up; not fired once the run is stopped. `agent_name`; for a continuation, also the prior `agent_run_id`. |
| `PostAgentCompletion` | A task finalized as completed. Agent name and run ID; `agent_summary` when nonempty. |
| `PostAgentFailure` | A task finalized as failed. Agent name, run ID, and `error_message`. |
| `PostAgentCancellation` | A task finalized as cancelled: stopped with the workflow run, stopped by the managed driver (over budget or after an interruption), or cancelled by any other `CancelTask` command or middleware decision. Agent name, run ID, and the task's reason in `error_message`. Cancelled tasks do not fire `PostAgentFailure`. |
| `PostAgentInterruption` | A task finalized as interrupted or in a nonterminal state. Agent name, run ID, and the reason in `error_message`. |
| `OnUserInputRequired` | A task finalized while awaiting user input. Agent name, run ID, and the questions in `error_message`. Not fired when a `UserInputHandler` answers the question. |
| `OnAgentPaused` | A task finalized paused for a decision the approval or error recovery policy leaves to the caller (`AgentRunStatus::Paused`); fires once per pause. Agent name, run ID, the pause's description in `error_message`, `metadata["pause"]` (`"tool_approval"` or `"error_recovery"`), and `metadata["cause"]` (such as `"retries_exhausted"`, `"monitor"`) or `metadata["tools"]` (each call's `tool_call_id`, `tool_name`, and `reason`). |
| `PreToolExecution` | Before a tool call executes. Agent name and run ID, `tool_name`, `tool_call_id`, and `tool_params`; `approval_reason` for a call that awaited approval. `Skip` rejects the call. |
| `PostToolExecution` | After a tool call, before result interception. The same fields, plus `tool_result` and `original_tool_result`, both holding the raw output or error text. |
| `PostToolResultInterception` | The managed driver receives an intercepted result. Agent name and run ID, tool name and call ID, the raw output in `original_tool_result`, and the delivered output in `effective_tool_result` and `tool_result`. |
| `OnWorkflowPause` | The pause flag changes from false to true. |
| `OnWorkflowResume` | The pause flag changes from true to false, before paused drivers are woken. |
| `OnWorkflowFailed` | `run_agent`, `continue_agent`, `resume_agent`, or `run_agents_parallel` is about to return a failure: an error other than `Cancelled`, `UserInputRequired`, or `AgentPaused`, including `AbortedByHook`. Fires at most once per call, also when several parallel agents fail. Not fired by a call refused because the run is already stopped. `error_message`; the agent name (and, for a continuation or resume, run ID) when available. |
| `OnWorkflowCancelled` | A `WorkflowCancelHandle` (or `Workflow::cancel`) cancelled the run. Fires once, from the task that called `cancel`; its result is ignored. A hook's abort does not fire it. |
| `OnBudgetExceeded` | Each automatic, manual, or distillation cost recorded while workflow spending strictly exceeds the limit. `budget_spent` is the new total. |
| `OnBudgetWarning` | The first recorded cost that brings spending to or above the warning threshold. `budget_spent` is the new total. |
| `OnAgentMessage` | The `send_message` tool queued a message. Sender name and run ID; `metadata["message_id"]`, `metadata["recipient"]`, and `metadata["content"]`. Direct `MessageBus::send` calls do not fire it. |
| `OnFreshnessConflict` | A successful tracked write (a mapped file tool, `write_memory`, or `mark_resource_dirty`) made another actor's read stale; fires once per stale reader. The reader's `agent_name` and `agent_run_id`, `metadata["resource"]` (the `ResourceId` in debug form, or `memory:<tier>`), and `metadata["writer"]` (the writer's agent name). A blocked write does not fire it. Its former name, `"OnMesiConflict"`, still deserializes to this variant. |

### HookContext fields

`hooks::context::HookContext` has these public fields. Optional fields are set only at the points listed above.

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | `chrono::DateTime<chrono::Utc>` | Hook invocation time. |
| `hook_point` | `HookPoint` | Current lifecycle point. |
| `workflow_run_id` | `Uuid` | Workflow run identity. |
| `workflow_name` | `String` | Configured name. |
| `agent_run_id` | `Option<Uuid>` | Task identity, when available. |
| `agent_name` | `Option<String>` | Agent registration name, when available. |
| `tool_name` | `Option<String>` | Tool being observed. |
| `tool_call_id` | `Option<Uuid>` | Specific tool invocation. |
| `tool_params` | `Option<HashMap<String, Value>>` | Tool input parameters. |
| `tool_result` | `Option<String>` | Tool output relevant to this point. |
| `original_tool_result` | `Option<String>` | Raw output before interception. |
| `effective_tool_result` | `Option<String>` | Output after interception. |
| `agent_summary` | `Option<String>` | Nonempty completion summary. |
| `approval_reason` | `Option<arko_agents::ApprovalReason>` | At `PreToolExecution` and `PostToolExecution`: why the call awaited approval, for a call that did. |
| `error_message` | `Option<String>` | Error, reason, or questions. |
| `budget_spent` | `f64` | Workflow dollars spent. |
| `budget_limit` | `Option<f64>` | Configured workflow limit. |
| `metadata` | `HashMap<String, Value>` | Point-specific data: `resource` and `writer` for `OnFreshnessConflict`; `message_id`, `recipient`, and `content` for `OnAgentMessage`; `pause` and `cause` or `tools` for `OnAgentPaused`. |

### UserInputHandler and UserInputRequest

```rust
#[async_trait::async_trait]
pub trait UserInputHandler: Send + Sync {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String>;
}
```

`UserInputRequest` has the fields `workflow_run_id: Uuid`, `agent_run_id: Uuid`, `agent_name: String`, and `questions: String`. Pass an `Arc<dyn UserInputHandler>` to `user_input_handler`. The returned string is submitted as the user's response; an `Err` string ends the invocation with `WorkflowError::Channel`.

Without a handler, the managed driver suspends the task on `ask_user` and returns `WorkflowError::UserInputRequired { agent, task_id, questions }`. To answer, take the matching `WaitingForInput` result from `workflow.result().agent_results` and pass it with the answer to `continue_agent`. With persistence, checkpoints include the suspended result and its task, so it can also be continued after `resume_from`.

## Workflow services and custom tools

Source: [services/mod.rs](src/services/mod.rs), [builder integration](src/workflow/mod.rs).

`WorkflowServicesConfig` (`Debug`, `Clone`, `Default`) configures agent-callable delegation:

| Field | Type | Default and meaning |
| --- | --- | --- |
| `allowed_callers` | `HashSet<String>` | The agents offered `run_workflow_agent`. Empty (default): none, unless `all_callers`. |
| `allowed_targets` | `HashSet<String>` | The agents a caller may run, independent of the caller list. Empty (default): none, unless `all_targets`. |
| `all_callers` | `bool` | Opt-in, default `false`: every agent is a caller, except completion-verifier agents. |
| `all_targets` | `bool` | Opt-in, default `false`: a caller may run every agent, except itself and completion-verifier agents. |
| `max_active_calls` | `u32` | `4`. Limit on simultaneous service calls, including nested delegation. |

```rust
// WorkflowServicesConfig
fn allow_caller(self, agent: impl Into<String>) -> Self;
fn allow_target(self, agent: impl Into<String>) -> Self;
fn allow_all_callers(self) -> Self;
fn allow_all_targets(self) -> Self;
fn max_active_calls(self, max: u32) -> Self;

// WorkflowServices: Clone
fn is_allowed(&self, caller: &str) -> bool;
fn is_target_allowed(&self, caller: &str, target: &str) -> bool;
async fn run_agent(&self, caller: &str, agent_name: &str, prompt: &str)
    -> Result<AgentResult, WorkflowError>;
```

Callers and targets must be named explicitly (`allow_caller`, `allow_target`), or admitted with the opt-ins, which leave out completion-verifier agents (and, as a target, the caller itself); an explicitly named agent is admitted either way (so a caller also named as a target may run itself). `max_active_calls` clamps to at least one; assigning the public field directly bypasses the clamp. `build` rejects a caller or target that is not a registered agent. The tool's `agent` parameter lists the caller's targets with their roles. `WorkflowServices` has no public constructor; after enabling services, `workflow.context().workflow_services.get()` returns it. It holds a weak reference to the workflow context and returns `Channel` once the workflow has been dropped.

`run_agent` checks the caller and target policies, then runs the target with the workflow's shared memory, budget, hooks, persistence, and completion verifiers. Once the run is stopped, it returns the stop's error. A disallowed caller or target returns `BuilderInvalid`; reaching the active-call limit returns `Channel` immediately instead of queuing. `caller` is a host-supplied identity, not authentication. Apart from the active-call limit, there is no call-depth or cycle limit.

Tool name constant: `services::RUN_WORKFLOW_AGENT_TOOL_NAME: &str = "run_workflow_agent"`.

| Tool | JSON parameters | Output |
| --- | --- | --- |
| `run_workflow_agent` | Required `agent: string` (an allowed target) and `prompt: string` (a bounded task). The `agent` parameter's description names the allowed targets with their role descriptions (every registered agent when no target is restricted); it depends only on the configuration. | The target's completion summary, after any configured verification. A missing string parameter is a `MissingParameter` error; workflow failures become `ExecutionFailed`. Not batch-safe. |

Custom domain tools implement `arko_agents::tool::Tool` and are registered with `global_tool` or `agent_tool`, which also add their names to the receiving agents' available tools. Use a `ToolResultInterceptor` to transform output and the freshness APIs to record custom resource effects. These traits are defined and documented in `arko_agents`.

## Shared context

Source: [context/mod.rs](src/context/mod.rs). `context::SharedContext`, returned by `Workflow::context`, exposes the public fields below and the messaging methods listed with the `messaging` field; it has no public constructor. The lock-protected fields expose live workflow state; prefer the `Workflow` methods where they add hooks, notifications, or accounting.

| Field | Type | Purpose |
| --- | --- | --- |
| `workflow_run_id` | `Uuid` | Persistent or ephemeral run identity. |
| `workflow_name` | `String` | Workflow name. |
| `workflow_metadata` | `WorkflowMetadata` | Discovery metadata. |
| `workspace` | `PathBuf` | Shared working directory. |
| `budget` | `BudgetManager` | Ledger and limits. |
| `durable_store` | `Option<Arc<durable::DurableStore>>` | Present when persistence is configured. |
| `memory` | `memory::MemoryStore` | Shared tier store. |
| `messages` | `messaging::MessageBus` | Queued and live messaging. |
| `messaging` | `Option<messaging::MessagingPolicy>` | Whom each agent may message with `send_message`; `None` when messaging is disabled. `fn message_recipients(&self, agent_name: &str) -> Vec<String>` returns an agent's permitted recipients, sorted by name (empty when disabled). `fn recipient_statuses(&self, agent_name: &str) -> Vec<(String, AgentStatus)>` pairs them with their current statuses, and `fn recipient_status(&self, agent: &str) -> AgentStatus` returns one agent's status (`NotStarted` for an unknown name). `fn message_senders(&self, agent_name: &str) -> Vec<String>` returns the agents that may message it, sorted by name (empty when disabled); an agent with any is offered `wait_for_message`. |
| `resource_tracker` | `Option<freshness::ResourceTracker>` | Context freshness tracker. |
| `hooks` | `Arc<hooks::HookRegistry>` | Registered callbacks. |
| `agent_configs` | `HashMap<String, AgentConfig>` | Agent configurations keyed by registration name. |
| `global_tools` | `Vec<Arc<dyn arko_agents::tool::Tool>>` | Tools for every agent. |
| `agent_tools` | `HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>` | Agent-specific tools. |
| `agent_results` | `RwLock<Vec<AgentResult>>` | Finalized results, including suspended and failed runs. |
| `global_interceptors` | `Vec<Arc<dyn ToolResultInterceptor>>` | Global interceptor chain. |
| `agent_interceptors` | `HashMap<String, Vec<Arc<dyn ToolResultInterceptor>>>` | Per-agent chains. |
| `interception_failure_policy` | `InterceptionFailurePolicy` | Interceptor error handling. |
| `completion_verifiers` | `Vec<verification::CompletionVerifierConfig>` | Completion gates in order. |
| `verification_attempts` | `Mutex<HashMap<(Uuid, String), (u32, u32)>>` | `(minor, major)` rejection counts per target task and verifier name. |
| `verification_records` | `RwLock<Vec<verification::VerificationRecord>>` | Verification audit history. |
| `distillation_configs` | `Vec<distillation::DistillationConfig>` | Distillers in order. |
| `distillation_records` | `RwLock<Vec<distillation::DistillationRecord>>` | Distillation audit history. |
| `workflow_services` | `OnceLock<services::WorkflowServices>` | Set only when services are enabled. |
| `paused` | `std::sync::atomic::AtomicBool` | Driver pause flag. |
| `resume_notify` | `tokio::sync::Notify` | Wakes paused drivers. |
| `user_input_handler` | `Option<Arc<dyn user_input::UserInputHandler>>` | Human response routing. |
| `ballots` | `Option<Arc<voting::BallotStore>>` | In-memory structured votes. |
| `event_tx` | `tokio::sync::broadcast::Sender<WorkflowEvent>` | Live managed event stream. |

## Memory

Module: `arko_workflows::memory`. Sources: [store and configuration](src/memory/mod.rs), [agent tools](src/memory/tools.rs), [read responses](src/memory/reads.rs).

Memory provides named tiers with independent read and write access, append or replace writes, write history, and optional size limits (none by default; opt in per tier with `max_bytes`, `global_tier_with_limit`, or `broadcast_tier_with_limit`). `MemoryStore` and `MemoryConfig` default to empty. Below, `DateTime<Utc>` is `chrono::DateTime<chrono::Utc>`.

| Public type | Fields or variants; purpose and defaults |
| --- | --- |
| `AccessScope` | `All`, `Only(HashSet<String>)`, `None`. Grants access by exact agent registration name. `fn allows(&self, agent_name: &str) -> bool` checks it. |
| `WriteMode` | `AppendOnly` (the Rust default) or `FullEdit`. An append adds a newline first when the content is nonempty; a full edit replaces the content. Both add a history entry. |
| `MemoryEntry` | `id: Uuid`, `author: String`, `task_id: Option<Uuid>`, `timestamp: DateTime<Utc>`, `content: String`. One write with its text, as `MemoryStore::entries` and `MemoryTier::history` return it. `task_id` defaults to `None` when omitted in JSON. |
| `MemoryWrite` | `id: Uuid`, `author: String`, `task_id: Option<Uuid>`, `timestamp: DateTime<Utc>`, `content: Option<String>`, `range: Option<(usize, usize)>`. One write as the tier stores it: each successful write records a new UUID and the current UTC time, and its text is stored once, as a byte range of the tier's `content` while it is there (every append, and the latest full edit) or in `content` (an earlier version of a full-edit tier). `content` and `range` are omitted from JSON when `None`. |
| `MemoryTier` | `name: String`, `content: String`, `readers: AccessScope`, `writers: AccessScope`, `write_mode: WriteMode`, `entries: Vec<MemoryWrite>`, `max_bytes: Option<usize>`. When omitted in JSON, `entries` defaults to `[]` and `max_bytes` to `None` (no limit). No `Default` implementation. `fn text_of(&self, write: &MemoryWrite) -> &str` and `fn history(&self) -> Vec<MemoryEntry>` read the texts. A tier saved by an earlier version, whose entries each held their text beside the content, loads converted: an append-only tier's writes are matched from the end of its content back, a full-edit tier's latest write to its content; the history reads the same. |

These five types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`. Enums serialize with their Rust variant names, for example `"AppendOnly"`, `"All"`, or `{"Only":["writer"]}`. `MemoryTier.write_mode` is required in JSON even though `WriteMode` has a Rust default.

### Memory configuration

Methods on `MemoryConfig`; setters consume and return the configuration.

| Signature | Purpose and parameters |
| --- | --- |
| `fn new() -> Self` | Empty configuration. |
| `fn tier(self, tier: MemoryTier) -> Self` | Add a fully specified tier, including initial content, policies, history, and byte limit. |
| `fn global_tier(self, name: impl Into<String>) -> Self` | Add an empty append-only tier that every agent can read and write; no limit. |
| `fn broadcast_tier(self, name: impl Into<String>, writers: impl IntoIterator<Item = impl Into<String>>) -> Self` | Add an empty append-only tier that every agent can read and only the named agents can write; no limit. |
| `fn global_tier_with_limit(self, name: impl Into<String>, max_bytes: usize) -> Self` | As `global_tier`, with a limit on the resulting UTF-8 content size in bytes. |
| `fn broadcast_tier_with_limit(self, name: impl Into<String>, writers: impl IntoIterator<Item = impl Into<String>>, max_bytes: usize) -> Self` | As `broadcast_tier`, with a byte limit. |
| `fn tiers(&self) -> &[MemoryTier]` | The configured tiers. |
| `fn build(self) -> MemoryStore` | Construct the store. `WorkflowBuilder::build` rejects duplicate tier names and access sets naming unregistered agents. |

### Memory store

Methods on `MemoryStore`. Read and write methods check the named agent's access to the tier.

| Signature | Purpose, result, and constraints |
| --- | --- |
| `fn new(tiers: Vec<MemoryTier>) -> Self` | Create a store from tier definitions; with duplicate names, the last wins. Initial content and history are kept as supplied, with each write's text stored once. |
| `fn is_empty(&self) -> bool` | Whether the store has no tiers. |
| `fn read(&self, agent_name: &str, tier_name: &str) -> Result<String, WorkflowError>` | Clone of the full current content. |
| `fn read_versioned(&self, agent_name: &str, tier_name: &str) -> Result<(usize, String), WorkflowError>` | Atomically read `(version, content)`; the version is the tier's history-entry count. |
| `fn read_window(&self, agent_name: &str, tier_name: &str, max_chars: Option<usize>) -> Result<String, WorkflowError>` | The full content, or, when longer than `max_chars` Unicode scalar values, its final `max_chars` characters prefixed with `[showing final {max_chars} of {count} characters]\n`. Accepts zero. |
| `fn entries(&self, agent_name: &str, tier_name: &str) -> Result<Vec<MemoryEntry>, WorkflowError>` | Clone of the write history (requires read access). Full-edit writes keep earlier history. |
| `fn write(&self, agent_name: &str, tier_name: &str, content: &str) -> Result<(), WorkflowError>` | Apply the tier's write mode and add a history entry without a task ID. |
| `fn write_with_context(&self, agent_name: &str, task_id: Option<Uuid>, tier_name: &str, content: &str) -> Result<(), WorkflowError>` | As `write`, recording the task ID in history. |
| `fn list_accessible(&self, agent_name: &str) -> Vec<(String, bool, bool)>` | `(tier_name, can_read, can_write)` for each tier the agent can read or write, sorted by name; tiers it can't access are left out. |
| `fn access_of(&self, agent_name: &str) -> (bool, bool)` | Whether the agent can read any tier, and whether it can write any: which memory tools it is offered. |
| `fn snapshot(&self) -> Vec<MemoryTier>` | Clone of all tiers, policies, content, and history, without access checks. Order is unspecified. |

An unknown tier returns `MemoryTierNotFound`; a denied operation returns `MemoryAccessDenied`; a write whose resulting content would exceed the tier's byte limit returns `MemoryLimitExceeded` and changes nothing. The newline added by an append counts toward the limit; history size is not limited. Direct store calls perform no freshness checks and send no notifications; the memory tools provide that integration.

### Memory tools and read JSON contract

`memory::tools` defines `ReadMemoryTool`, `WriteMemoryTool`, and `ListMemoriesTool`. Each has `fn new(context: Arc<crate::context::SharedContext>, agent_name: String) -> Self` and implements `Debug` and `arko_agents::tool::Tool`, including:

```rust
fn definition(&self) -> arko_agents::tool::ToolDefinition;
async fn execute(
    &self,
    task_context: &arko_agents::tool::TaskContext<'_>,
    params: HashMap<String, serde_json::Value>,
) -> Result<String, arko_agents::tool::ToolError>;
```

| Constant (`pub const …: &str`) and tool name | Input parameters | Output and behavior |
| --- | --- | --- |
| `READ_MEMORY_TOOL_NAME = "read_memory"` | Required `tier: string`; optional `full: boolean` (default `false`; the strings `"true"` and `"false"` are also accepted); optional `max_chars: string` holding a positive integer. | Versioned JSON, described below. Batch-safe. Requires read access. When freshness tracking is enabled, a complete read records a read of the tier. |
| `WRITE_MEMORY_TOOL_NAME = "write_memory"` | Required `tier: string` and `content: string`. | `Successfully wrote to memory tier '{tier}'.` Applies the tier's policy and records the current task ID. With freshness tracking, a write based on a stale read is rejected when `block_on_conflict` is set and allowed otherwise; after a successful write, each stale reader gets a notice if active, and `OnFreshnessConflict` fires for every stale reader, active or not. Not batch-safe. |
| `LIST_MEMORIES_TOOL_NAME = "list_memories"` | None. | `Available memory tiers:` followed by each tier the agent can access, sorted by name, with `read/write`, `read-only`, or `write-only`; tiers it can't access are left out. `No memory tiers are available to you.` when there are none. Batch-safe. |

Every successful `read_memory` response is a JSON object with `tier: string`, `version: integer`, and `kind`; the other fields depend on `kind`:

| `kind` | Additional fields | How to consume it |
| --- | --- | --- |
| `"full"` | `content: string` | Replace the local view with this complete snapshot. |
| `"append"` | `base_version: integer`, `content: string` | Append `content` exactly to the snapshot at `base_version`; it includes any required newline. |
| `"replace"` | `base_version: integer`, `start_line: integer`, `delete_lines: integer`, `content: string` | In the snapshot at `base_version`, replace `delete_lines` lines starting at zero-based `start_line` with `content`. Lines keep their newline terminators, as with Rust `split_inclusive('\n')`, so concatenating the lines yields the exact current content. |
| `"unchanged"` | None. | Keep the prior content and adopt the returned version, which can advance even when the text did not change. |
| `"partial"` | `total_chars: integer`, `content: string` | The trailing `max_chars` Unicode scalar values of a longer snapshot. An incomplete view, not a delta. |

The first complete read of a tier in a task is `"full"`. Later reads compare against the latest complete response confirmed as delivered into that task's context. A change is expressed as one exact append or line replacement; a `"full"` response is returned instead when the encoded delta would be at least as large. An unchanged tier always yields `"unchanged"`. `full=true` requests a complete snapshot, but a truncating `max_chars` still takes precedence and yields `"partial"`. Partial responses neither advance the baseline nor count as a complete read for freshness. Responses that are prepared but not yet delivered do not advance the baseline.

`ReadMemoryTool` also implements these `Tool` methods:

```rust
fn allow_result_interception(&self) -> bool; // false: interceptors never alter read results
fn reset_context(&self, task_id: uuid::Uuid); // discard this task's read baselines and pending reads
fn observe_context(&self, task_id: uuid::Uuid, message: &arko_ai::Message);
```

`observe_context` confirms delivery when a user-role message carries a successful matching `read_memory` tool result or text containing a pending response. When driving these tools outside the workflow engine, forward context observations and resets to keep deltas correct. Parameter errors are `MissingParameter` or `ParameterValidation`; store and freshness failures are `ExecutionFailed`. A hook's `Abort` after a write is returned as an `ExecutionFailed` error even though memory has already changed, and it stops the run.

## Messaging

Module: `arko_workflows::messaging`. Sources: [message bus, data, and policy](src/messaging/mod.rs), [agent statuses](src/messaging/registry.rs), [agent tools and start notices](src/messaging/tools.rs).

Messages are addressed to an agent name in a fixed roster. They stay queued until drained or acknowledged. Every active task of the recipient receives a live injection at its next safe turn boundary, in a `# Workflow context update` user message; a run that starts receives the queued messages in its prompt, and live any message queued while it was starting. A message reaches each run of the agent once: a continued or resumed run is never presented a message it already holds. Below, `UserCommand` is `arko_agents::runtime::command::UserCommand`.

| Public type | Fields or variants |
| --- | --- |
| `AgentMessage` | `id: Uuid`, `from: String`, `from_task_id: Option<Uuid>`, `to: String`, `content: String`, `attachments: Vec<MessageAttachment>`, `timestamp: chrono::DateTime<chrono::Utc>`, `to_task_id: Option<Uuid>` (the run a context notice is for), `held_by: Vec<Uuid>` (the runs that stopped holding it; see [Message bus](#message-bus)). When omitted in JSON, `id` is the nil UUID, `from_task_id` and `to_task_id` are `None`, and `attachments` and `held_by` are `[]`; `to_task_id` and `held_by` are omitted from JSON when empty. New messages get a fresh UUID and the current UTC time. `is_held_by(task_id)` and `is_for(task_id)` (not held by the run, and not another run's notice) read them. |
| `MessageAttachment` | `File { path: String }`, `MemoryTier { tier: String }`, `Transcript { task_id: Uuid }`, `Text { name: String, content: String }`. Typed references or inline text; the bus does not read the referenced files, tiers, or transcripts. |

Both types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`. Attachments serialize with an internal `type` tag in snake case:

```json
[
  {"type":"file","path":"src/lib.rs"},
  {"type":"memory_tier","tier":"notes"},
  {"type":"transcript","task_id":"00000000-0000-0000-0000-000000000001"},
  {"type":"text","name":"summary","content":"Review the parser."}
]
```

### Message bus

Methods on `MessageBus`. `Default` is an empty roster with no queues or active recipients.

| Signature | Purpose and behavior |
| --- | --- |
| `fn new(roster: impl IntoIterator<Item = String>) -> Self` | Bus with the given recipient names and empty queues. |
| `fn from_snapshot(roster: impl IntoIterator<Item = String>, queues: HashMap<String, Vec<AgentMessage>>) -> Self` | Restore pending queues. Active registrations are not restored. |
| `fn roster(&self) -> Vec<String>` | Recipient names, sorted. |
| `fn register_active(&self, agent_name: &str, task_id: Uuid, sender: tokio::sync::mpsc::Sender<UserCommand>)` | Register a task's command channel and start its delivery worker. Registering the same task again replaces the registration and stops the old worker. Can be called outside a Tokio runtime; delivery then starts at the next `send` or `inject_task` inside one. |
| `fn unregister_active(&self, agent_name: &str, task_id: Uuid)` | Remove the registration and stop forwarding; unacknowledged messages stay queued. |
| `async fn inject_task(&self, task_id: Uuid, content: String) -> bool` | Queue a context notice for one active task without waiting for channel capacity. Creates an `AgentMessage` from `workflow-context` (`messaging::WORKFLOW_CONTEXT_SENDER`) that checkpoints persist like any message; it doesn't end a `wait_for_message`. `true` means queued (not yet consumed or saved); an unknown or inactive task returns `false`. |
| `async fn send(&self, from: &str, from_task_id: Option<Uuid>, to: &str, content: String, attachments: Vec<MessageAttachment>) -> Result<AgentMessage, String>` | Validate the recipient, queue the message, and return it. Active delivery workers forward it without blocking the sender on a busy recipient. A message to the sender itself returns `'<to>' is not a permitted recipient`; an unknown recipient's error lists the valid names. The messaging policy and recipient statuses are not checked. A closed channel unregisters that task; its unacknowledged messages stay queued. The sender name is not validated. |
| `fn drain(&self, agent_name: &str) -> Vec<AgentMessage>` | Remove and return the agent's whole queue; `[]` for an unknown or empty name. |
| `fn pending(&self, agent_name: &str) -> Vec<AgentMessage>` | Clone of the whole queue without removing messages. |
| `fn pending_for(&self, agent_name: &str, task_id: Uuid) -> Vec<AgentMessage>` | The queued messages for run `task_id` (`AgentMessage::is_for`): those it doesn't hold, other than another run's context notices. What its prompt presents. |
| `fn acknowledge(&self, agent_name: &str, message_ids: &[Uuid])` | Remove the given IDs from the agent's queue. |
| `fn snapshot(&self) -> HashMap<String, Vec<AgentMessage>>` | Clone of the pending queues, for persistence. |

Queueing, channel acceptance, and acknowledgment are separate steps. When a run stops, the engine records what it holds before checkpointing: the queued messages its conversation holds (presented in one of its prompts, or added as a complete runtime `# Workflow context update`; an echo in an assistant turn, a tool result, or a quoted ID doesn't count) and those it took from its command channel but hadn't added (`Task::pending_context_injections`). Their `AgentMessage::held_by` lists the run, so a continuation or resume of it never presents them again (the context it took is added from the task, once). A run that completed acknowledges the messages in its conversation, which leave the queue; one it only took stays queued. Until acknowledged, a message is still presented to every other run of the agent. A context notice (`to_task_id` set) is for its run only and is dropped when that run stops, whatever the outcome. Messages reach disk only in a checkpoint; returning from `send` or `inject_task` is not a disk commit. A message can be delivered again to a run after interrupted work (a run stopped without being recorded), so consumers that need deduplication should use its stable ID. `MessageBus::send` fires no hooks; the `send_message` tool does.

### Messaging policy

`MessagingPolicy` sets whom each agent may message with the `send_message` tool. It implements `Debug`, `Clone`, and `Default`.

| Item | Behavior |
| --- | --- |
| `recipients: HashMap<String, HashSet<String>>` | For each agent listed, the only agents it may message (none when the set is empty). An agent without an entry may message every other registered agent. |
| `fn all() -> Self` | No restrictions; the same as `default()` and the builder's default. |
| `fn can_message<I, S>(self, agent: impl Into<String>, recipients: I) -> Self` (`I: IntoIterator<Item = S>`, `S: Into<String>`) | Restrict `agent` to `recipients`; an empty list means no one. Replaces an earlier setting for the agent. |
| `fn cannot_message(self, agent: impl Into<String>) -> Self` | `agent` may message no one. |
| `fn permits(&self, sender: &str, recipient: &str) -> bool` | Whether the policy lets `sender` message `recipient`; never itself. Does not check that either name is registered. |
| `fn recipients_of(&self, sender: &str, roster: &[String]) -> Vec<String>` | The roster agents `sender` may message, in roster order, never `sender` itself. |

An agent is offered `send_message` and `list_recipients` only when it has at least one permitted recipient: a single-agent workflow, an agent restricted to no one, and every agent of a workflow built with `disable_messaging` get neither. The offer depends on the policy alone, not on recipient statuses. Names are registration names, and `build` rejects a policy naming an unregistered agent. The policy is read whenever an agent run starts, so it applies to runs started through workflow services and to resumed runs alike. Message queues restored from a checkpoint are delivered whatever the policy; it governs only new `send_message` calls, not `MessageBus::send`.

```rust
use arko_workflows::messaging::MessagingPolicy;

// `worker_a` may message only `lead`; `auditor` may message no one; every
// other agent may message every other agent.
let policy = MessagingPolicy::all()
    .can_message("worker_a", ["lead"])
    .cannot_message("auditor");
```

### Agent statuses

One registry per workflow run records each agent's status, under one lock, with a version that every status change increases. `AgentStatus` and `WaitReason` are defined in `messaging::registry` and re-exported in the prelude. Both implement `Debug`, `Clone`, `PartialEq`, `Eq`, and Serde, with variant names in snake case.

| Type | Variants and meaning |
| --- | --- |
| `AgentStatus` | `NotStarted`: no run of the agent has started. `Running`: a run of the agent is running. `Waiting(WaitReason)`: every running run of the agent is waiting. `Finished(AgentRunStatus)`: no run is running, and the latest ended with this outcome. |
| `WaitReason` | `RateLimited { until: Option<chrono::DateTime<chrono::Utc>> }`: waiting out a provider rate limit. `Retrying { until }`: waiting to retry another failed model request. `UsageLimit { until }`: waiting for a plan usage limit to reset. `AwaitingApproval`: a tool call or error recovery awaits approval. `AwaitingInput`: questions await an answer. `AwaitingMessage`: waiting in `wait_for_message`. `Paused`: the workflow is paused. |

`AgentStatus` methods:

```rust
fn is_reachable(&self) -> bool; // Running or Waiting: messages are delivered
fn describe(&self) -> String;
```

`describe` returns the form agents are shown: `not_started`, `running`, `waiting (<reason>)` or `waiting (<reason> until <time>)`, and `finished (<outcome>)`. The reasons are `rate_limited`, `retrying`, `usage_limit`, `awaiting_approval`, `awaiting_input`, `awaiting_message`, and `paused`; the time is RFC 3339 UTC to the second, for example `waiting (rate_limited until 2026-09-27T12:00:00Z)`. The outcomes are `completed`, `failed`, `interrupted`, `cancelled`, `waiting_for_input`, and `paused`.

Statuses follow the managed driver's events:

- `RetryScheduled` starts a `RateLimited` wait when its `rate_limit` flag is set and a `Retrying` wait otherwise, until the current time plus its delay.
- `UsageLimitReached` starts a `UsageLimit` wait until its `waiting_until_unix_ms`, when known.
- A task state awaiting tool, batch, or error-recovery approval starts `AwaitingApproval`; `AwaitingUserResponse` and `UserQuestionsAsked` start `AwaitingInput`.
- Error reports, task logs, plan-usage and cache-efficiency reports, and the stream end leave a wait in place; any other event ends it.
- `wait_for_message` marks its run `AwaitingMessage` for as long as it waits, whatever events arrive meanwhile, and running again when it returns or its task is aborted.

A wait whose `until` time has passed counts as over: the agent is `Running` again while the retry is under way. While the workflow is paused, every agent with a running run is `Waiting(Paused)`. With several runs of one agent, the agent is `Running` while any run is not waiting; otherwise it is waiting for the reason of one of its runs. A run started while another run of the same agent is running is not a new start.

`Workflow::agent_status`, `SharedContext::recipient_statuses`, and `list_recipients` read the registry. An agent driven through `AgentHandle::into_parts` counts as running for the rest of the run, and one whose `AgentHandle` is dropped unused, or whose run call is dropped before it ends, counts as finished (cancelled). Checkpoints save the registry; see [Checkpoint payloads](#checkpoint-payloads).

### What agents are told about their recipients

An agent learns about the agents it may message in exactly four places. The workflow adds nothing about them to its system prompt or tool definitions, so both stay the same during a task and remain cacheable.

1. **The initial prompt of a new run** ends with a `# Agents you can message` section when the agent has permitted recipients: a sentence on how to reach them, then one line per recipient with its role description and its status when the run started, such as `` - `lead` (Coordinates the work): running ``. The agent's pending messages come before it. Continuations (`continue_agent`, including an answer to `ask_user`, and a verifier's re-review or format retry) add no list. A completion verifier's prompt has no list, and the target's task prompt that it quotes has its list removed.
2. **Start notices** arrive in `# Workflow context update` user messages, added at each safe turn boundary before a model request, one line per recipient: `` Agent `worker` (role: <role description>) is now running. You can message it with send_message. `` A run is told about every permitted recipient that started after the registry version it last saw (from its list or its previous notices) and is still running, and then has seen the current version, so no ordering of starts loses or repeats a notice. A continued run is told about recipients that started since it last ran, and a run whose task comes from a checkpoint saved without the registry is told about every running recipient once. Agents started together by `run_agents_parallel` are marked running before any is prompted, so each sees the others as running and gets no notice. There are no finish notices.
3. **`list_recipients`** reports current statuses on request.
4. **`wait_for_message`** results name an agent whose start or finish ended the wait; see [Waiting for a message](#waiting-for-a-message). The start notice still follows at the next turn boundary.

### Messaging tools

`messaging::tools` defines `SEND_MESSAGE_TOOL_NAME: &str = "send_message"` and `LIST_RECIPIENTS_TOOL_NAME: &str = "list_recipients"`, and re-exports `WAIT_FOR_MESSAGE_TOOL_NAME: &str = "wait_for_message"` and `WaitForMessageTool` from `messaging::wait`. `SendMessageTool`, `ListRecipientsTool` and `WaitForMessageTool` implement `Debug` and `arko_agents::tool::Tool`; `WaitForMessageTool::new` also takes the `Arc<arko_ai::LLMClient>` whose `prompt_cache_lifetime` sets the automatic limit:

```rust
fn new(context: Arc<crate::context::SharedContext>, agent_name: String) -> Self;
fn definition(&self) -> arko_agents::tool::ToolDefinition;
async fn execute(
    &self,
    task_context: &arko_agents::tool::TaskContext<'_>,
    params: HashMap<String, serde_json::Value>,
) -> Result<String, arko_agents::tool::ToolError>;
```

Their definitions name no agent, so they are the same for every agent and never change during a task.

| Tool | Parameters | Output and behavior |
| --- | --- | --- |
| `send_message` | Required `to: string` and `content: string`; optional `attachments_json: string`, a JSON array in the attachment format above (default: no attachments). | Delivers a message to a permitted recipient that is running or waiting. Not batch-safe. |
| `list_recipients` | None. | One line per permitted recipient, never the agent itself, in the initial prompt's format with current statuses; `There are no agents you can message.` when there are none. Batch-safe. |
| `wait_for_message` | Optional `timeout_seconds: string`, a positive number of seconds (a JSON number is accepted too). | Waits for a message; see [Waiting for a message](#waiting-for-a-message). Offered to each agent that another agent may message, whether or not it may send. Not batch-safe. |

`send_message` checks its parameters first: a missing `to` or `content` is a `MissingParameter` error, invalid attachment JSON is a `ParameterValidation` error, and a non-string `attachments_json` is ignored. A `to` that is not a permitted recipient, including the agent itself and unknown names, is a `ParameterValidation` error that names only the permitted recipients: `'<to>' is not an agent you can message. The agents you can message: <recipients>.` The recipient's status decides the rest:

- `not_started`: an `ExecutionFailed` error, `Agent '<to>' has not started, so it can't receive messages. A workflow context update will tell you when it starts.` Nothing is queued for later.
- `finished (<outcome>)`: an `ExecutionFailed` error, `Agent '<to>' is finished (<outcome>); it no longer receives messages.`
- `running`: the message is queued and delivered at the recipient's next safe turn boundary; the tool returns `Message <id> sent to agent '<to>'.`
- `waiting (<reason>)`: the message is queued and delivered likewise, and the result says so, for example `Message <id> sent to agent 'lead', which is waiting (usage_limit); it receives the message when it continues.` A recipient `waiting (awaiting_message)` stops waiting at once, and the result is the plain `Message <id> sent to agent '<to>'.`

On success the tool records the current task as the sender's task and fires `OnAgentMessage`. A hook's `Abort` is returned as an `ExecutionFailed` error even though the message has already been queued, and it stops the run.

### Waiting for a message

Source: [messaging/wait.rs](src/messaging/wait.rs). An agent in native tool mode must call a tool every turn; `wait_for_message` lets it wait for a reply without polling. Its description and results are kept short, since repeated waits add them to the context every time; to wait longer, the agent calls it again. It is offered to each agent that another agent may message (`SharedContext::message_senders` is not empty), whether or not the agent may send. Its definition names no agent and no time, so it never changes during a task.

A wait returns on the first of these, checked when it is called and whenever something changes:

- **A message.** An agent message for the run that its conversation doesn't hold yet: one sent after the run started and not yet added by a context update. Messages in the initial prompt, messages already delivered (which stay queued until the run completes), and `workflow-context` notices don't count. The result names the senders, ``Message from `lead` waiting.`` or ``Messages from `lead` and `worker_b` waiting.``, followed by any start or finish as below. The message itself is never in the result: it is delivered once, through the usual context update, and acknowledged as before. The tool returns only once the message is in the run's command channel (it waits up to a second for the delivery worker, which normally takes microseconds), and the runtime takes every queued command at a turn boundary, so the message reaches the very next request.
- **A start or finish** of an agent the run may message or that may message it: `` `worker` started.`` or `` `worker` finished (completed).`` Other status changes, such as another agent's rate-limit wait, don't end a wait. The start notice (see [What agents are told about their recipients](#what-agents-are-told-about-their-recipients)) still comes at the next turn boundary.
- **The time limit**: `No messages after 270 s.`, stating the wait used (the limit, or the shorter `timeout_seconds`). Durations read like `270 s`, `54 min`, or `0.5 s`.
- **A stop**: `Workflow stopping.` once the run is cancelled or aborted by a hook, and `Run stopping.` once the managed driver stops this run (over budget, or after an interruption). Either ends the wait at once, not at its limit; the agent then stops at the turn boundary. The runtime reads its `CancelTask` only after a tool call returns, so the wait watches the stop itself.

While it waits, the agent's status is `waiting (awaiting_message)`, which `list_recipients`, rosters, and `Workflow::agent_status` show; a `send_message` to it ends the wait. The status returns to running when the wait ends, also when the agent's task is aborted mid-wait. A `timeout_seconds` that is not a positive number is a `ParameterValidation` error. Nothing is persisted: the waiting status is not part of a checkpoint.

The time limit:

```rust
// messaging::wait; WaitForMessageTimeout is also in the prelude.
pub enum WaitForMessageTimeout {
    Auto,                      // the default
    Fixed(std::time::Duration), // also From<Duration>
}
fn resolve(self, lifetime: arko_ai::PromptCacheLifetime) -> std::time::Duration;
pub const DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT: std::time::Duration; // 270 seconds
pub const MAX_WAIT_FOR_MESSAGE_TIMEOUT: std::time::Duration; // 24 hours
```

`WaitForMessageTimeout` implements `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, and `Default`.

By default (`Auto`) a wait lasts up to about 4.5 minutes. Each wake-up is a cheap turn that keeps the provider's prompt cache warm, so a long wait is simply several short waits: the agent calls `wait_for_message` again. With Anthropic's 1-hour cache TTL (`GenerationConfig::anthropic_cache_ttl` set to `AnthropicCacheTtl::OneHour`, also for Claude on OpenRouter) the default stretches to about 54 minutes. Precisely, `Auto` takes the agent's cache lifetime from `LLMClient::prompt_cache_lifetime` (5 minutes for Anthropic and, conservatively, for OpenAI; 1 hour with the 1-hour TTL), subtracts a tenth of it, at least 30 seconds, and counts from the agent's last model request, when the cache was last used; an unknown lifetime or caching off gives `DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT`.

A longer wait can be set with `Fixed(duration)`, for the workflow or one agent (more than zero, at most 24 hours; it counts from the start of the wait). Provider caches often last longer than 5 minutes (OpenAI documents about 30, and a test over ChatGPT OAuth still hit after 40), but that isn't guaranteed, and a miss re-reads the whole context at full price. The model's `timeout_seconds` shortens a wait, never lengthens it.

## Context freshness

Module: `arko_workflows::freshness`. Source: [freshness tracking](src/freshness/mod.rs).

Version-based tracking detects stale reads of files, directory trees, memory tiers, and workflow-defined resources. Actors include the task ID, so concurrent runs of the same agent are tracked separately. The tracker records explicit observations and mutations; it does not watch the filesystem for outside changes.

| Public type | Fields or variants; purpose |
| --- | --- |
| `ResourceId` | `File(PathBuf)`, `DirectoryTree(PathBuf)`, `MemoryTier(String)`, `Custom { namespace: String, key: String }`. Resource identity. |
| `ActorId` | `task_id: Uuid`, `agent_name: String`. Constructor: `fn new(task_id: Uuid, agent_name: impl Into<String>) -> Self`. |
| `ToolResourceEffect` | `ReadFile { parameter: String }`, `ReadDirectoryTree { parameter: String }`, `WriteFile { parameter: String }`. `parameter` names the tool argument that holds the path string. |
| `ContextFreshnessConfig` | `block_on_conflict: bool`, `tool_effects: HashMap<String, ToolResourceEffect>`. The default blocks conflicts and uses the mappings below. |
| `FreshnessConflict` | `resource: ResourceId`, `observed_version: u64`, `current_version: u64`, `last_writer: Option<ActorId>`. A read that has become stale. |
| `FreshnessSnapshot` | Opaque tracking state (private fields); `Default` is empty. `ResourceTracker::snapshot` and `from_snapshot` round-trip it, including reads and writes. |

All six types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`; `ResourceId` and `ActorId` also implement `PartialEq`, `Eq`, and `Hash`, and `ContextFreshnessConfig` and `FreshnessSnapshot` implement `Default`. Enums serialize with their Rust variant names, for example `{"MemoryTier":"notes"}` and `{"ReadFile":{"parameter":"path"}}`.

| Default tool mapping | Effect and argument |
| --- | --- |
| `read_file` | `ReadFile { parameter: "path" }` |
| `list_files` | `ReadDirectoryTree { parameter: "directory" }` |
| `write_file`, `apply_diff` | `WriteFile { parameter: "path" }` |

`ContextFreshnessConfig::tool_effect(self, tool_name: impl Into<String>, effect: ToolResourceEffect) -> Self` adds or replaces a mapping; assign `tool_effects` to replace them all. Fields omitted in JSON do not take the Rust defaults.

With a tracker configured, the workflow checks each mapped write before the tool runs. When it overlaps context the agent read that another actor has since changed, the call is rejected with feedback to re-read if `block_on_conflict` is set, and allowed otherwise. After a successful mapped tool call, the read or write is recorded; for a write, every active stale reader receives a notice, and `OnFreshnessConflict` fires once per stale reader. An `Abort` from that hook stops the run; the write's own result stands.

**Finished readers.** When a task stops, a task that can still be continued (completed, waiting for input, or paused) keeps its reads but no longer receives notices or fires `OnFreshnessConflict`, and a notice queued for it and not yet read is dropped; any other task's reads are forgotten. When it is continued (`continue_agent`) or resumed (`resume_agent`), it gets one message listing the resources it read that another actor changed since, advising it to re-read any of them before editing it or relying on it: at the end of the continuation prompt, or as a context update before the resumed turn. Its kept reads still block a stale write under `block_on_conflict` until it reads again. After a restart from a checkpoint no task runs, so every recorded reader counts as finished until its task is continued.

### Resource tracker

Methods on `ResourceTracker`.

| Signature | Purpose and behavior |
| --- | --- |
| `fn new(workspace: PathBuf, config: ContextFreshnessConfig) -> Self` | Empty tracker rooted at the workspace. |
| `fn from_snapshot(workspace: PathBuf, config: ContextFreshnessConfig, snapshot: FreshnessSnapshot) -> Self` | Restore tracking state with the given workspace and configuration. Every recorded reader counts as finished. |
| `fn snapshot(&self) -> FreshnessSnapshot` | Clone of all tracking state. |
| `fn config(&self) -> &ContextFreshnessConfig` | The configuration. |
| `fn resource_for_tool(&self, tool_name: &str, parameters: &HashMap<String, Value>) -> Option<(ResourceId, bool)>` | Resolve a mapped tool's path argument; the boolean is `true` for a write. `None` for an unmapped tool, a missing or non-string argument, or a path no tool can open (one with a NUL byte). Paths resolve as the built-in file tools open them and the agent runtime's workspace check judges them, with `arko_agents::utils::resolve_path`: surrounding whitespace is trimmed, `~` and `$VAR` are not expanded, a relative path is joined to the workspace, and `.` and `..` are resolved lexically. Symlinks are not resolved. |
| `fn record_read(&self, actor: &ActorId, resource: ResourceId)` | Record the resource's current version as the actor's observation, replacing an earlier one. A new resource starts at version zero. |
| `fn check_write(&self, actor: &ActorId, resource: &ResourceId) -> Vec<FreshnessConflict>` | The actor's overlapping observations whose resource has since reached a newer version last written by another actor. No prior read means no conflict. Reports only; it does not block anything. |
| `fn record_write(&self, actor: &ActorId, resource: ResourceId) -> Vec<ActorId>` | Record a successful mutation: increment the affected versions, set the last writer, and return the other actors with affected reads, except finished ones (see `finish_actor`). A file write also increments already tracked directory trees that contain it. |
| `fn clear_actor(&self, actor: &ActorId)` | Remove the actor's observations; its writes and the resource versions remain. |
| `fn finish_actor(&self, actor: &ActorId, continued: bool)` | The actor's task stopped: with `continued`, keep its reads but leave it out of `record_write`'s result (no notices); otherwise `clear_actor`. The engine calls this when it records a task. |
| `fn resume_actor(&self, actor: &ActorId) -> Vec<ResourceId>` | The actor's task runs again: it is included in `record_write` again, and the resources it read that another actor changed since are returned, sorted (files, directory trees, memory tiers, custom). |

Resources overlap when they are identical, or when a directory tree that was read contains a written file. Memory and custom resources overlap only by exact identity. `record_read` and `record_write` use their arguments as supplied, so resources constructed manually need consistently normalized paths.

## Completion verification

Source: [verification/mod.rs](src/verification/mod.rs); execution: [engine](src/workflow/engine.rs). All public types and the protocol constant are in the prelude.

A `CompletionVerifierConfig` registered with `WorkflowBuilder::completion_verifier` gates an agent's `complete_task` call on a review by another registered agent. Configurations run in registration order. A rejection returns the verifier's feedback to the agent for repair; a correctness gate fails the task once its cap is exceeded. Targets are registration names. A configuration never reviews its own verifier agent, even when that agent is listed as a target. Targets are required: `build` rejects a configuration without one, so a verifier never attaches to agents it wasn't aimed at (or to other verifiers). Use distinct configuration names: repair counts and conversation reuse are keyed by target task ID and configuration name. A review in progress when the run is stopped decides nothing: the target agent is cancelled with the run, not failed.

### `CompletionVerifierConfig`, `VerificationKind`, and `RepairContextConfig`

`CompletionVerifierConfig` implements `Debug`, `Clone`, `Serialize`, and `Deserialize`. Public fields:

| Field | Type | Purpose and constructor default |
| --- | --- | --- |
| `name` | `String` | Configuration name; appears in records and feedback. |
| `verifier_agent` | `String` | Registered agent that reviews the completion. |
| `target_agents` | `HashSet<String>` | Target registration names; empty from the constructors, and at least one is required by `build`. |
| `kind` | `VerificationKind` | `Correctness`, or advisory `Efficiency`; set by the constructor. |
| `instructions` | `String` | Reviewer instructions; defaults below. |
| `max_minor_rejections` | `u32` | `3` for correctness, `2` for efficiency. |
| `max_major_rejections` | `u32` | `0` for both kinds. |
| `reuse_conversation` | `bool` | `false`; also `false` when omitted in JSON. |
| `repair_context` | `Option<RepairContextConfig>` | `None`; also `None` when omitted in JSON. |
| `include_prior_findings` | `bool` | `true` (also when omitted in JSON): a new review of a task after an earlier verdict (without conversation reuse) is given that verdict and its findings. |
| `max_format_retries` | `u32` | `1` (also when omitted in JSON): how many times an unparseable verdict is asked for again before the review counts as failed. |
| `max_review_retries` | `u32` | `1` (also when omitted in JSON): how many failed reviews of one task are retried before the task pauses and the run call returns `VerifierFailed`. |

`VerificationKind` has the variants `Correctness` and `Efficiency` and implements `Copy`, `Clone`, `PartialEq`, `Eq`, and Serde. `RepairContextConfig` implements `Clone`, `PartialEq`, `Eq`, and Serde and has one public field, `max_bytes: usize`, the byte budget for repair-delta text. Neither configuration implements `Default`.

| Signature on `CompletionVerifierConfig` | Parameters and result |
| --- | --- |
| `pub fn correctness(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self` | Correctness gate. Default instructions: "Check correctness, completeness, and maintainability. Inspect the actual workspace rather than trusting the candidate summary." |
| `pub fn efficiency(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self` | Advisory efficiency gate. Default instructions: "Look for avoidable repeated work, poor asymptotic behavior, unnecessary I/O or API calls, and needless resource use. Do not demand speculative optimization." |
| `pub fn target(mut self, agent: impl Into<String>) -> Self` | Add a target registration name; duplicates collapse. |
| `pub fn instructions(mut self, instructions: impl Into<String>) -> Self` | Replace the reviewer instructions. |
| `pub fn minor_rejection_cap(mut self, cap: u32) -> Self` | Number of minor rejections allowed. |
| `pub fn major_rejection_cap(mut self, cap: u32) -> Self` | Number of major rejections allowed (correctness only). |
| `pub fn reuse_conversation(mut self, enabled: bool) -> Self` | Continue the earlier verifier conversation when the same target task is reviewed again and that verifier result is available; otherwise start a new review. |
| `pub fn repair_context(mut self, max_bytes: usize) -> Self` | Enable repair-delta injection with `max_bytes.max(1_024)`. Does not enable conversation reuse, which repair deltas require. |
| `pub fn include_prior_findings(mut self, enabled: bool) -> Self` | Opt out (`false`) of giving a new review the prior findings. |
| `pub fn format_retries(mut self, retries: u32) -> Self` | Set `max_format_retries`. |
| `pub fn review_retries(mut self, retries: u32) -> Self` | Set `max_review_retries`. |

A correctness gate allows exactly the configured number of minor and of major rejections and fails the task on the next rejection of that severity; the two counts are independent, so a major cap of zero fails on the first major rejection. An efficiency gate treats every major rejection as minor, never uses the major cap, and stops running once its minor count reaches the cap; a minor cap of zero disables it. **A failed review doesn't fail the target.** A review fails when the verifier's run fails (it ends with an error, or pauses) or its verdict is still unusable after `max_format_retries` requests to restate it. Within `max_review_retries` failed reviews of a task, the completion is rejected with a message saying the verifier could not review it and nothing was decided, so the agent calls `complete_task` again and a new review runs. On the next failure the completion is neither accepted nor rejected: the agent's `complete_task` call is answered that nothing was decided, and the task pauses for the caller (status `Paused`, `OnAgentPaused`; an error-recovery pause with the cause `Middleware`, also inside another agent's call), and the run call returns `WorkflowError::VerifierFailed { verifier, agent, task_id, reason }`. Continue it with `continue_agent` on its result (`Workflow::agent_result(task_id)`): the prompt is the agent's next message, and its next `complete_task` is reviewed again. `resume_agent` with `AgentResume::ContinueAfterError` does the same, and `AgentResume::FailAfterError` fails it. The counts are kept per target task and verifier, and cleared when the task completes or pauses this way.

With conversation reuse, the continued review receives the previous verdict, the new completion summary, and focused re-review instructions. Without it, a new review of a task that already has a verdict from the same configuration is given that verdict and its findings and asked to check whether each is resolved, unless `include_prior_findings` is off; the first review of a task is a full first pass. A repair delta is added only when context freshness tracking is enabled; it covers tracked files inside the workspace and new memory entries the verifier can read, falling back to truncation or content hashes when text is unavailable. It does not enumerate directory-tree or custom resources and does not replace inspecting the affected artifacts. `repair_context` clamps the byte budget; setting the field directly or deserializing does not.

### Verdict protocol and records

`pub const VERIFIER_OUTPUT_CONTRACT: &str` holds the plain-text response instructions included in verifier prompts. The first non-empty line of the verifier's completion summary must be `ACCEPTED`, `REJECTED_MINOR`, or `REJECTED_MAJOR`. Rejections require non-empty feedback on the following lines; acceptance may omit it. Leading blank lines, whitespace around the decision, and CRLF line endings are accepted. JSON, Markdown wrappers, and feedback on the decision line are not; everything after the decision line is opaque feedback. A malformed response gets `max_format_retries` continuations (one by default) asking for a corrected format; if it is still unusable, the review has failed (see above).

`VerificationDecision` implements `Clone`, `PartialEq`, `Eq`, and Serde:

| Rust variant | Serde value | Meaning |
| --- | --- | --- |
| `Approved` | `"approved"` | Protocol `ACCEPTED`; the gate passes. |
| `RejectedMinor` | `"rejected_minor"` | Bounded corrections requested. |
| `RejectedMajor` | `"rejected_major"` | Major correctness issue; efficiency gates record it as `RejectedMinor`. |

`VerificationRecord` implements `Clone` and Serde. Public fields:

```rust
pub timestamp: chrono::DateTime<chrono::Utc>;
pub target_task_id: uuid::Uuid;
pub target_agent: String;
pub verifier_name: String;
pub verifier_agent: String;
pub verifier_task_id: uuid::Uuid;
pub decision: VerificationDecision;
pub feedback: String;
```

A record identifies the reviewed task and agent, the configuration, the reviewing agent and its run, the time, the effective decision (after the efficiency conversion), and the trimmed feedback. Records are available in `SharedContext::verification_records` and in checkpoints. Repair counts and delta baselines for a task are cleared when it completes successfully.

## Tool-result distillation

Source: [distillation/mod.rs](src/distillation/mod.rs); registration and order: [engine](src/workflow/engine.rs). Both public types implement `Clone` and Serde and are in the prelude.

A `DistillationConfig` registered with `WorkflowBuilder::distiller` replaces large matching tool results with an LLM-written summary. Each distiller is installed only for the agents it targets, and distills only the tools it targets; `build` requires at least one of each. Any tool whose results may be intercepted can be targeted, `read_file` included (for a large file the agent wants specific details from). Distillers run after the global and per-agent interceptors, in registration order; each sees the result of the stages before it, and all of a distiller's filters must match. Once the run is stopped, a distiller makes no request and returns the tool output unchanged. A distiller calls the workflow's LLM client with its model, reasoning, and output-token limit, in a fresh conversation with prompt caching turned off where the provider allows it (each request is a one-off ending with a tool output no later request repeats, so a cache write would never be read; ChatGPT OAuth caches implicitly and is asked nothing), giving the agent's initial task prompt, the tool name and parameters, and the current tool output as context. Results of tools that opt out of interception (`Tool::allow_result_interception`, for example `read_memory`) are never distilled.

### `DistillationConfig`

| Public field | Type | Purpose and default from `new` |
| --- | --- | --- |
| `name` | `String` | Configuration name used in records, errors, and cost descriptions. |
| `model` | `arko_ai::Model` | Distillation model supplied by the caller. |
| `target_agents` | `HashSet<String>` | Registration names to match, as given to `WorkflowBuilder::agent` (not `AgentConfig::name`). Required: `build` rejects an empty set. |
| `target_tools` | `HashSet<String>` | Tool names to match. Required: `build` rejects an empty set. |
| `min_chars` | `usize` | Minimum input size in Unicode scalar values, inclusive; default `8_000`. |
| `max_output_tokens` | `Option<usize>` | Generation limit, including reasoning tokens. Default `None`: `default_max_output_tokens()` (`16_000`, or the model's `Model::max_output_tokens` when that is known and smaller) is sent where the provider takes a limit, and no limit on ChatGPT OAuth, whose backend rejects one. A set value is sent on every request (arko_ai drops it on ChatGPT OAuth, with a warning logged once). Must be positive. A JSON configuration with a number loads as that set value. |
| `instructions` | `String` | Content-preservation instructions; default below. |
| `reasoning` | `Option<arko_ai::ReasoningConfig>` | Default `None`; omitted from JSON when `None` and `None` when omitted. |

| Signature on `DistillationConfig` | Parameters and result |
| --- | --- |
| `pub fn new(name: impl Into<String>, model: arko_ai::Model) -> Self` | Configuration with the defaults above. Default instructions: "Preserve every fact relevant to the task, exact errors, paths, identifiers, constraints, and actionable details. Remove repetition and irrelevant bulk. Never invent facts." |
| `pub fn target_agent(mut self, agent: impl Into<String>) -> Self` | Add an agent registration-name filter; duplicates collapse. |
| `pub fn target_tool(mut self, tool: impl Into<String>) -> Self` | Add a tool-name filter; duplicates collapse. |
| `pub fn min_chars(mut self, chars: usize) -> Self` | Inclusive size threshold; `0` also admits empty output. |
| `pub fn max_output_tokens(mut self, tokens: usize) -> Self` | Set the token limit; `build` rejects zero. |
| `pub fn default_max_output_tokens(&self) -> usize` | The limit sent when none is set. |
| `pub fn instructions(mut self, instructions: impl Into<String>) -> Self` | Replace the instructions. |
| `pub fn reasoning(mut self, reasoning: arko_ai::ReasoningConfig) -> Self` | Set reasoning; `build` rejects a reasoning configuration for a different provider than the model's. |

`DistillationConfig` has no `Default` implementation. `build` validates the token limit, the reasoning provider, and the targets (at least one agent and one tool; each agent must be registered); a target tool no agent is configured with is logged as a warning. A summary replaces the tool result only when the response has no refusal, stops with `StopReason::EndTurn` or `StopReason::StopSequence`, and contains non-empty text. Otherwise the raw result is kept:

- A refusal (`GenerationResponse::refusal` from any source, even with accompanying text, or an `arko_ai::Error::Refusal` such as an OpenRouter moderation 403) or a truncated response (`StopReason::is_truncated`: `MaxTokens`, when the summary reached `max_output_tokens`, or `ContextWindowExceeded`) would likely repeat, so the distiller logs a warning and returns `InterceptedResult::Unchanged`; the failure policy does not retry it.
- Model errors, empty text, and any other stop reason are interceptor errors, which the interception failure policy may retry before the raw result is kept. This includes `StopReason::Other("unknown")`, a response that reported no stop reason (OpenRouter `finish_reason: null`, Anthropic `stop_reason: null`), whose completeness cannot be determined.

A finished, non-empty summary replaces the result even when it is longer than the input; `max_output_tokens` caps generation and does not guarantee compression. With an OpenAI API key and with Anthropic, reasoning (thinking) tokens count against `max_output_tokens`, and on Anthropic a manual thinking budget must stay below it unless interleaved thinking is enabled; a limit too small for the model's reasoning yields truncated summaries, which are charged but not used. ChatGPT OAuth requests send no output limit.

Each distillation charges its billed amount, `response.usage.billed()` (zero for subscription calls such as ChatGPT OAuth), as a **manual workflow cost** with the description `distillation: <name>` and agent, task, and tool metadata. The charge counts toward the workflow budget but not toward the agent's own LLM spend. Warning and exceeded hooks fire as applicable, and the managed driver enforces an exceeded workflow budget. Every response the model returns is charged and recorded, including refused, truncated, and empty ones, since they are billed too; each retry of an interceptor error is another billed call. The distillation record is added after the budget hooks succeed.

### `DistillationRecord`

```rust
pub timestamp: chrono::DateTime<chrono::Utc>;
pub task_id: uuid::Uuid;
pub agent_name: String;
pub tool_name: String;
pub config_name: String;
pub input_chars: usize;
pub output_chars: usize;
pub cost: f64;
pub rejection: Option<String>;
```

A record identifies the task, the agent (by registration name), the tool, and the configuration, with the UTC time, input and output sizes in Unicode scalar values, and the billed dollar cost. `output_chars` counts the text the model generated, joined as a summary is (text parts separated by newlines): the message text plus any `Refusal::partial_output` that a safety-classifier or content-filter refusal moved out of the message; a refusal's explanation is not counted. `rejection` is `None` when the summary replaced the result and otherwise gives the reason it was not used (the refusal, the stop reason, or missing text); it is omitted from JSON when `None` and loads as `None` when absent. Records are available in `SharedContext::distillation_records` and in checkpoints.

## Budget and cost accounting

Source: [budget/mod.rs](src/budget/mod.rs). All five public types are in the prelude. Amounts, limits, thresholds, and returned totals are in dollars. Running workflow and per-agent totals are kept in whole micro-dollars (amounts are truncated); ledger entries keep the original `f64` amounts, so a ledger sum can differ slightly from the running total.

### `BudgetManager`

Thread-safe workflow spend, per-agent LLM spend and limits, and a detailed ledger. Fields are private, and there is no `Default` implementation. The manager records costs and reports thresholds; it neither cancels agents nor fires hooks. The managed workflow driver does both.

| Signature | Parameters, return, and behavior |
| --- | --- |
| `pub fn new(limit: Option<f64>, warning: Option<f64>) -> Self` | Zero spend, an empty ledger, and no agent limits. `None` disables the corresponding workflow threshold. |
| `pub fn from_ledger(limit: Option<f64>, warning: Option<f64>, ledger: CostLedger) -> Self` | Restore totals from ledger entries, rebuilding per-agent spend from `CostSource::Agent` entries. A warning threshold already reached counts as fired. Agent limits start empty. |
| `pub fn add_agent_cost(&self, agent_name: &str, usage: &arko_ai::UsageInfo) -> AddCostResult` | Add an agent entry for `usage.billed()` (zero for subscription calls such as ChatGPT OAuth, whose estimated cost stays in the stored usage), with a copy of the usage, the current time, and no metadata; update workflow and agent spend. |
| `pub fn set_agent_limit(&self, agent_name: impl Into<String>, limit: f64)` | Set or replace the agent's limit. |
| `pub fn agent_limit(&self, agent_name: &str) -> Option<f64>` | The agent's limit, if set. |
| `pub fn agent_spent(&self, agent_name: &str) -> f64` | Recorded agent LLM spend; zero for an unknown name. |
| `pub fn is_agent_exceeded(&self, agent_name: &str) -> bool` | Whether agent spend is strictly greater than its limit. |
| `pub fn add_manual_cost(&self, amount: f64, description: &str, metadata: Option<String>) -> AddCostResult` | Add a manual entry with the current time and optional free-form metadata; update workflow spend only. |
| `pub fn total_spent(&self) -> f64` | Running workflow spend. |
| `pub fn is_exceeded(&self) -> bool` | Whether workflow spend is strictly greater than the limit; `false` without a limit. |
| `pub fn check_warning(&self) -> bool` | `true` once, the first time spend is at or above the warning threshold; the call consumes the notification. |
| `pub fn remaining(&self) -> Option<f64>` | `limit - spent`, floored at zero; `None` without a limit. |
| `pub fn limit(&self) -> Option<f64>` | The workflow limit. |
| `pub fn snapshot_ledger(&self) -> CostLedger` | Clone of the ledger. |

The recording methods keep a cost even when it exceeds the budget, and their result consumes a newly reached warning. The manager does not validate its inputs; supply finite, nonnegative values. `WorkflowBuilder` validates configured limits and thresholds, and `Workflow::add_manual_cost` validates the amount, fires the budget hooks, and returns `WorkflowError::BudgetExceeded`.

### Cost data types

`CostSource` implements `Debug`, `Clone`, and Serde:

```rust
pub enum CostSource {
    Agent { agent_name: String, usage: Box<arko_ai::UsageInfo> },
    Manual { description: String },
}
```

`Agent` keeps the agent's registration name and full usage breakdown, including the estimated cost of subscription calls whose billed `amount` is zero. `Manual` keeps a free-form description; distillation charges are manual entries.

`CostEntry` implements `Debug`, `Clone`, and Serde. Public fields:

```rust
pub amount: f64;
pub source: CostSource;
pub timestamp: chrono::DateTime<chrono::Utc>;
pub metadata: Option<String>;
```

`amount` is the dollar charge as recorded; `timestamp` is when it was recorded; `metadata` is free-form accounting context with no prescribed schema.

`CostLedger` implements `Debug`, `Clone`, `Default` (empty), and Serde. Its one public field, `entries: Vec<CostEntry>`, is in insertion order. `pub fn total(&self) -> f64` sums the entry amounts.

`AddCostResult` implements `Debug` and `Clone`. Public fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `new_total` | `f64` | Running workflow total after the addition. |
| `warning_triggered` | `bool` | This addition reached the warning threshold for the first time. |
| `exceeded` | `bool` | The workflow limit is exceeded after this addition (not only on the first crossing). |

## Durable steps

Source: [durable/mod.rs](src/durable/mod.rs). Module: `arko_workflows::durable`; the prelude re-exports every public type except `DurableStore`. Durable steps replay recorded outputs of explicitly identified operations and hold uncertain outcomes for reconciliation. The `Workflow` methods require configured persistence; `DurableStore` can also be constructed directly.

In the signatures below, `Serialize` is `serde::Serialize`, `DeserializeOwned` is `serde::de::DeserializeOwned`, and `Future` is `std::future::Future`.

### Step specifications, execution context, and reconciliation

| Type | Public fields or variants | Purpose and defaults |
|---|---|---|
| `DurableStepSpec` | `key: String`, `instance: Option<String>`, `version: String`, `retry_policy: DurableRetryPolicy` | Stable operation definition. `new(key)` sets `instance = None`, `version = "v1"`, and `retry_policy = Never`. Key plus instance identify the step; the version and the hash of the serialized input must match the recorded definition. |
| `DurableRetryPolicy` | `Never`; `WhenDefinitelyNotCompleted { max_attempts: u32 }`; `RetrySafe { max_attempts: u32 }` | Default `Never`. `WhenDefinitelyNotCompleted` retries after a resolver confirms the operation did not complete; `RetrySafe` retries an uncertain operation without reconciliation. `max_attempts` includes the first attempt and is treated as at least one. |
| `DurableStepStatus` | `Started`, `Completed`, `InDoubt`, `PermanentlyFailed` | Journal states. A `Started` step without a later record is treated as in doubt on replay. |
| `DurableStepContext` | `workflow_run_id: Uuid`, `step_id: Uuid`, `operation_id: String`, `attempt: u32` | Closure input. The IDs are stable across retries; `attempt` starts at 1. Use `operation_id` as an idempotency or receipt-lookup key with external services. |
| `InDoubtContext` | `workflow_run_id: Uuid`, `step_id: Uuid`, `operation_id: String`, `attempt: u32`, `reason: Option<String>` | Details of an uncertain attempt for resolvers and operators. |
| `InDoubtResolution<O>` | `Completed(O)`; `DefinitelyNotCompleted`; `StillInDoubt { reason: String }`; `PermanentlyFailed { reason: String }` | Respectively: record the recovered output; allow a retry the policy permits; keep the step in doubt with a reason; record a permanent failure. |

`DurableStepSpec`, `DurableRetryPolicy`, and `DurableStepStatus` implement `Clone` and Serde; the retry policy and status implement `PartialEq` and `Eq`, and the status also implements `Copy`. Both context types implement `Clone`.

```rust
impl DurableStepSpec {
    pub fn new(key: impl Into<String>) -> Self;
    pub fn instance(mut self, instance: impl Into<String>) -> Self;
    pub fn version(mut self, version: impl Into<String>) -> Self;
    pub fn retry_policy(mut self, policy: DurableRetryPolicy) -> Self;
    pub fn display_key(&self) -> String;
}
impl DurableStepContext {
    pub fn operation_id(&self) -> &str;
}
#[async_trait::async_trait]
pub trait ResolveInDoubt<I, O>: Send + Sync {
    async fn resolve(
        &self, context: &InDoubtContext, input: &I,
    ) -> Result<InDoubtResolution<O>, WorkflowError>;
}
```

`instance`, `version`, and `retry_policy` set the matching field. `display_key` returns `key` or `key[instance]`. `operation_id()` borrows the operation ID. Implement `ResolveInDoubt` to check an external system using the attempt context and the original input; resolver errors propagate.

A spec is validated whenever a step executes or an operator resolution is applied: the key and version must contain non-whitespace characters, and the key and instance must not contain newline, carriage return, or NUL; an invalid spec returns `BuilderInvalid`. `None` and `Some("")` instances are the same identity. Reusing an identity with a different input or version returns `DurableDefinitionMismatch`; give a separate operation its own key or instance.

### DurableStore

`DurableStore` has no public fields. It keeps one run's append-only JSONL journal at `<run_dir>/durable/journal.jsonl`.

```rust
impl DurableStore {
    pub fn new(workflow_run_id: Uuid, run_dir: PathBuf) -> Self;
    pub fn journal_path(&self) -> &std::path::Path;
    pub fn workflow_run_id(&self) -> Uuid;
    pub async fn execute<I, O, F, Fut>(
        &self, spec: &DurableStepSpec, input: &I, execute: F,
    ) -> Result<O, WorkflowError>
    where
        I: Serialize,
        O: Serialize + DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: Future<Output = Result<O, WorkflowError>>;
    pub async fn execute_with_resolver<I, O, F, Fut, R>(
        &self, spec: &DurableStepSpec, input: &I, execute: F, resolver: &R,
    ) -> Result<O, WorkflowError>
    where
        I: Serialize + Sync,
        O: Serialize + DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: Future<Output = Result<O, WorkflowError>>,
        R: ResolveInDoubt<I, O>;
    pub async fn list_in_doubt(&self) -> Result<Vec<InDoubtContext>, WorkflowError>;
    pub async fn operator_complete<I, O>(
        &self, spec: &DurableStepSpec, input: &I, output: &O,
        actor: &str, reason: &str,
    ) -> Result<(), WorkflowError>
    where I: Serialize, O: Serialize;
    pub async fn operator_permanently_fail<I: Serialize>(
        &self, spec: &DurableStepSpec, input: &I, actor: &str, reason: &str,
    ) -> Result<(), WorkflowError>;
    pub async fn operator_authorize_retry<I: Serialize>(
        &self, spec: &DurableStepSpec, input: &I, actor: &str, reason: &str,
    ) -> Result<(), WorkflowError>;
}
```

| Method | Parameters and behavior |
|---|---|
| `new` | Associate a run UUID and directory. Creates no files and does not check an existing journal's run identity. |
| `journal_path`, `workflow_run_id` | The journal path and run ID. |
| `execute` | Hash the serialized `input`, validate `spec`, and either return the recorded output of a completed step or run `execute(context)` once and journal the outcome. When the closure returns an error or its output cannot be serialized, the attempt is recorded as in doubt with that error as the reason, and `DurableInDoubt` is returned. |
| `execute_with_resolver` | As `execute`, but when an uncertain earlier attempt may not already retry, `resolver` is consulted. `RetrySafe` steps and operator-authorized retries run without consulting it. |
| `list_in_doubt` | Latest `Started` and `InDoubt` attempts, including steps still executing. Order is unspecified. |
| `operator_complete` | Record `output` as the result of the matching unresolved step. It must deserialize as the output type later callers expect. |
| `operator_permanently_fail` | Record a permanent failure; later execution returns `DurablePermanentlyFailed`. |
| `operator_authorize_retry` | Allow the next execution call to retry, even with `Never`; a configured attempt limit still applies. Does not run the closure. |

The operator methods require an unresolved attempt with a matching version and input, and return `DurableDefinitionMismatch` otherwise. They append records carrying `actor` and `reason` and never rewrite earlier records. They take no execution lease, so resolve steps only after execution has stopped.

A retry happens on a later call; each call runs the closure at most once. When the configured attempts are exhausted, execution returns `DurableAttemptsExhausted`. Persistence and serialization errors propagate. A completed step replays without running its closure, but external side effects still need idempotency or reconciliation. Executing an identity that is already running on the same store returns `DurableInDoubt`. Locking and active-step protection are per store instance; there is no cross-process lock. Reading the journal tolerates an invalid final record but rejects an invalid earlier one.

## Persistence and checkpoint state

Source: [persistence/mod.rs](src/persistence/mod.rs), [persistence/state.rs](src/persistence/state.rs). Configuration and manager: `arko_workflows::persistence`; state types: `arko_workflows::persistence::state`. The prelude re-exports `PersistenceConfig`.

### PersistenceConfig and PersistenceManager

`PersistenceConfig` implements `Debug` and `Clone`. It has no `Default` implementation; `new` sets the defaults below. `PersistenceManager` implements `Clone` and has private fields; clones share the durable store, write lock, and event log.

| Public field | Type | Meaning and default from `new` |
|---|---|---|
| `checkpoint_dir` | `std::path::PathBuf` | The run directory. |
| `auto_checkpoint` | `bool` | Save a checkpoint after each finalized agent result, and once when a run call first returns a stopped run's error. Default `true`. |
| `retain_checkpoints` | `Option<usize>` | Keep only the newest this many checkpoint files, deleting older ones (earlier processes' first, by modification time) after each save; a file that can't be deleted is logged and kept. Default `None`: every checkpoint is kept. |
| `event_log` | `EventLog` | `Full` (default): every agent event, including each streamed fragment. `LifecycleOnly`: every event except `AssistantResponseChunk`, `ReasoningChunk`, `ToolParamName`, `ToolParamValueChunk`, and `ToolParamEnd`; the finished responses are in the task files, but streamed reasoning a provider doesn't return with the response is only in a `Full` log. |
| `event_flush_interval` | `std::time::Duration` | Buffered events are written at least this often; `TaskStateChanged`, `TaskStreamEnd`, `ErrorEncountered`, `RetryScheduled`, `UsageLimitReached`, `RefusalReceived`, and `OutputTokenLimitReached` are written at once, and the buffer before each checkpoint. Default `DEFAULT_EVENT_FLUSH_INTERVAL`, one second. |

```rust
impl PersistenceConfig {
    pub fn new(checkpoint_dir: impl Into<std::path::PathBuf>) -> Self;
    pub fn auto_checkpoint(self, enabled: bool) -> Self;
    pub fn retain_checkpoints(self, count: usize) -> Self; // at least 1
    pub fn event_log(self, event_log: EventLog) -> Self;
    pub fn event_flush_interval(self, interval: std::time::Duration) -> Self;
    pub fn user_data(workflow_name: &str) -> Result<Self, WorkflowError>;
}
impl PersistenceManager {
    pub fn resolve_run_id(config: &PersistenceConfig) -> Result<uuid::Uuid, WorkflowError>;
    pub fn new(config: PersistenceConfig, workflow_run_id: uuid::Uuid) -> Self;
    pub fn durable_store(&self) -> std::sync::Arc<crate::durable::DurableStore>;
    pub async fn save_checkpoint(
        &self, ctx: &crate::context::SharedContext,
    ) -> Result<std::path::PathBuf, WorkflowError>;
    pub async fn append_event(
        &self, agent_name: &str, agent_run_id: uuid::Uuid,
        event: &arko_agents::runtime::event::AgentEvent,
    );
    pub async fn flush_events(&self);
    pub async fn save_agent_task(
        &self, task: &arko_agents::Task,
    ) -> Result<std::path::PathBuf, WorkflowError>;
    pub async fn auto_checkpoint(
        &self, ctx: &crate::context::SharedContext,
    ) -> Result<(), WorkflowError>;
    pub async fn load_checkpoint(
        path: &std::path::Path,
    ) -> Result<state::WorkflowState, WorkflowError>;
}
```

| Method | Purpose, parameters, and result |
|---|---|
| `PersistenceConfig::new` | Use the given run directory. |
| `PersistenceConfig::user_data` | A new run directory `<user-data>/arko/workflows/<workflow_name>/<new UUID>`, not yet created. The user-data base is `LOCALAPPDATA` on Windows, `$HOME/Library/Application Support` on macOS, and `XDG_DATA_HOME` or `$HOME/.local/share` elsewhere. Returns `Persistence` when no base can be determined. |
| `PersistenceManager::resolve_run_id` | Create the run directory if needed and return the UUID in its `workflow-run-id` file, creating and syncing a new one when the file is absent. A malformed UUID returns `Persistence`. |
| `PersistenceManager::new` | Associate a configuration with a run identity; when reopening a run, pass the identity from `resolve_run_id`. |
| `durable_store` | Shared handle to the run's durable-step store. |
| `save_checkpoint` | Write the buffered events, then snapshot `ctx` into a new pretty-printed `checkpoint-<UUID>.json` (written to a temporary file, synced, and renamed), point `latest-checkpoint` at its file name, apply `retain_checkpoints`, and return its path. Agent results refer to their task files; a task not saved to its file yet (restored from a checkpoint that held it) is written first. |
| `append_event` | Record `event` in `events.jsonl` as `event_log` selects, with the current UTC time, the workflow run UUID, `agent_name`, and `agent_run_id`: buffered, and written as described for `event_flush_interval`. The file stays open between writes. A failed write is logged as a warning once until writing succeeds again, and the events stay buffered (the oldest are dropped past 64 MiB, with a warning); it never stops an agent. Dropping the last manager writes what is still buffered. |
| `flush_events` | Write the buffered events now. |
| `save_agent_task` | Write `task` to `agents/<task.id>.json` (temporary file, sync, rename), replacing the task's earlier snapshot, and return its path. |
| `auto_checkpoint` | Save a checkpoint only when `config.auto_checkpoint` is `true`; otherwise succeed without writing. |
| `load_checkpoint` | Read and deserialize the given checkpoint file. Does not follow `latest-checkpoint` or restore a workflow; use `WorkflowBuilder::resume_from` for that. `WorkflowState::agent_results_in(run_dir)` returns its agent results with their tasks, read from the task files. |

Filesystem and serialization failures are returned as `WorkflowError`. A run directory holds checkpoints, task snapshots, the event log, and the durable journal; running futures and ballots are not persisted.

`resume_from` restores memory content and history, pending messages, the cost ledger, finalized agent results, verification and distillation records, and the agent status registry. For memory, the new builder's `MemoryConfig` decides each tier's readers, writers, limit, and write mode, and the checkpoint supplies content and history by tier name. A configured tier the checkpoint lacks starts as configured. A checkpoint tier that matches no configured tier makes `build` fail with `BuilderInvalid` naming it, unless `rename_memory_tier_on_resume(from, to)` maps it to a configured tier or `drop_memory_tier_on_resume(name)` discards it; two checkpoint tiers mapped to one configured tier also fail. Content over a tier's configured limit is kept (with a warning); writes that grow it fail until it is smaller. When the write mode changed, each write's text is stored in its entry. Freshness state is restored only when freshness tracking is configured again. Agents, the client, tools, hooks, policies, budget limits, and metadata come from the new builder, not the checkpoint. The pause flag starts cleared, and active message channels, ballots, and verification repair counts and baselines start empty. Restoring does not continue agents or replay workflow control flow. Accounting is rebuilt from `cost_ledger`; `budget_spent` and `completed_agents` are not used on restore. A recorded `cancellation` is not restored: the resumed run starts again, as described under [Stopping a run](#stopping-a-run).

No agent is running after a resume. From `agent_registry`, an agent that was running when the checkpoint was saved counts as finished (interrupted), and each continuable run keeps the registry version it has seen, so a continued run is told only about recipients that started after that. From a checkpoint without `agent_registry`, an agent with recorded results takes its latest result's outcome, and any other agent is not started.

### Checkpoint payloads

`WorkflowState` implements Serde and exposes every field below; it has no `Default` implementation. The last column gives the value used when a field is missing from the checkpoint JSON.

| Field | Type | Purpose; value when missing |
|---|---|---|
| `workflow_run_id` | `uuid::Uuid` | Run identity; nil UUID. |
| `workflow_name` | `String` | Workflow name; required. |
| `workflow_metadata` | `crate::workflow::config::WorkflowMetadata` | Description, platform, and dependency metadata; `WorkflowMetadata::default()`. |
| `checkpoint_id` | `uuid::Uuid` | Snapshot ID; required. |
| `timestamp` | `chrono::DateTime<chrono::Utc>` | Snapshot time; required. |
| `memory_snapshot` | `Vec<crate::memory::MemoryTier>` | Memory tiers and entries; required. |
| `message_queues` | `std::collections::HashMap<String, Vec<crate::messaging::AgentMessage>>` | Pending messages per agent; required. |
| `budget_spent` | `f64` | Workflow spend at checkpoint time; required. |
| `cost_ledger` | `crate::budget::CostLedger` | Complete accounting history; empty ledger. |
| `freshness_snapshot` | `Option<crate::freshness::FreshnessSnapshot>` | Freshness tracking state; `None`. |
| `agent_results` | `Vec<crate::workflow::result::AgentResult>` | Finalized results with their tasks, as checkpoints written before `agent_runs` held them; omitted from JSON when empty (always, in new checkpoints); empty. |
| `agent_runs` | `Vec<AgentRunRecord>` | Finalized results, including failures and pauses, each referring to its task file; omitted from JSON when empty; empty. |
| `verification_records` | `Vec<crate::verification::VerificationRecord>` | Completion-verification history; empty. |
| `distillation_records` | `Vec<crate::distillation::DistillationRecord>` | Distillation history; empty. |
| `completed_agents` | `Vec<SerializableAgentResult>` | Name and summary of each recorded result; empty. |
| `cancellation` | `Option<crate::workflow::cancel::Cancellation>` | The run's stop, in checkpoints written after it (omitted from JSON when `None`); `None`. |
| `agent_registry` | `Option<crate::messaging::registry::RegistrySnapshot>` | Agent statuses and what each agent run has seen of them (omitted from JSON when `None`); `None`. |

`RegistrySnapshot` has public fields `version: u64`, which every status change increases; `agents: std::collections::BTreeMap<String, AgentRecord>`, keyed by registration name; and `seen: HashMap<Uuid, u64>`, the registry version each agent run (by task ID) has been told about. `AgentRecord` has `last_start: u64`, the version at the agent's latest start (`0` if it never started), and `finished: Option<AgentRunStatus>`, how its latest run ended when none is running (omitted from JSON when `None`). Every field defaults when missing. Both types implement `Debug`, `Clone`, `Default`, `PartialEq`, `Eq`, and Serde. What a run that cannot be continued (any status other than `Completed` or `WaitingForInput`) has seen is dropped when it ends.

Unknown fields are ignored on load, so checkpoints written by earlier versions with fields since removed, such as `oauth_exempt_agents`, still load; their ledger entries keep their recorded amounts. A task state saved as `Failed` with exactly the reason `Task cancelled by user.` deserializes as `TaskState::Cancelled`, in checkpoints, task files, and event-log records alike. Budget accounting records `UsageInfo::billed()`, which is zero for subscription calls such as ChatGPT OAuth.

`SerializableAgentResult` has public fields `agent_name: String` and `summary: String` and implements Serde, `Clone`, and `Default` (both strings empty).

`AgentRunRecord` (`Debug`, `Clone`, Serde) has public fields `agent_run_id: Uuid`, `agent_name: String`, `status: AgentRunStatus`, `summary: String`, `error: Option<String>` (omitted from JSON when `None`), and `task_file: String`, the task file relative to the run directory (`agents/<agent_run_id>.json`). A task file always holds the task's latest state, so an older checkpoint that refers to it reads the task as it is now. `WorkflowState::agent_results_in(&self, run_dir: &Path) -> Result<Vec<AgentResult>, WorkflowError>` returns the results `agent_results` holds, then those of `agent_runs` with their tasks read from `run_dir`; a task file that can't be read or parsed returns `Persistence` naming it. `resume_from` reads the task files from the checkpoint's directory. A checkpoint holding its tasks (from an earlier version) still loads; the next checkpoint writes their task files and refers to them.

## Structured voting

Source: [voting/mod.rs](src/voting/mod.rs). Module: `arko_workflows::voting`; the prelude re-exports `VoteOption`, `TieBreak`, `BallotConfig`, `CastVote`, and `BallotResult`. To run a ballot: call `WorkflowBuilder::enable_voting` with the agents that vote (only they are offered `cast_vote`), begin a ballot, give its UUID and options to the participating agents, and finish it after they vote.

| Type | Public fields or variants | Purpose and constraints |
|---|---|---|
| `VoteOption` | `id: String`, `description: String` | Machine-readable option ID and a description for agents. IDs must be nonblank and unique within a ballot. |
| `TieBreak` | `RejectTie`, `Lexicographic` | Reject a tie for the top total, or choose the lexicographically first tied option ID. |
| `BallotConfig` | `question: String`, `options: Vec<VoteOption>`, `eligible_voters: std::collections::HashSet<String>`, `quorum: usize`, `consensus_threshold: f64`, `weights: std::collections::HashMap<String, f64>`, `tie_break: TieBreak` | At least two options, a positive quorum, and a finite threshold in `[0, 1]`. An empty eligibility set admits every agent. A voter without a weight counts `1.0`; weights of actual voters must be finite and positive. No `Default` implementation. |
| `CastVote` | `voter: String`, `task_id: uuid::Uuid`, `option: String`, `reasoning: String`, `timestamp: chrono::DateTime<chrono::Utc>` | A recorded vote. The tool sets the voter, task, and time; the option and reasoning come from its arguments. |
| `BallotResult` | `ballot_id: uuid::Uuid`, `winner: String`, `consensus: f64`, `votes: Vec<CastVote>` | Winning option, its weight divided by the total cast weight, and the recorded votes (unspecified order). |

These types implement `Clone` and Serde; `TieBreak` also implements `Copy`, `PartialEq`, and `Eq`. `BallotStore` has private fields; `Default` creates an empty in-memory store.

```rust
pub const CAST_VOTE_TOOL_NAME: &str = "cast_vote";
impl BallotStore {
    pub fn begin(&self, config: BallotConfig) -> Result<uuid::Uuid, String>;
    pub fn finish(&self, ballot_id: uuid::Uuid) -> Result<BallotResult, String>;
}
```

`begin` validates the options, quorum, and threshold, stores an empty ballot, and returns its UUID. `finish` requires an existing ballot, checks the number of distinct voters against the quorum, validates the weights of cast votes, totals the weighted votes, applies the tie-break, and checks consensus. Its errors name the cause: unknown ballot, quorum not reached, invalid weight, tie, or insufficient consensus. Finishing computes a result without closing or removing the ballot, so eligible agents can still vote. Ballots live in memory and are not checkpointed. The `Workflow` ballot methods return store errors as `WorkflowError::Deliberation`.

### Agent tool: cast_vote

`cast_vote(ballot_id: string, option: string, reasoning: string) -> string`. All three arguments are required, and the tool is not batch-safe. `ballot_id` must be a UUID supplied by the workflow; `option` must exactly match a configured option ID; `reasoning` carries the voter's rationale. The tool sets `voter` to the running agent's registration name, `task_id` to its task, and the timestamp to the current UTC time. Only eligible agents may vote, once per ballot per agent name. Success returns `Vote recorded for option '<option>'.`; missing or non-string arguments, malformed UUIDs, unknown ballots or options, ineligible voters, and repeat votes are errors. The tool type is crate-private and votes can be cast only through it; `enable_voting` gives it to every agent.

## Python bindings

Source: [python/mod.rs](src/python/mod.rs). Cargo feature: `python`. The Python module `arko_workflows` exports three classes, `PyWorkflowBuilder`, `PyWorkflow`, and `PyAgentResult`, and the exceptions `WorkflowCancelled`, `UserInputRequired`, and `AgentPaused`, subclasses of `Exception`. `UserInputRequired` and `AgentPaused` carry `(message, agent_run_id)` as their arguments. The three class names are public Rust wrapper types in `arko_workflows::python`; their Python methods are not public Rust methods. `PyAgentResult` implements `From<arko_workflows::AgentResult>`.

The tables use Python-facing signatures: `self` and PyO3 interpreter arguments are omitted, and awaitable methods are written as `async def`. Builder setters mutate the builder and return `None`, so they cannot be chained like the Rust builder. Numbers map to the underlying Rust `f64` and `usize` types.

### PyWorkflowBuilder

| Python signature | Parameters and behavior |
| --- | --- |
| `PyWorkflowBuilder()` | Empty builder. Name, workspace, and agents are required as in Rust; the builder creates its own LLM client at build time. |
| `name(name: str) -> None` | Set the workflow name. |
| `workspace(path: str) -> None` | Set the existing workspace directory. |
| `agent(name: str, config_path: str) -> None` | Add an agent whose configuration file is loaded during `build`. |
| `budget(limit: float) -> None` | Set the workflow budget in dollars; validated at build. |
| `persistence(checkpoint_dir: str, auto_checkpoint: bool = True, retain_checkpoints: int \| None = None, event_log: str = "full") -> None` | Enable persistence in `checkpoint_dir`, with the Rust defaults: a checkpoint after every finalized agent run, every checkpoint kept (`retain_checkpoints` keeps the newest that many), and every event logged (`"lifecycle_only"` leaves out streamed fragments; other values raise `ValueError`). |
| `rename_memory_tier_on_resume(old: str, new: str) -> None` | As `WorkflowBuilder::rename_memory_tier_on_resume`. |
| `drop_memory_tier_on_resume(name: str) -> None` | As `WorkflowBuilder::drop_memory_tier_on_resume`. |
| `approval_policy(policy: str) -> None` | `"approve_all"` (the default) or `"respect_agent_config"`; other values raise `ValueError`. A handler is Rust-only. |
| `error_recovery_policy(policy: str, max: int \| None = None) -> None` | `"automatic"` (the default), `"limited"` with `max`, or `"return_to_caller"`; other combinations raise `ValueError`. |
| `max_parallel_agents(max: int) -> None` | As `WorkflowBuilder::max_parallel_agents`; `build()` raises `ValueError` for zero. |
| `fail_fast(enabled: bool) -> None` | As `WorkflowBuilder::fail_fast` (default off). |
| `resume_from(checkpoint: str) -> None` | Restore a checkpoint JSON file at build; requires `persistence`. |
| `global_memory_tier(name: str, max_bytes: int \| None = None) -> None` | Add an append-only tier every agent can read and write, with an optional byte limit. |
| `cancel_grace_period(seconds: float) -> None` | Seconds a stopped run waits for each running agent to stop before aborting it; default 30. A negative or non-finite value raises `ValueError`, and so does `build()` for a value too large for a duration. |
| `wait_for_message_timeout(seconds: float \| None = None) -> None` | The longest `wait_for_message` wait in seconds, as `WorkflowBuilder::wait_for_message_timeout`; `None` selects the automatic limit (the default). A non-positive or non-finite value raises `ValueError`; `build()` raises it for more than 24 hours. |
| `agent_wait_for_message_timeout(agent: str, seconds: float \| None = None) -> None` | The same for one agent, in place of the workflow's setting; `None` selects the automatic limit for it. `build()` raises `ValueError` for an unknown agent. |
| `build() -> PyWorkflow` | Create the LLM client, load the agent files, and build the workflow. Load and build failures raise `ValueError`. The builder is not consumed and can be reused. |

`build` creates the client with `LLMClient::from_env()`, which configures every provider whose credentials are in the process environment, ignoring blank values: OpenAI with ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (plus optional `OPENAI_OAUTH_REFRESH_TOKEN`), otherwise with an OpenAI API key from `OPENAI_API_KEY`; Anthropic from `ANTHROPIC_API_KEY`; Google Gemini from `GOOGLE_API_KEY`, else `GEMINI_API_KEY`; and OpenRouter from `OPENROUTER_API_KEY`. Setting only some of the `OPENAI_OAUTH_*` variables fails the build. When `OPENAI_OAUTH_REFRESH_TOKEN` is the refresh token in the Codex `auth.json` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`), that file is used as the credential store, so refreshed tokens are written back to it; otherwise refreshed tokens are kept only in memory, and the first refresh logs a warning. The Python API accepts no custom client, custom tools, hooks, approval handlers, verifiers, distillers, or service settings; those require the Rust API.

### PyWorkflow

Obtained from `PyWorkflowBuilder.build()`; there is no Python constructor.

| Python signature | Parameters, result, and behavior |
| --- | --- |
| `async def run_agent(agent_name: str, prompt: str) -> PyAgentResult` | Run a registered agent with the managed driver. |
| `async def run_agents_parallel(agents: list[tuple[str, str]]) -> list[PyAgentResult]` | Run `(name, prompt)` pairs concurrently; results in input order when none fails (an agent that paused is in the list, with its status). |
| `async def continue_agent(previous: PyAgentResult, prompt: str) -> PyAgentResult` | As `Workflow::continue_agent`: a follow-up for a completed run, or the answer for one waiting for input. |
| `async def resume_agent(previous: PyAgentResult, decision: str, note: str \| None = None) -> PyAgentResult` | As `Workflow::resume_agent` for a paused run: `"approve"` or `"reject"` every call awaiting approval (`note` is the message or reason), or `"continue"` (`note` is added for the model) or `"fail"` (`note` is the reason) after an error-recovery pause. Other values raise `ValueError`. |
| `agent_result(agent_run_id: str) -> PyAgentResult \| None` | The recorded result of a run, as `UserInputRequired` and `AgentPaused` give its ID. A malformed ID raises `ValueError`. |
| `async def finish() -> list[PyAgentResult]` | As `Workflow::finish`: fire `PostWorkflow`, save a checkpoint, and return the recorded agent results. |
| `cancel() -> bool` | Cancel the run, as `Workflow::cancel`: running agents stop, and pending and later runs raise `WorkflowCancelled`. Returns `False` when the run was already stopped. |
| `is_cancelled() -> bool` | Whether the run is stopped. |
| `add_manual_cost(amount: float, description: str, metadata: str \| None = None) -> float` | Record a dollar cost and return the new workflow total. A budget error still leaves the charge recorded. |
| `cost_ledger_json() -> str` | The current `CostLedger` as JSON. |
| `async def checkpoint() -> None` | Save a checkpoint; does nothing without persistence. |
| `read_memory(agent_name: str, tier: str, max_chars: int \| None = None) -> str` | Access-checked plain-text read of a tier, or of its trailing `max_chars` characters (`MemoryStore::read_window`). Not the agent tool's versioned JSON. |
| `write_memory(agent_name: str, tier: str, content: str) -> None` | Access-checked write using the tier's mode, through the store directly: no task ID, freshness notifications, or hooks. |

The run methods raise `arko_workflows.WorkflowCancelled` once the run is cancelled, `UserInputRequired` for an agent waiting for input, and `AgentPaused` for a paused agent; their other errors raise `RuntimeError` with the message `Workflow error: <error>`. Accounting, checkpoint, and memory errors raise `RuntimeError`. Raw handles, cancel handles, event subscription, workflow pause and resume, the aggregate `WorkflowResult`, durable steps, and ballots are not available from Python.

### PyAgentResult

Read-only properties: `agent_name: str`, `summary: str`, `total_cost: float` (the billed amount, zero for ChatGPT OAuth calls), `status: str`, `agent_run_id: str`, and `error: str | None` (the failure or cancellation reason, the questions, or the pause). `status` is the Rust variant name (`Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`, or `Paused`); `agent_run_id` is a UUID string. The Rust wrapper's six matching fields are public; it also keeps the full `AgentResult` (private) for `continue_agent` and `resume_agent`. The task and token totals are not exposed. There is no Python constructor.

## Workflow errors

Source: [error.rs](src/error.rs). `WorkflowError` is re-exported at the root and in the prelude and implements `std::error::Error` and `Display` (through `thiserror`). Match on variants for programmatic handling.

| Variant and payload | Meaning |
| --- | --- |
| `Agent(arko_agents::ArkoAgentsError)` | Agent setup, runtime, or task error, including a managed task that ended failed (`TaskExecution` with the failure reason). Automatic `From`. |
| `Llm(arko_ai::Error)` | Model client error. Automatic `From`. |
| `BuilderMissing { field: String }` | A required builder setting is missing. |
| `BuilderInvalid(String)` | Invalid configuration or checkpoint identity, an invalid durable step spec, a disabled feature (freshness, voting) used through a `Workflow` method, a non-continuable result passed to `continue_agent` or a non-paused result or mismatched decision passed to `resume_agent`, or a disallowed workflow-services caller or target. |
| `AgentNotFound(String)` | No agent is registered under the name. |
| `AgentInterrupted { agent: String, reason: String }` | A managed invocation was interrupted. |
| `AgentCancelled { agent: String, reason: String }` | A managed invocation's task ended `Cancelled` while the workflow run was not stopped; `reason` is the task's reason. |
| `VerifierFailed { verifier: String, agent: String, task_id: Uuid, reason: String }` | A completion verifier could not review the agent's completion more often than its `max_review_retries` allow. The completion was neither accepted nor rejected: the task is paused (`Paused`), to be continued with `continue_agent` on `Workflow::agent_result(task_id)`. |
| `UserInputRequired { agent: String, task_id: Uuid, questions: String }` | The run is suspended until an answer is supplied with `continue_agent`. A pause, not a failure. |
| `AgentPaused { agent: String, task_id: Uuid, pause: AgentPause }` | The agent is suspended for a tool approval or error-recovery decision the workflow's policies leave to the caller; continue it with `resume_agent` (its result: `agent_result(task_id)`). A pause, not a failure. |
| `MemoryTierNotFound(String)` | Unknown memory tier. |
| `MemoryAccessDenied { agent: String, tier: String, action: String }` | The tier's reader or writer policy denies the agent. |
| `MemoryLimitExceeded { tier: String, attempted: usize, limit: usize }` | The write would exceed the tier's byte limit. |
| `BudgetExceeded { spent: f64, limit: f64 }` | Workflow spending strictly exceeds the dollar limit. |
| `AgentBudgetExceeded { agent: String, spent: f64, limit: f64 }` | The agent's cumulative LLM spending strictly exceeds its limit. |
| `InvalidCost(f64)` | A manual cost is negative or non-finite. |
| `AbortedByHook { hook_point: String, reason: String }` | A hook returned `Abort`, which stopped the run. Returned by the operation that fired a lifecycle hook and by every run call in progress or made later. `hook_point` is the `HookPoint` variant name. A later hook `Abort` on the stopped run returns this first cause's error, not a new `AbortedByHook`. |
| `Persistence(String)` | Run identity, journal, or other persisted-state failure. |
| `DurableUnavailable` | A durable operation was requested without persistence. |
| `DurableDefinitionMismatch { step: String, details: String }` | The recorded step definition or input differs from the request, or an operator resolution found no unresolved attempt. |
| `DurableInDoubt { step: String, operation_id: String }` | The outcome of an earlier attempt is unresolved, or the step is already running on this store. |
| `DurablePermanentlyFailed { step: String, reason: String }` | The step is recorded as permanently failed. |
| `DurableAttemptsExhausted { step: String }` | The retry policy's attempt limit is reached. |
| `Cancelled` | The run was cancelled through `WorkflowCancelHandle::cancel` or `Workflow::cancel`. Returned by every run call and durable step in progress or started later, and by `finish`. |
| `Io(std::io::Error)` | I/O failure. Automatic `From`. |
| `Json(serde_json::Error)` | JSON encoding or decoding failure. Automatic `From`. |
| `Channel(String)` | Task join or channel failure, a `UserInputHandler` error, a workflow-services active-call limit, or services used after the workflow was dropped. |
| `Tool(arko_agents::ToolError)` | Tool-layer error. Automatic `From`. |
| `Deliberation(String)` | Ballot configuration or tally failure. |

## Minimal composition and examples

This function takes configured agents and an LLM client and shows the required builder inputs, a shared memory tier, a sequential handoff, and an explicit finish:

```rust
use std::{path::PathBuf, sync::Arc};
use arko_agents::AgentConfig;
use arko_ai::LLMClient;
use arko_workflows::prelude::*;

async fn run_workflow(
    workspace: PathBuf,
    client: Arc<LLMClient>,
    planner: AgentConfig,
    worker: AgentConfig,
) -> Result<WorkflowResult, WorkflowError> {
    let workflow = Workflow::builder()
        .name("generated-workflow")
        .workspace(workspace)
        .llm_client(client)
        .agent("planner", planner)
        .agent("worker", worker)
        .memory(MemoryConfig::new().global_tier("decisions"))
        .budget(5.0)
        .build()?;

    let plan = workflow.run_agent("planner", "Plan the requested work").await?;
    workflow.run_agent("worker", &plan.summary).await?;
    workflow.finish().await
}
```

| Example | Demonstrates |
| --- | --- |
| [basic_workflow.rs](examples/basic_workflow.rs) | Sequential architect-to-coder handoff, a client from `LLMClient::from_env()`, a shared memory tier, a budget limit and warning, and a completion hook. Uses the fixed paths `./my-project` and `configs/*.json` set in `main`. |
| [DeepCoder guide](examples/deepcoder/README.md), [deepcoder.rs](examples/deepcoder.rs) | Agent-driven delegation, correctness, efficiency, and final verification gates, conversation reuse with repair deltas, context freshness, per-role memory policies, a custom shell tool, checkpoint and resume, and workflow metadata. The CLI and shell policies are specific to the example. |
