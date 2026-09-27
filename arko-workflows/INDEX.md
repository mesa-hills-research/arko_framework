# arko-workflows Index

Workflow-author reference for `arko-workflows` **0.1.0**, checked against the local source on **2026-09-14**. Covers the public Rust API, optional Python API, injected agent tools, configuration, defaults, result contracts, and execution behavior. Your Rust or Python control flow defines the workflow; the library supplies orchestration primitives.

Signatures describe this crate's API, with method receivers included. Standard imports are abbreviated: `Arc`/`Weak`/`OnceLock` from `std::sync`, `HashMap`/`HashSet` from `std::collections`, `Path`/`PathBuf` from `std::path`, `Uuid` from `uuid`, `Value` from `serde_json`, and `Mutex`/`RwLock` from `parking_lot` unless qualified otherwise. `AgentConfig`, `Task`, `Tool`, `ToolError`, `ToolResultInterceptor`, and `InterceptionFailurePolicy` come from `arko_agents`; `LLMClient`, `Model`, and `UsageInfo` come from `arko_ai`. Signatures are declarations, not complete implementation blocks. Builder setters consume and return `Self` unless explicitly shown otherwise. Generated standard-trait implementations are not separately indexed.

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
- [MESI file tracking](#mesi-file-tracking)
- [Completion verification](#completion-verification)
- [Tool result distillation](#tool-result-distillation)
- [Budget and cost accounting](#budget-and-cost-accounting)
- [Durable steps](#durable-steps)
- [Persistence and checkpoint state](#persistence-and-checkpoint-state)
- [Structured voting](#structured-voting)
- [Python bindings](#python-bindings)
- [Workflow errors](#workflow-errors)
- [Minimal composition and examples](#minimal-composition-and-examples)

## Imports and feature map

Source: [Cargo.toml](Cargo.toml), [crate exports](src/lib.rs).

The Cargo package is `arko-workflows`; the Rust import is `arko_workflows`. The crate uses Rust edition 2024. Default Cargo features are empty. The optional `python` feature enables PyO3 and its Tokio bridge; see [Python bindings](#python-bindings).

Root imports: `Workflow`, `WorkflowBuilder`, `AgentHandle`, `WorkflowError`, `SupportedPlatform`, `WorkflowDependency`, `WorkflowMetadata`, `AgentResult`, `AgentRunStatus`, `WorkflowEvent`, `WorkflowResult`.

`use arko_workflows::prelude::*;` re-exports the following groups. Other public types remain accessible through their module paths.

| Module | Prelude exports |
| --- | --- |
| `workflow`, `workflow::config`, `workflow::result`, `error` | All root imports above **except `AgentHandle`** |
| `budget` | `AddCostResult`, `BudgetManager`, `CostEntry`, `CostLedger`, `CostSource` |
| `distillation` | `DistillationConfig`, `DistillationRecord` |
| `durable` | `DurableRetryPolicy`, `DurableStepContext`, `DurableStepSpec`, `DurableStepStatus`, `InDoubtContext`, `InDoubtResolution`, `ResolveInDoubt` |
| `freshness` | `ActorId`, `ContextFreshnessConfig`, `FreshnessConflict`, `ResourceId`, `ResourceTracker`, `ToolResourceEffect` |
| `hooks` | `HookCallback`, `HookPoint`, `HookResult` |
| `memory` | `AccessScope`, `MemoryConfig`, `MemoryEntry`, `MemoryTier`, `WriteMode` |
| `mesi` | `MesiConfig` |
| `messaging` | `AgentMessage`, `MessageAttachment` |
| `persistence` | `PersistenceConfig` |
| `services` | `WorkflowServices`, `WorkflowServicesConfig` |
| `user_input` | `UserInputHandler`, `UserInputRequest` |
| `verification` | `CompletionVerifierConfig`, `RepairContextConfig`, `VERIFIER_OUTPUT_CONTRACT`, `VerificationDecision`, `VerificationKind`, `VerificationRecord` |
| `voting` | `BallotConfig`, `BallotResult`, `CastVote`, `TieBreak`, `VoteOption` |

Public modules also include `context`, `hooks::context`, `memory::tools`, `messaging::tools`, `persistence::state`, and feature-gated `python`. `workflow::engine` is declared public but provides no externally accessible engine type; execution is exposed through `Workflow` and `AgentHandle`.

| Need | Entry point |
| --- | --- |
| Build and configure a workflow | `Workflow::builder()` / `WorkflowBuilder` |
| Sequential, parallel, or background agent execution | `run_agent`, `continue_agent`, `run_agents_parallel`, `spawn_agent` |
| Monitor progress or control lifecycle | `subscribe_events`, hooks, `pause`, `resume`, `finish` |
| Share structured text and deliver messages | `MemoryConfig`, `MemoryStore`, `MessageBus` |
| Track stale files, directories, memory, or custom resources | `ContextFreshnessConfig`, `ResourceTracker`; legacy `MesiConfig` |
| Gate completion, retain repair context | `CompletionVerifierConfig` |
| Condense selected tool outputs | `DistillationConfig`, custom tool result interceptors |
| Enforce and audit spending | Workflow/agent budgets, manual costs, `CostLedger` |
| Checkpoint, restore, or reconcile uncertain side effects | `PersistenceConfig`, durable steps |
| Let agents delegate or vote | `WorkflowServicesConfig`, `enable_voting` |
| Route human input | `UserInputHandler`, or handle `UserInputRequired` and continue |
| Add domain tools | `global_tool`, `agent_tool` |

## Workflow construction

Source: [workflow/mod.rs](src/workflow/mod.rs). Types: `workflow::Workflow`, `workflow::WorkflowBuilder` (both root-re-exported).

### Builder signatures and defaults

`Workflow::builder() -> WorkflowBuilder`, `WorkflowBuilder::new() -> Self`, and `WorkflowBuilder::default() -> Self` create an empty builder. Required: a name, an existing workspace directory, an `Arc<LLMClient>`, and at least one uniquely named agent. The default has no memory tiers, spending limits, hooks, persistence, freshness tracking, custom tools, interceptors, verifiers, distillers, workflow services, input handler, or voting. Metadata defaults to empty strings/lists. Tool interception defaults to `InterceptionFailurePolicy::UseRawResult`.

| Signature | Parameters, purpose, and default |
| --- | --- |
| `name(self, name: impl Into<String>) -> Self` | Workflow identity used in results, hooks, and checkpoint name checks. Required. |
| `workspace(self, path: impl Into<PathBuf>) -> Self` | Existing working directory for every agent. Required; the builder does not create it. |
| `llm_client(self, client: Arc<LLMClient>) -> Self` | Shared configured model client. Required; Rust builder does not load credentials from environment itself (use `LLMClient::from_env()` to configure every provider found in the environment). |
| `agent(self, name: impl Into<String>, config: AgentConfig) -> Self` | Register a configured agent under a unique workflow name. At least one required. |
| `metadata(self, metadata: WorkflowMetadata) -> Self` | Replace discovery metadata, including supported-host restrictions. |
| `memory(self, config: MemoryConfig) -> Self` | Configure shared memory tiers and access policies. Default empty. |
| `budget(self, limit: f64) -> Self` | Overall cost limit in dollars, finite and nonnegative. Default unlimited. |
| `budget_warning(self, threshold: f64) -> Self` | One-time warning threshold in dollars, finite and nonnegative. Default absent. |
| `agent_budget(self, agent: impl Into<String>, limit: f64) -> Self` | Cumulative limit across runs of the named agent; finite and nonnegative. Default unlimited. |
| `hook(self, point: HookPoint, callback: HookCallback) -> Self` | Append synchronous callback at a lifecycle point. Callbacks retain registration order. |
| `persistence(self, config: PersistenceConfig) -> Self` | Enable checkpoints, event/task files, persistent run identity, and durable steps. |
| `resume_from(self, checkpoint: impl Into<PathBuf>) -> Self` | Restore a checkpoint JSON file while building; also configure matching persistence. This accepts a file, not a run-directory shortcut. |
| `ephemeral(self) -> Self` | Clear persistence and resume settings currently on the builder. Subsequent setters can enable them again. |
| `enable_mesi(self) -> Self` | Enable legacy MESI defaults and derive equivalent generalized freshness tracking unless explicitly configured. |
| `mesi(self, config: MesiConfig) -> Self` | Configure legacy tracking and its generalized freshness compatibility mapping. |
| `context_freshness(self, config: ContextFreshnessConfig) -> Self` | Enable versioned resource tracking; an explicit config takes precedence over a derived MESI mapping. |
| `global_tool(self, tool: Arc<dyn arko_agents::tool::Tool>) -> Self` | Register a developer tool for every agent and add its name to their available tools. |
| `agent_tool(self, agent_name: impl Into<String>, tool: Arc<dyn arko_agents::tool::Tool>) -> Self` | Register a tool for one configured agent. |
| `tool_result_interceptor(self, interceptor: Arc<dyn ToolResultInterceptor>) -> Self` | Append a global tool-output interceptor. |
| `agent_tool_result_interceptor(self, agent_name: impl Into<String>, interceptor: Arc<dyn ToolResultInterceptor>) -> Self` | Append an interceptor for one agent, after global interceptors. |
| `interception_failure_policy(self, policy: InterceptionFailurePolicy) -> Self` | Set error handling for the interceptor chain. Default `UseRawResult`. |
| `completion_verifier(self, config: CompletionVerifierConfig) -> Self` | Append a gate on matching agents' `complete_task` calls. |
| `distiller(self, config: DistillationConfig) -> Self` | Append a configurable LLM output distiller, after custom interceptors. |
| `workflow_services(self, config: WorkflowServicesConfig) -> Self` | Enable bounded agent-to-agent delegation through a live workflow. |
| `user_input_handler(self, handler: Arc<dyn UserInputHandler>) -> Self` | Handle agents' `ask_user` questions asynchronously. Default suspends and returns an input-required error. |
| `enable_voting(self) -> Self` | Create an in-memory ballot store and inject `cast_vote` into all agents. |
| `build(self) -> Result<Workflow, WorkflowError>` | Validate, initialize/restore state, and fire `PreWorkflow`. May create persistence files; does not run an agent. |

### Build validation and tool composition

`build` rejects missing required settings; nonexistent/non-directory workspaces; unsupported hosts when platform restrictions are supplied; duplicate agent names; invalid workflow/agent budget values; unknown agent names in agent tools, agent interceptors, agent budgets, or completion verifier definitions; invalid distillation numeric settings or reasoning/model provider mismatches; and incompatible checkpoint identity/name. A warning threshold need not be below a budget limit. Workflow metadata dependencies are descriptive: executables are not installed or checked by the builder.

Duplicate global tool names, duplicate names within one agent-local list, and collisions between global and agent-local tool names are rejected. Global tools cannot use reserved names: `complete_task`, `ask_user`, `read_memory`, `write_memory`, `list_memories`, `send_message`, `run_workflow_agent`, `cast_vote`. The same reserved-name check is not applied to agent-local registrations; choose distinct names so built-in behavior is preserved. Configured built-ins and skills belong to `arko_agents`; this crate augments that configuration with workflow tools.

Automatic additions: `send_message` for every agent; memory tools when at least one memory tier exists; `run_workflow_agent` for authorized callers when services are enabled; `cast_vote` when voting is enabled. Multi-agent prompts include the sorted agent roster with role descriptions and queued messages. Skills configured on an agent are initialized against the workspace. No separate workflow graph or scheduler configuration is required.

Interceptors run in this order: globals, that agent's interceptors, then distillers; registration order is retained within each group. Each stage receives the preceding stage's effective result. Tool hooks, freshness policy, and verification operate at execution time, including automatically approved and batched calls.

### Workflow metadata

Source: [workflow/config.rs](src/workflow/config.rs). All four types serialize/deserialize through Serde. Only `WorkflowMetadata` implements `Default`.

| Type | Public fields and meaning |
| --- | --- |
| `SupportedPlatform` | `os: String` — Rust host OS name; `arch: Option<String>` — optional architecture restriction, default `None` when omitted in JSON. |
| `WorkflowDependency` | `name: String`, `description: String` — required descriptive values; `executable: Option<String>`, `install_hint: Option<String>`, `os: Option<String>`, `arch: Option<String>` — optional command/install/platform metadata, JSON defaults `None`. No enforcement method. |
| `WorkflowMetadata` | `description: String`, `version: String` default `""`; `supported_platforms: Vec<SupportedPlatform>`, `required_dependencies: Vec<WorkflowDependency>`, `tags: Vec<String>` default empty. Empty platforms means unrestricted host. |
| `WorkflowConfig` | `name: String`, `workspace: PathBuf`, `metadata: WorkflowMetadata` (metadata defaults when missing in JSON). A serializable descriptor; use the builder to construct an executable workflow. |

`SupportedPlatform` methods:

```rust
fn new(os: impl Into<String>) -> Self;
fn arch(self, arch: impl Into<String>) -> Self;
fn matches_host(&self) -> bool;
```

`new` has no architecture restriction. `matches_host` compares OS and, when present, architecture case-insensitively against `std::env::consts::{OS, ARCH}`. Multiple supported platforms are alternatives.

## Workflow execution and lifecycle

Source: [workflow/mod.rs](src/workflow/mod.rs), [execution behavior](src/workflow/engine.rs). All methods below belong to `Workflow`.

### Run, continue, and spawn

| Signature | Parameters, result, and behavior |
| --- | --- |
| `async run_agent(&self, agent_name: &str, prompt: &str) -> Result<AgentResult, WorkflowError>` | Start a fresh invocation of a registered agent and await completion. Adds queued messages/roster, drives events, records costs/results, and runs configured lifecycle policies. Unknown names return `AgentNotFound`. |
| `async continue_agent(&self, previous: &AgentResult, prompt: &str) -> Result<AgentResult, WorkflowError>` | Resume a `Completed` or `WaitingForInput` result, preserving task/conversation/run ID. `prompt` is the follow-up task or user answer. Other statuses are rejected. Accumulated results replace the prior result with that same run ID; task token/cost totals are cumulative. |
| `async run_agents_parallel(&self, agents: Vec<(&str, String)>) -> Result<Vec<AgentResult>, WorkflowError>` | Concurrent named-agent/prompt pairs, including multiple invocations of one configured agent. Awaits all invocations; on success returns input order. On any failure returns the first error encountered in input join order after joining all; accumulated context retains finalized outcomes. |
| `async spawn_agent(&self, agent_name: &str, prompt: &str) -> Result<AgentHandle, WorkflowError>` | Prepare/start the runtime and return its handle. Drive it with `wait` or raw channels. This does not automatically run the workflow event driver in another task. |

The managed driver automatically approves pending single and batch tool requests, permits one approval-required error-recovery attempt per invocation, and cancels on an exceeded budget or persistent interruption. Errors can be returned after an `AgentResult` has been recorded; inspect `result().agent_results` for finalized failures or input suspensions. `run_agent`, `continue_agent`, and `run_agents_parallel` fire `OnWorkflowFailed` when returning errors; the spawn/handle path does not wrap errors with that hook.

### AgentHandle

`AgentHandle` has no public constructor or fields; obtain it with `spawn_agent`.

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

`agent_name` borrows the configured name. `wait` consumes the handle, drives workflow events, finalizes/persists the task, and returns a completed result or error. `into_parts` consumes it into event receiver, command sender, task join handle, and agent name. Raw driving transfers event handling, approvals, usage accounting, event broadcasting/persistence, final result recording, and message acknowledgment/cleanup to the caller. Execution middleware and interceptors were already installed at spawn, so their tool-level behavior still runs. Use the managed path when you need the complete default workflow lifecycle.

### Results, state, costs, events, and pause

| Signature | Purpose and behavior |
| --- | --- |
| `context(&self) -> &Arc<SharedContext>` | Borrow shared stores, recorded results, and extension state. See [Shared context](#shared-context). |
| `result(&self) -> WorkflowResult` | Snapshot current aggregate results, metadata, and cost ledger. Does not finish or checkpoint. |
| `cost_ledger(&self) -> CostLedger` | Clone the chronological automatic/manual cost ledger. |
| `add_manual_cost(&self, amount: f64, description: &str, metadata: Option<String>) -> Result<AddCostResult, WorkflowError>` | Add finite, nonnegative dollar cost plus human-readable description and arbitrary metadata string. Fires warning/exceeded hooks. An over-limit cost stays recorded even when `BudgetExceeded` is returned. Invalid amount returns `InvalidCost`. |
| `subscribe_events(&self) -> tokio::sync::broadcast::Receiver<WorkflowEvent>` | Subscribe to future managed agent events. Buffer is 1,024 events; slow receivers can lag. Subscribe before starting runs; no historical replay here. |
| `async checkpoint(&self) -> Result<(), WorkflowError>` | Save configured persistence snapshot. Successful no-op without persistence. |
| `async finish(&self) -> Result<WorkflowResult, WorkflowError>` | Fire `PostWorkflow`, checkpoint, return snapshot. Caller decides when work is complete; this does not join outstanding agents or lock the workflow against later runs. Repeated calls repeat the hook/checkpoint. |
| `pause(&self) -> Result<(), WorkflowError>` | Set pause flag and fire `OnWorkflowPause` only on transition. Managed event drivers stop processing at their next event boundary; ongoing model/tool work is not forcibly suspended. |
| `resume(&self) -> Result<(), WorkflowError>` | Clear flag, fire `OnWorkflowResume` on transition, wake event drivers. |
| `is_paused(&self) -> bool` | Read current pause flag. Default false, including after checkpoint restore. |

### Workflow-facing resource and ballot operations

```rust
fn register_resource_read(&self, actor: ActorId, resource: ResourceId)
    -> Result<(), WorkflowError>;
async fn mark_resource_dirty(&self, actor: ActorId, resource: ResourceId)
    -> Result<Vec<ActorId>, WorkflowError>;
fn begin_ballot(&self, config: BallotConfig) -> Result<Uuid, WorkflowError>;
fn finish_ballot(&self, ballot_id: Uuid) -> Result<BallotResult, WorkflowError>;
```

The resource methods require freshness tracking (`BuilderInvalid` otherwise). `actor` identifies the observer/writer; `resource` identifies the changed context. `register_resource_read` establishes a version baseline. `mark_resource_dirty` increments version, returns stale readers, injects notices into active reader tasks, and fires `OnMesiConflict`; mutation precedes notification/hook errors. See the resource tracker section for identity and overlap rules.

Ballot methods require `enable_voting` (`BuilderInvalid` otherwise). `begin_ballot` validates/stores the config and returns its generated UUID. `finish_ballot` evaluates current votes; it does not close the ballot. Validation/tally errors map to `WorkflowError::Deliberation`.

### Workflow-facing durable operations

All require configured persistence; otherwise return `DurableUnavailable`. See the durable section for spec identity, retry, replay, and resolution contracts.

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

`spec` names/versions an operation, `input` establishes its persisted input identity, and `execute` performs one attempt with the supplied context. Completed steps replay typed `O` without executing again. `resolver` reconciles uncertain prior outcomes where policy calls for it. `in_doubt_steps` lists uncertain attempts. Operator methods record an externally established output, permanent failure, or authorization for a subsequent retry; `actor` and `reason` are audit strings, not automatic authorization checks.

## Result and event types

Source: [workflow/result.rs](src/workflow/result.rs). Types are root- and prelude-re-exported and Serde serializable/deserializable.

| Type | Public fields/variants |
| --- | --- |
| `AgentRunStatus` | `Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`. |
| `AgentResult` | `agent_run_id: Uuid` — stable task identity; `agent_name: String`; `status: AgentRunStatus`; `summary: String` — completion text, otherwise usually empty; `error: Option<String>` — failure/interruption reason or unanswered questions; `task: arko_agents::Task` — conversation, usage, and runtime state. |
| `WorkflowResult` | `workflow_run_id: Uuid`, `workflow_name: String`, `metadata: WorkflowMetadata`, `total_cost: f64` — running budget counter in dollars; `cost_ledger: CostLedger`; `agent_results: Vec<AgentResult>` — finalized outcomes, replacing earlier entries for continued run IDs. |
| `WorkflowEvent` | `timestamp: chrono::DateTime<chrono::Utc>`, `workflow_run_id: Uuid`, `agent_name: String`, `agent_run_id: Uuid`, `event: arko_agents::runtime::event::AgentEvent`. Wrapper for broadcast/persisted runtime events. |

`AgentResult` methods:

```rust
fn total_cost(&self) -> f64;
fn total_input_tokens(&self) -> usize;
fn total_output_tokens(&self) -> usize;
```

These return task-wide usage totals, including earlier conversation segments after continuation; `total_cost` is the billed amount (`UsageInfo::billed()`, zero for subscription calls such as ChatGPT OAuth). Workflow cost also includes manual charges; summing task costs need not match the workflow ledger.

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

Callbacks are synchronous and receive immutable context; capture shared state for application effects. Registration appends to the selected point; `get` borrows its ordered callbacks and `has_hooks` checks for a nonempty list. `Continue` proceeds to the next callback. The first `Skip` or `Abort` stops that point's callback chain. `Skip` rejects the current tool only at `PreToolExecution`; elsewhere it has no skip effect beyond stopping the callback chain. Hook results cannot replace tool output; use a tool result interceptor.

`Abort(reason)` on ordinary lifecycle dispatch returns `AbortedByHook`. Tool-boundary aborts surface as tool execution errors; they do not guarantee global cancellation. Automatic usage-report budget hooks ignore hook-returned errors, while manual-cost hooks propagate them. Hooks do not provide transaction rollback for operations already performed.

### HookPoint variants

| Variant | Trigger and additional context |
| --- | --- |
| `PreWorkflow` | Successful setup near end of `build`; before returning the workflow. |
| `PostWorkflow` | Explicit `finish`, before its checkpoint. |
| `PreAgentSpawn` | Before run/spawn/continuation setup; `agent_name`, plus prior `agent_run_id` for continuation. |
| `PostAgentCompletion` | Finalized completed task; agent name/run ID and nonempty summary. |
| `PostAgentFailure` | Finalized failed/cancelled task; name/run ID and error. |
| `PostAgentInterruption` | Finalized interruption or unexpected nonterminal ending; name/run ID and reason. |
| `OnUserInputRequired` | Finalization of a task suspended awaiting input; name/run ID, questions in `error_message`. A handled live question does not trigger this suspension hook. |
| `PreToolExecution` | Before a tool call; agent name/run ID, tool name/call ID, parameters. Can reject with `Skip`. |
| `PostToolExecution` | After a tool outcome; same identities/parameters, `tool_result` and `original_tool_result` contain pre-interception output or error text. |
| `PostToolResultInterception` | Managed driver receives an intercepted-result event; tool identity, raw `original_tool_result`, delivered `effective_tool_result` and `tool_result`. |
| `OnWorkflowPause` | Pause flag changes false→true. |
| `OnWorkflowResume` | Pause flag changes true→false, before waking paused event drivers. |
| `OnWorkflowFailed` | Public managed run/continue/parallel call returns an error; error message and available agent identity. |
| `OnBudgetExceeded` | A recorded automatic/manual cost strictly exceeds workflow limit. Contains updated spending. |
| `OnBudgetWarning` | Warning threshold first reached, with updated spending. |
| `OnAgentMessage` | `send_message` tool sends a message; sender name/task ID, content as `tool_result`, recipient in `metadata["to"]`. Direct bus sends do not dispatch it. |
| `OnMesiConflict` | Successful resource writes invalidate readers; stale reader identity and `metadata["resource"]`, `metadata["writer"]`. Also used for memory writes and `mark_resource_dirty`; it is not a generic event for every rejected stale write. |

### HookContext fields

`hooks::context::HookContext` has these public fields. Optional fields are populated only at applicable points.

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | `chrono::DateTime<chrono::Utc>` | Hook invocation time. |
| `hook_point` | `HookPoint` | Current lifecycle point. |
| `workflow_run_id` | `Uuid` | Workflow identity. |
| `workflow_name` | `String` | Configured name. |
| `agent_run_id` | `Option<Uuid>` | Task identity when available. |
| `agent_name` | `Option<String>` | Agent name when available. |
| `tool_name` | `Option<String>` | Tool being observed. |
| `tool_call_id` | `Option<Uuid>` | Specific tool invocation. |
| `tool_params` | `Option<HashMap<String, Value>>` | Tool input parameters. |
| `tool_result` | `Option<String>` | Output relevant to this point. |
| `original_tool_result` | `Option<String>` | Raw output before interception. |
| `effective_tool_result` | `Option<String>` | Replacement output after interception. |
| `agent_summary` | `Option<String>` | Nonempty completion summary. |
| `error_message` | `Option<String>` | Error/reason/questions at applicable points. |
| `budget_spent` | `f64` | Current workflow dollars spent. |
| `budget_limit` | `Option<f64>` | Configured total limit. |
| `metadata` | `HashMap<String, Value>` | Point-specific resource/writer/recipient data. |

### UserInputHandler and UserInputRequest

```rust
#[async_trait::async_trait]
pub trait UserInputHandler: Send + Sync {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String>;
}
```

`UserInputRequest` fields: `workflow_run_id: Uuid`, `agent_run_id: Uuid`, `agent_name: String`, `questions: String`. Implement `respond` and pass `Arc<dyn UserInputHandler>` to the builder. The returned string is submitted as the user's response; an error string becomes `WorkflowError::Channel`.

Without a handler, the managed driver suspends on `ask_user` and returns `WorkflowError::UserInputRequired { agent, task_id, questions }`. Find the matching `WaitingForInput` result in `workflow.result().agent_results`, obtain an answer externally, then pass that result and answer to `continue_agent`. A checkpoint can preserve the suspended task when persistence is enabled.

## Workflow services and custom tools

Source: [services/mod.rs](src/services/mod.rs), [builder integration](src/workflow/mod.rs).

`WorkflowServicesConfig` configures agent-callable delegation. Its public fields and default:

| Field | Type | Default/meaning |
| --- | --- | --- |
| `allowed_callers` | `HashSet<String>` | Empty: every configured caller may delegate. Nonempty: only listed callers receive the tool. |
| `allowed_targets` | `HashSet<String>` | Empty: any configured target. Nonempty: restrict target names independently from callers. |
| `max_active_calls` | `u32` | `4`; shared simultaneous service-call limit, including nested delegation. |

```rust
// WorkflowServicesConfig: Default
fn allow_caller(self, agent: impl Into<String>) -> Self;
fn allow_target(self, agent: impl Into<String>) -> Self;
fn max_active_calls(self, max: u32) -> Self;

// WorkflowServices: Clone
fn is_allowed(&self, caller: &str) -> bool;
fn is_target_allowed(&self, target: &str) -> bool;
async fn run_agent(&self, caller: &str, agent_name: &str, prompt: &str)
    -> Result<AgentResult, WorkflowError>;
```

Allow setters insert names; `max_active_calls` clamps to at least one. Setting the public field directly bypasses that clamp. Names in service allowlists are not checked at build time; targets still must resolve to a configured agent at execution. Obtain a service reference with `workflow.context().workflow_services.get()` after enabling it. There is no public constructor. A cloned service holds a weak workflow-context reference and fails if the workflow context has been dropped.

`run_agent` checks caller and target policy, then runs the target with shared memory, budget, hooks, persistence, and configured completion verifiers. Unauthorized names return `BuilderInvalid`; a reached concurrency limit returns `Channel` immediately rather than queuing. The `caller` parameter is a host-supplied identity, not an authentication mechanism. No call-depth or cycle policy is supplied beyond the active-call limit.

Agent tool constant: `services::RUN_WORKFLOW_AGENT_TOOL_NAME: &str = "run_workflow_agent"`.

| Tool | JSON parameters | Output |
| --- | --- | --- |
| `run_workflow_agent` | Required `agent: string` (configured allowed target), `prompt: string` (bounded task). | Target's completion summary string, after any configured verification. Missing strings are tool parameter errors; workflow failures become execution errors. Not batch-safe. |

Custom domain tools implement `arko_agents::tool::Tool` and are registered through `global_tool` or `agent_tool`; the library automatically exposes their names to the selected agents. Use `ToolResultInterceptor` for output transformation and the freshness APIs for custom resource effects. These extension traits and their full definitions belong to `arko_agents`; this index records the integration surface without duplicating that crate's API.

## Shared context

Source: [context/mod.rs](src/context/mod.rs). `context::SharedContext` is accessed through `Workflow::context`. It has public fields below and no public constructor or inherent methods. Lock-protected stores expose live workflow state; prefer workflow methods when their hook, notification, or accounting behavior is needed.

| Field | Type | Purpose |
| --- | --- | --- |
| `workflow_run_id` | `Uuid` | Current persistent or ephemeral run identity. |
| `workflow_name` | `String` | Workflow name. |
| `workflow_metadata` | `WorkflowMetadata` | Discovery metadata. |
| `workspace` | `PathBuf` | Shared working directory. |
| `budget` | `BudgetManager` | Ledger and limits. |
| `durable_store` | `Option<Arc<durable::DurableStore>>` | Present when persistence is configured. |
| `memory` | `memory::MemoryStore` | Shared tier store. |
| `messages` | `messaging::MessageBus` | Queued and live messaging. |
| `file_tracker` | `Option<mesi::FileTracker>` | Legacy tracker. |
| `resource_tracker` | `Option<freshness::ResourceTracker>` | Generalized freshness tracker. |
| `hooks` | `Arc<hooks::HookRegistry>` | Registered callbacks. |
| `agent_configs` | `HashMap<String, AgentConfig>` | Configured agents keyed by workflow name. |
| `global_tools` | `Vec<Arc<dyn arko_agents::tool::Tool>>` | Tools for every agent. |
| `agent_tools` | `HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>` | Agent-specific tools. |
| `agent_results` | `RwLock<Vec<AgentResult>>` | Finalized results, including suspended/failed runs. |
| `global_interceptors` | `Vec<Arc<dyn ToolResultInterceptor>>` | Global interceptor chain. |
| `agent_interceptors` | `HashMap<String, Vec<Arc<dyn ToolResultInterceptor>>>` | Per-agent chains. |
| `interception_failure_policy` | `InterceptionFailurePolicy` | Interceptor error handling. |
| `completion_verifiers` | `Vec<verification::CompletionVerifierConfig>` | Completion gates in order. |
| `verification_attempts` | `Mutex<HashMap<(Uuid, String), (u32, u32)>>` | Per-target-run/verifier rejection counters `(minor, major)`. |
| `verification_records` | `RwLock<Vec<verification::VerificationRecord>>` | Verification audit history. |
| `distillation_configs` | `Vec<distillation::DistillationConfig>` | Distillers in order. |
| `distillation_records` | `RwLock<Vec<distillation::DistillationRecord>>` | Distillation audit history. |
| `workflow_services` | `OnceLock<services::WorkflowServices>` | Initialized only when services enabled. |
| `paused` | `std::sync::atomic::AtomicBool` | Event-driver pause flag. |
| `resume_notify` | `tokio::sync::Notify` | Event-driver wake notification. |
| `user_input_handler` | `Option<Arc<dyn user_input::UserInputHandler>>` | Human response routing. |
| `ballots` | `Option<Arc<voting::BallotStore>>` | In-memory structured votes. |
| `event_tx` | `tokio::sync::broadcast::Sender<WorkflowEvent>` | Live managed event stream. |


## Memory

Module: `arko_workflows::memory`. Sources: [store and configuration](src/memory/mod.rs), [agent tools](src/memory/tools.rs), [read response behavior](src/memory/reads.rs).

Memory provides named tiers with independent read/write access, append or replacement writes, write history, and optional size limits. `MemoryStore` and `MemoryConfig` default to empty. In signatures below, `Uuid = uuid::Uuid`, `DateTime<Utc> = chrono::DateTime<chrono::Utc>`, and collection types are from `std::collections`.

| Public type | Fields or variants; purpose and defaults |
| --- | --- |
| `AccessScope` | `All`, `Only(HashSet<String>)`, `None`. Controls access using exact agent names. `fn allows(&self, agent_name: &str) -> bool` checks membership. |
| `WriteMode` | `AppendOnly` (Rust default), `FullEdit`. Append adds a newline when existing content is nonempty; full edit replaces the current content. Both record a new history entry. |
| `MemoryEntry` | `id: Uuid`, `author: String`, `task_id: Option<Uuid>`, `timestamp: DateTime<Utc>`, `content: String`. Each successful write records its supplied content, a generated UUID, and current UTC timestamp. Missing serialized `task_id` defaults to `None`. |
| `MemoryTier` | `name: String`, `content: String`, `readers: AccessScope`, `writers: AccessScope`, `write_mode: WriteMode`, `entries: Vec<MemoryEntry>`, `max_bytes: Option<usize>`. Missing serialized `entries` defaults to `[]`; `max_bytes` defaults to `None`. No `Default` implementation. |

These four data types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`. Enum serialization retains Rust variant names, e.g. `"AppendOnly"`, `"All"`, or `{"Only":["writer"]}`. `WriteMode`'s Rust default does not make a missing `MemoryTier.write_mode` field valid during deserialization.

### Memory configuration

All methods are on `MemoryConfig`; builder methods consume and return the configuration.

| Signature | Purpose and parameter behavior |
| --- | --- |
| `fn new() -> Self` | Empty configuration. |
| `fn tier(self, tier: MemoryTier) -> Self` | Add a fully configured tier, including initial content, policies, history, and byte limit. |
| `fn global_tier(self, name: impl Into<String>) -> Self` | Add an empty append-only tier readable and writable by all agents; no limit. |
| `fn broadcast_tier(self, name: impl Into<String>, writers: impl IntoIterator<Item = impl Into<String>>) -> Self` | Add an empty append-only tier readable by all and writable only by the supplied names; no limit. |
| `fn global_tier_with_limit(self, name: impl Into<String>, max_bytes: usize) -> Self` | Global append-only tier with a maximum resulting UTF-8 content size in bytes. |
| `fn build(self) -> MemoryStore` | Construct the store. Duplicate tier names resolve to the last supplied tier. |

### Memory store

All methods are on `MemoryStore`; read/write methods take an agent name whose permissions are checked against the selected tier.

| Signature | Purpose, result, and constraints |
| --- | --- |
| `fn new(tiers: Vec<MemoryTier>) -> Self` | Create a store from tier definitions; duplicate names use the last definition. Initial content/history are accepted as supplied. |
| `fn is_empty(&self) -> bool` | Whether no tiers exist. |
| `fn read(&self, agent_name: &str, tier_name: &str) -> Result<String, WorkflowError>` | Clone the current complete content after checking read access. |
| `fn read_versioned(&self, agent_name: &str, tier_name: &str) -> Result<(usize, String), WorkflowError>` | Atomically read `(version, content)`; version is the tier's history-entry count. |
| `fn read_window(&self, agent_name: &str, tier_name: &str, max_chars: Option<usize>) -> Result<String, WorkflowError>` | Return all content, or its final `max_chars` Unicode scalar values prefixed with `[showing final {max_chars} of {count} characters]\n` when truncated. This low-level method accepts zero. |
| `fn entries(&self, agent_name: &str, tier_name: &str) -> Result<Vec<MemoryEntry>, WorkflowError>` | Clone the write history after checking read access. Full-edit writes retain earlier history. |
| `fn write(&self, agent_name: &str, tier_name: &str, content: &str) -> Result<(), WorkflowError>` | Apply the tier's write mode and append history with no task ID. |
| `fn write_with_context(&self, agent_name: &str, task_id: Option<Uuid>, tier_name: &str, content: &str) -> Result<(), WorkflowError>` | Same write operation, with the task ID recorded in history. |
| `fn list_accessible(&self, agent_name: &str) -> Vec<(String, bool, bool)>` | Return `(tier_name, can_read, can_write)` for **all** tiers, including `(false, false)`. Order is unspecified. |
| `fn snapshot(&self) -> Vec<MemoryTier>` | Clone all tiers, policies, content, and history for persistence; no agent access check. Order is unspecified. |

Unknown tiers return `MemoryTierNotFound`; denied operations return `MemoryAccessDenied`; writes exceeding the resulting content's byte limit return `MemoryLimitExceeded` without changing content/history. The appended newline counts toward the byte limit; history size is not limited. Direct store operations do not perform resource-freshness checks or send notifications; the memory tools provide that integration.

### Memory tools and read JSON contract

Public types in `memory::tools`: `ReadMemoryTool`, `WriteMemoryTool`, `ListMemoriesTool`. Each has `fn new(context: Arc<crate::context::SharedContext>, agent_name: String) -> Self`, implements `Debug`, and implements `arko_agents::tool::Tool`. Their implemented common trait signatures are:

```rust
fn definition(&self) -> arko_agents::tool::ToolDefinition;
async fn execute(
    &self,
    task_context: &arko_agents::tool::TaskContext<'_>,
    params: HashMap<String, serde_json::Value>,
) -> Result<String, arko_agents::tool::ToolError>;
```

| Constant (`pub const …: &str`) / tool name | Input parameters | Output and behavior |
| --- | --- | --- |
| `READ_MEMORY_TOOL_NAME = "read_memory"` | Required `tier: string`; optional `full: boolean` (default `false`; also accepts exact strings `"true"`/`"false"`); optional `max_chars: string` containing a positive integer. | Versioned JSON described below. Batched calls supported. Read permission required. A complete read records resource freshness when tracking is enabled. |
| `WRITE_MEMORY_TOOL_NAME = "write_memory"` | Required `tier: string`, `content: string`. | `Successfully wrote to memory tier '{tier}'.` Applies tier policy and records the current task ID. Checks for stale memory reads; blocks conflicts when configured, otherwise writes. Successful writes attempt notifications to stale task readers and fire `OnMesiConflict` for each tracked stale reader, including inactive readers. Batched calls unsupported. |
| `LIST_MEMORIES_TOOL_NAME = "list_memories"` | None. | Text headed `Available memory tiers:` with tier names and `read/write`, `read-only`, `write-only`, or `no access`; or `No memory tiers are configured in this workflow.` Batched calls supported. |

Every successful `read_memory` response is an object with `tier: string`, `version: integer`, and `kind`; the additional fields depend on `kind`:

| `kind` | Additional fields | How to consume it |
| --- | --- | --- |
| `"full"` | `content: string` | Replace the local view with the full current snapshot. |
| `"append"` | `base_version: integer`, `content: string` | Append `content` exactly to the snapshot at `base_version`; it includes any required newline. |
| `"replace"` | `base_version: integer`, `start_line: integer`, `delete_lines: integer`, `content: string` | On the snapshot at `base_version`, replace `delete_lines` starting at zero-based `start_line` with `content`. Lines retain their newline terminators, equivalent to Rust `split_inclusive('\n')`; concatenating the resulting lines yields exact current content. |
| `"unchanged"` | None. | Retain the prior content; adopt the returned current version, which can advance even if text did not change. |
| `"partial"` | `total_chars: integer`, `content: string` | Trailing `max_chars` Unicode scalar values of a longer snapshot. This is an incomplete view, not a reconstructable delta. |

The first complete read for a task/tier is full. Later reads compare against the latest complete response confirmed as delivered into that task's context. Changes use one exact append or line-replacement operation; a full response is returned when the encoded delta would be as large or larger. Unchanged responses always use `"unchanged"`. Use `full=true` to request a fresh complete snapshot; a truncating `max_chars` still takes precedence and produces `"partial"`. Partial responses neither advance the complete baseline nor count as a complete freshness read. Concurrent pending responses do not advance the baseline merely by being prepared.

`ReadMemoryTool` additionally implements these `Tool` methods:

```rust
fn allow_result_interception(&self) -> bool; // false: preserve the exact read contract
fn reset_context(&self, task_id: uuid::Uuid); // discard this task's read baselines/pending reads
fn observe_context(&self, task_id: uuid::Uuid, message: &arko_ai::Message);
```

`observe_context` confirms delivery from successful matching user-role tool results or user-role text containing a pending response. When driving these tools outside the workflow engine, forward context observations and resets to preserve correct delta behavior. Tool parameter errors use `MissingParameter`/`ParameterValidation`; store or freshness failures become `ExecutionFailed`. A post-write hook failure can return an error after memory has already changed.

## Messaging

Module: `arko_workflows::messaging`. Sources: [message bus and data](src/messaging/mod.rs), [agent tool](src/messaging/tools.rs).

Messages target an agent name in a fixed roster. They remain queued until drained or acknowledged; active task instances receive an injection at their next safe turn boundary. All active instances of the recipient name receive the live injection. In signatures, `Uuid = uuid::Uuid` and `UserCommand = arko_agents::runtime::command::UserCommand`.

| Public type | Fields or variants |
| --- | --- |
| `AgentMessage` | `id: Uuid`, `from: String`, `from_task_id: Option<Uuid>`, `to: String`, `content: String`, `attachments: Vec<MessageAttachment>`, `timestamp: chrono::DateTime<chrono::Utc>`. Deserialization defaults: missing `id` is the nil UUID, `from_task_id` is `None`, and `attachments` is `[]`. New messages receive a fresh UUID and UTC timestamp. |
| `MessageAttachment` | `File { path: String }`, `MemoryTier { tier: String }`, `Transcript { task_id: Uuid }`, `Text { name: String, content: String }`. Typed references or inline text; references do not themselves read files, memory, or transcripts. |

Both types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`. Attachment JSON uses an internal `type` tag with snake-case names:

```json
[
  {"type":"file","path":"src/lib.rs"},
  {"type":"memory_tier","tier":"notes"},
  {"type":"transcript","task_id":"00000000-0000-0000-0000-000000000001"},
  {"type":"text","name":"summary","content":"Review the parser."}
]
```

### Message bus

All methods are on `MessageBus`; `Default` is an empty roster with empty queues and no active recipients.

| Signature | Purpose and behavior |
| --- | --- |
| `fn new(roster: impl IntoIterator<Item = String>) -> Self` | Construct a bus with valid recipient names and empty queues. |
| `fn from_snapshot(roster: impl IntoIterator<Item = String>, queues: HashMap<String, Vec<AgentMessage>>) -> Self` | Restore pending queues; active channel registrations are not restored. |
| `fn roster(&self) -> Vec<String>` | Return recipient names sorted alphabetically. |
| `fn register_active(&self, agent_name: &str, task_id: Uuid, sender: tokio::sync::mpsc::Sender<UserCommand>)` | Register a live command channel and a per-task delivery worker. Re-registering replaces the registration and cancels its old worker. Registration outside Tokio is supported; a later send/inject inside Tokio starts delivery. |
| `fn unregister_active(&self, agent_name: &str, task_id: Uuid)` | Remove the registration and cancel forwarding; unacknowledged messages remain queued. |
| `async fn inject_task(&self, task_id: Uuid, content: String) -> bool` | Queue a context notice for a known active task without waiting for its command-channel capacity. Creates a stable-ID `AgentMessage` from `workflow-context`, eligible for snapshot persistence. `true` means queued, not consumed or flushed to disk; an unknown/inactive task returns `false`. |
| `async fn send(&self, from: &str, from_task_id: Option<Uuid>, to: &str, content: String, attachments: Vec<MessageAttachment>) -> Result<AgentMessage, String>` | Validate the recipient and enqueue a stable-ID message. Active delivery workers forward it without blocking the sender on a busy recipient. Return the message. Unknown recipient errors list valid names; closed channels are unregistered and unacknowledged messages remain available. Sender membership is not validated. |
| `fn drain(&self, agent_name: &str) -> Vec<AgentMessage>` | Remove and return the agent's entire pending queue; unknown/empty names return `[]`. |
| `fn pending(&self, agent_name: &str) -> Vec<AgentMessage>` | Clone the pending queue without removing messages. |
| `fn acknowledge(&self, agent_name: &str, message_ids: &[Uuid])` | Remove the supplied IDs from that agent's queue. |
| `fn confirm_active_context(&self, agent_name: &str, task_id: Uuid, conversation: &arko_ai::Conversation)` | Confirm queued IDs whose complete injected payloads appear in finalized user messages with the runtime's workflow-context framing. The engine calls this for completed tasks before checkpointing. Arbitrary quoted IDs in prompts or tool results do not qualify. |
| `fn acknowledge_active(&self, agent_name: &str, task_id: Uuid)` | Remove only context-confirmed message IDs from the agent-wide queue. Command-channel acceptance alone does not acknowledge unread messages. |
| `fn snapshot(&self) -> HashMap<String, Vec<AgentMessage>>` | Clone pending queues for persistence. |

Queueing, channel acceptance, context confirmation, and queue acknowledgment are distinct. Snapshot persistence is explicit; returning from send/inject is not a disk commit. Delivery can repeat after interrupted work, so consumers should retain stable IDs when side effects need deduplication. Direct `MessageBus::send` does not fire lifecycle hooks; the agent tool provides that integration.

### Send-message tool

`messaging::tools::SEND_MESSAGE_TOOL_NAME: &str = "send_message"` names the tool. `SendMessageTool` implements `Debug` and `arko_agents::tool::Tool` and has:

```rust
fn new(context: Arc<crate::context::SharedContext>, agent_name: String) -> Self;
fn definition(&self) -> arko_agents::tool::ToolDefinition;
async fn execute(
    &self,
    task_context: &arko_agents::tool::TaskContext<'_>,
    params: HashMap<String, serde_json::Value>,
) -> Result<String, arko_agents::tool::ToolError>;
```

Parameters: required `to: string` and `content: string`; optional `attachments_json: string` containing an array in the attachment format above (default empty). The tool's definition lists valid recipients. Invalid JSON strings produce `ParameterValidation`; a non-string `attachments_json` currently behaves as absent. Unknown recipients become `ExecutionFailed`. Successful sending records the current task as sender context, fires `OnAgentMessage`, and returns `Message {id} sent to agent '{to}'.` Batched calls are unsupported. A hook failure may return an error after the message has already been queued/delivered.

## Context freshness

Module: `arko_workflows::freshness`. Source: [freshness tracking](src/freshness/mod.rs).

Version-based tracking detects stale reads of files, directory trees, memory tiers, and workflow-defined resources. Actors include task identity so concurrent runs with the same agent name remain distinct. Tracking records explicit observations/mutations; it does not watch the filesystem for outside changes. In signatures, `Uuid = uuid::Uuid`, `PathBuf = std::path::PathBuf`, and `Value = serde_json::Value`.

| Public type | Fields or variants; purpose |
| --- | --- |
| `ResourceId` | `File(PathBuf)`, `DirectoryTree(PathBuf)`, `MemoryTier(String)`, `Custom { namespace: String, key: String }`. Resource identity. |
| `ActorId` | `task_id: Uuid`, `agent_name: String`. Constructor: `fn new(task_id: Uuid, agent_name: impl Into<String>) -> Self`. |
| `ToolResourceEffect` | `ReadFile { parameter: String }`, `ReadDirectoryTree { parameter: String }`, `WriteFile { parameter: String }`. `parameter` names the tool argument containing a path string. |
| `ContextFreshnessConfig` | `block_on_conflict: bool`, `tool_effects: HashMap<String, ToolResourceEffect>`. Default blocks conflicts and uses the mappings below. |
| `FreshnessConflict` | `resource: ResourceId`, `observed_version: u64`, `current_version: u64`, `last_writer: Option<ActorId>`. Identifies a read that became stale. |
| `FreshnessSnapshot` | Opaque snapshot with private fields; `Default` is empty. Use `ResourceTracker::snapshot` and `from_snapshot` to round-trip tracking state, including observations and writes. |

All six data types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`; `ResourceId` and `ActorId` additionally implement `PartialEq`, `Eq`, and `Hash`. `ContextFreshnessConfig` and `FreshnessSnapshot` implement `Default`. Resource/effect enum serialization retains Rust variant names. For example, `{"MemoryTier":"notes"}` and `{"ReadFile":{"parameter":"path"}}`.

| Default tool mapping | Effect and argument |
| --- | --- |
| `read_file` | `ReadFile { parameter: "path" }` |
| `list_files` | `ReadDirectoryTree { parameter: "directory" }` |
| `write_file`, `apply_diff` | `WriteFile { parameter: "path" }` |

`ContextFreshnessConfig::tool_effect(self, tool_name: impl Into<String>, effect: ToolResourceEffect) -> Self` adds or replaces a mapping. To replace all mappings, assign the public `tool_effects` field. The Rust default configuration is not automatically applied to missing JSON fields.

### Resource tracker

All methods are on `ResourceTracker`.

| Signature | Purpose and behavior |
| --- | --- |
| `fn new(workspace: PathBuf, config: ContextFreshnessConfig) -> Self` | Create an empty tracker rooted at the supplied workspace. |
| `fn from_snapshot(workspace: PathBuf, config: ContextFreshnessConfig, snapshot: FreshnessSnapshot) -> Self` | Restore tracking state under the supplied workspace/configuration. |
| `fn snapshot(&self) -> FreshnessSnapshot` | Clone all tracking state. |
| `fn config(&self) -> &ContextFreshnessConfig` | Borrow the current configuration. |
| `fn resource_for_tool(&self, tool_name: &str, parameters: &HashMap<String, Value>) -> Option<(ResourceId, bool)>` | Resolve a configured tool's path argument; the boolean is `true` for a write. Return `None` for an unmapped tool, absent argument, or non-string argument. Relative paths are joined to the workspace and `.`/`..` are normalized lexically with `arko_agents::utils::normalize_path`; symlinks are not resolved. Paths are taken literally, as the built-in file tools open them: unlike `arko_agents::utils::is_path_in_workspace`, `~` and `$VAR` are not expanded. |
| `fn record_read(&self, actor: &ActorId, resource: ResourceId)` | Record this resource's current version as the actor's observation, replacing any older read. New resources begin at version zero. |
| `fn check_write(&self, actor: &ActorId, resource: &ResourceId) -> Vec<FreshnessConflict>` | Return stale overlapping observations whose current version is newer and last writer is another actor. No prior read means no conflict. This method reports conflicts; it does not block a write itself. |
| `fn record_write(&self, actor: &ActorId, resource: ResourceId) -> Vec<ActorId>` | Record a successful mutation, increment affected resource versions, set last writer, and return other actors with affected reads for notification. A file mutation also increments already tracked enclosing directory-tree resources. |
| `fn clear_actor(&self, actor: &ActorId)` | Remove the actor's read observations. Its recorded writes/resource versions remain. |

Overlap means identical resources, or a previously read directory tree containing a written file. Memory/custom resources overlap by exact identity. Direct `record_read`/`record_write` arguments are used as supplied; use consistent normalized paths when constructing resources manually. Freshness blocking through workflow tools requires the corresponding configuration to be enabled; the tracker provides the low-level reporting/recording API.

## MESI file tracking

Module: `arko_workflows::mesi`. Source: [MESI tracking](src/mesi/mod.rs).

This file tracker compares timestamps of reads and subsequent writes by other agent names. It offers configurable conflict decisions and read/write tool classification. It identifies agents by name; use context freshness when task-specific identity and directory/memory/custom resource tracking are needed. In signatures, `Path = std::path::Path` and `PathBuf = std::path::PathBuf`.

| Public type | Fields or variants; defaults |
| --- | --- |
| `FileReadRecord` | `agent: String`, `timestamp: chrono::DateTime<chrono::Utc>`. A recorded file read. |
| `FileWriteRecord` | `agent: String`, `timestamp: chrono::DateTime<chrono::Utc>`. A recorded file write. |
| `FileState` | `readers: Vec<FileReadRecord>`, `writers: Vec<FileWriteRecord>`. `Default` makes both empty. |
| `ConflictAction` | `Allow`, `Block { conflicting_writes: Vec<FileWriteRecord> }`, `NotifyAndAllow { conflicting_writes: Vec<FileWriteRecord> }`. Decision returned by conflict checking. |
| `MesiConfig` | `block_on_conflict: bool` (default `true`), `write_tools: HashSet<String>` (default `{"write_file", "apply_diff"}`), `read_tools: HashSet<String>` (default `{"read_file"}`). Configure through public fields. |

The records, `FileState`, and `MesiConfig` implement `Debug`, `Clone`, `Serialize`, and `Deserialize`; `ConflictAction` implements `Debug` and `Clone`. Rust defaults do not supply missing serialized configuration fields.

All methods below are on `FileTracker`; `Default` uses `new()`.

| Signature | Purpose and behavior |
| --- | --- |
| `fn new() -> Self` | Empty tracker with default MESI configuration. |
| `fn with_config(config: MesiConfig) -> Self` | Empty tracker with supplied tool names and conflict policy. |
| `fn register_read(&self, agent: &str, path: &Path)` | Record a read with the current UTC timestamp. |
| `fn register_write(&self, agent: &str, path: &Path)` | Record a write with the current UTC timestamp. |
| `fn check_write_conflict(&self, agent: &str, path: &PathBuf) -> ConflictAction` | Find other agents' writes strictly newer than this agent's latest read of the exact path. No read, no tracked file, or no newer other-agent write returns `Allow`; conflicts return `Block` or `NotifyAndAllow` according to configuration. |
| `fn is_write_tool(&self, tool_name: &str) -> bool` | Exact membership in configured write tools. |
| `fn is_read_tool(&self, tool_name: &str) -> bool` | Exact membership in configured read tools. |

Paths are stored as supplied. These low-level methods record/check state and return decisions; they do not execute file operations or deliver notifications.

## Completion verification

[Source](src/verification/mod.rs); execution behavior: [engine](src/workflow/engine.rs). All public types and the protocol constant are re-exported by `prelude`.

Register a `CompletionVerifierConfig` with `WorkflowBuilder::completion_verifier` to gate an agent's `complete_task` using a separately registered workflow agent. Configurations run in registration order; a rejection returns feedback for repair or fails the task when its cap is exceeded. Targets use workflow registration names. A configuration always excludes its own verifier, even when explicitly targeted. Empty targets select every other agent; with multiple verifiers, explicit targets avoid unintentionally reviewing other verifiers. Build rejects unknown verifier agents or explicit targets. Use distinct configuration names: repair counts and continuation history are keyed by target task ID and verifier name.

### `CompletionVerifierConfig`, `VerificationKind`, and `RepairContextConfig`

`CompletionVerifierConfig` is cloneable and Serde serializable/deserializable. Its public fields are:

| Field | Type | Purpose/default from constructors |
| --- | --- | --- |
| `name` | `String` | Configuration identifier supplied to constructor; appears in records and feedback. |
| `verifier_agent` | `String` | Registered agent that reviews the completion. |
| `target_agents` | `HashSet<String>` | Selected target registration names; initially empty (all except own verifier). |
| `kind` | `VerificationKind` | `Correctness` or advisory `Efficiency`, chosen by constructor. |
| `instructions` | `String` | Reviewer instructions; defaults described below. |
| `max_minor_rejections` | `u32` | Default `3` for correctness, `2` for efficiency. |
| `max_major_rejections` | `u32` | Default `0` for both kinds. |
| `reuse_conversation` | `bool` | Default `false`; also defaults to false on deserialization when absent. |
| `repair_context` | `Option<RepairContextConfig>` | Default `None`; also defaults to None on deserialization when absent. |

`VerificationKind` has exactly `Correctness` and `Efficiency`; it is `Copy`, cloneable, comparable (`PartialEq`, `Eq`), and Serde serializable/deserializable. `RepairContextConfig` is cloneable, comparable, and Serde serializable/deserializable, with one public field: `max_bytes: usize`, the repair-delta text budget. Neither configuration has a `Default` implementation.

| Exact signature on `CompletionVerifierConfig` | Parameters and result |
| --- | --- |
| `pub fn correctness(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self` | Create a correctness gate. Default instructions: "Check correctness, completeness, and maintainability. Inspect the actual workspace rather than trusting the candidate summary." |
| `pub fn efficiency(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self` | Create an advisory efficiency gate. Default instructions: "Look for avoidable repeated work, poor asymptotic behavior, unnecessary I/O or API calls, and needless resource use. Do not demand speculative optimization." |
| `pub fn target(mut self, agent: impl Into<String>) -> Self` | Add a target registration name; repeated names deduplicate. |
| `pub fn instructions(mut self, instructions: impl Into<String>) -> Self` | Replace reviewer instructions. |
| `pub fn minor_rejection_cap(mut self, cap: u32) -> Self` | Set permitted minor repair attempts. |
| `pub fn major_rejection_cap(mut self, cap: u32) -> Self` | Set permitted major repair attempts for correctness. |
| `pub fn reuse_conversation(mut self, enabled: bool) -> Self` | Continue the prior verifier conversation for subsequent reviews of the same target task when its result is available; otherwise start a new review. |
| `pub fn repair_context(mut self, max_bytes: usize) -> Self` | Enable repair-delta injection, setting `max_bytes.max(1_024)`; does not itself enable conversation reuse. |

Correctness permits exactly the configured number of minor/major rejections, then fails on the next rejection of that severity; the two counters are independent. A major cap of zero fails on the first major rejection. Efficiency converts every major rejection into a minor rejection and stops running the gate when its minor count reaches the cap; a minor cap of zero skips it entirely. It never uses the major cap. Verifier execution failure or an unusable response can still fail an efficiency-reviewed task.

Conversation reuse supplies the previous verdict, new completion summary, and focused re-review instructions. Repair deltas additionally require context freshness/resource tracking; they cover tracked workspace files and new readable memory entries, with truncation or hashes when text is unavailable. They do not enumerate arbitrary directory-tree or custom resources or replace inspecting affected artifacts. Configuration through the fluent method clamps the byte budget; direct field construction/deserialization does not.

### Verdict protocol and records

`pub const VERIFIER_OUTPUT_CONTRACT: &str` provides the plain-text response instructions for verifier prompts. The first non-empty line is one of `ACCEPTED`, `REJECTED_MINOR`, or `REJECTED_MAJOR`. Rejections require non-empty feedback on following lines; acceptance may omit feedback. Leading blank lines, surrounding whitespace on the decision line, and CRLF are accepted. JSON, Markdown wrappers, or feedback appended to the decision line do not match the decision protocol; subsequent feedback is opaque text. One continuation attempt repairs malformed formatting, then the target fails if the response remains unusable.

`VerificationDecision` is cloneable, comparable (`PartialEq`, `Eq`), and Serde serializable/deserializable:

| Rust variant | Serde value | Meaning |
| --- | --- | --- |
| `Approved` | `"approved"` | Protocol `ACCEPTED`; allow this gate to pass. |
| `RejectedMinor` | `"rejected_minor"` | Ask for bounded corrections. |
| `RejectedMajor` | `"rejected_major"` | Major correctness issue; efficiency converts this to `RejectedMinor`. |

`VerificationRecord` is cloneable and Serde serializable/deserializable. Public fields:

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

Each record identifies the reviewed target task/agent, named configuration, reviewing agent/run, timestamp, effective decision (after efficiency conversion), and trimmed feedback. Records are available through `SharedContext::verification_records` and checkpoint state; transient repair counters/baselines are cleared when the target successfully completes.

## Tool result distillation

[Source](src/distillation/mod.rs); registration/order: [engine](src/workflow/engine.rs). Both public types are cloneable, Serde serializable/deserializable, and re-exported by `prelude`.

Register `DistillationConfig` through `WorkflowBuilder::distiller` to replace sufficiently large matching tool results with an LLM-produced distilled result. Distillers run after global and per-agent result interceptors, in registration order. Each sees the result from preceding interceptors; all configured filters must match. Distillation uses the shared LLM client, the configured model/reasoning/output-token limit, a fresh conversation with caching disabled, and the agent's initial task prompt, tool name, parameters, and current tool output as context.

### `DistillationConfig`

| Public field | Type | Purpose/default from `new` |
| --- | --- | --- |
| `name` | `String` | Configuration name for records, errors, and cost descriptions. |
| `model` | `arko_ai::Model` | Distillation model supplied by caller. |
| `target_agents` | `HashSet<String>` | Matching workflow registration names; empty matches all. |
| `target_tools` | `HashSet<String>` | Matching tool names; empty matches all. |
| `min_chars` | `usize` | Minimum input Unicode character count, inclusive; default `8_000`. |
| `max_output_tokens` | `usize` | Generation limit including reasoning tokens; default `16_000`, or the model's output limit when smaller (`Model::max_output_tokens`, when known); must be positive. |
| `instructions` | `String` | Content-preservation instructions described below. |
| `reasoning` | `Option<arko_ai::ReasoningConfig>` | Default `None`; omitted when serializing None and defaults to None when absent on deserialization. |

| Exact signature on `DistillationConfig` | Parameters and result |
| --- | --- |
| `pub fn new(name: impl Into<String>, model: arko_ai::Model) -> Self` | Create the configuration with the defaults above. Default instructions: "Preserve every fact relevant to the task, exact errors, paths, identifiers, constraints, and actionable details. Remove repetition and irrelevant bulk. Never invent facts." |
| `pub fn target_agent(mut self, agent: impl Into<String>) -> Self` | Add an agent-name filter; names deduplicate. |
| `pub fn target_tool(mut self, tool: impl Into<String>) -> Self` | Add a tool-name filter; names deduplicate. |
| `pub fn min_chars(mut self, chars: usize) -> Self` | Set inclusive size threshold; `0` also admits empty input. |
| `pub fn max_output_tokens(mut self, tokens: usize) -> Self` | Set token limit; zero is rejected during workflow build. |
| `pub fn instructions(mut self, instructions: impl Into<String>) -> Self` | Replace distillation instructions. |
| `pub fn reasoning(mut self, reasoning: arko_ai::ReasoningConfig) -> Self` | Set reasoning; workflow build rejects a provider that differs from the model's provider. |

There is no `Default` implementation. Build validates token limit and reasoning/provider compatibility; it does not validate target agent/tool names. Only a finished summary replaces the result: a response with no refusal, a `StopReason::EndTurn` or `StopReason::StopSequence` stop reason, and non-empty text. Otherwise the raw result is kept:

- A refusal (`GenerationResponse::refusal`, from any source and even when text accompanies it, or an `arko_ai::Error::Refusal` such as an OpenRouter moderation 403) or a truncated stop reason (`StopReason::is_truncated`: `MaxTokens`, when the summary hit `max_output_tokens`, or `ContextWindowExceeded`) would most likely repeat, so the distiller logs a warning and returns `InterceptedResult::Unchanged`: the interception failure policy does not retry it.
- Model errors, empty text, and any other stop reason produce interceptor errors, which the workflow's interception failure policy may retry before keeping the raw result. That includes `StopReason::Other("unknown")`, a response that reported no stop reason (OpenRouter `finish_reason: null`, an Anthropic `stop_reason: null`), whose completeness can't be told.

A finished, non-empty summary replaces the result even if it is longer than its input; `max_output_tokens` is a generation cap, not a guaranteed compression ratio. Reasoning (thinking) tokens count against `max_output_tokens` with OpenAI API keys and on Anthropic (a manual thinking budget must stay below it unless interleaved thinking is on), so a limit that is too small for the model's reasoning yields truncated summaries, which are charged but not used. ChatGPT OAuth requests send no output limit.

Distillation charges the billed amount `response.usage.billed()` (zero for subscription calls such as ChatGPT OAuth) as a **manual workflow cost**, with description `distillation: <name>` and agent/task/tool metadata. It therefore affects the workflow budget and does not add to the agent's own LLM-spend counter. Warning/exceeded hooks run when indicated; normal managed execution enforces the exceeded workflow budget. Every response the model returns is charged and recorded, including refused, cut-off, and empty ones, since they are billed too (each retry of an interceptor error is another billed call); the distillation record is appended after the budget hooks succeed.

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

Public fields identify the target task/agent/tool and configuration, UTC timestamp, input/output Unicode character counts, and reported dollar cost. `output_chars` counts the text the model generated, joined as a summary is (text parts separated by newlines): the message text, plus the `Refusal::partial_output` that a safety-classifier or content-filter refusal moves out of the message; a refusal's explanation is not counted. `rejection` is `None` when the distilled text replaced the result; otherwise it says why the text was not used (the refusal, the stop reason, or no text). It is omitted when serializing `None`, and records saved before it existed load as `None`. Records are available through `SharedContext::distillation_records` and checkpoint state.

## Budget and cost accounting

[Source](src/budget/mod.rs). All five public types are re-exported by `prelude`. Amounts, limits, warnings, and returned balances use dollars. The running workflow/per-agent counters have micro-dollar precision; ledger entries retain the original `f64` amounts, so a ledger sum can differ slightly from the running total.

### `BudgetManager`

Thread-safe accumulated workflow cost, per-agent LLM spend/limits, and a detailed ledger. Fields are private; there is no `Default` implementation. The manager records accounting and returns threshold flags; it does not itself cancel agents or fire workflow hooks. Managed workflow execution supplies that behavior.

| Exact signature | Parameters, return, and behavior |
| --- | --- |
| `pub fn new(limit: Option<f64>, warning: Option<f64>) -> Self` | Start at zero with an empty ledger and no agent limits. `None` disables its respective workflow threshold. |
| `pub fn from_ledger(limit: Option<f64>, warning: Option<f64>, ledger: CostLedger) -> Self` | Restore accounting from entries; rebuild per-agent spend from `CostSource::Agent`. Already-reached warning is marked fired. Agent limits start empty and must be set again. |
| `pub fn add_agent_cost(&self, agent_name: &str, usage: &arko_ai::UsageInfo) -> AddCostResult` | Append an agent entry whose amount is `usage.billed()` (zero for subscription calls such as ChatGPT OAuth, whose estimate stays in the stored usage), with cloned usage details, current time, and no metadata; update workflow and agent spend. |
| `pub fn set_agent_limit(&self, agent_name: impl Into<String>, limit: f64)` | Set/replace that agent's dollar limit. |
| `pub fn agent_limit(&self, agent_name: &str) -> Option<f64>` | Return configured per-agent limit, or None. |
| `pub fn agent_spent(&self, agent_name: &str) -> f64` | Return recorded agent LLM cost, or zero for an unseen name. |
| `pub fn is_agent_exceeded(&self, agent_name: &str) -> bool` | True only when agent spend is strictly greater than its configured limit. |
| `pub fn add_manual_cost(&self, amount: f64, description: &str, metadata: Option<String>) -> AddCostResult` | Append a manual cost with current timestamp and arbitrary optional accounting metadata; update only workflow spend. |
| `pub fn total_spent(&self) -> f64` | Return running workflow spend. |
| `pub fn is_exceeded(&self) -> bool` | True only when workflow spend is strictly greater than its limit; false without a limit. |
| `pub fn check_warning(&self) -> bool` | Return true once, when spend is greater than or equal to the warning threshold. Calling this consumes the one-time notification. |
| `pub fn remaining(&self) -> Option<f64>` | Return `limit - spent`, clamped at zero; None without a limit. |
| `pub fn limit(&self) -> Option<f64>` | Return configured workflow limit. |
| `pub fn snapshot_ledger(&self) -> CostLedger` | Clone the current ledger for analysis/persistence. |

Recording methods retain the cost even if it exceeds the budget and consume a newly reached warning via their result. Supply finite, non-negative dollar values: direct manager construction/mutation does not validate these inputs. `WorkflowBuilder` validates configured limits/warnings. Use `Workflow::add_manual_cost` for manual costs that also trigger managed workflow hooks and return `WorkflowError::BudgetExceeded` when appropriate.

### Cost data types

`CostSource` is cloneable and Serde serializable/deserializable:

```rust
pub enum CostSource {
    Agent { agent_name: String, usage: Box<arko_ai::UsageInfo> },
    Manual { description: String },
}
```

`Agent` retains the registered agent name and full usage breakdown, including the per-token estimate of subscription calls whose billed `amount` is zero. `Manual` retains a free-form description, including charges from distillation.

`CostEntry` is cloneable and Serde serializable/deserializable, with public fields:

```rust
pub amount: f64;
pub source: CostSource;
pub timestamp: chrono::DateTime<chrono::Utc>;
pub metadata: Option<String>;
```

`amount` is the original dollar charge; `timestamp` identifies when it was recorded; optional metadata can contain service/API/accounting context without a prescribed schema.

`CostLedger` is cloneable, Serde serializable/deserializable, and `Default` (empty). It has one public field, `entries: Vec<CostEntry>`, in insertion order. `pub fn total(&self) -> f64` sums original entry amounts in dollars.

`AddCostResult` is cloneable, with public fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `new_total` | `f64` | Running workflow dollar total after recording the addition. |
| `warning_triggered` | `bool` | This recording observed the warning threshold for the first time. |
| `exceeded` | `bool` | Workflow limit is currently exceeded; this is not limited to the first crossing. |

## Durable steps

Source: [src/durable/mod.rs](src/durable/mod.rs). Import from `arko_workflows::durable`; the prelude re-exports its public types except `DurableStore`. Durable steps replay persisted outputs for explicitly identified operations and suspend uncertain outcomes for reconciliation. Configure workflow persistence to use the `Workflow` convenience methods. `DurableStore` is also directly constructible.

In the signatures below, `Uuid = uuid::Uuid`, `PathBuf = std::path::PathBuf`, `Serialize = serde::Serialize`, `DeserializeOwned = serde::de::DeserializeOwned`, and `Future = std::future::Future`.

### Step specifications, execution context, and reconciliation

| Type | Public fields or variants | Purpose and defaults |
|---|---|---|
| `DurableStepSpec` | `key: String`, `instance: Option<String>`, `version: String`, `retry_policy: DurableRetryPolicy` | Stable operation definition. `new(key)` sets `instance = None`, `version = "v1"`, and `retry_policy = Never`. Key plus instance identifies the step; version and serialized-input hash must match its recorded definition. |
| `DurableRetryPolicy` | `Never`; `WhenDefinitelyNotCompleted { max_attempts: u32 }`; `RetrySafe { max_attempts: u32 }` | Default is `Never`. The second permits retry after a resolver confirms noncompletion; the third allows retry of an uncertain operation without reconciliation. Limits count the first attempt and are clamped to at least one. |
| `DurableStepStatus` | `Started`, `Completed`, `InDoubt`, `PermanentlyFailed` | Journal lifecycle. `Started` without completion is treated as uncertain on replay. |
| `DurableStepContext` | `workflow_run_id: Uuid`, `step_id: Uuid`, `operation_id: String`, `attempt: u32` | Closure input. IDs remain stable across retries; `attempt` starts at 1. Use `operation_id` for a provider's idempotency or receipt lookup key. |
| `InDoubtContext` | `workflow_run_id: Uuid`, `step_id: Uuid`, `operation_id: String`, `attempt: u32`, `reason: Option<String>` | Resolver/operator inspection data for an uncertain attempt. |
| `InDoubtResolution<O>` | `Completed(O)`; `DefinitelyNotCompleted`; `StillInDoubt { reason: String }`; `PermanentlyFailed { reason: String }` | Respectively persist recovered output, permit a policy-approved retry, keep suspended with a reason, or persist terminal failure. |

`DurableStepSpec`, `DurableRetryPolicy`, and `DurableStepStatus` support Serde serialization/deserialization and cloning. The retry policy and status support equality comparison. Both context types are cloneable; `DurableStepStatus` is also `Copy`.

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

`instance`, `version`, and `retry_policy` set the matching fields and return the spec. `display_key` returns `key` or `key[instance]`. `operation_id()` borrows the operation ID. Implement `ResolveInDoubt` to check an external system using the supplied attempt context and original input; resolver errors propagate.

Validation occurs when executing or applying an operator resolution: keys must have nonwhitespace content; keys and instances cannot contain newline, carriage return, or NUL; versions must have nonwhitespace content. `None` and `Some("")` instances share one identity. Reusing an identity with different input or version yields `DurableDefinitionMismatch`; use a distinct key/instance for a separate operation.

### DurableStore

`DurableStore` has no public fields. It maintains one run's append-only JSONL journal at `<run_dir>/durable/journal.jsonl`.

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

| Method | Parameters and result/behavior |
|---|---|
| `new` | Associates the supplied run UUID and directory; construction does not create files or validate a preexisting journal's run identity. |
| `journal_path`, `workflow_run_id` | Return the configured journal path and run ID. |
| `execute` | Hashes serializable `input`, validates `spec`, and either returns a deserialized completed output or runs `execute(context)` once and journals the result. A closure error or output serialization failure leaves the outcome uncertain and normally returns `DurableInDoubt`. |
| `execute_with_resolver` | Same execution/replay contract, with `resolver` consulted when an existing uncertain attempt cannot already retry. `RetrySafe` and operator-authorized retries proceed before resolver consultation. |
| `list_in_doubt` | Returns latest `Started` and `InDoubt` attempts, including currently executing steps; order is unspecified. |
| `operator_complete` | Records supplied `output` as completed for the matching unresolved `spec`/`input`. The output must deserialize as the future caller's output type. |
| `operator_permanently_fail` | Records terminal failure; future execution returns `DurablePermanentlyFailed`. |
| `operator_authorize_retry` | Records permission for the next execution call to retry, including with `Never`; any configured attempt limit still applies. Does not itself execute the closure. |

All operator methods require an existing unresolved attempt with matching version/input. They append `actor` and `reason` audit information; they do not overwrite prior journal records. Resolve steps after active execution has stopped: operator methods do not acquire execution leases.

Retries occur on subsequent calls, with at most one closure execution per call. Exhausted configured attempts return `DurableAttemptsExhausted`. Persistence/serialization failures can also propagate. A completed result replays without rerunning its closure; external side effects still require appropriate idempotency/reconciliation semantics. A duplicate identity executing on the same store returns `DurableInDoubt`. Locks and active-step protection belong to the store instance; there is no cross-process lock. Journal reads tolerate an invalid final JSON record but reject an invalid earlier record.

## Persistence and checkpoint state

Sources: [src/persistence/mod.rs](src/persistence/mod.rs), [src/persistence/state.rs](src/persistence/state.rs). Import configuration/manager from `arko_workflows::persistence`, state types from `arko_workflows::persistence::state`. The prelude re-exports `PersistenceConfig`.

### PersistenceConfig and PersistenceManager

`PersistenceConfig` is cloneable and has public fields `checkpoint_dir: std::path::PathBuf` (run storage directory) and `auto_checkpoint: bool` (whether automatic checkpoints are enabled). It has no `Default` implementation; `new` sets `auto_checkpoint = true`. `PersistenceManager` is cloneable with private fields; clones share the durable store and write lock.

```rust
impl PersistenceConfig {
    pub fn new(checkpoint_dir: impl Into<std::path::PathBuf>) -> Self;
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
    ) -> Result<(), WorkflowError>;
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
| `PersistenceConfig::new` | Uses the given run directory. |
| `PersistenceConfig::user_data` | Returns a fresh directory `<user-data>/arko/workflows/<workflow_name>/<new-UUID>`; does not create it. User-data base: Windows `LOCALAPPDATA`; macOS `$HOME/Library/Application Support`; otherwise `XDG_DATA_HOME` or `$HOME/.local/share`. Errors if no base can be determined. |
| `PersistenceManager::resolve_run_id` | Creates the configured directory; reads its `workflow-run-id` UUID or creates/syncs a new one if absent. Returns the reusable identity; malformed existing UUIDs error. |
| `PersistenceManager::new` | Associates a config and supplied run identity. Use the matching directory identity when reopening a run. |
| `durable_store` | Returns a shared handle to that run's durable-step store. |
| `save_checkpoint` | Snapshots `ctx` into a new `checkpoint-<UUID>.json`; writes/syncs a temporary file, renames it, updates `latest-checkpoint` with the filename, and returns the checkpoint path. |
| `append_event` | Appends `event` to `events.jsonl`, annotated with current UTC timestamp, workflow run UUID, supplied `agent_name`, and `agent_run_id`. |
| `save_agent_task` | Writes/syncs/renames the serialized `task` to `agents/<task.id>.json`, replacing that task's prior snapshot; returns its path. |
| `auto_checkpoint` | Saves `ctx` only if `config.auto_checkpoint` is true; otherwise succeeds without writing. |
| `load_checkpoint` | Reads/deserializes the exact supplied JSON file into `WorkflowState`; does not follow `latest-checkpoint` or restore a live workflow. Use `WorkflowBuilder::resume_from` to restore. |

These operations return filesystem/serialization failures as `WorkflowError`. Persistence stores snapshots, task/event history, and the separate durable journal; checkpoints do not persist live running futures or ballots.

On `resume_from`, the builder restores memory tiers/history, pending messages, the cost ledger, finalized agent results, and verification/distillation records. Saved memory replaces any newly supplied `MemoryConfig`. Freshness state is restored only when tracking is enabled again. Supply agents, client, tools, hooks, policies, budget limits, and metadata through the new builder; they are not reconstructed from the checkpoint. Pause state, active channels, ballots, and transient verification counters/baselines start fresh. The builder does not automatically continue agents or replay workflow control flow. Accounting is rebuilt from `cost_ledger`; legacy `budget_spent` and `completed_agents` do not substitute for a missing ledger or full results.

### Checkpoint payloads

`WorkflowState` supports Serde serialization/deserialization and exposes all fields below. It has no `Default` implementation. “Missing JSON default” describes backward-compatible deserialization only.

| Field | Type | Purpose; missing JSON default |
|---|---|---|
| `workflow_run_id` | `uuid::Uuid` | Durable run identity; nil UUID if absent. |
| `workflow_name` | `String` | Workflow name; required. |
| `workflow_metadata` | `crate::workflow::config::WorkflowMetadata` | Workflow description/platform/dependency metadata; its default if absent. |
| `checkpoint_id` | `uuid::Uuid` | Unique snapshot ID; required. |
| `timestamp` | `chrono::DateTime<chrono::Utc>` | Snapshot time; required. |
| `memory_snapshot` | `Vec<crate::memory::MemoryTier>` | Memory tiers and entries; required. |
| `message_queues` | `std::collections::HashMap<String, Vec<crate::messaging::AgentMessage>>` | Pending messages per agent; required. |
| `budget_spent` | `f64` | Total recorded spend; required. |
| `cost_ledger` | `crate::budget::CostLedger` | Complete accounting history; empty/default ledger if absent. |
| `freshness_snapshot` | `Option<crate::freshness::FreshnessSnapshot>` | Resource version/read tracking state; `None` if absent. |
| `agent_results` | `Vec<crate::workflow::result::AgentResult>` | Full finalized results, including failures and input suspensions; empty if absent. |
| `verification_records` | `Vec<crate::verification::VerificationRecord>` | Completion-verification history; empty if absent. |
| `distillation_records` | `Vec<crate::distillation::DistillationRecord>` | Distillation history; empty if absent. |
| `completed_agents` | `Vec<SerializableAgentResult>` | Legacy/lightweight agent result summaries; empty if absent. |

Checkpoints written by older versions may contain an `oauth_exempt_agents` field. It is ignored when they are loaded: budget accounting now records `UsageInfo::billed()`, which is zero for subscription calls, so no exemption list is kept (the removed `BudgetManager::mark_oauth_exempt`, `is_oauth_exempt` and `oauth_exempt_snapshot` have no replacement). Ledger entries already in such a checkpoint keep their recorded amounts.

`SerializableAgentResult` has public `agent_name: String` and `summary: String` fields. It supports Serde, `Clone`, and `Default` (both strings empty).

## Structured voting

Source: [src/voting/mod.rs](src/voting/mod.rs). Import from `arko_workflows::voting`. The prelude re-exports `VoteOption`, `TieBreak`, `BallotConfig`, `CastVote`, and `BallotResult`. Enable workflow integration with `WorkflowBuilder::enable_voting`, begin a ballot, supply its UUID/options to participating agents, and finish after agents vote.

| Type | Public fields or variants | Purpose and constraints |
|---|---|---|
| `VoteOption` | `id: String`, `description: String` | Exact machine-readable option ID and human/agent description. IDs must be nonblank and unique within a ballot. |
| `TieBreak` | `RejectTie`, `Lexicographic` | Reject equal top totals or choose the lexicographically first tied option ID. |
| `BallotConfig` | `question: String`, `options: Vec<VoteOption>`, `eligible_voters: std::collections::HashSet<String>`, `quorum: usize`, `consensus_threshold: f64`, `weights: std::collections::HashMap<String, f64>`, `tie_break: TieBreak` | At least two options, positive quorum, finite threshold in `[0, 1]`. Empty eligibility set allows all agents. Absent voter weights count as `1.0`; weights for actual voters must be finite and positive. No `Default` implementation. |
| `CastVote` | `voter: String`, `task_id: uuid::Uuid`, `option: String`, `reasoning: String`, `timestamp: chrono::DateTime<chrono::Utc>` | A recorded agent vote. Identity/task/time are assigned by the tool; option and reasoning come from its arguments. |
| `BallotResult` | `ballot_id: uuid::Uuid`, `winner: String`, `consensus: f64`, `votes: Vec<CastVote>` | Winning option, winning weight divided by total cast weight, and recorded votes (unspecified order). |

These types support Serde serialization/deserialization and `Clone`; `TieBreak` is also `Copy` and equality-comparable. `BallotStore` has private fields and `Default` creates an empty in-memory store.

```rust
pub const CAST_VOTE_TOOL_NAME: &str = "cast_vote";
impl BallotStore {
    pub fn begin(&self, config: BallotConfig) -> Result<uuid::Uuid, String>;
    pub fn finish(&self, ballot_id: uuid::Uuid) -> Result<BallotResult, String>;
}
```

`begin(config)` validates options/quorum/threshold, stores an empty ballot, and returns its UUID. `finish(ballot_id)` requires that ballot to exist, checks the number of distinct voters against quorum, validates weights for cast votes, totals weighted votes, applies tie-breaking, and checks consensus. Errors explain unknown ballots, insufficient quorum, invalid weights, rejected ties, or insufficient consensus. Finishing computes a result without closing/removing the ballot; additional eligible agents can still vote. State is held in memory and is not checkpointed. Workflow convenience methods map store errors to `WorkflowError::Deliberation`.

### Agent tool: cast_vote

`cast_vote(ballot_id: string, option: string, reasoning: string) -> string`. All three arguments are required; no defaults, batching, or path parameters. `ballot_id` must parse as a UUID supplied by the workflow; `option` must exactly match a configured ID; `reasoning` carries the independent rationale. The tool binds `voter` to the running agent name and `task_id` to its task, and timestamps the vote in UTC. Only eligible agents may vote and each agent name may vote once per ballot. Success returns `Vote recorded for option '<option>'.`; missing/non-string arguments, malformed UUIDs, unknown ballots/options, ineligible voters, and duplicate votes error. The tool implementation is crate-private and the direct store casting method is private; workflow voting registration exposes the tool to agents.

## Python bindings

Source: [python/mod.rs](src/python/mod.rs). Cargo feature: `python`. The module exports three Python classes: `PyWorkflowBuilder`, `PyWorkflow`, `PyAgentResult`. These same names are public Rust wrapper types under `arko_workflows::python`; their Python methods are not public inherent Rust methods. `PyAgentResult` implements `From<arko_workflows::AgentResult>`.

The table uses Python-facing signatures: the implicit `self` and PyO3 interpreter arguments are omitted, and awaitable results are written as `async def`. Builder setters mutate in place and return `None`; they cannot be chained like the Rust builder. Numeric limits use the underlying Rust `f64`/`usize` types.

### PyWorkflowBuilder

| Python signature | Parameters and behavior |
| --- | --- |
| `PyWorkflowBuilder()` | Empty builder. Same required name/workspace/agents as Rust; it supplies its own LLM client at build time. |
| `name(name: str) -> None` | Set workflow name. |
| `workspace(path: str) -> None` | Set existing workspace directory. |
| `agent(name: str, config_path: str) -> None` | Queue a named agent configuration file to load during `build`. |
| `budget(limit: float) -> None` | Set total dollar budget; validated at build. |
| `persistence(checkpoint_dir: str, auto_checkpoint: bool = False) -> None` | Enable persistence with explicit directory and auto-checkpoint setting. |
| `resume_from(checkpoint: str) -> None` | Restore checkpoint JSON file; requires persistence config. |
| `global_memory_tier(name: str, max_bytes: int \| None = None) -> None` | Add globally readable/writable append-only memory tier; optional byte cap. |
| `build() -> PyWorkflow` | Load agent files, build LLM client, and construct workflow. Load/build failures raise `ValueError`. Builder is borrowed and can be reused. |

During `build`, the Python builder calls `LLMClient::from_env()`, which configures every provider found in the process environment: OpenAI (ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID`, with optional `OPENAI_OAUTH_REFRESH_TOKEN`, else `OPENAI_API_KEY`), Anthropic (`ANTHROPIC_API_KEY`), Google (`GOOGLE_API_KEY`, else `GEMINI_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`); blank values are ignored. A Codex `auth.json` is not read, and refreshed OAuth tokens are kept only in memory. No Python method accepts a custom client, custom tools, hooks, verifiers, distillers, or service settings. These capabilities require the Rust API or an extension of the bindings.

### PyWorkflow

No Python constructor is exposed; obtain this from the builder.

| Python signature | Parameters, result, and behavior |
| --- | --- |
| `async def run_agent(agent_name: str, prompt: str) -> PyAgentResult` | Run a registered agent with default managed behavior. |
| `async def run_agents_parallel(agents: list[tuple[str, str]]) -> list[PyAgentResult]` | Concurrent `(name, prompt)` pairs, results in input order when all succeed. |
| `add_manual_cost(amount: float, description: str, metadata: str \| None = None) -> float` | Record dollars and return new running workflow total; a budget error leaves charge recorded. |
| `cost_ledger_json() -> str` | Serialize current `CostLedger` as JSON. |
| `async def checkpoint() -> None` | Save snapshot; no-op without configured persistence. |
| `read_memory(agent_name: str, tier: str, max_chars: int \| None = None) -> str` | Access-checked plain-text tier read or trailing character window. Does not use the agent tool's versioned JSON/delivery baseline. |
| `write_memory(agent_name: str, tier: str, content: str) -> None` | Access-checked memory write using tier mode. Uses store API directly, without workflow tool notifications/hooks; no task ID attached. |

Workflow execution, accounting, checkpoint, and memory errors raise `RuntimeError`. Python does not expose continuation, raw handles, event subscription, pause/resume, finish, aggregate `WorkflowResult`, durable-step operations, or ballot operations in this implementation.

### PyAgentResult

Read-only Python properties: `agent_name: str`, `summary: str`, `total_cost: float` (the billed amount, zero for ChatGPT OAuth calls), `status: str`, `agent_run_id: str`. `status` is the Rust debug variant name (`Completed`, `Failed`, `Interrupted`, `Cancelled`, or `WaitingForInput`); run ID is a UUID string. The Rust wrapper's five corresponding fields are public (`String`, `String`, `f64`, `String`, `String`). No full task, token totals, or detailed error field is exposed. No Python constructor is exposed.

## Workflow errors

Source: [error.rs](src/error.rs). `WorkflowError` is root- and prelude-re-exported and implements `std::error::Error`/`Display` through `thiserror`. Match variants for programmatic handling.

| Variant and payload | Meaning |
| --- | --- |
| `Agent(arko_agents::ArkoAgentsError)` | Underlying agent setup/runtime/task error; automatic `From`. |
| `Llm(arko_ai::Error)` | Underlying model error; automatic `From`. |
| `BuilderMissing { field: String }` | Required setting is absent. |
| `BuilderInvalid(String)` | Invalid configuration, checkpoint identity, or disabled feature requested through an API using this variant. |
| `AgentNotFound(String)` | Requested agent registration does not exist. |
| `AgentInterrupted { agent: String, reason: String }` | Interrupted managed invocation. |
| `UserInputRequired { agent: String, task_id: Uuid, questions: String }` | Input-suspended run needs an external answer. |
| `MemoryTierNotFound(String)` | Unknown memory tier. |
| `MemoryAccessDenied { agent: String, tier: String, action: String }` | Reader/writer policy rejects access. |
| `MemoryLimitExceeded { tier: String, attempted: usize, limit: usize }` | Write would exceed tier byte cap. |
| `BudgetExceeded { spent: f64, limit: f64 }` | Overall spending strictly exceeds the configured dollar limit. |
| `AgentBudgetExceeded { agent: String, spent: f64, limit: f64 }` | Agent's cumulative LLM spending strictly exceeds its limit. |
| `InvalidCost(f64)` | Manual workflow charge is negative or non-finite. |
| `AbortedByHook { hook_point: String, reason: String }` | Propagated lifecycle hook abort. |
| `MesiConflict { agent: String, path: String, modifier: String }` | Declared legacy file-conflict error payload. Generalized tool freshness rejection normally returns tool feedback instead. |
| `Persistence(String)` | Persistent identity, file, or state validation failure represented as text. |
| `DurableUnavailable` | Durable operation requested without persistence. |
| `DurableDefinitionMismatch { step: String, details: String }` | Persisted step definition/input differs from the requested one. |
| `DurableInDoubt { step: String, operation_id: String }` | Prior side-effect outcome is unresolved. |
| `DurablePermanentlyFailed { step: String, reason: String }` | Persisted permanent failure. |
| `DurableAttemptsExhausted { step: String }` | Attempt limit reached. |
| `Cancelled` | Declared workflow cancellation error. Managed task cancellations can also appear as an agent error and a `Cancelled` result status. |
| `Io(std::io::Error)` | I/O failure; automatic `From`. |
| `Json(serde_json::Error)` | JSON encoding/decoding failure; automatic `From`. |
| `Channel(String)` | Join/channel failures, user-input handler failure, or unavailable/exhausted workflow services. |
| `Hook(String)` | General hook error payload. |
| `Tool(arko_agents::ToolError)` | Tool-layer error; automatic `From`. |
| `Deliberation(String)` | Ballot configuration/tally failure. |

## Minimal composition and examples

The following Rust function accepts already configured agents and an LLM client. It demonstrates the required builder inputs, a shared memory tier, a sequential handoff, and explicit completion.

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

| Existing example | Demonstrated features |
| --- | --- |
| [basic_workflow.rs](examples/basic_workflow.rs) | Sequential architect/coder handoff, environment-configured client, shared decisions, cost limit/warning, completion hook. Paths are hardcoded in the implementation. |
| [DeepCoder guide](examples/deepcoder/README.md), [deepcoder.rs](examples/deepcoder.rs) | Agent-driven delegation, correctness/efficiency/final gates, conversation reuse with repair deltas, per-role memory policies, a custom shell tool, checkpoint/resume, and workflow metadata. Application-specific CLI and shell policies live in the example. |
