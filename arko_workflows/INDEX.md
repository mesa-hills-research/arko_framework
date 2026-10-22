# Arko Workflows API index

Reference for `arko_workflows` 0.1.0, covering the public Rust API, the optional Python API, the agent tools a workflow injects, configuration defaults, result contracts, and execution behavior. Workflow control flow is ordinary Rust or Python code. The crate supplies the orchestration primitives.

Signatures are declarations with their receivers, not complete implementations. Standard imports are abbreviated, with `Arc`/`Weak`/`OnceLock` from `std::sync`, `HashMap`/`HashSet` from `std::collections`, `Path`/`PathBuf` from `std::path`, `Uuid` from `uuid`, `Value` from `serde_json`, and `Mutex`/`RwLock` from `parking_lot` unless qualified. `AgentConfig`, `Task`, `Tool`, `ToolError`, `ToolResultInterceptor`, and `InterceptionFailurePolicy` come from `arko_agents`. `LLMClient`, `Model`, and `UsageInfo` come from `arko_ai`. Builder setters consume and return `Self` unless shown otherwise. Derived standard traits are noted only where they matter to callers.

- [Imports and feature map](#imports-and-feature-map)
- [Workflow construction](#workflow-construction)
- [Workflow execution and lifecycle](#workflow-execution-and-lifecycle)
- [Result and event types](#result-and-event-types)
- [Hooks and human input](#hooks-and-human-input)
- [Workflow services and custom tools](#workflow-services-and-custom-tools)
- [Background subagents](#background-subagents)
- [Shared context](#shared-context)
- [Memory](#memory)
- [Messaging](#messaging)
- [Operator notes](#operator-notes)
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

Defined in [Cargo.toml](Cargo.toml), [crate exports](src/lib.rs).

No Cargo features are enabled by default. The `python` feature enables the PyO3 bindings and their Tokio bridge. See [Python bindings](#python-bindings).

The root re-exports `Workflow`, `WorkflowBuilder`, `AgentHandle`, `RunOptions`, `WorkflowError`, `WorkflowCancelHandle`, `CancelCause`, `Cancellation`, `SupportedPlatform`, `WorkflowDependency`, `WorkflowMetadata`, `AgentResult`, `AgentRunStatus`, `WorkflowEvent`, `WorkflowResult`, and, from `workflow::stop`, `StopKind`, `StopRequest`, `WrapUp`, `StopRecord`, `StoppedRun`, `Restart`, `stop_on_signals`.

`use arko_workflows::prelude::*;` re-exports the groups below. Other public types are reached through their module paths.

| Module | Prelude exports |
| --- | --- |
| `workflow`, `workflow::cancel`, `workflow::config`, `workflow::result`, `workflow::stop`, `error` | All root re-exports above **except `AgentHandle`** |
| `budget` | `AddCostResult`, `BudgetExceededPolicy`, `BudgetManager`, `CostEntry`, `CostLedger`, `CostSource`, `PlanBudget`, `PlanLimits`, `PlanUsageSpent`, `PlanWindow` |
| `distillation` | `DistillationConfig`, `DistillationRecord` |
| `durable` | `DurableRetryPolicy`, `DurableStepContext`, `DurableStepSpec`, `DurableStepStatus`, `InDoubtContext`, `InDoubtResolution`, `ResolveInDoubt` |
| `freshness` | `ActorId`, `ContextFreshnessConfig`, `FreshnessConflict`, `ResourceId`, `ResourceTracker`, `ToolResourceEffect` |
| `hooks` | `HookCallback`, `HookPoint`, `HookResult` |
| `memory` | `AccessScope`, `MemoryConfig`, `MemoryEntry`, `MemoryTier`, `MemoryWrite`, `WriteMode` |
| `messaging`, `messaging::registry`, `messaging::wait` | `AgentMessage`, `MessageAttachment`, `MessagingPolicy`, `AgentStatus`, `WaitReason`, `WaitForMessageTimeout` |
| `notes` | `Note`, `NoteAudience`, `NoteImage`, `NoteKind`, `NoteSource`, `NotesHandle`, `PostedNote` |
| `persistence` | `EventLog`, `PersistenceConfig` |
| `policy` | `AgentPause`, `AgentResume`, `ApprovalPolicy`, `ErrorRecoveryPolicy`, `ToolApprovalHandler`, `ToolApprovalOutcome`, `ToolApprovalRequest` |
| `services` | `WorkflowServices`, `WorkflowServicesConfig` |
| `subagents` | `GitWorktreeProvider`, `IsolatedWorkspace`, `MailItem`, `MailKind`, `MailSender`, `StartRequest`, `SubagentEvent`, `SubagentEventKind`, `SubagentHook`, `SubagentRecord`, `SubagentState`, `SubagentsConfig`, `WorkspaceProvider`, `WorkspaceRequest` |
| `user_input` | `UserInputHandler`, `UserInputRequest` |
| `verification` | `CompletionVerifierConfig`, `RepairContextConfig`, `VERIFIER_OUTPUT_CONTRACT`, `VerificationDecision`, `VerificationKind`, `VerificationRecord` |
| `voting` | `BallotConfig`, `BallotResult`, `CastVote`, `TieBreak`, `VoteOption` |

The other public modules are `context`, `hooks::context`, `memory::tools`, `messaging::tools`, `messaging::wait`, `notes::control`, `persistence::state`, and `python` (with the `python` feature). `workflow::engine` is public but exports no items. Execution goes through `Workflow` and `AgentHandle`.

| Need | Entry point |
| --- | --- |
| Build and configure a workflow | `Workflow::builder()` / `WorkflowBuilder` |
| Sequential, parallel, or background agent execution | `run_agent`, `continue_agent`, `run_agents_parallel`, `spawn_agent` |
| Monitor progress or control the lifecycle | `subscribe_events`, `subscribe_events_lossless`, hooks, `pause`, `resume`, `finish` |
| Stop a run from any task, or from a hook | `cancel_handle` / `WorkflowCancelHandle`, `HookResult::Abort` |
| Stop gracefully (Ctrl-C, `SIGTERM`, a redeploy) and restart what the stop ended | `stop`, `StopRequest`, `stop_on_signals`, `stopped_runs`, `restart` ([Graceful stop](#graceful-stop)) |
| Stop one agent run while the rest runs on, to retire it or replace it with another model | `stop_run`, `stop_agent`, `RunStopIntent` ([Stopping one agent run](#stopping-one-agent-run)) |
| Share structured text and deliver messages | `MemoryConfig`, `MemoryStore`, `MessagingPolicy`, `MessageBus` |
| See which agents are running, waiting, or finished | `agent_status`, `AgentStatus` |
| Steer running agents without changing their configuration | `post_note`, `notes_handle`, `control_folder` ([Operator notes](#operator-notes)) |
| Track stale files, directories, memory, or custom resources | `ContextFreshnessConfig`, `ResourceTracker` |
| Gate completion and retain repair context | `CompletionVerifierConfig` |
| Condense selected tool outputs | `DistillationConfig`, custom tool result interceptors |
| Enforce and audit spending | Workflow and agent budgets, manual costs, `CostLedger` |
| Checkpoint, restore, or reconcile uncertain side effects | `PersistenceConfig`, durable steps |
| Continue agents that were running when the workflow crashed | `resume_from`, `interrupted_runs`, `recover_agent` ([Task journal and recovery](#task-journal-and-recovery)) |
| Let agents delegate or vote | `WorkflowServicesConfig`, `enable_voting` |
| Let agents start runs in the background and message them | `subagents`, `SubagentsConfig` ([Background subagents](#background-subagents)) |
| Run an agent in a directory of its own | `run_agent_with`, `spawn_agent_with`, `RunOptions` |
| Route human input | `UserInputHandler`, or handle `UserInputRequired` and continue |
| Keep long-running agents' context small | `AgentConfig::handover` (arko_agents). See [Conversation handover](#conversation-handover) |
| Decide tool approvals and error recovery | `approval_policy`, `error_recovery_policy`, `ToolApprovalHandler`, or handle `AgentPaused` and `resume_agent` |
| Add domain tools | `global_tool`, `agent_tool` |

## Workflow construction

Defined in [workflow/mod.rs](src/workflow/mod.rs). `Workflow` and `WorkflowBuilder` are defined in `workflow` and re-exported at the root.

### Builder signatures and defaults

`Workflow::builder() -> WorkflowBuilder`, `WorkflowBuilder::new() -> Self`, and `WorkflowBuilder::default() -> Self` return an empty builder. A workflow needs a name, an existing workspace directory, an `Arc<LLMClient>`, and at least one uniquely named agent. By default a workflow has no memory tiers, spending limits, hooks, persistence, freshness tracking, custom tools, interceptors, verifiers, distillers, workflow services, input handler, or voting. Metadata is empty, the interception failure policy is `InterceptionFailurePolicy::UseRawResult`, the cancel grace period is 30 seconds, and `wait_for_message` waits up to an automatic limit inside each agent's prompt-cache lifetime. **Every tool call is approved** (`ApprovalPolicy::ApproveAll`) and **errors are recovered from automatically without an attempt limit** (`ErrorRecoveryPolicy::Automatic`), so a workflow runs without human input. See [Approvals and error recovery](#approvals-and-error-recovery).

| Signature | Parameters, purpose, and default |
| --- | --- |
| `name(self, name: impl Into<String>) -> Self` | Workflow identity used in results, hooks, and checkpoint name checks. Required. |
| `workspace(self, path: impl Into<PathBuf>) -> Self` | Working directory for every agent. Required. Must already exist (the builder does not create it). `build` stores it absolute, with `.` and `..` resolved lexically (`std::path::absolute`, then `arko_agents::utils::normalize_path`). Symbolic links are kept, and on Windows no verbatim `\\?\` form is produced (`std::fs::canonicalize` is not used). |
| `llm_client(self, client: Arc<LLMClient>) -> Self` | Model client shared by agents, verifiers, and distillers. Required. The builder reads no credentials itself. `LLMClient::from_env()` configures every provider found in the environment. |
| `agent(self, name: impl Into<String>, config: AgentConfig) -> Self` | Register an agent under a unique workflow registration name. At least one is required. |
| `metadata(self, metadata: WorkflowMetadata) -> Self` | Replace discovery metadata, including supported-host restrictions. |
| `memory(self, config: MemoryConfig) -> Self` | Shared memory tiers and access policies. None by default. |
| `budget(self, limit: f64) -> Self` | Workflow cost limit in dollars. Finite and nonnegative. Unlimited by default. |
| `budget_warning(self, threshold: f64) -> Self` | One-time warning threshold in dollars. Finite and nonnegative. None by default. |
| `agent_budget(self, agent: impl Into<String>, limit: f64) -> Self` | Limit on the named agent's cumulative LLM spend across all its runs. Finite and nonnegative. Unlimited by default. |
| `plan_budget(self, budget: PlanBudget) -> Self` | Limits on the subscription plans' five-hour and seven-day usage windows, in percentage points used since the run started (`PlanBudget { five_hour, seven_day, chatgpt, claude_code }`, with `five_hour` and `seven_day` for each plan's windows, and a plan's own `PlanLimits { five_hour, seven_day }` in place of them for that plan, each limit an `Option<f64>`, finite and nonnegative). None by default. See [Plan budgets](#plan-budgets). |
| `budget_exceeded_policy(self, policy: BudgetExceededPolicy) -> Self` | What happens over the workflow's budget or an agent's. **The default, `Pause`, holds each agent over it** before its next model request until the budget is raised, and nothing is cancelled. `Cancel` cancels the agent. See [Budget holds](#budget-holds). |
| `wait_for_budget_raise(self, enabled: bool) -> Self` | Under `Pause`, whether a run the host started waits in its call for a live raise (as runs inside another agent's tool call always do) instead of returning `AgentPaused`. Off by default. |
| `hook(self, point: HookPoint, callback: HookCallback) -> Self` | Append a synchronous callback at a lifecycle point. Callbacks run in registration order. |
| `persistence(self, config: PersistenceConfig) -> Self` | Enable checkpoints, event and task files, a persistent run identity, and durable steps. |
| `resume_from(self, checkpoint: impl Into<PathBuf>) -> Self` | Restore state from a checkpoint JSON file (not a run directory) during `build`. Requires `persistence` whose run directory holds the checkpoint's `workflow_run_id`. A nil ID in the checkpoint is accepted. Memory access and limits come from the builder. Resumable tasks (completed, waiting for input, paused, or interrupted), and tasks a [graceful stop](#graceful-stop) or `cancel` cancelled (marked `stopped_by`), reject changed user agent configuration. Workflow-injected tools and saved model registry snapshots are ignored. Agent runs that were running when the checkpoint was written come back `Interrupted` with their tasks (see [Task journal and recovery](#task-journal-and-recovery)). See [Persistence and checkpoint state](#persistence-and-checkpoint-state). |
| `rename_memory_tier_on_resume(self, from: impl Into<String>, to: impl Into<String>) -> Self` | On resume, restore the checkpoint's memory tier `from` into the configured tier `to`. A `from` the checkpoint lacks is logged as a warning. |
| `keep_saved_configs<I, S>(self, agents: I) -> Self` (`I: IntoIterator<Item = S>`, `S: Into<String>`) | On resume, these agents' saved runs keep the configuration saved in their tasks while their new runs use the registered one, so a host can change an agent's instructions or tools between runs (a chat's next conversation, say). `build` then checks, instead of an unchanged configuration, that every tool a saved run offers is still available, whether built in, added by the workflow, or registered globally or for the agent (for a Claude Code agent, the names in `arko_agents::claude_code::BUILT_IN_TOOLS` are the CLI's own). A missing one returns `BuilderInvalid` (`agent 'main' run <id> offers tool 'probe', which isn't registered: register it to resume the run`), and an unregistered agent name `keep_saved_configs names unknown agent 'x'`. Adds to earlier calls. |
| `drop_memory_tier_on_resume(self, name: impl Into<String>) -> Self` | On resume, discard the checkpoint's memory tier `name` instead of failing because it isn't configured. |
| `ephemeral(self) -> Self` | Clear the persistence and resume settings made so far. Later setters can enable them again. |
| `context_freshness(self, config: ContextFreshnessConfig) -> Self` | Enable versioned resource tracking. `ContextFreshnessConfig::default()` maps the built-in file tools `read_file`, `list_files`, `write_file`, `edit_file`, and `apply_diff`, and Claude Code's `Read`, `Write`, `Edit`, `MultiEdit`, and `NotebookEdit`. See [Context freshness](#context-freshness). |
| `global_tool(self, tool: Arc<dyn arko_agents::tool::Tool>) -> Self` | Register a tool for every agent and add its name to each agent's available tools. |
| `agent_tool(self, agent_name: impl Into<String>, tool: Arc<dyn arko_agents::tool::Tool>) -> Self` | Register a tool for one agent and add its name to that agent's available tools. |
| `tool_result_interceptor(self, interceptor: Arc<dyn ToolResultInterceptor>) -> Self` | Append an interceptor for every agent's tool results. |
| `agent_tool_result_interceptor(self, agent_name: impl Into<String>, interceptor: Arc<dyn ToolResultInterceptor>) -> Self` | Append an interceptor for one agent. It runs after the global interceptors. |
| `interception_failure_policy(self, policy: InterceptionFailurePolicy) -> Self` | Error handling for interceptors and distillers, applied by the agent runtime to each agent's chain as a whole. Default `UseRawResult`. |
| `tool_image_sink(self, sink: Arc<dyn arko_agents::ToolImageSink>) -> Self` | Give `sink` the images every agent's tool calls return (`read_file`, MCP tools, a host's tools, Claude Code's `Read`), once per call, keyed by the run (`task_id`) and the call's `tool_call_id` as its events name them, before the ones the model can't take are left out. Events carry no image bytes. None by default. |
| `background_jobs(self, jobs: Arc<arko_agents::terminal::TerminalManager>) -> Self` | Let every agent's long tool calls move to the background as jobs of `jobs` (Unix and Windows). While a call of a tool whose `background_mode` isn't `Never` runs, `jobs` lists it, and the host moves it with `move_to_background`. A moved call's result reaches its agent as a note, and `wait_for_agents` and `wait_for_message` end when moved. When a subagent's run ends, its moved calls and monitors stop, and its commands keep running. None by default. See [Moving calls to the background](../docs/arko_agents-guide.md#moving-calls-to-the-background). |
| `completion_verifier(self, config: CompletionVerifierConfig) -> Self` | Append a gate on matching agents' `complete_task` calls. See [Completion verification](#completion-verification). |
| `distiller(self, config: DistillationConfig) -> Self` | Append an LLM distiller. Distillers run after all custom interceptors. See [Tool-result distillation](#tool-result-distillation). |
| `workflow_services(self, config: WorkflowServicesConfig) -> Self` | Enable bounded agent-to-agent delegation through the `run_workflow_agent` tool. |
| `subagents(self, config: SubagentsConfig) -> Self` | Let the workflow services' callers start runs of their targets in the background, with messages between them. Requires `workflow_services`. Off by default. See [Background subagents](#background-subagents). |
| `messaging(self, policy: MessagingPolicy) -> Self` | Set whom each agent may message with `send_message`. The default, `MessagingPolicy::all()`, lets every agent message every other agent. See [Messaging policy](#messaging-policy). |
| `disable_messaging(self) -> Self` | Offer no agent `send_message`, `list_recipients`, or `wait_for_message`. |
| `notes_by_name_only<I, S>(self, agents: I) -> Self` (`I: IntoIterator<Item = S>`, `S: Into<String>`) | Agents (registration names) that get a note only when its audience names them, as completion verifiers do, for roles that must stay independent. Adds to earlier calls. None by default, besides the verifiers. See [Operator notes](#operator-notes). |
| `control_folder(self) -> Self` | Opt-in, off by default. Reads notes that another process writes to `<run dir>/control/` (see [The control folder](#the-control-folder)). `build` creates the folder. Requires `persistence`. `ephemeral` clears it. |
| `wait_for_message_timeout(self, timeout: impl Into<WaitForMessageTimeout>) -> Self` | How long `wait_for_message` waits at most, for agents without their own setting. Default `WaitForMessageTimeout::Auto`, just inside each agent's prompt-cache lifetime (4 min 30 s, or 54 min with Anthropic's 1-hour TTL). A `std::time::Duration` converts to a fixed limit, more than zero and at most 24 hours. See [Waiting for a message](#waiting-for-a-message). |
| `agent_wait_for_message_timeout(self, agent: impl Into<String>, timeout: impl Into<WaitForMessageTimeout>) -> Self` | The same for one agent, in place of the workflow's setting. |
| `user_input_handler(self, handler: Arc<dyn UserInputHandler>) -> Self` | Answer agents' `ask_user` questions asynchronously. Without a handler, the run suspends and returns `UserInputRequired`. |
| `approval_policy(self, policy: ApprovalPolicy) -> Self` | How tool-approval requests are answered. **The default, `ApprovalPolicy::ApproveAll`, approves every call**, including calls an agent's configuration marks for approval. See [Approvals and error recovery](#approvals-and-error-recovery). |
| `error_recovery_policy(self, policy: ErrorRecoveryPolicy) -> Self` | How error-recovery pauses are answered. **The default, `ErrorRecoveryPolicy::Automatic`, recovers without an attempt limit** from errors that may pass, and fails clearly on those that won't. |
| `enable_voting<I, S>(self, voters: I) -> Self` (`I: IntoIterator<Item = S>`, `S: Into<String>`) | Create an in-memory ballot store and offer the `cast_vote` tool to `voters` (registration names) only. |
| `max_parallel_agents(self, max: usize) -> Self` | Run at most `max` agents of a `run_agents_parallel` call at once. The others start in input order as places free up. No cap by default. `build` rejects zero. Workflow services have their own limit (`max_active_calls`). |
| `fail_fast(self, enabled: bool) -> Self` | Opt-in, off by default. In `run_agents_parallel`, the first agent that fails stops the others. An agent that pauses (including one whose completion its verifier could not review) doesn't. |
| `cancel_grace_period(self, grace_period: std::time::Duration) -> Self` | How long a stopped run waits for each running agent to stop before aborting it. Default `workflow::cancel::DEFAULT_CANCEL_GRACE_PERIOD`, 30 seconds. See [Stopping a run](#stopping-a-run). |
| `build(self) -> Result<Workflow, WorkflowError>` | Validate, initialize or restore state, and fire `PreWorkflow`. May create persistence files. Runs no agent. |

### Build validation and tool composition

`build` returns `BuilderMissing` when the name, workspace, client, or agents are missing, and `BuilderInvalid` for:

- a workspace that does not exist or is not a directory.
- a host that matches none of a nonempty `metadata.supported_platforms` list.
- duplicate agent names.
- a negative or non-finite workflow budget, warning threshold, or agent budget.
- an unknown agent name in `agent_tool`, `agent_tool_result_interceptor`, `agent_budget`, a completion verifier's verifier or targets, a distiller's targets (`distiller '<name>' targets unknown agent '<target>'`), the messaging policy (`messaging policy references unknown agent '<name>'`), `agent_wait_for_message_timeout` (`wait_for_message timeout references unknown agent '<name>'`), a memory tier's `AccessScope::Only` readers or writers (`memory tier '<tier>' readers name unknown agent '<name>'`), workflow services callers or targets (`workflow services caller '<name>' is not a registered agent`), or the voters (`voting names unknown agent '<name>'`).
- a completion verifier without a target (`completion verifier '<name>' has no target`), or a distiller without a target agent or a target tool.
- an unknown agent in `notes_by_name_only` (`notes_by_name_only names unknown agent '<name>'`), or `control_folder` without `persistence` (`control_folder requires persistence: ...`).
- two memory tiers with the same name.
- a fixed `wait_for_message` limit of zero or more than 24 hours, for the workflow or an agent.
- `max_parallel_agents(0)`.
- a distiller whose `max_output_tokens` is zero or whose `reasoning` configuration is for a different provider than its model.
- a distiller that targets one of Claude Code's built-in tools for an agent whose model runs in Claude Code (`distiller '<name>' targets Claude Code's '<tool>', whose output can't be distilled`), because Claude Code produces that output itself.
- `resume_from` without `persistence`, a checkpoint for a different workflow name, a checkpoint run ID that differs from the persistence directory's, or a resumable task (or one a stop cancelled) whose user agent configuration changed (a task the host reconfigured, `Task::reconfigured`, keeps the configuration it was given and is not compared, and the runs of agents set with `keep_saved_configs` are checked for their tools only). That error names the changed fields by their paths in the configuration's JSON, objects field by field and lists as a whole, the first ten, as in `agent 'coder' configuration changed since its checkpoint (instructions, tool_config.available_tool_names changed); use the original configuration to resume it or register the changed configuration under another agent name`. Tool descriptions and workflow-injected tools are not part of the comparison.

Checkpoint read and parse failures return `Io` or `Json`. A warning threshold may exceed the budget limit. Tool names are only warned about, since MCP servers and skills add tools whose names aren't known at build. A tool the context freshness mapping or a distiller's targets name that no agent is configured with (`available_tool_names`), no workflow or custom tool has, and no `arko_agents` or Claude Code built-in has is logged as a warning. Workflow services that name no caller are logged as a warning too, since no agent is offered `run_workflow_agent`. The builder neither installs nor checks metadata dependencies. `WorkflowMetadata::missing_dependencies` reports the missing ones, for the workflow to act on at startup.

Duplicate global tool names, duplicate names within one agent's tools, and a name shared by a global and an agent-local tool are rejected. Global tools cannot use the reserved names `complete_task`, `ask_user`, `read_memory`, `write_memory`, `list_memories`, `run_workflow_agent`, or `cast_vote`, nor the [subagent tools](#background-subagents)' names while `subagents` is set, nor `send_message` or `list_recipients` while any agent is offered the workflow's messaging tools, nor `wait_for_message` while any agent is offered it. An agent-local tool with one of these names is rejected for an agent that is offered the workflow's tool. Agent-local tools aren't checked against reserved names, and an agent-local tool with a reserved name replaces that workflow or built-in tool for its agent, except `complete_task`, which `arko_agents` never lets a non-built-in tool replace. Custom tools are registered after the workflow tools, so a custom tool named like an `arko_agents` built-in (for example `read_file`) replaces it for the agents that receive it.

The workflow adds tools automatically, except to a chat (`arko_agents::AgentToolConfig::chat`), which gets none of them, none of the tools registered for it or for every agent, and no list of the agents it may message. It adds `send_message` and `list_recipients` for each agent with at least one permitted recipient under the messaging policy. `wait_for_message` for each agent that another agent may message. `read_memory` for each agent that can read at least one memory tier, `write_memory` for each that can write at least one, and `list_memories` for each that can do either. `run_workflow_agent` for permitted callers when services are enabled. `cast_vote` for the voters named in `enable_voting`. The agent's pending messages are appended to the prompt of each new run and continuation, under `# Messages from other agents:`. A new run's initial prompt also lists the agents it may message. See [What agents are told about their recipients](#what-agents-are-told-about-their-recipients). Skills configured on an agent are initialized against the workspace. There is no separate graph or scheduler configuration.

Interceptors run in this order, first the global interceptors, then the agent's own interceptors, then the distillers that target the agent, each group in registration order. Each stage receives the previous stage's result. The interception failure policy applies to each stage. A stage that fails is retried up to the policy's attempts (the stages before it don't run, or get paid for, again) and, if it still fails, skipped, so the next stage and then the agent get the output of the last stage that succeeded. A failure never undoes an earlier stage's work, so an interceptor that redacts content keeps it redacted whatever fails after it. Under `InformAgent` and `RetryThenInformAgent` the result ends with a note that an interceptor failed. A stage fails open for its own work. An interceptor that fails can't transform what it was given, so one that must withhold content (for example to redact secrets) should replace the output with a placeholder rather than return an error when it can't process it, and should run before any interceptor that might fail. With a single interceptor, a failure after the last attempt delivers the original tool output. `load_skill` results are intercepted only when the agent's `SkillsConfig::load_skill_interception` opts in, under the policy it sets. Every agent's tool calls pass through the workflow's execution middleware, which enforces a stopped run and runs tool hooks, freshness checks, and completion verification at tool execution, so these also apply to automatically approved and batched calls. If an external execution-middleware `before` or `after` hook returns an error, the agent pauses with cause `Middleware`. A `before` failure leaves the call unrun, while an `after` failure keeps the tool result and does not repeat the call when the host continues it.

### Workflow metadata

Defined in [workflow/config.rs](src/workflow/config.rs). All four types implement Serde `Serialize` and `Deserialize`. Only `WorkflowMetadata` implements `Default`. `WorkflowConfig` is not re-exported.

| Type | Public fields and meaning |
| --- | --- |
| `SupportedPlatform` | `os: String`, a Rust host OS name (`std::env::consts::OS`). `arch: Option<String>`, an optional architecture restriction (`None` when omitted in JSON). |
| `WorkflowDependency` | `name: String`, `description: String` (required). `executable`, `install_hint`, `os` and `arch` are `Option<String>`, each `None` when omitted in JSON. `optional: bool`, `false` (required) when omitted. An optional dependency helps but isn't required, and the workflow works without it. |
| `WorkflowMetadata` | `description: String` and `version: String`, default `""`. `supported_platforms: Vec<SupportedPlatform>`, `required_dependencies: Vec<WorkflowDependency>` (required and optional ones, despite the name), and `tags: Vec<String>`, default empty. Every field may be omitted in JSON. An empty platform list means any host. |
| `WorkflowConfig` | `name: String`, `workspace: PathBuf`, `metadata: WorkflowMetadata` (default when omitted in JSON). A serializable descriptor. Workflows are constructed with the builder. Unknown JSON fields are ignored. |

`SupportedPlatform` has these methods:

```rust
fn new(os: impl Into<String>) -> Self;
fn arch(self, arch: impl Into<String>) -> Self;
fn matches_host(&self) -> bool;
```

`new` sets no architecture restriction. `matches_host` compares the OS and, when set, the architecture case-insensitively with `std::env::consts::{OS, ARCH}`. Multiple supported platforms are alternatives.

`WorkflowDependency` and `WorkflowMetadata` have these methods:

```rust
// WorkflowDependency: a required dependency on any host, then optional settings.
fn new(name: impl Into<String>, description: impl Into<String>) -> Self;
fn executable(self, executable: impl Into<String>) -> Self;
fn install_hint(self, hint: impl Into<String>) -> Self;
fn os(self, os: impl Into<String>) -> Self;
fn arch(self, arch: impl Into<String>) -> Self;
fn optional(self) -> Self;
// WorkflowMetadata
fn missing_dependencies(&self) -> Vec<&WorkflowDependency>;
```

`missing_dependencies` returns, in declaration order, the dependencies that apply to this host (`os` and `arch` unset or equal, ignoring case, to `std::env::consts::{OS, ARCH}`) whose `executable` isn't found, required and optional alike. Filter them by `optional`. A name is looked for in each `PATH` directory, on Windows also with each `PATHEXT` extension (`.COM;.EXE;.BAT;.CMD` when unset). An `executable` containing `/` or `\` is checked as a file. On Unix the file must have an execute bit. A dependency without an `executable` (a login, a service) is never reported. It prints nothing and installs nothing, and the workflow decides, for example stopping with each missing required dependency's description and `install_hint` before any model call, and warning about optional ones.

```rust,ignore
let metadata = WorkflowMetadata {
    required_dependencies: vec![
        WorkflowDependency::new("cargo", "Builds the crate").executable("cargo"),
        WorkflowDependency::new("ripgrep", "Fast code search")
            .executable("rg")
            .install_hint("apt install ripgrep")
            .optional(),
    ],
    ..WorkflowMetadata::default()
};
let (optional, required): (Vec<_>, Vec<_>) = metadata
    .missing_dependencies()
    .into_iter()
    .partition(|dependency| dependency.optional);
```

## Workflow execution and lifecycle

Defined in [workflow/mod.rs](src/workflow/mod.rs), [engine](src/workflow/engine.rs). All methods below are on `Workflow`.

### Run, continue, and spawn

| Signature | Parameters, result, and behavior |
| --- | --- |
| `async run_agent(&self, agent_name: &str, prompt: &str) -> Result<AgentResult, WorkflowError>` | Start a new invocation of a registered agent and await it. Adds pending messages and the list of agents it may message to the prompt, runs the managed driver, and records costs and the result. An unknown name returns `AgentNotFound`. |
| `async continue_agent(&self, previous: &AgentResult, prompt: &str) -> Result<AgentResult, WorkflowError>` | Resume a `Completed` or `WaitingForInput` result with its task, conversation, and run ID. `prompt` is the follow-up task or the user's answer. A `Paused` result whose completion a verifier could not review (`VerifierFailed`) continues as `resume_agent` with `AgentResume::ContinueAfterError { note: Some(prompt) }` does. The messages queued for the agent that the task doesn't already hold are appended (see [Message bus](#message-bus)), and no recipient list is added. The continuation message follows the agent's `EachUserMessage` environment-detail and reminder rules. Context the task took from its command channel but did not add before it stopped (`Task::pending_context_injections`) is added at its first turn boundary, not presented again. Other statuses, or a task whose state does not match its status, return `BuilderInvalid`. The new result replaces the recorded result with the same run ID. Token and cost totals are cumulative. |
| `async resume_agent(&self, previous: &AgentResult, resume: AgentResume) -> Result<AgentResult, WorkflowError>` | Continue a `Paused` result (returned as `AgentPaused`) with a decision, `ToolApprovals` for calls awaiting approval, `ContinueAfterError`, `FailAfterError`, or `ResolveRefusal` for an error-recovery pause. A result that isn't paused, or a decision for the other kind of pause (or naming a call that isn't awaiting approval), returns `BuilderInvalid`. So does `ResolveRefusal` for a pause whose cause isn't `Refusal` (or `Unknown`, a pause saved before causes were recorded, which the runtime checks), without a change, or for a Claude Code agent. When the runtime can't apply the changes (naming a message outside the refused input, for example), the host gets its error under the error recovery policy, as another error at that pause. Under `ReturnToCaller` (or for an `Unknown` pause) the call returns `AgentPaused` again. Under `Automatic` or `Limited` a refusal's run fails with that error. Fires `PreAgentSpawn` and runs the managed driver like a continuation. Messages queued while it was paused are delivered before the decision takes effect, and none it already holds is delivered again. |
| `agent_result(&self, agent_run_id: Uuid) -> Option<AgentResult>` | The recorded result of a run, such as the one `UserInputRequired`, `AgentPaused`, or `VerifierFailed` names by `task_id`. |
| `interrupted_runs(&self) -> Vec<AgentResult>` | The recorded `Interrupted` results the host must recover, not being recovered. After `resume_from` they are the runs that were running when the checkpoint was written, with their tasks as last saved (and runs that stopped as interrupted on their own, after a lasting connection failure). There are two kinds, top-level runs (`task.parent` is `None`) and orphaned nested runs (`task.parent` set), which are runs started inside another run's tool call that that run can no longer resume, because it ended or answered the call. A nested run whose parent still holds its call open is not listed. See [Task journal and recovery](#task-journal-and-recovery). |
| `async reopen_agent(&self, previous: &AgentResult, prompt: &str) -> Result<AgentResult, WorkflowError>` | Continue a `Cancelled` result (cancelled by the budget under `BudgetExceededPolicy::Cancel`, stopped with the workflow, or aborted after the grace period), including one recorded by an earlier version, from where its task stopped, with its run ID. It is recovered as an interrupted run is (calls without results are settled by their `recovery_mode`) and gets a message, `REOPENED_NOTE` (it was reopened, and calls reported cancelled or stopped didn't finish), then `prompt` if not empty. A nested run becomes the host's (its parent link is released) when it is orphaned, as for `recover_agent`. One its parent is running, or its parent's recovery settles, is refused, and so is a run recorded superseded (replaced after a restart). Another status returns `BuilderInvalid`. Under `Cancel`, a budget still exceeded refuses it (`BudgetExceeded`, `AgentBudgetExceeded`). Returns as `run_agent` does. |
| `async recover_agent(&self, previous: &AgentResult) -> Result<AgentResult, WorkflowError>` | Continue an `Interrupted` result where its task was saved, with its run ID and without a new message. Returns as `run_agent` does. Tool calls it was running are settled by their tools' `recovery_mode` (by default answered as possibly run, without running them again, while read-only built-ins run again and `run_workflow_agent` recovers the run it started). A wait it was saved in (tool approval, error-recovery pause, question) is answered again under the workflow's policies and handlers. Another status returns `BuilderInvalid`, and so do a nested run its parent can still resume (recover the parent, or, while the parent runs, let it resume the run) and a run being recovered already, since a run is never recovered twice. An orphaned nested run is recovered as the host's own, and its parent link (`task.parent`) is released. A `previous` that was `Interrupted` but has since been recovered (or resumed by an agent) returns the outcome recorded, without running again. Fires `PreAgentSpawn` like a continuation. |
| `async run_agents_parallel(&self, agents: Vec<(&str, String)>) -> Result<Vec<AgentResult>, WorkflowError>` | Run agent-name/prompt pairs concurrently. One agent may appear several times. All the agents are marked running before any is prompted, so each one's recipient list shows the others as running. Under `max_parallel_agents`, at most that many run at once, and each is marked running as it starts (a waiting one is `not_started`). Waits for every invocation, then returns the results in input order, or the first failure in input order. An agent waiting for a place was admitted (`PreAgentSpawn`) when the call began. When it starts, the run's stop and the workflow budget are checked again, and either ends it before it starts (its error in its place). With `fail_fast`, the first failure instead stops the others (each is cancelled, winding down within the cancel grace period, and recorded `Cancelled` with a reason naming the failed agent, and an agent inside one's tool call, such as a verifier, is aborted with it after the grace period), agents not yet started don't start and have no result, and the call returns that failure. An agent that pauses (waiting for input, paused for a decision, or paused because a completion verifier could not review its completion) isn't a failure, and doesn't trigger `fail_fast`. Its result comes back in its place with status `WaitingForInput` or `Paused`, and the others run on. So does a run its own stop cancelled (`AgentStopped`), with status `Cancelled`. Finalized outcomes are recorded either way. |
| `async spawn_agent(&self, agent_name: &str, prompt: &str) -> Result<AgentHandle, WorkflowError>` | Set up and start the agent and return its handle. Nothing drives its events until `wait` is called or the raw channels are used. |
| `async run_agent_with(&self, agent_name: &str, prompt: &str, options: RunOptions) -> Result<AgentResult, WorkflowError>`, `async spawn_agent_with(&self, agent_name: &str, prompt: &str, options: RunOptions) -> Result<AgentHandle, WorkflowError>` | `run_agent` and `spawn_agent` with options. `RunOptions` (`Debug`, `Clone`, `Default`, `PartialEq`, `Eq`, `#[non_exhaustive]`, built with `RunOptions::new().workspace(path)`) has the fields `workspace: Option<PathBuf>` and `images: Vec<arko_ai::Image>` (`RunOptions::new().images(images)`). `workspace` is the run's working directory instead of the workflow's workspace. It is the task's `current_working_directory`, so the shell, terminal and file tools work there and the path checks allow it, and skills load from it. A relative path is resolved against the workflow's workspace, and the result is absolute with `.` and `..` resolved lexically. A directory that doesn't exist returns `BuilderInvalid` (`... does not exist`). Saved with the task, so a continued, resumed or recovered run keeps it. `images` go after the prompt in the run's first message (`Task::initial_task_images`, moved into that message when it is written, `Task::initial_images` gives them), and again in the first message of a conversation that replaces it at a handover. An image over the model's limits is shrunk to fit them first (`arko_ai::Model::sendable_image`). One the agent's model can't take returns `BuilderInvalid` with the reason before the run starts (`agent 'x' can't be sent image 2: model 'claude-opus-5-5' takes images as data, not by URL`), as does one it can't read or shrink enough. |
| `async continue_agent_with(&self, previous: &AgentResult, prompt: &str, options: ContinueOptions) -> Result<AgentResult, WorkflowError>`, `async reopen_agent_with(..., options: ContinueOptions)` | `continue_agent` and `reopen_agent` with options. `ContinueOptions` (`Debug`, `Clone`, `Default`, `#[non_exhaustive]`, not `PartialEq`, since it can hold an `AgentConfig`, built with `ContinueOptions::new().images(images)`) has `images: Vec<arko_ai::Image>`, which go after the prompt, in the same message (for a reopened run, in the context update that gives it the prompt, each after `Image N:`). Images over the model's limits are shrunk to fit. Images the agent's model can't take return `BuilderInvalid` before anything runs, and so do images a request would then hold too many of, or too much image data (`... its conversation holds 100 images, and model 'claude-opus-5-5' takes at most 100 in a request (cut the conversation back or hand it over to make room)`), as do images for a run continued after its completion verifier couldn't review it. `config: Option<AgentConfig>` (`ContinueOptions::new().reconfigure(config)`, `continue_agent_with` only) continues the run under another configuration, such as a switch between a chat and an agent. It keeps the run ID, conversation and prompt cache key, with the new configuration's tools (the workflow's added as for a new run, none to a chat) and a system prompt written from it (`arko_agents::Task::reconfigure`). The model must stay the same (`BuilderInvalid` with `a reconfigured task keeps its model ...`), and the new configuration is checked as one given at a start is (a completion verifier's target can't end its task with text). The next request writes the provider's prompt cache afresh. A Claude Code agent's session resumes with the new flags and `--system-prompt-snapshot off`. The task keeps the configuration from then on, also across `resume_from`. A reopened or recovered run, and one continued after its completion verifier couldn't review it, refuse it. `cut_back_to: Option<usize>` (`ContinueOptions::new().cut_back_to(index)`, `continue_agent_with` only) first cuts the conversation back to just before message `index`, one of `Task::user_turns()`, and sends the prompt in its place (an edited message, or the same one to regenerate the reply), as `arko_agents::Task::cut_back` describes. It applies to a run of any status that isn't running (completed, waiting for input, paused, cancelled, failed, interrupted), under the same run ID. Index `0` starts the conversation over from the prompt, with the standing notes and the list of recipients as a new run's prompt has them. The notes the removed messages gave the run are given again at its next turn boundary, except one-time notes (they went with the messages around them, such as what the user wrote while it worked). Messages from other agents in them are not delivered again. The subagents the run started in the removed messages belong to a timeline it abandoned, so they are detached (`SubagentRecord::detached`), and their mail, whether given already, waiting or still to come, is withheld from it (`MailItem::withheld`) and stays in the subagent records for the host to show. The mail of the subagents it still started that the removed messages held is given again. The run forgets what it read in the removed messages (context freshness), as after a handover. `ConversationCutBack { message_index, removed_messages }` is published as the run goes on. A Claude Code agent's session resumes at the reply before the cut. An index that isn't a user turn returns `BuilderInvalid`. `workspace: Option<PathBuf>` (`ContinueOptions::new().workspace(dir)`, `continue_agent_with` only) continues the run in another directory, resolved and checked as `RunOptions::workspace` is. Its tools work there and skills load from it, and its system prompt is written again (it may state the directory). A Claude Code run's arko tools and checks move too, while the CLI resumes its session in the directory the session started in, where Claude Code finds it, with the new directory added (`--add-dir`) and the new prompt (`arko_agents::Task::move_to`, `Task::claude_code_directory`). Claude Code's own tools run in the new directory too, since the runtime changes each call to start there. A reconfiguration publishes `Reconfigured { chat, tools }` as the run goes on. |
| `import_agent_run(&self, agent_name: &str, task: arko_agents::Task) -> Result<AgentResult, WorkflowError>` | Adds a completed run of `agent_name` holding `task`, to continue like any completed run, such as a branch (`Task::branch(index)`, a copy with its own run ID whose Claude Code session forks the original's) of a run of this workflow or another. It keeps the configuration saved in `task` and counts as given every note posted so far. `AgentNotFound` for an unregistered agent. `BuilderInvalid` for a task that isn't `Completed` or whose ID the workflow has. The result is in the next checkpoint. |
| `async refresh_prompt_cache(&self, run: &AgentResult) -> Result<arko_agents::CacheRefresh, WorkflowError>` | Keeps the prompt cache of a run that isn't running (waiting for the user's next message, or paused) warm, by sending its next request with a short ping after it and discarding the reply (`arko_agents::AgentRuntime::refresh_prompt_cache`, where a Claude Code agent pings an unsaved fork of its session). A run paused with calls that have no results yet (for a tool approval) sends its last request again unchanged instead. A subagent's run is offered the tools of its depth. Nothing is added to the run's task or the event log. The cost is charged to the agent as a model call's is, with `OnBudgetWarning` and `OnBudgetExceeded`, and saved at once, in the live checkpoint with the task journal or else by an automatic checkpoint. Returns `CacheRefresh` with the usage, the output limit sent, whether the last request went again, and why a smaller limit was refused. With a budget exceeded (the workflow's, the agent's, or a plan window's) nothing is sent, and the call returns `BudgetExceeded`, `AgentBudgetExceeded` or `PlanBudgetExceeded`. A running run (one waiting in a host handler counts as running) and a run started inside another run's tool call return `BuilderInvalid`. A route that can't be kept warm returns `Agent(LLM(UnsupportedFeature))`. |

Under `BudgetExceededPolicy::Pause` (the default), a run over the budget starts (or continues) and is held at its next model request. See [Budget holds](#budget-holds). Under `Cancel`, all four return `BudgetExceeded` without starting an agent when the workflow budget is already exceeded, `AgentBudgetExceeded` without starting or continuing that agent when its own cumulative limit is already exceeded, and `PlanBudgetExceeded` when a [plan budget](#plan-budgets) window is. Once the run is stopped, all four return the stop's error (`Cancelled` or `AbortedByHook`) at once, without firing `PreAgentSpawn` or starting an agent, and during a graceful stop's wrap-up they return `Cancelled`. See [Stopping a run](#stopping-a-run) and [Graceful stop](#graceful-stop).

The managed driver (used by `run_agent`, `continue_agent`, `resume_agent`, `recover_agent`, `run_agents_parallel`, and `AgentHandle::wait`) answers tool approvals under the workflow's `ApprovalPolicy` and error-recovery pauses under its `ErrorRecoveryPolicy` (see [Approvals and error recovery](#approvals-and-error-recovery)), answers budget holds (see [Budget holds](#budget-holds)), and cancels the task when the task is interrupted or, under `BudgetExceededPolicy::Cancel`, when the workflow or agent budget, or a plan budget window, is exceeded. A task it cancels ends `Cancelled`, and the call returns `AgentInterrupted`, `BudgetExceeded`, `AgentBudgetExceeded`, or `PlanBudgetExceeded`. The driver waits for a stopped agent at most the run's grace period, as described under [Stopping a run](#stopping-a-run).

A managed call returns the `AgentResult` of a completed task. Otherwise it returns the driver's error, or an error for the finalized status, `UserInputRequired` for `WaitingForInput`, `AgentPaused` for `Paused` (`VerifierFailed` when a completion verifier could not review the completion), `AgentInterrupted` for `Interrupted`, `AgentCancelled` for `Cancelled` (`AgentStopped` when the run's own stop cancelled it, see [Stopping one agent run](#stopping-one-agent-run)), and `Agent(ArkoAgentsError::TaskExecution)` with the failure reason for `Failed`. Once the run is stopped, the stop's error replaces any of these outcomes. An error can be returned after the invocation's `AgentResult` has been recorded. `result().agent_results` and `agent_result` hold finalized failures, cancellations, and pauses.

`run_agent`, `continue_agent`, `resume_agent`, and `run_agents_parallel` fire `OnWorkflowFailed` once before returning a failure, which is an error other than `Cancelled`, `AgentStopped`, `UserInputRequired`, `AgentPaused`, or `VerifierFailed` (a pause isn't a failure, and `VerifierFailed` leaves the task paused), including `AbortedByHook`. An `Abort` from that hook stops the run, and the call returns the stop's error. When the run was already stopped, its first cause's error is kept. Calls refused because the run is already stopped do not fire the hook, and neither do `spawn_agent` and `AgentHandle::wait`.

### Approvals and error recovery

Defined in [policy.rs](src/policy.rs), [engine](src/workflow/engine.rs). The types are in the prelude.

#### Tool approvals

An agent asks for approval for a call its configuration doesn't auto-approve (`default_auto_approve_tool_use` and per-tool `auto_approve_tool_use`, which default to asking) and for a path outside the workspace under `PathSafetyPolicy::RequireApproval`. Each request carries an `arko_agents::ApprovalReason` saying which.

| `ApprovalPolicy` | Behavior |
| --- | --- |
| `ApproveAll` (**default**) | Approves every call, so the workflow runs without human input. The first such call of each agent is logged as a warning naming its tool and why it needed approval, and saying that the agent's other calls that need approval are approved too, without further warnings. |
| `RespectAgentConfig` | A call the agent's configuration marks for approval pauses the agent (it is suspended and recorded `Paused`). The call returns `AgentPaused` with `AgentPause::ToolApproval { calls }`, and `OnAgentPaused` fires. Decide with `resume_agent(&result, AgentResume::ToolApprovals(decisions))`. A call without a decision is rejected. |
| `Handler(Arc<dyn ToolApprovalHandler>)` | The handler decides each such call as it is asked, with `ToolApprovalOutcome::Approve`, `Reject { reason }` (the agent is told), or `Pause` (as under `RespectAgentConfig`). It may wait on a person. A stopped run doesn't wait for it. |

```rust
#[async_trait::async_trait]
pub trait ToolApprovalHandler: Send + Sync {
    async fn decide(&self, request: ToolApprovalRequest) -> ToolApprovalOutcome;
}
// ToolApprovalRequest: workflow_run_id, agent_run_id, agent_name (registration name),
// tool_call_id, tool_name, parameters, tool_description, reason: ApprovalReason.
```

The tool hooks receive the reason. `HookContext::approval_reason` is set at `PreToolExecution` and `PostToolExecution` for a call that awaited approval.

#### Error recovery

The agent runtime recovers on its own first, under its `llm_error_policy` and `llm_retry_config` (the default retries a failed request with backoff, a rate limit after the provider's delay, until 10 attempts or the 5-minute elapsed budget run out, and never cuts off a request in flight). Only then, under the pausing error policies (the default), does the agent pause for error-recovery approval, with an `arko_agents::ErrorRecoveryCause`. The workflow's policy then decides. The driver never sends a model request itself, so nothing is retried twice.

| `ErrorRecoveryPolicy` | Behavior |
| --- | --- |
| `Automatic` (**default**) | Recovers without an attempt limit. On `RetriesExhausted` (an outage, repeated rate limits), after a wait growing from 30 seconds to 15 minutes while pauses repeat without a model response in between, and at least the delay the provider asked for (`retry_after_ms`), the failed request is sent again unchanged, starting a new round of the agent's retries. On `UsageLimit` it waits until the reported reset plus 5 seconds (or, with no reset reported, the growing wait, and when the limit is reported again right after a reset, at least the growing wait, so a reset time already past is not retried every few seconds) and sends the request again. Each wait gets up to 30 seconds (at most a quarter of it) added at random, so agents sharing a plan don't all resend at once. On `ToolError` (the agent's opt-in `ToolErrorPolicy::PauseAndAwaitApproval`) it continues at once. The calls' results report the errors. On `NonRetryable`, `Refusal` (after the agent's opt-in `refusal_fallback`, if configured, also refused, since the same input would be refused again, while a fallback request that fails otherwise pauses with its own cause, such as `RetriesExhausted`, and is sent again), and `UnusableResponses`, the task fails with a clear reason, and the cause is recorded with the failure (`TaskState::Failed { cause }`, `AgentResult::failure_cause`, and the `PostAgentFailure` hook's `cause` metadata), so a host can tell an expired login or a bad key (`NonRetryable`) from other failures. The failure also keeps the error's kind when it was classified (`TaskState::Failed { error_kind }`, `AgentResult::failure_kind`, and the hook's `error_kind` metadata), such as `Authentication` (a lost login, so every run on that credential fails until it is renewed), `Billing`, `ContextOverflow` (a request over the model's input limit that the agent's overflow guard couldn't make fit, which concerns this run alone), `InvalidRequest`, and so on. The policy treats every kind alike. A host that wants a lost login to stop the whole run stops it (`Workflow::stop` with `StopRequest::failure`, see [Graceful stop](#graceful-stop)) when `failure_kind()` is `Some(ErrorKind::Authentication)`. On `Monitor` (an opt-in plan-usage or cache-efficiency monitor's pause), `Middleware` (a completion its verifier couldn't review or an external execution-middleware hook error, which the run call returns to its caller), and an unrecorded cause, the agent pauses for the caller, as configured. `BudgetExceeded` (a budget hold) isn't the policy's to decide (see [Budget holds](#budget-holds)). Continuing adds nothing to the conversation (`UserCommand::ContinueAfterError` without a note), so the model is never told a user approved. While it waits, the agent's status is `waiting (retrying until …)` or `waiting (usage_limit until …)`. |
| `Limited { max }` | As `Automatic`, for at most `max` automatic recoveries in each call that drives the agent (a run, continuation or resume). The next such pause fails the task with a reason naming the limit. `max: 0` recovers nothing automatically. |
| `ReturnToCaller` | Every error-recovery pause pauses the agent, and the call returns `AgentPaused` with `AgentPause::ErrorRecovery { error, cause, error_kind }` (`error_kind` is the model-request error's `arko_ai::ErrorKind`, when classified). Continue with `resume_agent` and `AgentResume::ContinueAfterError { note }` (without a note the request is sent again unchanged, and a note is added for the model first), `FailAfterError { reason }`, or, for a refusal, `ResolveRefusal(changes)`. |

An agent whose `llm_error_policy` is `RetryThenInformAgent` or `InformAgentAndContinue` doesn't pause. When its recovery ends it is interrupted, and the call returns `AgentInterrupted`.

#### Nested runs can't pause

An agent running inside another agent's tool call (a completion verifier, or an agent run with `run_workflow_agent`) can't be paused for the caller, since its parent is waiting on the call. Under `RespectAgentConfig` its `complete_task` is approved even when its configuration marks it for approval (as a default agent configuration does), since its parent waits for that result. Where it would otherwise pause (another call that needs approval under `RespectAgentConfig`, a handler's `Pause`, an error-recovery pause left to the caller, or an `ask_user` question without a user input handler), the workflow stops it instead, and it ends `Cancelled` with a reason saying what it needed. A verifier's run counts as a failed review (see [Completion verification](#completion-verification)). `run_workflow_agent` returns a tool error to the delegating agent. Under a `Handler` the handler decides its calls, `complete_task` included. Give such agents configurations that auto-approve the tools they use, or decide them with a `Handler`. (A delegated agent whose own completion its verifier could not review is the exception. It is left paused, and the delegating agent gets the `VerifierFailed` error at once.)

#### The agent's command timeout

The workflow decides every wait of the agents it drives, under the policies and handlers above, or pauses the agent for its caller. So it runs them with `arko_agents::HostDecisions` set. No wait of the run is timed, and an agent's opt-in `user_command_timeout_seconds` never cuts a workflow's decision short, whether it is a handler waiting on a person, automatic recovery's wait before it continues (a usage limit can take hours), or a paused workflow, whose driver may not have seen the wait yet. This covers `run_agent`, parallel runs, continuations, resumes and nested runs from their start, and a `spawn_agent` handle from `wait` on. The setting belongs to the run and isn't saved. The task keeps its timeout, which applies when it is run outside a workflow. An agent driven through `into_parts` is timed as configured, since the caller answers its waits. The caller can send `UserCommand::HostDeciding` for a wait it takes long to decide.

#### Pauses

A paused agent's task keeps its awaiting state (`TaskState::AwaitingToolApproval`, `AwaitingBatchToolApproval`, or `AwaitingErrorRecoveryApproval`), is recorded with status `AgentRunStatus::Paused` (`error` describes the pause), and is saved with the checkpoint, so it can be resumed after a restart. `OnAgentPaused` fires once per pause, and `OnWorkflowFailed` doesn't fire. A completion its verifier could not review pauses the task the same way (the call returns `VerifierFailed`). `AgentPause::of(&task.current_state)` reads the pause from a task.

### Conversation handover

An agent whose `AgentConfig::handover` is set (see [Conversation handover](../docs/arko_agents-guide.md#conversation-handover) in the arko_agents guide) writes a report once its context reaches the threshold and continues from it in a fresh conversation, in the same task and run. Its task ID, name, status, budget, and the messages sent to it are unaffected, and no one sees it stop. A workflow needs no setting of its own. Opt in per agent:

```rust
let mut lead = lead_config();
lead.handover = Some(arko_agents::HandoverConfig::at_context_fraction(0.4));
let workflow = Workflow::builder().agent("lead", lead) /* ... */ .build()?;
```

The engine gives every run a `HandoverHook`. When the conversation is replaced, it:

- records the queued messages the old conversation held as read by the run (`MessageBus::hand_over`, `AgentMessage::read_by`), so they aren't presented again, and are acknowledged when the run completes as if the conversation still held them. The report is asked to cover them, including the ones it hasn't answered and what it meant to reply, and anything the agent is still waiting for, such as a reply to a question it sent. A message that arrives once the report is requested isn't added to the old conversation. It is delivered, in full, in the fresh one.
- forgets the run's reads in the freshness tracker (with context freshness on), since the fresh conversation holds none of their content, and files it reads again are tracked again.
- starts the fresh conversation from the run's prompt without the messages it delivered and without its recipient list, then the report, then a current `# Agents you can message` list (when the prompt had one), which also counts as the list the run has seen for start and finish notices. A chat (`AgentToolConfig::chat`) starts instead from the user's messages since the model's last reply (`HandoverSeed::for_chat`), which the fresh conversation answers, with their images.

`Task::initial_task_prompt` is kept, so a completion verifier still reviews against the original task. `ConversationHandedOver` is written to the event journal at once. A caller can also ask a running agent to hand over by sending `arko_agents::UserCommand::RequestHandover` on the command channel of an `AgentHandle::into_parts` run. `WorkflowBuilder::agent_handover_threshold(agent, threshold)` gives an agent's runs an `arko_agents::HandoverThreshold` in place of the configured threshold. The host keeps a clone and changes the size with it at any time, larger or smaller, also while the runs run. It isn't saved with the checkpoint, so a host that resumes the workflow gives it again.

### Claude Code agents

An agent whose model is a Claude Code model (`Model::claude_code(...)`) runs in the installed Claude Code CLI, under its login. See [Claude Code models](../docs/arko_agents-guide.md#claude-code-models) in the arko_agents guide. A workflow drives it like any agent. Its workflow tools (`send_message`, `list_recipients`, `wait_for_message`, the memory tools, `run_workflow_agent`, `cast_vote`) are served to Claude Code with its other arko tools, and the workflow's approvals, error recovery, middleware, context updates, start and finish notices, messages, budgets and handovers apply. Its Claude Code built-in tools (listed in `available_tool_names`) pass the workflow's approval policy and execution middleware, including the `PreToolExecution` and `PostToolExecution` hooks and freshness tracking. They can't be distilled. Its calls are billed as a subscription (nothing per call), so they count toward the token totals but not the budget's spend. A turn that fails at a Claude plan limit pauses with `ErrorRecoveryCause::UsageLimit` and the limit's reset, which `ErrorRecoveryPolicy::Automatic` waits out as for other agents. The error recovery policy decides its other pauses by their causes as for any agent. Under `Automatic`, a session that can't start (the CLI is missing or exits at once) or can't sign in (Claude Code isn't signed in, or its login expired or was revoked, which it reports on the first turn), can't bill its account (`billing_error`) or isn't allowed to make the request (`oauth_org_not_allowed`), all `NonRetryable` with their `failure_kind` (`Authentication`, `Billing`, `Permission`), or a standing refusal, fails the run. A session that failed otherwise (`RetriesExhausted`) is resumed after the policy's wait. A tool error under `PauseAndAwaitApproval` is continued. And a completion its verifier couldn't review returns `VerifierFailed`. Its refused input can't be changed, so `ResolveRefusal` is refused. `examples/claude_code_team.rs` runs a lead and a worker in Claude Code that message each other while the lead hands its conversation over.

### AgentHandle

`AgentHandle` has no public constructor or fields. `spawn_agent` returns it.

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

`agent_name` returns the registration name. `wait` consumes the handle, runs the managed driver, finalizes and persists the task, and returns the completed result or the same errors as `run_agent`. Messages delivered in the agent's prompt are acknowledged when it completes, and the agent is unregistered from the message bus whatever the outcome. `into_parts` consumes the handle and returns the event receiver, command sender, task join handle, and agent name. A `wait_for_message` call such an agent is running ends at a message, a start or finish, its time limit, a workflow stop, or when the task is aborted, but not at a `CancelTask` the caller sends, since the runtime reads commands only after the call returns. With the raw parts, the caller is responsible for event handling, approvals, usage accounting, event broadcasting and persistence, result recording, and message acknowledgment and unregistration (through `context().messages`). Execution middleware and interceptors are installed at spawn, so tool hooks, freshness checks, verification, and interception still run. A stopped run still reaches the agent. It is sent `CancelTask`, and the middleware cancels it at its next tool call. The caller must keep receiving its events until it ends. The workflow cannot observe the end of an agent driven through `into_parts`, so it counts as running for the rest of the run.

Dropping a handle without calling `wait` or `into_parts` aborts the agent's runtime task, unregisters it, and marks it finished (cancelled). No result is recorded.

### Results, state, costs, events, and pause

| Signature | Purpose and behavior |
| --- | --- |
| `context(&self) -> &Arc<SharedContext>` | Shared stores, recorded results, and extension state. See [Shared context](#shared-context). |
| `result(&self) -> WorkflowResult` | Snapshot of recorded results, metadata, the cost ledger, and the run's `cancellation`. Does not finish or checkpoint. Also available once the run is stopped. |
| `agent_status(&self, agent_name: &str) -> Option<AgentStatus>` | A registered agent's status in this run, as `list_recipients` reports it. `None` for an unknown name. See [Agent statuses](#agent-statuses). |
| `cost_ledger(&self) -> CostLedger` | Clone of the chronological ledger of automatic and manual costs. |
| `cost_totals_by_agent(&self) -> BTreeMap<String, CostTotals>` | Each agent's running totals, kept as costs are added (rebuilt from the ledger on resume), without copying the ledger, as `CostTotals { billed, list_price, input_tokens, output_tokens, calls }`, where `list_price` counts the billed amount of metered calls and the list-price estimate of those a plan covered. Manual costs belong to no agent. |
| `set_budget(&self, limit: Option<f64>) -> Result<(), WorkflowError>` | Change the workflow's limit (`None` for none) while agents run. It applies from their next model requests. Raising it releases the agents held on it. Waiting runs continue at once, and a run returned as `AgentPaused` continues with `resume_agent`. Lowering it holds agents once spend is over it. A non-finite or negative limit returns `BuilderInvalid`. The limit is saved with the next checkpoint (with the task journal, a live checkpoint is written at once while agents run). See [Budget holds](#budget-holds) for resuming. `OnBudgetExceeded` fires again when spend goes over the new limit. |
| `set_agent_budget(&self, agent: &str, limit: Option<f64>) -> Result<(), WorkflowError>` | The same for an agent's own limit. An unknown agent returns `AgentNotFound`. |
| `budget(&self) -> Option<f64>` | The workflow's current limit. |
| `set_plan_budget(&self, budget: PlanBudget) -> Result<(), WorkflowError>` | Change the plan budget's limits while agents run, as `set_budget` does for the dollar limit. Raising a window's limit releases the agents held on it, and the limits are saved with the next checkpoint. A non-finite or negative limit returns `BuilderInvalid`. See [Plan budgets](#plan-budgets). |
| `plan_budget(&self) -> PlanBudget` | The plan budget's current limits. |
| `plan_usage_spent(&self) -> PlanUsageSpent` | The points of each window used so far (`five_hour`, `seven_day`, and `None` for a window no report has shown). With several plans it is the most any plan has used of the window. |
| `plan_usage_spent_of(&self, plan: arko_ai::SubscriptionPlan) -> PlanUsageSpent` | The points of each of `plan`'s windows used so far. `None` for a window no report of the plan has shown. |
| `is_budget_blocked(&self) -> bool` | Under `Pause`, true when some run waits for a raise and every other running run does too, waits for a run it delegated to, or waits for a message, so nothing runs until the budget is raised. |
| `async budget_blocked(&self)` | Waits until `is_budget_blocked` holds (never, while agents make progress). A host that can't raise the budget live stops there (see [Budget holds](#budget-holds)). |
| `add_manual_cost(&self, amount: f64, description: &str, metadata: Option<String>) -> Result<AddCostResult, WorkflowError>` | Record a finite, nonnegative dollar cost with a description and optional free-form metadata. Fires `OnBudgetWarning` and `OnBudgetExceeded` as applicable. A hook `Abort` stops the run and returns the stop's error. Returns `BudgetExceeded` when spending exceeds the limit. The cost stays recorded. An invalid amount returns `InvalidCost` and records nothing. |
| `subscribe_events(&self) -> tokio::sync::broadcast::Receiver<WorkflowEvent>` | Receive events that managed drivers emit after subscription, without replay. The buffer holds 8,192 events, and a receiver that lags behind it misses the oldest (`RecvError::Lagged`). |
| `subscribe_events_lossless(&self) -> EventReceiver` | Receive every event emitted after subscription, in order, however far the receiver lags. Each receiver has its own queue, and while it lags, the streamed chunks of one run's text (or reasoning) wait joined into one event, so the text arrives whole in fewer pieces. `EventReceiver::recv().await` returns `None` once the workflow is dropped and every event was received, and `try_recv`, `len` and `is_empty` look without waiting. For a host that shows streamed text or needs every lifecycle event. |
| `post_note(&self, note: Note) -> Result<PostedNote, WorkflowError>`, `notes(&self) -> Vec<PostedNote>`, `note_image(&self, image: &NoteImage) -> Result<arko_ai::Image, WorkflowError>`, `notes_handle(&self) -> NotesHandle` | Post a note to the agents while the workflow runs, list the notes posted, read a posted note's image, or get a handle that posts from any task. See [Operator notes](#operator-notes). |
| `async checkpoint(&self) -> Result<(), WorkflowError>` | Save a checkpoint. Succeeds without writing when persistence is not configured. |
| `async finish(&self) -> Result<WorkflowResult, WorkflowError>` | Fire `PostWorkflow`, save a checkpoint, and return the result snapshot. Does not wait for outstanding agents or prevent later runs. Each call repeats the hook and checkpoint. On a stopped run, saves a checkpoint without firing `PostWorkflow` and returns the stop's error. |
| `pause(&self) -> Result<(), WorkflowError>` | Set the pause flag, firing `OnWorkflowPause` only when the flag changes. Managed drivers stop before processing their next event. Model and tool work already in progress continues. A pause does not delay stopping the run. |
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

The resource methods require freshness tracking and return `BuilderInvalid` otherwise. `actor` is the observing or writing actor. `resource` identifies the context. `register_resource_read` records the actor's observation of the current version. `mark_resource_dirty` increments the version, returns the actors whose reads became stale, queues a notice for each active stale reader, and fires `OnFreshnessConflict` once per stale reader. A hook `Abort` stops the run and returns the stop's error. The version change is recorded before any notification or hook error is returned. See [Resource tracker](#resource-tracker) for identity and overlap rules.

The ballot methods require `enable_voting` and return `BuilderInvalid` otherwise. `begin_ballot` validates and stores the configuration and returns the new ballot's UUID. Its eligible voters must be among the workflow's voters (`Deliberation` otherwise), and an empty set becomes all of them. `finish_ballot` tallies the current votes without closing the ballot. Validation and tally errors are returned as `WorkflowError::Deliberation`.

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

`spec` names and versions the operation, `input` establishes its persisted input identity, and `execute` performs one attempt with the supplied context. A completed step returns its recorded `O` without executing again. `resolver` reconciles an uncertain earlier outcome where the retry policy requires it. `in_doubt_steps` lists uncertain attempts. `resolve_in_doubt_completed`, `resolve_in_doubt_failed`, and `authorize_in_doubt_retry` record an externally established output, a permanent failure, or permission for the next call to retry. `actor` and `reason` are audit strings, not authorization checks.

Once the run is stopped, `durable_step` and `durable_step_with_resolver` return the stop's error without journaling anything. A step already executing when the run stops is journaled as any attempt is. When its closure fails because of the stop (for example, its `run_agent` returned `Cancelled`), the attempt is recorded in doubt with the closure's error as the reason, and when the closure still succeeds, the step is recorded `Completed`. Either way the call returns the stop's error, not `DurableInDoubt` or the output. After `resume_from`, the step replays under its retry policy. `RetrySafe` runs it again, and otherwise it is resolved through a resolver or the operator methods above, which remain available on a stopped run.

### Stopping a run

Defined in [workflow/cancel.rs](src/workflow/cancel.rs). A `Workflow` is one run. The run is stopped when a `WorkflowCancelHandle` cancels or stops it (`stop`, after its wrap-up if it has one, see [Graceful stop](#graceful-stop)) or when any hook returns `HookResult::Abort` (see [Hook registration and results](#hook-registration-and-results)). The first cause stands, and a stopped run stays stopped. A hook's `Abort` on a run that is already stopped returns the first cause's error wherever it surfaces. During a graceful stop's wrap-up, `cancel` or a hook's `Abort` ends the wrap-up at once, and the stop's cause stands.

```rust
// Workflow
fn cancel_handle(&self) -> WorkflowCancelHandle;
fn cancel(&self) -> bool; // cancel_handle().cancel()
fn stop(&self, request: StopRequest) -> bool; // cancel_handle().stop(request)
fn is_stopping(&self) -> bool; // a wrap-up is under way
fn is_cancelled(&self) -> bool;
fn cancellation(&self) -> Option<Cancellation>;

// WorkflowCancelHandle: Clone, Debug, Send, Sync
fn cancel(&self) -> bool;
fn stop(&self, request: StopRequest) -> bool;
fn is_stopping(&self) -> bool;
fn is_cancelled(&self) -> bool;
fn cancellation(&self) -> Option<Cancellation>;
async fn cancelled(&self);

// workflow::cancel
pub const DEFAULT_CANCEL_GRACE_PERIOD: std::time::Duration; // 30 seconds
```

`cancel` stops the run with `CancelCause::Requested` and fires `OnWorkflowCancelled` once, from the calling task, ignoring its result. It returns `false` and changes nothing when the run is already stopped, whether cancelled or aborted by a hook. `is_cancelled` is `true` once the run is stopped for either cause, `cancellation` returns the cause and time, and `cancelled` resolves once the run is stopped. A handle can be cloned and used from any task or thread. Clones control the same run. It holds only a weak reference to the workflow's shared context.

| Type | Fields or variants and methods |
| --- | --- |
| `CancelCause` | `Requested` (a handle cancelled the run), `AbortedByHook { hook_point: String, reason: String }`, where `hook_point` is the `HookPoint` variant name, or `Stopped { kind: StopKind, reason: String }` (a [graceful stop](#graceful-stop)). `fn to_error(&self) -> WorkflowError` returns `Cancelled` (for `Requested` and `Stopped`) or `AbortedByHook` with the same fields. `fn agent_reason(&self) -> String` returns `Workflow cancelled.`, `Workflow aborted by {hook_point} hook: {reason}`, or `Workflow stopped ({kind}): {reason}.` (`Workflow stopped ({kind}).` with an empty reason). Implements `Clone`, `PartialEq`, `Eq`, and Serde. |
| `Cancellation` | `cause: CancelCause` and `timestamp: chrono::DateTime<chrono::Utc>`, the time the run stopped. Implements `Clone`, `PartialEq`, and Serde. Recorded in `WorkflowResult::cancellation` and `WorkflowState::cancellation`. A `Stopped` cause is serialized as `"cause": "Requested"` with `"stop": {"kind", "reason"}` beside it, which versions from before graceful stops ignore (they read a plain cancellation). This version reads it back as `Stopped`. |

Stopping the run has these effects:

- Every running agent of the run is sent `CancelTask` at once, queued before its next step. This includes agents started with `spawn_agent`, whether driven by `AgentHandle::wait` or through `into_parts` (see [AgentHandle](#agenthandle)).
- Every run call in progress (`run_agent`, `continue_agent`, `run_agents_parallel`, `AgentHandle::wait`) waits for its agents to end while recording their remaining events (event journal and broadcast) and usage (budget), then returns the stop's error, `Cancelled` or `AbortedByHook`. It does so even when an agent completed during the wind-down. That agent's `Completed` result is still recorded. A `durable_step` or `durable_step_with_resolver` call in progress returns the stop's error once its closure returns. Every later call returns the error at once without starting anything, with no `PreAgentSpawn` and no journal record.
- An agent stops at its next step. Model streams and waits are interrupted at once, `wait_for_message` returns at once, and an agent running another long tool call stops when the call returns. The grace period counts from the moment the run stopped. An agent that hasn't ended within it is aborted. Its Tokio task is dropped with any tool call in progress (an `arko_agents` command-template tool's child process is killed), and it is recorded as `Cancelled` with its task as the task journal last saved it (see [Task journal and recovery](#task-journal-and-recovery), and without it, the task as the run started it, so the record lacks this invocation's conversation, which the event journal has). `reopen_agent` continues it. Its reason is the stop reason described below followed by `The agent did not stop within <grace period> and was aborted.`, with the configured period in `Duration` debug form (for example `30s`). The same bounded wait applies whenever the managed driver stops an agent, for example over budget, after an interruption, or after a hook's abort. When the run itself is not stopped, it counts from that moment. A grace period too long to represent waits as long as possible.
- An agent running inside another agent's tool call, such as a completion verifier or an agent delegated through the `run_workflow_agent` tool, gets half its parent's grace period at each level of nesting, so when the run stops it is stopped and recorded before its parent can be aborted. When only the parent is stopped (for example over its budget) and it doesn't end within the grace period, it is aborted with the agent running inside its tool call, which is not recorded. Dropping the driver of a running agent, for example a `run_agent` or `run_agents_parallel` future, aborts the agent's task instead of leaving it running unsupervised, and the agent counts as finished (cancelled). No result is recorded.
- A paused workflow does not delay the stop, and a `UserInputHandler` still waiting for an answer is abandoned.
- Once the run is stopped, the execution middleware answers every tool call with `ToolExecutionDecision::CancelTask`. The call doesn't run (a `cast_vote` is not recorded, for example), and the agent ends `Cancelled` with the cause's agent reason. A completion verifier stopped with the run decides nothing, so its target agent is cancelled, not failed. Distillers make no further requests and leave tool output unchanged.
- An agent the managed driver stops records why in its `TaskState::Cancelled` reason (and `AgentResult::error`), in place of the runtime's generic `Task cancelled by user.` (`arko_agents::task::CANCELLED_BY_USER`), with the cause's agent reason when the run is stopped, and otherwise `Stopped by the workflow: <error>`, with the error the driver stopped it for (for example `BudgetExceeded`). The cancellation's `TaskStateChanged` event, as recorded in the event journal and broadcast during the wind-down, carries the same reason. An agent that ends because of the stop has status `AgentRunStatus::Cancelled`, and `PostAgentCancellation` fires for it.
- Budgets and limits are unchanged. Usage reported while agents wind down is charged.
- Votes cast before the stop stand, and `finish_ballot` works as usual, although the quorum may be unmet.
- `finish` saves a checkpoint and returns the stop's error without firing `PostWorkflow`. `result` still returns the recorded results, with `cancellation` set. `OnWorkflowFailed` does not fire for `Cancelled`.

With persistence, every checkpoint written after the stop records it in `WorkflowState::cancellation`. With `auto_checkpoint`, one is saved as each stopped agent's result is finalized and once more when the first run call returns the stop's error. `checkpoint` and `finish` save one regardless of `auto_checkpoint`. The pause flag is not persisted. A stopped run's checkpoint is recognized by its `cancellation` field and the `Cancelled` status of its stopped agents.

Resuming with `resume_from` from a checkpoint that records a cancellation starts the run again. The new workflow isn't stopped, and its checkpoints do not carry the earlier record. To keep a cancelled run stopped, check `WorkflowState::cancellation`, for example with `PersistenceManager::load_checkpoint`, before resuming. Durable steps left in doubt by the stop are handled as described under [Workflow-facing durable operations](#workflow-facing-durable-operations).

### Graceful stop

Defined in [workflow/stop.rs](src/workflow/stop.rs). `stop` stops the run for a stated reason and, optionally, lets running agents finish first. It is built on the cancellation above. After the optional wrap-up, the run is cancelled with `CancelCause::Stopped`, its agents get the grace period, and the checkpoint records the stop.

```rust
// workflow::stop (re-exported at the root and in the prelude)
#[non_exhaustive] enum StopKind { Operator, Drain, Limit, Failure } // Serde: "operator", "drain", ...
struct StopRequest { pub kind: StopKind, pub reason: String, pub wrap_up: Option<WrapUp> }
struct WrapUp { pub within: Duration, pub message: Option<String> }
struct StopRecord { pub kind: StopKind, pub reason: String, pub timestamp: DateTime<Utc>,
                   pub intent: Option<RunStopIntent> } // Serde. intent: see Stopping one agent run
struct StoppedRun { pub result: AgentResult, pub restart: Restart }
enum Restart { Recover, Reopen, Continue }

impl StopRequest {
    fn new(kind: StopKind, reason: impl Into<String>) -> Self; // no wrap-up
    fn operator(reason) -> Self; fn drain(reason) -> Self; fn limit(reason) -> Self; fn failure(reason) -> Self;
    fn wrap_up(self, within: Duration) -> Self;
    fn wrap_up_message(self, message: impl Into<String>) -> Self; // after wrap_up; replaces the notice
}
fn wrap_up_notice(reason: &str, within: Duration) -> String; // the default notice
fn stop_on_signals(handle: WorkflowCancelHandle, request: StopRequest)
    -> std::io::Result<tokio::task::JoinHandle<()>>;

// Workflow
fn stopped_runs(&self) -> Vec<StoppedRun>;
async fn restart(&self, run: &StoppedRun, note: &str) -> Result<AgentResult, WorkflowError>;
async fn restart_with(&self, run: &StoppedRun, note: &str, options: ContinueOptions) -> Result<AgentResult, WorkflowError>;
```

`StopKind` says why, for the host. It is `Operator` (Ctrl-C, `SIGTERM`, a person), `Drain` (the process is going away and the run will be resumed, as in a redeploy), `Limit` (a deadline, budget or count of the host's own), `Failure` (something outside the work failed, such as a lost login). Every kind stops the run the same way.

Without a wrap-up (the default, `StopRequest::new` and its shorthands), `stop` cancels the run at once, as `cancel` does, with `CancelCause::Stopped`. With `wrap_up(within)`:

1. No agent run starts. `run_agent`, `continue_agent`, `resume_agent`, `recover_agent`, `reopen_agent`, `run_agents_parallel` (including agents waiting for a place), `spawn_agent` and `restart` return `Cancelled` at once, and an agent's `run_workflow_agent` call fails with that error (`Workflow cancelled`). Only a completion verifier's review may still start, of a completion an agent made before it was told to wrap up (see step 3). An agent that is also a completion verifier, run through `run_workflow_agent`, is refused like any other. Durable steps still run.
2. The stop signal of every running agent's tool calls fires at once (`arko_agents::TaskContext::stop`, through `AgentRuntime::with_stop_signal`), with the reason `Workflow stopped ({kind}): {reason}.`. A tool that watches it saves its work and returns, an `arko_agents::StopAwareCommand` creates its stop file, and `wait_for_message` returns `Workflow stopping.`. A tool call started later in the wrap-up sees the signal already fired.
3. Each running agent, other than a completion verifier's review, gets the notice once, at its next turn boundary (before its next model request), as a workflow context update. [operator notes](#operator-notes) due at that boundary follow it in the same update, and their rule tells the agent that a note doesn't override a request to stop. The default notice (`wrap_up_notice`), for `StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(90))`:

   > The workflow is stopping: deploying a fix. You have about 90 seconds before you are stopped. Save your progress where your instructions say to keep it, and don't start new long work. Then call complete_task with a summary of where you stopped and what is left, so the work can be picked up from there.

   An agent in a long tool call reads it once the call returns. One waiting on a model response reads it with its next request. A `complete_task` call made after the notice reports progress, so completion verifiers don't review it (they would judge a partial summary against the whole task, and could fail the run), and the run is marked `stopped_by`. The run it is continued in after a restart is reviewed as usual. A run started by `run_workflow_agent` returns its progress report to its caller followed by a note naming the run (`Agent run <id> of agent '<agent>' stopped with the workflow before it finished: ...`), so that the restarted caller can resume it with `resume_run` (`stopped_runs` leaves it to its caller). A completion made before the agent was told (its model answered after the stop began) is reviewed as usual, during the wrap-up.
4. The wrap-up ends when no agent of the run is running any more (the last run's call then returns `Cancelled`), when its time is up, or at once on `cancel`, a stop without a wrap-up, or a hook's `Abort`. Then the run is stopped with the first stop's cause, as described above. The agents still running are cancelled (`Workflow stopped (drain): deploying a fix.`) within the grace period, and the stop is checkpointed. Agents run through `AgentHandle::into_parts` are never seen to end, so with one of them running the wrap-up lasts its full time.

During the wrap-up `is_stopping` is `true` and `is_cancelled` still `false`. A run call whose agent completed in the wrap-up returns its result. `finish` waits for the run to stop and returns `Cancelled`. `stop` returns `false`, changing nothing, when the run is stopped already, or when it is wrapping up and the new request has a wrap-up too. `OnWorkflowCancelled` fires once, when the run is stopped, from the task that stopped it (the wrap-up's timer, or the driver of the last agent to end), with metadata `stop_kind` (`operator`, `drain`, `limit`, `failure`) and `reason`. Timing a wrap-up needs a Tokio runtime, the current one or the one the handle was made in. Without either, the run is stopped at once.

#### Blameless results

`AgentResult::stopped_by()` returns the `StopRecord` of the stop that ended a run, such as a run cancelled once the run was stopped by `stop` or `cancel` (status `Cancelled`, including one aborted after the grace period, and a `cancel` is recorded as `StopKind::Operator` with the reason `Workflow cancelled.`), or a run that completed after it was given the wrap-up notice (status `Completed`, unreviewed). It is `None` for runs that ended otherwise, for a hook's abort, and in results recorded by an older version. A stopped run's outcome says nothing about the agent's work, so don't count it as a failure (or a finished success).

#### Restarting after a resume

After `resume_from` on the stopped run's checkpoint, `stopped_runs` lists the runs to relaunch, in the order they were recorded. They are runs a stop cancelled (`Restart::Reopen`), runs started by the host that completed in a wrap-up (`Restart::Continue`), and the `interrupted_runs` (`Restart::Recover`, where the process ended while they ran, for example when a service manager killed it mid-stop). Nested runs are listed only when orphaned, as for `interrupted_runs`, and not when the run that started them is listed too, because restarted, that run resumes a run inside a call it was running, and a call that failed when the stop cancelled its run named that run (`Agent run <id> of agent '<agent>' was stopped with the workflow and did not finish. You can resume it instead of starting over: ...`), so that it resumes it with `run_workflow_agent`'s `resume_run` instead of starting the work again. Runs running now and superseded runs are not listed, and neither are runs paused for the caller (`WaitingForInput`, `Paused`), which a stop doesn't end, so continue them as usual. `restart(run, note)` calls `recover_agent` (without a message, and `note` isn't used), `reopen_agent(result, note)`, or `continue_agent` with this message, then `note`:

> The workflow was stopped (deploying a fix) while you worked, so you finished early, and it has now restarted. Continue your task from where you stopped, and call complete_task when it is done.

`restart_with` sends `ContinueOptions::images` with the note of a reopened or continued run (through `reopen_agent_with` and `continue_agent_with`), and returns `BuilderInvalid` for images to a recovered run, which takes no message, so send them with the message after it. A restarted run's new result replaces the old one (without `stopped_by`), so it is listed no more. A run the host retired with `stop_run` is never listed, also when a workflow stop ended it in its wrap-up (see [Stopping one agent run](#stopping-one-agent-run)). Results recorded by versions from before graceful stops carry no `stopped_by`, so continue their cancelled runs with `reopen_agent`. A changed configuration of an agent with a run `stopped_runs` would relaunch (or one a stop cancelled inside another run's call) refuses the resume, as for any resumable run (see [Build validation and tool composition](#build-validation-and-tool-composition)). A cancelled run without `stopped_by` is not checked, and reopened it keeps the configuration saved in its task. What may change across a redeploy is listed in the [workflow design guide](../docs/guides/workflow-design.md#make-side-effects-durable).

#### Signals

`stop_on_signals(workflow.cancel_handle(), request)` installs handlers for Ctrl-C and `SIGTERM` (Ctrl-C and Ctrl-Break on Windows) before it returns, and spawns a task that stops the run with `request` at the first signal, ends a wrap-up at once at the second (`stop` without a wrap-up), and at the third kills the children of running `arko_agents::StopAwareCommand`s (on Unix, with `arko_agents::kill_stop_aware_commands`, because exiting runs no destructors, and each child is in its own process group, out of the terminal's reach) and exits the process with status 128 + the signal's number. Call it within a Tokio runtime. The handlers replace the signals' default action for the rest of the process, and aborting the returned task leaves them ignored. Under systemd, give the unit `KillMode=mixed` and a `TimeoutStopSec` longer than the wrap-up plus the grace period plus the longest tool stop grace.

```rust
let workflow = Arc::new(builder.build()?);
let _signals = stop_on_signals(
    workflow.cancel_handle(),
    StopRequest::drain("stopped by a signal").wrap_up(Duration::from_secs(90)),
)?;
match workflow.run_agent("worker", "Process the next batch.").await {
    Ok(result) => { /* done */ }
    Err(WorkflowError::Cancelled) => {
        // Stopped: the runs the stop ended are marked, and are not failures.
        let stopped = workflow.result().agent_results.into_iter()
            .filter(|result| result.stopped_by().is_some());
        /* report them, exit, and resume later */
    }
    Err(error) => return Err(error),
}

// The next process, after the redeploy:
let resumed = builder.resume_from(&checkpoint).build()?;
for run in resumed.stopped_runs() {
    resumed.restart(&run, "").await?;
}
```

With one agent the call can't return a result the stop marked, because the last agent to end ends the wrap-up, so its call returns `Cancelled`. With several agents running at once, check `stopped_by()` on `Ok` results too, since an agent that completed in the wrap-up while others still ran returns its result, finished early.

### Stopping one agent run

Defined in [workflow/cancel.rs](src/workflow/cancel.rs). `stop_run` stops one agent run the way a graceful stop stops the whole workflow, while the rest of the workflow runs on. The other runs carry on, and new runs start as usual. It is how a long run retires an agent, or replaces it with another model. The old run saves its work and reports where it stands, and then the host starts the replacement.

```rust
// Workflow, and WorkflowCancelHandle
fn stop_run(&self, agent_run_id: Uuid, request: StopRequest, intent: RunStopIntent) -> bool;
fn stop_agent(&self, agent_name: &str, request: StopRequest, intent: RunStopIntent)
    -> Result<Vec<Uuid>, WorkflowError>; // the runs whose stop began

// workflow::stop (re-exported at the root and in the prelude)
enum RunStopIntent { Retire, Relaunch } // Serde: "retire", "relaunch". as_str, Display
impl StopRecord { fn is_retired(&self) -> bool; } // intent == Some(Retire)
fn run_wrap_up_notice(reason: &str, within: Duration) -> String; // the default notice

// WorkflowError
AgentStopped { agent: String, task_id: Uuid, reason: String }
```

`request` is a `StopRequest`, as for `stop`. Its kind and reason are recorded, and its optional wrap-up gives the run time to finish.

1. With `wrap_up(within)`, the stop signal of the run's tool calls fires at once, with the reason `Run stopped ({kind}): {reason}.`. A `StopAwareCommand` creates its stop file, and `wait_for_message` returns `Run stopping.`. At its next turn boundary the run gets the notice (`run_wrap_up_notice`, or the request's `wrap_up_message`), here for `StopRequest::operator("moving it to Opus 5.5").wrap_up(Duration::from_secs(600))`:

   > Your run is being stopped: moving it to Opus 5.5. You have about 600 seconds before you are stopped. Save your progress where your instructions say to keep it, and don't start new long work. Then call complete_task with a summary of where you stopped and what is left, so the work can be picked up from there.

   A `complete_task` call made after the notice reports progress. Completion verifiers don't review it, and the run call returns the result, with `stopped_by` set. Its summary is the report a replacement starts from.
2. When `within` passes before the run ends, or at once without a wrap-up, the run is cancelled and wound down within the grace period, as the agents of a stopped workflow are. Its run call returns `AgentStopped`, and its result (`agent_result(task_id)`) is `Cancelled`, with the reason `Run stopped ({kind}): {reason}.` and `stopped_by` set. Such a run leaves no report. A replacement starts from what it saved, and the host can read its conversation in `result.task`. A run that completed just as it was cancelled returns its result instead. A recovered run still waiting out the pause it was saved in is stopped at once too.
3. Neither outcome is a failure. `OnWorkflowFailed` doesn't fire, `fail_fast` doesn't stop the other agents, and `run_agents_parallel` returns the stopped run's result in its place. `PostAgentCompletion` or `PostAgentCancellation` fires as usual.

The runs inside the run's tool calls (agents it delegated to with `run_workflow_agent`, at any depth) are stopped with it, with the same request and intent, and its `run_workflow_agent` calls are refused while it is being stopped. A run that starts inside its tool call all the same (a call already under way) takes over its stop, and every one of them is cancelled when the run is. A completion verifier's review is never stopped alone, so stop the run under review.

`stop_run` returns `false`, changing nothing, when the run isn't running, is a review, or the workflow is stopped, and when the run is being stopped already and `request` has a wrap-up too. A request without a wrap-up during the run's wrap-up cancels it at once, and the first stop's record stands. A workflow stop during the run's wrap-up gives the run the workflow's notice as well, and the run's own record stands on its result. `stop_agent` stops each running run of the agent this way, and returns `AgentNotFound` for an unknown agent.

#### Retire or relaunch

`intent` is saved on the run's `StopRecord` and says what the host means to do with the run:

- `Retire` means the run is done for good, because the agent is removed or another run replaces it. `stopped_runs` and `interrupted_runs` never list it, `restart` returns `BuilderInvalid` for it, and a delegating agent can't resume it (`resume_run` is refused). `continue_agent`, `reopen_agent` and `recover_agent` still take it, for a host that changes its mind.
- `Relaunch` means the host wants the run back. `stopped_runs` lists it as soon as its result is recorded, while the workflow runs and after a resume, and `restart` reopens or continues it, also while the old launch's driver is still finishing. A run continued this way is told `Your run was stopped (<reason>) while you worked, so you finished early, and it has now been restarted.`, then to carry on, then the restart's note.

With persistence and the task journal, the live checkpoint is written at once with the stop on the run's record (`AgentRunRecord::stopped_by`). A run the process ended in its wrap-up resumes `Interrupted` with `stopped_by` set, so `stopped_runs` lists it for `Restart::Recover` when it is to be relaunched, and leaves it out when it was retired. A run interrupted in its wrap-up while the process went on is marked the same way.

#### Delegated runs

A delegated run belongs to its caller. Stopped alone, it returns its progress report to its caller, followed by a note naming it, and the caller runs on:

> Agent run <id> of agent 'coder' was stopped by the workflow's host (pausing the coder) before it finished: its summary is a progress report that no completion verifier reviewed. You can resume it instead of starting over: call run_workflow_agent with agent "coder", resume_run "<id>" and a prompt telling it what happened and what to do next. It continues its own task.

A delegated run cancelled by its stop fails the caller's call with the `AgentStopped` message and a note that it `was stopped by the workflow's host (<reason>) and did not finish`. For a retired run the note ends `The host retired it, so it can't be resumed.` A retired run the process ended in its wrap-up stays stopped when its caller is recovered, and the call it ran in fails with the same note. `stopped_runs` leaves a delegated run to its caller while the caller can resume it, as after a workflow stop.

#### Replacing an agent

Register the new configuration under its own agent name (a saved run resumes only with the configuration it started with), stop the old run with a wrap-up and `Retire`, and start the new one from the old one's report:

```rust
// `worker` is the task running `workflow.run_agent("worker-7", &task)`.
let request = StopRequest::operator("moving it to Opus 5.5").wrap_up(Duration::from_secs(600));
workflow.stop_agent("worker-7", request, RunStopIntent::Retire)?;
let report = match worker.await.expect("the worker's task") {
    Ok(result) => result.summary, // stopped_by() is set when the stop ended it
    Err(WorkflowError::AgentStopped { .. }) => String::new(), // cancelled before it reported
    Err(error) => return Err(error),
};
let prompt = format!("{task}\n\nThe previous worker stopped here:\n{report}");
workflow.run_agent("worker-7-opus", &prompt).await?;
```

Another process stops runs through the [control folder](#the-control-folder) (`runctl stop`).

## Result and event types

Defined in [workflow/result.rs](src/workflow/result.rs). These types are re-exported at the root and in the prelude, and implement Serde `Serialize` and `Deserialize`.

| Type | Public fields or variants |
| --- | --- |
| `AgentRunStatus` | `Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`, `Paused`. `is_paused()` is true for the last two, where the run waits for input or a decision and can go on. `Paused` is a task suspended awaiting a tool approval or error-recovery decision the workflow's policies leave to the caller. `Cancelled` is the status of a task that ended in `arko_agents::TaskState::Cancelled`, stopped with the workflow run, stopped by the managed driver (over budget or after an interruption), or cancelled by any other `CancelTask` command or middleware decision. The status follows the task state, never the wording of a failure reason. |
| `AgentResult` | `agent_run_id: Uuid`, the stable task identity. `agent_name: String`, the registration name. `status: AgentRunStatus`. `summary: String`, the completion summary for `Completed` and empty otherwise. `error: Option<String>`, the failure, interruption, or cancellation reason, the unanswered questions, or a description of the pause. `task: arko_agents::Task`, with conversation, usage, and runtime state (a failure at a rejected error-recovery pause records the pause's cause in `TaskState::Failed { cause }`, see `failure_cause`). `stopped_by: Option<StopRecord>`, the stop that ended the run (see `stopped_by()`, omitted from JSON when `None`, and `None` when absent). Adding `stopped_by` broke struct literals of `AgentResult`, and adding `StopRecord::intent` (omitted from JSON when `None`, and `None` when absent) broke struct literals of `StopRecord`. |
| `WorkflowResult` | `workflow_run_id: Uuid`, `workflow_name: String`, `metadata: WorkflowMetadata`, `total_cost: f64` (running workflow total in dollars), `cost_ledger: CostLedger`, `agent_results: Vec<AgentResult>` (finalized outcomes, where a continuation replaces the entry with its run ID), `cancellation: Option<Cancellation>` (set once the run is stopped, omitted from JSON when `None` and `None` when absent). |
| `WorkflowEvent` | `timestamp: chrono::DateTime<chrono::Utc>`, `workflow_run_id: Uuid`, `agent_name: String`, `agent_run_id: Uuid`, `event: arko_agents::runtime::event::AgentEvent`. Wraps each broadcast and persisted runtime event. |

`AgentResult` has these methods:

```rust
fn total_cost(&self) -> f64;
fn total_input_tokens(&self) -> usize;
fn total_output_tokens(&self) -> usize;
fn failure_cause(&self) -> Option<&arko_agents::ErrorRecoveryCause>;
fn failure_kind(&self) -> Option<arko_ai::ErrorKind>;
fn stopped_by(&self) -> Option<&StopRecord>;
```

`stopped_by` is the stop that ended the run, when a stop the host asked for did (see [Graceful stop](#graceful-stop) and [Stopping one agent run](#stopping-one-agent-run)). A run stopped alone carries it also when it was interrupted in its wrap-up.

`failure_cause` is why a `Failed` run failed when it failed at an error-recovery pause the error recovery policy (or the caller, with `AgentResume::FailAfterError`) rejected, as its task recorded it. `NonRetryable` means the model request needs a new credential (an expired or invalidated login) or a configuration change, so a host can start the agent again once, or stop and tell the user to sign in, without parsing the reason or the event log. `None` for runs that didn't fail, failed otherwise, or were saved by an older version.

`failure_kind` is that failure's error kind (`TaskState::Failed { error_kind }`), when the error was classified (see [Error kinds](../docs/arko_ai-guide.md#error-kinds) in the arko_ai guide), such as `Authentication` for a lost login, `Billing` for a credit wall, `ContextOverflow` for a request over the model's input limit, `InvalidRequest`, and so on. Decide by it instead of matching error text, since a lost login fails every run on that credential, while a request too large fails only this run. `None` for runs that didn't fail, failed otherwise or on an unclassified error, or were saved by an older version.

These return task-wide totals, including earlier conversation segments after a continuation. `total_cost` is the billed amount (`UsageInfo::billed()`), which is zero for subscription calls such as ChatGPT OAuth. The workflow total also includes manual and distillation charges, so the sum of task costs need not match the workflow ledger.

## Hooks and human input

Defined in [hooks/mod.rs](src/hooks/mod.rs), [hooks/context.rs](src/hooks/context.rs), [user_input.rs](src/user_input.rs), [runtime integration](src/workflow/engine.rs).

### Hook registration and results

```rust
type HookCallback = Arc<
    dyn Fn(&hooks::context::HookContext) -> HookResult + Send + Sync + 'static
>;
enum HookResult { Continue, Abort(String), Skip, Reject(String) }

// hooks::HookRegistry (also Default)
fn new() -> Self;
fn register(&mut self, point: HookPoint, callback: HookCallback);
fn get(&self, point: &HookPoint) -> Option<&Vec<HookCallback>>;
fn has_hooks(&self, point: &HookPoint) -> bool;
```

Callbacks are synchronous and receive an immutable context. Capture shared state to act on it. `register` appends to a point's list, `get` borrows the ordered list, and `has_hooks` reports whether the list is nonempty. `Continue` proceeds to the next callback. The first `Skip`, `Reject` or `Abort` ends the chain for that point. `Skip` and `Reject` have an effect only at `PreToolExecution`, where they reject the tool call, `Skip` with the message `Skipped by PreToolExecution hook`, `Reject(message)` with `message`, so the agent learns why the call was refused and what to do instead. At other points they only end the chain. Hooks cannot replace tool output. Use a tool result interceptor for that.

`Abort(reason)` at any point stops the whole run with `CancelCause::AbortedByHook { hook_point, reason }`, where `hook_point` is the point's variant name. Running agents are cancelled, and run calls in progress or made later return `AbortedByHook`. On a run that is already stopped, the first cause stands, and its error is returned wherever the abort surfaces. See [Stopping a run](#stopping-a-run). Where the abort surfaces depends on the point:

- At lifecycle points (`PreAgentSpawn`, the `PostAgent*` points, `OnWorkflowPause`, `OnWorkflowResume`, `PostWorkflow`, `OnBudgetWarning` and `OnBudgetExceeded` from `add_manual_cost`, and others), the operation that fired the hook also returns the stop's error. The managed driver ignores the return value of budget hooks fired for automatic usage reports, including usage reported while agents wind down, but an `Abort` from them still stops the run. Budget hooks fired by a distillation charge also turn an `Abort` into an interceptor error.
- At `PreToolExecution`, the call does not run, and the agent ends `Cancelled` with the reason `Workflow aborted by PreToolExecution hook: <reason>`. The run call driving the agent returns `AbortedByHook` with `hook_point` `"PreToolExecution"`. `Skip` still rejects only that call.
- At `PostToolExecution`, and at `OnFreshnessConflict` fired from the execution middleware, the call has run and its result stands. The agent is sent `CancelTask` and ends before its next model request, and the run call returns `AbortedByHook`. The middleware never returns an error for a hook result.
- Hooks fired inside workflow tools (`OnAgentMessage` from `send_message`, `OnFreshnessConflict` from `write_memory`) return a tool error to the agent after the message or write has been made, and the run stops.

The result of `OnWorkflowCancelled` is ignored. Hooks do not roll back work already performed.

### HookPoint variants

| Variant | Trigger and populated context |
| --- | --- |
| `PreWorkflow` | End of a successful `build`, before the workflow is returned. |
| `PostWorkflow` | `finish`, before its checkpoint. |
| `PreAgentSpawn` | Before a run, spawn, or continuation is set up. Not fired once the run is stopped. `agent_name`. For a continuation, also the prior `agent_run_id`. |
| `PostAgentCompletion` | A task finalized as completed. Agent name and run ID. `agent_summary` when nonempty. |
| `PostAgentFailure` | A task finalized as failed. Agent name, run ID, and `error_message`. For a failure at a rejected error-recovery pause, its cause in `metadata["cause"]` (`non_retryable`, `refusal`, `unusable_responses`, …, as `OnAgentPaused` names a pause's cause), and its error kind, when classified, in `metadata["error_kind"]` (`authentication`, `context_overflow`, `billing`, …). |
| `PostAgentCancellation` | A task finalized as cancelled, stopped with the workflow run, stopped by the managed driver (over budget or after an interruption), or cancelled by any other `CancelTask` command or middleware decision. Agent name, run ID, and the task's reason in `error_message`. Cancelled tasks do not fire `PostAgentFailure`. |
| `PostAgentInterruption` | A task finalized as interrupted or in a nonterminal state. Agent name, run ID, and the reason in `error_message`. |
| `OnUserInputRequired` | A task finalized while awaiting user input. Agent name, run ID, and the questions in `error_message`. Not fired when a `UserInputHandler` answers the question. |
| `OnAgentPaused` | A task finalized paused for a decision the approval or error recovery policy leaves to the caller (`AgentRunStatus::Paused`). Fires once per pause. Agent name, run ID, the pause's description in `error_message`, `metadata["pause"]` (`"tool_approval"` or `"error_recovery"`), and `metadata["cause"]` (such as `"retries_exhausted"`, `"monitor"`, with `metadata["error_kind"]` when the error was classified) or `metadata["tools"]` (each call's `tool_call_id`, `tool_name`, and `reason`). |
| `PreToolExecution` | Before a tool call executes. Agent name and run ID, `tool_name`, `tool_call_id`, and `tool_params`. `approval_reason` for a call that awaited approval. `Skip` or `Reject(message)` rejects the call. |
| `PostToolExecution` | After a tool call, before result interception. The same fields, plus `tool_result` and `original_tool_result`, both holding the raw output or error text. |
| `PostToolResultInterception` | The managed driver receives an intercepted result. Agent name and run ID, tool name and call ID, the raw output in `original_tool_result`, and the delivered output in `effective_tool_result` and `tool_result`. |
| `OnWorkflowPause` | The pause flag changes from false to true. |
| `OnWorkflowResume` | The pause flag changes from true to false, before paused drivers are woken. |
| `OnWorkflowFailed` | `run_agent`, `continue_agent`, `resume_agent`, or `run_agents_parallel` is about to return a failure, which is an error other than `Cancelled`, `AgentStopped`, `UserInputRequired`, `AgentPaused`, or `VerifierFailed`, including `AbortedByHook`. Fires at most once per call, also when several parallel agents fail. Not fired by a call refused because the run is already stopped. `error_message`. The agent name (and, for a continuation or resume, run ID) when available. |
| `OnWorkflowCancelled` | A `WorkflowCancelHandle` (or `Workflow::cancel` or `Workflow::stop`) cancelled or stopped the run. Fires once, when the run is stopped (after a graceful stop's wrap-up), from the task that stopped it. Its result is ignored. For a graceful stop its metadata has `stop_kind` and `reason`. A hook's abort does not fire it, unless it ends a graceful stop's wrap-up. |
| `OnBudgetExceeded` | Under `BudgetExceededPolicy::Pause` (the default), the automatic, manual, or distillation cost that takes workflow spending strictly over the limit, once per crossing (again after a raise brings spend within the limit and it is exceeded anew). Under `Cancel`, each such cost recorded while spending exceeds the limit. `budget_spent` is the new total. Also the plan usage report that takes a [plan budget](#plan-budgets) window strictly over its limit, once per crossing under either policy, with `agent_name`, `agent_run_id` and `metadata` `plan` (`"chatgpt"` or `"claude_code"`, or `null` for counting without a plan), `plan_window` (`"five_hour"` or `"seven_day"`), `plan_points_used` and `plan_points_limit`. |
| `OnBudgetWarning` | The first recorded cost that brings spending to or above the warning threshold. `budget_spent` is the new total. |
| `OnAgentMessage` | The `send_message` tool queued a message. Sender name and run ID. `metadata["message_id"]`, `metadata["recipient"]`, and `metadata["content"]`. Direct `MessageBus::send` calls do not fire it. |
| `OnFreshnessConflict` | A successful tracked write (a mapped file tool, `write_memory`, or `mark_resource_dirty`) made another actor's read stale. Fires once per stale reader. The reader's `agent_name` and `agent_run_id`, `metadata["resource"]` (the `ResourceId` in debug form, or `memory:<tier>`), and `metadata["writer"]` (the writer's agent name). A blocked write does not fire it. Its former name, `"OnMesiConflict"`, still deserializes to this variant. |

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
| `approval_reason` | `Option<arko_agents::ApprovalReason>` | Why the call awaited approval, for a call that did, at `PreToolExecution` and `PostToolExecution`. |
| `error_message` | `Option<String>` | Error, reason, or questions. |
| `budget_spent` | `f64` | Workflow dollars spent. |
| `budget_limit` | `Option<f64>` | Configured workflow limit. |
| `metadata` | `HashMap<String, Value>` | Point-specific data, such as `resource` and `writer` for `OnFreshnessConflict`. `message_id`, `recipient`, and `content` for `OnAgentMessage`. `pause` and `cause` or `tools` for `OnAgentPaused`. |

### UserInputHandler and UserInputRequest

```rust
#[async_trait::async_trait]
pub trait UserInputHandler: Send + Sync {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String>;
}
```

`UserInputRequest` has the fields `workflow_run_id: Uuid`, `agent_run_id: Uuid`, `agent_name: String`, and `questions: String`. Pass an `Arc<dyn UserInputHandler>` to `user_input_handler`. The returned string is submitted as the user's response. An `Err` string ends the invocation with `WorkflowError::Channel`.

Without a handler, the managed driver suspends the task on `ask_user` and returns `WorkflowError::UserInputRequired { agent, task_id, questions }`. To answer, take the matching `WaitingForInput` result from `workflow.result().agent_results` and pass it with the answer to `continue_agent`. With persistence, checkpoints include the suspended result and its task, so it can also be continued after `resume_from`.

## Workflow services and custom tools

Defined in [services/mod.rs](src/services/mod.rs), [builder integration](src/workflow/mod.rs).

`WorkflowServicesConfig` (`Debug`, `Clone`, `Default`) configures agent-callable delegation. Build it with `Default` and the setters below (its public fields can be read and assigned afterwards). It has a private field, so a struct literal, `..Default::default()` included, doesn't compile outside the crate:

| Field | Type | Default and meaning |
| --- | --- | --- |
| `allowed_callers` | `HashSet<String>` | The agents offered `run_workflow_agent`. Empty by default, which means none unless `all_callers` is set. |
| `allowed_targets` | `HashSet<String>` | The agents a caller may run, independent of the caller list. Empty by default, which means none unless `all_targets` is set. |
| `all_callers` | `bool` | Opt-in, `false` by default. Every agent is a caller, except completion-verifier agents. |
| `all_targets` | `bool` | Opt-in, `false` by default. A caller may run every agent, except itself and completion-verifier agents. |
| `max_active_calls` | `NonZeroU32` | `DEFAULT_MAX_ACTIVE_CALLS`, 4. Limit on simultaneous service calls, including nested delegation. |
| `unlimited_active_calls` | `bool` | Opt-in, `false` by default. No limit on simultaneous calls (`max_active_calls` is ignored). |
| `queue_active_calls` | `bool` | Opt-in, `false` by default. A call over `max_active_calls` waits until a call ends, instead of failing. |

```rust
// WorkflowServicesConfig
fn allow_caller(self, agent: impl Into<String>) -> Self;
fn allow_target(self, agent: impl Into<String>) -> Self;
fn allow_all_callers(self) -> Self;
fn allow_all_targets(self) -> Self;
fn max_active_calls(self, max: u32) -> Self;
fn unlimited_active_calls(self) -> Self;
fn queue_active_calls(self) -> Self;

// WorkflowServices: Clone
fn is_allowed(&self, caller: &str) -> bool;
fn is_target_allowed(&self, caller: &str, target: &str) -> bool;
async fn run_agent(&self, caller: &str, agent_name: &str, prompt: &str)
    -> Result<AgentResult, WorkflowError>;
```

Callers and targets must be named explicitly (`allow_caller`, `allow_target`), or admitted with the opt-ins, which leave out completion-verifier agents (and, as a target, the caller itself). An explicitly named agent is admitted either way (so a caller also named as a target may run itself). `max_active_calls(0)` makes `build` fail ("max_active_calls needs at least 1; the default is 4"), and the field can't hold 0. `build` rejects a caller or target that is not a registered agent. The tool's `agent` parameter lists the caller's targets with their roles. `WorkflowServices` has no public constructor. After enabling services, `workflow.context().workflow_services.get()` returns it. It holds a weak reference to the workflow context and returns `Channel` once the workflow has been dropped.

`run_agent` checks the caller and target policies, then runs the target with the workflow's shared memory, budget, hooks, persistence, and completion verifiers. Once the run is stopped, it returns the stop's error. A disallowed caller or target returns `BuilderInvalid`. Reaching the active-call limit returns `Channel` immediately, or with `queue_active_calls` waits until a call ends (waiting calls go on in no set order). `caller` is a host-supplied identity, not authentication. Apart from the active-call limit, there is no call-depth or cycle limit.

A run the tool starts records its parent in its task (`Task::parent`, with the calling run's task ID and the call's `tool_call_id`). The tool's `recovery_mode` is `Resume`. When a calling run that was running at a crash is recovered, its open call finds that run instead of starting another. A run that completed returns its summary without running again. An interrupted one is recovered (in doubt about its own open calls, and so on down) and waited for. One that ended otherwise returns its error. And when the call started no run that was saved (a run is saved before its first tool call, so it can't have run one), the target runs now. Such a recovered call counts toward `max_active_calls` but is never refused by it (it was admitted before the crash), and the caller and target policies are checked again. A run another recovery is recovering already is waited for, and its recorded outcome returned, so it never runs twice.

### Unfinished runs are named, and can be resumed

When a `run_workflow_agent` call fails and the run it ran did not finish (interrupted, paused after an error, waiting for an answer, or cancelled by a [graceful stop](#graceful-stop) or `cancel`), for example because recovering the run was refused by a budget, the call's error names the run and says how to resume it, as in `Agent run <id> of agent '<agent>' was interrupted and did not finish. You can resume it instead of starting over: call run_workflow_agent with agent "<agent>", resume_run "<id>" and a prompt telling it what happened and what to do next; it continues its own task.` (When the caller may not run that agent, it says the run stays for the workflow's host to recover.) A Claude Code caller gets the same text in the note that opens its resumed session.

With the optional `resume_run` parameter (a run ID), the call resumes that run instead of starting a new one, and `prompt` is the caller's message to it. The run must be a run of `agent` that the caller's run started, or an orphaned one (the run that started it ended, or answered its call, and any caller permitted to run the agent may resume it). An `Interrupted` run is recovered where it was saved (its open calls settled as for any recovery) with the message added as a `# Workflow context update` user message after those calls' results. A `Completed` or `WaitingForInput` run continues with the message as its next user message. A run `Paused` after an error continues with it as the note. A `Cancelled` run (cancelled by the budget under `BudgetExceededPolicy::Cancel`, say) is reopened as `Workflow::reopen_agent` does, with the message after the reopened note. The run's parent link then points at the resuming call, so a later crash finds it there. It is refused, with the reason, for an unknown run ID, a run of another agent, a run started by the host (not by `run_workflow_agent`), a run whose parent run can still resume it, a run running already, a run paused for tool approvals, a run that ended `Failed`, and a superseded run (cancelled because another replaced it after a restart). A `resume_run` call counts toward `max_active_calls` like a new run, and is refused over it.

The tool name constant is `services::RUN_WORKFLOW_AGENT_TOOL_NAME: &str = "run_workflow_agent"`, `services::RESUME_RUN_PARAMETER: &str = "resume_run"`.

| Tool | JSON parameters | Output |
| --- | --- | --- |
| `run_workflow_agent` | Required `agent: string` (an allowed target) and `prompt: string` (a bounded task, or with `resume_run` the message to the resumed run). Optional `resume_run: string` (`RESUME_RUN_PARAMETER`), the ID of an unfinished run of `agent` to resume instead of starting one (see above). The `agent` parameter's description names the allowed targets with their role descriptions (every registered agent when no target is restricted). It depends only on the configuration. | The target's completion summary, after any configured verification. For a run that completed after a [graceful stop](#graceful-stop)'s wrap-up notice (unreviewed), the summary followed by the note naming the run. A missing string parameter is a `MissingParameter` error, a `resume_run` that is not a UUID a `ParameterValidation` error. Workflow failures become `ExecutionFailed`, followed by the note naming the run the call ran when it did not finish. Not batch-safe. |

Custom domain tools implement `arko_agents::tool::Tool` and are registered with `global_tool` or `agent_tool`, which also add their names to the receiving agents' available tools. Use a `ToolResultInterceptor` to transform output and the freshness APIs to record custom resource effects. These traits are defined and documented in `arko_agents`.

## Background subagents

Defined in [subagents/mod.rs](src/subagents/mod.rs), [board](src/subagents/board.rs), [tools](src/subagents/tools.rs), [workspaces](src/subagents/workspace.rs).

`WorkflowBuilder::subagents(SubagentsConfig)` lets the agents the workflow services name as callers start runs of their targets in the background. `start_agent` returns a short run id (`r1`, `r2`, ...) at once, the run works while its caller (its lead) goes on, and its report reaches the lead at the lead's next turn boundary. Callers and targets are `WorkflowServicesConfig`'s, so `subagents` without `workflow_services` fails the build.

A subagent is an ordinary run of the workflow, with its own run ID (`SubagentRecord::run_id`). Its costs go into the workflow's ledger, its task and result are saved like any other, and the budget, approval and error-recovery policies, hooks, notes and verifiers apply to it. It runs outside its lead's tool calls, so the lead's run can end or be stopped while it works on. `interrupted_runs()` and `stopped_runs()` leave subagents out, since `resume_subagents` relaunches them.

### `SubagentsConfig`

`SubagentsConfig` (`Clone`, `Default`) is built with `Default` and the setters below. Its fields are public.

| Field | Type | Default and meaning |
| --- | --- | --- |
| `max_running` | `Option<NonZeroUsize>` | `DEFAULT_MAX_RUNNING`, 4. Subagents running at once, workflow-wide. A start, or a restart, over the cap waits in a queue and starts when a run ends, oldest first. `None` (`unlimited()`) means no cap. |
| `max_depth` | `u32` | 1. A run may start subagents while its depth is below this. A host's run has depth 0 and its subagents depth 1, so by default subagents start none and aren't offered `run_workflow_agent`. |
| `peer_messages` | `bool` | `true`. The lead may let its subagents message each other (`start_agent`'s `peers`, `link_agents`). |
| `max_peer_messages` | `u32` | `DEFAULT_MAX_PEER_MESSAGES`, 20. Messages two subagents may send each other between two of their lead's messages to either of them. Over it, `message_agent` is refused and says to ask the lead. |
| `copy_peer_messages_to_lead` | `bool` | `false`. The lead gets a copy of each message its subagents send each other. A copy ends no wait and wakes no one. |
| `workspaces_anywhere` | `bool` | `false`. `start_agent`'s `workspace` may name any directory, instead of one inside the caller's working directory. |
| `offer_blocking_tool` | `bool` | `true`. Callers are offered `run_workflow_agent` too. |
| `questions_to_lead` | `bool` | `true`. A subagent's `ask_user` goes to its lead as a message (`MailKind::Question`), and the lead's next message to it is the answer. Off, it goes to the workflow's `UserInputHandler`, as other runs' questions do. |
| `workspace_provider` | `Option<Arc<dyn WorkspaceProvider>>` | None. Creates and removes isolated workspaces. With one, `start_agent` offers `isolated` and `stop_agent` offers `remove_workspace`. |
| `hook` | `Option<Arc<dyn SubagentHook>>` | None. Adjusts a subagent's `AgentConfig` as it starts. |
| `start_parameters` | `Vec<arko_agents::ToolParameterDefinition>` | Empty. The host's own `start_agent` parameters, whose values reach the hook. |

```rust
// SubagentsConfig
fn max_running(self, max: usize) -> Self;          // at least 1
fn unlimited(self) -> Self;
fn max_depth(self, depth: u32) -> Self;
fn peer_messages(self, enabled: bool) -> Self;
fn max_peer_messages(self, max: u32) -> Self;
fn copy_peer_messages_to_lead(self, enabled: bool) -> Self;
fn workspaces_anywhere(self) -> Self;
fn offer_blocking_tool(self, enabled: bool) -> Self;
fn questions_to_lead(self, enabled: bool) -> Self;
fn workspace_provider(self, provider: Arc<dyn WorkspaceProvider>) -> Self;
fn hook(self, hook: Arc<dyn SubagentHook>) -> Self;
fn start_parameter(self, parameter: ToolParameterDefinition) -> Self;

// The hook, also implemented by closures of the same signature
trait SubagentHook: Send + Sync {
    fn prepare(&self, request: &StartRequest, config: &mut AgentConfig) -> Result<(), String>;
}
```

`build` rejects a start parameter named like one of `start_agent`'s own (`agent`, `task`, `workspace`, `isolated`, `peers`) or another start parameter. `StartRequest` (`Debug`, `Clone`, `PartialEq`, Serde) has `workflow_run_id`, `run_id`, `handle`, `agent`, `caller_run_id`, `caller_agent`, `task` and `parameters: BTreeMap<String, Value>`, the host parameters the call gave. The hook runs once per start, before a workspace is made, and its error is the caller's `start_agent` error. It runs again on resume, for each saved subagent, so that a resumed task's configuration compares with the prepared one, not the agent's.

### Tools

The tool name constants are `subagents::START_AGENT_TOOL_NAME` and the like, all in `SUBAGENT_TOOL_NAMES`. While `subagents` is set, global tools may not use these names.

| Tool | Offered to | JSON parameters | Output |
| --- | --- | --- | --- |
| `start_agent` | Callers whose depth allows it | Required `agent` (a target, described with the caller's targets and their roles) and `task` (the whole task). Optional `workspace` (a directory, relative to the caller's), `isolated: bool` (with a provider), `peers` (with peer messages on, run ids separated by commas, or `all`), and the host's start parameters. | `Started r1 (agent `worker`).`, or `Queued r2 ...` over the cap, with where it works and how to reach it. It is refused, with the reason, for an unknown or disallowed agent, an empty task, a workflow or caller run that is stopping, a workspace that doesn't exist or is outside the caller's directory, `workspace` with `isolated`, a provider or hook error. |
| `message_agent` | Callers and subagents | Required `to` (`lead`, a run id such as `r1`, or a run ID) and `message`. | Where the message went, read at the recipient's next step, or the run it continued or restarted. Refused for a run that isn't the sender's lead, subagent or linked peer, over the peer cap, and for a run that can't go on (closed, retired by the host, or failed in a way a message can't continue). |
| `wait_for_agents` | Callers and subagents | Optional `runs` (ids separated by commas, default all of the caller's), `all: bool`, `timeout_seconds`. | The reports and messages themselves, whole, with the runs still running. Ends when a run waited for ends (each of them, with `all`), at any message or question, when a note arrives, when other context arrives for the run, apart from `workflow-context` notices (`An update for you arrived while you waited. It follows. Wait again if you still need to.`), when the host moves it to the background (`The user ended this wait so you can go on with other work. Reports and messages from your agent runs reach you as notes as they come.`), when the run or the workflow stops, or at the time limit (`Nothing new in N s.`). What it collected comes first. The limit is the agent's `wait_for_message` limit, and the model may only shorten it. Recovered by running again. |
| `list_agents` | Callers | None. | Each of the caller's runs, with its id, agent, state and for how long (a queued run's place in line), where it works (with the branch), its task, its report or error once it ended, its peers, the messages it hasn't read yet, and its cost so far. Recovered by running again. |
| `stop_agent` | Callers | Required `run`. Optional `reason` and `remove_workspace: bool` (with a provider). | A running subagent gets a two-minute wrap-up (`StopRequest::operator`, with intent `Relaunch`) to save its progress and report, and its report then reaches the lead as a stop. A queued one is stopped before it starts. With `remove_workspace`, once the run has ended, the provider removes its workspace and the record becomes `Closed`. |
| `link_agents` | Callers, with peer messages on | Required `runs`, two or more ids. | The runs may now message each other. |

A subagent's initial prompt is its task, then a section saying it is agent run `rN` started by its lead, how to message the lead and wait for it, and where it works (an isolated workspace's `description` included).

### Mail and delivery

Reports, failures, stops, messages and questions go through a mailbox kept per run. Each `MailItem` has `id: u64`, `to` (a run ID), `from: MailSender` (`Run(Uuid)` or `Host`), `kind: MailKind` (`Report`, `Failed`, `Stopped`, `Message`, `Question`, `Copy`), `heading`, `body`, `at`, `delivered_at` and `file`. `text()` is the whole item, `[mail N] heading`, a blank line and the body. An item reaches its recipient in the first way that applies.

1. A `wait_for_agents` call waiting for it returns it whole.
2. While the recipient runs, its next turn boundary adds it to the `# Workflow context update`, oldest first, up to about `UPDATE_MAX_MAIL_BYTES` (48 KiB) per update, with the rest at the next boundary. An item over `MAX_MAIL_BYTES` (16 KiB) is shown as an excerpt with the path of the file that holds all of it (`<checkpoint_dir>/subagents/mail-N.md` with persistence, else under the temp directory).
3. Otherwise it waits. A subagent that finished is continued with its conversation, and gets the item at its first turn boundary. A subagent that was stopped, failed or interrupted is restarted only by its lead's mail. For a run the host drives, the workflow sends `SubagentEventKind::NewsWaiting`, and the host continues that run (`continue_agent`, say), which then gets the news at its first turn boundary, or takes it with `take_news`.

Mail given to a run that ends before its task holds it is given again, at the run's next turn or to the host. Mail that reaches a subagent as it completes continues it, and its new report follows. When a subagent is stopped, its lead is told which run stopped and why, in three wordings, `you stopped it: <reason>`, `the workflow's host stopped it (<reason>)` (a `Workflow::stop_run`), or `it was stopped with the workflow (<reason>)`. A run the host retired (`RunStopIntent::Retire`) can't be restarted.

### Workspaces and isolation

`start_agent`'s `workspace` gives the run a working directory, as `RunOptions::workspace` does for the host's runs (see [Run, continue, and spawn](#run-continue-and-spawn)). Its tools work there, its path checks allow it, and it is saved with the task. With `isolated: true`, the `WorkspaceProvider` makes a workspace for the run, and the record keeps it (`SubagentRecord::isolated`), so its later messages, restarts and resumes use the same one.

```rust
#[async_trait]
trait WorkspaceProvider: Send + Sync {
    async fn create(&self, request: &WorkspaceRequest) -> Result<IsolatedWorkspace, String>;
    async fn remove(&self, workspace: &IsolatedWorkspace) -> Result<(), String>;
}

// GitWorktreeProvider: a worktree per subagent, at <folder>/<id> on branch <prefix><id>
fn new(folder: impl Into<PathBuf>) -> Self;          // branch prefix "arko/"
fn branch_prefix(self, prefix: impl Into<String>) -> Self;
```

`WorkspaceRequest` (`Debug`, `Clone`, `PartialEq`, `Eq`) has `workflow_run_id`, `run_id`, `handle`, `agent`, `caller_run_id` and `base`, the caller's working directory. `IsolatedWorkspace` (`Debug`, `Clone`, `PartialEq`, `Eq`, Serde, built with `IsolatedWorkspace::at(path)`) has `path`, the run's working directory, which must exist, and the optional `root` (the workspace's top directory, when `path` is below it), `repository`, `branch`, `base_commit` and `description` (what the subagent is told about where it works). `GitWorktreeProvider` checks out the caller's `HEAD` in a new worktree on a new branch, one at a time, with the subagent in the same subdirectory as its caller. `<id>` is the run's short id, with a suffix (`r1-2`) when an earlier workflow run left that folder or branch. Removing a worktree keeps its branch, and git refuses while it has changes that aren't committed.

### Records, events and host methods

`SubagentRecord` (`Debug`, `Clone`, Serde) has `run_id`, `handle`, `agent`, `caller_run_id`, `caller_agent`, `depth`, `task`, `workspace`, `isolated`, `parameters`, `peers`, `all_peers`, `state: SubagentState`, `created`, `since` (when it entered its state), `report`, `error`, `pending: Option<PendingLaunch>` (how a queued run starts, `Start { prompt }` or `Resume { message }`), `stopped_by_caller`, `stop_requested: Option<StopRecord>` (an accepted stop kept until the launch ends, including during startup) and `asking` (it waits for its lead's answer to a question). `SubagentState` is `Queued`, `Running`, `Finished`, `Failed`, `Stopped`, `Interrupted` (the workflow stopped while it ran) or `Closed` (its workspace was removed). `is_active()` is true for `Queued` and `Running`.

`SubagentEvent` (`Debug`, `Clone`, Serde) has `timestamp`, `workflow_run_id`, `run_id`, `handle`, `agent_name`, `caller_run_id`, `caller_agent` and `kind: SubagentEventKind`, one of `Queued`, `Started`, `Resumed`, `Finished { summary }`, `Failed { error }`, `Stopped { reason, by_caller }`, `Message { mail_id, from, to }`, `Delivered { to, mail_ids }`, `NewsWaiting { mail_ids }` (for a run that isn't running) and `WorkspaceRemoved`. Events are also written to the event log, as records with `"subagent": true` and the event's fields. Their order relative to `WorkflowEvent`s isn't guaranteed. `NewsWaiting` may come more than once for the same items.

| `Workflow` method | Purpose |
| --- | --- |
| `subagents(&self) -> Vec<SubagentRecord>` | Every subagent of the run, in start order. |
| `active_subagents(&self) -> Vec<SubagentRecord>` | The subagents running or queued. A host that runs to the end (a one-shot command) waits until this is empty and no news is pending. |
| `stop_subagent(&self, run_id: Uuid, request: StopRequest) -> bool` | Stops a queued or running subagent and sends its lead stop news. Queued runs stop before they start. Running runs use the request with intent `Relaunch`, and a later message from the lead can restart either. Returns whether the stop changed. |
| `subscribe_subagents(&self) -> broadcast::Receiver<SubagentEvent>` | The events above. |
| `pending_news(&self, agent_run_id: Uuid) -> Vec<MailItem>` | The mail for a run it hasn't received, oldest first. |
| `take_news(&self, agent_run_id: Uuid) -> Vec<MailItem>` | Takes that mail out of the mailbox, for a host that puts it in its own message to the run. |
| `resume_subagents(&self) -> Vec<String>` | After `resume_from`, relaunches the subagents that were running or queued when the workflow stopped, and returns their ids. A subagent not relaunched waits, and its lead's message restarts it. |
| `transfer_subagents(&self, from: Uuid, to: Uuid) -> usize` | Run `from`'s subagents, and its pending mail, become run `to`'s (a new conversation of the same lead). Returns how many. |
| `notes_given(&self, agent_run_id: Uuid) -> u64` | The number of the newest [note](#operator-notes) the run has been given (`NotesHandle::given` too). |

To stop a lead while its subagents go on, stop the lead's run (`stop_run`). `Workflow::stop` and `cancel` stop the subagents too. Dropping the `Workflow` aborts their drivers.

### Read receipts

Each context update a run receives is reported with `AgentEvent::ContextInjected { ids, bytes }` (arko_agents), through `subscribe_events`. The ids name what it held, `note:N` for an operator note, `mail:N` for a mailbox item, and `message:<uuid>` for a bus message.

### Persistence

`WorkflowState::subagents` (a `SubagentsSnapshot` with `runs`, `mail`, `last_handle`, `last_mail` and `peer_counts`, each defaulting when missing) saves the records, the mailbox and the peer counts with each checkpoint, and with the live checkpoint while runs are live. On resume, a record that was running is settled by its run's saved result, as `Finished` (with its report posted if it was missing), `Stopped`, `Failed` or `Interrupted`. A lead's saved task holds the mail it was given, so mail given after its last save is given again.

## Shared context

Defined in [context/mod.rs](src/context/mod.rs). `context::SharedContext`, returned by `Workflow::context`, exposes the public fields below and the messaging methods listed with the `messaging` field. It has no public constructor. The lock-protected fields expose live workflow state. Prefer the `Workflow` methods where they add hooks, notifications, or accounting.

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
| `messaging` | `Option<messaging::MessagingPolicy>` | Whom each agent may message with `send_message`. `None` when messaging is disabled. `fn message_recipients(&self, agent_name: &str) -> Vec<String>` returns an agent's permitted recipients, sorted by name (empty when disabled). `fn recipient_statuses(&self, agent_name: &str) -> Vec<(String, AgentStatus)>` pairs them with their current statuses, and `fn recipient_status(&self, agent: &str) -> AgentStatus` returns one agent's status (`NotStarted` for an unknown name). `fn message_senders(&self, agent_name: &str) -> Vec<String>` returns the agents that may message it, sorted by name (empty when disabled). An agent with any is offered `wait_for_message`. |
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

Module `arko_workflows::memory`, defined in [store and configuration](src/memory/mod.rs), [agent tools](src/memory/tools.rs), [read responses](src/memory/reads.rs).

Memory provides named tiers with independent read and write access, append or replace writes, write history, and optional size limits (none by default, opt in per tier with `max_bytes`, `global_tier_with_limit`, or `broadcast_tier_with_limit`). `MemoryStore` and `MemoryConfig` default to empty. Below, `DateTime<Utc>` is `chrono::DateTime<chrono::Utc>`.

| Public type | Fields or variants. Purpose and defaults |
| --- | --- |
| `AccessScope` | `All`, `Only(HashSet<String>)`, `None`. Grants access by exact agent registration name. `fn allows(&self, agent_name: &str) -> bool` checks it. |
| `WriteMode` | `AppendOnly` (the Rust default) or `FullEdit`. An append adds a newline first when the content is nonempty. A full edit replaces the content. Both add a history entry. |
| `MemoryEntry` | `id: Uuid`, `author: String`, `task_id: Option<Uuid>`, `timestamp: DateTime<Utc>`, `content: String`. One write with its text, as `MemoryStore::entries` and `MemoryTier::history` return it. `task_id` defaults to `None` when omitted in JSON. |
| `MemoryWrite` | `id: Uuid`, `author: String`, `task_id: Option<Uuid>`, `timestamp: DateTime<Utc>`, `content: Option<String>`, `range: Option<(usize, usize)>`. One write as the tier stores it. Each successful write records a new UUID and the current UTC time, and its text is stored once, as a byte range of the tier's `content` while it is there (every append, and the latest full edit) or in `content` (an earlier version of a full-edit tier). `content` and `range` are omitted from JSON when `None`. |
| `MemoryTier` | `name: String`, `content: String`, `readers: AccessScope`, `writers: AccessScope`, `write_mode: WriteMode`, `entries: Vec<MemoryWrite>`, `max_bytes: Option<usize>`. When omitted in JSON, `entries` defaults to `[]` and `max_bytes` to `None` (no limit). No `Default` implementation. `fn text_of(&self, write: &MemoryWrite) -> &str` and `fn history(&self) -> Vec<MemoryEntry>` read the texts. A tier saved by an earlier version, whose entries each held their text beside the content, loads converted. An append-only tier's writes are matched from the end of its content back, a full-edit tier's latest write to its content. The history reads the same. |

These five types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`. Enums serialize with their Rust variant names, for example `"AppendOnly"`, `"All"`, or `{"Only":["writer"]}`. `MemoryTier.write_mode` is required in JSON even though `WriteMode` has a Rust default.

### Memory configuration

Methods on `MemoryConfig`. Setters consume and return the configuration.

| Signature | Purpose and parameters |
| --- | --- |
| `fn new() -> Self` | Empty configuration. |
| `fn tier(self, tier: MemoryTier) -> Self` | Add a fully specified tier, including initial content, policies, history, and byte limit. |
| `fn global_tier(self, name: impl Into<String>) -> Self` | Add an empty append-only tier that every agent can read and write. No limit. |
| `fn broadcast_tier(self, name: impl Into<String>, writers: impl IntoIterator<Item = impl Into<String>>) -> Self` | Add an empty append-only tier that every agent can read and only the named agents can write. No limit. |
| `fn global_tier_with_limit(self, name: impl Into<String>, max_bytes: usize) -> Self` | As `global_tier`, with a limit on the resulting UTF-8 content size in bytes. |
| `fn broadcast_tier_with_limit(self, name: impl Into<String>, writers: impl IntoIterator<Item = impl Into<String>>, max_bytes: usize) -> Self` | As `broadcast_tier`, with a byte limit. |
| `fn tiers(&self) -> &[MemoryTier]` | The configured tiers. |
| `fn build(self) -> MemoryStore` | Construct the store. `WorkflowBuilder::build` rejects duplicate tier names and access sets naming unregistered agents. |

### Memory store

Methods on `MemoryStore`. Read and write methods check the named agent's access to the tier.

| Signature | Purpose, result, and constraints |
| --- | --- |
| `fn new(tiers: Vec<MemoryTier>) -> Self` | Create a store from tier definitions. With duplicate names, the last wins. Initial content and history are kept as supplied, with each write's text stored once. |
| `fn is_empty(&self) -> bool` | Whether the store has no tiers. |
| `fn read(&self, agent_name: &str, tier_name: &str) -> Result<String, WorkflowError>` | Clone of the full current content. |
| `fn read_versioned(&self, agent_name: &str, tier_name: &str) -> Result<(usize, String), WorkflowError>` | Atomically read `(version, content)`. The version is the tier's history-entry count. |
| `fn read_window(&self, agent_name: &str, tier_name: &str, max_chars: Option<usize>) -> Result<String, WorkflowError>` | The full content, or, when longer than `max_chars` Unicode scalar values, its final `max_chars` characters prefixed with `[showing final {max_chars} of {count} characters]\n`. Accepts zero. |
| `fn entries(&self, agent_name: &str, tier_name: &str) -> Result<Vec<MemoryEntry>, WorkflowError>` | Clone of the write history (requires read access). Full-edit writes keep earlier history. |
| `fn write(&self, agent_name: &str, tier_name: &str, content: &str) -> Result<(), WorkflowError>` | Apply the tier's write mode and add a history entry without a task ID. |
| `fn write_with_context(&self, agent_name: &str, task_id: Option<Uuid>, tier_name: &str, content: &str) -> Result<(), WorkflowError>` | As `write`, recording the task ID in history. |
| `fn list_accessible(&self, agent_name: &str) -> Vec<(String, bool, bool)>` | `(tier_name, can_read, can_write)` for each tier the agent can read or write, sorted by name. Tiers it can't access are left out. |
| `fn access_of(&self, agent_name: &str) -> (bool, bool)` | Whether the agent can read any tier, and whether it can write any, which decides the memory tools it is offered. |
| `fn snapshot(&self) -> Vec<MemoryTier>` | Clone of all tiers, policies, content, and history, without access checks. Order is unspecified. |

An unknown tier returns `MemoryTierNotFound`. A denied operation returns `MemoryAccessDenied`. A write whose resulting content would exceed the tier's byte limit returns `MemoryLimitExceeded` and changes nothing. The newline added by an append counts toward the limit. History size is not limited. Direct store calls perform no freshness checks and send no notifications. The memory tools provide that integration.

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
| `READ_MEMORY_TOOL_NAME = "read_memory"` | Required `tier: string`. Optional `full: boolean` (default `false`, and the strings `"true"` and `"false"` are also accepted). Optional `max_chars: string` holding a positive integer. | Versioned JSON, described below. Batch-safe. Requires read access. When freshness tracking is enabled, a complete read records a read of the tier. |
| `WRITE_MEMORY_TOOL_NAME = "write_memory"` | Required `tier: string` and `content: string`. | `Successfully wrote to memory tier '{tier}'.` Applies the tier's policy and records the current task ID. With freshness tracking, a write based on a stale read is rejected when `block_on_conflict` is set and allowed otherwise. After a successful write, each stale reader whose task is still running gets a notice and fires `OnFreshnessConflict`. A reader whose task has finished is told what changed when it is continued (see Finished readers). Not batch-safe. |
| `LIST_MEMORIES_TOOL_NAME = "list_memories"` | None. | `Available memory tiers:` followed by each tier the agent can access, sorted by name, with `read/write`, `read-only`, or `write-only`. Tiers it can't access are left out. `No memory tiers are available to you.` when there are none. Batch-safe. |

Every successful `read_memory` response is a JSON object with `tier: string`, `version: integer`, and `kind`. The other fields depend on `kind`:

| `kind` | Additional fields | How to consume it |
| --- | --- | --- |
| `"full"` | `content: string` | Replace the local view with this complete snapshot. |
| `"append"` | `base_version: integer`, `content: string` | Append `content` exactly to the snapshot at `base_version`. It includes any required newline. |
| `"replace"` | `base_version: integer`, `start_line: integer`, `delete_lines: integer`, `content: string` | In the snapshot at `base_version`, replace `delete_lines` lines starting at zero-based `start_line` with `content`. Lines keep their newline terminators, as with Rust `split_inclusive('\n')`, so concatenating the lines yields the exact current content. |
| `"unchanged"` | None. | Keep the prior content and adopt the returned version, which can advance even when the text did not change. |
| `"partial"` | `total_chars: integer`, `content: string` | The trailing `max_chars` Unicode scalar values of a longer snapshot. An incomplete view, not a delta. |

The first complete read of a tier in a task is `"full"`. Later reads compare against the latest complete response confirmed as delivered into that task's context. A change is expressed as one exact append or line replacement. A `"full"` response is returned instead when the encoded delta would be at least as large. An unchanged tier always yields `"unchanged"`. `full=true` requests a complete snapshot, but a truncating `max_chars` still takes precedence and yields `"partial"`. Partial responses neither advance the baseline nor count as a complete read for freshness. Responses that are prepared but not yet delivered do not advance the baseline.

`ReadMemoryTool` also implements these `Tool` methods:

```rust
fn allow_result_interception(&self) -> bool; // false: interceptors never alter read results
fn reset_context(&self, task_id: uuid::Uuid); // discard this task's read baselines and pending reads
fn observe_context(&self, task_id: uuid::Uuid, message: &arko_ai::Message);
```

`observe_context` confirms delivery when a user-role message carries a successful matching `read_memory` tool result or text containing a pending response. When driving these tools outside the workflow engine, forward context observations and resets to keep deltas correct. Parameter errors are `MissingParameter` or `ParameterValidation`. Store and freshness failures are `ExecutionFailed`. A hook's `Abort` after a write is returned as an `ExecutionFailed` error even though memory has already changed, and it stops the run.

## Messaging

Module `arko_workflows::messaging`, defined in [message bus, data, and policy](src/messaging/mod.rs), [agent statuses](src/messaging/registry.rs), [agent tools and start and finish notices](src/messaging/tools.rs).

Messages are addressed to an agent name in a fixed roster. They stay queued until drained or acknowledged. Every active task of the recipient receives a live injection at its next safe turn boundary, in a `# Workflow context update` user message. A run that starts receives the queued messages in its prompt, and live any message queued while it was starting. A message reaches each run of the agent once, and a continued or resumed run is never presented a message it already holds. Below, `UserCommand` is `arko_agents::runtime::command::UserCommand`.

| Public type | Fields or variants |
| --- | --- |
| `AgentMessage` | `id: Uuid`, `from: String`, `from_task_id: Option<Uuid>`, `to: String`, `content: String`, `attachments: Vec<MessageAttachment>`, `timestamp: chrono::DateTime<chrono::Utc>`, `to_task_id: Option<Uuid>` (the run a context notice is for), `held_by: Vec<Uuid>` (the runs that stopped holding it, see [Message bus](#message-bus)), `read_by: Vec<Uuid>` (the runs that read it in a conversation they handed over, see [Conversation handover](#conversation-handover)). When omitted in JSON, `id` is the nil UUID, `from_task_id` and `to_task_id` are `None`, and `attachments`, `held_by` and `read_by` are `[]`. `to_task_id`, `held_by` and `read_by` are omitted from JSON when empty. New messages get a fresh UUID and the current UTC time. `is_held_by(task_id)` and `is_for(task_id)` (not held by the run, and not another run's notice) read them. |
| `MessageAttachment` | `File { path: String }`, `MemoryTier { tier: String }`, `Transcript { task_id: Uuid }`, `Text { name: String, content: String }`. Typed references or inline text. The bus does not read the referenced files, tiers, or transcripts. |

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
| `fn register_active(&self, agent_name: &str, task_id: Uuid, sender: tokio::sync::mpsc::Sender<UserCommand>)` | Register a task's command channel and start its delivery worker. Registering the same task again replaces the registration and stops the old worker. Can be called outside a Tokio runtime. Delivery then starts at the next `send` or `inject_task` inside one. |
| `fn unregister_active(&self, agent_name: &str, task_id: Uuid)` | Remove the registration and stop forwarding. Unacknowledged messages stay queued. |
| `async fn inject_task(&self, task_id: Uuid, content: String) -> bool` | Queue a context notice for one active task without waiting for channel capacity. Creates an `AgentMessage` from `workflow-context` (`messaging::WORKFLOW_CONTEXT_SENDER`) that checkpoints persist like any message. It doesn't end a `wait_for_message` or a `wait_for_agents`. The run reads it at its next turn boundary. `true` means queued (not yet consumed or saved). An unknown or inactive task returns `false`. |
| `async fn send(&self, from: &str, from_task_id: Option<Uuid>, to: &str, content: String, attachments: Vec<MessageAttachment>) -> Result<AgentMessage, String>` | Validate the recipient, queue the message, and return it. Active delivery workers forward it without blocking the sender on a busy recipient. A message to the sender itself returns `'<to>' is not a permitted recipient`. An unknown recipient's error lists the valid names. The messaging policy and recipient statuses are not checked. A closed channel unregisters that task. Its unacknowledged messages stay queued. The sender name is not validated. |
| `fn drain(&self, agent_name: &str) -> Vec<AgentMessage>` | Remove and return the agent's whole queue. `[]` for an unknown or empty name. |
| `fn pending(&self, agent_name: &str) -> Vec<AgentMessage>` | Clone of the whole queue without removing messages. |
| `fn pending_for(&self, agent_name: &str, task_id: Uuid) -> Vec<AgentMessage>` | The queued messages for run `task_id` (`AgentMessage::is_for`), those it doesn't hold, other than another run's context notices. What its prompt presents. |
| `fn acknowledge(&self, agent_name: &str, message_ids: &[Uuid])` | Remove the given IDs from the agent's queue. |
| `fn hand_over(&self, agent_name: &str, task_id: Uuid, conversation: &arko_ai::Conversation, prompt_ids: &[Uuid])` | Record that run `task_id` is handing `conversation` over, the queued messages it holds (in `prompt_ids`, a prompt, or a complete context update) get the run in `read_by` and `held_by`, so they are not presented to it again and count as in its conversation when it stops. Its context notices in the conversation are dropped. The engine calls it for every handover. |
| `fn snapshot(&self) -> HashMap<String, Vec<AgentMessage>>` | Clone of the pending queues, for persistence. |

Queueing, channel acceptance, and acknowledgment are separate steps. When a run stops, the engine records what it holds before checkpointing, which is the queued messages its conversation holds (presented in one of its prompts, or added as a complete runtime `# Workflow context update`, while an echo in an assistant turn, a tool result, or a quoted ID doesn't count) and those it took from its command channel but hadn't added (`Task::pending_context_injections`). Their `AgentMessage::held_by` lists the run, so a continuation or resume of it never presents them again (the context it took is added from the task, once). A run that completed acknowledges the messages in its conversation, and those it read before a handover (`read_by`), which leave the queue. One it only took stays queued. Until acknowledged, a message is still presented to every other run of the agent. A context notice (`to_task_id` set) is for its run only and is dropped when that run stops, whatever the outcome. Messages reach disk only in a checkpoint. Returning from `send` or `inject_task` is not a disk commit. A message can be delivered again to a run after interrupted work (a run stopped without being recorded), so consumers that need deduplication should use its stable ID. `MessageBus::send` fires no hooks. The `send_message` tool does.

### Messaging policy

`MessagingPolicy` sets whom each agent may message with the `send_message` tool. It implements `Debug`, `Clone`, and `Default`.

| Item | Behavior |
| --- | --- |
| `recipients: HashMap<String, HashSet<String>>` | For each agent listed, the only agents it may message (none when the set is empty). An agent without an entry may message every other registered agent. |
| `fn all() -> Self` | No restrictions. The same as `default()` and the builder's default. |
| `fn can_message<I, S>(self, agent: impl Into<String>, recipients: I) -> Self` (`I: IntoIterator<Item = S>`, `S: Into<String>`) | Restrict `agent` to `recipients`. An empty list means no one. Replaces an earlier setting for the agent. |
| `fn cannot_message(self, agent: impl Into<String>) -> Self` | `agent` may message no one. |
| `fn permits(&self, sender: &str, recipient: &str) -> bool` | Whether the policy lets `sender` message `recipient`. Never itself. Does not check that either name is registered. |
| `fn recipients_of(&self, sender: &str, roster: &[String]) -> Vec<String>` | The roster agents `sender` may message, in roster order, never `sender` itself. |
| `keep_for_finished: bool`, `fn keep_messages_for_finished(self) -> Self` | Opt-in (off by default). `send_message` to a finished agent queues the message for the agent's next run, a continuation or a new run, which gets it in its prompt, instead of refusing it. `SharedContext::keeps_messages_for_finished` reads it. Suits a workflow that continues or reruns agents, such as a lead that finished before a worker replied. Without it, a message that may never be read is refused, and the sender knows. |

An agent is offered `send_message` and `list_recipients` only when it has at least one permitted recipient, so a single-agent workflow, an agent restricted to no one, and every agent of a workflow built with `disable_messaging` get neither. The offer depends on the policy alone, not on recipient statuses. Names are registration names, and `build` rejects a policy naming an unregistered agent. The policy is read whenever an agent run starts, so it applies to runs started through workflow services and to resumed runs alike. Message queues restored from a checkpoint are delivered whatever the policy. It governs only new `send_message` calls, not `MessageBus::send`.

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
| `AgentStatus` | `NotStarted` when no run of the agent has started. `Running` when a run of the agent is running. `Waiting(WaitReason)` when every running run of the agent is waiting. `Finished(AgentRunStatus)` when no run is running, and the latest ended with this outcome. |
| `WaitReason` | `RateLimited { until: Option<chrono::DateTime<chrono::Utc>> }` while waiting out a provider rate limit. `Retrying { until }` while waiting to retry another failed model request. `UsageLimit { until }` while waiting for a plan usage limit to reset. `AwaitingApproval` while a tool call or error recovery awaits approval. `AwaitingInput` while questions await an answer. `AwaitingMessage` while waiting in `wait_for_message`. `Paused` while the workflow is paused. `BudgetExceeded` while held before a model request until the budget is raised (described as `waiting (budget_exceeded)`). |

`AgentStatus` has these methods:

```rust
fn is_reachable(&self) -> bool; // Running or Waiting: messages are delivered
fn describe(&self) -> String;
```

`describe` returns the form agents are shown, `not_started`, `running`, `waiting (<reason>)` or `waiting (<reason> until <time>)`, and `finished (<outcome>)`. The reasons are `rate_limited`, `retrying`, `usage_limit`, `awaiting_approval`, `awaiting_input`, `awaiting_message`, and `paused`. The time is RFC 3339 UTC to the second, for example `waiting (rate_limited until 2026-09-27T12:00:00Z)`. The outcomes are `completed`, `failed`, `interrupted`, `cancelled`, `waiting_for_input`, and `paused`.

Statuses follow the managed driver's events:

- `RetryScheduled` starts a `RateLimited` wait when its `rate_limit` flag is set and a `Retrying` wait otherwise, until the current time plus its delay.
- `UsageLimitReached` starts a `UsageLimit` wait until its `waiting_until_unix_ms`, when known.
- A task state awaiting tool, batch, or error-recovery approval starts `AwaitingApproval`. `AwaitingUserResponse` and `UserQuestionsAsked` start `AwaitingInput`.
- Error reports, task logs, plan-usage and cache-efficiency reports, and the stream end leave a wait in place. Any other event ends it.
- `wait_for_message` marks its run `AwaitingMessage` for as long as it waits, whatever events arrive meanwhile, and running again when it returns or its task is aborted.

A wait whose `until` time has passed counts as over, and the agent is `Running` again while the retry is under way. While the workflow is paused, every agent with a running run is `Waiting(Paused)`. With several runs of one agent, the agent is `Running` while any run is not waiting. Otherwise it is waiting for the reason of one of its runs. A run started while another run of the same agent is running is not a new start.

`Workflow::agent_status`, `SharedContext::recipient_statuses`, and `list_recipients` read the registry. An agent driven through `AgentHandle::into_parts` counts as running for the rest of the run, and one whose `AgentHandle` is dropped unused, or whose run call is dropped before it ends, counts as finished (cancelled). Checkpoints save the registry. See [Checkpoint payloads](#checkpoint-payloads).

### What agents are told about their recipients

An agent learns about the agents it may message in exactly four places. The workflow adds nothing about them to its system prompt or tool definitions, so both stay the same during a task and remain cacheable.

1. The initial prompt of a new run ends with a `# Agents you can message` section when the agent has permitted recipients, with a sentence on how to reach them, then one line per recipient with its role description and its status when the run started, such as `` - `lead` (Coordinates the work): running ``. The agent's pending messages come before it. Continuations (`continue_agent`, including an answer to `ask_user`, and a verifier's re-review or format retry) add no list. A completion verifier's prompt has no list, and the target's task prompt that it quotes has its list removed.
2. Start and finish notices arrive in `# Workflow context update` user messages, added at each safe turn boundary before a model request, one line per recipient, such as `` Agent `worker` (role: <role description>) is now running. You can message it with send_message. `` or `` Agent `worker` (role: <role description>) is finished (completed). It no longer receives messages, so don't wait for a reply from it. `` (with `keep_messages_for_finished`, `` Messages you send it are kept until it is continued or runs again. ``). A run is told about every permitted recipient that started after the registry version it last saw (from its list or its previous notices) and is still running, and every one that finished after it (its last running run ended, `AgentRecord::last_finish`) and is not running. One that did both is reported finished. It then has seen the current version, so no ordering of starts and finishes loses or repeats a notice. A continued run is told about recipients that started since it last ran, and a run whose task comes from a checkpoint saved without the registry is told about every running recipient once, and about no finish. Agents started together by `run_agents_parallel` are marked running before any is prompted, so each sees the others as running and gets no notice. The notices come from the engine's `TurnContextSource`.
3. `list_recipients` reports current statuses on request.
4. `wait_for_message` results name an agent whose start or finish ended the wait. See [Waiting for a message](#waiting-for-a-message). The start or finish notice still follows at the next turn boundary.

A run that hands its conversation over gets a fresh list in its new first message. See [Conversation handover](#conversation-handover).

### Messaging tools

`messaging::tools` defines `SEND_MESSAGE_TOOL_NAME: &str = "send_message"` and `LIST_RECIPIENTS_TOOL_NAME: &str = "list_recipients"`, and re-exports `WAIT_FOR_MESSAGE_TOOL_NAME: &str = "wait_for_message"` and `WaitForMessageTool` from `messaging::wait`. `SendMessageTool`, `ListRecipientsTool` and `WaitForMessageTool` implement `Debug` and `arko_agents::tool::Tool`. `WaitForMessageTool::new` also takes the `Arc<arko_ai::LLMClient>` whose `prompt_cache_lifetime` sets the automatic limit:

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
| `send_message` | Required `to: string` and `content: string`. Optional `attachments_json: string`, a JSON array in the attachment format above (no attachments by default). | Delivers a message to a permitted recipient that is running or waiting. Not batch-safe. |
| `list_recipients` | None. | One line per permitted recipient, never the agent itself, in the initial prompt's format with current statuses. `There are no agents you can message.` when there are none. Batch-safe. |
| `wait_for_message` | Optional `timeout_seconds: string`, a positive number of seconds (a JSON number is accepted too). | Waits for a message. See [Waiting for a message](#waiting-for-a-message). Offered to each agent that another agent may message, whether or not it may send. Not batch-safe. |

`send_message` checks its parameters first. A missing `to` or `content` is a `MissingParameter` error, invalid attachment JSON is a `ParameterValidation` error, and a non-string `attachments_json` is ignored. A `to` that is not a permitted recipient, including the agent itself and unknown names, is a `ParameterValidation` error that names only the permitted recipients, `'<to>' is not an agent you can message. The agents you can message: <recipients>.` The recipient's status decides the rest:

- `not_started` gets an `ExecutionFailed` error, `Agent '<to>' has not started, so it can't receive messages. A workflow context update will tell you when it starts.` Nothing is queued for later.
- `finished (<outcome>)` gets an `ExecutionFailed` error, `Agent '<to>' is finished (<outcome>); it no longer receives messages.` With `keep_messages_for_finished`, the message is queued instead, and the tool returns `Message <id> queued for agent '<to>', which is finished (<outcome>); it receives the message if it is continued or runs again.` (its description then says so too).
- For `running`, the message is queued and delivered at the recipient's next safe turn boundary. The tool returns `Message <id> sent to agent '<to>'.`
- For `waiting (<reason>)`, the message is queued and delivered likewise, and the result says so, for example `Message <id> sent to agent 'lead', which is waiting (usage_limit); it receives the message when it continues.` A recipient `waiting (awaiting_message)` stops waiting at once, and the result is the plain `Message <id> sent to agent '<to>'.`

On success the tool records the current task as the sender's task and fires `OnAgentMessage`. A hook's `Abort` is returned as an `ExecutionFailed` error even though the message has already been queued, and it stops the run.

### Waiting for a message

Defined in [messaging/wait.rs](src/messaging/wait.rs). An agent in native tool mode must call a tool every turn. `wait_for_message` lets it wait for a reply without polling. Its description and results are kept short, since repeated waits add them to the context every time. To wait longer, the agent calls it again. It is offered to each agent that another agent may message (`SharedContext::message_senders` is not empty), whether or not the agent may send. Its definition names no agent and no time, so it never changes during a task.

A wait returns on the first of these, checked when it is called and whenever something changes:

- A message for the run that its conversation doesn't hold yet, one sent after the run started and not yet added by a context update. Messages in the initial prompt, messages already delivered (which stay queued until the run completes), and `workflow-context` notices don't count. The result names the senders, ``Message from `lead` waiting.`` or ``Messages from `lead` and `worker_b` waiting.``, followed by any start or finish as below. The message itself is never in the result. It is delivered once, through the usual context update, and acknowledged as before. The tool returns only once the message is in the run's command channel (it waits up to a second for the delivery worker, which normally takes microseconds), and the runtime takes every queued command at a turn boundary, so the message reaches the very next request.
- A start or finish of an agent the run may message or that may message it, as `` `worker` started.`` or `` `worker` finished (completed).`` Other status changes, such as another agent's rate-limit wait, don't end a wait. The start or finish notice (see [What agents are told about their recipients](#what-agents-are-told-about-their-recipients)) still comes at the next turn boundary.
- Context that arrives for the run while it waits (a note, a terminal update), with `An update for you arrived while you waited. It follows. Wait again if you still need to.` The runtime fires the call's `TaskContext::context_arrived`, and the update comes with the result. A message that arrives this way is named as above instead. A `workflow-context` notice (a context freshness notice, say) doesn't end a wait, and comes at the next turn boundary.
- A move to the background by the host (`TaskContext::moved`, since the tool's `background_mode` is `Own`), with `The user ended this wait so you can go on with other work. Messages from other agents reach you as notes as they come.`
- The time limit, with `No messages after 270 s.`, stating the wait used (the limit, or the shorter `timeout_seconds`). Durations read like `270 s`, `54 min`, or `0.5 s`.
- A stop, with `Workflow stopping.` once the run is cancelled or aborted by a hook, or a graceful stop's wrap-up begins (the call's stop signal fired), and `Run stopping.` once the managed driver stops this run (over budget, or after an interruption). Either ends the wait at once, not at its limit. The agent then stops at the turn boundary. The runtime reads its `CancelTask` only after a tool call returns, so the wait watches the stop itself.

While it waits, the agent's status is `waiting (awaiting_message)`, which `list_recipients`, rosters, and `Workflow::agent_status` show. A `send_message` to it ends the wait. The status returns to running when the wait ends, also when the agent's task is aborted mid-wait. A `timeout_seconds` that is not a positive number is a `ParameterValidation` error. Nothing is persisted, and the waiting status isn't part of a checkpoint.

The time limit is set with these:

```rust
// messaging::wait, with WaitForMessageTimeout also in the prelude.
pub enum WaitForMessageTimeout {
    Auto,                      // the default
    Fixed(std::time::Duration), // also From<Duration>
}
fn resolve(self, lifetime: arko_ai::PromptCacheLifetime) -> std::time::Duration;
pub const DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT: std::time::Duration; // 270 seconds
pub const MAX_WAIT_FOR_MESSAGE_TIMEOUT: std::time::Duration; // 24 hours
```

`WaitForMessageTimeout` implements `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, and `Default`.

By default (`Auto`) a wait lasts up to about 4.5 minutes. Each wake-up is a cheap turn that keeps the provider's prompt cache warm, so a long wait is simply several short waits, with the agent calling `wait_for_message` again. Over ChatGPT OAuth, whose cache lasts 30 minutes, the default is about 27 minutes, and with Anthropic's 1-hour cache TTL (`GenerationConfig::anthropic_cache_ttl` set to `AnthropicCacheTtl::OneHour`, also for Claude on OpenRouter) about 54 minutes. Precisely, `Auto` takes the agent's cache lifetime from `LLMClient::prompt_cache_lifetime`, subtracts a tenth of it, at least 30 seconds, and counts from the agent's last model request, when the cache was last used. The lifetime is 5 minutes for Anthropic and, conservatively, for OpenAI with an API key, 30 minutes for OpenAI over ChatGPT OAuth, 1 hour with the 1-hour TTL, and the host's `GenerationConfig::prompt_cache_lifetime_seconds` when set. An unknown lifetime or caching off gives `DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT`.

A longer wait can be set with `Fixed(duration)`, for the workflow or one agent (more than zero and at most 24 hours, counted from the start of the wait). Provider caches often last longer than the lifetime arko reports (OpenAI documents about 30 minutes on the API, and a test over ChatGPT OAuth still hit after 40), but that isn't guaranteed, and a miss re-reads the whole context at full price. A host that knows its route's lifetime can set `prompt_cache_lifetime_seconds` instead, and the automatic limit follows it. The model's `timeout_seconds` shortens a wait, never lengthens it.

## Operator notes

Module `arko_workflows::notes`, defined in [notes](src/notes/mod.rs), [control folder](src/notes/control.rs), [delivery](src/workflow/engine.rs).

A note is guidance the host gives the agents while the workflow runs, such as a correction, a workaround found meanwhile, or a change of priority. It reaches running agents at their next turn boundary and later runs in their initial prompt, and it never touches an agent's `AgentConfig`, so a resume still accepts the agents' saved tasks. Notes are the supported way to change instructions in a running or resumed workflow (a changed `AgentConfig`, such as new instructions, makes `resume_from` refuse the tasks saved with the old one). Notes are posted only through this API and the opt-in [control folder](#the-control-folder), and agents have no tool to post them.

```rust
// Workflow
fn post_note(&self, note: Note) -> Result<PostedNote, WorkflowError>;
fn notes(&self) -> Vec<PostedNote>; // every note, oldest first
fn note_image(&self, image: &NoteImage) -> Result<arko_ai::Image, WorkflowError>;
fn notes_handle(&self) -> NotesHandle;
fn notes_given(&self, agent_run_id: Uuid) -> u64; // the newest note's seq the run was given, 0 for none
fn skip_one_time_notes(&self, agent_run_id: Uuid); // the one-time notes posted so far don't reach the run

// NotesHandle: Clone, Debug, Send, Sync
fn post(&self, note: Note) -> Result<PostedNote, WorkflowError>;
fn notes(&self) -> Vec<PostedNote>;
fn image(&self, image: &NoteImage) -> Result<arko_ai::Image, WorkflowError>;
fn given(&self, agent_run_id: Uuid) -> u64;

// Note: a standing note from the operator to every agent, then setters.
fn new(text: impl Into<String>) -> Self;
fn to<I, S>(self, agents: I) -> Self;     // NoteAudience::Only
fn except<I, S>(self, agents: I) -> Self; // NoteAudience::Except
fn once(self) -> Self;                     // NoteKind::Once
fn source(self, source: NoteSource) -> Self;
fn images(self, images: impl IntoIterator<Item = arko_ai::Image>) -> Self; // a one-time note's only

// notes
pub const MAX_NOTE_BYTES: usize;   // 16 KiB
pub const PROMPT_MAX_NOTES: usize; // 20, per prompt or context update
pub const PROMPT_MAX_BYTES: usize; // 16 KiB, likewise
```

| Public type | Fields or variants |
| --- | --- |
| `Note` | `text: String`, `audience: NoteAudience`, `kind: NoteKind`, `source: NoteSource`, `images: Vec<arko_ai::Image>`. In JSON only `text` is required. The others default, and `images` is left out when empty. |
| `NoteAudience` | `All` (the default) for every agent. `Only(BTreeSet<String>)` for the agents named. `Except(BTreeSet<String>)` for every agent but those. Snake case in JSON, as `"all"`, `{"only": ["a"]}`, `{"except": ["b"]}`. |
| `NoteKind` | `Standing` (the default) is delivered to the running runs of its audience and shown in the initial prompt of every later run. `Once` is delivered once to each run of its audience that has started when it is posted. `"standing"`, `"once"`. |
| `NoteSource` | Who wrote it, as the agents are told, `Operator` (the default, "the operator (the person running this workflow)"), `Host` ("the workflow (the program running the agents)"), or `Agent(String)` ("agent `name`"), for example an observer reviewing the run. The name need not be a registered agent's. In JSON it is `"operator"`, `"host"` or `{"agent": "observer"}`. |
| `PostedNote` | `id: Uuid` (new, or the control folder entry's), `seq: u64` (the note's number in the run, from 1, in posting order), `posted_at: chrono::DateTime<chrono::Utc>` (when posted, or written to the control folder), `note: Note` (without its images, so `note.images` is empty), `images: Vec<NoteImage>` (omitted from JSON when empty). |
| `NoteImage` | A posted note's image, `Url(String)` or `Stored { name, mime_type }`, kept by the workflow under `name`, `<SHA-256 of its bytes>.<type>` (`png`, `jpg`, ...). A persisted run writes it to `images/<name>` in its run directory when the note is posted (a write that fails fails `post_note` with `Persistence`), a run without persistence keeps it in memory. `Workflow::note_image` and `NotesHandle::image` read it. Notes, checkpoints and the event log hold the names only, so a note's images cost nothing at later steps or checkpoints. |
| `NotesHandle` | From `notes_handle`. Clones post to the same run from any task or thread. It holds a weak reference, so once the workflow is dropped, `post` returns `BuilderInvalid` and `notes` returns none. |

All implement `Debug` and `Clone`. All but `NotesHandle` implement `PartialEq`, `Eq`, and Serde, and `Note`, `NoteAudience`, `NoteKind`, and `NoteSource` implement `Default`.

### Posting

`post_note` trims the text, makes every line break in it (CR LF, a lone CR, NEL, U+2028, U+2029) a LF, and checks the note. It returns `BuilderInvalid` for an empty text (allowed when the note has images), one over `MAX_NOTE_BYTES` (a document belongs in a file or a memory tier), an `Only` naming no agent, an `Agent` source whose name is blank or holds a line break or a control character, a note no agent receives, a standing note with images, or images an agent of its audience can't be sent (`arko_ai::Model::sendable_image` says why, such as a model without image input, a URL for Anthropic, Gemini or Claude Code, a GIF for Gemini, or an image over the model's limits that can't be read or shrunk enough, while one that can is shrunk when it is given), and `AgentNotFound` for an audience naming an unregistered agent. A posted note is logged once, as a `log::info!` line naming its number, source, recipients and first line, and, with persistence, a line in the event log, `{"timestamp", "workflow_run_id", "note": <PostedNote>}` (written at once inside a Tokio runtime, otherwise with the next write to the event log, unlike agent events it has no `agent_name` or `event`, a note restored on resume is not logged again). It is saved with the next checkpoint. While agents run under the task journal, the live checkpoint is written at once, as for a budget change. A stopped run takes notes too, for a resume.

### Audience

Names are registration names, as in [`MessagingPolicy`](#messaging-policy). Completion verifiers' reviews, and the agents set with `WorkflowBuilder::notes_by_name_only`, get a note only when its `Only` names them. Under `All` and `Except` they get nothing, so independent roles stay independent of the operator's steering. A verifier's agent run outside a review (by the host, or as a delegation target) is an ordinary run. An agent never gets a note whose source is that agent. Because an agent that gets a note can pass its content on in a message, posting a note that leaves out such an independent agent while one of its recipients may message it (under the messaging policy) logs a warning, once per agent. Restrict whom an independent agent hears from when that matters. For a completion verifier that isn't set with `notes_by_name_only`, it is a debug line unless messages are kept for finished agents (`keep_messages_for_finished`), because a verifier runs only while it reviews, and a message to it is refused at other times. Since a verifier judges the completion without the target's notes, address a note to the verifier too (`Note::to([.., verifier])`) when it changes what counts as done, or the verifier may keep rejecting completions the note asked for.

### Delivery

What a run has been given is tracked per run, so it gets each note once:

- A running run gets every note it receives that it hasn't been given, standing or one-time, at its next safe turn boundary (the same one as messages, start and finish notices and a [graceful stop](#graceful-stop)'s wrap-up notice, after those notices), in its `# Workflow context update`, as `A new note on your task.` (or `New notes on your task, oldest first.`), then `A note refines your task; it doesn't change your role's rules (what you may read, write or run, and whom you may message) or override the workflow's own notices, such as a request to stop. Where notes differ, the newer one takes precedence.`, then the notes. An update shows at most `PROMPT_MAX_NOTES` (20) new notes and about `PROMPT_MAX_BYTES` (16 KiB, the first always in full), oldest first, then `(N more notes follow at your next turn.)`. Only the notes shown count as given, and the rest come at the following boundaries, so none is lost. A run inside a long tool call gets them when the call returns. A one-time note's images follow the update's text in the same message, each after a label naming the note (`The image of note 4:`, or `Image 1 of note 4:` for several), and the note's item says `(With an image, shown after this update.)` or `(With N images, ...)`. An image that can't be read or sent when the note is given is left out, and the item says so (`(Its image can't be shown.)`, or how many of the images can't be shown) instead of announcing it. During a turn, a Claude Code agent gets the update's text at once (a hook's context is text), with `(The image of this update can't come during a turn: it comes with your next input.)`, and the images with its next input, at the turn's end.
- A new run's initial prompt gets the standing notes it receives in a `# Notes on your task` section, after the delivered messages and before the list of agents it may message, newest first, at most `PROMPT_MAX_NOTES` (20) and about `PROMPT_MAX_BYTES` (16 KiB, with the newest always shown in full), then `(N earlier notes are not shown.)` when some are left out. One-time notes posted before a run started never reach it.
- A continued, resumed or recovered run gets the notes posted since it was last given any, at its first turn boundary (and the next, past an update's cap), including one-time notes. `skip_one_time_notes(run)` keeps the one-time notes posted so far from a run that isn't running. A host that continues an earlier run of an agent in place of the one its user was talking to (going back to an earlier conversation) calls it, so lines typed during the other run's turns don't reach it. Standing notes, and one-time notes posted later, still do. A run recovered from its task as last saved (`recover_agent`, `reopen_agent`, after a crash, or aborted after the grace period) is given again any note it receives that its saved task doesn't hold, as messages are, so a note given in a turn the crash cut short is not lost, and may be given twice.
- After a [handover](#conversation-handover), the fresh conversation restates the standing notes the agent had been given (newest first and capped, as in an initial prompt), after the report and before the list of agents it may message. The notes it hadn't been given yet, such as one posted while it wrote its report, come in the context update before its first request. The one-time notes the old conversation held aren't repeated, because the report covers them.
- A completion verifier quotes its target's task without the target's notes, and gets only the notes that name it.

Each note is shown as `- Note <seq> from <source>, <YYYY-MM-DD HH:MM> UTC:`, then its text with every line indented two spaces (so no line of a note can pass for a heading of the prompt). Claude Code agents get notes the same way.

### Persistence

Checkpoints save the notes in `WorkflowState::notes` (a `NotesSnapshot`, see [Checkpoint payloads](#checkpoint-payloads)), with every note, what each run has been given, the runs the one-time notes were skipped for, and how much of the control folder has been read. `resume_from` restores them, so later runs and continued runs get them as before the resume. A checkpoint written before notes existed loads with none.

### The control folder

`WorkflowBuilder::control_folder()` (opt-in, and it requires persistence) lets another process post notes to the run, while it runs or while it is stopped, and stop agent runs while it runs, for a binary's `note` subcommand, a bot, an observer, or the [`runctl` example](#minimal-composition-and-examples). `build` creates `<run dir>/control/`.

```rust
// notes::control
pub const FOLDER: &str = "control";
pub const NOTES_FILE: &str = "notes.jsonl";
pub const LOCK_FILE: &str = "notes.lock";
fn folder(run_dir: &Path) -> PathBuf;
fn post_note(run_dir: &Path, note: &Note) -> std::io::Result<ControlNote>;
fn read_notes(run_dir: &Path) -> std::io::Result<Vec<ControlNote>>;
// ControlNote: Debug, Clone, PartialEq, Eq, Serde
pub struct ControlNote { pub id: Option<Uuid>, pub at: Option<chrono::DateTime<chrono::Utc>>, pub note: Note }
```

`NOTES_FILE` holds one `ControlNote` per line, a note's JSON fields with `id` and `at` beside them (both may be left out in a line written by hand). `post_note` appends one, with a new `id` and the current time, under an exclusive `std::fs::File::lock` on `LOCK_FILE` (never a lock of the run's own), and syncs it. A last line a crashed writer left unfinished is ended first. It returns `NotFound` when the run directory has no control folder (the workflow doesn't read one, so the note would never be read) and `InvalidInput` for an empty note or one over `MAX_NOTE_BYTES`. The audience is checked by the workflow. `read_notes` returns every note in the file, skipping lines that aren't notes, for status tools. Which of them the workflow posted is in its checkpoint (`NotesSnapshot::notes`, by `id`).

The workflow reads the lines added since it last read the file when a running agent reaches a turn boundary, when it starts an agent or hands one's conversation over, and in `notes()`, under a shared lock it doesn't wait for. While a writer holds the lock, the lines are read the next time. Each line is posted as `post_note` would. A line that isn't a note, or a note that can't be posted (an unknown agent, say), is skipped with a warning in the log. A line whose `id` was posted already is skipped, and an unfinished last line is left until a writer ends it. How far the file has been read is saved in checkpoints with the notes, so after a resume nothing is posted twice and notes written while the run was stopped are read. After a crash, a note read since the last checkpoint is read again. A line with an `id` is posted with the same identity, and is logged again.

#### Stop requests

While agents run, the workflow also reads the folder's stops file every second and stops the runs each request names, as [`stop_run` and `stop_agent`](#stopping-one-agent-run) do:

```rust
// notes::control
pub const STOPS_FILE: &str = "stops.jsonl";
pub const STOPS_POLL_INTERVAL: std::time::Duration; // 1 second
fn request_stop(run_dir: &Path, stop: &ControlStop) -> std::io::Result<ControlStop>;
fn read_stops(run_dir: &Path) -> std::io::Result<Vec<ControlStop>>;
// ControlStop: Debug, Clone, PartialEq, Eq, Serde
pub struct ControlStop {
    pub id: Option<Uuid>, pub at: Option<chrono::DateTime<chrono::Utc>>,
    pub run: Option<Uuid>,         // one run, or
    pub agent: Option<String>,     // every running run of an agent
    pub kind: StopKind,            // default Operator
    pub reason: String,            // default empty
    pub wrap_up_secs: Option<u64>, // none: cancelled at once
    pub intent: RunStopIntent,     // default Retire
}
// ControlStop::run(id, reason), ControlStop::agent(name, reason), then wrap_up(Duration),
// intent(RunStopIntent). request() is the StopRequest it asks for.
```

`request_stop` appends one line, with a new `id` and the current time, under the same lock as notes, and syncs it. It returns `NotFound` when the run directory has no control folder and `InvalidInput` for a request that names both a run and an agent, or neither. In JSON only `run` or `agent` is required, so the line `{"agent": "worker-12"}` retires every running run of `worker-12` at once. The workflow starts reading when its first agent run starts and stops once the workflow is stopped. A request that finds no run to stop, or names an unknown agent, is skipped with a warning in the log. Stop requests are for the live workflow. A workflow reads only the lines written after it was built, so a request written while it isn't running is never applied, and how far the file was read isn't saved. `read_stops` returns every request in the file, for status tools.

Whoever can write the control folder can post notes, as any source, and stop runs. Keep the run directory out of the agents' reach (outside their workspace, or behind a path policy), as a host keeps its own command line out of the agents' sandboxes.

## Context freshness

Module `arko_workflows::freshness`, defined in [freshness tracking](src/freshness/mod.rs).

Version-based tracking detects stale reads of files, directory trees, memory tiers, and workflow-defined resources. Actors include the task ID, so concurrent runs of the same agent are tracked separately. The tracker records explicit observations and mutations. It does not watch the filesystem for outside changes.

| Public type | Fields or variants. Purpose |
| --- | --- |
| `ResourceId` | `File(PathBuf)`, `DirectoryTree(PathBuf)`, `MemoryTier(String)`, `Custom { namespace: String, key: String }`. Resource identity. |
| `ActorId` | `task_id: Uuid`, `agent_name: String`, built with `fn new(task_id: Uuid, agent_name: impl Into<String>) -> Self`. |
| `ToolResourceEffect` | `ReadFile { parameter: String }`, `ReadDirectoryTree { parameter: String }`, `WriteFile { parameter: String }`. `parameter` names the tool argument that holds the path string. |
| `ContextFreshnessConfig` | `block_on_conflict: bool`, `tool_effects: HashMap<String, ToolResourceEffect>`. The default blocks conflicts and uses the mappings below. |
| `FreshnessConflict` | `resource: ResourceId`, `observed_version: u64`, `current_version: u64`, `last_writer: Option<ActorId>`. A read that has become stale. |
| `FreshnessSnapshot` | Opaque tracking state (private fields). `Default` is empty. `ResourceTracker::snapshot` and `from_snapshot` round-trip it, including reads and writes. |

All six types implement `Debug`, `Clone`, `Serialize`, and `Deserialize`. `ResourceId` and `ActorId` also implement `PartialEq`, `Eq`, and `Hash`, and `ContextFreshnessConfig` and `FreshnessSnapshot` implement `Default`. Enums serialize with their Rust variant names, for example `{"MemoryTier":"notes"}` and `{"ReadFile":{"parameter":"path"}}`.

| Default tool mapping | Effect and argument |
| --- | --- |
| `read_file` | `ReadFile { parameter: "path" }` |
| `list_files` | `ReadDirectoryTree { parameter: "directory" }` |
| `write_file`, `edit_file`, `apply_diff` | `WriteFile { parameter: "path" }` |
| `Read` (Claude Code) | `ReadFile { parameter: "file_path" }` |
| `Write`, `Edit`, `MultiEdit` (Claude Code) | `WriteFile { parameter: "file_path" }` |
| `NotebookEdit` (Claude Code) | `WriteFile { parameter: "notebook_path" }` |

A Claude Code agent's built-in calls pass the workflow's execution middleware, so the Claude Code mappings track them as the arko file tools are tracked. Claude Code's `Bash` is not tracked, like a shell command run through arko's terminal tools.

`ContextFreshnessConfig::tool_effect(self, tool_name: impl Into<String>, effect: ToolResourceEffect) -> Self` adds or replaces a mapping. Assign `tool_effects` to replace them all. Fields omitted in JSON take the `Default` values. A given `tool_effects` replaces the default mappings whole.

With a tracker configured, the workflow checks each mapped write before the tool runs. When it overlaps context the agent read that another actor has since changed, the call is rejected with feedback to re-read if `block_on_conflict` is set, and allowed otherwise. After a successful mapped tool call, the read or write is recorded. For a write, every active stale reader receives a notice at its next turn boundary (a wait it is in goes on), and `OnFreshnessConflict` fires once per stale reader. An `Abort` from that hook stops the run. The write's own result stands.

### Finished readers

When a task stops, a task that can still be continued (completed, waiting for input, or paused) keeps its reads but no longer receives notices or fires `OnFreshnessConflict`, and a notice queued for it and not yet read is dropped. Any other task's reads are forgotten. When it is continued (`continue_agent`) or resumed (`resume_agent`), it gets one message listing the resources it read that another actor changed since, advising it to re-read any of them before editing it or relying on it, at the end of the continuation prompt or as a context update before the resumed turn. Its kept reads still block a stale write under `block_on_conflict` until it reads again. After a restart from a checkpoint no task runs, so every recorded reader counts as finished until its task is continued.

### Resource tracker

Methods on `ResourceTracker`.

| Signature | Purpose and behavior |
| --- | --- |
| `fn new(workspace: PathBuf, config: ContextFreshnessConfig) -> Self` | Empty tracker rooted at the workspace. |
| `fn from_snapshot(workspace: PathBuf, config: ContextFreshnessConfig, snapshot: FreshnessSnapshot) -> Self` | Restore tracking state with the given workspace and configuration. Every recorded reader counts as finished. |
| `fn snapshot(&self) -> FreshnessSnapshot` | Clone of all tracking state. |
| `fn config(&self) -> &ContextFreshnessConfig` | The configuration. |
| `fn resource_for_tool(&self, tool_name: &str, parameters: &HashMap<String, Value>) -> Option<(ResourceId, bool)>` | Resolve a mapped tool's path argument. The boolean is `true` for a write. `None` for an unmapped tool, a missing or non-string argument, or a path no tool can open (one with a NUL byte). Paths resolve as the built-in file tools open them and the agent runtime's workspace check judges them, with `arko_agents::utils::resolve_path`. Surrounding whitespace is trimmed, `~` and `$VAR` are not expanded, a relative path is joined to the workspace, and `.` and `..` are resolved lexically. Symlinks are not resolved. |
| `fn record_read(&self, actor: &ActorId, resource: ResourceId)` | Record the resource's current version as the actor's observation, replacing an earlier one. A new resource starts at version zero. |
| `fn check_write(&self, actor: &ActorId, resource: &ResourceId) -> Vec<FreshnessConflict>` | The actor's overlapping observations whose resource has since reached a newer version last written by another actor. No prior read means no conflict. Reports only. It does not block anything. |
| `fn record_write(&self, actor: &ActorId, resource: ResourceId) -> Vec<ActorId>` | Record a successful mutation, incrementing the affected versions, setting the last writer, and returning the other actors with affected reads, except finished ones (see `finish_actor`). A file write also increments already tracked directory trees that contain it. |
| `fn clear_actor(&self, actor: &ActorId)` | Remove the actor's observations. Its writes and the resource versions remain. |
| `fn finish_actor(&self, actor: &ActorId, continued: bool)` | The actor's task stopped. With `continued`, keep its reads but leave it out of `record_write`'s result (no notices). Otherwise `clear_actor`. The engine calls this when it records a task. |
| `fn resume_actor(&self, actor: &ActorId) -> Vec<ResourceId>` | The actor's task runs again. It is included in `record_write` again, and the resources it read that another actor changed since are returned, sorted (files, directory trees, memory tiers, custom). |

Resources overlap when they are identical, or when a directory tree that was read contains a written file. Memory and custom resources overlap only by exact identity. `record_read` and `record_write` use their arguments as supplied, so resources constructed manually need consistently normalized paths.

## Completion verification

Defined in [verification/mod.rs](src/verification/mod.rs), and run by the [engine](src/workflow/engine.rs). All public types and the protocol constant are in the prelude.

A `CompletionVerifierConfig` registered with `WorkflowBuilder::completion_verifier` gates an agent's `complete_task` call on a review by another registered agent. Configurations run in registration order. A rejection returns the verifier's feedback to the agent for repair. A correctness gate fails the task once its cap is exceeded. Targets are registration names. A configuration never reviews its own verifier agent, even when that agent is listed as a target. Targets are required, and `build` rejects a configuration without one, so a verifier never attaches to agents it wasn't aimed at (or to other verifiers). Use distinct configuration names, because repair counts and conversation reuse are keyed by target task ID and configuration name. A review in progress when the run is stopped decides nothing, and the target agent is cancelled with the run, not failed. A completion an agent makes after a [graceful stop](#graceful-stop)'s wrap-up notice isn't reviewed, since it reports progress, and the run it is continued in after a restart is reviewed as usual.

A verifier reviews `complete_task` calls, and its own agent gives the verdict with one. So `build`, and the start of a run with a configuration of its own (a subagent's), refuse `AgentToolConfig::text_reply_ends_task` on a target agent and on a verifier agent, because with it the agent ends its task with a reply in text, which no verifier sees.

### `CompletionVerifierConfig`, `VerificationKind`, and `RepairContextConfig`

`CompletionVerifierConfig` implements `Debug`, `Clone`, `Serialize`, and `Deserialize`. Its public fields are these:

| Field | Type | Purpose and constructor default |
| --- | --- | --- |
| `name` | `String` | Configuration name. Appears in records and feedback. |
| `verifier_agent` | `String` | Registered agent that reviews the completion. |
| `target_agents` | `HashSet<String>` | Target registration names. Empty from the constructors, and at least one is required by `build`. |
| `kind` | `VerificationKind` | `Correctness`, or advisory `Efficiency`. Set by the constructor. |
| `instructions` | `String` | Reviewer instructions. Defaults below. |
| `max_minor_rejections` | `u32` | `3` for correctness, `2` for efficiency. |
| `max_major_rejections` | `u32` | `0` for both kinds. |
| `reuse_conversation` | `bool` | `false`. Also `false` when omitted in JSON. |
| `repair_context` | `Option<RepairContextConfig>` | `None`. Also `None` when omitted in JSON. |
| `include_prior_findings` | `bool` | `true` (also when omitted in JSON). A new review of a task after an earlier verdict (without conversation reuse) is given that verdict and its findings. |
| `max_format_retries` | `u32` | `1` (also when omitted in JSON). How many times an unparseable verdict is asked for again before the review counts as failed. |
| `max_review_retries` | `u32` | `1` (also when omitted in JSON). How many failed reviews of one task are retried before the task pauses and the run call returns `VerifierFailed`. |

`VerificationKind` has the variants `Correctness` and `Efficiency` and implements `Copy`, `Clone`, `PartialEq`, `Eq`, and Serde. `RepairContextConfig` implements `Clone`, `PartialEq`, `Eq`, and Serde and has one public field, `max_bytes: usize`, the byte budget for repair-delta text. Neither configuration implements `Default`.

| Signature on `CompletionVerifierConfig` | Parameters and result |
| --- | --- |
| `pub fn correctness(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self` | Correctness gate. The default instructions are "Check correctness, completeness, and maintainability. Inspect the actual workspace rather than trusting the candidate summary." |
| `pub fn efficiency(name: impl Into<String>, verifier_agent: impl Into<String>) -> Self` | Advisory efficiency gate. The default instructions are "Look for avoidable repeated work, poor asymptotic behavior, unnecessary I/O or API calls, and needless resource use. Do not demand speculative optimization." |
| `pub fn target(mut self, agent: impl Into<String>) -> Self` | Add a target registration name. Duplicates collapse. |
| `pub fn instructions(mut self, instructions: impl Into<String>) -> Self` | Replace the reviewer instructions. |
| `pub fn minor_rejection_cap(mut self, cap: u32) -> Self` | Number of minor rejections allowed. |
| `pub fn major_rejection_cap(mut self, cap: u32) -> Self` | Number of major rejections allowed (correctness only). |
| `pub fn reuse_conversation(mut self, enabled: bool) -> Self` | Continue the earlier verifier conversation when the same target task is reviewed again and that verifier result is available. Otherwise start a new review. |
| `pub fn repair_context(mut self, max_bytes: usize) -> Self` | Enable repair-delta injection with `max_bytes.max(1_024)`. Does not enable conversation reuse, which repair deltas require. |
| `pub fn include_prior_findings(mut self, enabled: bool) -> Self` | Opt out (`false`) of giving a new review the prior findings. |
| `pub fn format_retries(mut self, retries: u32) -> Self` | Set `max_format_retries`. |
| `pub fn review_retries(mut self, retries: u32) -> Self` | Set `max_review_retries`. |

A correctness gate allows exactly the configured number of minor and of major rejections and fails the task on the next rejection of that severity. The two counts are independent, so a major cap of zero fails on the first major rejection. An efficiency gate treats every major rejection as minor, never uses the major cap, and stops running once its minor count reaches the cap. A minor cap of zero disables it. **A failed review doesn't fail the target.** A review fails when the verifier's run fails (it ends with an error, or pauses) or its verdict is still unusable after `max_format_retries` requests to restate it. Within `max_review_retries` failed reviews of a task, the completion is rejected with a message saying the verifier could not review it and nothing was decided, so the agent calls `complete_task` again and a new review runs. On the next failure the completion is neither accepted nor rejected. The agent's `complete_task` call is answered that nothing was decided, and the task pauses for the caller (status `Paused`, `OnAgentPaused`, an error-recovery pause with the cause `Middleware`, also inside another agent's call), and the run call returns `WorkflowError::VerifierFailed { verifier, agent, task_id, reason }`. Continue it with `continue_agent` on its result (`Workflow::agent_result(task_id)`). The prompt is the agent's next message, and its next `complete_task` is reviewed again. `resume_agent` with `AgentResume::ContinueAfterError` does the same, and `AgentResume::FailAfterError` fails it. The failed-review count is kept per target task and verifier, and cleared when the task completes or pauses this way. The rejection counts and repair baselines carry over into a continuation.

With conversation reuse, the continued review receives the previous verdict, the new completion summary, and focused re-review instructions. Without it, a new review of a task that already has a verdict from the same configuration is given that verdict and its findings and asked to check whether each is resolved, unless `include_prior_findings` is off. The first review of a task is a full first pass. A repair delta is added only when context freshness tracking is enabled. It covers tracked files inside the workspace and new memory entries the verifier can read, falling back to truncation or content hashes when text is unavailable. It does not enumerate directory-tree or custom resources and does not replace inspecting the affected artifacts. `repair_context` clamps the byte budget. Setting the field directly or deserializing does not.

### Verdict protocol and records

`pub const VERIFIER_OUTPUT_CONTRACT: &str` holds the plain-text response instructions included in verifier prompts. The first non-empty line of the verifier's completion summary must be `ACCEPTED`, `REJECTED_MINOR`, or `REJECTED_MAJOR`. Rejections require non-empty feedback on the following lines. Acceptance may omit it. Leading blank lines, whitespace around the decision, and CRLF line endings are accepted. JSON, Markdown wrappers, and feedback on the decision line are not. Everything after the decision line is opaque feedback. A malformed response gets `max_format_retries` continuations (one by default) asking for a corrected format. If it is still unusable, the review has failed (see above).

`VerificationDecision` implements `Clone`, `PartialEq`, `Eq`, and Serde:

| Rust variant | Serde value | Meaning |
| --- | --- | --- |
| `Approved` | `"approved"` | Protocol `ACCEPTED`. The gate passes. |
| `RejectedMinor` | `"rejected_minor"` | Bounded corrections requested. |
| `RejectedMajor` | `"rejected_major"` | Major correctness issue. Efficiency gates record it as `RejectedMinor`. |

`VerificationRecord` implements `Clone` and Serde. Its public fields are these:

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

Defined in [distillation/mod.rs](src/distillation/mod.rs), with registration and order in the [engine](src/workflow/engine.rs). Both public types implement `Clone` and Serde and are in the prelude.

A `DistillationConfig` registered with `WorkflowBuilder::distiller` replaces large matching tool results with an LLM-written summary. Each distiller is installed only for the agents it targets, and distills only the tools it targets. `build` requires at least one of each. Any tool whose results may be intercepted can be targeted, `read_file` included (for a large file the agent wants specific details from). Distillers run after the global and per-agent interceptors, in registration order. Each sees the result of the stages before it, and all of a distiller's filters must match. Once the run is stopped, a distiller makes no request and returns the tool output unchanged. A distiller calls the workflow's LLM client with its model, reasoning, and output-token limit, in a fresh conversation with prompt caching turned off where the provider allows it (each request is a one-off ending with a tool output no later request repeats, so a cache write would never be read, and ChatGPT OAuth caches implicitly and is asked nothing), giving the agent's initial task prompt, the tool name and parameters, and the current tool output as context. Results of tools that opt out of interception (`Tool::allow_result_interception`, for example `read_memory`) are never distilled. A distiller never sends a request over its own model's input limit. When the request (estimated at 3 bytes per token) plus the output-token limit is over `LLMClient::max_input_tokens` for the distiller's model, the tool output is kept unchanged, nothing is charged, and a record notes the skip (`rejection` starting with `skipped:`). The agent's overflow guard (arko_agents' *Tool result size*) runs after the distillers, so an output too large for the agent's own model is still shortened before it reaches the model.

### `DistillationConfig`

| Public field | Type | Purpose and default from `new` |
| --- | --- | --- |
| `name` | `String` | Configuration name used in records, errors, and cost descriptions. |
| `model` | `arko_ai::Model` | Distillation model supplied by the caller. |
| `target_agents` | `HashSet<String>` | Registration names to match, as given to `WorkflowBuilder::agent` (not `AgentConfig::name`). Required, so `build` rejects an empty set. |
| `target_tools` | `HashSet<String>` | Tool names to match. Required, so `build` rejects an empty set. |
| `min_chars` | `usize` | Minimum input size in Unicode scalar values, inclusive. Default `8_000`. |
| `max_output_tokens` | `Option<usize>` | Generation limit, including reasoning tokens. The default, `None`, means `default_max_output_tokens()` (`16_000`, or the model's `Model::max_output_tokens` when that is known and smaller) is sent where the provider takes a limit, and no limit on ChatGPT OAuth, whose backend rejects one. A set value is sent on every request (arko_ai drops it on ChatGPT OAuth, with a warning logged once). Must be positive. A JSON configuration with a number loads as that set value. |
| `instructions` | `String` | Content-preservation instructions. Default below. |
| `reasoning` | `Option<arko_ai::ReasoningConfig>` | Default `None`. Omitted from JSON when `None` and `None` when omitted. |

| Signature on `DistillationConfig` | Parameters and result |
| --- | --- |
| `pub fn new(name: impl Into<String>, model: arko_ai::Model) -> Self` | Configuration with the defaults above. The default instructions are "Preserve every fact relevant to the task, exact errors, paths, identifiers, constraints, and actionable details. Remove repetition and irrelevant bulk. Never invent facts." |
| `pub fn target_agent(mut self, agent: impl Into<String>) -> Self` | Add an agent registration-name filter. Duplicates collapse. |
| `pub fn target_tool(mut self, tool: impl Into<String>) -> Self` | Add a tool-name filter. Duplicates collapse. |
| `pub fn min_chars(mut self, chars: usize) -> Self` | Inclusive size threshold. `0` also admits empty output. |
| `pub fn max_output_tokens(mut self, tokens: usize) -> Self` | Set the token limit. `build` rejects zero. |
| `pub fn default_max_output_tokens(&self) -> usize` | The limit sent when none is set. |
| `pub fn instructions(mut self, instructions: impl Into<String>) -> Self` | Replace the instructions. |
| `pub fn reasoning(mut self, reasoning: arko_ai::ReasoningConfig) -> Self` | Set reasoning. `build` rejects a reasoning configuration for a different provider than the model's. |

`DistillationConfig` has no `Default` implementation. `build` validates the token limit, the reasoning provider, and the targets (at least one agent and one tool, and each agent must be registered). A target tool no agent is configured with is logged as a warning. A summary replaces the tool result only when the response has no refusal, stops with `StopReason::EndTurn` or `StopReason::StopSequence`, and contains non-empty text. Otherwise the raw result is kept:

- A refusal (`GenerationResponse::refusal` from any source, even with accompanying text, or an `arko_ai::Error::Refusal` such as an OpenRouter moderation 403) or a truncated response (`StopReason::is_truncated`, which is `MaxTokens` when the summary reached `max_output_tokens`, or `ContextWindowExceeded`) would likely repeat, so the distiller logs a warning and returns `InterceptedResult::Unchanged`. The failure policy does not retry it.
- Model errors, empty text, and any other stop reason are interceptor errors, which the interception failure policy may retry before the raw result is kept. This includes `StopReason::Other("unknown")`, a response that reported no stop reason (OpenRouter `finish_reason: null`, Anthropic `stop_reason: null`), whose completeness cannot be determined.

A finished, non-empty summary replaces the result even when it is longer than the input. `max_output_tokens` caps generation and does not guarantee compression. With an OpenAI API key and with Anthropic, reasoning (thinking) tokens count against `max_output_tokens`, and on Anthropic a manual thinking budget must stay below it unless interleaved thinking is enabled. A limit too small for the model's reasoning yields truncated summaries, which are charged but not used. ChatGPT OAuth requests send no output limit.

Each distillation charges its billed amount, `response.usage.billed()` (zero for subscription calls such as ChatGPT OAuth), as a **manual workflow cost** with the description `distillation: <name>` and agent, task, and tool metadata. The charge counts toward the workflow budget but not toward the agent's own LLM spend. Warning and exceeded hooks fire as applicable, and the managed driver enforces an exceeded workflow budget. Every response the model returns is charged and recorded, including refused, truncated, and empty ones, since they are billed too. Each retry of an interceptor error is another billed call. The distillation record is added after the budget hooks succeed.

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

A record identifies the task, the agent (by registration name), the tool, and the configuration, with the UTC time, input and output sizes in Unicode scalar values, and the billed dollar cost. `output_chars` counts the text the model generated, joined as a summary is (text parts separated by newlines), which is the message text plus any `Refusal::partial_output` that a safety-classifier or content-filter refusal moved out of the message. A refusal's explanation is not counted. `rejection` is `None` when the summary replaced the result and otherwise gives the reason it was not used (the refusal, the stop reason, missing text, or a request skipped because it was over the distiller model's input limit, with `output_chars` and `cost` zero). It is omitted from JSON when `None` and loads as `None` when absent. Records are available in `SharedContext::distillation_records` and in checkpoints.

## Budget and cost accounting

Defined in [budget/mod.rs](src/budget/mod.rs). All six public types are in the prelude. Amounts, limits, thresholds, and returned totals are in dollars. Running workflow and per-agent totals are kept in whole micro-dollars (amounts are truncated). Ledger entries keep the original `f64` amounts, so a ledger sum can differ slightly from the running total.

### `BudgetManager`

Thread-safe workflow spend, per-agent LLM spend and limits, and a detailed ledger. Fields are private, and there is no `Default` implementation. The manager records costs and reports thresholds. It neither cancels agents nor fires hooks. The managed workflow driver does both.

| Signature | Parameters, return, and behavior |
| --- | --- |
| `pub fn new(limit: Option<f64>, warning: Option<f64>) -> Self` | Zero spend, an empty ledger, and no agent limits. `None` disables the corresponding workflow threshold. |
| `pub fn from_ledger(limit: Option<f64>, warning: Option<f64>, ledger: CostLedger) -> Self` | Restore totals from ledger entries, rebuilding per-agent spend from `CostSource::Agent` entries. A warning threshold already reached counts as fired. Agent limits start empty. |
| `pub fn add_agent_cost(&self, agent_name: &str, usage: &arko_ai::UsageInfo) -> AddCostResult` | Add an agent entry for `usage.billed()` (zero for subscription calls such as ChatGPT OAuth, whose estimated cost stays in the stored usage), with a copy of the usage, the current time, and no metadata. Update workflow and agent spend. |
| `pub fn with_exceeded_policy(self, policy: BudgetExceededPolicy, wait_for_raise: bool) -> Self`, `exceeded_policy(&self)`, `waits_for_raise(&self)` | The policy over the limits (default `Pause`, no waiting for a raise in runs the host started). The builder sets it. |
| `pub fn set_limit(&self, limit: Option<f64>)` | Change the workflow limit while agents run (see `Workflow::set_budget`). A limit that brings spend within it lets the next crossing be reported. Wakes the runs waiting for a raise. |
| `pub fn set_agent_limit(&self, agent_name: impl Into<String>, limit: f64)`, `clear_agent_limit(&self, agent_name: &str)` | Set, replace, or remove the agent's limit. Wakes the runs waiting for a raise. |
| `pub fn agent_limits(&self) -> HashMap<String, f64>` | The agents' limits. |
| `pub fn hold_reason(&self, agent_name: &str) -> Option<String>` | Under `Pause`, why a run of the agent is held (the workflow's or its own limit is exceeded, or a plan budget window). `None` under `Cancel` or within budget. |
| `pub fn record_plan_usage(&self, usage: &arko_ai::PlanUsage) -> Vec<PlanBudgetCrossing>`, `set_plan_limits(&self, limits: PlanBudget)`, `plan_limits(&self)`, `plan_spent(&self) -> PlanUsageSpent`, `plan_spent_of(&self, plan: arko_ai::SubscriptionPlan) -> PlanUsageSpent`, `plan_exceeded(&self) -> Option<PlanBudgetCrossing>`, `plan_state(&self) -> PlanBudgetState` | The [plan budget](#plan-budgets). Take a report, counted with the windows of the plan it names (returns the windows it took over their limits, each once per crossing), set the limits (wakes the runs waiting for a raise), and read the limits, the points used (the most any plan has used, or one plan's), a window over its limit (`plan`, `window`, `used`, `limit`, and `describe()` gives `Claude Code five-hour window 25.4 of 25 points`), and the state a checkpoint records. |
| `pub fn agent_limit(&self, agent_name: &str) -> Option<f64>` | The agent's limit, if set. |
| `pub fn agent_spent(&self, agent_name: &str) -> f64` | Recorded agent LLM spend. Zero for an unknown name. |
| `pub fn is_agent_exceeded(&self, agent_name: &str) -> bool` | Whether agent spend is strictly greater than its limit. |
| `pub fn add_manual_cost(&self, amount: f64, description: &str, metadata: Option<String>) -> AddCostResult` | Add a manual entry with the current time and optional free-form metadata. Update workflow spend only. |
| `pub fn total_spent(&self) -> f64` | Running workflow spend. |
| `pub fn is_exceeded(&self) -> bool` | Whether workflow spend is strictly greater than the limit. `false` without a limit. |
| `pub fn check_warning(&self) -> bool` | `true` once, the first time spend is at or above the warning threshold. The call consumes the notification. |
| `pub fn remaining(&self) -> Option<f64>` | `limit - spent`, floored at zero. `None` without a limit. |
| `pub fn limit(&self) -> Option<f64>` | The workflow limit, as it is now. |
| `pub fn snapshot_ledger(&self) -> CostLedger` | Clone of the ledger. |
| `pub fn ledger_len(&self) -> usize` | The number of ledger entries, one more per cost added (agent or manual), never fewer while the workflow runs. A host that derives figures from the ledger recomputes them when it changes instead of cloning the ledger on every read. |
| `pub fn totals_by_agent(&self) -> BTreeMap<String, CostTotals>` | Each agent's running totals (billed, at list prices, tokens, calls), kept as costs are added. |

The recording methods keep a cost even when it exceeds the budget, and their result consumes a newly reached warning. The manager does not validate its inputs. Supply finite, nonnegative values. `WorkflowBuilder` validates configured limits and thresholds, and `Workflow::add_manual_cost` validates the amount, fires the budget hooks, and returns `WorkflowError::BudgetExceeded`.

### Cost data types

`CostSource` implements `Debug`, `Clone`, and Serde:

```rust
pub enum CostSource {
    Agent { agent_name: String, usage: Box<arko_ai::UsageInfo> },
    Manual { description: String },
}
```

`Agent` keeps the agent's registration name and full usage breakdown, including the estimated cost of subscription calls whose billed `amount` is zero. `Manual` keeps a free-form description. Distillation charges are manual entries.

`CostEntry` implements `Debug`, `Clone`, and Serde. Its public fields are these:

```rust
pub amount: f64;
pub source: CostSource;
pub timestamp: chrono::DateTime<chrono::Utc>;
pub metadata: Option<String>;
```

`amount` is the dollar charge as recorded. `timestamp` is when it was recorded. `metadata` is free-form accounting context with no prescribed schema.

`CostLedger` implements `Debug`, `Clone`, `Default` (empty), and Serde. Its one public field, `entries: Vec<CostEntry>`, is in insertion order. `pub fn total(&self) -> f64` sums the entry amounts.

`AddCostResult` implements `Debug` and `Clone`. Its public fields are these:

| Field | Type | Meaning |
| --- | --- | --- |
| `new_total` | `f64` | Running workflow total after the addition. |
| `warning_triggered` | `bool` | This addition reached the warning threshold for the first time. |
| `exceeded` | `bool` | The workflow limit is exceeded after this addition (not only on the first crossing). |
| `exceeded_crossed` | `bool` | This addition exceeded the limit, the first over it since spend was last within it (a raise can bring it back within). |

### Budget holds

Budgets are soft limits. Spending is only known after a model request completes, so the request that crosses a limit is always paid for, and a run can end slightly above its budget. The limit stops the requests after it.

`BudgetExceededPolicy` (`Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Default`, Serde in snake case) has two variants, set with `WorkflowBuilder::budget_exceeded_policy`:

- Under `Pause` (the default) nothing is cancelled. Before each model request, an agent whose workflow or own budget is exceeded is held. The request isn't made and the task pauses for error recovery with `arko_agents::ErrorRecoveryCause::BudgetExceeded` (whatever its `llm_error_policy`, see `TurnContextSource::hold_request` in arko_agents). Tool calls it is running finish first. A run inside another agent's tool call (a delegated run or a completion verifier) waits for a raise, with status `waiting (budget_exceeded)`. So does a run the host started under `wait_for_budget_raise(true)`. Otherwise the host's call returns `WorkflowError::AgentPaused` with `AgentPause::ErrorRecovery { cause: BudgetExceeded, .. }` and the run is recorded `Paused` (`OnAgentPaused` fires). `set_budget` or `set_agent_budget` releases waiting runs at once (they continue without a note to the model). A paused run continues with `resume_agent(.., AgentResume::ContinueAfterError { note: None })` once the budget is raised, and is held again at once if it is still exceeded. The error recovery policy doesn't decide these pauses, and `Limited` doesn't count them. A stop of the run ends a wait. New runs start and are held at their first request. `OnBudgetExceeded` fires once per crossing of the workflow limit. Spending can go past the limit by the requests in flight when it is reached (at most about one per running agent, since a charge is processed as the agent reports it). A Claude Code agent is asked before each turn and at each message of a turn. A turn it holds is interrupted and continues when released (the session resumes).
- Under `Cancel`, each driven agent is cancelled at its next event once the workflow budget is exceeded (or at its usage report once its own is), its call returns `BudgetExceeded` or `AgentBudgetExceeded`, and later run calls are refused while the budget stays exceeded. `OnBudgetExceeded` fires for every charge over the limit. `Workflow::reopen_agent` continues a cancelled run once the budget is raised.

A host that can raise the budget while the run is live calls `set_budget` (from another task or a UI). One that can't (a command-line tool, say) stops when the budget blocks the run. It waits for `budget_blocked()` alongside its run call, or gets `AgentPaused` with the `BudgetExceeded` cause, saves a checkpoint (`Workflow::checkpoint`, with the run call still alive so its agents are recorded as they are) and exits. Resumed with a higher budget, `recover_agent` continues the runs that were running (a delegated run held on the budget replays its pause and continues from it) and `resume_agent` continues the paused ones. The forge example does this.

#### Limits on resume

Checkpoints record the limits in force (`WorkflowState::budget_limits`), including changes made live. On `resume_from`, for the workflow's limit and each agent's, the builder's limit applies unless the recorded one is higher. A limit the builder doesn't set is none, whatever was recorded, and a checkpoint without `budget_limits` (an earlier version's) leaves the builder's. So resuming with a higher budget raises it, and resuming with the limit the run was built with keeps a raise made while it was live. A lowering is not kept.

### Plan budgets

A plan budget limits how much of subscription plans' usage windows a run uses, in percentage points. `WorkflowBuilder::plan_budget(PlanBudget::none().five_hour(25.0).seven_day(5.0))` lets the run use 25 points of the five-hour window and 5 of the seven-day window. The points come from the plan usage agents report (`AgentEvent::PlanUsageReported`, from Claude Code's `rate_limit_event` and ChatGPT OAuth's `x-codex-*` headers) and count from each window's level in the first report the run sees. The points used are the latest level minus that baseline, plus the points of earlier periods. A window is told by its reported length (300 minutes for the five-hour window and 10,080 for the seven-day one, and another length isn't counted) or, without one, by position (the primary window is the five-hour one, the secondary the seven-day one). When a window resets (its reset time moves more than a minute later, or, without reset times, its level drops), the points used before the reset are kept and the new period counts from 0. A lower level with the same reset time is an earlier report arriving late, and is ignored.

Each plan is counted on its own. A report names its plan (`PlanUsage::plan`, `SubscriptionPlan::ChatGpt` for ChatGPT OAuth, `SubscriptionPlan::ClaudeCode` for Claude Code), so a run with agents on both counts the ChatGPT plan's windows and the Claude Code plan's windows separately, each from its own first report. `five_hour` and `seven_day` limit each plan's window of that length. A plan's own limits (`PlanBudget::chatgpt`, `PlanBudget::claude_code`, each a `PlanLimits { five_hour, seven_day }`) replace them for that plan's windows:

```rust
// ChatGPT: 25 points of its seven-day window; Claude Code: 85 points of its
// five-hour window. The dollar budget counts only what is billed:
// subscription calls (ChatGPT OAuth, Claude Code without metered billing)
// count as $0.
let workflow = Workflow::builder()
    .budget(35.0)
    .plan_budget(
        PlanBudget::none()
            .chatgpt(PlanLimits::none().seven_day(25.0))
            .claude_code(PlanLimits::none().five_hour(85.0)),
    )
    .build()?;
```

`PlanBudget::limit_for(plan, window)` gives the limit that applies to a plan's window. `Workflow::plan_usage_spent_of(plan)` gives one plan's points, and `plan_usage_spent()` the most any plan has used of each window, which for a run on one plan is that plan's.

Plan usage is account-wide. Reports from every agent update the same windows of their plan, the latest winning, and other use of the same plan during the run (another run, an interactive session) counts too. Like a dollar budget it is a soft limit, because the points are only known after a request completes, so the requests in flight when a limit is reached complete and can take the run past it.

A window over its limit, of any plan, is handled as an exceeded dollar budget under the same `BudgetExceededPolicy`, for all the run's agents whatever their plan. Under `Pause` each agent is held before its next request (the pause's error names the plan and window, for example `Claude Code five-hour window 85.4 of 85 points`), a nested run waits, `budget_blocked()` covers it, and `set_plan_budget` releases the held runs. Under `Cancel` the agent is cancelled and later runs are refused, with `WorkflowError::PlanBudgetExceeded`. `OnBudgetExceeded` fires once per crossing with the plan and window in its metadata (see [HookPoint variants](#hookpoint-variants)).

Checkpoints record the limits and each plan's counting (`WorkflowState::plan_budget`), so a resumed run keeps counting from the levels the run first saw, not from the levels at the resume. On `resume_from`, each limit (each window's, and each plan's own) follows the dollar rule above, the builder's unless the recorded one is higher. None when the builder sets none. A checkpoint without `plan_budget` (an earlier version's) leaves the builder's and counts afresh. A checkpoint written before plans were counted apart holds counting without a plan. It goes on under the general limits, and the first plan to report after the resume takes it over, whichever plan that is, so such a run is resumed with its agents on the plan it used.

## Durable steps

Defined in [durable/mod.rs](src/durable/mod.rs). Module `arko_workflows::durable`. The prelude re-exports every public type except `DurableStore`. Durable steps replay recorded outputs of explicitly identified operations and hold uncertain outcomes for reconciliation. The `Workflow` methods require configured persistence. `DurableStore` can also be constructed directly.

In the signatures below, `Serialize` is `serde::Serialize`, `DeserializeOwned` is `serde::de::DeserializeOwned`, and `Future` is `std::future::Future`.

### Step specifications, execution context, and reconciliation

| Type | Public fields or variants | Purpose and defaults |
|---|---|---|
| `DurableStepSpec` | `key: String`, `instance: Option<String>`, `version: String`, `retry_policy: DurableRetryPolicy` | Stable operation definition. `new(key)` sets `instance = None`, `version = "v1"`, and `retry_policy = Never`. Key plus instance identify the step. The version and the hash of the serialized input must match the recorded definition. |
| `DurableRetryPolicy` | `Never`. `WhenDefinitelyNotCompleted { max_attempts: u32 }`. `RetrySafe { max_attempts: u32 }` | Default `Never`. `WhenDefinitelyNotCompleted` retries after a resolver confirms the operation did not complete. `RetrySafe` retries an uncertain operation without reconciliation. `max_attempts` includes the first attempt and is treated as at least one. |
| `DurableStepStatus` | `Started`, `Completed`, `InDoubt`, `PermanentlyFailed` | Journal states. A `Started` step without a later record is treated as in doubt on replay. |
| `DurableStepContext` | `workflow_run_id: Uuid`, `step_id: Uuid`, `operation_id: String`, `attempt: u32` | Closure input. The IDs are stable across retries. `attempt` starts at 1. Use `operation_id` as an idempotency or receipt-lookup key with external services. |
| `InDoubtContext` | `workflow_run_id: Uuid`, `step_id: Uuid`, `operation_id: String`, `attempt: u32`, `reason: Option<String>` | Details of an uncertain attempt for resolvers and operators. |
| `InDoubtResolution<O>` | `Completed(O)`. `DefinitelyNotCompleted`. `StillInDoubt { reason: String }`. `PermanentlyFailed { reason: String }` | They record the recovered output, allow a retry the policy permits, keep the step in doubt with a reason, and record a permanent failure, in that order. |

`DurableStepSpec`, `DurableRetryPolicy`, and `DurableStepStatus` implement `Clone` and Serde. The retry policy and status implement `PartialEq` and `Eq`, and the status also implements `Copy`. Both context types implement `Clone`.

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

`instance`, `version`, and `retry_policy` set the matching field. `display_key` returns `key` or `key[instance]`. `operation_id()` borrows the operation ID. Implement `ResolveInDoubt` to check an external system using the attempt context and the original input. Resolver errors propagate.

A spec is validated whenever a step executes or an operator resolution is applied. The key and version must contain non-whitespace characters, and the key and instance must not contain newline, carriage return, or NUL. An invalid spec returns `BuilderInvalid`. `None` and `Some("")` instances are the same identity. Reusing an identity with a different input or version returns `DurableDefinitionMismatch`. Give a separate operation its own key or instance.

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
| `operator_permanently_fail` | Record a permanent failure. Later execution returns `DurablePermanentlyFailed`. |
| `operator_authorize_retry` | Allow the next execution call to retry, even with `Never`. A configured attempt limit still applies. Does not run the closure. |

The operator methods require an unresolved attempt with a matching version and input, and return `DurableDefinitionMismatch` otherwise. They append records carrying `actor` and `reason` and never rewrite earlier records. They take no execution lease, so resolve steps only after execution has stopped.

A retry happens on a later call. Each call runs the closure at most once. When the configured attempts are exhausted, execution returns `DurableAttemptsExhausted`. Persistence and serialization errors propagate. A completed step replays without running its closure, but external side effects still need idempotency or reconciliation. Executing an identity that is already running on the same store returns `DurableInDoubt`. Locking and active-step protection are per store instance. There is no cross-process lock. Reading the journal tolerates an invalid final record but rejects an invalid earlier one.

## Persistence and checkpoint state

Defined in [persistence/mod.rs](src/persistence/mod.rs), [persistence/state.rs](src/persistence/state.rs). The configuration and manager are in `arko_workflows::persistence`, and the state types in `arko_workflows::persistence::state`. The prelude re-exports `PersistenceConfig`, `CheckpointRetention` and `EventLog`.

### PersistenceConfig and PersistenceManager

`PersistenceConfig` implements `Debug` and `Clone`. It has no `Default` implementation. `new` sets the defaults below. Build it with `new` and the setters (its public fields can be read and assigned afterwards). It has a private field, so a struct literal doesn't compile outside the crate. `PersistenceManager` implements `Clone` and has private fields. Clones share the durable store, write lock, event log, and task journal.

| Public field | Type | Meaning and default from `new` |
|---|---|---|
| `checkpoint_dir` | `std::path::PathBuf` | The run directory. |
| `auto_checkpoint` | `bool` | Save a checkpoint after each finalized agent result, and once when a run call first returns a stopped run's error. Default `true`. |
| `retain_checkpoints` | `CheckpointRetention` | `KeepAll` (the default) keeps every checkpoint and task version. `KeepLast(n)` (`n: NonZeroUsize`, set with `keep_last_checkpoints`) keeps only the newest `n` checkpoint files are kept, deleting older ones (earlier processes' first, by modification time) after each save, and then the task versions that neither a kept checkpoint nor a recorded result refers to (none if a kept checkpoint can't be read). A file that can't be deleted is logged and kept. |
| `event_log` | `EventLog` | `Full` (the default) logs every agent event, including each streamed fragment. `LifecycleOnly` logs every event except `AssistantResponseChunk`, `ReasoningChunk`, `ToolParamName`, `ToolParamValueChunk`, and `ToolParamEnd`. The finished responses are in the task files, but streamed reasoning a provider doesn't return with the response is only in a `Full` log. |
| `event_flush_interval` | `std::time::Duration` | Buffered events are written at least this often. `TaskStateChanged`, `TaskStreamEnd`, `ErrorEncountered`, `RetryScheduled`, `UsageLimitReached`, `RefusalReceived`, `OutputTokenLimitReached`, and `ConversationHandedOver` are written at once, and the buffer before each checkpoint. Default `DEFAULT_EVENT_FLUSH_INTERVAL`, one second. |
| `task_journal` | `bool` | Save running agents as they run, each running task in `agents/<task id>/live.json` and the workflow's state in `live-checkpoint.json`, rewritten at the task's safe points, so that runs running at a crash are recovered where they were. Default `true`. See [Task journal and recovery](#task-journal-and-recovery). |
| `pretty_json` | `bool` | Pretty-print the task journal's files (`live.json`, `live-checkpoint.json`). Compact JSON (`false`) by default. Checkpoints and task versions are always pretty-printed. |

```rust
impl PersistenceConfig {
    pub fn new(checkpoint_dir: impl Into<std::path::PathBuf>) -> Self;
    pub fn auto_checkpoint(self, enabled: bool) -> Self;
    pub fn keep_last_checkpoints(self, count: usize) -> Self; // 0 fails `WorkflowBuilder::build`
    pub fn event_log(self, event_log: EventLog) -> Self;
    pub fn event_flush_interval(self, interval: std::time::Duration) -> Self;
    pub fn task_journal(self, enabled: bool) -> Self; // default true
    pub fn pretty_json(self, enabled: bool) -> Self; // default false
    pub fn user_data(workflow_name: &str) -> Result<Self, WorkflowError>;
}
pub const LIVE_CHECKPOINT_FILE: &str = "live-checkpoint.json";
impl PersistenceManager {
    pub fn resolve_run_id(config: &PersistenceConfig) -> Result<uuid::Uuid, WorkflowError>;
    pub fn new(config: PersistenceConfig, workflow_run_id: uuid::Uuid)
        -> Result<Self, WorkflowError>; // BuilderInvalid for keep_last_checkpoints(0)
    pub fn config(&self) -> &PersistenceConfig;
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
| `PersistenceManager::new` | Associate a configuration with a run identity. When reopening a run, pass the identity from `resolve_run_id`. Returns `BuilderInvalid`, as `WorkflowBuilder::build` does, for a configuration that asked to keep the last zero checkpoints. |
| `config` | The configuration it persists with. |
| `durable_store` | Shared handle to the run's durable-step store. |
| `save_checkpoint` | Write the buffered events, then snapshot `ctx` into a new pretty-printed `checkpoint-<UUID>.json`, point `latest-checkpoint` at its file name, apply `retain_checkpoints`, and return its path. Each agent result refers to the version of its task the checkpoint saw (see [Task versions](#task-versions)). A version not saved yet (a task restored from a checkpoint that held it, or one whose save failed) is written first. With the task journal, each agent run still running is recorded `Interrupted`, referring to a version of its latest saved snapshot. If that write fails, the checkpoint is still written. It refers to the task's last saved version (leaving out a run never saved), lists the run in `missing_task_states`, and logs a warning naming it. The next checkpoint tries again. Only a failure to write the checkpoint file itself returns an error. |
| `append_event` | Record `event` in `events.jsonl` as `event_log` selects, with the current UTC time, the workflow run UUID, `agent_name`, and `agent_run_id`, buffered, and written as described for `event_flush_interval`. The file stays open between writes. A failed write is logged as a warning once until writing succeeds again, and the events stay buffered (the oldest are dropped past 64 MiB, with a warning). It never stops an agent. Dropping the last manager writes what is still buffered. |
| `flush_events` | Write the buffered events now. |
| `save_agent_task` | Save `task` as a version, `agents/<task.id>/<content hash>.json`, unless that version exists already, and return its path. It doesn't change which versions checkpoints refer to. |
| `auto_checkpoint` | Save a checkpoint only when `config.auto_checkpoint` is `true`. Otherwise succeed without writing. |
| `load_checkpoint` | Read and deserialize the given checkpoint file. Does not follow `latest-checkpoint` or restore a workflow. Use `WorkflowBuilder::resume_from` for that. `WorkflowState::agent_results_in(run_dir)` returns its agent results with their tasks, read from the task files. |

Filesystem and serialization failures are returned as `WorkflowError`, except that saving a finalized agent's task, the automatic checkpoint after it, and the task journal's saves never fail the agent's run. A failure is logged as a warning and the next save tries again. A run directory holds checkpoints, task versions, the task journal's files, the event log, the durable journal, and, with `control_folder`, the [control folder](#the-control-folder). Running futures and ballots are not persisted. The event log also records each posted note, in a line of its own (see [Operator notes](#operator-notes)).

Every file of the run directory except the append-only logs (`events.jsonl`, the durable journal) and `workflow-run-id` (created once, exclusively) is replaced atomically. It is written to a temporary file in the same directory (`<name>.<process id>-<n>.tmp`), synced, and renamed over the file, so a crash leaves the old content or the new. On Unix the directory is synced after the rename (except for the task journal's background saves, see [Task journal and recovery](#task-journal-and-recovery)), and a directory created for a file is synced into its parent. On Windows, creating the temporary file or the rename, refused because another program has the file open (an antivirus scanner or indexer, giving access denied or a sharing violation), is tried again a few times. The temporary file is removed when a write fails, and a manager opening the run directory removes those an earlier process left.

### Task versions

A checkpoint doesn't hold the tasks of its agent results. It refers to task version files, `agents/<task id>/<content hash>.json`, each a task as the checkpoint recorded it. When a finalized run is recorded its task is saved as a version. A version is written only when the task's content changed, so checkpoints that saw the same task share one file, and a task that runs on (continued or resumed) gets a new version while older checkpoints still refer to, and resume, the task as it was then. Under `CheckpointRetention::KeepAll` (the default) every version is kept, so the run directory holds one full task per version. Under `KeepLast`, a version is deleted once no kept checkpoint (and no recorded result) refers to it. Task files that earlier versions wrote beside checkpoints holding their tasks (`agents/<task id>.json`) are not referred to or deleted.

#### Images

The images in what a run directory saves (the conversations of task versions and live files, the notes, any image in a checkpoint or an event) are saved once each, as files in `images/`, named `<SHA-256 of the image's bytes>.<type>` (`.b64` holding the text as given for data that isn't standard base64). The JSON holds `{"Base64": {"stored": "<name>", "mime_type": ...}}` in place of the data (see `arko_ai::image_store`), so a conversation saved at every step doesn't rewrite its screenshots. A file is written atomically and flushed before the JSON naming it, and never removed while the run directory exists, since older checkpoints, task versions and the event log may name it. `persistence::images::ImageFiles::in_run_dir(dir)` opens a run directory's images, shared within the process, and `ImageFiles::with(f)` runs `f` with them as the image store. A host that reads task files itself wraps its reads in it. `PersistenceManager::load_checkpoint`, `WorkflowState::agent_results_in` and `resume_from` do so already. To recognize images it saved without hashing them again, the store keeps the latest ones in memory, up to 256 MiB, and compares an image in full before it reuses a name. A file is read back only when its content still matches its name. An image whose file is gone or damaged doesn't keep a task from loading. It becomes the line `(image no longer available: REASON)` in its place (`arko_agents::Task::settle_unavailable_images`), the resume logs a warning, and the run reports it with `AgentEvent::ImagesUnavailable` when it goes on. Versions of the framework from before image files read JSON naming a stored image as an error (`missing field data`), so a session saved with images needs a build with image files to resume.

`resume_from` restores memory content and history, pending messages, the cost ledger, finalized agent results, the runs that were running (as `Interrupted` results), verification and distillation records, the agent status registry, and the [operator notes](#operator-notes). For memory, the new builder's `MemoryConfig` decides each tier's readers, writers, limit, and write mode, and the checkpoint supplies content and history by tier name. A configured tier the checkpoint lacks starts as configured. A checkpoint tier that matches no configured tier makes `build` fail with `BuilderInvalid` naming it, unless `rename_memory_tier_on_resume(from, to)` maps it to a configured tier or `drop_memory_tier_on_resume(name)` discards it. Two checkpoint tiers mapped to one configured tier also fail. Content over a tier's configured limit is kept (with a warning). Writes that grow it fail until it is smaller. When the write mode changed, each write's text is stored in its entry. Freshness state is restored only when freshness tracking is configured again. Agents, the client, tools, hooks, policies, and metadata come from the new builder, not the checkpoint. So do budget limits, except that a higher limit the checkpoint recorded (a raise made while the run was live) is kept (see [Budget holds](#budget-holds)). The pause flag starts cleared, and active message channels, ballots, and verification repair counts and baselines start empty. Restoring does not continue agents (`recover_agent` continues the interrupted ones) or replay workflow control flow. Accounting is rebuilt from `cost_ledger`. `budget_spent` and `completed_agents` are not used on restore. A recorded `cancellation` isn't restored, and the resumed run starts again, as described under [Stopping a run](#stopping-a-run).

No agent is running after a resume. From `agent_registry`, an agent that was running when the checkpoint was saved counts as finished (interrupted), and each continuable run keeps the registry version it has seen, so a continued run is told only about recipients that started after that. From a checkpoint without `agent_registry`, an agent with recorded results takes its latest result's outcome, and any other agent is not started. The runs that were running are restored as `Interrupted` results with their tasks, to be continued with `recover_agent` (see [Task journal and recovery](#task-journal-and-recovery)).

### Task journal and recovery

With `task_journal` on (the default whenever persistence is configured), the workflow saves each agent run it starts while it runs, through the agent runtime's `TaskSnapshotSink` (see the arko_agents guide). The task is rewritten whole to `agents/<task id>/live.json` at its safe points (after each model turn, before a group of tool calls runs, after their results, at pauses and handovers, and at its end), followed by `live-checkpoint.json`, a checkpoint of the whole workflow state (memory, messages, cost ledger, results, registry) that records each running run `Interrupted`, referring to its live file. Before a group of tool calls runs, both files are on disk (their directory entries flushed too) before the calls start, so a call with effects never runs without a saved record of it (`Task::running_tool_calls`). A group whose calls recovery simply runs again (every tool `ToolRecovery::Rerun`, such as `read_file` or `list_files`, and, for a Claude Code agent, the read-only built-ins of `claude_code::READ_ONLY_BUILT_IN_TOOLS` such as `Read`, `Glob` and `Grep`) doesn't wait, because a crash before its save has the calls run, or asked for, again. The other saves are written in the background, a newer snapshot replacing one not written yet, with the files flushed but not their directory (a crash of the machine may keep the previous file, still whole). The live checkpoint is written by one writer at a time with the state as it is when it starts, and only when a resume would read something new from it. A running task's progress alone is in its live file, and so is the usage its model calls reported (a resume charges what the checkpoint's ledger lacks), so a step that changes nothing shared (memory, messages, results, records, other costs) writes only the live file. `latest-checkpoint` names the newest checkpoint written, live or versioned, which is `live-checkpoint.json` while agents run, the versioned checkpoint written as a run ends (with `auto_checkpoint`), and the live checkpoint recording the end when no versioned one follows. Each versioned checkpoint also records the runs still running, referring to a version of each one's latest snapshot. A finished run's live file is deleted once the live checkpoint no longer refers to it (the versioned checkpoint recording its end refers to its task version). A run restored from its live file is saved as a version by the next checkpoint. Until one succeeds, checkpoints refer to the live file, which is kept. When a manager opens a run directory, it removes what an earlier process left unfinished, such as temporary files of interrupted writes, and live files no checkpoint refers to. Retention never counts or deletes `live-checkpoint.json` or live files, and keeps the task versions the live checkpoint refers to. A run driven through `AgentHandle::into_parts`, or dropped unrecorded, is no longer saved. Live files and the live checkpoint are compact JSON unless `pretty_json` is set. With `task_journal(false)` nothing is written while agents run. A run running at a crash is then lost (its agent counts as interrupted) and starts again.

A save before calls with effects costs a synced write of the task file and its directory, plus the live checkpoint's when something shared changed, next to the model's latency. The live checkpoint holds the whole cost ledger, which grows with each model call, and is serialized off the async threads. A run the workflow couldn't save is logged as a warning, and the checkpoint refers to its last saved snapshot (listed in `missing_task_states`). The run goes on.

To resume after a crash, build the workflow again with the same configuration, persistence and `resume_from(<run dir>/<latest-checkpoint>)`, then continue the runs of `interrupted_runs()` with `recover_agent`:

```rust
let workflow = builder.persistence(config).resume_from(latest).build()?;
for run in workflow.interrupted_runs() {
    let result = workflow.recover_agent(&run).await?;
}
```

A recovered run keeps its identity, conversation and costs, and gets no new message. Calls it was running are settled by their tools' `Tool::recovery_mode`. `InDoubt` (the default) answers the call that the task restarted while it ran, so it may have run partly or fully and its effects should be checked. `Rerun` runs it again (read-only built-ins such as `read_file`, `list_files`, `load_skill`, and `complete_task`, whose completion verifiers review it again). `Resume` runs it again knowing it is a recovery (`run_workflow_agent`, which recovers the run it started, see [Workflow services](#workflow-services-and-custom-tools)). A request the model was streaming is sent again. A wait the task was saved in (a tool approval, an error-recovery pause, or a question) is answered again under the workflow's policies and handlers. Runs started inside a tool call their parent still holds open are recovered with it and are not listed by `interrupted_runs`. `recover_agent` rejects them. When a run is recovered, the interrupted runs inside its open calls that it doesn't resume (a completion verifier that was reviewing its `complete_task` call, or a run inside a `run_workflow_agent` call settled in doubt because the tool is no longer offered to it, or answered with the host's own output) are recorded `Cancelled` ("Superseded: ..."), with every interrupted run inside theirs, and its completion is reviewed again by a new review. `build` doesn't check the configuration of runs that will be superseded this way (nobody can run them again).

No nested run is orphaned silently. When a parent can't recover a run it started (its `run_workflow_agent` call failed, for example over a budget), the call's result names the run, and the parent can resume it with `resume_run` (see [Workflow services](#workflow-services-and-custom-tools)). The run stays `Interrupted`, and is not superseded when the parent is recovered again. Once the parent can no longer resume it (the parent ended, or it answered the call and stopped), the run is orphaned. `interrupted_runs` lists it with its `task.parent` still set, and `recover_agent` recovers it as the host's own, releasing the link. Recovering a run that is being recovered already fails for the host ("is running already") and is waited for by a parent's call. Either way it runs once. Usage a run reported that the checkpoint's ledger lacks (recorded in `AgentRunRecord::usage_charged`) is charged when the workflow is built, so each model call is charged once. State outside checkpoints starts empty, as for any resume, such as verification repair counts and baselines, approval reasons, ballots, and the pause flag.

After a graceful stop, `stopped_runs()` lists the interrupted runs together with the runs the stop ended, and `restart` relaunches each. See [Graceful stop](#graceful-stop).

### Checkpoint payloads

`WorkflowState` implements Serde and exposes every field below. It has no `Default` implementation. The last column gives the value used when a field is missing from the checkpoint JSON.

| Field | Type | Purpose. Value when missing |
|---|---|---|
| `workflow_run_id` | `uuid::Uuid` | Run identity. Nil UUID. |
| `workflow_name` | `String` | Workflow name. Required. |
| `workflow_metadata` | `crate::workflow::config::WorkflowMetadata` | Description, platform, and dependency metadata. `WorkflowMetadata::default()`. |
| `checkpoint_id` | `uuid::Uuid` | Snapshot ID. Required. |
| `timestamp` | `chrono::DateTime<chrono::Utc>` | Snapshot time. Required. |
| `memory_snapshot` | `Vec<crate::memory::MemoryTier>` | Memory tiers and entries. Required. |
| `message_queues` | `std::collections::HashMap<String, Vec<crate::messaging::AgentMessage>>` | Pending messages per agent. Required. |
| `budget_spent` | `f64` | Workflow spend at checkpoint time. Required. |
| `cost_ledger` | `crate::budget::CostLedger` | Complete accounting history. Empty ledger. |
| `freshness_snapshot` | `Option<crate::freshness::FreshnessSnapshot>` | Freshness tracking state. `None`. |
| `agent_results` | `Vec<crate::workflow::result::AgentResult>` | Finalized results with their tasks, as checkpoints written before `agent_runs` held them. Omitted from JSON when empty (always, in new checkpoints). Empty. |
| `agent_runs` | `Vec<AgentRunRecord>` | Finalized results, including failures and pauses, each referring to its task version, and, with the task journal, the runs running when it was written (`Interrupted`). Omitted from JSON when empty. Empty. |
| `missing_task_states` | `Vec<MissingTaskState>` | The runs whose latest task state couldn't be saved. Their records refer to the last saved version, and a run never saved is not in `agent_runs`. Omitted from JSON when empty. Empty. |
| `verification_records` | `Vec<crate::verification::VerificationRecord>` | Completion-verification history. Empty. |
| `distillation_records` | `Vec<crate::distillation::DistillationRecord>` | Distillation history. Empty. |
| `completed_agents` | `Vec<SerializableAgentResult>` | Name and summary of each recorded result. Empty. |
| `cancellation` | `Option<crate::workflow::cancel::Cancellation>` | The run's stop, in checkpoints written after it (omitted from JSON when `None`). `None`. |
| `agent_registry` | `Option<crate::messaging::registry::RegistrySnapshot>` | Agent statuses and what each agent run has seen of them (omitted from JSON when `None`). `None`. |
| `notes` | `Option<crate::notes::NotesSnapshot>` | The [operator notes](#operator-notes) posted, what each run has been given, the runs the one-time notes were skipped for, and how much of the control folder has been read. Written once the run has any of these, omitted from JSON when `None`. `None` (no notes). |
| `plan_budget` | `Option<PlanBudgetState>` | The plan budget's `limits: PlanBudget`, each plan's counting in `plans: BTreeMap<SubscriptionPlan, PlanUsageTracking>` (keys `chatgpt` and `claude_code`, omitted when empty), and `usage: PlanUsageTracking`, the counting without a plan (an earlier version's, until a plan takes it over). A `PlanUsageTracking` has per window `five_hour`/`seven_day` a `PlanWindowUsage { baseline, latest, accumulated, resets_at }`. Written once the run has a plan budget or has seen plan usage. Omitted from JSON when `None`. `None`. See [Plan budgets](#plan-budgets). |
| `subagents` | `Option<crate::subagents::SubagentsSnapshot>` | The [background subagents](#background-subagents), with their records, the mailbox, and the peer message counts, omitted from JSON when `None`. `None` when missing. |
| `budget_limits` | `Option<BudgetLimits>` | The limits in force, `limit: Option<f64>` (no limit when `None`, and omitted then) and `agent_limits: BTreeMap<String, f64>` (omitted when empty). Omitted from JSON when `None`. `None`. See [Budget holds](#budget-holds). |

`RegistrySnapshot` has public fields `version: u64`, which every status change increases. `agents: std::collections::BTreeMap<String, AgentRecord>`, keyed by registration name. And `seen: HashMap<Uuid, u64>`, the registry version each agent run (by task ID) has been told about. `AgentRecord` has `last_start: u64`, the version at the agent's latest start (`0` if it never started), `finished: Option<AgentRunStatus>`, how its latest run ended when none is running (omitted from JSON when `None`), and `last_finish: u64`, the version at which its last running run ended (`0` if it never finished, or in a record saved before finish notices). Every field defaults when missing. Both types implement `Debug`, `Clone`, `Default`, `PartialEq`, `Eq`, and Serde. What a run that cannot be continued (any status other than `Completed` or `WaitingForInput`) has seen is dropped when it ends.

`notes::NotesSnapshot` (`Debug`, `Clone`, `Default`, `PartialEq`, `Eq`, Serde) has public fields `notes: Vec<PostedNote>`, every note posted, oldest first. `seen: BTreeMap<Uuid, u64>`, per agent run (task ID), the number of the newest note it has been given (none when it has no note). `started: BTreeMap<Uuid, u64>`, per run started while there were notes, the number of the newest note when it started (a recovered run's notes are read from its task after it). `skipped: BTreeMap<Uuid, u64>`, per run the host skipped the one-time notes for (`Workflow::skip_one_time_notes`), the number of the newest note then (the one-time notes up to it don't reach it). And `control_offset: u64`, how many bytes of the control folder's notes file have been read. Every field defaults when missing. `seen`, `started` and `skipped` are omitted from JSON when empty and `control_offset` when zero.

Unknown fields are ignored on load, so checkpoints written by earlier versions with fields since removed, such as `oauth_exempt_agents`, still load. Their ledger entries keep their recorded amounts. A task state saved as `Failed` with exactly the reason `Task cancelled by user.` deserializes as `TaskState::Cancelled`, in checkpoints, task files, and event-log records alike. Budget accounting records `UsageInfo::billed()`, which is zero for subscription calls such as ChatGPT OAuth.

`SerializableAgentResult` has public fields `agent_name: String` and `summary: String` and implements Serde, `Clone`, and `Default` (both strings empty).

`AgentRunRecord` (`Debug`, `Clone`, Serde) has public fields `agent_run_id: Uuid`, `agent_name: String`, `status: AgentRunStatus`, `summary: String`, `error: Option<String>` (omitted from JSON when `None`), `task_file: String`, the task version the checkpoint recorded, relative to the run directory (`agents/<agent_run_id>/<content hash>.json`, or for a running run in the live checkpoint its live file `agents/<agent_run_id>/live.json`), `usage_charged: Option<usize>` (omitted from JSON when `None`), for an `Interrupted` run how many entries of its task's `usage_history` the checkpoint's ledger has charged. `None` in older checkpoints and for other runs. And `stopped_by: Option<StopRecord>` (omitted from JSON when `None`), the stop that ended the run (`AgentResult::stopped_by`), or, on a running run's record in the live checkpoint, the stop of that run alone under way (`Workflow::stop_run`), which older versions ignore and older checkpoints lack. A running run's record has the error `The workflow stopped while this agent run was running; Workflow::recover_agent continues it.` and loads `Interrupted` whatever state its task was saved in. `MissingTaskState` (`Debug`, `Clone`, Serde) has public fields `agent_run_id: Uuid`, `agent_name: String`, `status: AgentRunStatus` (of the state that is missing), `last_saved: Option<String>` (the version referred to instead, omitted from JSON when `None`), and `reason: String` (why it could not be saved). `WorkflowState::agent_results_in(&self, run_dir: &Path) -> Result<Vec<AgentResult>, WorkflowError>` returns the results `agent_results` holds, then those of `agent_runs` with their tasks read from `run_dir`. A task file that can't be read or parsed returns `Persistence` naming it. A run in `missing_task_states` is returned as its last saved task records it (its status, summary, and error come from that task), and a run never saved is not returned. Each is logged as a warning saying the checkpoint lacks its latest state. `resume_from` reads the task files from the checkpoint's directory. A checkpoint holding its tasks (from an earlier version) still loads. The next checkpoint writes their task versions and refers to them.

## Structured voting

Defined in [voting/mod.rs](src/voting/mod.rs). Module `arko_workflows::voting`. The prelude re-exports `VoteOption`, `TieBreak`, `BallotConfig`, `CastVote`, and `BallotResult`. To run a ballot, call `WorkflowBuilder::enable_voting` with the agents that vote (only they are offered `cast_vote`), begin a ballot, give its UUID and options to the participating agents, and finish it after they vote.

| Type | Public fields or variants | Purpose and constraints |
|---|---|---|
| `VoteOption` | `id: String`, `description: String` | Machine-readable option ID and a description for agents. IDs must be nonblank and unique within a ballot. |
| `TieBreak` | `RejectTie`, `Lexicographic` | Reject a tie for the top total, or choose the lexicographically first tied option ID. |
| `BallotConfig` | `question: String`, `options: Vec<VoteOption>`, `eligible_voters: std::collections::HashSet<String>`, `quorum: usize`, `consensus_threshold: f64`, `weights: std::collections::HashMap<String, f64>`, `tie_break: TieBreak` | At least two options, a positive quorum, and a finite threshold in `[0, 1]`. An empty eligibility set admits every agent. A voter without a weight counts `1.0`. Weights of actual voters must be finite and positive. No `Default` implementation. |
| `CastVote` | `voter: String`, `task_id: uuid::Uuid`, `option: String`, `reasoning: String`, `timestamp: chrono::DateTime<chrono::Utc>` | A recorded vote. The tool sets the voter, task, and time. The option and reasoning come from its arguments. |
| `BallotResult` | `ballot_id: uuid::Uuid`, `winner: String`, `consensus: f64`, `votes: Vec<CastVote>` | Winning option, its weight divided by the total cast weight, and the recorded votes (unspecified order). |

These types implement `Clone` and Serde. `TieBreak` also implements `Copy`, `PartialEq`, and `Eq`. `BallotStore` has private fields. `Default` creates an empty in-memory store.

```rust
pub const CAST_VOTE_TOOL_NAME: &str = "cast_vote";
impl BallotStore {
    pub fn begin(&self, config: BallotConfig) -> Result<uuid::Uuid, String>;
    pub fn finish(&self, ballot_id: uuid::Uuid) -> Result<BallotResult, String>;
}
```

`begin` validates the options, quorum, and threshold, stores an empty ballot, and returns its UUID. `finish` requires an existing ballot, checks the number of distinct voters against the quorum, validates the weights of cast votes, totals the weighted votes, applies the tie-break, and checks consensus. Its errors name the cause, such as an unknown ballot, quorum not reached, invalid weight, tie, or insufficient consensus. Finishing computes a result without closing or removing the ballot, so eligible agents can still vote. Ballots live in memory and are not checkpointed. The `Workflow` ballot methods return store errors as `WorkflowError::Deliberation`.

### The cast_vote agent tool

`cast_vote(ballot_id: string, option: string, reasoning: string) -> string`. All three arguments are required, and the tool is not batch-safe. `ballot_id` must be a UUID supplied by the workflow. `option` must exactly match a configured option ID. `reasoning` carries the voter's rationale. The tool sets `voter` to the running agent's registration name, `task_id` to its task, and the timestamp to the current UTC time. Only eligible agents may vote, once per ballot per agent name. Success returns `Vote recorded for option '<option>'.`. Missing or non-string arguments, malformed UUIDs, unknown ballots or options, ineligible voters, and repeat votes are errors. The tool type is crate-private and votes can be cast only through it. `enable_voting` gives it to every agent.

## Python bindings

Defined in [python/mod.rs](src/python/mod.rs). It needs the Cargo feature `python`. The Python module `arko_workflows` exports three classes, `PyWorkflowBuilder`, `PyWorkflow`, and `PyAgentResult`, and the exceptions `WorkflowCancelled`, `UserInputRequired`, `AgentPaused`, and `VerifierFailed`, subclasses of `Exception`. `UserInputRequired`, `AgentPaused`, and `VerifierFailed` carry `(message, agent_run_id)` as their arguments. The three class names are public Rust wrapper types in `arko_workflows::python`. Their Python methods are not public Rust methods. `PyAgentResult` implements `From<arko_workflows::AgentResult>`.

The tables use Python-facing signatures, where `self` and PyO3 interpreter arguments are omitted, and awaitable methods are written as `async def`. Builder setters mutate the builder and return `None`, so they cannot be chained like the Rust builder. Numbers map to the underlying Rust `f64` and `usize` types. Graceful stops (`stop`, `stop_run`, `stop_agent`, `stopped_runs`, `restart`, `stop_on_signals`, and `PyAgentResult` stop marks) and operator notes (`post_note`, `notes`, `notes_by_name_only`, `control_folder`) are not exposed. `cancel` is.

### PyWorkflowBuilder

| Python signature | Parameters and behavior |
| --- | --- |
| `PyWorkflowBuilder()` | Empty builder. Name, workspace, and agents are required as in Rust. The builder creates its own LLM client at build time. |
| `name(name: str) -> None` | Set the workflow name. |
| `workspace(path: str) -> None` | Set the existing workspace directory. |
| `agent(name: str, config_path: str) -> None` | Add an agent whose configuration file is loaded during `build`. |
| `budget(limit: float) -> None` | Set the workflow budget in dollars. Validated at build. |
| `budget_exceeded_policy(policy: str, wait_for_raise: bool = False) -> None` | `"pause"` (the default) or `"cancel"`, as `WorkflowBuilder::budget_exceeded_policy`. `wait_for_raise` as `wait_for_budget_raise`. Other values raise `ValueError`. |
| `persistence(checkpoint_dir: str, auto_checkpoint: bool = True, retain_checkpoints: int \| None = None, event_log: str = "full", task_journal: bool = True, pretty_json: bool = False) -> None` | Enable persistence in `checkpoint_dir`, with the Rust defaults, a checkpoint after every finalized agent run, every checkpoint kept (`retain_checkpoints` keeps the newest that many, and 0 raises `ValueError`), every event logged (`"lifecycle_only"` leaves out streamed fragments, and other values raise `ValueError`), and running agents saved as they run in compact JSON. |
| `rename_memory_tier_on_resume(old: str, new: str) -> None` | As `WorkflowBuilder::rename_memory_tier_on_resume`. |
| `drop_memory_tier_on_resume(name: str) -> None` | As `WorkflowBuilder::drop_memory_tier_on_resume`. |
| `approval_policy(policy: str) -> None` | `"approve_all"` (the default) or `"respect_agent_config"`. Other values raise `ValueError`. A handler is Rust-only. |
| `error_recovery_policy(policy: str, max: int \| None = None) -> None` | `"automatic"` (the default), `"limited"` with `max`, or `"return_to_caller"`. Other combinations raise `ValueError`. |
| `max_parallel_agents(max: int) -> None` | As `WorkflowBuilder::max_parallel_agents`. `build()` raises `ValueError` for zero. |
| `fail_fast(enabled: bool) -> None` | As `WorkflowBuilder::fail_fast` (default off). |
| `resume_from(checkpoint: str) -> None` | Restore a checkpoint JSON file at build. Requires `persistence`. |
| `global_memory_tier(name: str, max_bytes: int \| None = None) -> None` | Add an append-only tier every agent can read and write, with an optional byte limit. |
| `cancel_grace_period(seconds: float) -> None` | Seconds a stopped run waits for each running agent to stop before aborting it. Default 30. A negative or non-finite value raises `ValueError`, and so does `build()` for a value too large for a duration. |
| `wait_for_message_timeout(seconds: float \| None = None) -> None` | The longest `wait_for_message` wait in seconds, as `WorkflowBuilder::wait_for_message_timeout`. `None` selects the automatic limit (the default). A non-positive or non-finite value raises `ValueError`. `build()` raises it for more than 24 hours. |
| `agent_wait_for_message_timeout(agent: str, seconds: float \| None = None) -> None` | The same for one agent, in place of the workflow's setting. `None` selects the automatic limit for it. `build()` raises `ValueError` for an unknown agent. |
| `build() -> PyWorkflow` | Create the LLM client, load the agent files, and build the workflow. Load and build failures raise `ValueError`. The builder is not consumed and can be reused. |

`build` creates the client with `LLMClient::from_env()`, which configures every provider whose credentials are in the process environment, ignoring blank values, as OpenAI with ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (plus optional `OPENAI_OAUTH_REFRESH_TOKEN`), otherwise with an OpenAI API key from `OPENAI_API_KEY`. Anthropic from `ANTHROPIC_API_KEY`. Google Gemini from `GOOGLE_API_KEY`, else `GEMINI_API_KEY`. And OpenRouter from `OPENROUTER_API_KEY`. Setting only some of the `OPENAI_OAUTH_*` variables fails the build. When `OPENAI_OAUTH_REFRESH_TOKEN` is the refresh token in the Codex `auth.json` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`), that file is used as the credential store, so refreshed tokens are written back to it. Otherwise refreshed tokens are kept only in memory, and the first refresh logs a warning. The Python API accepts no custom client, custom tools, hooks, approval handlers, verifiers, distillers, or service settings. Those require the Rust API.

### PyWorkflow

Obtained from `PyWorkflowBuilder.build()`. There is no Python constructor.

| Python signature | Parameters, result, and behavior |
| --- | --- |
| `async def run_agent(agent_name: str, prompt: str) -> PyAgentResult` | Run a registered agent with the managed driver. |
| `async def run_agents_parallel(agents: list[tuple[str, str]]) -> list[PyAgentResult]` | Run `(name, prompt)` pairs concurrently. Results in input order when none fails (an agent that paused is in the list, with its status). |
| `async def continue_agent(previous: PyAgentResult, prompt: str) -> PyAgentResult` | As `Workflow::continue_agent`, a follow-up for a completed run, the answer for one waiting for input, or the next message for one whose completion a verifier could not review. |
| `async def resume_agent(previous: PyAgentResult, decision: str, note: str \| None = None) -> PyAgentResult` | As `Workflow::resume_agent` for a paused run, with `"approve"` or `"reject"` every call awaiting approval (`note` is the message or reason), or `"continue"` (`note` is added for the model) or `"fail"` (`note` is the reason) after an error-recovery pause. Other values raise `ValueError`. |
| `agent_result(agent_run_id: str) -> PyAgentResult \| None` | The recorded result of a run, as `UserInputRequired`, `AgentPaused`, and `VerifierFailed` give its ID. A malformed ID raises `ValueError`. |
| `interrupted_runs() -> list[PyAgentResult]` | As `Workflow::interrupted_runs`. After `resume_from`, the runs that were running when the workflow stopped, and orphaned nested runs. |
| `async def recover_agent(previous: PyAgentResult) -> PyAgentResult` | As `Workflow::recover_agent`, continues an interrupted run where it was saved. |
| `async def reopen_agent(previous: PyAgentResult, prompt: str = "") -> PyAgentResult` | As `Workflow::reopen_agent`, continues a cancelled run. |
| `set_budget(limit: float \| None = None) -> None`, `set_agent_budget(agent: str, limit: float \| None = None) -> None` | As `Workflow::set_budget` and `set_agent_budget`. Invalid values raise `ValueError`. |
| `is_budget_blocked() -> bool` | As `Workflow::is_budget_blocked`. |
| `async def finish() -> list[PyAgentResult]` | As `Workflow::finish`, fires `PostWorkflow`, saves a checkpoint, and returns the recorded agent results. |
| `cancel() -> bool` | Cancel the run, as `Workflow::cancel`. Running agents stop, and pending and later runs raise `WorkflowCancelled`. Returns `False` when the run was already stopped. |
| `is_cancelled() -> bool` | Whether the run is stopped. |
| `add_manual_cost(amount: float, description: str, metadata: str \| None = None) -> float` | Record a dollar cost and return the new workflow total. A budget error still leaves the charge recorded. |
| `cost_ledger_json() -> str` | The current `CostLedger` as JSON. |
| `async def checkpoint() -> None` | Save a checkpoint. Does nothing without persistence. |
| `read_memory(agent_name: str, tier: str, max_chars: int \| None = None) -> str` | Access-checked plain-text read of a tier, or of its trailing `max_chars` characters (`MemoryStore::read_window`). Not the agent tool's versioned JSON. |
| `write_memory(agent_name: str, tier: str, content: str) -> None` | Access-checked write using the tier's mode, through the store directly, without a task ID, freshness notifications or hooks. |

The run methods raise `arko_workflows.WorkflowCancelled` once the run is cancelled, `UserInputRequired` for an agent waiting for input, `AgentPaused` for a paused agent, and `VerifierFailed` for an agent whose completion a verifier couldn't review (paused, and continued with `continue_agent` on `agent_result(agent_run_id)`). Their other errors raise `RuntimeError` with the message `Workflow error: <error>`. Accounting, checkpoint, and memory errors raise `RuntimeError`. Raw handles, cancel handles, event subscription, workflow pause and resume, the aggregate `WorkflowResult`, durable steps, and ballots are not available from Python.

### PyAgentResult

Its read-only properties are `agent_name: str`, `summary: str`, `total_cost: float` (the billed amount, zero for ChatGPT OAuth calls), `status: str`, `agent_run_id: str`, `error: str | None` (the failure or cancellation reason, the questions, or the pause), `failure_cause: str | None` (for a `Failed` run that failed at a rejected error-recovery pause, its cause, such as `"non_retryable"`, see `AgentResult::failure_cause`), and `failure_kind: str | None` (that error's kind, such as `"authentication"` or `"context_overflow"`, see `AgentResult::failure_kind`). `status` is the Rust variant name (`Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`, or `Paused`). `agent_run_id` is a UUID string. The Rust wrapper's eight matching fields are public. It also keeps the full `AgentResult` (private) for `continue_agent` and `resume_agent`. The task and token totals are not exposed. There is no Python constructor.

## Workflow errors

Defined in [error.rs](src/error.rs). `WorkflowError` is re-exported at the root and in the prelude and implements `std::error::Error` and `Display` (through `thiserror`). Match on variants for programmatic handling.

| Variant and payload | Meaning |
| --- | --- |
| `Agent(arko_agents::ArkoAgentsError)` | Agent setup, runtime, or task error, including a managed task that ended failed (`TaskExecution` with the failure reason). Automatic `From`. |
| `Llm(arko_ai::Error)` | Model client error. Automatic `From`. |
| `BuilderMissing { field: String }` | A required builder setting is missing. |
| `BuilderInvalid(String)` | Invalid configuration or checkpoint identity, an invalid durable step spec, a disabled feature (freshness, voting) used through a `Workflow` method, a non-continuable result passed to `continue_agent` or a non-paused result or mismatched decision passed to `resume_agent`, or a disallowed workflow-services caller or target. |
| `AgentNotFound(String)` | No agent is registered under the name. |
| `AgentInterrupted { agent: String, reason: String }` | A managed invocation was interrupted. |
| `AgentCancelled { agent: String, reason: String }` | A managed invocation's task ended `Cancelled` while the workflow run was not stopped. `reason` is the task's reason. |
| `AgentStopped { agent: String, task_id: Uuid, reason: String }` | The host stopped the run alone (`Workflow::stop_run`), and it was cancelled when its wrap-up was over, or at once without one. Its result (`agent_result(task_id)`) is `Cancelled`, with `stopped_by` set, and `reason` is its cancellation reason (`Run stopped (operator): ...`). Not a failure (see [Stopping one agent run](#stopping-one-agent-run)). |
| `VerifierFailed { verifier: String, agent: String, task_id: Uuid, reason: String }` | A completion verifier could not review the agent's completion more often than its `max_review_retries` allow. The completion was neither accepted nor rejected. The task is paused (`Paused`), to be continued with `continue_agent` on `Workflow::agent_result(task_id)`. |
| `UserInputRequired { agent: String, task_id: Uuid, questions: String }` | The run is suspended until an answer is supplied with `continue_agent`. A pause, not a failure. |
| `AgentPaused { agent: String, task_id: Uuid, pause: AgentPause }` | The agent is suspended for a tool approval or error-recovery decision the workflow's policies leave to the caller. Continue it with `resume_agent` (its result is `agent_result(task_id)`). A pause, not a failure. |
| `MemoryTierNotFound(String)` | Unknown memory tier. |
| `MemoryAccessDenied { agent: String, tier: String, action: String }` | The tier's reader or writer policy denies the agent. |
| `MemoryLimitExceeded { tier: String, attempted: usize, limit: usize }` | The write would exceed the tier's byte limit. |
| `BudgetExceeded { spent: f64, limit: f64 }` | Workflow spending strictly exceeds the dollar limit. |
| `AgentBudgetExceeded { agent: String, spent: f64, limit: f64 }` | The agent's cumulative LLM spending strictly exceeds its limit. |
| `PlanBudgetExceeded { plan: Option<String>, window: String, used: f64, limit: f64 }` | Under `BudgetExceededPolicy::Cancel`, a plan budget window (`plan` is `ChatGPT`, `Claude Code`, or `None` for counting without a plan, and `window` is `five-hour` or `seven-day`) is strictly over its limit, in points. |
| `InvalidCost(f64)` | A manual cost is negative or non-finite. |
| `AbortedByHook { hook_point: String, reason: String }` | A hook returned `Abort`, which stopped the run. Returned by the operation that fired a lifecycle hook and by every run call in progress or made later. `hook_point` is the `HookPoint` variant name. A later hook `Abort` on the stopped run returns this first cause's error, not a new `AbortedByHook`. |
| `Persistence(String)` | Run identity, journal, or other persisted-state failure. |
| `DurableUnavailable` | A durable operation was requested without persistence. |
| `DurableDefinitionMismatch { step: String, details: String }` | The recorded step definition or input differs from the request, or an operator resolution found no unresolved attempt. |
| `DurableInDoubt { step: String, operation_id: String }` | The outcome of an earlier attempt is unresolved, or the step is already running on this store. |
| `DurablePermanentlyFailed { step: String, reason: String }` | The step is recorded as permanently failed. |
| `DurableAttemptsExhausted { step: String }` | The retry policy's attempt limit is reached. |
| `Cancelled` | The run was cancelled through `WorkflowCancelHandle::cancel` or `Workflow::cancel`, or stopped through `stop` (`CancelCause::Stopped`). Returned by every run call and durable step in progress or started later, and by `finish`. During a graceful stop's wrap-up, run calls that would start an agent return it too. |
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
| [complete_workflow.rs](examples/complete_workflow.rs) | Explicit agent settings, built-in tools, shared memory, messaging, completion verification, tool-result distillation, persistence, dry-run validation, checkpoint resume, `continue_agent`, and `resume_agent`. |
| [runctl.rs](examples/runctl.rs) | `runctl note RUN_DIR "text" [--file PATH] [--to a,b \| --except c] [--once] [--from-agent NAME \| --from-host]` posts a note to any workflow built with `control_folder()`, live or stopped, through `notes::control::post_note`. `runctl stop RUN_DIR (--agent NAME \| --run ID) [--wrap-up SECS \| --now] [--reason TEXT] [--relaunch]` stops runs of the live workflow alone through `notes::control::request_stop` (a 600-second wrap-up unless `--wrap-up` or `--now`, retired unless `--relaunch`). `runctl status RUN_DIR` prints the latest checkpoint's runs (with the intent of those stopped alone), spend, newest notes, and the control-folder notes not yet posted. Makes no model requests. See [Operator notes](#operator-notes) and [Stopping one agent run](#stopping-one-agent-run). |
