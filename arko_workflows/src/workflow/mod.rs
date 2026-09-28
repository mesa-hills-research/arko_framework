pub mod cancel;
pub mod config;
pub mod engine;
pub mod result;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arko_agents::{AgentConfig, InterceptionFailurePolicy, ToolResultInterceptor};
use arko_ai::LLMClient;

use crate::budget::{AddCostResult, BudgetManager, CostLedger};
use crate::context::SharedContext;
use crate::distillation::DistillationConfig;
use crate::durable::{DurableStepContext, DurableStepSpec, InDoubtContext, ResolveInDoubt};
use crate::error::WorkflowError;
use crate::freshness::{ActorId, ResourceId};
use crate::freshness::{ContextFreshnessConfig, ResourceTracker};
use crate::hooks::{HookCallback, HookPoint, HookRegistry};
use crate::memory::MemoryConfig;
use crate::messaging::registry::{AgentRegistry, AgentStatus};
use crate::messaging::tools::{
    LIST_RECIPIENTS_TOOL_NAME, SEND_MESSAGE_TOOL_NAME, WAIT_FOR_MESSAGE_TOOL_NAME,
};
use crate::messaging::wait::{WaitForMessageTimeout, WaitTimeouts};
use crate::messaging::{MessageBus, MessagingPolicy};
use crate::persistence::{PersistenceConfig, PersistenceManager};
use crate::services::{RUN_WORKFLOW_AGENT_TOOL_NAME, WorkflowServices, WorkflowServicesConfig};
use crate::user_input::UserInputHandler;
use crate::verification::CompletionVerifierConfig;
use crate::voting::{BallotConfig, BallotResult, BallotStore, CAST_VOTE_TOOL_NAME};
use crate::workflow::cancel::{
    Cancellation, DEFAULT_CANCEL_GRACE_PERIOD, RunControl, WorkflowCancelHandle,
};
use crate::workflow::config::{WorkflowConfig, WorkflowMetadata};
use crate::workflow::engine::{StartedRun, WorkflowEngine, fire_hooks_on_context};
use crate::workflow::result::{AgentResult, WorkflowEvent, WorkflowResult};

/// A configured multi-agent workflow.
///
/// Created via [`WorkflowBuilder`]. Provides methods to run agents sequentially,
/// in parallel, or via non-blocking spawn with manual driving.
///
/// A `Workflow` is one run. Once the run is stopped, by
/// [`cancel`](Self::cancel) (or a [`WorkflowCancelHandle`]) or a hook's
/// `Abort`, every run call in progress returns the stop's error after its
/// agents have stopped, and every later one returns it at once.
pub struct Workflow {
    #[allow(dead_code)]
    config: WorkflowConfig,
    engine: WorkflowEngine,
    context: Arc<SharedContext>,
}

impl Workflow {
    /// Creates a new workflow builder.
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Run a named agent with the given prompt. Blocks until the agent completes.
    ///
    /// This is the simplest way to execute an agent. The workflow engine handles
    /// all event processing (hooks, context freshness, budget) internally.
    pub async fn run_agent(
        &self,
        agent_name: &str,
        prompt: &str,
    ) -> Result<AgentResult, WorkflowError> {
        self.ensure_running().await?;
        let result = self
            .engine
            .run_agent(agent_name, prompt, &self.context)
            .await;
        self.finish_call(result, |hook| {
            hook.agent_name = Some(agent_name.to_string());
        })
        .await
    }

    /// Settles a finished run call: a failure fires `OnWorkflowFailed` once
    /// (whose abort replaces the error), and once the run is stopped the call
    /// returns the stop's error and the stop is checkpointed. A pause (an
    /// agent waiting for input, or paused for a decision) and a cancellation
    /// are not failures.
    async fn finish_call<T>(
        &self,
        result: Result<T, WorkflowError>,
        configure: impl FnOnce(&mut crate::hooks::context::HookContext),
    ) -> Result<T, WorkflowError> {
        let mut result = result;
        if let Err(error) = &result
            && !matches!(error, WorkflowError::Cancelled)
            && !engine::is_pause(error)
            && let Err(hook_error) =
                fire_hooks_on_context(&self.context, HookPoint::OnWorkflowFailed, |hook| {
                    configure(hook);
                    hook.error_message = Some(error.to_string());
                })
        {
            result = Err(hook_error);
        }
        self.apply_stop(result).await
    }

    /// `Err` with the stop's error once the run is stopped.
    async fn ensure_running(&self) -> Result<(), WorkflowError> {
        self.apply_stop(Ok(())).await
    }

    /// Once the run is stopped, replaces the call's outcome with the stop's
    /// error and saves the checkpoint recording the stop.
    async fn apply_stop<T>(&self, result: Result<T, WorkflowError>) -> Result<T, WorkflowError> {
        match self.context.run_control.error() {
            Some(stop) => {
                self.engine.save_stop_checkpoint(&self.context).await;
                Err(stop)
            }
            None => result,
        }
    }

    /// Continue a completed agent with its prior conversation and stable run
    /// identity instead of starting again from an empty context, or answer
    /// an agent waiting for input (`AgentRunStatus::WaitingForInput`) with
    /// `prompt`.
    pub async fn continue_agent(
        &self,
        previous: &AgentResult,
        prompt: &str,
    ) -> Result<AgentResult, WorkflowError> {
        self.ensure_running().await?;
        let result = self
            .engine
            .continue_agent(previous, prompt, &self.context)
            .await;
        self.finish_call(result, |hook| {
            hook.agent_name = Some(previous.agent_name.clone());
            hook.agent_run_id = Some(previous.agent_run_id);
        })
        .await
    }

    /// Continue an agent paused for a decision (`AgentRunStatus::Paused`,
    /// returned as [`WorkflowError::AgentPaused`]): decide the tool calls it
    /// awaits approval for, or continue or fail it after an error-recovery
    /// pause. See [`AgentResume`](crate::policy::AgentResume).
    pub async fn resume_agent(
        &self,
        previous: &AgentResult,
        resume: crate::policy::AgentResume,
    ) -> Result<AgentResult, WorkflowError> {
        self.ensure_running().await?;
        let result = self
            .engine
            .resume_agent(previous, resume, &self.context)
            .await;
        self.finish_call(result, |hook| {
            hook.agent_name = Some(previous.agent_name.clone());
            hook.agent_run_id = Some(previous.agent_run_id);
        })
        .await
    }

    /// The recorded result of agent run `agent_run_id` (the task ID that
    /// [`WorkflowError::UserInputRequired`] and [`WorkflowError::AgentPaused`]
    /// carry), if it has one.
    pub fn agent_result(&self, agent_run_id: uuid::Uuid) -> Option<AgentResult> {
        self.context
            .agent_results
            .read()
            .iter()
            .find(|result| result.agent_run_id == agent_run_id)
            .cloned()
    }

    /// Run multiple agents in parallel.
    ///
    /// All agents run concurrently. Results are returned in the same order
    /// as the input once every agent has finished. An agent that pauses
    /// (waiting for input, or paused for a decision) is not a failure: its
    /// result is returned with status `WaitingForInput` or `Paused`, to be
    /// continued with [`continue_agent`](Self::continue_agent) or
    /// [`resume_agent`](Self::resume_agent), while the others run on. If an
    /// agent fails, the first failure is returned (and `OnWorkflowFailed`
    /// fires once); the other results stay available through
    /// [`result`](Self::result).
    pub async fn run_agents_parallel(
        &self,
        agents: Vec<(&str, String)>,
    ) -> Result<Vec<AgentResult>, WorkflowError> {
        self.ensure_running().await?;
        let result = self.engine.run_agents_parallel(agents, &self.context).await;
        self.finish_call(result, |_| {}).await
    }

    /// Spawn an agent without blocking. Returns a handle for driving it.
    ///
    /// Use [`AgentHandle::wait`] to drive the event loop with the workflow's
    /// behavior (its approval and error recovery policies, hooks, context
    /// freshness, budget) and collect the result.
    ///
    /// Use [`AgentHandle::into_parts`] to get the raw channels for custom
    /// event processing.
    pub async fn spawn_agent(
        &self,
        agent_name: &str,
        prompt: &str,
    ) -> Result<AgentHandle, WorkflowError> {
        self.ensure_running().await?;
        self.engine
            .spawn_agent(agent_name, prompt, &self.context)
            .await
    }

    /// A handle that cancels this run from any task or thread.
    pub fn cancel_handle(&self) -> WorkflowCancelHandle {
        WorkflowCancelHandle::new(&self.context)
    }

    /// Cancels the run; see [`WorkflowCancelHandle::cancel`]. Returns
    /// `false` when the run was already stopped.
    pub fn cancel(&self) -> bool {
        self.cancel_handle().cancel()
    }

    /// Whether the run is stopped, by cancellation or a hook's abort.
    pub fn is_cancelled(&self) -> bool {
        self.context.run_control.is_stopped()
    }

    /// Why and when the run stopped, once it has.
    pub fn cancellation(&self) -> Option<Cancellation> {
        self.context.run_control.cancellation()
    }

    /// The status of a registered agent in this run, as `list_recipients`
    /// reports it; `None` for an unknown name.
    pub fn agent_status(&self, agent_name: &str) -> Option<AgentStatus> {
        self.context
            .agent_configs
            .contains_key(agent_name)
            .then(|| self.context.recipient_status(agent_name))
    }

    /// Add a manual cost entry to the workflow budget.
    ///
    /// The cost is recorded in the cost ledger and counted toward the
    /// running total identically to agent LLM costs. Budget warning
    /// and exceeded hooks fire if their thresholds are crossed.
    ///
    /// Returns [`WorkflowError::BudgetExceeded`] if the addition pushes
    /// the total past the budget limit. The cost is still recorded in
    /// the ledger before the error is returned.
    ///
    /// The `metadata` field accepts a free-form string for accounting
    /// context — service name, URL, API call body/response, or any
    /// other information relevant for cost analysis.
    pub fn add_manual_cost(
        &self,
        amount: f64,
        description: &str,
        metadata: Option<String>,
    ) -> Result<AddCostResult, WorkflowError> {
        if !amount.is_finite() || amount < 0.0 {
            return Err(WorkflowError::InvalidCost(amount));
        }
        let cost_result = self
            .context
            .budget
            .add_manual_cost(amount, description, metadata);

        if cost_result.warning_triggered {
            fire_hooks_on_context(&self.context, HookPoint::OnBudgetWarning, |hctx| {
                hctx.budget_spent = cost_result.new_total;
            })?;
        }

        if cost_result.exceeded {
            fire_hooks_on_context(&self.context, HookPoint::OnBudgetExceeded, |hctx| {
                hctx.budget_spent = cost_result.new_total;
            })?;
            return Err(WorkflowError::BudgetExceeded {
                spent: cost_result.new_total,
                limit: self.context.budget.limit().unwrap_or(0.0),
            });
        }

        Ok(cost_result)
    }

    /// Returns a snapshot of the cost ledger for analysis.
    ///
    /// The ledger contains all cost entries — both automatic agent
    /// costs and manual additions — in chronological order.
    pub fn cost_ledger(&self) -> CostLedger {
        self.context.budget.snapshot_ledger()
    }

    /// Access the shared context (memory, messages, budget, etc.)
    pub fn context(&self) -> &Arc<SharedContext> {
        &self.context
    }

    /// Get accumulated workflow result.
    pub fn result(&self) -> WorkflowResult {
        let results = self.context.agent_results.read().clone();
        let total_cost = self.context.budget.total_spent();
        let cost_ledger = self.context.budget.snapshot_ledger();
        WorkflowResult {
            workflow_run_id: self.context.workflow_run_id,
            workflow_name: self.context.workflow_name.clone(),
            metadata: self.context.workflow_metadata.clone(),
            total_cost,
            cost_ledger,
            agent_results: results,
            cancellation: self.context.run_control.cancellation(),
        }
    }

    /// Save a checkpoint of the current workflow state.
    pub async fn checkpoint(&self) -> Result<(), WorkflowError> {
        self.engine.checkpoint(&self.context).await
    }

    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<WorkflowEvent> {
        self.context.event_tx.subscribe()
    }

    /// Record that workflow code or a custom tool observed a resource.
    pub fn register_resource_read(
        &self,
        actor: ActorId,
        resource: ResourceId,
    ) -> Result<(), WorkflowError> {
        let tracker = self.context.resource_tracker.as_ref().ok_or_else(|| {
            WorkflowError::BuilderInvalid("context freshness is not enabled".to_string())
        })?;
        tracker.record_read(&actor, resource);
        Ok(())
    }

    /// Mark an arbitrary resource dirty and proactively notify active readers.
    pub async fn mark_resource_dirty(
        &self,
        actor: ActorId,
        resource: ResourceId,
    ) -> Result<Vec<ActorId>, WorkflowError> {
        let tracker = self.context.resource_tracker.as_ref().ok_or_else(|| {
            WorkflowError::BuilderInvalid("context freshness is not enabled".to_string())
        })?;
        let stale = tracker.record_write(&actor, resource.clone());
        for reader in &stale {
            self.context
                .messages
                .inject_task(
                    reader.task_id,
                    format!(
                        "Context freshness notice: {resource:?} changed after you observed it. Refresh that context before using it."
                    ),
                )
                .await;
            fire_hooks_on_context(&self.context, HookPoint::OnFreshnessConflict, |hook| {
                hook.agent_name = Some(reader.agent_name.clone());
                hook.agent_run_id = Some(reader.task_id);
                hook.metadata.insert(
                    "resource".to_string(),
                    serde_json::json!(format!("{resource:?}")),
                );
                hook.metadata
                    .insert("writer".to_string(), serde_json::json!(actor.agent_name));
            })?;
        }
        Ok(stale)
    }

    /// Mark the workflow run complete, fire its final lifecycle hook, persist
    /// the final snapshot, and return the accumulated result.
    ///
    /// A stopped run is not complete: the snapshot, which records the stop,
    /// is persisted, `PostWorkflow` doesn't fire, and the stop's error is
    /// returned ([`result`](Self::result) still returns the results).
    pub async fn finish(&self) -> Result<WorkflowResult, WorkflowError> {
        if let Some(stop) = self.context.run_control.error() {
            self.checkpoint().await?;
            return Err(stop);
        }
        fire_hooks_on_context(&self.context, HookPoint::PostWorkflow, |_| {})?;
        self.checkpoint().await?;
        Ok(self.result())
    }

    pub fn pause(&self) -> Result<(), WorkflowError> {
        if !self
            .context
            .paused
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            fire_hooks_on_context(&self.context, HookPoint::OnWorkflowPause, |_| {})?;
        }
        Ok(())
    }

    pub fn resume(&self) -> Result<(), WorkflowError> {
        if self
            .context
            .paused
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            fire_hooks_on_context(&self.context, HookPoint::OnWorkflowResume, |_| {})?;
            self.context.resume_notify.notify_waiters();
        }
        Ok(())
    }

    pub fn is_paused(&self) -> bool {
        self.context
            .paused
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Open a ballot. Its eligible voters must be among the workflow's
    /// voters ([`WorkflowBuilder::enable_voting`]), the agents offered
    /// `cast_vote`; with none named, every workflow voter is eligible.
    pub fn begin_ballot(&self, mut config: BallotConfig) -> Result<uuid::Uuid, WorkflowError> {
        let ballots = self.context.ballots.as_ref().ok_or_else(|| {
            WorkflowError::BuilderInvalid("structured voting is not enabled".to_string())
        })?;
        let mut eligible = config.eligible_voters.iter().collect::<Vec<_>>();
        eligible.sort();
        if let Some(voter) = eligible
            .into_iter()
            .find(|voter| !self.context.voters.contains(*voter))
        {
            return Err(WorkflowError::Deliberation(format!(
                "eligible voter '{voter}' is not offered cast_vote; name it in WorkflowBuilder::enable_voting"
            )));
        }
        if config.eligible_voters.is_empty() {
            config.eligible_voters = self.context.voters.clone();
        }
        ballots.begin(config).map_err(WorkflowError::Deliberation)
    }

    pub fn finish_ballot(&self, ballot_id: uuid::Uuid) -> Result<BallotResult, WorkflowError> {
        self.context
            .ballots
            .as_ref()
            .ok_or_else(|| {
                WorkflowError::BuilderInvalid("structured voting is not enabled".to_string())
            })?
            .finish(ballot_id)
            .map_err(WorkflowError::Deliberation)
    }

    /// Execute an explicitly keyed replay-safe workflow step.
    ///
    /// A previously completed step returns its persisted output. An unresolved
    /// prior attempt returns [`WorkflowError::DurableInDoubt`] unless the step
    /// was explicitly declared retry-safe.
    pub async fn durable_step<I, O, F, Fut>(
        &self,
        spec: DurableStepSpec,
        input: &I,
        execute: F,
    ) -> Result<O, WorkflowError>
    where
        I: serde::Serialize,
        O: serde::Serialize + serde::de::DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: std::future::Future<Output = Result<O, WorkflowError>>,
    {
        self.ensure_running().await?;
        let store = self
            .context
            .durable_store
            .as_ref()
            .ok_or(WorkflowError::DurableUnavailable)?;
        let result = store.execute(&spec, input, execute).await;
        self.apply_stop(result).await
    }

    /// Execute a durable step with developer-supplied reconciliation for an
    /// uncertain prior outcome.
    pub async fn durable_step_with_resolver<I, O, F, Fut, R>(
        &self,
        spec: DurableStepSpec,
        input: &I,
        execute: F,
        resolver: &R,
    ) -> Result<O, WorkflowError>
    where
        I: serde::Serialize + Sync,
        O: serde::Serialize + serde::de::DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: std::future::Future<Output = Result<O, WorkflowError>>,
        R: ResolveInDoubt<I, O>,
    {
        self.ensure_running().await?;
        let store = self
            .context
            .durable_store
            .as_ref()
            .ok_or(WorkflowError::DurableUnavailable)?;
        let result = store
            .execute_with_resolver(&spec, input, execute, resolver)
            .await;
        self.apply_stop(result).await
    }

    /// List steps whose side-effect outcome is not known.
    pub async fn in_doubt_steps(&self) -> Result<Vec<InDoubtContext>, WorkflowError> {
        let store = self
            .context
            .durable_store
            .as_ref()
            .ok_or(WorkflowError::DurableUnavailable)?;
        store.list_in_doubt().await
    }

    pub async fn resolve_in_doubt_completed<I, O>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        output: &O,
        actor: &str,
        reason: &str,
    ) -> Result<(), WorkflowError>
    where
        I: serde::Serialize,
        O: serde::Serialize,
    {
        let store = self
            .context
            .durable_store
            .as_ref()
            .ok_or(WorkflowError::DurableUnavailable)?;
        store
            .operator_complete(spec, input, output, actor, reason)
            .await
    }

    pub async fn resolve_in_doubt_failed<I: serde::Serialize>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        actor: &str,
        reason: &str,
    ) -> Result<(), WorkflowError> {
        let store = self
            .context
            .durable_store
            .as_ref()
            .ok_or(WorkflowError::DurableUnavailable)?;
        store
            .operator_permanently_fail(spec, input, actor, reason)
            .await
    }

    pub async fn authorize_in_doubt_retry<I: serde::Serialize>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        actor: &str,
        reason: &str,
    ) -> Result<(), WorkflowError> {
        let store = self
            .context
            .durable_store
            .as_ref()
            .ok_or(WorkflowError::DurableUnavailable)?;
        store
            .operator_authorize_retry(spec, input, actor, reason)
            .await
    }
}

/// A handle to a running agent, enabling non-blocking spawn-then-wait patterns.
///
/// Obtained via [`Workflow::spawn_agent`]. Two usage modes:
///
/// 1. **Default driving**: Call [`wait`](AgentHandle::wait) to let the workflow
///    engine handle all events (hooks, context freshness, budget) and return the result.
///
/// 2. **Custom driving**: Call [`into_parts`](AgentHandle::into_parts) to get
///    the raw event receiver, command sender, and join handle for manual event
///    processing.
///
/// Dropping the handle without calling either stops the agent: its runtime
/// task is aborted and no result is recorded.
pub struct AgentHandle {
    /// Taken by `wait` or `into_parts`.
    run: Option<StartedRun>,
    context: Arc<SharedContext>,
    persistence: Option<Arc<PersistenceManager>>,
}

impl AgentHandle {
    pub(crate) fn new(
        run: StartedRun,
        context: Arc<SharedContext>,
        persistence: Option<Arc<PersistenceManager>>,
    ) -> Self {
        Self {
            run: Some(run),
            context,
            persistence,
        }
    }

    fn run(&self) -> &StartedRun {
        self.run
            .as_ref()
            .expect("the run is taken only by consuming the handle")
    }

    fn take_run(&mut self) -> StartedRun {
        self.run
            .take()
            .expect("the run is taken only by consuming the handle")
    }

    /// Returns the name of the agent this handle controls.
    pub fn agent_name(&self) -> &str {
        &self.run().agent_name
    }

    /// Drive the event loop with default workflow behavior and wait for completion.
    ///
    /// Applies the workflow's approval and error recovery policies, hooks,
    /// context freshness, and budget tracking. Returns the agent result on
    /// completion, and otherwise the same errors as [`Workflow::run_agent`].
    pub async fn wait(mut self) -> Result<AgentResult, WorkflowError> {
        let run = self.take_run();
        let result = engine::drive_to_result(run, &self.context, self.persistence.as_ref()).await;
        if self.context.run_control.is_stopped() {
            engine::save_stop_checkpoint(&self.context, self.persistence.as_ref()).await;
        }
        result
    }

    /// Decompose the handle into raw parts for custom event processing.
    ///
    /// Returns `(event_receiver, command_sender, join_handle, agent_name)`.
    ///
    /// When using this, the caller is responsible for processing
    /// [`AgentEvent`](arko_agents::runtime::event::AgentEvent)s and sending
    /// appropriate [`UserCommand`](arko_agents::runtime::command::UserCommand)s.
    /// Hooks and budget tracking are **not** applied automatically. Cancelling
    /// the run still sends the agent `CancelTask`, and the tool middleware
    /// still cancels it at its next tool call. The workflow can't observe
    /// the agent's end: other agents see it as running for the rest of the
    /// run.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        mut self,
    ) -> (
        tokio::sync::mpsc::Receiver<arko_agents::runtime::event::AgentEvent>,
        tokio::sync::mpsc::Sender<arko_agents::runtime::command::UserCommand>,
        tokio::task::JoinHandle<Result<arko_agents::Task, arko_agents::ArkoAgentsError>>,
        String,
    ) {
        let run = self.take_run();
        (
            run.event_rx,
            run.cmd_tx,
            run.join_handle.into_inner(),
            run.agent_name,
        )
    }
}

impl Drop for AgentHandle {
    fn drop(&mut self) {
        // Dropped unused: the runtime task is aborted with the run.
        if let Some(run) = self.run.take() {
            engine::abandon_run(&run, &self.context);
        }
    }
}

/// Builder for constructing [`Workflow`] instances.
pub struct WorkflowBuilder {
    name: Option<String>,
    metadata: WorkflowMetadata,
    workspace: Option<PathBuf>,
    llm_client: Option<Arc<LLMClient>>,
    agents: Vec<(String, AgentConfig)>,
    memory_config: Option<MemoryConfig>,
    /// On resume: checkpoint memory tier name -> configured tier name.
    memory_renames: HashMap<String, String>,
    /// On resume: checkpoint memory tiers to discard.
    dropped_memory_tiers: HashSet<String>,
    budget_limit: Option<f64>,
    budget_warning: Option<f64>,
    agent_budgets: HashMap<String, f64>,
    hooks: Vec<(HookPoint, HookCallback)>,
    persistence_config: Option<PersistenceConfig>,
    resume_from: Option<PathBuf>,
    freshness_config: Option<ContextFreshnessConfig>,
    global_tools: Vec<Arc<dyn arko_agents::tool::Tool>>,
    agent_tools: HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>,
    global_interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
    agent_interceptors: HashMap<String, Vec<Arc<dyn ToolResultInterceptor>>>,
    completion_verifiers: Vec<CompletionVerifierConfig>,
    distillation_configs: Vec<DistillationConfig>,
    services_config: Option<WorkflowServicesConfig>,
    messaging: Option<MessagingPolicy>,
    wait_timeouts: WaitTimeouts,
    user_input_handler: Option<Arc<dyn UserInputHandler>>,
    approval_policy: crate::policy::ApprovalPolicy,
    error_recovery_policy: crate::policy::ErrorRecoveryPolicy,
    recovery_waits: crate::policy::RecoveryWaits,
    parallel: crate::context::ParallelSettings,
    /// `Some(voters)` once voting is enabled.
    voters: Option<HashSet<String>>,
    interception_failure_policy: Option<InterceptionFailurePolicy>,
    cancel_grace_period: std::time::Duration,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            metadata: WorkflowMetadata::default(),
            workspace: None,
            llm_client: None,
            agents: Vec::new(),
            memory_config: None,
            memory_renames: HashMap::new(),
            dropped_memory_tiers: HashSet::new(),
            budget_limit: None,
            budget_warning: None,
            agent_budgets: HashMap::new(),
            hooks: Vec::new(),
            persistence_config: None,
            resume_from: None,
            freshness_config: None,
            global_tools: Vec::new(),
            agent_tools: HashMap::new(),
            global_interceptors: Vec::new(),
            agent_interceptors: HashMap::new(),
            completion_verifiers: Vec::new(),
            distillation_configs: Vec::new(),
            services_config: None,
            messaging: Some(MessagingPolicy::default()),
            wait_timeouts: WaitTimeouts::default(),
            user_input_handler: None,
            approval_policy: crate::policy::ApprovalPolicy::default(),
            error_recovery_policy: crate::policy::ErrorRecoveryPolicy::default(),
            recovery_waits: crate::policy::RecoveryWaits::default(),
            parallel: crate::context::ParallelSettings::default(),
            voters: None,
            interception_failure_policy: None,
            cancel_grace_period: DEFAULT_CANCEL_GRACE_PERIOD,
        }
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the workspace directory.
    pub fn workspace(mut self, path: impl Into<PathBuf>) -> Self {
        self.workspace = Some(path.into());
        self
    }

    /// Set the LLM client.
    pub fn llm_client(mut self, client: Arc<LLMClient>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Register a named agent configuration.
    pub fn agent(mut self, name: impl Into<String>, config: AgentConfig) -> Self {
        self.agents.push((name.into(), config));
        self
    }

    /// Configure the memory system.
    pub fn memory(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    /// Set the budget limit in dollars.
    pub fn budget(mut self, limit: f64) -> Self {
        self.budget_limit = Some(limit);
        self
    }

    /// Set the budget warning threshold in dollars.
    pub fn budget_warning(mut self, threshold: f64) -> Self {
        self.budget_warning = Some(threshold);
        self
    }

    /// Register a lifecycle hook.
    pub fn hook(mut self, point: HookPoint, callback: HookCallback) -> Self {
        self.hooks.push((point, callback));
        self
    }

    /// Configure persistence.
    pub fn persistence(mut self, config: PersistenceConfig) -> Self {
        self.persistence_config = Some(config);
        self
    }

    pub fn agent_budget(mut self, agent: impl Into<String>, limit: f64) -> Self {
        self.agent_budgets.insert(agent.into(), limit);
        self
    }

    /// Explicitly disable run persistence and durable steps.
    pub fn ephemeral(mut self) -> Self {
        self.persistence_config = None;
        self.resume_from = None;
        self
    }

    pub fn metadata(mut self, metadata: WorkflowMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// On resume, restore the checkpoint's memory tier `from` into the
    /// configured tier `to` (a tier renamed in code).
    pub fn rename_memory_tier_on_resume(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        self.memory_renames.insert(from.into(), to.into());
        self
    }

    /// On resume, discard the checkpoint's memory tier `name` (one no longer
    /// configured), with its content and history. Without this, a
    /// checkpoint tier that matches no configured tier makes `build` fail.
    pub fn drop_memory_tier_on_resume(mut self, name: impl Into<String>) -> Self {
        self.dropped_memory_tiers.insert(name.into());
        self
    }

    /// Restore workflow state from a checkpoint when building. Persistence
    /// must also be configured, and its run directory must match the
    /// checkpoint's workflow-run identity.
    ///
    /// Memory: the current configuration decides each tier's readers,
    /// writers, limit and write mode, and the checkpoint supplies content
    /// and history by tier name. A configured tier the checkpoint lacks
    /// starts as configured. A checkpoint tier matching no configured tier
    /// makes `build` fail, unless it is mapped to one with
    /// [`rename_memory_tier_on_resume`](Self::rename_memory_tier_on_resume)
    /// or discarded with
    /// [`drop_memory_tier_on_resume`](Self::drop_memory_tier_on_resume).
    pub fn resume_from(mut self, checkpoint: impl Into<PathBuf>) -> Self {
        self.resume_from = Some(checkpoint.into());
        self
    }

    /// Enable generalized context freshness for files, directory listings,
    /// memory tiers, and custom resources.
    pub fn context_freshness(mut self, config: ContextFreshnessConfig) -> Self {
        self.freshness_config = Some(config);
        self
    }

    /// Register a tool available to all agents.
    pub fn global_tool(mut self, tool: Arc<dyn arko_agents::tool::Tool>) -> Self {
        self.global_tools.push(tool);
        self
    }

    /// Register a tool available only to a specific agent.
    pub fn agent_tool(
        mut self,
        agent_name: impl Into<String>,
        tool: Arc<dyn arko_agents::tool::Tool>,
    ) -> Self {
        self.agent_tools
            .entry(agent_name.into())
            .or_default()
            .push(tool);
        self
    }

    /// Register a tool result interceptor applied to all agents.
    ///
    /// Multiple global interceptors are chained in registration order.
    /// Global interceptors run before any agent-specific interceptors.
    pub fn tool_result_interceptor(mut self, interceptor: Arc<dyn ToolResultInterceptor>) -> Self {
        self.global_interceptors.push(interceptor);
        self
    }

    /// Register a tool result interceptor applied only to a specific agent.
    ///
    /// Multiple interceptors for the same agent are chained in registration order,
    /// and run after any global interceptors.
    pub fn agent_tool_result_interceptor(
        mut self,
        agent_name: impl Into<String>,
        interceptor: Arc<dyn ToolResultInterceptor>,
    ) -> Self {
        self.agent_interceptors
            .entry(agent_name.into())
            .or_default()
            .push(interceptor);
        self
    }

    /// Set the failure policy for tool result interception.
    ///
    /// Defaults to `UseRawResult` (log a warning and continue without the
    /// interceptor's change if interception fails). An agent's interceptors
    /// run chained: global interceptors, then the agent's, then its
    /// distillers. The policy applies to each: an interceptor that fails is
    /// retried up to the policy's attempts, without running those before it
    /// again, and then skipped, so the next one (and the agent) gets the
    /// output of the last one that succeeded; a failure never undoes an
    /// earlier interceptor's work, such as a redaction. Under `InformAgent`
    /// and `RetryThenInformAgent` the result ends with a note that an
    /// interceptor failed. `load_skill` results are intercepted only when
    /// the agent's `SkillsConfig::load_skill_interception` opts in, under
    /// the policy it sets.
    pub fn interception_failure_policy(mut self, policy: InterceptionFailurePolicy) -> Self {
        self.interception_failure_policy = Some(policy);
        self
    }

    pub fn completion_verifier(mut self, config: CompletionVerifierConfig) -> Self {
        self.completion_verifiers.push(config);
        self
    }

    pub fn distiller(mut self, config: DistillationConfig) -> Self {
        self.distillation_configs.push(config);
        self
    }

    pub fn workflow_services(mut self, config: WorkflowServicesConfig) -> Self {
        self.services_config = Some(config);
        self
    }

    /// Set whom each agent may message with the `send_message` tool. By
    /// default every agent may message every other agent.
    pub fn messaging(mut self, policy: MessagingPolicy) -> Self {
        self.messaging = Some(policy);
        self
    }

    /// Offer no agent the `send_message` tool. The tool name is then free
    /// for a host tool.
    pub fn disable_messaging(mut self) -> Self {
        self.messaging = None;
        self
    }

    /// How long `wait_for_message` waits at most, for every agent without its
    /// own setting ([`agent_wait_for_message_timeout`](Self::agent_wait_for_message_timeout)).
    /// Default [`WaitForMessageTimeout::Auto`]: just inside the agent's
    /// prompt-cache lifetime, so the request after a wait still finds the
    /// prompt cached: 4 minutes 30 seconds, or 54 minutes with Anthropic's
    /// 1-hour TTL. An agent waits longer by calling the tool again. A
    /// `Duration` sets a fixed limit, more than zero and at most 24 hours; one
    /// past the cache lifetime can cost a cache miss on the next turn.
    pub fn wait_for_message_timeout(mut self, timeout: impl Into<WaitForMessageTimeout>) -> Self {
        self.wait_timeouts.workflow = timeout.into();
        self
    }

    /// How long `wait_for_message` waits at most for `agent`, in place of the
    /// workflow's setting ([`wait_for_message_timeout`](Self::wait_for_message_timeout)).
    pub fn agent_wait_for_message_timeout(
        mut self,
        agent: impl Into<String>,
        timeout: impl Into<WaitForMessageTimeout>,
    ) -> Self {
        self.wait_timeouts
            .agents
            .insert(agent.into(), timeout.into());
        self
    }

    pub fn user_input_handler(mut self, handler: Arc<dyn UserInputHandler>) -> Self {
        self.user_input_handler = Some(handler);
        self
    }

    /// How agents' tool-approval requests are answered. **The default,
    /// [`ApprovalPolicy::ApproveAll`](crate::policy::ApprovalPolicy::ApproveAll),
    /// approves every call**, including calls an agent's configuration marks
    /// for approval and paths outside the workspace under
    /// `PathSafetyPolicy::RequireApproval`, so the workflow runs without
    /// human input (the first such call per agent, tool and reason is logged
    /// as a warning). `RespectAgentConfig` pauses the agent for the caller;
    /// `Handler` decides each call in-process.
    pub fn approval_policy(mut self, policy: crate::policy::ApprovalPolicy) -> Self {
        self.approval_policy = policy;
        self
    }

    /// How agents' error-recovery pauses are answered. **The default,
    /// [`ErrorRecoveryPolicy::Automatic`](crate::policy::ErrorRecoveryPolicy::Automatic),
    /// recovers without an attempt limit** from errors that may pass
    /// (outages, rate limits, plan usage limits, which it waits out) and
    /// fails the run clearly on errors that won't; `Limited { max }` caps
    /// the automatic recoveries, and `ReturnToCaller` pauses the agent for
    /// the caller at every error-recovery pause.
    pub fn error_recovery_policy(mut self, policy: crate::policy::ErrorRecoveryPolicy) -> Self {
        self.error_recovery_policy = policy;
        self
    }

    /// Runs at most `max` agents of a `run_agents_parallel` call at once;
    /// the others start as places free up, in input order. Default: no cap,
    /// every agent starts at once. `build` rejects zero. Workflow services
    /// have their own limit (`WorkflowServicesConfig::max_active_calls`).
    pub fn max_parallel_agents(mut self, max: usize) -> Self {
        self.parallel.max_agents = Some(max);
        self
    }

    /// Opt-in (default off): in `run_agents_parallel`, the first agent that
    /// fails stops the others (each is cancelled, winding down within the
    /// cancel grace period), agents not started yet don't start, and the
    /// call returns that failure. A pause is not a failure. Off, every agent
    /// runs to its end and the first failure in input order is returned.
    pub fn fail_fast(mut self, enabled: bool) -> Self {
        self.parallel.fail_fast = enabled;
        self
    }

    /// Shortens automatic recovery's waits (tests only).
    #[cfg(test)]
    pub(crate) fn recovery_waits(mut self, waits: crate::policy::RecoveryWaits) -> Self {
        self.recovery_waits = waits;
        self
    }

    /// Enable structured ballots, with `voters` (registration names) offered
    /// the `cast_vote` tool; other agents are not. A ballot's eligible
    /// voters must be among them.
    pub fn enable_voting<I, S>(mut self, voters: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.voters = Some(voters.into_iter().map(Into::into).collect());
        self
    }

    /// How long a stopped run waits for each running agent to stop before
    /// aborting it (default [`DEFAULT_CANCEL_GRACE_PERIOD`], 30 seconds). An
    /// agent stops at its next step; one running a long tool call stops once
    /// the call returns.
    pub fn cancel_grace_period(mut self, grace_period: std::time::Duration) -> Self {
        self.cancel_grace_period = grace_period;
        self
    }

    /// Build the workflow.
    pub fn build(self) -> Result<Workflow, WorkflowError> {
        for config in &self.distillation_configs {
            config.validate()?;
        }
        let name = self.name.ok_or(WorkflowError::BuilderMissing {
            field: "name".into(),
        })?;
        let workspace = self.workspace.ok_or(WorkflowError::BuilderMissing {
            field: "workspace".into(),
        })?;
        let llm_client = self.llm_client.ok_or(WorkflowError::BuilderMissing {
            field: "llm_client".into(),
        })?;

        if self.agents.is_empty() {
            return Err(WorkflowError::BuilderMissing {
                field: "agents (at least one required)".into(),
            });
        }

        if !self.metadata.supported_platforms.is_empty()
            && !self
                .metadata
                .supported_platforms
                .iter()
                .any(|platform| platform.matches_host())
        {
            return Err(WorkflowError::BuilderInvalid(format!(
                "workflow does not support host platform {}/{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            )));
        }

        let workspace = normalize_workspace(&workspace)?;
        if self.parallel.max_agents == Some(0) {
            return Err(WorkflowError::BuilderInvalid(
                "max_parallel_agents must be at least 1".to_string(),
            ));
        }

        for (label, value) in [
            ("budget limit", self.budget_limit),
            ("budget warning", self.budget_warning),
        ] {
            if let Some(value) = value
                && (!value.is_finite() || value < 0.0)
            {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "{label} must be finite and non-negative"
                )));
            }
        }

        let mut agent_names = HashSet::new();
        for (agent_name, _) in &self.agents {
            if !agent_names.insert(agent_name.clone()) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "duplicate agent name '{agent_name}'"
                )));
            }
        }

        for referenced_agent in self
            .agent_tools
            .keys()
            .chain(self.agent_interceptors.keys())
        {
            if !agent_names.contains(referenced_agent) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "configuration references unknown agent '{referenced_agent}'"
                )));
            }
        }
        for (agent, limit) in &self.agent_budgets {
            if !agent_names.contains(agent) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "agent budget references unknown agent '{agent}'"
                )));
            }
            if !limit.is_finite() || *limit < 0.0 {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "budget for agent '{agent}' must be finite and non-negative"
                )));
            }
        }

        for verifier in &self.completion_verifiers {
            if !agent_names.contains(&verifier.verifier_agent) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "completion verifier '{}' references unknown agent '{}'",
                    verifier.name, verifier.verifier_agent
                )));
            }
            if verifier.target_agents.is_empty() {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "completion verifier '{}' has no target: name the agents it verifies with target(..)",
                    verifier.name
                )));
            }
            for target in &verifier.target_agents {
                if !agent_names.contains(target) {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "completion verifier '{}' targets unknown agent '{}'",
                        verifier.name, target
                    )));
                }
            }
        }

        for distiller in &self.distillation_configs {
            if let Some(target) = distiller
                .target_agents
                .iter()
                .find(|target| !agent_names.contains(*target))
            {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "distiller '{}' targets unknown agent '{target}'",
                    distiller.name
                )));
            }
        }

        if let Some(config) = &self.memory_config {
            config.validate(&agent_names)?;
        }
        if let Some(services) = &self.services_config {
            services.validate(&agent_names)?;
            if services.allowed_callers.is_empty() && !services.all_callers {
                log::warn!(
                    "Workflow '{name}': workflow services name no caller, so no agent is offered run_workflow_agent (WorkflowServicesConfig::allow_caller)"
                );
            }
        }
        if let Some(voters) = &self.voters {
            let mut voters = voters.iter().collect::<Vec<_>>();
            voters.sort();
            if let Some(unknown) = voters
                .into_iter()
                .find(|voter| !agent_names.contains(*voter))
            {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "voting names unknown agent '{unknown}'"
                )));
            }
        }
        warn_unknown_tool_names(
            &name,
            &self.agents,
            &self.global_tools,
            &self.agent_tools,
            self.freshness_config.as_ref(),
            &self.distillation_configs,
        );

        // Agents offered the workflow's send_message tool: those with a
        // permitted recipient.
        let mut roster = agent_names.iter().cloned().collect::<Vec<_>>();
        roster.sort();
        let (messaging_agents, receiving_agents) = match &self.messaging {
            Some(policy) => {
                policy.validate(&agent_names)?;
                let sending = roster
                    .iter()
                    .filter(|agent| !policy.recipients_of(agent, &roster).is_empty())
                    .cloned()
                    .collect::<HashSet<_>>();
                // Offered `wait_for_message`: another agent may message it.
                let receiving = roster
                    .iter()
                    .filter(|agent| roster.iter().any(|sender| policy.permits(sender, agent)))
                    .cloned()
                    .collect::<HashSet<_>>();
                (sending, receiving)
            }
            None => (HashSet::new(), HashSet::new()),
        };
        for (agent, timeout) in std::iter::once((None, &self.wait_timeouts.workflow)).chain(
            self.wait_timeouts
                .agents
                .iter()
                .map(|(agent, timeout)| (Some(agent), timeout)),
        ) {
            if let Some(agent) = agent
                && !agent_names.contains(agent)
            {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "wait_for_message timeout references unknown agent '{agent}'"
                )));
            }
            timeout.validate().map_err(WorkflowError::BuilderInvalid)?;
        }

        let mut global_tool_names = HashSet::new();
        for tool in &self.global_tools {
            let tool_name = tool.name();
            if matches!(
                tool_name.as_str(),
                "complete_task"
                    | "ask_user"
                    | "read_memory"
                    | "write_memory"
                    | "list_memories"
                    | RUN_WORKFLOW_AGENT_TOOL_NAME
                    | CAST_VOTE_TOOL_NAME
            ) || (matches!(
                tool_name.as_str(),
                SEND_MESSAGE_TOOL_NAME | LIST_RECIPIENTS_TOOL_NAME
            ) && !messaging_agents.is_empty())
                || (tool_name == WAIT_FOR_MESSAGE_TOOL_NAME && !receiving_agents.is_empty())
            {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "tool name '{tool_name}' is reserved"
                )));
            }
            if !global_tool_names.insert(tool_name.clone()) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "duplicate global tool name '{tool_name}'"
                )));
            }
        }

        for (agent_name, tools) in &self.agent_tools {
            let mut names = global_tool_names.clone();
            for tool in tools {
                let tool_name = tool.name();
                if matches!(
                    tool_name.as_str(),
                    SEND_MESSAGE_TOOL_NAME | LIST_RECIPIENTS_TOOL_NAME
                ) && messaging_agents.contains(agent_name)
                {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "tool name '{tool_name}' is reserved for agent '{agent_name}', which messaging permits to send"
                    )));
                }
                if tool_name == WAIT_FOR_MESSAGE_TOOL_NAME && receiving_agents.contains(agent_name)
                {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "tool name '{tool_name}' is reserved for agent '{agent_name}', which messaging permits to receive"
                    )));
                }
                if !names.insert(tool_name.clone()) {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "duplicate tool name '{tool_name}' for agent '{agent_name}'"
                    )));
                }
            }
        }

        let resume_state = match &self.resume_from {
            Some(path) => {
                let json = std::fs::read_to_string(path)?;
                let state: crate::persistence::state::WorkflowState = serde_json::from_str(&json)?;
                if state.workflow_name != name {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "checkpoint is for workflow '{}', not '{}'",
                        state.workflow_name, name
                    )));
                }
                Some(state)
            }
            None => None,
        };
        // The checkpoint's results, their tasks read from the task files it
        // refers to (in the checkpoint's directory).
        let resumed_results = match (&resume_state, &self.resume_from) {
            (Some(state), Some(path)) => {
                let run_dir = path.parent().unwrap_or(std::path::Path::new("."));
                state.agent_results_in(run_dir)?
            }
            _ => Vec::new(),
        };
        if resume_state.is_some() && self.persistence_config.is_none() {
            return Err(WorkflowError::BuilderInvalid(
                "resume_from requires persistence configuration".to_string(),
            ));
        }

        let workflow_run_id = match &self.persistence_config {
            Some(config) => {
                let persisted = PersistenceManager::resolve_run_id(config)?;
                if let Some(state) = &resume_state
                    && state.workflow_run_id != uuid::Uuid::nil()
                    && state.workflow_run_id != persisted
                {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "checkpoint run {} does not match persistence run {}",
                        state.workflow_run_id, persisted
                    )));
                }
                persisted
            }
            None => uuid::Uuid::new_v4(),
        };
        let persistence = self
            .persistence_config
            .map(|config| PersistenceManager::new(config, workflow_run_id));
        if let (Some(persistence), Some(state)) = (&persistence, &resume_state) {
            // Their task files hold these tasks already.
            persistence.mark_tasks_saved(state.agent_runs.iter().map(|run| run.agent_run_id));
        }
        let durable_store = persistence.as_ref().map(PersistenceManager::durable_store);

        let budget = if let Some(state) = &resume_state {
            BudgetManager::from_ledger(
                self.budget_limit,
                self.budget_warning,
                state.cost_ledger.clone(),
            )
        } else {
            BudgetManager::new(self.budget_limit, self.budget_warning)
        };
        for (agent, limit) in self.agent_budgets {
            budget.set_agent_limit(agent, limit);
        }
        // On resume the configuration decides access, limits and write
        // modes; the checkpoint supplies content and history by tier name.
        let memory = match &resume_state {
            Some(state) => self.memory_config.unwrap_or_default().restore(
                state.memory_snapshot.clone(),
                &self.memory_renames,
                &self.dropped_memory_tiers,
            )?,
            None => self.memory_config.unwrap_or_default().build(),
        };
        let messages = if let Some(state) = &resume_state {
            MessageBus::from_snapshot(agent_names.iter().cloned(), state.message_queues.clone())
        } else {
            MessageBus::new(agent_names.iter().cloned())
        };
        let resource_tracker = self.freshness_config.map(|config| {
            if let Some(snapshot) = resume_state
                .as_ref()
                .and_then(|state| state.freshness_snapshot.clone())
            {
                ResourceTracker::from_snapshot(workspace.clone(), config, snapshot)
            } else {
                ResourceTracker::new(workspace.clone(), config)
            }
        });

        let mut hook_registry = HookRegistry::new();
        for (point, callback) in self.hooks {
            hook_registry.register(point, callback);
        }

        let (event_tx, _) = tokio::sync::broadcast::channel(1_024);
        let context = Arc::new(SharedContext {
            workflow_run_id,
            workflow_name: name.clone(),
            workflow_metadata: self.metadata.clone(),
            workspace: workspace.clone(),
            budget,
            durable_store,
            memory,
            messages,
            messaging: self.messaging,
            wait_timeouts: self.wait_timeouts,
            resource_tracker,
            hooks: Arc::new(hook_registry),
            agent_configs: self.agents.iter().cloned().collect(),
            global_tools: self.global_tools,
            agent_tools: self.agent_tools,
            agent_results: parking_lot::RwLock::new(resumed_results.clone()),
            global_interceptors: self.global_interceptors,
            agent_interceptors: self.agent_interceptors,
            interception_failure_policy: self.interception_failure_policy.unwrap_or_default(),
            completion_verifiers: self.completion_verifiers,
            verification_attempts: parking_lot::Mutex::new(HashMap::new()),
            verification_baselines: parking_lot::Mutex::new(HashMap::new()),
            verification_failures: parking_lot::Mutex::new(HashMap::new()),
            verifier_failures: parking_lot::Mutex::new(HashMap::new()),
            verification_records: parking_lot::RwLock::new(
                resume_state
                    .as_ref()
                    .map(|state| state.verification_records.clone())
                    .unwrap_or_default(),
            ),
            distillation_configs: self.distillation_configs,
            distillation_records: parking_lot::RwLock::new(
                resume_state
                    .as_ref()
                    .map(|state| state.distillation_records.clone())
                    .unwrap_or_default(),
            ),
            workflow_services: std::sync::OnceLock::new(),
            paused: std::sync::atomic::AtomicBool::new(false),
            resume_notify: tokio::sync::Notify::new(),
            run_control: Arc::new(RunControl::new(self.cancel_grace_period)),
            registry: match &resume_state {
                Some(state) => {
                    AgentRegistry::restore(state.agent_registry.clone(), &resumed_results)
                }
                None => AgentRegistry::default(),
            },
            user_input_handler: self.user_input_handler,
            approval_policy: self.approval_policy,
            error_recovery_policy: self.error_recovery_policy,
            recovery_waits: self.recovery_waits,
            parallel: self.parallel,
            approval_reasons: parking_lot::Mutex::new(HashMap::new()),
            approval_warnings: parking_lot::Mutex::new(HashSet::new()),
            ballots: self
                .voters
                .is_some()
                .then(|| Arc::new(BallotStore::default())),
            voters: self.voters.clone().unwrap_or_default(),
            event_tx,
        });

        if let Some(services_config) = self.services_config {
            let services = WorkflowServices::new(
                Arc::downgrade(&context),
                llm_client.clone(),
                persistence.clone().map(Arc::new),
                services_config,
            );
            context.workflow_services.set(services).map_err(|_| {
                WorkflowError::BuilderInvalid("workflow services initialized twice".to_string())
            })?;
        }

        let engine = WorkflowEngine::new(llm_client, persistence);
        fire_hooks_on_context(&context, HookPoint::PreWorkflow, |_| {})?;

        Ok(Workflow {
            config: WorkflowConfig {
                name,
                workspace,
                metadata: self.metadata,
            },
            engine,
            context,
        })
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The workspace as an absolute path with `.` and `..` resolved lexically
/// (symbolic links are kept), so paths the agents and freshness tracking
/// derive from it compare equal. `std::fs::canonicalize` is not used: on
/// Windows it returns a verbatim `\\?\` path, which other tools and
/// comparisons treat as a different path. The directory must exist.
fn normalize_workspace(workspace: &std::path::Path) -> Result<PathBuf, WorkflowError> {
    let absolute = std::path::absolute(workspace).map_err(|error| {
        WorkflowError::BuilderInvalid(format!(
            "workspace '{}' can't be made absolute: {error}",
            workspace.display()
        ))
    })?;
    let normalized = arko_agents::utils::normalize_path(&absolute);
    if !normalized.is_dir() {
        return Err(WorkflowError::BuilderInvalid(format!(
            "workspace '{}' does not exist or is not a directory",
            workspace.display()
        )));
    }
    Ok(normalized)
}

/// Logs a warning for each tool name the freshness mapping or a distiller
/// names that no agent is configured with, no workflow tool has, and no
/// built-in tool has. Only a warning: MCP servers and skills add tools
/// whose names aren't known at build.
fn warn_unknown_tool_names(
    workflow: &str,
    agents: &[(String, AgentConfig)],
    global_tools: &[Arc<dyn arko_agents::tool::Tool>],
    agent_tools: &HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>,
    freshness: Option<&ContextFreshnessConfig>,
    distillers: &[DistillationConfig],
) {
    let mut known = arko_agents::ToolRegistry::new()
        .list_tool_names()
        .into_iter()
        .collect::<HashSet<_>>();
    known.extend(
        [
            crate::memory::tools::READ_MEMORY_TOOL_NAME,
            crate::memory::tools::WRITE_MEMORY_TOOL_NAME,
            crate::memory::tools::LIST_MEMORIES_TOOL_NAME,
            SEND_MESSAGE_TOOL_NAME,
            LIST_RECIPIENTS_TOOL_NAME,
            WAIT_FOR_MESSAGE_TOOL_NAME,
            RUN_WORKFLOW_AGENT_TOOL_NAME,
            CAST_VOTE_TOOL_NAME,
            arko_agents::LOAD_SKILL_TOOL_NAME,
        ]
        .map(str::to_string),
    );
    known.extend(
        agents
            .iter()
            .flat_map(|(_, config)| config.tool_config.available_tool_names.iter().cloned()),
    );
    known.extend(global_tools.iter().map(|tool| tool.name()));
    known.extend(agent_tools.values().flatten().map(|tool| tool.name()));
    let mut named = Vec::new();
    if let Some(freshness) = freshness {
        named.extend(
            freshness
                .tool_effects
                .keys()
                .map(|tool| ("the context freshness configuration".to_string(), tool)),
        );
    }
    for distiller in distillers {
        named.extend(
            distiller
                .target_tools
                .iter()
                .map(|tool| (format!("distiller '{}'", distiller.name), tool)),
        );
    }
    named.sort();
    for (source, tool) in named {
        if !known.contains(tool) {
            log::warn!(
                "Workflow '{workflow}': {source} names tool '{tool}', which no agent is configured with and no workflow or built-in tool has; check the name (an MCP or skill tool can't be checked before it runs)."
            );
        }
    }
}

#[cfg(test)]
#[path = "messaging_tests.rs"]
mod messaging_tests;

#[cfg(test)]
#[path = "wait_tests.rs"]
mod wait_tests;

#[cfg(test)]
#[path = "policy_tests.rs"]
mod policy_tests;

#[cfg(test)]
#[path = "build_tests.rs"]
mod build_tests;

#[cfg(test)]
#[path = "verification_tests.rs"]
mod verification_tests;

#[cfg(test)]
#[path = "persistence_tests.rs"]
mod persistence_tests;
