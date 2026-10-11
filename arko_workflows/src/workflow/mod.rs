pub mod cancel;
pub mod config;
pub mod engine;
pub(crate) mod options;
pub mod result;
pub mod stop;

pub use options::RunOptions;

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
use crate::workflow::result::{AgentResult, AgentRunStatus, WorkflowEvent, WorkflowResult};
use crate::workflow::stop::{Restart, RunStopIntent, StopRecord, StopRequest, StoppedRun};

/// A configured multi-agent workflow.
///
/// Created via [`WorkflowBuilder`]. Provides methods to run agents sequentially,
/// in parallel, or via non-blocking spawn with manual driving.
///
/// A `Workflow` is one run. Once the run is stopped, by
/// [`cancel`](Self::cancel) or [`stop`](Self::stop) (or a
/// [`WorkflowCancelHandle`]) or a hook's `Abort`, every run call in progress
/// returns the stop's error after its agents have stopped, and every later
/// one returns it at once.
pub struct Workflow {
    #[allow(dead_code)]
    config: WorkflowConfig,
    engine: WorkflowEngine,
    context: Arc<SharedContext>,
}

/// Dropping the workflow aborts its background subagents' drivers (see
/// [`crate::subagents`]): their runs stop where they are, as after a crash.
impl Drop for Workflow {
    fn drop(&mut self) {
        self.context.subagents.abort_all();
    }
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
        // Boxed: an `AgentResult` is large, and each layer holding one
        // grows the future a host awaits on its own stack.
        Box::pin(async move {
            self.ensure_admitting().await?;
            let result = self
                .engine
                .run_agent(agent_name, prompt, &self.context)
                .await;
            self.finish_call(result, |hook| {
                hook.agent_name = Some(agent_name.to_string());
            })
            .await
        })
        .await
    }

    /// [`run_agent`](Self::run_agent) with `options`, such as a working
    /// directory of the run's own ([`RunOptions::workspace`]).
    pub async fn run_agent_with(
        &self,
        agent_name: &str,
        prompt: &str,
        options: RunOptions,
    ) -> Result<AgentResult, WorkflowError> {
        Box::pin(async move {
            self.ensure_admitting().await?;
            let result = self
                .engine
                .run_agent_with(agent_name, prompt, options, &self.context)
                .await;
            self.finish_call(result, |hook| {
                hook.agent_name = Some(agent_name.to_string());
            })
            .await
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
            && !engine::is_run_stop(error)
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

    /// `Err` with the stop's error once the run is stopped, and with
    /// `Cancelled` during a graceful stop's wrap-up: no agent run starts
    /// then.
    async fn ensure_admitting(&self) -> Result<(), WorkflowError> {
        self.ensure_running().await?;
        self.context.run_control.check_admission(false)
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
    /// `prompt`. An agent whose completion a verifier could not review
    /// ([`WorkflowError::VerifierFailed`]) continues with `prompt` as its
    /// next message.
    pub async fn continue_agent(
        &self,
        previous: &AgentResult,
        prompt: &str,
    ) -> Result<AgentResult, WorkflowError> {
        // Boxed: an `AgentResult` is large, and each layer holding one
        // grows the future a host awaits on its own stack.
        Box::pin(async move {
            self.ensure_admitting().await?;
            let result = self
                .engine
                .continue_agent(previous, prompt, &self.context)
                .await;
            self.finish_call(result, |hook| {
                hook.agent_name = Some(previous.agent_name.clone());
                hook.agent_run_id = Some(previous.agent_run_id);
            })
            .await
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
        // Boxed: an `AgentResult` is large, and each layer holding one
        // grows the future a host awaits on its own stack.
        Box::pin(async move {
            self.ensure_admitting().await?;
            let result = self
                .engine
                .resume_agent(previous, resume, &self.context)
                .await;
            self.finish_call(result, |hook| {
                hook.agent_name = Some(previous.agent_name.clone());
                hook.agent_run_id = Some(previous.agent_run_id);
            })
            .await
        })
        .await
    }

    /// The agent runs that were running when the workflow stopped (it
    /// crashed, or its process ended), as the checkpoint it resumed from
    /// recorded them: `Interrupted` results carrying their tasks as last
    /// saved, to be continued with [`recover_agent`](Self::recover_agent).
    /// With the task journal on ([`PersistenceConfig::task_journal`], the
    /// default), each was saved at its last safe point, so it continues
    /// where it was. A run that stopped on its own as interrupted (a lasting
    /// connection failure) is listed too. Runs being recovered are not.
    ///
    /// Two kinds are listed:
    /// - top-level runs (`task.parent` is `None`), which the host started;
    /// - orphaned runs (`task.parent` is set): runs started inside another
    ///   run's tool call (a delegated agent) that the run which started
    ///   them can no longer resume, because it ended, or it stopped with
    ///   that call answered (the call failed; its result named the run).
    ///   Only the host recovers them, unless an agent resumes one first
    ///   (`run_workflow_agent` with `resume_run`).
    ///
    /// A run started inside a tool call its parent still holds open is not
    /// listed: recovering the parent recovers it (or supersedes it, for a
    /// completion verifier's review or a call that can't resume). Neither
    /// is a run the host retired ([`stop_run`](Self::stop_run) with
    /// [`RunStopIntent::Retire`]) that the process ended in its wrap-up;
    /// `recover_agent` still takes it.
    pub fn interrupted_runs(&self) -> Vec<AgentResult> {
        let interrupted = self
            .context
            .agent_results
            .read()
            .iter()
            .filter(|result| {
                result.status == AgentRunStatus::Interrupted
                    && !self.context.registry.is_running(result.agent_run_id)
                    && !result
                        .stopped_by
                        .as_ref()
                        .is_some_and(StopRecord::is_retired)
            })
            .cloned()
            .collect::<Vec<_>>();
        interrupted
            .into_iter()
            .filter(|result| {
                (result.task.parent.is_none() || engine::is_orphaned(&self.context, result))
                    && !self.context.subagents.is_subagent(result.agent_run_id)
            })
            .collect()
    }

    /// Continue an agent run that was running when the workflow stopped
    /// (one of [`interrupted_runs`](Self::interrupted_runs)) where its task
    /// was saved, with its run identity and without a new message. Tool
    /// calls it was running are settled by their tools'
    /// [`recovery_mode`](arko_agents::Tool::recovery_mode): by default the
    /// agent is told the call may have run and to check its effects;
    /// read-only built-in tools run again; `run_workflow_agent` recovers the
    /// agent run it started (or returns its result if it had finished), and
    /// a `complete_task` call is reviewed again by its completion verifiers.
    /// A wait it was saved in (a tool approval, an error-recovery pause, a
    /// question) is answered again under the workflow's policies. Returns as
    /// [`run_agent`](Self::run_agent) does. The agent's configuration must
    /// be the one it was saved with (`build` checks it).
    ///
    /// An orphaned run (listed with `task.parent` set) is recovered as the
    /// host's own: its parent link is released, so its result is returned
    /// here only. A run whose parent can still resume it is refused, and so
    /// is a run being recovered already ("is running already"): a run is
    /// never recovered twice. A run that was recovered (or resumed by an
    /// agent) since `previous` was listed returns the outcome it recorded,
    /// so recovering each run of an earlier `interrupted_runs` list is safe.
    pub async fn recover_agent(
        &self,
        previous: &AgentResult,
    ) -> Result<AgentResult, WorkflowError> {
        // Boxed: an `AgentResult` is large, and each layer holding one
        // grows the future a host awaits on its own stack.
        Box::pin(async move {
            self.ensure_admitting().await?;
            let result = self.engine.recover_agent(previous, &self.context).await;
            self.finish_call(result, |hook| {
                hook.agent_name = Some(previous.agent_name.clone());
                hook.agent_run_id = Some(previous.agent_run_id);
            })
            .await
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
        self.ensure_admitting().await?;
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
        self.spawn_agent_with(agent_name, prompt, RunOptions::default())
            .await
    }

    /// [`spawn_agent`](Self::spawn_agent) with `options`, such as a working
    /// directory of the run's own ([`RunOptions::workspace`]).
    pub async fn spawn_agent_with(
        &self,
        agent_name: &str,
        prompt: &str,
        options: RunOptions,
    ) -> Result<AgentHandle, WorkflowError> {
        self.ensure_admitting().await?;
        self.engine
            .spawn_agent_with(agent_name, prompt, options, &self.context)
            .await
    }

    /// Every background subagent of the run, in the order they were started
    /// (see [`crate::subagents`]).
    pub fn subagents(&self) -> Vec<crate::subagents::SubagentRecord> {
        self.context.subagents.records()
    }

    /// The background subagents running, or queued to run.
    pub fn active_subagents(&self) -> Vec<crate::subagents::SubagentRecord> {
        self.context.subagents.active()
    }

    /// Stops a queued or running background subagent and reports the stop
    /// to its lead. A queued run stops before it starts. A running run uses
    /// `request` with [`RunStopIntent::Relaunch`], so a later message from
    /// its lead can restart it. Returns whether this call changed its stop.
    pub fn stop_subagent(&self, run_id: uuid::Uuid, request: StopRequest) -> bool {
        self.context
            .subagents
            .stop_subagent(&self.context, run_id, request)
    }

    /// Background subagent events: starts, reports, failures, stops,
    /// messages, deliveries, and
    /// [`NewsWaiting`](crate::subagents::SubagentEventKind::NewsWaiting) for
    /// a run that isn't running and has news. Also recorded in the event
    /// log. Their order relative to [`subscribe_events`](Self::subscribe_events)'
    /// events isn't guaranteed.
    pub fn subscribe_subagents(
        &self,
    ) -> tokio::sync::broadcast::Receiver<crate::subagents::SubagentEvent> {
        self.context.subagents.subscribe()
    }

    /// The mailbox items for agent run `agent_run_id` it hasn't received:
    /// its subagents' reports, failures, stops and messages, oldest first.
    /// A run that is continued gets them at its first turn boundary.
    pub fn pending_news(&self, agent_run_id: uuid::Uuid) -> Vec<crate::subagents::MailItem> {
        self.context.subagents.pending(agent_run_id)
    }

    /// Takes the items [`pending_news`](Self::pending_news) lists out of the
    /// mailbox, for a host that puts them in its own message to the run
    /// ([`MailItem::text`](crate::subagents::MailItem::text) is each one
    /// whole).
    pub fn take_news(&self, agent_run_id: uuid::Uuid) -> Vec<crate::subagents::MailItem> {
        self.context.subagents.take(&self.context, agent_run_id)
    }

    /// Relaunches the background subagents that were running, or queued,
    /// when the workflow stopped (after a resume from a checkpoint), in the
    /// background: returns their ids (`r1`, ...). Their reports reach their
    /// leads as usual. A subagent that isn't relaunched waits, and a message
    /// from its lead restarts it.
    pub fn resume_subagents(&self) -> Vec<String> {
        self.context.subagents.resume_all(&self.context)
    }

    /// Agent run `from`'s background subagents become run `to`'s (a new
    /// conversation of the same lead, say), with the news for `from` it
    /// hasn't received. Returns how many.
    pub fn transfer_subagents(&self, from: uuid::Uuid, to: uuid::Uuid) -> usize {
        self.context.subagents.transfer(from, to)
    }

    /// The number of the newest note agent run `agent_run_id` has been
    /// given; see [`NotesHandle::given`](crate::notes::NotesHandle::given).
    pub fn notes_given(&self, agent_run_id: uuid::Uuid) -> u64 {
        self.context.notes.given(agent_run_id)
    }

    /// A handle that cancels or stops this run from any task or thread.
    pub fn cancel_handle(&self) -> WorkflowCancelHandle {
        WorkflowCancelHandle::new(&self.context)
    }

    /// Cancels the run; see [`WorkflowCancelHandle::cancel`]. Returns
    /// `false` when the run was already stopped.
    pub fn cancel(&self) -> bool {
        self.cancel_handle().cancel()
    }

    /// Stops the run for `request`'s [kind](crate::workflow::stop::StopKind)
    /// and reason; see the [`stop`](crate::workflow::stop) module.
    ///
    /// Without a wrap-up (the default), the run is cancelled at once, as
    /// [`cancel`](Self::cancel) does, with
    /// [`CancelCause::Stopped`](crate::CancelCause::Stopped).
    ///
    /// With a [wrap-up](StopRequest::wrap_up), first:
    /// - no new agent run starts: run calls return `Cancelled` (only a
    ///   completion verifier's review may start, of a completion an agent
    ///   made before it was told);
    /// - the tools of running agents see their stop signal at once
    ///   ([`TaskContext::stop`](arko_agents::TaskContext::stop)): a
    ///   [`StopAwareCommand`](arko_agents::StopAwareCommand) creates its
    ///   stop file, and `wait_for_message` returns;
    /// - each running agent (other than a completion verifier's review) is
    ///   told at its next turn boundary to save its progress and call
    ///   `complete_task`
    ///   ([`wrap_up_notice`](crate::workflow::stop::wrap_up_notice), or the
    ///   request's message). Completion verifiers don't review a completion
    ///   made after the notice: it reports progress (the run is marked, and
    ///   is continued after a restart).
    ///
    /// The wrap-up ends when no agent of the run is running any more, or
    /// after its time; then the run is cancelled with the stop's cause, its
    /// agents get the [grace period](WorkflowBuilder::cancel_grace_period),
    /// and the checkpoint records the stop. Run calls of agents that
    /// completed during the wrap-up return their results; once the run is
    /// stopped, calls return `Cancelled`. Results the stop ended are marked
    /// ([`AgentResult::stopped_by`]).
    ///
    /// Returns `false`, changing nothing, when the run is stopped already,
    /// or when it is wrapping up and `request` has a wrap-up too. A request
    /// without a wrap-up during a wrap-up ends the wrap-up at once; the
    /// first stop's cause stands. `OnWorkflowCancelled` fires when the run
    /// is stopped, with `stop_kind` and `reason` metadata. Timing a wrap-up
    /// needs a Tokio runtime (the current one, or the one the handle was
    /// made in); without one, the run is stopped at once.
    pub fn stop(&self, request: StopRequest) -> bool {
        self.cancel_handle().stop(request)
    }

    /// Stops agent run `agent_run_id` alone, for `request`'s kind and
    /// reason, while the rest of the workflow runs on: other runs carry on,
    /// and new runs start as usual. `intent` says what the host means to do
    /// with the run afterwards, and is saved with the stop
    /// ([`StopRecord::intent`](crate::workflow::stop::StopRecord::intent)):
    /// [`RunStopIntent::Retire`] (removed, or replaced by another run) or
    /// [`RunStopIntent::Relaunch`] (listed by
    /// [`stopped_runs`](Self::stopped_runs) for [`restart`](Self::restart)).
    ///
    /// With a [wrap-up](StopRequest::wrap_up), the run's tools see their
    /// stop signal at once, and the run is told at its next turn boundary
    /// to save its progress and call `complete_task`
    /// ([`run_wrap_up_notice`](crate::workflow::stop::run_wrap_up_notice),
    /// or the request's message). A completion after the notice reports
    /// progress: completion verifiers don't review it, and the run call
    /// returns the result, with [`AgentResult::stopped_by`] set. When the
    /// wrap-up's time is up first, or without a wrap-up, the run is
    /// cancelled, within the [grace period](WorkflowBuilder::cancel_grace_period)
    /// as a stopped workflow's agents are, and its run call returns
    /// [`WorkflowError::AgentStopped`] (its result is `Cancelled`, with
    /// `stopped_by` set). Neither is a failure: `OnWorkflowFailed` doesn't
    /// fire, `fail_fast` isn't triggered, and `run_agents_parallel` returns
    /// the result in its place.
    ///
    /// The runs inside the run's tool calls (agents it delegated to with
    /// `run_workflow_agent`) are stopped with it, with the same request and
    /// intent, and the run can't start new ones (one that starts all the
    /// same, from a call already under way, takes over the stop). They are
    /// cancelled when the run is. A delegated run stopped
    /// alone returns its progress report to its caller, followed by a note
    /// naming the run and saying whether the caller can resume it (not when
    /// it was retired).
    ///
    /// Returns `false`, changing nothing, when the run isn't running, is a
    /// completion verifier's review (stop the run under review instead),
    /// the workflow is stopped, or the run is being stopped alone already
    /// and `request` has a wrap-up too. A request without a wrap-up during
    /// the run's wrap-up cancels it at once; the first stop's record stands.
    /// With persistence and the task journal, the stop is saved with the
    /// live checkpoint at once, so a resume keeps the host's intent for a
    /// run the process ended in its wrap-up.
    pub fn stop_run(
        &self,
        agent_run_id: uuid::Uuid,
        request: StopRequest,
        intent: RunStopIntent,
    ) -> bool {
        self.cancel_handle().stop_run(agent_run_id, request, intent)
    }

    /// Stops every running run of `agent_name` alone, as
    /// [`stop_run`](Self::stop_run) does, and returns the runs whose stop
    /// began (none when none runs). An unknown agent returns
    /// `AgentNotFound`.
    pub fn stop_agent(
        &self,
        agent_name: &str,
        request: StopRequest,
        intent: RunStopIntent,
    ) -> Result<Vec<uuid::Uuid>, WorkflowError> {
        self.cancel_handle().stop_agent(agent_name, request, intent)
    }

    /// Whether a graceful stop's wrap-up is under way: the run is stopping
    /// but not stopped yet ([`is_cancelled`](Self::is_cancelled) is still
    /// `false`).
    pub fn is_stopping(&self) -> bool {
        self.context.run_control.is_wrapping_up()
    }

    /// The runs to relaunch after resuming a stopped run (with
    /// [`WorkflowBuilder::resume_from`]), or, while the workflow runs, after
    /// stopping runs alone to relaunch them ([`stop_run`](Self::stop_run)
    /// with [`RunStopIntent::Relaunch`]), in the order they were recorded,
    /// each with how [`restart`](Self::restart) relaunches it:
    /// - runs a stop cancelled ([`AgentResult::stopped_by`] set, status
    ///   `Cancelled`): [`Restart::Reopen`];
    /// - runs that completed in a stop's wrap-up (`stopped_by` set, status
    ///   `Completed`), started by the host: [`Restart::Continue`];
    /// - the [`interrupted_runs`](Self::interrupted_runs) (the process
    ///   ended while they ran, before or during a stop):
    ///   [`Restart::Recover`].
    ///
    /// Runs the host retired ([`RunStopIntent::Retire`]) are never listed,
    /// also when a later stop of the workflow ended them in their wrap-up.
    ///
    /// A run started inside another run's tool call is listed only when
    /// that run can no longer resume it, as for `interrupted_runs`, and
    /// isn't listed itself: restarted, that run resumes it (a call it was
    /// running), or its call failed naming the run, so that it can resume it
    /// with `run_workflow_agent`'s `resume_run`. Runs running now, runs
    /// superseded after a restart, and runs paused for the caller (which a
    /// stop doesn't end) are not listed. A
    /// restarted run's new result replaces its old one, so it is no longer
    /// listed unless a stop ends it again. Runs recorded by versions from
    /// before graceful stops have no `stopped_by`: only their interrupted
    /// runs are listed (continue cancelled ones with
    /// [`reopen_agent`](Self::reopen_agent)).
    pub fn stopped_runs(&self) -> Vec<StoppedRun> {
        let interrupted = self
            .interrupted_runs()
            .into_iter()
            .map(|result| result.agent_run_id)
            .collect::<HashSet<_>>();
        let results = self.context.agent_results.read().clone();
        let candidates = results
            .into_iter()
            .filter_map(|result| {
                let restart = if result
                    .stopped_by
                    .as_ref()
                    .is_some_and(StopRecord::is_retired)
                    || self.context.subagents.is_subagent(result.agent_run_id)
                {
                    return None;
                } else if interrupted.contains(&result.agent_run_id) {
                    Restart::Recover
                } else if result.stopped_by.is_none()
                    || self.context.registry.is_running(result.agent_run_id)
                    || engine::is_superseded(&result)
                {
                    return None;
                } else {
                    match result.status {
                        AgentRunStatus::Cancelled
                            if result.task.parent.is_none()
                                || engine::is_orphaned(&self.context, &result) =>
                        {
                            Restart::Reopen
                        }
                        AgentRunStatus::Completed if result.task.parent.is_none() => {
                            Restart::Continue
                        }
                        _ => return None,
                    }
                };
                Some(StoppedRun { result, restart })
            })
            .collect::<Vec<_>>();
        // A run started inside the tool call of a run listed too is left to
        // that run: restarted, it resumes it (a call it was running), or was
        // told its ID (a call that failed with it) to resume it with
        // `run_workflow_agent`'s `resume_run`.
        let listed = candidates
            .iter()
            .map(|run| run.result.agent_run_id)
            .collect::<HashSet<_>>();
        candidates
            .into_iter()
            .filter(|run| {
                run.result
                    .task
                    .parent
                    .is_none_or(|parent| !listed.contains(&parent.task_id))
            })
            .collect()
    }

    /// Relaunches a run [`stopped_runs`](Self::stopped_runs) listed, as its
    /// [`Restart`] says: [`recover_agent`](Self::recover_agent) (without a
    /// message; `note` is not used),
    /// [`reopen_agent`](Self::reopen_agent) with `note` after its reopened
    /// notice, or [`continue_agent`](Self::continue_agent) with a message
    /// saying the workflow (or, for a run stopped alone, the run) was
    /// stopped and has restarted, then `note`. `note` may be empty. Returns
    /// as those do. A retired run ([`RunStopIntent::Retire`]) returns
    /// `BuilderInvalid`.
    pub async fn restart(
        &self,
        run: &StoppedRun,
        note: &str,
    ) -> Result<AgentResult, WorkflowError> {
        if run
            .result
            .stopped_by
            .as_ref()
            .is_some_and(StopRecord::is_retired)
        {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent run {} of agent '{}' was retired (Workflow::stop_run with RunStopIntent::Retire), so restart doesn't relaunch it",
                run.result.agent_run_id, run.result.agent_name
            )));
        }
        match run.restart {
            Restart::Recover => self.recover_agent(&run.result).await,
            Restart::Reopen => self.reopen_agent(&run.result, note).await,
            Restart::Continue => {
                let message = crate::workflow::stop::continued_message(&run.result, note);
                self.continue_agent(&run.result, &message).await
            }
        }
    }

    /// Whether the run is stopped, by cancellation, a stop or a hook's
    /// abort.
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
            if engine::reports_exceeded(&self.context, &cost_result) {
                fire_hooks_on_context(&self.context, HookPoint::OnBudgetExceeded, |hctx| {
                    hctx.budget_spent = cost_result.new_total;
                })?;
            }
            return Err(WorkflowError::BudgetExceeded {
                spent: cost_result.new_total,
                limit: self.context.budget.limit().unwrap_or(0.0),
            });
        }

        Ok(cost_result)
    }

    /// Changes the workflow's budget limit, in dollars (`None`: no limit),
    /// also while agents run: it applies from their next model requests.
    /// Under [`BudgetExceededPolicy::Pause`](crate::budget::BudgetExceededPolicy::Pause)
    /// (the default), raising it releases the agents held on it: runs
    /// waiting for a raise continue at once, and a run returned as
    /// [`WorkflowError::AgentPaused`] continues with
    /// [`resume_agent`](Self::resume_agent). Lowering it holds agents at
    /// their next request once the spend is over it. The limit is saved
    /// with the next checkpoint, so a resume keeps it (see
    /// [`WorkflowBuilder::resume_from`]). `OnBudgetExceeded` fires again
    /// when the spend goes over the new limit.
    pub fn set_budget(&self, limit: Option<f64>) -> Result<(), WorkflowError> {
        if let Some(limit) = limit
            && (!limit.is_finite() || limit < 0.0)
        {
            return Err(WorkflowError::BuilderInvalid(
                "budget limit must be finite and non-negative".to_string(),
            ));
        }
        self.context.budget.set_limit(limit);
        self.budget_changed();
        Ok(())
    }

    /// Changes `agent`'s own budget limit, in dollars (`None`: none), also
    /// while it runs, as [`set_budget`](Self::set_budget) does for the
    /// workflow's.
    pub fn set_agent_budget(&self, agent: &str, limit: Option<f64>) -> Result<(), WorkflowError> {
        if !self.context.agent_configs.contains_key(agent) {
            return Err(WorkflowError::AgentNotFound(agent.to_string()));
        }
        match limit {
            Some(limit) if !limit.is_finite() || limit < 0.0 => {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "budget for agent '{agent}' must be finite and non-negative"
                )));
            }
            Some(limit) => self.context.budget.set_agent_limit(agent, limit),
            None => self.context.budget.clear_agent_limit(agent),
        }
        self.budget_changed();
        Ok(())
    }

    /// The workflow's budget limit in dollars, if it has one.
    pub fn budget(&self) -> Option<f64> {
        self.context.budget.limit()
    }

    /// Changes the plan budget's limits (see
    /// [`WorkflowBuilder::plan_budget`]), also while agents run, as
    /// [`set_budget`](Self::set_budget) does for the dollar budget: raising
    /// a limit releases the agents held on it, and the limits are saved
    /// with the next checkpoint.
    pub fn set_plan_budget(&self, budget: crate::budget::PlanBudget) -> Result<(), WorkflowError> {
        budget.validate().map_err(WorkflowError::BuilderInvalid)?;
        self.context.budget.set_plan_limits(budget);
        self.budget_changed();
        Ok(())
    }

    /// The plan budget's limits.
    pub fn plan_budget(&self) -> crate::budget::PlanBudget {
        self.context.budget.plan_limits()
    }

    /// The points of each plan window the run has used so far, counted
    /// from the levels it first saw (see [`WorkflowBuilder::plan_budget`]):
    /// with several plans, the most any plan has used of each window (see
    /// [`plan_usage_spent_of`](Self::plan_usage_spent_of)). `None` for a
    /// window no report has shown yet.
    pub fn plan_usage_spent(&self) -> crate::budget::PlanUsageSpent {
        self.context.budget.plan_spent()
    }

    /// The points of each of `plan`'s windows the run has used so far, as
    /// [`plan_usage_spent`](Self::plan_usage_spent) counts them; `None` for
    /// a window no report of the plan has shown yet.
    pub fn plan_usage_spent_of(
        &self,
        plan: arko_ai::SubscriptionPlan,
    ) -> crate::budget::PlanUsageSpent {
        self.context.budget.plan_spent_of(plan)
    }

    /// With the task journal saving running agents, a live checkpoint
    /// records the changed limits (it would otherwise wait for the next
    /// safe point of an agent, which a held agent doesn't reach).
    fn budget_changed(&self) {
        let Some(persistence) = self.engine.persistence().cloned() else {
            return;
        };
        if !persistence.has_live_runs() {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let context = self.context.clone();
        runtime.spawn(async move {
            if let Err(error) = persistence.write_live_checkpoint(&context, false).await {
                log::warn!(
                    "Workflow '{}': could not save the changed budget: {error}",
                    context.workflow_name
                );
            }
        });
    }

    /// Whether the budget holds up the whole run: agents are held until
    /// the budget is raised, and no running agent can go on without them
    /// (each is held, waits for a run it delegated to, or waits for a
    /// message). A host that can't raise the budget while the run is live
    /// can stop here: the running agents were saved (with the task journal,
    /// the default under persistence), and after a resume with a higher
    /// budget, [`recover_agent`](Self::recover_agent) and
    /// [`resume_agent`](Self::resume_agent) continue them.
    pub fn is_budget_blocked(&self) -> bool {
        self.context.budget.exceeded_policy() == crate::budget::BudgetExceededPolicy::Pause
            && self.context.registry.blocked_on_budget()
    }

    /// Waits until [`is_budget_blocked`](Self::is_budget_blocked) holds.
    /// Never returns while agents make progress or none is running.
    pub async fn budget_blocked(&self) {
        let mut versions = self.context.registry.subscribe();
        while !self.is_budget_blocked() {
            if versions.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }

    /// Continue an agent run that was cancelled (`AgentRunStatus::Cancelled`),
    /// such as one the budget stopped under
    /// [`BudgetExceededPolicy::Cancel`](crate::budget::BudgetExceededPolicy::Cancel)
    /// or one stopped with the workflow, from where its task stopped, with
    /// its run identity: it is told it was reopened and that calls reported
    /// cancelled did not finish, followed by `prompt` (may be empty). Tool
    /// calls it was running without results are settled as for
    /// [`recover_agent`](Self::recover_agent). Runs recorded by earlier
    /// versions can be reopened too. A run started inside another run's
    /// tool call becomes the host's (its parent link is released), if it
    /// is orphaned as for [`recover_agent`](Self::recover_agent): a run
    /// whose parent is running, or whose parent's recovery settles it, is
    /// refused. So is a run recorded superseded (another run replaced it
    /// after a restart, such as a completion verifier's review). Under
    /// `Cancel`, raise the budget first ([`set_budget`](Self::set_budget)),
    /// or it is refused again. Returns as [`run_agent`](Self::run_agent)
    /// does.
    pub async fn reopen_agent(
        &self,
        previous: &AgentResult,
        prompt: &str,
    ) -> Result<AgentResult, WorkflowError> {
        // Boxed: an `AgentResult` is large, and each layer holding one
        // grows the future a host awaits on its own stack.
        Box::pin(async move {
            self.ensure_admitting().await?;
            let result = self
                .engine
                .reopen_agent(previous, prompt, &self.context)
                .await;
            self.finish_call(result, |hook| {
                hook.agent_name = Some(previous.agent_name.clone());
                hook.agent_run_id = Some(previous.agent_run_id);
            })
            .await
        })
        .await
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

    /// Posts a note to the agents: guidance that refines their task while
    /// the workflow runs, without changing their configuration (so a resume
    /// still accepts their saved tasks). A running agent of the audience gets
    /// it at its next safe turn boundary, in its `# Workflow context
    /// update`; a standing note (the default) is also in the initial prompt
    /// of every later run of the audience, and in the fresh conversation
    /// after a handover. See [`notes`](crate::notes).
    ///
    /// The text is trimmed; an empty one, one over
    /// [`MAX_NOTE_BYTES`](crate::notes::MAX_NOTE_BYTES), or one no agent
    /// receives returns `BuilderInvalid`, and an audience naming an
    /// unregistered agent returns `AgentNotFound`. The note is logged once
    /// (and recorded in the event log with persistence), and saved with
    /// the next checkpoint: at once while agents run with the task journal.
    pub fn post_note(
        &self,
        note: crate::notes::Note,
    ) -> Result<crate::notes::PostedNote, WorkflowError> {
        self.notes_handle().post(note)
    }

    /// Every note posted to this run, oldest first, including those restored
    /// from the checkpoint and those read from the control folder.
    pub fn notes(&self) -> Vec<crate::notes::PostedNote> {
        self.context.notes.notes(&self.context)
    }

    /// A handle that posts notes from any task while the workflow runs.
    pub fn notes_handle(&self) -> crate::notes::NotesHandle {
        crate::notes::NotesHandle::new(&self.context)
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
    /// returned ([`result`](Self::result) still returns the results). During
    /// a graceful stop's wrap-up, it first waits for the run to stop.
    pub async fn finish(&self) -> Result<WorkflowResult, WorkflowError> {
        if self.context.run_control.is_wrapping_up() {
            self.context.run_control.stopped().await;
        }
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
    /// run, and the task journal doesn't save it
    /// ([`PersistenceConfig::task_journal`]).
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
        if let Some(journal) = &run.journal {
            journal.end_live_run(run.agent_run_id);
        }
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
    plan_budget: crate::budget::PlanBudget,
    budget_exceeded_policy: crate::budget::BudgetExceededPolicy,
    wait_for_budget_raise: bool,
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
    subagents: Option<crate::subagents::SubagentsConfig>,
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
    /// Agents that get a note only when it names them.
    notes_by_name_only: std::collections::BTreeSet<String>,
    /// Read notes from the run directory's control folder.
    control_folder: bool,
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
            plan_budget: crate::budget::PlanBudget::default(),
            budget_exceeded_policy: crate::budget::BudgetExceededPolicy::default(),
            wait_for_budget_raise: false,
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
            subagents: None,
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
            notes_by_name_only: std::collections::BTreeSet::new(),
            control_folder: false,
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

    /// Set the budget limit in dollars. What happens when it is exceeded is
    /// the [`budget_exceeded_policy`](Self::budget_exceeded_policy): by
    /// default the agents pause, to go on once it is raised
    /// ([`Workflow::set_budget`]). On resume, see
    /// [`resume_from`](Self::resume_from) for how it combines with the
    /// limit the checkpoint recorded.
    pub fn budget(mut self, limit: f64) -> Self {
        self.budget_limit = Some(limit);
        self
    }

    /// Set a plan budget: how many percentage points of the subscription
    /// plans' five-hour and seven-day usage windows the run may use,
    /// counted from each window's level in the first plan usage report the
    /// run sees (Claude Code's `rate_limit_event`, ChatGPT OAuth's usage
    /// headers). Each plan's windows are counted separately: `five_hour`
    /// and `seven_day` limit each plan's, and a plan's own limits
    /// (`chatgpt`, `claude_code`) replace them for that plan. After a
    /// window resets, the points used before the reset still count and the
    /// new period counts from 0. Plan usage is account-wide: reports from
    /// every agent update the same windows of their plan, and use of the
    /// same plan outside the run while it runs counts too. The points are
    /// known once a request completes, so, as with a dollar budget, the
    /// requests in flight when the limit is reached complete and can take
    /// the run past it.
    ///
    /// Exceeding any plan's window's limit is handled as exceeding the
    /// dollar budget is (see [`budget_exceeded_policy`](Self::budget_exceeded_policy)):
    /// by default agents are held until it is raised
    /// ([`Workflow::set_plan_budget`]); `OnBudgetExceeded` fires once per
    /// crossing, with `plan` (`chatgpt` or `claude_code`), `plan_window`
    /// (`five_hour` or `seven_day`), `plan_points_used` and
    /// `plan_points_limit` in its metadata. Checkpoints record the limits
    /// and the counting, so a resumed run counts from where the run
    /// started; on resume, each limit combines with the recorded one as
    /// dollar limits do (see [`resume_from`](Self::resume_from)).
    pub fn plan_budget(mut self, budget: crate::budget::PlanBudget) -> Self {
        self.plan_budget = budget;
        self
    }

    /// What happens when the workflow's budget, or an agent's, is exceeded:
    /// [`BudgetExceededPolicy::Pause`](crate::budget::BudgetExceededPolicy::Pause)
    /// (the default) holds each agent over it before its next model
    /// request until the budget is raised;
    /// [`BudgetExceededPolicy::Cancel`](crate::budget::BudgetExceededPolicy::Cancel)
    /// cancels it.
    pub fn budget_exceeded_policy(mut self, policy: crate::budget::BudgetExceededPolicy) -> Self {
        self.budget_exceeded_policy = policy;
        self
    }

    /// Under [`BudgetExceededPolicy::Pause`](crate::budget::BudgetExceededPolicy::Pause),
    /// whether a run the host started (`run_agent` and the like) that the
    /// budget holds waits in the call until the budget is raised with
    /// [`Workflow::set_budget`] or [`Workflow::set_agent_budget`], as runs
    /// inside another agent's tool call always do. Off by default: the
    /// call returns [`WorkflowError::AgentPaused`] (cause
    /// `BudgetExceeded`), and [`Workflow::resume_agent`] continues it once
    /// the budget is raised. Turn it on when the host can raise the budget
    /// while the run is live (from another task, or a UI).
    pub fn wait_for_budget_raise(mut self, enabled: bool) -> Self {
        self.wait_for_budget_raise = enabled;
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

    /// Set `agent`'s own budget limit in dollars: its runs together may
    /// spend this much. Exceeding it is handled as exceeding the workflow's
    /// budget is (see [`budget_exceeded_policy`](Self::budget_exceeded_policy)),
    /// for that agent's runs only.
    pub fn agent_budget(mut self, agent: impl Into<String>, limit: f64) -> Self {
        self.agent_budgets.insert(agent.into(), limit);
        self
    }

    /// Explicitly disable run persistence and durable steps, and with them
    /// the [control folder](Self::control_folder), which is in the run
    /// directory.
    pub fn ephemeral(mut self) -> Self {
        self.persistence_config = None;
        self.resume_from = None;
        self.control_folder = false;
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
    /// Agent runs that were running when the checkpoint was written come
    /// back `Interrupted`, with their tasks; see
    /// [`Workflow::interrupted_runs`] and [`Workflow::recover_agent`]. The
    /// usage they reported that the checkpoint's cost ledger lacks is
    /// charged now.
    ///
    /// Budget limits: the checkpoint records the limits in force when it
    /// was written, including changes made while the run was live
    /// ([`Workflow::set_budget`], [`Workflow::set_agent_budget`]). For the
    /// workflow's limit and each agent's, the limit set on this builder
    /// applies unless the recorded one is higher, which then applies. A
    /// limit not set on the builder is none, as in a new run, whatever the
    /// checkpoint recorded; a checkpoint written before limits were
    /// recorded leaves the builder's. So resuming with a higher budget
    /// raises it, and resuming with the budget the run was built with keeps
    /// a raise made while it was live (a lowering is not kept). Each limit
    /// of a [`plan_budget`](Self::plan_budget) (each window's, and each
    /// plan's own) follows the same rule, and the resumed run keeps
    /// counting plan usage from the levels the run first saw, not from the
    /// levels at the resume. The counting of a checkpoint written before
    /// plans were counted apart is taken over by the first plan to report.
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

    /// Lets the workflow services' callers start background subagents of
    /// their targets: runs that work while the caller goes on, with
    /// messages between them (see [`crate::subagents`]). Needs
    /// [`workflow_services`](Self::workflow_services).
    pub fn subagents(mut self, config: crate::subagents::SubagentsConfig) -> Self {
        self.subagents = Some(config);
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

    /// These agents (registration names) get a note ([`Workflow::post_note`])
    /// only when its audience names them, as completion verifiers do: roles
    /// that must stay independent of the operator's steering, such as a
    /// checker. Adds to earlier calls. Default: none besides the verifiers'
    /// reviews.
    pub fn notes_by_name_only<I, S>(mut self, agents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.notes_by_name_only
            .extend(agents.into_iter().map(Into::into));
        self
    }

    /// Opt-in (default off): read notes from the run directory's control
    /// folder, `<run dir>/control/`, which another process writes with
    /// [`notes::control::post_note`](crate::notes::control::post_note) (a
    /// binary's `note` subcommand, or the `runctl` example) while the
    /// workflow runs or is stopped. `build` creates the folder. The
    /// workflow reads it at its agents' turn boundaries and when it starts
    /// an agent, and posts each note as [`Workflow::post_note`] does.
    /// While agents run, it also reads stop requests from the folder
    /// ([`notes::control::request_stop`](crate::notes::control::request_stop))
    /// every second and stops the runs they name, as
    /// [`Workflow::stop_run`] does. Requires
    /// [`persistence`](Self::persistence). Whoever can write the folder can
    /// post notes and stop runs: keep the run directory out of the agents'
    /// reach.
    pub fn control_folder(mut self) -> Self {
        self.control_folder = true;
        self
    }

    /// How long `wait_for_message` waits at most, for every agent without its
    /// own setting ([`agent_wait_for_message_timeout`](Self::agent_wait_for_message_timeout)).
    /// Default [`WaitForMessageTimeout::Auto`]: just inside the agent's
    /// prompt-cache lifetime, so the request after a wait still finds the
    /// prompt cached: 4 minutes 30 seconds, 27 minutes over ChatGPT OAuth, or
    /// 54 minutes with Anthropic's 1-hour TTL. An agent waits longer by
    /// calling the tool again. A
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
    /// human input (the first such call of each agent is logged as a
    /// warning). `RespectAgentConfig` pauses the agent for the caller;
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
    /// call returns that failure. A pause is not a failure, including a
    /// completion its verifier could not review. Off, every agent
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
        self.plan_budget
            .validate()
            .map_err(WorkflowError::BuilderInvalid)?;

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

        // Claude Code runs its built-in tools itself: their output can't be
        // replaced.
        for distiller in &self.distillation_configs {
            let claude_code_target = self.agents.iter().any(|(name, config)| {
                distiller.target_agents.contains(name) && config.runs_in_claude_code()
            });
            if let Some(tool) = distiller
                .target_tools
                .iter()
                .find(|tool| arko_agents::claude_code::is_built_in_tool(tool))
                .filter(|_| claude_code_target)
            {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "distiller '{}' targets Claude Code's '{tool}', whose output can't be distilled: Claude Code runs its built-in tools itself.",
                    distiller.name
                )));
            }
        }

        if let Some(config) = &self.memory_config {
            config.validate(&agent_names)?;
        }
        if let Some(config) = &self.persistence_config {
            config.validate()?;
        }
        if let Some(services) = &self.services_config {
            services.validate(&agent_names)?;
            if services.allowed_callers.is_empty() && !services.all_callers {
                log::warn!(
                    "Workflow '{name}': workflow services name no caller, so no agent is offered run_workflow_agent (WorkflowServicesConfig::allow_caller)"
                );
            }
        }
        if let Some(subagents) = &self.subagents {
            if self.services_config.is_none() {
                return Err(WorkflowError::BuilderInvalid(
                    "subagents need workflow_services naming who may start which agent".to_string(),
                ));
            }
            subagents.validate()?;
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
        if let Some(unknown) = self
            .notes_by_name_only
            .iter()
            .find(|agent| !agent_names.contains(*agent))
        {
            return Err(WorkflowError::BuilderInvalid(format!(
                "notes_by_name_only names unknown agent '{unknown}'"
            )));
        }
        if self.control_folder && self.persistence_config.is_none() {
            return Err(WorkflowError::BuilderInvalid(
                "control_folder requires persistence: the control folder is in the run directory"
                    .to_string(),
            ));
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
            ) || (self.subagents.is_some()
                && crate::subagents::SUBAGENT_TOOL_NAMES.contains(&tool_name.as_str()))
                || (matches!(
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
        // A subagent's saved configuration is its agent's as the host's hook
        // changes it.
        let restored_subagents = resume_state
            .as_ref()
            .and_then(|state| state.subagents.clone());
        let mut prepared = HashMap::new();
        if let (Some(snapshot), Some(hook)) = (
            &restored_subagents,
            self.subagents
                .as_ref()
                .and_then(|config| config.hook.as_ref()),
        ) {
            for record in &snapshot.runs {
                let Some(mut config) = self
                    .agents
                    .iter()
                    .find(|(agent, _)| *agent == record.agent)
                    .map(|(_, config)| config.clone())
                else {
                    continue;
                };
                let request = crate::subagents::StartRequest {
                    workflow_run_id: resume_state
                        .as_ref()
                        .map_or_else(uuid::Uuid::nil, |state| state.workflow_run_id),
                    run_id: record.run_id,
                    handle: record.handle.clone(),
                    agent: record.agent.clone(),
                    caller_run_id: record.caller_run_id,
                    caller_agent: record.caller_agent.clone(),
                    task: record.task.clone(),
                    parameters: record.parameters.clone(),
                };
                if hook.prepare(&request, &mut config).is_ok() {
                    prepared.insert(record.run_id, config);
                }
            }
        }
        validate_resumed_agent_configs(
            &self.agents,
            &self.global_tools,
            &self.agent_tools,
            &resumed_results,
            &self
                .completion_verifiers
                .iter()
                .map(|verifier| verifier.verifier_agent.as_str())
                .collect(),
            &prepared,
        )?;
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
            .map(|config| PersistenceManager::new(config, workflow_run_id))
            .transpose()?;
        if let (Some(persistence), Some(state), Some(path)) =
            (&persistence, &resume_state, &self.resume_from)
        {
            // The tasks were loaded from these versions, which later
            // checkpoints refer to when they are in this run directory.
            let run_dir = path.parent().unwrap_or(std::path::Path::new("."));
            let same_dir = |a: &std::path::Path, b: &std::path::Path| {
                a.canonicalize()
                    .ok()
                    .is_some_and(|a| Some(a) == b.canonicalize().ok())
            };
            if same_dir(run_dir, &persistence.config().checkpoint_dir) {
                persistence.mark_tasks_saved(
                    state
                        .agent_runs
                        .iter()
                        .map(|run| (run.agent_run_id, run.task_file.clone())),
                );
            }
        }
        let durable_store = persistence.as_ref().map(PersistenceManager::durable_store);
        let control_folder = match &persistence {
            Some(persistence) if self.control_folder => {
                let folder = crate::notes::control::folder(&persistence.config().checkpoint_dir);
                std::fs::create_dir_all(&folder)?;
                Some(folder)
            }
            _ => None,
        };

        // The limits: the builder's, or the checkpoint's where it recorded
        // a higher one (see `resume_from`).
        let recorded_limits = resume_state
            .as_ref()
            .and_then(|state| state.budget_limits.clone());
        let budget_limit = match &recorded_limits {
            Some(recorded) => resumed_limit(self.budget_limit, recorded.limit),
            None => self.budget_limit,
        };
        let mut agent_budgets = self.agent_budgets.clone();
        if let Some(recorded) = &recorded_limits {
            for (agent, limit) in &mut agent_budgets {
                if let Some(recorded) = recorded.agent_limits.get(agent) {
                    *limit = limit.max(*recorded);
                }
            }
        }
        let budget = if let Some(state) = &resume_state {
            let budget = BudgetManager::from_ledger(
                budget_limit,
                self.budget_warning,
                state.cost_ledger.clone(),
            );
            // A run that was running may have reported usage the ledger
            // didn't record yet.
            for record in &state.agent_runs {
                let Some(charged) = record.usage_charged else {
                    continue;
                };
                if let Some(result) = resumed_results
                    .iter()
                    .find(|result| result.agent_run_id == record.agent_run_id)
                {
                    for usage in result.task.usage_history.iter().skip(charged) {
                        budget.add_agent_cost(&record.agent_name, usage);
                    }
                }
            }
            budget
        } else {
            BudgetManager::new(budget_limit, self.budget_warning)
        };
        let budget =
            budget.with_exceeded_policy(self.budget_exceeded_policy, self.wait_for_budget_raise);
        for (agent, limit) in agent_budgets {
            budget.set_agent_limit(agent, limit);
        }
        let recorded_plan = resume_state
            .as_ref()
            .and_then(|state| state.plan_budget.clone());
        let mut plan_budget = self.plan_budget;
        if let Some(recorded) = &recorded_plan {
            for (plan, window) in crate::budget::PlanBudget::slots() {
                plan_budget.set(
                    plan,
                    window,
                    resumed_limit(
                        plan_budget.own(plan, window),
                        recorded.limits.own(plan, window),
                    ),
                );
            }
        }
        budget.set_plan_limits(plan_budget);
        if let Some(recorded) = recorded_plan {
            budget.restore_plan_usage(recorded);
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
            persistence: persistence.clone(),
            control_stops: crate::notes::control::StopReader::new(control_folder.clone()),
            registry: match &resume_state {
                Some(state) => {
                    AgentRegistry::restore(state.agent_registry.clone(), &resumed_results)
                }
                None => AgentRegistry::default(),
            },
            notes: crate::notes::NoteBoard::new(
                resume_state.as_ref().and_then(|state| state.notes.clone()),
                self.notes_by_name_only,
                control_folder,
                persistence.clone(),
            ),
            subagents: crate::subagents::Board::new(
                self.subagents,
                restored_subagents,
                &resumed_results,
                workflow_run_id,
                persistence.clone(),
            ),
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

/// A limit on resume: the builder's (`configured`) unless the checkpoint
/// recorded a higher one; none when the builder sets none.
fn resumed_limit(configured: Option<f64>, recorded: Option<f64>) -> Option<f64> {
    match (configured, recorded) {
        (Some(configured), Some(recorded)) => Some(configured.max(recorded)),
        (configured, _) => configured,
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

/// A task that can be resumed (or restarted after a stop cancelled it) must
/// keep the settings that define the agent's behavior. The task stores the prepared configuration, so compare it with
/// the builder's user configuration after removing workflow-owned additions.
/// Model registry snapshots are deliberately ignored: the provider and model
/// id remain authoritative, while the registry may gain metadata between
/// checkpoints. An interrupted run no one can reach, because recovering the
/// run that started it supersedes it ([`superseded_on_recovery`]), is not
/// checked: it never runs again.
fn validate_resumed_agent_configs(
    agents: &[(String, AgentConfig)],
    global_tools: &[Arc<dyn arko_agents::tool::Tool>],
    agent_tools: &HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>,
    resumed_results: &[AgentResult],
    verifier_agents: &HashSet<&str>,
    prepared: &HashMap<uuid::Uuid, AgentConfig>,
) -> Result<(), WorkflowError> {
    let by_id = resumed_results
        .iter()
        .map(|result| (result.agent_run_id, result))
        .collect::<HashMap<_, _>>();
    let configured = agents
        .iter()
        .map(|(name, config)| (name, config))
        .collect::<HashMap<_, _>>();
    let mut workflow_tools = [
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
    .into_iter()
    .chain(crate::subagents::SUBAGENT_TOOL_NAMES.iter().copied())
    .map(str::to_string)
    .collect::<HashSet<_>>();
    workflow_tools.extend(global_tools.iter().map(|tool| tool.name()));
    workflow_tools.extend(agent_tools.values().flatten().map(|tool| tool.name()));

    for result in resumed_results {
        // Runs that can be continued, and runs a stop cancelled (restarted
        // after the resume; see `Workflow::stopped_runs`). Runs cancelled
        // otherwise, reopened only on request, keep the configuration saved
        // in their task.
        let continued = matches!(
            result.status,
            AgentRunStatus::Completed
                | AgentRunStatus::WaitingForInput
                | AgentRunStatus::Paused
                | AgentRunStatus::Interrupted
        );
        let stopped = result.status == AgentRunStatus::Cancelled
            && result.stopped_by.is_some()
            && !engine::is_superseded(result);
        if !continued && !stopped {
            continue;
        }
        let Some(current) = prepared
            .get(&result.agent_run_id)
            .or_else(|| configured.get(&result.agent_name).copied())
        else {
            continue;
        };
        if result.status == AgentRunStatus::Interrupted
            && superseded_on_recovery(result, &by_id, verifier_agents, 0)
        {
            continue;
        }
        let current = normalized_resume_agent_config(current, &workflow_tools)?;
        let saved = normalized_resume_agent_config(&result.task.agent_config, &workflow_tools)?;
        if current != saved {
            let mut changed = Vec::new();
            changed_fields(&saved, &current, "", &mut changed);
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{}' configuration changed since its checkpoint ({} changed); use the original configuration to resume it or register the changed configuration under another agent name",
                result.agent_name,
                describe_changed_fields(&changed)
            )));
        }
    }
    Ok(())
}

/// Whether interrupted run `result` is superseded, not run again, when the
/// run that started it is recovered (see `WorkflowEngine::recover_at`): its
/// parent is interrupted and holds the call it ran in open without resuming
/// it (a completion verifier's review, a call the host answered with its
/// own output), or its parent is superseded itself. A run whose parent
/// resumes it is superseded only with its parent, and an orphaned run (its
/// parent ended, or answered the call) is not.
fn superseded_on_recovery(
    result: &AgentResult,
    by_id: &HashMap<uuid::Uuid, &AgentResult>,
    verifier_agents: &HashSet<&str>,
    depth: usize,
) -> bool {
    let Some(link) = result.task.parent else {
        return false;
    };
    let Some(parent) = by_id.get(&link.task_id) else {
        return false;
    };
    if parent.status != AgentRunStatus::Interrupted {
        return false;
    }
    // Bounded, against a cycle of parent links in a damaged checkpoint.
    if depth < 64 && superseded_on_recovery(parent, by_id, verifier_agents, depth + 1) {
        return true;
    }
    let call = parent
        .task
        .running_tool_calls
        .iter()
        .find(|call| call.tool_call_id == link.tool_call_id && call.output.is_none());
    let resumed = call.is_some_and(|call| {
        call.tool_name == RUN_WORKFLOW_AGENT_TOOL_NAME && call.custom_output.is_none()
    });
    let settled = call.is_some() || verifier_agents.contains(result.agent_name.as_str());
    settled && !resumed
}

/// The paths of the fields that differ between two configurations as JSON
/// (`instructions`, `tool_config.available_tool_names`,
/// `llm_config.generation_config.max_tokens`), appended to `changed` in
/// key order: an object is compared field by field, anything else (an
/// array, a value) as a whole.
fn changed_fields(
    saved: &serde_json::Value,
    current: &serde_json::Value,
    path: &str,
    changed: &mut Vec<String>,
) {
    use serde_json::Value;
    match (saved, current) {
        (Value::Object(saved), Value::Object(current)) => {
            let keys: std::collections::BTreeSet<&String> =
                saved.keys().chain(current.keys()).collect();
            for key in keys {
                let field = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                match (saved.get(key), current.get(key)) {
                    (Some(saved), Some(current)) => {
                        changed_fields(saved, current, &field, changed);
                    }
                    (saved, current) if saved != current => changed.push(field),
                    _ => {}
                }
            }
        }
        (saved, current) if saved != current => {
            changed.push(if path.is_empty() {
                "the configuration".to_string()
            } else {
                path.to_string()
            });
        }
        _ => {}
    }
}

/// The changed fields as one line: the first ten, then how many more.
fn describe_changed_fields(changed: &[String]) -> String {
    const SHOWN: usize = 10;
    let mut description = changed
        .iter()
        .take(SHOWN)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if changed.len() > SHOWN {
        description.push_str(&format!(" and {} more", changed.len() - SHOWN));
    }
    description
}

fn normalized_resume_agent_config(
    config: &AgentConfig,
    workflow_tools: &HashSet<String>,
) -> Result<serde_json::Value, WorkflowError> {
    let mut value = serde_json::to_value(config).map_err(|error| {
        WorkflowError::BuilderInvalid(format!(
            "could not compare agent configuration while resuming: {error}"
        ))
    })?;
    let object = value
        .as_object_mut()
        .expect("AgentConfig serializes as an object");
    if let Some(tool_config) = object
        .get_mut("tool_config")
        .and_then(|value| value.as_object_mut())
    {
        if let Some(names) = tool_config
            .get_mut("available_tool_names")
            .and_then(|value| value.as_array_mut())
        {
            names.retain(|name| {
                name.as_str()
                    .is_none_or(|name| !workflow_tools.contains(name))
            });
        }
        if let Some(overrides) = tool_config
            .get_mut("tool_specific_configs")
            .and_then(|value| value.as_object_mut())
        {
            overrides.retain(|name, _| !workflow_tools.contains(name));
        }
    }
    for key in ["llm_config", "refusal_fallback"] {
        if let Some(model_config) = object.get_mut(key).and_then(|value| value.as_object_mut()) {
            model_config.remove("model");
        }
    }
    Ok(value)
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
        crate::subagents::SUBAGENT_TOOL_NAMES
            .iter()
            .map(|name| name.to_string()),
    );
    known.extend(
        agents
            .iter()
            .flat_map(|(_, config)| config.tool_config.available_tool_names.iter().cloned()),
    );
    // Claude Code's built-in tools, which the default freshness mapping
    // names for agents that run in Claude Code.
    known.extend(
        arko_agents::claude_code::BUILT_IN_TOOLS
            .iter()
            .map(|tool| tool.to_string()),
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

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod recovery_tests;

#[cfg(test)]
#[path = "budget_tests.rs"]
mod budget_tests;

#[cfg(test)]
#[path = "plan_budget_tests.rs"]
mod plan_budget_tests;

#[cfg(test)]
#[path = "stop_tests.rs"]
mod stop_tests;

#[cfg(test)]
#[path = "run_stop_tests.rs"]
mod run_stop_tests;
