use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::skill::{SkillManager, SkillPromptContext};
use arko_agents::{
    AgentConfig, AgentRuntime, InterceptedResult, InterceptorConfig, Task, ToolCallContext,
    ToolError, ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware,
    ToolExecutionOutcome, ToolRegistry, ToolResultInterceptor,
    runtime::{command::UserCommand, event::AgentEvent},
    task::{CANCELLED_BY_USER, TaskState},
    tool::TaskContext,
};
use arko_ai::LLMClient;

use crate::context::SharedContext;
use crate::distillation::DistillationInterceptor;
use crate::error::WorkflowError;
use crate::freshness::ActorId;
use crate::hooks::context::HookContext;
use crate::hooks::{HookPoint, HookResult};
use crate::memory::tools::*;
use crate::messaging::tools::*;
use crate::persistence::PersistenceManager;
use crate::services::{RUN_WORKFLOW_AGENT_TOOL_NAME, RunWorkflowAgentTool};
use crate::verification::{
    CompletionVerifierConfig, VerificationDecision, VerificationKind, VerificationRecord,
};
use crate::voting::{CAST_VOTE_TOOL_NAME, CastVoteTool};
use crate::workflow::cancel::CancelCause;
use crate::workflow::result::{AgentResult, AgentRunStatus};

use super::AgentHandle;

/// Chains multiple `ToolResultInterceptor` implementations, running them
/// in sequence. Each interceptor receives the output of the previous one.
struct ChainedInterceptor {
    interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
    failure_policy: arko_agents::InterceptionFailurePolicy,
}

impl std::fmt::Debug for ChainedInterceptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainedInterceptor")
            .field("count", &self.interceptors.len())
            .finish()
    }
}

#[async_trait]
impl ToolResultInterceptor for ChainedInterceptor {
    async fn intercept(
        &self,
        tool_name: &str,
        raw_result: &str,
        params: &HashMap<String, Value>,
        task_context: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        let mut current = raw_result.to_string();
        let mut was_modified = false;

        for interceptor in &self.interceptors {
            let attempts = match self.failure_policy {
                arko_agents::InterceptionFailurePolicy::UseRawResult => 1,
                arko_agents::InterceptionFailurePolicy::RetryThenUseRawResult { max_attempts } => {
                    max_attempts.max(1)
                }
            };
            for attempt in 1..=attempts {
                match interceptor
                    .intercept(tool_name, &current, params, task_context)
                    .await
                {
                    Ok(InterceptedResult::Unchanged) => break,
                    Ok(InterceptedResult::Replaced(new_value)) => {
                        current = new_value;
                        was_modified = true;
                        break;
                    }
                    Err(error) if attempt < attempts => {
                        log::warn!(
                            "Tool result interceptor failed on attempt {attempt}/{attempts}: {error}"
                        );
                    }
                    Err(error) => {
                        log::warn!(
                            "Tool result interceptor failed; preserving earlier transformations: {error}"
                        );
                        break;
                    }
                }
            }
        }

        if was_modified {
            Ok(InterceptedResult::Replaced(current))
        } else {
            Ok(InterceptedResult::Unchanged)
        }
    }
}

/// Fire lifecycle hooks using only the shared context.
///
/// Iterates all hooks registered at the given point. If any hook returns
/// `Abort`, the workflow run is stopped (see [`abort_run`]) and this returns
/// `WorkflowError::AbortedByHook`. `Skip` is ignored where there is no
/// operation to skip.
pub(crate) fn fire_hooks_on_context(
    ctx: &Arc<SharedContext>,
    point: HookPoint,
    configure: impl FnOnce(&mut HookContext),
) -> Result<(), WorkflowError> {
    match fire_hooks(ctx, &point, configure) {
        HookResult::Abort(reason) => Err(abort_run(ctx, &point, reason)),
        _ => Ok(()),
    }
}

/// Fires hooks at a point where their result has no effect.
pub(crate) fn fire_hooks_ignoring_result(
    ctx: &Arc<SharedContext>,
    point: HookPoint,
    configure: impl FnOnce(&mut HookContext),
) {
    let _ = fire_hooks(ctx, &point, configure);
}

/// Runs the hooks registered at `point` until one returns something other
/// than `Continue`, and returns that result.
fn fire_hooks(
    ctx: &SharedContext,
    point: &HookPoint,
    configure: impl FnOnce(&mut HookContext),
) -> HookResult {
    let Some(hooks) = ctx.hooks.get(point) else {
        return HookResult::Continue;
    };

    let mut hctx = HookContext::base(
        ctx.workflow_run_id,
        &ctx.workflow_name,
        point.clone(),
        ctx.budget.total_spent(),
        ctx.budget.limit(),
    );
    configure(&mut hctx);

    for hook in hooks {
        match hook(&hctx) {
            HookResult::Continue => continue,
            other => return other,
        }
    }

    HookResult::Continue
}

/// Stops the run because a hook at `point` returned `Abort(reason)`: every
/// running agent is told to stop and later run calls fail. Returns the run's
/// stop error: a run already stopped keeps its first cause.
pub(crate) fn abort_run(ctx: &SharedContext, point: &HookPoint, reason: String) -> WorkflowError {
    let cause = CancelCause::AbortedByHook {
        hook_point: format!("{point:?}"),
        reason,
    };
    let error = cause.to_error();
    ctx.run_control.stop(cause);
    ctx.run_control.error().unwrap_or(error)
}

/// The error of a stopped run.
fn stop_error(ctx: &SharedContext) -> WorkflowError {
    ctx.run_control.error().unwrap_or(WorkflowError::Cancelled)
}

/// Waits while the workflow is paused. Fails with the stop's error if the
/// run is stopped meanwhile.
async fn wait_while_paused(ctx: &SharedContext) -> Result<(), WorkflowError> {
    loop {
        let resumed = ctx.resume_notify.notified();
        if !ctx.paused.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }
        tokio::select! {
            biased;
            _ = ctx.run_control.stopped() => return Err(stop_error(ctx)),
            _ = resumed => {}
        }
    }
}

/// Broadcasts an agent event to workflow subscribers and appends it to the
/// run's event journal.
async fn publish_event(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    event: &AgentEvent,
    ctx: &SharedContext,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<(), WorkflowError> {
    let _ = ctx.event_tx.send(crate::workflow::result::WorkflowEvent {
        timestamp: chrono::Utc::now(),
        workflow_run_id: ctx.workflow_run_id,
        agent_name: agent_name.to_string(),
        agent_run_id,
        event: event.clone(),
    });
    if let Some(persistence) = persistence {
        persistence
            .append_event(agent_name, agent_run_id, event)
            .await?;
    }
    Ok(())
}

/// Charges reported usage to the budget and fires the budget hooks.
fn charge_usage(
    agent_name: &str,
    usage: &arko_ai::UsageInfo,
    ctx: &Arc<SharedContext>,
) -> crate::budget::AddCostResult {
    // Subscription (ChatGPT OAuth) calls are recorded with a billed amount
    // of zero.
    let cost_result = ctx.budget.add_agent_cost(agent_name, usage);
    if cost_result.warning_triggered {
        let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetWarning, |hctx| {
            hctx.budget_spent = cost_result.new_total;
        });
    }
    if cost_result.exceeded {
        let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetExceeded, |hctx| {
            hctx.budget_spent = cost_result.new_total;
        });
    }
    cost_result
}

/// Drive the agent event loop with default workflow behavior.
///
/// Handles tool approval, budget tracking, and error recovery automatically.
/// Tool hooks and freshness policy are enforced by execution middleware, so
/// auto-approved and batched calls follow the same path.
///
/// Returns when the agent ends, or with an error when the workflow must stop
/// it (including the stop's error once the run is cancelled or aborted); the
/// caller then stops the agent.
pub(crate) async fn drive_event_loop(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<(), WorkflowError> {
    let mut approved_error_recoveries = 0_u32;
    loop {
        let event = tokio::select! {
            biased;
            _ = ctx.run_control.stopped() => return Err(stop_error(ctx)),
            event = event_rx.recv() => match event {
                Some(event) => event,
                None => break,
            },
        };
        wait_while_paused(ctx).await?;
        publish_event(agent_name, agent_run_id, &event, ctx, persistence).await?;
        ctx.registry.observe(agent_run_id, &event);
        if ctx.budget.is_exceeded() && !matches!(&event, AgentEvent::UsageReported(_)) {
            let _ = cmd_tx.send(UserCommand::CancelTask).await;
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        match event {
            AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => {
                let _ = cmd_tx
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await;
            }

            AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } => {
                let mut decisions = HashMap::new();
                for call in &pending_calls {
                    decisions.insert(
                        call.tool_call_id,
                        arko_agents::runtime::command::ToolApprovalDecision::Approve {
                            custom_tool_output: None,
                            message_to_agent: None,
                        },
                    );
                }
                let _ = cmd_tx
                    .send(UserCommand::SubmitBatchToolApprovals {
                        batch_id,
                        decisions,
                    })
                    .await;
            }

            AgentEvent::UsageReported(ref usage) => {
                let cost_result = charge_usage(agent_name, usage, ctx);
                if cost_result.exceeded {
                    let _ = cmd_tx.send(UserCommand::CancelTask).await;
                    return Err(WorkflowError::BudgetExceeded {
                        spent: cost_result.new_total,
                        limit: ctx.budget.limit().unwrap_or(0.0),
                    });
                }
                if ctx.budget.is_agent_exceeded(agent_name) {
                    let _ = cmd_tx.send(UserCommand::CancelTask).await;
                    return Err(WorkflowError::AgentBudgetExceeded {
                        agent: agent_name.to_string(),
                        spent: ctx.budget.agent_spent(agent_name),
                        limit: ctx.budget.agent_limit(agent_name).unwrap_or(0.0),
                    });
                }

                if ctx.budget.is_exceeded() {
                    let _ = cmd_tx.send(UserCommand::CancelTask).await;
                    return Err(WorkflowError::BudgetExceeded {
                        spent: ctx.budget.total_spent(),
                        limit: ctx.budget.limit().unwrap_or(0.0),
                    });
                }
            }

            AgentEvent::ErrorEncountered {
                requires_approval_to_retry,
                ..
            } => {
                if requires_approval_to_retry {
                    if approved_error_recoveries == 0 {
                        approved_error_recoveries += 1;
                        let _ = cmd_tx
                            .send(UserCommand::ApproveErrorRecovery { message: None })
                            .await;
                    } else {
                        let _ = cmd_tx
                            .send(UserCommand::RejectErrorRecovery {
                                message: Some(
                                    "Workflow-managed error recovery limit reached".to_string(),
                                ),
                            })
                            .await;
                    }
                }
            }

            AgentEvent::UserQuestionsAsked { questions } => {
                if let Some(handler) = &ctx.user_input_handler {
                    let request = crate::user_input::UserInputRequest {
                        workflow_run_id: ctx.workflow_run_id,
                        agent_run_id,
                        agent_name: agent_name.to_string(),
                        questions: questions.clone(),
                    };
                    // The handler may wait on a person; a stop doesn't.
                    let response = tokio::select! {
                        biased;
                        _ = ctx.run_control.stopped() => return Err(stop_error(ctx)),
                        response = handler.respond(request) => response,
                    }
                    .map_err(WorkflowError::Channel)?;
                    cmd_tx
                        .send(UserCommand::SubmitUserResponse { response })
                        .await
                        .map_err(|error| WorkflowError::Channel(error.to_string()))?;
                } else {
                    let _ = cmd_tx.send(UserCommand::SuspendTask).await;
                    return Err(WorkflowError::UserInputRequired {
                        agent: agent_name.to_string(),
                        task_id: agent_run_id,
                        questions,
                    });
                }
            }

            AgentEvent::ToolResultIntercepted {
                tool_call_id,
                ref tool_name,
                ref original_output,
                ref intercepted_output,
            } => {
                fire_hooks_on_context(ctx, HookPoint::PostToolResultInterception, |hook| {
                    hook.agent_name = Some(agent_name.to_string());
                    hook.agent_run_id = Some(agent_run_id);
                    hook.tool_call_id = Some(tool_call_id);
                    hook.tool_name = Some(tool_name.clone());
                    hook.original_tool_result = Some(original_output.clone());
                    hook.effective_tool_result = Some(intercepted_output.clone());
                    hook.tool_result = Some(intercepted_output.clone());
                })?;
            }

            AgentEvent::TaskStateChanged(TaskState::Interrupted { reason }) => {
                // The agent runtime can wait for commands after a persistent
                // transport failure. A workflow invocation must return its error
                // instead of waiting forever for TaskStreamEnd.
                let _ = cmd_tx.send(UserCommand::CancelTask).await;
                return Err(WorkflowError::AgentInterrupted {
                    agent: agent_name.to_string(),
                    reason,
                });
            }

            AgentEvent::TaskStateChanged(ref state) if state.is_terminal() => {
                break;
            }

            AgentEvent::TaskStreamEnd => break,

            _ => {}
        }
    }

    Ok(())
}

/// How a run starts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StartOptions {
    /// The run's task ID, already marked running in the agent registry.
    pub reserved: Option<uuid::Uuid>,
    /// Whether the initial prompt lists the agents it may message.
    pub roster: bool,
    /// See [`StartedRun::depth`].
    pub depth: u32,
}

impl Default for StartOptions {
    fn default() -> Self {
        Self {
            reserved: None,
            roster: true,
            depth: 0,
        }
    }
}

/// The heading of the initial prompt's list of the agents it may message.
const ROSTER_HEADING: &str = "\n\n# Agents you can message\n\nMessage these agents with send_message while they are running or waiting; call list_recipients for their current status. You are told when one starts.\n";

/// `prompt` without the list of agents it may message.
pub(crate) fn without_roster(prompt: &str) -> &str {
    prompt
        .rfind(ROSTER_HEADING)
        .map_or(prompt, |start| &prompt[..start])
}

/// An agent run the engine started, to be driven to its result.
pub(crate) struct StartedRun {
    pub agent_name: String,
    pub agent_run_id: uuid::Uuid,
    pub event_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    pub cmd_tx: tokio::sync::mpsc::Sender<UserCommand>,
    pub join_handle: AbortOnDrop,
    /// The task as the run started it: the record of an agent that had to be
    /// aborted.
    pub started_task: Task,
    /// Messages delivered in the prompt, acknowledged once the agent
    /// completes.
    pub pending_message_ids: Vec<uuid::Uuid>,
    /// How many agent runs this one runs inside (a verifier or a delegated
    /// agent runs inside the tool call of the agent that started it).
    pub depth: u32,
}

/// What an agent runtime's task returned.
type Joined = Result<Result<Task, arko_agents::ArkoAgentsError>, tokio::task::JoinError>;

/// An agent runtime's task, aborted if its handle is dropped first: an
/// agent whose driver is dropped (for example the driver of a delegated
/// agent inside an agent that was itself aborted) doesn't keep running
/// unsupervised.
pub(crate) struct AbortOnDrop(
    pub Option<tokio::task::JoinHandle<Result<Task, arko_agents::ArkoAgentsError>>>,
);

impl AbortOnDrop {
    /// Hands the task over without aborting it on drop.
    pub fn into_inner(
        mut self,
    ) -> tokio::task::JoinHandle<Result<Task, arko_agents::ArkoAgentsError>> {
        self.0.take().expect("the task is handed over once")
    }

    fn handle(
        &mut self,
    ) -> &mut tokio::task::JoinHandle<Result<Task, arko_agents::ArkoAgentsError>> {
        self.0.as_mut().expect("the task is handed over once")
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
}

/// When a stopped agent at `depth` is aborted. Every driver of a stopped run
/// counts from the moment the run stopped; one nested inside another agent's
/// tool call gets half its parent's time, so it winds down, and is recorded,
/// before its parent can be aborted.
fn stop_deadline(ctx: &SharedContext, depth: u32) -> tokio::time::Instant {
    let grace = ctx.run_control.grace_period() / 2_u32.checked_pow(depth).unwrap_or(u32::MAX);
    let start = ctx
        .run_control
        .stopped_at()
        .unwrap_or_else(tokio::time::Instant::now);
    // A grace period too long to represent waits as long as possible.
    start
        .checked_add(grace)
        .unwrap_or_else(|| start + std::time::Duration::from_secs(30 * 365 * 86_400))
}

/// Waits for an agent that is being stopped to end: sends it `CancelTask`,
/// records its last events and usage meanwhile (its generic cancellation
/// reason replaced by `reason`), and aborts it at `deadline`. `None` means
/// the agent was aborted.
#[allow(clippy::too_many_arguments)]
async fn wind_down(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    join_handle: &mut AbortOnDrop,
    deadline: tokio::time::Instant,
    reason: &str,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Option<Joined> {
    let mut events_open = true;
    // Sent while events are drained: an agent blocked on a full event
    // channel must be able to take the command.
    let cancel = cmd_tx.send(UserCommand::CancelTask);
    tokio::pin!(cancel);
    let mut cancel_sent = false;
    loop {
        tokio::select! {
            biased;
            event = event_rx.recv(), if events_open => match event {
                Some(event) => {
                    let event = match event {
                        AgentEvent::TaskStateChanged(TaskState::Cancelled { reason: generic })
                            if generic == CANCELLED_BY_USER =>
                        {
                            AgentEvent::TaskStateChanged(TaskState::Cancelled {
                                reason: reason.to_string(),
                            })
                        }
                        event => event,
                    };
                    if let Err(error) =
                        publish_event(agent_name, agent_run_id, &event, ctx, persistence).await
                    {
                        log::warn!("Could not record an event of stopping agent '{agent_name}': {error}");
                    }
                    ctx.registry.observe(agent_run_id, &event);
                    if let AgentEvent::UsageReported(usage) = &event {
                        charge_usage(agent_name, usage, ctx);
                    }
                }
                None => events_open = false,
            },
            _ = &mut cancel, if !cancel_sent => cancel_sent = true,
            joined = &mut *join_handle.handle() => return Some(joined),
            _ = tokio::time::sleep_until(deadline) => {
                join_handle.handle().abort();
                return match join_handle.handle().await {
                    Err(error) if error.is_cancelled() => None,
                    joined => Some(joined),
                };
            }
        }
    }
}

/// Drives a started agent run to its result: runs the event loop, stops the
/// agent when the workflow must (aborting it after the run's grace period),
/// records the final task, and maps it to the call's outcome. Once the run
/// is stopped the outcome is the stop's error; the agent's result is
/// recorded either way.
pub(crate) async fn drive_to_result(
    run: StartedRun,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<AgentResult, WorkflowError> {
    let StartedRun {
        agent_name,
        agent_run_id,
        mut event_rx,
        cmd_tx,
        mut join_handle,
        started_task,
        pending_message_ids,
        depth,
    } = run;
    // Until the end is recorded below: if this future is dropped (and with
    // it, the agent's task aborted), the run is recorded as abandoned.
    let mut abandoned = Abandoned {
        ctx,
        agent_name: agent_name.clone(),
        agent_run_id,
        armed: true,
    };
    let drive_result = drive_event_loop(
        &agent_name,
        agent_run_id,
        &mut event_rx,
        &cmd_tx,
        ctx,
        persistence,
    )
    .await;

    let stopping = drive_result
        .as_ref()
        .is_err_and(|error| !matches!(error, WorkflowError::UserInputRequired { .. }));
    let wound_down = if stopping {
        let deadline = stop_deadline(ctx, depth);
        wind_down(
            &agent_name,
            agent_run_id,
            &mut event_rx,
            &cmd_tx,
            &mut join_handle,
            deadline,
            &stop_reason(ctx, &drive_result),
            ctx,
            persistence,
        )
        .await
    } else {
        Some(join_handle.handle().await)
    };

    let final_task = match wound_down {
        Some(Ok(Ok(task))) => Ok(task),
        Some(Ok(Err(error))) => Err(WorkflowError::Agent(error)),
        Some(Err(error)) => Err(WorkflowError::Channel(format!("Agent join error: {error}"))),
        None => {
            let mut task = started_task;
            task.set_state(TaskState::Cancelled {
                reason: format!(
                    "{} The agent did not stop within {:?} and was aborted.",
                    stop_reason(ctx, &drive_result),
                    ctx.run_control.grace_period()
                ),
            });
            Ok(task)
        }
    };

    abandoned.armed = false;
    ctx.registry.finish(
        agent_run_id,
        final_task.as_ref().map_or(AgentRunStatus::Failed, |task| {
            run_status(&task.current_state)
        }),
    );
    let result = match final_task {
        Ok(mut task) => {
            // The runtime records its generic reason for a CancelTask; the
            // workflow knows why it sent it.
            if stopping
                && matches!(&task.current_state, TaskState::Cancelled { reason } if reason == CANCELLED_BY_USER)
            {
                task.set_state(TaskState::Cancelled {
                    reason: stop_reason(ctx, &drive_result),
                });
            }
            let completed = matches!(task.current_state, TaskState::Completed { .. });
            let finalized = finalize_agent_task(agent_name.clone(), task, ctx, persistence).await;
            if completed {
                ctx.messages.acknowledge_active(&agent_name, agent_run_id);
                ctx.messages.acknowledge(&agent_name, &pending_message_ids);
            }
            finalized
        }
        Err(error) => Err(error),
    };
    ctx.messages.unregister_active(&agent_name, agent_run_id);
    ctx.run_control.unregister(agent_run_id);

    if let Some(stop) = ctx.run_control.error() {
        if let Err(error) = &result {
            log::warn!("Agent '{agent_name}' stopped with the workflow run: {error}");
        }
        return Err(stop);
    }
    let result = result?;
    drive_result?;
    match result.status {
        AgentRunStatus::Completed => Ok(result),
        AgentRunStatus::WaitingForInput => Err(WorkflowError::UserInputRequired {
            agent: agent_name,
            task_id: result.agent_run_id,
            questions: result.error.clone().unwrap_or_default(),
        }),
        AgentRunStatus::Interrupted => Err(WorkflowError::AgentInterrupted {
            agent: agent_name,
            reason: result.error.clone().unwrap_or_default(),
        }),
        AgentRunStatus::Cancelled => Err(WorkflowError::AgentCancelled {
            agent: agent_name,
            reason: result.error.clone().unwrap_or_default(),
        }),
        AgentRunStatus::Failed => Err(WorkflowError::Agent(
            arko_agents::ArkoAgentsError::TaskExecution(
                result
                    .error
                    .clone()
                    .unwrap_or_else(|| "Agent failed".to_string()),
            ),
        )),
    }
}

/// Unregisters a run dropped before it was driven (its runtime task is
/// aborted with it): its agent is finished, cancelled, unless another run
/// of it is running.
pub(crate) fn abandon_run(run: &StartedRun, ctx: &SharedContext) {
    abandon(ctx, &run.agent_name, run.agent_run_id);
}

fn abandon(ctx: &SharedContext, agent_name: &str, agent_run_id: uuid::Uuid) {
    ctx.registry.finish(agent_run_id, AgentRunStatus::Cancelled);
    ctx.messages.unregister_active(agent_name, agent_run_id);
    ctx.run_control.unregister(agent_run_id);
}

/// Records a run whose driver was dropped before it recorded the run's end.
struct Abandoned<'a> {
    ctx: &'a SharedContext,
    agent_name: String,
    agent_run_id: uuid::Uuid,
    armed: bool,
}

impl Drop for Abandoned<'_> {
    fn drop(&mut self) {
        if self.armed {
            abandon(self.ctx, &self.agent_name, self.agent_run_id);
        }
    }
}

/// The run status a final task state records.
fn run_status(state: &TaskState) -> AgentRunStatus {
    match state {
        TaskState::Completed { .. } => AgentRunStatus::Completed,
        TaskState::Cancelled { .. } => AgentRunStatus::Cancelled,
        TaskState::Failed { .. } => AgentRunStatus::Failed,
        TaskState::AwaitingUserResponse { .. } => AgentRunStatus::WaitingForInput,
        _ => AgentRunStatus::Interrupted,
    }
}

/// Why the workflow stopped an agent: the run's stop, or the error the
/// driver stopped it for (an exceeded budget, an interruption, a hook).
fn stop_reason(ctx: &SharedContext, drive_result: &Result<(), WorkflowError>) -> String {
    match (ctx.run_control.cancellation(), drive_result) {
        (Some(cancellation), _) => cancellation.cause.agent_reason(),
        (None, Err(error)) => format!("Stopped by the workflow: {error}"),
        (None, Ok(())) => "Stopped by the workflow.".to_string(),
    }
}

/// With auto-checkpointing, saves the checkpoint recording the run's stop,
/// once per run.
pub(crate) async fn save_stop_checkpoint(
    ctx: &SharedContext,
    persistence: Option<&Arc<PersistenceManager>>,
) {
    if let Some(persistence) = persistence
        && ctx.run_control.claim_stop_save()
        && let Err(error) = persistence.auto_checkpoint(ctx).await
    {
        log::warn!("Could not save the checkpoint recording the workflow's stop: {error}");
    }
}

pub(crate) async fn finalize_agent_task(
    agent_name: String,
    task: Task,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<AgentResult, WorkflowError> {
    let agent_run_id = task.id;
    if matches!(task.current_state, TaskState::Completed { .. }) {
        // Channel acceptance is not consumption: complete_task may finish with
        // unread InjectContext commands. Confirm only finalized user context.
        ctx.messages
            .confirm_active_context(&agent_name, agent_run_id, &task.conversation);
    }
    let (status, summary, error, hook_point) = match &task.current_state {
        TaskState::Completed { summary } => (
            crate::workflow::result::AgentRunStatus::Completed,
            summary.clone(),
            None,
            HookPoint::PostAgentCompletion,
        ),
        TaskState::Cancelled { reason } => (
            crate::workflow::result::AgentRunStatus::Cancelled,
            String::new(),
            Some(reason.clone()),
            HookPoint::PostAgentCancellation,
        ),
        TaskState::Failed { reason } => (
            crate::workflow::result::AgentRunStatus::Failed,
            String::new(),
            Some(reason.clone()),
            HookPoint::PostAgentFailure,
        ),
        TaskState::Interrupted { reason } => (
            crate::workflow::result::AgentRunStatus::Interrupted,
            String::new(),
            Some(reason.clone()),
            HookPoint::PostAgentInterruption,
        ),
        TaskState::AwaitingUserResponse { questions } => (
            crate::workflow::result::AgentRunStatus::WaitingForInput,
            String::new(),
            Some(questions.clone()),
            HookPoint::OnUserInputRequired,
        ),
        state => (
            crate::workflow::result::AgentRunStatus::Interrupted,
            String::new(),
            Some(format!("Agent ended in non-terminal state: {state:?}")),
            HookPoint::PostAgentInterruption,
        ),
    };

    let result = AgentResult {
        agent_run_id,
        agent_name: agent_name.clone(),
        status,
        summary: summary.clone(),
        error: error.clone(),
        task,
    };
    {
        let mut results = ctx.agent_results.write();
        if let Some(existing) = results
            .iter_mut()
            .find(|existing| existing.agent_run_id == agent_run_id)
        {
            *existing = result.clone();
        } else {
            results.push(result.clone());
        }
    }

    let hook_result = fire_hooks_on_context(ctx, hook_point, |hook| {
        hook.agent_name = Some(agent_name);
        hook.agent_run_id = Some(agent_run_id);
        hook.agent_summary = (!summary.is_empty()).then_some(summary);
        hook.error_message = error;
    });

    if let Some(persistence) = persistence {
        persistence.save_agent_task(&result.task).await?;
        persistence.auto_checkpoint(ctx).await?;
    }

    hook_result?;

    Ok(result)
}

pub(crate) struct WorkflowEngine {
    llm_client: Arc<LLMClient>,
    persistence: Option<Arc<PersistenceManager>>,
}

impl WorkflowEngine {
    fn setup_skills(
        config: &mut AgentConfig,
        registry: &mut ToolRegistry,
        workspace: &std::path::Path,
    ) -> Result<Option<SkillPromptContext>, WorkflowError> {
        let Some(source) = config.skills_config.as_ref() else {
            return Ok(None);
        };
        let manager = SkillManager::load(source, workspace)?;
        Ok(Some(manager.setup(registry, config)))
    }

    pub fn new(llm_client: Arc<LLMClient>, persistence: Option<PersistenceManager>) -> Self {
        Self {
            llm_client,
            persistence: persistence.map(Arc::new),
        }
    }

    pub(crate) fn from_shared(
        llm_client: Arc<LLMClient>,
        persistence: Option<Arc<PersistenceManager>>,
    ) -> Self {
        Self {
            llm_client,
            persistence,
        }
    }

    /// The workflow tools of `agent_name`'s run at `depth` (see
    /// [`StartedRun::depth`]).
    pub(crate) fn build_tool_registry(
        &self,
        agent_name: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
    ) -> ToolRegistry {
        let mut registry = ToolRegistry::new();

        if !ctx.memory.is_empty() {
            registry.register(Arc::new(ReadMemoryTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
            registry.register(Arc::new(WriteMemoryTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
            registry.register(Arc::new(ListMemoriesTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
        }

        if !ctx.message_recipients(agent_name).is_empty() {
            registry.register(Arc::new(SendMessageTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
            registry.register(Arc::new(ListRecipientsTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
        }
        if let Some(services) = ctx.workflow_services.get()
            && services.is_allowed(agent_name)
        {
            registry.register(Arc::new(RunWorkflowAgentTool::new(
                services.clone(),
                agent_name.to_string(),
                depth,
            )));
        }
        if let Some(ballots) = &ctx.ballots {
            registry.register(Arc::new(CastVoteTool::new(
                ballots.clone(),
                agent_name.to_string(),
            )));
        }

        for tool in &ctx.global_tools {
            registry.register(tool.clone());
        }
        if let Some(agent_tools) = ctx.agent_tools.get(agent_name) {
            for tool in agent_tools {
                registry.register(tool.clone());
            }
        }

        registry
    }

    /// Composes global and agent-specific interceptors into a single
    /// `InterceptorConfig`, or `None` if no interceptors are configured
    /// for this agent.
    fn build_interceptor_config(
        &self,
        agent_name: &str,
        ctx: &Arc<SharedContext>,
    ) -> Option<InterceptorConfig> {
        let mut all: Vec<Arc<dyn ToolResultInterceptor>> = Vec::new();
        all.extend(ctx.global_interceptors.iter().cloned());
        if let Some(agent_specific) = ctx.agent_interceptors.get(agent_name) {
            all.extend(agent_specific.iter().cloned());
        }
        for config in &ctx.distillation_configs {
            if !config.targets_agent(agent_name) {
                continue;
            }
            all.push(Arc::new(DistillationInterceptor {
                agent_name: agent_name.to_string(),
                config: config.clone(),
                client: self.llm_client.clone(),
                context: ctx.clone(),
            }));
        }

        if all.is_empty() {
            return None;
        }

        let interceptor: Arc<dyn ToolResultInterceptor> = if all.len() == 1 {
            all.remove(0)
        } else {
            Arc::new(ChainedInterceptor {
                interceptors: all,
                failure_policy: ctx.interception_failure_policy.clone(),
            })
        };

        Some(InterceptorConfig {
            interceptor,
            failure_policy: ctx.interception_failure_policy.clone(),
        })
    }

    /// The middleware every agent's tool calls pass through: the run's
    /// cancellation, tool hooks, context freshness, and completion
    /// verification.
    pub(crate) fn build_execution_middleware(
        &self,
        agent_name: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
    ) -> Arc<dyn ToolExecutionMiddleware> {
        Arc::new(WorkflowExecutionMiddleware {
            agent_name: agent_name.to_string(),
            context: ctx.clone(),
            llm_client: self.llm_client.clone(),
            persistence: self.persistence.clone(),
            depth,
        })
    }

    pub(crate) fn prepare_agent_config(
        &self,
        agent_name: &str,
        base_config: &AgentConfig,
        ctx: &Arc<SharedContext>,
    ) -> AgentConfig {
        let mut config = base_config.clone();

        if !ctx.memory.is_empty() {
            ensure_tool_name(&mut config, READ_MEMORY_TOOL_NAME);
            ensure_tool_name(&mut config, WRITE_MEMORY_TOOL_NAME);
            ensure_tool_name(&mut config, LIST_MEMORIES_TOOL_NAME);
        }
        if !ctx.message_recipients(agent_name).is_empty() {
            ensure_tool_name(&mut config, SEND_MESSAGE_TOOL_NAME);
            ensure_tool_name(&mut config, LIST_RECIPIENTS_TOOL_NAME);
        }
        if let Some(services) = ctx.workflow_services.get()
            && services.is_allowed(agent_name)
        {
            ensure_tool_name(&mut config, RUN_WORKFLOW_AGENT_TOOL_NAME);
        }
        if ctx.ballots.is_some() {
            ensure_tool_name(&mut config, CAST_VOTE_TOOL_NAME);
        }

        for tool in &ctx.global_tools {
            ensure_tool_name(&mut config, &tool.name());
        }
        if let Some(agent_tools) = ctx.agent_tools.get(agent_name) {
            for tool in agent_tools {
                ensure_tool_name(&mut config, &tool.name());
            }
        }

        config
    }

    /// The prompt section delivering the agent's pending messages, and
    /// their IDs, acknowledged once the run completes.
    fn pending_messages(
        &self,
        agent_name: &str,
        ctx: &Arc<SharedContext>,
    ) -> (String, Vec<uuid::Uuid>) {
        let messages = ctx.messages.pending(agent_name);
        let message_ids = messages.iter().map(|message| message.id).collect();
        let mut section = String::new();
        if !messages.is_empty() {
            section.push_str("\n\n# Messages from other agents:\n");
            for msg in messages {
                section.push_str(&format!(
                    "\n--- Message {} from '{}' ---\n{}{}\n--- End message ---\n",
                    msg.id,
                    msg.from,
                    msg.content,
                    if msg.attachments.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "\nContext references: {}",
                            serde_json::to_string(&msg.attachments).unwrap_or_default()
                        )
                    }
                ));
            }
        }
        (section, message_ids)
    }

    /// Checks that the run may start an agent and fires `PreAgentSpawn`.
    fn admit(
        &self,
        agent_name: &str,
        agent_run_id: Option<uuid::Uuid>,
        ctx: &Arc<SharedContext>,
    ) -> Result<(), WorkflowError> {
        ctx.run_control.check()?;
        if ctx.budget.is_exceeded() {
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        fire_hooks_on_context(ctx, HookPoint::PreAgentSpawn, |hook| {
            hook.agent_name = Some(agent_name.to_string());
            hook.agent_run_id = agent_run_id;
        })
    }

    /// Prepares a new run of a registered agent and starts it. A run
    /// `reserved` is already marked running in the agent registry (a batch
    /// started together) and is abandoned if it can't start.
    fn start_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
        options: StartOptions,
    ) -> Result<StartedRun, WorkflowError> {
        let prepared = self.admit(agent_name, None, ctx).and_then(|()| {
            let base_config = ctx
                .agent_configs
                .get(agent_name)
                .ok_or_else(|| WorkflowError::AgentNotFound(agent_name.to_string()))?;
            let mut config = self.prepare_agent_config(agent_name, base_config, ctx);
            let mut registry = self.build_tool_registry(agent_name, ctx, options.depth);
            let skill_prompt_context =
                Self::setup_skills(&mut config, &mut registry, &ctx.workspace)?;
            Ok((config, registry, skill_prompt_context))
        });
        let (config, registry, skill_prompt_context) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                if let Some(task_id) = options.reserved {
                    ctx.registry.abandon(task_id);
                }
                return Err(error);
            }
        };

        let mut task = Task::new(config, String::new(), ctx.workspace.clone(), None);
        match options.reserved {
            Some(task_id) => task.id = task_id,
            None => ctx.registry.start(&[(task.id, agent_name)]),
        }
        let (messages, pending_message_ids) = self.pending_messages(agent_name, ctx);
        let mut initial_prompt = format!("{prompt}{messages}");
        let recipients = ctx.message_recipients(agent_name);
        if !recipients.is_empty() {
            // The roster's statuses and the version the run has seen come
            // from one registry snapshot.
            let roster = ctx.registry.roster(
                task.id,
                &recipients,
                ctx.paused.load(std::sync::atomic::Ordering::Acquire),
            );
            if options.roster {
                initial_prompt.push_str(ROSTER_HEADING);
                initial_prompt.push_str(&render_recipients(ctx, &roster));
            }
        }
        task.initial_task_prompt = initial_prompt;
        Ok(self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            pending_message_ids,
            ctx,
            options.depth,
        ))
    }

    /// Starts the agent runtime on `task` and registers the running agent
    /// with the message bus and the run's cancellation.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        agent_name: &str,
        task: Task,
        registry: ToolRegistry,
        skill_prompt_context: Option<SkillPromptContext>,
        pending_message_ids: Vec<uuid::Uuid>,
        ctx: &Arc<SharedContext>,
        depth: u32,
    ) -> StartedRun {
        let interceptor_config = self.build_interceptor_config(agent_name, ctx);
        let execution_middleware = self.build_execution_middleware(agent_name, ctx, depth);
        let agent_run_id = task.id;
        let started_task = task.clone();
        let runtime =
            AgentRuntime::new(self.llm_client.clone()).with_turn_context(Arc::new(StartNotices {
                context: ctx.clone(),
                agent_name: agent_name.to_string(),
            }));
        let (event_rx, cmd_tx, join_handle) = runtime.run_task(
            task,
            Arc::new(registry),
            None,
            false,
            skill_prompt_context,
            interceptor_config,
            Some(execution_middleware),
        );
        ctx.messages
            .register_active(agent_name, agent_run_id, cmd_tx.clone());
        ctx.run_control.register(agent_run_id, &cmd_tx);
        StartedRun {
            agent_name: agent_name.to_string(),
            agent_run_id,
            event_rx,
            cmd_tx,
            join_handle: AbortOnDrop(Some(join_handle)),
            started_task,
            pending_message_ids,
            depth,
        }
    }

    /// Run a single agent to completion, blocking until it finishes.
    pub async fn run_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        let run = self.start_agent(agent_name, prompt, ctx, StartOptions::default())?;
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// Run an agent inside the tool call of a run at `depth - 1`.
    pub(crate) async fn run_nested(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
    ) -> Result<AgentResult, WorkflowError> {
        let options = StartOptions {
            depth,
            ..StartOptions::default()
        };
        let run = self.start_agent(agent_name, prompt, ctx, options)?;
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// Run a completion verifier at `depth`: its prompt gets no roster of
    /// the agents it may message.
    async fn run_verifier(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
    ) -> Result<AgentResult, WorkflowError> {
        let options = StartOptions {
            roster: false,
            depth,
            ..StartOptions::default()
        };
        let run = self.start_agent(agent_name, prompt, ctx, options)?;
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// Run multiple agents in parallel.
    ///
    /// The agents are marked running together before any of them is
    /// prompted, so each one's roster shows the others as running.
    pub async fn run_agents_parallel(
        &self,
        agents: Vec<(&str, String)>,
        ctx: &Arc<SharedContext>,
    ) -> Result<Vec<AgentResult>, WorkflowError> {
        ctx.run_control.check()?;
        let reserved = agents
            .iter()
            .map(|(name, _)| (uuid::Uuid::new_v4(), *name))
            .collect::<Vec<_>>();
        ctx.registry.start(&reserved);
        let mut join_handles = Vec::new();

        for ((task_id, name), (_, prompt)) in reserved.into_iter().zip(agents) {
            let engine_client = self.llm_client.clone();
            let persistence = self.persistence.clone();
            let ctx_clone = ctx.clone();
            let name_owned = name.to_string();

            let handle = tokio::spawn({
                let engine = WorkflowEngine {
                    llm_client: engine_client,
                    persistence,
                };
                async move {
                    let options = StartOptions {
                        reserved: Some(task_id),
                        ..StartOptions::default()
                    };
                    let run = engine.start_agent(&name_owned, &prompt, &ctx_clone, options)?;
                    drive_to_result(run, &ctx_clone, engine.persistence.as_ref()).await
                }
            });
            join_handles.push(handle);
        }

        let mut results = Vec::new();
        let mut first_error = None;
        while !join_handles.is_empty() {
            let handle = join_handles.remove(0);
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(WorkflowError::Channel(format!(
                            "Parallel agent join error: {error}"
                        )));
                    }
                }
            }
        }
        ctx.run_control.check()?;
        match first_error {
            Some(error) => Err(error),
            None => Ok(results),
        }
    }

    /// Spawn an agent without blocking, returning a handle for driving it.
    ///
    /// Performs all agent setup (hook firing, config preparation, tool registry
    /// construction, message augmentation, task creation) then starts the agent
    /// runtime and returns the raw channels wrapped in an [`AgentHandle`].
    ///
    /// The caller drives execution either by calling [`AgentHandle::wait`]
    /// (default workflow behavior) or [`AgentHandle::into_parts`] (custom
    /// event processing).
    pub async fn spawn_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentHandle, WorkflowError> {
        let run = self.start_agent(agent_name, prompt, ctx, StartOptions::default())?;
        Ok(AgentHandle::new(run, ctx.clone(), self.persistence.clone()))
    }

    pub async fn continue_agent(
        &self,
        previous: &AgentResult,
        continuation_prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        self.continue_at(previous, continuation_prompt, ctx, 0)
            .await
    }

    /// Continue an agent run at `depth` (see [`StartedRun::depth`]).
    async fn continue_at(
        &self,
        previous: &AgentResult,
        continuation_prompt: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
    ) -> Result<AgentResult, WorkflowError> {
        let resumable = match previous.status {
            AgentRunStatus::Completed => {
                matches!(previous.task.current_state, TaskState::Completed { .. })
            }
            AgentRunStatus::WaitingForInput => matches!(
                previous.task.current_state,
                TaskState::AwaitingUserResponse { .. }
            ),
            _ => false,
        };
        if !resumable {
            return Err(WorkflowError::BuilderInvalid(
                "only a completed or user-input-suspended agent run can be continued".to_string(),
            ));
        }
        let agent_name = &previous.agent_name;
        self.admit(agent_name, Some(previous.agent_run_id), ctx)?;
        let mut task = previous.task.clone();
        let mut registry = self.build_tool_registry(agent_name, ctx, depth);
        let skill_prompt_context =
            Self::setup_skills(&mut task.agent_config, &mut registry, &ctx.workspace)?;
        // No roster: the task's initial prompt has one. Start notices tell it
        // about recipients that started since it last ran.
        let (messages, pending_message_ids) = self.pending_messages(agent_name, ctx);
        let prompt = format!("{continuation_prompt}{messages}");
        match previous.status {
            AgentRunStatus::Completed => {
                task.prepare_for_resumption_from_completed(prompt)?;
            }
            _ => {
                task.prepare_for_resumption_with_user_response(prompt)?;
            }
        }
        ctx.registry.start(&[(task.id, agent_name.as_str())]);
        let run = self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            pending_message_ids,
            ctx,
            depth,
        );
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// With auto-checkpointing, saves the checkpoint recording the run's
    /// stop, once per run.
    pub(crate) async fn save_stop_checkpoint(&self, ctx: &Arc<SharedContext>) {
        save_stop_checkpoint(ctx, self.persistence.as_ref()).await;
    }

    /// Save a checkpoint of the current workflow state.
    pub async fn checkpoint(&self, ctx: &Arc<SharedContext>) -> Result<(), WorkflowError> {
        if let Some(ref persistence) = self.persistence {
            persistence.save_checkpoint(ctx).await?;
        }
        Ok(())
    }
}

fn ensure_tool_name(config: &mut AgentConfig, tool_name: &str) {
    if !config
        .tool_config
        .available_tool_names
        .iter()
        .any(|n| n == tool_name)
    {
        config
            .tool_config
            .available_tool_names
            .push(tool_name.to_string());
    }
}

/// Enforces workflow policy at the actual tool execution boundary, independent
/// of whether a call was auto-approved, manually approved, or part of a batch.
struct WorkflowExecutionMiddleware {
    /// Stable key used to register this agent in the workflow. AgentConfig::name
    /// is a human-readable display name and must not be used for policy matching.
    agent_name: String,
    context: Arc<SharedContext>,
    llm_client: Arc<LLMClient>,
    persistence: Option<Arc<PersistenceManager>>,
    /// The depth of this agent's run; its verifiers run one deeper.
    depth: u32,
}

impl WorkflowExecutionMiddleware {
    /// Once the run is stopped, every tool call cancels its agent.
    fn stop_decision(&self) -> Option<ToolExecutionDecision> {
        self.context.run_control.cancellation().map(|cancellation| {
            ToolExecutionDecision::CancelTask {
                reason: cancellation.cause.agent_reason(),
            }
        })
    }

    fn applicable_completion_verifiers(&self) -> impl Iterator<Item = &CompletionVerifierConfig> {
        self.context
            .completion_verifiers
            .iter()
            .filter(|config| config.applies_to(&self.agent_name))
    }

    async fn verify_completion(
        &self,
        execution: &ToolExecutionContext<'_>,
        summary: &str,
    ) -> Option<ToolExecutionDecision> {
        const MAX_FORMAT_RETRIES: usize = 1;

        for verifier in self.applicable_completion_verifiers() {
            if verifier.kind == VerificationKind::Efficiency {
                let repair_count = self
                    .context
                    .verification_attempts
                    .lock()
                    .get(&(execution.task_id, verifier.name.clone()))
                    .map_or(0, |counts| counts.0);
                if verifier.efficiency_repair_budget_exhausted(repair_count) {
                    continue;
                }
            }
            let engine = WorkflowEngine {
                llm_client: self.llm_client.clone(),
                persistence: self.persistence.clone(),
            };
            let session_key = (execution.task_id, verifier.name.clone());
            let prior_record = verifier
                .reuse_conversation
                .then(|| {
                    self.context
                        .verification_records
                        .read()
                        .iter()
                        .rev()
                        .find(|record| {
                            record.target_task_id == execution.task_id
                                && record.verifier_name == verifier.name
                        })
                        .cloned()
                })
                .flatten();
            let prior_result = prior_record.as_ref().and_then(|record| {
                self.context
                    .agent_results
                    .read()
                    .iter()
                    .find(|result| result.agent_run_id == record.verifier_task_id)
                    .cloned()
            });
            let prior_baseline = self
                .context
                .verification_baselines
                .lock()
                .get(&session_key)
                .cloned();
            let target_actor = ActorId::new(execution.task_id, &self.agent_name);

            let mut result = if let (Some(record), Some(previous)) =
                (prior_record.as_ref(), prior_result.as_ref())
            {
                let prior_decision = match record.decision {
                    VerificationDecision::Approved => "ACCEPTED",
                    VerificationDecision::RejectedMinor => "REJECTED_MINOR",
                    VerificationDecision::RejectedMajor => "REJECTED_MAJOR",
                };
                let repair_delta = verifier
                    .repair_context
                    .as_ref()
                    .zip(prior_baseline.as_ref())
                    .map(|(config, baseline)| {
                        crate::verification::render_repair_delta(
                            &self.context,
                            &verifier.verifier_agent,
                            baseline,
                            config.max_bytes,
                        )
                    })
                    .unwrap_or_else(|| {
                        "No bounded repair delta is available; perform only the focused reads needed to validate the new completion.".to_string()
                    });
                let prompt = format!(
                    "Continue your review of the same '{}' task.\n\nYour prior decision:\n{}\n{}\n\nNew candidate completion summary:\n{}\n\n{}\n\nRe-review instructions:\n- Verify every prior finding and any correctness or efficiency regression materially exposed by the delta.\n- Reuse unchanged passing evidence from your conversation; do not repeat the complete audit or rerun unchanged broad checks.\n- Run focused reproductions and directly affected checks.\n- Report a new issue only when it is material and tied to an explicit requirement or directly exposed by the repair.\n- Batch independent read-only tool calls when useful, but call non-batch-safe tools alone.\n\nVerification instructions remain:\n{}\n\nFinish with complete_task using this exact contract:\n\n{}",
                    self.agent_name,
                    prior_decision,
                    record.feedback,
                    summary,
                    repair_delta,
                    verifier.instructions,
                    crate::verification::VERIFIER_OUTPUT_CONTRACT,
                );
                match engine
                    .continue_at(previous, &prompt, &self.context, self.depth + 1)
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' could not continue: {error}",
                                verifier.name
                            ),
                        });
                    }
                }
            } else {
                let prompt = format!(
                    "You are a completion verifier for agent '{}'.\n\nOriginal task:\n{}\n\nCandidate completion summary:\n{}\n\nVerification instructions:\n{}\n\nThis is the first review pass. Inspect the actual workspace and relevant artifacts, systematically cover the applicable requirement and failure classes, and consolidate every evidenced related finding into this verdict rather than drip-feeding one counterexample per repair cycle. Do not approve merely because the summary claims success. Batch independent read-only tool calls when useful, but call non-batch-safe tools alone. Finish by calling complete_task using this exact contract:\n\n{}",
                    self.agent_name,
                    without_roster(execution.initial_task_prompt),
                    summary,
                    verifier.instructions,
                    crate::verification::VERIFIER_OUTPUT_CONTRACT,
                );
                match engine
                    .run_verifier(
                        &verifier.verifier_agent,
                        &prompt,
                        &self.context,
                        self.depth + 1,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' could not complete: {error}",
                                verifier.name
                            ),
                        });
                    }
                }
            };

            let mut format_attempt = 0;
            let response = loop {
                match crate::verification::parse_response(&result.summary) {
                    Ok(response) => break response,
                    Err(error) if format_attempt < MAX_FORMAT_RETRIES => {
                        format_attempt += 1;
                        let prompt = format!(
                            "Your verification conclusion could not be parsed ({error}). Do not repeat the review or change your conclusion. Re-emit it through complete_task using only this contract:\n\n{}",
                            crate::verification::VERIFIER_OUTPUT_CONTRACT,
                        );
                        result = match engine
                            .continue_at(&result, &prompt, &self.context, self.depth + 1)
                            .await
                        {
                            Ok(result) => result,
                            Err(retry_error) => {
                                return Some(ToolExecutionDecision::FailTask {
                                    reason: format!(
                                        "completion verifier '{}' could not repair its decision format: {retry_error}",
                                        verifier.name
                                    ),
                                });
                            }
                        };
                    }
                    Err(error) => {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' produced an unusable decision after {} attempts: {error}",
                                verifier.name,
                                format_attempt + 1,
                            ),
                        });
                    }
                }
            };
            let mut decision = response.decision;
            if verifier.kind == VerificationKind::Efficiency
                && decision == VerificationDecision::RejectedMajor
            {
                decision = VerificationDecision::RejectedMinor;
            }
            if let Some(config) = &verifier.repair_context {
                let verifier_actor = ActorId::new(result.agent_run_id, &verifier.verifier_agent);
                let baseline = crate::verification::capture_repair_baseline(
                    &self.context,
                    &target_actor,
                    &verifier_actor,
                    config.max_bytes,
                );
                self.context
                    .verification_baselines
                    .lock()
                    .insert(session_key, baseline);
            }
            self.context
                .verification_records
                .write()
                .push(VerificationRecord {
                    timestamp: chrono::Utc::now(),
                    target_task_id: execution.task_id,
                    target_agent: self.agent_name.clone(),
                    verifier_name: verifier.name.clone(),
                    verifier_agent: verifier.verifier_agent.clone(),
                    verifier_task_id: result.agent_run_id,
                    decision: decision.clone(),
                    feedback: response.feedback.clone(),
                });

            match decision {
                VerificationDecision::Approved => {}
                VerificationDecision::RejectedMinor => {
                    let count = {
                        let mut attempts = self.context.verification_attempts.lock();
                        let counts = attempts
                            .entry((execution.task_id, verifier.name.clone()))
                            .or_default();
                        counts.0 = counts.0.saturating_add(1);
                        counts.0
                    };
                    if count > verifier.max_minor_rejections {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' exhausted its minor-correction cap ({}): {}",
                                verifier.name, verifier.max_minor_rejections, response.feedback
                            ),
                        });
                    }
                    return Some(ToolExecutionDecision::Reject {
                        message_to_agent: format!(
                            "Completion was rejected with minor corrections by verifier '{}'. Fix the issues and try again ({count}/{}): {}",
                            verifier.name, verifier.max_minor_rejections, response.feedback
                        ),
                    });
                }
                VerificationDecision::RejectedMajor => {
                    let count = {
                        let mut attempts = self.context.verification_attempts.lock();
                        let counts = attempts
                            .entry((execution.task_id, verifier.name.clone()))
                            .or_default();
                        counts.1 = counts.1.saturating_add(1);
                        counts.1
                    };
                    if count > verifier.max_major_rejections {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' issued a major rejection: {}",
                                verifier.name, response.feedback
                            ),
                        });
                    }
                    return Some(ToolExecutionDecision::Reject {
                        message_to_agent: format!(
                            "Completion was rejected for a major issue by verifier '{}'. Rework it before trying again ({count}/{}): {}",
                            verifier.name, verifier.max_major_rejections, response.feedback
                        ),
                    });
                }
            }
        }
        None
    }
}

#[async_trait]
impl ToolExecutionMiddleware for WorkflowExecutionMiddleware {
    async fn before(
        &self,
        execution: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        if let Some(decision) = self.stop_decision() {
            return Ok(decision);
        }
        match fire_hooks(&self.context, &HookPoint::PreToolExecution, |hook| {
            hook.agent_name = Some(self.agent_name.clone());
            hook.agent_run_id = Some(execution.task_id);
            hook.tool_name = Some(call.tool_name.to_string());
            hook.tool_call_id = Some(call.tool_call_id);
            hook.tool_params = Some(call.parameters.clone());
        }) {
            HookResult::Continue => {}
            HookResult::Skip => {
                return Ok(ToolExecutionDecision::Reject {
                    message_to_agent: "Skipped by PreToolExecution hook".to_string(),
                });
            }
            // The call doesn't run, and the whole run stops.
            HookResult::Abort(reason) => {
                let error = abort_run(&self.context, &HookPoint::PreToolExecution, reason);
                return Ok(self.stop_decision().unwrap_or_else(|| {
                    ToolExecutionDecision::CancelTask {
                        reason: error.to_string(),
                    }
                }));
            }
        }

        if let Some(tracker) = &self.context.resource_tracker
            && let Some((resource, true)) =
                tracker.resource_for_tool(call.tool_name, call.parameters)
        {
            let actor = ActorId::new(execution.task_id, &self.agent_name);
            let conflicts = tracker.check_write(&actor, &resource);
            if tracker.config().block_on_conflict && !conflicts.is_empty() {
                let writers = conflicts
                    .iter()
                    .filter_map(|conflict| conflict.last_writer.as_ref())
                    .map(|writer| writer.agent_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Ok(ToolExecutionDecision::Reject {
                    message_to_agent: format!(
                        "Context freshness conflict for {resource:?}; your context was changed by {writers}. Re-read the affected resource before writing."
                    ),
                });
            }
        }

        if call.tool_name == arko_agents::tool::builtins::COMPLETE_TASK_TOOL_NAME {
            let summary = call
                .parameters
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(decision) = self.verify_completion(execution, summary).await {
                // A verifier stopped with the run decides nothing.
                return Ok(self.stop_decision().unwrap_or(decision));
            }
        }

        Ok(self
            .stop_decision()
            .unwrap_or(ToolExecutionDecision::Continue))
    }

    async fn after(
        &self,
        execution: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
        outcome: ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError> {
        let successful_output = match outcome {
            ToolExecutionOutcome::Success(output)
            | ToolExecutionOutcome::CallerProvided(output) => Some(output),
            ToolExecutionOutcome::Error(_) => None,
        };

        if successful_output.is_some()
            && call.tool_name == arko_agents::tool::builtins::COMPLETE_TASK_TOOL_NAME
        {
            self.context
                .verification_baselines
                .lock()
                .retain(|(task_id, _), _| *task_id != execution.task_id);
            self.context
                .verification_attempts
                .lock()
                .retain(|(task_id, _), _| *task_id != execution.task_id);
        }

        if successful_output.is_some()
            && let Some(tracker) = &self.context.resource_tracker
            && let Some((resource, is_write)) =
                tracker.resource_for_tool(call.tool_name, call.parameters)
        {
            let actor = ActorId::new(execution.task_id, &self.agent_name);
            if is_write {
                let stale_readers = tracker.record_write(&actor, resource.clone());
                for reader in stale_readers {
                    let notice = format!(
                        "Context freshness notice: {resource:?} changed after you read it. Re-read it before relying on or modifying that context."
                    );
                    self.context
                        .messages
                        .inject_task(reader.task_id, notice)
                        .await;
                    // An abort stops the run; the call's own result stands.
                    let _ = fire_hooks_on_context(
                        &self.context,
                        HookPoint::OnFreshnessConflict,
                        |hook| {
                            hook.agent_name = Some(reader.agent_name.clone());
                            hook.agent_run_id = Some(reader.task_id);
                            hook.metadata.insert(
                                "resource".to_string(),
                                serde_json::json!(format!("{resource:?}")),
                            );
                            hook.metadata
                                .insert("writer".to_string(), serde_json::json!(self.agent_name));
                        },
                    );
                }
            } else {
                tracker.record_read(&actor, resource);
            }
        }

        let result = match outcome {
            ToolExecutionOutcome::Success(output)
            | ToolExecutionOutcome::CallerProvided(output)
            | ToolExecutionOutcome::Error(output) => output,
        };

        let hook_result = fire_hooks(&self.context, &HookPoint::PostToolExecution, |hook| {
            hook.agent_name = Some(self.agent_name.clone());
            hook.agent_run_id = Some(execution.task_id);
            hook.tool_name = Some(call.tool_name.to_string());
            hook.tool_call_id = Some(call.tool_call_id);
            hook.tool_params = Some(call.parameters.clone());
            hook.tool_result = Some(result.to_string());
            hook.original_tool_result = Some(result.to_string());
        });
        // The call has run; the run stops, and the stop's CancelTask ends the
        // agent before its next step.
        if let HookResult::Abort(reason) = hook_result {
            abort_run(&self.context, &HookPoint::PostToolExecution, reason);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use arko_agents::AgentLLMConfig;
    use arko_ai::{GenerationConfig, Model};

    use crate::workflow::Workflow;

    #[tokio::test]
    async fn interrupted_runtime_is_cancelled_instead_of_waiting_for_stream_end() {
        let config = AgentConfig::builder(
            "designer".into(),
            "Designer".into(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_sol(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("Designer".into())
        .instructions("Design".into())
        .build()
        .unwrap();
        let workflow = Workflow::builder()
            .name("interruption-test")
            .workspace(std::env::current_dir().unwrap())
            .llm_client(Arc::new(LLMClient::new()))
            .agent("designer", config)
            .build()
            .unwrap();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(4);
        let (command_tx, mut command_rx) = tokio::sync::mpsc::channel(4);
        event_tx
            .send(AgentEvent::TaskStateChanged(TaskState::Interrupted {
                reason: "Persistent network failure".into(),
            }))
            .await
            .unwrap();
        // Keep event_tx alive: an interrupted runtime has not closed its stream.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            drive_event_loop(
                "designer",
                uuid::Uuid::new_v4(),
                &mut event_rx,
                &command_tx,
                workflow.context(),
                None,
            ),
        )
        .await
        .expect("interrupted workflow must not hang");
        assert!(
            matches!(result, Err(WorkflowError::AgentInterrupted { reason, .. }) if reason == "Persistent network failure")
        );
        assert!(matches!(
            command_rx.try_recv().unwrap(),
            UserCommand::CancelTask
        ));
    }

    /// The run status follows the task state, never the wording of a
    /// failure reason.
    #[tokio::test]
    async fn only_a_cancelled_task_is_a_cancelled_run() {
        let config = AgentConfig::builder(
            "designer".into(),
            "Designer".into(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_sol(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("Designer".into())
        .instructions("Design".into())
        .build()
        .unwrap();
        let workflow = Workflow::builder()
            .name("status-test")
            .workspace(std::env::current_dir().unwrap())
            .llm_client(Arc::new(LLMClient::new()))
            .agent("designer", config.clone())
            .build()
            .unwrap();
        for (state, status) in [
            (
                TaskState::Failed {
                    reason: "The upload was cancelled by the server.".into(),
                },
                crate::workflow::result::AgentRunStatus::Failed,
            ),
            (
                TaskState::Cancelled {
                    reason: arko_agents::task::CANCELLED_BY_USER.into(),
                },
                crate::workflow::result::AgentRunStatus::Cancelled,
            ),
        ] {
            let mut task = Task::new(config.clone(), "design".into(), ".".into(), None);
            task.set_state(state);
            let result = finalize_agent_task("designer".into(), task, workflow.context(), None)
                .await
                .unwrap();
            assert_eq!(result.status, status);
        }
    }

    #[test]
    fn workflow_skill_setup_injects_auto_load_and_registers_on_demand() {
        use arko_agents::skill::{InvalidSkillPolicy, SkillSource, SkillsConfig};
        let workspace =
            std::env::temp_dir().join(format!("arko-workflow-skills-{}", uuid::Uuid::new_v4()));
        for (name, auto_load) in [("principles", true), ("reference", false)] {
            let dir = workspace.join(".skills").join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("SKILL.md"), format!("---\nname: {name}\ndescription: Test skill\nauto_load: {auto_load}\n---\n\n{name} content")).unwrap();
        }
        let mut config = AgentConfig::builder(
            "designer".into(),
            "Designer".into(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_sol(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("Designer".into())
        .instructions("Design".into())
        .build()
        .unwrap();
        config.skills_config = Some(SkillsConfig {
            workspace_source: Some(SkillSource {
                path: ".skills".into(),
                include: None,
                exclude: None,
            }),
            global_sources: vec![],
            on_invalid_skill: InvalidSkillPolicy::Error,
        });
        let mut registry = ToolRegistry::new();
        let context = WorkflowEngine::setup_skills(&mut config, &mut registry, &workspace)
            .unwrap()
            .unwrap();
        assert_eq!(
            context.auto_loaded_skills[0].content.trim(),
            "principles content"
        );
        assert_eq!(context.on_demand_skills[0].name, "reference");
        assert!(registry.get("load_skill").is_some());
        assert!(
            config
                .tool_config
                .available_tool_names
                .contains(&"load_skill".into())
        );
        std::fs::write(
            workspace.join(".skills/principles/SKILL.md"),
            "invalid frontmatter",
        )
        .unwrap();
        assert!(WorkflowEngine::setup_skills(&mut config, &mut registry, &workspace).is_err());
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[tokio::test]
    async fn middleware_uses_registered_agent_name_instead_of_display_name() {
        let workspace = std::env::temp_dir().join(format!(
            "arko-workflow-agent-identity-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();
        let observed_agent = Arc::new(Mutex::new(None));
        let hook_observation = observed_agent.clone();
        let config = AgentConfig::builder(
            "coder-config".to_string(),
            "Friendly Coder Display Name".to_string(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_luna(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test coder".to_string())
        .instructions("test instructions".to_string())
        .build()
        .unwrap();
        let mut reviewer_config = config.clone();
        reviewer_config.id = "reviewer-config".to_string();
        reviewer_config.name = "Friendly Reviewer Display Name".to_string();
        let workflow = Workflow::builder()
            .name("agent-identity-test")
            .workspace(&workspace)
            .llm_client(Arc::new(LLMClient::new()))
            .agent("coder", config)
            .agent("reviewer", reviewer_config)
            .completion_verifier(
                CompletionVerifierConfig::correctness("correctness", "reviewer").target("coder"),
            )
            .hook(
                HookPoint::PreToolExecution,
                Arc::new(move |context| {
                    *hook_observation.lock().unwrap() = context.agent_name.clone();
                    HookResult::Continue
                }),
            )
            .build()
            .unwrap();
        let concrete_middleware = WorkflowExecutionMiddleware {
            agent_name: "coder".to_string(),
            context: workflow.context().clone(),
            llm_client: Arc::new(LLMClient::new()),
            persistence: None,
            depth: 0,
        };
        assert_eq!(
            concrete_middleware
                .applicable_completion_verifiers()
                .map(|verifier| verifier.name.as_str())
                .collect::<Vec<_>>(),
            vec!["correctness"]
        );
        let middleware = workflow
            .engine
            .build_execution_middleware("coder", workflow.context(), 0);
        let parameters = HashMap::new();
        let execution = ToolExecutionContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "Friendly Coder Display Name",
            initial_task_prompt: "test",
            current_working_directory: &workspace,
        };
        let call = ToolCallContext {
            tool_call_id: uuid::Uuid::new_v4(),
            tool_name: "test_tool",
            parameters: &parameters,
        };

        let decision = middleware.before(&execution, &call).await.unwrap();

        assert_eq!(decision, ToolExecutionDecision::Continue);
        assert_eq!(observed_agent.lock().unwrap().as_deref(), Some("coder"));
        std::fs::remove_dir(&workspace).unwrap();
    }

    /// A distiller's target is the agent's registration name; its display
    /// name (`AgentConfig::name`, which the runtime reports) doesn't match.
    #[tokio::test]
    async fn distillers_target_registration_names_not_display_names() {
        use crate::distillation::DistillationConfig;
        use crate::distillation::tests::{MockServer, anthropic_reply};
        let model = Model::anthropic_claude_opus_5_5();
        let coder = AgentConfig::builder(
            "coder-config".to_string(),
            "Friendly Coder Display Name".to_string(),
            AgentLLMConfig {
                model: model.clone(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test coder".to_string())
        .instructions("test instructions".to_string())
        .build()
        .unwrap();
        let mut reviewer = coder.clone();
        reviewer.id = "reviewer-config".to_string();
        reviewer.name = "Friendly Reviewer Display Name".to_string();
        let builder = |server: &MockServer, target: &str| {
            Workflow::builder()
                .name("distillation-target-test")
                .workspace(std::env::temp_dir())
                .llm_client(Arc::new(server.anthropic_client()))
                .agent("coder", coder.clone())
                .agent("reviewer", reviewer.clone())
                .ephemeral()
                .distiller(
                    DistillationConfig::new("bulk", model.clone())
                        .min_chars(10)
                        .target_agent(target),
                )
                .build()
        };
        let summary = "The failing test is in lib.rs.";
        let server = MockServer::start(anthropic_reply(summary, "end_turn")).await;
        assert!(matches!(
            builder(&server, "Friendly Coder Display Name"),
            Err(WorkflowError::BuilderInvalid(message)) if message.contains("unknown agent")
        ));

        let workflow = builder(&server, "coder").unwrap();
        assert!(
            workflow
                .engine
                .build_interceptor_config("reviewer", workflow.context())
                .is_none()
        );
        let interceptor = workflow
            .engine
            .build_interceptor_config("coder", workflow.context())
            .unwrap()
            .interceptor;
        let task = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "Friendly Coder Display Name",
            initial_task_prompt: "find the failing test",
            current_working_directory: &workflow.context().workspace,
        };
        let raw = "raw tool output ".repeat(10);
        let outcome = interceptor
            .intercept("search", &raw, &HashMap::new(), &task)
            .await
            .unwrap();
        assert!(matches!(outcome, InterceptedResult::Replaced(ref text) if text == summary));
        assert_eq!(server.requests(), 1);
        let records = workflow.context().distillation_records.read();
        assert_eq!(records[0].agent_name, "coder");
    }

    #[derive(Debug)]
    struct PassThrough;

    #[async_trait]
    impl ToolResultInterceptor for PassThrough {
        async fn intercept(
            &self,
            _tool_name: &str,
            _raw_result: &str,
            _params: &HashMap<String, Value>,
            _task_context: &TaskContext<'_>,
        ) -> Result<InterceptedResult, ToolError> {
            Ok(InterceptedResult::Unchanged)
        }
    }

    /// Under a retrying failure policy, a refused or truncated summary is not
    /// requested (and billed) again; an empty one is.
    #[tokio::test]
    async fn chained_distillers_retry_only_rejections_that_could_turn_out_differently() {
        use crate::distillation::DistillationConfig;
        use crate::distillation::tests::{MockServer, anthropic_reply};
        let config = AgentConfig::builder(
            "coder".to_string(),
            "Coder".to_string(),
            AgentLLMConfig {
                model: Model::anthropic_claude_opus_5_5(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test coder".to_string())
        .instructions("test instructions".to_string())
        .build()
        .unwrap();
        for (reply, requests) in [
            (anthropic_reply("The failing test is in", "max_tokens"), 1),
            (anthropic_reply("The failing test", "refusal"), 1),
            (anthropic_reply("", "end_turn"), 3),
        ] {
            let server = MockServer::start(reply).await;
            let workflow = Workflow::builder()
                .name("distillation-retry-test")
                .workspace(std::env::temp_dir())
                .llm_client(Arc::new(server.anthropic_client()))
                .agent("coder", config.clone())
                .ephemeral()
                // A second interceptor makes the engine chain them.
                .tool_result_interceptor(Arc::new(PassThrough))
                .distiller(
                    DistillationConfig::new("bulk", Model::anthropic_claude_opus_5_5())
                        .min_chars(10),
                )
                .interception_failure_policy(
                    arko_agents::InterceptionFailurePolicy::RetryThenUseRawResult {
                        max_attempts: 3,
                    },
                )
                .build()
                .unwrap();
            let interceptor = workflow
                .engine
                .build_interceptor_config("coder", workflow.context())
                .unwrap()
                .interceptor;
            let task = TaskContext {
                task_id: uuid::Uuid::new_v4(),
                agent_name: "coder",
                initial_task_prompt: "find the failing test",
                current_working_directory: &workflow.context().workspace,
            };
            let outcome = interceptor
                .intercept("search", &"raw tool output ".repeat(10), &HashMap::new(), &task)
                .await
                .unwrap();
            assert!(matches!(outcome, InterceptedResult::Unchanged));
            assert_eq!(server.requests(), requests);
            assert_eq!(workflow.context().distillation_records.read().len(), requests);
        }
    }
}
