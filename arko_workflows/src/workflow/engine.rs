use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::skill::{SkillManager, SkillPromptContext};
use arko_agents::tool::builtins::COMPLETE_TASK_TOOL_NAME;
use arko_agents::{
    AgentConfig, AgentRuntime, HostDecisions, InterceptedResult, InterceptionFailurePolicy,
    InterceptorConfig, PendingToolCallInfo, Task, ToolCallContext, ToolError, ToolExecutionContext,
    ToolExecutionDecision, ToolExecutionMiddleware, ToolExecutionOutcome, ToolRegistry,
    ToolResultInterceptor,
    runtime::{
        command::{ToolApprovalDecision, UserCommand},
        event::AgentEvent,
    },
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
use crate::messaging::registry::WaitReason;
use crate::messaging::tools::*;
use crate::persistence::PersistenceManager;
use crate::policy::{
    AgentPause, AgentResume, ApprovalPolicy, RecoveryAction, ToolApprovalOutcome,
    ToolApprovalRequest,
};
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
///
/// The chain applies the agent's
/// [`InterceptionFailurePolicy`] to
/// each stage: a failing stage is retried up to the policy's attempts and,
/// if it still fails, skipped, so the next stage (and the agent) gets the
/// output of the last stage that succeeded. A failure never undoes an
/// earlier stage's work (such as a redaction), and a stage that succeeded
/// is never run (or, for a distiller, paid for) again. Under `InformAgent`
/// and `RetryThenInformAgent` the result ends with a note naming the
/// failures. The chain itself never fails, so the runtime doesn't retry it.
struct ChainedInterceptor {
    interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
    /// The policy for the tools the workflow's policy covers.
    policy: InterceptionFailurePolicy,
    /// The policy for `load_skill` results, which the runtime passes to
    /// the chain only when the agent's skills configuration sets one.
    load_skill_policy: Option<InterceptionFailurePolicy>,
}

impl ChainedInterceptor {
    fn new(
        interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
        policy: InterceptionFailurePolicy,
        load_skill_policy: Option<InterceptionFailurePolicy>,
    ) -> Self {
        Self {
            interceptors,
            policy,
            load_skill_policy,
        }
    }

    fn policy_for(&self, tool_name: &str) -> &InterceptionFailurePolicy {
        match &self.load_skill_policy {
            Some(policy) if tool_name == arko_agents::skill::LOAD_SKILL_TOOL_NAME => policy,
            _ => &self.policy,
        }
    }
}

impl std::fmt::Debug for ChainedInterceptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainedInterceptor")
            .field("count", &self.interceptors.len())
            .field("policy", &self.policy)
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
        let policy = self.policy_for(tool_name);
        let max_attempts = policy.max_attempts();
        let mut current = raw_result.to_string();
        let mut modified = false;
        let mut failures = Vec::new();

        for (stage, interceptor) in self.interceptors.iter().enumerate() {
            let mut last_error = None;
            for attempt in 1..=max_attempts {
                match interceptor
                    .intercept(tool_name, &current, params, task_context)
                    .await
                {
                    Ok(InterceptedResult::Unchanged) => {
                        last_error = None;
                        break;
                    }
                    Ok(InterceptedResult::Replaced(new_value)) => {
                        current = new_value;
                        modified = true;
                        last_error = None;
                        break;
                    }
                    Err(error) => {
                        log::warn!(
                            "Task {}: tool-result interceptor {} of {} failed for '{}' (attempt {attempt}/{max_attempts}): {error}",
                            task_context.task_id,
                            stage + 1,
                            self.interceptors.len(),
                            tool_name,
                        );
                        last_error = Some(error);
                    }
                }
            }
            if let Some(error) = last_error {
                log::warn!(
                    "Task {}: skipping tool-result interceptor {} of {} for '{}'; the next gets the output so far",
                    task_context.task_id,
                    stage + 1,
                    self.interceptors.len(),
                    tool_name,
                );
                failures.push(error.to_string());
            }
        }

        if !failures.is_empty() && policy.informs_agent() {
            let errors = failures.join("; ");
            current = if modified {
                format!(
                    "{current}\n\n[A tool-result interceptor failed ({errors}), so this result skipped that step.]"
                )
            } else {
                format!(
                    "{current}\n\n[This is the full result: it was to be shortened, but the tool-result interceptor failed ({errors}).]"
                )
            };
            modified = true;
        }

        Ok(if modified {
            InterceptedResult::Replaced(current)
        } else {
            InterceptedResult::Unchanged
        })
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
) {
    let _ = ctx.event_tx.send(crate::workflow::result::WorkflowEvent {
        timestamp: chrono::Utc::now(),
        workflow_run_id: ctx.workflow_run_id,
        agent_name: agent_name.to_string(),
        agent_run_id,
        event: event.clone(),
    });
    if let Some(persistence) = persistence {
        // Never fails: a failed write is logged and retried.
        persistence
            .append_event(agent_name, agent_run_id, event)
            .await;
    }
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

/// The error-recovery state of one drive of an agent.
#[derive(Default)]
struct RecoveryState {
    /// The last error-recovery pause the agent reported, and its cause.
    pause: Option<(String, arko_agents::ErrorRecoveryCause)>,
    /// Automatic recoveries in this drive (for `ErrorRecoveryPolicy::Limited`).
    recovered: u32,
    /// Pauses in a row without a model response in between.
    repeat: u32,
}

/// Suspends the agent for its caller's decision: the run call returns
/// [`WorkflowError::AgentPaused`].
async fn pause_for_caller(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    pause: AgentPause,
) -> Result<(), WorkflowError> {
    let _ = cmd_tx.send(UserCommand::SuspendTask).await;
    Err(WorkflowError::AgentPaused {
        agent: agent_name.to_string(),
        task_id: agent_run_id,
        pause,
    })
}

/// Logs, once per agent, that a call the agent's configuration marks for
/// approval was approved under `ApproveAll`.
fn warn_auto_approved(ctx: &SharedContext, agent_name: &str, call: &PendingToolCallInfo) {
    if let Some(warning) = auto_approval_warning(ctx, agent_name, call) {
        log::warn!("{warning}");
    }
}

/// The warning for the first call of `agent_name` its configuration marks
/// for approval that `ApproveAll` approves; `None` for its later ones.
pub(super) fn auto_approval_warning(
    ctx: &SharedContext,
    agent_name: &str,
    call: &PendingToolCallInfo,
) -> Option<String> {
    if !ctx.approval_warnings.lock().insert(agent_name.to_string()) {
        return None;
    }
    Some(format!(
        "Workflow '{}': approving '{}' for agent '{agent_name}' under ApprovalPolicy::ApproveAll (the default), although {}. The agent's other calls that need approval are approved too, without further warnings. Set WorkflowBuilder::approval_policy to decide such calls.",
        ctx.workflow_name,
        call.tool_name,
        call.approval_reason.describe()
    ))
}

/// Stops a nested run (inside another agent's tool call: a completion
/// verifier or a delegated run) that needs a decision the workflow leaves
/// to the caller: its parent waits on the call, so it can't be paused.
/// Ending it keeps the outcome bounded: a verifier's failed review, or a
/// tool error for the delegating agent.
fn nested_cannot_pause(ctx: &SharedContext, agent_name: &str, needs: &str) -> WorkflowError {
    let reason = format!(
        "Stopped: agent '{agent_name}' runs inside another agent's tool call and can't be paused, but it needs {needs}. Configure it to auto-approve the tools it uses, or decide such calls with ApprovalPolicy::Handler."
    );
    log::warn!("Workflow '{}': {reason}", ctx.workflow_name);
    WorkflowError::AgentCancelled {
        agent: agent_name.to_string(),
        reason,
    }
}

/// Decides tool calls awaiting approval under the workflow's approval
/// policy: a decision per call, or `None` to pause the agent for its
/// caller. A `nested` run (inside another agent's tool call) can't pause:
/// under `RespectAgentConfig` its `complete_task` is approved (its caller
/// waits for its result), and a call left undecided stops it instead. Each
/// approved call's approval reason is kept for the tool hooks until it has
/// run.
async fn decide_tool_calls(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    calls: &[PendingToolCallInfo],
    nested: bool,
    ctx: &Arc<SharedContext>,
) -> Result<Option<Vec<(uuid::Uuid, ToolApprovalDecision)>>, WorkflowError> {
    let decisions = if nested && matches!(ctx.approval_policy, ApprovalPolicy::RespectAgentConfig) {
        approve_nested_completions(calls)
    } else {
        decide_tool_calls_by_policy(agent_name, agent_run_id, calls, ctx).await?
    };
    let Some(decisions) = decisions else {
        if nested {
            let tools = calls
                .iter()
                .filter(|call| {
                    !matches!(ctx.approval_policy, ApprovalPolicy::RespectAgentConfig)
                        || call.tool_name != COMPLETE_TASK_TOOL_NAME
                })
                .map(|call| format!("'{}'", call.tool_name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(nested_cannot_pause(
                ctx,
                agent_name,
                &format!("approval for {tools}"),
            ));
        }
        return Ok(None);
    };
    record_approval_reasons(ctx, agent_run_id, calls, &decisions);
    Ok(Some(decisions))
}

/// A nested run's calls under `RespectAgentConfig`: `complete_task` is
/// approved; any other call is left undecided (`None`), since no one can
/// be asked.
fn approve_nested_completions(
    calls: &[PendingToolCallInfo],
) -> Option<Vec<(uuid::Uuid, ToolApprovalDecision)>> {
    calls
        .iter()
        .map(|call| {
            (call.tool_name == COMPLETE_TASK_TOOL_NAME).then_some((
                call.tool_call_id,
                ToolApprovalDecision::Approve {
                    custom_tool_output: None,
                    message_to_agent: None,
                },
            ))
        })
        .collect()
}

/// [`decide_tool_calls`] by the policy alone.
async fn decide_tool_calls_by_policy(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    calls: &[PendingToolCallInfo],
    ctx: &Arc<SharedContext>,
) -> Result<Option<Vec<(uuid::Uuid, ToolApprovalDecision)>>, WorkflowError> {
    let approve = || ToolApprovalDecision::Approve {
        custom_tool_output: None,
        message_to_agent: None,
    };
    let decisions = match &ctx.approval_policy {
        ApprovalPolicy::ApproveAll => {
            for call in calls {
                warn_auto_approved(ctx, agent_name, call);
            }
            calls
                .iter()
                .map(|call| (call.tool_call_id, approve()))
                .collect()
        }
        ApprovalPolicy::RespectAgentConfig => return Ok(None),
        ApprovalPolicy::Handler(handler) => {
            let mut decisions = Vec::with_capacity(calls.len());
            for call in calls {
                let request = ToolApprovalRequest {
                    workflow_run_id: ctx.workflow_run_id,
                    agent_run_id,
                    agent_name: agent_name.to_string(),
                    tool_call_id: call.tool_call_id,
                    tool_name: call.tool_name.clone(),
                    parameters: call.parameters.clone(),
                    tool_description: call.tool_description.clone(),
                    reason: call.approval_reason.clone(),
                };
                // The handler may wait on a person; a stop doesn't.
                let outcome = tokio::select! {
                    biased;
                    _ = ctx.run_control.stopped() => return Err(stop_error(ctx)),
                    outcome = handler.decide(request) => outcome,
                };
                decisions.push((
                    call.tool_call_id,
                    match outcome {
                        ToolApprovalOutcome::Approve => approve(),
                        ToolApprovalOutcome::Reject { reason } => {
                            ToolApprovalDecision::Reject { reason }
                        }
                        ToolApprovalOutcome::Pause => return Ok(None),
                    },
                ));
            }
            decisions
        }
    };
    Ok(Some(decisions))
}

/// Keeps the approval reasons of the approved `calls` for the tool hooks,
/// until each has run.
fn record_approval_reasons(
    ctx: &SharedContext,
    agent_run_id: uuid::Uuid,
    calls: &[PendingToolCallInfo],
    decisions: &[(uuid::Uuid, ToolApprovalDecision)],
) {
    let mut reasons = ctx.approval_reasons.lock();
    for call in calls {
        let approved = decisions.iter().any(|(id, decision)| {
            *id == call.tool_call_id && matches!(decision, ToolApprovalDecision::Approve { .. })
        });
        if approved {
            reasons.insert(
                call.tool_call_id,
                (agent_run_id, call.approval_reason.clone()),
            );
        }
    }
}

/// Sends the decisions for `calls`, one call's or a batch's.
async fn send_tool_decisions(
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    batch_id: Option<uuid::Uuid>,
    decisions: Vec<(uuid::Uuid, ToolApprovalDecision)>,
) {
    let command = match batch_id {
        Some(batch_id) => UserCommand::SubmitBatchToolApprovals {
            batch_id,
            decisions: decisions.into_iter().collect(),
        },
        None => match decisions.into_iter().next() {
            Some((
                tool_call_id,
                ToolApprovalDecision::Approve {
                    custom_tool_output,
                    message_to_agent,
                },
            )) => UserCommand::ApproveToolUse {
                tool_call_id,
                custom_tool_output,
                message_to_agent,
            },
            Some((tool_call_id, ToolApprovalDecision::Reject { reason })) => {
                UserCommand::RejectToolUse {
                    tool_call_id,
                    reason,
                }
            }
            None => return,
        },
    };
    let _ = cmd_tx.send(command).await;
}

/// Answers an error-recovery pause under the workflow's error recovery
/// policy: continues (after a wait), fails the task, or pauses the agent
/// for its caller; a `nested` run (inside another agent's tool call) can't
/// pause, so it fails instead.
async fn answer_recovery_pause(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    fallback_error: String,
    recovery: &mut RecoveryState,
    nested: bool,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &Arc<SharedContext>,
) -> Result<(), WorkflowError> {
    let (error, cause) = recovery
        .pause
        .clone()
        .unwrap_or((fallback_error, arko_agents::ErrorRecoveryCause::Unknown));
    recovery.repeat = recovery.repeat.saturating_add(1);
    let now_unix_ms = u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0);
    match ctx.error_recovery_policy.action(
        &cause,
        &error,
        recovery.recovered,
        recovery.repeat,
        &ctx.recovery_waits,
        now_unix_ms,
    ) {
        RecoveryAction::ContinueAfter(wait) => {
            let wait = ctx
                .recovery_waits
                .with_jitter(wait, arko_ai::retry::random_sample());
            if !wait.is_zero() {
                let until = chrono::Utc::now()
                    + chrono::Duration::from_std(wait).unwrap_or(chrono::Duration::MAX);
                ctx.registry.set_waiting(
                    agent_run_id,
                    match cause {
                        arko_agents::ErrorRecoveryCause::UsageLimit { .. } => {
                            WaitReason::UsageLimit { until: Some(until) }
                        }
                        _ => WaitReason::Retrying { until: Some(until) },
                    },
                );
                log::info!(
                    "Workflow '{}': agent '{agent_name}' paused ({}); continuing in {wait:?}: {error}",
                    ctx.workflow_name,
                    crate::policy::cause_name(&cause)
                );
                tokio::select! {
                    biased;
                    _ = ctx.run_control.stopped() => return Err(stop_error(ctx)),
                    _ = tokio::time::sleep(wait) => {}
                }
                wait_while_paused(ctx).await?;
            }
            recovery.recovered = recovery.recovered.saturating_add(1);
            let _ = cmd_tx
                .send(UserCommand::ContinueAfterError { note: None })
                .await;
            Ok(())
        }
        RecoveryAction::Fail(reason) => {
            let _ = cmd_tx
                .send(UserCommand::RejectErrorRecovery {
                    message: Some(reason),
                })
                .await;
            Ok(())
        }
        // A completion a verifier could not review is left paused even inside
        // another agent's call: the run call returns `VerifierFailed` at
        // once, so nothing waits on the pause.
        RecoveryAction::ReturnToCaller
            if nested && cause != arko_agents::ErrorRecoveryCause::Middleware =>
        {
            Err(nested_cannot_pause(
                ctx,
                agent_name,
                &format!(
                    "an error-recovery decision ({}): {error}",
                    crate::policy::cause_name(&cause)
                ),
            ))
        }
        RecoveryAction::ReturnToCaller => {
            pause_for_caller(
                agent_name,
                agent_run_id,
                cmd_tx,
                AgentPause::ErrorRecovery { error, cause },
            )
            .await
        }
    }
}

/// Drive the agent event loop with default workflow behavior.
///
/// Answers tool approvals under the workflow's [`ApprovalPolicy`] and
/// error-recovery pauses under its [`ErrorRecoveryPolicy`], and tracks
/// budget. Tool hooks and freshness policy are enforced by execution
/// middleware, so auto-approved and batched calls follow the same path.
///
/// Returns when the agent ends; with [`WorkflowError::UserInputRequired`] or
/// [`WorkflowError::AgentPaused`] once it is suspended for its caller; or
/// with another error when the workflow must stop it (including the stop's
/// error once the run is cancelled or aborted), and the caller then stops
/// the agent.
///
/// A `batch_stop` (fail-fast in `run_agents_parallel`) that is set stops the
/// agent too: the loop returns `AgentCancelled` with its reason.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn drive_event_loop(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
    batch_stop: Option<tokio::sync::watch::Receiver<Option<String>>>,
    nested: bool,
) -> Result<(), WorkflowError> {
    let drive = Drive {
        agent_name,
        agent_run_id,
        nested,
        ctx,
        persistence,
    };
    let Some(mut batch_stop) = batch_stop else {
        return drive_events(&drive, event_rx, cmd_tx).await;
    };
    let stopped = tokio::select! {
        biased;
        stopped = batch_stop.wait_for(Option::is_some) => {
            stopped.map(|reason| reason.clone().unwrap_or_default())
        }
        result = drive_events(&drive, event_rx, cmd_tx) => {
            return result;
        }
    };
    match stopped {
        Ok(reason) => Err(WorkflowError::AgentCancelled {
            agent: agent_name.to_string(),
            reason,
        }),
        // The call that could stop it is gone: drive on.
        Err(_) => drive_events(&drive, event_rx, cmd_tx).await,
    }
}

/// The run a driver drives.
struct Drive<'a> {
    agent_name: &'a str,
    agent_run_id: uuid::Uuid,
    /// Runs inside another agent's tool call, so it can't pause for the
    /// caller.
    nested: bool,
    ctx: &'a Arc<SharedContext>,
    persistence: Option<&'a Arc<PersistenceManager>>,
}

/// [`drive_event_loop`] without a batch stop.
async fn drive_events(
    drive: &Drive<'_>,
    event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
) -> Result<(), WorkflowError> {
    let Drive {
        agent_name,
        agent_run_id,
        nested,
        ctx,
        persistence,
    } = *drive;
    let mut recovery = RecoveryState::default();
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
        publish_event(agent_name, agent_run_id, &event, ctx, persistence).await;
        ctx.registry.observe(agent_run_id, &event);
        if ctx.budget.is_exceeded() && !matches!(&event, AgentEvent::UsageReported(_)) {
            let _ = cmd_tx.try_send(UserCommand::CancelTask);
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        match event {
            AgentEvent::ToolCallAwaitingApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
                approval_reason,
            } => {
                let calls = [PendingToolCallInfo {
                    tool_call_id,
                    tool_name,
                    parameters,
                    tool_description,
                    approval_reason,
                }];
                match decide_tool_calls(agent_name, agent_run_id, &calls, nested, ctx).await? {
                    Some(decisions) => send_tool_decisions(cmd_tx, None, decisions).await,
                    None => {
                        return pause_for_caller(
                            agent_name,
                            agent_run_id,
                            cmd_tx,
                            AgentPause::ToolApproval {
                                calls: calls.to_vec(),
                            },
                        )
                        .await;
                    }
                }
            }

            AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } => match decide_tool_calls(agent_name, agent_run_id, &pending_calls, nested, ctx)
                .await?
            {
                Some(decisions) => send_tool_decisions(cmd_tx, Some(batch_id), decisions).await,
                None => {
                    return pause_for_caller(
                        agent_name,
                        agent_run_id,
                        cmd_tx,
                        AgentPause::ToolApproval {
                            calls: pending_calls,
                        },
                    )
                    .await;
                }
            },

            AgentEvent::UsageReported(ref usage) => {
                // A model call went through: the next pause starts a new
                // series of automatic waits.
                recovery.repeat = 0;
                let cost_result = charge_usage(agent_name, usage, ctx);
                if cost_result.exceeded {
                    let _ = cmd_tx.try_send(UserCommand::CancelTask);
                    return Err(WorkflowError::BudgetExceeded {
                        spent: cost_result.new_total,
                        limit: ctx.budget.limit().unwrap_or(0.0),
                    });
                }
                if ctx.budget.is_agent_exceeded(agent_name) {
                    let _ = cmd_tx.try_send(UserCommand::CancelTask);
                    return Err(WorkflowError::AgentBudgetExceeded {
                        agent: agent_name.to_string(),
                        spent: ctx.budget.agent_spent(agent_name),
                        limit: ctx.budget.agent_limit(agent_name).unwrap_or(0.0),
                    });
                }

                if ctx.budget.is_exceeded() {
                    let _ = cmd_tx.try_send(UserCommand::CancelTask);
                    return Err(WorkflowError::BudgetExceeded {
                        spent: ctx.budget.total_spent(),
                        limit: ctx.budget.limit().unwrap_or(0.0),
                    });
                }
            }

            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                ref error_message,
                ref cause,
            }) => {
                recovery.pause = Some((error_message.clone(), cause.clone()));
            }

            // The runtime reports every error-recovery pause this way, after
            // the state change that gives its cause.
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                error_message,
                ..
            } => {
                answer_recovery_pause(
                    agent_name,
                    agent_run_id,
                    error_message,
                    &mut recovery,
                    nested,
                    cmd_tx,
                    ctx,
                )
                .await?;
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
                } else if nested {
                    return Err(nested_cannot_pause(
                        ctx,
                        agent_name,
                        "an answer to its question (no user input handler is set)",
                    ));
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
                let _ = cmd_tx.try_send(UserCommand::CancelTask);
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

/// Whether `error` is an agent suspended for its caller, not a failure: it
/// waits for input or is paused (also when a completion verifier could not
/// review its completion), and the run can go on.
pub(crate) fn is_pause(error: &WorkflowError) -> bool {
    matches!(
        error,
        WorkflowError::UserInputRequired { .. }
            | WorkflowError::AgentPaused { .. }
            | WorkflowError::VerifierFailed { .. }
    )
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
    /// Already admitted (`PreAgentSpawn` fired, budget and stop checked).
    pub admitted: bool,
    /// The workflow drives the run from its start, so it decides every wait
    /// (see [`StartedRun::host_decisions`]). Off for `spawn_agent`, whose
    /// caller chooses: `AgentHandle::wait` turns it on.
    pub driven: bool,
}

impl Default for StartOptions {
    fn default() -> Self {
        Self {
            reserved: None,
            roster: true,
            depth: 0,
            admitted: false,
            driven: true,
        }
    }
}

/// Spawned agent drivers, aborted if the call holding them is dropped.
struct AbortAll<T>(Vec<tokio::task::JoinHandle<T>>);

impl<T> Drop for AbortAll<T> {
    fn drop(&mut self) {
        for handle in &self.0 {
            handle.abort();
        }
    }
}

/// A run marked running in the registry before its start: abandoned unless
/// released once the start has been attempted (the start abandons it itself
/// if it fails).
struct Reservation {
    ctx: Arc<SharedContext>,
    task_id: uuid::Uuid,
    armed: bool,
}

impl Reservation {
    fn release(mut self) {
        self.armed = false;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.armed {
            self.ctx.registry.abandon(self.task_id);
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
    /// Stops the run when set, with the reason (fail-fast in
    /// `run_agents_parallel`).
    pub batch_stop: Option<tokio::sync::watch::Receiver<Option<String>>>,
    /// Set while the workflow drives the run: it decides every wait the
    /// agent reaches (under its policies and handlers, however long they
    /// take, also while the workflow is paused) or suspends the agent for
    /// its caller, so the agent's own command timeout doesn't apply. Not
    /// saved with the task: resumed outside a workflow, it is timed as
    /// configured.
    pub host_decisions: HostDecisions,
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
    // A tool call waiting for something (`wait_for_message`) returns now; the
    // runtime reads the command only once the call has returned.
    ctx.run_control.stop_agent(agent_run_id);
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
                    publish_event(agent_name, agent_run_id, &event, ctx, persistence).await;
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
///
/// Boxed: an agent run's driver is large, and its callers (including a
/// verifier's run inside the agent runtime's tool call) keep only a
/// pointer to it on the stack.
pub(crate) fn drive_to_result<'a>(
    run: StartedRun,
    ctx: &'a Arc<SharedContext>,
    persistence: Option<&'a Arc<PersistenceManager>>,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
> {
    Box::pin(drive_run_to_result(run, ctx, persistence))
}

/// [`drive_to_result`], unboxed.
async fn drive_run_to_result(
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
        batch_stop,
        host_decisions,
    } = run;
    host_decisions.set(true);
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
        batch_stop,
        depth > 0,
    )
    .await;

    let stopping = drive_result.as_ref().is_err_and(|error| !is_pause(error));
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
        if drive_result.is_err() {
            // Suspended for its caller: the agent's last events (its
            // suspension) are still recorded.
            while let Some(event) = event_rx.recv().await {
                publish_event(&agent_name, agent_run_id, &event, ctx, persistence).await;
                ctx.registry.observe(agent_run_id, &event);
            }
        }
        Some(join_handle.handle().await)
    };

    let final_task = match wound_down {
        Some(Ok(Ok(task))) => Ok(task),
        Some(Ok(Err(error))) => Err(WorkflowError::Agent(error)),
        Some(Err(error)) => Err(WorkflowError::Channel(format!("Agent join error: {error}"))),
        None => {
            let mut task = started_task;
            let reason = stop_reason(ctx, &drive_result);
            let separator = if reason.ends_with('.') { " " } else { ". " };
            task.set_state(TaskState::Cancelled {
                reason: format!(
                    "{reason}{separator}The agent did not stop within {:?} and was aborted.",
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
    // The runtime has ended: nothing more is delivered to it, and no notice
    // is queued for it while its messages are settled.
    ctx.messages.unregister_active(&agent_name, agent_run_id);
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
            finalize_agent_task(
                agent_name.clone(),
                task,
                &pending_message_ids,
                ctx,
                persistence,
            )
            .await
        }
        Err(error) => Err(error),
    };
    ctx.run_control.unregister(agent_run_id);
    // Set when a verifier that failed paused the task.
    let verifier_failure = ctx.verifier_failures.lock().remove(&agent_run_id);

    if let Some(stop) = ctx.run_control.error() {
        if let Err(error) = &result {
            log::warn!("Agent '{agent_name}' stopped with the workflow run: {error}");
        }
        return Err(stop);
    }
    let result = result?;
    // Paused, neither accepted nor rejected, for its caller.
    if let Some((verifier, reason)) = verifier_failure
        && matches!(
            result.task.current_state,
            TaskState::AwaitingErrorRecoveryApproval {
                cause: arko_agents::ErrorRecoveryCause::Middleware,
                ..
            }
        )
    {
        return Err(WorkflowError::VerifierFailed {
            verifier,
            agent: agent_name,
            task_id: result.agent_run_id,
            reason,
        });
    }
    drive_result?;
    match result.status {
        AgentRunStatus::Completed => Ok(result),
        AgentRunStatus::WaitingForInput => Err(WorkflowError::UserInputRequired {
            agent: agent_name,
            task_id: result.agent_run_id,
            questions: result.error.clone().unwrap_or_default(),
        }),
        AgentRunStatus::Paused => Err(WorkflowError::AgentPaused {
            agent: agent_name,
            task_id: result.agent_run_id,
            pause: AgentPause::of(&result.task.current_state)
                .expect("a paused task awaits a decision"),
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
    // Not recorded, so it can't be continued: freshness forgets its reads.
    if let Some(tracker) = &ctx.resource_tracker {
        tracker.finish_actor(&ActorId::new(agent_run_id, agent_name), false);
    }
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
    AgentRunStatus::of_state(state)
}

/// Why the workflow stopped an agent: the run's stop, or the error the
/// driver stopped it for (an exceeded budget, an interruption, a hook).
fn stop_reason(ctx: &SharedContext, drive_result: &Result<(), WorkflowError>) -> String {
    match (ctx.run_control.cancellation(), drive_result) {
        (Some(cancellation), _) => cancellation.cause.agent_reason(),
        // A stop the workflow already worded (fail-fast).
        (None, Err(WorkflowError::AgentCancelled { reason, .. })) => reason.clone(),
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

/// Records `result` in the run's results, replacing an earlier result of
/// its task (without persistence; see [`PersistenceManager::record_result`]).
fn record_result(ctx: &SharedContext, result: AgentResult) {
    crate::workflow::result::upsert_result(&mut ctx.agent_results.write(), result);
}

/// Records an agent run that stopped: its result (replacing an earlier one
/// of the same task), the messages it holds (acknowledged when it completed;
/// `prompt_ids` are the messages its prompt presented), its lifecycle hook,
/// and, with persistence, its task and a checkpoint.
pub(crate) async fn finalize_agent_task(
    agent_name: String,
    task: Task,
    prompt_ids: &[uuid::Uuid],
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<AgentResult, WorkflowError> {
    let agent_run_id = task.id;
    // Approved calls the run didn't get to run.
    ctx.approval_reasons
        .lock()
        .retain(|_, (task_id, _)| *task_id != agent_run_id);
    // A task that can be continued keeps its reads; freshness no longer
    // notifies it (a continuation is told what changed). First, so that no
    // notice is queued for it once its messages are settled.
    if let Some(tracker) = &ctx.resource_tracker {
        let continued = matches!(
            task.current_state,
            TaskState::Completed { .. }
                | TaskState::AwaitingUserResponse { .. }
                | TaskState::AwaitingToolApproval { .. }
                | TaskState::AwaitingBatchToolApproval { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. }
        );
        tracker.finish_actor(&ActorId::new(agent_run_id, &agent_name), continued);
    }
    // Channel acceptance is not consumption: complete_task may finish with
    // unread InjectContext commands. Only context in the conversation counts
    // as read; context taken but not added stays with the task.
    ctx.messages.settle_run(
        &agent_name,
        agent_run_id,
        &task.conversation,
        prompt_ids,
        &task.pending_context_injections,
        matches!(task.current_state, TaskState::Completed { .. }),
    );
    let pause = AgentPause::of(&task.current_state);
    let result = AgentResult::of_task(agent_name.clone(), task);
    let hook_point = match result.status {
        AgentRunStatus::Completed => HookPoint::PostAgentCompletion,
        AgentRunStatus::Cancelled => HookPoint::PostAgentCancellation,
        AgentRunStatus::Failed => HookPoint::PostAgentFailure,
        AgentRunStatus::Interrupted => HookPoint::PostAgentInterruption,
        AgentRunStatus::WaitingForInput => HookPoint::OnUserInputRequired,
        AgentRunStatus::Paused => HookPoint::OnAgentPaused,
    };
    let summary = result.summary.clone();
    let error = result.error.clone();
    // With persistence, its task is saved first (a failure is logged and
    // retried at the next checkpoint; it never fails the run). Spawned, so
    // the result is recorded even if this call is dropped while it waits
    // for another save.
    match persistence {
        Some(persistence) => {
            let recording = {
                let (persistence, ctx, result) = (persistence.clone(), ctx.clone(), result.clone());
                tokio::spawn(async move { persistence.record_result(&ctx, result).await })
            };
            if let Err(error) = recording.await {
                log::warn!(
                    "Workflow '{}': recording the result of agent '{agent_name}' ({agent_run_id}) failed: {error}",
                    ctx.workflow_name
                );
            }
        }
        None => record_result(ctx, result.clone()),
    }

    let hook_result = fire_hooks_on_context(ctx, hook_point, |hook| {
        hook.agent_name = Some(agent_name);
        hook.agent_run_id = Some(agent_run_id);
        hook.agent_summary = (!summary.is_empty()).then_some(summary);
        hook.error_message = error;
        match &pause {
            Some(AgentPause::ToolApproval { calls }) => {
                hook.metadata
                    .insert("pause".to_string(), serde_json::json!("tool_approval"));
                hook.metadata.insert(
                    "tools".to_string(),
                    serde_json::json!(
                        calls
                            .iter()
                            .map(|call| serde_json::json!({
                                "tool_call_id": call.tool_call_id,
                                "tool_name": call.tool_name,
                                "reason": call.approval_reason.describe(),
                            }))
                            .collect::<Vec<_>>()
                    ),
                );
            }
            Some(AgentPause::ErrorRecovery { cause, .. }) => {
                hook.metadata
                    .insert("pause".to_string(), serde_json::json!("error_recovery"));
                hook.metadata.insert(
                    "cause".to_string(),
                    serde_json::json!(crate::policy::cause_name(cause)),
                );
            }
            None => {}
        }
    });

    // A checkpoint that can't be saved is logged; the next one tries again.
    if let Some(persistence) = persistence
        && let Err(error) = Box::pin(persistence.auto_checkpoint(ctx)).await
    {
        log::warn!(
            "Workflow '{}': could not save the checkpoint after agent '{}' ({agent_run_id}) ended: {error}. The next checkpoint tries again.",
            ctx.workflow_name,
            result.agent_name
        );
    }

    hook_result?;

    Ok(result)
}

pub(crate) struct WorkflowEngine {
    llm_client: Arc<LLMClient>,
    persistence: Option<Arc<PersistenceManager>>,
}

impl WorkflowEngine {
    #[cfg(test)]
    pub(crate) fn persistence(&self) -> Option<&Arc<PersistenceManager>> {
        self.persistence.as_ref()
    }

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

        let (reads, writes) = ctx.memory.access_of(agent_name);
        if reads {
            registry.register(Arc::new(ReadMemoryTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
        }
        if writes {
            registry.register(Arc::new(WriteMemoryTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
        }
        if reads || writes {
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
        if !ctx.message_senders(agent_name).is_empty() {
            registry.register(Arc::new(WaitForMessageTool::new(
                ctx.clone(),
                agent_name.to_string(),
                self.llm_client.clone(),
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
        if let Some(ballots) = &ctx.ballots
            && ctx.voters.contains(agent_name)
        {
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
        load_skill_policy: Option<InterceptionFailurePolicy>,
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
            Arc::new(ChainedInterceptor::new(
                all,
                ctx.interception_failure_policy.clone(),
                load_skill_policy,
            ))
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

        let (reads, writes) = ctx.memory.access_of(agent_name);
        if reads {
            ensure_tool_name(&mut config, READ_MEMORY_TOOL_NAME);
        }
        if writes {
            ensure_tool_name(&mut config, WRITE_MEMORY_TOOL_NAME);
        }
        if reads || writes {
            ensure_tool_name(&mut config, LIST_MEMORIES_TOOL_NAME);
        }
        if !ctx.message_recipients(agent_name).is_empty() {
            ensure_tool_name(&mut config, SEND_MESSAGE_TOOL_NAME);
            ensure_tool_name(&mut config, LIST_RECIPIENTS_TOOL_NAME);
        }
        if !ctx.message_senders(agent_name).is_empty() {
            ensure_tool_name(&mut config, WAIT_FOR_MESSAGE_TOOL_NAME);
        }
        if let Some(services) = ctx.workflow_services.get()
            && services.is_allowed(agent_name)
        {
            ensure_tool_name(&mut config, RUN_WORKFLOW_AGENT_TOOL_NAME);
        }
        if ctx.ballots.is_some() && ctx.voters.contains(agent_name) {
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

    /// The prompt section delivering the messages queued for the agent that
    /// run `task_id` doesn't hold yet, and their IDs, acknowledged once the
    /// run completes.
    fn pending_messages(
        &self,
        agent_name: &str,
        task_id: uuid::Uuid,
        ctx: &Arc<SharedContext>,
    ) -> (String, Vec<uuid::Uuid>) {
        let messages = ctx.messages.pending_for(agent_name, task_id);
        let message_ids = messages.iter().map(|message| message.id).collect();
        let mut section = String::new();
        if !messages.is_empty() {
            section.push_str("\n\n# Messages from other agents:\n");
            for msg in messages {
                section.push_str(&format!(
                    "\n{}\n{}{}\n--- End message ---\n",
                    crate::messaging::prompt_block_marker(msg.id, &msg.from),
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
        let admission = if options.admitted {
            Ok(())
        } else {
            self.admit(agent_name, None, ctx)
        };
        let prepared = admission.and_then(|()| {
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
        let (messages, pending_message_ids) = self.pending_messages(agent_name, task.id, ctx);
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
            options.driven,
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
        driven: bool,
    ) -> StartedRun {
        let load_skill_policy = task
            .agent_config
            .skills_config
            .as_ref()
            .and_then(|skills| skills.load_skill_interception.clone());
        let interceptor_config = self.build_interceptor_config(agent_name, load_skill_policy, ctx);
        let execution_middleware = self.build_execution_middleware(agent_name, ctx, depth);
        let agent_run_id = task.id;
        let started_task = task.clone();
        let host_decisions = HostDecisions::new(driven);
        let runtime = AgentRuntime::new(self.llm_client.clone())
            .with_turn_context(Arc::new(StartNotices {
                context: ctx.clone(),
                agent_name: agent_name.to_string(),
            }))
            .with_host_decisions(host_decisions.clone());
        let (event_rx, cmd_tx, join_handle) = runtime.run_task(
            task,
            Arc::new(registry),
            None,
            false,
            skill_prompt_context,
            interceptor_config,
            Some(execution_middleware),
        );
        ctx.messages.register_run(
            agent_name,
            agent_run_id,
            cmd_tx.clone(),
            &pending_message_ids,
        );
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
            batch_stop: None,
            host_decisions,
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
    /// [`continue_at`](Self::continue_at) for a run inside another agent's
    /// tool call: boxed here, so the caller keeps only a pointer to its
    /// large state on the stack.
    fn continue_nested<'a>(
        &'a self,
        previous: &'a AgentResult,
        prompt: &'a str,
        ctx: &'a Arc<SharedContext>,
        depth: u32,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.continue_at(previous, prompt, ctx, depth))
    }

    /// [`run_verifier`](Self::run_verifier), boxed as
    /// [`continue_nested`](Self::continue_nested) is.
    fn run_verifier_nested<'a>(
        &'a self,
        agent_name: &'a str,
        prompt: &'a str,
        ctx: &'a Arc<SharedContext>,
        depth: u32,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.run_verifier(agent_name, prompt, ctx, depth))
    }

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
    /// prompted, so each one's roster shows the others as running. With a
    /// concurrency cap (`WorkflowBuilder::max_parallel_agents`), at most that
    /// many run at once: each is marked running when it starts. With
    /// fail-fast (`WorkflowBuilder::fail_fast`), the first failure stops the
    /// others, and agents not started yet don't start.
    pub async fn run_agents_parallel(
        &self,
        agents: Vec<(&str, String)>,
        ctx: &Arc<SharedContext>,
    ) -> Result<Vec<AgentResult>, WorkflowError> {
        ctx.run_control.check()?;
        let limit = ctx.parallel.max_agents;
        // Each agent is admitted (PreAgentSpawn, budget, the run's stop)
        // before any is marked running; one refused is that agent's error.
        let admitted = agents
            .iter()
            .map(|(name, _)| self.admit(name, None, ctx))
            .collect::<Vec<_>>();
        // Without a cap every admitted agent starts now, marked running
        // together.
        let mut reserved = if limit.is_none() {
            let reserved = agents
                .iter()
                .zip(&admitted)
                .filter(|(_, admission)| admission.is_ok())
                .map(|((name, _), _)| (uuid::Uuid::new_v4(), *name))
                .collect::<Vec<_>>();
            ctx.registry.start(&reserved);
            reserved.into_iter().map(|(id, _)| id).collect::<Vec<_>>()
        } else {
            Vec::new()
        }
        .into_iter();
        let permits = limit.map(|limit| Arc::new(tokio::sync::Semaphore::new(limit)));
        let (stop, _) = tokio::sync::watch::channel(None::<String>);
        // The index of the agent whose failure stopped the others.
        let trigger = Arc::new(std::sync::OnceLock::<usize>::new());
        // Dropping this call stops its agents.
        let mut join_handles = AbortAll(Vec::new());

        for (index, ((name, prompt), admission)) in agents.into_iter().zip(admitted).enumerate() {
            let engine = WorkflowEngine {
                llm_client: self.llm_client.clone(),
                persistence: self.persistence.clone(),
            };
            let ctx_clone = ctx.clone();
            let name_owned = name.to_string();
            let reservation = admission.map(|()| {
                reserved.next().map(|task_id| Reservation {
                    ctx: ctx.clone(),
                    task_id,
                    armed: true,
                })
            });
            let permits = permits.clone();
            let mut stopped = stop.subscribe();
            let stop = stop.clone();
            let trigger = trigger.clone();

            let handle = tokio::spawn(async move {
                // A capped batch's place, held until this agent's run ends.
                let mut _place = None;
                let reservation = match reservation? {
                    Some(reservation) => reservation,
                    None => {
                        // Capped: wait for a place, unless the batch stops.
                        let permits = permits.expect("a capped batch has permits");
                        let place = tokio::select! {
                            biased;
                            reason = stopped.wait_for(Option::is_some) => {
                                return Err(WorkflowError::AgentCancelled {
                                    agent: name_owned,
                                    reason: reason
                                        .map(|reason| reason.clone().unwrap_or_default())
                                        .unwrap_or_default(),
                                });
                            }
                            place = permits.acquire_owned() => place,
                        };
                        _place = place.ok();
                        // Admitted when the call began; the run may have
                        // stopped, or the budget run out, since.
                        ctx_clone.run_control.check()?;
                        if ctx_clone.budget.is_exceeded() {
                            return Err(WorkflowError::BudgetExceeded {
                                spent: ctx_clone.budget.total_spent(),
                                limit: ctx_clone.budget.limit().unwrap_or(0.0),
                            });
                        }
                        let task_id = uuid::Uuid::new_v4();
                        ctx_clone.registry.start(&[(task_id, name_owned.as_str())]);
                        Reservation {
                            ctx: ctx_clone.clone(),
                            task_id,
                            armed: true,
                        }
                    }
                };
                let options = StartOptions {
                    reserved: Some(reservation.task_id),
                    admitted: true,
                    ..StartOptions::default()
                };
                let run = engine.start_agent(&name_owned, &prompt, &ctx_clone, options);
                reservation.release();
                let mut run = run?;
                run.batch_stop = Some(stopped);
                let result = drive_to_result(run, &ctx_clone, engine.persistence.as_ref()).await;
                if ctx_clone.parallel.fail_fast
                    && let Err(error) = &result
                    && !is_pause(error)
                    && trigger.set(index).is_ok()
                {
                    stop.send_replace(Some(format!(
                        "Stopped because agent '{name_owned}' failed in the same parallel run (fail_fast): {error}"
                    )));
                }
                // The place frees up as this returns, after a failure has
                // stopped the batch, so a waiting agent doesn't start then.
                result
            });
            join_handles.0.push(handle);
        }

        let mut results = Vec::new();
        let mut errors = Vec::new();
        for (index, handle) in join_handles.0.iter_mut().enumerate() {
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(error)) => {
                    // An agent that paused is not a failure: its recorded
                    // result, to be continued, is returned with the others.
                    let paused = match &error {
                        WorkflowError::UserInputRequired { task_id, .. }
                        | WorkflowError::AgentPaused { task_id, .. }
                        | WorkflowError::VerifierFailed { task_id, .. } => ctx
                            .agent_results
                            .read()
                            .iter()
                            .find(|result| result.agent_run_id == *task_id)
                            .cloned(),
                        _ => None,
                    };
                    match paused {
                        Some(result) => results.push(result),
                        None => errors.push((index, error)),
                    }
                }
                Err(error) => errors.push((
                    index,
                    WorkflowError::Channel(format!("Parallel agent join error: {error}")),
                )),
            }
        }
        ctx.run_control.check()?;
        // The failure that stopped the others, else the first in input
        // order.
        let first = trigger
            .get()
            .and_then(|trigger| errors.iter().position(|(index, _)| index == trigger))
            .unwrap_or(0);
        if errors.is_empty() {
            Ok(results)
        } else {
            Err(errors.swap_remove(first).1)
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
        // The caller chooses who drives it: `AgentHandle::wait` hands it to
        // the workflow.
        let options = StartOptions {
            driven: false,
            ..StartOptions::default()
        };
        let run = self.start_agent(agent_name, prompt, ctx, options)?;
        Ok(AgentHandle::new(run, ctx.clone(), self.persistence.clone()))
    }

    pub async fn continue_agent(
        &self,
        previous: &AgentResult,
        continuation_prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        // A completion a verifier could not review (`VerifierFailed`): the
        // prompt is the agent's next message.
        if previous.status == AgentRunStatus::Paused
            && matches!(
                previous.task.current_state,
                TaskState::AwaitingErrorRecoveryApproval {
                    cause: arko_agents::ErrorRecoveryCause::Middleware,
                    ..
                }
            )
        {
            return self
                .resume_agent(
                    previous,
                    AgentResume::ContinueAfterError {
                        note: Some(continuation_prompt.to_string()),
                    },
                    ctx,
                )
                .await;
        }
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
                "only a completed or user-input-suspended agent run, or one a completion verifier could not review, can be continued; resume a paused one with resume_agent".to_string(),
            ));
        }
        let agent_name = &previous.agent_name;
        self.admit(agent_name, Some(previous.agent_run_id), ctx)?;
        let mut task = previous.task.clone();
        let mut registry = self.build_tool_registry(agent_name, ctx, depth);
        let skill_prompt_context =
            Self::setup_skills(&mut task.agent_config, &mut registry, &ctx.workspace)?;
        // What the task holds, recorded again for a task saved before
        // messages recorded who holds them.
        ctx.messages.settle_run(
            agent_name,
            task.id,
            &task.conversation,
            &[],
            &task.pending_context_injections,
            false,
        );
        // No roster: the task's initial prompt has one. Start notices tell it
        // about recipients that started since it last ran.
        let (messages, pending_message_ids) = self.pending_messages(agent_name, task.id, ctx);
        // The prompt presents the messages the task doesn't hold. A task
        // saved before holding was recorded may still have taken one it
        // stopped before adding (`Task::pending_context_injections`): it
        // doesn't get it twice.
        task.pending_context_injections.retain(|content| {
            !pending_message_ids
                .iter()
                .any(|id| crate::messaging::is_injection_of(content, *id))
        });
        // The resources it read that changed since, when freshness is on.
        // It counts as finished (no notices) until it is registered below.
        let actor = ActorId::new(task.id, agent_name.as_str());
        let changed = ctx
            .resource_tracker
            .as_ref()
            .map(|tracker| tracker.changed_since_read(&actor))
            .unwrap_or_default();
        let changed_notice = ctx
            .resource_tracker
            .as_ref()
            .and_then(|tracker| tracker.changed_notice(&changed))
            .map_or_else(String::new, |notice| format!("\n\n{notice}"));
        let prompt = format!("{continuation_prompt}{messages}{changed_notice}");
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
            true,
        );
        // Now registered, it is notified of writes; one made while it
        // started is told now.
        if let Some(tracker) = &ctx.resource_tracker {
            let since = tracker
                .resume_actor(&actor)
                .into_iter()
                .filter(|resource| !changed.contains(resource))
                .collect::<Vec<_>>();
            if let Some(notice) = tracker.changed_notice(&since) {
                ctx.messages.inject_task(run.agent_run_id, notice).await;
            }
        }
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// Continue a paused agent run (see [`Workflow::resume_agent`](super::Workflow::resume_agent)).
    pub async fn resume_agent(
        &self,
        previous: &AgentResult,
        resume: AgentResume,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        let pause = (previous.status == AgentRunStatus::Paused)
            .then(|| AgentPause::of(&previous.task.current_state))
            .flatten()
            .ok_or_else(|| {
                WorkflowError::BuilderInvalid(
                    "only a paused agent run can be resumed; continue a completed or input-waiting run with continue_agent".to_string(),
                )
            })?;
        enum Decision {
            Tools(Option<uuid::Uuid>, Vec<(uuid::Uuid, ToolApprovalDecision)>),
            Command(UserCommand),
        }
        let decision = match (&pause, resume) {
            (AgentPause::ToolApproval { calls }, AgentResume::ToolApprovals(mut decided)) => {
                if let Some(unknown) = decided
                    .keys()
                    .find(|id| !calls.iter().any(|call| call.tool_call_id == **id))
                {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "tool call {unknown} is not awaiting approval"
                    )));
                }
                let decisions = calls
                    .iter()
                    .map(|call| {
                        (
                            call.tool_call_id,
                            decided
                                .remove(&call.tool_call_id)
                                .unwrap_or(ToolApprovalDecision::Reject { reason: None }),
                        )
                    })
                    .collect::<Vec<_>>();
                let batch_id = match &previous.task.current_state {
                    TaskState::AwaitingBatchToolApproval { batch_id, .. } => Some(*batch_id),
                    _ => None,
                };
                Decision::Tools(batch_id, decisions)
            }
            (AgentPause::ErrorRecovery { .. }, AgentResume::ContinueAfterError { note }) => {
                Decision::Command(UserCommand::ContinueAfterError { note })
            }
            (AgentPause::ErrorRecovery { .. }, AgentResume::FailAfterError { reason }) => {
                Decision::Command(UserCommand::RejectErrorRecovery { message: reason })
            }
            (AgentPause::ErrorRecovery { .. }, AgentResume::ResolveRefusal(changes)) => {
                Decision::Command(UserCommand::ResolveRefusal { changes })
            }
            (pause, resume) => {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "{resume:?} doesn't answer an agent {pause}"
                )));
            }
        };
        let agent_name = &previous.agent_name;
        self.admit(agent_name, Some(previous.agent_run_id), ctx)?;
        let mut task = previous.task.clone();
        // What the task holds, recorded again for a task saved before
        // messages recorded who holds them.
        ctx.messages.settle_run(
            agent_name,
            task.id,
            &task.conversation,
            &[],
            &task.pending_context_injections,
            false,
        );
        let mut registry = self.build_tool_registry(agent_name, ctx, 0);
        let skill_prompt_context =
            Self::setup_skills(&mut task.agent_config, &mut registry, &ctx.workspace)?;
        ctx.registry.start(&[(task.id, agent_name.as_str())]);
        // Resumed without a prompt: the messages it doesn't hold arrive as
        // context updates.
        let run = self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            Vec::new(),
            ctx,
            0,
            true,
        );
        // Messages queued while it was paused, and what changed since it
        // read it, go ahead of the decision, so they are added at the turn
        // boundary the decision leads to.
        tokio::select! {
            biased;
            _ = ctx.run_control.stopped() => {}
            _ = ctx.messages.forwarded(agent_name, run.agent_run_id) => {}
        }
        if let Some(notice) = ctx.resource_tracker.as_ref().and_then(|tracker| {
            tracker.changed_notice(
                &tracker.resume_actor(&ActorId::new(run.agent_run_id, agent_name.as_str())),
            )
        }) {
            let _ = run
                .cmd_tx
                .send(UserCommand::InjectContext { content: notice })
                .await;
        }
        match decision {
            Decision::Tools(batch_id, decisions) => {
                if let AgentPause::ToolApproval { calls } = &pause {
                    record_approval_reasons(ctx, previous.agent_run_id, calls, &decisions);
                }
                send_tool_decisions(&run.cmd_tx, batch_id, decisions).await
            }
            Decision::Command(command) => {
                let _ = run.cmd_tx.send(command).await;
            }
        }
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

    /// Answers a review that failed (the verifier's run failed, or its
    /// verdict stayed unusable): within the verifier's `max_review_retries`,
    /// the completion is rejected so the agent calls `complete_task` again
    /// and a new review runs; after that, the completion is neither accepted
    /// nor rejected: the task is paused for its caller (continuable with
    /// `continue_agent`) and the run call returns `VerifierFailed`. The
    /// count starts again after the pause.
    fn review_failed(
        &self,
        execution: &ToolExecutionContext<'_>,
        verifier: &CompletionVerifierConfig,
        reason: String,
    ) -> ToolExecutionDecision {
        let failures = {
            let mut failures = self.context.verification_failures.lock();
            let count = failures
                .entry((execution.task_id, verifier.name.clone()))
                .or_default();
            *count = count.saturating_add(1);
            *count
        };
        log::warn!(
            "Workflow '{}': completion verifier '{}' could not review agent '{}' (failure {failures}): {reason}",
            self.context.workflow_name,
            verifier.name,
            self.agent_name
        );
        if failures <= verifier.max_review_retries {
            return ToolExecutionDecision::Reject {
                message_to_agent: format!(
                    "Completion verifier '{}' could not review this completion ({reason}). Nothing was decided about your work: call complete_task again to have it reviewed again.",
                    verifier.name
                ),
            };
        }
        self.context
            .verification_failures
            .lock()
            .remove(&(execution.task_id, verifier.name.clone()));
        self.context
            .verifier_failures
            .lock()
            .insert(execution.task_id, (verifier.name.clone(), reason.clone()));
        ToolExecutionDecision::PauseTask {
            message_to_agent: format!(
                "Completion verifier '{}' could not review this completion ({reason}). Nothing was decided about your work; the task is paused until the caller decides how to go on.",
                verifier.name
            ),
            reason: format!(
                "Completion verifier '{}' could not review the completion: {reason}",
                verifier.name
            ),
        }
    }

    async fn verify_completion(
        &self,
        execution: &ToolExecutionContext<'_>,
        summary: &str,
    ) -> Option<ToolExecutionDecision> {
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
            // The latest verdict on this task by this verifier, if any.
            let prior_record = self
                .context
                .verification_records
                .read()
                .iter()
                .rev()
                .find(|record| {
                    record.target_task_id == execution.task_id
                        && record.verifier_name == verifier.name
                })
                .cloned();
            let prior_decision = |record: &VerificationRecord| match record.decision {
                VerificationDecision::Approved => "ACCEPTED",
                VerificationDecision::RejectedMinor => "REJECTED_MINOR",
                VerificationDecision::RejectedMajor => "REJECTED_MAJOR",
            };
            let prior_result = prior_record
                .as_ref()
                .filter(|_| verifier.reuse_conversation)
                .and_then(|record| {
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
                    prior_decision(record),
                    record.feedback,
                    summary,
                    repair_delta,
                    verifier.instructions,
                    crate::verification::VERIFIER_OUTPUT_CONTRACT,
                );
                match engine
                    .continue_nested(previous, &prompt, &self.context, self.depth + 1)
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Some(self.review_failed(
                            execution,
                            verifier,
                            format!("its review could not continue: {error}"),
                        ));
                    }
                }
            } else {
                // A new review, given the prior findings on this task
                // unless the verifier opts out.
                let pass = match prior_record
                    .as_ref()
                    .filter(|_| verifier.include_prior_findings)
                {
                    Some(record) => format!(
                        "This is a re-review. An earlier review of this task returned:\n{}\n{}\n\nThe agent has since revised its work. Check whether each prior finding is resolved, then inspect the actual workspace and relevant artifacts for any other material issue. Do not approve merely because the summary claims success.",
                        prior_decision(record),
                        record.feedback,
                    ),
                    None => "This is the first review pass. Inspect the actual workspace and relevant artifacts, systematically cover the applicable requirement and failure classes, and consolidate every evidenced related finding into this verdict rather than drip-feeding one counterexample per repair cycle. Do not approve merely because the summary claims success.".to_string(),
                };
                let prompt = format!(
                    "You are a completion verifier for agent '{}'.\n\nOriginal task:\n{}\n\nCandidate completion summary:\n{}\n\nVerification instructions:\n{}\n\n{pass} Batch independent read-only tool calls when useful, but call non-batch-safe tools alone. Finish by calling complete_task using this exact contract:\n\n{}",
                    self.agent_name,
                    without_roster(execution.initial_task_prompt),
                    summary,
                    verifier.instructions,
                    crate::verification::VERIFIER_OUTPUT_CONTRACT,
                );
                match engine
                    .run_verifier_nested(
                        &verifier.verifier_agent,
                        &prompt,
                        &self.context,
                        self.depth + 1,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Some(self.review_failed(
                            execution,
                            verifier,
                            format!("its review did not complete: {error}"),
                        ));
                    }
                }
            };

            let mut format_attempt = 0;
            let response = loop {
                match crate::verification::parse_response(&result.summary) {
                    Ok(response) => break response,
                    Err(error) if format_attempt < verifier.max_format_retries => {
                        format_attempt += 1;
                        let prompt = format!(
                            "Your verification conclusion could not be parsed ({error}). Do not repeat the review or change your conclusion. Re-emit it through complete_task using only this contract:\n\n{}",
                            crate::verification::VERIFIER_OUTPUT_CONTRACT,
                        );
                        result = match engine
                            .continue_nested(&result, &prompt, &self.context, self.depth + 1)
                            .await
                        {
                            Ok(result) => result,
                            Err(retry_error) => {
                                return Some(self.review_failed(
                                    execution,
                                    verifier,
                                    format!("it could not restate its verdict: {retry_error}"),
                                ));
                            }
                        };
                    }
                    Err(error) => {
                        return Some(self.review_failed(
                            execution,
                            verifier,
                            format!(
                                "its verdict was unusable after {} attempts: {error}",
                                format_attempt + 1
                            ),
                        ));
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

impl WorkflowExecutionMiddleware {
    /// What `before` decides for a call.
    async fn gate(
        &self,
        execution: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        if let Some(decision) = self.stop_decision() {
            return Ok(decision);
        }
        let approval_reason = self
            .context
            .approval_reasons
            .lock()
            .get(&call.tool_call_id)
            .map(|(_, reason)| reason.clone());
        match fire_hooks(&self.context, &HookPoint::PreToolExecution, |hook| {
            hook.agent_name = Some(self.agent_name.clone());
            hook.agent_run_id = Some(execution.task_id);
            hook.tool_name = Some(call.tool_name.to_string());
            hook.tool_call_id = Some(call.tool_call_id);
            hook.tool_params = Some(call.parameters.clone());
            hook.approval_reason = approval_reason;
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

        if call.tool_name == COMPLETE_TASK_TOOL_NAME {
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
}

#[async_trait]
impl ToolExecutionMiddleware for WorkflowExecutionMiddleware {
    async fn before(
        &self,
        execution: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        let decision = self.gate(execution, call).await;
        // A call that won't run needs no approval reason for its hooks.
        if !matches!(decision, Ok(ToolExecutionDecision::Continue)) {
            self.context
                .approval_reasons
                .lock()
                .remove(&call.tool_call_id);
        }
        decision
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

        if successful_output.is_some() && call.tool_name == COMPLETE_TASK_TOOL_NAME {
            self.context
                .verification_baselines
                .lock()
                .retain(|(task_id, _), _| *task_id != execution.task_id);
            self.context
                .verification_attempts
                .lock()
                .retain(|(task_id, _), _| *task_id != execution.task_id);
            self.context
                .verification_failures
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

        // The call has run: its approval reason is no longer needed.
        let approval_reason = self
            .context
            .approval_reasons
            .lock()
            .remove(&call.tool_call_id)
            .map(|(_, reason)| reason);
        let hook_result = fire_hooks(&self.context, &HookPoint::PostToolExecution, |hook| {
            hook.agent_name = Some(self.agent_name.clone());
            hook.agent_run_id = Some(execution.task_id);
            hook.tool_name = Some(call.tool_name.to_string());
            hook.tool_call_id = Some(call.tool_call_id);
            hook.tool_params = Some(call.parameters.clone());
            hook.approval_reason = approval_reason;
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
                None,
                false,
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
            let result =
                finalize_agent_task("designer".into(), task, &[], workflow.context(), None)
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
            load_skill_interception: None,
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
                        .target_agent(target)
                        .target_tool("search"),
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
                .build_interceptor_config("reviewer", None, workflow.context())
                .is_none()
        );
        let interceptor = workflow
            .engine
            .build_interceptor_config("coder", None, workflow.context())
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

    /// Under a retrying failure policy, a refused or truncated summary is not
    /// requested (and billed) again; an empty one is. The chain retries the
    /// distiller alone, not the interceptor before it, and keeps the tool
    /// output once the retries run out.
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
        let policy =
            arko_agents::InterceptionFailurePolicy::RetryThenUseRawResult { max_attempts: 3 };
        for (reply, requests) in [
            (anthropic_reply("The failing test is in", "max_tokens"), 1),
            (anthropic_reply("The failing test", "refusal"), 1),
            (anthropic_reply("", "end_turn"), 3),
        ] {
            let server = MockServer::start(reply).await;
            let first = Arc::new(Counting::default());
            let workflow = Workflow::builder()
                .name("distillation-retry-test")
                .workspace(std::env::temp_dir())
                .llm_client(Arc::new(server.anthropic_client()))
                .agent("coder", config.clone())
                .ephemeral()
                // A second interceptor makes the engine chain them.
                .tool_result_interceptor(first.clone())
                .distiller(
                    DistillationConfig::new("bulk", Model::anthropic_claude_opus_5_5())
                        .min_chars(10)
                        .target_agent("coder")
                        .target_tool("search"),
                )
                .interception_failure_policy(policy.clone())
                .build()
                .unwrap();
            let interceptor = workflow
                .engine
                .build_interceptor_config("coder", None, workflow.context())
                .unwrap()
                .interceptor;
            let task = TaskContext {
                task_id: uuid::Uuid::new_v4(),
                agent_name: "coder",
                initial_task_prompt: "find the failing test",
                current_working_directory: &workflow.context().workspace,
            };
            let outcome = interceptor
                .intercept(
                    "search",
                    &"raw tool output ".repeat(10),
                    &HashMap::new(),
                    &task,
                )
                .await
                .expect("the chain applies the policy itself and never fails");
            assert!(matches!(outcome, InterceptedResult::Unchanged));
            assert_eq!(server.requests(), requests);
            assert_eq!(
                workflow.context().distillation_records.read().len(),
                requests
            );
            assert_eq!(first.calls(), 1, "the interceptor before it ran once");
        }
    }

    /// A reader whose task finished is not notified of a write (no notice,
    /// no conflict hook); continuing it sends one message listing what
    /// changed since it read it, advising a reread before editing.
    #[tokio::test]
    async fn a_continued_reader_learns_what_changed_since_it_read() {
        use crate::freshness::{ContextFreshnessConfig, ResourceId};
        use crate::test_support::{
            ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
        };
        let workspace = Workspace::new();
        std::fs::write(workspace.0.join("a.txt"), "first").unwrap();
        let server = ResponsesServer::start(|index, _| {
            Some(match index {
                0 => tool_call_turn("r", "read_file", serde_json::json!({"path": "a.txt"})),
                _ => complete_turn("done", "done"),
            })
        })
        .await;
        let mut reader = agent_config("Friendly Reader");
        reader
            .tool_config
            .available_tool_names
            .push("read_file".into());
        let conflicts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let workflow = Workflow::builder()
            .name("freshness-continue-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("reader", reader)
            .context_freshness(ContextFreshnessConfig::default())
            .hook(HookPoint::OnFreshnessConflict, {
                let conflicts = conflicts.clone();
                Arc::new(move |_| {
                    conflicts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    HookResult::Continue
                })
            })
            .build()
            .unwrap();
        let first = workflow.run_agent("reader", "Read a.txt.").await.unwrap();
        let file =
            arko_agents::utils::resolve_path("a.txt", &workflow.context().workspace).unwrap();
        let stale = workflow
            .mark_resource_dirty(
                ActorId::new(uuid::Uuid::new_v4(), "writer"),
                ResourceId::File(file),
            )
            .await
            .unwrap();
        assert!(stale.is_empty(), "a finished reader is not notified");
        assert_eq!(conflicts.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(workflow.context().messages.pending("reader").is_empty());

        workflow.continue_agent(&first, "Carry on.").await.unwrap();
        let last = server.bodies().last().unwrap().clone();
        assert!(last.contains("Carry on."), "{last}");
        assert!(last.contains("Changed since you last read them"), "{last}");
        assert!(last.contains("- file `a.txt`"), "{last}");
        assert!(
            last.contains("Re-read any of these before you edit it"),
            "{last}"
        );
        // Told once: the first request of the continuation only.
        assert_eq!(server.requests(), 3);
    }

    /// Counts its calls; replaces the result with `replacement`, if any.
    #[derive(Debug, Default)]
    struct Counting {
        calls: std::sync::atomic::AtomicUsize,
        replacement: Option<&'static str>,
    }

    impl Counting {
        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ToolResultInterceptor for Counting {
        async fn intercept(
            &self,
            _tool_name: &str,
            _raw_result: &str,
            _params: &HashMap<String, Value>,
            _task_context: &TaskContext<'_>,
        ) -> Result<InterceptedResult, ToolError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(match self.replacement {
                Some(text) => InterceptedResult::Replaced(text.to_string()),
                None => InterceptedResult::Unchanged,
            })
        }
    }

    /// Always fails; counts its calls.
    #[derive(Debug, Default)]
    struct Failing(std::sync::atomic::AtomicUsize);

    #[async_trait]
    impl ToolResultInterceptor for Failing {
        async fn intercept(
            &self,
            _tool_name: &str,
            _raw_result: &str,
            _params: &HashMap<String, Value>,
            _task_context: &TaskContext<'_>,
        ) -> Result<InterceptedResult, ToolError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(ToolError::ExecutionFailed("summarizer offline".to_string()))
        }
    }

    /// A failing stage in a chain gets the agent's failure policy (its
    /// retries, then its note) and is skipped: the agent gets the output of
    /// the stage before it, so a redaction stays in place. (The chain once
    /// failed as a whole and the runtime delivered the raw result, undoing
    /// the redaction.)
    #[tokio::test]
    async fn a_failing_stage_keeps_the_earlier_stages_output_such_as_a_redaction() {
        use crate::test_support::{
            ProbeTool, ResponsesServer, Workspace, agent_config, complete_turn, tool_call_turn,
        };
        let workspace = Workspace::new();
        let server = ResponsesServer::start(|index, _| {
            Some(match index {
                0 => tool_call_turn("1", "probe", serde_json::json!({})),
                _ => complete_turn("2", "done"),
            })
        })
        .await;
        let redact = Arc::new(Counting {
            replacement: Some("[redacted]"),
            ..Counting::default()
        });
        let failing = Arc::new(Failing::default());
        let after = Arc::new(Counting::default());
        let workflow = Workflow::builder()
            .name("chain-failure-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("coder", agent_config("Coder"))
            .global_tool(ProbeTool::new("probe", false))
            .tool_result_interceptor(redact.clone())
            .tool_result_interceptor(failing.clone())
            .tool_result_interceptor(after.clone())
            .interception_failure_policy(
                arko_agents::InterceptionFailurePolicy::RetryThenInformAgent { max_attempts: 3 },
            )
            .build()
            .unwrap();
        let result = workflow.run_agent("coder", "Probe.").await.unwrap();
        assert_eq!(result.summary, "done");
        assert_eq!(redact.calls(), 1, "a stage that succeeded runs once");
        assert_eq!(failing.0.load(std::sync::atomic::Ordering::SeqCst), 3);
        assert_eq!(after.calls(), 1, "the stages after a skipped one run");
        let second = &server.bodies()[1];
        assert!(
            second.contains(
                "[redacted]\\n\\n[A tool-result interceptor failed (Execution failed: summarizer offline), so this result skipped that step.]"
            ),
            "{second}"
        );
        assert!(
            !second.contains("probed"),
            "the raw result never reaches the agent"
        );
    }

    /// `load_skill` results get the skills configuration's failure policy,
    /// other tools the workflow's.
    #[tokio::test]
    async fn a_chain_applies_the_load_skill_policy_to_load_skill_results() {
        let failing = Arc::new(Failing::default());
        let chain = ChainedInterceptor::new(
            vec![Arc::new(Counting::default()), failing.clone()],
            InterceptionFailurePolicy::UseRawResult,
            Some(InterceptionFailurePolicy::RetryThenInformAgent { max_attempts: 2 }),
        );
        let workspace = std::env::temp_dir();
        let task = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "coder",
            initial_task_prompt: "task",
            current_working_directory: &workspace,
        };
        let search = chain
            .intercept("search", "output", &HashMap::new(), &task)
            .await
            .unwrap();
        assert!(matches!(search, InterceptedResult::Unchanged));
        assert_eq!(failing.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        let skill = chain
            .intercept(
                arko_agents::LOAD_SKILL_TOOL_NAME,
                "skill text",
                &HashMap::new(),
                &task,
            )
            .await
            .unwrap();
        assert!(matches!(
            skill,
            InterceptedResult::Replaced(ref text) if text.starts_with("skill text\n\n[This is the full result")
        ));
        assert_eq!(failing.0.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
