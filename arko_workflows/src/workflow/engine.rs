use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::handover::{HandoverHook, HandoverSeed};
use arko_agents::skill::{
    LOAD_SKILL_TOOL_NAME, LoadSkillTool, SkillManager, SkillPromptContext, SkillsConfig,
};
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

use crate::budget::BudgetExceededPolicy;
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
use crate::workflow::cancel::{CancelCause, Cancellation};
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
                    "Task {}: skipping tool-result interceptor {} of {} for '{}'. The next gets the output so far",
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
                    "{current}\n\n[This is the full result. It was to be shortened, but the tool-result interceptor failed ({errors}).]"
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
    // During a graceful stop's wrap-up the abort ends the wrap-up, and the
    // stop, which came first, is made.
    if ctx.run_control.stop(cause)
        && matches!(
            ctx.run_control.cancellation(),
            Some(Cancellation {
                cause: CancelCause::Stopped { .. },
                ..
            })
        )
    {
        fire_stop_hook(ctx);
    }
    ctx.run_control.error().unwrap_or(error)
}

/// Fires `OnWorkflowCancelled` for the run's stop, ignoring its result. For
/// a [`CancelCause::Stopped`] stop the hook's metadata has `stop_kind`
/// (`operator`, `drain`, `limit` or `failure`) and `reason`.
pub(crate) fn fire_stop_hook(ctx: &SharedContext) {
    let cancellation = ctx.run_control.cancellation();
    let _ = fire_hooks(ctx, &HookPoint::OnWorkflowCancelled, |hook| {
        if let Some(Cancellation {
            cause: CancelCause::Stopped { kind, reason },
            ..
        }) = &cancellation
        {
            hook.metadata
                .insert("stop_kind".to_string(), serde_json::json!(kind.as_str()));
            hook.metadata
                .insert("reason".to_string(), serde_json::json!(reason));
        }
    });
}

/// Unregisters agent run `agent_run_id` from the run's stop and, when it was
/// the last running agent of a graceful stop's wrap-up, stops the run.
fn unregister_run(ctx: &SharedContext, agent_run_id: uuid::Uuid, launch: u64) {
    ctx.run_control.unregister(agent_run_id, launch);
    if ctx.run_control.end_wrap_up_if_idle() {
        fire_stop_hook(ctx);
    }
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
    let event = crate::workflow::result::WorkflowEvent {
        timestamp: chrono::Utc::now(),
        workflow_run_id: ctx.workflow_run_id,
        agent_name: agent_name.to_string(),
        agent_run_id,
        event: event.clone(),
    };
    ctx.event_hub.publish(&event);
    let _ = ctx.event_tx.send(event.clone());
    let event = &event.event;
    if let Some(persistence) = persistence {
        // Never fails: a failed write is logged and retried.
        persistence
            .append_event(agent_name, agent_run_id, event)
            .await;
    }
}

/// Charges usage agent run `agent_run_id` reported to the budget and fires
/// the budget hooks. With the task journal, the run's count of charged
/// usage goes up with the charge (see [`PersistenceManager::charge_usage`]).
fn charge_usage(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    usage: &arko_ai::UsageInfo,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> crate::budget::AddCostResult {
    // Subscription (ChatGPT OAuth) calls are recorded with a billed amount
    // of zero.
    let charge = || ctx.budget.add_agent_cost(agent_name, usage);
    let cost_result = match persistence {
        Some(persistence) => persistence.charge_usage(agent_run_id, charge),
        None => charge(),
    };
    if cost_result.warning_triggered {
        let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetWarning, |hctx| {
            hctx.budget_spent = cost_result.new_total;
        });
    }
    if reports_exceeded(ctx, &cost_result) {
        let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetExceeded, |hctx| {
            hctx.budget_spent = cost_result.new_total;
        });
    }
    cost_result
}

/// Whether a charge fires `OnBudgetExceeded`: under
/// [`BudgetExceededPolicy::Pause`] once per crossing of the limit, under
/// `Cancel` for every charge over it.
pub(crate) fn reports_exceeded(
    ctx: &SharedContext,
    cost_result: &crate::budget::AddCostResult,
) -> bool {
    match ctx.budget.exceeded_policy() {
        BudgetExceededPolicy::Pause => cost_result.exceeded_crossed,
        BudgetExceededPolicy::Cancel => cost_result.exceeded,
    }
}

/// The error for a request agent `agent_name` may not make because a budget
/// is exceeded (the workflow's, the agent's, or a plan window's), whatever
/// the policy: for a request no run waits on, such as a prompt cache
/// refresh.
fn budget_refusal(ctx: &SharedContext, agent_name: &str) -> Option<WorkflowError> {
    let budget = &ctx.budget;
    if budget.is_exceeded() {
        return Some(WorkflowError::BudgetExceeded {
            spent: budget.total_spent(),
            limit: budget.limit().unwrap_or(0.0),
        });
    }
    if budget.is_agent_exceeded(agent_name) {
        return Some(WorkflowError::AgentBudgetExceeded {
            agent: agent_name.to_string(),
            spent: budget.agent_spent(agent_name),
            limit: budget.agent_limit(agent_name).unwrap_or(0.0),
        });
    }
    budget
        .plan_exceeded()
        .map(|crossing| WorkflowError::PlanBudgetExceeded {
            plan: crossing.plan.map(|plan| plan.label().to_string()),
            window: crossing.window.label().to_string(),
            used: crossing.used,
            limit: crossing.limit,
        })
}

/// Whether the budget cancels agents over it.
fn budget_cancels(ctx: &SharedContext) -> bool {
    ctx.budget.exceeded_policy() == BudgetExceededPolicy::Cancel
}

/// Under [`BudgetExceededPolicy::Cancel`], the error for a plan budget
/// window over its limit.
fn plan_budget_error(ctx: &SharedContext) -> Option<WorkflowError> {
    if !budget_cancels(ctx) {
        return None;
    }
    ctx.budget
        .plan_exceeded()
        .map(|crossing| WorkflowError::PlanBudgetExceeded {
            plan: crossing.plan.map(|plan| plan.label().to_string()),
            window: crossing.window.label().to_string(),
            used: crossing.used,
            limit: crossing.limit,
        })
}

/// Answers an agent held before a model request because a budget is
/// exceeded ([`ErrorRecoveryCause::BudgetExceeded`](arko_agents::ErrorRecoveryCause::BudgetExceeded)):
/// once the budget is raised, it continues. A run inside another agent's
/// tool call (`nested`) waits for the raise, and so does one the host
/// started under `WorkflowBuilder::wait_for_budget_raise`. Any other is
/// paused for its caller ([`WorkflowError::AgentPaused`]). A stop of the
/// run ends the wait. Not an automatic recovery: `ErrorRecoveryPolicy`
/// doesn't count it.
async fn answer_budget_hold(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    error: String,
    nested: bool,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &Arc<SharedContext>,
) -> Result<(), WorkflowError> {
    let mut changes = ctx.budget.subscribe();
    if ctx.budget.hold_reason(agent_name).is_some() {
        if !nested && !ctx.budget.waits_for_raise() {
            log::info!(
                "Workflow '{}': agent '{agent_name}' paused: {error}",
                ctx.workflow_name
            );
            return pause_for_caller(
                agent_name,
                agent_run_id,
                cmd_tx,
                AgentPause::ErrorRecovery {
                    error,
                    cause: arko_agents::ErrorRecoveryCause::BudgetExceeded,
                    error_kind: None,
                },
            )
            .await;
        }
        ctx.registry
            .set_waiting(agent_run_id, WaitReason::BudgetExceeded);
        log::info!(
            "Workflow '{}': agent '{agent_name}' waits for the budget to be raised: {error}",
            ctx.workflow_name
        );
        while ctx.budget.hold_reason(agent_name).is_some() {
            tokio::select! {
                biased;
                _ = ctx.run_control.stopped() => return Err(stop_error(ctx)),
                changed = changes.changed() => {
                    if changed.is_err() {
                        return Err(stop_error(ctx));
                    }
                }
            }
        }
    }
    wait_while_paused(ctx).await?;
    let _ = cmd_tx
        .send(UserCommand::ContinueAfterError { note: None })
        .await;
    Ok(())
}

/// The error-recovery state of one drive of an agent.
#[derive(Default)]
struct RecoveryState {
    /// The cause of the error-recovery pause the agent is in (its latest
    /// state, or the state it was resumed in), if it is in one. An error it
    /// reports needing a decision is about that pause.
    paused: Option<arko_agents::ErrorRecoveryCause>,
    /// That pause's error kind, when it recorded one.
    paused_kind: Option<arko_ai::ErrorKind>,
    /// An error needing a decision that arrived before the state change
    /// giving its cause, answered once that pause arrives, so the decision
    /// doesn't depend on the order a runtime reports them in.
    unanswered: Option<String>,
    /// Automatic recoveries in this drive (for `ErrorRecoveryPolicy::Limited`).
    recovered: u32,
    /// Pauses in a row without a model response in between.
    repeat: u32,
    /// Failed model requests in a row without a model response in between
    /// (for `SubagentsConfig::max_failed_requests`). The agent's own retries
    /// count, and so does the request that ended a round of them.
    failed_requests: u32,
    /// The error of the latest failed request the agent retried.
    last_error: String,
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
/// for approval that `ApproveAll` approves. `None` for its later ones.
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
        "Stopped because agent '{agent_name}' runs inside another agent's tool call and can't be paused, but it needs {needs}. Configure it to auto-approve the tools it uses, or decide such calls with ApprovalPolicy::Handler."
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
/// approved. Any other call is left undecided (`None`), since no one can
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

/// Why subagent `agent_run_id` fails when `failed` model requests have failed
/// in a row, if that reaches its cap
/// ([`SubagentsConfig::max_failed_requests`](crate::subagents::SubagentsConfig::max_failed_requests)).
fn failed_requests_cap_reached(
    ctx: &SharedContext,
    agent_run_id: uuid::Uuid,
    failed: u32,
    error: &str,
) -> Option<String> {
    let cap = ctx.subagents.max_failed_requests()?;
    (failed >= cap && ctx.subagents.is_subagent(agent_run_id)).then(|| {
        let requests = if failed == 1 { "request" } else { "requests" };
        format!("It stopped after {failed} failed {requests} in a row. The last error was {error}")
    })
}

/// Counts a failed request that the agent retries or waits out on its own. A
/// subagent over its cap is cancelled (it is waiting and would never pause)
/// and fails with the cap's reason.
fn count_failed_request(
    recovery: &mut RecoveryState,
    agent_run_id: uuid::Uuid,
    error: &str,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &SharedContext,
) -> Result<(), WorkflowError> {
    recovery.failed_requests = recovery.failed_requests.saturating_add(1);
    match failed_requests_cap_reached(ctx, agent_run_id, recovery.failed_requests, error) {
        Some(reason) => {
            let _ = cmd_tx.try_send(UserCommand::CancelTask);
            Err(WorkflowError::Agent(
                arko_agents::ArkoAgentsError::TaskExecution(reason),
            ))
        }
        None => Ok(()),
    }
}

/// Answers `error`, reported at an error-recovery pause, under the
/// workflow's error recovery policy and the pause's cause: continues (after
/// a wait), fails the task, or pauses the agent for its caller. A `nested`
/// run (inside another agent's tool call) can't pause, so it fails instead.
async fn answer_recovery_pause(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    error: String,
    recovery: &mut RecoveryState,
    nested: bool,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &Arc<SharedContext>,
) -> Result<(), WorkflowError> {
    let cause = recovery
        .paused
        .clone()
        .unwrap_or(arko_agents::ErrorRecoveryCause::Unknown);
    if cause == arko_agents::ErrorRecoveryCause::BudgetExceeded {
        return answer_budget_hold(agent_name, agent_run_id, error, nested, cmd_tx, ctx).await;
    }
    recovery.repeat = recovery.repeat.saturating_add(1);
    // The request that ended this round of the agent's retries failed too.
    let request_failed = matches!(
        cause,
        arko_agents::ErrorRecoveryCause::RetriesExhausted { .. }
            | arko_agents::ErrorRecoveryCause::UsageLimit { .. }
    );
    if request_failed {
        recovery.failed_requests = recovery.failed_requests.saturating_add(1);
    }
    let now_unix_ms = u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0);
    let mut action = ctx.error_recovery_policy.action(
        &cause,
        &error,
        recovery.recovered,
        recovery.repeat,
        &ctx.recovery_waits,
        now_unix_ms,
    );
    // A subagent's own cap on failed requests in a row.
    if request_failed
        && matches!(action, RecoveryAction::ContinueAfter(_))
        && let Some(reason) =
            failed_requests_cap_reached(ctx, agent_run_id, recovery.failed_requests, &error)
    {
        action = RecoveryAction::Fail(reason);
    }
    match action {
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
                    "Workflow '{}': agent '{agent_name}' paused ({}). Continuing in {wait:?}: {error}",
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
                AgentPause::ErrorRecovery {
                    error,
                    cause,
                    error_kind: recovery.paused_kind,
                },
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
/// Returns when the agent ends. With [`WorkflowError::UserInputRequired`] or
/// [`WorkflowError::AgentPaused`] once it is suspended for its caller. Or
/// with another error when the workflow must stop it (including the stop's
/// error once the run is cancelled or aborted), and the caller then stops
/// the agent.
///
/// A `batch_stop` (fail-fast in `run_agents_parallel`) that is set stops the
/// agent too: the loop returns `AgentCancelled` with its reason. So does the
/// run's own stop ([`Workflow::stop_run`](super::Workflow::stop_run)) once
/// it cancels the run: the loop returns `AgentStopped`.
///
/// `paused` is the cause (and error kind) of the error-recovery pause the
/// agent starts in (a resumed or recovered run): a runtime reports an error
/// about it (such as a refusal change it can't apply) without reporting the
/// pause again.
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
    paused: Option<(arko_agents::ErrorRecoveryCause, Option<arko_ai::ErrorKind>)>,
) -> Result<(), WorkflowError> {
    let drive = Drive {
        agent_name,
        agent_run_id,
        nested,
        ctx,
        persistence,
    };
    let mut recovery = RecoveryState {
        paused_kind: paused.as_ref().and_then(|(_, kind)| *kind),
        paused: paused.map(|(cause, _)| cause),
        ..RecoveryState::default()
    };
    let run_stop = ctx.run_control.run_cancel(agent_run_id);
    tokio::select! {
        biased;
        reason = stop_set(batch_stop) => Err(WorkflowError::AgentCancelled {
            agent: agent_name.to_string(),
            reason,
        }),
        reason = stop_set(run_stop) => Err(WorkflowError::AgentStopped {
            agent: agent_name.to_string(),
            task_id: agent_run_id,
            reason,
        }),
        result = drive_events(&drive, event_rx, cmd_tx, &mut recovery) => result,
    }
}

/// Resolves with the reason once `stop` is set. Never when there is none,
/// or when what could set it is gone (the run drives on, where it was).
async fn stop_set(stop: Option<tokio::sync::watch::Receiver<Option<String>>>) -> String {
    if let Some(mut stop) = stop
        && let Ok(reason) = stop.wait_for(Option::is_some).await
    {
        return reason.clone().unwrap_or_default();
    }
    std::future::pending().await
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
    recovery: &mut RecoveryState,
) -> Result<(), WorkflowError> {
    let Drive {
        agent_name,
        agent_run_id,
        nested,
        ctx,
        persistence,
    } = *drive;
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
        if budget_cancels(ctx)
            && ctx.budget.is_exceeded()
            && !matches!(&event, AgentEvent::UsageReported(_))
        {
            let _ = cmd_tx.try_send(UserCommand::CancelTask);
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        if !matches!(&event, AgentEvent::PlanUsageReported(_))
            && let Some(error) = plan_budget_error(ctx)
        {
            let _ = cmd_tx.try_send(UserCommand::CancelTask);
            return Err(error);
        }
        // Any other state: the agent is past its pause, and an error it
        // reported without one needs no decision.
        if let AgentEvent::TaskStateChanged(state) = &event
            && !matches!(state, TaskState::AwaitingErrorRecoveryApproval { .. })
        {
            recovery.paused = None;
            recovery.paused_kind = None;
            recovery.unanswered = None;
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
                recovery.failed_requests = 0;
                let cost_result = charge_usage(agent_name, agent_run_id, usage, ctx, persistence);
                // Under `Pause`, the agent's next request is held instead
                // (see `answer_budget_hold`).
                if !budget_cancels(ctx) {
                    continue;
                }
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

            AgentEvent::PlanUsageReported(ref usage) => {
                // Under `Pause`, agents are held before their next request
                // (`BudgetManager::hold_reason`).
                for crossing in ctx.budget.record_plan_usage(usage) {
                    let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetExceeded, |hook| {
                        hook.agent_name = Some(agent_name.to_string());
                        hook.agent_run_id = Some(agent_run_id);
                        hook.metadata.insert(
                            "plan".to_string(),
                            crossing.plan.map(|plan| plan.key()).into(),
                        );
                        hook.metadata
                            .insert("plan_window".to_string(), crossing.window.key().into());
                        hook.metadata
                            .insert("plan_points_used".to_string(), crossing.used.into());
                        hook.metadata
                            .insert("plan_points_limit".to_string(), crossing.limit.into());
                    });
                }
                if let Some(error) = plan_budget_error(ctx) {
                    let _ = cmd_tx.try_send(UserCommand::CancelTask);
                    return Err(error);
                }
            }

            // An error-recovery pause is reported as the state change that
            // gives its cause and an error needing a decision; it is answered
            // once both have arrived, whichever comes first (arko's runtimes
            // report the state first; a pause they are still in, also one a
            // run was resumed in, can be reported again with an error alone,
            // such as a refusal change they can't apply). The error's text
            // is what the host is told.
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                ref cause,
                error_kind,
                ..
            }) => {
                recovery.paused = Some(cause.clone());
                recovery.paused_kind = error_kind;
                if let Some(error) = recovery.unanswered.take() {
                    answer_recovery_pause(
                        agent_name,
                        agent_run_id,
                        error,
                        recovery,
                        nested,
                        cmd_tx,
                        ctx,
                    )
                    .await?;
                }
            }

            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: true,
                error_message,
                ..
            } => {
                if recovery.paused.is_some() {
                    answer_recovery_pause(
                        agent_name,
                        agent_run_id,
                        error_message,
                        recovery,
                        nested,
                        cmd_tx,
                        ctx,
                    )
                    .await?;
                } else {
                    recovery.unanswered = Some(error_message);
                }
            }

            AgentEvent::UserQuestionsAsked {
                questions,
                asked,
                wait: true,
            } => {
                answer_questions(drive, questions, asked, cmd_tx).await?;
            }

            AgentEvent::UserQuestionsAsked {
                questions,
                asked,
                wait: false,
            } => ask_later(drive, questions, asked),

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

            // A failed request the agent retries (or waits out) on its own.
            AgentEvent::ErrorEncountered {
                requires_approval_to_retry: false,
                error_message,
                ..
            } => recovery.last_error = error_message,

            AgentEvent::RetryScheduled { .. } => {
                let error = recovery.last_error.clone();
                count_failed_request(recovery, agent_run_id, &error, cmd_tx, ctx)?;
            }

            AgentEvent::UsageLimitReached {
                waiting_until_unix_ms: Some(_),
                error_message,
                ..
            } => count_failed_request(recovery, agent_run_id, &error_message, cmd_tx, ctx)?,

            _ => {}
        }
    }

    Ok(())
}

/// Answers an agent's questions with the workflow's user input handler, or
/// suspends the agent for its caller (`UserInputRequired`). A nested run
/// without a handler is stopped instead.
async fn answer_questions(
    drive: &Drive<'_>,
    questions: String,
    asked: Vec<arko_agents::questions::AskedQuestion>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
) -> Result<(), WorkflowError> {
    let Drive {
        agent_name,
        agent_run_id,
        nested,
        ctx,
        ..
    } = *drive;
    // A subagent asks its lead, which asks the user if it needs to.
    if ctx
        .subagents
        .config()
        .is_some_and(|config| config.questions_to_lead)
        && ctx.subagents.is_subagent(agent_run_id)
    {
        let response = ctx.subagents.ask(ctx, agent_run_id, &questions).await?;
        return cmd_tx
            .send(UserCommand::SubmitUserResponse { response })
            .await
            .map_err(|error| WorkflowError::Channel(error.to_string()));
    }
    if let Some(handler) = &ctx.user_input_handler {
        let request = crate::user_input::UserInputRequest {
            workflow_run_id: ctx.workflow_run_id,
            agent_run_id,
            agent_name: agent_name.to_string(),
            questions: questions.clone(),
            asked,
            wait: true,
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
            .map_err(|error| WorkflowError::Channel(error.to_string()))
    } else if nested {
        Err(nested_cannot_pause(
            ctx,
            agent_name,
            "an answer to its question (no user input handler is set)",
        ))
    } else {
        let _ = cmd_tx.send(UserCommand::SuspendTask).await;
        Err(WorkflowError::UserInputRequired {
            agent: agent_name.to_string(),
            task_id: agent_run_id,
            questions,
        })
    }
}

/// What an agent is told when it asked without waiting and no one can
/// answer: the workflow has no user input handler.
pub(crate) const NO_ONE_TO_ASK: &str = "No one can answer questions in this workflow, so go on with your defaults and say in your result what you assumed.";

/// Questions the agent asked without waiting for the answer: the user input
/// handler takes them in the background
/// ([`UserInputHandler::asked`](crate::user_input::UserInputHandler::asked)),
/// and an answer it returns reaches the agent in its mailbox, as a
/// subagent's report does. Without a handler, the agent is told no one can
/// answer.
fn ask_later(
    drive: &Drive<'_>,
    questions: String,
    asked: Vec<arko_agents::questions::AskedQuestion>,
) {
    let ctx = drive.ctx.clone();
    let agent_run_id = drive.agent_run_id;
    let agent_name = drive.agent_name.to_string();
    let ids = asked
        .iter()
        .map(|question| question.id.clone())
        .collect::<Vec<_>>();
    // (The handler's answer may also say that no one can answer.)
    let heading = match ids.as_slice() {
        [one] => format!("About your question {one}:"),
        _ => format!(
            "About your questions {}:",
            arko_agents::questions::id_list(&ids)
        ),
    };
    let Some(handler) = ctx.user_input_handler.clone() else {
        ctx.subagents.host_mail(
            &ctx,
            agent_run_id,
            &agent_name,
            String::new(),
            NO_ONE_TO_ASK.to_string(),
        );
        return;
    };
    let request = crate::user_input::UserInputRequest {
        workflow_run_id: ctx.workflow_run_id,
        agent_run_id,
        agent_name: agent_name.clone(),
        questions,
        asked,
        wait: false,
    };
    tokio::spawn(async move {
        let answer = tokio::select! {
            biased;
            _ = ctx.run_control.stopped() => return,
            answer = handler.asked(request) => answer,
        };
        match answer {
            Ok(Some(answer)) if !answer.trim().is_empty() => {
                ctx.subagents
                    .host_mail(&ctx, agent_run_id, &agent_name, heading, answer);
            }
            Ok(_) => {}
            Err(error) => log::warn!("The answer to {ids:?} of run {agent_run_id} failed: {error}"),
        }
    });
}

/// Answers the wait a recovered task was saved in, as the driver answered
/// it when the agent reached it: a recovered agent waits for the command
/// without announcing the wait again. The tool approvals it awaits are
/// decided under the approval policy, an error-recovery pause under the
/// error recovery policy, and its questions by the user input handler. Each
/// may suspend the agent for its caller, as it would have. (A Claude Code
/// agent doesn't keep a saved approval wait, and approved calls that ran
/// are recovered by the runtime.)
async fn answer_saved_wait(
    drive: &Drive<'_>,
    task: &Task,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
) -> Result<(), WorkflowError> {
    let Drive {
        agent_name,
        agent_run_id,
        nested,
        ctx,
        ..
    } = *drive;
    match &task.current_state {
        TaskState::AwaitingToolApproval { .. } | TaskState::AwaitingBatchToolApproval { .. }
            if task.running_tool_calls.is_empty() && !task.agent_config.runs_in_claude_code() =>
        {
            let Some(AgentPause::ToolApproval { calls }) = AgentPause::of(&task.current_state)
            else {
                return Ok(());
            };
            let batch_id = match &task.current_state {
                TaskState::AwaitingBatchToolApproval { batch_id, .. } => Some(*batch_id),
                _ => None,
            };
            match decide_tool_calls(agent_name, agent_run_id, &calls, nested, ctx).await? {
                Some(decisions) => {
                    send_tool_decisions(cmd_tx, batch_id, decisions).await;
                    Ok(())
                }
                None => {
                    pause_for_caller(
                        agent_name,
                        agent_run_id,
                        cmd_tx,
                        AgentPause::ToolApproval { calls },
                    )
                    .await
                }
            }
        }
        TaskState::AwaitingErrorRecoveryApproval {
            error_message,
            cause,
            error_kind,
        } => {
            let mut recovery = RecoveryState {
                paused: Some(cause.clone()),
                paused_kind: *error_kind,
                ..RecoveryState::default()
            };
            answer_recovery_pause(
                agent_name,
                agent_run_id,
                error_message.clone(),
                &mut recovery,
                nested,
                cmd_tx,
                ctx,
            )
            .await
        }
        TaskState::AwaitingUserResponse { questions } => {
            let waiting = &task.questions.waiting;
            let asked = task
                .questions
                .open
                .iter()
                .filter(|question| waiting.contains(&question.id))
                .cloned()
                .collect();
            answer_questions(drive, questions.clone(), asked, cmd_tx).await
        }
        _ => Ok(()),
    }
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

/// Whether `error` says the host stopped the run alone
/// ([`WorkflowError::AgentStopped`]): not a failure.
pub(crate) fn is_run_stop(error: &WorkflowError) -> bool {
    matches!(error, WorkflowError::AgentStopped { .. })
}

/// How a run starts.
#[derive(Debug, Clone)]
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
    /// caller chooses. `AgentHandle::wait` turns it on.
    pub driven: bool,
    /// The tool call the run runs inside (a delegated agent's or a
    /// verifier's), recorded in its task ([`Task::parent`]).
    pub parent: Option<arko_agents::TaskParent>,
    /// A completion verifier's review, which may start during a graceful
    /// stop's wrap-up, and isn't told to wrap up.
    pub review: bool,
    /// The run's working directory, when not the workflow's workspace (see
    /// [`RunOptions::workspace`](super::RunOptions::workspace)).
    pub workspace: Option<std::path::PathBuf>,
    /// The run is a background subagent's. Its tools are a subagent's (see
    /// [`crate::subagents`]).
    pub subagent: bool,
    /// The configuration it starts with, instead of its agent's registered
    /// one (a subagent's, as the host's hook changed it, or the host's own,
    /// see `host_config`).
    pub config: Option<AgentConfig>,
    /// `config` is the host's ([`RunOptions::config`](super::RunOptions::config)):
    /// the task keeps it as one the host gave it ([`Task::reconfigured`]).
    pub host_config: bool,
    /// Images that go with the prompt (see
    /// [`RunOptions::images`](super::RunOptions::images)).
    pub images: Vec<arko_ai::Image>,
}

/// A message to an agent run: its text, images after it, and the
/// configuration the run continues with, when the host changes it (see
/// [`ContinueOptions::reconfigure`](super::ContinueOptions::reconfigure)).
pub(crate) struct UserInput<'a> {
    pub text: &'a str,
    pub images: Vec<arko_ai::Image>,
    pub config: Option<AgentConfig>,
    /// The message the conversation is cut back to first (see
    /// [`ContinueOptions::cut_back_to`](super::ContinueOptions::cut_back_to)).
    pub cut_back_to: Option<usize>,
    /// The directory the run goes on in (see
    /// [`ContinueOptions::workspace`](super::ContinueOptions::workspace)).
    pub workspace: Option<std::path::PathBuf>,
}

impl<'a> From<&'a str> for UserInput<'a> {
    fn from(text: &'a str) -> Self {
        Self {
            text,
            images: Vec::new(),
            config: None,
            cut_back_to: None,
            workspace: None,
        }
    }
}

/// The list of the agents a new run of `agent_name` may message, for its
/// initial prompt (empty without `shown`, or for a chat, which can't message
/// anyone), recording the version of their statuses the run has seen.
fn roster_section(ctx: &SharedContext, agent_name: &str, task: &Task, shown: bool) -> String {
    let recipients = ctx.message_recipients(agent_name);
    if recipients.is_empty() {
        return String::new();
    }
    // The roster's statuses and the version the run has seen come from one
    // registry snapshot.
    let roster = ctx.registry.roster(
        task.id,
        &recipients,
        ctx.paused.load(std::sync::atomic::Ordering::Acquire),
    );
    if !shown || task.agent_config.tool_config.chat {
        return String::new();
    }
    format!("{ROSTER_HEADING}{}", render_recipients(ctx, &roster))
}

/// A task's refusal (a cut back or reconfiguration the task can't take) as
/// the host's error.
fn builder_invalid(error: arko_agents::ArkoAgentsError) -> WorkflowError {
    match error {
        arko_agents::ArkoAgentsError::InvalidTaskState(reason) => {
            WorkflowError::BuilderInvalid(reason)
        }
        other => WorkflowError::BuilderInvalid(other.to_string()),
    }
}

/// `images` as `config`'s model can be sent them: each one over the model's
/// limits shrunk to fit them ([`arko_ai::Model::sendable_image`]), or an
/// error naming the one that can't be sent and why. With `held`, the
/// conversation they join, also an error when a request would then hold
/// more images, or more image data, than the model's provider takes
/// ([`arko_ai::Model::image_limits`]): every later request would fail.
pub(crate) fn sendable_images(
    agent_name: &str,
    config: &AgentConfig,
    images: Vec<arko_ai::Image>,
    held: Option<&arko_ai::Conversation>,
) -> Result<Vec<arko_ai::Image>, WorkflowError> {
    let model = &config.llm_config.model;
    let count = images.len();
    let images = images
        .into_iter()
        .enumerate()
        .map(|(index, image)| match model.sendable_image(&image) {
            Ok(fitted) => Ok(fitted.unwrap_or(image)),
            Err(reason) => {
                let which = if count == 1 {
                    "the image".to_string()
                } else {
                    format!("image {}", index + 1)
                };
                Err(WorkflowError::BuilderInvalid(format!(
                    "agent '{agent_name}' can't be sent {which}: {reason}"
                )))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(conversation) = held
        && !images.is_empty()
    {
        let limits = model.image_limits();
        let (held_images, held_bytes) = image_load(conversation.messages.iter());
        let (_, new_bytes) =
            image_load(std::iter::once(&arko_ai::Message::new_user_with_contents(
                images
                    .iter()
                    .cloned()
                    .map(arko_ai::Content::Image)
                    .collect(),
            )));
        let room = "cut the conversation back or hand it over to make room";
        if held_images + images.len() > limits.max_images {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{agent_name}' can't be sent {}. Its conversation holds {held_images} images, and model '{}' takes at most {} in a request ({room})",
                if images.len() == 1 {
                    "another image".to_string()
                } else {
                    format!("{} more images", images.len())
                },
                model.id,
                limits.max_images
            )));
        }
        if held_bytes + new_bytes > limits.max_request_image_bytes {
            const MIB: f64 = 1024.0 * 1024.0;
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{agent_name}' can't be sent {:.1} MB more of images. Its conversation holds {:.1} MB, and a request to model '{}' takes at most {:.1} MB ({room})",
                new_bytes as f64 / MIB,
                held_bytes as f64 / MIB,
                model.id,
                limits.max_request_image_bytes as f64 / MIB
            )));
        }
    }
    Ok(images)
}

/// An error unless `config`'s model can be sent each of `images`, as it is
/// or fitted, judged from each one's header without decoding it
/// ([`arko_ai::Model::check_image`]): for checking every recipient before
/// fitting for one.
pub(crate) fn check_sendable_images(
    agent_name: &str,
    config: &AgentConfig,
    images: &[arko_ai::Image],
) -> Result<(), WorkflowError> {
    for (index, image) in images.iter().enumerate() {
        if let Err(reason) = config.llm_config.model.check_image(image) {
            let which = if images.len() == 1 {
                "the image".to_string()
            } else {
                format!("image {}", index + 1)
            };
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent '{agent_name}' can't be sent {which}: {reason}"
            )));
        }
    }
    Ok(())
}

/// How many images `messages` hold (in tool results too) and how many bytes
/// of base64 data.
fn image_load<'a>(messages: impl Iterator<Item = &'a arko_ai::Message>) -> (usize, usize) {
    let mut load = (0, 0);
    for message in messages {
        for content in &message.content {
            let images: Vec<&arko_ai::Image> = match content {
                arko_ai::Content::Image(image) => vec![image],
                arko_ai::Content::ToolResult(result) => {
                    result.images.iter().map(|image| &image.image).collect()
                }
                arko_ai::Content::Text(_) => Vec::new(),
            };
            for image in images {
                load.0 += 1;
                if let arko_ai::Image::Base64 { data, .. } = image {
                    load.1 += data.len();
                }
            }
        }
    }
    load
}

/// What a run is, for the workflow tools it gets: how deep it is (see
/// [`StartedRun::depth`]), whether it is a background subagent, and whether
/// the host's hook offers it the tools that start agent runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunRole {
    pub depth: u32,
    pub subagent: bool,
    /// Always true for a run the host drives. For a subagent it is the
    /// answer of [`SubagentHook::offers_start_tools`](crate::subagents::SubagentHook::offers_start_tools).
    pub starts: bool,
}

impl Default for RunRole {
    fn default() -> Self {
        Self {
            depth: 0,
            subagent: false,
            starts: true,
        }
    }
}

impl From<u32> for RunRole {
    fn from(depth: u32) -> Self {
        Self {
            depth,
            ..Self::default()
        }
    }
}

impl RunRole {
    /// The role of run `task_id` at `depth`.
    pub(crate) fn of(ctx: &SharedContext, depth: u32, task_id: uuid::Uuid) -> Self {
        let subagent = ctx.subagents.is_subagent(task_id);
        Self {
            depth,
            subagent,
            starts: !subagent || ctx.subagents.offers_start_tools(task_id),
        }
    }
}

/// Which subagent tools a run of `agent_name` in `role` gets: the caller's
/// (it may start subagents), and those for talking with its runs or its
/// lead.
fn subagent_tools(ctx: &SharedContext, agent_name: &str, role: RunRole) -> (bool, bool) {
    let caller = role.starts
        && ctx.subagents.may_start_at(role.depth)
        && ctx
            .workflow_services
            .get()
            .is_some_and(|services| services.is_allowed(agent_name));
    let talks = ctx.subagents.config().is_some() && (caller || role.subagent);
    (caller, talks)
}

/// Whether a run of `agent_name` in `role` is offered `run_workflow_agent`:
/// a caller, unless the subagents configuration hides it from callers, or
/// from subagents as deep as subagents may nest.
fn offers_blocking_tool(ctx: &SharedContext, agent_name: &str, role: RunRole) -> bool {
    let allowed = ctx
        .workflow_services
        .get()
        .is_some_and(|services| services.is_allowed(agent_name));
    match ctx.subagents.config() {
        Some(config) => {
            allowed
                && role.starts
                && config.offer_blocking_tool
                && !(role.subagent && role.depth >= config.max_depth)
        }
        None => allowed,
    }
}

impl Default for StartOptions {
    fn default() -> Self {
        Self {
            reserved: None,
            roster: true,
            depth: 0,
            admitted: false,
            driven: true,
            parent: None,
            review: false,
            workspace: None,
            subagent: false,
            config: None,
            host_config: false,
            images: Vec::new(),
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

/// Why an interrupted run inside a recovered run's tool call that doesn't
/// resume it is recorded cancelled.
pub(crate) const SUPERSEDED_AFTER_RESTART: &str = "Superseded because the workflow stopped while this agent ran inside another agent's tool call, and that call was not resumed when its agent was recovered (a completion is reviewed again by a new review).";

/// Whether `run` was cancelled because another run replaced it (see
/// [`SUPERSEDED_AFTER_RESTART`]): it is not reopened, as that would do its
/// work twice (a review's verdict would reach no one).
pub(crate) fn is_superseded(run: &AgentResult) -> bool {
    run.task().is_ok_and(|task| {
        matches!(
            &task.current_state,
            TaskState::Cancelled { reason } if reason == SUPERSEDED_AFTER_RESTART
        )
    })
}

/// The heading of the initial prompt's list of the agents it may message.
const ROSTER_HEADING: &str = "\n\n# Agents you can message\n\nMessage these agents with send_message while they are running or waiting; call list_recipients for their current status. You are told when one starts or finishes.\n";

/// `prompt` without the list of agents it may message.
pub(crate) fn without_roster(prompt: &str) -> &str {
    prompt
        .rfind(ROSTER_HEADING)
        .map_or(prompt, |start| &prompt[..start])
}

/// The heading of the initial prompt's messages from other agents (see
/// `WorkflowEngine::pending_messages`).
const MESSAGES_HEADING: &str = "\n\n# Messages from other agents:\n";

/// The task part of an initial prompt: `prompt` without the messages it
/// delivered, its notes, and the list of agents it may message.
pub(crate) fn task_part(prompt: &str) -> &str {
    let prompt = crate::notes::without_notes(without_roster(prompt));
    match prompt.rfind(MESSAGES_HEADING) {
        Some(start) if prompt[start..].contains("\n--- Message ") => &prompt[..start],
        _ => prompt,
    }
}

/// What a workflow does when one of its agents hands its conversation over
/// ([`arko_agents::handover`]): the messages the old conversation held are
/// recorded as read by the run (and acknowledged when it completes, as if
/// its conversation still held them), the freshness tracker forgets the
/// run's reads (the fresh conversation holds none of their content), and
/// the fresh conversation restates the task without the delivered messages
/// (the report covers them), with the standing notes for the agent it had
/// been given (newer ones come in its next context update), and with the
/// current list of agents it may message, when its prompt had one, and the
/// subagents it started that still run (`# Your workers`). After
/// Claude Code compacted the conversation of an agent whose model runs in
/// it, the agent gets those standing notes again, which a summary may leave
/// out.
pub(crate) struct WorkflowHandover {
    pub context: Arc<SharedContext>,
    pub agent_name: String,
    /// The messages delivered in the run's initial prompt.
    pub prompt_message_ids: Vec<uuid::Uuid>,
    /// The run is a completion verifier's review (see
    /// [`StartOptions::review`]).
    pub review: bool,
}

impl WorkflowHandover {
    /// The user input handler, for a run whose questions it takes (not a
    /// subagent whose questions go to its lead).
    fn question_handler(
        &self,
        task: &Task,
    ) -> Option<&Arc<dyn crate::user_input::UserInputHandler>> {
        let ctx = &self.context;
        let to_lead = ctx
            .subagents
            .config()
            .is_some_and(|config| config.questions_to_lead)
            && ctx.subagents.is_subagent(task.id);
        if to_lead {
            return None;
        }
        ctx.user_input_handler.as_ref()
    }
}

impl HandoverHook for WorkflowHandover {
    fn on_handover(&self, task: &Task, _report: &str) -> HandoverSeed {
        let ctx = &self.context;
        ctx.messages.hand_over(
            &self.agent_name,
            task.id,
            &task.conversation,
            &self.prompt_message_ids,
        );
        if let Some(tracker) = &ctx.resource_tracker {
            tracker.clear_actor(&ActorId::new(task.id, &self.agent_name));
        }
        let prompt = &task.initial_task_prompt;
        let mut context = ctx
            .notes
            .handover_section(&self.agent_name, task.id, self.review);
        let recipients = ctx.message_recipients(&self.agent_name);
        if prompt.contains(ROSTER_HEADING) && !recipients.is_empty() {
            let roster = ctx.registry.roster(
                task.id,
                &recipients,
                ctx.paused.load(std::sync::atomic::Ordering::Acquire),
            );
            context.push_str(ROSTER_HEADING);
            context.push_str(&render_recipients(ctx, &roster));
        }
        // The subagents it started and still waits for: the report may not
        // name them.
        if let Some(workers) = ctx.subagents.workers_section(task.id) {
            if !context.is_empty() {
                context.push_str("\n\n");
            }
            context.push_str(&workers);
        }
        // A chat goes on from the user's latest input, not from its first
        // message.
        let task_prompt = if task.agent_config.tool_config.chat {
            HandoverSeed::for_chat(task).task_prompt
        } else {
            task_part(prompt).to_string()
        };
        HandoverSeed {
            task_prompt,
            context: (!context.is_empty()).then_some(context),
        }
    }

    /// The questions the user input handler says are still open.
    fn open_questions(&self, task: &Task) -> Option<Vec<arko_agents::questions::AskedQuestion>> {
        self.question_handler(task)?.open_questions(task.id)
    }

    /// The user input handler numbers the questions it takes.
    fn question_numbers(&self, task: &Task, count: u32) -> Option<u32> {
        self.question_handler(task)?
            .question_numbers(task.id, &self.agent_name, count)
    }

    /// The user input handler hears of the questions it takes at once.
    fn questions_asked(
        &self,
        task: &Task,
        asked: &[arko_agents::questions::AskedQuestion],
        wait: bool,
    ) {
        if let Some(handler) = self.question_handler(task) {
            handler.questions_recorded(task.id, &self.agent_name, asked, wait);
        }
    }

    fn on_compaction(&self, task: &Task) -> Option<String> {
        let notes = self
            .context
            .notes
            .handover_section(&self.agent_name, task.id, self.review);
        let workers = self.context.subagents.workers_section(task.id);
        let parts = [notes.trim(), workers.as_deref().unwrap_or_default()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        (!parts.is_empty()).then(|| parts.join("\n\n"))
    }
}

/// An agent run the engine started, to be driven to its result.
pub(crate) struct StartedRun {
    pub agent_name: String,
    pub agent_run_id: uuid::Uuid,
    pub event_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    pub cmd_tx: tokio::sync::mpsc::Sender<UserCommand>,
    pub join_handle: AbortOnDrop,
    /// The task as the run started it, the record of an agent that had to be
    /// aborted, unless the task journal saved a later state.
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
    /// Set while the workflow drives the run. It decides every wait the
    /// agent reaches (under its policies and handlers, however long they
    /// take, also while the workflow is paused) or suspends the agent for
    /// its caller, so the agent's own command timeout doesn't apply. Not
    /// saved with the task. Resumed outside a workflow, it is timed as
    /// configured.
    pub host_decisions: HostDecisions,
    /// The run was recovered after the workflow stopped while it ran. A
    /// wait its task was saved in is answered first (see
    /// [`answer_saved_wait`]).
    pub recovered: bool,
    /// The task journal saving the run while it runs, if any.
    pub journal: Option<Arc<PersistenceManager>>,
    /// This launch of the run (see `RunControl::begin_run`).
    pub launch: u64,
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
/// counts from the moment the run stopped. One nested inside another agent's
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
/// replaced by `ended`), and aborts it at `deadline`. `None` means the agent
/// was aborted.
#[allow(clippy::too_many_arguments)]
async fn wind_down(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    join_handle: &mut AbortOnDrop,
    deadline: tokio::time::Instant,
    ended: &TaskState,
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
                            AgentEvent::TaskStateChanged(ended.clone())
                        }
                        event => event,
                    };
                    publish_event(agent_name, agent_run_id, &event, ctx, persistence).await;
                    ctx.registry.observe(agent_run_id, &event);
                    if let AgentEvent::UsageReported(usage) = &event {
                        charge_usage(agent_name, agent_run_id, usage, ctx, persistence);
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
/// is stopped the outcome is the stop's error. The agent's result is
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
        recovered,
        journal,
        launch,
    } = run;
    host_decisions.set(true);
    // Until the end is recorded below: if this future is dropped (and with
    // it, the agent's task aborted), the run is recorded as abandoned.
    let mut abandoned = Abandoned {
        ctx,
        agent_name: agent_name.clone(),
        agent_run_id,
        launch,
        journal,
        armed: true,
    };
    let saved_wait = if recovered {
        let drive = Drive {
            agent_name: &agent_name,
            agent_run_id,
            nested: depth > 0,
            ctx,
            persistence,
        };
        // The run's own stop ends the wait (an automatic recovery's can last
        // until a usage limit resets), as it ends the event loop's.
        tokio::select! {
            biased;
            reason = stop_set(ctx.run_control.run_cancel(agent_run_id)) => {
                Err(WorkflowError::AgentStopped {
                    agent: agent_name.clone(),
                    task_id: agent_run_id,
                    reason,
                })
            }
            answered = answer_saved_wait(&drive, &started_task, &cmd_tx) => answered,
        }
    } else {
        Ok(())
    };
    // A run resumed (or recovered) at an error-recovery pause starts in it.
    let paused = match &started_task.current_state {
        TaskState::AwaitingErrorRecoveryApproval {
            cause, error_kind, ..
        } => Some((cause.clone(), *error_kind)),
        _ => None,
    };
    let drive_result = match saved_wait {
        Ok(()) => {
            drive_event_loop(
                &agent_name,
                agent_run_id,
                &mut event_rx,
                &cmd_tx,
                ctx,
                persistence,
                batch_stop,
                depth > 0,
                paused,
            )
            .await
        }
        Err(error) => Err(error),
    };

    let stopping = drive_result.as_ref().is_err_and(|error| !is_pause(error));
    // Once, so what is published is what is recorded.
    let ended = stopping.then(|| stopped_state(ctx, &drive_result));
    let wound_down = if let Some(ended) = &ended {
        let deadline = stop_deadline(ctx, depth);
        wind_down(
            &agent_name,
            agent_run_id,
            &mut event_rx,
            &cmd_tx,
            &mut join_handle,
            deadline,
            ended,
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
            // Its latest saved state, when the task journal saves it: the
            // progress it made before it was aborted is kept.
            let mut task = abandoned
                .journal
                .as_ref()
                .and_then(|journal| journal.latest_live_task(agent_run_id))
                .unwrap_or(started_task);
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
            if let Some(ended) = ended
                && matches!(&task.current_state, TaskState::Cancelled { reason } if reason == CANCELLED_BY_USER)
            {
                task.set_state(ended);
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
    unregister_run(ctx, agent_run_id, launch);
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
        && result.task().is_ok_and(|task| {
            matches!(
                task.current_state,
                TaskState::AwaitingErrorRecoveryApproval {
                    cause: arko_agents::ErrorRecoveryCause::Middleware,
                    ..
                }
            )
        })
    {
        return Err(WorkflowError::VerifierFailed {
            verifier,
            agent: agent_name,
            task_id: result.agent_run_id,
            reason,
        });
    }
    match drive_result {
        // Its own stop's cancel came after it completed: the completion
        // (a progress report, if it was told) is its outcome.
        Err(WorkflowError::AgentStopped { .. }) if result.status == AgentRunStatus::Completed => {
            Ok(result)
        }
        drive_result => {
            drive_result?;
            run_outcome(result)
        }
    }
}

/// What a run call returns for a run that ended as `result`: the result
/// when it completed, else the error its status stands for.
pub(crate) fn run_outcome(result: AgentResult) -> Result<AgentResult, WorkflowError> {
    let agent = result.agent_name.clone();
    match result.status {
        AgentRunStatus::Completed => Ok(result),
        AgentRunStatus::WaitingForInput => Err(WorkflowError::UserInputRequired {
            agent,
            task_id: result.agent_run_id,
            questions: result.error.clone().unwrap_or_default(),
        }),
        AgentRunStatus::Paused => match result
            .task()
            .ok()
            .and_then(|task| AgentPause::of(&task.current_state))
        {
            Some(pause) => Err(WorkflowError::AgentPaused {
                agent,
                task_id: result.agent_run_id,
                pause,
            }),
            None => Err(WorkflowError::AgentInterrupted {
                agent,
                reason: result.error.clone().unwrap_or_default(),
            }),
        },
        AgentRunStatus::Interrupted => Err(WorkflowError::AgentInterrupted {
            agent,
            reason: result.error.clone().unwrap_or_default(),
        }),
        AgentRunStatus::Cancelled => Err(WorkflowError::AgentCancelled {
            agent,
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

/// Whether task `parent` holds call `tool_call_id` open: recorded running,
/// without a result. Recovering the task settles the call, and with it the
/// agent run the call started: resumed with it, or superseded.
pub(crate) fn holds_open_call(parent: &Task, tool_call_id: uuid::Uuid) -> bool {
    parent
        .running_tool_calls
        .iter()
        .any(|call| call.tool_call_id == tool_call_id && call.output.is_none())
}

/// Whether interrupted run `parent`, once recovered, settles `child`, an
/// interrupted run started inside its tool call `tool_call_id`: resumes it
/// or supersedes it (see `WorkflowEngine::supersede_children`). It does
/// when the call is open ([`holds_open_call`]), and always for a
/// completion verifier's review (its `complete_task` call runs again).
fn settles_child(
    ctx: &SharedContext,
    parent: &AgentResult,
    child: &AgentResult,
    tool_call_id: uuid::Uuid,
) -> bool {
    parent.status == AgentRunStatus::Interrupted
        && (parent
            .task()
            .is_ok_and(|task| holds_open_call(task, tool_call_id))
            || is_verifier_agent(ctx, &child.agent_name))
}

pub(crate) fn is_verifier_agent(ctx: &SharedContext, agent_name: &str) -> bool {
    ctx.completion_verifiers
        .iter()
        .any(|config| config.verifier_agent == agent_name)
}

/// Whether `run`, started inside another run's tool call, is orphaned: that
/// run can't resume it any more. It isn't running, and it ended (or has no
/// record), or it stopped with the call answered (the call failed, and its
/// result named this run). An orphaned run is its own: the host recovers
/// it ([`Workflow::recover_agent`](super::Workflow::recover_agent) lists
/// and accepts it, releasing its parent link), and any caller permitted to
/// run its agent may resume it (`run_workflow_agent` with `resume_run`).
pub(crate) fn is_orphaned(ctx: &SharedContext, run: &AgentResult) -> bool {
    let Some(parent) = run.info().parent else {
        return false;
    };
    if ctx.registry.is_running(parent.task_id) {
        return false;
    }
    !ctx.agent_results
        .read()
        .iter()
        .find(|result| result.agent_run_id == parent.task_id)
        .is_some_and(|result| settles_child(ctx, result, run, parent.tool_call_id))
}

/// Fails, for the host, unless `run` (started inside another run's tool
/// call) is orphaned ([`is_orphaned`]): otherwise the run that started it
/// resumes it, or recovering that run does.
fn refuse_unless_orphaned(ctx: &SharedContext, run: &AgentResult) -> Result<(), WorkflowError> {
    let Some(parent) = run.info().parent else {
        return Ok(());
    };
    if is_orphaned(ctx, run) {
        return Ok(());
    }
    Err(WorkflowError::BuilderInvalid(
        if ctx.registry.is_running(parent.task_id) {
            format!(
                "agent run {} was started by a tool call of agent run {}, which is running and resumes it (or can, with run_workflow_agent's resume_run)",
                run.agent_run_id, parent.task_id
            )
        } else {
            format!(
                "agent run {} was started by a tool call of agent run {}. Recover that run, which recovers this one",
                run.agent_run_id, parent.task_id
            )
        },
    ))
}

/// The recorded result of agent run `agent_run_id`, if it has one.
pub(crate) fn recorded_result(
    ctx: &SharedContext,
    agent_run_id: uuid::Uuid,
) -> Option<AgentResult> {
    ctx.agent_results
        .read()
        .iter()
        .find(|result| result.agent_run_id == agent_run_id)
        .cloned()
}

/// Waits until the recovery `recovering` (see
/// [`AgentRegistry::recovery`](crate::messaging::registry::AgentRegistry::recovery))
/// has ended.
async fn recovery_ended(mut recovering: tokio::sync::watch::Receiver<()>) {
    while recovering.changed().await.is_ok() {}
}

/// How [`WorkflowEngine::recover_at`] recovers an interrupted run.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Recovery<'a> {
    /// The run's depth (see [`StartedRun::depth`]).
    pub depth: u32,
    /// Replaces the run's parent link. `Some(None)` releases it (the host
    /// recovers an orphaned run), `Some(Some(call))` points it at the tool
    /// call resuming it, so that a later stop finds it there.
    pub parent: Option<Option<arko_agents::TaskParent>>,
    /// Given to the run as a user message at its next turn boundary, once
    /// the calls it was running are settled.
    pub message: Option<&'a str>,
    /// Images that go with `message`, after its text.
    pub images: &'a [arko_ai::Image],
    /// When another recovery runs it, wait for that recovery and return
    /// the outcome it recorded, instead of failing.
    pub wait: bool,
    /// Reopens a cancelled run. It is recovered as an interrupted one
    /// would be (see [`WorkflowEngine::reopen_agent`]).
    pub reopen: bool,
}

/// What a reopened run is told first (see [`WorkflowEngine::reopen_agent`]).
pub(crate) const REOPENED_NOTE: &str = "Your run was stopped before it finished and has now been reopened: continue your task where you were. Tool calls reported as cancelled or stopped did not finish; check their effects before repeating them.";

/// The message a reopened run gets: [`REOPENED_NOTE`], then `prompt`.
pub(crate) fn reopened_message(prompt: &str) -> String {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        REOPENED_NOTE.to_string()
    } else {
        format!("{REOPENED_NOTE}\n\n{prompt}")
    }
}

/// Unregisters a run dropped before it was driven (its runtime task is
/// aborted with it): its agent is finished, cancelled, unless another run
/// of it is running.
pub(crate) fn abandon_run(run: &StartedRun, ctx: &SharedContext) {
    abandon(
        ctx,
        &run.agent_name,
        run.agent_run_id,
        run.launch,
        run.journal.as_ref(),
    );
}

fn abandon(
    ctx: &SharedContext,
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    launch: u64,
    journal: Option<&Arc<PersistenceManager>>,
) {
    ctx.registry.finish(agent_run_id, AgentRunStatus::Cancelled);
    // Not recorded: no longer saved either.
    if let Some(journal) = journal {
        journal.end_live_run(agent_run_id);
    }
    // Not recorded, so it can't be continued: freshness forgets its reads.
    if let Some(tracker) = &ctx.resource_tracker {
        tracker.finish_actor(&ActorId::new(agent_run_id, agent_name), false);
    }
    ctx.messages.unregister_active(agent_name, agent_run_id);
    unregister_run(ctx, agent_run_id, launch);
}

/// Records a run whose driver was dropped before it recorded the run's end.
struct Abandoned<'a> {
    ctx: &'a SharedContext,
    agent_name: String,
    agent_run_id: uuid::Uuid,
    launch: u64,
    journal: Option<Arc<PersistenceManager>>,
    armed: bool,
}

impl Drop for Abandoned<'_> {
    fn drop(&mut self) {
        if self.armed {
            abandon(
                self.ctx,
                &self.agent_name,
                self.agent_run_id,
                self.launch,
                self.journal.as_ref(),
            );
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
        // A stop the workflow already worded (fail-fast, or the run's own).
        (None, Err(WorkflowError::AgentCancelled { reason, .. }))
        | (None, Err(WorkflowError::AgentStopped { reason, .. })) => reason.clone(),
        (None, Err(error)) => format!("Stopped by the workflow: {error}"),
        (None, Ok(())) => "Stopped by the workflow.".to_string(),
    }
}

/// The state a stopped agent's generic cancellation becomes. It is a failure
/// when a subagent's cap on failed requests stopped it, as at a pause.
fn stopped_state(ctx: &SharedContext, drive_result: &Result<(), WorkflowError>) -> TaskState {
    match drive_result {
        Err(WorkflowError::Agent(arko_agents::ArkoAgentsError::TaskExecution(reason)))
            if ctx.run_control.cancellation().is_none() =>
        {
            TaskState::Failed {
                reason: reason.clone(),
                cause: None,
                error_kind: None,
            }
        }
        _ => TaskState::Cancelled {
            reason: stop_reason(ctx, drive_result),
        },
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
        && let Err(error) = Box::pin(persistence.auto_checkpoint(ctx)).await
    {
        log::warn!("Could not save the checkpoint recording the workflow's stop: {error}");
    }
}

/// Records `result` in the run's results, replacing an earlier result of
/// its task (without persistence, see [`PersistenceManager::record_result`]).
pub(crate) fn record_result(ctx: &SharedContext, result: AgentResult) {
    crate::workflow::result::upsert_result(&mut ctx.agent_results.write(), result);
}

/// Records an agent run that stopped: its result (replacing an earlier one
/// of the same task), the messages it holds (acknowledged when it completed,
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
    // Its subagent mail too: what it holds leaves the mailbox, the rest is
    // given again.
    ctx.subagents.settle_run(ctx, &task);
    let pause = AgentPause::of(&task.current_state);
    let mut result = AgentResult::of_task(agent_name.clone(), task);
    // Ended by a stop the host asked for: not the agent's failure.
    result.stopped_by = ctx.run_control.stopped_by(agent_run_id, &result.status);
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
    let failure_cause = result.failure_cause().cloned();
    let failure_kind = result.failure_kind();
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
            Some(AgentPause::ErrorRecovery {
                cause, error_kind, ..
            }) => {
                hook.metadata
                    .insert("pause".to_string(), serde_json::json!("error_recovery"));
                hook.metadata.insert(
                    "cause".to_string(),
                    serde_json::json!(crate::policy::cause_name(cause)),
                );
                if let Some(kind) = error_kind {
                    hook.metadata
                        .insert("error_kind".to_string(), serde_json::json!(kind.as_str()));
                }
            }
            None => {}
        }
        // A failure at a rejected error-recovery pause: its cause.
        if let Some(cause) = &failure_cause {
            hook.metadata.insert(
                "cause".to_string(),
                serde_json::json!(crate::policy::cause_name(cause)),
            );
        }
        if let Some(kind) = failure_kind {
            hook.metadata
                .insert("error_kind".to_string(), serde_json::json!(kind.as_str()));
        }
    });

    // The live checkpoint no longer refers to the run's live file (which is
    // deleted then), and then the checkpoint recording its end is saved. A
    // checkpoint that can't be saved is logged; the next one tries again.
    if let Some(persistence) = persistence
        && let Err(error) = Box::pin(persistence.release_live_run(ctx, agent_run_id)).await
    {
        log::warn!(
            "Workflow '{}': could not update the live checkpoint after agent '{}' ({agent_run_id}) ended: {error}",
            ctx.workflow_name,
            result.agent_name
        );
    }
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
    pub(crate) fn persistence(&self) -> Option<&Arc<PersistenceManager>> {
        self.persistence.as_ref()
    }

    /// Gives a task that was saved, and loaded again, the embedded skills of
    /// its agent's registered configuration. A saved configuration doesn't
    /// hold them, so a run that goes on takes the current ones, and a
    /// built-in skill that changed doesn't stop a resume. A task with
    /// embedded skills of its own, and one whose agent has none, are left as
    /// they are. A task the host reconfigured gets them when its own skills
    /// configuration is set, and stays without when it has none. Only the
    /// embedded skills come from the registered configuration, and the
    /// invalid skill policy of a task that had no skills configuration.
    fn restore_embedded_skills(task: &mut Task, ctx: &SharedContext, agent_name: &str) {
        let Some(current) = ctx.registered_skills(agent_name) else {
            return;
        };
        if current.embedded.is_empty() {
            return;
        }
        let skills = if task.reconfigured {
            match task.agent_config.skills_config.as_mut() {
                Some(skills) => skills,
                None => return,
            }
        } else {
            task.agent_config.skills_config.get_or_insert_with(|| {
                arko_agents::skill::SkillsConfig {
                    on_invalid_skill: current.on_invalid_skill.clone(),
                    ..Default::default()
                }
            })
        };
        if skills.embedded.is_empty() {
            skills.embedded = current.embedded;
        }
    }

    /// Loads the skills of `config` and registers its `load_skill`, for a
    /// run in `role`.
    ///
    /// A subagent's `load_skill` never loads a skill kept to the main agent
    /// (`workers: false`), also when its configuration was saved before
    /// [`SkillsConfig::subagent`](arko_agents::skill::SkillsConfig::subagent)
    /// was set. Only the tool's own copy of the configuration says so, so a
    /// saved conversation keeps its prompt and its tool list. A new run's
    /// configuration has it set already (`prepare_agent_config_for`).
    fn setup_skills(
        config: &mut AgentConfig,
        registry: &mut ToolRegistry,
        workspace: &std::path::Path,
        role: RunRole,
    ) -> Result<Option<SkillPromptContext>, WorkflowError> {
        let Some(source) = config.skills_config.as_ref() else {
            return Ok(None);
        };
        let manager = SkillManager::load(source, workspace)?;
        let narrowed = (role.subagent && !source.subagent).then(|| SkillsConfig {
            subagent: true,
            ..source.clone()
        });
        let context = manager.setup(registry, config);
        if let Some(narrowed) = narrowed
            && registry.get(LOAD_SKILL_TOOL_NAME).is_some()
        {
            registry.register(Arc::new(LoadSkillTool::new(narrowed, workspace)));
        }
        Ok(Some(context))
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
        role: impl Into<RunRole>,
    ) -> ToolRegistry {
        let role = role.into();
        let depth = role.depth;
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
            && offers_blocking_tool(ctx, agent_name, role)
        {
            registry.register(Arc::new(RunWorkflowAgentTool::new(
                services.clone(),
                agent_name.to_string(),
                depth,
            )));
        }
        let (caller, talks) = subagent_tools(ctx, agent_name, role);
        if caller {
            registry.register(Arc::new(crate::subagents::StartAgentTool::new(
                ctx.clone(),
                agent_name.to_string(),
                depth,
            )));
            registry.register(Arc::new(crate::subagents::ListAgentsTool::new(ctx.clone())));
            registry.register(Arc::new(crate::subagents::StopAgentTool::new(ctx.clone())));
            if ctx
                .subagents
                .config()
                .is_some_and(|config| config.peer_messages)
            {
                registry.register(Arc::new(crate::subagents::LinkAgentsTool::new(ctx.clone())));
            }
        }
        if talks {
            registry.register(Arc::new(crate::subagents::MessageAgentTool::new(
                ctx.clone(),
                caller,
            )));
            registry.register(Arc::new(crate::subagents::WaitForAgentsTool::new(
                ctx.clone(),
                agent_name.to_string(),
                self.llm_client.clone(),
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

    #[cfg(test)]
    pub(crate) fn prepare_agent_config(
        &self,
        agent_name: &str,
        base_config: &AgentConfig,
        ctx: &Arc<SharedContext>,
    ) -> AgentConfig {
        self.prepare_agent_config_for(agent_name, base_config, ctx, RunRole::default())
    }

    /// [`prepare_agent_config`](Self::prepare_agent_config) for a run in
    /// `role`.
    pub(crate) fn prepare_agent_config_for(
        &self,
        agent_name: &str,
        base_config: &AgentConfig,
        ctx: &Arc<SharedContext>,
        role: RunRole,
    ) -> AgentConfig {
        let mut config = base_config.clone();
        // The prompt written from this configuration is a new one, so a
        // subagent's lists and holds no skill kept to the main agent, also
        // when the host left `subagent` off. (A saved prompt stays, and only
        // its `load_skill` is narrowed, in `setup_skills`.)
        if role.subagent
            && let Some(skills) = config.skills_config.as_mut()
        {
            skills.subagent = true;
        }
        // A chat (`AgentToolConfig::chat`) stays one: it gets none of the
        // workflow's tools.
        if config.tool_config.chat {
            return config;
        }
        // A subagent whose questions go to its lead waits for the answer.
        if role.subagent
            && ctx
                .subagents
                .config()
                .is_some_and(|config| config.questions_to_lead)
        {
            config.tool_config.async_questions = false;
        }

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
        if offers_blocking_tool(ctx, agent_name, role) {
            ensure_tool_name(&mut config, RUN_WORKFLOW_AGENT_TOOL_NAME);
        }
        let (caller, talks) = subagent_tools(ctx, agent_name, role);
        if caller {
            ensure_tool_name(&mut config, crate::subagents::START_AGENT_TOOL_NAME);
            ensure_tool_name(&mut config, crate::subagents::LIST_AGENTS_TOOL_NAME);
            ensure_tool_name(&mut config, crate::subagents::STOP_AGENT_TOOL_NAME);
            if ctx
                .subagents
                .config()
                .is_some_and(|config| config.peer_messages)
            {
                ensure_tool_name(&mut config, crate::subagents::LINK_AGENTS_TOOL_NAME);
            }
        }
        if talks {
            ensure_tool_name(&mut config, crate::subagents::MESSAGE_AGENT_TOOL_NAME);
            ensure_tool_name(&mut config, crate::subagents::WAIT_FOR_AGENTS_TOOL_NAME);
        }
        if ctx.ballots.is_some() && ctx.voters.contains(agent_name) {
            ensure_tool_name(&mut config, CAST_VOTE_TOOL_NAME);
        }

        for tool in &ctx.global_tools {
            ensure_tool_name(&mut config, &tool.name());
        }
        if let Some(agent_tools) = ctx.agent_tools.get(agent_name) {
            for tool in agent_tools {
                let name = tool.name();
                if !ctx
                    .resume_only_tools
                    .contains(&(agent_name.to_string(), name.clone()))
                {
                    ensure_tool_name(&mut config, &name);
                }
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
    /// During a graceful stop's wrap-up only a completion verifier's
    /// `review` may start (of a completion made in the wrap-up).
    fn admit(
        &self,
        agent_name: &str,
        agent_run_id: Option<uuid::Uuid>,
        review: bool,
        ctx: &Arc<SharedContext>,
    ) -> Result<(), WorkflowError> {
        ctx.run_control.check_admission(review)?;
        // Under `Pause` a run over the budget starts, and is held at its
        // first request.
        if budget_cancels(ctx) && ctx.budget.is_exceeded() {
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        if budget_cancels(ctx) && ctx.budget.is_agent_exceeded(agent_name) {
            return Err(WorkflowError::AgentBudgetExceeded {
                agent: agent_name.to_string(),
                spent: ctx.budget.agent_spent(agent_name),
                limit: ctx.budget.agent_limit(agent_name).unwrap_or(0.0),
            });
        }
        if let Some(error) = plan_budget_error(ctx) {
            return Err(error);
        }
        fire_hooks_on_context(ctx, HookPoint::PreAgentSpawn, |hook| {
            hook.agent_name = Some(agent_name.to_string());
            hook.agent_run_id = agent_run_id;
        })
    }

    /// Prepares a new run of a registered agent and starts it. A run
    /// `reserved` is already marked running in the agent registry (a batch
    /// started together) and is abandoned if it can't start.
    pub(crate) fn start_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
        mut options: StartOptions,
    ) -> Result<StartedRun, WorkflowError> {
        let admission = if options.admitted {
            Ok(())
        } else {
            self.admit(agent_name, None, options.review, ctx)
        };
        // A subagent's record is on the board already (`options.reserved`
        // is its run id), and the hook says if it may start runs.
        let role = RunRole {
            depth: options.depth,
            subagent: options.subagent,
            starts: !options.subagent
                || options
                    .reserved
                    .is_none_or(|run_id| ctx.subagents.offers_start_tools(run_id)),
        };
        let prepared = admission.and_then(|()| {
            let registered;
            let base_config = match &options.config {
                Some(config) => config,
                None => {
                    registered = ctx
                        .registered_config(agent_name)
                        .ok_or_else(|| WorkflowError::AgentNotFound(agent_name.to_string()))?;
                    &registered
                }
            };
            // A configuration given at the start (a subagent's) is checked
            // here. The builder checked the workflow's own.
            crate::verification::refuse_text_endings(
                &ctx.completion_verifiers,
                agent_name,
                base_config,
            )
            .map_err(WorkflowError::BuilderInvalid)?;
            options.images = sendable_images(
                agent_name,
                base_config,
                std::mem::take(&mut options.images),
                None,
            )?;
            let workspace = match &options.workspace {
                Some(workspace) => super::options::run_workspace(&ctx.workspace, workspace)?,
                None => ctx.workspace.clone(),
            };
            let mut config = self.prepare_agent_config_for(agent_name, base_config, ctx, role);
            let mut registry = self.build_tool_registry(agent_name, ctx, role);
            let skill_prompt_context =
                Self::setup_skills(&mut config, &mut registry, &workspace, role)?;
            Ok((config, registry, skill_prompt_context, workspace))
        });
        let (config, registry, skill_prompt_context, workspace) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                if let Some(task_id) = options.reserved {
                    ctx.registry.abandon(task_id);
                }
                return Err(error);
            }
        };

        let mut task = Task::new(config, String::new(), workspace, None);
        task.parent = options.parent;
        task.reconfigured = options.host_config;
        task.initial_task_images = options.images;
        match options.reserved {
            Some(task_id) => task.id = task_id,
            None => ctx.registry.start(&[(task.id, agent_name)]),
        }
        let (messages, pending_message_ids) = self.pending_messages(agent_name, task.id, ctx);
        let notes = ctx
            .notes
            .initial_section(ctx, agent_name, task.id, options.review);
        let roster = roster_section(ctx, agent_name, &task, options.roster);
        task.initial_task_prompt = format!("{prompt}{messages}{notes}{roster}");
        Ok(self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            pending_message_ids,
            ctx,
            options.depth,
            options.driven,
            options.review,
        ))
    }

    /// Starts the agent runtime on `task` and registers the running agent
    /// with the message bus and the run's cancellation. `review` for a
    /// completion verifier's review (see [`StartOptions::review`]).
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
        review: bool,
    ) -> StartedRun {
        let load_skill_policy = task
            .agent_config
            .skills_config
            .as_ref()
            .and_then(|skills| skills.load_skill_interception.clone());
        let interceptor_config = self.build_interceptor_config(agent_name, load_skill_policy, ctx);
        let execution_middleware = self.build_execution_middleware(agent_name, ctx, depth);
        let agent_run_id = task.id;
        ctx.note_run_llm_config(agent_run_id, &task.agent_config.llm_config);
        ctx.registry
            .set_parent(agent_run_id, task.parent.map(|parent| parent.task_id));
        let started_task = task.clone();
        let host_decisions = HostDecisions::new(driven);
        let (stop_signal, launch) = ctx.run_control.begin_run(
            agent_run_id,
            review,
            task.parent.map(|parent| parent.task_id),
        );
        ctx.subagents.registered(ctx, agent_run_id);
        let mut runtime = AgentRuntime::new(self.llm_client.clone())
            .with_turn_context(Arc::new(crate::notes::WithNotes {
                inner: ActivityNotices {
                    context: ctx.clone(),
                    agent_name: agent_name.to_string(),
                    review,
                },
                context: ctx.clone(),
                agent_name: agent_name.to_string(),
                review,
            }))
            .with_host_decisions(host_decisions.clone())
            .with_stop_signal(stop_signal)
            .with_handover_hook(Arc::new(WorkflowHandover {
                context: ctx.clone(),
                agent_name: agent_name.to_string(),
                prompt_message_ids: pending_message_ids.clone(),
                review,
            }));
        if let Some(threshold) = ctx.handover_thresholds.get(agent_name) {
            runtime = runtime.with_handover_threshold(threshold.clone());
        }
        if let Some(sink) = &ctx.tool_image_sink {
            runtime = runtime.with_tool_image_sink(sink.clone());
        }
        #[cfg(any(unix, windows))]
        if let Some(jobs) = &ctx.background_jobs {
            runtime = runtime.with_background_jobs(jobs.clone());
        }
        // Saved as it runs, with the usage it has been charged for so far.
        let journal = self
            .persistence
            .clone()
            .filter(|persistence| persistence.journals_tasks());
        if let Some(journal) = &journal {
            journal.begin_live_run(agent_run_id, agent_name, task.usage_history.len());
            runtime = runtime.with_task_snapshots(Arc::new(crate::persistence::TaskJournal::new(
                journal.clone(),
                ctx.clone(),
            )));
        }
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
        ctx.control_stops.start(ctx);
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
            recovered: false,
            journal,
            launch,
        }
    }

    /// Run a single agent to completion, blocking until it finishes.
    pub async fn run_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        self.run_agent_with(agent_name, prompt, super::RunOptions::default(), ctx)
            .await
    }

    /// [`run_agent`](Self::run_agent) with `options`.
    pub async fn run_agent_with(
        &self,
        agent_name: &str,
        prompt: &str,
        options: super::RunOptions,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        let options = StartOptions {
            workspace: options.workspace,
            images: options.images,
            host_config: options.config.is_some(),
            config: options.config,
            ..StartOptions::default()
        };
        let run = self.start_agent(agent_name, prompt, ctx, options)?;
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// Run an agent inside the tool call of a run at `depth - 1`, `parent`.
    pub(crate) async fn run_nested(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
        parent: Option<arko_agents::TaskParent>,
    ) -> Result<AgentResult, WorkflowError> {
        let options = StartOptions {
            depth,
            parent,
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
    pub(crate) fn continue_nested<'a>(
        &'a self,
        previous: &'a AgentResult,
        prompt: &'a str,
        ctx: &'a Arc<SharedContext>,
        depth: u32,
        parent: Option<arko_agents::TaskParent>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.continue_at(previous, prompt.into(), ctx, depth, parent, false))
    }

    /// [`continue_nested`](Self::continue_nested) for a completion
    /// verifier's review, continued (a re-review, or its verdict restated).
    fn continue_review_nested<'a>(
        &'a self,
        previous: &'a AgentResult,
        prompt: &'a str,
        ctx: &'a Arc<SharedContext>,
        depth: u32,
        parent: arko_agents::TaskParent,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.continue_at(previous, prompt.into(), ctx, depth, Some(parent), true))
    }

    /// [`run_verifier`](Self::run_verifier), boxed as
    /// [`continue_nested`](Self::continue_nested) is.
    fn run_verifier_nested<'a>(
        &'a self,
        agent_name: &'a str,
        prompt: &'a str,
        ctx: &'a Arc<SharedContext>,
        depth: u32,
        parent: arko_agents::TaskParent,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.run_verifier(agent_name, prompt, ctx, depth, parent))
    }

    async fn run_verifier(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
        depth: u32,
        parent: arko_agents::TaskParent,
    ) -> Result<AgentResult, WorkflowError> {
        let options = StartOptions {
            roster: false,
            depth,
            parent: Some(parent),
            review: true,
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
    /// many run at once. Each is marked running when it starts. With
    /// fail-fast (`WorkflowBuilder::fail_fast`), the first failure stops the
    /// others, and agents not started yet don't start.
    pub async fn run_agents_parallel(
        &self,
        agents: Vec<(&str, String)>,
        ctx: &Arc<SharedContext>,
    ) -> Result<Vec<AgentResult>, WorkflowError> {
        ctx.run_control.check_admission(false)?;
        let limit = ctx.parallel.max_agents;
        // Each agent is admitted (PreAgentSpawn, budget, the run's stop)
        // before any is marked running; one refused is that agent's error.
        let admitted = agents
            .iter()
            .map(|(name, _)| self.admit(name, None, false, ctx))
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
                        ctx_clone.run_control.check_admission(false)?;
                        if budget_cancels(&ctx_clone) && ctx_clone.budget.is_exceeded() {
                            return Err(WorkflowError::BudgetExceeded {
                                spent: ctx_clone.budget.total_spent(),
                                limit: ctx_clone.budget.limit().unwrap_or(0.0),
                            });
                        }
                        if let Some(error) = plan_budget_error(&ctx_clone) {
                            return Err(error);
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
                    && !is_run_stop(error)
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
                    // An agent that paused, or that the host stopped alone,
                    // is not a failure: its recorded result is returned
                    // with the others.
                    let paused = match &error {
                        WorkflowError::UserInputRequired { task_id, .. }
                        | WorkflowError::AgentPaused { task_id, .. }
                        | WorkflowError::VerifierFailed { task_id, .. }
                        | WorkflowError::AgentStopped { task_id, .. } => ctx
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
    /// event processing). `options` set up the run (its working directory).
    pub async fn spawn_agent_with(
        &self,
        agent_name: &str,
        prompt: &str,
        options: super::RunOptions,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentHandle, WorkflowError> {
        // The caller chooses who drives it: `AgentHandle::wait` hands it to
        // the workflow.
        let options = StartOptions {
            driven: false,
            workspace: options.workspace,
            images: options.images,
            host_config: options.config.is_some(),
            config: options.config,
            ..StartOptions::default()
        };
        let run = self.start_agent(agent_name, prompt, ctx, options)?;
        Ok(AgentHandle::new(run, ctx.clone(), self.persistence.clone()))
    }

    /// Continue an agent run, as
    /// [`Workflow::continue_agent_with`](super::Workflow::continue_agent_with)
    /// says.
    pub async fn continue_agent_with(
        &self,
        previous: &AgentResult,
        continuation_prompt: &str,
        options: super::ContinueOptions,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        // A completion a verifier could not review (`VerifierFailed`): the
        // prompt is the agent's next message (a cut back replaces it).
        if options.cut_back_to.is_none()
            && previous.status == AgentRunStatus::Paused
            && matches!(
                previous.task()?.current_state,
                TaskState::AwaitingErrorRecoveryApproval {
                    cause: arko_agents::ErrorRecoveryCause::Middleware,
                    ..
                }
            )
        {
            if !options.images.is_empty() || options.config.is_some() || options.workspace.is_some()
            {
                return Err(WorkflowError::BuilderInvalid(
                    "a run whose completion a verifier could not review continues with its message alone. Continue it without images or a new configuration, then send them".to_string(),
                ));
            }
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
        let input = UserInput {
            text: continuation_prompt,
            images: options.images,
            config: options.config,
            cut_back_to: options.cut_back_to,
            workspace: options.workspace,
        };
        self.continue_at(previous, input, ctx, 0, None, false).await
    }

    /// Continue an agent run at `depth` (see [`StartedRun::depth`]), inside
    /// tool call `parent` (`None` for a run the host continues). `review`
    /// for a completion verifier's review (see [`StartOptions::review`]).
    async fn continue_at(
        &self,
        previous: &AgentResult,
        mut input: UserInput<'_>,
        ctx: &Arc<SharedContext>,
        depth: u32,
        parent: Option<arko_agents::TaskParent>,
        review: bool,
    ) -> Result<AgentResult, WorkflowError> {
        let continuation_prompt = input.text;
        let previous_task = previous.task()?;
        let resumable = match previous.status {
            // A cut back continues a run that isn't running, whatever its
            // status (`start_unless_running` below refuses a running one).
            _ if input.cut_back_to.is_some() => true,
            AgentRunStatus::Completed => {
                matches!(previous_task.current_state, TaskState::Completed { .. })
            }
            AgentRunStatus::WaitingForInput => matches!(
                previous_task.current_state,
                TaskState::AwaitingUserResponse { .. }
            ),
            _ => false,
        };
        if !resumable {
            return Err(WorkflowError::BuilderInvalid(
                "only a completed or user-input-suspended agent run, or one a completion verifier could not review, can be continued. Resume a paused one with resume_agent, recover an interrupted one with recover_agent".to_string(),
            ));
        }
        let agent_name = &previous.agent_name;
        let mut task = previous_task.clone();
        task.parent = parent;
        // What changed in the task, reported as the run goes on.
        let mut lifecycle = Vec::new();
        if let Some(index) = input.cut_back_to {
            let before = task.conversation.messages.len();
            task.cut_back(index).map_err(builder_invalid)?;
            lifecycle.push(AgentEvent::ConversationCutBack {
                message_index: index,
                removed_messages: before.saturating_sub(index),
            });
        }
        if let Some(workspace) = &input.workspace {
            let directory = super::options::run_workspace(&ctx.workspace, workspace)?;
            // Its tools work there, and its prompt (which may state the
            // directory) is written again. A Claude Code session stays where
            // it started, where Claude Code finds it.
            task.move_to(directory);
        }
        if let Some(config) = input.config {
            // A configuration given now is checked as one given at a start
            // is, and gets the workflow's tools as a new run's does.
            crate::verification::refuse_text_endings(
                &ctx.completion_verifiers,
                agent_name,
                &config,
            )
            .map_err(WorkflowError::BuilderInvalid)?;
            let prepared = self.prepare_agent_config_for(
                agent_name,
                &config,
                ctx,
                RunRole::of(ctx, depth, task.id),
            );
            task.reconfigure(prepared).map_err(builder_invalid)?;
            lifecycle.push(AgentEvent::Reconfigured {
                chat: task.agent_config.tool_config.chat,
                tools: task.agent_config.tool_config.offered_tool_names().len(),
            });
        }
        input.images = sendable_images(
            agent_name,
            &task.agent_config,
            std::mem::take(&mut input.images),
            Some(&task.conversation),
        )?;
        self.admit(agent_name, Some(previous.agent_run_id), review, ctx)?;
        let mut registry =
            self.build_tool_registry(agent_name, ctx, RunRole::of(ctx, depth, task.id));
        Self::restore_embedded_skills(&mut task, ctx, agent_name);
        let skill_prompt_context = Self::setup_skills(
            &mut task.agent_config,
            &mut registry,
            &task.current_working_directory,
            RunRole::of(ctx, depth, task.id),
        )?;
        // Continued once at a time.
        if !ctx.registry.start_unless_running(task.id, agent_name) {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent run {} is running already",
                task.id
            )));
        }
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
        // Cut back: the standing notes the removed messages gave it are
        // given again (one-time notes went with them, and a cut to the first
        // message gets its notes in its prompt). The subagents it started in
        // the removed messages are detached: their mail is withheld. The
        // mail of the others the removed messages held is given again.
        if let Some(index) = input.cut_back_to {
            if index > 0 {
                ctx.notes.cut_back(agent_name, &task, review);
            }
            let messages = &previous_task.conversation.messages;
            ctx.subagents
                .cut_back(ctx, &task, &messages[index.min(messages.len())..]);
        }
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
        // Cut back, it no longer holds what it read in the removed messages
        // (and can't tell which reads were kept): it reads again what it
        // needs, as after a handover.
        if input.cut_back_to.is_some()
            && let Some(tracker) = &ctx.resource_tracker
        {
            tracker.clear_actor(&actor);
        }
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
        let prepared = match (input.cut_back_to, &previous.status) {
            // Cut back to its first message: it starts over from the prompt
            // as a new run does, with the standing notes and its roster.
            (Some(0), _) => {
                let notes = ctx.notes.initial_section(ctx, agent_name, task.id, review);
                let roster = roster_section(ctx, agent_name, &task, true);
                task.initial_task_prompt =
                    format!("{continuation_prompt}{messages}{notes}{roster}{changed_notice}");
                task.initial_task_images = input.images;
                Ok(())
            }
            (Some(_), _) | (None, AgentRunStatus::Completed) => task
                .prepare_for_resumption_from_completed_with_images(
                    format!("{continuation_prompt}{messages}{changed_notice}"),
                    input.images,
                ),
            _ => task.prepare_for_resumption_with_user_response_and_images(
                format!("{continuation_prompt}{messages}{changed_notice}"),
                input.images,
            ),
        };
        if let Err(error) = prepared {
            ctx.registry.abandon(task.id);
            return Err(error.into());
        }
        for event in &lifecycle {
            publish_event(agent_name, task.id, event, ctx, self.persistence.as_ref()).await;
        }
        let run = self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            pending_message_ids,
            ctx,
            depth,
            true,
            review,
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
        self.resume_at(previous, resume, ctx, 0, None).await
    }

    /// [`resume_at`](Self::resume_at), boxed as
    /// [`continue_nested`](Self::continue_nested) is.
    pub(crate) fn resume_nested<'a>(
        &'a self,
        previous: &'a AgentResult,
        resume: AgentResume,
        ctx: &'a Arc<SharedContext>,
        depth: u32,
        parent: Option<arko_agents::TaskParent>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.resume_at(previous, resume, ctx, depth, parent))
    }

    /// Continue paused agent run `previous` at `depth` (see
    /// [`StartedRun::depth`]). With `parent`, inside that tool call (its
    /// parent link points there), else keeping its link.
    async fn resume_at(
        &self,
        previous: &AgentResult,
        resume: AgentResume,
        ctx: &Arc<SharedContext>,
        depth: u32,
        parent: Option<arko_agents::TaskParent>,
    ) -> Result<AgentResult, WorkflowError> {
        let (pause, previous_task) = match previous.status {
            AgentRunStatus::Paused => {
                let task = previous.task()?;
                AgentPause::of(&task.current_state).map(|pause| (pause, task))
            }
            _ => None,
        }
        .ok_or_else(|| {
            WorkflowError::BuilderInvalid(
                "only a paused agent run can be resumed. Continue a completed or input-waiting run with continue_agent".to_string(),
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
                let batch_id = match &previous_task.current_state {
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
            (AgentPause::ErrorRecovery { cause, .. }, AgentResume::ResolveRefusal(changes)) => {
                // A pause saved before causes were recorded (`Unknown`) may
                // be a refusal: the runtime checks there is refused input.
                if !matches!(
                    cause,
                    arko_agents::ErrorRecoveryCause::Refusal
                        | arko_agents::ErrorRecoveryCause::Unknown
                ) {
                    return Err(WorkflowError::BuilderInvalid(format!(
                        "ResolveRefusal answers a refusal. The agent is paused for another error ({}). Continue or fail it",
                        crate::policy::cause_name(cause)
                    )));
                }
                if changes.is_empty() {
                    return Err(WorkflowError::BuilderInvalid(
                        "ResolveRefusal needs at least one change to the refused input".to_string(),
                    ));
                }
                if previous_task.agent_config.runs_in_claude_code() {
                    return Err(WorkflowError::BuilderInvalid(
                        "a Claude Code agent's refused input can't be changed (Claude Code keeps its own conversation). Continue or fail it".to_string(),
                    ));
                }
                Decision::Command(UserCommand::ResolveRefusal { changes })
            }
            (pause, resume) => {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "{resume:?} doesn't answer an agent {pause}"
                )));
            }
        };
        let agent_name = &previous.agent_name;
        self.admit(agent_name, Some(previous.agent_run_id), false, ctx)?;
        let mut task = previous_task.clone();
        if parent.is_some() {
            task.parent = parent;
        }
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
        let mut registry =
            self.build_tool_registry(agent_name, ctx, RunRole::of(ctx, depth, task.id));
        Self::restore_embedded_skills(&mut task, ctx, agent_name);
        let skill_prompt_context = Self::setup_skills(
            &mut task.agent_config,
            &mut registry,
            &task.current_working_directory,
            RunRole::of(ctx, depth, task.id),
        )?;
        // Resumed once at a time.
        if !ctx.registry.start_unless_running(task.id, agent_name) {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent run {} is running already",
                task.id
            )));
        }
        // Resumed without a prompt: the messages it doesn't hold arrive as
        // context updates.
        let run = self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            Vec::new(),
            ctx,
            depth,
            true,
            false,
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

    /// Continue an agent run that was running when the workflow stopped
    /// (see [`Workflow::recover_agent`](super::Workflow::recover_agent)). A
    /// run started inside another run's tool call is recovered with that
    /// run, unless it is orphaned ([`is_orphaned`]). Then the host recovers
    /// it, and its parent link is released.
    pub async fn recover_agent(
        &self,
        previous: &AgentResult,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        let current = recorded_result(ctx, previous.agent_run_id);
        let current = current.as_ref().unwrap_or(previous);
        let mut recovery = Recovery::default();
        if current.info().parent.is_some() {
            refuse_unless_orphaned(ctx, current)?;
            // Nobody consumes its result but the host: a later stop doesn't
            // look for it in its parent's call.
            recovery.parent = Some(None);
        }
        // Boxed: the recovery future is large, and a host awaits it on
        // its own stack (a 2 MiB thread in debug builds).
        self.recover_nested(previous, ctx, recovery).await
    }

    /// Keeps the prompt cache of a run that isn't running warm (see
    /// [`Workflow::refresh_prompt_cache`](super::Workflow::refresh_prompt_cache)).
    pub(crate) async fn refresh_prompt_cache(
        &self,
        previous: &AgentResult,
        ctx: &Arc<SharedContext>,
    ) -> Result<arko_agents::CacheRefresh, WorkflowError> {
        if ctx.registry.is_running(previous.agent_run_id) {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent run {} is running (a run waiting in a host handler, for a tool approval or an answer, counts as running). Only a run that isn't running is refreshed",
                previous.agent_run_id
            )));
        }
        let current =
            recorded_result(ctx, previous.agent_run_id).unwrap_or_else(|| previous.clone());
        let mut task = current.task()?.clone();
        let agent_name = current.agent_name;
        // The depth its next request runs at, which decides its tools.
        let depth = match ctx.subagents.record(task.id) {
            Some(record) => record.depth,
            None if task.parent.is_some() => {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "agent run {} was started inside another run's tool call, which continues it. It isn't refreshed on its own",
                    previous.agent_run_id
                )));
            }
            None => 0,
        };
        if let Some(error) = budget_refusal(ctx, &agent_name) {
            return Err(error);
        }
        // The tools and skills the run is offered when it goes on, so the
        // request has its prefix.
        let mut registry =
            self.build_tool_registry(&agent_name, ctx, RunRole::of(ctx, depth, task.id));
        Self::restore_embedded_skills(&mut task, ctx, &agent_name);
        let skill_prompt_context = Self::setup_skills(
            &mut task.agent_config,
            &mut registry,
            &task.current_working_directory,
            RunRole::of(ctx, depth, task.id),
        )?;
        let refresh = AgentRuntime::new(self.llm_client.clone())
            .refresh_prompt_cache(task, Arc::new(registry), skill_prompt_context)
            .await?;
        charge_usage(
            &agent_name,
            previous.agent_run_id,
            &refresh.usage,
            ctx,
            self.persistence.as_ref(),
        );
        // Saved now: an idle workflow has no running agent to save it with.
        if let Some(persistence) = &self.persistence {
            let saved = if persistence.journals_tasks() {
                persistence.write_live_checkpoint(ctx, false).await
            } else {
                Box::pin(persistence.auto_checkpoint(ctx)).await
            };
            if let Err(error) = saved {
                log::warn!(
                    "Workflow '{}': could not save the cost of agent '{agent_name}''s prompt cache refresh: {error}. The next checkpoint saves it.",
                    ctx.workflow_name
                );
            }
        }
        Ok(refresh)
    }

    /// Continue a cancelled agent run (see
    /// [`Workflow::reopen_agent`](super::Workflow::reopen_agent)) where its
    /// task stopped, with `prompt` after [`REOPENED_NOTE`]. It is recovered
    /// as an interrupted run is (calls without results are settled in
    /// doubt, and the like), under its run identity. A run started inside
    /// another run's tool call becomes the host's. Its parent link is
    /// released. `options`' images go after `prompt`.
    pub async fn reopen_agent_with(
        &self,
        previous: &AgentResult,
        prompt: &str,
        options: super::ContinueOptions,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        if options.config.is_some() || options.cut_back_to.is_some() || options.workspace.is_some()
        {
            return Err(WorkflowError::BuilderInvalid(
                "a reopened run keeps its configuration and conversation. Cut it back or reconfigure it with continue_agent_with instead".to_string(),
            ));
        }
        let previous_task = previous.task()?;
        let images = sendable_images(
            &previous.agent_name,
            &previous_task.agent_config,
            options.images,
            Some(&previous_task.conversation),
        )?;
        let current =
            recorded_result(ctx, previous.agent_run_id).unwrap_or_else(|| previous.clone());
        if current.status != AgentRunStatus::Cancelled {
            return Err(WorkflowError::BuilderInvalid(format!(
                "only a cancelled agent run can be reopened. Agent run {} is {:?}",
                previous.agent_run_id, current.status
            )));
        }
        if is_superseded(&current) {
            return Err(WorkflowError::BuilderInvalid(format!(
                "agent run {} was superseded by a run that replaced it after a restart. It is not reopened",
                previous.agent_run_id
            )));
        }
        // As for `recover_agent`: a run its parent still settles is left to
        // it (a completion verifier's review is done again by a new one).
        if current.info().parent.is_some() {
            refuse_unless_orphaned(ctx, &current)?;
        }
        let message = reopened_message(prompt);
        let recovery = Recovery {
            parent: current.info().parent.is_some().then_some(None),
            message: Some(&message),
            images: &images,
            reopen: true,
            ..Recovery::default()
        };
        self.recover_nested(&current, ctx, recovery).await
    }

    /// [`recover_at`](Self::recover_at), boxed as
    /// [`continue_nested`](Self::continue_nested) is: for a run inside the
    /// tool call that recovers it.
    pub(crate) fn recover_nested<'a>(
        &'a self,
        previous: &'a AgentResult,
        ctx: &'a Arc<SharedContext>,
        recovery: Recovery<'a>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentResult, WorkflowError>> + Send + 'a>,
    > {
        Box::pin(self.recover_at(previous, ctx, recovery))
    }

    /// Continues interrupted agent run `previous` where its task was saved,
    /// as `recovery` says: the calls it was running are settled by their
    /// tools' recovery modes as it starts, and a wait it was saved in is
    /// answered again. The run is recovered once at a time: another
    /// recovery of it fails (or is waited for, see [`Recovery::wait`]), and
    /// a run whose recovery ended since `previous` was read isn't run
    /// again: the outcome it recorded is returned.
    async fn recover_at(
        &self,
        previous: &AgentResult,
        ctx: &Arc<SharedContext>,
        recovery: Recovery<'_>,
    ) -> Result<AgentResult, WorkflowError> {
        let agent_run_id = previous.agent_run_id;
        let running = || {
            WorkflowError::BuilderInvalid(format!("agent run {agent_run_id} is running already"))
        };
        // Whether another recovery of the run ended while this one waited:
        // its outcome is this one's.
        let mut waited = false;
        let (claim, current, registry, skill_prompt_context, mut task) = loop {
            // Its record now: a recovery that ended since `previous` was
            // read recorded its outcome.
            let current = recorded_result(ctx, agent_run_id).unwrap_or_else(|| previous.clone());
            let reopening = recovery.reopen && current.status == AgentRunStatus::Cancelled;
            if current.status != AgentRunStatus::Interrupted && !reopening {
                // Recovered (or resumed by an agent) since `previous` was
                // read: its outcome is this call's.
                if waited || previous.status == AgentRunStatus::Interrupted {
                    return run_outcome(current);
                }
                return Err(WorkflowError::BuilderInvalid(
                    "only an interrupted agent run can be recovered. Continue a completed or input-waiting run with continue_agent, resume a paused one with resume_agent".to_string(),
                ));
            }
            if let Some(recovering) = ctx.registry.recovery(agent_run_id) {
                if !recovery.wait {
                    return Err(running());
                }
                recovery_ended(recovering).await;
                waited = true;
                continue;
            }
            if ctx.registry.is_running(agent_run_id) {
                return Err(running());
            }
            let agent_name = &current.agent_name;
            self.admit(agent_name, Some(agent_run_id), false, ctx)?;
            let mut task = current.task()?.clone();
            let mut registry = self.build_tool_registry(
                agent_name,
                ctx,
                RunRole::of(ctx, recovery.depth, task.id),
            );
            Self::restore_embedded_skills(&mut task, ctx, agent_name);
            let skill_prompt_context = Self::setup_skills(
                &mut task.agent_config,
                &mut registry,
                &task.current_working_directory,
                RunRole::of(ctx, recovery.depth, task.id),
            )?;
            // Claimed before anything is recorded, so that a second recovery
            // of the same run records nothing twice.
            match ctx.registry.claim_recovery(agent_run_id, agent_name) {
                Ok(claim) => {
                    // A recovery that ended between the read above and the
                    // claim recorded the run's outcome.
                    if recorded_result(ctx, agent_run_id)
                        .is_some_and(|recorded| recorded.status != current.status)
                    {
                        ctx.registry.abandon(agent_run_id);
                        drop(claim);
                        waited = recovery.wait;
                        continue;
                    }
                    break (claim, current, registry, skill_prompt_context, task);
                }
                Err(Some(recovering)) if recovery.wait => {
                    recovery_ended(recovering).await;
                    waited = true;
                }
                Err(_) => return Err(running()),
            }
        };
        // Held until the run's outcome is recorded.
        let _claim = claim;
        let agent_name = &current.agent_name;
        if let Some(parent) = recovery.parent {
            task.parent = parent;
        }
        if let TaskState::Cancelled { reason } = &task.current_state {
            // Reopened: it runs on as an interrupted task does, its open
            // calls settled first.
            let reason = format!("Reopened after it was cancelled: {reason}");
            task.set_state(TaskState::Interrupted { reason });
        }
        // The `run_workflow_agent` calls the run resumes (see
        // `ToolRecovery::Resume`): open, without the host's own output or a
        // result already, and still offered. Their runs are recovered by
        // them; every other interrupted run inside its open calls is
        // superseded (a run inside a call answered already is orphaned, see
        // `is_orphaned`, and stays to be resumed).
        let resumed = if registry.get(RUN_WORKFLOW_AGENT_TOOL_NAME).is_some() {
            task.running_tool_calls
                .iter()
                .filter(|call| {
                    call.tool_name == RUN_WORKFLOW_AGENT_TOOL_NAME
                        && call.custom_output.is_none()
                        && call.output.is_none()
                })
                .map(|call| call.tool_call_id)
                .collect()
        } else {
            Vec::new()
        };
        self.supersede_children(&task, Some(&resumed), ctx).await;
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
        // The notes it was given after its task was saved, given again (it
        // is launched as an ordinary run, below).
        ctx.notes.recovered(agent_name, &task, false);
        ctx.subagents.recovered(&task);
        // Its caller's message goes with the task, as context taken and not
        // yet added: it is added at the first turn boundary, after the
        // results of the calls it was running (and is saved with the task
        // until then).
        if let Some(message) = recovery
            .message
            .filter(|message| !message.trim().is_empty())
        {
            task.pending_context_injections.push(message.to_string());
        }
        task.pending_context_images
            .extend(recovery.images.iter().enumerate().map(|(index, image)| {
                arko_agents::ContextImage::new(format!("Image {}:", index + 1), image.clone())
            }));
        // Recovered without a prompt: the messages it doesn't hold arrive as
        // context updates.
        let mut run = self.launch(
            agent_name,
            task,
            registry,
            skill_prompt_context,
            Vec::new(),
            ctx,
            recovery.depth,
            true,
            false,
        );
        run.recovered = true;
        // Messages queued meanwhile, and what changed since it read it, are
        // added at its next turn boundary.
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
        drive_to_result(run, ctx, self.persistence.as_ref()).await
    }

    /// Before interrupted run `task` is recovered: the interrupted runs
    /// that ran inside its open tool calls (recorded running, without a
    /// result), other than those inside the `run_workflow_agent` calls it
    /// resumes (`resumed`, which recover theirs), are recorded cancelled,
    /// superseded. They are the completion verifiers reviewing a
    /// `complete_task` call, which runs again and starts a new review, and
    /// runs inside calls settled in doubt or answered with the host's own
    /// output. A run inside a call answered before the stop is orphaned
    /// ([`is_orphaned`]): it stays, to be resumed. With `resumed` `None`
    /// (`task` is superseded itself: it resumes nothing), every interrupted
    /// run inside its calls goes, whatever call it ran in.
    fn supersede_children<'a>(
        &'a self,
        task: &'a Task,
        resumed: Option<&'a [uuid::Uuid]>,
        ctx: &'a Arc<SharedContext>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let superseded = ctx
                .agent_results
                .read()
                .iter()
                .filter(|result| {
                    result.status == AgentRunStatus::Interrupted
                        && result.info().parent.is_some_and(|parent| {
                            parent.task_id == task.id
                                && resumed.is_none_or(|resumed| {
                                    !resumed.contains(&parent.tool_call_id)
                                        && (holds_open_call(task, parent.tool_call_id)
                                            || is_verifier_agent(ctx, &result.agent_name))
                                })
                        })
                        && !ctx.registry.is_running(result.agent_run_id)
                })
                .cloned()
                .collect::<Vec<_>>();
            for child in superseded {
                // Interrupted, so its task was read with the checkpoint.
                let task = match child.task() {
                    Ok(task) => task,
                    Err(error) => {
                        log::warn!(
                            "Workflow '{}': agent run {} of agent '{}' can't be superseded: {error}",
                            ctx.workflow_name,
                            child.agent_run_id,
                            child.agent_name
                        );
                        continue;
                    }
                };
                self.supersede_children(task, None, ctx).await;
                let mut child_task = task.clone();
                child_task.set_state(TaskState::Cancelled {
                    reason: SUPERSEDED_AFTER_RESTART.to_string(),
                });
                if let Err(error) = finalize_agent_task(
                    child.agent_name.clone(),
                    child_task,
                    &[],
                    ctx,
                    self.persistence.as_ref(),
                )
                .await
                {
                    log::warn!(
                        "Workflow '{}': recording superseded agent run {} of agent '{}' failed: {error}",
                        ctx.workflow_name,
                        child.agent_run_id,
                        child.agent_name
                    );
                }
            }
        })
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
    /// The depth of this agent's run. Its verifiers run one deeper.
    depth: u32,
}

impl WorkflowExecutionMiddleware {
    /// Once the run is stopped, or the agent run `task_id`'s own stop
    /// cancels it, every tool call cancels its agent.
    fn stop_decision(&self, task_id: uuid::Uuid) -> Option<ToolExecutionDecision> {
        let reason = self
            .context
            .run_control
            .cancellation()
            .map(|cancellation| cancellation.cause.agent_reason())
            .or_else(|| self.context.run_control.run_cancel_reason(task_id))?;
        Some(ToolExecutionDecision::CancelTask { reason })
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
    /// and a new review runs. After that, the completion is neither accepted
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
                "Completion verifier '{}' could not review this completion ({reason}). Nothing was decided about your work. The task is paused until the caller decides how to go on.",
                verifier.name
            ),
            reason: format!(
                "Completion verifier '{}' could not review the completion: {reason}",
                verifier.name
            ),
        }
    }

    /// Reviews a completion (`complete_task` call `tool_call_id` with
    /// `summary`) with each verifier that applies. Its reviews run inside
    /// the call.
    async fn verify_completion(
        &self,
        execution: &ToolExecutionContext<'_>,
        tool_call_id: uuid::Uuid,
        summary: &str,
    ) -> Option<ToolExecutionDecision> {
        let parent = arko_agents::TaskParent {
            task_id: execution.task_id,
            tool_call_id,
        };
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
            // Only a completed review continues: one that was stopped (its
            // workflow stopped during a re-review, say) is replaced by a new
            // review, given the prior findings.
            let prior_result = prior_record
                .as_ref()
                .filter(|_| verifier.reuse_conversation)
                .and_then(|record| {
                    self.context
                        .agent_results
                        .read()
                        .iter()
                        .find(|result| {
                            result.agent_run_id == record.verifier_task_id
                                && result.status == AgentRunStatus::Completed
                        })
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
                    .continue_review_nested(
                        previous,
                        &prompt,
                        &self.context,
                        self.depth + 1,
                        parent,
                    )
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
                    crate::notes::without_notes(without_roster(execution.initial_task_prompt)),
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
                        parent,
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
                            .continue_review_nested(
                                &result,
                                &prompt,
                                &self.context,
                                self.depth + 1,
                                parent,
                            )
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
        if let Some(decision) = self.stop_decision(execution.task_id) {
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
            HookResult::Reject(message) => {
                return Ok(ToolExecutionDecision::Reject {
                    message_to_agent: message,
                });
            }
            // The call doesn't run, and the whole run stops.
            HookResult::Abort(reason) => {
                let error = abort_run(&self.context, &HookPoint::PreToolExecution, reason);
                return Ok(self.stop_decision(execution.task_id).unwrap_or_else(|| {
                    ToolExecutionDecision::CancelTask {
                        reason: error.to_string(),
                    }
                }));
            }
        }

        if let Some(tracker) = &self.context.resource_tracker
            && let Some((resource, true)) = tracker.resource_for_tool_in(
                call.tool_name,
                call.parameters,
                execution.current_working_directory,
            )
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

        // A subagent that started subagents isn't done while they run: its
        // report would reach its lead with their work still going on. A run
        // that is being stopped reports at once, whatever it still has.
        if call.tool_name != COMPLETE_TASK_TOOL_NAME {
            self.context.subagents.forget_refusals(execution.task_id);
        } else if !self.context.run_control.was_told(execution.task_id)
            && !self.context.run_control.is_run_stopping(execution.task_id)
            && !self.context.run_control.is_wrapping_up()
            && let Some(message) = self.context.subagents.unfinished_workers(execution.task_id)
        {
            return Ok(ToolExecutionDecision::Reject {
                message_to_agent: message,
            });
        }

        // A run told to wrap up reports its progress: its completion is not
        // reviewed (the run it is continued in after a restart is).
        if call.tool_name == COMPLETE_TASK_TOOL_NAME
            && !self.context.run_control.was_told(execution.task_id)
        {
            let summary = call
                .parameters
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(decision) = self
                .verify_completion(execution, call.tool_call_id, summary)
                .await
            {
                // A verifier stopped with the run decides nothing.
                return Ok(self.stop_decision(execution.task_id).unwrap_or(decision));
            }
        }

        Ok(self
            .stop_decision(execution.task_id)
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
            && let Some((resource, is_write)) = tracker.resource_for_tool_in(
                call.tool_name,
                call.parameters,
                execution.current_working_directory,
            )
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

    /// An error-recovery pause is decided by its cause whichever a runtime
    /// reports first: the state change that gives the cause, or the error
    /// needing a decision. A non-retryable cause fails the run under
    /// `Automatic` in both orders (it was answered with an unknown cause,
    /// returned to the caller, when the error came first).
    #[tokio::test]
    async fn a_recovery_pause_is_decided_by_its_cause_in_either_order() {
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
            .name("order-test")
            .workspace(std::env::current_dir().unwrap())
            .llm_client(Arc::new(LLMClient::new()))
            .agent("designer", config)
            .build()
            .unwrap();
        let pause = || {
            AgentEvent::TaskStateChanged(TaskState::AwaitingErrorRecoveryApproval {
                error_message: "401 refresh_token_invalidated".into(),
                cause: arko_agents::ErrorRecoveryCause::NonRetryable,
                error_kind: Some(arko_ai::ErrorKind::Authentication),
            })
        };
        let error = || AgentEvent::ErrorEncountered {
            error_message: "401 refresh_token_invalidated".into(),
            requires_approval_to_retry: true,
            tool_call_id: None,
            error_kind: None,
        };
        let decide_from = |paused: Option<arko_agents::ErrorRecoveryCause>,
                           events: Vec<AgentEvent>| {
            let context = workflow.context().clone();
            async move {
                let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(8);
                let (command_tx, mut command_rx) = tokio::sync::mpsc::channel(8);
                for event in events {
                    event_tx.send(event).await.unwrap();
                }
                drop(event_tx);
                let driven = drive_event_loop(
                    "designer",
                    uuid::Uuid::new_v4(),
                    &mut event_rx,
                    &command_tx,
                    &context,
                    None,
                    None,
                    false,
                    paused.map(|cause| (cause, None)),
                )
                .await;
                assert!(driven.is_ok(), "{driven:?}");
                let mut commands = Vec::new();
                while let Ok(command) = command_rx.try_recv() {
                    commands.push(command);
                }
                commands
            }
        };
        let decide = |events: Vec<AgentEvent>| decide_from(None, events);
        let rejected = |commands: &[UserCommand]| match commands {
            [
                UserCommand::RejectErrorRecovery {
                    message: Some(message),
                },
            ] => message.contains("can't succeed without a change to the request, a credential"),
            _ => false,
        };
        let state_first = decide(vec![pause(), error()]).await;
        let error_first = decide(vec![error(), pause()]).await;
        assert!(rejected(&state_first), "{state_first:?}");
        assert!(rejected(&error_first), "{error_first:?}");
        // A pause the agent is still in, reported again with an error alone,
        // is decided by that pause; an error the agent moved past without a
        // pause needs no decision.
        let again = decide(vec![pause(), error(), error()]).await;
        assert_eq!(again.len(), 2, "{again:?}");
        assert!(rejected(&again[1..]));
        let moved_on = decide(vec![
            error(),
            AgentEvent::TaskStateChanged(TaskState::InProgress),
        ])
        .await;
        assert!(moved_on.is_empty(), "{moved_on:?}");
        // Once the agent is past a pause, a new error waits for its own
        // pause rather than taking the old one's cause.
        let past = decide(vec![
            pause(),
            error(),
            AgentEvent::TaskStateChanged(TaskState::InProgress),
            error(),
        ])
        .await;
        assert_eq!(past.len(), 1, "{past:?}");
        // A run resumed at a pause starts in it: an error about it alone
        // (a refusal change the runtime can't apply) is decided by its
        // cause, and its own text is what the host is told.
        let resumed = decide_from(
            Some(arko_agents::ErrorRecoveryCause::NonRetryable),
            vec![AgentEvent::ErrorEncountered {
                error_message: "Could not resolve the refusal: no refused input".into(),
                requires_approval_to_retry: true,
                tool_call_id: None,
                error_kind: None,
            }],
        )
        .await;
        assert!(rejected(&resumed), "{resumed:?}");
        assert!(
            matches!(&resumed[..], [UserCommand::RejectErrorRecovery { message: Some(message) }]
                if message.contains("Could not resolve the refusal")),
            "{resumed:?}"
        );
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
                    cause: None,
                    error_kind: None,
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
                optional: false,
            }),
            global_sources: vec![],
            embedded: vec![],
            on_invalid_skill: InvalidSkillPolicy::Error,
            load_skill_interception: None,
            ..Default::default()
        });
        let mut registry = ToolRegistry::new();
        let context = WorkflowEngine::setup_skills(
            &mut config,
            &mut registry,
            &workspace,
            RunRole::default(),
        )
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
        assert!(
            WorkflowEngine::setup_skills(
                &mut config,
                &mut registry,
                &workspace,
                RunRole::default()
            )
            .is_err()
        );
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

    /// A distiller's target is the agent's registration name. Its display
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
        let task = TaskContext::new(
            uuid::Uuid::new_v4(),
            "Friendly Coder Display Name",
            "find the failing test",
            &workflow.context().workspace,
        );
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
    /// requested (and billed) again. An empty one is. The chain retries the
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
            let task = TaskContext::new(
                uuid::Uuid::new_v4(),
                "coder",
                "find the failing test",
                &workflow.context().workspace,
            );
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
    /// no conflict hook). Continuing it sends one message listing what
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

    /// Counts its calls. Replaces the result with `replacement`, if any.
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

    /// Always fails. Counts its calls.
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
        let task = TaskContext::new(uuid::Uuid::new_v4(), "coder", "task", &workspace);
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

#[cfg(test)]
mod prompt_tests {
    use super::*;

    #[test]
    fn the_task_part_drops_delivered_messages_and_the_roster() {
        let prompt = format!(
            "Draft it.{MESSAGES_HEADING}\n--- Message 1 from 'lead' ---\nhi\n--- End message ---\n{ROSTER_HEADING}- `lead` (lead role): running"
        );
        assert_eq!(task_part(&prompt), "Draft it.");
        assert_eq!(task_part("Draft it."), "Draft it.");
        // A heading the task itself uses is kept.
        let own = format!("Draft it.{MESSAGES_HEADING}none yet");
        assert_eq!(task_part(&own), own);
    }
}
