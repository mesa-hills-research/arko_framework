//! How the managed driver answers an agent's tool-approval requests and
//! error-recovery pauses, and how a paused agent is continued.
//!
//! Both defaults keep a workflow fully autonomous, for unattended runs:
//! [`ApprovalPolicy::ApproveAll`] approves every tool call, and
//! [`ErrorRecoveryPolicy::Automatic`] recovers from errors that may pass
//! (outages, rate limits, plan usage limits) without an attempt limit. The
//! other variants hand decisions to the application.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use arko_agents::runtime::command::ToolApprovalDecision;
use arko_agents::{ApprovalReason, ErrorRecoveryCause, PendingToolCallInfo, RefusalResolution};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// How the managed driver answers an agent's tool-approval requests.
///
/// An agent asks for approval for a call its configuration doesn't
/// auto-approve (`default_auto_approve_tool_use`, per-tool
/// `auto_approve_tool_use`), and for a path outside the workspace under
/// `PathSafetyPolicy::RequireApproval`. Each request carries an
/// [`ApprovalReason`], which the tool hooks also receive
/// ([`HookContext::approval_reason`](crate::hooks::context::HookContext::approval_reason)).
#[derive(Clone, Default)]
pub enum ApprovalPolicy {
    /// **The default: approve every call**, whatever the agent's
    /// configuration asks, so a workflow runs without human input. The first
    /// call each agent makes to each tool for each reason is logged as a
    /// warning.
    #[default]
    ApproveAll,
    /// Honor the agent's configuration: a call it marks for approval pauses
    /// the agent, and the run call returns [`WorkflowError::AgentPaused`]
    /// with [`AgentPause::ToolApproval`]. Decide the calls and continue with
    /// [`Workflow::resume_agent`](crate::Workflow::resume_agent) and
    /// [`AgentResume::ToolApprovals`].
    ///
    /// [`WorkflowError::AgentPaused`]: crate::WorkflowError::AgentPaused
    RespectAgentConfig,
    /// Honor the agent's configuration, with this handler deciding each
    /// call it marks for approval as it is asked (the handler may wait on a
    /// person; a stopped run doesn't wait for it). A
    /// [`ToolApprovalOutcome::Pause`] pauses the agent as under
    /// `RespectAgentConfig`.
    Handler(Arc<dyn ToolApprovalHandler>),
}

impl fmt::Debug for ApprovalPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApproveAll => formatter.write_str("ApproveAll"),
            Self::RespectAgentConfig => formatter.write_str("RespectAgentConfig"),
            Self::Handler(_) => formatter.write_str("Handler(..)"),
        }
    }
}

/// Decides tool calls an agent's configuration marks for approval; see
/// [`ApprovalPolicy::Handler`].
#[async_trait]
pub trait ToolApprovalHandler: Send + Sync {
    async fn decide(&self, request: ToolApprovalRequest) -> ToolApprovalOutcome;
}

/// A tool call awaiting approval, as a [`ToolApprovalHandler`] receives it.
#[derive(Debug, Clone)]
pub struct ToolApprovalRequest {
    pub workflow_run_id: Uuid,
    pub agent_run_id: Uuid,
    /// The agent's registration name.
    pub agent_name: String,
    pub tool_call_id: Uuid,
    pub tool_name: String,
    pub parameters: HashMap<String, Value>,
    pub tool_description: String,
    /// Why the call awaits approval.
    pub reason: ApprovalReason,
}

/// A [`ToolApprovalHandler`]'s decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolApprovalOutcome {
    /// Run the call.
    Approve,
    /// Don't run it; the agent is told, with the reason when given.
    Reject { reason: Option<String> },
    /// Pause the agent and return the decision to the run's caller
    /// ([`AgentPause::ToolApproval`]).
    Pause,
}

/// How the managed driver answers an agent that pauses for error-recovery
/// approval.
///
/// The agent runtime first recovers on its own, as its `llm_error_policy`
/// and `llm_retry_config` say: a failed request is retried with backoff (a
/// rate limit after the provider's delay) until its attempts or elapsed
/// budget run out, and a request in flight is never cut off. Only then does
/// the agent pause, with an
/// [`ErrorRecoveryCause`](arko_agents::ErrorRecoveryCause), and this policy
/// decides. The driver never sends a request itself, so nothing is retried
/// twice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ErrorRecoveryPolicy {
    /// **The default: recover automatically, without an attempt limit.**
    ///
    /// - Retries that ran out (an outage, repeated rate limits): after a
    ///   wait that grows from 30 seconds to 15 minutes while the pauses
    ///   repeat without a model response in between (and at least the delay
    ///   the provider asked for), the failed request is sent again
    ///   unchanged, starting a new round of the agent's retries.
    /// - A plan usage limit: the driver waits until it resets (plus 5
    ///   seconds; when the limit is reported again right after, at least the
    ///   growing wait above), or, when the provider didn't say when, for the
    ///   growing wait, and sends the request again unchanged.
    ///
    /// Each wait gets up to 30 seconds (at most a quarter of it) added at
    /// random, so agents sharing a plan don't all resend at once.
    /// - Tool errors (under the agent's opt-in
    ///   `ToolErrorPolicy::PauseAndAwaitApproval`): the agent continues;
    ///   the calls' results report the errors.
    /// - A refusal: the agent's refusal settings apply first (the opt-in
    ///   `refusal_fallback` model has already been tried when configured);
    ///   a refusal that stands fails the run with a clear reason, since the
    ///   same input would be refused again.
    /// - Errors retrying can't fix (a request, credential or configuration
    ///   error) and responses the runtime can't use: the run fails with a
    ///   clear reason.
    /// - A pause by an opt-in monitor (the plan-usage threshold's `Pause`
    ///   action, the cache monitor's `pause_below`) stops for the
    ///   application as configured: the agent pauses and the run call
    ///   returns [`WorkflowError::AgentPaused`](crate::WorkflowError::AgentPaused).
    ///
    /// Continuing adds nothing to the conversation: the model is never told
    /// that a user approved.
    #[default]
    Automatic,
    /// As `Automatic`, for at most `max` automatic recoveries in each call
    /// that drives the agent (a run, continuation or resume); the next
    /// pause fails the run with a clear reason. `max: 0` recovers nothing
    /// automatically.
    Limited { max: u32 },
    /// Every error-recovery pause pauses the agent and returns
    /// [`WorkflowError::AgentPaused`](crate::WorkflowError::AgentPaused) with
    /// [`AgentPause::ErrorRecovery`]; continue with
    /// [`Workflow::resume_agent`](crate::Workflow::resume_agent).
    ReturnToCaller,
}

/// Why an agent paused, returned with
/// [`WorkflowError::AgentPaused`](crate::WorkflowError::AgentPaused). Its
/// task (in the recorded [`AgentResult`](crate::AgentResult), with status
/// [`AgentRunStatus::Paused`](crate::AgentRunStatus::Paused)) keeps the
/// state; continue it with
/// [`Workflow::resume_agent`](crate::Workflow::resume_agent).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentPause {
    /// Tool calls await approval; decide them with
    /// [`AgentResume::ToolApprovals`].
    ToolApproval { calls: Vec<PendingToolCallInfo> },
    /// The agent awaits error-recovery approval; continue with
    /// [`AgentResume::ContinueAfterError`], fail with
    /// [`AgentResume::FailAfterError`], or, for a refusal, change the input
    /// with [`AgentResume::ResolveRefusal`].
    ErrorRecovery {
        error: String,
        cause: ErrorRecoveryCause,
    },
}

impl AgentPause {
    /// The pause of a task in `state`, if it is paused.
    pub fn of(state: &arko_agents::TaskState) -> Option<Self> {
        use arko_agents::TaskState;
        match state {
            TaskState::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
                approval_reason,
            } => Some(Self::ToolApproval {
                calls: vec![PendingToolCallInfo {
                    tool_call_id: *tool_call_id,
                    tool_name: tool_name.clone(),
                    parameters: parameters.clone(),
                    tool_description: tool_description.clone(),
                    approval_reason: approval_reason.clone(),
                }],
            }),
            TaskState::AwaitingBatchToolApproval { pending_calls, .. } => {
                Some(Self::ToolApproval {
                    calls: pending_calls.clone(),
                })
            }
            TaskState::AwaitingErrorRecoveryApproval {
                error_message,
                cause,
            } => Some(Self::ErrorRecovery {
                error: error_message.clone(),
                cause: cause.clone(),
            }),
            _ => None,
        }
    }

    /// `"tool_approval"` or `"error_recovery"`, as hooks receive it.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ToolApproval { .. } => "tool_approval",
            Self::ErrorRecovery { .. } => "error_recovery",
        }
    }
}

impl fmt::Display for AgentPause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolApproval { calls } => {
                let tools = calls
                    .iter()
                    .map(|call| format!("{} ({})", call.tool_name, call.approval_reason.describe()))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(formatter, "awaiting approval of {tools}")
            }
            Self::ErrorRecovery { error, cause } => {
                write!(
                    formatter,
                    "awaiting error-recovery approval ({}): {error}",
                    cause_name(cause)
                )
            }
        }
    }
}

/// A cause's name, as hooks and messages give it.
pub(crate) fn cause_name(cause: &ErrorRecoveryCause) -> &'static str {
    match cause {
        ErrorRecoveryCause::Unknown => "unknown",
        ErrorRecoveryCause::RetriesExhausted { .. } => "retries_exhausted",
        ErrorRecoveryCause::NonRetryable => "non_retryable",
        ErrorRecoveryCause::UnusableResponses => "unusable_responses",
        ErrorRecoveryCause::UsageLimit { .. } => "usage_limit",
        ErrorRecoveryCause::Refusal => "refusal",
        ErrorRecoveryCause::ToolError => "tool_error",
        ErrorRecoveryCause::Monitor => "monitor",
    }
}

/// How [`Workflow::resume_agent`](crate::Workflow::resume_agent) continues a
/// paused agent.
#[derive(Debug, Clone)]
pub enum AgentResume {
    /// Decide the calls of an [`AgentPause::ToolApproval`]; a call without
    /// a decision is rejected.
    ToolApprovals(HashMap<Uuid, ToolApprovalDecision>),
    /// Continue after an [`AgentPause::ErrorRecovery`]: without a note the
    /// failed request is sent again unchanged; a note is added for the
    /// model first.
    ContinueAfterError { note: Option<String> },
    /// Fail the task, with `reason` when given.
    FailAfterError { reason: Option<String> },
    /// Change the refused input of a refusal pause and continue; see
    /// `arko_agents::Task::resolve_refusal`.
    ResolveRefusal(Vec<RefusalResolution>),
}

/// What the driver does with an error-recovery pause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecoveryAction {
    /// Wait this long, then send the request again unchanged.
    ContinueAfter(Duration),
    /// Fail the task with this reason.
    Fail(String),
    /// Pause the agent for the run's caller.
    ReturnToCaller,
}

/// The waits of automatic recovery; tests shorten them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecoveryWaits {
    /// The first wait after retries ran out.
    pub initial: Duration,
    /// The longest.
    pub max: Duration,
    /// Added to a usage limit's reset.
    pub reset_margin: Duration,
    /// Up to this much more is added at random to each wait (at most a
    /// quarter of it), so agents sharing a plan don't all resend at once.
    pub max_jitter: Duration,
}

impl Default for RecoveryWaits {
    fn default() -> Self {
        Self {
            initial: Duration::from_secs(30),
            max: Duration::from_secs(15 * 60),
            reset_margin: Duration::from_secs(5),
            max_jitter: Duration::from_secs(30),
        }
    }
}

impl RecoveryWaits {
    /// The wait before the `repeat`-th continuation (from 1) in a row
    /// without a model response in between.
    pub fn backoff(&self, repeat: u32) -> Duration {
        let doublings = repeat.saturating_sub(1).min(16);
        self.initial
            .saturating_mul(1_u32 << doublings)
            .min(self.max)
    }

    /// `wait` plus its jitter: a `sample` share of the smaller of
    /// [`max_jitter`](Self::max_jitter) and a quarter of `wait`.
    pub fn with_jitter(&self, wait: Duration, sample: u64) -> Duration {
        let spread = self.max_jitter.min(wait / 4);
        let spread_ms = u64::try_from(spread.as_millis()).unwrap_or(u64::MAX);
        wait.saturating_add(Duration::from_millis(
            sample.checked_rem(spread_ms.saturating_add(1)).unwrap_or(0),
        ))
    }
}

impl ErrorRecoveryPolicy {
    /// What to do with a pause for `cause` (`error` is its message), after
    /// `recovered` automatic recoveries in this drive and `repeat` pauses
    /// in a row (this one included) without a model response between.
    pub(crate) fn action(
        &self,
        cause: &ErrorRecoveryCause,
        error: &str,
        recovered: u32,
        repeat: u32,
        waits: &RecoveryWaits,
        now_unix_ms: u64,
    ) -> RecoveryAction {
        let limit = match self {
            Self::ReturnToCaller => return RecoveryAction::ReturnToCaller,
            Self::Automatic => None,
            Self::Limited { max } => Some(*max),
        };
        // After the first pause of a series the growing wait applies too, so
        // a provider that keeps reporting a reset or delay already passed
        // (or one too short) is not asked again every few seconds.
        let at_least_backoff = |wait: Duration| {
            if repeat > 1 {
                wait.max(waits.backoff(repeat))
            } else {
                wait
            }
        };
        let wait = match cause {
            ErrorRecoveryCause::RetriesExhausted {
                retry_after_ms: Some(delay),
            } => Duration::from_millis(*delay).max(waits.backoff(repeat)),
            ErrorRecoveryCause::RetriesExhausted {
                retry_after_ms: None,
            } => waits.backoff(repeat),
            ErrorRecoveryCause::UsageLimit {
                resets_at_unix_ms: Some(resets_at),
            } => at_least_backoff(
                Duration::from_millis(resets_at.saturating_sub(now_unix_ms))
                    .saturating_add(waits.reset_margin),
            ),
            ErrorRecoveryCause::UsageLimit {
                resets_at_unix_ms: None,
            } => waits.backoff(repeat),
            ErrorRecoveryCause::ToolError => Duration::ZERO,
            ErrorRecoveryCause::NonRetryable => {
                return RecoveryAction::Fail(format!(
                    "The model request can't succeed without a change to the request, a credential, or the configuration, so the workflow did not retry it: {error}"
                ));
            }
            ErrorRecoveryCause::Refusal => {
                return RecoveryAction::Fail(format!(
                    "The model refused, and the workflow does not send a refused input again (use ErrorRecoveryPolicy::ReturnToCaller to change the input and continue): {error}"
                ));
            }
            ErrorRecoveryCause::UnusableResponses => {
                return RecoveryAction::Fail(format!(
                    "The model's responses could not be used: {error}"
                ));
            }
            // A monitor's pause stops for the application, as configured;
            // a cause not recorded is left to it too.
            ErrorRecoveryCause::Monitor | ErrorRecoveryCause::Unknown => {
                return RecoveryAction::ReturnToCaller;
            }
        };
        if let Some(max) = limit
            && recovered >= max
        {
            return RecoveryAction::Fail(format!(
                "The workflow's error recovery limit ({max} automatic recoveries) was reached: {error}"
            ));
        }
        RecoveryAction::ContinueAfter(wait)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn waits() -> RecoveryWaits {
        RecoveryWaits {
            max_jitter: Duration::ZERO,
            ..RecoveryWaits::default()
        }
    }

    /// A usage limit reported again right after its reset (a reset already
    /// past, or reported wrongly) is not resent every few seconds: from the
    /// second pause in a row, the growing wait applies too.
    #[test]
    fn a_usage_limit_reported_again_waits_at_least_the_growing_wait() {
        let waits = waits();
        let now = 1_000_000;
        let stale = ErrorRecoveryCause::UsageLimit {
            resets_at_unix_ms: Some(now + 1_000),
        };
        let wait =
            |repeat| ErrorRecoveryPolicy::Automatic.action(&stale, "limit", 0, repeat, &waits, now);
        assert_eq!(
            wait(1),
            RecoveryAction::ContinueAfter(Duration::from_secs(6))
        );
        assert_eq!(
            wait(2),
            RecoveryAction::ContinueAfter(Duration::from_secs(60))
        );
        assert_eq!(
            wait(9),
            RecoveryAction::ContinueAfter(Duration::from_secs(900))
        );
    }

    #[test]
    fn jitter_adds_at_most_a_quarter_of_the_wait_and_30_seconds() {
        let waits = RecoveryWaits::default();
        for (wait, spread) in [
            (Duration::from_secs(20), Duration::from_secs(5)),
            (Duration::from_secs(3_600), Duration::from_secs(30)),
            (Duration::ZERO, Duration::ZERO),
        ] {
            for sample in [0, 1, 7_777, u64::MAX] {
                let jittered = waits.with_jitter(wait, sample);
                assert!(
                    jittered >= wait && jittered <= wait + spread,
                    "{wait:?} {sample}"
                );
            }
        }
    }

    #[test]
    fn automatic_recovery_waits_grow_to_the_cap() {
        let waits = waits();
        assert_eq!(waits.backoff(1), Duration::from_secs(30));
        assert_eq!(waits.backoff(2), Duration::from_secs(60));
        assert_eq!(waits.backoff(5), Duration::from_secs(480));
        assert_eq!(waits.backoff(6), Duration::from_secs(900));
        assert_eq!(waits.backoff(u32::MAX), Duration::from_secs(900));
    }

    /// Each cause under each policy: `Automatic` recovers from what may
    /// pass, without a limit; `Limited` the same, up to its count;
    /// `ReturnToCaller` returns every pause.
    #[test]
    fn each_cause_gets_its_action_under_each_policy() {
        let waits = waits();
        let now = 1_000_000;
        let action = |policy: &ErrorRecoveryPolicy, cause: ErrorRecoveryCause, recovered| {
            policy.action(&cause, "boom", recovered, 1, &waits, now)
        };
        let automatic = ErrorRecoveryPolicy::Automatic;
        let exhausted = |retry_after_ms| ErrorRecoveryCause::RetriesExhausted { retry_after_ms };
        assert_eq!(
            action(&automatic, exhausted(None), 1_000),
            RecoveryAction::ContinueAfter(Duration::from_secs(30))
        );
        // The provider's longer delay is waited out.
        assert_eq!(
            action(&automatic, exhausted(Some(600_000)), 0),
            RecoveryAction::ContinueAfter(Duration::from_secs(600))
        );
        assert_eq!(
            action(&automatic, exhausted(Some(1_000)), 0),
            RecoveryAction::ContinueAfter(Duration::from_secs(30))
        );
        assert_eq!(
            action(
                &automatic,
                ErrorRecoveryCause::UsageLimit {
                    resets_at_unix_ms: Some(now + 3_600_000)
                },
                0
            ),
            RecoveryAction::ContinueAfter(Duration::from_secs(3_605))
        );
        // A reset already past: the margin alone.
        assert_eq!(
            action(
                &automatic,
                ErrorRecoveryCause::UsageLimit {
                    resets_at_unix_ms: Some(now - 10)
                },
                0
            ),
            RecoveryAction::ContinueAfter(Duration::from_secs(5))
        );
        assert_eq!(
            action(
                &automatic,
                ErrorRecoveryCause::UsageLimit {
                    resets_at_unix_ms: None
                },
                0
            ),
            RecoveryAction::ContinueAfter(Duration::from_secs(30))
        );
        assert_eq!(
            action(&automatic, ErrorRecoveryCause::ToolError, 0),
            RecoveryAction::ContinueAfter(Duration::ZERO)
        );
        for cause in [
            ErrorRecoveryCause::NonRetryable,
            ErrorRecoveryCause::Refusal,
            ErrorRecoveryCause::UnusableResponses,
        ] {
            assert!(
                matches!(action(&automatic, cause.clone(), 0), RecoveryAction::Fail(reason) if reason.ends_with(": boom")),
                "{cause:?}"
            );
        }
        for cause in [ErrorRecoveryCause::Monitor, ErrorRecoveryCause::Unknown] {
            assert_eq!(action(&automatic, cause, 0), RecoveryAction::ReturnToCaller);
        }

        let limited = ErrorRecoveryPolicy::Limited { max: 2 };
        assert!(matches!(
            action(&limited, exhausted(None), 1),
            RecoveryAction::ContinueAfter(_)
        ));
        assert!(matches!(
            action(&limited, exhausted(None), 2),
            RecoveryAction::Fail(reason) if reason.contains("limit (2 automatic recoveries)")
        ));
        assert!(matches!(
            action(
                &ErrorRecoveryPolicy::Limited { max: 0 },
                ErrorRecoveryCause::ToolError,
                0
            ),
            RecoveryAction::Fail(_)
        ));
        assert_eq!(
            action(&limited, ErrorRecoveryCause::Monitor, 0),
            RecoveryAction::ReturnToCaller
        );

        for cause in [
            exhausted(None),
            ErrorRecoveryCause::NonRetryable,
            ErrorRecoveryCause::ToolError,
            ErrorRecoveryCause::Refusal,
        ] {
            assert_eq!(
                action(&ErrorRecoveryPolicy::ReturnToCaller, cause, 0),
                RecoveryAction::ReturnToCaller
            );
        }
    }
}
