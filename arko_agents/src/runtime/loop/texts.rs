//! The texts the loop gives the model and the host: the results of calls it
//! didn't run or stopped, and its notes.

use super::*;

/// The result of a tool call made while a handover report is due.
pub(super) const HANDOVER_REPORT_DUE: &str = "This call was not run: your handover report is due. Reply with the report as plain text, without calling a tool (call complete_task instead if the task is done).";
/// [`HANDOVER_REPORT_DUE`] for an agent not offered `complete_task` (its
/// reply in text ends the task, once the report is in).
pub(super) const HANDOVER_REPORT_DUE_NO_COMPLETION: &str = "This call was not run: your handover report is due. Reply with the report as plain text, without calling a tool.";

/// The note after a reply to a report request that had no text.
pub(super) const NO_HANDOVER_REPORT: &str = "Your reply had no text, so no handover report was received. Reply with the handover report as plain text, without calling a tool.";

/// For an agent offered `handover_report`: the result of another tool call
/// made while the report is due.
pub(super) const HANDOVER_REPORT_TOOL_DUE: &str = "This call was not run: your handover report is due. Call handover_report with the report (or complete_task if the task is done).";
/// [`HANDOVER_REPORT_TOOL_DUE`] for an agent not offered `complete_task`.
pub(super) const HANDOVER_REPORT_TOOL_DUE_NO_COMPLETION: &str =
    "This call was not run: your handover report is due. Call handover_report with the report.";

/// For an agent offered `handover_report`: the note after a reply without a
/// tool call while the report is due.
pub(super) const NO_REPORT_TOOL_CALL: &str = "Your response did not call a tool, so no handover report was received. Call handover_report with the report (or complete_task if the task is done).";
/// [`NO_REPORT_TOOL_CALL`] for an agent not offered `complete_task`: its
/// reply ends the task only once the report is in.
pub(super) const NO_REPORT_TOOL_CALL_NO_COMPLETION: &str = "Your response did not call a tool, so no handover report was received, and the task is still open: your reply ends it only after the handover. Call handover_report with the report first.";

/// The results of `handover_report` calls: the report taken, no report
/// asked for, and a call without a report.
pub(super) const REPORT_RECEIVED: &str =
    "Handover report received. Your conversation continues from it in a fresh one.";
pub(super) const REPORT_NOT_REQUESTED: &str = "This call was not run: no handover report was asked for. Continue with your task: you will be asked for a report if your conversation needs to be handed over.";
pub(super) const EMPTY_REPORT_CALL: &str = "No handover report was received: the call had no report. Call handover_report again with the full report in its `report` parameter.";

/// The runtime's notes on a response it can't use, added to the
/// conversation before it asks again. See
/// [`AgentLoop::handle_model_output_failure`].
pub(super) const NO_NATIVE_TOOL_CALL: &str = "Your response did not call a tool, so the task is still open. Continue with your tools, or call complete_task with a summary if the task is finished.";
/// [`NO_NATIVE_TOOL_CALL`] for an agent whose reply in text ends the task
/// ([`AgentToolConfig::text_reply_ends_task`](crate::agent_config::AgentToolConfig::text_reply_ends_task)),
/// after a response with neither.
pub(super) const NO_NATIVE_TOOL_CALL_OR_REPLY: &str = "Your response had no text and no tool call, so the task is still open. Continue with your tools, or reply in plain text, without a tool call, if the task is finished: that reply ends the task.";
pub(super) const NO_TEXT_TOOL_CALL: &str = "Your response did not contain a tool call. Use one of your tools in the format the system prompt describes, or complete_task if the task is finished.";

/// The note after a reply in plain text that was sent to the user as a
/// message ([`AgentToolConfig::text_reply_as_message`](crate::agent_config::AgentToolConfig::text_reply_as_message)).
pub(super) const TEXT_REPLY_SENT: &str = "Your reply was sent to the user as a message. The task is still open: carry on with your tools, or call complete_task with a summary if the task is finished.";

/// `ToolRejected` reasons for calls a stream announced but never finished.
pub(super) const INTERRUPTED_CALL: &str =
    "The model's response was interrupted before this call finished; it was not run.";
pub(super) const FAILED_STREAM_CALL: &str =
    "The model's response failed before this call finished; it was not run.";
pub(super) const REFUSED_TURN_CALL: &str = "turn refused";
pub(super) const UNDELIVERED_CALL: &str =
    "The model's final response did not include this call; it was not run.";
pub(super) const UNREACHED_CALL: &str = "The turn ended before this call ran; it was not run.";
/// Why a call awaiting approval was not run, when its wait ended without a
/// decision. See [`AgentLoop::settle_interrupted_wait`]. For a task saved
/// that way by an older version.
pub(super) const UNDECIDED_CALL: &str =
    "The approval wait ended without a decision, so this call was not run.";
/// The result of a call that was running when the task stopped (its host
/// crashed, say), settled as [`ToolRecovery::InDoubt`] when the task runs
/// again.
pub(super) const RESTARTED_CALL: &str = "The task was restarted while this call was running (its host stopped), so it may have run partly or fully. Check its effects before calling it again.";
/// Added to the output-based results of a turn the task stopped in the
/// middle of, once they are settled when it runs again.
pub(super) const RESTARTED_TURN: &str = "The task was restarted while this turn's tool calls ran (its host stopped). A call of the turn without a result here was not run; call it again if it is still needed.";
/// When the host's command channel closed during the wait.
pub(super) const HOST_STOPPED_CALL: &str =
    "The approval wait ended because the host stopped responding, so this call was not run.";
/// Results of calls a tool error under `ToolErrorPolicy::FailTask` stopped.
pub(super) const TASK_FAILED_CALL: &str =
    "The task failed on an earlier tool error, so this call was not run.";

/// Results of calls a `CancelTask` command stopped. The command carries no
/// reason: it comes from the user or from a host such as a workflow
/// stopping its agents.
pub(super) const TASK_CANCELLED_CALL: &str = "The task was cancelled, so this call was not run.";
/// Results of calls a `SuspendTask` command left unrun.
pub(super) const TASK_SUSPENDED_CALL: &str = "The task was suspended before this call ran, so it was not run. Call it again if it is still needed.";
/// Results of calls that were running when `CancelTask` or `SuspendTask`
/// arrived. See [`AgentLoop::run_group`].
pub(super) const STOPPED_BY_CANCEL: &str = "The task was cancelled while this call was running, so the call was stopped before it finished.";
pub(super) const STOPPED_BY_SUSPEND: &str = "The task was suspended while this call was running, so the call was stopped before it finished. Call it again if it is still needed.";

/// The result of a call the host rejected without giving a reason.
pub(super) const REJECTED_BY_USER: &str = "Rejected by user";
/// A call of a batch the host decided without a decision for it: the
/// `ToolRejected` reason, and its result (native) or the reason in its
/// result (output-based).
pub(super) const NO_DECISION: &str = "No decision provided";
pub(super) const NO_DECISION_CALL: &str = "No decision provided, implicitly rejected";
pub(super) const NO_DECISION_TEXT_CALL: &str = "Implicitly rejected: No decision provided.";

/// The runtime's request to continue after a decision that leaves no
/// message of its own (a decision on a pause, say).
pub(super) const CONTINUE_AFTER_DECISION: &str =
    "Please continue with the task based on the previous actions.";
/// The runtime's request to continue after an output-based turn that left
/// no results (its calls returned nothing).
pub(super) const CONTINUE_WITH_A_TOOL: &str = "Please continue with the task or use the 'complete_task' tool if finished. Ensure you use a tool.";
/// The request to continue a conversation that ends on an assistant turn
/// (output kept from an interrupted stream, or a kept refusal).
pub(super) const CONTINUE_FROM_PARTIAL_TURN: &str =
    "Continue the task from where your previous response ended.";

/// Results of calls held behind a call awaiting approval (they never needed
/// approval themselves) when the wait ended without a decision.
pub(super) const HELD_CALL_NOT_RUN: &str = "An earlier call in this turn was still waiting for approval when the wait ended, so this call was not run. Call it again if it is still needed.";

/// The result of `complete_task` called after an earlier call of its turn
/// was rejected or failed: the model chose to finish before it saw those
/// results.
pub(super) const COMPLETION_AFTER_FAILURE: &str = "The task was not completed: an earlier call in this turn was rejected or failed (see its result). Review the results, and call complete_task again if the task is done.";

/// Results of calls that were running when the task ended another way (a
/// call beside them failed or completed it) as a stop arrived.
pub(super) const STOPPED_TASK_ENDED: &str =
    "The task ended while this call was running, so the call was stopped before it finished.";

/// The result text of a call a stop ended while it ran, given the task's
/// state once the stop is applied: a cancellation it made, a suspension, or
/// an end another call of the group had already brought about.
pub(super) fn stopped_call_reason(stop: ToolsStopped, state: &TaskState) -> &'static str {
    match (stop, state) {
        (_, TaskState::Cancelled { reason }) if reason == CANCELLED_BY_USER => STOPPED_BY_CANCEL,
        (_, state) if state.is_terminal() => STOPPED_TASK_ENDED,
        (ToolsStopped::Suspended, _) => STOPPED_BY_SUSPEND,
        (ToolsStopped::Cancelled, _) => STOPPED_BY_CANCEL,
    }
}

/// The name of a task's state, without its details (a timeout's reason
/// names the state the task waited in).
pub(super) fn state_name(state: &TaskState) -> &'static str {
    match state {
        TaskState::Pending => "Pending",
        TaskState::InProgress => "InProgress",
        TaskState::AwaitingToolApproval { .. } => "AwaitingToolApproval",
        TaskState::AwaitingBatchToolApproval { .. } => "AwaitingBatchToolApproval",
        TaskState::AwaitingUserResponse { .. } => "AwaitingUserResponse",
        TaskState::AwaitingErrorRecoveryApproval { .. } => "AwaitingErrorRecoveryApproval",
        TaskState::Interrupted { .. } => "Interrupted",
        TaskState::Completed { .. } => "Completed",
        TaskState::Failed { .. } => "Failed",
        TaskState::Cancelled { .. } => "Cancelled",
    }
}

/// When `user_command_timeout_seconds` ran out during the wait.
pub(super) fn approval_timed_out_call(seconds: u64) -> String {
    format!(
        "Human approval timed out after {seconds} seconds, so this call was not run. You can call it again if it is still needed."
    )
}
