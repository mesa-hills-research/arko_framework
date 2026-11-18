//! Calls a tool makes from inside its own call, such as code that calls the
//! agent's tools. Each goes through the checks a call of the model's gets
//! (see [`NestedCalls`]), in both runtimes, since both run calls with
//! [`run_tool`](super::tool_run::run_tool).

use super::admission::{self, PathVerdict};
use super::event::{AgentEvent, NESTED_OUTPUT_BYTES, NestedCallOutcome};
use super::tool_run::{self, RunEnvironment, RunOutcome, ToolRun};
use crate::agent_config::AgentToolConfig;
use crate::handover::HANDOVER_REPORT_TOOL_NAME;
use crate::stop::StopSignal;
use crate::tool::builtins::{
    ASK_USER_TOOL_NAME, CLOSE_QUESTIONS_TOOL_NAME, COMPLETE_TASK_TOOL_NAME,
};
use crate::tool::{ToolBackground, ToolOutput, ToolRegistry};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;
use uuid::Uuid;

/// The tools no call made from inside a call may call, since they are the
/// turn's own protocol. The calling tool itself is refused too.
pub const NEVER_NESTED: [&str; 4] = [
    COMPLETE_TASK_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    CLOSE_QUESTIONS_TOOL_NAME,
    HANDOVER_REPORT_TOOL_NAME,
];

/// What the checks of a nested call need beside the run's environment.
pub(super) struct Gate {
    /// The tools the task's configuration offers, the only ones that run.
    pub registry: Arc<ToolRegistry>,
    /// The agent's tool configuration, for its path and approval policies
    /// and its time limits.
    pub tool_config: AgentToolConfig,
    /// The task's event channel, for the nested calls' events.
    pub events: mpsc::Sender<AgentEvent>,
}

/// How the task stops at the outer call once it returns, because the
/// middleware decided so at a call it made.
#[derive(Debug, Clone)]
pub(super) enum Halt {
    /// The task fails, or with `cancel` is cancelled.
    Stop { reason: String, cancel: bool },
    /// The task pauses for its host's decision, with the outer call's
    /// result kept.
    Pause { reason: String },
}

/// How a call made from inside a call ended.
#[derive(Debug)]
pub enum NestedOutcome {
    /// The tool ran and returned.
    Finished(ToolOutput),
    /// The tool ran and failed, with the error's text.
    Failed(String),
    /// A check refused the call, with why, and the tool didn't run. The
    /// checks are the agent's tools, the path and approval policies (a call
    /// that would wait for approval is refused), the execution middleware,
    /// and checks inside the tool ([`ToolError::Refused`](crate::tool::ToolError::Refused)).
    Refused(String),
    /// The middleware failed, cancelled or paused the task at the call, with
    /// its message. The task does the same at the outer call once that
    /// returns, so the outer call should return now. Later calls are
    /// answered this way too.
    Halted(String),
}

/// Calls the agent's tools from inside a call, as
/// [`TaskContext::nested`](crate::tool::TaskContext::nested) gives it to
/// the tool being run. Each call is checked as a call of the model's is,
/// and the checks are shared with the native loop's.
///
/// 1. The tool is one the task's configuration offers, and not one of
///    [`NEVER_NESTED`] or the calling tool itself.
/// 2. Its paths pass the path policy, and it needs no approval. A call
///    that would wait for approval (the tool isn't auto-approved, or a path
///    outside the workspace asks for it) is refused at once, with a message
///    to call the tool directly.
/// 3. The execution middleware's `before` and `after` see it, so every hook
///    judges it. A decision that fails, cancels or pauses the task at it
///    does so at the outer call.
/// 4. It runs under its tool's own time limit, with a [`TaskContext`](crate::tool::TaskContext) of
///    its own, a new call ID and the outer call's stop signal. It doesn't
///    move to the background, and it gets no `nested` of its own.
/// 5. Calls run one at a time, in the order they were made.
/// 6. Each emits [`AgentEvent::NestedToolCallStarted`] and
///    [`AgentEvent::NestedToolCallFinished`], naming the outer call. A
///    nested call has no `ToolCallFormatted` or `ToolResult`, so nothing
///    that counts or shows the model's calls takes it for one, and its
///    failure doesn't count under the tool-error policy.
///
/// Dropping a call's future stops it, as dropping the outer call's does.
#[derive(Clone)]
pub struct NestedCalls {
    environment: Arc<RunEnvironment>,
    gate: Arc<Gate>,
    outer_tool_call_id: Uuid,
    outer_tool_name: String,
    stop: StopSignal,
    /// Held while a call runs, so calls run one at a time in order.
    order: Arc<tokio::sync::Mutex<()>>,
    /// The first stop or pause a call met, for the outer call.
    halt: Arc<OnceLock<Halt>>,
}

impl std::fmt::Debug for NestedCalls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NestedCalls")
            .field("outer_tool_call_id", &self.outer_tool_call_id)
            .field("outer_tool_name", &self.outer_tool_name)
            .finish_non_exhaustive()
    }
}

impl NestedCalls {
    /// The calls of the outer call `outer_tool_call_id` of `outer_tool_name`,
    /// which stop with `stop`.
    pub(super) fn new(
        environment: Arc<RunEnvironment>,
        gate: Arc<Gate>,
        outer_tool_call_id: Uuid,
        outer_tool_name: &str,
        stop: StopSignal,
    ) -> Self {
        Self {
            environment,
            gate,
            outer_tool_call_id,
            outer_tool_name: outer_tool_name.to_string(),
            stop,
            order: Arc::default(),
            halt: Arc::default(),
        }
    }

    /// Where the outer call finds a stop or pause its calls met.
    pub(super) fn halt(&self) -> Arc<OnceLock<Halt>> {
        self.halt.clone()
    }

    /// The ID of the call these calls are made from.
    pub fn outer_tool_call_id(&self) -> Uuid {
        self.outer_tool_call_id
    }

    /// Whether `tool_name` is a tool these calls may reach, meaning one the
    /// task's configuration offers, not one of [`NEVER_NESTED`] and not the
    /// calling tool. It says nothing of a call's paths, approval or hooks.
    pub fn may_call(&self, tool_name: &str) -> bool {
        self.refusal(tool_name).is_none()
    }

    /// Calls `tool_name` with `params`, after the calls made before it end.
    pub async fn call(&self, tool_name: &str, params: HashMap<String, Value>) -> NestedOutcome {
        let _turn = self.order.lock().await;
        let tool_call_id = Uuid::new_v4();
        self.emit(AgentEvent::NestedToolCallStarted {
            outer_tool_call_id: self.outer_tool_call_id,
            tool_call_id,
            tool_name: tool_name.to_string(),
            parameters: params.clone(),
        })
        .await;
        // A call stopped before its end still ends in the events.
        let mut unfinished = Unfinished {
            events: Some(self.gate.events.clone()),
            outer_tool_call_id: self.outer_tool_call_id,
            tool_call_id,
            tool_name,
        };
        let outcome = self.run(tool_call_id, tool_name, params).await;
        unfinished.events = None;
        let (kind, text) = match &outcome {
            NestedOutcome::Finished(output) => (NestedCallOutcome::Finished, output.text.as_str()),
            NestedOutcome::Failed(text) => (NestedCallOutcome::Failed, text.as_str()),
            NestedOutcome::Refused(text) => (NestedCallOutcome::Refused, text.as_str()),
            NestedOutcome::Halted(text) => (NestedCallOutcome::Halted, text.as_str()),
        };
        self.emit(finished(
            self.outer_tool_call_id,
            tool_call_id,
            tool_name,
            kind,
            text,
        ))
        .await;
        outcome
    }

    /// Why the gate refuses any call of `tool_name`, whatever its arguments.
    fn refusal(&self, tool_name: &str) -> Option<String> {
        if NEVER_NESTED.contains(&tool_name) || tool_name == self.outer_tool_name {
            return Some(format!(
                "{tool_name} can't be called from inside a call of {}.",
                self.outer_tool_name
            ));
        }
        if self.gate.registry.get(tool_name).is_none() {
            return Some(format!("Tool '{tool_name}' is not available."));
        }
        None
    }

    async fn run(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        mut params: HashMap<String, Value>,
    ) -> NestedOutcome {
        if let Some(halt) = self.halt.get() {
            return NestedOutcome::Halted(halt.reason().to_string());
        }
        if let Some(refusal) = self.refusal(tool_name) {
            return NestedOutcome::Refused(refusal);
        }
        let Some(tool) = self.gate.registry.get(tool_name) else {
            return NestedOutcome::Refused(format!("Tool '{tool_name}' is not available."));
        };
        let config = &self.gate.tool_config;
        let definition = tool.definition();
        let needs_approval = match admission::path_verdict(
            config,
            tool_name,
            &definition,
            &params,
            &self.environment.working_directory,
        ) {
            PathVerdict::Allowed => !config.auto_approves(tool_name),
            PathVerdict::NeedsApproval(_) => true,
            PathVerdict::Outside(param_name) => {
                return NestedOutcome::Refused(format!(
                    "Path for parameter '{param_name}' is outside the workspace."
                ));
            }
            PathVerdict::Invalid(reason) => return NestedOutcome::Refused(reason),
        };
        if needs_approval {
            return NestedOutcome::Refused(format!(
                "This call of {tool_name} needs the user's approval, which a call from inside {} can't wait for. Call {tool_name} directly instead.",
                self.outer_tool_name
            ));
        }
        crate::tool::builtins::read_file::ignore_line_ranges(tool.as_ref(), config, &mut params);
        let run = ToolRun {
            tool_call_id,
            tool_name: tool_name.to_string(),
            timeout: config.tool_timeout(tool_name, tool.uses_host_timeout()),
            custom_output: None,
            interception: None,
            parallel_safe: false,
            interruptible: true,
            background: ToolBackground::Never,
            tool,
            params,
            call_id: None,
            message_to_agent: None,
            recovering: false,
        };
        let outcome = tool_run::run_nested(self.environment.clone(), run, self.stop.clone()).await;
        let (halt, text) = match outcome {
            RunOutcome::Finished { output, .. } => return NestedOutcome::Finished(output),
            // (A nested call never moves to the background.)
            RunOutcome::Moved { text } => return NestedOutcome::Finished(ToolOutput::text(text)),
            RunOutcome::Failed(crate::tool::ToolError::Refused(message)) => {
                return NestedOutcome::Refused(message);
            }
            RunOutcome::Failed(error) => return NestedOutcome::Failed(error.to_string()),
            RunOutcome::Rejected(message) => return NestedOutcome::Refused(message),
            RunOutcome::Stop { reason, cancel } => (
                Halt::Stop {
                    reason: reason.clone(),
                    cancel,
                },
                reason,
            ),
            RunOutcome::Pause { message, reason } => (Halt::Pause { reason }, message),
            RunOutcome::MiddlewareErrorBefore { reason } => (
                Halt::Pause {
                    reason: reason.clone(),
                },
                reason,
            ),
            RunOutcome::MiddlewareErrorAfter { reason, .. } => (
                Halt::Pause {
                    reason: reason.clone(),
                },
                reason,
            ),
        };
        let _ = self.halt.set(halt);
        NestedOutcome::Halted(text)
    }

    async fn emit(&self, event: AgentEvent) {
        // The task's consumer is gone only once the task ended.
        let _ = self.gate.events.send(event).await;
    }
}

impl Halt {
    fn reason(&self) -> &str {
        match self {
            Halt::Stop { reason, .. } | Halt::Pause { reason } => reason,
        }
    }
}

/// The end event of a nested call, with at most [`NESTED_OUTPUT_BYTES`] of
/// its result.
fn finished(
    outer_tool_call_id: Uuid,
    tool_call_id: Uuid,
    tool_name: &str,
    outcome: NestedCallOutcome,
    text: &str,
) -> AgentEvent {
    let mut cut = text.len().min(NESTED_OUTPUT_BYTES);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    AgentEvent::NestedToolCallFinished {
        outer_tool_call_id,
        tool_call_id,
        tool_name: tool_name.to_string(),
        outcome,
        output: text[..cut].to_string(),
        output_bytes: text.len(),
    }
}

/// Sends a stopped call's end event when the call is dropped before it
/// ends, without waiting for room in the channel.
struct Unfinished<'a> {
    events: Option<mpsc::Sender<AgentEvent>>,
    outer_tool_call_id: Uuid,
    tool_call_id: Uuid,
    tool_name: &'a str,
}

impl Drop for Unfinished<'_> {
    fn drop(&mut self) {
        if let Some(events) = self.events.take() {
            let _ = events.try_send(finished(
                self.outer_tool_call_id,
                self.tool_call_id,
                self.tool_name,
                NestedCallOutcome::Stopped,
                "The call was stopped before it ended.",
            ));
        }
    }
}
