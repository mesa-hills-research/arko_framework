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
use std::sync::{Arc, Mutex, OnceLock};
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
    /// The calls that got past the checks, in order.
    records: Arc<Mutex<Vec<NestedRecord>>>,
}

/// A call that got past the gate's checks, so its tool may have run, as
/// [`NestedCalls::records`] keeps it. It is there before the tool starts,
/// so a call dropped before its end (the outer call was stopped) is too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedRecord {
    pub tool_call_id: Uuid,
    pub tool_name: String,
    pub state: NestedState,
}

/// How far a recorded call got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestedState {
    /// It runs, or it was dropped before its end, so it may have done
    /// anything its tool does.
    Running,
    /// The tool returned, or with `failed` failed.
    Ran { failed: bool },
    /// The middleware or a check inside the tool refused it, and it did
    /// nothing.
    Refused,
    /// The middleware stopped or paused the task at it, before the tool
    /// ran, or with `ran` after.
    Halted { ran: bool },
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
            records: Arc::default(),
        }
    }

    /// The calls that got past the checks so far, in the order made, each
    /// as far as it got. A caller whose own run ends early (stopped, timed
    /// out) reads what its calls may have done here.
    pub fn records(&self) -> Vec<NestedRecord> {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn record(&self, tool_call_id: Uuid, tool_name: &str, state: NestedState) {
        let mut records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match records
            .iter_mut()
            .find(|record| record.tool_call_id == tool_call_id)
        {
            Some(record) => record.state = state,
            None => records.push(NestedRecord {
                tool_call_id,
                tool_name: tool_name.to_string(),
                state,
            }),
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
        let started = AgentEvent::NestedToolCallStarted {
            outer_tool_call_id: self.outer_tool_call_id,
            tool_call_id,
            tool_name: tool_name.to_string(),
            parameters: params.clone(),
        };
        // Once the start is queued, an end is owed, a stop's until the
        // call's own is known.
        let mut end = Owed::start(
            &self.gate.events,
            started,
            finished(
                self.outer_tool_call_id,
                tool_call_id,
                tool_name,
                NestedCallOutcome::Stopped,
                "The call was stopped before it ended.",
            ),
        )
        .await;
        let outcome = self.run(tool_call_id, tool_name, params).await;
        let (kind, text) = match &outcome {
            NestedOutcome::Finished(output) => (NestedCallOutcome::Finished, output.text.as_str()),
            NestedOutcome::Failed(text) => (NestedCallOutcome::Failed, text.as_str()),
            NestedOutcome::Refused(text) => (NestedCallOutcome::Refused, text.as_str()),
            NestedOutcome::Halted(text) => (NestedCallOutcome::Halted, text.as_str()),
        };
        if let Some(end) = &mut end {
            end.owe(finished(
                self.outer_tool_call_id,
                tool_call_id,
                tool_name,
                kind,
                text,
            ));
            end.deliver().await;
        }
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
        self.record(tool_call_id, tool_name, NestedState::Running);
        let outcome = tool_run::run_nested(self.environment.clone(), run, self.stop.clone()).await;
        let (state, ended) = match outcome {
            RunOutcome::Finished { output, .. } => (
                NestedState::Ran { failed: false },
                Ok(NestedOutcome::Finished(output)),
            ),
            // (A nested call never moves to the background.)
            RunOutcome::Moved { text } => (
                NestedState::Ran { failed: false },
                Ok(NestedOutcome::Finished(ToolOutput::text(text))),
            ),
            RunOutcome::Failed(crate::tool::ToolError::Refused(message)) => {
                (NestedState::Refused, Ok(NestedOutcome::Refused(message)))
            }
            RunOutcome::Failed(error) => (
                NestedState::Ran { failed: true },
                Ok(NestedOutcome::Failed(error.to_string())),
            ),
            RunOutcome::Rejected(message) => {
                (NestedState::Refused, Ok(NestedOutcome::Refused(message)))
            }
            RunOutcome::Stop { reason, cancel } => (
                NestedState::Halted { ran: false },
                Err((
                    Halt::Stop {
                        reason: reason.clone(),
                        cancel,
                    },
                    reason,
                )),
            ),
            RunOutcome::Pause { message, reason } => (
                NestedState::Halted { ran: false },
                Err((Halt::Pause { reason }, message)),
            ),
            RunOutcome::MiddlewareErrorBefore { reason } => (
                NestedState::Halted { ran: false },
                Err((
                    Halt::Pause {
                        reason: reason.clone(),
                    },
                    reason,
                )),
            ),
            // The tool ran, and the hook after it failed.
            RunOutcome::MiddlewareErrorAfter { reason, .. } => (
                NestedState::Halted { ran: true },
                Err((
                    Halt::Pause {
                        reason: reason.clone(),
                    },
                    reason,
                )),
            ),
        };
        self.record(tool_call_id, tool_name, state);
        match ended {
            Ok(outcome) => outcome,
            Err((halt, text)) => {
                let _ = self.halt.set(halt);
                NestedOutcome::Halted(text)
            }
        }
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

/// The end event a started call owes. Delivered, it waits for room in the
/// channel. Dropped first (the outer call was stopped), it goes on a task of
/// its own, which waits for room, so every start has its end.
struct Owed {
    events: mpsc::Sender<AgentEvent>,
    end: Option<AgentEvent>,
}

impl Owed {
    /// Sends `started`, and owes `end` from then on. `None` when the
    /// task's events have no reader, as once it has ended.
    async fn start(
        events: &mpsc::Sender<AgentEvent>,
        started: AgentEvent,
        end: AgentEvent,
    ) -> Option<Self> {
        let permit = events.reserve().await.ok()?;
        permit.send(started);
        Some(Self {
            events: events.clone(),
            end: Some(end),
        })
    }

    /// Owes `end` in place of the end owed until now.
    fn owe(&mut self, end: AgentEvent) {
        self.end = Some(end);
    }

    /// Sends the end owed, once the channel has room.
    async fn deliver(&mut self) {
        if let Ok(permit) = self.events.reserve().await
            && let Some(end) = self.end.take()
        {
            permit.send(end);
        }
    }
}

impl Drop for Owed {
    fn drop(&mut self) {
        let Some(end) = self.end.take() else {
            return;
        };
        match self.events.try_send(end) {
            Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => {}
            Err(mpsc::error::TrySendError::Full(end)) => {
                if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                    let events = self.events.clone();
                    runtime.spawn(async move {
                        let _ = events.send(end).await;
                    });
                }
            }
        }
    }
}
