//! Conversation handover: when an agent's context grows large, it writes a
//! report of its progress and continues from that report in a fresh
//! conversation, in the same task.
//!
//! Performance and cost both suffer as a conversation grows. A handover keeps
//! the task (its ID, state, budget and usage history) and replaces only the
//! conversation: the runtime asks the agent for a report, then starts a new
//! conversation whose first message is the task prompt followed by the
//! report. Nothing outside the task sees the agent stop, so hosts that
//! address the agent by name or task ID are unaffected.
//!
//! [`AgentConfig::handover`](crate::agent_config::AgentConfig::handover) turns
//! it on at a context size; a host can also ask for one at any time with
//! [`UserCommand::RequestHandover`](crate::runtime::command::UserCommand::RequestHandover).
//! A [`HandoverHook`] lets the host adjust the fresh conversation's first
//! message and update its own records at the moment the conversation is
//! replaced.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::ArkoAgentsError;
use crate::task::Task;

/// What the runtime asks for, as a user message, when a handover starts.
/// The agent answers with the report as plain text; other tool calls are
/// rejected until it does (it may still call `complete_task` if the task is
/// done).
pub const DEFAULT_REPORT_REQUEST: &str = "Your conversation is getting long, so it will be replaced with a fresh one that starts from your task and a report you write now. Reply with that report as plain text, without calling a tool. Cover what the task is, what you have done and the evidence for it (files, commands, results), decisions you made and why, what you were doing when you stopped, and the next steps. List anything you are still waiting for, such as a reply to a question you asked, and from whom, and every message you received that you haven't fully answered yet: who sent it, what it asked, and what you meant to reply. Be complete: the fresh conversation will have only the task and this report. Messages that arrive while you write it are delivered after it, in the fresh conversation.";

/// The heading of the report in the fresh conversation's first message.
pub const REPORT_HEADING: &str = "# Handover report";

/// Said before the report in the fresh conversation's first message.
pub const REPORT_INTRODUCTION: &str = "Your earlier conversation on this task was replaced with this one to keep your context small. Before it was replaced, you wrote the report below. Continue the task from it.";

/// When an agent hands over; see the [module docs](self). At least one
/// threshold must be set. In arko's own runtime it is checked before each
/// model request against the estimated input of that request: the latest
/// request's reported input, plus what the conversation gained since (the
/// reply, at least its reported output tokens, and the turn's tool
/// results) at the bytes-per-token ratio calibrated from the task's own
/// requests: the overflow guard's calibrated estimate, not the
/// 10-bytes-per-token lower bound it decides by (see
/// [`ToolResultLimits`](crate::agent_config::ToolResultLimits)). So a turn
/// whose results take the conversation over the threshold hands over
/// before the request that would carry them. The Claude Code runtime
/// checks the latest request's input, as Claude Code reports it.
/// [`HandoverRequest::input_tokens`] and [`HandoverRecord::input_tokens`]
/// record the latest request's reported input.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HandoverConfig {
    /// Hand over once a request's input reaches this many tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_input_tokens: Option<usize>,
    /// Hand over once a request's input reaches this share of the model's
    /// input limit: above 0 and at most 1. Ignored for a model whose limit
    /// is unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_context_fraction: Option<f64>,
    /// Replaces [`DEFAULT_REPORT_REQUEST`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_request: Option<String>,
}

impl HandoverConfig {
    /// A handover once a request's input reaches `tokens` tokens.
    pub fn at_input_tokens(tokens: usize) -> Self {
        Self {
            at_input_tokens: Some(tokens),
            ..Self::default()
        }
    }

    /// A handover once a request's input reaches `fraction` of the model's
    /// input limit.
    pub fn at_context_fraction(fraction: f64) -> Self {
        Self {
            at_context_fraction: Some(fraction),
            ..Self::default()
        }
    }

    /// The input size, in tokens, at which to hand over for a model that
    /// accepts `max_input_tokens` (0 when unknown): the lower of the two
    /// thresholds that apply.
    pub fn threshold(&self, max_input_tokens: usize) -> Option<usize> {
        let fraction = self
            .at_context_fraction
            .filter(|_| max_input_tokens > 0)
            .map(|fraction| (max_input_tokens as f64 * fraction) as usize);
        match (self.at_input_tokens, fraction) {
            (Some(tokens), Some(fraction)) => Some(tokens.min(fraction)),
            (tokens, fraction) => tokens.or(fraction),
        }
    }

    /// The request the agent receives when the handover starts.
    pub fn report_request(&self) -> &str {
        self.report_request
            .as_deref()
            .unwrap_or(DEFAULT_REPORT_REQUEST)
    }

    pub fn validate(&self) -> Result<(), ArkoAgentsError> {
        let error = |message: &str| ArkoAgentsError::BuilderError {
            struct_name: "HandoverConfig".into(),
            message: message.into(),
        };
        if self.at_input_tokens.is_none() && self.at_context_fraction.is_none() {
            return Err(error(
                "set at_input_tokens or at_context_fraction; a host can also ask for a handover with UserCommand::RequestHandover.",
            ));
        }
        if self.at_input_tokens == Some(0) {
            return Err(error("at_input_tokens must be positive."));
        }
        if self
            .at_context_fraction
            .is_some_and(|fraction| !(fraction > 0.0 && fraction <= 1.0))
        {
            return Err(error("at_context_fraction must be above 0 and at most 1."));
        }
        if self
            .report_request
            .as_deref()
            .is_some_and(|request| request.trim().is_empty())
        {
            return Err(error("report_request must not be empty."));
        }
        Ok(())
    }
}

/// A handover the runtime has asked the agent for and is waiting on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HandoverRequest {
    pub requested_at: DateTime<Utc>,
    /// The input size of the latest request when the handover started; 0
    /// when a host asked for it before any request.
    pub input_tokens: usize,
    /// The report request is in the conversation. A host's request is noted
    /// at once and the report requested at the next turn boundary where a
    /// user message can go.
    #[serde(default)]
    pub report_requested: bool,
}

/// A handover that has taken place.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HandoverRecord {
    pub at: DateTime<Utc>,
    /// The input size of the latest request when the handover started.
    pub input_tokens: usize,
    /// How many messages the replaced conversation had.
    pub replaced_messages: usize,
    /// The report the agent wrote.
    pub report: String,
}

/// The fresh conversation's first message, before the runtime adds the
/// environment details and reminder it adds to every task prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct HandoverSeed {
    /// The task, restated: by default the task's initial prompt.
    pub task_prompt: String,
    /// Context the host adds after the report, such as the agents the task
    /// can message.
    pub context: Option<String>,
}

impl HandoverSeed {
    /// The default: the task's initial prompt, and no further context.
    pub fn for_task(task: &Task) -> Self {
        Self {
            task_prompt: task.initial_task_prompt.clone(),
            context: None,
        }
    }

    /// The message text: the task prompt, the report under
    /// [`REPORT_HEADING`], then the context.
    pub fn message(&self, report: &str) -> String {
        let mut message = format!(
            "{}\n\n{REPORT_HEADING}\n\n{REPORT_INTRODUCTION}\n\n{}",
            self.task_prompt.trim_end(),
            report.trim()
        );
        if let Some(context) = self.context.as_deref().filter(|c| !c.trim().is_empty()) {
            message.push_str("\n\n");
            message.push_str(context.trim());
        }
        message
    }
}

/// Called by the runtime at the moment a task's conversation is replaced.
/// Set with [`AgentRuntime::with_handover_hook`](crate::runtime::AgentRuntime::with_handover_hook).
pub trait HandoverHook: Send + Sync {
    /// `task` still holds the conversation being replaced. Returns the fresh
    /// conversation's first message; the default restates the task's
    /// initial prompt. A host that tracks what the conversation received
    /// (delivered messages, files read) updates its records here, before the
    /// conversation is gone.
    fn on_handover(&self, task: &Task, report: &str) -> HandoverSeed {
        let _ = report;
        HandoverSeed::for_task(task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_take_the_lower_limit() {
        let both = HandoverConfig {
            at_input_tokens: Some(150_000),
            at_context_fraction: Some(0.25),
            ..HandoverConfig::default()
        };
        assert_eq!(both.threshold(1_000_000), Some(150_000));
        assert_eq!(both.threshold(200_000), Some(50_000));
        assert_eq!(both.threshold(0), Some(150_000));
        assert_eq!(HandoverConfig::at_context_fraction(0.3).threshold(0), None);
        assert_eq!(
            HandoverConfig::at_context_fraction(0.3).threshold(1_000_000),
            Some(300_000)
        );
    }

    #[test]
    fn validation_needs_a_sensible_threshold() {
        assert!(HandoverConfig::default().validate().is_err());
        assert!(HandoverConfig::at_input_tokens(0).validate().is_err());
        assert!(HandoverConfig::at_context_fraction(1.5).validate().is_err());
        assert!(HandoverConfig::at_context_fraction(0.0).validate().is_err());
        assert!(HandoverConfig::at_context_fraction(0.25).validate().is_ok());
        let blank = HandoverConfig {
            report_request: Some("  ".into()),
            ..HandoverConfig::at_input_tokens(10)
        };
        assert!(blank.validate().is_err());
    }

    #[test]
    fn the_seed_message_restates_the_task_then_the_report() {
        let seed = HandoverSeed {
            task_prompt: "Fix the parser.\n".into(),
            context: Some("# Agents you can message\n- `lead`".into()),
        };
        let message = seed.message("  Done: lexer. Next: parser.  ");
        assert!(message.starts_with("Fix the parser.\n\n# Handover report\n\n"));
        assert!(message.contains("Done: lexer. Next: parser.\n\n# Agents you can message"));
    }
}
