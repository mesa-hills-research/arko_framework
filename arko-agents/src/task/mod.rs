use crate::agent_config::{AgentConfig, DeveloperDefinableEnvironmentDetail};
use crate::error::ArkoAgentsError;
use arko_ai::{ModelCallUsage, UsageInfo};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Information about a single tool call pending approval, typically part of a batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingToolCallInfo {
    /// Individual ID for this specific call.
    pub tool_call_id: Uuid,
    pub tool_name: String,
    pub parameters: HashMap<String, Value>,
    pub tool_description: String,
}

/// The input a refused turn answered: the last user message before it, in
/// [`Task::conversation`]. Reported with `AgentEvent::RefusedInput` and kept
/// in [`Task::refused_input`] until the refusal is resolved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RefusedInput {
    /// A user message: the task prompt, a user response, a context update or
    /// a recovery message. In output-based tool mode, tool results are sent
    /// as such messages too.
    UserMessage { message_index: usize },
    /// Native tool results (possibly with text alongside).
    ToolResults {
        message_index: usize,
        results: Vec<RefusedToolResult>,
    },
}

impl RefusedInput {
    pub fn message_index(&self) -> usize {
        match self {
            Self::UserMessage { message_index } | Self::ToolResults { message_index, .. } => {
                *message_index
            }
        }
    }
}

/// A tool result in a [`RefusedInput::ToolResults`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RefusedToolResult {
    /// The provider's call id, which [`RefusalResolution::RedactToolResults`] takes.
    pub call_id: Option<String>,
    pub tool_name: String,
}

/// How the host changes a refused input before the task continues; see
/// [`Task::resolve_refusal`] and `UserCommand::ResolveRefusal`. The refused
/// turn (and anything after the input) is removed, and the next request
/// sends the changed input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RefusalResolution {
    /// Replace the text of a refused user message (images are kept).
    RephraseUserMessage { text: String },
    /// Remove a refused user message. The task prompt, when it is the only
    /// user message, can't be removed; rephrase it instead.
    RemoveUserMessage,
    /// Replace the output of refused tool results (by provider call id),
    /// keeping each call paired with its result.
    RedactToolResults {
        call_ids: Vec<String>,
        replacement: String,
    },
}

/// Represents the current state of a task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskState {
    Pending,
    InProgress,
    /// Awaiting approval for a single tool call.
    AwaitingToolApproval {
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
        tool_description: String,
    },
    /// Awaiting approval for a batch where one or more calls require approval.
    AwaitingBatchToolApproval {
        /// An ID for the entire batch operation.
        batch_id: Uuid,
        /// Details of each call needing approval.
        pending_calls: Vec<PendingToolCallInfo>,
    },
    /// Awaiting user response to questions asked by the agent via `ask_user` tool.
    AwaitingUserResponse {
        /// The questions asked by the agent.
        questions: String,
    },
    AwaitingErrorRecoveryApproval {
        error_message: String,
    },
    Interrupted {
        reason: String,
    },
    Completed {
        summary: String,
    },
    Failed {
        reason: String,
    },
}

impl TaskState {
    /// Returns true if the task is in a terminal state (Completed or Failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskState::Completed { .. } | TaskState::Failed { .. })
    }

    /// Returns true if the task is awaiting some form of user approval or response.
    pub fn is_awaiting_approval(&self) -> bool {
        matches!(
            self,
            TaskState::AwaitingToolApproval { .. }
                | TaskState::AwaitingBatchToolApproval { .. }
                | TaskState::AwaitingUserResponse { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. }
        )
    }
}

/// Represents a task being executed by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub agent_config_id: String,
    /// The config this task was started with.
    pub agent_config: AgentConfig,
    pub initial_task_prompt: String,
    pub conversation: arko_ai::Conversation,
    pub current_state: TaskState,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub start_time: DateTime<Utc>,
    #[serde(with = "chrono::serde::ts_seconds_option", default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<DateTime<Utc>>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub developer_env_details: Option<Vec<DeveloperDefinableEnvironmentDetail>>,

    #[serde(default)]
    pub usage_history: Vec<UsageInfo>,

    /// Per-turn model call usage metrics with computed input deltas.
    ///
    /// Each entry corresponds to one LLM call. The `input_delta_from_previous_call`
    /// field is populated by the agent runtime when the usage is recorded.
    #[serde(default)]
    pub model_call_usage_history: Vec<ModelCallUsage>,

    /// Task-specific current working directory.
    pub current_working_directory: PathBuf,

    /// Counter for the number of user messages sent to the LLM.
    /// Used for determining when to inject user reminders.
    #[serde(default)]
    pub user_message_count: usize,

    /// Tool outputs waiting for the remainder of a native batch or approval.
    /// Persist them together so image attachments retain their call correlation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_native_tool_results: Vec<arko_ai::ToolResult>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub pending_native_tool_calls: HashMap<Uuid, arko_ai::ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_tool_images: Vec<crate::tool::ToolImage>,

    /// The input of the latest refused turn, until the refusal is resolved
    /// or recovery is approved without a change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused_input: Option<RefusedInput>,
}

impl Task {
    /// Creates a new task with the given configuration.
    pub fn new(
        agent_config: AgentConfig,
        initial_task_prompt: String,
        initial_cwd: PathBuf,
        developer_env_details: Option<Vec<DeveloperDefinableEnvironmentDetail>>,
    ) -> Self {
        let task_id = Uuid::new_v4();
        let agent_config_id = agent_config.id.clone();
        let conversation = arko_ai::Conversation::new(agent_config.llm_config.model.clone());

        Self {
            id: task_id,
            agent_config_id,
            agent_config,
            initial_task_prompt,
            conversation,
            current_state: TaskState::Pending,
            start_time: Utc::now(),
            end_time: None,
            developer_env_details,
            usage_history: Vec::new(),
            model_call_usage_history: Vec::new(),
            current_working_directory: initial_cwd,
            user_message_count: 0,
            pending_native_tool_results: Vec::new(),
            pending_native_tool_calls: HashMap::new(),
            pending_tool_images: Vec::new(),
            refused_input: None,
        }
    }

    /// Changes the input of the latest refused turn ([`Task::refused_input`])
    /// as `resolution` says, removing the refused turn, so that the task can
    /// continue with a request the provider has not refused. Used by the
    /// runtime for `UserCommand::ResolveRefusal`; call it directly on a task
    /// that was returned interrupted, before running it again.
    pub fn resolve_refusal(&mut self, resolution: RefusalResolution) -> Result<(), ArkoAgentsError> {
        let invalid = |message: String| ArkoAgentsError::InvalidTaskState(message);
        let input = self
            .refused_input
            .clone()
            .ok_or_else(|| invalid("The task has no refused input to resolve.".into()))?;
        let index = input.message_index();
        let conversation = &mut self.conversation;
        if index >= conversation.messages.len() {
            return Err(invalid("The refused input is no longer in the conversation.".into()));
        }
        match (resolution, &input) {
            (RefusalResolution::RephraseUserMessage { text }, RefusedInput::UserMessage { .. }) => {
                if conversation.last_user_turn() != Some(index) {
                    return Err(invalid("The refused message is no longer the last user message.".into()));
                }
                conversation
                    .rephrase_last_user_turn(&text)
                    .map_err(|error| invalid(error.to_string()))?;
            }
            (RefusalResolution::RemoveUserMessage, RefusedInput::UserMessage { .. }) => {
                if conversation.last_user_turn() != Some(index) {
                    return Err(invalid("The refused message is no longer the last user message.".into()));
                }
                if !conversation.messages[..index]
                    .iter()
                    .any(|message| message.role == arko_ai::Role::User)
                {
                    return Err(invalid(
                        "The task prompt is the only user message; rephrase it instead of removing it.".into(),
                    ));
                }
                conversation
                    .remove_last_user_turn()
                    .map_err(|error| invalid(error.to_string()))?;
            }
            (
                RefusalResolution::RedactToolResults { call_ids, replacement },
                RefusedInput::ToolResults { results, .. },
            ) => {
                if call_ids.is_empty() {
                    return Err(invalid("Name at least one tool result to redact.".into()));
                }
                for call_id in &call_ids {
                    if !results.iter().any(|result| result.call_id.as_deref() == Some(call_id)) {
                        return Err(invalid(format!(
                            "Call {call_id} is not among the refused tool results."
                        )));
                    }
                }
                for call_id in &call_ids {
                    let found = conversation
                        .redact_tool_result(call_id, &replacement)
                        .map_err(|error| invalid(error.to_string()))?;
                    if found != index {
                        return Err(invalid(format!(
                            "The result of call {call_id} moved in the conversation."
                        )));
                    }
                }
                conversation.messages.truncate(index + 1);
            }
            (_, input) => {
                let kind = match input {
                    RefusedInput::UserMessage { .. } => "a user message; rephrase or remove it",
                    RefusedInput::ToolResults { .. } => "tool results; redact them",
                };
                return Err(invalid(format!("The refused input is {kind}.")));
            }
        }
        self.refused_input = None;
        Ok(())
    }

    /// Records a model call's usage metrics, computing the input token delta
    /// from the previous call in this task's history.
    ///
    /// Returns the usage with `input_delta_from_previous_call` populated.
    pub fn record_model_call_usage(&mut self, mut usage: ModelCallUsage) -> ModelCallUsage {
        if let Some(previous) = self.model_call_usage_history.last() {
            let delta = usage.input_tokens as isize - previous.input_tokens as isize;
            usage = usage.with_input_delta(delta);
        }
        self.model_call_usage_history.push(usage.clone());
        usage
    }

    /// Loads a task from a JSON file.
    pub fn load_from_path(path: &Path) -> Result<Self, ArkoAgentsError> {
        let file_content =
            std::fs::read_to_string(path).map_err(|e| ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            })?;
        serde_json::from_str(&file_content).map_err(ArkoAgentsError::Json)
    }

    /// Saves the task to a JSON file.
    pub fn save_to_path(&self, path: &Path) -> Result<(), ArkoAgentsError> {
        let json_string = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json_string).map_err(|e| ArkoAgentsError::FileOperation {
            path: path.to_path_buf(),
            source: e,
        })
    }

    /// Gets the runtime duration of the task, if it has ended.
    pub fn get_run_time(&self) -> Option<chrono::Duration> {
        self.end_time.map(|et| et - self.start_time)
    }

    /// Sets the task state, handling terminal state transitions appropriately.
    pub fn set_state(&mut self, new_state: TaskState) {
        let is_already_terminal = self.current_state.is_terminal() && self.end_time.is_some();
        let new_state_is_terminal = new_state.is_terminal();

        self.current_state = new_state;

        if new_state_is_terminal {
            if !is_already_terminal {
                self.end_time = Some(Utc::now());
            }
        } else {
            self.end_time = None;
        }
    }

    /// Prepares a task that was 'Completed' for resumption.
    /// Sets its state to 'InProgress' and adds the continuation prompt to the conversation.
    pub fn prepare_for_resumption_from_completed(
        &mut self,
        continuation_prompt: String,
    ) -> Result<(), ArkoAgentsError> {
        if !matches!(self.current_state, TaskState::Completed { .. }) {
            return Err(ArkoAgentsError::InvalidTaskState(
                "Task must be in a 'Completed' state to be resumed this way.".to_string(),
            ));
        }
        self.set_state(TaskState::InProgress);
        self.conversation.add_user_message(&continuation_prompt);
        self.user_message_count += 1;
        Ok(())
    }

    /// Resume a task suspended by `ask_user` with the application's answer.
    pub fn prepare_for_resumption_with_user_response(
        &mut self,
        response: String,
    ) -> Result<(), ArkoAgentsError> {
        let questions = match &self.current_state {
            TaskState::AwaitingUserResponse { questions } => questions.clone(),
            _ => {
                return Err(ArkoAgentsError::InvalidTaskState(
                    "Task must be awaiting user input to resume with a response".to_string(),
                ));
            }
        };
        self.set_state(TaskState::InProgress);
        self.conversation.add_user_message(&format!(
            "Response to the agent's questions:\nQuestions: {questions}\nResponse: {response}"
        ));
        self.user_message_count += 1;
        Ok(())
    }

    /// Returns the summed usage of every LLM call in this task.
    pub fn total_usage(&self) -> UsageInfo {
        self.usage_history.iter().sum()
    }

    /// Returns the amount billed across all LLM usage for this task.
    /// Subscription (ChatGPT OAuth) calls contribute nothing; see
    /// [`UsageInfo::billed`].
    pub fn total_usage_cost(&self) -> f64 {
        self.total_usage().billed()
    }

    /// Returns the total input tokens (including cached input) across all LLM
    /// usage for this task.
    pub fn total_input_tokens(&self) -> usize {
        self.total_usage().input_tokens
    }

    /// Returns the total output tokens across all LLM usage for this task.
    pub fn total_output_tokens(&self) -> usize {
        self.total_usage().output_tokens
    }

    /// Returns the context size of the most recent LLM call, if available:
    /// its total input tokens, including cached input.
    pub fn current_context_size(&self) -> Option<usize> {
        self.usage_history.last().map(|u| u.input_tokens)
    }

    /// Increments the user message count and returns the new count.
    pub fn increment_user_message_count(&mut self) -> usize {
        self.user_message_count += 1;
        self.user_message_count
    }
}
