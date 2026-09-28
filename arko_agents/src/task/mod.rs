use crate::agent_config::{AgentConfig, DeveloperDefinableEnvironmentDetail};
use crate::error::ArkoAgentsError;
use arko_ai::{ModelCallUsage, UsageInfo};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use uuid::Uuid;

mod refusal;
pub use refusal::{RefusalResolution, RefusedInput, RefusedMessage, RefusedToolResult};

/// Information about a single tool call pending approval, typically part of a batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingToolCallInfo {
    /// Individual ID for this specific call.
    pub tool_call_id: Uuid,
    pub tool_name: String,
    pub parameters: HashMap<String, Value>,
    pub tool_description: String,
}

/// The reason a task stopped by [`UserCommand::CancelTask`](crate::runtime::command::UserCommand::CancelTask) records.
pub const CANCELLED_BY_USER: &str = "Task cancelled by user.";

/// Represents the current state of a task.
///
/// Tasks saved before [`TaskState::Cancelled`] existed recorded a cancelled
/// task as `Failed` with the reason [`CANCELLED_BY_USER`]; that state loads
/// as `Cancelled`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(from = "TaskStateRepr")]
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
    /// Stopped on request (a [`CancelTask`](crate::runtime::command::UserCommand::CancelTask)
    /// command or a middleware's
    /// [`ToolExecutionDecision::CancelTask`](crate::tool::ToolExecutionDecision::CancelTask))
    /// rather than by a failure.
    Cancelled {
        reason: String,
    },
}

/// The serialized form of [`TaskState`], read before the legacy mapping.
#[derive(Deserialize)]
enum TaskStateRepr {
    Pending,
    InProgress,
    AwaitingToolApproval {
        tool_call_id: Uuid,
        tool_name: String,
        parameters: HashMap<String, Value>,
        tool_description: String,
    },
    AwaitingBatchToolApproval {
        batch_id: Uuid,
        pending_calls: Vec<PendingToolCallInfo>,
    },
    AwaitingUserResponse {
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
    Cancelled {
        reason: String,
    },
}

impl From<TaskStateRepr> for TaskState {
    fn from(state: TaskStateRepr) -> Self {
        match state {
            TaskStateRepr::Pending => TaskState::Pending,
            TaskStateRepr::InProgress => TaskState::InProgress,
            TaskStateRepr::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
            } => TaskState::AwaitingToolApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
            },
            TaskStateRepr::AwaitingBatchToolApproval {
                batch_id,
                pending_calls,
            } => TaskState::AwaitingBatchToolApproval {
                batch_id,
                pending_calls,
            },
            TaskStateRepr::AwaitingUserResponse { questions } => {
                TaskState::AwaitingUserResponse { questions }
            }
            TaskStateRepr::AwaitingErrorRecoveryApproval { error_message } => {
                TaskState::AwaitingErrorRecoveryApproval { error_message }
            }
            TaskStateRepr::Interrupted { reason } => TaskState::Interrupted { reason },
            TaskStateRepr::Completed { summary } => TaskState::Completed { summary },
            // Saved before cancellation had its own state.
            TaskStateRepr::Failed { reason } if reason == CANCELLED_BY_USER => {
                TaskState::Cancelled { reason }
            }
            TaskStateRepr::Failed { reason } => TaskState::Failed { reason },
            TaskStateRepr::Cancelled { reason } => TaskState::Cancelled { reason },
        }
    }
}

impl TaskState {
    /// Returns true if the task is in a terminal state (Completed, Failed or
    /// Cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed { .. } | TaskState::Failed { .. } | TaskState::Cancelled { .. }
        )
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
    /// The config this task was started with. It is saved with its tool
    /// calling mode; a task saved without one ran output-based and keeps
    /// that mode (see [`ToolCallingMode`](crate::agent_config::ToolCallingMode)).
    #[serde(deserialize_with = "crate::agent_config::deserialize_task_agent_config")]
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
    /// Output-based tool results of a turn or approval that ended waiting
    /// for a command (an approval, or a pause on a tool error): they are
    /// added before the message that ends the wait. The output-based
    /// counterpart of `pending_native_tool_results`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_text_tool_results: Vec<String>,
    /// Tool errors the task pauses on under
    /// `ToolErrorPolicy::PauseAndAwaitApproval` once the other calls of
    /// their turn are done, including calls still awaiting approval.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred_tool_errors: Vec<DeferredToolError>,

    /// The input of the latest refused turn, until the refusal is resolved,
    /// recovery is approved without a change, or a later turn completes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused_input: Option<RefusedInput>,
    /// Indices of conversation messages the runtime added (file-tree
    /// updates, error and recovery notes, requests to continue), which
    /// [`RefusalResolution`] leaves alone.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub runtime_messages: BTreeSet<usize>,
    /// Indices of tool-results messages whose text beside the results the
    /// runtime added (a recovery note), which [`RefusalResolution`] doesn't
    /// rephrase.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub runtime_text_messages: BTreeSet<usize>,
    /// Set after a refusal when the opt-in refusal fallback will answer the
    /// refused input: the refusal the fallback request answers. The next
    /// model request goes to the fallback model; saved so a task suspended
    /// in between still makes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_for: Option<arko_ai::Refusal>,
    /// Indices in `usage_history` of calls made to the refusal fallback
    /// model, which the cache monitor skips.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_calls: Vec<usize>,
    /// The plan-usage window last reported over the opt-in threshold, so a
    /// resumed task doesn't report (or pause for) it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_usage_threshold_reported: Option<PlanUsageWindowMark>,
    /// The task is paused by a monitor (plan usage or cache threshold)
    /// rather than an error; approving it adds no error note.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub monitor_pause: bool,
    /// Context the runtime has taken from its command channel
    /// ([`UserCommand::InjectContext`](crate::runtime::command::UserCommand::InjectContext))
    /// and not yet added to the conversation, in the order it was sent. It
    /// is added, as one workflow context update, at the next safe turn
    /// boundary, before any context sent later. A task that stops keeps here
    /// what could not be added before it stopped: in an awaiting state (a
    /// user message must not come between a call and its result), while a
    /// refusal is unresolved (a pending fallback answers exactly the refused
    /// input, and [`Task::resolve_refusal`] needs it to be the last input),
    /// and once completed (the model has not read it). Saved so a task that
    /// runs again still gets it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_context_injections: Vec<String>,
}

/// A failed tool call the task will pause on; see
/// [`Task::deferred_tool_errors`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeferredToolError {
    /// The error, as `AgentEvent::ErrorEncountered` reports it.
    pub message: String,
    pub tool_call_id: Option<Uuid>,
}

/// A plan-usage window and its reset; see
/// [`Task::plan_usage_threshold_reported`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanUsageWindowMark {
    /// `"primary"` or `"secondary"`.
    pub window: String,
    /// The window's reset, in Unix seconds, when reported.
    pub resets_at: Option<u64>,
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
            pending_text_tool_results: Vec::new(),
            deferred_tool_errors: Vec::new(),
            refused_input: None,
            runtime_messages: BTreeSet::new(),
            runtime_text_messages: BTreeSet::new(),
            fallback_for: None,
            fallback_calls: Vec::new(),
            plan_usage_threshold_reported: None,
            monitor_pause: false,
            pending_context_injections: Vec::new(),
        }
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

    /// Saves the task to a JSON file, replacing it in one step (see
    /// [`write_file_atomically`](crate::utils::write_file_atomically)), so
    /// that a crash during the save leaves the previous file whole.
    pub fn save_to_path(&self, path: &Path) -> Result<(), ArkoAgentsError> {
        let json_string = serde_json::to_string_pretty(self)?;
        crate::utils::write_file_atomically(path, json_string.as_bytes()).map_err(|e| {
            ArkoAgentsError::FileOperation {
                path: path.to_path_buf(),
                source: e,
            }
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
    /// Tool results of the question's turn that are still waiting (a task
    /// saved by an older version can hold an approved `ask_user`'s result)
    /// are added with the answer, so every call keeps its result.
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
        let answer = self.with_pending_text_tool_results(format!(
            "Response to the agent's questions:\nQuestions: {questions}\nResponse: {response}"
        ));
        self.add_pending_tool_results(Some(&answer));
        Ok(())
    }

    /// `message` after the output-based tool results still waiting (see
    /// [`Task::pending_text_tool_results`]), which it takes.
    pub(crate) fn with_pending_text_tool_results(&mut self, message: String) -> String {
        let mut parts = std::mem::take(&mut self.pending_text_tool_results);
        if parts.is_empty() {
            return message;
        }
        if !message.is_empty() {
            parts.push(message);
        }
        parts.join("\n\n---\n\n")
    }

    /// Adds the tool results waiting for this turn to the conversation as
    /// one user message: the native results, then `text`, then the tool
    /// images, each after its description. Adds nothing when there is
    /// nothing to add; returns whether a message was added.
    pub(crate) fn add_pending_tool_results(&mut self, text: Option<&str>) -> bool {
        let results = std::mem::take(&mut self.pending_native_tool_results);
        if results.is_empty() && text.is_none() && self.pending_tool_images.is_empty() {
            return false;
        }
        let mut content: Vec<_> = results
            .into_iter()
            .map(arko_ai::Content::ToolResult)
            .collect();
        if let Some(text) = text {
            content.push(arko_ai::Content::Text(text.to_string()));
        }
        content.extend(self.take_tool_image_content());
        self.conversation.add_user_message_with_contents(content);
        self.increment_user_message_count();
        true
    }

    /// The waiting tool images, each after its description, as message
    /// content.
    pub(crate) fn take_tool_image_content(&mut self) -> Vec<arko_ai::Content> {
        std::mem::take(&mut self.pending_tool_images)
            .into_iter()
            .flat_map(|attachment| {
                [
                    arko_ai::Content::Text(attachment.description),
                    arko_ai::Content::Image(attachment.image),
                ]
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::AgentLLMConfig;
    use crate::runtime::event::AgentEvent;

    fn state(json: Value) -> TaskState {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn a_cancellation_saved_as_a_failure_loads_as_cancelled() {
        assert_eq!(
            state(serde_json::json!({"Failed": {"reason": "Task cancelled by user."}})),
            TaskState::Cancelled {
                reason: CANCELLED_BY_USER.to_string()
            }
        );
        // Only the exact reason the runtime recorded is a cancellation.
        for reason in [
            "Task cancelled by user",
            "Tool call cancelled by the server.",
        ] {
            assert_eq!(
                state(serde_json::json!({"Failed": {"reason": reason}})),
                TaskState::Failed {
                    reason: reason.to_string()
                }
            );
        }
    }

    #[test]
    fn every_state_round_trips() {
        let states = [
            TaskState::Pending,
            TaskState::InProgress,
            TaskState::AwaitingToolApproval {
                tool_call_id: Uuid::new_v4(),
                tool_name: "write_file".into(),
                parameters: HashMap::from([("path".into(), serde_json::json!("a.txt"))]),
                tool_description: "Writes".into(),
            },
            TaskState::AwaitingBatchToolApproval {
                batch_id: Uuid::new_v4(),
                pending_calls: vec![PendingToolCallInfo {
                    tool_call_id: Uuid::new_v4(),
                    tool_name: "read_file".into(),
                    parameters: HashMap::new(),
                    tool_description: "Reads".into(),
                }],
            },
            TaskState::AwaitingUserResponse {
                questions: "Which?".into(),
            },
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "boom".into(),
            },
            TaskState::Interrupted {
                reason: "network".into(),
            },
            TaskState::Completed {
                summary: "done".into(),
            },
            TaskState::Failed {
                reason: "verifier rejected".into(),
            },
            TaskState::Cancelled {
                reason: "Workflow cancelled.".into(),
            },
        ];
        for original in states {
            // Naming every variant here makes a new one fail to compile until
            // it is added to this list (and to `TaskStateRepr`).
            match &original {
                TaskState::Pending
                | TaskState::InProgress
                | TaskState::AwaitingToolApproval { .. }
                | TaskState::AwaitingBatchToolApproval { .. }
                | TaskState::AwaitingUserResponse { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. }
                | TaskState::Interrupted { .. }
                | TaskState::Completed { .. }
                | TaskState::Failed { .. }
                | TaskState::Cancelled { .. } => {}
            }
            let json = serde_json::to_value(&original).unwrap();
            assert_eq!(state(json), original);
        }
        assert_eq!(
            serde_json::to_value(TaskState::Cancelled {
                reason: CANCELLED_BY_USER.into()
            })
            .unwrap(),
            serde_json::json!({"Cancelled": {"reason": CANCELLED_BY_USER}})
        );
        assert!(
            TaskState::Cancelled {
                reason: String::new()
            }
            .is_terminal()
        );
    }

    /// Context taken but not yet added is saved only when there is some, in
    /// order, and a task saved before the field existed loads without it.
    #[test]
    fn context_taken_but_not_added_is_saved_with_the_task() {
        let config = AgentConfig::builder(
            "saved".into(),
            "Saved".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let task = Task::new(config, "prompt".into(), PathBuf::from("/workspace"), None);
        let json = serde_json::to_value(&task).unwrap();
        assert!(json.get("pending_context_injections").is_none());
        let mut loaded: Task = serde_json::from_value(json).unwrap();
        assert!(loaded.pending_context_injections.is_empty());

        loaded.pending_context_injections = vec!["First.".into(), "Second.".into()];
        let json = serde_json::to_value(&loaded).unwrap();
        assert_eq!(
            json["pending_context_injections"],
            serde_json::json!(["First.", "Second."])
        );
        let loaded: Task = serde_json::from_str(&json.to_string()).unwrap();
        assert_eq!(loaded.pending_context_injections, ["First.", "Second."]);
    }

    /// A new configuration without a tool calling mode gets native; a saved
    /// task records its mode, and one saved without it (by a version whose
    /// default was output-based) keeps output-based.
    #[test]
    fn saved_tasks_keep_their_tool_calling_mode_and_new_configs_default_to_native() {
        use crate::agent_config::ToolCallingMode;
        let config = serde_json::json!({
            "id": "a", "name": "A", "role_description": "r", "instructions": "i",
            "llm_config": {"provider": "OpenAI", "model_id": "gpt-5.6-luna"},
            "tool_config": {"available_tool_names": ["read_file"]}
        });
        let loaded: AgentConfig = serde_json::from_value(config.clone()).unwrap();
        assert_eq!(
            loaded.tool_config.tool_calling_mode,
            ToolCallingMode::Native
        );
        let mut bare = config.clone();
        bare.as_object_mut().unwrap().remove("tool_config");
        let loaded: AgentConfig = serde_json::from_value(bare.clone()).unwrap();
        assert_eq!(
            loaded.tool_config.tool_calling_mode,
            ToolCallingMode::Native
        );
        let yaml = "id: a\nname: A\nrole_description: r\ninstructions: i\nllm_config:\n  provider: OpenAI\n  model_id: gpt-5.6-luna\n";
        let loaded: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            loaded.tool_config.tool_calling_mode,
            ToolCallingMode::Native
        );

        let task = Task::new(loaded, "prompt".into(), PathBuf::from("/workspace"), None);
        let mut saved = serde_json::to_value(&task).unwrap();
        assert_eq!(
            saved["agent_config"]["tool_config"]["tool_calling_mode"],
            "Native"
        );
        assert_eq!(
            serde_json::from_value::<Task>(saved.clone())
                .unwrap()
                .agent_config
                .tool_config
                .tool_calling_mode,
            ToolCallingMode::Native
        );
        for old_config in [config, bare] {
            saved["agent_config"] = old_config;
            let old: Task = serde_json::from_value(saved.clone()).unwrap();
            assert_eq!(
                old.agent_config.tool_config.tool_calling_mode,
                ToolCallingMode::OutputBased
            );
        }
        // Loading a task doesn't change what a configuration loaded next gets.
        let after: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(after.tool_config.tool_calling_mode, ToolCallingMode::Native);
        assert!(serde_json::from_value::<Task>(serde_json::json!({"agent_config": {}})).is_err());
        let after: AgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(after.tool_config.tool_calling_mode, ToolCallingMode::Native);
    }

    #[test]
    fn saved_tasks_and_events_from_before_the_cancelled_state_still_load() {
        let config = AgentConfig::builder(
            "saved".into(),
            "Saved".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let mut task = Task::new(config, "prompt".into(), PathBuf::from("/workspace"), None);
        task.set_state(TaskState::Completed {
            summary: String::new(),
        });
        let mut json = serde_json::to_value(&task).unwrap();
        json["current_state"] =
            serde_json::json!({"Failed": {"reason": "Task cancelled by user."}});
        let loaded: Task = serde_json::from_str(&json.to_string()).unwrap();
        assert_eq!(
            loaded.current_state,
            TaskState::Cancelled {
                reason: CANCELLED_BY_USER.to_string()
            }
        );
        // Saved in whole seconds.
        assert_eq!(
            loaded.end_time.map(|time| time.timestamp()),
            task.end_time.map(|time| time.timestamp())
        );

        let event: AgentEvent = serde_json::from_value(serde_json::json!({
            "TaskStateChanged": {"Failed": {"reason": "Task cancelled by user."}}
        }))
        .unwrap();
        assert!(matches!(
            event,
            AgentEvent::TaskStateChanged(TaskState::Cancelled { .. })
        ));
    }
}
