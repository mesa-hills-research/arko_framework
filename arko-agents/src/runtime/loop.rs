use super::command::{ToolApprovalDecision, UserCommand};
use super::event::AgentEvent;
use crate::agent_config::{
    AgentConfig, LLMErrorPolicy, PathSafetyPolicy, PlanUsageAction, ToolCallingMode,
    ToolErrorPolicy, UsageLimitPolicy,
};
use crate::error::ArkoAgentsError;
use crate::parsing::{
    ParsedToolCall, PromptBuilder, RawParserEvent, ToolCallGroup, ToolCallParser,
    ToolParsingConfig, assemble_tool_calls,
};
use crate::skill::SkillPromptContext;
use crate::task::{PendingToolCallInfo, Task, TaskState};
use crate::tool::{
    InterceptedResult, InterceptionFailurePolicy, InterceptorConfig, TaskContext, ToolCallContext,
    ToolDefinition, ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware,
    ToolExecutionOutcome, ToolImage, ToolOutput, ToolRegistry,
    builtins::{ASK_USER_TOOL_NAME, COMPLETE_TASK_TOOL_NAME},
};
use arko_ai::{self as ai};
use futures::TryStreamExt;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep as tokio_sleep, timeout};
use uuid::Uuid;

const OAUTH_REFRESH_MARGIN_SECS: u64 = 5 * 60;

/// A usage-limit wait retries this long after the reported reset, plus up
/// to [`USAGE_LIMIT_RESET_JITTER`].
const USAGE_LIMIT_RESET_MARGIN: Duration = Duration::from_secs(1);
const USAGE_LIMIT_RESET_JITTER: Duration = Duration::from_secs(30);

/// How recovery from a failed model request waits.
enum RecoveryWait {
    /// Bounded by the error policy's attempts and the elapsed budget; a
    /// message about the error is added before the retry.
    Ordinary,
    /// Opt-in indefinite capacity recovery (`capacity_retry`).
    Capacity(crate::agent_config::LLMCapacityRetryConfig),
    /// Default recovery from short rate limits and capacity errors
    /// (`rate_limit_retry`).
    RateLimit(crate::agent_config::LLMRateLimitRetryConfig),
    /// A plan usage limit, handled under `usage_limit_policy`.
    UsageLimit,
}

fn unix_millis_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, duration_millis)
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// `3h 12m`, `12m 5s` or `40s`.
fn describe_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (days, hours, minutes, rest) = (
        seconds / 86_400,
        seconds % 86_400 / 3_600,
        seconds % 3_600 / 60,
        seconds % 60,
    );
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {rest}s")
    } else {
        format!("{rest}s")
    }
}

/// `ToolRejected` reasons for calls a stream announced but never finished.
const INTERRUPTED_CALL: &str = "The model's response was interrupted before this call finished; it was not run.";
const FAILED_STREAM_CALL: &str = "The model's response failed before this call finished; it was not run.";

async fn wait_for_recovery_deadline(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending::<()>().await,
    }
}


/// Tool names excluded from result interception because their results
/// carry semantic meaning used directly by the runtime (task state
/// transitions, user-facing questions, etc.).
const INTERCEPTION_EXCLUDED_TOOLS: &[&str] = &[COMPLETE_TASK_TOOL_NAME, ASK_USER_TOOL_NAME];

fn native_batch_rejection_reason(
    config: &AgentConfig,
    registry: &ToolRegistry,
    tool_calls: &[ai::ToolCall],
) -> Option<String> {
    if tool_calls.len() <= 1 {
        return None;
    }
    let mut unsafe_tools = tool_calls
        .iter()
        .filter_map(|call| {
            registry.get(&call.name).and_then(|tool| {
                (!config.allows_batched_calls_for_tool(&tool.definition()))
                    .then(|| call.name.clone())
            })
        })
        .collect::<Vec<_>>();
    unsafe_tools.sort();
    unsafe_tools.dedup();
    (!unsafe_tools.is_empty()).then(|| {
        format!(
            "Native tool batch rejected without execution because these tools are not marked batch-safe: {}. Retry batch-safe tools together and call every other tool alone.",
            unsafe_tools.join(", ")
        )
    })
}

#[cfg(test)]
type ScriptedStreamResult =
    Result<futures::stream::BoxStream<'static, Result<ai::StreamEvent, ai::Error>>, ai::Error>;

/// The main agent execution loop.
pub struct AgentLoop {
    task: Task,
    llm_client: Arc<ai::LLMClient>,
    tool_registry: Arc<ToolRegistry>,
    event_sender: mpsc::Sender<AgentEvent>,
    command_receiver: mpsc::Receiver<UserCommand>,
    parsing_config: ToolParsingConfig,
    event_throttle_duration: Duration,
    simplified_tool_events: bool,

    text_buffer: String,
    reasoning_buffer: String,
    last_text_emit_time: Instant,
    last_reasoning_emit_time: Instant,

    llm_interaction_attempts: u32,
    recovery_started: Option<tokio::time::Instant>,
    capacity_recovery_started: Option<tokio::time::Instant>,
    capacity_retry_attempts: u32,
    /// Requests in this recovery episode that ended in a wait that doesn't
    /// count against the ordinary limits (capacity, rate limits).
    indefinite_interactions: u32,
    /// Ordinary recovery time used before its clock was paused for such a
    /// wait.
    paused_recovery_elapsed: Option<Duration>,
    /// The plan usage window (and its reset) last reported over the
    /// threshold, so it is reported once.
    plan_usage_threshold_reported: Option<(String, Option<u64>)>,
    stop_requested: bool,
    #[cfg(test)]
    scripted_responses: std::sync::Mutex<VecDeque<ScriptedStreamResult>>,

    skill_prompt_context: Option<SkillPromptContext>,

    interceptor_config: Option<InterceptorConfig>,
    execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    pending_context_injections: Vec<String>,
    tool_context_cursor: usize,
    file_tree_baseline: Option<crate::file_tree::FileTreeSnapshot>,
    file_tree_error: Option<String>,
}

impl AgentLoop {
    /// Creates a new AgentLoop.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: Task,
        llm_client: Arc<ai::LLMClient>,
        tool_registry: Arc<ToolRegistry>,
        event_sender: mpsc::Sender<AgentEvent>,
        command_receiver: mpsc::Receiver<UserCommand>,
        event_throttle_ms: Option<u64>,
        simplified_tool_events: bool,
        skill_prompt_context: Option<SkillPromptContext>,
        interceptor_config: Option<InterceptorConfig>,
        execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    ) -> Self {
        let parsing_config = task.agent_config.get_tool_parsing_config();
        let throttle_ms = event_throttle_ms.unwrap_or(0);
        let now = Instant::now();

        for name in tool_registry.list_tool_names() {
            if let Some(tool) = tool_registry.get(&name) {
                tool.reset_context(task.id);
            }
        }

        Self {
            task,
            llm_client,
            tool_registry,
            event_sender,
            command_receiver,
            parsing_config,
            event_throttle_duration: Duration::from_millis(throttle_ms),
            simplified_tool_events,
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            last_text_emit_time: now,
            last_reasoning_emit_time: now,
            llm_interaction_attempts: 0,
            recovery_started: None,
            capacity_recovery_started: None,
            capacity_retry_attempts: 0,
            indefinite_interactions: 0,
            paused_recovery_elapsed: None,
            plan_usage_threshold_reported: None,
            stop_requested: false,
            #[cfg(test)]
            scripted_responses: std::sync::Mutex::new(VecDeque::new()),
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
            pending_context_injections: Vec::new(),
            tool_context_cursor: 0,
            file_tree_baseline: None,
            file_tree_error: None,
        }
    }

    fn queue_context_injection(&mut self, content: String) {
        if !content.trim().is_empty() {
            self.pending_context_injections.push(content);
        }
    }

    fn observe_committed_tool_context(&mut self) {
        let tools = self
            .tool_registry
            .list_tool_names()
            .into_iter()
            .filter_map(|name| self.tool_registry.get(&name))
            .collect::<Vec<_>>();
        if self.tool_context_cursor > self.task.conversation.messages.len() {
            for tool in &tools {
                tool.reset_context(self.task.id);
            }
            self.tool_context_cursor = 0;
        }
        for message in &self.task.conversation.messages[self.tool_context_cursor..] {
            for tool in &tools {
                tool.observe_context(self.task.id, message);
            }
        }
        self.tool_context_cursor = self.task.conversation.messages.len();
    }

    fn refresh_file_tree_context(&mut self) {
        let config = &self.task.agent_config.environment_details_config;
        if !config.file_tree.include
            || config.file_tree_mode != crate::agent_config::FileTreeMode::Diff
        {
            self.file_tree_baseline = None;
            self.file_tree_error = None;
            return;
        }
        let depth = config.file_tree.depth.unwrap_or(3);
        let message = match crate::file_tree::FileTreeSnapshot::capture(
            &self.task.current_working_directory,
            Some(depth),
        ) {
            Ok(snapshot) => {
                let recovered = self.file_tree_error.take().is_some();
                let update = match &self.file_tree_baseline {
                    Some(previous) => snapshot.changes_from(previous),
                    None => Some(format!("Full snapshot:\n{}", snapshot.full())),
                };
                self.file_tree_baseline = Some(snapshot);
                update
                    .map(|text| format!("# File tree (max depth {depth})\n{text}"))
                    .or_else(|| {
                        recovered
                            .then(|| "File-tree scan recovered; visible tree is unchanged.".into())
                    })
            }
            Err(error) => {
                let error = error.to_string();
                if self.file_tree_error.as_ref() == Some(&error) {
                    None
                } else {
                    self.file_tree_error = Some(error.clone());
                    Some(format!(
                        "File-tree scan failed; current inventory could not be checked: {error}"
                    ))
                }
            }
        };
        if let Some(message) = message {
            self.task.conversation.add_user_message(&message);
            self.task.increment_user_message_count();
        }
    }

    fn flush_context_injections(&mut self) -> Result<(), ArkoAgentsError> {
        if self.pending_context_injections.is_empty() {
            return Ok(());
        }

        let content = self
            .pending_context_injections
            .drain(..)
            .collect::<Vec<_>>()
            .join("\n\n");
        let prompt_builder =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
        let next_message_number = self.task.user_message_count + 1;
        let message = prompt_builder.augment_user_message_with_env_details_and_reminder(
            &self.task,
            &format!("# Workflow context update\n\n{content}"),
            next_message_number,
        )?;
        self.task.conversation.add_user_message(&message);
        self.task.increment_user_message_count();
        Ok(())
    }

    fn task_context(&self) -> TaskContext<'_> {
        TaskContext {
            task_id: self.task.id,
            agent_name: &self.task.agent_config.name,
            initial_task_prompt: &self.task.initial_task_prompt,
            current_working_directory: &self.task.current_working_directory,
        }
    }

    fn tool_execution_context(&self) -> ToolExecutionContext<'_> {
        ToolExecutionContext {
            task_id: self.task.id,
            agent_name: &self.task.agent_config.name,
            initial_task_prompt: &self.task.initial_task_prompt,
            current_working_directory: &self.task.current_working_directory,
        }
    }

    async fn before_tool_execution(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        parameters: &HashMap<String, Value>,
    ) -> Result<ToolExecutionDecision, crate::tool::ToolError> {
        let Some(middleware) = &self.execution_middleware else {
            return Ok(ToolExecutionDecision::Continue);
        };
        middleware
            .before(
                &self.tool_execution_context(),
                &ToolCallContext {
                    tool_call_id,
                    tool_name,
                    parameters,
                },
            )
            .await
    }

    async fn after_tool_execution(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        parameters: &HashMap<String, Value>,
        outcome: ToolExecutionOutcome<'_>,
    ) -> Result<(), crate::tool::ToolError> {
        let Some(middleware) = &self.execution_middleware else {
            return Ok(());
        };
        middleware
            .after(
                &self.tool_execution_context(),
                &ToolCallContext {
                    tool_call_id,
                    tool_name,
                    parameters,
                },
                outcome,
            )
            .await
    }

    fn is_native_mode(&self) -> bool {
        self.task.agent_config.tool_config.tool_calling_mode == ToolCallingMode::Native
    }

    fn build_native_tool_config(&self) -> ai::ToolConfig {
        let mut ai_tools = Vec::new();
        for name in &self.task.agent_config.tool_config.available_tool_names {
            if let Some(tool) = self.tool_registry.get(name) {
                ai_tools.push(tool.definition().to_arko_ai_tool_definition());
            }
        }
        if !self
            .task
            .agent_config
            .tool_config
            .available_tool_names
            .iter()
            .any(|n| n == COMPLETE_TASK_TOOL_NAME)
            && let Some(ct) = self.tool_registry.get(COMPLETE_TASK_TOOL_NAME)
        {
            ai_tools.push(ct.definition().to_arko_ai_tool_definition());
        }

        let mut config = ai::ToolConfig::new(ai_tools);

        if !self.task.agent_config.tool_config.allow_batched_tool_calls
            && self
                .task
                .conversation
                .model
                .supports_disabling_parallel_tool_calls()
        {
            config = config.with_parallel_tool_calls(false);
        }

        config
    }

    fn build_generation_config(&self) -> ai::GenerationConfig {
        let mut config = self.task.agent_config.llm_config.generation_config.clone();
        if self.is_native_mode() {
            config = config.with_tool_config(self.build_native_tool_config());
        }
        config
    }

    fn tool_call_args_to_map(tc: &ai::ToolCall) -> HashMap<String, Value> {
        if let Value::Object(map) = &tc.arguments {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            HashMap::new()
        }
    }

    fn reset_llm_recovery(&mut self) {
        self.llm_interaction_attempts = 0;
        self.recovery_started = None;
        self.capacity_recovery_started = None;
        self.capacity_retry_attempts = 0;
        self.indefinite_interactions = 0;
        self.paused_recovery_elapsed = None;
    }

    /// Ordinary recovery attempts in this episode: requests made, less those
    /// that ended in a wait that doesn't count against the ordinary limits.
    fn ordinary_attempts(&self) -> u32 {
        self.llm_interaction_attempts
            .saturating_sub(self.indefinite_interactions)
            .max(1)
    }

    /// Stops the ordinary elapsed-recovery clock (and its deadline) while an
    /// indefinite wait lasts; see [`resume_ordinary_recovery_clock`](Self::resume_ordinary_recovery_clock).
    fn pause_ordinary_recovery_clock(&mut self) {
        if let Some(started) = self.recovery_started.take() {
            self.paused_recovery_elapsed = Some(
                self.paused_recovery_elapsed
                    .unwrap_or_default()
                    .saturating_add(started.elapsed()),
            );
        }
    }

    /// Restarts a paused ordinary recovery clock with the time it had
    /// already used.
    fn resume_ordinary_recovery_clock(&mut self) {
        if self.recovery_started.is_none()
            && let Some(used) = self.paused_recovery_elapsed.take()
        {
            let now = tokio::time::Instant::now();
            self.recovery_started = Some(now.checked_sub(used).unwrap_or(now));
        }
    }

    fn recovery_deadline(&self) -> Option<tokio::time::Instant> {
        self.recovery_started.map(|start| {
            start + Duration::from_millis(self.task.agent_config.llm_retry_config.max_elapsed_ms)
        })
    }

    /// Handle control commands at every interruptible model/recovery wait.
    async fn handle_recovery_command(
        &mut self,
        command: Option<UserCommand>,
    ) -> Result<bool, ArkoAgentsError> {
        match command {
            Some(UserCommand::InjectContext { content }) => self.queue_context_injection(content),
            Some(UserCommand::CancelTask) => {
                self.task.set_state(TaskState::Failed {
                    reason: "Task cancelled by user.".into(),
                });
                self.stop_requested = true;
            }
            Some(UserCommand::SuspendTask) => self.stop_requested = true,
            None => {
                self.task.set_state(TaskState::Interrupted {
                    reason: "Command channel disconnected during model recovery.".into(),
                });
                self.stop_requested = true;
            }
            Some(_) => {}
        }
        if self.stop_requested {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        Ok(self.stop_requested)
    }

    /// Refresh and model stream initialization share one attempt and deadline.
    async fn run_llm_attempt<T>(
        &mut self,
        operation: impl std::future::Future<Output = Result<T, ai::Error>>,
    ) -> Result<Option<T>, ArkoAgentsError> {
        self.llm_interaction_attempts = self.llm_interaction_attempts.saturating_add(1);
        let deadline = self.recovery_deadline();
        tokio::pin!(operation);
        loop {
            tokio::select! {
                biased;
                _ = wait_for_recovery_deadline(deadline) => {
                    self.finish_llm_recovery("Elapsed model recovery budget exhausted during request.".into(), None).await?;
                    return Ok(None);
                }
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(None); }
                }
                result = &mut operation => {
                    return match result {
                        Ok(value) => Ok(Some(value)),
                        Err(ai::Error::Refusal(refusal)) => {
                            self.handle_refusal(*refusal, ai::Message::new_assistant("")).await?;
                            Ok(None)
                        }
                        Err(error) => { self.handle_llm_error(error, None).await?; Ok(None) }
                    };
                }
            }
        }
    }

    async fn emit_throttled_text(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.text_buffer.is_empty()
            && (self.event_throttle_duration.is_zero()
                || self.last_text_emit_time.elapsed() >= self.event_throttle_duration)
        {
            self.emit_event(AgentEvent::AssistantResponseChunk(self.text_buffer.clone()))
                .await?;
            self.text_buffer.clear();
            self.last_text_emit_time = Instant::now();
        }
        Ok(())
    }

    async fn flush_throttled_text(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.text_buffer.is_empty() {
            self.emit_event(AgentEvent::AssistantResponseChunk(self.text_buffer.clone()))
                .await?;
            self.text_buffer.clear();
        }
        Ok(())
    }

    async fn emit_throttled_reasoning(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.reasoning_buffer.is_empty()
            && (self.event_throttle_duration.is_zero()
                || self.last_reasoning_emit_time.elapsed() >= self.event_throttle_duration)
        {
            self.emit_event(AgentEvent::ReasoningChunk(self.reasoning_buffer.clone()))
                .await?;
            self.reasoning_buffer.clear();
            self.last_reasoning_emit_time = Instant::now();
        }
        Ok(())
    }

    async fn flush_throttled_reasoning(&mut self) -> Result<(), ArkoAgentsError> {
        if !self.reasoning_buffer.is_empty() {
            self.emit_event(AgentEvent::ReasoningChunk(self.reasoning_buffer.clone()))
                .await?;
            self.reasoning_buffer.clear();
        }
        Ok(())
    }

    fn flush_native_results_to_conversation(&mut self, extra_text: Option<&str>) {
        let results = std::mem::take(&mut self.task.pending_native_tool_results);
        if results.is_empty() && extra_text.is_none() && self.task.pending_tool_images.is_empty() {
            return;
        }
        let mut content: Vec<_> = results.into_iter().map(ai::Content::ToolResult).collect();
        if let Some(text) = extra_text {
            content.push(ai::Content::Text(text.to_string()));
        }
        content.extend(self.take_tool_image_content());
        self.task.conversation.add_user_message_with_contents(content);
        self.task.increment_user_message_count();
    }

    fn queue_tool_images(
        &mut self,
        tool: &str,
        call_id: impl std::fmt::Display,
        images: Vec<ToolImage>,
    ) {
        self.task
            .pending_tool_images
            .extend(images.into_iter().map(|mut attachment| {
                attachment.description = format!(
                    "Image from tool {tool}, call {call_id}: {}",
                    attachment.description
                );
                attachment
            }));
    }

    fn take_tool_image_content(&mut self) -> Vec<ai::Content> {
        std::mem::take(&mut self.task.pending_tool_images)
            .into_iter()
            .flat_map(|attachment| {
                [
                    ai::Content::Text(attachment.description),
                    ai::Content::Image(attachment.image),
                ]
            })
            .collect()
    }

    fn add_text_tool_results_to_conversation(&mut self, text: &str) {
        let mut content = vec![ai::Content::Text(text.to_string())];
        content.extend(self.take_tool_image_content());
        self.task.conversation.add_user_message_with_contents(content);
    }

    /// Records model call usage on the task (computing the input delta)
    /// and emits the `ModelCallUsageReported` event.
    async fn record_and_emit_model_call_usage(
        &mut self,
        raw_usage: ai::ModelCallUsage,
    ) -> Result<(), ArkoAgentsError> {
        let usage_with_delta = self.task.record_model_call_usage(raw_usage);
        let plan_usage = usage_with_delta.plan_usage.clone();
        self.emit_event(AgentEvent::ModelCallUsageReported(usage_with_delta))
            .await?;
        if let Some(plan_usage) = plan_usage {
            self.emit_event(AgentEvent::PlanUsageReported(plan_usage)).await?;
        }
        Ok(())
    }

    /// Applies tool result interception if configured and the tool is eligible.
    ///
    /// Returns `(effective_result, was_intercepted)`. When no interceptor is
    /// configured, or the tool is in the exclusion list, returns the raw result
    /// unchanged. On interceptor failure, retries according to the configured
    /// `InterceptionFailurePolicy` and falls back to the raw result.
    async fn apply_tool_result_interception(
        &self,
        tool_name: &str,
        raw_result: &str,
        params: &HashMap<String, Value>,
    ) -> (String, bool) {
        if self
            .tool_registry
            .get(tool_name)
            .is_some_and(|tool| !tool.allow_result_interception())
        {
            return (raw_result.to_string(), false);
        }
        let Some(ref config) = self.interceptor_config else {
            return (raw_result.to_string(), false);
        };

        if INTERCEPTION_EXCLUDED_TOOLS.contains(&tool_name) {
            return (raw_result.to_string(), false);
        }

        let task_context = self.task_context();

        let max_attempts = match config.failure_policy {
            InterceptionFailurePolicy::UseRawResult => 1,
            InterceptionFailurePolicy::RetryThenUseRawResult { max_attempts } => {
                max_attempts.max(1)
            }
        };

        for attempt in 1..=max_attempts {
            match config
                .interceptor
                .intercept(tool_name, raw_result, params, &task_context)
                .await
            {
                Ok(InterceptedResult::Unchanged) => return (raw_result.to_string(), false),
                Ok(InterceptedResult::Replaced(new_result)) => return (new_result, true),
                Err(e) => {
                    log::warn!(
                        "Task {}: Tool result interceptor failed for '{}' (attempt {}/{}): {}",
                        self.task.id,
                        tool_name,
                        attempt,
                        max_attempts,
                        e
                    );
                }
            }
        }

        log::warn!(
            "Task {}: Interceptor exhausted all attempts for '{}', using raw result.",
            self.task.id,
            tool_name
        );
        (raw_result.to_string(), false)
    }

    /// Emits `ToolResult` with the original output and, when interception
    /// modified the result, a follow-up `ToolResultIntercepted` event.
    async fn emit_tool_result_events(
        &self,
        tool_call_id: Uuid,
        tool_name: &str,
        original_output: &str,
        effective_output: &str,
        was_intercepted: bool,
    ) -> Result<(), ArkoAgentsError> {
        self.emit_event(AgentEvent::ToolResult {
            tool_call_id,
            tool_name: tool_name.to_string(),
            output: original_output.to_string(),
        })
        .await?;

        if was_intercepted {
            self.emit_event(AgentEvent::ToolResultIntercepted {
                tool_call_id,
                tool_name: tool_name.to_string(),
                original_output: original_output.to_string(),
                intercepted_output: effective_output.to_string(),
            })
            .await?;
        }

        Ok(())
    }

    /// Runs the agent loop to completion.
    pub async fn run(mut self) -> Result<Task, ArkoAgentsError> {
        self.task.agent_config.llm_retry_config.validate()?;
        log::info!(
            "Agent loop starting for task ID: {} (mode: {:?})",
            self.task.id,
            self.task.agent_config.tool_config.tool_calling_mode
        );

        if matches!(self.task.current_state, TaskState::Pending)
            || matches!(self.task.current_state, TaskState::Interrupted { .. })
        {
            self.task.set_state(TaskState::InProgress);
        }

        if matches!(self.task.current_state, TaskState::InProgress) {
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }

        if self.task.conversation.system_message.is_none() {
            let mut tool_defs_for_prompt: Vec<ToolDefinition> = Vec::new();
            for name in &self.task.agent_config.tool_config.available_tool_names {
                if let Some(tool) = self.tool_registry.get(name) {
                    tool_defs_for_prompt.push(tool.definition());
                } else {
                    log::warn!(
                        "Agent {} configured with tool '{}' but it was not found in the registry.",
                        self.task.agent_config.name,
                        name
                    );
                }
            }
            if !tool_defs_for_prompt
                .iter()
                .any(|def| def.name == COMPLETE_TASK_TOOL_NAME)
                && let Some(ct_tool) = self.tool_registry.get(COMPLETE_TASK_TOOL_NAME)
            {
                tool_defs_for_prompt.push(ct_tool.definition());
            }

            let prompt_builder = PromptBuilder::new(
                &self.task.agent_config,
                tool_defs_for_prompt.iter().collect(),
                &self.parsing_config,
                self.skill_prompt_context.as_ref(),
            );
            let system_prompt = prompt_builder.build_system_prompt(&self.task)?;
            self.task.conversation.system_message = Some(system_prompt);

            if self.task.conversation.messages.is_empty()
                && !self.task.initial_task_prompt.is_empty()
            {
                let initial_user_prompt = prompt_builder
                    .format_initial_task_prompt(&self.task, &self.task.initial_task_prompt)?;
                self.task
                    .conversation
                    .add_user_message(&initial_user_prompt);
                self.task.increment_user_message_count();
                self.reset_llm_recovery();
            }
        }

        'main_loop: loop {
            if self.stop_requested {
                break 'main_loop;
            }
            if self.task.current_state.is_terminal() {
                log::info!(
                    "Task {} reached terminal state: {:?}",
                    self.task.id,
                    self.task.current_state
                );
                break 'main_loop;
            }

            if self.task.current_state.is_awaiting_approval() {
                let command_future = self.command_receiver.recv();
                let command_result =
                    if let Some(seconds) = self.task.agent_config.user_command_timeout_seconds {
                        timeout(Duration::from_secs(seconds), command_future).await
                    } else {
                        Ok(command_future.await)
                    };

                match command_result {
                    Ok(Some(cmd)) => {
                        if let UserCommand::InjectContext { ref content } = cmd {
                            self.queue_context_injection(content.clone());
                            continue 'main_loop;
                        }
                        let mut next_user_message_from_command = String::new();
                        let mut native_handled = false;
                        let original_state_before_command_processing =
                            self.task.current_state.clone();

                        match self.task.current_state.clone() {
                            TaskState::AwaitingToolApproval {
                                tool_call_id,
                                tool_name,
                                parameters,
                                tool_description: _,
                            } => match cmd {
                                UserCommand::ApproveToolUse {
                                    tool_call_id: cmd_tool_id,
                                    ref custom_tool_output,
                                    ref message_to_agent,
                                } => {
                                    if cmd_tool_id != tool_call_id {
                                        log::warn!(
                                            "Task {}: Received ApproveToolUse for unexpected tool_call_id. Ignoring.",
                                            self.task.id
                                        );
                                        continue 'main_loop;
                                    }

                                    if self.is_native_mode() {
                                        if let Some(ai_call) =
                                            self.task.pending_native_tool_calls.remove(&tool_call_id)
                                        {
                                            let result = self
                                                .execute_native_tool_for_approval(
                                                    &ai_call,
                                                    tool_call_id,
                                                    &tool_name,
                                                    parameters,
                                                    custom_tool_output.clone(),
                                                    message_to_agent.clone(),
                                                )
                                                .await?;
                                            if let Some(r) = result {
                                                self.task.pending_native_tool_results.push(r);
                                            }
                                        }
                                        if matches!(
                                            self.task.current_state,
                                            TaskState::AwaitingToolApproval { .. }
                                        ) {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                        if !self.task.current_state.is_awaiting_approval() {
                                            self.flush_native_results_to_conversation(None);
                                        }
                                        native_handled = true;
                                    } else {
                                        match self
                                            .execute_tool_and_handle_result(
                                                &tool_name,
                                                parameters,
                                                tool_call_id,
                                                custom_tool_output.clone(),
                                                message_to_agent.clone(),
                                            )
                                            .await
                                        {
                                            Ok(msg_part) => {
                                                next_user_message_from_command = msg_part
                                            }
                                            Err(e) => {
                                                log::error!(
                                                    "Task {}: Error during approved tool execution: {}",
                                                    self.task.id,
                                                    e
                                                );
                                            }
                                        }
                                        if matches!(self.task.current_state, TaskState::AwaitingToolApproval { tool_call_id: current_id, .. } if current_id == tool_call_id)
                                        {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                    }
                                }
                                UserCommand::RejectToolUse {
                                    tool_call_id: cmd_tool_id,
                                    ref reason,
                                } => {
                                    if cmd_tool_id != tool_call_id {
                                        log::warn!(
                                            "Task {}: Received RejectToolUse for unexpected tool_call_id. Ignoring.",
                                            self.task.id
                                        );
                                        self.emit_event(AgentEvent::TaskStateChanged(
                                            self.task.current_state.clone(),
                                        ))
                                        .await?;
                                        continue 'main_loop;
                                    }
                                    self.emit_event(AgentEvent::ToolRejected {
                                        tool_call_id,
                                        tool_name: tool_name.clone(),
                                        reason: (*reason).clone(),
                                    })
                                    .await?;

                                    if self.is_native_mode() {
                                        if let Some(ai_call) =
                                            self.task.pending_native_tool_calls.remove(&tool_call_id)
                                        {
                                            let reason_str =
                                                reason.as_deref().unwrap_or("Rejected by user");
                                            self.task.pending_native_tool_results.push(
                                                ai::ToolResult::error_from_tool_call(
                                                    &ai_call, reason_str,
                                                ),
                                            );
                                        }
                                        self.task.set_state(TaskState::InProgress);
                                        self.flush_native_results_to_conversation(None);
                                        native_handled = true;
                                    } else {
                                        let pb = PromptBuilder::new(
                                            &self.task.agent_config,
                                            vec![],
                                            &self.parsing_config,
                                            None,
                                        );
                                        next_user_message_from_command = pb
                                            .format_tool_rejection_as_user_message(
                                                &tool_name,
                                                (*reason).as_deref(),
                                            );
                                        self.task.set_state(TaskState::InProgress);
                                    }
                                }
                                UserCommand::CancelTask => {}
                                UserCommand::SuspendTask => {}
                                _ => {
                                    log::warn!(
                                        "Task {}: Received inappropriate command while AwaitingToolApproval. Ignoring.",
                                        self.task.id
                                    );
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    continue 'main_loop;
                                }
                            },
                            TaskState::AwaitingBatchToolApproval {
                                batch_id,
                                ref pending_calls,
                            } => {
                                if let UserCommand::SubmitBatchToolApprovals {
                                    batch_id: cmd_batch_id,
                                    ref decisions,
                                } = cmd
                                {
                                    if cmd_batch_id != batch_id {
                                        log::warn!(
                                            "Task {}: Received SubmitBatchToolApprovals for unexpected batch_id. Ignoring.",
                                            self.task.id
                                        );
                                        continue 'main_loop;
                                    }

                                    if self.is_native_mode() {
                                        for pending_call_info in pending_calls {
                                            match decisions.get(&pending_call_info.tool_call_id) {
                                                Some(ToolApprovalDecision::Approve {
                                                    custom_tool_output,
                                                    message_to_agent,
                                                }) => {
                                                    if let Some(ai_call) = self.task
                                                        .pending_native_tool_calls
                                                        .remove(&pending_call_info.tool_call_id)
                                                    {
                                                        let result = self
                                                            .execute_native_tool_for_approval(
                                                                &ai_call,
                                                                pending_call_info.tool_call_id,
                                                                &pending_call_info.tool_name,
                                                                pending_call_info
                                                                    .parameters
                                                                    .clone(),
                                                                custom_tool_output.clone(),
                                                                message_to_agent.clone(),
                                                            )
                                                            .await?;
                                                        if let Some(r) = result {
                                                            self.task.pending_native_tool_results.push(r);
                                                        }
                                                    }
                                                }
                                                Some(ToolApprovalDecision::Reject { reason }) => {
                                                    self.emit_event(AgentEvent::ToolRejected {
                                                        tool_call_id: pending_call_info
                                                            .tool_call_id,
                                                        tool_name: pending_call_info
                                                            .tool_name
                                                            .clone(),
                                                        reason: (*reason).clone(),
                                                    })
                                                    .await?;
                                                    if let Some(ai_call) = self.task
                                                        .pending_native_tool_calls
                                                        .remove(&pending_call_info.tool_call_id)
                                                    {
                                                        let reason_str = reason
                                                            .as_deref()
                                                            .unwrap_or("Rejected by user");
                                                        self.task.pending_native_tool_results.push(
                                                            ai::ToolResult::error_from_tool_call(
                                                                &ai_call, reason_str,
                                                            ),
                                                        );
                                                    }
                                                }
                                                None => {
                                                    if let Some(ai_call) = self.task
                                                        .pending_native_tool_calls
                                                        .remove(&pending_call_info.tool_call_id)
                                                    {
                                                        self.task.pending_native_tool_results.push(ai::ToolResult::error_from_tool_call(&ai_call, "No decision provided, implicitly rejected"));
                                                    }
                                                }
                                            }
                                            if self.task.current_state.is_terminal() {
                                                break;
                                            }
                                        }
                                        if matches!(
                                            self.task.current_state,
                                            TaskState::AwaitingBatchToolApproval { .. }
                                        ) {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                        if !self.task.current_state.is_awaiting_approval() {
                                            self.flush_native_results_to_conversation(None);
                                        }
                                        native_handled = true;
                                    } else {
                                        let mut combined_results_for_llm = Vec::new();
                                        for pending_call_info in pending_calls {
                                            match decisions.get(&pending_call_info.tool_call_id) {
                                                Some(ToolApprovalDecision::Approve {
                                                    custom_tool_output,
                                                    message_to_agent,
                                                }) => {
                                                    match self
                                                        .execute_tool_and_handle_result(
                                                            &pending_call_info.tool_name,
                                                            pending_call_info.parameters.clone(),
                                                            pending_call_info.tool_call_id,
                                                            custom_tool_output.clone(),
                                                            message_to_agent.clone(),
                                                        )
                                                        .await
                                                    {
                                                        Ok(msg_part) => {
                                                            combined_results_for_llm.push(msg_part)
                                                        }
                                                        Err(e) => log::error!(
                                                            "Task {}: Error during approved batched tool execution: {}",
                                                            self.task.id,
                                                            e
                                                        ),
                                                    }
                                                }
                                                Some(ToolApprovalDecision::Reject { reason }) => {
                                                    self.emit_event(AgentEvent::ToolRejected {
                                                        tool_call_id: pending_call_info
                                                            .tool_call_id,
                                                        tool_name: pending_call_info
                                                            .tool_name
                                                            .clone(),
                                                        reason: (*reason).clone(),
                                                    })
                                                    .await?;
                                                    let pb = PromptBuilder::new(
                                                        &self.task.agent_config,
                                                        vec![],
                                                        &self.parsing_config,
                                                        None,
                                                    );
                                                    combined_results_for_llm.push(
                                                        pb.format_tool_rejection_as_user_message(
                                                            &pending_call_info.tool_name,
                                                            (*reason).as_deref(),
                                                        ),
                                                    );
                                                }
                                                None => {
                                                    let pb = PromptBuilder::new(
                                                        &self.task.agent_config,
                                                        vec![],
                                                        &self.parsing_config,
                                                        None,
                                                    );
                                                    combined_results_for_llm.push(pb.format_tool_rejection_as_user_message(&pending_call_info.tool_name, Some("Implicitly rejected: No decision provided.")));
                                                }
                                            }
                                            if !matches!(
                                                self.task.current_state,
                                                TaskState::AwaitingBatchToolApproval { .. }
                                            ) {
                                                break;
                                            }
                                        }
                                        next_user_message_from_command =
                                            combined_results_for_llm.join("\n\n---\n\n");
                                        if matches!(self.task.current_state, TaskState::AwaitingBatchToolApproval { batch_id: current_state_batch_id, .. } if current_state_batch_id == cmd_batch_id)
                                        {
                                            self.task.set_state(TaskState::InProgress);
                                        }
                                    }
                                } else if matches!(cmd, UserCommand::CancelTask) {
                                } else {
                                    log::warn!(
                                        "Task {}: Received inappropriate command while AwaitingBatchToolApproval. Ignoring.",
                                        self.task.id
                                    );
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    continue 'main_loop;
                                }
                            }
                            TaskState::AwaitingUserResponse { ref questions } => match cmd {
                                UserCommand::SubmitUserResponse { ref response } => {
                                    log::info!("Task {}: User response received.", self.task.id);
                                    let pb = PromptBuilder::new(
                                        &self.task.agent_config,
                                        vec![],
                                        &self.parsing_config,
                                        None,
                                    );
                                    let formatted = pb
                                        .format_user_response_as_user_message(questions, response);
                                    self.task.set_state(TaskState::InProgress);

                                    if self.is_native_mode() {
                                        self.flush_native_results_to_conversation(Some(&formatted));
                                        native_handled = true;
                                    } else {
                                        next_user_message_from_command = formatted;
                                    }
                                }
                                UserCommand::CancelTask => {}
                                UserCommand::SuspendTask => {}
                                _ => {
                                    log::warn!(
                                        "Task {}: Received inappropriate command while AwaitingUserResponse. Ignoring.",
                                        self.task.id
                                    );
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    continue 'main_loop;
                                }
                            },
                            TaskState::AwaitingErrorRecoveryApproval { ref error_message } => {
                                match cmd {
                                    UserCommand::ApproveErrorRecovery {
                                        message: ref user_msg_on_recovery,
                                    } => {
                                        log::info!(
                                            "Task {}: Error recovery approved by user.",
                                            self.task.id
                                        );
                                        let mut recovery_prompt = format!(
                                            "The previous attempt resulted in an error: {}. The user has approved to continue.",
                                            error_message
                                        );
                                        if let Some(user_input) = user_msg_on_recovery
                                            && !(*user_input).is_empty()
                                        {
                                            recovery_prompt.push_str("\nUser provided additional context for recovery:\n");
                                            recovery_prompt.push_str(user_input);
                                        }

                                        if self.is_native_mode()
                                            && !self.task.pending_native_tool_results.is_empty()
                                        {
                                            self.flush_native_results_to_conversation(Some(
                                                &recovery_prompt,
                                            ));
                                            native_handled = true;
                                        } else {
                                            next_user_message_from_command = recovery_prompt;
                                        }
                                        self.task.set_state(TaskState::InProgress);
                                        self.reset_llm_recovery();
                                    }
                                    UserCommand::RejectErrorRecovery { message: _ } => {
                                        log::info!(
                                            "Task {}: Error recovery rejected by user.",
                                            self.task.id
                                        );
                                        self.task.set_state(TaskState::Failed {
                                            reason: format!(
                                                "Error recovery rejected by user for error: {}",
                                                error_message
                                            ),
                                        });
                                    }
                                    UserCommand::CancelTask => {}
                                    UserCommand::SuspendTask => {}
                                    _ => {
                                        log::warn!(
                                            "Task {}: Received inappropriate command while AwaitingErrorRecoveryApproval. Ignoring.",
                                            self.task.id
                                        );
                                        self.emit_event(AgentEvent::TaskStateChanged(
                                            self.task.current_state.clone(),
                                        ))
                                        .await?;
                                        continue 'main_loop;
                                    }
                                }
                            }
                            _ => {}
                        }

                        if let UserCommand::CancelTask = cmd {
                            log::info!("Task {}: Cancelled by user command.", self.task.id);
                            self.task.set_state(TaskState::Failed {
                                reason: "Task cancelled by user.".to_string(),
                            });
                        }

                        if let UserCommand::SuspendTask = cmd {
                            self.emit_event(AgentEvent::TaskStateChanged(
                                self.task.current_state.clone(),
                            ))
                            .await?;
                            break 'main_loop;
                        }

                        if self.task.current_state != original_state_before_command_processing {
                            self.emit_event(AgentEvent::TaskStateChanged(
                                self.task.current_state.clone(),
                            ))
                            .await?;
                        }

                        if self.task.current_state.is_terminal() {
                            break 'main_loop;
                        }

                        if matches!(self.task.current_state, TaskState::InProgress)
                            && !native_handled
                        {
                            if !next_user_message_from_command.is_empty() {
                                let pb = PromptBuilder::new(
                                    &self.task.agent_config,
                                    vec![],
                                    &self.parsing_config,
                                    None,
                                );
                                let next_msg_num = self.task.user_message_count + 1;
                                let augmented_msg = pb
                                    .augment_user_message_with_env_details_and_reminder(
                                        &self.task,
                                        &next_user_message_from_command,
                                        next_msg_num,
                                    )?;
                                self.add_text_tool_results_to_conversation(&augmented_msg);
                                self.task.increment_user_message_count();
                                self.reset_llm_recovery();
                            } else {
                                let pb = PromptBuilder::new(
                                    &self.task.agent_config,
                                    vec![],
                                    &self.parsing_config,
                                    None,
                                );
                                let generic_prompt =
                                    "Please continue with the task based on the previous actions.";
                                let next_msg_num = self.task.user_message_count + 1;
                                let augmented_generic_prompt = pb
                                    .augment_user_message_with_env_details_and_reminder(
                                        &self.task,
                                        generic_prompt,
                                        next_msg_num,
                                    )?;
                                self.task
                                    .conversation
                                    .add_user_message(&augmented_generic_prompt);
                                self.task.increment_user_message_count();
                                self.reset_llm_recovery();
                            }
                        }

                        if self.task.current_state.is_awaiting_approval() {
                            continue 'main_loop;
                        }
                    }
                    Ok(None) => {
                        let reason =
                            "Command channel disconnected during approval state.".to_string();
                        log::error!("Task {}: {}", self.task.id, reason);
                        self.task.set_state(TaskState::Interrupted {
                            reason: reason.clone(),
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        self.emit_event(AgentEvent::ErrorEncountered {
                            error_message: reason,
                            requires_approval_to_retry: false,
                            tool_call_id: None,
                        })
                        .await?;
                        break 'main_loop;
                    }
                    Err(_timeout_err) => {
                        let seconds = self
                            .task
                            .agent_config
                            .user_command_timeout_seconds
                            .unwrap_or(0);
                        let reason = format!(
                            "Timeout ({}s) waiting for user command in state {:?}.",
                            seconds, self.task.current_state
                        );
                        log::error!("Task {}: {}", self.task.id, reason);
                        self.task.set_state(TaskState::Interrupted {
                            reason: reason.clone(),
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        self.emit_event(AgentEvent::ErrorEncountered {
                            error_message: reason,
                            requires_approval_to_retry: false,
                            tool_call_id: None,
                        })
                        .await?;
                        break 'main_loop;
                    }
                }
            } else {
                match self.command_receiver.try_recv() {
                    Ok(UserCommand::CancelTask) => {
                        self.task.set_state(TaskState::Failed {
                            reason: "Task cancelled by user.".to_string(),
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        break 'main_loop;
                    }
                    Ok(UserCommand::SuspendTask) => {
                        break 'main_loop;
                    }
                    Ok(UserCommand::InjectContext { content }) => {
                        self.queue_context_injection(content);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => {}
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        let reason = "Command channel disconnected.".to_string();
                        log::error!("Task {}: {}", self.task.id, reason);
                        self.task.set_state(TaskState::Interrupted {
                            reason: reason.clone(),
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        self.emit_event(AgentEvent::ErrorEncountered {
                            error_message: reason,
                            requires_approval_to_retry: false,
                            tool_call_id: None,
                        })
                        .await?;
                        break 'main_loop;
                    }
                    Ok(other_cmd) => {
                        log::warn!(
                            "Task {}: Unexpected command {:?} received outside of an approval state. Ignoring.",
                            self.task.id,
                            other_cmd
                        );
                    }
                }
            }

            if !matches!(self.task.current_state, TaskState::InProgress) {
                if self.task.current_state.is_terminal() {
                    break 'main_loop;
                }
                tokio_sleep(Duration::from_millis(100)).await;
                continue 'main_loop;
            }

            self.flush_context_injections()?;

            self.refresh_file_tree_context();
            self.ensure_user_turn_before_request()?;
            if !self.apply_plan_usage_threshold().await? {
                continue 'main_loop;
            }
            self.observe_committed_tool_context();

            let llm_client = Arc::clone(&self.llm_client);
            let conversation = self.task.conversation.clone();
            let gen_config = self.build_generation_config();
            #[cfg(test)]
            let scripted_response = self.scripted_responses.get_mut().unwrap().pop_front();
            let request = async move {
                #[cfg(test)]
                if let Some(response) = scripted_response {
                    return response;
                }
                if conversation.model.provider == ai::Provider::OpenAI
                    && llm_client
                        .is_openai_token_expired(Duration::from_secs(OAUTH_REFRESH_MARGIN_SECS))
                        .await
                {
                    llm_client.refresh_openai_oauth_token_once().await?;
                }
                llm_client.generate_stream(&conversation, gen_config).await
            };
            let Some(mut response_stream) = self.run_llm_attempt(request).await? else {
                continue 'main_loop;
            };

            let mut raw_events_for_turn: Vec<RawParserEvent> = Vec::new();
            let mut available_tool_names_hs: HashSet<String> = self
                .task
                .agent_config
                .tool_config
                .available_tool_names
                .iter()
                .cloned()
                .collect();
            available_tool_names_hs.insert(COMPLETE_TASK_TOOL_NAME.to_string());

            let mut parser = if !self.is_native_mode() {
                Some(ToolCallParser::new(
                    available_tool_names_hs,
                    self.parsing_config.clone(),
                ))
            } else {
                None
            };

            let mut had_stream_error = false;
            let mut final_generation_response: Option<ai::GenerationResponse> = None;
            let mut current_assistant_response_text_for_conv = String::new();
            let mut native_stream_call_ids: HashMap<usize, (Uuid, String)> = HashMap::new();
            let mut streamed_refusal: Option<ai::Refusal> = None;

            let stream_recovery_deadline = self.recovery_deadline();
            loop {
                tokio::select! {
                    biased;
                    _ = wait_for_recovery_deadline(stream_recovery_deadline) => {
                        if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                            self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                        }
                        self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                        self.finish_llm_recovery("Elapsed model recovery budget exhausted during stream.".into(), None).await?;
                        had_stream_error = true;
                        break;
                    }
                    maybe_cmd = self.command_receiver.recv() => {
                        match maybe_cmd {
                            Some(UserCommand::CancelTask) => {
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                                self.task.set_state(TaskState::Failed { reason: "Task cancelled by user.".to_string() });
                                self.emit_event(AgentEvent::TaskStateChanged(self.task.current_state.clone())).await?;
                                had_stream_error = true;
                                break;
                            }
                            Some(UserCommand::SuspendTask) => {
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                self.reject_announced_calls(&mut native_stream_call_ids, INTERRUPTED_CALL).await?;
                                self.stop_requested = true;
                                had_stream_error = true;
                                break;
                            }
                            Some(UserCommand::InjectContext { content }) => {
                                self.queue_context_injection(content);
                            }
                            Some(_) => {}
                            None => {
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                self.stop_requested = true;
                                let reason = "Command channel disconnected during LLM streaming.".to_string();
                                self.task.set_state(TaskState::Interrupted { reason: reason.clone() });
                                self.emit_event(AgentEvent::TaskStateChanged(self.task.current_state.clone())).await?;
                                self.emit_event(AgentEvent::ErrorEncountered { error_message: reason, requires_approval_to_retry: false, tool_call_id: None }).await?;
                                had_stream_error = true;
                                break;
                            }
                        }
                    }

                    stream_item = response_stream.try_next() => {
                        match stream_item {
                            Ok(Some(ai::StreamEvent::TextChunk(text_chunk))) => {
                                current_assistant_response_text_for_conv.push_str(&text_chunk);
                                if self.is_native_mode() {
                                    self.text_buffer.push_str(&text_chunk);
                                    self.emit_throttled_text().await?;
                                } else {
                                    raw_events_for_turn.extend(parser.as_mut().unwrap().ingest_chunk(&text_chunk));
                                }
                            }
                            Ok(Some(ai::StreamEvent::ReasoningChunk(reasoning))) => {
                                self.reasoning_buffer.push_str(&reasoning);
                                self.emit_throttled_reasoning().await?;
                            }
                            Ok(Some(ai::StreamEvent::ToolCallStart { index, name })) => {
                                if self.is_native_mode() {
                                    self.flush_throttled_text().await?;
                                    let uuid = Uuid::new_v4();
                                    native_stream_call_ids.insert(index, (uuid, name.clone()));
                                    self.emit_event(AgentEvent::ToolCallInitiated {
                                        tool_call_id: uuid,
                                        tool_name: name,
                                    }).await?;
                                }
                            }
                            Ok(Some(ai::StreamEvent::ToolCallArgumentsDelta { .. })) => {}
                            // Providers repeat the refusal on `End`; it is
                            // kept only for a stream that ends without one.
                            Ok(Some(ai::StreamEvent::Refusal(refusal))) => {
                                streamed_refusal = Some(refusal);
                            }
                            Ok(Some(ai::StreamEvent::End(boxed_generation_response))) => {
                                let generation_response = *boxed_generation_response;
                                self.task.usage_history.push(generation_response.usage.clone());
                                self.emit_event(AgentEvent::UsageReported(generation_response.usage.clone())).await?;
                                self.record_and_emit_model_call_usage(generation_response.model_call_usage.clone()).await?;
                                // A refused turn is reported, never retried
                                // unchanged, and never treated as a turn that
                                // failed to call a tool.
                                if let Some(refusal) = generation_response.refusal.clone() {
                                    // The call's routing token still belongs
                                    // to this turn, as for any other response.
                                    if let Some(token) = &generation_response.openai_turn_state {
                                        self.task.conversation.record_openai_turn_state(token);
                                    }
                                    if let Some(ref mut p) = parser {
                                        raw_events_for_turn.extend(p.finalize());
                                    }
                                    self.handle_refusal(refusal, generation_response.message).await?;
                                    had_stream_error = true;
                                    break;
                                }
                                self.task.conversation.add_assistant_message_from_response(generation_response.clone());
                                final_generation_response = Some(generation_response);
                                if let Some(ref mut p) = parser {
                                    raw_events_for_turn.extend(p.finalize());
                                }
                                break;
                            }
                            Ok(None) => {
                                if let Some(ref mut p) = parser {
                                    raw_events_for_turn.extend(p.finalize());
                                }
                                if let Some(refusal) = streamed_refusal.take() {
                                    let partial = ai::Message::new_assistant(&current_assistant_response_text_for_conv);
                                    self.handle_refusal(refusal, partial).await?;
                                    had_stream_error = true;
                                    break;
                                }
                                // Without a final response no announced call runs.
                                self.reject_announced_calls(&mut native_stream_call_ids, FAILED_STREAM_CALL).await?;
                                if !current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.task.conversation.add_assistant_message(&current_assistant_response_text_for_conv);
                                }
                                if current_assistant_response_text_for_conv.is_empty() && final_generation_response.is_none() {
                                    self.handle_llm_error(ai::Error::ApiError("Stream ended prematurely with no output".to_string()), None).await?;
                                    had_stream_error = true;
                                }
                                break;
                            }
                            // A refusal reported as an error (Gemini
                            // generation-blocked codes): its partial text is
                            // not kept.
                            Err(ai::Error::Refusal(refusal)) => {
                                self.handle_refusal(*refusal, ai::Message::new_assistant("")).await?;
                                had_stream_error = true;
                                break;
                            }
                            Err(e) => {
                                if final_generation_response.is_none() {
                                    self.keep_partial_output(&current_assistant_response_text_for_conv, &e);
                                }
                                self.reject_announced_calls(&mut native_stream_call_ids, FAILED_STREAM_CALL).await?;
                                self.handle_llm_error(e, None).await?;
                                had_stream_error = true;
                                break;
                            }
                        }
                    }
                }
            }
            self.flush_throttled_reasoning().await?;
            self.flush_throttled_text().await?;

            if had_stream_error
                || self.task.current_state.is_terminal()
                || !matches!(self.task.current_state, TaskState::InProgress)
            {
                if self.task.current_state.is_terminal() {
                    break 'main_loop;
                }
                continue 'main_loop;
            }

            if self.is_native_mode() {
                let tool_calls = final_generation_response
                    .as_ref()
                    .map(|r| r.message.tool_calls.clone())
                    .unwrap_or_default();

                if tool_calls.is_empty() {
                    log::warn!(
                        "LLM response for task {} did not contain a native tool call.",
                        self.task.id
                    );
                    self.handle_llm_error(ai::Error::ApiError("Agent failed to produce a tool call. You MUST use one or more tools in every response, even if it is just 'complete_task'.".to_string()), None).await?;
                    if self.task.current_state.is_terminal() {
                        break 'main_loop;
                    }
                    continue 'main_loop;
                }

                self.reset_llm_recovery();
                let mut calls_needing_approval: Vec<PendingToolCallInfo> = Vec::new();
                let overall_batch_id = if tool_calls.len() > 1 {
                    Some(Uuid::new_v4())
                } else {
                    None
                };
                let rejected_batch_reason = native_batch_rejection_reason(
                    &self.task.agent_config,
                    &self.tool_registry,
                    &tool_calls,
                );

                for (call_index, ai_tool_call) in tool_calls.iter().enumerate() {
                    let tool_call_id = native_stream_call_ids
                        .remove(&call_index)
                        .map_or_else(Uuid::new_v4, |(id, _)| id);
                    let params_map = Self::tool_call_args_to_map(ai_tool_call);

                    self.emit_event(AgentEvent::ToolCallFormatted {
                        tool_call_id,
                        tool_name: ai_tool_call.name.clone(),
                        parameters: params_map.clone(),
                    })
                    .await?;

                    if let Some(reason) = &rejected_batch_reason {
                        self.emit_event(AgentEvent::ToolRejected {
                            tool_call_id,
                            tool_name: ai_tool_call.name.clone(),
                            reason: Some(reason.clone()),
                        })
                        .await?;
                        self.task.pending_native_tool_results
                            .push(ai::ToolResult::error_from_tool_call(ai_tool_call, reason));
                        continue;
                    }

                    if let Some(raw) = &ai_tool_call.invalid_arguments {
                        let reason = format!(
                            "The arguments for tool '{}' are not valid JSON, so the tool was not run. Call it again with a JSON object. Received: {}",
                            ai_tool_call.name, raw
                        );
                        self.emit_event(AgentEvent::ToolRejected {
                            tool_call_id,
                            tool_name: ai_tool_call.name.clone(),
                            reason: Some(reason.clone()),
                        })
                        .await?;
                        self.task.pending_native_tool_results
                            .push(ai::ToolResult::error_from_tool_call(ai_tool_call, &reason));
                        continue;
                    }

                    let tool_arc = match self.tool_registry.get(&ai_tool_call.name) {
                        Some(t) => t,
                        None => {
                            self.emit_event(AgentEvent::UnavailableToolAttempted {
                                tool_name: ai_tool_call.name.clone(),
                            })
                            .await?;
                            self.task.pending_native_tool_results.push(
                                ai::ToolResult::error_from_tool_call(
                                    ai_tool_call,
                                    &format!("Tool '{}' is not available.", ai_tool_call.name),
                                ),
                            );
                            continue;
                        }
                    };
                    let tool_definition = tool_arc.definition();

                    let mut is_out_of_workspace = false;
                    let mut offending_param_name: Option<String> = None;
                    for param_def in &tool_definition.parameters {
                        if param_def.is_path
                            && let Some(path_str) =
                                params_map.get(&param_def.name).and_then(|v| v.as_str())
                            && !path_str.is_empty()
                        {
                            match crate::utils::is_path_in_workspace(
                                path_str,
                                &self.task.current_working_directory,
                                &self.task.current_working_directory,
                            ) {
                                Ok(true) => {}
                                Ok(false) => {
                                    is_out_of_workspace = true;
                                    offending_param_name = Some(param_def.name.clone());
                                    break;
                                }
                                Err(e) => {
                                    self.task.pending_native_tool_results.push(
                                        ai::ToolResult::error_from_tool_call(
                                            ai_tool_call,
                                            &format!("Path validation error: {}", e),
                                        ),
                                    );
                                    continue;
                                }
                            }
                        }
                    }

                    let mut force_approval = false;
                    if is_out_of_workspace {
                        let path_policy = self
                            .task
                            .agent_config
                            .get_path_safety_policy(&ai_tool_call.name);
                        match path_policy {
                            PathSafetyPolicy::AutoReject => {
                                self.task.pending_native_tool_results.push(
                                    ai::ToolResult::error_from_tool_call(
                                        ai_tool_call,
                                        &format!(
                                            "Path for parameter '{}' is outside the workspace.",
                                            offending_param_name.unwrap_or_default()
                                        ),
                                    ),
                                );
                                continue;
                            }
                            PathSafetyPolicy::RequireApproval => {
                                force_approval = true;
                            }
                            PathSafetyPolicy::AutoApprove => {}
                        }
                    }

                    let normal_approval_needed = !self
                        .task
                        .agent_config
                        .should_auto_approve_tool(&ai_tool_call.name);

                    if force_approval || normal_approval_needed {
                        self.task.pending_native_tool_calls
                            .insert(tool_call_id, ai_tool_call.clone());
                        calls_needing_approval.push(PendingToolCallInfo {
                            tool_call_id,
                            tool_name: ai_tool_call.name.clone(),
                            parameters: params_map,
                            tool_description: tool_definition.description.clone(),
                        });
                    } else {
                        let result = self.execute_native_tool(ai_tool_call, tool_call_id).await?;
                        if let Some(r) = result {
                            self.task.pending_native_tool_results.push(r);
                        }
                        if !matches!(self.task.current_state, TaskState::InProgress) {
                            break;
                        }
                    }
                }

                if !calls_needing_approval.is_empty()
                    && matches!(self.task.current_state, TaskState::InProgress)
                {
                    if calls_needing_approval.len() == 1 && overall_batch_id.is_none() {
                        let single_pending = calls_needing_approval.remove(0);
                        self.task.set_state(TaskState::AwaitingToolApproval {
                            tool_call_id: single_pending.tool_call_id,
                            tool_name: single_pending.tool_name.clone(),
                            parameters: single_pending.parameters.clone(),
                            tool_description: single_pending.tool_description.clone(),
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        self.emit_event(AgentEvent::ToolCallAwaitingApproval {
                            tool_call_id: single_pending.tool_call_id,
                            tool_name: single_pending.tool_name,
                            parameters: single_pending.parameters,
                            tool_description: single_pending.tool_description,
                        })
                        .await?;
                    } else {
                        let batch_id_for_state = overall_batch_id.unwrap_or_else(Uuid::new_v4);
                        self.task.set_state(TaskState::AwaitingBatchToolApproval {
                            batch_id: batch_id_for_state,
                            pending_calls: calls_needing_approval,
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        self.emit_event(AgentEvent::BatchToolCallAwaitingApproval {
                            batch_id: batch_id_for_state,
                            pending_calls: match &self.task.current_state {
                                TaskState::AwaitingBatchToolApproval { pending_calls, .. } => {
                                    pending_calls.clone()
                                }
                                _ => vec![],
                            },
                        })
                        .await?;
                    }
                } else if matches!(
                    self.task.current_state,
                    TaskState::AwaitingUserResponse { .. }
                ) || !self.task.current_state.is_awaiting_approval()
                {
                    self.flush_native_results_to_conversation(None);
                }
            } else {
                let mut assistant_text_from_raw_events_for_pass1 = String::new();
                let mut parsed_call_ids = VecDeque::new();

                if !self.simplified_tool_events {
                    let mut current_tool_call_id_granular: Option<Uuid> = None;
                    let mut current_param_name_granular: Option<String> = None;

                    for raw_event in &raw_events_for_turn {
                        match raw_event {
                            RawParserEvent::Text(text) => {
                                assistant_text_from_raw_events_for_pass1.push_str(text);
                            }
                            RawParserEvent::Error(err) => {
                                log::warn!("Raw parser emitted error event: {}", err);
                            }
                            RawParserEvent::ToolCallStart(name) => {
                                let tool_call_id = Uuid::new_v4();
                                parsed_call_ids.push_back(tool_call_id);
                                current_tool_call_id_granular = Some(tool_call_id);
                                self.emit_event(AgentEvent::ToolCallInitiated {
                                    tool_call_id,
                                    tool_name: name.clone(),
                                })
                                .await?;
                            }
                            RawParserEvent::ParamName(p_name) => {
                                if let Some(id) = current_tool_call_id_granular {
                                    current_param_name_granular = Some(p_name.clone());
                                    self.emit_event(AgentEvent::ToolParamName {
                                        tool_call_id: id,
                                        param_name: p_name.clone(),
                                    })
                                    .await?;
                                }
                            }
                            RawParserEvent::ParamValueChunk(chunk) => {
                                if let Some(id) = current_tool_call_id_granular
                                    && let Some(ref param_name) = current_param_name_granular
                                {
                                    self.emit_event(AgentEvent::ToolParamValueChunk {
                                        tool_call_id: id,
                                        param_name: param_name.clone(),
                                        chunk: chunk.clone(),
                                    })
                                    .await?;
                                }
                            }
                            RawParserEvent::ParamEnd => {
                                if let Some(id) = current_tool_call_id_granular
                                    && let Some(ref param_name) = current_param_name_granular
                                {
                                    self.emit_event(AgentEvent::ToolParamEnd {
                                        tool_call_id: id,
                                        param_name: param_name.clone(),
                                    })
                                    .await?;
                                }
                                current_param_name_granular = None;
                            }
                            RawParserEvent::ToolCallEnd(_) => {
                                current_tool_call_id_granular = None;
                                current_param_name_granular = None;
                            }
                            RawParserEvent::BatchedToolCallStart
                            | RawParserEvent::BatchedToolCallEnd => {}
                        }
                    }
                    if !assistant_text_from_raw_events_for_pass1.is_empty() {
                        self.text_buffer
                            .push_str(&assistant_text_from_raw_events_for_pass1);
                        self.emit_throttled_text().await?;
                    }
                }
                self.flush_throttled_text().await?;

                let mut next_user_message_str_accumulated = String::new();

                match assemble_tool_calls(raw_events_for_turn) {
                    Ok((plain_text_from_assembler, mut tool_call_groups)) => {
                        if self.simplified_tool_events && !plain_text_from_assembler.is_empty() {
                            self.text_buffer.push_str(&plain_text_from_assembler);
                            self.emit_throttled_text().await?;
                            self.flush_throttled_text().await?;
                        }

                        if tool_call_groups.len() > 1 {
                            let error_msg = "Error: Only one tool call or one batch of the same tool can be used per turn.".to_string();
                            if !next_user_message_str_accumulated.is_empty() {
                                next_user_message_str_accumulated.push_str("\n\n");
                            }
                            next_user_message_str_accumulated.push_str(&error_msg);
                        } else if tool_call_groups.is_empty() {
                            if !self.task.current_state.is_terminal() {
                                self.handle_llm_error(
                                    ai::Error::ApiError(
                                        "Agent failed to produce a tool call.".to_string(),
                                    ),
                                    None,
                                )
                                .await?;
                                if self.task.current_state.is_terminal() {
                                    break 'main_loop;
                                }
                                continue 'main_loop;
                            }
                        } else {
                            let current_group = tool_call_groups.remove(0);
                            let calls_to_process_with_ids: Vec<(ParsedToolCall, Uuid)> =
                                match current_group {
                                    ToolCallGroup::Single(pc) => vec![(
                                        pc,
                                        parsed_call_ids.pop_front().unwrap_or_else(Uuid::new_v4),
                                    )],
                                    ToolCallGroup::Batch(pcs) => {
                                        if let Some(first_tool_name_in_batch) =
                                            pcs.first().map(|call| &call.name)
                                        {
                                            if let Some(tool_def_arc) =
                                                self.tool_registry.get(first_tool_name_in_batch)
                                            {
                                                if !self
                                                    .task
                                                    .agent_config
                                                    .allows_batched_calls_for_tool(
                                                        &tool_def_arc.definition(),
                                                    )
                                                {
                                                    let err_msg = format!(
                                                        "Error: Tool '{}' cannot be used in a batch.",
                                                        first_tool_name_in_batch
                                                    );
                                                    if !next_user_message_str_accumulated.is_empty()
                                                    {
                                                        next_user_message_str_accumulated
                                                            .push_str("\n\n");
                                                    }
                                                    next_user_message_str_accumulated
                                                        .push_str(&err_msg);
                                                    Vec::new()
                                                } else {
                                                    pcs.into_iter()
                                                        .map(|pc| {
                                                            let id = parsed_call_ids
                                                                .pop_front()
                                                                .unwrap_or_else(Uuid::new_v4);
                                                            (pc, id)
                                                        })
                                                        .collect()
                                                }
                                            } else {
                                                pcs.into_iter()
                                                    .map(|pc| {
                                                        let id = parsed_call_ids
                                                            .pop_front()
                                                            .unwrap_or_else(Uuid::new_v4);
                                                        (pc, id)
                                                    })
                                                    .collect()
                                            }
                                        } else {
                                            Vec::new()
                                        }
                                    }
                                };

                            let mut calls_needing_approval = Vec::new();
                            let mut auto_approved_call_results = Vec::new();
                            let overall_batch_id = if calls_to_process_with_ids.len() > 1 {
                                Some(Uuid::new_v4())
                            } else {
                                None
                            };

                            for (parsed_tool_call_ref, tool_call_id_ref) in
                                &calls_to_process_with_ids
                            {
                                let tool_call_id = *tool_call_id_ref;

                                if self.simplified_tool_events {
                                    self.emit_event(AgentEvent::ToolCallInitiated {
                                        tool_call_id,
                                        tool_name: parsed_tool_call_ref.name.clone(),
                                    })
                                    .await?;
                                }
                                self.emit_event(AgentEvent::ToolCallFormatted {
                                    tool_call_id,
                                    tool_name: parsed_tool_call_ref.name.clone(),
                                    parameters: parsed_tool_call_ref.parameters.clone(),
                                })
                                .await?;

                                let tool_arc =
                                    match self.tool_registry.get(&parsed_tool_call_ref.name) {
                                        Some(t) => t,
                                        None => {
                                            self.emit_event(AgentEvent::UnavailableToolAttempted {
                                                tool_name: parsed_tool_call_ref.name.clone(),
                                            })
                                            .await?;
                                            auto_approved_call_results.push(format!(
                                                "Error: Tool '{}' is not available.",
                                                parsed_tool_call_ref.name
                                            ));
                                            continue;
                                        }
                                    };
                                let tool_definition = tool_arc.definition();

                                let mut is_out_of_workspace = false;
                                let mut path_check_error: Option<ArkoAgentsError> = None;
                                let mut offending_param_name: Option<String> = None;

                                for param_def in &tool_definition.parameters {
                                    if param_def.is_path
                                        && let Some(path_str) = parsed_tool_call_ref
                                            .parameters
                                            .get(&param_def.name)
                                            .and_then(|v| v.as_str())
                                        && !path_str.is_empty()
                                    {
                                        match crate::utils::is_path_in_workspace(
                                            path_str,
                                            &self.task.current_working_directory,
                                            &self.task.current_working_directory,
                                        ) {
                                            Ok(true) => {}
                                            Ok(false) => {
                                                is_out_of_workspace = true;
                                                offending_param_name = Some(param_def.name.clone());
                                                break;
                                            }
                                            Err(e) => {
                                                path_check_error = Some(e);
                                                offending_param_name = Some(param_def.name.clone());
                                                break;
                                            }
                                        }
                                    }
                                }

                                if let Some(err) = path_check_error {
                                    let tool_error =
                                        crate::tool::ToolError::ExecutionFailed(format!(
                                            "Path validation failed for parameter '{}': {}",
                                            offending_param_name.unwrap_or_default(),
                                            err
                                        ));
                                    let error_message_for_llm = self
                                        .handle_tool_exec_error(
                                            &parsed_tool_call_ref.name,
                                            tool_error,
                                            Some(tool_call_id),
                                        )
                                        .await?;
                                    auto_approved_call_results.push(error_message_for_llm);
                                    continue;
                                }

                                let mut force_approval = false;
                                if is_out_of_workspace {
                                    let path_policy = self
                                        .task
                                        .agent_config
                                        .get_path_safety_policy(&parsed_tool_call_ref.name);
                                    match path_policy {
                                        PathSafetyPolicy::AutoReject => {
                                            let tool_error =
                                                crate::tool::ToolError::PathAccessRejected {
                                                    param_name: offending_param_name
                                                        .unwrap_or_default(),
                                                    workspace_path: self
                                                        .task
                                                        .current_working_directory
                                                        .to_string_lossy()
                                                        .to_string(),
                                                };
                                            let error_message_for_llm = self
                                                .handle_tool_exec_error(
                                                    &parsed_tool_call_ref.name,
                                                    tool_error,
                                                    Some(tool_call_id),
                                                )
                                                .await?;
                                            auto_approved_call_results.push(error_message_for_llm);
                                            continue;
                                        }
                                        PathSafetyPolicy::RequireApproval => {
                                            force_approval = true;
                                        }
                                        PathSafetyPolicy::AutoApprove => {}
                                    }
                                }

                                let normal_approval_needed = !self
                                    .task
                                    .agent_config
                                    .should_auto_approve_tool(&parsed_tool_call_ref.name);

                                if force_approval || normal_approval_needed {
                                    calls_needing_approval.push(PendingToolCallInfo {
                                        tool_call_id,
                                        tool_name: parsed_tool_call_ref.name.clone(),
                                        parameters: parsed_tool_call_ref.parameters.clone(),
                                        tool_description: tool_definition.description.clone(),
                                    });
                                } else {
                                    match self
                                        .execute_tool_and_handle_result(
                                            &parsed_tool_call_ref.name,
                                            parsed_tool_call_ref.parameters.clone(),
                                            tool_call_id,
                                            None,
                                            None,
                                        )
                                        .await
                                    {
                                        Ok(msg_part) => auto_approved_call_results.push(msg_part),
                                        Err(e) => log::error!(
                                            "Task {}: Error during auto-approved tool execution: {}",
                                            self.task.id,
                                            e
                                        ),
                                    }
                                    if !matches!(self.task.current_state, TaskState::InProgress) {
                                        break;
                                    }
                                }
                            }

                            if !calls_needing_approval.is_empty()
                                && matches!(self.task.current_state, TaskState::InProgress)
                            {
                                if calls_needing_approval.len() == 1 && overall_batch_id.is_none() {
                                    let single_pending = calls_needing_approval.remove(0);
                                    self.task.set_state(TaskState::AwaitingToolApproval {
                                        tool_call_id: single_pending.tool_call_id,
                                        tool_name: single_pending.tool_name.clone(),
                                        parameters: single_pending.parameters.clone(),
                                        tool_description: single_pending.tool_description.clone(),
                                    });
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    self.emit_event(AgentEvent::ToolCallAwaitingApproval {
                                        tool_call_id: single_pending.tool_call_id,
                                        tool_name: single_pending.tool_name,
                                        parameters: single_pending.parameters,
                                        tool_description: single_pending.tool_description,
                                    })
                                    .await?;
                                } else {
                                    let batch_id_for_state =
                                        overall_batch_id.unwrap_or_else(Uuid::new_v4);
                                    self.task.set_state(TaskState::AwaitingBatchToolApproval {
                                        batch_id: batch_id_for_state,
                                        pending_calls: calls_needing_approval,
                                    });
                                    self.emit_event(AgentEvent::TaskStateChanged(
                                        self.task.current_state.clone(),
                                    ))
                                    .await?;
                                    self.emit_event(AgentEvent::BatchToolCallAwaitingApproval {
                                        batch_id: batch_id_for_state,
                                        pending_calls: match &self.task.current_state {
                                            TaskState::AwaitingBatchToolApproval {
                                                pending_calls,
                                                ..
                                            } => pending_calls.clone(),
                                            _ => vec![],
                                        },
                                    })
                                    .await?;
                                }
                                if !auto_approved_call_results.is_empty() {
                                    next_user_message_str_accumulated =
                                        auto_approved_call_results.join("\n\n---\n\n");
                                }
                                if self.task.current_state.is_terminal() {
                                    break 'main_loop;
                                }
                                if self.task.current_state.is_awaiting_approval() {
                                    continue 'main_loop;
                                }
                            } else {
                                next_user_message_str_accumulated =
                                    auto_approved_call_results.join("\n\n---\n\n");
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to assemble tool calls for task {}: {}",
                            self.task.id,
                            e
                        );
                        self.handle_llm_error(
                            ai::Error::ApiError(format!("Assembly failed: {}", e)),
                            None,
                        )
                        .await?;
                        if self.task.current_state.is_terminal() {
                            break 'main_loop;
                        }
                        continue 'main_loop;
                    }
                }

                if final_generation_response.is_some()
                    && !had_stream_error
                    && !self.task.current_state.is_awaiting_approval()
                {
                    self.reset_llm_recovery();
                }

                if !next_user_message_str_accumulated.is_empty()
                    && !self.task.current_state.is_terminal()
                    && !self.task.current_state.is_awaiting_approval()
                {
                    let pb = PromptBuilder::new(
                        &self.task.agent_config,
                        vec![],
                        &self.parsing_config,
                        None,
                    );
                    let next_msg_num = self.task.user_message_count + 1;
                    let augmented_user_message = pb
                        .augment_user_message_with_env_details_and_reminder(
                            &self.task,
                            &next_user_message_str_accumulated,
                            next_msg_num,
                        )?;
                    self.add_text_tool_results_to_conversation(&augmented_user_message);
                    self.task.increment_user_message_count();
                } else if next_user_message_str_accumulated.is_empty()
                    && !self.task.current_state.is_terminal()
                    && !self.task.current_state.is_awaiting_approval()
                    && final_generation_response.is_some()
                {
                    let pb = PromptBuilder::new(
                        &self.task.agent_config,
                        vec![],
                        &self.parsing_config,
                        None,
                    );
                    let generic_retry_prompt = "Please continue with the task or use the 'complete_task' tool if finished. Ensure you use a tool.";
                    let next_msg_num = self.task.user_message_count + 1;
                    let augmented_generic_prompt = pb
                        .augment_user_message_with_env_details_and_reminder(
                            &self.task,
                            generic_retry_prompt,
                            next_msg_num,
                        )?;
                    self.task
                        .conversation
                        .add_user_message(&augmented_generic_prompt);
                    self.task.increment_user_message_count();
                }
            }

            if self.task.current_state.is_terminal() {
                break 'main_loop;
            }
        }

        self.flush_throttled_text().await?;
        self.flush_throttled_reasoning().await?;
        self.emit_event(AgentEvent::TaskStreamEnd)
            .await
            .map_err(|e| {
                log::error!(
                    "Task {}: Failed to send TaskStreamEnd event: {}",
                    self.task.id,
                    e
                );
                e
            })?;

        log::info!(
            "Agent loop finished for task ID: {}. Final state: {:?}",
            self.task.id,
            self.task.current_state
        );
        Ok(self.task)
    }

    /// Reports calls a stream announced (`ToolCallInitiated`) that will not
    /// run because the turn ended without them, so every announced call gets
    /// an outcome.
    async fn reject_announced_calls(
        &self,
        announced: &mut HashMap<usize, (Uuid, String)>,
        reason: &str,
    ) -> Result<(), ArkoAgentsError> {
        let mut calls: Vec<_> = announced.drain().collect();
        calls.sort_by_key(|(index, _)| *index);
        for (_, (tool_call_id, tool_name)) in calls {
            self.emit_event(AgentEvent::ToolRejected {
                tool_call_id,
                tool_name,
                reason: Some(reason.to_string()),
            })
            .await?;
        }
        Ok(())
    }

    async fn emit_event(&self, event: AgentEvent) -> Result<(), ArkoAgentsError> {
        if self.event_sender.send(event.clone()).await.is_err() {
            let err_msg = format!("Failed to send AgentEvent: {:?}", event);
            log::error!("{}", err_msg);
            return Err(ArkoAgentsError::ChannelSend(err_msg));
        }
        Ok(())
    }

    async fn execute_native_tool(
        &mut self,
        ai_tool_call: &ai::ToolCall,
        internal_id: Uuid,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        let tool_name = ai_tool_call.name.clone();
        let params_map = Self::tool_call_args_to_map(ai_tool_call);

        let tool_arc = self
            .tool_registry
            .get(&tool_name)
            .ok_or_else(|| ArkoAgentsError::ToolNotFound(tool_name.clone()))?;

        match self
            .before_tool_execution(internal_id, &tool_name, &params_map)
            .await?
        {
            ToolExecutionDecision::Continue => {}
            ToolExecutionDecision::Reject { message_to_agent } => {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: internal_id,
                    tool_name: tool_name.clone(),
                    reason: Some(message_to_agent.clone()),
                })
                .await?;
                return Ok(Some(ai::ToolResult::error_from_tool_call(
                    ai_tool_call,
                    &message_to_agent,
                )));
            }
            ToolExecutionDecision::FailTask { reason } => {
                self.emit_event(AgentEvent::ToolRejected {
                    tool_call_id: internal_id,
                    tool_name: tool_name.clone(),
                    reason: Some(reason.clone()),
                })
                .await?;
                self.task.set_state(TaskState::Failed {
                    reason: reason.clone(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                return Ok(Some(ai::ToolResult::error_from_tool_call(
                    ai_tool_call,
                    &reason,
                )));
            }
        }

        let task_context = self.task_context();

        let params_for_interceptor = if self.interceptor_config.is_some() {
            Some(params_map.clone())
        } else {
            None
        };

        match tool_arc
            .execute_with_output(&task_context, params_map.clone())
            .await
        {
            Ok(ToolOutput {
                text: result_str,
                images,
            }) => {
                self.after_tool_execution(
                    internal_id,
                    &tool_name,
                    &params_map,
                    ToolExecutionOutcome::Success(&result_str),
                )
                .await?;
                self.queue_tool_images(
                    &tool_name,
                    ai_tool_call.id.as_deref().unwrap_or("unidentified"),
                    images,
                );
                if tool_name == ASK_USER_TOOL_NAME {
                    self.task.set_state(TaskState::AwaitingUserResponse {
                        questions: result_str.clone(),
                    });
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                    self.emit_event(AgentEvent::UserQuestionsAsked {
                        questions: result_str.clone(),
                    })
                    .await?;
                    self.emit_event(AgentEvent::ToolResult {
                        tool_call_id: internal_id,
                        tool_name,
                        output: result_str.clone(),
                    })
                    .await?;
                    return Ok(Some(ai::ToolResult::from_tool_call(
                        ai_tool_call,
                        &result_str,
                    )));
                }

                let (effective_result, was_intercepted) = self
                    .apply_tool_result_interception(
                        &tool_name,
                        &result_str,
                        params_for_interceptor.as_ref().unwrap_or(&HashMap::new()),
                    )
                    .await;

                self.emit_tool_result_events(
                    internal_id,
                    &tool_name,
                    &result_str,
                    &effective_result,
                    was_intercepted,
                )
                .await?;

                if tool_name == COMPLETE_TASK_TOOL_NAME {
                    self.task.set_state(TaskState::Completed {
                        summary: result_str.clone(),
                    });
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                }

                Ok(Some(ai::ToolResult::from_tool_call(
                    ai_tool_call,
                    &effective_result,
                )))
            }
            Err(e) => {
                let error_str = format!("Tool '{}' execution failed: {}", tool_name, e);
                self.after_tool_execution(
                    internal_id,
                    &tool_name,
                    &params_map,
                    ToolExecutionOutcome::Error(&error_str),
                )
                .await?;
                let _ = self
                    .handle_tool_exec_error(&tool_name, e, Some(internal_id))
                    .await?;
                Ok(Some(ai::ToolResult::error_from_tool_call(
                    ai_tool_call,
                    &error_str,
                )))
            }
        }
    }

    async fn execute_native_tool_for_approval(
        &mut self,
        ai_call: &ai::ToolCall,
        internal_id: Uuid,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        custom_output: Option<String>,
        message_to_agent: Option<String>,
    ) -> Result<Option<ai::ToolResult>, ArkoAgentsError> {
        let params_for_interceptor = if self.interceptor_config.is_some() {
            Some(parameters.clone())
        } else {
            None
        };

        let mut output_str = if let Some(output) = custom_output {
            self.after_tool_execution(
                internal_id,
                tool_name,
                &parameters,
                ToolExecutionOutcome::CallerProvided(&output),
            )
            .await?;
            output
        } else {
            match self
                .before_tool_execution(internal_id, tool_name, &parameters)
                .await?
            {
                ToolExecutionDecision::Continue => {}
                ToolExecutionDecision::Reject { message_to_agent } => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: internal_id,
                        tool_name: tool_name.to_string(),
                        reason: Some(message_to_agent.clone()),
                    })
                    .await?;
                    return Ok(Some(ai::ToolResult::error_from_tool_call(
                        ai_call,
                        &message_to_agent,
                    )));
                }
                ToolExecutionDecision::FailTask { reason } => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id: internal_id,
                        tool_name: tool_name.to_string(),
                        reason: Some(reason.clone()),
                    })
                    .await?;
                    self.task.set_state(TaskState::Failed {
                        reason: reason.clone(),
                    });
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                    return Ok(Some(ai::ToolResult::error_from_tool_call(ai_call, &reason)));
                }
            }

            let tool_arc = self
                .tool_registry
                .get(tool_name)
                .ok_or_else(|| ArkoAgentsError::ToolNotFound(tool_name.to_string()))?;

            let task_context = self.task_context();

            match tool_arc
                .execute_with_output(&task_context, parameters.clone())
                .await
            {
                Ok(ToolOutput {
                    text: result_str,
                    images,
                }) => {
                    self.after_tool_execution(
                        internal_id,
                        tool_name,
                        &parameters,
                        ToolExecutionOutcome::Success(&result_str),
                    )
                    .await?;
                    self.queue_tool_images(
                        tool_name,
                        ai_call.id.as_deref().unwrap_or("unidentified"),
                        images,
                    );
                    result_str
                }
                Err(e) => {
                    let error_str = format!("Tool '{}' execution failed: {}", tool_name, e);
                    self.after_tool_execution(
                        internal_id,
                        tool_name,
                        &parameters,
                        ToolExecutionOutcome::Error(&error_str),
                    )
                    .await?;
                    let _ = self
                        .handle_tool_exec_error(tool_name, e, Some(internal_id))
                        .await?;
                    return Ok(Some(ai::ToolResult::error_from_tool_call(
                        ai_call, &error_str,
                    )));
                }
            }
        };

        if tool_name == COMPLETE_TASK_TOOL_NAME {
            self.task.set_state(TaskState::Completed {
                summary: output_str.clone(),
            });
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
        }
        if tool_name == ASK_USER_TOOL_NAME {
            self.task.set_state(TaskState::AwaitingUserResponse {
                questions: output_str.clone(),
            });
            self.emit_event(AgentEvent::TaskStateChanged(
                self.task.current_state.clone(),
            ))
            .await?;
            self.emit_event(AgentEvent::UserQuestionsAsked {
                questions: output_str.clone(),
            })
            .await?;
        }

        let (effective_output, was_intercepted) = self
            .apply_tool_result_interception(
                tool_name,
                &output_str,
                params_for_interceptor.as_ref().unwrap_or(&HashMap::new()),
            )
            .await;

        if let Some(msg) = message_to_agent {
            if !effective_output.is_empty() {
                output_str = effective_output;
                output_str.push_str("\n\n");
            } else {
                output_str = effective_output;
            }
            output_str.push_str("Message from user on approval:\n");
            output_str.push_str(&msg);

            self.emit_tool_result_events(internal_id, tool_name, &output_str, &output_str, false)
                .await?;
            return Ok(Some(ai::ToolResult::from_tool_call(ai_call, &output_str)));
        }

        self.emit_tool_result_events(
            internal_id,
            tool_name,
            &output_str,
            &effective_output,
            was_intercepted,
        )
        .await?;
        Ok(Some(ai::ToolResult::from_tool_call(
            ai_call,
            &effective_output,
        )))
    }

    fn append_error_recovery_user_message(&mut self, message: &str) -> Result<(), ArkoAgentsError> {
        let pb = PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
        let next_msg_num = self.task.user_message_count + 1;
        let augmented_message = pb.augment_user_message_with_env_details_and_reminder(
            &self.task,
            message,
            next_msg_num,
        )?;
        self.task.conversation.add_user_message(&augmented_message);
        self.task.increment_user_message_count();
        Ok(())
    }

    async fn finish_llm_recovery(
        &mut self,
        message: String,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let pause = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::PauseAndAwaitApproval
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if pause {
            self.task
                .set_state(TaskState::AwaitingErrorRecoveryApproval {
                    error_message: message.clone(),
                });
        } else {
            // Exhaustion must return control to the host rather than starting
            // an unbounded second retry cycle under RetryThenInformAgent.
            self.task.set_state(TaskState::Interrupted {
                reason: message.clone(),
            });
            self.stop_requested = true;
        }
        self.emit_event(AgentEvent::TaskStateChanged(
            self.task.current_state.clone(),
        ))
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: pause,
            tool_call_id,
        })
        .await
    }

    /// Reports a refusal and stops recovery: the same input is expected to be
    /// refused again, so it is never retried unchanged. A model's own refusal
    /// ([`ai::RefusalSource::keeps_turn`]) stays in history so the model can
    /// see what it declined, with the refusal explanation as its text when the
    /// turn has none. Its tool calls are not run, so they are not kept either
    /// (a call without a result would invalidate the history; OpenAI output
    /// items stored with the turn replay only the calls it still holds), and
    /// a turn with no text at all is not kept. Classifier and content-filter
    /// refusals are removed from history.
    async fn handle_refusal(
        &mut self,
        refusal: ai::Refusal,
        mut message: ai::Message,
    ) -> Result<(), ArkoAgentsError> {
        self.flush_throttled_reasoning().await?;
        self.flush_throttled_text().await?;
        if refusal.source.keeps_turn() {
            message.tool_calls.clear();
            let has_text = |message: &ai::Message| {
                message
                    .content
                    .iter()
                    .filter_map(ai::Content::as_text)
                    .any(|text| !text.trim().is_empty())
            };
            if !has_text(&message)
                && let Some(explanation) = refusal.explanation.as_deref()
            {
                message
                    .content
                    .retain(|content| content.as_text().is_none());
                message
                    .content
                    .insert(0, ai::Content::Text(explanation.to_string()));
            }
            if has_text(&message) {
                self.task.conversation.messages.push(message);
            }
        }
        self.emit_event(AgentEvent::RefusalReceived(refusal.clone()))
            .await?;
        self.handle_llm_error(ai::Error::Refusal(Box::new(refusal)), None)
            .await
    }

    /// How recovery from `error` waits.
    fn recovery_wait(&self, error: &ai::Error) -> RecoveryWait {
        if error.is_usage_limit() {
            return RecoveryWait::UsageLimit;
        }
        let retrying = matches!(
            self.task.agent_config.llm_error_policy,
            LLMErrorPolicy::RetryThenInformAgent { .. }
                | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { .. }
        );
        if !retrying {
            return RecoveryWait::Ordinary;
        }
        let config = &self.task.agent_config.llm_retry_config;
        if error.is_capacity_error()
            && let Some(capacity) = &config.capacity_retry
        {
            return RecoveryWait::Capacity(capacity.clone());
        }
        if let Some(rate_limit) = &config.rate_limit_retry
            && (error.is_rate_limit() || error.is_capacity_error())
        {
            let too_long = error
                .retry_after()
                .is_some_and(|hint| hint > Duration::from_millis(rate_limit.max_wait_ms));
            if !too_long {
                return RecoveryWait::RateLimit(rate_limit.clone());
            }
            // A rate limit that resets this far off is a usage limit in all
            // but name.
            if error.is_rate_limit() {
                return RecoveryWait::UsageLimit;
            }
        }
        RecoveryWait::Ordinary
    }

    /// When a usage limit is waited out under
    /// [`UsageLimitPolicy::WaitUntilReset`], the delay before the retry.
    fn usage_limit_wait(&self, error: &ai::Error) -> Option<Duration> {
        let UsageLimitPolicy::WaitUntilReset { max_wait_ms } =
            self.task.agent_config.llm_retry_config.usage_limit_policy
        else {
            return None;
        };
        let reset_in = error.retry_after()?;
        if reset_in > Duration::from_millis(max_wait_ms) {
            return None;
        }
        // A little past the reset, spread so that agents sharing a plan do
        // not all retry at once.
        let jitter = if self.task.agent_config.llm_retry_config.jitter {
            ai::retry::random_sample() % (USAGE_LIMIT_RESET_JITTER.as_millis() as u64 + 1)
        } else {
            0
        };
        Some(
            reset_in
                .saturating_add(USAGE_LIMIT_RESET_MARGIN)
                .saturating_add(Duration::from_millis(jitter)),
        )
    }

    /// Whether recovering from `error` resends the same request without
    /// adding to the conversation. Output a failed stream delivered is then
    /// dropped: kept, it would end the retried request on an assistant turn
    /// (a prefill, which Claude 4.6 and later reject).
    fn retries_unchanged(&self, error: &ai::Error) -> bool {
        match self.recovery_wait(error) {
            RecoveryWait::Capacity(_) | RecoveryWait::RateLimit(_) => true,
            RecoveryWait::UsageLimit => self.usage_limit_wait(error).is_some(),
            RecoveryWait::Ordinary => false,
        }
    }

    /// Records the output a failed stream delivered before it failed with
    /// `error`, unless the retry resends the request unchanged; see
    /// [`retries_unchanged`](Self::retries_unchanged).
    fn keep_partial_output(&mut self, text: &str, error: &ai::Error) {
        if !text.is_empty() && !self.retries_unchanged(error) {
            self.task.conversation.add_assistant_message(text);
        }
    }

    /// Before a model request: a conversation that ends on an assistant
    /// turn (output kept from an interrupted stream, or a kept refusal, in a
    /// task resumed without a new message) gets a user message asking to
    /// continue, since several providers reject a request that ends on an
    /// assistant turn (Claude 4.6 and later reject prefill).
    fn ensure_user_turn_before_request(&mut self) -> Result<(), ArkoAgentsError> {
        let ends_on_assistant_text = self.task.conversation.messages.last().is_some_and(|message| {
            message.role == ai::Role::Assistant && message.tool_calls.is_empty()
        });
        if ends_on_assistant_text {
            self.append_error_recovery_user_message(
                "Continue the task from where your previous response ended.",
            )?;
        }
        Ok(())
    }

    async fn handle_llm_error(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let mut config = self.task.agent_config.llm_retry_config.clone();
        let policy = self.task.agent_config.llm_error_policy.clone();
        let wait = self.recovery_wait(&error);
        if let RecoveryWait::UsageLimit = wait {
            return self.handle_usage_limit(error, tool_call_id).await;
        }
        let indefinite = !matches!(wait, RecoveryWait::Ordinary);
        let (attempt, started) = if indefinite {
            // These waits neither consume nor reset ordinary recovery: its
            // elapsed clock stops while they last, and the requests they
            // make are not counted as its attempts.
            self.pause_ordinary_recovery_clock();
            self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
            let started = *self
                .capacity_recovery_started
                .get_or_insert_with(tokio::time::Instant::now);
            self.capacity_retry_attempts = self.capacity_retry_attempts.saturating_add(1);
            (self.capacity_retry_attempts, started)
        } else {
            self.resume_ordinary_recovery_clock();
            (
                self.ordinary_attempts(),
                *self
                    .recovery_started
                    .get_or_insert_with(tokio::time::Instant::now),
            )
        };
        match &wait {
            RecoveryWait::Capacity(capacity) => {
                config.initial_delay_ms = capacity.initial_delay_ms;
                config.max_delay_ms = capacity.max_delay_ms;
            }
            RecoveryWait::RateLimit(rate_limit) => {
                config.initial_delay_ms = rate_limit.initial_delay_ms;
                config.max_delay_ms = rate_limit.max_delay_ms;
            }
            RecoveryWait::Ordinary | RecoveryWait::UsageLimit => {}
        }
        let elapsed = started.elapsed();
        let max_attempts = match policy {
            LLMErrorPolicy::RetryThenInformAgent { max_attempts }
            | LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts } => {
                max_attempts.max(1)
            }
            LLMErrorPolicy::InformAgentAndContinue => u32::MAX,
            LLMErrorPolicy::PauseAndAwaitApproval => 1,
        };
        let message = match &wait {
            RecoveryWait::Capacity(_) => format!(
                "LLM capacity error (attempt {attempt}; waiting until service recovers): {error}"
            ),
            RecoveryWait::RateLimit(_) => {
                let kind = if error.is_capacity_error() { "capacity error" } else { "rate limit" };
                format!("LLM {kind} (attempt {attempt}; retrying when the provider allows): {error}")
            }
            RecoveryWait::Ordinary | RecoveryWait::UsageLimit => {
                format!("LLM error (attempt {attempt}/{max_attempts}): {error}")
            }
        };
        let stop_reason = if !error.is_retryable() {
            Some("Failure requires a request, credential, or configuration change")
        } else if !indefinite && attempt >= max_attempts {
            Some("Retry attempt limit exhausted")
        } else {
            None
        };
        if let Some(reason) = stop_reason {
            return self
                .finish_llm_recovery(format!("{reason}. {message}"), tool_call_id)
                .await;
        }
        // A rate limit's own delay is followed as given (plus jitter) rather
        // than grown by backoff; backoff applies when there is none.
        let backoff_attempt = match wait {
            RecoveryWait::RateLimit(_) if error.retry_after().is_some() => 1,
            _ => attempt,
        };
        let delay = super::retry::retry_delay(
            &config,
            backoff_attempt,
            error.retry_after(),
            ai::retry::random_sample(),
        );
        if !indefinite
            && elapsed.saturating_add(delay) >= Duration::from_millis(config.max_elapsed_ms) {
            return self
                .finish_llm_recovery(
                    format!("Next retry cannot fit the elapsed recovery budget. {message}"),
                    tool_call_id,
                )
                .await;
        }
        let Some(retry_at) = tokio::time::Instant::now().checked_add(delay) else {
            return self.finish_llm_recovery(
                format!("Server retry delay cannot be represented safely. {message}"), tool_call_id,
            ).await;
        };
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: message,
            requires_approval_to_retry: false,
            tool_call_id,
        })
        .await?;
        self.emit_event(AgentEvent::RetryScheduled {
            attempt,
            delay_ms: delay.as_millis() as u64,
            elapsed_ms: elapsed.as_millis() as u64,
        })
        .await?;
        if !indefinite {
            self.append_error_recovery_user_message(&format!("An error occurred during the previous LLM call (attempt {attempt}): {error}. Please retry or adjust your approach."))?;
        }
        self.wait_before_retry(retry_at, tool_call_id).await
    }

    /// Handles a plan usage limit under
    /// [`LLMRetryConfig::usage_limit_policy`](crate::agent_config::LLMRetryConfig::usage_limit_policy):
    /// waits for the reset and retries unchanged (opt-in), or stops
    /// retrying.
    async fn handle_usage_limit(
        &mut self,
        error: ai::Error,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let now_ms = unix_millis_now();
        let reset_in = error.retry_after();
        let resets_at_unix_ms =
            reset_in.map(|delay| now_ms.saturating_add(duration_millis(delay)));
        let when = match reset_in {
            Some(delay) => format!("resets in about {}", describe_duration(delay)),
            None => "reset time not reported".to_string(),
        };
        let retry_at = self
            .usage_limit_wait(&error)
            .and_then(|delay| tokio::time::Instant::now().checked_add(delay).map(|at| (delay, at)));
        let Some((delay, retry_at)) = retry_at else {
            self.emit_event(AgentEvent::UsageLimitReached {
                resets_at_unix_ms,
                waiting_until_unix_ms: None,
                error_message: error.to_string(),
            })
            .await?;
            return self
                .finish_llm_recovery(format!("Usage limit reached ({when}). {error}"), tool_call_id)
                .await;
        };
        // Like the other unchanged retries, the wait doesn't count against
        // ordinary recovery.
        self.pause_ordinary_recovery_clock();
        self.indefinite_interactions = self.indefinite_interactions.saturating_add(1);
        self.emit_event(AgentEvent::UsageLimitReached {
            resets_at_unix_ms,
            waiting_until_unix_ms: Some(now_ms.saturating_add(duration_millis(delay))),
            error_message: error.to_string(),
        })
        .await?;
        self.emit_event(AgentEvent::ErrorEncountered {
            error_message: format!(
                "Usage limit reached ({when}); waiting {} before retrying. {error}",
                describe_duration(delay)
            ),
            requires_approval_to_retry: false,
            tool_call_id,
        })
        .await?;
        self.wait_before_retry(retry_at, tool_call_id).await
    }

    /// Waits until `retry_at` before a retry, handling commands; the task is
    /// then in progress again. An ordinary recovery deadline that passes
    /// first ends recovery.
    async fn wait_before_retry(
        &mut self,
        retry_at: tokio::time::Instant,
        tool_call_id: Option<Uuid>,
    ) -> Result<(), ArkoAgentsError> {
        let deadline = self.recovery_deadline();
        let sleep = tokio::time::sleep_until(retry_at);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                _ = wait_for_recovery_deadline(deadline) => {
                    return self.finish_llm_recovery("Elapsed model recovery budget exhausted during backoff.".into(), tool_call_id).await;
                }
                command = self.command_receiver.recv() => {
                    if self.handle_recovery_command(command).await? { return Ok(()); }
                }
                _ = &mut sleep => {
                    self.task.set_state(TaskState::InProgress);
                    return Ok(());
                }
            }
        }
    }

    /// Before a model request: applies the opt-in plan usage threshold
    /// ([`MonitoringConfig::plan_usage`](crate::agent_config::MonitoringConfig::plan_usage))
    /// to the most recent report. Returns whether to make the request.
    async fn apply_plan_usage_threshold(&mut self) -> Result<bool, ArkoAgentsError> {
        let Some(threshold) = self.task.agent_config.monitoring.plan_usage.clone() else {
            return Ok(true);
        };
        let Some(usage) = self
            .task
            .model_call_usage_history
            .last()
            .and_then(|call| call.plan_usage.clone())
        else {
            return Ok(true);
        };
        let Some((window, reached)) = usage
            .most_used_window()
            .filter(|(_, window)| window.used_percent >= threshold.percent)
        else {
            return Ok(true);
        };
        let key = (window.to_string(), reached.resets_at);
        let first = self.plan_usage_threshold_reported.as_ref() != Some(&key);
        if first {
            self.plan_usage_threshold_reported = Some(key);
            self.emit_event(AgentEvent::PlanUsageThresholdReached {
                window: window.to_string(),
                used_percent: reached.used_percent,
                threshold_percent: threshold.percent,
                resets_at_unix_ms: reached.resets_at.map(|seconds| seconds.saturating_mul(1000)),
                action: threshold.action.clone(),
            })
            .await?;
        }
        match threshold.action {
            PlanUsageAction::Pause if first => {
                let message = format!(
                    "Plan usage reached {:.1}% of the {window} usage-limit window (threshold {:.1}%).",
                    reached.used_percent, threshold.percent
                );
                self.finish_llm_recovery(message, None).await?;
                Ok(false)
            }
            PlanUsageAction::Pause => Ok(true),
            PlanUsageAction::SlowDown { delay_ms } => {
                let Some(until) =
                    tokio::time::Instant::now().checked_add(Duration::from_millis(delay_ms))
                else {
                    return Ok(true);
                };
                let sleep = tokio::time::sleep_until(until);
                tokio::pin!(sleep);
                loop {
                    tokio::select! {
                        biased;
                        command = self.command_receiver.recv() => {
                            if self.handle_recovery_command(command).await? { return Ok(false); }
                        }
                        _ = &mut sleep => return Ok(true),
                    }
                }
            }
        }
    }

    async fn handle_tool_exec_error(
        &mut self,
        tool_name: &str,
        error: crate::tool::ToolError,
        tool_call_id: Option<Uuid>,
    ) -> Result<String, ArkoAgentsError> {
        let error_message_for_event = format!("Tool '{}' execution failed: {}", tool_name, error);
        log::warn!("Task {}: {}", self.task.id, error_message_for_event);

        let prompt_builder_for_error_msg =
            PromptBuilder::new(&self.task.agent_config, vec![], &self.parsing_config, None);
        let error_message_for_llm = prompt_builder_for_error_msg
            .format_tool_execution_error_as_user_message(tool_name, &error.to_string());

        let policy = &self
            .task
            .agent_config
            .error_recovery_config
            .on_tool_execution_error;
        match policy {
            ToolErrorPolicy::InformAgentAndContinue => {
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event.clone(),
                    requires_approval_to_retry: false,
                    tool_call_id,
                })
                .await?;
                self.task.set_state(TaskState::InProgress);
                Ok(error_message_for_llm)
            }
            ToolErrorPolicy::PauseAndAwaitApproval => {
                self.task
                    .set_state(TaskState::AwaitingErrorRecoveryApproval {
                        error_message: error_message_for_event.clone(),
                    });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event.clone(),
                    requires_approval_to_retry: true,
                    tool_call_id,
                })
                .await?;
                Ok(String::new())
            }
            ToolErrorPolicy::FailTask => {
                self.task.set_state(TaskState::Failed {
                    reason: error_message_for_event.clone(),
                });
                self.emit_event(AgentEvent::TaskStateChanged(
                    self.task.current_state.clone(),
                ))
                .await?;
                self.emit_event(AgentEvent::ErrorEncountered {
                    error_message: error_message_for_event,
                    requires_approval_to_retry: false,
                    tool_call_id,
                })
                .await?;
                Ok(String::new())
            }
        }
    }

    async fn execute_tool_and_handle_result(
        &mut self,
        tool_name: &str,
        parameters: HashMap<String, Value>,
        tool_call_id: Uuid,
        custom_tool_output: Option<String>,
        message_to_agent_on_approval: Option<String>,
    ) -> Result<String, ArkoAgentsError> {
        let tool_arc = self
            .tool_registry
            .get(tool_name)
            .ok_or_else(|| ArkoAgentsError::ToolNotFound(tool_name.to_string()))?;

        let result_str_or_error_msg_for_llm: String;
        let tool_definition_for_pb = tool_arc.definition();
        let pb_for_tool_results = PromptBuilder::new(
            &self.task.agent_config,
            vec![&tool_definition_for_pb],
            &self.parsing_config,
            None,
        );

        if let Some(custom_output) = custom_tool_output {
            self.after_tool_execution(
                tool_call_id,
                tool_name,
                &parameters,
                ToolExecutionOutcome::CallerProvided(&custom_output),
            )
            .await?;
            let (effective_output, was_intercepted) = self
                .apply_tool_result_interception(tool_name, &custom_output, &parameters)
                .await;

            self.emit_tool_result_events(
                tool_call_id,
                tool_name,
                &custom_output,
                &effective_output,
                was_intercepted,
            )
            .await?;
            result_str_or_error_msg_for_llm = pb_for_tool_results
                .format_tool_result_as_user_message(
                    tool_name,
                    &effective_output,
                    message_to_agent_on_approval.as_deref(),
                );
        } else {
            let params_for_interceptor = if self.interceptor_config.is_some() {
                Some(parameters.clone())
            } else {
                None
            };

            match self
                .before_tool_execution(tool_call_id, tool_name, &parameters)
                .await?
            {
                ToolExecutionDecision::Continue => {}
                ToolExecutionDecision::Reject { message_to_agent } => {
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id,
                        tool_name: tool_name.to_string(),
                        reason: Some(message_to_agent.clone()),
                    })
                    .await?;
                    return Ok(
                        pb_for_tool_results.format_tool_execution_error_as_user_message(
                            tool_name,
                            &message_to_agent,
                        ),
                    );
                }
                ToolExecutionDecision::FailTask { reason } => {
                    let formatted_error = pb_for_tool_results
                        .format_tool_execution_error_as_user_message(tool_name, &reason);
                    self.emit_event(AgentEvent::ToolRejected {
                        tool_call_id,
                        tool_name: tool_name.to_string(),
                        reason: Some(reason.clone()),
                    })
                    .await?;
                    self.task.set_state(TaskState::Failed {
                        reason: reason.clone(),
                    });
                    self.emit_event(AgentEvent::TaskStateChanged(
                        self.task.current_state.clone(),
                    ))
                    .await?;
                    return Ok(formatted_error);
                }
            }

            let task_context = self.task_context();
            match tool_arc
                .execute_with_output(&task_context, parameters.clone())
                .await
            {
                Ok(ToolOutput {
                    text: result_str,
                    images,
                }) => {
                    self.after_tool_execution(
                        tool_call_id,
                        tool_name,
                        &parameters,
                        ToolExecutionOutcome::Success(&result_str),
                    )
                    .await?;
                    if tool_name == ASK_USER_TOOL_NAME {
                        self.task.set_state(TaskState::AwaitingUserResponse {
                            questions: result_str.clone(),
                        });
                        self.emit_event(AgentEvent::TaskStateChanged(
                            self.task.current_state.clone(),
                        ))
                        .await?;
                        self.emit_event(AgentEvent::UserQuestionsAsked {
                            questions: result_str.clone(),
                        })
                        .await?;
                        self.emit_event(AgentEvent::ToolResult {
                            tool_call_id,
                            tool_name: tool_name.to_string(),
                            output: result_str,
                        })
                        .await?;
                        self.queue_tool_images(tool_name, tool_call_id, images);
                        return Ok(String::new());
                    }

                    let (effective_result, was_intercepted) = self
                        .apply_tool_result_interception(
                            tool_name,
                            &result_str,
                            params_for_interceptor.as_ref().unwrap_or(&HashMap::new()),
                        )
                        .await;

                    self.emit_tool_result_events(
                        tool_call_id,
                        tool_name,
                        &result_str,
                        &effective_result,
                        was_intercepted,
                    )
                    .await?;
                    result_str_or_error_msg_for_llm = pb_for_tool_results
                        .format_tool_result_as_user_message(
                            tool_name,
                            &effective_result,
                            message_to_agent_on_approval.as_deref(),
                        );
                    self.queue_tool_images(tool_name, tool_call_id, images);

                    if tool_name == COMPLETE_TASK_TOOL_NAME {
                        self.task.set_state(TaskState::Completed {
                            summary: result_str,
                        });
                    }
                }
                Err(e) => {
                    let error_str = format!("Tool '{}' execution failed: {}", tool_name, e);
                    self.after_tool_execution(
                        tool_call_id,
                        tool_name,
                        &parameters,
                        ToolExecutionOutcome::Error(&error_str),
                    )
                    .await?;
                    result_str_or_error_msg_for_llm = self
                        .handle_tool_exec_error(tool_name, e, Some(tool_call_id))
                        .await?;
                }
            }
        }
        Ok(result_str_or_error_msg_for_llm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentLLMConfig, AgentToolConfig};
    use arko_ai::{GenerationConfig, Model};

    pub(super) fn test_config() -> AgentConfig {
        AgentConfig::builder(
            "batch-test".to_string(),
            "Batch Test".to_string(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_luna(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test".to_string())
        .instructions("test".to_string())
        .tool_config(AgentToolConfig {
            allow_batched_tool_calls: true,
            ..AgentToolConfig::default()
        })
        .build()
        .unwrap()
    }

    fn call(name: &str) -> ai::ToolCall {
        ai::ToolCall {
            id: Some(format!("call-{name}")),
            name: name.to_string(),
            arguments: serde_json::json!({}),
            thought_signature: None,
            provider_id: None,
            invalid_arguments: None,
        }
    }

    #[test]
    fn native_batches_allow_mixed_safe_reads_but_reject_unsafe_tools() {
        let config = test_config();
        let registry = ToolRegistry::new();
        assert!(
            native_batch_rejection_reason(
                &config,
                &registry,
                &[call("read_file"), call("list_files")],
            )
            .is_none()
        );

        let rejection = native_batch_rejection_reason(
            &config,
            &registry,
            &[call("read_file"), call("complete_task")],
        )
        .unwrap();
        assert!(rejection.contains("complete_task"));
        assert!(!rejection.contains("read_file, complete_task"));
    }
}

#[cfg(test)]
#[path = "retry_tests.rs"]
mod retry_tests;

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod recovery_tests;

#[cfg(test)]
#[path = "long_run_tests.rs"]
mod long_run_tests;

#[cfg(test)]
#[path = "image_tests.rs"]
mod image_tests;

#[cfg(test)]
#[path = "context_tests.rs"]
mod context_tests;
