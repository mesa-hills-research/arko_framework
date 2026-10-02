use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

pub mod builtins;
pub mod command_builder;
pub mod definition;
pub mod error;
pub mod interceptor;
pub mod middleware;
pub mod registry;
pub mod types;

pub use definition::{ToolDefinition, ToolParameterDefinition, ToolParameterType};
pub use error::ToolError;
pub use interceptor::{
    InterceptedResult, InterceptionFailurePolicy, InterceptorConfig, ToolResultInterceptor,
};
pub use middleware::{
    ToolCallContext, ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware,
    ToolExecutionOutcome,
};
pub use registry::ToolRegistry;
pub use types::ToolTypeDetails;

/// A real image input returned by a tool, with a human-readable source label.
/// Image bytes belong in saved task state, never in text tool-result events.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolImage {
    pub description: String,
    pub image: arko_ai::Image,
}

impl Debug for ToolImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.image {
            arko_ai::Image::Base64 { mime_type, .. } => mime_type.as_str(),
            arko_ai::Image::Url(_) => "URL",
        };
        f.debug_struct("ToolImage")
            .field("description", &self.description)
            .field("kind", &kind)
            .finish_non_exhaustive()
    }
}

/// Structured output is opt-in; existing string-returning tools need no changes.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub text: String,
    pub images: Vec<ToolImage>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }
}

/// Context provided to a tool during its execution.
///
/// This allows tools to access relevant information from the current task's environment,
/// such as the current working directory.
///
/// A host that builds one itself (to call a tool outside a runtime, or in a
/// test) uses [`TaskContext::new`], which leaves the call fields unset and
/// the [stop signal](Self::stop) one that never fires; the fields stay
/// public to set afterwards. The struct is `#[non_exhaustive]`, so fields
/// can be added without breaking callers: outside this crate it can't be
/// built with a struct literal.
#[derive(Debug)]
#[non_exhaustive]
pub struct TaskContext<'a> {
    pub task_id: uuid::Uuid,
    pub agent_name: &'a str,
    pub initial_task_prompt: &'a str,
    pub current_working_directory: &'a Path,
    /// The runtime's id of the call being executed, as its events and the
    /// execution middleware see it (`None` outside a call, such as for a
    /// result interceptor run by a host). A call recovered after its host
    /// stopped keeps its id; see [`Tool::recovery_mode`].
    pub tool_call_id: Option<uuid::Uuid>,
    /// The call is being run again because the task stopped while it ran
    /// (its host crashed, say): its tool declared
    /// [`ToolRecovery::Rerun`] or [`ToolRecovery::Resume`]. A
    /// [`Resume`](ToolRecovery::Resume) tool finds the work the first run
    /// started, by [`tool_call_id`](Self::tool_call_id), and continues it.
    pub recovering: bool,
    /// Fires when the task is being stopped while the call runs: the
    /// runtime received `CancelTask` or `SuspendTask`, or the host fired the
    /// signal it gave the runtime (a workflow's graceful stop). A long call
    /// can watch it to save its work and return early; see
    /// [`StopSignal`](crate::stop::StopSignal) and
    /// [`StopAwareCommand`](crate::stop::StopAwareCommand) for a child
    /// process. An [interruptible](Tool::interruptible) call is dropped as
    /// soon as `CancelTask` or `SuspendTask` arrives whatever it does, so
    /// it sees the signal only for a host's graceful stop.
    pub stop: crate::stop::StopSignal,
}

impl<'a> TaskContext<'a> {
    /// The context of a task outside a call: no call ID, not recovering, and
    /// a stop signal that never fires. Set the public fields to change
    /// them, or [`with_stop`](Self::with_stop).
    pub fn new(
        task_id: uuid::Uuid,
        agent_name: &'a str,
        initial_task_prompt: &'a str,
        current_working_directory: &'a Path,
    ) -> Self {
        Self {
            task_id,
            agent_name,
            initial_task_prompt,
            current_working_directory,
            tool_call_id: None,
            recovering: false,
            stop: crate::stop::StopSignal::never(),
        }
    }

    /// This context with `stop` as its [stop signal](Self::stop).
    pub fn with_stop(mut self, stop: crate::stop::StopSignal) -> Self {
        self.stop = stop;
        self
    }
}

/// How a tool call that was running when its task stopped (its host
/// crashed, say) is settled when the task runs again; see
/// [`Tool::recovery_mode`] and
/// [`Task::running_tool_calls`](crate::task::Task::running_tool_calls).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRecovery {
    /// The call is not run again: its result tells the model the task
    /// restarted while the call ran, that it may have run partly or fully,
    /// and to check its effects before calling it again. For a tool with
    /// effects outside the task (writing files, running commands).
    #[default]
    InDoubt,
    /// The call runs again, as it was made: for a tool without side
    /// effects, which returns the same result (reading a file, listing a
    /// directory, loading a skill).
    Rerun,
    /// The call runs again with
    /// [`TaskContext::recovering`] set and its original
    /// [`TaskContext::tool_call_id`]: the tool reattaches to the work its
    /// first run started (a workflow's `run_workflow_agent` recovers the
    /// agent run it started) and returns its result, or else runs the call.
    Resume,
}

/// A trait for tools that an agent can use.
///
/// Tools must be `Send + Sync` as they can be executed in concurrent environments.
#[async_trait]
pub trait Tool: Debug + Send + Sync {
    /// Returns the definition of the tool.
    fn definition(&self) -> ToolDefinition;

    /// Stateful, exact-result protocols can opt out of result transformation.
    fn allow_result_interception(&self) -> bool {
        true
    }

    /// Whether the agent's host-side time limit applies to this tool's calls
    /// ([`AgentToolConfig::tool_timeout_seconds`](crate::agent_config::AgentToolConfig::tool_timeout_seconds)).
    /// Tools with their own wait and job handling return `false`, as the
    /// terminal tools do. A per-tool setting applies either way.
    fn uses_host_timeout(&self) -> bool {
        true
    }

    /// Whether `CancelTask` and `SuspendTask` stop a running call of this
    /// tool: the runtime drops the call (killing a command it started), and
    /// its result says it was stopped. Tools that wind their own work down
    /// when the task stops return `false`, as a workflow's
    /// `run_workflow_agent` does (it stops the agent it runs within the
    /// workflow's grace period); the stop then takes effect once the call
    /// returns.
    fn interruptible(&self) -> bool {
        true
    }

    /// How a call of this tool that was running when its task stopped (its
    /// host crashed while the call ran) is settled when the task runs
    /// again: by default [`ToolRecovery::InDoubt`], answered without
    /// running it again. Read-only tools return [`ToolRecovery::Rerun`].
    fn recovery_mode(&self) -> ToolRecovery {
        ToolRecovery::InDoubt
    }

    /// Discard delivery baselines when a task starts or resumes in a runtime.
    fn reset_context(&self, _task_id: uuid::Uuid) {}

    /// Observe messages committed to the conversation before its next model call.
    /// This is synchronous bookkeeping, not a place for I/O or model calls.
    fn observe_context(&self, _task_id: uuid::Uuid, _message: &arko_ai::Message) {}

    /// Bind background tools to this task's context channel. A weak sender must
    /// not keep a stopped runtime alive. Hosts can also bind custom runners.
    fn bind_runtime(
        &self,
        _task_id: uuid::Uuid,
        _sender: tokio::sync::mpsc::WeakSender<crate::runtime::command::UserCommand>,
    ) {
    }

    /// Executes the tool with the given parameters and task context.
    ///
    /// # Arguments
    /// * `context`: Contextual information about the current task, like CWD.
    /// * `params`: A map of parameter names to their `serde_json::Value` representations.
    ///
    /// # Returns
    /// A `Result` containing a `String` output of the tool on success, or a `ToolError` on failure.
    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError>;

    /// Execute with optional image attachments. The runtime preserves images
    /// independently of text interception and sends them as image content blocks.
    async fn execute_with_output(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        self.execute(context, params).await.map(ToolOutput::text)
    }

    /// Helper method to get the tool's name from its definition.
    fn name(&self) -> String {
        self.definition().name
    }
}

/// A tool defined by an end user via a command template.
#[derive(Debug, Clone)]
pub(crate) struct EndUserTool {
    definition: ToolDefinition,
}

impl EndUserTool {
    /// Creates a new EndUserTool from a tool definition.
    pub fn new(definition: ToolDefinition) -> Result<Self, ToolError> {
        if !matches!(
            definition.tool_type_details,
            ToolTypeDetails::EndUserDefined { .. }
        ) {
            return Err(ToolError::Configuration {
                tool_name: definition.name.clone(),
                message: "ToolDefinition is not for an EndUserDefined tool.".to_string(),
            });
        }
        Ok(Self { definition })
    }
}

#[async_trait]
impl Tool for EndUserTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let command_template_str = match &self.definition.tool_type_details {
            ToolTypeDetails::EndUserDefined { command_template } => command_template,
            _ => return Err(ToolError::ExecutionFailed(
                "EndUserTool executed with non-EndUserDefined details. This is an internal error."
                    .to_string(),
            )),
        };

        command_builder::execute_command_from_template(
            &self.definition.name,
            command_template_str,
            &self.definition.parameters,
            &params,
            context.current_working_directory,
        )
        .await
    }
}
