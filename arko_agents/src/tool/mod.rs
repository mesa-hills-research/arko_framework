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
#[derive(Debug)]
pub struct TaskContext<'a> {
    pub task_id: uuid::Uuid,
    pub agent_name: &'a str,
    pub initial_task_prompt: &'a str,
    pub current_working_directory: &'a Path,
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
