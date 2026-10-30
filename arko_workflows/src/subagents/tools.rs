//! The subagent tools, `start_agent`, `message_agent`, `wait_for_agents`,
//! `list_agents`, `stop_agent` and `link_agents`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolRecovery, ToolTypeDetails,
};

use super::board::split_ids;
use crate::context::SharedContext;
use crate::messaging::wait::WaitForMessageTool;

pub const START_AGENT_TOOL_NAME: &str = "start_agent";
pub const MESSAGE_AGENT_TOOL_NAME: &str = "message_agent";
pub const WAIT_FOR_AGENTS_TOOL_NAME: &str = "wait_for_agents";
pub const LIST_AGENTS_TOOL_NAME: &str = "list_agents";
pub const STOP_AGENT_TOOL_NAME: &str = "stop_agent";
pub const LINK_AGENTS_TOOL_NAME: &str = "link_agents";

/// Every subagent tool's name.
pub const SUBAGENT_TOOL_NAMES: &[&str] = &[
    START_AGENT_TOOL_NAME,
    MESSAGE_AGENT_TOOL_NAME,
    WAIT_FOR_AGENTS_TOOL_NAME,
    LIST_AGENTS_TOOL_NAME,
    STOP_AGENT_TOOL_NAME,
    LINK_AGENTS_TOOL_NAME,
];

/// `start_agent`'s own parameters. A host parameter may not take one of
/// these names.
pub(crate) const START_PARAMETERS: &[&str] =
    &["agent", "task", "workspace", "isolated", "peers", "group"];

fn param(
    name: &str,
    description: impl Into<String>,
    param_type: ToolParameterType,
    required: bool,
) -> ToolParameterDefinition {
    ToolParameterDefinition {
        name: name.to_string(),
        description: description.into(),
        param_type,
        required,
        is_path: false,
        flag: None,
        default_value: None,
    }
}

fn definition(
    name: &str,
    description: &str,
    parameters: Vec<ToolParameterDefinition>,
    parallel_safe: bool,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        output_based_example: None,
        parameters,
        parallel_safe,
        tool_type_details: ToolTypeDetails::DeveloperDefined,
        read_only: false,
        raw_json_schema_override: None,
    }
}

/// A text parameter, trimmed. `None` when absent or empty.
fn text(params: &HashMap<String, Value>, name: &str) -> Option<String> {
    match params.get(name)? {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn required(params: &HashMap<String, Value>, name: &str) -> Result<String, ToolError> {
    text(params, name).ok_or_else(|| ToolError::MissingParameter {
        param_name: name.to_string(),
    })
}

/// A true-or-false parameter, read leniently (`"true"` counts). Absent is
/// false.
fn flag(params: &HashMap<String, Value>, name: &str) -> Result<bool, ToolError> {
    match params.get(name) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(false),
        Some(Value::String(text)) if text.trim().eq_ignore_ascii_case("true") => Ok(true),
        Some(Value::String(text)) if text.trim().eq_ignore_ascii_case("false") => Ok(false),
        Some(other) => Err(ToolError::ParameterValidation {
            param_name: name.to_string(),
            message: format!("`{name}` is true or false, not {other}"),
        }),
    }
}

/// Run ids, from a list or a text with commas or spaces.
fn ids(params: &HashMap<String, Value>, name: &str) -> Option<Vec<String>> {
    let ids = match params.get(name)? {
        Value::String(text) => split_ids(text),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str())
            .flat_map(split_ids)
            .collect(),
        _ => return None,
    };
    (!ids.is_empty()).then_some(ids)
}

fn failed(message: String) -> ToolError {
    ToolError::ExecutionFailed(message)
}

/// Starts a subagent: offered to the workflow services' callers while their
/// depth allows.
pub(crate) struct StartAgentTool {
    context: Arc<SharedContext>,
    caller: String,
    depth: u32,
}

impl StartAgentTool {
    pub(crate) fn new(context: Arc<SharedContext>, caller: String, depth: u32) -> Self {
        Self {
            context,
            caller,
            depth,
        }
    }
}

impl std::fmt::Debug for StartAgentTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StartAgentTool")
            .field("caller", &self.caller)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for StartAgentTool {
    fn definition(&self) -> ToolDefinition {
        let guidance = self
            .context
            .workflow_services
            .get()
            .map_or_else(String::new, |services| {
                services.target_guidance(&self.caller)
            });
        let config = self.context.subagents.config();
        let mut parameters = vec![
            param("agent", guidance, ToolParameterType::String, true),
            param(
                "task",
                "The whole task: the goal, the context and constraints it needs (it doesn't see your conversation), what done looks like, and what to report.",
                ToolParameterType::String,
                true,
            ),
            param(
                "workspace",
                "Optional: the directory it works in, relative to your working directory (default: yours).",
                ToolParameterType::String,
                false,
            ),
        ];
        if config.is_some_and(|config| config.workspace_provider.is_some()) {
            parameters.push(param(
                "isolated",
                "true: it works in a workspace of its own (a copy of your workspace that its changes stay in until they are merged). Use it for runs that change files while others do. Default false.",
                ToolParameterType::Bool,
                false,
            ));
        }
        if config.is_some_and(|config| config.peer_messages) {
            parameters.push(param(
                "peers",
                "Optional: the ids of your other agent runs it may message, and that may message it, separated by commas, or `all` for every run you start.",
                ToolParameterType::String,
                false,
            ));
            parameters.push(param(
                "group",
                "Optional: a name for a team of your runs, such as `openrouter`. The runs you start with the same group may message each other, whichever starts first, so runs started in one response can be linked.",
                ToolParameterType::String,
                false,
            ));
        }
        if let Some(config) = config {
            parameters.extend(config.start_parameters.iter().cloned());
        }
        definition(
            START_AGENT_TOOL_NAME,
            "Starts an agent on a task and returns its run id (r1, r2, ...) at once: it works while you go on, and its report reaches you when it finishes. Start several in one response to run them in parallel.",
            parameters,
            true,
        )
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let agent = required(&params, "agent")?;
        let task = match params.get("task") {
            Some(Value::String(task)) if !task.trim().is_empty() => task.clone(),
            _ => {
                return Err(ToolError::MissingParameter {
                    param_name: "task".to_string(),
                });
            }
        };
        let services = self
            .context
            .workflow_services
            .get()
            .ok_or_else(|| failed("the workflow has no services".to_string()))?;
        services
            .authorize(&self.caller, &agent)
            .map_err(|error| failed(error.to_string()))?;
        let host = self
            .context
            .subagents
            .config()
            .map(|config| config.start_parameters.clone())
            .unwrap_or_default();
        let parameters = host
            .iter()
            .filter_map(|parameter| {
                params
                    .get(&parameter.name)
                    .filter(|value| !value.is_null())
                    .map(|value| (parameter.name.clone(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let workspace = text(&params, "workspace");
        let peers = text(&params, "peers");
        let group = text(&params, "group");
        self.context
            .subagents
            .start(
                &self.context,
                context.task_id,
                &self.caller,
                self.depth,
                context.current_working_directory,
                &agent,
                &task,
                workspace.as_deref(),
                flag(&params, "isolated")?,
                peers.as_deref(),
                group.as_deref(),
                parameters,
            )
            .await
            .map_err(failed)
    }
}

/// Sends a message to an agent run: a lead's to its runs, a run's to its
/// lead or its peers.
pub(crate) struct MessageAgentTool {
    context: Arc<SharedContext>,
    caller: bool,
}

impl MessageAgentTool {
    /// `caller`: offered to a run that may start runs.
    pub(crate) fn new(context: Arc<SharedContext>, caller: bool) -> Self {
        Self { context, caller }
    }
}

impl std::fmt::Debug for MessageAgentTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MessageAgentTool")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for MessageAgentTool {
    fn definition(&self) -> ToolDefinition {
        let to = if self.caller {
            "The run id of one of your agent runs (such as r1), or `lead` for the agent that started you, if one did."
        } else {
            "`lead` for the agent that started you, or the run id of another agent run your lead linked you with."
        };
        definition(
            MESSAGE_AGENT_TOOL_NAME,
            "Sends a message to an agent run without waiting. A running one reads it at its next step. One that finished continues its task with it, with its context, and one that was stopped restarts where it stopped.",
            vec![
                param("to", to, ToolParameterType::String, true),
                param("message", "The message.", ToolParameterType::String, true),
            ],
            true,
        )
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let to = required(&params, "to")?;
        let message = required(&params, "message")?;
        self.context
            .subagents
            .message(&self.context, context.task_id, &to, &message)
            .map_err(failed)
    }
}

/// Waits for the reports and messages of the agent runs a run works with.
pub(crate) struct WaitForAgentsTool {
    context: Arc<SharedContext>,
    agent_name: String,
    /// The time limit, as `wait_for_message`'s.
    limits: WaitForMessageTool,
}

impl WaitForAgentsTool {
    pub(crate) fn new(
        context: Arc<SharedContext>,
        agent_name: String,
        llm_client: Arc<arko_ai::LLMClient>,
    ) -> Self {
        Self {
            limits: WaitForMessageTool::new(context.clone(), agent_name.clone(), llm_client),
            context,
            agent_name,
        }
    }
}

impl std::fmt::Debug for WaitForAgentsTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WaitForAgentsTool")
            .field("agent_name", &self.agent_name)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for WaitForAgentsTool {
    fn definition(&self) -> ToolDefinition {
        definition(
            WAIT_FOR_AGENTS_TOOL_NAME,
            "Waits until one of your agent runs finishes, a message reaches you (from your lead or an agent run), or a note arrives, and returns the reports and messages themselves (any that came in earlier come back at once). Use it when you can't go on without them. Call it again to keep waiting.",
            vec![
                param(
                    "runs",
                    "Optional: the ids of your own agent runs (or of runs your lead linked you with) to wait for, separated by commas (default: all of yours). Messages from your lead always end the wait.",
                    ToolParameterType::String,
                    false,
                ),
                param(
                    "all",
                    "true: wait until every one of them has finished (or a message comes), not just the first. Default false.",
                    ToolParameterType::Bool,
                    false,
                ),
                param(
                    "timeout_seconds",
                    "Optional: the most seconds to wait.",
                    ToolParameterType::Integer,
                    false,
                ),
            ],
            false,
        )
    }

    /// The wait has its own limit.
    fn uses_host_timeout(&self) -> bool {
        false
    }

    fn recovery_mode(&self) -> ToolRecovery {
        ToolRecovery::Rerun
    }

    /// Moving the wait to the background ends it, since reports and
    /// messages reach the agent as notes anyway.
    fn background_mode(&self) -> arko_agents::ToolBackground {
        arko_agents::ToolBackground::Own
    }

    fn observe_context(&self, task_id: uuid::Uuid, message: &arko_ai::Message) {
        self.limits.observe_context(task_id, message);
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let requested = match params.get("timeout_seconds") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let seconds = match value {
                    Value::Number(number) => number.as_f64(),
                    Value::String(text) if text.trim().is_empty() => None,
                    Value::String(text) => Some(text.trim().parse::<f64>().map_err(|_| {
                        ToolError::ParameterValidation {
                            param_name: "timeout_seconds".to_string(),
                            message: "Expected a positive number of seconds.".to_string(),
                        }
                    })?),
                    _ => None,
                };
                match seconds {
                    Some(seconds) if seconds.is_finite() && seconds > 0.0 => Some(seconds),
                    None => None,
                    Some(_) => {
                        return Err(ToolError::ParameterValidation {
                            param_name: "timeout_seconds".to_string(),
                            message: "Expected a positive number of seconds.".to_string(),
                        });
                    }
                }
            }
        };
        let limit = self.limits.wait_limit(context.task_id).await;
        // The model may only shorten the wait.
        let wait = match requested {
            Some(seconds) if seconds < limit.as_secs_f64() => Duration::from_secs_f64(seconds),
            _ => limit,
        };
        self.context
            .subagents
            .wait(
                &self.context,
                context.task_id,
                &self.agent_name,
                ids(&params, "runs"),
                flag(&params, "all")?,
                wait,
                &super::board::WaitSignals {
                    stop: &context.stop,
                    moved: &context.moved,
                    context_arrived: &context.context_arrived,
                    claude_code: context.runs_in_claude_code,
                },
            )
            .await
            .map_err(failed)
    }
}

/// Lists a run's agent runs.
pub(crate) struct ListAgentsTool {
    context: Arc<SharedContext>,
}

impl ListAgentsTool {
    pub(crate) fn new(context: Arc<SharedContext>) -> Self {
        Self { context }
    }
}

impl std::fmt::Debug for ListAgentsTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ListAgentsTool")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for ListAgentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            read_only: true,
            ..definition(
                LIST_AGENTS_TOOL_NAME,
                "Lists the agent runs you started: what each is doing or reported, for how long, and where it works.",
                vec![],
                true,
            )
        }
    }

    fn recovery_mode(&self) -> ToolRecovery {
        ToolRecovery::Rerun
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        Ok(self.context.subagents.list(&self.context, context.task_id))
    }
}

/// Stops one of a run's agent runs.
pub(crate) struct StopAgentTool {
    context: Arc<SharedContext>,
}

impl StopAgentTool {
    pub(crate) fn new(context: Arc<SharedContext>) -> Self {
        Self { context }
    }
}

impl std::fmt::Debug for StopAgentTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StopAgentTool")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for StopAgentTool {
    fn definition(&self) -> ToolDefinition {
        let mut parameters = vec![
            param(
                "run",
                "The run id, such as r1.",
                ToolParameterType::String,
                true,
            ),
            param(
                "reason",
                "Optional: why, as it is told.",
                ToolParameterType::String,
                false,
            ),
        ];
        if self
            .context
            .subagents
            .config()
            .is_some_and(|config| config.workspace_provider.is_some())
        {
            parameters.push(param(
                "remove_workspace",
                "true: once it has stopped, also remove its isolated workspace (it can't run again then), as when its changes are merged. Default false.",
                ToolParameterType::Bool,
                false,
            ));
        }
        definition(
            STOP_AGENT_TOOL_NAME,
            "Stops one of your agent runs: it saves its progress and reports, and message_agent restarts it later where it stopped.",
            parameters,
            true,
        )
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let run = required(&params, "run")?;
        let reason = text(&params, "reason").unwrap_or_default();
        self.context
            .subagents
            .stop(
                &self.context,
                context.task_id,
                &run,
                &reason,
                flag(&params, "remove_workspace")?,
            )
            .await
            .map_err(failed)
    }
}

/// Lets a run's agent runs message each other.
pub(crate) struct LinkAgentsTool {
    context: Arc<SharedContext>,
}

impl LinkAgentsTool {
    pub(crate) fn new(context: Arc<SharedContext>) -> Self {
        Self { context }
    }
}

impl std::fmt::Debug for LinkAgentsTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LinkAgentsTool")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for LinkAgentsTool {
    fn definition(&self) -> ToolDefinition {
        definition(
            LINK_AGENTS_TOOL_NAME,
            "Lets some of your agent runs message each other directly (message_agent with each other's ids), as when one needs what another finds.",
            vec![param(
                "runs",
                "The run ids, at least two, separated by commas.",
                ToolParameterType::String,
                true,
            )],
            true,
        )
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let runs = ids(&params, "runs").ok_or_else(|| ToolError::MissingParameter {
            param_name: "runs".to_string(),
        })?;
        self.context
            .subagents
            .link(context.task_id, &runs)
            .map_err(failed)
    }
}
