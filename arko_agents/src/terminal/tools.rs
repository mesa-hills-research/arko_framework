use super::*;
use crate::tool::{
    Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType, ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
enum Operation {
    Start,
    Status,
    Input,
    Control,
}

#[derive(Debug)]
pub struct TerminalTool {
    manager: Arc<TerminalManager>,
    operation: Operation,
}
impl TerminalTool {
    pub const NAMES: [&'static str; 4] = [
        "terminal_start",
        "terminal_status",
        "terminal_input",
        "terminal_control",
    ];
    pub fn all(manager: Arc<TerminalManager>) -> Vec<Self> {
        [
            Operation::Start,
            Operation::Status,
            Operation::Input,
            Operation::Control,
        ]
        .into_iter()
        .map(|operation| Self {
            manager: manager.clone(),
            operation,
        })
        .collect()
    }
}
pub(super) fn parameter(name: &str, description: &str, required: bool) -> ToolParameterDefinition {
    ToolParameterDefinition {
        name: name.into(),
        description: description.into(),
        param_type: ToolParameterType::String,
        required,
        is_path: false,
        flag: None,
        default_value: None,
    }
}
pub(super) fn string<'a>(
    params: &'a HashMap<String, Value>,
    name: &str,
) -> std::result::Result<&'a str, ToolError> {
    params
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::ParameterValidation {
            param_name: name.into(),
            message: "expected a string".into(),
        })
}
/// A whole-number parameter: a number, or text holding one (as calls saved
/// before the parameters were integers give it).
pub(super) fn number(
    params: &HashMap<String, Value>,
    name: &str,
) -> std::result::Result<Option<u64>, ToolError> {
    params
        .get(name)
        .map(|value| {
            crate::tool::definition::integer_argument(value)
                .and_then(|number| u64::try_from(number).ok())
                .ok_or_else(|| ToolError::ParameterValidation {
                    param_name: name.into(),
                    message: "expected a whole number, 0 or more".into(),
                })
        })
        .transpose()
}
pub(super) fn boolean(
    params: &HashMap<String, Value>,
    name: &str,
    default: bool,
) -> std::result::Result<bool, ToolError> {
    match params.get(name) {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::String(value)) if value == "true" || value == "false" => Ok(value == "true"),
        _ => Err(ToolError::ParameterValidation {
            param_name: name.into(),
            message: "expected true or false".into(),
        }),
    }
}
fn revision(params: &HashMap<String, Value>) -> std::result::Result<u64, ToolError> {
    number(params, "revision")?.ok_or_else(|| ToolError::MissingParameter {
        param_name: "revision".into(),
    })
}
pub(super) fn tool_error(error: impl std::fmt::Display) -> ToolError {
    ToolError::ExecutionFailed(error.to_string())
}

#[async_trait]
impl Tool for TerminalTool {
    fn definition(&self) -> ToolDefinition {
        let (name, description, parameters) = match self.operation {
            Operation::Start => (
                Self::NAMES[0],
                "Starts a command in the background and returns its job ID at once, so you can go on with other work. A terminal update tells you when the job finishes, and when it seems to wait for input. Read its output with terminal_status and answer it with terminal_input.",
                vec![
                    parameter(
                        "command",
                        "The command, for the configured shell (by default `/bin/sh -c` on Unix and `cmd.exe /D /S /C` on Windows).",
                        true,
                    ),
                    parameter(
                        "terminal",
                        "true (the default): run it in a terminal (a PTY), as programs run for a person. false: with plain pipes.",
                        false,
                    ),
                    parameter(
                        "detect_prompts",
                        "true (the default): also tell you when its output looks like a prompt waiting for input.",
                        false,
                    ),
                    parameter(
                        "timeout_ms",
                        "Stop the job after this many milliseconds (default: no limit).",
                        false,
                    ),
                ],
            ),
            Operation::Status => (
                Self::NAMES[1],
                "Shows a job's state and latest output. Pass the `output.end` of your last read as after_output to get only what's new. `gap: true` means part of that output is no longer kept: the file at `log_path` has it all, unless `log_truncated`. `input` is set while the job seems to wait for input: source `program` means the program asked for it, and `prompt_rule` that its output looks like a prompt. A job with no new output may still be working.",
                vec![
                    parameter("job_id", "The job's ID.", true),
                    parameter(
                        "after_output",
                        "The `output.end` of your last read, to get only the output after it.",
                        false,
                    ),
                ],
            ),
            Operation::Input => (
                Self::NAMES[2],
                "Sends text to a running job, exactly as given. End a line with \\n (\\r in a terminal on Windows), or leave it out to send a single key. Give the job's latest revision, so input never goes to a prompt you haven't seen: if the job changed since, the input is refused, so read its status again. If input can't be fully delivered, the job is stopped. A job's request for input doesn't allow anything your task doesn't.",
                vec![
                    parameter("job_id", "The job's ID.", true),
                    parameter(
                        "revision",
                        "The job's latest revision, from terminal_status or a terminal update.",
                        true,
                    ),
                    parameter("text", "The exact input, 1 to 4,096 bytes.", true),
                ],
            ),
            Operation::Control => (
                Self::NAMES[3],
                "Lists your jobs (action `list`), waits until a job changes, for at most 30 seconds (`wait`), cancels a job and the processes it started (`cancel`), or forgets a finished job but keeps its log (`forget`). A terminal update tells you when a job finishes or seems to wait for input, so wait only when you can't go on without it.",
                vec![
                    parameter("action", "list, wait, cancel or forget.", true),
                    parameter("job_id", "The job's ID (not for list).", false),
                    parameter(
                        "revision",
                        "wait: the job's latest revision you saw (required). The wait returns once it changes.",
                        false,
                    ),
                    parameter(
                        "wait_ms",
                        "wait: the most milliseconds to wait (default 1000, at most 30000).",
                        false,
                    ),
                ],
            ),
        };
        let mut parameters = parameters;
        for param in &mut parameters {
            match param.name.as_str() {
                "terminal" | "detect_prompts" => param.param_type = ToolParameterType::Bool,
                "timeout_ms" | "after_output" | "revision" | "wait_ms" => {
                    param.param_type = ToolParameterType::Integer
                }
                _ => {}
            }
        }
        ToolDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
            output_based_example: None,
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            // The others start, type into or stop jobs.
            read_only: matches!(self.operation, Operation::Status),
            raw_json_schema_override: None,
        }
    }
    fn allow_result_interception(&self) -> bool {
        false
    }
    /// Jobs run in the background with their own deadlines, and waits end
    /// after at most 30 seconds.
    fn uses_host_timeout(&self) -> bool {
        false
    }
    fn bind_runtime(&self, task_id: Uuid, sender: mpsc::WeakSender<UserCommand>) {
        self.manager.bind_runtime(task_id, sender);
    }
    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> std::result::Result<String, ToolError> {
        let result = match self.operation {
            Operation::Start => serde_json::to_value(
                self.manager
                    .start(
                        context,
                        string(&params, "command")?,
                        StartOptions {
                            terminal: boolean(&params, "terminal", true)?,
                            detect_prompts: boolean(&params, "detect_prompts", true)?,
                            timeout: number(&params, "timeout_ms")?.map(Duration::from_millis),
                            ..Default::default()
                        },
                    )
                    .map_err(tool_error)?,
            )
            .map_err(tool_error)?,
            Operation::Status => serde_json::to_value(
                self.manager
                    .status(
                        context.task_id,
                        Uuid::parse_str(string(&params, "job_id")?).map_err(tool_error)?,
                        number(&params, "after_output")?,
                    )
                    .map_err(tool_error)?,
            )
            .map_err(tool_error)?,
            Operation::Input => {
                self.manager
                    .input(
                        context.task_id,
                        Uuid::parse_str(string(&params, "job_id")?).map_err(tool_error)?,
                        revision(&params)?,
                        string(&params, "text")?.to_string(),
                    )
                    .await
                    .map_err(tool_error)?;
                json!({"input_written": true})
            }
            Operation::Control => {
                let action = string(&params, "action")?;
                if action == "list" {
                    let jobs = self.manager.jobs.lock().unwrap();
                    Value::Array(jobs.values().filter_map(|job| {
                        let data = job.data.lock().unwrap();
                        (data.record.owner == context.task_id && !data.record.held).then(|| json!({"job_id":data.record.job_id,"state":data.record.state,"revision":data.record.revision}))
                    }).collect())
                } else {
                    let id = Uuid::parse_str(string(&params, "job_id")?).map_err(tool_error)?;
                    match action {
                        "wait" => serde_json::to_value(
                            self.manager
                                .wait(
                                    context.task_id,
                                    id,
                                    revision(&params)?,
                                    Duration::from_millis(
                                        number(&params, "wait_ms")?.unwrap_or(1000),
                                    ),
                                )
                                .await
                                .map_err(tool_error)?,
                        )
                        .map_err(tool_error)?,
                        "cancel" => {
                            self.manager
                                .cancel(context.task_id, id)
                                .map_err(tool_error)?;
                            json!({"cancellation_requested":true})
                        }
                        "forget" => {
                            self.manager
                                .forget(context.task_id, id)
                                .map_err(tool_error)?;
                            json!({"forgotten":true})
                        }
                        _ => return Err(tool_error("unknown terminal action")),
                    }
                }
            }
        };
        serde_json::to_string(&result).map_err(tool_error)
    }
}
