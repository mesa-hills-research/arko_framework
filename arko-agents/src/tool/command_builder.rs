use super::ToolError;
use super::definition::ToolParameterDefinition;
use super::definition::ToolParameterType;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

/// Executes a shell command from a template with the provided parameters.
pub(crate) async fn execute_command_from_template(
    tool_name: &str,
    command_template_base: &str,
    param_defs: &[ToolParameterDefinition],
    provided_params: &HashMap<String, Value>,
    task_cwd: &Path,
) -> Result<String, ToolError> {
    let mut command_args: Vec<String> = Vec::new();
    let mut positional_args_values: HashMap<usize, String> = HashMap::new();
    let mut max_pos_idx = 0;

    for def in param_defs {
        let value_or_default = provided_params
            .get(&def.name)
            .or(def.default_value.as_ref());

        let current_value = match value_or_default {
            Some(v) => v.clone(),
            None if def.required => {
                return Err(ToolError::MissingParameter {
                    param_name: def.name.clone(),
                });
            }
            None => {
                if def.param_type == ToolParameterType::Bool {
                    Value::Bool(false)
                } else {
                    continue;
                }
            }
        };

        match def.param_type {
            ToolParameterType::String => {
                let str_val =
                    current_value
                        .as_str()
                        .ok_or_else(|| ToolError::ParameterValidation {
                            param_name: def.name.clone(),
                            message: format!(
                                "Expected string for parameter '{}', got {:?}",
                                def.name, current_value
                            ),
                        })?;
                if str_val.is_empty() && !def.required {
                    continue;
                }
                if let Some(flag_str) = &def.flag {
                    command_args.push(flag_str.clone());
                    command_args.push(str_val.to_string());
                } else {
                    let pos_idx = positional_args_values.len();
                    positional_args_values.insert(pos_idx, str_val.to_string());
                    if pos_idx > max_pos_idx {
                        max_pos_idx = pos_idx;
                    }
                }
            }
            ToolParameterType::Bool => {
                let bool_val =
                    current_value
                        .as_bool()
                        .ok_or_else(|| ToolError::ParameterValidation {
                            param_name: def.name.clone(),
                            message: format!(
                                "Expected boolean for parameter '{}', got {:?}",
                                def.name, current_value
                            ),
                        })?;
                if bool_val {
                    if let Some(flag_str) = &def.flag {
                        command_args.push(flag_str.clone());
                    } else {
                        return Err(ToolError::Configuration {
                            tool_name: tool_name.to_string(),
                            message: format!("Boolean parameter '{}' must have a flag.", def.name),
                        });
                    }
                }
            }
        }
    }

    let mut final_cmd_parts: Vec<String> = Vec::new();
    let expanded_base_command = shellexpand::full(command_template_base)
        .map_err(|e| ToolError::ShellCommandTemplate {
            tool_name: tool_name.to_string(),
            message: format!(
                "Failed to expand command template base '{}': {}",
                command_template_base, e
            ),
        })?
        .into_owned();

    expanded_base_command.split_whitespace().for_each(|part| {
        final_cmd_parts.push(part.to_string());
    });

    for i in 0..=max_pos_idx {
        if let Some(val) = positional_args_values.get(&i) {
            final_cmd_parts.push(val.clone());
        }
    }

    final_cmd_parts.extend(command_args);

    if final_cmd_parts.is_empty() {
        return Err(ToolError::ShellCommandTemplate {
            tool_name: tool_name.to_string(),
            message: "Command template resulted in an empty command.".to_string(),
        });
    }

    let program = final_cmd_parts[0].clone();
    let args = &final_cmd_parts[1..];

    log::debug!(
        "Executing command for tool '{}' in CWD '{}': {} {:?}",
        tool_name,
        task_cwd.display(),
        program,
        args
    );

    let mut cmd = Command::new(&program);
    cmd.args(args);
    cmd.current_dir(task_cwd);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    // A task stopped mid-call (its runtime aborted) doesn't leave the
    // command running.
    cmd.kill_on_drop(true);

    let output = cmd
        .spawn()
        .map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to spawn command '{}': {}", program, e))
        })?
        .wait_with_output()
        .await
        .map_err(|e| {
            ToolError::ExecutionFailed(format!("Error waiting for command '{}': {}", program, e))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if output.status.success() {
        Ok(stdout)
    } else {
        Err(ToolError::ShellCommandFailed {
            tool_name: tool_name.to_string(),
            exit_code: output.status.code(),
            stdout,
            stderr,
        })
    }
}
