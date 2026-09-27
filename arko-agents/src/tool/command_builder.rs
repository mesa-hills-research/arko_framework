use super::ToolError;
use super::definition::ToolParameterDefinition;
use super::definition::{ToolParameterType, parse_bool_text};
use serde_json::Value;
use std::collections::HashMap;
use std::ffi::OsString;
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
    let mut command_args: Vec<OsString> = Vec::new();
    let mut positional_args_values: HashMap<usize, OsString> = HashMap::new();
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
                // A path parameter passes the path the workspace check
                // judged: absolute, so the program can't read it as an
                // option, and with `..` already resolved, so a symbolic link
                // on the way can't take it elsewhere.
                let argument = if def.is_path && !str_val.is_empty() {
                    crate::utils::resolve_path(str_val, task_cwd)
                        .map_err(|error| ToolError::ParameterValidation {
                            param_name: def.name.clone(),
                            message: format!("Invalid path '{str_val}': {error}"),
                        })?
                        .into_os_string()
                } else {
                    OsString::from(str_val)
                };
                if let Some(flag_str) = &def.flag {
                    command_args.push(flag_str.into());
                    command_args.push(argument);
                } else {
                    let pos_idx = positional_args_values.len();
                    positional_args_values.insert(pos_idx, argument);
                    if pos_idx > max_pos_idx {
                        max_pos_idx = pos_idx;
                    }
                }
            }
            ToolParameterType::Bool => {
                // Also `"true"` or `"false"` text, as output-based calls and
                // hand-written defaults give it.
                let bool_val = current_value
                    .as_bool()
                    .or_else(|| current_value.as_str().and_then(parse_bool_text))
                    .ok_or_else(|| ToolError::ParameterValidation {
                        param_name: def.name.clone(),
                        message: format!(
                            "Expected boolean for parameter '{}', got {:?}",
                            def.name, current_value
                        ),
                    })?;
                if bool_val {
                    if let Some(flag_str) = &def.flag {
                        command_args.push(flag_str.into());
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

    let mut final_cmd_parts: Vec<OsString> = Vec::new();
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
        final_cmd_parts.push(part.into());
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
        program.display(),
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
            ToolError::ExecutionFailed(format!(
                "Failed to spawn command '{}': {}",
                program.display(),
                e
            ))
        })?
        .wait_with_output()
        .await
        .map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Error waiting for command '{}': {}",
                program.display(),
                e
            ))
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn boolean_parameters_accept_true_and_false_text() {
        let definitions = [ToolParameterDefinition {
            name: "shout".into(),
            description: "Shout".into(),
            param_type: ToolParameterType::Bool,
            required: false,
            is_path: false,
            flag: Some("--shout".into()),
            default_value: None,
        }];
        let run = |value: Value| {
            let definitions = definitions.clone();
            async move {
                let params = HashMap::from([("shout".to_string(), value)]);
                execute_command_from_template(
                    "say",
                    "echo hi",
                    &definitions,
                    &params,
                    &std::env::temp_dir(),
                )
                .await
            }
        };
        assert_eq!(run(Value::Bool(true)).await.unwrap(), "hi --shout\n");
        assert_eq!(run(Value::from("True")).await.unwrap(), "hi --shout\n");
        assert_eq!(run(Value::from(" false ")).await.unwrap(), "hi\n");
        assert!(matches!(
            run(Value::from("loudly")).await,
            Err(ToolError::ParameterValidation { .. })
        ));
    }

    fn path_parameter() -> [ToolParameterDefinition; 1] {
        [ToolParameterDefinition {
            name: "file".into(),
            description: "A file".into(),
            param_type: ToolParameterType::String,
            required: true,
            is_path: true,
            flag: None,
            default_value: None,
        }]
    }

    async fn run_with_path(template: &str, file: &str, cwd: &Path) -> Result<String, ToolError> {
        let params = HashMap::from([("file".to_string(), Value::from(file))]);
        execute_command_from_template("fixture", template, &path_parameter(), &params, cwd).await
    }

    #[tokio::test]
    async fn path_parameters_pass_the_path_the_workspace_check_judged() {
        let root = std::fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("arko-command-paths-{}", uuid::Uuid::new_v4()));
        let workspace = root.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(root.join("outside/deep")).unwrap();
        std::fs::write(root.join("outside/x"), "OUTSIDE").unwrap();
        std::fs::write(workspace.join("x"), "INSIDE").unwrap();
        std::os::unix::fs::symlink(root.join("outside/deep"), workspace.join("link")).unwrap();

        // An option-like value is inside the workspace as a path, and the
        // program gets it as one: absolute, never an option.
        assert!(crate::utils::is_path_in_workspace("-n", &workspace).unwrap());
        assert_eq!(
            run_with_path("echo", "-n", &workspace).await.unwrap(),
            format!("{}\n", workspace.join("-n").display())
        );
        // `link/..` is the workspace lexically, which the check judged; the
        // kernel would follow the link to `outside` first.
        assert!(crate::utils::is_path_in_workspace("link/../x", &workspace).unwrap());
        assert_eq!(
            run_with_path("cat", "link/../x", &workspace).await.unwrap(),
            "INSIDE"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
