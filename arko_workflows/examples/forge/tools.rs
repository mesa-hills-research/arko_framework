//! The forge's own tools: lessons, trial runs, and a shell for the API
//! backend (agents in Claude Code use its `Bash`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use arko_agents::terminal::{CommandGuard, ShellDialect};
use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::AsyncWriteExt;

use crate::scaffold::tail;

fn parameter(name: &str, description: &str, required: bool) -> ToolParameterDefinition {
    ToolParameterDefinition {
        name: name.to_string(),
        description: description.to_string(),
        param_type: ToolParameterType::String,
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
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        output_based_example: None,
        parameters,
        parallel_safe: false,
        tool_type_details: ToolTypeDetails::DeveloperDefined,
        raw_json_schema_override: None,
    }
}

fn string<'a>(params: &'a HashMap<String, Value>, name: &str) -> Option<&'a str> {
    params.get(name).and_then(Value::as_str)
}

fn required<'a>(params: &'a HashMap<String, Value>, name: &str) -> Result<&'a str, ToolError> {
    string(params, name)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ToolError::MissingParameter {
            param_name: name.to_string(),
        })
}

/// Appends a lesson to the forge's lessons file, read at the start of every
/// run.
#[derive(Debug)]
pub struct RecordLesson {
    pub path: PathBuf,
}

#[async_trait]
impl Tool for RecordLesson {
    fn definition(&self) -> ToolDefinition {
        definition(
            "record_lesson",
            "Save a lesson for future forge runs: a general, durable insight about the framework, the tools, or building workflows, worth reading before every run. Not facts about this use case.",
            vec![parameter(
                "lesson",
                "The lesson, in one to three sentences, specific enough to act on.",
                true,
            )],
        )
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let lesson = required(&params, "lesson")?;
        let io = |source| ToolError::Io {
            tool_name: "record_lesson".into(),
            source,
        };
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(io)?;
        }
        let line = format!(
            "- ({}) {}\n",
            chrono::Utc::now().format("%Y-%m-%d"),
            lesson.split_whitespace().collect::<Vec<_>>().join(" ")
        );
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .map_err(io)?;
        file.write_all(line.as_bytes()).await.map_err(io)?;
        Ok("Recorded.".into())
    }
}

/// Builds and runs the forged workflow for real, a limited number of times.
#[derive(Debug)]
pub struct TrialRun {
    pub crate_dir: PathBuf,
    pub target_dir: PathBuf,
    pub log_dir: PathBuf,
    pub allowed: u32,
    pub used: AtomicU32,
    pub max_minutes: u64,
}

#[async_trait]
impl Tool for TrialRun {
    fn definition(&self) -> ToolDefinition {
        definition(
            "trial_run",
            "Build the workflow and run it for real (`cargo run -- <arguments>` in the crate), with real model usage. Returns the exit status, the end of its output, and the path of the full log. The number of runs is limited.",
            vec![
                parameter(
                    "arguments",
                    "The workflow's command-line arguments, as you would type them after `--` (shell quoting applies).",
                    true,
                ),
                parameter(
                    "input",
                    "Text for the workflow's standard input, such as answers to the questions it will ask, one per line. Optional.",
                    false,
                ),
                parameter(
                    "timeout_minutes",
                    "Stop the run after this many minutes. Optional; defaults to the most allowed.",
                    false,
                ),
            ],
        )
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let arguments = string(&params, "arguments").unwrap_or_default();
        let minutes = match string(&params, "timeout_minutes") {
            Some(value) => value
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|minutes| (1..=self.max_minutes).contains(minutes))
                .ok_or_else(|| ToolError::ParameterValidation {
                    param_name: "timeout_minutes".into(),
                    message: format!("a whole number from 1 to {}", self.max_minutes),
                })?,
            None => self.max_minutes,
        };
        let run = self.used.fetch_add(1, Ordering::SeqCst) + 1;
        if run > self.allowed {
            self.used.fetch_sub(1, Ordering::SeqCst);
            return Err(ToolError::ExecutionFailed(format!(
                "No trial runs left: all {} have been used.",
                self.allowed
            )));
        }
        let command = format!("cargo run --quiet -- {arguments}");
        let mut process = shell(&command);
        process
            .current_dir(&self.crate_dir)
            .env("CARGO_TARGET_DIR", &self.target_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        process.process_group(0);
        let started = Instant::now();
        let mut child = process.spawn().map_err(|source| ToolError::Io {
            tool_name: "trial_run".into(),
            source,
        })?;
        let pid = child.id();
        if let Some(mut stdin) = child.stdin.take() {
            let input = string(&params, "input").unwrap_or_default().to_string();
            tokio::spawn(async move {
                let _ = stdin.write_all(input.as_bytes()).await;
            });
        }
        let outcome =
            tokio::time::timeout(Duration::from_secs(minutes * 60), child.wait_with_output()).await;
        let (status, stdout, stderr) = match outcome {
            Ok(Ok(output)) => (
                match output.status.code() {
                    Some(code) => format!("exit code {code}"),
                    None => "killed by a signal".into(),
                },
                String::from_utf8_lossy(&output.stdout).into_owned(),
                String::from_utf8_lossy(&output.stderr).into_owned(),
            ),
            Ok(Err(source)) => {
                return Err(ToolError::Io {
                    tool_name: "trial_run".into(),
                    source,
                });
            }
            Err(_) => {
                kill_group(pid);
                (
                    format!("stopped after {minutes} minutes"),
                    String::new(),
                    String::new(),
                )
            }
        };
        let log = self.log_dir.join(format!("trial-{run}.log"));
        let _ = std::fs::create_dir_all(&self.log_dir);
        let _ = std::fs::write(
            &log,
            format!("$ {command}\n{status}\n\n## stdout\n{stdout}\n\n## stderr\n{stderr}\n"),
        );
        Ok(format!(
            "Trial run {run} of {allowed}: {status} after {seconds}s. Full log: {log}\n\n## stdout (end)\n{stdout}\n\n## stderr (end)\n{stderr}",
            allowed = self.allowed,
            seconds = started.elapsed().as_secs(),
            log = log.display(),
            stdout = tail(&stdout, 8_000),
            stderr = tail(&stderr, 8_000),
        ))
    }
}

/// Stops a timed-out trial and everything it started.
fn kill_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
    }
    #[cfg(not(unix))]
    let _ = pid;
}

fn shell(command: &str) -> tokio::process::Command {
    if cfg!(windows) {
        let mut process = tokio::process::Command::new("cmd.exe");
        process.args(["/D", "/S", "/C", command]);
        process
    } else {
        let mut process = tokio::process::Command::new("bash");
        process.args(["-lc", command]);
        process
    }
}

/// A shell for agents on the API backend, behind arko's command guard.
#[derive(Debug)]
pub struct RunShell {
    pub guard: CommandGuard,
    pub target_dir: PathBuf,
}

#[async_trait]
impl Tool for RunShell {
    fn definition(&self) -> ToolDefinition {
        definition(
            "run_shell",
            "Run one shell command in the workspace and return its output: searching (grep, find), building and testing (cargo). Commands that would delete or overwrite paths outside the workspace are refused. Stops after 20 minutes.",
            vec![parameter("command", "The command.", true)],
        )
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let command = required(&params, "command")?;
        let dialect = if cfg!(windows) {
            ShellDialect::Cmd
        } else {
            ShellDialect::Posix
        };
        self.guard
            .check(command, dialect, context.current_working_directory)
            .map_err(|refusal| ToolError::ExecutionFailed(refusal.message))?;
        let mut process = shell(command);
        process
            .current_dir(context.current_working_directory)
            .env("CARGO_TARGET_DIR", &self.target_dir)
            .stdin(Stdio::null())
            .kill_on_drop(true);
        let output = tokio::time::timeout(Duration::from_secs(20 * 60), process.output())
            .await
            .map_err(|_| ToolError::ExecutionFailed("timed out after 20 minutes".into()))?
            .map_err(|source| ToolError::Io {
                tool_name: "run_shell".into(),
                source,
            })?;
        let stdout = tail(&String::from_utf8_lossy(&output.stdout), 24_000);
        let stderr = tail(&String::from_utf8_lossy(&output.stderr), 12_000);
        if !output.status.success() {
            return Err(ToolError::ShellCommandFailed {
                tool_name: "run_shell".into(),
                exit_code: output.status.code(),
                stdout,
                stderr,
            });
        }
        Ok(format!("stdout:\n{stdout}\nstderr:\n{stderr}"))
    }
}

/// Why the forge refuses a shell `command`, if it does: version control and
/// publishing are the user's, never the agents'.
pub fn refused_command(command: &str) -> Option<&'static str> {
    let words: Vec<&str> = command
        .split(|c: char| c.is_whitespace() || ";|&()`$<>'\"{}".contains(c))
        .filter(|word| !word.is_empty())
        .collect();
    let program = |word: &str| {
        let name = word.rsplit(['/', '\\']).next().unwrap_or(word);
        name.strip_suffix(".exe").unwrap_or(name).to_string()
    };
    for (index, word) in words.iter().enumerate() {
        match program(word).as_str() {
            "git" | "gh" => {
                return Some(
                    "The forge doesn't run git or gh: version control and publishing are the user's. Leave the crate's files as they are; the user reviews and commits them.",
                );
            }
            "cargo"
                if words.get(index + 1).is_some_and(|next| {
                    ["publish", "login", "logout", "owner", "yank"].contains(next)
                }) =>
            {
                return Some("The forge doesn't publish crates or change registry logins.");
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::refused_command;

    #[test]
    fn version_control_and_publishing_are_refused() {
        for command in [
            "git status",
            "cd /repo && git commit -am wip",
            "/usr/bin/git push origin main",
            "echo $(git rev-parse HEAD)",
            "gh pr create",
            "cargo publish --dry-run",
            "(cargo login token)",
        ] {
            assert!(refused_command(command).is_some(), "{command}");
        }
        for command in [
            "cargo build --all-targets",
            "grep -rn digit src",
            "cat .gitignore",
            "rg legit_name",
            "cargo test publish_tests",
        ] {
            assert!(refused_command(command).is_none(), "{command}");
        }
    }
}
