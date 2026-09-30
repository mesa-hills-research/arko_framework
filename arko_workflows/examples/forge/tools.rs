//! The forge's own tools: lessons, trial runs, and a shell for the API
//! backend (agents in Claude Code use its `Bash`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use arko_agents::terminal::{CommandGuard, ShellDialect};
use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use arko_workflows::budget::CostSource;
use arko_workflows::context::SharedContext;
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
    /// The most all trials together may reserve (`--trial-budget`).
    pub trial_budget: Option<f64>,
    /// The workflow, once built: trials reserve against its budget ledger,
    /// so they count toward the forge's own `--budget`.
    pub workflow: Arc<OnceLock<Arc<SharedContext>>>,
}

/// The ledger description of a trial's reservation.
pub const TRIAL_RESERVATION: &str = "trial run reservation";

impl TrialRun {
    /// What the next trial may still reserve: the trial budget less what
    /// earlier trials reserved, and the forge's budget less what it has
    /// spent. `None` when neither is set.
    fn reservable(&self) -> Option<f64> {
        let context = self.workflow.get();
        let reserved: f64 = context
            .map(|context| {
                context
                    .budget
                    .snapshot_ledger()
                    .entries
                    .iter()
                    .filter(|entry| {
                        matches!(&entry.source, CostSource::Manual { description }
                            if description == TRIAL_RESERVATION)
                    })
                    .map(|entry| entry.amount)
                    .sum()
            })
            .unwrap_or(0.0);
        let trials = self.trial_budget.map(|limit| limit - reserved);
        let forge = context.and_then(|context| {
            context
                .budget
                .limit()
                .map(|limit| limit - context.budget.total_spent())
        });
        match (trials, forge) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
        .map(|left| left.max(0.0))
    }
}

/// Whether `arguments` pass `budget` to the workflow: some number in them
/// equals it.
fn passes_budget(arguments: &str, budget: f64) -> bool {
    arguments
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .filter_map(|word| word.trim_matches('.').parse::<f64>().ok())
        .any(|number| (number - budget).abs() < 1e-9)
}

#[async_trait]
impl Tool for TrialRun {
    fn definition(&self) -> ToolDefinition {
        definition(
            "trial_run",
            "Build the workflow and run it for real (`cargo run -- <arguments>` in the crate), with real model usage. Returns the exit status, the end of its output, and the path of the full log. The number of runs is limited, and so is what they may spend: each run reserves its `budget` in full, which the arguments must pass to the workflow's own budget flag.",
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
                    "budget",
                    "The most this run may spend, in dollars at list prices. It is reserved in full from the trial budget, and the same amount must appear in `arguments` as the workflow's budget flag. Required when a trial or forge budget is set.",
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
        let budget = match string(&params, "budget") {
            Some(value) => Some(
                value
                    .trim()
                    .trim_start_matches('$')
                    .parse::<f64>()
                    .ok()
                    .filter(|budget| budget.is_finite() && *budget > 0.0)
                    .ok_or_else(|| ToolError::ParameterValidation {
                        param_name: "budget".into(),
                        message: "a positive number of dollars".into(),
                    })?,
            ),
            None => None,
        };
        let reservable = self.reservable();
        if let Some(left) = reservable {
            let Some(budget) = budget else {
                return Err(ToolError::MissingParameter {
                    param_name: "budget".into(),
                });
            };
            if budget > left {
                return Err(ToolError::ExecutionFailed(format!(
                    "The trial budget has ${left:.2} left; reserve at most that, or report that trials need more."
                )));
            }
        }
        if let Some(budget) = budget
            && !passes_budget(arguments, budget)
        {
            return Err(ToolError::ExecutionFailed(format!(
                "The arguments don't pass the reserved ${budget} to the workflow: add its budget flag with that amount (for example `--budget {budget}`)."
            )));
        }
        let run = self.used.fetch_add(1, Ordering::SeqCst) + 1;
        if run > self.allowed {
            self.used.fetch_sub(1, Ordering::SeqCst);
            return Err(ToolError::ExecutionFailed(format!(
                "No trial runs left: all {} have been used.",
                self.allowed
            )));
        }
        if let (Some(budget), Some(context)) = (budget, self.workflow.get()) {
            // Reserved in full: the trial's own spend isn't visible here.
            context.budget.add_manual_cost(
                budget,
                TRIAL_RESERVATION,
                Some(format!("trial run {run}")),
            );
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
                    "The forge doesn't run git or gh: version control and publishing are the user's. Leave the crate's files as they are; the user reviews and commits them. Don't try to get around this refusal; if it blocks work you need, say so in your summary.",
                );
            }
            "cargo"
                if words.get(index + 1).is_some_and(|next| {
                    ["publish", "login", "logout", "owner", "yank"].contains(next)
                }) =>
            {
                return Some(
                    "The forge doesn't publish crates or change registry logins. Don't try to get around this refusal; if it blocks work you need, say so in your summary.",
                );
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
    fn a_trial_must_pass_its_reserved_budget() {
        use super::passes_budget;
        assert!(passes_budget("--repo x --budget 2.5 --verbose", 2.5));
        assert!(passes_budget("--budget=3", 3.0));
        assert!(!passes_budget("--budget 25", 2.5));
        assert!(!passes_budget("--repo x", 2.0));
    }

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
