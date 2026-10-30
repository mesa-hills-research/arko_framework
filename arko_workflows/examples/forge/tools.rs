//! The forge's own tools: lessons, trial runs, and a shell for the API
//! backend (agents in Claude Code use its `Bash`), with the command rules
//! every agent's shell follows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use arko_agents::terminal::{CommandGuard, CommandRule, ShellDialect};
use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use arko_workflows::budget::CostSource;
use arko_workflows::context::SharedContext;
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

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
        read_only: false,
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
        // Output is collected as it arrives, so a run stopped at its time
        // limit still reports what it printed.
        let stdout = collect(child.stdout.take());
        let stderr = collect(child.stderr.take());
        let outcome = tokio::time::timeout(Duration::from_secs(minutes * 60), child.wait()).await;
        let status = match outcome {
            Ok(Ok(status)) => match status.code() {
                Some(code) => format!("exit code {code}"),
                None => "killed by a signal".into(),
            },
            Ok(Err(source)) => {
                return Err(ToolError::Io {
                    tool_name: "trial_run".into(),
                    source,
                });
            }
            Err(_) => {
                kill_group(pid);
                format!("stopped after {minutes} minutes")
            }
        };
        let stdout = collected(stdout).await;
        let stderr = collected(stderr).await;
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

/// Reads a trial's output stream as it arrives, until it ends.
fn collect(
    stream: Option<impl AsyncRead + Unpin + Send + 'static>,
) -> tokio::task::JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        let mut output = Vec::new();
        if let Some(mut stream) = stream {
            let mut chunk = [0u8; 8192];
            while let Ok(read) = stream.read(&mut chunk).await {
                if read == 0 {
                    break;
                }
                output.extend_from_slice(&chunk[..read]);
            }
        }
        output
    })
}

/// What [`collect`] read: all of it once the stream has ended, or, when
/// something the trial started still holds the stream open a few seconds
/// after it stopped, nothing (the log says so).
async fn collected(reader: tokio::task::JoinHandle<Vec<u8>>) -> String {
    match tokio::time::timeout(Duration::from_secs(10), reader).await {
        Ok(Ok(output)) => String::from_utf8_lossy(&output).into_owned(),
        _ => "(the output stream was still open 10 s after the run stopped)".into(),
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
    /// Further environment variables for its commands.
    pub env: Vec<(String, String)>,
}

impl RunShell {
    /// The shell of an agent, or of a spike when `spike_root` is its
    /// scratch crates' folder.
    pub fn new(spike_root: Option<&Path>, target_dir: PathBuf) -> Self {
        Self {
            guard: CommandGuard {
                rules: command_rules(spike_root),
                ..CommandGuard::default()
            },
            target_dir,
            env: spike_root.map(spike_environment).unwrap_or_default(),
        }
    }
}

/// The environment of a spike's shell: git looks for a repository no higher
/// than the spikes' folder, so a git command outside a scratch crate's toy
/// repository fails rather than acting on a repository around the state
/// directory (`--state-dir` can put it inside the user's). git never looks
/// in a ceiling directory or above it, except when it starts in the ceiling
/// itself, so the ceiling is the folder above the spikes'.
pub fn spike_environment(spike_root: &Path) -> Vec<(String, String)> {
    let ceiling = spike_root.parent().unwrap_or(spike_root);
    vec![(
        "GIT_CEILING_DIRECTORIES".to_string(),
        ceiling.display().to_string(),
    )]
}

#[async_trait]
impl Tool for RunShell {
    fn definition(&self) -> ToolDefinition {
        definition(
            "run_shell",
            "Run one shell command in the workspace and return its output: searching (grep, find), building and testing (cargo). Commands that would delete or overwrite paths outside the workspace are refused, and so are git and gh commands that change anything (reading runs) and publishing. Stops after 20 minutes.",
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
            .envs(self.env.iter().map(|(name, value)| (name, value)))
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

/// The command rules of every agent's shell (Claude Code's `Bash` and
/// `run_shell`): version control and publishing are the user's, so git and
/// gh may only read, and cargo doesn't publish. A spike (`spike_root`, its
/// scratch crates' folder) may also use git fully there, to test claims
/// about git in toy repositories.
pub fn command_rules(spike_root: Option<&Path>) -> Vec<CommandRule> {
    const VERSION_CONTROL: &str = "Version control and publishing are the user's: the forge allows git and gh only to read (status, log, diff, show and the like) and leaves commits, branches and the index to the user. Don't try to get around this refusal; if it blocks work you need, say so in your summary or submission.";
    const PUBLISHING: &str = "The forge doesn't publish crates or change registry logins: publishing is the user's. Don't try to get around this refusal; if it blocks work you need, say so in your summary or submission.";
    let mut rules: Vec<CommandRule> = [
        "git --version",
        "git status *",
        "git log *",
        "git diff *",
        "git show *",
        "git rev-parse *",
        "git rev-list *",
        "git merge-base *",
        "git cat-file *",
        "git ls-files *",
        "git ls-tree *",
        "git blame *",
        "git describe *",
        "git shortlog *",
        "git grep *",
        // Only these forms: `git branch <name>` creates a branch.
        "git branch",
        "git branch -a",
        "git branch -r",
        "git branch -v",
        "git branch -vv",
        "git branch -a -v",
        "git branch -a -vv",
        "git branch --list *",
        "git branch --show-current",
        "git tag",
        "git tag --list *",
        "git stash list *",
        "git reflog",
        "git reflog show *",
        "git show-ref *",
        "git for-each-ref *",
        "git worktree list *",
        "git submodule status *",
        "git check-ignore *",
        "git help *",
        "git remote -v",
        "git config --get *",
        "git config --list *",
        "gh --version",
        "gh pr view *",
        "gh pr list *",
        "gh pr diff *",
        "gh pr checks *",
        "gh pr status *",
        "gh issue view *",
        "gh issue list *",
        "gh repo view *",
        "gh auth status *",
        "gh run list *",
        "gh run view *",
    ]
    .into_iter()
    .map(|pattern| CommandRule::allow(pattern).because(VERSION_CONTROL))
    .collect();
    // `log`, `diff` and `show` write files with `--output`.
    rules.push(CommandRule::deny("git * --output* *").because(
        "`--output` writes a file, which the forge's read-only git doesn't do: read the output from the command instead.",
    ));
    // `git grep -O` opens the matching files in a program, also bundled
    // with other short options (`-nO`).
    for option in ["-*O*", "--open-files-in-pager*"] {
        rules.push(CommandRule::deny(format!("git grep * {option} *")).because(
            "`git grep -O` runs a program on the matching files, which the forge's read-only git doesn't do: read the matches from the command's output instead.",
        ));
    }
    // `cargo run -- publish` and `cargo test login` run.
    for command in ["publish", "login", "logout", "owner", "yank"] {
        for pattern in [
            format!("cargo {command} *"),
            format!("cargo +* {command} *"),
            format!("cargo -* {command} *"),
        ] {
            rules.push(CommandRule::deny(pattern).because(PUBLISHING));
        }
    }
    if let Some(root) = spike_root {
        rules.push(CommandRule::allow("git *").within(root));
    }
    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trial stopped at its time limit still reports what it printed.
    #[cfg(unix)]
    #[tokio::test]
    async fn output_printed_before_a_stop_is_kept() {
        let mut child = shell("echo printed before the stop; echo on stderr >&2; sleep 30")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .process_group(0)
            .spawn()
            .unwrap();
        let pid = child.id();
        let stdout = collect(child.stdout.take());
        let stderr = collect(child.stderr.take());
        assert!(
            tokio::time::timeout(Duration::from_secs(2), child.wait())
                .await
                .is_err()
        );
        kill_group(pid);
        assert!(collected(stdout).await.contains("printed before the stop"));
        assert!(collected(stderr).await.contains("on stderr"));
    }

    #[test]
    fn a_trial_must_pass_its_reserved_budget() {
        use super::passes_budget;
        assert!(passes_budget("--repo x --budget 2.5 --verbose", 2.5));
        assert!(passes_budget("--budget=3", 3.0));
        assert!(!passes_budget("--budget 25", 2.5));
        assert!(!passes_budget("--repo x", 2.0));
    }

    #[test]
    fn git_and_gh_only_read_and_cargo_doesnt_publish() {
        let spikes = std::env::temp_dir().join("forge-state/deliberations");
        let guard = |spike_root: Option<&Path>| CommandGuard {
            rules: command_rules(spike_root),
            ..CommandGuard::default()
        };
        let runs = |guard: &CommandGuard, command: &str, directory: &Path| {
            guard.check(command, ShellDialect::Posix, directory).is_ok()
        };
        let workspace = std::env::temp_dir().join("forge-crate");
        let agent = guard(None);
        for command in [
            "git status --short",
            "git --no-pager log --oneline -10",
            "git diff HEAD~1 -- src",
            "git -C ../arko show HEAD:README.md",
            "git branch",
            "git branch --show-current",
            "git remote -v",
            "git config --get user.name",
            "gh pr view 12 --comments",
            "ls /usr/share/doc/git",
            "grep -rn 'git push' src",
            "cat .gitignore",
            "cargo build --all-targets",
            "cargo test publish_tests",
            "git branch -a",
            "git branch -vv",
            "git branch -a -vv",
            "git tag",
            "git stash list --stat",
            "git reflog",
            "git reflog show HEAD -5",
            "git show-ref --heads",
            "git for-each-ref refs/heads",
            "git worktree list --porcelain",
            "git submodule status --recursive",
            "git check-ignore -v target",
            "git help log",
            "git grep -n TODO -- src",
            "gh --version",
            "gh pr checks 12",
            "gh pr status",
            "cargo run -- publish",
            "cargo test login",
            "cargo --locked test owner",
        ] {
            assert!(runs(&agent, command, &workspace), "{command}");
        }
        for command in [
            "git commit -am wip",
            "cd /repo && git push origin main",
            "git log --output=log.txt",
            "git branch -D old",
            "git -c alias.x='!sh' x",
            "git stash",
            "gh pr create --fill",
            "gh api repos/o/r",
            "cargo publish --dry-run",
            "(cargo login token)",
            "cargo +nightly yank --version 1.0.0",
            "cargo --locked publish",
            "cargo -Zunstable-options logout",
            "cargo owner --add someone",
            "git branch new-feature",
            "git branch -a -d old",
            "git tag v1.0",
            "git stash drop",
            "git reflog expire --all",
            "git worktree add ../other",
            "git submodule update --init",
            "git grep -O TODO",
            "git grep -n -Ovim TODO",
            "git grep -nOvim TODO",
            "git grep -inO'touch x' TODO",
            "git grep TODO --open-files-in-pager=vim",
            "gh pr checkout 12",
        ] {
            assert!(!runs(&agent, command, &workspace), "{command}");
        }
        let refusal = |command: &str| {
            agent
                .check(command, ShellDialect::Posix, &workspace)
                .unwrap_err()
                .message
        };
        assert!(
            refusal("git log --output=log.txt").contains("read the output from the command"),
            "{}",
            refusal("git log --output=log.txt")
        );
        assert!(refusal("git grep -O TODO").contains("runs a program"));
        // A spike may use git fully in its scratch crate, and only there.
        let spike = guard(Some(&spikes));
        let scratch = spikes.join("D1/spike-1");
        assert!(runs(&spike, "git init && git commit -m x", &scratch));
        assert!(runs(&spike, "git rebase -i HEAD~2", &scratch));
        assert!(!runs(&spike, "git commit -m x", &workspace));
        assert!(!runs(&agent, "git init", &scratch));
        assert!(!runs(&spike, "git log --output=x", &scratch));
    }
}
