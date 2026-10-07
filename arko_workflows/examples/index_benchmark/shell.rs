//! The agents' terminal: one `shell` tool that runs a command in the
//! read-only checkout and returns its output. Every arm gets the same tool;
//! the arms differ only in the programs on its `PATH` and its environment.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arko_agents::terminal::{CommandGuard, ShellDialect};
use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::AsyncReadExt;

pub const SHELL_TOOL: &str = "shell";

/// How every arm's shell runs commands.
#[derive(Debug, Clone)]
pub struct ShellSettings {
    /// Default and maximum time a command may run.
    pub timeout: Duration,
    /// Output returned to the agent: the first and last half of this many
    /// bytes, with a note of what was left out.
    pub max_output_bytes: usize,
    /// Calls after which the tool answers with a request to finish instead
    /// of running the command.
    pub max_calls: usize,
}

/// Called once when an agent goes on calling after the limit: the job
/// stops the run.
pub type LimitCallback = Arc<dyn Fn(String) + Send + Sync>;

/// The environment a job's commands run in.
#[derive(Debug, Clone)]
pub struct ShellEnvironment {
    /// Variables set on top of the (scrubbed) process environment.
    pub vars: Vec<(String, OsString)>,
    /// Directories put first on `PATH` (the arm's tools).
    pub path_prefix: Vec<PathBuf>,
}

pub struct ShellTool {
    settings: ShellSettings,
    environment: ShellEnvironment,
    guard: CommandGuard,
    calls: AtomicUsize,
    on_limit: Option<LimitCallback>,
}

impl std::fmt::Debug for ShellTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellTool")
            .field("settings", &self.settings)
            .field("calls", &self.calls)
            .finish()
    }
}

impl ShellTool {
    pub fn new(
        settings: ShellSettings,
        environment: ShellEnvironment,
        on_limit: Option<LimitCallback>,
    ) -> Self {
        Self {
            settings,
            environment,
            guard: CommandGuard::default(),
            calls: AtomicUsize::new(0),
            on_limit,
        }
    }

    #[cfg(test)]
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// The `PATH` for a command.
    fn path_variable(&self) -> OsString {
        let inherited = std::env::var_os("PATH").unwrap_or_default();
        let mut parts: Vec<PathBuf> = self.environment.path_prefix.clone();
        parts.extend(std::env::split_paths(&inherited));
        std::env::join_paths(parts).unwrap_or(inherited)
    }

    async fn run(&self, command: &str, cwd: &Path, timeout: Duration) -> Result<String, ToolError> {
        #[cfg(windows)]
        let mut process = {
            let mut process = tokio::process::Command::new("cmd.exe");
            process.args(["/D", "/S", "/C", command]);
            process
        };
        #[cfg(not(windows))]
        let mut process = {
            let mut process = tokio::process::Command::new("bash");
            process.args(["-c", command]);
            process.process_group(0);
            process
        };
        process
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (name, _) in std::env::vars_os() {
            if is_sensitive(&name.to_string_lossy()) {
                process.env_remove(name);
            }
        }
        process.env("PATH", self.path_variable());
        for (name, value) in [
            ("PAGER", "cat"),
            ("GIT_PAGER", "cat"),
            ("TERM", "dumb"),
            ("NO_COLOR", "1"),
        ] {
            process.env(name, value);
        }
        for (name, value) in &self.environment.vars {
            process.env(name, value);
        }
        let mut child = process.spawn().map_err(|source| ToolError::Io {
            tool_name: SHELL_TOOL.to_string(),
            source,
        })?;
        let pid = child.id();
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut stderr = child.stderr.take().expect("piped stderr");
        let cap = self.settings.max_output_bytes;
        let read_out = tokio::spawn(async move { read_capped(&mut stdout, cap).await });
        let read_err = tokio::spawn(async move { read_capped(&mut stderr, cap).await });
        let status = match tokio::time::timeout(timeout, child.wait()).await {
            Ok(status) => Some(status.map_err(|source| ToolError::Io {
                tool_name: SHELL_TOOL.to_string(),
                source,
            })?),
            Err(_) => {
                kill_group(pid).await;
                let _ = child.kill().await;
                None
            }
        };
        // Output still arriving from a background process the command left
        // running is not waited for long.
        let collect = |handle: tokio::task::JoinHandle<CappedOutput>| async move {
            match tokio::time::timeout(Duration::from_secs(5), handle).await {
                Ok(Ok(output)) => output,
                _ => CappedOutput::default(),
            }
        };
        let (out, err) = (collect(read_out).await, collect(read_err).await);
        let mut text = String::new();
        match status {
            Some(status) => match status.code() {
                Some(code) => text.push_str(&format!("exit code: {code}\n")),
                None => text.push_str("exit code: none (killed by a signal)\n"),
            },
            None => text.push_str(&format!(
                "the command did not finish within {} seconds and was stopped\n",
                timeout.as_secs()
            )),
        }
        let output = out.render(cap);
        let errors = err.render(cap / 4);
        if !output.is_empty() {
            text.push_str(&output);
            if !output.ends_with('\n') {
                text.push('\n');
            }
        }
        if !errors.is_empty() {
            text.push_str("stderr:\n");
            text.push_str(&errors);
        }
        Ok(text)
    }
}

/// Output read up to a cap: the first `cap / 2` bytes, the last `cap / 2`
/// bytes, and how many bytes there were in all.
#[derive(Debug, Default)]
struct CappedOutput {
    head: Vec<u8>,
    tail: std::collections::VecDeque<u8>,
    total: usize,
}

impl CappedOutput {
    fn render(&self, cap: usize) -> String {
        let shown = self.head.len() + self.tail.len();
        let head = String::from_utf8_lossy(&self.head);
        let tail: Vec<u8> = self.tail.iter().copied().collect();
        let tail = String::from_utf8_lossy(&tail);
        if self.total <= shown || self.total <= cap {
            return format!("{head}{tail}");
        }
        format!(
            "{head}\n[... {} bytes of output omitted; narrow the command (for example with head, a more specific pattern, or a line range) to see them ...]\n{tail}",
            self.total - shown
        )
    }
}

async fn read_capped(reader: &mut (impl AsyncReadExt + Unpin), cap: usize) -> CappedOutput {
    let half = cap / 2;
    let mut output = CappedOutput::default();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let n = match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        output.total += n;
        for &byte in &buffer[..n] {
            if output.head.len() < half {
                output.head.push(byte);
            } else {
                output.tail.push_back(byte);
                if output.tail.len() > half {
                    output.tail.pop_front();
                }
            }
        }
    }
    output
}

/// Kills the process group a timed-out command leads, so pipelines and
/// their children stop too.
async fn kill_group(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        let _ = tokio::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
    #[cfg(not(unix))]
    let _ = pid;
}

/// Variables that hold credentials: never passed to the agents' commands.
pub fn is_sensitive(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.ends_with("_API_KEY")
        || name.ends_with("_ACCESS_TOKEN")
        || name.ends_with("_REFRESH_TOKEN")
        || name.ends_with("_AUTH_TOKEN")
        || name.ends_with("_TOKEN")
        || name.ends_with("_PASSWORD")
        || name.ends_with("_SECRET")
        || name == "CODEX_HOME"
        || [
            "OPENAI_",
            "ANTHROPIC_",
            "GOOGLE_",
            "GEMINI_",
            "VOYAGE_",
            "AWS_",
            "AZURE_",
            "GITHUB_",
            "GH_",
            "CLOUDFLARE_",
            "TELEGRAM_",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

#[async_trait]
impl Tool for ShellTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: SHELL_TOOL.to_string(),
            description: format!(
                "Run one shell command (bash) in the repository's read-only checkout and return its exit code and output. Output beyond {} bytes is cut in the middle; narrow the command when that happens. Commands stop after {} seconds.",
                self.settings.max_output_bytes,
                self.settings.timeout.as_secs()
            ),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "command".to_string(),
                    description: "The command, for example `rg -n 'fn parse' src | head -50` or `sed -n '120,180p' src/lib.rs`.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "timeout_seconds".to_string(),
                    description: format!(
                        "Optional: stop the command after this many seconds (at most {}).",
                        self.settings.timeout.as_secs()
                    ),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    fn recovery_mode(&self) -> arko_agents::ToolRecovery {
        arko_agents::ToolRecovery::Rerun
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let command = params
            .get("command")
            .and_then(Value::as_str)
            .filter(|command| !command.trim().is_empty())
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "command".to_string(),
            })?;
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call > self.settings.max_calls {
            if call > self.settings.max_calls + 10
                && let Some(on_limit) = &self.on_limit
            {
                on_limit(format!(
                    "the agent kept calling the shell after its limit of {} calls",
                    self.settings.max_calls
                ));
            }
            return Ok(format!(
                "Not run: this task's limit of {} commands is reached. Answer now with `complete_task`, from what you have found.",
                self.settings.max_calls
            ));
        }
        let timeout = match params.get("timeout_seconds") {
            Some(value) => {
                let seconds = value
                    .as_u64()
                    .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
                    .ok_or_else(|| ToolError::ParameterValidation {
                        param_name: "timeout_seconds".to_string(),
                        message: "expected a number of seconds".to_string(),
                    })?;
                Duration::from_secs(seconds.clamp(1, self.settings.timeout.as_secs().max(1)))
            }
            None => self.settings.timeout,
        };
        let dialect = if cfg!(windows) {
            ShellDialect::Cmd
        } else {
            ShellDialect::Posix
        };
        self.guard
            .check(command, dialect, context.current_working_directory)
            .map_err(|refusal| ToolError::ExecutionFailed(refusal.message))?;
        self.run(command, context.current_working_directory, timeout)
            .await
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn tool(max_calls: usize, max_output_bytes: usize, prefix: Vec<PathBuf>) -> ShellTool {
        ShellTool::new(
            ShellSettings {
                timeout: Duration::from_secs(5),
                max_output_bytes,
                max_calls,
            },
            ShellEnvironment {
                vars: vec![("BENCH_TEST_VAR".into(), "visible".into())],
                path_prefix: prefix,
            },
            None,
        )
    }

    #[tokio::test]
    async fn runs_commands_with_the_arm_environment() {
        let dir = crate::test_dir("shell-env");
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let script = bin.join("arm_only_tool");
        std::fs::write(&script, "#!/bin/sh\necho arm tool ran\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let shell = tool(10, 4096, vec![bin.clone()]);
        let output = shell
            .run(
                "arm_only_tool; echo $BENCH_TEST_VAR; echo oops >&2; exit 3",
                &dir,
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert!(output.starts_with("exit code: 3\n"), "{output}");
        assert!(output.contains("arm tool ran\nvisible\n"), "{output}");
        assert!(output.contains("stderr:\noops"), "{output}");
        let without = tool(10, 4096, vec![]);
        let output = without
            .run(
                "command -v arm_only_tool || echo missing",
                &dir,
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert!(output.contains("missing"), "{output}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn caps_output_and_stops_slow_commands() {
        let dir = crate::test_dir("shell-cap");
        let shell = tool(10, 1000, vec![]);
        let output = shell
            .run("seq 1 100000", &dir, Duration::from_secs(5))
            .await
            .unwrap();
        assert!(output.contains("bytes of output omitted"), "{output}");
        assert!(output.len() < 1400, "{}", output.len());
        assert!(output.contains("\n1\n2\n"));
        assert!(output.trim_end().ends_with("100000"));
        let started = std::time::Instant::now();
        let output = shell
            .run("sleep 30 | cat", &dir, Duration::from_secs(1))
            .await
            .unwrap();
        assert!(
            output.contains("did not finish within 1 seconds"),
            "{output}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn asks_for_an_answer_after_the_call_limit_then_stops_the_run() {
        let dir = crate::test_dir("shell-limit");
        let tripped = Arc::new(parking_lot::Mutex::new(None::<String>));
        let seen = tripped.clone();
        let shell = ShellTool::new(
            ShellSettings {
                timeout: Duration::from_secs(5),
                max_output_bytes: 4096,
                max_calls: 2,
            },
            ShellEnvironment {
                vars: vec![],
                path_prefix: vec![],
            },
            Some(Arc::new(move |reason| *seen.lock() = Some(reason))),
        );
        let context = TaskContext::new(uuid::Uuid::new_v4(), "solver", "", &dir);
        let call = |command: &str| {
            HashMap::from([("command".to_string(), Value::String(command.to_string()))])
        };
        for _ in 0..2 {
            let output = shell.execute(&context, call("echo ran")).await.unwrap();
            assert!(output.contains("ran"), "{output}");
        }
        let output = shell.execute(&context, call("echo ran")).await.unwrap();
        assert!(output.starts_with("Not run"), "{output}");
        assert!(tripped.lock().is_none());
        for _ in 0..10 {
            shell.execute(&context, call("echo ran")).await.unwrap();
        }
        assert!(tripped.lock().as_deref().unwrap().contains("limit of 2"));
        assert_eq!(shell.calls(), 13);
        // Destructive commands outside the checkout are refused.
        let fresh = tool(10, 4096, vec![]);
        assert!(fresh.execute(&context, call("rm -rf /usr")).await.is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn credentials_are_not_passed_on() {
        assert!(is_sensitive("OPENAI_API_KEY"));
        assert!(is_sensitive("VOYAGE_API_KEY"));
        assert!(is_sensitive("CODEX_HOME"));
        assert!(is_sensitive("github_token"));
        assert!(!is_sensitive("PATH"));
        assert!(!is_sensitive("ARKO_INDEX_STATE"));
    }
}
