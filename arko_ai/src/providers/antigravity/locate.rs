//! Finds the Antigravity CLI, checks its version, and checks its sign-in.
//!
//! The sign-in check runs `agy acp` in a new profile and an empty folder,
//! both temporary, and goes as far as `session/new`, which needs the sign-in
//! but no model call. The new profile keeps the user's sign-in, plan and
//! models, and nothing of the user's own profile or project loads.

use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::process::NoConsoleWindow;
use crate::process_tree::ProcessTree;

use super::acp::{
    AcpClient, AcpError, AcpOptions, NO_AUTO_UPDATE, PermissionDecision, PermissionPolicy,
    PermissionRequest, config_values,
};
use super::{AvailableModel, MINIMUM_VERSION, SignIn, SignInProblem};

/// How long `agy --version` may take.
const VERSION_TIMEOUT: Duration = Duration::from_secs(30);

/// The names `agy` is installed under.
#[cfg(windows)]
const PROGRAM_NAMES: [&str; 2] = ["agy.exe", "agy.cmd"];
#[cfg(not(windows))]
const PROGRAM_NAMES: [&str; 1] = ["agy"];

/// The `agy` executable, from `PATH` or else `~/.local/bin`, where its
/// installer puts it.
pub fn find_agy() -> Option<PathBuf> {
    let path = std::env::var_os("PATH");
    let home = std::env::home_dir();
    find_in(path.as_deref(), home.as_deref(), &is_executable)
}

fn find_in(
    path: Option<&OsStr>,
    home: Option<&Path>,
    is_executable: &dyn Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let installed = home.map(|home| home.join(".local").join("bin"));
    path.into_iter()
        .flat_map(|path| std::env::split_paths(path))
        .filter(|directory| !directory.as_os_str().is_empty())
        .chain(installed)
        .flat_map(|directory| PROGRAM_NAMES.map(|name| directory.join(name)))
        .find(|candidate| is_executable(candidate.as_path()))
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// The version `agy --version` prints, such as `1.5.14`.
pub async fn cli_version(cli: &Path) -> Result<String, SignInProblem> {
    version_of(&Launch::new(cli)).await
}

/// Whether version `found` is `minimum` or later. Each is compared by its
/// numbers (`1.5.14`), and a suffix such as `-beta` is left out.
pub fn version_at_least(found: &str, minimum: &str) -> bool {
    match (version_numbers(found), version_numbers(minimum)) {
        (Some(found), Some(minimum)) => found >= minimum,
        _ => false,
    }
}

fn version_numbers(version: &str) -> Option<Vec<u64>> {
    let version = version.trim();
    let version = version.strip_prefix('v').unwrap_or(version);
    let numbers = version.split(['-', '+']).next()?;
    numbers.split('.').map(|part| part.parse().ok()).collect()
}

/// The first word of `--version`'s output that looks like a version.
fn parse_version(output: &str) -> Option<&str> {
    output
        .split_whitespace()
        .map(|word| word.strip_prefix('v').unwrap_or(word))
        .find(|word| word.starts_with(|c: char| c.is_ascii_digit()) && word.contains('.'))
}

/// Checks that the CLI (by default the one [`find_agy`] finds) is installed,
/// new enough and signed in, and lists the models the sign-in offers with
/// their thought levels.
pub async fn check_sign_in(cli: Option<&Path>) -> Result<SignIn, SignInProblem> {
    let program = match cli {
        Some(cli) => cli.to_path_buf(),
        None => tokio::task::spawn_blocking(find_agy)
            .await
            .ok()
            .flatten()
            .ok_or(SignInProblem::Missing)?,
    };
    check(Launch::new(&program)).await
}

async fn check(launch: Launch) -> Result<SignIn, SignInProblem> {
    let version = version_of(&launch).await?;
    if !version_at_least(&version, MINIMUM_VERSION) {
        return Err(SignInProblem::Outdated { found: version });
    }
    let folder_error = |error: io::Error| {
        SignInProblem::Failed(format!("Couldn't make a temporary folder. {error}"))
    };
    let scratch = Scratch::create("check").await.map_err(folder_error)?;
    let profile = scratch.folder("profile").await.map_err(folder_error)?;
    let work = scratch.folder("work").await.map_err(folder_error)?;
    let mut options = AcpOptions::new(launch.program.clone(), profile, work.clone());
    options.prefix_args = launch.prefix_args.clone();
    options.env = launch.env.clone();
    // The default agent asks before its own tools, and none should run.
    let refuse: Arc<dyn PermissionPolicy> =
        Arc::new(|_: &PermissionRequest| PermissionDecision::Cancel);
    let client = AcpClient::start(options, refuse).await.map_err(problem)?;
    let session = client.new_session(&work).await.map_err(problem)?;
    let families = session
        .config_options
        .iter()
        .find(|option| option.id == "model")
        .map(|option| option.options.clone())
        .unwrap_or_default();
    let mut models = Vec::with_capacity(families.len());
    // The thought levels on offer depend on the model, so each model is
    // chosen in turn. It changes only this session.
    for family in families {
        let settings = client
            .set_config_option(&session.session_id, "model", &family.value)
            .await
            .map_err(problem)?;
        models.push(AvailableModel {
            id: family.value,
            name: family.name,
            thought_levels: config_values(&settings, "thought_level")
                .into_iter()
                .map(str::to_string)
                .collect(),
        });
    }
    // The check is done, so a failure to close changes nothing.
    let _ = client.close().await;
    drop(scratch);
    Ok(SignIn {
        cli_path: launch.program,
        version,
        tier: session.tier,
        models,
    })
}

/// The sign-in problem an ACP error shows.
fn problem(error: AcpError) -> SignInProblem {
    match &error {
        AcpError::Spawn { source, .. } if source.kind() == io::ErrorKind::NotFound => {
            SignInProblem::Missing
        }
        // How agy 1.5.14 refuses a session without a sign-in.
        AcpError::Rpc { method, error: rpc }
            if method == "session/new"
                && rpc.code == -32000
                && rpc
                    .message
                    .to_ascii_lowercase()
                    .contains("authentication required") =>
        {
            SignInProblem::SignedOut
        }
        _ => SignInProblem::Failed(error.to_string()),
    }
}

async fn version_of(launch: &Launch) -> Result<String, SignInProblem> {
    let output = launch
        .output(&["--version"], None, VERSION_TIMEOUT)
        .await
        .map_err(|error| match error {
            RunError::NotFound => SignInProblem::Missing,
            RunError::Failed(message) => SignInProblem::Failed(message),
        })?;
    if !output.status.success() {
        return Err(SignInProblem::Failed(output.failure("agy --version")));
    }
    parse_version(&output.stdout)
        .map(str::to_string)
        .ok_or_else(|| {
            SignInProblem::Failed(format!(
                "`agy --version` printed no version (\"{}\").",
                output.stdout.trim()
            ))
        })
}

/// How to run a plain `agy` command.
#[derive(Debug, Clone)]
pub(super) struct Launch {
    pub program: PathBuf,
    /// Arguments before agy's own, such as a script run by an interpreter.
    pub prefix_args: Vec<String>,
    /// Variables set on top of the inherited environment, before
    /// [`NO_AUTO_UPDATE`].
    pub env: Vec<(String, String)>,
}

/// What a command printed, and how it ended.
pub(super) struct Output {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    /// Says that `command` failed, with the end of what it wrote to stderr.
    pub fn failure(&self, command: &str) -> String {
        let stderr = self.stderr.trim();
        let tail = stderr
            .char_indices()
            .rev()
            .nth(2000)
            .map_or(stderr, |(start, _)| &stderr[start..]);
        if tail.is_empty() {
            format!("`{command}` ended with {}.", self.status)
        } else {
            format!("`{command}` ended with {}. {tail}", self.status)
        }
    }
}

pub(super) enum RunError {
    NotFound,
    Failed(String),
}

impl Launch {
    pub fn new(program: &Path) -> Self {
        Self {
            program: program.to_path_buf(),
            prefix_args: Vec::new(),
            env: Vec::new(),
        }
    }

    /// Runs agy with `args` in `cwd` (by default the current folder) and
    /// collects its output. Once agy exits, what it left running is killed,
    /// such as the language server some commands start. A run that takes
    /// longer than `timeout` is killed too.
    pub async fn output(
        &self,
        args: &[&str],
        cwd: Option<&Path>,
        timeout: Duration,
    ) -> Result<Output, RunError> {
        let mut command = Command::new(&self.program);
        command
            .args(&self.prefix_args)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .no_console_window();
        for (name, value) in &self.env {
            command.env(name, value);
        }
        command.env(NO_AUTO_UPDATE.0, NO_AUTO_UPDATE.1);
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        let shown = || {
            std::iter::once(self.program.display().to_string())
                .chain(args.iter().map(|arg| arg.to_string()))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let (mut child, mut tree) = ProcessTree::spawn(&mut command).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                RunError::NotFound
            } else {
                RunError::Failed(format!("Couldn't start `{}`. {error}", shown()))
            }
        })?;
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let mut stderr = child.stderr.take().expect("stderr is piped");
        let run = async {
            // The pipes close once the tree is killed, so the reads end even
            // if a program agy started holds them.
            let reading = async {
                let (mut out, mut err) = (Vec::new(), Vec::new());
                let _ = tokio::join!(stdout.read_to_end(&mut out), stderr.read_to_end(&mut err));
                (out, err)
            };
            let (status, (out, err)) = tokio::join!(tree.wait(&mut child), reading);
            io::Result::Ok(Output {
                status: status?,
                stdout: String::from_utf8_lossy(&out).into_owned(),
                stderr: String::from_utf8_lossy(&err).into_owned(),
            })
        };
        match tokio::time::timeout(timeout, run).await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(error)) => Err(RunError::Failed(format!("`{}` failed. {error}", shown()))),
            // Dropping the tree and the child kills them.
            Err(_) => Err(RunError::Failed(format!(
                "`{}` didn't finish within {} seconds.",
                shown(),
                timeout.as_secs()
            ))),
        }
    }
}

/// A private temporary folder, removed when dropped.
pub(super) struct Scratch {
    path: PathBuf,
}

impl Scratch {
    /// Makes a new folder in the system's temporary folder, readable only by
    /// the user on Unix.
    pub async fn create(purpose: &str) -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!("agy-{purpose}-{}", uuid::Uuid::new_v4()));
        private_folder(&path).await?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Makes a private folder `name` inside.
    pub async fn folder(&self, name: &str) -> io::Result<PathBuf> {
        let path = self.path.join(name);
        private_folder(&path).await?;
        Ok(path)
    }
}

async fn private_folder(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let mut builder = tokio::fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(path).await
    }
    #[cfg(not(unix))]
    {
        tokio::fs::create_dir(path).await
    }
}

impl Drop for Scratch {
    /// Removes the folder on a thread of its own, as a profile can hold many
    /// files.
    fn drop(&mut self) {
        let path = std::mem::take(&mut self.path);
        std::thread::spawn(move || {
            if let Err(error) = std::fs::remove_dir_all(&path) {
                log::debug!("Couldn't remove {}: {error}", path.display());
            }
        });
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::acp::tests::{Fake, python, timed};
    use super::*;

    /// A launch of the fake agy, which writes down what it gets.
    pub(crate) fn fake_launch(fake: &Fake) -> Launch {
        Launch {
            program: python(),
            prefix_args: fake.prefix_args(),
            env: fake.env(),
        }
    }

    /// A made-up home folder on this system.
    fn home() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\Users\ada")
        } else {
            PathBuf::from("/home/ada")
        }
    }

    fn folder(name: &str) -> PathBuf {
        home().join(name)
    }

    fn find_with(path: &[PathBuf], files: &[PathBuf]) -> Option<PathBuf> {
        let path = std::env::join_paths(path).unwrap();
        let home = home();
        let is_executable = |candidate: &Path| files.iter().any(|file| file == candidate);
        find_in(Some(path.as_os_str()), Some(home.as_path()), &is_executable)
    }

    #[test]
    fn agy_is_found_on_the_path_first() {
        let name = PROGRAM_NAMES[0];
        let on_path = folder("tools").join(name);
        let installed = folder(".local").join("bin").join(name);
        assert_eq!(
            find_with(
                &[folder("bin"), folder("tools")],
                &[on_path.clone(), installed.clone()]
            ),
            Some(on_path)
        );
        assert_eq!(
            find_with(&[folder("bin")], &[installed.clone()]),
            Some(installed)
        );
        assert_eq!(find_with(&[folder("bin")], &[]), None);
        assert_eq!(find_in(None, None, &|_| true), None);
    }

    #[test]
    fn versions_compare_by_their_numbers() {
        assert!(version_at_least("1.5.14", MINIMUM_VERSION));
        assert!(version_at_least("1.5.15", "1.5.14"));
        assert!(version_at_least("1.6", "1.5.14"));
        assert!(version_at_least("2.0.0", "1.5.14"));
        assert!(version_at_least("v1.5.14", "1.5.14"));
        assert!(version_at_least("1.5.14-beta.1", "1.5.14"));
        assert!(!version_at_least("1.5.13", "1.5.14"));
        assert!(!version_at_least("1.5", "1.5.14"));
        assert!(!version_at_least("1.4.99", "1.5.14"));
        assert!(!version_at_least("unknown", "1.5.14"));
        assert_eq!(parse_version("1.5.13\n"), Some("1.5.13"));
        assert_eq!(parse_version("agy version 1.5.14 (linux)"), Some("1.5.14"));
        assert_eq!(parse_version(""), None);
    }

    #[tokio::test]
    async fn the_version_comes_from_dash_dash_version() {
        let fake = Fake::new();
        assert_eq!(
            timed(version_of(&fake_launch(&fake))).await.unwrap(),
            "1.5.14"
        );
        let launch = &fake.records()[0];
        assert_eq!(launch["args"], serde_json::json!(["--version"]));
        assert_eq!(launch["no_update"], "true");
    }

    #[tokio::test]
    async fn a_missing_cli_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("agy");
        assert_eq!(
            timed(check_sign_in(Some(missing.as_path())))
                .await
                .unwrap_err(),
            SignInProblem::Missing
        );
        assert_eq!(
            timed(cli_version(&missing)).await.unwrap_err(),
            SignInProblem::Missing
        );
    }

    #[tokio::test]
    async fn an_old_cli_is_outdated() {
        let fake = Fake::new();
        let mut launch = fake_launch(&fake);
        launch
            .env
            .push(("FAKE_AGY_VERSION".into(), "1.5.13".into()));
        assert_eq!(
            timed(check(launch)).await.unwrap_err(),
            SignInProblem::Outdated {
                found: "1.5.13".into()
            }
        );
        // It stops before `agy acp`.
        assert_eq!(fake.records().len(), 1);
    }

    #[tokio::test]
    async fn the_check_lists_the_models_and_their_levels() {
        let fake = Fake::new();
        let sign_in = timed(check(fake_launch(&fake))).await.unwrap();
        assert_eq!(sign_in.cli_path, python());
        assert_eq!(sign_in.version, "1.5.14");
        assert_eq!(sign_in.tier.as_deref(), Some("Google AI Pro"));
        let levels = |levels: &[&str]| -> Vec<String> {
            levels.iter().map(|level| level.to_string()).collect()
        };
        assert_eq!(
            sign_in.models,
            [
                AvailableModel {
                    id: "gemini-4-neon".into(),
                    name: "Gemini 4 Neon".into(),
                    thought_levels: levels(&["low", "medium", "high", "xhigh"]),
                },
                AvailableModel {
                    id: "gemini-3.8-flash".into(),
                    name: "Gemini 3.8 Flash".into(),
                    thought_levels: levels(&["low", "medium", "high"]),
                },
                AvailableModel {
                    id: "gpt-oss-120b-medium".into(),
                    name: "GPT-OSS 120B (Medium)".into(),
                    thought_levels: Vec::new(),
                },
            ]
        );

        let records = fake.records();
        let acp = records
            .iter()
            .find(|record| record["args"][0] == "acp")
            .expect("agy acp ran");
        assert_eq!(acp["no_update"], "true");
        let arguments = acp["args"].as_array().unwrap();
        assert!(!arguments.iter().any(|argument| argument == "--agent"));
        // A new profile beside an empty working folder, both removed after.
        let profile = PathBuf::from(arguments.last().unwrap().as_str().unwrap());
        assert!(profile.ends_with("profile"));
        let scratch = profile.parent().unwrap().to_path_buf();
        assert!(scratch.starts_with(std::env::temp_dir()));
        assert!(acp["cwd"].as_str().unwrap().ends_with("work"));
        for _ in 0..200 {
            if !scratch.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(!scratch.exists());
        // No model call was made.
        assert!(
            !fake
                .received()
                .iter()
                .any(|message| message["method"] == "session/prompt")
        );
    }

    #[tokio::test]
    async fn a_signed_out_cli_is_signed_out() {
        let fake = Fake::new();
        let mut launch = fake_launch(&fake);
        launch.env.push(("FAKE_AGY_SIGNED_OUT".into(), "1".into()));
        assert_eq!(
            timed(check(launch)).await.unwrap_err(),
            SignInProblem::SignedOut
        );
    }

    #[tokio::test]
    async fn another_error_fails_the_check() {
        let fake = Fake::new();
        let mut launch = fake_launch(&fake);
        launch.env.push(("FAKE_AGY_BROKEN".into(), "1".into()));
        let SignInProblem::Failed(message) = timed(check(launch)).await.unwrap_err() else {
            panic!("another problem");
        };
        assert!(message.contains("Internal error"), "{message}");
    }
}
