//! Opt-in interactive background jobs with explicit and inferred input notices.
//!
//! Program declarations travel over a dedicated stream. Prompt rules report
//! possible input only. Silence and descriptor writability are never detectors.
mod detection;
mod guard;
mod io;
mod runner;
mod tools;

pub use detection::PromptRule;
pub use guard::{CommandGuard, CommandRefusal, CommandRule, RuleAction, ShellDialect};
pub use tools::TerminalTool;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, watch};
use uuid::Uuid;

use crate::{TaskContext, ToolRegistry, UserCommand};
use detection::Detector;

type Result<T> = std::result::Result<T, TerminalError>;

#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    #[error("{0}")]
    Invalid(String),
    #[error("terminal job not found or owned by another task")]
    NotOwned,
    #[error("terminal view changed; read status before sending input")]
    StaleView,
    #[error("terminal job has finished")]
    Finished,
    /// The [`CommandGuard`] refused the command.
    #[error(transparent)]
    Refused(#[from] CommandRefusal),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct TerminalConfig {
    pub directory: PathBuf,
    pub shell: PathBuf,
    /// Arguments before the command string. Defaults to `-c` for `sh`, or
    /// `/D /S /C` for `cmd`.
    pub shell_args: Vec<String>,
    pub output_buffer_bytes: usize,
    pub max_log_bytes: u64,
    pub excerpt_bytes: usize,
    pub prompt_quiet_ms: u64,
    pub prompt_rules: Vec<PromptRule>,
    pub default_timeout: Option<Duration>,
    /// PTY jobs get [`QUIET_TERMINAL_ENVIRONMENT`] on top of the host's
    /// environment (default `true`), so programs that see a terminal print
    /// plain UTF-8 text without colours or pagers. Full-screen programs
    /// (editors, `top`, `less`) refuse to run or degrade under `TERM=dumb`.
    /// Set `false` to give PTY jobs the host's environment unchanged. Pipe
    /// jobs always get it unchanged.
    pub quiet_terminal_environment: bool,
    /// Refuses commands that would delete, move, overwrite, or reformat a
    /// protected path (the filesystem root, a drive root, the home directory
    /// or its parents, top-level and system directories, mounted volumes,
    /// devices) or a path outside the job's working directory. On by
    /// default. See [`CommandGuard`] for the opt-ins.
    pub command_guard: CommandGuard,
}

/// The UTF-8 locale PTY jobs get, set to `C.UTF-8`, except on macOS, which may
/// not have it (a shell would then print locale warnings into the output),
/// where it is `en_US.UTF-8`.
pub const QUIET_TERMINAL_LOCALE: &str = if cfg!(target_os = "macos") {
    "en_US.UTF-8"
} else {
    "C.UTF-8"
};

/// The variables PTY jobs get when
/// [`TerminalConfig::quiet_terminal_environment`] is on (the default), on
/// every platform, with no colour (`TERM=dumb`, `NO_COLOR=1`), no pager
/// (`PAGER=cat`, `GIT_PAGER=cat`), and a UTF-8 locale
/// ([`QUIET_TERMINAL_LOCALE`] as `LANG`, `LC_CTYPE` and `LC_ALL`).
pub const QUIET_TERMINAL_ENVIRONMENT: [(&str, &str); 7] = [
    ("TERM", "dumb"),
    ("NO_COLOR", "1"),
    ("PAGER", "cat"),
    ("GIT_PAGER", "cat"),
    ("LANG", QUIET_TERMINAL_LOCALE),
    ("LC_CTYPE", QUIET_TERMINAL_LOCALE),
    ("LC_ALL", QUIET_TERMINAL_LOCALE),
];

impl TerminalConfig {
    /// The variables a job gets on top of the host's environment.
    pub(super) fn job_environment(
        &self,
        terminal: bool,
    ) -> &'static [(&'static str, &'static str)] {
        if terminal && self.quiet_terminal_environment {
            &QUIET_TERMINAL_ENVIRONMENT
        } else {
            &[]
        }
    }

    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            shell: if cfg!(windows) { "cmd.exe" } else { "/bin/sh" }.into(),
            shell_args: if cfg!(windows) {
                vec!["/D", "/S", "/C"]
            } else {
                vec!["-c"]
            }
            .into_iter()
            .map(str::to_string)
            .collect(),
            output_buffer_bytes: 65_536,
            max_log_bytes: 4 * 1024 * 1024,
            excerpt_bytes: 2_000,
            prompt_quiet_ms: 150,
            prompt_rules: detection::default_rules(),
            default_timeout: None,
            quiet_terminal_environment: true,
            command_guard: CommandGuard::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StartOptions {
    pub terminal: bool,
    pub detect_prompts: bool,
    pub timeout: Option<Duration>,
    /// The job is held for its host (default `false`). It sends no terminal
    /// updates and isn't listed until [`TerminalManager::release`] moves it
    /// to the background. For a host that waits for the job itself (a
    /// command run to completion) and may hand it to the agent later.
    pub held: bool,
    /// The arguments before the command for this job, instead of
    /// [`TerminalConfig::shell_args`] (default `None`).
    pub shell_args: Option<Vec<String>>,
    /// What [`TerminalManager::background_jobs`] shows as the command, for
    /// a host that wraps the command it was given. Defaults to `None`, which
    /// shows the command.
    pub label: Option<String>,
}
impl Default for StartOptions {
    fn default() -> Self {
        Self {
            terminal: true,
            detect_prompts: true,
            timeout: None,
            held: false,
            shell_args: None,
            label: None,
        }
    }
}

/// A job running in the background (not held), as
/// [`TerminalManager::background_jobs`] lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundJob {
    pub owner: Uuid,
    pub job_id: Uuid,
    /// The command, or its [`StartOptions::label`].
    pub command: String,
    pub started: std::time::Instant,
}

/// A job that finished in the background (not held), as
/// [`TerminalManager::subscribe_finished`] reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobFinished {
    pub owner: Uuid,
    pub job_id: Uuid,
    pub state: JobState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
    Interrupted,
}
impl JobState {
    pub fn is_terminal(self) -> bool {
        self != Self::Running
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputSource {
    /// An application declared a request. This is not OS-level wait detection.
    Program { request_id: String },
    /// A configured rule matched output. It is always an inference.
    PromptRule { rule: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputNotice {
    pub source: InputSource,
    pub prompt: String,
    pub prompt_truncated: bool,
}

/// Whether PTY output shown to the agent loses its escape sequences: on
/// Windows, where ConPTY adds its own (cursor moves, window titles, colour
/// resets) whatever the program prints. The log and byte offsets keep the
/// raw output.
const STRIPS_ESCAPES: bool = cfg!(windows);

/// The display text of an output window: its bytes as UTF-8 (invalid or
/// split sequences replaced), without escape sequences when `strip` is set
/// (cursor-forward moves become the spaces they stand for).
fn window_text(bytes: &[u8], strip: bool) -> String {
    let text = String::from_utf8_lossy(bytes);
    if strip {
        detection::strip_escapes(&text)
    } else {
        text.into_owned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputWindow {
    pub start: u64,
    pub end: u64,
    pub available_end: u64,
    /// True when the requested beginning is no longer in the output buffer.
    pub gap: bool,
    /// UTF-8 display of this byte range, replacing invalid/incomplete
    /// sequences. On Windows, PTY output is shown without escape sequences.
    /// `start`, `end` and the log still count the raw bytes.
    pub text: String,
    /// Exact bytes when the display range contains invalid/incomplete UTF-8.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_hex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalSnapshot {
    pub job_id: Uuid,
    pub revision: u64,
    pub state: JobState,
    pub input: Option<InputNotice>,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
    pub output: OutputWindow,
    pub log_path: PathBuf,
    pub log_truncated: bool,
    /// Both modes retain combined arrival-order output. PTY mode adds terminal semantics.
    pub terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    job_id: Uuid,
    owner: Uuid,
    revision: u64,
    state: JobState,
    input: Option<InputNotice>,
    exit_code: Option<i32>,
    error: Option<String>,
    terminal: bool,
    output_end: u64,
    log_truncated: bool,
    /// See [`StartOptions::held`]. False in records saved before it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    held: bool,
}

#[derive(Debug)]
struct JobData {
    record: Record,
    output: VecDeque<u8>,
    last_event: Option<TerminalSnapshot>,
}
impl JobData {
    fn snapshot(
        &self,
        config: &TerminalConfig,
        after: Option<u64>,
        limit: usize,
    ) -> Result<TerminalSnapshot> {
        let end = self.record.output_end;
        let base = end - self.output.len() as u64;
        let requested = after.unwrap_or(end.saturating_sub(limit as u64));
        if requested > end {
            return Err(TerminalError::Invalid(
                "output cursor is beyond this job's output".into(),
            ));
        }
        let start = requested.max(base);
        let take = limit.min((end - start) as usize);
        let bytes = self
            .output
            .iter()
            .skip((start - base) as usize)
            .take(take)
            .copied()
            .collect::<Vec<_>>();
        Ok(TerminalSnapshot {
            job_id: self.record.job_id,
            revision: self.record.revision,
            state: self.record.state,
            input: self.record.input.clone(),
            exit_code: self.record.exit_code,
            error: self.record.error.clone(),
            output: OutputWindow {
                start,
                end: start + take as u64,
                available_end: end,
                gap: requested < base,
                text: window_text(&bytes, STRIPS_ESCAPES && self.record.terminal),
                raw_hex: std::str::from_utf8(&bytes)
                    .is_err()
                    .then(|| bytes.iter().map(|byte| format!("{byte:02x}")).collect()),
            },
            log_path: config
                .directory
                .join(self.record.job_id.to_string())
                .join("output.log"),
            log_truncated: self.record.log_truncated,
            terminal: self.record.terminal,
        })
    }
}

pub(super) enum Control {
    Input {
        revision: u64,
        text: String,
        response: oneshot::Sender<Result<()>>,
    },
}
struct Job {
    data: Arc<Mutex<JobData>>,
    /// The directory the job started in, for checking input typed into a
    /// shell it runs. `None` for jobs restored from an earlier process.
    working_directory: Option<PathBuf>,
    /// The command as [`BackgroundJob`] shows it, and when it started
    /// (empty for jobs restored from an earlier process, which never run).
    command: String,
    started: std::time::Instant,
    controls: mpsc::Sender<Control>,
    cancel: watch::Sender<bool>,
    changes: watch::Receiver<u64>,
    done: watch::Receiver<bool>,
}

fn completed_job(data: Arc<Mutex<JobData>>) -> Arc<Job> {
    let (controls, _) = mpsc::channel(1);
    let (cancel, _) = watch::channel(false);
    let (_, changes) = watch::channel(data.lock().unwrap().record.revision);
    let (_, done) = watch::channel(true);
    Arc::new(Job {
        data,
        working_directory: None,
        command: String::new(),
        started: std::time::Instant::now(),
        controls,
        cancel,
        changes,
        done,
    })
}

#[derive(Clone)]
struct RuntimeBinding {
    sender: mpsc::WeakSender<UserCommand>,
    pending: Arc<Mutex<HashSet<Uuid>>>,
}

/// Share one manager across tools. It owns jobs. Dropping it cancels running jobs.
/// Its directory is exclusive. On process restart, previous running jobs become
/// interrupted records. Their PIDs are never trusted, reattached, or rerun.
pub struct TerminalManager {
    config: TerminalConfig,
    detector: Arc<Detector>,
    jobs: Mutex<HashMap<Uuid, Arc<Job>>>,
    runtimes: Mutex<HashMap<Uuid, RuntimeBinding>>,
    closing: std::sync::atomic::AtomicBool,
    /// The command guard's `allow_outside_workspace`, which can change
    /// while jobs run.
    allow_outside_workspace: std::sync::atomic::AtomicBool,
    /// Where jobs that finish in the background are reported.
    finished: tokio::sync::broadcast::Sender<JobFinished>,
    _directory_lock: Arc<std::fs::File>,
}
impl std::fmt::Debug for TerminalManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalManager")
            .field("directory", &self.config.directory)
            .finish_non_exhaustive()
    }
}

impl TerminalManager {
    pub fn new(mut config: TerminalConfig) -> Result<Arc<Self>> {
        if config.output_buffer_bytes < 1024
            || config.excerpt_bytes == 0
            || config.excerpt_bytes > config.output_buffer_bytes
            || config.max_log_bytes == 0
            || config.prompt_quiet_ms == 0
            || config.prompt_quiet_ms > 60_000
            || config.default_timeout.is_some_and(|value| {
                value.is_zero() || std::time::Instant::now().checked_add(value).is_none()
            })
        {
            return Err(TerminalError::Invalid("invalid terminal limits".into()));
        }
        let detector = Arc::new(
            Detector::new(&config.prompt_rules)
                .map_err(|e| TerminalError::Invalid(e.to_string()))?,
        );
        std::fs::create_dir_all(&config.directory)?;
        config.directory = config.directory.canonicalize()?;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(config.directory.join(".lock"))?;
        lock.try_lock().map_err(|error| {
            TerminalError::Invalid(format!("terminal directory is already in use: {error}"))
        })?;
        let manager = Arc::new(Self {
            allow_outside_workspace: config.command_guard.allow_outside_workspace.into(),
            config,
            detector,
            jobs: Mutex::new(HashMap::new()),
            runtimes: Mutex::new(HashMap::new()),
            closing: false.into(),
            finished: tokio::sync::broadcast::channel(64).0,
            _directory_lock: Arc::new(lock),
        });
        for entry in std::fs::read_dir(&manager.config.directory)? {
            let path = entry?.path().join("job.json");
            if !path.is_file() {
                continue;
            }
            let mut record: Record = serde_json::from_slice(&std::fs::read(&path)?)?;
            if path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                != Some(&record.job_id.to_string())
            {
                return Err(TerminalError::Invalid(
                    "job record ID does not match its directory".into(),
                ));
            }
            if record.state == JobState::Running {
                record.state = JobState::Interrupted;
                record.input = None;
                record.revision += 1;
                record.error = Some(
                    "host stopped before recording completion; inspect artifacts before retrying"
                        .into(),
                );
            }
            persist(&manager.config, &record)?;
            let data = Arc::new(Mutex::new(JobData {
                record,
                output: VecDeque::new(),
                last_event: None,
            }));
            let id = data.lock().unwrap().record.job_id;
            manager.jobs.lock().unwrap().insert(id, completed_job(data));
        }
        Ok(manager)
    }

    /// Sets the command guard's
    /// [`allow_outside_workspace`](CommandGuard::allow_outside_workspace)
    /// while the manager runs. It applies from the next `terminal_start`
    /// command or `terminal_input` text. Jobs already running keep going.
    pub fn set_allow_outside_workspace(&self, allow: bool) {
        self.allow_outside_workspace
            .store(allow, std::sync::atomic::Ordering::Release);
    }

    /// Whether the command guard allows destructive commands on paths
    /// outside the working directory, as last set.
    pub fn allows_outside_workspace(&self) -> bool {
        self.allow_outside_workspace
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Checks a command run in `working_directory` against the command
    /// guard, with outside access as last set.
    fn check_command(&self, command: &str, working_directory: &Path) -> Result<()> {
        let guard = CommandGuard {
            allow_outside_workspace: self.allows_outside_workspace(),
            ..self.config.command_guard.clone()
        };
        guard.check(
            command,
            ShellDialect::from_shell(&self.config.shell),
            working_directory,
        )?;
        Ok(())
    }

    /// Registers the tools. Add their names to an agent's allowlist to offer them.
    pub fn register_tools(self: &Arc<Self>, registry: &mut ToolRegistry) {
        for tool in TerminalTool::all(self.clone()) {
            registry.register(Arc::new(tool));
        }
    }

    pub fn bind_runtime(self: &Arc<Self>, owner: Uuid, sender: mpsc::WeakSender<UserCommand>) {
        {
            let mut bindings = self.runtimes.lock().unwrap();
            if bindings
                .get(&owner)
                .and_then(|old| old.sender.upgrade())
                .zip(sender.upgrade())
                .is_some_and(|(old, new)| old.same_channel(&new))
            {
                return;
            }
            bindings.retain(|_, binding| binding.sender.strong_count() > 0);
            bindings.insert(
                owner,
                RuntimeBinding {
                    sender,
                    pending: Default::default(),
                },
            );
        }
        let replay = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .filter_map(|job| {
                let data = job.data.lock().unwrap();
                (data.record.owner == owner)
                    .then(|| data.last_event.clone())
                    .flatten()
            })
            .collect::<Vec<_>>();
        for snapshot in replay {
            self.deliver(owner, snapshot);
        }
    }

    fn deliver(self: &Arc<Self>, owner: Uuid, snapshot: TerminalSnapshot) {
        let Some(binding) = self.runtimes.lock().unwrap().get(&owner).cloned() else {
            return;
        };
        let Some(sender) = binding.sender.upgrade() else {
            return;
        };
        let id = snapshot.job_id;
        if !binding.pending.lock().unwrap().insert(id) {
            return;
        }
        let manager = Arc::downgrade(self);
        // One pending delivery per job. Reserve channel capacity before reading
        // its latest event, so backpressure coalesces superseded questions.
        tokio::spawn(async move {
            let permit = sender.reserve_owned().await;
            let mut pending = binding.pending.lock().unwrap();
            if let (Ok(permit), Some(manager)) = (permit, manager.upgrade()) {
                let current_binding = manager.runtimes.lock().unwrap().get(&owner).cloned();
                let current = current_binding
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(&current.pending, &binding.pending));
                if current && let Ok(job) = manager.owned(owner, id) {
                    let mut data = job.data.lock().unwrap();
                    // A held job's events wait for its release.
                    if data.last_event.is_some()
                        && !data.record.held
                        && let Ok(latest) =
                            data.snapshot(&manager.config, None, manager.config.excerpt_bytes)
                    {
                        let content = format!(
                            "Terminal update:\n{}",
                            serde_json::to_string(&latest)
                                .expect("terminal snapshot serialization")
                        );
                        permit.send(UserCommand::InjectContext { content });
                        // Sent: a runtime the owner binds later (its next
                        // turn) doesn't get it again.
                        data.last_event = None;
                    }
                }
            }
            pending.remove(&id);
        });
    }

    fn owned(&self, owner: Uuid, id: Uuid) -> Result<Arc<Job>> {
        let job = self
            .jobs
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or(TerminalError::NotOwned)?;
        if job.data.lock().unwrap().record.owner != owner {
            return Err(TerminalError::NotOwned);
        }
        Ok(job)
    }

    pub fn start(
        self: &Arc<Self>,
        context: &TaskContext<'_>,
        command: &str,
        options: StartOptions,
    ) -> Result<TerminalSnapshot> {
        if command.is_empty()
            || command.len() > 65_536
            || options.timeout.is_some_and(|timeout| {
                timeout.is_zero() || std::time::Instant::now().checked_add(timeout).is_none()
            })
        {
            return Err(TerminalError::Invalid(
                "empty/oversized command or zero timeout".into(),
            ));
        }
        self.check_command(command, context.current_working_directory)?;
        let mut jobs = self.jobs.lock().unwrap();
        if self.closing.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TerminalError::Invalid(
                "terminal manager is shut down".into(),
            ));
        }
        let id = Uuid::new_v4();
        std::fs::create_dir(self.config.directory.join(id.to_string()))?;
        let record = Record {
            job_id: id,
            owner: context.task_id,
            revision: 0,
            state: JobState::Running,
            input: None,
            exit_code: None,
            error: None,
            terminal: options.terminal,
            output_end: 0,
            log_truncated: false,
            held: options.held,
        };
        let data = Arc::new(Mutex::new(JobData {
            record,
            output: VecDeque::new(),
            last_event: None,
        }));
        let mut log = std::fs::File::create(
            self.config
                .directory
                .join(id.to_string())
                .join("output.log"),
        )?;
        let own_args;
        let config = match &options.shell_args {
            Some(args) => {
                own_args = TerminalConfig {
                    shell_args: args.clone(),
                    ..self.config.clone()
                };
                &own_args
            }
            None => &self.config,
        };
        let process = match io::spawn(
            config,
            command,
            context.current_working_directory,
            options.terminal,
            &self.config.directory.join(id.to_string()),
        ) {
            Ok(process) => process,
            Err(error) => {
                {
                    let mut data = data.lock().unwrap();
                    data.record.state = JobState::Failed;
                    data.record.error = Some(error.to_string());
                    persist(&self.config, &data.record)?;
                }
                jobs.insert(id, completed_job(data));
                return Err(TerminalError::Invalid(format!(
                    "could not start job {id}: {error}"
                )));
            }
        };
        // ProcessGroup and Child guards clean up if persistence fails here.
        persist(&self.config, &data.lock().unwrap().record)?;
        let (controls, receiver) = mpsc::channel(16);
        let (cancel, cancellation) = watch::channel(false);
        let (changed, changes) = watch::channel(0);
        let (completed, done) = watch::channel(false);
        let snapshot =
            data.lock()
                .unwrap()
                .snapshot(&self.config, None, self.config.excerpt_bytes)?;
        jobs.insert(
            id,
            Arc::new(Job {
                data: data.clone(),
                working_directory: Some(context.current_working_directory.to_path_buf()),
                command: options.label.clone().unwrap_or_else(|| command.to_string()),
                started: std::time::Instant::now(),
                controls,
                cancel,
                changes,
                done,
            }),
        );
        let weak = Arc::downgrade(self);
        let config = self.config.clone();
        let detector = self.detector.clone();
        let directory_lock = self._directory_lock.clone();
        tokio::spawn(async move {
            runner::run(
                weak,
                config,
                detector,
                data,
                process,
                &mut log,
                options,
                receiver,
                cancellation,
                changed,
            )
            .await;
            drop(directory_lock);
            completed.send_replace(true);
        });
        Ok(snapshot)
    }

    pub fn status(&self, owner: Uuid, id: Uuid, after: Option<u64>) -> Result<TerminalSnapshot> {
        let job = self.owned(owner, id)?;
        job.data
            .lock()
            .unwrap()
            .snapshot(&self.config, after, self.config.excerpt_bytes)
    }

    pub async fn input(&self, owner: Uuid, id: Uuid, revision: u64, text: String) -> Result<()> {
        if text.is_empty() || text.len() > 4096 {
            return Err(TerminalError::Invalid(
                "input must contain 1–4096 bytes".into(),
            ));
        }
        let job = self.owned(owner, id)?;
        if job.data.lock().unwrap().record.state.is_terminal() {
            return Err(TerminalError::Finished);
        }
        // Input may be a command typed into a shell the job runs.
        if let Some(directory) = &job.working_directory {
            self.check_command(&text, directory)?;
        }
        let (response, answer) = oneshot::channel();
        job.controls
            .send(Control::Input {
                revision,
                text,
                response,
            })
            .await
            .map_err(|_| TerminalError::Finished)?;
        answer.await.map_err(|_| TerminalError::Finished)?
    }

    pub fn cancel(&self, owner: Uuid, id: Uuid) -> Result<()> {
        self.owned(owner, id)?.cancel.send_replace(true);
        Ok(())
    }

    /// Moves a held job (see [`StartOptions::held`]) to the background. It
    /// is listed, and its terminal updates go to its owner from now on,
    /// starting with an event it already has. A job that has finished
    /// stays held ([`TerminalError::Finished`]), as its host has its result.
    pub fn release(self: &Arc<Self>, owner: Uuid, id: Uuid) -> Result<TerminalSnapshot> {
        let job = self.owned(owner, id)?;
        let (snapshot, event) = {
            // Under the jobs' lock, as `close` checks them: a job doesn't
            // move to the background while the host exits.
            let _jobs = self.jobs.lock().unwrap();
            let mut data = job.data.lock().unwrap();
            if data.record.state.is_terminal() {
                return Err(TerminalError::Finished);
            }
            if self.closing.load(std::sync::atomic::Ordering::Acquire) {
                return Err(TerminalError::Invalid(
                    "terminal manager is shut down".into(),
                ));
            }
            if data.record.held {
                let record = Record {
                    held: false,
                    ..data.record.clone()
                };
                persist(&self.config, &record)?;
                data.record.held = false;
            }
            (
                data.snapshot(&self.config, None, self.config.excerpt_bytes)?,
                data.last_event.clone(),
            )
        };
        if let Some(event) = event {
            self.deliver(owner, event);
        }
        Ok(snapshot)
    }

    /// The jobs that finish in the background (not held) from now on. A
    /// host can wake an idle agent for them.
    pub fn subscribe_finished(&self) -> tokio::sync::broadcast::Receiver<JobFinished> {
        self.finished.subscribe()
    }

    /// The jobs running in the background, of every owner, oldest first.
    /// These are what [`shutdown`](Self::shutdown) would stop, for a host to say so
    /// before it exits.
    pub fn background_jobs(&self) -> Vec<BackgroundJob> {
        background(&self.jobs.lock().unwrap())
    }

    /// Closes the manager for its host's exit, unless a job runs in the
    /// background that isn't one of `known` (those the host's user agreed
    /// to stop). In that case it stays open, and the jobs running in the background
    /// come back. Closed, it starts no job and moves none to the background
    /// ([`release`](Self::release)), so [`shutdown`](Self::shutdown) stops
    /// no background job the user didn't know of. The check and the closing
    /// are one step, so no job gets past it.
    pub fn close(&self, known: &[Uuid]) -> std::result::Result<(), Vec<BackgroundJob>> {
        let jobs = self.jobs.lock().unwrap();
        let running = background(&jobs);
        if running.iter().any(|job| !known.contains(&job.job_id)) {
            return Err(running);
        }
        self.closing
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Wait for a revision change, input request, or completion. Never more than 30s.
    /// The wait deadline does not terminate the process.
    pub async fn wait(
        &self,
        owner: Uuid,
        id: Uuid,
        revision: u64,
        duration: Duration,
    ) -> Result<TerminalSnapshot> {
        let job = self.owned(owner, id)?;
        let mut changes = job.changes.clone();
        changes.borrow_and_update();
        let should_wait = {
            let data = job.data.lock().unwrap();
            data.record.revision == revision && !data.record.state.is_terminal()
        };
        if should_wait {
            let _ = tokio::time::timeout(duration.min(Duration::from_secs(30)), changes.changed())
                .await;
        }
        self.status(owner, id, None)
    }

    pub fn forget(&self, owner: Uuid, id: Uuid) -> Result<()> {
        let job = self.owned(owner, id)?;
        if !job.data.lock().unwrap().record.state.is_terminal() {
            return Err(TerminalError::Invalid(
                "cancel or finish a job before forgetting it".into(),
            ));
        }
        // Keep artifacts; rename the manifest out of the active catalog.
        let directory = self.config.directory.join(id.to_string());
        std::fs::rename(directory.join("job.json"), directory.join("finished.json"))?;
        self.jobs.lock().unwrap().remove(&id);
        Ok(())
    }

    pub async fn shutdown(&self) {
        self.closing
            .store(true, std::sync::atomic::Ordering::Release);
        let jobs = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for job in &jobs {
            job.cancel.send_replace(true);
        }
        for job in jobs {
            let mut done = job.done.clone();
            let _ = tokio::time::timeout(Duration::from_secs(3), async {
                while !*done.borrow_and_update() {
                    if done.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await;
        }
    }
}
impl Drop for TerminalManager {
    fn drop(&mut self) {
        for job in self.jobs.get_mut().unwrap().values() {
            job.cancel.send_replace(true);
        }
    }
}

/// The jobs running in the background (not held) in `jobs`, oldest first.
fn background(jobs: &HashMap<Uuid, Arc<Job>>) -> Vec<BackgroundJob> {
    let mut running: Vec<BackgroundJob> = jobs
        .values()
        .filter_map(|job| {
            let data = job.data.lock().unwrap();
            (data.record.state == JobState::Running && !data.record.held).then(|| BackgroundJob {
                owner: data.record.owner,
                job_id: data.record.job_id,
                command: job.command.clone(),
                started: job.started,
            })
        })
        .collect();
    running.sort_by_key(|job| job.started);
    running
}

fn persist(config: &TerminalConfig, record: &Record) -> Result<()> {
    let directory = config.directory.join(record.job_id.to_string());
    let temporary = directory.join("job.json.tmp");
    std::fs::write(&temporary, serde_json::to_vec(record)?)?;
    std::fs::rename(temporary, directory.join("job.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests;
