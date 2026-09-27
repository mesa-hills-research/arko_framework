//! Opt-in interactive background jobs with explicit and inferred input notices.
//!
//! Program declarations travel over a dedicated stream. Prompt rules report
//! possible input only: silence and descriptor writability are never detectors.
mod detection;
mod io;
mod runner;
mod tools;

pub use detection::PromptRule;
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
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct TerminalConfig {
    pub directory: PathBuf,
    pub shell: PathBuf,
    /// Arguments before the command string (default: sh -c or cmd /D /S /C).
    pub shell_args: Vec<String>,
    pub max_jobs: usize,
    pub max_running: usize,
    pub output_buffer_bytes: usize,
    pub max_log_bytes: u64,
    pub excerpt_bytes: usize,
    pub prompt_quiet_ms: u64,
    pub prompt_rules: Vec<PromptRule>,
    pub default_timeout: Option<Duration>,
}
impl TerminalConfig {
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
            max_jobs: 64,
            max_running: 4,
            output_buffer_bytes: 65_536,
            max_log_bytes: 4 * 1024 * 1024,
            excerpt_bytes: 2_000,
            prompt_quiet_ms: 150,
            prompt_rules: detection::default_rules(),
            default_timeout: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StartOptions {
    pub terminal: bool,
    pub detect_prompts: bool,
    pub timeout: Option<Duration>,
}
impl Default for StartOptions {
    fn default() -> Self {
        Self {
            terminal: true,
            detect_prompts: true,
            timeout: None,
        }
    }
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
    /// An application declared a request; this is not OS-level wait detection.
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputWindow {
    pub start: u64,
    pub end: u64,
    pub available_end: u64,
    /// True when the requested beginning is no longer in the output buffer.
    pub gap: bool,
    /// UTF-8 display of this byte range, replacing invalid/incomplete sequences.
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
    /// Both modes retain combined arrival-order output; PTY mode adds terminal semantics.
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
                text: String::from_utf8_lossy(&bytes).into_owned(),
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

/// Share one manager across tools. It owns jobs; dropping it cancels running jobs.
/// Its directory is exclusive. On process restart, previous running jobs become
/// interrupted records; their PIDs are never trusted, reattached, or rerun.
pub struct TerminalManager {
    config: TerminalConfig,
    detector: Arc<Detector>,
    jobs: Mutex<HashMap<Uuid, Arc<Job>>>,
    runtimes: Mutex<HashMap<Uuid, RuntimeBinding>>,
    closing: std::sync::atomic::AtomicBool,
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
        if config.max_jobs == 0
            || config.max_running == 0
            || config.output_buffer_bytes < 1024
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
            config,
            detector,
            jobs: Mutex::new(HashMap::new()),
            runtimes: Mutex::new(HashMap::new()),
            closing: false.into(),
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

    /// Tools remain opt-in: registering them does not add their names to an agent's allowlist.
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
                    let data = job.data.lock().unwrap();
                    if data.last_event.is_some()
                        && let Ok(latest) =
                            data.snapshot(&manager.config, None, manager.config.excerpt_bytes)
                    {
                        let content = format!(
                            "Terminal update:\n{}",
                            serde_json::to_string(&latest)
                                .expect("terminal snapshot serialization")
                        );
                        permit.send(UserCommand::InjectContext { content });
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
        let mut jobs = self.jobs.lock().unwrap();
        if self.closing.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TerminalError::Invalid(
                "terminal manager is shut down".into(),
            ));
        }
        if jobs.len() >= self.config.max_jobs
            || jobs
                .values()
                .filter(|job| job.data.lock().unwrap().record.state == JobState::Running)
                .count()
                >= self.config.max_running
        {
            return Err(TerminalError::Invalid(
                "terminal job limit reached; finish jobs or forget completed records".into(),
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
        let process = match io::spawn(
            &self.config,
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

    /// Wait for a revision change, input request, or completion; never more than 30s.
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

fn persist(config: &TerminalConfig, record: &Record) -> Result<()> {
    let directory = config.directory.join(record.job_id.to_string());
    let temporary = directory.join("job.json.tmp");
    std::fs::write(&temporary, serde_json::to_vec(record)?)?;
    std::fs::rename(temporary, directory.join("job.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests;
