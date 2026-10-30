//! Checkpoint and resume support for workflows.
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::workflow::result::AgentRunStatus;

pub mod images;
pub mod state;

/// What the event log (`events.jsonl` in the run directory) records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EventLog {
    /// Every agent event (the default), including each streamed fragment
    /// of response text and reasoning, one line each. A long response
    /// writes thousands of lines.
    #[default]
    Full,
    /// Every event except the streamed fragments
    /// (`AssistantResponseChunk`, `ReasoningChunk`, and output-based tool
    /// calling's `ToolParamName`, `ToolParamValueChunk`, `ToolParamEnd`), including
    /// state changes, tool calls and results, usage, errors, retries and
    /// the rest. The finished responses are in the task files
    /// (`agents/<task id>/<version>.json`). Streamed reasoning a provider
    /// doesn't return with the response is only in a `Full` log.
    LifecycleOnly,
}

impl EventLog {
    /// Whether `event` is recorded.
    fn records(self, event: &arko_agents::runtime::event::AgentEvent) -> bool {
        self == Self::Full || !is_fragment(event)
    }
}

/// A streamed fragment of a response.
fn is_fragment(event: &arko_agents::runtime::event::AgentEvent) -> bool {
    use arko_agents::runtime::event::AgentEvent;
    matches!(
        event,
        AgentEvent::AssistantResponseChunk(_)
            | AgentEvent::ReasoningChunk(_)
            | AgentEvent::ToolParamName { .. }
            | AgentEvent::ToolParamValueChunk { .. }
            | AgentEvent::ToolParamEnd { .. }
    )
}

/// An event written to disk at once: a state change, the end of a run, an
/// error or recovery report, or a handover (whose report the replaced
/// conversation no longer holds).
fn is_urgent(event: &arko_agents::runtime::event::AgentEvent) -> bool {
    use arko_agents::runtime::event::AgentEvent;
    matches!(
        event,
        AgentEvent::TaskStateChanged(_)
            | AgentEvent::TaskStreamEnd
            | AgentEvent::ErrorEncountered { .. }
            | AgentEvent::RetryScheduled { .. }
            | AgentEvent::UsageLimitReached { .. }
            | AgentEvent::RefusalReceived(_)
            | AgentEvent::OutputTokenLimitReached { .. }
            | AgentEvent::ConversationHandedOver { .. }
    )
}

/// The default [`PersistenceConfig::event_flush_interval`] of one second.
pub const DEFAULT_EVENT_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Which checkpoint files a run directory keeps. See
/// [`PersistenceConfig::retain_checkpoints`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CheckpointRetention {
    /// Every checkpoint and task version is kept (the default).
    #[default]
    KeepAll,
    /// When enabled, only the newest this many checkpoint files are kept. After
    /// each save, older ones are deleted (earlier processes' first, by
    /// modification time), then the task versions that neither a kept
    /// checkpoint nor a recorded result refers to.
    KeepLast(NonZeroUsize),
}

/// Configuration for workflow persistence.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Directory where checkpoint files are stored.
    pub checkpoint_dir: PathBuf,
    /// Automatically checkpoint after each agent completion.
    pub auto_checkpoint: bool,
    /// Which checkpoint files are kept. Keeps every one
    /// ([`CheckpointRetention::KeepAll`], the default), or the newest few,
    /// with the task versions they refer to (set with
    /// [`keep_last_checkpoints`](Self::keep_last_checkpoints)).
    pub retain_checkpoints: CheckpointRetention,
    /// What the event log records. Default [`EventLog::Full`].
    pub event_log: EventLog,
    /// Events are buffered and written at least this often. State changes,
    /// run ends and error reports are written at once, and so is the
    /// buffer before each checkpoint. Default
    /// [`DEFAULT_EVENT_FLUSH_INTERVAL`].
    pub event_flush_interval: Duration,
    /// Whether running agents are saved as they run (`true` by default). Each
    /// running task in `agents/<task id>/live.json`, rewritten at its safe
    /// points (before its tool calls run, after their results, after each
    /// model turn), with the workflow's state in `live-checkpoint.json`
    /// (rewritten when a resume would read something new from it).
    /// `latest-checkpoint` names the newest checkpoint, live or versioned.
    /// A workflow that stops (a crash, a reboot) then resumes its running
    /// agents where they were. See
    /// [`Workflow::interrupted_runs`](crate::Workflow::interrupted_runs)
    /// and [`Workflow::recover_agent`](crate::Workflow::recover_agent).
    /// Set with [`task_journal`](Self::task_journal).
    pub task_journal: bool,
    /// Whether the files the task journal rewrites as agents run (their
    /// live task files and the live checkpoint) are pretty-printed JSON.
    /// Default `false`, compact. Checkpoints and task versions are always
    /// pretty-printed. Set with [`pretty_json`](Self::pretty_json).
    pub pretty_json: bool,
    /// `keep_last_checkpoints(0)` was asked for. `WorkflowBuilder::build`
    /// reports it.
    zero_kept_checkpoints: bool,
}

/// Manages checkpointing and restoration of workflow state.
#[derive(Clone)]
pub struct PersistenceManager {
    config: PersistenceConfig,
    durable_store: Arc<crate::durable::DurableStore>,
    write_lock: Arc<tokio::sync::Mutex<()>>,
    events: Arc<EventWriter>,
    /// The run's images, saved once each. What it writes refers to them.
    images: Arc<images::ImageFiles>,
    /// The saved versions of the recorded results' tasks, by task ID,
    /// written by this manager, or loaded on resume. Changed only with the
    /// write lock held, together with the results.
    saved_tasks: Arc<parking_lot::Mutex<HashMap<uuid::Uuid, SavedTask>>>,
    /// Checkpoint files this manager wrote, oldest first (for retention).
    written: Arc<parking_lot::Mutex<VecDeque<PathBuf>>>,
    /// The task journal's state holds the running agent runs and their live
    /// files ([`PersistenceConfig::task_journal`]).
    live: Arc<parking_lot::Mutex<LiveState>>,
    /// Writes the live checkpoint, one write at a time.
    live_checkpoints: Arc<LiveCheckpoints>,
    /// Numbers the checkpoints as their states are built (with the write
    /// lock held). A higher number holds a newer state.
    builds: Arc<AtomicU64>,
    /// What `latest-checkpoint` names.
    latest: Arc<parking_lot::Mutex<Latest>>,
    /// Held while `latest-checkpoint` is written.
    pointer_writes: Arc<tokio::sync::Mutex<()>>,
    /// Makes task writes fail, as a full or read-only disk would.
    #[cfg(test)]
    pub(crate) fail_task_writes: Arc<AtomicBool>,
    /// Set by [`simulate_crash`](Self::simulate_crash), so nothing is written
    /// or deleted any more.
    #[cfg(test)]
    crashed: Arc<AtomicBool>,
    /// Where [`crash_at`](Self::crash_at) crashes.
    #[cfg(test)]
    crash_point: Arc<parking_lot::Mutex<Option<CrashPoint>>>,
}

/// A point of the finalization of an agent run a test can crash at. See
/// [`PersistenceManager::crash_at`].
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CrashPoint {
    /// The run's result is recorded (its task version written), and no
    /// checkpoint records it yet.
    ResultRecorded,
    /// The live checkpoint recording the run's end is written, and the
    /// versioned checkpoint after it isn't.
    LiveRunReleased,
}

#[cfg(test)]
impl PersistenceManager {
    /// Holds off every write (task versions, checkpoints) until the guard
    /// is dropped, as a long write would.
    pub(crate) async fn hold_writes(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.write_lock.clone().lock_owned().await
    }

    /// From now on nothing is written, renamed or deleted: the run
    /// directory stays as a process killed now would leave it.
    pub(crate) fn simulate_crash(&self) {
        self.crashed.store(true, Ordering::SeqCst);
    }

    /// The flag [`simulate_crash`](Self::simulate_crash) sets, for a hook
    /// to crash at a chosen point.
    pub(crate) fn crash_switch(&self) -> Arc<AtomicBool> {
        self.crashed.clone()
    }

    /// Crashes (see [`simulate_crash`](Self::simulate_crash)) as the next
    /// agent run to end reaches `point`.
    pub(crate) fn crash_at(&self, point: CrashPoint) {
        *self.crash_point.lock() = Some(point);
    }

    /// Crashes if a test asked to at `point`.
    fn reached(&self, point: CrashPoint) {
        if *self.crash_point.lock() == Some(point) {
            self.simulate_crash();
        }
    }
}

/// The live checkpoint's file in the run directory. See
/// [`PersistenceConfig::task_journal`].
pub const LIVE_CHECKPOINT_FILE: &str = "live-checkpoint.json";

/// A running task's file in its task directory.
const LIVE_TASK_FILE: &str = "live.json";

/// The error a checkpoint records for an agent run that was running when
/// it was written.
pub(crate) const RUNNING_AT_CHECKPOINT: &str =
    "The workflow stopped while this agent run was running. Workflow::recover_agent continues it.";

/// The live file of task `id`, relative to the run directory.
fn live_task_file(id: uuid::Uuid) -> String {
    format!("{TASKS_DIR}/{id}/{LIVE_TASK_FILE}")
}

/// The task journal's state.
#[derive(Default)]
struct LiveState {
    /// The running agent runs the journal saves, by task ID.
    runs: HashMap<uuid::Uuid, LiveRun>,
    /// Orders `runs` by start.
    started: u64,
    /// The live task files, on disk or to be written, with the lock that
    /// writing or deleting each takes.
    files: HashMap<uuid::Uuid, Arc<tokio::sync::Mutex<()>>>,
    /// The runs whose live file the live checkpoint on disk refers to (or
    /// the one being written will). Their files are kept until a live
    /// checkpoint no longer does.
    referenced: HashSet<uuid::Uuid>,
}

/// The checkpoint `latest-checkpoint` names, as this manager last wrote it.
#[derive(Default)]
struct Latest {
    /// Its file name.
    name: Option<String>,
    /// Its build number (see [`PersistenceManager::builds`]).
    build: u64,
    /// The agent runs it records as running.
    running: HashSet<uuid::Uuid>,
    /// When it is the live checkpoint, the fingerprint of the state it
    /// holds (see [`Built::fingerprint`]).
    live: Option<u64>,
    /// Its directory entry is flushed to disk.
    synced: bool,
}

/// A checkpoint's state, as [`PersistenceManager::build_state`] builds it.
struct Built {
    state: state::WorkflowState,
    /// Its build number (see [`PersistenceManager::builds`]).
    build: u64,
    /// The agent runs it records as running.
    running: HashSet<uuid::Uuid>,
    /// The agent runs whose live files it refers to.
    live_files: HashSet<uuid::Uuid>,
    /// Equal for two states that differ only where a resume can tell the
    /// newer from the running tasks' live files, including the cost ledger, as far as
    /// the running runs' usage goes (see `usage_charged`), and the time.
    fingerprint: u64,
}

/// A running agent run, as the task journal saves it.
struct LiveRun {
    agent_name: String,
    /// When it started, in the order of the journal's runs.
    started: u64,
    /// How many of its task's `usage_history` entries the budget was
    /// charged for, from the reports the workflow processed. A checkpoint
    /// records it with the ledger, so a resume charges the rest.
    usage_charged: usize,
    /// The latest snapshot of its task.
    task: Option<Arc<arko_agents::Task>>,
    /// `task` is not in the live file yet.
    pending: bool,
    /// The live file holds a snapshot of the run.
    written: bool,
    /// The live file's directory entry is flushed to disk, as a durable
    /// save needs.
    synced: bool,
    /// Why the latest snapshot could not be written.
    unsaved: Option<String>,
}

/// Writes the live checkpoint for its requests, one write at a time: each
/// write saves the state as it is when it starts, so requests made while
/// another write runs are answered by the next one.
#[derive(Default)]
struct LiveCheckpoints {
    requested: AtomicU64,
    /// The requests answered by a live checkpoint on disk.
    written: AtomicU64,
    /// The requests answered by a live checkpoint whose directory entry is
    /// flushed to disk too.
    synced: AtomicU64,
    lock: tokio::sync::Mutex<()>,
}

/// A recorded result's task, as checkpoints refer to it.
#[derive(Debug, Clone, Default)]
struct SavedTask {
    /// Its last saved version, a file relative to the run directory.
    file: Option<String>,
    /// Why its latest state (the recorded result's task) isn't saved. The
    /// write failed. Checkpoints refer to `file` until a later one saves it.
    unsaved: Option<String>,
}

impl PersistenceConfig {
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            auto_checkpoint: true,
            retain_checkpoints: CheckpointRetention::KeepAll,
            event_log: EventLog::default(),
            event_flush_interval: DEFAULT_EVENT_FLUSH_INTERVAL,
            task_journal: true,
            pretty_json: false,
            zero_kept_checkpoints: false,
        }
    }

    /// Whether a checkpoint is saved after each finalized agent result
    /// (default `true`). The task journal's live checkpoint
    /// ([`task_journal`](Self::task_journal)) is written either way.
    pub fn auto_checkpoint(mut self, enabled: bool) -> Self {
        self.auto_checkpoint = enabled;
        self
    }

    /// Whether running agents are saved as they run, so that a workflow
    /// that stops resumes them where they were (default `true`). See
    /// [`PersistenceConfig::task_journal`]. Off, a run is saved only once it
    /// ends, and one running when the workflow stops starts again.
    pub fn task_journal(mut self, enabled: bool) -> Self {
        self.task_journal = enabled;
        self
    }

    /// Opt in to pretty-printing the files the task journal rewrites as agents
    /// run (default compact JSON, which is smaller and quicker to write).
    /// See [`PersistenceConfig::pretty_json`].
    pub fn pretty_json(mut self, enabled: bool) -> Self {
        self.pretty_json = enabled;
        self
    }

    /// Opt in to keeping only the newest `count` checkpoints
    /// ([`CheckpointRetention::KeepLast`]), deleting older checkpoint files
    /// after each save, with the task versions no kept checkpoint refers
    /// to. By default every checkpoint and task version is kept. A `count`
    /// of 0 makes `WorkflowBuilder::build` fail.
    pub fn keep_last_checkpoints(mut self, count: usize) -> Self {
        match NonZeroUsize::new(count) {
            Some(count) => {
                self.retain_checkpoints = CheckpointRetention::KeepLast(count);
                self.zero_kept_checkpoints = false;
            }
            None => self.zero_kept_checkpoints = true,
        }
        self
    }

    /// The settings `WorkflowBuilder::build` rejects.
    pub(crate) fn validate(&self) -> Result<(), WorkflowError> {
        if self.zero_kept_checkpoints {
            return Err(WorkflowError::BuilderInvalid(
                "keep_last_checkpoints needs at least 1. Every checkpoint is kept by default"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// What the event log records. See [`EventLog`].
    pub fn event_log(mut self, event_log: EventLog) -> Self {
        self.event_log = event_log;
        self
    }

    /// How often buffered events are written at least (default one
    /// second).
    pub fn event_flush_interval(mut self, interval: Duration) -> Self {
        self.event_flush_interval = interval;
        self
    }

    /// Construct an OS-appropriate per-user run directory without requiring
    /// the workflow developer to hard-code a platform path.
    pub fn user_data(workflow_name: &str) -> Result<Self, WorkflowError> {
        let base = if cfg!(target_os = "windows") {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join("Library").join("Application Support"))
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .map(|path| path.join(".local/share"))
                })
        }
        .ok_or_else(|| {
            WorkflowError::Persistence("could not determine the user data directory".to_string())
        })?;
        Ok(Self::new(
            base.join("arko")
                .join("workflows")
                .join(workflow_name)
                .join(uuid::Uuid::new_v4().to_string()),
        ))
    }
}

/// The directory of task versions, relative to the run directory.
const TASKS_DIR: &str = "agents";

/// A version of `task`: its file, relative to the run directory
/// (`agents/<task id>/<content hash>.json`), and its content. The same
/// content has the same file, so a task that didn't change isn't written
/// again.
fn task_version(
    task: &arko_agents::Task,
    images: &Arc<images::ImageFiles>,
) -> Result<(String, Vec<u8>), WorkflowError> {
    // Through a JSON value, whose maps keep their keys in order: the same
    // task gives the same bytes (a `HashMap` field's order varies). Its
    // images are saved apart and named.
    let content = serde_json::to_vec_pretty(&images.with(|| serde_json::to_value(task))?)?;
    let hash = crate::digest::sha256_hex(&content);
    Ok((
        format!("{TASKS_DIR}/{}/{}.json", task.id, &hash[..32]),
        content,
    ))
}

impl PersistenceManager {
    /// Open an existing run identity or create one for a new run directory.
    pub fn resolve_run_id(config: &PersistenceConfig) -> Result<uuid::Uuid, WorkflowError> {
        std::fs::create_dir_all(&config.checkpoint_dir)?;
        let path = config.checkpoint_dir.join("workflow-run-id");
        match std::fs::read_to_string(&path) {
            Ok(value) => uuid::Uuid::parse_str(value.trim())
                .map_err(|error| WorkflowError::Persistence(error.to_string())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let run_id = uuid::Uuid::new_v4();
                let mut options = std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                use std::io::Write;
                let mut file = options.open(&path)?;
                file.write_all(run_id.to_string().as_bytes())?;
                file.sync_data()?;
                Ok(run_id)
            }
            Err(error) => Err(error.into()),
        }
    }

    /// A manager persisting run `workflow_run_id` with `config`. Fails, as
    /// `WorkflowBuilder::build` does, on a setting the config rejects
    /// (`keep_last_checkpoints(0)`).
    pub fn new(
        config: PersistenceConfig,
        workflow_run_id: uuid::Uuid,
    ) -> Result<Self, WorkflowError> {
        config.validate()?;
        let durable_store = Arc::new(crate::durable::DurableStore::new(
            workflow_run_id,
            config.checkpoint_dir.clone(),
        ));
        let images = images::ImageFiles::in_run_dir(&config.checkpoint_dir);
        let events = Arc::new(EventWriter::new(
            config.checkpoint_dir.join("events.jsonl"),
            workflow_run_id,
            config.event_log,
            config.event_flush_interval,
            images.clone(),
        ));
        // The live files the live checkpoint on disk (an earlier process's)
        // refers to are kept until this manager's live checkpoint replaces
        // it; what an earlier process left behind otherwise goes.
        let mut live = LiveState::default();
        for id in sweep_run_directory(&config.checkpoint_dir) {
            live.files.insert(id, Arc::default());
            live.referenced.insert(id);
        }
        Ok(Self {
            config,
            durable_store,
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
            events,
            images,
            saved_tasks: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            written: Arc::new(parking_lot::Mutex::new(VecDeque::new())),
            live: Arc::new(parking_lot::Mutex::new(live)),
            live_checkpoints: Arc::default(),
            builds: Arc::default(),
            latest: Arc::default(),
            pointer_writes: Arc::default(),
            #[cfg(test)]
            fail_task_writes: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            crashed: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            crash_point: Arc::default(),
        })
    }

    /// The configuration it persists with.
    pub fn config(&self) -> &PersistenceConfig {
        &self.config
    }

    pub fn durable_store(&self) -> Arc<crate::durable::DurableStore> {
        self.durable_store.clone()
    }

    /// The run's image files. What it saves refers to its images by name.
    pub fn images(&self) -> &Arc<images::ImageFiles> {
        &self.images
    }

    /// Records that the recorded results' tasks were loaded from these task
    /// versions (task ID, file relative to the run directory). A task
    /// loaded from its live file is saved as a version by the next
    /// checkpoint (that file changes once the run is recovered). Until one
    /// is, checkpoints refer to the live file, which is kept.
    pub(crate) fn mark_tasks_saved(&self, tasks: impl IntoIterator<Item = (uuid::Uuid, String)>) {
        let mut saved = self.saved_tasks.lock();
        for (id, file) in tasks {
            let unsaved = live_file_task(&file).is_some().then(|| {
                "it was restored from its live file and isn't saved as a version yet".to_string()
            });
            saved.insert(
                id,
                SavedTask {
                    file: Some(file),
                    unsaved,
                },
            );
        }
    }

    /// Records that the recorded result of task `id` changed since it was
    /// saved, such as a run put back as it was before a report on request
    /// (with the model calls and the context the report run took), so the
    /// next checkpoint saves it as a new version. Until one does,
    /// checkpoints refer to the version last saved.
    pub(crate) fn mark_task_changed(&self, id: uuid::Uuid) {
        let mut saved = self.saved_tasks.lock();
        let entry = saved.entry(id).or_default();
        entry.unsaved = Some(
            "it was put back as it was before a report that didn't come, and isn't saved as a version yet"
                .to_string(),
        );
    }

    /// Records a finalized agent result in `ctx`, replacing an earlier one
    /// of its task, after saving its task as a new version (unless it didn't
    /// change). A failed save is logged. Checkpoints then refer to the
    /// task's last saved version and say its latest state is missing, and
    /// the next checkpoint tries again. It never fails the run.
    pub(crate) async fn record_result(
        &self,
        ctx: &SharedContext,
        result: crate::workflow::result::AgentResult,
    ) {
        // A report on request is never saved: the run stays as it was until
        // the report comes, and is recorded then.
        if ctx
            .reports_in_flight
            .lock()
            .contains_key(&result.agent_run_id)
        {
            return;
        }
        let _guard = self.write_lock.lock().await;
        let saved = match result.task() {
            Ok(task) => self.write_task(task).await,
            Err(error) => Err(error),
        };
        if let Err(error) = &saved {
            log::warn!(
                "Workflow '{}': could not save the task of agent '{}' ({}): {error}. Checkpoints refer to its last saved state until one saves it.",
                ctx.workflow_name,
                result.agent_name,
                result.agent_run_id
            );
        }
        let id = result.agent_run_id;
        let mut results = ctx.agent_results.write();
        let mut tasks = self.saved_tasks.lock();
        crate::workflow::result::upsert_result(&mut results, result);
        let entry = tasks.entry(id).or_default();
        match saved {
            Ok(file) => {
                entry.file = Some(file);
                entry.unsaved = None;
            }
            Err(error) => entry.unsaved = Some(error.to_string()),
        }
        // Recorded: the run is no longer running, and checkpoints record
        // its result from now on.
        self.live.lock().runs.remove(&id);
        #[cfg(test)]
        self.reached(CrashPoint::ResultRecorded);
    }

    /// Save a checkpoint of the current workflow state.
    ///
    /// The checkpoint refers to the version of each agent run's task it
    /// records (`agents/<task id>/<version>.json`) instead of holding the
    /// task, so an older checkpoint still reads the task as it was then. A
    /// task version not saved yet (a task restored from a checkpoint that
    /// held it, or one whose save failed) is written first. If that fails
    /// again, the checkpoint refers to the task's last saved version (a run
    /// never saved is left out) and lists it in
    /// [`missing_task_states`](state::WorkflowState::missing_task_states),
    /// with a warning. With the task journal
    /// ([`PersistenceConfig::task_journal`]), the agent runs still running
    /// are recorded `Interrupted`, each with a version of its latest task
    /// snapshot, so resuming from the checkpoint can recover them. Buffered
    /// events are written first as well. With
    /// [`CheckpointRetention::KeepLast`], older checkpoint files beyond the
    /// count are then deleted, with the task versions no remaining
    /// checkpoint refers to.
    pub async fn save_checkpoint(&self, ctx: &SharedContext) -> Result<PathBuf, WorkflowError> {
        self.flush_events().await;
        let _guard = self.write_lock.lock().await;
        let Some(built) = self.build_state(ctx, false).await else {
            unreachable!("a versioned checkpoint is always built");
        };
        let ws = &built.state;
        let path = self
            .config
            .checkpoint_dir
            .join(format!("checkpoint-{}.json", ws.checkpoint_id));
        let json = self.images.with(|| serde_json::to_string_pretty(ws))?;
        self.write_file(&path, json.as_bytes(), true).await?;
        self.point_latest_at(&path, built.build, built.running, None)
            .await?;
        self.written.lock().push_back(path.clone());
        for missing in &ws.missing_task_states {
            log::warn!(
                "Workflow '{}': checkpoint {} lacks the latest state of agent '{}' ({}): {}. {} The next checkpoint tries again.",
                ctx.workflow_name,
                path.display(),
                missing.agent_name,
                missing.agent_run_id,
                missing.reason,
                match &missing.last_saved {
                    Some(file) => format!("It refers to the last saved state, {file}."),
                    None =>
                        "No state of it was saved, so the checkpoint leaves it out.".to_string(),
                }
            );
        }
        if let CheckpointRetention::KeepLast(keep) = self.config.retain_checkpoints {
            self.prune_checkpoints(keep).await;
        }
        self.delete_finished_live_files().await;
        log::info!("Checkpoint saved to {}", path.display());
        Ok(path)
    }

    /// The workflow's state for a checkpoint. The caller holds the write
    /// lock. Task versions not saved yet are written first. The agent runs
    /// the task journal saves are recorded `Interrupted`: in the live
    /// checkpoint (`live`) with their live files, in a versioned one with a
    /// version of their latest snapshot (so that it stays valid once the
    /// live file changes). For the live checkpoint, `None` when the live
    /// checkpoint `latest-checkpoint` names holds this state already (see
    /// [`Built::fingerprint`]): the cost ledger isn't even copied then.
    async fn build_state(&self, ctx: &SharedContext, live: bool) -> Option<Built> {
        // Results change only with the write lock held. Only the results
        // whose tasks must be written are taken (a clone shares the task).
        let pending = {
            let results = ctx.agent_results.read();
            let tasks = self.saved_tasks.lock();
            results
                .iter()
                .filter(|result| {
                    tasks
                        .get(&result.agent_run_id)
                        .is_none_or(|saved| saved.unsaved.is_some() || saved.file.is_none())
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        for result in pending {
            let saved = match result.task() {
                Ok(task) => self.write_task(task).await,
                Err(error) => Err(error),
            };
            let mut tasks = self.saved_tasks.lock();
            let entry = tasks.entry(result.agent_run_id).or_default();
            match saved {
                Ok(file) => {
                    entry.file = Some(file);
                    entry.unsaved = None;
                }
                Err(error) => entry.unsaved = Some(error.to_string()),
            }
        }
        // A versioned checkpoint's versions of the running tasks.
        let mut live_versions = HashMap::new();
        if !live {
            let snapshots = self
                .live
                .lock()
                .runs
                .iter()
                .filter_map(|(id, run)| run.task.clone().map(|task| (*id, task)))
                .collect::<Vec<_>>();
            for (id, task) in snapshots {
                let saved = self
                    .write_task(&task)
                    .await
                    .map_err(|error| error.to_string());
                live_versions.insert(id, saved);
            }
        }

        // The state shared by the agents, which the fingerprint covers.
        let memory_snapshot = ctx.memory.snapshot();
        let message_queues = ctx.messages.snapshot();
        let freshness_snapshot = ctx
            .resource_tracker
            .as_ref()
            .map(|tracker| tracker.snapshot());
        let verification_records = ctx.verification_records.read().clone();
        let distillation_records = ctx.distillation_records.read().clone();
        let cancellation = ctx.run_control.cancellation();
        let agent_registry = ctx.registry.snapshot();
        let budget_limits = state::BudgetLimits {
            limit: ctx.budget.limit(),
            agent_limits: ctx.budget.agent_limits().into_iter().collect(),
        };
        let plan_budget = ctx.budget.plan_state();
        let notes = ctx.notes.snapshot();
        let subagents = ctx.subagents.snapshot();
        let shared = self.images.with(|| {
            fingerprint(&(
                &memory_snapshot,
                &message_queues,
                &freshness_snapshot,
                &verification_records,
                &distillation_records,
                &cancellation,
                &agent_registry,
                &budget_limits,
                &plan_budget,
                &notes,
                &subagents,
            ))
        });

        let mut agent_runs = Vec::new();
        let mut missing_task_states = Vec::new();
        let mut running_ids = HashSet::new();
        let mut live_files = HashSet::new();
        let completed_agents: Vec<state::SerializableAgentResult>;
        let (budget_spent, cost_ledger, fingerprint);
        {
            let results = ctx.agent_results.read();
            let tasks = self.saved_tasks.lock();
            // Held while the ledger is read, so that it and each run's
            // charged usage agree (see `charge_usage`).
            let journal = self.live.lock();
            let mut running = journal.runs.iter().collect::<Vec<_>>();
            running.sort_by_key(|(_, run)| run.started);
            let mut running_records = Vec::new();
            for (id, run) in running {
                let (file, unsaved) = if live {
                    (
                        run.written.then(|| live_task_file(*id)),
                        run.unsaved.clone(),
                    )
                } else {
                    match live_versions.remove(id) {
                        Some(Ok(file)) => (Some(file), None),
                        Some(Err(error)) => (None, Some(error)),
                        None => (None, None),
                    }
                };
                if let Some(reason) = unsaved {
                    missing_task_states.push(state::MissingTaskState {
                        agent_run_id: *id,
                        agent_name: run.agent_name.clone(),
                        status: AgentRunStatus::Interrupted,
                        last_saved: file.clone(),
                        reason,
                    });
                }
                // A run with no saved snapshot yet keeps the result it had
                // (a continued run), if any.
                let Some(file) = file else {
                    continue;
                };
                running_ids.insert(*id);
                running_records.push(state::AgentRunRecord {
                    agent_run_id: *id,
                    agent_name: run.agent_name.clone(),
                    status: AgentRunStatus::Interrupted,
                    summary: String::new(),
                    error: Some(RUNNING_AT_CHECKPOINT.to_string()),
                    task_file: file,
                    usage_charged: Some(run.usage_charged),
                    // A stop of the run alone under way: a resume keeps the
                    // host's intent for it.
                    stopped_by: ctx.run_control.run_stop_record(*id),
                    // Its task is read with the checkpoint (it is recovered).
                    info: None,
                });
            }
            for result in results.iter() {
                if running_ids.contains(&result.agent_run_id) {
                    continue;
                }
                let saved = tasks.get(&result.agent_run_id).cloned().unwrap_or_default();
                if let Some(reason) = saved.unsaved {
                    missing_task_states.push(state::MissingTaskState {
                        agent_run_id: result.agent_run_id,
                        agent_name: result.agent_name.clone(),
                        status: result.status.clone(),
                        last_saved: saved.file.clone(),
                        reason,
                    });
                }
                if let Some(file) = saved.file {
                    agent_runs.push(state::AgentRunRecord::of(result, file));
                }
            }
            agent_runs.extend(running_records);
            if live {
                live_files.extend(
                    agent_runs
                        .iter()
                        .filter_map(|record| live_file_task(&record.task_file)),
                );
            }
            completed_agents = results
                .iter()
                .filter(|result| !running_ids.contains(&result.agent_run_id))
                .map(|r| state::SerializableAgentResult {
                    agent_name: r.agent_name.clone(),
                    summary: r.summary.clone(),
                })
                .collect();
            // The ledger's entries the running runs' records don't count:
            // a resume can't tell those from the live files.
            let counted = agent_runs
                .iter()
                .filter(|record| running_ids.contains(&record.agent_run_id))
                .filter_map(|record| record.usage_charged)
                .sum::<usize>();
            let uncounted = ctx.budget.ledger_len().saturating_sub(counted);
            let records = agent_runs
                .iter()
                .map(|record| {
                    (
                        record.agent_run_id,
                        &record.agent_name,
                        &record.status,
                        &record.summary,
                        &record.error,
                        &record.task_file,
                        &record.stopped_by,
                    )
                })
                .collect::<Vec<_>>();
            fingerprint = self::fingerprint(&(
                shared,
                &records,
                &missing_task_states,
                &completed_agents,
                uncounted,
            ));
            if live {
                let latest = self.latest.lock();
                if latest.name.as_deref() == Some(LIVE_CHECKPOINT_FILE)
                    && latest.live == Some(fingerprint)
                {
                    return None;
                }
            }
            budget_spent = ctx.budget.total_spent();
            cost_ledger = ctx.budget.snapshot_ledger();
        }
        let state = state::WorkflowState {
            workflow_run_id: ctx.workflow_run_id,
            workflow_name: ctx.workflow_name.clone(),
            workflow_metadata: ctx.workflow_metadata.clone(),
            checkpoint_id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            memory_snapshot,
            message_queues,
            budget_spent,
            cost_ledger,
            freshness_snapshot,
            agent_results: Vec::new(),
            agent_runs,
            missing_task_states,
            verification_records,
            distillation_records,
            completed_agents,
            cancellation,
            agent_registry: Some(agent_registry),
            notes,
            budget_limits: Some(budget_limits),
            // Recorded once the run has a plan budget or has seen plan usage.
            plan_budget: (plan_budget != Default::default()).then_some(plan_budget),
            subagents,
        };
        Some(Built {
            state,
            build: self.builds.fetch_add(1, Ordering::SeqCst) + 1,
            running: running_ids,
            live_files,
            fingerprint,
        })
    }

    /// Points `latest-checkpoint` at checkpoint file `path` (in the run
    /// directory) of build `build`, which records `running` as running,
    /// unless it names a newer checkpoint (built later) already. For the
    /// live checkpoint, `live` is its state's fingerprint and whether its
    /// directory entry is flushed.
    async fn point_latest_at(
        &self,
        path: &Path,
        build: u64,
        running: HashSet<uuid::Uuid>,
        live: Option<(u64, bool)>,
    ) -> Result<(), WorkflowError> {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        let _writing = self.pointer_writes.lock().await;
        let rewrite = {
            let latest = self.latest.lock();
            if build < latest.build {
                return Ok(());
            }
            latest.name.as_deref() != Some(name.as_str())
        };
        if rewrite {
            self.write_file(
                &self.config.checkpoint_dir.join("latest-checkpoint"),
                name.as_bytes(),
                true,
            )
            .await?;
        }
        *self.latest.lock() = Latest {
            name: Some(name),
            build,
            running,
            live: live.map(|(fingerprint, _)| fingerprint),
            synced: live.is_none_or(|(_, synced)| synced),
        };
        Ok(())
    }

    /// Deletes checkpoint files beyond the newest `keep`: those of earlier
    /// processes (oldest first by modification time), then this manager's
    /// in the order it wrote them. Then the task versions no remaining
    /// checkpoint refers to. A file that can't be deleted is logged and
    /// kept. The caller holds the write lock.
    async fn prune_checkpoints(&self, keep: NonZeroUsize) {
        let written = self.written.lock().iter().cloned().collect::<Vec<_>>();
        let mut earlier = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.config.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("checkpoint-")
                    && name.ends_with(".json")
                    && !written.contains(&path)
                {
                    let modified = entry
                        .metadata()
                        .and_then(|metadata| metadata.modified())
                        .ok();
                    earlier.push((modified, path));
                }
            }
        }
        earlier.sort();
        let ordered = earlier
            .into_iter()
            .map(|(_, path)| path)
            .chain(written)
            .collect::<Vec<_>>();
        let excess = ordered.len().saturating_sub(keep.get());
        let mut retained = Vec::new();
        for (index, path) in ordered.into_iter().enumerate() {
            if index >= excess {
                retained.push(path);
                continue;
            }
            match self.remove_file(&path).await {
                Ok(()) => self.written.lock().retain(|written| *written != path),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    self.written.lock().retain(|written| *written != path);
                }
                Err(error) => {
                    log::warn!(
                        "Could not delete old checkpoint {}: {error}",
                        path.display()
                    );
                    retained.push(path);
                }
            }
        }
        self.prune_task_versions(&retained).await;
    }

    /// Deletes the task versions that neither a checkpoint in `retained`,
    /// the live checkpoint (while `latest-checkpoint` names it), nor a
    /// recorded result refers to. A live checkpoint that a newer checkpoint
    /// replaced as the latest is stale: it is never named again unless it
    /// is written anew, from the current state. Nothing is
    /// deleted if one of those checkpoints can't be read (it may refer to
    /// any version). Live task files and the task files of older versions
    /// (`agents/<task id>.json`) are left alone.
    async fn prune_task_versions(&self, retained: &[PathBuf]) {
        let mut referenced = self
            .saved_tasks
            .lock()
            .values()
            .filter_map(|saved| saved.file.clone())
            .collect::<HashSet<_>>();
        let live_checkpoint = self.config.checkpoint_dir.join(LIVE_CHECKPOINT_FILE);
        let live_named = self.latest.lock().name.as_deref() == Some(LIVE_CHECKPOINT_FILE);
        let live_checkpoint = (live_named && live_checkpoint.is_file()).then_some(live_checkpoint);
        for path in retained.iter().chain(&live_checkpoint) {
            match checkpoint_references(path) {
                Ok(references) => referenced.extend(references),
                Err(error) => {
                    log::warn!(
                        "Not deleting task versions because checkpoint {} can't be read ({error})",
                        path.display()
                    );
                    return;
                }
            }
        }
        let tasks_dir = self.config.checkpoint_dir.join(TASKS_DIR);
        let Ok(task_dirs) = std::fs::read_dir(&tasks_dir) else {
            return;
        };
        for task_dir in task_dirs.flatten() {
            if !task_dir.path().is_dir() {
                continue;
            }
            let Ok(versions) = std::fs::read_dir(task_dir.path()) else {
                continue;
            };
            for version in versions.flatten() {
                let name = version.file_name().to_string_lossy().into_owned();
                if !name.ends_with(".json") || name == LIVE_TASK_FILE {
                    continue;
                }
                let file = format!(
                    "{TASKS_DIR}/{}/{name}",
                    task_dir.file_name().to_string_lossy()
                );
                if referenced.contains(&file) {
                    continue;
                }
                if let Err(error) = self.remove_file(&version.path()).await
                    && error.kind() != std::io::ErrorKind::NotFound
                {
                    log::warn!(
                        "Could not delete task version {}: {error}",
                        version.path().display()
                    );
                }
            }
        }
    }

    /// Records `event` in the event log, as [`PersistenceConfig::event_log`]
    /// selects. Events are buffered and written at least every
    /// [`PersistenceConfig::event_flush_interval`]. State changes, run ends
    /// and error reports are written at once. A failed write is logged as a
    /// warning and retried with the next write. It never stops an agent.
    pub async fn append_event(
        &self,
        agent_name: &str,
        agent_run_id: uuid::Uuid,
        event: &arko_agents::runtime::event::AgentEvent,
    ) {
        self.events.append(agent_name, agent_run_id, event).await;
    }

    /// Writes the buffered events now.
    pub async fn flush_events(&self) {
        self.events.flush().await;
    }

    /// Records a posted note in the event log, whatever
    /// [`PersistenceConfig::event_log`] selects: a line with the current UTC
    /// time, the workflow run UUID, and `note`, the [`PostedNote`](crate::notes::PostedNote).
    /// Written at once when a Tokio runtime is running, otherwise with the
    /// next write.
    pub(crate) fn append_note(&self, note: &crate::notes::PostedNote) {
        #[derive(serde::Serialize)]
        struct PostedNoteRecord<'a> {
            timestamp: chrono::DateTime<chrono::Utc>,
            workflow_run_id: uuid::Uuid,
            note: &'a crate::notes::PostedNote,
        }
        self.events.append_record(
            &PostedNoteRecord {
                timestamp: chrono::Utc::now(),
                workflow_run_id: self.events.workflow_run_id,
                note,
            },
            "a posted note",
        );
    }

    /// Records a background subagent's event in the event log, whatever
    /// [`PersistenceConfig::event_log`] selects: a line with the event's
    /// fields and `"subagent": true` (see
    /// [`SubagentEvent`](crate::subagents::SubagentEvent)).
    pub(crate) fn append_subagent_event(&self, event: &crate::subagents::SubagentEvent) {
        #[derive(serde::Serialize)]
        struct SubagentEventRecord<'a> {
            subagent: bool,
            #[serde(flatten)]
            event: &'a crate::subagents::SubagentEvent,
        }
        self.events.append_record(
            &SubagentEventRecord {
                subagent: true,
                event,
            },
            "a subagent event",
        );
    }

    /// Saves `task` as a version in the run directory, unless that version
    /// is there already, and returns its path. A checkpoint refers to the
    /// versions of its results' tasks. This call doesn't change which, so
    /// with [`CheckpointRetention::KeepLast`] the next checkpoint deletes a
    /// version no checkpoint or recorded result refers to.
    pub async fn save_agent_task(
        &self,
        task: &arko_agents::Task,
    ) -> Result<PathBuf, WorkflowError> {
        let _guard = self.write_lock.lock().await;
        let file = self.write_task(task).await?;
        Ok(self.config.checkpoint_dir.join(file))
    }

    /// Writes `task`'s version, unless it is there already, and returns its
    /// file relative to the run directory. The caller holds the write lock.
    async fn write_task(&self, task: &arko_agents::Task) -> Result<String, WorkflowError> {
        #[cfg(test)]
        if self.fail_task_writes.load(Ordering::SeqCst) {
            return Err(WorkflowError::Persistence(
                "injected task write failure".to_string(),
            ));
        }
        let (file, content) = task_version(task, &self.images)?;
        let path = self.config.checkpoint_dir.join(&file);
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            return Ok(file);
        }
        self.write_file(&path, &content, true).await?;
        Ok(file)
    }

    /// Writes `bytes` to `path` with [`write_replacing`], unless a crash is
    /// being simulated.
    async fn write_file(
        &self,
        path: &Path,
        bytes: &[u8],
        sync_directory: bool,
    ) -> std::io::Result<()> {
        #[cfg(test)]
        if self.crashed.load(Ordering::SeqCst) {
            return Ok(());
        }
        write_replacing(path, bytes, sync_directory).await
    }

    /// Flushes the entries of the directory holding `path` to disk, unless a
    /// crash is being simulated.
    async fn sync_directory_of(&self, path: &Path) -> std::io::Result<()> {
        #[cfg(test)]
        if self.crashed.load(Ordering::SeqCst) {
            return Ok(());
        }
        sync_directory(parent_directory(path)).await
    }

    /// Deletes `path`, unless a crash is being simulated.
    async fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        #[cfg(test)]
        if self.crashed.load(Ordering::SeqCst) {
            return Ok(());
        }
        tokio::fs::remove_file(path).await
    }

    /// Whether the task journal saves running agents
    /// ([`PersistenceConfig::task_journal`]).
    pub(crate) fn journals_tasks(&self) -> bool {
        self.config.task_journal
    }

    /// Starts saving agent run `id` of `agent_name` as it runs, its budget
    /// charged for the first `usage_charged` entries of its task's usage
    /// history. Its snapshots arrive through a [`TaskJournal`]. It is saved
    /// until its result is recorded or [`end_live_run`](Self::end_live_run)
    /// drops it.
    pub(crate) fn begin_live_run(&self, id: uuid::Uuid, agent_name: &str, usage_charged: usize) {
        if !self.config.task_journal {
            return;
        }
        let mut live = self.live.lock();
        live.started += 1;
        let started = live.started;
        live.runs.insert(
            id,
            LiveRun {
                agent_name: agent_name.to_string(),
                started,
                usage_charged,
                task: None,
                pending: false,
                written: false,
                synced: false,
                unsaved: None,
            },
        );
        live.files.entry(id).or_default();
    }

    /// Stops saving agent run `id` without recording it (the run was
    /// dropped, or its caller drives it outside the workflow): later
    /// checkpoints leave it out, and its live file is deleted once no
    /// checkpoint refers to it.
    pub(crate) fn end_live_run(&self, id: uuid::Uuid) {
        self.live.lock().runs.remove(&id);
    }

    /// The latest snapshot the task journal has of running agent run `id`,
    /// if it saves the run and has one.
    pub(crate) fn latest_live_task(&self, id: uuid::Uuid) -> Option<arko_agents::Task> {
        self.live.lock().runs.get(&id)?.task.as_deref().cloned()
    }

    /// Where the images of running agent run `id`'s latest snapshot are in
    /// memory, to see which a later snapshot kept.
    #[cfg(test)]
    pub(crate) fn live_image_addresses(&self, id: uuid::Uuid) -> Vec<usize> {
        let live = self.live.lock();
        let Some(task) = live.runs.get(&id).and_then(|run| run.task.as_deref()) else {
            return Vec::new();
        };
        task.conversation
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|content| match content {
                arko_ai::Content::Image(arko_ai::Image::Base64 { data, .. }) => {
                    Some(data.as_ptr() as usize)
                }
                _ => None,
            })
            .collect()
    }

    /// Whether the task journal saves any running agent run now.
    pub(crate) fn has_live_runs(&self) -> bool {
        !self.live.lock().runs.is_empty()
    }

    /// Charges usage the workflow processed for running agent run `id`:
    /// `charge` adds it to the budget, and the run's count of charged usage
    /// goes up with it, both while checkpoints can't read them, so that a
    /// checkpoint's ledger and counts agree.
    pub(crate) fn charge_usage<T>(&self, id: uuid::Uuid, charge: impl FnOnce() -> T) -> T {
        let mut live = self.live.lock();
        let charged = charge();
        if let Some(run) = live.runs.get_mut(&id) {
            run.usage_charged += 1;
        }
        charged
    }

    /// A snapshot of running task `task` arrived (see [`TaskJournal`]): it
    /// is written to the task's live file, then the live checkpoint. With
    /// `durable`, both are on disk (their directory entries flushed too)
    /// when this returns. Otherwise they are written in the background, a
    /// later snapshot replacing one not written yet, and only the files are
    /// flushed (a crash of the machine may then keep the older file, never
    /// a part).
    async fn live_task_changed(
        &self,
        ctx: &Arc<SharedContext>,
        task: &arko_agents::Task,
        durable: bool,
    ) -> Result<(), String> {
        let id = task.id;
        // The latest snapshot becomes this one in place when no one holds it
        // (the file isn't being written from it), keeping the messages and
        // images it has instead of cloning them again. A task with many
        // images costs a comparison of them, not a copy.
        let refreshed = {
            let mut live = self.live.lock();
            match live.runs.get_mut(&id) {
                // No longer saved: recorded, or dropped.
                None => return Ok(()),
                Some(run) => match run.task.as_mut().and_then(Arc::get_mut) {
                    Some(latest) => {
                        task.refresh_copy(latest);
                        run.pending = true;
                        true
                    }
                    None => false,
                },
            }
        };
        if !refreshed {
            let copy = Arc::new(task.clone());
            let mut live = self.live.lock();
            let Some(run) = live.runs.get_mut(&id) else {
                return Ok(());
            };
            run.task = Some(copy);
            run.pending = true;
        }
        if durable {
            return self.save_live_run(ctx, id, true).await;
        }
        let (manager, ctx) = (self.clone(), ctx.clone());
        tokio::spawn(async move {
            if let Err(error) = manager.save_live_run(&ctx, id, false).await {
                log::warn!(
                    "Workflow '{}': could not save running agent run {id}: {error}. The next save tries again.",
                    ctx.workflow_name
                );
            }
        });
        Ok(())
    }

    /// Writes running task `id`'s latest snapshot to its live file, if it
    /// isn't there yet, and then the live checkpoint. When another call
    /// wrote the snapshot already, that call writes the checkpoint. A
    /// `durable` save still waits for a checkpoint written after it began.
    async fn save_live_run(
        &self,
        ctx: &SharedContext,
        id: uuid::Uuid,
        durable: bool,
    ) -> Result<(), String> {
        let written = match self.write_live_task(id, durable).await {
            Some(written) => written,
            None if durable => Ok(()),
            None => return Ok(()),
        };
        let checkpoint = self
            .write_live_checkpoint(ctx, durable)
            .await
            .map_err(|error| format!("the live checkpoint could not be written: {error}"));
        written.and(checkpoint)
    }

    /// Writes running task `id`'s latest snapshot to its live file. With
    /// `durable`, the file's directory entry is flushed too (also for a
    /// snapshot written already). `None` when there was nothing to do: the
    /// file holds the latest snapshot already, or the run is no longer
    /// saved.
    async fn write_live_task(&self, id: uuid::Uuid, durable: bool) -> Option<Result<(), String>> {
        let writer = self.live.lock().files.entry(id).or_default().clone();
        let _writing = writer.lock().await;
        let task = {
            let mut live = self.live.lock();
            let run = live.runs.get_mut(&id)?;
            if std::mem::take(&mut run.pending) {
                Some(run.task.clone()?)
            } else if durable && run.written && !run.synced {
                None
            } else {
                return None;
            }
        };
        let path = self.config.checkpoint_dir.join(live_task_file(id));
        let rewritten = task.is_some();
        let result = match task {
            Some(task) => {
                let (pretty, images) = (self.config.pretty_json, self.images.clone());
                // Off the async threads: a long conversation takes a while.
                let json = tokio::task::spawn_blocking(move || {
                    images.with(|| {
                        if pretty {
                            serde_json::to_vec_pretty(&*task)
                        } else {
                            serde_json::to_vec(&*task)
                        }
                    })
                })
                .await;
                match json {
                    Ok(Ok(json)) => self
                        .write_file(&path, &json, durable)
                        .await
                        .map_err(|error| error.to_string()),
                    Ok(Err(error)) => Err(error.to_string()),
                    Err(error) => Err(error.to_string()),
                }
            }
            None => self
                .sync_directory_of(&path)
                .await
                .map_err(|error| error.to_string()),
        };
        let mut live = self.live.lock();
        let run = live.runs.get_mut(&id)?;
        Some(match result {
            Ok(()) => {
                run.written = true;
                run.synced = durable || (run.synced && !rewritten);
                if run.unsaved.take().is_some() {
                    log::info!("Running agent run {id} is saved again");
                }
                Ok(())
            }
            Err(error) => {
                // Written with the next snapshot, or the next checkpoint.
                run.pending |= rewritten;
                let first = run.unsaved.replace(error.clone()).is_none();
                if first {
                    log::warn!(
                        "Could not save running agent run {id} of agent '{}' to {}: {error}. Checkpoints refer to its last saved state until a save succeeds.",
                        run.agent_name,
                        path.display()
                    );
                }
                Err(error)
            }
        })
    }

    /// Writes the live checkpoint ([`LIVE_CHECKPOINT_FILE`]) with the state
    /// as it is once earlier writes are done: a request made while another
    /// write runs is answered by the next one. `latest-checkpoint` then
    /// points to it, unless a newer checkpoint was written meanwhile. When
    /// the live checkpoint `latest-checkpoint` names holds that state
    /// already (a resume would read the same from it), nothing is written.
    /// With `durable`, its directory entry is flushed if it isn't yet.
    pub(crate) async fn write_live_checkpoint(
        &self,
        ctx: &SharedContext,
        durable: bool,
    ) -> Result<(), WorkflowError> {
        use std::sync::atomic::Ordering::SeqCst;
        let writer = &self.live_checkpoints;
        let request = writer.requested.fetch_add(1, SeqCst) + 1;
        let _writing = writer.lock.lock().await;
        let answered = if durable {
            &writer.synced
        } else {
            &writer.written
        };
        if answered.load(SeqCst) >= request {
            return Ok(());
        }
        let covered = writer.requested.load(SeqCst);
        // Every snapshot not written yet is written first (a failed write
        // is tried again): the checkpoint then lists each run at its latest
        // snapshot. Otherwise a run whose snapshot was still on its way
        // (a non-durable save) would be left out while a run it started
        // (a completion verifier's review) is listed, and after a crash
        // the parent would be lost and its child listed as an orphan.
        let unwritten = self
            .live
            .lock()
            .runs
            .iter()
            .filter(|(_, run)| run.pending)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in unwritten {
            let _ = self.write_live_task(id, durable).await;
        }
        let built = {
            let _guard = self.write_lock.lock().await;
            let built = self.build_state(ctx, true).await;
            if let Some(built) = &built {
                // Kept from now on, until the file is written without them.
                self.live
                    .lock()
                    .referenced
                    .extend(built.live_files.iter().copied());
            }
            built
        };
        let path = self.config.checkpoint_dir.join(LIVE_CHECKPOINT_FILE);
        match built {
            None => {
                if durable && !self.latest.lock().synced {
                    self.sync_directory_of(&path).await?;
                    let mut latest = self.latest.lock();
                    if latest.name.as_deref() == Some(LIVE_CHECKPOINT_FILE) {
                        latest.synced = true;
                    }
                }
            }
            Some(built) => {
                let (pretty, state) = (self.config.pretty_json, built.state);
                let images = self.images.clone();
                // Off the async threads: the cost ledger can be long.
                let json = tokio::task::spawn_blocking(move || {
                    images.with(|| {
                        if pretty {
                            serde_json::to_vec_pretty(&state)
                        } else {
                            serde_json::to_vec(&state)
                        }
                    })
                })
                .await
                .map_err(std::io::Error::other)??;
                if let Err(error) = self.write_file(&path, &json, durable).await {
                    // The file may hold this state or the one before.
                    self.latest.lock().live = None;
                    return Err(error.into());
                }
                self.live.lock().referenced = built.live_files;
                self.point_latest_at(
                    &path,
                    built.build,
                    built.running,
                    Some((built.fingerprint, durable)),
                )
                .await?;
            }
        }
        writer.written.fetch_max(covered, SeqCst);
        if durable {
            writer.synced.fetch_max(covered, SeqCst);
        }
        self.delete_finished_live_files().await;
        Ok(())
    }

    /// After agent run `id`'s result is recorded: when the live checkpoint
    /// refers to the run's live file, or `latest-checkpoint` names a
    /// checkpoint recording the run as running, the live checkpoint is
    /// written again, recording its end: the file can go, and a resume
    /// finds the run ended.
    pub(crate) async fn release_live_run(
        &self,
        ctx: &SharedContext,
        id: uuid::Uuid,
    ) -> Result<(), WorkflowError> {
        let recorded_running =
            self.live.lock().referenced.contains(&id) || self.latest.lock().running.contains(&id);
        if recorded_running {
            self.write_live_checkpoint(ctx, false).await?;
        }
        #[cfg(test)]
        self.reached(CrashPoint::LiveRunReleased);
        Ok(())
    }

    /// Deletes the live files of the runs no longer running that the live
    /// checkpoint no longer refers to (a versioned checkpoint refers to one
    /// only while it is the task's last saved state), unless the run's
    /// recorded state could not be saved as a version. A file that can't
    /// be deleted is tried again after the next checkpoint.
    async fn delete_finished_live_files(&self) {
        let finished = {
            let tasks = self.saved_tasks.lock();
            let live = self.live.lock();
            live.files
                .iter()
                .filter(|(id, _)| {
                    !live.runs.contains_key(id)
                        && !live.referenced.contains(id)
                        && tasks
                            .get(id)
                            .is_none_or(|saved| saved.unsaved.is_none() && saved.file.is_some())
                })
                .map(|(id, writer)| (*id, writer.clone()))
                .collect::<Vec<_>>()
        };
        for (id, writer) in finished {
            // After a write in progress.
            let _writing = writer.lock().await;
            if self.live.lock().runs.contains_key(&id) {
                // Running again (recovered).
                continue;
            }
            let path = self.config.checkpoint_dir.join(live_task_file(id));
            match self.remove_file(&path).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    log::warn!(
                        "Could not delete the live file {} of a finished agent run: {error}",
                        path.display()
                    );
                    continue;
                }
            }
            let mut live = self.live.lock();
            if !live.runs.contains_key(&id) {
                live.files.remove(&id);
            }
        }
    }

    /// Auto-checkpoint if configured.
    pub async fn auto_checkpoint(&self, ctx: &SharedContext) -> Result<(), WorkflowError> {
        if self.config.auto_checkpoint {
            self.save_checkpoint(ctx).await?;
        }
        Ok(())
    }

    /// Load a checkpoint from disk. Its agent results are in
    /// [`WorkflowState::agent_results_in`](state::WorkflowState::agent_results_in).
    pub async fn load_checkpoint(path: &Path) -> Result<state::WorkflowState, WorkflowError> {
        let content = tokio::fs::read_to_string(path).await?;
        let images = images::ImageFiles::in_run_dir(parent_directory(path));
        let ws: state::WorkflowState = images.with(|| serde_json::from_str(&content))?;
        Ok(ws)
    }
}

/// Saves an agent run's task at its safe points for the task journal
/// ([`PersistenceConfig::task_journal`]): the workflow gives one to each run
/// it starts while the journal is on.
pub(crate) struct TaskJournal {
    persistence: Arc<PersistenceManager>,
    context: Arc<SharedContext>,
}

impl TaskJournal {
    pub(crate) fn new(persistence: Arc<PersistenceManager>, context: Arc<SharedContext>) -> Self {
        Self {
            persistence,
            context,
        }
    }
}

#[async_trait::async_trait]
impl arko_agents::TaskSnapshotSink for TaskJournal {
    async fn task_changed(&self, task: &arko_agents::Task, durable: bool) -> Result<(), String> {
        self.persistence
            .live_task_changed(&self.context, task, durable)
            .await
    }
}

/// The task files the checkpoint at `path` refers to, relative to its
/// directory.
fn checkpoint_references(path: &Path) -> std::io::Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct References {
        #[serde(default)]
        agent_runs: Vec<Reference>,
    }
    #[derive(serde::Deserialize)]
    struct Reference {
        task_file: String,
    }
    let json = std::fs::read(path)?;
    let references = serde_json::from_slice::<References>(&json).map_err(std::io::Error::other)?;
    Ok(references
        .agent_runs
        .into_iter()
        .map(|reference| reference.task_file)
        .collect())
}

/// The task whose live file `file` (relative to the run directory) is, if
/// it is one.
fn live_file_task(file: &str) -> Option<uuid::Uuid> {
    file.strip_prefix(TASKS_DIR)?
        .strip_prefix('/')?
        .strip_suffix(LIVE_TASK_FILE)?
        .strip_suffix('/')?
        .parse()
        .ok()
}

/// Writes `bytes` to `path` and flushes them to disk through the same handle.
/// Windows can only flush a handle opened for writing, so reopening the file
/// read-only to sync it fails there with "Access is denied".
async fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = retry_while_shared(|| tokio::fs::File::create(path)).await?;
    file.write_all(bytes).await?;
    file.sync_all().await
}

/// Replaces `path` with `bytes`, so that a reader, or a crash, finds the
/// old content or the new, never a part: the bytes are written to a
/// temporary file beside it (`<name>.<process>-<n>.tmp`, in the directory,
/// created if missing), flushed to disk, and renamed over `path`. On Unix
/// the directory is flushed too, so the rename survives a crash of the
/// machine (a directory created for the file is flushed into its parent
/// as well). On Windows a rename refused because another program has the
/// file open (an antivirus scanner, an indexer, an editor) is tried again a
/// few times, as is creating the temporary file. The temporary file is
/// removed when the write fails. Every
/// file of a run directory is written this way except the append-only logs
/// and `workflow-run-id`, which is created once, exclusively.
///
/// With `sync_directory` unset, the directory isn't flushed after the
/// rename: a crash of the machine (not of the process) may then leave the
/// old file, still whole.
async fn write_replacing(path: &Path, bytes: &[u8], sync_directory: bool) -> std::io::Result<()> {
    /// Makes the temporary files' names unique within the process.
    static TEMPORARIES: AtomicU64 = AtomicU64::new(0);
    let directory = parent_directory(path);
    create_directory(directory).await?;
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(
        ".{}-{}{TEMPORARY_SUFFIX}",
        std::process::id(),
        TEMPORARIES.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = PathBuf::from(temporary);
    let written = async {
        write_synced(&temporary, bytes).await?;
        rename_replacing(&temporary, path).await
    }
    .await;
    if let Err(error) = written {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error);
    }
    if sync_directory {
        self::sync_directory(directory).await?;
    }
    Ok(())
}

/// The ending of the temporary files [`write_replacing`] writes.
const TEMPORARY_SUFFIX: &str = ".tmp";

/// The directory holding `path`.
fn parent_directory(path: &Path) -> &Path {
    match path.parent() {
        Some(directory) if !directory.as_os_str().is_empty() => directory,
        _ => Path::new("."),
    }
}

/// Creates `directory` and its missing parents, each flushed into its
/// parent (Unix), so that files written into it are found after a crash
/// of the machine.
async fn create_directory(directory: &Path) -> std::io::Result<()> {
    if tokio::fs::metadata(directory)
        .await
        .is_ok_and(|metadata| metadata.is_dir())
    {
        return Ok(());
    }
    // The directories to create, the deepest first.
    let mut missing = Vec::new();
    let mut next = Some(directory);
    while let Some(current) = next {
        if current.as_os_str().is_empty() || tokio::fs::try_exists(current).await.unwrap_or(true) {
            break;
        }
        missing.push(current);
        next = current.parent();
    }
    tokio::fs::create_dir_all(directory).await?;
    for created in missing.into_iter().rev() {
        sync_directory(parent_directory(created)).await?;
    }
    Ok(())
}

/// Removes what an earlier process left in run directory `directory`: the
/// temporary files of writes it didn't finish, and the live task files no
/// checkpoint refers to (written just before it stopped, or of runs it
/// finished). Returns the tasks whose live files the live checkpoint refers
/// to. Nothing is removed when a checkpoint that could refer to live files
/// can't be read. A file that can't be removed is left.
fn sweep_run_directory(directory: &Path) -> HashSet<uuid::Uuid> {
    let mut referenced = HashSet::new();
    let mut kept = HashSet::new();
    let mut known = true;
    let latest = std::fs::read_to_string(directory.join("latest-checkpoint"))
        .ok()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty() && !name.contains(['/', '\\']));
    let live_checkpoint = directory.join(LIVE_CHECKPOINT_FILE);
    for path in
        std::iter::once(live_checkpoint.clone()).chain(latest.map(|name| directory.join(name)))
    {
        match checkpoint_references(&path) {
            Ok(files) => {
                let tasks = files.iter().filter_map(|file| live_file_task(file));
                if path == live_checkpoint {
                    let tasks = tasks.collect::<Vec<_>>();
                    referenced.extend(tasks.iter().copied());
                    kept.extend(tasks);
                } else {
                    kept.extend(tasks);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => known = false,
        }
    }
    let remove = |path: &Path| {
        if let Err(error) = std::fs::remove_file(path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            log::warn!("Could not remove {}: {error}", path.display());
        }
    };
    let temporary = |path: &Path| {
        path.file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with(TEMPORARY_SUFFIX))
            && path.is_file()
    };
    for entry in std::fs::read_dir(directory)
        .into_iter()
        .chain(std::fs::read_dir(directory.join(images::IMAGES_DIR)))
        .flatten()
        .flatten()
    {
        if temporary(&entry.path()) {
            remove(&entry.path());
        }
    }
    for task_dir in std::fs::read_dir(directory.join(TASKS_DIR))
        .into_iter()
        .flatten()
        .flatten()
    {
        let task = task_dir
            .file_name()
            .to_string_lossy()
            .parse::<uuid::Uuid>()
            .ok();
        for entry in std::fs::read_dir(task_dir.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let path = entry.path();
            let unreferenced_live_file = entry.file_name() == LIVE_TASK_FILE
                && known
                && task.is_some_and(|task| !kept.contains(&task));
            if temporary(&path) || unreferenced_live_file {
                remove(&path);
            }
        }
    }
    referenced
}

/// A hash of `value`'s JSON form (through a JSON value, whose maps are
/// ordered): equal for equal values. A value that can't be serialized
/// hashes to a random number, equal to nothing.
fn fingerprint(value: &impl serde::Serialize) -> u64 {
    use std::hash::Hasher;
    struct HashWriter(std::collections::hash_map::DefaultHasher);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.write(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(Default::default());
    match serde_json::to_value(value).and_then(|value| serde_json::to_writer(&mut writer, &value)) {
        Ok(()) => writer.0.finish(),
        Err(_) => uuid::Uuid::new_v4().as_u64_pair().0,
    }
}

/// Renames `from` over `to`, [retrying](retry_while_shared) on Windows.
async fn rename_replacing(from: &Path, to: &Path) -> std::io::Result<()> {
    retry_while_shared(|| tokio::fs::rename(from, to)).await
}

/// Runs `operation`, and on Windows runs it again a few times while it
/// fails because another program holds the file open without sharing it
/// (an antivirus scanner or an indexer): "access denied" (5) or a sharing
/// violation (32), which pass in a moment.
async fn retry_while_shared<T, F, Fut>(mut operation: F) -> std::io::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::io::Result<T>>,
{
    #[cfg(windows)]
    {
        const RETRIES: u32 = 5;
        let mut attempt = 0;
        loop {
            match operation().await {
                Err(error) if attempt < RETRIES && matches!(error.raw_os_error(), Some(5 | 32)) => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(10 << attempt)).await;
                }
                done => return done,
            }
        }
    }
    #[cfg(not(windows))]
    operation().await
}

/// Flushes `directory`'s entries to disk (Unix), so a file renamed into it
/// stays renamed after a crash. Windows can't flush a directory, and a
/// file system that doesn't support it is skipped.
async fn sync_directory(directory: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let directory = directory.to_path_buf();
        let synced =
            tokio::task::spawn_blocking(move || std::fs::File::open(&directory)?.sync_all())
                .await
                .map_err(std::io::Error::other)?;
        match synced {
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::InvalidInput | std::io::ErrorKind::Unsupported
                ) =>
            {
                Ok(())
            }
            synced => synced,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        Ok(())
    }
}

/// The most buffered event bytes kept while writes fail. The oldest lines
/// beyond it are dropped (and counted in a warning).
const MAX_BUFFERED_EVENT_BYTES: usize = 64 * 1024 * 1024;

/// The event log's buffered writer. Lines wait in `pending` and are
/// appended to the file, in order, by one writer at a time.
struct EventWriter {
    path: PathBuf,
    workflow_run_id: uuid::Uuid,
    policy: EventLog,
    interval: Duration,
    /// The run's images. A line names an image rather than holding it.
    images: Arc<images::ImageFiles>,
    pending: parking_lot::Mutex<PendingEvents>,
    /// The open file. Held while writing, so writes keep their order.
    file: tokio::sync::Mutex<Option<tokio::fs::File>>,
    /// A timed flush is scheduled.
    timer: AtomicBool,
}

#[derive(Default)]
struct PendingEvents {
    bytes: Vec<u8>,
    last_flush: Option<tokio::time::Instant>,
    /// Writes are failing (warned once until they succeed again).
    failing: bool,
    /// Lines dropped since the last warning.
    dropped: usize,
}

impl EventWriter {
    fn new(
        path: PathBuf,
        workflow_run_id: uuid::Uuid,
        policy: EventLog,
        interval: Duration,
        images: Arc<images::ImageFiles>,
    ) -> Self {
        Self {
            path,
            workflow_run_id,
            policy,
            interval,
            images,
            pending: parking_lot::Mutex::new(PendingEvents::default()),
            file: tokio::sync::Mutex::new(None),
            timer: AtomicBool::new(false),
        }
    }

    async fn append(
        self: &Arc<Self>,
        agent_name: &str,
        agent_run_id: uuid::Uuid,
        event: &arko_agents::runtime::event::AgentEvent,
    ) {
        #[derive(serde::Serialize)]
        struct PersistedEvent<'a> {
            timestamp: chrono::DateTime<chrono::Utc>,
            workflow_run_id: uuid::Uuid,
            agent_name: &'a str,
            agent_run_id: uuid::Uuid,
            event: &'a arko_agents::runtime::event::AgentEvent,
        }

        if !self.policy.records(event) {
            return;
        }
        let mut line = match self.images.with(|| {
            serde_json::to_vec(&PersistedEvent {
                timestamp: chrono::Utc::now(),
                workflow_run_id: self.workflow_run_id,
                agent_name,
                agent_run_id,
                event,
            })
        }) {
            Ok(line) => line,
            Err(error) => {
                log::warn!(
                    "Could not serialize an event of agent '{agent_name}' for the event log: {error}"
                );
                return;
            }
        };
        line.push(b'\n');
        let due = {
            let mut pending = self.pending.lock();
            pending.bytes.extend_from_slice(&line);
            if pending.bytes.len() > MAX_BUFFERED_EVENT_BYTES {
                drop_oldest_lines(&mut pending);
            }
            pending
                .last_flush
                .is_none_or(|last| last.elapsed() >= self.interval)
        };
        if due || is_urgent(event) {
            self.flush().await;
        } else {
            self.schedule();
        }
    }

    /// Appends `record`, a line that isn't an agent event (`what` names it
    /// in a warning), to be written at once when a Tokio runtime is running
    /// and otherwise with the next write.
    fn append_record(self: &Arc<Self>, record: &impl serde::Serialize, what: &str) {
        let mut line = match self.images.with(|| serde_json::to_vec(record)) {
            Ok(line) => line,
            Err(error) => {
                log::warn!("Could not serialize {what} for the event log: {error}");
                return;
            }
        };
        line.push(b'\n');
        {
            let mut pending = self.pending.lock();
            pending.bytes.extend_from_slice(&line);
            if pending.bytes.len() > MAX_BUFFERED_EVENT_BYTES {
                drop_oldest_lines(&mut pending);
            }
        }
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let writer = self.clone();
            runtime.spawn(async move { writer.flush().await });
        }
    }

    /// Flushes once the interval has passed, unless a flush is already
    /// scheduled.
    fn schedule(self: &Arc<Self>) {
        if self.timer.swap(true, Ordering::AcqRel) {
            return;
        }
        let writer: Weak<Self> = Arc::downgrade(self);
        let interval = self.interval;
        tokio::spawn(async move {
            tokio::time::sleep(interval).await;
            if let Some(writer) = writer.upgrade() {
                writer.timer.store(false, Ordering::Release);
                writer.flush().await;
            }
        });
    }

    /// Appends the buffered lines to the file. On failure the lines not
    /// written stay first in the buffer, for the next flush.
    async fn flush(&self) {
        use tokio::io::AsyncWriteExt;
        let mut file = self.file.lock().await;
        let bytes = std::mem::take(&mut self.pending.lock().bytes);
        if bytes.is_empty() {
            return;
        }
        let mut written = 0;
        let result = async {
            if file.is_none() {
                if let Some(directory) = self.path.parent() {
                    tokio::fs::create_dir_all(directory).await?;
                }
                *file = Some(
                    tokio::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&self.path)
                        .await?,
                );
            }
            let handle = file.as_mut().expect("opened above");
            while written < bytes.len() {
                match handle.write(&bytes[written..]).await? {
                    0 => return Err(std::io::ErrorKind::WriteZero.into()),
                    count => written += count,
                }
            }
            handle.flush().await
        }
        .await;
        let mut pending = self.pending.lock();
        pending.last_flush = Some(tokio::time::Instant::now());
        match result {
            Ok(()) => {
                if std::mem::take(&mut pending.failing) {
                    log::info!("Writing the event log {} again", self.path.display());
                }
            }
            Err(error) => {
                // Reopened at the next flush.
                *file = None;
                let mut rest = bytes[written..].to_vec();
                rest.append(&mut pending.bytes);
                pending.bytes = rest;
                if !std::mem::replace(&mut pending.failing, true) {
                    log::warn!(
                        "Could not write the event log {}: {error}. Events stay buffered and are written when writing succeeds again. The agents go on.",
                        self.path.display()
                    );
                }
            }
        }
        if pending.dropped > 0 {
            log::warn!(
                "The event log {} dropped {} of its oldest buffered events while writes failed",
                self.path.display(),
                std::mem::take(&mut pending.dropped)
            );
        }
    }
}

/// Drops the oldest whole lines until the buffer fits
/// [`MAX_BUFFERED_EVENT_BYTES`].
fn drop_oldest_lines(pending: &mut PendingEvents) {
    let excess = pending.bytes.len() - MAX_BUFFERED_EVENT_BYTES;
    let mut cut = 0;
    let mut lines = 0;
    while cut < excess {
        match pending.bytes[cut..].iter().position(|byte| *byte == b'\n') {
            Some(end) => {
                cut += end + 1;
                lines += 1;
            }
            None => {
                cut = pending.bytes.len();
                lines += 1;
            }
        }
    }
    pending.bytes.drain(..cut);
    pending.dropped += lines;
}

impl Drop for EventWriter {
    /// Writes what is still buffered, blocking briefly: the last events of
    /// a run whose workflow is dropped before a timed flush.
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.pending.get_mut().bytes);
        if bytes.is_empty() {
            return;
        }
        use std::io::Write;
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut file| file.write_all(&bytes));
        if let Err(error) = result {
            log::warn!(
                "Could not write the last events to the event log {}: {error}",
                self.path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arko_agents::runtime::event::AgentEvent;

    fn lines(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn manager(config: PersistenceConfig) -> PersistenceManager {
        PersistenceManager::new(config, uuid::Uuid::new_v4()).unwrap()
    }

    /// A manager made directly (not through `WorkflowBuilder::build`)
    /// rejects keeping zero checkpoints too, with the same error (it once
    /// kept every checkpoint silently).
    #[test]
    fn a_manager_rejects_keeping_zero_checkpoints() {
        let workspace = crate::test_support::Workspace::new();
        let error = PersistenceManager::new(
            PersistenceConfig::new(&workspace.0).keep_last_checkpoints(0),
            uuid::Uuid::new_v4(),
        )
        .err()
        .unwrap();
        assert!(
            matches!(&error, WorkflowError::BuilderInvalid(message)
                if message == "keep_last_checkpoints needs at least 1. Every checkpoint is kept by default"),
            "{error}"
        );
        let kept = manager(PersistenceConfig::new(&workspace.0).keep_last_checkpoints(2));
        assert_eq!(
            kept.config().retain_checkpoints,
            CheckpointRetention::KeepLast(NonZeroUsize::new(2).unwrap())
        );
    }

    fn state_change() -> AgentEvent {
        AgentEvent::TaskStateChanged(arko_agents::TaskState::InProgress)
    }

    /// Fragments are buffered and written at the next state change (or the
    /// interval). A lifecycle-only log leaves them out.
    #[tokio::test]
    async fn events_are_buffered_until_a_state_change_or_the_interval() {
        for policy in [EventLog::Full, EventLog::LifecycleOnly] {
            let workspace = crate::test_support::Workspace::new();
            let manager = manager(
                PersistenceConfig::new(&workspace.0)
                    .event_log(policy)
                    // Long enough that only the state change flushes here.
                    .event_flush_interval(Duration::from_secs(3600)),
            );
            let log = workspace.0.join("events.jsonl");
            let run = uuid::Uuid::new_v4();
            // The first event is written (nothing was written yet).
            manager
                .append_event("coder", run, &AgentEvent::TaskStreamEnd)
                .await;
            assert_eq!(lines(&log).len(), 1);
            for chunk in ["a", "b", "c"] {
                manager
                    .append_event(
                        "coder",
                        run,
                        &AgentEvent::AssistantResponseChunk(chunk.into()),
                    )
                    .await;
            }
            assert_eq!(lines(&log).len(), 1, "buffered");
            manager.append_event("coder", run, &state_change()).await;
            let written = lines(&log);
            let fragments = if policy == EventLog::Full { 3 } else { 0 };
            assert_eq!(written.len(), 2 + fragments);
            assert_eq!(
                written.last().unwrap()["event"]["TaskStateChanged"],
                "InProgress"
            );
        }
    }

    /// A buffered event is written once the interval passes, without a
    /// later event.
    #[tokio::test]
    async fn buffered_events_are_written_after_the_interval() {
        let workspace = crate::test_support::Workspace::new();
        let manager = manager(
            PersistenceConfig::new(&workspace.0).event_flush_interval(Duration::from_millis(20)),
        );
        let log = workspace.0.join("events.jsonl");
        let run = uuid::Uuid::new_v4();
        manager
            .append_event("coder", run, &AgentEvent::TaskStreamEnd)
            .await;
        manager
            .append_event("coder", run, &AgentEvent::ReasoningChunk("thinking".into()))
            .await;
        tokio::time::timeout(Duration::from_secs(10), async {
            while lines(&log).len() < 2 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the timed flush wrote the event");
    }

    /// A write that fails is warned about and retried with the next flush:
    /// no event is lost and nothing fails.
    #[tokio::test]
    async fn a_failed_write_keeps_the_events_for_the_next_flush() {
        let workspace = crate::test_support::Workspace::new();
        // The log's path is a directory, so opening it fails.
        let log = workspace.0.join("events.jsonl");
        std::fs::create_dir(&log).unwrap();
        let manager = manager(PersistenceConfig::new(&workspace.0));
        let run = uuid::Uuid::new_v4();
        manager
            .append_event("coder", run, &AgentEvent::TaskStreamEnd)
            .await;
        manager.append_event("coder", run, &state_change()).await;
        std::fs::remove_dir(&log).unwrap();
        manager.flush_events().await;
        assert_eq!(lines(&log).len(), 2);
    }

    /// Dropping the manager writes what is still buffered.
    #[tokio::test]
    async fn dropping_the_manager_writes_the_buffered_events() {
        let workspace = crate::test_support::Workspace::new();
        let manager = manager(
            PersistenceConfig::new(&workspace.0).event_flush_interval(Duration::from_secs(3600)),
        );
        let run = uuid::Uuid::new_v4();
        manager
            .append_event("coder", run, &AgentEvent::TaskStreamEnd)
            .await;
        manager
            .append_event(
                "coder",
                run,
                &AgentEvent::AssistantResponseChunk("x".into()),
            )
            .await;
        drop(manager);
        assert_eq!(lines(&workspace.0.join("events.jsonl")).len(), 2);
    }

    /// `write_replacing` replaces a file whole (creating its directory), and
    /// a failed write leaves the target as it was, without a temporary
    /// file.
    #[tokio::test]
    async fn atomic_writes_replace_whole_files_and_clean_up_after_failures() {
        let workspace = crate::test_support::Workspace::new();
        let file = workspace.0.join("nested/state.json");
        write_replacing(&file, b"first", true).await.unwrap();
        write_replacing(&file, b"second", true).await.unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "second");
        let names = |directory: &Path| {
            let mut names = std::fs::read_dir(directory)
                .unwrap()
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            names.sort();
            names
        };
        assert_eq!(names(&workspace.0.join("nested")), ["state.json"]);

        // A directory can't be replaced by a file.
        let occupied = workspace.0.join("occupied");
        std::fs::create_dir(&occupied).unwrap();
        assert!(write_replacing(&occupied, b"data", true).await.is_err());
        assert!(occupied.is_dir());
        assert_eq!(names(&workspace.0), ["nested", "occupied"]);
    }

    /// Checkpoints point `latest-checkpoint` at themselves through atomic
    /// writes: no temporary file is left.
    #[tokio::test]
    async fn the_latest_checkpoint_pointer_is_written_atomically() {
        let workspace = crate::test_support::Workspace::new();
        let workflow = crate::Workflow::builder()
            .name("pointer-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("coder", crate::test_support::agent_config("Coder"))
            .persistence(PersistenceConfig::new(workspace.0.join(".run")))
            .build()
            .unwrap();
        workflow.checkpoint().await.unwrap();
        let run = workspace.0.join(".run");
        let latest = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
        assert!(run.join(latest.trim()).is_file());
        assert!(
            std::fs::read_dir(&run)
                .unwrap()
                .flatten()
                .all(|entry| !entry.file_name().to_string_lossy().ends_with(".tmp"))
        );
    }

    /// A manager opening a run directory removes what an earlier process
    /// left unfinished: temporary files, and live files no checkpoint
    /// refers to. The live files the live checkpoint refers to stay.
    #[test]
    fn leftovers_of_an_earlier_process_are_removed() {
        let workspace = crate::test_support::Workspace::new();
        let run = workspace.0.join(".run");
        let (kept, orphan) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        for id in [kept, orphan] {
            std::fs::create_dir_all(run.join(format!("{TASKS_DIR}/{id}"))).unwrap();
            std::fs::write(run.join(live_task_file(id)), "{}").unwrap();
        }
        let leftovers = [
            run.join("checkpoint-1.json.7-0.tmp"),
            run.join(format!("{}.7-1.tmp", live_task_file(kept))),
        ];
        for file in &leftovers {
            std::fs::write(file, "half").unwrap();
        }
        let version = run.join(format!("{TASKS_DIR}/{kept}/0123.json"));
        std::fs::write(&version, "{}").unwrap();
        std::fs::write(
            run.join(LIVE_CHECKPOINT_FILE),
            serde_json::json!({"agent_runs": [{"task_file": live_task_file(kept)}]}).to_string(),
        )
        .unwrap();
        let opened = manager(PersistenceConfig::new(&run));
        assert!(run.join(live_task_file(kept)).is_file());
        assert!(version.is_file());
        assert!(!run.join(live_task_file(orphan)).exists());
        for file in &leftovers {
            assert!(!file.exists(), "{}", file.display());
        }
        assert!(opened.live.lock().referenced.contains(&kept));

        // An unreadable live checkpoint may refer to any live file.
        std::fs::write(run.join(LIVE_CHECKPOINT_FILE), "{").unwrap();
        std::fs::write(run.join(live_task_file(orphan)), "{}").unwrap();
        drop(opened);
        let _reopened = manager(PersistenceConfig::new(&run));
        assert!(run.join(live_task_file(orphan)).is_file());
    }

    /// Only a task's live file is taken for one.
    #[test]
    fn live_files_are_recognized_by_path() {
        let id = uuid::Uuid::new_v4();
        assert_eq!(live_file_task(&live_task_file(id)), Some(id));
        assert_eq!(live_file_task(&format!("agents/{id}/0123.json")), None);
        assert_eq!(live_file_task(&format!("agents/{id}.json")), None);
    }

    #[test]
    fn the_oldest_lines_are_dropped_past_the_buffer_limit() {
        let mut pending = PendingEvents {
            bytes: [
                b"first\n".as_slice(),
                &vec![b'x'; MAX_BUFFERED_EVENT_BYTES - 10],
                b"\nlast\n",
            ]
            .concat(),
            ..PendingEvents::default()
        };
        drop_oldest_lines(&mut pending);
        assert_eq!(pending.dropped, 1);
        assert_eq!(pending.bytes.len(), MAX_BUFFERED_EVENT_BYTES - 4);
        assert!(pending.bytes.ends_with(b"x\nlast\n"));
    }
}
