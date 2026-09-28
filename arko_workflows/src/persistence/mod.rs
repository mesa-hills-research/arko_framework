/// Checkpoint and resume support for workflows.
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::context::SharedContext;
use crate::error::WorkflowError;

pub mod state;

/// What the event log (`events.jsonl` in the run directory) records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EventLog {
    /// Every agent event (the default), including each streamed fragment
    /// of response text and reasoning, one line each: a long response
    /// writes thousands of lines.
    #[default]
    Full,
    /// Every event except the streamed fragments
    /// (`AssistantResponseChunk`, `ReasoningChunk`, and output-based tool
    /// calling's `ToolParamName`, `ToolParamValueChunk`, `ToolParamEnd`):
    /// state changes, tool calls and results, usage, errors, retries and
    /// the rest. The finished responses are in the task files
    /// (`agents/<task id>.json`); streamed reasoning a provider doesn't
    /// return with the response is only in a `Full` log.
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

/// An event written to disk at once: a state change, the end of a run, or
/// an error or recovery report.
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
    )
}

/// The default [`PersistenceConfig::event_flush_interval`]: one second.
pub const DEFAULT_EVENT_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Configuration for workflow persistence.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Directory where checkpoint files are stored.
    pub checkpoint_dir: PathBuf,
    /// Automatically checkpoint after each agent completion.
    pub auto_checkpoint: bool,
    /// Keep only the newest this many checkpoint files, deleting older
    /// ones after each save. `None` (the default) keeps every checkpoint.
    pub retain_checkpoints: Option<usize>,
    /// What the event log records; default [`EventLog::Full`].
    pub event_log: EventLog,
    /// Events are buffered and written at least this often; state changes,
    /// run ends and error reports are written at once, and so is the
    /// buffer before each checkpoint. Default
    /// [`DEFAULT_EVENT_FLUSH_INTERVAL`].
    pub event_flush_interval: Duration,
}

/// Manages checkpointing and restoration of workflow state.
#[derive(Clone)]
pub struct PersistenceManager {
    config: PersistenceConfig,
    durable_store: Arc<crate::durable::DurableStore>,
    write_lock: Arc<tokio::sync::Mutex<()>>,
    events: Arc<EventWriter>,
    /// Tasks whose file holds the recorded result's task: written by this
    /// manager, or loaded from their files on resume.
    saved_tasks: Arc<parking_lot::Mutex<HashSet<uuid::Uuid>>>,
    /// Checkpoint files this manager wrote, oldest first (for retention).
    written: Arc<parking_lot::Mutex<VecDeque<PathBuf>>>,
}

impl PersistenceConfig {
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            auto_checkpoint: true,
            retain_checkpoints: None,
            event_log: EventLog::default(),
            event_flush_interval: DEFAULT_EVENT_FLUSH_INTERVAL,
        }
    }

    /// Whether a checkpoint is saved after each finalized agent result
    /// (default `true`).
    pub fn auto_checkpoint(mut self, enabled: bool) -> Self {
        self.auto_checkpoint = enabled;
        self
    }

    /// Opt-in: keep only the newest `count` checkpoints (at least one),
    /// deleting older checkpoint files after each save. By default every
    /// checkpoint is kept.
    pub fn retain_checkpoints(mut self, count: usize) -> Self {
        self.retain_checkpoints = Some(count.max(1));
        self
    }

    /// What the event log records; see [`EventLog`].
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

/// The file of task `id`, relative to the run directory.
pub(crate) fn task_file(id: uuid::Uuid) -> String {
    format!("agents/{id}.json")
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

    pub fn new(config: PersistenceConfig, workflow_run_id: uuid::Uuid) -> Self {
        let durable_store = Arc::new(crate::durable::DurableStore::new(
            workflow_run_id,
            config.checkpoint_dir.clone(),
        ));
        let events = Arc::new(EventWriter::new(
            config.checkpoint_dir.join("events.jsonl"),
            workflow_run_id,
            config.event_log,
            config.event_flush_interval,
        ));
        Self {
            config,
            durable_store,
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
            events,
            saved_tasks: Arc::new(parking_lot::Mutex::new(HashSet::new())),
            written: Arc::new(parking_lot::Mutex::new(VecDeque::new())),
        }
    }

    pub fn durable_store(&self) -> Arc<crate::durable::DurableStore> {
        self.durable_store.clone()
    }

    /// Records that the files of `tasks` hold the recorded results' tasks
    /// (they were loaded from them).
    pub(crate) fn mark_tasks_saved(&self, tasks: impl IntoIterator<Item = uuid::Uuid>) {
        self.saved_tasks.lock().extend(tasks);
    }

    /// Save a checkpoint of the current workflow state.
    ///
    /// The checkpoint refers to each agent run's task file
    /// (`agents/<task id>.json`) instead of holding the task; a task not
    /// saved yet (one restored from a checkpoint that held it) is written
    /// first. Buffered events are written first as well. With
    /// [`PersistenceConfig::retain_checkpoints`], older checkpoint files
    /// beyond the count are then deleted.
    pub async fn save_checkpoint(&self, ctx: &SharedContext) -> Result<PathBuf, WorkflowError> {
        self.flush_events().await;
        let _guard = self.write_lock.lock().await;
        let results = ctx.agent_results.read().clone();
        for result in &results {
            if !self.saved_tasks.lock().contains(&result.agent_run_id) {
                self.write_task(&result.task).await?;
            }
        }
        let ws = state::WorkflowState {
            workflow_run_id: ctx.workflow_run_id,
            workflow_name: ctx.workflow_name.clone(),
            workflow_metadata: ctx.workflow_metadata.clone(),
            checkpoint_id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            memory_snapshot: ctx.memory.snapshot(),
            message_queues: ctx.messages.snapshot(),
            budget_spent: ctx.budget.total_spent(),
            cost_ledger: ctx.budget.snapshot_ledger(),
            freshness_snapshot: ctx
                .resource_tracker
                .as_ref()
                .map(|tracker| tracker.snapshot()),
            agent_results: Vec::new(),
            agent_runs: results.iter().map(state::AgentRunRecord::of).collect(),
            verification_records: ctx.verification_records.read().clone(),
            distillation_records: ctx.distillation_records.read().clone(),
            completed_agents: results
                .iter()
                .map(|r| state::SerializableAgentResult {
                    agent_name: r.agent_name.clone(),
                    summary: r.summary.clone(),
                })
                .collect(),
            cancellation: ctx.run_control.cancellation(),
            agent_registry: Some(ctx.registry.snapshot()),
        };

        std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        let path = self
            .config
            .checkpoint_dir
            .join(format!("checkpoint-{}.json", ws.checkpoint_id));
        let json = serde_json::to_string_pretty(&ws)?;
        let temporary = path.with_extension("json.tmp");
        write_synced(&temporary, json.as_bytes()).await?;
        tokio::fs::rename(&temporary, &path).await?;
        let latest = self.config.checkpoint_dir.join("latest-checkpoint");
        let latest_temporary = self.config.checkpoint_dir.join("latest-checkpoint.tmp");
        tokio::fs::write(
            &latest_temporary,
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default(),
        )
        .await?;
        tokio::fs::rename(&latest_temporary, &latest).await?;
        self.written.lock().push_back(path.clone());
        if let Some(keep) = self.config.retain_checkpoints {
            self.prune_checkpoints(keep).await;
        }
        log::info!("Checkpoint saved to {}", path.display());
        Ok(path)
    }

    /// Deletes checkpoint files beyond the newest `keep`: those of earlier
    /// processes (oldest first by modification time), then this manager's
    /// in the order it wrote them. A file that can't be deleted is logged
    /// and kept.
    async fn prune_checkpoints(&self, keep: usize) {
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
        let excess = ordered.len().saturating_sub(keep);
        for path in ordered.into_iter().take(excess) {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => self.written.lock().retain(|written| *written != path),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    self.written.lock().retain(|written| *written != path);
                }
                Err(error) => log::warn!(
                    "Could not delete old checkpoint {}: {error}",
                    path.display()
                ),
            }
        }
    }

    /// Records `event` in the event log, as [`PersistenceConfig::event_log`]
    /// selects. Events are buffered and written at least every
    /// [`PersistenceConfig::event_flush_interval`]; state changes, run ends
    /// and error reports are written at once. A failed write is logged as a
    /// warning and retried with the next write; it never stops an agent.
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

    pub async fn save_agent_task(
        &self,
        task: &arko_agents::Task,
    ) -> Result<PathBuf, WorkflowError> {
        let _guard = self.write_lock.lock().await;
        self.write_task(task).await
    }

    /// Writes `task` to its file; the caller holds the write lock. Until the
    /// write succeeds the task counts as not saved, so a checkpoint writes
    /// it again rather than refer to an older file.
    async fn write_task(&self, task: &arko_agents::Task) -> Result<PathBuf, WorkflowError> {
        self.saved_tasks.lock().remove(&task.id);
        let directory = self.config.checkpoint_dir.join("agents");
        tokio::fs::create_dir_all(&directory).await?;
        let path = self.config.checkpoint_dir.join(task_file(task.id));
        let temporary = directory.join(format!("{}.json.tmp", task.id));
        write_synced(&temporary, &serde_json::to_vec_pretty(task)?).await?;
        tokio::fs::rename(&temporary, &path).await?;
        self.saved_tasks.lock().insert(task.id);
        Ok(path)
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
        let ws: state::WorkflowState = serde_json::from_str(&content)?;
        Ok(ws)
    }
}

/// Writes `bytes` to `path` and flushes them to disk through the same handle.
/// Windows can only flush a handle opened for writing, so reopening the file
/// read-only to sync it fails there with "Access is denied".
async fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::File::create(path).await?;
    file.write_all(bytes).await?;
    file.sync_all().await
}

/// The most buffered event bytes kept while writes fail; the oldest lines
/// beyond it are dropped (and counted in a warning).
const MAX_BUFFERED_EVENT_BYTES: usize = 64 * 1024 * 1024;

/// The event log's buffered writer. Lines wait in `pending` and are
/// appended to the file, in order, by one writer at a time.
struct EventWriter {
    path: PathBuf,
    workflow_run_id: uuid::Uuid,
    policy: EventLog,
    interval: Duration,
    pending: parking_lot::Mutex<PendingEvents>,
    /// The open file; held while writing, so writes keep their order.
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
    ) -> Self {
        Self {
            path,
            workflow_run_id,
            policy,
            interval,
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
        let mut line = match serde_json::to_vec(&PersistedEvent {
            timestamp: chrono::Utc::now(),
            workflow_run_id: self.workflow_run_id,
            agent_name,
            agent_run_id,
            event,
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
                        "Could not write the event log {}: {error}. Events stay buffered and are written when writing succeeds again; the agents go on.",
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
        PersistenceManager::new(config, uuid::Uuid::new_v4())
    }

    fn state_change() -> AgentEvent {
        AgentEvent::TaskStateChanged(arko_agents::TaskState::InProgress)
    }

    /// Fragments are buffered and written at the next state change (or the
    /// interval); a lifecycle-only log leaves them out.
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

    /// A task whose latest write failed is written again by the next
    /// checkpoint, instead of the checkpoint referring to its older file.
    #[tokio::test]
    async fn a_task_whose_write_failed_is_written_by_the_next_checkpoint() {
        let workspace = crate::test_support::Workspace::new();
        let manager = manager(PersistenceConfig::new(&workspace.0));
        let config = crate::test_support::agent_config("Friendly Coder");
        let mut task = arko_agents::Task::new(config, "first".into(), workspace.0.clone(), None);
        manager.save_agent_task(&task).await.unwrap();
        assert!(manager.saved_tasks.lock().contains(&task.id));
        // The next write fails: its temporary file's path is a directory.
        let temporary = workspace.0.join(format!("agents/{}.json.tmp", task.id));
        std::fs::create_dir(&temporary).unwrap();
        task.initial_task_prompt = "second".into();
        assert!(manager.save_agent_task(&task).await.is_err());
        assert!(!manager.saved_tasks.lock().contains(&task.id));
        std::fs::remove_dir(&temporary).unwrap();
        manager.save_agent_task(&task).await.unwrap();
        let saved = std::fs::read_to_string(workspace.0.join(task_file(task.id))).unwrap();
        assert!(saved.contains("second"));
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
