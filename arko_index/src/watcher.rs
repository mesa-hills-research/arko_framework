use crate::{
    ArkoIndex, BuildSpec, Result,
    error::invalid,
    source,
    storage::{INTERRUPTED, decode},
};
use notify::Watcher;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Component, Path, PathBuf},
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::{
    sync::{broadcast, watch},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum FileChange {
    Added { path: String },
    Deleted { path: String },
    Modified { path: String },
    Moved { from: String, to: String },
}
/// Reconcile final states rather than replaying raw filesystem events. Rename
/// inference requires a unique deleted/added content match; duplicates stay explicit.
pub fn coalesce(old: &BTreeMap<String, String>, new: &BTreeMap<String, String>) -> Vec<FileChange> {
    let mut deleted: BTreeMap<_, _> = old
        .iter()
        .filter(|(p, _)| !new.contains_key(*p))
        .map(|(p, h)| (p.clone(), h.clone()))
        .collect();
    let mut added: BTreeMap<_, _> = new
        .iter()
        .filter(|(p, _)| !old.contains_key(*p))
        .map(|(p, h)| (p.clone(), h.clone()))
        .collect();
    let mut out = vec![];
    for (path, hash) in old {
        if let Some(current) = new.get(path)
            && current != hash
        {
            out.push(FileChange::Modified { path: path.clone() });
        }
    }
    for (from, hash) in deleted.clone() {
        let from_count = deleted.values().filter(|h| **h == hash).count();
        let matches: Vec<_> = added
            .iter()
            .filter(|(_, h)| **h == hash)
            .map(|(p, _)| p.clone())
            .collect();
        if from_count == 1 && matches.len() == 1 {
            let to = matches[0].clone();
            deleted.remove(&from);
            added.remove(&to);
            out.push(FileChange::Moved { from, to });
        }
    }
    out.extend(deleted.into_keys().map(|path| FileChange::Deleted { path }));
    out.extend(added.into_keys().map(|path| FileChange::Added { path }));
    out
}

/// Settings of [`ArkoIndex::watch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatchOptions {
    /// How long the source must be quiet after a file-system event before it
    /// is scanned. 500 ms by default.
    pub debounce: Duration,
    /// The interval of full scans that catch events the file system did not
    /// report. 30 s by default; `None` scans only at start and after events.
    pub reconcile: Option<Duration>,
    /// Resume an update whose process stopped mid-run (paused with the
    /// problem "process interrupted"). On by default. Jobs paused with
    /// [`ArkoIndex::pause`] or by a usage limit are left for the host.
    pub resume_interrupted: bool,
    /// Opt-in: resume an update blocked by transient provider failures (rate
    /// limits, server errors or capacity waits that ran out of attempts)
    /// this long after the watcher found it blocked. `None` by default.
    pub retry_blocked_after: Option<Duration>,
}
impl Default for WatchOptions {
    fn default() -> Self {
        Self {
            debounce: Duration::from_millis(500),
            reconcile: Some(Duration::from_secs(30)),
            resume_interrupted: true,
            retry_blocked_after: None,
        }
    }
}
impl WatchOptions {
    /// The defaults with another debounce period.
    pub fn new(debounce: Duration) -> Self {
        Self {
            debounce,
            ..Default::default()
        }
    }
    pub fn with_reconcile(mut self, reconcile: Option<Duration>) -> Self {
        self.reconcile = reconcile;
        self
    }
    pub fn with_resume_interrupted(mut self, resume: bool) -> Self {
        self.resume_interrupted = resume;
        self
    }
    pub fn with_retry_blocked_after(mut self, delay: Option<Duration>) -> Self {
        self.retry_blocked_after = delay;
        self
    }
    fn validate(&self) -> Result<()> {
        if self.debounce.is_zero()
            || self.reconcile.is_some_and(|d| d.is_zero())
            || self.retry_blocked_after.is_some_and(|d| d.is_zero())
        {
            return Err(invalid("watch intervals must be positive"));
        }
        Ok(())
    }
}

/// What a watcher did or found; see [`WatchHandle::subscribe`]. Each is also
/// logged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum WatchEvent {
    /// A scan of the source finished: `files` files, of which `changes`
    /// differ from the active generation.
    Scanned { files: usize, changes: usize },
    /// The watcher started an update.
    UpdateStarted { job: String },
    /// The watcher resumed an update: `reason` is `interrupted` (its process
    /// stopped mid-run) or `blocked` (see
    /// [`WatchOptions::retry_blocked_after`]).
    Resumed { job: String, reason: String },
    /// A paused or blocked update the watcher doesn't resume; no other update
    /// starts until the host resumes or cancels it. Reported once per job,
    /// phase and problem.
    Held {
        job: String,
        phase: String,
        problem: Option<String>,
    },
    /// A scan, update or resume failed; the watcher keeps running.
    Failed { error: String },
}

pub struct WatchHandle {
    stop: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
    events: broadcast::Sender<WatchEvent>,
}
impl WatchHandle {
    /// Receives the watcher's events from now on. A receiver that falls more
    /// than 64 events behind misses the oldest.
    pub fn subscribe(&self) -> broadcast::Receiver<WatchEvent> {
        self.events.subscribe()
    }
    pub async fn stop(self) -> Result<()> {
        self.stop.send_replace(true);
        self.task
            .await
            .map_err(|e| crate::Error::Worker(e.to_string()))
    }
}

/// Decides which file-system events can change the index: not those under a
/// directory the scan skips, by name or by ignore rules.
struct EventFilter {
    root: PathBuf,
    skipped: Vec<String>,
    /// Directories the last scan did not enter (ignored, skipped or not
    /// followed), as absolute paths.
    pruned: RwLock<HashSet<PathBuf>>,
}
impl EventFilter {
    fn ignores(&self, path: &Path) -> bool {
        let Ok(relative) = path.strip_prefix(&self.root) else {
            return false;
        };
        if relative.components().any(|component| {
            matches!(component, Component::Normal(name)
                if self.skipped.iter().any(|skipped| name == skipped.as_str()))
        }) {
            return true;
        }
        let Ok(pruned) = self.pruned.read() else {
            return false;
        };
        path.ancestors()
            .take_while(|ancestor| *ancestor != self.root)
            .any(|ancestor| pruned.contains(ancestor))
    }
    fn relevant(&self, event: &notify::Result<notify::Event>) -> bool {
        match event {
            Ok(event) if event.kind.is_access() => false,
            Ok(event) if !event.paths.is_empty() => {
                !event.paths.iter().all(|path| self.ignores(path))
            }
            // Errors and events without paths may hide changes.
            _ => true,
        }
    }
}

/// The directories inside `root` that a scan which entered `visited`
/// (root-relative, `/`-separated; `""` is the root) did not enter.
fn pruned_directories(root: &Path, visited: &[String]) -> HashSet<PathBuf> {
    let entered: HashSet<PathBuf> = visited
        .iter()
        .map(|dir| {
            if dir.is_empty() {
                root.to_owned()
            } else {
                root.join(dir)
            }
        })
        .collect();
    let mut pruned = HashSet::new();
    for dir in &entered {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) && !entered.contains(&entry.path()) {
                pruned.insert(entry.path());
            }
        }
    }
    pruned
}

/// The newest unfinished job of an index: `(job, phase, problem, retryable)`.
type Pending = Option<(String, String, Option<String>, bool)>;

impl ArkoIndex {
    /// Watches the index root and keeps the index up to date: after each
    /// burst of file-system events and at every reconcile interval, it scans
    /// the source with the saved spec and starts an update when any file
    /// changed. Events under directories the scan skips (by name or by
    /// ignore rules) are ignored. Updates stay serialized: while one is
    /// running the watcher waits for it; one whose process stopped is
    /// resumed ([`WatchOptions::resume_interrupted`]); other paused or
    /// blocked updates hold further updates and are reported as
    /// [`WatchEvent::Held`].
    pub async fn watch(&self, index: &str, options: WatchOptions) -> Result<WatchHandle> {
        options.validate()?;
        let index = index.to_owned();
        let key = index.clone();
        let spec: BuildSpec = self
            .inner
            .db
            .call(move |c| {
                let s: String =
                    c.query_row("SELECT spec FROM indexes WHERE id=?1", [key], |r| r.get(0))?;
                decode(&s)
            })
            .await?;
        let filter = Arc::new(EventFilter {
            root: spec.root.clone(),
            skipped: spec.skip_directories.clone(),
            pruned: RwLock::new(HashSet::new()),
        });
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let events_filter = filter.clone();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if events_filter.relevant(&event) {
                    let _ = tx.try_send(());
                }
            })
            .map_err(|e| invalid(e.to_string()))?;
        watcher
            .watch(&spec.root, notify::RecursiveMode::Recursive)
            .map_err(|e| invalid(e.to_string()))?;
        let (stop, mut stopped) = watch::channel(false);
        let (events, _) = broadcast::channel(64);
        let this = self.clone();
        let sender = events.clone();
        let task = tokio::spawn(async move {
            let _watcher = watcher;
            let mut watching = Watching {
                index: this,
                id: index,
                spec,
                options: options.clone(),
                filter,
                events: sender,
                reported: HashSet::new(),
                blocked_since: HashMap::new(),
                unresumable: HashSet::new(),
            };
            let mut interval = options.reconcile.map(tokio::time::interval);
            // The first check scans at once, catching changes made while no
            // watcher ran.
            let mut due = Some(Instant::now());
            loop {
                let timer = async {
                    match due {
                        Some(t) => tokio::time::sleep_until(t).await,
                        None => std::future::pending().await,
                    }
                };
                let tick = async {
                    match &mut interval {
                        Some(interval) => {
                            interval.tick().await;
                        }
                        None => std::future::pending().await,
                    }
                };
                tokio::select! {
                    _ = stopped.changed() => break,
                    event = rx.recv() => {
                        if event.is_none() { break; }
                        due = Some(Instant::now() + options.debounce);
                    }
                    _ = tick => {
                        if due.is_none() { due = Some(Instant::now() + options.debounce); }
                    }
                    _ = timer => due = watching.check().await,
                }
            }
        });
        Ok(WatchHandle { stop, task, events })
    }
}

struct Watching {
    index: ArkoIndex,
    id: String,
    spec: BuildSpec,
    options: WatchOptions,
    filter: Arc<EventFilter>,
    events: broadcast::Sender<WatchEvent>,
    /// Held jobs already reported: (job, phase, problem).
    reported: HashSet<(String, String, Option<String>)>,
    /// When the watcher first found each job blocked by transient failures.
    blocked_since: HashMap<String, Instant>,
    /// Jobs the watcher could not resume; they are held from then on.
    unresumable: HashSet<String>,
}

impl Watching {
    fn emit(&self, event: WatchEvent) {
        match &event {
            WatchEvent::Scanned { .. } => {}
            WatchEvent::UpdateStarted { job } => {
                log::info!("index {}: source changed; started update {job}", self.id)
            }
            WatchEvent::Resumed { job, reason } => {
                log::info!("index {}: resumed update {job} ({reason})", self.id)
            }
            WatchEvent::Held {
                job,
                phase,
                problem,
            } => log::warn!(
                "index {}: update {job} is {phase}{}; no other update starts until it is resumed or cancelled",
                self.id,
                problem
                    .as_ref()
                    .map(|p| format!(" ({p})"))
                    .unwrap_or_default()
            ),
            WatchEvent::Failed { error } => log::warn!("index {}: watcher: {error}", self.id),
        }
        let _ = self.events.send(event);
    }

    /// One check after a debounce period or reconcile tick. Returns when the
    /// watcher must check again without a new event, if ever.
    async fn check(&mut self) -> Option<Instant> {
        let now = Instant::now();
        let recheck = Some(now + self.options.debounce);
        let index = self.id.clone();
        let pending: Pending = match self
            .index
            .inner
            .db
            .call(move |c| {
                let row: Option<(String, String, Option<String>)> = c
                    .query_row(
                        "SELECT id,phase,problem FROM jobs WHERE index_id=?1 AND phase IN ('snapshotting','chunking','embedding','paused','blocked') ORDER BY rowid DESC LIMIT 1",
                        [&index],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                    )
                    .optional()?;
                let Some((job, phase, problem)) = row else {
                    return Ok(None);
                };
                let retryable = phase == "blocked"
                    && c.query_row(
                        "SELECT json_extract(detail,'$.retryable') FROM events WHERE job=?1 AND kind='phase' ORDER BY sequence DESC LIMIT 1",
                        [&job],
                        |r| r.get::<_, Option<bool>>(0),
                    )
                    .optional()?
                    .flatten()
                    .unwrap_or(false);
                Ok(Some((job, phase, problem, retryable)))
            })
            .await
        {
            Ok(pending) => pending,
            Err(error) => {
                self.emit(WatchEvent::Failed {
                    error: format!("could not read the index status: {error}"),
                });
                return None;
            }
        };
        let Some((job, phase, problem, retryable)) = pending else {
            self.blocked_since.clear();
            return self.scan().await;
        };
        match phase.as_str() {
            // Another check after the update ends picks up edits made meanwhile.
            "snapshotting" | "chunking" | "embedding" => {
                // Resumed by the host: the watcher may resume it again later.
                self.unresumable.remove(&job);
                recheck
            }
            _ if self.unresumable.contains(&job) => {
                self.hold(job, phase, problem);
                None
            }
            "paused"
                if self.options.resume_interrupted && problem.as_deref() == Some(INTERRUPTED) =>
            {
                self.resume(&job, "interrupted").await
            }
            "blocked" if retryable && self.options.retry_blocked_after.is_some() => {
                let delay = self.options.retry_blocked_after.unwrap_or_default();
                let since = *self.blocked_since.entry(job.clone()).or_insert(now);
                if now >= since + delay {
                    self.blocked_since.remove(&job);
                    return self.resume(&job, "blocked").await;
                }
                self.hold(job, phase, problem);
                Some(since + delay)
            }
            _ => {
                self.hold(job, phase, problem);
                None
            }
        }
    }

    fn hold(&mut self, job: String, phase: String, problem: Option<String>) {
        if self
            .reported
            .insert((job.clone(), phase.clone(), problem.clone()))
        {
            self.emit(WatchEvent::Held {
                job,
                phase,
                problem,
            });
        }
    }

    async fn resume(&mut self, job: &str, reason: &str) -> Option<Instant> {
        match self.index.resume(job).await {
            Ok(()) => {
                self.reported.retain(|(held, _, _)| held != job);
                self.emit(WatchEvent::Resumed {
                    job: job.to_owned(),
                    reason: reason.to_owned(),
                });
                Some(Instant::now() + self.options.debounce)
            }
            Err(error) => {
                // Not retried on every event: the job is held for the host.
                self.unresumable.insert(job.to_owned());
                self.emit(WatchEvent::Failed {
                    error: format!("could not resume update {job}: {error}"),
                });
                None
            }
        }
    }

    /// Scans the source and starts an update when it differs from the active
    /// generation.
    async fn scan(&mut self) -> Option<Instant> {
        let index = self.id.clone();
        let old = match self
            .index
            .inner
            .db
            .call(move |c| {
                let mut map = BTreeMap::new();
                let mut statement = c.prepare(
                    "SELECT path,hash FROM files WHERE job=(SELECT active FROM indexes WHERE id=?1)",
                )?;
                for row in statement.query_map([index], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })? {
                    let (path, hash) = row?;
                    map.insert(path, hash);
                }
                Ok(map)
            })
            .await
        {
            Ok(old) => old,
            Err(error) => {
                self.emit(WatchEvent::Failed {
                    error: format!("could not read the active generation: {error}"),
                });
                return None;
            }
        };
        let spec = self.spec.clone();
        let scan = tokio::task::spawn_blocking(move || {
            source::scan(&spec).map(|manifest| {
                let pruned = pruned_directories(&spec.root, &manifest.directories);
                (manifest, pruned)
            })
        })
        .await;
        let manifest = match scan {
            Ok(Ok((manifest, pruned))) => {
                if let Ok(mut current) = self.filter.pruned.write() {
                    *current = pruned;
                }
                manifest
            }
            Ok(Err(error)) => {
                self.emit(WatchEvent::Failed {
                    error: format!("scan failed: {error}"),
                });
                return None;
            }
            Err(error) => {
                self.emit(WatchEvent::Failed {
                    error: format!("scan worker failed: {error}"),
                });
                return None;
            }
        };
        let files = manifest.files.len();
        let new = manifest
            .files
            .into_iter()
            .map(|s| (s.path, s.hash))
            .collect();
        let changes = coalesce(&old, &new).len();
        self.emit(WatchEvent::Scanned { files, changes });
        if changes == 0 {
            return None;
        }
        match self.index.start_update(&self.id).await {
            Ok(handle) => {
                self.emit(WatchEvent::UpdateStarted { job: handle.job });
                Some(Instant::now() + self.options.debounce)
            }
            Err(error) => {
                self.emit(WatchEvent::Failed {
                    error: format!("could not start an update: {error}"),
                });
                None
            }
        }
    }
}
