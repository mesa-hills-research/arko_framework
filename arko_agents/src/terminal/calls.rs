//! Tool calls the host can move to the background: the calls running in the
//! foreground that can move, and the jobs that run the calls that moved.

use super::*;
use crate::stop::StopSource;
use crate::tool::{MoveSignal, MoveSource, ToolImage};
use futures::FutureExt;
use futures::future::BoxFuture;
use std::time::Instant;

/// How long a moved call has to return after its job is stopped before its
/// future is dropped.
pub(super) const CALL_STOP_GRACE: Duration = Duration::from_secs(3);

/// The most characters of a moved call's result its note carries. A longer
/// result is cut to its first and last parts, and its job's log has all of
/// it.
pub const CALL_NOTE_CHARS: usize = 30_000;

/// The first line of the result a call gets when it moves to the background
/// starts with this, followed by the job's ID.
pub const IN_THE_BACKGROUND: &str = "in the background as job ";

/// A call running in the foreground that its host may move to the
/// background, as [`TerminalManager::foreground_calls`] lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ForegroundCall {
    /// The task whose call it is.
    pub owner: Uuid,
    /// The runtime's ID of the call, as its events give it.
    pub tool_call_id: Uuid,
    pub tool_name: String,
    /// What a list shows.
    pub label: String,
    /// When it was registered, as it started running.
    pub started: Instant,
}

/// What a job runs, as [`JobInfo`] says.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum JobKind {
    /// A command, started by `terminal_start` or a host's tool.
    Command,
    /// A tool call that moved to the background.
    Call { tool_name: String },
}

/// A job, as [`TerminalManager::jobs`] and
/// [`TerminalManager::interrupted_at_open`] list it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct JobInfo {
    pub owner: Uuid,
    pub job_id: Uuid,
    /// The command, its [`StartOptions::label`], or a moved call's tool.
    pub label: String,
    /// When the job started. A moved call's job starts as it moves. A job
    /// restored from an earlier process has the time it was read.
    pub started: Instant,
    pub kind: JobKind,
    /// The call that made the job (the call that started the command, or
    /// the call that moved), when it was made in a call.
    pub tool_call_id: Option<Uuid>,
}

/// What a moved call returned, for [`TerminalManager::adopt_call`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CallOutput {
    /// The result's text, or the error's.
    pub text: String,
    /// The call failed, and `text` says why.
    pub is_error: bool,
    pub images: Vec<ToolImage>,
}

impl CallOutput {
    /// A result with `text`.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// A failure that `text` describes.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
            ..Self::default()
        }
    }

    /// This output with `images`.
    pub fn with_images(mut self, images: Vec<ToolImage>) -> Self {
        self.images = images;
        self
    }
}

/// A call in the foreground registry.
pub(super) struct Foreground {
    pub(super) call: ForegroundCall,
    pub(super) source: MoveSource,
    /// Tells this registration from a later one of the same call.
    pub(super) serial: u64,
}

/// A call registered with [`TerminalManager::hold_call`]. It is listed among
/// the calls that can move until it moves or the guard is dropped.
#[derive(Debug)]
pub struct HeldCall {
    manager: Weak<TerminalManager>,
    tool_call_id: Uuid,
    serial: u64,
    source: MoveSource,
    started: Instant,
}

impl HeldCall {
    /// The signal that fires when the host moves the call.
    pub fn signal(&self) -> MoveSignal {
        self.source.signal()
    }

    /// Whether the host moved the call.
    pub fn is_moved(&self) -> bool {
        self.source.is_moved()
    }

    /// Resolves once the host moves the call.
    pub async fn moved(&self) {
        self.source.signal().moved().await
    }

    /// When the call was registered.
    pub fn started(&self) -> Instant {
        self.started
    }
}

impl Drop for HeldCall {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade() {
            manager.unregister(self.tool_call_id, self.serial);
        }
    }
}

/// Who stopped a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StopCause {
    /// Its owner, with `terminal_control cancel` or [`TerminalManager::cancel`].
    Owner,
    /// The user, through [`TerminalManager::stop_for_user`].
    User,
    /// Its owner's run ended ([`TerminalManager::stop_owner_watchers`]).
    OwnerEnded,
}

/// A moved call's job, recorded and waiting for its call. See
/// [`TerminalManager::reserve_call`].
pub(crate) struct CallSlot {
    manager: Arc<TerminalManager>,
    data: Arc<Mutex<JobData>>,
    controls: mpsc::Receiver<Control>,
    cancellation: watch::Receiver<bool>,
    changed: watch::Sender<u64>,
    completed: watch::Sender<bool>,
    snapshot: TerminalSnapshot,
}

impl CallSlot {
    /// Runs `call` as the job, `stop` firing when the job is stopped, and
    /// returns the job's first snapshot.
    pub(crate) fn run(
        self,
        stop: StopSource,
        call: BoxFuture<'static, CallOutput>,
    ) -> TerminalSnapshot {
        let Self {
            manager,
            data,
            controls,
            cancellation,
            changed,
            completed,
            snapshot,
        } = self;
        let weak = Arc::downgrade(&manager);
        let config = manager.config.clone();
        let directory_lock = manager._directory_lock.clone();
        drop(manager);
        tokio::spawn(async move {
            run_call(
                weak,
                config,
                data,
                call,
                stop,
                controls,
                cancellation,
                changed,
            )
            .await;
            drop(directory_lock);
            completed.send_replace(true);
        });
        snapshot
    }
}

impl TerminalManager {
    /// Registers a call running in the foreground, which its host may move
    /// to the background with [`move_to_background`](Self::move_to_background).
    /// A runtime with background jobs
    /// ([`AgentRuntime::with_background_jobs`](crate::runtime::AgentRuntime::with_background_jobs))
    /// registers the calls of tools whose
    /// [`background_mode`](crate::tool::Tool::background_mode) isn't
    /// `Never`, once the execution middleware lets them run. The call is
    /// listed until it moves or the guard is dropped.
    pub fn hold_call(
        self: &Arc<Self>,
        owner: Uuid,
        tool_call_id: Uuid,
        tool_name: &str,
        label: String,
    ) -> HeldCall {
        let source = MoveSource::new();
        let started = Instant::now();
        let serial = {
            let mut foreground = self.foreground.lock().unwrap();
            foreground.1 += 1;
            let serial = foreground.1;
            foreground.0.push(Foreground {
                call: ForegroundCall {
                    owner,
                    tool_call_id,
                    tool_name: tool_name.to_string(),
                    label,
                    started,
                },
                source: source.clone(),
                serial,
            });
            serial
        };
        self.foreground_changed();
        HeldCall {
            manager: Arc::downgrade(self),
            tool_call_id,
            serial,
            source,
            started,
        }
    }

    fn unregister(&self, tool_call_id: Uuid, serial: u64) {
        let removed = {
            let mut foreground = self.foreground.lock().unwrap();
            let before = foreground.0.len();
            foreground
                .0
                .retain(|held| !(held.call.tool_call_id == tool_call_id && held.serial == serial));
            foreground.0.len() != before
        };
        if removed {
            self.foreground_changed();
        }
    }

    fn foreground_changed(&self) {
        self.foreground_changes
            .send_modify(|count| *count = count.wrapping_add(1));
    }

    /// The calls of every owner that can move to the background now, oldest
    /// first.
    pub fn foreground_calls(&self) -> Vec<ForegroundCall> {
        self.foreground
            .lock()
            .unwrap()
            .0
            .iter()
            .map(|held| held.call.clone())
            .collect()
    }

    /// Changes each time a call starts or stops being one that can move
    /// (see [`foreground_calls`](Self::foreground_calls)), for a host that
    /// shows which calls can move.
    pub fn watch_foreground_calls(&self) -> watch::Receiver<u64> {
        self.foreground_changes.subscribe()
    }

    /// Moves `owner`'s call `tool_call_id` to the background, or all of
    /// `owner`'s calls that can move with `None`. Returns how many moved. A
    /// manager [closed](Self::close) for its host's exit moves none.
    pub fn move_to_background(&self, owner: Uuid, tool_call_id: Option<Uuid>) -> usize {
        if self.closing.load(std::sync::atomic::Ordering::Acquire) {
            return 0;
        }
        let moved = {
            let mut foreground = self.foreground.lock().unwrap();
            let mut moved = 0;
            foreground.0.retain(|held| {
                let chosen = held.call.owner == owner
                    && tool_call_id.is_none_or(|id| id == held.call.tool_call_id);
                if chosen {
                    held.source.fire();
                    moved += 1;
                }
                !chosen
            });
            moved
        };
        if moved > 0 {
            self.foreground_changed();
        }
        moved
    }

    /// Sets after how long a running call of a
    /// [`Detach`](crate::tool::ToolBackground::Detach) tool moves to the
    /// background by itself, or `None` (the default) for never. Calls
    /// already running keep the time they started with.
    pub fn set_background_after(&self, after: Option<Duration>) {
        *self.background_after.lock().unwrap() = after.filter(|after| !after.is_zero());
    }

    /// After how long calls move to the background by themselves, as last
    /// set.
    pub fn background_after(&self) -> Option<Duration> {
        *self.background_after.lock().unwrap()
    }

    /// Runs the rest of `owner`'s call `tool_call_id`, which moved to the
    /// background, as a job, and returns the job's first snapshot. `stop`
    /// fires when the job is stopped (by its owner, the user, its owner's
    /// run ending, or the host's exit), and the future is dropped
    /// [3 seconds](CALL_STOP_GRACE) later if it hasn't ended. When the call
    /// ends its result goes to `owner` as a note, and its log holds all of
    /// it. Its images are saved in the job's directory, and the note names
    /// them. The job is listed with the [background jobs](Self::background_jobs)
    /// meanwhile, `terminal_status` reads its state, and the result once it
    /// ended.
    pub fn adopt_call(
        self: &Arc<Self>,
        owner: Uuid,
        tool_call_id: Uuid,
        tool_name: &str,
        label: String,
        stop: StopSource,
        call: BoxFuture<'static, CallOutput>,
    ) -> Result<TerminalSnapshot> {
        Ok(self
            .reserve_call(owner, tool_call_id, tool_name, label)?
            .run(stop, call))
    }

    /// The job for a moved call, recorded, before the call is handed over,
    /// so a runtime that can't move it keeps it in the foreground.
    pub(crate) fn reserve_call(
        self: &Arc<Self>,
        owner: Uuid,
        tool_call_id: Uuid,
        tool_name: &str,
        label: String,
    ) -> Result<CallSlot> {
        let mut jobs = self.jobs.lock().unwrap();
        if self.closing.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TerminalError::Invalid(
                "terminal manager is shut down".into(),
            ));
        }
        let id = Uuid::new_v4();
        let directory = self.config.directory.join(id.to_string());
        std::fs::create_dir(&directory)?;
        std::fs::File::create(directory.join("output.log"))?;
        let record = Record {
            job_id: id,
            owner,
            revision: 0,
            state: JobState::Running,
            input: None,
            exit_code: None,
            error: None,
            terminal: false,
            output_end: 0,
            log_truncated: false,
            held: false,
            tool_call_id: Some(tool_call_id),
            label: Some(label.clone()),
            call: Some(CallRecord {
                tool_name: tool_name.to_string(),
            }),
        };
        persist(&self.config, &record)?;
        let data = Arc::new(Mutex::new(JobData::new(record)));
        let snapshot =
            data.lock()
                .unwrap()
                .snapshot(&self.config, None, self.config.excerpt_bytes)?;
        let (controls, receiver) = mpsc::channel(4);
        let (cancel, cancellation) = watch::channel(false);
        let (changed, changes) = watch::channel(0);
        let (completed, done) = watch::channel(false);
        jobs.insert(
            id,
            Arc::new(Job {
                data: data.clone(),
                working_directory: None,
                command: label,
                started: Instant::now(),
                controls,
                cancel,
                changes,
                done,
            }),
        );
        Ok(CallSlot {
            manager: self.clone(),
            data,
            controls: receiver,
            cancellation,
            changed,
            completed,
            snapshot,
        })
    }

    /// The jobs running in the background (not held), of every owner, oldest
    /// first, with what each runs and the call it came from.
    pub fn jobs(&self) -> Vec<JobInfo> {
        let jobs = self.jobs.lock().unwrap();
        let mut running: Vec<JobInfo> = jobs
            .values()
            .filter_map(|job| {
                let data = job.data.lock().unwrap();
                (data.record.state == JobState::Running && !data.record.held)
                    .then(|| info(&data.record, job.command.clone(), job.started))
            })
            .collect();
        running.sort_by_key(|job| job.started);
        running
    }

    /// The jobs that were running when the last process using this
    /// directory stopped, which are now
    /// [`Interrupted`](JobState::Interrupted) records. They never run again,
    /// since their effects are unknown. A host can tell their owners.
    pub fn interrupted_at_open(&self) -> Vec<JobInfo> {
        self.interrupted.lock().unwrap().clone()
    }

    /// Whether `owner` has news it hasn't been given, a job's update (it
    /// finished, it waits for input) that went to no runtime because none
    /// was bound. A host can start a turn for it.
    pub fn has_news(&self, owner: Uuid) -> bool {
        self.jobs.lock().unwrap().values().any(|job| {
            let data = job.data.lock().unwrap();
            data.record.owner == owner && !data.record.held && data.last_event.is_some()
        })
    }

    /// Stops `owner`'s job `id` for the user. The job's update (the note of
    /// a moved call) says the user stopped it. It goes to the owner with its
    /// next turn, and isn't reported to
    /// [`subscribe_finished`](Self::subscribe_finished), so it doesn't wake
    /// an idle agent.
    pub fn stop_for_user(&self, owner: Uuid, id: Uuid) -> Result<()> {
        self.stop_job(owner, id, StopCause::User)
    }

    /// Stops `owner`'s jobs that only report to it, its moved calls, because
    /// its run ended (a worker finished, failed or was stopped) and their
    /// notes would reach no one. Its commands keep running, since it may
    /// have started a server for another agent. These stops aren't reported
    /// to [`subscribe_finished`](Self::subscribe_finished). Returns how many
    /// were stopped.
    pub fn stop_owner_watchers(&self, owner: Uuid) -> usize {
        let watchers: Vec<Uuid> = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .filter_map(|job| {
                let data = job.data.lock().unwrap();
                (data.record.owner == owner
                    && data.record.call.is_some()
                    && data.record.state == JobState::Running)
                    .then_some(data.record.job_id)
            })
            .collect();
        watchers
            .into_iter()
            .filter(|id| self.stop_job(owner, *id, StopCause::OwnerEnded).is_ok())
            .count()
    }

    /// Stops a running job, recording who stopped it.
    pub(super) fn stop_job(&self, owner: Uuid, id: Uuid, cause: StopCause) -> Result<()> {
        let job = self.owned(owner, id)?;
        {
            let mut data = job.data.lock().unwrap();
            if data.record.state.is_terminal() {
                return Err(TerminalError::Finished);
            }
            data.stopped_by.get_or_insert(cause);
        }
        job.cancel.send_replace(true);
        Ok(())
    }
}

/// A job as [`JobInfo`] gives it.
pub(super) fn info(record: &Record, label: String, started: Instant) -> JobInfo {
    JobInfo {
        owner: record.owner,
        job_id: record.job_id,
        label,
        started,
        kind: match &record.call {
            Some(call) => JobKind::Call {
                tool_name: call.tool_name.clone(),
            },
            None => JobKind::Command,
        },
        tool_call_id: record.tool_call_id,
    }
}

/// How a moved call's job ended.
enum Ending {
    /// The call returned. `stopped` when it did after its job was stopped.
    Returned { output: CallOutput, stopped: bool },
    /// It panicked.
    Panicked,
    /// It was stopped, and dropped when it didn't return in time.
    Dropped,
}

#[allow(clippy::too_many_arguments)]
async fn run_call(
    manager: Weak<TerminalManager>,
    config: TerminalConfig,
    data: Arc<Mutex<JobData>>,
    call: BoxFuture<'static, CallOutput>,
    stop: StopSource,
    mut controls: mpsc::Receiver<Control>,
    mut cancellation: watch::Receiver<bool>,
    changed: watch::Sender<u64>,
) {
    let moved_at = Instant::now();
    let mut call = std::panic::AssertUnwindSafe(call).catch_unwind();
    let finished = loop {
        tokio::select! {
            biased;
            output = &mut call => break Some(output),
            _ = cancellation.changed() => break None,
            control = controls.recv() => match control {
                Some(Control::Input { response, .. }) => {
                    let _ = response.send(Err(TerminalError::Invalid(
                        "this job runs a tool call, which takes no input".into(),
                    )));
                }
                // The manager is gone.
                None => break None,
            },
        }
    };
    let ending = match finished {
        Some(Ok(output)) => Ending::Returned {
            output,
            stopped: false,
        },
        Some(Err(_)) => Ending::Panicked,
        None => {
            let reason = match data.lock().unwrap().stopped_by {
                Some(StopCause::User) => "The user stopped the job.",
                Some(StopCause::OwnerEnded) => "The agent's run ended.",
                _ => "The job was stopped.",
            };
            stop.stop(reason);
            match tokio::time::timeout(CALL_STOP_GRACE, &mut call).await {
                Ok(Ok(output)) => Ending::Returned {
                    output,
                    stopped: true,
                },
                Ok(Err(_)) => Ending::Panicked,
                Err(_) => Ending::Dropped,
            }
        }
    };
    drop(call);
    let directory = config
        .directory
        .join(data.lock().unwrap().record.job_id.to_string());
    let (body, images) = match &ending {
        Ending::Returned { output, .. } => {
            let images = save_images(&directory, &output.images);
            (output.text.clone(), images)
        }
        _ => (String::new(), None),
    };
    let body = match images {
        Some(images) if body.is_empty() => images,
        Some(images) => format!("{body}\n\n{images}"),
        None => body,
    };
    let (logged, log_truncated) = write_log(&directory.join("output.log"), &body, &config);
    let final_notice = {
        let mut data = data.lock().unwrap();
        let stopped_by = data.stopped_by;
        let bytes = body.as_bytes();
        data.output = bytes[bytes.len().saturating_sub(config.output_buffer_bytes)..]
            .iter()
            .copied()
            .collect();
        data.record.output_end = bytes.len() as u64;
        data.record.log_truncated = log_truncated;
        data.record.revision += 1;
        let (state, error) = match &ending {
            Ending::Returned {
                stopped: false,
                output,
            } => (
                if output.is_error {
                    JobState::Failed
                } else {
                    JobState::Succeeded
                },
                None,
            ),
            Ending::Returned { stopped: true, .. } | Ending::Dropped => (
                JobState::Cancelled,
                match stopped_by {
                    Some(StopCause::User) => Some("stopped by the user".to_string()),
                    Some(StopCause::OwnerEnded) => {
                        Some("stopped because its agent's run ended".to_string())
                    }
                    _ => None,
                },
            ),
            Ending::Panicked => (
                JobState::Failed,
                Some("the call failed unexpectedly".to_string()),
            ),
        };
        data.record.state = state;
        data.record.error = error;
        if let Err(error) = &logged {
            log::warn!(
                "Terminal job {}: could not write the moved call's result: {error}",
                data.record.job_id
            );
        }
        if let Err(error) = persist(&config, &data.record) {
            data.record.state = JobState::Failed;
            data.record.error = Some(format!("could not persist terminal result: {error}"));
        }
        let tool_name = data
            .record
            .call
            .as_ref()
            .map(|call| call.tool_name.clone())
            .unwrap_or_default();
        let log_path = directory.join("output.log");
        data.note = Some(call_note(
            data.record.job_id,
            &tool_name,
            &ending,
            stopped_by,
            moved_at.elapsed(),
            &body,
            &log_path,
            log_truncated,
        ));
        let snapshot = data
            .snapshot(&config, None, config.excerpt_bytes)
            .expect("valid output window");
        data.last_event = Some(snapshot.clone());
        (data.record.owner, snapshot, stopped_by)
    };
    if let Some(manager) = manager.upgrade() {
        let (owner, snapshot, stopped_by) = final_notice;
        // The user's stop, and a stop because the owner's run ended, wake
        // no one.
        if !matches!(stopped_by, Some(StopCause::User | StopCause::OwnerEnded)) {
            let _ = manager.finished.send(JobFinished {
                owner,
                job_id: snapshot.job_id,
                state: snapshot.state,
            });
        }
        manager.deliver(owner, snapshot);
    }
    changed.send_replace(data.lock().unwrap().record.revision);
}

/// Writes `body` to the job's log, at most the configured log size.
/// Returns whether it was cut.
fn write_log(path: &Path, body: &str, config: &TerminalConfig) -> (std::io::Result<()>, bool) {
    let bytes = body.as_bytes();
    let keep = bytes.len().min(config.max_log_bytes as usize);
    (std::fs::write(path, &bytes[..keep]), keep < bytes.len())
}

/// Saves the images a moved call returned in its job's directory, and says
/// where they are, or `None` without images.
fn save_images(directory: &Path, images: &[ToolImage]) -> Option<String> {
    use base64::Engine as _;
    if images.is_empty() {
        return None;
    }
    let mut places = Vec::new();
    let mut unsaved = 0;
    for (index, image) in images.iter().enumerate() {
        match &image.image {
            arko_ai::Image::Url(url) => places.push(url.clone()),
            arko_ai::Image::Base64 { data, mime_type } => {
                let extension = match mime_type.as_str() {
                    "image/png" => "png",
                    "image/jpeg" | "image/jpg" => "jpg",
                    "image/gif" => "gif",
                    "image/webp" => "webp",
                    _ => "img",
                };
                let path = directory.join(format!("image-{}.{extension}", index + 1));
                let saved = base64::engine::general_purpose::STANDARD
                    .decode(data.trim())
                    .map_err(std::io::Error::other)
                    .and_then(|bytes| std::fs::write(&path, bytes));
                match saved {
                    Ok(()) => places.push(path.display().to_string()),
                    Err(error) => {
                        log::warn!("Could not save a moved call's image: {error}");
                        unsaved += 1;
                    }
                }
            }
        }
    }
    let mut sentence = match places.len() {
        0 => String::new(),
        1 if images.len() == 1 => format!("It returned an image, which is at {}.", places[0]),
        _ => format!(
            "It returned {} images, which are at {}.",
            images.len(),
            list(&places)
        ),
    };
    match unsaved {
        0 => {}
        1 => sentence.push_str(" One image couldn't be saved."),
        count => sentence.push_str(&format!(" {count} images couldn't be saved.")),
    }
    Some(sentence.trim_start().to_string())
}

/// `a`, `a and b`, `a, b and c`.
fn list(items: &[String]) -> String {
    match items.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => items.concat(),
    }
}

/// `40 s`, `4 min`, `2 h 5 min`.
pub(super) fn describe(duration: Duration) -> String {
    let seconds = duration.as_secs();
    match seconds {
        0..60 => format!("{seconds} s"),
        60..3600 => format!("{} min", seconds / 60),
        _ if seconds % 3600 < 60 => format!("{} h", seconds / 3600),
        _ => format!("{} h {} min", seconds / 3600, seconds % 3600 / 60),
    }
}

/// The note a moved call's job sends its owner when it ends.
#[allow(clippy::too_many_arguments)]
fn call_note(
    id: Uuid,
    tool_name: &str,
    ending: &Ending,
    stopped_by: Option<StopCause>,
    since_move: Duration,
    body: &str,
    log_path: &Path,
    log_truncated: bool,
) -> String {
    let ago = describe(since_move);
    let call = format!("your {tool_name} call");
    let stopped = |returned: bool| {
        let end = if returned {
            ". It returned this as it stopped."
        } else {
            ", so that call has no result."
        };
        match stopped_by {
            Some(StopCause::User) => {
                format!("The user stopped job {id} ({call}) before it ended{end}")
            }
            Some(StopCause::Owner) => {
                format!("Job {id} ({call}) was stopped before it ended, as you asked{end}")
            }
            _ => format!("Job {id} ({call}) was stopped before it ended{end}"),
        }
    };
    let (heading, with_body) = match ending {
        Ending::Returned {
            stopped: false,
            output,
        } if output.is_error => (
            format!(
                "Job {id} ended with an error. It ran {call}, which moved to the background {ago} ago. The error follows."
            ),
            true,
        ),
        Ending::Returned { stopped: false, .. } => (
            format!(
                "Job {id} ended. It ran {call}, which moved to the background {ago} ago. Its result follows."
            ),
            true,
        ),
        Ending::Returned { stopped: true, .. } => (stopped(!body.is_empty()), !body.is_empty()),
        Ending::Dropped => (stopped(false), false),
        Ending::Panicked => (
            format!(
                "Job {id} failed. Your {tool_name} call, which moved to the background {ago} ago, failed unexpectedly, so it has no result."
            ),
            false,
        ),
    };
    if !with_body {
        return heading;
    }
    let body = if body.is_empty() {
        "(no output)".to_string()
    } else {
        cut(body, log_path, log_truncated)
    };
    format!("{heading}\n\n{body}")
}

/// `body` as a note carries it: all of it, or its first and last parts with
/// a line saying how much was left out and where all of it is.
fn cut(body: &str, log_path: &Path, log_truncated: bool) -> String {
    let count = body.chars().count();
    if count <= CALL_NOTE_CHARS {
        return body.to_string();
    }
    let half = CALL_NOTE_CHARS / 2;
    let head_end = body
        .char_indices()
        .nth(half)
        .map_or(body.len(), |(at, _)| at);
    let tail_start = body
        .char_indices()
        .nth(count - half)
        .map_or(body.len(), |(at, _)| at);
    let whole = if log_truncated {
        format!("{} holds its first part", log_path.display())
    } else {
        format!("{} holds all of it", log_path.display())
    };
    format!(
        "{}\n\n[{} characters in the middle are left out. {whole}.]\n\n{}",
        &body[..head_end],
        count - 2 * half,
        &body[tail_start..]
    )
}

/// The result a call gets as it moves to the background, by the user or
/// (`after`) because it ran longer than that. `elapsed` is how long it ran,
/// `limit` its host time limit, and `terminal_tools` whether the agent has
/// `terminal_status` and `terminal_control`.
pub(crate) fn moved_result(
    job: &TerminalSnapshot,
    elapsed: Duration,
    after: Option<Duration>,
    limit: Option<Duration>,
    terminal_tools: bool,
) -> String {
    let why = match after {
        Some(after) => format!(
            "It ran longer than {}, so it moved to the background.",
            describe_words(after)
        ),
        None => "The user moved this call to the background.".to_string(),
    };
    let limit = match limit {
        Some(limit) => format!(
            " It goes on running until it ends or reaches its time limit of {}.",
            describe_words(limit)
        ),
        None => " It goes on running.".to_string(),
    };
    let tools = if terminal_tools {
        " terminal_status shows whether it still runs, and terminal_control waits for it or stops it."
    } else {
        ""
    };
    format!(
        "{IN_THE_BACKGROUND}{}, {:.1} s\n{why}{limit} Its result reaches you as a note when it ends.{tools}",
        job.job_id,
        elapsed.as_secs_f64()
    )
}

/// `10 minutes`, `1 hour`, `90 seconds`.
fn describe_words(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (count, unit) = match seconds {
        seconds if seconds >= 3600 && seconds % 3600 == 0 => (seconds / 3600, "hour"),
        seconds if seconds >= 60 && seconds % 60 == 0 => (seconds / 60, "minute"),
        seconds => (seconds, "second"),
    };
    format!("{count} {unit}{}", if count == 1 { "" } else { "s" })
}
