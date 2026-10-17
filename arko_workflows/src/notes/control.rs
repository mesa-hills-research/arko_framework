//! The control folder: notes for a workflow written by another process,
//! such as a binary's `note` subcommand or the `runctl` example, while the
//! workflow runs or while it is stopped, and requests to stop agent runs
//! alone while it runs.
//!
//! A workflow built with
//! [`WorkflowBuilder::control_folder`](crate::WorkflowBuilder::control_folder)
//! creates `<run dir>/control/` and reads its notes file,
//! [`NOTES_FILE`]: one JSON [`ControlNote`] per line, appended under an
//! exclusive lock on [`LOCK_FILE`] (a [`std::fs::File::lock`], never a lock
//! of the run's own). The workflow reads new lines under a shared lock at
//! its agents' turn boundaries, when it starts an agent, and in
//! [`Workflow::notes`](crate::Workflow::notes), and posts each note as
//! [`Workflow::post_note`](crate::Workflow::post_note) would. How much of
//! the file it has read is saved in its checkpoints, so a note written while
//! the workflow is stopped is read after it resumes, and none is read twice.
//!
//! Its stops file, [`STOPS_FILE`], holds one [`ControlStop`] per line,
//! appended under the same lock ([`request_stop`]). A running workflow reads
//! it every [`STOPS_POLL_INTERVAL`] and stops the runs each request names
//! that are running then, as [`Workflow::stop_run`](crate::Workflow::stop_run)
//! and [`Workflow::stop_agent`](crate::Workflow::stop_agent) do. It reads
//! only the requests written after it was built.
//!
//! Whoever can write the folder can post notes, as any source, and stop
//! runs: keep the run
//! directory out of the agents' reach (outside their workspace, or behind a
//! path policy), as a host keeps its own command line out of their sandbox.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Note;
use crate::context::SharedContext;
use crate::workflow::stop::{RunStopIntent, StopKind, StopRequest};
use std::sync::Arc;

/// The control folder's name in the run directory.
pub const FOLDER: &str = "control";

/// The notes file in the control folder: one [`ControlNote`] per line.
pub const NOTES_FILE: &str = "notes.jsonl";

/// The file locked while the notes file or the stops file is written
/// (exclusively) or read (shared).
pub const LOCK_FILE: &str = "notes.lock";

/// The stops file in the control folder: one [`ControlStop`] per line.
pub const STOPS_FILE: &str = "stops.jsonl";

/// How often a running workflow reads the stops file.
pub const STOPS_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// The control folder of run directory `run_dir`: `<run_dir>/control`.
pub fn folder(run_dir: &Path) -> PathBuf {
    run_dir.join(FOLDER)
}

/// A line of the notes file: a [`Note`]'s fields, with the identity and time
/// [`post_note`] gives it. A line written by hand may leave both out. The
/// workflow then gives the note an identity when it reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlNote {
    /// The posted note's identity. A line whose `id` the workflow has posted
    /// already is skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Uuid>,
    /// When it was written. The posted note's time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<DateTime<Utc>>,
    #[serde(flatten)]
    pub note: Note,
}

/// Appends `note` to the notes file of the workflow whose run directory is
/// `run_dir`, durably (synced), under the control folder's lock, and returns
/// the line written, with a new `id` and the current time. Works whether the
/// workflow is running or stopped: the workflow posts the note at its next
/// read.
///
/// Fails with [`io::ErrorKind::NotFound`] when the run directory has no
/// control folder (the workflow doesn't read one: see
/// [`WorkflowBuilder::control_folder`](crate::WorkflowBuilder::control_folder)),
/// and with [`io::ErrorKind::InvalidInput`] for an empty note or one over
/// [`MAX_NOTE_BYTES`](super::MAX_NOTE_BYTES). The audience is checked when the
/// workflow reads the note: one naming an unknown agent is skipped there,
/// with a warning in the workflow's log.
pub fn post_note(run_dir: &Path, note: &Note) -> io::Result<ControlNote> {
    let folder = folder(run_dir);
    if !folder.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{} doesn't exist: the workflow of this run directory doesn't read a control folder (WorkflowBuilder::control_folder)",
                folder.display()
            ),
        ));
    }
    let text = super::checked_text(&note.text)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let entry = ControlNote {
        id: Some(Uuid::new_v4()),
        at: Some(Utc::now()),
        note: Note {
            text,
            ..note.clone()
        },
    };
    let mut line = serde_json::to_vec(&entry).map_err(io::Error::other)?;
    line.push(b'\n');
    let lock = open_lock(&folder)?;
    lock.lock()?;
    let appended = append(&folder.join(NOTES_FILE), &line);
    let _ = lock.unlock();
    appended?;
    Ok(entry)
}

/// Every note in the notes file of run directory `run_dir`, oldest first,
/// read under the control folder's lock. Lines that aren't notes are
/// skipped. Empty when there is no notes file. For status tools: which of
/// them the workflow has posted is in its checkpoint
/// ([`NotesSnapshot::notes`](super::NotesSnapshot::notes), by `id`).
pub fn read_notes(run_dir: &Path) -> io::Result<Vec<ControlNote>> {
    let folder = folder(run_dir);
    let path = folder.join(NOTES_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let lock = open_lock(&folder)?;
    lock.lock_shared()?;
    let bytes = std::fs::read(&path);
    let _ = lock.unlock();
    Ok(lines(&bytes?)
        .filter_map(|line| serde_json::from_slice(line).ok())
        .collect())
}

/// Appends `line` to `path`, ending a last line a crashed write left
/// unfinished first, and syncs it.
fn append(path: &Path, line: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path)?;
    let len = file.metadata()?.len();
    if len > 0 {
        let mut last = [0_u8];
        file.seek(SeekFrom::Start(len - 1))?;
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            file.write_all(b"\n")?;
        }
    }
    file.write_all(line)?;
    file.sync_all()?;
    // The new file's directory entry, too.
    #[cfg(unix)]
    {
        if len == 0
            && let Some(parent) = path.parent()
        {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

fn open_lock(folder: &Path) -> io::Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(folder.join(LOCK_FILE))
}

/// The nonblank lines of `bytes`, without their line ends.
fn lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.trim_ascii().is_empty())
}

/// The lines of the notes file after `offset`, read once no writer holds the
/// lock.
pub(crate) struct Batch {
    /// Each complete line: a note, or why it isn't one.
    pub entries: Vec<Result<ControlNote, String>>,
    /// The offset after the last complete line: an unfinished last line
    /// (one a writer crashed in) is read once a later write ends it.
    pub end: u64,
}

/// The notes written to `folder`'s notes file after byte `offset`. `None`
/// when there are none, or a writer holds the lock (they are read next
/// time). A file shorter than `offset` was replaced: it is read from the
/// start, and the notes it repeats are skipped by their `id`.
pub(crate) fn read_from(folder: &Path, offset: u64) -> io::Result<Option<Batch>> {
    Ok(
        read_lines_from(folder, NOTES_FILE, offset)?.map(|read| Batch {
            entries: read
                .lines
                .iter()
                .map(|line| serde_json::from_slice(line).map_err(|error| error.to_string()))
                .collect(),
            end: read.end,
        }),
    )
}

/// The complete lines read from a control file, and the offset after them.
struct Lines {
    lines: Vec<Vec<u8>>,
    end: u64,
}

/// The complete lines written to `folder`'s file `name` after byte
/// `offset`. `None` when there are none, or a writer holds the lock (they
/// are read next time). A file shorter than `offset` was replaced: it is
/// read from the start. An unfinished last line (one a writer crashed in)
/// is read once a later write ends it.
fn read_lines_from(folder: &Path, name: &str, offset: u64) -> io::Result<Option<Lines>> {
    let path = folder.join(name);
    let len = match std::fs::metadata(&path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let offset = if len < offset {
        log::warn!(
            "{} is shorter than the part already read; reading it again from the start",
            path.display()
        );
        0
    } else {
        offset
    };
    if len == offset {
        return Ok(None);
    }
    let lock = open_lock(folder)?;
    match lock.try_lock_shared() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
        Err(std::fs::TryLockError::Error(error)) => return Err(error),
    }
    let read = (|| {
        let mut file = File::open(&path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok::<_, io::Error>(bytes)
    })();
    let _ = lock.unlock();
    let bytes = read?;
    let complete = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |end| end + 1);
    Ok(Some(Lines {
        lines: lines(&bytes[..complete]).map(<[u8]>::to_vec).collect(),
        end: offset + complete as u64,
    }))
}

/// A line of the stops file: a request to stop agent runs alone
/// ([`Workflow::stop_run`](crate::Workflow::stop_run) or
/// [`Workflow::stop_agent`](crate::Workflow::stop_agent)), written by
/// another process while the workflow runs. It names one run (`run`) or
/// every running run of an agent (`agent`). In JSON only `run` or `agent`
/// is required. The others default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlStop {
    /// The request's identity, given by [`request_stop`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Uuid>,
    /// When it was written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<DateTime<Utc>>,
    /// The agent run to stop, by its ID (`agent_run_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<Uuid>,
    /// Stop every running run of this agent (its registration name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// Default [`StopKind::Operator`].
    #[serde(default = "operator")]
    pub kind: StopKind,
    #[serde(default)]
    pub reason: String,
    /// The wrap-up's length in seconds. None: the runs are cancelled at
    /// once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrap_up_secs: Option<u64>,
    /// Default [`RunStopIntent::Retire`].
    #[serde(default = "retire")]
    pub intent: RunStopIntent,
}

fn operator() -> StopKind {
    StopKind::Operator
}

fn retire() -> RunStopIntent {
    RunStopIntent::Retire
}

impl ControlStop {
    /// A request to stop agent run `run` and retire it, without a wrap-up.
    pub fn run(run: Uuid, reason: impl Into<String>) -> Self {
        Self::new(Some(run), None, reason.into())
    }

    /// A request to stop every running run of `agent` and retire them,
    /// without a wrap-up.
    pub fn agent(agent: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(None, Some(agent.into()), reason.into())
    }

    fn new(run: Option<Uuid>, agent: Option<String>, reason: String) -> Self {
        Self {
            id: None,
            at: None,
            run,
            agent,
            kind: StopKind::Operator,
            reason,
            wrap_up_secs: None,
            intent: RunStopIntent::Retire,
        }
    }

    /// A wrap-up of `within` (in whole seconds) before the runs are
    /// cancelled.
    pub fn wrap_up(mut self, within: std::time::Duration) -> Self {
        self.wrap_up_secs = Some(within.as_secs());
        self
    }

    /// What the host means to do with the stopped runs.
    pub fn intent(mut self, intent: RunStopIntent) -> Self {
        self.intent = intent;
        self
    }

    /// The [`StopRequest`] it asks for.
    pub fn request(&self) -> StopRequest {
        let request = StopRequest::new(self.kind, self.reason.clone());
        match self.wrap_up_secs {
            Some(seconds) => request.wrap_up(std::time::Duration::from_secs(seconds)),
            None => request,
        }
    }

    fn check(&self) -> Result<(), String> {
        match (&self.run, &self.agent) {
            (Some(_), None) => Ok(()),
            (None, Some(agent)) if !agent.trim().is_empty() => Ok(()),
            (None, Some(_)) => Err("its agent name is blank".to_string()),
            (Some(_), Some(_)) => Err("it names both a run and an agent".to_string()),
            (None, None) => Err("it names neither a run nor an agent".to_string()),
        }
    }
}

/// Appends `stop` to the stops file of the workflow whose run directory is
/// `run_dir`, durably (synced), under the control folder's lock, and
/// returns the line written, with a new `id` and the current time. The
/// running workflow reads it within [`STOPS_POLL_INTERVAL`] and stops the
/// runs it names that are running then. A workflow reads only the requests
/// written after it was built, so one written while the workflow isn't
/// running is never applied.
///
/// Fails with [`io::ErrorKind::NotFound`] when the run directory has no
/// control folder (see
/// [`WorkflowBuilder::control_folder`](crate::WorkflowBuilder::control_folder)),
/// and with [`io::ErrorKind::InvalidInput`] for a request that names both a
/// run and an agent, or neither. The agent's name is checked when the
/// workflow reads the request.
pub fn request_stop(run_dir: &Path, stop: &ControlStop) -> io::Result<ControlStop> {
    let folder = folder(run_dir);
    if !folder.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{} doesn't exist: the workflow of this run directory doesn't read a control folder (WorkflowBuilder::control_folder)",
                folder.display()
            ),
        ));
    }
    stop.check()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let entry = ControlStop {
        id: Some(Uuid::new_v4()),
        at: Some(Utc::now()),
        ..stop.clone()
    };
    let mut line = serde_json::to_vec(&entry).map_err(io::Error::other)?;
    line.push(b'\n');
    let lock = open_lock(&folder)?;
    lock.lock()?;
    let appended = append(&folder.join(STOPS_FILE), &line);
    let _ = lock.unlock();
    appended?;
    Ok(entry)
}

/// Every request in the stops file of run directory `run_dir`, oldest
/// first, read under the control folder's lock. Lines that aren't requests
/// are skipped. Empty when there is no stops file.
pub fn read_stops(run_dir: &Path) -> io::Result<Vec<ControlStop>> {
    let folder = folder(run_dir);
    let path = folder.join(STOPS_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let lock = open_lock(&folder)?;
    lock.lock_shared()?;
    let bytes = std::fs::read(&path);
    let _ = lock.unlock();
    Ok(lines(&bytes?)
        .filter_map(|line| serde_json::from_slice(line).ok())
        .collect())
}

/// Reads the stops file while the workflow runs, and applies each new
/// request (see [`request_stop`]).
pub(crate) struct StopReader {
    /// The control folder, when enabled.
    folder: Option<PathBuf>,
    /// How much of the stops file has been read: from its length when the
    /// workflow was built, so that the requests written before are skipped.
    offset: parking_lot::Mutex<u64>,
    /// The reading task was started.
    started: std::sync::atomic::AtomicBool,
}

impl StopReader {
    pub(crate) fn new(folder: Option<PathBuf>) -> Self {
        let offset = folder
            .as_ref()
            .and_then(|folder| std::fs::metadata(folder.join(STOPS_FILE)).ok())
            .map_or(0, |metadata| metadata.len());
        Self {
            folder,
            offset: parking_lot::Mutex::new(offset),
            started: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Starts the task that reads the stops file every
    /// [`STOPS_POLL_INTERVAL`] (once, from the first agent run's start, it
    /// needs a Tokio runtime). The task ends once the workflow is stopped or
    /// dropped.
    pub(crate) fn start(&self, context: &Arc<SharedContext>) {
        use std::sync::atomic::Ordering;
        if self.folder.is_none() || self.started.load(Ordering::Acquire) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if self.started.swap(true, Ordering::AcqRel) {
            return;
        }
        let context = Arc::downgrade(context);
        runtime.spawn(async move {
            loop {
                tokio::time::sleep(STOPS_POLL_INTERVAL).await;
                let Some(context) = context.upgrade() else {
                    return;
                };
                if context.run_control.is_stopped() {
                    return;
                }
                context.control_stops.poll(&context);
            }
        });
    }

    /// Applies the requests written since the file was last read. A writer
    /// holding the lock leaves them for the next read.
    pub(crate) fn poll(&self, context: &Arc<SharedContext>) {
        let Some(folder) = &self.folder else {
            return;
        };
        let read = {
            let mut offset = self.offset.lock();
            match read_lines_from(folder, STOPS_FILE, *offset) {
                Ok(Some(read)) => {
                    *offset = read.end;
                    read
                }
                Ok(None) => return,
                Err(error) => {
                    log::warn!(
                        "Workflow '{}': can't read the stop requests in {}: {error}",
                        context.workflow_name,
                        folder.display()
                    );
                    return;
                }
            }
        };
        let handle = crate::workflow::cancel::WorkflowCancelHandle::new(context);
        for line in read.lines {
            let stop = serde_json::from_slice::<ControlStop>(&line)
                .map_err(|error| format!("it isn't a stop request ({error})"))
                .and_then(|stop| stop.check().map(|()| stop));
            let stop = match stop {
                Ok(stop) => stop,
                Err(reason) => {
                    log::warn!(
                        "Workflow '{}': skipped an entry of {}: {reason}",
                        context.workflow_name,
                        folder.join(STOPS_FILE).display()
                    );
                    continue;
                }
            };
            let id = stop
                .id
                .map_or_else(|| "(without an id)".to_string(), |id| id.to_string());
            match (&stop.run, &stop.agent) {
                (Some(run), _) => {
                    if !handle.stop_run(*run, stop.request(), stop.intent) {
                        log::warn!(
                            "Workflow '{}': stop request {id}: agent run {run} isn't running, or is being stopped already",
                            context.workflow_name
                        );
                    }
                }
                (None, Some(agent)) => {
                    match handle.stop_agent(agent, stop.request(), stop.intent) {
                        Ok(stopped) if stopped.is_empty() => log::warn!(
                            "Workflow '{}': stop request {id}: no run of agent '{agent}' is running, or each is being stopped already",
                            context.workflow_name
                        ),
                        Ok(_) => {}
                        Err(error) => log::warn!(
                            "Workflow '{}': skipped stop request {id}: {error}",
                            context.workflow_name
                        ),
                    }
                }
                (None, None) => {}
            }
        }
    }
}
