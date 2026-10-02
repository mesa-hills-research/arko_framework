//! The control folder: notes for a workflow written by another process,
//! such as a binary's `note` subcommand or the `runctl` example, while the
//! workflow runs or while it is stopped.
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
//! Whoever can write the folder can post notes, as any source: keep the run
//! directory out of the agents' reach (outside their workspace, or behind a
//! path policy), as a host keeps its own command line out of their sandbox.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Note;

/// The control folder's name in the run directory.
pub const FOLDER: &str = "control";

/// The notes file in the control folder: one [`ControlNote`] per line.
pub const NOTES_FILE: &str = "notes.jsonl";

/// The file locked while the notes file is written (exclusively) or read
/// (shared).
pub const LOCK_FILE: &str = "notes.lock";

/// The control folder of run directory `run_dir`: `<run_dir>/control`.
pub fn folder(run_dir: &Path) -> PathBuf {
    run_dir.join(FOLDER)
}

/// A line of the notes file: a [`Note`]'s fields, with the identity and time
/// [`post_note`] gives it. A line written by hand may leave both out; the
/// workflow then gives the note an identity when it reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlNote {
    /// The posted note's identity. A line whose `id` the workflow has posted
    /// already is skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Uuid>,
    /// When it was written; the posted note's time.
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
/// read under the control folder's lock; lines that aren't notes are
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

/// The notes written to `folder`'s notes file after byte `offset`; `None`
/// when there are none, or a writer holds the lock (they are read next
/// time). A file shorter than `offset` was replaced: it is read from the
/// start, and the notes it repeats are skipped by their `id`.
pub(crate) fn read_from(folder: &Path, offset: u64) -> io::Result<Option<Batch>> {
    let path = folder.join(NOTES_FILE);
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
    let entries = lines(&bytes[..complete])
        .map(|line| serde_json::from_slice(line).map_err(|error| error.to_string()))
        .collect();
    Ok(Some(Batch {
        entries,
        end: offset + complete as u64,
    }))
}
