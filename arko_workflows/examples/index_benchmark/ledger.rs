//! Append-only JSON-lines files: the durable record of finished work that
//! makes every stage resumable. A record is written (and synced) when its
//! work is done; a crash loses at most the work in progress, which is done
//! again on the next start.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::BenchResult;

/// Reads every record of `path` (none when it doesn't exist). A line that
/// doesn't parse (a write cut short by a crash) is skipped with a warning:
/// its work is done again.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> BenchResult<Vec<T>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("reading {}: {error}", path.display()).into()),
    };
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(line) {
            Ok(record) => records.push(record),
            Err(error) => eprintln!(
                "warning: {} line {}: skipping a record that doesn't parse ({error}); a write was probably cut short",
                path.display(),
                index + 1
            ),
        }
    }
    Ok(records)
}

/// Appends records to a JSON-lines file, one synced line per record.
#[derive(Debug)]
pub struct Appender {
    path: PathBuf,
    file: parking_lot::Mutex<std::fs::File>,
}

impl Appender {
    pub fn open(path: &Path) -> BenchResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // A last line cut short by a crash gets a newline first, so the next
        // record starts on a line of its own.
        let needs_newline = std::fs::read(path)
            .map(|bytes| !bytes.is_empty() && !bytes.ends_with(b"\n"))
            .unwrap_or(false);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        if needs_newline {
            file.write_all(b"\n")?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: parking_lot::Mutex::new(file),
        })
    }

    pub fn append<T: Serialize>(&self, record: &T) -> BenchResult<()> {
        let mut line = serde_json::to_string(record)?;
        line.push('\n');
        let mut file = self.file.lock();
        file.write_all(line.as_bytes())
            .and_then(|()| file.sync_data())
            .map_err(|error| format!("writing {}: {error}", self.path.display()).into())
    }
}

/// A `LOCK` file holding this process's id: one process at a time works
/// in a directory (a run, the task generation, a `prepare`). A lock whose
/// process is gone (it crashed) is taken over. Removed on drop.
#[derive(Debug)]
pub struct DirLock {
    path: PathBuf,
}

impl DirLock {
    pub fn acquire(dir: &Path, what: &str) -> BenchResult<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("LOCK");
        if let Some(pid) = Self::holder(dir) {
            return Err(format!(
                "{what} is in progress in another process (pid {pid}): wait for it or stop it, or remove {} if that process isn't this benchmark",
                path.display()
            )
            .into());
        }
        std::fs::write(&path, std::process::id().to_string())?;
        Ok(Self { path })
    }

    /// The live process holding `dir`'s lock, if any.
    pub fn holder(dir: &Path) -> Option<u32> {
        let pid: u32 = std::fs::read_to_string(dir.join("LOCK"))
            .ok()?
            .trim()
            .parse()
            .ok()?;
        (pid != std::process::id() && process_alive(pid)).then_some(pid)
    }
}

impl Drop for DirLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Whether a process with this id runs (Linux: `/proc`; elsewhere a lock is
/// taken as held, since there is no cheap check).
fn process_alive(pid: u32) -> bool {
    if cfg!(target_os = "linux") {
        Path::new(&format!("/proc/{pid}")).exists()
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Record {
        n: u32,
    }

    #[test]
    fn appends_and_reads_back_skipping_a_cut_last_line() {
        let dir = crate::test_dir("ledger");
        let path = dir.join("records.jsonl");
        assert!(read_jsonl::<Record>(&path).unwrap().is_empty());
        let appender = Appender::open(&path).unwrap();
        appender.append(&Record { n: 1 }).unwrap();
        appender.append(&Record { n: 2 }).unwrap();
        drop(appender);
        // A crash in the middle of a write.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"n\": 3")
            .unwrap();
        assert_eq!(
            read_jsonl::<Record>(&path).unwrap(),
            vec![Record { n: 1 }, Record { n: 2 }]
        );
        let appender = Appender::open(&path).unwrap();
        appender.append(&Record { n: 4 }).unwrap();
        drop(appender);
        assert_eq!(
            read_jsonl::<Record>(&path).unwrap(),
            vec![Record { n: 1 }, Record { n: 2 }, Record { n: 4 }]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn locks_refuse_a_live_holder_and_take_over_a_dead_one() {
        let dir = crate::test_dir("lock");
        // A live process (this test's parent, say) holds it.
        let parent = std::fs::read_to_string("/proc/self/stat")
            .unwrap()
            .split_whitespace()
            .nth(3)
            .unwrap()
            .to_string();
        std::fs::write(dir.join("LOCK"), &parent).unwrap();
        assert!(DirLock::acquire(&dir, "the run").is_err());
        // A process that is gone.
        std::fs::write(dir.join("LOCK"), "4000000000").unwrap();
        let lock = DirLock::acquire(&dir, "the run").unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("LOCK")).unwrap(),
            std::process::id().to_string()
        );
        drop(lock);
        assert!(!dir.join("LOCK").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
