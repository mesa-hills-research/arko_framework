use crate::{
    bench_v1::{Chunk, Delta, Error, Options, chunk_str},
    files::{FileData, FileRules, Report, read_dir, write_report},
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[cfg(test)]
thread_local! {
    static CHUNK_WORK: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[derive(Clone, Serialize, Deserialize)]
struct State {
    version: u32,
    root: String,
    max_bytes: usize,
    #[serde(default)]
    engine: String,
    #[serde(default)]
    file_hashes: BTreeMap<String, String>,
    sequence: u64,
    acked: u64,
    chunks: Vec<Chunk>,
    pending: Vec<Delta>,
    #[serde(default)]
    fingerprints: HashMap<String, String>,
    /// File selection rules of the last run (absent in states from earlier versions).
    #[serde(default)]
    rules: Option<FileRules>,
}
// S-f5530629 qualifies persisted parser caches by engine and options. Hash the
// actual boundary/identity policy and locked dependencies, rather than relying on
// a manually bumped string when a grammar or normalization implementation changes.
/// [`engine_id`], qualified by the options other than `max_bytes` (which the state
/// records itself) that change chunks: the default options keep the plain engine id.
pub(crate) fn engine_for(options: &Options) -> String {
    if options.front_matter_description {
        format!("{}-front-matter-description", engine_id())
    } else {
        engine_id()
    }
}
fn engine_id() -> String {
    crate::files::hash(
        concat!(
            "arko-index-cache-v1\0",
            env!("CARGO_PKG_VERSION"),
            include_str!("syntax.rs"),
            include_str!("languages.rs"),
            include_str!("rules/mod.rs"),
            include_str!("rules/tables.rs"),
            include_str!("rules/tags.rs"),
            include_str!("headers.rs"),
            include_str!("normalize.rs"),
            include_str!("files.rs"),
            include_str!("bench_v1.rs"),
            include_str!(concat!(env!("OUT_DIR"), "/engine_lock.txt")),
        )
        .as_bytes(),
    )
}

#[derive(Clone, Default)]
struct CachedFile {
    hash: String,
    chunks: Vec<Chunk>,
    // S-ba66f1e3 caches exact chunk byte hashes together with the parsed chunks.
    // Reuse only after hashing the entire file, never by size/mtime.
    fingerprints: Vec<String>,
}
impl CachedFile {
    fn matches(&self, path: &str, file: &FileData) -> bool {
        if self.hash != file.hash || self.chunks.len() != self.fingerprints.len() {
            return false;
        }
        let mut end = 0;
        for c in &self.chunks {
            if c.path != path
                || c.start_byte != end
                || c.end_byte <= end
                || c.end_byte > file.text.len()
                || !file.text.is_char_boundary(c.end_byte)
            {
                return false;
            }
            end = c.end_byte;
        }
        end == file.text.len()
    }
    fn parse(path: &str, file: &FileData, options: &Options) -> Result<Self, Error> {
        #[cfg(test)]
        CHUNK_WORK.with(|v| {
            let (p, h) = v.get();
            v.set((p + 1, h));
        });
        let chunks = chunk_str(Path::new(path), &file.text, options)?;
        #[cfg(test)]
        CHUNK_WORK.with(|v| {
            let (p, h) = v.get();
            v.set((p, h + chunks.len()));
        });
        let fingerprints = chunks
            .iter()
            .map(|c| crate::files::hash(&file.text.as_bytes()[c.start_byte..c.end_byte]))
            .collect();
        Ok(Self {
            hash: file.hash.clone(),
            chunks,
            fingerprints,
        })
    }
}

/// Durable, at-least-once event delivery. Polling deliberately does not depend on an OS
/// event queue: missed notifications/queue overflow cannot silently lose filesystem changes.
pub struct Watcher {
    root: PathBuf,
    dir: PathBuf,
    options: Options,
    state: State,
    cache: BTreeMap<String, CachedFile>,
    engine: String,
    delivered: u64,
    poisoned: bool,
    observed: Option<(Vec<(String, String)>, Instant)>,
    rules: FileRules,
    report: Report,
    rules_changed: bool,
    _lock: File,
}

pub(crate) fn atomic_json<T: Serialize>(dir: &Path, name: &str, value: &T) -> Result<(), Error> {
    let tmp = dir.join(format!(".{}.{}.tmp", name, std::process::id()));
    {
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        // S-21956424 (r06): buffer serde's small writes, but explicitly flush
        // before fsync/rename. Drop alone would silently discard flush errors.
        {
            let mut buffered = BufWriter::new(&mut f);
            serde_json::to_writer(&mut buffered, value)?;
            buffered.write_all(b"\n")?;
            buffered.flush()?;
        }
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(name))?;
    sync_parent(dir).map_err(|e| -> Error { Box::new(UncertainCommit(e)) })?;
    Ok(())
}
/// Rename committed a new journal, but directory durability could not be confirmed.
/// Unlike a pre-rename error, this cannot safely be retried against old memory.
#[derive(Debug)]
struct UncertainCommit(std::io::Error);
impl std::fmt::Display for UncertainCommit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "journal rename succeeded but directory sync failed; drop/reopen before continuing: {}",
            self.0
        )
    }
}
impl std::error::Error for UncertainCommit {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
// Test-only injection exercises the post-rename error boundary; never an env hook.
#[cfg(test)]
thread_local! {
    static FAIL_PARENT_SYNC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
fn sync_parent(dir: &Path) -> std::io::Result<()> {
    #[cfg(test)]
    if FAIL_PARENT_SYNC.with(|flag| flag.replace(false)) {
        return Err(std::io::Error::other(
            "injected parent directory sync failure",
        ));
    }
    crate::files::sync_dir(dir)
}
impl Watcher {
    fn check_health(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err("watch commit outcome is uncertain; drop and reopen the watcher".into());
        }
        Ok(())
    }
    fn persistence_result(&mut self, result: Result<(), Error>) -> Result<(), Error> {
        if result
            .as_ref()
            .is_err_and(|e| e.downcast_ref::<UncertainCommit>().is_some())
        {
            self.poisoned = true;
        }
        result
    }
    pub(crate) fn open(root: &Path, state_dir: &Path, options: &Options) -> Result<Self, Error> {
        Self::open_with(root, state_dir, options, &FileRules::default())
    }
    pub(crate) fn open_with(
        root: &Path,
        state_dir: &Path,
        options: &Options,
        rules: &FileRules,
    ) -> Result<Self, Error> {
        if options.max_bytes < 4 {
            return Err("max_bytes must be at least 4 (one UTF-8 scalar)".into());
        }
        rules.validate()?;
        let root = root.canonicalize()?;
        if !root.is_dir() {
            return Err("watch root must be a directory".into());
        }
        std::fs::create_dir_all(state_dir)?;
        let dir = state_dir.canonicalize()?;
        if dir == root || root.starts_with(&dir) {
            return Err("state directory may not contain the watched root".into());
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(".arko-index-state"))?;
        lock.try_lock_exclusive()
            .map_err(|e| format!("watch state already in use: {e}"))?;
        lock.sync_all()?;
        let root_name = root
            .to_str()
            .ok_or("watch root path must be UTF-8")?
            .to_owned();
        let state_path = dir.join("watch.json");
        let existed = state_path.is_file();
        let mut state = if existed {
            let s: State = serde_json::from_reader(BufReader::new(File::open(state_path)?))?;
            if s.version != 1 || s.root != root_name {
                return Err("watch state version/root mismatch".into());
            }
            s
        } else {
            State {
                version: 1,
                root: root_name,
                max_bytes: options.max_bytes,
                engine: String::new(),
                file_hashes: BTreeMap::new(),
                sequence: 0,
                acked: 0,
                chunks: vec![],
                pending: vec![],
                fingerprints: HashMap::new(),
                rules: Some(rules.clone()),
            }
        };
        // Different rules (or a state from before rules were recorded): the startup
        // reconciliation below applies the current rules like any offline change.
        let rules_changed = existed && state.rules.as_ref() != Some(rules);
        state.rules = Some(rules.clone());
        let delivered = state.acked;
        let engine = engine_for(options);
        let mut cache = BTreeMap::new();
        if state.engine == engine && state.max_bytes == options.max_bytes {
            for (path, hash) in &state.file_hashes {
                cache.insert(
                    path.clone(),
                    CachedFile {
                        hash: hash.clone(),
                        ..CachedFile::default()
                    },
                );
            }
            for c in &state.chunks {
                if let Some(entry) = cache.get_mut(&c.path) {
                    entry.chunks.push(c.clone());
                    if let Some(fp) = state.fingerprints.get(&c.id) {
                        entry.fingerprints.push(fp.clone());
                    }
                }
            }
        }
        let mut watcher = Self {
            root,
            dir,
            options: options.clone(),
            state,
            cache,
            engine,
            delivered,
            poisoned: false,
            observed: None,
            rules: rules.clone(),
            report: Report::default(),
            rules_changed,
            _lock: lock,
        };
        let (inventory, report) = read_dir(&watcher.root, Some(&watcher.dir), &watcher.rules)?;
        watcher.reconcile(inventory)?;
        watcher.flush()?;
        watcher.set_report(report)?;
        Ok(watcher)
    }
    /// What the latest pass indexed and skipped; also kept in `STATE_DIR/report.json`.
    pub fn report(&self) -> &Report {
        &self.report
    }
    /// Whether this open found a state written with different file selection rules
    /// (or by a version that did not record them); the open reconciled to the new rules.
    pub fn rules_changed(&self) -> bool {
        self.rules_changed
    }
    fn set_report(&mut self, report: Report) -> Result<(), Error> {
        if report != self.report {
            write_report(&self.dir, &report)?;
            self.report = report;
        }
        Ok(())
    }
    fn reconcile(&mut self, inventory: BTreeMap<String, FileData>) -> Result<(), Error> {
        let mut cache = BTreeMap::new();
        let mut chunks = Vec::new();
        let mut fingerprints = HashMap::new();
        let mut file_hashes = BTreeMap::new();
        for (path, file) in inventory {
            let entry = match self
                .cache
                .get(&path)
                .filter(|cached| cached.matches(&path, &file))
            {
                Some(cached) => cached.clone(),
                None => CachedFile::parse(&path, &file, &self.options)?,
            };
            for (c, fp) in entry.chunks.iter().zip(&entry.fingerprints) {
                fingerprints.insert(c.id.clone(), fp.clone());
            }
            chunks.extend(entry.chunks.iter().cloned());
            file_hashes.insert(path.clone(), entry.hash.clone());
            cache.insert(path, entry);
        }
        let old: HashMap<&str, &Chunk> = self
            .state
            .chunks
            .iter()
            .map(|c| (c.id.as_str(), c))
            .collect();
        let new: HashMap<&str, &Chunk> = chunks.iter().map(|c| (c.id.as_str(), c)).collect();
        let added = chunks
            .iter()
            .filter(|c| !old.contains_key(c.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let changed = chunks
            .iter()
            .filter(|c| {
                old.get(c.id.as_str()).is_some_and(|before| {
                    *before != *c || self.state.fingerprints.get(&c.id) != fingerprints.get(&c.id)
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        let removed = self
            .state
            .chunks
            .iter()
            .filter(|c| !new.contains_key(c.id.as_str()))
            .map(|c| c.id.clone())
            .collect::<Vec<_>>();
        let changed = Delta {
            sequence: 0,
            added,
            changed,
            removed,
        };
        let has_delta =
            !changed.added.is_empty() || !changed.changed.is_empty() || !changed.removed.is_empty();
        let cache_changed = self.state.engine != self.engine
            || self.state.max_bytes != self.options.max_bytes
            || self.state.file_hashes != file_hashes;
        if has_delta || cache_changed {
            let mut state = self.state.clone();
            if has_delta {
                state.sequence = state
                    .sequence
                    .checked_add(1)
                    .ok_or("watch sequence exhausted")?;
                state.pending.push(Delta {
                    sequence: state.sequence,
                    ..changed
                });
            }
            state.chunks = chunks;
            state.fingerprints = fingerprints;
            state.max_bytes = self.options.max_bytes;
            state.engine = self.engine.clone();
            state.file_hashes = file_hashes;
            // Both durable and in-memory state advance only after the transaction is committed.
            let result = atomic_json(&self.dir, "watch.json", &state);
            self.persistence_result(result)?;
            self.state = state;
        }
        self.cache = cache;
        Ok(())
    }
    fn fingerprint(inventory: &BTreeMap<String, FileData>) -> Vec<(String, String)> {
        inventory
            .iter()
            .map(|(p, f)| (p.clone(), f.hash.clone()))
            .collect()
    }
    /// Returns each pending event once in this process, replaying unacked events on reopen.
    /// A quiet timeout is None. Changes are committed only after a stable 75ms debounce window.
    pub fn next_change(&mut self, timeout: Duration) -> Result<Option<Delta>, Error> {
        self.check_health()?;
        if let Some(delta) = self
            .state
            .pending
            .iter()
            .find(|d| d.sequence > self.delivered)
        {
            self.delivered = delta.sequence;
            return Ok(Some(delta.clone()));
        }
        let began = Instant::now();
        loop {
            let (inventory, report) = read_dir(&self.root, Some(&self.dir), &self.rules)?;
            self.set_report(report)?;
            let current = Self::fingerprint(&inventory);
            let known: Vec<_> = self
                .cache
                .iter()
                .map(|(p, entry)| (p.clone(), entry.hash.clone()))
                .collect();
            if current != known {
                let stable = self.observed.as_ref().is_some_and(|(fp, since)| {
                    fp == &current && since.elapsed() >= Duration::from_millis(75)
                });
                if stable {
                    self.reconcile(inventory)?;
                    if let Some(d) = self
                        .state
                        .pending
                        .iter()
                        .find(|d| d.sequence > self.delivered)
                    {
                        self.delivered = d.sequence;
                        return Ok(Some(d.clone()));
                    }
                    self.observed = None;
                } else if self.observed.as_ref().is_none_or(|(fp, _)| fp != &current) {
                    self.observed = Some((current, Instant::now()));
                }
            } else {
                self.observed = None;
            }
            let elapsed = began.elapsed();
            if elapsed >= timeout {
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(30).min(timeout - elapsed));
        }
    }
    pub fn snapshot(&self) -> Result<Vec<Chunk>, Error> {
        self.check_health()?;
        Ok(self.state.chunks.clone())
    }
    /// Cumulative durable acknowledgement; a later sequence acknowledges every prior delta.
    pub fn ack(&mut self, sequence: u64) -> Result<(), Error> {
        self.check_health()?;
        if sequence > self.state.sequence {
            return Err("cannot ack a future watch sequence".into());
        }
        if sequence <= self.state.acked {
            return Ok(());
        }
        let mut state = self.state.clone();
        state.acked = sequence;
        state.pending.retain(|d| d.sequence > sequence);
        let result = atomic_json(&self.dir, "watch.json", &state);
        self.persistence_result(result)?;
        self.state = state;
        self.delivered = self.delivered.max(sequence);
        Ok(())
    }
    pub fn flush(&mut self) -> Result<(), Error> {
        self.check_health()?;
        let result = atomic_json(&self.dir, "watch.json", &self.state);
        self.persistence_result(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (tempfile::TempDir, tempfile::TempDir, Watcher) {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.rs"), "fn initial() {}\n").unwrap();
        let watch = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        (root, state, watch)
    }
    fn assert_poisoned(w: &mut Watcher) {
        assert!(
            w.snapshot().is_err(),
            "snapshot must not expose stale state"
        );
        assert!(w.next_change(Duration::ZERO).is_err());
        assert!(w.ack(0).is_err());
        assert!(
            w.flush().is_err(),
            "flush must not overwrite the renamed journal"
        );
    }
    #[test]
    fn uncertain_reconcile_requires_reopen_without_reusing_sequence() {
        let (root, state, mut w) = setup();
        let first = w.next_change(Duration::ZERO).unwrap().unwrap();
        w.ack(first.sequence).unwrap();
        std::fs::write(root.path().join("a.rs"), "fn second() {}\n").unwrap();
        FAIL_PARENT_SYNC.with(|flag| flag.set(true));
        assert!(w.next_change(Duration::from_secs(1)).is_err());
        let disk: State = serde_json::from_reader(BufReader::new(
            File::open(state.path().join("watch.json")).unwrap(),
        ))
        .unwrap();
        assert_eq!(disk.sequence, first.sequence + 1);
        assert_eq!(w.state.sequence, first.sequence);
        assert_poisoned(&mut w);
        drop(w);
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        let replay = w.next_change(Duration::ZERO).unwrap().unwrap();
        assert_eq!(replay.sequence, first.sequence + 1);
        w.ack(replay.sequence).unwrap();
        std::fs::write(root.path().join("a.rs"), "fn third() {}\n").unwrap();
        let third = w.next_change(Duration::from_secs(1)).unwrap().unwrap();
        assert_eq!(third.sequence, replay.sequence + 1);
    }
    #[test]
    fn uncertain_ack_requires_reopen_without_resurrecting_pending() {
        let (root, state, mut w) = setup();
        let first = w.next_change(Duration::ZERO).unwrap().unwrap();
        FAIL_PARENT_SYNC.with(|flag| flag.set(true));
        assert!(w.ack(first.sequence).is_err());
        let disk: State = serde_json::from_reader(BufReader::new(
            File::open(state.path().join("watch.json")).unwrap(),
        ))
        .unwrap();
        assert_eq!(disk.acked, first.sequence);
        assert_eq!(w.state.acked, 0);
        assert_poisoned(&mut w);
        drop(w);
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert!(w.next_change(Duration::ZERO).unwrap().is_none());
        std::fs::write(root.path().join("a.rs"), "fn second() {}\n").unwrap();
        let second = w.next_change(Duration::from_secs(1)).unwrap().unwrap();
        assert_eq!(second.sequence, first.sequence + 1);
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, Watcher) {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.rs"), "fn first() {}\n").unwrap();
        std::fs::write(root.path().join("b.rs"), "fn stable() {}\n").unwrap();
        std::fs::write(root.path().join("empty.txt"), "").unwrap();
        let watch = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        (root, state, watch)
    }
    fn reset_work() {
        CHUNK_WORK.with(|v| v.set((0, 0)));
    }
    fn work() -> (usize, usize) {
        CHUNK_WORK.with(|v| v.get())
    }
    #[test]
    fn reopen_reuses_validated_exact_content_cache_and_replays_unacked() {
        let (root, state, mut w) = fixture();
        let initial = w.next_change(Duration::ZERO).unwrap().unwrap();
        let snapshot = w.snapshot().unwrap();
        drop(w);
        reset_work();
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(
            work(),
            (0, 0),
            "unchanged files need neither parsing nor per-chunk rehashing"
        );
        assert_eq!(snapshot, w.snapshot().unwrap());
        assert_eq!(
            w.next_change(Duration::ZERO).unwrap().unwrap().sequence,
            initial.sequence
        );
        w.ack(initial.sequence).unwrap();
        drop(w);
        reset_work();
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (0, 0));
        assert!(w.next_change(Duration::ZERO).unwrap().is_none());
    }
    #[test]
    fn one_file_edit_does_not_rehash_other_files_chunks() {
        let (root, _state, mut w) = fixture();
        let initial = w.next_change(Duration::ZERO).unwrap().unwrap();
        w.ack(initial.sequence).unwrap();
        // Same length, same normalized ID, same metadata: only exact bytes distinguish it.
        std::fs::write(root.path().join("a.rs"), "fn first() {} ").unwrap();
        reset_work();
        let d = w.next_change(Duration::from_secs(1)).unwrap().unwrap();
        assert_eq!(work(), (1, 1));
        assert!(d.added.is_empty() && d.removed.is_empty());
        assert_eq!(d.changed.len(), 1);
        assert_eq!(d.changed[0].path, "a.rs");
    }
    fn read_journal(dir: &Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.join("watch.json")).unwrap()).unwrap()
    }
    fn write_journal(dir: &Path, v: &serde_json::Value) {
        std::fs::write(dir.join("watch.json"), serde_json::to_vec(v).unwrap()).unwrap();
    }
    #[test]
    fn legacy_engine_and_options_rebuild_cache_without_fabricating_events() {
        let (root, state, mut w) = fixture();
        let d = w.next_change(Duration::ZERO).unwrap().unwrap();
        w.ack(d.sequence).unwrap();
        let expected = w.snapshot().unwrap();
        drop(w);
        for field in ["engine", "file_hashes"] {
            let mut journal = read_journal(state.path());
            journal.as_object_mut().unwrap().remove(field);
            write_journal(state.path(), &journal);
            reset_work();
            let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
            assert_eq!(work(), (3, 2));
            assert_eq!(expected, w.snapshot().unwrap());
            assert!(w.next_change(Duration::ZERO).unwrap().is_none());
            assert_eq!(w.state.sequence, d.sequence);
            assert_eq!(w.state.engine, engine_id());
        }
        let mut journal = read_journal(state.path());
        journal["engine"] = serde_json::json!("other-parser-build");
        write_journal(state.path(), &journal);
        reset_work();
        let w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (3, 2));
        assert_eq!(expected, w.snapshot().unwrap());
        drop(w);
        reset_work();
        let options = Options {
            max_bytes: 64,
            ..Options::default()
        };
        let mut w = Watcher::open(root.path(), state.path(), &options).unwrap();
        assert_eq!(work(), (3, 2));
        assert!(w.next_change(Duration::ZERO).unwrap().is_none());
        assert_eq!(w.state.max_bytes, 64);
        drop(w);
        reset_work();
        let _w = Watcher::open(root.path(), state.path(), &options).unwrap();
        assert_eq!(work(), (0, 0));
    }
    #[test]
    fn offline_same_length_mtime_preserving_edit_invalidates_only_edited_file() {
        let (root, state, mut w) = fixture();
        let initial = w.next_change(Duration::ZERO).unwrap().unwrap();
        let before = w.snapshot().unwrap();
        drop(w); // Keep initial pending, so restart must replay it before reconciliation.
        let file = root.path().join("a.rs");
        let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
        let old_len = std::fs::metadata(&file).unwrap().len();
        std::fs::write(&file, "fn other() {}\n").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_eq!(old_len, std::fs::metadata(&file).unwrap().len());
        assert_eq!(
            modified,
            std::fs::metadata(&file).unwrap().modified().unwrap()
        );
        reset_work();
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (1, 1));
        assert_eq!(
            w.next_change(Duration::ZERO).unwrap().unwrap().sequence,
            initial.sequence
        );
        let edit = w.next_change(Duration::ZERO).unwrap().unwrap();
        assert_eq!(edit.sequence, initial.sequence + 1);
        assert_eq!(edit.added.len(), 1);
        assert_eq!(edit.removed.len(), 1);
        assert!(edit.changed.is_empty());
        assert_eq!(before[1], w.snapshot().unwrap()[1]);
        w.ack(edit.sequence).unwrap();
        drop(w);
        reset_work();
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (0, 0));
        assert!(w.next_change(Duration::ZERO).unwrap().is_none());
    }
    #[test]
    fn incomplete_persisted_cache_rebuilds_instead_of_dropping_chunks() {
        let (root, state, mut w) = fixture();
        let initial = w.next_change(Duration::ZERO).unwrap().unwrap();
        w.ack(initial.sequence).unwrap();
        let expected = w.snapshot().unwrap();
        drop(w);
        let mut journal = read_journal(state.path());
        journal["fingerprints"]
            .as_object_mut()
            .unwrap()
            .remove(&expected[0].id);
        write_journal(state.path(), &journal);
        reset_work();
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (1, 1));
        assert_eq!(expected, w.snapshot().unwrap());
        // Missing raw fingerprint is conservatively a content change.
        let d = w.next_change(Duration::ZERO).unwrap().unwrap();
        assert_eq!(d.changed.len(), 1);
        w.ack(d.sequence).unwrap();
        drop(w);
        let mut journal = read_journal(state.path());
        journal["chunks"].as_array_mut().unwrap().remove(0);
        write_journal(state.path(), &journal);
        reset_work();
        let w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (1, 1));
        assert_eq!(expected, w.snapshot().unwrap());
    }
    #[test]
    fn cache_only_changes_are_failure_atomic_without_sequence_events() {
        let (root, state, mut w) = fixture();
        let d = w.next_change(Duration::ZERO).unwrap().unwrap();
        w.ack(d.sequence).unwrap();
        let journal = read_journal(state.path());
        let blocked = state
            .path()
            .join(format!(".watch.json.{}.tmp", std::process::id()));
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(root.path().join("another_empty.txt"), "").unwrap();
        assert!(w.next_change(Duration::from_secs(1)).is_err());
        assert!(!w.cache.contains_key("another_empty.txt"));
        assert!(!w.state.file_hashes.contains_key("another_empty.txt"));
        assert_eq!(journal, read_journal(state.path()));
        std::fs::remove_dir(blocked).unwrap();
        assert!(w.next_change(Duration::from_millis(120)).unwrap().is_none());
        assert!(w.cache.contains_key("another_empty.txt"));
        assert_eq!(w.state.sequence, d.sequence);
        drop(w);
        reset_work();
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(work(), (0, 0));
        std::fs::remove_file(root.path().join("another_empty.txt")).unwrap();
        FAIL_PARENT_SYNC.with(|flag| flag.set(true));
        assert!(w.next_change(Duration::from_secs(1)).is_err());
        assert!(w.flush().is_err());
        drop(w);
        let mut w = Watcher::open(root.path(), state.path(), &Options::default()).unwrap();
        assert_eq!(w.state.sequence, d.sequence);
        assert!(w.next_change(Duration::ZERO).unwrap().is_none());
        assert!(!w.state.file_hashes.contains_key("another_empty.txt"));
    }
    #[test]
    fn invalid_options_fail_before_state_creation_even_for_empty_tree() {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        assert!(
            Watcher::open(
                root.path(),
                &state,
                &Options {
                    max_bytes: 0,
                    ..Options::default()
                }
            )
            .is_err()
        );
        assert!(!state.exists());
    }
}
