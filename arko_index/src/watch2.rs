//! Scaling prototype of the watcher (Linux only): inotify events and metadata
//! reconciliation decide *which* files to look at; per-file records and an
//! append-only journal make each commit proportional to the change, not the repo.
//!
//! Durability model (same guarantees as `watch.rs`, different storage):
//! * `watch2.snap` is a full checkpoint written atomically (tmp, fsync, rename, fsync dir).
//!   It holds the file table (path -> metadata, content hash, record location) and the
//!   undelivered/unacknowledged deltas, not the chunks.
//! * `watch2.recs.N` (the record store) holds each file version's chunks, appended
//!   and fdatasync'ed before the journal record that references them. Only the
//!   file table stays in memory; a file's chunks are read back when it changes.
//!   On open the store is cut back to the end of its last referenced record (an
//!   uncommitted tail is garbage). When dead versions dominate and no delta is
//!   pending, the live records are copied to `watch2.recs.N+1` and a checkpoint
//!   switches to it.
//! * `watch2.wal` holds one JSON line per transaction, appended and fdatasync'ed
//!   **before** the delta is delivered. Every record carries a log sequence number;
//!   replay skips records already folded into the checkpoint and ignores a torn tail.
//!   A delta is stored as the (old, new) record versions of the files it covers and
//!   rendered from the store when delivered: the same per-file diff as before.
//! * Startup (and inotify queue overflow, and any directory-level event) runs a full
//!   reconciliation: walk + lstat every file; a file is re-read and re-hashed only if
//!   its (size, mtime, ctime, inode, device) differ from the record or the record is
//!   "racy" (ctime within RACY_NS of when it was hashed, the racy-git rule), so
//!   same-size, mtime-restoring edits are still found (they change ctime).
//! * File events re-list only the affected directories (their direct files), walking
//!   from the root so every file selection rule applies exactly as in a full walk.
//! * Rule changes: a `CACHEDIR.TAG` event (also in an excluded, tagged directory, which
//!   keeps a watch for its tag only) and, when ignore files are honoured, a
//!   `.gitignore`/`.ignore` event or a change of `.git/info/exclude` (each `.git/info`
//!   directory gets a watch for that file only) run a full reconciliation.
use crate::{
    bench_v1::{Chunk, Delta, Error, Options},
    files,
    files::{FileRules, Report, SkipReason, Skipped},
    par,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ffi::CString,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    os::unix::{
        ffi::OsStrExt,
        fs::{FileExt as _, MetadataExt},
    },
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const RACY_NS: i128 = 2_000_000_000;
const QUIET: Duration = Duration::from_millis(50);
const MAX_WAIT: Duration = Duration::from_millis(1000);
const WAL_CHECKPOINT_BYTES: u64 = 64 << 20;
/// Compact the record store once it exceeds twice its live records plus this.
const STORE_SLACK_BYTES: u64 = 64 << 20;
const SNAP_VERSION: u32 = 3;
const STORE_PREFIX: &str = "watch2.recs.";

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
struct Meta {
    size: u64,
    mtime: i128,
    ctime: i128,
    ino: u64,
    dev: u64,
}
fn meta_of(m: &std::fs::Metadata) -> Meta {
    Meta {
        size: m.size(),
        mtime: m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128,
        ctime: m.ctime() as i128 * 1_000_000_000 + m.ctime_nsec() as i128,
        ino: m.ino(),
        dev: m.dev(),
    }
}
fn now_ns() -> i128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0)
}

/// (offset, length) of one file version's records in the record store; length 0
/// means no chunks.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
struct Loc(u64, u64);
impl Loc {
    fn end(self) -> u64 {
        if self.1 == 0 { 0 } else { self.0 + self.1 }
    }
}

/// SHA-256 of a file's bytes, inline (no heap allocation per file), written as hex;
/// all zeros (written "") means unknown and never matches.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct Hash32([u8; 32]);
impl Serialize for Hash32 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0 == [0; 32] {
            return s.serialize_str("");
        }
        let mut hex = String::with_capacity(64);
        for b in self.0 {
            hex.push_str(&format!("{b:02x}"));
        }
        s.serialize_str(&hex)
    }
}
impl<'de> Deserialize<'de> for Hash32 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(d)?;
        let mut out = [0u8; 32];
        if hex.is_empty() {
            return Ok(Self(out));
        }
        if hex.len() != 64 {
            return Err(serde::de::Error::custom("bad content hash"));
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o =
                u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(serde::de::Error::custom)?;
        }
        Ok(Self(out))
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Default)]
struct FileRec {
    meta: Meta,
    hashed_at: i128,
    hash: Hash32,
    rec: Loc,
}
impl FileRec {
    fn trusted(&self, m: &Meta) -> bool {
        self.meta == *m && self.meta.ctime + RACY_NS < self.hashed_at
    }
}

/// A delta not yet acknowledged: the (old, new) record versions of each file with a
/// non-empty diff, in path order. Its JSON is rendered from the store on delivery.
#[derive(Clone, Serialize, Deserialize, Default)]
struct PendingDelta {
    sequence: u64,
    files: Vec<(Loc, Loc)>,
}

#[derive(Serialize, Deserialize, Default)]
struct Snap {
    version: u32,
    root: String,
    engine: String,
    max_bytes: usize,
    lsn: u64,
    sequence: u64,
    acked: u64,
    /// The record store (file name in the state directory) every `Loc` refers to.
    store: String,
    files: BTreeMap<String, FileRec>,
    pending: Vec<PendingDelta>,
    /// File selection rules of the last run (absent in states from earlier versions).
    #[serde(default)]
    rules: Option<FileRules>,
}

#[derive(Serialize, Deserialize)]
struct Record {
    lsn: u64,
    #[serde(default)]
    delta: Option<PendingDelta>,
    #[serde(default)]
    set: Vec<(String, FileRec)>,
    #[serde(default)]
    del: Vec<String>,
    #[serde(default)]
    ack: Option<u64>,
}

// ---- Record store entries ----
//
// One file version = its chunks, each as: u32 LE JSON length, u8 id length,
// 32-byte SHA-256 of the chunk's raw source bytes, then the Chunk's JSON exactly as
// serde_json writes it. The id is `json[7..7 + id length]` (`{"id":"<id>",...`).
const HEAD: usize = 4 + 1 + 32;
const ID_AT: usize = 7;

fn encode_entry(chunks: &[Chunk], text: &str, out: &mut Vec<u8>) -> Result<(), Error> {
    for c in chunks {
        let start = out.len();
        out.extend_from_slice(&[0; HEAD]);
        serde_json::to_writer(&mut *out, c)?;
        let json = &out[start + HEAD..];
        let id = c.id.as_bytes();
        if id.len() > 255
            || !json.starts_with(b"{\"id\":\"")
            || json.get(ID_AT..ID_AT + id.len()) != Some(id)
            || json.get(ID_AT + id.len()) != Some(&b'"')
        {
            return Err("chunk id does not serialize verbatim".into());
        }
        let json_len = u32::try_from(json.len()).map_err(|_| "chunk record too large")?;
        out[start..start + 4].copy_from_slice(&json_len.to_le_bytes());
        out[start + 4] = id.len() as u8;
        let fp = Sha256::digest(&text.as_bytes()[c.start_byte..c.end_byte]);
        out[start + 5..start + HEAD].copy_from_slice(&fp);
    }
    Ok(())
}

struct EntryChunk<'a> {
    json: &'a [u8],
    id: &'a [u8],
    fp: &'a [u8],
}
fn decode_entry(buf: &[u8]) -> Result<Vec<EntryChunk<'_>>, Error> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < buf.len() {
        let head = buf.get(at..at + HEAD).ok_or("truncated chunk record")?;
        let n = u32::from_le_bytes(head[..4].try_into().unwrap()) as usize;
        let json = buf
            .get(at + HEAD..at + HEAD + n)
            .ok_or("truncated chunk record")?;
        let id = json
            .get(ID_AT..ID_AT + head[4] as usize)
            .ok_or("bad chunk record")?;
        out.push(EntryChunk {
            json,
            id,
            fp: &head[5..],
        });
        at += HEAD + n;
    }
    Ok(out)
}

/// The original watcher's per-file diff: chunks of `new` whose id is not in `old`
/// (added) or whose chunk or raw bytes differ (changed), and ids of `old` missing
/// from `new` (removed). Chunk equality is JSON equality (serde_json is
/// deterministic and these fields serialize injectively).
#[derive(Default)]
struct FileDiff<'a> {
    added: Vec<&'a [u8]>,
    changed: Vec<&'a [u8]>,
    removed: Vec<&'a [u8]>,
}
impl FileDiff<'_> {
    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.changed.is_empty() && self.removed.is_empty()
    }
}
fn diff<'a>(old: &[EntryChunk<'a>], new: &[EntryChunk<'a>]) -> FileDiff<'a> {
    let mut d = FileDiff::default();
    if old.is_empty() {
        d.added = new.iter().map(|c| c.json).collect();
        return d;
    }
    if new.is_empty() {
        d.removed = old.iter().map(|c| c.id).collect();
        return d;
    }
    let old_ids: HashMap<&[u8], &EntryChunk> = old.iter().map(|c| (c.id, c)).collect();
    for c in new {
        match old_ids.get(c.id) {
            None => d.added.push(c.json),
            Some(o) if o.json != c.json || o.fp != c.fp => d.changed.push(c.json),
            Some(_) => {}
        }
    }
    let new_ids: HashSet<&[u8]> = new.iter().map(|c| c.id).collect();
    d.removed = old
        .iter()
        .filter(|c| !new_ids.contains(c.id))
        .map(|c| c.id)
        .collect();
    d
}

/// The record store: appends are buffered and written with `pwrite`; `sync` makes
/// everything appended so far durable (before the journal record that refers to it).
struct Store {
    file: File,
    name: String,
    len: u64,
    buf: Vec<u8>,
    dirty: bool,
}
impl Store {
    fn open(dir: &Path, name: &str, keep: u64) -> Result<Self, Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(name))?;
        let len = file.metadata()?.len();
        if len < keep {
            return Err(
                format!("record store {name} is shorter than its committed records").into(),
            );
        }
        if len > keep {
            // Records appended for a transaction whose journal line never committed.
            file.set_len(keep)?;
            file.sync_all()?;
        }
        Ok(Self {
            file,
            name: name.to_owned(),
            len: keep,
            buf: Vec::new(),
            dirty: false,
        })
    }
    fn append(&mut self, bytes: &[u8]) -> Result<Loc, Error> {
        if bytes.is_empty() {
            return Ok(Loc::default());
        }
        let loc = Loc(self.len, bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
        self.len += bytes.len() as u64;
        if self.buf.len() >= 1 << 20 {
            self.write_buf()?;
        }
        Ok(loc)
    }
    fn write_buf(&mut self) -> Result<(), Error> {
        if !self.buf.is_empty() {
            let at = self.len - self.buf.len() as u64;
            self.file.write_all_at(&self.buf, at)?;
            self.buf.clear();
            self.dirty = true;
        }
        Ok(())
    }
    fn sync(&mut self) -> Result<(), Error> {
        self.write_buf()?;
        if self.dirty {
            self.file.sync_data()?;
            self.dirty = false;
        }
        Ok(())
    }
    fn read(&self, loc: Loc) -> Result<Vec<u8>, Error> {
        let mut v = vec![0u8; loc.1 as usize];
        if loc.1 == 0 {
            return Ok(v);
        }
        let buffered = self.len - self.buf.len() as u64;
        if loc.0 >= buffered {
            let at = (loc.0 - buffered) as usize;
            v.copy_from_slice(&self.buf[at..at + loc.1 as usize]);
        } else {
            self.file.read_exact_at(&mut v, loc.0)?;
        }
        Ok(v)
    }
}

/// Writes `d` exactly as `serde_json::to_writer(&Delta)` followed by a newline:
/// `{"sequence":N,"added":[..],"removed":[..],"changed":[..]}`, each list in path
/// order (files) then chunk order.
fn write_delta(store: &Store, d: &PendingDelta, out: &mut dyn Write) -> Result<(), Error> {
    write!(out, "{{\"sequence\":{},\"added\":[", d.sequence)?;
    for (pass, opening) in ["", "],\"removed\":[", "],\"changed\":["]
        .iter()
        .enumerate()
    {
        out.write_all(opening.as_bytes())?;
        let mut first = true;
        for &(old, new) in &d.files {
            // Only the passes that can produce output need the file's records.
            let wanted = match pass {
                0 => new.1 > 0,
                1 => old.1 > 0,
                _ => old.1 > 0 && new.1 > 0,
            };
            if !wanted {
                continue;
            }
            let (ob, nb) = (store.read(old)?, store.read(new)?);
            let (o, n) = (decode_entry(&ob)?, decode_entry(&nb)?);
            let fd = diff(&o, &n);
            let items = match pass {
                0 => fd.added,
                1 => fd.removed,
                _ => fd.changed,
            };
            for item in items {
                if !first {
                    out.write_all(b",")?;
                }
                first = false;
                if pass == 1 {
                    out.write_all(b"\"")?;
                    out.write_all(item)?;
                    out.write_all(b"\"")?;
                } else {
                    out.write_all(item)?;
                }
            }
        }
    }
    out.write_all(b"]}\n")?;
    Ok(())
}

struct Inotify {
    fd: i32,
    wd_dir: HashMap<i32, String>, // wd -> relative dir ("" = root)
    dir_wd: HashMap<String, i32>,
    /// Watches kept for one file name only: excluded CACHEDIR.TAG directories
    /// (`CACHEDIR.TAG`) and, when ignore files are honoured, `.git/info` (`exclude`).
    trigger_only: HashMap<i32, &'static [u8]>,
}
const MASK: u32 = libc::IN_CLOSE_WRITE
    | libc::IN_MODIFY
    | libc::IN_ATTRIB
    | libc::IN_MOVED_FROM
    | libc::IN_MOVED_TO
    | libc::IN_CREATE
    | libc::IN_DELETE
    | libc::IN_DELETE_SELF
    | libc::IN_MOVE_SELF
    | libc::IN_ONLYDIR
    | libc::IN_EXCL_UNLINK;

impl Inotify {
    fn new() -> Result<Self, Error> {
        let fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self {
            fd,
            wd_dir: HashMap::new(),
            dir_wd: HashMap::new(),
            trigger_only: HashMap::new(),
        })
    }
    /// Adds (or refreshes) the watch of `rel`; `None` if it failed, as for walked
    /// directories (a vanished directory's parent event triggers a rescan).
    fn watch(&mut self, root: &Path, rel: &str) -> Option<i32> {
        let abs = if rel.is_empty() {
            root.to_path_buf()
        } else {
            root.join(rel)
        };
        let c = CString::new(abs.as_os_str().as_bytes()).ok()?;
        let wd = unsafe { libc::inotify_add_watch(self.fd, c.as_ptr(), MASK) };
        if wd < 0 {
            return None;
        }
        self.wd_dir.insert(wd, rel.to_owned());
        self.dir_wd.insert(rel.to_owned(), wd);
        Some(wd)
    }
    fn forget(&mut self, wd: i32) {
        if let Some(d) = self.wd_dir.remove(&wd) {
            // The path may have a newer watch (directory replaced): keep that mapping.
            if self.dir_wd.get(&d) == Some(&wd) {
                self.dir_wd.remove(&d);
            }
        }
        self.trigger_only.remove(&wd);
    }
    /// Reads all queued events. Returns (any relevant event, full rescan needed).
    fn drain(
        &mut self,
        dirty: &mut BTreeSet<String>,
        respect_ignore: bool,
    ) -> Result<(bool, bool), Error> {
        let mut buf = vec![0u8; 1 << 16];
        let (mut any, mut full) = (false, false);
        loop {
            let n = unsafe { libc::read(self.fd, buf.as_mut_ptr().cast(), buf.len()) };
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::WouldBlock {
                    return Ok((any, full));
                }
                return Err(e.into());
            }
            let mut at = 0usize;
            while at + 16 <= n as usize {
                let wd = i32::from_ne_bytes(buf[at..at + 4].try_into().unwrap());
                let mask = u32::from_ne_bytes(buf[at + 4..at + 8].try_into().unwrap());
                let len = u32::from_ne_bytes(buf[at + 12..at + 16].try_into().unwrap()) as usize;
                let name = &buf[at + 16..at + 16 + len];
                let name = &name[..name.iter().position(|&b| b == 0).unwrap_or(len)];
                at += 16 + len;
                if mask & libc::IN_Q_OVERFLOW != 0 {
                    (any, full) = (true, true);
                    continue;
                }
                if mask & libc::IN_IGNORED != 0 {
                    self.forget(wd);
                    continue;
                }
                if let Some(trigger) = self.trigger_only.get(&wd) {
                    // A watch kept for one file (a tag, `.git/info/exclude`).
                    if name == *trigger {
                        (any, full) = (true, true);
                    }
                    continue;
                }
                any = true;
                if mask & libc::IN_ISDIR != 0
                    || mask & (libc::IN_DELETE_SELF | libc::IN_MOVE_SELF) != 0
                    || name == b"CACHEDIR.TAG"
                    || (respect_ignore && (name == b".gitignore" || name == b".ignore"))
                {
                    full = true;
                    continue;
                }
                if let Some(d) = self.wd_dir.get(&wd) {
                    dirty.insert(d.clone());
                }
            }
        }
    }
    fn wait(&self, timeout: Duration) -> Result<bool, Error> {
        let mut p = libc::pollfd {
            fd: self.fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = timeout.as_millis().min(i32::MAX as u128) as i32;
        let r = unsafe { libc::poll(&mut p, 1, ms) };
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(e.into());
        }
        Ok(r > 0)
    }
}
impl Drop for Inotify {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

pub struct Watcher2 {
    root: PathBuf,
    dir: PathBuf,
    options: Options,
    snap: Snap,
    wal: BufWriter<File>,
    wal_bytes: u64,
    store: Store,
    /// Total length of the records the file table references (compaction trigger).
    live_bytes: u64,
    /// Compact when the store exceeds twice `live_bytes` plus this.
    store_slack: u64,
    /// Set when a durable write failed midway; every later operation errors.
    poisoned: Option<String>,
    /// Number of tracked paths per (device, inode), for hard-linked siblings.
    inodes: HashMap<(u64, u64), u32>,
    delivered: u64,
    threads: usize,
    ino: Inotify,
    rules: FileRules,
    report: Report,
    rules_changed: bool,
    /// A full reconciliation is due (a tag changed while the last one set its watches).
    pending_full: bool,
    /// `report` changed since `report.json` was written.
    report_dirty: bool,
    /// The report's warnings for files whose parser failed (they were split as text),
    /// by path: kept while the file is unchanged, in this process.
    parse_warnings: BTreeMap<String, String>,
    _lock: File,
    pub stats: BTreeMap<&'static str, f64>,
}

fn write_snapshot(dir: &Path, snap: &Snap) -> Result<(), Error> {
    let tmp = dir.join(format!(".watch2.snap.{}.tmp", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        let mut b = BufWriter::with_capacity(1 << 20, &mut f);
        serde_json::to_writer(&mut b, snap)?;
        b.flush()?;
        drop(b);
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join("watch2.snap"))?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

/// Adjusts the hard-link counts and live record bytes for one table change.
fn track(
    inodes: &mut HashMap<(u64, u64), u32>,
    live: &mut u64,
    old: Option<&FileRec>,
    new: Option<&FileRec>,
) {
    if let Some(o) = old {
        *live -= o.rec.1;
        if o.meta.ino != 0 {
            let key = (o.meta.dev, o.meta.ino);
            if let Some(n) = inodes.get_mut(&key) {
                *n -= 1;
                if *n == 0 {
                    inodes.remove(&key);
                }
            }
        }
    }
    if let Some(r) = new {
        *live += r.rec.1;
        if r.meta.ino != 0 {
            *inodes.entry((r.meta.dev, r.meta.ino)).or_insert(0) += 1;
        }
    }
}

/// Returns freed heap to the OS (glibc keeps it otherwise), so a long-running
/// watcher's RSS reflects its file table, not its startup peak.
fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// One candidate's outcome: `None` if it vanished or is not text; otherwise its new
/// record, with the encoded chunks if its content hash changed.
type Update = Option<(FileRec, Option<Vec<u8>>)>;

/// A candidate's index, [`Update`] and report entry if it was skipped, and, when it was
/// chunked, its parse warning (`Some(None)`: none).
type CandidateRead = (usize, Update, Option<Skipped>, Option<Option<String>>);

impl Watcher2 {
    pub fn open(root: &Path, state_dir: &Path, options: &Options) -> Result<Self, Error> {
        Self::open_with(root, state_dir, options, &FileRules::default())
    }
    /// [`Watcher2::open`] with explicit file selection rules. A restart with rules that
    /// differ from the previous run's reconciles to the new rules (see README).
    pub fn open_with(
        root: &Path,
        state_dir: &Path,
        options: &Options,
        rules: &FileRules,
    ) -> Result<Self, Error> {
        rules.validate()?;
        let t0 = Instant::now();
        let root = root.canonicalize()?;
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
        let root_name = root.to_str().ok_or("root must be UTF-8")?.to_owned();
        let engine = format!(
            "{}-max{}",
            crate::watch::engine_for(options),
            options.max_bytes
        );
        let mut snap = match File::open(dir.join("watch2.snap")) {
            Ok(f) => {
                // Check the version before the shape: older prototype states differ.
                #[derive(Deserialize)]
                struct Version {
                    version: u32,
                }
                let raw = std::fs::read(dir.join("watch2.snap"))?;
                drop(f);
                let v: Version = serde_json::from_slice(&raw)?;
                if v.version != SNAP_VERSION {
                    return Err(format!(
                        "watch2 state version {} is from an earlier prototype; remove {} to rebuild it",
                        v.version,
                        dir.display()
                    )
                    .into());
                }
                serde_json::from_slice(&raw)?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Snap {
                version: SNAP_VERSION,
                root: root_name.clone(),
                store: format!("{STORE_PREFIX}0"),
                ..Snap::default()
            },
            Err(e) => return Err(e.into()),
        };
        if snap.root != root_name {
            return Err("watch state root mismatch".into());
        }
        let load_s = t0.elapsed().as_secs_f64();
        // Replay the journal; a torn final line was never acknowledged as committed.
        let snap_existed = dir.join("watch2.snap").is_file();
        let mut replayed = 0usize;
        let wal_path = dir.join("watch2.wal");
        if let Ok(f) = File::open(&wal_path) {
            for line in BufReader::with_capacity(1 << 20, f).split(b'\n') {
                let line = line?;
                let Ok(r) = serde_json::from_slice::<Record>(&line) else {
                    break;
                };
                if r.lsn <= snap.lsn {
                    continue;
                }
                apply(&mut snap, r);
                replayed += 1;
            }
        }
        // Different engine/options: drop cached partitions (reparse) but keep sequences;
        // the old records stay readable for the diff.
        let engine_changed = snap.engine != engine;
        if engine_changed {
            for rec in snap.files.values_mut() {
                rec.hash = Hash32::default();
                rec.meta = Meta::default();
            }
            snap.engine = engine;
            snap.max_bytes = options.max_bytes;
        }
        // Different rules (or a state from before rules were recorded): the startup
        // reconciliation applies the current rules like any offline change.
        let rules_changed = snap_existed && snap.rules.as_ref() != Some(rules);
        snap.rules = Some(rules.clone());
        // Everything committed ends at the last referenced record.
        let keep = snap
            .files
            .values()
            .map(|r| r.rec.end())
            .chain(
                snap.pending
                    .iter()
                    .flat_map(|d| d.files.iter().flat_map(|(o, n)| [o.end(), n.end()])),
            )
            .max()
            .unwrap_or(0);
        let store = Store::open(&dir, &snap.store, keep)?;
        // Fold the journal into a fresh checkpoint, then start an empty journal.
        if !snap_existed || replayed > 0 || engine_changed || rules_changed {
            write_snapshot(&dir, &snap)?;
        }
        // Stores left by an interrupted compaction.
        for e in std::fs::read_dir(&dir)? {
            let name = e?.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(STORE_PREFIX) && name != snap.store.as_str() {
                std::fs::remove_file(dir.join(name.as_ref()))?;
            }
        }
        let wal_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&wal_path)?;
        wal_file.sync_all()?;
        File::open(&dir)?.sync_all()?;
        let delivered = snap.acked;
        let (mut inodes, mut live_bytes) = (HashMap::new(), 0);
        for r in snap.files.values() {
            track(&mut inodes, &mut live_bytes, None, Some(r));
        }
        let mut w = Self {
            root,
            dir,
            options: options.clone(),
            snap,
            wal: BufWriter::with_capacity(1 << 20, wal_file),
            wal_bytes: 0,
            store,
            live_bytes,
            store_slack: STORE_SLACK_BYTES,
            poisoned: None,
            inodes,
            delivered,
            threads: par::threads(),
            ino: Inotify::new()?,
            rules: rules.clone(),
            report: Report::default(),
            rules_changed,
            pending_full: false,
            report_dirty: false,
            parse_warnings: BTreeMap::new(),
            _lock: lock,
            stats: BTreeMap::new(),
        };
        w.stats.insert("open_load_replay_s", load_s);
        let t = Instant::now();
        w.reconcile(None)?;
        w.stats
            .insert("open_reconcile_s", t.elapsed().as_secs_f64());
        w.write_report()?;
        release_memory();
        Ok(w)
    }

    /// What the latest reconciliation indexed and skipped; also kept in
    /// `STATE_DIR/report.json`. Pattern counts are refreshed by full reconciliations.
    pub fn report(&self) -> &Report {
        &self.report
    }
    /// Whether this open found a state written with different file selection rules
    /// (or by a version that did not record them); the open reconciled to the new rules.
    pub fn rules_changed(&self) -> bool {
        self.rules_changed
    }

    fn append(&mut self, mut r: Record) -> Result<(), Error> {
        self.healthy()?;
        r.lsn = self.snap.lsn + 1;
        let mut line = serde_json::to_vec(&r)?;
        line.push(b'\n');
        // Once bytes may have reached the journal, a failure leaves its contents
        // uncertain: refuse further work until reopen (replay decides).
        let written = self
            .wal
            .write_all(&line)
            .and_then(|()| self.wal.flush())
            .and_then(|()| self.wal.get_ref().sync_data());
        if let Err(e) = written {
            self.poisoned = Some(format!("journal append: {e}"));
            return Err(e.into());
        }
        self.wal_bytes += line.len() as u64;
        for (p, rec) in &r.set {
            track(
                &mut self.inodes,
                &mut self.live_bytes,
                self.snap.files.get(p),
                Some(rec),
            );
        }
        for p in &r.del {
            track(
                &mut self.inodes,
                &mut self.live_bytes,
                self.snap.files.get(p),
                None,
            );
        }
        apply(&mut self.snap, r);
        if self.wal_bytes > WAL_CHECKPOINT_BYTES {
            self.checkpoint()?;
        }
        self.maybe_compact()
    }

    fn healthy(&self) -> Result<(), Error> {
        match &self.poisoned {
            Some(why) => Err(format!(
                "watch state uncertain after a failed write ({why}); reopen the watcher"
            )
            .into()),
            None => Ok(()),
        }
    }

    fn checkpoint(&mut self) -> Result<(), Error> {
        write_snapshot(&self.dir, &self.snap)?;
        let f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(self.dir.join("watch2.wal"))?;
        f.sync_all()?;
        self.wal = BufWriter::with_capacity(1 << 20, f);
        self.wal_bytes = 0;
        Ok(())
    }

    /// Copies the live records into a new store once dead versions dominate. Only
    /// while no delta is pending (pending deltas may refer to dead versions).
    fn maybe_compact(&mut self) -> Result<(), Error> {
        if !self.snap.pending.is_empty()
            || self.store.len <= self.live_bytes.saturating_mul(2) + self.store_slack
        {
            return Ok(());
        }
        let n: u64 = self.store.name[STORE_PREFIX.len()..].parse()?;
        let name = format!("{STORE_PREFIX}{}", n + 1);
        let mut next = Store::open(&self.dir, &name, 0)?;
        let mut locs = Vec::with_capacity(self.snap.files.len());
        for r in self.snap.files.values() {
            locs.push(next.append(&self.store.read(r.rec)?)?);
        }
        next.write_buf()?;
        next.file.sync_all()?;
        // Publish: the checkpoint names the new store; the journal restarts empty.
        // If the checkpoint fails, the disk holds either the old checkpoint (old store,
        // journal intact) or the new one; both are consistent, but this instance can no
        // longer tell which, so it refuses further work until reopened.
        for (r, l) in self.snap.files.values_mut().zip(&locs) {
            r.rec = *l;
        }
        let old_name = std::mem::replace(&mut self.snap.store, name);
        self.store = next;
        if let Err(e) = self.checkpoint() {
            self.poisoned = Some(format!("store compaction: {e}"));
            return Err(e);
        }
        std::fs::remove_file(self.dir.join(&old_name))?;
        release_memory();
        Ok(())
    }

    /// `dirs = None`: full reconciliation (walk, lstat all, add watches).
    /// `dirs = Some(set)`: re-list only those directories.
    fn reconcile(&mut self, dirs: Option<&BTreeSet<String>>) -> Result<(), Error> {
        let t = Instant::now();
        // 1. Current inventory of the scope, as relative paths.
        let (scope_present, scope_known, walk_report): (Vec<String>, Vec<String>, Report) =
            match dirs {
                None => {
                    // Watch each directory as the walker accepts it, before its entries are
                    // read, so nothing created after the listing can be missed.
                    // With ignore files honoured, a repository's `.git/info` is watched
                    // (for `exclude` only) before the walk reads it, for the same reason.
                    let fd = self.ino.fd;
                    let respect_ignore = self.rules.respect_ignore;
                    type Added = std::sync::Arc<std::sync::Mutex<Vec<(i32, String)>>>;
                    let added: Added = Default::default();
                    let info_added: Added = Default::default();
                    let (sink, info_sink) = (added.clone(), info_added.clone());
                    let add = move |abs: &Path| -> Option<i32> {
                        let c = CString::new(abs.as_os_str().as_bytes()).ok()?;
                        let wd = unsafe { libc::inotify_add_watch(fd, c.as_ptr(), MASK) };
                        (wd >= 0).then_some(wd)
                    };
                    let on_dir: files::OnDir =
                        std::sync::Arc::new(move |rel: &[u8], abs: &Path| {
                            let rel = String::from_utf8_lossy(rel).into_owned();
                            if respect_ignore
                                && abs.join(".git/info").is_dir()
                                && let Some(wd) = add(&abs.join(".git/info"))
                            {
                                let info = if rel.is_empty() {
                                    ".git/info".to_owned()
                                } else {
                                    format!("{rel}/.git/info")
                                };
                                info_sink.lock().unwrap().push((wd, info));
                            }
                            if let Some(wd) = add(abs) {
                                sink.lock().unwrap().push((wd, rel));
                            }
                        });
                    let walked = files::walk_files(
                        &self.root,
                        Some(&self.dir),
                        &self.rules,
                        None,
                        Some(on_dir),
                    )?;
                    let mut live = HashSet::new();
                    for (wd, rel) in added.lock().unwrap().drain(..) {
                        live.insert(wd);
                        self.ino.dir_wd.insert(rel.clone(), wd);
                        self.ino.wd_dir.insert(wd, rel);
                    }
                    // Excluded CACHEDIR.TAG directories keep a watch for their tag, so that
                    // removing the tag brings their files back. Check the tag again after
                    // the watch exists: a removal in between would otherwise go unseen.
                    let mut trigger_only = HashMap::new();
                    for (wd, rel) in info_added.lock().unwrap().drain(..) {
                        live.insert(wd);
                        trigger_only.insert(wd, &b"exclude"[..]);
                        self.ino.dir_wd.insert(rel.clone(), wd);
                        self.ino.wd_dir.insert(wd, rel);
                    }
                    for s in &walked.report.skipped {
                        if s.reason != SkipReason::CachedirTag {
                            continue;
                        }
                        let rel = s.path.trim_end_matches('/');
                        if let Some(wd) = self.ino.watch(&self.root, rel) {
                            live.insert(wd);
                            trigger_only.insert(wd, &b"CACHEDIR.TAG"[..]);
                        }
                        if !files::is_cachedir_tagged(&self.root.join(rel)) {
                            self.pending_full = true;
                        }
                    }
                    self.ino.trigger_only = trigger_only;
                    // Directories no longer walked (now ignored, excluded or gone).
                    let stale: Vec<i32> = self
                        .ino
                        .wd_dir
                        .keys()
                        .filter(|wd| !live.contains(wd))
                        .copied()
                        .collect();
                    for wd in stale {
                        unsafe { libc::inotify_rm_watch(self.ino.fd, wd) };
                        self.ino.forget(wd);
                    }
                    // The walker lists only UTF-8 names.
                    let present = walked
                        .files
                        .into_iter()
                        .filter_map(|i| String::from_utf8(i.key).ok())
                        .collect();
                    (
                        present,
                        self.snap.files.keys().cloned().collect(),
                        walked.report,
                    )
                }
                Some(dirs) => {
                    let walked = files::walk_files(
                        &self.root,
                        Some(&self.dir),
                        &self.rules,
                        Some(dirs),
                        None,
                    )?;
                    let mut present: Vec<String> = walked
                        .files
                        .into_iter()
                        .filter_map(|i| String::from_utf8(i.key).ok())
                        .collect();
                    let mut known = Vec::new();
                    for d in dirs {
                        let parent_of =
                            |p: &str| p.rsplit_once('/').map_or("", |(a, _)| a).to_owned();
                        let prefix = if d.is_empty() {
                            String::new()
                        } else {
                            format!("{d}/")
                        };
                        known.extend(
                            self.snap
                                .files
                                .range(prefix.clone()..)
                                .take_while(|(k, _)| k.starts_with(&prefix))
                                .filter(|(k, _)| parent_of(k) == *d)
                                .map(|(k, _)| k.clone()),
                        );
                    }
                    // Hard links: a write through one path notifies only that path's
                    // directory, but every other path of the inode changed too (measured).
                    // Inode counts are maintained incrementally; the table is scanned for
                    // siblings only when an in-scope file shares its inode.
                    let in_scope: BTreeSet<&String> = present.iter().chain(known.iter()).collect();
                    let linked: HashSet<(u64, u64)> = in_scope
                        .iter()
                        .filter_map(|p| self.snap.files.get(*p))
                        .map(|r| (r.meta.dev, r.meta.ino))
                        .filter(|k| k.1 != 0 && self.inodes.get(k).is_some_and(|&n| n > 1))
                        .collect();
                    let mut extra: Vec<String> = Vec::new();
                    if !linked.is_empty() {
                        extra = self
                            .snap
                            .files
                            .iter()
                            .filter(|(p, r)| {
                                linked.contains(&(r.meta.dev, r.meta.ino)) && !in_scope.contains(p)
                            })
                            .map(|(p, _)| p.clone())
                            .collect();
                    }
                    present.extend(extra.iter().cloned());
                    known.extend(extra);
                    (present, known, walked.report)
                }
            };
        self.stats.insert("last_list_s", t.elapsed().as_secs_f64());
        // 2. Metadata pre-check in parallel; only changed/racy/new files are read.
        let t = Instant::now();
        let root = self.root.clone();
        let batches: Vec<&[String]> = scope_present.chunks(256).collect();
        let mut candidates: Vec<(String, Meta)> = Vec::new();
        let files_ref = &self.snap.files;
        par::ordered_map(
            &batches,
            self.threads,
            self.threads * 4,
            |b| {
                let mut out = Vec::new();
                for p in b.iter() {
                    match std::fs::symlink_metadata(root.join(p)) {
                        Ok(m) if m.is_file() => {
                            let meta = meta_of(&m);
                            if !files_ref.get(p).is_some_and(|r| r.trusted(&meta)) {
                                out.push((p.clone(), meta));
                            }
                        }
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e.into()),
                    }
                }
                Ok(out)
            },
            |v| {
                candidates.extend(v);
                Ok(())
            },
        )?;
        self.stats.insert("last_stat_s", t.elapsed().as_secs_f64());
        // 3. Read, hash and (only if the hash changed) chunk the candidates in parallel.
        //    New records go to the store in candidate order; each file's diff against
        //    its previous version is computed as it arrives (only "is it empty" is kept).
        let t = Instant::now();
        let options = self.options.clone();
        let store = &mut self.store;
        let old_warnings: BTreeSet<String> = self.parse_warnings.values().cloned().collect();
        let parse_warnings = &mut self.parse_warnings;
        let mut changes: BTreeMap<String, (Option<FileRec>, bool)> = BTreeMap::new();
        let mut read_skips: Vec<Skipped> = Vec::new();
        let weights: Vec<usize> = candidates
            .iter()
            .map(|(_, m)| m.size as usize + 4096)
            .collect();
        let order: Vec<usize> = (0..candidates.len()).collect();
        let cands = &candidates;
        par::ordered_map_weighted(
            &order,
            self.threads,
            par::window_bytes(self.threads),
            |i| weights[i],
            |&i| -> Result<CandidateRead, Error> {
                let (p, meta) = &cands[i];
                let hashed_at = now_ns();
                let item = files::Listed {
                    key: p.clone().into_bytes(),
                    path: root.join(p),
                };
                let data = match files::read_listed(&item, false) {
                    files::Read::Text(_, data) => data,
                    files::Read::Skip(s) => return Ok((i, None, Some(s), None)),
                    files::Read::Gone => return Ok((i, None, None, None)),
                };
                let hash = Hash32(Sha256::digest(data.text.as_bytes()).into());
                // Metadata after the read: a write racing the read changes ctime, so the
                // record stays untrusted and the next pass re-reads it.
                let meta = std::fs::symlink_metadata(&item.path)
                    .map(|m| meta_of(&m))
                    .unwrap_or(*meta);
                if let Some(old) = files_ref.get(p).filter(|r| r.hash == hash) {
                    let rec = FileRec {
                        meta,
                        hashed_at,
                        ..*old
                    };
                    return Ok((i, Some((rec, None)), None, None));
                }
                let (chunks, warning) =
                    crate::syntax::chunk_noted(Path::new(p), &data.text, &options)?;
                let chunks = chunks.chunks;
                let mut entry = Vec::new();
                encode_entry(&chunks, &data.text, &mut entry)?;
                let rec = FileRec {
                    meta,
                    hashed_at,
                    hash,
                    rec: Loc::default(),
                };
                Ok((i, Some((rec, Some(entry))), None, Some(warning)))
            },
            |(i, update, skip, warning)| {
                read_skips.extend(skip);
                let p = &cands[i].0;
                match (update.is_some(), warning) {
                    (_, Some(Some(w))) => {
                        parse_warnings.insert(p.clone(), w);
                    }
                    (false, _) | (true, Some(None)) => {
                        parse_warnings.remove(p);
                    }
                    (true, None) => {}
                }
                let old = files_ref.get(p).map_or(Loc::default(), |r| r.rec);
                let change = match update {
                    None => (None, old.1 > 0),
                    Some((rec, None)) => (Some(rec), false),
                    Some((mut rec, Some(entry))) => {
                        rec.rec = store.append(&entry)?;
                        let before = store.read(old)?;
                        let differs =
                            !diff(&decode_entry(&before)?, &decode_entry(&entry)?).is_empty();
                        (Some(rec), differs)
                    }
                };
                // Long-lived keys are allocated here, on the main thread, not in a worker's
                // arena, where they would pin pages of freed file text.
                changes.insert(p.as_str().to_owned(), change);
                Ok(())
            },
        )?;
        self.stats
            .insert("last_hash_chunk_s", t.elapsed().as_secs_f64());
        self.stats
            .insert("last_candidates", candidates.len() as f64);
        // 4. Per-file diff (chunk IDs are path-qualified, so this equals a global diff).
        let present: BTreeSet<&String> = scope_present.iter().collect();
        for p in &scope_known {
            if !present.contains(p) {
                self.parse_warnings.remove(p);
                let old = self.snap.files.get(p).map_or(Loc::default(), |r| r.rec);
                changes.insert(p.clone(), (None, old.1 > 0));
            }
        }
        let mut delta = PendingDelta::default();
        let mut set = Vec::new();
        let mut del = Vec::new();
        for (p, (rec, differs)) in changes {
            let old = self.snap.files.get(&p).map(|r| r.rec);
            if differs {
                let new = rec.as_ref().map_or(Loc::default(), |r| r.rec);
                delta.files.push((old.unwrap_or_default(), new));
            }
            match rec {
                Some(r) => set.push((p, r)),
                None if old.is_some() => del.push(p),
                None => {}
            }
        }
        let has_delta = !delta.files.is_empty();
        if has_delta || !set.is_empty() || !del.is_empty() {
            self.store.sync()?;
            self.append(Record {
                lsn: 0,
                delta: has_delta.then_some(delta),
                set,
                del,
                ack: None,
            })?;
        }
        // 5. The report. A full pass replaces it; a re-listing replaces the entries of
        // the files directly inside the re-listed directories and of the files it read
        // (directory entries change only through full passes, which every directory
        // event, tag change and, when honoured, ignore-file change triggers).
        // Updated in place (no copy of a large report per batch); `report.json` is
        // written by the next `next_change` call, after this batch's delta is delivered.
        let files = self.snap.files.len() as u64;
        match dirs {
            None => {
                let mut report = walk_report;
                report.skipped.extend(read_skips);
                report.files = files;
                report
                    .warnings
                    .extend(self.parse_warnings.values().cloned());
                report.finish();
                if report != self.report {
                    self.report = report;
                    self.report_dirty = true;
                }
            }
            Some(dirs) => {
                let read: HashSet<&str> = candidates.iter().map(|(p, _)| p.as_str()).collect();
                let mut gone = Vec::new();
                self.report.skipped.retain(|s| {
                    let parent = s.path.rsplit_once('/').map_or("", |(d, _)| d);
                    let keep = s.path.ends_with('/')
                        || !(dirs.contains(parent) || read.contains(s.path.as_str()));
                    if !keep {
                        gone.push(s.clone());
                    }
                    keep
                });
                let mut new: Vec<Skipped> = walk_report
                    .skipped
                    .into_iter()
                    .filter(|s| !s.path.ends_with('/'))
                    .chain(read_skips)
                    .collect();
                new.sort();
                new.dedup();
                gone.sort();
                self.report_dirty |= new != gone || self.report.files != files;
                let warnings: BTreeSet<String> = self.parse_warnings.values().cloned().collect();
                if warnings != old_warnings {
                    self.report.warnings.retain(|w| !old_warnings.contains(w));
                    self.report
                        .warnings
                        .extend(self.parse_warnings.values().cloned());
                    self.report_dirty = true;
                }
                self.report.skipped.extend(new);
                self.report.files = files;
                self.report.finish();
            }
        }
        Ok(())
    }

    /// Writes `report.json` if the report changed since it was last written.
    fn write_report(&mut self) -> Result<(), Error> {
        if self.report_dirty {
            files::write_report(&self.dir, &self.report)?;
            self.report_dirty = false;
        }
        Ok(())
    }

    /// Waits up to `timeout` for the next undelivered delta; returns its index in
    /// `snap.pending`.
    fn next_pending(&mut self, timeout: Duration) -> Result<Option<usize>, Error> {
        self.write_report()?;
        let undelivered = |w: &Self| w.snap.pending.iter().position(|d| d.sequence > w.delivered);
        if let Some(i) = undelivered(self) {
            return Ok(Some(i));
        }
        let began = Instant::now();
        let mut dirty = BTreeSet::new();
        let mut full = std::mem::take(&mut self.pending_full);
        let mut first: Option<Instant> = full.then(Instant::now);
        let mut last = Instant::now();
        loop {
            let wait = match first {
                None => timeout.saturating_sub(began.elapsed()),
                Some(f) => QUIET.min(MAX_WAIT.saturating_sub(f.elapsed())),
            };
            if self.ino.wait(wait)? {
                let (any, f) = self.ino.drain(&mut dirty, self.rules.respect_ignore)?;
                full |= f;
                if any {
                    last = Instant::now();
                    first.get_or_insert(last);
                }
            }
            match first {
                None if began.elapsed() >= timeout => return Ok(None),
                None => continue,
                Some(f) if last.elapsed() < QUIET && f.elapsed() < MAX_WAIT => continue,
                Some(_) => {}
            }
            let t = Instant::now();
            if full {
                self.reconcile(None)?;
            } else {
                self.reconcile(Some(&dirty))?;
            }
            if full
                || self
                    .stats
                    .get("last_candidates")
                    .is_some_and(|&n| n >= 256.0)
            {
                let t = Instant::now();
                release_memory();
                self.stats
                    .insert("last_release_s", t.elapsed().as_secs_f64());
            }
            self.stats
                .insert("last_reconcile_s", t.elapsed().as_secs_f64());
            if let Some(i) = undelivered(self) {
                return Ok(Some(i));
            }
            dirty.clear();
            full = std::mem::take(&mut self.pending_full);
            first = full.then(Instant::now);
            if began.elapsed() >= timeout {
                return Ok(None);
            }
        }
    }

    /// Writes the next delta as one JSON line (exactly `serde_json` of [`Delta`] plus a
    /// newline) without holding its chunks in memory; returns its sequence.
    pub fn next_change_to(
        &mut self,
        timeout: Duration,
        out: &mut dyn Write,
    ) -> Result<Option<u64>, Error> {
        self.healthy()?;
        let Some(i) = self.next_pending(timeout)? else {
            return Ok(None);
        };
        let d = &self.snap.pending[i];
        self.delivered = d.sequence;
        write_delta(&self.store, d, out)?;
        Ok(Some(d.sequence))
    }

    pub fn next_change(&mut self, timeout: Duration) -> Result<Option<Delta>, Error> {
        let mut line = Vec::new();
        Ok(match self.next_change_to(timeout, &mut line)? {
            Some(_) => Some(serde_json::from_slice(&line)?),
            None => None,
        })
    }

    pub fn ack(&mut self, sequence: u64) -> Result<(), Error> {
        if sequence > self.snap.sequence {
            return Err("cannot ack a future watch sequence".into());
        }
        if sequence <= self.snap.acked {
            return Ok(());
        }
        self.append(Record {
            lsn: 0,
            delta: None,
            set: vec![],
            del: vec![],
            ack: Some(sequence),
        })?;
        self.delivered = self.delivered.max(sequence);
        Ok(())
    }

    pub fn file_count(&self) -> usize {
        self.snap.files.len()
    }
}

fn apply(snap: &mut Snap, r: Record) {
    snap.lsn = r.lsn;
    if let Some(mut d) = r.delta {
        snap.sequence += 1;
        d.sequence = snap.sequence;
        snap.pending.push(d);
    }
    for (p, rec) in r.set {
        snap.files.insert(p, rec);
    }
    for p in r.del {
        snap.files.remove(&p);
    }
    if let Some(a) = r.ack {
        snap.acked = snap.acked.max(a);
        snap.pending.retain(|d| d.sequence > a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;

    fn apply_delta(model: &mut Map<String, Chunk>, d: &Delta) {
        for id in &d.removed {
            model.remove(id);
        }
        for c in d.added.iter().chain(&d.changed) {
            model.insert(c.id.clone(), c.clone());
        }
    }
    /// Collects deltas until none arrives for `idle`; checks that the materialized
    /// delta re-serializes to the streamed line.
    fn drain(w: &mut Watcher2, model: &mut Map<String, Chunk>, idle: Duration) -> usize {
        let mut n = 0;
        while let Some(d) = w.next_change(idle).unwrap() {
            let mut again = Vec::new();
            w.delivered = d.sequence - 1;
            w.next_change_to(idle, &mut again).unwrap();
            let mut line = serde_json::to_vec(&d).unwrap();
            line.push(b'\n');
            assert_eq!(again, line);
            apply_delta(model, &d);
            w.ack(d.sequence).unwrap();
            n += 1;
        }
        n
    }
    fn fresh(root: &Path) -> Map<String, Chunk> {
        crate::bench_v1::chunk_dir(root, &Options::default())
            .unwrap()
            .into_iter()
            .map(|c| (c.id.clone(), c))
            .collect()
    }

    #[test]
    fn deltas_restart_torn_tail_and_compaction() {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let r = root.path();
        std::fs::create_dir_all(r.join("a/b")).unwrap();
        for i in 0..20 {
            std::fs::write(
                r.join(format!("a/f{i}.rs")),
                format!("fn first_{i}() {{ one(); }}\n\nfn second_{i}() {{ two({i}); }}\n"),
            )
            .unwrap();
        }
        std::fs::write(r.join("a/b/x.py"), "def x():\n    return 1\n").unwrap();
        std::fs::write(r.join("empty.rs"), "").unwrap();
        let options = Options::default();
        let idle = Duration::from_millis(400);
        let mut model = Map::new();
        let mut w = Watcher2::open(r, state.path(), &options).unwrap();
        assert_eq!(drain(&mut w, &mut model, idle), 1);
        assert_eq!(model, fresh(r));
        // Live edits: change, append, delete, create, whitespace-only, hard link.
        std::fs::write(r.join("a/f1.rs"), "fn first_1() { changed(); }\n").unwrap();
        std::fs::write(
            r.join("a/f2.rs"),
            "fn first_2() { one(); }\n\nfn second_2() { two(2); }\n\n\n",
        )
        .unwrap();
        std::fs::remove_file(r.join("a/f3.rs")).unwrap();
        std::fs::write(r.join("a/b/new.rs"), "struct New { a: u8 }\n").unwrap();
        std::fs::hard_link(r.join("a/f4.rs"), r.join("a/b/link.rs")).unwrap();
        assert!(drain(&mut w, &mut model, idle) >= 1);
        assert_eq!(model, fresh(r));
        std::fs::write(r.join("a/f4.rs"), "fn through_link() {}\n").unwrap();
        drain(&mut w, &mut model, idle);
        assert_eq!(model, fresh(r), "hard-linked sibling updated");
        drop(w);
        // Clean restart: nothing to report.
        let mut w = Watcher2::open(r, state.path(), &options).unwrap();
        assert_eq!(drain(&mut w, &mut model, idle), 0);
        drop(w);
        // Offline edits plus an uncommitted (torn) store tail.
        std::fs::write(r.join("a/f5.rs"), "fn offline() {}\n").unwrap();
        std::fs::remove_file(r.join("a/f6.rs")).unwrap();
        let store_path = state.path().join(format!("{STORE_PREFIX}0"));
        let committed = std::fs::metadata(&store_path).unwrap().len();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&store_path)
            .unwrap()
            .write_all(b"garbage from a transaction that never committed")
            .unwrap();
        let mut w = Watcher2::open(r, state.path(), &options).unwrap();
        assert_eq!(drain(&mut w, &mut model, idle), 1);
        assert_eq!(model, fresh(r));
        let bytes = std::fs::read(&store_path).unwrap();
        assert!(bytes.len() as u64 > committed);
        assert!(
            !bytes.windows(7).any(|x| x == b"garbage"),
            "torn tail removed"
        );
        // Compaction once dead versions dominate (no slack): a new store, same results.
        w.store_slack = 0;
        for i in 7..20 {
            std::fs::write(r.join(format!("a/f{i}.rs")), format!("fn v2_{i}() {{}}\n")).unwrap();
        }
        drain(&mut w, &mut model, idle);
        assert_eq!(model, fresh(r));
        assert_ne!(w.snap.store, format!("{STORE_PREFIX}0"));
        assert!(!store_path.exists());
        std::fs::write(r.join("a/f7.rs"), "fn v3() {}\n").unwrap();
        drain(&mut w, &mut model, idle);
        assert_eq!(model, fresh(r));
        drop(w);
        let mut w = Watcher2::open(r, state.path(), &options).unwrap();
        assert_eq!(drain(&mut w, &mut model, idle), 0);
        drop(w);
        assert_eq!(model, fresh(r));
    }

    #[test]
    fn a_failed_parser_is_a_report_warning_while_the_file_is_unchanged() {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let r = root.path();
        std::fs::create_dir(r.join("docs")).unwrap();
        let bad = format!("{}# Notes\n", crate::syntax::TEST_PANIC);
        std::fs::write(r.join("docs/bad.md"), &bad).unwrap();
        std::fs::write(r.join("good.rs"), "fn main() {}\n").unwrap();
        let warning = "docs/bad.md: the markdown parser panicked (a grammar bug (test)), chunked as plain text";
        let idle = Duration::from_millis(400);
        let mut model = Map::new();
        let mut w = Watcher2::open(r, state.path(), &Options::default()).unwrap();
        drain(&mut w, &mut model, idle);
        assert_eq!(w.report().warnings, [warning]);
        std::fs::write(r.join("good.rs"), "fn main() { changed(); }\n").unwrap();
        drain(&mut w, &mut model, idle);
        assert_eq!(w.report().warnings, [warning]);
        std::fs::write(r.join("docs/bad.md"), "# Notes\n").unwrap();
        drain(&mut w, &mut model, idle);
        assert!(w.report().warnings.is_empty());
        std::fs::write(r.join("docs/bad.md"), &bad).unwrap();
        drain(&mut w, &mut model, idle);
        assert_eq!(w.report().warnings, [warning]);
        std::fs::remove_file(r.join("docs/bad.md")).unwrap();
        drain(&mut w, &mut model, idle);
        assert!(w.report().warnings.is_empty());
        assert_eq!(model, fresh(r));
    }

    #[test]
    fn earlier_prototype_state_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(
            state.path().join("watch2.snap"),
            r#"{"version":2,"root":"/x","engine":"","max_bytes":4096,"lsn":0,"sequence":0,"acked":0,"files":{},"pending":[]}"#,
        )
        .unwrap();
        let e = Watcher2::open(root.path(), state.path(), &Options::default())
            .err()
            .unwrap();
        assert!(e.to_string().contains("earlier prototype"), "{e}");
    }
}
