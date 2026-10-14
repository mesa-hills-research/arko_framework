//! Which files a directory walk indexes, and the report of what it skipped.
//!
//! Every command and library entry point that walks a tree goes through
//! [`walk_files`], so the rules are the same everywhere (`docs/arko_index-reference.md`,
//! "File selection").
use crate::bench_v1::Error;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read as _,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub(crate) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
/// Makes a rename into `dir` durable: fsyncs the directory on Unix. Windows cannot
/// open a directory with `File::open`; there this does nothing.
pub(crate) fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}
pub(crate) fn text(bytes: &[u8]) -> Option<&str> {
    if bytes.contains(&0) {
        return None;
    }
    std::str::from_utf8(bytes).ok()
}

/// At most this many entries removed by ignore files are listed in `Report::ignored`.
pub const IGNORED_LIST_CAP: usize = 1000;

/// Marker file in every state directory the tool creates (index and watch state).
pub(crate) const STATE_MARKER: &str = ".arko-index-state";
/// The first 43 bytes of a valid `CACHEDIR.TAG` (https://bford.info/cachedir/).
pub(crate) const CACHEDIR_SIGNATURE: &[u8; 43] = b"Signature: 8a477f597d28d172789f06886806bc55";

/// Whether `dir` holds a `CACHEDIR.TAG` file that starts with the standard signature.
pub(crate) fn is_cachedir_tagged(dir: &Path) -> bool {
    let mut head = [0u8; 43];
    std::fs::File::open(dir.join("CACHEDIR.TAG"))
        .and_then(|mut f| f.read_exact(&mut head))
        .is_ok()
        && &head == CACHEDIR_SIGNATURE
}

/// Which files a directory walk indexes.
///
/// The default indexes every regular file under the root except: `.git` directories,
/// directories tagged with a valid `CACHEDIR.TAG`, and the tool's own state directories.
/// Ignore files are not read unless `respect_ignore` is set, and ignore files in
/// directories *above* the root never apply. Symlinks are never followed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct FileRules {
    /// Honour `.gitignore` and `.ignore` files in the root and its subdirectories, and
    /// `.git/info/exclude` of repositories whose `.git` directory is at or below the root
    /// (Git's syntax and precedence, also outside a Git checkout). The global Git
    /// excludes file and ignore files above the root are never read.
    pub respect_ignore: bool,
    /// Gitignore-syntax patterns relative to the root (`vendor/`, `*.min.js`,
    /// `/docs/generated/`, `!keep.md`). A matching file is not indexed; a matching
    /// directory is not entered.
    pub exclude: Vec<String>,
    /// Gitignore-syntax patterns relative to the root. When non-empty, only files
    /// that match one (directly or through one of their directories) are indexed.
    /// A pattern that matches a `CACHEDIR.TAG` directory itself brings it back.
    /// Exclusions always win over inclusions.
    pub include: Vec<String>,
}

/// Why a file or directory under the root was not indexed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum SkipReason {
    /// The file contains a NUL byte.
    Binary,
    /// The file has no NUL byte but is not valid UTF-8 (Latin-1 text, for example).
    NotUtf8,
    /// The file or directory name is not valid UTF-8 (chunk paths must be UTF-8).
    NameNotUtf8,
    /// Reading the file or listing the directory failed (`detail` has the error).
    Unreadable,
    /// A symbolic link; links are never followed.
    Symlink,
    /// Not a regular file, directory or symlink (FIFO, socket, device).
    Special,
    /// A `.git` directory.
    GitDir,
    /// A `.git` file (the `gitdir:` pointer of a submodule or linked worktree).
    GitFile,
    /// A directory holding a valid `CACHEDIR.TAG` (Cargo's `target/`, other caches).
    CachedirTag,
    /// The tool's own index or watch state directory.
    StateDir,
    /// Matched by a `.gitignore` (only with `respect_ignore`; listed in `ignored`).
    Gitignore,
    /// Matched by a `.ignore` file (only with `respect_ignore`; listed in `ignored`).
    IgnoreFile,
    /// Matched by `.git/info/exclude` (only with `respect_ignore`; listed in `ignored`).
    GitInfoExclude,
}
impl SkipReason {
    /// The built-in exclusions (`.git`, `CACHEDIR.TAG`, state directories): reported,
    /// but not worth a line on stderr.
    pub fn is_default_exclusion(self) -> bool {
        matches!(
            self,
            SkipReason::GitDir
                | SkipReason::GitFile
                | SkipReason::CachedirTag
                | SkipReason::StateDir
        )
    }
    pub fn as_str(self) -> &'static str {
        match self {
            SkipReason::Binary => "binary",
            SkipReason::NotUtf8 => "not-utf8",
            SkipReason::NameNotUtf8 => "name-not-utf8",
            SkipReason::Unreadable => "unreadable",
            SkipReason::Symlink => "symlink",
            SkipReason::Special => "special",
            SkipReason::GitDir => "git-dir",
            SkipReason::GitFile => "git-file",
            SkipReason::CachedirTag => "cachedir-tag",
            SkipReason::StateDir => "state-dir",
            SkipReason::Gitignore => "gitignore",
            SkipReason::IgnoreFile => "ignore-file",
            SkipReason::GitInfoExclude => "git-info-exclude",
        }
    }
}

/// One entry under the root that was not indexed.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Skipped {
    /// Relative and slash-separated, as in chunk paths. Directories end with `/` (their
    /// contents were not visited). Bytes that are not UTF-8 are shown as `\xHH`.
    pub path: String,
    pub reason: SkipReason,
    /// The error message, for `unreadable`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// What a walk indexed and what it did not. Everything not indexed for a reason other
/// than the user's own rules is listed in `skipped`. Entries removed by ignore files are
/// counted per source and listed in `ignored` (at most [`IGNORED_LIST_CAP`]); entries
/// removed by `exclude`/`include` are counted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct Report {
    /// The rules in effect.
    pub rules: FileRules,
    /// Files indexed.
    pub files: u64,
    /// Number of `skipped` entries per reason.
    pub counts: BTreeMap<SkipReason, u64>,
    /// Every entry not indexed for a reason other than the user's rules, sorted by path.
    pub skipped: Vec<Skipped>,
    /// Files and directories matching an `exclude` pattern (a directory counts once).
    pub excluded_by_pattern: u64,
    /// Files matching no `include` pattern.
    pub not_included: u64,
    /// Files and directories removed by `.gitignore` files (a directory counts once).
    pub ignored_by_gitignore: u64,
    /// Files and directories removed by `.ignore` files.
    pub ignored_by_ignore_file: u64,
    /// Files and directories removed by `.git/info/exclude`.
    pub ignored_by_git_info_exclude: u64,
    /// The entries removed by ignore files, sorted by path: the first
    /// [`IGNORED_LIST_CAP`]; the three counts above are complete.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ignored: Vec<Skipped>,
    /// Problems that did not skip anything, such as an invalid line in an ignore file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}
impl Report {
    pub(crate) fn new(rules: &FileRules) -> Self {
        Self {
            rules: rules.clone(),
            ..Self::default()
        }
    }
    /// Sorts `skipped` and recomputes `counts`; sorts and caps `ignored`.
    pub(crate) fn finish(&mut self) {
        self.ignored.sort();
        self.ignored.truncate(IGNORED_LIST_CAP);
        self.skipped.sort();
        self.skipped.dedup();
        self.counts.clear();
        for s in &self.skipped {
            *self.counts.entry(s.reason).or_insert(0) += 1;
        }
    }
    /// One line: files indexed, skipped entries per reason, and pattern exclusions.
    pub fn summary(&self) -> String {
        let mut s = format!("{} files indexed", self.files);
        let skipped: u64 = self.counts.values().sum();
        if skipped > 0 {
            let parts: Vec<String> = self
                .counts
                .iter()
                .map(|(r, n)| format!("{n} {}", r.as_str()))
                .collect();
            s += &format!(", {skipped} skipped ({})", parts.join(", "));
        }
        if self.excluded_by_pattern > 0 {
            s += &format!(", {} excluded by pattern", self.excluded_by_pattern);
        }
        if self.not_included > 0 {
            s += &format!(", {} not included", self.not_included);
        }
        let ignored = self.ignored_total();
        if ignored > 0 {
            s += &format!(
                ", {ignored} ignored ({} .gitignore, {} .ignore, {} .git/info/exclude)",
                self.ignored_by_gitignore,
                self.ignored_by_ignore_file,
                self.ignored_by_git_info_exclude
            );
        }
        if self.rules.respect_ignore {
            s += ", ignore files honoured";
        }
        if !self.warnings.is_empty() {
            s += &format!(", {} warnings", self.warnings.len());
        }
        s
    }
    /// Entries removed by ignore files (the sum of the three counts).
    pub fn ignored_total(&self) -> u64 {
        self.ignored_by_gitignore + self.ignored_by_ignore_file + self.ignored_by_git_info_exclude
    }
    /// Whether a one-line summary is worth printing for an interactive command:
    /// something other than the built-in exclusions was skipped, there are warnings,
    /// or nothing was indexed. The user's own rules alone do not count.
    pub fn worth_printing(&self) -> bool {
        self.files == 0
            || !self.warnings.is_empty()
            || self
                .skipped
                .iter()
                .any(|s| !s.reason.is_default_exclusion())
    }
    /// Entries a person is likely to want to know about one by one (everything but
    /// binary files, symlinks and the built-in exclusions other than `CACHEDIR.TAG`,
    /// which are only counted), at most `limit` per reason, then the warnings.
    pub fn notable(&self, limit: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut shown: BTreeMap<SkipReason, usize> = BTreeMap::new();
        for s in &self.skipped {
            if matches!(
                s.reason,
                SkipReason::Binary
                    | SkipReason::Symlink
                    | SkipReason::GitDir
                    | SkipReason::GitFile
                    | SkipReason::StateDir
            ) {
                continue;
            }
            let n = shown.entry(s.reason).or_insert(0);
            *n += 1;
            if *n <= limit {
                match &s.detail {
                    Some(d) => {
                        out.push(format!("skipped ({}): {} ({d})", s.reason.as_str(), s.path))
                    }
                    None => out.push(format!("skipped ({}): {}", s.reason.as_str(), s.path)),
                }
            }
        }
        for (reason, n) in shown {
            if n > limit {
                out.push(format!("... and {} more {}", n - limit, reason.as_str()));
            }
        }
        out.extend(self.warnings.iter().map(|w| format!("warning: {w}")));
        out
    }
}

/// Writes `report.json` into a state directory (replaced atomically, not fsynced: it
/// is rebuilt by every index build and every watcher start).
pub(crate) fn write_report(dir: &Path, report: &Report) -> Result<(), Error> {
    let tmp = dir.join(format!(".report.json.{}.tmp", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, dir.join("report.json"))?;
    Ok(())
}

/// A relative path for the report: slash-joined, invalid UTF-8 bytes as `\xHH`.
fn display(rel: &[u8], dir: bool) -> String {
    let mut s = String::with_capacity(rel.len() + 1);
    for chunk in rel.utf8_chunks() {
        s.push_str(chunk.valid());
        for b in chunk.invalid() {
            s.push_str(&format!("\\x{b:02X}"));
        }
    }
    if dir {
        s.push('/');
    }
    s
}
fn skipped(rel: &[u8], dir: bool, reason: SkipReason, detail: Option<String>) -> Skipped {
    Skipped {
        path: display(rel, dir),
        reason,
        detail,
    }
}

#[derive(Clone)]
pub(crate) struct FileData {
    pub text: String,
    pub hash: String,
}

/// A walked regular file: its relative path as slash-joined raw bytes (the sort key;
/// for UTF-8 names it orders exactly like a `BTreeMap<String, _>`) and its absolute
/// path. Nothing is read yet. The walker only lists files with UTF-8 names.
pub(crate) struct Listed {
    pub key: Vec<u8>,
    pub path: PathBuf,
}

/// The sorted file list of a walk and the report of what the walk did not list.
pub(crate) struct Walked {
    pub files: Vec<Listed>,
    pub report: Report,
}

/// Called for every included directory (relative key, absolute path) before the
/// walker reads its entries, so a watch added here cannot miss later changes.
pub(crate) type OnDir = std::sync::Arc<dyn Fn(&[u8], &Path) + Send + Sync>;

struct Matchers {
    exclude: Option<Gitignore>,
    include: Option<Gitignore>,
}
fn patterns(root: &Path, list: &[String], what: &str) -> Result<Option<Gitignore>, Error> {
    if list.is_empty() {
        return Ok(None);
    }
    let mut b = GitignoreBuilder::new(root);
    for p in list {
        b.add_line(None, p)
            .map_err(|e| format!("invalid {what} pattern {p:?}: {e}"))?;
    }
    Ok(Some(b.build()?))
}
impl FileRules {
    /// Checks that every pattern parses.
    pub fn validate(&self) -> Result<(), Error> {
        self.compile(Path::new("/")).map(|_| ())
    }
    fn compile(&self, root: &Path) -> Result<Matchers, Error> {
        Ok(Matchers {
            exclude: patterns(root, &self.exclude, "exclude")?,
            include: patterns(root, &self.include, "include")?,
        })
    }
}

#[derive(Default)]
struct Tally {
    skipped: Vec<Skipped>,
    excluded_by_pattern: u64,
    not_included: u64,
    ignored: Vec<Skipped>,
    ignored_counts: [u64; 3],
    warnings: Vec<String>,
    /// Ignore matchers of the directories on the current walk path, by depth.
    ignore_stack: Vec<Arc<DirIgnore>>,
}

/// The ignore files of one directory, and its parent's (with `respect_ignore`).
/// The `ignore` crate's own handling is not used, because it hides what it removes;
/// this follows its precedence (ignore-0.4.33 dir.rs `matched_ignore`, with
/// `require_git(false)` and no parents): over the directories from the entry's parent
/// up to the root, the nearest `.ignore` match wins, then the nearest `.gitignore`
/// match, then the nearest `.git/info/exclude` match; a `!` whitelist ends the search.
struct DirIgnore {
    ignore: Option<Gitignore>,
    gitignore: Option<Gitignore>,
    git_exclude: Option<Gitignore>,
    parent: Option<Arc<DirIgnore>>,
}
fn load_ignore_file(dir: &Path, file: &Path, warnings: &mut Vec<String>) -> Option<Gitignore> {
    if !file.is_file() {
        return None;
    }
    let mut b = GitignoreBuilder::new(dir);
    if let Some(e) = b.add(file) {
        warnings.push(e.to_string());
    }
    match b.build() {
        Ok(g) if !g.is_empty() => Some(g),
        Ok(_) => None,
        Err(e) => {
            warnings.push(e.to_string());
            None
        }
    }
}
impl DirIgnore {
    fn load(dir: &Path, parent: Option<Arc<DirIgnore>>, warnings: &mut Vec<String>) -> Self {
        Self {
            ignore: load_ignore_file(dir, &dir.join(".ignore"), warnings),
            gitignore: load_ignore_file(dir, &dir.join(".gitignore"), warnings),
            // Only a `.git` directory at or below the root (like the crate without
            // require_git); a `.git` file's real git directory lies elsewhere.
            git_exclude: load_ignore_file(dir, &dir.join(".git/info/exclude"), warnings),
            parent,
        }
    }
    /// The ignore file kind that removes `path`, if any.
    fn ignored_by(self: &Arc<Self>, path: &Path, is_dir: bool) -> Option<SkipReason> {
        // None = undecided, Some(true) = ignored, Some(false) = whitelisted.
        let decide = |g: &Option<Gitignore>, slot: &mut Option<bool>| {
            if slot.is_none()
                && let Some(g) = g
            {
                let m = g.matched(path, is_dir);
                if !m.is_none() {
                    *slot = Some(m.is_ignore());
                }
            }
        };
        let (mut m_ignore, mut m_gitignore, mut m_exclude) = (None, None, None);
        let mut node = Some(self);
        while let Some(n) = node {
            decide(&n.ignore, &mut m_ignore);
            decide(&n.gitignore, &mut m_gitignore);
            decide(&n.git_exclude, &mut m_exclude);
            node = n.parent.as_ref();
        }
        match (m_ignore, m_gitignore, m_exclude) {
            (Some(true), _, _) => Some(SkipReason::IgnoreFile),
            (Some(false), _, _) => None,
            (None, Some(true), _) => Some(SkipReason::Gitignore),
            (None, Some(false), _) => None,
            (None, None, Some(true)) => Some(SkipReason::GitInfoExclude),
            _ => None,
        }
    }
}

/// Whether a scoped walk (watcher re-listing `scope` directories) needs this entry:
/// directories on the way to a scope directory, and entries directly inside one.
fn in_scope(scope: &BTreeSet<String>, rel: &[u8], is_dir: bool) -> bool {
    let Ok(rel) = std::str::from_utf8(rel) else {
        return false;
    };
    if is_dir {
        let prefix = format!("{rel}/");
        scope.contains(rel)
            || scope
                .range(prefix.clone()..)
                .next()
                .is_some_and(|s| s.starts_with(&prefix))
    } else {
        scope.contains(rel.rsplit_once('/').map_or("", |(dir, _)| dir))
    }
}

/// The underlying I/O error and the path of a walk error, if any.
fn error_path(e: &ignore::Error) -> Option<&Path> {
    match e {
        ignore::Error::WithPath { path, .. } => Some(path),
        ignore::Error::WithDepth { err, .. } | ignore::Error::WithLineNumber { err, .. } => {
            error_path(err)
        }
        ignore::Error::Partial(v) => v.iter().find_map(error_path),
        _ => None,
    }
}

/// Lists the regular files under `root` that the rules select, sorted by path.
///
/// * `state`: the caller's own state directory, never listed.
/// * `scope`: list only the files directly inside these relative directories ("" is
///   the root), walking from the root so that every rule, including ignore files on the
///   way down, applies exactly as in a full walk. Used by the watcher's re-listing.
/// * `on_dir`: see [`OnDir`].
///
/// An entry is attributed to the first rule that removes it, in this order: ignore
/// files (when honoured), symlink, special file, `.git` (directory or file), state
/// directory, `exclude` pattern, `include` list (files), name not UTF-8, `CACHEDIR.TAG`
/// (unless an `include` pattern matches the tagged directory itself).
pub(crate) fn walk_files(
    root: &Path,
    state: Option<&Path>,
    rules: &FileRules,
    scope: Option<&BTreeSet<String>>,
    on_dir: Option<OnDir>,
) -> Result<Walked, Error> {
    if !root.is_dir() {
        return Err(format!("not a directory: {}", root.display()).into());
    }
    let root = root.canonicalize()?;
    let state: Option<PathBuf> = state.map(|p| p.canonicalize()).transpose()?;
    let matchers = rules.compile(&root)?;
    let mut walk = ignore::WalkBuilder::new(&root);
    // The crate reads no ignore files (`DirIgnore` does, when opted in), so ignore
    // files above the root never apply, and the global excludes file is never read.
    walk.hidden(false)
        .parents(false)
        .ignore(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .require_git(false)
        .follow_links(false);
    if let (Some(cb), None) = (&on_dir, scope) {
        cb(b"", &root);
    }
    let respect_ignore = rules.respect_ignore;
    let mut first = Tally::default();
    if respect_ignore {
        let node = DirIgnore::load(&root, None, &mut first.warnings);
        first.ignore_stack.push(Arc::new(node));
    }
    let tally: Arc<Mutex<Tally>> = Arc::new(Mutex::new(first));
    let sink = tally.clone();
    let walk_root = root.clone();
    let scope: Option<BTreeSet<String>> = scope.cloned();
    walk.filter_entry(move |e| {
        if e.depth() == 0 {
            return true;
        }
        let rel = e
            .path()
            .strip_prefix(&walk_root)
            .unwrap_or(e.path())
            .as_os_str()
            .as_encoded_bytes();
        let ft = e.file_type();
        let is_dir = ft.is_some_and(|t| t.is_dir());
        if scope.as_ref().is_some_and(|s| !in_scope(s, rel, is_dir)) {
            return false;
        }
        // The walk is depth-first, so the matchers of this entry's parent are at
        // depth - 1 of the stack (deeper ones belong to finished siblings).
        let depth = e.depth();
        if respect_ignore {
            let mut t = sink.lock().unwrap();
            t.ignore_stack.truncate(depth);
            let reason = t
                .ignore_stack
                .get(depth - 1)
                .and_then(|n| n.ignored_by(e.path(), is_dir));
            if let Some(reason) = reason {
                t.ignored_counts[match reason {
                    SkipReason::Gitignore => 0,
                    SkipReason::IgnoreFile => 1,
                    _ => 2,
                }] += 1;
                t.ignored.push(skipped(rel, is_dir, reason, None));
                return false;
            }
        }
        let skip = |reason: SkipReason, dir: bool| {
            sink.lock()
                .unwrap()
                .skipped
                .push(skipped(rel, dir, reason, None));
            false
        };
        if ft.is_some_and(|t| t.is_symlink()) {
            return skip(SkipReason::Symlink, false);
        }
        if !is_dir && !ft.is_some_and(|t| t.is_file()) {
            return skip(SkipReason::Special, false);
        }
        if e.file_name() == ".git" {
            return skip(
                if is_dir {
                    SkipReason::GitDir
                } else {
                    SkipReason::GitFile
                },
                is_dir,
            );
        }
        if state.as_ref().is_some_and(|p| e.path().starts_with(p))
            || (is_dir && e.path().join(STATE_MARKER).is_file())
        {
            return skip(SkipReason::StateDir, is_dir);
        }
        if let Some(m) = &matchers.exclude
            && m.matched(e.path(), is_dir).is_ignore()
        {
            sink.lock().unwrap().excluded_by_pattern += 1;
            return false;
        }
        if !is_dir
            && let Some(m) = &matchers.include
            && !m.matched_path_or_any_parents(e.path(), false).is_ignore()
        {
            sink.lock().unwrap().not_included += 1;
            return false;
        }
        if std::str::from_utf8(e.file_name().as_encoded_bytes()).is_err() {
            return skip(SkipReason::NameNotUtf8, is_dir);
        }
        if is_dir {
            // An include pattern naming the tagged directory itself brings it back.
            if is_cachedir_tagged(e.path())
                && !matchers
                    .include
                    .as_ref()
                    .is_some_and(|m| m.matched(e.path(), true).is_ignore())
            {
                return skip(SkipReason::CachedirTag, true);
            }
            if let Some(cb) = &on_dir {
                cb(rel, e.path());
            }
            if respect_ignore {
                let mut t = sink.lock().unwrap();
                let parent = t.ignore_stack.get(depth - 1).cloned();
                let mut warnings = Vec::new();
                let node = DirIgnore::load(e.path(), parent, &mut warnings);
                t.warnings.extend(warnings);
                t.ignore_stack.truncate(depth);
                t.ignore_stack.push(Arc::new(node));
            }
        }
        true
    });
    let mut files = Vec::new();
    let mut walk_errors = Vec::new();
    let mut warnings = Vec::new();
    for entry in walk.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                if e.io_error()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                {
                    continue; // vanished while walking
                }
                match error_path(&e) {
                    Some(p) if p == root => return Err(e.into()),
                    Some(p) if e.io_error().is_some() => {
                        let rel = p
                            .strip_prefix(&root)
                            .unwrap_or(p)
                            .as_os_str()
                            .as_encoded_bytes();
                        let dir = std::fs::symlink_metadata(p).is_ok_and(|m| m.is_dir());
                        walk_errors.push(skipped(
                            rel,
                            dir,
                            SkipReason::Unreadable,
                            e.io_error().map(|e| e.to_string()),
                        ));
                    }
                    _ => warnings.push(e.to_string()),
                }
                continue;
            }
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let rel = entry.path().strip_prefix(&root)?;
        let mut key = Vec::new();
        for (i, c) in rel.components().enumerate() {
            if i > 0 {
                key.push(b'/');
            }
            key.extend_from_slice(c.as_os_str().as_encoded_bytes());
        }
        files.push(Listed {
            key,
            path: entry.into_path(),
        });
    }
    files.sort_unstable_by(|a, b| a.key.cmp(&b.key));
    let tally = std::mem::take(&mut *tally.lock().unwrap());
    let mut report = Report::new(rules);
    report.skipped = tally.skipped;
    report.skipped.extend(walk_errors);
    report.excluded_by_pattern = tally.excluded_by_pattern;
    report.not_included = tally.not_included;
    [
        report.ignored_by_gitignore,
        report.ignored_by_ignore_file,
        report.ignored_by_git_info_exclude,
    ] = tally.ignored_counts;
    report.ignored = tally.ignored;
    // Ignore-file problems (an invalid line, an unreadable file): the valid lines apply.
    report.warnings = tally.warnings;
    report.warnings.extend(warnings);
    report.finish();
    Ok(Walked { files, report })
}

/// The whole tree's file list (see [`walk_files`]).
pub(crate) fn list_files(
    root: &Path,
    state: Option<&Path>,
    rules: &FileRules,
) -> Result<Walked, Error> {
    walk_files(root, state, rules, None, None)
}

/// The outcome of reading one listed file.
pub(crate) enum Read {
    /// UTF-8 text without NUL bytes: the relative path and the contents.
    Text(String, FileData),
    /// Not indexable; the report entry says why.
    Skip(Skipped),
    /// Vanished since it was listed.
    Gone,
}

/// Reads one listed file: NUL-containing files are binary, other invalid UTF-8 is
/// `not-utf8`, read errors are `unreadable`; a vanished file is `Gone`.
pub(crate) fn read_listed(item: &Listed, with_hash: bool) -> Read {
    let bytes = match std::fs::read(&item.path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Read::Gone,
        Err(e) => {
            return Read::Skip(skipped(
                &item.key,
                false,
                SkipReason::Unreadable,
                Some(e.to_string()),
            ));
        }
    };
    if bytes.contains(&0) {
        return Read::Skip(skipped(&item.key, false, SkipReason::Binary, None));
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return Read::Skip(skipped(&item.key, false, SkipReason::NotUtf8, None));
    };
    let Ok(name) = String::from_utf8(item.key.clone()) else {
        return Read::Skip(skipped(&item.key, false, SkipReason::NameNotUtf8, None));
    };
    // `chunk`/`index` never use the file hash; only the watcher needs it.
    let hash = if with_hash {
        hash(text.as_bytes())
    } else {
        String::new()
    };
    Read::Text(name, FileData { text, hash })
}

/// Walks and reads the whole tree sequentially: path -> text and hash, plus the report.
pub(crate) fn read_dir(
    root: &Path,
    state: Option<&Path>,
    rules: &FileRules,
) -> Result<(BTreeMap<String, FileData>, Report), Error> {
    let Walked { files, mut report } = list_files(root, state, rules)?;
    let mut result = BTreeMap::new();
    for item in &files {
        match read_listed(item, true) {
            Read::Text(path, file) => {
                result.insert(path, file);
            }
            Read::Skip(s) => report.skipped.push(s),
            Read::Gone => {}
        }
    }
    report.files = result.len() as u64;
    report.finish();
    Ok((result, report))
}
