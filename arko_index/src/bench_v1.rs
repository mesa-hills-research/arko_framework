use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Options {
    pub max_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self { max_bytes: 4096 }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Chunk {
    pub id: String,
    pub path: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub symbol_path: String,
    pub kind: String,
    pub token_estimate: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Delta {
    pub sequence: u64,
    pub added: Vec<Chunk>,
    pub removed: Vec<String>,
    pub changed: Vec<Chunk>,
}
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub use crate::files::{FileRules, Report, SkipReason, Skipped};
pub use crate::watch::Watcher;

pub fn chunk_str(path: &Path, text: &str, options: &Options) -> Result<Vec<Chunk>, Error> {
    crate::syntax::chunk(path, text, options)
}
pub fn chunk_file(path: &Path, options: &Options) -> Result<Vec<Chunk>, Error> {
    let bytes = std::fs::read(path)?;
    let Some(text) = crate::files::text(&bytes) else {
        return Ok(Vec::new());
    };
    chunk_str(
        Path::new(path.file_name().ok_or("file path has no name")?),
        text,
        options,
    )
}
/// Chunks every file the default [`FileRules`] select (see README, "What gets indexed").
/// Files that cannot be indexed are skipped; [`chunk_dir_with`] also returns the report.
pub fn chunk_dir(root: &Path, options: &Options) -> Result<Vec<Chunk>, Error> {
    Ok(chunk_dir_with(root, options, &FileRules::default())?.0)
}
/// [`chunk_dir`] with explicit file selection rules; also returns what was skipped.
pub fn chunk_dir_with(
    root: &Path,
    options: &Options,
    rules: &FileRules,
) -> Result<(Vec<Chunk>, Report), Error> {
    let mut chunks = Vec::new();
    let report = chunk_dir_each_with(root, options, rules, crate::par::threads(), |c| {
        chunks.extend(c);
        Ok(())
    })?;
    Ok((chunks, report))
}
/// Scaling prototype: chunks a directory with `threads` workers and hands each file's
/// chunks to `sink` in the exact order (and with the exact content) of the sequential
/// `read_dir` + `chunk_str` loop. Memory is bounded by the reorder window, not the tree.
pub fn chunk_dir_each<S>(
    root: &Path,
    options: &Options,
    threads: usize,
    sink: S,
) -> Result<(), Error>
where
    S: FnMut(Vec<Chunk>) -> Result<(), Error>,
{
    chunk_dir_each_with(root, options, &FileRules::default(), threads, sink).map(|_| ())
}
/// [`chunk_dir_each`] with explicit file selection rules; returns the report.
pub fn chunk_dir_each_with<S>(
    root: &Path,
    options: &Options,
    rules: &FileRules,
    threads: usize,
    sink: S,
) -> Result<Report, Error>
where
    S: FnMut(Vec<Chunk>) -> Result<(), Error>,
{
    chunk_dir_map_with(root, options, rules, threads, Ok, sink)
}
/// Like [`chunk_dir_each`], but first applies `map` to each file's chunks on the worker
/// thread (for example to render output there), and hands `map`'s results to `sink`
/// in file order. The reorder window is a byte budget (`ARKO_INDEX_WINDOW_MB`, default
/// 32 MB of source per thread).
pub fn chunk_dir_map<R, M, S>(
    root: &Path,
    options: &Options,
    threads: usize,
    map: M,
    sink: S,
) -> Result<(), Error>
where
    R: Send,
    M: Fn(Vec<Chunk>) -> Result<R, Error> + Sync,
    S: FnMut(R) -> Result<(), Error>,
{
    chunk_dir_map_with(root, options, &FileRules::default(), threads, map, sink).map(|_| ())
}
/// [`chunk_dir_map`] with explicit file selection rules; returns the report.
pub fn chunk_dir_map_with<R, M, S>(
    root: &Path,
    options: &Options,
    rules: &FileRules,
    threads: usize,
    map: M,
    mut sink: S,
) -> Result<Report, Error>
where
    R: Send,
    M: Fn(Vec<Chunk>) -> Result<R, Error> + Sync,
    S: FnMut(R) -> Result<(), Error>,
{
    use crate::files::Read;
    let crate::files::Walked { files, mut report } = crate::files::list_files(root, None, rules)?;
    let weights = crate::par::source_weights(&files, threads);
    crate::par::ordered_map_weighted(
        &files,
        threads,
        crate::par::window_bytes(threads),
        |i| weights.get(i).copied().unwrap_or(1),
        |item| match crate::files::read_listed(item, false) {
            Read::Text(path, file) => {
                Ok(Ok(map(chunk_str(Path::new(&path), &file.text, options)?)?))
            }
            Read::Skip(s) => Ok(Err(Some(s))),
            Read::Gone => Ok(Err(None)),
        },
        |r| match r {
            Ok(rendered) => {
                report.files += 1;
                sink(rendered)
            }
            Err(skip) => {
                report.skipped.extend(skip);
                Ok(())
            }
        },
    )?;
    report.finish();
    Ok(report)
}
pub fn open_watch(root: &Path, state_dir: &Path, options: &Options) -> Result<Watcher, Error> {
    Watcher::open(root, state_dir, options)
}
/// [`open_watch`] with explicit file selection rules. A restart with rules that differ
/// from the previous run's reconciles: the first delta removes the chunks of files the
/// new rules exclude and adds the newly included ones.
pub fn open_watch_with(
    root: &Path,
    state_dir: &Path,
    options: &Options,
    rules: &FileRules,
) -> Result<Watcher, Error> {
    Watcher::open_with(root, state_dir, options, rules)
}
