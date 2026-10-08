use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Options {
    pub max_bytes: usize,
    /// Markdown files: the front matter's `description` joins its `title` at the root
    /// of every chunk's scope (`Title — Description › ## Section`). Off by default:
    /// the root is the title alone.
    #[serde(default)]
    pub front_matter_description: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_bytes: 4096,
            front_matter_description: false,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}
/// The version of the chunk JSON format (the `chunk` command's lines, deltas, index
/// documents). Format 1 had the ten fields up to `token_estimate`; format 2 adds
/// `scope`, serialized only when it is not empty. Boundaries and IDs are the same in
/// both formats. Format 3 adds file entries ([`FileMetadata`]) to the `chunk`
/// command's output: a Markdown file's metadata region (front matter, MDX
/// import/export lines), which its chunks leave out.
pub const CHUNK_FORMAT: u32 = 3;

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
    /// The chunk's context: the chain of enclosing definitions from the syntax tree
    /// (`impl Display for Point › fn fmt`, `class X › def y`), or for Markdown the
    /// heading chain at its start (`# Guide › ## Install`), joined with ` › `. Empty
    /// for top-level text. Metadata only: it never affects boundaries or IDs. Absent
    /// from the JSON when empty, and in data written before chunk format 2.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scope: String,
}

impl Chunk {
    /// The context header shown with a search result: `path:start_line-end_line`,
    /// then ` · ` and the scope when there is one.
    pub fn header(&self) -> String {
        let mut header = format!("{}:{}-{}", self.path, self.start_line, self.end_line);
        if !self.scope.is_empty() {
            header.push_str(" · ");
            header.push_str(&self.scope);
        }
        header
    }
}
/// A file's metadata region: bytes that belong to the file's record instead of a
/// chunk. A Markdown file has one when it starts with front matter (a YAML `---` or
/// TOML `+++` block) or, in MDX, with `import`/`export` lines. Every byte of a file is
/// in exactly one chunk or in this region, which always starts at byte 0. As a line of
/// the `chunk` command's output it comes before the file's chunks and has a `file` key
/// where chunks have `id`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileMetadata {
    /// The file's path, as in its chunks.
    pub file: String,
    pub metadata_range: SourceSpan,
    /// The front matter block as written, delimiter lines included; empty when the
    /// region holds only MDX import/export lines.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub front_matter: String,
    /// The front matter's `title`, the root of the file's chunk scopes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The front matter's `description` (in scopes with
    /// [`Options::front_matter_description`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
/// One file's chunks and, when it has one, its metadata region.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileChunks {
    pub metadata: Option<FileMetadata>,
    pub chunks: Vec<Chunk>,
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

/// A file's chunks. A Markdown file's metadata region (see [`FileMetadata`]) is left
/// out; [`chunk_str_with_metadata`] also returns it.
pub fn chunk_str(path: &Path, text: &str, options: &Options) -> Result<Vec<Chunk>, Error> {
    Ok(crate::syntax::chunk(path, text, options)?.chunks)
}
/// [`chunk_str`], with the file's metadata region.
pub fn chunk_str_with_metadata(
    path: &Path,
    text: &str,
    options: &Options,
) -> Result<FileChunks, Error> {
    crate::syntax::chunk(path, text, options)
}
pub fn chunk_file(path: &Path, options: &Options) -> Result<Vec<Chunk>, Error> {
    Ok(chunk_file_with_metadata(path, options)?.chunks)
}
/// [`chunk_file`], with the file's metadata region.
pub fn chunk_file_with_metadata(path: &Path, options: &Options) -> Result<FileChunks, Error> {
    let bytes = std::fs::read(path)?;
    let Some(text) = crate::files::text(&bytes) else {
        return Ok(FileChunks::default());
    };
    chunk_str_with_metadata(
        Path::new(path.file_name().ok_or("file path has no name")?),
        text,
        options,
    )
}
/// Chunks every file the default [`FileRules`] select (see `docs/arko_index-reference.md`,
/// "File selection").
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
    chunk_dir_map_with(root, options, rules, threads, |file| Ok(file.chunks), sink)
}
/// Like [`chunk_dir_each`], but first applies `map` to each file's chunks (and metadata
/// region, see [`FileMetadata`]) on the worker thread (for example to render output
/// there), and hands `map`'s results to `sink` in file order. The reorder window is a byte budget (`ARKO_INDEX_WINDOW_MB`, default
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
    M: Fn(FileChunks) -> Result<R, Error> + Sync,
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
    M: Fn(FileChunks) -> Result<R, Error> + Sync,
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
                let (chunks, note) =
                    crate::syntax::chunk_noted(Path::new(&path), &file.text, options)?;
                Ok(Ok((map(chunks)?, note)))
            }
            Read::Skip(s) => Ok(Err(Some(s))),
            Read::Gone => Ok(Err(None)),
        },
        |r| match r {
            Ok((rendered, note)) => {
                report.files += 1;
                report.warnings.extend(note);
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
