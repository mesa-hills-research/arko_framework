use crate::{BuildSpec, Result, error::invalid};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{ops::Range, path::Path};
use syn::{spanned::Spanned, visit::Visit};

pub fn hash(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}
/// The byte offset of each line of normalized `text`.
fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![];
    if !text.is_empty() {
        starts.push(0);
    }
    starts.extend(
        text.bytes()
            .enumerate()
            .filter(|&(i, c)| c == b'\n' && i + 1 < text.len())
            .map(|(i, _)| i + 1),
    );
    starts
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub path: String,
    pub original_hash: String,
    pub hash: String,
    pub text: String,
    pub line_starts: Vec<usize>,
}
impl Snapshot {
    pub fn new(path: impl Into<String>, bytes: &[u8]) -> Result<Self> {
        let original = std::str::from_utf8(bytes).map_err(|_| invalid("invalid UTF-8"))?;
        if original.contains('\0') {
            return Err(invalid("binary NUL"));
        }
        let text = original.replace("\r\n", "\n").replace('\r', "\n");
        let line_starts = line_starts(&text);
        Ok(Self {
            path: path.into(),
            original_hash: hash(bytes),
            hash: hash(&text),
            text,
            line_starts,
        })
    }
    /// A snapshot as stored: `text` is already normalized, and `hash` and
    /// `original_hash` are its hashes.
    pub(crate) fn stored(
        path: impl Into<String>,
        text: String,
        hash: String,
        original_hash: String,
    ) -> Self {
        let line_starts = line_starts(&text);
        Self {
            path: path.into(),
            original_hash,
            hash,
            text,
            line_starts,
        }
    }
    pub fn rows(&self) -> usize {
        self.line_starts.len()
    }
    pub fn line_end(&self, row: usize) -> Result<usize> {
        if row == 0 || row > self.rows() {
            return Err(invalid(format!("row {row} outside 1..{}", self.rows())));
        }
        Ok(self
            .line_starts
            .get(row)
            .copied()
            .unwrap_or(self.text.len()))
    }
    pub fn row_at(&self, byte: usize) -> usize {
        self.line_starts
            .partition_point(|&start| start <= byte)
            .max(1)
    }
    pub fn line(&self, row: usize) -> Result<&str> {
        let end = self.line_end(row)?;
        Ok(self.text[self.line_starts[row - 1]..end]
            .strip_suffix('\n')
            .unwrap_or(&self.text[self.line_starts[row - 1]..end]))
    }
    pub fn numbered(&self, first: usize, last: usize) -> String {
        let width = self.rows().max(1).to_string().len();
        (first..=last.min(self.rows()))
            .filter_map(|row| {
                self.line(row)
                    .ok()
                    .map(|s| format!("{row:0width$} | {s}\n"))
            })
            .collect()
    }
    pub fn tail_matches(&self, row: usize, tail: &str) -> bool {
        if tail.is_empty() {
            return false;
        }
        self.line_end(row).is_ok_and(|mut end| {
            if self.text.as_bytes().get(end.wrapping_sub(1)) == Some(&b'\n') {
                end -= 1;
            }
            self.text[..end].ends_with(tail)
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Exclusion {
    pub path: String,
    pub reason: String,
}
#[derive(Debug)]
pub struct Manifest {
    pub files: Vec<Snapshot>,
    pub excluded: Vec<Exclusion>,
    pub hash: String,
    /// The directories the scan entered, root-relative with `/` separators;
    /// `""` is the root.
    pub directories: Vec<String>,
}
fn recognized(path: &Path) -> bool {
    let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
    if matches!(
        name,
        "Dockerfile"
            | "Makefile"
            | "CMakeLists.txt"
            | "Justfile"
            | "LICENSE"
            | "README"
            | "AGENTS.md"
    ) {
        return true;
    }
    matches!(
        path.extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "rs" | "py"
            | "pyi"
            | "js"
            | "jsx"
            | "ts"
            | "tsx"
            | "mjs"
            | "cjs"
            | "go"
            | "c"
            | "h"
            | "cc"
            | "cpp"
            | "hpp"
            | "cs"
            | "java"
            | "kt"
            | "swift"
            | "rb"
            | "php"
            | "lua"
            | "sh"
            | "ps1"
            | "sql"
            | "ex"
            | "exs"
            | "erl"
            | "hs"
            | "scala"
            | "vue"
            | "svelte"
            | "html"
            | "css"
            | "scss"
            | "md"
            | "mdx"
            | "rst"
            | "txt"
            | "adoc"
            | "toml"
            | "yaml"
            | "yml"
            | "json"
            | "xml"
            | "proto"
            | "graphql"
            | "tf"
    )
}
/// `path` relative to `root` with `/` separators (lossy for names that are
/// not UTF-8); paths outside `root` as they are.
fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}
/// The path a walk error is about, when it names one.
fn walk_error_path(error: &ignore::Error) -> Option<&Path> {
    match error {
        ignore::Error::WithPath { path, .. } => Some(path),
        ignore::Error::Loop { child, .. } => Some(child),
        ignore::Error::WithDepth { err, .. } | ignore::Error::WithLineNumber { err, .. } => {
            walk_error_path(err)
        }
        ignore::Error::Partial(errors) => errors.iter().find_map(walk_error_path),
        _ => None,
    }
}
/// The exclusion reason of an entry that couldn't be read.
fn unreadable(io: Option<&std::io::Error>, error: &dyn std::fmt::Display) -> String {
    match io.map(std::io::Error::kind) {
        Some(std::io::ErrorKind::PermissionDenied) => "unreadable: permission denied".into(),
        Some(std::io::ErrorKind::NotFound) => "removed during the scan".into(),
        _ => format!("unreadable: {error}"),
    }
}
/// The exclusion reason of a file over [`BuildSpec::max_file_bytes`].
pub const TOO_LARGE_TO_READ: &str = "too large (over max_file_bytes)";
/// The exclusion reason of a file left out under
/// [`OversizePolicy::Skip`](crate::OversizePolicy::Skip).
pub const TOO_LARGE_FOR_A_REQUEST: &str = "too large (over the chunk request budget)";
pub fn scan(spec: &BuildSpec) -> Result<Manifest> {
    let root = std::fs::canonicalize(&spec.root)?;
    if !root.is_dir() {
        return Err(invalid("index root must be a directory"));
    }
    let includes: Vec<_> = spec
        .include
        .iter()
        .map(|s| glob::Pattern::new(s).map_err(|e| invalid(e.to_string())))
        .collect::<Result<_>>()?;
    let excludes: Vec<_> = spec
        .exclude
        .iter()
        .map(|s| glob::Pattern::new(s).map_err(|e| invalid(e.to_string())))
        .collect::<Result<_>>()?;
    let skipped = spec.skip_directories.clone();
    // Ignore files apply even outside a git repository, including those in
    // the root's parent directories; hidden files are indexed.
    let walker = ignore::WalkBuilder::new(&root)
        .hidden(false)
        .follow_links(false)
        .require_git(false)
        .filter_entry(move |e| {
            !e.file_type().is_some_and(|t| t.is_dir())
                || !skipped
                    .iter()
                    .any(|name| e.file_name().to_str() == Some(name.as_str()))
        })
        .build();
    let mut files = vec![];
    let mut excluded = vec![];
    let mut directories = vec![];
    for entry in walker {
        // An entry the walk can't read (a directory without permission, a
        // file that vanished, an ignore file with invalid rules) is reported
        // and the scan continues.
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                excluded.push(Exclusion {
                    path: walk_error_path(&error)
                        .map(|path| relative_path(path, &root))
                        .unwrap_or_default(),
                    reason: unreadable(error.io_error(), &error),
                });
                continue;
            }
        };
        if entry.file_type().is_some_and(|t| t.is_dir()) {
            if let Ok(relative) = entry.path().strip_prefix(&root) {
                directories.push(
                    relative
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/"),
                );
            }
            continue;
        }
        let path = relative_path(entry.path(), &root);
        let valid_name = entry
            .path()
            .strip_prefix(&root)
            .is_ok_and(|relative| relative.to_str().is_some());
        let reason = if entry.file_type().is_some_and(|t| t.is_symlink()) {
            Some("symlink")
        } else if excludes.iter().any(|p| p.matches(&path)) {
            Some("exclude pattern")
        } else if !includes.is_empty() && !includes.iter().any(|p| p.matches(&path)) {
            Some("outside include patterns")
        } else if includes.is_empty() && !recognized(entry.path()) {
            Some("unsupported extension")
        } else if entry
            .metadata()
            .is_ok_and(|metadata| metadata.len() > spec.max_file_bytes)
        {
            Some(TOO_LARGE_TO_READ)
        } else if !valid_name {
            Some("path is not valid UTF-8")
        } else {
            None
        };
        if let Some(reason) = reason {
            excluded.push(Exclusion {
                path,
                reason: reason.into(),
            });
            continue;
        }
        let bytes = match std::fs::read(entry.path()) {
            Ok(bytes) => bytes,
            Err(error) => {
                excluded.push(Exclusion {
                    path,
                    reason: unreadable(Some(&error), &error),
                });
                continue;
            }
        };
        match Snapshot::new(&path, &bytes) {
            Ok(file) => files.push(file),
            Err(error) => excluded.push(Exclusion {
                path,
                reason: error.to_string(),
            }),
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    excluded.sort_by(|a, b| a.path.cmp(&b.path));
    let manifest_hash = hash(serde_json::to_vec(&(
        files
            .iter()
            .map(|s| (&s.path, &s.hash, &s.original_hash))
            .collect::<Vec<_>>(),
        &excluded,
    ))?);
    Ok(Manifest {
        files,
        excluded,
        hash: manifest_hash,
        directories,
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Structure {
    pub status: String,
    pub protected: Vec<Range<usize>>,
}
impl Structure {
    pub fn legal(&self, byte: usize) -> bool {
        !self
            .protected
            .iter()
            .any(|r| r.start < byte && byte < r.end)
    }
}
struct RustGuards<'a> {
    source: &'a Snapshot,
    ranges: Vec<Range<usize>>,
}
impl RustGuards<'_> {
    fn protect(&mut self, span: proc_macro2::Span) {
        let a = span.start();
        let b = span.end();
        if a.line == 0 || b.line > self.source.rows() {
            return;
        }
        let start = self.source.line_starts[a.line - 1];
        let end = self
            .source
            .line_end(b.line)
            .unwrap_or(self.source.text.len());
        self.ranges.push(start..end);
    }
}
impl<'ast> Visit<'ast> for RustGuards<'_> {
    fn visit_item_use(&mut self, n: &'ast syn::ItemUse) {
        self.protect(n.span());
        syn::visit::visit_item_use(self, n);
    }

    fn visit_item_impl(&mut self, n: &'ast syn::ItemImpl) {
        self.protect(n.span());
        syn::visit::visit_item_impl(self, n);
    }

    fn visit_item_trait(&mut self, n: &'ast syn::ItemTrait) {
        self.protect(n.span());
        syn::visit::visit_item_trait(self, n);
    }

    fn visit_item_struct(&mut self, n: &'ast syn::ItemStruct) {
        self.protect(n.span());
        syn::visit::visit_item_struct(self, n);
    }

    fn visit_item_enum(&mut self, n: &'ast syn::ItemEnum) {
        self.protect(n.span());
        syn::visit::visit_item_enum(self, n);
    }

    fn visit_item_mod(&mut self, n: &'ast syn::ItemMod) {
        self.protect(n.span());
        syn::visit::visit_item_mod(self, n);
    }

    fn visit_item_fn(&mut self, n: &'ast syn::ItemFn) {
        self.protect(n.span());
        syn::visit::visit_item_fn(self, n);
    }
    fn visit_impl_item_fn(&mut self, n: &'ast syn::ImplItemFn) {
        self.protect(n.span());
        syn::visit::visit_impl_item_fn(self, n);
    }
    fn visit_trait_item_fn(&mut self, n: &'ast syn::TraitItemFn) {
        self.protect(n.span());
        syn::visit::visit_trait_item_fn(self, n);
    }
    fn visit_stmt(&mut self, n: &'ast syn::Stmt) {
        self.protect(n.span());
        syn::visit::visit_stmt(self, n);
    }
    fn visit_lit_str(&mut self, n: &'ast syn::LitStr) {
        self.protect(n.span());
    }
}
pub fn structure(source: &Snapshot) -> Structure {
    let ext = Path::new(&source.path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if ext == "rs" {
        if let Ok(ast) = syn::parse_file(&source.text) {
            let mut visitor = RustGuards {
                source,
                ranges: vec![],
            };
            visitor.visit_file(&ast);
            return Structure {
                status: "rust-guarded".into(),
                protected: visitor.ranges,
            };
        }
    } else if matches!(ext, "md" | "mdx") {
        use pulldown_cmark::{Event, Parser, Tag};
        let mut ranges = vec![];
        let mut heading = None;
        for (event, range) in Parser::new(&source.text).into_offset_iter() {
            if let Event::Start(tag) = event {
                if matches!(tag, Tag::Heading { .. }) {
                    heading = Some(range.start);
                }
                if matches!(tag, Tag::Paragraph | Tag::CodeBlock(_) | Tag::Item)
                    && let Some(start) = heading.take()
                {
                    ranges.push(start..range.end);
                }
                if matches!(
                    tag,
                    Tag::Paragraph | Tag::CodeBlock(_) | Tag::Item | Tag::Heading { .. }
                ) {
                    ranges.push(range);
                }
            }
        }
        return Structure {
            status: "markdown-guarded".into(),
            protected: ranges,
        };
    }
    Structure {
        status: "unchecked".into(),
        protected: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_snapshot_is_the_snapshot_it_was_made_from() {
        for text in ["", "one line", "a\r\nb\rc\n", "a\n\nb\n", "\n"] {
            let fresh = Snapshot::new("a.txt", text.as_bytes()).unwrap();
            let stored = Snapshot::stored(
                "a.txt",
                fresh.text.clone(),
                fresh.hash.clone(),
                fresh.original_hash.clone(),
            );
            assert_eq!(stored.line_starts, fresh.line_starts, "{text:?}");
            assert_eq!(
                (stored.text, stored.hash, stored.original_hash),
                (fresh.text, fresh.hash, fresh.original_hash)
            );
        }
    }
}
