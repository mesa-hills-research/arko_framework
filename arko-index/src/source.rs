use crate::{BuildSpec, Result, error::invalid};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{ops::Range, path::Path};
use syn::{spanned::Spanned, visit::Visit};

pub fn hash(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
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
        let mut line_starts = vec![];
        if !text.is_empty() {
            line_starts.push(0);
        }
        for (i, c) in text.bytes().enumerate() {
            if c == b'\n' && i + 1 < text.len() {
                line_starts.push(i + 1);
            }
        }
        Ok(Self {
            path: path.into(),
            original_hash: hash(bytes),
            hash: hash(&text),
            text,
            line_starts,
        })
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
    let walker = ignore::WalkBuilder::new(&root)
        .hidden(false)
        .follow_links(false)
        .require_git(false)
        .filter_entry(|e| {
            !e.file_type().is_some_and(|t| t.is_dir())
                || !matches!(
                    e.file_name().to_str().unwrap_or(""),
                    ".git"
                        | "target"
                        | "node_modules"
                        | ".runtime"
                        | ".cache"
                        | "__pycache__"
                        | ".venv"
                        | "dist"
                )
        })
        .build();
    let mut files = vec![];
    let mut excluded = vec![];
    for entry in walker {
        let entry = entry.map_err(|e| invalid(format!("source enumeration: {e}")))?;
        if entry.file_type().is_some_and(|t| t.is_dir()) {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(&root)
            .map_err(|_| invalid("source outside root"))?;
        let path = relative
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 source path is unsupported"))?
            .replace(std::path::MAIN_SEPARATOR, "/");
        let reason = if entry.file_type().is_some_and(|t| t.is_symlink()) {
            Some("symlink")
        } else if excludes.iter().any(|p| p.matches(&path)) {
            Some("exclude pattern")
        } else if !includes.is_empty() && !includes.iter().any(|p| p.matches(&path)) {
            Some("outside include patterns")
        } else if includes.is_empty() && !recognized(entry.path()) {
            Some("unsupported extension")
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
        let bytes = std::fs::read(entry.path())?;
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
