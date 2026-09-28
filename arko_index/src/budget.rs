//! Chunk-request budgets: measuring the rendered request, and splitting files
//! whose rendering exceeds the budget into windows at structural boundaries.
use crate::{
    ChunkingOptions, OversizePolicy,
    backend::{ChunkInput, ChunkTokenizer},
    chunker::Window,
    source::{Snapshot, Structure},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The input limit assumed for a chunk model whose limit is unknown (a custom
/// backend, or an OpenRouter model built without the catalog).
pub const FALLBACK_MAX_INPUT_TOKENS: usize = 128_000;
/// The default budget's share of the chunk model's input limit, in percent.
pub const DEFAULT_BUDGET_PERCENT: usize = 10;
/// Bytes per token assumed when a budget in tokens is converted to bytes.
pub const BYTES_PER_TOKEN: usize = 4;
/// Longer lines count as their rendered byte length instead of being
/// tokenized. A token is at least one byte, so this never undercounts.
const LINE_TOKENIZE_LIMIT: usize = 128 * 1024;

/// A job's resolved chunk-request budget, saved in its `snapshot` event as
/// `chunk_budget` and used for the rest of the job, resumes included.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Budget {
    /// The tokenizer that measures `max_tokens`.
    pub tokenizer: Option<ChunkTokenizer>,
    pub max_tokens: Option<usize>,
    pub max_bytes: Option<usize>,
    /// The chunk model's input limit the defaults were derived from.
    pub max_input_tokens: usize,
    /// Whether that limit came from the backend, rather than
    /// [`FALLBACK_MAX_INPUT_TOKENS`].
    pub max_input_known: bool,
    pub oversize: OversizePolicy,
}

impl Budget {
    /// The budget for `options` with a chunk model described by `input`:
    /// explicit limits where set; otherwise [`DEFAULT_BUDGET_PERCENT`] of the
    /// model's input limit, in tokens when the model has a tokenizer and in
    /// bytes ([`BYTES_PER_TOKEN`] per token) when it has none.
    pub fn resolve(options: &ChunkingOptions, input: &ChunkInput) -> Self {
        let max_input_tokens = input.max_input_tokens.unwrap_or(FALLBACK_MAX_INPUT_TOKENS);
        let default_tokens = (max_input_tokens * DEFAULT_BUDGET_PERCENT / 100).max(1);
        let (max_tokens, max_bytes) = match input.tokenizer {
            Some(_) => (
                Some(options.max_request_tokens.unwrap_or(default_tokens)),
                options.max_request_bytes,
            ),
            None => (
                None,
                Some(options.max_request_bytes.unwrap_or(
                    options.max_request_tokens.unwrap_or(default_tokens) * BYTES_PER_TOKEN,
                )),
            ),
        };
        Self {
            tokenizer: input.tokenizer.filter(|_| max_tokens.is_some()),
            max_tokens,
            max_bytes,
            max_input_tokens,
            max_input_known: input.max_input_tokens.is_some(),
            oversize: options.oversize,
        }
    }

    /// The share of the budget `size` takes: the largest of its token and
    /// byte shares. Up to 1.0 fits. Without a token count, bytes stand in for
    /// tokens (a token is at least one byte).
    pub fn share(&self, size: Size) -> f64 {
        let tokens = self.max_tokens.map_or(0.0, |max| {
            size.tokens.unwrap_or(size.bytes) as f64 / max as f64
        });
        let bytes = self
            .max_bytes
            .map_or(0.0, |max| size.bytes as f64 / max as f64);
        tokens.max(bytes)
    }

    pub fn fits(&self, size: Size) -> bool {
        self.share(size) <= 1.0
    }

    /// A short description for messages, such as `27200 tokens`.
    pub fn describe(&self) -> String {
        match (self.max_tokens, self.max_bytes) {
            (Some(tokens), Some(bytes)) => format!("{tokens} tokens and {bytes} bytes"),
            (Some(tokens), None) => format!("{tokens} tokens"),
            (None, Some(bytes)) => format!("{bytes} bytes"),
            (None, None) => "no limit".into(),
        }
    }
}

/// The rendered size of part of a chunk request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Size {
    pub bytes: usize,
    pub tokens: Option<usize>,
}

impl std::ops::Add for Size {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            bytes: self.bytes + other.bytes,
            tokens: match (self.tokens, other.tokens) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
        }
    }
}

/// Counts `text` with `tokenizer`, as the provider does for plain text:
/// special-token text such as `<|endoftext|>` counts as ordinary text.
pub(crate) fn count(tokenizer: ChunkTokenizer, text: &str) -> usize {
    match tokenizer {
        ChunkTokenizer::O200kBase => {
            // The tokenizer's regex gives up on extreme whitespace runs, which
            // lines up to LINE_TOKENIZE_LIMIT stay well below; a failure
            // counts the text's bytes, an upper bound.
            std::panic::catch_unwind(|| tiktoken_rs::o200k_base_singleton().count_ordinary(text))
                .unwrap_or(text.len())
        }
    }
}

/// The header line of a window in the initial request, as
/// [`crate::chunker::request`] renders it.
pub(crate) fn header(id: &str, s: &Snapshot, start: usize, end: usize) -> String {
    format!(
        "FILE {id} path={} snapshot={} bytes={start}..{end} (absolute rows)\n",
        s.path, s.hash
    )
}

/// The rendered byte size of window `w` in an initial request: its header
/// and numbered rows, plus the newline joining it to the previous file.
pub(crate) fn rendered_bytes(s: &Snapshot, w: &Window) -> usize {
    if w.start >= w.end {
        return 0;
    }
    let width = s.rows().max(1).to_string().len();
    let rows = s.row_at(w.start)..=s.row_at(w.end - 1);
    let lines: usize = rows
        .map(|row| width + 3 + s.line(row).map_or(0, str::len) + 1)
        .sum();
    header("f00", s, w.start, w.end).len() + 1 + lines
}

/// The size of window `w` for batching: its saved token count, and its
/// rendered bytes.
pub(crate) fn window_size(s: &Snapshot, w: &Window) -> Size {
    Size {
        bytes: rendered_bytes(s, w),
        tokens: w.tokens,
    }
}

/// Prefix sums of the rendered rows of one file.
struct Sizes {
    bytes: Vec<usize>,
    tokens: Option<Vec<usize>>,
    header: Size,
}

impl Sizes {
    fn measure(s: &Snapshot, tokenizer: Option<ChunkTokenizer>) -> Self {
        let width = s.rows().max(1).to_string().len();
        let mut bytes = Vec::with_capacity(s.rows() + 1);
        let mut tokens = tokenizer.map(|_| Vec::with_capacity(s.rows() + 1));
        bytes.push(0);
        if let Some(tokens) = &mut tokens {
            tokens.push(0);
        }
        let mut row_text = String::new();
        for row in 1..=s.rows() {
            let line = s.line(row).unwrap_or_default();
            let rendered = width + 3 + line.len() + 1;
            bytes.push(bytes[row - 1] + rendered);
            if let (Some(tokens), Some(tokenizer)) = (&mut tokens, tokenizer) {
                let count = if line.len() > LINE_TOKENIZE_LIMIT {
                    rendered
                } else {
                    use std::fmt::Write;
                    row_text.clear();
                    let _ = writeln!(row_text, "{row:0width$} | {line}");
                    count(tokenizer, &row_text)
                };
                tokens.push(tokens[row - 1] + count);
            }
        }
        // The widest header this file can have, plus the joining newline and
        // a little room for a longer file ID.
        let text = header("f00", s, s.text.len(), s.text.len());
        let header = Size {
            bytes: text.len() + 2,
            tokens: tokenizer.map(|tokenizer| count(tokenizer, &text) + 2),
        };
        Self {
            bytes,
            tokens,
            header,
        }
    }

    /// The size of rows `first..=last` as one window.
    fn rows(&self, first: usize, last: usize) -> Size {
        self.header
            + Size {
                bytes: self.bytes[last] - self.bytes[first - 1],
                tokens: self.tokens.as_ref().map(|t| t[last] - t[first - 1]),
            }
    }
}

/// How a file is presented to the chunk model under a budget.
pub(crate) struct Plan {
    /// Contiguous windows covering the file. Empty for an oversize file that
    /// is not split.
    pub windows: Vec<Window>,
    /// The size of the whole file as one window.
    pub size: Size,
    /// Whether the whole file exceeds the budget.
    pub oversize: bool,
    /// Rows that exceed the budget on their own; each is a forced window.
    pub forced_rows: Vec<usize>,
}

/// Plans the windows of `s`, which has at least one row. A file that fits is
/// one window. An oversize file is split under [`OversizePolicy::Split`]:
/// at the strongest structural boundaries that give pieces within the
/// budget, into pieces of similar size; a single row that exceeds the budget
/// becomes a forced window (one chunk, no model request). Other policies get
/// no windows. Windows without a legal internal boundary are marked as
/// needing no model request.
pub(crate) fn plan(s: &Snapshot, structure: &Structure, budget: &Budget) -> Plan {
    let sizes = Sizes::measure(s, budget.tokenizer);
    let rows = s.rows();
    let size = sizes.rows(1, rows);
    let oversize = !budget.fits(size);
    let pieces = if !oversize {
        vec![(1, rows, false)]
    } else if budget.oversize == OversizePolicy::Split {
        let splitter = Splitter {
            sizes: &sizes,
            budget,
            levels: levels(s, structure),
        };
        let mut pieces = vec![];
        splitter.split(1, rows, 0, &mut pieces);
        pieces
    } else {
        vec![]
    };
    let forced_rows = pieces
        .iter()
        .filter(|(_, _, forced)| *forced)
        .map(|(row, _, _)| *row)
        .collect();
    let windows = pieces
        .into_iter()
        .map(|(first, last, forced)| {
            let mut window = Window::new(
                s.line_starts[first - 1],
                s.line_end(last).unwrap_or(s.text.len()),
            );
            window.tokens = sizes.rows(first, last).tokens;
            if forced {
                window.forced = true;
                window.cuts = Some(vec![]);
            } else if !(first..last).any(|row| {
                s.line_end(row).is_ok_and(|end| structure.legal(end))
                    && s.line(row).is_ok_and(|line| !line.trim().is_empty())
            }) {
                // Nothing to choose: the window is one chunk.
                window.cuts = Some(vec![]);
            }
            window
        })
        .collect();
    Plan {
        windows,
        size,
        oversize,
        forced_rows,
    }
}

struct Splitter<'a> {
    sizes: &'a Sizes,
    budget: &'a Budget,
    /// Boundary rows (split before the row), sorted, strongest level first;
    /// the last level holds every row.
    levels: Vec<Vec<usize>>,
}

impl Splitter<'_> {
    fn share(&self, first: usize, last: usize) -> f64 {
        self.budget.share(self.sizes.rows(first, last))
    }

    /// Splits rows `first..=last` into pieces `(first, last, forced)`, using
    /// boundaries of `level` or weaker.
    fn split(&self, first: usize, last: usize, level: usize, out: &mut Vec<(usize, usize, bool)>) {
        if self.share(first, last) <= 1.0 {
            out.push((first, last, false));
            return;
        }
        if first == last {
            out.push((first, last, true));
            return;
        }
        for (index, boundaries) in self.levels.iter().enumerate().skip(level) {
            let from = boundaries.partition_point(|&row| row <= first);
            let to = boundaries.partition_point(|&row| row <= last);
            if from == to {
                continue;
            }
            let mut segments = vec![];
            let mut start = first;
            for &row in &boundaries[from..to] {
                segments.push((start, row - 1));
                start = row;
            }
            segments.push((start, last));
            self.pack(&segments, index + 1, out);
            return;
        }
        // Unreachable: the last level holds every row.
        out.push((first, last, true));
    }

    /// Packs consecutive segments into pieces of similar size within the
    /// budget; a segment that is too large on its own is split further.
    fn pack(&self, segments: &[(usize, usize)], next: usize, out: &mut Vec<(usize, usize, bool)>) {
        let (first, last) = (segments[0].0, segments[segments.len() - 1].1);
        let total = self.share(first, last);
        let target = total / total.ceil().max(1.0);
        let mut pieces: Vec<(usize, usize, bool)> = vec![];
        let mut current: Option<(usize, usize)> = None;
        for &(start, end) in segments {
            if self.share(start, end) > 1.0 {
                pieces.extend(current.take().map(|(a, b)| (a, b, false)));
                self.split(start, end, next, &mut pieces);
                continue;
            }
            current = match current {
                None => Some((start, end)),
                Some((a, b)) => {
                    let with = self.share(a, end);
                    let without = self.share(a, b);
                    if with > 1.0 || (without >= target && with - target > target - without) {
                        pieces.push((a, b, false));
                        Some((start, end))
                    } else {
                        Some((a, end))
                    }
                }
            };
        }
        pieces.extend(current.map(|(a, b)| (a, b, false)));
        // Merge a small piece into a neighbor when both fit together, so a
        // lone closing brace doesn't become its own window.
        let small = target * 0.25;
        let mut merged: Vec<(usize, usize, bool)> = vec![];
        for piece in pieces {
            if let Some(previous) = merged.last_mut()
                && !previous.2
                && !piece.2
                && previous.1 + 1 == piece.0
                && (self.share(previous.0, previous.1) < small
                    || self.share(piece.0, piece.1) < small)
                && self.share(previous.0, piece.1) <= 1.0
            {
                previous.1 = piece.1;
                continue;
            }
            merged.push(piece);
        }
        out.extend(merged);
    }
}

/// Boundary levels of `s`, strongest first; each lists the rows a window may
/// start at. Boundaries that cut no guarded unit come first, the same levels
/// without that restriction next, and every row last.
fn levels(s: &Snapshot, structure: &Structure) -> Vec<Vec<usize>> {
    let ext = Path::new(&s.path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let structural = match ext.as_str() {
        "md" | "mdx" => markdown_levels(s),
        "rs" if structure.status == "rust-guarded" => rust_levels(s, structure),
        _ => code_levels(s, &ext),
    };
    let legal = |row: &usize| structure.legal(s.line_starts[row - 1]);
    let mut levels: Vec<Vec<usize>> = structural
        .iter()
        .map(|level| level.iter().copied().filter(legal).collect())
        .collect();
    levels.push((2..=s.rows()).filter(legal).collect());
    levels.extend(structural);
    levels.push((2..=s.rows()).collect());
    levels.retain(|level| !level.is_empty());
    levels
}

fn is_blank(s: &Snapshot, row: usize) -> bool {
    s.line(row).is_ok_and(|line| line.trim().is_empty())
}

/// Nonblank rows that follow a blank row.
fn after_blank(s: &Snapshot, row: usize) -> bool {
    row >= 2 && !is_blank(s, row) && is_blank(s, row - 1)
}

/// Markdown: before headings of level 1, then of level 2 or higher, and so
/// on to level 6; then before any top-level block. Fenced code is one block.
fn markdown_levels(s: &Snapshot) -> Vec<Vec<usize>> {
    use pulldown_cmark::{Event, Parser, Tag};
    let mut headings: Vec<(usize, usize)> = vec![];
    let mut blocks = vec![];
    let mut depth = 0usize;
    let starts_line = |offset: usize| s.line_starts.binary_search(&offset).ok().map(|i| i + 1);
    for (event, range) in Parser::new(&s.text).into_offset_iter() {
        match event {
            Event::Start(tag) => {
                if depth == 0
                    && let Some(row) = starts_line(range.start).filter(|&row| row >= 2)
                {
                    blocks.push(row);
                    if let Tag::Heading { level, .. } = tag {
                        headings.push((row, level as usize));
                    }
                }
                depth += 1;
            }
            Event::End(_) => depth = depth.saturating_sub(1),
            Event::Rule | Event::Html(_) if depth == 0 => {
                blocks.extend(starts_line(range.start).filter(|&row| row >= 2));
            }
            _ => {}
        }
    }
    let mut levels: Vec<Vec<usize>> = (1..=6)
        .map(|level| {
            headings
                .iter()
                .filter(|(_, l)| *l <= level)
                .map(|(row, _)| *row)
                .collect()
        })
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    levels.push(blocks);
    levels.dedup();
    levels
}

/// Rust files that parse: before items at each nesting depth (top-level
/// items, then members of impls, traits and modules, then statements of
/// function bodies), each followed by the rows after a blank line at that
/// depth. Depth counts the guarded units a boundary is inside.
fn rust_levels(s: &Snapshot, structure: &Structure) -> Vec<Vec<usize>> {
    let Ok(file) = syn::parse_file(&s.text) else {
        return code_levels(s, "rs");
    };
    let mut items: Vec<(usize, usize)> = vec![];
    rust_items(&file.items, 0, &mut items);
    let depths = guard_depths(s, structure);
    let mut levels = vec![];
    for depth in 0..=3 {
        let mut level: Vec<usize> = items
            .iter()
            .filter(|(row, d)| *d == depth && *row >= 2)
            .map(|(row, _)| *row)
            .collect();
        level.sort_unstable();
        level.dedup();
        levels.push(level);
        levels.push(
            (2..=s.rows())
                .filter(|&row| depths[row] == depth && after_blank(s, row))
                .collect(),
        );
    }
    levels
}

fn rust_items(items: &[syn::Item], depth: usize, out: &mut Vec<(usize, usize)>) {
    use syn::spanned::Spanned;
    let row = |span: proc_macro2::Span| span.start().line;
    let statements = |block: &syn::Block, depth: usize, out: &mut Vec<(usize, usize)>| {
        out.extend(block.stmts.iter().map(|stmt| (row(stmt.span()), depth)));
    };
    for item in items {
        out.push((row(item.span()), depth));
        match item {
            syn::Item::Impl(item) => {
                for member in &item.items {
                    out.push((row(member.span()), depth + 1));
                    if let syn::ImplItem::Fn(function) = member {
                        statements(&function.block, depth + 2, out);
                    }
                }
            }
            syn::Item::Trait(item) => {
                out.extend(
                    item.items
                        .iter()
                        .map(|member| (row(member.span()), depth + 1)),
                );
            }
            syn::Item::Mod(item) => {
                if let Some((_, items)) = &item.content {
                    rust_items(items, depth + 1, out);
                }
            }
            syn::Item::Fn(function) => statements(&function.block, depth + 1, out),
            _ => {}
        }
    }
}

/// For each row (index 1..=rows), how many guarded units contain its start.
fn guard_depths(s: &Snapshot, structure: &Structure) -> Vec<usize> {
    let rows = s.rows();
    let mut delta = vec![0isize; rows + 2];
    for range in &structure.protected {
        // Rows whose start byte b satisfies range.start < b < range.end.
        let from = s.line_starts.partition_point(|&b| b <= range.start) + 1;
        let to = s.line_starts.partition_point(|&b| b < range.end) + 1;
        if from < to {
            delta[from] += 1;
            delta[to] -= 1;
        }
    }
    let mut depths = vec![0usize; rows + 1];
    let mut depth = 0isize;
    for row in 1..=rows {
        depth += delta[row];
        depths[row] = depth.max(0) as usize;
    }
    depths
}

/// Other files: for bracket depths 0 to 3, the rows after a blank line at
/// that depth, then the rows at that depth's smallest indentation that don't
/// start with a closer. Depth counts brackets outside strings and line
/// comments; when brackets don't balance, indentation alone is used.
fn code_levels(s: &Snapshot, ext: &str) -> Vec<Vec<usize>> {
    let hash_comments = matches!(
        ext,
        "py" | "pyi"
            | "rb"
            | "sh"
            | "yaml"
            | "yml"
            | "toml"
            | "ex"
            | "exs"
            | "ps1"
            | "tf"
            | "graphql"
            | "txt"
            | ""
    );
    let dash_comments = matches!(ext, "sql" | "lua" | "hs");
    let rows = s.rows();
    let mut depths = vec![0usize; rows + 1];
    let mut depth = 0isize;
    for (row, slot) in depths.iter_mut().enumerate().skip(1) {
        *slot = depth.max(0) as usize;
        let line = s.line(row).unwrap_or_default();
        let mut quote: Option<char> = None;
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            match quote {
                Some(q) => {
                    if c == '\\' {
                        chars.next();
                    } else if c == q {
                        quote = None;
                    }
                }
                None => match c {
                    '"' | '\'' | '`' => quote = Some(c),
                    '/' if chars.peek() == Some(&'/') => break,
                    '#' if hash_comments => break,
                    '-' if dash_comments && chars.peek() == Some(&'-') => break,
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => depth = (depth - 1).max(0),
                    _ => {}
                },
            }
        }
    }
    if depth != 0 {
        depths.iter_mut().for_each(|d| *d = 0);
    }
    let indent = |row: usize| {
        s.line(row)
            .map_or(0, |line| line.len() - line.trim_start().len())
    };
    let closer = |row: usize| {
        let line = s.line(row).unwrap_or_default().trim_start();
        line.starts_with([')', ']', '}'])
            || ["else", "elif", "except", "finally", "end", "catch"]
                .iter()
                .any(|word| {
                    line.strip_prefix(word).is_some_and(|rest| {
                        !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')
                    })
                })
    };
    let mut levels = vec![];
    for level in 0..=3 {
        let at_depth: Vec<usize> = (2..=rows)
            .filter(|&row| depths[row] == level && !is_blank(s, row))
            .collect();
        levels.push(
            at_depth
                .iter()
                .copied()
                .filter(|&row| after_blank(s, row))
                .collect(),
        );
        let openers: Vec<usize> = at_depth.into_iter().filter(|&row| !closer(row)).collect();
        let smallest = openers.iter().map(|&row| indent(row)).min();
        levels.push(
            openers
                .into_iter()
                .filter(|&row| Some(indent(row)) == smallest)
                .collect(),
        );
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::structure;

    fn budget(tokens: Option<usize>, bytes: Option<usize>) -> Budget {
        Budget {
            tokenizer: tokens.map(|_| ChunkTokenizer::O200kBase),
            max_tokens: tokens,
            max_bytes: bytes,
            max_input_tokens: 272_000,
            max_input_known: true,
            oversize: OversizePolicy::Split,
        }
    }

    /// Checks that windows cover `s` exactly, in order, at line starts.
    fn assert_covers(s: &Snapshot, windows: &[Window]) {
        let mut at = 0;
        for w in windows {
            assert_eq!(w.start, at);
            assert!(w.end > w.start);
            assert!(w.end == s.text.len() || s.line_starts.contains(&w.end));
            at = w.end;
        }
        assert_eq!(at, s.text.len());
    }

    #[test]
    fn o200k_counts_match_the_hugging_face_encoding_and_the_api() {
        // The same counts as Xenova/gpt-4o (o200k_base) at revision
        // 7956d98f2a83b2751a98ea7136fdf7fe6cf54e69, which match the input
        // tokens GPT-5.6 and GPT-6 report.
        let o200k = |text: &str| count(ChunkTokenizer::O200kBase, text);
        assert_eq!(
            o200k("The quick brown fox jumps over the lazy dog, then naps in the sun.\n"),
            16
        );
        assert_eq!(
            o200k(
                "fn main() {\n    let total: u64 = (1..=10).map(|n| n * n).sum();\n    println!(\"{total}\");\n}\n"
            ),
            33
        );
        assert_eq!(
            o200k(
                "नमस्ते दुनिया। Привет, мир! 你好，世界。こんにちは世界。مرحبا بالعالم. Γειά σου Κόσμε.\n"
            ),
            33
        );
        assert_eq!(o200k("0042 | fn main() {\n"), 7);
        // Special-token text counts as ordinary text, as the API counts it
        // (the Hugging Face tokenizer counts each such token as one: 12).
        assert_eq!(o200k("Before <|endoftext|> after <|im_start|>user\n"), 17);
        // Rows never merge across their trailing newline, so per-row counts
        // add up to the whole.
        let s = Snapshot::new("a.rs", b"fn a() {}\n\nfn b() {\n    1\n}\n").unwrap();
        let numbered = s.numbered(1, s.rows());
        let rows: usize = numbered.split_inclusive('\n').map(o200k).sum();
        assert_eq!(rows, o200k(&numbered));
    }

    #[test]
    fn extreme_lines_are_counted_without_failing() {
        let spaces = " ".repeat(1 << 20);
        assert!(count(ChunkTokenizer::O200kBase, &spaces) <= spaces.len());
        let text = format!("a\n{spaces}\nb\n");
        let s = Snapshot::new("a.txt", text.as_bytes()).unwrap();
        let sizes = Sizes::measure(&s, Some(ChunkTokenizer::O200kBase));
        // The long row counts as its rendered bytes.
        assert_eq!(
            sizes.rows(2, 2).tokens.unwrap() - sizes.header.tokens.unwrap(),
            (1 << 20) + 5
        );
    }

    #[test]
    fn budgets_default_to_a_tenth_of_the_input_limit() {
        let options = ChunkingOptions::default();
        let gpt = |max| ChunkInput {
            tokenizer: Some(ChunkTokenizer::O200kBase),
            max_input_tokens: max,
        };
        let api = Budget::resolve(&options, &gpt(Some(922_000)));
        assert_eq!((api.max_tokens, api.max_bytes), (Some(92_200), None));
        let oauth = Budget::resolve(&options, &gpt(Some(272_000)));
        assert_eq!((oauth.max_tokens, oauth.max_bytes), (Some(27_200), None));
        // Without a tokenizer: bytes, four per token.
        let claude = ChunkInput {
            tokenizer: None,
            max_input_tokens: Some(1_000_000),
        };
        let bytes = Budget::resolve(&options, &claude);
        assert_eq!((bytes.max_tokens, bytes.max_bytes), (None, Some(400_000)));
        assert_eq!(bytes.tokenizer, None);
        // Unknown limits use the fallback.
        let unknown = Budget::resolve(&options, &ChunkInput::default());
        assert_eq!(unknown.max_bytes, Some(51_200));
        assert!(!unknown.max_input_known);
        // Explicit limits: both apply with a tokenizer; a token limit becomes
        // bytes without one.
        let explicit = ChunkingOptions {
            max_request_tokens: Some(1_000),
            max_request_bytes: Some(3_000),
            ..Default::default()
        };
        let both = Budget::resolve(&explicit, &gpt(Some(922_000)));
        assert_eq!(
            (both.max_tokens, both.max_bytes),
            (Some(1_000), Some(3_000))
        );
        let tokens_only = ChunkingOptions {
            max_request_tokens: Some(1_000),
            ..Default::default()
        };
        assert_eq!(
            Budget::resolve(&tokens_only, &claude).max_bytes,
            Some(4_000)
        );
        // The larger share decides.
        assert!(both.fits(Size {
            bytes: 3_000,
            tokens: Some(1_000)
        }));
        assert!(!both.fits(Size {
            bytes: 3_001,
            tokens: Some(10)
        }));
        assert!(!both.fits(Size {
            bytes: 10,
            tokens: Some(1_001)
        }));
    }

    #[test]
    fn a_file_within_the_budget_is_one_window() {
        let s = Snapshot::new("a.rs", b"fn a() {}\n\nfn b() {}\n").unwrap();
        let plan = plan(&s, &structure(&s), &budget(Some(1_000), None));
        assert!(!plan.oversize);
        assert_eq!(plan.windows.len(), 1);
        assert_eq!(plan.windows[0].tokens, plan.size.tokens);
        assert!(plan.windows[0].cuts.is_none());
        assert_covers(&s, &plan.windows);
        // The window's size is its rendering in the initial request.
        let rendered = crate::chunker::request(&[("f0".into(), &s, &plan.windows[0])]).prompt;
        let exact = count(ChunkTokenizer::O200kBase, &rendered);
        let tokens = plan.size.tokens.unwrap();
        assert!(
            (exact..=exact + 8).contains(&tokens),
            "{tokens} for {exact}"
        );
        assert!((rendered.len()..=rendered.len() + 8).contains(&plan.size.bytes));
        assert_eq!(rendered_bytes(&s, &plan.windows[0]), rendered.len() + 2);
    }

    #[test]
    fn rust_splits_between_items_into_similar_pieces() {
        let function = |n: usize| {
            format!(
                "/// Function {n}.\nfn f{n}() {{\n{}}}\n\n",
                "    let x = 1;\n".repeat(40)
            )
        };
        let text: String = (0..12).map(function).collect();
        let s = Snapshot::new("a.rs", text.as_bytes()).unwrap();
        let guard = structure(&s);
        let whole = plan(&s, &guard, &budget(Some(100_000), None)).size;
        let budget = budget(Some(whole.tokens.unwrap() / 3), None);
        let plan = plan(&s, &guard, &budget);
        assert!(plan.oversize);
        assert_covers(&s, &plan.windows);
        assert!(plan.forced_rows.is_empty());
        assert!(
            (3..=5).contains(&plan.windows.len()),
            "{}",
            plan.windows.len()
        );
        for w in &plan.windows {
            // Every window starts at a function's doc comment and fits.
            assert!(s.text[w.start..].starts_with("/// Function"));
            assert!(guard.legal(w.start));
            assert!(w.tokens.unwrap() <= budget.max_tokens.unwrap());
        }
        let sizes: Vec<_> = plan.windows.iter().map(|w| w.tokens.unwrap()).collect();
        let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
        assert!(max - min <= max / 3, "{sizes:?}");
    }

    #[test]
    fn markdown_splits_at_the_strongest_headings_first() {
        let section = |heading: &str| {
            format!(
                "{heading}\n\n{}\n```sh\n# not a heading\n```\n\n",
                "Words here. ".repeat(60)
            )
        };
        let text = [
            section("# One"),
            section("## One A"),
            section("## One B"),
            section("# Two"),
            section("## Two A"),
            section("## Two B"),
        ]
        .concat();
        let s = Snapshot::new("a.md", text.as_bytes()).unwrap();
        let guard = structure(&s);
        let whole = plan(&s, &guard, &budget(None, Some(1_000_000))).size;
        let plan = plan(&s, &guard, &budget(None, Some(whole.bytes * 6 / 10)));
        assert_covers(&s, &plan.windows);
        let starts: Vec<_> = plan
            .windows
            .iter()
            .map(|w| s.text[w.start..].lines().next().unwrap())
            .collect();
        assert_eq!(starts, ["# One", "# Two"]);
        // A smaller budget uses second-level headings, never a fence line.
        let plan = super::plan(&s, &guard, &budget(None, Some(whole.bytes / 4)));
        assert_covers(&s, &plan.windows);
        for w in &plan.windows {
            assert!(s.text[w.start..].starts_with('#'));
            assert!(!s.text[w.start..].starts_with("# not"));
        }
    }

    #[test]
    fn json_splits_between_keys_by_bracket_depth() {
        let entry = |n: usize| {
            format!(
                "  \"key{n}\": {{\n{}  }}",
                (0..30)
                    .map(|i| format!("    \"field{i}\": \"value {i} {{ ] \",\n"))
                    .collect::<String>()
            )
        };
        let text = format!(
            "{{\n{}\n}}\n",
            (0..8).map(entry).collect::<Vec<_>>().join(",\n")
        );
        let s = Snapshot::new("a.json", text.as_bytes()).unwrap();
        let guard = structure(&s);
        let whole = plan(&s, &guard, &budget(None, Some(10_000_000))).size;
        let plan = plan(&s, &guard, &budget(None, Some(whole.bytes / 3)));
        assert_covers(&s, &plan.windows);
        assert!(plan.windows.len() >= 3);
        for w in &plan.windows[1..] {
            let line = s.text[w.start..].lines().next().unwrap();
            assert!(line.starts_with("  \"key"), "{line}");
        }
    }

    #[test]
    fn a_row_over_the_budget_is_forced_and_the_rest_is_split() {
        let text = format!("start\n{}\nend\n", "x".repeat(5_000));
        let s = Snapshot::new("a.txt", text.as_bytes()).unwrap();
        let plan = plan(&s, &structure(&s), &budget(None, Some(1_000)));
        assert_covers(&s, &plan.windows);
        assert_eq!(plan.forced_rows, [2]);
        let forced: Vec<_> = plan.windows.iter().filter(|w| w.forced).collect();
        assert_eq!(forced.len(), 1);
        assert_eq!(
            &s.text[forced[0].start..forced[0].end],
            format!("{}\n", "x".repeat(5_000))
        );
        assert_eq!(forced[0].cuts, Some(vec![]));
    }

    #[test]
    fn skip_and_block_plan_no_windows_for_oversize_files() {
        let s = Snapshot::new("a.txt", "line\n".repeat(1_000).as_bytes()).unwrap();
        for policy in [OversizePolicy::Skip, OversizePolicy::Block] {
            let budget = Budget {
                oversize: policy,
                ..budget(None, Some(1_000))
            };
            let plan = plan(&s, &structure(&s), &budget);
            assert!(plan.oversize);
            assert!(plan.windows.is_empty());
        }
    }

    #[test]
    fn saved_budgets_round_trip() {
        let budget = budget(Some(27_200), None);
        let saved = serde_json::to_value(&budget).unwrap();
        assert_eq!(saved["tokenizer"], "o200k_base");
        assert_eq!(saved["oversize"], "split");
        assert_eq!(serde_json::from_value::<Budget>(saved).unwrap(), budget);
    }
}
