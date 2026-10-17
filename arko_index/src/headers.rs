//! Context headers: the parts of `Chunk::scope` that do not come from a syntax tree
//! (Markdown heading chains under the document's title) and the bounds every scope
//! keeps. The labels of definitions (`fn f`, `class C`) are made where the tree is
//! walked, in `syntax.rs`.
use std::path::Path;

/// Joins the elements of a scope chain.
pub(crate) const SCOPE_SEPARATOR: &str = " › ";
/// A scope longer than this keeps its innermost elements, after a leading `…`.
pub(crate) const MAX_SCOPE_BYTES: usize = 1000;
/// A label (a name, an impl's type, a heading) longer than this is cut, ending in `…`.
const MAX_LABEL_BYTES: usize = 200;

/// Whitespace runs (newlines included) become one space.
pub(crate) fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// At most [`MAX_LABEL_BYTES`], cut on a character boundary and ending in `…`.
pub(crate) fn clip(mut label: String) -> String {
    if label.len() > MAX_LABEL_BYTES {
        let mut at = MAX_LABEL_BYTES - '…'.len_utf8();
        while !label.is_char_boundary(at) {
            at -= 1;
        }
        label.truncate(at);
        label.push('…');
    }
    label
}

/// At most [`MAX_SCOPE_BYTES`]: a longer chain keeps its innermost elements (from an
/// element boundary when there is one in range) after `… › `.
pub(crate) fn bounded(scope: String) -> String {
    if scope.len() <= MAX_SCOPE_BYTES {
        return scope;
    }
    let prefix = format!("…{SCOPE_SEPARATOR}");
    let mut at = scope.len() - (MAX_SCOPE_BYTES - prefix.len());
    while !scope.is_char_boundary(at) {
        at += 1;
    }
    let tail = &scope[at..];
    let tail = tail
        .find(SCOPE_SEPARATOR)
        .map_or(tail, |i| &tail[i + SCOPE_SEPARATOR.len()..]);
    format!("{prefix}{tail}")
}

/// Markdown files (`.md`, `.markdown`, `.mdx`) get heading chains as their scope.
pub(crate) fn is_markdown(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        e.eq_ignore_ascii_case("md")
            || e.eq_ignore_ascii_case("markdown")
            || e.eq_ignore_ascii_case("mdx")
    })
}

struct Heading {
    /// Byte offset of the heading's (first) line.
    at: usize,
    level: usize,
    title: String,
}

/// Lines with their byte offsets, without line terminators.
fn lines(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        out.push((at, line.trim_end_matches(['\n', '\r'])));
        at += line.len();
    }
    out
}

/// Leading spaces (a tab counts as four: such a line is never a heading).
fn indent(line: &str) -> usize {
    let mut n = 0;
    for b in line.bytes() {
        match b {
            b' ' => n += 1,
            b'\t' => n += 4,
            _ => break,
        }
    }
    n
}

/// The ATX heading on a line (`## Title ##`): level and title.
fn atx(body: &str) -> Option<(usize, String)> {
    let level = body.bytes().take_while(|&b| b == b'#').count();
    let rest = &body[level..];
    if !(1..=6).contains(&level) || !(rest.is_empty() || rest.starts_with([' ', '\t'])) {
        return None;
    }
    let mut title = rest.trim();
    // An optional closing sequence: spaces, then only `#`s.
    let stripped = title.trim_end_matches('#');
    if stripped.is_empty() {
        title = "";
    } else if stripped.len() < title.len() && stripped.ends_with([' ', '\t']) {
        title = stripped.trim_end();
    }
    Some((level, title.to_owned()))
}

/// A line that can be the text of a setext heading: not blank, not indented as code,
/// not a list item, quote, table row, HTML or fence.
fn plain(line: &str) -> bool {
    let body = line.trim_start();
    if body.is_empty() || indent(line) > 3 {
        return false;
    }
    // A list marker is followed by a space, a tab or the end of the line.
    let marker_end = |at: usize| matches!(body.as_bytes().get(at), None | Some(b' ' | b'\t'));
    let first = body.as_bytes()[0];
    if matches!(first, b'>' | b'|' | b'<')
        || (matches!(first, b'-' | b'*' | b'+') && marker_end(1))
        || body.starts_with("```")
        || body.starts_with("~~~")
    {
        return false;
    }
    // Ordered list items: digits, then `.` or `)`, then a space (not `5.1.0`).
    let digits = body.bytes().take_while(u8::is_ascii_digit).count();
    !(digits > 0
        && matches!(body.as_bytes().get(digits), Some(b'.' | b')'))
        && marker_end(digits + 1))
}

/// The headings of a Markdown text: ATX (`# Title`) and setext (a one-line paragraph
/// underlined with `===` or `---`). Headings inside fenced code blocks and a leading
/// YAML (`---`) or TOML (`+++`) front matter block are not headings.
fn headings(text: &str) -> Vec<Heading> {
    let lines = lines(text);
    let mut out = Vec::new();
    let mut i = 0;
    let open = lines
        .first()
        .map(|(_, l)| l.trim_start_matches('\u{feff}').trim_end());
    if let Some(open @ ("---" | "+++")) = open
        && let Some(close) = lines.iter().skip(1).position(|(_, l)| {
            let l = l.trim_end();
            l == open || (open == "---" && l == "...")
        })
    {
        i = close + 2;
    }
    // (fence character, fence length) while inside a fenced code block.
    let mut fence: Option<(u8, usize)> = None;
    // The previous line, when it can be a setext heading's text.
    let mut candidate: Option<usize> = None;
    // Whether the previous line ends a block (start, blank line, heading, fence).
    let mut boundary = true;
    while i < lines.len() {
        let (at, line) = lines[i];
        let body = line.trim_start_matches([' ', '\t']);
        let small_indent = indent(line) <= 3;
        if let Some((ch, n)) = fence {
            let run = body.bytes().take_while(|&b| b == ch).count();
            if small_indent && run >= n && body[run..].trim().is_empty() {
                fence = None;
            }
            (candidate, boundary) = (None, true);
            i += 1;
            continue;
        }
        if small_indent && (body.starts_with("```") || body.starts_with("~~~")) {
            let ch = body.as_bytes()[0];
            fence = Some((ch, body.bytes().take_while(|&b| b == ch).count()));
            (candidate, boundary) = (None, true);
            i += 1;
            continue;
        }
        if small_indent && let Some((level, title)) = atx(body) {
            out.push(Heading {
                at,
                level,
                title: clip(collapse(&title)),
            });
            (candidate, boundary) = (None, true);
            i += 1;
            continue;
        }
        let underline = body.trim_end();
        if small_indent
            && let Some(c) = candidate
            && !underline.is_empty()
            && (underline.bytes().all(|b| b == b'=') || underline.bytes().all(|b| b == b'-'))
        {
            let (text_at, text_line) = lines[c];
            out.push(Heading {
                at: text_at,
                level: if underline.starts_with('=') { 1 } else { 2 },
                title: clip(collapse(text_line)),
            });
            (candidate, boundary) = (None, true);
            i += 1;
            continue;
        }
        let blank = line.trim().is_empty();
        candidate = (boundary && plain(line)).then_some(i);
        boundary = blank;
        i += 1;
    }
    out
}

/// The scope of each range of a Markdown file: the document's root, then the heading
/// chain at the range (see [`markdown_chains`]). The root is `title` (the front
/// matter's), with ` — description` when one is given. Without a title, the first
/// level-1 heading's text. The heading that gives the root, and a level-1 heading
/// that repeats the title, are left out of the chains.
pub(crate) fn markdown_scopes(
    text: &str,
    ranges: impl Iterator<Item = (usize, usize)>,
    title: Option<&str>,
    description: Option<&str>,
) -> Vec<String> {
    let headings = headings(text);
    let first_h1 = headings.iter().position(|h| h.level == 1);
    let (base, skip) = match title {
        Some(title) => (Some(clip(collapse(title))), None),
        None => (first_h1.map(|i| headings[i].title.clone()), first_h1),
    };
    let base = base.filter(|b| !b.is_empty());
    let root = match (base, description.map(collapse).filter(|d| !d.is_empty())) {
        (Some(base), Some(d)) => Some(format!("{base} — {}", clip_to(d, 300))),
        (Some(base), None) => Some(base),
        (None, Some(d)) => Some(clip_to(d, 300)),
        (None, None) => None,
    };
    let repeats = |h: &Heading| {
        h.level == 1 && title.is_some_and(|t| collapse(t).eq_ignore_ascii_case(&h.title))
    };
    let keep: Vec<bool> = headings
        .iter()
        .enumerate()
        .map(|(i, h)| Some(i) != skip && !repeats(h))
        .collect();
    chains(&headings, &keep, text, ranges)
        .into_iter()
        .map(|chain| {
            let scope = match (&root, chain.is_empty()) {
                (Some(root), true) => root.clone(),
                (Some(root), false) => format!("{root}{SCOPE_SEPARATOR}{chain}"),
                (None, _) => chain,
            };
            bounded(scope)
        })
        .collect()
}

/// At most `max` bytes, cut on a character boundary and ending in `…`.
fn clip_to(mut text: String, max: usize) -> String {
    if text.len() > max {
        let mut at = max - '…'.len_utf8();
        while !text.is_char_boundary(at) {
            at -= 1;
        }
        text.truncate(at);
        text.push('…');
    }
    text
}

/// The heading chain of each range `(start, end)` (ascending, not overlapping): the
/// headings in effect at the range's first non-whitespace byte, each closing the
/// headings of its own and deeper levels, as `# A › ## B`. A range that starts before
/// any heading (front matter, a preamble) but contains one gets the chain of its
/// first heading.
#[cfg(test)]
pub(crate) fn markdown_chains(
    text: &str,
    ranges: impl Iterator<Item = (usize, usize)>,
) -> Vec<String> {
    let headings = headings(text);
    let keep = vec![true; headings.len()];
    chains(&headings, &keep, text, ranges)
        .into_iter()
        .map(bounded)
        .collect()
}

/// The chains of [`markdown_chains`], leaving out the headings whose `keep` is false
/// (they still close the headings of their level).
fn chains(
    headings: &[Heading],
    keep: &[bool],
    text: &str,
    ranges: impl Iterator<Item = (usize, usize)>,
) -> Vec<String> {
    // Indices into `headings`.
    let mut stack: Vec<usize> = Vec::new();
    let mut next = 0;
    let mut out = Vec::new();
    for (start, end) in ranges {
        let first = text[start..end]
            .find(|c: char| !c.is_whitespace())
            .map_or(start, |i| start + i);
        while let Some(h) = headings.get(next).filter(|h| h.at <= first) {
            while stack
                .last()
                .is_some_and(|&top| headings[top].level >= h.level)
            {
                stack.pop();
            }
            stack.push(next);
            next += 1;
        }
        let mut chain = stack.clone();
        if chain.is_empty() && headings.get(next).is_some_and(|h| h.at < end) {
            chain.push(next);
        }
        let chain: Vec<String> = chain
            .into_iter()
            .filter(|&i| keep[i])
            .map(|i| {
                let h = &headings[i];
                let hashes = "#".repeat(h.level);
                if h.title.is_empty() {
                    hashes
                } else {
                    format!("{hashes} {}", h.title)
                }
            })
            .collect();
        out.push(chain.join(SCOPE_SEPARATOR));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chains of ranges from each start to the next (the last to the end).
    fn chains(text: &str, starts: &[usize]) -> Vec<String> {
        let ends = starts.iter().skip(1).copied().chain([text.len()]);
        markdown_chains(text, starts.iter().copied().zip(ends))
    }

    #[test]
    fn heading_chains_follow_levels() {
        let text = "# Guide\nintro\n## Install\n### Linux\napt\n## Use\nrun\n";
        let at = |s: &str| text.find(s).unwrap();
        assert_eq!(
            chains(
                text,
                &[0, at("intro"), at("### Linux"), at("apt"), at("## Use")]
            ),
            [
                "# Guide",
                "# Guide",
                "# Guide › ## Install › ### Linux",
                "# Guide › ## Install › ### Linux",
                "# Guide › ## Use",
            ]
        );
    }

    #[test]
    fn fences_front_matter_and_setext() {
        let text = "---\ntitle: x\n# not a heading\n---\nTitle\n=====\n\n```sh\n# comment\n```\nSub\n---\ntext\n";
        let at = |s: &str| text.find(s).unwrap();
        assert_eq!(
            chains(text, &[0, at("Title"), at("```sh"), at("text\n")]),
            ["", "# Title", "# Title", "# Title › ## Sub"]
        );
        // A range from the start that contains the first heading takes its chain.
        assert_eq!(
            chains(text, &[0, at("text\n")]),
            ["# Title", "# Title › ## Sub"]
        );
        // A version number is not a list item; a list item is not a heading's text.
        let text = "5.1.0 / 2025-03-31\n===\n\n* item\n- x\n---\n";
        assert_eq!(
            chains(text, &[text.find("- x").unwrap()]),
            ["# 5.1.0 / 2025-03-31"]
        );
    }

    #[test]
    fn closing_hashes_and_limits() {
        assert_eq!(atx("## Install ##"), Some((2, "Install".into())));
        assert_eq!(atx("# C#"), Some((1, "C#".into())));
        assert_eq!(atx("#hashtag"), None);
        assert_eq!(atx("####### seven"), None);
        let long = format!("x{}", " › y".repeat(400));
        let b = bounded(long);
        assert!(b.len() <= MAX_SCOPE_BYTES && b.starts_with("… › y"), "{b}");
        assert!(clip("é".repeat(150)).ends_with('…'));
        assert!(clip("é".repeat(150)).len() <= MAX_LABEL_BYTES);
    }
}
