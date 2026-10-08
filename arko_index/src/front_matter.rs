//! A Markdown file's metadata region: its front matter (a YAML `---` or TOML `+++`
//! block at the top) and, in MDX, the `import`/`export` lines after it. The region is
//! file metadata: chunks leave it out, and its `title` (or the first `# H1`) becomes
//! the root of every chunk's scope.

/// The metadata region at the start of a Markdown file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    /// The region is `0..end` (a byte order mark at the start included).
    pub(crate) end: usize,
    /// The front matter block, delimiter lines included (empty when the region holds
    /// only MDX import/export lines).
    pub(crate) front_matter: String,
    pub(crate) title: Option<String>,
    pub(crate) description: Option<String>,
}

/// Lines with their byte offsets, terminators included.
fn lines(text: &str, from: usize) -> impl Iterator<Item = (usize, &str)> {
    let mut at = from;
    text[from..].split_inclusive('\n').map(move |line| {
        let start = at;
        at += line.len();
        (start, line)
    })
}

fn bare(line: &str) -> &str {
    line.trim_end_matches(['\n', '\r'])
}

/// The metadata region of a Markdown (`mdx`: MDX) file, if it has one.
pub(crate) fn region(text: &str, mdx: bool) -> Option<Region> {
    let start = if text.starts_with('\u{feff}') { 3 } else { 0 };
    let mut end = start;
    let mut front_matter = String::new();
    let (mut title, mut description) = (None, None);
    let first = lines(text, start).next().map(|(_, l)| bare(l).trim_end());
    if let Some(open @ ("---" | "+++")) = first {
        let toml = open == "+++";
        let close = lines(text, start).skip(1).find(|(_, l)| {
            let l = bare(l).trim_end();
            if toml {
                l == "+++"
            } else {
                l == "---" || l == "..."
            }
        });
        if let Some((at, line)) = close {
            end = at + line.len();
            front_matter = text[start..end].to_owned();
            let first_len = text[start..].find('\n').map_or(0, |i| i + 1);
            let body = &text[start + first_len..at];
            title = field(body, "title", toml);
            description = field(body, "description", toml);
        }
    }
    if mdx {
        // ESM paragraphs: from a line starting with `import` or `export` to the next
        // blank line, after any blank lines.
        let mut at = end;
        loop {
            let mut iter = lines(text, at).peekable();
            while iter.peek().is_some_and(|(_, l)| l.trim().is_empty()) {
                iter.next();
            }
            let Some(&(start_of, line)) = iter.peek() else {
                break;
            };
            let word = line.trim_end();
            let esm = ["import ", "import{", "export ", "export{"]
                .iter()
                .any(|p| word.starts_with(p));
            if !esm {
                break;
            }
            let mut stop = start_of;
            for (line_at, line) in iter {
                if line.trim().is_empty() {
                    break;
                }
                stop = line_at + line.len();
            }
            at = stop;
            end = stop;
        }
    }
    (end > start).then_some(Region {
        end,
        front_matter,
        title,
        description,
    })
}

/// A top-level string field of YAML (`key: value`, quoted, plain or a block scalar)
/// or TOML (`key = "value"` before the first table), whitespace collapsed.
fn field(body: &str, key: &str, toml: bool) -> Option<String> {
    let lines: Vec<&str> = body.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if toml {
            if line.trim_start().starts_with('[') {
                return None;
            }
            let Some(rest) = line.strip_prefix(key) else {
                continue;
            };
            let Some(value) = rest.trim_start().strip_prefix('=') else {
                continue;
            };
            return unquote(value.trim()).filter(|v| !v.is_empty());
        }
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let Some(value) = rest.strip_prefix(':') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() || value.starts_with('|') || value.starts_with('>') {
            // A block scalar (or a value on the following, more indented lines).
            let more: Vec<&str> = lines[i + 1..]
                .iter()
                .take_while(|l| l.trim().is_empty() || l.starts_with([' ', '\t']))
                .map(|l| l.trim())
                .collect();
            let joined = crate::headers::collapse(&more.join(" "));
            return (!joined.is_empty()).then_some(joined);
        }
        if value.starts_with(['"', '\'']) {
            return unquote(value).filter(|v| !v.is_empty());
        }
        // A plain scalar ends at a comment.
        let plain = value.split(" #").next().unwrap_or(value).trim();
        return (!plain.is_empty()).then(|| crate::headers::collapse(plain));
    }
    None
}

/// A quoted string's content (`"a \" b"`, `'it''s'`), or the text when unquoted.
fn unquote(value: &str) -> Option<String> {
    let collapse = crate::headers::collapse;
    if let Some(inner) = value.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            match c {
                '"' => return Some(collapse(&out)),
                '\\' => match chars.next()? {
                    'n' | 't' => out.push(' '),
                    other => out.push(other),
                },
                c => out.push(c),
            }
        }
        return None;
    }
    if let Some(inner) = value.strip_prefix('\'') {
        let mut out = String::new();
        let mut rest = inner;
        loop {
            let i = rest.find('\'')?;
            out.push_str(&rest[..i]);
            if rest[i + 1..].starts_with('\'') {
                out.push('\'');
                rest = &rest[i + 2..];
            } else {
                return Some(collapse(&out));
            }
        }
    }
    Some(collapse(value.split(" #").next().unwrap_or(value)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_front_matter_and_mdx_imports() {
        let text = "---\ntitle: \"SQLite storage\"\nsidebar_position: 3\ndescription: >\n  How chunks\n  are stored.\n---\n\nimport A from \"a\";\nimport {\n  B,\n} from \"b\";\n\nexport const meta = {};\n\n# Storage\n";
        let r = region(text, true).unwrap();
        assert_eq!(&text[r.end..], "\n# Storage\n");
        assert_eq!(r.title.as_deref(), Some("SQLite storage"));
        assert_eq!(r.description.as_deref(), Some("How chunks are stored."));
        assert!(r.front_matter.starts_with("---\n") && r.front_matter.ends_with("---\n"));
        // In plain Markdown the import lines are content.
        let r = region(text, false).unwrap();
        assert!(text[r.end..].starts_with("\nimport A"));
    }

    #[test]
    fn toml_and_missing_or_unclosed_front_matter() {
        let r = region(
            "+++\ntitle = \"Quoted \\\"title\\\"\" # c\n[extra]\ntitle = \"no\"\n+++\nbody\n",
            false,
        )
        .unwrap();
        assert_eq!(r.title.as_deref(), Some("Quoted \"title\""));
        let r = region("---\ntitle: 'It''s here'\n---\n", false).unwrap();
        assert_eq!(r.title.as_deref(), Some("It's here"));
        assert_eq!(region("# Title\n", false), None);
        assert_eq!(region("---\nno close\n", false), None);
        assert_eq!(region("import x from 'y'\n\n# T\n", true).unwrap().end, 18);
        let r = region("\u{feff}---\ntitle: plain # comment\n---\n", false).unwrap();
        assert_eq!(r.title.as_deref(), Some("plain"));
        assert_eq!(r.end, 3 + "---\ntitle: plain # comment\n---\n".len());
    }
}
