//! Lexical paths for the command guard. They are parsed from text rather
//! than with `std::path`, so Windows and Unix rules can be checked (and
//! tested) on any host, and nothing on disk is touched.

/// Placeholder text for a value known only when the command runs, which may
/// be empty.
pub(super) const MAYBE_EMPTY: char = '\u{1}';
/// Placeholder text for a value known only when the command runs, which the
/// shell guarantees is not empty (`${X:?}`).
pub(super) const NONEMPTY: char = '\u{2}';
/// A child name that matches no real directory.
pub(super) const ANY_CHILD: &str = "\u{0}";

#[derive(Debug, Clone, Copy)]
pub(super) struct PathStyle {
    pub windows: bool,
    pub ignore_case: bool,
}

impl PathStyle {
    pub fn host() -> Self {
        Self {
            windows: cfg!(windows),
            ignore_case: cfg!(any(windows, target_os = "macos")),
        }
    }

    pub fn is_separator(self, c: char) -> bool {
        c == '/' || (self.windows && c == '\\')
    }

    fn separator(self) -> char {
        if self.windows { '\\' } else { '/' }
    }

    pub fn same(self, a: &str, b: &str) -> bool {
        if self.ignore_case {
            a.to_lowercase() == b.to_lowercase()
        } else {
            a == b
        }
    }
}

/// An absolute, normalized path: a root (`/`, `C:\`, `\\server\share\`, or
/// the device namespace `\\.\`) and the names below it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AbsPath {
    pub root: String,
    pub parts: Vec<String>,
}

impl AbsPath {
    /// Parses `text`, relative to `cwd` when it is relative. `None` when it
    /// is relative and `cwd` is unknown.
    pub fn parse(style: PathStyle, text: &str, cwd: Option<&AbsPath>) -> Option<Self> {
        let (root, rest) = split_root(style, text);
        let (root, mut parts) = match root {
            Root::Absolute(root) => (root, Vec::new()),
            Root::CurrentDrive => {
                let cwd = cwd?;
                (cwd.root.clone(), Vec::new())
            }
            Root::DriveRelative(drive) => match cwd {
                Some(cwd) if style.same(&cwd.root[..2.min(cwd.root.len())], &drive) => {
                    (cwd.root.clone(), cwd.parts.clone())
                }
                _ => (format!("{drive}\\"), Vec::new()),
            },
            Root::Relative => {
                let cwd = cwd?;
                (cwd.root.clone(), cwd.parts.clone())
            }
        };
        for part in rest.split(|c| style.is_separator(c)) {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                name => parts.push(name.to_string()),
            }
        }
        Some(Self { root, parts })
    }

    pub fn display(&self, style: PathStyle) -> String {
        let text = format!(
            "{}{}",
            self.root,
            self.parts.join(&style.separator().to_string())
        );
        text.replace([MAYBE_EMPTY, NONEMPTY], "…")
            .replace(ANY_CHILD, "*")
    }

    pub fn child(&self) -> Self {
        let mut parts = self.parts.clone();
        parts.push(ANY_CHILD.to_string());
        Self {
            root: self.root.clone(),
            parts,
        }
    }

    pub fn parent(&self) -> Option<Self> {
        let mut parts = self.parts.clone();
        parts.pop()?;
        Some(Self {
            root: self.root.clone(),
            parts,
        })
    }

    /// Whether `self` is `other` or contains it.
    pub fn contains(&self, style: PathStyle, other: &AbsPath) -> bool {
        style.same(&self.root, &other.root)
            && self.parts.len() <= other.parts.len()
            && self
                .parts
                .iter()
                .zip(&other.parts)
                .all(|(a, b)| style.same(a, b))
    }

    pub fn equals(&self, style: PathStyle, other: &AbsPath) -> bool {
        self.parts.len() == other.parts.len() && self.contains(style, other)
    }

    /// The first part that holds a wildcard or a placeholder, with the parts
    /// before it.
    pub fn first_pattern(&self, windows: bool) -> Option<(AbsPath, &str)> {
        let index = self.parts.iter().position(|part| {
            part.chars().any(|c| {
                c == '*' || c == '?' || c == MAYBE_EMPTY || c == NONEMPTY || (!windows && c == '[')
            })
        })?;
        Some((
            AbsPath {
                root: self.root.clone(),
                parts: self.parts[..index].to_vec(),
            },
            &self.parts[index],
        ))
    }
}

enum Root {
    Absolute(String),
    /// `\foo` on Windows: the root of the current drive.
    CurrentDrive,
    /// `C:foo`.
    DriveRelative(String),
    Relative,
}

fn split_root(style: PathStyle, text: &str) -> (Root, &str) {
    if !style.windows {
        return match text.strip_prefix('/') {
            Some(rest) => (Root::Absolute("/".into()), rest),
            None => (Root::Relative, text),
        };
    }
    let text = match text
        .strip_prefix(r"\\?\")
        .or_else(|| text.strip_prefix("//?/"))
    {
        Some(rest) => match rest.get(..4) {
            Some(prefix) if prefix.eq_ignore_ascii_case(r"UNC\") => {
                return split_unc(style, &rest[4..]);
            }
            _ => rest,
        },
        None => text,
    };
    let mut chars = text.chars();
    let first = chars.next();
    let second = chars.next();
    if first.is_some_and(|c| style.is_separator(c)) && second.is_some_and(|c| style.is_separator(c))
    {
        let rest = &text[2..];
        if let Some(device) = rest.strip_prefix(".\\").or_else(|| rest.strip_prefix("./")) {
            return (Root::Absolute(r"\\.\".into()), device);
        }
        return split_unc(style, rest);
    }
    if first.is_some_and(|c| style.is_separator(c)) {
        return (Root::CurrentDrive, &text[1..]);
    }
    if first.is_some_and(|c| c.is_ascii_alphabetic()) && second == Some(':') {
        let drive = text[..2].to_ascii_uppercase();
        let rest = &text[2..];
        return match rest.chars().next() {
            Some(c) if style.is_separator(c) => (Root::Absolute(format!("{drive}\\")), &rest[1..]),
            _ => (Root::DriveRelative(drive), rest),
        };
    }
    (Root::Relative, text)
}

/// `server\share\rest` after the leading `\\`.
fn split_unc(style: PathStyle, text: &str) -> (Root, &str) {
    let mut pieces = text.splitn(3, |c| style.is_separator(c));
    let server = pieces.next().unwrap_or_default();
    let share = pieces.next().unwrap_or_default();
    let rest = pieces.next().unwrap_or_default();
    (Root::Absolute(format!(r"\\{server}\{share}\")), rest)
}
