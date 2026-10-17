//! Splits a command line into simple commands and words for the command
//! guard. The lexers keep what matters for finding a command's targets:
//! words, variable references, output redirections, where commands and
//! pipelines begin, and where POSIX subshells open and close. Everything
//! else (arithmetic, most expansions, parameter parsing) is left to the
//! analysis or treated as unknown text.

use super::ShellDialect;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Piece {
    Text(String),
    Var(String, VarForm),
    /// A value only known when the command runs, such as `$((...))`.
    Unknown,
    /// The output of a command substitution (`$(...)` or backticks, and in
    /// PowerShell `$(...)` and `@(...)`): only known when the command runs.
    /// The text is the command it runs.
    Substitution(String),
    /// `$(pwd)`: the working directory.
    WorkingDirectory,
    /// `$(mktemp)` and `$(mktemp -d)`: a new path in the temporary directory.
    TempPath,
    /// A PowerShell parenthesized argument, kept raw (`foreach ($f in $x)`).
    Group(String),
    /// A leading `~`, or `~user`.
    Tilde(Option<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum VarForm {
    Plain,
    /// `${X:?}`: the shell stops when the value is empty.
    Required,
    /// `${X:-word}`: the word stands in for an empty value.
    Default(String),
    /// A transformation of the value (`${X%/*}`), empty when the value is.
    Transform,
    /// A special parameter such as `$1`, never known in advance.
    Opaque,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Word {
    pub pieces: Vec<Piece>,
    pub quoted: bool,
}

impl Word {
    fn push_char(&mut self, c: char) {
        match self.pieces.last_mut() {
            Some(Piece::Text(text)) => text.push(c),
            _ => self.pieces.push(Piece::Text(c.to_string())),
        }
    }

    fn push(&mut self, piece: Piece) {
        self.pieces.push(piece);
    }

    /// The word's text when it contains no expansions.
    pub fn literal(&self) -> Option<String> {
        let mut text = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Text(part) => text.push_str(part),
                _ => return None,
            }
        }
        Some(text)
    }
}

#[derive(Debug, Clone)]
pub(super) struct Redirect {
    pub append: bool,
    pub target: Word,
}

#[derive(Debug, Clone, Default)]
pub(super) struct Command {
    pub words: Vec<Word>,
    pub redirects: Vec<Redirect>,
    /// Reads the previous command's output through a pipe.
    pub piped: bool,
    /// Commands of one pipeline, including PowerShell script blocks inside
    /// it, share this number.
    pub pipeline: usize,
    /// PowerShell's call operator `&` came first: the first word names the
    /// command to run, even when it is quoted or a variable.
    pub call: bool,
    /// POSIX subshells that open before this command, outermost first.
    pub opens: Vec<Subshell>,
    /// How many POSIX subshells close after this command.
    pub closes: usize,
    /// Later POSIX pipelines its output reaches beyond its own: in a
    /// subshell, group or compound command that is a pipeline's member, the
    /// next member's.
    pub feeds: Vec<usize>,
}

/// Commands of a POSIX line that run in a subshell: a `cd` or assignment
/// there ends with it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Subshell {
    /// `( … )`, a background list (`… &`), or a pipeline's member other
    /// than the last.
    Always,
    /// A pipeline's last member: a subshell in bash, the shell itself in
    /// zsh, ksh, and bash with `lastpipe`.
    Maybe,
}

/// A POSIX list being read: the line, or a `( … )`, `{ … }` or compound
/// command (`while … done`, `if … fi`, …) in it. Indexes are of the
/// commands read.
struct List {
    /// The word that closes it.
    close: &'static str,
    /// Where its current and-or list began: `&` runs that in the background.
    start: usize,
    /// Where its current pipeline's member began.
    member: usize,
    /// Its current pipeline has a `|` already.
    piped: bool,
}

/// What ends a POSIX command.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Separator {
    /// `;` or a newline.
    List,
    /// `&&` or `||`.
    AndOr,
    /// `&`: the and-or list runs in the background.
    Background,
    /// `|` or `|&`.
    Pipe,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Pending {
    None,
    Output { append: bool },
    Discard,
    Heredoc { strip_tabs: bool },
}

struct Builder {
    commands: Vec<Command>,
    current: Command,
    word: Option<Word>,
    pending: Pending,
    pipeline: usize,
    heredocs: Vec<(String, bool)>,
    /// POSIX only: reserved words, groups and subshells are read.
    posix: bool,
    /// `( … )` subshells opened since the last command.
    opens: usize,
    /// `( … )` subshells open.
    subshells: usize,
    /// For each `case` statement open, the subshells open where it began.
    cases: Vec<usize>,
    /// The POSIX lists open, the line first.
    lists: Vec<List>,
}

impl Builder {
    fn new(posix: bool) -> Self {
        Self {
            commands: Vec::new(),
            current: Command::default(),
            word: None,
            pending: Pending::None,
            pipeline: 0,
            heredocs: Vec::new(),
            posix,
            opens: 0,
            subshells: 0,
            cases: Vec::new(),
            lists: vec![List {
                close: "",
                start: 0,
                member: 0,
                piped: false,
            }],
        }
    }

    fn word(&mut self) -> &mut Word {
        self.word.get_or_insert_with(Word::default)
    }

    fn at_command_start(&self) -> bool {
        self.word.is_none() && self.current.words.is_empty()
    }

    fn end_word(&mut self) {
        let Some(word) = self.word.take() else {
            return;
        };
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::None => {
                // A reserved word counts at the start of a command, also
                // after `do`, `then`, `else`, `elif` and `!`.
                let reserved = self.posix
                    && !word.quoted
                    && self.current.words.iter().all(|before| {
                        !before.quoted
                            && matches!(
                                before.literal().as_deref(),
                                Some("do" | "then" | "else" | "elif" | "!")
                            )
                    });
                if reserved {
                    match word.literal().as_deref() {
                        // The braces of a group aren't words of its commands.
                        Some("{") => return self.open_list("}"),
                        Some("}") if self.close_list("}") => return,
                        Some("while" | "until" | "for" | "select") => self.open_list("done"),
                        Some("if") => self.open_list("fi"),
                        Some("case") => {
                            self.cases.push(self.subshells);
                            self.open_list("esac");
                        }
                        Some("done") => {
                            self.close_list("done");
                        }
                        Some("fi") => {
                            self.close_list("fi");
                        }
                        Some("esac") => {
                            self.cases.pop();
                            self.close_list("esac");
                        }
                        _ => {}
                    }
                }
                self.current.words.push(word);
            }
            Pending::Output { append } => self.current.redirects.push(Redirect {
                append,
                target: word,
            }),
            Pending::Discard => {}
            Pending::Heredoc { strip_tabs } => {
                let delimiter = word
                    .pieces
                    .iter()
                    .map(|piece| match piece {
                        Piece::Text(text) => text.clone(),
                        _ => String::new(),
                    })
                    .collect();
                self.heredocs.push((delimiter, strip_tabs));
            }
        }
    }

    /// Ends the current command. `piped`: the next one reads its output.
    /// `new_pipeline`: the next one starts a new pipeline.
    fn end_command(&mut self, piped: bool, new_pipeline: bool) {
        self.end_word();
        self.pending = Pending::None;
        let pipeline = self.pipeline;
        if !self.current.words.is_empty() || !self.current.redirects.is_empty() {
            self.current.opens = vec![Subshell::Always; std::mem::take(&mut self.opens)];
            self.commands.push(std::mem::take(&mut self.current));
        }
        if new_pipeline {
            self.pipeline += 1;
        }
        self.current = Command {
            piped,
            pipeline: if new_pipeline {
                self.pipeline
            } else {
                pipeline
            },
            ..Command::default()
        };
    }

    /// Ends a POSIX command at `separator`, marking what runs in a subshell:
    /// a pipeline's members, and a background list.
    fn separate(&mut self, separator: Separator) {
        if separator == Separator::Pipe {
            self.end_command(true, false);
            let member = self.list().member;
            self.subshell(member, Subshell::Always);
            // The next member reads the output of every command in this one.
            let pipeline = self.current.pipeline;
            for command in &mut self.commands[member..] {
                if command.pipeline != pipeline {
                    command.feeds.push(pipeline);
                }
            }
            let next = self.commands.len();
            let list = self.list();
            list.member = next;
            list.piped = true;
            return;
        }
        self.end_command(false, true);
        self.end_pipeline();
        if separator == Separator::Background {
            let start = self.list().start;
            self.subshell(start, Subshell::Always);
        }
        if separator != Separator::AndOr {
            let next = self.commands.len();
            self.list().start = next;
        }
    }

    /// The current pipeline ends: its last member, after a `|`, may run in
    /// a subshell.
    fn end_pipeline(&mut self) {
        let list = self.list();
        let (piped, member) = (std::mem::take(&mut list.piped), list.member);
        if piped {
            self.subshell(member, Subshell::Maybe);
        }
        let next = self.commands.len();
        self.list().member = next;
    }

    /// The commands read since `start` run in a subshell of `kind`, which
    /// holds any subshell marked in them already.
    fn subshell(&mut self, start: usize, kind: Subshell) {
        if start < self.commands.len() {
            self.commands[start].opens.insert(0, kind);
            if let Some(last) = self.commands.last_mut() {
                last.closes += 1;
            }
        }
    }

    fn list(&mut self) -> &mut List {
        self.lists.last_mut().expect("the line's list stays open")
    }

    /// A list closed by the word `close` opens.
    fn open_list(&mut self, close: &'static str) {
        let start = self.commands.len();
        self.lists.push(List {
            close,
            start,
            member: start,
            piped: false,
        });
    }

    /// Closes the innermost list if `close` closes it.
    fn close_list(&mut self, close: &str) -> bool {
        if self.lists.len() < 2 || self.list().close != close {
            return false;
        }
        self.end_pipeline();
        self.lists.pop();
        true
    }

    /// A POSIX subshell opens (`(`). One that begins a pipeline's member
    /// keeps its place in the pipeline.
    fn open_subshell(&mut self) {
        if !self.at_command_start() {
            self.end_command(false, true);
        }
        self.opens += 1;
        self.subshells += 1;
        self.open_list(")");
    }

    /// A POSIX subshell closes (`)`). One that holds no command closes with
    /// nothing to mark. A `)` that closes no subshell (a `case` pattern's)
    /// only ends the command.
    fn close_subshell(&mut self) {
        self.end_command(false, true);
        if self.subshells == 0 || self.cases.last() == Some(&self.subshells) {
            let next = self.commands.len();
            let list = self.list();
            list.start = next;
            list.member = next;
            return;
        }
        self.close_list(")");
        self.subshells -= 1;
        if self.opens > 0 {
            self.opens -= 1;
        } else if let Some(command) = self.commands.last_mut() {
            command.closes += 1;
        }
    }

    fn redirect(&mut self, pending: Pending) {
        self.end_word();
        self.pending = pending;
    }

    fn finish(mut self) -> Vec<Command> {
        self.end_command(false, true);
        if self.posix {
            self.end_pipeline();
        }
        self.commands
    }
}

pub(super) fn lex(dialect: ShellDialect, source: &str) -> Vec<Command> {
    let chars: Vec<char> = source.chars().collect();
    match dialect {
        ShellDialect::Posix => posix(&chars),
        ShellDialect::Cmd => cmd(&chars),
        ShellDialect::PowerShell => powershell(&chars),
    }
}

/// The index after the `close` that balances the `open` at `start`,
/// skipping quoted text.
fn balanced(chars: &[char], start: usize, open: char, close: char) -> usize {
    let mut depth = 0usize;
    let mut i = start;
    while i < chars.len() {
        match chars[i] {
            '\\' | '`' => i += 1,
            '\'' => {
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' || chars[i] == '`' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            c if c == open => depth += 1,
            c if c == close => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    chars.len()
}

fn skip_line(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && chars[i] != '\n' {
        i += 1;
    }
    i
}

fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// ---------------------------------------------------------------- POSIX sh

fn posix(chars: &[char]) -> Vec<Command> {
    let mut b = Builder::new(true);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            ' ' | '\t' | '\r' => {
                b.end_word();
                i += 1;
            }
            '\n' => {
                b.separate(Separator::List);
                i = skip_heredocs(chars, i + 1, &mut b.heredocs);
            }
            ';' => {
                b.separate(Separator::List);
                i += 1;
            }
            '&' if next == Some('&') => {
                b.separate(Separator::AndOr);
                i += 2;
            }
            '&' if next == Some('>') => {
                let append = chars.get(i + 2) == Some(&'>');
                b.redirect(Pending::Output { append });
                i += if append { 3 } else { 2 };
            }
            '&' => {
                b.separate(Separator::Background);
                i += 1;
            }
            '|' if next == Some('|') => {
                b.separate(Separator::AndOr);
                i += 2;
            }
            '|' => {
                b.separate(Separator::Pipe);
                i += if next == Some('&') { 2 } else { 1 };
            }
            '(' => {
                b.open_subshell();
                i += 1;
            }
            ')' => {
                b.close_subshell();
                i += 1;
            }
            '#' if b.word.is_none() => i = skip_line(chars, i),
            '0'..='9' if b.word.is_none() && matches!(next, Some('>') | Some('<')) => i += 1,
            '>' => {
                match next {
                    Some('>') => {
                        b.redirect(Pending::Output { append: true });
                        i += 2;
                    }
                    Some('&') => {
                        b.redirect(Pending::Discard);
                        i += 2;
                    }
                    Some('|') => {
                        b.redirect(Pending::Output { append: false });
                        i += 2;
                    }
                    _ => {
                        b.redirect(Pending::Output { append: false });
                        i += 1;
                    }
                };
            }
            '<' => {
                if next == Some('<') && chars.get(i + 2) == Some(&'<') {
                    b.redirect(Pending::Discard);
                    i += 3;
                } else if next == Some('<') {
                    let strip_tabs = chars.get(i + 2) == Some(&'-');
                    b.redirect(Pending::Heredoc { strip_tabs });
                    i += if strip_tabs { 3 } else { 2 };
                } else {
                    b.redirect(Pending::Discard);
                    i += if matches!(next, Some('&') | Some('>')) {
                        2
                    } else {
                        1
                    };
                }
            }
            '\'' => {
                let word = b.word();
                word.quoted = true;
                word.push(Piece::Text(String::new()));
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    word.push_char(chars[i]);
                    i += 1;
                }
                i += 1;
            }
            '"' => {
                let word = b.word();
                word.quoted = true;
                word.push(Piece::Text(String::new()));
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    match chars[i] {
                        '\\' if matches!(chars.get(i + 1), Some('$' | '`' | '"' | '\\' | '\n')) => {
                            if chars[i + 1] != '\n' {
                                word.push_char(chars[i + 1]);
                            }
                            i += 2;
                        }
                        '$' => i = posix_dollar(chars, i, word),
                        '`' => {
                            let end = backtick_end(chars, i);
                            word.push(substitution(
                                &chars[i + 1..end.saturating_sub(1).max(i + 1)],
                            ));
                            i = end;
                        }
                        other => {
                            word.push_char(other);
                            i += 1;
                        }
                    }
                }
                i += 1;
            }
            '\\' => {
                match next {
                    Some('\n') => {}
                    Some(escaped) => b.word().push_char(escaped),
                    None => {}
                }
                i += 2;
            }
            '$' => i = posix_dollar(chars, i, b.word()),
            '`' => {
                let end = backtick_end(chars, i);
                b.word().push(substitution(
                    &chars[i + 1..end.saturating_sub(1).max(i + 1)],
                ));
                i = end;
            }
            '~' if b.word.is_none() => {
                let mut end = i + 1;
                while end < chars.len()
                    && (is_name_char(chars[end]) || matches!(chars[end], '.' | '-'))
                {
                    end += 1;
                }
                let terminates = chars.get(end).is_none_or(|c| {
                    matches!(
                        c,
                        '/' | ' ' | '\t' | '\n' | ';' | '&' | '|' | ')' | '<' | '>'
                    )
                });
                if terminates {
                    let name: String = chars[i + 1..end].iter().collect();
                    b.word()
                        .push(Piece::Tilde((!name.is_empty()).then_some(name)));
                    i = end;
                } else {
                    b.word().push_char('~');
                    i += 1;
                }
            }
            other => {
                b.word().push_char(other);
                i += 1;
            }
        }
    }
    b.finish()
}

/// The value of a command substitution: the working directory for `pwd`, a
/// new temporary path for a `mktemp` that names no directory of its own, and
/// the command's output otherwise.
fn substitution(inner: &[char]) -> Piece {
    let text: String = inner.iter().collect();
    let words: Vec<&str> = text.split_whitespace().collect();
    match words.as_slice() {
        ["pwd"] | ["pwd", "-L" | "-P"] => Piece::WorkingDirectory,
        ["mktemp", rest @ ..] => {
            let options_only = rest.iter().all(|word| word.starts_with('-'));
            let in_temp = rest.iter().any(|word| *word == "-t" || *word == "--tmpdir");
            if options_only || in_temp {
                Piece::TempPath
            } else {
                Piece::Unknown
            }
        }
        _ => Piece::Substitution(text),
    }
}

fn backtick_end(chars: &[char], start: usize) -> usize {
    let mut i = start + 1;
    while i < chars.len() && chars[i] != '`' {
        if chars[i] == '\\' {
            i += 1;
        }
        i += 1;
    }
    i + 1
}

fn skip_heredocs(chars: &[char], mut i: usize, heredocs: &mut Vec<(String, bool)>) -> usize {
    for (delimiter, strip_tabs) in heredocs.drain(..) {
        while i < chars.len() {
            let end = skip_line(chars, i);
            let line: String = chars[i..end].iter().collect();
            let line = if strip_tabs {
                line.trim_start_matches('\t')
            } else {
                &line
            };
            let matched = line.trim_end_matches('\r') == delimiter;
            i = (end + 1).min(chars.len());
            if matched {
                break;
            }
        }
    }
    i
}

fn posix_dollar(chars: &[char], i: usize, word: &mut Word) -> usize {
    match chars.get(i + 1) {
        Some('(') => {
            let end = balanced(chars, i + 1, '(', ')');
            let inner = &chars[i + 2..end.saturating_sub(1).max(i + 2)];
            // `$((...))` is arithmetic, not a command.
            word.push(if inner.first() == Some(&'(') {
                Piece::Unknown
            } else {
                substitution(inner)
            });
            end
        }
        Some('{') => {
            let end = balanced(chars, i + 1, '{', '}');
            let inner: String = chars[i + 2..end.saturating_sub(1).max(i + 2)]
                .iter()
                .collect();
            word.push(brace_parameter(&inner));
            end
        }
        Some('\'') => {
            word.quoted = true;
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '\'' {
                if chars[j] == '\\' && j + 1 < chars.len() {
                    j += 1;
                    word.push_char(match chars[j] {
                        'n' => '\n',
                        't' => '\t',
                        other => other,
                    });
                } else {
                    word.push_char(chars[j]);
                }
                j += 1;
            }
            j + 1
        }
        Some(&c) if is_name_start(c) => {
            let mut end = i + 1;
            while end < chars.len() && is_name_char(chars[end]) {
                end += 1;
            }
            word.push(Piece::Var(
                chars[i + 1..end].iter().collect(),
                VarForm::Plain,
            ));
            end
        }
        Some(&c) if c.is_ascii_digit() || "@*#?$!-".contains(c) => {
            word.push(Piece::Var(c.to_string(), VarForm::Opaque));
            i + 2
        }
        _ => {
            word.push_char('$');
            i + 1
        }
    }
}

fn brace_parameter(inner: &str) -> Piece {
    if inner.starts_with('#') || inner.starts_with('!') {
        return Piece::Var(inner.to_string(), VarForm::Opaque);
    }
    let name_end = inner
        .find(|c: char| !is_name_char(c))
        .unwrap_or(inner.len());
    let (name, rest) = inner.split_at(name_end);
    if name.is_empty() || name.chars().all(|c| c.is_ascii_digit()) {
        return Piece::Var(name.to_string(), VarForm::Opaque);
    }
    let rest_after_colon = rest.strip_prefix(':').unwrap_or(rest);
    let form = if rest.is_empty() {
        VarForm::Plain
    } else if rest_after_colon.starts_with('?') {
        VarForm::Required
    } else if let Some(default) = rest_after_colon
        .strip_prefix('-')
        .or_else(|| rest_after_colon.strip_prefix('='))
    {
        VarForm::Default(default.trim_matches(|c| c == '"' || c == '\'').to_string())
    } else if rest_after_colon.starts_with('+') {
        VarForm::Opaque
    } else {
        VarForm::Transform
    };
    Piece::Var(name.to_string(), form)
}

// --------------------------------------------------------------------- cmd

fn cmd(chars: &[char]) -> Vec<Command> {
    let mut b = Builder::new(false);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            ' ' | '\t' | ',' | ';' => {
                b.end_word();
                i += 1;
            }
            '\r' | '\n' => {
                b.end_command(false, true);
                i += 1;
            }
            '&' => {
                b.end_command(false, true);
                i += if next == Some('&') { 2 } else { 1 };
            }
            '|' if next == Some('|') => {
                b.end_command(false, true);
                i += 2;
            }
            '|' => {
                b.end_command(true, false);
                i += 1;
            }
            '(' | ')' => {
                b.end_command(false, true);
                i += 1;
            }
            '@' if b.at_command_start() => i += 1,
            '0'..='9' if b.word.is_none() && matches!(next, Some('>') | Some('<')) => i += 1,
            '>' => {
                if next == Some('&') {
                    b.redirect(Pending::Discard);
                    i += 2;
                } else {
                    let append = next == Some('>');
                    b.redirect(Pending::Output { append });
                    i += if append { 2 } else { 1 };
                }
            }
            '<' => {
                b.redirect(Pending::Discard);
                i += 1;
            }
            '^' => {
                if let Some(escaped) = next {
                    b.word().push_char(escaped);
                }
                i += 2;
            }
            '"' => {
                let word = b.word();
                word.quoted = true;
                word.push(Piece::Text(String::new()));
                i += 1;
                while i < chars.len() && chars[i] != '"' && chars[i] != '\n' {
                    if chars[i] == '%' {
                        i = cmd_percent(chars, i, word);
                    } else {
                        word.push_char(chars[i]);
                        i += 1;
                    }
                }
                i += 1;
            }
            '%' => i = cmd_percent(chars, i, b.word()),
            '!' => {
                let close = chars[i + 1..].iter().position(|&c| c == '!');
                match close {
                    Some(len) if len > 0 && chars[i + 1..i + 1 + len].iter().all(|&c| c != ' ') => {
                        b.word().push(Piece::Unknown);
                        i += len + 2;
                    }
                    _ => {
                        b.word().push_char('!');
                        i += 1;
                    }
                }
            }
            other => {
                b.word().push_char(other);
                i += 1;
            }
        }
    }
    b.finish()
}

fn cmd_percent(chars: &[char], i: usize, word: &mut Word) -> usize {
    match chars.get(i + 1) {
        // `%%i` loop variables, `%1` and `%*` arguments, `%~dp0` modifiers.
        Some('%') => {
            word.push(Piece::Var("%".into(), VarForm::Opaque));
            (i + 3).min(chars.len())
        }
        Some(c) if c.is_ascii_digit() || *c == '*' => {
            word.push(Piece::Var(c.to_string(), VarForm::Opaque));
            i + 2
        }
        Some('~') => {
            let mut end = i + 2;
            while end < chars.len() && chars[end].is_ascii_alphabetic() {
                end += 1;
            }
            word.push(Piece::Var("~".into(), VarForm::Opaque));
            (end + 1).min(chars.len())
        }
        _ => {
            let close = chars[i + 1..]
                .iter()
                .position(|&c| matches!(c, '%' | '"' | '\n' | ' '));
            match close {
                Some(len) if len > 0 && chars[i + 1 + len] == '%' => {
                    let name: String = chars[i + 1..i + 1 + len].iter().collect();
                    let piece = match name.split_once(':') {
                        Some((name, _)) => Piece::Var(name.to_string(), VarForm::Transform),
                        None => Piece::Var(name, VarForm::Plain),
                    };
                    word.push(piece);
                    i + len + 2
                }
                // A single `%a` inside a `for` loop typed at the prompt.
                _ if chars.get(i + 1).is_some_and(|c| c.is_ascii_alphabetic())
                    && !chars.get(i + 2).is_some_and(|c| c.is_ascii_alphanumeric()) =>
                {
                    word.push(Piece::Var("%".into(), VarForm::Opaque));
                    i + 2
                }
                _ => {
                    word.push_char('%');
                    i + 1
                }
            }
        }
    }
}

// -------------------------------------------------------------- PowerShell

fn powershell(chars: &[char]) -> Vec<Command> {
    let mut b = Builder::new(false);
    let mut depth = 0usize;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            ' ' | '\t' | ',' => {
                b.end_word();
                i += 1;
            }
            '\r' | '\n' | ';' => {
                b.end_command(false, depth == 0);
                i += 1;
            }
            '{' => {
                b.end_command(false, false);
                depth += 1;
                i += 1;
            }
            '}' => {
                b.end_command(false, false);
                depth = depth.saturating_sub(1);
                i += 1;
            }
            '&' if next == Some('&') => {
                b.end_command(false, depth == 0);
                i += 2;
            }
            '&' if b.at_command_start() => {
                b.current.call = true;
                i += 1;
            }
            '&' => {
                b.end_command(false, depth == 0);
                i += 1;
            }
            '|' if next == Some('|') => {
                b.end_command(false, depth == 0);
                i += 2;
            }
            '|' => {
                b.end_command(true, false);
                i += 1;
            }
            '(' if b.at_command_start() => {
                b.end_command(false, false);
                i += 1;
            }
            '(' => {
                let end = balanced(chars, i, '(', ')');
                let raw: String = chars[i + 1..end.saturating_sub(1).max(i + 1)]
                    .iter()
                    .collect();
                b.word().push(Piece::Group(raw));
                i = end;
            }
            ')' => {
                b.end_command(false, false);
                i += 1;
            }
            '<' if next == Some('#') => {
                let mut j = i + 2;
                while j + 1 < chars.len() && !(chars[j] == '#' && chars[j + 1] == '>') {
                    j += 1;
                }
                i = j + 2;
            }
            '#' if b.word.is_none() => i = skip_line(chars, i),
            '0'..='9' | '*' if b.word.is_none() && next == Some('>') => i += 1,
            '>' => {
                if next == Some('&') {
                    b.redirect(Pending::Discard);
                    i += 3;
                } else {
                    let append = next == Some('>');
                    b.redirect(Pending::Output { append });
                    i += if append { 2 } else { 1 };
                }
            }
            '@' if matches!(next, Some('"') | Some('\''))
                && chars.get(i + 2).is_some_and(|c| *c == '\n' || *c == '\r') =>
            {
                let quote = chars[i + 1];
                let mut j = i + 2;
                let mut text = String::new();
                while j < chars.len() {
                    if chars[j] == '\n'
                        && chars.get(j + 1) == Some(&quote)
                        && chars.get(j + 2) == Some(&'@')
                    {
                        j += 3;
                        break;
                    }
                    text.push(chars[j]);
                    j += 1;
                }
                let word = b.word();
                word.quoted = true;
                word.push(Piece::Text(text));
                i = j;
            }
            '@' if next == Some('(') => {
                let end = balanced(chars, i + 1, '(', ')');
                b.word().push(Piece::Substitution(
                    chars[i + 2..end.saturating_sub(1).max(i + 2)]
                        .iter()
                        .collect(),
                ));
                i = end;
            }
            '\'' => {
                let word = b.word();
                word.quoted = true;
                word.push(Piece::Text(String::new()));
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\'' {
                        if chars.get(i + 1) == Some(&'\'') {
                            word.push_char('\'');
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    word.push_char(chars[i]);
                    i += 1;
                }
                i += 1;
            }
            '"' => {
                let word = b.word();
                word.quoted = true;
                word.push(Piece::Text(String::new()));
                i += 1;
                while i < chars.len() {
                    match chars[i] {
                        '"' if chars.get(i + 1) == Some(&'"') => {
                            word.push_char('"');
                            i += 2;
                        }
                        '"' => break,
                        '`' => {
                            if let Some(&escaped) = chars.get(i + 1) {
                                word.push_char(escaped);
                            }
                            i += 2;
                        }
                        '$' => i = powershell_dollar(chars, i, word),
                        other => {
                            word.push_char(other);
                            i += 1;
                        }
                    }
                }
                i += 1;
            }
            '`' => {
                match next {
                    Some('\n') | None => {}
                    Some(escaped) => b.word().push_char(escaped),
                }
                i += 2;
            }
            '$' => i = powershell_dollar(chars, i, b.word()),
            '~' if b.word.is_none()
                && chars.get(i + 1).is_none_or(|c| {
                    matches!(c, '/' | '\\' | ' ' | '\t' | ';' | '|' | ')' | '\n')
                }) =>
            {
                b.word().push(Piece::Tilde(None));
                i += 1;
            }
            other => {
                b.word().push_char(other);
                i += 1;
            }
        }
    }
    b.finish()
}

fn powershell_dollar(chars: &[char], i: usize, word: &mut Word) -> usize {
    match chars.get(i + 1) {
        Some('(') => {
            let end = balanced(chars, i + 1, '(', ')');
            word.push(Piece::Substitution(
                chars[i + 2..end.saturating_sub(1).max(i + 2)]
                    .iter()
                    .collect(),
            ));
            end
        }
        Some('{') => {
            let end = chars[i + 2..]
                .iter()
                .position(|&c| c == '}')
                .map_or(chars.len(), |len| i + 2 + len);
            let name: String = chars[i + 2..end].iter().collect();
            word.push(Piece::Var(scoped_name(&name), VarForm::Plain));
            (end + 1).min(chars.len())
        }
        Some(&c) if is_name_char(c) => {
            let mut end = i + 1;
            while end < chars.len()
                && (is_name_char(chars[end])
                    || (chars[end] == ':' && chars.get(end + 1).is_some_and(|c| is_name_char(*c))))
            {
                end += 1;
            }
            let name: String = chars[i + 1..end].iter().collect();
            word.push(Piece::Var(scoped_name(&name), VarForm::Plain));
            end
        }
        Some('?' | '^' | '$') => {
            word.push(Piece::Var("?".into(), VarForm::Opaque));
            i + 2
        }
        _ => {
            word.push_char('$');
            i + 1
        }
    }
}

/// `env:X` keeps its scope. Other scopes (`global:`, `script:`) are dropped.
/// Names are case-insensitive.
fn scoped_name(name: &str) -> String {
    let name = name.to_ascii_lowercase();
    match name.split_once(':') {
        Some(("env", rest)) => format!("env:{rest}"),
        Some((_, rest)) => rest.to_string(),
        None => name,
    }
}
