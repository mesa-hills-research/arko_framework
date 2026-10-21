//! Refuses terminal commands that would accidentally delete, move, overwrite,
//! or reformat the filesystem root, a drive root, the home
//! directory or its parents, top-level and system directories, mounted
//! volumes, or devices. Paths outside the working directory need opt-in.
//!
//! This is a guard against accidents, not a sandbox. It reads the command
//! text, including the shell's own words, variables it can resolve, `cd`, redirections,
//! pipelines, and nested `sh -c`, `cmd /c`, and `powershell -Command`
//! strings. A script or program the command runs is not inspected.
//!
//! [`CommandRule`]s, off by default, allow or refuse programs and their
//! arguments, matched against each command the guard finds.

mod lex;
mod paths;
mod rules;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use lex::{Command, Piece, Subshell, VarForm, Word};
use paths::{AbsPath, MAYBE_EMPTY, NONEMPTY, PathStyle};
use rules::Rules;
pub use rules::{CommandRule, RuleAction};

/// The command language a shell speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellDialect {
    /// `sh`, `bash`, `zsh`, and other POSIX shells.
    Posix,
    /// Windows `cmd.exe`.
    Cmd,
    /// Windows PowerShell and PowerShell 7 (`pwsh`).
    PowerShell,
}

impl ShellDialect {
    /// The dialect of a shell program. `cmd` and `cmd.exe` are
    /// [`Cmd`](Self::Cmd), `powershell` and `pwsh` are
    /// [`PowerShell`](Self::PowerShell), anything else is
    /// [`Posix`](Self::Posix).
    pub fn from_shell(program: &Path) -> Self {
        match base_name(&program.to_string_lossy()).as_str() {
            "cmd" => Self::Cmd,
            "powershell" | "pwsh" => Self::PowerShell,
            _ => Self::Posix,
        }
    }
}

/// Which destructive commands the terminal runs. The default refuses any
/// command that would delete, move, overwrite, or reformat a protected path
/// or a path outside the working directory.
#[derive(Debug, Clone, Default)]
pub struct CommandGuard {
    /// Allow destructive commands on paths outside the working directory.
    /// Protected paths stay refused.
    pub allow_outside_workspace: bool,
    /// Paths outside the working directory that destructive commands may
    /// target, with everything below them. Relative paths are relative to
    /// the job's working directory. Protected paths can't be allowed here,
    /// except a top-level directory named exactly.
    pub allowed_paths: Vec<PathBuf>,
    /// Turn the guard off. Every command runs as written, including ones
    /// that delete the filesystem root or the home directory, and the
    /// `rules` don't apply.
    pub dangerously_allow_all_commands: bool,
    /// Programs and arguments to allow or refuse, matched against every
    /// command the guard finds in the line, including each pipeline and list item,
    /// commands after `cd` and wrappers (`sudo`, `env`, `timeout`, …), in
    /// `sh -c`, `eval`, `$(…)` and backticks, and run by `xargs` and
    /// `find -exec`. Empty by default, with no rules.
    ///
    /// A command is refused when a [`Deny`](RuleAction::Deny) rule matches
    /// it, or when its program has [`Allow`](RuleAction::Allow) rules and
    /// none matches it. Otherwise only the path checks apply. While there
    /// are rules, a command whose program, or nested command text, is only
    /// known when it runs is refused too, since the rules can't check it.
    pub rules: Vec<CommandRule>,
}

/// Why the guard refused a command. The message tells the model what the
/// command would have done and what to do instead.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct CommandRefusal {
    pub message: String,
    /// The path the command would have affected, as the guard resolved it.
    /// For a refusal by the [`rules`](CommandGuard::rules), the command.
    pub target: String,
    /// The path is protected, so only `dangerously_allow_all_commands` allows it.
    pub protected: bool,
}

impl CommandGuard {
    /// Checks `command`, run by a `dialect` shell in `working_directory`,
    /// against the guard. Variables resolve from this process's environment.
    pub fn check(
        &self,
        command: &str,
        dialect: ShellDialect,
        working_directory: &Path,
    ) -> Result<(), CommandRefusal> {
        if self.dangerously_allow_all_commands {
            return Ok(());
        }
        self.check_on(
            &Host::from_environment(self, working_directory),
            dialect,
            command,
        )
    }

    fn check_on(
        &self,
        host: &Host,
        dialect: ShellDialect,
        command: &str,
    ) -> Result<(), CommandRefusal> {
        if self.dangerously_allow_all_commands {
            return Ok(());
        }
        let rules = Rules::new(&self.rules, host.style, host.workspace.as_ref());
        let mut analysis = Analysis {
            host,
            rules: &rules,
            findings: Vec::new(),
            refusal: None,
            pipeline: 0,
        };
        let mut scope = Scope::new(host.workspace.clone());
        analysis.run(dialect, command, &mut scope, 0);
        let mut outside = None;
        for finding in analysis.findings {
            match finding.class {
                Class::Protected(_) => return Err(finding.refusal(host)),
                Class::Outside if !self.allow_outside_workspace && outside.is_none() => {
                    outside = Some(finding);
                }
                _ => {}
            }
        }
        if let Some(refusal) = analysis.refusal {
            return Err(refusal);
        }
        outside.map_or(Ok(()), |finding| Err(finding.refusal(host)))
    }
}

// ------------------------------------------------------------------ host

/// What the guard knows about the machine: path rules, the working
/// directory, the home and temporary directories, and the environment.
struct Host {
    style: PathStyle,
    workspace: Option<AbsPath>,
    /// False when the working directory is itself protected (a home
    /// directory or a root), so being inside it allows nothing.
    workspace_usable: bool,
    home: Option<AbsPath>,
    temp: Vec<AbsPath>,
    allowed: Vec<AbsPath>,
    env: HashMap<String, String>,
}

impl Host {
    fn new(
        style: PathStyle,
        workspace: Option<AbsPath>,
        home: Option<AbsPath>,
        temp: Vec<AbsPath>,
        allowed: Vec<AbsPath>,
        env: HashMap<String, String>,
    ) -> Self {
        let mut host = Self {
            style,
            workspace,
            workspace_usable: false,
            home,
            temp,
            allowed,
            env,
        };
        // A workspace that is itself a protected place (a root, the home
        // directory, a system directory) grants nothing. One right under the
        // root, such as `D:\RoboEnv` or `/data`, is fine. The rule that keeps
        // top-level directories from being deleted guards the directory
        // itself, not what's inside it, so it doesn't count here. Speculative
        // paths skip it the same way.
        host.workspace_usable = host.workspace.as_ref().is_some_and(|workspace| {
            !matches!(
                host.classify_path(workspace, Op::Delete, true),
                Class::Protected(_)
            )
        });
        host
    }

    fn from_environment(guard: &CommandGuard, working_directory: &Path) -> Self {
        let style = PathStyle::host();
        let env: HashMap<String, String> = std::env::vars_os()
            .map(|(name, value)| {
                (
                    name.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                )
            })
            .collect();
        let absolute = |path: &Path| -> Option<AbsPath> {
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir().ok()?.join(path)
            };
            AbsPath::parse(style, &path.to_string_lossy(), None)
        };
        let home_variable = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        let home = env
            .get(home_variable)
            .or_else(|| env.get("HOME"))
            .and_then(|home| absolute(Path::new(home)));
        let mut temp_roots = vec![std::env::temp_dir()];
        for name in ["TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = env.get(name) {
                temp_roots.push(value.into());
            }
        }
        if !cfg!(windows) {
            for root in ["/tmp", "/var/tmp", "/private/tmp", "/dev/shm"] {
                temp_roots.push(root.into());
            }
        }
        let mut temp = Vec::new();
        for root in temp_roots {
            let canonical = std::fs::canonicalize(&root).ok();
            temp.extend(
                [Some(root), canonical]
                    .into_iter()
                    .flatten()
                    .filter_map(|p| absolute(&p)),
            );
        }
        let allowed = guard
            .allowed_paths
            .iter()
            .filter_map(|path| absolute(&working_directory.join(path)))
            .collect();
        Self::new(style, absolute(working_directory), home, temp, allowed, env)
    }

    fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned().or_else(|| {
            self.style.windows.then(|| {
                self.env
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case(name))
                    .map(|(_, value)| value.clone())
            })?
        })
    }

    fn home_text(&self) -> Option<String> {
        self.home.as_ref().map(|home| home.display(self.style))
    }

    /// Where a resolved target is. A value only known when the command runs
    /// at the start of the path can be anywhere. When it may be empty, the
    /// rest of the path is also checked from the root.
    fn locate(&self, text: &str, cwd: Option<&AbsPath>) -> Vec<Located> {
        if text.is_empty() {
            return Vec::new();
        }
        let rest = text.trim_start_matches([MAYBE_EMPTY, NONEMPTY]);
        if rest.len() != text.len() {
            let mut located = vec![Located::unknown()];
            let may_be_empty = !text[..text.len() - rest.len()].contains(NONEMPTY);
            if may_be_empty && rest.starts_with(|c| self.style.is_separator(c)) {
                located.extend(self.locate(rest, cwd).into_iter().map(|mut place| {
                    place.speculative = true;
                    place
                }));
            }
            return located;
        }
        let Some(path) = AbsPath::parse(self.style, text, cwd) else {
            return vec![Located::unknown()];
        };
        match path.first_pattern(self.style.windows) {
            None => vec![Located {
                path: Some(path),
                span: Span::Exact,
                speculative: false,
            }],
            Some((prefix, part)) => {
                let everything = part.chars().all(|c| c == '*' || c == MAYBE_EMPTY)
                    || (self.style.windows && part == "*.*");
                vec![Located {
                    path: Some(prefix),
                    span: if everything {
                        Span::Contents
                    } else {
                        Span::Child
                    },
                    speculative: false,
                }]
            }
        }
    }

    fn judge(&self, place: &Located, op: Op) -> Class {
        let Some(path) = &place.path else {
            return match op {
                Op::Append | Op::Permissions { recursive: false } => Class::Allowed,
                _ => Class::Outside,
            };
        };
        match place.span {
            Span::Exact | Span::Contents => self.classify(path, op, place.speculative),
            Span::Child => {
                let holds_home = self.home.as_ref().is_some_and(|home| {
                    path.contains(self.style, home) && !path.equals(self.style, home)
                });
                if holds_home {
                    return Class::Protected("a parent of the home directory");
                }
                self.classify(&path.child(), op, place.speculative)
            }
        }
    }

    fn classify(&self, path: &AbsPath, op: Op, speculative: bool) -> Class {
        match (op, self.classify_path(path, op, speculative)) {
            (Op::Append | Op::Permissions { recursive: false }, Class::Outside) => Class::Allowed,
            (_, class) => class,
        }
    }

    fn classify_path(&self, path: &AbsPath, op: Op, speculative: bool) -> Class {
        let style = self.style;
        let writes = matches!(op, Op::Overwrite | Op::Append);
        if style.windows {
            let is_nul = path
                .parts
                .last()
                .is_some_and(|name| name.eq_ignore_ascii_case("nul"));
            if writes && is_nul {
                return Class::Allowed;
            }
            if path.root == r"\\.\" {
                return Class::Protected("a device");
            }
        } else if path.parts.first().is_some_and(|name| name == "dev") {
            let shared_memory = path.parts.len() > 2 && path.parts[1] == "shm";
            let harmless = path.parts.len() == 2
                && matches!(
                    path.parts[1].as_str(),
                    "null"
                        | "zero"
                        | "random"
                        | "urandom"
                        | "full"
                        | "stdin"
                        | "stdout"
                        | "stderr"
                        | "tty"
                )
                || path.parts.get(1).is_some_and(|name| name == "fd");
            if shared_memory || (writes && harmless) {
                return Class::Allowed;
            }
            return Class::Protected("a device");
        }
        if path.parts.is_empty() {
            return Class::Protected(if !style.windows {
                "the filesystem root"
            } else if path.root.starts_with(r"\\") {
                "the root of a network share"
            } else {
                "the root of a drive"
            });
        }
        if let Some(home) = &self.home {
            if path.contains(style, home) {
                return Class::Protected(if path.equals(style, home) {
                    "the home directory"
                } else {
                    "a parent of the home directory"
                });
            }
            let sibling = path.parts.len() == home.parts.len()
                && home
                    .parent()
                    .is_some_and(|parent| parent.contains(style, path));
            if sibling {
                return Class::Protected("another user's home directory");
            }
        }
        if !style.windows {
            let mount = (path.parts.len() == 2
                && matches!(path.parts[0].as_str(), "Volumes" | "mnt" | "media"))
                || (path.parts.len() == 4 && path.parts[0] == "run" && path.parts[1] == "media");
            if mount {
                return Class::Protected("a mounted volume");
            }
        }
        if self.granted(path) {
            return Class::Allowed;
        }
        if path.parts.len() == 1 && !speculative {
            return Class::Protected("a top-level directory");
        }
        let top = path.parts[0].to_lowercase();
        let system = if style.windows {
            matches!(
                top.as_str(),
                "windows" | "program files" | "program files (x86)" | "programdata"
            )
        } else {
            matches!(
                top.as_str(),
                "bin"
                    | "boot"
                    | "dev"
                    | "etc"
                    | "lib"
                    | "lib32"
                    | "lib64"
                    | "libx32"
                    | "proc"
                    | "sbin"
                    | "sys"
                    | "usr"
                    | "var"
                    | "system"
                    | "library"
                    | "private"
                    | "snap"
            )
        };
        if system {
            return Class::Protected("a system directory");
        }
        Class::Outside
    }

    fn granted(&self, path: &AbsPath) -> bool {
        let style = self.style;
        (self.workspace_usable
            && self
                .workspace
                .as_ref()
                .is_some_and(|workspace| workspace.contains(style, path)))
            || self
                .temp
                .iter()
                .any(|temp| temp.contains(style, path) && !temp.equals(style, path))
            || self
                .allowed
                .iter()
                .any(|allowed| allowed.contains(style, path))
    }
}

#[derive(Debug, Clone)]
struct Located {
    /// `None`: the path is only known when the command runs.
    path: Option<AbsPath>,
    span: Span,
    /// The path assumes an unknown value is empty.
    speculative: bool,
}

impl Located {
    fn unknown() -> Self {
        Self {
            path: None,
            span: Span::Exact,
            speculative: false,
        }
    }

    fn display(&self, style: PathStyle, text: &str) -> String {
        let separator = if style.windows { "\\" } else { "/" };
        match (&self.path, self.span) {
            (None, _) => text.replace([MAYBE_EMPTY, NONEMPTY], "…"),
            (Some(path), Span::Exact) => path.display(style),
            (Some(path), Span::Contents) => {
                let shown = path.display(style);
                if shown.ends_with(separator) {
                    format!("{shown}*")
                } else {
                    format!("{shown}{separator}*")
                }
            }
            (Some(path), Span::Child) => path.child().display(style),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Span {
    Exact,
    /// Everything inside the path, as with `dir/*`.
    Contents,
    /// Some of the entries inside the path, as with `dir/*.log`.
    Child,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Class {
    Allowed,
    Outside,
    Protected(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Delete,
    Move,
    Overwrite,
    Append,
    Permissions { recursive: bool },
    Disk,
}

impl Op {
    fn verb(self) -> &'static str {
        match self {
            Op::Delete => "delete",
            Op::Move => "move",
            Op::Overwrite => "overwrite",
            Op::Append => "write to",
            Op::Permissions { .. } => "change the permissions of",
            Op::Disk => "reformat or repartition",
        }
    }
}

struct Finding {
    program: String,
    op: Op,
    target: String,
    class: Class,
    note: Option<String>,
}

impl Finding {
    fn refusal(self, host: &Host) -> CommandRefusal {
        let note = self
            .note
            .map(|note| format!(" ({note})"))
            .unwrap_or_default();
        let action = format!(
            "`{}` would {} `{}`",
            self.program,
            self.op.verb(),
            self.target
        );
        let (message, protected) = match self.class {
            Class::Protected(reason) => (
                format!(
                    "Refused: {action}, which is {reason}{note}. Terminal commands can't delete, move, overwrite or reformat the filesystem root, drive roots, the home directory or its parents, top-level or system directories, mounted volumes, or devices."
                ),
                true,
            ),
            _ => {
                let workspace = host
                    .workspace
                    .as_ref()
                    .map(|workspace| workspace.display(host.style))
                    .unwrap_or_default();
                (
                    format!(
                        "Refused {action} because it is outside the working directory `{workspace}`{note}. Keep commands that delete, move or overwrite files inside the working directory or a temporary directory. Other paths need the user's permission in the terminal settings."
                    ),
                    false,
                )
            }
        };
        CommandRefusal {
            message,
            target: self.target,
            protected,
        }
    }
}

// -------------------------------------------------------------- analysis

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Known(String),
    Unknown,
}

struct Scope {
    cwd: Option<AbsPath>,
    vars: HashMap<String, Value>,
    /// The command sources a script or evaluates text the guard can't read,
    /// so an unset variable may be set by then.
    opaque: bool,
    /// Paths listed by the first command of each pipeline (`find`, `ls`,
    /// `Get-ChildItem`), which `xargs rm` or `Remove-Item` downstream receive.
    listings: HashMap<usize, Vec<Located>>,
    /// Paths listed in a POSIX subshell, group or compound command that is a
    /// pipeline's member, by the pipeline of the member after it.
    fed_listings: HashMap<usize, Vec<Located>>,
}

impl Scope {
    fn new(cwd: Option<AbsPath>) -> Self {
        Self {
            cwd,
            vars: HashMap::new(),
            opaque: false,
            listings: HashMap::new(),
            fed_listings: HashMap::new(),
        }
    }

    /// A nested shell: same directory, none of the parent's shell variables.
    fn child(&self) -> Self {
        Self {
            opaque: self.opaque,
            ..Self::new(self.cwd.clone())
        }
    }

    /// A subshell (`$(…)`, a command `find` runs): the same directory and
    /// variables.
    fn subshell(&self, cwd: Option<AbsPath>) -> Self {
        Self {
            vars: self.vars.clone(),
            opaque: self.opaque,
            ..Self::new(cwd)
        }
    }

    /// What a subshell in the line (`( … )`, a background list, a pipeline's
    /// member) can change and loses when it ends: the directory and the
    /// variables.
    fn save(&self) -> Saved {
        (self.cwd.clone(), self.vars.clone())
    }

    /// The shell after a subshell ends.
    fn restore(&mut self, (cwd, vars): Saved) {
        self.cwd = cwd;
        self.vars = vars;
    }

    /// The shell after commands that may have run in a subshell or in this
    /// shell: what they changed is only known when they run.
    fn merge(&mut self, (cwd, vars): Saved) {
        if self.cwd != cwd {
            self.cwd = None;
        }
        for (name, value) in &mut self.vars {
            if vars.get(name) != Some(value) {
                *value = Value::Unknown;
            }
        }
    }
}

type Saved = (Option<AbsPath>, HashMap<String, Value>);

#[derive(Debug, Default)]
struct Resolved {
    text: String,
    empty_vars: Vec<String>,
    unknown: bool,
    /// The shell stops before running the command (`${X:?}` with X empty).
    aborts: bool,
}

const MAX_NESTING: usize = 4;

const RM_FAMILY: &[&str] = &["rm", "unlink", "rmdir", "shred", "srm", "wipe"];

static DOTNET_DELETE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r#"(?i)\[(?:system\.)?io\.(?:directory|file)\]::delete\(\s*('(?:[^']|'')*'|"[^"]*"|\$[\w:]+)"#,
    )
    .unwrap()
});

struct Analysis<'h> {
    host: &'h Host,
    rules: &'h Rules<'h>,
    findings: Vec<Finding>,
    /// The first refusal by the rules.
    refusal: Option<CommandRefusal>,
    /// The pipeline of the command being checked.
    pipeline: usize,
}

impl Analysis<'_> {
    fn run(&mut self, dialect: ShellDialect, source: &str, scope: &mut Scope, depth: usize) {
        if depth > MAX_NESTING {
            if !self.rules.is_empty() && self.refusal.is_none() {
                self.refusal = Some(rules::too_deep());
            }
            return;
        }
        let commands = lex::lex(dialect, source);
        if commands
            .iter()
            .any(|command| sources_script(dialect, command))
        {
            scope.opaque = true;
        }
        if dialect == ShellDialect::PowerShell {
            for capture in DOTNET_DELETE.captures_iter(source) {
                if let Some(word) = lex::lex(dialect, &capture[1])
                    .into_iter()
                    .next()
                    .and_then(|command| command.words.into_iter().next())
                {
                    self.target(dialect, scope, &word, Op::Delete, "[IO.Directory]::Delete");
                }
            }
        }
        // What a POSIX subshell changes ends with it; what a pipeline's last
        // member changes is only known when it runs.
        let mut subshells = Vec::new();
        for command in &commands {
            subshells.extend(command.opens.iter().map(|kind| (*kind, scope.save())));
            self.command(dialect, command, scope, depth);
            for _ in 0..command.closes {
                match subshells.pop() {
                    Some((Subshell::Always, saved)) => scope.restore(saved),
                    Some((Subshell::Maybe, saved)) => scope.merge(saved),
                    None => {}
                }
            }
        }
    }

    fn command(
        &mut self,
        dialect: ShellDialect,
        command: &Command,
        scope: &mut Scope,
        depth: usize,
    ) {
        self.substitutions(dialect, command, scope, depth);
        self.pipeline = command.pipeline;
        for redirect in &command.redirects {
            let op = if redirect.append {
                Op::Append
            } else {
                Op::Overwrite
            };
            let symbol = if redirect.append { ">>" } else { ">" };
            self.target(dialect, scope, &redirect.target, op, symbol);
        }
        let mut words: &[Word] = &command.words;
        match dialect {
            ShellDialect::Posix => {
                let assignments = words.iter().take_while(|w| assignment(w).is_some()).count();
                if assignments == words.len() {
                    for word in words {
                        self.assign_posix(scope, word);
                    }
                    return;
                }
                words = &words[assignments..];
            }
            ShellDialect::PowerShell if self.assign_powershell(scope, words) => return,
            _ => {}
        }
        // PowerShell runs a quoted or computed first word only after `&`;
        // otherwise the statement is an expression.
        let expression = dialect == ShellDialect::PowerShell
            && !command.call
            && words
                .first()
                .is_some_and(|word| word.quoted || word.literal().is_none());
        if !expression {
            self.check_rules(dialect, &command.words, scope);
        }
        // The directory a wrapper (`env -C`, `sudo -D`, `unshare -w`) runs
        // the command in.
        let mut directory = None;
        loop {
            let Some(name) = words.first().and_then(command_name) else {
                return;
            };
            match skip_prefix(dialect, &name, &words[1..]) {
                Some(skipped) => {
                    let end = (1 + skipped).min(words.len());
                    let from = directory.clone().unwrap_or_else(|| scope.cwd.clone());
                    if dialect == ShellDialect::Posix
                        && let Some(changed) =
                            self.wrapper_directory(&name, &words[1..end], scope, from.as_ref())
                    {
                        directory = Some(changed);
                    }
                    words = &words[end..];
                }
                None => break,
            }
        }
        let Some(name) = words.first().and_then(command_name) else {
            return;
        };
        let program = words[0].literal().unwrap_or_else(|| name.clone());
        let rest = &words[1..];
        // The command runs there; the shell stays where it is.
        let shell_directory = directory.map(|cwd| std::mem::replace(&mut scope.cwd, cwd));
        if !command.piped
            && let Some(listing) = self.listing(dialect, &name, rest, scope)
        {
            for pipeline in &command.feeds {
                let fed = scope.fed_listings.entry(*pipeline).or_default();
                fed.extend(listing.iter().cloned());
            }
            scope.listings.insert(command.pipeline, listing);
        }
        match dialect {
            ShellDialect::Posix => self.posix(&name, &program, rest, command, scope, depth),
            ShellDialect::Cmd => self.cmd(&name, &program, rest, command, scope, depth),
            ShellDialect::PowerShell => {
                self.powershell(&name, &program, rest, command, scope, depth)
            }
        }
        if let Some(cwd) = shell_directory {
            scope.cwd = cwd;
        }
    }

    // -------------------------------------------------------------- rules

    /// Checks the command `words` run against the rules, with the commands
    /// that wrappers (`sudo`, `env`, `timeout`, …) in front of it run.
    fn check_rules(&mut self, dialect: ShellDialect, words: &[Word], scope: &Scope) {
        if self.rules.is_empty() || self.refusal.is_some() {
            return;
        }
        let assignments = match dialect {
            ShellDialect::Posix => words.iter().take_while(|w| assignment(w).is_some()).count(),
            _ => 0,
        };
        // A git command run with another repository's variables isn't in
        // the directory it runs in.
        let mut elsewhere = words[..assignments]
            .iter()
            .filter_map(assignment)
            .any(|(name, _)| names_repository(&name))
            || REPOSITORY_VARIABLES.iter().any(|&name| {
                // Shell variables as each dialect stores them.
                let lower = name.to_ascii_lowercase();
                let set = [name.to_string(), format!("env:{lower}"), lower]
                    .iter()
                    .filter_map(|key| scope.vars.get(key))
                    .any(|value| !matches!(value, Value::Known(text) if text.is_empty()));
                set || self.host.env(name).is_some_and(|value| !value.is_empty())
            });
        let mut directory = scope.cwd.clone();
        let words = self.rule_words(dialect, scope, &words[assignments..]);
        let mut rest = words.as_slice();
        while let Some(program) = rest.first() {
            let name = program.as_deref().map(base_name);
            if let Err(refusal) = self.rules.check(rest, directory.as_ref(), elsewhere) {
                self.refusal = Some(refusal);
                return;
            }
            let Some(name) = name else {
                return;
            };
            let arguments: Vec<Word> = rest[1..]
                .iter()
                .map(|word| match word {
                    Some(text) => literal_word(text),
                    None => Word {
                        pieces: vec![Piece::Unknown],
                        quoted: false,
                    },
                })
                .collect();
            let Some(skipped) = skip_prefix(dialect, &name, &arguments) else {
                return;
            };
            let (options, command) = rest.split_at((1 + skipped).min(rest.len()));
            let any = |test: fn(&str) -> bool| options[1..].iter().flatten().any(|o| test(o));
            match name.as_str() {
                // `command -v git` only looks the program up.
                "command" if any(|o| o == "-v" || o == "-V") => return,
                // `env -C`, `sudo -D` and `unshare -w` run the command in
                // another directory, `unshare -R` under another root.
                "env" | "sudo" if any(|o| matches!(o, "-C" | "-D") || o.starts_with("--chdir")) => {
                    directory = None;
                }
                "unshare"
                    if any(|o| {
                        ["-w", "--wd", "-R", "--root"]
                            .iter()
                            .any(|option| o.starts_with(option))
                    }) =>
                {
                    directory = None;
                }
                _ => {}
            }
            elsewhere |= matches!(name.as_str(), "env" | "sudo" | "strace") && any(sets_repository);
            rest = command;
        }
    }

    /// `words` as the rules see them: resolved, with an unquoted variable
    /// split into words as the shell splits it (PowerShell doesn't), and
    /// `None` for a word only known when the command runs.
    fn rule_words(
        &self,
        dialect: ShellDialect,
        scope: &Scope,
        words: &[Word],
    ) -> Vec<Option<String>> {
        let mut resolved_words = Vec::new();
        for word in words {
            if let Some(text) = word.literal() {
                resolved_words.push(Some(text));
                continue;
            }
            let resolved = self.resolve(dialect, scope, word);
            let splits = !word.quoted
                && dialect != ShellDialect::PowerShell
                && word
                    .pieces
                    .iter()
                    .any(|piece| matches!(piece, Piece::Var(..)));
            if resolved.unknown || resolved.aborts {
                resolved_words.push(None);
            } else if splits {
                resolved_words.extend(
                    resolved
                        .text
                        .split_whitespace()
                        .map(|text| Some(text.to_string())),
                );
            } else {
                resolved_words.push(Some(resolved.text));
            }
        }
        resolved_words
    }

    /// Checks the commands in `command`'s substitutions (`$(…)`, backticks,
    /// PowerShell's `(…)`) against the rules. Only the rules: the guard
    /// doesn't follow the paths they act on.
    fn substitutions(
        &mut self,
        dialect: ShellDialect,
        command: &Command,
        scope: &Scope,
        depth: usize,
    ) {
        if self.rules.is_empty() {
            return;
        }
        let words = command
            .words
            .iter()
            .chain(command.redirects.iter().map(|redirect| &redirect.target));
        for piece in words.flat_map(|word| &word.pieces) {
            if let Piece::Substitution(text) | Piece::Group(text) = piece {
                let mut subshell = scope.subshell(scope.cwd.clone());
                self.rules_only(|analysis| analysis.run(dialect, text, &mut subshell, depth + 1));
            }
        }
    }

    /// The commands `find -exec`, `-execdir`, `-ok` and `-okdir` run,
    /// checked against the rules only: the deletes `find` itself does are
    /// checked with its other arguments.
    fn find_commands(&mut self, rest: &[Word], command: &Command, scope: &Scope, depth: usize) {
        if self.rules.is_empty() {
            return;
        }
        let mut index = 0;
        while let Some(word) = rest.get(index) {
            index += 1;
            let flag = word.literal().unwrap_or_default();
            if !matches!(flag.as_str(), "-exec" | "-execdir" | "-ok" | "-okdir") {
                continue;
            }
            let end = rest[index..]
                .iter()
                .position(|word| matches!(word.literal().as_deref(), Some(";" | "+")))
                .map_or(rest.len(), |at| index + at);
            let inner = &rest[index..end];
            index = end + 1;
            // `-execdir` runs the command in each found file's directory.
            let cwd = (!flag.ends_with("dir"))
                .then(|| scope.cwd.clone())
                .flatten();
            let mut subshell = scope.subshell(cwd);
            self.check_rules(ShellDialect::Posix, inner, &subshell);
            if let Some(name) = inner.first().and_then(command_name) {
                let program = inner[0].literal().unwrap_or_else(|| name.clone());
                self.rules_only(|analysis| {
                    analysis.posix(
                        &name,
                        &program,
                        &inner[1..],
                        command,
                        &mut subshell,
                        depth + 1,
                    )
                });
            }
        }
    }

    /// Runs `check` for the rules only: the path findings it adds are
    /// dropped.
    fn rules_only(&mut self, check: impl FnOnce(&mut Self)) {
        let (findings, pipeline) = (self.findings.len(), self.pipeline);
        check(self);
        self.findings.truncate(findings);
        self.pipeline = pipeline;
    }

    /// A command the guard can't read in advance, which `what` runs:
    /// refused while there are rules, since they can't check it.
    fn unchecked(&mut self, what: &str) {
        if !self.rules.is_empty() && self.refusal.is_none() {
            self.refusal = Some(rules::unchecked(what));
        }
    }

    // ------------------------------------------------------------ targets

    fn target(&mut self, dialect: ShellDialect, scope: &Scope, word: &Word, op: Op, program: &str) {
        let mut resolved = self.resolve(dialect, scope, word);
        if resolved.aborts || resolved.text.is_empty() {
            return;
        }
        if dialect == ShellDialect::PowerShell {
            match powershell_provider(&resolved.text, self.temp_text().as_deref()) {
                None => {}
                Some(ProviderPath::File(path)) => resolved.text = path,
                Some(ProviderPath::NotFile) => return,
                Some(ProviderPath::Elsewhere) => {
                    let text = resolved.text.clone();
                    self.record(&Located::unknown(), op, program, &text, None);
                    return;
                }
            }
        }
        let note = if !resolved.empty_vars.is_empty() {
            Some(format!(
                "{} {} empty or not set here",
                resolved.empty_vars.join(", "),
                if resolved.empty_vars.len() == 1 {
                    "is"
                } else {
                    "are"
                }
            ))
        } else if resolved.unknown {
            Some("part of the path is only known when the command runs".to_string())
        } else {
            None
        };
        for place in self.host.locate(&resolved.text, scope.cwd.as_ref()) {
            self.record(&place, op, program, &resolved.text, note.clone());
        }
    }

    /// The temporary directory, for PowerShell's `Temp:` drive.
    fn temp_text(&self) -> Option<String> {
        self.host
            .temp
            .first()
            .map(|temp| temp.display(self.host.style))
    }

    fn record(&mut self, place: &Located, op: Op, program: &str, text: &str, note: Option<String>) {
        let class = self.host.judge(place, op);
        if class != Class::Allowed {
            self.findings.push(Finding {
                program: program.to_string(),
                op,
                target: place.display(self.host.style, text),
                class,
                note,
            });
        }
    }

    fn targets(
        &mut self,
        dialect: ShellDialect,
        scope: &Scope,
        words: &[&Word],
        op: Op,
        program: &str,
    ) {
        for word in words {
            if is_pipeline_item(word) {
                self.pipeline_input(scope, op, program);
            } else {
                self.target(dialect, scope, word, op, program);
            }
        }
    }

    /// What a command receives through the pipeline: the upstream listing,
    /// or the working directory's contents when that is unknown.
    fn pipeline_input(&mut self, scope: &Scope, op: Op, program: &str) {
        let listed = scope.listings.get(&self.pipeline).cloned();
        let mut places = listed.unwrap_or_else(|| match &scope.cwd {
            Some(cwd) => vec![Located {
                path: Some(cwd.clone()),
                span: Span::Contents,
                speculative: false,
            }],
            None => vec![Located::unknown()],
        });
        if let Some(fed) = scope.fed_listings.get(&self.pipeline) {
            places.extend(fed.iter().cloned());
        }
        for place in places {
            self.record(&place, op, program, "the pipeline input", None);
        }
    }

    fn disk(&mut self, program: &str, what: &str) {
        self.findings.push(Finding {
            program: program.to_string(),
            op: Op::Disk,
            target: what.to_string(),
            class: Class::Protected("a disk"),
            note: None,
        });
    }

    fn listing(
        &self,
        dialect: ShellDialect,
        name: &str,
        rest: &[Word],
        scope: &Scope,
    ) -> Option<Vec<Located>> {
        let (words, span): (Vec<&Word>, Span) = match (dialect, name) {
            (ShellDialect::Posix, "find") => (find_paths(rest), Span::Exact),
            (ShellDialect::Posix, "ls") => (posix_operands(rest, &[]), Span::Contents),
            (ShellDialect::PowerShell, "get-childitem" | "gci" | "ls" | "dir") => {
                (powershell_arguments(rest).paths, Span::Contents)
            }
            (ShellDialect::PowerShell, "get-item" | "gi") => {
                (powershell_arguments(rest).paths, Span::Exact)
            }
            _ => return None,
        };
        let mut places = Vec::new();
        if words.is_empty() {
            places.push(Located {
                path: scope.cwd.clone(),
                span,
                speculative: false,
            });
        }
        for word in words {
            let resolved = self.resolve(dialect, scope, word);
            for mut place in self.host.locate(&resolved.text, scope.cwd.as_ref()) {
                if place.span == Span::Exact {
                    place.span = span;
                }
                places.push(place);
            }
        }
        Some(places)
    }

    // ------------------------------------------------------------ values

    fn resolve(&self, dialect: ShellDialect, scope: &Scope, word: &Word) -> Resolved {
        let mut resolved = Resolved::default();
        let separator = if self.host.style.windows { '\\' } else { '/' };
        for piece in &word.pieces {
            match piece {
                Piece::Text(text) => resolved.text.push_str(text),
                Piece::Tilde(None) => match self.host.home_text() {
                    Some(home) => resolved.text.push_str(&home),
                    None => resolved.text.push(NONEMPTY),
                },
                Piece::Tilde(Some(user)) => {
                    match self.host.home.as_ref().and_then(AbsPath::parent) {
                        Some(parent) => {
                            let parent = parent.display(self.host.style);
                            resolved.text.push_str(parent.trim_end_matches(separator));
                            resolved.text.push(separator);
                            resolved.text.push_str(user);
                        }
                        None => resolved.text.push(NONEMPTY),
                    }
                }
                Piece::WorkingDirectory => match &scope.cwd {
                    Some(cwd) => resolved.text.push_str(&cwd.display(self.host.style)),
                    None => {
                        resolved.text.push(NONEMPTY);
                        resolved.unknown = true;
                    }
                },
                Piece::TempPath => match self.host.temp.first() {
                    Some(temp) => {
                        let temp = temp.display(self.host.style);
                        resolved.text.push_str(temp.trim_end_matches(separator));
                        resolved.text.push(separator);
                        resolved.text.push_str("tmp.XXXXXXXXXX");
                    }
                    None => {
                        resolved.text.push(NONEMPTY);
                        resolved.unknown = true;
                    }
                },
                Piece::Unknown | Piece::Substitution(_) | Piece::Group(_) => {
                    resolved.text.push(MAYBE_EMPTY);
                    resolved.unknown = true;
                }
                Piece::Var(name, form) => {
                    let value = match form {
                        VarForm::Opaque => Value::Unknown,
                        _ => self.lookup(dialect, scope, name),
                    };
                    let shown = variable_name(dialect, name);
                    match (form, value) {
                        (VarForm::Required, Value::Known(value)) if value.is_empty() => {
                            resolved.aborts = true;
                        }
                        (VarForm::Default(default), Value::Known(value)) if value.is_empty() => {
                            resolved.text.push_str(default);
                        }
                        (VarForm::Transform, Value::Known(value)) if !value.is_empty() => {
                            resolved.text.push(MAYBE_EMPTY);
                            resolved.unknown = true;
                        }
                        (_, Value::Known(value)) => {
                            if value.is_empty() {
                                resolved.empty_vars.push(shown);
                            }
                            resolved.text.push_str(&value);
                        }
                        (VarForm::Required, Value::Unknown) => {
                            resolved.text.push(NONEMPTY);
                            resolved.unknown = true;
                        }
                        (VarForm::Default(default), Value::Unknown) if !default.is_empty() => {
                            resolved.text.push(NONEMPTY);
                            resolved.unknown = true;
                        }
                        (_, Value::Unknown) => {
                            resolved.text.push(MAYBE_EMPTY);
                            resolved.unknown = true;
                        }
                    }
                }
            }
        }
        resolved
    }

    fn lookup(&self, dialect: ShellDialect, scope: &Scope, name: &str) -> Value {
        let unset = |fallback: String| {
            if scope.opaque {
                Value::Unknown
            } else {
                Value::Known(fallback)
            }
        };
        let cwd = || match &scope.cwd {
            Some(cwd) => Value::Known(cwd.display(self.host.style)),
            None => Value::Unknown,
        };
        match dialect {
            ShellDialect::Posix => {
                if let Some(value) = scope.vars.get(name) {
                    return value.clone();
                }
                if name == "PWD" {
                    return cwd();
                }
                if let Some(value) = self.host.env(name) {
                    return Value::Known(value);
                }
                if name == "HOME"
                    && let Some(home) = self.host.home_text()
                {
                    return Value::Known(home);
                }
                unset(String::new())
            }
            ShellDialect::Cmd => {
                let lower = name.to_ascii_lowercase();
                if let Some(value) = scope.vars.get(&lower) {
                    return value.clone();
                }
                if lower == "cd" {
                    return cwd();
                }
                match self.host.env(name) {
                    Some(value) => Value::Known(value),
                    // cmd leaves an undefined `%NAME%` in the command as typed.
                    None => unset(format!("%{name}%")),
                }
            }
            ShellDialect::PowerShell => {
                if let Some(variable) = name.strip_prefix("env:") {
                    return match self.host.env(variable) {
                        Some(value) => Value::Known(value),
                        None => unset(String::new()),
                    };
                }
                if let Some(value) = scope.vars.get(name) {
                    return value.clone();
                }
                match name {
                    "home" => self.host.home_text().map_or(Value::Unknown, Value::Known),
                    "pwd" => cwd(),
                    "null" => Value::Known(String::new()),
                    "true" => Value::Known("True".into()),
                    "false" => Value::Known("False".into()),
                    "_" | "psitem" | "psscriptroot" | "pscommandpath" | "myinvocation" | "args"
                    | "input" | "matches" | "lastexitcode" | "profile" => Value::Unknown,
                    _ => unset(String::new()),
                }
            }
        }
    }

    fn value(&self, dialect: ShellDialect, scope: &Scope, word: &Word) -> Value {
        let resolved = self.resolve(dialect, scope, word);
        if resolved.unknown || resolved.aborts {
            Value::Unknown
        } else {
            Value::Known(resolved.text)
        }
    }

    fn assign_posix(&self, scope: &mut Scope, word: &Word) {
        if let Some((name, value)) = assignment(word) {
            let value = match value {
                Some(mut value) => {
                    if let Some(Piece::Text(text)) = value.pieces.first_mut()
                        && (text == "~" || text.starts_with("~/"))
                        && let Some(home) = self.host.home_text()
                    {
                        *text = text.replacen('~', &home, 1);
                    }
                    self.value(ShellDialect::Posix, scope, &value)
                }
                None => Value::Unknown,
            };
            scope.vars.insert(name, value);
        }
    }

    /// `$name = value`. True when `words` is an assignment.
    fn assign_powershell(&self, scope: &mut Scope, words: &[Word]) -> bool {
        const OPERATORS: [&str; 5] = ["+=", "-=", "*=", "/=", "="];
        let Some(first) = words.first() else {
            return false;
        };
        let Some(Piece::Var(name, VarForm::Plain)) = first.pieces.first() else {
            return false;
        };
        let (operator, rhs) = match first.pieces.get(1) {
            // `$x='a'`: the operator and value are in the same word.
            Some(Piece::Text(text)) => {
                let Some(operator) = OPERATORS.into_iter().find(|op| text.starts_with(op)) else {
                    return false;
                };
                let tail = Word {
                    pieces: std::iter::once(Piece::Text(text[operator.len()..].to_string()))
                        .chain(first.pieces[2..].iter().cloned())
                        .filter(|piece| *piece != Piece::Text(String::new()))
                        .collect(),
                    quoted: first.quoted,
                };
                let mut rhs = vec![tail];
                rhs.extend(words[1..].iter().cloned());
                rhs.retain(|word| !word.pieces.is_empty());
                (operator, rhs)
            }
            None => match words.get(1).and_then(Word::literal) {
                Some(text) => match OPERATORS.into_iter().find(|op| *op == text) {
                    Some(operator) => (operator, words[2..].to_vec()),
                    None => return false,
                },
                None => return false,
            },
            _ => return false,
        };
        let value = match rhs.as_slice() {
            [word]
                if operator == "="
                    && (word.quoted
                        || matches!(
                            word.pieces.first(),
                            Some(Piece::Var(..) | Piece::Tilde(_))
                        )) =>
            {
                self.value(ShellDialect::PowerShell, scope, word)
            }
            _ => Value::Unknown,
        };
        scope.vars.insert(name.clone(), value);
        true
    }

    /// The directory a wrapper's `options` run its command in, when they name
    /// one (`env -C`, `env --chdir=`, `sudo -D`, `unshare -w`): resolved from
    /// `from` as `cd` resolves it, `None` inside when it is only known when the
    /// command runs.
    fn wrapper_directory(
        &self,
        name: &str,
        options: &[Word],
        scope: &Scope,
        from: Option<&AbsPath>,
    ) -> Option<Option<AbsPath>> {
        let (short, long) = match name {
            "env" => ("-C", "--chdir"),
            "sudo" => ("-D", "--chdir"),
            "unshare" => ("-w", "--wd"),
            _ => return None,
        };
        let mut value = None;
        let mut words = options.iter();
        while let Some(word) = words.next() {
            match word.literal().as_deref() {
                Some(option) if option == short || option == long => {
                    value = words.next().cloned();
                }
                _ => {
                    if let Some(attached) = strip_word_prefix(word, &format!("{long}="))
                        .or_else(|| strip_word_prefix(word, short))
                    {
                        value = Some(attached);
                    }
                }
            }
        }
        let resolved = self.resolve(ShellDialect::Posix, scope, &value?);
        Some(
            if resolved.unknown || resolved.aborts || resolved.text.is_empty() {
                None
            } else {
                AbsPath::parse(self.host.style, &resolved.text, from)
            },
        )
    }

    fn change_directory(&self, dialect: ShellDialect, scope: &mut Scope, rest: &[Word]) {
        let operand = match dialect {
            ShellDialect::Posix => posix_operands(rest, &[]).first().copied(),
            ShellDialect::Cmd => cmd_operands(rest).first().copied(),
            ShellDialect::PowerShell => powershell_arguments(rest).paths.first().copied(),
        };
        let home = || self.host.home.clone();
        scope.cwd = match operand {
            None if dialect == ShellDialect::Cmd => return,
            None => home(),
            Some(word) => {
                let resolved = self.resolve(dialect, scope, word);
                if resolved.aborts {
                    return;
                }
                if resolved.text.is_empty() && !word.quoted && dialect == ShellDialect::Posix {
                    home()
                } else if resolved.unknown
                    || matches!(resolved.text.as_str(), "-" | "+")
                    || (dialect == ShellDialect::Cmd
                        && resolved.text.len() == 2
                        && resolved.text.ends_with(':'))
                {
                    None
                } else if dialect == ShellDialect::PowerShell {
                    // `cd Env:` or `cd HKCU:\Software` leaves the filesystem,
                    // so later relative paths aren't known.
                    match powershell_provider(&resolved.text, self.temp_text().as_deref()) {
                        None => AbsPath::parse(self.host.style, &resolved.text, scope.cwd.as_ref()),
                        Some(ProviderPath::File(path)) => {
                            AbsPath::parse(self.host.style, &path, scope.cwd.as_ref())
                        }
                        Some(ProviderPath::NotFile | ProviderPath::Elsewhere) => None,
                    }
                } else {
                    AbsPath::parse(self.host.style, &resolved.text, scope.cwd.as_ref())
                }
            }
        };
    }

    // ------------------------------------------------------ nested shells

    fn nested(&mut self, dialect: ShellDialect, text: &str, scope: &Scope, depth: usize) {
        let mut child = scope.child();
        self.run(dialect, text, &mut child, depth + 1);
    }

    /// Joins `words` into command text. `None` when part of it is unknown.
    fn join(&self, dialect: ShellDialect, scope: &Scope, words: &[Word]) -> Option<String> {
        let mut parts = Vec::new();
        for word in words {
            let resolved = self.resolve(dialect, scope, word);
            if resolved.unknown {
                return None;
            }
            if word.quoted && resolved.text.contains(' ') && words.len() > 1 {
                parts.push(format!("\"{}\"", resolved.text));
            } else {
                parts.push(resolved.text);
            }
        }
        Some(parts.join(" "))
    }

    /// Checks command `text` a program runs through a POSIX shell. `what`
    /// names it when the text is only known when it runs.
    fn shell_text(&mut self, text: Option<&Word>, what: &str, scope: &Scope, depth: usize) {
        let Some(word) = text else {
            return;
        };
        let resolved = self.resolve(ShellDialect::Posix, scope, word);
        if resolved.unknown {
            self.unchecked(what);
        } else {
            self.nested(ShellDialect::Posix, &resolved.text, scope, depth);
        }
    }

    fn posix_shell(&mut self, dialect: ShellDialect, rest: &[Word], scope: &Scope, depth: usize) {
        let mut index = 0;
        let mut script = false;
        while let Some(option) = rest.get(index).and_then(Word::literal) {
            if !option.starts_with('-') {
                break;
            }
            if !option.starts_with("--") && option[1..].contains('c') {
                script = true;
            }
            index += 1;
        }
        if let (true, Some(word)) = (script, rest.get(index)) {
            let resolved = self.resolve(dialect, scope, word);
            if !resolved.unknown {
                self.nested(ShellDialect::Posix, &resolved.text, scope, depth);
            } else {
                self.unchecked("`sh -c`");
            }
        }
    }

    fn cmd_shell(&mut self, dialect: ShellDialect, rest: &[Word], scope: &Scope, depth: usize) {
        let start = rest.iter().position(|word| {
            word.literal().is_some_and(|flag| {
                matches!(flag.to_ascii_lowercase().as_str(), "/c" | "/k" | "/r")
            })
        });
        if let Some(start) = start {
            match self.join(dialect, scope, &rest[start + 1..]) {
                Some(text) => self.nested(ShellDialect::Cmd, &text, scope, depth),
                None => self.unchecked("`cmd /c`"),
            }
        }
    }

    fn powershell_shell(
        &mut self,
        dialect: ShellDialect,
        name: &str,
        rest: &[Word],
        scope: &Scope,
        depth: usize,
    ) {
        const VALUED: &[&str] = &[
            "executionpolicy",
            "ep",
            "windowstyle",
            "w",
            "workingdirectory",
            "wd",
            "configurationname",
            "outputformat",
            "of",
            "inputformat",
            "if",
            "settingsfile",
            "custompipename",
            "version",
            "v",
            "psconsolefile",
        ];
        let mut index = 0;
        while let Some(word) = rest.get(index) {
            let Some(flag) = word
                .literal()
                .and_then(|text| text.strip_prefix('-').map(str::to_ascii_lowercase))
            else {
                // powershell.exe runs a bare argument as a command; pwsh runs it as a file.
                if name == "powershell" {
                    match self.join(dialect, scope, &rest[index..]) {
                        Some(text) => self.nested(ShellDialect::PowerShell, &text, scope, depth),
                        None => self.unchecked("`powershell`"),
                    }
                }
                return;
            };
            if !flag.is_empty() && "command".starts_with(&flag) && flag != "co" {
                match self.join(dialect, scope, &rest[index + 1..]) {
                    Some(text) => self.nested(ShellDialect::PowerShell, &text, scope, depth),
                    None => self.unchecked("`powershell -Command`"),
                }
                return;
            }
            if !flag.is_empty() && "encodedcommand".starts_with(&flag) || flag == "ec" {
                match rest
                    .get(index + 1)
                    .and_then(Word::literal)
                    .and_then(|encoded| decode_utf16_base64(&encoded))
                {
                    Some(text) => self.nested(ShellDialect::PowerShell, &text, scope, depth),
                    None => self.unchecked("`powershell -EncodedCommand`"),
                }
                return;
            }
            if flag == "file" || flag == "f" {
                return;
            }
            index += if VALUED.contains(&flag.as_str()) {
                2
            } else {
                1
            };
        }
    }

    // ------------------------------------------------------ POSIX commands

    fn posix(
        &mut self,
        name: &str,
        program: &str,
        rest: &[Word],
        command: &Command,
        scope: &mut Scope,
        depth: usize,
    ) {
        const DIALECT: ShellDialect = ShellDialect::Posix;
        match name {
            "cd" | "pushd" | "chdir" => self.change_directory(DIALECT, scope, rest),
            "popd" => scope.cwd = None,
            "export" | "local" | "declare" | "typeset" | "readonly" => {
                for word in rest {
                    self.assign_posix(scope, word);
                }
            }
            "unset" => {
                for name in rest.iter().filter_map(Word::literal) {
                    scope.vars.insert(name, Value::Known(String::new()));
                }
            }
            "read" | "mapfile" | "readarray" | "for" | "select" | "getopts" => {
                for name in posix_operands(rest, &["-d", "-n", "-N", "-p", "-t", "-u"])
                    .into_iter()
                    .filter_map(Word::literal)
                {
                    scope.vars.insert(name, Value::Unknown);
                }
            }
            "eval" => match self.join(DIALECT, scope, rest) {
                Some(text) => self.run(DIALECT, &text, scope, depth + 1),
                None => self.unchecked("`eval`"),
            },
            "sh" | "bash" | "zsh" | "dash" | "ksh" | "mksh" | "ash" | "fish" => {
                self.posix_shell(DIALECT, rest, scope, depth)
            }
            "cmd" => self.cmd_shell(DIALECT, rest, scope, depth),
            "powershell" | "pwsh" => self.powershell_shell(DIALECT, name, rest, scope, depth),
            // Programs that run command text through a shell.
            "su" => {
                let text = command_option(rest, &["--command", "--session-command"]);
                self.shell_text(text.as_ref(), "`su -c`", scope, depth);
            }
            "script" => {
                let text = command_option(rest, &["--command"]);
                self.shell_text(text.as_ref(), "`script -c`", scope, depth);
            }
            "flock" => {
                let lock = leading_options(rest, FLOCK_VALUED, false);
                if matches!(
                    rest.get(lock + 1).and_then(Word::literal).as_deref(),
                    Some("-c" | "--command")
                ) {
                    self.shell_text(rest.get(lock + 2), "`flock -c`", scope, depth);
                }
            }
            "watch" => {
                let start = leading_options(rest, WATCH_VALUED, false).min(rest.len());
                match self.join(DIALECT, scope, &rest[start..]) {
                    Some(text) => self.nested(DIALECT, &text, scope, depth),
                    None => self.unchecked("`watch`"),
                }
            }
            "xargs" => self.xargs(rest, command, scope, depth),
            "find" => {
                self.find_commands(rest, command, scope, depth);
                self.posix_rules(name, program, rest, scope);
            }
            _ => self.posix_rules(name, program, rest, scope),
        }
    }

    fn posix_rules(&mut self, name: &str, program: &str, rest: &[Word], scope: &Scope) {
        const DIALECT: ShellDialect = ShellDialect::Posix;
        match name {
            _ if RM_FAMILY.contains(&name) => {
                let operands = posix_operands(
                    rest,
                    &["-n", "-s", "--iterations", "--size", "--random-source"],
                );
                self.targets(DIALECT, scope, &operands, Op::Delete, program);
            }
            "find" => {
                let literals: Vec<String> = rest.iter().filter_map(Word::literal).collect();
                let deletes = literals.iter().enumerate().any(|(index, flag)| {
                    flag == "-delete"
                        || (matches!(flag.as_str(), "-exec" | "-execdir" | "-ok" | "-okdir")
                            && literals
                                .get(index + 1)
                                .is_some_and(|next| RM_FAMILY.contains(&base_name(next).as_str())))
                });
                if deletes {
                    let paths = find_paths(rest);
                    if paths.is_empty() {
                        self.target(DIALECT, scope, &literal_word("."), Op::Delete, program);
                    }
                    self.targets(DIALECT, scope, &paths, Op::Delete, program);
                }
            }
            "mv" => {
                let target_directory = rest
                    .iter()
                    .filter_map(Word::literal)
                    .any(|flag| flag == "-t" || flag.starts_with("--target-directory"));
                let mut operands =
                    posix_operands(rest, &["-t", "-S", "--suffix", "--target-directory"]);
                if !target_directory {
                    operands.pop();
                }
                self.targets(DIALECT, scope, &operands, Op::Move, program);
            }
            "git" => self.git(program, rest, scope),
            "rsync" => self.rsync(program, rest, scope),
            "truncate" => {
                let operands = posix_operands(rest, &["-s", "-r", "--size", "--reference"]);
                self.targets(DIALECT, scope, &operands, Op::Overwrite, program);
            }
            "dd" => {
                for word in rest {
                    if let Some(output) = strip_word_prefix(word, "of=") {
                        self.target(DIALECT, scope, &output, Op::Overwrite, program);
                    }
                }
            }
            "tee" => {
                let append = rest
                    .iter()
                    .filter_map(Word::literal)
                    .any(|flag| flag == "-a" || flag == "--append");
                let op = if append { Op::Append } else { Op::Overwrite };
                self.targets(DIALECT, scope, &posix_operands(rest, &[]), op, program);
            }
            "chmod" | "chown" | "chgrp" | "setfacl" => {
                let literals: Vec<String> = rest.iter().filter_map(Word::literal).collect();
                let recursive = literals.iter().any(|flag| {
                    flag == "--recursive"
                        || (flag.starts_with('-') && !flag.starts_with("--") && flag.contains('R'))
                });
                let mut operands = posix_operands(
                    rest,
                    &[
                        "-m",
                        "-M",
                        "-x",
                        "-X",
                        "--set",
                        "--set-file",
                        "--modify",
                        "--modify-file",
                        "--remove",
                        "--remove-file",
                        "--from",
                    ],
                );
                let has_reference = literals.iter().any(|flag| flag.starts_with("--reference"));
                if name != "setfacl" && !has_reference && !operands.is_empty() {
                    operands.remove(0);
                }
                self.targets(
                    DIALECT,
                    scope,
                    &operands,
                    Op::Permissions { recursive },
                    program,
                );
            }
            "mke2fs" | "mkswap" | "wipefs" | "blkdiscard" => {
                self.targets(
                    DIALECT,
                    scope,
                    &posix_operands(rest, &["-t", "-L", "-b"]),
                    Op::Disk,
                    program,
                );
            }
            _ if name.starts_with("mkfs") => {
                self.targets(
                    DIALECT,
                    scope,
                    &posix_operands(rest, &["-t", "-L", "-b"]),
                    Op::Disk,
                    program,
                );
            }
            "fdisk" | "sfdisk" | "cfdisk" | "gdisk" | "sgdisk" | "parted" => {
                let lists = rest.iter().filter_map(Word::literal).any(|flag| {
                    matches!(
                        flag.as_str(),
                        "-l" | "--list" | "-p" | "--print" | "print" | "-d" | "--dump" | "-L"
                    )
                });
                if !lists {
                    self.targets(
                        DIALECT,
                        scope,
                        &posix_operands(rest, &[]),
                        Op::Disk,
                        program,
                    );
                }
            }
            "diskutil" => {
                let operands: Vec<String> = posix_operands(rest, &[])
                    .into_iter()
                    .filter_map(Word::literal)
                    .map(|operand| operand.to_ascii_lowercase())
                    .collect();
                let erase = |verb: &str| {
                    matches!(
                        verb,
                        "erasedisk"
                            | "erasevolume"
                            | "reformat"
                            | "partitiondisk"
                            | "zerodisk"
                            | "randomdisk"
                            | "secureerase"
                            | "splitpartition"
                            | "mergepartitions"
                            | "deletecontainer"
                            | "deletevolume"
                    )
                };
                let first = operands.first().map(String::as_str).unwrap_or_default();
                if erase(first)
                    || (first == "apfs" && operands.get(1).is_some_and(|verb| erase(verb)))
                {
                    self.disk(program, &format!("diskutil {}", operands.join(" ")));
                }
            }
            _ => {}
        }
    }

    fn xargs(&mut self, rest: &[Word], command: &Command, scope: &mut Scope, depth: usize) {
        const VALUED: &[&str] = &[
            "-n",
            "-L",
            "-P",
            "-s",
            "-d",
            "-E",
            "-I",
            "-a",
            "--delimiter",
            "--max-args",
            "--max-procs",
            "--arg-file",
            "--replace",
        ];
        let mut index = 0;
        while let Some(flag) = rest.get(index).and_then(Word::literal) {
            if !flag.starts_with('-') {
                break;
            }
            index += if VALUED.contains(&flag.as_str()) {
                2
            } else {
                1
            };
        }
        let inner = &rest[index.min(rest.len())..];
        self.check_rules(ShellDialect::Posix, inner, scope);
        let Some(name) = inner.first().and_then(command_name) else {
            return;
        };
        let program = inner[0].literal().unwrap_or_else(|| name.clone());
        if RM_FAMILY.contains(&name.as_str()) {
            self.pipeline_input(scope, Op::Delete, &program);
        } else if name == "mv" {
            self.pipeline_input(scope, Op::Move, &program);
        }
        self.posix(&name, &program, &inner[1..], command, scope, depth);
    }

    fn git(&mut self, program: &str, rest: &[Word], scope: &Scope) {
        let mut index = 0;
        let mut directory = scope.cwd.clone();
        while let Some(flag) = rest.get(index).and_then(Word::literal) {
            if !flag.starts_with('-') {
                break;
            }
            if flag == "-C" {
                if let Some(word) = rest.get(index + 1) {
                    let resolved = self.resolve(ShellDialect::Posix, scope, word);
                    directory = if resolved.unknown {
                        None
                    } else {
                        AbsPath::parse(self.host.style, &resolved.text, directory.as_ref())
                    };
                }
                index += 2;
            } else if matches!(
                flag.as_str(),
                "-c" | "--git-dir" | "--work-tree" | "--namespace"
            ) {
                index += 2;
            } else {
                index += 1;
            }
        }
        if rest.get(index).and_then(Word::literal).as_deref() != Some("clean") {
            return;
        }
        let args = &rest[index + 1..];
        let force = args.iter().filter_map(Word::literal).any(|flag| {
            flag == "--force"
                || (flag.starts_with('-') && !flag.starts_with("--") && flag.contains('f'))
        });
        if !force {
            return;
        }
        let scope = scope.subshell(directory.clone());
        let paths = posix_operands(args, &["-e", "--exclude"]);
        if paths.is_empty() {
            let place = Located {
                path: directory,
                span: Span::Contents,
                speculative: false,
            };
            self.record(&place, Op::Delete, program, ".", None);
        }
        self.targets(ShellDialect::Posix, &scope, &paths, Op::Delete, program);
    }

    fn rsync(&mut self, program: &str, rest: &[Word], scope: &Scope) {
        const VALUED: &[&str] = &[
            "-e",
            "--rsh",
            "--exclude",
            "--include",
            "--filter",
            "-f",
            "--files-from",
            "--exclude-from",
            "--include-from",
            "--backup-dir",
            "--suffix",
            "--partial-dir",
            "--temp-dir",
            "-T",
            "--compare-dest",
            "--copy-dest",
            "--link-dest",
            "--chmod",
            "--chown",
            "--log-file",
            "--password-file",
            "--port",
            "--timeout",
            "--bwlimit",
            "-B",
            "--block-size",
            "--max-size",
            "--min-size",
            "--max-delete",
        ];
        let literals: Vec<String> = rest.iter().filter_map(Word::literal).collect();
        let deletes = literals.iter().any(|flag| flag.starts_with("--del"));
        let removes_sources = literals.iter().any(|flag| flag == "--remove-source-files");
        let mut operands: Vec<&Word> = posix_operands(rest, VALUED)
            .into_iter()
            .filter(|word| !word.literal().is_some_and(|text| is_remote(&text)))
            .collect();
        let Some(destination) = operands.pop() else {
            return;
        };
        if deletes {
            self.target(ShellDialect::Posix, scope, destination, Op::Delete, program);
        }
        if removes_sources {
            self.targets(ShellDialect::Posix, scope, &operands, Op::Move, program);
        }
    }

    // --------------------------------------------------------- cmd commands

    fn cmd(
        &mut self,
        name: &str,
        program: &str,
        rest: &[Word],
        _command: &Command,
        scope: &mut Scope,
        depth: usize,
    ) {
        const DIALECT: ShellDialect = ShellDialect::Cmd;
        let switches: Vec<String> = rest
            .iter()
            .filter_map(Word::literal)
            .filter(|word| word.starts_with('/'))
            .map(|word| word.to_ascii_lowercase())
            .collect();
        let has = |switch: &str| {
            switches
                .iter()
                .any(|s| s == switch || s.starts_with(&format!("{switch}:")))
        };
        match name {
            "cd" | "chdir" | "pushd" => self.change_directory(DIALECT, scope, rest),
            "popd" => scope.cwd = None,
            "set" => self.cmd_set(scope, rest),
            "rd" | "rmdir" | "del" | "erase" => {
                self.targets(DIALECT, scope, &cmd_operands(rest), Op::Delete, program);
            }
            "move" => {
                let mut operands = cmd_operands(rest);
                operands.pop();
                self.targets(DIALECT, scope, &operands, Op::Move, program);
            }
            "format" => self.targets(DIALECT, scope, &cmd_operands(rest), Op::Disk, program),
            "diskpart" => self.disk(program, "a disk"),
            "robocopy" if has("/mir") || has("/purge") => {
                if let Some(destination) = cmd_operands(rest).get(1) {
                    self.target(DIALECT, scope, destination, Op::Delete, program);
                }
            }
            "icacls" if has("/t") => {
                if let Some(path) = cmd_operands(rest).first() {
                    self.target(
                        DIALECT,
                        scope,
                        path,
                        Op::Permissions { recursive: true },
                        program,
                    );
                }
            }
            "takeown" if has("/r") => {
                let path = rest.iter().position(|word| {
                    word.literal()
                        .is_some_and(|flag| flag.eq_ignore_ascii_case("/f"))
                });
                if let Some(word) = path.and_then(|index| rest.get(index + 1)) {
                    self.target(
                        DIALECT,
                        scope,
                        word,
                        Op::Permissions { recursive: true },
                        program,
                    );
                }
            }
            "rm" | "unlink" | "shred" => {
                self.targets(
                    DIALECT,
                    scope,
                    &posix_operands(rest, &[]),
                    Op::Delete,
                    program,
                );
            }
            "cmd" => self.cmd_shell(DIALECT, rest, scope, depth),
            "powershell" | "pwsh" => self.powershell_shell(DIALECT, name, rest, scope, depth),
            "bash" | "sh" => self.posix_shell(DIALECT, rest, scope, depth),
            _ => {}
        }
    }

    fn cmd_set(&self, scope: &mut Scope, rest: &[Word]) {
        let Some(first) = rest.first() else {
            return;
        };
        if first.literal().is_some_and(|flag| flag.starts_with('/')) {
            if let Some(name) = rest
                .get(1)
                .and_then(Word::literal)
                .and_then(|text| text.split('=').next().map(str::to_ascii_lowercase))
            {
                scope.vars.insert(name, Value::Unknown);
            }
            return;
        }
        let Some(text) = self.join(ShellDialect::Cmd, scope, rest) else {
            if let Some(Piece::Text(text)) = first.pieces.first()
                && let Some((name, _)) = text.split_once('=')
            {
                scope.vars.insert(name.to_ascii_lowercase(), Value::Unknown);
            }
            return;
        };
        let text = text.trim_matches('"');
        if let Some((name, value)) = text.split_once('=') {
            scope.vars.insert(
                name.trim().to_ascii_lowercase(),
                Value::Known(value.to_string()),
            );
        }
    }

    // --------------------------------------------------- PowerShell commands

    fn powershell(
        &mut self,
        name: &str,
        program: &str,
        rest: &[Word],
        command: &Command,
        scope: &mut Scope,
        depth: usize,
    ) {
        const DIALECT: ShellDialect = ShellDialect::PowerShell;
        let arguments = powershell_arguments(rest);
        match name {
            "set-location" | "sl" | "cd" | "chdir" | "push-location" | "pushd" => {
                self.change_directory(DIALECT, scope, rest)
            }
            "pop-location" | "popd" => scope.cwd = None,
            "remove-item" | "ri" | "rm" | "rmdir" | "rd" | "del" | "erase" => {
                if arguments.paths.is_empty() {
                    if command.piped || scope.listings.contains_key(&command.pipeline) {
                        self.pipeline_input(scope, Op::Delete, program);
                    }
                } else {
                    self.targets(DIALECT, scope, &arguments.paths, Op::Delete, program);
                }
            }
            "move-item" | "mi" | "move" | "mv" => {
                let sources = arguments.named.first().or(arguments.positional.first());
                if let Some(source) = sources {
                    self.targets(DIALECT, scope, &[*source], Op::Move, program);
                } else if command.piped {
                    self.pipeline_input(scope, Op::Move, program);
                }
            }
            "clear-content" | "clc" => {
                self.targets(DIALECT, scope, &arguments.paths, Op::Overwrite, program);
            }
            "set-content" | "sc" | "out-file" => {
                let path = arguments.named.first().or(arguments.positional.first());
                if let Some(path) = path {
                    self.targets(DIALECT, scope, &[*path], Op::Overwrite, program);
                }
            }
            "format-volume" | "clear-disk" | "initialize-disk" | "remove-partition" => {
                self.disk(program, "a disk");
            }
            "invoke-expression" | "iex" => {
                let resolved = arguments
                    .positional
                    .first()
                    .map(|word| self.resolve(DIALECT, scope, word))
                    .filter(|resolved| !resolved.unknown);
                match resolved {
                    Some(resolved) => self.run(DIALECT, &resolved.text, scope, depth + 1),
                    None => self.unchecked("`Invoke-Expression`"),
                }
            }
            "foreach" | "for" => {
                for word in rest {
                    for piece in &word.pieces {
                        if let Piece::Group(raw) = piece {
                            for variable in raw.split('$').skip(1) {
                                let name: String = variable
                                    .chars()
                                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                                    .collect();
                                scope.vars.insert(name.to_ascii_lowercase(), Value::Unknown);
                            }
                        }
                    }
                }
            }
            "cmd" => self.cmd_shell(DIALECT, rest, scope, depth),
            "powershell" | "pwsh" => self.powershell_shell(DIALECT, name, rest, scope, depth),
            "bash" | "sh" => self.posix_shell(DIALECT, rest, scope, depth),
            _ => {}
        }
    }
}

// ------------------------------------------------------------- word helpers

/// The command a word names: its file name, lowercase, without `.exe`.
fn command_name(word: &Word) -> Option<String> {
    let text = word.literal()?;
    let name = base_name(&text);
    (!name.is_empty()).then_some(name)
}

fn base_name(text: &str) -> String {
    let name = text
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(text)
        .to_ascii_lowercase();
    for extension in [".exe", ".com", ".cmd", ".bat"] {
        if let Some(stem) = name.strip_suffix(extension) {
            return stem.to_string();
        }
    }
    name
}

fn literal_word(text: &str) -> Word {
    Word {
        pieces: vec![Piece::Text(text.to_string())],
        quoted: false,
    }
}

fn variable_name(dialect: ShellDialect, name: &str) -> String {
    match dialect {
        ShellDialect::Posix => format!("`${name}`"),
        ShellDialect::Cmd => format!("`%{name}%`"),
        ShellDialect::PowerShell => match name.strip_prefix("env:") {
            Some(variable) => format!("`$env:{variable}`"),
            None => format!("`${name}`"),
        },
    }
}

/// The word without `prefix` when its text starts with it.
fn strip_word_prefix(word: &Word, prefix: &str) -> Option<Word> {
    let Some(Piece::Text(text)) = word.pieces.first() else {
        return None;
    };
    let rest = text.strip_prefix(prefix)?;
    let mut pieces = vec![Piece::Text(rest.to_string())];
    pieces.extend(word.pieces[1..].iter().cloned());
    Some(Word {
        pieces,
        quoted: word.quoted,
    })
}

/// `NAME=value`: the name and the value (`None` for `NAME+=value`).
fn assignment(word: &Word) -> Option<(String, Option<Word>)> {
    let Some(Piece::Text(text)) = word.pieces.first() else {
        return None;
    };
    let end = text.find('=')?;
    let (name, append) = match text[..end].strip_suffix('+') {
        Some(name) => (name, true),
        None => (&text[..end], false),
    };
    let valid = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return None;
    }
    let value = (!append)
        .then(|| strip_word_prefix(word, &text[..=end]))
        .flatten();
    Some((name.to_string(), value))
}

fn is_pipeline_item(word: &Word) -> bool {
    matches!(word.pieces.as_slice(), [Piece::Var(name, VarForm::Plain)] if name == "_" || name == "psitem")
}

fn is_remote(text: &str) -> bool {
    let colon = text.find(':');
    let slash = text.find('/');
    colon.is_some_and(|colon| colon > 1 && slash.is_none_or(|slash| colon < slash))
}

/// git's variables that point it at another repository, work tree, index or
/// object store than the directory's.
const REPOSITORY_VARIABLES: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
];

fn names_repository(name: &str) -> bool {
    REPOSITORY_VARIABLES.contains(&name)
}

/// Whether a wrapper's option sets one of git's repository variables:
/// `env GIT_DIR=…`, `sudo GIT_DIR=…`, `strace -E GIT_DIR=…`.
fn sets_repository(option: &str) -> bool {
    let option = option
        .strip_prefix("--env=")
        .or_else(|| option.strip_prefix("-E"))
        .unwrap_or(option);
    option
        .split_once('=')
        .is_some_and(|(name, _)| names_repository(name))
}

/// How many of `rest`'s first words are options (and with `assignments`,
/// `NAME=value` words): `valued` options take the next word as their value.
fn leading_options(rest: &[Word], valued: &[&str], assignments: bool) -> usize {
    let mut index = 0;
    while let Some(word) = rest.get(index) {
        let Some(text) = word.literal() else {
            break;
        };
        if assignments && assignment(word).is_some() {
            index += 1;
        } else if text.starts_with('-') && text.len() > 1 {
            index += if valued.contains(&text.as_str()) {
                2
            } else {
                1
            };
        } else {
            break;
        }
    }
    index
}

/// `flock`'s options that take a value.
const FLOCK_VALUED: &[&str] = &["-w", "--timeout", "-E", "--conflict-exit-code"];

/// `watch`'s options that take a value.
const WATCH_VALUED: &[&str] = &["-n", "--interval", "-q", "--equexit"];

/// The text a program runs through a shell from its `-c` option or one of
/// its `long` options: `su -c 'git push'`, `script -qc make log`,
/// `su --command='git push'`.
fn command_option(rest: &[Word], long: &[&str]) -> Option<Word> {
    let mut words = rest.iter();
    while let Some(word) = words.next() {
        let Some(Piece::Text(text)) = word.pieces.first() else {
            continue;
        };
        // `-c`, short options ending with it (`-qc`), or a long option: the
        // text is the next word.
        let short = text.len() > 1
            && text.starts_with('-')
            && text[1..].chars().all(|c| c.is_ascii_alphabetic());
        let whole = word.pieces.len() == 1;
        if whole && ((short && text.ends_with('c')) || long.contains(&text.as_str())) {
            return words.next().cloned();
        }
        // `--command=…`, `-c…`.
        if let Some(value) = long
            .iter()
            .find_map(|option| strip_word_prefix(word, &format!("{option}=")))
        {
            return Some(value);
        }
        if text.starts_with("-c") {
            return strip_word_prefix(word, "-c");
        }
    }
    None
}

/// Words that name what `name` runs, if `name` only prefixes another command
/// (`sudo`, `env`, `then`): how many following words belong to it.
fn skip_prefix(dialect: ShellDialect, name: &str, rest: &[Word]) -> Option<usize> {
    let options =
        |valued: &[&str], skip_assignments: bool| leading_options(rest, valued, skip_assignments);
    match dialect {
        ShellDialect::Posix => match name {
            "if" | "then" | "else" | "elif" | "fi" | "do" | "done" | "while" | "until" | "!"
            | "{" | "}" | "nohup" | "builtin" | "noglob" | "nocorrect" | "unbuffer"
            | "caffeinate" => Some(0),
            // sudo, like env, takes `NAME=value` assignments for the command.
            "sudo" => Some(options(
                &[
                    "-u",
                    "-g",
                    "-h",
                    "-p",
                    "-U",
                    "-C",
                    "-D",
                    "-r",
                    "-t",
                    "-T",
                    "-R",
                    "--user",
                    "--group",
                    "--host",
                    "--prompt",
                    "--other-user",
                    "--close-from",
                    "--chdir",
                    "--role",
                    "--type",
                    "--command-timeout",
                    "--chroot",
                ],
                true,
            )),
            "doas" => Some(options(&["-u", "-C"], false)),
            "env" => Some(options(&["-u", "-C", "-S", "--unset", "--chdir"], true)),
            "nice" => Some(options(&["-n", "--adjustment"], false)),
            "ionice" => Some(options(&["-c", "-n", "-p", "-t"], false)),
            "time" | "command" | "setsid" | "chronic" => Some(options(&[], false)),
            "exec" => Some(options(&["-a"], false)),
            "stdbuf" => Some(options(&["-i", "-o", "-e"], false)),
            "timeout" => Some(options(&["-s", "-k", "--signal", "--kill-after"], false) + 1),
            // `flock <lock> <command>`. After the lock, `-c <text>` runs the
            // text through a shell instead (checked as nested shell text).
            "flock" => {
                let lock = options(FLOCK_VALUED, false);
                match rest.get(lock + 1).and_then(Word::literal).as_deref() {
                    Some("-c" | "--command") => None,
                    _ => Some(lock + 1),
                }
            }
            // `taskset <mask> <command>` (`-c`: a list for the mask); with
            // `-p` it changes a running process instead.
            "taskset" => {
                let skipped = options(&[], false);
                let process = rest[..skipped.min(rest.len())]
                    .iter()
                    .filter_map(Word::literal)
                    .any(|option| {
                        option == "--pid" || (!option.starts_with("--") && option.contains('p'))
                    });
                (!process).then_some(skipped + 1)
            }
            "unshare" => Some(options(
                &[
                    "-l",
                    "--load-interp",
                    "--propagation",
                    "-R",
                    "--root",
                    "-w",
                    "--wd",
                    "-S",
                    "--setuid",
                    "-G",
                    "--setgid",
                    "--map-user",
                    "--map-group",
                    "--map-users",
                    "--map-groups",
                    "--owner",
                    "--setgroups",
                    "--monotonic",
                    "--boottime",
                ],
                false,
            )),
            "strace" => Some(options(
                &[
                    "-a",
                    "-b",
                    "-e",
                    "-E",
                    "-I",
                    "-o",
                    "-O",
                    "-p",
                    "-P",
                    "-s",
                    "-S",
                    "-u",
                    "-U",
                    "-X",
                    "--attach",
                    "--argv0",
                    "--columns",
                    "--const-print-style",
                    "--detach-on",
                    "--env",
                    "--inject",
                    "--interruptible",
                    "--output",
                    "--signal",
                    "--status",
                    "--string-limit",
                    "--summary-columns",
                    "--summary-sort-by",
                    "--summary-syscall-overhead",
                    "--syscall-limit",
                    "--trace",
                    "--trace-path",
                    "--user",
                ],
                false,
            )),
            "ltrace" => Some(options(
                &[
                    "-a",
                    "-A",
                    "-D",
                    "-e",
                    "-F",
                    "-l",
                    "-n",
                    "-o",
                    "-p",
                    "-s",
                    "-u",
                    "-w",
                    "-x",
                    "--align",
                    "--config",
                    "--debug",
                    "--indent",
                    "--library",
                    "--output",
                    "--where",
                ],
                false,
            )),
            // `watch -x` runs its arguments; without `-x`, their text
            // through `sh -c` (checked as nested shell text).
            "watch" => {
                let skipped = options(WATCH_VALUED, false);
                let exec = rest[..skipped.min(rest.len())]
                    .iter()
                    .filter_map(Word::literal)
                    .any(|option| {
                        option == "--exec" || (!option.starts_with("--") && option.contains('x'))
                    });
                exec.then_some(skipped)
            }
            _ => None,
        },
        ShellDialect::Cmd => match name {
            "else" | "do" | "call" => Some(0),
            "if" => {
                let mut index = 0;
                let flag = |index: usize| {
                    rest.get(index)
                        .and_then(Word::literal)
                        .map(|text| text.to_ascii_lowercase())
                };
                while matches!(flag(index).as_deref(), Some("/i" | "not")) {
                    index += 1;
                }
                index += match flag(index).as_deref() {
                    Some("exist" | "defined" | "errorlevel" | "cmdextversion") => 2,
                    _ if matches!(
                        flag(index + 1).as_deref(),
                        Some("equ" | "neq" | "lss" | "leq" | "gtr" | "geq" | "==")
                    ) =>
                    {
                        3
                    }
                    _ => 1,
                };
                Some(index)
            }
            "start" => {
                let mut index = 0;
                if rest.first().is_some_and(|word| word.quoted) {
                    index += 1;
                }
                while rest
                    .get(index)
                    .and_then(Word::literal)
                    .is_some_and(|text| text.starts_with('/'))
                {
                    index += 1;
                }
                Some(index)
            }
            _ => None,
        },
        ShellDialect::PowerShell => match name {
            "sudo" => Some(0),
            _ => None,
        },
    }
}

/// Whether `command` runs a script the guard can't read in the current
/// shell (`source env.sh`, `call setup.cmd`, `. .\env.ps1`).
fn sources_script(dialect: ShellDialect, command: &Command) -> bool {
    let name = command
        .words
        .first()
        .and_then(Word::literal)
        .map(|n| n.to_ascii_lowercase());
    matches!(
        (dialect, name.as_deref()),
        (ShellDialect::Posix, Some("source" | "."))
            | (ShellDialect::Cmd, Some("call"))
            | (ShellDialect::PowerShell, Some(".")),
    )
}

/// Operands of a POSIX command: words after its options. `valued` options
/// take the next word as their value.
fn posix_operands<'w>(rest: &'w [Word], valued: &[&str]) -> Vec<&'w Word> {
    let mut operands = Vec::new();
    let mut options_done = false;
    let mut index = 0;
    while let Some(word) = rest.get(index) {
        index += 1;
        if !options_done && let Some(text) = word.literal() {
            if text == "--" {
                options_done = true;
                continue;
            }
            if text.starts_with('-') && text.len() > 1 {
                if valued.contains(&text.as_str()) {
                    index += 1;
                }
                continue;
            }
        }
        operands.push(word);
    }
    operands
}

fn cmd_operands(rest: &[Word]) -> Vec<&Word> {
    rest.iter()
        .filter(|word| {
            !word.literal().is_some_and(|text| {
                text.starts_with('/') && text.len() <= 12 && !text[1..].contains(['/', '\\'])
            })
        })
        .collect()
}

/// `find` start points: the words before the first expression.
fn find_paths(rest: &[Word]) -> Vec<&Word> {
    rest.iter()
        .take_while(|word| {
            !word
                .literal()
                .is_some_and(|text| text.starts_with('-') || text == "(" || text == "!")
        })
        .collect()
}

struct PowerShellArguments<'w> {
    /// `-Path`, `-LiteralPath`, and `-FilePath` values, then positional words.
    paths: Vec<&'w Word>,
    named: Vec<&'w Word>,
    positional: Vec<&'w Word>,
}

/// What a PowerShell path that names a provider or a drive of more than one
/// letter points to.
#[derive(Debug, PartialEq)]
enum ProviderPath {
    /// A file path, as the text to check.
    File(String),
    /// Not a file, such as an environment variable, a variable, a function
    /// or an alias (`Env:\X`, `Variable:\X`, `Function:\X`, `Alias:\X`).
    NotFile,
    /// Outside the working directory by nature, such as the registry, the
    /// certificate store, WS-Management, or a drive the session defines,
    /// whose place isn't known.
    Elsewhere,
}

/// What `text` names in PowerShell when it starts with a provider
/// (`Registry::HKEY_CURRENT_USER\…`, `Microsoft.PowerShell.Core\FileSystem::C:\…`)
/// or a drive whose name is longer than one letter (`Env:`, `HKCU:`,
/// `Cert:`, `Temp:`). `None` for an ordinary path. `temp` is the temporary
/// directory, where PowerShell's `Temp:` drive points.
fn powershell_provider(text: &str, temp: Option<&str>) -> Option<ProviderPath> {
    let is_name = |name: &str| {
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    if let Some((qualifier, rest)) = text.split_once("::") {
        let provider = match qualifier.rsplit_once('\\') {
            // A module-qualified provider, as Get-Item shows it.
            Some((module, provider))
                if module
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_') =>
            {
                provider
            }
            Some(_) => return None,
            None => qualifier,
        };
        if !is_name(provider) {
            return None;
        }
        return Some(match provider.to_ascii_lowercase().as_str() {
            "filesystem" => ProviderPath::File(rest.to_string()),
            "environment" | "variable" | "function" | "alias" => ProviderPath::NotFile,
            _ => ProviderPath::Elsewhere,
        });
    }
    let (drive, rest) = text.split_once(':')?;
    if drive.len() < 2 || !is_name(drive) || !drive.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    Some(match drive.to_ascii_lowercase().as_str() {
        "env" | "variable" | "function" | "alias" => ProviderPath::NotFile,
        "temp" => match temp {
            Some(temp) => {
                let rest = rest.trim_start_matches(['\\', '/']);
                let separator = if temp.ends_with(['\\', '/']) {
                    ""
                } else if temp.starts_with('/') {
                    "/"
                } else {
                    "\\"
                };
                ProviderPath::File(format!("{temp}{separator}{rest}"))
            }
            None => ProviderPath::Elsewhere,
        },
        _ => ProviderPath::Elsewhere,
    })
}

fn powershell_arguments(rest: &[Word]) -> PowerShellArguments<'_> {
    const PATHS: &[&str] = &["path", "literalpath", "filepath", "pspath", "lp"];
    const VALUED: &[&str] = &[
        "include",
        "exclude",
        "filter",
        "credential",
        "stream",
        "destination",
        "value",
        "encoding",
        "inputobject",
        "width",
        "delimiter",
        "depth",
        "attributes",
        "itemtype",
        "name",
        "stackname",
    ];
    const SWITCHES: &[&str] = &[
        "recurse",
        "force",
        "whatif",
        "confirm",
        "verbose",
        "debug",
        "passthru",
        "nonewline",
        "append",
        "noclobber",
        "directory",
        "file",
        "hidden",
        "readonly",
        "system",
    ];
    let mut named = Vec::new();
    let mut positional = Vec::new();
    let mut index = 0;
    while let Some(word) = rest.get(index) {
        index += 1;
        let parameter = word
            .literal()
            .filter(|text| text.starts_with('-') && text.len() > 1 && !word.quoted)
            .map(|text| {
                text[1..]
                    .split(':')
                    .next()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            });
        let Some(parameter) = parameter else {
            positional.push(word);
            continue;
        };
        let matches = |names: &[&str]| {
            names
                .iter()
                .any(|name| name.starts_with(parameter.as_str()))
        };
        if PATHS.contains(&parameter.as_str())
            || (matches(PATHS) && !matches(SWITCHES) && !matches(VALUED))
        {
            if let Some(value) = rest.get(index) {
                named.push(value);
                index += 1;
            }
        } else if VALUED.contains(&parameter.as_str()) || (matches(VALUED) && !matches(SWITCHES)) {
            index += 1;
        }
    }
    let paths = named.iter().chain(&positional).copied().collect();
    PowerShellArguments {
        paths,
        named,
        positional,
    }
}

/// PowerShell's `-EncodedCommand`: base64 of UTF-16LE text.
fn decode_utf16_base64(text: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0;
    for c in text.chars().filter(|c| !c.is_whitespace() && *c != '=') {
        let value = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' | '-' => 62,
            '/' | '_' => 63,
            _ => return None,
        };
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    String::from_utf16(&units).ok()
}
