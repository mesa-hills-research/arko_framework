//! Command rules: which programs, with which arguments, the guard allows or
//! refuses. They are matched against each command the guard finds in a
//! line, never against the raw text.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::paths::{AbsPath, PathStyle};
use super::{CommandRefusal, base_name};

/// Whether a [`CommandRule`] allows or refuses the commands it matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleAction {
    /// Allow the commands the rule matches. Once a program has an `Allow`
    /// rule, only the commands of it that an `Allow` rule matches run.
    Allow,
    /// Refuse the commands the rule matches, whatever allows them.
    Deny,
}

/// A rule of the [`CommandGuard`](super::CommandGuard): a pattern of words
/// for the commands it allows or refuses.
///
/// The pattern's first word is the program, compared with the base name of
/// the command's program (`/usr/bin/git` is `git`; case and `.exe` are
/// ignored). The other words match the command's arguments in order: `*`
/// alone matches any number of words, none included; `*` inside a word
/// matches any text in that word (`--output*`); any other word matches
/// itself exactly, after the shell's quoting is removed. An argument only
/// known when the command runs (`$(…)` output) matches only `*`. For git,
/// the global options `-C <dir>`, `--no-pager`, `-P` and
/// `--no-optional-locks` are skipped first, so `git log *` matches
/// `git -C repo log`; other global options (`-c`, `--git-dir`, …) stay.
///
/// ```
/// use arko_agents::terminal::CommandRule;
///
/// let rules = vec![
///     CommandRule::allow("git status *"),
///     CommandRule::allow("git log *"),
///     CommandRule::deny("cargo * publish *").because("Publishing is the user's."),
///     CommandRule::allow("git *").within("scratch"),
/// ];
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandRule {
    pub action: RuleAction,
    /// Whitespace-separated words: the program, then its arguments.
    pub pattern: String,
    /// Only for commands run in this directory or below it: after `cd`, and
    /// for git `-C`. Relative paths are relative to the job's working
    /// directory. A command whose directory isn't known (it changes to a
    /// directory only known when it runs, or git is given `--git-dir` or
    /// `--work-tree`, or one of its location variables such as `GIT_DIR`,
    /// `GIT_WORK_TREE` or `GIT_INDEX_FILE` is set, whatever `-C` follows) is
    /// never within it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<PathBuf>,
    /// Why, for the model: the refusal's message says it instead of a
    /// generic explanation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl CommandRule {
    /// A rule allowing the commands `pattern` matches.
    pub fn allow(pattern: impl Into<String>) -> Self {
        Self::new(RuleAction::Allow, pattern)
    }

    /// A rule refusing the commands `pattern` matches.
    pub fn deny(pattern: impl Into<String>) -> Self {
        Self::new(RuleAction::Deny, pattern)
    }

    fn new(action: RuleAction, pattern: impl Into<String>) -> Self {
        Self {
            action,
            pattern: pattern.into(),
            within: None,
            reason: None,
        }
    }

    /// Applies the rule only to commands run in `directory` or below it.
    pub fn within(mut self, directory: impl Into<PathBuf>) -> Self {
        self.within = Some(directory.into());
        self
    }

    /// The reason a refusal by this rule gives the model.
    pub fn because(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// The guard's rules, ready to match: paths resolved, patterns split.
pub(super) struct Rules<'g> {
    style: PathStyle,
    rules: Vec<Rule<'g>>,
}

struct Rule<'g> {
    rule: &'g CommandRule,
    program: String,
    arguments: Vec<&'g str>,
    /// `None`: anywhere. `Some(None)`: a directory that couldn't be
    /// resolved, which no command is within.
    within: Option<Option<AbsPath>>,
}

impl<'g> Rules<'g> {
    pub fn new(rules: &'g [CommandRule], style: PathStyle, workspace: Option<&AbsPath>) -> Self {
        let rules = rules
            .iter()
            .filter_map(|rule| {
                let mut words = rule.pattern.split_whitespace();
                let program = base_name(words.next()?);
                Some(Rule {
                    rule,
                    program,
                    arguments: words.collect(),
                    within: rule
                        .within
                        .as_ref()
                        .map(|within| AbsPath::parse(style, &within.to_string_lossy(), workspace)),
                })
            })
            .collect();
        Self { style, rules }
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Checks one command: `words` from its program on (`None` for a word
    /// only known when it runs), run in `directory`. `other_repository`:
    /// one of git's location variables (`GIT_DIR`, `GIT_INDEX_FILE`, …) is
    /// set for it, so git works elsewhere.
    pub fn check(
        &self,
        words: &[Option<String>],
        directory: Option<&AbsPath>,
        other_repository: bool,
    ) -> Result<(), CommandRefusal> {
        let Some((Some(program), arguments)) = words.split_first() else {
            return Err(unknown_program(words));
        };
        let program = base_name(program);
        let (arguments, directory) = if program == "git" {
            self.git(arguments, directory.cloned(), other_repository)
        } else {
            (arguments.to_vec(), directory.cloned())
        };
        let applies = |rule: &Rule| match &rule.within {
            None => true,
            Some(within) => within.as_ref().is_some_and(|within| {
                directory
                    .as_ref()
                    .is_some_and(|directory| within.contains(self.style, directory))
            }),
        };
        let matches = |rule: &Rule| words_match(&rule.arguments, &arguments) && applies(rule);
        let of_program: Vec<&Rule> = self
            .rules
            .iter()
            .filter(|rule| glob(&rule.program, &program))
            .collect();
        if let Some(rule) = of_program
            .iter()
            .find(|rule| rule.rule.action == RuleAction::Deny && matches(rule))
        {
            let reason = rule.rule.reason.clone().unwrap_or_else(|| {
                "Do the work another way; other commands need the user's permission in the terminal settings.".into()
            });
            return Err(refusal(
                words,
                format!("matches the command rule `{}`. {reason}", rule.rule.pattern),
            ));
        }
        let allowed: Vec<&&Rule> = of_program
            .iter()
            .filter(|rule| rule.rule.action == RuleAction::Allow)
            .collect();
        if allowed.is_empty() || allowed.iter().any(|rule| matches(rule)) {
            return Ok(());
        }
        let patterns = allowed
            .iter()
            .map(|rule| match &rule.rule.within {
                Some(within) => format!("`{}` (in `{}`)", rule.rule.pattern, within.display()),
                None => format!("`{}`", rule.rule.pattern),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let reason = allowed
            .iter()
            .find_map(|rule| rule.rule.reason.clone())
            .unwrap_or_else(|| {
                "Use one of those, or do the work another way; other commands need the user's permission in the terminal settings.".into()
            });
        Err(refusal(
            words,
            format!("isn't one of the `{program}` commands allowed here: {patterns}. {reason}"),
        ))
    }

    /// git's arguments without the global options rules skip, and the
    /// directory it runs in: none known when it works on another repository
    /// or work tree, whatever `-C` says.
    fn git(
        &self,
        arguments: &[Option<String>],
        mut directory: Option<AbsPath>,
        mut other_repository: bool,
    ) -> (Vec<Option<String>>, Option<AbsPath>) {
        const VALUED: &[&str] = &[
            "-c",
            "--git-dir",
            "--work-tree",
            "--namespace",
            "--config-env",
            "--attr-source",
            "--super-prefix",
        ];
        let mut kept = Vec::new();
        let mut index = 0;
        while let Some(Some(option)) = arguments.get(index) {
            if !option.starts_with('-') {
                break;
            }
            match option.as_str() {
                "-C" => {
                    directory = match arguments.get(index + 1) {
                        Some(Some(path)) => AbsPath::parse(self.style, path, directory.as_ref()),
                        _ => None,
                    };
                    index += 2;
                    continue;
                }
                "--no-pager" | "-P" | "--no-optional-locks" => {
                    index += 1;
                    continue;
                }
                _ => {}
            }
            if option.starts_with("--git-dir") || option.starts_with("--work-tree") {
                other_repository = true;
            }
            let taken = if VALUED.contains(&option.as_str()) {
                2
            } else {
                1
            };
            kept.extend(
                arguments[index..(index + taken).min(arguments.len())]
                    .iter()
                    .cloned(),
            );
            index += taken;
        }
        kept.extend(arguments[index.min(arguments.len())..].iter().cloned());
        (kept, directory.filter(|_| !other_repository))
    }
}

/// A refusal of `words` (shown as the command), for `why`.
fn refusal(words: &[Option<String>], why: String) -> CommandRefusal {
    let command = shown(words);
    CommandRefusal {
        message: format!("Refused: `{command}` {why}"),
        target: command,
        protected: false,
    }
}

fn unknown_program(words: &[Option<String>]) -> CommandRefusal {
    let command = shown(words);
    CommandRefusal {
        message: format!(
            "Refused: `{command}`: its program is only known when the command runs, so the command rules can't check it. Name the program directly."
        ),
        target: command,
        protected: false,
    }
}

/// A refusal of a command whose text the guard can't know in advance:
/// `what` runs it.
pub(super) fn unchecked(what: &str) -> CommandRefusal {
    CommandRefusal {
        message: format!(
            "Refused: the command {what} runs is only known when it runs, so the command rules can't check it. Write the command out directly."
        ),
        target: what.to_string(),
        protected: false,
    }
}

/// A refusal of shells nested deeper than the guard reads.
pub(super) fn too_deep() -> CommandRefusal {
    CommandRefusal {
        message: "Refused: the command nests shells too deeply for the command rules to check it. Write the command out directly.".into(),
        target: String::new(),
        protected: false,
    }
}

/// The command as the guard read it, shortened.
fn shown(words: &[Option<String>]) -> String {
    let text = words
        .iter()
        .map(|word| word.as_deref().unwrap_or("…"))
        .collect::<Vec<_>>()
        .join(" ");
    if text.chars().count() > 200 {
        format!("{}…", text.chars().take(200).collect::<String>())
    } else {
        text
    }
}

/// Whether the pattern `words` match the command's `arguments`.
fn words_match(pattern: &[&str], arguments: &[Option<String>]) -> bool {
    match pattern.split_first() {
        None => arguments.is_empty(),
        Some((&"*", rest)) => {
            (0..=arguments.len()).any(|skip| words_match(rest, &arguments[skip..]))
        }
        Some((word, rest)) => match arguments.split_first() {
            Some((Some(argument), others)) => glob(word, argument) && words_match(rest, others),
            _ => false,
        },
    }
}

/// Whether `text` matches `pattern`, where `*` matches any text.
fn glob(pattern: &str, text: &str) -> bool {
    let Some((first, rest)) = pattern.split_once('*') else {
        return pattern == text;
    };
    let Some(mut text) = text.strip_prefix(first) else {
        return false;
    };
    let mut parts: Vec<&str> = rest.split('*').collect();
    let last = parts.pop().unwrap_or_default();
    for part in parts {
        match text.find(part) {
            Some(at) => text = &text[at + part.len()..],
            None => return false,
        }
    }
    text.len() >= last.len() && text.ends_with(last)
}

#[cfg(test)]
mod tests {
    use super::glob;

    #[test]
    fn globs_match_within_a_word() {
        assert!(glob("--output*", "--output"));
        assert!(glob("--output*", "--output=log.txt"));
        assert!(!glob("--output*", "--outpu"));
        assert!(glob("*", ""));
        assert!(glob("a*b*c", "abc"));
        assert!(glob("a*b*c", "axxbyyc"));
        assert!(!glob("a*b*c", "axxbyy"));
        assert!(!glob("ab*ba", "aba"));
        assert!(glob("git", "git"));
        assert!(!glob("git", "gitk"));
    }
}
