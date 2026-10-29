/// Whether a skill's `models` patterns match the names of a model.
///
/// A pattern is matched against the whole of a name, ignoring case, where `*`
/// matches any run of characters (none included) and `?` matches one
/// character. A pattern with neither must equal the name. No `patterns` is a
/// skill for every model, so it matches any names, none included.
///
/// A model goes by names of three kinds, and a pattern is matched only
/// against names of its own kind.
/// - A pattern that starts with `cc:` (or `claude-code:`, read as `cc:`) is
///   matched against names that start so, the names of a model run through
///   Claude Code.
/// - A pattern that starts with `openrouter:` is matched against names that
///   start so, the model's slug on OpenRouter.
/// - Any other pattern is matched against the plain names, which have neither
///   prefix. So `*haiku*` never matches `cc:haiku` or an OpenRouter slug.
///
/// ```
/// use arko_agents::skill::models_match;
///
/// let names = ["claude-haiku-5-5", "haiku", "cc:haiku"];
/// assert!(models_match(&["Haiku"], &names));
/// assert!(models_match(&["claude-haiku-*", "sol"], &names));
/// assert!(models_match(&["cc:*"], &names));
/// assert!(!models_match(&["opus"], &names));
/// assert!(!models_match(&["openrouter:*"], &names));
/// assert!(models_match::<&str, &str>(&[], &names));
/// ```
pub fn models_match<P, N>(patterns: &[P], names: &[N]) -> bool
where
    P: AsRef<str>,
    N: AsRef<str>,
{
    if patterns.is_empty() {
        return true;
    }
    let names: Vec<(Kind, Vec<char>)> = names.iter().map(|name| classify(name.as_ref())).collect();
    patterns.iter().any(|pattern| {
        let (kind, pattern) = classify(pattern.as_ref());
        names
            .iter()
            .any(|(name_kind, name)| *name_kind == kind && glob(&pattern, name))
    })
}

/// The kinds of names a model goes by.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Plain,
    ClaudeCode,
    OpenRouter,
}

/// The kind of a name or pattern, and its characters in lower case, with
/// `claude-code:` read as `cc:`.
fn classify(text: &str) -> (Kind, Vec<char>) {
    let text = text.trim().to_lowercase();
    let text = match text.strip_prefix("claude-code:") {
        Some(rest) => format!("cc:{rest}"),
        None => text,
    };
    let kind = if text.starts_with("cc:") {
        Kind::ClaudeCode
    } else if text.starts_with("openrouter:") {
        Kind::OpenRouter
    } else {
        Kind::Plain
    };
    (kind, text.chars().collect())
}

/// Whether `pattern` matches the whole of `text`, where `*` is any run of
/// characters and `?` one.
fn glob(pattern: &[char], text: &[char]) -> bool {
    let (mut p, mut t) = (0, 0);
    // Where the last `*` was in the pattern, and how much text it has taken.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        match pattern.get(p) {
            Some('*') => {
                star = Some((p, t));
                p += 1;
            }
            Some(c) if *c == '?' || *c == text[t] => {
                p += 1;
                t += 1;
            }
            _ => match star {
                Some((star_p, star_t)) => {
                    // Let the star take one more character.
                    star = Some((star_p, star_t + 1));
                    p = star_p + 1;
                    t = star_t + 1;
                }
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(patterns: &[&str], names: &[&str]) -> bool {
        models_match(patterns, names)
    }

    #[test]
    fn a_pattern_matches_the_whole_name_ignoring_case() {
        let names = ["claude-haiku-5-5", "haiku"];
        assert!(matches(&["haiku"], &names));
        assert!(matches(&["HAIKU"], &names));
        assert!(matches(&["Claude-Haiku-5-5"], &names));
        // Never a part of a name.
        assert!(!matches(&["hai"], &names));
        assert!(!matches(&["claude-haiku"], &names));
        assert!(!matches(&["aiku"], &names));
        assert!(!matches(&["opus"], &names));
    }

    #[test]
    fn stars_and_question_marks_match_runs_and_single_characters() {
        let names = ["gpt-6.1-sol"];
        assert!(matches(&["gpt-6*"], &names));
        assert!(matches(&["*sol"], &names));
        assert!(matches(&["*"], &names));
        assert!(matches(&["g*t-*l"], &names));
        assert!(matches(&["gpt-6.?-sol"], &names));
        assert!(matches(&["gpt-6.1-sol*"], &names));
        assert!(matches(&["*gpt*sol*"], &names));
        assert!(!matches(&["gpt-6.??-sol"], &names));
        assert!(!matches(&["gpt-6?"], &names));
        assert!(!matches(&["gpt-7*"], &names));
        assert!(!matches(&["*luna"], &names));
        // Backtracking over a star that first takes too little.
        assert!(matches(&["*a*b"], &["xaabb"]));
        assert!(matches(&["*ab*ab"], &["abab"]));
        assert!(matches(&["a*a*a"], &["aaaa"]));
        assert!(!matches(&["a*a*a"], &["aa"]));
        // `?` is one character, whatever its width in bytes.
        assert!(matches(&["h?iku"], &["häiku"]));
        assert!(matches(&["HÄIKU"], &["häiku"]));
    }

    #[test]
    fn any_pattern_and_any_name_will_do() {
        let names = ["claude-haiku-5-5", "haiku"];
        assert!(matches(&["opus", "haiku"], &names));
        assert!(matches(&["opus", "claude-haiku-*"], &names));
        assert!(!matches(&["opus", "sol"], &names));
        assert!(!matches(&["haiku"], &["opus", "claude-opus-5-5"]));
    }

    #[test]
    fn no_patterns_is_every_model_and_no_names_matches_nothing_else() {
        assert!(matches(&[], &["haiku"]));
        assert!(matches(&[], &[]));
        assert!(!matches(&["haiku"], &[]));
        assert!(!matches(&["*"], &[]));
    }

    #[test]
    fn a_pattern_is_matched_only_against_names_of_its_kind() {
        let names = [
            "claude-haiku-5-5",
            "haiku",
            "cc:haiku",
            "cc:claude-haiku-5-5",
            "openrouter:anthropic/claude-haiku-5.5",
        ];
        // Plain patterns reach plain names only.
        assert!(matches(&["haiku"], &names));
        assert!(!matches(&["haiku"], &names[2..]));
        assert!(!matches(&["*haiku*"], &names[2..]));
        assert!(!matches(&["*"], &names[2..]));
        assert!(!matches(&["*anthropic*"], &names));
        assert!(!matches(&["anthropic/*"], &names));
        // A `cc:` pattern reaches Claude Code names only.
        assert!(matches(&["cc:haiku"], &names));
        assert!(!matches(&["cc:haiku"], &names[..2]));
        assert!(!matches(&["cc:haiku"], &names[4..]));
        assert!(matches(&["cc:*"], &names));
        assert!(matches(&["cc:claude-haiku-*"], &names));
        assert!(!matches(&["cc:*"], &names[..2]));
        // `claude-code:` is read as `cc:`, in a pattern or a name.
        assert!(matches(&["claude-code:haiku"], &names));
        assert!(matches(&["Claude-Code:HAIKU"], &names));
        assert!(matches(&["cc:haiku"], &["claude-code:haiku"]));
        assert!(!matches(&["claude-code:haiku"], &names[..2]));
        // An `openrouter:` pattern reaches the OpenRouter name only.
        assert!(matches(&["openrouter:anthropic/claude-haiku-5.5"], &names));
        assert!(matches(&["openrouter:anthropic/*"], &names));
        assert!(matches(&["openrouter:*"], &names));
        assert!(!matches(&["openrouter:*"], &names[..4]));
        assert!(!matches(
            &["openrouter:anthropic/claude-haiku-5.5:floor"],
            &names
        ));
        assert!(!matches(&["openrouter:openai/gpt-6*"], &names));
    }

    #[test]
    fn a_name_with_a_routing_suffix_is_its_own_name() {
        // Only the full slug reaches a slug that isn't in a table.
        let names = ["openrouter:anthropic/claude-haiku-5.5:floor"];
        assert!(matches(&["openrouter:anthropic/claude-haiku-5.5*"], &names));
        assert!(matches(
            &["openrouter:anthropic/claude-haiku-5.5:floor"],
            &names
        ));
        assert!(!matches(&["openrouter:anthropic/claude-haiku-5.5"], &names));
        assert!(!matches(&["haiku", "claude-haiku-*", "*haiku*"], &names));
    }

    #[test]
    fn owned_strings_work_as_patterns_and_names() {
        let patterns = vec!["haiku".to_string()];
        let names = vec!["haiku".to_string()];
        assert!(models_match(&patterns, &names));
    }
}
