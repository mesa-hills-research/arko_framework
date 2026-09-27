use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptRule {
    pub name: String,
    pub pattern: String,
}

pub(super) struct Detector {
    rules: Vec<(String, Regex)>,
    cursor_forward: Regex,
    escapes: Regex,
}
impl Detector {
    pub fn new(rules: &[PromptRule]) -> Result<Self, regex::Error> {
        Ok(Self {
            rules: rules
                .iter()
                .map(|rule| Ok((rule.name.clone(), Regex::new(&rule.pattern)?)))
                .collect::<Result<_, regex::Error>>()?,
            cursor_forward: Regex::new(r"\x1b\[(\d{0,4})C")?,
            escapes: Regex::new(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")?,
        })
    }
    pub fn detect(&self, output: &[u8]) -> Option<(String, String)> {
        let text = String::from_utf8_lossy(output);
        // ConPTY renders runs of spaces as cursor-forward (CSI n C). Restore
        // them so rules match trailing spaces as other platforms print them.
        let text = self.cursor_forward.replace_all(&text, |c: &regex::Captures| {
            " ".repeat(c[1].parse::<usize>().unwrap_or(1).clamp(1, 256))
        });
        let text = self.escapes.replace_all(&text, "");
        let text = text.trim_end_matches(['\r', '\n']);
        let line = text.rsplit(['\n', '\r']).next().unwrap_or("");
        self.rules
            .iter()
            .find(|(_, rule)| rule.is_match(line))
            .map(|(name, _)| (name.clone(), line.to_string()))
    }
}

pub(super) fn default_rules() -> Vec<PromptRule> {
    vec![
        PromptRule {
            name: "yes_no".into(),
            pattern: r"(?i)(?:\[y/n\]|\(y/n\)|\[yes/no\]|\(yes/no\))\s*[:?]?\s*$".into(),
        },
        PromptRule {
            name: "press_key".into(),
            pattern: r"(?i)press (?:enter|return|any key)(?: to [^\r\n]{0,100})?[.: ]*$".into(),
        },
        PromptRule {
            name: "password".into(),
            pattern: r"(?i)(?:password|passphrase)(?: for [^\r\n]{0,100})?:\s*$".into(),
        },
    ]
}

/// This side channel carries application declarations, never inferred from stdout.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Declaration {
    InputRequest { request_id: String, prompt: String },
    InputCancelled { request_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conpty_cursor_forward_counts_as_trailing_space() {
        let rule = PromptRule {
            name: "region".into(),
            pattern: "Pick a region: $".into(),
        };
        let detector = Detector::new(&[rule]).unwrap();
        // Captured from a Windows CI run: the prompt's trailing space arrives as CSI 1 C.
        let output = b"\x1b[?25l\x1b[2J\x1b[m\x1b[HPick a region:\x1b[1C\x1b]0;C:\\Windows\\SYSTEM32\\cmd.exe\x07\x1b[?25h";
        assert_eq!(
            detector.detect(output),
            Some(("region".into(), "Pick a region: ".into()))
        );
        assert_eq!(
            detector.detect(b"Pick a region:\x1b[C").map(|(_, line)| line),
            Some("Pick a region: ".into())
        );
        assert_eq!(detector.detect(b"Pick a region:\x1b[1K"), None);
    }
}
