use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptRule {
    pub name: String,
    pub pattern: String,
}

pub(super) struct Detector {
    rules: Vec<(String, Regex)>,
    escapes: Regex,
}
impl Detector {
    pub fn new(rules: &[PromptRule]) -> Result<Self, regex::Error> {
        Ok(Self {
            rules: rules
                .iter()
                .map(|rule| Ok((rule.name.clone(), Regex::new(&rule.pattern)?)))
                .collect::<Result<_, regex::Error>>()?,
            escapes: Regex::new(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")?,
        })
    }
    pub fn detect(&self, output: &[u8]) -> Option<(String, String)> {
        let text = String::from_utf8_lossy(output);
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
