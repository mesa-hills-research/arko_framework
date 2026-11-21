//! Settings for agents whose models run through the installed Antigravity CLI.
//! Wrap a base model with [`arko_ai::Model::antigravity`], or look it up
//! with [`arko_ai::Provider::Antigravity`]. Run `agy` once to sign in.
//!
//! Arko supplies the prompt and tools through a private profile. Set
//! [`AntigravitySettings::helper_command`] to a program and arguments that
//! call the [`arko_ai::antigravity`] relay and hook helpers.
//!
//! Antigravity 1.5.14 still exposes `list_resources`, `read_resource` and
//! `manage_task`. Arko’s MCP relay offers no resources, and this setup
//! starts no Antigravity tasks for `manage_task` to manage.
//!
//! Usage is billed to the Google plan. A resumed task starts a fresh
//! CLI session with the saved transcript.

use std::collections::BTreeMap;
use std::path::PathBuf;

use arko_ai::antigravity::HelperCommand;
use serde::{Deserialize, Serialize};

/// How the agent starts a private Antigravity session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AntigravitySettings {
    /// The CLI to run. When unset, look on `PATH` and then in `~/.local/bin`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cli_path: Option<PathBuf>,
    /// The host's command for the MCP relay and fallback hooks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helper_command: Option<HelperCommand>,
    /// The parent directory for private profiles. Defaults to the system temporary directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_root: Option<PathBuf>,
    /// Additional environment variables for the CLI.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// The requested thought level. None leaves the model's default in place.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_level: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_settings_take_the_defaults() {
        let settings: AntigravitySettings = serde_json::from_str("{}").unwrap();
        assert_eq!(settings, AntigravitySettings::default());
        assert_eq!(serde_json::to_string(&settings).unwrap(), "{}");
    }
}
