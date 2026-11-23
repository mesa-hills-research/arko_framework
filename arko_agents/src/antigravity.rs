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
//! Usage is billed to the Google plan.
//!
//! With [`AntigravitySettings::profile_root`] set, a task keeps its profile,
//! which holds the CLI's copy of the conversation, and its next run resumes
//! the session ([`AntigravitySession`]), also in another process. A new
//! session starts from a record of the conversation when there is no root,
//! when the system prompt changed, after a handover, a cut back or a failed
//! turn, and when the CLI refuses to resume.

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
    /// The parent directory of the tasks' private profiles, each kept in a
    /// folder named for its task until a new session replaces it. The host
    /// deletes them with the tasks. Without one, each run writes its profile
    /// in the system temporary directory, removes it when it ends, and starts
    /// a new session from a record of the conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_root: Option<PathBuf>,
    /// Additional environment variables for the CLI.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// The requested thought level. None leaves the model's default in place.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_level: Option<String>,
}

/// The Antigravity session holding a task's conversation, which the task's
/// next run resumes in the task's profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AntigravitySession {
    /// The session's ACP ID.
    pub id: String,
    /// The digest of the system prompt it started with
    /// ([`arko_ai::antigravity::prompt_digest`]). A resumed session keeps
    /// that prompt, so a task whose prompt changed starts a new session.
    pub prompt: String,
    /// Message index at the start of a turn with no final reply yet. A
    /// snapshot here can't tell how much of the turn the CLI saved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pending_from: Option<usize>,
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
