//! Models that run through the installed Antigravity CLI on the user's plan.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

mod acp;
mod locate;
mod profile;
mod relay;
mod usage;

pub use acp::{
    ARKO_TOOL_PREFIX, AcpClient, AcpError, AcpEvent, AcpOptions, ArkoToolsOnly, ClientInfo,
    ConfigOption, ConfigValue, MAX_FRAME, NO_AUTO_UPDATE, NewSession, PermissionDecision,
    PermissionOption, PermissionPolicy, PermissionRequest, PromptResult, PromptUsage, RpcError,
    SessionUpdate, ToolCall, WIRE_LOG, agent_fell_back, config_values, image_block,
    log_shows_fallback, text_block, turn_failure,
};
pub use locate::{check_sign_in, cli_version, find_agy, version_at_least};
pub use profile::{
    AGENT_NAME, HelperCommand, MCP_SERVER_NAME, Profile, ProfileSpec, qualified_tool_name,
    run_gate, run_tripwire,
};
pub use relay::{RELAY_TOKEN_ENV, RelayHost, run_relay, write_address_file};
pub use usage::{parse_usage, plan_usage, plan_usage_report};

/// The usage windows Antigravity reports for a set of models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageGroup {
    pub name: String,
    pub description: String,
    pub buckets: Vec<UsageBucket>,
}

/// One window in the Antigravity plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageBucket {
    pub id: String,
    pub name: String,
    pub window: String,
    pub remaining_fraction: f64,
    pub reset_time: String,
}

/// The usage group that names the model. The CLI lists the model names in
/// the description unless the group is for just one model.
pub fn group_for_model<'a>(groups: &'a [UsageGroup], display_name: &str) -> Option<&'a UsageGroup> {
    groups
        .iter()
        .find(|group| {
            group
                .description
                .strip_prefix("Models within this group:")
                .is_some_and(|list| {
                    list.split(',')
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                        .any(|name| {
                            display_name == name
                                || display_name
                                    .strip_prefix(name)
                                    .is_some_and(|rest| rest.starts_with(" ("))
                        })
                })
        })
        .or_else(|| groups.iter().find(|group| group.name == display_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_match_listed_names_or_the_group_name() {
        let groups = [
            UsageGroup {
                name: "Claude and GPT models".into(),
                description:
                    "Models within this group: Claude Opus 5.5, Claude Sonnet 5.5, GPT-OSS 120B"
                        .into(),
                buckets: Vec::new(),
            },
            UsageGroup {
                name: "Gemini 4 Argon".into(),
                description: "Argon has a separate quota.".into(),
                buckets: Vec::new(),
            },
        ];
        assert_eq!(
            group_for_model(&groups, "GPT-OSS 120B (Medium)"),
            Some(&groups[0])
        );
        assert_eq!(group_for_model(&groups, "Gemini 4 Argon"), Some(&groups[1]));
        assert_eq!(group_for_model(&groups, "Claude Opus 5.50"), None);
    }
}

/// The lowest version Arko supports.
pub const MINIMUM_VERSION: &str = "1.5.14";

/// The installed CLI and the models available on its sign-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignIn {
    pub cli_path: PathBuf,
    pub version: String,
    pub tier: Option<String>,
    pub models: Vec<AvailableModel>,
}

/// A model family and the thought levels the CLI offers for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableModel {
    pub id: String,
    pub name: String,
    pub thought_levels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SignInProblem {
    #[error("The Antigravity CLI was not found.")]
    Missing,
    #[error("Antigravity {found} is too old. Install {MINIMUM_VERSION} or later.")]
    Outdated { found: String },
    #[error("Sign in by running `agy` once.")]
    SignedOut,
    #[error("Could not check the Antigravity CLI. {0}")]
    Failed(String),
}
