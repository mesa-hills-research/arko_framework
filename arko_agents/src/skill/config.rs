use serde::{Deserialize, Serialize};

/// Behavior when a skill fails to parse or load.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum InvalidSkillPolicy {
    /// Log a warning and skip the invalid skill. Other skills continue loading.
    #[default]
    Warn,
    /// Return an error immediately. No skills are loaded.
    Error,
}

/// A source directory from which skills are loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSource {
    /// Path to the directory containing skills.
    /// For workspace sources: relative to the `workspace_root` given to
    /// [`SkillManager::load`](super::SkillManager::load) (e.g., ".skills").
    /// For global sources: absolute path (e.g., "/home/user/.skills/global").
    pub path: String,

    /// If set, only skills whose names are in this list are loaded from this source.
    /// Acts as a whitelist. Applied before `exclude`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    /// If set, skills whose names are in this list are excluded from this source.
    /// Acts as a blacklist. Applied after `include` (if both are set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<Vec<String>>,
}

/// Configuration for the skills system within an agent config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsConfig {
    /// Optional workspace-relative skills directory (e.g., ".skills", ".claude/skills").
    /// Path is resolved relative to the `workspace_root` given to
    /// [`SkillManager::load`](super::SkillManager::load).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_source: Option<SkillSource>,

    /// Zero or more global skill directories (absolute paths).
    /// Processed in order; earlier sources have priority for deduplication.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub global_sources: Vec<SkillSource>,

    /// Behavior when a skill fails to parse or load.
    /// Defaults to `Warn` (skip the skill, log a warning).
    #[serde(default)]
    pub on_invalid_skill: InvalidSkillPolicy,

    /// Opt-in: let the tool-result interceptor given to the runtime (such as
    /// a workflow's distiller) replace `load_skill` results, with this
    /// failure policy for the failures the interceptor reports: retry, fall
    /// back to the full result, or give the full result with a note that it
    /// could not be shortened. `None`, the default, sends skills as written:
    /// a skill carries exact instructions, which a summary would lose.
    /// (When arko_workflows chains several interceptors, the chain applies
    /// this policy to each interceptor in it and reports no failures
    /// itself.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_skill_interception: Option<crate::tool::InterceptionFailurePolicy>,
}
