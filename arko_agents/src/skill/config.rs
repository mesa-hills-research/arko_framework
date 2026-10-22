use serde::{Deserialize, Serialize};

/// Behavior when a skill fails to parse or load.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum InvalidSkillPolicy {
    /// Log a warning and skip the invalid skill, or one that can't be read
    /// (a skill file or folder, or a whole source directory). Other skills
    /// continue loading.
    #[default]
    Warn,
    /// Return an error immediately. No skills are loaded.
    Error,
}

/// A source directory from which skills are loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSource {
    /// Path to the directory containing skills.
    /// For workspace sources, relative to the `workspace_root` given to
    /// [`SkillManager::load`](super::SkillManager::load) (e.g., ".skills").
    /// For global sources, an absolute path (e.g., "/home/user/.skills/global").
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

/// A skill held in memory, such as one built into a program, instead of read
/// from a folder.
///
/// It is parsed and checked as a folder skill's `SKILL.md` is, so it needs
/// the same front matter. It has no folder, so it has no resource files to
/// load beside it and no scripts. Build one with [`new`](Self::new).
///
/// ```
/// use arko_agents::skill::EmbeddedSkill;
///
/// let skill = EmbeddedSkill::new(
///     "commit-style",
///     "---\nname: commit-style\ndescription: How to word a commit message.\n---\nStart with the area.\n",
/// );
/// assert_eq!(skill.name, "commit-style");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddedSkill {
    /// Names the skill in warnings and errors, as a folder skill's folder
    /// name does. The agents see the `name` in the front matter, which wins
    /// (with a warning) when the two differ.
    pub name: String,

    /// The whole text of the skill's `SKILL.md`, front matter included.
    pub content: String,
}

impl EmbeddedSkill {
    /// A skill with the given `name` and the full text of its `SKILL.md`.
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
        }
    }
}

/// Configuration for the skills system within an agent config.
///
/// `SkillsConfig::default()` has no skills, so a config can be written as
/// `SkillsConfig { embedded, ..Default::default() }`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillsConfig {
    /// Optional workspace-relative skills directory (e.g., ".skills", ".claude/skills").
    /// Path is resolved relative to the `workspace_root` given to
    /// [`SkillManager::load`](super::SkillManager::load).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_source: Option<SkillSource>,

    /// Zero or more global skill directories (absolute paths).
    /// Processed in order. Earlier sources have priority for deduplication.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub global_sources: Vec<SkillSource>,

    /// Skills given in memory. They load after those of the folders and have
    /// the lowest priority. A folder skill with the same name replaces the
    /// embedded one, which is how a user overrides a built-in skill.
    ///
    /// They are program code, so a saved configuration (and so a saved task)
    /// doesn't hold them, though one that has them still loads. A host that
    /// loads a saved task gives them again. arko_workflows does for a run that
    /// goes on, from the configuration its agent is registered with (a run the
    /// host reconfigured gets them when its own skills configuration is set).
    /// A subagent has the skills of its own agent, not its lead's.
    #[serde(default, skip_serializing)]
    pub embedded: Vec<EmbeddedSkill>,

    /// Behavior when a skill fails to parse or load.
    /// Defaults to `Warn` (skip the skill, log a warning).
    #[serde(default)]
    pub on_invalid_skill: InvalidSkillPolicy,

    /// Lets the tool-result interceptor given to the runtime (such as
    /// a workflow's distiller) replace `load_skill` results. This policy
    /// handles the failures the interceptor reports by retrying, falling
    /// back to the full result, or giving the full result with a note that it
    /// could not be shortened. `None`, the default, sends skills as written
    /// because a skill carries exact instructions, which a summary would lose.
    /// (When arko_workflows chains several interceptors, the chain applies
    /// this policy to each interceptor in it and reports no failures
    /// itself.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_skill_interception: Option<crate::tool::InterceptionFailurePolicy>,
}
