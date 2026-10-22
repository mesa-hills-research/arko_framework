use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during skill loading.
#[derive(Debug, Error)]
pub enum SkillLoadError {
    #[error("Missing YAML frontmatter (expected '---' delimiters) in {path}")]
    MissingFrontmatter { path: String },

    #[error("Malformed YAML frontmatter (missing closing '---') in {path}")]
    MalformedFrontmatter { path: String },

    #[error("Failed to parse YAML frontmatter in {path}: {source}")]
    FrontmatterParseError {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },

    #[error("Missing required frontmatter field '{field}' in {path}")]
    MissingRequiredField { field: String, path: String },

    #[error("I/O error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Zip extraction error for {path}: {message}")]
    ZipError { path: String, message: String },

    #[error("Skill bundle {path} does not contain a SKILL.md")]
    MissingSkillMd { path: String },
}

/// The deserialized YAML frontmatter from a SKILL.md or flat .md file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillFrontmatter {
    /// Required. The skill identifier. Should be kebab-case.
    pub name: String,

    /// Required. Description shown to the agent.
    pub description: String,

    /// If true, the skill's full content is injected into the system prompt
    /// rather than being available on-demand via the load_skill tool.
    #[serde(default)]
    pub auto_load: bool,

    /// Remaining fields are captured but not validated.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

/// A fully loaded skill with all content in memory.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    /// Unique name, possibly modified by deduplication (e.g., "skill-name-2").
    pub name: String,

    /// Original name from frontmatter, before any dedup renaming.
    pub original_name: String,

    /// Description from frontmatter.
    pub description: String,

    /// Whether this skill should be auto-loaded into the system prompt.
    pub auto_load: bool,

    /// The body content of SKILL.md (or the flat .md file), after frontmatter
    /// has been stripped.
    pub content: String,

    /// Sub-resources mapped from relative paths (e.g., "FORMS.md") to file content.
    /// Only .md files are included. Empty for flat/minimal skills.
    pub resources: HashMap<String, String>,

    /// Human-readable label indicating where this skill was loaded from,
    /// `workspace:<path>`, `global:<path>` or `embedded`.
    pub source_label: String,
}

/// Lightweight view used for prompt building (name + description only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSummary {
    pub name: String,
    pub description: String,
}

/// Content of an auto-loaded skill for prompt injection.
#[derive(Debug, Clone)]
pub struct AutoLoadedSkillContent {
    pub name: String,
    pub content: String,
}

/// Everything the prompt builder needs, produced by the loader.
#[derive(Debug, Clone, Default)]
pub struct SkillPromptContext {
    /// Skills available for on-demand loading.
    pub on_demand_skills: Vec<SkillSummary>,

    /// Skills whose full content should be directly injected into the system prompt.
    pub auto_loaded_skills: Vec<AutoLoadedSkillContent>,
}

/// A skill file or folder in the sources of a [`SkillsConfig`](super::SkillsConfig),
/// as [`scan`](super::scan) lists it, with what loading the skills makes of
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillScan {
    /// Where it is, a folder skill's folder or a bundle's (`.skill`) or a
    /// flat skill's (`.md`) file. A source directory that can't be read is
    /// listed by its own path. Empty for an embedded skill, which has none
    /// (see `embedded`).
    pub path: PathBuf,
    /// It is an [`EmbeddedSkill`](super::EmbeddedSkill), given in memory and
    /// not read from a folder.
    pub embedded: bool,
    /// Its name as agents load it, renamed (`name-2`) when a skill loaded
    /// before it has the name. One that can't be read goes by its folder's
    /// name, or its file's without the extension.
    pub name: String,
    /// Its description (empty for one that can't be read).
    pub description: String,
    /// Its content goes in the system prompt instead of being loaded on
    /// demand.
    pub auto_load: bool,
    /// Why it isn't loaded, if it isn't. That may be why it is invalid or
    /// can't be read, its source's `include` or `exclude` list, a skill
    /// with the same content loaded before it, or, for an embedded skill,
    /// the folder skill with its name that replaces it.
    pub skipped: Option<String>,
}
