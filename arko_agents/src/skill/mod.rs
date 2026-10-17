pub mod config;
mod frontmatter;
pub mod load_skill;
mod loader;
pub mod types;

pub use config::{InvalidSkillPolicy, SkillSource, SkillsConfig};
pub use load_skill::{LOAD_SKILL_TOOL_NAME, LoadSkillTool};
pub use types::{
    AutoLoadedSkillContent, LoadedSkill, SkillLoadError, SkillPromptContext, SkillScan,
    SkillSummary,
};

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::agent_config::{AgentConfig, PerToolConfigOverride};
use crate::error::ArkoAgentsError;
use crate::tool::ToolRegistry;

/// Lists the skill files and folders in the sources of `config`, resolving
/// workspace paths relative to `workspace_root`, with what
/// [`SkillManager::load`] makes of each, the name it loads it under or why
/// it skips it. It reads them by the same code as loading, under
/// [`InvalidSkillPolicy::Warn`], and never fails. A source directory that
/// doesn't exist has nothing to list, and one that can't be read is listed
/// as skipped. Within a source, entries are taken in the order of their
/// names, and earlier sources come first.
pub fn scan(config: &SkillsConfig, workspace_root: &Path) -> Vec<SkillScan> {
    loader::scan(config, workspace_root)
}

/// Manages skill loading, deduplication, and tool creation.
///
/// Created from a `SkillsConfig` at agent setup time. Provides these items:
/// - A `LoadSkillTool` to register in the `ToolRegistry`
/// - A `SkillPromptContext` for the `PromptBuilder`
/// - The tool name to add to `available_tool_names`
pub struct SkillManager {
    on_demand_skills: Arc<HashMap<String, LoadedSkill>>,
    prompt_context: SkillPromptContext,
}

impl SkillManager {
    /// Loads skills from the given config, resolving workspace paths relative
    /// to `workspace_root`.
    ///
    /// Returns `Err` only if `config.on_invalid_skill` is `Error` and a
    /// skill is invalid, can't be read, or a source directory is missing
    /// or can't be read. Under `Warn` those are logged and skipped.
    pub fn load(config: &SkillsConfig, workspace_root: &Path) -> Result<Self, ArkoAgentsError> {
        let (all_skills, prompt_context) = loader::load_skills(config, workspace_root)?;

        let on_demand_map: HashMap<String, LoadedSkill> = all_skills
            .into_iter()
            .filter(|s| !s.auto_load)
            .map(|s| (s.name.clone(), s))
            .collect();

        Ok(Self {
            on_demand_skills: Arc::new(on_demand_map),
            prompt_context,
        })
    }

    /// Creates the `LoadSkillTool` instance.
    pub fn create_tool(&self) -> LoadSkillTool {
        LoadSkillTool::new(Arc::clone(&self.on_demand_skills))
    }

    /// Registers the `load_skill` tool in the given registry.
    pub fn register_tool(&self, registry: &mut ToolRegistry) {
        let tool = Arc::new(self.create_tool());
        registry.register(tool);
    }

    /// Returns the tool name constant for adding to `available_tool_names`.
    pub fn tool_name(&self) -> &'static str {
        LOAD_SKILL_TOOL_NAME
    }

    /// Returns the prompt context for the `PromptBuilder`.
    pub fn prompt_context(&self) -> &SkillPromptContext {
        &self.prompt_context
    }

    /// Returns true if there are any on-demand skills available.
    pub fn has_on_demand_skills(&self) -> bool {
        !self.on_demand_skills.is_empty()
    }

    /// Returns true if there are any auto-loaded skills.
    pub fn has_auto_loaded_skills(&self) -> bool {
        !self.prompt_context.auto_loaded_skills.is_empty()
    }

    /// Returns the total number of loaded skills (on-demand + auto-loaded).
    pub fn total_skill_count(&self) -> usize {
        self.on_demand_skills.len() + self.prompt_context.auto_loaded_skills.len()
    }

    /// Sets up on-demand skills and returns the `SkillPromptContext`.
    ///
    /// When on-demand skills exist, it performs these steps:
    /// - Registers the `load_skill` tool in the registry.
    /// - Adds the tool name to the agent config's available tools.
    /// - Adds an auto-approve override for `load_skill` if it has no
    ///   per-tool configuration entry.
    pub fn setup(
        &self,
        registry: &mut ToolRegistry,
        agent_config: &mut AgentConfig,
    ) -> SkillPromptContext {
        if self.has_on_demand_skills() {
            self.register_tool(registry);

            let tool_name = LOAD_SKILL_TOOL_NAME.to_string();
            if !agent_config
                .tool_config
                .available_tool_names
                .contains(&tool_name)
            {
                agent_config
                    .tool_config
                    .available_tool_names
                    .push(tool_name.clone());
            }

            agent_config
                .tool_config
                .tool_specific_configs
                .entry(tool_name)
                .or_insert_with(|| PerToolConfigOverride {
                    auto_approve_tool_use: Some(true),
                    ..Default::default()
                });
        }

        self.prompt_context.clone()
    }
}
