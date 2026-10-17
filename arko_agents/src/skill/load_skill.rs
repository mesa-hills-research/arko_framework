use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::types::LoadedSkill;
use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

/// Name constant for the load_skill built-in tool.
pub const LOAD_SKILL_TOOL_NAME: &str = "load_skill";
const NAME_PARAM: &str = "name";
const RESOURCE_PARAM: &str = "resource";

/// Built-in tool that loads on-demand skill content into the agent's context.
///
/// The tool holds a reference-counted map of all on-demand skills (those with
/// `auto_load: false`). Auto-loaded skills are not in this map because their
/// content is already injected into the system prompt.
#[derive(Debug, Clone)]
pub struct LoadSkillTool {
    skills: Arc<HashMap<String, LoadedSkill>>,
}

impl LoadSkillTool {
    /// Creates a new LoadSkillTool with the given on-demand skills.
    pub fn new(skills: Arc<HashMap<String, LoadedSkill>>) -> Self {
        Self { skills }
    }
}

#[async_trait]
impl Tool for LoadSkillTool {
    /// Loading a skill has no side effects. A call running when its task
    /// stopped loads it again.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LOAD_SKILL_TOOL_NAME.to_string(),
            description: "Load a skill's full content on-demand. Available skills are listed \
                in the system prompt under 'Available Skills'. Use this tool when a skill's \
                domain is relevant to your current task. Some skills have additional sub-resources \
                that are listed when the main skill is loaded."
                .to_string(),
            output_based_example: Some(
                "<load_skill>\n<name>frontend-design</name>\n</load_skill>".to_string(),
            ),
            parameters: vec![
                ToolParameterDefinition {
                    name: NAME_PARAM.to_string(),
                    description: "The name of the skill to load.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: RESOURCE_PARAM.to_string(),
                    description: "Optional. A specific resource file within the skill \
                        (e.g., 'FORMS.md' or 'reference/evaluation.md'). If omitted, the \
                        main skill content is returned. Available resources are listed when \
                        you load the main skill."
                        .to_string(),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let skill_name = params
            .get(NAME_PARAM)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: NAME_PARAM.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: NAME_PARAM.to_string(),
                message: "Parameter 'name' must be a string.".to_string(),
            })?
            .trim();

        if skill_name.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: NAME_PARAM.to_string(),
                message: "Skill name cannot be empty.".to_string(),
            });
        }

        let resource_name = params
            .get(RESOURCE_PARAM)
            .and_then(|v| v.as_str())
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());

        let skill = match self.skills.get(skill_name) {
            Some(s) => s,
            None => {
                let available: Vec<&str> = self.skills.keys().map(|k| k.as_str()).collect();
                let listing = if available.is_empty() {
                    "No on-demand skills are currently available.".to_string()
                } else {
                    let mut names: Vec<&str> = available;
                    names.sort();
                    names
                        .iter()
                        .map(|n| format!("- {}", n))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                return Ok(format!(
                    "Skill '{}' not found. Available skills:\n{}",
                    skill_name, listing
                ));
            }
        };

        if let Some(resource) = resource_name {
            match skill.resources.get(resource) {
                Some(resource_content) => Ok(format!(
                    "--- BEGIN SKILL RESOURCE: {}/{} ---\n{}\n--- END SKILL RESOURCE: {}/{} ---",
                    skill_name,
                    resource,
                    resource_content.trim(),
                    skill_name,
                    resource
                )),
                None => {
                    if skill.resources.is_empty() {
                        Ok(format!(
                            "Resource '{}' not found in skill '{}'. This skill has no additional resources.",
                            resource, skill_name
                        ))
                    } else {
                        let mut resource_names: Vec<&str> =
                            skill.resources.keys().map(|k| k.as_str()).collect();
                        resource_names.sort();
                        let listing = resource_names
                            .iter()
                            .map(|n| format!("- {}", n))
                            .collect::<Vec<_>>()
                            .join("\n");
                        Ok(format!(
                            "Resource '{}' not found in skill '{}'. Available resources:\n{}",
                            resource, skill_name, listing
                        ))
                    }
                }
            }
        } else {
            let mut output = format!(
                "--- BEGIN SKILL: {} ---\n{}\n--- END SKILL: {} ---",
                skill_name,
                skill.content.trim(),
                skill_name
            );

            if !skill.resources.is_empty() {
                let mut resource_names: Vec<&str> =
                    skill.resources.keys().map(|k| k.as_str()).collect();
                resource_names.sort();
                let listing = resource_names
                    .iter()
                    .map(|n| format!("- {}", n))
                    .collect::<Vec<_>>()
                    .join("\n");
                output.push_str(&format!(
                    "\n\nThis skill has additional resources that can be loaded individually:\n{}\n\n\
                     To load a resource, call this tool again with name=\"{}\" and resource=\"<resource_name>\".",
                    listing, skill_name
                ));
            }

            Ok(output)
        }
    }
}
