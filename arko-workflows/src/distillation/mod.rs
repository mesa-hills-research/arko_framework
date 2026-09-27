//! LLM-backed, context-aware tool-result distillation.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use arko_agents::{InterceptedResult, TaskContext, ToolError, ToolResultInterceptor};

use crate::context::SharedContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    pub name: String,
    pub model: arko_ai::Model,
    pub target_agents: HashSet<String>,
    pub target_tools: HashSet<String>,
    pub min_chars: usize,
    pub max_output_tokens: usize,
    pub instructions: String,
    /// Optional reasoning configuration for the distillation model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<arko_ai::ReasoningConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arko_ai::{Model, OpenAIEffort, ReasoningConfig};

    #[test]
    fn reasoning_reaches_generation_config_and_legacy_configs_still_load() {
        let config = DistillationConfig::new("bulk", Model::openai_gpt_5_6_luna())
            .reasoning(ReasoningConfig::openai(OpenAIEffort::High, false))
            .max_output_tokens(2_000);
        let mut json = serde_json::to_value(&config).unwrap();
        let restored: DistillationConfig = serde_json::from_value(json.clone()).unwrap();
        restored.validate().unwrap();
        let generation = restored.generation_config();
        assert_eq!(generation.max_tokens, Some(2_000));
        assert!(matches!(
            generation.reasoning,
            Some(ReasoningConfig::OpenAI {
                effort: OpenAIEffort::High,
                include_summaries: false,
            })
        ));
        json.as_object_mut().unwrap().remove("reasoning");
        let legacy: DistillationConfig = serde_json::from_value(json).unwrap();
        assert!(legacy.generation_config().reasoning.is_none());
    }

    #[test]
    fn invalid_distillers_are_rejected_during_workflow_build() {
        for config in [
            DistillationConfig::new("zero", Model::openai_gpt_5_6_luna()).max_output_tokens(0),
            DistillationConfig::new("mismatch", Model::openai_gpt_5_6_luna())
                .reasoning(ReasoningConfig::anthropic(1_024)),
        ] {
            assert!(matches!(
                crate::Workflow::builder().distiller(config).build(),
                Err(crate::WorkflowError::BuilderInvalid(_))
            ));
        }
    }

    #[test]
    fn thresholds_and_target_filters_are_composed() {
        let config = DistillationConfig::new("bulk", Model::openai_gpt_5_6_luna())
            .min_chars(5)
            .target_agent("researcher")
            .target_tool("search");
        assert!(!config.applies("researcher", "search", "éééé".chars().count()));
        assert!(config.applies("researcher", "search", "ééééé".chars().count()));
        assert!(!config.applies("builder", "search", 100));
        assert!(!config.applies("researcher", "read_file", 100));
        assert!(
            DistillationConfig::new("all", Model::openai_gpt_5_6_luna())
                .min_chars(0)
                .applies("any", "any", 0)
        );
    }
}

impl DistillationConfig {
    pub fn new(name: impl Into<String>, model: arko_ai::Model) -> Self {
        Self {
            name: name.into(),
            model,
            target_agents: HashSet::new(),
            target_tools: HashSet::new(),
            min_chars: 8_000,
            max_output_tokens: 4_000,
            instructions: "Preserve every fact relevant to the task, exact errors, paths, identifiers, constraints, and actionable details. Remove repetition and irrelevant bulk. Never invent facts.".to_string(),
            reasoning: None,
        }
    }

    pub fn target_agent(mut self, agent: impl Into<String>) -> Self {
        self.target_agents.insert(agent.into());
        self
    }

    pub fn target_tool(mut self, tool: impl Into<String>) -> Self {
        self.target_tools.insert(tool.into());
        self
    }

    pub fn min_chars(mut self, chars: usize) -> Self {
        self.min_chars = chars;
        self
    }

    pub fn max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    pub fn reasoning(mut self, reasoning: arko_ai::ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), crate::WorkflowError> {
        if self.max_output_tokens == 0 {
            return Err(crate::WorkflowError::BuilderInvalid(format!(
                "distiller '{}' output token limit must be positive",
                self.name
            )));
        }
        if self
            .reasoning
            .as_ref()
            .is_some_and(|reasoning| reasoning.provider() != self.model.provider)
        {
            return Err(crate::WorkflowError::BuilderInvalid(format!(
                "distiller '{}' reasoning provider does not match its model",
                self.name
            )));
        }
        Ok(())
    }

    fn generation_config(&self) -> arko_ai::GenerationConfig {
        let mut config =
            arko_ai::GenerationConfig::default().with_max_tokens(self.max_output_tokens);
        config.reasoning = self.reasoning.clone();
        config
    }

    fn applies(&self, agent: &str, tool: &str, chars: usize) -> bool {
        chars >= self.min_chars
            && (self.target_agents.is_empty() || self.target_agents.contains(agent))
            && (self.target_tools.is_empty() || self.target_tools.contains(tool))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub task_id: uuid::Uuid,
    pub agent_name: String,
    pub tool_name: String,
    pub config_name: String,
    pub input_chars: usize,
    pub output_chars: usize,
    pub cost: f64,
}

pub(crate) struct DistillationInterceptor {
    pub config: DistillationConfig,
    pub client: Arc<arko_ai::LLMClient>,
    pub context: Arc<SharedContext>,
}

#[async_trait]
impl ToolResultInterceptor for DistillationInterceptor {
    async fn intercept(
        &self,
        tool_name: &str,
        raw_result: &str,
        params: &HashMap<String, Value>,
        task_context: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        if !self.config.applies(
            task_context.agent_name,
            tool_name,
            raw_result.chars().count(),
        ) {
            return Ok(InterceptedResult::Unchanged);
        }

        let mut conversation = arko_ai::Conversation::new(self.config.model.clone())
            .with_system_message(&format!(
                "You distill tool output for another agent. {}",
                self.config.instructions
            ))
            .with_cache_disabled();
        conversation.add_user_message(&format!(
            "Agent task:\n{}\n\nTool: {}\nParameters: {}\n\nRaw tool result:\n{}",
            task_context.initial_task_prompt,
            tool_name,
            serde_json::to_string(params).unwrap_or_else(|_| "{}".to_string()),
            raw_result
        ));
        let response = self
            .client
            .generate_text(&conversation, self.config.generation_config())
            .await
            .map_err(|error| {
                ToolError::ExecutionFailed(format!(
                    "distiller '{}' failed: {error}",
                    self.config.name
                ))
            })?;
        let distilled = response
            .message
            .content
            .iter()
            .filter_map(arko_ai::Content::as_text)
            .collect::<Vec<_>>()
            .join("\n");
        if distilled.trim().is_empty() {
            return Err(ToolError::ExecutionFailed(format!(
                "distiller '{}' returned no text",
                self.config.name
            )));
        }

        let cost_result = self.context.budget.add_manual_cost(
            response.usage.billed(),
            &format!("distillation: {}", self.config.name),
            Some(format!(
                "agent={}, task={}, tool={}",
                task_context.agent_name, task_context.task_id, tool_name
            )),
        );
        if cost_result.warning_triggered {
            crate::workflow::engine::fire_hooks_on_context(
                &self.context,
                crate::hooks::HookPoint::OnBudgetWarning,
                |hook| hook.budget_spent = cost_result.new_total,
            )
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        }
        if cost_result.exceeded {
            crate::workflow::engine::fire_hooks_on_context(
                &self.context,
                crate::hooks::HookPoint::OnBudgetExceeded,
                |hook| hook.budget_spent = cost_result.new_total,
            )
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        }
        self.context
            .distillation_records
            .write()
            .push(DistillationRecord {
                timestamp: chrono::Utc::now(),
                task_id: task_context.task_id,
                agent_name: task_context.agent_name.to_string(),
                tool_name: tool_name.to_string(),
                config_name: self.config.name.clone(),
                input_chars: raw_result.chars().count(),
                output_chars: distilled.chars().count(),
                cost: response.usage.billed(),
            });
        Ok(InterceptedResult::Replaced(distilled))
    }
}
