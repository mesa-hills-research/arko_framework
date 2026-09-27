//! Memory tools injected into agent ToolRegistries.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

use crate::context::SharedContext;
use crate::freshness::{ActorId, ResourceId};

pub const READ_MEMORY_TOOL_NAME: &str = "read_memory";
pub const WRITE_MEMORY_TOOL_NAME: &str = "write_memory";
pub const LIST_MEMORIES_TOOL_NAME: &str = "list_memories";

/// Tool for reading memory tier content.
#[derive(Debug)]
pub struct ReadMemoryTool {
    context: Arc<SharedContext>,
    agent_name: String,
    reads: super::reads::MemoryReads,
}

impl ReadMemoryTool {
    pub fn new(context: Arc<SharedContext>, agent_name: String) -> Self {
        Self {
            context,
            agent_name,
            reads: Default::default(),
        }
    }
}

#[async_trait]
impl Tool for ReadMemoryTool {
    fn allow_result_interception(&self) -> bool {
        false
    }
    fn reset_context(&self, task_id: uuid::Uuid) {
        self.reads.reset(task_id);
    }
    fn observe_context(&self, task_id: uuid::Uuid, message: &arko_ai::Message) {
        self.reads.observe(task_id, message);
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: READ_MEMORY_TOOL_NAME.to_string(),
            description: "Read versioned workflow memory. First read is full; later reads return exact changes or unchanged. Use full=true for a fresh snapshot. Changes identify their base version; replacements use zero-based lines retaining newline terminators. Use list_memories to discover tiers and access.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "full".into(), description: "Return the full current snapshot instead of a diff.".into(),
                    param_type: ToolParameterType::Bool, required: false, is_path: false, flag: None, default_value: None,
                },
                ToolParameterDefinition {
                    name: "tier".to_string(),
                    description: "The name of the memory tier to read.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "max_chars".to_string(),
                    description: "Optional maximum number of trailing characters to return, encoded as a positive integer string.".to_string(),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            supports_batched_calls: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        task_context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let tier_name = params.get("tier").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::MissingParameter {
                param_name: "tier".to_string(),
            }
        })?;

        let max_chars = params
            .get("max_chars")
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| ToolError::ParameterValidation {
                        param_name: "max_chars".into(),
                        message: "Expected a positive integer string".into(),
                    })?
                    .parse::<usize>()
                    .map_err(|error| ToolError::ParameterValidation {
                        param_name: "max_chars".into(),
                        message: error.to_string(),
                    })
            })
            .transpose()?;
        let full = match params.get("full") {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(Value::String(value)) if value == "true" || value == "false" => value == "true",
            Some(_) => {
                return Err(ToolError::ParameterValidation {
                    param_name: "full".into(),
                    message: "Expected true or false".into(),
                });
            }
        };
        if max_chars == Some(0) {
            return Err(ToolError::ParameterValidation {
                param_name: "max_chars".into(),
                message: "Must be positive".into(),
            });
        }
        let (version, content) = self
            .context
            .memory
            .read_versioned(&self.agent_name, tier_name)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        if let Some(tracker) = &self.context.resource_tracker
            && max_chars.is_none_or(|limit| content.chars().count() <= limit)
        {
            tracker.record_read(
                &ActorId::new(task_context.task_id, &self.agent_name),
                ResourceId::MemoryTier(tier_name.to_string()),
            );
        }
        self.reads
            .prepare(
                task_context.task_id,
                tier_name,
                version,
                content,
                full,
                max_chars,
            )
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))
    }
}

/// Tool for writing to a memory tier.
#[derive(Debug)]
pub struct WriteMemoryTool {
    context: Arc<SharedContext>,
    agent_name: String,
}

impl WriteMemoryTool {
    pub fn new(context: Arc<SharedContext>, agent_name: String) -> Self {
        Self {
            context,
            agent_name,
        }
    }
}

#[async_trait]
impl Tool for WriteMemoryTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: WRITE_MEMORY_TOOL_NAME.to_string(),
            description: "Write content to a workflow memory tier. The behavior (append or replace) depends on the tier's write mode.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "tier".to_string(),
                    description: "The name of the memory tier to write to.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "content".to_string(),
                    description: "The content to write.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            supports_batched_calls: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        task_context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let tier_name = params.get("tier").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::MissingParameter {
                param_name: "tier".to_string(),
            }
        })?;
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "content".to_string(),
            })?;

        let actor = ActorId::new(task_context.task_id, &self.agent_name);
        let resource = ResourceId::MemoryTier(tier_name.to_string());
        if let Some(tracker) = &self.context.resource_tracker {
            let conflicts = tracker.check_write(&actor, &resource);
            if tracker.config().block_on_conflict && !conflicts.is_empty() {
                return Err(ToolError::ExecutionFailed(format!(
                    "Context freshness conflict for memory tier '{tier_name}'; read it again before writing"
                )));
            }
        }

        self.context
            .memory
            .write_with_context(
                &self.agent_name,
                Some(task_context.task_id),
                tier_name,
                content,
            )
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if let Some(tracker) = &self.context.resource_tracker {
            for reader in tracker.record_write(&actor, resource) {
                self.context
                    .messages
                    .inject_task(
                        reader.task_id,
                        format!(
                            "Context freshness notice: memory tier '{tier_name}' changed after you read it. Read it again before relying on that content."
                        ),
                    )
                    .await;
                crate::workflow::engine::fire_hooks_on_context(
                    &self.context,
                    crate::hooks::HookPoint::OnFreshnessConflict,
                    |hook| {
                        hook.agent_name = Some(reader.agent_name.clone());
                        hook.agent_run_id = Some(reader.task_id);
                        hook.metadata.insert(
                            "resource".to_string(),
                            serde_json::json!(format!("memory:{tier_name}")),
                        );
                        hook.metadata
                            .insert("writer".to_string(), serde_json::json!(self.agent_name));
                    },
                )
                .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
            }
        }

        Ok(format!(
            "Successfully wrote to memory tier '{}'.",
            tier_name
        ))
    }
}

/// Tool for listing available memory tiers and access levels.
#[derive(Debug)]
pub struct ListMemoriesTool {
    context: Arc<SharedContext>,
    agent_name: String,
}

impl ListMemoriesTool {
    pub fn new(context: Arc<SharedContext>, agent_name: String) -> Self {
        Self {
            context,
            agent_name,
        }
    }
}

#[async_trait]
impl Tool for ListMemoriesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LIST_MEMORIES_TOOL_NAME.to_string(),
            description: "List all workflow memory tiers and your access level for each."
                .to_string(),
            output_based_example: None,
            parameters: vec![],
            supports_batched_calls: true,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        _params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let accessible = self.context.memory.list_accessible(&self.agent_name);
        if accessible.is_empty() {
            return Ok("No memory tiers are configured in this workflow.".to_string());
        }
        let mut output = String::from("Available memory tiers:\n");
        for (name, can_read, can_write) in accessible {
            let access = match (can_read, can_write) {
                (true, true) => "read/write",
                (true, false) => "read-only",
                (false, true) => "write-only",
                (false, false) => "no access",
            };
            output.push_str(&format!("- {} ({})\n", name, access));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Workflow;
    use crate::freshness::ContextFreshnessConfig;
    use crate::memory::MemoryConfig;

    fn workflow() -> Workflow {
        let config = arko_agents::AgentConfig::builder(
            "reader".into(),
            "Reader".into(),
            arko_agents::AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        Workflow::builder()
            .name("memory-tests")
            .workspace(std::env::temp_dir())
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("reader", config)
            .ephemeral()
            .context_freshness(ContextFreshnessConfig::default())
            .memory(
                MemoryConfig::new()
                    .global_tier("notes")
                    .tier(super::super::MemoryTier {
                        name: "private".into(),
                        content: "private content".into(),
                        readers: super::super::AccessScope::None,
                        writers: super::super::AccessScope::None,
                        write_mode: Default::default(),
                        entries: vec![],
                        max_bytes: None,
                    }),
            )
            .build()
            .unwrap()
    }

    fn params(value: Value) -> HashMap<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[tokio::test]
    async fn tool_validates_parameters_permissions_and_full_overrides_in_both_calling_formats() {
        let workflow = workflow();
        workflow
            .context()
            .memory
            .write("writer", "notes", &"long initial content\n".repeat(30))
            .unwrap();
        let tool = ReadMemoryTool::new(workflow.context().clone(), "reader".into());
        let task = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "reader",
            initial_task_prompt: "read",
            current_working_directory: &workflow.context().workspace,
        };
        let full = tool
            .execute(&task, params(serde_json::json!({"tier":"notes"})))
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&full).unwrap()["kind"],
            "full"
        );
        tool.observe_context(task.task_id, &arko_ai::Message::new_user(&full));
        workflow
            .context()
            .memory
            .write("writer", "notes", "new line")
            .unwrap();
        let change = tool
            .execute(&task, params(serde_json::json!({"tier":"notes"})))
            .await
            .unwrap();
        let change: Value = serde_json::from_str(&change).unwrap();
        assert_eq!(change["kind"], "append");
        assert_eq!(change["base_version"], 1);
        assert_eq!(change["version"], 2);
        for full in [Value::Bool(true), Value::String("true".into())] {
            let result = tool
                .execute(
                    &task,
                    params(serde_json::json!({"tier":"notes", "full":full})),
                )
                .await
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&result).unwrap()["kind"],
                "full"
            );
        }
        for invalid in [
            serde_json::json!({"tier":"notes", "full":"yes"}),
            serde_json::json!({"tier":"notes", "max_chars":"0"}),
            serde_json::json!({"tier":"notes", "max_chars":5}),
            serde_json::json!({"tier":"notes", "max_chars":"invalid"}),
            serde_json::json!({"tier":"private"}),
            serde_json::json!({"tier":"missing"}),
        ] {
            assert!(tool.execute(&task, params(invalid)).await.is_err());
        }
        assert!(!tool.allow_result_interception());
    }

    #[tokio::test]
    async fn partial_reads_do_not_clear_a_freshness_conflict() {
        let workflow = workflow();
        let context = workflow.context();
        let tool = ReadMemoryTool::new(context.clone(), "reader".into());
        let task = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "reader",
            initial_task_prompt: "read",
            current_working_directory: &context.workspace,
        };
        let resource = ResourceId::MemoryTier("notes".into());
        let reader = ActorId::new(task.task_id, "reader");
        let writer = ActorId::new(uuid::Uuid::new_v4(), "writer");
        let tracker = context.resource_tracker.as_ref().unwrap();
        context.memory.write("writer", "notes", "baseline").unwrap();
        tool.execute(&task, params(serde_json::json!({"tier":"notes"})))
            .await
            .unwrap();
        context
            .memory
            .write("writer", "notes", "more content")
            .unwrap();
        tracker.record_write(&writer, resource.clone());
        assert!(!tracker.check_write(&reader, &resource).is_empty());
        let result = tool
            .execute(
                &task,
                params(serde_json::json!({"tier":"notes", "max_chars":"4"})),
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&result).unwrap()["kind"],
            "partial"
        );
        assert!(!tracker.check_write(&reader, &resource).is_empty());
        tool.execute(
            &task,
            params(serde_json::json!({"tier":"notes", "full":true})),
        )
        .await
        .unwrap();
        assert!(tracker.check_write(&reader, &resource).is_empty());
    }
}
