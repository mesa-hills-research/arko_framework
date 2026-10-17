use super::builtins::{
    ApplyDiffTool, AskUserTool, COMPLETE_TASK_TOOL_NAME, CompleteTaskTool, EditFileTool,
    ListFilesTool, MessageUserTool, ReadFileTool, WriteFileTool,
};
use super::{EndUserTool, Tool, ToolDefinition, ToolError};
use std::collections::HashMap;
use std::sync::Arc;

/// A registry for managing available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates a new ToolRegistry with all built-in tools registered.
    pub fn new() -> Self {
        let mut tools = HashMap::new();

        let complete_task_tool = Arc::new(CompleteTaskTool::new());
        tools.insert(
            complete_task_tool.name(),
            complete_task_tool.clone() as Arc<dyn Tool>,
        );

        let read_file_tool = Arc::new(ReadFileTool::new());
        tools.insert(
            read_file_tool.name(),
            read_file_tool.clone() as Arc<dyn Tool>,
        );

        let write_file_tool = Arc::new(WriteFileTool::new());
        tools.insert(
            write_file_tool.name(),
            write_file_tool.clone() as Arc<dyn Tool>,
        );

        let edit_file_tool = Arc::new(EditFileTool::new());
        tools.insert(
            edit_file_tool.name(),
            edit_file_tool.clone() as Arc<dyn Tool>,
        );

        let apply_diff_tool = Arc::new(ApplyDiffTool::new());
        tools.insert(
            apply_diff_tool.name(),
            apply_diff_tool.clone() as Arc<dyn Tool>,
        );

        let list_files_tool = Arc::new(ListFilesTool::new());
        tools.insert(
            list_files_tool.name(),
            list_files_tool.clone() as Arc<dyn Tool>,
        );

        let ask_user_tool = Arc::new(AskUserTool::new());
        tools.insert(ask_user_tool.name(), ask_user_tool.clone() as Arc<dyn Tool>);

        let message_user_tool = Arc::new(MessageUserTool::new());
        tools.insert(
            message_user_tool.name(),
            message_user_tool.clone() as Arc<dyn Tool>,
        );

        Self { tools }
    }

    /// Registers a tool in the registry, replacing a tool registered under
    /// its name before (a built-in included), except that `complete_task`
    /// can't be replaced by a tool that isn't built-in. A host's own
    /// `message_user` replaces the built-in: it runs as any other tool, and
    /// the runtime sends no
    /// [`AgentEvent::MessageToUser`](crate::AgentEvent::MessageToUser) for
    /// its calls.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let tool_name = tool.name();
        if tool_name == COMPLETE_TASK_TOOL_NAME
            && tool.definition().tool_type_details != super::ToolTypeDetails::BuiltIn
        {
            log::warn!(
                "Attempted to overwrite mandatory '{}' tool with a non-built-in type. Registration skipped.",
                COMPLETE_TASK_TOOL_NAME
            );
            return;
        }
        self.tools.insert(tool_name, tool);
    }

    /// Gets a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// A registry with only the tools of this one that `names` lists. Names
    /// not registered here are skipped. The runtime runs a task with the
    /// tools its configuration offers
    /// ([`AgentToolConfig::offered_tool_names`](crate::agent_config::AgentToolConfig::offered_tool_names)).
    pub fn restricted_to<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Self {
        let tools = names
            .into_iter()
            .filter_map(|name| Some((name.to_string(), self.tools.get(name)?.clone())))
            .collect();
        Self { tools }
    }

    /// Loads and registers an end-user defined tool from its definition.
    pub fn load_end_user_tool(&mut self, definition: ToolDefinition) -> Result<(), ToolError> {
        let tool_name = definition.name.clone();
        match definition.tool_type_details {
            super::ToolTypeDetails::EndUserDefined { .. } => {
                let end_user_tool = Arc::new(EndUserTool::new(definition)?);
                self.register(end_user_tool);
                Ok(())
            }
            _ => Err(ToolError::Configuration {
                tool_name,
                message: "ToolDefinition is not for an EndUserDefined tool.".to_string(),
            }),
        }
    }

    /// Lists all registered tool names.
    pub fn list_tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Lists all registered tool definitions.
    pub fn list_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
