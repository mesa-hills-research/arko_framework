use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use self::command::UserCommand;
use self::event::AgentEvent;
use self::r#loop::AgentLoop;
use crate::error::ArkoAgentsError;
use crate::skill::SkillPromptContext;
use crate::task::Task;
use crate::tool::{InterceptorConfig, ToolExecutionMiddleware, ToolRegistry};
use arko_ai::LLMClient;

pub mod command;
pub mod event;
mod r#loop;
mod monitor;
mod retry;

const DEFAULT_EVENT_CHANNEL_SIZE: usize = 100;
const DEFAULT_COMMAND_CHANNEL_SIZE: usize = 10;

/// The main runtime for executing agent tasks.
pub struct AgentRuntime {
    llm_client: Arc<LLMClient>,
}

impl AgentRuntime {
    /// Creates a new AgentRuntime with the given LLM client.
    pub fn new(llm_client: Arc<LLMClient>) -> Self {
        Self { llm_client }
    }

    /// Runs a task with the given agent configuration and tool registry.
    ///
    /// An optional `InterceptorConfig` enables tool result interception,
    /// allowing results to be transformed (e.g., distilled) before they
    /// enter the agent's conversation. `complete_task` results are never
    /// intercepted.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn run_task(
        &self,
        task: Task,
        tool_registry: Arc<ToolRegistry>,
        event_throttle_ms: Option<u64>,
        simplified_tool_events: bool,
        skill_prompt_context: Option<SkillPromptContext>,
        interceptor_config: Option<InterceptorConfig>,
        execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    ) -> (
        mpsc::Receiver<AgentEvent>,
        mpsc::Sender<UserCommand>,
        JoinHandle<Result<Task, ArkoAgentsError>>,
    ) {
        let (event_sender, event_receiver) = mpsc::channel(DEFAULT_EVENT_CHANNEL_SIZE);
        let (command_sender, command_receiver) = mpsc::channel(DEFAULT_COMMAND_CHANNEL_SIZE);

        for name in tool_registry.list_tool_names() {
            if let Some(tool) = tool_registry.get(&name) {
                tool.bind_runtime(task.id, command_sender.downgrade());
            }
        }

        let agent_loop = AgentLoop::new(
            task,
            Arc::clone(&self.llm_client),
            tool_registry,
            event_sender,
            command_receiver,
            event_throttle_ms,
            simplified_tool_events,
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
        );

        let task_join_handle = tokio::spawn(async move { agent_loop.run().await });

        (event_receiver, command_sender, task_join_handle)
    }
}
