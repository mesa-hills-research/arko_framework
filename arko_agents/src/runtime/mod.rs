use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use self::command::UserCommand;
use self::event::AgentEvent;
use self::r#loop::AgentLoop;
pub use self::turn_context::TurnContextSource;
use crate::error::ArkoAgentsError;
use crate::handover::HandoverHook;
use crate::skill::SkillPromptContext;
use crate::task::Task;
use crate::tool::{InterceptorConfig, ToolExecutionMiddleware, ToolRegistry};
use arko_ai::LLMClient;

pub mod command;
pub mod event;
mod r#loop;
mod monitor;
mod retry;
mod tool_run;
pub mod turn_context;

const DEFAULT_EVENT_CHANNEL_SIZE: usize = 100;
const DEFAULT_COMMAND_CHANNEL_SIZE: usize = 10;

/// Whether the host decides every wait of a run itself: every tool
/// approval, answer and error-recovery decision the task waits for is
/// decided by the host (or the host stops the run), however long it takes.
/// While it is set, the task's
/// [`user_command_timeout_seconds`](crate::agent_config::AgentConfig::user_command_timeout_seconds)
/// doesn't apply: no wait of the run is timed. A workflow sets it for the
/// runs it drives.
///
/// A run-level setting: it is not part of the task and is not saved, so a
/// task run later without it is timed as configured. Clones share the
/// setting, and the host may change it while the run runs: turned on during
/// a wait, that wait is no longer timed; turned off, the waits that start
/// afterwards are.
#[derive(Clone, Debug, Default)]
pub struct HostDecisions(Arc<AtomicBool>);

impl HostDecisions {
    /// `true`: the host decides every wait of the run.
    pub fn new(host_decides: bool) -> Self {
        Self(Arc::new(AtomicBool::new(host_decides)))
    }

    /// Changes the setting for every clone.
    pub fn set(&self, host_decides: bool) {
        self.0.store(host_decides, Ordering::Release);
    }

    /// Whether the host decides every wait of the run.
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The main runtime for executing agent tasks.
pub struct AgentRuntime {
    llm_client: Arc<LLMClient>,
    turn_context: Option<Arc<dyn TurnContextSource>>,
    host_decisions: HostDecisions,
    handover_hook: Option<Arc<dyn HandoverHook>>,
}

impl AgentRuntime {
    /// Creates a new AgentRuntime with the given LLM client.
    pub fn new(llm_client: Arc<LLMClient>) -> Self {
        Self {
            llm_client,
            turn_context: None,
            host_decisions: HostDecisions::default(),
            handover_hook: None,
        }
    }

    /// Runs tasks with `decisions`: while it is set, the host decides every
    /// wait of the runs itself, and their command timeout doesn't apply.
    /// See [`HostDecisions`].
    pub fn with_host_decisions(mut self, decisions: HostDecisions) -> Self {
        self.host_decisions = decisions;
        self
    }

    /// Asks `source` for context before each model request of the tasks this
    /// runtime runs; see [`TurnContextSource`].
    pub fn with_turn_context(mut self, source: Arc<dyn TurnContextSource>) -> Self {
        self.turn_context = Some(source);
        self
    }

    /// Calls `hook` when a task this runtime runs hands its conversation
    /// over; see [`HandoverHook`].
    pub fn with_handover_hook(mut self, hook: Arc<dyn HandoverHook>) -> Self {
        self.handover_hook = Some(hook);
        self
    }

    /// Runs a task with the given agent configuration and tool registry.
    ///
    /// The task runs only the registry's tools its configuration offers
    /// ([`AgentToolConfig::offered_tool_names`](crate::agent_config::AgentToolConfig::offered_tool_names));
    /// those tools are bound to the task's command channel
    /// ([`Tool::bind_runtime`](crate::tool::Tool::bind_runtime)).
    ///
    /// An optional `InterceptorConfig` enables tool result interception,
    /// allowing results to be transformed (e.g., distilled) before they
    /// enter the agent's conversation. `complete_task` and `ask_user`
    /// results are never intercepted.
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

        let mut agent_loop = AgentLoop::new(
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
        agent_loop.turn_context = self.turn_context.clone();
        agent_loop.host_decisions = self.host_decisions.clone();
        agent_loop.handover_hook = self.handover_hook.clone();
        agent_loop.bind_tools(&command_sender);

        let task_join_handle = tokio::spawn(async move { agent_loop.run().await });

        (event_receiver, command_sender, task_join_handle)
    }
}
