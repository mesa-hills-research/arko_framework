use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;

use arko_agents::{AgentConfig, InterceptionFailurePolicy, ToolResultInterceptor};

use crate::budget::BudgetManager;
use crate::freshness::ResourceTracker;
use crate::hooks::HookRegistry;
use crate::memory::MemoryStore;
use crate::messaging::MessageBus;
use crate::workflow::result::AgentResult;

/// Shared state for a workflow, accessed by the engine, tools, and hooks.
pub struct SharedContext {
    pub workflow_run_id: uuid::Uuid,
    pub workflow_name: String,
    pub workflow_metadata: crate::workflow::config::WorkflowMetadata,
    pub workspace: PathBuf,
    pub budget: BudgetManager,
    pub durable_store: Option<Arc<crate::durable::DurableStore>>,
    pub memory: MemoryStore,
    pub messages: MessageBus,
    /// Who may message whom with `send_message`; `None` disables it.
    pub messaging: Option<crate::messaging::MessagingPolicy>,
    /// How long `wait_for_message` waits, for the workflow and per agent.
    pub(crate) wait_timeouts: crate::messaging::wait::WaitTimeouts,
    pub resource_tracker: Option<ResourceTracker>,
    pub hooks: Arc<HookRegistry>,
    pub agent_configs: HashMap<String, AgentConfig>,
    pub global_tools: Vec<Arc<dyn arko_agents::tool::Tool>>,
    pub agent_tools: HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>,
    pub agent_results: RwLock<Vec<AgentResult>>,
    /// Interceptors applied to every agent's tool results.
    pub global_interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
    /// Interceptors applied only to a specific agent's tool results.
    pub agent_interceptors: HashMap<String, Vec<Arc<dyn ToolResultInterceptor>>>,
    /// Failure policy used when composing interceptors for agent runs.
    pub interception_failure_policy: InterceptionFailurePolicy,
    pub completion_verifiers: Vec<crate::verification::CompletionVerifierConfig>,
    pub verification_attempts: parking_lot::Mutex<HashMap<(uuid::Uuid, String), (u32, u32)>>,
    pub(crate) verification_baselines:
        parking_lot::Mutex<HashMap<(uuid::Uuid, String), crate::verification::RepairBaseline>>,
    pub verification_records: RwLock<Vec<crate::verification::VerificationRecord>>,
    pub distillation_configs: Vec<crate::distillation::DistillationConfig>,
    pub distillation_records: RwLock<Vec<crate::distillation::DistillationRecord>>,
    pub workflow_services: std::sync::OnceLock<crate::services::WorkflowServices>,
    pub paused: std::sync::atomic::AtomicBool,
    pub resume_notify: tokio::sync::Notify,
    /// Agent statuses: rosters, start notices, and `list_recipients`.
    pub(crate) registry: crate::messaging::registry::AgentRegistry,
    /// Whether and why the run was stopped (cancelled or aborted by a hook).
    pub(crate) run_control: Arc<crate::workflow::cancel::RunControl>,
    pub user_input_handler: Option<Arc<dyn crate::user_input::UserInputHandler>>,
    pub ballots: Option<Arc<crate::voting::BallotStore>>,
    pub event_tx: tokio::sync::broadcast::Sender<crate::workflow::result::WorkflowEvent>,
}

impl SharedContext {
    /// The agents `agent_name` may message with `send_message`: registered
    /// agents other than itself that the messaging policy permits. Empty when
    /// messaging is disabled; the agent is then not offered the tool.
    pub fn message_recipients(&self, agent_name: &str) -> Vec<String> {
        self.messaging.as_ref().map_or_else(Vec::new, |policy| {
            policy.recipients_of(agent_name, &self.messages.roster())
        })
    }

    /// The agents that may message `agent_name` with `send_message`:
    /// registered agents other than itself whose messaging policy permits it.
    /// Empty when messaging is disabled; the agent is then not offered
    /// `wait_for_message`.
    pub fn message_senders(&self, agent_name: &str) -> Vec<String> {
        self.messaging.as_ref().map_or_else(Vec::new, |policy| {
            self.messages
                .roster()
                .into_iter()
                .filter(|sender| policy.permits(sender, agent_name))
                .collect()
        })
    }

    /// The agents whose starts and finishes end `agent_name`'s
    /// `wait_for_message`: those it may message and those that may message
    /// it, sorted by name.
    pub(crate) fn status_peers(&self, agent_name: &str) -> Vec<String> {
        let mut peers = self.message_recipients(agent_name);
        peers.extend(self.message_senders(agent_name));
        peers.sort();
        peers.dedup();
        peers
    }

    /// The current statuses of the agents `agent_name` may message.
    pub fn recipient_statuses(
        &self,
        agent_name: &str,
    ) -> Vec<(String, crate::messaging::registry::AgentStatus)> {
        self.registry.statuses(
            &self.message_recipients(agent_name),
            self.paused.load(std::sync::atomic::Ordering::Acquire),
        )
    }

    /// The current status of `agent`.
    pub fn recipient_status(&self, agent: &str) -> crate::messaging::registry::AgentStatus {
        self.registry
            .statuses(
                &[agent.to_string()],
                self.paused.load(std::sync::atomic::Ordering::Acquire),
            )
            .remove(0)
            .1
    }
}

impl fmt::Debug for SharedContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedContext")
            .field("workflow_name", &self.workflow_name)
            .field("workflow_run_id", &self.workflow_run_id)
            .field("workflow_metadata", &self.workflow_metadata)
            .field("workspace", &self.workspace)
            .field(
                "agent_configs",
                &self.agent_configs.keys().collect::<Vec<_>>(),
            )
            .field("durability_enabled", &self.durable_store.is_some())
            .field("global_tools", &self.global_tools.len())
            .field(
                "context_freshness_enabled",
                &self.resource_tracker.is_some(),
            )
            .field("agent_tools", &self.agent_tools.keys().collect::<Vec<_>>())
            .field("agent_results_count", &self.agent_results.read().len())
            .field("global_interceptors", &self.global_interceptors.len())
            .field(
                "agent_interceptors",
                &self.agent_interceptors.keys().collect::<Vec<_>>(),
            )
            .field(
                "interception_failure_policy",
                &self.interception_failure_policy,
            )
            .field("completion_verifiers", &self.completion_verifiers.len())
            .field("distillation_configs", &self.distillation_configs.len())
            .field("cancellation", &self.run_control.cancellation())
            .finish_non_exhaustive()
    }
}
