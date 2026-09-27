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
    pub user_input_handler: Option<Arc<dyn crate::user_input::UserInputHandler>>,
    pub ballots: Option<Arc<crate::voting::BallotStore>>,
    pub event_tx: tokio::sync::broadcast::Sender<crate::workflow::result::WorkflowEvent>,
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
            .finish_non_exhaustive()
    }
}
