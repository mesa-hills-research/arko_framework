use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;

use arko_agents::skill::SkillsConfig;
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
    /// Who may message whom with `send_message`. `None` disables it.
    pub messaging: Option<crate::messaging::MessagingPolicy>,
    /// How long `wait_for_message` waits, for the workflow and per agent.
    pub(crate) wait_timeouts: crate::messaging::wait::WaitTimeouts,
    pub resource_tracker: Option<ResourceTracker>,
    pub hooks: Arc<HookRegistry>,
    /// The agents' configurations as registered. `Workflow::set_agent_skills`
    /// changes an agent's skills after that, which this map doesn't show.
    /// Inside the crate, read them through `registered_config`.
    pub agent_configs: HashMap<String, AgentConfig>,
    /// The skills configurations the host set after the build, by agent.
    /// `None` is no skills.
    pub(crate) agent_skills: RwLock<HashMap<String, Option<SkillsConfig>>>,
    /// The model settings each run started or went on with, by run, which
    /// may not be its agent's registered ones. Read through `run_llm_config`.
    pub(crate) run_llm_configs: RwLock<HashMap<uuid::Uuid, arko_agents::AgentLLMConfig>>,
    pub global_tools: Vec<Arc<dyn arko_agents::tool::Tool>>,
    pub agent_tools: HashMap<String, Vec<Arc<dyn arko_agents::tool::Tool>>>,
    /// The (agent, tool name) pairs of `agent_tools` that new and
    /// reconfigured runs are not offered.
    pub(crate) resume_only_tools: HashSet<(String, String)>,
    pub agent_results: RwLock<Vec<AgentResult>>,
    /// Interceptors applied to every agent's tool results.
    pub global_interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
    /// Interceptors applied only to a specific agent's tool results.
    pub agent_interceptors: HashMap<String, Vec<Arc<dyn ToolResultInterceptor>>>,
    /// The host's handover thresholds, by agent: in place of the
    /// configuration's (`WorkflowBuilder::agent_handover_threshold`).
    pub(crate) handover_thresholds: HashMap<String, arko_agents::HandoverThreshold>,
    /// Receives the images tool calls return (`WorkflowBuilder::tool_image_sink`).
    pub(crate) tool_image_sink: Option<Arc<dyn arko_agents::ToolImageSink>>,
    /// The terminal manager calls move to the background with
    /// (`WorkflowBuilder::background_jobs`).
    #[cfg(any(unix, windows))]
    pub(crate) background_jobs: Option<Arc<arko_agents::terminal::TerminalManager>>,
    /// Failure policy used when composing interceptors for agent runs.
    pub interception_failure_policy: InterceptionFailurePolicy,
    pub completion_verifiers: Vec<crate::verification::CompletionVerifierConfig>,
    pub verification_attempts: parking_lot::Mutex<HashMap<(uuid::Uuid, String), (u32, u32)>>,
    pub(crate) verification_baselines:
        parking_lot::Mutex<HashMap<(uuid::Uuid, String), crate::verification::RepairBaseline>>,
    pub verification_records: RwLock<Vec<crate::verification::VerificationRecord>>,
    /// Failed reviews per target task and verifier.
    pub(crate) verification_failures: parking_lot::Mutex<HashMap<(uuid::Uuid, String), u32>>,
    /// Target tasks a verifier stopped because it failed: the verifier's
    /// name and why.
    pub(crate) verifier_failures: parking_lot::Mutex<HashMap<uuid::Uuid, (String, String)>>,
    pub distillation_configs: Vec<crate::distillation::DistillationConfig>,
    pub distillation_records: RwLock<Vec<crate::distillation::DistillationRecord>>,
    pub workflow_services: std::sync::OnceLock<crate::services::WorkflowServices>,
    pub paused: std::sync::atomic::AtomicBool,
    pub resume_notify: tokio::sync::Notify,
    /// Agent statuses: rosters, start notices, and `list_recipients`.
    pub(crate) registry: crate::messaging::registry::AgentRegistry,
    /// The notes posted to the agents (`Workflow::post_note`).
    pub(crate) notes: crate::notes::NoteBoard,
    /// The background subagents, their mailboxes and their drivers.
    pub(crate) subagents: crate::subagents::Board,
    /// Whether and why the run was stopped (cancelled or aborted by a hook).
    pub(crate) run_control: Arc<crate::workflow::cancel::RunControl>,
    /// The run's persistence, when configured: changes the host makes
    /// while agents run (such as a stop of one run) are saved with it.
    pub(crate) persistence: Option<crate::persistence::PersistenceManager>,
    /// Reads stop requests from the control folder, when enabled.
    pub(crate) control_stops: crate::notes::control::StopReader,
    pub user_input_handler: Option<Arc<dyn crate::user_input::UserInputHandler>>,
    /// `run_agents_parallel`'s concurrency cap and fail-fast setting.
    pub(crate) parallel: ParallelSettings,
    /// How agents' tool-approval requests are answered.
    pub approval_policy: crate::policy::ApprovalPolicy,
    /// How agents' error-recovery pauses are answered.
    pub error_recovery_policy: crate::policy::ErrorRecoveryPolicy,
    pub(crate) recovery_waits: crate::policy::RecoveryWaits,
    /// Why each approved call (by call ID) of a run (task ID) awaited
    /// approval, until it has run, won't run, or its run ended: for the tool
    /// hooks.
    pub(crate) approval_reasons:
        parking_lot::Mutex<HashMap<uuid::Uuid, (uuid::Uuid, arko_agents::ApprovalReason)>>,
    /// The agents whose auto-approvals under `ApproveAll` were warned about.
    pub(crate) approval_warnings: parking_lot::Mutex<std::collections::HashSet<String>>,
    pub ballots: Option<Arc<crate::voting::BallotStore>>,
    /// The agents offered `cast_vote` (see `WorkflowBuilder::enable_voting`).
    pub voters: std::collections::HashSet<String>,
    pub event_tx: tokio::sync::broadcast::Sender<crate::workflow::result::WorkflowEvent>,
    /// The lossless subscribers of the events.
    pub(crate) event_hub: crate::workflow::event_stream::EventHub,
}

/// How `run_agents_parallel` runs its agents.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ParallelSettings {
    /// At most this many at once. `None` means all at once.
    pub max_agents: Option<usize>,
    /// The first failure stops the others.
    pub fail_fast: bool,
}

impl SharedContext {
    /// `agent`'s registered configuration as it is now, with the skills the
    /// host set last.
    pub(crate) fn registered_config(&self, agent: &str) -> Option<AgentConfig> {
        let mut config = self.agent_configs.get(agent)?.clone();
        config.skills_config = self.registered_skills(agent);
        Some(config)
    }

    /// The model settings run `task_id` of `agent` works with, which may
    /// not be the agent's registered ones
    /// ([`RunOptions::config`](crate::RunOptions::config)). The registered
    /// ones for a run that hasn't started.
    pub(crate) fn run_llm_config(
        &self,
        agent: &str,
        task_id: uuid::Uuid,
    ) -> Option<arko_agents::AgentLLMConfig> {
        if let Some(config) = self.run_llm_configs.read().get(&task_id) {
            return Some(config.clone());
        }
        Some(self.agent_configs.get(agent)?.llm_config.clone())
    }

    /// Records the model settings run `task_id` works with, as it starts or
    /// goes on, for [`run_llm_config`](Self::run_llm_config).
    pub(crate) fn note_run_llm_config(
        &self,
        task_id: uuid::Uuid,
        config: &arko_agents::AgentLLMConfig,
    ) {
        self.run_llm_configs.write().insert(task_id, config.clone());
    }

    /// The skills configuration `agent` is registered with now.
    pub(crate) fn registered_skills(&self, agent: &str) -> Option<SkillsConfig> {
        match self.agent_skills.read().get(agent) {
            Some(skills) => skills.clone(),
            None => self.agent_configs.get(agent)?.skills_config.clone(),
        }
    }

    /// The agents `agent_name` may message with `send_message`, among registered
    /// agents other than itself that the messaging policy permits. Empty when
    /// messaging is disabled. The agent is then not offered the tool.
    pub fn message_recipients(&self, agent_name: &str) -> Vec<String> {
        self.messaging.as_ref().map_or_else(Vec::new, |policy| {
            policy.recipients_of(agent_name, &self.messages.roster())
        })
    }

    /// Whether messages to a finished agent are kept for its next run. See
    /// [`MessagingPolicy::keep_for_finished`](crate::messaging::MessagingPolicy::keep_for_finished).
    pub fn keeps_messages_for_finished(&self) -> bool {
        self.messaging
            .as_ref()
            .is_some_and(|policy| policy.keep_for_finished)
    }

    /// The agents that may message `agent_name` with `send_message` are
    /// registered agents other than itself whose messaging policy permits it.
    /// Empty when messaging is disabled. The agent is then not offered
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

impl SharedContext {
    /// With the task journal, writes the live checkpoint soon (spawned)
    /// while agents run, so that a change the host made to the running
    /// workflow (`what`) is saved before their next turns.
    pub(crate) fn save_live_checkpoint_soon(self: &Arc<Self>, what: &'static str) {
        let Some(persistence) = self.persistence.clone() else {
            return;
        };
        if !persistence.has_live_runs() {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let context = self.clone();
        runtime.spawn(async move {
            if let Err(error) = persistence.write_live_checkpoint(&context, false).await {
                log::warn!(
                    "Workflow '{}': could not save {what}: {error}",
                    context.workflow_name
                );
            }
        });
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
            .field("approval_policy", &self.approval_policy)
            .field("error_recovery_policy", &self.error_recovery_policy)
            .field("cancellation", &self.run_control.cancellation())
            .finish_non_exhaustive()
    }
}
