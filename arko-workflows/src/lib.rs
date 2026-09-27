/// # arko_workflows
///
/// Multi-agent workflow orchestration built on `arko_agents`.
///
/// `arko_workflows` enables composing multiple LLM agent runs into cohesive
/// workflows with shared memory, inter-agent messaging, budget management,
/// lifecycle hooks, context freshness tracking, and persistence.
///
/// ## Quick Start
///
/// ```rust,no_run
/// use arko_workflows::prelude::*;
/// use arko_ai::LLMClient;
/// use arko_agents::AgentConfig;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), WorkflowError> {
/// let client = Arc::new(LLMClient::new().with_anthropic_api_key("sk-..."));
///
/// let workflow = Workflow::builder()
///     .name("my-workflow")
///     .workspace("./project")
///     .llm_client(client)
///     .agent("coder", AgentConfig::load_from_path("configs/coder.json".as_ref()).unwrap())
///     .budget(5.0)
///     .build()?;
///
/// let result = workflow.run_agent("coder", "Implement the auth module").await?;
/// println!("Cost: ${:.6}", result.total_cost());
/// # Ok(())
/// # }
/// ```
pub mod budget;
pub mod context;
mod digest;
pub mod distillation;
pub mod durable;
pub mod error;
pub mod freshness;
pub mod hooks;
pub mod memory;
pub mod messaging;
pub mod persistence;
pub mod services;
pub mod user_input;
pub mod verification;
pub mod voting;
pub mod workflow;

#[cfg(feature = "python")]
pub mod python;

pub use error::WorkflowError;
pub use workflow::config::{SupportedPlatform, WorkflowDependency, WorkflowMetadata};
pub use workflow::result::{AgentResult, AgentRunStatus, WorkflowEvent, WorkflowResult};
pub use workflow::{AgentHandle, Workflow, WorkflowBuilder};

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::budget::{AddCostResult, BudgetManager, CostEntry, CostLedger, CostSource};
    pub use crate::distillation::{DistillationConfig, DistillationRecord};
    pub use crate::durable::{
        DurableRetryPolicy, DurableStepContext, DurableStepSpec, DurableStepStatus, InDoubtContext,
        InDoubtResolution, ResolveInDoubt,
    };
    pub use crate::error::WorkflowError;
    pub use crate::freshness::{
        ActorId, ContextFreshnessConfig, FreshnessConflict, ResourceId, ResourceTracker,
        ToolResourceEffect,
    };
    pub use crate::hooks::{HookCallback, HookPoint, HookResult};
    pub use crate::memory::{AccessScope, MemoryConfig, MemoryEntry, MemoryTier, WriteMode};
    pub use crate::messaging::{AgentMessage, MessageAttachment};
    pub use crate::persistence::PersistenceConfig;
    pub use crate::services::{WorkflowServices, WorkflowServicesConfig};
    pub use crate::user_input::{UserInputHandler, UserInputRequest};
    pub use crate::verification::{
        CompletionVerifierConfig, RepairContextConfig, VERIFIER_OUTPUT_CONTRACT,
        VerificationDecision, VerificationKind, VerificationRecord,
    };
    pub use crate::voting::{BallotConfig, BallotResult, CastVote, TieBreak, VoteOption};
    pub use crate::workflow::config::{SupportedPlatform, WorkflowDependency, WorkflowMetadata};
    pub use crate::workflow::result::{AgentResult, AgentRunStatus, WorkflowEvent, WorkflowResult};
    pub use crate::workflow::{Workflow, WorkflowBuilder};
}
