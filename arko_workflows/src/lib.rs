//! Multi-agent workflows built on `arko_agents` and `arko_ai`.
//!
//! A workflow is ordinary Rust code. [`Workflow`] runs the agents and gives them shared
//! memory, messaging, budgets, lifecycle hooks, context-freshness tracking, completion
//! verification, durable steps and persistence, and your code decides which agents run
//! and in what order.
//!
//! ```rust,no_run
//! use arko_workflows::prelude::*;
//! use arko_ai::LLMClient;
//! use arko_agents::AgentConfig;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), WorkflowError> {
//! let client = Arc::new(LLMClient::new().with_anthropic_api_key("sk-..."));
//!
//! let workflow = Workflow::builder()
//!     .name("my-workflow")
//!     .workspace("./project")
//!     .llm_client(client)
//!     .agent("coder", AgentConfig::load_from_path("configs/coder.json".as_ref()).unwrap())
//!     .budget(5.0)
//!     .build()?;
//!
//! let result = workflow.run_agent("coder", "Implement the auth module").await?;
//! println!("Cost: ${:.6}", result.total_cost());
//! # Ok(())
//! # }
//! ```
//!
//! The crate's README introduces it, and `docs/arko_workflows-guide.md` and
//! `arko_workflows/INDEX.md` in the repository cover it in depth.

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
pub mod notes;
pub mod persistence;
pub mod policy;
pub mod services;
pub mod subagents;
#[cfg(test)]
pub(crate) mod test_support;
pub mod user_input;
pub mod verification;
pub mod voting;
pub mod workflow;

#[cfg(feature = "python")]
pub mod python;

pub use error::WorkflowError;
pub use workflow::cancel::{CancelCause, Cancellation, WorkflowCancelHandle};
pub use workflow::config::{SupportedPlatform, WorkflowDependency, WorkflowMetadata};
pub use workflow::event_stream::EventReceiver;
pub use workflow::result::{AgentResult, AgentRunStatus, WorkflowEvent, WorkflowResult};
pub use workflow::stop::{
    Restart, RunStopIntent, StopKind, StopRecord, StopRequest, StoppedRun, WrapUp, stop_on_signals,
};
pub use workflow::{AgentHandle, ContinueOptions, RunOptions, Workflow, WorkflowBuilder};

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::budget::{
        AddCostResult, BudgetExceededPolicy, BudgetManager, CostEntry, CostLedger, CostSource,
        CostTotals, PlanBudget, PlanLimits, PlanUsageSpent, PlanWindow,
    };
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
    pub use crate::memory::{
        AccessScope, MemoryConfig, MemoryEntry, MemoryTier, MemoryWrite, WriteMode,
    };
    pub use crate::messaging::registry::{AgentStatus, WaitReason};
    pub use crate::messaging::wait::WaitForMessageTimeout;
    pub use crate::messaging::{AgentMessage, MessageAttachment, MessagingPolicy};
    pub use crate::notes::{
        Note, NoteAudience, NoteImage, NoteKind, NoteSource, NotesHandle, PostedNote,
    };
    pub use crate::persistence::{CheckpointRetention, EventLog, PersistenceConfig};
    pub use crate::policy::{
        AgentPause, AgentResume, ApprovalPolicy, ErrorRecoveryPolicy, ToolApprovalHandler,
        ToolApprovalOutcome, ToolApprovalRequest,
    };
    pub use crate::services::{WorkflowServices, WorkflowServicesConfig};
    pub use crate::subagents::{
        GitWorktreeProvider, IsolatedWorkspace, MailItem, MailKind, MailSender, StartRequest,
        SubagentEvent, SubagentEventKind, SubagentHook, SubagentRecord, SubagentState,
        SubagentsConfig, WorkspaceProvider, WorkspaceRequest,
    };
    pub use crate::user_input::{UserInputHandler, UserInputRequest};
    pub use crate::verification::{
        CompletionVerifierConfig, RepairContextConfig, VERIFIER_OUTPUT_CONTRACT,
        VerificationDecision, VerificationKind, VerificationRecord,
    };
    pub use crate::voting::{BallotConfig, BallotResult, CastVote, TieBreak, VoteOption};
    pub use crate::workflow::cancel::{CancelCause, Cancellation, WorkflowCancelHandle};
    pub use crate::workflow::config::{SupportedPlatform, WorkflowDependency, WorkflowMetadata};
    pub use crate::workflow::event_stream::EventReceiver;
    pub use crate::workflow::result::{AgentResult, AgentRunStatus, WorkflowEvent, WorkflowResult};
    pub use crate::workflow::stop::{
        Restart, RunStopIntent, StopKind, StopRecord, StopRequest, StoppedRun, WrapUp,
        stop_on_signals,
    };
    pub use crate::workflow::{ContinueOptions, RunOptions, Workflow, WorkflowBuilder};
}
