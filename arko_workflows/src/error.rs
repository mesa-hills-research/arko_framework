use thiserror::Error;

/// Errors that can occur during workflow construction or execution.
#[derive(Error, Debug)]
pub enum WorkflowError {
    #[error("Agent error: {0}")]
    Agent(#[from] arko_agents::ArkoAgentsError),

    #[error("LLM error: {0}")]
    Llm(#[from] arko_ai::Error),

    #[error("Builder error: missing required field '{field}'")]
    BuilderMissing { field: String },

    #[error("Builder validation error: {0}")]
    BuilderInvalid(String),

    #[error("Agent '{0}' not found in workflow")]
    AgentNotFound(String),

    #[error("Agent '{agent}' was interrupted: {reason}")]
    AgentInterrupted { agent: String, reason: String },

    /// The agent's task was cancelled while the workflow run was not (for
    /// example by a custom tool or middleware).
    #[error("Agent '{agent}' was cancelled: {reason}")]
    AgentCancelled { agent: String, reason: String },

    /// The agent asked the user a question and no user input handler is
    /// set: it is suspended, recorded with status `WaitingForInput`; answer
    /// with `Workflow::continue_agent`. A pause, not a failure.
    #[error("Agent '{agent}' requires user input: {questions}")]
    UserInputRequired {
        agent: String,
        task_id: uuid::Uuid,
        questions: String,
    },

    /// The agent paused for a decision the workflow's approval or error
    /// recovery policy leaves to the caller: it is suspended, recorded with
    /// status `Paused`; continue it with `Workflow::resume_agent`. A pause,
    /// not a failure.
    #[error("Agent '{agent}' paused, {pause}")]
    AgentPaused {
        agent: String,
        task_id: uuid::Uuid,
        pause: crate::policy::AgentPause,
    },

    /// A completion verifier couldn't review the agent's completion (its run
    /// failed, or its verdict stayed unusable) more times than its
    /// `max_review_retries` allow. The completion was neither accepted nor
    /// rejected: the agent's task is paused (status `Paused`, an
    /// error-recovery pause with the cause `Middleware`). Continue it with
    /// [`Workflow::continue_agent`](crate::Workflow::continue_agent) on
    /// [`Workflow::agent_result`](crate::Workflow::agent_result)`(task_id)`:
    /// the prompt is its next message, and its next `complete_task` is
    /// reviewed again. [`Workflow::resume_agent`](crate::Workflow::resume_agent)
    /// can also continue or fail it.
    #[error("Completion verifier '{verifier}' could not review agent '{agent}': {reason}")]
    VerifierFailed {
        verifier: String,
        agent: String,
        task_id: uuid::Uuid,
        reason: String,
    },

    #[error("Memory tier '{0}' not found")]
    MemoryTierNotFound(String),

    #[error("Memory access denied: agent '{agent}' cannot {action} tier '{tier}'")]
    MemoryAccessDenied {
        agent: String,
        tier: String,
        action: String,
    },

    #[error("Budget exceeded: spent ${spent:.6}, limit ${limit:.6}")]
    BudgetExceeded { spent: f64, limit: f64 },

    #[error("Agent '{agent}' budget exceeded: spent ${spent:.6}, limit ${limit:.6}")]
    AgentBudgetExceeded {
        agent: String,
        spent: f64,
        limit: f64,
    },

    #[error("Invalid cost amount {0}: costs must be finite and non-negative")]
    InvalidCost(f64),

    /// A hook returned `HookResult::Abort`; the run is stopped.
    #[error("Workflow aborted by hook at {hook_point}: {reason}")]
    AbortedByHook { hook_point: String, reason: String },

    #[error("Persistence error: {0}")]
    Persistence(String),

    #[error("Durable steps require workflow persistence to be configured")]
    DurableUnavailable,

    #[error("Durable step definition changed for '{step}': {details}")]
    DurableDefinitionMismatch { step: String, details: String },

    #[error("Durable step '{step}' is in doubt (operation id: {operation_id})")]
    DurableInDoubt { step: String, operation_id: String },

    #[error(
        "Memory tier '{tier}' would grow to {attempted} bytes, exceeding its {limit}-byte limit"
    )]
    MemoryLimitExceeded {
        tier: String,
        attempted: usize,
        limit: usize,
    },

    #[error("Durable step '{step}' permanently failed: {reason}")]
    DurablePermanentlyFailed { step: String, reason: String },

    #[error("Durable step '{step}' exhausted its retry limit")]
    DurableAttemptsExhausted { step: String },

    /// A `WorkflowCancelHandle` cancelled the run.
    #[error("Workflow cancelled")]
    Cancelled,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Tool error: {0}")]
    Tool(#[from] arko_agents::ToolError),

    #[error("Deliberation error: {0}")]
    Deliberation(String),
}
