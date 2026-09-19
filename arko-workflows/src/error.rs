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

    #[error("Agent '{agent}' requires user input: {questions}")]
    UserInputRequired {
        agent: String,
        task_id: uuid::Uuid,
        questions: String,
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

    #[error("Workflow aborted by hook at {hook_point}: {reason}")]
    AbortedByHook { hook_point: String, reason: String },

    #[error("MESI conflict: agent '{agent}' write to '{path}' conflicts with '{modifier}'")]
    MesiConflict {
        agent: String,
        path: String,
        modifier: String,
    },

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

    #[error("Workflow cancelled")]
    Cancelled,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Hook error: {0}")]
    Hook(String),

    #[error("Tool error: {0}")]
    Tool(#[from] arko_agents::ToolError),

    #[error("Deliberation error: {0}")]
    Deliberation(String),
}
