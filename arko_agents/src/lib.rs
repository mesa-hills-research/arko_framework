//! Tool-using agents built on `arko_ai`.
//!
//! An [`AgentConfig`] describes an agent with its role and instructions, model, tools, and
//! approval and error policies. A [`Task`] holds one run's conversation and state.
//! [`AgentRuntime`] runs a task, streams [`AgentEvent`]s and takes [`UserCommand`]s for
//! approvals, answers, error recovery, context updates, cancellation and suspension.
//! [`ToolRegistry`] holds the built-in and custom tools an agent can be offered.
//!
//! The crate's README has a first example, and `docs/arko_agents-guide.md` in the
//! repository covers tools, approvals, recovery, Claude Code models and conversation
//! handover.

pub mod agent_config;
pub mod claude_code;
#[cfg(all(test, windows))]
mod console_probe;
pub mod error;
mod file_tree;
pub mod handover;
pub mod mcp;
pub mod parsing;
pub mod runtime;
pub mod skill;
pub mod snapshot;
pub mod stop;
pub mod task;
#[cfg(any(unix, windows))]
pub mod terminal;
pub mod tool;
pub mod utils;

pub use agent_config::{
    AgentConfig, AgentConfigBuilder, AgentLLMConfig, AgentToolConfig, BuiltInEnvironmentDetail,
    CacheMonitorConfig, DEFAULT_TOOL_TIMEOUT_SECONDS, DeveloperDefinableEnvironmentDetail,
    DynamicContentProvider, EnvironmentDetailPlacement, EnvironmentDetailsConfig,
    ErrorRecoveryConfig, FileTreeMode, LLMCapacityRetryConfig, LLMErrorPolicy,
    LLMRateLimitRetryConfig, LLMRetryConfig, MonitoringConfig, PerToolConfigOverride,
    PlanUsageAction, PlanUsageThreshold, ReminderInjectionFrequency, ToolCallingMode,
    ToolErrorPolicy, ToolResultLimits, UsageLimitPolicy, UserReminder,
};
pub use claude_code::{ClaudeCodeSettings, ClaudeCodeTools};
pub use error::ArkoAgentsError;
pub use handover::{
    HandoverConfig, HandoverHook, HandoverRecord, HandoverRequest, HandoverSeed, HandoverThreshold,
};
pub use mcp::{McpConfig, McpManager, McpServerConfig};
pub use parsing::{
    ParsedToolCall, RawParserEvent, ToolAssemblyError, ToolCallGroup, ToolCallParser,
    ToolParsingConfig, assemble_tool_calls, strip_output_based_only_blocks,
};
pub use runtime::{
    AgentRuntime, CacheRefresh, ContextImage, DEFAULT_BYTES_PER_TOKEN, HostDecisions, TurnContext,
    TurnContextSource, calibrated_bytes_per_token,
    command::{ToolApprovalDecision, UserCommand},
    event::{AgentEvent, ShortenReason},
};
pub use skill::{
    AutoLoadedSkillContent, InvalidSkillPolicy, LOAD_SKILL_TOOL_NAME, LoadSkillTool, SkillManager,
    SkillPromptContext, SkillSource, SkillSummary, SkillsConfig,
};
pub use snapshot::TaskSnapshotSink;
pub use stop::{
    ChildStop, STOP_FILE_ENV, StopAwareCommand, StopAwareOutput, StopSignal, StopSource,
    kill_stop_aware_commands,
};
pub use task::{
    ApprovalReason, DeferredToolError, ErrorRecoveryCause, PendingToolCallInfo,
    PlanUsageWindowMark, RefusalResolution, RefusedInput, RefusedMessage, RefusedToolResult,
    RunningToolCall, Task, TaskParent, TaskState, UserTurn, UserTurnKind,
};
pub use tool::{
    ContextSignal, ContextSource, ImageRoom, InterceptedResult, InterceptionFailurePolicy,
    InterceptorConfig, TaskContext, Tool, ToolCallContext, ToolDefinition, ToolError,
    ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware, ToolExecutionOutcome,
    ToolImage, ToolImageSink, ToolOutput, ToolParameterDefinition, ToolParameterType, ToolRecovery,
    ToolRegistry, ToolResultInterceptor, ToolTypeDetails, builtins,
};

pub use arko_ai::ModelCallUsage;
