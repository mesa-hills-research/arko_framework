use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::TaskContext;
use super::error::ToolError;

/// The outcome of a tool result interception attempt.
pub enum InterceptedResult {
    /// The interceptor chose not to modify the result.
    Unchanged,
    /// The interceptor produced a replacement result.
    Replaced(String),
}

/// A trait for intercepting and potentially transforming tool results before
/// they enter the agent's conversation.
///
/// Implementations may perform arbitrary async work, including LLM calls
/// (e.g., for distillation/summarization of large tool outputs).
///
/// The interceptor receives the tool name, raw result, parameters, and task
/// context. It returns `Unchanged` to pass through or `Replaced(String)` to
/// substitute a new result.
///
/// `complete_task` results are never passed to interceptors.
#[async_trait]
pub trait ToolResultInterceptor: Send + Sync {
    async fn intercept(
        &self,
        tool_name: &str,
        raw_result: &str,
        params: &HashMap<String, Value>,
        task_context: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError>;
}

/// Policy for handling interceptor failures (e.g., a distillation LLM call fails).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum InterceptionFailurePolicy {
    /// Use the original (raw) tool result and continue. This is the default.
    #[default]
    UseRawResult,
    /// Retry interception up to `max_attempts` times, then fall back to raw result.
    RetryThenUseRawResult { max_attempts: u32 },
}

/// Bundles a tool result interceptor with its failure handling policy.
pub struct InterceptorConfig {
    pub interceptor: Arc<dyn ToolResultInterceptor>,
    pub failure_policy: InterceptionFailurePolicy,
}
