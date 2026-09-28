//! Running one tool call: middleware, the tool itself under the host's time
//! limit, and result interception. A call's inputs are owned, so the
//! parallel-safe calls of a response can run at the same time while the
//! loop watches its command channel.

use crate::tool::{
    InterceptedResult, InterceptionFailurePolicy, TaskContext, Tool, ToolCallContext, ToolError,
    ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware, ToolExecutionOutcome,
    ToolOutput, ToolResultInterceptor,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Owned copies of what tools, middleware and interceptors see of the task.
pub(super) struct RunEnvironment {
    pub task_id: Uuid,
    pub agent_name: String,
    pub initial_task_prompt: String,
    pub working_directory: PathBuf,
    pub middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
}

impl RunEnvironment {
    fn task_context(&self) -> TaskContext<'_> {
        TaskContext {
            task_id: self.task_id,
            agent_name: &self.agent_name,
            initial_task_prompt: &self.initial_task_prompt,
            current_working_directory: &self.working_directory,
        }
    }

    fn execution_context(&self) -> ToolExecutionContext<'_> {
        ToolExecutionContext {
            task_id: self.task_id,
            agent_name: &self.agent_name,
            initial_task_prompt: &self.initial_task_prompt,
            current_working_directory: &self.working_directory,
        }
    }
}

/// A result interceptor and how its failures are handled.
#[derive(Clone)]
pub(super) struct Interception {
    pub interceptor: Arc<dyn ToolResultInterceptor>,
    pub policy: InterceptionFailurePolicy,
}

/// One call to run.
pub(super) struct ToolRun {
    pub tool_call_id: Uuid,
    pub tool_name: String,
    pub tool: Arc<dyn Tool>,
    pub params: HashMap<String, Value>,
    /// The host-side time limit; `None` runs without one.
    pub timeout: Option<Duration>,
    /// Output the host gave with the approval instead of running the tool.
    pub custom_output: Option<String>,
    /// How the result is intercepted, if it is.
    pub interception: Option<Interception>,
    /// The call may run at the same time as its parallel-safe neighbours.
    pub parallel_safe: bool,
    /// A cancellation or suspension stops the call while it runs
    /// ([`Tool::interruptible`]).
    pub interruptible: bool,
}

/// How a call ended.
pub(super) enum RunOutcome {
    /// The middleware rejected the call; it did not run.
    Rejected(String),
    /// The middleware failed (or, with `cancel`, cancelled) the task at the
    /// call; it did not run.
    Stop { reason: String, cancel: bool },
    /// The tool returned, or the host supplied its output. `effective` is
    /// the text after interception.
    Finished {
        output: ToolOutput,
        effective: String,
        intercepted: bool,
    },
    /// The tool failed, a timeout included.
    Failed(ToolError),
}

/// Runs one call. An error is the middleware's own failure.
pub(super) async fn run_tool(
    environment: Arc<RunEnvironment>,
    run: ToolRun,
) -> Result<RunOutcome, ToolError> {
    let call = ToolCallContext {
        tool_call_id: run.tool_call_id,
        tool_name: &run.tool_name,
        parameters: &run.params,
    };
    let execution = environment.execution_context();
    let middleware = environment.middleware.as_ref();
    let output = match &run.custom_output {
        Some(output) => {
            if let Some(middleware) = middleware {
                middleware
                    .after(
                        &execution,
                        &call,
                        ToolExecutionOutcome::CallerProvided(output),
                    )
                    .await?;
            }
            ToolOutput::text(output.clone())
        }
        None => {
            if let Some(middleware) = middleware {
                match middleware.before(&execution, &call).await? {
                    ToolExecutionDecision::Continue => {}
                    ToolExecutionDecision::Reject { message_to_agent } => {
                        return Ok(RunOutcome::Rejected(message_to_agent));
                    }
                    ToolExecutionDecision::FailTask { reason } => {
                        return Ok(RunOutcome::Stop {
                            reason,
                            cancel: false,
                        });
                    }
                    ToolExecutionDecision::CancelTask { reason } => {
                        return Ok(RunOutcome::Stop {
                            reason,
                            cancel: true,
                        });
                    }
                }
            }
            let context = environment.task_context();
            let execution_future = run.tool.execute_with_output(&context, run.params.clone());
            // Dropping the call at the limit stops it: a command it started
            // is killed with it.
            let result = match run.timeout {
                Some(limit) => tokio::time::timeout(limit, execution_future)
                    .await
                    .unwrap_or(Err(ToolError::TimedOut { limit })),
                None => execution_future.await,
            };
            match result {
                Ok(output) => {
                    if let Some(middleware) = middleware {
                        middleware
                            .after(
                                &execution,
                                &call,
                                ToolExecutionOutcome::Success(&output.text),
                            )
                            .await?;
                    }
                    output
                }
                Err(error) => {
                    if let Some(middleware) = middleware {
                        let error_text =
                            format!("Tool '{}' execution failed: {}", run.tool_name, error);
                        middleware
                            .after(&execution, &call, ToolExecutionOutcome::Error(&error_text))
                            .await?;
                    }
                    return Ok(RunOutcome::Failed(error));
                }
            }
        }
    };
    let (effective, intercepted) = match &run.interception {
        Some(interception) => {
            intercept(
                &environment,
                interception,
                &run.tool_name,
                &output.text,
                &run.params,
            )
            .await
        }
        None => (output.text.clone(), false),
    };
    Ok(RunOutcome::Finished {
        output,
        effective,
        intercepted,
    })
}

/// Applies `interception` to a result; returns the text for the
/// conversation and whether it differs from the raw result. A failed
/// interceptor is retried under its policy, and then the raw result is used.
async fn intercept(
    environment: &RunEnvironment,
    interception: &Interception,
    tool_name: &str,
    raw_result: &str,
    params: &HashMap<String, Value>,
) -> (String, bool) {
    let max_attempts = match interception.policy {
        InterceptionFailurePolicy::UseRawResult => 1,
        InterceptionFailurePolicy::RetryThenUseRawResult { max_attempts } => max_attempts.max(1),
    };
    let context = environment.task_context();
    for attempt in 1..=max_attempts {
        match interception
            .interceptor
            .intercept(tool_name, raw_result, params, &context)
            .await
        {
            Ok(InterceptedResult::Unchanged) => return (raw_result.to_string(), false),
            Ok(InterceptedResult::Replaced(new_result)) => return (new_result, true),
            Err(e) => {
                log::warn!(
                    "Task {}: Tool result interceptor failed for '{}' (attempt {}/{}): {}",
                    environment.task_id,
                    tool_name,
                    attempt,
                    max_attempts,
                    e
                );
            }
        }
    }
    log::warn!(
        "Task {}: Interceptor exhausted all attempts for '{}', using raw result.",
        environment.task_id,
        tool_name
    );
    (raw_result.to_string(), false)
}

/// Splits calls, in order, into the groups that run at the same time: each
/// run of consecutive parallel-safe calls, and each other call alone.
pub(super) fn groups<T>(calls: Vec<T>, parallel_safe: impl Fn(&T) -> bool) -> Vec<Vec<T>> {
    let mut groups: Vec<Vec<T>> = Vec::new();
    let mut shared = false;
    for call in calls {
        let safe = parallel_safe(&call);
        match groups.last_mut() {
            Some(group) if safe && shared => group.push(call),
            _ => groups.push(vec![call]),
        }
        shared = safe;
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_safe_neighbours_share_a_group_and_others_run_alone() {
        let calls = vec![
            ("read a", true),
            ("read b", true),
            ("write", false),
            ("read c", true),
            ("write 2", false),
            ("write 3", false),
            ("read d", true),
            ("read e", true),
        ];
        let grouped: Vec<Vec<&str>> = groups(calls, |call| call.1)
            .into_iter()
            .map(|group| group.into_iter().map(|call| call.0).collect())
            .collect();
        assert_eq!(
            grouped,
            [
                vec!["read a", "read b"],
                vec!["write"],
                vec!["read c"],
                vec!["write 2"],
                vec!["write 3"],
                vec!["read d", "read e"],
            ]
        );
        assert!(groups(Vec::<(&str, bool)>::new(), |call| call.1).is_empty());
    }
}
