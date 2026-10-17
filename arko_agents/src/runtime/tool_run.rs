//! Running one tool call: middleware, the tool itself under the host's time
//! limit, and result interception. A call's inputs are owned, so the
//! parallel-safe calls of a response can run at the same time while the
//! loop watches its command channel.

use crate::tool::{
    InterceptedResult, InterceptionFailurePolicy, TaskContext, Tool, ToolCallContext, ToolError,
    ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware, ToolExecutionOutcome,
    ToolImageSink, ToolOutput, ToolResultInterceptor,
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
    /// What the task's calls see of a stop: see [`TaskContext::stop`].
    pub stop: crate::stop::StopSignal,
    /// The model whose image capabilities apply to tool results.
    pub model: arko_ai::Model,
    /// Receives the images calls return. See [`ToolImageSink`].
    pub image_sink: Option<Arc<dyn ToolImageSink>>,
    /// How many more images the conversation takes before a request would
    /// hold more than the model's provider takes, or `None` for no count.
    pub image_room: Option<Arc<crate::tool::ImageRoom>>,
}

impl RunEnvironment {
    /// What the tool (or the interceptor) of `run` sees of the task.
    fn task_context(&self, run: &ToolRun) -> TaskContext<'_> {
        TaskContext {
            task_id: self.task_id,
            agent_name: &self.agent_name,
            initial_task_prompt: &self.initial_task_prompt,
            current_working_directory: &self.working_directory,
            tool_call_id: Some(run.tool_call_id),
            recovering: run.recovering,
            stop: self.stop.clone(),
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
    /// The model's id of the call, recorded with it while it runs
    /// ([`RunningToolCall::call_id`](crate::task::RunningToolCall::call_id)).
    pub call_id: Option<String>,
    /// The host's message given with the call's approval, recorded with it
    /// while it runs.
    pub message_to_agent: Option<String>,
    /// The call runs again after its task stopped while it ran; see
    /// [`TaskContext::recovering`].
    pub recovering: bool,
}

impl ToolRun {
    /// The call as recorded while it runs.
    pub fn running(&self) -> crate::task::RunningToolCall {
        crate::task::RunningToolCall {
            tool_call_id: self.tool_call_id,
            call_id: self.call_id.clone(),
            tool_name: self.tool_name.clone(),
            parameters: self.params.clone(),
            custom_output: self.custom_output.clone(),
            message_to_agent: self.message_to_agent.clone(),
            built_in: false,
            output: None,
            output_images: Vec::new(),
        }
    }
}

/// How a call ended.
pub(super) enum RunOutcome {
    /// The middleware rejected the call; it did not run.
    Rejected(String),
    /// The middleware hook itself failed before the tool ran; the task must
    /// pause so the host can decide whether to continue.
    MiddlewareErrorBefore { reason: String },
    /// The middleware hook itself failed after the tool returned. The output
    /// is kept so a continuation does not execute the tool again.
    MiddlewareErrorAfter {
        output: ToolOutput,
        effective: String,
        intercepted: bool,
        reason: String,
        is_error: bool,
    },
    /// The middleware failed (or, with `cancel`, cancelled) the task at the
    /// call; it did not run.
    Stop { reason: String, cancel: bool },
    /// The middleware paused the task at the call, answering it with
    /// `message`; it did not run.
    Pause { message: String, reason: String },
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

/// Runs one call. Middleware hook errors become pausable outcomes so the task
/// and any completed tool result remain available to the host.
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
            let output = ToolOutput::text(output.clone());
            if let Some(middleware) = middleware
                && let Err(error) = middleware
                    .after(
                        &execution,
                        &call,
                        ToolExecutionOutcome::CallerProvided(&output.text),
                    )
                    .await
            {
                let reason = middleware_error_reason("after", &run.tool_name, &error);
                let (effective, intercepted) =
                    intercepted_output(&environment, &run, &output.text).await;
                return Ok(RunOutcome::MiddlewareErrorAfter {
                    output,
                    effective,
                    intercepted,
                    reason,
                    is_error: false,
                });
            }
            output
        }
        None => {
            if let Some(middleware) = middleware {
                match middleware.before(&execution, &call).await {
                    Err(error) => {
                        return Ok(RunOutcome::MiddlewareErrorBefore {
                            reason: middleware_error_reason("before", &run.tool_name, &error),
                        });
                    }
                    Ok(decision) => match decision {
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
                        ToolExecutionDecision::PauseTask {
                            message_to_agent,
                            reason,
                        } => {
                            return Ok(RunOutcome::Pause {
                                message: message_to_agent,
                                reason,
                            });
                        }
                    },
                }
            }
            let context = environment.task_context(&run);
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
                Ok(mut output) => {
                    if let Some(sink) = &environment.image_sink
                        && !output.images.is_empty()
                    {
                        sink.tool_images(
                            environment.task_id,
                            run.tool_call_id,
                            &run.tool_name,
                            &output.images,
                        );
                    }
                    match &environment.image_room {
                        Some(room) => output.admit_images_within(&environment.model, room),
                        None => output.admit_images(&environment.model),
                    }
                    if let Some(middleware) = middleware
                        && let Err(error) = middleware
                            .after(
                                &execution,
                                &call,
                                ToolExecutionOutcome::Success(&output.text),
                            )
                            .await
                    {
                        let reason = middleware_error_reason("after", &run.tool_name, &error);
                        let (effective, intercepted) =
                            intercepted_output(&environment, &run, &output.text).await;
                        return Ok(RunOutcome::MiddlewareErrorAfter {
                            output,
                            effective,
                            intercepted,
                            reason,
                            is_error: false,
                        });
                    }
                    output
                }
                Err(error) => {
                    if let Some(middleware) = middleware {
                        let error_text =
                            format!("Tool '{}' execution failed: {}", run.tool_name, error);
                        if let Err(middleware_error) = middleware
                            .after(&execution, &call, ToolExecutionOutcome::Error(&error_text))
                            .await
                        {
                            return Ok(RunOutcome::MiddlewareErrorAfter {
                                output: ToolOutput::text(error_text.clone()),
                                effective: error_text,
                                intercepted: false,
                                reason: middleware_error_reason(
                                    "after",
                                    &run.tool_name,
                                    &middleware_error,
                                ),
                                is_error: true,
                            });
                        }
                    }
                    return Ok(RunOutcome::Failed(error));
                }
            }
        }
    };
    let (effective, intercepted) = intercepted_output(&environment, &run, &output.text).await;
    Ok(RunOutcome::Finished {
        output,
        effective,
        intercepted,
    })
}

async fn intercepted_output(
    environment: &RunEnvironment,
    run: &ToolRun,
    raw_output: &str,
) -> (String, bool) {
    match &run.interception {
        Some(interception) => intercept(environment, interception, run, raw_output).await,
        None => (raw_output.to_string(), false),
    }
}

fn middleware_error_reason(phase: &str, tool_name: &str, error: &ToolError) -> String {
    format!("Tool execution middleware {phase} hook failed for '{tool_name}': {error}")
}

/// Applies `interception` to a result; returns the text for the
/// conversation and whether it differs from the raw result. A failed
/// interceptor is retried under its policy, and then the raw result is used.
async fn intercept(
    environment: &RunEnvironment,
    interception: &Interception,
    run: &ToolRun,
    raw_result: &str,
) -> (String, bool) {
    let (tool_name, params) = (run.tool_name.as_str(), &run.params);
    let max_attempts = interception.policy.max_attempts();
    let context = environment.task_context(run);
    let mut last_error = None;
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
                last_error = Some(e);
            }
        }
    }
    log::warn!(
        "Task {}: Interceptor exhausted all attempts for '{}', using raw result.",
        environment.task_id,
        tool_name
    );
    if interception.policy.informs_agent() {
        let error = last_error
            .map(|error| error.to_string())
            .unwrap_or_default();
        return (
            format!(
                "{raw_result}\n\n[This is the full result: it was to be shortened, but the tool-result interceptor failed ({error}).]"
            ),
            true,
        );
    }
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
