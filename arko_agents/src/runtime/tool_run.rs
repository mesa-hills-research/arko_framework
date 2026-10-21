//! Running one tool call: middleware, the tool itself under the host's time
//! limit, and result interception. A call's inputs are owned, so the
//! parallel-safe calls of a response can run at the same time while the
//! loop watches its command channel. Once the middleware lets a call run,
//! the rest is an owned future, so a call that moves to the background goes
//! on as a job of the host's terminal manager.

use crate::stop::{StopSignal, StopSource};
use crate::tool::{
    ContextSignal, ContextSource, InterceptedResult, InterceptionFailurePolicy, MoveSignal,
    TaskContext, Tool, ToolBackground, ToolCallContext, ToolError, ToolExecutionContext,
    ToolExecutionDecision, ToolExecutionMiddleware, ToolExecutionOutcome, ToolImageSink,
    ToolOutput, ToolResultInterceptor,
};
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
    pub stop: StopSignal,
    /// The model whose image capabilities apply to tool results.
    pub model: arko_ai::Model,
    /// Receives the images calls return. See [`ToolImageSink`].
    pub image_sink: Option<Arc<dyn ToolImageSink>>,
    /// How many more images the conversation takes before a request would
    /// hold more than the model's provider takes, or `None` for no count.
    pub image_room: Option<Arc<crate::tool::ImageRoom>>,
    /// Notified as context arrives for the task. Each call gets a signal
    /// made as it starts. See [`TaskContext::context_arrived`].
    pub context: ContextSource,
    /// The terminal manager calls move to the background with. See
    /// [`AgentRuntime::with_background_jobs`](super::AgentRuntime::with_background_jobs).
    /// A moved call's future owns the environment, so the manager is held
    /// weakly. Dropping the host's handles still drops it, which stops the
    /// call.
    #[cfg(any(unix, windows))]
    pub jobs: Option<std::sync::Weak<crate::terminal::TerminalManager>>,
    /// The agent is offered `terminal_status` and `terminal_control`, which
    /// a moved call's result names.
    pub terminal_tools: bool,
}

/// What a call sees of a stop, a move and context arriving.
struct CallSignals {
    stop: StopSignal,
    moved: MoveSignal,
    context_arrived: ContextSignal,
}

impl RunEnvironment {
    /// What the tool (or the interceptor) of `run` sees of the task, with
    /// the call's signals.
    fn task_context(&self, run: &ToolRun, signals: &CallSignals) -> TaskContext<'_> {
        TaskContext {
            task_id: self.task_id,
            agent_name: &self.agent_name,
            initial_task_prompt: &self.initial_task_prompt,
            current_working_directory: &self.working_directory,
            tool_call_id: Some(run.tool_call_id),
            recovering: run.recovering,
            stop: signals.stop.clone(),
            moved: signals.moved.clone(),
            context_arrived: signals.context_arrived.clone(),
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

    /// Whether `tools` offers what a moved call's result names.
    pub fn offers_terminal_tools(tools: &crate::tool::ToolRegistry) -> bool {
        ["terminal_status", "terminal_control"]
            .iter()
            .all(|name| tools.get(name).is_some())
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
    /// The host-side time limit. `None` runs without one.
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
    /// How the call can move to the background ([`Tool::background_mode`]).
    pub background: ToolBackground,
    /// The model's id of the call, recorded with it while it runs
    /// ([`RunningToolCall::call_id`](crate::task::RunningToolCall::call_id)).
    pub call_id: Option<String>,
    /// The host's message given with the call's approval, recorded with it
    /// while it runs.
    pub message_to_agent: Option<String>,
    /// The call runs again after its task stopped while it ran. See
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
    /// The middleware rejected the call. It did not run.
    Rejected(String),
    /// The middleware hook itself failed before the tool ran. The task must
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
    /// call. It did not run.
    Stop { reason: String, cancel: bool },
    /// The middleware paused the task at the call, answering it with
    /// `message`. It did not run.
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
    /// The call moved to the background, and goes on as a job. `text` is
    /// its result for now, which names the job. The middleware's `after`
    /// and interception run on the real result in the background.
    Moved { text: String },
}

/// Runs one call. Middleware hook errors become pausable outcomes so the task
/// and any completed tool result remain available to the host.
pub(super) async fn run_tool(
    environment: Arc<RunEnvironment>,
    run: ToolRun,
) -> Result<RunOutcome, ToolError> {
    if let Some(output) = run.custom_output.clone() {
        return Ok(caller_provided(&environment, &run, output).await);
    }
    if let Some(outcome) = before(&environment, &run).await {
        return Ok(outcome);
    }
    let signals = CallSignals {
        stop: environment.stop.clone(),
        moved: MoveSignal::never(),
        // Context that arrives from now on reaches the call.
        context_arrived: environment.context.signal(),
    };
    #[cfg(any(unix, windows))]
    if run.background != ToolBackground::Never
        && let Some(jobs) = environment.jobs.as_ref().and_then(std::sync::Weak::upgrade)
    {
        return Ok(movable::run(environment, run, signals, jobs).await);
    }
    Ok(execute(environment, run, signals, None).await)
}

/// A call answered with the host's output instead of running: the
/// middleware's `after` sees it, and it is intercepted like a result.
async fn caller_provided(
    environment: &RunEnvironment,
    run: &ToolRun,
    output: String,
) -> RunOutcome {
    let call = ToolCallContext {
        tool_call_id: run.tool_call_id,
        tool_name: &run.tool_name,
        parameters: &run.params,
    };
    let output = ToolOutput::text(output);
    if let Some(middleware) = &environment.middleware
        && let Err(error) = middleware
            .after(
                &environment.execution_context(),
                &call,
                ToolExecutionOutcome::CallerProvided(&output.text),
            )
            .await
    {
        let reason = middleware_error_reason("after", &run.tool_name, &error);
        let (effective, intercepted) = intercepted_output(environment, run, &output.text).await;
        return RunOutcome::MiddlewareErrorAfter {
            output,
            effective,
            intercepted,
            reason,
            is_error: false,
        };
    }
    let (effective, intercepted) = intercepted_output(environment, run, &output.text).await;
    RunOutcome::Finished {
        output,
        effective,
        intercepted,
    }
}

/// The middleware's `before`: `None` when it lets the call run, else how
/// the call ends.
async fn before(environment: &RunEnvironment, run: &ToolRun) -> Option<RunOutcome> {
    let middleware = environment.middleware.as_ref()?;
    let call = ToolCallContext {
        tool_call_id: run.tool_call_id,
        tool_name: &run.tool_name,
        parameters: &run.params,
    };
    let decision = match middleware
        .before(&environment.execution_context(), &call)
        .await
    {
        Err(error) => {
            return Some(RunOutcome::MiddlewareErrorBefore {
                reason: middleware_error_reason("before", &run.tool_name, &error),
            });
        }
        Ok(decision) => decision,
    };
    match decision {
        ToolExecutionDecision::Continue => None,
        ToolExecutionDecision::Reject { message_to_agent } => {
            Some(RunOutcome::Rejected(message_to_agent))
        }
        ToolExecutionDecision::FailTask { reason } => Some(RunOutcome::Stop {
            reason,
            cancel: false,
        }),
        ToolExecutionDecision::CancelTask { reason } => Some(RunOutcome::Stop {
            reason,
            cancel: true,
        }),
        ToolExecutionDecision::PauseTask {
            message_to_agent,
            reason,
        } => Some(RunOutcome::Pause {
            message: message_to_agent,
            reason,
        }),
    }
}

/// The rest of a call the middleware let run: the tool under the host's
/// time limit, the image sink, the middleware's `after` and interception.
/// It owns what it uses, so a call that moves to the background goes on as
/// a job. Once `adopted` is set (the call moved), its images are kept as
/// the tool returned them, since the turn's image room no longer applies.
fn execute(
    environment: Arc<RunEnvironment>,
    run: ToolRun,
    signals: CallSignals,
    adopted: Option<Arc<AtomicBool>>,
) -> BoxFuture<'static, RunOutcome> {
    Box::pin(async move {
        let call = ToolCallContext {
            tool_call_id: run.tool_call_id,
            tool_name: &run.tool_name,
            parameters: &run.params,
        };
        let execution = environment.execution_context();
        let middleware = environment.middleware.as_ref();
        let context = environment.task_context(&run, &signals);
        let execution_future = run.tool.execute_with_output(&context, run.params.clone());
        // Dropping the call at the limit stops it: a command it started
        // is killed with it.
        let result = match run.timeout {
            Some(limit) => tokio::time::timeout(limit, execution_future)
                .await
                .unwrap_or(Err(ToolError::TimedOut { limit })),
            None => execution_future.await,
        };
        let mut output = match result {
            Ok(output) => output,
            Err(error) => {
                if let Some(middleware) = middleware {
                    let error_text =
                        format!("Tool '{}' execution failed: {}", run.tool_name, error);
                    if let Err(middleware_error) = middleware
                        .after(&execution, &call, ToolExecutionOutcome::Error(&error_text))
                        .await
                    {
                        return RunOutcome::MiddlewareErrorAfter {
                            output: ToolOutput::text(error_text.clone()),
                            effective: error_text,
                            intercepted: false,
                            reason: middleware_error_reason(
                                "after",
                                &run.tool_name,
                                &middleware_error,
                            ),
                            is_error: true,
                        };
                    }
                }
                return RunOutcome::Failed(error);
            }
        };
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
        if !adopted.is_some_and(|adopted| adopted.load(Ordering::Acquire)) {
            match &environment.image_room {
                Some(room) => output.admit_images_within(&environment.model, room),
                None => output.admit_images(&environment.model),
            }
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
            return RunOutcome::MiddlewareErrorAfter {
                output,
                effective,
                intercepted,
                reason,
                is_error: false,
            };
        }
        let (effective, intercepted) = intercepted_output(&environment, &run, &output.text).await;
        RunOutcome::Finished {
            output,
            effective,
            intercepted,
        }
    })
}

/// Calls that can move to the background.
#[cfg(any(unix, windows))]
mod movable {
    use super::*;
    use crate::terminal::{CallOutput, TerminalManager};

    /// Runs a call of a tool that can move, listed with `jobs` while it runs
    /// in the foreground. An `Own` tool sees the move as
    /// [`TaskContext::moved`]. A `Detach` call that moves goes on as a job
    /// and returns [`RunOutcome::Moved`].
    pub(super) async fn run(
        environment: Arc<RunEnvironment>,
        run: ToolRun,
        mut signals: CallSignals,
        jobs: Arc<TerminalManager>,
    ) -> RunOutcome {
        let held = jobs.hold_call(
            environment.task_id,
            run.tool_call_id,
            &run.tool_name,
            run.tool_name.clone(),
        );
        if run.background != ToolBackground::Detach {
            signals.moved = held.signal();
            let outcome = execute(environment, run, signals, None).await;
            drop(held);
            return outcome;
        }
        let (owner, id, tool_name, limit) = (
            environment.task_id,
            run.tool_call_id,
            run.tool_name.clone(),
            run.timeout,
        );
        let started = held.started();
        // The call's own stop. It follows the turn's while the call runs in
        // the foreground, and the job's once it moved, so stopping the turn
        // no longer stops it.
        let stop = StopSource::new();
        let turn_stop = std::mem::replace(&mut signals.stop, stop.signal());
        // Forwards the turn's stop until the call moves or ends. A task of
        // its own, since a stop that drops the call (an interruptible call
        // cancelled) drops this future before it runs again, and work the
        // call left on another thread still has to hear of it.
        let (forwarding, forwarded) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn({
            let (turn_stop, stop) = (turn_stop.clone(), stop.clone());
            async move {
                tokio::select! {
                    biased;
                    () = turn_stop.stopping() => {
                        stop.stop(turn_stop.reason().unwrap_or_default());
                    }
                    _ = forwarded => {}
                }
            }
        });
        let adopted = Arc::new(AtomicBool::new(false));
        let mut call = execute(environment.clone(), run, signals, Some(adopted.clone()));
        // Moves by itself after this long. A time too far off to count means
        // never.
        let auto_after = jobs.background_after();
        let mut moves_at =
            auto_after.and_then(|after| tokio::time::Instant::from_std(started).checked_add(after));
        let mut held = Some(held);
        let mut following = true;
        loop {
            let after = tokio::select! {
                // A call that ended as it was to move keeps its result.
                biased;
                outcome = &mut call => return outcome,
                () = turn_stop.stopping(), if following => {
                    stop.stop(turn_stop.reason().unwrap_or_default());
                    following = false;
                    // A call being stopped doesn't move.
                    held = None;
                    continue;
                }
                () = moved(&held), if held.is_some() => None,
                () = tokio::time::sleep_until(moves_at.unwrap_or_else(tokio::time::Instant::now)),
                    if moves_at.is_some() && held.is_some() =>
                {
                    if jobs.move_to_background(owner, Some(id)) == 0
                        && !held.as_ref().is_some_and(|held| held.is_moved())
                    {
                        // The host is exiting.
                        moves_at = None;
                        continue;
                    }
                    auto_after
                }
            };
            held = None;
            match jobs.reserve_call(owner, id, &tool_name, tool_name.clone()) {
                Ok(slot) => {
                    adopted.store(true, Ordering::Release);
                    // From now on only the job's stop stops it.
                    drop(forwarding);
                    let name = tool_name.clone();
                    let job = slot.run(
                        stop,
                        Box::pin(async move { call_output(&name, call.await) }),
                    );
                    return RunOutcome::Moved {
                        text: crate::terminal::moved_result(
                            &job,
                            started.elapsed(),
                            after,
                            limit,
                            environment.terminal_tools,
                        ),
                    };
                }
                Err(error) => {
                    // It goes on in the foreground.
                    log::warn!(
                        "Task {owner}: the '{tool_name}' call couldn't move to the background: {error}"
                    );
                }
            }
        }
    }

    /// Resolves when the call `held` registered moves.
    async fn moved(held: &Option<crate::terminal::HeldCall>) {
        match held {
            Some(held) => held.moved().await,
            None => std::future::pending().await,
        }
    }

    /// A moved call's outcome as its job's output.
    fn call_output(tool_name: &str, outcome: RunOutcome) -> CallOutput {
        match outcome {
            RunOutcome::Finished {
                output, effective, ..
            } => CallOutput::text(effective).with_images(output.images),
            RunOutcome::MiddlewareErrorAfter {
                output,
                effective,
                reason,
                is_error,
                ..
            } => {
                // The task can't pause for a call that moved on.
                log::warn!("{reason}");
                let result = if is_error {
                    CallOutput::error(effective)
                } else {
                    CallOutput::text(effective)
                };
                result.with_images(output.images)
            }
            RunOutcome::Failed(error) => {
                CallOutput::error(format!("Tool '{tool_name}' execution failed: {error}"))
            }
            // The middleware's `before` ran before the call started.
            RunOutcome::Rejected(text)
            | RunOutcome::MiddlewareErrorBefore { reason: text }
            | RunOutcome::Stop { reason: text, .. }
            | RunOutcome::Pause { message: text, .. } => CallOutput::error(text),
            RunOutcome::Moved { text } => CallOutput::text(text),
        }
    }
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

/// Applies `interception` to a result. Returns the text for the
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
    let context = environment.task_context(
        run,
        &CallSignals {
            stop: environment.stop.clone(),
            moved: MoveSignal::never(),
            context_arrived: ContextSignal::never(),
        },
    );
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
                "{raw_result}\n\n[This is the full result. It was to be shortened, but the tool-result interceptor failed ({error}).]"
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
