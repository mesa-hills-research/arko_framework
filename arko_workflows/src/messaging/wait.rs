//! The `wait_for_message` tool lets an agent in native tool mode, which must
//! call a tool every turn, wait for a message from another agent without
//! polling.
//!
//! The wait returns when an agent message arrives that the run's conversation
//! doesn't hold yet, an agent it can exchange messages with starts or finishes,
//! the time limit expires, or the workflow stops this run or all runs. The result names the senders, while
//! the message arrives, once, through the usual context update at the next
//! turn boundary, and the wait returns only once the message is in the run's
//! command channel, so that boundary is the very next one.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::time::Instant;

use arko_agents::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use arko_ai::{LLMClient, PromptCacheLifetime};

use crate::context::SharedContext;
use crate::messaging::registry::{Activity, AgentRegistry, AgentStatus};

pub const WAIT_FOR_MESSAGE_TOOL_NAME: &str = "wait_for_message";

/// How a wait that the user moved to the background ends.
const WAIT_MOVED: &str = "The user ended this wait so you can go on with other work. Messages from other agents reach you as notes as they come.";

/// The time limit for a 5-minute prompt cache (Anthropic's default, and
/// OpenAI's as reported), and when the lifetime is unknown or caching is
/// off, 4 minutes 30 seconds.
pub const DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT: Duration = Duration::from_secs(270);

/// The longest time limit the builder accepts, 24 hours.
pub const MAX_WAIT_FOR_MESSAGE_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// The least time an automatic limit leaves between the end of a wait and
/// the end of the cache lifetime, for the model request that follows.
const MIN_CACHE_MARGIN: Duration = Duration::from_secs(30);

/// The least an automatic limit leaves for a wait, however long ago the last
/// model request was.
const MIN_AUTO_WAIT: Duration = Duration::from_secs(1);

/// How long a wait that found a message still waits for the run's delivery
/// worker to put it in the command channel, so that it arrives at the next
/// turn boundary. It takes microseconds unless the channel is full.
const FORWARD_WAIT: Duration = Duration::from_secs(1);

/// How long `wait_for_message` waits at most, set by the workflow's setting
/// ([`WorkflowBuilder::wait_for_message_timeout`]) or an agent's
/// ([`WorkflowBuilder::agent_wait_for_message_timeout`]).
///
/// Every wait ends with a model request, and a request within the
/// provider's prompt-cache lifetime finds the prompt cached (a cache hit
/// restarts the lifetime) and pays only for cheap cached reads. A request
/// after it re-reads the whole prompt at full price. So waits are kept short,
/// and a long wait is several of them. The agent calls `wait_for_message`
/// again, keeping the cache warm. `Auto` keeps each wait inside the agent's
/// lifetime, from [`LLMClient::prompt_cache_lifetime`] for its model and
/// settings, calculated as the lifetime minus a margin of a tenth of it, at least 30
/// seconds. That is 4 minutes 30 seconds for a 5-minute cache (Anthropic's
/// default, and OpenAI's with an API key before GPT-5.6, reported
/// conservatively), 27 minutes for a 30-minute cache (OpenAI GPT-5.6 and
/// later, and ChatGPT OAuth), and 54 minutes with Anthropic's 1-hour TTL, an
/// explicit opt-in. A host's
/// `GenerationConfig::prompt_cache_lifetime_seconds` replaces the lifetime.
/// With an unknown lifetime, or caching off, it is
/// [`DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT`]. With nothing to keep warm, a
/// bounded wait still lets the agent look around now and then.
///
/// An automatic limit counts from the agent's last model request, when the
/// cache was last used, so the time the model took to call the tool is
/// included (a second is always left). A `Fixed` limit counts from the start
/// of the wait.
///
/// A `Fixed` limit may be longer than the cache lifetime, at the price of a
/// possible cache miss on the next turn after a wait that runs to it.
///
/// [`WorkflowBuilder::wait_for_message_timeout`]: crate::WorkflowBuilder::wait_for_message_timeout
/// [`WorkflowBuilder::agent_wait_for_message_timeout`]: crate::WorkflowBuilder::agent_wait_for_message_timeout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaitForMessageTimeout {
    /// Just inside the agent's prompt-cache lifetime.
    #[default]
    Auto,
    /// This long, more than zero and at most
    /// [`MAX_WAIT_FOR_MESSAGE_TIMEOUT`].
    Fixed(Duration),
}

impl From<Duration> for WaitForMessageTimeout {
    fn from(timeout: Duration) -> Self {
        Self::Fixed(timeout)
    }
}

impl WaitForMessageTimeout {
    /// The time limit for an agent whose prompt cache has `lifetime`.
    pub fn resolve(self, lifetime: PromptCacheLifetime) -> Duration {
        match (self, lifetime) {
            (Self::Fixed(timeout), _) => timeout,
            (Self::Auto, PromptCacheLifetime::Known(lifetime)) => lifetime
                .saturating_sub((lifetime / 10).max(MIN_CACHE_MARGIN))
                .clamp(Duration::from_secs(1), MAX_WAIT_FOR_MESSAGE_TIMEOUT),
            (Self::Auto, PromptCacheLifetime::Unknown | PromptCacheLifetime::Disabled) => {
                DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT
            }
        }
    }

    pub(crate) fn validate(self) -> Result<(), String> {
        match self {
            Self::Fixed(timeout) if timeout.is_zero() || timeout > MAX_WAIT_FOR_MESSAGE_TIMEOUT => {
                Err(format!(
                    "wait_for_message timeout must be more than zero and at most 24 hours, not {timeout:?}"
                ))
            }
            _ => Ok(()),
        }
    }
}

/// The workflow's `wait_for_message` limits: its own and agents'.
#[derive(Debug, Clone, Default)]
pub(crate) struct WaitTimeouts {
    pub workflow: WaitForMessageTimeout,
    pub agents: HashMap<String, WaitForMessageTimeout>,
}

impl WaitTimeouts {
    pub fn for_agent(&self, agent_name: &str) -> WaitForMessageTimeout {
        self.agents
            .get(agent_name)
            .copied()
            .unwrap_or(self.workflow)
    }
}

/// Waits for a message from another agent. Offered to every agent that
/// another agent may message. Its definition names no agent and no time, so
/// it never changes during a task.
pub struct WaitForMessageTool {
    context: Arc<SharedContext>,
    agent_name: String,
    llm_client: Arc<LLMClient>,
    /// Per run, when its conversation last got new messages, just before a
    /// model request, which is when its prompt cache was last used.
    last_request: parking_lot::Mutex<HashMap<uuid::Uuid, Instant>>,
}

impl std::fmt::Debug for WaitForMessageTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WaitForMessageTool")
            .field("agent_name", &self.agent_name)
            .finish_non_exhaustive()
    }
}

impl WaitForMessageTool {
    pub fn new(
        context: Arc<SharedContext>,
        agent_name: String,
        llm_client: Arc<LLMClient>,
    ) -> Self {
        Self {
            context,
            agent_name,
            llm_client,
            last_request: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// How long run `task_id` may wait: the agent's limit. An automatic
    /// limit counts from the run's last model request, when the cache was
    /// last used, so the time the model took to answer is included. At least
    /// a second is left.
    pub(crate) async fn wait_limit(&self, task_id: uuid::Uuid) -> Duration {
        let limit = self.limit_of(Some(task_id)).await;
        if self.context.wait_timeouts.for_agent(&self.agent_name) != WaitForMessageTimeout::Auto {
            return limit;
        }
        let since = self
            .last_request
            .lock()
            .get(&task_id)
            .map_or(Duration::ZERO, |at| at.elapsed());
        limit.saturating_sub(since).max(MIN_AUTO_WAIT)
    }

    /// The agent's time limit, for a run of its registered configuration.
    #[cfg(test)]
    pub(crate) async fn limit(&self) -> Duration {
        self.limit_of(None).await
    }

    /// The time limit of run `task_id`, by the prompt cache of the model it
    /// works with, which may not be its agent's (or of the agent's
    /// registered configuration without one).
    async fn limit_of(&self, task_id: Option<uuid::Uuid>) -> Duration {
        let timeout = self.context.wait_timeouts.for_agent(&self.agent_name);
        let config = match task_id {
            Some(task_id) => self.context.run_llm_config(&self.agent_name, task_id),
            None => self
                .context
                .agent_configs
                .get(&self.agent_name)
                .map(|config| config.llm_config.clone()),
        };
        let lifetime = match (timeout, config) {
            (WaitForMessageTimeout::Auto, Some(config)) => {
                self.llm_client
                    .prompt_cache_lifetime(&config.model, &config.generation_config)
                    .await
            }
            _ => PromptCacheLifetime::Unknown,
        };
        timeout.resolve(lifetime)
    }
}

/// Marks a run as waiting for a message while it lives. Dropping it (the wait
/// returned, or its task was aborted) marks the run running again.
struct AwaitingMessage<'a> {
    registry: &'a AgentRegistry,
    task_id: uuid::Uuid,
}

impl<'a> AwaitingMessage<'a> {
    fn start(registry: &'a AgentRegistry, task_id: uuid::Uuid) -> Self {
        registry.set_awaiting_message(task_id, true);
        Self { registry, task_id }
    }
}

impl Drop for AwaitingMessage<'_> {
    fn drop(&mut self) {
        self.registry.set_awaiting_message(self.task_id, false);
    }
}

/// The model's `timeout_seconds`, if given: a positive number of seconds,
/// as a number (the parameter's type) or a string (as conversations saved
/// before it was an integer hold it).
fn requested_timeout(params: &HashMap<String, Value>) -> Result<Option<f64>, ToolError> {
    let Some(value) = params.get("timeout_seconds") else {
        return Ok(None);
    };
    let seconds = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    };
    match seconds {
        Some(seconds) if seconds.is_finite() && seconds > 0.0 => Ok(Some(seconds)),
        _ => Err(ToolError::ParameterValidation {
            param_name: "timeout_seconds".into(),
            message: "Expected a positive number of seconds.".into(),
        }),
    }
}

/// `270 s`, `27 min`, `0.5 s`.
fn describe_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 10.0 && duration.subsec_millis() != 0 {
        return format!("{seconds:.1} s");
    }
    let seconds = seconds.round() as u64;
    if seconds >= 120 && seconds.is_multiple_of(60) {
        format!("{} min", seconds / 60)
    } else {
        format!("{seconds} s")
    }
}

/// `` `a` ``, `` `a` and `b` ``, `` `a`, `b` and `c` ``.
fn names(agents: &[String]) -> String {
    let quoted = agents
        .iter()
        .map(|agent| format!("`{agent}`"))
        .collect::<Vec<_>>();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => quoted.concat(),
    }
}

#[async_trait]
impl Tool for WaitForMessageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: WAIT_FOR_MESSAGE_TOOL_NAME.to_string(),
            description: "Wait for a message from another agent. Returns when one arrives (delivered with your next turn), when an agent you work with starts or finishes, or at a time limit. Call again to keep waiting.".to_string(),
            output_based_example: None,
            parameters: vec![ToolParameterDefinition {
                name: "timeout_seconds".to_string(),
                description: "Optional: the most seconds to wait.".to_string(),
                param_type: ToolParameterType::Integer,
                required: false,
                is_path: false,
                flag: None,
                default_value: None,
            }],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    /// The wait has its own limit (up to a day, when configured so).
    fn uses_host_timeout(&self) -> bool {
        false
    }

    /// Moving the wait to the background ends it, since messages reach the
    /// agent with its next turns anyway.
    fn background_mode(&self) -> arko_agents::ToolBackground {
        arko_agents::ToolBackground::Own
    }

    fn observe_context(&self, task_id: uuid::Uuid, message: &arko_ai::Message) {
        self.context
            .messages
            .observe_context(&self.agent_name, task_id, message);
        // The runtime reports new messages just before each model request.
        self.last_request.lock().insert(task_id, Instant::now());
    }

    async fn execute(
        &self,
        task_context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let requested = requested_timeout(&params)?;
        let limit = self.wait_limit(task_context.task_id).await;
        // The model may only shorten the wait.
        let wait = match requested {
            Some(seconds) if seconds < limit.as_secs_f64() => Duration::from_secs_f64(seconds),
            _ => limit,
        };
        let context = &self.context;
        let task_id = task_context.task_id;
        let _waiting = AwaitingMessage::start(&context.registry, task_id);

        // Taken, and marked seen, before the first check: whatever happens
        // after the check wakes the wait.
        let mut arrivals = context.messages.watch_arrivals(task_id);
        arrivals.borrow_and_update();
        let mut versions = context.registry.subscribe();
        versions.borrow_and_update();
        let mut stopping = context.run_control.agent_stopping(task_id);
        let peers = context.status_peers(&self.agent_name);
        let baseline = context.registry.activity(&peers);

        let deadline = Instant::now() + wait;
        let mut forward_deadline = None;
        let (mut arrivals_open, mut versions_open, mut stopping_open) = (true, true, true);
        loop {
            // Stopped, or a graceful stop's wrap-up began: the agent is
            // told what to do at its next turn boundary.
            if context.run_control.is_stopped() || context.run_control.is_wrapping_up() {
                return Ok("Workflow stopping.".to_string());
            }
            // The run alone is being stopped (its tools' stop signal fired).
            if *stopping.borrow_and_update() || task_context.stop.is_stopping() {
                return Ok("Run stopping.".to_string());
            }
            let unseen = context.messages.unseen_messages(&self.agent_name, task_id);
            let changes = status_changes(context, &peers, &baseline);
            let now = Instant::now();
            if unseen.senders.is_empty() {
                forward_deadline = None;
                if !changes.is_empty() {
                    return Ok(changes.join(" "));
                }
                if task_context.moved.is_moved() {
                    return Ok(WAIT_MOVED.to_string());
                }
                // A message arriving does too, but its sender is named
                // above.
                if task_context.context_arrived.has_arrived() {
                    return Ok(crate::subagents::CONTEXT_ARRIVED.to_string());
                }
                if now >= deadline {
                    return Ok(format!("No messages after {}.", describe_duration(wait)));
                }
            } else {
                if unseen.forwarded || forward_deadline.is_some_and(|until| now >= until) {
                    return Ok(message_result(&unseen.senders, &changes));
                }
                forward_deadline.get_or_insert(now + FORWARD_WAIT);
            }
            let until = forward_deadline.unwrap_or(deadline);
            tokio::select! {
                biased;
                _ = context.run_control.stopped() => {}
                _ = task_context.stop.stopping() => {}
                changed = stopping.changed(), if stopping_open => stopping_open = changed.is_ok(),
                changed = arrivals.changed(), if arrivals_open => arrivals_open = changed.is_ok(),
                changed = versions.changed(), if versions_open => versions_open = changed.is_ok(),
                // Checked above once they fired, while a message waits to
                // be forwarded.
                () = task_context.moved.moved(), if !task_context.moved.is_moved() => {}
                () = task_context.context_arrived.arrived(),
                    if !task_context.context_arrived.has_arrived() => {}
                _ = tokio::time::sleep_until(until) => {}
            }
        }
    }
}

/// A sentence for each of `peers` that started or finished since `baseline`:
/// `` `worker` started. `` or `` `worker` finished (completed). ``
fn status_changes(context: &SharedContext, peers: &[String], baseline: &[Activity]) -> Vec<String> {
    let current = context.registry.activity(peers);
    peers
        .iter()
        .zip(baseline.iter().zip(&current))
        .filter(|(_, (before, now))| before != now)
        .map(|(agent, _)| match context.recipient_status(agent) {
            AgentStatus::Running | AgentStatus::Waiting(_) => format!("`{agent}` started."),
            status => format!("`{agent}` {}.", status.describe()),
        })
        .collect()
}

fn message_result(senders: &[String], changes: &[String]) -> String {
    let mut result = if senders.len() == 1 {
        format!("Message from {} waiting.", names(senders))
    } else {
        format!("Messages from {} waiting.", names(senders))
    };
    for change in changes {
        result.push(' ');
        result.push_str(change);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: Duration = Duration::from_secs(60);

    #[test]
    fn auto_limits_stay_inside_the_cache_lifetime() {
        let auto = WaitForMessageTimeout::Auto;
        let known = |lifetime| auto.resolve(PromptCacheLifetime::Known(lifetime));
        assert_eq!(known(5 * MINUTE), Duration::from_secs(270));
        assert_eq!(known(30 * MINUTE), 27 * MINUTE);
        assert_eq!(known(60 * MINUTE), 54 * MINUTE);
        // The margin is at least 30 seconds, and the limit at least 1 second.
        assert_eq!(known(2 * MINUTE), Duration::from_secs(90));
        assert_eq!(known(Duration::from_secs(10)), Duration::from_secs(1));
        assert_eq!(known(48 * 60 * MINUTE), MAX_WAIT_FOR_MESSAGE_TIMEOUT);
        for lifetime in [PromptCacheLifetime::Unknown, PromptCacheLifetime::Disabled] {
            assert_eq!(auto.resolve(lifetime), DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT);
        }
        let fixed = WaitForMessageTimeout::from(Duration::from_secs(600));
        assert_eq!(
            fixed.resolve(PromptCacheLifetime::Known(5 * MINUTE)),
            Duration::from_secs(600)
        );
    }

    #[test]
    fn fixed_limits_are_validated_and_agents_override_the_workflow() {
        assert!(WaitForMessageTimeout::Auto.validate().is_ok());
        for valid in [Duration::from_millis(1), MAX_WAIT_FOR_MESSAGE_TIMEOUT] {
            assert!(WaitForMessageTimeout::Fixed(valid).validate().is_ok());
        }
        for invalid in [
            Duration::ZERO,
            MAX_WAIT_FOR_MESSAGE_TIMEOUT + Duration::from_secs(1),
        ] {
            assert!(WaitForMessageTimeout::Fixed(invalid).validate().is_err());
        }
        let timeouts = WaitTimeouts {
            workflow: WaitForMessageTimeout::Fixed(MINUTE),
            agents: HashMap::from([("lead".to_string(), WaitForMessageTimeout::Auto)]),
        };
        assert_eq!(timeouts.for_agent("lead"), WaitForMessageTimeout::Auto);
        assert_eq!(
            timeouts.for_agent("worker"),
            WaitForMessageTimeout::Fixed(MINUTE)
        );
    }

    #[test]
    fn requested_timeouts_are_positive_numbers() {
        let params = |value: Value| HashMap::from([("timeout_seconds".to_string(), value)]);
        assert_eq!(requested_timeout(&HashMap::new()).unwrap(), None);
        assert_eq!(
            requested_timeout(&params(Value::from("90"))).unwrap(),
            Some(90.0)
        );
        assert_eq!(
            requested_timeout(&params(Value::from(0.5))).unwrap(),
            Some(0.5)
        );
        for invalid in [
            Value::from("0"),
            Value::from("-5"),
            Value::from("soon"),
            Value::from("NaN"),
            Value::Bool(true),
        ] {
            assert!(requested_timeout(&params(invalid)).is_err());
        }
    }

    #[test]
    fn durations_and_names_read_naturally() {
        assert_eq!(describe_duration(Duration::from_secs(270)), "270 s");
        assert_eq!(describe_duration(27 * MINUTE), "27 min");
        assert_eq!(describe_duration(Duration::from_millis(300)), "0.3 s");
        assert_eq!(describe_duration(Duration::from_secs(90)), "90 s");
        let agents = |list: &[&str]| list.iter().map(|a| a.to_string()).collect::<Vec<_>>();
        assert_eq!(names(&agents(&["lead"])), "`lead`");
        assert_eq!(names(&agents(&["a", "b"])), "`a` and `b`");
        assert_eq!(names(&agents(&["a", "b", "c"])), "`a`, `b` and `c`");
    }
}
