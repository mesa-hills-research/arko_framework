//! Inter-agent messaging system.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Weak},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Notify, mpsc, watch},
    task::AbortHandle,
};
use uuid::Uuid;

use arko_agents::runtime::command::UserCommand;

pub mod registry;
pub mod tools;
pub mod wait;

/// The sender name of the context notices [`MessageBus::inject_task`]
/// queues, such as context-freshness notices.
pub const WORKFLOW_CONTEXT_SENDER: &str = "workflow-context";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageAttachment {
    File { path: String },
    MemoryTier { tier: String },
    Transcript { task_id: uuid::Uuid },
    Text { name: String, content: String },
}

/// Whom each agent may message with the `send_message` tool.
///
/// Set with [`WorkflowBuilder::messaging`](crate::WorkflowBuilder::messaging).
/// An agent the policy doesn't name may message every other registered agent
/// (the default for every agent). [`can_message`](Self::can_message)
/// restricts an agent to named recipients, or to none. An agent is offered
/// `send_message` only when it has at least one permitted recipient, so a
/// single-agent workflow has none.
///
/// ```rust
/// use arko_workflows::messaging::MessagingPolicy;
///
/// // Workers report to the lead only, and the lead may message everyone.
/// let policy = MessagingPolicy::all()
///     .can_message("worker_a", ["lead"])
///     .can_message("worker_b", ["lead"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MessagingPolicy {
    /// For each agent listed, the only agents it may message (none when the
    /// set is empty). An agent without an entry may message every other
    /// agent.
    pub recipients: HashMap<String, HashSet<String>>,
    /// When enabled, `send_message` to a finished agent queues the message for
    /// the agent's next run (a continuation or a new run) instead of
    /// failing. Off by default. A message to a finished agent may never be
    /// read, and the sender should know. See
    /// [`keep_messages_for_finished`](Self::keep_messages_for_finished).
    pub keep_for_finished: bool,
}

impl MessagingPolicy {
    /// Every agent may message every other agent (the default).
    pub fn all() -> Self {
        Self::default()
    }

    /// `agent` may message only `recipients`. An empty list means no one.
    /// Replaces an earlier setting for `agent`.
    pub fn can_message<I, S>(mut self, agent: impl Into<String>, recipients: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.recipients.insert(
            agent.into(),
            recipients.into_iter().map(Into::into).collect(),
        );
        self
    }

    /// Messages to a finished agent are kept for its next run instead of
    /// being refused. Its finish notice says so. Suits workflows that
    /// continue or run agents again. For example, a lead that finished before a worker
    /// replied gets the reply when it is continued.
    pub fn keep_messages_for_finished(mut self) -> Self {
        self.keep_for_finished = true;
        self
    }

    /// `agent` may message no one and is not offered `send_message`.
    pub fn cannot_message(self, agent: impl Into<String>) -> Self {
        self.can_message(agent, std::iter::empty::<String>())
    }

    /// Whether `sender` may message `recipient`. Never itself.
    pub fn permits(&self, sender: &str, recipient: &str) -> bool {
        sender != recipient
            && self
                .recipients
                .get(sender)
                .is_none_or(|allowed| allowed.contains(recipient))
    }

    /// The agents of `roster` that `sender` may message, in roster order.
    /// Never the sender itself.
    pub fn recipients_of(&self, sender: &str, roster: &[String]) -> Vec<String> {
        roster
            .iter()
            .filter(|agent| self.permits(sender, agent))
            .cloned()
            .collect()
    }

    /// Every agent the policy names must be registered.
    pub(crate) fn validate(&self, agents: &HashSet<String>) -> Result<(), crate::WorkflowError> {
        let named = self
            .recipients
            .iter()
            .flat_map(|(sender, recipients)| std::iter::once(sender).chain(recipients));
        for agent in named {
            if !agents.contains(agent) {
                return Err(crate::WorkflowError::BuilderInvalid(format!(
                    "messaging policy references unknown agent '{agent}'"
                )));
            }
        }
        Ok(())
    }
}

/// A message sent between agents in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    #[serde(default)]
    pub id: Uuid,
    pub from: String,
    #[serde(default)]
    pub from_task_id: Option<Uuid>,
    pub to: String,
    pub content: String,
    #[serde(default)]
    pub attachments: Vec<MessageAttachment>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// The agent run a context notice is for (set on notices
    /// [`MessageBus::inject_task`] queues). It is dropped when that run
    /// stops, whatever its outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_task_id: Option<Uuid>,
    /// The agent runs (task IDs) that stopped holding this message, in
    /// their conversation (in a prompt, or added as a context update) or
    /// taken to be added when they run again. It is not presented to them
    /// again when they are continued or resumed. Other runs of the agent
    /// still get it until one that completed with it in its conversation
    /// acknowledges it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub held_by: Vec<Uuid>,
    /// The agent runs that read this message in a conversation they later
    /// handed over (see [`arko_agents::handover`]). It counts as in their
    /// conversation when they stop, though the fresh conversation no
    /// longer holds it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub read_by: Vec<Uuid>,
}

impl AgentMessage {
    /// Whether run `task_id` holds this message. See
    /// [`held_by`](Self::held_by).
    pub fn is_held_by(&self, task_id: Uuid) -> bool {
        self.held_by.contains(&task_id)
    }

    /// Whether the message is still to be presented to run `task_id`. The
    /// run doesn't hold it, and it isn't another run's context notice.
    pub fn is_for(&self, task_id: Uuid) -> bool {
        !self.is_held_by(task_id) && self.to_task_id.is_none_or(|to| to == task_id)
    }
}

/// Per-agent message queue. Active recipients receive asynchronous context
/// injections. Later invocations receive pending messages in their prompt.
/// Messages remain snapshot-backed until the workflow acknowledges completion.
pub struct MessageBus {
    state: Arc<Mutex<BusState>>,
    roster: HashSet<String>,
}

#[derive(Default)]
struct BusState {
    queues: HashMap<String, Vec<AgentMessage>>,
    active: HashMap<String, HashMap<Uuid, ActiveRecipient>>,
    /// Per agent run (task ID), a counter bumped whenever an agent message
    /// arrives for the run or the run's delivery worker forwards a message.
    /// Created by whichever comes first, the registration or a waiter, and
    /// removed when the run is unregistered.
    arrivals: HashMap<Uuid, watch::Sender<u64>>,
}

fn signal_arrival(arrivals: &HashMap<Uuid, watch::Sender<u64>>, task_id: Uuid) {
    if let Some(arrivals) = arrivals.get(&task_id) {
        arrivals.send_modify(|count| *count = count.wrapping_add(1));
    }
}

struct ActiveRecipient {
    generation: Uuid,
    sender: mpsc::Sender<UserCommand>,
    outbound: VecDeque<(Uuid, String)>,
    /// Agent messages sent to this run that its conversation doesn't hold
    /// yet, as `(id, sender, injected payload)`. Context notices are left out.
    unseen: Vec<(Uuid, String, String)>,
    wake: Arc<Notify>,
    pump: Option<AbortHandle>,
}

/// The agent messages a run hasn't seen yet. See
/// [`MessageBus::unseen_messages`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct UnseenMessages {
    /// Their senders, each once, in the order their messages arrived.
    pub senders: Vec<String>,
    /// Whether every one of them is in the run's command channel, so the
    /// runtime adds it at its next turn boundary.
    pub forwarded: bool,
}

impl Drop for ActiveRecipient {
    fn drop(&mut self) {
        if let Some(pump) = &self.pump {
            pump.abort();
        }
    }
}

/// Reserve outside the state lock, then commit channel delivery and its
/// acknowledgement ID together. A busy agent must never block its callers.
async fn forward(
    state: Weak<Mutex<BusState>>,
    agent_name: String,
    task_id: Uuid,
    generation: Uuid,
    wake: Arc<Notify>,
) {
    loop {
        let sender = {
            let Some(state) = state.upgrade() else {
                return;
            };
            let state = state.lock();
            let Some(active) = state
                .active
                .get(&agent_name)
                .and_then(|instances| instances.get(&task_id))
                .filter(|active| active.generation == generation)
            else {
                return;
            };
            (!active.outbound.is_empty()).then(|| active.sender.clone())
        };
        let Some(sender) = sender else {
            wake.notified().await;
            continue;
        };
        let permit = match sender.reserve_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                if let Some(state) = state.upgrade() {
                    let mut state = state.lock();
                    if let Some(instances) = state.active.get_mut(&agent_name) {
                        if instances
                            .get(&task_id)
                            .is_some_and(|active| active.generation == generation)
                        {
                            instances.remove(&task_id);
                        }
                        if instances.is_empty() {
                            state.active.remove(&agent_name);
                        }
                    }
                    // A waiter learns that the run is gone.
                    signal_arrival(&state.arrivals, task_id);
                }
                // Nothing has been acknowledged: every unsent message remains
                // in the checkpoint-backed queue for the next invocation.
                return;
            }
        };
        let Some(state) = state.upgrade() else {
            return;
        };
        let mut state = state.lock();
        let Some(active) = state
            .active
            .get_mut(&agent_name)
            .and_then(|instances| instances.get_mut(&task_id))
            .filter(|active| active.generation == generation)
        else {
            return;
        };
        if let Some((_, content)) = active.outbound.pop_front() {
            permit.send(UserCommand::InjectContext { content });
            signal_arrival(&state.arrivals, task_id);
        }
    }
}

/// The context injection that delivers `message` to a running agent.
pub(crate) fn injection(message: &AgentMessage) -> String {
    let attachments = if message.attachments.is_empty() {
        String::new()
    } else {
        format!(
            "\nAttachments/context references:\n{}",
            serde_json::to_string_pretty(&message.attachments).unwrap_or_default()
        )
    };
    format!(
        "Message {} from agent '{}':\n{}{}",
        message.id, message.from, message.content, attachments
    )
}

/// Whether `content`, an injected context payload, is the injection of a
/// context notice [`MessageBus::inject_task`] queued.
pub(crate) fn is_context_notice(content: &str) -> bool {
    content
        .strip_prefix("Message ")
        .and_then(|rest| rest.split_once(' '))
        .is_some_and(|(id, rest)| {
            Uuid::parse_str(id).is_ok()
                && rest.starts_with(&format!("from agent '{WORKFLOW_CONTEXT_SENDER}':\n"))
        })
}

/// Whether `content`, an injected context payload, is message `id`'s
/// [`injection`].
pub(crate) fn is_injection_of(content: &str, id: Uuid) -> bool {
    content.starts_with(&format!("Message {id} from agent '"))
}

/// The line that opens message `id`'s block in a prompt (see
/// `WorkflowEngine::pending_messages`).
pub(crate) fn prompt_block_marker(id: Uuid, from: &str) -> String {
    format!("--- Message {id} from '{from}' ---")
}

/// Whether `conversation` holds `message`: presented in one of its prompts,
/// or added as a runtime context update.
fn conversation_holds(conversation: &arko_ai::Conversation, message: &AgentMessage) -> bool {
    let marker = prompt_block_marker(message.id, &message.from);
    let payload = injection(message);
    conversation.messages.iter().any(|entry| {
        context_update(entry).is_some_and(|update| contains_injection(update, &payload))
            || (entry.role == arko_ai::Role::User
                && entry
                    .content
                    .iter()
                    .filter_map(arko_ai::Content::as_text)
                    .any(|text| text.contains(&marker)))
    })
}

fn acknowledge(state: &mut BusState, agent_name: &str, message_ids: &[Uuid]) {
    if message_ids.is_empty() {
        return;
    }
    let ids = message_ids.iter().copied().collect::<HashSet<_>>();
    if let Some(messages) = state.queues.get_mut(agent_name) {
        messages.retain(|message| !ids.contains(&message.id));
        if messages.is_empty() {
            state.queues.remove(agent_name);
        }
    }
}

/// The runtime's context update a conversation message carries, without the
/// environment details and reminder `PromptBuilder` appends after it (copies
/// in those suffixes are not deliveries). An identical marker inside message
/// content conservatively doesn't count.
fn context_update(message: &arko_ai::Message) -> Option<&str> {
    if message.role != arko_ai::Role::User {
        return None;
    }
    let [arko_ai::Content::Text(text)] = message.content.as_slice() else {
        return None;
    };
    let update = text.strip_prefix("# Workflow context update\n\n")?;
    let update = update
        .split("\n\n# Current Environment Information (updated this turn)\n")
        .next()?;
    update.split("\n\n# Reminder\n").next()
}

fn contains_injection(update: &str, payload: &str) -> bool {
    update.match_indices(payload).any(|(start, _)| {
        (start == 0 || update[..start].ends_with("\n\n"))
            && (start + payload.len() == update.len()
                || update[start + payload.len()..].starts_with("\n\n"))
    })
}

impl MessageBus {
    pub fn new(roster: impl IntoIterator<Item = String>) -> Self {
        Self {
            state: Arc::new(Mutex::new(BusState::default())),
            roster: roster.into_iter().collect(),
        }
    }

    pub fn from_snapshot(
        roster: impl IntoIterator<Item = String>,
        queues: HashMap<String, Vec<AgentMessage>>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(BusState {
                queues,
                ..BusState::default()
            })),
            roster: roster.into_iter().collect(),
        }
    }

    pub fn roster(&self) -> Vec<String> {
        let mut roster = self.roster.iter().cloned().collect::<Vec<_>>();
        roster.sort();
        roster
    }

    /// Register live delivery without requiring a Tokio runtime at registration
    /// time. If called outside Tokio, a later send/inject inside Tokio starts the
    /// pump. Pending messages are always available to the invocation's prompt.
    pub fn register_active(
        &self,
        agent_name: &str,
        task_id: Uuid,
        sender: mpsc::Sender<UserCommand>,
    ) {
        self.register_locked(&mut self.state.lock(), agent_name, task_id, sender);
    }

    fn register_locked(
        &self,
        state: &mut BusState,
        agent_name: &str,
        task_id: Uuid,
        sender: mpsc::Sender<UserCommand>,
    ) {
        let active = state.active.entry(agent_name.to_string()).or_default();
        active.insert(
            task_id,
            ActiveRecipient {
                generation: Uuid::new_v4(),
                sender,
                outbound: VecDeque::new(),
                unseen: Vec::new(),
                wake: Arc::new(Notify::new()),
                pump: None,
            },
        );
        self.start_pump(agent_name, task_id, active.get_mut(&task_id).unwrap());
        state
            .arrivals
            .entry(task_id)
            .or_insert_with(|| watch::Sender::new(0));
    }

    /// Registers a run started with the messages `in_prompt` in its prompt,
    /// and delivers live every other message queued for the agent that the
    /// run doesn't hold (see [`AgentMessage::held_by`]): those queued since
    /// the prompt was built, and, for a run resumed without a prompt, those
    /// it hasn't seen.
    pub(crate) fn register_run(
        &self,
        agent_name: &str,
        task_id: Uuid,
        sender: mpsc::Sender<UserCommand>,
        in_prompt: &[Uuid],
    ) {
        // One critical section: a message sent meanwhile is either late here
        // or delivered by `send`, never both.
        let mut guard = self.state.lock();
        let state = &mut *guard;
        self.register_locked(state, agent_name, task_id, sender);
        let late = state
            .queues
            .get(agent_name)
            .into_iter()
            .flatten()
            .filter(|message| !in_prompt.contains(&message.id) && message.is_for(task_id))
            .collect::<Vec<_>>();
        if late.is_empty() {
            return;
        }
        let Some(active) = state
            .active
            .get_mut(agent_name)
            .and_then(|instances| instances.get_mut(&task_id))
        else {
            return;
        };
        for message in late {
            let payload = injection(message);
            if message.from != WORKFLOW_CONTEXT_SENDER {
                active
                    .unseen
                    .push((message.id, message.from.clone(), payload.clone()));
            }
            active.outbound.push_back((message.id, payload));
        }
        active.wake.notify_one();
        signal_arrival(&state.arrivals, task_id);
    }

    fn start_pump(&self, agent_name: &str, task_id: Uuid, active: &mut ActiveRecipient) {
        if active.pump.as_ref().is_some_and(|pump| !pump.is_finished()) {
            return;
        }
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let task = runtime.spawn(forward(
                Arc::downgrade(&self.state),
                agent_name.to_string(),
                task_id,
                active.generation,
                active.wake.clone(),
            ));
            active.pump = Some(task.abort_handle());
        }
    }

    pub fn unregister_active(&self, agent_name: &str, task_id: Uuid) {
        let mut state = self.state.lock();
        if let Some(instances) = state.active.get_mut(agent_name) {
            instances.remove(&task_id);
            if instances.is_empty() {
                state.active.remove(agent_name);
            }
        }
        state.arrivals.remove(&task_id);
    }

    /// Counts agent messages arriving for run `task_id` and their forwarding
    /// to it: take the receiver, mark its value seen, then check
    /// [`unseen_messages`](Self::unseen_messages), and no arrival after the
    /// check goes unnoticed.
    pub(crate) fn watch_arrivals(&self, task_id: Uuid) -> watch::Receiver<u64> {
        self.state
            .lock()
            .arrivals
            .entry(task_id)
            .or_insert_with(|| watch::Sender::new(0))
            .subscribe()
    }

    /// The agent messages sent to run `task_id` of `agent_name` since it
    /// started that its conversation doesn't hold yet, as far as
    /// [`observe_context`](Self::observe_context) has been told. Messages in
    /// its initial prompt and context notices don't count.
    pub(crate) fn unseen_messages(&self, agent_name: &str, task_id: Uuid) -> UnseenMessages {
        let state = self.state.lock();
        let Some(active) = state
            .active
            .get(agent_name)
            .and_then(|instances| instances.get(&task_id))
        else {
            return UnseenMessages::default();
        };
        let mut senders = Vec::new();
        for (_, from, _) in &active.unseen {
            if !senders.contains(from) {
                senders.push(from.clone());
            }
        }
        let forwarded = active
            .unseen
            .iter()
            .all(|(id, _, _)| !active.outbound.iter().any(|(outbound, _)| outbound == id));
        UnseenMessages { senders, forwarded }
    }

    /// Resolves once every agent message queued for run `task_id` of
    /// `agent_name` is in its command channel, or the run can't take them
    /// (its channel closed, or it is not registered).
    pub(crate) async fn forwarded(&self, agent_name: &str, task_id: Uuid) {
        let mut arrivals = self.watch_arrivals(task_id);
        loop {
            arrivals.borrow_and_update();
            if !self.is_registered(agent_name, task_id)
                || self.unseen_messages(agent_name, task_id).forwarded
                || arrivals.changed().await.is_err()
            {
                return;
            }
        }
    }

    /// Whether run `task_id` of `agent_name` is registered for delivery.
    fn is_registered(&self, agent_name: &str, task_id: Uuid) -> bool {
        self.state
            .lock()
            .active
            .get(agent_name)
            .is_some_and(|instances| instances.contains_key(&task_id))
    }

    /// Notes a message committed to run `task_id`'s conversation: the agent
    /// messages whose injected payload its context update holds are seen.
    pub(crate) fn observe_context(
        &self,
        agent_name: &str,
        task_id: Uuid,
        message: &arko_ai::Message,
    ) {
        let Some(update) = context_update(message) else {
            return;
        };
        let mut state = self.state.lock();
        if let Some(active) = state
            .active
            .get_mut(agent_name)
            .and_then(|instances| instances.get_mut(&task_id))
        {
            active
                .unseen
                .retain(|(_, _, payload)| !contains_injection(update, payload));
        }
    }

    /// Queue a task-specific context notice without waiting for the recipient.
    /// True means retained for a known active task, not yet consumed. The
    /// queue is persisted by workflow checkpoints, just like inter-agent
    /// messages. A notice is for its run only. It is dropped when that run
    /// stops (`settle_run`).
    pub async fn inject_task(&self, task_id: Uuid, content: String) -> bool {
        let mut state = self.state.lock();
        let agent_name = state
            .active
            .iter()
            .find_map(|(name, instances)| instances.contains_key(&task_id).then(|| name.clone()));
        let Some(agent_name) = agent_name else {
            return false;
        };
        let message = AgentMessage {
            id: Uuid::new_v4(),
            from: WORKFLOW_CONTEXT_SENDER.to_string(),
            from_task_id: None,
            to: agent_name.clone(),
            content,
            attachments: Vec::new(),
            timestamp: chrono::Utc::now(),
            to_task_id: Some(task_id),
            held_by: Vec::new(),
            read_by: Vec::new(),
        };
        let content = injection(&message);
        state
            .queues
            .entry(agent_name.clone())
            .or_default()
            .push(message.clone());
        let active = state
            .active
            .get_mut(&agent_name)
            .unwrap()
            .get_mut(&task_id)
            .unwrap();
        active.outbound.push_back((message.id, content));
        self.start_pump(&agent_name, task_id, active);
        active.wake.notify_one();
        true
    }

    /// Retain a message for checkpoint persistence and schedule active delivery
    /// without waiting for the recipient's command channel. It leaves the
    /// queue once a run of the recipient completes with it in its
    /// conversation (`settle_run`). Until then it is
    /// presented to every run of the recipient that doesn't hold it.
    pub async fn send(
        &self,
        from: &str,
        from_task_id: Option<Uuid>,
        to: &str,
        content: String,
        attachments: Vec<MessageAttachment>,
    ) -> Result<AgentMessage, String> {
        if to == from {
            return Err(format!("'{to}' is not a permitted recipient"));
        }
        if !self.roster.contains(to) {
            return Err(format!(
                "unknown recipient '{to}'; valid recipients: {}",
                self.roster().join(", ")
            ));
        }

        let msg = AgentMessage {
            id: Uuid::new_v4(),
            from: from.to_string(),
            from_task_id,
            to: to.to_string(),
            content,
            attachments,
            timestamp: chrono::Utc::now(),
            to_task_id: None,
            held_by: Vec::new(),
            read_by: Vec::new(),
        };

        let mut state = self.state.lock();
        state
            .queues
            .entry(to.to_string())
            .or_default()
            .push(msg.clone());

        let state = &mut *state;
        if let Some(instances) = state.active.get_mut(to) {
            let content = injection(&msg);
            for (task_id, active) in instances {
                if msg.from != WORKFLOW_CONTEXT_SENDER {
                    active
                        .unseen
                        .push((msg.id, msg.from.clone(), content.clone()));
                }
                active.outbound.push_back((msg.id, content.clone()));
                self.start_pump(to, *task_id, active);
                active.wake.notify_one();
                signal_arrival(&state.arrivals, *task_id);
            }
        }

        Ok(msg)
    }

    /// Drain all pending messages for the given agent.
    pub fn drain(&self, agent_name: &str) -> Vec<AgentMessage> {
        self.state
            .lock()
            .queues
            .remove(agent_name)
            .unwrap_or_default()
    }

    /// Every message queued for the agent.
    pub fn pending(&self, agent_name: &str) -> Vec<AgentMessage> {
        self.state
            .lock()
            .queues
            .get(agent_name)
            .cloned()
            .unwrap_or_default()
    }

    /// The messages queued for the agent that are for run `task_id` (see
    /// [`AgentMessage::is_for`]), matching what its prompt presents.
    pub fn pending_for(&self, agent_name: &str, task_id: Uuid) -> Vec<AgentMessage> {
        self.state
            .lock()
            .queues
            .get(agent_name)
            .into_iter()
            .flatten()
            .filter(|message| message.is_for(task_id))
            .cloned()
            .collect()
    }

    pub fn acknowledge(&self, agent_name: &str, message_ids: &[Uuid]) {
        acknowledge(&mut self.state.lock(), agent_name, message_ids);
    }

    /// Records what run `task_id` of `agent_name` holds as it stops (see
    /// [`AgentMessage::held_by`]): the queued messages its conversation holds
    /// (presented in `prompt_ids` or an earlier prompt, or added as a
    /// runtime context update, an echo in an assistant turn, a tool result,
    /// or a suffix doesn't count) and those it took from its command channel
    /// but hadn't added yet (`taken`, its `pending_context_injections`).
    /// None of them is presented to the run again when it is continued or
    /// resumed. When the run `completed`, the messages in its conversation
    /// are acknowledged and leave the queue. One it only took stays for the
    /// runs that haven't read it. The run's context notices are dropped
    /// whatever the outcome: they describe the run's own context.
    pub(crate) fn settle_run(
        &self,
        agent_name: &str,
        task_id: Uuid,
        conversation: &arko_ai::Conversation,
        prompt_ids: &[Uuid],
        taken: &[String],
        completed: bool,
    ) {
        // The conversation is searched outside the lock (it can be long);
        // messages queued meanwhile aren't in it.
        let queued = self
            .state
            .lock()
            .queues
            .get(agent_name)
            .cloned()
            .unwrap_or_default();
        let mut in_conversation = HashSet::new();
        let mut held = HashSet::new();
        for message in &queued {
            if prompt_ids.contains(&message.id)
                || message.read_by.contains(&task_id)
                || conversation_holds(conversation, message)
            {
                in_conversation.insert(message.id);
                held.insert(message.id);
            } else if taken
                .iter()
                .any(|content| is_injection_of(content, message.id))
            {
                held.insert(message.id);
            }
        }
        let mut state = self.state.lock();
        let Some(messages) = state.queues.get_mut(agent_name) else {
            return;
        };
        messages.retain_mut(|message| {
            if message.to_task_id == Some(task_id) {
                return false;
            }
            if completed && in_conversation.contains(&message.id) {
                return false;
            }
            if held.contains(&message.id) && !message.is_held_by(task_id) {
                message.held_by.push(task_id);
            }
            true
        });
        if messages.is_empty() {
            state.queues.remove(agent_name);
        }
    }

    /// Records that run `task_id` of `agent_name` is handing over
    /// `conversation` (see [`arko_agents::handover`]). The queued messages
    /// it holds (presented in `prompt_ids` or a prompt, or added as a
    /// runtime context update) are marked as read by the run
    /// ([`AgentMessage::read_by`]) and held by it, so they are neither
    /// presented to it again nor left unacknowledged when it completes. Its
    /// context notices in the conversation are dropped. Those it hasn't
    /// received yet stay.
    pub fn hand_over(
        &self,
        agent_name: &str,
        task_id: Uuid,
        conversation: &arko_ai::Conversation,
        prompt_ids: &[Uuid],
    ) {
        let queued = self
            .state
            .lock()
            .queues
            .get(agent_name)
            .cloned()
            .unwrap_or_default();
        let read: HashSet<Uuid> = queued
            .iter()
            .filter(|message| {
                prompt_ids.contains(&message.id) || conversation_holds(conversation, message)
            })
            .map(|message| message.id)
            .collect();
        let mut state = self.state.lock();
        let Some(messages) = state.queues.get_mut(agent_name) else {
            return;
        };
        messages.retain_mut(|message| {
            if !read.contains(&message.id) {
                return true;
            }
            if message.to_task_id == Some(task_id) {
                return false;
            }
            if !message.read_by.contains(&task_id) {
                message.read_by.push(task_id);
            }
            if !message.is_held_by(task_id) {
                message.held_by.push(task_id);
            }
            true
        });
        if messages.is_empty() {
            state.queues.remove(agent_name);
        }
    }

    /// Snapshot for persistence.
    pub fn snapshot(&self) -> HashMap<String, Vec<AgentMessage>> {
        self.state.lock().queues.clone()
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn conversation() -> arko_ai::Conversation {
        arko_ai::Conversation::new(arko_ai::Model::openai_gpt_6_astra())
    }

    /// Settles run `task_id` as completed with `content` added as a context
    /// update.
    fn confirm(bus: &MessageBus, agent_name: &str, task_id: Uuid, content: &str) {
        let mut conversation = conversation();
        conversation.add_user_message(&format!("# Workflow context update\n\n{content}"));
        bus.settle_run(agent_name, task_id, &conversation, &[], &[], true);
    }

    /// Settles run `task_id` as completed with nothing in its conversation.
    fn complete_empty(bus: &MessageBus, agent_name: &str, task_id: Uuid) {
        bus.settle_run(agent_name, task_id, &conversation(), &[], &[], true);
    }

    // Test watchdogs detect deadlocks; production delivery has no forced timeout.
    async fn received(receiver: &mut mpsc::Receiver<UserCommand>) -> String {
        let command = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
            .await
            .expect("delivery stalled")
            .expect("delivery channel closed");
        let UserCommand::InjectContext { content } = command else {
            panic!("expected a context injection");
        };
        content
    }

    fn update(content: &str) -> arko_ai::Message {
        arko_ai::Message::new_user(&format!("# Workflow context update\n\n{content}"))
    }

    /// A run's unseen agent messages: sent after it started, not yet in its
    /// conversation. Messages in its prompt, delivered ones (still queued
    /// until acknowledged) and context notices don't count. Each arrival and
    /// forward is signalled.
    #[tokio::test]
    async fn a_run_tracks_the_agent_messages_its_conversation_lacks() {
        let bus = MessageBus::new(["lead".to_string(), "worker".to_string()]);
        let run = Uuid::new_v4();
        let early = bus
            .send("worker", None, "lead", "in the prompt".into(), vec![])
            .await
            .unwrap();
        let mut arrivals = bus.watch_arrivals(run);
        arrivals.borrow_and_update();
        let (sender, mut receiver) = mpsc::channel(4);
        bus.register_run("lead", run, sender, &[early.id]);
        assert_eq!(
            bus.unseen_messages("lead", run),
            UnseenMessages {
                senders: vec![],
                forwarded: true,
            }
        );

        assert!(bus.inject_task(run, "file changed".into()).await);
        assert!(received(&mut receiver).await.contains("file changed"));
        assert!(bus.unseen_messages("lead", run).senders.is_empty());

        bus.send("worker", None, "lead", "reply".into(), vec![])
            .await
            .unwrap();
        assert!(arrivals.has_changed().unwrap());
        let delivered = received(&mut receiver).await;
        let unseen = bus.unseen_messages("lead", run);
        assert_eq!(unseen.senders, ["worker"]);
        assert!(unseen.forwarded);

        // An update without it, or a copy outside a context update, is not
        // the delivery.
        bus.observe_context("lead", run, &update("unrelated"));
        bus.observe_context("lead", run, &arko_ai::Message::new_user(&delivered));
        assert_eq!(bus.unseen_messages("lead", run).senders, ["worker"]);
        bus.observe_context("lead", run, &update(&delivered));
        assert!(bus.unseen_messages("lead", run).senders.is_empty());
        // Still queued, with the prompt's message and the notice, until the
        // run completes and acknowledges them.
        assert_eq!(bus.pending("lead").len(), 3);

        // Unregistered: nothing is tracked, and the signal is gone.
        arrivals.borrow_and_update();
        bus.unregister_active("lead", run);
        assert!(bus.unseen_messages("lead", run).senders.is_empty());
        assert!(arrivals.changed().await.is_err());
    }

    /// A message sent while a run registers reaches it once: with the late
    /// messages its registration queues, or through the send, never both.
    /// (With registration and the search for late messages in two critical
    /// sections, a send between them was delivered twice.)
    #[test]
    fn a_message_sent_while_a_run_registers_is_delivered_once() {
        for _ in 0..2_000 {
            let bus = Arc::new(MessageBus::new(["lead".to_string(), "worker".to_string()]));
            let run = Uuid::new_v4();
            let (sender, _receiver) = mpsc::channel(4);
            let start = Arc::new(std::sync::Barrier::new(2));
            let registering = std::thread::spawn({
                let (bus, start) = (bus.clone(), start.clone());
                move || {
                    start.wait();
                    bus.register_run("lead", run, sender, &[]);
                }
            });
            let sending = std::thread::spawn({
                let (bus, start) = (bus.clone(), start.clone());
                move || {
                    start.wait();
                    let send = bus.send("worker", None, "lead", "hello".into(), vec![]);
                    let mut send = std::pin::pin!(send);
                    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
                    let std::task::Poll::Ready(sent) = send.as_mut().poll(&mut context) else {
                        panic!("a send never waits");
                    };
                    sent.unwrap().id
                }
            });
            registering.join().unwrap();
            let id = sending.join().unwrap();
            // Outside a Tokio runtime no delivery worker runs: every delivery
            // stays in the run's outbound queue.
            let state = bus.state.lock();
            let deliveries = state.active["lead"][&run]
                .outbound
                .iter()
                .filter(|(queued, _)| *queued == id)
                .count();
            assert_eq!(deliveries, 1);
        }
    }

    /// Waiting for the queue to be forwarded ends when the run's channel
    /// closes before it could take them (a runtime that exited at once).
    #[tokio::test]
    async fn waiting_for_forwards_ends_when_the_run_closes() {
        let bus = MessageBus::new(["lead".to_string(), "worker".to_string()]);
        let run = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel(1);
        sender
            .try_send(UserCommand::InjectContext {
                content: "fills the channel".into(),
            })
            .unwrap();
        bus.send("worker", None, "lead", "queued".into(), vec![])
            .await
            .unwrap();
        bus.register_run("lead", run, sender, &[]);
        let wait = bus.forwarded("lead", run);
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(10), wait)
            .await
            .expect("the wait ended");
        // An unregistered run doesn't wait at all.
        tokio::time::timeout(
            Duration::from_secs(10),
            bus.forwarded("lead", Uuid::new_v4()),
        )
        .await
        .expect("no wait for an unknown run");
    }

    /// A message the delivery worker has not forwarded yet (the command
    /// channel is full) is unseen but not forwarded.
    #[tokio::test]
    async fn an_unforwarded_message_is_reported_as_such() {
        let bus = MessageBus::new(["lead".to_string(), "worker".to_string()]);
        let run = Uuid::new_v4();
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .try_send(UserCommand::InjectContext {
                content: "fills the channel".into(),
            })
            .unwrap();
        bus.register_run("lead", run, sender, &[]);
        let mut arrivals = bus.watch_arrivals(run);
        arrivals.borrow_and_update();
        bus.send("worker", None, "lead", "first".into(), vec![])
            .await
            .unwrap();
        bus.send("lead", None, "worker", "not for lead".into(), vec![])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            bus.unseen_messages("lead", run),
            UnseenMessages {
                senders: vec!["worker".into()],
                forwarded: false,
            }
        );
        arrivals.borrow_and_update();
        assert!(received(&mut receiver).await.contains("fills the channel"));
        // The forward is signalled.
        tokio::time::timeout(Duration::from_secs(5), arrivals.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(bus.unseen_messages("lead", run).forwarded);
        assert!(received(&mut receiver).await.contains("first"));
    }

    #[tokio::test]
    async fn active_delivery_remains_pending_until_acknowledged() {
        let bus = MessageBus::new(["coder".to_string(), "reviewer".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, mut receiver) = mpsc::channel(1);
        bus.register_active("reviewer", task_id, sender);

        let message = bus
            .send(
                "coder",
                Some(Uuid::new_v4()),
                "reviewer",
                "Please inspect this file".to_string(),
                vec![MessageAttachment::File {
                    path: "src/lib.rs".to_string(),
                }],
            )
            .await
            .unwrap();

        let UserCommand::InjectContext { content } = receiver.recv().await.unwrap() else {
            panic!("expected a context injection");
        };
        assert!(content.contains(&message.id.to_string()));
        assert!(content.contains("src/lib.rs"));
        assert_eq!(bus.pending("reviewer").len(), 1);

        confirm(&bus, "reviewer", task_id, &content);
        assert!(bus.pending("reviewer").is_empty());
    }

    #[tokio::test]
    async fn completed_task_does_not_acknowledge_unread_context() {
        let bus = MessageBus::new(["creator".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel(2);
        bus.register_active("creator", task_id, sender);
        let message = bus
            .send(
                "critic",
                None,
                "creator",
                "consequential correction".into(),
                vec![],
            )
            .await
            .unwrap();
        assert!(bus.inject_task(task_id, "evidence changed".into()).await);
        tokio::time::timeout(Duration::from_secs(5), async {
            while receiver.len() != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("messages were not forwarded to the runtime channel");

        // complete_task can return before the runtime reads these commands or
        // commits their text to Task.conversation. Finalization must preserve
        // the message even though channel delivery was successful; the
        // run's own context notice ends with it.
        complete_empty(&bus, "creator", task_id);
        bus.unregister_active("creator", task_id);
        drop(receiver);
        let pending = bus.pending("creator");
        assert_eq!(pending.len(), 1, "completion discarded unread context");
        assert_eq!(pending[0].id, message.id);
        assert!(pending[0].held_by.is_empty());
    }

    #[tokio::test]
    async fn confirmation_requires_complete_runtime_update_not_echoes_or_suffixes() {
        let bus = MessageBus::new(["creator".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, mut receiver) = mpsc::channel(1);
        bus.register_active("creator", task_id, sender);
        let message = bus
            .send(
                "critic",
                None,
                "creator",
                "exact full correction".into(),
                vec![],
            )
            .await
            .unwrap();
        let payload = received(&mut receiver).await;
        let framed = format!("# Workflow context update\n\n{payload}");
        let invalid = [
            arko_ai::Message::new_user(&payload),
            arko_ai::Message::new_assistant(&framed),
            arko_ai::Message::new_user_with_contents(vec![arko_ai::Content::ToolResult(
                arko_ai::ToolResult::new(None, "echo", &framed),
            )]),
            arko_ai::Message::new_user(&format!("# Workflow context update\n\n{}", message.id)),
            arko_ai::Message::new_user(&framed[..framed.len() - 1]),
            arko_ai::Message::new_user(&format!(
                "# Workflow context update\n\nnot-a-boundary{payload}"
            )),
            arko_ai::Message::new_user(&format!(
                "# Workflow context update\n\n{payload}incomplete-boundary"
            )),
            arko_ai::Message::new_user(&format!(
                "# Workflow context update\n\nother update\n\n# Reminder\n{payload}"
            )),
            arko_ai::Message::new_user(&format!(
                "# Workflow context update\n\nother update\n\n# Current Environment Information (updated this turn)\n{payload}"
            )),
        ];
        for context in invalid {
            let mut conversation = conversation();
            conversation.messages.push(context);
            bus.settle_run("creator", task_id, &conversation, &[], &[], true);
            assert_eq!(bus.pending("creator").len(), 1);
            assert!(bus.pending("creator")[0].held_by.is_empty());
        }
        let mut conversation = conversation();
        conversation.add_user_message(&format!("{framed}\n\n# Current Environment Information (updated this turn)\nignored metadata\n\n# Reminder\nContinue carefully"));
        bus.settle_run("creator", task_id, &conversation, &[], &[], true);
        assert!(bus.pending("creator").is_empty());
    }

    #[tokio::test]
    async fn finalization_confirms_consumed_context_and_preserves_unread_message() {
        let config = arko_agents::AgentConfig::builder(
            "creator".into(),
            "Creator".into(),
            arko_agents::AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_6_astra(),
                generation_config: arko_ai::GenerationConfig::default(),
            },
        )
        .role_description("Test creator".into())
        .instructions("Test only".into())
        .build()
        .unwrap();
        // No runtime/model request: feed a finished Task to the real shared
        // finalizer, which all three workflow completion paths use.
        let workflow = crate::workflow::Workflow::builder()
            .name("message-finalization-test")
            .workspace(std::env::temp_dir())
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("creator", config.clone())
            .build()
            .unwrap();
        let mut task = arko_agents::Task::new(config, "Finish".into(), std::env::temp_dir(), None);
        let task_id = task.id;
        let bus = &workflow.context().messages;
        let (sender, mut receiver) = mpsc::channel(2);
        bus.register_active("creator", task_id, sender);
        let consumed = bus
            .send("critic", None, "creator", "seen correction".into(), vec![])
            .await
            .unwrap();
        let unread = bus
            .send("critic", None, "creator", "late correction".into(), vec![])
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while receiver.len() != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let payload = received(&mut receiver).await;
        assert_eq!(payload, injection(&consumed));
        task.conversation
            .add_user_message(&format!("# Workflow context update\n\n{payload}"));
        task.set_state(arko_agents::task::TaskState::Completed {
            summary: "Done".into(),
        });
        crate::workflow::engine::finalize_agent_task(
            "creator".into(),
            task,
            &[],
            workflow.context(),
            None,
        )
        .await
        .unwrap();
        bus.unregister_active("creator", task_id);
        drop(receiver);
        let restored = MessageBus::from_snapshot(bus.roster(), bus.snapshot());
        assert_eq!(
            restored
                .pending("creator")
                .iter()
                .map(|message| message.id)
                .collect::<Vec<_>>(),
            vec![unread.id]
        );
    }

    /// A message queued after a run's prompt was built, before the run was
    /// registered, is delivered to it live. One in its prompt is not
    /// delivered again.
    #[tokio::test]
    async fn a_run_receives_messages_sent_while_it_started() {
        let bus = MessageBus::new(["coder".to_string(), "reviewer".to_string()]);
        let in_prompt = bus
            .send("coder", None, "reviewer", "in the prompt".into(), vec![])
            .await
            .unwrap();
        bus.send("coder", None, "reviewer", "sent meanwhile".into(), vec![])
            .await
            .unwrap();
        let (sender, mut receiver) = mpsc::channel(4);
        bus.register_run("reviewer", Uuid::new_v4(), sender, &[in_prompt.id]);
        let delivered = received(&mut receiver).await;
        assert!(delivered.contains("sent meanwhile"), "{delivered}");
        assert!(receiver.try_recv().is_err());
    }

    #[tokio::test]
    async fn unknown_recipients_are_rejected_without_queueing() {
        let bus = MessageBus::new(["coder".to_string()]);
        let error = bus
            .send("coder", None, "ghost", "hello".to_string(), vec![])
            .await
            .unwrap_err();
        assert!(error.contains("unknown recipient"));
        assert!(bus.snapshot().is_empty());
    }

    #[tokio::test]
    async fn saturated_recipient_does_not_block_messages_or_freshness_notices() {
        let bus = MessageBus::new(["creator".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(UserCommand::InjectContext {
                content: "runtime is executing consult_design".into(),
            })
            .await
            .unwrap();
        bus.register_active("creator", task_id, sender);

        tokio::time::timeout(Duration::from_secs(5), async {
            for index in 0..32 {
                bus.send(
                    "designer",
                    None,
                    "creator",
                    format!("finding {index}"),
                    vec![],
                )
                .await
                .unwrap();
                assert!(bus.inject_task(task_id, format!("refresh {index}")).await);
            }
        })
        .await
        .expect("collaborators waited for a busy recipient");

        let pending = bus.pending("creator");
        assert_eq!(pending.len(), 64);
        assert_eq!(
            pending
                .iter()
                .map(|msg| msg.id)
                .collect::<HashSet<_>>()
                .len(),
            64
        );
        assert_eq!(
            received(&mut receiver).await,
            "runtime is executing consult_design"
        );
        for expected in &pending {
            assert_eq!(received(&mut receiver).await, injection(expected));
        }
        confirm(
            &bus,
            "creator",
            task_id,
            &pending
                .iter()
                .map(injection)
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
        assert!(bus.pending("creator").is_empty());
        bus.unregister_active("creator", task_id);
    }

    #[tokio::test]
    async fn slow_active_instance_does_not_block_another_instance() {
        let bus = MessageBus::new(["reviewer".to_string()]);
        let (slow_sender, mut slow_receiver) = mpsc::channel(1);
        slow_sender
            .send(UserCommand::InjectContext {
                content: "busy".into(),
            })
            .await
            .unwrap();
        let (fast_sender, mut fast_receiver) = mpsc::channel(1);
        bus.register_active("reviewer", Uuid::new_v4(), slow_sender);
        bus.register_active("reviewer", Uuid::new_v4(), fast_sender);
        let message = bus
            .send("creator", None, "reviewer", "review this".into(), vec![])
            .await
            .unwrap();
        assert_eq!(received(&mut fast_receiver).await, injection(&message));
        assert_eq!(received(&mut slow_receiver).await, "busy");
        assert_eq!(received(&mut slow_receiver).await, injection(&message));
    }

    #[tokio::test]
    async fn unregister_stops_forwarding_and_snapshot_preserves_unsent_notices() {
        let bus = MessageBus::new(["creator".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(UserCommand::InjectContext {
                content: "busy".into(),
            })
            .await
            .unwrap();
        bus.register_active("creator", task_id, sender);
        let message = bus
            .send(
                "critic",
                None,
                "creator",
                "unfinished finding".into(),
                vec![],
            )
            .await
            .unwrap();
        assert!(bus.inject_task(task_id, "changed evidence".into()).await);
        tokio::task::yield_now().await;
        bus.unregister_active("creator", task_id);
        assert_eq!(received(&mut receiver).await, "busy");
        assert!(
            tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                .await
                .unwrap()
                .is_none()
        );
        let snapshot = bus.snapshot();
        let encoded = serde_json::to_vec(&snapshot).unwrap();
        let restored =
            MessageBus::from_snapshot(bus.roster(), serde_json::from_slice(&encoded).unwrap());
        let pending = restored.pending("creator");
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].id, message.id);
        assert_eq!(pending[1].id, snapshot["creator"][1].id);
        assert_eq!(pending[1].content, "changed evidence");
        // The workflow's next invocation includes these pending messages in its
        // prompt, and acknowledges these exact IDs only on completion.
        restored.acknowledge(
            "creator",
            &pending.iter().map(|message| message.id).collect::<Vec<_>>(),
        );
        assert!(restored.pending("creator").is_empty());
    }

    #[tokio::test]
    async fn replaced_task_cannot_receive_or_acknowledge_old_queued_delivery() {
        let bus = MessageBus::new(["creator".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, mut old_receiver) = mpsc::channel(1);
        sender
            .send(UserCommand::InjectContext {
                content: "busy".into(),
            })
            .await
            .unwrap();
        bus.register_active("creator", task_id, sender);
        let old = bus
            .send("critic", None, "creator", "before restart".into(), vec![])
            .await
            .unwrap();
        tokio::task::yield_now().await;
        let (sender, mut new_receiver) = mpsc::channel(1);
        bus.register_active("creator", task_id, sender);
        let new = bus
            .send("critic", None, "creator", "after restart".into(), vec![])
            .await
            .unwrap();
        assert_eq!(received(&mut new_receiver).await, injection(&new));
        assert_eq!(received(&mut old_receiver).await, "busy");
        assert!(
            tokio::time::timeout(Duration::from_secs(5), old_receiver.recv())
                .await
                .unwrap()
                .is_none()
        );
        confirm(&bus, "creator", task_id, &injection(&new));
        assert_eq!(
            bus.pending("creator")
                .iter()
                .map(|msg| msg.id)
                .collect::<Vec<_>>(),
            vec![old.id]
        );
    }

    #[tokio::test]
    async fn closed_recipient_retains_queued_message_and_notice_for_recovery() {
        let bus = MessageBus::new(["creator".to_string()]);
        let task_id = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel(1);
        let weak_sender = sender.downgrade();
        bus.register_active("creator", task_id, sender);
        let message = bus
            .send("critic", None, "creator", "finding".into(), vec![])
            .await
            .unwrap();
        assert!(bus.inject_task(task_id, "freshness notice".into()).await);
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(5), async {
            while weak_sender.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("closed recipient retained its forwarding task");
        assert_eq!(bus.pending("creator").len(), 2);
        assert_eq!(bus.pending("creator")[0].id, message.id);
        assert!(!bus.inject_task(task_id, "no longer active".into()).await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn immediate_completion_acknowledgement_cannot_race_delivery_bookkeeping() {
        let bus = Arc::new(MessageBus::new(["reviewer".to_string()]));
        let task_id = Uuid::new_v4();
        let (sender, mut receiver) = mpsc::channel(1);
        bus.register_active("reviewer", task_id, sender);
        let consumer_bus = bus.clone();
        let consumer = tokio::spawn(async move {
            for _ in 0..256 {
                let content = received(&mut receiver).await;
                confirm(&consumer_bus, "reviewer", task_id, &content);
            }
        });
        for index in 0..256 {
            bus.send(
                "creator",
                None,
                "reviewer",
                format!("message {index}"),
                vec![],
            )
            .await
            .unwrap();
        }
        tokio::time::timeout(Duration::from_secs(5), consumer)
            .await
            .unwrap()
            .unwrap();
        assert!(bus.pending("reviewer").is_empty());
    }

    #[test]
    fn synchronous_registration_outside_tokio_starts_delivery_on_later_send() {
        let bus = MessageBus::new(["reviewer".to_string()]);
        let (sender, mut receiver) = mpsc::channel(1);
        let task_id = Uuid::new_v4();
        bus.register_active("reviewer", task_id, sender);
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let message = bus
                    .send("creator", None, "reviewer", "later runtime".into(), vec![])
                    .await
                    .unwrap();
                assert_eq!(received(&mut receiver).await, injection(&message));
                confirm(&bus, "reviewer", task_id, &injection(&message));
                assert!(bus.pending("reviewer").is_empty());
            });
    }

    #[tokio::test]
    async fn dropping_bus_releases_a_pump_waiting_for_channel_capacity() {
        let bus = MessageBus::new(["creator".to_string()]);
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(UserCommand::InjectContext {
                content: "busy".into(),
            })
            .await
            .unwrap();
        bus.register_active("creator", Uuid::new_v4(), sender);
        bus.send("critic", None, "creator", "pending".into(), vec![])
            .await
            .unwrap();
        tokio::task::yield_now().await;
        drop(bus);
        assert_eq!(received(&mut receiver).await, "busy");
        assert!(
            tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                .await
                .unwrap()
                .is_none()
        );
    }

    /// A handover marks the messages the old conversation held as read by
    /// the run: they are not presented to it again, and they are
    /// acknowledged when it completes though the fresh conversation lacks
    /// them. Its own notices in the old conversation go. Unread messages
    /// stay.
    #[tokio::test]
    async fn a_handover_keeps_what_the_old_conversation_read() {
        let bus = MessageBus::new(["lead".to_string(), "worker".to_string()]);
        let run = Uuid::new_v4();
        let read = bus
            .send("lead", None, "worker", "read".into(), vec![])
            .await
            .unwrap();
        let unread = bus
            .send("lead", None, "worker", "unread".into(), vec![])
            .await
            .unwrap();
        let notice = AgentMessage {
            id: Uuid::new_v4(),
            from: WORKFLOW_CONTEXT_SENDER.to_string(),
            from_task_id: None,
            to: "worker".into(),
            content: "a file changed".into(),
            attachments: Vec::new(),
            timestamp: chrono::Utc::now(),
            to_task_id: Some(run),
            held_by: Vec::new(),
            read_by: Vec::new(),
        };
        bus.state
            .lock()
            .queues
            .get_mut("worker")
            .unwrap()
            .push(notice.clone());
        let mut old = conversation();
        old.messages.push(update(&injection(&read)));
        old.messages.push(update(&injection(&notice)));
        bus.hand_over("worker", run, &old, &[]);

        let pending = bus.pending("worker");
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].id, read.id);
        assert_eq!(pending[0].read_by, [run]);
        assert!(pending[0].is_held_by(run));
        assert_eq!(pending[1].id, unread.id);
        assert!(pending[1].read_by.is_empty());
        assert_eq!(
            bus.pending_for("worker", run)
                .iter()
                .map(|message| message.id)
                .collect::<Vec<_>>(),
            [unread.id]
        );

        // The fresh conversation holds neither.
        complete_empty(&bus, "worker", run);
        assert_eq!(
            bus.pending("worker")
                .iter()
                .map(|message| message.id)
                .collect::<Vec<_>>(),
            [unread.id]
        );
    }
}
