//! Inter-agent messaging system.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Weak},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Notify, mpsc},
    task::AbortHandle,
};
use uuid::Uuid;

use arko_agents::runtime::command::UserCommand;

pub mod registry;
pub mod tools;

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
/// (the default for every agent); [`can_message`](Self::can_message)
/// restricts an agent to named recipients, or to none. An agent is offered
/// `send_message` only when it has at least one permitted recipient, so a
/// single-agent workflow has none.
///
/// ```rust
/// use arko_workflows::messaging::MessagingPolicy;
///
/// // Workers report to the lead only; the lead may message everyone.
/// let policy = MessagingPolicy::all()
///     .can_message("worker_a", ["lead"])
///     .can_message("worker_b", ["lead"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MessagingPolicy {
    /// For each agent listed, the only agents it may message (none when the
    /// set is empty); an agent without an entry may message every other
    /// agent.
    pub recipients: HashMap<String, HashSet<String>>,
}

impl MessagingPolicy {
    /// Every agent may message every other agent (the default).
    pub fn all() -> Self {
        Self::default()
    }

    /// `agent` may message only `recipients`; an empty list means no one.
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

    /// `agent` may message no one and is not offered `send_message`.
    pub fn cannot_message(self, agent: impl Into<String>) -> Self {
        self.can_message(agent, std::iter::empty::<String>())
    }

    /// Whether `sender` may message `recipient`; never itself.
    pub fn permits(&self, sender: &str, recipient: &str) -> bool {
        sender != recipient
            && self
                .recipients
                .get(sender)
                .is_none_or(|allowed| allowed.contains(recipient))
    }

    /// The agents of `roster` that `sender` may message, in roster order;
    /// never the sender itself.
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
}

/// Per-agent message queue. Active recipients receive asynchronous context
/// injections; later invocations receive pending messages in their prompt.
/// Messages remain snapshot-backed until the workflow acknowledges completion.
pub struct MessageBus {
    state: Arc<Mutex<BusState>>,
    roster: HashSet<String>,
}

#[derive(Default)]
struct BusState {
    queues: HashMap<String, Vec<AgentMessage>>,
    active: HashMap<String, HashMap<Uuid, ActiveRecipient>>,
}

struct ActiveRecipient {
    generation: Uuid,
    sender: mpsc::Sender<UserCommand>,
    pending_message_ids: Vec<Uuid>,
    confirmed_message_ids: HashSet<Uuid>,
    outbound: VecDeque<(Uuid, String)>,
    wake: Arc<Notify>,
    pump: Option<AbortHandle>,
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
        if let Some((id, content)) = active.outbound.pop_front() {
            active.pending_message_ids.push(id);
            permit.send(UserCommand::InjectContext { content });
        }
    }
}

fn injection(message: &AgentMessage) -> String {
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
                active: HashMap::new(),
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
        let mut state = self.state.lock();
        let active = state.active.entry(agent_name.to_string()).or_default();
        active.insert(
            task_id,
            ActiveRecipient {
                generation: Uuid::new_v4(),
                sender,
                pending_message_ids: Vec::new(),
                confirmed_message_ids: HashSet::new(),
                outbound: VecDeque::new(),
                wake: Arc::new(Notify::new()),
                pump: None,
            },
        );
        self.start_pump(agent_name, task_id, active.get_mut(&task_id).unwrap());
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
    }

    /// Queue a task-specific context notice without waiting for the recipient.
    /// True means retained for a known active task, not yet consumed. The queue
    /// is persisted by workflow checkpoints, just like inter-agent messages.
    /// Unacknowledged notices survive snapshot/recovery as ordinary messages.
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
            from: "workflow-context".to_string(),
            from_task_id: None,
            to: agent_name.clone(),
            content,
            attachments: Vec::new(),
            timestamp: chrono::Utc::now(),
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
    /// without waiting for the recipient's command channel. Only confirmed
    /// context consumption followed by completion acknowledgement removes it;
    /// interrupted delivery may be presented again on resume.
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
        };

        let mut state = self.state.lock();
        state
            .queues
            .entry(to.to_string())
            .or_default()
            .push(msg.clone());

        if let Some(instances) = state.active.get_mut(to) {
            let content = injection(&msg);
            for (task_id, active) in instances {
                active.outbound.push_back((msg.id, content.clone()));
                self.start_pump(to, *task_id, active);
                active.wake.notify_one();
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

    pub fn pending(&self, agent_name: &str) -> Vec<AgentMessage> {
        self.state
            .lock()
            .queues
            .get(agent_name)
            .cloned()
            .unwrap_or_default()
    }

    pub fn acknowledge(&self, agent_name: &str, message_ids: &[Uuid]) {
        acknowledge(&mut self.state.lock(), agent_name, message_ids);
    }

    /// Confirm channel-enqueued messages committed to a finalized conversation.
    /// Only the runtime's complete User context-update text qualifies, never an
    /// assistant echo, tool result, arbitrary prompt, or bare message ID.
    pub fn confirm_active_context(
        &self,
        agent_name: &str,
        task_id: Uuid,
        conversation: &arko_ai::Conversation,
    ) {
        let (generation, payloads) = {
            let state = self.state.lock();
            let Some(active) = state
                .active
                .get(agent_name)
                .and_then(|instances| instances.get(&task_id))
            else {
                return;
            };
            let enqueued = active
                .pending_message_ids
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let payloads = state
                .queues
                .get(agent_name)
                .into_iter()
                .flatten()
                .filter(|message| enqueued.contains(&message.id))
                .map(|message| (message.id, injection(message)))
                .collect::<Vec<_>>();
            (active.generation, payloads)
        };
        // PromptBuilder appends environment details/reminders after the runtime
        // update. Do not mistake copies appearing in those suffixes for delivery.
        // An identical marker inside message content is conservatively unconfirmed.
        let updates = conversation
            .messages
            .iter()
            .filter_map(|message| {
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
            })
            .collect::<Vec<_>>();
        let confirmed = payloads
            .into_iter()
            .filter_map(|(id, payload)| {
                updates
                    .iter()
                    .any(|update| contains_injection(update, &payload))
                    .then_some(id)
            })
            .collect::<Vec<_>>();
        let mut state = self.state.lock();
        if let Some(active) = state
            .active
            .get_mut(agent_name)
            .and_then(|instances| instances.get_mut(&task_id))
            .filter(|active| active.generation == generation)
        {
            active.confirmed_message_ids.extend(confirmed);
        }
    }

    /// Acknowledge only payloads confirmed in the finalized task's context.
    /// Merely accepting an InjectContext command into its channel is insufficient.
    pub fn acknowledge_active(&self, agent_name: &str, task_id: Uuid) {
        let mut state = self.state.lock();
        let ids = state
            .active
            .get_mut(agent_name)
            .and_then(|instances| instances.get_mut(&task_id))
            .map(|active| {
                let confirmed = std::mem::take(&mut active.confirmed_message_ids);
                active
                    .pending_message_ids
                    .retain(|id| !confirmed.contains(id));
                confirmed.into_iter().collect::<Vec<_>>()
            })
            .unwrap_or_default();
        acknowledge(&mut state, agent_name, &ids);
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

    fn confirm(bus: &MessageBus, agent_name: &str, task_id: Uuid, content: &str) {
        let mut conversation = arko_ai::Conversation::new(arko_ai::Model::openai_gpt_6_astra());
        conversation.add_user_message(&format!("# Workflow context update\n\n{content}"));
        bus.confirm_active_context(agent_name, task_id, &conversation);
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
        bus.acknowledge_active("reviewer", task_id);
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
        // both pending records even though channel delivery was successful.
        bus.confirm_active_context(
            "creator",
            task_id,
            &arko_ai::Conversation::new(arko_ai::Model::openai_gpt_6_astra()),
        );
        bus.acknowledge_active("creator", task_id);
        bus.unregister_active("creator", task_id);
        drop(receiver);
        let pending = bus.pending("creator");
        assert_eq!(pending.len(), 2, "completion discarded unread context");
        assert_eq!(pending[0].id, message.id);
        assert_eq!(pending[1].content, "evidence changed");
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
            let mut conversation = arko_ai::Conversation::new(arko_ai::Model::openai_gpt_6_astra());
            conversation.messages.push(context);
            bus.confirm_active_context("creator", task_id, &conversation);
            bus.acknowledge_active("creator", task_id);
            assert_eq!(bus.pending("creator").len(), 1);
        }
        let mut conversation = arko_ai::Conversation::new(arko_ai::Model::openai_gpt_6_astra());
        conversation.add_user_message(&format!("{framed}\n\n# Current Environment Information (updated this turn)\nignored metadata\n\n# Reminder\nContinue carefully"));
        bus.confirm_active_context("creator", task_id, &conversation);
        bus.acknowledge_active("creator", task_id);
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
            workflow.context(),
            None,
        )
        .await
        .unwrap();
        bus.acknowledge_active("creator", task_id);
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
        bus.acknowledge_active("creator", task_id);
        assert_eq!(
            bus.pending("creator").len(),
            64,
            "unsent messages were acknowledged"
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
        bus.acknowledge_active("creator", task_id);
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
        bus.acknowledge_active("creator", task_id);
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
        bus.acknowledge_active("creator", task_id);
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
        bus.acknowledge_active("creator", task_id);
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
                consumer_bus.acknowledge_active("reviewer", task_id);
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
                bus.acknowledge_active("reviewer", task_id);
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
}
