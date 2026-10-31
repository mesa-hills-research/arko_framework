//! The context waiting for the model ([`Task::pending_context_injections`])
//! and the task's record of what its conversation holds of it
//! ([`Task::context_records`]), so a host learns what a task holds from
//! ids rather than from its text.

use serde::{Deserialize, Deserializer, Serialize};

use super::Task;

/// One item of context waiting for the model, in
/// [`Task::pending_context_injections`]. It stays there, whole, from the
/// step the runtime takes it until the step the conversation records its
/// last part.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PendingContext {
    /// The task's number for it ([`Task::next_context_item`]), never used
    /// again. 0 for an item saved as bare text, which a run numbers as it
    /// starts.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub id: u64,
    #[serde(default, skip_serializing_if = "PendingKind::is_context")]
    pub kind: PendingKind,
    /// The whole item, also once its first parts are in the conversation.
    pub text: String,
    /// The host's ids of what it carries
    /// ([`TurnContext::ids`](crate::runtime::TurnContext::ids),
    /// [`TurnContextSource::identify`](crate::runtime::TurnContextSource::identify)).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub host_ids: Vec<String>,
    /// Saved as bare text, before items carried their host's ids: only its
    /// text says what it holds (see [`Task::unidentified_context`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unidentified: bool,
    /// The bytes of `text` the conversation holds already, where a Claude
    /// Code agent got its first parts in hooks.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub given: usize,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

/// What a [`PendingContext`] is.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingKind {
    /// Context a host sent, its turn context source's update, or the
    /// runtime's own.
    #[default]
    Context,
    /// What the user typed ([`TurnContext::user_content`](crate::runtime::TurnContext::user_content)),
    /// which a Claude Code agent gets as their message during a turn.
    Typed,
    /// The rest of the runtime's answer to call `call` (a denial, or a
    /// host's output for a built-in) too long for the answer itself.
    DenialRest { call: String },
}

impl PendingKind {
    fn is_context(&self) -> bool {
        *self == Self::Context
    }
}

impl PendingContext {
    /// Context that `text` holds, which the host identifies by `host_ids`.
    /// [`Task::queue_context`] numbers it.
    pub fn new(kind: PendingKind, text: impl Into<String>, host_ids: Vec<String>) -> Self {
        Self {
            id: 0,
            kind,
            text: text.into(),
            host_ids,
            unidentified: false,
            given: 0,
        }
    }
}

/// Bare text, as tasks saved it before items had ids: what it holds is
/// known from its text only.
impl From<String> for PendingContext {
    fn from(text: String) -> Self {
        Self {
            unidentified: true,
            ..Self::new(PendingKind::Context, text, Vec::new())
        }
    }
}

impl From<&str> for PendingContext {
    fn from(text: &str) -> Self {
        text.to_string().into()
    }
}

/// An item equals a text when its text does.
impl PartialEq<str> for PendingContext {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for PendingContext {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

impl PartialEq<String> for PendingContext {
    fn eq(&self, other: &String) -> bool {
        self.text == *other
    }
}

impl<'de> Deserialize<'de> for PendingContext {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            #[serde(default)]
            id: u64,
            #[serde(default)]
            kind: PendingKind,
            text: String,
            #[serde(default)]
            host_ids: Vec<String>,
            #[serde(default)]
            unidentified: bool,
            #[serde(default)]
            given: usize,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Saved {
            Text(String),
            Item(Fields),
        }
        Ok(match Saved::deserialize(deserializer)? {
            Saved::Text(text) => text.into(),
            Saved::Item(fields) => Self {
                id: fields.id,
                kind: fields.kind,
                given: fields.given.min(fields.text.len()),
                text: fields.text,
                host_ids: fields.host_ids,
                unidentified: fields.unidentified,
            },
        })
    }
}

/// What one message of the conversation holds of the task's context
/// ([`Task::context_records`]).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextRecord {
    /// The message's index in the conversation.
    pub message: usize,
    pub parts: Vec<RecordedPart>,
}

/// An item, or a part of one, that a message holds.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedPart {
    /// The item's number, 0 for what a host's prompt presents.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub item: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub host_ids: Vec<String>,
    /// Of an item saved as bare text (see [`PendingContext::unidentified`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unidentified: bool,
    /// It holds the item's end: the whole item, or its last part.
    pub last: bool,
}

impl RecordedPart {
    /// The part of `item` a message holds, its last (or the whole item)
    /// when `last`.
    pub fn of(item: &PendingContext, last: bool) -> Self {
        Self {
            item: item.id,
            host_ids: item.host_ids.clone(),
            unidentified: item.unidentified,
            last,
        }
    }

    /// What a host's prompt presents, by the host's ids.
    pub fn presented(host_ids: Vec<String>) -> Self {
        Self {
            item: 0,
            host_ids,
            unidentified: false,
            last: true,
        }
    }
}

/// What a task holds of the item a host's id names (see [`Task::holding`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Holding {
    /// Its conversation holds all of it.
    Recorded,
    /// It waits in the task's pending context, perhaps with its first parts
    /// in the conversation.
    Pending,
    /// Neither. A first part whose rest a cut back removed counts so too.
    NotHeld,
    /// Only text could say: the task has context without ids (see
    /// [`Task::unidentified_context`]).
    Unknown,
}

/// What only text can say of what a task holds: context saved without ids,
/// and the messages from before the task kept records.
#[derive(Debug, Default)]
pub struct UnidentifiedContext<'a> {
    /// The pending items saved as bare text.
    pub pending: Vec<&'a str>,
    /// The messages whose records hold such items, and the messages
    /// recorded before [`Task::context_records_from`].
    pub messages: Vec<&'a arko_ai::Message>,
}

impl UnidentifiedContext<'_> {
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.messages.is_empty()
    }
}

impl Task {
    /// Puts `item` at the end of the pending context, numbered, and returns
    /// its number. Context that is only white space isn't kept (`None`).
    pub fn queue_context(&mut self, item: PendingContext) -> Option<u64> {
        let at = self.pending_context_injections.len();
        self.insert_context(at, item)
    }

    /// [`queue_context`](Self::queue_context) at `index`.
    pub(crate) fn insert_context(&mut self, index: usize, mut item: PendingContext) -> Option<u64> {
        if item.text.trim().is_empty() {
            return None;
        }
        self.next_context_item += 1;
        item.id = self.next_context_item;
        let index = index.min(self.pending_context_injections.len());
        self.pending_context_injections.insert(index, item);
        Some(self.next_context_item)
    }

    /// As a run starts: the items saved as bare text get numbers, and a task
    /// saved before it kept records starts them from its conversation's end.
    pub(crate) fn number_context(&mut self) {
        for index in 0..self.pending_context_injections.len() {
            if self.pending_context_injections[index].id == 0 {
                self.next_context_item += 1;
                self.pending_context_injections[index].id = self.next_context_item;
            }
        }
        if self.context_records_from.is_none() {
            self.context_records_from = Some(self.conversation.messages.len());
        }
    }

    /// Records that message `message` holds `parts`.
    pub(crate) fn record_context(&mut self, message: usize, parts: Vec<RecordedPart>) {
        if parts.is_empty() {
            return;
        }
        match self
            .context_records
            .iter_mut()
            .find(|record| record.message == message)
        {
            Some(record) => record.parts.extend(parts),
            None => self.context_records.push(ContextRecord { message, parts }),
        }
    }

    /// Records that message `message` of the conversation, one the host
    /// added (a continuation prompt), presents what `host_ids` name, so
    /// [`holding`](Self::holding) finds it there.
    pub fn record_held_ids(&mut self, message: usize, host_ids: Vec<String>) {
        if !host_ids.is_empty() && message < self.conversation.messages.len() {
            self.record_context(message, vec![RecordedPart::presented(host_ids)]);
        }
    }

    /// Records [`initial_prompt_ids`](Self::initial_prompt_ids) with the
    /// first message, which the runtime has just written from the initial
    /// prompt.
    pub(crate) fn record_initial_prompt(&mut self) {
        let ids = self.initial_prompt_ids.clone();
        self.record_held_ids(0, ids);
    }

    /// The pending texts, in order.
    pub fn pending_texts(&self) -> Vec<&str> {
        self.pending_context_injections
            .iter()
            .map(|item| item.text.as_str())
            .collect()
    }

    /// What the task holds of the item that `host_id` names, by the ids its
    /// pending context and its records carry.
    pub fn holding(&self, host_id: &str) -> Holding {
        let names = |ids: &[String]| ids.iter().any(|id| id == host_id);
        if self
            .pending_context_injections
            .iter()
            .any(|item| names(&item.host_ids))
        {
            return Holding::Pending;
        }
        let parts = self
            .context_records
            .iter()
            .flat_map(|record| &record.parts)
            .filter(|part| names(&part.host_ids))
            .collect::<Vec<_>>();
        if parts.iter().any(|part| part.last) {
            return Holding::Recorded;
        }
        if parts.is_empty() && !self.unidentified_context().is_empty() {
            return Holding::Unknown;
        }
        Holding::NotHeld
    }

    /// What only text can say of what the task holds, for a host to search
    /// as it did before tasks kept ids: the pending items saved as bare
    /// text, the messages that hold such items, and the messages from before
    /// [`context_records_from`](Self::context_records_from) (all of them for
    /// a task saved before records existed).
    pub fn unidentified_context(&self) -> UnidentifiedContext<'_> {
        let messages = &self.conversation.messages;
        let from = self
            .context_records_from
            .unwrap_or(messages.len())
            .min(messages.len());
        let mut indices: Vec<usize> = (0..from).collect();
        for record in &self.context_records {
            if record.parts.iter().any(|part| part.unidentified)
                && record.message >= from
                && record.message < messages.len()
            {
                indices.push(record.message);
            }
        }
        indices.sort_unstable();
        indices.dedup();
        UnidentifiedContext {
            pending: self
                .pending_context_injections
                .iter()
                .filter(|item| item.unidentified)
                .map(|item| item.text.as_str())
                .collect(),
            messages: indices.into_iter().map(|index| &messages[index]).collect(),
        }
    }

    /// The conversation keeps its first `at` messages (a cut back): the
    /// records of the others go, and those from before records began stay
    /// within the conversation.
    pub(crate) fn cut_context_records(&mut self, at: usize) {
        self.context_records.retain(|record| record.message < at);
        if let Some(from) = &mut self.context_records_from {
            *from = (*from).min(at);
        }
    }

    /// The conversation's messages moved: `moved(index)` is where message
    /// `index` is now, or `None` for one that is gone. The records follow
    /// their messages, and those of messages that are gone go with them.
    pub(crate) fn move_context_records(&mut self, moved: impl Fn(usize) -> Option<usize>) {
        self.context_records
            .retain_mut(|record| match moved(record.message) {
                Some(index) => {
                    record.message = index;
                    true
                }
                None => false,
            });
        if let Some(from) = self.context_records_from {
            // The messages before `from` that are still there.
            let kept = (0..from).filter_map(&moved).count();
            self.context_records_from = Some(kept);
        }
    }

    /// A fresh conversation replaced the conversation: it holds nothing of
    /// the context yet, so every pending item goes whole again.
    pub(crate) fn reset_context_records(&mut self) {
        self.context_records.clear();
        self.context_records_from = Some(0);
        for item in &mut self.pending_context_injections {
            item.given = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentConfig, AgentLLMConfig};

    fn task(messages: usize) -> Task {
        let config = AgentConfig::builder(
            "pending".into(),
            "Pending".into(),
            AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into())
        .build()
        .unwrap();
        let mut task = Task::new(
            config,
            "prompt".into(),
            std::path::PathBuf::from("/w"),
            None,
        );
        for index in 0..messages {
            task.conversation
                .add_user_message(&format!("message {index}"));
        }
        task
    }

    fn part(item: u64, id: &str, last: bool) -> RecordedPart {
        RecordedPart {
            item,
            host_ids: vec![id.to_string()],
            unidentified: false,
            last,
        }
    }

    #[test]
    fn holding_reads_the_ids_the_task_keeps() {
        let mut task = task(3);
        task.record_context(1, vec![part(1, "mail:1", true), part(2, "mail:2", false)]);
        let mut item = PendingContext::new(PendingKind::Context, "the rest", vec!["mail:2".into()]);
        item.given = 4;
        task.queue_context(item);
        task.record_held_ids(0, vec!["message:9".into()]);
        assert_eq!(task.holding("mail:1"), Holding::Recorded);
        assert_eq!(task.holding("mail:2"), Holding::Pending);
        assert_eq!(task.holding("message:9"), Holding::Recorded);
        assert_eq!(task.holding("mail:3"), Holding::NotHeld);
        // A first part whose rest is gone holds nothing.
        task.pending_context_injections.clear();
        assert_eq!(task.holding("mail:2"), Holding::NotHeld);
        // What only text can say.
        task.queue_context("A bare text.".into());
        assert_eq!(task.holding("mail:3"), Holding::Unknown);
        assert_eq!(task.unidentified_context().pending, ["A bare text."]);
    }

    #[test]
    fn a_bare_text_stays_unidentified_once_numbered_and_recorded() {
        let mut task = task(0);
        task.context_records_from = None;
        task.pending_context_injections = vec!["A bare text.".into()];
        task.number_context();
        let item = task.pending_context_injections.remove(0);
        assert!(item.id > 0 && item.unidentified);
        task.conversation
            .add_user_message("# Workflow context update\n\nA bare text.");
        task.record_context(0, vec![RecordedPart::of(&item, true)]);
        assert_eq!(task.holding("mail:1"), Holding::Unknown);
        let unknown = task.unidentified_context();
        assert!(unknown.pending.is_empty());
        assert_eq!(unknown.messages.len(), 1);
    }

    #[test]
    fn records_follow_their_messages_through_edits() {
        let mut task = task(6);
        task.context_records_from = Some(2);
        for message in [2, 3, 5] {
            task.record_context(
                message,
                vec![part(message as u64, &format!("mail:{message}"), true)],
            );
        }
        let mut item = PendingContext::new(PendingKind::Context, "long item", Vec::new());
        item.given = 3;
        task.queue_context(item);
        // Messages 1 and 4 go, the others move down.
        task.move_context_records(|index| match index {
            0 => Some(0),
            1 | 4 => None,
            2 | 3 => Some(index - 1),
            index => Some(index - 2),
        });
        let at = |id: &str| {
            task.context_records
                .iter()
                .find(|record| record.parts[0].host_ids == [id])
                .map(|record| record.message)
        };
        assert_eq!(
            (at("mail:2"), at("mail:3"), at("mail:5")),
            (Some(1), Some(2), Some(3))
        );
        assert_eq!(
            task.context_records_from,
            Some(1),
            "message 0 is the one left before it"
        );
        // A cut back keeps the records before its point, and records start
        // there at the latest.
        task.cut_context_records(2);
        assert_eq!(task.context_records.len(), 1);
        assert_eq!(task.context_records_from, Some(1));
        task.cut_context_records(0);
        assert_eq!(task.context_records_from, Some(0));
        // A fresh conversation holds none of the context: every item goes
        // whole again.
        task.record_context(0, vec![part(9, "mail:9", true)]);
        assert_eq!(task.start_fresh_conversation(), 6);
        assert!(task.context_records.is_empty());
        assert_eq!(task.context_records_from, Some(0));
        assert_eq!(task.pending_context_injections[0].given, 0);
    }

    #[test]
    fn a_cut_back_drops_the_records_past_its_point() {
        let mut task = task(0);
        task.conversation.add_user_message("first");
        task.conversation.add_assistant_message("reply");
        task.conversation
            .add_user_message("# Workflow context update\n\nmail 1");
        task.record_context(2, vec![part(1, "mail:1", true)]);
        task.context_records_from = Some(3);
        task.cut_back(2).unwrap();
        assert!(task.context_records.is_empty());
        assert_eq!(task.context_records_from, Some(2));
        assert_eq!(
            task.holding("mail:1"),
            Holding::Unknown,
            "messages before the records"
        );
    }
}
