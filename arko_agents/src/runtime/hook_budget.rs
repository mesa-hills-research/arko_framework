//! The budget of a Claude Code hook's text, and how the context that waits
//! for the model ([`Task::pending_context_injections`](crate::Task::pending_context_injections),
//! the task's queue) goes out in hooks.
//!
//! Claude Code replaces a hook's text of over 10,000 characters with a
//! preview of 2 KB and the name of a file. So no text the runtime gives the
//! model through a hook (a PostToolUse hook's context, a PreToolUse denial,
//! an answer to a permission request) is over [`HOOK_MAX_BYTES`], whatever
//! the producers give. Bytes are never fewer than UTF-16 units or
//! characters, so the budget holds whichever unit Claude Code counts.
//!
//! The queue is the only place context waits, and it is saved with the
//! task. A hook carries the queue's items in order, by three rules.
//! 1. Whole items go while they fit.
//! 2. An item that would fit an empty hook but not the room left waits
//!    whole for the next hook.
//! 3. An item longer than an empty hook is split when it is first in line.
//!    The part that fits goes now, cut at a line break in its last fifth
//!    when there is one, and ends with [`GOES_ON`]. The rest starts with
//!    [`CONTINUED`].
//!
//! A hook leaves the queue as it was. Its items leave it, and its text joins
//! the conversation, in one step, once Claude Code reports its call's
//! results, in the order the hooks were given. A stop or a crash in between
//! leaves the queue whole, so the next input carries what wasn't
//! acknowledged. The worst outcome is a repeat, when Claude Code kept a
//! hook's text and the runtime never read the results.
//!
//! The user's typed notes ([`TurnContext::user_content`](super::TurnContext::user_content))
//! go as their own message, once everything queued ahead of them has gone
//! in a hook. Until then they wait here with their marks, and they join the
//! queue at their place when they can't go that way.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::task::CONTEXT_UPDATE;

/// The most bytes of text a hook gives the model.
pub(super) const HOOK_MAX_BYTES: usize = 9_500;

/// Ends the part of an item that goes on in the next hook.
pub(super) const GOES_ON: &str = "\n(This update goes on in the next one.)";

/// Starts the rest of an item whose first part went in a hook.
pub(super) const CONTINUED: &str = "(Continued from the previous update.)\n";

type Marks = BTreeMap<String, u64>;

/// Where a split item's part ends, in bytes, and where its rest starts.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Part {
    end: usize,
    rest: usize,
}

/// A hook given out whose call's results Claude Code hasn't reported.
#[derive(Debug)]
struct Hook {
    /// The call (its `tool_use_id`) whose results acknowledge it.
    call: String,
    /// The queue items it carried whole, after those of the hooks before it.
    whole: usize,
    /// The part it carried of the item after those, when it split one.
    part: Option<Part>,
    /// Its text, heading included. Empty when it carried a typed note only.
    text: String,
    /// The typed note sent with it, by the UUID of its message.
    note: Option<String>,
    /// Claude Code reported the call's results, and the hook waits for an
    /// earlier one.
    reported: bool,
}

/// Typed notes from one update.
#[derive(Debug, Clone)]
struct Note {
    text: String,
    marks: Marks,
    /// The update's identifiers, when it had no other text.
    ids: Vec<String>,
    /// Where it belongs in the queue, after this many of its items.
    place: usize,
    /// Its hook's text joined the conversation with it.
    recorded: bool,
    /// The CLI said it completed it.
    completed: bool,
}

/// What joins the conversation once a hook is acknowledged.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Delivered {
    /// The record's text: the hook's, then the typed note sent with it.
    pub text: String,
    /// The host's identifiers of the items whose first part it carried.
    pub ids: Vec<String>,
    /// The marks the task keeps from now on (a typed note's).
    pub marks: Marks,
    /// Its size in bytes, without its heading.
    pub bytes: usize,
}

/// The state of the queue's delivery the runtime keeps beside the task: the
/// hooks in flight, a typed note waiting, the notes sent and the rests of
/// denials too long for their hook.
#[derive(Debug, Default)]
pub(super) struct Delivery {
    in_flight: VecDeque<Hook>,
    /// The typed notes of the latest update, until they can go as a message.
    waiting: Option<Note>,
    /// The note just chosen to go, until its message is sent.
    sending: Option<Note>,
    /// Notes sent as messages, by UUID, until the CLI says what became of them.
    sent: HashMap<String, Note>,
    /// The rests of denials too long for their hook, by call, until the
    /// call's result is reported.
    denied: Vec<(String, String)>,
    /// The identifiers of the source's updates that wait in the queue, by
    /// the update's text (empty for an update of images alone).
    update_ids: Vec<(String, Vec<String>)>,
}

/// Where `text`, longer than `room` bytes, is cut so its part fits `room`.
fn cut(text: &str, room: usize) -> Part {
    let mut end = room.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let from = room - room / 5;
    match text[..end].rfind('\n') {
        Some(newline) if newline >= from => Part {
            end: newline,
            rest: newline,
        },
        _ => Part { end, rest: end },
    }
}

/// `text` as its part and its rest, each with its marker.
fn split(text: &str, part: Part) -> (String, String) {
    (
        format!("{}{GOES_ON}", &text[..part.end]),
        format!("{CONTINUED}{}", &text[part.rest..]),
    )
}

fn heading() -> String {
    format!("{CONTEXT_UPDATE}\n\n")
}

impl Delivery {
    /// The queue's items in hooks, and whether a hook may carry more: not
    /// while a split item's part is in flight, whose rest comes first.
    fn covered(&self) -> (usize, bool) {
        let whole = self.in_flight.iter().map(|hook| hook.whole).sum::<usize>();
        if self.in_flight.iter().any(|hook| hook.part.is_some()) {
            (whole + 1, false)
        } else {
            (whole, true)
        }
    }

    /// Nothing in the queue waits outside a hook, and no typed note waits:
    /// the turn context source may be asked for its update.
    pub(super) fn caught_up(&self, queue: &[String]) -> bool {
        let (covered, open) = self.covered();
        open && covered >= queue.len() && self.waiting.is_none()
    }

    /// Puts `text` in the queue at `index`, behind the hooks in flight. The
    /// notes' places after it move with it.
    fn insert(&mut self, queue: &mut Vec<String>, index: usize, text: String) {
        let index = index.max(self.covered().0).min(queue.len());
        queue.insert(index, text);
        let notes = self.waiting.iter_mut().chain(self.sent.values_mut());
        for note in notes {
            if note.place >= index {
                note.place += 1;
            }
        }
    }

    /// A typed note joins the queue at its place. Returns its marks, which
    /// the task keeps from now on.
    fn queue_note(&mut self, queue: &mut Vec<String>, note: Note) -> Marks {
        self.update_ids.push((note.text.clone(), note.ids));
        self.insert(queue, note.place, note.text);
        note.marks
    }

    /// The turn context source's update joins the queue. During a turn
    /// (`in_turn`), its typed notes wait to go as a message once what is
    /// ahead of them has gone. Returns the marks the task keeps now, which
    /// for an update with typed notes come with the notes.
    pub(super) fn take_update(
        &mut self,
        queue: &mut Vec<String>,
        content: String,
        typed: String,
        ids: Vec<String>,
        marks: Marks,
        in_turn: bool,
    ) -> Marks {
        let content = Some(content).filter(|content| !content.trim().is_empty());
        let typed = Some(typed).filter(|typed| !typed.trim().is_empty());
        let (content_ids, note_ids) = match (&content, &typed) {
            (Some(_), _) => (ids, Vec::new()),
            (None, Some(_)) => (Vec::new(), ids),
            (None, None) => {
                // Images alone, which wait for the next input with the ids.
                if !ids.is_empty() {
                    self.update_ids.push((String::new(), ids));
                }
                return marks;
            }
        };
        if let Some(content) = content {
            self.update_ids.push((content.clone(), content_ids));
            queue.push(content);
        }
        let Some(text) = typed else {
            return marks;
        };
        let note = Note {
            text,
            marks,
            ids: note_ids,
            place: queue.len(),
            recorded: false,
            completed: false,
        };
        if in_turn {
            self.waiting = Some(note);
            return Marks::new();
        }
        self.queue_note(queue, note)
    }

    /// The text of a denial (a PreToolUse hook's or a permission answer's)
    /// for call `call`, within the budget. The rest of a longer one joins
    /// the queue when the call's result is reported, or at once without a
    /// call.
    pub(super) fn deny(
        &mut self,
        queue: &mut Vec<String>,
        call: Option<&str>,
        text: String,
    ) -> String {
        if text.len() <= HOOK_MAX_BYTES {
            return text;
        }
        let (part, rest) = split(&text, cut(&text, HOOK_MAX_BYTES - GOES_ON.len()));
        match call {
            Some(call) => self.denied.push((call.to_string(), rest)),
            None => {
                let at = self.covered().0;
                self.insert(queue, at, rest);
            }
        }
        part
    }

    /// The context of call `call`'s PostToolUse hook: the queue's items from
    /// the first one not in flight, up to the typed note waiting, by the
    /// rules in the [module](self). `None` when it carries none.
    pub(super) fn fill(&mut self, queue: &[String], call: &str) -> Option<String> {
        let (start, open) = self.covered();
        if !open || start >= queue.len() {
            return None;
        }
        let end = self
            .waiting
            .as_ref()
            .map_or(queue.len(), |note| note.place.min(queue.len()));
        let mut text = heading();
        let empty = text.len();
        let (mut whole, mut part) = (0, None);
        for item in queue.get(start..end).unwrap_or_default() {
            let separator = if whole == 0 { "" } else { "\n\n" };
            if text.len() + separator.len() + item.len() <= HOOK_MAX_BYTES {
                text.push_str(separator);
                text.push_str(item);
                whole += 1;
                continue;
            }
            // It waits whole for a hook it fits, or it is split once it is
            // first in line.
            if whole > 0 || empty + item.len() <= HOOK_MAX_BYTES {
                break;
            }
            let at = cut(item, HOOK_MAX_BYTES - empty - GOES_ON.len());
            text.push_str(&split(item, at).0);
            part = Some(at);
            break;
        }
        if whole == 0 && part.is_none() {
            return None;
        }
        debug_assert!(text.len() <= HOOK_MAX_BYTES, "{}", text.len());
        self.in_flight.push_back(Hook {
            call: call.to_string(),
            whole,
            part,
            text: text.clone(),
            note: None,
            reported: false,
        });
        Some(text)
    }

    /// The typed note that goes now as the user's message, once a hook
    /// carried everything ahead of it. It is sent with
    /// [`note_sent`](Self::note_sent) or put back with
    /// [`note_unsent`](Self::note_unsent).
    pub(super) fn due_note(&mut self, queue: &[String]) -> Option<String> {
        let (covered, open) = self.covered();
        let note = self.waiting.as_ref()?;
        if !open || covered < note.place.min(queue.len()) {
            return None;
        }
        let note = self.waiting.take()?;
        let text = note.text.clone();
        self.sending = Some(note);
        Some(text)
    }

    /// The due note went as message `uuid`, with call `call`'s hook.
    pub(super) fn note_sent(&mut self, call: &str, uuid: String) {
        let Some(mut note) = self.sending.take() else {
            return;
        };
        // Behind what was ahead of it, all in hooks now.
        note.place = self.covered().0;
        match self
            .in_flight
            .iter_mut()
            .rev()
            .find(|hook| hook.call == call)
        {
            Some(hook) => hook.note = Some(uuid.clone()),
            None => self.in_flight.push_back(Hook {
                call: call.to_string(),
                whole: 0,
                part: None,
                text: String::new(),
                note: Some(uuid.clone()),
                reported: false,
            }),
        }
        self.sent.insert(uuid, note);
    }

    /// The due note couldn't be sent: it joins the queue at its place.
    /// Returns its marks.
    pub(super) fn note_unsent(&mut self, queue: &mut Vec<String>) -> Marks {
        match self.sending.take() {
            Some(note) => self.queue_note(queue, note),
            None => Marks::new(),
        }
    }

    /// The CLI said what became of note message `uuid`. One it cancelled
    /// never reached the model, so it goes back to its place in the queue,
    /// left out of its hook's record when that is still to come. One it
    /// completed is done with once its hook's record holds it (see
    /// [`session_ended`](Self::session_ended)). Returns the marks the task
    /// keeps now, when the note went back.
    pub(super) fn note_settled(
        &mut self,
        queue: &mut Vec<String>,
        uuid: &str,
        cancelled: bool,
    ) -> Option<Marks> {
        if !cancelled {
            let note = self.sent.get_mut(uuid)?;
            note.completed = true;
            if note.recorded {
                self.sent.remove(uuid);
            }
            return None;
        }
        let note = self.sent.remove(uuid)?;
        for hook in &mut self.in_flight {
            if hook.note.as_deref() == Some(uuid) {
                hook.note = None;
            }
        }
        Some(self.queue_note(queue, note))
    }

    /// Whether a note sent waits for the CLI's word on it.
    pub(super) fn notes_unsettled(&self) -> bool {
        self.sent.values().any(|note| !note.completed)
    }

    /// The identifiers of `item` as it first goes: the source's, else the
    /// host's (`identify`). A rest has none.
    fn ids_of(&mut self, item: &str, identify: &dyn Fn(&str) -> Option<String>) -> Vec<String> {
        if item.starts_with(CONTINUED) {
            return Vec::new();
        }
        match self.update_ids.iter().position(|(text, _)| text == item) {
            Some(index) => self.update_ids.remove(index).1,
            None => identify(item).into_iter().collect(),
        }
    }

    /// Claude Code reported call `call`'s result. A denial's rest joins the
    /// queue ahead of what waits, and the call's hook is acknowledged. The
    /// hooks acknowledged, from the first given, leave the queue in order,
    /// and are returned for the conversation. A hook reported before an
    /// earlier one waits for it.
    pub(super) fn reported(
        &mut self,
        queue: &mut Vec<String>,
        call: &str,
        identify: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<Delivered> {
        if let Some(index) = self.denied.iter().position(|(denied, _)| denied == call) {
            let (_, rest) = self.denied.remove(index);
            let at = self.covered().0;
            self.insert(queue, at, rest);
        }
        for hook in &mut self.in_flight {
            if hook.call == call {
                hook.reported = true;
            }
        }
        let mut delivered = Vec::new();
        while self.in_flight.front().is_some_and(|hook| hook.reported) {
            let hook = self.in_flight.pop_front().expect("checked");
            let firsts = hook.whole + usize::from(hook.part.is_some());
            let mut ids = Vec::new();
            for item in queue.iter().take(firsts) {
                ids.extend(self.ids_of(item, identify));
            }
            let removed = hook.whole.min(queue.len());
            queue.drain(..removed);
            if let (Some(part), Some(item)) = (hook.part, queue.first_mut()) {
                *item = split(item, part).1;
            }
            for note in self.waiting.iter_mut().chain(self.sent.values_mut()) {
                note.place = note.place.saturating_sub(removed);
            }
            let mut text = hook.text;
            let mut bytes = text.len().saturating_sub(heading().len());
            let mut marks = Marks::new();
            let note = hook.note.as_ref().and_then(|uuid| self.sent.get_mut(uuid));
            let mut done = None;
            if let Some(note) = note {
                note.recorded = true;
                bytes += note.text.len();
                if text.is_empty() {
                    // A note alone has the update's heading in the record.
                    text = format!("{}{}", heading(), note.text);
                } else {
                    text = format!("{text}\n\n{}", note.text);
                }
                ids.extend(note.ids.iter().cloned());
                marks = note.marks.clone();
                done = hook.note.as_ref().filter(|_| note.completed);
            }
            if let Some(uuid) = done {
                self.sent.remove(uuid);
            }
            if text.is_empty() {
                continue;
            }
            delivered.push(Delivered {
                text,
                ids,
                marks,
                bytes,
            });
        }
        delivered
    }

    /// The session's hooks end (a stop, a turn's end, or the next input):
    /// those not acknowledged are dropped, their items still in the queue,
    /// and the typed note waiting and the denials' rests join it at their
    /// places. Returns the marks the task keeps now.
    pub(super) fn settle(&mut self, queue: &mut Vec<String>) -> Marks {
        self.in_flight.clear();
        for (_, rest) in std::mem::take(&mut self.denied).into_iter().rev() {
            self.insert(queue, 0, rest);
        }
        let mut marks = Marks::new();
        for note in [self.sending.take(), self.waiting.take()]
            .into_iter()
            .flatten()
        {
            marks.extend(self.queue_note(queue, note));
        }
        marks
    }

    /// The session ended: a note sent goes back to its place unless its
    /// hook's record holds it, also one the CLI said it completed, since the
    /// results it went with never came. The worst outcome is a repeat.
    /// Returns the marks the task keeps now.
    pub(super) fn session_ended(&mut self, queue: &mut Vec<String>) -> Marks {
        let mut marks = self.settle(queue);
        let mut notes = std::mem::take(&mut self.sent)
            .into_values()
            .filter(|note| !note.recorded)
            .collect::<Vec<_>>();
        notes.sort_by_key(|note| note.place);
        for note in notes.into_iter().rev() {
            marks.extend(self.queue_note(queue, note));
        }
        marks
    }

    /// The identifiers of everything in the queue, as the next input
    /// carries it all: each item's, then those of updates of images alone.
    pub(super) fn all_ids(
        &mut self,
        queue: &[String],
        identify: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<String> {
        let mut ids = Vec::new();
        for item in queue {
            ids.extend(self.ids_of(item, identify));
        }
        for (_, rest) in std::mem::take(&mut self.update_ids) {
            ids.extend(rest);
        }
        ids
    }
}

#[cfg(test)]
#[path = "hook_budget_tests.rs"]
mod tests;
