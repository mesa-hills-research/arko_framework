//! The budget of a Claude Code hook's text, and how the context that waits
//! for the model ([`Task::pending_context_injections`](crate::Task::pending_context_injections),
//! the task's list) goes out in hooks.
//!
//! Claude Code replaces a hook's text of over 10,000 characters with a
//! preview of 2 KB and the name of a file. So no text the runtime gives the
//! model through a hook (a PostToolUse hook's context, a PreToolUse denial,
//! an answer to a permission request) is over [`HOOK_MAX_BYTES`], whatever
//! the producers give. Bytes are never fewer than UTF-16 units or
//! characters, so the budget holds whichever unit Claude Code counts.
//!
//! The task's list is the only place context waits, and it is saved with
//! the task. A hook carries its items in order, by three rules.
//! 1. Whole items go while they fit.
//! 2. An item that would fit an empty hook but not the room left waits
//!    whole for the next hook.
//! 3. An item longer than an empty hook is split when it is first in line.
//!    The part that fits goes now, cut at a line break in its last fifth
//!    when there is one, and ends with [`GOES_ON`]. The next part starts
//!    with [`CONTINUED`]. An item saved as bare text
//!    ([`unidentified`](crate::PendingContext::unidentified)) is never
//!    split, and waits whole for the next input.
//!
//! A hook leaves the list as it was. Once Claude Code reports its call's
//! results, in the order the hooks were given, the hook's text joins the
//! conversation with a record of the items it held, and in the same step a
//! whole item (or its last part) leaves the list and a part moves the
//! item's [`given`](crate::PendingContext::given). A stop or a crash in
//! between leaves the list as it was, so the next input carries what wasn't
//! acknowledged. The worst outcome is a repeat.
//!
//! [`Delivery`] keeps only what means nothing after the run, by item
//! number: the hooks in flight, the typed items sent as the user's message,
//! and the denials' rests waiting for their call's result. Nothing in the
//! list ever moves.

use std::collections::{HashSet, VecDeque};

use crate::task::{CONTEXT_UPDATE, PendingContext, PendingKind, RecordedPart};

/// The most bytes of text a hook gives the model.
pub(super) const HOOK_MAX_BYTES: usize = 9_500;

/// Ends the part of an item that goes on in the next hook.
pub(super) const GOES_ON: &str = "\n(This update goes on in the next one.)";

/// Starts the next part of an item whose first part went in a hook.
pub(super) const CONTINUED: &str = "(Continued from the previous update.)\n";

/// Ends a denial cut to the budget, whose rest comes in a context update.
pub(super) const DENIAL_GOES_ON: &str = "\n(This answer goes on in the next context update.)";

/// Starts a denial's rest.
pub(super) const DENIAL_REST: &str = "(The rest of arko's answer to your call.)\n";

/// Where a split item's part ends, in bytes, and where its rest starts.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cut {
    end: usize,
}

/// A part of an item a hook carried, up to `end` (a byte offset of its
/// text).
#[derive(Debug, Clone, Copy)]
struct Carried {
    item: u64,
    end: usize,
}

/// A hook given out whose call's results Claude Code hasn't reported.
#[derive(Debug)]
struct Hook {
    /// The call (its `tool_use_id`) whose results acknowledge it.
    call: String,
    parts: Vec<Carried>,
    /// Its text, heading included. Empty when it carried a typed item only.
    text: String,
    /// The typed item sent as the user's message with it.
    note: Option<u64>,
    /// Claude Code reported the call's results, and the hook waits for an
    /// earlier one.
    reported: bool,
}

/// A typed item sent as the user's message.
#[derive(Debug)]
struct Sent {
    uuid: String,
    item: u64,
    /// Its hook was acknowledged.
    acked: bool,
    /// The CLI said it completed the message.
    completed: bool,
}

/// What joins the conversation once a hook is acknowledged, or a typed item
/// is done.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Delivered {
    /// The record's text: the hook's, then the typed item sent with it.
    pub text: String,
    /// The items, or parts of them, the text holds.
    pub parts: Vec<RecordedPart>,
    /// The host's ids of the items whose first part it holds.
    pub ids: Vec<String>,
    /// Its size in bytes, without its heading.
    pub bytes: usize,
}

/// What the CLI's word on a typed item's message did.
#[derive(Debug, PartialEq)]
pub(super) enum Verdict {
    /// Nothing the runtime knows of.
    None,
    /// The item is done, and joins the conversation.
    Done(Delivered),
    /// It waits for its hook's acknowledgement too.
    Completed,
    /// Cancelled, it waits again where it is.
    Waiting,
}

/// The state of the list's delivery the runtime keeps beside the task. See
/// the [module](self).
#[derive(Debug, Default)]
pub(super) struct Delivery {
    in_flight: VecDeque<Hook>,
    /// The typed item chosen to go, until its message is sent.
    sending: Option<u64>,
    sent: Vec<Sent>,
    /// Typed items whose message couldn't be sent, which go in hooks as
    /// context instead.
    as_context: HashSet<u64>,
    /// Denials' rests waiting for their call's result, by call.
    held: Vec<(String, u64)>,
}

/// Where `text`, longer than `room` bytes, is cut so its start fits `room`.
fn cut(text: &str, room: usize) -> Cut {
    let mut end = room.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let from = room - room / 5;
    match text[..end].rfind('\n') {
        Some(newline) if newline >= from => Cut { end: newline },
        _ => Cut { end },
    }
}

/// `text`, a denial longer than the budget, cut to fit, and its rest.
pub(super) fn cut_denial(text: String) -> (String, Option<String>) {
    if text.len() <= HOOK_MAX_BYTES {
        return (text, None);
    }
    let at = cut(&text, HOOK_MAX_BYTES - DENIAL_GOES_ON.len());
    (
        format!("{}{DENIAL_GOES_ON}", &text[..at.end]),
        Some(text[at.end..].to_string()),
    )
}

/// What the model reads of `item` from `from` on: a part after the first
/// starts with [`CONTINUED`], and a denial's rest with [`DENIAL_REST`].
pub(super) fn shown_from(item: &PendingContext, from: usize) -> String {
    let rest = &item.text[from.min(item.text.len())..];
    match (&item.kind, from) {
        (_, 1..) => format!("{CONTINUED}{rest}"),
        (PendingKind::DenialRest { .. }, 0) => format!("{DENIAL_REST}{rest}"),
        _ => rest.to_string(),
    }
}

fn heading() -> String {
    format!("{CONTEXT_UPDATE}\n\n")
}

impl Delivery {
    /// Where item `id`'s part in a hook ends, when a hook carries one.
    fn in_flight_end(&self, id: u64) -> Option<usize> {
        self.in_flight
            .iter()
            .flat_map(|hook| &hook.parts)
            .filter(|part| part.item == id)
            .map(|part| part.end)
            .max()
    }

    fn sent(&self, id: u64) -> bool {
        self.sent.iter().any(|sent| sent.item == id) || self.sending == Some(id)
    }

    fn held(&self, id: u64) -> bool {
        self.held.iter().any(|(_, held)| *held == id)
    }

    /// Whether `item` goes as a typed note, apart from the hook's text.
    fn typed(&self, item: &PendingContext) -> bool {
        item.kind == PendingKind::Typed && !self.as_context.contains(&item.id)
    }

    /// Whether `item` is all in a hook already: its last part, or as a typed
    /// item sent.
    fn gone(&self, item: &PendingContext) -> bool {
        self.in_flight_end(item.id) == Some(item.text.len())
            || (self.typed(item) && self.sent(item.id))
    }

    /// Whether `item`, saved as bare text and too long for a hook, waits for
    /// the next input with everything behind it.
    fn waits_for_input(&self, item: &PendingContext) -> bool {
        item.unidentified
            && self.in_flight_end(item.id).is_none()
            && heading().len() + shown_from(item, item.given).len() > HOOK_MAX_BYTES
    }

    /// Every item is in a hook already, or waits for the next input: the
    /// turn context source may be asked for its update.
    pub(super) fn caught_up(&self, list: &[PendingContext]) -> bool {
        list.iter()
            .find(|item| !self.gone(item))
            .is_none_or(|item| self.waits_for_input(item))
    }

    /// Where a denial's rest joins the list: behind the items in hooks, an
    /// item whose first part the model has read, and the rests already at
    /// the front, ahead of what waits.
    pub(super) fn rest_place(&self, list: &[PendingContext]) -> usize {
        list.iter()
            .position(|item| {
                self.in_flight_end(item.id).is_none()
                    && item.given == 0
                    && !self.sent(item.id)
                    && !matches!(item.kind, PendingKind::DenialRest { .. })
            })
            .unwrap_or(list.len())
    }

    /// Denial rest `item` waits for call `call`'s result.
    pub(super) fn hold_rest(&mut self, call: &str, item: u64) {
        self.held.push((call.to_string(), item));
    }

    /// The context of call `call`'s PostToolUse hook: the list's items from
    /// the first one not all in a hook, by the rules in the
    /// [module](self). A typed item, a part in flight and a rest still
    /// held stop it there. `None` when it carries nothing.
    pub(super) fn fill(&mut self, list: &[PendingContext], call: &str) -> Option<String> {
        let mut text = heading();
        let empty = text.len();
        let mut parts = Vec::new();
        for item in list {
            let end = self.in_flight_end(item.id);
            if end == Some(item.text.len()) {
                continue;
            }
            if end.is_some() || self.held(item.id) || self.typed(item) {
                break;
            }
            let shown = shown_from(item, item.given);
            let separator = if parts.is_empty() { "" } else { "\n\n" };
            if text.len() + separator.len() + shown.len() <= HOOK_MAX_BYTES {
                text.push_str(separator);
                text.push_str(&shown);
                parts.push(Carried {
                    item: item.id,
                    end: item.text.len(),
                });
                continue;
            }
            // It waits whole for a hook it fits, or it is split once it is
            // first in line. An item saved as bare text is never split,
            // since only its whole text tells a host what it holds. It waits
            // for the next input, which has no budget.
            if !parts.is_empty() || empty + shown.len() <= HOOK_MAX_BYTES || item.unidentified {
                break;
            }
            let marker = shown.len() - (item.text.len() - item.given);
            let at = cut(&shown, HOOK_MAX_BYTES - empty - GOES_ON.len());
            text.push_str(&shown[..at.end]);
            text.push_str(GOES_ON);
            parts.push(Carried {
                item: item.id,
                end: item.given + at.end.saturating_sub(marker),
            });
            break;
        }
        if parts.is_empty() {
            return None;
        }
        debug_assert!(text.len() <= HOOK_MAX_BYTES, "{}", text.len());
        self.in_flight.push_back(Hook {
            call: call.to_string(),
            parts,
            text: text.clone(),
            note: None,
            reported: false,
        });
        Some(text)
    }

    /// The typed item that goes now as the user's message, once the hooks
    /// carry everything ahead of it, with its text. It is sent with
    /// [`note_sent`](Self::note_sent) or goes as context with
    /// [`note_unsent`](Self::note_unsent).
    pub(super) fn due_note(&mut self, list: &[PendingContext]) -> Option<(u64, String)> {
        // Behind a typed item sent, nothing goes until it is done.
        let item = list
            .iter()
            .find(|item| self.in_flight_end(item.id) != Some(item.text.len()))?;
        if !self.typed(item) || self.sent(item.id) {
            return None;
        }
        self.sending = Some(item.id);
        Some((item.id, item.text.clone()))
    }

    /// The due typed item went as message `uuid`, with call `call`'s hook.
    pub(super) fn note_sent(&mut self, call: &str, uuid: String) {
        let Some(item) = self.sending.take() else {
            return;
        };
        match self
            .in_flight
            .iter_mut()
            .rev()
            .find(|hook| hook.call == call)
        {
            Some(hook) => hook.note = Some(item),
            None => self.in_flight.push_back(Hook {
                call: call.to_string(),
                parts: Vec::new(),
                text: String::new(),
                note: Some(item),
                reported: false,
            }),
        }
        self.sent.push(Sent {
            uuid,
            item,
            acked: false,
            completed: false,
        });
    }

    /// The due typed item couldn't be sent: from now on it goes in hooks
    /// as context, where it is.
    pub(super) fn note_unsent(&mut self) {
        if let Some(item) = self.sending.take() {
            self.as_context.insert(item);
        }
    }

    /// Whether a typed item sent waits for the CLI's word on it.
    pub(super) fn notes_unsettled(&self) -> bool {
        self.sent.iter().any(|sent| !sent.completed)
    }

    /// The typed item `id` is done: it leaves `list` for the record.
    fn note_done(&mut self, list: &mut Vec<PendingContext>, id: u64) -> Option<PendingContext> {
        self.sent.retain(|sent| sent.item != id);
        let index = list.iter().position(|item| item.id == id)?;
        Some(list.remove(index))
    }

    /// The CLI said what became of message `uuid`. A typed item is done
    /// once its hook is acknowledged and the CLI completed its message, in
    /// either order. One it cancelled never reached the model, and waits
    /// again where it is, left out of its hook's record.
    pub(super) fn note_settled(
        &mut self,
        list: &mut Vec<PendingContext>,
        uuid: &str,
        cancelled: bool,
    ) -> Verdict {
        let Some(sent) = self.sent.iter_mut().find(|sent| sent.uuid == uuid) else {
            return Verdict::None;
        };
        let item = sent.item;
        if cancelled {
            self.sent.retain(|sent| sent.uuid != uuid);
            for hook in &mut self.in_flight {
                if hook.note == Some(item) {
                    hook.note = None;
                }
            }
            return Verdict::Waiting;
        }
        sent.completed = true;
        if !sent.acked {
            return Verdict::Completed;
        }
        match self.note_done(list, item) {
            Some(note) => Verdict::Done(Delivered {
                text: format!("{}{}", heading(), note.text),
                bytes: note.text.len(),
                ids: note.host_ids.clone(),
                parts: vec![RecordedPart::of(&note, true)],
            }),
            None => Verdict::None,
        }
    }

    /// Claude Code reported call `call`'s result. A denial's rest held for
    /// it may go now, which the returned flag says, and the call's hook is
    /// acknowledged. The hooks acknowledged, from the first given, are
    /// returned for the conversation in order. A hook reported before an
    /// earlier one waits for it.
    pub(super) fn reported(
        &mut self,
        list: &mut Vec<PendingContext>,
        call: &str,
    ) -> (Vec<Delivered>, bool) {
        let held = self.held.len();
        self.held.retain(|(denied, _)| denied != call);
        let released = self.held.len() < held;
        for hook in &mut self.in_flight {
            if hook.call == call {
                hook.reported = true;
            }
        }
        let mut delivered = Vec::new();
        while self.in_flight.front().is_some_and(|hook| hook.reported) {
            let hook = self.in_flight.pop_front().expect("checked");
            let mut parts = Vec::new();
            let mut ids = Vec::new();
            for carried in &hook.parts {
                let Some(index) = list.iter().position(|item| item.id == carried.item) else {
                    continue;
                };
                let item = &mut list[index];
                if item.given == 0 {
                    ids.extend(item.host_ids.iter().cloned());
                }
                let last = carried.end >= item.text.len();
                parts.push(RecordedPart::of(item, last));
                if last {
                    list.remove(index);
                } else {
                    item.given = carried.end;
                }
            }
            let mut text = hook.text;
            let mut bytes = text.len().saturating_sub(heading().len());
            if let Some(id) = hook.note
                && let Some(sent) = self.sent.iter_mut().find(|sent| sent.item == id)
            {
                sent.acked = true;
                if sent.completed
                    && let Some(note) = self.note_done(list, id)
                {
                    bytes += note.text.len();
                    text = if text.is_empty() {
                        // A note alone has the update's heading in the record.
                        format!("{}{}", heading(), note.text)
                    } else {
                        format!("{text}\n\n{}", note.text)
                    };
                    ids.extend(note.host_ids.iter().cloned());
                    parts.push(RecordedPart::of(&note, true));
                }
            }
            if !text.is_empty() {
                delivered.push(Delivered {
                    text,
                    parts,
                    ids,
                    bytes,
                });
            }
        }
        (delivered, released)
    }

    /// The session's hooks end (a stop, a turn's end, or the next input):
    /// those not acknowledged are dropped, the typed items sent and the rests
    /// held wait again, and nothing in the list moves.
    pub(super) fn settle(&mut self) {
        self.in_flight.clear();
        self.sending = None;
        self.sent.clear();
        self.held.clear();
        self.as_context.clear();
    }
}

#[cfg(test)]
#[path = "hook_budget_tests.rs"]
mod tests;
