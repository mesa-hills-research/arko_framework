//! The packing and delivery rules of hooks, without a CLI.

use super::*;

/// A list of items, numbered as a task numbers them.
#[derive(Default)]
struct List {
    items: Vec<PendingContext>,
    next: u64,
}

impl List {
    fn of(texts: &[&str]) -> Self {
        let mut list = Self::default();
        for text in texts {
            list.push(PendingKind::Context, text, &[]);
        }
        list
    }

    fn insert(&mut self, at: usize, kind: PendingKind, text: &str, ids: &[&str]) -> u64 {
        self.next += 1;
        let mut item =
            PendingContext::new(kind, text, ids.iter().map(|id| id.to_string()).collect());
        item.id = self.next;
        self.items.insert(at.min(self.items.len()), item);
        self.next
    }

    fn push(&mut self, kind: PendingKind, text: &str, ids: &[&str]) -> u64 {
        self.insert(self.items.len(), kind, text, ids)
    }

    fn texts(&self) -> Vec<&str> {
        self.items.iter().map(|item| item.text.as_str()).collect()
    }
}

/// A hook's items, its heading taken out.
fn body(hook: &str) -> String {
    hook.strip_prefix(&heading()).unwrap_or(hook).to_string()
}

/// Texts the model was given in order, joined as the items they carry, each
/// split item's markers taken out.
fn stream(texts: &[String]) -> String {
    texts
        .iter()
        .map(|text| body(text))
        .collect::<Vec<_>>()
        .join("\n\n")
        .replace(&format!("{GOES_ON}\n\n{CONTINUED}"), "")
}

/// Hooks for calls `c1`, `c2`, … each reported at once, until the list is
/// empty. Returns their texts.
fn drain(delivery: &mut Delivery, list: &mut List) -> Vec<String> {
    let mut texts = Vec::new();
    for call in 1.. {
        if list.items.is_empty() {
            break;
        }
        let call = format!("c{call}");
        let text = delivery
            .fill(&list.items, &call)
            .expect("a hook carries something");
        assert!(text.len() <= HOOK_MAX_BYTES, "{}", text.len());
        texts.push(text);
        delivery.reported(&mut list.items, &call);
    }
    texts
}

#[test]
fn an_item_that_fits_an_empty_hook_waits_whole() {
    let mut delivery = Delivery::default();
    let mut list = List::of(&[&"a".repeat(6_000), &"b".repeat(6_000)]);
    let first = delivery.fill(&list.items, "c1").unwrap();
    assert_eq!(body(&first), "a".repeat(6_000));
    // Not acknowledged yet: a parallel call's hook carries what isn't in
    // flight.
    let second = delivery.fill(&list.items, "c2").unwrap();
    assert_eq!(body(&second), "b".repeat(6_000));
    assert!(delivery.fill(&list.items, "c3").is_none());
    assert_eq!(list.items.len(), 2);
    let (done, _) = delivery.reported(&mut list.items, "c1");
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].bytes, 6_000);
    assert_eq!(
        done[0]
            .parts
            .iter()
            .map(|part| (part.item, part.last))
            .collect::<Vec<_>>(),
        [(1, true)]
    );
    assert_eq!(list.texts(), ["b".repeat(6_000)]);
}

/// An item saved as bare text is never split, so its whole text lands in one
/// message. Longer than a hook, it waits for the next input with what is
/// behind it, and the hooks count as caught up meanwhile.
#[test]
fn a_long_item_saved_as_bare_text_waits_whole_for_the_next_input() {
    let mut delivery = Delivery::default();
    let legacy = "a legacy note line\n".repeat(650);
    assert!(legacy.len() > 12_000);
    let mut list = List::of(&[]);
    list.items.push(PendingContext {
        id: 1,
        ..legacy.as_str().into()
    });
    list.next = 1;
    list.push(PendingKind::Context, "Behind it.", &["b"]);
    assert!(delivery.fill(&list.items, "c1").is_none());
    assert!(delivery.caught_up(&list.items));
    // One that fits a hook goes in one whole.
    let mut short = List::of(&[]);
    short.items.push(PendingContext {
        id: 1,
        ..PendingContext::from("A short legacy note.")
    });
    assert!(!delivery.caught_up(&short.items));
    let hook = delivery.fill(&short.items, "c2").unwrap();
    assert_eq!(body(&hook), "A short legacy note.");
}

#[test]
fn a_long_item_moves_its_cursor_and_stays_whole_in_the_list() {
    let mut delivery = Delivery::default();
    let lines = (0..1_000)
        .map(|line| format!("line {line:04} of the update"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut list = List::of(&[&lines]);
    let first = delivery.fill(&list.items, "c1").unwrap();
    assert!(first.len() <= HOOK_MAX_BYTES);
    assert!(
        first.ends_with(&format!("of the update{GOES_ON}")),
        "{first}"
    );
    // A parallel call's hook carries nothing of it meanwhile.
    assert!(delivery.fill(&list.items, "c2").is_none());
    let (done, _) = delivery.reported(&mut list.items, "c1");
    assert!(!done[0].parts[0].last);
    // The item keeps its whole text, and its cursor says what went.
    assert_eq!(list.texts(), [lines.as_str()]);
    let given = list.items[0].given;
    assert!(given > 0 && lines[given..].starts_with("\nline "));
    let mut texts = vec![first];
    texts.extend(drain(&mut delivery, &mut list));
    assert_eq!(texts.len(), 3);
    assert_eq!(stream(&texts), lines);
}

#[test]
fn an_item_without_a_line_break_is_cut_on_a_character() {
    let mut delivery = Delivery::default();
    let text = "😀é".repeat(4_000);
    let mut list = List::of(&[&text]);
    let texts = drain(&mut delivery, &mut list);
    assert_eq!(texts.len(), 3);
    assert_eq!(stream(&texts), text);
}

#[test]
fn hooks_reported_out_of_order_leave_the_list_in_order() {
    let mut delivery = Delivery::default();
    let mut list = List::of(&[&"a".repeat(6_000), &"b".repeat(6_000), "c"]);
    delivery.fill(&list.items, "c1").unwrap();
    delivery.fill(&list.items, "c2").unwrap();
    assert!(delivery.reported(&mut list.items, "c2").0.is_empty());
    assert_eq!(list.items.len(), 3);
    let (done, _) = delivery.reported(&mut list.items, "c1");
    let texts = done.iter().map(|done| body(&done.text)).collect::<Vec<_>>();
    assert_eq!(
        texts,
        ["a".repeat(6_000), format!("{}\n\nc", "b".repeat(6_000))]
    );
    assert!(list.items.is_empty());
}

#[test]
fn a_typed_item_waits_until_the_hooks_carry_what_is_ahead_of_it() {
    let mut delivery = Delivery::default();
    let mut list = List::default();
    list.push(PendingKind::Context, &"x".repeat(16_000), &["note:2"]);
    list.push(PendingKind::Typed, "Use tabs.", &["note:3"]);
    list.push(PendingKind::Context, "Later mail.", &["mail:1"]);
    delivery.fill(&list.items, "c1").unwrap();
    assert!(
        delivery.due_note(&list.items).is_none(),
        "a rest is ahead of it"
    );
    delivery.reported(&mut list.items, "c1");
    let rest = delivery.fill(&list.items, "c2").unwrap();
    assert!(rest.contains(CONTINUED));
    assert!(
        !rest.contains("Later mail."),
        "nothing behind the note goes"
    );
    assert_eq!(
        delivery.due_note(&list.items),
        Some((2, "Use tabs.".to_string()))
    );
    delivery.note_sent("c2", "u1".into());
    // Behind a note sent, nothing goes until it is done.
    assert!(delivery.fill(&list.items, "c3").is_none());
    assert!(delivery.due_note(&list.items).is_none());
    assert_eq!(
        delivery.note_settled(&mut list.items, "u1", false),
        Verdict::Completed
    );
    let (done, _) = delivery.reported(&mut list.items, "c2");
    assert!(done[0].text.ends_with("\n\nUse tabs."));
    assert_eq!(done[0].ids, ["note:3"]);
    assert_eq!(list.texts(), ["Later mail."]);
}

#[test]
fn a_typed_item_is_done_when_both_its_ack_and_its_verdict_came() {
    for verdict_first in [true, false] {
        let mut delivery = Delivery::default();
        let mut list = List::default();
        list.push(PendingKind::Typed, "Use tabs.", &["note:3"]);
        list.push(PendingKind::Context, "Later.", &[]);
        assert!(delivery.fill(&list.items, "c1").is_none());
        delivery.due_note(&list.items).unwrap();
        delivery.note_sent("c1", "u1".into());
        let mut recorded = Vec::new();
        if verdict_first {
            assert_eq!(
                delivery.note_settled(&mut list.items, "u1", false),
                Verdict::Completed
            );
            assert_eq!(
                list.items.len(),
                2,
                "pending until its hook is acknowledged"
            );
            recorded.extend(delivery.reported(&mut list.items, "c1").0);
        } else {
            assert!(delivery.reported(&mut list.items, "c1").0.is_empty());
            assert_eq!(list.items.len(), 2, "pending until the CLI completed it");
            assert!(
                delivery.fill(&list.items, "c2").is_none(),
                "its barrier stays"
            );
            match delivery.note_settled(&mut list.items, "u1", false) {
                Verdict::Done(done) => recorded.push(done),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].text, format!("{CONTEXT_UPDATE}\n\nUse tabs."));
        assert_eq!(recorded[0].ids, ["note:3"]);
        assert_eq!(list.texts(), ["Later."]);
    }
}

#[test]
fn a_cancelled_typed_item_waits_where_it_is() {
    let mut delivery = Delivery::default();
    let mut list = List::default();
    list.push(PendingKind::Context, "First.", &[]);
    list.push(PendingKind::Typed, "Use tabs.", &[]);
    delivery.fill(&list.items, "c1").unwrap();
    delivery.due_note(&list.items).unwrap();
    delivery.note_sent("c1", "u1".into());
    list.push(PendingKind::Context, "Later.", &[]);
    // A parallel call's hook carries nothing behind the note sent.
    assert!(delivery.fill(&list.items, "c2").is_none());
    assert_eq!(
        delivery.note_settled(&mut list.items, "u1", true),
        Verdict::Waiting
    );
    assert_eq!(list.texts(), ["First.", "Use tabs.", "Later."]);
    // Its hook's record leaves it out, and it goes again before what came
    // after it.
    let (done, _) = delivery.reported(&mut list.items, "c1");
    assert_eq!(body(&done[0].text), "First.");
    assert!(delivery.fill(&list.items, "c3").is_none());
    assert_eq!(
        delivery.due_note(&list.items),
        Some((2, "Use tabs.".to_string()))
    );
}

#[test]
fn nothing_moves_when_the_session_ends() {
    let mut delivery = Delivery::default();
    let mut list = List::default();
    list.push(PendingKind::Context, &"x".repeat(16_000), &[]);
    list.push(PendingKind::Typed, "Use tabs.", &[]);
    list.push(PendingKind::Typed, "And spaces.", &[]);
    delivery.fill(&list.items, "c1").unwrap();
    delivery.reported(&mut list.items, "c1");
    delivery.fill(&list.items, "c2").unwrap();
    delivery.due_note(&list.items).unwrap();
    delivery.note_sent("c2", "u1".into());
    let before = list.items.clone();
    delivery.settle();
    assert_eq!(list.items, before);
    // Whatever was in flight goes again.
    let again = delivery.fill(&list.items, "c3").unwrap();
    assert!(again.starts_with(&format!("{CONTEXT_UPDATE}\n\n{CONTINUED}")));
}

#[test]
fn a_long_denial_is_cut_and_its_rest_waits_for_the_calls_result() {
    let mut delivery = Delivery::default();
    let mut list = List::default();
    list.push(PendingKind::Context, &"a".repeat(9_450), &[]);
    list.push(PendingKind::Context, "Older waiting context.", &[]);
    delivery.fill(&list.items, "c1").unwrap();
    let mut rests = Vec::new();
    for (call, letter) in [("d1", "y"), ("d2", "z")] {
        let denial = format!("arko did not run this call: {}", letter.repeat(12_000));
        let (cut, rest) = cut_denial(denial.clone());
        assert!(cut.len() <= HOOK_MAX_BYTES && cut.ends_with(DENIAL_GOES_ON));
        let rest = rest.unwrap();
        assert_eq!(
            format!("{}{rest}", cut.strip_suffix(DENIAL_GOES_ON).unwrap()),
            denial
        );
        let at = delivery.rest_place(&list.items);
        let id = list.insert(
            at,
            PendingKind::DenialRest { call: call.into() },
            &rest,
            &[],
        );
        delivery.hold_rest(call, id);
        rests.push(rest);
    }
    // Behind what is in flight, ahead of what waits, in the order given.
    assert_eq!(
        list.texts()[1..],
        [
            rests[0].as_str(),
            rests[1].as_str(),
            "Older waiting context."
        ]
    );
    delivery.reported(&mut list.items, "c1");
    assert!(
        delivery.fill(&list.items, "c2").is_none(),
        "held for its call"
    );
    assert!(delivery.reported(&mut list.items, "d1").1, "released");
    let next = delivery.fill(&list.items, "c3").unwrap();
    assert!(next.starts_with(&format!("{CONTEXT_UPDATE}\n\n{DENIAL_REST}")));
}

#[test]
fn ids_are_reported_with_an_items_first_part() {
    let mut delivery = Delivery::default();
    let mut list = List::default();
    list.push(PendingKind::Context, &"m".repeat(12_000), &["mail:1"]);
    list.push(
        PendingKind::Context,
        "Message 7 from the host.",
        &["message:7"],
    );
    delivery.fill(&list.items, "c1").unwrap();
    let (first, _) = delivery.reported(&mut list.items, "c1");
    assert_eq!(first[0].ids, ["mail:1"]);
    delivery.fill(&list.items, "c2").unwrap();
    let (second, _) = delivery.reported(&mut list.items, "c2");
    assert_eq!(second[0].ids, ["message:7"], "the rest has none of its own");
    assert_eq!(
        second[0]
            .parts
            .iter()
            .map(|part| (part.item, part.last))
            .collect::<Vec<_>>(),
        [(1, true), (2, true)]
    );
}

/// A small random generator, so the property test needs no crate.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    /// Text of up to `max` bytes, with line breaks and multi-byte and astral
    /// characters, never only white space.
    fn text(&mut self, max: usize) -> String {
        let target = match self.below(4) {
            0 => self.below(200),
            _ => self.below(max + 1),
        };
        let mut text = String::from("t");
        while text.len() < target {
            text.push(match self.below(14) {
                0 | 1 => '\n',
                2 => 'é',
                3 => '…',
                4 => '𝄞',
                5 => '😀',
                _ => (b'a' + self.below(26) as u8) as char,
            });
        }
        text
    }
}

/// Random sequences of context, updates with typed items, denials' rests,
/// hooks reported out of order, the CLI's verdicts in either order and
/// sessions that end, from small to 40,000 bytes. Every hook stays within
/// the budget, nothing in the list moves, and the records, then the next
/// input, give back every item exactly once and in the list's order,
/// markers taken out.
#[test]
fn every_hook_is_within_the_budget_and_nothing_is_lost_or_reordered() {
    // What the runs exercised.
    let (mut splits, mut notes, mut late, mut rests) = (0, 0, 0, 0);
    for seed in 1..=400u64 {
        let mut random = Random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut delivery = Delivery::default();
        let mut list = List::default();
        // The items as the model should read them, in the list's order.
        let mut expected: Vec<(u64, String)> = Vec::new();
        let mut given: Vec<String> = Vec::new();
        // The items the records hold, part by part, in order.
        let mut order: Vec<u64> = Vec::new();
        let mut unreported: Vec<String> = Vec::new();
        let mut sent: Vec<String> = Vec::new();
        let mut calls = 0;
        for step in 0..random.below(50) + 1 {
            // The items recorded left the list from its front, and nothing
            // in it moved.
            let recorded = expected.len() - list.items.len();
            let tail = expected[recorded..]
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            let ids = list.items.iter().map(|item| item.id).collect::<Vec<_>>();
            assert_eq!(tail, ids, "seed {seed}, step {step}");
            let mut seen = order.clone();
            seen.dedup();
            let first = expected
                .iter()
                .map(|(id, _)| *id)
                .take(seen.len())
                .collect::<Vec<_>>();
            assert_eq!(
                seen, first,
                "seed {seed}, step {step}: recorded out of order"
            );
            calls += 1;
            let call = format!("c{calls}");
            let op = random.below(9);
            match op {
                0 | 1 => {
                    let text = random.text(40_000);
                    let id = list.push(PendingKind::Context, &text, &[]);
                    expected.push((id, text));
                }
                2 => {
                    for kind in [PendingKind::Context, PendingKind::Typed] {
                        if random.below(2) == 0 {
                            let text = random.text(20_000);
                            let id = list.push(kind, &text, &[]);
                            expected.push((id, text));
                        }
                    }
                }
                3 => {
                    let (cut, rest) = cut_denial(random.text(40_000));
                    assert!(cut.len() <= HOOK_MAX_BYTES);
                    if let Some(rest) = rest {
                        rests += 1;
                        let at = delivery.rest_place(&list.items);
                        // The items recorded so far left the list from its
                        // front, so the list's place is after them.
                        let place = expected.len() - list.items.len() + at;
                        let id = list.insert(
                            at,
                            PendingKind::DenialRest { call: call.clone() },
                            &rest,
                            &[],
                        );
                        delivery.hold_rest(&call, id);
                        expected.insert(place, (id, format!("{DENIAL_REST}{rest}")));
                        unreported.push(call);
                    }
                }
                4 if !unreported.is_empty() => {
                    late += 1;
                    let late = unreported.remove(random.below(unreported.len()));
                    for done in delivery.reported(&mut list.items, &late).0 {
                        order.extend(done.parts.iter().map(|part| part.item));
                        given.push(done.text);
                    }
                }
                5 if !sent.is_empty() => {
                    let uuid = sent.remove(random.below(sent.len()));
                    let cancel = random.below(4) == 0;
                    if let Verdict::Done(done) =
                        delivery.note_settled(&mut list.items, &uuid, cancel)
                    {
                        order.extend(done.parts.iter().map(|part| part.item));
                        given.push(done.text);
                    }
                }
                6 if random.below(4) == 0 => {
                    // The session ends: what is in flight goes again.
                    delivery.settle();
                    unreported.clear();
                    sent.clear();
                }
                _ => {
                    let before = list.texts().join("\u{0}");
                    if let Some(text) = delivery.fill(&list.items, &call) {
                        assert!(text.len() <= HOOK_MAX_BYTES, "{}", text.len());
                        splits += usize::from(text.ends_with(GOES_ON));
                    }
                    if delivery.due_note(&list.items).is_some() {
                        notes += 1;
                        let uuid = format!("u{calls}");
                        delivery.note_sent(&call, uuid.clone());
                        sent.push(uuid);
                    }
                    assert_eq!(list.texts().join("\u{0}"), before, "a hook moves nothing");
                    unreported.push(call);
                }
            }
        }
        // Every call is reported and every message completed, then the turn
        // ends and the next input carries the rest.
        while !unreported.is_empty() || !sent.is_empty() {
            if let Some(call) = unreported.pop() {
                given.extend(
                    delivery
                        .reported(&mut list.items, &call)
                        .0
                        .into_iter()
                        .map(|done| done.text),
                );
            }
            if let Some(uuid) = sent.pop()
                && let Verdict::Done(done) = delivery.note_settled(&mut list.items, &uuid, false)
            {
                given.push(done.text);
            }
        }
        delivery.settle();
        given.extend(list.items.iter().map(|item| shown_from(item, item.given)));
        let expected = expected
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>()
            .join("\n\n");
        let got = stream(&given);
        assert!(
            got == expected,
            "seed {seed}: given {} bytes, expected {}",
            got.len(),
            expected.len()
        );
    }
    assert!(
        splits > 100 && notes > 50 && late > 50 && rests > 50,
        "{splits} {notes} {late} {rests}"
    );
}
