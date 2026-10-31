//! The packing and delivery rules of hooks, without a CLI.

use super::*;

fn no_ids(_: &str) -> Option<String> {
    None
}

/// A hook's items, its heading taken out.
fn body(hook: &str) -> String {
    hook.strip_prefix(&heading()).unwrap_or(hook).to_string()
}

/// Texts the model was given in order, joined as the items they carry,
/// each split item's markers taken out.
fn stream(texts: &[String]) -> String {
    texts
        .iter()
        .map(|text| body(text))
        .collect::<Vec<_>>()
        .join("\n\n")
        .replace(&format!("{GOES_ON}\n\n{CONTINUED}"), "")
}

/// Hooks for calls `c1`, `c2`, … each reported at once, until the queue
/// is empty. Returns their texts.
fn drain(delivery: &mut Delivery, queue: &mut Vec<String>) -> Vec<String> {
    let mut texts = Vec::new();
    for call in 1.. {
        if queue.is_empty() {
            break;
        }
        let call = format!("c{call}");
        let text = delivery
            .fill(queue, &call)
            .expect("a hook carries something");
        assert!(text.len() <= HOOK_MAX_BYTES, "{}", text.len());
        texts.push(text);
        delivery.reported(queue, &call, &no_ids);
    }
    texts
}

#[test]
fn an_item_that_fits_an_empty_hook_waits_whole() {
    let mut delivery = Delivery::default();
    let mut queue = vec!["a".repeat(6_000), "b".repeat(6_000)];
    let first = delivery.fill(&queue, "c1").unwrap();
    assert_eq!(body(&first), "a".repeat(6_000));
    // Not acknowledged yet: a parallel call's hook carries what isn't in
    // flight.
    let second = delivery.fill(&queue, "c2").unwrap();
    assert_eq!(body(&second), "b".repeat(6_000));
    assert!(delivery.fill(&queue, "c3").is_none());
    assert_eq!(queue.len(), 2);
    let done = delivery.reported(&mut queue, "c1", &no_ids);
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].bytes, 6_000);
    assert_eq!(queue, ["b".repeat(6_000)]);
}

#[test]
fn a_long_item_is_split_at_a_line_break_near_the_cut() {
    let mut delivery = Delivery::default();
    let lines = (0..1_000)
        .map(|line| format!("line {line:04} of the update"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut queue = vec![lines.clone()];
    let first = delivery.fill(&queue, "c1").unwrap();
    assert!(first.len() <= HOOK_MAX_BYTES);
    assert!(
        first.ends_with(&format!("of the update{GOES_ON}")),
        "{first}"
    );
    // The whole item stays queued until the part is acknowledged, and a
    // parallel call's hook carries nothing of it meanwhile.
    assert!(delivery.fill(&queue, "c2").is_none());
    assert_eq!(queue, std::slice::from_ref(&lines));
    delivery.reported(&mut queue, "c1", &no_ids);
    assert!(
        queue[0].starts_with(&format!("{CONTINUED}\nline ")),
        "{}",
        &queue[0][..80]
    );
    let mut texts = vec![first];
    texts.extend(drain(&mut delivery, &mut queue));
    assert_eq!(texts.len(), 3);
    assert_eq!(stream(&texts), lines);
}

#[test]
fn an_item_without_a_line_break_is_cut_on_a_character() {
    let mut delivery = Delivery::default();
    let text = "😀é".repeat(4_000);
    let mut queue = vec![text.clone()];
    let texts = drain(&mut delivery, &mut queue);
    assert_eq!(texts.len(), 3);
    assert_eq!(stream(&texts), text);
}

#[test]
fn hooks_reported_out_of_order_leave_the_queue_in_order() {
    let mut delivery = Delivery::default();
    let mut queue = vec!["a".repeat(6_000), "b".repeat(6_000), "c".into()];
    delivery.fill(&queue, "c1").unwrap();
    delivery.fill(&queue, "c2").unwrap();
    assert!(delivery.reported(&mut queue, "c2", &no_ids).is_empty());
    assert_eq!(queue.len(), 3);
    let done = delivery.reported(&mut queue, "c1", &no_ids);
    let texts = done.iter().map(|done| body(&done.text)).collect::<Vec<_>>();
    assert_eq!(
        texts,
        ["a".repeat(6_000), format!("{}\n\nc", "b".repeat(6_000))]
    );
    assert!(queue.is_empty());
}

#[test]
fn a_typed_note_waits_for_what_is_ahead_of_it() {
    let mut delivery = Delivery::default();
    let mut queue = vec!["x".repeat(16_000)];
    let marks = Marks::from([("notes".to_string(), 3)]);
    let kept = delivery.take_update(
        &mut queue,
        String::new(),
        "Use tabs.".into(),
        vec!["note:3".into()],
        marks.clone(),
        true,
    );
    assert!(kept.is_empty(), "the note keeps them");
    assert!(!delivery.caught_up(&queue));
    delivery.fill(&queue, "c1").unwrap();
    assert!(delivery.due_note(&queue).is_none(), "a rest is ahead of it");
    delivery.reported(&mut queue, "c1", &no_ids);
    let rest = delivery.fill(&queue, "c2").unwrap();
    assert!(rest.contains(CONTINUED));
    assert_eq!(delivery.due_note(&queue).as_deref(), Some("Use tabs."));
    delivery.note_sent("c2", "u1".into());
    let done = delivery.reported(&mut queue, "c2", &no_ids);
    assert!(done[0].text.ends_with("\n\nUse tabs."));
    assert_eq!(done[0].ids, ["note:3"]);
    assert_eq!(done[0].marks, marks);
    assert!(queue.is_empty());
}

#[test]
fn a_typed_note_alone_goes_at_once() {
    let mut delivery = Delivery::default();
    let mut queue = Vec::new();
    delivery.take_update(
        &mut queue,
        String::new(),
        "Use tabs.".into(),
        Vec::new(),
        Marks::new(),
        true,
    );
    assert!(delivery.fill(&queue, "c1").is_none());
    assert_eq!(delivery.due_note(&queue).as_deref(), Some("Use tabs."));
    delivery.note_sent("c1", "u1".into());
    let done = delivery.reported(&mut queue, "c1", &no_ids);
    assert_eq!(done[0].text, format!("{CONTEXT_UPDATE}\n\nUse tabs."));
}

#[test]
fn a_stop_puts_a_waiting_note_at_its_place() {
    let mut delivery = Delivery::default();
    let mut queue = vec!["x".repeat(16_000)];
    delivery.take_update(
        &mut queue,
        String::new(),
        "Use tabs.".into(),
        Vec::new(),
        Marks::from([("notes".to_string(), 3)]),
        true,
    );
    queue.push("Later mail.".into());
    delivery.fill(&queue, "c1").unwrap();
    let marks = delivery.session_ended(&mut queue);
    assert_eq!(
        queue,
        ["x".repeat(16_000), "Use tabs.".into(), "Later mail.".into()]
    );
    assert_eq!(marks.get("notes"), Some(&3));
}

#[test]
fn a_cancelled_note_goes_back_behind_what_was_ahead_of_it() {
    let mut delivery = Delivery::default();
    let mut queue = vec!["First.".to_string()];
    delivery.take_update(
        &mut queue,
        String::new(),
        "Use tabs.".into(),
        Vec::new(),
        Marks::new(),
        true,
    );
    delivery.fill(&queue, "c1").unwrap();
    delivery.due_note(&queue).unwrap();
    delivery.note_sent("c1", "u1".into());
    queue.push("Later.".into());
    delivery.note_settled(&mut queue, "u1", true);
    assert_eq!(queue, ["First.", "Use tabs.", "Later."]);
    // Its hook's record leaves it out.
    let done = delivery.reported(&mut queue, "c1", &no_ids);
    assert_eq!(body(&done[0].text), "First.");
    assert_eq!(queue, ["Use tabs.", "Later."]);
}

#[test]
fn a_long_denial_is_cut_and_its_rest_comes_first() {
    let mut delivery = Delivery::default();
    let mut queue = vec!["Older context.".to_string()];
    let denial = format!("arko did not run this call: {}", "y".repeat(12_000));
    let cut = delivery.deny(&mut queue, Some("c1"), denial.clone());
    assert!(cut.len() <= HOOK_MAX_BYTES);
    assert!(cut.ends_with(GOES_ON));
    assert_eq!(queue.len(), 1, "the rest waits for the call's result");
    delivery.reported(&mut queue, "c1", &no_ids);
    assert_eq!(queue.len(), 2);
    assert!(queue[0].starts_with(CONTINUED));
    assert_eq!(stream(&[cut, queue[0].clone()]), denial);
}

#[test]
fn ids_are_reported_with_an_items_first_part() {
    let mut delivery = Delivery::default();
    let mut queue = Vec::new();
    let mail = "m".repeat(12_000);
    delivery.take_update(
        &mut queue,
        mail,
        String::new(),
        vec!["mail:1".into()],
        Marks::new(),
        true,
    );
    queue.push("Message 7 from the host.".into());
    let identify = |text: &str| {
        text.starts_with("Message 7")
            .then(|| "message:7".to_string())
    };
    delivery.fill(&queue, "c1").unwrap();
    let first = delivery.reported(&mut queue, "c1", &identify);
    assert_eq!(first[0].ids, ["mail:1"]);
    delivery.fill(&queue, "c2").unwrap();
    let second = delivery.reported(&mut queue, "c2", &identify);
    assert_eq!(second[0].ids, ["message:7"], "the rest has none of its own");
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

    /// Text of up to `max` bytes (any size from empty), with line breaks
    /// and multi-byte and astral characters.
    fn text(&mut self, max: usize) -> String {
        let target = match self.below(4) {
            0 => self.below(200),
            _ => self.below(max + 1),
        };
        let mut text = String::new();
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

/// Random sequences of injections, updates with typed notes, approval
/// messages and denials, from empty to 40,000 bytes, with calls reported
/// out of order. Every hook and denial stays within the budget, and the
/// hooks, then the next input, give back every item once and in order,
/// markers taken out.
#[test]
fn every_hook_is_within_the_budget_and_nothing_is_lost_or_reordered() {
    // What the runs exercised: split items, typed notes sent and calls
    // reported after a later one.
    let (mut splits, mut notes, mut late) = (0, 0, 0);
    for seed in 1..=400u64 {
        let mut random = Random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut delivery = Delivery::default();
        let mut queue: Vec<String> = Vec::new();
        // What the model is given in order, and what was queued or typed
        // in the order it should come.
        let mut given: Vec<String> = Vec::new();
        let mut expected: Vec<String> = Vec::new();
        let mut unreported: Vec<String> = Vec::new();
        let mut calls = 0;
        let steps = random.below(40) + 1;
        for _ in 0..steps {
            calls += 1;
            let call = format!("c{calls}");
            match random.below(7) {
                // An injection, or an approval message, joins the back.
                0 | 1 => {
                    let text = random.text(40_000);
                    if !text.trim().is_empty() {
                        expected.push(text.clone());
                        queue.push(text);
                    }
                }
                // The source's update, with typed notes or not, once
                // nothing waits outside a hook.
                2 if delivery.caught_up(&queue) => {
                    let content = random.text(40_000);
                    let typed = match random.below(2) {
                        0 => random.text(20_000),
                        _ => String::new(),
                    };
                    for text in [&content, &typed] {
                        if !text.trim().is_empty() {
                            expected.push(text.clone());
                        }
                    }
                    delivery.take_update(
                        &mut queue,
                        content,
                        typed,
                        Vec::new(),
                        Marks::new(),
                        true,
                    );
                }
                // A denied call, when nothing waits or is in flight, so
                // that its rest comes next.
                3 if delivery.caught_up(&queue)
                    && delivery.in_flight.is_empty()
                    && queue.is_empty() =>
                {
                    let text = random.text(40_000);
                    let denial = delivery.deny(&mut queue, Some(&call), text.clone());
                    assert!(denial.len() <= HOOK_MAX_BYTES, "{}", denial.len());
                    expected.push(text);
                    given.push(denial);
                    delivery.reported(&mut queue, &call, &no_ids);
                }
                // A call reported late.
                4 if !unreported.is_empty() => {
                    late += 1;
                    let late = unreported.remove(random.below(unreported.len()));
                    for done in delivery.reported(&mut queue, &late, &no_ids) {
                        given.push(done.text);
                    }
                }
                // A call's hook, reported now or later.
                _ => {
                    if let Some(text) = delivery.fill(&queue, &call) {
                        assert!(text.len() <= HOOK_MAX_BYTES, "{}", text.len());
                        splits += usize::from(text.ends_with(GOES_ON));
                    }
                    if delivery.due_note(&queue).is_some() {
                        notes += 1;
                        delivery.note_sent(&call, format!("u{calls}"));
                    }
                    if random.below(3) == 0 {
                        unreported.push(call);
                    } else {
                        for done in delivery.reported(&mut queue, &call, &no_ids) {
                            given.push(done.text);
                        }
                    }
                }
            }
        }
        // Every call is reported, and the turn ends: the next input
        // carries the rest whole.
        while !unreported.is_empty() {
            let late = unreported.remove(random.below(unreported.len()));
            for done in delivery.reported(&mut queue, &late, &no_ids) {
                given.push(done.text);
            }
        }
        delivery.session_ended(&mut queue);
        given.extend(queue.iter().cloned());
        let expected = expected.join("\n\n");
        assert!(
            stream(&given) == expected,
            "seed {seed}: given {} bytes, expected {}",
            stream(&given).len(),
            expected.len()
        );
    }
    assert!(
        splits > 100 && notes > 50 && late > 100,
        "{splits} {notes} {late}"
    );
}
