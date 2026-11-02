//! What a tool call did, as the tool saw it, for a host to show without
//! doing the work again. A receipt is small and bounded, so it travels with
//! the call's result event ([`AgentEvent::ToolResult`](crate::AgentEvent)).

use serde::{Deserialize, Serialize};

/// A tool's account of what its call did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolReceipt {
    /// What `edit_file` changed.
    Edit(EditReceipt),
}

/// What an edit changed in its file, as diff hunks of whole lines, from the
/// text the tool read, the matches it found and the text it wrote.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditReceipt {
    /// The changes in the file's order, each with up to
    /// [`EDIT_RECEIPT_CONTEXT`] unchanged lines on each side.
    pub hunks: Vec<EditHunk>,
    /// Lines or changes were left out to keep it within its limits
    /// ([`EDIT_RECEIPT_LINES`], [`EDIT_RECEIPT_BYTES`] and
    /// [`EDIT_RECEIPT_LINE_BYTES`]).
    #[serde(default)]
    pub truncated: bool,
}

/// One stretch of an edit, its unchanged, removed and added lines in order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditHunk {
    /// The line its first line was on before the edit, numbered from 1.
    /// For lines added where nothing was removed, with no line before them,
    /// it is the line they came before.
    pub old_start: usize,
    /// The line its first line is on after the edit.
    pub new_start: usize,
    pub lines: Vec<EditLine>,
}

/// A line of a hunk, without its line break.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditLine {
    pub change: LineChange,
    pub text: String,
}

/// What became of a line of a hunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineChange {
    /// Unchanged, around a change.
    Kept,
    /// In the file before the edit only.
    Removed,
    /// In the file after the edit only.
    Added,
}

/// The unchanged lines a hunk shows on each side of a change.
pub const EDIT_RECEIPT_CONTEXT: usize = 3;
/// The most lines an edit receipt holds.
pub const EDIT_RECEIPT_LINES: usize = 400;
/// The most bytes of text an edit receipt holds.
pub const EDIT_RECEIPT_BYTES: usize = 64 * 1024;
/// The most bytes a line of an edit receipt holds. A longer one is cut at a
/// character boundary.
pub const EDIT_RECEIPT_LINE_BYTES: usize = 2_000;

/// The receipt of an edit that turned `before` into `after`, replacing the
/// `old_len` bytes at each of `found` (in order, not overlapping) with the
/// `new_len` bytes that start at the same place of `starts` in `after`. The
/// work is linear in the two texts and the receipt.
pub(crate) fn edit_receipt(
    before: &str,
    after: &str,
    found: &[usize],
    starts: &[usize],
    old_len: usize,
    new_len: usize,
) -> EditReceipt {
    let mut built = Builder {
        before,
        after,
        receipt: EditReceipt::default(),
        hunk: None,
        lines: 0,
        bytes: 0,
    };
    // The line of `before` the scan is on, and where that line starts.
    let mut scanned = 0;
    let mut line = 1;
    let mut line_start = 0;
    // How many more lines `after` has than `before` up to the last change.
    let mut shift: isize = 0;
    let mut at = 0;
    while at < found.len() && !built.full() {
        let first = at;
        let a = found[at];
        for (offset, byte) in before.as_bytes()[scanned..a].iter().enumerate() {
            if *byte == b'\n' {
                line += 1;
                line_start = scanned + offset + 1;
            }
        }
        scanned = a;
        // The change's whole lines, from the start of the first match's line
        // to the end of the last match's, with every match that starts on a
        // line it holds.
        let start = line_start;
        let mut end;
        loop {
            let match_end = found[at] + old_len;
            let next = found.get(at + 1).copied();
            end = if before.as_bytes()[match_end - 1] == b'\n' {
                Some(match_end)
            } else {
                line_end(before, match_end, next)
            };
            // The match ended at a line break, and its new text doesn't, so
            // the line after it is part of the change too.
            if end == Some(match_end)
                && match_end < before.len()
                && !at_line_break(after, starts[at] + new_len)
            {
                end = line_end(before, match_end, next);
            }
            match end {
                Some(end) if next.is_none_or(|next| next >= end) => break,
                _ => at += 1,
            }
        }
        let end = end.unwrap_or(before.len());
        let last = at;
        at += 1;
        // The same lines after the edit.
        let after_start = starts[first] - (found[first] - start);
        let after_end = end - (found[last] + old_len) + (starts[last] + new_len);
        let old_line = line;
        let new_line = (line as isize + shift) as usize;
        shift += lines_of(&after[after_start..after_end]).count() as isize
            - lines_of(&before[start..end]).count() as isize;
        built.change(Change {
            old_line,
            new_line,
            start,
            end,
            after_start,
            after_end,
        });
        // (The scan goes on from the change's end.)
        for (offset, byte) in before.as_bytes()[scanned..end].iter().enumerate() {
            if *byte == b'\n' {
                line += 1;
                line_start = scanned + offset + 1;
            }
        }
        scanned = end;
    }
    if at < found.len() {
        built.receipt.truncated = true;
    }
    built.finish()
}

/// Where the line of `text` that holds byte `from` ends (after its line
/// break, or at the end of the text), or `None` when the match at `next`
/// starts on it first.
fn line_end(text: &str, from: usize, next: Option<usize>) -> Option<usize> {
    let limit = next.unwrap_or(text.len()).max(from);
    match text.as_bytes()[from..limit]
        .iter()
        .position(|byte| *byte == b'\n')
    {
        Some(offset) => Some(from + offset + 1),
        None if next.is_some() => None,
        None => Some(text.len()),
    }
}

/// Whether byte `at` of `text` starts a line, or is its end.
fn at_line_break(text: &str, at: usize) -> bool {
    at == 0 || at == text.len() || text.as_bytes()[at - 1] == b'\n'
}

/// The lines of `text`, whole lines, without their line breaks.
fn lines_of(text: &str) -> impl Iterator<Item = &str> {
    text.split_inclusive('\n').map(|line| {
        let line = line.strip_suffix('\n').unwrap_or(line);
        line.strip_suffix('\r').unwrap_or(line)
    })
}

/// [`lines_of`], from the last line back.
fn lines_back(text: &str) -> impl Iterator<Item = &str> {
    text.rsplit_terminator('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
}

/// `at` moved on past `lines` lines of `text` (each with its line break).
fn advance(text: &str, mut at: usize, lines: usize) -> usize {
    for _ in 0..lines {
        at = text[at..]
            .find('\n')
            .map_or(text.len(), |offset| at + offset + 1);
    }
    at
}

/// `end`, which ends a line of `text`, moved back by `lines` lines.
fn back(text: &str, mut end: usize, lines: usize) -> usize {
    for _ in 0..lines {
        let upto = if end > 0 && text.as_bytes()[end - 1] == b'\n' {
            end - 1
        } else {
            end
        };
        end = text[..upto].rfind('\n').map_or(0, |at| at + 1);
    }
    end
}

/// A change's whole lines, by their lines and bytes before and after.
struct Change {
    old_line: usize,
    new_line: usize,
    /// Its bytes in the text before.
    start: usize,
    end: usize,
    /// Its bytes in the text after.
    after_start: usize,
    after_end: usize,
}

/// A receipt as it's made. Its lines are copied in only while they fit,
/// so making it takes memory in proportion to the receipt.
struct Builder<'a> {
    before: &'a str,
    after: &'a str,
    receipt: EditReceipt,
    /// The hunk being made, with the line after its last change (before the
    /// edit) and the byte where that line starts.
    hunk: Option<(EditHunk, usize, usize)>,
    lines: usize,
    bytes: usize,
}

impl<'a> Builder<'a> {
    fn full(&self) -> bool {
        self.lines >= EDIT_RECEIPT_LINES || self.bytes >= EDIT_RECEIPT_BYTES
    }

    /// Adds `change`, with the unchanged lines around it, to the hunk of the
    /// change before it when their context meets, else to a new one.
    fn change(&mut self, mut change: Change) {
        // The lines both sides share at either end are unchanged, unless
        // nothing tells the two apart once their line breaks are off, when
        // the change shows as it is.
        let old = &self.before[change.start..change.end];
        let new = &self.after[change.after_start..change.after_end];
        let (old_lines, new_lines) = (lines_of(old).count(), lines_of(new).count());
        let lead = lines_of(old)
            .zip(lines_of(new))
            .take_while(|(old, new)| old == new)
            .count();
        let tail = lines_back(old)
            .zip(lines_back(new))
            .take(old_lines.min(new_lines) - lead)
            .take_while(|(old, new)| old == new)
            .count();
        let mut removed = old_lines - lead - tail;
        if removed > 0 || new_lines - lead - tail > 0 {
            change.start = advance(self.before, change.start, lead);
            change.end = back(self.before, change.end, tail);
            change.after_start = advance(self.after, change.after_start, lead);
            change.after_end = back(self.after, change.after_end, tail);
            change.old_line += lead;
            change.new_line += lead;
        } else {
            removed = old_lines;
        }
        let gap = self
            .hunk
            .as_ref()
            .map(|(_, next, _)| change.old_line.saturating_sub(*next));
        match (self.hunk.take(), gap) {
            (Some((mut hunk, _, from)), Some(gap)) if gap <= 2 * EDIT_RECEIPT_CONTEXT => {
                for line in lines_of(&self.before[from..change.start]) {
                    self.push(&mut hunk, LineChange::Kept, line);
                }
                self.hunk = Some((hunk, 0, 0));
            }
            (Some((hunk, _, from)), _) => {
                let hunk = self.close(hunk, from);
                self.receipt.hunks.push(hunk);
                self.open(&change);
            }
            (None, _) => self.open(&change),
        }
        let Some((mut hunk, _, _)) = self.hunk.take() else {
            return;
        };
        // (Each side's lines until the receipt is full, and no further.)
        let (before, after) = (self.before, self.after);
        for line in lines_of(&before[change.start..change.end]) {
            if !self.push(&mut hunk, LineChange::Removed, line) {
                break;
            }
        }
        for line in lines_of(&after[change.after_start..change.after_end]) {
            if !self.push(&mut hunk, LineChange::Added, line) {
                break;
            }
        }
        self.hunk = Some((hunk, change.old_line + removed, change.end));
    }

    /// Starts the hunk of `change`, with up to [`EDIT_RECEIPT_CONTEXT`]
    /// unchanged lines before it.
    fn open(&mut self, change: &Change) {
        let mut from = change.start;
        let mut context = 0;
        while context < EDIT_RECEIPT_CONTEXT && from > 0 {
            from = self.before[..from - 1].rfind('\n').map_or(0, |at| at + 1);
            context += 1;
        }
        let mut hunk = EditHunk {
            old_start: change.old_line - context,
            new_start: change.new_line - context,
            lines: Vec::new(),
        };
        for line in lines_of(&self.before[from..change.start]) {
            self.push(&mut hunk, LineChange::Kept, line);
        }
        self.hunk = Some((hunk, 0, 0));
    }

    /// `hunk` with up to [`EDIT_RECEIPT_CONTEXT`] unchanged lines from byte
    /// `from` on.
    fn close(&mut self, mut hunk: EditHunk, from: usize) -> EditHunk {
        for line in lines_of(&self.before[from..]).take(EDIT_RECEIPT_CONTEXT) {
            self.push(&mut hunk, LineChange::Kept, line);
        }
        hunk
    }

    /// Adds `text` to `hunk`, cut to [`EDIT_RECEIPT_LINE_BYTES`] and to the
    /// bytes the receipt has room for, at a character boundary. False when
    /// the receipt is full, which leaves it out.
    fn push(&mut self, hunk: &mut EditHunk, change: LineChange, text: &str) -> bool {
        let room = EDIT_RECEIPT_BYTES.saturating_sub(self.bytes);
        if self.lines >= EDIT_RECEIPT_LINES || room == 0 {
            self.receipt.truncated = true;
            return false;
        }
        let mut cut = text.len().min(EDIT_RECEIPT_LINE_BYTES).min(room);
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut < text.len() {
            self.receipt.truncated = true;
            if cut == 0 {
                // (Not a character of it fits, so the receipt is full.)
                self.bytes = EDIT_RECEIPT_BYTES;
                return false;
            }
        }
        self.lines += 1;
        self.bytes += cut;
        hunk.lines.push(EditLine {
            change,
            text: text[..cut].to_string(),
        });
        true
    }

    fn finish(mut self) -> EditReceipt {
        if let Some((hunk, _, from)) = self.hunk.take() {
            let hunk = self.close(hunk, from);
            self.receipt.hunks.push(hunk);
        }
        self.receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The receipt of replacing every `old` in `before` with `new`, as
    /// `edit_file` does.
    fn receipt(before: &str, old: &str, new: &str) -> EditReceipt {
        let found: Vec<usize> = before.match_indices(old).map(|(at, _)| at).collect();
        let (mut after, mut starts, mut rest) = (String::new(), Vec::new(), 0);
        for at in &found {
            after.push_str(&before[rest..*at]);
            starts.push(after.len());
            after.push_str(new);
            rest = at + old.len();
        }
        after.push_str(&before[rest..]);
        edit_receipt(before, &after, &found, &starts, old.len(), new.len())
    }

    /// A receipt as a diff reads, each hunk's line numbers before and after,
    /// then its lines.
    fn diff(receipt: &EditReceipt) -> Vec<String> {
        let mut lines = Vec::new();
        for hunk in &receipt.hunks {
            lines.push(format!("@{} {}", hunk.old_start, hunk.new_start));
            for line in &hunk.lines {
                let mark = match line.change {
                    LineChange::Kept => ' ',
                    LineChange::Removed => '-',
                    LineChange::Added => '+',
                };
                lines.push(format!("{mark}{}", line.text));
            }
        }
        lines
    }

    #[test]
    fn a_change_is_its_whole_lines_with_the_lines_around_it() {
        let file = "1\n2\n3\nfoo bar\n5\n6\n7\n8\n";
        assert_eq!(
            diff(&receipt(file, "foo", "baz")),
            [
                "@1 1", " 1", " 2", " 3", "-foo bar", "+baz bar", " 5", " 6", " 7"
            ]
        );
        // New text the old one holds, at the start of the file.
        assert_eq!(
            diff(&receipt("x ax\nend\n", "ax", "x")),
            ["@1 1", "-x ax", "+x x", " end"]
        );
    }

    #[test]
    fn line_breaks_added_and_taken_away_change_whole_lines() {
        // A line added, and the lines both sides share are unchanged.
        assert_eq!(
            diff(&receipt("a\nb\n", "a\n", "a\nz\n")),
            ["@1 1", " a", "+z", " b"]
        );
        // Two lines joined.
        assert_eq!(
            diff(&receipt("a\nb\nc\n", "a\nb", "ab")),
            ["@1 1", "-a", "-b", "+ab", " c"]
        );
        // A line taken out whole, and one taken out with its break, so the
        // next line joins it.
        assert_eq!(
            diff(&receipt("a\nb\nc\n", "b\n", "")),
            ["@1 1", " a", "-b", " c"]
        );
        assert_eq!(
            diff(&receipt("ab\ncd\n", "b\n", "")),
            ["@1 1", "-ab", "-cd", "+acd"]
        );
        // At the end of a file without a last line break.
        assert_eq!(diff(&receipt("a", "a", "a\nb")), ["@1 1", " a", "+b"]);
    }

    #[test]
    fn its_lines_have_no_line_breaks_and_crlf_ones_show_as_lines_too() {
        assert_eq!(
            diff(&receipt("a\r\nb\r\nc\r\n", "b\r\n", "B\r\n")),
            ["@1 1", " a", "-b", "+B", " c"]
        );
        // Only a line break changed, so the line shows removed and added.
        assert_eq!(
            diff(&receipt("a\r\nb\n", "a\r\n", "a\n")),
            ["@1 1", "-a", "+a", " b"]
        );
    }

    #[test]
    fn changes_far_apart_are_hunks_of_their_own_numbered_after_the_ones_before() {
        let mut file: String = (1..=20).map(|n| format!("line {n}\n")).collect();
        file = file.replace("line 2\n", "x\n").replace("line 18\n", "x\n");
        let receipt = receipt(&file, "x\n", "y\ny\n");
        assert_eq!(
            diff(&receipt),
            [
                "@1 1", " line 1", "-x", "+y", "+y", " line 3", " line 4", " line 5", "@15 16",
                " line 15", " line 16", " line 17", "-x", "+y", "+y", " line 19", " line 20",
            ]
        );
        assert!(!receipt.truncated);
        // Changes whose lines around them meet share a hunk.
        let near = "x\n1\n2\n3\n4\n5\n6\nx\n";
        assert_eq!(
            diff(&super::tests::receipt(near, "x", "y"))
                .iter()
                .filter(|line| line.starts_with('@'))
                .count(),
            1
        );
        // Several on one line are one change.
        assert_eq!(
            diff(&super::tests::receipt("x x x\n", "x", "y")),
            ["@1 1", "-x x x", "+y y y"]
        );
    }

    /// A result's receipt goes with its event and comes back, and a result
    /// saved before receipts loads without one.
    #[test]
    fn a_result_event_keeps_its_receipt_and_one_saved_without_loads() {
        let event = crate::AgentEvent::ToolResult {
            tool_call_id: uuid::Uuid::new_v4(),
            tool_name: "edit_file".into(),
            output: "Edited a.rs at line 1:\n1 | b".into(),
            receipt: Some(ToolReceipt::Edit(receipt("a\n", "a", "b"))),
        };
        let saved = serde_json::to_value(&event).unwrap();
        assert_eq!(
            serde_json::from_value::<crate::AgentEvent>(saved).unwrap(),
            event
        );
        let older = serde_json::json!({"ToolResult": {
            "tool_call_id": uuid::Uuid::nil(),
            "tool_name": "edit_file",
            "output": "Edited a.rs.",
        }});
        assert!(matches!(
            serde_json::from_value::<crate::AgentEvent>(older).unwrap(),
            crate::AgentEvent::ToolResult { receipt: None, .. }
        ));
    }

    /// Every receipt of a non-truncated edit, laid over the file before and
    /// after, shows exactly their lines, with what it leaves out the same
    /// on both sides.
    fn shows_the_written_lines(before: &str, old: &str, new: &str) {
        let found: Vec<usize> = before.match_indices(old).map(|(at, _)| at).collect();
        if found.is_empty() {
            return;
        }
        let receipt = receipt(before, old, new);
        let after = before.replace(old, new);
        let context = format!("{before:?} {old:?} {new:?} {receipt:?}");
        if receipt.truncated {
            return;
        }
        let (was, is): (Vec<&str>, Vec<&str>) =
            (lines_of(before).collect(), lines_of(&after).collect());
        let (mut old_at, mut new_at) = (0, 0);
        for hunk in &receipt.hunks {
            let (from_old, from_new) = (hunk.old_start - 1, hunk.new_start - 1);
            assert_eq!(was[old_at..from_old], is[new_at..from_new], "gap {context}");
            (old_at, new_at) = (from_old, from_new);
            for line in &hunk.lines {
                if line.change != LineChange::Added {
                    assert_eq!(was.get(old_at), Some(&line.text.as_str()), "{context}");
                    old_at += 1;
                }
                if line.change != LineChange::Removed {
                    assert_eq!(is.get(new_at), Some(&line.text.as_str()), "{context}");
                    new_at += 1;
                }
            }
        }
        assert_eq!(was[old_at..], is[new_at..], "tail {context}");
    }

    #[test]
    fn small_edits_of_every_shape_show_the_lines_written() {
        let alphabet = ['a', 'b', '\n'];
        for len in 1..=7u32 {
            for mut mask in 0..3usize.pow(len) {
                let mut before = String::new();
                for _ in 0..len {
                    before.push(alphabet[mask % 3]);
                    mask /= 3;
                }
                for start in 0..before.len() {
                    for end in start + 1..=before.len().min(start + 4) {
                        let old = &before[start..end];
                        for new in ["", "a", "b", "\n", "a\n", "\na", "a\nb", "\n\n"] {
                            if old != new {
                                shows_the_written_lines(&before, old, new);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn varied_edits_show_the_lines_written() {
        let mut seed: u64 = 73;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        let alphabet = ["a", "b", "\n", "\r\n", "界", "é", "c"];
        for _ in 0..10_000 {
            let before: String = (0..next() % 300 + 1)
                .map(|_| alphabet[next() % alphabet.len()])
                .collect();
            let edges: Vec<usize> = before
                .char_indices()
                .map(|(at, _)| at)
                .chain([before.len()])
                .collect();
            let from = next() % (edges.len() - 1);
            let to = (from + next() % 12 + 1).min(edges.len() - 1);
            let old = &before[edges[from]..edges[to]];
            let new: String = (0..next() % 10)
                .map(|_| alphabet[next() % alphabet.len()])
                .collect();
            if old != new {
                shows_the_written_lines(&before, old, &new);
            }
        }
    }

    /// A receipt's text is within its budget also where a line would cross
    /// it, cut at a character boundary.
    #[test]
    fn a_receipt_keeps_to_its_byte_budget_where_a_line_crosses_it() {
        let file = format!("{}x\n", "a".repeat(1_998)).repeat(40);
        let receipt = receipt(&file, "x", "y");
        let bytes: usize = receipt
            .hunks
            .iter()
            .flat_map(|hunk| &hunk.lines)
            .map(|line| line.text.len())
            .sum();
        assert!(bytes <= EDIT_RECEIPT_BYTES, "{bytes}");
        assert!(receipt.truncated);
        // Where the budget ends inside a character.
        let file = format!("{}x\n", "é".repeat(997)).repeat(40);
        let receipt = super::tests::receipt(&file, "x", "y");
        let lines: Vec<&EditLine> = receipt.hunks.iter().flat_map(|hunk| &hunk.lines).collect();
        let bytes: usize = lines.iter().map(|line| line.text.len()).sum();
        assert!(bytes <= EDIT_RECEIPT_BYTES, "{bytes}");
        assert!(lines.iter().all(|line| !line.text.is_empty()));
    }

    #[test]
    fn a_receipt_keeps_within_its_limits() {
        let file = "x\n".repeat(10_000);
        let receipt = receipt(&file, "x", "y");
        let lines: usize = receipt.hunks.iter().map(|hunk| hunk.lines.len()).sum();
        assert_eq!(lines, EDIT_RECEIPT_LINES);
        assert!(receipt.truncated);
        // A long line is cut at a character boundary.
        let long = format!("{}\n", "é".repeat(EDIT_RECEIPT_LINE_BYTES));
        let receipt = super::tests::receipt(&long, "é", "e");
        let line = &receipt.hunks[0].lines[0].text;
        assert!(line.len() <= EDIT_RECEIPT_LINE_BYTES && line.chars().all(|c| c == 'é'));
        assert!(receipt.truncated);
    }
}
