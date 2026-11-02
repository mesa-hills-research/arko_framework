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
        let old: Vec<&str> = lines_of(&before[start..end]).collect();
        let new: Vec<&str> = lines_of(&after[after_start..after_end]).collect();
        let old_line = line;
        let new_line = (line as isize + shift) as usize;
        shift += new.len() as isize - old.len() as isize;
        built.change(Change {
            old_line,
            new_line,
            start,
            end,
            old,
            new,
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

/// A change's whole lines before and after, and where they are.
struct Change<'a> {
    old_line: usize,
    new_line: usize,
    /// Its bytes in the text before.
    start: usize,
    end: usize,
    old: Vec<&'a str>,
    new: Vec<&'a str>,
}

/// A receipt as it's made.
struct Builder<'a> {
    before: &'a str,
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
    fn change(&mut self, mut change: Change<'a>) {
        // The lines both sides share at either end are unchanged, unless
        // nothing tells the two apart once their line breaks are off, when
        // the change shows as it is.
        let lead = change
            .old
            .iter()
            .zip(&change.new)
            .take_while(|(old, new)| old == new)
            .count();
        let tail = change.old[lead..]
            .iter()
            .rev()
            .zip(change.new[lead..].iter().rev())
            .take_while(|(old, new)| old == new)
            .count();
        let (removed, added) = (
            change.old.len() - lead - tail,
            change.new.len() - lead - tail,
        );
        if removed > 0 || added > 0 {
            change.start = advance(self.before, change.start, lead);
            change.end = back(self.before, change.end, tail);
            change.old_line += lead;
            change.new_line += lead;
            change.old = change.old[lead..lead + removed].to_vec();
            change.new = change.new[lead..lead + added].to_vec();
        }
        let gap = self
            .hunk
            .as_ref()
            .map(|(_, next, _)| change.old_line.saturating_sub(*next));
        match (self.hunk.take(), gap) {
            (Some((mut hunk, _, from)), Some(gap)) if gap <= 2 * EDIT_RECEIPT_CONTEXT => {
                let between: Vec<&str> = lines_of(&self.before[from..change.start]).collect();
                for line in between {
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
        for line in &change.old {
            self.push(&mut hunk, LineChange::Removed, line);
        }
        for line in &change.new {
            self.push(&mut hunk, LineChange::Added, line);
        }
        self.hunk = Some((hunk, change.old_line + change.old.len(), change.end));
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
        let lines: Vec<&str> = lines_of(&self.before[from..change.start]).collect();
        for line in lines {
            self.push(&mut hunk, LineChange::Kept, line);
        }
        self.hunk = Some((hunk, 0, 0));
    }

    /// `hunk` with up to [`EDIT_RECEIPT_CONTEXT`] unchanged lines from byte
    /// `from` on.
    fn close(&mut self, mut hunk: EditHunk, from: usize) -> EditHunk {
        let lines: Vec<&str> = lines_of(&self.before[from..])
            .take(EDIT_RECEIPT_CONTEXT)
            .collect();
        for line in lines {
            self.push(&mut hunk, LineChange::Kept, line);
        }
        hunk
    }

    /// Adds `text` to `hunk`, cut to [`EDIT_RECEIPT_LINE_BYTES`], unless the
    /// receipt is full.
    fn push(&mut self, hunk: &mut EditHunk, change: LineChange, text: &str) {
        if self.full() {
            self.receipt.truncated = true;
            return;
        }
        let mut cut = text.len().min(EDIT_RECEIPT_LINE_BYTES);
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut < text.len() {
            self.receipt.truncated = true;
        }
        self.lines += 1;
        self.bytes += cut;
        hunk.lines.push(EditLine {
            change,
            text: text[..cut].to_string(),
        });
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
