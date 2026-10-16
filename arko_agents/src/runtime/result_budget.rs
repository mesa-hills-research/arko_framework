//! The overflow guard and the opt-in caps on tool results
//! ([`ToolResultLimits`](crate::agent_config::ToolResultLimits)): token
//! estimates, which results to shorten, head-and-tail shortening, and the
//! notices the model reads.
//!
//! Tokens are estimated from bytes; no tokenizer is in this crate's
//! dependencies. Two estimates are used:
//!
//! - **Whether to act** uses a lower bound: the latest request's reported
//!   input tokens, plus every byte added since at
//!   [`LEAST_BYTES_PER_TOKEN`] (10). No content measured packs that many
//!   bytes into a token (o200k, the GPT-5 and GPT-6 tokenizer: code 3.6 to
//!   4.8, English 3.8 to 4.3, padded tables 8.6; Claude's tokenizer counts
//!   more tokens for the same text, not fewer), so the guard acts only when
//!   the next request would certainly be over the model's input limit.
//!   When it is unsure it sends the request; an underestimate is caught by
//!   the provider's overflow error and the bounded recovery.
//! - **How much to keep** uses the calibrated estimate: the bytes per token
//!   of the task's own requests (each request's size in bytes over the
//!   input tokens the provider reported for it). Content without bytes here
//!   (images, reasoning a provider replays) counts in the reported tokens
//!   only, which lowers the ratio, so a preview errs toward fewer bytes and
//!   the request toward fitting. The ratio is clamped to
//!   [`MIN_BYTES_PER_TOKEN`]..[`MAX_BYTES_PER_TOKEN`]; before a task's first
//!   request it is [`DEFAULT_BYTES_PER_TOKEN`].

use arko_ai as ai;

use super::event::ShortenReason;

/// Bytes per token before a task's own usage calibrates it: about right
/// for code and English with current tokenizers.
pub const DEFAULT_BYTES_PER_TOKEN: f64 = 3.0;
/// The calibrated ratio's bounds: dense text (JSON, non-Latin scripts) has
/// fewer bytes per token, prose more.
pub(crate) const MIN_BYTES_PER_TOKEN: f64 = 2.0;
pub(crate) const MAX_BYTES_PER_TOKEN: f64 = 4.0;
/// The bytes per token of the lower bound that decides whether the guard
/// acts: more than any content measured (see the module docs).
pub(crate) const LEAST_BYTES_PER_TOKEN: f64 = 10.0;

/// The default margin is this share of the input limit...
const MARGIN_SHARE: f64 = 0.02;
/// ...and at least this many tokens.
const MIN_MARGIN_TOKENS: usize = 4_000;
/// A preview is this share of the input limit...
const PREVIEW_SHARE: f64 = 0.02;
/// ...between these bounds.
const MIN_PREVIEW_TOKENS: usize = 4_000;
const MAX_PREVIEW_TOKENS: usize = 16_000;
/// A preview smaller than this isn't worth keeping: the result is replaced
/// by its notice alone. A result no larger than this beyond its notice is
/// never shortened by the guard.
pub(crate) const MIN_KEPT_TOKENS: usize = 500;
/// Room for what a notice says beyond its text without lines kept: the
/// lines shown, and the marker between head and tail.
const NOTICE_SLACK_BYTES: usize = 80;
/// The share of a preview taken from the start of the result.
const HEAD_SHARE: f64 = 0.8;

/// Tokens in `bytes` bytes at `ratio` bytes per token, rounded up.
pub(crate) fn tokens(bytes: usize, ratio: f64) -> usize {
    (bytes as f64 / ratio).ceil() as usize
}

/// The fewest tokens `bytes` bytes of text take: at
/// [`LEAST_BYTES_PER_TOKEN`], rounded down.
pub(crate) fn least_tokens(bytes: usize) -> usize {
    (bytes as f64 / LEAST_BYTES_PER_TOKEN) as usize
}

/// Bytes in `tokens` tokens at `ratio` bytes per token.
fn bytes(tokens: usize, ratio: f64) -> usize {
    (tokens as f64 * ratio) as usize
}

/// The ratio a request of `request_bytes` bytes that the provider counted
/// as `input_tokens` tokens shows, clamped; `None` when it says nothing.
pub(crate) fn calibrated_ratio(request_bytes: usize, input_tokens: usize) -> Option<f64> {
    (request_bytes > 0 && input_tokens > 0).then(|| {
        (request_bytes as f64 / input_tokens as f64).clamp(MIN_BYTES_PER_TOKEN, MAX_BYTES_PER_TOKEN)
    })
}

/// The default margin for an input limit of `limit` tokens: 2%, at least
/// 4,000 tokens.
pub(crate) fn default_margin(limit: usize) -> usize {
    ((limit as f64 * MARGIN_SHARE) as usize).max(MIN_MARGIN_TOKENS)
}

/// The size of a shortened result's preview for an input limit of `limit`
/// tokens: 2%, between 4,000 and 16,000 tokens.
pub(crate) fn preview_tokens(limit: usize) -> usize {
    ((limit as f64 * PREVIEW_SHARE) as usize).clamp(MIN_PREVIEW_TOKENS, MAX_PREVIEW_TOKENS)
}

/// The bytes a message adds to a request, as far as they are known here:
/// its text, its tool calls' names and arguments, and its tool results'
/// output. Images and replayed reasoning are not counted (see the module
/// docs).
pub(crate) fn message_bytes(message: &ai::Message) -> usize {
    let content: usize = message
        .content
        .iter()
        .map(|content| match content {
            ai::Content::Text(text) => text.len(),
            ai::Content::ToolResult(result) => result.output.len() + result.name.len(),
            _ => 0,
        })
        .sum();
    let calls: usize = message
        .tool_calls
        .iter()
        .map(|call| call.name.len() + call.arguments.to_string().len())
        .sum();
    content + calls
}

/// The bytes of a request: the system message, the messages, and the tool
/// definitions ([`ai::RequestSizes::text_bytes`]).
pub(crate) fn request_bytes(
    conversation: &ai::Conversation,
    generation_config: &ai::GenerationConfig,
) -> usize {
    request_bytes_of(conversation, &conversation.messages, generation_config)
}

/// [`request_bytes`] of a request with `conversation`'s system message and
/// only `messages`.
pub(crate) fn request_bytes_of(
    conversation: &ai::Conversation,
    messages: &[ai::Message],
    generation_config: &ai::GenerationConfig,
) -> usize {
    ai::RequestSizes::of_messages(conversation, messages, generation_config.offered_tools())
        .text_bytes()
}

/// The bytes per token a request of `request_bytes` bytes of text (see
/// [`ai::RequestSizes::text_bytes`]) that the provider counted as
/// `input_tokens` input tokens shows, clamped to 2..4 (dense text such as
/// JSON has fewer bytes per token, prose more; images and replayed
/// reasoning count in the tokens only, which lowers it). `None` when either
/// is zero. The ratio arko's runtime estimates requests by, and what
/// [`AgentEvent::ModelRequestStarted`](crate::runtime::event::AgentEvent::ModelRequestStarted)
/// reports; [`DEFAULT_BYTES_PER_TOKEN`] before a task's first answered
/// request.
pub fn calibrated_bytes_per_token(request_bytes: usize, input_tokens: usize) -> Option<f64> {
    calibrated_ratio(request_bytes, input_tokens)
}

/// What happens to one result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cut {
    /// Tokens of the result to keep (head and tail); 0 keeps none.
    pub keep: usize,
    pub reason: ShortenReason,
}

/// The room the overflow guard plans the turn's results in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Room {
    /// Tokens the results may take, at the calibrated estimate, for the
    /// request to fit the input limit less the margin: how much shortened
    /// results keep.
    pub available: usize,
    /// Tokens left under the input limit by the lower bound of the rest of
    /// the request: the results certainly overflow it when their own lower
    /// bound is more.
    pub certain: usize,
    /// A result's lower bound, in tokens per calibrated token (the
    /// calibrated ratio over [`LEAST_BYTES_PER_TOKEN`]).
    pub least_share: f64,
    /// The size of a preview.
    pub preview: usize,
}

/// Which results to shorten, given each result's estimated tokens
/// (`sizes`), the tokens its notice would add (`notices`), its opt-in cap
/// (`caps`), and with the overflow guard on, the [`Room`]. A result over
/// its cap is cut to the cap. Then the guard acts only while the results
/// certainly overflow (their lower bound is over [`Room::certain`]): the
/// largest are cut first, each to the preview; if that leaves the overflow
/// certain, the room the calibrated estimate leaves is shared between the
/// results above [`MIN_KEPT_TOKENS`], a share below it keeping the notice
/// alone. A result is never replaced by something larger (notices count at
/// their size), and results no larger than [`MIN_KEPT_TOKENS`] beyond their
/// notice are never cut by the guard. When the overflow isn't certain,
/// nothing is cut: the request is sent, and the provider's overflow error
/// and the bounded recovery catch an underestimate.
pub(crate) fn plan(
    sizes: &[usize],
    notices: &[usize],
    caps: &[Option<usize>],
    room: Option<Room>,
) -> Vec<Option<Cut>> {
    let mut cuts: Vec<Option<Cut>> = sizes
        .iter()
        .zip(caps)
        .map(|(size, cap)| {
            cap.filter(|cap| size > cap).map(|cap| Cut {
                keep: cap,
                reason: ShortenReason::ResultCap,
            })
        })
        .collect();
    let Some(room) = room else {
        return cuts;
    };
    // What a result takes: all of it, or what it keeps and its notice.
    let kept = |cuts: &[Option<Cut>], index: usize| {
        cuts[index].map_or(sizes[index], |cut| cut.keep + notices[index])
    };
    let certain = |cuts: &[Option<Cut>]| {
        (0..sizes.len())
            .map(|index| (kept(cuts, index) as f64 * room.least_share) as usize)
            .sum::<usize>()
            > room.certain
    };
    if !certain(&cuts) {
        return cuts;
    }
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|index| std::cmp::Reverse(kept(&cuts, *index)));
    let overflow = |keep: usize| {
        Some(Cut {
            keep,
            reason: ShortenReason::Overflow,
        })
    };
    // The largest first, each to the preview, while the overflow is
    // certain.
    for &index in &order {
        if !certain(&cuts) {
            return cuts;
        }
        if kept(&cuts, index) > room.preview + notices[index] {
            cuts[index] = overflow(room.preview);
        }
    }
    if !certain(&cuts) {
        return cuts;
    }
    // The previews leave it certain (the context is nearly full): the room
    // the calibrated estimate leaves is shared between the results above
    // the smallest worth keeping, smallest first, each keeping all it has
    // while its share allows.
    let candidates: Vec<usize> = order
        .iter()
        .rev()
        .copied()
        .filter(|&index| kept(&cuts, index) > MIN_KEPT_TOKENS + notices[index])
        .collect();
    let others: usize = (0..sizes.len())
        .filter(|index| !candidates.contains(index))
        .map(|index| kept(&cuts, index))
        .sum();
    let mut budget = room.available.saturating_sub(others);
    let mut remaining = candidates.len();
    for &index in &candidates {
        let share = budget / remaining;
        remaining -= 1;
        let current = kept(&cuts, index);
        if current <= share {
            budget -= current;
            continue;
        }
        let keep = share.saturating_sub(notices[index]);
        let keep = if keep >= MIN_KEPT_TOKENS { keep } else { 0 };
        cuts[index] = overflow(keep);
        budget = budget.saturating_sub(keep + notices[index]);
    }
    cuts
}

/// What a notice says about a result; see [`notice`].
pub(crate) struct NoticeFacts<'a> {
    pub tool_name: &'a str,
    /// The file a `read_file` call read.
    pub path: Option<&'a str>,
    pub reason: ShortenReason,
    /// The result's estimated tokens.
    pub tokens: usize,
    /// The result's size in bytes and lines.
    pub bytes: usize,
    pub lines: usize,
    /// What was kept: the first and last lines (both 0 for none).
    pub head_lines: usize,
    pub tail_lines: usize,
    /// The cap a [`ShortenReason::ResultCap`] result was over.
    pub cap: Option<usize>,
    /// The agent is offered arko's terminal tools.
    pub terminal: bool,
    /// The agent's `read_file` takes line ranges.
    pub line_ranges: bool,
}

/// A size in bytes, as people read it.
fn describe_bytes(bytes: usize) -> String {
    match bytes {
        bytes if bytes >= 1 << 20 => format!("{:.1} MB", bytes as f64 / (1 << 20) as f64),
        bytes if bytes >= 1 << 10 => format!("{:.0} KB", bytes as f64 / (1 << 10) as f64),
        bytes => format!("{bytes} bytes"),
    }
}

/// A number with thousands separators.
fn grouped(number: usize) -> String {
    let digits = number.to_string();
    let mut grouped = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// The notice after a shortened result, the only part of the guard the
/// model sees: how large the result was, what of it is shown, and how to
/// get the rest. With a terminal, it names commands (for a `read_file`
/// result, on its file); without one, `read_file`'s line ranges when they
/// are on, or else to ask for less. Read ranges are never mentioned unless
/// the agent's `read_file` takes them.
pub(crate) fn notice(facts: &NoticeFacts) -> String {
    let size = format!(
        "about {} tokens ({}, {} lines)",
        grouped(facts.tokens),
        describe_bytes(facts.bytes),
        grouped(facts.lines)
    );
    let why = match facts.reason {
        ShortenReason::ResultCap => format!(
            "this result is {size}, over the host's limit of {} tokens per tool result",
            grouped(facts.cap.unwrap_or_default())
        ),
        ShortenReason::Overflow | ShortenReason::OverflowRecovery => {
            format!("this result is {size}, too large for your context")
        }
    };
    let shown = if facts.head_lines + facts.tail_lines == 0 {
        "None of it is shown.".to_string()
    } else if facts.tail_lines == 0 {
        format!("Shown: the first {} lines.", grouped(facts.head_lines))
    } else {
        format!(
            "Shown: the first {} and the last {} lines.",
            grouped(facts.head_lines),
            grouped(facts.tail_lines)
        )
    };
    let read_file = facts.tool_name == crate::tool::builtins::READ_FILE_TOOL_NAME;
    let advice = match (facts.terminal, facts.path.filter(|_| read_file)) {
        (true, Some(path)) => {
            let path = shell_quote(path);
            format!(
                "Look at the parts you need with the terminal, for example: wc -l {path}; grep -n 'pattern' {path}; head -n 100 {path}; sed -n '5000,5400p' {path}; or a short script."
            )
        }
        (true, None) => "Ask for less (a narrower query, fewer results), or, where the data is in a file or a command's output, look at the parts you need with the terminal: wc -l, head, grep -n, sed -n 'a,bp', or a short script.".to_string(),
        (false, Some(path)) if facts.line_ranges => format!(
            "Read the other parts with read_file's start_line and end_line, for example {{\"path\": {}, \"start_line\": 5000, \"end_line\": 5400}}.",
            serde_json::Value::from(path)
        ),
        (false, _) => "The rest was left out because the output was too large. Ask for less: a smaller file, a narrower query, or fewer results.".to_string(),
    };
    format!("[Shortened: {why}. {shown} {advice}]")
}

/// The tokens `facts`' notice adds to a shortened result, at `ratio`, with
/// room for the lines it names and the marker between head and tail.
pub(crate) fn notice_tokens(facts: &NoticeFacts, ratio: f64) -> usize {
    tokens(notice(facts).len() + NOTICE_SLACK_BYTES, ratio)
}

/// `path` quoted for a POSIX shell when it needs quoting.
fn shell_quote(path: &str) -> String {
    let plain = !path.is_empty()
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-+,:@%".contains(c));
    if plain {
        path.to_string()
    } else {
        format!("'{}'", path.replace('\'', r"'\''"))
    }
}

/// A result shortened to its head and tail.
pub(crate) struct Shortened {
    pub text: String,
    pub head_lines: usize,
    pub tail_lines: usize,
}

/// The largest index at or below `index` that is a character boundary.
fn floor_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// The smallest index at or above `index` that is a character boundary.
fn ceil_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// `text` cut to about `keep_bytes` bytes: [`HEAD_SHARE`] from its start
/// and the rest from its end, on line boundaries where a line fits, with a
/// marker between them saying how much was left out. `keep_bytes` of 0
/// keeps nothing.
pub(crate) fn head_and_tail(text: &str, keep_bytes: usize) -> Shortened {
    if keep_bytes == 0 || text.is_empty() {
        return Shortened {
            text: String::new(),
            head_lines: 0,
            tail_lines: 0,
        };
    }
    if keep_bytes >= text.len() {
        return Shortened {
            text: text.to_string(),
            head_lines: text.lines().count(),
            tail_lines: 0,
        };
    }
    let head_budget = (keep_bytes as f64 * HEAD_SHARE) as usize;
    let tail_budget = keep_bytes - head_budget;
    // The head ends after the last whole line that fits, or mid-line when
    // not even one does.
    let head_end = floor_boundary(text, head_budget);
    let head_end = text[..head_end]
        .rfind('\n')
        .map_or(head_end, |newline| newline + 1);
    // The tail starts at the first whole line that fits.
    let tail_start = ceil_boundary(text, text.len().saturating_sub(tail_budget)).max(head_end);
    let tail_start = match text[tail_start..].find('\n') {
        Some(newline) if tail_start > 0 && !text[..tail_start].ends_with('\n') => {
            let start = tail_start + newline + 1;
            if start < text.len() {
                start
            } else {
                tail_start
            }
        }
        _ => tail_start,
    };
    let head = &text[..head_end];
    let tail = &text[tail_start..];
    let line_count = |part: &str| part.lines().count();
    let omitted = &text[head_end..tail_start];
    let mut shortened = head.to_string();
    if !shortened.is_empty() && !shortened.ends_with('\n') {
        shortened.push('\n');
    }
    shortened.push_str(&format!(
        "[... {} lines ({}) left out ...]\n",
        grouped(line_count(omitted)),
        describe_bytes(omitted.len())
    ));
    shortened.push_str(tail);
    Shortened {
        text: shortened,
        head_lines: line_count(head),
        tail_lines: line_count(tail),
    }
}

/// `output` shortened to about `keep` tokens at `ratio`, followed by its
/// notice (`facts`, with the lines kept filled in); returns the new text
/// and the estimated tokens kept.
pub(crate) fn shorten(
    output: &str,
    keep: usize,
    ratio: f64,
    mut facts: NoticeFacts,
) -> (String, usize) {
    let shortened = head_and_tail(output, bytes(keep, ratio));
    facts.head_lines = shortened.head_lines;
    facts.tail_lines = shortened.tail_lines;
    let notice = notice(&facts);
    let kept = tokens(shortened.text.len(), ratio);
    let text = if shortened.text.is_empty() {
        notice
    } else {
        format!("{}\n\n{notice}", shortened.text.trim_end())
    };
    (text, kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_margins_and_previews() {
        assert_eq!(tokens(3_000, 3.0), 1_000);
        assert_eq!(tokens(3_001, 3.0), 1_001);
        assert_eq!(calibrated_ratio(350_000, 100_000), Some(3.5));
        assert_eq!(
            calibrated_ratio(900_000, 100_000),
            Some(MAX_BYTES_PER_TOKEN)
        );
        assert_eq!(
            calibrated_ratio(100_000, 100_000),
            Some(MIN_BYTES_PER_TOKEN)
        );
        assert_eq!(calibrated_ratio(0, 10), None);
        assert_eq!(least_tokens(99), 9);
        assert_eq!(default_margin(272_000), 5_440);
        assert_eq!(default_margin(100_000), 4_000);
        assert_eq!(preview_tokens(272_000), 5_440);
        assert_eq!(preview_tokens(100_000), 4_000);
        assert_eq!(preview_tokens(1_000_000), 16_000);
    }

    fn keeps(cuts: &[Option<Cut>]) -> Vec<Option<usize>> {
        cuts.iter().map(|cut| cut.map(|cut| cut.keep)).collect()
    }

    /// A room for results at 4 bytes per token (a lower bound of 0.4 of
    /// that), with a preview of 5,000 tokens.
    fn room(available: usize, certain: usize) -> Option<Room> {
        Some(Room {
            available,
            certain,
            least_share: 4.0 / LEAST_BYTES_PER_TOKEN,
            preview: 5_000,
        })
    }

    /// The plan with notices of 100 tokens and no caps.
    fn guard(sizes: &[usize], room: Option<Room>) -> Vec<Option<Cut>> {
        plan(
            sizes,
            &vec![100; sizes.len()],
            &vec![None; sizes.len()],
            room,
        )
    }

    #[test]
    fn results_that_fit_are_left_alone() {
        assert_eq!(
            keeps(&guard(&[1_000, 2_000], room(3_000, 3_000))),
            [None, None]
        );
        // Over the calibrated estimate's room, but not certainly over the
        // limit: sent as they are.
        assert_eq!(
            keeps(&guard(&[5_000, 4_000], room(3_000, 4_000))),
            [None, None]
        );
        assert_eq!(keeps(&guard(&[100_000], room(80_000, 90_000))), [None]);
        // Without the guard nothing is cut.
        assert_eq!(keeps(&guard(&[9_000_000], None)), [None]);
    }

    #[test]
    fn the_largest_result_is_cut_first_to_the_preview() {
        let cuts = guard(&[1_000, 2_000_000, 3_000], room(50_000, 60_000));
        assert_eq!(keeps(&cuts), [None, Some(5_000), None]);
        assert_eq!(cuts[1].unwrap().reason, ShortenReason::Overflow);
        // Two large parallel results: both, largest first, while the
        // overflow is certain.
        let cuts = guard(&[600_000, 2_000_000, 100], room(50_000, 100_000));
        assert_eq!(keeps(&cuts), [Some(5_000), Some(5_000), None]);
        let cuts = guard(&[40_000, 2_000_000, 100], room(30_000, 60_000));
        assert_eq!(keeps(&cuts), [None, Some(5_000), None]);
    }

    #[test]
    fn a_nearly_full_context_shares_what_is_left_and_keeps_small_results() {
        // 6,000 tokens left for two large results: 3,000 each, notices
        // included.
        let cuts = guard(&[90_000, 80_000], room(6_000, 1_000));
        assert_eq!(keeps(&cuts), [Some(2_900), Some(2_900)]);
        // A result under the preview shares the room too.
        let cuts = guard(&[4_000, 100], room(3_000, 1_000));
        assert_eq!(keeps(&cuts), [Some(2_800), None]);
        // Nothing left: the large result keeps its notice; a small one is
        // never replaced, by a notice larger than it or at all.
        let cuts = guard(&[90_000, 10, 550], room(0, 0));
        assert_eq!(keeps(&cuts), [Some(0), None, None]);
        let cuts = plan(&[150], &[200], &[None], room(0, 0));
        assert_eq!(keeps(&cuts), [None]);
        // Too little to share: the large ones keep their notices.
        let cuts = guard(&[90_000, 80_000, 200], room(700, 0));
        assert_eq!(keeps(&cuts), [Some(0), Some(0), None]);
    }

    #[test]
    fn caps_apply_with_or_without_the_guard() {
        let cuts = plan(
            &[30_000, 500],
            &[100, 100],
            &[Some(20_000), Some(20_000)],
            None,
        );
        assert_eq!(keeps(&cuts), [Some(20_000), None]);
        assert_eq!(cuts[0].unwrap().reason, ShortenReason::ResultCap);
        // A capped result the guard must cut further.
        let cuts = plan(&[300_000], &[100], &[Some(20_000)], room(10_000, 5_000));
        assert_eq!(keeps(&cuts), [Some(5_000)]);
        assert_eq!(cuts[0].unwrap().reason, ShortenReason::Overflow);
    }

    #[test]
    fn the_head_and_tail_keep_whole_lines() {
        let text: String = (1..=1_000).map(|n| format!("{n} | line {n}\n")).collect();
        let shortened = head_and_tail(&text, 1_000);
        assert!(shortened.text.starts_with("1 | line 1\n"));
        assert!(shortened.text.ends_with("1000 | line 1000\n"));
        assert!(shortened.text.contains("lines ("));
        assert!(shortened.head_lines > 30 && shortened.tail_lines > 5);
        assert!(shortened.text.len() < 1_200, "{}", shortened.text.len());
        for line in shortened
            .text
            .lines()
            .filter(|line| !line.starts_with("[..."))
        {
            let (number, rest) = line.split_once(" | ").unwrap();
            assert_eq!(rest, format!("line {number}"));
        }
        // One long line (minified code) is cut mid-line, on a character
        // boundary: the head's end (803 bytes for 1,004) and the tail's
        // start (201 bytes from the end) both fall inside a 2-byte
        // character.
        let minified = "é".repeat(10_000);
        for keep in [1_001, 1_004] {
            let shortened = head_and_tail(&minified, keep);
            assert!(shortened.text.len() <= keep + 100);
            assert!(shortened.text.starts_with('é'));
            assert!(shortened.text.ends_with('é'));
            let head = shortened.text.split('\n').next().unwrap();
            assert_eq!(head.len(), if keep == 1_004 { 802 } else { 800 });
        }
        // Nothing kept.
        assert!(head_and_tail(&text, 0).text.is_empty());
    }

    fn facts(terminal: bool, line_ranges: bool, tool: &str) -> NoticeFacts<'static> {
        NoticeFacts {
            tool_name: if tool == "read_file" {
                "read_file"
            } else {
                "mcp__db__query"
            },
            path: Some("data/big file.py"),
            reason: ShortenReason::Overflow,
            tokens: 2_100_000,
            bytes: 8_600_000,
            lines: 81_431,
            head_lines: 410,
            tail_lines: 80,
            cap: None,
            terminal,
            line_ranges,
        }
    }

    #[test]
    fn notices_point_to_the_terminal_or_ask_for_less() {
        let with_terminal = notice(&facts(true, false, "read_file"));
        assert!(
            with_terminal.starts_with("[Shortened: this result is about 2,100,000 tokens (8.2 MB, 81,431 lines), too large for your context. Shown: the first 410 and the last 80 lines."),
            "{with_terminal}"
        );
        for command in [
            "wc -l 'data/big file.py'",
            "grep -n",
            "head -n 100",
            "sed -n '5000,5400p'",
            "short script",
        ] {
            assert!(with_terminal.contains(command), "{with_terminal}");
        }
        assert!(!with_terminal.contains("start_line"));
        let other_tool = notice(&facts(true, false, "other"));
        assert!(
            other_tool.contains("wc -l, head, grep -n, sed -n 'a,bp'"),
            "{other_tool}"
        );
        let without = notice(&facts(false, false, "read_file"));
        assert!(without.contains("too large. Ask for less"), "{without}");
        for word in ["terminal", "start_line", "sed"] {
            assert!(!without.contains(word), "{without}");
        }
        // Ranges are named only when read_file takes them.
        let ranges = notice(&facts(false, true, "read_file"));
        assert!(
            ranges
                .contains(r#"{"path": "data/big file.py", "start_line": 5000, "end_line": 5400}"#),
            "{ranges}"
        );
        assert!(!notice(&facts(false, true, "other")).contains("start_line"));
        let capped = notice(&NoticeFacts {
            reason: ShortenReason::ResultCap,
            cap: Some(20_000),
            head_lines: 0,
            tail_lines: 0,
            ..facts(false, false, "other")
        });
        assert!(
            capped.contains(
                "over the host's limit of 20,000 tokens per tool result. None of it is shown."
            ),
            "{capped}"
        );
    }
}
