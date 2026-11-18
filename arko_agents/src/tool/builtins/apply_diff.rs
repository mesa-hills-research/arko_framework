//! `apply_diff` changes an existing file with SEARCH/REPLACE blocks, several
//! in one call.
//!
//! A block's search lines match whole lines of the file exactly, and must be
//! found once, or at least once with `replace_all`. The one adjustment is for
//! a file with CRLF line breaks, whose lines the search lines match and whose
//! line breaks the replacement lines get. Every block matches the file as it
//! was before the call, so blocks can't overlap, and their order in the diff
//! doesn't matter. A call applies every block or none. It reads the file,
//! checks every block and writes the file once, and an error names each block
//! that failed and why. Everything outside the matched lines is kept byte for
//! byte, a byte-order mark and the last line break included. A new file
//! takes the file's place in one step, with its owner, group, permissions,
//! ACL and extended attributes, and a file whose identity a new file can't
//! have, such as one with a second hard link, is refused and left as it was.
//!
//! A line of the file that is itself a marker is written in a block with a
//! backslash before it (`\=======`), as the error that meets one says. Blocks
//! in the form earlier versions took, opening with a `:start_line:` line and
//! a `-------` line, still parse, and the line number isn't used. Only both
//! lines together are that header, and a `:start_line:` line of the file is
//! written with a backslash too.
//!
//! Its output carries a receipt of what it changed
//! ([`EditReceipt`](crate::tool::EditReceipt)), as `edit_file`'s does.

use super::edit_file::{
    BOM, CONTEXT_LINES, LISTED_LINES, SHOWN_LINES, line_at, line_range, lines_at, lines_named,
    replace_all_param, starts_with_line_number, text_param,
};
use super::rewrite::RewriteError;
use crate::tool::receipt::{Replacement, edit_receipt};
use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolOutput, ToolParameterDefinition,
    ToolParameterType, ToolReceipt, ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const APPLY_DIFF_TOOL_NAME: &str = "apply_diff";
const PATH_PARAM_NAME: &str = "path";
const DIFF_PARAM_NAME: &str = "diff";
const REPLACE_ALL_PARAM_NAME: &str = "replace_all";

const SEARCH: &str = "<<<<<<< SEARCH";
const DIVIDER: &str = "=======";
const REPLACE: &str = ">>>>>>> REPLACE";
/// The line that opened a block's search lines in earlier versions, and the
/// line after it.
const START_LINE: &str = ":start_line:";
const OLD_DIVIDER: &str = "-------";

/// How a block is written, for an error about the diff's form.
const FORM: &str = "A block is <<<<<<< SEARCH, the lines to find, =======, the lines to put in their place, and >>>>>>> REPLACE, each marker on a line of its own.";
/// How a block writes a line of the file that is itself a marker.
const ESCAPE: &str = "A line of the file that is itself a marker is written with a backslash before it, as \\=======.";
/// The most bytes of a file's line an error quotes.
const QUOTED_BYTES: usize = 200;

const DESCRIPTION: &str = "Changes an existing file with one or more SEARCH/REPLACE blocks. Each block's search lines must match whole lines of the file exactly, indentation included, and be found once unless `replace_all` is set. Copy them from `read_file`'s output without the line-number prefix (`12 | `). Every block applies, or none does, and the error says which block failed and why. An empty replace part deletes the lines. Use `write_file` to create a file or replace all of it.

Example with two blocks:
<<<<<<< SEARCH
    let total = 0;
=======
    let mut total = 0;
>>>>>>> REPLACE
<<<<<<< SEARCH
fn sum(items: &[i32]) -> i32 {
=======
fn sum(items: &[i64]) -> i64 {
>>>>>>> REPLACE";

#[derive(Debug, Clone)]
pub struct ApplyDiffTool;

impl ApplyDiffTool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ApplyDiffTool {
    fn default() -> Self {
        Self::new()
    }
}

/// A block of the diff.
#[derive(Debug, PartialEq)]
struct Block {
    search: Vec<String>,
    replace: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Marker {
    Search,
    Divider,
    Replace,
}

fn marker(line: &str) -> Option<Marker> {
    match line.trim_end() {
        SEARCH => Some(Marker::Search),
        DIVIDER => Some(Marker::Divider),
        REPLACE => Some(Marker::Replace),
        _ => None,
    }
}

/// A line of a block's text. A marker, or a `:start_line:` line, with
/// backslashes before it stands for itself with one backslash fewer, and
/// any other line for itself.
fn text_line(line: &str) -> String {
    let unescaped = line.trim_start_matches('\\');
    if unescaped.len() < line.len() && (marker(unescaped).is_some() || is_start_line(unescaped)) {
        line[1..].to_string()
    } else {
        line.to_string()
    }
}

/// Whether `line` is the `:start_line:` line of the old form.
fn is_start_line(line: &str) -> bool {
    line.trim()
        .strip_prefix(START_LINE)
        .is_some_and(|number| number.trim().bytes().all(|byte| byte.is_ascii_digit()))
}

/// The blocks of `diff`, or why it doesn't parse. Text outside the blocks,
/// such as a code fence around them, is left out.
fn parse(diff: &str) -> Result<Vec<Block>, String> {
    enum Part {
        Outside,
        Search,
        Replace,
    }
    let mut blocks = Vec::new();
    let mut part = Part::Outside;
    let (mut search, mut replace) = (Vec::new(), Vec::new());
    // The lines of the block's search part so far, and its first line while
    // it may open the old form's header, `:start_line:` and `-------`.
    let (mut opening, mut start_line) = (0, None);
    for (index, line) in diff.split('\n').enumerate() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let (at, number) = (index + 1, blocks.len() + 1);
        match (&part, marker(line)) {
            (Part::Outside, Some(Marker::Search)) => {
                part = Part::Search;
                (opening, start_line) = (0, None);
            }
            (Part::Outside, Some(_)) => {
                return Err(format!(
                    "Line {at} of the diff, {}, is outside a block. {FORM} {ESCAPE}",
                    line.trim_end()
                ));
            }
            (Part::Outside, None) => {}
            (Part::Search, Some(Marker::Divider)) => {
                search.extend(start_line.take().map(text_line));
                part = Part::Replace;
            }
            (Part::Search, Some(Marker::Search)) => {
                return Err(format!(
                    "Block {number} has a second <<<<<<< SEARCH line, line {at} of the diff, before its =======. {ESCAPE}"
                ));
            }
            (Part::Search, Some(Marker::Replace)) => {
                return Err(format!(
                    "Block {number} has no ======= line before its >>>>>>> REPLACE on line {at} of the diff. {ESCAPE}"
                ));
            }
            (Part::Search, None) => {
                opening += 1;
                if opening == 1 && is_start_line(line) {
                    start_line = Some(line);
                } else if opening == 2 && start_line.is_some() && line.trim_end() == OLD_DIVIDER {
                    // (The whole header, which is left out.)
                    start_line = None;
                } else {
                    search.extend(start_line.take().map(text_line));
                    search.push(text_line(line));
                }
            }
            (Part::Replace, Some(Marker::Replace)) => {
                blocks.push(Block {
                    search: std::mem::take(&mut search),
                    replace: std::mem::take(&mut replace),
                });
                part = Part::Outside;
            }
            (Part::Replace, Some(Marker::Divider)) => {
                return Err(format!(
                    "Block {number} has a second ======= line, line {at} of the diff. {ESCAPE}"
                ));
            }
            (Part::Replace, Some(Marker::Search)) => {
                return Err(format!(
                    "Block {number} has no >>>>>>> REPLACE line before the <<<<<<< SEARCH on line {at} of the diff."
                ));
            }
            (Part::Replace, None) => replace.push(text_line(line)),
        }
    }
    let number = blocks.len() + 1;
    match part {
        Part::Search => Err(format!("Block {number} has no ======= line. {FORM}")),
        Part::Replace => Err(format!(
            "Block {number} has no >>>>>>> REPLACE line. {FORM}"
        )),
        Part::Outside if blocks.is_empty() => Err(format!("The diff has no blocks. {FORM}")),
        Part::Outside => Ok(blocks),
    }
}

/// Where a block's lines are in the file. They start at `at` and end at
/// `end`, and their line break, if they have one, ends at `next`.
#[derive(Debug, Clone, Copy)]
struct Found {
    block: usize,
    at: usize,
    end: usize,
    next: usize,
}

/// Where `needle`, whole lines joined with the file's line break, is in
/// `text` as whole lines, each place's `at`, `end` and `next` (see
/// [`Found`]). With `all`, the places that don't overlap, from the start, as
/// [`str::replace`] takes them. Without, every place, overlapping ones
/// included, so lines found twice are never taken as found once.
fn find_lines(text: &str, needle: &str, all: bool) -> Vec<(usize, usize, usize)> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = text[from..].find(needle) {
        let at = from + offset;
        let end = at + needle.len();
        let rest = &text[end..];
        let next = end
            + if rest.starts_with("\r\n") {
                2
            } else {
                usize::from(rest.starts_with('\n'))
            };
        let whole = (at == 0 || text.as_bytes()[at - 1] == b'\n')
            && (next > end || end == text.len())
            // (Past the last line break there is no line.)
            && next > at;
        if whole {
            found.push((at, end, next));
            if all {
                if next == text.len() {
                    break;
                }
                from = next;
                continue;
            }
        }
        match text[at..].find('\n') {
            Some(offset) => from = at + offset + 1,
            None => break,
        }
    }
    found
}

/// `line` for an error, in backticks and cut to [`QUOTED_BYTES`].
fn quoted(line: &str) -> String {
    if line.is_empty() {
        return "empty".to_string();
    }
    let mut cut = line.len().min(QUOTED_BYTES);
    while !line.is_char_boundary(cut) {
        cut -= 1;
    }
    let more = if cut < line.len() { "…" } else { "" };
    format!("`{}{more}`", &line[..cut])
}

/// What an error adds about search lines not found in a file of `lines`
/// (without their line breaks).
fn not_found_hint(lines: &[&str], search: &[String]) -> String {
    if starts_with_line_number(&search.join("\n")) {
        return " They start with a line number from read_file's output (`12 | `). Leave those prefixes out.".to_string();
    }
    let trimmed: Vec<&str> = search
        .iter()
        .map(|line| line.trim_matches([' ', '\t']))
        .collect();
    if trimmed.iter().any(|line| !line.is_empty()) && lines.len() >= search.len() {
        let matches = |first: usize| {
            lines[first..first + search.len()]
                .iter()
                .zip(&trimmed)
                .all(|(line, search)| line.trim_matches([' ', '\t']) == *search)
        };
        if let Some(first) = (0..=lines.len() - search.len()).find(|first| matches(*first)) {
            return format!(
                " They match {} apart from spaces and tabs at the start or end of lines. Copy those exactly as the file has them.",
                line_range(first + 1, first + search.len())
            );
        }
    }
    if search.len() > 1 {
        // The place where the most of the search lines match, from the first.
        let mut best: Option<(usize, usize)> = None;
        for (first, line) in lines.iter().enumerate() {
            if *line != search[0] {
                continue;
            }
            let matched = lines[first..]
                .iter()
                .zip(search)
                .take_while(|pair| *pair.0 == pair.1.as_str())
                .count();
            if best.is_none_or(|(_, most)| matched > most) {
                best = Some((first, matched));
            }
        }
        if let Some((first, matched)) = best {
            let start = if matched == 1 {
                format!(" Its first line matches line {}", first + 1)
            } else {
                format!(
                    " Its lines 1-{matched} match lines {}-{}",
                    first + 1,
                    first + matched
                )
            };
            return match lines.get(first + matched) {
                Some(line) => format!(
                    "{start}, and its line {} differs from line {}, which is {}.",
                    matched + 1,
                    first + matched + 1,
                    quoted(line)
                ),
                None => format!("{start}, and the file ends there."),
            };
        }
    }
    " Read the lines again and copy them exactly.".to_string()
}

/// `line 3`, `lines 3-5`, or `lines 3-5, 9 and 12-14`, with at most
/// [`LISTED_LINES`] of them.
fn ranges_named(ranges: &[(usize, usize)]) -> String {
    let mut ranges = ranges.to_vec();
    ranges.dedup();
    if let [(first, last)] = ranges[..] {
        return line_range(first, last);
    }
    let mut names: Vec<String> = ranges
        .iter()
        .take(LISTED_LINES)
        .map(|(first, last)| {
            if first == last {
                first.to_string()
            } else {
                format!("{first}-{last}")
            }
        })
        .collect();
    let more = ranges.len().saturating_sub(LISTED_LINES);
    if more > 0 {
        names.push(format!("{more} more"));
    }
    match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("lines {} and {last}", rest.join(", ")),
        _ => format!("lines {}", names.concat()),
    }
}

/// `[Line 4 not shown.]` or `[Lines 4-9 not shown.]`.
fn not_shown(first: usize, last: usize) -> String {
    if first == last {
        format!("[Line {first} not shown.]")
    } else {
        format!("[Lines {first}-{last} not shown.]")
    }
}

/// What a call that changed `text` (the file's content without its
/// byte-order mark) at `changes` returns. Each change is the first and last
/// of its new lines, and shows with [`CONTEXT_LINES`] either side, at most
/// [`SHOWN_LINES`] of them, numbered as `read_file` numbers them. A stretch
/// between them, or after the last, that isn't shown is named.
fn shown_changes(text: &str, changes: &[(usize, usize)]) -> String {
    let count = text.lines().count();
    if count == 0 {
        return String::new();
    }
    let mut wanted = BTreeSet::new();
    let mut reach = 0;
    for (first, last) in changes {
        let start = first.saturating_sub(CONTEXT_LINES).max(1).min(count);
        let end = (last + CONTEXT_LINES).min(count);
        wanted.extend(start..=end.min(start + SHOWN_LINES - 1));
        reach = reach.max(end);
    }
    let lines: BTreeMap<usize, &str> = text
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
        .filter(|(number, _)| wanted.contains(number))
        .collect();
    let mut shown = Vec::new();
    let mut previous = None;
    for (number, line) in lines {
        if let Some(previous) = previous
            && number > previous + 1
        {
            shown.push(not_shown(previous + 1, number - 1));
        }
        shown.push(format!("{number} | {line}"));
        previous = Some(number);
    }
    if let Some(previous) = previous
        && reach > previous
    {
        shown.push(not_shown(previous + 1, reach));
    }
    shown.join("\n")
}

#[async_trait]
impl Tool for ApplyDiffTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: APPLY_DIFF_TOOL_NAME.to_string(),
            description: DESCRIPTION.to_string(),
            output_based_example: Some(
                "<apply_diff>\n\
                <path>src/config.rs</path>\n\
                <diff>\n\
                <<<<<<< SEARCH\n\
                pub const VERSION: &str = \"1.0.0\";\n\
                =======\n\
                pub const VERSION: &str = \"1.1.0\";\n\
                >>>>>>> REPLACE\n\
                </diff>\n\
                </apply_diff>"
                    .to_string(),
            ),
            parameters: vec![
                ToolParameterDefinition {
                    name: PATH_PARAM_NAME.to_string(),
                    description: "The path to the file to edit (relative to the current working directory, e.g. 'src/main.rs').".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: true,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: DIFF_PARAM_NAME.to_string(),
                    description: "One or more SEARCH/REPLACE blocks.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: REPLACE_ALL_PARAM_NAME.to_string(),
                    description: "Replace every occurrence of each block's search lines (default false).".to_string(),
                    param_type: ToolParameterType::Bool,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.execute_with_output(context, params)
            .await
            .map(|output| output.text)
    }

    async fn execute_with_output(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        let path = text_param(&params, PATH_PARAM_NAME)?.trim().to_string();
        if path.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "File path cannot be empty.".to_string(),
            });
        }
        let diff = text_param(&params, DIFF_PARAM_NAME)?;
        let replace_all = replace_all_param(&params)?;
        let blocks = parse(diff).map_err(|reason| ToolError::ParameterValidation {
            param_name: DIFF_PARAM_NAME.to_string(),
            message: format!("{path} is unchanged. {reason}"),
        })?;

        let absolute_path = super::resolve_path_parameter(
            PATH_PARAM_NAME,
            &path,
            context.current_working_directory,
        )?;
        let io = |source| ToolError::Io {
            tool_name: APPLY_DIFF_TOOL_NAME.to_string(),
            source,
        };
        let metadata = match tokio::fs::metadata(&absolute_path).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::ExecutionFailed(format!(
                    "{path} doesn't exist. To create it, use write_file."
                )));
            }
            Err(error) => return Err(io(error)),
        };
        if metadata.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "{path} is a directory, not a file."
            )));
        }
        // (A pipe or a device could keep the read waiting.)
        if !metadata.is_file() {
            return Err(ToolError::ExecutionFailed(format!(
                "{path} isn't a regular file, so it can't be edited."
            )));
        }
        let bytes = tokio::fs::read(&absolute_path).await.map_err(io)?;
        let (bom, body) = match bytes.strip_prefix(BOM) {
            Some(body) => (BOM, body),
            None => (&[][..], &bytes[..]),
        };
        let text = match std::str::from_utf8(body) {
            Ok(text) if !text.contains('\0') => text,
            _ => {
                return Err(ToolError::ExecutionFailed(format!(
                    "{path} isn't a text file (UTF-8 text), so it can't be edited."
                )));
            }
        };
        if text.is_empty() {
            return Err(ToolError::ExecutionFailed(format!(
                "{path} is empty, so there is nothing to match. To give it content, use write_file."
            )));
        }

        // Every block is checked against the file as it is, and each that
        // fails gives its reason, so one error names them all.
        let line_break = if text.contains("\r\n") { "\r\n" } else { "\n" };
        let mut failures: Vec<String> = Vec::new();
        let mut found: Vec<Found> = Vec::new();
        let mut file_lines: Option<Vec<&str>> = None;
        for (index, block) in blocks.iter().enumerate() {
            let number = index + 1;
            if block.search.is_empty() {
                failures.push(format!("Block {number} has no search lines. To add lines, search for the lines next to them and repeat those in the replace part."));
                continue;
            }
            if block.search == block.replace {
                failures.push(format!("Block {number}'s replace lines are the same as its search lines, so it changes nothing."));
                continue;
            }
            let places = find_lines(text, &block.search.join(line_break), replace_all);
            match places.as_slice() {
                [] => {
                    let lines = file_lines.get_or_insert_with(|| text.lines().collect());
                    failures.push(format!(
                        "Block {number}'s search lines weren't found.{}",
                        not_found_hint(lines, &block.search)
                    ));
                }
                [_, _, ..] if !replace_all => {
                    let ats: Vec<usize> = places.iter().map(|(at, _, _)| *at).collect();
                    failures.push(format!(
                        "Block {number}'s search lines were found {} times, at {}. Add lines around them to make them unique, or set replace_all to change every occurrence.",
                        places.len(),
                        lines_named(&lines_at(text, &ats))
                    ));
                }
                _ => found.extend(places.iter().map(|(at, end, next)| Found {
                    block: index,
                    at: *at,
                    end: *end,
                    next: *next,
                })),
            }
        }
        found.sort_by_key(|place| place.at);
        // The block whose place reaches furthest so far, and where it ends.
        let mut reach: Option<(usize, usize)> = None;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for place in &found {
            if let Some((block, end)) = reach
                && place.at < end
                && block != place.block
            {
                let pair = (block.min(place.block), block.max(place.block));
                if !pairs.contains(&pair) {
                    pairs.push(pair);
                    let last = line_at(text, end.min(place.next) - 1);
                    failures.push(format!(
                        "Blocks {} and {} both change {}. Make them one block.",
                        pair.0 + 1,
                        pair.1 + 1,
                        line_range(line_at(text, place.at), last)
                    ));
                }
            }
            if reach.is_none_or(|(_, end)| place.next > end) {
                reach = Some((place.block, place.next));
            }
        }
        if !failures.is_empty() {
            let lead = if blocks.len() == 1 {
                format!("{path} is unchanged.")
            } else {
                format!(
                    "None of the {} blocks applied, so {path} is unchanged.",
                    blocks.len()
                )
            };
            return Err(ToolError::ExecutionFailed(format!(
                "{lead}\n{}",
                failures.join("\n")
            )));
        }

        let joined: Vec<String> = blocks
            .iter()
            .map(|block| block.replace.join(line_break))
            .collect();
        let mut edited = String::with_capacity(text.len());
        let mut replaced = Vec::with_capacity(found.len());
        let mut rest = 0;
        for place in &found {
            edited.push_str(&text[rest..place.at]);
            let start = edited.len();
            if !blocks[place.block].replace.is_empty() {
                edited.push_str(&joined[place.block]);
                edited.push_str(&text[place.end..place.next]);
            }
            replaced.push(Replacement {
                at: place.at,
                old_len: place.next - place.at,
                start,
                new_len: edited.len() - start,
            });
            rest = place.next;
        }
        edited.push_str(&text[rest..]);

        let mut contents = Vec::with_capacity(bom.len() + edited.len());
        contents.extend_from_slice(bom);
        contents.extend_from_slice(edited.as_bytes());
        let target = absolute_path.clone();
        tokio::task::spawn_blocking(move || super::rewrite::rewrite_file(&target, &contents))
            .await
            .map_err(|error| ToolError::ExecutionFailed(format!("The write stopped: {error}")))?
            .map_err(|error| match error {
                RewriteError::Kept(why) => ToolError::ExecutionFailed(format!(
                    "{path} is unchanged. {why} To change it, use write_file or the shell, which write it in place."
                )),
                RewriteError::Io(error) => io(error),
            })?;
        let receipt = ToolReceipt::Edit(edit_receipt(text, &edited, &replaced));

        let starts: Vec<usize> = replaced
            .iter()
            .map(|replacement| replacement.start)
            .collect();
        let first_lines = lines_at(&edited, &starts);
        if replace_all {
            return Ok(ToolOutput::text(format!(
                "Replaced {} in {path}, at {}.",
                if starts.len() == 1 {
                    "1 occurrence".to_string()
                } else {
                    format!("{} occurrences", starts.len())
                },
                lines_named(&first_lines)
            ))
            .with_receipt(receipt));
        }
        let count = edited.lines().count().max(1);
        let changes: Vec<(usize, usize)> = found
            .iter()
            .zip(first_lines)
            .map(|(place, first)| {
                let first = first.min(count);
                let lines = blocks[place.block].replace.len().max(1);
                (first, first + lines - 1)
            })
            .collect();
        let shown = shown_changes(&edited, &changes);
        Ok(ToolOutput::text(if shown.is_empty() {
            format!("Edited {path}. It is now empty.")
        } else {
            format!("Edited {path} at {}:\n{shown}", ranges_named(&changes))
        })
        .with_receipt(receipt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Workspace {
        root: std::path::PathBuf,
    }

    impl Workspace {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("arko-apply-diff-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn write(&self, name: &str, bytes: &[u8]) {
            std::fs::write(self.root.join(name), bytes).unwrap();
        }

        fn read(&self, name: &str) -> Vec<u8> {
            std::fs::read(self.root.join(name)).unwrap()
        }

        fn text(&self, name: &str) -> String {
            String::from_utf8(self.read(name)).unwrap()
        }

        /// Calls `apply_diff` with `params`, and returns its output.
        async fn call(&self, params: Value) -> Result<ToolOutput, ToolError> {
            let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &self.root);
            ApplyDiffTool
                .execute_with_output(&context, serde_json::from_value(params).unwrap())
                .await
        }

        /// Applies `diff` to `name`, and returns the result's text.
        async fn apply(&self, name: &str, diff: &str) -> String {
            self.call(json!({"path": name, "diff": diff}))
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .text
        }

        /// Calls `apply_diff` on `name` and returns its error, checking that
        /// the file is byte for byte as it was, and that nothing else
        /// appeared in the folder.
        async fn refused(&self, name: &str, params: Value) -> String {
            let before = std::fs::read(self.root.join(name)).ok();
            let entries = self.entries();
            let error = match self.call(params).await {
                Ok(output) => panic!("applied: {}", output.text),
                Err(error) => error.to_string(),
            };
            assert_eq!(std::fs::read(self.root.join(name)).ok(), before, "{error}");
            assert_eq!(self.entries(), entries, "{error}");
            error
        }

        /// [`Workspace::refused`] for `diff` applied to `name`.
        async fn refused_diff(&self, name: &str, diff: &str) -> String {
            self.refused(name, json!({"path": name, "diff": diff}))
                .await
        }

        fn entries(&self) -> Vec<std::ffi::OsString> {
            let mut names: Vec<_> = std::fs::read_dir(&self.root)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Workspace {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// A block replacing `search` with `replace`, each lines joined with
    /// `\n` (empty for none).
    fn block(search: &str, replace: &str) -> String {
        let part = |text: &str| {
            if text.is_empty() {
                String::new()
            } else {
                format!("{text}\n")
            }
        };
        format!(
            "{SEARCH}\n{}{DIVIDER}\n{}{REPLACE}\n",
            part(search),
            part(replace)
        )
    }

    /// The block numbers an error names as failing, from its `Block N` and
    /// `Blocks N and M`.
    fn blocks_named(error: &str) -> Vec<usize> {
        let words: Vec<&str> = error.split_whitespace().collect();
        let number = |at: usize| {
            words
                .get(at)
                .and_then(|word| word.trim_end_matches("'s").parse::<usize>().ok())
        };
        let mut named = Vec::new();
        for (at, word) in words.iter().enumerate() {
            match *word {
                "Block" => named.extend(number(at + 1)),
                "Blocks" if words.get(at + 2) == Some(&"and") => {
                    named.extend(number(at + 1));
                    named.extend(number(at + 3));
                }
                _ => {}
            }
        }
        named
    }

    const FILE: &str = "fn one() {\n    1\n}\n\nfn two() {\n    2\n}\n\nfn three() {\n    3\n}\n";

    #[tokio::test]
    async fn several_blocks_apply_in_one_call_in_any_order() {
        let ws = Workspace::new();
        ws.write("a.rs", FILE.as_bytes());
        let diff = [
            block("    3", "    30"),
            block("fn one() {", "fn uno() {"),
            block("    2", "    20\n    21"),
        ]
        .concat();
        let result = ws.apply("a.rs", &diff).await;
        assert_eq!(
            ws.text("a.rs"),
            "fn uno() {\n    1\n}\n\nfn two() {\n    20\n    21\n}\n\nfn three() {\n    30\n}\n"
        );
        // It shows each block's new lines, numbered as read_file numbers them.
        for line in ["1 | fn uno() {", "6 |     20", "7 |     21", "11 |     30"] {
            assert!(result.lines().any(|shown| shown == line), "{result}");
        }
    }

    #[tokio::test]
    async fn a_failing_block_in_the_middle_leaves_the_file_byte_identical() {
        let ws = Workspace::new();
        ws.write("a.rs", FILE.as_bytes());
        let diff = [
            block("fn one() {", "fn uno() {"),
            block("    4", "    40"),
            block("    3", "    30"),
        ]
        .concat();
        let error = ws.refused_diff("a.rs", &diff).await;
        assert_eq!(blocks_named(&error), [2], "{error}");
    }

    /// Every block applies, or none does, whatever the reason a block
    /// fails. Each failing block is named, and only those.
    #[tokio::test]
    async fn every_block_applies_or_none_does() {
        let ws = Workspace::new();
        ws.write("a.rs", FILE.as_bytes());
        let good = block("fn one() {", "fn uno() {");
        for (bad, why) in [
            (block("    4", "    40"), "not found"),
            (block("}", "};"), "found three times"),
            (block("", "    0"), "no search lines"),
            (block("    2", "    2"), "changes nothing"),
        ] {
            let error = ws.refused_diff("a.rs", &[good.clone(), bad].concat()).await;
            assert_eq!(blocks_named(&error), [2], "{why}: {error}");
        }
        // Several failing blocks are named in one error.
        let diff = [
            block("    4", "x"),
            good.clone(),
            block("}", "x"),
            block("    3", "x"),
        ]
        .concat();
        let error = ws.refused_diff("a.rs", &diff).await;
        assert_eq!(blocks_named(&error), [1, 3], "{error}");
        // A diff that doesn't parse changes nothing either.
        let unfinished = format!("{good}{SEARCH}\n    3\n{DIVIDER}\n    30\n");
        let error = ws.refused_diff("a.rs", &unfinished).await;
        assert_eq!(blocks_named(&error), [2], "{error}");
        assert_eq!(ws.text("a.rs"), FILE);
    }

    #[tokio::test]
    async fn overlapping_blocks_are_refused_and_named() {
        let ws = Workspace::new();
        ws.write("a.rs", FILE.as_bytes());
        // Two blocks sharing a line.
        let diff = [
            block("fn two() {\n    2", "fn two() {\n    20"),
            block("fn one() {", "fn uno() {"),
            block("    2\n}", "    22\n}"),
        ]
        .concat();
        let error = ws.refused_diff("a.rs", &diff).await;
        assert_eq!(blocks_named(&error), [1, 3], "{error}");
        assert!(error.contains("line 6"), "{error}");
        // The same lines twice.
        let diff = [block("    1", "    10"), block("    1", "    11")].concat();
        let error = ws.refused_diff("a.rs", &diff).await;
        assert_eq!(blocks_named(&error), [1, 2], "{error}");
        // Blocks on lines next to each other don't overlap, a deletion
        // included.
        let diff = [block("fn one() {", ""), block("    1", "    10")].concat();
        ws.apply("a.rs", &diff).await;
        assert!(ws.text("a.rs").starts_with("    10\n}\n\nfn two"));
    }

    /// Every block matches the file as it was before the call, not what
    /// another block of the call wrote.
    #[tokio::test]
    async fn blocks_match_the_file_as_it_was_before_the_call() {
        let ws = Workspace::new();
        ws.write("a.txt", b"a\nb\n");
        let diff = [block("a", "c"), block("c", "d")].concat();
        let error = ws.refused_diff("a.txt", &diff).await;
        assert_eq!(blocks_named(&error), [2], "{error}");
        // Swapping two lines in one call.
        let diff = [block("a", "b"), block("b", "a")].concat();
        ws.apply("a.txt", &diff).await;
        assert_eq!(ws.text("a.txt"), "b\na\n");
    }

    #[tokio::test]
    async fn search_lines_match_whole_lines_exactly() {
        let ws = Workspace::new();
        ws.write("a.txt", b"x = 1\nxx = 1\n  y = 2\nlast");
        // Part of a line, other indentation and trailing spaces don't match.
        for search in ["x = ", "y = 2", "  y = 2 ", "= 1\nxx"] {
            let error = ws.refused_diff("a.txt", &block(search, "z")).await;
            assert_eq!(blocks_named(&error), [1], "{search:?}: {error}");
        }
        // A whole line matches even where its text is part of another line.
        ws.apply("a.txt", &block("x = 1", "x = 3")).await;
        assert_eq!(ws.text("a.txt"), "x = 3\nxx = 1\n  y = 2\nlast");
        // The last line, without a line break, is a whole line too, and
        // keeps having none.
        ws.apply("a.txt", &block("last", "end\nmore")).await;
        assert_eq!(ws.text("a.txt"), "x = 3\nxx = 1\n  y = 2\nend\nmore");
    }

    #[tokio::test]
    async fn an_empty_replace_part_deletes_the_lines() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\ntwo\nthree\n");
        ws.apply("a.txt", &block("two", "")).await;
        assert_eq!(ws.text("a.txt"), "one\nthree\n");
        // An empty line in its place leaves an empty line.
        ws.apply("a.txt", &format!("{SEARCH}\nthree\n{DIVIDER}\n\n{REPLACE}"))
            .await;
        assert_eq!(ws.text("a.txt"), "one\n\n");
        let result = ws.apply("a.txt", &block("one\n", "")).await;
        assert_eq!(ws.text("a.txt"), "");
        assert!(!result.contains(" | "), "{result}");
    }

    #[tokio::test]
    async fn crlf_files_match_their_lines_and_keep_their_line_breaks() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\r\ntwo\r\nthree\r\nfour");
        let diff = [block("one\ntwo", "1\n2\n2.5"), block("four", "4\n5")].concat();
        let output = ws
            .call(json!({"path": "a.txt", "diff": diff}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"1\r\n2\r\n2.5\r\nthree\r\n4\r\n5");
        // A diff written with CRLF line breaks reads the same.
        ws.apply("a.txt", &block("three", "3").replace('\n', "\r\n"))
            .await;
        assert_eq!(ws.read("a.txt"), b"1\r\n2\r\n2.5\r\n3\r\n4\r\n5");
        // The receipt shows the lines without their line breaks.
        let Some(ToolReceipt::Edit(receipt)) = output.receipt else {
            panic!("no receipt");
        };
        assert!(
            receipt
                .hunks
                .iter()
                .flat_map(|hunk| &hunk.lines)
                .all(|line| !line.text.contains('\r'))
        );
        // An LF file gets no CRLF.
        ws.write("b.txt", b"one\ntwo\n");
        ws.apply("b.txt", &block("one", "1\n0").replace('\n', "\r\n"))
            .await;
        assert_eq!(ws.read("b.txt"), b"1\n0\ntwo\n");
    }

    #[tokio::test]
    async fn a_byte_order_mark_is_kept() {
        let ws = Workspace::new();
        ws.write("a.txt", b"\xEF\xBB\xBFname = old\nrest\n");
        ws.apply("a.txt", &block("name = old", "name = new")).await;
        assert_eq!(ws.read("a.txt"), b"\xEF\xBB\xBFname = new\nrest\n");
    }

    #[tokio::test]
    async fn lines_found_more_than_once_need_replace_all() {
        let ws = Workspace::new();
        ws.write("a.txt", b"x\ny\nx\ny\nx\n");
        let error = ws.refused_diff("a.txt", &block("x", "z")).await;
        assert!(error.contains("1, 3 and 5"), "{error}");
        // Places that overlap count too.
        let error = ws.refused_diff("a.txt", &block("x\ny\nx", "z")).await;
        assert!(error.contains("1 and 3"), "{error}");
        let output = ws
            .call(json!({"path": "a.txt", "replace_all": true, "diff": block("x", "z") + &block("y", "w")}))
            .await
            .unwrap();
        assert_eq!(ws.text("a.txt"), "z\nw\nz\nw\nz\n");
        assert!(output.text.contains("5 occurrences"), "{}", output.text);
        // Overlapping places are taken from the start, as str::replace
        // takes them.
        ws.write("b.txt", b"a\na\na\nb\n");
        ws.call(json!({"path": "b.txt", "replace_all": "true", "diff": block("a\na", "c")}))
            .await
            .unwrap();
        assert_eq!(ws.text("b.txt"), "c\na\nb\n");
        // A block replace_all doesn't find fails too.
        let error = ws
            .refused(
                "b.txt",
                json!({"path": "b.txt", "replace_all": true, "diff": block("c", "d") + &block("q", "r")}),
            )
            .await;
        assert_eq!(blocks_named(&error), [2], "{error}");
    }

    #[tokio::test]
    async fn a_miss_is_explained() {
        let ws = Workspace::new();
        ws.write(
            "a.rs",
            b"fn main() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n}\n",
        );
        // Copied with read_file's line numbers.
        let error = ws
            .refused_diff("a.rs", &block("2 |     let x = 1;", "x"))
            .await;
        assert!(error.contains("`12 | `"), "{error}");
        // Right apart from indentation, so the lines are named.
        let error = ws
            .refused_diff("a.rs", &block("  let x = 1;\n  let y = 2;", "x"))
            .await;
        assert!(error.contains("lines 2-3"), "{error}");
        // The first lines match, so the line that differs is named and
        // quoted.
        let error = ws
            .refused_diff(
                "a.rs",
                &block("fn main() {\n    let x = 1;\n    let y = 5;", "x"),
            )
            .await;
        assert!(error.contains("lines 1-2"), "{error}");
        assert!(
            error.contains("line 3") && error.contains("`    let y = 2;`"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_marker_line_of_the_file_is_written_with_a_backslash() {
        let ws = Workspace::new();
        ws.write(
            "conflict.txt",
            b"<<<<<<< ours\na\n=======\nb\n>>>>>>> theirs\n",
        );
        // Unescaped, the block's form is unclear, and nothing is written.
        let diff = block("<<<<<<< ours\na\n=======\nb\n>>>>>>> theirs", "b");
        let error = ws.refused_diff("conflict.txt", &diff).await;
        assert!(error.contains("\\======="), "{error}");
        let diff = block("<<<<<<< ours\na\n\\=======\nb\n>>>>>>> theirs", "b");
        ws.apply("conflict.txt", &diff).await;
        assert_eq!(ws.text("conflict.txt"), "b\n");
        // In the replace part, and with a backslash of its own.
        ws.apply(
            "conflict.txt",
            &block("b", "\\>>>>>>> REPLACE\n\\\\<<<<<<< SEARCH"),
        )
        .await;
        assert_eq!(
            ws.text("conflict.txt"),
            ">>>>>>> REPLACE\n\\<<<<<<< SEARCH\n"
        );
    }

    #[tokio::test]
    async fn blocks_in_the_old_form_still_apply() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\ntwo\nthree\n");
        // The line number isn't used, so a wrong one is no matter.
        let diff = format!(
            "{SEARCH}\n:start_line:9\n-------\ntwo\n{DIVIDER}\n2\n{REPLACE}\n\n{SEARCH}\n:start_line: 1\n-------\none\n{DIVIDER}\n1\n{REPLACE}"
        );
        ws.apply("a.txt", &diff).await;
        assert_eq!(ws.text("a.txt"), "1\n2\nthree\n");
        // A `-------` line without `:start_line:` before it is a line of
        // the file.
        ws.write("b.md", b"Title\n-------\n");
        ws.apply("b.md", &block("-------", "=====")).await;
        assert_eq!(ws.text("b.md"), "Title\n=====\n");
        // Nor is a `:start_line:` line without `-------` after it.
        ws.write("c.txt", b":start_line:42\na\nb\n");
        ws.apply("c.txt", &block(":start_line:42\na", "x")).await;
        assert_eq!(ws.text("c.txt"), "x\nb\n");
        ws.write("d.txt", b"a\n:start_line:42\n");
        ws.apply("d.txt", &block(":start_line:42", "x")).await;
        assert_eq!(ws.text("d.txt"), "a\nx\n");
        // Both header lines, as lines of the file, are written with a
        // backslash.
        ws.write("e.txt", b":start_line:1\n-------\nx\n");
        ws.apply("e.txt", &block("\\:start_line:1\n-------\nx", "y"))
            .await;
        assert_eq!(ws.text("e.txt"), "y\n");
    }

    #[tokio::test]
    async fn text_around_the_blocks_is_left_out() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\n");
        ws.apply("a.txt", &format!("```diff\n{}```\n", block("one", "1")))
            .await;
        assert_eq!(ws.text("a.txt"), "1\n");
    }

    #[tokio::test]
    async fn a_diff_whose_form_is_wrong_is_refused() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\n=======\n");
        for diff in [
            String::new(),
            "one\n".to_string(),
            format!("{SEARCH}\none\n{REPLACE}\n"),
            format!("{SEARCH}\none\n{DIVIDER}\n1\n"),
            format!("{SEARCH}\none\n{DIVIDER}\n1\n{DIVIDER}\n{REPLACE}\n"),
            format!("{SEARCH}\none\n{SEARCH}\n"),
            format!("{}{DIVIDER}\n", block("one", "1")),
        ] {
            ws.refused_diff("a.txt", &diff).await;
        }
    }

    /// A call's output carries one receipt with the changes of every block.
    #[tokio::test]
    async fn the_receipt_covers_every_block() {
        let ws = Workspace::new();
        let file: String = (1..=30).map(|n| format!("line {n}\n")).collect();
        ws.write("a.txt", file.as_bytes());
        let diff = [block("line 25", "line 25!"), block("line 3", "")].concat();
        let output = ws
            .call(json!({"path": "a.txt", "diff": diff}))
            .await
            .unwrap();
        let Some(ToolReceipt::Edit(receipt)) = output.receipt else {
            panic!("no receipt");
        };
        let changed: Vec<(crate::tool::LineChange, &str)> = receipt
            .hunks
            .iter()
            .flat_map(|hunk| &hunk.lines)
            .filter(|line| line.change != crate::tool::LineChange::Kept)
            .map(|line| (line.change, line.text.as_str()))
            .collect();
        use crate::tool::LineChange::{Added, Removed};
        assert_eq!(
            changed,
            [
                (Removed, "line 3"),
                (Removed, "line 25"),
                (Added, "line 25!")
            ]
        );
        assert_eq!(receipt.hunks.len(), 2);
        assert_eq!(
            (receipt.hunks[1].old_start, receipt.hunks[1].new_start),
            (22, 21)
        );
    }

    #[tokio::test]
    async fn other_files_are_refused_and_left_alone() {
        let ws = Workspace::new();
        let diff = block("a", "b");
        let error = ws.refused_diff("missing.txt", &diff).await;
        assert!(error.contains("write_file"), "{error}");
        ws.write("empty.txt", b"");
        let error = ws.refused_diff("empty.txt", &diff).await;
        assert!(error.contains("write_file"), "{error}");
        std::fs::create_dir_all(ws.root.join("dir")).unwrap();
        ws.refused_diff("dir", &diff).await;
        ws.write("image.png", b"\x89PNG\r\n\x1a\n\xff\xfe");
        ws.refused_diff("image.png", &diff).await;
        ws.write("a.txt", b"a\n");
        for (params, missing) in [
            (json!({"diff": diff}), "path"),
            (json!({"path": "a.txt"}), "diff"),
        ] {
            let error = ws.refused("a.txt", params).await;
            assert!(error.contains(missing), "{error}");
        }
        ws.refused(
            "a.txt",
            json!({"path": "a.txt", "diff": diff, "replace_all": "maybe"}),
        )
        .await;
    }

    /// A pipe is refused without being read, which would wait for a writer.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_pipe_is_refused_without_waiting_for_it() {
        let ws = Workspace::new();
        let made = std::process::Command::new("mkfifo")
            .arg(ws.root.join("pipe"))
            .status()
            .unwrap();
        assert!(made.success());
        let call = ws.call(json!({"path": "pipe", "diff": block("a", "b")}));
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), call)
            .await
            .expect("the call waited");
        assert!(result.is_err());
    }

    /// The file is replaced whole, keeping its permissions, and a symbolic
    /// link stays a link to the file it names.
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[tokio::test]
    async fn the_write_keeps_permissions_and_links() {
        use std::os::unix::fs::PermissionsExt;
        let ws = Workspace::new();
        ws.write("run.sh", b"echo one\n");
        let path = ws.root.join("run.sh");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o750)).unwrap();
        ws.apply("run.sh", &block("echo one", "echo two")).await;
        assert_eq!(ws.text("run.sh"), "echo two\n");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o750);

        std::os::unix::fs::symlink("run.sh", ws.root.join("link.sh")).unwrap();
        ws.apply("link.sh", &block("echo two", "echo three")).await;
        let link = std::fs::symlink_metadata(ws.root.join("link.sh")).unwrap();
        assert!(link.file_type().is_symlink());
        assert_eq!(ws.text("run.sh"), "echo three\n");
        assert_eq!(
            ws.entries(),
            [
                std::ffi::OsString::from("link.sh"),
                std::ffi::OsString::from("run.sh")
            ]
        );

        // A file the process can't write is refused, as a write in place
        // would be.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        if std::fs::OpenOptions::new().write(true).open(&path).is_err() {
            ws.refused_diff("run.sh", &block("echo three", "echo four"))
                .await;
        }
    }

    /// Who can read and write the file stays as it was. A file of another
    /// group keeps its group and permissions, and an ACL is kept. A file
    /// with a second hard link, which a new file in its place would leave
    /// behind, is refused.
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[tokio::test]
    async fn the_write_keeps_who_can_read_the_file() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let ws = Workspace::new();
        ws.write("shared.txt", b"one\n");
        let path = ws.root.join("shared.txt");
        let groups = std::process::Command::new("id").arg("-G").output().unwrap();
        let other = String::from_utf8(groups.stdout)
            .unwrap()
            .split_whitespace()
            .nth(1)
            .map(|group| group.parse::<u32>().unwrap());
        match other {
            Some(group) => {
                std::os::unix::fs::chown(&path, None, Some(group)).unwrap();
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
                ws.apply("shared.txt", &block("one", "two")).await;
                let metadata = std::fs::metadata(&path).unwrap();
                assert_eq!((metadata.gid(), metadata.mode() & 0o7777), (group, 0o640));
                assert_eq!(ws.text("shared.txt"), "two\n");
            }
            None => eprintln!("skipped the other group, since the process has one group"),
        }

        ws.write("a.txt", b"one\n");
        std::fs::hard_link(ws.root.join("a.txt"), ws.root.join("b.txt")).unwrap();
        ws.refused_diff("a.txt", &block("one", "two")).await;
        assert_eq!(ws.text("b.txt"), "one\n");
        assert_eq!(std::fs::metadata(ws.root.join("a.txt")).unwrap().nlink(), 2);

        ws.write("acl.txt", b"one\n");
        let acl = ws.root.join("acl.txt");
        let set = std::process::Command::new("setfacl")
            .args(["-m", "u:nobody:r"])
            .arg(&acl)
            .output();
        match set {
            Ok(set) if set.status.success() => {
                ws.apply("acl.txt", &block("one", "two")).await;
                let listed = std::process::Command::new("getfacl")
                    .arg(&acl)
                    .output()
                    .unwrap();
                let listed = String::from_utf8(listed.stdout).unwrap();
                assert!(listed.contains("user:nobody:r--"), "{listed}");
                assert_eq!(ws.text("acl.txt"), "two\n");
            }
            _ => eprintln!(
                "skipped the ACL, since setfacl isn't here or the file system has no ACLs"
            ),
        }
    }

    #[test]
    fn its_path_is_checked_as_write_file_s() {
        let definition = ApplyDiffTool.definition();
        let paths: Vec<&str> = definition
            .parameters
            .iter()
            .filter(|parameter| parameter.is_path)
            .map(|parameter| parameter.name.as_str())
            .collect();
        assert_eq!(paths, ["path"]);
        assert!(!definition.parallel_safe);
        assert_eq!(
            ApplyDiffTool.recovery_mode(),
            crate::tool::ToolRecovery::InDoubt
        );
    }

    /// The examples in the description and the output-based example are
    /// diffs that parse.
    #[test]
    fn its_examples_parse() {
        let definition = ApplyDiffTool.definition();
        let example = definition.description.split_once(SEARCH).unwrap().1;
        assert_eq!(parse(&format!("{SEARCH}{example}")).unwrap().len(), 2);
        let output_based = definition.output_based_example.unwrap();
        let diff = output_based
            .split_once("<diff>")
            .unwrap()
            .1
            .split_once("</diff>")
            .unwrap()
            .0;
        assert_eq!(parse(diff).unwrap().len(), 1);
    }
}
