//! `edit_file`: replaces one exact piece of text in a file, or every
//! occurrence of it.
//!
//! The match is exact, on the file's whole content: no fuzzy matching and
//! no whitespace tolerance. The one adjustment is for a file with CRLF line
//! breaks, where the `\n` line breaks of `old_string` and `new_string` are
//! written as `\r\n` before matching, since `read_file` shows the file's
//! lines without them. Everything outside the match is kept byte for byte,
//! a byte-order mark and the last line break included. A call applies in
//! full or not at all: it reads the file, checks the match and writes the
//! file once, and any error leaves the file as it was.

use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

pub const EDIT_FILE_TOOL_NAME: &str = "edit_file";
const PATH_PARAM_NAME: &str = "path";
const OLD_STRING_PARAM_NAME: &str = "old_string";
const NEW_STRING_PARAM_NAME: &str = "new_string";
const REPLACE_ALL_PARAM_NAME: &str = "replace_all";

const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
/// The lines around a change that its result shows, on each side.
const CONTEXT_LINES: usize = 2;
/// The most lines a result shows of the file.
const SHOWN_LINES: usize = 20;
/// The most line numbers a result or an error lists.
const LISTED_LINES: usize = 20;

#[derive(Debug, Clone)]
pub struct EditFileTool;

impl EditFileTool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new()
    }
}

/// A required text parameter.
fn text_param<'a>(params: &'a HashMap<String, Value>, name: &str) -> Result<&'a str, ToolError> {
    params
        .get(name)
        .ok_or_else(|| ToolError::MissingParameter {
            param_name: name.to_string(),
        })?
        .as_str()
        .ok_or_else(|| ToolError::ParameterValidation {
            param_name: name.to_string(),
            message: format!("Parameter '{name}' must be a string."),
        })
}

/// `replace_all`: a boolean, or the text `true` or `false`. Absent or null
/// is `false`.
fn replace_all_param(params: &HashMap<String, Value>) -> Result<bool, ToolError> {
    match params.get(REPLACE_ALL_PARAM_NAME) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(false),
        Some(Value::String(text)) => {
            crate::tool::definition::parse_bool_text(text).ok_or_else(|| {
                ToolError::ParameterValidation {
                    param_name: REPLACE_ALL_PARAM_NAME.to_string(),
                    message: "Parameter 'replace_all' must be true or false.".to_string(),
                }
            })
        }
        Some(_) => Err(ToolError::ParameterValidation {
            param_name: REPLACE_ALL_PARAM_NAME.to_string(),
            message: "Parameter 'replace_all' must be true or false.".to_string(),
        }),
    }
}

/// `text` with its `\n` line breaks written as `\r\n` (and those already
/// written so left as they are).
fn with_crlf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\n', "\r\n")
}

/// The line, numbered from 1 as `read_file` numbers them, that byte
/// `offset` of `text` is on.
fn line_at(text: &str, offset: usize) -> usize {
    1 + text.as_bytes()[..offset]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
}

/// `3`, `3 and 17`, `3, 17 and 40`, with at most [`LISTED_LINES`] of them
/// (`3, 17, ... and 12 more`). `lines` is sorted, and a line listed twice
/// is named once.
fn line_list(lines: &[usize]) -> String {
    let mut lines = lines.to_vec();
    lines.dedup();
    let more = lines.len().saturating_sub(LISTED_LINES);
    let mut names: Vec<String> = lines
        .iter()
        .take(LISTED_LINES)
        .map(ToString::to_string)
        .collect();
    if more > 0 {
        names.push(format!("{more} more"));
    }
    match names.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// `line 3` or `lines 3 and 17` (see [`line_list`]).
fn lines_named(lines: &[usize]) -> String {
    let mut distinct = lines.to_vec();
    distinct.dedup();
    let word = if distinct.len() == 1 { "line" } else { "lines" };
    format!("{word} {}", line_list(lines))
}

/// `line 3` or `lines 3-5`.
fn line_range(first: usize, last: usize) -> String {
    if first == last {
        format!("line {first}")
    } else {
        format!("lines {first}-{last}")
    }
}

/// Whether `text` starts with a line number as `read_file` shows it
/// (`12 | `), copied from its output.
fn starts_with_line_number(text: &str) -> bool {
    let first = text
        .trim_start_matches(['\r', '\n'])
        .lines()
        .next()
        .unwrap_or_default()
        .trim_start();
    let digits = first.bytes().take_while(u8::is_ascii_digit).count();
    let rest = &first[digits..];
    digits > 0 && (rest == " |" || rest.starts_with(" | "))
}

/// `text` with the spaces and tabs at the start of each line removed.
fn unindented(text: &str) -> String {
    text.split('\n')
        .map(|line| line.trim_start_matches([' ', '\t']))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The lines where `old` is in `text` apart from the indentation of its
/// lines, when it is.
fn match_apart_from_indentation(text: &str, old: &str) -> Option<(usize, usize)> {
    let old = unindented(old);
    if old.trim().is_empty() {
        return None;
    }
    let text = unindented(text);
    let at = text.find(&old)?;
    let first = line_at(&text, at);
    let last = line_at(&text, at + old.trim_end_matches(['\r', '\n']).len());
    Some((first, last.max(first)))
}

/// What a call that changed `text` (the file's content without its
/// byte-order mark) at lines `first` to `last` returns: those lines, and
/// [`CONTEXT_LINES`] either side, numbered as `read_file` numbers them, at
/// most [`SHOWN_LINES`] of them.
fn shown_lines(text: &str, first: usize, last: usize) -> String {
    let normalized = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let start = first.saturating_sub(CONTEXT_LINES).max(1).min(lines.len());
    let end = (last + CONTEXT_LINES).min(lines.len());
    let shown_end = end.min(start + SHOWN_LINES - 1);
    let mut shown: Vec<String> = (start..=shown_end)
        .map(|number| format!("{number} | {}", lines[number - 1]))
        .collect();
    if shown_end < end {
        shown.push(format!("[Lines {}-{end} not shown.]", shown_end + 1));
    }
    shown.join("\n")
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: EDIT_FILE_TOOL_NAME.to_string(),
            description: "Replaces text in an existing file. `old_string` must match the file exactly, including indentation and line breaks, and appear exactly once. Set `replace_all` to change every occurrence. Copy it from `read_file`'s output without the line-number prefix (`12 | `), with enough surrounding lines to make it unique. If it isn't found, or is found more than once, the file is left unchanged and the error says which. For several changes, call it once for each. Calls in one response run in order. Use `write_file` to create a file or replace all of it.".to_string(),
            output_based_example: Some(
                "<edit_file>\n\
                <path>src/config.rs</path>\n\
                <old_string>pub const VERSION: &str = \"1.0.0\";</old_string>\n\
                <new_string>pub const VERSION: &str = \"1.1.0\";</new_string>\n\
                </edit_file>"
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
                    name: OLD_STRING_PARAM_NAME.to_string(),
                    description: "The exact text to replace.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: NEW_STRING_PARAM_NAME.to_string(),
                    description: "The text to put in its place (empty to delete it).".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: REPLACE_ALL_PARAM_NAME.to_string(),
                    description: "Replace every occurrence of old_string (default false).".to_string(),
                    param_type: ToolParameterType::Bool,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let path = text_param(&params, PATH_PARAM_NAME)?.trim().to_string();
        if path.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "File path cannot be empty.".to_string(),
            });
        }
        let old = text_param(&params, OLD_STRING_PARAM_NAME)?;
        let new = text_param(&params, NEW_STRING_PARAM_NAME)?;
        let replace_all = replace_all_param(&params)?;
        if old.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: OLD_STRING_PARAM_NAME.to_string(),
                message:
                    "old_string is empty. To create a file or replace all of it, use write_file."
                        .to_string(),
            });
        }
        if old == new {
            return Err(ToolError::ParameterValidation {
                param_name: NEW_STRING_PARAM_NAME.to_string(),
                message: "new_string is the same as old_string, so there is nothing to change."
                    .to_string(),
            });
        }

        let absolute_path = super::resolve_path_parameter(
            PATH_PARAM_NAME,
            &path,
            context.current_working_directory,
        )?;
        let io = |source| ToolError::Io {
            tool_name: EDIT_FILE_TOOL_NAME.to_string(),
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

        let (old, new) = if text.contains("\r\n") {
            (with_crlf(old), with_crlf(new))
        } else {
            (old.to_string(), new.to_string())
        };
        let found: Vec<usize> = text.match_indices(&old).map(|(at, _)| at).collect();
        if found.is_empty() {
            let mut message =
                format!("old_string wasn't found in {path}, so the file is unchanged.");
            let mut hinted = false;
            if starts_with_line_number(&old) {
                message.push_str(" It starts with a line number from read_file's output (`12 | `). Leave those prefixes out.");
                hinted = true;
            }
            if let Some((first, last)) = match_apart_from_indentation(text, &old) {
                message.push_str(&format!(
                    " It matches {} apart from indentation. Copy the indentation exactly as the file has it.",
                    line_range(first, last)
                ));
                hinted = true;
            }
            if !hinted {
                message.push_str(" Read the lines again and copy them exactly.");
            }
            return Err(ToolError::ExecutionFailed(message));
        }
        let found_lines: Vec<usize> = found.iter().map(|at| line_at(text, *at)).collect();
        if found.len() > 1 && !replace_all {
            return Err(ToolError::ExecutionFailed(format!(
                "old_string was found {} times in {path}, at {}, so the file is unchanged. Add surrounding lines to old_string to make it unique, or set replace_all to change every occurrence.",
                found.len(),
                lines_named(&found_lines)
            )));
        }

        // The new content, and where each replacement starts in it.
        let mut edited = String::with_capacity(text.len() + found.len() * new.len());
        let mut starts = Vec::with_capacity(found.len());
        let mut rest = 0;
        for at in &found {
            edited.push_str(&text[rest..*at]);
            starts.push(edited.len());
            edited.push_str(&new);
            rest = at + old.len();
        }
        edited.push_str(&text[rest..]);

        let mut contents = Vec::with_capacity(bom.len() + edited.len());
        contents.extend_from_slice(bom);
        contents.extend_from_slice(edited.as_bytes());
        tokio::fs::write(&absolute_path, contents)
            .await
            .map_err(io)?;

        if replace_all {
            let lines: Vec<usize> = starts.iter().map(|at| line_at(&edited, *at)).collect();
            return Ok(format!(
                "Replaced {} in {path}, at {}.",
                if starts.len() == 1 {
                    "1 occurrence".to_string()
                } else {
                    format!("{} occurrences", starts.len())
                },
                lines_named(&lines)
            ));
        }
        let start = starts[0];
        let first = line_at(&edited, start);
        let last = if new.is_empty() {
            first
        } else {
            line_at(&edited, start + new.len() - 1)
        };
        let shown = shown_lines(&edited, first, last);
        Ok(if shown.is_empty() {
            format!("Edited {path}. It is now empty.")
        } else {
            format!("Edited {path} at {}:\n{shown}", line_range(first, last))
        })
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
                std::env::temp_dir().join(format!("arko-edit-file-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn write(&self, name: &str, bytes: &[u8]) {
            std::fs::write(self.root.join(name), bytes).unwrap();
        }

        fn read(&self, name: &str) -> Vec<u8> {
            std::fs::read(self.root.join(name)).unwrap()
        }

        /// Calls `edit_file` with `params`.
        async fn edit(&self, params: Value) -> Result<String, ToolError> {
            let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &self.root);
            EditFileTool
                .execute(&context, serde_json::from_value(params).unwrap())
                .await
        }

        /// Calls `edit_file` on `name` and returns its error, checking that
        /// the file is unchanged.
        async fn refused(&self, name: &str, params: Value) -> String {
            let before = std::fs::read(self.root.join(name)).ok();
            let error = self.edit(params).await.unwrap_err().to_string();
            assert_eq!(std::fs::read(self.root.join(name)).ok(), before, "{error}");
            error
        }
    }

    impl Drop for Workspace {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[tokio::test]
    async fn replaces_one_exact_match_and_keeps_everything_else() {
        let ws = Workspace::new();
        ws.write(
            "a.rs",
            b"fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n",
        );
        let result = ws
            .edit(json!({"path": "a.rs", "old_string": "    two();", "new_string": "    two();\n    three();"}))
            .await
            .unwrap();
        assert_eq!(
            String::from_utf8(ws.read("a.rs")).unwrap(),
            "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n    three();\n}\n"
        );
        assert_eq!(
            result,
            "Edited a.rs at lines 6-7:\n4 | \n5 | fn b() {\n6 |     two();\n7 |     three();\n8 | }"
        );
        // Without a last line break, none is added.
        ws.write("b.txt", b"x = 1");
        let result = ws
            .edit(json!({"path": "b.txt", "old_string": "1", "new_string": "2"}))
            .await
            .unwrap();
        assert_eq!(ws.read("b.txt"), b"x = 2");
        assert_eq!(result, "Edited b.txt at line 1:\n1 | x = 2");
    }

    #[tokio::test]
    async fn an_empty_new_string_deletes_the_text() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\ntwo\nthree\n");
        let result = ws
            .edit(json!({"path": "a.txt", "old_string": "two\n", "new_string": ""}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"one\nthree\n");
        assert_eq!(result, "Edited a.txt at line 2:\n1 | one\n2 | three");
        let result = ws
            .edit(json!({"path": "a.txt", "old_string": "one\nthree\n", "new_string": ""}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"");
        assert_eq!(result, "Edited a.txt. It is now empty.");
    }

    #[tokio::test]
    async fn crlf_files_match_text_with_plain_line_breaks_and_keep_theirs() {
        let ws = Workspace::new();
        ws.write("a.txt", b"one\r\ntwo\r\nthree\r\n");
        ws.edit(json!({"path": "a.txt", "old_string": "one\ntwo\n", "new_string": "1\n2\n"}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"1\r\n2\r\nthree\r\n");
        // Text already written with CRLF matches too.
        ws.edit(json!({"path": "a.txt", "old_string": "2\r\nthree", "new_string": "3"}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"1\r\n3\r\n");
        // An LF file gets no CRLF.
        ws.write("b.txt", b"one\ntwo\n");
        ws.edit(json!({"path": "b.txt", "old_string": "one\n", "new_string": "1\n0\n"}))
            .await
            .unwrap();
        assert_eq!(ws.read("b.txt"), b"1\n0\ntwo\n");
        // No other adjustment: trailing spaces must match.
        ws.write("c.txt", b"one  \r\ntwo\r\n");
        let error = ws
            .refused(
                "c.txt",
                json!({"path": "c.txt", "old_string": "one\ntwo", "new_string": "x"}),
            )
            .await;
        assert!(error.contains("wasn't found"), "{error}");
    }

    #[tokio::test]
    async fn a_byte_order_mark_is_kept() {
        let ws = Workspace::new();
        ws.write("a.txt", b"\xEF\xBB\xBFname = old\n");
        let result = ws
            .edit(json!({"path": "a.txt", "old_string": "old", "new_string": "new"}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"\xEF\xBB\xBFname = new\n");
        assert_eq!(result, "Edited a.txt at line 1:\n1 | name = new");
        // The text at the very start, after the mark.
        ws.edit(json!({"path": "a.txt", "old_string": "name", "new_string": "key"}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"\xEF\xBB\xBFkey = new\n");
    }

    #[tokio::test]
    async fn replace_all_changes_every_occurrence_and_says_where() {
        let ws = Workspace::new();
        ws.write("a.txt", b"foo\nbar foo\nbaz\nfoo foo\n");
        let result = ws
            .edit(json!({"path": "a.txt", "old_string": "foo", "new_string": "qux", "replace_all": true}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"qux\nbar qux\nbaz\nqux qux\n");
        // A line with two is named once.
        assert_eq!(
            result,
            "Replaced 4 occurrences in a.txt, at lines 1, 2 and 4."
        );
        // The lines are the file's as it is now.
        let result = ws
            .edit(json!({"path": "a.txt", "old_string": "qux", "new_string": "1\n2", "replace_all": true}))
            .await
            .unwrap();
        assert_eq!(ws.read("a.txt"), b"1\n2\nbar 1\n2\nbaz\n1\n2 1\n2\n");
        assert_eq!(
            result,
            "Replaced 4 occurrences in a.txt, at lines 1, 3, 6 and 7."
        );
        // As text, and with one occurrence.
        let result = ws
            .edit(json!({"path": "a.txt", "old_string": "baz", "new_string": "z", "replace_all": "true"}))
            .await
            .unwrap();
        assert_eq!(result, "Replaced 1 occurrence in a.txt, at line 5.");
        let error = ws
            .refused("a.txt", json!({"path": "a.txt", "old_string": "z", "new_string": "y", "replace_all": "maybe"}))
            .await;
        assert!(error.contains("must be true or false"), "{error}");
    }

    #[tokio::test]
    async fn text_found_more_than_once_is_refused_with_its_lines() {
        let ws = Workspace::new();
        ws.write("a.txt", b"x = 1\ny = 2\nx = 1\n\nx = 1\n");
        let error = ws
            .refused(
                "a.txt",
                json!({"path": "a.txt", "old_string": "x = 1", "new_string": "x = 3"}),
            )
            .await;
        assert!(
            error.contains("old_string was found 3 times in a.txt, at lines 1, 3 and 5, so the file is unchanged."),
            "{error}"
        );
        assert!(error.contains("set replace_all"), "{error}");
        // `replace_all: false` is the same.
        ws.refused("a.txt", json!({"path": "a.txt", "old_string": "x = 1", "new_string": "x = 3", "replace_all": false}))
            .await;
        // Many lines are listed in part.
        ws.write("b.txt", "a\n".repeat(30).as_bytes());
        let error = ws
            .refused(
                "b.txt",
                json!({"path": "b.txt", "old_string": "a", "new_string": "b"}),
            )
            .await;
        assert!(
            error.contains("found 30 times in b.txt, at lines 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 and 10 more,"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn text_not_found_is_refused_with_hints() {
        let ws = Workspace::new();
        ws.write("a.rs", b"fn main() {\n    let x = 1;\n    let y = 2;\n}\n");
        let error = ws
            .refused(
                "a.rs",
                json!({"path": "a.rs", "old_string": "let z = 3;", "new_string": "x"}),
            )
            .await;
        assert!(
            error.contains(
                "old_string wasn't found in a.rs, so the file is unchanged. Read the lines again"
            ),
            "{error}"
        );
        // Copied with read_file's line numbers.
        let error = ws
            .refused("a.rs", json!({"path": "a.rs", "old_string": "2 |     let x = 1;\n3 |     let y = 2;", "new_string": "x"}))
            .await;
        assert!(error.contains("starts with a line number"), "{error}");
        assert!(!error.contains("Read the lines again"), "{error}");
        // Right apart from indentation: the lines are named.
        let error = ws
            .refused("a.rs", json!({"path": "a.rs", "old_string": "  let x = 1;\n  let y = 2;", "new_string": "x"}))
            .await;
        assert!(
            error.contains("It matches lines 2-3 apart from indentation"),
            "{error}"
        );
        let error = ws
            .refused(
                "a.rs",
                json!({"path": "a.rs", "old_string": "\tlet y = 2;\n", "new_string": "x"}),
            )
            .await;
        assert!(
            error.contains("It matches line 3 apart from indentation"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn other_mistakes_are_refused_and_change_nothing() {
        let ws = Workspace::new();
        ws.write("a.txt", b"same\n");
        let error = ws
            .refused(
                "a.txt",
                json!({"path": "a.txt", "old_string": "", "new_string": "x"}),
            )
            .await;
        assert!(
            error.contains(
                "old_string is empty. To create a file or replace all of it, use write_file."
            ),
            "{error}"
        );
        let error = ws
            .refused(
                "a.txt",
                json!({"path": "a.txt", "old_string": "same", "new_string": "same"}),
            )
            .await;
        assert!(error.contains("nothing to change"), "{error}");
        let error = ws
            .refused(
                "missing.txt",
                json!({"path": "missing.txt", "old_string": "a", "new_string": "b"}),
            )
            .await;
        assert!(
            error.contains("missing.txt doesn't exist. To create it, use write_file."),
            "{error}"
        );
        assert!(!ws.root.join("missing.txt").exists());
        std::fs::create_dir_all(ws.root.join("dir")).unwrap();
        let error = ws
            .refused(
                "dir",
                json!({"path": "dir", "old_string": "a", "new_string": "b"}),
            )
            .await;
        assert!(error.contains("dir is a directory, not a file."), "{error}");
        ws.write("image.png", b"\x89PNG\r\n\x1a\n\xff\xfe");
        let error = ws
            .refused(
                "image.png",
                json!({"path": "image.png", "old_string": "PNG", "new_string": "GIF"}),
            )
            .await;
        assert!(error.contains("isn't a text file"), "{error}");
        ws.write("nul.bin", b"a\0b");
        let error = ws
            .refused(
                "nul.bin",
                json!({"path": "nul.bin", "old_string": "a", "new_string": "c"}),
            )
            .await;
        assert!(error.contains("isn't a text file"), "{error}");
        for (params, missing) in [
            (json!({"old_string": "a", "new_string": "b"}), "path"),
            (json!({"path": "a.txt", "new_string": "b"}), "old_string"),
            (json!({"path": "a.txt", "old_string": "same"}), "new_string"),
        ] {
            let error = ws.refused("a.txt", params).await;
            assert!(error.contains(missing), "{error}");
        }
        let error = ws
            .refused(
                "a.txt",
                json!({"path": " ", "old_string": "a", "new_string": "b"}),
            )
            .await;
        assert!(error.contains("cannot be empty"), "{error}");
        assert_eq!(ws.read("a.txt"), b"same\n");
    }

    #[test]
    fn its_path_is_checked_as_write_file_s() {
        let definition = EditFileTool.definition();
        let paths: Vec<&str> = definition
            .parameters
            .iter()
            .filter(|parameter| parameter.is_path)
            .map(|parameter| parameter.name.as_str())
            .collect();
        assert_eq!(paths, ["path"]);
        assert!(!definition.parallel_safe);
        assert_eq!(
            EditFileTool.recovery_mode(),
            crate::tool::ToolRecovery::InDoubt
        );
    }
}
