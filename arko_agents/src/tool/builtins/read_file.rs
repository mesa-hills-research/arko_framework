use crate::tool::{
    MAX_TOOL_IMAGE_BYTES, TaskContext, Tool, ToolDefinition, ToolError, ToolImage, ToolOutput,
    ToolParameterDefinition, ToolParameterType, ToolTypeDetails,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

pub const READ_FILE_TOOL_NAME: &str = "read_file";
const PATH_PARAM_NAME: &str = "path";
/// The optional first line to read, when the agent's configuration turns
/// line ranges on ([`AgentToolConfig::read_file_line_ranges`](crate::agent_config::AgentToolConfig::read_file_line_ranges)).
pub const START_LINE_PARAM_NAME: &str = "start_line";
/// The optional last line to read, when line ranges are on.
pub const END_LINE_PARAM_NAME: &str = "end_line";

/// The sentence `read_file`'s description gains when the agent is also
/// offered a terminal, so large files can be read in parts with it.
pub const TERMINAL_HINT: &str = "For a large or generated file (data, logs, minified code), look at the parts you need with the terminal instead: `wc -l`, `grep -n`, `head`, `sed -n '100,200p'`.";

/// The sentence `read_file`'s description gains with line ranges on.
pub const LINE_RANGES_HINT: &str =
    "To read part of a file, give start_line and end_line (1-based, inclusive).";

#[derive(Debug, Clone)]
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

/// `read_file`'s definition as an agent with `tool_config` is shown it. The
/// built-in `read_file` (not a host's own tool of that name) gains one
/// sentence, [`TERMINAL_HINT`], when the agent is also offered arko's
/// terminal tools, and, with
/// [`read_file_line_ranges`](crate::agent_config::AgentToolConfig::read_file_line_ranges)
/// on, the optional `start_line` and `end_line` parameters and
/// [`LINE_RANGES_HINT`]. Any other definition is returned unchanged.
///
/// Only the description and schema change, and they follow from the
/// configuration, so the same configuration shows the same definition. Tool
/// definitions are not part of the configuration a workflow compares when
/// it resumes, so a saved run resumes as before. (In output-based mode the
/// definitions are written into the system prompt when a task starts, so a
/// saved task keeps the one it started with.)
pub fn offered_definition(
    mut definition: ToolDefinition,
    tool_config: &crate::agent_config::AgentToolConfig,
) -> ToolDefinition {
    if definition.name != READ_FILE_TOOL_NAME
        || !matches!(definition.tool_type_details, ToolTypeDetails::BuiltIn)
    {
        return definition;
    }
    if tool_config.read_file_line_ranges {
        definition.description = format!("{} {LINE_RANGES_HINT}", definition.description);
        for (name, description) in [
            (
                START_LINE_PARAM_NAME,
                "Optional: the first line to read (1-based). Leave out start_line and end_line to read the whole file.",
            ),
            (
                END_LINE_PARAM_NAME,
                "Optional: the last line to read (inclusive); past the end of the file reads to its end.",
            ),
        ] {
            definition.parameters.push(ToolParameterDefinition {
                name: name.to_string(),
                description: description.to_string(),
                param_type: ToolParameterType::Integer,
                required: false,
                is_path: false,
                flag: None,
                default_value: None,
            });
        }
    }
    if tool_config.offers_terminal() {
        definition.description = format!("{} {TERMINAL_HINT}", definition.description);
    }
    definition
}

/// Removes the line-range parameters from a call to the built-in
/// `read_file` whose agent hasn't turned line ranges on, so the call reads
/// the whole file, as its schema says.
pub(crate) fn ignore_line_ranges(
    tool: &dyn Tool,
    tool_config: &crate::agent_config::AgentToolConfig,
    params: &mut HashMap<String, Value>,
) {
    if tool_config.read_file_line_ranges {
        return;
    }
    let definition = tool.definition();
    if definition.name == READ_FILE_TOOL_NAME
        && matches!(definition.tool_type_details, ToolTypeDetails::BuiltIn)
    {
        params.remove(START_LINE_PARAM_NAME);
        params.remove(END_LINE_PARAM_NAME);
    }
}

/// A line-number parameter: a positive integer, as a number or as text.
fn line_number(params: &HashMap<String, Value>, name: &str) -> Result<Option<usize>, ToolError> {
    let Some(value) = params.get(name) else {
        return Ok(None);
    };
    let number = match value {
        Value::Null => return Ok(None),
        Value::Number(number) => number.as_u64(),
        Value::String(text) if text.trim().is_empty() => return Ok(None),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    };
    match number {
        Some(number) if number >= 1 => Ok(Some(number as usize)),
        _ => Err(ToolError::ParameterValidation {
            param_name: name.to_string(),
            message: format!("Parameter '{name}' must be a line number (1 or more)."),
        }),
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    /// Reading has no side effects. A call running when its task stopped
    /// reads again.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        crate::tool::ToolRecovery::Rerun
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: READ_FILE_TOOL_NAME.to_string(),
            description: "Reads the content of a file from the workspace, prepending line numbers to each line (e.g., '1 | line content'). An image file (PNG, JPEG, GIF or WebP) is shown as an image.".to_string(),
            output_based_example: Some(
                "<read_file>\n<path>src/main.rs</path>\n</read_file>".to_string()
            ),
            parameters: vec![ToolParameterDefinition {
                name: PATH_PARAM_NAME.to_string(),
                description: "The path to the file to read (relative to the current working directory, e.g. 'src/main.rs').".to_string(),
                param_type: ToolParameterType::String,
                required: true,
                is_path: true,
                flag: None,
                default_value: None,
            }],
            parallel_safe: true,
            tool_type_details: ToolTypeDetails::BuiltIn,
            read_only: true,
            raw_json_schema_override: None,
        }
    }

    /// The file's text. See [`execute_with_output`](Self::execute_with_output).
    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.execute_with_output(context, params)
            .await
            .map(|output| output.text)
    }

    /// Reads the file, or with `start_line` and/or `end_line` (which the
    /// runtime passes only when the agent's configuration turns line
    /// ranges on) only those lines, followed by a line giving the range and
    /// the file's line count. An image file (PNG, JPEG, GIF or WebP, told
    /// by its first bytes) is returned as an image, with a line naming it,
    /// unless it is larger than [`MAX_TOOL_IMAGE_BYTES`].
    async fn execute_with_output(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        let file_path_str = params
            .get(PATH_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: PATH_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: format!("Parameter '{}' must be a string.", PATH_PARAM_NAME),
            })?
            .trim()
            .to_string();

        if file_path_str.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "File path cannot be empty.".to_string(),
            });
        }
        let start_line = line_number(&params, START_LINE_PARAM_NAME)?;
        let end_line = line_number(&params, END_LINE_PARAM_NAME)?;
        if let (Some(start), Some(end)) = (start_line, end_line)
            && end < start
        {
            return Err(ToolError::ParameterValidation {
                param_name: END_LINE_PARAM_NAME.to_string(),
                message: format!("end_line ({end}) is before start_line ({start})."),
            });
        }

        let absolute_path = super::resolve_path_parameter(
            PATH_PARAM_NAME,
            &file_path_str,
            context.current_working_directory,
        )?;

        log::debug!(
            "ReadFileTool: Attempting to read file at absolute path: {:?}",
            absolute_path
        );

        let bytes = tokio::fs::read(&absolute_path)
            .await
            .map_err(|e| ToolError::Io {
                tool_name: self.name(),
                source: e,
            })?;

        if let Some(mime_type) = image_type(&bytes) {
            return Ok(image_output(&file_path_str, mime_type, &bytes));
        }

        let content_bytes = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            bytes[3..].to_vec()
        } else {
            bytes
        };

        let content = String::from_utf8(content_bytes)
            .map_err(|_| ToolError::ExecutionFailed("File is not valid UTF-8".to_string()))?;
        let normalized = content.replace("\r\n", "\n");
        let numbered = |(i, line): (usize, &str)| format!("{} | {}", i + 1, line);
        if start_line.is_none() && end_line.is_none() {
            return Ok(ToolOutput::text(
                normalized
                    .lines()
                    .enumerate()
                    .map(numbered)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ));
        }
        let total = normalized.lines().count();
        let start = start_line.unwrap_or(1);
        if start > total {
            return Err(ToolError::ParameterValidation {
                param_name: START_LINE_PARAM_NAME.to_string(),
                message: format!(
                    "start_line ({start}) is past the end of the file ({total} lines)."
                ),
            });
        }
        let end = end_line.unwrap_or(total).min(total);
        let mut lines = normalized
            .lines()
            .enumerate()
            .skip(start - 1)
            .take(end + 1 - start)
            .map(numbered)
            .collect::<Vec<_>>();
        lines.push(format!("[Lines {start}-{end} of {total}.]"));
        Ok(ToolOutput::text(lines.join("\n")))
    }
}

/// The image type of a file that starts with `bytes`: PNG, JPEG, GIF or
/// WebP.
fn image_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// An image file's result: a line naming it, and the image, or for a file
/// over [`MAX_TOOL_IMAGE_BYTES`] a line saying it is too large to show.
fn image_output(path: &str, mime_type: &str, bytes: &[u8]) -> ToolOutput {
    let size = if bytes.len() < 1024 * 1024 {
        format!("{} KB", bytes.len().div_ceil(1024))
    } else {
        format!("{:.1} MB", bytes.len() as f64 / (1024.0 * 1024.0))
    };
    if bytes.len() > MAX_TOOL_IMAGE_BYTES {
        return ToolOutput::text(format!(
            "{path} is an image ({mime_type}, {size}), too large to show. The limit is {} MB.",
            MAX_TOOL_IMAGE_BYTES / (1024 * 1024)
        ));
    }
    ToolOutput {
        text: format!("{path} is an image ({mime_type}, {size})."),
        images: vec![ToolImage {
            description: format!("{path}, {size}"),
            image: arko_ai::Image::from_bytes(bytes, mime_type),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::AgentToolConfig;
    use serde_json::json;

    fn tool_config(names: &[&str], ranges: bool) -> AgentToolConfig {
        AgentToolConfig {
            available_tool_names: names.iter().map(|name| name.to_string()).collect(),
            read_file_line_ranges: ranges,
            ..AgentToolConfig::default()
        }
    }

    fn parameter_names(definition: &ToolDefinition) -> Vec<String> {
        definition
            .parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect()
    }

    #[test]
    fn the_terminal_hint_and_line_ranges_follow_the_configuration() {
        let plain = ReadFileTool.definition();
        // No terminal, no ranges: the definition as it was.
        let offered = offered_definition(plain.clone(), &tool_config(&["read_file"], false));
        assert_eq!(offered.description, plain.description);
        assert_eq!(parameter_names(&offered), ["path"]);
        // A terminal adds one sentence and nothing else.
        let offered = offered_definition(
            plain.clone(),
            &tool_config(&["read_file", "terminal_start", "terminal_status"], false),
        );
        assert_eq!(
            offered.description,
            format!("{} {TERMINAL_HINT}", plain.description)
        );
        assert_eq!(parameter_names(&offered), ["path"]);
        let schema = offered.to_json_schema();
        assert!(!schema.to_string().contains("start_line"), "{schema}");
        // Ranges on: the optional parameters and their sentence.
        let offered = offered_definition(plain.clone(), &tool_config(&["read_file"], true));
        assert_eq!(
            parameter_names(&offered),
            ["path", "start_line", "end_line"]
        );
        assert!(offered.description.ends_with(LINE_RANGES_HINT));
        assert!(offered.parameters[1..].iter().all(|p| !p.required));
        // Another tool, or a host's own read_file, is left alone.
        let custom = ToolDefinition {
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            ..plain.clone()
        };
        let config = tool_config(&["read_file", "terminal_start"], true);
        assert_eq!(
            offered_definition(custom.clone(), &config).description,
            custom.description
        );
        let list = crate::tool::builtins::ListFilesTool::new().definition();
        assert_eq!(
            offered_definition(list.clone(), &config).description,
            list.description
        );
    }

    fn context(directory: &std::path::Path) -> TaskContext<'_> {
        TaskContext::new(uuid::Uuid::new_v4(), "a", "", directory)
    }

    fn params(value: Value) -> HashMap<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[tokio::test]
    async fn image_signatures_are_detected_without_relying_on_the_file_extension() {
        let directory =
            std::env::temp_dir().join(format!("arko-read-images-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let context = context(&directory);
        // Only the file signature matters here; the reader does not decode pixels.
        for (name, bytes, mime_type) in [
            (
                "png.txt",
                b"\x89PNG\r\n\x1a\n\x00\xff".as_slice(),
                "image/png",
            ),
            ("jpeg.txt", b"\xff\xd8\xff\x00".as_slice(), "image/jpeg"),
            ("gif87.txt", b"GIF87a\x00\xff".as_slice(), "image/gif"),
            ("gif89.txt", b"GIF89a\x00\xff".as_slice(), "image/gif"),
            (
                "webp.txt",
                b"RIFF\x04\x00\x00\x00WEBP\xff".as_slice(),
                "image/webp",
            ),
        ] {
            std::fs::write(directory.join(name), bytes).unwrap();
            let output = ReadFileTool
                .execute_with_output(&context, params(json!({"path": name})))
                .await
                .unwrap();
            assert_eq!(output.images.len(), 1, "{name}");
            assert_eq!(
                output.images[0].image,
                arko_ai::Image::from_bytes(bytes, mime_type)
            );
            assert!(output.images[0].description.contains(name));
            // The image follows the line, which says only what it is.
            assert_eq!(
                output.text,
                format!("{name} is an image ({mime_type}, 1 KB).")
            );
            assert_eq!(
                ReadFileTool
                    .execute(&context, params(json!({"path": name})))
                    .await
                    .unwrap(),
                output.text,
                "legacy string reads still describe {name}"
            );
        }
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[tokio::test]
    async fn structured_file_reads_preserve_text_normalization_ranges_and_errors() {
        let directory =
            std::env::temp_dir().join(format!("arko-read-text-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        // An image extension cannot override text detection, including a UTF-8 BOM.
        std::fs::write(
            directory.join("notes.png"),
            "\u{feff}first\r\n二\r\nlast\r\n",
        )
        .unwrap();
        std::fs::write(directory.join("empty"), []).unwrap();
        std::fs::write(directory.join("not-an-image"), [0xff, 0x00, 0x01]).unwrap();
        let context = context(&directory);
        for (arguments, expected) in [
            (json!({"path": "notes.png"}), "1 | first\n2 | 二\n3 | last"),
            (
                json!({"path": "notes.png", "start_line": 2, "end_line": 2}),
                "2 | 二\n[Lines 2-2 of 3.]",
            ),
            (json!({"path": "empty"}), ""),
        ] {
            let output = ReadFileTool
                .execute_with_output(&context, params(arguments.clone()))
                .await
                .unwrap();
            assert!(output.images.is_empty());
            assert_eq!(output.text, expected);
            assert_eq!(
                ReadFileTool
                    .execute(&context, params(arguments))
                    .await
                    .unwrap(),
                expected
            );
        }
        let error = ReadFileTool
            .execute_with_output(&context, params(json!({"path": "not-an-image"})))
            .await
            .unwrap_err();
        assert!(
            matches!(error, ToolError::ExecutionFailed(message) if message == "File is not valid UTF-8")
        );
        let missing = ReadFileTool
            .execute_with_output(&context, params(json!({"path": "missing.png"})))
            .await
            .unwrap_err();
        assert!(matches!(missing, ToolError::Io { .. }));
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[tokio::test]
    async fn file_images_at_the_size_limit_are_shown_but_larger_images_are_described() {
        let directory =
            std::env::temp_dir().join(format!("arko-read-image-limit-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let context = context(&directory);
        let mut bytes = vec![0; MAX_TOOL_IMAGE_BYTES];
        bytes[..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        std::fs::write(directory.join("limit.png"), &bytes).unwrap();
        let output = ReadFileTool
            .execute_with_output(&context, params(json!({"path": "limit.png"})))
            .await
            .unwrap();
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.images[0].decoded_len(), Some(MAX_TOOL_IMAGE_BYTES));
        bytes.push(0);
        std::fs::write(directory.join("too-large.png"), &bytes).unwrap();
        let output = ReadFileTool
            .execute_with_output(&context, params(json!({"path": "too-large.png"})))
            .await
            .unwrap();
        assert!(output.images.is_empty());
        assert!(output.text.contains("too-large.png"));
        assert!(output.text.contains("too large to show"));
        assert!(output.text.contains("limit is 5 MB"));
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[tokio::test]
    async fn line_ranges_read_part_of_the_file_and_are_ignored_unless_turned_on() {
        let directory =
            std::env::temp_dir().join(format!("arko-read-ranges-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let text = (1..=10)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(directory.join("f.txt"), text).unwrap();
        let context = context(&directory);
        let read = |value: Value| {
            let context = &context;
            async move { ReadFileTool.execute(context, params(value)).await }
        };
        assert_eq!(
            read(json!({"path": "f.txt", "start_line": 3, "end_line": "4"}))
                .await
                .unwrap(),
            "3 | line 3\n4 | line 4\n[Lines 3-4 of 10.]"
        );
        assert_eq!(
            read(json!({"path": "f.txt", "start_line": "9", "end_line": 50}))
                .await
                .unwrap(),
            "9 | line 9\n10 | line 10\n[Lines 9-10 of 10.]"
        );
        assert!(
            read(json!({"path": "f.txt", "end_line": 1}))
                .await
                .unwrap()
                .starts_with("1 | line 1\n[Lines 1-1 of 10.]")
        );
        for bad in [
            json!({"path": "f.txt", "start_line": 0}),
            json!({"path": "f.txt", "start_line": "x"}),
            json!({"path": "f.txt", "start_line": 5, "end_line": 4}),
            json!({"path": "f.txt", "start_line": 11}),
        ] {
            assert!(read(bad.clone()).await.is_err(), "{bad}");
        }
        // Without the setting, the runtime drops the parameters: the whole file.
        let mut call = params(json!({"path": "f.txt", "start_line": 3, "end_line": 4}));
        ignore_line_ranges(
            &ReadFileTool,
            &tool_config(&["read_file"], false),
            &mut call,
        );
        assert_eq!(call.len(), 1);
        let whole = ReadFileTool.execute(&context, call).await.unwrap();
        assert!(whole.starts_with("1 | line 1\n") && whole.ends_with("10 | line 10"));
        let mut call = params(json!({"path": "f.txt", "start_line": 3}));
        ignore_line_ranges(&ReadFileTool, &tool_config(&["read_file"], true), &mut call);
        assert_eq!(call.len(), 2);
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
