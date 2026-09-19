use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

mod prompt_builder;
pub use self::prompt_builder::PromptBuilder;

/// Default opening tag delimiter.
pub const DEFAULT_TAG_OPEN: &str = "<";
/// Default closing tag delimiter.
pub const DEFAULT_TAG_CLOSE: &str = ">";
/// Default tag name for batched tool calls.
pub const DEFAULT_BATCHED_TOOL_CALL_TAG: &str = "batched_tool_call";
/// Default escape prefix for literal tags.
pub const DEFAULT_ESCAPE_PREFIX: &str = "!!--!!";

/// Configuration for parsing tool calls from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolParsingConfig {
    pub tag_open: String,
    pub tag_close: String,
    pub batched_tool_call_tag: String,
    pub escape_prefix: Option<String>,
}

impl Default for ToolParsingConfig {
    fn default() -> Self {
        Self {
            tag_open: DEFAULT_TAG_OPEN.to_string(),
            tag_close: DEFAULT_TAG_CLOSE.to_string(),
            batched_tool_call_tag: DEFAULT_BATCHED_TOOL_CALL_TAG.to_string(),
            escape_prefix: Some(DEFAULT_ESCAPE_PREFIX.to_string()),
        }
    }
}

/// Strips content between `{{#output_based_only}}` and `{{/output_based_only}}` markers.
///
/// This is used to remove output-based tool calling examples and instructions
/// from system prompts and agent instructions when native tool calling is
/// chosen in the agent config.
///
/// # Usage in agent config fields
///
/// Wrap any output-based-specific content in your `instructions` or
/// `role_description` fields:
///
/// ```text
/// You are a helpful coding assistant.
///
/// {{#output_based_only}}
/// When using tools, format your tool calls using XML-like syntax:
/// <tool_name>
/// <param>value</param>
/// </tool_name>
/// {{/output_based_only}}
///
/// Always explain your reasoning before making changes.
/// ```
///
/// When the agent config has `tool_calling_mode: Native`, the content
/// between the markers (inclusive) is removed. When `OutputBased` is
/// selected, the markers themselves are stripped but the content is kept.
pub fn strip_output_based_only_blocks(text: &str) -> String {
    let open_marker = "{{#output_based_only}}";
    let close_marker = "{{/output_based_only}}";
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    loop {
        match remaining.find(open_marker) {
            Some(start) => {
                result.push_str(&remaining[..start]);
                let after_open = &remaining[start + open_marker.len()..];
                match after_open.find(close_marker) {
                    Some(end) => {
                        remaining = &after_open[end + close_marker.len()..];
                    }
                    None => {
                        result.push_str(&remaining[start..]);
                        break;
                    }
                }
            }
            None => {
                result.push_str(remaining);
                break;
            }
        }
    }

    result
}

/// Removes the `{{#output_based_only}}` / `{{/output_based_only}}` marker
/// tags but preserves the content between them. Used when output-based mode
/// is active and the content should be shown.
pub fn remove_output_based_only_markers(text: &str) -> String {
    text.replace("{{#output_based_only}}", "")
        .replace("{{/output_based_only}}", "")
}

/// A parsed tool call with its name and parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub parameters: HashMap<String, Value>,
}

/// Events emitted by the raw tool call parser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RawParserEvent {
    /// Plain text content.
    Text(String),
    /// Start of a tool call.
    ToolCallStart(String),
    /// Parameter name encountered.
    ParamName(String),
    /// Chunk of parameter value.
    ParamValueChunk(String),
    /// End of current parameter.
    ParamEnd,
    /// End of tool call.
    ToolCallEnd(String),
    /// Start of batched tool calls.
    BatchedToolCallStart,
    /// End of batched tool calls.
    BatchedToolCallEnd,
    /// Parser error.
    Error(String),
}

#[derive(Debug, PartialEq)]
enum ParserState {
    ConsumingText,
    AwaitingTagName,
    InOpeningTagName,
    InParamValue {
        current_param_name: String,
        param_value_buffer: String,
        expected_closing_tag: String,
        escaped_expected_closing_tag: Option<String>,
    },
    InClosingTagName,
}

/// A streaming parser for tool calls in LLM output.
pub struct ToolCallParser {
    config: ToolParsingConfig,
    available_tool_names: HashSet<String>,
    state: ParserState,
    buffer: String,

    current_tool_name_context: Option<String>,
    tag_name_buffer: String,
}

impl ToolCallParser {
    /// Helper function to find the previous valid UTF-8 character boundary.
    fn prev_char_boundary(s: &str, idx: usize) -> usize {
        if idx >= s.len() {
            s.len()
        } else {
            let mut i = idx;
            while i > 0 && !s.is_char_boundary(i) {
                i -= 1;
            }
            i
        }
    }

    /// Creates a new parser with the given tool names and configuration.
    pub fn new(available_tool_names: HashSet<String>, config: ToolParsingConfig) -> Self {
        Self {
            config,
            available_tool_names,
            state: ParserState::ConsumingText,
            buffer: String::new(),
            current_tool_name_context: None,
            tag_name_buffer: String::new(),
        }
    }

    /// Ingests a chunk of text and returns any events parsed from it.
    pub fn ingest_chunk(&mut self, chunk: &str) -> Vec<RawParserEvent> {
        let mut events = Vec::new();
        self.buffer.push_str(chunk);

        loop {
            let initial_buffer_len = self.buffer.len();
            let initial_state = std::mem::discriminant(&self.state);
            match self.state {
                ParserState::ConsumingText => {
                    let open = &self.config.tag_open;
                    let close = &self.config.tag_close;
                    let escape_opt = self.config.escape_prefix.as_deref();

                    let mut made_progress = false;

                    loop {
                        if self.buffer.is_empty() {
                            break;
                        }

                        let next_open = self.buffer.find(open);
                        if let Some(open_idx) = next_open {
                            let is_escaped = escape_opt.is_some_and(|esc| {
                                open_idx >= esc.len()
                                    && &self.buffer[open_idx - esc.len()..open_idx] == esc
                            });

                            if is_escaped {
                                let esc_len = escape_opt.unwrap().len();
                                let safe_prefix_start = open_idx - esc_len;

                                if safe_prefix_start > 0 {
                                    let before = &self.buffer[..safe_prefix_start];
                                    if !before.is_empty() {
                                        made_progress = true;
                                        events.push(RawParserEvent::Text(before.to_string()));
                                    }
                                }

                                if let Some(rel_close) = self.buffer[open_idx..].find(close) {
                                    let end_idx = open_idx + rel_close + close.len();
                                    let literal_tag = &self.buffer[open_idx..end_idx];
                                    events.push(RawParserEvent::Text(literal_tag.to_string()));
                                    self.buffer.drain(..end_idx);
                                    made_progress = true;
                                    continue;
                                } else {
                                    if safe_prefix_start > 0 {
                                        self.buffer.drain(..safe_prefix_start);
                                        made_progress = true;
                                    }
                                    break;
                                }
                            } else {
                                if open_idx > 0 {
                                    let before = &self.buffer[..open_idx];
                                    if !before.is_empty() {
                                        events.push(RawParserEvent::Text(before.to_string()));
                                    }
                                }
                                self.buffer.drain(..open_idx + open.len());
                                self.tag_name_buffer.clear();
                                self.state = ParserState::AwaitingTagName;
                                made_progress = true;
                                break;
                            }
                        } else {
                            let esc_len = escape_opt.map(|s| s.len()).unwrap_or(0);
                            let lookbehind = esc_len + open.len().saturating_sub(1);
                            let split_point = self.buffer.len().saturating_sub(lookbehind);
                            let safe_split_point =
                                Self::prev_char_boundary(&self.buffer, split_point);

                            if safe_split_point > 0 {
                                let segment = &self.buffer[..safe_split_point];
                                if !segment.trim().is_empty() {
                                    events.push(RawParserEvent::Text(segment.to_string()));
                                }
                                self.buffer.drain(..safe_split_point);
                                made_progress = true;
                            }
                            break;
                        }
                    }

                    if !made_progress {
                        break;
                    }
                }
                ParserState::AwaitingTagName => {
                    if self.buffer.is_empty() {
                        break;
                    }

                    let first_char = self.buffer.chars().next().unwrap();
                    if first_char == '/' {
                        self.buffer.drain(..1);
                        self.state = ParserState::InClosingTagName;
                    } else if first_char.is_alphanumeric() || first_char == '_' || first_char == '-'
                    {
                        self.state = ParserState::InOpeningTagName;
                    } else {
                        events.push(RawParserEvent::Text(self.config.tag_open.clone()));
                        self.state = ParserState::ConsumingText;
                    }
                }
                ParserState::InOpeningTagName => {
                    if let Some(idx) = self.buffer.find(&self.config.tag_close) {
                        self.tag_name_buffer.push_str(&self.buffer[..idx]);
                        self.buffer.drain(..idx + self.config.tag_close.len());

                        let tag_name = self.tag_name_buffer.trim().to_string();
                        self.tag_name_buffer.clear();

                        if tag_name == self.config.batched_tool_call_tag {
                            events.push(RawParserEvent::BatchedToolCallStart);
                            self.state = ParserState::ConsumingText;
                        } else if self.available_tool_names.contains(&tag_name)
                            && self.current_tool_name_context.is_none()
                        {
                            self.current_tool_name_context = Some(tag_name.clone());
                            events.push(RawParserEvent::ToolCallStart(tag_name));
                            self.state = ParserState::ConsumingText;
                        } else if self.current_tool_name_context.is_some() {
                            events.push(RawParserEvent::ParamName(tag_name.clone()));
                            let expected_closing_tag = format!(
                                "{}{}{}{}",
                                self.config.tag_open, "/", tag_name, self.config.tag_close
                            );
                            let escaped_expected_closing_tag = self
                                .config
                                .escape_prefix
                                .as_ref()
                                .map(|prefix| format!("{}{}", prefix, expected_closing_tag));
                            self.state = ParserState::InParamValue {
                                current_param_name: tag_name,
                                param_value_buffer: String::new(),
                                expected_closing_tag,
                                escaped_expected_closing_tag,
                            };
                        } else {
                            events.push(RawParserEvent::Text(format!(
                                "{}{}{}",
                                self.config.tag_open, tag_name, self.config.tag_close
                            )));
                            self.state = ParserState::ConsumingText;
                        }
                    } else {
                        self.tag_name_buffer.push_str(&self.buffer);
                        self.buffer.clear();
                        break;
                    }
                }
                ParserState::InClosingTagName => {
                    if let Some(idx) = self.buffer.find(&self.config.tag_close) {
                        self.tag_name_buffer.push_str(&self.buffer[..idx]);
                        self.buffer.drain(..idx + self.config.tag_close.len());

                        let tag_name = self.tag_name_buffer.trim().to_string();
                        self.tag_name_buffer.clear();

                        if Some(&tag_name) == self.current_tool_name_context.as_ref() {
                            events.push(RawParserEvent::ToolCallEnd(tag_name.clone()));
                            self.current_tool_name_context = None;
                            self.state = ParserState::ConsumingText;
                        } else if tag_name == self.config.batched_tool_call_tag {
                            events.push(RawParserEvent::BatchedToolCallEnd);
                            self.state = ParserState::ConsumingText;
                        } else {
                            if self.current_tool_name_context.is_some() {
                                events.push(RawParserEvent::Error(format!("Mismatched or unexpected closing tag: </{}>. Current tool context: {:?}.", tag_name, self.current_tool_name_context)));
                            }
                            events.push(RawParserEvent::Text(format!(
                                "{}{}/{}{}",
                                self.config.tag_open, "/", tag_name, self.config.tag_close
                            )));
                            self.state = ParserState::ConsumingText;
                        }
                    } else {
                        self.tag_name_buffer.push_str(&self.buffer);
                        self.buffer.clear();
                        break;
                    }
                }
                ParserState::InParamValue {
                    current_param_name: _,
                    ref mut param_value_buffer,
                    ref expected_closing_tag,
                    ref escaped_expected_closing_tag,
                } => {
                    let closing_tag_pos = self.buffer.find(expected_closing_tag);
                    let escaped_tag_pos = escaped_expected_closing_tag
                        .as_ref()
                        .and_then(|tag| self.buffer.find(tag));

                    match (closing_tag_pos, escaped_tag_pos) {
                        (Some(pos), Some(esc_pos)) if esc_pos < pos => {
                            let escaped_tag = escaped_expected_closing_tag.as_ref().unwrap();
                            param_value_buffer.push_str(&self.buffer[..esc_pos]);
                            param_value_buffer.push_str(expected_closing_tag);
                            self.buffer.drain(..esc_pos + escaped_tag.len());
                        }
                        (Some(pos), _) => {
                            param_value_buffer.push_str(&self.buffer[..pos]);
                            if !param_value_buffer.is_empty() {
                                events.push(RawParserEvent::ParamValueChunk(std::mem::take(
                                    param_value_buffer,
                                )));
                            }
                            events.push(RawParserEvent::ParamEnd);
                            self.buffer.drain(..pos + expected_closing_tag.len());
                            self.state = ParserState::ConsumingText;
                        }
                        (None, Some(esc_pos)) => {
                            let escaped_tag = escaped_expected_closing_tag.as_ref().unwrap();
                            param_value_buffer.push_str(&self.buffer[..esc_pos]);
                            param_value_buffer.push_str(expected_closing_tag);
                            self.buffer.drain(..esc_pos + escaped_tag.len());
                        }
                        (None, None) => {
                            let max_lookbehind = expected_closing_tag.len().saturating_sub(1);
                            let split_point_bytes =
                                self.buffer.len().saturating_sub(max_lookbehind);
                            let safe_split_point =
                                Self::prev_char_boundary(&self.buffer, split_point_bytes);

                            if safe_split_point > 0 {
                                param_value_buffer.push_str(&self.buffer[..safe_split_point]);
                                self.buffer.drain(..safe_split_point);
                            }
                            break;
                        }
                    }
                }
            }

            if self.buffer.len() == initial_buffer_len
                && std::mem::discriminant(&self.state) == initial_state
                && !self.buffer.is_empty()
            {
                break;
            }
        }
        events
    }

    /// Finalizes parsing, returning any remaining events.
    pub fn finalize(&mut self) -> Vec<RawParserEvent> {
        let mut events = Vec::new();

        if let ParserState::InParamValue {
            ref mut param_value_buffer,
            ref current_param_name,
            ..
        } = self.state
        {
            if !param_value_buffer.is_empty() {
                events.push(RawParserEvent::ParamValueChunk(std::mem::take(
                    param_value_buffer,
                )));
            }
            if !self.buffer.is_empty() {
                events.push(RawParserEvent::ParamValueChunk(std::mem::take(
                    &mut self.buffer,
                )));
            }
            events.push(RawParserEvent::Error(format!(
                "Input ended with unclosed parameter value for param: <{}>",
                current_param_name
            )));
        }

        if !self.buffer.is_empty() {
            match self.state {
                ParserState::ConsumingText | ParserState::AwaitingTagName => {
                    events.push(RawParserEvent::Text(self.buffer.clone()));
                }
                ParserState::InOpeningTagName | ParserState::InClosingTagName => {
                    let effective_open_delim =
                        if matches!(self.state, ParserState::InClosingTagName) {
                            format!("{}{}", self.config.tag_open, "/")
                        } else {
                            self.config.tag_open.clone()
                        };

                    let incomplete_tag_text =
                        format!("{}{}", effective_open_delim, self.tag_name_buffer);
                    events.push(RawParserEvent::Error(format!(
                        "Input ended with incomplete tag: {}",
                        incomplete_tag_text
                    )));

                    events.push(RawParserEvent::Text(incomplete_tag_text));
                    events.push(RawParserEvent::Text(self.buffer.clone()));
                }
                ParserState::InParamValue { .. } => {}
            }
            self.buffer.clear();
        }

        if let Some(tool_name) = &self.current_tool_name_context {
            events.push(RawParserEvent::Error(format!(
                "Input ended with unclosed tool: {}",
                tool_name
            )));
        }

        self.state = ParserState::ConsumingText;
        self.current_tool_name_context = None;
        self.tag_name_buffer.clear();

        events
    }
}

/// Represents a group of tool calls (either single or batched).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolCallGroup {
    Single(ParsedToolCall),
    Batch(Vec<ParsedToolCall>),
}

/// Errors that can occur during tool call assembly.
#[derive(Error, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolAssemblyError {
    #[error("Mismatched tool closing tag: expected '{expected}', got '{actual}' during assembly.")]
    MismatchedClosingTag { expected: String, actual: String },
    #[error("Batch tool calls must all be for the same tool, found mixed types.")]
    MixedToolsInBatch,
    #[error("Unexpected parameter end event.")]
    UnexpectedParamEnd,
    #[error("Parameter name encountered without active tool call.")]
    ParamNameOutsideToolCall,
    #[error("Parameter value encountered without active parameter name.")]
    ParamValueWithoutName,
    #[error("Tool call ended without active tool call context.")]
    ToolEndWithoutActiveCall,
    #[error("Unexpected raw parser event encountered: {event:?}")]
    UnexpectedRawEvent { event: RawParserEvent },
    #[error("Tool call started within another unclosed tool call.")]
    NestedToolCall,
    #[error("Batch call started within an unclosed tool call.")]
    BatchInsideTool,
    #[error("Tool call started within an unclosed batch call without closing the batch first.")]
    ToolInsideBatch,
    #[error("Batched tool call closed without being started.")]
    UnopenedBatchClosed,
    #[error("Batched tool call started within another unclosed batched call.")]
    NestedBatchCall,
}

/// Assembles raw parser events into tool call groups.
///
/// # Returns
/// A tuple of (plain_text_output, tool_call_groups) or an assembly error.
pub fn assemble_tool_calls(
    events: Vec<RawParserEvent>,
) -> Result<(String, Vec<ToolCallGroup>), ToolAssemblyError> {
    let mut plain_text_output = String::new();
    let mut tool_call_groups = Vec::new();

    let mut current_tool_call: Option<ParsedToolCall> = None;
    let mut current_param_name: Option<String> = None;
    let mut current_param_value: String = String::new();

    let mut in_batch_mode = false;
    let mut current_batch_calls: Vec<ParsedToolCall> = Vec::new();
    let mut batch_tool_name_check: Option<String> = None;

    for event in events {
        match event {
            RawParserEvent::Text(text) => {
                if current_param_name.is_some() {
                    current_param_value.push_str(&text);
                } else {
                    plain_text_output.push_str(&text);
                }
            }
            RawParserEvent::ToolCallStart(name) => {
                if current_tool_call.is_some() {
                    return Err(ToolAssemblyError::NestedToolCall);
                }
                if in_batch_mode && current_batch_calls.is_empty() {
                    batch_tool_name_check = Some(name.clone());
                } else if in_batch_mode && batch_tool_name_check.as_ref() != Some(&name) {
                    return Err(ToolAssemblyError::MixedToolsInBatch);
                }
                current_tool_call = Some(ParsedToolCall {
                    name,
                    parameters: HashMap::new(),
                });
                current_param_name = None;
                current_param_value.clear();
            }
            RawParserEvent::ParamName(name) => {
                if current_tool_call.is_none() {
                    return Err(ToolAssemblyError::ParamNameOutsideToolCall);
                }
                if let Some(prev_param_name) = current_param_name.take() {
                    current_tool_call
                        .as_mut()
                        .unwrap()
                        .parameters
                        .insert(prev_param_name, Value::String(current_param_value.clone()));
                    current_param_value.clear();
                }
                current_param_name = Some(name);
            }
            RawParserEvent::ParamValueChunk(chunk) => {
                if current_param_name.is_none() {
                    return Err(ToolAssemblyError::ParamValueWithoutName);
                }
                current_param_value.push_str(&chunk);
            }
            RawParserEvent::ParamEnd => {
                if let Some(p_name) = current_param_name.take() {
                    if let Some(ref mut tool) = current_tool_call {
                        tool.parameters
                            .insert(p_name, Value::String(current_param_value.clone()));
                        current_param_value.clear();
                    } else {
                        return Err(ToolAssemblyError::UnexpectedParamEnd);
                    }
                } else {
                    return Err(ToolAssemblyError::UnexpectedParamEnd);
                }
            }
            RawParserEvent::ToolCallEnd(name) => {
                if let Some(mut tool) = current_tool_call.take() {
                    if tool.name != name {
                        return Err(ToolAssemblyError::MismatchedClosingTag {
                            expected: tool.name,
                            actual: name,
                        });
                    }
                    if let Some(p_name) = current_param_name.take() {
                        tool.parameters
                            .insert(p_name, Value::String(current_param_value.clone()));
                        current_param_value.clear();
                    }

                    if in_batch_mode {
                        current_batch_calls.push(tool);
                    } else {
                        tool_call_groups.push(ToolCallGroup::Single(tool));
                    }
                } else {
                    return Err(ToolAssemblyError::ToolEndWithoutActiveCall);
                }
            }
            RawParserEvent::BatchedToolCallStart => {
                if current_tool_call.is_some() {
                    return Err(ToolAssemblyError::BatchInsideTool);
                }
                if in_batch_mode {
                    return Err(ToolAssemblyError::NestedBatchCall);
                }
                in_batch_mode = true;
                current_batch_calls.clear();
                batch_tool_name_check = None;
            }
            RawParserEvent::BatchedToolCallEnd => {
                if !in_batch_mode {
                    return Err(ToolAssemblyError::UnopenedBatchClosed);
                }
                if current_tool_call.is_some() {
                    return Err(ToolAssemblyError::ToolEndWithoutActiveCall);
                }
                in_batch_mode = false;
                if !current_batch_calls.is_empty() {
                    tool_call_groups.push(ToolCallGroup::Batch(std::mem::take(
                        &mut current_batch_calls,
                    )));
                }
                batch_tool_name_check = None;
            }
            RawParserEvent::Error(e) => {
                log::warn!("Raw parser emitted error event during assembly: {}", e);
            }
        }
    }

    if current_param_name.is_some() {
        return Err(ToolAssemblyError::UnexpectedParamEnd);
    }
    if current_tool_call.is_some() {
        return Err(ToolAssemblyError::ToolEndWithoutActiveCall);
    }
    if in_batch_mode {
        return Err(ToolAssemblyError::UnopenedBatchClosed);
    }

    Ok((plain_text_output, tool_call_groups))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_calls_parse_independently_of_stream_chunk_boundaries() {
        let input = "<read_memory><tier>mémoire</tier><full>true</full></read_memory>";
        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let mut parser =
                ToolCallParser::new(HashSet::from(["read_memory".into()]), Default::default());
            let mut events = parser.ingest_chunk(&input[..split]);
            events.extend(parser.ingest_chunk(&input[split..]));
            events.extend(parser.finalize());
            let (_, calls) = assemble_tool_calls(events).unwrap();
            assert_eq!(
                calls,
                vec![ToolCallGroup::Single(ParsedToolCall {
                    name: "read_memory".into(),
                    parameters: HashMap::from([
                        ("tier".into(), Value::String("mémoire".into())),
                        ("full".into(), Value::String("true".into())),
                    ]),
                })],
                "stream split at byte {split}"
            );
        }
    }
}
