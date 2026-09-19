use serde::{Deserialize, Serialize};

/// Defines a tool that the model can invoke during generation.
///
/// Tool definitions describe the function's name, purpose, and expected
/// parameters using JSON Schema. All three supported providers (OpenAI,
/// Anthropic, Google Gemini) share this common schema format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// When enabled, the provider guarantees that the model's generated
    /// arguments will conform exactly to the parameter schema. Supported
    /// by OpenAI (`strict` field) and Anthropic (`strict` field). Ignored
    /// by Google Gemini.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ToolDefinition {
    pub fn new(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
            strict: None,
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

/// A tool invocation produced by the model.
///
/// Contains the function name and parsed arguments. The `id` field is
/// populated by all supported providers and is used to match results back
/// to calls: OpenAI (`call_id`), Anthropic (`tool_use` `id`), and Google
/// Gemini via the Interactions API (`function_call` step `id`). This is
/// especially important for parallel tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
    /// Opaque signature that the legacy Gemini generateContent API attached
    /// to function-call parts to maintain reasoning context across
    /// tool-call turns.
    ///
    /// The Gemini Interactions API carries thought signatures on dedicated
    /// `thought` steps (preserved via the message's reasoning artifact)
    /// instead, so the Google provider no longer populates or sends this
    /// field. It is retained for compatibility with previously serialized
    /// conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Provider-specific item identifier that may differ from the
    /// correlation `id`.
    ///
    /// For the OpenAI Responses API, function call output items carry
    /// two identifiers: an item `id` prefixed with `fc_` and a
    /// `call_id` prefixed with `call_`. The `call_id` is stored in
    /// [`id`](Self::id) for result matching, while the `fc_`-prefixed
    /// item identifier is stored here so it can be round-tripped when
    /// the conversation is sent back to the API.
    ///
    /// Other providers leave this as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
}

/// The result of executing a tool locally in response to a `ToolCall`.
///
/// Both `call_id` and `name` should be populated to ensure correct
/// matching across all providers:
/// - OpenAI, Anthropic, and Google Gemini (Interactions API) use `call_id`
///   to correlate results with calls.
/// - Google Gemini's `function_result` steps additionally carry the
///   function `name`.
///
/// The easiest way to construct a result is via [`ToolResult::from_tool_call`],
/// which copies both fields from the originating call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: Option<String>,
    pub name: String,
    pub output: String,
    /// When `true`, indicates that the tool execution failed. Anthropic
    /// and Google Gemini surface this to the model via `is_error`; for
    /// OpenAI the error details in `output` convey the failure.
    pub is_error: bool,
}

impl ToolResult {
    pub fn new(call_id: Option<String>, name: &str, output: &str) -> Self {
        Self {
            call_id,
            name: name.to_string(),
            output: output.to_string(),
            is_error: false,
        }
    }

    pub fn error(call_id: Option<String>, name: &str, output: &str) -> Self {
        Self {
            call_id,
            name: name.to_string(),
            output: output.to_string(),
            is_error: true,
        }
    }

    /// Creates a result from the originating `ToolCall`, copying its
    /// `id` and `name` so that all providers can match the result.
    pub fn from_tool_call(tool_call: &ToolCall, output: &str) -> Self {
        Self {
            call_id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            output: output.to_string(),
            is_error: false,
        }
    }

    /// Creates an error result from the originating `ToolCall`.
    pub fn error_from_tool_call(tool_call: &ToolCall, output: &str) -> Self {
        Self {
            call_id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            output: output.to_string(),
            is_error: true,
        }
    }
}

/// Controls how the model selects tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    /// The model decides whether to call any tools (default).
    Auto,
    /// The model must call at least one tool.
    Required,
    /// The model must call the specific named tool.
    Specific(String),
    /// The model must not call any tools, even if definitions are provided.
    None,
}

/// Configuration for tool calling, passed as part of [`GenerationConfig`].
///
/// [`GenerationConfig`]: crate::config::GenerationConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Controls whether the model may issue multiple tool calls in a
    /// single response.
    ///
    /// - `None` — use provider default (parallel calls enabled).
    /// - `Some(true)` — explicitly enable parallel tool calls.
    /// - `Some(false)` — restrict the model to at most one tool call
    ///   per turn.
    ///
    /// Supported by OpenAI and Anthropic. Google Gemini does **not**
    /// support disabling parallel tool calls; setting this to
    /// `Some(false)` for a Gemini model will return
    /// [`Error::UnsupportedFeature`](crate::error::Error::UnsupportedFeature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

impl ToolConfig {
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        Self {
            tools,
            tool_choice: None,
            parallel_tool_calls: None,
        }
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }
}