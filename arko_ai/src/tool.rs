use crate::message::Image;
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
/// to calls on OpenAI (`call_id`), Anthropic (`tool_use` `id`), and Google
/// Gemini via the Interactions API (`function_call` step `id`). This is
/// especially important for parallel tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    /// Parsed arguments. `Value::Null` when the provider returned text that
    /// is not valid JSON. See [`invalid_arguments`](Self::invalid_arguments).
    pub arguments: serde_json::Value,
    /// Opaque Gemini signature for this function call.
    ///
    /// The Gemini Interactions API may attach a `signature` to
    /// `function_call` steps (in addition to signatures on `thought` steps,
    /// which are kept in the message's reasoning artifact). The Google
    /// provider stores it here and sends it back with the call. Other
    /// providers leave it `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Provider-specific item identifier that may differ from the
    /// correlation `id`.
    ///
    /// For the OpenAI Responses API, function call output items carry
    /// both an item `id` prefixed with `fc_` and a
    /// `call_id` prefixed with `call_`. The `call_id` is stored in
    /// [`id`](Self::id) for result matching, while the `fc_`-prefixed
    /// item identifier is stored here so it can be round-tripped when
    /// the conversation is sent back to the API.
    ///
    /// Other providers leave this as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
    /// The raw argument text when it was not valid JSON. Such a call must
    /// not be executed. Report the problem to the model instead. The text is
    /// replayed verbatim to providers that accept argument strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid_arguments: Option<String>,
}

impl ToolCall {
    /// Builds a call from provider argument text. Empty text means no
    /// arguments. Text that isn't valid JSON is kept in `invalid_arguments`.
    pub(crate) fn from_argument_text(id: Option<String>, name: String, text: &str) -> Self {
        let (arguments, invalid_arguments) = if text.trim().is_empty() {
            (serde_json::Value::Object(Default::default()), None)
        } else {
            match serde_json::from_str(text) {
                Ok(arguments) => (arguments, None),
                Err(_) => (serde_json::Value::Null, Some(text.to_string())),
            }
        };
        Self {
            id,
            name,
            arguments,
            thought_signature: None,
            provider_id: None,
            invalid_arguments,
        }
    }

    /// Arguments for providers that require a JSON object when the call is
    /// sent back as history (Anthropic `tool_use.input`, Gemini
    /// `function_call.arguments`). Invalid arguments become `{}`. The
    /// accompanying tool result tells the model what went wrong.
    pub(crate) fn replay_arguments(&self) -> serde_json::Value {
        if self.invalid_arguments.is_some() {
            serde_json::Value::Object(Default::default())
        } else {
            self.arguments.clone()
        }
    }
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
    /// and Google Gemini surface this to the model via `is_error`. For
    /// OpenAI the error details in `output` convey the failure.
    pub is_error: bool,
    /// Images the tool returned with its output (a screenshot, a rendered
    /// chart). Where the model accepts images inside a tool result
    /// ([`Model::accepts_tool_result_images`]) they are sent inside this
    /// result, after the output text. Otherwise they follow all of the
    /// turn's results, each after the text
    /// `"Image from tool {name}, call {call_id}: {label}"`. Results without
    /// images are sent exactly as before this field existed, and results
    /// saved before it load with none.
    ///
    /// [`Model::accepts_tool_result_images`]: crate::model::Model::accepts_tool_result_images
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ToolResultImage>,
}

/// An image returned with a [`ToolResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultImage {
    /// Short text sent before the image, such as its file name and size.
    /// Empty sends none (inside a result) or only the tool and call names
    /// (after the results).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    pub image: Image,
}

impl ToolResultImage {
    pub fn new(label: impl Into<String>, image: Image) -> Self {
        Self {
            label: label.into(),
            image,
        }
    }
}

impl ToolResult {
    pub fn new(call_id: Option<String>, name: &str, output: &str) -> Self {
        Self {
            call_id,
            name: name.to_string(),
            output: output.to_string(),
            is_error: false,
            images: Vec::new(),
        }
    }

    pub fn error(call_id: Option<String>, name: &str, output: &str) -> Self {
        Self {
            call_id,
            name: name.to_string(),
            output: output.to_string(),
            is_error: true,
            images: Vec::new(),
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
            images: Vec::new(),
        }
    }

    /// Attaches images the tool returned. See [`images`](Self::images).
    pub fn with_images(mut self, images: Vec<ToolResultImage>) -> Self {
        self.images = images;
        self
    }

    /// Creates an error result from the originating `ToolCall`.
    pub fn error_from_tool_call(tool_call: &ToolCall, output: &str) -> Self {
        Self {
            call_id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            output: output.to_string(),
            is_error: true,
            images: Vec::new(),
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
    /// support disabling parallel tool calls. Setting this to
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
