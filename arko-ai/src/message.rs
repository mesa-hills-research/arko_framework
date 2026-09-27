use crate::reasoning::{ReasoningArtifact, ReasoningEffort};
use crate::tool::{ToolCall, ToolResult};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Content {
    Text(String),
    Image(Image),
    ToolResult(ToolResult),
}

impl Content {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&Image> {
        match self {
            Content::Image(image) => Some(image),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResult> {
        match self {
            Content::ToolResult(result) => Some(result),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Image {
    Url(String),
    Base64 { data: String, mime_type: String },
}

impl Image {
    pub fn new_url(url: &str) -> Self {
        Image::Url(url.to_string())
    }

    pub fn new_base64(data: &str, mime_type: &str) -> Self {
        Image::Base64 {
            data: data.to_string(),
            mime_type: mime_type.to_string(),
        }
    }
}

/// A request setting changed from a point in the conversation onward,
/// without editing earlier turns, so the provider's prompt cache for the
/// turns before it stays valid. Added with
/// [`Conversation::add_reasoning_effort_update`].
///
/// [`Conversation::add_reasoning_effort_update`]: crate::conversation::Conversation::add_reasoning_effort_update
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationUpdate {
    /// Reasoning effort for the turns after this point.
    pub reasoning_effort: ReasoningEffort,
}

impl ConfigurationUpdate {
    pub fn reasoning_effort(effort: ReasoningEffort) -> Self {
        Self {
            reasoning_effort: effort,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
    /// Provider state an assistant message needs to be replayed: reasoning
    /// signatures, or for OpenAI the response's output items (see
    /// [`ReasoningArtifact::OpenAIOutputItems`]).
    pub reasoning_artifact: Option<ReasoningArtifact>,
    /// Tool calls produced by the model in this assistant message.
    /// Empty for user messages and assistant messages without tool use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub id: Option<String>,
    /// Set on a content-less system message that changes request settings
    /// from its position onward (see [`ConfigurationUpdate`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration_update: Option<ConfigurationUpdate>,
}

impl Message {
    pub fn new_user(text: &str) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_user_with_image(text: &str, image: Image) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.to_string()), Content::Image(image)],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_user_with_contents(contents: Vec<Content>) -> Self {
        Self {
            role: Role::User,
            content: contents,
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_assistant_with_reasoning(text: &str, reasoning_artifact: ReasoningArtifact) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: Some(reasoning_artifact),
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_system(text: &str) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    /// A content-less system message carrying a configuration update.
    pub fn new_configuration_update(update: ConfigurationUpdate) -> Self {
        Self {
            role: Role::System,
            content: Vec::new(),
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: Some(update),
        }
    }

    /// Returns `true` if this assistant message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Whether this message is a configuration update rather than a turn.
    pub fn is_configuration_update(&self) -> bool {
        self.configuration_update.is_some()
    }
}