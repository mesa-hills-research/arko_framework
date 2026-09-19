use crate::reasoning::ReasoningArtifact;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
    pub reasoning_artifact: Option<ReasoningArtifact>,
    /// Tool calls produced by the model in this assistant message.
    /// Empty for user messages and assistant messages without tool use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub id: Option<String>,
}

impl Message {
    pub fn new_user(text: &str) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
        }
    }

    pub fn new_user_with_image(text: &str, image: Image) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.to_string()), Content::Image(image)],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
        }
    }

    pub fn new_user_with_contents(contents: Vec<Content>) -> Self {
        Self {
            role: Role::User,
            content: contents,
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
        }
    }

    pub fn new_assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
        }
    }

    pub fn new_assistant_with_reasoning(text: &str, reasoning_artifact: ReasoningArtifact) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: Some(reasoning_artifact),
            tool_calls: Vec::new(),
            id: None,
        }
    }

    pub fn new_system(text: &str) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
        }
    }

    /// Returns `true` if this assistant message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}