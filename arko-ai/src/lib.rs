pub mod client;
pub mod config;
pub mod conversation;
pub mod error;
pub mod message;
pub mod model;
pub mod providers;
pub mod reasoning;
pub mod tool;
pub mod usage;

pub mod embedding_client;
pub mod embedding_config;
pub mod embedding_model;
pub mod embedding_provider_identifier;
pub mod token_utils;

pub use client::LLMClient;
pub use config::{GenerationConfig, Verbosity};
pub use conversation::Conversation;
pub use error::Error;
pub use message::{Content, Image, Message, Role};
pub use model::{Model, Provider};
pub use providers::{GenerationResponse, StreamEvent};
pub use reasoning::{
    AnthropicReasoningEffort, GeminiThinkingLevel, OpenAIEffort,
    OpenAIReasoningMode, ReasoningArtifact, ReasoningConfig,
};
pub use tool::{ToolCall, ToolChoice, ToolConfig, ToolDefinition, ToolResult};
pub use usage::{ContextEvent, ContextEventType, ModelCallUsage, UsageInfo};

pub use embedding_client::EmbeddingClient;
pub use embedding_config::{EmbeddingConfig, VoyageInputType};
pub use embedding_model::EmbeddingModelDetails;
pub use embedding_provider_identifier::EmbeddingProviderIdentifier;
pub use providers::EmbeddingResponse;
pub use token_utils::count_tokens;

pub use providers::openai::{
    ChatGptOAuthCredentials, OAuthRefreshResult, OpenAIAuth,
    is_token_expired, parse_jwt_expiry, refresh_chatgpt_oauth_token,
};