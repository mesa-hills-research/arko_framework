pub mod client;
pub mod codex_auth;
pub mod config;
pub mod conversation;
pub mod error;
pub mod message;
pub mod model;
pub mod providers;
pub mod pricing;
pub mod reasoning;
pub mod retry;
pub mod stop;
pub mod tool;
pub mod trace;
pub mod usage;

pub mod embedding_client;
pub mod embedding_config;
pub mod embedding_model;
pub mod embedding_provider_identifier;
pub mod token_utils;

pub use client::LLMClient;
pub use codex_auth::{CodexAuthFile, CodexAuthSync};
pub use config::{
    AnthropicCacheTtl, AnthropicInferenceGeo, GenerationConfig, OpenAIServiceTier, Verbosity,
};
pub use conversation::Conversation;
pub use error::{Error, RetryClass};
pub use message::{ConfigurationUpdate, Content, Image, Message, Role};
pub use model::{Model, ModelPricing, Provider};
pub use pricing::{ResidencyPricing, ScheduledPricing, ServiceTier, TierPricing};
pub use providers::{GenerationResponse, StreamEvent, http_client, http_client_builder};
pub use reasoning::{
    AnthropicReasoningEffort, GeminiThinkingLevel, OpenAIEffort,
    OpenAIReasoningItem, OpenAIReasoningMode, OpenRouterReasoning, ReasoningArtifact,
    ReasoningConfig, ReasoningContext, ReasoningEffort, ReasoningSummary,
};
pub use stop::{Refusal, RefusalSource, StopReason};
pub use tool::{ToolCall, ToolChoice, ToolConfig, ToolDefinition, ToolResult};
pub use retry::BackoffPolicy;
pub use trace::RequestTrace;
pub use usage::{BillingKind, ContextEvent, ContextEventType, ModelCallUsage, UsageInfo};

pub use embedding_client::EmbeddingClient;
pub use embedding_config::{EmbeddingConfig, VoyageInputType};
pub use embedding_model::EmbeddingModelDetails;
pub use embedding_provider_identifier::EmbeddingProviderIdentifier;
pub use providers::EmbeddingResponse;
pub use token_utils::{TokenCounter, count_tokens};

pub use providers::openrouter::{
    AppAttribution, AutoExactoPolicy, AutoRouterCostTier, AutoRouterOptions, DataCollection,
    OpenRouterCatalog, OpenRouterClient, OpenRouterModelInfo, OpenRouterOptions,
    ProviderPreferences, SortBy,
};

pub use providers::openai::{
    ChatGptOAuthCredentials, OAuthRefreshResult, OpenAIAuth, OpenAIRequestHook,
    is_token_expired, parse_jwt_expiry, refresh_chatgpt_oauth_token,
};