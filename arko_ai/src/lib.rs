//! Clients for language models from OpenAI (API key or ChatGPT OAuth), Anthropic, Google
//! Gemini and OpenRouter, for Claude models run through an installed Claude Code CLI, and
//! for Voyage text embeddings.
//!
//! [`LLMClient`] sends a [`Conversation`] to the provider of its model and returns a
//! [`GenerationResponse`] with the text and reasoning, tool calls, stop reason, refusal,
//! token usage and cost. [`EmbeddingClient`] embeds text with Voyage.
//!
//! The crate's README has a first example, and `docs/arko_ai-guide.md` in the repository
//! covers each provider, ChatGPT OAuth, usage and cost, and errors and retries.

pub mod client;
pub mod codex_auth;
pub mod config;
pub mod conversation;
pub mod error;
pub mod image_fit;
pub mod image_store;
pub mod image_tokens;
pub mod input_limit;
pub mod message;
pub mod model;
mod notice;
pub mod pricing;
pub mod process;
pub mod prompt_cache;
pub mod providers;
pub mod reasoning;
pub mod request_size;
pub mod retry;
pub mod stop;
pub mod tool;
mod tool_history;
pub use tool_history::TOOL_HISTORY_NOTE;
pub mod trace;
pub mod usage;

pub mod embedding_client;
pub mod embedding_config;
pub mod embedding_model;
pub mod embedding_provider_identifier;
pub mod token_utils;

pub use client::LLMClient;
pub use codex_auth::{CodexAuthFile, CodexAuthStore, CodexAuthSync};
pub use config::{
    AnthropicCacheTtl, AnthropicInferenceGeo, GenerationConfig, OpenAIServiceTier, Verbosity,
};
pub use conversation::{Conversation, OpenAITurnState};
pub use error::{Error, ErrorKind, RetryClass};
pub use input_limit::{CHATGPT_OAUTH_MAX_INPUT_TOKENS, effective_max_input_tokens};
pub use message::{ConfigurationUpdate, Content, Image, Message, Role};
pub use model::{Model, ModelPricing, Provider};
pub use pricing::{ResidencyPricing, ScheduledPricing, ServiceTier, TierPricing};
pub use prompt_cache::PromptCacheLifetime;
pub use providers::claude_code;
pub use providers::{GenerationResponse, StreamEvent, http_client, http_client_builder};
pub use reasoning::{
    AnthropicReasoningEffort, GeminiThinkingLevel, OpenAIEffort, OpenAIReasoningItem,
    OpenAIReasoningMode, OpenRouterReasoning, ReasoningArtifact, ReasoningConfig, ReasoningContext,
    ReasoningEffort, ReasoningSummary,
};
pub use request_size::RequestSizes;
pub use retry::BackoffPolicy;
pub use stop::{Refusal, RefusalSource, StopReason};
pub use tool::{ToolCall, ToolChoice, ToolConfig, ToolDefinition, ToolResult, ToolResultImage};
pub use trace::RequestTrace;
pub use usage::{
    BillingKind, ContextEvent, ContextEventType, ModelCallUsage, PlanCredits, PlanUsage,
    PlanUsageWindow, SubscriptionPlan, UsageInfo,
};

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
    ChatGptAccount, ChatGptAccountEvent, ChatGptAccountStatus, ChatGptAccounts,
    ChatGptOAuthCredentials, ChatGptOAuthStore, ChatGptUsageStatus, OAuthRefreshResult, OpenAIAuth,
    OpenAIRequestHook, is_token_expired, parse_jwt_expiry, refresh_chatgpt_oauth_token,
};
