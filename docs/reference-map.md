# arko_framework reference map

This is the all-in-one map of the public arko_framework API. It names each configurable
feature and public primitive, points to the defining Rust file, and links to the detailed
behavior and defaults for using it correctly.

- [arko_ai](#arko_ai)
- [arko_agents](#arko_agents)
- [arko_workflows](#arko_workflows)
- [arko_index](#arko_index)
- [Complete workflow example](#complete-workflow-example)

The crates build on one another: `arko_agents` uses `arko_ai`, `arko_workflows` uses both,
and `arko_index` uses `arko_ai`. Code paths in the tables are repository-relative. Defaults
and omitted serialized fields are called out separately because they can differ.

## arko_ai

Clients for OpenAI (API key or ChatGPT OAuth), Anthropic, Google Gemini and OpenRouter models, and for Voyage text embeddings, with one conversation, response, usage and error model across providers.

Guides: [arko_ai README](../arko_ai/README.md) ([OpenAI and ChatGPT OAuth](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth), [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials), [OpenRouter](../arko_ai/README.md#openrouter), [Voyage embeddings](../arko_ai/README.md#voyage-embeddings)).

### Map (arko_ai)

Code paths are relative to the repository root. Items are listed by their shortest public path: most types are re-exported at the crate root (`arko_ai::LLMClient`); the others are reached through their module (`arko_ai::providers::openai::OpenAIClient`, `arko_ai::model::ModelType`).

**Modules**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_ai::client` | `LLMClient`, the client that sends conversations to every provider. | `arko_ai/src/client.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `arko_ai::codex_auth` | Reading a Codex `auth.json` and keeping it in step with rotated ChatGPT OAuth tokens (`CodexAuthFile`, `CodexAuthStore`, `CodexAuthSync`). | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `arko_ai::config` | `GenerationConfig` and the enums of its fields. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `arko_ai::conversation` | `Conversation` and `OpenAITurnState`. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `arko_ai::error` | `Error` and `RetryClass`. | `arko_ai/src/error.rs` | [Errors and retries](../arko_ai/README.md#errors-and-retries) |
| `arko_ai::input_limit` | Input-token limit of a model on the auth path in use. | `arko_ai/src/input_limit.rs` | [Models](../arko_ai/README.md#models) |
| `arko_ai::message` | `Message`, `Role`, `Content`, `Image`, `ConfigurationUpdate`. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `arko_ai::model` | The model registry: `Model`, `Provider`, `ModelType`, `ModelPricing`. | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `arko_ai::pricing` | Price variants: `ServiceTier`, `TierPricing`, `ResidencyPricing`, `ScheduledPricing`. | `arko_ai/src/pricing.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `arko_ai::prompt_cache` | How long providers keep prompt-cache entries. | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `arko_ai::providers` | The provider clients, `GenerationResponse`, `StreamEvent`, `ProviderClient`, `EmbeddingResponse`, and the shared HTTP client. | `arko_ai/src/providers/mod.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `arko_ai::providers::openai` | OpenAI Responses API client, API key and ChatGPT OAuth. | `arko_ai/src/providers/openai.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `arko_ai::providers::anthropic` | Anthropic Messages API client. | `arko_ai/src/providers/anthropic.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `arko_ai::providers::google` | Gemini client on the Interactions API. | `arko_ai/src/providers/google.rs` | [Generation settings and tool calls](../arko_ai/README.md#generation-settings-and-tool-calls) |
| `arko_ai::providers::openrouter` | OpenRouter Chat Completions client, its options and model catalog. | `arko_ai/src/providers/openrouter.rs` | [OpenRouter](../arko_ai/README.md#openrouter) |
| `arko_ai::providers::voyage` | Voyage embeddings client. | `arko_ai/src/providers/voyage.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `arko_ai::reasoning` | `ReasoningConfig`, the effort and level enums, and `ReasoningArtifact`. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `arko_ai::retry` | `BackoffPolicy` and `MAX_SERVER_DELAY_HINT`. | `arko_ai/src/retry.rs` | [Errors and retries](../arko_ai/README.md#errors-and-retries) |
| `arko_ai::stop` | `StopReason`, `Refusal`, `RefusalSource`. | `arko_ai/src/stop.rs` | [Stop reasons and refusals](../arko_ai/README.md#stop-reasons-and-refusals) |
| `arko_ai::tool` | Tool definitions, calls, results and the tool choice. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `arko_ai::trace` | `RequestTrace`, the opt-in request trace. | `arko_ai/src/trace.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `arko_ai::usage` | `UsageInfo`, `ModelCallUsage`, `PlanUsage`, `BillingKind`, `ContextEvent`. | `arko_ai/src/usage.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `arko_ai::embedding_client` | `EmbeddingClient`. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `arko_ai::embedding_config` | `EmbeddingConfig` and `VoyageInputType`. | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `arko_ai::embedding_model` | `EmbeddingModelDetails`, the embedding model registry. | `arko_ai/src/embedding_model.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `arko_ai::embedding_provider_identifier` | `EmbeddingProviderIdentifier`. | `arko_ai/src/embedding_provider_identifier.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `arko_ai::token_utils` | Local token counting with Hugging Face tokenizers (`TokenCounter`, `count_tokens`). | `arko_ai/src/token_utils.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |

**Clients and credentials**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `LLMClient` | Holds one client per provider (OpenAI, Anthropic, Google, OpenRouter) and sends each conversation to the client of its model's provider. Clones share the OpenAI credentials. | `arko_ai/src/client.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `LLMClient::new`, `impl Default for LLMClient` | A client with no provider configured; reads nothing from the environment. | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::from_env`, `LLMClient::from_env_with` | Configures every provider whose credentials are in the environment (`from_env_with` reads each variable through a function). | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_openai_api_key`, `with_anthropic_api_key`, `with_google_api_key`, `with_openrouter_api_key` | Configures one provider with an API key, replacing any client configured for it before. | `arko_ai/src/client.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `LLMClient::with_openai_chatgpt_oauth`, `with_openai_chatgpt_oauth_rate_limit_retries` | Configures OpenAI with ChatGPT OAuth credentials kept in memory; the second also sets the token-refresh rate-limit retries (off with the first). | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_codex_auth_file` | Configures OpenAI with the ChatGPT OAuth credentials in a Codex `auth.json`, kept in step with the file. | `arko_ai/src/client.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `LLMClient::with_openai_auth`, `LLMClient::update_openai_auth` | Installs an OpenAI client with the given `OpenAIAuth`; replaces the credentials of the configured OpenAI client in place, for every clone (`Error::MissingApiKey` without one). | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_openai_client`, `with_anthropic_client`, `with_google_client`, `with_openrouter_client` | Installs a preconfigured provider client. The `LLMClient`'s own settings (OpenAI trace, hook, `originator`, `User-Agent`; the stream idle timeout) are applied to it only where they are set. | `arko_ai/src/client.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `LLMClient::with_openai_originator`, `LLMClient::with_openai_user_agent` | Sends an `originator` or `User-Agent` header with every OpenAI request; none by default. | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_request_trace` | Writes every OpenAI request and response to a `RequestTrace` directory. | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_openai_request_hook` | Installs an `OpenAIRequestHook` on the OpenAI client. | `arko_ai/src/client.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `LLMClient::with_stream_idle_timeout` | Opt-in limit on the time between data on a streamed response, for every provider client. | `arko_ai/src/client.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `LLMClient::generate_text`, `LLMClient::generate_stream` | Sends a `Conversation` with a `GenerationConfig` to the model's provider; returns a `GenerationResponse`, or a stream of `StreamEvent`s ending with `End`. | `arko_ai/src/client.rs` | [Quick start](../arko_ai/README.md#quick-start) |
| `LLMClient::new_conversation` | `Conversation::new(model)`. | `arko_ai/src/client.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `LLMClient::has_provider`, `LLMClient::openrouter` | Whether a provider is configured; the configured `OpenRouterClient`, for its catalog. | `arko_ai/src/client.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `LLMClient::is_openai_chatgpt_oauth`, `openai_chatgpt_oauth_credentials`, `is_openai_token_expired` | Whether OpenAI uses ChatGPT OAuth; a snapshot of the current credentials (it waits for a refresh in flight); whether the access token expires within a margin (`true` when its expiry can't be read). | `arko_ai/src/client.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `LLMClient::refresh_openai_oauth_token`, `refresh_openai_oauth_token_once`, `refresh_openai_oauth_token_if_expiring` | Refreshes the ChatGPT OAuth credentials: explicitly (with the opt-in rate-limit retries), explicitly with one request, or only when the access token expires within a margin. | `arko_ai/src/client.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `LLMClient::prompt_cache_lifetime` | The prompt-cache lifetime of this client's requests for a model and config, using the client's OpenAI auth path. | `arko_ai/src/client.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `LLMClient::max_input_tokens` | The input-token limit of a model on this client's auth path (ChatGPT OAuth caps OpenAI models at 272,000). | `arko_ai/src/client.rs` | [ChatGPT OAuth (arko_ai)](#chatgpt-oauth-arko_ai) |
| `providers::openai::OpenAIClient`, `::new`, `::new_chatgpt_oauth`, `::with_auth` | The OpenAI provider client, with an API key, ChatGPT OAuth credentials, or an `OpenAIAuth`. | `arko_ai/src/providers/openai.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `OpenAIClient::with_base_url`, `with_oauth_token_url`, `with_stream_idle_timeout` | Sends requests to `{base_url}/responses`; sends token refreshes to another URL; sets or clears (`None`) the stream idle timeout. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIClient::with_oauth_rate_limit_retries`, `with_credential_store` | Opt-in retries of a rate-limited explicit token refresh; a `ChatGptOAuthStore` kept in step with the credentials. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIClient::with_request_trace`, `with_request_hook`, `with_originator`, `with_user_agent` | The client-level forms of the `LLMClient` settings; each takes an `Option`, so `None` clears it. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIClient::update_auth`, `is_chatgpt_oauth`, `chatgpt_oauth_credentials`, `is_oauth_token_expired`, `refresh_oauth_token`, `refresh_oauth_token_once`, `refresh_oauth_token_if_expiring` | The client-level forms of the `LLMClient` credential methods. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `providers::anthropic::AnthropicClient`, `::new`, `::with_base_url`, `::with_stream_idle_timeout` | The Anthropic provider client; requests go to `{base_url}/v1/messages`. | `arko_ai/src/providers/anthropic.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `providers::google::GoogleClient`, `::new`, `::with_base_url`, `::with_stream_idle_timeout` | The Gemini provider client; requests go to `{base_url}/interactions`. | `arko_ai/src/providers/google.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIAuth` (`ApiKey`, `ChatGptOAuth`) | OpenAI credentials: an API key, or `ChatGptOAuthCredentials`. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `ChatGptOAuthCredentials` (`access_token`, `refresh_token`, `account_id`) | ChatGPT OAuth credentials; `Debug` output redacts all three. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `OAuthRefreshResult` (`access_token`, `refresh_token`, `id_token`) | The tokens a refresh returned; `Debug` output redacts them. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `ChatGptOAuthStore` (`before_refresh`, `save_refreshed`, `refresh_lock`) | Extension trait: storage shared with other programs that a client re-reads before every refresh and writes rotated tokens to; `refresh_lock` defaults to `None`. | `arko_ai/src/providers/openai/credential_store.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `CodexAuthFile`, `::default_path`, `::load_default`, `::load`, `::path`, `::credentials` | A Codex `auth.json` (nested `tokens` or top-level token fields): its default location (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`), loading it (a relative path is made absolute), and its credentials. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `CodexAuthFile::openai_client`, `CodexAuthFile::client` | An `OpenAIClient` (or an `LLMClient` holding one) with the file's credentials and a `CodexAuthStore` of the file. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `CodexAuthFile::persist`, `persist_tokens`, `write_recovery`, `write_recovery_tokens`, `sync` | Writes credentials to the file (backup, then atomic replace; never over newer credentials another process wrote); writes them to a new `auth.json.recovered-<unix seconds>` file; reconciles the file with a client configured another way. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `CodexAuthStore`, `::new`, `::path`, `impl ChatGptOAuthStore for CodexAuthStore` | The `ChatGptOAuthStore` of a Codex `auth.json`; every store of the same file in a process shares one refresh lock. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `CodexAuthSync` (`Unchanged`, `Saved`, `Reloaded`) | The result of `CodexAuthFile::persist` and `sync`. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `refresh_chatgpt_oauth_token` | Exchanges a refresh token for new tokens once, with a caller-supplied HTTP client; `client_id` and `token_url` default to the Codex CLI's; 60-second timeout; refuses a refresh token this process knows is spent. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `parse_jwt_expiry`, `is_token_expired` | The `exp` claim of a JWT; whether it expires within a margin (`true` when it can't be read). | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials) |
| `http_client`, `http_client_builder` | The HTTP client (or its builder) the providers use: rustls with the OS trust store, HTTP/1.1, 30-second connect timeout, no total deadline, no redirects. | `arko_ai/src/providers/mod.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `RequestTrace`, `::new`, `::dir` | An opt-in trace directory for OpenAI requests and responses (`calls.jsonl` plus one request and response file per call). | `arko_ai/src/trace.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIRequestHook` (`prepare`, `observe_response`) | Extension trait: adjusts each OpenAI request's headers and body before sending, and observes response headers; both methods default to doing nothing. | `arko_ai/src/providers/openai/request_hook.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |

**Requests and responses**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `GenerationConfig`, `impl Default` | Per-request settings. Every field defaults to `None`; omitted fields deserialize to their defaults, so partial JSON or YAML is valid. A setting the model's provider has no use for is dropped with a warning. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::temperature`, `with_temperature` | Sampling temperature; `None` sends none. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::new` | Deprecated: `GenerationConfig::default().with_temperature(t)`. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::max_tokens`, `with_max_tokens` | Output-token limit; `None` means the model's own maximum. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::verbosity`, `with_verbosity`; `Verbosity` (`Low`, `Medium`, `High`) | Output verbosity (OpenAI GPT-5 and GPT-6, OpenRouter). | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::reasoning`, `with_reasoning` | The `ReasoningConfig` for the request; each provider takes its own variants. | `arko_ai/src/config.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `GenerationConfig::openai_reasoning_mode`, `with_openai_reasoning_mode` | OpenAI `reasoning.mode` (`standard`, `pro`) for GPT-5.6 and GPT-6 on API-key requests; ChatGPT OAuth omits it and warns once. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::anthropic_thinking_enabled`, `with_anthropic_thinking_enabled` | Turns Anthropic thinking on (adaptive where no reasoning config is given) or off. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::prompt_caching`, `with_prompt_caching` | Turns prompt caching on or off for the request; `None` inherits `Conversation::enable_cache`. | `arko_ai/src/config.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `GenerationConfig::anthropic_summarized_thinking`, `with_anthropic_summarized_thinking` | Requests summarized Claude thinking text on models that omit it by default. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::long_context`, `with_long_context` | Requests Claude's 1M-token context (standard on Claude 4.6 and later; nothing is sent). | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::tool_config`, `with_tool_config` | Tool definitions, tool choice and parallel-call setting for the request. | `arko_ai/src/config.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `GenerationConfig::interleaved_thinking`, `with_interleaved_thinking` | Claude interleaved thinking between tool calls. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::openai_service_tier`, `with_openai_service_tier`; `OpenAIServiceTier` (`Auto`, `Default`, `Flex`, `Fast`), `OpenAIServiceTier::pricing_tier` | OpenAI `service_tier` (`Fast` is sent as `priority`); API key only. `pricing_tier` maps it to the `ServiceTier` used for cost (`Auto` and `Default` price as `Standard`). | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::anthropic_cache_ttl`, `with_anthropic_cache_ttl` | `ttl` of the `cache_control` arko sends for Claude (`AnthropicCacheTtl`). | `arko_ai/src/config.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `GenerationConfig::anthropic_fast_mode`, `with_anthropic_fast_mode` | Anthropic fast mode (`speed: "fast"`), Opus 5.5, Opus 5 and Opus 4.8. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::anthropic_inference_geo`, `with_anthropic_inference_geo`; `AnthropicInferenceGeo` (`Global`, `Us`) | Anthropic `inference_geo`, Claude 4.6 and later. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::openrouter`, `with_openrouter_options` | Per-request `OpenRouterOptions`, merged over the client's default options. | `arko_ai/src/config.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `providers::ProviderClient` (`generate_text`, `generate_stream`) | The trait the four provider clients implement. `LLMClient` holds only its own concrete clients, so another implementation can't be plugged into it. | `arko_ai/src/providers/mod.rs` | [Providers and credentials](../arko_ai/README.md#providers-and-credentials) |
| `GenerationResponse` (`message`, `usage`, `model_call_usage`, `stop_reason`, `refusal`, `openai_turn_state`) | The result of a model call: the assistant message, its `UsageInfo` and `ModelCallUsage`, the normalized `StopReason`, a `Refusal` when one happened, and the ChatGPT backend's turn-state token. | `arko_ai/src/providers/mod.rs` | [Quick start](../arko_ai/README.md#quick-start) |
| `GenerationResponse::text`, `served_model`, `rerouted` | The concatenated message text; the model the provider reports serving; whether another model than requested served the call. | `arko_ai/src/providers/mod.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `GenerationResponse::single_tool_call` | The one call to a named tool in a response forced to call it; `Error::Refusal`, `Error::UnexpectedToolCalls` or `Error::InvalidToolArguments` otherwise. | `arko_ai/src/providers/mod.rs` | [Generation settings and tool calls](../arko_ai/README.md#generation-settings-and-tool-calls) |
| `StreamEvent` (`ReasoningChunk`, `TextChunk`, `Refusal`, `ToolCallStart { index, name }`, `ToolCallArgumentsDelta { index, delta }`, `End`) | Events of `generate_stream`: reasoning and text fragments, a refusal (once, just before `End`), native tool-call starts and argument fragments, and `End` with the complete `GenerationResponse`. | `arko_ai/src/providers/mod.rs` | [Quick start](../arko_ai/README.md#quick-start) |

**Reasoning**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ReasoningConfig::OpenAI { effort, include_summaries }`, `ReasoningConfig::openai` | OpenAI reasoning effort; `include_summaries: true` sends `reasoning.summary: "auto"`. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::Anthropic { budget_tokens, effort }`, `anthropic`, `anthropic_with_effort` | Claude manual thinking budget, with an optional effort. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::AnthropicAdaptive { effort }`, `anthropic_adaptive`, `anthropic_adaptive_with_effort` | Claude adaptive thinking (Claude 4.6 and later), with an optional effort. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::Gemini { thinking_level, include_summaries }`, `gemini` | Gemini thinking level. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::Google { thinking_budget, include_summaries }`, `google` | A Gemini thinking budget, mapped to a thinking level before sending. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::OpenRouter(OpenRouterReasoning)`, `openrouter`, `openrouter_effort` | OpenRouter's `reasoning` object; the only variant OpenRouter models take. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::to_openrouter` | The closest `OpenRouterReasoning` for another variant; `Error::InvalidRequest` where none exists (a budget with an effort; a negative Google budget other than -1). | `arko_ai/src/reasoning.rs` | [Reasoning](../arko_ai/README.md#reasoning) |
| `ReasoningConfig::is_enabled`, `ReasoningConfig::provider` | Whether the config turns reasoning on (false only for an Anthropic or Google budget of 0, and for an OpenRouter config with `enabled: false` or effort `none`); the `Provider` whose models take it. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `OpenAIEffort` (`None`, `Minimal`, `Low`, `Medium`, `High`, `XHigh`, `Max`) | OpenAI efforts, validated per model when the request is built. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `OpenAIReasoningMode` (`Standard`, `Pro`) | OpenAI reasoning mode (`GenerationConfig::openai_reasoning_mode`, `OpenRouterReasoning::mode`). | `arko_ai/src/reasoning.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `AnthropicReasoningEffort` (`Low`, `Medium`, `High`, `XHigh`, `Max`) | Claude efforts, validated per model. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `GeminiThinkingLevel` (`Minimal`, `Low`, `Medium`, `High`), `GeminiThinkingLevel::as_str` | Gemini thinking levels; saved as `"MINIMAL"` to `"HIGH"`, sent lowercase; `as_str` gives the uppercase form. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningEffort` (`None`, `Minimal`, `Low`, `Medium`, `High`, `XHigh`, `Max`), `ReasoningEffort::as_str`, `From<OpenAIEffort>`, `From<AnthropicReasoningEffort>`, `From<GeminiThinkingLevel>` | Provider-independent effort used by `OpenRouterReasoning` and mid-conversation effort updates; `as_str` is the wire value. | `arko_ai/src/reasoning.rs` | [Reasoning](../arko_ai/README.md#reasoning) |
| `ReasoningSummary` (`Auto`, `Concise`, `Detailed`), `ReasoningContext` (`Auto`, `AllTurns`, `CurrentTurn`) | OpenRouter `reasoning.summary` and `reasoning.context` values. | `arko_ai/src/reasoning.rs` | [Reasoning](../arko_ai/README.md#reasoning) |
| `OpenRouterReasoning` (`effort`, `max_tokens`, `exclude`, `enabled`, `summary`, `context`, `mode`), `::effort`, `::max_tokens`, `::enabled`, `::disabled`, `with_exclude`, `with_summary`, `with_context`, `with_mode`, `impl Default` | OpenRouter's `reasoning` object; unset fields are omitted, so the model's defaults apply; `effort` and `max_tokens` are exclusive. | `arko_ai/src/reasoning.rs` | [Reasoning](../arko_ai/README.md#reasoning) |
| `ReasoningArtifact` (`OpenAIEncryptedReasoning { items }`, `OpenAIOutputItems { items }`, `AnthropicReasoning { fragments }`, `GoogleReasoning { fragments }`, `OpenRouter { details }`), `ReasoningArtifact::provider` | Provider state kept on an assistant message so the turn can be replayed: OpenAI output items (or older encrypted reasoning items), Claude thinking blocks, Gemini thought signatures, OpenRouter `reasoning_details`. | `arko_ai/src/reasoning.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `ReasoningArtifact::new_openai_encrypted`, `new_anthropic_thinking`, `new_anthropic_redacted`, `new_anthropic_mixed`, `new_google_reasoning`, `new_openrouter` | Constructors of the artifact variants. | `arko_ai/src/reasoning.rs` | [Persisted data (arko_ai)](#persisted-data-arko_ai) |
| `reasoning::AnthropicReasoningFragment` (`Thinking { thinking, signature }`, `RedactedThinking { data }`), `reasoning::GoogleReasoningFragment` (`summary`, `signature`), `OpenAIReasoningItem` (`encrypted_content`, `summaries`, `id`) | The parts of the Claude, Gemini and older OpenAI artifacts. | `arko_ai/src/reasoning.rs` | [Persisted data (arko_ai)](#persisted-data-arko_ai) |

**Conversations and messages**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Conversation` (`model`, `system_message`, `enable_cache`, `prompt_cache_key`, `messages`, `openai_turn_state`) | The model, system message, caching switch, prompt-cache key, message history and ChatGPT turn state sent with each request. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::new`, `with_system_message`, `with_cache_disabled`, `with_prompt_cache_key` | A new conversation (caching on, a random UUID key); set the system message; turn caching off; set the key. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_user_message`, `add_user_message_with_image`, `add_user_message_with_contents` | Append a user message; one without tool results starts a new turn (and drops the ChatGPT turn-state token). | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_assistant_message`, `add_assistant_message_with_reasoning` | Append an assistant message, optionally with a `ReasoningArtifact`. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_assistant_message_from_response` | Append a response's message (not for classifier and content-filter refusals) and record its turn-state token. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_tool_results`, `add_tool_results_with_text` | Append a user message of tool results, optionally followed by text. | `arko_ai/src/conversation.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `Conversation::add_reasoning_effort_update`, `add_configuration_update` | Append a mid-conversation `ConfigurationUpdate` (OpenRouter only; replaces a pending update). | `arko_ai/src/conversation.rs` | [Mid-conversation effort updates](../arko_ai/README.md#mid-conversation-effort-updates) |
| `Conversation::current_openai_turn_state`, `record_openai_turn_state`; `OpenAITurnState` (`token`, `turn_start`, `turn_start_fingerprint`) | The ChatGPT `x-codex-turn-state` token the current turn echoes, recording one, and the saved token with the index and SHA-256 of the turn's starting message. | `arko_ai/src/conversation.rs` | [ChatGPT OAuth (arko_ai)](#chatgpt-oauth-arko_ai) |
| `Conversation::last_user_turn`, `rephrase_last_user_turn`, `remove_last_user_turn`, `redact_tool_result` | Edits for a refused input: find, rephrase or remove the last user turn (dropping later messages); replace one tool result's text (its images stay). `Error::InvalidRequest` when there is nothing to edit. | `arko_ai/src/conversation.rs` | [Stop reasons and refusals](../arko_ai/README.md#stop-reasons-and-refusals) |
| `Message` (`role`, `content`, `reasoning_artifact`, `tool_calls`, `id`, `configuration_update`) | One message of the history. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Message::new_user`, `new_user_with_image`, `new_user_with_contents`, `new_assistant`, `new_assistant_with_reasoning`, `new_system`, `new_configuration_update` | Message constructors. A `Role::System` message in `messages` is an error on OpenAI, Anthropic and Gemini (OpenRouter sends it as a system message); use `Conversation::system_message`. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Message::image_count`, `has_images`, `remove_images`, `has_tool_calls`, `is_configuration_update` | Counts and removes images, including those inside tool results (their text stays). | `arko_ai/src/message.rs` | [Tool images](../arko_ai/README.md#tool-images) |
| `Role` (`User`, `Assistant`, `System`) | Message roles. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Content` (`Text`, `Image`, `ToolResult`), `Content::as_text`, `as_image`, `as_tool_result` | Message content parts. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Image` (`Url`, `Base64 { data, mime_type }`), `Image::new_url`, `Image::new_base64` | Image input; Anthropic and Gemini accept only `Base64`. | `arko_ai/src/message.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `ConfigurationUpdate` (`reasoning_effort`), `ConfigurationUpdate::reasoning_effort` | A reasoning-effort change from a point of the conversation onward. | `arko_ai/src/message.rs` | [Mid-conversation effort updates](../arko_ai/README.md#mid-conversation-effort-updates) |

**Tools**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ToolDefinition` (`name`, `description`, `parameters`, `strict`), `ToolDefinition::new`, `with_strict` | A tool the model may call, with a JSON Schema for its parameters. Not the same type as arko_agents' `ToolDefinition`. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `ToolCall` (`id`, `name`, `arguments`, `thought_signature`, `provider_id`, `invalid_arguments`) | A call the model made; arguments that aren't valid JSON are kept as text in `invalid_arguments`, with `arguments` `null`. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `ToolResult` (`call_id`, `name`, `output`, `is_error`, `images`), `ToolResult::new`, `error`, `from_tool_call`, `error_from_tool_call`, `with_images` | The result of a call, matched by `call_id`, optionally with images. | `arko_ai/src/tool.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `ToolResultImage` (`label`, `image`), `ToolResultImage::new` | An image returned with a tool result, with a short label. | `arko_ai/src/tool.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `ToolChoice` (`Auto`, `Required`, `Specific`, `None`) | How the model selects tools. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `ToolConfig` (`tools`, `tool_choice`, `parallel_tool_calls`), `ToolConfig::new`, `with_tool_choice`, `with_parallel_tool_calls` | The tool setup of a request. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |

**Stop reasons and refusals**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `StopReason` (`EndTurn`, `ToolUse`, `MaxTokens`, `StopSequence`, `Refusal`, `PauseTurn`, `ContextWindowExceeded`, `Other`), `StopReason::is_truncated` | Why generation stopped, normalized across providers; `is_truncated` is true for `MaxTokens` and `ContextWindowExceeded`. | `arko_ai/src/stop.rs` | [Stop reasons and refusals](../arko_ai/README.md#stop-reasons-and-refusals) |
| `RefusalSource` (`SafetyClassifier`, `Model`, `ContentFilter`), `RefusalSource::keeps_turn` | Who refused; only a `Model` refusal keeps the turn in history. | `arko_ai/src/stop.rs` | [Stop reasons and refusals](../arko_ai/README.md#stop-reasons-and-refusals) |
| `Refusal` (`source`, `category`, `explanation`, `partial_output`, `provider_details`), `Refusal::new`, `impl Display` | A refusal's details; displayed as "refused by <source> (<category>): <explanation>". | `arko_ai/src/stop.rs` | [Stop reasons and refusals](../arko_ai/README.md#stop-reasons-and-refusals) |

**Models and pricing**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Provider` (`OpenAI`, `Anthropic`, `Google`, `OpenRouter`), `Provider::as_telemetry_str` | A model's provider; the telemetry names are `openai`, `anthropic`, `gemini`, `openrouter`. | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `model::ModelType` (`Standard`, `Reasoning`) | Whether a model reasons. On OpenAI it decides whether `temperature` is accepted and `reasoning.encrypted_content` is requested. | `arko_ai/src/model.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Model` (`provider`, `id`, `model_type`, `max_input_tokens`, `max_output_tokens`, `supports_image_input`, `supports_cache`, `pricing`, `openrouter`) | A model's identity, limits, capabilities and prices. `max_output_tokens` is what OpenAI (API key), Anthropic and Gemini requests send when `max_tokens` is unset. | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::list`, `Model::all`, `Model::find`, `Model::get` | The registered models of a provider (empty for OpenRouter); all of them; one by id; one by provider and id (`Error::ModelNotSupported` otherwise, always for OpenRouter). | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `providers::openai::list_models`, `providers::anthropic::list_models`, `providers::google::list_models` | The same lists as `Model::list` for each provider. | `arko_ai/src/providers/openai.rs`, `anthropic.rs`, `google.rs` | [Models](../arko_ai/README.md#models) |
| `Model::openrouter`, `Model::openrouter_info` | An OpenRouter model by slug, built without a network call (limits and prices 0, image input and caching assumed); the catalog metadata of a model taken from the OpenRouter catalog. | `arko_ai/src/model.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `Provider::ClaudeCode`, `Model::claude_code` | Anthropic models run through the user's installed Claude Code CLI; `Model::claude_code(model)` wraps a registered Anthropic model (same id, limits and list prices). `Model::list(Provider::ClaudeCode)` mirrors the Anthropic registry; `Model::all` and `Model::find` leave it out. `LLMClient` rejects these models (`Error::ModelNotSupported`). | `arko_ai/src/model.rs` | [Claude Code](../arko_ai/README.md#claude-code) |
| `arko_ai::claude_code` (`ClaudeSession`, `ClaudeOptions`, `SessionStart`, `SystemPrompt`, `BuiltInTools`, `Effort`, `PermissionMode`, `Hook`, `HookCallback`, `HookEvent`, `PermissionHandler`, `PermissionRequest`, `PermissionDecision`, `ToolServer`, `ClaudeTool`, `ToolOutput`, `Message`, `ContentBlock`, `ResultMessage`, `Turn`, `ClaudeError`) | A client for the installed Claude Code CLI over its stream-json control protocol: turns, resume, approvals, hooks, in-process MCP tools, settings and setting sources. | `arko_ai/src/providers/claude_code/` | [Claude Code](../arko_ai/README.md#claude-code) |
| `Model::openai_gpt_6_astra`, `openai_gpt_6_1_sol`, `openai_gpt_6_sol`, `openai_gpt_6_luna`, `openai_gpt_5_6_sol`, `openai_gpt_5_6_terra`, `openai_gpt_5_6_luna`, `openai_gpt_5_6` | Registered models `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, and `gpt-5.6` (an alias of GPT-5.6 Sol with its prices and limits). | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::openai_gpt_5_5`, `openai_gpt_5_5_pro`, `openai_gpt_5_4`, `openai_gpt_5_4_pro`, `openai_gpt_5_4_mini`, `openai_gpt_5_4_nano`, `openai_gpt_5_3_codex`, `openai_gpt_5_2`, `openai_gpt_5_2_pro`, `openai_gpt_5_1`, `openai_gpt_5`, `openai_gpt_5_pro`, `openai_gpt_5_mini`, `openai_gpt_5_nano` | Registered models `gpt-5.5-2026-04-23`, `gpt-5.5-pro-2026-04-23`, `gpt-5.4-2026-03-05`, `gpt-5.4-pro-2026-03-05`, `gpt-5.4-mini-2026-03-17`, `gpt-5.4-nano-2026-03-17`, `gpt-5.3-codex`, `gpt-5.2`, `gpt-5.2-pro`, `gpt-5.1-2025-11-13`, `gpt-5-2025-08-07`, `gpt-5-pro-2025-10-06`, `gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07`. | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::openai_o4_mini`, `openai_o3`, `openai_o3_pro`, `openai_gpt_4_1`, `openai_gpt_4_1_mini`, `openai_gpt_4_1_nano` | Registered models `o4-mini-2025-04-16`, `o3-2025-04-16`, `o3-pro-2025-06-10`, `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`, `gpt-4.1-nano-2025-04-14` (the three `gpt-4.1` models are the only `Standard` OpenAI models). | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::anthropic_claude_fable_5_1`, `anthropic_claude_fable_5`, `anthropic_claude_opus_5_5`, `anthropic_claude_opus_5`, `anthropic_claude_opus_4_8`, `anthropic_claude_sonnet_5_5`, `anthropic_claude_sonnet_5`, `anthropic_claude_opus_4_7`, `anthropic_claude_opus_4_6`, `anthropic_claude_sonnet_4_6` | Registered models `claude-fable-5-1`, `claude-fable-5`, `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-sonnet-5-5`, `claude-sonnet-5`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6` (the "Claude 4.6 and later" set). | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::anthropic_claude_opus_4_5`, `anthropic_claude_opus_4_1`, `anthropic_claude_opus_4`, `anthropic_claude_sonnet_4_5`, `anthropic_claude_sonnet_4`, `anthropic_claude_haiku_4_5`, `anthropic_claude_3_7_sonnet`, `anthropic_claude_3_5_sonnet_20241022`, `anthropic_claude_3_5_sonnet_20240620` | Registered models `claude-opus-4-5-20251101`, `claude-opus-4-1-20250805`, `claude-opus-4-20250514`, `claude-sonnet-4-5-20250929`, `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001`, `claude-3-7-sonnet-20250219`, `claude-3-5-sonnet-20241022`, `claude-3-5-sonnet-20240620`. | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::google_gemini_3_8_flash`, `google_gemini_3_7_flash`, `google_gemini_3_6_flash`, `google_gemini_3_5_flash`, `google_gemini_3_5_flash_lite`, `google_gemini_3_1_pro`, `google_gemini_3_1_pro_customtools`, `google_gemini_3_1_flash_lite_stable`, `google_gemini_3_1_flash_lite`, `google_gemini_3_0_pro`, `google_gemini_3_0_flash` | Registered models `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools`, `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-3-pro-preview`, `gemini-3-flash-preview`. | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::google_gemini_2_5_pro`, `google_gemini_2_5_flash`, `google_gemini_2_5_flash_lite`, `google_gemini_2_0_flash` | Registered models `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash` (the only Gemini model without thinking or caching). | `arko_ai/src/model.rs` | [Models](../arko_ai/README.md#models) |
| `Model::supports_forced_tool_choice`, `supports_disabling_parallel_tool_calls`, `reports_stop_reason`, `accepts_tool_result_images` | Whether a forced tool choice is accepted with a given config; whether `parallel_tool_calls: Some(false)` is (false only for Gemini); always true; whether images go inside tool results (OpenAI, Anthropic, Gemini 3 and later, with image input). | `arko_ai/src/model.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `Model::is_openai_gpt_6_astra`, `is_openai_gpt_6_sol_or_luna`, `is_openai_gpt_5_6`, `is_openai_gpt_5_5_pro`, `is_openai_gpt_5_2_pro`, `is_openai_gpt_5_pro`, `supports_openai_prompt_cache_options` | OpenAI model-id tests; `supports_openai_prompt_cache_options` is true for GPT-5.6 and GPT-6. | `arko_ai/src/model.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `Model::is_fable_5_1`, `is_fable_5`, `is_opus_5_5`, `is_opus_5`, `is_opus_4_8`, `is_opus_4_7`, `is_opus_4_6`, `is_opus_4_5`, `is_sonnet_5_5`, `is_sonnet_5`, `is_sonnet_4_6` | Claude model-id tests (exact ids). | `arko_ai/src/model.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Model::supports_anthropic_adaptive_thinking`, `defaults_anthropic_adaptive_thinking`, `always_uses_anthropic_adaptive_thinking`, `rejects_anthropic_manual_thinking`, `omits_anthropic_sampling_parameters`, `supports_anthropic_summarized_thinking_display`, `supports_anthropic_effort`, `supports_anthropic_xhigh_effort`, `supports_anthropic_max_effort`, `rejects_anthropic_forced_tool_choice`, `supports_anthropic_fast_mode`, `supports_anthropic_inference_geo`, `has_standard_anthropic_1m_context`, `had_retired_anthropic_long_context_beta`, `supports_long_context`, `supports_interleaved_thinking`, `auto_enables_interleaved_thinking` | The Claude capability rules the Anthropic provider (and, for forced tool choice, OpenRouter) checks before sending. | `arko_ai/src/model.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Model::is_gemini_3_pro`, `is_gemini_3_flash`, `is_gemini_3_1_pro`, `is_gemini_3_1_flash_lite`, `is_gemini_3_5_flash`, `is_gemini_3_5_flash_lite`, `is_gemini_3_6_flash`, `is_gemini_3_7_flash`, `is_gemini_3_8_flash`, `is_gemini_2_5_pro`, `is_gemini_2_5_flash`, `is_gemini_2_5_flash_lite`, `is_gemini_2_5`, `uses_gemini_thinking_levels` | Gemini model-id tests; `uses_gemini_thinking_levels` is true for the Gemini 3.x models. | `arko_ai/src/model.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `CHATGPT_OAUTH_MAX_INPUT_TOKENS`, `effective_max_input_tokens` | 272,000, the ChatGPT OAuth backend's input limit; a model's input limit on an auth path (`None` when unknown). Requests aren't checked against it. | `arko_ai/src/input_limit.rs` | [ChatGPT OAuth (arko_ai)](#chatgpt-oauth-arko_ai) |
| `ModelPricing` (`input_cost_per_million`, `output_cost_per_million`, `cached_input_cost_per_million`, `cache_write_cost_per_million`, `large_input_threshold`, `large_input_cost_per_million`, `large_output_cost_per_million`, `large_cached_input_cost_per_million`, `large_cache_write_cost_per_million`, `reasoning_output_cost_per_million`, `cache_write_1h_cost_per_million`, `large_cache_write_1h_cost_per_million`, `service_tiers`, `residency`, `schedule`), `impl Default` | List prices in USD per million tokens, long-context prices, tier and residency multipliers, and dated price changes. A token class without its own price is billed at the input rate; `reasoning_output_cost_per_million` is unused. | `arko_ai/src/model.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `ModelPricing::at`, `tier_multiplier`, `offers_tier`, `residency_multiplier` | The prices in effect at a time; a tier's multiplier (1.0 for unpriced tiers); whether a tier can be requested; a region's multiplier. | `arko_ai/src/pricing.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `ServiceTier` (`Standard`, `Flex`, `Fast`, `Priority`, `Scale`, `Batch`, `Other`), `ServiceTier::from_openai`, `ServiceTier::from_anthropic` | The processing tier used for pricing, normalized from OpenAI `service_tier` and Anthropic `service_tier`/`speed`. | `arko_ai/src/pricing.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `TierPricing` (`tier`, `multiplier`, `applies_to_long_context`), `::new`, `::flex`, `::fast`, `short_context_only` | A tier's price multiplier; `applies_to_long_context` defaults to `true`. | `arko_ai/src/pricing.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `ResidencyPricing` (`region`, `multiplier`), `::new`; `ScheduledPricing` (`effective_from`, `pricing`), `::new`, `effective_from_unix` | A multiplier for a reported region (Anthropic `inference_geo`); prices from a UTC date (`YYYY-MM-DD`). | `arko_ai/src/pricing.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |

**Prompt caching**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `PromptCacheLifetime` (`Known(Duration)`, `Unknown`, `Disabled`) | How long a provider keeps a request's prompt cache after its last use. | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `prompt_cache::prompt_cache_lifetime` | The lifetime for a model, config and auth path; reads `GenerationConfig::prompt_caching`, not `Conversation::enable_cache`. | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `prompt_cache::ANTHROPIC_PROMPT_CACHE_LIFETIME`, `ANTHROPIC_EXTENDED_PROMPT_CACHE_LIFETIME`, `OPENAI_PROMPT_CACHE_LIFETIME` | 5 minutes, 1 hour, and 5 minutes (a conservative value for every OpenAI path). | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `AnthropicCacheTtl` (`FiveMinutes` = `"5m"`, `OneHour` = `"1h"`), `AnthropicCacheTtl::duration` | Claude cache lifetimes and their durations. | `arko_ai/src/config.rs`, `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |

**Usage and cost**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `UsageInfo` (`input_tokens`, `output_tokens`, `input_cost`, `output_cost`, `cached_tokens`, `cache_read_cost`, `cache_write_tokens`, `cache_write_1h_tokens`, `cache_write_cost`, `reasoning_tokens`, `service_tier`, `inference_geo`, `total_cost`, `reported_cost`, `upstream_cost`, `billing`, `billed_cost`), `impl Default` | Tokens and cost of a call or a sum of calls. `input_tokens` includes cache reads and writes; `output_tokens` includes reasoning tokens; `total_cost` is the list-price estimate. | `arko_ai/src/usage.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `UsageInfo::billed` | What the account was charged: `billed_cost`, or `total_cost` for records saved before billing kinds existed. | `arko_ai/src/usage.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `UsageInfo::new`, `with_cached_tokens`, `with_cache_write_tokens`, `calculate_cost`, `calculate_cost_at`, `with_reported_cost`, `with_billing`, `for_embedding` | Constructors and cost computation: `calculate_cost_at` prices the tokens at the list prices in effect at a time, for the tier and residency that applied, and sets `billed_cost` from `billing`. | `arko_ai/src/usage.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `impl AddAssign for UsageInfo`, `impl Sum for UsageInfo` | Adds calls: counts saturate; tier and region become unknown and `billing` `Mixed` when the calls differ; `billed_cost` is the sum of `billed()`. | `arko_ai/src/usage.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `BillingKind` (`Metered` (default), `Subscription`, `ProviderReported`, `NotCharged`, `Mixed`) | How a call is paid for, which decides `billed_cost`: the estimate, zero (ChatGPT OAuth; Anthropic classifier refusals in unbilled categories), the provider's reported charge, or a sum. | `arko_ai/src/usage.rs` | [Usage and cost](../arko_ai/README.md#usage-and-cost) |
| `ModelCallUsage` (`call_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `max_input_tokens`, `context_usage_percent`, `input_delta_from_previous_call`, `stop_reason`, `stop_details`, `requested_model`, `upstream_provider`, `served_model`, `rerouted`, `plan_usage`) | Per-call telemetry: the serving model, token counts, the context share of the model's input limit, the raw stop reason, reroute information and plan usage. | `arko_ai/src/usage.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `ModelCallUsage::new`, `with_plan_usage`, `served_model_changed`, `with_served_model`, `with_input_delta`, `with_stop_metadata` | Constructors; `served_model_changed` is true when `requested_model` is set (OpenRouter) or `rerouted` (OpenAI). | `arko_ai/src/usage.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `PlanUsage` (`plan_type`, `active_limit`, `primary`, `secondary`), `PlanUsage::from_codex_headers`, `most_used_window`; `PlanUsageWindow` (`used_percent`, `window_minutes`, `resets_at`, `reset_after_seconds`) | ChatGPT plan usage parsed from the `x-codex-*` response headers; the window with the highest used percentage. | `arko_ai/src/usage.rs` | [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `ContextEvent` (`call_id`, `event_type`, `char_count`, `byte_count`, `token_count`), `ContextEvent::new`, `for_tool_call`, `for_tool_result`, `for_text`, `with_token_count`; `ContextEventType` (`System`, `UserMessage`, `AssistantMessage`, `AssistantToolCall`, `ToolResult`, `ToolDefinition`, `FileAttachment`) | A per-call record of context sizes. No arko crate creates or reads it. | `arko_ai/src/usage.rs` | — |

**Errors and retries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Error::ProviderError { message, status, code, retry_after }` | A provider's error response or error event, with its HTTP status (or a status its code stands for), code and delay hint. | `arko_ai/src/error.rs` | [Errors and retries](../arko_ai/README.md#errors-and-retries) |
| `Error::HttpError`, `Error::JsonError`, `Error::IoError` | Transport failures (connection, timeout, body); JSON failures; file I/O failures (Codex `auth.json`). | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `Error::ApiError` | An unexpected provider response (no `choices`, a missing field, a malformed stream event). | `arko_ai/src/error.rs` | [Errors and retries](../arko_ai/README.md#errors-and-retries) |
| `Error::AuthenticationFailed`, `Error::MissingApiKey` | Credential problems (partial `OPENAI_OAUTH_*` variables, an expired OAuth access token without a refresh token, Codex `auth.json` problems, a spent refresh token); a provider that isn't configured. | `arko_ai/src/error.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `Error::InvalidRequest`, `Error::UnsupportedFeature` | A request rejected before sending: a value the model can't take; a feature the provider or model lacks (images without image input, image URLs on Anthropic and Gemini, mid-conversation updates outside OpenRouter, ...). | `arko_ai/src/error.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Error::ProviderNotSupported`, `Error::ModelNotSupported` | A provider client given another provider's conversation (only when calling a provider client directly); an unknown model id (`Model::get`, an embedding model, an OpenRouter slug missing from a loaded catalog). | `arko_ai/src/error.rs` | [Models](../arko_ai/README.md#models) |
| `Error::Refusal` | A refusal raised as an error (Gemini blocked generation, OpenRouter refusal errors, `single_tool_call` on a refused response); classed `Permanent`. | `arko_ai/src/error.rs` | [Stop reasons and refusals](../arko_ai/README.md#stop-reasons-and-refusals) |
| `Error::StreamTruncated` | A stream that ended before the provider reported completion, or was cut off by the stream idle timeout. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `Error::UnexpectedToolCalls { expected, stop_reason, found }`, `Error::InvalidToolArguments { name, raw }` | `GenerationResponse::single_tool_call` found no call, several, or another tool; or the call's arguments aren't valid JSON. | `arko_ai/src/error.rs` | [Generation settings and tool calls](../arko_ai/README.md#generation-settings-and-tool-calls) |
| `Error::TokenizerError`, `TokenLimitExceeded`, `MissingHuggingFaceToken`, `BatchSizeLimitExceeded`, `InvalidDimension` | Embedding failures before sending: tokenizer loading or counting; an input or request over the token limit; a tokenizer that needs a token (no registered model does); more inputs than allowed; an unsupported dimension. | `arko_ai/src/error.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `Error::Base64Error` | A `From<base64::DecodeError>` conversion; arko_ai's own code doesn't produce it. | `arko_ai/src/error.rs` | — |
| `Error::retry_class`, `is_retryable`, `is_capacity_error`, `is_usage_limit`, `is_rate_limit`, `retry_after` | How a failure may be recovered; views of that class; the provider's delay hint. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `RetryClass` (`Permanent`, `Transient`, `Capacity`, `UsageLimit`), `RetryClass::is_retryable` | Recovery classes; every class but `Permanent` is retryable. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `BackoffPolicy` (`initial`, `max`, `jitter`), `BackoffPolicy::new`, `delay`, `next_delay`; `retry::random_sample` | Exponential backoff shared by the other crates: `new` turns jitter on; a server hint is a minimum. `random_sample` is the random input of `delay`. | `arko_ai/src/retry.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `retry::MAX_SERVER_DELAY_HINT` | 30 days, the cap on every provider delay hint. | `arko_ai/src/retry.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |

**Embeddings**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `EmbeddingClient`, `EmbeddingClient::new`, `impl Default` | The embeddings client (Voyage only); `new` configures no key. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingClient::with_voyage_api_key`, `from_env`, `with_voyage_client` | Configures Voyage with a key, from `VOYAGE_API_KEY` (empty counts as unset), or with a preconfigured `VoyageClient`. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingClient::embed` | Embeds texts after local checks (dimension, token counts, input count); vectors come back in input order. `Error::MissingApiKey` without a Voyage key. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingClient::token_counter`, `count_tokens`, `get_embedding_cost`, `get_embedding_cost_for_token_count` | A reusable `TokenCounter` for a config; the token count of one text; the list-price cost of texts or of a token count (no request is made). | `arko_ai/src/embedding_client.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `EmbeddingConfig` (`provider`, `model_id`, `output_dimensionality`, `voyage_input_type`, `voyage_truncation`, `hf_token`, `tokenizer_revision`), `EmbeddingConfig::new` | An embedding request's model and options; `new` sets every option to `None`. `Debug` output redacts `hf_token`. | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingConfig::with_output_dimensionality`, `with_voyage_input_type`, `with_voyage_truncation`, `with_tokenizer_revision`, `with_hf_token` | Setters of the options. | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `VoyageInputType` (`Query`, `Document`), `VoyageInputType::as_str` | Voyage `input_type` (`"query"`, `"document"`). | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingModelDetails` (`provider`, `id`, `api_id`, `max_input_tokens_per_item`, `max_batch_total_tokens`, `max_batch_size`, `tokenizer_id`, `requires_hf_token`, `cost_per_million_input_tokens`, `default_dimensions`, `supported_dimensions`), `EmbeddingModelDetails::get` | An embedding model's limits, tokenizer and price; `get` looks one up (`Error::ModelNotSupported` otherwise). | `arko_ai/src/embedding_model.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `EmbeddingModelDetails::voyage_4_large`, `voyage_4`, `voyage_4_lite`, `voyage_code_4`, `voyage_3_5`, `voyage_3_large`, `voyage_3_5_lite`, `voyage_code_3` | Registered models `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3.5`, `voyage-3-large`, `voyage-3.5-lite`, `voyage-code-3`: 32,000 tokens per input, 1,000 inputs per request, 1,024 dimensions by default (256, 512, 1,024, 2,048 supported). | `arko_ai/src/embedding_model.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `EmbeddingProviderIdentifier` (`Voyage`) | The embeddings provider (Voyage is the only one). | `arko_ai/src/embedding_provider_identifier.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `providers::EmbeddingResponse` (`embeddings`, `usage`) | The vectors, in input order, and their `UsageInfo`. | `arko_ai/src/providers/mod.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `providers::EmbeddingProviderTrait` (`embed_texts`), `impl EmbeddingProviderTrait for VoyageClient` | The trait `VoyageClient` implements; `EmbeddingClient` holds only a `VoyageClient`, so another implementation can't be plugged into it. | `arko_ai/src/providers/mod.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `providers::voyage::VoyageClient`, `::new`, `::with_base_url` | The Voyage client; requests go to `{base_url}/embeddings` (default `https://api.voyageai.com/v1`). | `arko_ai/src/providers/voyage.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `TokenCounter`, `TokenCounter::count` | Counts a text's tokens with a loaded tokenizer (no padding, truncation or special tokens). | `arko_ai/src/token_utils.rs` | [Voyage embeddings](../arko_ai/README.md#voyage-embeddings) |
| `count_tokens`, `token_utils::count_tokens_with_revision`, `token_utils::count_tokens_for_text` | Counts texts' tokens with a Hugging Face tokenizer repository (at `main`, or at a revision), loading and caching it per repository and revision. | `arko_ai/src/token_utils.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |

**OpenRouter**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `OpenRouterClient`, `OpenRouterClient::new`, `providers::openrouter::OPENROUTER_BASE_URL` | The OpenRouter client (clones share the loaded catalog); chat requests go to `{base_url}/chat/completions`, default `https://openrouter.ai/api/v1`. | `arko_ai/src/providers/openrouter.rs` | [OpenRouter](../arko_ai/README.md#openrouter) |
| `OpenRouterClient::with_base_url`, `with_stream_idle_timeout` | Another base URL; sets or clears (`None`) the stream idle timeout. | `arko_ai/src/providers/openrouter.rs` | [OpenRouter](../arko_ai/README.md#openrouter) |
| `OpenRouterClient::with_default_options`, `default_options` | Options applied to every request, which `GenerationConfig::openrouter` overrides field by field. | `arko_ai/src/providers/openrouter.rs` | [Options](../arko_ai/README.md#options) |
| `OpenRouterClient::with_app_attribution`; `AppAttribution` (`url`, `title`, `categories`, `hidden`), `AppAttribution::new`, `with_title`, `with_categories`, `hidden` | Opt-in app attribution headers: `HTTP-Referer`, `X-OpenRouter-Title`, `X-OpenRouter-Categories`, `X-OpenRouter-App-Visibility: hidden`. | `arko_ai/src/providers/openrouter.rs`, `openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `OpenRouterClient::fetch_models`, `fetch_user_models`, `set_catalog`, `catalog` | Fetches the public catalog (`GET /models`, no key sent) or the key's catalog (`GET /models/user`) and keeps it; installs or clears a catalog; the loaded catalog. | `arko_ai/src/providers/openrouter.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `OpenRouterClient::model`, `OpenRouterClient::resolve_model` | The catalog's model for a slug; the catalog's model, or `Model::openrouter(slug)` without a catalog (`Error::ModelNotSupported` for a slug the loaded catalog lacks; slugs starting with `@` skip the check). | `arko_ai/src/providers/openrouter.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `OpenRouterCatalog`, `OpenRouterCatalog::from_json`, `from_models`, `models`, `source`, `fetched_at`, `get`, `contains`; `CatalogSource` (`Public`, `User`, `Supplied`) | A parsed model catalog: from a saved `/models` response or from models, their list, where it came from and when; lookup by slug (variants and `model@preset/...` resolve to the base model, which keeps the requested slug). | `arko_ai/src/providers/openrouter/catalog.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `OpenRouterModelInfo` (`name`, `canonical_slug`, `alias_target`, `context_length`, `max_completion_tokens`, `input_modalities`, `output_modalities`, `supported_parameters`, `reasoning`, `pricing_known`, `expiration_date`), `supports_parameter`, `accepts_images` | Catalog metadata of a model; an empty `supported_parameters` list counts as accepting every parameter. | `arko_ai/src/providers/openrouter/catalog.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `providers::openrouter::OpenRouterReasoningInfo` (`effort_selectable`, `supported_efforts`, `default_effort`, `default_enabled`, `mandatory`, `supports_max_tokens`) | A catalog model's reasoning controls. | `arko_ai/src/providers/openrouter/catalog.rs` | [Models and the catalog](../arko_ai/README.md#models-and-the-catalog) |
| `OpenRouterOptions` (`provider`, `fallback_models`, `auto_router`, `auto_exacto`, `user`, `service_tier`), `OpenRouterOptions::new`, `with_provider`, `with_fallback_models`, `with_auto_router`, `with_auto_exacto`, `with_user`, `with_service_tier`, `merged_over` | Routing and request options; unset fields aren't sent. `merged_over` lays request options over defaults: `provider` and `auto_router` merge field by field, every other field replaces the default whole. | `arko_ai/src/providers/openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `providers::openrouter::OpenRouterServiceTier` (`Auto`, `Default`, `Flex`, `Priority`, `Scale`) | OpenRouter `service_tier` values. | `arko_ai/src/providers/openrouter/options.rs` | [Options](../arko_ai/README.md#options) |
| `AutoExactoPolicy` (`CacheAware` (default), `AllowAutoExacto`, `DisableAutoExacto`) | What arko does about Auto Exacto on tool requests. | `arko_ai/src/providers/openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `ProviderPreferences` (`order`, `only`, `ignore`, `allow_fallbacks`, `require_parameters`, `data_collection`, `zdr`, `enforce_distillable_text`, `quantizations`, `sort`, `max_price`, `preferred_min_throughput`, `preferred_max_latency`, `options`) | OpenRouter's `provider` routing object; unset fields are omitted. | `arko_ai/src/providers/openrouter/options.rs` | [Options](../arko_ai/README.md#options) |
| `ProviderPreferences::new`, `pinned`, `zdr_only`, `with_order`, `with_only`, `with_ignore`, `with_allow_fallbacks`, `with_require_parameters`, `with_data_collection`, `with_zdr`, `with_enforce_distillable_text`, `with_quantizations`, `with_sort`, `with_max_price`, `with_preferred_min_throughput`, `with_preferred_max_latency`, `with_provider_options`, `merged_over`, `is_empty` | Builders; `pinned(slug)` sets `only` plus `allow_fallbacks: false`; `zdr_only()` sets `zdr: true`; `merged_over` takes each set field over the base. | `arko_ai/src/providers/openrouter/options.rs` | [Options](../arko_ai/README.md#options) |
| `DataCollection` (`Allow`, `Deny`), `providers::openrouter::Quantization` (`Int4`, `Int8`, `Fp4`, `Mxfp4`, `Nvfp4`, `Fp6`, `Fp8`, `Mxfp8`, `Fp16`, `Bf16`, `Fp32`, `Unknown`), `SortBy` (`Price`, `Throughput`, `Latency`, `Exacto`), `providers::openrouter::SortPartition` (`Model`, `None`), `providers::openrouter::ProviderSort` (`By`, `Config { by, partition }`), `ProviderSort::by`, `From<SortBy> for ProviderSort` | Values of the `provider` fields. | `arko_ai/src/providers/openrouter/options.rs` | [Options](../arko_ai/README.md#options) |
| `providers::openrouter::MaxPrice` (`prompt`, `completion`, `request`, `image`, `audio`), `MaxPrice::per_million_tokens`; `providers::openrouter::PerformanceThreshold` (`P50`, `Percentiles { p50, p75, p90, p99 }`) | `provider.max_price` (sent as JSON numbers) and throughput or latency thresholds. | `arko_ai/src/providers/openrouter/options.rs` | [Options](../arko_ai/README.md#options) |
| `AutoRouterOptions` (`allowed_models`, `excluded_models`, `cost_tier`, `provider`), `AutoRouterOptions::new`, `zdr`, `with_allowed_models`, `with_excluded_models`, `with_cost_tier`, `with_provider`, `merged_over`; `AutoRouterCostTier` (`Low`, `Medium`, `High`, `XHigh`, `Max`) | Auto Router settings, used only for `openrouter/auto` and `openrouter/auto-beta` and ignored for other models. | `arko_ai/src/providers/openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |

**Example binaries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `llm_client_cli` `text`, `chat`, `models` | One-shot generation (optionally streamed, with an image), an interactive streamed chat, and the list of registered models with the aliases the CLI accepts. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `-m`, `--model`; `-s`, `--system`; `-t`, `--temperature`; `-n`, `--max-tokens`; `--long-context` | The model (registered id, alias, or OpenRouter slug) and generation settings. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `-r`, `--reasoning`; `--reasoning-level`; `--reasoning-effort`; `--reasoning-mode` | Reasoning at the model's defaults, or an explicit effort, level, budget or mode. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `text -p`, `--prompt`; `text -i`, `--image PATH`; `text --stream` | The prompt, an image file, and streaming for `text`. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `--codex-auth`, `--codex-auth-json PATH` | OpenAI ChatGPT OAuth credentials from a Codex `auth.json` instead of the environment. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `embeddings_demo` | Embeds two texts with `voyage-4-lite` at 256 dimensions; takes no flags. | `arko_ai/examples/embeddings_demo.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `oauth_cache_probe` `--out`, `--auth-file`, `--conversations`, `--turns`, `--idle-gap`, `--model`, `--effort`, `--label`, `--summarize-only` | Measures ChatGPT OAuth prompt-cache hits with a traced tool-loop workload. | `arko_ai/examples/oauth_cache_probe.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `oauth_cache_probe` `--legacy`, `--session-id-header`, `--thread-id-header`, `--client-request-id-header`, `--echo-turn-state`, `--no-legacy-beta-header`, `--replay-output-items`, `--reasoning-context` | Request-shape experiments: the provider as it was before its Codex-parity changes, with single pieces turned back on. | `arko_ai/examples/oauth_cache_probe.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |

### Client and credentials (arko_ai)

| Setting | Default | Notes |
|---|---|---|
| `LLMClient::new()` | No provider | A request to a provider without a client fails with `Error::MissingApiKey`. The client reads no environment variables unless `from_env` is called. |
| `LLMClient::from_env()` | — | Reads `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (and optionally `OPENAI_OAUTH_REFRESH_TOKEN`), otherwise `OPENAI_API_KEY`; `ANTHROPIC_API_KEY`; `GOOGLE_API_KEY`, otherwise `GEMINI_API_KEY`; `OPENROUTER_API_KEY`. Empty or blank values count as unset. An `OPENAI_OAUTH_*` variable set without both the access token and the account ID is `Error::AuthenticationFailed`, never a fallback to the API key. `from_env_with(lookup)` reads through a function instead. |
| Refresh token from the environment | Kept in memory | If it matches the refresh token in the Codex `auth.json` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`) for the same account (compared by SHA-256), the client uses that file as its credential store instead, as `with_codex_auth_file` does. Otherwise the first rotation logs a warning (once per process), since the environment's token stops working once used. |
| `with_openai_chatgpt_oauth(access, refresh, account)` | In memory only | Rotated tokens are not saved and no warning is logged; save them from `openai_chatgpt_oauth_credentials()`. `with_openai_auth(OpenAIAuth::ChatGptOAuth(..))` behaves the same. |
| `with_codex_auth_file(&file)`, `CodexAuthFile::client()` | — | Every refresh re-reads `auth.json` first (adopting tokens another program refreshed, for the same account only), and rotated tokens are saved at once. See [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials). |
| Credentials without a refresh token | — | The access token is used until the server rejects it as expired (HTTP 401 `token_expired`), which then fails with `Error::AuthenticationFailed`; any other 401 is `Error::ProviderError` with status 401. `refresh_openai_oauth_token_if_expiring` returns `None` for them. |
| Token-refresh rate-limit retries | Off | `with_openai_chatgpt_oauth_rate_limit_retries(.., true)` or `OpenAIClient::with_oauth_rate_limit_retries(true)`: an explicit `refresh_openai_oauth_token()` retries an HTTP 429 up to 8 times, with jittered waits doubling from 30 s to 15 min; a server hint, capped at 15 min, is a minimum. The refresh after a 401, `refresh_openai_oauth_token_once` and `refresh_openai_oauth_token_if_expiring` (which the arko_agents runtime uses) never wait, so the setting does nothing for a client that only runs agents. |
| `with_openai_originator`, `with_openai_user_agent` | Not sent | No `originator` or `User-Agent` header by default; the shared HTTP client sets no `User-Agent`. Token-refresh requests never carry them. |
| `with_request_trace(RequestTrace::new(dir))` | Off | Writes every OpenAI model request and response (full prompts; credential headers redacted, `x-codex-turn-state` hashed) with default file permissions. Token refreshes and other providers are not traced; the first request to each other provider logs a warning. |
| `with_openai_request_hook(hook)` | None | `prepare` can change the headers (the `Authorization` header included) and body of each attempt before sending; `observe_response` gets every response's headers, never its body. |
| `OpenAIClient::with_credential_store(Some(store))`, `with_oauth_token_url(url)` | No store; `https://auth.openai.com/oauth/token` | A `ChatGptOAuthStore` keeps rotated tokens in step with other storage (`CodexAuthStore` for a Codex `auth.json`). `LLMClient::with_openai_auth(OpenAIAuth)` installs either kind of credentials, without a store. |
| `with_stream_idle_timeout(duration)` | Off | Applies to every provider client of the `LLMClient`, including clients installed later, over their own setting; it can't be cleared on the `LLMClient` again (the provider clients' setters take `None`). See [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai). |
| Provider clients | Base URLs: `https://api.openai.com/v1` (API key), `https://chatgpt.com/backend-api/codex` (ChatGPT OAuth), `https://api.anthropic.com`, `https://generativelanguage.googleapis.com/v1`, `https://openrouter.ai/api/v1`, `https://api.voyageai.com/v1` | Requests go to `/responses` (OpenAI Responses API), `/v1/messages` (Anthropic, `anthropic-version: 2023-06-01`, no beta header by default), `/interactions` (Gemini Interactions API; `?alt=sse` when streaming), `/chat/completions` (OpenRouter Chat Completions API) and `/embeddings` (Voyage). Each client takes `with_base_url` (one base for both OpenAI auth kinds); install them with `with_openai_client` and its siblings. |

The HTTP transport is fixed: a 30-second connect timeout, no total request deadline, no
redirects (a 3xx response is an error, so credentials never reach another host), TCP
keep-alive every 30 seconds, pooled connections closed after 90 seconds idle, HTTP/1.1,
rustls with the operating system's trust store, and the environment's and system's
proxy settings (`HTTPS_PROXY` and the like).

### GenerationConfig (arko_ai)

Every field defaults to `None`, and omitted fields take their defaults, so a partial JSON
or YAML config is valid. One config can serve models of several providers: a setting
the model's provider has no use for is dropped with a warning (logged once per process
for each provider and setting), while a value the provider can't accept for a setting it
does use is an error before anything is sent. Exceptions: `temperature` and
`parallel_tool_calls: Some(false)` on Gemini, and `openai_service_tier` over ChatGPT
OAuth, are `Error::UnsupportedFeature`; Gemini ignores `ToolDefinition::strict` without a
warning; Claude on OpenRouter accepts `anthropic_summarized_thinking` and `long_context`
without sending anything.

| Field | `None` means | Allowed values and notes |
|---|---|---|
| `temperature` | Not sent: the model's default | `Error::InvalidRequest` on OpenAI reasoning models (every registered OpenAI model except the `gpt-4.1` ones); `Error::UnsupportedFeature` on Gemini. Dropped with a warning (once per model) on Claude models that take no sampling parameters (Fable 5.x, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5, Sonnet 5); other Claude models accept only 1.0 while thinking. Sent to OpenRouter as given. `GenerationConfig::new(t)` is deprecated for `default().with_temperature(t)`. |
| `max_tokens` | The model's own maximum: OpenAI with an API key, Anthropic and Gemini send the model's registered output limit; OpenRouter and ChatGPT OAuth send none | A set value is sent as given, not checked against the model's limit. Over ChatGPT OAuth it is never sent (the backend rejects it): a set value is ignored with a warning logged once. On Claude through OpenRouter it must exceed a `reasoning.max_tokens` budget. |
| `verbosity` | Not sent | `low`, `medium`, `high`. OpenAI GPT-5.x and GPT-6 models (by model id), on both auth paths; `Error::InvalidRequest` on other OpenAI models. Sent to OpenRouter, where on Claude it can't be combined with `reasoning.effort` (`Error::InvalidRequest`). Dropped with a warning on Anthropic and Gemini. |
| `reasoning` | The model's default; see [Reasoning (arko_ai)](#reasoning-arko_ai) | Each provider takes its own `ReasoningConfig` variants; another provider's variant is dropped with a warning. |
| `openai_reasoning_mode` | Not sent | `standard`, `pro`, sent as `reasoning.mode` to GPT-5.6 and GPT-6 on API-key requests; ChatGPT OAuth omits it and warns once. `Error::InvalidRequest` on other OpenAI models; dropped with a warning elsewhere (on OpenRouter, set `OpenRouterReasoning::mode`). |
| `anthropic_thinking_enabled` | The model's default | `Some(true)` without an Anthropic reasoning config sends adaptive thinking (`Error::InvalidRequest` on models before Claude 4.6). `Some(false)` sends `thinking.type: "disabled"`; it is `Error::InvalidRequest` on Fable 5.x, Opus 5.5 and Sonnet 5.5 (arko does not send Sonnet 5.5's `between_tools`), together with an Anthropic reasoning config, and with `anthropic_summarized_thinking: Some(true)`. Dropped with a warning on the other providers, OpenRouter included. |
| `prompt_caching` | `Conversation::enable_cache` (default `true`) | See [Prompt caching (arko_ai)](#prompt-caching-arko_ai). |
| `anthropic_summarized_thinking` | The model's default display | `Some(true)` sends `thinking.display: "summarized"` on models that omit thinking text by default (Fable 5.x, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5, Sonnet 5), nothing on others; `Error::InvalidRequest` without active thinking. On a model that thinks adaptively by default, `Some(true)` makes arko send the adaptive `thinking` object explicitly. `Some(false)` sends nothing. |
| `long_context` | Nothing sent | `Some(true)` is accepted on Claude 4.6 and later, where 1M context is standard and nothing extra is sent; `Error::InvalidRequest` on other Claude models, Sonnet 4 and 4.5 included. `Some(false)` sends nothing. |
| `tool_config` | No tools | See [Tool calls (arko_ai)](#tool-calls-arko_ai). |
| `interleaved_thinking` | Automatic on Claude 4.6 and later while adaptive thinking is active | On the Anthropic provider, `Some(true)` needs a model that supports it and active thinking (not a manual budget on Opus 4.6); with a manual budget it sends the `interleaved-thinking-2025-05-14` beta header, with adaptive thinking nothing. `Some(false)` is `Error::InvalidRequest` while adaptive thinking is active on 4.6 and later, and sends nothing otherwise. For Claude on OpenRouter, `Some(true)` adds the beta header and `Some(false)` is `Error::UnsupportedFeature`. |
| `openai_service_tier` | Not sent: the project's default tier | `auto`, `default`, `flex`, `priority` (the `Fast` variant). Flex and Fast only on models that offer them (`Error::InvalidRequest` otherwise); API key only (`Error::UnsupportedFeature` over ChatGPT OAuth). |
| `anthropic_cache_ttl` | No `ttl`: 5 minutes | `"5m"`, `"1h"` (1-hour cache writes cost 2x base input). `Error::InvalidRequest` while caching is off. Also used for Claude on OpenRouter. |
| `anthropic_fast_mode` | Nothing sent | `Some(true)` sends `speed: "fast"` with the `fast-mode-2026-02-01` beta header on Opus 5.5, Opus 5 and Opus 4.8 (`Error::InvalidRequest` on other models); billed at fast-mode rates. |
| `anthropic_inference_geo` | Not sent: the workspace's default | `global`, `us` (1.1x price); Claude 4.6 and later (`Error::InvalidRequest` on older models). |
| `openrouter` | The client's default options | `OpenRouterOptions`, merged over `OpenRouterClient::with_default_options`; see [OpenRouter (arko_ai)](#openrouter-arko_ai). |

### Reasoning (arko_ai)

What is sent without `GenerationConfig::reasoning`:

| Provider | Sent |
|---|---|
| OpenAI, API key | No effort, so the model's default applies (none on GPT-5.1, 5.2 and 5.4, which then don't reason). GPT-5.6 reasoning models get `reasoning.context: "all_turns"`; GPT-6 gets no `reasoning` object. |
| OpenAI, ChatGPT OAuth | `effort: "medium"` on GPT-5.x and GPT-6 models; `context: "all_turns"` on GPT-5.6 and GPT-6. |
| Anthropic | Nothing: the model's default. Fable 5.x, Opus 5.5, Opus 5, Sonnet 5.5 and Sonnet 5 think adaptively by default (Fable 5.x, Opus 5.5 and Sonnet 5.5 can't turn it off); earlier models don't think. |
| Gemini | No `thinking_level` or `thinking_summaries`: the model's default level (medium on Gemini 3.5 to 3.8 Flash, minimal on 3.1 and 3.5 Flash-Lite, high on 3 Pro, 3 Flash and 3.1 Pro) and no thought summaries. |
| OpenRouter | Nothing: the model's default from OpenRouter. |

Every OpenAI request sets `store: false`, and reasoning models also ask for
`reasoning.encrypted_content`, which later requests replay (reasoning that nothing
follows is left out).

| `ReasoningConfig` | Values and checks |
|---|---|
| `openai(effort, include_summaries)` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, validated per model: an effort the model lacks, or any reasoning config on the `gpt-4.1` models, is `Error::InvalidRequest`. `include_summaries: true` sends `reasoning.summary: "auto"`. |
| `anthropic(budget_tokens)`, `anthropic_with_effort(budget, effort)` | Manual budget: at least 1,024 (0 is an error too) and below `max_tokens` unless `interleaved_thinking` is `Some(true)`; at most 52,000 on Opus 4.5 and Sonnet 4.6 and 102,400 on Opus 4.6; `Error::InvalidRequest` on Fable 5.x, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5 and Sonnet 5. The effort is sent as `output_config.effort`: `low` to `high` on Opus 4.5, `max` too on Opus 4.6 and Sonnet 4.6, all five on Fable 5.x, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5 and Sonnet 5, none on older models. |
| `anthropic_adaptive()`, `anthropic_adaptive_with_effort(effort)` | Claude 4.6 and later (`Error::InvalidRequest` on others); efforts as above. |
| `gemini(level, include_summaries)` | `minimal`, `low`, `medium`, `high` (`"MINIMAL"` to `"HIGH"` in saved JSON; sent lowercase), validated per model with `Error::InvalidRequest`: 3 Pro rejects minimal and medium; 3.1 Pro, 3.7 Flash, 3.8 Flash and 2.5 reject minimal; 2.0 Flash rejects any level. `thinking_summaries` is `"auto"` with summaries and `"none"` without. |
| `google(budget, include_summaries)` | Mapped to a `thinking_level`, then validated as above (no budget is ever sent): negative = no level (the model's default), 0 = minimal (on 2.5 Pro and 2.5 Flash an error, on 2.5 Flash-Lite no level), up to 4,096 = low, up to 16,384 = medium, above = high. `thinking_summaries` is sent either way. |
| `openrouter(OpenRouterReasoning)`, `openrouter_effort(effort)` | The only variant OpenRouter takes; `effort` or `max_tokens`, not both (`Error::InvalidRequest`). `ReasoningConfig::to_openrouter()` converts the other variants explicitly. |

### Tool calls (arko_ai)

| Setting | Default | Notes |
|---|---|---|
| `ToolConfig::tool_choice` | `None`: nothing sent to OpenAI, OpenRouter or Gemini, so the provider's `auto` applies; Anthropic sends `{"type": "auto"}` | `Auto`, `Required`, `None`, `Specific(name)`. A forced choice (`Required`, `Specific`) is `Error::InvalidRequest` before sending on Claude Fable 5.1, Opus 5.5 and Sonnet 5.5 and with a manual thinking budget, on the Anthropic provider and on OpenRouter for `anthropic/` slugs of registered Claude models (where OpenRouter reasoning counts as a budget on Claude before 4.6, and `reasoning.max_tokens` on Opus 4.6 and Sonnet 4.6); with an OpenRouter catalog, also on models that don't list `tool_choice`. `Model::supports_forced_tool_choice(config)` tells in advance. |
| `ToolConfig::parallel_tool_calls` | `None`: the provider's default, several calls allowed | `Some(false)`: OpenAI and OpenRouter send `parallel_tool_calls: false`, Anthropic `disable_parallel_tool_use: true` inside `tool_choice` (not with `ToolChoice::None`); Gemini fails with `Error::UnsupportedFeature`. |
| `ToolDefinition::strict` | `None`: not sent | Sent to OpenAI, Anthropic and OpenRouter (for Claude on OpenRouter with the `structured-outputs-2025-11-13` beta header); Gemini ignores it without a warning. |
| Arguments that aren't valid JSON | — | Kept as text in `ToolCall::invalid_arguments` with `arguments: null`; replayed verbatim to OpenAI and to OpenRouter `openai/` models, as `{}` to the others. |

### Conversation (arko_ai)

| Field | Default | Notes |
|---|---|---|
| `enable_cache` | `true` | `with_cache_disabled()`. Required when loading JSON. |
| `prompt_cache_key` | A random UUID per new conversation | `with_prompt_cache_key`. Sent as OpenAI's `prompt_cache_key` (both auth paths), as ChatGPT OAuth's `session-id`, `thread-id` and `x-client-request-id` headers, and as OpenRouter's `session_id` (at most 256 characters, else `Error::InvalidRequest`). A loaded conversation keeps its key; one saved without a key gets a new one. Give each agent or task its own conversation. |
| `system_message` | None | `with_system_message`. A `Role::System` message inside `messages` is `Error::InvalidRequest` on OpenAI, Anthropic and Gemini; OpenRouter sends it as a system message. |
| `openai_turn_state` | None | The ChatGPT OAuth turn-state token of the current user turn, recorded by `add_assistant_message_from_response` and saved with the conversation. |

`add_assistant_message_from_response` leaves out classifier and content-filter refusals,
which must not stay in history; a model's own refusal stays. History that holds another
provider's reasoning artifacts is `Error::InvalidRequest` on Anthropic and Gemini, so a
conversation can't switch to them mid-way; OpenAI drops such artifacts silently and
OpenRouter with a debug log, sending the turn's text and tool calls. A mid-conversation
reasoning-effort update is `Error::UnsupportedFeature` except on OpenRouter.

### Prompt caching (arko_ai)

Caching is on by default. Turning it off (`with_prompt_caching(false)` or
`Conversation::with_cache_disabled()`) takes effect only where the provider has a
switch:

| Provider | Caching | With caching off |
|---|---|---|
| Anthropic, and Claude on OpenRouter | Only through the top-level `cache_control` arko sends | Nothing is cached |
| OpenAI GPT-5.6 and GPT-6, API key | `prompt_cache_options.mode: "implicit"` | `"explicit"` with no breakpoints: nothing is cached |
| OpenAI before GPT-5.6, API key | Automatic | Still cached; can't be turned off (no warning) |
| ChatGPT OAuth, every model | Implicit, managed by the backend | Still cached; ignored with a warning logged once |
| Gemini | Implicit, managed by Google | Still cached; can't be turned off (no warning) |
| OpenRouter, other models | `prompt_cache_key` sent | No key sent; OpenAI GPT-5.6 and later get `"explicit"` mode, other upstreams keep caching implicitly |

The conversation's key goes with every OpenAI request, and as `session_id` with every
OpenRouter request, either way; it routes requests but doesn't turn caching on. On
Anthropic and OpenRouter, caching also needs the model's cache support
(`Model::supports_cache`). `prompt_cache_lifetime` reports 5 minutes for Anthropic and
Claude on OpenRouter (1 hour with `anthropic_cache_ttl: OneHour`) and a conservative
5 minutes for OpenAI on both auth paths; Gemini and other OpenRouter models are
`Unknown`. It reports `Disabled` for models without cache support (the OpenAI Pro models,
`o3-pro`, `gemini-2.0-flash`, OpenRouter catalog models without cache prices), and with
caching off on Anthropic, Claude on OpenRouter, and GPT-5.6 and later with an API key.
`LLMClient::prompt_cache_lifetime(&model, &config)` reads only `config.prompt_caching`,
not `Conversation::enable_cache`.

### ChatGPT OAuth (arko_ai)

Requests over ChatGPT OAuth go to the Codex backend and differ from API-key requests:

- They always stream; `generate_text` collects the stream, so the stream idle timeout
  applies to it too.
- `max_tokens` is never sent: the model's own output limit applies.
- Caching is implicit and can't be turned off. The conversation's `prompt_cache_key` goes
  in the body and in the `session-id`, `thread-id` and `x-client-request-id` headers; the
  backend derives cache affinity from `session-id`.
- Without a reasoning config, `effort: "medium"` is sent to GPT-5.x and GPT-6 models;
  `openai_reasoning_mode` is omitted and warns once, because the backend does not accept
  `reasoning.mode`.
- `openai_service_tier` is `Error::UnsupportedFeature`.
- GPT-5.4 and GPT-5.5 snapshot IDs are sent without their date.
- The input limit is capped at 272,000 tokens (`CHATGPT_OAUTH_MAX_INPUT_TOKENS`), or
  the model's limit if lower; `LLMClient::max_input_tokens(&model)` returns it. Requests
  are not checked against it, and `ModelCallUsage::max_input_tokens` reports the
  effective limit for the auth path (272,000 over OAuth).
- The first `x-codex-turn-state` token the backend returns in a user turn is echoed on
  the turn's later requests (tool-result follow-ups and retries), once the response has
  been added with `add_assistant_message_from_response`.
- Usage is billed as `BillingKind::Subscription`: `billed()` is zero, and the list-price
  estimate stays in `total_cost`. Plan usage from the `x-codex-*` headers is in
  `ModelCallUsage::plan_usage`.
- A request rejected with HTTP 401 `token_expired` is refreshed and resent once. A
  token refresh times out after 60 seconds (`Error::HttpError`, transient).

See [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth).

### Tool images and Gemini 3.x (arko_ai)

`ToolResult::with_images` attaches images to a tool result. Where they go is
`Model::accepts_tool_result_images()`:

| Provider | Result images |
|---|---|
| OpenAI (API key and ChatGPT OAuth) | Inside `function_call_output.output` |
| Anthropic | Inside `tool_result.content` |
| Gemini 3 and later | Inside `function_result.result` |
| Gemini 2.x, and OpenRouter | After all of the turn's results, in a user turn, each after the text `Image from tool {name}, call {call_id}: {label}` (`unidentified` without a call ID) |

Anthropic and Gemini accept only base64 images (`Image::Url` is
`Error::UnsupportedFeature`); OpenAI and OpenRouter also send image URLs. A model
without image input rejects any image, result images included, before sending
(`Error::UnsupportedFeature`).

On Gemini 3 and later, content that follows tool results is folded into the message's
last `function_result`: the other content of a tool-result message (an `ask_user`
answer, a recovery note, environment details), and the text and image messages that
follow it before the next assistant or tool-result message. Texts are appended to the
result's text after a blank line (no blank line when the output is empty); images follow
the result's own images, and a text directly followed by an image in the same message
becomes that image's label. Anthropic sends a tool-result message's other content in that
same user message, in the message's own order (after the `tool_result` blocks when the
results come first, as arko's builders put them), and following messages as separate
user messages; OpenAI, OpenRouter and Gemini 2.x send it all as separate user input after
the results. See [Tool images](../arko_ai/README.md#tool-images).

### Retries and timeouts (arko_ai)

- arko_ai never retries a model call itself. The exceptions, over ChatGPT OAuth: a
  request rejected with 401 `token_expired` is refreshed and resent once, and with a
  credential store any other 401 re-reads the store and resends once if it holds newer
  credentials.
- `Error::retry_class()` is `Permanent`, `Transient`, `Capacity`, or `UsageLimit`.
  HTTP 408, 409, 429 and 5xx are retryable unless the body has a permanent code (quota,
  authentication, invalid request, context length, content policy, ...); a plan usage
  limit (`usage_limit_reached`) is `UsageLimit`; an overload code (`overloaded_error`,
  `service_unavailable`, ...) with a retryable status or none is `Capacity`; any other
  error reported after an HTTP 200 is `Transient`, as are connection failures, timeouts
  and `Error::StreamTruncated`. Server delay hints are capped at 30 days.
- `BackoffPolicy` has no default; `BackoffPolicy::new(initial, max)` jitters each delay
  between half and all of its step, which doubles from `initial` to `max`; a server hint
  is a minimum, with jitter added above it.
- There is no model-request timeout by default: a response is waited for as long as its
  connection stays open, since models can reason silently for over an hour (GPT Pro).
  `LLMClient::with_stream_idle_timeout(duration)` opts in to a limit on the gap between
  data on a streamed response; expiry is `Error::StreamTruncated` (transient), naming the
  timeout. It covers neither the wait for headers nor responses that don't stream
  (`generate_text`, except over ChatGPT OAuth), and there is no total deadline. Make it
  generous, for example 2 hours. Connecting has its own 30-second timeout and a token
  refresh a 60-second one. See [Long requests](../arko_ai/README.md#long-requests).

### Embeddings (arko_ai)

| Setting | Default | Notes |
|---|---|---|
| `EmbeddingClient::new()` | No Voyage key | `with_voyage_api_key`, or `from_env()` (reads `VOYAGE_API_KEY`; blank counts as unset). `with_voyage_client(VoyageClient::new(key).with_base_url(url))` replaces the base URL `https://api.voyageai.com/v1`. Without a key, `embed` fails with `Error::MissingApiKey` after the local checks. |
| `EmbeddingConfig::new(provider, model_id)` | No default model | An unsupported model is `Error::ModelNotSupported` when embedding or counting tokens. |
| `EmbeddingConfig::output_dimensionality` | `None`: not sent, the model's default, 1,024 | 256, 512, 1,024, 2,048; another value is `Error::InvalidDimension` before sending. |
| `voyage_input_type` | `None`: not sent | `Document`, `Query` (for search queries). |
| `voyage_truncation` | `None`: `truncation: false` is sent | Voyage's own default is `true`. arko never truncates silently: whatever this is set to, an input over the 32,000-token limit fails locally with `Error::TokenLimitExceeded` before sending, so none reaches Voyage to be truncated; split long inputs first. |
| `tokenizer_revision` | `None`: `main`, unpinned | Inputs are counted locally before sending, with the `voyageai/<model>` tokenizer downloaded from the Hugging Face Hub on first use and cached per repository and revision (`HF_HOME`, `HF_ENDPOINT`). A failed download is `Error::TokenizerError`, and nothing is sent; a revision the repository doesn't have (the Hub answers 404) is an error naming the repository and revision. |
| `hf_token` | `None`: the Hugging Face token saved on the machine (`$HF_HOME/token`, else `~/.cache/huggingface/token`), if any | Never serialized, and redacted in `Debug`: after loading a config, set it again with `with_hf_token`, in code or from an environment variable the application reads. A config saved with it by an earlier version still loads it. No registered model requires a token. |

Requests are checked locally in this order: the dimension, 32,000 tokens per input, the
model's total tokens per request (unpublished for `voyage-code-4`, so not checked), and
1,000 inputs. Vectors come back in input order; missing, duplicate, wrong-dimension,
non-finite or all-zero vectors are rejected. See
[Voyage embeddings](../arko_ai/README.md#voyage-embeddings).

### OpenRouter (arko_ai)

See [OpenRouter](../arko_ai/README.md#openrouter) for every option. The defaults:

- No attribution or metadata headers; `OpenRouterClient::with_app_attribution` opts in.
- No catalog is fetched: `Model::openrouter(slug)` has unknown limits and prices, and
  requests aren't checked against a catalog (arko's own checks still apply). Once a
  catalog is loaded (`fetch_models`, `fetch_user_models`, `set_catalog`), a slug it
  doesn't list is `Error::ModelNotSupported` (slugs starting with `@` skip the check),
  and `temperature`, `max_tokens`, `verbosity`, `reasoning`, `tools` or `tool_choice` on
  a model that doesn't list them is `Error::InvalidRequest`.
- `session_id` is always sent; `prompt_cache_key`, and `cache_control` for Claude, only
  while caching is on.
- `auto_exacto`: `CacheAware`. On tool requests with caching on, for a model that bills
  cache writes (from the catalog, or without one every `anthropic/` and `openai/` slug),
  `provider.sort: "price"` is sent, which turns Auto Exacto off and keeps sticky routing
  and its cache. `AllowAutoExacto` never adds it; `DisableAutoExacto` always does on tool
  requests. An explicit `sort` or `order`, or a `:exacto`, `:nitro` or `:floor` variant,
  always wins.
- `fallback_models`: off. With it, OpenRouter tries the next model after an error, a
  refusal included, so another model may answer a refused prompt.
- `provider` preferences, `user` and `service_tier`: sent only when set; `auto_router`
  only for `openrouter/auto` and `openrouter/auto-beta`, ignored for other models.
- `GenerationConfig::openrouter` is merged over `with_default_options`: `provider` and
  `auto_router` merge field by field, and every other field (lists and maps included)
  replaces the default whole, so a request can't unset a default.
- `ReasoningConfig` variants of other providers are dropped with a warning, not
  converted.
- `Conversation::add_reasoning_effort_update` works only on OpenRouter; the other
  providers reject it with `Error::UnsupportedFeature`. OpenRouter requests reject two
  adjacent updates, an update before tool results, updates with pro reasoning or `-pro`
  models, and an update to `none` on Claude.

### Persisted data (arko_ai)

- `GenerationConfig`: configs with the old `temperature_specified` flag load (`false`
  means no temperature; `true` without a `temperature` means 0.0). A `temperature`
  without the flag is sent.
- `Conversation`: `model`, `enable_cache` and `messages` are required. The saved key is
  kept, including old `arko:conversation:<uuid>` keys. A registry model's saved data is
  replaced by the registry's current data on load; other IDs keep the saved copy.
- `ToolResult`: `is_error` is required; `images` defaults to none.
- `ReasoningArtifact`: the old single-item `OpenAIEncryptedReasoning` loads as a
  one-item list.
- `GenerationResponse`: a response saved before stop reasons existed loads with
  `stop_reason` `Other("unknown")`.
- `UsageInfo`: `billing` defaults to `Metered`, and a record without `billed_cost`
  reports its `total_cost` as `billed()`. Anthropic records saved before cache counts
  were added back hold only uncached input in `input_tokens`.
- `EmbeddingConfig`: `hf_token` is never written; a config saved with it by an earlier
  version still loads it.
- A saved conversation whose tool-result message also holds text or images, or is
  followed by text or image user messages before the next assistant message, is sent to
  Gemini 3 and later folded as described above, so its first request after an upgrade
  misses Gemini's cache from that message on.

### Example binaries (arko_ai)

`llm_client_cli` and `embeddings_demo` load a `.env` file; `oauth_cache_probe` doesn't.

**`llm_client_cli`** (`text`, `chat`, `models`; credentials from `LLMClient::from_env`):

| Flag | Default | Notes |
|---|---|---|
| `-m`, `--model` | Required | A registered ID or an alias (the ID without its date or `-preview`; for Claude also the dotted version, such as `claude-opus-4.5`); an ID with `/` is an OpenRouter slug. |
| `-s`, `--system`, `-t`, `--temperature`, `-n`, `--max-tokens` | Unset: not sent | A temperature for a Claude model that takes none prints a warning. |
| `-r`, `--reasoning` | Off | Reasoning at the model's defaults: OpenAI reasoning models get no reasoning config (their default effort, no summaries; other OpenAI models are an error); Claude models with adaptive thinking get it without an effort; Claude models that need a manual budget get half of `--max-tokens` (or of the output limit), 1,024 to 16,000; Gemini gets its default level with thought summaries; OpenRouter gets `enabled: true`. |
| `--reasoning-level`, `--reasoning-effort` | Unset | An effort (OpenAI, sent with summaries; Claude adaptive; OpenRouter), a level (Gemini) or a token budget (Claude manual, Gemini, OpenRouter). Given together, they must agree, except a Claude budget with an effort. |
| `--reasoning-mode` | Unset | `standard` or `pro`, for OpenAI GPT-5.6 and GPT-6 API-key requests, or through OpenRouter; ChatGPT OAuth omits it; an error for other providers. |
| `--long-context` | Off | |
| `text -p`, `--prompt` | Required | `-i`, `--image PATH`; `--stream`. |
| `--codex-auth`, `--codex-auth-json PATH` | Off | OpenAI credentials from a Codex `auth.json` (the default location, or PATH), kept in step with the file; the `OPENAI_OAUTH_*` variables are then not read. |

**`embeddings_demo`**: `VOYAGE_API_KEY` (skipped without it); `voyage-4-lite`, 256
dimensions, `Document` input. **`oauth_cache_probe`**: `--out DIR` (required; new or
empty unless `--summarize-only`), `--auth-file` (default: the Codex `auth.json`),
`--model gpt-6-astra` (a registered OpenAI ID), `--effort xhigh`, `--conversations 3`,
`--turns 15` (2 with `--idle-gap SECS`, which can't be combined with
`--conversations`), `--label baseline`; `--legacy` with its request-shape flags, and
`--reasoning-context`, for comparisons; `--summarize-only` recomputes the summary of an
earlier run.


## arko_agents

Tool-using agents built on arko_ai: an `AgentConfig` describes an agent, a `Task` holds one run, and `AgentRuntime` runs the task as a Tokio task that streams `AgentEvent`s and takes `UserCommand`s.

Guides: [arko_agents README](../arko_agents/README.md) · [interactive terminals](interactive-terminals.md) · [context management](context-efficiency.md)

Most items are re-exported at the crate root (`arko_agents::AgentConfig`). Items without a root re-export are written with their module: `agent_config::PathSafetyPolicy`, `task::CANCELLED_BY_USER`, `mcp::McpTool`, `parsing::PromptBuilder`, `parsing::remove_output_based_only_markers`, the `parsing::DEFAULT_*` constants, `skill::{LoadedSkill, SkillLoadError}`, `skill::types::SkillFrontmatter`, the built-in tools under `builtins` (`arko_agents::builtins`), and everything in `terminal` (Unix and Windows only) and `utils`.

### Map (arko_agents)

#### Map: agent configuration (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentConfig` | An agent's definition: role, instructions, model, tools and policies. Serializable; no `Default`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::builder`, `AgentConfigBuilder::new` | Start a builder from `id`, `name` and an `AgentLLMConfig`. | `arko_agents/src/agent_config.rs` | [Agent configuration](../arko_agents/README.md#agent-configuration) |
| `AgentConfigBuilder::{role_description, instructions, tool_config, environment_details_config, complete_task_summary_prompt_override, parsing_config_override, error_recovery_config, llm_error_policy, llm_retry_config, user_command_timeout_seconds, user_reminder, skills_config, monitoring, refusal_fallback, build}` | One setter per field. `build` fails without `role_description` or `instructions`, gives unset fields their defaults, and runs `validate`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::load_from_path`, `AgentConfig::save_to_path` | Load JSON without validating; save pretty JSON through a temporary file and a rename. | `arko_agents/src/agent_config.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `AgentConfig::validate` | Checks retry timing, monitors, tool time limits, a command timeout and reminder interval of at least 1, forced tool choices the model accepts, and a refusal fallback on the same provider. Fails with `ArkoAgentsError::BuilderError`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::{id, name, role_description, instructions, llm_config}` | Required fields. `name` heads the system prompt and is the `agent_name` tools receive. | `arko_agents/src/agent_config.rs` | [Agent configuration](../arko_agents/README.md#agent-configuration) |
| `AgentConfig::tool_config` | The agent's `AgentToolConfig`. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentConfig::environment_details_config` | The built-in environment details (`EnvironmentDetailsConfig`). | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `AgentConfig::error_recovery_config`, `ErrorRecoveryConfig::on_tool_execution_error`, `ToolErrorPolicy::{InformAgentAndContinue, PauseAndAwaitApproval, FailTask}` | What a failed tool call does. | `arko_agents/src/agent_config.rs` | [Tool errors](../arko_agents/README.md#tool-errors) |
| `AgentConfig::llm_error_policy`, `LLMErrorPolicy::{RetryThenPauseAndAwaitApproval, RetryThenInformAgent, InformAgentAndContinue, PauseAndAwaitApproval}` | What a failed model request does. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentConfig::llm_retry_config` | Model-call recovery timing (`LLMRetryConfig`). | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentConfig::user_command_timeout_seconds` | Opt-in limit on each wait for a command. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::user_reminder`, `UserReminder { content, frequency }`, `UserReminder::{new, initial_only, every_message, every_n_messages}`, `ReminderInjectionFrequency::{InitialTaskPromptOnly, EveryUserMessage, EveryNUserMessages}` | Text appended to user messages under `# Reminder`, and how often. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `AgentConfig::skills_config` | Skill sources (`SkillsConfig`), read by `SkillManager::load`. | `arko_agents/src/agent_config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `AgentConfig::monitoring` | Opt-in model-call monitors (`MonitoringConfig`). | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentConfig::refusal_fallback` | Opt-in `AgentLLMConfig` a refused turn is sent to once. | `arko_agents/src/agent_config.rs` | [Refusals (arko_agents)](#refusals-arko_agents) |
| `AgentConfig::claude_code`, `ClaudeCodeSettings { cli_path, cli_args, setting_sources, replace_system_prompt, env, allow_commands_outside_workspace, allowed_command_paths, dangerously_allow_all_commands }`, `ClaudeCodeSettings::command_guard`, `AgentConfig::runs_in_claude_code` | How the Claude Code CLI runs for an agent whose model is a Claude Code model. | `arko_agents/src/claude_code.rs` | [Claude Code models](../arko_agents/README.md#claude-code-models) |
| `claude_code::{BUILT_IN_TOOLS, UNSUPPORTED_BUILT_IN_TOOLS, is_built_in_tool, built_in_path_parameters}`, `ClaudeCodeTools::{read_only, editing, shell, web, all}` | Claude Code's built-in tools, offered to a Claude Code agent when `available_tool_names` lists them; their path inputs for the workspace path policy. | `arko_agents/src/claude_code.rs` | [Claude Code models](../arko_agents/README.md#claude-code-models) |
| `Task::{claude_code_session, claude_code_synced}` | The Claude Code session a task resumes, and how much of its conversation the session has. | `arko_agents/src/task/mod.rs` | [Claude Code models](../arko_agents/README.md#claude-code-models) |
| `AgentConfig::handover`, `HandoverConfig { at_input_tokens, at_context_fraction, report_request }`, `HandoverConfig::{at_input_tokens, at_context_fraction, threshold, report_request, validate}` | Opt-in conversation handover and its threshold. | `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `HandoverHook::on_handover`, `HandoverSeed { task_prompt, context }`, `HandoverSeed::{for_task, message}`, `AgentRuntime::with_handover_hook` | A host's hook, called as the conversation is replaced; it returns the fresh conversation's first message. | `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `Task::{handover_requested, handovers, conversation_usage_start, conversation_model_call_start}`, `HandoverRequest`, `HandoverRecord`, `Task::{start_fresh_conversation, current_conversation_context_size}`, `handover::{DEFAULT_REPORT_REQUEST, REPORT_HEADING, REPORT_INTRODUCTION}` | A pending handover, the handovers made, where the current conversation's usage starts, and the primitives the runtime uses. | `arko_agents/src/task/mod.rs`, `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `AgentConfig::complete_task_summary_prompt_override` | Replaces the description of `complete_task`'s `summary` parameter. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::parsing_config_override`, `AgentConfig::get_tool_parsing_config` | Output-based tag syntax; the getter returns the override or `ToolParsingConfig::default()`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::{should_auto_approve_tool, get_path_safety_policy, is_parallel_safe, allows_batched_calls_for_tool, asks_one_tool_call_per_turn, is_native_tool_calling}` | Queries that resolve per-tool overrides against the agent's defaults, as the runtime does. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentLLMConfig { model, generation_config }` | An `arko_ai::Model` and its `arko_ai::GenerationConfig`. Saved as `provider`, `model_id`, a model snapshot and `generation_config`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |

#### Map: tool configuration (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentToolConfig` | Offered tools, tool calling mode, approvals, workspace path policy, parallel calls and time limits. Has `Default`. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::available_tool_names` | The tools offered besides `complete_task`. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::tool_calling_mode`, `ToolCallingMode::{Native, OutputBased}` | Native tool calling, or legacy calls parsed from the response text. | `arko_agents/src/agent_config.rs` | [Tool calling modes](../arko_agents/README.md#tool-calling-modes) |
| `AgentToolConfig::default_auto_approve_tool_use` | Whether calls run without approval unless a per-tool entry says otherwise. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::default_out_of_workspace_path_policy`, `agent_config::PathSafetyPolicy::{AutoReject, RequireApproval, AutoApprove}` | What a path parameter outside the working directory does. | `arko_agents/src/agent_config.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `AgentToolConfig::one_tool_call_per_turn` | Opt-in: one tool call per response. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::tool_timeout_seconds`, `DEFAULT_TOOL_TIMEOUT_SECONDS` | The host-side time limit of a tool call; the constant is 1,200 seconds. | `arko_agents/src/agent_config.rs` | [Tool time limit](../arko_agents/README.md#tool-time-limit) |
| `AgentToolConfig::tool_specific_configs`, `PerToolConfigOverride { auto_approve_tool_use, parallel_safe, out_of_workspace_path_policy, tool_timeout_seconds }` | Per-tool overrides; each `None` inherits the agent's setting. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::offered_tool_names`, `AgentToolConfig::tool_timeout` | The offered names (`available_tool_names` plus `complete_task`), and a tool's effective time limit. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |

#### Map: environment details (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `EnvironmentDetailsConfig { current_working_directory, file_tree, file_tree_mode, current_datetime, operating_system }` | The built-in details; none is included by default. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `BuiltInEnvironmentDetail { include, placement, depth }` | One detail's switch, placement and (file tree only) depth. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `EnvironmentDetailPlacement::{SystemPrompt, InitialTaskPrompt, EachUserMessage}` | Where a detail goes. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `FileTreeMode::{Diff, Full}` | File tree as a first snapshot and then changes, or in full at its placement. | `arko_agents/src/agent_config.rs` | [File-tree context](context-efficiency.md#file-tree-context) |
| `DeveloperDefinableEnvironmentDetail { name, placement, content_provider, static_content_for_serde, dynamic_provider_name_for_serde }`, `DeveloperDefinableEnvironmentDetail::{new_static, new_dynamic_placeholder, missing_content_error}` | A host-defined detail passed to `Task::new`, with static content or a provider. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `DynamicContentProvider::{Static, Dynamic}` | Static text, or an `Fn(&Task) -> Result<String, String>` called when the detail is rendered; not serialized. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |

#### Map: model-call recovery and monitoring (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `LLMRetryConfig { initial_delay_ms, max_delay_ms, max_elapsed_ms, jitter, capacity_retry, rate_limit_retry, usage_limit_policy, stream_idle_timeout_ms }`, `LLMRetryConfig::validate` | Timing of model-call recovery. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `LLMRateLimitRetryConfig { initial_delay_ms, max_delay_ms, max_wait_ms }` | Waits for short rate limits and capacity errors. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `LLMCapacityRetryConfig { initial_delay_ms, max_delay_ms }` | Opt-in indefinite waits for capacity errors. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `UsageLimitPolicy::{Pause, WaitUntilReset { max_wait_ms }}` | What a plan usage limit does. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `MonitoringConfig { plan_usage, cache }`, `MonitoringConfig::{is_empty, validate}` | The opt-in monitors. | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `PlanUsageThreshold { percent, action }`, `PlanUsageAction::{Pause, SlowDown { delay_ms }}` | A plan-usage threshold checked before each request, and what reaching it does. | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `CacheMonitorConfig { warmup_calls, window_calls, warn_below, pause_below }` | The prompt-cache efficiency monitor. | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |

#### Map: tasks (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Task` | One run: the configuration it runs with, its conversation, state and usage. Serializable. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `Task::new(agent_config, initial_task_prompt, initial_cwd, developer_env_details)` | A `Pending` task with an empty conversation for the configured model. | `arko_agents/src/task/mod.rs` | [Quick start](../arko_agents/README.md#quick-start) |
| `Task::load_from_path`, `Task::save_to_path` | Load JSON; save pretty JSON through a temporary file and a rename. | `arko_agents/src/task/mod.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `Task::{id, agent_config_id, agent_config, initial_task_prompt, conversation, current_state, start_time, end_time, developer_env_details, current_working_directory}` | The task's identity, the configuration and prompt it runs with, its `arko_ai::Conversation`, state, times, developer details and working directory. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `Task::{usage_history, model_call_usage_history}`, `Task::{total_usage, total_usage_cost, total_input_tokens, total_output_tokens, current_context_size, record_model_call_usage}` | Per-call usage and totals. `total_usage_cost` is the billed amount: ChatGPT OAuth calls add zero. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `Task::{user_message_count, pending_native_tool_results, pending_native_tool_calls, pending_tool_images, pending_text_tool_results, deferred_tool_errors, runtime_messages, runtime_text_messages, fallback_for, fallback_calls, plan_usage_threshold_reported, monitor_pause, pending_context_injections}`, `Task::{increment_user_message_count, set_state, get_run_time}` | State the runtime keeps between runs, saved with the task, and helpers it uses. `set_state` records `end_time` on entering a terminal state. | `arko_agents/src/task/mod.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `Task::prepare_for_resumption_from_completed(prompt)` | Puts a `Completed` task back in progress with the prompt as a new user message. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `Task::prepare_for_resumption_with_user_response(response)` | Answers an `AwaitingUserResponse` task before it runs again. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `TaskState::{Pending, InProgress, AwaitingToolApproval, AwaitingBatchToolApproval, AwaitingUserResponse, AwaitingErrorRecoveryApproval, Interrupted, Completed, Failed, Cancelled}`, `TaskState::{is_terminal, is_awaiting_approval}` | A task's state. `Completed`, `Failed` and `Cancelled` are terminal. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `ErrorRecoveryCause::{Unknown, RetriesExhausted, NonRetryable, UnusableResponses, UsageLimit, Refusal, ToolError, Monitor, Middleware}` | Why a task awaits error-recovery approval. | `arko_agents/src/task/mod.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `ApprovalReason { tool_not_auto_approved, path_outside_workspace }`, `ApprovalReason::{is_unrecorded, describe}` | Why a call awaits approval. | `arko_agents/src/task/mod.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `PendingToolCallInfo { tool_call_id, tool_name, parameters, tool_description, approval_reason }` | One call of a batch awaiting approval. | `arko_agents/src/task/mod.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `DeferredToolError { message, tool_call_id, middleware_pause }`, `PlanUsageWindowMark { window, resets_at }` | Saved records of a pending tool-error pause and of the last plan-usage window reported. | `arko_agents/src/task/mod.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `task::CANCELLED_BY_USER` | `"Task cancelled by user."`, the reason a `CancelTask` records. | `arko_agents/src/task/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |

#### Map: refusals (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Task::resolve_refusal(changes)` | Applies changes to the refused input, all or none, and removes the refused turn. | `arko_agents/src/task/refusal.rs` | [Refusals](../arko_agents/README.md#refusals) |
| `RefusalResolution::{RephraseUserMessage, RemoveUserMessage, RemoveImages, RedactToolResults}` | One change to a refused input. | `arko_agents/src/task/refusal.rs` | [Refusals](../arko_agents/README.md#refusals) |
| `Task::refused_input`, `RefusedInput { messages }`, `RefusedMessage { message_index, runtime, runtime_text, tool_results, images }`, `RefusedToolResult { call_id, tool_name }` | The user messages after the last assistant turn that a refused turn answered. | `arko_agents/src/task/refusal.rs` | [Refusals](../arko_agents/README.md#refusals) |

#### Map: runtime and commands (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentRuntime::new(client)` | A runtime over an `Arc<arko_ai::LLMClient>`. | `arko_agents/src/runtime/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentRuntime::run_task(task, tool_registry, event_throttle_ms, simplified_tool_events, skill_prompt_context, interceptor_config, execution_middleware)` | Spawns the run; returns the event receiver, the command sender, and a join handle that resolves to the task. | `arko_agents/src/runtime/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentRuntime::with_turn_context`, `TurnContextSource::turn_context` | A source asked for context before each model request. | `arko_agents/src/runtime/turn_context.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentRuntime::with_host_decisions`, `HostDecisions::{new, set, get}` | A run-level switch: while `true`, no wait of the run is timed. | `arko_agents/src/runtime/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `UserCommand::{ApproveToolUse, RejectToolUse, SubmitBatchToolApprovals, SubmitUserResponse}`, `ToolApprovalDecision::{Approve, Reject}` | Answers to approval and question waits. An approval can carry `custom_tool_output` (used instead of running the tool) and `message_to_agent`. | `arko_agents/src/runtime/command.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `UserCommand::{ApproveErrorRecovery, ContinueAfterError, RejectErrorRecovery, ResolveRefusal}` | Answers to an error-recovery pause. | `arko_agents/src/runtime/command.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `UserCommand::{InjectContext, HostDeciding, RequestHandover, CancelTask, SuspendTask}` | Context updates, wait control, a handover, and stopping the run. | `arko_agents/src/runtime/command.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |

#### Map: events (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentEvent::{TaskStateChanged, TaskStreamEnd}` | A state change; `TaskStreamEnd` is the last event of a run. | `arko_agents/src/runtime/event.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `AgentEvent::{AssistantResponseChunk, ReasoningChunk}` | Response text and reasoning, chunk by chunk. | `arko_agents/src/runtime/event.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentEvent::{ToolCallInitiated, ToolCallFormatted}` | A call announced, then parsed with its parameters. | `arko_agents/src/runtime/event.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `AgentEvent::{ToolParamName, ToolParamValueChunk, ToolParamEnd}` | Parameter-by-parameter parsing events, in output-based mode with `simplified_tool_events` off. | `arko_agents/src/runtime/event.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentEvent::{ToolCallAwaitingApproval, BatchToolCallAwaitingApproval, UserQuestionsAsked}` | The task waits for an approval or an answer. | `arko_agents/src/runtime/event.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `AgentEvent::{ToolResult, ToolResultIntercepted}` | A call's result, and the replacement when an interceptor changed it. | `arko_agents/src/runtime/event.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |
| `AgentEvent::{ToolRejected, UnavailableToolAttempted}` | A call that was rejected, stopped or not run; a call to a tool the task can't run. | `arko_agents/src/runtime/event.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `AgentEvent::{ErrorEncountered, RetryScheduled}` | An error (`requires_approval_to_retry` marks a pause), and a scheduled retry with its attempt, delay and elapsed time. | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::UsageLimitReached` | A plan usage limit, with the reset time and the planned wait, when known. | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::OutputTokenLimitReached` | A response ended at its output-token limit. | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::{RefusalReceived, RefusedInput, RefusalFallback}` | A refusal, the input it answered, and the fallback model being asked. | `arko_agents/src/runtime/event.rs` | [Refusals (arko_agents)](#refusals-arko_agents) |
| `AgentEvent::{UsageReported, ModelCallUsageReported, PlanUsageReported, ModelRerouted}` | Usage after each model call; plan usage when the provider reports it; a model served other than the one requested. | `arko_agents/src/runtime/event.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentEvent::{PlanUsageThresholdReached, CacheEfficiencyReported, CacheEfficiencyLow}` | Events of the opt-in monitors. | `arko_agents/src/runtime/event.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentEvent::{HandoverRequested, ConversationHandedOver}` | A handover started; the conversation was replaced, with the report. | `arko_agents/src/runtime/event.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |

#### Map: tools (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Tool` (`definition`, `execute`) | The trait a tool implements; `execute` returns the result text or a `ToolError`. | `arko_agents/src/tool/mod.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `Tool::execute_with_output`, `ToolOutput { text, images }`, `ToolOutput::text`, `ToolImage { description, image }` | A result with images; by default it calls `execute`. | `arko_agents/src/tool/mod.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `Tool::{uses_host_timeout, interruptible, allow_result_interception}` | Opt-outs, each `true` by default: the host time limit, being stopped mid-call, result interception. | `arko_agents/src/tool/mod.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `Tool::{bind_runtime, reset_context, observe_context, name}` | A weak command sender for background tools; callbacks that track what reached the conversation (no-ops by default); `name` is the definition's name. | `arko_agents/src/tool/mod.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `TaskContext { task_id, agent_name, initial_task_prompt, current_working_directory }` | What `execute` receives about the task. | `arko_agents/src/tool/mod.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `ToolDefinition { name, description, output_based_example, parameters, parallel_safe, tool_type_details, raw_json_schema_override }` | A tool's name, description and parameters as offered to the model. | `arko_agents/src/tool/definition.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolDefinition::{load_from_path, save_to_path, validate_params, coerce_text_arguments, to_json_schema, to_arko_ai_tool_definition}` | File load and save; an argument check the runtime doesn't call; text-to-boolean coercion for output-based calls; the JSON Schema and arko_ai definition sent to the model. | `arko_agents/src/tool/definition.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `ToolParameterDefinition { name, description, param_type, required, is_path, flag, default_value }`, `ToolParameterType::{String, Bool}` | One parameter. `is_path` marks a path checked against the workspace; `flag` and `default_value` are used by end-user command tools. | `arko_agents/src/tool/definition.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `ToolTypeDetails::{BuiltIn, DeveloperDefined, EndUserDefined { command_template }}` | The tool's kind; `EndUserDefined` runs a command. | `arko_agents/src/tool/types.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `ToolRegistry::new`, `ToolRegistry::default` | A registry holding the six built-in tools. | `arko_agents/src/tool/registry.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolRegistry::register` | Adds or replaces a tool by name; a `complete_task` that isn't `BuiltIn` is refused with a warning. | `arko_agents/src/tool/registry.rs` | [Custom tools](../arko_agents/README.md#custom-tools) |
| `ToolRegistry::{get, list_tool_names, list_tool_definitions, restricted_to}` | Lookups; `restricted_to` copies only the named tools, as the runtime does for the offered tools. | `arko_agents/src/tool/registry.rs` | [Tools](../arko_agents/README.md#tools) |
| `ToolRegistry::load_end_user_tool` | Registers an `EndUserDefined` definition as a command tool. | `arko_agents/src/tool/registry.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolError::{ParameterValidation, MissingParameter, ExecutionFailed, Configuration, PathAccessRejected, ShellCommandFailed, ShellCommandTemplate, TimedOut, Io}` | A tool failure; its text goes into the call's error result. | `arko_agents/src/tool/error.rs` | [Tool errors](../arko_agents/README.md#tool-errors) |

#### Map: result interception and execution middleware (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `InterceptorConfig { interceptor, failure_policy }` | An interceptor and its failure policy, passed to `run_task`. | `arko_agents/src/tool/interceptor.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |
| `ToolResultInterceptor::intercept`, `InterceptedResult::{Unchanged, Replaced}` | A trait that may replace a result's text before it enters the conversation. | `arko_agents/src/tool/interceptor.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |
| `InterceptionFailurePolicy::{UseRawResult, RetryThenUseRawResult, InformAgent, RetryThenInformAgent}`, `InterceptionFailurePolicy::{max_attempts, informs_agent}` | What an interceptor error does: the raw result is used, with a note under the `InformAgent` variants. | `arko_agents/src/tool/interceptor.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolExecutionMiddleware::{before, after}` | A trait called around every tool execution. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |
| `ToolExecutionDecision::{Continue, Reject, FailTask, CancelTask, PauseTask}` | The decision `before` returns. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |
| `ToolExecutionOutcome::{Success, Error, CallerProvided}` | What `after` receives. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |
| `ToolExecutionContext { task_id, agent_name, initial_task_prompt, current_working_directory }`, `ToolCallContext { tool_call_id, tool_name, parameters }` | The task and the call the middleware sees. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware) |

#### Map: built-in tools (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `complete_task`: `builtins::{CompleteTaskTool, COMPLETE_TASK_TOOL_NAME}`, `builtins::complete_task::COMPLETE_TASK_SUMMARY_PARAM_NAME`, `CompleteTaskTool::default_summary_description` | Ends the task as `Completed { summary }`; offered whether listed or not. | `arko_agents/src/tool/builtins/complete_task.rs` | [Tools](../arko_agents/README.md#tools) |
| `ask_user`: `builtins::{AskUserTool, ASK_USER_TOOL_NAME}`, `builtins::ask_user::ASK_USER_QUESTIONS_PARAM_NAME` | Puts the task in `AwaitingUserResponse` with its `questions`; runs without approval by default. | `arko_agents/src/tool/builtins/ask_user.rs` | [Tools](../arko_agents/README.md#tools) |
| `read_file`: `builtins::{ReadFileTool, READ_FILE_TOOL_NAME}` | Returns a UTF-8 file with numbered lines (`1 \| …`); parallel-safe. | `arko_agents/src/tool/builtins/read_file.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `write_file`: `builtins::{WriteFileTool, WRITE_FILE_TOOL_NAME}` | Writes `content` to `path`, creating parent directories. | `arko_agents/src/tool/builtins/write_file.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `apply_diff`: `builtins::{ApplyDiffTool, APPLY_DIFF_TOOL_NAME}` | Applies SEARCH/REPLACE blocks to an existing file. | `arko_agents/src/tool/builtins/apply_diff.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `list_files`: `builtins::{ListFilesTool, LIST_FILES_TOOL_NAME}` | Lists a directory's tree at every depth, honouring ignore files; parallel-safe; results never intercepted. | `arko_agents/src/tool/builtins/list_files.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `load_skill`: `LoadSkillTool`, `LoadSkillTool::new`, `LOAD_SKILL_TOOL_NAME` | Returns an on-demand skill, or one of its resources; registered by `SkillManager`; parallel-safe. | `arko_agents/src/skill/load_skill.rs` | [Skills](../arko_agents/README.md#skills) |

#### Map: output-based parsing and prompts (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ToolParsingConfig { tag_open, tag_close, batched_tool_call_tag, escape_prefix }`, `parsing::{DEFAULT_TAG_OPEN, DEFAULT_TAG_CLOSE, DEFAULT_BATCHED_TOOL_CALL_TAG, DEFAULT_ESCAPE_PREFIX}` | Tag syntax of output-based calls; the defaults are `<`, `>`, `batched_tool_call` and `!!--!!`. | `arko_agents/src/parsing/mod.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `strip_output_based_only_blocks`, `parsing::remove_output_based_only_markers` | Remove `{{#output_based_only}}…{{/output_based_only}}` blocks (native mode), or only the markers (output-based mode). | `arko_agents/src/parsing/mod.rs` | [Agent configuration](../arko_agents/README.md#agent-configuration) |
| `ToolCallParser::{new, ingest_chunk, finalize}`, `RawParserEvent::{Text, ToolCallStart, ParamName, ParamValueChunk, ParamEnd, ToolCallEnd, BatchedToolCallStart, BatchedToolCallEnd, Error}` | The streaming parser of output-based calls. | `arko_agents/src/parsing/mod.rs` | [Tool calling modes](../arko_agents/README.md#tool-calling-modes) |
| `assemble_tool_calls`, `ToolCallGroup::{Single, Batch}`, `ParsedToolCall { name, parameters }`, `ToolAssemblyError` (12 variants) | Builds call groups from parser events. | `arko_agents/src/parsing/mod.rs` | [Tool calling modes](../arko_agents/README.md#tool-calling-modes) |
| `parsing::PromptBuilder::{new, build_system_prompt, format_initial_task_prompt, augment_user_message_with_env_details_and_reminder, text_beside_tool_results, format_tool_result_as_user_message, format_tool_rejection_as_user_message, format_tool_execution_error_as_user_message, format_user_response_as_user_message}` | Builds the system prompt and the user messages the runtime sends. | `arko_agents/src/parsing/prompt_builder.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |

#### Map: skills (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `SkillsConfig { workspace_source, global_sources, on_invalid_skill, load_skill_interception }` | Where skills come from, invalid-skill handling, and opt-in interception of `load_skill` results. | `arko_agents/src/skill/config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `SkillSource { path, include, exclude }`, `InvalidSkillPolicy::{Warn, Error}` | A skills directory with name filters; skip or fail on invalid skills. | `arko_agents/src/skill/config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `SkillManager::load(config, workspace_root)` | Reads the skills: directories with a `SKILL.md`, `.skill` zip bundles, and single Markdown files other than `README.md`. | `arko_agents/src/skill/mod.rs` | [Skills](../arko_agents/README.md#skills) |
| `SkillManager::setup(registry, agent_config)` | With on-demand skills, registers `load_skill` and offers it with auto-approval; returns the `SkillPromptContext`. | `arko_agents/src/skill/mod.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `SkillManager::{create_tool, register_tool, tool_name, prompt_context, has_on_demand_skills, has_auto_loaded_skills, total_skill_count}` | The steps `setup` combines, and counts. | `arko_agents/src/skill/mod.rs` | [Skills](../arko_agents/README.md#skills) |
| `SkillPromptContext { on_demand_skills, auto_loaded_skills }`, `SkillSummary { name, description }`, `AutoLoadedSkillContent { name, content }` | What the system prompt lists (on-demand skills) and includes (auto-loaded skills); passed to `run_task`. | `arko_agents/src/skill/types.rs` | [Skills](../arko_agents/README.md#skills) |
| `skill::LoadedSkill { name, original_name, description, auto_load, content, resources, source_label }`, `skill::types::SkillFrontmatter { name, description, auto_load, extra }` | A loaded skill and its parsed front matter. | `arko_agents/src/skill/types.rs` | [Skills](../arko_agents/README.md#skills) |
| `skill::SkillLoadError::{MissingFrontmatter, MalformedFrontmatter, FrontmatterParseError, MissingRequiredField, Io, ZipError, MissingSkillMd}` | Why a skill is invalid; returned in `ArkoAgentsError::Skill` under `InvalidSkillPolicy::Error`. | `arko_agents/src/skill/types.rs` | [Skills](../arko_agents/README.md#skills) |

#### Map: MCP (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `McpConfig { mcp_servers }`, `McpConfig::load_from_path`, `McpServerConfig { name, command, args, env }` | The server list (JSON key `mcpServers`); each server is a command to spawn. | `arko_agents/src/mcp/config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `McpManager::from_config` | Spawns each server, completes the MCP handshake and lists its tools. On a failure it stops the servers already started and returns `ArkoAgentsError::Runtime`. | `arko_agents/src/mcp/manager.rs` | [MCP servers](../arko_agents/README.md#mcp-servers) |
| `McpManager::{register_tools, tool_names, tools}` | Registers the tools; the names to add to `available_tool_names`; the tools. | `arko_agents/src/mcp/manager.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `McpManager::shutdown` | Closes each connection, gives each server 3 seconds to exit, then kills it. | `arko_agents/src/mcp/manager.rs` | [MCP servers](../arko_agents/README.md#mcp-servers) |
| `mcp::McpTool`, `McpTool::{new, server_name, remote_tool_name}` | An MCP server's tool as a `Tool` named `mcp__<server>__<tool>`. | `arko_agents/src/mcp/tool.rs` | [MCP servers](../arko_agents/README.md#mcp-servers) |

#### Map: interactive terminals (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `terminal::TerminalConfig { directory, shell, shell_args, output_buffer_bytes, max_log_bytes, excerpt_bytes, prompt_quiet_ms, prompt_rules, default_timeout, quiet_terminal_environment, command_guard }`, `TerminalConfig::new(directory)` | The terminal manager's settings. | `arko_agents/src/terminal/mod.rs` | [Configuration](interactive-terminals.md#configuration) |
| `terminal::CommandGuard { allow_outside_workspace, allowed_paths, dangerously_allow_all_commands }`, `CommandGuard::check(command, dialect, working_directory)`, `terminal::CommandRefusal { message, target, protected }`, `terminal::ShellDialect::{Posix, Cmd, PowerShell}`, `ShellDialect::from_shell(program)`, `TerminalError::Refused` | Refuses commands that would delete, move, overwrite or reformat protected paths or paths outside the working directory. | `arko_agents/src/terminal/guard/mod.rs` | [Command guard](interactive-terminals.md#command-guard) |
| `terminal::QUIET_TERMINAL_ENVIRONMENT`, `terminal::QUIET_TERMINAL_LOCALE` | The variables PTY jobs get: `TERM=dumb`, `NO_COLOR=1`, `PAGER=cat`, `GIT_PAGER=cat`, and `LANG`, `LC_CTYPE`, `LC_ALL` set to `C.UTF-8` (`en_US.UTF-8` on macOS). | `arko_agents/src/terminal/mod.rs` | [Configuration](interactive-terminals.md#configuration) |
| `terminal::PromptRule { name, pattern }` | A named regular expression marking a possible input prompt. | `arko_agents/src/terminal/detection.rs` | [Input requests](interactive-terminals.md#input-requests) |
| `terminal::TerminalManager::new(config)` | Checks the limits, creates and locks the directory, and loads saved job records. | `arko_agents/src/terminal/mod.rs` | [Registering the tools](interactive-terminals.md#registering-the-tools) |
| `TerminalManager::register_tools`, `terminal::TerminalTool::{all, NAMES}` | The four tools `terminal_start`, `terminal_status`, `terminal_input` and `terminal_control`; registering doesn't offer them. | `arko_agents/src/terminal/tools.rs` | [Registering the tools](interactive-terminals.md#registering-the-tools) |
| `TerminalManager::bind_runtime(owner, sender)` | Sends a task's job notices to it as `InjectContext`; `run_task` binds offered terminal tools itself. | `arko_agents/src/terminal/mod.rs` | [Notifications and lifecycle](interactive-terminals.md#notifications-and-lifecycle) |
| `TerminalManager::{start, status, input, cancel, wait, forget}`, `terminal::StartOptions { terminal, detect_prompts, timeout }` | The job operations behind the tools. `StartOptions::default()` is a PTY with prompt detection and no deadline. | `arko_agents/src/terminal/mod.rs` | [Registering the tools](interactive-terminals.md#registering-the-tools) |
| `TerminalManager::shutdown` | Cancels the jobs, waits up to 3 seconds for each, and refuses new starts. | `arko_agents/src/terminal/mod.rs` | [Notifications and lifecycle](interactive-terminals.md#notifications-and-lifecycle) |
| `terminal::TerminalSnapshot { job_id, revision, state, input, exit_code, error, output, log_path, log_truncated, terminal }`, `terminal::OutputWindow { start, end, available_end, gap, text, raw_hex }` | A job's state and output window, as the tools return them. | `arko_agents/src/terminal/mod.rs` | [Reading output](interactive-terminals.md#reading-output) |
| `terminal::JobState::{Running, Succeeded, Failed, Cancelled, TimedOut, Interrupted}`, `JobState::is_terminal`, `terminal::InputNotice { source, prompt, prompt_truncated }`, `terminal::InputSource::{Program, PromptRule}` | A job's state, and a declared or inferred input request. | `arko_agents/src/terminal/mod.rs` | [Input requests](interactive-terminals.md#input-requests) |
| `terminal::TerminalError::{Invalid, NotOwned, StaleView, Finished, Io, Json}` | Terminal operation errors; the tools return their text as `ToolError::ExecutionFailed`. | `arko_agents/src/terminal/mod.rs` | [Input requests](interactive-terminals.md#input-requests) |

#### Map: errors and utilities (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ArkoAgentsError::{LLM, Tool, Io, Json, Configuration, Parsing, AssemblyError, Runtime, TaskExecution, ToolNotFound, InvalidTaskState, ChannelSend, ChannelReceive, FileOperation, UuidError, WalkDir, IgnoreError, JoinError, BuilderError, Skill}` | The crate's error. `BuilderError` comes from validation, `Configuration` from an environment detail without content, `ChannelSend` from a dropped event receiver, `InvalidTaskState` from the `Task` resumption and refusal methods, `FileOperation` from loading and saving. | `arko_agents/src/error.rs` | [Running tasks](../arko_agents/README.md#running-tasks) |
| `utils::resolve_path` | The path a built-in file tool opens: trimmed, taken literally (no `~` or `$VAR`), joined to the working directory, made absolute, with `.` and `..` resolved without following links. | `arko_agents/src/utils.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `utils::is_path_in_workspace` | Whether that path lies inside the working directory, compared component by component (ignoring ASCII case on Windows). | `arko_agents/src/utils.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `utils::normalize_path` | Resolves `.` and `..` without touching the file system. | `arko_agents/src/utils.rs` | [Approvals and workspace paths](../arko_agents/README.md#approvals-and-workspace-paths) |
| `utils::write_file_atomically` | Writes a temporary file in the same directory, syncs it, and renames it over the target. | `arko_agents/src/utils.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `utils::get_file_tree` | A directory tree as text, honouring ignore files, to an optional depth. | `arko_agents/src/utils.rs` | [File-tree context](context-efficiency.md#file-tree-context) |
| `utils::{get_current_working_directory_string, get_operating_system, get_current_datetime_utc}` | The process's working directory, `std::env::consts::OS`, and the UTC time in RFC 3339. | `arko_agents/src/utils.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |

#### Map: example binaries (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_agents_cli`: `-c`/`--agent-config`, `-w`/`--workspace-dir`, `-p`/`--prompt`, `--tool-calling-mode`, `--event-throttle-ms`, `--simplified-tool-events`, `--user-command-timeout`, `--mcp-config`, `--openai-auth-json`, `--save-task-state` | Runs one agent from a JSON configuration, asking at the terminal for approvals, answers and recovery. | `arko_agents/examples/arko_agents_cli.rs` | [Example binaries (arko_agents)](#example-binaries-arko_agents) |
| `interactive_terminal` | An offline terminal demo; no arguments. | `arko_agents/examples/interactive_terminal.rs` | [Example binaries (arko_agents)](#example-binaries-arko_agents) |
| `examples/coder_agent.json`, `examples/native-tools-agent.json` | Example agent configurations. | `arko_agents/examples/` | [Example binaries (arko_agents)](#example-binaries-arko_agents) |

### `AgentConfig` (arko_agents)

`AgentConfig::builder(id, name, llm_config)` requires `role_description` and `instructions`, gives the other fields their defaults, and runs `AgentConfig::validate`. `AgentConfig::load_from_path` doesn't validate; the runtime runs `validate` again when a task starts. There is no `Default`.

| Field | Default and omitted | Notes |
|---|---|---|
| `id`, `name`, `role_description`, `instructions`, `llm_config` | Required | In `role_description`, `instructions` and the task prompt, text between `{{#output_based_only}}` and `{{/output_based_only}}` is kept (without the markers) only in output-based mode. |
| `tool_config` | `AgentToolConfig::default()` | Omitted in the configuration saved inside a task: the same, but in output-based mode. See [Tools (arko_agents)](#tools-arko_agents). |
| `environment_details_config` | Nothing included | See [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents). |
| `error_recovery_config.on_tool_execution_error` | `InformAgentAndContinue`: the error is the call's result | `PauseAndAwaitApproval` lets the turn's other calls run, then pauses once. `FailTask` fails the task at the first error; the calls it didn't run get a result saying so. |
| `llm_error_policy` | `RetryThenPauseAndAwaitApproval { max_attempts: 10 }` | See [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents). |
| `llm_retry_config` | `LLMRetryConfig::default()` | See [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents). |
| `user_command_timeout_seconds` | `None`: a waiting task waits indefinitely | Opt-in. In any awaiting state, a wait with no decision this many seconds after it started returns the task `Interrupted`, and calls awaiting approval get a result saying approval timed out. The deadline is set once per wait: `InjectContext` and commands the wait ignores don't move it. A wait is not timed after `UserCommand::HostDeciding`, nor in a run whose `HostDecisions` is `true` (see [Runtime (arko_agents)](#runtime-arko_agents)). `Some(0)` is rejected by the builder and at task start. |
| `user_reminder` | `None` | `UserReminder { content, frequency }`, appended under `# Reminder`. |
| `skills_config` | `None` | Read by `SkillManager::load`; the runtime itself reads only `load_skill_interception`. See [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents). |
| `monitoring` | Nothing monitored | See [Monitoring (arko_agents)](#monitoring-arko_agents). |
| `refusal_fallback` | `None` | Opt-in `AgentLLMConfig`; a provider other than the configured model's is rejected. See [Refusals (arko_agents)](#refusals-arko_agents). |
| `handover` | `None`: the conversation grows until the task ends | Opt-in `HandoverConfig`. See [Conversation handover (arko_agents)](#conversation-handover-arko_agents). |
| `claude_code` | `None`: `claude` on the `PATH`, none of the user's settings files (`--setting-sources=` and `--strict-mcp-config`), the agent's system prompt appended to Claude Code's | Only for a Claude Code model (`validate` rejects it otherwise). A Claude Code agent must use `Native` tool calling, and its `refusal_fallback` must be a Claude Code model (it answers one turn after a refusal that stands); built-in tools only for Claude Code models, never `AskUserQuestion`, `ExitPlanMode` or `EnterPlanMode`, and without `tool_timeout_seconds` or `parallel_safe` overrides. See [Claude Code models](../arko_agents/README.md#claude-code-models). |
| `complete_task_summary_prompt_override` | `None`: the built-in description | Replaces the description of `complete_task`'s `summary` parameter, in both tool calling modes. |
| `parsing_config_override` | `None`: tags `<`, `>`, `batched_tool_call`, escape prefix `!!--!!` | Output-based mode only. A field omitted in JSON or YAML takes its default; `escape_prefix: null` turns escaping off. |

`AgentLLMConfig` is saved as `provider`, `model_id`, a snapshot of the model, and `generation_config` (omitted: `GenerationConfig::default()`). On load, a model the arko_ai registry knows takes the registry's current data; an unknown ID keeps its snapshot; an OpenRouter slug without one loads without catalog data; any other unknown ID fails to load.

### Tools (arko_agents)

`ToolRegistry::new()` registers `complete_task`, `read_file`, `write_file`, `apply_diff`, `list_files` and `ask_user`. A task runs only the registry's tools that its configuration offers.

| `AgentToolConfig` field | Default | Omitted in JSON | Notes |
|---|---|---|---|
| `available_tool_names` | `["complete_task"]` | Required when `tool_config` is given | The tools offered and run, plus `complete_task`, which is offered whether it is listed or not. A native call to another tool gets an error result and `UnavailableToolAttempted`; in output-based mode its markup stays text. MCP, terminal and custom tools must be listed; `SkillManager::setup` adds `load_skill` itself. |
| `tool_calling_mode` | `Native` | `Native`; `OutputBased` in the configuration saved inside a task | `OutputBased` is legacy: calls are parsed from the response text, one call or one batch of calls to one parallel-safe tool per response. |
| `default_auto_approve_tool_use` | `false` | `false` | Every call waits for approval, `complete_task` included, except `ask_user`, which runs without approval unless its own entry sets `auto_approve_tool_use: Some(false)`. |
| `default_out_of_workspace_path_policy` | `AutoReject` | `AutoReject` | `RequireApproval` asks, even for auto-approved tools; `AutoApprove` allows. Only `is_path` parameters are checked, lexically against the working directory; symbolic links are not followed, so it is not a sandbox. |
| `one_tool_call_per_turn` | `false` | `false` | `true` disables parallel calls in the request where the provider can (OpenAI, Anthropic and OpenRouter; not Gemini) and runs a response's calls one at a time. `parallel_tool_calls: Some(false)` in the model's `tool_config` means the same. |
| `tool_timeout_seconds` | `Some(1200)` (`DEFAULT_TOOL_TIMEOUT_SECONDS`) | `Some(1200)`; `null` is no limit | Host-side limit, not a tool parameter the model sees. A call over it is stopped (a command it started is killed) and handled as a tool error. `0` is rejected. Tools whose `uses_host_timeout` is `false` are exempt: the terminal tools, and arko_workflows' `run_workflow_agent` and `wait_for_message`. |
| `tool_specific_configs` | Empty | Empty | A `PerToolConfigOverride` per tool name, below. |

`PerToolConfigOverride` fields all default to `None`, which inherits the agent's setting: `auto_approve_tool_use`, `parallel_safe` (instead of the tool definition's), `out_of_workspace_path_policy`, and `tool_timeout_seconds` (omitted: inherit; `null`: no limit; a number: that limit, which applies to exempt tools too).

A custom tool sets its own behaviour: `ToolDefinition::parallel_safe` (omitted in JSON: `false`), `ToolParameterDefinition::is_path` (omitted in JSON: `false`), and the `Tool` methods `uses_host_timeout`, `interruptible` (`false`: `CancelTask` and `SuspendTask` wait for the call to return) and `allow_result_interception`, each `true` by default.

How a native response's calls run (an output-based response holds one call, or one batch of calls to one tool, run one after another):

- Native calls get exactly one result each. Calls run in call order under a fair read/write lock: a parallel-safe call starts once every earlier call that isn't parallel-safe has finished, and any other call waits for every earlier call. `read_file`, `list_files` and `load_skill` are parallel-safe; the other built-in tools and the terminal tools are not; an MCP tool is parallel-safe only when its server marks it `readOnlyHint: true`.
- A call awaiting approval counts as unfinished: the calls after it that must wait are held with it until the approvals are decided.
- Calls after one that ends the turn (`complete_task`, `ask_user`, a tool error under `FailTask`, a middleware's `FailTask` or `CancelTask`) are not run, and their results say so; parallel-safe calls already running beside it finish. Among the calls run once approvals are decided, only a call that ends the task stops the others. `CancelTask` and `SuspendTask` stop the calls still running (their results say so) unless one of them isn't interruptible.
- `complete_task` always runs alone, after the calls before it. After an earlier call of its turn was rejected or failed, it doesn't complete the task, and its result says so.
- A response without a tool call doesn't end the task: a note asks the model to continue or call `complete_task`, and the response counts as a model-call recovery attempt (under `LLMErrorPolicy::PauseAndAwaitApproval` the task pauses at once).
- A `tool_config` in the model's `GenerationConfig` supplies `tool_choice` and `parallel_tool_calls`; its tool list is ignored, with a warning in the log. Without one, no tool choice is sent.

In both modes:

- Tool results have no size limit: `read_file` returns the whole file, `list_files` the whole tree, and command and MCP output are returned in full.
- `write_file` is not atomic. `apply_diff` writes the blocks that applied even when others didn't; its result says "Applied N of M" with a line per block. Only a diff where no block applied is a tool error.
- End-user command tools expand `~` and `$VAR` in their command template, split it on whitespace (no shell, no quoting), and run it in the working directory with a null stdin; the process is killed when the call is stopped. The result is stdout; a non-zero exit is a tool error carrying the exit code, stdout and stderr. Tool parameters are never expanded: an `is_path` value `~/notes` names a directory called `~`.

`InterceptionFailurePolicy` (in the `InterceptorConfig` given to `run_task`) defaults to `UseRawResult`. `RetryThenUseRawResult { max_attempts }` retries first; `InformAgent` and `RetryThenInformAgent { max_attempts }` add a note that the result is the full one; `max_attempts` 0 counts as 1. Error results are never intercepted, nor are the results of `complete_task`, `ask_user` and tools whose `allow_result_interception` is `false` (`list_files`, the terminal tools). `load_skill` results are intercepted only with `SkillsConfig::load_skill_interception`, whose policy then applies. See [Result interception and execution middleware](../arko_agents/README.md#result-interception-and-execution-middleware).

### Environment details and reminders (arko_agents)

`EnvironmentDetailsConfig` has `current_working_directory`, `file_tree`, `current_datetime` and `operating_system`, each a `BuiltInEnvironmentDetail`, plus `file_tree_mode`. None is included by default.

| Setting | Default | Notes |
|---|---|---|
| `BuiltInEnvironmentDetail::include` | `false` | Also when omitted in JSON or YAML. |
| `BuiltInEnvironmentDetail::placement` | `SystemPrompt` | Also `InitialTaskPrompt` and `EachUserMessage`. The system prompt is written once, when the task starts: a date there is labelled `Task Start Datetime (UTC)`, elsewhere `Current Datetime (UTC)`. `EachUserMessage` details go with the task prompt and with the user messages that carry tool results, answers, notes and context updates, in native mode as text after the results. Three messages don't carry them: file-tree updates, the output-based results added when a wait ends without a decision, and the prompt `Task::prepare_for_resumption_from_completed` adds. A context update doesn't repeat them after a message of the same turn that carried them. |
| `BuiltInEnvironmentDetail::depth` | `None`: 3 | File tree only. |
| `file_tree_mode` | `Diff` | JSON `"diff"`, `"full"`. `Diff` sends a full snapshot before the first request, then a message listing changes before later requests, whatever the placement. `Full` renders the whole tree at `placement`. |
| `UserReminder::frequency` | `InitialTaskPromptOnly` | Also `EveryUserMessage`, and `EveryNUserMessages(n)`: the first user message and every nth after it, counting the messages the runtime adds. `n = 0` is rejected by the builder and at task start. |

Developer-defined details (`DeveloperDefinableEnvironmentDetail`) are passed to `Task::new`; a dynamic provider isn't saved with the task. A task that would show a detail without content fails at start with `ArkoAgentsError::Configuration`; a dynamic provider that returns an error leaves the detail out of that message.

### Model-call recovery (arko_agents)

| `LLMErrorPolicy` | Attempts | When recovery ends |
|---|---|---|
| `RetryThenPauseAndAwaitApproval { max_attempts }` (default, 10) | `max_attempts` (0 counts as 1) | Pauses in `AwaitingErrorRecoveryApproval` |
| `RetryThenInformAgent { max_attempts }` | `max_attempts` (0 counts as 1) | Returns the task `Interrupted` |
| `InformAgentAndContinue` | No limit within the elapsed budget; 10 for unusable responses | Returns the task `Interrupted` |
| `PauseAndAwaitApproval` | 1 | Pauses at the first failure |

| `LLMRetryConfig` field | Default | Notes |
|---|---|---|
| `initial_delay_ms` | 2,000 | Must be above 0. |
| `max_delay_ms` | 64,000 | At least `initial_delay_ms`. A delay the provider asks for may exceed it. |
| `max_elapsed_ms` | 300,000 | Budget from the first failure of an episode: it bounds the waits between attempts and the decision to retry. It never cuts off a request in flight, and it stops counting while an accepted response streams. Must be above 0. |
| `jitter` | `true` | Each backoff delay is drawn between half and all of its step; a provider's delay gets up to half a step added. |
| `rate_limit_retry` | `Some(LLMRateLimitRetryConfig::default())`: 1 s to 60 s backoff, `max_wait_ms` 900,000 | Under the `RetryThen*` policies, rate limits and (without `capacity_retry`) capacity errors are waited out outside the attempt limit and the budget. A rate limit asking for more than `max_wait_ms`, or arriving after `max_wait_ms` of rate-limit waits in one episode, is handled as a usage limit; a capacity error asking for longer gets ordinary recovery. Under the other policies, and with JSON `null`, both get ordinary recovery. |
| `capacity_retry` | `None` | Opt-in `LLMCapacityRetryConfig` (30 s to 300 s backoff): under the `RetryThen*` policies, capacity errors are retried until the service recovers. |
| `usage_limit_policy` | `Pause` | A wait longer than the prompt-cache lifetime costs one uncached request when it ends. A plan usage limit ends recovery under every policy. The opt-in `WaitUntilReset { max_wait_ms }`, under every policy, waits until the reported reset plus 1 s (and up to 30 s of jitter while `jitter` is on) and retries unchanged, if the provider reports a reset and the reset plus 31 s fits in `max_wait_ms`; otherwise recovery ends as under `Pause`. |
| `stream_idle_timeout_ms` | `None` | Opt-in cut-off of this agent's streamed responses after that long without data (arko_ai's `LLMClient::with_stream_idle_timeout`); the cut is a truncated stream, which recovery resends. Make it generous. `0` is rejected. |

A failed request (a connection failure, HTTP 408, 409, 429 or 5xx, a truncated stream) is resent unchanged, with nothing added to the conversation. A response the runtime can't use (native: no tool call; output-based: no call, or calls it can't read) is retried at once, without backoff, with a note to the model. Errors that can't be retried, such as authentication failures, invalid requests and refusals, end recovery at once; the pause records an `ErrorRecoveryCause`. Before each OpenAI request, the runtime refreshes ChatGPT OAuth credentials that expire within 5 minutes. See [Model-call recovery](../arko_agents/README.md#model-call-recovery) and [Errors and retries](../arko_ai/README.md#errors-and-retries).

A response that ends at the output-token limit emits `OutputTokenLimitReached` and is then handled as usual: nothing continues it automatically, and a call the stream announced but the response doesn't hold gets `ToolRejected`.

### Monitoring (arko_agents)

A call that reports plan usage (the ChatGPT backend on ChatGPT OAuth calls) emits `PlanUsageReported`, and a call served by another model than the one requested emits `ModelRerouted`. `AgentConfig::monitoring` adds opt-in monitors; both default to `None`:

| Setting | Default | Notes |
|---|---|---|
| `plan_usage: Some(PlanUsageThreshold { percent, action })` | `action`: `Pause` | `percent` is required, above 0 and at most 100. Checked before each request against the latest report; `PlanUsageThresholdReached` once per window and reset. `Pause` pauses once per window and reset; `SlowDown { delay_ms }` waits before every request while the most-used window stays at or above `percent` (a Claude Code agent holds each tool call instead); it spends the same usage over more time, so it helps only when the reset is close, and a delay at or past the prompt-cache lifetime makes each request miss the cache (warned once per run). See [Choosing a plan-usage action](../arko_agents/README.md#monitoring). |
| `cache: Some(CacheMonitorConfig::default())` | `warmup_calls` 2, `window_calls` 5, `warn_below` `Some(0.5)`, `pause_below` `None` | `CacheEfficiencyReported` after each judged call (not the warm-up calls, calls without a cache figure, or refusal-fallback calls). `CacheEfficiencyLow` once per drop of the rolling efficiency below `warn_below`, after `window_calls` judged calls; `pause_below` also pauses before the next request. |

A monitor pause waits in `AwaitingErrorRecoveryApproval` (cause `Monitor`) under the pausing error policies, and returns the task `Interrupted` under the others. Approving it adds no error note.

### Conversation handover (arko_agents)

| Setting | Default | Notes |
|---|---|---|
| `HandoverConfig::at_input_tokens` | `None` | Hand over once a request's input reaches this many tokens. At least 1. |
| `HandoverConfig::at_context_fraction` | `None` | Above 0 and at most 1, of the model's input limit; ignored for a model whose limit is unknown. With both set, the lower threshold applies; at least one must be set. |
| `HandoverConfig::report_request` | `DEFAULT_REPORT_REQUEST` | The user message asking for the report; not blank. |
| `AgentRuntime::with_handover_hook` | `HandoverSeed::for_task`: the fresh conversation restates `initial_task_prompt` | The hook sees the task with the old conversation. |

Checked before each model request (not a refusal fallback's) against the latest request of the current conversation, and only where a user message can go. During the report turn every tool call except `complete_task` is rejected; a reply without text gets a note. The fresh conversation keeps the system prompt, the pending context injections and the task's usage; tools observing the conversation are reset, the file tree is sent in full again, and the cache monitor judges only the fresh conversation. `initial_task_prompt` is unchanged. `UserCommand::RequestHandover` starts one without a threshold. See [Conversation handover](../arko_agents/README.md#conversation-handover).

### Refusals (arko_agents)

Model-call recovery does not automatically retry a refusal unchanged. `RefusalReceived` and `RefusedInput` are emitted, and the refusal is handled as an error that can't be retried by the agent's own recovery: the task pauses (cause `Refusal`) or is returned `Interrupted`, as the error policy decides. A model's own refusal (`RefusalSource::Model`) stays in history without its tool calls; classifier and content-filter refusals are removed. `UserCommand::ResolveRefusal` (while the task awaits recovery approval) or `Task::resolve_refusal` (on a returned task) changes the refused input; `ApproveErrorRecovery` (which adds its note) and `ContinueAfterError` continue without changing it. With the opt-in `refusal_fallback`, the same input is instead sent once to the fallback model (`RefusalFallback`): the refused turn is not kept, and only the fallback's own refusal makes the refusal stand. See [Refusals](../arko_agents/README.md#refusals).

### Runtime (arko_agents)

`AgentRuntime::new(client).run_task(task, tool_registry, event_throttle_ms, simplified_tool_events, skill_prompt_context, interceptor_config, execution_middleware)` returns the event receiver, the command sender, and a join handle that resolves to `Result<Task, ArkoAgentsError>`.

| Parameter or setter | Unset means | Notes |
|---|---|---|
| `event_throttle_ms` | `None` or `0`: every text and reasoning chunk is its own event | `Some(ms)` coalesces chunks. |
| `simplified_tool_events` | Required `bool` | Output-based mode only: `true` leaves out `ToolParamName`, `ToolParamValueChunk` and `ToolParamEnd`. |
| `skill_prompt_context` | No skills in the system prompt | From `SkillManager::setup`. |
| `interceptor_config` | No result interception | `InterceptorConfig { interceptor, failure_policy }`. |
| `execution_middleware` | None | `before` and `after` every tool execution; for output supplied with an approval, only `after` (with `CallerProvided`). |
| `AgentRuntime::with_turn_context(source)` | No turn context | Asked before each model request (not a refusal-fallback request); its text joins that turn's context update. |
| `AgentRuntime::with_host_decisions(decisions)` | `HostDecisions` `false`: waits are timed as configured | While `true`, `user_command_timeout_seconds` times no wait of the run. Not saved with the task; clones share the setting. |

The event channel holds 100 events and the command channel 10. A host that stops reading events blocks the run; a host that drops the event receiver ends it with `ArkoAgentsError::ChannelSend`, losing the task value. When every command sender is dropped, the task is returned `Interrupted` at its next wait or turn boundary (tools hold only weak senders).

- `InjectContext` is added at the next turn boundary as one `# Workflow context update` user message, with all context sent before it, in order. Context taken before the task stops is added before it is returned, or kept in `Task::pending_context_injections` when a user message can't go there yet.
- `CancelTask` ends the task `Cancelled` with `CANCELLED_BY_USER`; `SuspendTask` returns it in its current state, to be saved and run again.
- `ApproveErrorRecovery` adds a note that the user approved continuing (not after a monitor pause or for a pending refusal fallback). `ContinueAfterError { note: None }` resends the failed request unchanged, or after a tool-error pause sends the results that were waiting. `RejectErrorRecovery` fails the task.
- In `SubmitBatchToolApprovals`, a call without a decision is rejected.
- A command the current state doesn't take is ignored.

See [Running tasks](../arko_agents/README.md#running-tasks).

### Skills, MCP and terminals (arko_agents)

| Setting | Default | Notes |
|---|---|---|
| `SkillsConfig::workspace_source` | `None` | A `SkillSource { path, include, exclude }` whose `path` is joined to the `workspace_root` given to `SkillManager::load`. |
| `SkillsConfig::global_sources` | Empty | Paths are used as written; `~` is not expanded. Identical skills load once; another skill with a name already taken gets a numeric suffix. |
| `SkillsConfig::on_invalid_skill` | `Warn`: an invalid skill, or a missing directory, is skipped | `Error` makes `SkillManager::load` fail. |
| `SkillsConfig::load_skill_interception` | `None`: `load_skill` results are never intercepted | `Some(policy)` lets the runtime's interceptor replace them, under `policy`. |
| `McpServerConfig::args`, `McpServerConfig::env` | Empty | The server inherits the host's environment, plus `env`. |
| `TerminalConfig::new(directory)` | Shell `/bin/sh -c` (Windows `cmd.exe /D /S /C`); `output_buffer_bytes` 65,536; `max_log_bytes` 4 MiB; `excerpt_bytes` 2,000; `prompt_quiet_ms` 150; `default_timeout` `None`; `prompt_rules` for yes/no, press-a-key and password prompts; `quiet_terminal_environment` `true`; `command_guard` on (`CommandGuard::default()`) | The directory holds job records and logs and is locked by one manager. PTY jobs get `TERM=dumb`, `NO_COLOR=1`, `PAGER=cat`, `GIT_PAGER=cat` and a UTF-8 locale. The number of jobs is not limited. The command guard refuses commands and input that would delete, move, overwrite or reformat protected paths (only `dangerously_allow_all_commands` allows them) or paths outside the working directory (`allow_outside_workspace`, `allowed_paths`). |

Skills load only when the host calls `SkillManager::load` and then `setup` on the configuration the task will run with (before `Task::new`, which copies it, or on a loaded task's `agent_config`); arko_workflows does this for its agents. When there are on-demand skills, `setup` registers `load_skill`, adds it to `available_tool_names`, and auto-approves it unless the agent already has a per-tool entry for it.

MCP tools are named `mcp__<server>__<tool>` and return text only: images are dropped, and a result without text gives its structured content as JSON. They have no `is_path` parameters, so the workspace path check doesn't apply to them. `McpManager::register_tools` doesn't offer them (add `tool_names()` to `available_tool_names`), and only the host tool time limit bounds their calls. A server's stderr goes to the log at info level; a server that exits logs one warning, and its tools then fail as tool errors. Terminal tools are registered with `TerminalManager::register_tools` and must be listed in `available_tool_names` too. See [MCP servers](../arko_agents/README.md#mcp-servers), [Skills](../arko_agents/README.md#skills), and [Configuration](interactive-terminals.md#configuration).

### Persisted data (arko_agents)

- Tasks, configurations and tool definitions (`save_to_path`) are saved as pretty JSON through a temporary file and a rename. Tasks carry no format version; their optional fields default, so tasks saved by earlier versions load.
- A task's configuration snapshot without a tool calling mode loads as `OutputBased`, the mode it ran with; a standalone configuration without one loads as `Native`.
- A task runs with the configuration and system prompt saved in it: editing the agent's configuration doesn't change a saved task.
- `Failed { reason: "Task cancelled by user." }` loads as `Cancelled`; a pause saved without a cause loads with `ErrorRecoveryCause::Unknown`; a call saved without an approval reason loads with an unrecorded one.
- `allow_batched_tool_calls` is ignored (`false` logs a warning once); a per-tool `allow_batched_calls: false` loads as `parallel_safe: Some(false)` and `true` is ignored; a tool definition's `supports_batched_calls` is ignored.
- Dynamic environment-detail providers aren't saved: set them again on a loaded task.
- Terminal job records left `running` by a stopped host load as `interrupted`; processes are never reattached.

### Example binaries (arko_agents)

`arko_agents_cli` loads a `.env` file, then reads provider credentials from the environment (`LLMClient::from_env_with`).

| Flag | Default | Notes |
|---|---|---|
| `-c`, `--agent-config`; `-w`, `--workspace-dir`; `-p`, `--prompt` | Required | The workspace is created if missing, then canonicalized. |
| `--tool-calling-mode` | The configuration's | `native`, or `output_based` (also `outputbased`, `output-based`). |
| `--event-throttle-ms` | Unset: no coalescing | |
| `--simplified-tool-events <true\|false>` | `true` | |
| `--user-command-timeout SECONDS` | The configuration's | |
| `--mcp-config PATH` | No MCP | Adds the servers' tools to the agent's tools; with at least one server, switches the agent to native mode. |
| `--openai-auth-json PATH` | Environment credentials | ChatGPT OAuth from a Codex `auth.json` (`with_codex_auth_file`); the `OPENAI_OAUTH_*` variables are then not read. |
| `--save-task-state PATH` | Not saved | The final task, when the run returns it. |

The CLI asks at the terminal for every tool approval (a batch call by call), every `ask_user` question, and every error-recovery pause (approve or reject). It loads the configuration's skills and continues without them if loading fails. `examples/coder_agent.json` uses Anthropic `claude-opus-5-5` in native mode (it names no mode), asks approval for every call, and uses `PauseAndAwaitApproval` for both tool and model errors; `examples/native-tools-agent.json` uses OpenAI `gpt-6-sol` and auto-approves every tool. `interactive_terminal` takes no arguments; `ARKO_TEST_PYTHON` selects the Python interpreter (default `python3`, or `python` on Windows).


## arko_workflows

Multi-agent workflow orchestration on top of arko_agents: application code runs agents in order, in parallel or in the background, with shared memory, messaging, budgets, hooks, approval and error-recovery policies, context freshness, completion verification, distillation, durable steps and checkpoints.

Guides: [arko_workflows README](../arko_workflows/README.md), [API index](../arko_workflows/INDEX.md), [DeepCoder example](../arko_workflows/examples/deepcoder/README.md), [memory reads](context-efficiency.md#memory-reads) and [tool-result distillation](context-efficiency.md#tool-result-distillation) in the context management guide.

### Map (arko_workflows)

Agents are named by their registration name (the one given to `WorkflowBuilder::agent`) everywhere: policies, lists, hooks, results and tools.

**Crate**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_workflows` modules | `budget`, `context`, `distillation`, `durable`, `error`, `freshness`, `hooks` (`hooks::context`), `memory` (`memory::tools`), `messaging` (`registry`, `tools`, `wait`), `persistence` (`persistence::state`), `policy`, `services`, `user_input`, `verification`, `voting`, `workflow` (`cancel`, `config`, `engine`, `result`); `python` with the `python` feature. `workflow::engine` exports no items. | `arko_workflows/src/lib.rs` | [Imports and feature map](../arko_workflows/INDEX.md#imports-and-feature-map) |
| Root re-exports | `Workflow`, `WorkflowBuilder`, `AgentHandle`, `WorkflowError`, `WorkflowCancelHandle`, `CancelCause`, `Cancellation`, `SupportedPlatform`, `WorkflowDependency`, `WorkflowMetadata`, `AgentResult`, `AgentRunStatus`, `WorkflowEvent`, `WorkflowResult`. | `arko_workflows/src/lib.rs` | [Imports and feature map](../arko_workflows/INDEX.md#imports-and-feature-map) |
| `prelude` | The root re-exports except `AgentHandle`, plus each module's configuration and data types (`CheckpointRetention` included). Not in it: `SharedContext`, `HookContext`, `HookRegistry`, `MemoryStore`, `MessageBus`, `BallotStore`, `DurableStore`, `PersistenceManager`, `FreshnessSnapshot`, the checkpoint state types, the tool types, and the constants other than `VERIFIER_OUTPUT_CONTRACT`. | `arko_workflows/src/lib.rs` | [Imports and feature map](../arko_workflows/INDEX.md#imports-and-feature-map) |
| Feature `python` | Off by default; compiles the PyO3 bindings. The crate builds no `cdylib` and has no `pyproject.toml`, so it is not packaged as a Python extension. | `arko_workflows/Cargo.toml` | [Installation](../arko_workflows/README.md#installation) |

**Building a workflow**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::builder`, `WorkflowBuilder::new` (`Default`) | An empty builder. | `arko_workflows/src/workflow/mod.rs` | [Builder signatures and defaults](../arko_workflows/INDEX.md#builder-signatures-and-defaults) |
| `WorkflowBuilder::name`, `WorkflowBuilder::workspace`, `WorkflowBuilder::llm_client`, `WorkflowBuilder::agent` | The required settings: workflow name, existing workspace directory, shared `Arc<LLMClient>`, and at least one agent under a unique registration name. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::memory` | Memory tiers, from a `MemoryConfig`. | `arko_workflows/src/workflow/mod.rs` | [Memory](#memory-arko_workflows) |
| `WorkflowBuilder::budget`, `WorkflowBuilder::budget_warning`, `WorkflowBuilder::agent_budget` | Workflow spending limit, one-time warning threshold, and one agent's cumulative limit, in dollars. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::hook` | Registers a synchronous `HookCallback` at a `HookPoint`. | `arko_workflows/src/workflow/mod.rs` | [Hook registration and results](../arko_workflows/INDEX.md#hook-registration-and-results) |
| `WorkflowBuilder::persistence`, `WorkflowBuilder::ephemeral` | Turns on checkpoints, task and event files, a persistent run ID and durable steps; `ephemeral` clears the persistence and resume settings made so far. | `arko_workflows/src/workflow/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `WorkflowBuilder::resume_from`, `WorkflowBuilder::rename_memory_tier_on_resume`, `WorkflowBuilder::drop_memory_tier_on_resume` | Restores a checkpoint file at `build`; maps a checkpoint memory tier to a configured one, or discards it. | `arko_workflows/src/workflow/mod.rs` | [Resume](#resume-arko_workflows) |
| `WorkflowBuilder::metadata` | Sets the `WorkflowMetadata`. | `arko_workflows/src/workflow/mod.rs` | [Workflow metadata](../arko_workflows/INDEX.md#workflow-metadata) |
| `WorkflowBuilder::context_freshness` | Turns on context freshness tracking with a `ContextFreshnessConfig`. | `arko_workflows/src/workflow/mod.rs` | [Context freshness](#context-freshness-arko_workflows) |
| `WorkflowBuilder::global_tool`, `WorkflowBuilder::agent_tool` | Adds an `arko_agents` `Tool` for every agent, or for one agent, and adds its name to that agent's available tools. | `arko_workflows/src/workflow/mod.rs` | [Build validation and tool composition](../arko_workflows/INDEX.md#build-validation-and-tool-composition) |
| `WorkflowBuilder::tool_result_interceptor`, `WorkflowBuilder::agent_tool_result_interceptor`, `WorkflowBuilder::interception_failure_policy` | Tool-result interceptors for every agent or one agent, and the `InterceptionFailurePolicy` applied to each stage of an agent's interceptor chain. | `arko_workflows/src/workflow/mod.rs` | [Build validation and tool composition](../arko_workflows/INDEX.md#build-validation-and-tool-composition) |
| `WorkflowBuilder::completion_verifier`, `WorkflowBuilder::distiller` | Adds a completion verifier or a tool-result distiller. | `arko_workflows/src/workflow/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `WorkflowBuilder::workflow_services` | Enables agent-to-agent delegation through `run_workflow_agent`. | `arko_workflows/src/workflow/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::messaging`, `WorkflowBuilder::disable_messaging` | Sets whom each agent may message (`MessagingPolicy`), or offers no agent the messaging tools. | `arko_workflows/src/workflow/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::wait_for_message_timeout`, `WorkflowBuilder::agent_wait_for_message_timeout` | The `wait_for_message` time limit, for the workflow or one agent. | `arko_workflows/src/workflow/mod.rs` | [Waiting for a message](../arko_workflows/INDEX.md#waiting-for-a-message) |
| `WorkflowBuilder::user_input_handler` | Answers agents' `ask_user` questions in-process. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::approval_policy`, `WorkflowBuilder::error_recovery_policy` | How tool-approval requests and error-recovery pauses are answered. | `arko_workflows/src/workflow/mod.rs` | [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows) |
| `WorkflowBuilder::max_parallel_agents`, `WorkflowBuilder::fail_fast` | Concurrency cap and fail-fast for `run_agents_parallel`. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::enable_voting` | Creates the ballot store and offers `cast_vote` to the named agents. | `arko_workflows/src/workflow/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::cancel_grace_period` | How long a stopped run waits for each running agent before aborting it. | `arko_workflows/src/workflow/mod.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `WorkflowBuilder::build` | Validates the settings, restores a checkpoint when set, and fires `PreWorkflow`; returns `BuilderMissing` or `BuilderInvalid` for a bad configuration. Runs no agent. | `arko_workflows/src/workflow/mod.rs` | [Build validation and tool composition](../arko_workflows/INDEX.md#build-validation-and-tool-composition) |
| `WorkflowMetadata` (`description`, `version`, `supported_platforms`, `required_dependencies`, `tags`), `SupportedPlatform` (`os`, `arch`; `new`, `arch()`, `matches_host`), `WorkflowDependency` (`name`, `description`, `executable`, `install_hint`, `os`, `arch`) | Serializable discovery metadata; every `WorkflowMetadata` field may be omitted in JSON. A nonempty `supported_platforms` list that the host matches none of fails `build`; dependencies are descriptive only. | `arko_workflows/src/workflow/config.rs` | [Workflow metadata](../arko_workflows/INDEX.md#workflow-metadata) |
| `workflow::config::WorkflowConfig` | A serializable descriptor (`name`, `workspace`, `metadata`); workflows are built with the builder, and it is not re-exported. | `arko_workflows/src/workflow/config.rs` | [Workflow metadata](../arko_workflows/INDEX.md#workflow-metadata) |

**Running agents**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow` | One run of a configured workflow, created by `build`. | `arko_workflows/src/workflow/mod.rs` | [Workflow execution and lifecycle](../arko_workflows/INDEX.md#workflow-execution-and-lifecycle) |
| `Workflow::run_agent` | Starts a new run of a registered agent and drives it to its result. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::continue_agent` | Continues a `Completed` or `WaitingForInput` result, or one whose completion its verifier couldn't review, with a prompt; keeps the task and run ID. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::resume_agent` | Continues a `Paused` result with an `AgentResume` decision. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::run_agents_parallel` | Runs `(agent, prompt)` pairs concurrently (an agent may appear more than once); returns the results in input order, or the first failure. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::agent_result` | The recorded result of a run, by the task ID an error carries. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::spawn_agent`, `AgentHandle` (`agent_name`, `wait`, `into_parts`, `Drop`) | Starts an agent without driving it. `wait` drives it like `run_agent`; `into_parts` hands the event receiver, command sender and join handle to the caller; dropping the handle unused aborts the agent and records no result. | `arko_workflows/src/workflow/mod.rs` | [AgentHandle](../arko_workflows/INDEX.md#agenthandle) |
| `Workflow::result`, `Workflow::subscribe_events`, `Workflow::agent_status` | A `WorkflowResult` snapshot; a broadcast of `WorkflowEvent`s (1,024 buffered, no replay); a registered agent's `AgentStatus`. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `Workflow::pause`, `Workflow::resume`, `Workflow::is_paused` | The pause flag: managed drivers stop before their next event, and model and tool work in progress continues. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `Workflow::finish` | Fires `PostWorkflow`, saves a checkpoint and returns the result; on a stopped run, saves a checkpoint and returns the stop's error. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |

**Stopping a run**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::cancel_handle`, `Workflow::cancel`, `Workflow::is_cancelled`, `Workflow::cancellation` | Stops the run, or reports whether and why it stopped. | `arko_workflows/src/workflow/mod.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `WorkflowCancelHandle` (`cancel`, `is_cancelled`, `cancellation`, `cancelled`) | A clonable handle usable from any task or thread. `cancel` stops the run and fires `OnWorkflowCancelled`, or returns `false` when the run is already stopped; `cancelled` resolves once it is stopped. | `arko_workflows/src/workflow/cancel.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `CancelCause` (`Requested`, `AbortedByHook { hook_point, reason }`, `to_error`, `agent_reason`), `Cancellation` (`cause`, `timestamp`) | Why and when the run stopped; recorded in `WorkflowResult` and in checkpoints. | `arko_workflows/src/workflow/cancel.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `workflow::cancel::DEFAULT_CANCEL_GRACE_PERIOD` | 30 seconds. | `arko_workflows/src/workflow/cancel.rs` | [Builder defaults](#builder-defaults-arko_workflows) |

**Approvals and error recovery**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ApprovalPolicy` (`ApproveAll`, `RespectAgentConfig`, `Handler`) | How calls an agent asks approval for are answered; default `ApproveAll`. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows) |
| `ToolApprovalHandler::decide`, `ToolApprovalRequest` (`workflow_run_id`, `agent_run_id`, `agent_name`, `tool_call_id`, `tool_name`, `parameters`, `tool_description`, `reason`), `ToolApprovalOutcome` (`Approve`, `Reject { reason }`, `Pause`) | Extension trait for `ApprovalPolicy::Handler`: gets each call awaiting approval, with its `ApprovalReason`, and approves, rejects or pauses it. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery) |
| `ErrorRecoveryPolicy` (`Automatic`, `Limited { max }`, `ReturnToCaller`) | How error-recovery pauses are answered; default `Automatic`. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows) |
| `AgentPause` (`ToolApproval { calls }`, `ErrorRecovery { error, cause }`, `of`, `kind`) | Why an agent paused, as `WorkflowError::AgentPaused` carries it; `kind` is `"tool_approval"` or `"error_recovery"`. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery) |
| `AgentResume` (`ToolApprovals`, `ContinueAfterError { note }`, `FailAfterError { reason }`, `ResolveRefusal`) | The decision `resume_agent` applies; a call without a decision is rejected. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery) |

**Results and events**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentResult` (`agent_run_id`, `agent_name`, `status`, `summary`, `error`, `task`), `total_cost`, `total_input_tokens`, `total_output_tokens` | One run's outcome with its `arko_agents::Task`; the totals cover the whole task, and `total_cost` is the billed amount. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |
| `AgentRunStatus` (`Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`, `Paused`), `is_paused`, `of_state` | A run's status, from its final task state; `is_paused` is true for `WaitingForInput` and `Paused`. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |
| `WorkflowResult` (`workflow_run_id`, `workflow_name`, `metadata`, `total_cost`, `cost_ledger`, `agent_results`, `cancellation`) | A snapshot of the run: its results, total billed cost, ledger, and stop if any. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |
| `WorkflowEvent` (`timestamp`, `workflow_run_id`, `agent_name`, `agent_run_id`, `event`) | An `arko_agents` agent event with the workflow run ID, agent name and agent run ID, as broadcast and logged. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |

**Hooks and human input**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `HookPoint::PreWorkflow`, `HookPoint::PostWorkflow` | The end of a successful `build`; `finish`. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PreAgentSpawn` | Before a run, spawn, continuation or resume starts. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PostAgentCompletion`, `HookPoint::PostAgentFailure`, `HookPoint::PostAgentCancellation`, `HookPoint::PostAgentInterruption`, `HookPoint::OnUserInputRequired`, `HookPoint::OnAgentPaused` | A run's task was finalized as `Completed`, `Failed`, `Cancelled`, `Interrupted`, `WaitingForInput` or `Paused`. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PreToolExecution`, `HookPoint::PostToolExecution` | Before and after each tool call; `Skip` at `PreToolExecution` rejects the call. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PostToolResultInterception` | The managed driver received an intercepted tool result. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnWorkflowPause`, `HookPoint::OnWorkflowResume` | The pause flag changed. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnWorkflowFailed` | A run call is about to return a failure (not a pause, and not `Cancelled`); at most once per call. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnWorkflowCancelled` | A cancel handle stopped the run; its result is ignored. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnBudgetWarning`, `HookPoint::OnBudgetExceeded` | A recorded cost reached the warning threshold for the first time, or left spending above the limit. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnAgentMessage` | `send_message` queued a message. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnFreshnessConflict` | A tracked write made another active run's read stale; once per stale reader. `"OnMesiConflict"` loads as this variant. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookResult` (`Continue`, `Abort(String)`, `Skip`), `HookCallback` | A hook's answer; `HookCallback` is `Arc<dyn Fn(&HookContext) -> HookResult + Send + Sync>`. | `arko_workflows/src/hooks/mod.rs` | [Hook registration and results](../arko_workflows/INDEX.md#hook-registration-and-results) |
| `hooks::context::HookContext` (`timestamp`, `hook_point`, `workflow_run_id`, `workflow_name`, `agent_run_id`, `agent_name`, `tool_name`, `tool_call_id`, `tool_params`, `tool_result`, `original_tool_result`, `effective_tool_result`, `agent_summary`, `approval_reason`, `error_message`, `budget_spent`, `budget_limit`, `metadata`) | What a hook receives; the optional fields are set only at the points that fill them. | `arko_workflows/src/hooks/context.rs` | [HookContext fields](../arko_workflows/INDEX.md#hookcontext-fields) |
| `hooks::HookRegistry` (`new`, `register`, `get`, `has_hooks`) | The registered hooks by point (`SharedContext::hooks`). | `arko_workflows/src/hooks/mod.rs` | [Hook registration and results](../arko_workflows/INDEX.md#hook-registration-and-results) |
| `user_input::UserInputHandler::respond`, `UserInputRequest` | Extension trait that answers an agent's `ask_user` questions; an `Err` ends the call with `WorkflowError::Channel`. | `arko_workflows/src/user_input.rs` | [UserInputHandler and UserInputRequest](../arko_workflows/INDEX.md#userinputhandler-and-userinputrequest) |

**Workflow tools** (added to an agent's tools by the workflow)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `read_memory` (`memory::tools::ReadMemoryTool`, `READ_MEMORY_TOOL_NAME`) | For agents that can read a tier. Parameters `tier`, optional `full` and `max_chars`; returns versioned JSON (`full`, `append`, `replace`, `unchanged` or `partial`). Its results are never intercepted or distilled (`allow_result_interception` is false); per-task read baselines follow `reset_context` and `observe_context`. | `arko_workflows/src/memory/tools.rs` | [Memory tools and read JSON contract](../arko_workflows/INDEX.md#memory-tools-and-read-json-contract) |
| `write_memory` (`WriteMemoryTool`, `WRITE_MEMORY_TOOL_NAME`) | For agents that can write a tier. Parameters `tier` and `content`; appends or replaces by the tier's write mode. | `arko_workflows/src/memory/tools.rs` | [Memory tools and read JSON contract](../arko_workflows/INDEX.md#memory-tools-and-read-json-contract) |
| `list_memories` (`ListMemoriesTool`, `LIST_MEMORIES_TOOL_NAME`) | For agents that can read or write a tier: lists the tiers it can access, with its access to each. | `arko_workflows/src/memory/tools.rs` | [Memory tools and read JSON contract](../arko_workflows/INDEX.md#memory-tools-and-read-json-contract) |
| `send_message` (`messaging::tools::SendMessageTool`, `SEND_MESSAGE_TOOL_NAME`) | For agents with a permitted recipient. Parameters `to`, `content`, optional `attachments_json`; accepts only recipients that are running or waiting. | `arko_workflows/src/messaging/tools.rs` | [Messaging tools](../arko_workflows/INDEX.md#messaging-tools) |
| `list_recipients` (`ListRecipientsTool`, `LIST_RECIPIENTS_TOOL_NAME`) | Offered with `send_message`: the permitted recipients with their roles and current statuses. | `arko_workflows/src/messaging/tools.rs` | [Messaging tools](../arko_workflows/INDEX.md#messaging-tools) |
| `wait_for_message` (`messaging::wait::WaitForMessageTool`, `WAIT_FOR_MESSAGE_TOOL_NAME`, both re-exported from `messaging::tools`) | For agents another agent may message. Returns at a message, a start or finish of a peer, its time limit, or a stop; the optional `timeout_seconds` only shortens it. Not subject to the agent's tool time limit (`uses_host_timeout` is false). | `arko_workflows/src/messaging/wait.rs` | [Waiting for a message](../arko_workflows/INDEX.md#waiting-for-a-message) |
| `run_workflow_agent` (`services::RUN_WORKFLOW_AGENT_TOOL_NAME`) | For allowed callers when services are enabled. Parameters `agent` and `prompt`; runs the target to its end and returns its completion summary. Not subject to the agent's tool time limit (`uses_host_timeout` is false). | `arko_workflows/src/services/mod.rs` | [Workflow services and custom tools](../arko_workflows/INDEX.md#workflow-services-and-custom-tools) |
| `cast_vote` (`voting::CAST_VOTE_TOOL_NAME`) | For the agents named in `enable_voting`. Parameters `ballot_id`, `option` and `reasoning`; one vote per agent per ballot. | `arko_workflows/src/voting/mod.rs` | [Agent tool: cast_vote](../arko_workflows/INDEX.md#agent-tool-cast_vote) |

**Memory**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `MemoryConfig` (`new`, `tier`, `global_tier`, `global_tier_with_limit`, `broadcast_tier`, `broadcast_tier_with_limit`, `tiers`, `build`) | Builds the memory tiers given to `WorkflowBuilder::memory`. | `arko_workflows/src/memory/mod.rs` | [Memory](#memory-arko_workflows) |
| `MemoryTier` (`name`, `content`, `readers`, `writers`, `write_mode`, `entries`, `max_bytes`), `text_of`, `history` | One tier: content, access, write mode, write history and optional byte limit. | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `AccessScope` (`All`, `Only(names)`, `None`), `allows` | Who may read or write a tier. | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `WriteMode` (`AppendOnly`, `FullEdit`) | Whether a write appends (after a newline) or replaces the content. | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `MemoryEntry`, `MemoryWrite` | One write with its text, as read back; one write as stored (text held once, as a range of the content or in the entry). | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `MemoryStore` (`new`, `is_empty`, `read`, `read_versioned`, `read_window`, `entries`, `write`, `write_with_context`, `list_accessible`, `access_of`, `snapshot`) | The run's store (`SharedContext::memory`): access-checked reads and writes, without freshness checks or hooks. | `arko_workflows/src/memory/mod.rs` | [Memory store](../arko_workflows/INDEX.md#memory-store) |

**Messaging**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `MessagingPolicy` (`recipients`, `keep_for_finished`, `all`, `can_message`, `cannot_message`, `keep_messages_for_finished`, `permits`, `recipients_of`) | Whom each agent may message with `send_message`, never itself, and whether messages to a finished agent are kept for its next run. | `arko_workflows/src/messaging/mod.rs` | [Messaging policy](../arko_workflows/INDEX.md#messaging-policy) |
| `AgentMessage` (`id`, `from`, `from_task_id`, `to`, `content`, `attachments`, `timestamp`, `to_task_id`, `held_by`, `read_by`), `is_held_by`, `is_for`; `MessageAttachment` (`File`, `MemoryTier`, `Transcript`, `Text`) | A queued message and its typed references. | `arko_workflows/src/messaging/mod.rs` | [Messaging](../arko_workflows/INDEX.md#messaging) |
| `MessageBus` (`new`, `from_snapshot`, `roster`, `register_active`, `unregister_active`, `inject_task`, `send`, `drain`, `pending`, `pending_for`, `acknowledge`, `hand_over`, `snapshot`) | The run's message queues and live delivery (`SharedContext::messages`). `send` checks neither the policy nor statuses, and fires no hook. | `arko_workflows/src/messaging/mod.rs` | [Message bus](../arko_workflows/INDEX.md#message-bus) |
| `messaging::WORKFLOW_CONTEXT_SENDER` | `"workflow-context"`: the sender of context notices such as freshness notices. | `arko_workflows/src/messaging/mod.rs` | [Message bus](../arko_workflows/INDEX.md#message-bus) |
| `AgentStatus` (`NotStarted`, `Running`, `Waiting(WaitReason)`, `Finished(AgentRunStatus)`), `is_reachable`, `describe`; `WaitReason` (`RateLimited`, `Retrying`, `UsageLimit`, `AwaitingApproval`, `AwaitingInput`, `AwaitingMessage`, `Paused`) | An agent's status in the run, as `agent_status`, rosters and `list_recipients` report it. | `arko_workflows/src/messaging/registry.rs` | [Agent statuses](../arko_workflows/INDEX.md#agent-statuses) |
| `WaitForMessageTimeout` (`Auto`, `Fixed(Duration)`, `resolve`, `From<Duration>`), `DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT` (270 s), `MAX_WAIT_FOR_MESSAGE_TIMEOUT` (24 h) | The `wait_for_message` limit setting and its bounds. | `arko_workflows/src/messaging/wait.rs` | [Waiting for a message](../arko_workflows/INDEX.md#waiting-for-a-message) |

**Services and voting**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `WorkflowServicesConfig` (`allowed_callers`, `allowed_targets`, `all_callers`, `all_targets`, `max_active_calls`; `allow_caller`, `allow_target`, `allow_all_callers`, `allow_all_targets`, `max_active_calls(n)`), `DEFAULT_MAX_ACTIVE_CALLS` | Which agents may call `run_workflow_agent`, which agents they may run, and the workflow-wide limit on calls at once (default 4). | `arko_workflows/src/services/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowServices` (`Clone`; `is_allowed`, `is_target_allowed`, `run_agent`) | The live services, from `SharedContext::workflow_services`; `run_agent` runs a target for a caller under the same checks as the tool. | `arko_workflows/src/services/mod.rs` | [Workflow services and custom tools](../arko_workflows/INDEX.md#workflow-services-and-custom-tools) |
| `Workflow::begin_ballot`, `Workflow::finish_ballot` | Opens a ballot among the workflow's voters, and tallies it without closing it. | `arko_workflows/src/workflow/mod.rs` | [Workflow-facing resource and ballot operations](../arko_workflows/INDEX.md#workflow-facing-resource-and-ballot-operations) |
| `BallotConfig` (`question`, `options`, `eligible_voters`, `quorum`, `consensus_threshold`, `weights`, `tie_break`), `VoteOption` (`id`, `description`), `TieBreak` (`RejectTie`, `Lexicographic`), `CastVote` (`voter`, `task_id`, `option`, `reasoning`, `timestamp`), `BallotResult` (`ballot_id`, `winner`, `consensus`, `votes`), `BallotStore` (`begin`, `finish`) | Ballot configuration, votes, results and the in-memory store (`SharedContext::ballots`); ballots are not checkpointed. | `arko_workflows/src/voting/mod.rs` | [Structured voting](../arko_workflows/INDEX.md#structured-voting) |

**Context freshness**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ContextFreshnessConfig` (`block_on_conflict`, `tool_effects`, `tool_effect`) | Which tool calls read or write which resources, and whether a stale write is blocked. | `arko_workflows/src/freshness/mod.rs` | [Context freshness](#context-freshness-arko_workflows) |
| `ToolResourceEffect` (`ReadFile`, `ReadDirectoryTree`, `WriteFile`, each with `parameter`) | The effect of a mapped tool, naming its path parameter. | `arko_workflows/src/freshness/mod.rs` | [Context freshness](../arko_workflows/INDEX.md#context-freshness) |
| `ResourceId` (`File`, `DirectoryTree`, `MemoryTier`, `Custom { namespace, key }`), `ActorId` (`task_id`, `agent_name`, `new`) | A tracked resource, and a reader or writer (one agent run). | `arko_workflows/src/freshness/mod.rs` | [Context freshness](../arko_workflows/INDEX.md#context-freshness) |
| `FreshnessConflict` (`resource`, `observed_version`, `current_version`, `last_writer`), `FreshnessSnapshot` | A stale read; the tracker's saved state (private fields). | `arko_workflows/src/freshness/mod.rs` | [Context freshness](../arko_workflows/INDEX.md#context-freshness) |
| `ResourceTracker` (`new`, `from_snapshot`, `snapshot`, `config`, `resource_for_tool`, `record_read`, `check_write`, `record_write`, `clear_actor`, `finish_actor`, `resume_actor`, `changed_since_read`) | The run's tracker (`SharedContext::resource_tracker`). | `arko_workflows/src/freshness/mod.rs` | [Resource tracker](../arko_workflows/INDEX.md#resource-tracker) |
| `Workflow::register_resource_read`, `Workflow::mark_resource_dirty` | Records a read, or a write that notifies stale active readers, of any resource from application code; `BuilderInvalid` without freshness. | `arko_workflows/src/workflow/mod.rs` | [Workflow-facing resource and ballot operations](../arko_workflows/INDEX.md#workflow-facing-resource-and-ballot-operations) |

**Completion verification and distillation**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `CompletionVerifierConfig::correctness`, `CompletionVerifierConfig::efficiency` | A gate on target agents' `complete_task` calls, reviewed by another registered agent; `Efficiency` is advisory. | `arko_workflows/src/verification/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `CompletionVerifierConfig::target`, `CompletionVerifierConfig::instructions`, `CompletionVerifierConfig::minor_rejection_cap`, `CompletionVerifierConfig::major_rejection_cap`, `CompletionVerifierConfig::reuse_conversation`, `CompletionVerifierConfig::include_prior_findings`, `CompletionVerifierConfig::format_retries`, `CompletionVerifierConfig::review_retries`, `CompletionVerifierConfig::repair_context` | The verifier's setters; each sets the field of the same meaning. | `arko_workflows/src/verification/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `CompletionVerifierConfig` fields (`name`, `verifier_agent`, `target_agents`, `kind`, `instructions`, `max_minor_rejections`, `max_major_rejections`, `reuse_conversation`, `repair_context`, `include_prior_findings`, `max_format_retries`, `max_review_retries`), `VerificationKind`, `RepairContextConfig` (`max_bytes`) | The verifier configuration as stored and serialized. | `arko_workflows/src/verification/mod.rs` | [CompletionVerifierConfig, VerificationKind, and RepairContextConfig](../arko_workflows/INDEX.md#completionverifierconfig-verificationkind-and-repaircontextconfig) |
| `VERIFIER_OUTPUT_CONTRACT`, `VerificationDecision` (`Approved`, `RejectedMinor`, `RejectedMajor`), `VerificationRecord` (`timestamp`, `target_task_id`, `target_agent`, `verifier_name`, `verifier_agent`, `verifier_task_id`, `decision`, `feedback`) | The verdict format verifiers must return (first non-empty line `ACCEPTED`, `REJECTED_MINOR` or `REJECTED_MAJOR`), and the recorded verdicts. | `arko_workflows/src/verification/mod.rs` | [Verdict protocol and records](../arko_workflows/INDEX.md#verdict-protocol-and-records) |
| `DistillationConfig::new` and fields (`name`, `model`, `target_agents`, `target_tools`, `min_chars`, `max_output_tokens`, `instructions`, `reasoning`) | An LLM distiller that replaces large results of the named tools, for the named agents, with a summary. | `arko_workflows/src/distillation/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `DistillationConfig::target_agent`, `DistillationConfig::target_tool`, `DistillationConfig::min_chars`, `DistillationConfig::max_output_tokens`, `DistillationConfig::instructions`, `DistillationConfig::reasoning`, `DistillationConfig::default_max_output_tokens` | The distiller's setters, and the output limit sent when none is set. | `arko_workflows/src/distillation/mod.rs` | [DistillationConfig](../arko_workflows/INDEX.md#distillationconfig) |
| `DistillationRecord` (`timestamp`, `task_id`, `agent_name`, `tool_name`, `config_name`, `input_chars`, `output_chars`, `cost`, `rejection`) | One distillation request: task, agent, tool, sizes, billed cost, and why its summary wasn't used (`rejection`). | `arko_workflows/src/distillation/mod.rs` | [DistillationRecord](../arko_workflows/INDEX.md#distillationrecord) |

**Budget and costs**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::add_manual_cost`, `Workflow::cost_ledger` | Records an outside cost (finite, nonnegative) and fires the budget hooks; returns `BudgetExceeded` over the limit, with the cost recorded. `cost_ledger` clones the ledger. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `BudgetManager` (`new`, `from_ledger`, `add_agent_cost`, `add_manual_cost`, `set_agent_limit`, `agent_limit`, `agent_spent`, `is_agent_exceeded`, `total_spent`, `is_exceeded`, `check_warning`, `remaining`, `limit`, `snapshot_ledger`) | The run's spend, limits and ledger (`SharedContext::budget`). It records and reports; it neither cancels agents nor fires hooks. | `arko_workflows/src/budget/mod.rs` | [BudgetManager](../arko_workflows/INDEX.md#budgetmanager) |
| `CostLedger` (`entries`, `total`), `CostEntry` (`amount`, `source`, `timestamp`, `metadata`), `CostSource` (`Agent { agent_name, usage }`, `Manual { description }`), `AddCostResult` (`new_total`, `warning_triggered`, `exceeded`) | The cost history and one recorded cost; agent entries keep the full `UsageInfo`. | `arko_workflows/src/budget/mod.rs` | [Cost data types](../arko_workflows/INDEX.md#cost-data-types) |

**Durable steps**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::durable_step`, `Workflow::durable_step_with_resolver` | Runs a keyed side effect once and replays its recorded output; the resolver reconciles an earlier attempt whose outcome is unknown. | `arko_workflows/src/workflow/mod.rs` | [Durable steps](#durable-steps-arko_workflows) |
| `Workflow::in_doubt_steps`, `Workflow::resolve_in_doubt_completed`, `Workflow::resolve_in_doubt_failed`, `Workflow::authorize_in_doubt_retry` | Lists steps in doubt, and records an operator's resolution. | `arko_workflows/src/workflow/mod.rs` | [Workflow-facing durable operations](../arko_workflows/INDEX.md#workflow-facing-durable-operations) |
| `DurableStepSpec` (`key`, `instance`, `version`, `retry_policy`; `new`, `instance`, `version`, `retry_policy`, `display_key`) | A step's identity (key and instance), definition version and retry policy. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `DurableRetryPolicy` (`Never`, `WhenDefinitelyNotCompleted { max_attempts }`, `RetrySafe { max_attempts }`) | When an in-doubt step may run again; default `Never`. | `arko_workflows/src/durable/mod.rs` | [Durable steps](#durable-steps-arko_workflows) |
| `DurableStepStatus` (`Started`, `Completed`, `InDoubt`, `PermanentlyFailed`) | A journal record's state. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `DurableStepContext` (`workflow_run_id`, `step_id`, `operation_id`, `attempt`, `operation_id()`), `InDoubtContext` | What the step closure receives (use `operation_id` as an idempotency key), and what a resolver or operator sees. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `ResolveInDoubt::resolve`, `InDoubtResolution` (`Completed(O)`, `DefinitelyNotCompleted`, `StillInDoubt`, `PermanentlyFailed`) | Extension trait that checks an external system for an in-doubt step's outcome. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `durable::DurableStore` (`new`, `journal_path`, `workflow_run_id`, `execute`, `execute_with_resolver`, `list_in_doubt`, `operator_complete`, `operator_permanently_fail`, `operator_authorize_retry`) | The append-only journal at `<run dir>/durable/journal.jsonl`; usable directly. | `arko_workflows/src/durable/mod.rs` | [DurableStore](../arko_workflows/INDEX.md#durablestore) |

**Persistence and checkpoints**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `PersistenceConfig` (`checkpoint_dir`, `auto_checkpoint`, `retain_checkpoints`, `event_log`, `event_flush_interval`; `new`, `auto_checkpoint(b)`, `keep_last_checkpoints`, `event_log(e)`, `event_flush_interval(d)`, `user_data`) | The run directory and what is saved there. | `arko_workflows/src/persistence/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `CheckpointRetention` (`KeepAll`, `KeepLast(NonZeroUsize)`) | Which checkpoint files are kept; default `KeepAll`. | `arko_workflows/src/persistence/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `EventLog` (`Full`, `LifecycleOnly`), `DEFAULT_EVENT_FLUSH_INTERVAL` (1 s) | What `events.jsonl` records, and how often buffered events are written at least. | `arko_workflows/src/persistence/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `Workflow::checkpoint` | Saves a checkpoint now (nothing without persistence); a failed checkpoint write is returned as an error. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `PersistenceManager` (`resolve_run_id`, `new`, `config`, `durable_store`, `save_checkpoint`, `auto_checkpoint`, `append_event`, `flush_events`, `save_agent_task`, `load_checkpoint`) | Writes a run directory; usable directly, for example `load_checkpoint` to inspect a checkpoint before resuming. | `arko_workflows/src/persistence/mod.rs` | [PersistenceConfig and PersistenceManager](../arko_workflows/INDEX.md#persistenceconfig-and-persistencemanager) |
| `persistence::state::WorkflowState` (`workflow_run_id`, `workflow_name`, `workflow_metadata`, `checkpoint_id`, `timestamp`, `memory_snapshot`, `message_queues`, `budget_spent`, `cost_ledger`, `freshness_snapshot`, `agent_results`, `agent_runs`, `missing_task_states`, `verification_records`, `distillation_records`, `completed_agents`, `cancellation`, `agent_registry`), `WorkflowState::agent_results_in` | A checkpoint's content; `agent_results_in` returns its results with their tasks read from the run directory. | `arko_workflows/src/persistence/state.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| `persistence::state::AgentRunRecord` (`agent_run_id`, `agent_name`, `status`, `summary`, `error`, `task_file`), `MissingTaskState` (`agent_run_id`, `agent_name`, `status`, `last_saved`, `reason`), `SerializableAgentResult` (`agent_name`, `summary`) | A checkpoint's reference to a result's task version, a run whose latest task state couldn't be saved, and a name-and-summary record. | `arko_workflows/src/persistence/state.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| `messaging::registry::RegistrySnapshot` (`version`, `agents`, `seen`), `AgentRecord` (`last_start`, `finished`, `last_finish`) | The agent status registry as a checkpoint saves it. | `arko_workflows/src/messaging/registry.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| Event log (`events.jsonl`) | One JSON line per recorded agent event, with time, run ID, agent name and agent run ID. | `arko_workflows/src/persistence/mod.rs` | [PersistenceConfig and PersistenceManager](../arko_workflows/INDEX.md#persistenceconfig-and-persistencemanager) |

**Shared context**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::context`, `context::SharedContext` | The run's shared state. Public fields: `workflow_run_id`, `workflow_name`, `workflow_metadata`, `workspace`, `budget`, `durable_store`, `memory`, `messages`, `messaging`, `resource_tracker`, `hooks`, `agent_configs`, `global_tools`, `agent_tools`, `agent_results`, `global_interceptors`, `agent_interceptors`, `interception_failure_policy`, `completion_verifiers`, `verification_attempts`, `verification_records`, `distillation_configs`, `distillation_records`, `workflow_services`, `paused`, `resume_notify`, `user_input_handler`, `approval_policy`, `error_recovery_policy`, `ballots`, `voters`, `event_tx`. No public constructor. | `arko_workflows/src/context/mod.rs` | [Shared context](../arko_workflows/INDEX.md#shared-context) |
| `SharedContext::message_recipients`, `SharedContext::message_senders`, `SharedContext::recipient_statuses`, `SharedContext::recipient_status` | The agents one agent may message, the agents that may message it, and their statuses. | `arko_workflows/src/context/mod.rs` | [Shared context](../arko_workflows/INDEX.md#shared-context) |

**Errors**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `WorkflowError::BuilderMissing { field }`, `WorkflowError::BuilderInvalid(String)` | A required builder setting is missing; an invalid configuration or checkpoint, an invalid durable step spec, a feature used while off (freshness, voting), a result that can't be continued or resumed, or a disallowed services caller or target. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::AgentNotFound` | No agent is registered under the name. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::UserInputRequired { agent, task_id, questions }`, `WorkflowError::AgentPaused { agent, task_id, pause }`, `WorkflowError::VerifierFailed { verifier, agent, task_id, reason }` | Pauses, not failures: the agent waits for an answer (`continue_agent`), a decision (`resume_agent`), or a continuation after its verifier couldn't review its completion. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::AgentInterrupted { agent, reason }`, `WorkflowError::AgentCancelled { agent, reason }` | The driven task ended interrupted, or cancelled while the run was not stopped. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::Agent`, `WorkflowError::Llm`, `WorkflowError::Tool` | Errors from arko_agents, arko_ai and tools; a task that ended `Failed` returns `Agent(ArkoAgentsError::TaskExecution(reason))`. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::BudgetExceeded { spent, limit }`, `WorkflowError::AgentBudgetExceeded { agent, spent, limit }`, `WorkflowError::InvalidCost` | Spending strictly above the workflow or agent limit; a negative or non-finite manual cost. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::MemoryTierNotFound`, `WorkflowError::MemoryAccessDenied { agent, tier, action }`, `WorkflowError::MemoryLimitExceeded { tier, attempted, limit }` | Memory store errors. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::Cancelled`, `WorkflowError::AbortedByHook { hook_point, reason }` | The run was stopped by a cancel handle, or by a hook's `Abort`; every run call in progress or made later returns the first cause's error. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::DurableUnavailable`, `WorkflowError::DurableDefinitionMismatch { step, details }`, `WorkflowError::DurableInDoubt { step, operation_id }`, `WorkflowError::DurablePermanentlyFailed { step, reason }`, `WorkflowError::DurableAttemptsExhausted { step }` | Durable steps without persistence; a changed definition or input for a recorded step; an unresolved earlier attempt (or one running now); a recorded permanent failure; the retry limit reached. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::Persistence`, `WorkflowError::Io`, `WorkflowError::Json`, `WorkflowError::Channel`, `WorkflowError::Deliberation` | Persisted-state, file and JSON errors; `Channel` for a join or channel failure, a `UserInputHandler` error or the services call limit; `Deliberation` for ballot errors. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |

**Python bindings** (`python` feature)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `python::PyWorkflowBuilder`: `name`, `workspace`, `agent(name, config_path)`, `budget`, `persistence(checkpoint_dir, auto_checkpoint, retain_checkpoints, event_log)`, `rename_memory_tier_on_resume`, `drop_memory_tier_on_resume`, `approval_policy`, `error_recovery_policy`, `max_parallel_agents`, `fail_fast`, `resume_from`, `global_memory_tier`, `cancel_grace_period`, `wait_for_message_timeout`, `agent_wait_for_message_timeout`, `build` | The Python builder; setters return `None`, and `build` creates the client from the environment and loads the agent files. | `arko_workflows/src/python/mod.rs` | [Python bindings](#python-bindings-arko_workflows) |
| `python::PyWorkflow`: `run_agent`, `run_agents_parallel`, `continue_agent`, `resume_agent(previous, decision, note)`, `agent_result`, `finish`, `cancel`, `is_cancelled`, `add_manual_cost`, `cost_ledger_json`, `checkpoint`, `read_memory`, `write_memory` | The Python workflow; the run methods are awaitable. | `arko_workflows/src/python/mod.rs` | [PyWorkflow](../arko_workflows/INDEX.md#pyworkflow) |
| `python::PyAgentResult` (`agent_name`, `summary`, `total_cost`, `status`, `agent_run_id`, `error`) | A result's read-only properties; the full result is kept for continuing it. | `arko_workflows/src/python/mod.rs` | [PyAgentResult](../arko_workflows/INDEX.md#pyagentresult) |
| Exceptions `WorkflowCancelled`, `UserInputRequired`, `AgentPaused`, `VerifierFailed` | Raised by the run methods; the last three carry `(message, agent_run_id)`. Other run errors raise `RuntimeError`. | `arko_workflows/src/python/mod.rs` | [Python bindings](../arko_workflows/INDEX.md#python-bindings) |

**Example binaries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `basic_workflow` | A two-agent sequential example; takes no flags. | `arko_workflows/examples/basic_workflow.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `deepcoder --workspace`, `--spec` | Required: the project to modify and the specification file. | `arko_workflows/examples/deepcoder.rs` | [Run](../arko_workflows/examples/deepcoder/README.md#run) |
| `deepcoder --state-dir`, `--resume`, `--ephemeral` | Run directory, resuming a run, and running without persistence. | `arko_workflows/examples/deepcoder.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `deepcoder --budget`, `--max-minor-corrections`, `--max-efficiency-corrections`, `--max-final-corrections` | Budget and the verifiers' minor-correction caps. | `arko_workflows/examples/deepcoder.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `deepcoder --shell-timeout-seconds`, `--shell-output-bytes` | Default and maximum `run_shell` timeout, and the output returned to an agent. | `arko_workflows/examples/deepcoder.rs` | [Shell tool and isolation](../arko_workflows/examples/deepcoder/README.md#shell-tool-and-isolation) |
| `complete_workflow --data-dir DIR`, `--dry-run`, `--resume PATH`, `--owner TEXT` | Complete three-agent incident-report workflow; validates without credentials in dry-run mode and can resume a checkpoint. | `arko_workflows/examples/complete_workflow.rs` | [Complete workflow example](#complete-workflow-example) |

### Builder defaults (arko_workflows)

`build()` requires a name, an existing workspace directory (stored absolute, with `.` and `..` resolved and symbolic links kept), an `Arc<LLMClient>`, and at least one uniquely named agent. The builder reads no credentials. With persistence set, `build()` creates the run directory and its `workflow-run-id` file.

| Setting | Default | Notes |
|---|---|---|
| `approval_policy` | **`ApproveAll`: every call an agent asks approval for is approved** | Includes paths outside the workspace under `RequireApproval`. Logs one warning per agent. See [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows). |
| `error_recovery_policy` | **`Automatic`: recovery without an attempt limit** | See [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows). |
| `metadata` | Empty | Only `supported_platforms` is enforced: a nonempty list the host matches none of fails `build`. |
| `memory` | No tiers, no memory tools | See [Memory](#memory-arko_workflows). |
| `budget`, `budget_warning` | None: unlimited, no warning | Dollars, finite and nonnegative. Spending counts `UsageInfo::billed()`, so ChatGPT OAuth calls add zero. Once spending exceeds the limit, each driven agent is cancelled at its next event and its call returns `BudgetExceeded`; later run calls return it without starting an agent. |
| `agent_budget(agent, limit)` | None | The agent's own model spend across its runs, checked each time it reports usage: once over the limit, the agent is cancelled and its call returns `AgentBudgetExceeded` (a new run of an agent already over it is stopped at its first usage report). Distillation charges don't count toward it. |
| `hook(point, callback)` | None | Synchronous; run in registration order, and the first result other than `Continue` ends the point. `Abort` stops the run; `Skip` is honoured only at `PreToolExecution`. |
| `persistence` | None: nothing is saved, durable steps unavailable | See [Persistence](#persistence-arko_workflows). |
| `resume_from(checkpoint_file)` | None | Needs `persistence` whose run directory holds the checkpoint's run ID; see [Resume](#resume-arko_workflows). |
| `rename_memory_tier_on_resume`, `drop_memory_tier_on_resume` | None | See [Resume](#resume-arko_workflows). |
| `ephemeral()` | — | Clears the persistence and resume settings made so far. |
| `context_freshness` | Off | See [Context freshness](#context-freshness-arko_workflows). The default mapping covers arko's file tools and Claude Code's `Read`, `Write`, `Edit`, `MultiEdit` and `NotebookEdit`. |
| `global_tool`, `agent_tool` | None | Added to every agent's (or one agent's) available tools. A custom tool named like an arko_agents built-in replaces it, except `complete_task`. A global tool can't take the name `complete_task`, `ask_user`, a memory tool's, `run_workflow_agent` or `cast_vote`, nor a messaging tool's while any agent is offered it; an agent tool can't take a messaging tool's name that agent is offered (`build` fails). |
| `tool_result_interceptor`, `agent_tool_result_interceptor` | None | Global interceptors, then the agent's, then distillers. |
| `interception_failure_policy` | `UseRawResult` | Applied to each stage of the chain: a failing stage is retried as the policy allows and then skipped, keeping the earlier stages' output. |
| `completion_verifier`, `distiller` | None | See [Verification and distillation](#verification-and-distillation-arko_workflows). |
| `workflow_services` | None: no `run_workflow_agent` | See [Messaging, services and voting](#messaging-services-and-voting-arko_workflows). |
| `messaging` | `MessagingPolicy::all()`: every agent may message every other | `disable_messaging()` offers no messaging tools. |
| `wait_for_message_timeout`, `agent_wait_for_message_timeout` | `WaitForMessageTimeout::Auto` | Just inside the agent's prompt-cache lifetime, counted from its last model request: 4 min 30 s, or 54 min with Anthropic's 1-hour TTL. A fixed `Duration` must be above zero and at most 24 hours. |
| `user_input_handler` | None: an `ask_user` question suspends the agent and the call returns `UserInputRequired` | A handler's `Err` ends the call with `WorkflowError::Channel`. |
| `enable_voting(voters)` | Off | Offers `cast_vote` to the named agents only. |
| `max_parallel_agents(n)` | No cap | For `run_agents_parallel`; `0` fails `build`. |
| `fail_fast(bool)` | `false` | `true`: in `run_agents_parallel` the first failure stops the others. A pause is not a failure. |
| `cancel_grace_period` | 30 s (`DEFAULT_CANCEL_GRACE_PERIOD`) | How long a stopped run waits for each running agent before aborting it; nested runs get half per level. |

Every agent-name list is validated at build; tool names that the freshness mapping or a distiller names are only warned about. See [Builder signatures and defaults](../arko_workflows/INDEX.md#builder-signatures-and-defaults).

### Approvals and error recovery (arko_workflows)

Workflows run without human input by default. The managed paths (`run_agent`, `continue_agent`, `resume_agent`, `run_agents_parallel`, `AgentHandle::wait`) answer approvals and error-recovery pauses under these policies; `AgentHandle::into_parts` hands both to the caller.

| `ApprovalPolicy` | Behaviour |
|---|---|
| `ApproveAll` (default) | Approves every call the agent asks approval for. |
| `RespectAgentConfig` | Pauses the agent: the call returns `WorkflowError::AgentPaused` with `AgentPause::ToolApproval`; continue with `resume_agent`. A call without a decision is rejected. |
| `Handler(Arc<dyn ToolApprovalHandler>)` | Decides each call: approve, reject with a reason, or pause. |

| `ErrorRecoveryPolicy` | Behaviour |
|---|---|
| `Automatic` (default) | After the agent's own retries run out: outages wait 30 s, doubling to 15 min while pauses repeat (and at least the provider's requested delay), then resend unchanged; plan usage limits wait until the reset plus 5 s; tool errors continue at once. Each wait gets up to 30 s of random extra time. Non-retryable errors, refusals and unusable responses fail the task with a reason. Monitor pauses and failed verifier reviews return to the caller. Nothing is added to the conversation. |
| `Limited { max }` | As `Automatic`, for at most `max` recoveries per call that drives the agent; after that, a pause it would recover from fails the task. `0` recovers nothing. |
| `ReturnToCaller` | Every error-recovery pause returns `AgentPaused` with `AgentPause::ErrorRecovery` (`VerifierFailed` for a completion its verifier couldn't review). |

An agent that is interrupted instead of pausing (as the `RetryThenInformAgent` and `InformAgentAndContinue` error policies do when recovery ends) is cancelled, and the call returns `AgentInterrupted`. An agent running inside another agent's tool call (a verifier or a delegated run) can't pause: under `RespectAgentConfig` its `complete_task` is approved, and anything else that would pause it stops it (a completion its verifier couldn't review is left paused). `UserInputRequired`, `AgentPaused` and `VerifierFailed` are pauses, not failures: `OnUserInputRequired` fires for the first and `OnAgentPaused` for the others; `OnWorkflowFailed` fires for none. See [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery).

### Memory (arko_workflows)

| Setting | Default | Notes |
|---|---|---|
| `MemoryConfig::global_tier(name)` | Readers and writers `All`, `AppendOnly`, no size limit | `global_tier_with_limit`, `broadcast_tier(name, writers)` (writers `Only`), `broadcast_tier_with_limit`. Other readers, writers or `FullEdit` need `tier(MemoryTier { .. })`. |
| `MemoryTier::max_bytes` | `None`: unlimited | Limits the content (not the history); a write over it fails and changes nothing. |
| `MemoryTier::write_mode` | No default on the tier (required in JSON); `WriteMode::default()` and the `MemoryConfig` helpers use `AppendOnly` | `FullEdit` replaces the content; the history keeps each version. |
| `AccessScope` | — | `All`, `Only(names)`, `None`. |

An agent gets `read_memory` if it can read a tier, `write_memory` if it can write one, and `list_memories` if either; `list_memories` hides tiers it can't access. `read_memory` results are never intercepted or distilled. See [Memory](../arko_workflows/INDEX.md#memory).

### Messaging, services and voting (arko_workflows)

`WorkflowServicesConfig` is built with `Default` and its setters; a struct literal (`..Default::default()` included) doesn't compile outside the crate.

| Setting | Default | Notes |
|---|---|---|
| `MessagingPolicy` | Every agent may message every other | `can_message(agent, recipients)` limits one agent; `cannot_message(agent)`. `send_message` and `list_recipients` go to agents with a permitted recipient, `wait_for_message` to agents another may message. `send_message` reaches only running or waiting agents. Running agents are told when a recipient starts or finishes. |
| `MessagingPolicy::keep_messages_for_finished()` | Off: a message to a finished agent is refused | On: it is queued for the agent's next run (a continuation or a new run), and the finish notice says so. |
| `WorkflowServicesConfig::allowed_callers`, `allowed_targets` | Empty: **nobody** | Name them with `allow_caller` and `allow_target`; an empty caller list is only warned about. |
| `allow_all_callers()`, `allow_all_targets()` | Off | Every agent except verifier agents (and, as a target, the caller itself). |
| `max_active_calls` | 4 | Delegated calls running at once, workflow-wide; a call over the limit fails at once. A `NonZeroU32`: `max_active_calls(0)` fails the build. |
| `BallotConfig` | No default | `begin_ballot` requires at least 2 options with nonblank, unique IDs, `quorum` at least 1 and `consensus_threshold` from 0 to 1. `eligible_voters` empty = every enabled voter; a voter without a weight weighs 1.0. `finish_ballot` rejects a cast vote whose weight isn't finite and positive; `tie_break` is `RejectTie` or `Lexicographic` (smallest option ID). |

### Verification and distillation (arko_workflows)

`CompletionVerifierConfig::correctness(name, verifier_agent)` and `efficiency(..)` set a default `instructions` text; in JSON, `name`, `verifier_agent`, `kind` and `instructions` are required.

| `CompletionVerifierConfig` | `correctness()` | `efficiency()` | Omitted in JSON |
|---|---|---|---|
| `target_agents` | Empty: `build` fails | Same | Required |
| `max_minor_rejections` | 3 | 2 | Required |
| `max_major_rejections` | 0: the first major rejection fails the task | 0 (a major rejection counts as minor) | Required |
| `reuse_conversation` | `false` | `false` | `false` |
| `repair_context` | `None` (`repair_context(bytes)`, at least 1,024) | `None` | `None` |
| `include_prior_findings` | `true` | `true` | `true` |
| `max_format_retries` | 1 | 1 | 1 |
| `max_review_retries` | 1 | 1 | 1 |

A review that fails doesn't fail the target: the agent is asked to call `complete_task` again, up to `max_review_retries` times, then the call returns `WorkflowError::VerifierFailed` with the task paused. A correctness verifier fails the task at the rejection after its cap; an efficiency verifier never fails a task by rejecting it and stops reviewing once its minor cap is reached. `repair_context` needs `reuse_conversation` and context freshness.

In JSON, every `DistillationConfig` field except `max_output_tokens` and `reasoning` is required.

| `DistillationConfig::new(name, model)` | Default | Notes |
|---|---|---|
| `target_agents`, `target_tools` | Empty: `build` fails | Both required; `read_file` is not exempt. |
| `min_chars` | 8,000 | Results shorter than this are left alone. |
| `max_output_tokens` | `None`: 16,000, or the model's output limit when smaller (nothing over ChatGPT OAuth) | A set value is always sent (arko_ai drops it over ChatGPT OAuth, with a warning). `Some(0)` is rejected. |
| `reasoning` | `None`: the model's default | Must be for the model's provider. |

Distillation requests turn prompt caching off (except over ChatGPT OAuth), count toward the workflow budget but not the agent budget, and keep the raw output when the summary is refused or cut off. See [Tool-result distillation](../arko_workflows/INDEX.md#tool-result-distillation).

### Context freshness (arko_workflows)

`ContextFreshnessConfig::default()`: `block_on_conflict: true`, and `read_file`, `list_files` (a directory-tree read), `write_file` and `apply_diff` mapped to their path parameters; `tool_effect(name, effect)` maps other tools. When loaded from JSON, an omitted field takes its default; a given `tool_effects` replaces the default map whole. A write is blocked when the writer read that file, or listed a directory containing it, and another agent run (another run of the same agent included) has since written it (for a listing, any file under the directory); the call is rejected with a message to re-read. With `block_on_conflict: false` the write is allowed. In both modes, active readers whose view a write made stale get a notice, and `OnFreshnessConflict` fires for each stale reader. Memory tiers are tracked the same way through `read_memory` (complete reads only) and `write_memory`, and custom resources through `Workflow::register_resource_read` and `mark_resource_dirty`. See [Context freshness](../arko_workflows/INDEX.md#context-freshness).

### Durable steps (arko_workflows)

`DurableStepSpec::new(key)`: version `"v1"`, no instance, retry policy `Never` (in JSON `version` and `retry_policy` are required). A step whose closure returns an error is recorded in doubt, and the call returns `DurableInDoubt` (the closure's error becomes the recorded reason). Under `Never` it stays in doubt until resolved. `WhenDefinitelyNotCompleted { max_attempts }` runs it again only after a resolver (`durable_step_with_resolver`) reports it definitely not completed; `RetrySafe { max_attempts }` runs it again at the next call for the step, without a resolver; `authorize_in_doubt_retry` allows one more run (under `Never` always, otherwise within `max_attempts`). `max_attempts` 0 counts as 1. Durable steps need persistence and replay from the run directory's `durable/journal.jsonl`. See [Durable steps](../arko_workflows/INDEX.md#durable-steps).

### Persistence (arko_workflows)

`PersistenceConfig` is built with `new(dir)` and its setters; a struct literal doesn't compile outside the crate.

| `PersistenceConfig::new(dir)` | Default | Notes |
|---|---|---|
| `auto_checkpoint` | `true` | A checkpoint after every finalized agent run and when a run is stopped. Task versions are saved either way. |
| `retain_checkpoints` | `CheckpointRetention::KeepAll`: every checkpoint and task version is kept | `keep_last_checkpoints(n)` (`KeepLast`) keeps the newest `n` and the task versions they refer to; `n` of 0 fails the build. In Python, `retain_checkpoints=n` (0 raises `ValueError`). |
| `event_log` | `EventLog::Full`: every event, including each streamed fragment | `LifecycleOnly` leaves the fragments out. The event log is never truncated or rotated. |
| `event_flush_interval` | 1 s | State changes, run ends and errors are written at once. |

`PersistenceConfig::user_data(name)` picks a new run directory, `<user data>/arko/workflows/<name>/<UUID>`, on each call. A configuration pointing at an existing run directory reuses its `workflow-run-id` and durable journal, also without `resume_from`. Checkpoints and task versions are pretty JSON written through a temporary file, synced and renamed; `latest-checkpoint` names the newest checkpoint; the event log and the durable journal are appended JSON lines. All are written with default permissions; task versions hold full conversations, and the event log the agents' output and tool results. A save that fails never fails an agent's run: it is logged and retried by the next checkpoint. `checkpoint()` and `finish()` save regardless of `auto_checkpoint`, and return a failed checkpoint write as an error. See [Persistence](../arko_workflows/INDEX.md#persistence-and-checkpoint-state).

### Resume (arko_workflows)

`WorkflowBuilder::resume_from(path)` takes a checkpoint file, not a run directory.

- The new builder supplies agents, client, tools, hooks, policies, budget limits and metadata. Memory access, limits and write mode come from the builder; the checkpoint supplies each tier's content and history by name. For every restored task that can still be continued or resumed, a changed user-configured agent configuration is rejected; use the original configuration or register the changed one under another name. Workflow-injected tools and refreshed model registry snapshots are ignored for this comparison.
- A restored result that `continue_agent` or `resume_agent` runs again keeps the `AgentConfig` saved in its task (model, instructions and offered tools); new runs use the builder's.
- A tier configured but missing from the checkpoint starts as configured. A checkpoint tier with no configured tier fails `build`, unless renamed with `rename_memory_tier_on_resume` or dropped with `drop_memory_tier_on_resume`.
- Restored: memory, pending messages, the cost ledger, finalized agent results (with their tasks), verification and distillation records, agent statuses, and freshness state when context freshness is configured again. Not restored: a recorded stop (the run starts again), ballots, verifier repair counts, the pause flag, and running agents, which count as finished.
- The checkpoint's workflow name and run ID must match. `continue_agent` continues a `Completed` or `WaitingForInput` result; `resume_agent` answers a `Paused` one.

### Python bindings (arko_workflows)

Behind the `python` feature. The defaults match Rust (auto-checkpoint on, approve all, automatic recovery, no parallel cap, fail-fast off, 30 s grace, automatic `wait_for_message` limit). Differences: the client always comes from `LLMClient::from_env()`, agents load from configuration files, memory offers only global tiers, and messaging is always all-to-all. Not exposed: hooks, custom tools, interceptors and `interception_failure_policy`, verifiers, distillers, services, freshness, voting, durable steps, user-input handlers, the `Handler` approval policy, `ResolveRefusal`, `budget_warning`, `agent_budget`, `metadata`, the messaging policy, `event_flush_interval`, `PersistenceConfig::user_data`, `spawn_agent`, workflow pause and resume, and event subscription. See [Python bindings](../arko_workflows/INDEX.md#python-bindings).

### Persisted data (arko_workflows)

- Checkpoints have no format version. Their optional fields default and unknown fields are ignored, so older checkpoints load, including those that hold their tasks inline; the next checkpoint writes those tasks as task versions.
- A checkpoint without a run ID skips the run-ID check. One without a cost ledger resumes with nothing spent (`budget_spent` is not read).
- Checkpoints may refer to task files that earlier versions wrote beside them (`agents/<task id>.json`); those files are read when referenced and are not deleted by retention.
- `Failed` tasks with the reason `Task cancelled by user.` load as `Cancelled`.
- `resume_from` refuses a missing or unreadable task file, another workflow's checkpoint, a run-ID mismatch, and a checkpoint-only memory tier.
- While event-log writes fail, up to 64 MiB of events stay buffered for the next write; older ones are dropped with a warning.

### Example binaries (arko_workflows)

**`basic_workflow`**: no flags; `LLMClient::from_env()`, agent configurations `configs/architect.json` and `configs/coder.json` (not shipped; supply them) and an existing `./my-project` directory, all relative to the current directory; budget $5 (warning at $4); the default `ApproveAll`.

**`deepcoder`**:

| Flag | Default | Notes |
|---|---|---|
| `--workspace`, `--spec` | Required | |
| `--state-dir` | `<workspace parent>/.arko-deepcoder-runs/<name>-<time>-<id>` | Must be outside the workspace; one already holding a run is refused without `--resume`. Persistence is on unless `--ephemeral`. |
| `--resume PATH` | New run | A checkpoint file, or a run directory; its directory is the state directory. The workspace and specification must match the run's. Conflicts with `--ephemeral`. |
| `--budget` | 50.0 | Warning at 80%. |
| `--max-minor-corrections`, `--max-efficiency-corrections`, `--max-final-corrections` | 20, 3, 4 | The library's verifier caps are 3, 2 and 3. |
| `--shell-timeout-seconds`, `--shell-output-bytes` | 1800, 65,536 | |

`deepcoder` configures only OpenAI (the `OPENAI_OAUTH_*` variables, then the Codex `auth.json`, then `OPENAI_API_KEY`), auto-approves every tool, and asks `ask_user` questions at the terminal. Its agents use `LLMErrorPolicy::PauseAndAwaitApproval`, so a failed model request goes straight to the workflow's automatic recovery. It declares Linux and macOS only, so `build` fails on other hosts. See the [DeepCoder README](../arko_workflows/examples/deepcoder/README.md).

**`complete_workflow`**: `--data-dir` is required; `--dry-run` builds and checkpoints without
credentials or model requests; `--resume` restores a checkpoint; and `--owner` supplies the
answer to the writer's user question. See [Complete workflow example](#complete-workflow-example).


## arko_index

Incremental indexing of source code and documentation: a chunk model chooses validated chunk boundaries, Voyage embeds the chunks, and a local SQLite database serves lexical, semantic and hybrid search.

Guides: [arko_index README](../arko_index/README.md) · [arko_index reference](arko_index-reference.md)

### Map (arko_index)

The crate root re-exports the items of `config`, `index`, `search` and `judge`, `JobUsage`, the watcher types, `Error` and `Result`. The modules `backend`, `budget`, `chunker`, `config`, `error`, `packing` and `source` are public too; `budget`, `chunker`, `packing` and `source` hold the building blocks the index calls itself.

**Opening and configuration**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ArkoIndex` | Cloneable handle to one index database; the calls below go through it. | `arko_index/src/index.rs` | [README](../arko_index/README.md#build-and-search) |
| `ArkoIndex::open(path, backend, config)` | Validates `config`, requires the backend's profile to embed as `config.embedding` does, and opens or creates the SQLite database, holding `<path>.lock` so that one `ArkoIndex` serves it at a time. | `arko_index/src/index.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `Config`, `Config::validate` | Index settings: `batch_count`, `concurrency`, `embedding_concurrency`, `embedding`, `vector_cache_bytes`, `max_transient_attempts`, `max_capacity_wait_ms`, `query_priority`, `query_max_transient_attempts`, `embedding_batch_tokens`, `embedding_batch_items`, `keep_generations`, `synchronous`, `stream_idle_timeout_ms`. | `arko_index/src/config.rs` | [`Config` (arko_index)](#config-arko_index) |
| `Synchronous::{Full, Normal}` | SQLite `synchronous` mode of the database (`Config::synchronous`). | `arko_index/src/config.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `EmbeddingProfile`, `EmbeddingProfile::same_embeddings`, `EmbeddingProfile::revision_label` | What embeds an index: Voyage `model`, `dimensions` and `tokenizer_revision`; `same_embeddings` compares two profiles, and `revision_label` names the revision (`unpinned` when unset). | `arko_index/src/config.rs` | [Embeddings](arko_index-reference.md#embeddings) |
| `DEFAULT_MAX_CAPACITY_WAIT_MS` | 900,000 (15 minutes), the default `Config::max_capacity_wait_ms`. | `arko_index/src/config.rs` | [`Config` (arko_index)](#config-arko_index) |
| `FORMER_DEFAULT_TOKENIZER_REVISION` | The `voyageai/voyage-4` commit earlier versions pinned by default; an unset `tokenizer_revision` compares, and keys stored vectors, as this one. | `arko_index/src/config.rs` | [Embeddings](arko_index-reference.md#embeddings) |
| `Error`, `Result<T>` | The crate's error: `Invalid`, `Blocked`, `Paused`, `UsageLimit`, `Cancelled`, `Stale`, `Provider { message, transient, retry_after, source, usage }`, `Io`, `Sql`, `Json`, `Worker`. `search` returns `Blocked` before the first generation is published and `Stale` for a cursor or generation no longer available; `Paused`, `UsageLimit` and `Cancelled` are job stop reasons, seen as the job's phase and problem. An `arko_ai::Error` converts to `Provider`. | `arko_index/src/error.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `Error::with_usage`, `Error::usage`, `Error::retry_class` | Attach or read the usage of a provider response the backend rejected; the recovery class (`arko_ai::RetryClass`) of a provider failure, from its source error or else from `transient`. | `arko_index/src/error.rs` | [Usage and cost](arko_index-reference.md#usage-and-cost) |

**Builds and jobs**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `BuildSpec`, `BuildSpec::new`, `with_chunking`, `with_max_file_bytes`, `with_skip_directories`, `validate` | What an index covers: `name`, `root`, `include`, `exclude`, `skip_directories`, `max_file_bytes`, `chunking`; saved with the index at `start_build`. | `arko_index/src/config.rs` | [`BuildSpec` and chunking (arko_index)](#buildspec-and-chunking-arko_index) |
| `ChunkingOptions`, `ChunkingOptions::validate` | How chunk-model requests are sized: `max_request_tokens`, `max_request_bytes`, `oversize`, `max_repair_rounds`, `repair_limit`. | `arko_index/src/config.rs` | [Request budget and large files](arko_index-reference.md#request-budget-and-large-files) |
| `OversizePolicy::{Split, Skip, Block}` | What happens to a file over the chunk request budget. | `arko_index/src/config.rs` | [Request budget and large files](arko_index-reference.md#request-budget-and-large-files) |
| `RepairLimitPolicy::{Merge, Block}` | What happens to a window whose cuts are still invalid after `max_repair_rounds` repair requests. | `arko_index/src/config.rs` | [Repairs](arko_index-reference.md#repairs) |
| `DEFAULT_SKIPPED_DIRECTORIES`, `DEFAULT_MAX_FILE_BYTES` | The default `BuildSpec::skip_directories` list and `BuildSpec::max_file_bytes` (16 MiB). | `arko_index/src/config.rs` | [Which files are indexed](arko_index-reference.md#which-files-are-indexed) |
| `ArkoIndex::start_build(spec)` | Creates a new index with its own ID, even when the name is already used, and starts its first job in the background; returns a `JobHandle`. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `ArkoIndex::start_update(index)` | Starts a job that rescans the source with the saved spec; fails while another job of the index is running, paused or blocked. | `arko_index/src/index.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `ArkoIndex::pause(job)` | Drains a job running in this `ArkoIndex` to a checkpoint: queued requests and retry waits stop, requests in flight finish. Fails for a job being cancelled. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `ArkoIndex::cancel(job)` | Cancels a job, dropping its requests in flight; a `ready` or `pruned` job can't be cancelled. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `ArkoIndex::resume(job)` | Continues a `paused`, `blocked` or `failed` job from its checkpoints with the current runtime settings. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `ArkoIndex::status(job)`, `JobStatus` | A job's `job` and `index` IDs, `phase`, `problem`, file counts (`files`, `validated`, `bypassed`, `unchecked`, `skipped`), requests in flight across all jobs (`active_chunking`, `active_embeddings`) and `usage`. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| Job phases (`JobStatus::phase`) | `snapshotting`, `chunking`, `embedding`, `ready`; the stops `paused`, `blocked`, `cancelled` and `failed`; `pruned` for a generation removed by `keep_generations`. | `arko_index/src/index.rs` | [Jobs (arko_index)](#jobs-arko_index) |
| `ArkoIndex::events(job, after, limit)`, `Event` | Up to `limit` of a job's events with a sequence number above `after`: `sequence`, `kind`, JSON `detail`. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| Event kinds (`Event::kind`) | `phase`, `snapshot`, `oversize`, `request_attempt`, `request_finished`, `deterministic_repair`, `pause_requested`, `settings_changed`, `embedding_plan`, `embedded`, `embedding_rejected`, `embedding_failed`, `pruned`. A `request_attempt` event's `detail.event.phase` is `started`, `failed`, `retry_wait`, `retries_exhausted`, `usage_limit` or `completed`. | `arko_index/src/index.rs`, `arko_index/src/scheduler.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `ArkoIndex::request_artifact(job, id)` | A saved chunk-model request with its response or error and its attempt events. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `ArkoIndex::changes(job)`, `FileChange` | A job's file changes against the generation that was active when it scanned: `Added`, `Modified`, `Deleted`, `Moved { from, to }`. | `arko_index/src/index.rs`, `arko_index/src/watcher.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `coalesce(old, new)` | The `FileChange`s between two path-to-content-hash maps; a move needs exactly one deleted and one added file with that content. | `arko_index/src/watcher.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `ArkoIndex::list()`, `IndexInfo` | The database's indexes, ordered by name: `id`, `name` and active `generation`. | `arko_index/src/index.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `ArkoIndex::delete(index)` | Removes an index and the data no remaining index uses; fails while one of its jobs is snapshotting, chunking or embedding. | `arko_index/src/index.rs` | [Storage and updates](arko_index-reference.md#storage-and-updates) |
| `JobHandle` | The `index` and `job` IDs that `start_build` and `start_update` return. | `arko_index/src/index.rs` | [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries) |
| `JobUsage`, `JobUsage::total`, `JobUsage::billed` | Token usage and cost of a job's provider responses (`JobStatus::usage`): `chunk_calls`, `chunking`, `embedding_calls`, `embedding`, `unreported_calls`, `rejected_calls`. | `arko_index/src/usage.rs` | [Usage and cost](arko_index-reference.md#usage-and-cost) |

**Backend (module `backend`, not re-exported at the root)**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `backend::Backend` | Extension trait the index calls for providers. Required: `chunk`, `embed`, `tokens`, `limits`, `profile`, `output_tokenizer`; with defaults: `input_tokens`, `chunk_model_info`, `chunk_input`. `ArkoBackend` implements it; tests and offline hosts can supply their own. | `arko_index/src/backend.rs` | [Backend and chunk model (arko_index)](#backend-and-chunk-model-arko_index) |
| `backend::ArkoBackend::new(llm, embeddings, profile)` | The shipped backend: Voyage embeddings and token counts for `profile` (loading the `voyageai/<model>` tokenizer), and chunk boundaries from GPT-5.6 Luna with `default_chunk_config()`. | `arko_index/src/backend.rs` | [Chunk model](arko_index-reference.md#chunk-model) |
| `backend::ArkoBackend::with_chunk_model(model, config)` | Chunk boundaries from another `arko_ai` model and generation settings; the backend keeps ownership of the tool configuration. | `arko_index/src/backend.rs` | [Chunk model](arko_index-reference.md#chunk-model) |
| `backend::default_chunk_config()` | The default chunk-model settings: `high` reasoning effort with summaries, `low` verbosity. | `arko_index/src/backend.rs` | [Chunk model](arko_index-reference.md#chunk-model) |
| `backend::ModelRequest` | One chunk-model request: `system`, `prompt`, `tool`, `schema`, `cache_key` (the job's ID) and `stream_idle_timeout_ms` (from `Config`). | `arko_index/src/backend.rs` | [Chunk model](arko_index-reference.md#chunk-model) |
| `backend::ModelReply`, `backend::VectorReply` | A backend's answers: the tool call's arguments (`value`), the provider's call record (`usage`) and `usage_info`; embedding `vectors`, `billed_tokens` and `usage_info`. | `arko_index/src/backend.rs` | [Usage and cost](arko_index-reference.md#usage-and-cost) |
| `backend::ChunkInput`, `backend::ChunkTokenizer::O200kBase`, `backend::chunk_tokenizer(model)` | What sizes chunk requests: the chunk model's `tokenizer` (`o200k_base` for models of the OpenAI provider, none for others) and `max_input_tokens`. | `arko_index/src/backend.rs` | [Request budget and large files](arko_index-reference.md#request-budget-and-large-files) |
| `backend::Limits`, `backend::embedding_limits(details)`, `backend::UNPUBLISHED_BATCH_TOKENS` | Embedding request limits (`item_tokens`, `batch_tokens`, `batch_items`) of a Voyage model, with 120,000 tokens per request where Voyage publishes no limit (`voyage-code-4`). | `arko_index/src/backend.rs` | [Embeddings](arko_index-reference.md#embeddings) |
| `backend::ChunkModelInfo`, `backend::ChunkModelInfo::new(model, config)` | The chunk `model` and `reasoning` recorded in each job's `snapshot` event. | `arko_index/src/backend.rs` | [Chunk model](arko_index-reference.md#chunk-model) |

**Search**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ArkoIndex::search(index, query, options)` | Searches the active generation, or a cursor's generation, and returns a `SearchPage`. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `SearchMode::{Lexical, Semantic, Hybrid}` | FTS5 BM25 over path, text and identifier parts; exact cosine similarity; or both fused by reciprocal rank. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `SearchOptions` | `mode`, `top_k`, `min_chunk_bytes`, `path_prefix`, `literal`, `cursor`, `rebuild_file_limit`. | `arko_index/src/search.rs` | [Search and judge (arko_index)](#search-and-judge-arko_index) |
| `SearchPage` | `generation`, `mode`, `hits`, `next`, `omitted_hits`, `candidate_count`, `fresh_at_publication`, `rebuild`, `search_tokens`, `rebuild_tokens`, `tokenizer`, and `judge` for judged searches. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `SearchHit` | One result: `id`, `path`, `snapshot`, `start_byte`, `end_byte`, `first_line`, `last_line`, `structure`, `score`, `score_type`, the complete `text`, `other_locations` and `may_be_stale`. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `SearchCursor` | The `generation`, `query_hash` and `offset` of the next page (`SearchPage::next`). | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `RebuildNotice` | `SearchPage::rebuild` while an update is unfinished: its `job`, `phase`, `affected_files`, `omitted_files`, `determining_changes` and `message`. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `ArkoIndex::read_chunks(index, generation, ids, neighbors)`, `ChunkRead` | The named chunks of a complete generation, plus up to `neighbors` adjacent chunks on each side in the same file, ordered by path and position. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |

**Judge**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ArkoIndex::search_with_judge(index, query, options, judge, judging)` | Opt-in: searches for `candidate_limit` candidates, lets the judge filter and order them, then applies `top_k`. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |
| `SearchJudge` | Extension trait for judges: `select` is required; `name` (default `custom`) and `select_with_usage` have defaults. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |
| `LlmSearchJudge::new(client, model)`, `with_generation_config`, `with_instructions`, `with_stream_idle_timeout` | The shipped judge: one `select_results` tool call to an `arko_ai` model; the judge owns the tool configuration. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |
| `JudgeOptions` | `candidate_limit`, `max_attempts`, `max_transient_attempts`. | `arko_index/src/judge.rs` | [Search and judge (arko_index)](#search-and-judge-arko_index) |
| `JudgeSelection`, `JudgeExclusion` | A judge's answer: `ordered_ids`, and the `excluded` IDs, each with a `reason`. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |
| `JudgeReport` | `SearchPage::judge`: `name`, `attempts`, `usage`, `examined_candidates`, `unexamined_candidates`, `selected_count`, `selected_beyond_limit`, `excluded`. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |
| `JudgeAttempt` | One judge call's `result` and the `usage` of its response, from `select_with_usage`. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |
| `MAX_JUDGE_RETRY_WAIT` | 60 s: a judged search fails instead of waiting longer before a retry. | `arko_index/src/judge.rs` | [LLM judge](arko_index-reference.md#llm-judge) |

**Watcher**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ArkoIndex::watch(index, options)` | Watches the index root and starts, resumes or holds updates; returns a `WatchHandle`. | `arko_index/src/watcher.rs` | [Watching for changes](arko_index-reference.md#watching-for-changes) |
| `WatchOptions`, `WatchOptions::new(debounce)`, `with_reconcile`, `with_resume_interrupted`, `with_retry_blocked_after` | `debounce`, `reconcile`, `resume_interrupted`, `retry_blocked_after`. | `arko_index/src/watcher.rs` | [Watcher (arko_index)](#watcher-arko_index) |
| `WatchHandle::subscribe()`, `WatchHandle::stop()` | Receive the watcher's events from then on (a receiver more than 64 events behind misses the oldest); stop the watcher. | `arko_index/src/watcher.rs` | [Watching for changes](arko_index-reference.md#watching-for-changes) |
| `WatchEvent` | `Scanned { files, changes }`, `UpdateStarted { job }`, `Resumed { job, reason }`, `Held { job, phase, problem }`, `Failed { error }`. | `arko_index/src/watcher.rs` | [Watching for changes](arko_index-reference.md#watching-for-changes) |

**Building blocks (`source`, `chunker`, `packing`, `budget`)**

The index calls these itself; a host doesn't need them to build, update or search.

| Item | What it is | Code | Docs |
|---|---|---|---|
| `source::scan(spec)`, `source::Manifest`, `source::Exclusion` | The scan a job runs: the files a spec selects (`files`), each exclusion with its reason (`excluded`), the manifest `hash` and the `directories` entered. | `arko_index/src/source.rs` | [Which files are indexed](arko_index-reference.md#which-files-are-indexed) |
| `source::TOO_LARGE_TO_READ`, `source::TOO_LARGE_FOR_A_REQUEST` | The exclusion reasons "too large (over max_file_bytes)" and "too large (over the chunk request budget)". | `arko_index/src/source.rs` | [Which files are indexed](arko_index-reference.md#which-files-are-indexed) |
| `source::Snapshot`, `source::hash` | A file normalized to LF line endings (`path`, `hash`, `original_hash`, `text`, `line_starts`) with row helpers (`new`, `rows`, `line_end`, `row_at`, `line`, `numbered`, `tail_matches`); `hash` is a SHA-256 hex digest. | `arko_index/src/source.rs` | [Chunking and validation](arko_index-reference.md#chunking-and-validation) |
| `source::Structure`, `source::structure(snapshot)` | A file's guarded byte ranges (`protected`) and `status` (`rust-guarded`, `markdown-guarded` or `unchecked`); `legal(byte)` tells whether a cut may end there. | `arko_index/src/source.rs` | [Chunking and validation](arko_index-reference.md#chunking-and-validation) |
| `chunker::POLICY_VERSION`, `GROUPING_GUIDANCE`, `PROMPT`, `REPAIR_PROMPT`, `initial_prompt()`, `policy_hash()` | The chunking policy: its version, its prompts, and the hash each `snapshot` event records as `chunking_policy`. | `arko_index/src/chunker.rs` | [Repairs](arko_index-reference.md#repairs) |
| `chunker::{Cut, Cuts, Window, Region, Repair}`, `windows`, `request`, `failures`, `spans`, `remove_impossible_cuts`, `merge_invalid_cuts`, `repair`, `apply_repair` | The cut format (`after_line` and an exact `tail`), the `submit_cuts` and `repair_cuts` requests, and the validation and repair steps the index runs on replies. | `arko_index/src/chunker.rs` | [Chunking and validation](arko_index-reference.md#chunking-and-validation) |
| `packing::{Payload, payload, pack, normalize}` | Embedding payloads keyed by profile and text, first-fit-decreasing packing into requests within `Limits`, and normalization of vectors to unit length. | `arko_index/src/packing.rs` | [Embeddings](arko_index-reference.md#embeddings) |
| `budget::FALLBACK_MAX_INPUT_TOKENS`, `DEFAULT_BUDGET_PERCENT`, `BYTES_PER_TOKEN` | 128,000 tokens, assumed when the chunk model's input limit is unknown; the default budget's 10% share of that limit; 4 bytes per token for budgets in bytes. | `arko_index/src/budget.rs` | [Request budget and large files](arko_index-reference.md#request-budget-and-large-files) |

**Example binaries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_index_cli` | Example CLI: `build ROOT`, `update INDEX_ID`, `resume JOB_ID`, `status JOB_ID`, `list`, `search INDEX_ID QUERY`, `read INDEX_ID GENERATION CHUNK_ID...`, `watch INDEX_ID`. | `arko_index/examples/arko_index_cli.rs`, `arko_index/examples/support/search_judge.rs` | [Example CLI (arko_index)](#example-cli-arko_index) |
| `retrieval_eval DATABASE INDEX_ID QUERIES_JSON OUTPUT_DIRECTORY` | Research example: runs a semantic and a hybrid search (top 10, 40-byte minimum) for each query in the JSON file and saves every page; needs `VOYAGE_API_KEY`. | `arko_index/examples/retrieval_eval.rs` | [Example CLI (arko_index)](#example-cli-arko_index) |
| `retrieval_judge OAUTH_FILE EVALUATION_DIRECTORY` | Research example: grades each query's candidates in `judging-pool.json` with GPT-5.6 Luna at `max` effort over ChatGPT OAuth. | `arko_index/examples/retrieval_judge.rs` | [Example CLI (arko_index)](#example-cli-arko_index) |
| `retrieval_select OAUTH_FILE EVALUATION_DIRECTORY [compact\|conservative] [max\|high]` | Research example: GPT-5.6 Luna selection over saved search results; `conservative` and `max` by default. | `arko_index/examples/retrieval_select.rs` | [Example CLI (arko_index)](#example-cli-arko_index) |

### `Config` (arko_index)

Given to `ArkoIndex::open`; fields omitted from JSON take their defaults. `open` rejects a zero `batch_count`, `concurrency`, `embedding_concurrency` or `max_transient_attempts`; `Some(0)` for `query_max_transient_attempts`, `keep_generations`, `stream_idle_timeout_ms`, `embedding_batch_tokens` or `embedding_batch_items`; and an empty `tokenizer_revision`. `vector_cache_bytes: 0` caches nothing, and `max_capacity_wait_ms: Some(0)` fails a request at its first capacity signal. A resumed job takes every field except `embedding` from the current configuration.

| Field | Default and omitted | Notes |
|---|---|---|
| `batch_count` | 6 | Most windows per initial chunk request, within the request budget; a file within the budget is one window. |
| `concurrency` | 2 | Chunk-model requests in flight, shared by all jobs of the `ArkoIndex`. |
| `embedding_concurrency` | 2 | Embedding requests in flight, search query embeddings included. |
| `embedding` | `voyage-4`, 1,024 dimensions, `tokenizer_revision: None` | Must embed as the backend's profile does. An unset revision loads the tokenizer repository's current version (`main`) and compares, and keys vectors, as `FORMER_DEFAULT_TOKENIZER_REVISION`, so indexes built with that former default keep working; `Some("main")` is a different profile. See [Embeddings](arko_index-reference.md#embeddings). |
| `vector_cache_bytes` | 64 MiB | Memory for the vectors of the most recent semantic or hybrid search's generation and filters; 0 caches nothing. Results are the same either way. |
| `max_transient_attempts` | 10 | A request fails at this many transient failures; capacity signals don't count. |
| `max_capacity_wait_ms` | `Some(900_000)` (15 min); JSON `null` gives `None` | Total time one request may wait out capacity signals; `None` waits as long as the provider reports overload. Search query embeddings wait too. |
| `query_priority` | `false` | Opt-in: search query embeddings are dispatched before queued build embeddings. |
| `query_max_transient_attempts` | `None`: `max_transient_attempts` | Opt-in retry budget for search query embeddings. |
| `embedding_batch_tokens`, `embedding_batch_items` | `None`: the model's limits | Opt-in: fewer tokens or inputs per embedding request. They can't raise the model's limits, and `embedding_batch_tokens` can't be below the per-input limit. |
| `keep_generations` | `None`: every generation is kept until the index is deleted | Opt-in: `Some(n)` prunes the index's complete generations beyond the newest `n` each time a job publishes. |
| `synchronous` | `Full` | `Normal` syncs at WAL checkpoints; power loss can lose the last commits, never corrupting the database. |
| `stream_idle_timeout_ms` | `None` | Opt-in: a chunk-model response is cut off after this many milliseconds without data, a transient failure that is retried. No total deadline; make it generous. See [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries). |

### `BuildSpec` and chunking (arko_index)

`BuildSpec::new(name, root)` sets the defaults below; in JSON, `name` and `root` are required and the other fields take these defaults. `start_build` rejects `max_file_bytes: 0` and zero chunk limits or repair rounds, makes `root` canonical, and saves the spec with the index. No call changes it later: updates, the watcher and search's staleness check use the saved spec. The library has no default name.

| Field | Default and omitted | Notes |
|---|---|---|
| `include`, `exclude` | Empty | Case-sensitive globs on the root-relative path with `/` separators; `*` also matches `/`. With `include`, exactly the matching files are considered, whatever their extension, instead of the built-in list of file names and extensions; `exclude` and the other filters still apply. |
| `skip_directories` | `DEFAULT_SKIPPED_DIRECTORIES`: `.git`, `target`, `node_modules`, `.runtime`, `.cache`, `__pycache__`, `.venv`, `dist` | Exact names, never entered, at any depth; an empty list skips none, `.git` included. |
| `max_file_bytes` | 16 MiB (`DEFAULT_MAX_FILE_BYTES`) | Larger files are excluded unread, with the reason "too large (over max_file_bytes)". |
| `chunking.max_request_tokens` | `None`: 10% of the chunk model's input limit on the client's auth path | For chunk models of the OpenAI provider, counted in `o200k_base` tokens: 92,200 for GPT-5.6 and GPT-6 with an API key, 27,200 over ChatGPT OAuth. The budget counts only the rendered file content (header lines and numbered rows); the instructions and tool schema come on top. For models measured in bytes, a set value applies as 4 bytes per token unless `max_request_bytes` is set. |
| `chunking.max_request_bytes` | `None`: no byte limit for OpenAI chunk models; for others, 4 bytes per token of the default share | Models of other providers, OpenRouter included, are measured in bytes. An unknown input limit is taken as 128,000 tokens, a 51,200-byte default, and the job logs a warning. For OpenAI models both limits apply when both are set; for others this one replaces `max_request_tokens`. |
| `chunking.oversize` | `Split` | A file over the budget is split into windows at its strongest boundaries: Markdown headings, then top-level blocks; in Rust files that parse, items, then members, then statements; in other files, blank lines and the outermost indentation and bracket depth; then single lines. `Skip` leaves it out with the reason "too large (over the chunk request budget)"; `Block` blocks the job before any model request, naming the files. |
| `chunking.max_repair_rounds` | `Some(5)`; JSON `null` gives `None` | Repair requests per window; `None` repairs until every cut is valid. |
| `chunking.repair_limit` | `Merge` | When the repairs run out: `Merge` drops the invalid cuts, merging each disputed region into its neighbor (a `deterministic_repair` event); `Block` leaves the file unfinished and blocks the job. |

The OpenAI-or-bytes rule is `ArkoBackend`'s; a custom `Backend` reports its model's tokenizer and input limit through `Backend::chunk_input`. A job resolves its budget when it scans the source, records it as `chunk_budget` in its `snapshot` event, and keeps it when resumed, even if the client's auth path changed. Hidden files are indexed; `.gitignore` and `.ignore` files (the root's parent directories included), `.git/info/exclude` and the global git excludes file apply, even outside a repository. Symbolic links, files that aren't valid UTF-8 and files with NUL bytes are excluded with a reason. See [Which files are indexed](arko_index-reference.md#which-files-are-indexed) and [Request budget and large files](arko_index-reference.md#request-budget-and-large-files).

### Backend and chunk model (arko_index)

| Setting | Default | Notes |
|---|---|---|
| Chunk model of `ArkoBackend::new` | GPT-5.6 Luna with `default_chunk_config()`: `high` reasoning effort with summaries, `low` verbosity | `with_chunk_model(model, config)` selects another; the client passed to `new` must hold credentials for that model's provider. Changing the model keeps the boundaries of unchanged files. |
| Chunk tool configuration | The boundary tool (`submit_cuts`, or `repair_cuts` for repairs) replaces any `tool_config` in the settings | Forced where the model allows forcing it with those settings, with parallel calls disabled where supported. A reply must be exactly one call to the tool in a turn that ended with `EndTurn` or `ToolUse`: a refusal or a token or context limit fails the request, and a missing, extra or malformed call is retried as a transient failure. |
| Prompt-cache key of chunk requests | The job's ID (`ModelRequest::cache_key`) | The same for initial requests and repairs, before and after a resume. `ArkoBackend` sends it as the conversation's prompt-cache key, which over ChatGPT OAuth is also the `session-id` header. |
| Embedding requests of `ArkoBackend` | Voyage, the profile's dimensions, truncation off | Input type `document` for chunks and `query` for search queries. |
| `Backend::chunk_input` | No tokenizer, unknown input limit | Chunk requests are then measured in bytes against 51,200 bytes (10% of an assumed 128,000 tokens, at 4 bytes per token), with a warning. |
| `Backend::input_tokens` | `Backend::tokens` | The count used for embedding inputs and the query limit. |
| `Backend::chunk_model_info` | `None` | The `snapshot` event then records no chunk model. |

### Search and judge (arko_index)

`SearchOptions` and `JudgeOptions` fields omitted from JSON take these defaults.

| Setting | Default and omitted | Notes |
|---|---|---|
| `SearchOptions::mode` | `Hybrid` | `Lexical`, `Semantic`. Semantic and hybrid searches of a generation embedded with another profile fail with `Error::Invalid`; lexical search still works. |
| `SearchOptions::top_k` | `Some(10)` | Results per page, with a cursor for the rest; `None` returns every eligible result; `Some(0)` is an error. |
| `SearchOptions::min_chunk_bytes` | 40 | Chunks shorter than this many bytes are left out of search, not of `read_chunks`; 0 keeps them. |
| `SearchOptions::path_prefix`, `SearchOptions::literal` | `None` | A plain prefix of the root-relative path; a case-sensitive substring of the chunk text. |
| `SearchOptions::cursor` | `None` | A previous page's `next`. It needs the same query, `mode`, `top_k`, `min_chunk_bytes`, `path_prefix` and `literal`, or the search returns `Error::Stale`. |
| `SearchOptions::rebuild_file_limit` | `Some(100)` | Affected paths a rebuild notice lists; `None` lists all. |
| `JudgeOptions::candidate_limit` | 20 | Candidates retrieved and sent to the judge; must be positive and at least `top_k`. |
| `JudgeOptions::max_attempts` | 2 | The search fails when this many judge replies have failed validation, refused or incomplete ones included. |
| `JudgeOptions::max_transient_attempts` | 3 | The search fails when this many transient provider failures have occurred. A retry delay over 60 s (`MAX_JUDGE_RETRY_WAIT`) fails it at once; other provider errors, plan usage limits included, are returned at once. |
| `LlmSearchJudge::new(client, model)` | `GenerationConfig::default()`, no extra instructions, no idle timeout | No default model. Judged searches take no cursor and return none. |

Every search hit carries `may_be_stale`: true when the newest unfinished update (running, paused or blocked) lists its file as affected, or when the file under the index root no longer matches the chunk's snapshot (changed, removed, unreadable or over `max_file_bytes`). When the root isn't accessible, only the first applies. Search returns `Error::Blocked` until an index's first build has published.

### Watcher (arko_index)

`WatchOptions` has no serde form. `watch` rejects zero durations.

| Field | Default | Notes |
|---|---|---|
| `debounce` | 500 ms | How long the source must be quiet after a file-system event before it is scanned. |
| `reconcile` | `Some(30 s)` | Interval of full scans that catch unreported events; `None` scans at start and after events, and, while an update is held, looks again every twenty debounce periods. |
| `resume_interrupted` | `true` | Resumes an update paused with the problem "process interrupted; resume persisted work". |
| `retry_blocked_after` | `None` | Opt-in: resumes an update blocked only by transient provider failures, this long after the watcher found it blocked. |

An update paused with `pause` or by a usage limit, blocked for another reason, or one the watcher failed to resume, is held: a `Held` event is sent once per job, phase and problem, and no other update starts until the host resumes or cancels it. Events under directories the last scan didn't enter are ignored. See [Watching for changes](arko_index-reference.md#watching-for-changes).

### Jobs (arko_index)

- `start_build` and `start_update` return at once and run the job in the background: `snapshotting`, `chunking`, `embedding`, then `ready`. A pause request or a plan usage limit stops it as `paused`; a provider error, an invalid response or files that still need repair as `blocked`; `cancel` as `cancelled`; a local I/O, database, JSON or worker failure as `failed`.
- Transient failures are retried after a backoff that starts at 1 s and doubles up to 60 s, jittered and never shorter than the provider's delay. After `max_transient_attempts` a chunk request fails and the job is `blocked`; its `phase` event has `retryable: true` when only such failures blocked it. An embedding request that exhausts retries records `embedding_failed` before the job stops. Capacity signals wait without using attempts, up to `max_capacity_wait_ms`.
- Initial chunk requests and build embeddings are dispatched before repairs and retries; with `query_priority`, search query embeddings come first.
- A plan usage limit (ChatGPT `usage_limit_reached`) stops the job's queued requests and pauses it, with the reset time when the provider gave one. Nothing resumes it automatically.
- A job left running by a stopped process is `paused` when the database is next opened, with the problem "process interrupted; resume persisted work", or "paused" if a pause had been requested. Opening writes a matching `phase` event in the same transaction. Only a watcher resumes such a job automatically.
- `resume` requires the job's embedding profile, the current chunking policy (for a job that has scanned its source) and no newer published generation of the index. Every other `Config` field comes from the current configuration; changes are logged and recorded as a `settings_changed` event.
- A paused or blocked job keeps new updates of its index from starting; a failed or cancelled one doesn't.
- An update reuses validated boundaries of files with the same content and extension from the active generation and from the newest cancelled or failed job of the index, only when those ran with the same embedding profile and chunking policy. Vectors are reused per text and profile.

See [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries).

### Persisted data (arko_index)

- The database schema is version 2. A version 1 database is migrated when opened; a database with tables but no version, or with a newer version, is refused.
- Content is stored once: a snapshot per normalized file content, a chunk text and its full-text entry per location and content (shared by the generations and indexes that hold it), a vector per text and embedding profile. `delete` and pruning remove data nothing refers to, and remove vectors only while no job of the database is unfinished.
- The database stores no credentials: the saved configuration, specs and chunk requests hold none.
- A saved `BuildSpec` without `skip_directories` loads with the default list; one without `max_file_bytes` loads with 16 MiB, so at its next update an older index excludes files over 16 MiB with the reason "too large (over max_file_bytes)". To keep them, build again with a larger `with_max_file_bytes`.
- A job saved without a chunk budget batches its initial requests by `batch_count` only.
- `JobStatus::active_chunking` also reads the former name `active_luna`.
- A profile with an unset `tokenizer_revision` is saved without it; a saved profile holding `FORMER_DEFAULT_TOKENIZER_REVISION` matches an unset one.
- Saved usage totals whose costs aren't numbers load with those costs as zero; totals that can't be read at all are logged and restart from zero.

### Example CLI (arko_index)

`arko_index_cli`: global flags come before the command. Settings without a flag keep the library defaults.

| Flag | Default | Notes |
|---|---|---|
| `--database PATH` | Required | Created if missing, with its parent directories. |
| `--oauth-file PATH` | Unset: `LLMClient::from_env()` | An OpenAI client with ChatGPT OAuth credentials from a Codex `auth.json`, kept in step with the file; see [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials). |
| `--batch-count N`, `--concurrency N` | 6, 2 | `Config::batch_count` and `Config::concurrency`. |
| `--stream-idle-timeout-secs S` | Unset | `Config::stream_idle_timeout_ms`, also applied to the judge. |
| `build ROOT --name NAME` | The root directory's name | |
| `search --lexical`, `--semantic` | Hybrid | Mutually exclusive. |
| `search --top-k N`, `--min-chunk-bytes N`, `--rebuild-file-limit N` | 10, 40, 100 | |
| `watch --debounce-ms MS`, `--reconcile-secs S` | 500, 30 | `--reconcile-secs 0` turns periodic scans off. |
| `search --judge` | Off | Every other `--judge-*` flag requires it. |
| `--judge-provider` | `openai` | Or `anthropic`. |
| `--judge-model ID` | `gpt-5.6-luna` for OpenAI | Required for Anthropic. |
| `--judge-effort LEVEL` | `high` for an OpenAI reasoning model, unless `--judge-config` sets `reasoning` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`; requires an OpenAI reasoning model. |
| `--judge-candidates N`, `--judge-attempts N` | 20, 2 | `JudgeOptions::candidate_limit` and `max_attempts`; transient failures keep the library's 3. |
| `--judge-auth` | `inherit` | `inherit` reuses the global client (OpenAI only); `api-key` reads `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, or the variable `--judge-api-key-env NAME` names; `oauth` reads `--judge-oauth-file PATH`, else `--oauth-file`. |
| `--judge-config PATH` | Unset: `GenerationConfig` defaults | JSON `GenerationConfig` fields; unknown fields and `tool_config` are rejected. |
| `--judge-instructions-file PATH` | Unset | Selection preferences appended to the judge's instructions. |

`list`, `status` and `read` load no backend, so they need no credentials and no tokenizer. The other commands load the Voyage tokenizer; those that embed need `VOYAGE_API_KEY`. `build`, `update`, `resume` and `watch` need OpenAI credentials for the chunk model (GPT-5.6 Luna; no flag changes it), and so does `search --judge` with the default `--judge-auth inherit`. Ctrl-C pauses a running `build`, `update` or `resume` job. In `watch`, it stops the watcher and leaves an update it started unfinished: the database lists that update as paused when next opened, and the next `watch` resumes it. Logs go to standard error at `warn,arko_index=info` unless `RUST_LOG` is set. See [Example CLI](arko_index-reference.md#example-cli) and [Judge options](arko_index-reference.md#judge-options).

## Across crates

These defaults interact across the crates:

- **Model requests have no total deadline.** A response is awaited while its connection
  stays open. The opt-in stream-idle limits are `LLMClient::with_stream_idle_timeout`,
  `LLMRetryConfig::stream_idle_timeout_ms`, `Config::stream_idle_timeout_ms` for
  `arko_index`, and `LlmSearchJudge::with_stream_idle_timeout`. The other relevant
  limits are the 30-second connect timeout, the 60-second OAuth refresh timeout, the
  `arko_agents` host tool timeout (20 minutes by default), `wait_for_message`'s default
  270-second limit, the terminal wait cap, the judge's 60-second retry-delay cap, and the
  workflow's 30-second cancellation grace period.
- **Retries are layered.** `arko_ai` only resends an expired ChatGPT OAuth request once.
  `arko_agents` owns model-call retries and pauses at plan usage limits. `arko_workflows`
  answers those pauses automatically by default. `arko_index` gives chunk and embedding
  requests up to ten transient attempts and waits up to 15 minutes for capacity; judge
  validation and transient failures have separate budgets.
- **Approvals are layered.** A bare agent configuration asks for approval for calls other
  than `ask_user`; a workflow uses `ApproveAll` by default. Workspace path policy applies
  only to parameters marked as paths. Terminal, MCP and command tools are not confined by
  that file-path policy.
- **Caching is enabled by default.** Each conversation has its own cache key, while an
  `arko_index` job uses one key for all of its chunk requests and repairs.
- **Credentials are explicit.** Only `LLMClient::from_env` and
  `EmbeddingClient::from_env` read environment credentials. `WorkflowBuilder` and
  `ArkoIndex` do not; the Python workflow binding constructs its client with
  `LLMClient::from_env()`.
- **Costs use billed amounts.** Budgets and task totals use `UsageInfo::billed()`;
  ChatGPT OAuth subscription calls contribute zero billed cost while their list-price
  estimate remains in `total_cost`.
- **Persistence is durable but compatible.** Task files, checkpoints, event logs and
  request traces can contain full conversations and use default file permissions. New
  serialized fields have defaults and unknown fields are ignored; crate-specific
  migration rules are documented in each persisted-data section.

## Complete workflow example

[`complete_workflow.rs`](../arko_workflows/examples/complete_workflow.rs) is a runnable
workflow that combines explicit agent configuration, built-in tools, memory, messaging,
completion verification, tool-result distillation, persistence, and resuming. It supports a
credential-free `--dry-run` that validates the setup and writes a checkpoint, live execution
over ChatGPT OAuth, and `--resume` from a checkpoint. The source is the canonical example;
its comments identify the defaults and the settings that are intentionally explicit.
