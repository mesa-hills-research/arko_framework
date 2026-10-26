# arko_framework reference map

This is the all-in-one map of the public arko_framework API. It names each configurable
feature and public primitive, points to the defining Rust file, and links to the detailed
behavior and defaults for using it correctly.

- [arko_ai](#arko_ai)
- [arko_agents](#arko_agents)
- [arko_workflows](#arko_workflows)
- [arko_index](#arko_index)
- [Complete workflow example](#complete-workflow-example)

The crates build on one another. `arko_agents` uses `arko_ai`, and `arko_workflows` uses both.
`arko_index` stands alone. Code paths in the tables are repository-relative. Defaults
and omitted serialized fields are called out separately because they can differ.

## arko_ai

Clients for OpenAI (API key or ChatGPT OAuth), Anthropic, Google Gemini and OpenRouter models, and for Voyage text embeddings, with one conversation, response, usage and error model across providers.

See [arko_ai guide](arko_ai-guide.md) ([OpenAI and ChatGPT OAuth](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth), [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials), [OpenRouter](arko_ai-guide.md#openrouter), [Voyage embeddings](arko_ai-guide.md#voyage-embeddings)).

### Map (arko_ai)

Code paths are relative to the repository root. Items are listed by their shortest public path. Most types are re-exported at the crate root (`arko_ai::LLMClient`). The others are reached through their module (`arko_ai::providers::openai::OpenAIClient`, `arko_ai::model::ModelType`).

**Modules**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_ai::client` | `LLMClient`, the client that sends conversations to every provider. | `arko_ai/src/client.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `arko_ai::codex_auth` | Reading a Codex `auth.json` and keeping it in step with rotated ChatGPT OAuth tokens (`CodexAuthFile`, `CodexAuthStore`, `CodexAuthSync`). | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `arko_ai::config` | `GenerationConfig` and the enums of its fields. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `arko_ai::conversation` | `Conversation` and `OpenAITurnState`. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `arko_ai::error` | `Error`, `RetryClass` and `ErrorKind`. | `arko_ai/src/error.rs` | [Errors and retries](arko_ai-guide.md#errors-and-retries) |
| `arko_ai::input_limit` | Input-token limit of a model on the auth path in use. | `arko_ai/src/input_limit.rs` | [Models](arko_ai-guide.md#models) |
| `arko_ai::message` | `Message`, `Role`, `Content`, `Image`, `ConfigurationUpdate`. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `arko_ai::model` | The model registry, with `Model`, `Provider`, `ModelType`, `ModelPricing`. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `arko_ai::pricing` | Price variants, with `ServiceTier`, `TierPricing`, `ResidencyPricing`, `ScheduledPricing`. | `arko_ai/src/pricing.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `arko_ai::prompt_cache` | How long providers keep prompt-cache entries. | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `arko_ai::providers` | The provider clients, `GenerationResponse`, `StreamEvent`, `ProviderClient`, `EmbeddingResponse`, and the shared HTTP client. | `arko_ai/src/providers/mod.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `arko_ai::providers::openai` | OpenAI Responses API client, API key and ChatGPT OAuth. | `arko_ai/src/providers/openai.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `arko_ai::providers::anthropic` | Anthropic Messages API client. | `arko_ai/src/providers/anthropic.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `arko_ai::providers::google` | Gemini client on the Interactions API. | `arko_ai/src/providers/google.rs` | [Generation settings and tool calls](arko_ai-guide.md#generation-settings-and-tool-calls) |
| `arko_ai::providers::openrouter` | OpenRouter Chat Completions client, its options and model catalog. | `arko_ai/src/providers/openrouter.rs` | [OpenRouter](arko_ai-guide.md#openrouter) |
| `arko_ai::providers::voyage` | Voyage embeddings client. | `arko_ai/src/providers/voyage.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `arko_ai::reasoning` | `ReasoningConfig`, the effort and level enums, and `ReasoningArtifact`. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `arko_ai::retry` | `BackoffPolicy` and `MAX_SERVER_DELAY_HINT`. | `arko_ai/src/retry.rs` | [Errors and retries](arko_ai-guide.md#errors-and-retries) |
| `arko_ai::stop` | `StopReason`, `Refusal`, `RefusalSource`. | `arko_ai/src/stop.rs` | [Stop reasons and refusals](arko_ai-guide.md#stop-reasons-and-refusals) |
| `arko_ai::tool` | Tool definitions, calls, results and the tool choice. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `arko_ai::trace` | `RequestTrace`, the opt-in request trace. | `arko_ai/src/trace.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `arko_ai::usage` | `UsageInfo`, `ModelCallUsage`, `PlanUsage`, `BillingKind`, `ContextEvent`. | `arko_ai/src/usage.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `arko_ai::embedding_client` | `EmbeddingClient`. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `arko_ai::embedding_config` | `EmbeddingConfig` and `VoyageInputType`. | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `arko_ai::embedding_model` | `EmbeddingModelDetails`, the embedding model registry. | `arko_ai/src/embedding_model.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `arko_ai::embedding_provider_identifier` | `EmbeddingProviderIdentifier`. | `arko_ai/src/embedding_provider_identifier.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `arko_ai::token_utils` | Local token counting with Hugging Face tokenizers (`TokenCounter`, `count_tokens`). | `arko_ai/src/token_utils.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |

**Clients and credentials**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `LLMClient` | Holds one client per provider (OpenAI, Anthropic, Google, OpenRouter) and sends each conversation to the client of its model's provider. Clones share the OpenAI credentials. | `arko_ai/src/client.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `LLMClient::new`, `impl Default for LLMClient` | A client with no provider configured. Reads nothing from the environment. | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::from_env`, `LLMClient::from_env_with` | Configures every provider whose credentials are in the environment (`from_env_with` reads each variable through a function). | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_openai_api_key`, `with_anthropic_api_key`, `with_google_api_key`, `with_openrouter_api_key` | Configures one provider with an API key, replacing any client configured for it before. | `arko_ai/src/client.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `LLMClient::with_openai_chatgpt_oauth`, `with_openai_chatgpt_oauth_rate_limit_retries` | Configures OpenAI with ChatGPT OAuth credentials kept in memory. The second also sets the token-refresh rate-limit retries (off with the first). | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_codex_auth_file` | Configures OpenAI with the ChatGPT OAuth credentials in a Codex `auth.json`, kept in step with the file. | `arko_ai/src/client.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `LLMClient::with_chatgpt_accounts`, `with_codex_auth_files`, `chatgpt_accounts` | Configures OpenAI with an ordered list of ChatGPT OAuth accounts that fail over to each other on usage limits. Returns the list. | `arko_ai/src/client.rs` | [Several ChatGPT accounts](arko_ai-guide.md#several-chatgpt-accounts) |
| `ChatGptAccounts` (`new`, `from_codex_auth_files`, `with_status_check_interval`, `with_usage_url`, `with_observer`, `accounts`, `active_account`, `statuses`, `check_now`, `mark_usable`, `read_usage`). `ChatGptAccount` (`new`, `from_codex_auth_file`, `label`, `client`) | The account list. Each request goes to the first account the backend has not refused. Refused accounts come back at their reset time, on a usage check, or when marked usable. | `arko_ai/src/providers/openai/accounts.rs` | [Several ChatGPT accounts](arko_ai-guide.md#several-chatgpt-accounts) |
| `ChatGptAccountEvent::{Exhausted, Switched, Usable, AllExhausted, UsingCredits}`, `SwitchReason`, `UsableReason`. `ChatGptAccountStatus` (`drawing_credits`, `using_credits`). `ChatGptUsageStatus` (`from_json`, `usable`). `CHATGPT_USAGE_URL`, `DEFAULT_STATUS_CHECK_INTERVAL` | What happened to the accounts (`Display` writes a notification sentence), each account's state, and the usage endpoint's report. | `arko_ai/src/providers/openai/accounts.rs` | [Several ChatGPT accounts](arko_ai-guide.md#several-chatgpt-accounts) |
| `LLMClient::with_openai_auth`, `LLMClient::update_openai_auth` | Installs an OpenAI client with the given `OpenAIAuth`. Replaces the credentials of the configured OpenAI client in place, for every clone (`Error::MissingApiKey` without one). | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_openai_client`, `with_anthropic_client`, `with_google_client`, `with_openrouter_client` | Installs a preconfigured provider client. The `LLMClient`'s own settings (OpenAI trace, hook, `originator`, `User-Agent`, the stream idle timeout) are applied to it only where they are set. | `arko_ai/src/client.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `LLMClient::set_openai_client`, `set_anthropic_client`, `set_google_client`, `set_openrouter_client` | Sets a provider's client on a client in use, for every clone made before or after. It comes before the one the client was built with, and requests that start afterwards use it. | `arko_ai/src/client.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `LLMClient::with_openai_originator`, `LLMClient::with_openai_user_agent` | Sends an `originator` or `User-Agent` header with every OpenAI request. None by default. | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_request_trace` | Writes every OpenAI request and response to a `RequestTrace` directory. | `arko_ai/src/client.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `LLMClient::with_openai_request_hook` | Installs an `OpenAIRequestHook` on the OpenAI client. | `arko_ai/src/client.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `LLMClient::with_stream_idle_timeout` | Opt-in limit on the time between data on a streamed response, for every provider client. | `arko_ai/src/client.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `LLMClient::generate_text`, `LLMClient::generate_stream` | Sends a `Conversation` with a `GenerationConfig` to the model's provider. Returns a `GenerationResponse`, or a stream of `StreamEvent`s ending with `End`. | `arko_ai/src/client.rs` | [Quick start](arko_ai-guide.md#generating-text) |
| `LLMClient::new_conversation` | `Conversation::new(model)`. | `arko_ai/src/client.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `LLMClient::has_provider`, `LLMClient::openrouter` | Whether a provider is configured. The configured `OpenRouterClient`, for its catalog. | `arko_ai/src/client.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `LLMClient::is_openai_chatgpt_oauth`, `openai_chatgpt_oauth_credentials`, `is_openai_token_expired` | Whether OpenAI uses ChatGPT OAuth. A snapshot of the current credentials (it waits for a refresh in flight). Whether the access token expires within a margin (`true` when its expiry can't be read). | `arko_ai/src/client.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `LLMClient::refresh_openai_oauth_token`, `refresh_openai_oauth_token_once`, `refresh_openai_oauth_token_if_expiring` | Refreshes the ChatGPT OAuth credentials, explicitly (with the opt-in rate-limit retries), explicitly with one request, or only when the access token expires within a margin. | `arko_ai/src/client.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `LLMClient::prompt_cache_lifetime` | The prompt-cache lifetime of this client's requests for a model and config, using the client's OpenAI auth path. | `arko_ai/src/client.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `LLMClient::max_input_tokens` | The input-token limit of a model on this client's auth path (ChatGPT OAuth caps OpenAI models at 272,000). | `arko_ai/src/client.rs` | [ChatGPT OAuth (arko_ai)](#chatgpt-oauth-arko_ai) |
| `providers::openai::OpenAIClient`, `::new`, `::new_chatgpt_oauth`, `::with_auth` | The OpenAI provider client, with an API key, ChatGPT OAuth credentials, or an `OpenAIAuth`. | `arko_ai/src/providers/openai.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `OpenAIClient::with_base_url`, `with_oauth_token_url`, `with_stream_idle_timeout` | Sends requests to `{base_url}/responses`. Sends token refreshes to another URL. Sets or clears (`None`) the stream idle timeout. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIClient::with_oauth_rate_limit_retries`, `with_credential_store` | Opt-in retries of a rate-limited explicit token refresh. A `ChatGptOAuthStore` kept in step with the credentials. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIClient::with_request_trace`, `with_request_hook`, `with_originator`, `with_user_agent` | The client-level forms of the `LLMClient` settings. Each takes an `Option`, so `None` clears it. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIClient::update_auth`, `is_chatgpt_oauth`, `chatgpt_oauth_credentials`, `is_oauth_token_expired`, `refresh_oauth_token`, `refresh_oauth_token_once`, `refresh_oauth_token_if_expiring` | The client-level forms of the `LLMClient` credential methods. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `providers::anthropic::AnthropicClient`, `::new`, `::with_base_url`, `::with_stream_idle_timeout` | The Anthropic provider client. Requests go to `{base_url}/v1/messages`. | `arko_ai/src/providers/anthropic.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `providers::google::GoogleClient`, `::new`, `::with_base_url`, `::with_stream_idle_timeout` | The Gemini provider client. Requests go to `{base_url}/interactions`. | `arko_ai/src/providers/google.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIAuth` (`ApiKey`, `ChatGptOAuth`) | OpenAI credentials, an API key, or `ChatGptOAuthCredentials`. | `arko_ai/src/providers/openai.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `ChatGptOAuthCredentials` (`access_token`, `refresh_token`, `account_id`) | ChatGPT OAuth credentials. `Debug` output redacts all three. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `OAuthRefreshResult` (`access_token`, `refresh_token`, `id_token`) | The tokens a refresh returned. `Debug` output redacts them. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `ChatGptOAuthStore` (`before_refresh`, `save_refreshed`, `refresh_lock`) | Extension trait, storage shared with other programs that a client re-reads before every refresh and writes rotated tokens to. `refresh_lock` defaults to `None`. | `arko_ai/src/providers/openai/credential_store.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `CodexAuthFile`, `::default_path`, `::load_default`, `::load`, `::path`, `::credentials` | A Codex `auth.json` (nested `tokens` or top-level token fields). Its default location (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`), loading it (a relative path is made absolute), and its credentials. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `CodexAuthFile::openai_client`, `CodexAuthFile::client` | An `OpenAIClient` (or an `LLMClient` holding one) with the file's credentials and a `CodexAuthStore` of the file. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `CodexAuthFile::persist`, `persist_tokens`, `write_recovery`, `write_recovery_tokens`, `sync` | Writes credentials to the file (backup, then atomic replace, never over newer credentials another process wrote). Writes them to a new `auth.json.recovered-<unix seconds>` file. Reconciles the file with a client configured another way. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `CodexAuthStore`, `::new`, `::path`, `impl ChatGptOAuthStore for CodexAuthStore` | The `ChatGptOAuthStore` of a Codex `auth.json`. Every store of the same file in a process shares one refresh lock. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `CodexAuthSync` (`Unchanged`, `Saved`, `Reloaded`) | The result of `CodexAuthFile::persist` and `sync`. | `arko_ai/src/codex_auth.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `refresh_chatgpt_oauth_token` | Exchanges a refresh token for new tokens once, with a caller-supplied HTTP client. `client_id` and `token_url` default to the Codex CLI's. 60-second timeout. Refuses a refresh token this process knows is spent. | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `parse_jwt_expiry`, `is_token_expired` | The `exp` claim of a JWT. Whether it expires within a margin (`true` when it can't be read). | `arko_ai/src/providers/openai.rs` | [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials) |
| `http_client`, `http_client_builder` | The HTTP client (or its builder) the providers use, rustls with the OS trust store, HTTP/1.1, 30-second connect timeout, no total deadline, no redirects. | `arko_ai/src/providers/mod.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `RequestTrace`, `::new`, `::dir` | An opt-in trace directory for OpenAI requests and responses (`calls.jsonl` plus one request and response file per call). | `arko_ai/src/trace.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `OpenAIRequestHook` (`prepare`, `observe_response`) | Extension trait, adjusts each OpenAI request's headers and body before sending, and observes response headers. Both methods default to doing nothing. | `arko_ai/src/providers/openai/request_hook.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |

**Requests and responses**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `GenerationConfig`, `impl Default` | Per-request settings. Every field defaults to `None`. Omitted fields deserialize to their defaults, so partial JSON or YAML is valid. A setting the model's provider has no use for is dropped with a warning. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::temperature`, `with_temperature` | Sampling temperature. `None` sends none. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::new` | Deprecated, with `GenerationConfig::default().with_temperature(t)`. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::max_tokens`, `with_max_tokens` | Output-token limit. `None` means the model's own maximum. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::verbosity`, `with_verbosity`. `Verbosity` (`Low`, `Medium`, `High`) | Output verbosity (OpenAI GPT-5 and GPT-6, OpenRouter). | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::reasoning`, `with_reasoning` | The `ReasoningConfig` for the request. Each provider takes its own variants. | `arko_ai/src/config.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `GenerationConfig::openai_reasoning_mode`, `with_openai_reasoning_mode` | OpenAI `reasoning.mode` (`standard`, `pro`) for GPT-5.6 and GPT-6 on API-key requests. ChatGPT OAuth omits it and warns once. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::anthropic_thinking_enabled`, `with_anthropic_thinking_enabled` | Turns Anthropic thinking on (adaptive where no reasoning config is given) or off. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::prompt_caching`, `with_prompt_caching` | Turns prompt caching on or off for the request. `None` inherits `Conversation::enable_cache`. | `arko_ai/src/config.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `GenerationConfig::anthropic_summarized_thinking`, `with_anthropic_summarized_thinking` | Requests summarized Claude thinking text on models that omit it by default. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::long_context`, `with_long_context` | Requests Claude's 1M-token context (standard on Claude 4.6 and later, where nothing is sent). | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::tool_config`, `with_tool_config` | Tool definitions, tool choice and parallel-call setting for the request. | `arko_ai/src/config.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `GenerationConfig::interleaved_thinking`, `with_interleaved_thinking` | Claude interleaved thinking between tool calls. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::openai_service_tier`, `with_openai_service_tier`. `OpenAIServiceTier` (`Auto`, `Default`, `Flex`, `Fast`), `OpenAIServiceTier::pricing_tier` | OpenAI `service_tier` (`Fast` is sent as `priority`). Over ChatGPT OAuth only `Fast` (on the models Codex offers it for) and `Default` (sends no tier). `pricing_tier` maps it to the `ServiceTier` used for cost (`Auto` and `Default` price as `Standard`). | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::anthropic_cache_ttl`, `with_anthropic_cache_ttl` | `ttl` of the `cache_control` arko sends for Claude (`AnthropicCacheTtl`). | `arko_ai/src/config.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `GenerationConfig::anthropic_fast_mode`, `with_anthropic_fast_mode` | Anthropic fast mode (`speed: "fast"`), Opus 5.5, Opus 5 and Opus 4.8. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::anthropic_inference_geo`, `with_anthropic_inference_geo`. `AnthropicInferenceGeo` (`Global`, `Us`) | Anthropic `inference_geo`, Claude 4.6 and later. | `arko_ai/src/config.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `GenerationConfig::openrouter`, `with_openrouter_options` | Per-request `OpenRouterOptions`, merged over the client's default options. | `arko_ai/src/config.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `providers::ProviderClient` (`generate_text`, `generate_stream`) | The trait the four provider clients implement. `LLMClient` holds only its own concrete clients, so another implementation can't be plugged into it. | `arko_ai/src/providers/mod.rs` | [Providers and credentials](arko_ai-guide.md#providers-and-credentials) |
| `GenerationResponse` (`message`, `usage`, `model_call_usage`, `stop_reason`, `refusal`, `openai_turn_state`) | The result of a model call. The assistant message, its `UsageInfo` and `ModelCallUsage`, the normalized `StopReason`, a `Refusal` when one happened, and the ChatGPT backend's turn-state token. | `arko_ai/src/providers/mod.rs` | [Quick start](arko_ai-guide.md#generating-text) |
| `GenerationResponse::text`, `served_model`, `rerouted` | The concatenated message text. The model the provider reports serving. Whether another model than requested served the call. | `arko_ai/src/providers/mod.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `GenerationResponse::single_tool_call` | The one call to a named tool in a response forced to call it. `Error::Refusal`, `Error::UnexpectedToolCalls` or `Error::InvalidToolArguments` otherwise. | `arko_ai/src/providers/mod.rs` | [Generation settings and tool calls](arko_ai-guide.md#generation-settings-and-tool-calls) |
| `StreamEvent` (`ReasoningChunk`, `TextChunk`, `Refusal`, `ToolCallStart { index, name }`, `ToolCallArgumentsDelta { index, delta }`, `End`) | Events of `generate_stream`, reasoning and text fragments, a refusal (once, just before `End`), native tool-call starts and argument fragments, and `End` with the complete `GenerationResponse`. | `arko_ai/src/providers/mod.rs` | [Quick start](arko_ai-guide.md#generating-text) |

**Reasoning**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ReasoningConfig::OpenAI { effort, include_summaries }`, `ReasoningConfig::openai` | OpenAI reasoning effort. `include_summaries: true` sends `reasoning.summary: "auto"`. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::Anthropic { budget_tokens, effort }`, `anthropic`, `anthropic_with_effort` | Claude manual thinking budget, with an optional effort. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::AnthropicAdaptive { effort }`, `anthropic_adaptive`, `anthropic_adaptive_with_effort` | Claude adaptive thinking (Claude 4.6 and later), with an optional effort. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::Gemini { thinking_level, include_summaries }`, `gemini` | Gemini thinking level. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::Google { thinking_budget, include_summaries }`, `google` | A Gemini thinking budget, mapped to a thinking level before sending. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::OpenRouter(OpenRouterReasoning)`, `openrouter`, `openrouter_effort` | OpenRouter's `reasoning` object. The only variant OpenRouter models take. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningConfig::to_openrouter` | The closest `OpenRouterReasoning` for another variant. `Error::InvalidRequest` where none exists (a budget with an effort, a negative Google budget other than -1). | `arko_ai/src/reasoning.rs` | [Reasoning](arko_ai-guide.md#reasoning) |
| `ReasoningConfig::is_enabled`, `ReasoningConfig::provider` | Whether the config turns reasoning on (false only for an Anthropic or Google budget of 0, and for an OpenRouter config with `enabled: false` or effort `none`). The `Provider` whose models take it. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `OpenAIEffort` (`None`, `Minimal`, `Low`, `Medium`, `High`, `XHigh`, `Max`) | OpenAI efforts, validated per model when the request is built. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `OpenAIReasoningMode` (`Standard`, `Pro`) | OpenAI reasoning mode (`GenerationConfig::openai_reasoning_mode`, `OpenRouterReasoning::mode`). | `arko_ai/src/reasoning.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `AnthropicReasoningEffort` (`Low`, `Medium`, `High`, `XHigh`, `Max`) | Claude efforts, validated per model. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `GeminiThinkingLevel` (`Minimal`, `Low`, `Medium`, `High`), `GeminiThinkingLevel::as_str` | Gemini thinking levels. Saved as `"MINIMAL"` to `"HIGH"`, sent lowercase. `as_str` gives the uppercase form. | `arko_ai/src/reasoning.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `ReasoningEffort` (`None`, `Minimal`, `Low`, `Medium`, `High`, `XHigh`, `Max`), `ReasoningEffort::as_str`, `From<OpenAIEffort>`, `From<AnthropicReasoningEffort>`, `From<GeminiThinkingLevel>` | Provider-independent effort used by `OpenRouterReasoning` and mid-conversation effort updates. `as_str` is the wire value. | `arko_ai/src/reasoning.rs` | [Reasoning](arko_ai-guide.md#reasoning) |
| `ReasoningSummary` (`Auto`, `Concise`, `Detailed`), `ReasoningContext` (`Auto`, `AllTurns`, `CurrentTurn`) | OpenRouter `reasoning.summary` and `reasoning.context` values. | `arko_ai/src/reasoning.rs` | [Reasoning](arko_ai-guide.md#reasoning) |
| `OpenRouterReasoning` (`effort`, `max_tokens`, `exclude`, `enabled`, `summary`, `context`, `mode`), `::effort`, `::max_tokens`, `::enabled`, `::disabled`, `with_exclude`, `with_summary`, `with_context`, `with_mode`, `impl Default` | OpenRouter's `reasoning` object. Unset fields are omitted, so the model's defaults apply. `effort` and `max_tokens` are exclusive. | `arko_ai/src/reasoning.rs` | [Reasoning](arko_ai-guide.md#reasoning) |
| `ReasoningArtifact` (`OpenAIEncryptedReasoning { items }`, `OpenAIOutputItems { items }`, `AnthropicReasoning { fragments }`, `GoogleReasoning { fragments }`, `OpenRouter { details }`), `ReasoningArtifact::provider` | Provider state kept on an assistant message so the turn can be replayed, OpenAI output items (or older encrypted reasoning items), Claude thinking blocks, Gemini thought signatures, OpenRouter `reasoning_details`. | `arko_ai/src/reasoning.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `ReasoningArtifact::new_openai_encrypted`, `new_anthropic_thinking`, `new_anthropic_redacted`, `new_anthropic_mixed`, `new_google_reasoning`, `new_openrouter` | Constructors of the artifact variants. | `arko_ai/src/reasoning.rs` | [Persisted data (arko_ai)](#persisted-data-arko_ai) |
| `reasoning::AnthropicReasoningFragment` (`Thinking { thinking, signature }`, `RedactedThinking { data }`), `reasoning::GoogleReasoningFragment` (`summary`, `signature`), `OpenAIReasoningItem` (`encrypted_content`, `summaries`, `id`) | The parts of the Claude, Gemini and older OpenAI artifacts. | `arko_ai/src/reasoning.rs` | [Persisted data (arko_ai)](#persisted-data-arko_ai) |

**Conversations and messages**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Conversation` (`model`, `system_message`, `enable_cache`, `prompt_cache_key`, `messages`, `openai_turn_state`) | The model, system message, caching switch, prompt-cache key, message history and ChatGPT turn state sent with each request. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::new`, `with_system_message`, `with_cache_disabled`, `with_prompt_cache_key` | A new conversation (caching on, a random UUID key). Set the system message. Turn caching off. Set the key. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_user_message`, `add_user_message_with_image`, `add_user_message_with_contents` | Append a user message. One without tool results starts a new turn (and drops the ChatGPT turn-state token). | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::refresh_from(&source)` | Makes the conversation equal to `source`, as a clone does, but keeps its leading messages that `source` still has instead of cloning them again. For a host that keeps a copy of a growing conversation, such as one saved at every step. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_assistant_message`, `add_assistant_message_with_reasoning` | Append an assistant message, optionally with a `ReasoningArtifact`. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_assistant_message_from_response` | Append a response's message (not for classifier and content-filter refusals) and record its turn-state token. | `arko_ai/src/conversation.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Conversation::add_tool_results`, `add_tool_results_with_text` | Append a user message of tool results, optionally followed by text. | `arko_ai/src/conversation.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `Conversation::tool_history_as_text`, `into_tool_history_as_text`, `TOOL_HISTORY_NOTE` | The conversation for a request without tools. Each tool call as a line `[Earlier tool call ID: NAME, arguments ARGS]` (without the reasoning of its turn), and each message of results as one record, opened by a note that the tools are gone and the quoted output is data, with each result's output quoted line by line under its call's ID and tool, followed by its images. Borrowed (or returned as it is) when there are no calls. | `arko_ai/src/tool_history.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `Conversation::add_reasoning_effort_update`, `add_configuration_update` | Append a mid-conversation `ConfigurationUpdate` (OpenRouter only, replaces a pending update). | `arko_ai/src/conversation.rs` | [Mid-conversation effort updates](arko_ai-guide.md#mid-conversation-effort-updates) |
| `Conversation::current_openai_turn_state`, `record_openai_turn_state`. `OpenAITurnState` (`token`, `turn_start`, `turn_start_fingerprint`) | The ChatGPT `x-codex-turn-state` token the current turn echoes, recording one, and the saved token with the index and SHA-256 of the turn's starting message. | `arko_ai/src/conversation.rs` | [ChatGPT OAuth (arko_ai)](#chatgpt-oauth-arko_ai) |
| `Conversation::last_user_turn`, `rephrase_last_user_turn`, `remove_last_user_turn`, `redact_tool_result` | Edits for a refused input, find, rephrase or remove the last user turn (dropping later messages). Replace one tool result's text (its images stay). `Error::InvalidRequest` when there is nothing to edit. | `arko_ai/src/conversation.rs` | [Stop reasons and refusals](arko_ai-guide.md#stop-reasons-and-refusals) |
| `Message` (`role`, `content`, `reasoning_artifact`, `tool_calls`, `id`, `configuration_update`) | One message of the history. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Message::new_user`, `new_user_with_image`, `new_user_with_contents`, `new_assistant`, `new_assistant_with_reasoning`, `new_system`, `new_configuration_update` | Message constructors. A `Role::System` message in `messages` is an error on OpenAI, Anthropic and Gemini (OpenRouter sends it as a system message). Use `Conversation::system_message`. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Message::image_count`, `has_images`, `remove_images`, `has_tool_calls`, `is_configuration_update` | Counts and removes images, including those inside tool results (their text stays). | `arko_ai/src/message.rs` | [Tool images](arko_ai-guide.md#tool-images) |
| `image_store::ImageStore` (`save`, `load`), `image_store::with_image_store`, `image_store::sample_hash` | Inside `with_image_store(store, f)` a base64 `Image` serializes as `{"Base64": {"stored": <the store's name for it>, "mime_type": ...}}` and such JSON reads back through the store, so a host saving conversations often keeps each image once. Outside, images serialize with their data, and JSON holding a name fails to read with `image '...' is saved outside this JSON, and no image store is set to read it`. `sample_hash` hashes an image's length and 32 slices of its data, to find candidates cheaply. | `arko_ai/src/image_store.rs` | [Persistence](../arko_workflows/INDEX.md#persistence-and-checkpoint-state) |
| `Role` (`User`, `Assistant`, `System`) | Message roles. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Content` (`Text`, `Image`, `ToolResult`), `Content::as_text`, `as_image`, `as_tool_result` | Message content parts. | `arko_ai/src/message.rs` | [Conversation (arko_ai)](#conversation-arko_ai) |
| `Image` (`Url`, `Base64 { data, mime_type }`), `Image::new_url`, `Image::new_base64`, `Image::from_bytes` | Image input. Anthropic and Gemini accept only `Base64`. The data is an `Arc<str>`, so a cloned message or conversation shares its bytes. | `arko_ai/src/message.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `Image::dimensions`, `Image::estimated_tokens`, `Model::image_tokens` | An inline image's size from its header (PNG, JPEG, GIF, WebP, decoding only the header's bytes), and its input tokens by the model's provider's rule, Anthropic's `⌈w/28⌉ × ⌈h/28⌉` within 2,576 px and 4,784 tokens on Claude 4.7 and later (1,568 and 1,568 before), OpenAI's 32 px patches times 1.2 or 512 px tiles, Google's 258-token tiles. `estimated_tokens` counts as for a current Claude model. | `arko_ai/src/message.rs`, `arko_ai/src/image_tokens.rs` | [Request size by section](arko_ai-guide.md#request-size-by-section) |
| `RequestSizes` (`system_prompt`, `tool_definitions`, `user_text`, `assistant_text`, `tool_calls`, `tool_results`, `images`, `image_tokens`), `RequestSizes::{new, of, of_request, of_messages, text_of_messages, add_message, add_message_for, text_bytes, tokens, estimated_tokens}` | A request's size by section, bytes of text per section, and images with their estimated tokens. `text_of_messages` measures the text only, without reading the images' headers. | `arko_ai/src/request_size.rs` | [Request size by section](arko_ai-guide.md#request-size-by-section) |
| `ConfigurationUpdate` (`reasoning_effort`), `ConfigurationUpdate::reasoning_effort` | A reasoning-effort change from a point of the conversation onward. | `arko_ai/src/message.rs` | [Mid-conversation effort updates](arko_ai-guide.md#mid-conversation-effort-updates) |

**Tools**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ToolDefinition` (`name`, `description`, `parameters`, `strict`), `ToolDefinition::new`, `with_strict` | A tool the model may call, with a JSON Schema for its parameters. Not the same type as arko_agents' `ToolDefinition`. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `ToolCall` (`id`, `name`, `arguments`, `thought_signature`, `provider_id`, `invalid_arguments`) | A call the model made. Arguments that aren't valid JSON are kept as text in `invalid_arguments`, with `arguments` `null`. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `ToolResult` (`call_id`, `name`, `output`, `is_error`, `images`), `ToolResult::new`, `error`, `from_tool_call`, `error_from_tool_call`, `with_images` | The result of a call, matched by `call_id`, optionally with images. | `arko_ai/src/tool.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `ToolResultImage` (`label`, `image`), `ToolResultImage::new` | An image returned with a tool result, with a short label. | `arko_ai/src/tool.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `ToolChoice` (`Auto`, `Required`, `Specific`, `None`) | How the model selects tools. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `ToolConfig` (`tools`, `tool_choice`, `parallel_tool_calls`), `ToolConfig::new`, `with_tool_choice`, `with_parallel_tool_calls` | The tool setup of a request. | `arko_ai/src/tool.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |

**Stop reasons and refusals**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `StopReason` (`EndTurn`, `ToolUse`, `MaxTokens`, `StopSequence`, `Refusal`, `PauseTurn`, `ContextWindowExceeded`, `Other`), `StopReason::is_truncated` | Why generation stopped, normalized across providers. `is_truncated` is true for `MaxTokens` and `ContextWindowExceeded`. | `arko_ai/src/stop.rs` | [Stop reasons and refusals](arko_ai-guide.md#stop-reasons-and-refusals) |
| `RefusalSource` (`SafetyClassifier`, `Model`, `ContentFilter`), `RefusalSource::keeps_turn` | Who refused. Only a `Model` refusal keeps the turn in history. | `arko_ai/src/stop.rs` | [Stop reasons and refusals](arko_ai-guide.md#stop-reasons-and-refusals) |
| `Refusal` (`source`, `category`, `explanation`, `partial_output`, `provider_details`), `Refusal::new`, `impl Display` | A refusal's details. Displayed as "refused by <source> (<category>): <explanation>". | `arko_ai/src/stop.rs` | [Stop reasons and refusals](arko_ai-guide.md#stop-reasons-and-refusals) |

**Models and pricing**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Provider` (`OpenAI`, `Anthropic`, `Google`, `OpenRouter`), `Provider::as_telemetry_str` | A model's provider. The telemetry names are `openai`, `anthropic`, `gemini`, `openrouter`. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `model::ModelType` (`Standard`, `Reasoning`) | Whether a model reasons. On OpenAI it decides whether `temperature` is accepted and `reasoning.encrypted_content` is requested. | `arko_ai/src/model.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Model` (`provider`, `id`, `model_type`, `max_input_tokens`, `max_output_tokens`, `supports_image_input`, `supports_cache`, `pricing`, `openrouter`) | A model's identity, limits, capabilities and prices. `max_output_tokens` is what OpenAI (API key), Anthropic and Gemini requests send when `max_tokens` is unset. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::list`, `Model::all`, `Model::find`, `Model::get` | The registered models of a provider (empty for OpenRouter). All of them. One by id. One by provider and id (`Error::ModelNotSupported` otherwise, always for OpenRouter). | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `providers::openai::list_models`, `providers::anthropic::list_models`, `providers::google::list_models` | The same lists as `Model::list` for each provider. | `arko_ai/src/providers/openai.rs`, `anthropic.rs`, `google.rs` | [Models](arko_ai-guide.md#models) |
| `Model::openrouter`, `Model::openrouter_info` | An OpenRouter model by slug, built without a network call (limits and prices 0, image input and caching assumed). The catalog metadata of a model taken from the OpenRouter catalog. | `arko_ai/src/model.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `Provider::ClaudeCode`, `Model::claude_code` | Anthropic models run through the user's installed Claude Code CLI. `Model::claude_code(model)` wraps a registered Anthropic model (same id, limits and list prices). `Model::list(Provider::ClaudeCode)` mirrors the Anthropic registry. `Model::all` and `Model::find` leave it out. `LLMClient` rejects these models (`Error::ModelNotSupported`). | `arko_ai/src/model.rs` | [Claude Code](arko_ai-guide.md#claude-code) |
| `arko_ai::claude_code` (`ClaudeSession`, `ClaudeOptions`, `SessionStart`, `SystemPrompt`, `BuiltInTools`, `Effort`, `PermissionMode`, `Hook`, `HookCallback`, `HookEvent`, `PermissionHandler`, `PermissionRequest`, `PermissionDecision`, `ToolServer`, `ClaudeTool`, `ToolOutput`, `Message`, `ContentBlock`, `ResultMessage`, `Turn`, `ClaudeError`) | A client for the installed Claude Code CLI over its stream-json control protocol, turns, resume (also from an earlier message, `SessionStart::Resume { at }`, and with a changed system prompt, `ClaudeOptions::system_prompt_snapshot`, a fork under a new ID, `SessionStart::Fork`), approvals, hooks, in-process MCP tools, settings and setting sources. The CLI runs in its own process group (Windows, job object), killed with the programs it started on `close` or drop. | `arko_ai/src/providers/claude_code/` | [Claude Code](arko_ai-guide.md#claude-code) |
| `Model::openai_gpt_6_astra`, `openai_gpt_6_1_sol`, `openai_gpt_6_sol`, `openai_gpt_6_luna`, `openai_gpt_5_6_sol`, `openai_gpt_5_6_terra`, `openai_gpt_5_6_luna`, `openai_gpt_5_6` | Registered models `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, and `gpt-5.6` (an alias of GPT-5.6 Sol with its prices and limits). | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::openai_gpt_5_5`, `openai_gpt_5_5_pro`, `openai_gpt_5_4`, `openai_gpt_5_4_pro`, `openai_gpt_5_4_mini`, `openai_gpt_5_4_nano`, `openai_gpt_5_3_codex`, `openai_gpt_5_2`, `openai_gpt_5_2_pro`, `openai_gpt_5_1`, `openai_gpt_5`, `openai_gpt_5_pro`, `openai_gpt_5_mini`, `openai_gpt_5_nano` | Registered models `gpt-5.5-2026-04-23`, `gpt-5.5-pro-2026-04-23`, `gpt-5.4-2026-03-05`, `gpt-5.4-pro-2026-03-05`, `gpt-5.4-mini-2026-03-17`, `gpt-5.4-nano-2026-03-17`, `gpt-5.3-codex`, `gpt-5.2`, `gpt-5.2-pro`, `gpt-5.1-2025-11-13`, `gpt-5-2025-08-07`, `gpt-5-pro-2025-10-06`, `gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07`. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::openai_o4_mini`, `openai_o3`, `openai_o3_pro`, `openai_gpt_4_1`, `openai_gpt_4_1_mini`, `openai_gpt_4_1_nano` | Registered models `o4-mini-2025-04-16`, `o3-2025-04-16`, `o3-pro-2025-06-10`, `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`, `gpt-4.1-nano-2025-04-14` (the three `gpt-4.1` models are the only `Standard` OpenAI models). | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::anthropic_claude_fable_5_1`, `anthropic_claude_mythos_5_1`, `anthropic_claude_fable_5`, `anthropic_claude_opus_5_5`, `anthropic_claude_opus_5`, `anthropic_claude_opus_4_8`, `anthropic_claude_sonnet_5_5`, `anthropic_claude_sonnet_5`, `anthropic_claude_haiku_5_5`, `anthropic_claude_opus_4_7`, `anthropic_claude_opus_4_6`, `anthropic_claude_sonnet_4_6` | Registered models `claude-fable-5-1`, `claude-mythos-5-1`, `claude-fable-5`, `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-sonnet-5-5`, `claude-sonnet-5`, `claude-haiku-5-5`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6` (the "Claude 4.6 and later" set). Mythos 5.1 is Fable 5.1 for organizations verified through Anthropic's verification programs, with the same rules and prices. Haiku 5.5 prices a prompt over 100,000 tokens higher. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::anthropic_claude_opus_4_5`, `anthropic_claude_opus_4_1`, `anthropic_claude_opus_4`, `anthropic_claude_sonnet_4_5`, `anthropic_claude_sonnet_4`, `anthropic_claude_haiku_4_5`, `anthropic_claude_3_7_sonnet`, `anthropic_claude_3_5_sonnet_20241022`, `anthropic_claude_3_5_sonnet_20240620` | Registered models `claude-opus-4-5-20251101`, `claude-opus-4-1-20250805`, `claude-opus-4-20250514`, `claude-sonnet-4-5-20250929`, `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001`, `claude-3-7-sonnet-20250219`, `claude-3-5-sonnet-20241022`, `claude-3-5-sonnet-20240620`. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::google_gemini_3_8_flash`, `google_gemini_3_7_flash`, `google_gemini_3_6_flash`, `google_gemini_3_5_flash`, `google_gemini_3_5_flash_lite`, `google_gemini_3_1_pro`, `google_gemini_3_1_pro_customtools`, `google_gemini_3_1_flash_lite_stable`, `google_gemini_3_1_flash_lite`, `google_gemini_3_0_pro`, `google_gemini_3_0_flash` | Registered models `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools`, `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-3-pro-preview`, `gemini-3-flash-preview`. | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::google_gemini_2_5_pro`, `google_gemini_2_5_flash`, `google_gemini_2_5_flash_lite`, `google_gemini_2_0_flash` | Registered models `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash` (the only Gemini model without thinking or caching). | `arko_ai/src/model.rs` | [Models](arko_ai-guide.md#models) |
| `Model::supports_forced_tool_choice`, `supports_disabling_parallel_tool_calls`, `reports_stop_reason`, `accepts_tool_result_images` | Whether a forced tool choice is accepted with a given config. Whether `parallel_tool_calls: Some(false)` is (false only for Gemini). Always true. Whether images go inside tool results (OpenAI, Anthropic, Gemini 3 and later, with image input). | `arko_ai/src/model.rs` | [Tool calls (arko_ai)](#tool-calls-arko_ai) |
| `Model::image_refusal` | Why an image can't go to the model in a user message (no image input, a URL for Anthropic, Gemini or Claude Code, a GIF for Gemini), or `None`. |
| `Model::image_limits`, `Model::sendable_image`, `Model::check_image`, `image_fit::{ImageLimits, PatchLimit, fit_image, MAX_DECODE_BYTES}` | A provider's image limits (bytes per image, pixels a side, OpenAI's 30,000 squares of 32 px, images and image bytes per request, formats, animations), and an image fitted to them, unchanged, or turned upright, converted (BMP, TIFF, an animation's first frame) and scaled down and re-encoded (PNG kept for lossless images when small enough, else JPEG), or the reason it can't go (a format that can't be read here such as HEIC, an image too large to decode, can't shrink enough, or an `image_refusal` reason). Decoding takes at most 256 MB and runs off Tokio's workers. `check_image` judges from the header alone. | `arko_ai/src/image_fit.rs` | [Tool images and Gemini 3.x (arko_ai)](#tool-images-and-gemini-3x-arko_ai) |
| `Model::is_openai_gpt_6_astra`, `is_openai_gpt_6_sol_or_luna`, `is_openai_gpt_5_6`, `is_openai_gpt_5_5_pro`, `is_openai_gpt_5_2_pro`, `is_openai_gpt_5_pro`, `supports_openai_prompt_cache_options` | OpenAI model-id tests. `supports_openai_prompt_cache_options` is true for GPT-5.6 and GPT-6. | `arko_ai/src/model.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `Model::is_fable_5_1`, `is_mythos_5_1`, `is_fable_or_mythos_5_1`, `is_fable_5`, `is_opus_5_5`, `is_opus_5`, `is_opus_4_8`, `is_opus_4_7`, `is_opus_4_6`, `is_opus_4_5`, `is_sonnet_5_5`, `is_sonnet_5`, `is_sonnet_4_6`, `is_haiku_5_5` | Claude model-id tests (exact ids). `is_fable_or_mythos_5_1` matches either id, since every Fable 5.1 rule applies to both. | `arko_ai/src/model.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Model::supports_anthropic_adaptive_thinking`, `defaults_anthropic_adaptive_thinking`, `always_uses_anthropic_adaptive_thinking`, `rejects_anthropic_manual_thinking`, `omits_anthropic_sampling_parameters`, `supports_anthropic_summarized_thinking_display`, `supports_anthropic_effort`, `supports_anthropic_xhigh_effort`, `supports_anthropic_max_effort`, `rejects_anthropic_forced_tool_choice`, `supports_anthropic_fast_mode`, `supports_anthropic_inference_geo`, `has_standard_anthropic_1m_context`, `had_retired_anthropic_long_context_beta`, `supports_long_context`, `supports_interleaved_thinking`, `auto_enables_interleaved_thinking` | The Claude capability rules the Anthropic provider (and, for forced tool choice, OpenRouter) checks before sending. | `arko_ai/src/model.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Model::is_gemini_3_pro`, `is_gemini_3_flash`, `is_gemini_3_1_pro`, `is_gemini_3_1_flash_lite`, `is_gemini_3_5_flash`, `is_gemini_3_5_flash_lite`, `is_gemini_3_6_flash`, `is_gemini_3_7_flash`, `is_gemini_3_8_flash`, `is_gemini_2_5_pro`, `is_gemini_2_5_flash`, `is_gemini_2_5_flash_lite`, `is_gemini_2_5`, `uses_gemini_thinking_levels` | Gemini model-id tests. `uses_gemini_thinking_levels` is true for the Gemini 3.x models. | `arko_ai/src/model.rs` | [Reasoning (arko_ai)](#reasoning-arko_ai) |
| `CHATGPT_OAUTH_MAX_INPUT_TOKENS`, `effective_max_input_tokens` | 272,000, the ChatGPT OAuth backend's input limit. A model's input limit on an auth path (`None` when unknown). Requests aren't checked against it. | `arko_ai/src/input_limit.rs` | [ChatGPT OAuth (arko_ai)](#chatgpt-oauth-arko_ai) |
| `ModelPricing` (`input_cost_per_million`, `output_cost_per_million`, `cached_input_cost_per_million`, `cache_write_cost_per_million`, `large_input_threshold`, `large_input_cost_per_million`, `large_output_cost_per_million`, `large_cached_input_cost_per_million`, `large_cache_write_cost_per_million`, `reasoning_output_cost_per_million`, `cache_write_1h_cost_per_million`, `large_cache_write_1h_cost_per_million`, `service_tiers`, `residency`, `schedule`), `impl Default` | List prices in USD per million tokens, long-context prices, tier and residency multipliers, and dated price changes. A token class without its own price is billed at the input rate. `reasoning_output_cost_per_million` is unused. | `arko_ai/src/model.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `ModelPricing::at`, `tier_multiplier`, `offers_tier`, `residency_multiplier` | The prices in effect at a time. A tier's multiplier (1.0 for unpriced tiers). Whether a tier can be requested. A region's multiplier. | `arko_ai/src/pricing.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `ServiceTier` (`Standard`, `Flex`, `Fast`, `Priority`, `Scale`, `Batch`, `Other`), `ServiceTier::from_openai`, `ServiceTier::from_anthropic` | The processing tier used for pricing, normalized from OpenAI `service_tier` and Anthropic `service_tier`/`speed`. | `arko_ai/src/pricing.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `TierPricing` (`tier`, `multiplier`, `applies_to_long_context`), `::new`, `::flex`, `::fast`, `short_context_only` | A tier's price multiplier. `applies_to_long_context` defaults to `true`. | `arko_ai/src/pricing.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `ResidencyPricing` (`region`, `multiplier`), `::new`. `ScheduledPricing` (`effective_from`, `pricing`), `::new`, `effective_from_unix` | A multiplier for a reported region (Anthropic `inference_geo`). Prices from a UTC date (`YYYY-MM-DD`). | `arko_ai/src/pricing.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |

**Prompt caching**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `PromptCacheLifetime` (`Known(Duration)`, `Unknown`, `Disabled`) | How long a provider keeps a request's prompt cache after its last use. | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `prompt_cache::prompt_cache_lifetime` | The lifetime for a model, config and auth path. Reads `GenerationConfig::prompt_caching` and `prompt_cache_lifetime_seconds`, not `Conversation::enable_cache`. | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `prompt_cache::ANTHROPIC_PROMPT_CACHE_LIFETIME`, `ANTHROPIC_EXTENDED_PROMPT_CACHE_LIFETIME`, `OPENAI_PROMPT_CACHE_LIFETIME`, `CHATGPT_OAUTH_PROMPT_CACHE_LIFETIME` | 5 minutes, 1 hour, 5 minutes (a conservative value for OpenAI with an API key), and 30 minutes (OpenAI over ChatGPT OAuth). | `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `GenerationConfig::prompt_cache_lifetime_seconds`, `GenerationConfig::with_prompt_cache_lifetime` | The host's prompt-cache lifetime, reported in place of the provider's. | `arko_ai/src/config.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |
| `AnthropicCacheTtl` (`FiveMinutes` = `"5m"`, `OneHour` = `"1h"`), `AnthropicCacheTtl::duration` | Claude cache lifetimes and their durations. | `arko_ai/src/config.rs`, `arko_ai/src/prompt_cache.rs` | [Prompt caching (arko_ai)](#prompt-caching-arko_ai) |

**Usage and cost**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `UsageInfo` (`input_tokens`, `output_tokens`, `input_cost`, `output_cost`, `cached_tokens`, `cache_read_cost`, `cache_write_tokens`, `cache_write_1h_tokens`, `cache_write_cost`, `reasoning_tokens`, `service_tier`, `inference_geo`, `total_cost`, `reported_cost`, `upstream_cost`, `billing`, `billed_cost`), `impl Default` | Tokens and cost of a call or a sum of calls. `input_tokens` includes cache reads and writes. `output_tokens` includes reasoning tokens. `total_cost` is the list-price estimate. | `arko_ai/src/usage.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `UsageInfo::billed` | What the account was charged, with `billed_cost`, or `total_cost` for records saved before billing kinds existed. | `arko_ai/src/usage.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `UsageInfo::new`, `with_cached_tokens`, `with_cache_write_tokens`, `calculate_cost`, `calculate_cost_at`, `with_reported_cost`, `with_billing`, `for_embedding` | Constructors and cost computation, with `calculate_cost_at` prices the tokens at the list prices in effect at a time, for the tier and residency that applied, and sets `billed_cost` from `billing`. | `arko_ai/src/usage.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `impl AddAssign for UsageInfo`, `impl Sum for UsageInfo` | Adds calls, with saturating counts. Tier and region become unknown and `billing` `Mixed` when the calls differ. `billed_cost` is the sum of `billed()`. | `arko_ai/src/usage.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `BillingKind` (`Metered` (default), `Subscription`, `ProviderReported`, `NotCharged`, `Mixed`) | How a call is paid for, which decides `billed_cost`, the estimate, zero (ChatGPT OAuth, Anthropic classifier refusals in unbilled categories), the provider's reported charge, or a sum. | `arko_ai/src/usage.rs` | [Usage and cost](arko_ai-guide.md#usage-and-cost) |
| `ModelCallUsage` (`call_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `max_input_tokens`, `context_usage_percent`, `input_delta_from_previous_call`, `stop_reason`, `stop_details`, `requested_model`, `upstream_provider`, `served_model`, `rerouted`, `plan_usage`, `account`, `subagent`) | Per-call telemetry. The serving model, token counts, the context share of the model's input limit, the raw stop reason, reroute information, plan usage, the label of the ChatGPT account that served the call (with several), and the subagent a call was made for (Claude Code's `Agent` tool). | `arko_ai/src/usage.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `ModelCallUsage::new`, `with_plan_usage`, `served_model_changed`, `with_served_model`, `with_input_delta`, `with_stop_metadata` | Constructors. `served_model_changed` is true when `requested_model` is set (OpenRouter) or `rerouted` (OpenAI). | `arko_ai/src/usage.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `PlanUsage` (`plan`, `plan_type`, `active_limit`, `primary`, `secondary`, `credits`, `extra_usage`), `PlanUsage::from_codex_headers`, `most_used_window`. `PlanUsageWindow` (`used_percent`, `window_minutes`, `resets_at`, `reset_after_seconds`). `PlanCredits` (`has_credits`, `unlimited`, `balance`, `usable`). `ExtraUsage` (`enabled`, `in_use`, `status`, `disabled_reason`, `resets_at`, `monthly_limit`, `used_credits`, `utilization`, `currency`, `decimal_places`), a Claude account's extra usage (usage credits), with the flags from the `rate_limit_event` and the amounts, in minor units, from Claude Code's usage data. `SubscriptionPlan::{ChatGpt, ClaudeCode}` (`ALL`, `key`, `label`) | ChatGPT plan usage parsed from the `x-codex-*` response headers (Claude Code's reports use the same type). The plan a report is of (`chatgpt`, `claude_code`, `None` in records saved before it existed). The window with the highest used percentage. | `arko_ai/src/usage.rs` | [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth) |
| `ContextEvent` (`call_id`, `event_type`, `char_count`, `byte_count`, `token_count`), `ContextEvent::new`, `for_tool_call`, `for_tool_result`, `for_text`, `with_token_count`. `ContextEventType` (`System`, `UserMessage`, `AssistantMessage`, `AssistantToolCall`, `ToolResult`, `ToolDefinition`, `FileAttachment`) | A per-call record of context sizes. No arko crate creates or reads it. | `arko_ai/src/usage.rs` | — |

**Errors and retries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Error::ProviderError { message, status, code, retry_after }` | A provider's error response or error event, with its HTTP status (or a status its code stands for), code and delay hint. | `arko_ai/src/error.rs` | [Errors and retries](arko_ai-guide.md#errors-and-retries) |
| `Error::HttpError`, `Error::JsonError`, `Error::IoError` | Transport failures (connection, timeout, body). JSON failures. File I/O failures (Codex `auth.json`). | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `Error::ApiError` | An unexpected provider response (no `choices`, a missing field, a malformed stream event). | `arko_ai/src/error.rs` | [Errors and retries](arko_ai-guide.md#errors-and-retries) |
| `Error::AuthenticationFailed`, `Error::MissingApiKey` | Credential problems (partial `OPENAI_OAUTH_*` variables, an expired OAuth access token without a refresh token, Codex `auth.json` problems, a spent refresh token). A provider that isn't configured. | `arko_ai/src/error.rs` | [Client and credentials (arko_ai)](#client-and-credentials-arko_ai) |
| `Error::InvalidRequest`, `Error::UnsupportedFeature` | A request rejected before sending. A value the model can't take. A feature the provider or model lacks (images without image input, image URLs on Anthropic and Gemini, mid-conversation updates outside OpenRouter, ...). | `arko_ai/src/error.rs` | [GenerationConfig (arko_ai)](#generationconfig-arko_ai) |
| `Error::ProviderNotSupported`, `Error::ModelNotSupported` | A provider client given another provider's conversation (only when calling a provider client directly). An unknown model id (`Model::get`, an embedding model, an OpenRouter slug missing from a loaded catalog). | `arko_ai/src/error.rs` | [Models](arko_ai-guide.md#models) |
| `Error::Refusal` | A refusal raised as an error (Gemini blocked generation, OpenRouter refusal errors, `single_tool_call` on a refused response). Classed `Permanent`. | `arko_ai/src/error.rs` | [Stop reasons and refusals](arko_ai-guide.md#stop-reasons-and-refusals) |
| `Error::StreamTruncated` | A stream that ended before the provider reported completion, or was cut off by the stream idle timeout. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `Error::UnexpectedToolCalls { expected, stop_reason, found }`, `Error::InvalidToolArguments { name, raw }` | `GenerationResponse::single_tool_call` found no call, several, or another tool. Or the call's arguments aren't valid JSON. | `arko_ai/src/error.rs` | [Generation settings and tool calls](arko_ai-guide.md#generation-settings-and-tool-calls) |
| `Error::TokenizerError`, `TokenLimitExceeded`, `MissingHuggingFaceToken`, `BatchSizeLimitExceeded`, `InvalidDimension` | Embedding failures before sending, tokenizer loading or counting. An input or request over the token limit. A tokenizer that needs a token (no registered model does). More inputs than allowed. An unsupported dimension. | `arko_ai/src/error.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `Error::Base64Error` | A `From<base64::DecodeError>` conversion. arko_ai's own code doesn't produce it. | `arko_ai/src/error.rs` | — |
| `Error::retry_class`, `is_retryable`, `is_capacity_error`, `is_usage_limit`, `is_rate_limit`, `retry_after` | How a failure may be recovered. Views of that class. The provider's delay hint. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `RetryClass` (`Permanent`, `Transient`, `Capacity`, `UsageLimit`), `RetryClass::is_retryable` | Recovery classes. Every class but `Permanent` is retryable. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `ErrorKind` (`Authentication`, `Permission`, `Billing`, `ContextOverflow`, `RequestTooLarge`, `InvalidRequest`, `NotFound`, `Refusal`, `Unsupported`, `Unknown`), `ErrorKind::as_str` | What kind of failure an error is, beside its retry class, which it doesn't change, a lost credential, a credit wall, a request over the model's input limit, and so on. Saved in snake_case (`as_str`). A name a later version adds loads as `Unknown`. `#[non_exhaustive]`. | `arko_ai/src/error.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `Error::kind` | The error's kind, from its code, then its HTTP status, then documented phrases of its message. Temporary failures and anything unrecognised are `Unknown`. | `arko_ai/src/error.rs` | [Error kinds](arko_ai-guide.md#error-kinds) |
| `ErrorKind::from_claude_code(error, api_error_status, terminal_reason, text)` | The kind of a failed Claude Code CLI turn, from the assistant message's `error`, the result's `api_error_status` and `terminal_reason`, and its text. arko_agents' Claude Code runtime records it, and pauses `NonRetryable` on `Authentication`, `Billing` and `Permission` (retrying doesn't fix them). | `arko_ai/src/error.rs` | [Error kinds](arko_ai-guide.md#error-kinds) |
| `BackoffPolicy` (`initial`, `max`, `jitter`), `BackoffPolicy::new`, `delay`, `next_delay`. `retry::random_sample` | Exponential backoff shared by the other crates, with `new` turns jitter on. A server hint is a minimum. `random_sample` is the random input of `delay`. | `arko_ai/src/retry.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |
| `retry::MAX_SERVER_DELAY_HINT` | 30 days, the cap on every provider delay hint. | `arko_ai/src/retry.rs` | [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai) |

**Embeddings**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `EmbeddingClient`, `EmbeddingClient::new`, `impl Default` | The embeddings client (Voyage only). `new` configures no key. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingClient::with_voyage_api_key`, `from_env`, `with_voyage_client` | Configures Voyage with a key, from `VOYAGE_API_KEY` (empty counts as unset), or with a preconfigured `VoyageClient`. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingClient::embed` | Embeds texts after local checks (dimension, token counts, input count). Vectors come back in input order. `Error::MissingApiKey` without a Voyage key. | `arko_ai/src/embedding_client.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingClient::token_counter`, `count_tokens`, `get_embedding_cost`, `get_embedding_cost_for_token_count` | A reusable `TokenCounter` for a config. The token count of one text. The list-price cost of texts or of a token count (no request is made). | `arko_ai/src/embedding_client.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `EmbeddingConfig` (`provider`, `model_id`, `output_dimensionality`, `voyage_input_type`, `voyage_truncation`, `hf_token`, `tokenizer_revision`), `EmbeddingConfig::new` | An embedding request's model and options. `new` sets every option to `None`. `Debug` output redacts `hf_token`. | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingConfig::with_output_dimensionality`, `with_voyage_input_type`, `with_voyage_truncation`, `with_tokenizer_revision`, `with_hf_token` | Setters of the options. | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `VoyageInputType` (`Query`, `Document`), `VoyageInputType::as_str` | Voyage `input_type` (`"query"`, `"document"`). | `arko_ai/src/embedding_config.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `EmbeddingModelDetails` (`provider`, `id`, `api_id`, `max_input_tokens_per_item`, `max_batch_total_tokens`, `max_batch_size`, `tokenizer_id`, `requires_hf_token`, `cost_per_million_input_tokens`, `default_dimensions`, `supported_dimensions`), `EmbeddingModelDetails::get` | An embedding model's limits, tokenizer and price. `get` looks one up (`Error::ModelNotSupported` otherwise). | `arko_ai/src/embedding_model.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `EmbeddingModelDetails::voyage_4_large`, `voyage_4`, `voyage_4_lite`, `voyage_code_4`, `voyage_3_5`, `voyage_3_large`, `voyage_3_5_lite`, `voyage_code_3` | Registered models `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3.5`, `voyage-3-large`, `voyage-3.5-lite`, `voyage-code-3`, 32,000 tokens per input, 1,000 inputs per request, 1,024 dimensions by default (256, 512, 1,024, 2,048 supported). | `arko_ai/src/embedding_model.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `EmbeddingProviderIdentifier` (`Voyage`) | The embeddings provider (Voyage is the only one). | `arko_ai/src/embedding_provider_identifier.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `providers::EmbeddingResponse` (`embeddings`, `usage`) | The vectors, in input order, and their `UsageInfo`. | `arko_ai/src/providers/mod.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `providers::EmbeddingProviderTrait` (`embed_texts`), `impl EmbeddingProviderTrait for VoyageClient` | The trait `VoyageClient` implements. `EmbeddingClient` holds only a `VoyageClient`, so another implementation can't be plugged into it. | `arko_ai/src/providers/mod.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `providers::voyage::VoyageClient`, `::new`, `::with_base_url` | The Voyage client. Requests go to `{base_url}/embeddings` (default `https://api.voyageai.com/v1`). | `arko_ai/src/providers/voyage.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |
| `TokenCounter`, `TokenCounter::count` | Counts a text's tokens with a loaded tokenizer (no padding, truncation or special tokens). | `arko_ai/src/token_utils.rs` | [Voyage embeddings](arko_ai-guide.md#voyage-embeddings) |
| `count_tokens`, `token_utils::count_tokens_with_revision`, `token_utils::count_tokens_for_text` | Counts texts' tokens with a Hugging Face tokenizer repository (at `main`, or at a revision), loading and caching it per repository and revision. | `arko_ai/src/token_utils.rs` | [Embeddings (arko_ai)](#embeddings-arko_ai) |

**OpenRouter**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `OpenRouterClient`, `OpenRouterClient::new`, `providers::openrouter::OPENROUTER_BASE_URL` | The OpenRouter client (clones share the loaded catalog). Chat requests go to `{base_url}/chat/completions`, default `https://openrouter.ai/api/v1`. | `arko_ai/src/providers/openrouter.rs` | [OpenRouter](arko_ai-guide.md#openrouter) |
| `OpenRouterClient::with_base_url`, `with_base_url_for`, `with_stream_idle_timeout` | Another base URL. A base URL per model slug for chat requests (`None` keeps the client's). Sets or clears (`None`) the stream idle timeout. | `arko_ai/src/providers/openrouter.rs` | [OpenRouter](arko_ai-guide.md#openrouter) |
| `OpenRouterClient::with_default_options`, `default_options` | Options applied to every request, which `GenerationConfig::openrouter` overrides field by field. | `arko_ai/src/providers/openrouter.rs` | [Options](arko_ai-guide.md#options) |
| `OpenRouterClient::with_app_attribution`. `AppAttribution` (`url`, `title`, `categories`, `hidden`), `AppAttribution::new`, `with_title`, `with_categories`, `hidden` | Opt-in app attribution headers, with `HTTP-Referer`, `X-OpenRouter-Title`, `X-OpenRouter-Categories`, `X-OpenRouter-App-Visibility: hidden`. | `arko_ai/src/providers/openrouter.rs`, `openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `OpenRouterClient::fetch_models`, `fetch_user_models`, `set_catalog`, `catalog` | Fetches the public catalog (`GET /models`, no key sent) or the key's catalog (`GET /models/user`) and keeps it. Installs or clears a catalog. The loaded catalog. | `arko_ai/src/providers/openrouter.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `OpenRouterClient::model`, `OpenRouterClient::resolve_model` | The catalog's model for a slug. The catalog's model, or `Model::openrouter(slug)` without a catalog (`Error::ModelNotSupported` for a slug the loaded catalog lacks, slugs starting with `@` skip the check). | `arko_ai/src/providers/openrouter.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `OpenRouterCatalog`, `OpenRouterCatalog::from_json`, `from_models`, `models`, `source`, `fetched_at`, `get`, `contains`. `CatalogSource` (`Public`, `User`, `Supplied`) | A parsed model catalog, from a saved `/models` response or from models, their list, where it came from and when. Lookup by slug (variants and `model@preset/...` resolve to the base model, which keeps the requested slug). | `arko_ai/src/providers/openrouter/catalog.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `OpenRouterModelInfo` (`name`, `canonical_slug`, `alias_target`, `context_length`, `max_completion_tokens`, `input_modalities`, `output_modalities`, `supported_parameters`, `reasoning`, `pricing_known`, `expiration_date`), `supports_parameter`, `accepts_images` | Catalog metadata of a model. An empty `supported_parameters` list counts as accepting every parameter. | `arko_ai/src/providers/openrouter/catalog.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `providers::openrouter::OpenRouterReasoningInfo` (`effort_selectable`, `supported_efforts`, `default_effort`, `default_enabled`, `mandatory`, `supports_max_tokens`) | A catalog model's reasoning controls. | `arko_ai/src/providers/openrouter/catalog.rs` | [Models and the catalog](arko_ai-guide.md#models-and-the-catalog) |
| `OpenRouterOptions` (`provider`, `fallback_models`, `auto_router`, `auto_exacto`, `user`, `service_tier`), `OpenRouterOptions::new`, `with_provider`, `with_fallback_models`, `with_auto_router`, `with_auto_exacto`, `with_user`, `with_service_tier`, `merged_over` | Routing and request options. Unset fields aren't sent. `merged_over` lays request options over defaults, `provider` and `auto_router` merge field by field, every other field replaces the default whole. | `arko_ai/src/providers/openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `providers::openrouter::OpenRouterServiceTier` (`Auto`, `Default`, `Flex`, `Priority`, `Scale`) | OpenRouter `service_tier` values. | `arko_ai/src/providers/openrouter/options.rs` | [Options](arko_ai-guide.md#options) |
| `AutoExactoPolicy` (`CacheAware` (default), `AllowAutoExacto`, `DisableAutoExacto`) | What arko does about Auto Exacto on tool requests. | `arko_ai/src/providers/openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |
| `ProviderPreferences` (`order`, `only`, `ignore`, `allow_fallbacks`, `require_parameters`, `data_collection`, `zdr`, `enforce_distillable_text`, `quantizations`, `sort`, `max_price`, `preferred_min_throughput`, `preferred_max_latency`, `options`) | OpenRouter's `provider` routing object. Unset fields are omitted. | `arko_ai/src/providers/openrouter/options.rs` | [Options](arko_ai-guide.md#options) |
| `ProviderPreferences::new`, `pinned`, `zdr_only`, `with_order`, `with_only`, `with_ignore`, `with_allow_fallbacks`, `with_require_parameters`, `with_data_collection`, `with_zdr`, `with_enforce_distillable_text`, `with_quantizations`, `with_sort`, `with_max_price`, `with_preferred_min_throughput`, `with_preferred_max_latency`, `with_provider_options`, `merged_over`, `is_empty` | Builders. `pinned(slug)` sets `only` plus `allow_fallbacks: false`. `zdr_only()` sets `zdr: true`. `merged_over` takes each set field over the base. | `arko_ai/src/providers/openrouter/options.rs` | [Options](arko_ai-guide.md#options) |
| `DataCollection` (`Allow`, `Deny`), `providers::openrouter::Quantization` (`Int4`, `Int8`, `Fp4`, `Mxfp4`, `Nvfp4`, `Fp6`, `Fp8`, `Mxfp8`, `Fp16`, `Bf16`, `Fp32`, `Unknown`), `SortBy` (`Price`, `Throughput`, `Latency`, `Exacto`), `providers::openrouter::SortPartition` (`Model`, `None`), `providers::openrouter::ProviderSort` (`By`, `Config { by, partition }`), `ProviderSort::by`, `From<SortBy> for ProviderSort` | Values of the `provider` fields. | `arko_ai/src/providers/openrouter/options.rs` | [Options](arko_ai-guide.md#options) |
| `providers::openrouter::MaxPrice` (`prompt`, `completion`, `request`, `image`, `audio`), `MaxPrice::per_million_tokens`. `providers::openrouter::PerformanceThreshold` (`P50`, `Percentiles { p50, p75, p90, p99 }`) | `provider.max_price` (sent as JSON numbers) and throughput or latency thresholds. | `arko_ai/src/providers/openrouter/options.rs` | [Options](arko_ai-guide.md#options) |
| `AutoRouterOptions` (`allowed_models`, `excluded_models`, `cost_tier`, `provider`), `AutoRouterOptions::new`, `zdr`, `with_allowed_models`, `with_excluded_models`, `with_cost_tier`, `with_provider`, `merged_over`. `AutoRouterCostTier` (`Low`, `Medium`, `High`, `XHigh`, `Max`) | Auto Router settings, used only for `openrouter/auto` and `openrouter/auto-beta` and ignored for other models. | `arko_ai/src/providers/openrouter/options.rs` | [OpenRouter (arko_ai)](#openrouter-arko_ai) |

**Example binaries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `llm_client_cli` `text`, `chat`, `models` | One-shot generation (optionally streamed, with an image), an interactive streamed chat, and the list of registered models with the aliases the CLI accepts. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `-m`, `--model`. `-s`, `--system`. `-t`, `--temperature`. `-n`, `--max-tokens`. `--long-context` | The model (registered id, alias, or OpenRouter slug) and generation settings. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `-r`, `--reasoning`. `--reasoning-level`. `--reasoning-effort`. `--reasoning-mode` | Reasoning at the model's defaults, or an explicit effort, level, budget or mode. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `text -p`, `--prompt`. `text -i`, `--image PATH`. `text --stream` | The prompt, an image file, and streaming for `text`. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `llm_client_cli` `--codex-auth`, `--codex-auth-json PATH` | OpenAI ChatGPT OAuth credentials from a Codex `auth.json` instead of the environment. | `arko_ai/examples/llm_client_cli.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `embeddings_demo` | Embeds two texts with `voyage-4-lite` at 256 dimensions. Takes no flags. | `arko_ai/examples/embeddings_demo.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `oauth_cache_probe` `--out`, `--auth-file`, `--conversations`, `--turns`, `--idle-gap`, `--model`, `--effort`, `--label`, `--summarize-only` | Measures ChatGPT OAuth prompt-cache hits with a traced tool-loop workload. | `arko_ai/examples/oauth_cache_probe.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |
| `oauth_cache_probe` `--legacy`, `--session-id-header`, `--thread-id-header`, `--client-request-id-header`, `--echo-turn-state`, `--no-legacy-beta-header`, `--replay-output-items`, `--reasoning-context` | Request-shape experiments. The provider as it was before its Codex-parity changes, with single pieces turned back on. | `arko_ai/examples/oauth_cache_probe.rs` | [Example binaries (arko_ai)](#example-binaries-arko_ai) |

### Client and credentials (arko_ai)

| Setting | Default | Notes |
|---|---|---|
| `LLMClient::new()` | No provider | A request to a provider without a client fails with `Error::MissingApiKey`. The client reads no environment variables unless `from_env` is called. |
| `LLMClient::from_env()` | — | Reads `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (and optionally `OPENAI_OAUTH_REFRESH_TOKEN`), otherwise `OPENAI_API_KEY`. `ANTHROPIC_API_KEY`. `GOOGLE_API_KEY`, otherwise `GEMINI_API_KEY`. `OPENROUTER_API_KEY`. Empty or blank values count as unset. An `OPENAI_OAUTH_*` variable set without both the access token and the account ID is `Error::AuthenticationFailed`, never a fallback to the API key. `from_env_with(lookup)` reads through a function instead. |
| Refresh token from the environment | Kept in memory | If it matches the refresh token in the Codex `auth.json` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`) for the same account (compared by SHA-256), the client uses that file as its credential store instead, as `with_codex_auth_file` does. Otherwise the first rotation logs a warning (once per process), since the environment's token stops working once used. |
| `with_openai_chatgpt_oauth(access, refresh, account)` | In memory only | Rotated tokens are not saved and no warning is logged. Save them from `openai_chatgpt_oauth_credentials()`. `with_openai_auth(OpenAIAuth::ChatGptOAuth(..))` behaves the same. |
| `with_codex_auth_file(&file)`, `CodexAuthFile::client()` | — | Every refresh re-reads `auth.json` first (adopting tokens another program refreshed, for the same account only), and rotated tokens are saved at once. See [ChatGPT OAuth credentials](arko_ai-guide.md#chatgpt-oauth-credentials). |
| `with_chatgpt_accounts(ChatGptAccounts)` | Off (one account) | Opt-in failover between ChatGPT accounts, first preferred. An account is set aside only when the backend refuses it (`usage_limit_reached`, or an `ErrorKind::Billing` code such as `credit_balance_exhausted`). Credits keep it serving. A refused request goes to the next account at once. With every account set aside, the request fails with `usage_limit_reached` and the earliest known reset. Accounts above the serving one are checked on the usage endpoint at most every 5 minutes (`with_status_check_interval`), as requests are made. Nothing runs in the background. Each request replays only the serving account's encrypted reasoning and turn-state token. See [Several ChatGPT accounts](arko_ai-guide.md#several-chatgpt-accounts). |
| Credentials without a refresh token | — | The access token is used until the server rejects it as expired (HTTP 401 `token_expired`), which then fails with `Error::AuthenticationFailed`. Any other 401 is `Error::ProviderError` with status 401. `refresh_openai_oauth_token_if_expiring` returns `None` for them. |
| Token-refresh rate-limit retries | Off | `with_openai_chatgpt_oauth_rate_limit_retries(.., true)` or `OpenAIClient::with_oauth_rate_limit_retries(true)`. An explicit `refresh_openai_oauth_token()` retries an HTTP 429 up to 8 times, with jittered waits doubling from 30 s to 15 min. A server hint, capped at 15 min, is a minimum. The refresh after a 401, `refresh_openai_oauth_token_once` and `refresh_openai_oauth_token_if_expiring` (which the arko_agents runtime uses) never wait, so the setting does nothing for a client that only runs agents. |
| `with_openai_originator`, `with_openai_user_agent` | Not sent | No `originator` or `User-Agent` header by default. The shared HTTP client sets no `User-Agent`. Token-refresh requests never carry them. |
| `with_request_trace(RequestTrace::new(dir))` | Off | Writes every OpenAI model request and response (full prompts, credential headers redacted, `x-codex-turn-state` hashed) with default file permissions. Token refreshes and other providers are not traced. The first request to each other provider logs a warning. |
| `with_openai_request_hook(hook)` | None | `prepare` can change the headers (the `Authorization` header included) and body of each attempt before sending. `observe_response` gets every response's headers, never its body. |
| `OpenAIClient::with_credential_store(Some(store))`, `with_oauth_token_url(url)` | No store. `https://auth.openai.com/oauth/token` | A `ChatGptOAuthStore` keeps rotated tokens in step with other storage (`CodexAuthStore` for a Codex `auth.json`). `LLMClient::with_openai_auth(OpenAIAuth)` installs either kind of credentials, without a store. |
| `with_stream_idle_timeout(duration)` | Off | Applies to every provider client of the `LLMClient`, including clients installed later, over their own setting. It can't be cleared on the `LLMClient` again (the provider clients' setters take `None`). See [Retries and timeouts (arko_ai)](#retries-and-timeouts-arko_ai). |
| Provider clients | Base URLs, with `https://api.openai.com/v1` (API key), `https://chatgpt.com/backend-api/codex` (ChatGPT OAuth), `https://api.anthropic.com`, `https://generativelanguage.googleapis.com/v1`, `https://openrouter.ai/api/v1`, `https://api.voyageai.com/v1` | Requests go to `/responses` (OpenAI Responses API), `/v1/messages` (Anthropic, `anthropic-version: 2023-06-01`, no beta header by default), `/interactions` (Gemini Interactions API, `?alt=sse` when streaming), `/chat/completions` (OpenRouter Chat Completions API) and `/embeddings` (Voyage). Each client takes `with_base_url` (one base for both OpenAI auth kinds). Install them with `with_openai_client` and its siblings. |

The HTTP transport is fixed, a 30-second connect timeout, no total request deadline, no
redirects (a 3xx response is an error, so credentials never reach another host), TCP
keep-alive every 30 seconds, pooled connections closed after 90 seconds idle, HTTP/1.1,
rustls with the operating system's trust store, and the environment's and system's
proxy settings (`HTTPS_PROXY` and the like).

### GenerationConfig (arko_ai)

Every field defaults to `None`, and omitted fields take their defaults, so a partial JSON
or YAML config is valid. One config can serve models of several providers, a setting
the model's provider has no use for is dropped with a warning (logged once per process
for each provider and setting), while a value the provider can't accept for a setting it
does use is an error before anything is sent. Exceptions, `temperature` and
`parallel_tool_calls: Some(false)` on Gemini, and the Flex and Auto
`openai_service_tier` over ChatGPT OAuth, are `Error::UnsupportedFeature`. Gemini ignores
`ToolDefinition::strict` without a warning. Claude on OpenRouter accepts
`anthropic_summarized_thinking` and `long_context` without sending anything.

| Field | `None` means | Allowed values and notes |
|---|---|---|
| `temperature` | Not sent, the model's default | `Error::InvalidRequest` on OpenAI reasoning models (every registered OpenAI model except the `gpt-4.1` ones). `Error::UnsupportedFeature` on Gemini. Dropped with a warning (once per model) on Claude models that take no sampling parameters (Fable 5.x, Mythos 5.1, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5, Sonnet 5, Haiku 5.5). Other Claude models accept only 1.0 while thinking. Sent to OpenRouter as given. `GenerationConfig::new(t)` is deprecated for `default().with_temperature(t)`. |
| `max_tokens` | The model's own maximum, OpenAI with an API key, Anthropic and Gemini send the model's registered output limit. OpenRouter and ChatGPT OAuth send none | A set value is sent as given, not checked against the model's limit. Over ChatGPT OAuth it is never sent (the backend rejects it). A set value is ignored with a warning logged once. On Claude through OpenRouter it must exceed a `reasoning.max_tokens` budget. |
| `verbosity` | Not sent | `low`, `medium`, `high`. OpenAI GPT-5.x and GPT-6 models (by model id), on both auth paths. `Error::InvalidRequest` on other OpenAI models. Sent to OpenRouter, where on Claude it can't be combined with `reasoning.effort` (`Error::InvalidRequest`). Dropped with a warning on Anthropic and Gemini. |
| `reasoning` | The model's default. See [Reasoning (arko_ai)](#reasoning-arko_ai) | Each provider takes its own `ReasoningConfig` variants. Another provider's variant is dropped with a warning. |
| `openai_reasoning_mode` | Not sent | `standard`, `pro`, sent as `reasoning.mode` to GPT-5.6 and GPT-6 on API-key requests. ChatGPT OAuth omits it and warns once. `Error::InvalidRequest` on other OpenAI models. Dropped with a warning elsewhere (on OpenRouter, set `OpenRouterReasoning::mode`). |
| `anthropic_thinking_enabled` | The model's default | `Some(true)` without an Anthropic reasoning config sends adaptive thinking (`Error::InvalidRequest` on models before Claude 4.6). `Some(false)` sends `thinking.type: "disabled"`. It is `Error::InvalidRequest` on Fable 5.x, Mythos 5.1, Opus 5.5 and Sonnet 5.5 (arko does not send Sonnet 5.5's `between_tools`), together with an Anthropic reasoning config, and with `anthropic_summarized_thinking: Some(true)`. Opus 5 and Haiku 5.5 accept it at `high` effort or below, and without a reasoning config the default effort applies. Dropped with a warning on the other providers, OpenRouter included. |
| `prompt_caching` | `Conversation::enable_cache` (default `true`) | See [Prompt caching (arko_ai)](#prompt-caching-arko_ai). |
| `anthropic_summarized_thinking` | The model's default display | `Some(true)` sends `thinking.display: "summarized"` on models that omit thinking text by default (Fable 5.x, Mythos 5.1, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5, Sonnet 5, Haiku 5.5), nothing on others. `Error::InvalidRequest` without active thinking. On a model that thinks adaptively by default, `Some(true)` makes arko send the adaptive `thinking` object explicitly. `Some(false)` sends nothing. |
| `long_context` | Nothing sent | `Some(true)` is accepted on Claude 4.6 and later, where 1M context is standard and nothing extra is sent (prices don't depend on it, Haiku 5.5's higher rates for a prompt over 100,000 tokens apply either way). `Error::InvalidRequest` on other Claude models, Sonnet 4 and 4.5 included. `Some(false)` sends nothing. |
| `tool_config` | No tools | See [Tool calls (arko_ai)](#tool-calls-arko_ai). |
| `interleaved_thinking` | Automatic on Claude 4.6 and later while adaptive thinking is active | On the Anthropic provider, `Some(true)` needs a model that supports it and active thinking (not a manual budget on Opus 4.6). With a manual budget it sends the `interleaved-thinking-2025-05-14` beta header, with adaptive thinking nothing. `Some(false)` is `Error::InvalidRequest` while adaptive thinking is active on 4.6 and later, and sends nothing otherwise. For Claude on OpenRouter, `Some(true)` adds the beta header and `Some(false)` is `Error::UnsupportedFeature`. |
| `openai_service_tier` | Not sent, the project's default tier | `auto`, `default`, `flex`, `priority` (the `Fast` variant). With an API key, Flex and Fast only on models that offer them (`Error::InvalidRequest` otherwise). Over ChatGPT OAuth, as in Codex, Fast (`priority`, plus `tier=priority` in the `x-codex-routing-hint` header) on GPT-6 Astra, Sol, Luna and 6.1 Sol, GPT-5.6 Sol, Terra and Luna, and GPT-5.5 (`Error::InvalidRequest` on other models). `Default` sends no tier. Flex and Auto are `Error::UnsupportedFeature`. |
| `anthropic_cache_ttl` | No `ttl`, 5 minutes | `"5m"`, `"1h"` (1-hour cache writes cost 2x base input). `Error::InvalidRequest` while caching is off. Also used for Claude on OpenRouter. |
| `prompt_cache_lifetime_seconds` | `None`, the provider's lifetime | Never sent. Replaces the lifetime `prompt_cache_lifetime` reports, `Known` or `Unknown`. An uncached request stays `Disabled`, and `0` reports `Disabled`. Left out of saved configs when `None`. |
| `anthropic_fast_mode` | Nothing sent | `Some(true)` sends `speed: "fast"` with the `fast-mode-2026-02-01` beta header on Opus 5.5, Opus 5 and Opus 4.8 (`Error::InvalidRequest` on other models). Billed at fast-mode rates. |
| `anthropic_inference_geo` | Not sent, the workspace's default | `global`, `us` (1.1x price). Claude 4.6 and later (`Error::InvalidRequest` on older models). |
| `openrouter` | The client's default options | `OpenRouterOptions`, merged over `OpenRouterClient::with_default_options`. See [OpenRouter (arko_ai)](#openrouter-arko_ai). |

### Reasoning (arko_ai)

What is sent without `GenerationConfig::reasoning`:

| Provider | Sent |
|---|---|
| OpenAI, API key | No effort, so the model's default applies (none on GPT-5.1, 5.2 and 5.4, which then don't reason). GPT-5.6 reasoning models get `reasoning.context: "all_turns"`. GPT-6 gets no `reasoning` object. |
| OpenAI, ChatGPT OAuth | `effort: "medium"` on GPT-5.x and GPT-6 models. `context: "all_turns"` on GPT-5.6 and GPT-6. |
| Anthropic | Nothing, the model's default. Fable 5.x, Mythos 5.1, Opus 5.5, Opus 5, Sonnet 5.5, Sonnet 5 and Haiku 5.5 think adaptively by default (Fable 5.x, Mythos 5.1, Opus 5.5 and Sonnet 5.5 can't turn it off). Earlier models don't think. |
| Gemini | No `thinking_level` or `thinking_summaries`, the model's default level (medium on Gemini 3.5 to 3.8 Flash, minimal on 3.1 and 3.5 Flash-Lite, high on 3 Pro, 3 Flash and 3.1 Pro) and no thought summaries. |
| OpenRouter | Nothing, the model's default from OpenRouter. |

Every OpenAI request sets `store: false`, and reasoning models also ask for
`reasoning.encrypted_content`, which later requests replay (reasoning that nothing
follows is left out).

| `ReasoningConfig` | Values and checks |
|---|---|
| `openai(effort, include_summaries)` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, validated per model. An effort the model lacks, or any reasoning config on the `gpt-4.1` models, is `Error::InvalidRequest`. `include_summaries: true` sends `reasoning.summary: "auto"`. |
| `anthropic(budget_tokens)`, `anthropic_with_effort(budget, effort)` | Manual budget, at least 1,024 (0 is an error too) and below `max_tokens` unless `interleaved_thinking` is `Some(true)`. At most 52,000 on Opus 4.5 and Sonnet 4.6 and 102,400 on Opus 4.6. `Error::InvalidRequest` on Fable 5.x, Mythos 5.1, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5, Sonnet 5 and Haiku 5.5. The effort is sent as `output_config.effort`, `low` to `high` on Opus 4.5, `max` too on Opus 4.6 and Sonnet 4.6, all five on Fable 5.x, Mythos 5.1, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5.5, Sonnet 5 and Haiku 5.5, none on older models. |
| `anthropic_adaptive()`, `anthropic_adaptive_with_effort(effort)` | Claude 4.6 and later (`Error::InvalidRequest` on others). Efforts as above. |
| `gemini(level, include_summaries)` | `minimal`, `low`, `medium`, `high` (`"MINIMAL"` to `"HIGH"` in saved JSON, sent lowercase), validated per model with `Error::InvalidRequest`, 3 Pro rejects minimal and medium. 3.1 Pro, 3.7 Flash, 3.8 Flash and 2.5 reject minimal. 2.0 Flash rejects any level. `thinking_summaries` is `"auto"` with summaries and `"none"` without. |
| `google(budget, include_summaries)` | Mapped to a `thinking_level`, then validated as above (no budget is ever sent), negative = no level (the model's default), 0 = minimal (on 2.5 Pro and 2.5 Flash an error, on 2.5 Flash-Lite no level), up to 4,096 = low, up to 16,384 = medium, above = high. `thinking_summaries` is sent either way. |
| `openrouter(OpenRouterReasoning)`, `openrouter_effort(effort)` | The only variant OpenRouter takes. `effort` or `max_tokens`, not both (`Error::InvalidRequest`). `ReasoningConfig::to_openrouter()` converts the other variants explicitly. |

### Tool calls (arko_ai)

| Setting | Default | Notes |
|---|---|---|
| `ToolConfig::tool_choice` | `None`. Nothing sent to OpenAI, OpenRouter or Gemini, so the provider's `auto` applies. Anthropic sends `{"type": "auto"}` | `Auto`, `Required`, `None`, `Specific(name)`. A forced choice (`Required`, `Specific`) is `Error::InvalidRequest` before sending on Claude Fable 5.1, Mythos 5.1, Opus 5.5 and Sonnet 5.5 and with a manual thinking budget, on the Anthropic provider and on OpenRouter for `anthropic/` slugs of registered Claude models (where OpenRouter reasoning counts as a budget on Claude before 4.6, and `reasoning.max_tokens` on Opus 4.6 and Sonnet 4.6). With an OpenRouter catalog, also on models that don't list `tool_choice`. `Model::supports_forced_tool_choice(config)` tells in advance. |
| `ToolConfig::parallel_tool_calls` | `None`, the provider's default, several calls allowed | `Some(false)`, OpenAI and OpenRouter send `parallel_tool_calls: false`, Anthropic `disable_parallel_tool_use: true` inside `tool_choice` (not with `ToolChoice::None`). Gemini fails with `Error::UnsupportedFeature`. |
| `ToolConfig::tools` empty | — | A configuration without tools is sent as none, no tools, tool choice or parallel-call setting on any provider (`GenerationConfig::offered_tools`). |
| `ToolDefinition::strict` | `None`, not sent | Sent to OpenAI, Anthropic and OpenRouter (for Claude on OpenRouter with the `structured-outputs-2025-11-13` beta header). Gemini ignores it without a warning. |
| Arguments that aren't valid JSON | — | Kept as text in `ToolCall::invalid_arguments` with `arguments: null`. Replayed verbatim to OpenAI and to OpenRouter `openai/` models, as `{}` to the others. |

### Conversation (arko_ai)

| Field | Default | Notes |
|---|---|---|
| `enable_cache` | `true` | `with_cache_disabled()`. Required when loading JSON. |
| `prompt_cache_key` | A random UUID per new conversation | `with_prompt_cache_key`. Sent as OpenAI's `prompt_cache_key` (both auth paths), as ChatGPT OAuth's `session-id`, `thread-id` and `x-client-request-id` headers, and as OpenRouter's `session_id` (at most 256 characters, else `Error::InvalidRequest`). A loaded conversation keeps its key. One saved without a key gets a new one. Give each agent or task its own conversation. |
| `system_message` | None | `with_system_message`. A `Role::System` message inside `messages` is `Error::InvalidRequest` on OpenAI, Anthropic and Gemini. OpenRouter sends it as a system message. |
| `openai_turn_state` | None | The ChatGPT OAuth turn-state token of the current user turn, recorded by `add_assistant_message_from_response` and saved with the conversation. |

`add_assistant_message_from_response` leaves out classifier and content-filter refusals,
which must not stay in history. A model's own refusal stays. History that holds another
provider's reasoning artifacts is `Error::InvalidRequest` on Anthropic and Gemini, so a
conversation can't switch to them mid-way. OpenAI drops such artifacts silently and
OpenRouter with a debug log, sending the turn's text and tool calls. A mid-conversation
reasoning-effort update is `Error::UnsupportedFeature` except on OpenRouter.

### Prompt caching (arko_ai)

Caching is on by default. Turning it off (`with_prompt_caching(false)` or
`Conversation::with_cache_disabled()`) takes effect only where the provider has a
switch:

| Provider | Caching | With caching off |
|---|---|---|
| Anthropic, and Claude on OpenRouter | Only through the top-level `cache_control` arko sends | Nothing is cached |
| OpenAI GPT-5.6 and GPT-6, API key | `prompt_cache_options.mode: "implicit"` | `"explicit"` with no breakpoints. Nothing is cached |
| OpenAI before GPT-5.6, API key | Automatic | Still cached. Can't be turned off (no warning) |
| ChatGPT OAuth, every model | Implicit, managed by the backend | Still cached. Ignored with a warning logged once |
| Gemini | Implicit, managed by Google | Still cached. Can't be turned off (no warning) |
| OpenRouter, other models | `prompt_cache_key` sent | No key sent. OpenAI GPT-5.6 and later get `"explicit"` mode, other upstreams keep caching implicitly |

The conversation's key goes with every OpenAI request, and as `session_id` with every
OpenRouter request, either way. It routes requests but doesn't turn caching on. On
Anthropic and OpenRouter, caching also needs the model's cache support
(`Model::supports_cache`). `prompt_cache_lifetime` reports 5 minutes for Anthropic and
Claude on OpenRouter (1 hour with `anthropic_cache_ttl: OneHour`), a conservative
5 minutes for OpenAI with an API key, and 30 minutes for OpenAI over ChatGPT OAuth. Gemini
and other OpenRouter models are `Unknown`. `GenerationConfig::prompt_cache_lifetime_seconds`
replaces any of these, except for an uncached request. It reports `Disabled` for models without cache support (the OpenAI Pro models,
`o3-pro`, `gemini-2.0-flash`, OpenRouter catalog models without cache prices), and with
caching off on Anthropic, Claude on OpenRouter, and GPT-5.6 and later with an API key.
`LLMClient::prompt_cache_lifetime(&model, &config)` reads only `config.prompt_caching`,
not `Conversation::enable_cache`.

### ChatGPT OAuth (arko_ai)

Requests over ChatGPT OAuth go to the Codex backend and differ from API-key requests:

- They always stream. `generate_text` collects the stream, so the stream idle timeout
  applies to it too.
- `max_tokens` is never sent. The model's own output limit applies.
- Caching is implicit and can't be turned off. The conversation's `prompt_cache_key` goes
  in the body and in the `session-id`, `thread-id` and `x-client-request-id` headers. The
  backend derives cache affinity from `session-id`.
- Without a reasoning config, `effort: "medium"` is sent to GPT-5.x and GPT-6 models.
  `openai_reasoning_mode` is omitted and warns once, because the backend does not accept
  `reasoning.mode`.
- For `openai_service_tier`, Fast mode (`service_tier: "priority"`) on the models Codex
  offers it for, `Error::InvalidRequest` on others. `Default` sends no tier. Flex and
  Auto are `Error::UnsupportedFeature`. Every request carries Codex's
  `x-codex-routing-hint` header (`model=<model>`, plus `;tier=priority` in Fast mode).
  Fast calls are estimated at the API's Fast rates and, like every OAuth call, not
  billed. `UsageInfo::service_tier` is the requested tier, the backend's echoed
  `service_tier` is ignored, as in Codex (it reads `default` or `auto` for Fast calls).
- GPT-5.4 and GPT-5.5 snapshot IDs are sent without their date.
- The input limit is capped at 272,000 tokens (`CHATGPT_OAUTH_MAX_INPUT_TOKENS`), or
  the model's limit if lower. `LLMClient::max_input_tokens(&model)` returns it. Requests
  are not checked against it, and `ModelCallUsage::max_input_tokens` reports the
  effective limit for the auth path (272,000 over OAuth).
- The first `x-codex-turn-state` token the backend returns in a user turn is echoed on
  the turn's later requests (tool-result follow-ups and retries), once the response has
  been added with `add_assistant_message_from_response`.
- Usage is billed as `BillingKind::Subscription`, so `billed()` is zero, and the list-price
  estimate stays in `total_cost`. Plan usage from the `x-codex-*` headers is in
  `ModelCallUsage::plan_usage`, with the credits (`x-codex-credits-*`) in
  `PlanUsage::credits`. With several accounts, `ModelCallUsage::account` names the one
  that served the call.
- A request rejected with HTTP 401 `token_expired` is refreshed and resent once. A
  token refresh times out after 60 seconds (`Error::HttpError`, transient).

See [OpenAI Responses](arko_ai-guide.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth).

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
`Error::UnsupportedFeature`). OpenAI and OpenRouter also send image URLs. A model
without image input rejects any image, result images included, before sending
(`Error::UnsupportedFeature`).

A user's images go in their user message after its text, as each provider's image input:
`input_image` items (OpenAI), `image` blocks (Anthropic, Gemini), `image_url` parts
(OpenRouter). `Model::image_refusal(image)` gives the reason an image can't go before any
request is built, a model without image input, an image by URL for Anthropic, Gemini or
Claude Code (which would pass the URL on as text), and a GIF for Gemini.
`Model::sendable_image(image)` also fits an image over the provider's limits
(`Model::image_limits`) by shrinking it, refusing only one it can't read or shrink enough.

On Gemini 3 and later, content that follows tool results is folded into the message's
last `function_result`, the other content of a tool-result message (an `ask_user`
answer, a recovery note, environment details), and the text and image messages that
follow it before the next assistant or tool-result message. Texts are appended to the
result's text after a blank line (no blank line when the output is empty). Images follow
the result's own images, and a text directly followed by an image in the same message
becomes that image's label. Anthropic sends a tool-result message's other content in that
same user message, in the message's own order (after the `tool_result` blocks when the
results come first, as arko's builders put them), and following messages as separate
user messages. OpenAI, OpenRouter and Gemini 2.x send it all as separate user input after
the results. See [Tool images](arko_ai-guide.md#tool-images).

### Retries and timeouts (arko_ai)

- arko_ai never retries a model call itself. The exceptions, over ChatGPT OAuth, a
  request rejected with 401 `token_expired` is refreshed and resent once, and with a
  credential store any other 401 re-reads the store and resends once if it holds newer
  credentials.
- `Error::retry_class()` is `Permanent`, `Transient`, `Capacity`, or `UsageLimit`.
  HTTP 408, 409, 429 and 5xx are retryable unless the body has a permanent code (quota,
  authentication, invalid request, context length, content policy, ...). A plan usage
  limit (`usage_limit_reached`) is `UsageLimit`. An overload code (`overloaded_error`,
  `service_unavailable`, ...) with a retryable status or none is `Capacity`. Any other
  error reported after an HTTP 200 is `Transient`, as are connection failures, timeouts
  and `Error::StreamTruncated`. Server delay hints are capped at 30 days.
- `Error::kind()` says what kind of failure it is (`ErrorKind`), beside the retry class,
  which it doesn't change. A code that names a kind decides first (`invalid_api_key`,
  `authentication_error`, `token_expired` and `invalid_grant` are `Authentication`,
  `permission_error` and `unsupported_country_region_territory` `Permission`,
  `insufficient_quota`, `billing_error`, `payment_required`, `quota_exceeded` and Codex's
  spend walls `Billing`, `context_length_exceeded` `ContextOverflow`,
  `request_too_large` and `string_above_max_length` `RequestTooLarge`,
  `model_not_found` and `not_found_error` `NotFound`, and policy codes `Refusal`), then the
  HTTP status (401, 402, 403, 404, 413, 501), then documented phrases of a rejected
  request's message, "prompt is too long" and "input length and max_tokens exceed context
  limit" (Anthropic), "exceeds the context window" and "maximum context length" (OpenAI,
  the ChatGPT backend, OpenRouter), "exceeds the maximum number of tokens allowed"
  (Gemini), "API key not valid" (Gemini), "string too long". Any other rejected request
  is `InvalidRequest`. Rate limits, capacity, server and network errors are `Unknown`.
  Arko's own credential errors (`AuthenticationFailed`, `MissingApiKey`) are
  `Authentication`. A provider that rewords a message makes its kind `Unknown` or
  `InvalidRequest`, never another kind.
- `BackoffPolicy` has no default. `BackoffPolicy::new(initial, max)` jitters each delay
  between half and all of its step, which doubles from `initial` to `max`. A server hint
  is a minimum, with jitter added above it.
- There is no model-request timeout by default. A response is waited for as long as its
  connection stays open, since models can reason silently for over an hour (GPT Pro).
  `LLMClient::with_stream_idle_timeout(duration)` opts in to a limit on the gap between
  data on a streamed response. Expiry is `Error::StreamTruncated` (transient), naming the
  timeout. It covers neither the wait for headers nor responses that don't stream
  (`generate_text`, except over ChatGPT OAuth), and there is no total deadline. Make it
  generous, for example 2 hours. Connecting has its own 30-second timeout and a token
  refresh a 60-second one. See [Long requests](arko_ai-guide.md#long-requests).

### Embeddings (arko_ai)

| Setting | Default | Notes |
|---|---|---|
| `EmbeddingClient::new()` | No Voyage key | `with_voyage_api_key`, or `from_env()` (reads `VOYAGE_API_KEY`, blank counts as unset). `with_voyage_client(VoyageClient::new(key).with_base_url(url))` replaces the base URL `https://api.voyageai.com/v1`. Without a key, `embed` fails with `Error::MissingApiKey` after the local checks. |
| `EmbeddingConfig::new(provider, model_id)` | No default model | An unsupported model is `Error::ModelNotSupported` when embedding or counting tokens. |
| `EmbeddingConfig::output_dimensionality` | `None`, not sent, the model's default, 1,024 | 256, 512, 1,024, 2,048. Another value is `Error::InvalidDimension` before sending. |
| `voyage_input_type` | `None`, not sent | `Document`, `Query` (for search queries). |
| `voyage_truncation` | `None`, `truncation: false` is sent | Voyage's own default is `true`. arko never truncates silently, whatever this is set to, an input over the 32,000-token limit fails locally with `Error::TokenLimitExceeded` before sending, so none reaches Voyage to be truncated. Split long inputs first. |
| `tokenizer_revision` | `None`, `main`, unpinned | Inputs are counted locally before sending, with the `voyageai/<model>` tokenizer downloaded from the Hugging Face Hub on first use and cached per repository and revision (`HF_HOME`, `HF_ENDPOINT`). A failed download is `Error::TokenizerError`, and nothing is sent. A revision the repository doesn't have (the Hub answers 404) is an error naming the repository and revision. |
| `hf_token` | `None`, the Hugging Face token saved on the machine (`$HF_HOME/token`, else `~/.cache/huggingface/token`), if any | Never serialized, and redacted in `Debug`. After loading a config, set it again with `with_hf_token`, in code or from an environment variable the application reads. A config saved with it by an earlier version still loads it. No registered model requires a token. |

Requests are checked locally in this order, the dimension, 32,000 tokens per input, the
model's total tokens per request (unpublished for `voyage-code-4`, so not checked), and
1,000 inputs. Vectors come back in input order. Missing, duplicate, wrong-dimension,
non-finite or all-zero vectors, like a 200 that doesn't parse, are a permanent
`Error::JsonError` of kind `Unknown` (Voyage answered, and may have billed, the request).
arko_ai never retries an embedding request. Voyage error responses keep their status and
body, 400, 401 and 403 are permanent, 429 and 500 transient, and 502, 503 and 504
("Service Unavailable", "high traffic") `Capacity`. See
[Voyage embeddings](arko_ai-guide.md#voyage-embeddings).

### OpenRouter (arko_ai)

See [OpenRouter](arko_ai-guide.md#openrouter) for every option. The defaults:

- No attribution or metadata headers. `OpenRouterClient::with_app_attribution` opts in.
- No catalog is fetched, `Model::openrouter(slug)` has unknown limits and prices, and
  requests aren't checked against a catalog (arko's own checks still apply). Once a
  catalog is loaded (`fetch_models`, `fetch_user_models`, `set_catalog`), a slug it
  doesn't list is `Error::ModelNotSupported` (slugs starting with `@` skip the check),
  and `temperature`, `max_tokens`, `verbosity`, `reasoning`, `tools` or `tool_choice` on
  a model that doesn't list them is `Error::InvalidRequest`.
- `session_id` is always sent. `prompt_cache_key`, and `cache_control` for Claude, only
  while caching is on.
- `auto_exacto` is `CacheAware`. On tool requests with caching on, for a model that bills
  cache writes (from the catalog, or without one every `anthropic/` and `openai/` slug),
  `provider.sort: "price"` is sent, which turns Auto Exacto off and keeps sticky routing
  and its cache. `AllowAutoExacto` never adds it. `DisableAutoExacto` always does on tool
  requests. An explicit `sort` or `order`, or a `:exacto`, `:nitro` or `:floor` variant,
  always wins.
- `fallback_models` is off. With it, OpenRouter tries the next model after an error, a
  refusal included, so another model may answer a refused prompt.
- `provider` preferences, `user` and `service_tier`, sent only when set. `auto_router`
  only for `openrouter/auto` and `openrouter/auto-beta`, ignored for other models.
- `GenerationConfig::openrouter` is merged over `with_default_options`, `provider` and
  `auto_router` merge field by field, and every other field (lists and maps included)
  replaces the default whole, so a request can't unset a default.
- `ReasoningConfig` variants of other providers are dropped with a warning, not
  converted.
- `Conversation::add_reasoning_effort_update` works only on OpenRouter. The other
  providers reject it with `Error::UnsupportedFeature`. OpenRouter requests reject two
  adjacent updates, an update before tool results, updates with pro reasoning or `-pro`
  models, and an update to `none` on Claude.

### Persisted data (arko_ai)

- For `GenerationConfig`, configs with the old `temperature_specified` flag load (`false`
  means no temperature, and `true` without a `temperature` means 0.0). A `temperature`
  without the flag is sent.
- For `Conversation`, `model`, `enable_cache` and `messages` are required. The saved key is
  kept, including old `arko:conversation:<uuid>` keys. A registry model's saved data is
  replaced by the registry's current data on load. Other IDs keep the saved copy.
- For `ToolResult`, `is_error` is required. `images` defaults to none.
- For `ReasoningArtifact`, the old single-item `OpenAIEncryptedReasoning` loads as a
  one-item list.
- For `GenerationResponse`, a response saved before stop reasons existed loads with
  `stop_reason` `Other("unknown")`.
- `ModelCallUsage` and `PlanUsage` saved before `account`, `credits` and `extra_usage` existed load
  without them.
- For `UsageInfo`, `billing` defaults to `Metered`, and a record without `billed_cost`
  reports its `total_cost` as `billed()`. Anthropic records saved before cache counts
  were added back hold only uncached input in `input_tokens`.
- For `EmbeddingConfig`, `hf_token` is never written. A config saved with it by an earlier
  version still loads it.
- A saved conversation whose tool-result message also holds text or images, or is
  followed by text or image user messages before the next assistant message, is sent to
  Gemini 3 and later folded as described above, so its first request after an upgrade
  misses Gemini's cache from that message on.

### Example binaries (arko_ai)

`llm_client_cli` and `embeddings_demo` load a `.env` file. `oauth_cache_probe` doesn't.

#### `chatgpt_accounts_check`

`chatgpt_accounts_check` (`--auth PATH` per account, first preferred, `--model`,
default `gpt-6-luna`, and `--user-agent`) reads each account's usage. `--request` sends one
short request through the account list. `--portability` sends the first account's
encrypted reasoning to the second. See
[Several ChatGPT accounts](arko_ai-guide.md#several-chatgpt-accounts).

#### `llm_client_cli`

`llm_client_cli` (`text`, `chat`, `models`, credentials from `LLMClient::from_env`):

| Flag | Default | Notes |
|---|---|---|
| `-m`, `--model` | Required | A registered ID or an alias (the ID without its date or `-preview`, for Claude also the dotted version, such as `claude-opus-4.5`). An ID with `/` is an OpenRouter slug. |
| `-s`, `--system`, `-t`, `--temperature`, `-n`, `--max-tokens` | Unset, not sent | A temperature for a Claude model that takes none prints a warning. |
| `-r`, `--reasoning` | Off | Reasoning at the model's defaults, OpenAI reasoning models get no reasoning config (their default effort, no summaries, other OpenAI models are an error). Claude models with adaptive thinking get it without an effort. Claude models that need a manual budget get half of `--max-tokens` (or of the output limit), 1,024 to 16,000. Gemini gets its default level with thought summaries. OpenRouter gets `enabled: true`. |
| `--reasoning-level`, `--reasoning-effort` | Unset | An effort (OpenAI, sent with summaries, Claude adaptive, OpenRouter), a level (Gemini) or a token budget (Claude manual, Gemini, OpenRouter). Given together, they must agree, except a Claude budget with an effort. |
| `--reasoning-mode` | Unset | `standard` or `pro`, for OpenAI GPT-5.6 and GPT-6 API-key requests, or through OpenRouter. ChatGPT OAuth omits it. An error for other providers. |
| `--long-context` | Off | |
| `text -p`, `--prompt` | Required | `-i`, `--image PATH`. `--stream`. |
| `--codex-auth`, `--codex-auth-json PATH` | Off | OpenAI credentials from a Codex `auth.json` (the default location, or PATH), kept in step with the file. The `OPENAI_OAUTH_*` variables are then not read. |

#### `embeddings_demo`

`VOYAGE_API_KEY` (skipped without it). `voyage-4-lite`, 256
dimensions, `Document` input. **`oauth_cache_probe`**, `--out DIR` (required, new or
empty unless `--summarize-only`), `--auth-file` (default the Codex `auth.json`),
`--model gpt-6-astra` (a registered OpenAI ID), `--effort xhigh`, `--conversations 3`,
`--turns 15` (2 with `--idle-gap SECS`, which can't be combined with
`--conversations`), `--label baseline`. `--legacy` with its request-shape flags, and
`--reasoning-context`, for comparisons. `--summarize-only` recomputes the summary of an
earlier run.


## arko_agents

Tool-using agents built on arko_ai. An `AgentConfig` describes an agent, a `Task` holds one run, and `AgentRuntime` runs the task as a Tokio task that streams `AgentEvent`s and takes `UserCommand`s.

See [arko_agents guide](arko_agents-guide.md) · [interactive terminals](interactive-terminals.md) · [context management](context-efficiency.md)

Most items are re-exported at the crate root (`arko_agents::AgentConfig`). Items without a root re-export are written with their module, `agent_config::PathSafetyPolicy`, `task::CANCELLED_BY_USER`, `mcp::McpTool`, `parsing::PromptBuilder`, `parsing::remove_output_based_only_markers`, the `parsing::DEFAULT_*` constants, `skill::{LoadedSkill, SkillLoadError}`, `skill::types::SkillFrontmatter`, the built-in tools under `builtins` (`arko_agents::builtins`), and everything in `terminal` (Unix and Windows only) and `utils`.

### Map (arko_agents)

#### Map of agent configuration (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentConfig` | An agent's definition, role, instructions, model, tools and policies. Serializable. No `Default`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::builder`, `AgentConfigBuilder::new` | Start a builder from `id`, `name` and an `AgentLLMConfig`. | `arko_agents/src/agent_config.rs` | [Agent configuration](arko_agents-guide.md#agent-configuration) |
| `AgentConfigBuilder::{role_description, instructions, tool_config, environment_details_config, complete_task_summary_prompt_override, parsing_config_override, error_recovery_config, llm_error_policy, llm_retry_config, user_command_timeout_seconds, user_reminder, skills_config, monitoring, refusal_fallback, build}` | One setter per field. `build` fails without `role_description` or `instructions`, gives unset fields their defaults, and runs `validate`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::load_from_path`, `AgentConfig::save_to_path` | Load JSON without validating. Save pretty JSON through a temporary file and a rename. | `arko_agents/src/agent_config.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `AgentConfig::validate` | Checks retry timing, monitors, tool time limits, a command timeout and reminder interval of at least 1, forced tool choices the model accepts, and a refusal fallback on the same provider. Fails with `ArkoAgentsError::BuilderError`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::{id, name, role_description, instructions, llm_config}` | Required fields. `name` heads the system prompt and is the `agent_name` tools receive. | `arko_agents/src/agent_config.rs` | [Agent configuration](arko_agents-guide.md#agent-configuration) |
| `AgentConfig::tool_config` | The agent's `AgentToolConfig`. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentConfig::environment_details_config` | The built-in environment details (`EnvironmentDetailsConfig`). | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `AgentConfig::error_recovery_config`, `ErrorRecoveryConfig::on_tool_execution_error`, `ToolErrorPolicy::{InformAgentAndContinue, PauseAndAwaitApproval, FailTask}` | What a failed tool call does. | `arko_agents/src/agent_config.rs` | [Tool errors](arko_agents-guide.md#tool-errors) |
| `AgentConfig::llm_error_policy`, `LLMErrorPolicy::{RetryThenPauseAndAwaitApproval, RetryThenInformAgent, InformAgentAndContinue, PauseAndAwaitApproval}` | What a failed model request does. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentConfig::llm_retry_config` | Model-call recovery timing (`LLMRetryConfig`). | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentConfig::user_command_timeout_seconds` | Opt-in limit on each wait for a command. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::user_reminder`, `UserReminder { content, frequency }`, `UserReminder::{new, initial_only, every_message, every_n_messages}`, `ReminderInjectionFrequency::{InitialTaskPromptOnly, EveryUserMessage, EveryNUserMessages}` | Text appended to user messages under `# Reminder`, and how often. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `AgentConfig::skills_config` | Skill sources and embedded skills (`SkillsConfig`), read by `SkillManager::load`. | `arko_agents/src/agent_config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `AgentConfig::monitoring` | Opt-in model-call monitors (`MonitoringConfig`). | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentConfig::refusal_fallback` | Opt-in `AgentLLMConfig` a refused turn is sent to once. | `arko_agents/src/agent_config.rs` | [Refusals (arko_agents)](#refusals-arko_agents) |
| `AgentConfig::claude_code`, `ClaudeCodeSettings { cli_path, cli_args, setting_sources, replace_system_prompt, env, allow_commands_outside_workspace, allowed_command_paths, dangerously_allow_all_commands, command_rules }`, `ClaudeCodeSettings::command_guard`, `AgentConfig::runs_in_claude_code` | How the Claude Code CLI runs for an agent whose model is a Claude Code model. | `arko_agents/src/claude_code.rs` | [Claude Code models](arko_agents-guide.md#claude-code-models) |
| `claude_code::{BUILT_IN_TOOLS, READ_ONLY_BUILT_IN_TOOLS, UNSUPPORTED_BUILT_IN_TOOLS, is_built_in_tool, built_in_path_parameters}`, `ClaudeCodeTools::{read_only, editing, shell, web, all}` | Claude Code's built-in tools, offered to a Claude Code agent when `available_tool_names` lists them. Those that only read (their calls don't wait for the task snapshot). Their path inputs for the workspace path policy. | `arko_agents/src/claude_code.rs` | [Claude Code models](arko_agents-guide.md#claude-code-models) |
| `claude_code::HELD_FOR_COMPACTION` | The `ToolRejected` reason of a call held while a compaction is due, for hosts to recognise. | `arko_agents/src/claude_code.rs` | [Claude Code models](arko_agents-guide.md#claude-code-models) |
| `Task::{claude_code_session, claude_code_synced, claude_code_directory}`, `Task::move_to(directory)` | The Claude Code session a task resumes, how much of its conversation the session has, and the directory the CLI runs in for it after the task moved. `move_to` moves the task's working directory and writes its prompt again, and a Claude Code session stays where it started, where Claude Code finds it. The runtime changes the calls of Claude Code's own tools (`Bash`, `PowerShell`, `Glob`, `Grep`, path arguments) so that they run in the task's directory, and the paths of its file tools (`Read`, `Edit`, `Write`, `NotebookEdit`) to be the task's. | `arko_agents/src/task/mod.rs` | [Claude Code models](arko_agents-guide.md#claude-code-models) |
| `Task::refresh_copy(&self, copy: &mut Task)` | Makes `copy` equal to the task, as `*copy = task.clone()` does, but keeps the messages and images `copy` holds when they are the same. The workflow's task journal refreshes its snapshot of a running task this way, so a task with many images costs a comparison at each safe point instead of a copy. | `arko_agents/src/task/copy.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentConfig::handover`, `HandoverConfig { at_input_tokens, at_context_fraction, report_request, compact_in_claude_code }`, `HandoverConfig::{at_input_tokens, at_context_fraction, threshold, report_request, validate}` | Opt-in conversation handover and its threshold. | `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `handover::{HANDOVER_REPORT_TOOL_NAME, HANDOVER_REPORT_PARAM_NAME, report_tool_definition}` | The `handover_report` tool that native agents with `handover` set give the report with. The runtime answers its calls itself (it isn't in a `ToolRegistry`), and `AgentConfig::validate` rejects an agent tool of the same name beside `handover`. | `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `HandoverHook::{on_handover, on_compaction}`, `HandoverSeed { task_prompt, context }`, `HandoverSeed::{for_task, for_chat, message}`, `handover::unanswered_user_messages`, `AgentRuntime::with_handover_hook` | A host's hook, called as the conversation is replaced, where it returns the fresh conversation's first message, and after Claude Code compacted it, where it returns context the agent gets again. | `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `Task::{handover_requested, handovers, conversation_usage_start, conversation_model_call_start}`, `HandoverRequest` (with the `report` given with the tool, until the next turn boundary), `HandoverRecord`, `Task::{start_fresh_conversation, current_conversation_context_size}`, `handover::{DEFAULT_REPORT_REQUEST, DEFAULT_TEXT_REPORT_REQUEST, DEFAULT_CHAT_REPORT_REQUEST, default_report_request, REPORT_HEADING, REPORT_INTRODUCTION}`, `HandoverConfig::chat_report_request` | A pending handover, the handovers made, where the current conversation's usage starts, and the primitives the runtime uses. | `arko_agents/src/task/mod.rs`, `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `AgentConfig::complete_task_summary_prompt_override` | Replaces the description of `complete_task`'s `summary` parameter. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::parsing_config_override`, `AgentConfig::get_tool_parsing_config` | Output-based tag syntax. The getter returns the override or `ToolParsingConfig::default()`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `AgentConfig::{should_auto_approve_tool, get_path_safety_policy, is_parallel_safe, allows_batched_calls_for_tool, asks_one_tool_call_per_turn, is_native_tool_calling}` | Queries that resolve per-tool overrides against the agent's defaults, as the runtime does. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentLLMConfig { model, generation_config }` | An `arko_ai::Model` and its `arko_ai::GenerationConfig`. Saved as `provider`, `model_id`, a model snapshot and `generation_config`. | `arko_agents/src/agent_config.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |

#### Map of tool configuration (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentToolConfig` | Offered tools, tool calling mode, approvals, workspace path policy, parallel calls and time limits. Has `Default`. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::available_tool_names` | The tools offered besides `complete_task`. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::tool_calling_mode`, `ToolCallingMode::{Native, OutputBased}` | Native tool calling, or legacy calls parsed from the response text. | `arko_agents/src/agent_config.rs` | [Tool calling modes](arko_agents-guide.md#tool-calling-modes) |
| `AgentToolConfig::default_auto_approve_tool_use` | Whether calls run without approval unless a per-tool entry says otherwise. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::default_out_of_workspace_path_policy`, `agent_config::PathSafetyPolicy::{AutoReject, RequireApproval, AutoApprove}` | What a path parameter outside the working directory does. | `arko_agents/src/agent_config.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `AgentToolConfig::one_tool_call_per_turn` | Opt-in, one tool call per response. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::tool_timeout_seconds`, `DEFAULT_TOOL_TIMEOUT_SECONDS` | The host-side time limit of a tool call. The constant is 1,200 seconds. | `arko_agents/src/agent_config.rs` | [Tool time limit](arko_agents-guide.md#tool-time-limit) |
| `AgentToolConfig::tool_specific_configs`, `PerToolConfigOverride { auto_approve_tool_use, parallel_safe, out_of_workspace_path_policy, tool_timeout_seconds, max_result_tokens }` | Per-tool overrides. Each `None` inherits the agent's setting. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::offered_tool_names`, `AgentToolConfig::tool_timeout` | The offered names (`available_tool_names` plus `complete_task`, without `complete_task` under `text_reply_ends_task`), and a tool's effective time limit. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::result_limits`, `ToolResultLimits { overflow_guard, margin_tokens, max_result_tokens }`, `ToolResultLimits::{is_default, without_overflow_guard, margin_tokens, max_result_tokens}` | Host-side limits on what tool results add to the conversation, the overflow guard (on by default) and an opt-in cap per result. Never shown to the model. | `arko_agents/src/agent_config.rs` | [Tool result size (arko_agents)](#tool-result-size-arko_agents) |
| `AgentToolConfig::max_result_tokens(tool_name)`, `PerToolConfigOverride::max_result_tokens` | A tool's effective cap. A tool's own cap (`null`, none for this tool). | `arko_agents/src/agent_config.rs` | [Tool result size (arko_agents)](#tool-result-size-arko_agents) |
| `AgentToolConfig::read_file_line_ranges` | Opt-in. The built-in `read_file` takes `start_line` and `end_line`. | `arko_agents/src/agent_config.rs` | [Tool result size (arko_agents)](#tool-result-size-arko_agents) |
| `AgentToolConfig::text_reply_as_message` | Opt-in. A reply in plain text, without a tool call, is sent to the user as `MessageToUser` and the task carries on. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::text_reply_ends_task`, `AgentToolConfig::ends_task_with_text` | Opt-in. A reply in plain text, without a tool call, is the final reply and completes the task with the text as its summary. The agent isn't offered `complete_task`. `ends_task_with_text` is the setting in native mode, which is where it applies. | `arko_agents/src/agent_config.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `AgentToolConfig::chat`, `AgentToolConfig::chat()` | Opt-in. The agent is a chat, offered no tools (not `complete_task` either), its reply in text ending its task. Hosts add none of their tools, and its handover asks for a summary in text and answers the user's latest message. Native mode only, with no other listed tools and without `text_reply_as_message`. | `arko_agents/src/agent_config.rs` | [Tool calling modes](arko_agents-guide.md#tool-calling-modes) |

#### Map of environment details (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `EnvironmentDetailsConfig { current_working_directory, file_tree, file_tree_mode, current_datetime, operating_system }` | The built-in details. None is included by default. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `BuiltInEnvironmentDetail { include, placement, depth }` | One detail's switch, placement and (file tree only) depth. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `EnvironmentDetailPlacement::{SystemPrompt, InitialTaskPrompt, EachUserMessage}` | Where a detail goes. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `FileTreeMode::{Diff, Full}` | File tree as a first snapshot and then changes, or in full at its placement. | `arko_agents/src/agent_config.rs` | [File-tree context](context-efficiency.md#file-tree-context) |
| `DeveloperDefinableEnvironmentDetail { name, placement, content_provider, static_content_for_serde, dynamic_provider_name_for_serde }`, `DeveloperDefinableEnvironmentDetail::{new_static, new_dynamic_placeholder, missing_content_error}` | A host-defined detail passed to `Task::new`, with static content or a provider. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |
| `DynamicContentProvider::{Static, Dynamic}` | Static text, or an `Fn(&Task) -> Result<String, String>` called when the detail is rendered. Not serialized. | `arko_agents/src/agent_config.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |

#### Map of model-call recovery and monitoring (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `LLMRetryConfig { initial_delay_ms, max_delay_ms, max_elapsed_ms, jitter, capacity_retry, rate_limit_retry, usage_limit_policy, stream_idle_timeout_ms }`, `LLMRetryConfig::validate` | Timing of model-call recovery. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `LLMRateLimitRetryConfig { initial_delay_ms, max_delay_ms, max_wait_ms }` | Waits for short rate limits and capacity errors. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `LLMCapacityRetryConfig { initial_delay_ms, max_delay_ms }` | Opt-in indefinite waits for capacity errors. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `UsageLimitPolicy::{Pause, WaitUntilReset { max_wait_ms }}` | What a plan usage limit does. | `arko_agents/src/agent_config.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `MonitoringConfig { plan_usage, cache }`, `MonitoringConfig::{is_empty, validate}` | The opt-in monitors. | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `PlanUsageThreshold { percent, action }`, `PlanUsageAction::{Pause, SlowDown { delay_ms }}` | A plan-usage threshold checked before each request, and what reaching it does. | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `CacheMonitorConfig { warmup_calls, window_calls, warn_below, pause_below }` | The prompt-cache efficiency monitor. | `arko_agents/src/agent_config.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |

#### Map of tasks (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Task` | One run. The configuration it runs with, its conversation, state and usage. Serializable. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::new(agent_config, initial_task_prompt, initial_cwd, developer_env_details)` | A `Pending` task with an empty conversation for the configured model. | `arko_agents/src/task/mod.rs` | [Quick start](arko_agents-guide.md#events-and-commands) |
| `Task::load_from_path`, `Task::save_to_path` | Load JSON. Save pretty JSON through a temporary file and a rename. | `arko_agents/src/task/mod.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `Task::{id, agent_config_id, agent_config, initial_task_prompt, initial_task_images, conversation, current_state, start_time, end_time, developer_env_details, current_working_directory}` | The task's identity, the configuration and prompt it runs with (and the prompt's images, which the runtime moves to follow its text in the first message, `Task::initial_images` giving them either way), its `arko_ai::Conversation`, state, times, developer details and working directory. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::{usage_history, model_call_usage_history}`, `Task::{total_usage, total_usage_cost, total_input_tokens, total_output_tokens, current_context_size, record_model_call_usage}` | Per-call usage and totals. `total_usage_cost` is the billed amount, ChatGPT OAuth calls add zero. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::{user_message_count, pending_native_tool_results, pending_native_tool_calls, pending_tool_images, pending_text_tool_results, deferred_tool_errors, runtime_messages, runtime_text_messages, fallback_for, fallback_calls, plan_usage_threshold_reported, monitor_pause, pending_context_injections, pending_context_images}`, `Task::{increment_user_message_count, set_state, get_run_time}` | State the runtime keeps between runs, saved with the task, and helpers it uses. `set_state` records `end_time` on entering a terminal state. | `arko_agents/src/task/mod.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `Task::prepare_for_resumption_from_completed(prompt)`, `prepare_for_resumption_from_completed_with_images(prompt, images)` | Puts a `Completed` task back in progress with the prompt (and its images after it) as a new user message. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::prepare_for_resumption_with_user_response(response)`, `prepare_for_resumption_with_user_response_and_images(response, images)` | Answers an `AwaitingUserResponse` task before it runs again, with the answer's images after it. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::reconfigure(config)`, `Task::reconfigured`, `Task::claude_code_prompt_changed` | Continues a task (pending, completed, or awaiting an answer) under another configuration with the same model, new tools and a system message written from `config` at the next run, the same conversation and prompt cache key. A Claude Code session resumes with `--system-prompt-snapshot off`. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::user_turns`, `UserTurn` (`index`, `kind`, `text`, `images`), `UserTurnKind::{Prompt, Message, ContextUpdate}` | The messages of the current conversation the host sent, with their text without the runtime's environment details and reminder. | `arko_agents/src/task/cut_back.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::cut_back(index)`, `Task::branch(index)` | Cut the conversation back to just before a user turn (the task `Completed`, or `Pending` at `0`), dropping what waited with the later messages, or copy it, cut back or whole, into a new task with its own ID and no usage. | `arko_agents/src/task/cut_back.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `Task::claude_code_entries`, `Task::claude_code_resume_at`, `Task::claude_code_fork_of` | Each message's last entry in the Claude Code transcript, by index. The entry a cut back resumes at. The session a branch forks on its first run. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `TaskState::{Pending, InProgress, AwaitingToolApproval, AwaitingBatchToolApproval, AwaitingUserResponse, AwaitingErrorRecoveryApproval, Interrupted, Completed, Failed, Cancelled}`, `TaskState::{is_terminal, is_awaiting_approval}` | A task's state. `Completed`, `Failed` and `Cancelled` are terminal. `Failed { reason, cause, error_kind }`, `cause` is the pause's `ErrorRecoveryCause` when the host rejected an error-recovery pause (`NonRetryable` for a request that needs a new credential or a configuration change), else `None`. `AwaitingErrorRecoveryApproval { error_message, cause, error_kind }` and `Failed` record the `arko_ai::ErrorKind` of the model-request error that ended recovery, when it was classified (`Authentication` for a lost login, `ContextOverflow` for a request over the model's input limit). `None` otherwise. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `ErrorRecoveryCause::{Unknown, RetriesExhausted, NonRetryable, UnusableResponses, UsageLimit, Refusal, ToolError, Monitor, Middleware, BudgetExceeded}` | Why a task awaits error-recovery approval. Both runtimes report a pause as the state change that gives its cause, then `ErrorEncountered`. | `arko_agents/src/task/mod.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `ApprovalReason { tool_not_auto_approved, path_outside_workspace }`, `ApprovalReason::{is_unrecorded, describe}` | Why a call awaits approval. | `arko_agents/src/task/mod.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `PendingToolCallInfo { tool_call_id, tool_name, parameters, tool_description, approval_reason }` | One call of a batch awaiting approval. | `arko_agents/src/task/mod.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `DeferredToolError { message, tool_call_id, middleware_pause }`, `PlanUsageWindowMark { window, resets_at }` | Saved records of a pending tool-error pause and of the last plan-usage window reported. | `arko_agents/src/task/mod.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `task::CANCELLED_BY_USER` | `"Task cancelled by user."`, the reason a `CancelTask` records. | `arko_agents/src/task/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `Task::running_tool_calls`, `RunningToolCall { tool_call_id, call_id, tool_name, parameters, custom_output, message_to_agent, built_in, output, output_images }` | The calls that started and have no recorded result yet, recorded before they run. A task saved with some settles them first when it runs again. Claude Code recovery keeps completed tool images in `output_images`, which defaults to empty in older snapshots. | `arko_agents/src/task/mod.rs` | [Task snapshots and recovery](arko_agents-guide.md#task-snapshots-and-recovery) |
| `Task::parent`, `TaskParent { task_id, tool_call_id }` | The task and tool call a task runs inside, recorded by a host that runs tasks inside tool calls (arko_workflows' delegated agents and verifiers) to find them again on recovery. The runtime doesn't use it. | `arko_agents/src/task/mod.rs` | [Task snapshots and recovery](arko_agents-guide.md#task-snapshots-and-recovery) |

#### Map of refusals (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Task::resolve_refusal(changes)` | Applies changes to the refused input, all or none, and removes the refused turn. | `arko_agents/src/task/refusal.rs` | [Refusals](arko_agents-guide.md#refusals) |
| `RefusalResolution::{RephraseUserMessage, RemoveUserMessage, RemoveImages, RedactToolResults}` | One change to a refused input. | `arko_agents/src/task/refusal.rs` | [Refusals](arko_agents-guide.md#refusals) |
| `Task::refused_input`, `RefusedInput { messages }`, `RefusedMessage { message_index, runtime, runtime_text, tool_results, images }`, `RefusedToolResult { call_id, tool_name }` | The user messages after the last assistant turn that a refused turn answered. | `arko_agents/src/task/refusal.rs` | [Refusals](arko_agents-guide.md#refusals) |

#### Map of runtime and commands (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentRuntime::new(client)` | A runtime over an `Arc<arko_ai::LLMClient>`. | `arko_agents/src/runtime/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentRuntime::run_task(task, tool_registry, event_throttle_ms, simplified_tool_events, skill_prompt_context, interceptor_config, execution_middleware)` | Spawns the run. Returns the event receiver, the command sender, and a join handle that resolves to the task. | `arko_agents/src/runtime/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentRuntime::with_turn_context`, `TurnContextSource::turn_context`, `TurnContextSource::hold_request` | A source asked for context before each model request, and whether to hold the request (the task then pauses with `ErrorRecoveryCause::BudgetExceeded`, whatever its error policy, default, never). | `arko_agents/src/runtime/turn_context.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `TurnContextSource::context_update`, `TurnContextSource::context_update_within`, `TurnContextSource::identify`, `TurnContextSource::wakes_calls`, `TurnContext` (`content`, `user_content`, `ids`, `images`), `ContextImage` (`label`, `image`), `ContextImage::new` | The context with ids naming its items (default `turn_context` without ids) and its images, each after its label, an id for a context injection from the command channel (default none), and whether that injection fires the running calls' `TaskContext::context_arrived` (default yes, a workflow's context notices no). Both ids feed `AgentEvent::ContextInjected`. `context_update_within` is the update within a number of characters, which the runtime asks of a Claude Code agent's hook (default `context_update`). `user_content` is the part the user typed, which follows `content`. During a turn, a Claude Code agent gets an update's text at once in a hook, saying its images follow, and the images with its next input, and `user_content` as a message from the user. | `arko_agents/src/runtime/turn_context.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentRuntime::with_host_decisions`, `HostDecisions::{new, set, get}` | A run-level switch, while `true`, no wait of the run is timed. | `arko_agents/src/runtime/mod.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentRuntime::with_handover_threshold`, `HandoverThreshold::{new, set, get}` | A handover threshold the host can change while the task runs, in place of the configured one. | `arko_agents/src/handover.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `AgentRuntime::with_background_jobs(manager)` (Unix and Windows), `WorkflowBuilder::background_jobs` | The terminal manager the tasks' calls move to the background with, and whose notes they get. Without it no call moves. A workflow gives it to every agent's runtime, and stops a subagent's moved calls and monitors when its run ends. | `arko_agents/src/runtime/mod.rs`, `arko_workflows/src/workflow/mod.rs` | [Moving calls to the background](arko_agents-guide.md#moving-calls-to-the-background) |
| `AgentRuntime::with_stop_signal(signal)` | The host's `StopSignal` for the tasks' tool calls. It reaches each call as part of `TaskContext::stop`. Firing it changes nothing else. The default is a signal that never fires. | `arko_agents/src/runtime/mod.rs` | [Stopping tool calls (arko_agents)](#stopping-tool-calls-arko_agents) |
| `AgentRuntime::with_task_snapshots`, `TaskSnapshotSink::task_changed(task, durable)` (`arko_agents::snapshot`) | A sink the running task is reported to at its safe points. `durable` reports (before tool calls start, unless every call of the group is `ToolRecovery::Rerun` or a Claude Code read-only built-in) return once it is saved. | `arko_agents/src/snapshot.rs`, `arko_agents/src/runtime/mod.rs` | [Task snapshots and recovery](arko_agents-guide.md#task-snapshots-and-recovery) |
| `ToolImageSink::tool_images(task_id, tool_call_id, tool_name, images)`, `ToolImageSink::tool_images_with_call_id(task_id, tool_call_id, provider_call_id, tool_name, images)`, `AgentRuntime::with_tool_image_sink`, `WorkflowBuilder::tool_image_sink` | Gives the host the images each tool call returns, once per call, keyed by the call's runtime ID (as its events name it), before the model's filtering. The runtime calls `tool_images_with_call_id`, which adds the ID the model's provider gave the call (the `id` of the conversation's `ToolCall`, a Claude Code `tool_use_id`, `None` for a call written as text). Its default calls `tool_images`, so a sink that needs only the runtime's ID implements that. Events carry no image bytes. | `arko_agents/src/tool/mod.rs`, `arko_workflows/src/workflow/mod.rs` | [Tool images (arko_agents)](#tool-images-arko_agents) |
| `UserCommand::{ApproveToolUse, RejectToolUse, SubmitBatchToolApprovals, SubmitUserResponse}`, `ToolApprovalDecision::{Approve, Reject}` | Answers to approval and question waits. An approval can carry `custom_tool_output` (used instead of running the tool) and `message_to_agent`. | `arko_agents/src/runtime/command.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `UserCommand::{ApproveErrorRecovery, ContinueAfterError, RejectErrorRecovery, ResolveRefusal}` | Answers to an error-recovery pause. | `arko_agents/src/runtime/command.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `UserCommand::{InjectContext, HostDeciding, RequestHandover, CancelTask, SuspendTask}` | Context updates, wait control, a handover, and stopping the run. | `arko_agents/src/runtime/command.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |

#### Map of events (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentEvent::{TaskStateChanged, TaskStreamEnd}` | A state change. `TaskStreamEnd` is the last event of a run. | `arko_agents/src/runtime/event.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentEvent::{AssistantResponseChunk, ReasoningChunk}` | Response text and reasoning, chunk by chunk, on Claude Code models too (from its partial messages, without repeating them when the complete message arrives). | `arko_agents/src/runtime/event.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentEvent::{ToolCallInitiated, ToolCallFormatted}` | A call announced, then parsed with its parameters. | `arko_agents/src/runtime/event.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentEvent::{ToolParamName, ToolParamValueChunk, ToolParamEnd}` | Parameter-by-parameter parsing events, in output-based mode with `simplified_tool_events` off. | `arko_agents/src/runtime/event.rs` | [Runtime (arko_agents)](#runtime-arko_agents) |
| `AgentEvent::{ToolCallAwaitingApproval, BatchToolCallAwaitingApproval}` | The task waits for an approval. | `arko_agents/src/runtime/event.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `AgentEvent::{UserQuestionsAsked { questions, asked, wait }, QuestionsClosed { ids, why }}` | The agent asked (the task waits when `wait`), or closed questions. | `arko_agents/src/runtime/event.rs` | [Questions](arko_agents-guide.md#questions) |
| `AgentEvent::MessageToUser { message, tool_call_id }` | A message for the user, and the task carries on, a `message_user` call, or (`tool_call_id` `None`) a reply in plain text under `text_reply_as_message`. | `arko_agents/src/runtime/event.rs` | [Tools](arko_agents-guide.md#tools) |
| `AgentEvent::{ToolResult, ToolResultIntercepted}` | A call's result, and the replacement when an interceptor changed it. | `arko_agents/src/runtime/event.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |
| `AgentEvent::{ToolRejected, UnavailableToolAttempted}` | A call that was rejected, stopped or not run. A call to a tool the task can't run. | `arko_agents/src/runtime/event.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentEvent::{ErrorEncountered, RetryScheduled}` | An error (`requires_approval_to_retry` marks a pause, `error_kind` is a failed model request's `arko_ai::ErrorKind`, when classified), and a scheduled retry with its attempt, delay and elapsed time. For a Claude Code model, Claude Code's own retries (`api_retry`). | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::ServiceWarning { text }` | Claude Code's own message about a failed model request. A host shows it as a warning line, apart from the reply, and the task keeps no message for it. The `ErrorEncountered` and pause that follow carry the same text, so a host shows them as a state without repeating it. | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::ToolResultShortened { call_id, tool_name, original_tokens, kept_tokens, reason }`, `ShortenReason::{Overflow, ResultCap, OverflowRecovery}` | A tool result was shortened before it went into the conversation, with the estimated sizes. The `ToolResult` event before it has the whole output. | `arko_agents/src/runtime/event.rs` | [Tool result size (arko_agents)](#tool-result-size-arko_agents) |
| `AgentEvent::UsageLimitReached` | A plan usage limit, with the reset time and the planned wait, when known. | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::OutputTokenLimitReached` | A response ended at its output-token limit. | `arko_agents/src/runtime/event.rs` | [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents) |
| `AgentEvent::{RefusalReceived, RefusedInput, RefusalFallback}` | A refusal, the input it answered, and the fallback model being asked. | `arko_agents/src/runtime/event.rs` | [Refusals (arko_agents)](#refusals-arko_agents) |
| `AgentEvent::{UsageReported, ModelCallUsageReported, PlanUsageReported, ModelRerouted}` | Usage after each model call. Plan usage when the provider reports it. A model served other than the one requested. | `arko_agents/src/runtime/event.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentEvent::ModelRequestStarted { model, subagent, sizes, bytes_per_token }`, `calibrated_bytes_per_token`, `DEFAULT_BYTES_PER_TOKEN` | A model request going out (before each attempt, or for Claude Code models when the response starts), with its `RequestSizes` and the bytes per token calibrated from the task's earlier requests. | `arko_agents/src/runtime/event.rs`, `arko_agents/src/runtime/result_budget.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentEvent::StreamedOutputDiscarded { text_bytes, reasoning_bytes }` | A response failed after streaming part of it, the text and reasoning chunks since the latest `ModelRequestStarted` (conversation's own) belong to no response and a host drops them. Comes before the failure's `ErrorEncountered` and `RetryScheduled`, in both runtimes (Claude Code's at its `api_retry`, or as a stream stops unfinished). | `arko_agents/src/runtime/event.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentEvent::ImagesUnavailable { reasons }`, `Task::settle_unavailable_images`, `Task::unavailable_images`, `arko_ai::Image::{unavailable, unavailable_reason}`, `image_store::MISSING_IMAGE_TYPE` | An image a store can't give back (its file deleted or damaged) reads as a placeholder with the reason. A task turns each into the line `(image no longer available: REASON)` (dropped from images still to send), keeps the reasons until its run reports them as it starts, and the runtimes settle them before any request. | `arko_agents/src/task/mod.rs`, `arko_ai/src/image_store.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentEvent::{PlanUsageThresholdReached, CacheEfficiencyReported, CacheEfficiencyLow}` | Events of the opt-in monitors. | `arko_agents/src/runtime/event.rs` | [Monitoring (arko_agents)](#monitoring-arko_agents) |
| `AgentEvent::{HandoverRequested, ConversationHandedOver}` | A handover started. The conversation was replaced, with the report. |
| `AgentEvent::ConversationCompacted { input_tokens, compacted_tokens }` | Claude Code compacted a Claude Code agent's conversation, with its size before and after. | `arko_agents/src/runtime/event.rs` | [Conversation handover (arko_agents)](#conversation-handover-arko_agents) |
| `AgentEvent::ContextInjected { ids, bytes }` | A context update reached the conversation, with the ids of what it held (from the turn-context source and `TurnContextSource::identify`) and its size. | `arko_agents/src/runtime/event.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `AgentEvent::ConversationCutBack { message_index, removed_messages }`, `AgentEvent::Reconfigured { chat, tools }` | The host cut the conversation back or reconfigured the task. A workflow publishes them before the run's own events (`ContinueOptions::cut_back_to`, `reconfigure`). | `arko_agents/src/runtime/event.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |

#### Map of tools (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Tool` (`definition`, `execute`) | The trait a tool implements. `execute` returns the result text or a `ToolError`. | `arko_agents/src/tool/mod.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `Tool::execute_with_output`, `ToolOutput { text, images }`, `ToolOutput::{text, admit_images, admit_images_within}`, `ToolImage { description, image }`, `ToolImage::{base64, decoded_len}`, `tool::MAX_TOOL_IMAGE_BYTES` | A result with images. By default it calls `execute`. The runtime fits each image to the model (`arko_ai::Model::sendable_image`), and leaves out, with a notice in the text, those it can't send and those past the images a request takes. `MAX_TOOL_IMAGE_BYTES` (5 MiB) is the largest image file `read_file` shows. | `arko_agents/src/tool/mod.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `Tool::{uses_host_timeout, interruptible, allow_result_interception}` | Opt-outs, each `true` by default, the host time limit, being stopped mid-call, result interception. | `arko_agents/src/tool/mod.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `Tool::{bind_runtime, reset_context, observe_context, name}` | A weak command sender for background tools. Callbacks that track what reached the conversation (no-ops by default). `name` is the definition's name. | `arko_agents/src/tool/mod.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `TaskContext { task_id, agent_name, initial_task_prompt, current_working_directory, tool_call_id, recovering, stop, moved, context_arrived, runs_in_claude_code }`, `TaskContext::new(task_id, agent_name, initial_task_prompt, current_working_directory)`, `TaskContext::with_stop` | What `execute` receives about the task and the call. Its runtime id (`None` outside a call), whether it runs again after its task stopped while it ran, its stop signal, the signal that it moved to the background, the signal that context arrived for its task, and whether the task's model runs in Claude Code. `new` leaves the call fields unset, the signals ones that never fire and the last false. `#[non_exhaustive]`, outside the crate, build it with `new`. | `arko_agents/src/tool/mod.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `ContextSignal::{never, has_arrived, arrived}`, `ContextSource::{new, signal, notify}` | The signal of `TaskContext::context_arrived`, and the source a host or a test fires it with. A signal from `signal()` counts the context that arrived after it was made. The runtime's signal for a call also counts the context taken after the request the call came from. | `arko_agents/src/tool/background.rs` | [Context that arrives while a call runs](arko_agents-guide.md#context-that-arrives-while-a-call-runs) |
| `Tool::background_mode`, `ToolBackground::{Never, Detach, Own}` | How a running call can move to the background. `Never` by default. A `Detach` call's future goes on as a job of the runtime's terminal manager, and its result comes as a note. An `Own` tool watches `TaskContext::moved` and moves itself (or ends a wait). `McpTool` and command-template tools are `Detach`. `#[non_exhaustive]`. | `arko_agents/src/tool/background.rs`, `arko_agents/src/tool/mod.rs` | [Moving calls to the background](arko_agents-guide.md#moving-calls-to-the-background) |
| `MoveSignal::{never, is_moved, moved}`, `MoveSource::{new, signal, fire, is_moved}` | The signal of `TaskContext::moved`, and the source a host or a test fires it with. | `arko_agents/src/tool/background.rs` | [Moving calls to the background](arko_agents-guide.md#moving-calls-to-the-background) |
| `StopSignal::{never, is_stopping, reason, stopping}`, `StopSource::{new, signal, stop, is_stopped}` (`arko_agents::stop`) | Whether a task is being stopped, for a tool to observe, and the source a host fires it with. `TaskContext::stop` fires on `CancelTask` or `SuspendTask` while the call runs, or when the host's signal fires (a workflow's wrap-up). | `arko_agents/src/stop.rs` | [Stopping tool calls (arko_agents)](#stopping-tool-calls-arko_agents) |
| `StopAwareCommand::{new, stop_file, grace, terminate_grace, command_mut, output, status}`, `StopAwareOutput { status, stdout, stderr, stopped }`, `ChildStop::{Exited, Terminated, Killed}`, `STOP_FILE_ENV` (`ARKO_STOP_FILE`), `DEFAULT_STOP_GRACE` (10 s), `DEFAULT_TERMINATE_GRACE` (5 s), `kill_stop_aware_commands()` | A child process that stops cooperatively, on a stop, the stop file named by `ARKO_STOP_FILE` is created, then (Unix) its process group gets `SIGTERM`, then it is killed. Dropping the call kills it at once. `kill_stop_aware_commands` kills every such child's group in the process, for a host about to exit without unwinding. | `arko_agents/src/stop.rs` | [Stopping tool calls (arko_agents)](#stopping-tool-calls-arko_agents) |
| `Tool::recovery_mode`, `ToolRecovery::{InDoubt, Rerun, Resume}` | How a call running when its task stopped is settled when the task runs again. `InDoubt` by default. | `arko_agents/src/tool/mod.rs` | [Task snapshots and recovery](arko_agents-guide.md#task-snapshots-and-recovery) |
| `ToolDefinition { name, description, output_based_example, parameters, parallel_safe, tool_type_details, raw_json_schema_override }` | A tool's name, description and parameters as offered to the model. | `arko_agents/src/tool/definition.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolDefinition::{load_from_path, save_to_path, validate_params, coerce_text_arguments, to_json_schema, to_arko_ai_tool_definition}` | File load and save. An argument check the runtime doesn't call. Text-to-boolean coercion for output-based calls. The JSON Schema and arko_ai definition sent to the model. | `arko_agents/src/tool/definition.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `ToolParameterDefinition { name, description, param_type, required, is_path, flag, default_value }`, `ToolParameterType::{String, Bool, Integer}`, `definition::integer_argument` | One parameter. `is_path` marks a path checked against the workspace. `flag` and `default_value` are used by end-user command tools. An `Integer` is an `integer` in the JSON schema. Tools also take its value as text (`integer_argument` reads both), since output-based calls and conversations saved before a parameter was an integer carry text. In output-based calls, `coerce_text_arguments` turns a `Bool`'s `true`/`false` and an `Integer`'s whole number into JSON values. | `arko_agents/src/tool/definition.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `ToolTypeDetails::{BuiltIn, DeveloperDefined, EndUserDefined { command_template }}` | The tool's kind. `EndUserDefined` runs a command. | `arko_agents/src/tool/types.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `ToolRegistry::new`, `ToolRegistry::default` | A registry holding the six built-in tools. | `arko_agents/src/tool/registry.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolRegistry::register` | Adds or replaces a tool by name. A `complete_task` that isn't `BuiltIn` is refused with a warning. | `arko_agents/src/tool/registry.rs` | [Custom tools](arko_agents-guide.md#custom-tools) |
| `ToolRegistry::{get, list_tool_names, list_tool_definitions, restricted_to}` | Lookups. `restricted_to` copies only the named tools, as the runtime does for the offered tools. | `arko_agents/src/tool/registry.rs` | [Tools](arko_agents-guide.md#tools) |
| `ToolRegistry::load_end_user_tool` | Registers an `EndUserDefined` definition as a command tool. | `arko_agents/src/tool/registry.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolError::{ParameterValidation, MissingParameter, ExecutionFailed, Configuration, PathAccessRejected, ShellCommandFailed, ShellCommandTemplate, TimedOut, Io}` | A tool failure. Its text goes into the call's error result. | `arko_agents/src/tool/error.rs` | [Tool errors](arko_agents-guide.md#tool-errors) |

#### Map of result interception and execution middleware (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `InterceptorConfig { interceptor, failure_policy }` | An interceptor and its failure policy, passed to `run_task`. | `arko_agents/src/tool/interceptor.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |
| `ToolResultInterceptor::intercept`, `InterceptedResult::{Unchanged, Replaced}` | A trait that may replace a result's text before it enters the conversation. | `arko_agents/src/tool/interceptor.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |
| `InterceptionFailurePolicy::{UseRawResult, RetryThenUseRawResult, InformAgent, RetryThenInformAgent}`, `InterceptionFailurePolicy::{max_attempts, informs_agent}` | What an interceptor error does. The raw result is used, with a note under the `InformAgent` variants. | `arko_agents/src/tool/interceptor.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `ToolExecutionMiddleware::{before, after}` | A trait called around every tool execution. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |
| `ToolExecutionDecision::{Continue, Reject, FailTask, CancelTask, PauseTask}` | The decision `before` returns. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |
| `ToolExecutionOutcome::{Success, Error, CallerProvided}` | What `after` receives. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |
| `ToolExecutionContext { task_id, agent_name, initial_task_prompt, current_working_directory }`, `ToolCallContext { tool_call_id, tool_name, parameters }` | The task and the call the middleware sees. | `arko_agents/src/tool/middleware.rs` | [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware) |

#### Map of built-in tools (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `complete_task`, `builtins::{CompleteTaskTool, COMPLETE_TASK_TOOL_NAME}`, `builtins::complete_task::COMPLETE_TASK_SUMMARY_PARAM_NAME`, `CompleteTaskTool::default_summary_description` | Ends the task as `Completed { summary }`. Offered whether listed or not. | `arko_agents/src/tool/builtins/complete_task.rs` | [Tools](arko_agents-guide.md#tools) |
| `ask_user`, `builtins::{AskUserTool, ASK_USER_TOOL_NAME}`, `builtins::ask_user::{ASK_USER_QUESTIONS_PARAM_NAME, ASK_USER_WAIT_PARAM_NAME}` | Numbers its questions and puts the task in `AwaitingUserResponse`, or with `async_questions` and no `wait` returns at once. Runs without approval by default. | `arko_agents/src/tool/builtins/ask_user.rs` | [Questions](arko_agents-guide.md#questions) |
| `close_questions`, `builtins::{CloseQuestionsTool, CLOSE_QUESTIONS_TOOL_NAME}` | Closes questions the task asked. Runs without approval by default. | `arko_agents/src/tool/builtins/close_questions.rs` | [Questions](arko_agents-guide.md#questions) |
| `questions::{Question, AskedQuestion, QuestionLog, Answer, QuestionAnswer, Answers, parse_questions, questions_text, answer_note, dismissed_note, message_answers, still_open, open_section}`, `Task::questions`, `AgentToolConfig::async_questions`, `HandoverHook::open_questions` | The questions' types, the task's log, and the texts of their notes. | `arko_agents/src/questions.rs` | [Questions](arko_agents-guide.md#questions) |
| `message_user`, `builtins::{MessageUserTool, MESSAGE_USER_TOOL_NAME}`, `builtins::message_user::{MESSAGE_USER_MESSAGE_PARAM_NAME, MESSAGE_USER_RESULT}` | Sends its `message` to the host as `AgentEvent::MessageToUser` and the task carries on, with no wait and no state change. Parallel-safe. A host tool registered under the name replaces it, and its calls send no `MessageToUser`. | `arko_agents/src/tool/builtins/message_user.rs` | [Tools](arko_agents-guide.md#tools) |
| `read_file`, `builtins::{ReadFileTool, READ_FILE_TOOL_NAME}`, `builtins::read_file::{offered_definition, TERMINAL_HINT, LINE_RANGES_HINT, START_LINE_PARAM_NAME, END_LINE_PARAM_NAME}` | Returns a UTF-8 file with numbered lines (`1 \| …`), or PNG, JPEG, GIF and WebP content as an image, detected by its signature. Images above 5 MiB produce a size notice. Parallel-safe. With `AgentToolConfig::read_file_line_ranges`, `start_line` and `end_line` read only those lines. Offered beside the terminal tools, its description gains `TERMINAL_HINT`. | `arko_agents/src/tool/builtins/read_file.rs` | [Tool result size (arko_agents)](#tool-result-size-arko_agents) |
| `write_file`, `builtins::{WriteFileTool, WRITE_FILE_TOOL_NAME}`, `builtins::write_file::{offered_definition, EDIT_FILE_HINT}` | Writes `content` to `path`, creating parent directories. Offered beside `edit_file`, its description gains `EDIT_FILE_HINT`. | `arko_agents/src/tool/builtins/write_file.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `edit_file`, `builtins::{EditFileTool, EDIT_FILE_TOOL_NAME}` | Replaces `old_string` with `new_string` in an existing file, where it appears exactly once (or every occurrence with `replace_all`). The match is exact. In a CRLF file the strings' `\n` line breaks are written as `\r\n` first. Text not found or found more than once, and other mistakes, are errors that leave the file unchanged. Otherwise the file is written once, with everything outside the match (a byte-order mark, the last line break) as it was. The result names the lines changed and shows them with two lines either side. | `arko_agents/src/tool/builtins/edit_file.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `apply_diff`, `builtins::{ApplyDiffTool, APPLY_DIFF_TOOL_NAME}` | Applies SEARCH/REPLACE blocks to an existing file. | `arko_agents/src/tool/builtins/apply_diff.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `list_files`, `builtins::{ListFilesTool, LIST_FILES_TOOL_NAME}` | Lists a directory's tree at every depth, honouring ignore files. Parallel-safe. Results never intercepted. | `arko_agents/src/tool/builtins/list_files.rs` | [Tools (arko_agents)](#tools-arko_agents) |
| `load_skill`, `LoadSkillTool`, `LoadSkillTool::new`, `LOAD_SKILL_TOOL_NAME` | Returns an on-demand skill, or one of its resources. Registered by `SkillManager`. Parallel-safe. | `arko_agents/src/skill/load_skill.rs` | [Skills](arko_agents-guide.md#skills) |

#### Map of output-based parsing and prompts (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ToolParsingConfig { tag_open, tag_close, batched_tool_call_tag, escape_prefix }`, `parsing::{DEFAULT_TAG_OPEN, DEFAULT_TAG_CLOSE, DEFAULT_BATCHED_TOOL_CALL_TAG, DEFAULT_ESCAPE_PREFIX}` | Tag syntax of output-based calls. The defaults are `<`, `>`, `batched_tool_call` and `!!--!!`. | `arko_agents/src/parsing/mod.rs` | [AgentConfig (arko_agents)](#agentconfig-arko_agents) |
| `strip_output_based_only_blocks`, `parsing::remove_output_based_only_markers` | Remove `{{#output_based_only}}…{{/output_based_only}}` blocks (native mode), or only the markers (output-based mode). | `arko_agents/src/parsing/mod.rs` | [Agent configuration](arko_agents-guide.md#agent-configuration) |
| `ToolCallParser::{new, ingest_chunk, finalize}`, `RawParserEvent::{Text, ToolCallStart, ParamName, ParamValueChunk, ParamEnd, ToolCallEnd, BatchedToolCallStart, BatchedToolCallEnd, Error}` | The streaming parser of output-based calls. | `arko_agents/src/parsing/mod.rs` | [Tool calling modes](arko_agents-guide.md#tool-calling-modes) |
| `assemble_tool_calls`, `ToolCallGroup::{Single, Batch}`, `ParsedToolCall { name, parameters }`, `ToolAssemblyError` (12 variants) | Builds call groups from parser events. | `arko_agents/src/parsing/mod.rs` | [Tool calling modes](arko_agents-guide.md#tool-calling-modes) |
| `parsing::PromptBuilder::{new, build_system_prompt, format_initial_task_prompt, augment_user_message_with_env_details_and_reminder, text_beside_tool_results, format_tool_result_as_user_message, format_tool_rejection_as_user_message, format_tool_execution_error_as_user_message, format_user_response_as_user_message}` | Builds the system prompt and the user messages the runtime sends. | `arko_agents/src/parsing/prompt_builder.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |

#### Map of skills (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `SkillsConfig { workspace_source, global_sources, embedded, on_invalid_skill, load_skill_interception }`, `SkillsConfig::default()` | Where skills come from, invalid-skill handling, and opt-in interception of `load_skill` results. `Default` has no skills, so a configuration can be written with `..Default::default()`. | `arko_agents/src/skill/config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `skill::EmbeddedSkill { name, content }`, `EmbeddedSkill::new(name, content)` | A skill given in memory, with the full text of its `SKILL.md`. It has no resources. `#[non_exhaustive]`, outside the crate build it with `new`. | `arko_agents/src/skill/config.rs` | [Skills](arko_agents-guide.md#skills) |
| `SkillSource { path, include, exclude }`, `InvalidSkillPolicy::{Warn, Error}` | A skills directory with name filters. Skip or fail on invalid skills. | `arko_agents/src/skill/config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `SkillManager::load(config, workspace_root)` | Reads the skills, directories with a `SKILL.md`, `.skill` zip bundles, and single Markdown files other than `README.md`. | `arko_agents/src/skill/mod.rs` | [Skills](arko_agents-guide.md#skills) |
| `SkillManager::setup(registry, agent_config)` | With on-demand skills, registers `load_skill` and offers it with auto-approval. Returns the `SkillPromptContext`. | `arko_agents/src/skill/mod.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `SkillManager::{create_tool, register_tool, tool_name, prompt_context, has_on_demand_skills, has_auto_loaded_skills, total_skill_count}` | The steps `setup` combines, and counts. | `arko_agents/src/skill/mod.rs` | [Skills](arko_agents-guide.md#skills) |
| `SkillPromptContext { on_demand_skills, auto_loaded_skills }`, `SkillSummary { name, description }`, `AutoLoadedSkillContent { name, content }` | What the system prompt lists (on-demand skills) and includes (auto-loaded skills). Passed to `run_task`. | `arko_agents/src/skill/types.rs` | [Skills](arko_agents-guide.md#skills) |
| `skill::scan(config, workspace_root)`, `skill::SkillScan { path, embedded, name, description, auto_load, skipped }` | Every skill file, folder and embedded skill of a configuration with the name it loads under or why it is skipped. `embedded` skills come last with an empty `path`, and one a folder skill replaces names that skill's path in `skipped`. | `arko_agents/src/skill/mod.rs`, `arko_agents/src/skill/types.rs` | [Skills](arko_agents-guide.md#skills) |
| `skill::LoadedSkill { name, original_name, description, auto_load, content, resources, source_label }`, `skill::types::SkillFrontmatter { name, description, auto_load, extra }` | A loaded skill and its parsed front matter. | `arko_agents/src/skill/types.rs` | [Skills](arko_agents-guide.md#skills) |
| `skill::SkillLoadError::{MissingFrontmatter, MalformedFrontmatter, FrontmatterParseError, MissingRequiredField, Io, ZipError, MissingSkillMd}` | Why a skill is invalid. Returned in `ArkoAgentsError::Skill` under `InvalidSkillPolicy::Error`. | `arko_agents/src/skill/types.rs` | [Skills](arko_agents-guide.md#skills) |

#### Map of MCP (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `McpConfig { mcp_servers }`, `McpConfig::load_from_path`, `McpServerConfig { name, command, args, env }` | The server list (JSON key `mcpServers`). Each server is a command to spawn. | `arko_agents/src/mcp/config.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `McpManager::from_config` | Spawns each server, completes the MCP handshake and lists its tools. On a failure it stops the servers already started and returns `ArkoAgentsError::Runtime`. | `arko_agents/src/mcp/manager.rs` | [MCP servers](arko_agents-guide.md#mcp-servers) |
| `McpManager::{register_tools, tool_names, tools}` | Registers the tools. The names to add to `available_tool_names`. The tools. | `arko_agents/src/mcp/manager.rs` | [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents) |
| `McpManager::shutdown` | Closes each connection, gives each server 3 seconds to exit, then kills it. | `arko_agents/src/mcp/manager.rs` | [MCP servers](arko_agents-guide.md#mcp-servers) |
| `mcp::McpTool`, `McpTool::{new, server_name, remote_tool_name}` | An MCP server's tool as a `Tool` named `mcp__<server>__<tool>`. Image blocks and embedded image resources become result images. Embedded text is included, and unsupported media gets a text description. A call dropped before the server answers cancels its request (`notifications/cancelled`). | `arko_agents/src/mcp/tool.rs` | [MCP servers](arko_agents-guide.md#mcp-servers) |

#### Map of interactive terminals (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `terminal::TerminalConfig { directory, shell, shell_args, output_buffer_bytes, max_log_bytes, excerpt_bytes, prompt_quiet_ms, prompt_rules, default_timeout, quiet_terminal_environment, command_guard }`, `TerminalConfig::new(directory)` | The terminal manager's settings. | `arko_agents/src/terminal/mod.rs` | [Configuration](interactive-terminals.md#configuration) |
| `terminal::CommandGuard { allow_outside_workspace, allowed_paths, dangerously_allow_all_commands, rules }`, `CommandGuard::check(command, dialect, working_directory)`, `terminal::CommandRefusal { message, target, protected }`, `terminal::ShellDialect::{Posix, Cmd, PowerShell}`, `ShellDialect::from_shell(program)`, `TerminalError::Refused` | Refuses commands that would delete, move, overwrite or reformat protected paths or paths outside the working directory, and commands its rules refuse. | `arko_agents/src/terminal/guard/mod.rs` | [Command guard](interactive-terminals.md#command-guard) |
| `terminal::CommandRule { action, pattern, within, reason }`, `CommandRule::{allow, deny}(pattern)`, `CommandRule::within(directory)`, `CommandRule::because(reason)`, `terminal::RuleAction::{Allow, Deny}` | A command rule of the guard. The programs and arguments it allows or refuses, matched against each command the guard finds in a line (`git log *`, `cargo * publish *`), optionally only in a directory. | `arko_agents/src/terminal/guard/rules.rs` | [Command rules](interactive-terminals.md#command-rules) |
| `terminal::QUIET_TERMINAL_ENVIRONMENT`, `terminal::QUIET_TERMINAL_LOCALE` | The variables PTY jobs get, with `TERM=dumb`, `NO_COLOR=1`, `PAGER=cat`, `GIT_PAGER=cat`, and `LANG`, `LC_CTYPE`, `LC_ALL` set to `C.UTF-8` (`en_US.UTF-8` on macOS). | `arko_agents/src/terminal/mod.rs` | [Configuration](interactive-terminals.md#configuration) |
| `terminal::PromptRule { name, pattern }` | A named regular expression marking a possible input prompt. | `arko_agents/src/terminal/detection.rs` | [Input requests](interactive-terminals.md#input-requests) |
| `terminal::TerminalManager::new(config)` | Checks the limits, creates and locks the directory, and loads saved job records. | `arko_agents/src/terminal/mod.rs` | [Registering the tools](interactive-terminals.md#registering-the-tools) |
| `TerminalManager::register_tools`, `terminal::TerminalTool::{all, NAMES}` | The four tools `terminal_start`, `terminal_status`, `terminal_input` and `terminal_control`, and the monitor tools (`MonitorTool`). Registering doesn't offer them. | `arko_agents/src/terminal/tools.rs` | [Registering the tools](interactive-terminals.md#registering-the-tools) |
| `TerminalManager::bind_runtime(owner, sender)` | Sends a task's job notices to it as `InjectContext`. `run_task` binds offered terminal tools itself. | `arko_agents/src/terminal/mod.rs` | [Notifications and lifecycle](interactive-terminals.md#notifications-and-lifecycle) |
| `TerminalManager::{start, status, input, cancel, wait, forget}`, `terminal::StartOptions { terminal, detect_prompts, timeout }` | The job operations behind the tools. `StartOptions::default()` is a PTY with prompt detection and no deadline. | `arko_agents/src/terminal/mod.rs` | [Registering the tools](interactive-terminals.md#registering-the-tools) |
| `TerminalManager::{set_allow_outside_workspace, allows_outside_workspace}` | Changes and reads the command guard's `allow_outside_workspace` while the manager runs. A change applies from the next command or input text. | `arko_agents/src/terminal/mod.rs` | [Command guard](interactive-terminals.md#command-guard) |
| `TerminalManager::shutdown` | Cancels the jobs, waits up to 3 seconds for them (4 for a moved call), refuses new starts, and releases the directory so it opens again at once. A job still ending then saves no record once a newer manager has opened the directory. The next open lists the background jobs it stopped (`interrupted_at_open`). | `arko_agents/src/terminal/mod.rs` | [Notifications and lifecycle](interactive-terminals.md#notifications-and-lifecycle) |
| `StartOptions { held, shell_args, label }`, `TerminalManager::{release, subscribe_finished, background_jobs, close}`, `terminal::BackgroundJob { owner, job_id, command, started }`, `terminal::JobFinished { owner, job_id, state }` | Held jobs a host waits for and may move to the background, the jobs that finish in the background, the jobs running there, and closing for the host's exit unless a job the user didn't agree to stop runs. | `arko_agents/src/terminal/mod.rs` | [Held jobs](interactive-terminals.md#held-jobs) |
| `TerminalManager::{jobs, has_news, has_alerts, stop_for_user, interrupted_at_open}`, `terminal::JobInfo { owner, job_id, label, started, kind, tool_call_id }`, `terminal::JobKind::{Command, Call { tool_name }, Monitor(MonitorInfo)}` | The running background jobs with what each runs and the call that made it. Whether an owner has an update or monitor news no runtime got (`has_news`, all of it), or such news that should wake its idle agent (`has_alerts`). The user's stop, which `subscribe_finished` doesn't report. The background jobs the last process left running or stopped as it closed, listed by one open. `JobInfo` and `JobKind` are `#[non_exhaustive]`. | `arko_agents/src/terminal/calls.rs` | [Held jobs](interactive-terminals.md#held-jobs) |
| `TerminalManager::{hold_call, foreground_calls, watch_foreground_calls, move_to_background, set_background_after, background_after}`, `terminal::HeldCall::{signal, is_moved, moved, started}`, `terminal::ForegroundCall { owner, tool_call_id, tool_name, label, started }` | The calls running in the foreground that can move to the background, and moving them. `background_after` is `None` (never) by default. `ForegroundCall` is `#[non_exhaustive]`. | `arko_agents/src/terminal/calls.rs` | [Moved tool calls](interactive-terminals.md#moved-tool-calls) |
| `TerminalManager::{adopt_call, stop_owner_watchers}`, `terminal::CallOutput { text, is_error, images }`, `CallOutput::{text, error, with_images}`, `terminal::CALL_NOTE_CHARS` (30,000), `terminal::IN_THE_BACKGROUND` | A moved call's rest run as a job, which sends its owner a note with the result (cut past `CALL_NOTE_CHARS`, images saved in the job's directory) and gives the call 3 seconds after a stop. Stopping an owner's moved calls and monitors when its run ended. The prefix of a moved call's result. `CallOutput` is `#[non_exhaustive]`. | `arko_agents/src/terminal/calls.rs` | [Moved tool calls](interactive-terminals.md#moved-tool-calls) |
| `TerminalManager::start_monitor(context, command, options)`, `terminal::MonitorOptions { trigger, expires_in, label }`, `MonitorOptions::{lines, check, with_expiry, with_label}`, `terminal::MonitorTrigger::{Lines { pattern }, Check { every_seconds, when }}`, `MonitorTrigger::watching`, `terminal::CheckChange::{Success, Failure, Change, Output}`, `CheckChange::from_name`, `terminal::MonitorStarted { job_id, expires_at, summary }`, `terminal::MonitorInfo { trigger, expires_at, alerts }` | A monitor of a command's output lines that match a pattern, or of a check run every 10 to 600 seconds whose result changes. Its alerts reach its owner as notes. The start returns the tool's text, with the output so far or the first check's result. All `#[non_exhaustive]`. | `arko_agents/src/terminal/monitor.rs`, `arko_agents/src/terminal/checks.rs` | [Monitors](interactive-terminals.md#monitors) |
| `TerminalManager::{subscribe_alerts, rearm}`, `terminal::JobAlert { owner, job_id, kind }`, `terminal::AlertKind::{Lines, Check, EndsSoon}`, `terminal::note_wakes(note)` | Monitor news no runtime took that should wake its owner, at most one alert a minute per owner. What a note waiting in a task should wake its idle agent for. Keeping a monitor for up to 30 more minutes, which arms its warning again. `JobAlert` and `AlertKind` are `#[non_exhaustive]`. | `arko_agents/src/terminal/monitor.rs`, `arko_agents/src/terminal/warnings.rs` | [The outbox](interactive-terminals.md#the-outbox) |
| `TerminalManager::{set_prompt_cache_expiry, set_user_active_at, set_warning_wakes_within, warning_wakes_within}`, `terminal::DEFAULT_WARNINGS_WAKE_WITHIN` (2 hours) | When an agent's prompt cache expires, so a warning that would come after it comes a minute before it. When the user last wrote, and how long after that warnings still wake an idle agent. | `arko_agents/src/terminal/warnings.rs` | [Ending and rearming](interactive-terminals.md#ending-and-rearming) |
| `TerminalManager::{set_monitor_limits, monitor_limits}`, `terminal::MonitorLimits { per_owner, in_all, longest, warning_before, warned_from, cache_margin, alert_interval, gather, lines_per_note, bytes_per_note, line_bytes, partial_line_after, start_wait, shortest_check, longest_check, check_run_limit, check_output_bytes, first_check_wait, pattern_chars, noisy_lines, noisy_window }` | The limits of monitors started from then on. `Default` gives the values in [Ending and rearming](interactive-terminals.md#ending-and-rearming). `#[non_exhaustive]`. | `arko_agents/src/terminal/monitor.rs` | [Ending and rearming](interactive-terminals.md#ending-and-rearming) |
| `terminal::MonitorTool::{all, NAMES}` | The tools `monitor_start` and `monitor_control`, which start an agent's monitors and list, rearm and stop them. | `arko_agents/src/terminal/monitor_tools.rs` | [The monitor tools](interactive-terminals.md#the-monitor-tools) |
| `terminal::TerminalSnapshot { job_id, revision, state, input, exit_code, error, output, log_path, log_truncated, terminal }`, `terminal::OutputWindow { start, end, available_end, gap, text, raw_hex }` | A job's state and output window, as the tools return them. | `arko_agents/src/terminal/mod.rs` | [Reading output](interactive-terminals.md#reading-output) |
| `terminal::JobState::{Running, Succeeded, Failed, Cancelled, TimedOut, Interrupted}`, `JobState::is_terminal`, `terminal::InputNotice { source, prompt, prompt_truncated }`, `terminal::InputSource::{Program, PromptRule}` | A job's state, and a declared or inferred input request. | `arko_agents/src/terminal/mod.rs` | [Input requests](interactive-terminals.md#input-requests) |
| `terminal::TerminalError::{Invalid, NotOwned, StaleView, Finished, Io, Json}` | Terminal operation errors. The tools return their text as `ToolError::ExecutionFailed`. | `arko_agents/src/terminal/mod.rs` | [Input requests](interactive-terminals.md#input-requests) |

#### Map of errors and utilities (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ArkoAgentsError::{LLM, Tool, Io, Json, Configuration, Parsing, AssemblyError, Runtime, TaskExecution, ToolNotFound, InvalidTaskState, ChannelSend, ChannelReceive, FileOperation, UuidError, WalkDir, IgnoreError, JoinError, BuilderError, Skill}` | The crate's error. `BuilderError` comes from validation, `Configuration` from an environment detail without content, `ChannelSend` from a dropped event receiver, `InvalidTaskState` from the `Task` resumption and refusal methods, `FileOperation` from loading and saving. | `arko_agents/src/error.rs` | [Running tasks](arko_agents-guide.md#running-tasks) |
| `utils::resolve_path` | The path a built-in file tool opens, trimmed, taken literally (no `~` or `$VAR`), joined to the working directory, made absolute, with `.` and `..` resolved without following links. | `arko_agents/src/utils.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `utils::is_path_in_workspace` | Whether that path lies inside the working directory, compared component by component (ignoring ASCII case on Windows). A path starting with `~` is always outside. | `arko_agents/src/utils.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `utils::normalize_path` | Resolves `.` and `..` without touching the file system. | `arko_agents/src/utils.rs` | [Approvals and workspace paths](arko_agents-guide.md#approvals-and-workspace-paths) |
| `utils::write_file_atomically` | Writes a temporary file in the same directory, syncs it, and renames it over the target. | `arko_agents/src/utils.rs` | [Persisted data (arko_agents)](#persisted-data-arko_agents) |
| `utils::get_file_tree` | A directory tree as text, honouring ignore files, to an optional depth. | `arko_agents/src/utils.rs` | [File-tree context](context-efficiency.md#file-tree-context) |
| `utils::{get_current_working_directory_string, get_operating_system, get_current_datetime_utc}` | The process's working directory, `std::env::consts::OS`, and the UTC time in RFC 3339. | `arko_agents/src/utils.rs` | [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents) |

#### Map of example binaries (arko_agents)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_agents_cli`, `-c`/`--agent-config`, `-w`/`--workspace-dir`, `-p`/`--prompt`, `--tool-calling-mode`, `--event-throttle-ms`, `--simplified-tool-events`, `--user-command-timeout`, `--mcp-config`, `--openai-auth-json`, `--save-task-state` | Runs one agent from a JSON configuration, asking at the terminal for approvals, answers and recovery. | `arko_agents/examples/arko_agents_cli.rs` | [Example binaries (arko_agents)](#example-binaries-arko_agents) |
| `interactive_terminal` | An offline terminal demo. No arguments. | `arko_agents/examples/interactive_terminal.rs` | [Example binaries (arko_agents)](#example-binaries-arko_agents) |
| `examples/coder_agent.json`, `examples/native-tools-agent.json` | Example agent configurations. | `arko_agents/examples/` | [Example binaries (arko_agents)](#example-binaries-arko_agents) |

### `AgentConfig` (arko_agents)

`AgentConfig::builder(id, name, llm_config)` requires `role_description` and `instructions`, gives the other fields their defaults, and runs `AgentConfig::validate`. `AgentConfig::load_from_path` doesn't validate. The runtime runs `validate` again when a task starts. There is no `Default`.

| Field | Default and omitted | Notes |
|---|---|---|
| `id`, `name`, `role_description`, `instructions`, `llm_config` | Required | In `role_description`, `instructions` and the task prompt, text between `{{#output_based_only}}` and `{{/output_based_only}}` is kept (without the markers) only in output-based mode. |
| `tool_config` | `AgentToolConfig::default()` | Omitted in the configuration saved inside a task, the same, but in output-based mode. See [Tools (arko_agents)](#tools-arko_agents). |
| `environment_details_config` | Nothing included | See [Environment details and reminders (arko_agents)](#environment-details-and-reminders-arko_agents). |
| `error_recovery_config.on_tool_execution_error` | `InformAgentAndContinue`. The error is the call's result | `PauseAndAwaitApproval` lets the turn's other calls run, then pauses once. `FailTask` fails the task at the first error. The calls it didn't run get a result saying so. |
| `llm_error_policy` | `RetryThenPauseAndAwaitApproval { max_attempts: 10 }` | See [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents). |
| `llm_retry_config` | `LLMRetryConfig::default()` | See [Model-call recovery (arko_agents)](#model-call-recovery-arko_agents). |
| `user_command_timeout_seconds` | `None`. A waiting task waits indefinitely | Opt-in. In any awaiting state, a wait with no decision this many seconds after it started returns the task `Interrupted`, and calls awaiting approval get a result saying approval timed out. The deadline is set once per wait, `InjectContext` and commands the wait ignores don't move it. A wait is not timed after `UserCommand::HostDeciding`, nor in a run whose `HostDecisions` is `true` (see [Runtime (arko_agents)](#runtime-arko_agents)). `Some(0)` is rejected by the builder and at task start. |
| `user_reminder` | `None` | `UserReminder { content, frequency }`, appended under `# Reminder`. |
| `skills_config` | `None` | Read by `SkillManager::load`. The runtime itself reads only `load_skill_interception`. See [Skills, MCP and terminals (arko_agents)](#skills-mcp-and-terminals-arko_agents). |
| `monitoring` | Nothing monitored | See [Monitoring (arko_agents)](#monitoring-arko_agents). |
| `refusal_fallback` | `None` | Opt-in `AgentLLMConfig`. A provider other than the configured model's is rejected. See [Refusals (arko_agents)](#refusals-arko_agents). |
| `handover` | `None`. The conversation grows until the task ends | Opt-in `HandoverConfig`. See [Conversation handover (arko_agents)](#conversation-handover-arko_agents). |
| `claude_code` | `None`, `claude` on the `PATH`, none of the user's settings files (`--setting-sources=` and `--strict-mcp-config`), the agent's system prompt appended to Claude Code's, the command guard's defaults on `Bash` and `PowerShell` with no command rules (`command_rules` empty), each `Bash` call starting in the workspace (`CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1` unless `env` sets it), and no self-updates (`DISABLE_AUTOUPDATER=1` unless `env` sets it) | Only for a Claude Code model (`validate` rejects it otherwise). A Claude Code agent must use `Native` tool calling, and its `refusal_fallback` must be a Claude Code model (it answers one turn after a refusal that stands). Built-in tools only for Claude Code models, never `AskUserQuestion`, `ExitPlanMode` or `EnterPlanMode`, and without `tool_timeout_seconds` or `parallel_safe` overrides. See [Claude Code models](arko_agents-guide.md#claude-code-models). |
| `complete_task_summary_prompt_override` | `None`, the built-in description | Replaces the description of `complete_task`'s `summary` parameter, in both tool calling modes. |
| `parsing_config_override` | `None`, tags `<`, `>`, `batched_tool_call`, escape prefix `!!--!!` | Output-based mode only. A field omitted in JSON or YAML takes its default. `escape_prefix: null` turns escaping off. |

`AgentLLMConfig` is saved as `provider`, `model_id`, a snapshot of the model, and `generation_config` (omitted, `GenerationConfig::default()`). On load, a model the arko_ai registry knows takes the registry's current data. An unknown ID keeps its snapshot. An OpenRouter slug without one loads without catalog data. Any other unknown ID fails to load.

### Tools (arko_agents)

`ToolRegistry::new()` registers `complete_task`, `read_file`, `write_file`, `edit_file`, `apply_diff`, `list_files`, `ask_user`, `close_questions` and `message_user`. A task runs only the registry's tools that its configuration offers. `ToolRegistry::register` replaces a tool registered under the same name, a built-in included, except that only a built-in can replace `complete_task`.

| `AgentToolConfig` field | Default | Omitted in JSON | Notes |
|---|---|---|---|
| `available_tool_names` | `["complete_task"]` | Required when `tool_config` is given | The tools offered and run, plus `complete_task`, which is offered whether it is listed or not. A native call to another tool gets an error result and `UnavailableToolAttempted`. In output-based mode its markup stays text. MCP, terminal and custom tools must be listed. `SkillManager::setup` adds `load_skill` itself. |
| `tool_calling_mode` | `Native` | `Native`. `OutputBased` in the configuration saved inside a task | `OutputBased` is legacy. Calls are parsed from the response text, one call or one batch of calls to one parallel-safe tool per response. |
| `default_auto_approve_tool_use` | `false` | `false` | Every call waits for approval, `complete_task` included, except `ask_user` and `close_questions`, which run without approval unless their own entries set `auto_approve_tool_use: Some(false)`. |
| `async_questions` | `false` | `false`, left out when `false` | `true` lets `ask_user` return at once unless the agent passes `wait: true`, and the host delivers the answers later. See [Questions](arko_agents-guide.md#questions). |
| `default_out_of_workspace_path_policy` | `AutoReject` | `AutoReject` | `RequireApproval` asks, even for auto-approved tools. `AutoApprove` allows. Only `is_path` parameters are checked, lexically against the working directory. Symbolic links are not followed, so it is not a sandbox. |
| `one_tool_call_per_turn` | `false` | `false` | `true` disables parallel calls in the request where the provider can (OpenAI, Anthropic and OpenRouter, not Gemini) and runs a response's calls one at a time. `parallel_tool_calls: Some(false)` in the model's `tool_config` means the same. |
| `tool_timeout_seconds` | `Some(1200)` (`DEFAULT_TOOL_TIMEOUT_SECONDS`) | `Some(1200)`. `null` is no limit | Host-side limit, not a tool parameter the model sees. A call over it is stopped (a command it started is killed) and handled as a tool error. `0` is rejected. Tools whose `uses_host_timeout` is `false` are exempt, the terminal tools, and arko_workflows' `run_workflow_agent` and `wait_for_message`. |
| `tool_specific_configs` | Empty | Empty | A `PerToolConfigOverride` per tool name, below. |
| `result_limits` | `ToolResultLimits::default()`, the overflow guard on, no caps | The default. Left out when default | See [Tool result size (arko_agents)](#tool-result-size-arko_agents). |
| `read_file_line_ranges` | `false` | `false`. Left out when `false` | `true` adds `start_line` and `end_line` to `read_file`. |
| `text_reply_as_message` | `false` | `false`. Left out when `false` | `true` makes a native reply in plain text a message to the user (`MessageToUser` without a `tool_call_id`), followed by a note, outside error recovery. Only the first such reply since the last tool call or new message counts, another one right after it is handled as a response without a tool call. A Claude Code agent sends the final text of a turn that ends without `complete_task`. Output-based mode ignores it. |
| `text_reply_ends_task` | `false` | `false`, left out when `false` | `true` makes a native reply in plain text the final reply, the task completes with the trimmed text as its summary (`TaskStateChanged(Completed { summary })`, as for `complete_task`), and `complete_task` isn't offered, even when listed. Text beside tool calls stays progress. A response with neither text nor a tool call gets the note and the recovery attempt it gets today. A handover report that is due comes first. A Claude Code agent's turn that ends with text completes the task, except the turns after `ask_user`, a handover's report turn and a `/compact`. The reply needs no approval and passes no execution middleware, so arko_workflows refuses the setting on agents a completion verifier reviews or runs on. Refused together with `text_reply_as_message`. Output-based mode ignores it. |
| `chat` | `false` | `false`, left out when `false` | `true` makes the agent a chat. No tools, a text reply ends its task, no tools from its host, a handover summary in text. |

`PerToolConfigOverride` fields all default to `None`, which inherits the agent's setting, `auto_approve_tool_use`, `parallel_safe` (instead of the tool definition's), `out_of_workspace_path_policy`, `tool_timeout_seconds` (omitted to inherit, `null` for no limit, or a number for that limit, which applies to exempt tools too), and `max_result_tokens` (omitted to inherit, `null` for no cap on this tool, or a number for that cap).

A custom tool sets its own behaviour, `ToolDefinition::parallel_safe` (omitted in JSON, `false`), `ToolParameterDefinition::is_path` (omitted in JSON, `false`), and the `Tool` methods `uses_host_timeout`, `interruptible` (`false`, `CancelTask` and `SuspendTask` wait for the call to return, and its stop signal fires, see [Stopping tool calls (arko_agents)](#stopping-tool-calls-arko_agents)) and `allow_result_interception`, each `true` by default.

How a native response's calls run (an output-based response holds one call, or one batch of calls to one tool, run one after another):

- Native calls get exactly one result each. Calls run in call order under a fair read/write lock. A parallel-safe call starts once every earlier call that isn't parallel-safe has finished, and any other call waits for every earlier call. `read_file`, `list_files`, `message_user` and `load_skill` are parallel-safe. The other built-in tools and the terminal tools are not. An MCP tool is parallel-safe only when its server marks it `readOnlyHint: true`.
- A call awaiting approval counts as unfinished. The calls after it that must wait are held with it until the approvals are decided.
- Calls after one that ends the turn (`complete_task`, `ask_user`, a tool error under `FailTask`, a middleware's `FailTask` or `CancelTask`) are not run, and their results say so. Parallel-safe calls already running beside it finish. Among the calls run once approvals are decided, only a call that ends the task stops the others. `CancelTask` and `SuspendTask` stop the calls still running (their results say so) unless one of them isn't interruptible.
- `complete_task` always runs alone, after the calls before it. After an earlier call of its turn was rejected or failed, it doesn't complete the task, and its result says so.
- A response without a tool call doesn't end the task. A note asks the model to continue or call `complete_task`, and the response counts as a model-call recovery attempt (under `LLMErrorPolicy::PauseAndAwaitApproval` the task pauses at once). With `text_reply_as_message`, a reply with text is first sent to the user as a message instead, and with `text_reply_ends_task` it completes the task (see the table above).
- A `tool_config` in the model's `GenerationConfig` supplies `tool_choice` and `parallel_tool_calls`. Its tool list is ignored, with a warning in the log. Without one, no tool choice is sent.
- An agent offered no tools (no listed tool its registry holds, and `text_reply_ends_task` on, so no `complete_task`) sends no tool configuration at all, and its system prompt has no "Tool Usage" section (no "Tool names" section on Claude Code models, which run with `--tools ""`).

In both modes:

- A tool returns its whole output (`read_file` the whole file, `list_files` the whole tree, command and MCP tools everything they print). The [overflow guard](#tool-result-size-arko_agents) shortens a turn's results only when they would certainly put the next request over the model's input limit (native tool calling in arko's own runtime), and opt-in caps shorten results over them.
- `write_file` is not atomic. `edit_file` applies in full or not at all. It writes the file once, only when the text was found as the call asks. `apply_diff` writes the blocks that applied even when others didn't. Its result says "Applied N of M" with a line per block. Only a diff where no block applied is a tool error.
- End-user command tools expand `~` and `$VAR` in their command template, split it on whitespace (no shell, no quoting), and run it in the working directory with a null stdin. The process is killed when the call is stopped. The result is stdout. A non-zero exit is a tool error carrying the exit code, stdout and stderr. Tool parameters are never expanded. An `is_path` value `~/notes` names a directory called `~` but counts as outside the workspace.

`InterceptionFailurePolicy` (in the `InterceptorConfig` given to `run_task`) defaults to `UseRawResult`. `RetryThenUseRawResult { max_attempts }` retries first. `InformAgent` and `RetryThenInformAgent { max_attempts }` add a note that the result is the full one. `max_attempts` 0 counts as 1. Error results are never intercepted, nor are the results of `complete_task`, `ask_user` and tools whose `allow_result_interception` is `false` (`list_files`, the terminal tools). `load_skill` results are intercepted only with `SkillsConfig::load_skill_interception`, whose policy then applies. See [Result interception and execution middleware](arko_agents-guide.md#result-interception-and-execution-middleware).

### Tool images (arko_agents)

Custom tools return `ToolOutput { text, images }` from `execute_with_output`.
Each `ToolImage` has a source description and an `arko_ai::Image`. The runtime
keeps images with their tool result, saves them with the task, and forwards them
to the model. Events and result interceptors carry the text alone.

A host that shows the images (thumbnails under a call) gets them from a
`ToolImageSink` (`AgentRuntime::with_tool_image_sink`,
`WorkflowBuilder::tool_image_sink`), `tool_images(task_id, tool_call_id,
tool_name, images)` is called once for each call that returned images, as the
call ends, with the runtime's call ID its events carry and the images as the
tool returned them, before the ones the model can't take are left out. It covers
`read_file`, MCP tools and a host's own tools, and on a Claude Code model also
Claude Code's own `Read`. A call recovered after its task stopped is not
reported again. The runtime calls `tool_images_with_call_id`, which also gives the
ID the model's provider gave the call, to match the call with the conversation's
messages. Its default calls `tool_images`, so a sink written for that keeps
working. Image bytes stay out of the events because a workflow's
broadcast copies each event to every subscriber and into its event log.

The built-in `read_file` detects PNG, JPEG, GIF and WebP by file signature.
MCP tools accept those MIME types in image blocks and embedded blob resources,
with `image/jpg` normalized to `image/jpeg`. Embedded text resources are included
in the result text. Audio, unsupported image formats and other binary resources
get a description. Structured content supplies the text when the server omits it,
including results that also contain images.

The runtime fits each image to the model's provider (`arko_ai::Model::sendable_image`):
one over its limits is scaled down and re-encoded. It omits, with a notice for each, the
images a model without image input can't take, URL images where images go as data, GIF
images on Google (with a conversion hint, following its
[supported formats](https://ai.google.dev/gemini-api/docs/image-understanding#supported-image-formats)),
images it can't read or shrink enough, and, in the native runtime, images past the number a
request to the model takes (a handover or a cut back makes room). URL images have no local
size check.

The Claude Code runtime forwards base64 images as MCP image blocks. Its
`arko_ai::claude_code::ToolOutput::with_images` also serves images from custom
in-process tools. URL images become a text link on this path.

### Tool result size (arko_agents)

`AgentToolConfig::result_limits` (`ToolResultLimits`) holds host-side limits on what tool results add to the conversation. The model sees none of them, only a shortened result's notice.

| Setting | Default | Notes |
|---|---|---|
| `overflow_guard` | `true` | Acts only when the turn's results would certainly put the next request over the model's input limit (`LLMClient::max_input_tokens`, the model's limit, or 272,000 tokens on ChatGPT OAuth). When even a lower bound on the request is over it. Results that may fit are never changed, unless their text packs more than 10 bytes into a token (as long runs of spaces can). A model whose limit is unknown (an OpenRouter model without catalog data) is not guarded. `false` adds every result whole and turns off the overflow recovery. |
| `margin_tokens` | `None`, 2% of the input limit, at least 4,000 tokens | Kept free when the guard shares the room left between shortened results (a nearly full context), for what the runtime adds to the request and for estimate error. It doesn't decide whether the guard acts. |
| `max_result_tokens` | `None`, no cap | Opt-in cap on each result, in estimated tokens. A tool's `PerToolConfigOverride::max_result_tokens` replaces it (`null`, none). `0` is rejected. |
| `AgentToolConfig::read_file_line_ranges` | `false` | `true` adds the optional `start_line` and `end_line` (1-based, inclusive) to `read_file`'s schema and description, and its notice names them. Off, they are neither shown nor mentioned, and a call that passes them reads the whole file. |

- The guard runs once per turn, over all the turn's results (parallel calls included), as they go into the conversation, after result interceptors and distillers, before the next request. It decides by a lower bound on that request. The latest request's reported input tokens, plus every byte added since (the reply and the results) at 10 bytes per token, more than any content measured (with o200k, code 3.6 to 4.8, English 3.8 to 4.3, padded tables 8.6 bytes per token, Claude's tokenizer counts more tokens for the same text). It sizes what shortened results keep by a bytes-per-token ratio calibrated from the task's own requests (between 2 and 4, 3 until the first response). No tokenizer is used. When the request may fit, it is sent. An overflow the provider reports is recovered from as below. The turn boundary's context update is added after the guard and isn't counted by it (see [Conversation handover](#conversation-handover-arko_agents)), an error toward acting late, the safe direction for a lower bound.
- When the overflow is certain, the largest results are shortened first, each to a head-and-tail preview (2% of the input limit, between 4,000 and 16,000 tokens, 80% from the start, cut on line boundaries) with a marker for what was left out and a notice, until it no longer is. When even the previews leave it certain (a nearly full context), the room the calibrated estimate leaves, less the margin, is shared between the results above 500 tokens beyond their notice, a share under 500 tokens keeping the notice alone. Smaller results are never shortened by the guard, and no result is replaced by a notice larger than it. A capped result keeps its head and tail up to its cap. `ToolResultShortened` reports each one. `ToolResult` keeps the whole output.
- The notice says how large the result was and what is shown, then, with the terminal tools offered, terminal commands to look at the rest (for a `read_file` result, on its file, `wc -l`, `grep -n`, `head`, `sed -n '5000,5400p'`, or a short script). Without one, with `read_file_line_ranges` on, `read_file`'s ranges. Otherwise that the output was too large and to ask for less.
- Offered beside the terminal tools (`terminal_start`), the built-in `read_file`'s description gains one sentence pointing to them for large or generated files (`read_file::TERMINAL_HINT`), and offered beside `edit_file`, the built-in `write_file`'s gains one pointing to it for changes to part of a file (`write_file::EDIT_FILE_HINT`). Tool definitions are not part of the configuration a workflow compares on resume, so a saved run resumes as before. In native tool calling the definitions go with each request and the system prompt doesn't change. In output-based mode they are part of the system prompt, written when a task starts, so a saved task keeps the one it was saved with.
- If the provider still answers that a request is over the model's input limit (`ErrorKind::ContextOverflow`), the largest tool result added since the last answered request is shortened (to a preview when it is larger than one, otherwise, or when it was shortened before, to its notice alone) and the request sent again, at most twice in a row. After that, or at once when that result is under about 500 tokens (it is then left alone), an agent that hands over (`handover` set, or a host's `HandoverThreshold`) leaves out the step that request added (the model's reply, its results, and the runtime's messages after them) and hands over from the conversation of the last answered request, its report request starting with `handover::STEP_LEFT_OUT`. Context updates added since stay. Without a handover, recovery ends as for any error that retrying can't fix, with `error_kind` `ContextOverflow`.
- Covered, native tool calling in arko's own runtime. Not covered, output-based (legacy) tool calling, and Claude Code agents, whose context Claude Code manages (its own `Read` and `Bash` have their own limits). A Claude Code agent that hands over still leaves out the step when a turn goes over the limit (see [Conversation handover](#conversation-handover-arko_agents)).

### Stopping tool calls (arko_agents)

A tool learns that its task is being stopped from `TaskContext::stop`, a `StopSignal` (`arko_agents::stop`). It fires when the runtime receives `CancelTask` or `SuspendTask` while the call runs (reason `Task cancelled.` or `Task suspended.`), or when the host fires the signal it gave the runtime with `AgentRuntime::with_stop_signal` (an arko_workflows graceful stop fires it when its wrap-up begins). Firing the host's signal changes nothing else. The task goes on until the host cancels or suspends it.

| Item | Default | Notes |
|---|---|---|
| `AgentRuntime::with_stop_signal` | A signal that never fires | Not saved with the task. |
| `TaskContext::new` | No call id, not recovering, `StopSignal::never()`, `MoveSignal::never()`, `ContextSignal::never()` | For hosts that build a context themselves. `TaskContext` is `#[non_exhaustive]`, adding `stop` broke struct literals, and `new` replaces them. |
| `Tool::interruptible` | `true`, `CancelTask` and `SuspendTask` drop the call at once | Such a call sees its signal only for a host's signal (a wrap-up). A tool that returns `false` is waited for, and its signal fires so it can wind down. A workflow aborts its agent after the grace period if it doesn't. |
| `StopAwareCommand::grace` | `DEFAULT_STOP_GRACE`, 10 seconds | After the stop file is created, before `SIGTERM` (outside Unix, before the child is terminated). |
| `StopAwareCommand::terminate_grace` | `DEFAULT_TERMINATE_GRACE`, 5 seconds | After `SIGTERM`, before `SIGKILL` to the process group (Unix). Also bounds reading output a descendant holds open after a stop. |
| `StopAwareCommand::stop_file` | A fresh path in the system's temporary directory, removed afterwards | A path the caller names (inside a sandbox's bound directory, say) is cleared before the child starts and left afterwards. |

`StopAwareCommand` wraps a `tokio::process::Command`. The child gets `ARKO_STOP_FILE` (`STOP_FILE_ENV`), a path that doesn't exist until a stop. While the signal hasn't fired, the child runs as long as it takes. The graces apply only once a stop is under way, so nothing is timed by default. `output` captures stdout and stderr with a null stdin. `status` keeps the command's own stdio. `StopAwareOutput::stopped` says how a stop ended it (`ChildStop::Exited` within the grace, `Terminated` by `SIGTERM`, `Killed`), or `None` without a stop. On Unix the child leads a new process group, signals go to the group, and the group is sent `SIGKILL` once the child ended after a stop. A process that leaves the group (`setsid`) is not reached. Out of the terminal's foreground group, the child doesn't get the terminal's Ctrl-C and can't read the terminal (a read stops it with `SIGTTIN` until a stop kills it), give a `status` child that may read its standard input a null one or a file. `kill_stop_aware_commands()` kills the groups of all running children at once, for a host about to call `std::process::exit` (which runs no destructors, so no call kills its child). `stop_on_signals` calls it at its third signal. Elsewhere there is no `SIGTERM` step and only the child itself is terminated, so processes it started may outlive it. Dropping the call (an interruptible call cancelled, or over its time limit) kills the child at once (its group on Unix). End-user command tools don't use it. Their child is killed when the call is stopped, as before.

### Environment details and reminders (arko_agents)

`EnvironmentDetailsConfig` has `current_working_directory`, `file_tree`, `current_datetime` and `operating_system`, each a `BuiltInEnvironmentDetail`, plus `file_tree_mode`. None is included by default.

| Setting | Default | Notes |
|---|---|---|
| `BuiltInEnvironmentDetail::include` | `false` | Also when omitted in JSON or YAML. |
| `BuiltInEnvironmentDetail::placement` | `SystemPrompt` | Also `InitialTaskPrompt` and `EachUserMessage`. The system prompt is written once, when the task starts. A date there is labelled `Task Start Datetime (UTC)`, elsewhere `Current Datetime (UTC)`. `EachUserMessage` details go with the task prompt and with the user messages that carry tool results, answers, notes and context updates, in native mode as text after the results. Three messages don't carry them, file-tree updates, the output-based results added when a wait ends without a decision, and the prompt `Task::prepare_for_resumption_from_completed` adds. A context update doesn't repeat them after a message of the same turn that carried them. |
| `BuiltInEnvironmentDetail::depth` | `None`, 3 | File tree only. |
| `file_tree_mode` | `Diff` | JSON `"diff"`, `"full"`. `Diff` sends a full snapshot before the first request, then a message listing changes before later requests, whatever the placement. `Full` renders the whole tree at `placement`. |
| `UserReminder::frequency` | `InitialTaskPromptOnly` | Also `EveryUserMessage`, and `EveryNUserMessages(n)`. The first user message and every nth after it, counting the messages the runtime adds. `n = 0` is rejected by the builder and at task start. |

Developer-defined details (`DeveloperDefinableEnvironmentDetail`) are passed to `Task::new`. A dynamic provider isn't saved with the task. A task that would show a detail without content fails at start with `ArkoAgentsError::Configuration`. A dynamic provider that returns an error leaves the detail out of that message.

### Model-call recovery (arko_agents)

| `LLMErrorPolicy` | Attempts | When recovery ends |
|---|---|---|
| `RetryThenPauseAndAwaitApproval { max_attempts }` (default, 10) | `max_attempts` (0 counts as 1) | Pauses in `AwaitingErrorRecoveryApproval` |
| `RetryThenInformAgent { max_attempts }` | `max_attempts` (0 counts as 1) | Returns the task `Interrupted` |
| `InformAgentAndContinue` | No limit within the elapsed budget. 10 for unusable responses | Returns the task `Interrupted` |
| `PauseAndAwaitApproval` | 1 | Pauses at the first failure |

| `LLMRetryConfig` field | Default | Notes |
|---|---|---|
| `initial_delay_ms` | 2,000 | Must be above 0. |
| `max_delay_ms` | 64,000 | At least `initial_delay_ms`. A delay the provider asks for may exceed it. |
| `max_elapsed_ms` | 300,000 | Budget from the first failure of an episode. It bounds the waits between attempts and the decision to retry. It never cuts off a request in flight, and it stops counting while an accepted response streams. Must be above 0. |
| `jitter` | `true` | Each backoff delay is drawn between half and all of its step. A provider's delay gets up to half a step added. |
| `rate_limit_retry` | `Some(LLMRateLimitRetryConfig::default())`, 1 s to 60 s backoff, `max_wait_ms` 900,000 | Under the `RetryThen*` policies, rate limits and (without `capacity_retry`) capacity errors are waited out outside the attempt limit and the budget. A rate limit asking for more than `max_wait_ms`, or arriving after `max_wait_ms` of rate-limit waits in one episode, is handled as a usage limit. A capacity error asking for longer gets ordinary recovery. Under the other policies, and with JSON `null`, both get ordinary recovery. |
| `capacity_retry` | `None` | Opt-in `LLMCapacityRetryConfig` (30 s to 300 s backoff), under the `RetryThen*` policies, capacity errors are retried until the service recovers. |
| `usage_limit_policy` | `Pause` | After a wait longer than the prompt-cache lifetime, the retried request reads the context uncached, once. A plan usage limit ends recovery under every policy. The opt-in `WaitUntilReset { max_wait_ms }`, under every policy, waits until the reported reset plus 1 s (and up to 30 s of jitter while `jitter` is on) and retries unchanged, if the provider reports a reset and the reset plus 31 s fits in `max_wait_ms`. Otherwise recovery ends as under `Pause`. |
| `stream_idle_timeout_ms` | `None` | Opt-in cut-off of this agent's streamed responses after that long without data (arko_ai's `LLMClient::with_stream_idle_timeout`). The cut is a truncated stream, which recovery resends. Make it generous. `0` is rejected. |

A failed request (a connection failure, HTTP 408, 409, 429 or 5xx, a truncated stream) is resent unchanged, with nothing added to the conversation. A response the runtime can't use (in native mode one without a tool call, in output-based mode one without a call or with calls it can't read) is retried at once, without backoff, with a note to the model. Errors that can't be retried, such as authentication failures, invalid requests and refusals, end recovery at once. The pause records an `ErrorRecoveryCause`, and the error's `arko_ai::ErrorKind` when it was classified (`error_kind`, also on `ErrorEncountered` and, after a rejected recovery, on `Failed`). A request over the model's input limit is first made smaller and sent again, at most twice, under the overflow guard. See [Tool result size (arko_agents)](#tool-result-size-arko_agents). Before each OpenAI request, the runtime refreshes ChatGPT OAuth credentials that expire within 5 minutes. See [Model-call recovery](arko_agents-guide.md#model-call-recovery) and [Errors and retries](arko_ai-guide.md#errors-and-retries).

A response that ends at the output-token limit emits `OutputTokenLimitReached` and is then handled as usual. Nothing continues it automatically, and a call the stream announced but the response doesn't hold gets `ToolRejected`.

### Monitoring (arko_agents)

A call that reports plan usage (the ChatGPT backend on ChatGPT OAuth calls) emits `PlanUsageReported`, and a call served by another model than the one requested emits `ModelRerouted`. `AgentConfig::monitoring` adds opt-in monitors. Both default to `None`:

| Setting | Default | Notes |
|---|---|---|
| `plan_usage: Some(PlanUsageThreshold { percent, action })` | `action`, `Pause` | `percent` is required, above 0 and at most 100. Checked before each request against the latest report. `PlanUsageThresholdReached` once per window and reset. `Pause` pauses once per window and reset. `SlowDown { delay_ms }` waits before every request while the most-used window stays at or above `percent` (a Claude Code agent holds each tool call instead). It spends the same usage over more time, so it helps only when the reset is close, and a delay at or past the prompt-cache lifetime makes each request miss the cache (warned once per run). See [Choosing a plan-usage action](arko_agents-guide.md#monitoring). |
| `cache: Some(CacheMonitorConfig::default())` | `warmup_calls` 2, `window_calls` 5, `warn_below` `Some(0.5)`, `pause_below` `None` | `CacheEfficiencyReported` after each judged call (not the warm-up calls, calls without a cache figure, or refusal-fallback calls). `CacheEfficiencyLow` once per drop of the rolling efficiency below `warn_below`, after `window_calls` judged calls. `pause_below` also pauses before the next request. |

A monitor pause waits in `AwaitingErrorRecoveryApproval` (cause `Monitor`) under the pausing error policies, and returns the task `Interrupted` under the others. Approving it adds no error note.

### Conversation handover (arko_agents)

| Setting | Default | Notes |
|---|---|---|
| `HandoverConfig::at_input_tokens` | `None` | Hand over once a request's input reaches this many tokens. At least 1. |
| `HandoverConfig::at_context_fraction` | `None` | Above 0 and at most 1, of the model's input limit. Ignored for a model whose limit is unknown. With both set, the lower threshold applies. At least one must be set. |
| `HandoverConfig::report_request` | `DEFAULT_REPORT_REQUEST` with the `handover_report` tool, `DEFAULT_TEXT_REPORT_REQUEST` without it, `DEFAULT_CHAT_REPORT_REQUEST` for a chat (`AgentToolConfig::chat`) | The user message asking for the report. Not blank. |
| `HandoverConfig::compact_in_claude_code` | `false`, a handover | On a Claude Code model, Claude Code's `/compact` at the threshold instead, in the same session. Falls back to a handover when Claude Code doesn't compact. Other models hand over. |
| `AgentRuntime::with_handover_hook` | `HandoverSeed::for_task`, the fresh conversation restates `initial_task_prompt`. `HandoverSeed::for_chat` for a chat (`AgentToolConfig::chat`), the user's unanswered messages, with their images | The hook sees the task with the old conversation. |
| `AgentRuntime::with_handover_threshold` | `HandoverThreshold` unset, the configured threshold | The host's threshold, larger or smaller than the configured one, read at each check, so a change applies from the next one. Not saved with the task, and clones share it. The host keeps it under the model's input limit. |

Checked before each model request (not a refusal fallback's) against the estimated input of that request (the latest request's reported input, plus the reply, at least its reported output tokens, and the turn's results, at the bytes-per-token ratio calibrated from the task's own requests. The overflow guard's calibrated estimate, not the 10-bytes-per-token lower bound it decides by), and only where a user message can go. A turn whose results take the conversation over the threshold hands over before the request that would carry them. The turn boundary's context update (`InjectContext` context, a turn context source's text, the file-tree refresh) is added after this check and after the overflow guard, so neither counts it, it shows in the next request's reported input, one request late. In arko_workflows that update carries the start and finish notices, a graceful stop's wrap-up notice, messages, and at most one update's operator notes, about 16 KiB (`PROMPT_MAX_BYTES`, a single longer note is shown whole, up to `MAX_NOTE_BYTES` plus its label and two spaces of indentation a line). `HandoverRequested` and `HandoverRecord::input_tokens` record the latest request's reported input. The Claude Code runtime checks the latest request's input. With native tool calling and `handover` set, every request offers the `handover_report` tool (`HANDOVER_REPORT_TOOL_NAME`, `report_tool_definition`), and the report is its `report` argument. A reply without a tool call gets a note, a call when no report is due or without a report is rejected, and the conversation is replaced at the next turn boundary. Other agents give the report as a reply without a tool call, and a reply without text gets a note. During the report turn every other tool call except `complete_task` is rejected. The fresh conversation keeps the system prompt, the pending context injections and the task's usage. Tools observing the conversation are reset, the file tree is sent in full again, and the cache monitor judges only the fresh conversation. `initial_task_prompt` is unchanged. `UserCommand::RequestHandover` starts one without a threshold. A request still over the model's input limit after the overflow guard's recovery hands over too, from the last answered request and without the step that took it there (see [Tool result size](#tool-result-size-arko_agents)). A Claude Code agent that hands over does the same when a turn goes over the limit. The next session resumes the Claude Code session at the message that ended the last request Claude Code answered (the CLI's `--resume-session-at`, at the UUID of the user message arko sent or of the tool result Claude Code reported), which leaves out the model's reply, its results and the runtime's messages after them, and the report request there starts with `STEP_LEFT_OUT`. Context updates delivered since go with it. A step is left out only when tool results came after the last answered request, and once per handover, so a report request that goes over the limit as well ends the run on the error. See [Conversation handover](arko_agents-guide.md#conversation-handover). With `compact_in_claude_code`, a Claude Code agent has Claude Code compact the conversation in its session instead, at the threshold its next tool call is held and its turn interrupted, `/compact` is sent between turns (first, when a session starts over the threshold, and at once after a turn that went over the model's input limit), and the agent continues from Claude Code's summary with a message naming the held calls, followed by `HandoverHook::on_compaction`'s context. Without a `compact_boundary` before the `/compact` turn ends, it hands over instead, and when that report request goes over the model's input limit too, the step that went over it is left out as above. Every compaction Claude Code reports, its own included, emits `ConversationCompacted`, resets the tools observing the conversation and starts `conversation_usage_start` over. See [Claude Code models](arko_agents-guide.md#claude-code-models).

### Refusals (arko_agents)

Model-call recovery does not automatically retry a refusal unchanged. `RefusalReceived` and `RefusedInput` are emitted, and the refusal is handled as an error that can't be retried by the agent's own recovery. The task pauses (cause `Refusal`) or is returned `Interrupted`, as the error policy decides. A model's own refusal (`RefusalSource::Model`) stays in history without its tool calls. Classifier and content-filter refusals are removed. `UserCommand::ResolveRefusal` (while the task awaits recovery approval) or `Task::resolve_refusal` (on a returned task) changes the refused input. `ApproveErrorRecovery` (which adds its note) and `ContinueAfterError` continue without changing it. With the opt-in `refusal_fallback`, the same input is instead sent once to the fallback model (`RefusalFallback`). The refused turn is not kept, and only the fallback's own refusal makes the refusal stand. See [Refusals](arko_agents-guide.md#refusals).

### Runtime (arko_agents)

`AgentRuntime::new(client).run_task(task, tool_registry, event_throttle_ms, simplified_tool_events, skill_prompt_context, interceptor_config, execution_middleware)` returns the event receiver, the command sender, and a join handle that resolves to `Result<Task, ArkoAgentsError>`.

| Parameter or setter | Unset means | Notes |
|---|---|---|
| `event_throttle_ms` | `None` or `0`. Every text and reasoning chunk is its own event | `Some(ms)` coalesces chunks. |
| `simplified_tool_events` | Required `bool` | Output-based mode only, `true` leaves out `ToolParamName`, `ToolParamValueChunk` and `ToolParamEnd`. |
| `skill_prompt_context` | No skills in the system prompt | From `SkillManager::setup`. |
| `interceptor_config` | No result interception | `InterceptorConfig { interceptor, failure_policy }`. |
| `execution_middleware` | None | `before` and `after` every tool execution. For output supplied with an approval, only `after` (with `CallerProvided`). |
| `AgentRuntime::with_turn_context(source)` | No turn context | Asked before each model request (not a refusal-fallback request). Its text joins that turn's context update. |
| `AgentRuntime::with_host_decisions(decisions)` | `HostDecisions` `false`, waits are timed as configured | While `true`, `user_command_timeout_seconds` times no wait of the run. Not saved with the task. Clones share the setting. |
| `AgentRuntime::with_task_snapshots(sink)` | No snapshots. The task is returned only when the run ends | The sink receives the task at its safe points. Before tool calls start the run waits until it is saved. A sink error is logged and the run goes on. |
| `AgentRuntime::with_stop_signal(signal)` | A signal that never fires | Reaches the tasks' tool calls as part of `TaskContext::stop`. See [Stopping tool calls (arko_agents)](#stopping-tool-calls-arko_agents). |
| `AgentRuntime::with_background_jobs(manager)` | No call moves to the background | The calls of tools whose `background_mode` isn't `Never` can move. The tasks are bound to the manager for its notes. See [Moving calls to the background](arko_agents-guide.md#moving-calls-to-the-background). |

The event channel holds 100 events and the command channel 10. A host that stops reading events blocks the run. A host that drops the event receiver ends it with `ArkoAgentsError::ChannelSend`, losing the task value. When every command sender is dropped, the task is returned `Interrupted` at its next wait or turn boundary (tools hold only weak senders).

- `InjectContext` is added at the next turn boundary as one `# Workflow context update` user message, with all context sent before it, in order. It fires the `TaskContext::context_arrived` of the calls running, unless the turn context source's `wakes_calls` says it doesn't. Context taken before the task stops is added before it is returned, or kept in `Task::pending_context_injections` when a user message can't go there yet. Context that reaches the channel after the run's last read of it is taken as the run ends. Once the run has ended, sending fails, so a terminal manager keeps its note for the owner's next runtime.
- `CancelTask` ends the task `Cancelled` with `CANCELLED_BY_USER`. `SuspendTask` returns it in its current state, to be saved and run again. Either fires the stop signal of the tool calls running then (`TaskContext::stop`).
- `ApproveErrorRecovery` adds a note that the user approved continuing (not after a monitor pause or for a pending refusal fallback). `ContinueAfterError { note: None }` resends the failed request unchanged, or after a tool-error pause sends the results that were waiting. `RejectErrorRecovery` fails the task.
- In `SubmitBatchToolApprovals`, a call without a decision is rejected.
- A command the current state doesn't take is ignored.

See [Running tasks](arko_agents-guide.md#running-tasks).

### Skills, MCP and terminals (arko_agents)

| Setting | Default | Notes |
|---|---|---|
| `SkillsConfig::workspace_source` | `None` | A `SkillSource { path, include, exclude }` whose `path` is joined to the `workspace_root` given to `SkillManager::load`. |
| `SkillsConfig::global_sources` | Empty | Paths are used as written. `~` is not expanded. Identical skills load once. Another skill with a name already taken gets a numeric suffix. |
| `SkillsConfig::embedded` | Empty | `EmbeddedSkill` values, loaded after the folder sources. A folder skill with the same name replaces an embedded one. They have no resources. A saved configuration leaves them out. A workflow's continued, resumed, recovered and kept-warm runs take those of the agent's registered configuration when their task holds none, as a task loaded from a save does (a reconfigured run only when its own skills configuration is set), and `resume_from` doesn't compare them. A subagent has its own agent's. |
| `SkillsConfig::on_invalid_skill` | `Warn`. An invalid skill (an embedded one too), or a missing directory, is skipped | `Error` makes `SkillManager::load` fail. |
| `SkillsConfig::load_skill_interception` | `None`, `load_skill` results are never intercepted | `Some(policy)` lets the runtime's interceptor replace them, under `policy`. |
| `McpServerConfig::args`, `McpServerConfig::env` | Empty | The server inherits the host's environment, plus `env`. |
| `TerminalConfig::new(directory)` | Shell `/bin/sh -c` (Windows `cmd.exe /D /S /C`). `output_buffer_bytes` 65,536. `max_log_bytes` 4 MiB. `excerpt_bytes` 2,000. `prompt_quiet_ms` 150. `default_timeout` `None`. `prompt_rules` for yes/no, press-a-key and password prompts. `quiet_terminal_environment` `true`. `command_guard` on (`CommandGuard::default()`) | The directory holds job records and logs and is locked by one manager. PTY jobs get `TERM=dumb`, `NO_COLOR=1`, `PAGER=cat`, `GIT_PAGER=cat` and a UTF-8 locale. The number of jobs is not limited. The command guard refuses commands and input that would delete, move, overwrite or reformat protected paths (only `dangerously_allow_all_commands` allows them) or paths outside the working directory (`allow_outside_workspace`, `allowed_paths`). It has no command rules (`rules` empty). |

Skills load only when the host calls `SkillManager::load` and then `setup` on the configuration the task will run with (before `Task::new`, which copies it, or on a loaded task's `agent_config`). arko_workflows does this for its agents. When there are on-demand skills, `setup` registers `load_skill`, adds it to `available_tool_names`, and auto-approves it unless the agent already has a per-tool entry for it.

`MonitorLimits::default()` allows 5 running monitors per owner and 20 in all, for up to 30 minutes each, warns 5 minutes before the end (monitors armed for 10 minutes or more), and sends an owner at most one note of monitor news a minute. Warnings wake an idle agent for 2 hours after its user last wrote (`DEFAULT_WARNINGS_WAKE_WITHIN`). The monitor tools are registered with the terminal tools and offered only when listed. See [Monitors](interactive-terminals.md#monitors).

MCP tools are named `mcp__<server>__<tool>` and return text only, images are dropped, and a result without text gives its structured content as JSON. They have no `is_path` parameters, so the workspace path check doesn't apply to them. `McpManager::register_tools` doesn't offer them (add `tool_names()` to `available_tool_names`), and only the host tool time limit bounds their calls. A server's stderr goes to the log at info level. A server that exits logs one warning, and its tools then fail as tool errors. Terminal tools are registered with `TerminalManager::register_tools` and must be listed in `available_tool_names` too. See [MCP servers](arko_agents-guide.md#mcp-servers), [Skills](arko_agents-guide.md#skills), and [Configuration](interactive-terminals.md#configuration).

### Persisted data (arko_agents)

- Tasks, configurations and tool definitions (`save_to_path`) are saved as pretty JSON through a temporary file and a rename. Tasks carry no format version. Their optional fields default, so tasks saved by earlier versions load.
- A task's configuration snapshot without a tool calling mode loads as `OutputBased`, the mode it ran with. A standalone configuration without one loads as `Native`.
- A task runs with the configuration and system prompt saved in it, editing the agent's configuration doesn't change a saved task.
- `Failed { reason: "Task cancelled by user." }` loads as `Cancelled`. A pause saved without a cause loads with `ErrorRecoveryCause::Unknown`. A call saved without an approval reason loads with an unrecorded one. A task paused by a budget hold is saved with the cause `budget_exceeded`, which versions before it can't read.
- `allow_batched_tool_calls` is ignored (`false` logs a warning once). A per-tool `allow_batched_calls: false` loads as `parallel_safe: Some(false)` and `true` is ignored. A tool definition's `supports_batched_calls` is ignored.
- Dynamic environment-detail providers aren't saved, set them again on a loaded task.
- Terminal job records left `running` by a stopped host load as `interrupted`. Processes are never reattached. A monitor's record keeps what it watched, its end and its count of alerts, and a monitor never starts again after a restart. Records saved before monitors load as other jobs, and versions before them read a monitor's record as a command's.
- A task saved without `parent` has none (it ran at the top level).
- A task saved without `running_tool_calls` has none. One saved with some (by a host that crashed while its calls ran) settles them by each tool's `recovery_mode` before it goes on.
- A pause, failure or `ErrorEncountered` event saved without `error_kind` loads with `None`, and `None` is not written, so older versions read new tasks as before. A kind a later version adds loads as `Unknown`. A configuration saved without `result_limits`, `read_file_line_ranges`, `text_reply_as_message` or `text_reply_ends_task` loads with the defaults (the guard on, no caps, no ranges, text replies neither messages nor endings), which are not written either, so a resumed run compares equal.

### Example binaries (arko_agents)

`arko_agents_cli` loads a `.env` file, then reads provider credentials from the environment (`LLMClient::from_env_with`).

| Flag | Default | Notes |
|---|---|---|
| `-c`, `--agent-config`. `-w`, `--workspace-dir`. `-p`, `--prompt` | Required | The workspace is created if missing, then canonicalized. |
| `--tool-calling-mode` | The configuration's | `native`, or `output_based` (also `outputbased`, `output-based`). |
| `--event-throttle-ms` | Unset, no coalescing | |
| `--simplified-tool-events <true\|false>` | `true` | |
| `--user-command-timeout SECONDS` | The configuration's | |
| `--mcp-config PATH` | No MCP | Adds the servers' tools to the agent's tools. With at least one server, switches the agent to native mode. |
| `--openai-auth-json PATH` | Environment credentials | ChatGPT OAuth from a Codex `auth.json` (`with_codex_auth_file`). The `OPENAI_OAUTH_*` variables are then not read. |
| `--save-task-state PATH` | Not saved | The final task, when the run returns it. |

The CLI asks at the terminal for every tool approval (a batch call by call), every `ask_user` question, and every error-recovery pause (approve or reject). It loads the configuration's skills and continues without them if loading fails. `examples/coder_agent.json` uses Anthropic `claude-opus-5-5` in native mode (it names no mode), asks approval for every call, and uses `PauseAndAwaitApproval` for both tool and model errors. `examples/native-tools-agent.json` uses OpenAI `gpt-6-sol` and auto-approves every tool. `interactive_terminal` takes no arguments. `ARKO_TEST_PYTHON` selects the Python interpreter (default `python3`, or `python` on Windows).


## arko_workflows

Multi-agent workflow orchestration on top of arko_agents, application code runs agents in order, in parallel or in the background, with shared memory, messaging, budgets, hooks, approval and error-recovery policies, context freshness, completion verification, distillation, durable steps and checkpoints.

See [arko_workflows guide](arko_workflows-guide.md), [API index](../arko_workflows/INDEX.md), [DeepCoder example](../arko_workflows/examples/deepcoder/README.md), [memory reads](context-efficiency.md#memory-reads) and [tool-result distillation](context-efficiency.md#tool-result-distillation) in the context management guide.

### Map (arko_workflows)

Agents are named by their registration name (the one given to `WorkflowBuilder::agent`) everywhere, policies, lists, hooks, results and tools.

**Crate**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `arko_workflows` modules | `budget`, `context`, `distillation`, `durable`, `error`, `freshness`, `hooks` (`hooks::context`), `memory` (`memory::tools`), `messaging` (`registry`, `tools`, `wait`), `notes` (`notes::control`), `persistence` (`persistence::state`), `policy`, `services`, `user_input`, `verification`, `voting`, `workflow` (`cancel`, `config`, `engine`, `result`, `stop`). `python` with the `python` feature. `workflow::engine` exports no items. | `arko_workflows/src/lib.rs` | [Imports and feature map](../arko_workflows/INDEX.md#imports-and-feature-map) |
| Root re-exports | `Workflow`, `WorkflowBuilder`, `AgentHandle`, `WorkflowError`, `WorkflowCancelHandle`, `CancelCause`, `Cancellation`, `SupportedPlatform`, `WorkflowDependency`, `WorkflowMetadata`, `AgentResult`, `AgentRunStatus`, `TaskInfo`, `WorkflowEvent`, `WorkflowResult`, and from `workflow::stop`, `StopKind`, `StopRequest`, `WrapUp`, `StopRecord`, `StoppedRun`, `Restart`, `RunStopIntent`, `stop_on_signals`. | `arko_workflows/src/lib.rs` | [Imports and feature map](../arko_workflows/INDEX.md#imports-and-feature-map) |
| `prelude` | The root re-exports except `AgentHandle`, plus each module's configuration and data types (`CheckpointRetention` included). Not in it, `SharedContext`, `HookContext`, `HookRegistry`, `MemoryStore`, `MessageBus`, `BallotStore`, `DurableStore`, `PersistenceManager`, `FreshnessSnapshot`, the checkpoint state types, the tool types, and the constants other than `VERIFIER_OUTPUT_CONTRACT`. | `arko_workflows/src/lib.rs` | [Imports and feature map](../arko_workflows/INDEX.md#imports-and-feature-map) |
| Feature `python` | Off by default. Compiles the PyO3 bindings. The crate builds no `cdylib` and has no `pyproject.toml`, so it is not packaged as a Python extension. | `arko_workflows/Cargo.toml` | [Python bindings](arko_workflows-guide.md#python-bindings) |

**Building a workflow**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::builder`, `WorkflowBuilder::new` (`Default`) | An empty builder. | `arko_workflows/src/workflow/mod.rs` | [Builder signatures and defaults](../arko_workflows/INDEX.md#builder-signatures-and-defaults) |
| `WorkflowBuilder::name`, `WorkflowBuilder::workspace`, `WorkflowBuilder::llm_client`, `WorkflowBuilder::agent` | The required settings, workflow name, existing workspace directory, shared `Arc<LLMClient>`, and at least one agent under a unique registration name. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::memory` | Memory tiers, from a `MemoryConfig`. | `arko_workflows/src/workflow/mod.rs` | [Memory](#memory-arko_workflows) |
| `WorkflowBuilder::budget`, `WorkflowBuilder::budget_warning`, `WorkflowBuilder::agent_budget` | Workflow spending limit, one-time warning threshold, and one agent's cumulative limit, in dollars. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::hook` | Registers a synchronous `HookCallback` at a `HookPoint`. | `arko_workflows/src/workflow/mod.rs` | [Hook registration and results](../arko_workflows/INDEX.md#hook-registration-and-results) |
| `WorkflowBuilder::persistence`, `WorkflowBuilder::ephemeral` | Turns on checkpoints, task and event files, a persistent run ID and durable steps. `ephemeral` clears the persistence and resume settings made so far. | `arko_workflows/src/workflow/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `WorkflowBuilder::resume_from`, `WorkflowBuilder::rename_memory_tier_on_resume`, `WorkflowBuilder::drop_memory_tier_on_resume`, `WorkflowBuilder::keep_saved_configs` | Restores a checkpoint file at `build`. Maps a checkpoint memory tier to a configured one, or discards it. `keep_saved_configs(agents)` lets those agents' saved runs keep their own configuration while new runs take the registered one, checking only that the saved runs' tools are still there. | `arko_workflows/src/workflow/mod.rs` | [Resume](#resume-arko_workflows) |
| `WorkflowBuilder::metadata` | Sets the `WorkflowMetadata`. | `arko_workflows/src/workflow/mod.rs` | [Workflow metadata](../arko_workflows/INDEX.md#workflow-metadata) |
| `WorkflowBuilder::context_freshness` | Turns on context freshness tracking with a `ContextFreshnessConfig`. | `arko_workflows/src/workflow/mod.rs` | [Context freshness](#context-freshness-arko_workflows) |
| `WorkflowBuilder::global_tool`, `WorkflowBuilder::agent_tool` | Adds an `arko_agents` `Tool` for every agent, or for one agent, and adds its name to that agent's available tools. | `arko_workflows/src/workflow/mod.rs` | [Build validation and tool composition](../arko_workflows/INDEX.md#build-validation-and-tool-composition) |
| `WorkflowBuilder::tool_result_interceptor`, `WorkflowBuilder::agent_tool_result_interceptor`, `WorkflowBuilder::interception_failure_policy` | Tool-result interceptors for every agent or one agent, and the `InterceptionFailurePolicy` applied to each stage of an agent's interceptor chain. | `arko_workflows/src/workflow/mod.rs` | [Build validation and tool composition](../arko_workflows/INDEX.md#build-validation-and-tool-composition) |
| `WorkflowBuilder::agent_handover_threshold` | Gives one agent's runs a `HandoverThreshold` the host can change while they run. | `arko_workflows/src/workflow/mod.rs` | [Conversation handover](../arko_workflows/INDEX.md#conversation-handover) |
| `WorkflowBuilder::completion_verifier`, `WorkflowBuilder::distiller` | Adds a completion verifier or a tool-result distiller. | `arko_workflows/src/workflow/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `WorkflowBuilder::workflow_services` | Enables agent-to-agent delegation through `run_workflow_agent`. | `arko_workflows/src/workflow/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::messaging`, `WorkflowBuilder::disable_messaging` | Sets whom each agent may message (`MessagingPolicy`), or offers no agent the messaging tools. | `arko_workflows/src/workflow/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::wait_for_message_timeout`, `WorkflowBuilder::agent_wait_for_message_timeout` | The `wait_for_message` time limit, for the workflow or one agent. | `arko_workflows/src/workflow/mod.rs` | [Waiting for a message](../arko_workflows/INDEX.md#waiting-for-a-message) |
| `WorkflowBuilder::user_input_handler` | Answers agents' `ask_user` questions in-process. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::approval_policy`, `WorkflowBuilder::error_recovery_policy` | How tool-approval requests and error-recovery pauses are answered. | `arko_workflows/src/workflow/mod.rs` | [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows) |
| `WorkflowBuilder::max_parallel_agents`, `WorkflowBuilder::fail_fast` | Concurrency cap and fail-fast for `run_agents_parallel`. | `arko_workflows/src/workflow/mod.rs` | [Builder defaults](#builder-defaults-arko_workflows) |
| `WorkflowBuilder::enable_voting` | Creates the ballot store and offers `cast_vote` to the named agents. | `arko_workflows/src/workflow/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::cancel_grace_period` | How long a stopped run waits for each running agent before aborting it. | `arko_workflows/src/workflow/mod.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `WorkflowBuilder::build` | Validates the settings, restores a checkpoint when set, and fires `PreWorkflow`. Returns `BuilderMissing` or `BuilderInvalid` for a bad configuration. Runs no agent. | `arko_workflows/src/workflow/mod.rs` | [Build validation and tool composition](../arko_workflows/INDEX.md#build-validation-and-tool-composition) |
| `WorkflowMetadata` (`description`, `version`, `supported_platforms`, `required_dependencies`, `tags`, `missing_dependencies`), `SupportedPlatform` (`os`, `arch`, `new`, `arch()`, `matches_host`), `WorkflowDependency` (`name`, `description`, `executable`, `install_hint`, `os`, `arch`, `optional`, `new`, `executable()`, `install_hint()`, `os()`, `arch()`, `optional()`) | Serializable discovery metadata. Every `WorkflowMetadata` field may be omitted in JSON, and `optional` defaults to `false` (required). A nonempty `supported_platforms` list that the host matches none of fails `build`. The builder doesn't check dependencies, `missing_dependencies` reports those that apply to the host and whose `executable` isn't found on `PATH` (or as a file), required and optional alike, for the workflow to act on at startup. | `arko_workflows/src/workflow/config.rs` | [Workflow metadata](../arko_workflows/INDEX.md#workflow-metadata) |
| `workflow::config::WorkflowConfig` | A serializable descriptor (`name`, `workspace`, `metadata`). Workflows are built with the builder, and it is not re-exported. | `arko_workflows/src/workflow/config.rs` | [Workflow metadata](../arko_workflows/INDEX.md#workflow-metadata) |

**Running agents**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow` | One run of a configured workflow, created by `build`. | `arko_workflows/src/workflow/mod.rs` | [Workflow execution and lifecycle](../arko_workflows/INDEX.md#workflow-execution-and-lifecycle) |
| `Workflow::run_agent` | Starts a new run of a registered agent and drives it to its result. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::continue_agent`, `Workflow::continue_agent_with`, `ContinueOptions` (`images`, `config`, `cut_back_to`, `workspace`) | Continues a `Completed` or `WaitingForInput` result, or one whose completion its verifier couldn't review, with a prompt, keeping the task and run ID. `continue_agent_with` sends images after the prompt, refused up front when the agent's model can't take them (`Model::image_refusal`), and with `ContinueOptions::reconfigure(config)` continues the run under another configuration (same model, the workflow's tools added as for a new run, none to a chat), which the task keeps across a resume. `ContinueOptions::cut_back_to(index)` first cuts the conversation back to just before a user turn (edit, regenerate), for a run of any status that isn't running. The notes the removed messages gave are given again, except one-time notes, and the subagents started in the removed messages are detached, their mail withheld, and those still running stopped. `ContinueOptions::workspace(dir)` continues in another directory (a Claude Code session stays in its own, `Task::move_to`). |
| `Workflow::import_agent_run(agent_name, task)` | Adds a completed task (a branch, `Task::branch`) as a run of the agent, to continue like any completed run. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::refresh_prompt_cache(run)`, `AgentRuntime::refresh_prompt_cache(task, registry, skills)`, `CacheRefresh` (`usage`, `output_cap`, `resent_last_request`, `refused_cap`), `LLMClient::openai_uses_chatgpt_oauth` | Keep an idle run's prompt cache warm, its next request plus a short ping, output capped (Anthropic 1 token, OpenAI 16, none on ChatGPT OAuth), reply discarded. A run paused with unanswered calls sends its last request again unchanged. A refused cap is raised once to 1,024 and reported. Claude Code goes through an unsaved fork. The workflow refuses it over budget, charges the cost to the agent with the budget hooks, and saves it. Unsupported for Gemini, OpenRouter models other than Anthropic's and Anthropic's with reasoning on, and manual Anthropic thinking budgets. | `arko_agents/src/runtime/mod.rs`, `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::resume_agent` | Continues a `Paused` result with an `AgentResume` decision. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::run_agents_parallel` | Runs `(agent, prompt)` pairs concurrently (an agent may appear more than once). Returns the results in input order, or the first failure. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::agent_result` | The recorded result of a run, by the task ID an error carries. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::run_agent_with`, `Workflow::spawn_agent_with`, `RunOptions` (`workspace`, `images`) | `run_agent` and `spawn_agent` with a working directory of the run's own (its tools work there, its path checks allow it, and it is saved with the task), and with images after the prompt in the first message (`Task::initial_task_images`), refused up front when the agent's model can't take them. | `arko_workflows/src/workflow/options.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::spawn_agent`, `AgentHandle` (`agent_name`, `wait`, `into_parts`, `Drop`) | Starts an agent without driving it. `wait` drives it like `run_agent`. `into_parts` hands the event receiver, command sender and join handle to the caller. Dropping the handle unused aborts the agent and records no result. | `arko_workflows/src/workflow/mod.rs` | [AgentHandle](../arko_workflows/INDEX.md#agenthandle) |
| `Workflow::result`, `Workflow::subscribe_events`, `Workflow::subscribe_events_lossless`, `EventReceiver`, `Workflow::agent_status` | A `WorkflowResult` snapshot, a broadcast of `WorkflowEvent`s (8,192 buffered, no replay, a lagging receiver misses the oldest), every event in order for a receiver of its own (a lagging one's streamed chunks wait joined, nothing is lost), and a registered agent's `AgentStatus`. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `Workflow::pause`, `Workflow::resume`, `Workflow::is_paused` | The pause flag, managed drivers stop before their next event, and model and tool work in progress continues. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `Workflow::finish` | Fires `PostWorkflow`, saves a checkpoint and returns the result. On a stopped run, saves a checkpoint and returns the stop's error. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |

**Stopping a run**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::cancel_handle`, `Workflow::cancel`, `Workflow::is_cancelled`, `Workflow::cancellation` | Stops the run, or reports whether and why it stopped. | `arko_workflows/src/workflow/mod.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `Workflow::stop(StopRequest)`, `Workflow::is_stopping` | A graceful stop, with its kind and reason, after an optional wrap-up in which no run starts (only a completion verifier's review, of a completion made before the agent was told), tools see their stop signal and agents are told to save their progress and call `complete_task` (that completion is not reviewed). Then the run is cancelled with `CancelCause::Stopped`. `is_stopping`. A wrap-up is under way. | `arko_workflows/src/workflow/mod.rs` | [Graceful stop](../arko_workflows/INDEX.md#graceful-stop) |
| `StopRequest` (`kind`, `reason`, `wrap_up`, `new`, `operator`, `drain`, `limit`, `failure`, `wrap_up`, `wrap_up_message`), `WrapUp` (`within`, `message`), `StopKind::{Operator, Drain, Limit, Failure}` (`as_str`), `workflow::stop::wrap_up_notice` | What a stop asks for. The default is no wrap-up (cancel at once). The default notice asks agents to save their progress and call `complete_task`. | `arko_workflows/src/workflow/stop.rs` | [Graceful stop](../arko_workflows/INDEX.md#graceful-stop) |
| `AgentResult::stopped_by`, `StopRecord` (`kind`, `reason`, `timestamp`, `intent`, `is_retired`) | The stop that ended a run, cancelled by a stop, a cancel or its own stop, or completed after its wrap-up notice (unreviewed by completion verifiers). A run stopped alone carries its `intent`, also when interrupted in its wrap-up. A stopped run is blameless. | `arko_workflows/src/workflow/result.rs`, `arko_workflows/src/workflow/stop.rs` | [Graceful stop](../arko_workflows/INDEX.md#graceful-stop) |
| `Workflow::stopped_runs`, `Workflow::restart(run, note)`, `Workflow::restart_with(run, note, options)`, `StoppedRun` (`result`, `restart`), `Restart::{Recover, Reopen, Continue}` | After `resume_from`, or once a run stopped alone to relaunch has ended, the runs to relaunch (stopped ones, and the interrupted runs) and how. `restart` recovers, reopens or continues each. Retired runs are never listed, and `restart` refuses them. A run started inside the call of a listed run is left to it, restarted, that run resumes it (its failed call named the run). `restart_with` sends `ContinueOptions::images` with the note of a reopened or continued run, and refuses them for a recovered one, which takes no message. | `arko_workflows/src/workflow/mod.rs`, `arko_workflows/src/workflow/stop.rs` | [Graceful stop](../arko_workflows/INDEX.md#graceful-stop) |
| `Workflow::stop_run(id, request, intent)`, `Workflow::stop_agent(name, request, intent)` (also on `WorkflowCancelHandle`), `RunStopIntent::{Retire, Relaunch}`, `workflow::stop::run_wrap_up_notice`, `WorkflowError::AgentStopped { agent, task_id, reason }` | Stop one agent run, or an agent's running runs, while the workflow runs on, with a `StopRequest` as for `stop`. After the optional wrap-up (tools see their stop signal, the run is told to save its progress and call `complete_task`, unreviewed), the run is cancelled within the grace period (`AgentStopped`, not a failure). Runs inside its tool calls are stopped with it. The intent is saved on the result and with a running run in the live checkpoint. A retired run is never relaunched, one to relaunch is listed by `stopped_runs`. A delegated run's caller gets its report and a note saying whether it can resume it. | `arko_workflows/src/workflow/cancel.rs`, `arko_workflows/src/workflow/stop.rs` | [Stopping one agent run](../arko_workflows/INDEX.md#stopping-one-agent-run) |
| `stop_on_signals(handle, request)` | Ctrl-C and `SIGTERM` (Ctrl-C and Ctrl-Break on Windows) stop the run with `request`. A second signal ends a wrap-up at once. A third kills the `StopAwareCommand` children (on Unix, `kill_stop_aware_commands`) and exits the process. | `arko_workflows/src/workflow/stop.rs` | [Graceful stop](../arko_workflows/INDEX.md#graceful-stop) |
| `WorkflowCancelHandle` (`cancel`, `stop`, `is_cancelled`, `is_stopping`, `cancellation`, `cancelled`) | A clonable handle usable from any task or thread. `cancel` stops the run and fires `OnWorkflowCancelled`, or returns `false` when the run is already stopped. `stop` is `Workflow::stop`. `cancelled` resolves once it is stopped. | `arko_workflows/src/workflow/cancel.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `CancelCause` (`Requested`, `AbortedByHook { hook_point, reason }`, `Stopped { kind, reason }`, `to_error`, `agent_reason`), `Cancellation` (`cause`, `timestamp`) | Why and when the run stopped. Recorded in `WorkflowResult` and in checkpoints. A `Stopped` cause is saved as `Requested` with a `stop` field beside it, which older versions ignore. | `arko_workflows/src/workflow/cancel.rs` | [Stopping a run](../arko_workflows/INDEX.md#stopping-a-run) |
| `workflow::cancel::DEFAULT_CANCEL_GRACE_PERIOD` | 30 seconds. | `arko_workflows/src/workflow/cancel.rs` | [Builder defaults](#builder-defaults-arko_workflows) |

**Approvals and error recovery**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ApprovalPolicy` (`ApproveAll`, `RespectAgentConfig`, `Handler`) | How calls an agent asks approval for are answered. Default `ApproveAll`. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows) |
| `ToolApprovalHandler::decide`, `ToolApprovalRequest` (`workflow_run_id`, `agent_run_id`, `agent_name`, `tool_call_id`, `tool_name`, `parameters`, `tool_description`, `reason`), `ToolApprovalOutcome` (`Approve`, `Reject { reason }`, `Pause`) | Extension trait for `ApprovalPolicy::Handler`, gets each call awaiting approval, with its `ApprovalReason`, and approves, rejects or pauses it. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery) |
| `ErrorRecoveryPolicy` (`Automatic`, `Limited { max }`, `ReturnToCaller`) | How error-recovery pauses are answered. Default `Automatic`. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows) |
| `AgentPause` (`ToolApproval { calls }`, `ErrorRecovery { error, cause, error_kind }`, `of`, `kind`) | Why an agent paused, as `WorkflowError::AgentPaused` carries it. `kind` is `"tool_approval"` or `"error_recovery"`. `error_kind` is the model-request error's `arko_ai::ErrorKind`, when classified. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery) |
| `AgentResume` (`ToolApprovals`, `ContinueAfterError { note }`, `FailAfterError { reason }`, `ResolveRefusal`) | The decision `resume_agent` applies. A call without a decision is rejected. `ResolveRefusal` answers only a `Refusal` pause (or an `Unknown` one, saved before causes were recorded), with at least one change, of an agent not in Claude Code. | `arko_workflows/src/policy.rs` | [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery) |

**Results and events**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `AgentResult` (`agent_run_id`, `agent_name`, `status`, `summary`, `error`, `stopped_by`), `of_task`, `task()`, `info()`, `total_cost`, `total_input_tokens`, `total_output_tokens`, `failure_cause`, `failure_kind`, `stopped_by()`, `TaskInfo` (`agent_config`, `initial_task_prompt`, `start_time`, `end_time`, `parent`, `reconfigured`, `usage`, `model_calls`) | One run's outcome. `task()` returns its `arko_agents::Task`, which a result restored for a run that completed reads from its task file the first time, once for all clones. `info()` has what the task holds besides its conversation, without reading it. The totals cover the whole task, and `total_cost` is the billed amount. `failure_cause`, why a failed run failed at a rejected error-recovery pause (`NonRetryable`, an expired login, a bad key or model). `failure_kind`, that error's `arko_ai::ErrorKind` (`Authentication`, `ContextOverflow`, ...), when classified. Decide by it instead of matching error text. `stopped_by`. The stop that ended it (omitted from JSON when `None`). | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |
| `AgentRunStatus` (`Completed`, `Failed`, `Interrupted`, `Cancelled`, `WaitingForInput`, `Paused`), `is_paused`, `of_state` | A run's status, from its final task state. `is_paused` is true for `WaitingForInput` and `Paused`. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |
| `WorkflowResult` (`workflow_run_id`, `workflow_name`, `metadata`, `total_cost`, `cost_ledger`, `agent_results`, `cancellation`) | A snapshot of the run. Its results, total billed cost, ledger, and stop if any. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |
| `WorkflowEvent` (`timestamp`, `workflow_run_id`, `agent_name`, `agent_run_id`, `event`) | An `arko_agents` agent event with the workflow run ID, agent name and agent run ID, as broadcast and logged. | `arko_workflows/src/workflow/result.rs` | [Result and event types](../arko_workflows/INDEX.md#result-and-event-types) |

**Hooks and human input**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `HookPoint::PreWorkflow`, `HookPoint::PostWorkflow` | The end of a successful `build`. `finish`. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PreAgentSpawn` | Before a run, spawn, continuation or resume starts. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PostAgentCompletion`, `HookPoint::PostAgentFailure`, `HookPoint::PostAgentCancellation`, `HookPoint::PostAgentInterruption`, `HookPoint::OnUserInputRequired`, `HookPoint::OnAgentPaused` | A run's task was finalized as `Completed`, `Failed`, `Cancelled`, `Interrupted`, `WaitingForInput` or `Paused`. A failure at a rejected error-recovery pause has its cause in `metadata["cause"]`, and its error kind, when classified, in `metadata["error_kind"]` (`"authentication"`, `"context_overflow"`, ...). An error-recovery pause has both too. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PreToolExecution`, `HookPoint::PostToolExecution` | Before and after each tool call. `Skip` at `PreToolExecution` rejects the call. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::PostToolResultInterception` | The managed driver received an intercepted tool result. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnWorkflowPause`, `HookPoint::OnWorkflowResume` | The pause flag changed. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnWorkflowFailed` | A run call is about to return a failure (not a pause, and not `Cancelled` or `AgentStopped`). At most once per call. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnWorkflowCancelled` | A cancel handle cancelled or stopped the run (after a wrap-up). Its result is ignored. A graceful stop's has `stop_kind` and `reason` metadata. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnBudgetWarning`, `HookPoint::OnBudgetExceeded` | A recorded cost reached the warning threshold for the first time, or took spending above the limit (once per crossing under the default `Pause`, every charge over it under `Cancel`). | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnAgentMessage` | `send_message` queued a message. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookPoint::OnFreshnessConflict` | A tracked write made another active run's read stale. Once per stale reader. `"OnMesiConflict"` loads as this variant. | `arko_workflows/src/hooks/mod.rs` | [HookPoint variants](../arko_workflows/INDEX.md#hookpoint-variants) |
| `HookResult` (`Continue`, `Abort(String)`, `Skip`), `HookCallback` | A hook's answer. `HookCallback` is `Arc<dyn Fn(&HookContext) -> HookResult + Send + Sync>`. | `arko_workflows/src/hooks/mod.rs` | [Hook registration and results](../arko_workflows/INDEX.md#hook-registration-and-results) |
| `hooks::context::HookContext` (`timestamp`, `hook_point`, `workflow_run_id`, `workflow_name`, `agent_run_id`, `agent_name`, `tool_name`, `tool_call_id`, `tool_params`, `tool_result`, `original_tool_result`, `effective_tool_result`, `agent_summary`, `approval_reason`, `error_message`, `budget_spent`, `budget_limit`, `metadata`) | What a hook receives. The optional fields are set only at the points that fill them. | `arko_workflows/src/hooks/context.rs` | [HookContext fields](../arko_workflows/INDEX.md#hookcontext-fields) |
| `hooks::HookRegistry` (`new`, `register`, `get`, `has_hooks`) | The registered hooks by point (`SharedContext::hooks`). | `arko_workflows/src/hooks/mod.rs` | [Hook registration and results](../arko_workflows/INDEX.md#hook-registration-and-results) |
| `user_input::UserInputHandler::{respond, asked, open_questions}`, `UserInputRequest` | Extension trait that answers an agent's `ask_user` questions, at once or later. An `Err` ends the call with `WorkflowError::Channel`. | `arko_workflows/src/user_input.rs` | [UserInputHandler and UserInputRequest](../arko_workflows/INDEX.md#userinputhandler-and-userinputrequest) |

#### Workflow tools

Workflow tools (added to an agent's tools by the workflow)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `read_memory` (`memory::tools::ReadMemoryTool`, `READ_MEMORY_TOOL_NAME`) | For agents that can read a tier. Parameters `tier`, optional `full` and `max_chars`. Returns versioned JSON (`full`, `append`, `replace`, `unchanged` or `partial`). Its results are never intercepted or distilled (`allow_result_interception` is false). Per-task read baselines follow `reset_context` and `observe_context`. | `arko_workflows/src/memory/tools.rs` | [Memory tools and read JSON contract](../arko_workflows/INDEX.md#memory-tools-and-read-json-contract) |
| `write_memory` (`WriteMemoryTool`, `WRITE_MEMORY_TOOL_NAME`) | For agents that can write a tier. Parameters `tier` and `content`. Appends or replaces by the tier's write mode. | `arko_workflows/src/memory/tools.rs` | [Memory tools and read JSON contract](../arko_workflows/INDEX.md#memory-tools-and-read-json-contract) |
| `list_memories` (`ListMemoriesTool`, `LIST_MEMORIES_TOOL_NAME`) | For agents that can read or write a tier, lists the tiers it can access, with its access to each. | `arko_workflows/src/memory/tools.rs` | [Memory tools and read JSON contract](../arko_workflows/INDEX.md#memory-tools-and-read-json-contract) |
| `send_message` (`messaging::tools::SendMessageTool`, `SEND_MESSAGE_TOOL_NAME`) | For agents with a permitted recipient. Parameters `to`, `content`, optional `attachments_json`. Accepts only recipients that are running or waiting. | `arko_workflows/src/messaging/tools.rs` | [Messaging tools](../arko_workflows/INDEX.md#messaging-tools) |
| `list_recipients` (`ListRecipientsTool`, `LIST_RECIPIENTS_TOOL_NAME`) | Offered with `send_message`, the permitted recipients with their roles and current statuses. | `arko_workflows/src/messaging/tools.rs` | [Messaging tools](../arko_workflows/INDEX.md#messaging-tools) |
| `wait_for_message` (`messaging::wait::WaitForMessageTool`, `WAIT_FOR_MESSAGE_TOOL_NAME`, both re-exported from `messaging::tools`) | For agents another agent may message. Returns at a message, a start or finish of a peer, other context arriving for the run, a move to the background (`background_mode` is `Own`), its time limit, or a stop. The optional `timeout_seconds` only shortens it. Not subject to the agent's tool time limit (`uses_host_timeout` is false). | `arko_workflows/src/messaging/wait.rs` | [Waiting for a message](../arko_workflows/INDEX.md#waiting-for-a-message) |
| `run_workflow_agent` (`services::RUN_WORKFLOW_AGENT_TOOL_NAME`, `services::RESUME_RUN_PARAMETER`) | For allowed callers when services are enabled. Parameters `agent`, `prompt`, and optional `resume_run` (a run ID). Runs the target to its end and returns its completion summary, or with `resume_run` resumes that unfinished run of the target (one the caller's run started, or an orphaned one, a cancelled one is reopened) with `prompt` as its message, re-pointing its parent link to the call. Counts toward `max_active_calls` either way. A failed call names the run it ran when that run did not finish, and how to resume it. So does a successful call, after the summary, when the run completed after a stop's wrap-up notice (unreviewed). Not subject to the agent's tool time limit (`uses_host_timeout` is false). The run records its parent (`Task::parent`). Recovery mode `Resume`. A recovered call returns or recovers that run (waiting for another recovery of it) instead of starting another. | `arko_workflows/src/services/mod.rs` | [Workflow services and custom tools](../arko_workflows/INDEX.md#workflow-services-and-custom-tools) |
| `start_agent`, `message_agent`, `wait_for_agents`, `list_agents`, `stop_agent`, `link_agents` (`subagents::SUBAGENT_TOOL_NAMES`) | With `WorkflowBuilder::subagents`, callers start runs of their targets in the background (`start_agent` returns `r1` at once), list and stop them, and link them as peers. Callers and subagents message each other by run id and wait for reports and messages, which `wait_for_agents` returns whole (under about 40,000 bytes for an agent in Claude Code). A wait also ends when context arrives for the run or the host moves it to the background (`Own`). | `arko_workflows/src/subagents/tools.rs` | [Background subagents](../arko_workflows/INDEX.md#background-subagents) |
| `cast_vote` (`voting::CAST_VOTE_TOOL_NAME`) | For the agents named in `enable_voting`. Parameters `ballot_id`, `option` and `reasoning`. One vote per agent per ballot. | `arko_workflows/src/voting/mod.rs` | [The cast_vote agent tool](../arko_workflows/INDEX.md#the-cast_vote-agent-tool) |

**Memory**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `MemoryConfig` (`new`, `tier`, `global_tier`, `global_tier_with_limit`, `broadcast_tier`, `broadcast_tier_with_limit`, `tiers`, `build`) | Builds the memory tiers given to `WorkflowBuilder::memory`. | `arko_workflows/src/memory/mod.rs` | [Memory](#memory-arko_workflows) |
| `MemoryTier` (`name`, `content`, `readers`, `writers`, `write_mode`, `entries`, `max_bytes`), `text_of`, `history` | One tier, content, access, write mode, write history and optional byte limit. | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `AccessScope` (`All`, `Only(names)`, `None`), `allows` | Who may read or write a tier. | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `WriteMode` (`AppendOnly`, `FullEdit`) | Whether a write appends (after a newline) or replaces the content. | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `MemoryEntry`, `MemoryWrite` | One write with its text, as read back. One write as stored (text held once, as a range of the content or in the entry). | `arko_workflows/src/memory/mod.rs` | [Memory](../arko_workflows/INDEX.md#memory) |
| `MemoryStore` (`new`, `is_empty`, `read`, `read_versioned`, `read_window`, `entries`, `write`, `write_with_context`, `list_accessible`, `access_of`, `snapshot`) | The run's store (`SharedContext::memory`), access-checked reads and writes, without freshness checks or hooks. | `arko_workflows/src/memory/mod.rs` | [Memory store](../arko_workflows/INDEX.md#memory-store) |

**Messaging**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `MessagingPolicy` (`recipients`, `keep_for_finished`, `all`, `can_message`, `cannot_message`, `keep_messages_for_finished`, `permits`, `recipients_of`) | Whom each agent may message with `send_message`, never itself, and whether messages to a finished agent are kept for its next run. | `arko_workflows/src/messaging/mod.rs` | [Messaging policy](../arko_workflows/INDEX.md#messaging-policy) |
| `AgentMessage` (`id`, `from`, `from_task_id`, `to`, `content`, `attachments`, `timestamp`, `to_task_id`, `held_by`, `read_by`), `is_held_by`, `is_for`. `MessageAttachment` (`File`, `MemoryTier`, `Transcript`, `Text`) | A queued message and its typed references. | `arko_workflows/src/messaging/mod.rs` | [Messaging](../arko_workflows/INDEX.md#messaging) |
| `MessageBus` (`new`, `from_snapshot`, `roster`, `register_active`, `unregister_active`, `inject_task`, `send`, `drain`, `pending`, `pending_for`, `acknowledge`, `hand_over`, `snapshot`) | The run's message queues and live delivery (`SharedContext::messages`). `send` checks neither the policy nor statuses, and fires no hook. | `arko_workflows/src/messaging/mod.rs` | [Message bus](../arko_workflows/INDEX.md#message-bus) |
| `messaging::WORKFLOW_CONTEXT_SENDER` | `"workflow-context"`, the sender of context notices such as freshness notices. | `arko_workflows/src/messaging/mod.rs` | [Message bus](../arko_workflows/INDEX.md#message-bus) |
| `AgentStatus` (`NotStarted`, `Running`, `Waiting(WaitReason)`, `Finished(AgentRunStatus)`), `is_reachable`, `describe`. `WaitReason` (`RateLimited`, `Retrying`, `UsageLimit`, `AwaitingApproval`, `AwaitingInput`, `AwaitingMessage`, `Paused`) | An agent's status in the run, as `agent_status`, rosters and `list_recipients` report it. | `arko_workflows/src/messaging/registry.rs` | [Agent statuses](../arko_workflows/INDEX.md#agent-statuses) |
| `WaitForMessageTimeout` (`Auto`, `Fixed(Duration)`, `resolve`, `From<Duration>`), `DEFAULT_WAIT_FOR_MESSAGE_TIMEOUT` (270 s), `MAX_WAIT_FOR_MESSAGE_TIMEOUT` (24 h) | The `wait_for_message` limit setting and its bounds. | `arko_workflows/src/messaging/wait.rs` | [Waiting for a message](../arko_workflows/INDEX.md#waiting-for-a-message) |

**Operator notes**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::post_note`, `Workflow::notes`, `Workflow::note_image`, `Workflow::notes_handle`, `Workflow::skip_one_time_notes`, `notes::NotesHandle` (`post`, `notes`, `image`), `Note::images`, `notes::NoteImage` | Post guidance to the agents while the workflow runs, without touching their `AgentConfig` (so a resume still accepts their saved tasks), running agents get it at their next turn boundary, later runs in their initial prompt. Each is logged once and saved in checkpoints. A one-time note may carry images (its text may then be empty), each after a label naming the note, after the update's text. A standing note with images, or images an audience agent's model can't take, are refused. Once posted, a note's images are kept apart (a persisted run's in `images/` of its run directory) and `PostedNote::images` names them. The handle posts from any task. `skip_one_time_notes` keeps the one-time notes posted so far from a run the host continues in place of another. | `arko_workflows/src/notes/mod.rs` | [Operator notes](../arko_workflows/INDEX.md#operator-notes) |
| `notes::Note` (`text`, `audience`, `kind`, `source`, `new`, `to`, `except`, `once`, `source`), `NoteAudience` (`All`, `Only`, `Except`), `NoteKind` (`Standing`, `Once`), `NoteSource` (`Operator`, `Host`, `Agent(name)`), `PostedNote` (`id`, `seq`, `posted_at`, `note`, `images`) | A note, who gets it (registration names, completion verifiers' reviews and `notes_by_name_only` agents only when named), whether it stands for later runs or is given once, and the source the agents are told. | `arko_workflows/src/notes/mod.rs` | [Operator notes](../arko_workflows/INDEX.md#operator-notes) |
| `notes::MAX_NOTE_BYTES` (16 KiB), `PROMPT_MAX_NOTES` (20), `PROMPT_MAX_BYTES` (16 KiB) | The longest note accepted, and how many notes and about how many bytes a prompt shows (the newest standing notes, older ones left out) or a context update shows (the oldest new notes, the rest at the next turn boundary, so none is lost). The first always in full. | `arko_workflows/src/notes/mod.rs` | [Operator notes](../arko_workflows/INDEX.md#operator-notes) |
| `WorkflowBuilder::notes_by_name_only` | Agents that get a note only when it names them, as completion verifiers do, independent roles. | `arko_workflows/src/workflow/mod.rs` | [Operator notes](../arko_workflows/INDEX.md#operator-notes) |
| `WorkflowBuilder::notes_reply_hint(agent, hint)` | A line the host words, which ends the context updates that give `agent` a note when at least one note in the update is the operator's. Never in a prompt or a handover. | `arko_workflows/src/workflow/mod.rs` | [Operator notes](../arko_workflows/INDEX.md#operator-notes) |
| `WorkflowBuilder::control_folder`, `notes::control` (`post_note`, `read_notes`, `request_stop`, `read_stops`, `folder`, `ControlNote` (`id`, `at`, `note`), `ControlStop` (`id`, `at`, `run`, `agent`, `kind`, `reason`, `wrap_up_secs`, `intent`), `FOLDER`, `NOTES_FILE`, `STOPS_FILE`, `LOCK_FILE`, `STOPS_POLL_INTERVAL`) | Opt-in, another process posts notes to the run through `<run dir>/control/notes.jsonl`, appended under the folder's own lock. The workflow reads new lines at turn boundaries and agent starts. Stop requests go to `stops.jsonl`, which the workflow reads every second while agents run (only the lines written after it was built), stopping runs alone as `stop_run` and `stop_agent` do. | `arko_workflows/src/notes/control.rs` | [The control folder](../arko_workflows/INDEX.md#the-control-folder) |

**Services and voting**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `WorkflowServicesConfig` (`allowed_callers`, `allowed_targets`, `all_callers`, `all_targets`, `max_active_calls`, `unlimited_active_calls`, `queue_active_calls`, `allow_caller`, `allow_target`, `allow_all_callers`, `allow_all_targets`, `max_active_calls(n)`, `unlimited_active_calls()`, `queue_active_calls()`), `DEFAULT_MAX_ACTIVE_CALLS` | Which agents may call `run_workflow_agent`, which agents they may run, and the workflow-wide limit on calls at once (default 4), which can queue calls over it or be lifted. | `arko_workflows/src/services/mod.rs` | [Messaging, services and voting](#messaging-services-and-voting-arko_workflows) |
| `WorkflowBuilder::subagents`, `subagents::SubagentsConfig` (`max_running`, `max_depth`, `peer_messages`, `max_peer_messages`, `copy_peer_messages_to_lead`, `workspaces_anywhere`, `offer_blocking_tool`, `questions_to_lead`, `workspace_provider`, `hook`, `start_parameters`, `max_failed_requests`), `DEFAULT_MAX_RUNNING`, `DEFAULT_MAX_PEER_MESSAGES`, `MAX_MAIL_BYTES`, `UPDATE_MAX_MAIL_BYTES` | Background subagents for the services' callers, the cap and queue, nesting depth, messages between subagents, workspaces, questions to the lead, and the host's hook. | `arko_workflows/src/subagents/mod.rs` | [Background subagents (arko_workflows)](#background-subagents-arko_workflows) |
| `subagents::{SubagentHook, StartRequest}` | The host adjusts a subagent's `AgentConfig` as it starts, given the start and the host's own `start_agent` parameters. `describe` adds a line about a run to `start_agent`'s result and `list_agents`, `report_note` ends the news of a run that finished, failed or stopped with a line, and `offers_start_tools` says whether a subagent is offered the tools that start runs. | `arko_workflows/src/subagents/mod.rs` | [Background subagents](../arko_workflows/INDEX.md#background-subagents) |
| `subagents::{WorkspaceProvider, WorkspaceRequest, IsolatedWorkspace, GitWorktreeProvider}` | Isolated workspaces for `start_agent`'s `isolated`, created per run, kept with its record, removed with `stop_agent`'s `remove_workspace`. `GitWorktreeProvider` makes a git worktree on its own branch. | `arko_workflows/src/subagents/workspace.rs` | [Background subagents](../arko_workflows/INDEX.md#background-subagents) |
| `subagents::{SubagentRecord, SubagentState, PendingLaunch, MailItem, MailKind, MailSender, SubagentsSnapshot}` | A subagent's record and state, its mailbox items, and the state saved in checkpoints (`WorkflowState::subagents`). `stop_requested` preserves a stop before runtime registration and defaults to absent in older records. | `arko_workflows/src/subagents/mod.rs` | [Background subagents](../arko_workflows/INDEX.md#background-subagents) |
| `subagents::{SubagentEvent, SubagentEventKind}`, `Workflow::{subagents, active_subagents, subscribe_subagents, pending_news, take_news, resume_subagents, transfer_subagents, set_max_failed_requests, stop_subagent, notes_given}` | Subagent events (`NewsWaiting` tells the host an idle run has news), also in the event log, and the host's view and control of the subagents and their mail. `stop_subagent(run_id, request)` also stops queued runs and runs still starting, while allowing later messages to restart them, and stops the subagents the run started and theirs in turn, with the same kind and wrap-up. `resume_subagents` stops a subagent whose own lead won't run again instead of relaunching it. `set_max_failed_requests` changes the retry cap of running subagents. | `arko_workflows/src/workflow/mod.rs` | [Background subagents](../arko_workflows/INDEX.md#background-subagents) |
| `WorkflowServices` (`Clone`, `is_allowed`, `is_target_allowed`, `run_agent`) | The live services, from `SharedContext::workflow_services`. `run_agent` runs a target for a caller under the same checks as the tool. | `arko_workflows/src/services/mod.rs` | [Workflow services and custom tools](../arko_workflows/INDEX.md#workflow-services-and-custom-tools) |
| `Workflow::begin_ballot`, `Workflow::finish_ballot` | Opens a ballot among the workflow's voters, and tallies it without closing it. | `arko_workflows/src/workflow/mod.rs` | [Workflow-facing resource and ballot operations](../arko_workflows/INDEX.md#workflow-facing-resource-and-ballot-operations) |
| `BallotConfig` (`question`, `options`, `eligible_voters`, `quorum`, `consensus_threshold`, `weights`, `tie_break`), `VoteOption` (`id`, `description`), `TieBreak` (`RejectTie`, `Lexicographic`), `CastVote` (`voter`, `task_id`, `option`, `reasoning`, `timestamp`), `BallotResult` (`ballot_id`, `winner`, `consensus`, `votes`), `BallotStore` (`begin`, `finish`) | Ballot configuration, votes, results and the in-memory store (`SharedContext::ballots`). Ballots are not checkpointed. | `arko_workflows/src/voting/mod.rs` | [Structured voting](../arko_workflows/INDEX.md#structured-voting) |

**Context freshness**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `ContextFreshnessConfig` (`block_on_conflict`, `tool_effects`, `tool_effect`) | Which tool calls read or write which resources, and whether a stale write is blocked. | `arko_workflows/src/freshness/mod.rs` | [Context freshness](#context-freshness-arko_workflows) |
| `ToolResourceEffect` (`ReadFile`, `ReadDirectoryTree`, `WriteFile`, each with `parameter`) | The effect of a mapped tool, naming its path parameter. | `arko_workflows/src/freshness/mod.rs` | [Context freshness](../arko_workflows/INDEX.md#context-freshness) |
| `ResourceId` (`File`, `DirectoryTree`, `MemoryTier`, `Custom { namespace, key }`), `ActorId` (`task_id`, `agent_name`, `new`) | A tracked resource, and a reader or writer (one agent run). | `arko_workflows/src/freshness/mod.rs` | [Context freshness](../arko_workflows/INDEX.md#context-freshness) |
| `FreshnessConflict` (`resource`, `observed_version`, `current_version`, `last_writer`), `FreshnessSnapshot` | A stale read. The tracker's saved state (private fields). | `arko_workflows/src/freshness/mod.rs` | [Context freshness](../arko_workflows/INDEX.md#context-freshness) |
| `ResourceTracker` (`new`, `from_snapshot`, `snapshot`, `config`, `resource_for_tool`, `record_read`, `check_write`, `record_write`, `clear_actor`, `finish_actor`, `resume_actor`, `changed_since_read`) | The run's tracker (`SharedContext::resource_tracker`). | `arko_workflows/src/freshness/mod.rs` | [Resource tracker](../arko_workflows/INDEX.md#resource-tracker) |
| `Workflow::register_resource_read`, `Workflow::mark_resource_dirty` | Records a read, or a write that notifies stale active readers, of any resource from application code. `BuilderInvalid` without freshness. | `arko_workflows/src/workflow/mod.rs` | [Workflow-facing resource and ballot operations](../arko_workflows/INDEX.md#workflow-facing-resource-and-ballot-operations) |

**Completion verification and distillation**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `CompletionVerifierConfig::correctness`, `CompletionVerifierConfig::efficiency` | A gate on target agents' `complete_task` calls, reviewed by another registered agent. `Efficiency` is advisory. | `arko_workflows/src/verification/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `CompletionVerifierConfig::target`, `CompletionVerifierConfig::instructions`, `CompletionVerifierConfig::minor_rejection_cap`, `CompletionVerifierConfig::major_rejection_cap`, `CompletionVerifierConfig::reuse_conversation`, `CompletionVerifierConfig::include_prior_findings`, `CompletionVerifierConfig::format_retries`, `CompletionVerifierConfig::review_retries`, `CompletionVerifierConfig::repair_context` | The verifier's setters. Each sets the field of the same meaning. | `arko_workflows/src/verification/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `CompletionVerifierConfig` fields (`name`, `verifier_agent`, `target_agents`, `kind`, `instructions`, `max_minor_rejections`, `max_major_rejections`, `reuse_conversation`, `repair_context`, `include_prior_findings`, `max_format_retries`, `max_review_retries`), `VerificationKind`, `RepairContextConfig` (`max_bytes`) | The verifier configuration as stored and serialized. | `arko_workflows/src/verification/mod.rs` | [CompletionVerifierConfig, VerificationKind, and RepairContextConfig](../arko_workflows/INDEX.md#completionverifierconfig-verificationkind-and-repaircontextconfig) |
| `VERIFIER_OUTPUT_CONTRACT`, `VerificationDecision` (`Approved`, `RejectedMinor`, `RejectedMajor`), `VerificationRecord` (`timestamp`, `target_task_id`, `target_agent`, `verifier_name`, `verifier_agent`, `verifier_task_id`, `decision`, `feedback`) | The verdict format verifiers must return (first non-empty line `ACCEPTED`, `REJECTED_MINOR` or `REJECTED_MAJOR`), and the recorded verdicts. | `arko_workflows/src/verification/mod.rs` | [Verdict protocol and records](../arko_workflows/INDEX.md#verdict-protocol-and-records) |
| `DistillationConfig::new` and fields (`name`, `model`, `target_agents`, `target_tools`, `min_chars`, `max_output_tokens`, `instructions`, `reasoning`) | An LLM distiller that replaces large results of the named tools, for the named agents, with a summary. | `arko_workflows/src/distillation/mod.rs` | [Verification and distillation](#verification-and-distillation-arko_workflows) |
| `DistillationConfig::target_agent`, `DistillationConfig::target_tool`, `DistillationConfig::min_chars`, `DistillationConfig::max_output_tokens`, `DistillationConfig::instructions`, `DistillationConfig::reasoning`, `DistillationConfig::default_max_output_tokens` | The distiller's setters, and the output limit sent when none is set. | `arko_workflows/src/distillation/mod.rs` | [DistillationConfig](../arko_workflows/INDEX.md#distillationconfig) |
| `DistillationRecord` (`timestamp`, `task_id`, `agent_name`, `tool_name`, `config_name`, `input_chars`, `output_chars`, `cost`, `rejection`) | One distillation request, task, agent, tool, sizes, billed cost, and why its summary wasn't used (`rejection`). Also an output skipped for the distiller model's input limit (`rejection` starts with `skipped:`, no cost). | `arko_workflows/src/distillation/mod.rs` | [DistillationRecord](../arko_workflows/INDEX.md#distillationrecord) |

**Budget and costs**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::add_manual_cost`, `Workflow::cost_ledger` | Records an outside cost (finite, nonnegative) and fires the budget hooks. Returns `BudgetExceeded` over the limit, with the cost recorded. `cost_ledger` clones the ledger. |
| `Workflow::cost_totals_by_agent`, `BudgetManager::totals_by_agent`, `CostTotals` (`billed`, `list_price`, `input_tokens`, `output_tokens`, `calls`), `BudgetManager::ledger_len` | Each agent's running cost totals, kept as costs are added and rebuilt from the ledger on resume, for a host that reads them often, and the ledger's length, which grows by one per cost, for a host that recomputes figures from the ledger only when it changes. | `arko_workflows/src/budget/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `BudgetManager` (`new`, `from_ledger`, `with_exceeded_policy`, `exceeded_policy`, `waits_for_raise`, `add_agent_cost`, `add_manual_cost`, `set_limit`, `set_agent_limit`, `clear_agent_limit`, `agent_limits`, `agent_limit`, `agent_spent`, `is_agent_exceeded`, `hold_reason`, `total_spent`, `is_exceeded`, `check_warning`, `remaining`, `limit`, `snapshot_ledger`) | The run's spend, limits and ledger (`SharedContext::budget`). It records and reports. It neither cancels agents nor fires hooks. | `arko_workflows/src/budget/mod.rs` | [BudgetManager](../arko_workflows/INDEX.md#budgetmanager) |
| `CostLedger` (`entries`, `total`), `CostEntry` (`amount`, `source`, `timestamp`, `metadata`), `CostSource` (`Agent { agent_name, usage }`, `Manual { description }`), `AddCostResult` (`new_total`, `warning_triggered`, `exceeded`, `exceeded_crossed`) | The cost history and one recorded cost. Agent entries keep the full `UsageInfo`. | `arko_workflows/src/budget/mod.rs` | [Cost data types](../arko_workflows/INDEX.md#cost-data-types) |

**Durable steps**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::durable_step`, `Workflow::durable_step_with_resolver` | Runs a keyed side effect once and replays its recorded output. The resolver reconciles an earlier attempt whose outcome is unknown. | `arko_workflows/src/workflow/mod.rs` | [Durable steps](#durable-steps-arko_workflows) |
| `Workflow::in_doubt_steps`, `Workflow::resolve_in_doubt_completed`, `Workflow::resolve_in_doubt_failed`, `Workflow::authorize_in_doubt_retry` | Lists steps in doubt, and records an operator's resolution. | `arko_workflows/src/workflow/mod.rs` | [Workflow-facing durable operations](../arko_workflows/INDEX.md#workflow-facing-durable-operations) |
| `DurableStepSpec` (`key`, `instance`, `version`, `retry_policy`, `new`, `instance`, `version`, `retry_policy`, `display_key`) | A step's identity (key and instance), definition version and retry policy. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `DurableRetryPolicy` (`Never`, `WhenDefinitelyNotCompleted { max_attempts }`, `RetrySafe { max_attempts }`) | When an in-doubt step may run again. Default `Never`. | `arko_workflows/src/durable/mod.rs` | [Durable steps](#durable-steps-arko_workflows) |
| `DurableStepStatus` (`Started`, `Completed`, `InDoubt`, `PermanentlyFailed`) | A journal record's state. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `DurableStepContext` (`workflow_run_id`, `step_id`, `operation_id`, `attempt`, `operation_id()`), `InDoubtContext` | What the step closure receives (use `operation_id` as an idempotency key), and what a resolver or operator sees. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `ResolveInDoubt::resolve`, `InDoubtResolution` (`Completed(O)`, `DefinitelyNotCompleted`, `StillInDoubt`, `PermanentlyFailed`) | Extension trait that checks an external system for an in-doubt step's outcome. | `arko_workflows/src/durable/mod.rs` | [Step specifications, execution context, and reconciliation](../arko_workflows/INDEX.md#step-specifications-execution-context-and-reconciliation) |
| `durable::DurableStore` (`new`, `journal_path`, `workflow_run_id`, `execute`, `execute_with_resolver`, `list_in_doubt`, `operator_complete`, `operator_permanently_fail`, `operator_authorize_retry`) | The append-only journal at `<run dir>/durable/journal.jsonl`. Usable directly. | `arko_workflows/src/durable/mod.rs` | [DurableStore](../arko_workflows/INDEX.md#durablestore) |

**Persistence and checkpoints**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `PersistenceConfig` (`checkpoint_dir`, `auto_checkpoint`, `retain_checkpoints`, `event_log`, `event_flush_interval`, `task_journal`, `pretty_json`, `new`, `auto_checkpoint(b)`, `keep_last_checkpoints`, `event_log(e)`, `event_flush_interval(d)`, `task_journal(b)`, `pretty_json(b)`, `user_data`) | The run directory and what is saved there. | `arko_workflows/src/persistence/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `CheckpointRetention` (`KeepAll`, `KeepLast(NonZeroUsize)`) | Which checkpoint files are kept. Default `KeepAll`. | `arko_workflows/src/persistence/mod.rs` | [Persistence](#persistence-arko_workflows) |
| `EventLog` (`Full`, `LifecycleOnly`), `DEFAULT_EVENT_FLUSH_INTERVAL` (1 s) | What `events.jsonl` records, and how often buffered events are written at least. | `arko_workflows/src/persistence/mod.rs` | [Persistence](#persistence-arko_workflows) |
| Task journal (`agents/<task id>/live.json`, `persistence::LIVE_CHECKPOINT_FILE` `live-checkpoint.json`) | Running agents saved at their safe points, with the workflow's state. On by default with persistence (`task_journal(false)` turns it off). | `arko_workflows/src/persistence/mod.rs` | [Task journal and recovery](../arko_workflows/INDEX.md#task-journal-and-recovery) |
| `Workflow::set_budget`, `Workflow::set_agent_budget`, `Workflow::budget`, `Workflow::is_budget_blocked`, `Workflow::budget_blocked` | Change the workflow's or an agent's limit while agents run (raising releases held agents, saved with the next checkpoint). The current limit. Whether (or, awaited, once) the budget holds up the whole run, so a host that can't raise it live can checkpoint and stop. | `arko_workflows/src/workflow/mod.rs` | [Budget holds](../arko_workflows/INDEX.md#budget-holds) |
| `Workflow::set_agent_skills` | Replaces an agent's skills configuration while the workflow runs. Runs that start after it (a new conversation, later subagents) have the new skills, and a run going keeps its own. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `PlanBudget` (`five_hour`, `seven_day`, `chatgpt`, `claude_code`, `none`, `five_hour`, `seven_day`, `chatgpt`, `claude_code`, `limit`, `limit_for`, `is_set`), `PlanLimits` (`five_hour`, `seven_day`, `none`, `five_hour`, `seven_day`, `limit`, `is_empty`), `WorkflowBuilder::plan_budget`, `Workflow::set_plan_budget`, `Workflow::plan_budget`, `Workflow::plan_usage_spent`, `Workflow::plan_usage_spent_of`, `PlanUsageSpent`, `PlanWindow::{FiveHour, SevenDay}`, `PlanBudgetCrossing` (`plan`, `window`, `used`, `limit`), `PlanBudgetState` (`limits`, `usage`, `plans`), `PlanUsageTracking`, `PlanWindowUsage`, `WorkflowError::PlanBudgetExceeded` | A budget in percentage points of subscription plans' five-hour and seven-day windows, each plan (ChatGPT, Claude Code) counted separately from its windows' levels when the run started (resets folded in, other use of the plan counts too). General limits for each plan's windows and a plan's own limits in their place. Held or cancelled as the dollar budget is, raised live, counting kept across resumes (`WorkflowState::plan_budget`). | `arko_workflows/src/budget/plan.rs` | [Plan budgets](../arko_workflows/INDEX.md#plan-budgets) |
| `BudgetExceededPolicy::{Pause, Cancel}`, `WorkflowBuilder::budget_exceeded_policy`, `WorkflowBuilder::wait_for_budget_raise` | What happens over a budget (default `Pause`, held, not cancelled), and whether runs the host started wait for a raise. | `arko_workflows/src/budget/mod.rs` | [Budget holds](../arko_workflows/INDEX.md#budget-holds) |
| `Workflow::reopen_agent`, `Workflow::reopen_agent_with` | Continue a `Cancelled` run (cancelled by the budget under `Cancel`, stopped with the workflow, aborted after the grace period, also one recorded by an earlier version) from where its task stopped, told it was reopened, with a prompt (and with `reopen_agent_with`, its images). A nested run must be orphaned (as for `recover_agent`). A superseded run is refused. | `arko_workflows/src/workflow/mod.rs` | [Run, continue, and spawn](../arko_workflows/INDEX.md#run-continue-and-spawn) |
| `Workflow::interrupted_runs`, `Workflow::recover_agent` | After `resume_from`, the runs the host must recover. Those that were running when the workflow stopped (not those started inside a tool call their parent still holds open), and orphaned nested runs (`task.parent` set, their parent ended or answered the call). And continuing one where it was saved (an orphan's parent link is released). A run is never recovered twice at once. | `arko_workflows/src/workflow/mod.rs` | [Task journal and recovery](../arko_workflows/INDEX.md#task-journal-and-recovery) |
| `Workflow::checkpoint` | Saves a checkpoint now (nothing without persistence). A failed checkpoint write is returned as an error. | `arko_workflows/src/workflow/mod.rs` | [Results, state, costs, events, and pause](../arko_workflows/INDEX.md#results-state-costs-events-and-pause) |
| `PersistenceManager` (`resolve_run_id`, `new`, `config`, `durable_store`, `save_checkpoint`, `auto_checkpoint`, `append_event`, `flush_events`, `save_agent_task`, `load_checkpoint`) | Writes a run directory. Usable directly, for example `load_checkpoint` to inspect a checkpoint before resuming. | `arko_workflows/src/persistence/mod.rs` | [PersistenceConfig and PersistenceManager](../arko_workflows/INDEX.md#persistenceconfig-and-persistencemanager) |
| `persistence::state::WorkflowState` (`workflow_run_id`, `workflow_name`, `workflow_metadata`, `checkpoint_id`, `timestamp`, `memory_snapshot`, `message_queues`, `budget_spent`, `cost_ledger`, `freshness_snapshot`, `agent_results`, `agent_runs`, `missing_task_states`, `verification_records`, `distillation_records`, `completed_agents`, `cancellation`, `agent_registry`, `notes`, `budget_limits`, `plan_budget`), `state::BudgetLimits` (`limit`, `agent_limits`), `WorkflowState::agent_results_in` | A checkpoint's content. `agent_results_in` returns its results with their tasks in the run directory, read when `task()` is first called for a run that completed, and at once for the others. | `arko_workflows/src/persistence/state.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| `persistence::state::AgentRunRecord` (`agent_run_id`, `agent_name`, `status`, `summary`, `error`, `task_file`, `usage_charged`, `stopped_by`, `info`), `MissingTaskState` (`agent_run_id`, `agent_name`, `status`, `last_saved`, `reason`), `SerializableAgentResult` (`agent_name`, `summary`) | A checkpoint's reference to a result's task version, with what the task holds besides its conversation (`info`), a run whose latest task state couldn't be saved, and a name-and-summary record. | `arko_workflows/src/persistence/state.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| `messaging::registry::RegistrySnapshot` (`version`, `agents`, `seen`), `AgentRecord` (`last_start`, `finished`, `last_finish`) | The agent status registry as a checkpoint saves it. | `arko_workflows/src/messaging/registry.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| `notes::NotesSnapshot` (`notes`, `seen`, `started`, `control_offset`) | The operator notes as a checkpoint saves them (`WorkflowState::notes`, absent in checkpoints without notes). | `arko_workflows/src/notes/mod.rs` | [Checkpoint payloads](../arko_workflows/INDEX.md#checkpoint-payloads) |
| Event log (`events.jsonl`) | One JSON line per recorded agent event, with time, run ID, agent name and agent run ID. And one per posted operator note, with time, run ID and the `note`. | `arko_workflows/src/persistence/mod.rs` | [PersistenceConfig and PersistenceManager](../arko_workflows/INDEX.md#persistenceconfig-and-persistencemanager) |

**Shared context**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `Workflow::context`, `context::SharedContext` | The run's shared state. Public fields, with `workflow_run_id`, `workflow_name`, `workflow_metadata`, `workspace`, `budget`, `durable_store`, `memory`, `messages`, `messaging`, `resource_tracker`, `hooks`, `agent_configs`, `global_tools`, `agent_tools`, `agent_results`, `global_interceptors`, `agent_interceptors`, `interception_failure_policy`, `completion_verifiers`, `verification_attempts`, `verification_records`, `distillation_configs`, `distillation_records`, `workflow_services`, `paused`, `resume_notify`, `user_input_handler`, `approval_policy`, `error_recovery_policy`, `ballots`, `voters`, `event_tx`. No public constructor. | `arko_workflows/src/context/mod.rs` | [Shared context](../arko_workflows/INDEX.md#shared-context) |
| `SharedContext::message_recipients`, `SharedContext::message_senders`, `SharedContext::recipient_statuses`, `SharedContext::recipient_status` | The agents one agent may message, the agents that may message it, and their statuses. | `arko_workflows/src/context/mod.rs` | [Shared context](../arko_workflows/INDEX.md#shared-context) |

**Errors**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `WorkflowError::BuilderMissing { field }`, `WorkflowError::BuilderInvalid(String)` | A required builder setting is missing. An invalid configuration or checkpoint, an invalid durable step spec, a feature used while off (freshness, voting), a result that can't be continued or resumed, or a disallowed services caller or target. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::AgentNotFound` | No agent is registered under the name. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::UserInputRequired { agent, task_id, questions }`, `WorkflowError::AgentPaused { agent, task_id, pause }`, `WorkflowError::VerifierFailed { verifier, agent, task_id, reason }` | Pauses, not failures. The agent waits for an answer (`continue_agent`), a decision (`resume_agent`), or a continuation after its verifier couldn't review its completion. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::AgentInterrupted { agent, reason }`, `WorkflowError::AgentCancelled { agent, reason }`, `WorkflowError::AgentStopped { agent, task_id, reason }` | The driven task ended interrupted, or cancelled while the run was not stopped. `AgentStopped`, cancelled by its own stop (`stop_run`), not a failure. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::Agent`, `WorkflowError::Llm`, `WorkflowError::Tool` | Errors from arko_agents, arko_ai and tools. A task that ended `Failed` returns `Agent(ArkoAgentsError::TaskExecution(reason))`. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::BudgetExceeded { spent, limit }`, `WorkflowError::AgentBudgetExceeded { agent, spent, limit }`, `WorkflowError::InvalidCost` | Spending strictly above the workflow or agent limit. A negative or non-finite manual cost. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::MemoryTierNotFound`, `WorkflowError::MemoryAccessDenied { agent, tier, action }`, `WorkflowError::MemoryLimitExceeded { tier, attempted, limit }` | Memory store errors. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::Cancelled`, `WorkflowError::AbortedByHook { hook_point, reason }` | The run was stopped by a cancel handle, or by a hook's `Abort`. Every run call in progress or made later returns the first cause's error. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::DurableUnavailable`, `WorkflowError::DurableDefinitionMismatch { step, details }`, `WorkflowError::DurableInDoubt { step, operation_id }`, `WorkflowError::DurablePermanentlyFailed { step, reason }`, `WorkflowError::DurableAttemptsExhausted { step }` | Durable steps without persistence. A changed definition or input for a recorded step. An unresolved earlier attempt (or one running now). A recorded permanent failure. The retry limit reached. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |
| `WorkflowError::Persistence`, `WorkflowError::Io`, `WorkflowError::Json`, `WorkflowError::Channel`, `WorkflowError::Deliberation` | Persisted-state, file and JSON errors. `Channel` for a join or channel failure, a `UserInputHandler` error or the services call limit. `Deliberation` for ballot errors. | `arko_workflows/src/error.rs` | [Workflow errors](../arko_workflows/INDEX.md#workflow-errors) |

#### Python bindings

Python bindings (`python` feature)

| Item | What it is | Code | Docs |
|---|---|---|---|
| `python::PyWorkflowBuilder`, `name`, `workspace`, `agent(name, config_path)`, `budget`, `budget_exceeded_policy(policy, wait_for_raise)`, `persistence(checkpoint_dir, auto_checkpoint, retain_checkpoints, event_log)`, `rename_memory_tier_on_resume`, `drop_memory_tier_on_resume`, `approval_policy`, `error_recovery_policy`, `max_parallel_agents`, `fail_fast`, `resume_from`, `global_memory_tier`, `cancel_grace_period`, `wait_for_message_timeout`, `agent_wait_for_message_timeout`, `build` | The Python builder. Setters return `None`, and `build` creates the client from the environment and loads the agent files. | `arko_workflows/src/python/mod.rs` | [Python bindings](#python-bindings-arko_workflows) |
| `python::PyWorkflow`, `run_agent`, `run_agents_parallel`, `continue_agent`, `resume_agent(previous, decision, note)`, `interrupted_runs`, `recover_agent`, `reopen_agent`, `set_budget`, `set_agent_budget`, `is_budget_blocked`, `agent_result`, `finish`, `cancel`, `is_cancelled`, `add_manual_cost`, `cost_ledger_json`, `checkpoint`, `read_memory`, `write_memory` | The Python workflow. The run methods are awaitable. | `arko_workflows/src/python/mod.rs` | [PyWorkflow](../arko_workflows/INDEX.md#pyworkflow) |
| `python::PyAgentResult` (`agent_name`, `summary`, `total_cost`, `status`, `agent_run_id`, `error`, `failure_cause`, `failure_kind`) | A result's read-only properties. The full result is kept for continuing it. | `arko_workflows/src/python/mod.rs` | [PyAgentResult](../arko_workflows/INDEX.md#pyagentresult) |
| Exceptions `WorkflowCancelled`, `UserInputRequired`, `AgentPaused`, `VerifierFailed` | Raised by the run methods. The last three carry `(message, agent_run_id)`. Other run errors raise `RuntimeError`. | `arko_workflows/src/python/mod.rs` | [Python bindings](../arko_workflows/INDEX.md#python-bindings) |

**Example binaries**

| Item | What it is | Code | Docs |
|---|---|---|---|
| `basic_workflow` | A two-agent sequential example. Takes no flags. | `arko_workflows/examples/basic_workflow.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `deepcoder --workspace`, `--spec` | Required, the project to modify and the specification file. | `arko_workflows/examples/deepcoder.rs` | [Run](../arko_workflows/examples/deepcoder/README.md#run) |
| `deepcoder --state-dir`, `--resume`, `--ephemeral` | Run directory, resuming a run, and running without persistence. | `arko_workflows/examples/deepcoder.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `deepcoder --budget`, `--max-minor-corrections`, `--max-efficiency-corrections`, `--max-final-corrections` | Budget and the verifiers' minor-correction caps. | `arko_workflows/examples/deepcoder.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `deepcoder --shell-timeout-seconds`, `--shell-output-bytes` | Default and maximum `run_shell` timeout, and the output returned to an agent. | `arko_workflows/examples/deepcoder.rs` | [Shell tool and isolation](../arko_workflows/examples/deepcoder/README.md#shell-tool-and-isolation) |
| `forge --use-case`, `--output`, `--involvement`, `--backend` | A workflow that builds workflows, describe a use case, get a checked, documented crate. | `arko_workflows/examples/forge/main.rs` | [Forge](../arko_workflows/examples/forge/README.md) |
| `forge --trial-runs`, `--no-explorers`, `--max-active-calls`, `--resume` | Live trial runs, the explorer agents, and resuming a run. | `arko_workflows/examples/forge/main.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `complete_workflow --data-dir DIR`, `--dry-run`, `--resume PATH`, `--owner TEXT` | Complete three-agent incident-report workflow. Validates without credentials in dry-run mode and can resume a checkpoint. | `arko_workflows/examples/complete_workflow.rs` | [Complete workflow example](#complete-workflow-example) |
| `index_benchmark prepare`, `prepare-variants`, `generate`, `run`, `report` (`--out`, `--arms`, `--variants`, `--concurrency`, `--codex-home`, `--mock`) | The arko_index agent benchmark, code-understanding questions on pinned repositories, answered with a terminal alone or with arko_index (and variant arms with other arko_index options), graded blind against verified keys. | `arko_workflows/examples/index_benchmark/main.rs` | [Example binaries](#example-binaries-arko_workflows) |
| `runctl note RUN_DIR TEXT` (`--file`, `--to`, `--except`, `--once`, `--from-agent`, `--from-host`), `runctl stop RUN_DIR` (`--agent`, `--run`, `--wrap-up`, `--now`, `--reason`, `--relaunch`), `runctl status RUN_DIR` | Post a note to any workflow run built with `control_folder()`, live or stopped. Stop runs of a live one alone. Print a run's status from its latest checkpoint. No model requests. | `arko_workflows/examples/runctl.rs` | [Example binaries](#example-binaries-arko_workflows) |

### Builder defaults (arko_workflows)

`build()` requires a name, an existing workspace directory (stored absolute, with `.` and `..` resolved and symbolic links kept), an `Arc<LLMClient>`, and at least one uniquely named agent. The builder reads no credentials. With persistence set, `build()` creates the run directory and its `workflow-run-id` file.

| Setting | Default | Notes |
|---|---|---|
| `approval_policy` | **`ApproveAll`. Every call an agent asks approval for is approved** | Includes paths outside the workspace under `RequireApproval`. Logs one warning per agent. See [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows). |
| `error_recovery_policy` | **`Automatic`, recovery without an attempt limit** | See [Approvals and error recovery](#approvals-and-error-recovery-arko_workflows). |
| `metadata` | Empty | Only `supported_platforms` is enforced. A nonempty list the host matches none of fails `build`. Dependencies are checked only when the workflow calls `WorkflowMetadata::missing_dependencies`. A dependency is required unless `optional`. |
| `memory` | No tiers, no memory tools | See [Memory](#memory-arko_workflows). |
| `budget`, `budget_warning` | None, unlimited, no warning | Dollars, finite and nonnegative. Spending counts `UsageInfo::billed()`, so ChatGPT OAuth calls add zero. What happens over the limit is `budget_exceeded_policy`. On resume, a higher limit the checkpoint recorded (raised live with `Workflow::set_budget`) is kept. A limit not set on the builder is none. |
| `agent_budget(agent, limit)` | None | The agent's own model spend across its runs, checked before each of its model requests (under `Pause`) or at each usage report (under `Cancel`, where the agent is cancelled and its call returns `AgentBudgetExceeded`). Distillation charges don't count toward it. |
| `budget_exceeded_policy` | **`BudgetExceededPolicy::Pause`** | Over the workflow's or an agent's budget, nothing is cancelled. The agent is held before its next model request (an error-recovery pause with cause `BudgetExceeded`, whatever its error policy, running tool calls finish first). A run inside another agent's tool call waits for a raise (`Workflow::set_budget`, `set_agent_budget`). A run the host started returns `AgentPaused`, continued with `resume_agent` after a raise. New runs start and are held. `OnBudgetExceeded` fires once per crossing. Spend can overshoot by the requests in flight. `Cancel`, today's earlier behaviour. Each driven agent is cancelled (its call returns `BudgetExceeded` or `AgentBudgetExceeded`), later run calls are refused, and `OnBudgetExceeded` fires per charge over the limit. `reopen_agent` continues a cancelled run. |
| `wait_for_budget_raise` | Off | Under `Pause`, a held run the host started waits for a live raise instead of returning `AgentPaused`. Turn on only when something can call `set_budget` while the run is live. |
| `hook(point, callback)` | None | Synchronous. Run in registration order, and the first result other than `Continue` ends the point. `Abort` stops the run. `Skip` is honoured only at `PreToolExecution`. |
| `persistence` | None. Nothing is saved, durable steps unavailable | See [Persistence](#persistence-arko_workflows). |
| `resume_from(checkpoint_file)` | None | Needs `persistence` whose run directory holds the checkpoint's run ID. See [Resume](#resume-arko_workflows). |
| `rename_memory_tier_on_resume`, `drop_memory_tier_on_resume` | None | See [Resume](#resume-arko_workflows). |
| `ephemeral()` | — | Clears the persistence and resume settings made so far. |
| `context_freshness` | Off | See [Context freshness](#context-freshness-arko_workflows). The default mapping covers arko's file tools and Claude Code's `Read`, `Write`, `Edit`, `MultiEdit` and `NotebookEdit`. |
| `global_tool`, `agent_tool` | None | Added to every agent's (or one agent's) available tools. A custom tool named like an arko_agents built-in replaces it, except `complete_task`. A global tool can't take the name `complete_task`, `ask_user`, a memory tool's, `run_workflow_agent` or `cast_vote`, nor a subagent tool's while `subagents` is set, nor a messaging tool's while any agent is offered it. An agent tool can't take a messaging tool's name that agent is offered (`build` fails). |
| `tool_result_interceptor`, `agent_tool_result_interceptor` | None | Global interceptors, then the agent's, then distillers. |
| `agent_handover_threshold(agent, threshold)` | None, the configured threshold | The agent's runs hand over at the `HandoverThreshold` the host keeps a clone of, in place of the configured threshold. Not saved with checkpoints. An unknown agent fails `build`. |
| `tool_image_sink(sink)` | None, tool images reach only the model | Gives the sink every tool call's images, keyed by run and call ID. Events carry no image bytes. |
| `background_jobs(manager)` | None, no call moves to the background | Every agent's runtime gets `AgentRuntime::with_background_jobs(manager)`. A subagent's moved calls and monitors stop when its run ends (`TerminalManager::stop_owner_watchers`). |
| `interception_failure_policy` | `UseRawResult` | Applied to each stage of the chain. A failing stage is retried as the policy allows and then skipped, keeping the earlier stages' output. |
| `completion_verifier`, `distiller` | None | See [Verification and distillation](#verification-and-distillation-arko_workflows). |
| `workflow_services` | None, no `run_workflow_agent` | See [Messaging, services and voting](#messaging-services-and-voting-arko_workflows). |
| `subagents` | None, no background subagents | Requires `workflow_services`. See [Background subagents (arko_workflows)](#background-subagents-arko_workflows). |
| `messaging` | `MessagingPolicy::all()`. Every agent may message every other | `disable_messaging()` offers no messaging tools. |
| `notes_by_name_only(agents)` | None besides the completion verifiers' reviews, which get a note only when it names them | Other agents get the notes addressed to all agents (`NoteAudience::All`). See [Operator notes](../arko_workflows/INDEX.md#operator-notes). |
| `notes_reply_hint(agent, hint)` | No hint | An update that gives an agent a note ends with its hint only when you set one and the update holds a note from the operator. See [Operator notes](../arko_workflows/INDEX.md#operator-notes). |
| `control_folder()` | **Off**. No other process can post notes or stop runs | Opt-in. Requires `persistence` (`build` fails without it, `ephemeral()` turns it off). `build` creates `<run dir>/control/`. Whoever can write that folder can post notes and stop runs, so keep the run directory out of the agents' reach. A stop request without a `wrap_up_secs` cancels at once, and without an `intent` retires the runs. |
| `wait_for_message_timeout`, `agent_wait_for_message_timeout` | `WaitForMessageTimeout::Auto` | Just inside the agent's prompt-cache lifetime, counted from its last model request, 4 min 30 s, or 54 min with Anthropic's 1-hour TTL. A fixed `Duration` must be above zero and at most 24 hours. |
| `user_input_handler` | None. An `ask_user` question suspends the agent and the call returns `UserInputRequired` | A handler's `Err` ends the call with `WorkflowError::Channel`. |
| `enable_voting(voters)` | Off | Offers `cast_vote` to the named agents only. |
| `max_parallel_agents(n)` | No cap | For `run_agents_parallel`. `0` fails `build`. |
| `fail_fast(bool)` | `false` | `true`, in `run_agents_parallel` the first failure stops the others. A pause is not a failure. |
| `cancel_grace_period` | 30 s (`DEFAULT_CANCEL_GRACE_PERIOD`) | How long a stopped run waits for each running agent before aborting it. Nested runs get half per level. A graceful stop's wrap-up comes before it, and only when a `StopRequest` asks for one (no wrap-up by default). |

Every agent-name list is validated at build. Tool names that the freshness mapping or a distiller names are only warned about. See [Builder signatures and defaults](../arko_workflows/INDEX.md#builder-signatures-and-defaults).

### Approvals and error recovery (arko_workflows)

Workflows run without human input by default. The managed paths (`run_agent`, `continue_agent`, `resume_agent`, `run_agents_parallel`, `AgentHandle::wait`) answer approvals and error-recovery pauses under these policies. `AgentHandle::into_parts` hands both to the caller.

| `ApprovalPolicy` | Behaviour |
|---|---|
| `ApproveAll` (default) | Approves every call the agent asks approval for. |
| `RespectAgentConfig` | Pauses the agent. The call returns `WorkflowError::AgentPaused` with `AgentPause::ToolApproval`. Continue with `resume_agent`. A call without a decision is rejected. |
| `Handler(Arc<dyn ToolApprovalHandler>)` | Decides each call, approve, reject with a reason, or pause. |

| `ErrorRecoveryPolicy` | Behaviour |
|---|---|
| `Automatic` (default) | After the agent's own retries run out, outages wait 30 s, doubling to 15 min while pauses repeat (and at least the provider's requested delay), then resend unchanged. Plan usage limits wait until the reset plus 5 s. Tool errors continue at once. Each wait gets up to 30 s of random extra time. Non-retryable errors, refusals and unusable responses fail the task with a reason. The failure keeps the error's kind (`AgentResult::failure_kind`), whatever it is. Monitor pauses and failed verifier reviews return to the caller. Nothing is added to the conversation. |
| `Limited { max }` | As `Automatic`, for at most `max` recoveries per call that drives the agent. After that, a pause it would recover from fails the task. `0` recovers nothing. |
| `ReturnToCaller` | Every error-recovery pause returns `AgentPaused` with `AgentPause::ErrorRecovery` (`VerifierFailed` for a completion its verifier couldn't review). |

An agent that is interrupted instead of pausing (as the `RetryThenInformAgent` and `InformAgentAndContinue` error policies do when recovery ends) is cancelled, and the call returns `AgentInterrupted`. An agent running inside another agent's tool call (a verifier or a delegated run) can't pause, under `RespectAgentConfig` its `complete_task` is approved, and anything else that would pause it stops it (a completion its verifier couldn't review is left paused). `UserInputRequired`, `AgentPaused` and `VerifierFailed` are pauses, not failures, `OnUserInputRequired` fires for the first and `OnAgentPaused` for the others. `OnWorkflowFailed` fires for none. See [Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery).

### Memory (arko_workflows)

| Setting | Default | Notes |
|---|---|---|
| `MemoryConfig::global_tier(name)` | Readers and writers `All`, `AppendOnly`, no size limit | `global_tier_with_limit`, `broadcast_tier(name, writers)` (writers `Only`), `broadcast_tier_with_limit`. Other readers, writers or `FullEdit` need `tier(MemoryTier { .. })`. |
| `MemoryTier::max_bytes` | `None`, unlimited | Limits the content (not the history). A write over it fails and changes nothing. |
| `MemoryTier::write_mode` | No default on the tier (required in JSON). `WriteMode::default()` and the `MemoryConfig` helpers use `AppendOnly` | `FullEdit` replaces the content. The history keeps each version. |
| `AccessScope` | — | `All`, `Only(names)`, `None`. |

An agent gets `read_memory` if it can read a tier, `write_memory` if it can write one, and `list_memories` if either. `list_memories` hides tiers it can't access. `read_memory` results are never intercepted or distilled. See [Memory](../arko_workflows/INDEX.md#memory).

### Messaging, services and voting (arko_workflows)

`WorkflowServicesConfig` is built with `Default` and its setters. A struct literal (`..Default::default()` included) doesn't compile outside the crate.

| Setting | Default | Notes |
|---|---|---|
| `MessagingPolicy` | Every agent may message every other | `can_message(agent, recipients)` limits one agent. `cannot_message(agent)`. `send_message` and `list_recipients` go to agents with a permitted recipient, `wait_for_message` to agents another may message. `send_message` reaches only running or waiting agents. Running agents are told when a recipient starts or finishes. |
| `MessagingPolicy::keep_messages_for_finished()` | Off. A message to a finished agent is refused | On. It is queued for the agent's next run (a continuation or a new run), and the finish notice says so. |
| `WorkflowServicesConfig::allowed_callers`, `allowed_targets` | Empty, **nobody** | Name them with `allow_caller` and `allow_target`. An empty caller list is only warned about. |
| `allow_all_callers()`, `allow_all_targets()` | Off | Every agent except verifier agents (and, as a target, the caller itself). |
| `max_active_calls` | 4 | Delegated calls running at once, workflow-wide (`resume_run` calls included). A call over the limit fails at once. Calls recovered after a crash count but are never refused. A `NonZeroU32`, `max_active_calls(0)` fails the build. |
| `queue_active_calls()` | Off. A call over `max_active_calls` fails | On. It waits until a call ends. |
| `unlimited_active_calls()` | Off | On, no limit on calls at once. |
| `BallotConfig` | No default | `begin_ballot` requires at least 2 options with nonblank, unique IDs, `quorum` at least 1 and `consensus_threshold` from 0 to 1. `eligible_voters` empty = every enabled voter. A voter without a weight weighs 1.0. `finish_ballot` rejects a cast vote whose weight isn't finite and positive. `tie_break` is `RejectTie` or `Lexicographic` (smallest option ID). |

### Background subagents (arko_workflows)

`WorkflowBuilder::subagents(SubagentsConfig::default())` offers the workflow services' callers `start_agent`, `message_agent`, `wait_for_agents`, `list_agents` and `stop_agent`, and their subagents `message_agent` and `wait_for_agents`.

| `SubagentsConfig` | Default | Notes |
|---|---|---|
| `max_running` | 4 (`DEFAULT_MAX_RUNNING`) | Subagents running at once, workflow-wide. Starts and restarts over it wait in a queue, oldest first. `unlimited()` lifts it. |
| `max_depth` | 1 | Subagents start no subagents and aren't offered `run_workflow_agent`. Higher, every subagent of a caller agent under that depth gets the start tools (`SubagentHook::offers_start_tools` narrows that per run). A stop of a subagent, or its end, stops the subagents it started. It can't report while they work (`complete_task` is refused five times in a row, then let through and they are stopped), and while it waits for them in `wait_for_agents` it holds no `max_running` place. |
| `peer_messages` | On | The lead may let its subagents message each other (`peers`, `group`, `link_agents`). Off, subagents message only their lead. |
| `max_peer_messages` | 20 (`DEFAULT_MAX_PEER_MESSAGES`) | Messages between two subagents until their lead writes to either. |
| `copy_peer_messages_to_lead` | Off | On. The lead gets a copy of each, which wakes no one. |
| `workspaces_anywhere` | Off. A `workspace` must be inside the caller's directory | |
| `offer_blocking_tool` | On, callers keep `run_workflow_agent` | |
| `questions_to_lead` | On. A subagent's `ask_user` goes to its lead | Off, to the workflow's `UserInputHandler`. |
| `workspace_provider`, `hook`, `start_parameters` | None | No `isolated` parameter without a provider. |
| `max_failed_requests` | None, no cap | A subagent fails as soon as this many of its model requests have failed in a row, counting the agent's own retries and its waits for rate limits and overloaded errors. A model response resets the count, and it starts again when the run is resumed or continued. The run the host drives is never capped. |

Mail over `MAX_MAIL_BYTES` (16 KiB) reaches a running agent as an excerpt with a file holding all of it, and one context update holds about `UPDATE_MAX_MAIL_BYTES` (48 KiB) of mail, or about 9,500 characters for a Claude Code agent, whose hook Claude Code cuts at 10,000. `stop_agent` gives a subagent a two-minute wrap-up. See [Background subagents](../arko_workflows/INDEX.md#background-subagents).

### Verification and distillation (arko_workflows)

`CompletionVerifierConfig::correctness(name, verifier_agent)` and `efficiency(..)` set a default `instructions` text. In JSON, `name`, `verifier_agent`, `kind` and `instructions` are required.

| `CompletionVerifierConfig` | `correctness()` | `efficiency()` | Omitted in JSON |
|---|---|---|---|
| `target_agents` | Empty, with `build` fails | Same | Required |
| `max_minor_rejections` | 3 | 2 | Required |
| `max_major_rejections` | 0: the first major rejection fails the task | 0 (a major rejection counts as minor) | Required |
| `reuse_conversation` | `false` | `false` | `false` |
| `repair_context` | `None` (`repair_context(bytes)`, at least 1,024) | `None` | `None` |
| `include_prior_findings` | `true` | `true` | `true` |
| `max_format_retries` | 1 | 1 | 1 |
| `max_review_retries` | 1 | 1 | 1 |

A review that fails doesn't fail the target. The agent is asked to call `complete_task` again, up to `max_review_retries` times, then the call returns `WorkflowError::VerifierFailed` with the task paused. A correctness verifier fails the task at the rejection after its cap. An efficiency verifier never fails a task by rejecting it and stops reviewing once its minor cap is reached. `repair_context` needs `reuse_conversation` and context freshness.

In JSON, every `DistillationConfig` field except `max_output_tokens` and `reasoning` is required.

| `DistillationConfig::new(name, model)` | Default | Notes |
|---|---|---|
| `target_agents`, `target_tools` | Empty, with `build` fails | Both required. `read_file` is not exempt. |
| `min_chars` | 8,000 | Results shorter than this are left alone. |
| `max_output_tokens` | `None`, 16,000, or the model's output limit when smaller (nothing over ChatGPT OAuth) | A set value is always sent (arko_ai drops it over ChatGPT OAuth, with a warning). `Some(0)` is rejected. |
| `reasoning` | `None`, the model's default | Must be for the model's provider. |

Distillation requests turn prompt caching off (except over ChatGPT OAuth), count toward the workflow budget but not the agent budget, and keep the raw output when the summary is refused or cut off. An output whose request (estimated at 3 bytes per token) plus the output limit is over the distiller model's input limit is not sent. The raw output is kept, a record notes the skip, and the agent's overflow guard shortens the output if the agent's own model can't take it either. See [Tool-result distillation](../arko_workflows/INDEX.md#tool-result-distillation).

### Context freshness (arko_workflows)

`ContextFreshnessConfig::default()`, `block_on_conflict: true`, and `read_file`, `list_files` (a directory-tree read), `write_file`, `edit_file` and `apply_diff` mapped to their path parameters. `tool_effect(name, effect)` maps other tools. When loaded from JSON, an omitted field takes its default. A given `tool_effects` replaces the default map whole. A write is blocked when the writer read that file, or listed a directory containing it, and another agent run (another run of the same agent included) has since written it (for a listing, any file under the directory). The call is rejected with a message to re-read. With `block_on_conflict: false` the write is allowed. In both modes, active readers whose view a write made stale get a notice, and `OnFreshnessConflict` fires for each stale reader. Memory tiers are tracked the same way through `read_memory` (complete reads only) and `write_memory`, and custom resources through `Workflow::register_resource_read` and `mark_resource_dirty`. See [Context freshness](../arko_workflows/INDEX.md#context-freshness).

### Durable steps (arko_workflows)

`DurableStepSpec::new(key)`, version `"v1"`, no instance, retry policy `Never` (in JSON `version` and `retry_policy` are required). A step whose closure returns an error is recorded in doubt, and the call returns `DurableInDoubt` (the closure's error becomes the recorded reason). Under `Never` it stays in doubt until resolved. `WhenDefinitelyNotCompleted { max_attempts }` runs it again only after a resolver (`durable_step_with_resolver`) reports it definitely not completed. `RetrySafe { max_attempts }` runs it again at the next call for the step, without a resolver. `authorize_in_doubt_retry` allows one more run (under `Never` always, otherwise within `max_attempts`). `max_attempts` 0 counts as 1. Durable steps need persistence and replay from the run directory's `durable/journal.jsonl`. See [Durable steps](../arko_workflows/INDEX.md#durable-steps).

### Persistence (arko_workflows)

`PersistenceConfig` is built with `new(dir)` and its setters. A struct literal doesn't compile outside the crate.

| `PersistenceConfig::new(dir)` | Default | Notes |
|---|---|---|
| `auto_checkpoint` | `true` | A checkpoint after every finalized agent run and when a run is stopped. Task versions are saved either way. |
| `retain_checkpoints` | `CheckpointRetention::KeepAll`. Every checkpoint and task version is kept | `keep_last_checkpoints(n)` (`KeepLast`) keeps the newest `n` and the task versions they refer to. `n` of 0 fails the build. In Python, `retain_checkpoints=n` (0 raises `ValueError`). |
| `event_log` | `EventLog::Full`, every event, including each streamed fragment | `LifecycleOnly` leaves the fragments out. The event log is never truncated or rotated. |
| `event_flush_interval` | 1 s | State changes, run ends and errors are written at once. |
| `task_journal` | `true`, running agents are saved as they run | Each running task in `agents/<task id>/live.json` and the state in `live-checkpoint.json`, rewritten at its safe points (both on disk before tool calls with effects run, groups of `ToolRecovery::Rerun` calls and Claude Code's read-only built-ins don't wait). The live checkpoint only when what a resume reads from it changes. `latest-checkpoint` names the newest checkpoint, live or versioned. `false` saves a run only when it ends. In Python, `task_journal=`. |
| `pretty_json` | `false`. The task journal's files are compact JSON | Checkpoints and task versions are pretty JSON either way. In Python, `pretty_json=`. |

`PersistenceConfig::user_data(name)` picks a new run directory, `<user data>/arko/workflows/<name>/<UUID>`, on each call. A configuration pointing at an existing run directory reuses its `workflow-run-id` and durable journal, also without `resume_from`. Checkpoints and task versions are pretty JSON, the task journal's files compact JSON unless `pretty_json`. Each is written through a uniquely named temporary file, synced and renamed (the directory synced on Unix, except for the task journal's background saves, creating the temporary file and the rename retried on Windows while another program holds the file). Leftover temporary files and unreferenced live files are removed when a run directory is opened. `latest-checkpoint` names the newest checkpoint, live or versioned. The event log and the durable journal are appended JSON lines. Images are saved once each in `images/<SHA-256>.<type>`, and the JSON files name them (`persistence::images::ImageFiles`). All are written with default permissions. Task versions hold full conversations, and the event log the agents' output and tool results. A save that fails never fails an agent's run. It is logged and retried by the next checkpoint. `checkpoint()` and `finish()` save regardless of `auto_checkpoint`, and return a failed checkpoint write as an error. See [Persistence](../arko_workflows/INDEX.md#persistence-and-checkpoint-state).

### Resume (arko_workflows)

`WorkflowBuilder::resume_from(path)` takes a checkpoint file, not a run directory.

- The new builder supplies agents, client, tools, hooks, policies, budget limits and metadata. A higher limit the checkpoint recorded (`budget_limits`, raised live, each `plan_budget` limit likewise, per window and per plan) is kept for a limit the builder sets, and plan budget counting continues from the run's original baseline, per plan. Counting from a checkpoint written before plans were counted apart is taken over by the first plan to report. Runs paused by the budget continue with `resume_agent`, and cancelled ones with `reopen_agent`. Memory access, limits and write mode come from the builder. The checkpoint supplies each tier's content and history by name. For every restored task that can still be continued or resumed, or that a stop or `cancel` cancelled (marked `stopped_by`, so `stopped_runs` reopens it), a changed user-configured agent configuration is rejected, and the error names the changed fields by their paths (`agent 'coder' configuration changed since its checkpoint (instructions, tool_config.available_tool_names changed); ...`, the first ten). A task cancelled otherwise is not checked, and reopened it keeps the configuration saved in it. Use the original configuration, register the changed one under another name, or name the agent in `keep_saved_configs`, which keeps each saved run's own configuration and checks only that its tools are still there. Workflow-injected tools and refreshed model registry snapshots are ignored for this comparison.
- A restored result that `continue_agent` or `resume_agent` runs again keeps the `AgentConfig` saved in its task (model, instructions and offered tools). New runs use the builder's.
- A tier configured but missing from the checkpoint starts as configured. A checkpoint tier with no configured tier fails `build`, unless renamed with `rename_memory_tier_on_resume` or dropped with `drop_memory_tier_on_resume`.
- Restored, memory, pending messages, the cost ledger, finalized agent results (with their tasks), the runs that were running (as `Interrupted` results with their saved tasks, when the task journal is on, their uncharged usage is charged), verification and distillation records, agent statuses, operator notes (with what each run was given), background subagents with their mailboxes (relaunched by `resume_subagents`), and freshness state when context freshness is configured again. Operator notes never touch `AgentConfig`, so they change the agents' instructions without making the configuration check refuse a resume. Not restored. A recorded stop (the run starts again, `stopped_runs` lists the runs it ended, marked `stopped_by`, for `restart`), ballots, verifier repair counts, and the pause flag. No agent runs after a resume. Agents that were running count as finished (interrupted).
- The checkpoint's workflow name and run ID must match. `continue_agent` continues a `Completed` or `WaitingForInput` result. `resume_agent` answers a `Paused` one. `recover_agent` continues an `Interrupted` one from `interrupted_runs()` where it was saved (its open calls settled by their tools' `recovery_mode`, delegated runs recovered with it, a completion reviewed again). A delegated run its caller can't recover (or that a stop cancelled) is named in the call's result, and the caller can resume it with `run_workflow_agent`'s `resume_run`. Once the caller can't (it ended), `interrupted_runs()` lists it as orphaned (and `stopped_runs()` too, unless the caller is listed there). `build` skips the configuration check for interrupted runs that recovery supersedes (a stopped review).

### Python bindings (arko_workflows)

Behind the `python` feature. The defaults match Rust (auto-checkpoint on, approve all, automatic recovery, no parallel cap, fail-fast off, 30 s grace, automatic `wait_for_message` limit). Differences. The client always comes from `LLMClient::from_env()`, agents load from configuration files, memory offers only global tiers, and messaging is always all-to-all. Not exposed, hooks, custom tools, interceptors and `interception_failure_policy`, verifiers, distillers, services, freshness, voting, durable steps, graceful stops (`stop`, `stop_run`, `stop_agent`, `stopped_runs`, `restart`, `stop_on_signals`, `cancel` is exposed), user-input handlers, the `Handler` approval policy, `ResolveRefusal`, `budget_warning`, `agent_budget`, `metadata`, the messaging policy, `event_flush_interval`, `PersistenceConfig::user_data`, `spawn_agent`, workflow pause and resume, event subscription, and operator notes (`post_note`, `notes_by_name_only`, `control_folder`). See [Python bindings](../arko_workflows/INDEX.md#python-bindings).

### Persisted data (arko_workflows)

- Checkpoints have no format version. Their optional fields default and unknown fields are ignored, so older checkpoints load, including those that hold their tasks inline. The next checkpoint writes those tasks as task versions.
- A checkpoint without a run ID skips the run-ID check. One without a cost ledger resumes with nothing spent (`budget_spent` is not read). One without `notes` (written before notes existed, or by a run without any) resumes with none.
- Checkpoints may refer to task files that earlier versions wrote beside them (`agents/<task id>.json`). Those files are read when referenced and are not deleted by retention.
- `Failed` tasks with the reason `Task cancelled by user.` load as `Cancelled`.
- A graceful stop is saved as `"cause": "Requested"` with a `stop` field (`kind`, `reason`) beside it, and stopped runs' records carry `stopped_by`. Versions from before graceful stops ignore both and read the stop as a cancellation. Checkpoints from those versions load without stop marks.
- A run stopped alone carries `stopped_by` with an `intent` (`retire`, `relaunch`), also on a running run's record in the live checkpoint while its stop is under way. Records saved before runs could be stopped alone load with no intent, and versions from before ignore it.
- `resume_from` refuses a missing or unreadable task file, another workflow's checkpoint, a run-ID mismatch, and a checkpoint-only memory tier.
- While event-log writes fail, up to 64 MiB of events stay buffered for the next write. Older ones are dropped with a warning.

### Example binaries (arko_workflows)

#### `basic_workflow`

No flags. `LLMClient::from_env()`, agent configurations `configs/architect.json` and `configs/coder.json` (not shipped, supply them) and an existing `./my-project` directory, all relative to the current directory. Budget $5 (warning at $4). The default `ApproveAll`.

#### `deepcoder`

| Flag | Default | Notes |
|---|---|---|
| `--workspace`, `--spec` | Required | |
| `--state-dir` | `<workspace parent>/.arko-deepcoder-runs/<name>-<time>-<id>` | Must be outside the workspace. One already holding a run is refused without `--resume`. Persistence is on unless `--ephemeral`. |
| `--resume PATH` | New run | A checkpoint file, or a run directory. Its directory is the state directory. The workspace and specification must match the run's. Conflicts with `--ephemeral`. |
| `--budget` | 50.0 | Warning at 80%. |
| `--max-minor-corrections`, `--max-efficiency-corrections`, `--max-final-corrections` | 20, 3, 4 | The library's verifier caps are 3, 2 and 3. |
| `--shell-timeout-seconds`, `--shell-output-bytes` | 1800, 65,536 | |

`deepcoder` configures only OpenAI (the `OPENAI_OAUTH_*` variables, then the Codex `auth.json`, then `OPENAI_API_KEY`), auto-approves every tool, and asks `ask_user` questions at the terminal. Its agents use `LLMErrorPolicy::PauseAndAwaitApproval`, so a failed model request goes straight to the workflow's automatic recovery. It declares Linux and macOS only, so `build` fails on other hosts. See the [DeepCoder README](../arko_workflows/examples/deepcoder/README.md).

#### `forge`

`--output` is required for a new run (a new or empty directory), and the use case
comes from `--use-case`, `--use-case-file`, or the terminal. The agents run in the installed Claude
Code CLI by default (`--backend api` uses `ANTHROPIC_API_KEY`), an architect (`claude-opus-5-5`,
xhigh) and up to `--max-active-calls` (4) explorers (`claude-sonnet-5-5`, high). `--involvement`
is `clarify` (one batch of questions) or `none`. `--trial-runs` (0) allows live runs of the new
workflow. The run's settings are saved in its state directory (`~/.arko/forge/runs/...`), so
`--resume <state dir>` takes no other flags. Lessons are kept in `~/.arko/forge/lessons.md`. See
the [forge README](../arko_workflows/examples/forge/README.md).

#### `complete_workflow`

`--data-dir` is required. `--dry-run` builds and checkpoints without
credentials or model requests. `--resume` restores a checkpoint. And `--owner` supplies the
answer to the writer's user question. See [Complete workflow example](#complete-workflow-example).

#### `runctl`

`note RUN_DIR TEXT` (or `--file PATH`) appends a standing note from the operator to
every agent of the run in `RUN_DIR` (a workflow's persistence directory). `--to a,b` or
`--except c` sets the audience, `--once` makes it a one-time note, and `--from-agent NAME` or
`--from-host` changes the source. It needs a workflow built with `control_folder()` and fails
otherwise (exit status 1). The workflow posts the note at its agents' next turn boundaries or when
it resumes. `status RUN_DIR` prints the latest checkpoint's runs, spend, newest notes and the
control-folder notes not yet posted. `stop RUN_DIR --agent NAME` (or `--run ID`) asks the live
workflow to stop those runs alone. Each gets a 600-second wrap-up (`--wrap-up SECS`, or `--now`
for none) and is retired unless `--relaunch`. `--reason TEXT` is what the runs are told. It makes
no model requests and reads no credentials. See
[Operator notes](../arko_workflows/INDEX.md#operator-notes) and
[Stopping one agent run](../arko_workflows/INDEX.md#stopping-one-agent-run).

#### `index_benchmark`

Subcommands `prepare` (fetch the pinned repositories of `repos.json`,
make them read-only, build their arko_index indexes with the `arko_index_cli` next to the
binary or `--index-cli`), `generate` (writer and verifier agents, with a terminal unless
`--writer-tools index`, while `assemble` rebuilds the task set without agents), `prepare-variants`
(`--variants FILE`, one index per variant arm and repository, with that variant's `index2`
flags), `validate`, `run` and `report`. Every one takes `--out DIR`. Every agent
runs on `--model` (`gpt-6-astra:xhigh`) in Fast mode (`--no-fast` for standard) over the ChatGPT
login of `--codex-home` (repeatable, failover between accounts, first preferred). `--index-cli` on `prepare`, `prepare-variants`, `generate` and `run`. `run` defaults, `--arms A,B`, `--samples 1`,
`--concurrency 24`, `--max-output-bytes 16384`, `--command-timeout 120`, `--max-shell-calls 80`,
`--max-input-tokens 8000000`, `--attempt-timeout-minutes 60`, `--max-attempts 3`. Resumable by
`--name`, stopped gracefully by a signal or a `STOP` file. `--mock` runs every role against a
local scripted model. See the
[benchmark README](../arko_workflows/examples/index_benchmark/README.md).


## arko_index

Syntax-aware partitioning of source trees into chunks (tree-sitter for 53 languages, a bounded fallback for other UTF-8 text) with context headers, offline BM25 search, opt-in dense and hybrid search through a caller-supplied embedder, and durable watchers that report chunk deltas. The library makes no network calls or model requests and reads no credentials. It depends on no other arko crate (its examples use arko_ai for Voyage).

See [arko_index README](../arko_index/README.md) · [arko_index reference](arko_index-reference.md) · [arko_index languages](arko_index-languages.md) · [arko_index tests and evaluation](arko_index-evaluation.md)

### Map (arko_index)

`bench_v1` holds the chunking and polling-watcher API, `search` and `index2` the two index formats, `dense`, `vector` and `hybrid` embeddings, vector storage and lexical, dense and hybrid search, and `watch2` (Linux) the inotify watcher. The crate root re-exports the file selection types and `threads()`.

| Item | What it is | Code | Docs |
|---|---|---|---|
| `bench_v1::Options` | `max_bytes`, the source-byte bound of a chunk (default 4,096, values below 4 rejected). A rendered chunk is also capped at 24,000 bytes. `front_matter_description` (default `false`) adds the front matter's description to Markdown scopes. | `arko_index/src/bench_v1.rs` | [Boundaries and identities](arko_index-reference.md#boundaries-and-identities) |
| `bench_v1::Chunk`, `SourceSpan`, `CHUNK_FORMAT` | A chunk, with `id`, `path`, `start_byte`, `end_byte`, `start_line`, `end_line`, `language`, `symbol_path`, `kind`, `token_estimate`, `scope` (enclosing definitions or Markdown headings, metadata only, added in format 2), and `header()`. | `arko_index/src/bench_v1.rs`, `arko_index/src/headers.rs` | [Context headers](arko_index-reference.md#context-headers) |
| `bench_v1::chunk_str`, `chunk_file` | Chunk one text (under a given path) or one file. | `arko_index/src/bench_v1.rs` | [Chunking](arko_index-reference.md#chunking) |
| `bench_v1::chunk_str_with_metadata`, `chunk_file_with_metadata`, `FileChunks`, `FileMetadata` | The same with a Markdown file's metadata region (front matter, MDX import/export lines), which its chunks leave out (chunk format 3). | `arko_index/src/bench_v1.rs`, `arko_index/src/front_matter.rs` | [arko_index languages](arko_index-languages.md#markdown-and-mdx) |
| Languages and rules | 53 languages by extension and file name, a rules table per grammar (`rules/tables.rs`, from `tags.scm` where a grammar has one). | `arko_index/src/languages.rs`, `arko_index/src/rules/` | [arko_index languages](arko_index-languages.md) |
| `bench_v1::chunk_dir[_with]`, `chunk_dir_each[_with]`, `chunk_dir_map[_with]` | Chunk every selected file in path order, collected, streamed to a sink, or mapped on worker threads, with identical output at any thread count. `_with` takes `FileRules` and returns the `Report`. | `arko_index/src/bench_v1.rs`, `arko_index/src/par.rs` | [Chunking](arko_index-reference.md#chunking) |
| `FileRules` | `respect_ignore` (`false`), `exclude`, `include` (gitignore patterns, empty), and `validate()`. | `arko_index/src/files.rs` | [File selection](arko_index-reference.md#file-selection) |
| `Report`, `Skipped`, `SkipReason` | What a walk indexed and skipped, with reasons, per-source ignore counts and warnings (such as files whose parser failed and were chunked as plain text), and `summary()`, `notable(limit)`. | `arko_index/src/files.rs` | [File selection](arko_index-reference.md#file-selection) |
| `threads()` | Worker count, with `ARKO_INDEX_THREADS`, else the CPUs the process may use. | `arko_index/src/par.rs` | [Environment variables](arko_index-reference.md#environment-variables) |
| `search::index[_with]`, `search::search`, `search::outline` | The JSON index (`index.json`), loaded whole by a search, with at most ten hits. | `arko_index/src/search.rs` | [Search](arko_index-reference.md#search) |
| `index2::index[_with]`, `index2::build`, `BuildOptions`, `index2::search`, `index2::dense_info`, `BuildStats` | The disk-resident index, generations published through `CURRENT2`, bounded build memory, searches that read only the query terms' postings (same results as `search` unless built with `lexical_scope`). `build` with `BuildOptions::dense` publishes the generation, then embeds the chunks (or leaves that to `index2::embed` with `DenseBuild::embed_later`). | `arko_index/src/index2.rs` | [Search](arko_index-reference.md#search) |
| `index2::embed`, `index2::status`, `IndexStatus`, `dense::EmbedState`, `dense::StopSignal` | Embedding runs. The vectors of the current generation, stored and committed request by request while the index is searched, superseded by a newer generation, resumed from the embedding cache after a stop (with its reason) or a crash. `status` gives the current generation's coverage, state and progress. | `arko_index/src/index2.rs` | [Embedding runs](arko_index-reference.md#embedding-runs) |
| `dense::Embedder`, `ModelDescriptor`, `HashEmbedder`, `DenseConfig`, `DenseBuild`, `EmbedStats`, `render`, `render_with`, `EmbedText` | The embedding interface providers implement, the model's identity (the cache key), the offline test embedder, the per-index settings (`dense.json`), one build's embedding settings and statistics, the embedded text and its parts. | `arko_index/src/dense/` | [Dense and hybrid search](arko_index-reference.md#dense-and-hybrid-search) |
| `dense::local::LocalEmbedder`, `LocalModel`, `Precision`, `Device`, `Placement`, `MODELS` | Feature `local-embed`, embedding models run on the CPU in pure Rust (CodeRankEmbed by default, Jina's code model), pinned by commit and SHA-256, in int8 by default or f32, and on a GPU in f32, f16 or bf16 with `local-embed-cuda` or `local-embed-gpu`. | `arko_index/src/dense/local/` | [arko_index local embeddings](arko_index-local-embeddings.md) |
| `vector::VectorStore`, `CaprockStore`, `StoreConfig`, `Quantization`, `VectorIndex`, `Filter` | Vector storage (upsert, delete, top-k with a path-prefix or language filter, atomic persist) and its implementation on Caprock, int8 codes scanned flat with full-precision rescoring by default, f32 codes and Vamana graphs opt-in, one store per model and settings, updated in place by embedding runs. A path prefix matches whole components. | `arko_index/src/vector.rs` | [Dense and hybrid search](arko_index-reference.md#dense-and-hybrid-search) |
| `hybrid::search`, `SearchRequest`, `Mode`, `Fusion`, `SearchResults` | Lexical, dense or hybrid search of one generation, with filters, headers and the query's tokens, and reciprocal rank fusion by default. | `arko_index/src/hybrid.rs` | [Dense and hybrid search](arko_index-reference.md#dense-and-hybrid-search) |
| `bench_v1::Delta` | `sequence`, `added`, `removed` (IDs), `changed`. | `arko_index/src/bench_v1.rs` | [Watchers](arko_index-reference.md#watchers) |
| `bench_v1::open_watch[_with]`, `Watcher` | The polling watcher (any platform), with `next_change`, `ack`, `snapshot`, `flush`, `report`, `rules_changed`. | `arko_index/src/watch.rs` | [Watchers](arko_index-reference.md#watchers) |
| `watch2::Watcher2` | Linux, inotify plus metadata reconciliation, a record store and a write-ahead journal, with `open[_with]`, `next_change`, `next_change_to`, `ack`, `report`, `rules_changed`, `file_count`, `stats`. | `arko_index/src/watch2.rs` | [Watchers](arko_index-reference.md#watchers) |
| `bench_scan[_with]` | Unix, prints timings of the scan primitives. | `arko_index/src/scan.rs` | [Command line](arko_index-reference.md#command-line) |
| Parsing | Mesquite, a pure-Rust port of tree-sitter 0.25.10, and its grammar crates (no C compiler needed). The build script records their locked versions for the watchers' engine digest. | `arko_index/src/syntax.rs`, `arko_index/build.rs` | [Parsing](arko_index-reference.md#parsing) |

### Defaults (arko_index)

- Every regular file under the root is indexed, hidden and gitignored files included. Built-in exclusions, `.git` directories and files, directories with a valid `CACHEDIR.TAG` (not the root), and the tool's state directories. Ignore files are read only with `respect_ignore`, and never above the root.
- Binary, non-UTF-8, unreadable and undecodably named entries are skipped and reported, and the walk goes on. Symbolic links are not followed.
- `max_bytes` 4,096, at most ten search hits, all available CPUs (chunking, `index2`, `watch2`), a reorder window of 32 MB of source per thread, and an `index2` postings budget of 256 MB in the CLI.
- Dense search is opt-in (`BuildOptions::dense` is `None`, CLI `--embed`). Local models need the `local-embed` feature (off). The local embedder runs in int8 (`Precision::Int8`) on the CPU (`Device::Auto`), uses `threads()` workers and 8,192 tokens per forward pass, and embeds texts over 8,192 tokens in windows of that size. GPUs need `local-embed-cuda` or `local-embed-gpu` (off), `Device::Auto` puts f32 on CUDA or a discrete wgpu GPU when present, a GPU runs f32 unless f16 or bf16 is named, with 4,096 padded tokens per forward pass. Once chosen, the model is saved per index (`dense.json`) and the CLI's later builds follow it. A build publishes for lexical search before embedding, `build` embeds before it returns (`DenseBuild::embed_later` is `false`), and the CLI's `index2` embeds in a background process (`--wait-embeddings` waits). No token cap (`DenseBuild::max_tokens` `None`), and a cap applies to each embedding run. Vectors kept as int8 codes, searched flat, candidates rescored at full precision (`Quantization::F32` and `VectorIndex::Graph` opt-in). `SearchRequest`, lexical, ten results, 50 hybrid candidates per side, reciprocal rank fusion (k = 60, weights 1), no filter. The CLI's `search2` is hybrid when the index has embeddings. Render version 1 (path, scope, trimmed source, `EmbedText::PathScopeSource`). Lexical search indexes path and source, the scope only with `BuildOptions::lexical_scope`.

### Persisted data (arko_index)

- State directories hold the `.arko-index-state` marker and `report.json`, plus `index.json` (`search`), `CURRENT2` and `index2-*` generations (`index2`, with `.index.lock` and `.embed.lock`, and with embeddings each generation also holds `dense.json`, `vkeys.dat`, `vhash.dat` and, until it is embedded, `embed.spool`), `vectors-<digest>/`, `dense.json`, `embed-cache/` and `embed-progress.json` (builds with embeddings and embedding runs, and the examples keep local model files in the Hugging Face cache), `watch.json` (`open_watch`) or `watch2.snap`, `watch2.recs.N` and `watch2.wal` (`Watcher2`).
- Chunk JSON format 2 adds `scope` (written only when not empty). Format 1 data loads with an empty scope, and a watcher reports chunks that gain a scope as `changed` once. Format 3 adds `FileMetadata` lines (Markdown front matter) to the `chunk` output.
- Markdown scopes start with the front matter's title (or the first `# H1`), without the description (`Options::front_matter_description` is `false`).
- arko_index is unpublished and its on-disk data has no compatibility guarantee between versions. The SQLite databases of the earlier, model-chunked arko_index are not readable. A watcher reuses its parse cache only when the engine digest (crate version, chunking sources and locked tree-sitter versions) and `max_bytes` match, and otherwise re-chunks without spurious deltas.

### Example CLI (arko_index)

`arko_index_cli` (`cargo run --release -p arko_index --example arko_index_cli -- COMMAND`), `chunk`, `outline`, `index`/`search`, `index2`/`search2`, `embed`, `status`, `watch`, `watch2` (Linux) and `bench-scan` (Unix). Commands that walk a tree take `--respect-ignore`, `--exclude PATTERN`, `--include PATTERN`, `--report FILE` and `--front-matter-description`. `index2` also takes `--lexical-scope`, `--embed MODEL` (Voyage, `local[:NAME][:PRECISION]`, `hash`), `--dimensions`, `--embed-text`, `--embed-config`, `--quantize`, `--vector-index`, `--max-embed-tokens`, `--no-embed` and `--wait-embeddings` (with embeddings it returns once lexical search is ready and embeds in a background process unless told to wait), and `--owner-pid PID` (the background process ends when its owner ends, by default the session leader on Unix, the parent on Windows). `embed STATE_DIR` runs the embedding in the foreground (`--max-embed-tokens`, `--owner-pid`), `status STATE_DIR` prints the coverage and progress, and `watch`/`watch2` take `--index INDEX_STATE_DIR` (and `--lexical-scope`) to rebuild and embed an index as the tree changes. `search2` takes `--mode`, `--limit`, `--candidates`, `--fusion`, `--rrf-k`, `--lexical-weight`, `--dense-weight`, `--path-prefix`, `--language` and `--text`. `dense_live_check` runs the retrieval evaluation with Voyage (spends tokens) or a local model, and `local_embed_bench` measures a local model's throughput and index size. See [Command line](arko_index-reference.md#command-line).

## Across crates

These defaults interact across the crates:

- Model requests have no total deadline. A response is awaited while its connection
  stays open. The opt-in stream-idle limits are `LLMClient::with_stream_idle_timeout`,
  and `LLMRetryConfig::stream_idle_timeout_ms`. The other relevant
  limits are the 30-second connect timeout, the 60-second OAuth refresh timeout, the
  `arko_agents` host tool timeout (20 minutes by default), `wait_for_message`'s default
  270-second limit, the terminal wait cap, and the workflow's 30-second cancellation grace period. A graceful stop's wrap-up and a
  `StopAwareCommand`'s stop graces (10 s, then 5 s after `SIGTERM`) count only once a stop
  is under way. Nothing asks for a wrap-up by default.
- Retries are layered. `arko_ai` only resends an expired ChatGPT OAuth request once.
  `arko_agents` owns model-call retries and pauses at plan usage limits. `arko_workflows`
  answers those pauses automatically by default.
- Error kinds travel with the failure. `arko_ai::Error::kind` classifies a failure
  (`Authentication`, `ContextOverflow`, `Billing`, ...) beside its retry class.
  `arko_agents` records it on error-recovery pauses, failures and `ErrorEncountered`.
  `arko_workflows` exposes it as `AgentResult::failure_kind`, on `AgentPause::ErrorRecovery`
  and in hook metadata (`error_kind`). Nothing stops a whole run on a kind by default.
- Tool results are guarded, not capped. The `arko_agents` overflow guard (on) shortens
  a turn's results only when the next request would certainly be over the model's input
  limit (otherwise the provider decides, and an overflow it reports is recovered from),
  after `arko_workflows` distillers, which skip outputs over their own model's limit.
  Per-agent and per-tool caps are opt-in.
- Approvals are layered. A bare agent configuration asks for approval for calls other
  than `ask_user`. A workflow uses `ApproveAll` by default. Workspace path policy applies
  only to parameters marked as paths. Terminal, MCP and command tools are not confined by
  that file-path policy.
- Caching is enabled by default. Each conversation has its own cache key.
- Credentials are explicit. Only `LLMClient::from_env` and
  `EmbeddingClient::from_env` read environment credentials. `WorkflowBuilder` does
  not. arko_index needs none. The Python workflow binding constructs its client with
  `LLMClient::from_env()`.
- Costs use billed amounts. Budgets and task totals use `UsageInfo::billed()`.
  ChatGPT OAuth subscription calls contribute zero billed cost while their list-price
  estimate remains in `total_cost`.
- Budgets pause, not cancel. A workflow's default `BudgetExceededPolicy::Pause` holds
  each agent over its budget before its next model request (the `arko_agents`
  `TurnContextSource::hold_request` hook, cause `BudgetExceeded`). A run the host started
  returns `AgentPaused`, and nested runs wait for `Workflow::set_budget`. A host that can't
  raise the budget live should stop on `budget_blocked()` or that `AgentPaused` and resume
  with a higher budget later. Set `Cancel` for the earlier hard stop.
- Persistence is durable but compatible. Task files, checkpoints, event logs and
  request traces can contain full conversations and use default file permissions. New
  serialized fields have defaults and unknown fields are ignored. Crate-specific
  migration rules are documented in each persisted-data section. arko_index's index and
  watcher state is the exception. No compatibility guarantee while the crate is
  unpublished.

## Complete workflow example

[`complete_workflow.rs`](../arko_workflows/examples/complete_workflow.rs) is a runnable
workflow that combines explicit agent configuration, built-in tools, memory, messaging,
completion verification, tool-result distillation, persistence, and resuming. It supports a
credential-free `--dry-run` that validates the setup and writes a checkpoint, live execution
over ChatGPT OAuth, and `--resume` from a checkpoint. The source is the canonical example.
Its comments identify the defaults and the settings that are intentionally explicit.
