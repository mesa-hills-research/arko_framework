# Defaults reference

Every public setting of the four crates, with its default and what leaving it unset
means, so that code built on arko_framework can configure everything explicitly. Each
section also lists the opt-ins, the behaviours that are easy to assume wrongly, and what
data saved by earlier versions loads as. This page is a reference: the linked guides
have the details and examples.

- [arko_ai](#arko_ai) · [arko_agents](#arko_agents) · [arko_workflows](#arko_workflows) ·
  [arko_index](#arko_index) · [Across crates](#across-crates) ·
  [Example binaries](#example-binaries)

Conventions:

- **Default** is what `Default::default()`, the constructor, or the builder gives when a
  setting is not set. **Omitted** is what a field missing from JSON or YAML loads as. The
  tables say so where the two differ.
- For an `Option` setting whose default is `None`, an omitted field is `None` too. Where
  the default is `Some(..)`, an omitted field takes that default, and JSON `null` gives
  `None`, unless a row says otherwise.
- Unknown fields are ignored when loading, so files with fields that were since removed
  still load.

## arko_ai

Guide: [arko_ai README](../arko_ai/README.md).

### Client and credentials

| Setting | Default | Notes |
|---|---|---|
| `LLMClient::new()` | No provider | A request to a provider without credentials fails with `Error::MissingApiKey`. The client reads no credentials from the environment unless `from_env` is called. |
| `LLMClient::from_env()` | — | Reads `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (and optionally `OPENAI_OAUTH_REFRESH_TOKEN`), otherwise `OPENAI_API_KEY`; `ANTHROPIC_API_KEY`; `GOOGLE_API_KEY`, otherwise `GEMINI_API_KEY`; `OPENROUTER_API_KEY`. Empty values count as unset. Setting only some of the `OPENAI_OAUTH_*` variables is `Error::AuthenticationFailed`, never a fallback to the API key. `from_env_with(lookup)` reads through a function instead. |
| Refresh token from the environment | Kept in memory | If it matches the refresh token in the Codex `auth.json` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`) for the same account, the client uses that file as its credential store instead: the file's tokens, with rotations written back. Otherwise the first rotation logs a warning, since the environment's token stops working once used. |
| `with_openai_chatgpt_oauth(access, refresh, account)` | In memory only | Rotated tokens are not saved and no warning is logged; save them from `openai_chatgpt_oauth_credentials()`. |
| `with_codex_auth_file(&file)`, `CodexAuthFile::client()` | — | Every refresh re-reads `auth.json` first, and rotated tokens are saved at once. See [ChatGPT OAuth credentials](../arko_ai/README.md#chatgpt-oauth-credentials). |
| Credentials without a refresh token | — | The access token is used until the server rejects it, then `Error::AuthenticationFailed`. |
| Token-refresh rate-limit retries | Off | `with_openai_chatgpt_oauth_rate_limit_retries(.., true)` or `OpenAIClient::with_oauth_rate_limit_retries(true)`: up to 8 retries of an HTTP 429, backing off from 30 s to 15 min; a server hint is honoured up to 15 min per wait. Applies only to explicit `refresh_openai_oauth_token()` calls: the refresh after a 401 and `refresh_openai_oauth_token_if_expiring`, which the arko_agents runtime uses, never wait. |
| `with_openai_originator`, `with_openai_user_agent` | Not sent | No `originator` or `User-Agent` header by default. |
| `with_request_trace(RequestTrace::new(dir))` | Off | Writes every OpenAI request and response, full prompts included (credential headers redacted), with default file permissions. Other providers are not traced; the first request to each logs a warning. |
| `with_openai_request_hook(hook)` | None | Adjusts OpenAI headers and body before sending. |
| `OpenAIClient::with_credential_store(Some(store))` | None | A `ChatGptOAuthStore` that keeps rotated tokens in step with other storage (`CodexAuthStore` for a Codex `auth.json`). `with_oauth_token_url` sets the token endpoint (default `https://auth.openai.com/oauth/token`); `LLMClient::with_openai_auth(OpenAIAuth)` installs either kind of credentials. |
| `with_stream_idle_timeout(duration)` | Off | Applies to every provider client of the `LLMClient` and can't be cleared on it again. See [Retries and timeouts](#retries-and-timeouts). |
| Provider clients | Base URLs: `https://api.openai.com/v1` (API key), `https://chatgpt.com/backend-api/codex` (ChatGPT OAuth), `https://api.anthropic.com`, `https://generativelanguage.googleapis.com/v1`, `https://openrouter.ai/api/v1` | `OpenAIClient`, `AnthropicClient`, `GoogleClient` and `OpenRouterClient` take `with_base_url` and `with_stream_idle_timeout(Option)`; install them with `with_openai_client` and its siblings. OpenAI uses the Responses API, Gemini the Interactions API, OpenRouter the Chat Completions API. |

The HTTP transport is fixed: a 30-second connect timeout, no total request deadline,
no redirects (a 3xx response is an error, so credentials never reach another host),
pooled connections closed after 90 seconds idle, HTTP/1.1, rustls with the OS trust
store, and the system's proxy settings (`HTTPS_PROXY` and the like).

### `GenerationConfig`

Every field defaults to `None`, and omitted fields take their defaults, so a partial JSON
or YAML config is valid. One config can serve models of several providers: a setting the
model's provider has no use for is dropped with a warning (logged once per provider and
setting), while a value the provider can't accept for a setting it does use is an error.
Exceptions: `openai_service_tier` over ChatGPT OAuth, and `temperature` and
`parallel_tool_calls: Some(false)` on Gemini, are errors; Gemini ignores
`ToolDefinition::strict`, and Claude on OpenRouter accepts `anthropic_summarized_thinking`
and `long_context` without sending anything.

| Field | `None` means | Allowed values and notes |
|---|---|---|
| `temperature` | Not sent: the model's default | An error for OpenAI reasoning models and for Gemini. Dropped with a warning on Claude models that take no sampling parameters (Fable 5.x, Opus 4.7 and later, Sonnet 5); other Claude models accept only 1.0 while thinking. `GenerationConfig::new(t)` is deprecated for `default().with_temperature(t)`. |
| `max_tokens` | The model's own maximum: OpenAI with an API key, Anthropic and Gemini send the model's registered output limit; OpenRouter sends none; ChatGPT OAuth never sends one | Over ChatGPT OAuth a set value is ignored with a warning, since the backend rejects it. On OpenRouter a set value also restricts routing to endpoints that accept it. |
| `verbosity` | Not sent | `low`, `medium`, `high`. OpenAI GPT-5 family and GPT-6 only (an error on other OpenAI models); dropped on Claude and Gemini; on OpenRouter it can't be combined with `reasoning.effort` on Claude. |
| `reasoning` | The model's default; see [Reasoning](#reasoning) | Each provider takes its own `ReasoningConfig` variant; another provider's variant is dropped with a warning. |
| `openai_reasoning_mode` | Not sent | `standard`, `pro`; GPT-5.6 and GPT-6 only. |
| `anthropic_thinking_enabled` | The model's default | `Some(true)` without a reasoning config selects adaptive thinking (an error on models without it). `Some(false)` sends `thinking.type: "disabled"`: an error on Fable 5.x and Opus 5.5, and together with an Anthropic reasoning config. |
| `prompt_caching` | `Conversation::enable_cache` (default `true`) | See [Prompt caching](#prompt-caching). |
| `anthropic_summarized_thinking` | The model's default display | `Some(true)` sends `thinking.display: "summarized"` on models that omit thinking text by default (Fable 5.x, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5); an error without active thinking. |
| `long_context` | — | `Some(true)` is accepted on Claude 4.6 and later, where 1M context is standard and nothing extra is sent; an error on other Claude models, Sonnet 4 and 4.5 included. |
| `tool_config` | No tools | See [Tool calls](#tool-calls). |
| `interleaved_thinking` | Automatic on Claude 4.6 and later while adaptive thinking is active | On the Anthropic provider, `Some(true)` needs a model that supports it and active thinking (not a manual budget on Opus 4.6); on OpenRouter it adds the beta header. `Some(false)` is an error while adaptive thinking is active on 4.6 and later, and for Claude on OpenRouter. |
| `openai_service_tier` | Not sent: the project's default tier | `auto`, `default`, `flex`, `priority` (the `Fast` variant). Flex and Fast only on models that offer them; API key only. |
| `anthropic_cache_ttl` | No `ttl`: 5 minutes | `"5m"`, `"1h"` (1-hour cache writes cost 2x input). An error while caching is off. Also used for Claude on OpenRouter. |
| `anthropic_fast_mode` | Nothing sent | `Some(true)` on Opus 5.5, Opus 5 and Opus 4.8 only; billed at fast-mode rates. |
| `anthropic_inference_geo` | Not sent: the workspace's default | `global`, `us` (1.1x price); Claude 4.6 and later. |
| `openrouter` | The client's default options | `OpenRouterOptions`, merged field by field over `OpenRouterClient::with_default_options`. |

### Reasoning

What is sent without `GenerationConfig::reasoning`:

| Provider | Sent |
|---|---|
| OpenAI, API key | No effort, so the model's default applies (none on GPT-5.1, 5.2 and 5.4, which then don't reason). GPT-5.6 reasoning models get `reasoning.context: "all_turns"`. |
| OpenAI, ChatGPT OAuth | `effort: "medium"` on GPT-5 and GPT-6 models, as Codex sends; `context: "all_turns"` on GPT-5.6 and GPT-6. |
| Anthropic | Nothing: the model's default. Fable 5.x, Opus 5.5, Opus 5 and Sonnet 5 think adaptively by default (Fable 5.x and Opus 5.5 can't turn it off); earlier models don't think. |
| Gemini | No `thinking_level` or `thinking_summaries`: the model's default level (medium on Gemini 3.5 to 3.8 Flash, minimal on 3.1 and 3.5 Flash-Lite, high on 3 Pro, 3 Flash and 3.1 Pro, dynamic on 2.5) and no thought summaries. |
| OpenRouter | Nothing: the model's default from OpenRouter. |

Every OpenAI request sets `store: false`, and reasoning models also ask for
`reasoning.encrypted_content`, which later requests replay.

| `ReasoningConfig` | Values and checks |
|---|---|
| `openai(effort, include_summaries)` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; an effort the model lacks is an error. |
| `anthropic(budget_tokens)`, `anthropic_with_effort(budget, effort)` | Manual budget: at least 1,024 and below `max_tokens` unless thinking is interleaved; at most 52,000 on Opus 4.5 and Sonnet 4.6 and 102,400 on Opus 4.6; rejected on Fable 5.x, Opus 4.7 and later, and Sonnet 5. Effort `low` to `max`, per model. |
| `anthropic_adaptive()`, `anthropic_adaptive_with_effort(effort)` | Claude 4.6 and later. |
| `gemini(level, include_summaries)` | `minimal`, `low`, `medium`, `high` (`"MINIMAL"` to `"HIGH"` in JSON), validated per model. `include_summaries: false` sends `thinking_summaries: "none"`. |
| `google(budget, include_summaries)` | Mapped to a level, then validated per model: negative = the model's default, 0 = minimal, up to 4,096 = low, up to 16,384 = medium, above = high. |
| `openrouter(OpenRouterReasoning)`, `openrouter_effort(effort)` | The only variant OpenRouter takes; `effort` or `max_tokens`, not both. `ReasoningConfig::to_openrouter()` converts the others explicitly. |

### Tool calls

| Setting | Default | Notes |
|---|---|---|
| `ToolConfig::tool_choice` | `None`: the provider's `auto` (Anthropic sends `auto` explicitly) | `Auto`, `Required`, `None`, `Specific(name)`. The Anthropic provider rejects a forced choice before sending on Fable 5.1 and Opus 5.5 and with a manual thinking budget; `Model::supports_forced_tool_choice(config)` tells in advance, also for Claude on OpenRouter, which doesn't check. |
| `ToolConfig::parallel_tool_calls` | `None`: the provider's default, several calls allowed | `Some(false)`: OpenAI and OpenRouter send `parallel_tool_calls: false`, Anthropic `disable_parallel_tool_use`; Gemini fails with `Error::UnsupportedFeature`. |
| `ToolDefinition::strict` | `None` | Sent to OpenAI, Anthropic and OpenRouter (with the structured-outputs beta header for Claude on OpenRouter); Gemini ignores it. |
| Arguments that aren't valid JSON | — | Kept as text in `ToolCall::invalid_arguments` with `arguments: null`; replayed verbatim to OpenAI and OpenRouter `openai/` models, as `{}` to the others. |

### Conversation

| Field | Default | Notes |
|---|---|---|
| `enable_cache` | `true` | `with_cache_disabled()`. Required when loading JSON. |
| `prompt_cache_key` | A random UUID per new conversation | `with_prompt_cache_key`. Sent as OpenAI's `prompt_cache_key`, as ChatGPT OAuth's `session-id`, `thread-id` and `x-client-request-id` headers, and as OpenRouter's `session_id` (at most 256 characters, else an error). A loaded conversation keeps its key; one saved without a key gets a new one. Give each agent or task its own conversation. |
| `system_message` | None | `with_system_message`. A `Role::System` message inside `messages` is an error on OpenAI, Anthropic and Gemini. |
| `openai_turn_state` | None | The ChatGPT OAuth routing token of the current user turn, kept and saved with the conversation. |

`add_assistant_message_from_response` leaves out classifier and content-filter
refusals, which must not stay in history. History that holds another provider's
reasoning artifacts is an error on Anthropic and Gemini, so a conversation can't switch
to them mid-way. `LLMClient::prompt_cache_lifetime(&model, &config)` reads only
`config.prompt_caching`, not `Conversation::enable_cache`.

### Prompt caching

Caching is on by default. Turning it off (`with_prompt_caching(false)` or
`Conversation::with_cache_disabled()`) takes effect only where the provider has a
switch:

| Provider | Caching | With caching off |
|---|---|---|
| Anthropic, and Claude on OpenRouter | Only through the `cache_control` arko sends | Nothing is cached |
| OpenAI GPT-5.6 and GPT-6, API key | `prompt_cache_options.mode: "implicit"` | `"explicit"` with no breakpoints: nothing is cached |
| OpenAI before GPT-5.6, API key | Automatic | Still cached; can't be turned off |
| ChatGPT OAuth, every model | Implicit, managed by the backend | Still cached; ignored with a warning |
| Gemini | Implicit, managed by Google | Still cached; can't be turned off |
| OpenRouter, other models | `prompt_cache_key` sent | No key sent; GPT-5.6 and later get `"explicit"` mode, other upstreams keep caching implicitly |

The conversation's key goes with every OpenAI and OpenRouter request either way; it
routes requests but doesn't turn caching on. `prompt_cache_lifetime` reports 5 minutes
for Anthropic (1 hour with `anthropic_cache_ttl: OneHour`) and a conservative 5 minutes
for OpenAI; Gemini and other OpenRouter models are `Unknown`. It reports `Disabled` for
models without caching, and with caching off on Anthropic, Claude on OpenRouter, and
GPT-5.6 and later with an API key.

### ChatGPT OAuth

Requests over ChatGPT OAuth go to the Codex backend and differ from API-key requests:

- They always stream; `generate_text` collects the stream.
- `max_tokens` is never sent: the model's own output limit applies.
- Caching is implicit and can't be turned off. The backend derives cache affinity from
  the `session-id` header, set to the conversation's `prompt_cache_key`.
- Without a reasoning config, `effort: "medium"` is sent to GPT-5 and GPT-6 models.
- `openai_service_tier` is an error.
- The input limit is capped at 272,000 tokens (`CHATGPT_OAUTH_MAX_INPUT_TOKENS`), or
  the model's limit if lower. `LLMClient::max_input_tokens(&model)` returns it; requests
  are not checked against it.
- The first `x-codex-turn-state` token the backend returns in a user turn is echoed on
  the turn's later requests (tool-result follow-ups and retries).
- Usage is billed as `BillingKind::Subscription`: `billed()` is zero, and the list-price
  estimate stays in `total_cost`. Plan usage from the `x-codex-*` headers is in
  `ModelCallUsage::plan_usage`.
- A request rejected with HTTP 401 `token_expired` is refreshed and resent once. A
  token refresh times out after 60 seconds.

See [OpenAI Responses](../arko_ai/README.md#openai-responses-replay-prompt-caching-and-chatgpt-oauth).

### Tool images and Gemini 3.x

`ToolResult::with_images` attaches images to a tool result. Where they go:

| Provider | Result images |
|---|---|
| OpenAI (API key and ChatGPT OAuth) | Inside `function_call_output.output` |
| Anthropic | Inside `tool_result.content` |
| Gemini 3 and later | Inside `function_result.result` |
| Gemini 2.x, and OpenRouter | After all of the turn's results, in a user turn, each after the text `Image from tool {name}, call {call_id}: {label}` |

Anthropic and Gemini accept only base64 images, not `Image::Url`. A model without image
input rejects any image before sending.

On Gemini 3 and later, content that follows tool results is folded into the last
`function_result`: the other content of a tool-result message (an `ask_user` answer, a
recovery note, environment details), and the text and image messages that follow it
before the next assistant message. Texts are appended to the result's text after a
blank line; images follow the result's own images. Anthropic sends the tool-result
message's own text in that user message, after the `tool_result` blocks, and the
following messages as separate user messages; OpenAI, OpenRouter and Gemini 2.x send it
all as separate user input. See [Tool images](../arko_ai/README.md#tool-images).

### Retries and timeouts

- arko_ai never retries a model call itself. The exceptions: a ChatGPT OAuth request
  rejected with 401 `token_expired` is refreshed and resent once, and with a credential
  store any other 401 re-reads the store and resends once if it holds newer
  credentials.
- `Error::retry_class()` is `Permanent`, `Transient`, `Capacity`, or `UsageLimit`.
  HTTP 408, 409, 429 and 5xx are retryable unless the body has a permanent code (quota,
  authentication, invalid request, context length, content policy). Server delay hints
  are capped at 30 days.
- `BackoffPolicy` has no default; `BackoffPolicy::new(initial, max)` jitters each delay
  between half and all of its step.
- There is no model-request timeout by default: a response is waited for as long as its
  connection stays open, since models can reason silently for over an hour (GPT Pro).
  `LLMClient::with_stream_idle_timeout(duration)` opts in to a limit on the gap between
  data on a streamed response; expiry is `Error::StreamTruncated` (transient). It covers
  neither the wait for headers nor responses that don't stream, and there is no total
  deadline. Make it generous, for example 2 hours. See
  [Long requests](../arko_ai/README.md#long-requests).

### Embeddings

| Setting | Default | Notes |
|---|---|---|
| `EmbeddingClient::new()` | No Voyage key | `with_voyage_api_key`, or `from_env()` (reads `VOYAGE_API_KEY`). `with_voyage_client(VoyageClient::new(key).with_base_url(url))` replaces the base URL `https://api.voyageai.com/v1`. |
| `EmbeddingConfig::new(provider, model_id)` | No default model | An unsupported model is `Error::ModelNotSupported` when embedding or counting tokens. |
| `EmbeddingConfig::output_dimensionality` | `None`: the model's default, 1,024 | 256, 512, 1,024, 2,048; another value is an error before sending. |
| `voyage_input_type` | `None`: not sent | `Document`, `Query` (for search queries). |
| `voyage_truncation` | `None`: `truncation: false` is sent | Voyage's own default is `true`. Either way an input over the 32,000-token limit fails locally with `Error::TokenLimitExceeded` before sending; split long inputs first. |
| `tokenizer_revision` | `None`: `main`, unpinned | Inputs are counted locally before sending, with the `voyageai/<model>` tokenizer downloaded from the Hugging Face Hub on first use (`HF_HOME`, `HF_ENDPOINT`). A failed download is an error, and nothing is sent. |
| `hf_token` | `None`: the Hugging Face token saved on the machine, if any | Serialized in clear text with the config. |

Requests are checked locally against 32,000 tokens per input, 1,000 inputs, and the
model's total tokens per request (unpublished for `voyage-code-4`, so not checked). See
[Voyage embeddings](../arko_ai/README.md#voyage-embeddings).

### OpenRouter

See [OpenRouter](../arko_ai/README.md#openrouter) for every option. The defaults:

- No attribution or metadata headers; `OpenRouterClient::with_app_attribution` opts in.
- No catalog is fetched: `Model::openrouter(slug)` has unknown limits and prices, and
  requests go unchecked. Once a catalog is loaded (`fetch_models`, `fetch_user_models`,
  `set_catalog`), unknown slugs (except `@preset/…`) and parameters the model doesn't
  list are errors.
- `session_id` is always sent; `prompt_cache_key`, and `cache_control` for Claude, only
  while caching is on.
- `auto_exacto`: `CacheAware`. On tool requests with caching on for a model that bills
  cache writes (from the catalog, or without one every `anthropic/` and `openai/` slug),
  `provider.sort: "price"` is sent, which keeps sticky routing and its cache.
  `AllowAutoExacto` never adds it; `DisableAutoExacto` always does on tool requests. An
  explicit `sort` or `order`, or a `:exacto`, `:nitro` or `:floor` variant, always wins.
- `fallback_models`: off. With it, another model may answer after a refusal error.
- `provider` preferences, `user` and `service_tier`: sent only when set; `auto_router`
  only for `openrouter/auto` and `openrouter/auto-beta`.
- `ReasoningConfig` variants of other providers are dropped with a warning, not
  converted.
- `Conversation::add_reasoning_effort_update` works only on OpenRouter; the other
  providers reject it.

### Persisted data

- `GenerationConfig`: configs with the old `temperature_specified` flag load (`false`
  means no temperature). A `temperature` without the flag is sent.
- `Conversation`: `model`, `enable_cache` and `messages` are required. The saved key is
  kept, including old `arko:conversation:<uuid>` keys. A registry model's saved data is
  replaced by the registry's current data on load.
- `ToolResult`: `is_error` is required; `images` defaults to none.
- `ReasoningArtifact`: the old single-item `OpenAIEncryptedReasoning` loads as a
  one-item list.
- `UsageInfo`: `billing` defaults to `Metered`. Anthropic records saved before cache
  counts were added back hold only uncached input in `input_tokens`.
- A saved conversation whose tool-result message also holds text or images is sent to
  Gemini 3 and later folded as described above, so its first request after an upgrade
  misses Gemini's cache from that message on.

## arko_agents

Guides: [arko_agents README](../arko_agents/README.md),
[interactive terminals](interactive-terminals.md), and
[context management](context-efficiency.md).

### `AgentConfig`

`AgentConfig::builder(id, name, llm_config)` requires `role_description` and
`instructions` and validates the configuration. `AgentConfig::load_from_path` doesn't
validate; the runtime validates again when a task starts. There is no `Default`.

| Field | Default and omitted | Notes |
|---|---|---|
| `id`, `name`, `role_description`, `instructions`, `llm_config` | Required | In `role_description`, `instructions` and the task prompt, text between `{{#output_based_only}}` and `{{/output_based_only}}` is kept only in output-based mode. |
| `tool_config` | `AgentToolConfig::default()` | Omitted inside a saved task: output-based mode, as below. See [Tools](#tools). |
| `environment_details_config` | Nothing included | See [Environment details and reminders](#environment-details-and-reminders). |
| `error_recovery_config.on_tool_execution_error` | `InformAgentAndContinue`: the error is the call's result | `PauseAndAwaitApproval` lets the turn's other calls run, then pauses once; `FailTask` fails at the first error. |
| `llm_error_policy` | `RetryThenPauseAndAwaitApproval { max_attempts: 10 }` | See [Model-call recovery](#model-call-recovery). |
| `llm_retry_config` | `LLMRetryConfig::default()` | See [Model-call recovery](#model-call-recovery). |
| `user_command_timeout_seconds` | `None`: a task waiting for a command waits indefinitely | Opt-in: in any awaiting state, this many seconds after the wait started without a decision, the task is returned `Interrupted`, and calls awaiting approval get a result saying the approval timed out. The deadline is set once per wait: `InjectContext` (a workflow delivers messages that way) and commands the wait ignores don't move it; a decision ends the wait, and the next wait gets its own. `UserCommand::HostDeciding` turns it off for the current wait; a workflow sends it for every wait of its agents, which it decides itself (handlers, automatic recovery's waits) or pauses for the caller, so the timeout never cuts a workflow's decision short. `Some(0)` expires at once. |
| `user_reminder` | `None` | `UserReminder { content, frequency }`, appended as `# Reminder`. |
| `skills_config` | `None` | The host passes it to `SkillManager::load` (arko_workflows does this itself); the runtime reads only `load_skill_interception`. See [Skills, MCP and terminals](#skills-mcp-and-terminals). |
| `monitoring` | Nothing monitored | See [Monitoring](#monitoring). |
| `refusal_fallback` | `None` | Opt-in `AgentLLMConfig` on the same provider (another provider is rejected): the refused input goes to it once. |
| `complete_task_summary_prompt_override` | `None`: the built-in description | Replaces the description of `complete_task`'s `summary` parameter. |
| `parsing_config_override` | `None`: tags `<`, `>`, `batched_tool_call`, escape prefix `!!--!!` | Output-based mode only. In JSON the three tag fields are required, and an omitted `escape_prefix` means no escaping. |

`AgentLLMConfig` saves `provider`, `model_id`, a snapshot of the model, and
`generation_config` (omitted: `GenerationConfig::default()`). On load, a model the
registry knows takes the registry's current data; an unknown ID keeps its snapshot; an
OpenRouter slug without one loads without catalog data; any other unknown ID fails to
load.

### Tools

`ToolRegistry::new()` registers `complete_task`, `read_file`, `write_file`,
`apply_diff`, `list_files` and `ask_user`.

| `AgentToolConfig` field | Default | Omitted in JSON | Notes |
|---|---|---|---|
| `available_tool_names` | `["complete_task"]` | Required when `tool_config` is given | The only tools offered and run, plus `complete_task`, which is always offered. In native mode a call to another tool gets an error result and `UnavailableToolAttempted`; in output-based mode its markup stays text. MCP, terminal and custom tools must be listed too. |
| `tool_calling_mode` | `Native` | `Native`, but `OutputBased` in the configuration saved inside a task | `OutputBased` is legacy: calls are parsed from text, one call or one batch of calls to one parallel-safe tool per response. MCP tools need `Native`. |
| `default_auto_approve_tool_use` | `false` | `false` | Every call waits for approval, `complete_task` included; `ask_user` never does. |
| `default_out_of_workspace_path_policy` | `AutoReject` | `AutoReject` | `RequireApproval` (asks even for auto-approved tools), `AutoApprove`. Checks `is_path` parameters against the working directory lexically; symbolic links are not followed, so it is not a sandbox. |
| `one_tool_call_per_turn` | `false` | `false` | `true` disables parallel calls where the provider can (OpenAI, Anthropic, OpenRouter) and runs Gemini's calls one at a time. `parallel_tool_calls: Some(false)` in the model's `tool_config` means the same. |
| `tool_timeout_seconds` | `Some(1200)` (`DEFAULT_TOOL_TIMEOUT_SECONDS`, 20 min) | `Some(1200)`; JSON `null` is no limit | The host-side limit of a call; never shown to the model. A call over it is stopped (its command killed) and handled as a tool error. `0` is rejected. Terminal tools, `run_workflow_agent` and `wait_for_message` are exempt. |
| `tool_specific_configs` | Empty | Empty | Per-tool `PerToolConfigOverride`, below. |

`PerToolConfigOverride` fields all default to `None`, which inherits the agent's
setting: `auto_approve_tool_use` (`Some(false)` on `ask_user` makes it ask),
`parallel_safe` (overrides the tool definition's), `out_of_workspace_path_policy`, and
`tool_timeout_seconds` (omitted: inherit; `null`: no limit; a number: that limit, also
for exempt tools).

A custom tool sets its own behaviour: `ToolDefinition::parallel_safe` (omitted in JSON:
`false`), `ToolParameterDefinition::is_path` (omitted in JSON: `false`; only such
parameters are checked against the workspace), and the `Tool` methods
`uses_host_timeout` (default `true`), `interruptible` (default `true`; `false` makes
`CancelTask` and `SuspendTask` wait for the call) and `allow_result_interception`
(default `true`).

How calls run:

- Every call of a response runs and gets exactly one result. Calls run in call order as
  under a fair read/write lock: a parallel-safe call starts once every earlier call that
  isn't parallel-safe has finished; any other call waits for every earlier call.
  `read_file`, `list_files`, `load_skill`, `read_memory`, `list_memories` and
  `list_recipients` are parallel-safe; the other built-in and workflow tools are not;
  an MCP tool is parallel-safe only when its server marks it `readOnlyHint: true`.
- A call awaiting approval counts as unfinished, so later calls wait behind it.
- Calls after one that ends the turn (`complete_task`, `ask_user`, a `FailTask` error,
  a middleware `FailTask` or `CancelTask`) are not run and their results say so;
  parallel-safe calls already running beside it finish. Among calls run once approvals
  are decided, only one that ends the task stops the others. `CancelTask` and
  `SuspendTask` also stop the calls still running (their results say so) unless one of
  them isn't interruptible. `complete_task` always runs alone, and after an earlier call
  of its turn was rejected or failed it doesn't complete the task.
- In native mode a response without a tool call doesn't end the task: a note asks the
  model to continue or call `complete_task`, and the response counts as an attempt of
  model-call recovery.
- In native mode, a `tool_config` in the model's `GenerationConfig` supplies
  `tool_choice` and `parallel_tool_calls`; its tool list is ignored. Without one, no
  tool choice is sent, so the provider's `auto` applies and a plain-text answer is
  possible.
- Tool results have no size limit. `read_file` returns the whole file, `list_files` walks
  the whole tree, and command and MCP output are returned in full.
- `write_file` is not atomic. `apply_diff` writes the blocks that applied even when
  another block didn't, and reports success ("Applied N of M"); only a diff with no
  block applied is a tool error.
- End-user command tools expand `~` and `$VAR` in their command template, split it on
  whitespace (no shell, no quoting), run with a null stdin, and are killed when stopped.
  Their result is stdout; a non-zero exit is a tool error with the exit code, stdout and
  stderr. The runtime never expands tool parameters: an `is_path` value `~/notes` names a
  directory called `~`.

`InterceptionFailurePolicy` (for `InterceptorConfig`, given to `run_task`) defaults to
`UseRawResult`; the others are `RetryThenUseRawResult { max_attempts }`, `InformAgent`
and `RetryThenInformAgent { max_attempts }`. Error results, and results of
`complete_task`, `ask_user`, `list_files`, the terminal tools, `read_memory` and any tool
whose `allow_result_interception` is `false`, are never intercepted; `load_skill`
results only with `SkillsConfig::load_skill_interception`. See
[Result interception](../arko_agents/README.md#result-interception-and-execution-middleware).

### Environment details and reminders

`EnvironmentDetailsConfig` has `current_working_directory`, `file_tree`,
`current_datetime` and `operating_system`, each a `BuiltInEnvironmentDetail`, plus
`file_tree_mode`. All default to not included.

| Setting | Default | Notes |
|---|---|---|
| `BuiltInEnvironmentDetail::include` | `false` | Required in JSON when the detail is given. |
| `placement` | `SystemPrompt` | `InitialTaskPrompt`, `EachUserMessage`. The system prompt is written once, when the task starts, so a date there is the task's start time. `EachUserMessage` details go with every user message, tool results included (in native mode as text after the results). |
| `depth` | `None`: 3 | File tree only. |
| `file_tree_mode` | `Diff`: a full snapshot, then a message listing changes before a model request | JSON `"diff"`, `"full"`. `placement` is ignored in `Diff` mode; `Full` puts the whole tree at `placement`. |
| `UserReminder::frequency` | `InitialTaskPromptOnly` | `EveryUserMessage`; `EveryNUserMessages(n)`: the first message and every nth after it; `n = 0` never. Messages the runtime adds count. |

Developer-defined details are passed to `Task::new`; a dynamic provider isn't saved with
the task. A task that would show a detail without content fails at start with
`ArkoAgentsError::Configuration`; a dynamic provider that returns an error leaves the
detail out for that message.

### Model-call recovery

| `LLMErrorPolicy` | Attempts | When recovery ends |
|---|---|---|
| `RetryThenPauseAndAwaitApproval { max_attempts }` (**default**, 10) | `max_attempts` (0 counts as 1) | Pauses in `AwaitingErrorRecoveryApproval` |
| `RetryThenInformAgent { max_attempts }` | `max_attempts` | Returns the task `Interrupted` |
| `InformAgentAndContinue` | Unlimited within the elapsed budget; unusable responses 10 | Returns the task `Interrupted` |
| `PauseAndAwaitApproval` | 1 | Pauses at the first failure |

| `LLMRetryConfig` field | Default | Notes |
|---|---|---|
| `initial_delay_ms` | 2,000 | Must be above 0. |
| `max_delay_ms` | 64,000 | At least `initial_delay_ms`. A provider's delay may exceed it. |
| `max_elapsed_ms` | 300,000 | Budget from the first failure of an episode. It bounds the waits and the decision to retry; it never cuts off a request in flight, and it stops counting while an accepted response streams. |
| `jitter` | `true` | Each delay between half and all of its step. |
| `rate_limit_retry` | `Some(LLMRateLimitRetryConfig::default())` | 1 s to 60 s backoff, `max_wait_ms` 900,000. Under the `RetryThen*` policies, rate limits and (without `capacity_retry`) capacity errors are waited out outside the attempt limit and budget. A rate limit that asks for more than `max_wait_ms`, or comes after `max_wait_ms` of rate-limit waits in one recovery episode, is handled as a usage limit; a capacity error that asks for longer gets ordinary recovery. Under the other policies, and with JSON `null`, both get ordinary recovery. |
| `capacity_retry` | `None` | Opt-in `LLMCapacityRetryConfig` (30 s to 300 s backoff): under the `RetryThen*` policies, capacity errors are retried until the service recovers. |
| `usage_limit_policy` | `Pause` | A plan usage limit ends recovery under every policy. Opt-in `WaitUntilReset { max_wait_ms }`, under every policy, waits until the reported reset plus 1 s (and up to 30 s of jitter while `jitter` is on), if the provider reports a reset and the reset plus 31 s fits in `max_wait_ms`; otherwise recovery ends as under `Pause`. |
| `stream_idle_timeout_ms` | `None` | Opt-in idle limit on this agent's streamed responses (arko_ai's `with_stream_idle_timeout`); make it generous. |

A failed request (connection failure, HTTP 408, 409, 429 or 5xx, truncated stream) is
resent unchanged, with nothing added to the conversation. A response the runtime can't
use is retried at once with a note to the model. Non-retryable errors (authentication,
invalid request, refusal) end recovery at once. Before each OpenAI request the runtime
refreshes ChatGPT OAuth credentials that expire within 5 minutes. See
[Model-call recovery](../arko_agents/README.md#model-call-recovery).

A turn that ends at the output-token limit emits `OutputTokenLimitReached` and is
handled as usual: nothing continues it automatically, and a tool call cut off mid-way is
dropped.

### Monitoring

ChatGPT OAuth calls report plan usage (`PlanUsageReported`) whenever the backend sends
it, and a call served by another model than requested emits `ModelRerouted`.
`AgentConfig::monitoring` adds opt-in monitors; both default to `None`:

| Setting | Default | Notes |
|---|---|---|
| `plan_usage: Some(PlanUsageThreshold { percent, action })` | `action: Pause` | Checked before each request. `Pause` pauses once per window and reset; `SlowDown { delay_ms }` waits before every request while usage stays at or above `percent`. |
| `cache: Some(CacheMonitorConfig::default())` | `warmup_calls` 2, `window_calls` 5, `warn_below` `Some(0.5)`, `pause_below` `None` | `CacheEfficiencyReported` after each judged call (not the warm-up calls); `CacheEfficiencyLow` once per drop of the rolling efficiency below `warn_below`, after `window_calls` judged calls; `pause_below` also pauses. |

A monitor pause waits in `AwaitingErrorRecoveryApproval` under the pausing error
policies and returns the task `Interrupted` under the others.

### Refusals

A refusal is never retried unchanged. It is an error that can't be retried: the task
pauses or is returned `Interrupted`, as the error policy decides. A model's own refusal
stays in history, without its tool calls; classifier and content-filter refusals are
removed. `UserCommand::ResolveRefusal` (while the task awaits recovery approval) or
`Task::resolve_refusal` (on a returned task) changes the refused input. The opt-in
`refusal_fallback` sends the same input once to another model of the same provider; the
refused turn is then not kept. See [Refusals](../arko_agents/README.md#refusals).

### Runtime

`AgentRuntime::new(client).run_task(task, registry, event_throttle_ms,
simplified_tool_events, skill_prompt_context, interceptor_config,
execution_middleware)`:

| Parameter | Unset means | Notes |
|---|---|---|
| `event_throttle_ms` | `None` or 0: every text and reasoning chunk is its own event | `Some(ms)` coalesces chunks. |
| `simplified_tool_events` | Required `bool` | Output-based mode only: `true` leaves out `ToolParamName`, `ToolParamValueChunk` and `ToolParamEnd`. |
| `skill_prompt_context` | No skills in the prompt | From `SkillManager::setup`. |
| `interceptor_config` | No result interception | `InterceptorConfig { interceptor, failure_policy }`. |
| `execution_middleware` | None | `before`/`after` every tool execution; for output supplied with an approval, only `after`. |
| `AgentRuntime::with_turn_context(source)` | No turn context | Asked for context before each model request. |

The event channel holds 100 events and the command channel 10: a host that stops
reading events blocks the task, and a host that drops the event receiver ends the run
with `ArkoAgentsError::ChannelSend`, losing the task value. Dropping the command sender
returns the task `Interrupted` at its next wait or boundary. `CancelTask` ends the task
`Cancelled`; `SuspendTask` returns it as it is, to be saved and run again.
`InjectContext` is added at the next turn boundary as a `# Workflow context update`
message. `ApproveErrorRecovery` adds a note that the user approved continuing (not
after a monitor pause or for a pending refusal fallback); `ContinueAfterError { note:
None }` resends the failed request unchanged, or after a tool-error pause sends the
results that were waiting; `RejectErrorRecovery` fails the task. In
`SubmitBatchToolApprovals`, a call without a decision is rejected. A command the
current state doesn't take is ignored. See
[Running tasks](../arko_agents/README.md#running-tasks).

### Skills, MCP and terminals

| Setting | Default | Notes |
|---|---|---|
| `SkillsConfig::workspace_source` | `None` | A `SkillSource { path, include, exclude }` relative to the workspace root. |
| `SkillsConfig::global_sources` | Empty | Paths are used as written; `~` is not expanded. |
| `SkillsConfig::on_invalid_skill` | `Warn`: the skill is skipped | `Error` fails `SkillManager::load`. |
| `SkillsConfig::load_skill_interception` | `None`: `load_skill` results are never intercepted or distilled | `Some(policy)` opts in. |
| `McpServerConfig` | `args` and `env` empty | The server inherits the host environment plus `env`. |
| `TerminalConfig::new(dir)` | Shell `/bin/sh -c` (Windows `cmd.exe /D /S /C`); `output_buffer_bytes` 65,536; `max_log_bytes` 4 MiB; `excerpt_bytes` 2,000; `prompt_quiet_ms` 150; `default_timeout` `None`; `prompt_rules` yes/no, press-key and password prompts; `quiet_terminal_environment` `true` | The directory holds job records and logs and is locked by one manager. PTY jobs get `TERM=dumb`, `NO_COLOR=1`, `PAGER=cat`, `GIT_PAGER=cat` and a UTF-8 locale. There is no limit on the number of jobs. |

Skills are loaded only when the host calls `SkillManager::load` and then `setup` on the
configuration the task will run with (before `Task::new`, which copies it, or on a loaded
task's `agent_config`); arko_workflows does this for its agents. When there are
on-demand skills, `setup` registers `load_skill`, adds it to the agent's tools and
auto-approves it unless the agent has a per-tool entry for it.

MCP tools are named `mcp__<server>__<tool>`, return text only (images are dropped; with
no text, the structured content as JSON), and skip the workspace path check.
`McpManager::register_tools` doesn't add them to `available_tool_names` (use
`tool_names()`), and only the host tool timeout bounds their calls. A server's stderr
goes to the log at info level; a server that exits logs one warning. Terminal tools are
registered with `TerminalManager::register_tools` and must be listed in
`available_tool_names` too. See [MCP servers](../arko_agents/README.md#mcp-servers),
[Skills](../arko_agents/README.md#skills), and
[interactive terminals](interactive-terminals.md#configuration).

### Persisted data

- Tasks and configurations are saved as pretty JSON through a temporary file and a
  rename. Tasks carry no format version; their optional fields default, so tasks saved
  by earlier versions load.
- A task's configuration snapshot without a tool calling mode loads as `OutputBased`,
  the mode it ran with; a standalone configuration without one loads as `Native`.
- A running task uses the configuration and system prompt saved in it; editing the
  agent's configuration doesn't change a saved task.
- `Failed { reason: "Task cancelled by user." }` loads as `Cancelled`; a pause saved
  without a cause loads with `ErrorRecoveryCause::Unknown`.
- `allow_batched_tool_calls` is ignored (`false` logs a warning once), a per-tool
  `allow_batched_calls: false` loads as `parallel_safe: Some(false)`, and a tool
  definition's `supports_batched_calls` is ignored.
- Dynamic environment-detail providers aren't saved: set them again on a loaded task.
- Terminal job records left `running` by a stopped host load as `interrupted`; processes
  are never reattached.

## arko_workflows

Guides: [arko_workflows README](../arko_workflows/README.md) and the
[API index](../arko_workflows/INDEX.md).

### `WorkflowBuilder`

`build()` requires a name, an existing workspace directory (stored absolute, `.` and
`..` resolved, symbolic links kept), an `Arc<LLMClient>`, and at least one uniquely
named agent. The builder reads no credentials. `build()` creates the persistence
directory and its `workflow-run-id` file when persistence is set.

| Setting | Default | Notes |
|---|---|---|
| `approval_policy` | **`ApproveAll`: every call an agent asks approval for is approved** | Includes paths outside the workspace under `RequireApproval`. Logs one warning per agent. See [Approvals and error recovery](#approvals-and-error-recovery). |
| `error_recovery_policy` | **`Automatic`: recovery without an attempt limit** | See [Approvals and error recovery](#approvals-and-error-recovery). |
| `metadata` | Empty | Only `supported_platforms` is enforced: a nonempty list the host matches none of fails `build`. |
| `memory` | No tiers, no memory tools | See [Memory](#memory). |
| `budget`, `budget_warning` | None: unlimited, no warning | Dollars, finite and nonnegative. Spending counts `UsageInfo::billed()`, so ChatGPT OAuth calls add zero. |
| `agent_budget(agent, limit)` | None | The agent's own model spend across its runs. |
| `hook(point, callback)` | None | Synchronous; run in registration order, and the first result other than `Continue` ends the point. `Abort` stops the run; `Skip` is honoured only at `PreToolExecution`. |
| `persistence` | None: nothing is saved, durable steps unavailable | See [Persistence](#persistence). |
| `resume_from(checkpoint_file)` | None | Needs `persistence` on the same run directory; see [Resume](#resume). |
| `rename_memory_tier_on_resume`, `drop_memory_tier_on_resume` | None | See [Resume](#resume). |
| `ephemeral()` | — | Clears the persistence and resume settings made so far. |
| `context_freshness` | Off | See [Context freshness](#context-freshness). |
| `global_tool`, `agent_tool` | None | Added to every agent's (or one agent's) available tools. A custom tool named like an arko_agents built-in replaces it, except `complete_task`. A global tool can't take `complete_task`, `ask_user` or a workflow tool's name (`build` fails). |
| `tool_result_interceptor`, `agent_tool_result_interceptor` | None | Global interceptors, then the agent's, then distillers. |
| `interception_failure_policy` | `UseRawResult` | Applied to each stage of the chain. |
| `completion_verifier`, `distiller` | None | See [Verification and distillation](#verification-and-distillation). |
| `workflow_services` | None: no `run_workflow_agent` | See [Messaging, services and voting](#messaging-services-and-voting). |
| `messaging` | `MessagingPolicy::all()`: every agent may message every other | `disable_messaging()` offers no messaging tools. |
| `wait_for_message_timeout`, `agent_wait_for_message_timeout` | `WaitForMessageTimeout::Auto` | Just inside the agent's prompt-cache lifetime: 4 min 30 s, or 54 min with Anthropic's 1-hour TTL. A fixed `Duration` must be above zero and at most 24 hours. |
| `user_input_handler` | None: an `ask_user` question suspends the agent and the call returns `UserInputRequired` | |
| `enable_voting(voters)` | Off | Offers `cast_vote` to the named agents only. |
| `max_parallel_agents(n)` | No cap | For `run_agents_parallel`; `0` is rejected. |
| `fail_fast(bool)` | `false` | `true`: in `run_agents_parallel` the first failure stops the others. A pause is not a failure. |
| `cancel_grace_period` | 30 s (`DEFAULT_CANCEL_GRACE_PERIOD`) | How long a stopped run waits for each running agent before aborting it; nested runs get half per level. |

Agents are named by their registration name (the one given to `agent(name, config)`),
not `AgentConfig::name`, in every policy, list and hook. Every agent-name list is
validated at build; tool names the freshness mapping or a distiller names are only
warned about. See
[Builder signatures and defaults](../arko_workflows/INDEX.md#builder-signatures-and-defaults).

### Approvals and error recovery

Workflows run without human input by default. The managed paths (`run_agent`,
`continue_agent`, `resume_agent`, `run_agents_parallel`, `AgentHandle::wait`) answer
approvals and error-recovery pauses under these policies; `AgentHandle::into_parts`
hands both to the caller.

| `ApprovalPolicy` | Behaviour |
|---|---|
| `ApproveAll` (default) | Approves every call the agent's configuration marks for approval. |
| `RespectAgentConfig` | Pauses the agent: the call returns `WorkflowError::AgentPaused` with `AgentPause::ToolApproval`; continue with `resume_agent`. A call without a decision is rejected. |
| `Handler(Arc<dyn ToolApprovalHandler>)` | Decides each call: approve, reject with a reason, or pause. |

| `ErrorRecoveryPolicy` | Behaviour |
|---|---|
| `Automatic` (default) | After the agent's own retries run out: outages wait 30 s, doubling to 15 min while pauses repeat, then resend unchanged; plan usage limits wait until the reset plus 5 s; tool errors continue at once. Non-retryable errors, refusals and unusable responses fail the task with a reason. Monitor pauses and failed verifier reviews return to the caller. Nothing is added to the conversation. |
| `Limited { max }` | As `Automatic`, for at most `max` recoveries per run call; after that, a pause it would recover from fails the task. `0` recovers nothing. |
| `ReturnToCaller` | Every error-recovery pause returns `AgentPaused` with `AgentPause::ErrorRecovery` (`VerifierFailed` for a completion its verifier couldn't review). |

An agent whose `llm_error_policy` doesn't pause (`RetryThenInformAgent`,
`InformAgentAndContinue`) is interrupted instead, and the call returns
`AgentInterrupted`. An agent running inside another agent's tool call (a verifier or a
delegated run) can't pause: under `RespectAgentConfig` its `complete_task` is approved,
and anything else that would pause it stops it (a completion its verifier couldn't
review is left paused). `UserInputRequired`, `AgentPaused` and `VerifierFailed` are
pauses, not failures: `OnUserInputRequired` fires for the first and `OnAgentPaused` for
the others; `OnWorkflowFailed` fires for none. See
[Approvals and error recovery](../arko_workflows/INDEX.md#approvals-and-error-recovery).

### Memory

| Setting | Default | Notes |
|---|---|---|
| `MemoryConfig::global_tier(name)` | Readers and writers `All`, `AppendOnly`, no size limit | `global_tier_with_limit`, `broadcast_tier(name, writers)` (writers `Only`), `broadcast_tier_with_limit`. Other readers, writers or `FullEdit` need `tier(MemoryTier { .. })`. |
| `MemoryTier::max_bytes` | `None`: unlimited | Limits the content (not the history); a write over it fails. |
| `MemoryTier::write_mode` | `AppendOnly` | `FullEdit` replaces the content; history keeps each version. |
| `AccessScope` | — | `All`, `Only(names)`, `None`. |

An agent gets `read_memory` if it can read a tier, `write_memory` if it can write one,
and `list_memories` if either; `list_memories` hides tiers it can't access.
`read_memory` results are never intercepted or distilled. See
[Memory](../arko_workflows/INDEX.md#memory).

### Messaging, services and voting

| Setting | Default | Notes |
|---|---|---|
| `MessagingPolicy` | Every agent may message every other | `can_message(agent, recipients)` limits one agent; `cannot_message(agent)`. `send_message` and `list_recipients` go to agents with a permitted recipient, `wait_for_message` to agents another may message. `send_message` reaches only running or waiting agents. |
| `WorkflowServicesConfig::allowed_callers`, `allowed_targets` | Empty: **nobody** | Name them with `allow_caller` and `allow_target`; an empty caller list is only warned about. |
| `allow_all_callers()`, `allow_all_targets()` | Off | Every agent except verifier agents (and, as a target, the caller itself). |
| `max_active_calls` | 4 | Delegated calls running at once, workflow-wide; a call over the limit fails at once. A `NonZeroU32`: `max_active_calls(0)` fails the build. |
| `BallotConfig` | No default | At least 2 options with nonempty, unique IDs, `quorum` at least 1, `consensus_threshold` from 0 to 1, weights finite and positive. `eligible_voters` empty = every enabled voter; a voter without a weight weighs 1.0; `tie_break` is `RejectTie` or `Lexicographic` (smallest option ID). |

### Verification and distillation

`CompletionVerifierConfig::correctness(name, verifier_agent)` and `efficiency(..)` set a
default `instructions` text; in JSON, `name`, `verifier_agent`, `kind` and
`instructions` are required.

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

A review that fails doesn't fail the target: the agent is asked to call `complete_task`
again, up to `max_review_retries` times, then the call returns
`WorkflowError::VerifierFailed` with the task paused. A correctness verifier fails the
task at the rejection after its cap; an efficiency verifier never fails a task and stops
reviewing once its minor cap is reached. `repair_context` needs `reuse_conversation` and
context freshness.

In JSON, every `DistillationConfig` field except `max_output_tokens` and `reasoning` is
required.

| `DistillationConfig::new(name, model)` | Default | Notes |
|---|---|---|
| `target_agents`, `target_tools` | Empty: `build` fails | Both required; `read_file` is not exempt. |
| `min_chars` | 8,000 | Results shorter than this are left alone. |
| `max_output_tokens` | `None`: at most 16,000 (nothing over ChatGPT OAuth) | `Some(0)` is rejected. |
| `reasoning` | `None`: the model's default | Must be for the model's provider. |

Distillation requests turn prompt caching off (except over ChatGPT OAuth), count toward
the workflow budget but not the agent budget, and keep the raw output when the summary
is refused or cut off. See
[Tool-result distillation](../arko_workflows/INDEX.md#tool-result-distillation).

### Context freshness

`ContextFreshnessConfig::default()`: `block_on_conflict: true`, and `read_file`,
`list_files` (a directory-tree read), `write_file` and `apply_diff` mapped to their path
parameters; `tool_effect(name, effect)` maps other tools. When loaded from JSON, both
fields are required. A write is blocked when the writer read that file, or listed a
directory containing it, and another agent run (another run of the same agent included)
has since written it (for a listing, any file under the directory). With
`block_on_conflict: false` the write is allowed. In both modes, active readers whose
view a write made stale get a notice, and `OnFreshnessConflict` fires for each stale
reader. See
[Context freshness](../arko_workflows/INDEX.md#context-freshness).

### Durable steps

`DurableStepSpec::new(key)`: version `"v1"`, no instance, retry policy `Never` (in JSON
`version` and `retry_policy` are required). A step whose closure returns an error is
recorded in doubt, and under `Never` it stays in doubt until resolved.
`WhenDefinitelyNotCompleted { max_attempts }` runs it again only after a resolver
(`durable_step_with_resolver`) reports it definitely not completed; `RetrySafe {
max_attempts }` runs it again at the next call for the step, without a resolver;
`authorize_in_doubt_retry` allows one more run (under `Never` always, otherwise within
`max_attempts`). `max_attempts` 0 counts as 1. Durable steps need persistence and replay
from the run directory's `durable/journal.jsonl`. See
[Durable steps](../arko_workflows/INDEX.md#durable-steps).

### Persistence

| `PersistenceConfig::new(dir)` | Default | Notes |
|---|---|---|
| `auto_checkpoint` | `true` | A checkpoint after every finalized agent run and when a run is stopped. Task versions are saved either way. |
| `retain_checkpoints` | `CheckpointRetention::KeepAll`: every checkpoint and task version is kept | `keep_last_checkpoints(n)` (`KeepLast`) keeps the newest `n` and the task versions they refer to; `n` of 0 fails the build. In Python, `retain_checkpoints=n` (0 raises `ValueError`). |
| `event_log` | `EventLog::Full`: every event, including each streamed fragment | `LifecycleOnly` leaves the fragments out. The event log is never truncated or rotated. |
| `event_flush_interval` | 1 s | State changes, run ends and errors are written at once. |

`PersistenceConfig::user_data(name)` picks a new run directory,
`<user data>/arko/workflows/<name>/<UUID>`, on each call. A configuration pointing at an
existing run directory reuses its `workflow-run-id` and durable journal, also without
`resume_from`. Checkpoints and task versions are pretty JSON written through a temporary
file, synced and renamed; `latest-checkpoint` names the newest checkpoint; the event log
and the durable journal are appended JSON lines. All are written with default
permissions; task versions hold full conversations, and the event log the agents' output
and tool results. A save that fails never fails an agent's run:
it is logged and retried by the next checkpoint.
`checkpoint()` and `finish()` save on request. See
[Persistence](../arko_workflows/INDEX.md#persistence-and-checkpoint-state).

### Resume

`WorkflowBuilder::resume_from(path)` takes a checkpoint file, not a run directory.

- The new builder supplies agents, client, tools, hooks, policies, budget limits and
  metadata. Memory access, limits and write mode come from the builder; the checkpoint
  supplies each tier's content and history by name.
- A tier configured but missing from the checkpoint starts as configured. A checkpoint
  tier with no configured tier fails `build`, unless renamed with
  `rename_memory_tier_on_resume` or dropped with `drop_memory_tier_on_resume`.
- Restored: memory, pending messages, the cost ledger, finalized agent results (with
  their tasks), verification and distillation records, agent statuses, and freshness
  state when context freshness is configured again. Not restored: a recorded stop (the
  run starts again), ballots, verifier repair counts, the pause flag, and running
  agents, which count as finished.
- The checkpoint's workflow name and run ID must match. `continue_agent` continues a
  `Completed` or `WaitingForInput` result; `resume_agent` answers a `Paused` one.

### Python bindings

Behind the `python` feature. The defaults match Rust (auto-checkpoint on, approve all,
automatic recovery, no parallel cap, fail-fast off, 30 s grace). Differences: the client
always comes from `LLMClient::from_env()`, agents load from configuration files, memory
offers only global tiers, and messaging is always all-to-all. Not exposed: hooks, custom
tools, interceptors and `interception_failure_policy`, verifiers, distillers, services,
freshness, voting, durable steps, user-input handlers, the `Handler` approval policy,
`budget_warning`, `agent_budget`, `metadata`, the messaging policy,
`event_flush_interval` and `PersistenceConfig::user_data`. See
[Python bindings](../arko_workflows/INDEX.md#python-bindings).

### Persisted data

- Checkpoints have no format version. Their optional fields default and unknown fields
  are ignored, so older checkpoints load, including those that hold their tasks inline;
  the next checkpoint writes those tasks as task files.
- A checkpoint without a run ID skips the run-ID check. One without a cost ledger
  resumes with nothing spent (`budget_spent` is not read).
- Task files from earlier versions beside the checkpoints (`agents/<task id>.json`) are
  neither read nor deleted.
- `Failed` tasks with the reason `Task cancelled by user.` load as `Cancelled`.
- `resume_from` refuses a missing or unreadable task file, another workflow's
  checkpoint, a run-ID mismatch, and a checkpoint-only memory tier.

## arko_index

Guides: [arko_index README](../arko_index/README.md) and the
[arko_index reference](arko_index-reference.md).

### `Config`

Given to `ArkoIndex::open`; one `ArkoIndex` at a time can hold a database. Omitted fields
take their defaults. Counts and limits of zero, `Some(0)` included, are rejected, except
`vector_cache_bytes` (0 caches nothing) and `max_capacity_wait_ms` (`Some(0)` fails a
request at its first capacity signal). A resumed job takes every field except
`embedding` from the current configuration.

| Field | Default | Notes |
|---|---|---|
| `batch_count` | 6 | Most files per initial chunk request (a split file's windows count separately), within the request budget. |
| `concurrency` | 2 | Chunk-model requests in flight, shared by all jobs. |
| `embedding_concurrency` | 2 | Embedding requests in flight, search queries included. |
| `embedding` | `voyage-4`, 1,024 dimensions, a pinned tokenizer revision of `voyageai/voyage-4` | A Voyage model and a dimension it supports; must equal the backend's profile. Set `tokenizer_revision` together with `model`. An update with another profile doesn't reuse the earlier generation's boundaries or vectors. A generation embedded with another profile fails semantic and hybrid search; search it lexically. |
| `vector_cache_bytes` | 64 MiB | 0 caches nothing. |
| `max_transient_attempts` | 10 | Per request; capacity signals don't count. |
| `max_capacity_wait_ms` | `Some(900_000)` (15 min) | `None` waits as long as the provider is overloaded. Search query embeddings wait too. |
| `query_priority` | `false` | Opt-in: query embeddings go before queued build embeddings. |
| `query_max_transient_attempts` | `None`: `max_transient_attempts` | |
| `embedding_batch_tokens`, `embedding_batch_items` | `None`: the model's limits | Can only lower them; `embedding_batch_tokens` not below the per-input limit. |
| `keep_generations` | `None`: every generation is kept until the index is deleted | `Some(n)` prunes older complete generations after each publish. |
| `synchronous` | `Full` | `Normal` syncs less often; a power loss can lose recent commits. |
| `stream_idle_timeout_ms` | `None` | Opt-in idle limit on chunk-model responses; make it generous. |

### `BuildSpec` and chunking

`BuildSpec::new(name, root)`. The spec is saved with the index and can't be changed:
updates, the watcher and search use the saved spec. The library has no default name.

| Field | Default | Notes |
|---|---|---|
| `include`, `exclude` | Empty | Globs on the root-relative path, `*` matching `/`. With `include`, only matching files are considered instead of the built-in list of source and documentation extensions and file names; `exclude` and the other filters still apply. |
| `skip_directories` | `.git`, `target`, `node_modules`, `.runtime`, `.cache`, `__pycache__`, `.venv`, `dist` | Exact names, at any depth. An empty list skips none, `.git` included. |
| `max_file_bytes` | 16 MiB | Larger files are skipped unread. |
| `chunking.max_request_tokens` | `None`: 10% of the chunk model's input limit on the client's auth path | Counted with `o200k_base` for OpenAI chunk models: 92,200 tokens for GPT-5.6 and GPT-6 with an API key, 27,200 over ChatGPT OAuth. For models measured in bytes, a set value applies as 4 bytes per token unless `max_request_bytes` is set. |
| `chunking.max_request_bytes` | `None`: no byte limit for OpenAI models; for the others the default share | Models of other providers, OpenRouter included, are measured in bytes: 4 bytes per token of the 10% share. A model whose input limit is unknown is assumed to take 128,000 tokens (12,800 tokens or 51,200 bytes), with a warning. For OpenAI models both limits apply when both are set; for the others `max_request_bytes` replaces `max_request_tokens`. |
| `chunking.oversize` | `Split` | A file over the budget is split at headings, top-level items, blank lines, then lines. `Skip` leaves it out; `Block` blocks the job before any request. |
| `chunking.max_repair_rounds` | `Some(5)` | `None`: repair until every cut is valid. |
| `chunking.repair_limit` | `Merge`: invalid cuts are dropped | `Block` leaves the job blocked. |

The chunk budget is resolved when a job scans and kept when it resumes, even if the auth
path changed. Hidden files are indexed; `.gitignore`, `.ignore` and git's exclude files
apply even outside a repository, including those in the root's parent directories;
symbolic links, files that aren't UTF-8, and files with NUL bytes are excluded.

### Backend, search, judge and watcher

| Setting | Default | Notes |
|---|---|---|
| `ArkoBackend::new(llm, embeddings, profile)` | Chunk model GPT-5.6 Luna, `high` effort with summaries, `low` verbosity | `with_chunk_model(model, config)`; changing the chunk model keeps earlier boundaries. The backend replaces the config's tools with its boundary tool, forced where the model allows it. The client must hold credentials for the model's provider. |
| Prompt-cache key of chunk requests | The job's ID | The same for initial requests and repairs, across resumes; over ChatGPT OAuth it is also the `session-id`. |
| `SearchOptions::mode` | `Hybrid` | `Lexical`, `Semantic`. |
| `top_k` | `Some(10)` | `None` returns every match. |
| `min_chunk_bytes` | 40 | Shorter chunks are left out; 0 keeps them. |
| `path_prefix`, `literal` | `None` | A plain prefix; a case-sensitive substring. |
| `rebuild_file_limit` | `Some(100)` | Files listed in a rebuild notice; `None` lists all. |
| `JudgeOptions` | `candidate_limit` 20, `max_attempts` 2, `max_transient_attempts` 3 | The judge is opt-in (`search_with_judge`). It takes no cursor and a `top_k` of at most `candidate_limit`; a provider delay over 60 s fails the search. |
| `LlmSearchJudge::new(client, model)` | `GenerationConfig::default()`, no instructions, no idle timeout | No default model. |
| `WatchOptions::debounce` | 500 ms | |
| `WatchOptions::reconcile` | `Some(30 s)`: a full rescan every 30 s | `None`: rescans only at start and after file events. |
| `WatchOptions::resume_interrupted` | `true` | Resumes jobs left paused by an interrupted process. |
| `WatchOptions::retry_blocked_after` | `None` | Opt-in: resumes a job blocked only by transient failures after this delay. |

Every search hit carries `may_be_stale`, true when its file changed on disk or is in the
newest unfinished job's changes.

### Jobs

- A job left running by a stopped process is `paused` when the database is next opened;
  only a watcher resumes it automatically.
- A plan usage limit pauses the job with the reset time; nothing resumes it
  automatically.
- A paused or blocked job keeps new updates of its index from starting; a failed or
  cancelled one doesn't.
- Transient failures back off from 1 s to 60 s with jitter, at least the provider's
  delay; after `max_transient_attempts` the job is blocked.
- `resume` requires the same embedding profile and the same library chunking policy
  (its prompts and policy version), and no newer published generation. Other settings
  come from the current `Config`; changes are logged and recorded as a
  `settings_changed` event.
- An update reuses earlier boundaries only from jobs with the same embedding profile and
  chunking policy.

See [Jobs, scheduling and retries](arko_index-reference.md#jobs-scheduling-and-retries).

### Persisted data

- The database schema is version 2. A version 1 database is migrated when opened; a
  database from before versioning, or from a newer version, is refused.
- A chunk's text is stored once per location and file content, shared by the
  generations and indexes of a database that hold it; a vector once per text and
  embedding profile.
  `delete` and pruning remove data nothing refers to; vectors are removed only while no
  job is unfinished.
- A saved `BuildSpec` without `skip_directories` loads with the default list, so an
  index that used to include `dist/` or another listed directory skips it at its next
  update; one without `max_file_bytes` loads with 16 MiB.
- A job saved without a chunk budget runs without one (batches by `batch_count` only).
- `JobStatus::active_chunking` also reads the old name `active_luna`.

## Across crates

- **No model-request timeouts.** No layer puts a deadline on a model request; a
  response is waited for as long as its connection stays open. The opt-in stream idle
  timeout exists at each layer: `LLMClient::with_stream_idle_timeout`,
  `LLMRetryConfig::stream_idle_timeout_ms`, `Config::stream_idle_timeout_ms` (arko_index)
  and `LlmSearchJudge::with_stream_idle_timeout`. Make it generous. The main other
  limits are the 30 s connect timeout, the 60 s OAuth token-refresh timeout, the
  20-minute host tool timeout of arko_agents (terminal tools, `run_workflow_agent` and
  `wait_for_message` exempt), `wait_for_message`'s own limit (4 min 30 s by default),
  the terminal tools' 30 s cap per wait, the judge's 60 s limit on a provider's retry
  delay (a longer one fails the search), and a workflow's 30 s cancel grace period after
  an explicit stop.
- **Retries by layer.** arko_ai doesn't retry failed requests (apart from the 401
  resend over ChatGPT OAuth). arko_agents makes up to 10 attempts within a 5-minute
  budget, waits out rate limits and capacity errors, and pauses at plan usage limits.
  arko_workflows, by default, resumes those pauses without limit. arko_index gives each
  chunk and embedding request up to 10 attempts through transient failures and waits up
  to 15 minutes for capacity; its judge makes up to 3 attempts.
- **Approvals.** A bare `AgentConfig` asks approval for every call except `ask_user`;
  a workflow approves them all by default. Path parameters (`is_path`) outside the
  workspace are rejected either way unless the path policy says otherwise; terminal, MCP
  and command tools are not confined.
- **Prompt caching** is on everywhere by default. Each conversation has its own cache
  key, so one conversation per agent keeps its cache; arko_index uses one key per job.
- **Credentials** are read from the environment only by `LLMClient::from_env` and
  `EmbeddingClient::from_env`. `WorkflowBuilder` and `ArkoIndex` read none; the Python
  `WorkflowBuilder` calls `LLMClient::from_env`.
- **Costs.** Budgets and task totals count `UsageInfo::billed()`: ChatGPT OAuth calls
  add zero, and their list-price estimate stays in `total_cost`.
- **Logging** goes through the `log` crate. arko_ai logs its warnings about dropped or
  ignored request settings once per process; a workflow's `ApproveAll` warning comes
  once per agent.
- **Saved files** (tasks, checkpoints, event logs, request traces) hold full
  conversations and are written with default file permissions.
- **Loading older data.** Unknown fields are ignored and new fields default, so data
  from earlier versions loads. The exceptions are listed under each crate's persisted
  data.

## Example binaries

The examples read credentials as listed; `llm_client_cli`, `embeddings_demo` and
`arko_agents_cli` also load a `.env` file.

**`arko_ai` `llm_client_cli`** (`text`, `chat`, `models`):

| Flag | Default | Notes |
|---|---|---|
| `-m`, `--model` | Required | A registered ID or alias; an ID with `/` is an OpenRouter slug. |
| `-s`, `--system`, `-t`, `--temperature`, `-n`, `--max-tokens` | Unset: not sent | |
| `-r`, `--reasoning` | Off | The model's default reasoning: no effort is sent (OpenAI without summaries); Claude models that need a manual budget get half of `--max-tokens` (or the output limit), 1,024 to 16,000. |
| `--reasoning-level`, `--reasoning-effort`, `--reasoning-mode` | Unset | An explicit effort, level or budget (OpenAI with summaries); a mode. |
| `--long-context` | Off | |
| `text -p`, `--prompt` | Required | `-i`, `--image PATH`; `--stream`. |
| `--codex-auth`, `--codex-auth-json PATH` | Off | OpenAI credentials from a Codex `auth.json`; the `OPENAI_OAUTH_*` variables are then not read. |

**`arko_ai` `embeddings_demo`**: `VOYAGE_API_KEY` (skipped without it); `voyage-4-lite`,
256 dimensions. **`oauth_cache_probe`**: `--out DIR` (required), `--auth-file` (default
Codex path), `--model gpt-6-astra`, `--effort xhigh`, `--conversations 3`, `--turns 15`
(2 with `--idle-gap SECS`), `--label baseline`; `--reasoning-context`, and `--legacy`
with its request-shape flags for comparisons; `--summarize-only` recomputes the summary
of an earlier run.

**`arko_agents` `arko_agents_cli`**:

| Flag | Default | Notes |
|---|---|---|
| `-c`, `--agent-config`, `-w`, `--workspace-dir`, `-p`, `--prompt` | Required | The workspace is created if missing. |
| `--tool-calling-mode` | The configuration's | |
| `--event-throttle-ms` | Unset: no coalescing | |
| `--simplified-tool-events <true\|false>` | `true` | |
| `--user-command-timeout SECONDS` | The configuration's (none) | |
| `--mcp-config PATH` | No MCP | Servers switch the agent to native mode and add their tools. |
| `--openai-auth-json PATH` | Environment credentials | |
| `--save-task-state PATH` | Not saved | |

The CLI asks at the terminal for every approval and error recovery.
`examples/coder_agent.json` uses Anthropic `claude-opus-5-5` and pauses at the first
model error and at tool errors (`PauseAndAwaitApproval` for both);
`examples/native-tools-agent.json` uses OpenAI `gpt-6-sol` and auto-approves every tool.
`interactive_terminal` takes no arguments (`ARKO_TEST_PYTHON` selects Python).

**`arko_workflows` `basic_workflow`**: no flags; `LLMClient::from_env()`, agent
configurations `configs/architect.json` and `configs/coder.json` (not shipped; supply
them) and an existing `./my-project` directory, all relative to the current directory;
budget $5 (warning at $4); the default `ApproveAll`.

**`arko_workflows` `deepcoder`**:

| Flag | Default | Notes |
|---|---|---|
| `--workspace`, `--spec` | Required | |
| `--state-dir` | `<workspace parent>/.arko-deepcoder-runs/<name>-<time>-<id>` | Must be outside the workspace. Persistence is on unless `--ephemeral`. |
| `--resume PATH` | New run | A checkpoint file, or a run directory; its directory is the state directory. Conflicts with `--ephemeral`. |
| `--budget` | 50.0 | Warning at 80%. |
| `--max-minor-corrections`, `--max-efficiency-corrections`, `--max-final-corrections` | 20, 3, 4 | The library's verifier caps are 3, 2 and 3. |
| `--shell-timeout-seconds`, `--shell-output-bytes` | 1800, 65,536 | |

`deepcoder` configures only OpenAI (the `OPENAI_OAUTH_*` variables, then the Codex
`auth.json`, then `OPENAI_API_KEY`), turns on token-refresh rate-limit retries (used
only by explicit refresh calls, which the agent runtime doesn't make), and auto-approves
every tool. Its agents use `LLMErrorPolicy::PauseAndAwaitApproval`, so a
failed model request goes straight to the workflow's automatic recovery. See the
[DeepCoder README](../arko_workflows/examples/deepcoder/README.md).

**`arko_index` `arko_index_cli`** (`build`, `update`, `resume`, `status`, `list`,
`search`, `read`, `watch`; global flags come before the command):

| Flag | Default | Notes |
|---|---|---|
| `--database` | Required | |
| `--oauth-file PATH` | `LLMClient::from_env()` | An OpenAI-only client from a Codex `auth.json`. |
| `--batch-count`, `--concurrency` | 6, 2 | |
| `--stream-idle-timeout-secs` | None | Also applied to the judge. |
| `build --name` | The root directory's name | |
| `search --lexical`, `--semantic` | Hybrid | |
| `search --top-k`, `--min-chunk-bytes`, `--rebuild-file-limit` | 10, 40, 100 | |
| `search --judge` | Off | `--judge-model` defaults to `gpt-5.6-luna` for OpenAI, `--judge-effort` to `high`; see [Judge options](arko_index-reference.md#judge-options). |
| `watch --debounce-ms`, `--reconcile-secs` | 500, 30 | `--reconcile-secs 0` turns periodic rescans off. |

Commands that embed need `VOYAGE_API_KEY`; `build`, `update`, `resume` and `watch`
also need OpenAI credentials for the chunk model (GPT-5.6 Luna; no flag changes it), and
so does `search --judge` by default. `list`, `status` and `read` need no credentials.
Settings without a flag keep the library defaults. Ctrl-C pauses a running `build`,
`update` or `resume` job. It stops `watch` and leaves an update it started unfinished;
the database lists it as paused when next opened, and the next `watch` resumes it. The
`retrieval_eval`, `retrieval_judge` and `retrieval_select` examples take positional
arguments; see their source.
