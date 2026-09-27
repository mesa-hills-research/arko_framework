# Arko AI

Rust clients for language models (OpenAI, Anthropic, Google Gemini and
[OpenRouter](#openrouter)) and embeddings, with provider configuration, streaming
responses, tool calls, reasoning controls, usage metadata, and OpenAI OAuth support.

## Use in a Rust project

```toml
[dependencies]
arko-ai = { path = "../arko-ai" }
```

Rust imports use `arko_ai`, as in `use arko_ai::LLMClient;`.

## Examples and checks

From the framework workspace:

```bash
cargo run -p arko-ai --example llm_client_cli -- --help
cargo test -p arko-ai --lib
```

The `embeddings_demo` example uses `VOYAGE_API_KEY` to request embeddings.

## Voyage text embeddings

`EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, model_id)` accepts the
current text embedding models below. Each supports 32,000 tokens per input,
up to 1,000 inputs per request, and 256, 512, 1024 (default), or 2048 output
dimensions. Token counting uses the matching `voyageai/<model_id>` tokenizer
from Hugging Face. The tokenizer dependency is updated to 0.23.2 so these
assets can be downloaded and loaded through the current Hub interface.

| Model ID | Total tokens per request | Standard USD per million tokens |
| --- | ---: | ---: |
| `voyage-4-large` | 120,000 | $0.12 |
| `voyage-4` | 320,000 | $0.06 |
| `voyage-4-lite` | 1,000,000 | $0.02 |
| `voyage-code-4` | Not published | $0.12 |

For `voyage-code-4`, `max_batch_total_tokens` is `None`: aggregate limits are
validated by the provider. This does not mean unlimited requests. The known
per-input and item-count limits still apply locally.

```rust
use arko_ai::{EmbeddingConfig, EmbeddingProviderIdentifier, VoyageInputType};

let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-code-4")
    .with_voyage_input_type(VoyageInputType::Document)
    .with_output_dimensionality(1024);
```

Use `VoyageInputType::Query` when embedding a search query. Truncation defaults
to false. The existing `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large`, and
`voyage-code-3` IDs remain supported. Cost estimates use standard endpoint rates
before account credits or discounts.

Model metadata checked 2026-09-13 against Voyage's [model guide](https://docs.voyageai.com/docs/embeddings),
[API reference](https://docs.voyageai.com/reference/embeddings-api),
[pricing](https://docs.voyageai.com/docs/pricing), and
[tokenizer documentation](https://docs.voyageai.com/docs/tokenization).

## Stop reasons and refusals

Every `GenerationResponse` carries a normalized `stop_reason` (`EndTurn`, `ToolUse`, `MaxTokens`, `StopSequence`, `Refusal`, `PauseTurn`, `ContextWindowExceeded`, or `Other` with the provider's value) and, when the model or a filter declined, a `refusal`. Streams emit `StreamEvent::Refusal` once, just before `End`, which carries the same data. `GenerationResponse::single_tool_call(name)` returns the one expected tool call or an error that says whether the response was refused, cut off by a limit, or called something else.

The provider's raw `stop_reason` and `stop_details` remain on `GenerationResponse.model_call_usage` (and in the agent runtime's `ModelCallUsageReported` events and task usage history). Missing fields in older saved records deserialize as `None`.

`Refusal.source` says how to treat the turn:

- `SafetyClassifier`: Anthropic's `stop_reason: "refusal"`, with `stop_details.category` and `explanation` when supplied (both can be null). Anthropic requires the refused turn to be removed before continuing, so the message is left empty and text generated before the refusal moves to `partial_output`. `Conversation::add_assistant_message_from_response` doesn't append such a turn.
- `Model`: OpenAI's structured `refusal` content. The refusal text is in `explanation`, never in the message text or streamed as `TextChunk`. Ordinary text mentioning refusal is not classified as a refusal.
- `ContentFilter`: OpenAI `incomplete_details.reason: "content_filter"`. Partial output moves to `partial_output`, as for classifier refusals.

A stream that ends before the provider reports completion yields `Error::StreamTruncated` (retryable) instead of an `End`. When a token limit cuts a response short, tool calls whose arguments finished are kept and a call that was still being written is dropped. Tool-call arguments that aren't valid JSON are kept in `ToolCall::invalid_arguments` (with `arguments` set to `null`) instead of being replaced by `{}`.

Google's Interactions API reports only an interaction `status`: `completed` maps to `EndTurn` (or `ToolUse` with function calls), `requires_action` to `ToolUse`, and `incomplete` to `MaxTokens`. It has no refusal or safety reason to report.

OpenRouter maps `finish_reason`, `native_finish_reason` and `message.refusal` onto the same sources, and reports refusals raised as errors as `Error::Refusal`; see [Responses and streams](#responses-and-streams).

References: [Anthropic streaming refusals](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals), [official Anthropic streaming event schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_delta_event.py), [OpenAI refusal and content-filter handling](https://developers.openai.com/api/docs/guides/structured-outputs), [OpenAI streaming refusal events](https://platform.openai.com/docs/api-reference/responses-streaming/response/refusal), [Gemini Interactions API](https://ai.google.dev/api/interactions-api-v1).

## OpenAI Responses: replay, prompt caching and ChatGPT OAuth

Every OpenAI request sends the conversation's `prompt_cache_key` (for every model; `prompt_cache_options` stays limited to GPT-5.6 and GPT-6 with API keys). GPT-5.6 and GPT-6 requests also send `reasoning.context: "all_turns"`, so reasoning from earlier turns is rendered back to the model. Without a configured effort, API-key requests send no `reasoning.effort` (the model's default applies), while ChatGPT OAuth requests send `medium`, as Codex always sends an effort.

**Output items and replay.** A response's assistant output is assembled from the `response.output_item.done` stream events, in order, as Codex does; the ChatGPT backend's `response.completed` usually carries no items, and items only it holds are merged in. The assistant message keeps every replayable item (`reasoning` with `encrypted_content`, `message` with its `phase`, finished `function_call`) verbatim, minus `status`, in `ReasoningArtifact::OpenAIOutputItems`. Later requests replay them unchanged and in order, for example `reasoning`, `message` (commentary), `function_call`, then the call's `function_call_output`; no empty assistant message is added for tool-only turns. A replayed `function_call` needs the message to still hold that call (by `call_id`), and calls the message holds without an item follow in the reconstructed shape. Messages without items (older saved conversations with `OpenAIEncryptedReasoning`, application-built messages, other providers) are reconstructed as before: reasoning, function calls, then the text.

**ChatGPT OAuth requests** (`chatgpt.com/backend-api/codex/responses`) always stream (`generate_text` collects the stream) and carry the conversation identity, like Codex: `session-id`, `thread-id` and `x-client-request-id`, all set to the conversation's `prompt_cache_key`. The backend derives prompt-cache affinity from `session-id`, so give each agent its own conversation. `OpenAI-Beta: responses=experimental` is no longer sent on either path. No `originator` or `User-Agent` is sent unless the application sets one with `LLMClient::with_openai_originator` / `with_openai_user_agent` (use your own application's name).

**Turn state.** The backend returns an `x-codex-turn-state` routing token. `GenerationResponse::openai_turn_state` carries it, and `Conversation::add_assistant_message_from_response` keeps the first token of each user turn in `Conversation::openai_turn_state` (serialized, so a resumed task keeps it). Every later OAuth request of that turn, tool-result follow-ups and retries, echoes it unchanged. A user message without tool results starts a new turn and drops the token; it is never sent in another turn or with API keys.

**Reroutes.** `ModelCallUsage::served_model` is the model the provider reports (`openai-model` / `x-openai-model` header, else the response's `model`, else `response.created`), and `rerouted` is true when any reported model differs from the requested one, ignoring case and snapshot date suffixes. `GenerationResponse::served_model()` reads the first; `GenerationResponse::rerouted()` and `ModelCallUsage::served_model_changed()` are true for an OpenAI reroute and for an OpenRouter call whose `requested_model` is set.

References: [OpenAI reasoning context](https://developers.openai.com/api/docs/guides/reasoning), [openai/codex](https://github.com/openai/codex) (`codex-rs/core/src/client.rs`, `codex-rs/codex-api/src/sse/responses.rs`).

## Usage and cost

`UsageInfo` token counts mean the same thing for every provider:

- `input_tokens` is the total input, including cache reads (`cached_tokens`) and cache writes (`cache_write_tokens`). Anthropic reports uncached input separately, so arko adds the cache counts back; for streams it uses the cumulative counts in the final `message_delta`. **Usage records saved before this change stored only uncached input for Anthropic.**
- `output_tokens` includes reasoning tokens; `reasoning_tokens` reports them separately when the provider does (OpenAI `output_tokens_details.reasoning_tokens`, Gemini thought tokens).
- `cache_write_1h_tokens` is the part of Anthropic cache writes made with a 1-hour TTL.

Costs are estimates at list prices from `model.rs`, computed for what the call actually used:

- **Service tier.** The tier the response reports wins over the one requested (OpenAI echoes `service_tier`, for example `default` when a Fast request was downgraded; Anthropic reports `speed: "fast"` and `service_tier`). `GenerationConfig::with_openai_service_tier` requests OpenAI Flex or Fast (sent as `priority`) on models that offer it, with API-key auth only. Tier prices are per model (`ModelPricing::service_tiers`); tiers without a published price (Anthropic priority, OpenAI scale) are estimated at standard rates.
- **Cache TTL, residency and dates.** Anthropic 1-hour cache writes use their own rate; `inference_geo: "us"` applies the 1.1x multiplier on Claude 4.6 and later; `ModelPricing::schedule` switches prices on a date (for example the end of promotional pricing). Tier and residency multipliers stack.
- **Missing prices.** A token class without its own price (cached input on Pro models, cache writes on models that don't charge extra for them) is billed at the input rate, including long-context rates.

`total_cost` is the estimate; `billed()` is what the account is charged, decided by `billing`: `Metered` (the estimate), `Subscription` (ChatGPT OAuth: zero, while the estimate stays visible), `NotCharged` (zero: an Anthropic classifier refusal before any output in a category Anthropic does not bill) or `ProviderReported` (`reported_cost` plus any bring-your-own-key `upstream_cost`, used by providers that report their actual charge, such as [OpenRouter](#usage-and-cost-1)). `UsageInfo` implements `AddAssign` and `Sum`; summing calls billed differently yields `BillingKind::Mixed` with the billed amounts added. Workflow budgets count `billed()`.

## Errors and retries

`Error::retry_class()` returns `Permanent`, `Transient` or `Capacity`; `is_retryable()` and `is_capacity_error()` are views of it. Provider error bodies are read generically: string codes (`error.code`, `error.type`, `error.status`), OpenRouter's `error.metadata.error_type`, and numeric HTTP-style codes, which stand in for the status on stream errors that arrive after a 200. Documented string codes stand in for it too (Gemini Interactions codes such as `service_unavailable` or `deadline_exceeded`, Anthropic error types such as `api_error` or `timeout_error`). Permanent codes (authentication, quota such as `insufficient_quota` or Gemini's daily `quota_exceeded`, invalid requests, context length, content policy, ...) always win; any other error reported after a 200 is `Transient`, so callers retry it a bounded number of times. Gemini generation-blocked codes (`safety`, `recitation`, `prohibited_content`, ...) become `Error::Refusal` with a `ContentFilter` source. Delay hints come from `Retry-After`, OpenAI's `x-ratelimit-reset-*` headers (on a 429 only, for the exhausted limit), or body fields (`retry_after_ms`, `retry_after`, `resets_at`) for every provider.

`arko_ai::retry::BackoffPolicy` is the shared backoff: exponential with a cap and equal jitter, treating a server hint as a minimum with jitter above it. arko-ai does not retry model calls itself; arko-agents, arko-index and the OpenAI OAuth refresh loop use this policy.

### Long requests and embedding tokenizers

The shared model HTTP client has no total request deadline. Connection setup and idle
connection-pool cleanup are separately bounded. Hosts can cancel requests explicitly.

`EmbeddingConfig::with_tokenizer_revision()` pins a Hugging Face tokenizer revision.
Tokenizers are cached per repository/revision with truncation and padding disabled;
concurrent first loads share initialization. Embedding tokenization runs on a blocking
worker. Voyage responses are reordered by provider item indexes and reject missing,
duplicate, dimension-invalid, nonfinite, or zero vectors. Provider failures preserve
HTTP status and Retry-After metadata for caller-controlled retries.

## OpenRouter

`Provider::OpenRouter` talks to OpenRouter's native Chat Completions API
(`POST https://openrouter.ai/api/v1/chat/completions`) with its routing, caching and
reasoning extensions. It does not use OpenRouter's Responses or Anthropic Messages
compatibility APIs.

```rust
use arko_ai::{GenerationConfig, LLMClient, Model, ReasoningConfig, ReasoningEffort};

let client = LLMClient::from_env(); // reads OPENROUTER_API_KEY among others
let mut conversation = client.new_conversation(Model::openrouter("anthropic/claude-sonnet-5"));
conversation.add_user_message("Hello");
let config = GenerationConfig::default()
    .with_max_tokens(4096)
    .with_reasoning(ReasoningConfig::openrouter_effort(ReasoningEffort::High));
let response = client.generate_text(&conversation, config).await?;
```

`LLMClient::with_openrouter_api_key` or `with_openrouter_client(OpenRouterClient::new(key)...)`
configure it explicitly; `OpenRouterClient::with_base_url` points it at a regional
hostname, proxy or test server.

### Models and the catalog

`Model::openrouter(slug)` accepts any slug (`anthropic/claude-sonnet-5`,
`openrouter/auto`, `~anthropic/claude-sonnet-latest`, `openai/gpt-6-luna:floor`,
`@preset/name`) without a network call. Such a model has unknown limits (0) and prices
(0: costs come from OpenRouter's reported `usage.cost`), and requests pass through
unchecked. OpenRouter models are not in the static registry (`Model::list` is empty for
the provider); saved agent configs keep a model snapshot, and a hand-written
`provider: OpenRouter` config loads a bare model.

The catalog is fetched only when asked: `OpenRouterClient::fetch_models()` (`GET /models`,
public, no key sent) or `fetch_user_models()` (`GET /models/user`, filtered by the key's
provider preferences, privacy settings such as ZDR, and guardrails). It is kept in memory
and shared by clones; `OpenRouterCatalog::from_json` loads a saved response instead.
`OpenRouterClient::model(slug)` then returns a `Model` with context length, the top
provider's output limit, image input, prices (cache reads, cache writes, 1-hour writes and,
as the long-context tier, the first `pricing.overrides` entry whose only condition is
`min_prompt_tokens`; entries with time windows or unknown conditions are skipped) and
`Model::openrouter_info()`: supported parameters, reasoning controls (supported and default
efforts, mandatory reasoning) and input/output modalities. Variant suffixes resolve to the
base model.

With a catalog loaded, or a model taken from it, requests are validated before sending:
unknown slugs fail (except `@preset/…`; `model@preset/…` is checked by its model), as do
parameters the catalog doesn't list for the model (`temperature`, `reasoning`, `tools`,
`tool_choice`, `verbosity`, `max_tokens`), which OpenRouter would otherwise drop silently,
images for text-only models, and `effort: none` or `enabled: false` on models with
mandatory reasoning.

### What is sent

Nothing is sent without a reason, so OpenRouter and the serving provider apply their own
defaults. In particular no attribution or metadata headers are sent by default.

| Field | Sent when | Otherwise |
| --- | --- | --- |
| `model`, `messages` | always (the system message first) | |
| `session_id` | always: `Conversation::prompt_cache_key` (at most 256 characters) | sticky routing keyed on the opening messages |
| `prompt_cache_key` | prompt caching is enabled | |
| `cache_control` (top level, `ttl` from `anthropic_cache_ttl`) | caching is enabled and the model is `anthropic/…` (Claude is cached only on request) | |
| `prompt_cache_options: {mode: "explicit"}` | caching is disabled on an OpenAI model the catalog shows billing cache writes (GPT-5.6 and later) | automatic caching |
| `max_tokens` | `GenerationConfig::max_tokens` is set | provider default (sending it also limits routing to endpoints that allow it) |
| `temperature` | set explicitly | provider default |
| `reasoning` | `ReasoningConfig::OpenRouter` is set | the model's default (`reasoning.default_enabled` / `default_effort` in the catalog) |
| `verbosity` | set; rejected together with `reasoning.effort` on Claude, where OpenRouter maps verbosity to `output_config.effort` | `medium` |
| `tools`, `tool_choice`, `parallel_tool_calls` | from `ToolConfig` (`strict` is passed through) | |
| assistant `tool_calls[].function.arguments` | always; arguments that weren't valid JSON are replayed verbatim to `openai/` models and as `{}` to others, which need an object (the tool result explains the error) | |
| `provider` | preferences are configured, or the Auto Exacto policy applies | price-weighted load balancing |
| `models` | fallback models are configured | no model fallback |
| `plugins` | Auto Router settings with `openrouter/auto` (`auto-router`) or `openrouter/auto-beta` (`auto-beta-router`) | |
| `user`, `service_tier` | configured | account-level identity, default tier |
| `usage`, `stream_options`, `transforms` | never (deprecated no-ops); usage is always returned | |

Headers: `Authorization: Bearer …`; `x-anthropic-beta: structured-outputs-2025-11-13`
when a Claude tool is `strict` (OpenRouter strips `strict` otherwise) and
`interleaved-thinking-2025-05-14` for `interleaved_thinking: Some(true)` on Claude.

### Options

`OpenRouterOptions` holds routing options. Client defaults
(`OpenRouterClient::with_default_options`) and per-request options
(`GenerationConfig::with_openrouter_options`) are merged field by field, the request
winning (also inside `provider` and `auto_router`). All options serialize, so they can
live in saved agent configs.

- `provider: ProviderPreferences`: every documented field (`order`, `only`, `ignore`,
  `allow_fallbacks`, `require_parameters`, `data_collection`, `zdr`,
  `enforce_distillable_text`, `quantizations`, `sort` as a strategy or `{by, partition}`,
  `max_price`, `preferred_min_throughput`, `preferred_max_latency`, per-provider
  `options`). `ProviderPreferences::pinned("anthropic")` sends `only` plus
  `allow_fallbacks: false` (`order` would disable sticky routing and the prompt cache that
  comes with it); `ProviderPreferences::zdr_only()` restricts routing to Zero Data
  Retention endpoints (then Claude runs only on Bedrock or Vertex, and OpenAI models mostly
  on Azure).
- `fallback_models` (`models`): off by default. With it OpenRouter tries the next model on
  any error of the previous one, including moderation and refusal errors raised before any
  output, so a refused prompt may be answered by another model; this is opt-in only.
- `auto_router: AutoRouterOptions`: `allowed_models`, `excluded_models` and `cost_tier`,
  plus `provider` overrides, applied only when the model is `openrouter/auto` or
  `openrouter/auto-beta` and ignored otherwise, so they can be set once on the client.
  `AutoRouterOptions::zdr()` keeps router requests on ZDR endpoints with data collection
  denied. How far the router's own task classifier honours ZDR is not documented; for a
  hard guarantee use a dedicated API key with a ZDR guardrail on OpenRouter.
- `auto_exacto: AutoExactoPolicy`. Auto Exacto reorders providers by tool-calling quality
  on every request with tools, overriding sticky routing, so agent loops can miss their
  prompt cache. The default, `CacheAware`, sends `provider.sort: "price"` (which turns Auto
  Exacto off) on tool requests with caching enabled when the model bills cache writes
  (catalog `input_cache_write` above zero, or without a catalog an `anthropic/` or
  `openai/` model), and leaves OpenRouter's routing alone otherwise. `AllowAutoExacto` never
  adds a sort; `DisableAutoExacto` always does on tool requests. An explicit `sort` or
  `order`, or a sorting variant (`:exacto`, `:nitro`, `:floor`), always wins.
- `user` (per-end-user identifier for abuse isolation) and `service_tier`.

App attribution (`HTTP-Referer`, `X-OpenRouter-Title`, `X-OpenRouter-Categories`,
`X-OpenRouter-App-Visibility`) is sent only with
`OpenRouterClient::with_app_attribution(AppAttribution::new(url)...)`, for binaries that
want their own app credited.

### Reasoning

OpenRouter models take `ReasoningConfig::OpenRouter(OpenRouterReasoning)` only: `effort`
(`none` to `max`, clamped by OpenRouter to the nearest level the model supports) or
`max_tokens` (not both), `exclude`, `enabled`, `summary`, `context` and `mode`. Other
variants are rejected; `ReasoningConfig::to_openrouter()` converts them explicitly (OpenAI
effort and summaries, Anthropic budgets or adaptive effort, Gemini budgets and levels) and
fails where there is no equivalent. On Claude, a reasoning `max_tokens` budget must stay
below the request's `max_tokens`; this is checked before sending. `exclude: true` hides
the reasoning text (it is still billed); whether it also withholds the `reasoning_details`
needed to continue tool calls is undocumented, so avoid it in tool loops.

Responses keep `message.reasoning_details` verbatim as
`ReasoningArtifact::OpenRouter { details }`, and every assistant turn sends them back
unmodified and in order: Claude signatures and Gemini thought signatures must match for
tool calls to continue. Streamed detail chunks are merged by `index` (`text`, `summary` and
`data` concatenated, other non-null fields such as `signature` taken). Reasoning artifacts
from other providers are dropped with a debug log. `StreamEvent::ReasoningChunk` carries
`delta.reasoning`, or the text of `reasoning.text` / `reasoning.summary` items; encrypted
items are never shown. A streamed `[REDACTED]` placeholder is never spliced into encrypted
data that arrives in other chunks.

`Model::supports_forced_tool_choice` applies Anthropic's rules to `anthropic/` slugs (and to
a catalog alias's target): no forced tool on Claude Fable 5.1 and Opus 5.5, nor with a
manual thinking budget, which OpenRouter reasoning means on Claude models without adaptive
thinking or with a reasoning `max_tokens` on models that take budgets.

### Mid-conversation effort updates

`Conversation::add_reasoning_effort_update(ReasoningEffort::Low)` changes the effort from
that point on without touching earlier turns, so the prompt cache for them stays valid.
Call it just before the user turn it applies to and keep it in history; it is sent as a
content-less system message carrying `configuration_update` (OpenRouter's Chat
Completions form). A second call before another message replaces the pending update, since
adjacent updates are rejected. An update between an assistant's tool calls and their
results is rejected before sending (it would split the pair), and updates can't be combined
with pro reasoning or OpenAI `-pro` models. Support is per model: OpenRouter names Claude
Fable 5.1 and OpenAI GPT-6 Astra as the first models to accept updates and Claude Opus 5
as accepting them too, enables more as providers roll it out, and answers 400 for a model
without support. Claude rejects an update to `none` (`minimal` is sent as `low`), so arko
rejects one for `anthropic/` models before sending. The other providers reject a
conversation containing an update with `Error::UnsupportedFeature`.

### Responses and streams

| `finish_reason` | `StopReason` |
| --- | --- |
| `stop` | `EndTurn` (`ToolUse` with tool calls; `StopSequence` for a native `stop_sequence`) |
| `tool_calls` | `ToolUse` |
| `length` | `MaxTokens`; the first tool call whose arguments were cut off is dropped with every call after it |
| `content_filter` | `Refusal`: `SafetyClassifier` when `native_finish_reason` is `refusal` (Anthropic's classifier), `Model` when `message.refusal` is set, otherwise `ContentFilter` |
| `error` | an error (see below) |
| anything else | `Other` with the raw value |

A `message.refusal` with any other finish reason is a `Model` refusal; refusal text is never
message text or a `TextChunk`. The raw `finish_reason` is `ModelCallUsage::stop_reason` and
`native_finish_reason` is in `stop_details`. `ModelCallUsage::model` is the model that
served the call, named by its dated permaslug; when it differs from the requested one
(fallbacks, routers, `~…-latest` aliases) `requested_model` is set, and `upstream_provider`
records OpenRouter's `provider` field when present. Permaslugs can reorder the name
(`anthropic/claude-4.5-sonnet-20250929` for `anthropic/claude-sonnet-4.5`), so without catalog
metadata only recognizable changes count: routers, `~` aliases and presets, configured
fallbacks, or another vendor. With it, the served model is compared with the catalog's
canonical slug. Variant suffixes and trailing dates never count.

Streams skip `: OPENROUTER PROCESSING` keep-alives, assemble tool calls by `index` (a new
id at an index already in use starts another call), and treat the final usage chunk, which
repeats the finish reason, as accounting rather than a second end. `ToolCallStart` and
`ToolCallArgumentsDelta` carry the call's 0-based position in the final
`message.tool_calls` (calls in the order they first appear), not OpenRouter's raw index. A
token limit keeps only the calls before the first cut-off one, so every kept call keeps
its announced position; a dropped call may have had a `ToolCallStart`. `End` follows a
finish reason once `[DONE]` or the usage chunk arrives; a stream that stops earlier yields
`Error::StreamTruncated`.

### Usage and cost

`input_tokens` is `prompt_tokens` (the total, including cache reads and writes),
`output_tokens` is `completion_tokens` (including reasoning), with `cached_tokens`,
`cache_write_tokens` and `reasoning_tokens` from the token details. The reported
`usage.cost` is the charge (`BillingKind::ProviderReported`). With bring-your-own-key
routing it is only OpenRouter's fee: the provider's own charge
(`cost_details.upstream_inference_cost`) is recorded as `UsageInfo::upstream_cost` and
both are billed. Prices are used for an estimate only when no cost is reported.

### Errors

Error bodies are classified by `error.metadata.error_type` first, then by status (the
numeric `error.code` stands in for the status on errors that arrive after a 200: in a
stream chunk, as a 200 body with `error` instead of choices, or in `choices[0].error`).

- `refusal` (a provider refusing as an error, such as GPT-6 `bio_policy`/`cyber_policy`
  blocks) is `Error::Refusal` with `SafetyClassifier`; `content_policy_violation` and
  moderation flags (`metadata.reasons`, `flagged_input`) are `ContentFilter`. Never retried.
- 402 is retryable only for the in-flight spending budget
  (`limit_source: openrouter_in_flight_budget`) with a `Retry-After`; exhausted credits or
  key limits are permanent.
- "No available model provider meets your routing requirements" and "No allowed
  providers" (503, sometimes 404) are a permanent configuration error
  (`code: no_eligible_provider`), unless a 503 carries a retry hint (`Retry-After` or a
  delay in the body): OpenRouter sends `Retry-After` "when every attempted provider
  returned a retry hint", so the error is then transient and keeps the delay.
- Rate limits, `provider_overloaded` (capacity), `provider_unavailable`, `timeout` and
  `server` are retryable; `Retry-After` is kept.

### Other generation settings

`openai_reasoning_mode`, `anthropic_thinking_enabled`, `openai_service_tier`,
`anthropic_fast_mode`, `anthropic_inference_geo` and `interleaved_thinking` (except `true`
on Claude) are rejected rather than dropped; use `OpenRouterReasoning` and
`OpenRouterOptions` instead. `anthropic_summarized_thinking` and `long_context` are
accepted as no-ops: OpenRouter already returns summarized Claude thinking and enables
extended context where the model has it.

### Known gaps

- Chat Completions only. Features of the Responses and Messages APIs are unavailable:
  Anthropic `stop_details` category and explanation, other Claude Messages-only features,
  and OpenAI's `phase` for GPT-5.x (dropping it can degrade multi-turn quality on
  GPT-5.3-codex to GPT-5.5; use the OpenAI provider for those). OpenRouter strips OpenAI
  async tools on this API.
- Not exposed: server tools, `response_format`, stop sequences, logprobs, `n > 1`, the
  debug echo and `GET /generation` statistics.
- `provider.max_price` is sent as JSON numbers, as in OpenRouter's routing guide (the
  OpenAPI schema says strings).
- Reasoning returned only as plain `message.reasoning` text, without `reasoning_details`,
  is streamed but not stored, so it isn't replayed on later turns.
- `parallel_tool_calls` is passed through (`Model::supports_disabling_parallel_tool_calls`
  is true) although the catalog lists it for few models; the live suite checks that it
  reaches Claude, GPT and Gemini. Which `error_type` accompanies the "routing
  requirements" 503 is undocumented: if OpenRouter tags it as an outage it stays
  retryable.
- Behaviour the docs leave open is checked by the opt-in live suite: `reasoning_details`
  round trips and tool continuation on Claude, Gemini 3 and GPT-5.x (including the stream
  merge rule and whether streamed encrypted data arrives as `[REDACTED]`), the effect of
  `exclude`, `prompt_tokens` including cached tokens, cache hits on turn 2 with
  `session_id` (also under ZDR), `prompt_cache_key` reaching OpenAI, `require_parameters`
  with `cache_control`/`session_id`, the Auto Router with ZDR and `allowed_models`, cache hit
  rates over tool turns with and without Auto Exacto, effort without `max_tokens` on older
  Claude, numeric `max_price`, GPT-6 effort updates, `parallel_tool_calls: false` on the
  major models, and the `provider` response field:

  ```bash
  OPENROUTER_API_KEY=... cargo test -p arko-ai openrouter::live_tests -- --ignored --nocapture --test-threads=1
  ```
