# Arko AI

Rust clients for language models and embeddings, with provider configuration,
streaming responses, tool calls, reasoning controls, usage metadata, and OpenAI
OAuth support.

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

References: [Anthropic streaming refusals](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals), [official Anthropic streaming event schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_delta_event.py), [OpenAI refusal and content-filter handling](https://developers.openai.com/api/docs/guides/structured-outputs), [OpenAI streaming refusal events](https://platform.openai.com/docs/api-reference/responses-streaming/response/refusal), [Gemini Interactions API](https://ai.google.dev/api/interactions-api-v1).

## Usage and cost

`UsageInfo` token counts mean the same thing for every provider:

- `input_tokens` is the total input, including cache reads (`cached_tokens`) and cache writes (`cache_write_tokens`). Anthropic reports uncached input separately, so arko adds the cache counts back; for streams it uses the cumulative counts in the final `message_delta`. **Usage records saved before this change stored only uncached input for Anthropic.**
- `output_tokens` includes reasoning tokens; `reasoning_tokens` reports them separately when the provider does (OpenAI `output_tokens_details.reasoning_tokens`, Gemini thought tokens).
- `cache_write_1h_tokens` is the part of Anthropic cache writes made with a 1-hour TTL.

Costs are estimates at list prices from `model.rs`, computed for what the call actually used:

- **Service tier.** The tier the response reports wins over the one requested (OpenAI echoes `service_tier`, for example `default` when a Fast request was downgraded; Anthropic reports `speed: "fast"` and `service_tier`). `GenerationConfig::with_openai_service_tier` requests OpenAI Flex or Fast (sent as `priority`) on models that offer it, with API-key auth only. Tier prices are per model (`ModelPricing::service_tiers`); tiers without a published price (Anthropic priority, OpenAI scale) are estimated at standard rates.
- **Cache TTL, residency and dates.** Anthropic 1-hour cache writes use their own rate; `inference_geo: "us"` applies the 1.1x multiplier on Claude 4.6 and later; `ModelPricing::schedule` switches prices on a date (for example the end of promotional pricing). Tier and residency multipliers stack.
- **Missing prices.** A token class without its own price (cached input on Pro models, cache writes on models that don't charge extra for them) is billed at the input rate, including long-context rates.

`total_cost` is the estimate; `billed()` is what the account is charged, decided by `billing`: `Metered` (the estimate), `Subscription` (ChatGPT OAuth: zero, while the estimate stays visible), `NotCharged` (zero: an Anthropic classifier refusal before any output in a category Anthropic does not bill) or `ProviderReported` (`reported_cost`, used by providers that report their actual charge). `UsageInfo` implements `AddAssign` and `Sum`; summing calls billed differently yields `BillingKind::Mixed` with the billed amounts added. Workflow budgets count `billed()`.

## Errors and retries

`Error::retry_class()` returns `Permanent`, `Transient` or `Capacity`; `is_retryable()` and `is_capacity_error()` are views of it. Provider error bodies are read generically: string codes (`error.code`, `error.type`, `error.status`), OpenRouter's `error.metadata.error_type`, and numeric HTTP-style codes, which stand in for the status on stream errors that arrive after a 200. Delay hints come from `Retry-After`, OpenAI's `x-ratelimit-reset-*` headers, or body fields (`retry_after_ms`, `retry_after`, `resets_at`) for every provider.

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
