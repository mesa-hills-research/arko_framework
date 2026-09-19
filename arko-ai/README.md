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

## Provider stop metadata

Anthropic responses now retain `stop_reason` and, when supplied, `stop_details` on `GenerationResponse.model_call_usage`. Both streaming `message_delta.delta` and non-streaming responses are covered.

The agent runtime already persists this object in `ModelCallUsageReported` events and task usage history. Consumers can distinguish `refusal`, `max_tokens`, `tool_use`, `end_turn`, and future provider values without inspecting response text. Refusal details remain provider-supplied JSON; category and explanation can be null. Missing fields in older saved records deserialize as `None`, and absent or null details are omitted when serialized.

OpenAI's Responses API exposes typed `refusal` content and `response.refusal.delta` / `response.refusal.done` events rather than Anthropic's literal `stop_reason` field. Arko normalizes those signals to `stop_reason: "refusal"`; `stop_details` records the response status, item identifiers, content indices, and supplied refusal explanations. Ordinary text mentioning refusal is not classified as a refusal. A finalized refusal event replaces accumulated deltas rather than duplicating their text.

OpenAI `response.incomplete` terminal events now preserve usage and the reason in `incomplete_details`, including `content_filter` and token limits. Otherwise the response status is used as the normalized reason. This does not identify which internal classifier produced a refusal. Refusal content without an ordinary `text` field is parsed correctly and its explanation is retained in the assistant message. Partial streamed tool calls are not reconstructed after refusal or incomplete termination.

Retry policies, conversation resets, and model fallbacks are unchanged. These diagnostics cannot reconstruct fields discarded by earlier versions or explain a process signal when no terminal API response was recorded.

References: [Anthropic streaming refusals](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals), [official Anthropic streaming event schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_delta_event.py), [OpenAI refusal and content-filter handling](https://developers.openai.com/api/docs/guides/structured-outputs), [OpenAI streaming refusal events](https://platform.openai.com/docs/api-reference/responses-streaming/response/refusal).

### Long requests and embedding tokenizers

The shared model HTTP client has no total request deadline. Connection setup and idle
connection-pool cleanup are separately bounded. Hosts can cancel requests explicitly.

`EmbeddingConfig::with_tokenizer_revision()` pins a Hugging Face tokenizer revision.
Tokenizers are cached per repository/revision with truncation and padding disabled;
concurrent first loads share initialization. Embedding tokenization runs on a blocking
worker. Voyage responses are reordered by provider item indexes and reject missing,
duplicate, dimension-invalid, nonfinite, or zero vectors. Provider failures preserve
HTTP status and Retry-After metadata for caller-controlled retries.
