# Arko AI

Rust clients for language models from OpenAI (API key or ChatGPT OAuth), Anthropic,
Google Gemini, and [OpenRouter](#openrouter), for Claude models run through the user's
installed [Claude Code](#claude-code) CLI, and for Voyage text embeddings. One
`LLMClient` sends a `Conversation` to the provider of its model and returns a normalized
response: text and reasoning, tool calls, stop reasons and refusals, token usage, and
cost.

The [defaults reference](../docs/reference-map.md#arko_ai) lists every setting's
default.

## Installation

```toml
[dependencies]
arko_ai = { path = "../arko_ai" }
tokio = { version = "1.37", features = ["full"] }
futures = "0.3"    # for consuming response streams
serde_json = "1"   # for tool definitions
```

## Quick start

```rust
use arko_ai::{GenerationConfig, LLMClient, Model, StreamEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), arko_ai::Error> {
    // Configures every provider whose credentials are in the environment.
    let client = LLMClient::from_env()?;
    let mut conversation = client
        .new_conversation(Model::anthropic_claude_opus_5_5())
        .with_system_message("Answer in one short paragraph.");
    conversation.add_user_message("What does a borrow checker do?");

    let config = GenerationConfig::default().with_max_tokens(2048);
    let response = client.generate_text(&conversation, config.clone()).await?;
    println!("{}", response.text());
    println!("Billed: ${:.6}", response.usage.billed());
    conversation.add_assistant_message_from_response(response);

    // The same conversation, streamed.
    conversation.add_user_message("And what is a lifetime?");
    let mut stream = client.generate_stream(&conversation, config.clone()).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextChunk(text) => print!("{text}"),
            StreamEvent::End(response) => {
                conversation.add_assistant_message_from_response(*response);
            }
            _ => {}
        }
    }
    Ok(())
}
```

`generate_text` returns a `GenerationResponse` with the assistant `message`, its
`usage` and `model_call_usage`, a normalized `stop_reason`, and any `refusal`.
`generate_stream` yields `TextChunk` and `ReasoningChunk` events, `ToolCallStart` and
`ToolCallArgumentsDelta` for native tool calls, `Refusal`, and finally `End` with the
complete response. Add each response to the conversation with
`add_assistant_message_from_response`, which keeps the provider's reasoning artifacts for
replay and leaves out refused turns that must not stay in history.

## Providers and credentials

`LLMClient::from_env()` configures every provider whose credentials are set and ignores
empty values. `LLMClient` reads no environment variables unless it is called. Setting only
some of the `OPENAI_OAUTH_*` variables is an error (`Error::AuthenticationFailed`), not a
silent switch to `OPENAI_API_KEY` and API billing; see
[ChatGPT OAuth credentials](#chatgpt-oauth-credentials) for what happens to an
`OPENAI_OAUTH_REFRESH_TOKEN`. `LLMClient::from_env_with(lookup)` reads the variables
through a function instead, for example to hide the `OPENAI_OAUTH_*` variables when the
OpenAI credentials come from a file.

| Provider | Variables |
| --- | --- |
| OpenAI | ChatGPT OAuth from `OPENAI_OAUTH_ACCESS_TOKEN` and `OPENAI_OAUTH_ACCOUNT_ID` (optional `OPENAI_OAUTH_REFRESH_TOKEN`); otherwise `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY`, otherwise `GEMINI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

The builder methods configure providers explicitly: `with_openai_api_key`,
`with_openai_chatgpt_oauth(access_token, refresh_token, account_id)`,
`with_codex_auth_file` (ChatGPT OAuth from a Codex `auth.json`; see
[ChatGPT OAuth credentials](#chatgpt-oauth-credentials)), `with_anthropic_api_key`,
`with_google_api_key`, and `with_openrouter_api_key`. `with_openai_client`,
`with_anthropic_client`, `with_google_client`, and `with_openrouter_client` install a
provider client built with its own options, such as a base URL for a proxy or test
server. A request to a provider that is not configured fails with
`Error::MissingApiKey`. Clones of an `LLMClient` share its credentials.

Gemini requests use Google's Interactions API. OpenAI requests carry no `originator` or
`User-Agent` header unless the application sets one with `with_openai_originator` or
`with_openai_user_agent`; use your own application's name. `with_request_trace` writes
every OpenAI request and response to a directory for diagnosing prompt caching
(`RequestTrace`); requests to other providers are not traced, and the first one to each
provider logs a warning saying so. A request that fails while refreshing OAuth credentials
or detecting an account change is recorded as an error with its cause, rather than as an
abandoned call. `with_openai_request_hook` can adjust OpenAI requests before they are sent.

## Models

`Model` describes a model: its provider, ID, context and output limits, image support,
and prices. The registry covers the OpenAI, Anthropic, and Google models arko_ai
supports: `Model::list(provider)`, `Model::all()`, `Model::find(id)`, and
`Model::get(provider, id)` look models up, and constructors such as
`Model::anthropic_claude_opus_5_5()` or `Model::openai_gpt_6_sol()` name them directly.
The example CLI lists the registered IDs and the aliases it accepts:

```bash
cargo run -p arko_ai --example llm_client_cli -- models
```

`Provider::ClaudeCode` models are the Anthropic models run through Claude Code:
`Model::claude_code(Model::anthropic_claude_opus_5_5())`, or
`Model::get(Provider::ClaudeCode, "claude-opus-5-5")`. `Model::list(Provider::ClaudeCode)`
mirrors the Anthropic registry; `Model::all()` and `Model::find` leave the mirror out, so
IDs stay unique. See [Claude Code](#claude-code).

OpenRouter models are not in the registry: `Model::openrouter(slug)` accepts any slug,
and the OpenRouter catalog supplies limits and prices on request; see
[Models and the catalog](#models-and-the-catalog).

`Model::max_input_tokens` is the provider's published input limit. The ChatGPT OAuth
backend accepts less: `LLMClient::max_input_tokens(&model)` returns the limit for the
client's auth path, capping OpenAI models at `CHATGPT_OAUTH_MAX_INPUT_TOKENS` (272,000)
over ChatGPT OAuth, and `effective_max_input_tokens(&model, chatgpt_oauth)` computes the
same without a client. Both return `None` when the limit is unknown.

## Generation settings and tool calls

`GenerationConfig` holds the per-request settings, built from
`GenerationConfig::default()` with methods such as `with_max_tokens`, `with_temperature`,
`with_reasoning`, `with_verbosity`, `with_prompt_caching`, and `with_tool_config`. A config
deserializes from partial JSON or YAML, with omitted fields at their defaults.

The default sends no temperature: `temperature` is an `Option<f64>`, and `None` lets the
model's default apply. A set temperature is rejected for OpenAI reasoning models and
Gemini, and dropped with a warning for Claude models that take no sampling parameters.
Configs saved with the earlier `temperature_specified` flag still load (`false` means no
temperature). A `temperature` written without that flag is now sent; earlier versions
ignored it.

Each provider takes its own `ReasoningConfig` variant:
`ReasoningConfig::openai(effort, include_summaries)`, `anthropic(budget_tokens)` or
`anthropic_adaptive()`, `gemini(thinking_level, include_summaries)` or
`google(thinking_budget, include_summaries)` (mapped to a thinking level), and
`openrouter_effort(effort)`. Without a reasoning configuration, Gemini requests send no
`thinking_level` or `thinking_summaries`, so each model thinks at its documented default
(medium on Gemini 3.5 to 3.8 Flash, minimal on 3.1 and 3.5 Flash-Lite, high on 3 Pro,
3 Flash and 3.1 Pro, dynamic on 2.5) and returns no thought summaries; an explicit level
is validated against the model.

One config can serve models of several providers. Settings the model's provider has no
use for are dropped from the request, with a warning logged once per provider and
setting: another provider's `ReasoningConfig` variant (the model's default reasoning then
applies), `openai_*` settings and `verbosity` on Claude and Gemini, `anthropic_*`
settings, `long_context` and `interleaved_thinking` on GPT and Gemini, `openrouter` on
the direct providers, and on OpenRouter the settings listed under
[Other generation settings](#other-generation-settings). A value the provider can't
accept for a setting it does use, such as an effort the model lacks, is an error.

Prompt caching is on by default (`Conversation::enable_cache`); a new conversation's
`prompt_cache_key` is a random UUID (`with_prompt_cache_key` sets another; restored
conversations keep their saved key), so give each independent agent or task its own
conversation.

`LLMClient::prompt_cache_lifetime(&model, &config)` reports how long the provider keeps a
request's prompt cache after its last use: `PromptCacheLifetime::Known(duration)`,
`Unknown` or `Disabled`. Every cache hit starts the lifetime again, so a conversation that
makes a request within it stays cached however long it waits in between. Anthropic, and
Claude on OpenRouter: 5 minutes, or 1 hour with `anthropic_cache_ttl` set to `OneHour`.
OpenAI, every model, with an API key or ChatGPT OAuth: a conservative 5 minutes, the lower
end of the documented in-memory retention, on purpose. Longer retention exists (GPT-5.6
and later default to a 30-minute `prompt_cache_options.ttl` on the API, so none is sent),
but the ChatGPT OAuth backend caches implicitly only, with a lifetime neither documented
nor configurable, and a request after a longer idle time can miss the cache. Gemini and
other OpenRouter models: `Unknown`. Models without caching, and caching turned off where
the request can turn it off (Anthropic, Claude on OpenRouter, GPT-5.6 and later with an
API key): `Disabled`. The `prompt_cache` module has the constants and the providers'
documentation links.

`with_prompt_caching(false)`, or `Conversation::with_cache_disabled()`, turns caching off
where the provider has a switch. Where caching is implicit it stays on:

| Provider | Caching | With caching turned off |
| --- | --- | --- |
| Anthropic, and Claude on OpenRouter | only through the `cache_control` arko sends while caching is on | nothing is cached |
| OpenAI GPT-5.6 and GPT-6, API key | `prompt_cache_options.mode: "implicit"` | `"explicit"` with no breakpoints: nothing is cached |
| OpenAI before GPT-5.6, API key | automatic | still cached; it can't be turned off |
| ChatGPT OAuth, every model | implicit, managed by the backend | still cached; ignored, with a warning logged once |
| Gemini | implicit, managed by Google | still cached; it can't be turned off |
| OpenRouter, other models | `prompt_cache_key` sent | no key is sent; OpenAI GPT-5.6 and later get explicit mode as above, other upstreams keep caching implicitly |

Every OpenAI request carries the conversation's `prompt_cache_key` either way, and every
OpenRouter request its `session_id`; the key routes requests but doesn't turn caching on
by itself.

Native tool calls use JSON Schema definitions (built here with `serde_json`), continuing
the quick start's conversation:

```rust
use arko_ai::{ToolConfig, ToolDefinition, ToolResult};
use serde_json::json;

let weather = ToolDefinition::new(
    "get_weather",
    "Returns the current weather for a city.",
    json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }),
);
let config = config.with_tool_config(ToolConfig::new(vec![weather]));

let response = client.generate_text(&conversation, config.clone()).await?;
let calls = response.message.tool_calls.clone();
conversation.add_assistant_message_from_response(response);
if !calls.is_empty() {
    // Run each call, then send every result back in one message.
    let results = calls
        .iter()
        .map(|call| ToolResult::from_tool_call(call, "18 °C and sunny"))
        .collect();
    conversation.add_tool_results(results);
    let response = client.generate_text(&conversation, config).await?;
}
```

`ToolConfig::with_tool_choice` selects `ToolChoice::Auto` (the default), `Required`,
`None`, or `Specific(name)`. A forced choice (`Required` or `Specific`) is rejected before
sending (`Error::InvalidRequest`) where the model does not accept it: on Claude Fable 5.1,
Opus 5.5 and Sonnet 5.5, and on Claude models with a manual thinking budget, on the Anthropic
provider and for Claude models on OpenRouter; `Model::supports_forced_tool_choice(config)`
tells in advance. `with_parallel_tool_calls(false)` asks for one call per response. It
is available where `Model::supports_disabling_parallel_tool_calls` is true (OpenAI,
Anthropic, OpenRouter); for Gemini the request fails with `Error::UnsupportedFeature`.

A call's `arguments` are parsed JSON; arguments that are not valid JSON are kept as text
in `ToolCall::invalid_arguments`, with `arguments` set to `null`, and
`ToolResult::error_from_tool_call` reports a failed call to the model.
`GenerationResponse::single_tool_call(name)` returns the one call a forced tool choice
asked for, or an error saying whether the response was refused, cut off by a limit, or
called something else, or whether the call's arguments are not valid JSON.

### Tool images

A tool can return images with its result:
`ToolResult::with_images(vec![ToolResultImage::new(label, image)])`, where the label is
short text such as a file name and size. Where the model takes images inside a tool
result (`Model::accepts_tool_result_images()`), they go inside that result, after its
output text; otherwise they follow all of the turn's results, each after the text
`"Image from tool {name}, call {call_id}: {label}"` (`unidentified` when the result has
no call id), the layout arko_agents has always used for tool images:

| Provider and model | Result images |
| --- | --- |
| OpenAI, API key and ChatGPT OAuth | Inside `function_call_output.output`: `input_text` items (the output, then each label) and `input_image` items |
| Anthropic | Inside `tool_result.content`: `text` blocks (the output, then each label) and `image` blocks |
| Gemini 3 and later | Inside `function_result.result`: text (the output, then each label) and image blocks, as Google advises for Gemini 3.x |
| Gemini 2.x, and Gemini IDs whose version can't be read | After the results, in a `user_input` step; Gemini 2.5 rejects images in function responses |
| OpenRouter | After the `tool` messages, in a user message; how each upstream receives images in a tool message is not documented |

**Text after tool results on Gemini 3.x.** A user message that holds tool results plus
other content (an `ask_user` answer, a recovery note, environment details, or images
saved as sibling content) is sent to Gemini 3 and later without a `user_input` step after
the results: the other content is folded into the message's last `function_result`, as
Google advises ("append any extra instructions to the end of the function response text
separated by two newlines"). Each text is appended to that result's output text after a
blank line, and each image, with the text just before it as its label, follows the
result's own images; order is kept. User messages of only text and images that come
directly after the tool-result message, before the next assistant message (such as the
context updates an agent runtime adds as separate messages: a file-tree change, or a
message from another workflow agent), are folded into the same `function_result` the same
way, after the message's own content and in order; an image's label is taken only from
its own message. A user message after an assistant message is a new input and is sent as
a `user_input` step. Gemini 2.x and the other providers keep the text in a separate user
turn after the results, which every provider accepts; requests without text or images
after their results are unchanged.

Empty texts (an empty output with images, an empty label) are left out of the result. A
result without images is sent exactly as before result images existed, and conversations
saved before then (with tool images as sibling content of the message) load and are sent
unchanged, except to Gemini 3 and later: there a tool-result message that also holds text
or images is now folded as described above, so a saved conversation with such a message
is sent differently than before, and its first request after the upgrade misses Gemini's
implicit cache from that message on.

Result images count as image input: a model without it rejects them before sending.
`Message::has_images()`, `image_count()` and `remove_images()` cover images both as content
and inside results; `Conversation::redact_tool_result` replaces only a result's text and
keeps its images.

## Stop reasons and refusals

Every `GenerationResponse` carries a normalized `stop_reason` (`EndTurn`, `ToolUse`,
`MaxTokens`, `StopSequence`, `Refusal`, `PauseTurn`, `ContextWindowExceeded`, or `Other`
with the provider's value) and, when the model or a filter declined, a `refusal`.
Streams emit `StreamEvent::Refusal` once, just before `End`, which carries the same data.
The provider's raw `stop_reason` and `stop_details` stay on
`GenerationResponse::model_call_usage` (and in arko_agents' `ModelCallUsageReported`
events and task usage history); saved records without them load with `None`.

`Refusal.source` says how to treat the turn:

- `SafetyClassifier`: Anthropic's `stop_reason: "refusal"`, with `stop_details.category`
  and `explanation` when supplied (both can be null). Anthropic requires the refused turn
  to be removed before continuing, so the message is left empty and text generated before
  the refusal moves to `partial_output`. `Conversation::add_assistant_message_from_response`
  doesn't append such a turn.
- `Model`: OpenAI's structured `refusal` content. The refusal text is in `explanation`,
  never in the message text or streamed as `TextChunk`. Ordinary text that mentions a
  refusal is not classified as one.
- `ContentFilter`: OpenAI `incomplete_details.reason: "content_filter"`. Partial output
  moves to `partial_output`, as for classifier refusals.

A stream that ends before the provider reports completion yields `Error::StreamTruncated`
(retryable) instead of an `End`. When a token limit cuts a response short, tool calls
whose arguments finished are kept and a call that was still being written is dropped.

Google's Interactions API reports only an interaction `status`: `completed` maps to
`EndTurn` (or `ToolUse` with function calls), `requires_action` to `ToolUse`, and
`incomplete` to `MaxTokens`. It has no refusal or safety reason to report. Gemini
generation-blocked error codes (`safety`, `recitation`, `prohibited_content`, ...) become
`Error::Refusal` with a `ContentFilter` source.

OpenRouter maps `finish_reason`, `native_finish_reason` and `message.refusal` onto the
same sources, and reports refusals raised as errors as `Error::Refusal`; see
[Responses and streams](#responses-and-streams).

References: [Anthropic streaming refusals](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals), [official Anthropic streaming event schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_delta_event.py), [OpenAI refusal and content-filter handling](https://developers.openai.com/api/docs/guides/structured-outputs), [OpenAI streaming refusal events](https://platform.openai.com/docs/api-reference/responses-streaming/response/refusal), [Gemini Interactions API](https://ai.google.dev/api/interactions-api-v1).

## OpenAI Responses: replay, prompt caching and ChatGPT OAuth

OpenAI requests use the Responses API: `api.openai.com/v1/responses` with an API key,
and the ChatGPT backend's `chatgpt.com/backend-api/codex/responses` with ChatGPT OAuth.

**Prompt caching and reasoning context.** Every OpenAI request sends the conversation's
`prompt_cache_key`, for every model; `prompt_cache_options` is sent only to GPT-5.6 and
GPT-6 with API keys. Reasoning GPT-5.6 requests also send `reasoning.context:
"all_turns"`, so reasoning from earlier turns is rendered back to the model; over ChatGPT
OAuth, GPT-6 requests send it too, as Codex does. With API keys it is left out for GPT-6,
since the API documents the value only for GPT-5.6. Without a configured effort, API-key
requests send no `reasoning.effort` (the model's default applies), while ChatGPT OAuth
requests send `medium`, as Codex always sends an effort.

**Output items and replay.** A response's assistant output is assembled from the
`response.output_item.done` stream events, in order, as Codex does; the ChatGPT backend's
`response.completed` usually carries no items, and items only it holds are merged in. The
assistant message keeps every replayable item (`reasoning` with `encrypted_content`,
`message` with its `phase`, finished `function_call`) verbatim, minus `status`, in
`ReasoningArtifact::OpenAIOutputItems`. Later requests replay them unchanged and in
order, for example `reasoning`, `message` (commentary), `function_call`, then the call's
`function_call_output`; no empty assistant message is added for tool-only turns. A
replayed `function_call` needs the message to still hold that call (by `call_id`), and
calls the message holds without an item follow in the reconstructed shape. When the items
hold no `message`, the message's own text follows them; arko_agents relies on both when it
keeps a model's refusal without its tool calls and with the refusal explanation as its
text. Reasoning that nothing follows (a response cut off after its reasoning, or whose
calls were all dropped) is left out of the replay, since the API rejects a reasoning item
without its following item ([openai/codex#1498](https://github.com/openai/codex/issues/1498)).
Messages without items (conversations saved with `OpenAIEncryptedReasoning`,
application-built messages, and other providers' messages) are sent as reasoning,
function calls, then the text.

**ChatGPT OAuth requests** always stream (`generate_text` collects the stream) and carry
the conversation's identity, like Codex: `session-id`, `thread-id` and
`x-client-request-id`, all set to the conversation's `prompt_cache_key`. The backend
derives prompt-cache affinity from `session-id`, so give each agent its own conversation.
No `OpenAI-Beta` header is sent on either path. The backend rejects `max_output_tokens`
("Unsupported parameter"), so `GenerationConfig::max_tokens` is never sent over ChatGPT
OAuth and the model's own output limit applies; a set limit is ignored with a warning
logged once. The backend also caches implicitly and takes no cache options, so turning
prompt caching off has no effect there (also warned about once). It also does not accept
`reasoning.mode`; `GenerationConfig::openai_reasoning_mode` is omitted on this auth path
and warns once when configured.

**Turn state.** The backend returns an `x-codex-turn-state` routing token.
`GenerationResponse::openai_turn_state` carries it, and
`Conversation::add_assistant_message_from_response` keeps the first token of each user
turn in `Conversation::openai_turn_state` (serialized, so a resumed task keeps it). Every
later OAuth request of that turn, tool-result follow-ups and retries alike, echoes it
unchanged. A user message without tool results starts a new turn and drops the token; it
is never sent in another turn or with API keys. The token is tied to its turn's starting
message by index and by a hash of the message, so editing `messages` directly (rewinding
and pushing a different message at the same index) also drops it.

**Plan usage.** Every ChatGPT OAuth response reports the plan's usage-limit windows in
`x-codex-*` headers; they are parsed into `ModelCallUsage::plan_usage` (`PlanUsage`: plan
type, active limit, and the primary and secondary windows' used percentage, length, and
reset time). arko_agents reports them as events and can act on a threshold.

**Reroutes.** `ModelCallUsage::served_model` is the model the provider reports
(`openai-model` / `x-openai-model` header, else the response's `model`, else
`response.created`), and `rerouted` is true when any reported model differs from the
requested one, ignoring case and snapshot date suffixes.
`GenerationResponse::served_model()` reads the first; `GenerationResponse::rerouted()` and
`ModelCallUsage::served_model_changed()` are true for an OpenAI reroute and for an
OpenRouter call whose `requested_model` is set.

References: [OpenAI reasoning context](https://developers.openai.com/api/docs/guides/reasoning), [openai/codex](https://github.com/openai/codex) (`codex-rs/core/src/client.rs`, `codex-rs/codex-api/src/sse/responses.rs`).

### ChatGPT OAuth credentials

ChatGPT OAuth refresh tokens rotate: a refresh returns a new refresh token and the old one
stops working. The OAuth server treats a second use of the old token as a reused token,
which signs the user out of Codex. Credentials shared with Codex therefore have to stay in
step with its `auth.json`.

**Where credentials come from.**

- `with_openai_chatgpt_oauth(access_token, refresh_token, account_id)` keeps the
  credentials in memory only. Save rotated tokens yourself from
  `LLMClient::openai_chatgpt_oauth_credentials()`.
- `from_env` compares `OPENAI_OAUTH_REFRESH_TOKEN` with the refresh token in the Codex
  `auth.json` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`) by SHA-256
  fingerprint. When they match for the same account, the client uses that file as its
  credential store, as if built with `with_codex_auth_file`, so rotated tokens are written
  back and Codex stays signed in; this is logged at info level. Otherwise the
  credentials are kept in memory, and the first rotation logs a warning: the refresh
  token in the environment stops working once it is used, so save the new tokens from
  `openai_chatgpt_oauth_credentials()`, or load the credentials with `CodexAuthFile`
  instead.
- A client built from a Codex `auth.json` keeps the file in step with it through a
  `CodexAuthStore`. Load the file with `CodexAuthFile::load(path)` or
  `CodexAuthFile::load_default()` (`$CODEX_HOME/auth.json`, else `~/.codex/auth.json`),
  then build the client with `LLMClient::new().with_codex_auth_file(&file)?`,
  `file.openai_client()?`, or `file.client()?`. Both the nested `tokens` layout Codex
  writes and top-level token fields are read.
- Missing or malformed login files report the path; a missing file also tells the caller to
  run `codex login`.
- Other storage implements `ChatGptOAuthStore` and is installed with
  `OpenAIClient::with_credential_store`.

**Without a refresh token** the access token is used until the server rejects it:
`refresh_openai_oauth_token_if_expiring` returns `None`, even once the token's expiry has
passed, and a request the server rejects with `token_expired` fails with
`Error::AuthenticationFailed` saying there is no refresh token to renew it.

**When a refresh happens.** A client refreshes when asked
(`refresh_openai_oauth_token`, or `refresh_openai_oauth_token_if_expiring`, which the
arko_agents runtime calls before each request), and once after an HTTP 401
`token_expired`, then resends the request. On a client with a store, any other HTTP 401
re-reads the store, as Codex's unauthorized recovery re-reads `auth.json`: newer
credentials there are adopted and the request is sent once more, without a refresh;
otherwise the 401 is returned, as it is at once on a client without a store. When a
refresh request fails, the store is read again, and credentials another program rotated
in the meantime are adopted instead of returning the error.

**Every refresh re-reads `auth.json` first.** What the file holds decides what happens;
the first matching row applies:

| The file holds | Then |
| --- | --- |
| Nothing readable (for example after `codex logout` deleted it), or another account | The refresh stops with `Error::AuthenticationFailed`, a permanent error. |
| Spent credentials (see below) | The client's credentials are saved over them and refreshed. If those are spent too, or have no refresh token, the refresh stops: sign in to Codex again. |
| The client's credentials | They are refreshed. |
| Other live credentials of the same account | They are adopted without a refresh: Codex refreshed, or the user signed in again. Credentials the host installed in the client, which the file never held, give way to the file's. For `refresh_openai_oauth_token_if_expiring`, adopted credentials that also expire within the margin are refreshed. |

An unsaved rotation of the client's account changes the first two rows (see "When the
save fails" below). A file that is not valid JSON is read up to five times, 30 ms apart,
before it counts as an error, since Codex writes `auth.json` in place.

**Rotated tokens are saved at once.** As soon as a refresh succeeds, the new tokens and
the refresh's `id_token` are written to `auth.json`. Other fields and the token layout are
kept; `last_refresh` is set to the current time (in Codex's nested layout, or when the
file has the field), and the file is written back as pretty-printed JSON. The previous
contents go to `auth.json.bak` (one backup, replaced each time), and
the new contents go to a temporary file that is synced and then renamed over `auth.json`.
On Unix the backup and the temporary file are readable only by the owner until they get
`auth.json`'s permissions. Tokens another process wrote are never overwritten: when the
file holds other credentials than the ones read before the refresh, nothing is written
and the save fails, and the file is compared once more just before the rename. Codex
takes no lock on `auth.json`, so a write in the last instant cannot be ruled out.

**Spent refresh tokens are never used again.** A refresh token is spent once a refresh
with it has returned a new one, or the server has refused it as reused, expired, or
revoked (`refresh_token_reused`, `refresh_token_expired`, `refresh_token_invalidated`).
The process keeps one record of spent tokens, as SHA-256 digests, and checks it just
before every refresh request, including those of `refresh_chatgpt_oauth_token`. Refresh
tokens are unique, so the record applies whichever client or token URL spent a token. A
spent token is never sent again, and credentials holding one are neither adopted from the
store nor saved to it; the refresh fails with `Error::AuthenticationFailed` instead.
`CodexAuthFile::persist` and `persist_tokens` write the credentials they are given
without this check.

**When the save fails.** Rotated tokens that cannot be written to `auth.json` go to a new
file next to it, `auth.json.recovered-<unix seconds>`, which on Unix only the owner can
read (mode 0600), and the refresh returns an error naming that file. If even that file
cannot be written, the error says to sign in to Codex again. The client keeps using the
new tokens. Every client of the same `auth.json` in the process, including ones created
later, then knows of this unsaved rotation (the latest one of each account) until the
file holds live credentials of that account again:

- While the file still holds the spent credentials, the next refresh of the account
  through any client saves the rotation first, with its refresh's `id_token`; the other
  clients then take it from the file. No client is handed the rotated tokens before they
  are in the file.
- While the rotation cannot be saved, or the file cannot be read or holds another
  account, every refresh of the account stops with an error naming the recovery file (if
  there is one), so the client holding the rotation does not refresh it again.
- If the file gets other live credentials meanwhile (the user signed in again), every
  client uses those, including the one holding the rotation, which is dropped; the tokens
  stay in the recovery file.

To recover by hand, put the recovery file in place of `auth.json` before Codex next
refreshes, or sign in to Codex again.

**Concurrent refreshes.** Clones of a client share its credentials and refresh one at a
time. Clients created separately from one `auth.json` (each `openai_client()` call
creates its own store) share one refresh lock per file in the process, held from the
re-read to the save, so the second client finds the first one's tokens in the file
instead of refreshing with the rotated refresh token. A client takes that lock before its
own credential lock, so while it waits for another client's refresh, its requests go on
with the token it holds.

**Timeouts and shutdown.** A refresh request times out after 60 seconds with a retryable
error, so a stalled token endpoint cannot hold the credentials longer than that. Codex
sets no such timeout: a server that rotates the token but answers after 60 seconds loses
that rotation, as when the connection drops mid-answer. Once a refresh request is sent,
the refresh runs to completion, installing and saving the new tokens, even if its caller
is dropped. A Tokio runtime shutting down still cancels it, and tokens it had not saved
are lost. A host that must not lose a rotation stops starting refreshes (finishes or
cancels its agent runs) and then awaits `LLMClient::openai_chatgpt_oauth_credentials()`
on each client before shutting the runtime down: a refresh in flight holds the credential
lock until its tokens are installed and saved, so the call returns after it.

**What cannot be detected.**

- A refresh whose response is lost (the connection drops, or the request times out after
  it was sent) may still have rotated the token. The old token is not recorded as spent,
  so the next refresh sends it; the server refuses it, and it is recorded then.
- Clients that share neither their credentials nor a store's refresh lock can send the
  same token at the same time: a token is known to be spent only once a refresh with it
  has been answered.
- Tokens spent by another process, or by an earlier run of this one, look live. An old
  `auth.json` restored from a backup is adopted like new credentials, and an unsaved
  rotation of its account is dropped (the tokens stay in their recovery file).

`CodexAuthFile::sync(&client)` brings a client configured another way into step with the
file. Compared with the credentials that `CodexAuthFile` value last read or wrote, it
saves credentials the client rotated or loads newer ones the file holds. It fails when
both rotated, since only one rotation can be valid, and it never saves or loads another
account's credentials or a spent refresh token.

**Refreshing proactively.** `LLMClient::refresh_openai_oauth_token_if_expiring(margin)`
does nothing while the held access token expires later than `margin`. Otherwise it takes
the credential lock, checks again, and re-reads the store before refreshing, so
concurrent callers rotate once (checking `is_openai_token_expired` and then refreshing
can rotate twice). It returns `Some` when it
installed new credentials (refreshed, or adopted from the store) and `None` when nothing
needed refreshing, including for API-key and unconfigured clients. An explicit
`refresh_openai_oauth_token` that waits while another refresh replaces the credentials
returns those instead of refreshing again. The opt-in OAuth rate-limit retries of
`refresh_openai_oauth_token` (`with_openai_chatgpt_oauth_rate_limit_retries`, or
`OpenAIClient::with_oauth_rate_limit_retries`) stop after 8 retries and honor at most 15
minutes of a server hint per wait; no lock is held while waiting, and a retry that finds
the credentials replaced returns them. They apply only to explicit
`refresh_openai_oauth_token` calls: the refreshes requests make on their own (after a 401
`token_expired`, and `refresh_openai_oauth_token_if_expiring`, which the arko_agents
runtime calls before each request) never wait, and a 429 from them is the request's
error, which the agent's own rate-limit recovery handles.

## Usage and cost

`UsageInfo` token counts mean the same thing for every provider:

- `input_tokens` is the total input, including cache reads (`cached_tokens`) and cache
  writes (`cache_write_tokens`). Anthropic reports uncached input separately, so arko_ai
  adds the cache counts back; for streams it uses the cumulative counts in the final
  `message_delta`. Anthropic usage records saved by earlier versions of arko_ai hold only
  the uncached input in `input_tokens`.
- `output_tokens` includes reasoning tokens; `reasoning_tokens` reports them separately
  when the provider does (OpenAI `output_tokens_details.reasoning_tokens`, Gemini thought
  tokens).
- `cache_write_1h_tokens` is the part of Anthropic cache writes made with a 1-hour TTL.

Costs are estimates at list prices from the model registry, computed for what the call
actually used:

- **Service tier.** The tier the response reports wins over the one requested (OpenAI
  echoes `service_tier`, for example `default` when a Fast request was downgraded;
  Anthropic reports `speed: "fast"` and `service_tier`).
  `GenerationConfig::with_openai_service_tier` requests OpenAI Flex or Fast (sent as
  `priority`) on models that offer it, with API-key auth only. Tier prices are per model
  (`ModelPricing::service_tiers`); tiers without a published price (Anthropic priority,
  OpenAI scale) are estimated at standard rates.
- **Cache TTL, residency and dates.** Anthropic 1-hour cache writes use their own rate;
  `inference_geo: "us"` applies the 1.1x multiplier on Claude 4.6 and later;
  `ModelPricing::schedule` switches prices on a date (for example the end of promotional
  pricing). Tier and residency multipliers stack.
- **Missing prices.** A token class without its own price (cached input on Pro models,
  cache writes on models that don't charge extra for them) is billed at the input rate,
  including long-context rates.

`total_cost` is the estimate; `billed()` is what the account is charged, decided by
`billing`: `Metered` (the estimate), `Subscription` (ChatGPT OAuth: zero, while the
estimate stays visible), `NotCharged` (zero: an Anthropic classifier refusal before any
output in a category Anthropic does not bill) or `ProviderReported` (`reported_cost` plus
any bring-your-own-key `upstream_cost`, used by providers that report their actual
charge, such as [OpenRouter](#usage-and-cost-1)). `UsageInfo` implements `AddAssign` and
`Sum`; summing calls billed differently yields `BillingKind::Mixed` with the billed
amounts added. Workflow budgets count `billed()`.

## Errors and retries

`Error::retry_class()` returns `Permanent`, `Transient`, `Capacity` or `UsageLimit` (a
plan usage limit such as the ChatGPT backend's `usage_limit_reached`, retryable only once
it resets; `retry_after()` holds the time left). `is_retryable()`, `is_capacity_error()`
and `is_usage_limit()` are views of it, and `is_rate_limit()` picks the short rate limits
out of `Transient` (HTTP 429, `rate_limit_exceeded`, `slow_down`).

Provider error bodies are read generically: string codes (`error.code`, `error.type`,
`error.status`), OpenRouter's `error.metadata.error_type`, and numeric HTTP-style codes,
which stand in for the status on stream errors that arrive after a 200. Documented string
codes stand in for it too (Gemini Interactions codes such as `service_unavailable` or
`deadline_exceeded`, Anthropic error types such as `api_error` or `timeout_error`).
Permanent codes (authentication, quota such as `insufficient_quota`,
`organization_usage_limit_exceeded` or Gemini's daily `quota_exceeded`, the ChatGPT
backend's `flex_unavailable`, invalid requests, context length, content policy, ...)
always win, even on a 429; any other error reported after a 200 is `Transient`, so
callers retry it a bounded number of times.

Delay hints come from `Retry-After`, `x-ratelimit-reset-*` headers (on a 429 only, for
the exhausted limit), or body fields (`retry_after_ms`, `retry_after`, `resets_at`,
OpenRouter's `error.metadata.headers.X-RateLimit-Reset`) for every provider; a rate
limit without any of these takes the wait its message names ("Please try again in 2.5s",
read as Codex reads it). All are capped at `retry::MAX_SERVER_DELAY_HINT` (30 days), so a
hint can always be added to the current time.

arko_ai does not retry model calls itself. `arko_ai::retry::BackoffPolicy` is the shared
backoff used by arko_agents, arko_index and the OAuth refresh retries: exponential with a
cap and equal jitter, treating a server hint as a minimum with jitter above it.

## Long requests

The shared model HTTP client has no total request deadline, since a response can stream
for a long time; connection setup (30 seconds) and idle pool connections are bounded
separately, and a request is cancelled by dropping its future. It follows no redirects: a
3xx response is returned as an error, so API keys and other credential headers never
reach another host.

There is no timeout on a response while its connection stays open, and this is
deliberate. Models can reason silently for a long time: a stream may carry no event
until the reasoning ends, and GPT Pro models especially can think for over an hour. A
short timeout cuts healthy responses off, and since the failure is retryable, a caller
that retries sends them again and pays for the work again.

`LLMClient::with_stream_idle_timeout(duration)` opts in to a stream idle timeout on every
provider: a streamed response that receives no data (events, keep-alive comments or
pings) for `duration` ends with `Error::StreamTruncated`, which is transient, so existing
retry logic applies. If you enable it, make it generous, well above the longest silent
stretch your models can have; for example `Duration::from_secs(2 * 60 * 60)`. It counts
only the time between data on a streamed response once its body has started: there is no
total deadline, the wait for the response's status and headers is not covered, and
responses that don't stream (`generate_text`, except over ChatGPT OAuth, whose backend
always streams) are unaffected. The provider clients take it too
(`with_stream_idle_timeout(Some(duration))` on `OpenAIClient`, `AnthropicClient`,
`GoogleClient` and `OpenRouterClient`). The setting is never sent to the model.

## Voyage embeddings

`EmbeddingClient` embeds text with Voyage (`with_voyage_api_key`, or `from_env`, which
reads `VOYAGE_API_KEY`):

```rust
use arko_ai::{EmbeddingClient, EmbeddingConfig, EmbeddingProviderIdentifier, VoyageInputType};

let client = EmbeddingClient::from_env();
let config = EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-code-4")
    .with_voyage_input_type(VoyageInputType::Document)
    .with_output_dimensionality(1024);
let response = client.embed(vec!["fn main() {}".into()], &config).await?;
```

The current text embedding models support 32,000 tokens per input, up to 1,000 inputs per
request, and 256, 512, 1024 (default), or 2048 output dimensions:

| Model ID | Total tokens per request | Standard USD per million tokens |
| --- | ---: | ---: |
| `voyage-4-large` | 120,000 | $0.12 |
| `voyage-4` | 320,000 | $0.06 |
| `voyage-4-lite` | 1,000,000 | $0.02 |
| `voyage-code-4` | Not published | $0.12 |

`voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large`, and `voyage-code-3` are also accepted.
For `voyage-code-4`, `max_batch_total_tokens` is `None`: the provider validates the
aggregate limit, and the per-input and item-count limits still apply locally.
Use `VoyageInputType::Query` when embedding a search query. arko never truncates
silently: `truncation: false` is sent unless `with_voyage_truncation(true)` asks for Voyage's
truncation, and either way an input over the per-input limit fails locally with
`Error::TokenLimitExceeded` before any request, so none reaches Voyage to be truncated;
split long inputs first. Cost estimates use standard endpoint rates,
before account credits or discounts.

Requests are checked against these limits before sending. Tokens are counted with the
matching `voyageai/<model_id>` tokenizer from Hugging Face, on a blocking worker; the
tokenizer is cached per repository and revision (`with_tokenizer_revision` pins a
revision; `with_hf_token` gives a token for gated tokenizers, which is never serialized
with the config, so set it again after loading one), with truncation and padding disabled, and concurrent first loads share one
initialization. The first load downloads the tokenizer from the Hugging Face Hub
(huggingface.co) into the Hugging Face cache (`HF_HOME`, by default
`~/.cache/huggingface`); later loads, in this and later runs, read the cached copy with no
network request. Where the Hub is unreachable, set `HF_ENDPOINT` to a mirror or copy the
cache from a machine that has the tokenizer. A failed download is an
`Error::TokenizerError` that says so and names these settings; a repository or revision
the Hub doesn't have (404) is one naming them, without that advice (other load failures,
such as a damaged cached file, are reported as they are), and nothing is sent to Voyage; with
no home directory and no `HF_HOME` it is an error too. Responses are reordered by the
provider's item indexes, and missing, duplicate, wrong-dimension, non-finite, or zero
vectors are rejected. Provider failures keep the HTTP status and `Retry-After` for
retries by the caller.

Sources: Voyage's [model guide](https://docs.voyageai.com/docs/embeddings),
[API reference](https://docs.voyageai.com/reference/embeddings-api),
[pricing](https://docs.voyageai.com/docs/pricing), and
[tokenizer documentation](https://docs.voyageai.com/docs/tokenization).

## OpenRouter

`Provider::OpenRouter` talks to OpenRouter's native Chat Completions API
(`POST https://openrouter.ai/api/v1/chat/completions`) with its routing, caching and
reasoning extensions. It does not use OpenRouter's Responses or Anthropic Messages
compatibility APIs.

```rust
use arko_ai::{GenerationConfig, LLMClient, Model, ReasoningConfig, ReasoningEffort};

let client = LLMClient::from_env()?; // reads OPENROUTER_API_KEY among others
let mut conversation = client.new_conversation(Model::openrouter("anthropic/claude-opus-5.5"));
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

`Model::openrouter(slug)` accepts any slug (`anthropic/claude-opus-5.5`,
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
| `prompt_cache_options: {mode: "explicit"}` | caching is disabled on an OpenAI model that bills cache writes (GPT-5.6 and later): known from the catalog, or without one from the registry model of the same id | automatic caching |
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
variants are dropped with a warning, and nothing is converted implicitly;
`ReasoningConfig::to_openrouter()` converts them explicitly (OpenAI effort and summaries,
Anthropic budgets or adaptive effort, Gemini budgets and levels) and fails where there is
no equivalent. On Claude, a reasoning `max_tokens` budget must stay
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
a catalog alias's target): no forced tool on Claude Fable 5.1, Opus 5.5 and Sonnet 5.5, nor with a
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
`anthropic_fast_mode` and `anthropic_inference_geo` have no OpenRouter equivalent sent by
arko: they are dropped with a warning (logged once per setting); use `OpenRouterReasoning`
and `OpenRouterOptions` instead. On Claude models, `anthropic_cache_ttl` and
`interleaved_thinking: Some(true)` are sent (see [What is sent](#what-is-sent)),
`interleaved_thinking: Some(false)` is an error because OpenRouter can't turn it off, and
`anthropic_summarized_thinking` and `long_context` are accepted as no-ops: OpenRouter
already returns summarized Claude thinking and enables extended context where the model
has it. On other models these four Claude settings are dropped with a warning too.

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
  OPENROUTER_API_KEY=... cargo test -p arko_ai openrouter::live_tests -- --ignored --nocapture --test-threads=1
  ```

## Claude Code

`arko_ai::claude_code` drives the user's installed Claude Code CLI over its stream-json
control protocol, the protocol Anthropic's Python and TypeScript Agent SDKs use. There is
no official Rust Agent SDK; this module speaks the protocol directly, so it needs neither
Node nor Python. It is the transport of `Provider::ClaudeCode` models. Claude Code runs
its own agent loop, so these models take no `LLMClient` requests (`LLMClient` returns
`Error::ModelNotSupported`); arko_agents runs an agent whose model is one of them in a
Claude Code session (see its README's *Claude Code models*).

The CLI runs unmodified, under whatever login the user already has (or an API key in its
environment). This module never reads, stores, or refreshes credentials and makes no
requests of its own; Claude Code makes the model calls. Plan limits assume ordinary,
individual use, so shared or production automation should run Claude Code with an API
key.

```rust,ignore
use std::sync::Arc;
use arko_ai::claude_code::*;

let mut session = ClaudeSession::start(ClaudeOptions {
    model: Some("claude-opus-5-5".into()),
    effort: Some(Effort::High),
    permission_handler: Some(Arc::new(my_approvals)),
    tool_servers: vec![Arc::new(ToolServer::new("arko").tool(Arc::new(my_tool)))],
    setting_sources: Some(Vec::new()),
    ..ClaudeOptions::default()
})
.await?;
session.send("Summarize the open TODOs in this repository.").await?;
let turn = session.receive_turn().await?;
println!("{:?}", turn.result.result);
let id = session.session_id();
session.close().await?;
```

| Capability | How |
| --- | --- |
| Turns | One process per conversation; `send` starts a turn, `receive_turn` collects it up to its `Result` |
| Model and effort | `--model` and `--effort` per process; `set_model` switches the model mid-session. To change effort, start a new session with `SessionStart::Resume` |
| Conversation history | Claude Code keeps the transcript; hold on to `session_id()` and resume with `SessionStart::Resume { session, fork }`. `SessionStart::WithId` names a new session |
| Approvals | `permission_handler` answers the CLI's `can_use_tool` requests (`--permission-prompt-tool stdio`) |
| Hooks | `hooks` registers host callbacks for events such as `PreToolUse`; `Hook::deny_tool` blocks a call |
| Host tools | `tool_servers` are served in-process as MCP servers over the same pipes; the model sees `mcp__<server>__<tool>` |
| Settings | `settings` (`--settings`, for example `{"autoCompactEnabled": false}` or `{"promptCacheTtl": "1h"}`; unset, Claude Code caches the main conversation for 1 hour on a subscription within its limits and 5 minutes on an API key, and reports each request's split in `usage.cache_creation`) and `setting_sources` (`--setting-sources`; `Some(vec![])` loads none of the user's settings files) |
| Control | `interrupt`, `set_model`, `set_permission_mode` |
| Messages | `Message` variants keep their raw JSON, so message types this client does not model are still available. Usage follows Anthropic's exclusive counts: `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens` add up to a request's whole input. An `assistant` message's output count is the one when its stream started (one response can arrive as several messages with the same id), and the turn's `result` sums the turn's requests; with `include_partial_messages`, each request's `message_delta` stream event has its final usage |

On Windows, npm installs Claude Code as a `claude.cmd` launcher script, which can't be
run without a shell. The client follows it to the package's `bin/claude.exe`, or to
`cli.js` run with `node` for older releases, as T3 Code does; native installs are used as
found. A program running inside a Claude Code session inherits variables that describe
that session (`CLAUDE_CODE_SESSION_ID`, its messaging socket, its effort); the client
removes them, so the CLI it starts runs as its own session.

## Examples and tests

The `llm_client_cli` example sends one prompt or runs an interactive chat with any
configured provider, reading credentials with `LLMClient::from_env()` (and a `.env`
file). `--codex-auth` takes the OpenAI credentials from the Codex `auth.json` instead
(`--codex-auth-json PATH` from another file), through `with_codex_auth_file`; the
`OPENAI_OAUTH_*` variables are then not read:

```bash
cargo run -p arko_ai --example llm_client_cli -- --help
cargo run -p arko_ai --example llm_client_cli -- text -m claude-opus-5-5 -p "Hello"
cargo run -p arko_ai --example llm_client_cli -- text -m gpt-6-astra -p "Hello" --codex-auth
cargo run -p arko_ai --example llm_client_cli -- chat -m anthropic/claude-opus-5.5 --reasoning-level high
```

`--reasoning` reasons at the model's defaults: no effort or level is sent. On OpenAI that
means the model's default effort (none on GPT-5.1, 5.2 and 5.4, which then don't reason)
and no reasoning summaries; pass `--reasoning-level` for an effort and summaries. Claude
models that need a manual thinking budget get half of `--max-tokens` (or of the model's
output limit), between 1,024 and 16,000 tokens. `--reasoning-level` and
`--reasoning-effort` set the effort, level or budget explicitly.

`claude_code_live` runs a short conversation against the installed Claude Code CLI (a
tool call, a hook, a second turn, and a resume), and `claude_code_delivery_probe`
measures when a message sent mid-turn reaches the model. Both use the CLI's login and
count against its usage. `RUST_LOG=arko_ai::claude_code::wire=trace` logs the raw
stream-json traffic of any Claude Code session, one line per message (`->` to the CLI,
`<-` from it); `claude_code::WIRE_LOG` names the target.

`embeddings_demo` requests Voyage embeddings with `VOYAGE_API_KEY`, and
`oauth_cache_probe` measures prompt-cache hits on the ChatGPT OAuth path with the
credentials in a Codex `auth.json`; with `--idle-gap` it measures how long an idle
conversation's cache lasts.

The unit tests use local fixtures and make no provider calls; the Claude Code tests run
a fake CLI (`tests/fixtures/fake_claude.py`) and need Python 3 (`ARKO_TEST_PYTHON`
selects the interpreter):

```bash
cargo test -p arko_ai --lib
```

Tests that need the network (a Hugging Face tokenizer download, live OpenRouter
requests) are marked `#[ignore]`; see [Known gaps](#known-gaps) for the OpenRouter suite.

## License

MIT OR Apache-2.0
