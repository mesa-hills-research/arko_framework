# Arko Agents

`arko-agents` is a Rust library for building and running LLM-powered agents
capable of interacting with their environment through a flexible system of tools.
It builds upon the `arko-ai` crate for LLM interactions.

## Features

Opt-in [interactive background terminals](../docs/interactive-terminals.md) provide
PTY/pipe sessions, correlated program input requests, optional prompt matching,
bounded output, and automatic runtime notifications through four terminal tools.

File-tree context defaults to a full first snapshot followed by changes at each
model-call boundary. Unchanged trees add no message. Configure `file_tree_mode`
in `EnvironmentDetailsConfig`; see [context management](../docs/context-efficiency.md)
for refresh behavior, full mode, and move detection.

*   Define agents with specific roles, instructions, and LLM configurations.
*   Equip agents with tools:
    *   Built-in tools (e.g., task completion).
    *   Developer-defined tools (custom Rust logic).
    *   End-user defined tools (no-code shell command mapping).
*   Runtime configuration of agents and end-user tools via JSON.
*   Human-in-the-loop paradigm for tool usage and error recovery (configurable).
*   Streaming of agent events for dynamic UIs.
*   Customizable tool invocation syntax parsing.
*   Environment details (CWD, file trees, time, OS) injectable into prompts.

## Getting Started

Run the included CLI example:

```bash
cargo run -p arko-agents --example arko_agents_cli -- --help
```

## Model and token-refresh recovery

`LLMErrorPolicy` controls attempt limits and whether recovery pauses for approval.
`LLMRetryConfig` controls timing; it is available on `AgentConfig` and its builder:

```rust
.llm_error_policy(LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 10 })
.llm_retry_config(LLMRetryConfig {
    initial_delay_ms: 250,
    max_delay_ms: 64_000,
    max_elapsed_ms: 300_000,
    jitter: true,
    capacity_retry: None,
})
```

To wait through explicit provider capacity errors until service recovers, opt in:

```rust
.llm_retry_config(LLMRetryConfig {
    capacity_retry: Some(LLMCapacityRetryConfig::default()),
    ..Default::default()
})
```

Under either `RetryThen*` policy, this bypasses attempt and elapsed limits only for
retryable `server_is_overloaded`, `overloaded_error`, and
`service_unavailable_error` codes. Capacity backoff starts at 30 seconds and caps
at five minutes, with the same jitter and server-minimum handling. Cancel,
suspend, and channel closure remain interruptible. Capacity retries emit events
without adding repeated errors to the model conversation. Success clears the
capacity episode; another error category begins ordinary bounded recovery, and
permanent errors stop promptly. A new runtime starts a new backoff episode.
`PauseAndAwaitApproval` remains an immediate pause even with this option enabled.
Other applications retain bounded recovery by default. Rust `LLMRetryConfig`
struct literals must supply `capacity_retry` or use `..Default::default()`.

These timing values are the defaults. Old serialized configurations receive them
through serde defaults. Zero timing values and a maximum delay smaller than the
initial delay are rejected. Attempt limits count the initial attempt; zero is
normalized to one attempt.

- HTTP 408, 409, 429, and 5xx failures are eligible for bounded retries. Invalid
  requests, missing provider configuration, authentication failures, and explicit
  permanent provider codes (including quota exhaustion and invalid refresh grants)
  stop automatic recovery. Unknown legacy `ApiError` values retain bounded retries.
  Providers expose status, error code, and optional delay through `ProviderError`.
- `Retry-After` seconds and HTTP dates set a minimum delay. OpenAI also supplies
  reset-header and response-body hints. Server hints may exceed `max_delay_ms`,
  which caps client-generated exponential backoff; they must fit the remaining
  elapsed budget. Arko pauses/returns control if they cannot fit, rather than
  retrying before the server's minimum.
- Equal jitter randomizes client backoff between half and all of its capped
  exponential delay. Server minimums also receive nonnegative jitter so shared
  reset times do not synchronize agents. Disable jitter for
  deterministic timing when needed.
- The elapsed recovery clock starts at the first failure and includes subsequent
  waits, token-refresh requests, model requests, and streams. A successful model
  turn with valid tool calls clears recovery; token refresh alone and peer context
  messages do not. The initial request remains subject to provider/host timeouts.
- An OAuth request rejected with HTTP 401 `token_expired` triggers one provider
  refresh and one resend, even if the JWT expiry is still in the future.
  Concurrent rejections share rotated credentials. A second rejection, other
  authentication errors, or a permanent refresh error returns to this policy.
  Transient refresh errors also return here without a provider retry loop.
- Proactive OAuth refresh participates in the same attempt and elapsed limits as
  model calls. The runtime explicitly bypasses the legacy provider retry loop.
  With credentials from a Codex `auth.json`, build the client with
  `LLMClient::with_codex_auth_file` so both refresh paths re-read the file first
  and save rotated tokens at once (see arko-ai's README, "Shared Codex
  credentials"); the example CLI's `--openai-auth-json` does.
  `CancelTask`, `SuspendTask`, and command-channel closure interrupt backoff and
  pending requests. Suspending returns the current task for persistence.
- At exhaustion, `RetryThenPauseAndAwaitApproval` waits for explicit recovery or
  suspension. `RetryThenInformAgent` returns an `Interrupted` task to the host;
  it no longer enters an unbounded second retry cycle. `InformAgentAndContinue`
  is also subject to classification and the elapsed limit. Explicit approval or
  a new runtime can begin a new recovery episode.
- Refusals are never retried unchanged: a `RefusalReceived` event carries the
  `arko_ai::Refusal`, then the refusal follows the permanent-error path above.
  A model's own refusal (`RefusalSource::Model`, such as OpenAI structured
  refusals) stays in history with its explanation as the assistant text, so the
  model can see what it declined; its tool calls are not run. Classifier and
  content-filter refusals are removed from history.
- Waits that don't count against the ordinary limits (opt-in `capacity_retry`)
  neither reset nor consume them: ordinary attempts keep counting across
  interleaved capacity failures, and the ordinary elapsed clock stops while such
  a wait lasts.
- Output a failed stream delivered is kept (followed by the error message) when
  recovery adds a message, and dropped when the retry resends the request
  unchanged (capacity waits), so no retried request ends on an assistant turn.
  A request whose conversation still ends on assistant text (for example a task
  resumed after `SuspendTask` mid-stream) first gets a user message asking the
  model to continue; Claude 4.6 and later reject prefill.
- Calls a native stream announced (`ToolCallInitiated`) but that never run,
  because the stream failed, was interrupted or was refused, get a
  `ToolRejected` event.
- Tasks and conversations loaded from JSON take the registry's current data
  (pricing, limits) for their model, as `AgentLLMConfig` does; unknown ids and
  OpenRouter slugs keep their saved copy.
- `RetryScheduled` events expose attempt, delay, and elapsed milliseconds.
  Tool executions are not replayed by model recovery. Interrupted streams may
  have incurred provider usage without delivering final usage metadata.

Compatibility: HTTP failures now use `arko_ai::Error::ProviderError` instead of
string-only API/authentication variants. Update exhaustive error/event matches
for `ProviderError` and `RetryScheduled`. Rust `AgentConfig` struct literals need
the new `llm_retry_config` field; existing builder calls and saved JSON keep
working through defaults.

Tests use virtual time and synthetic errors; the refresh transport test uses only
a local HTTP fixture and fake credentials. Run `cargo test --offline --locked -p
arko-agents -p arko-ai -p arko-workflows --lib` from the framework workspace.
Status classification follows [OpenAI's error guidance](https://developers.openai.com/api/docs/guides/error-codes)
and [documented retry categories](https://developers.openai.com/api/reference/ruby).

## License

Licensed under the MIT license.
