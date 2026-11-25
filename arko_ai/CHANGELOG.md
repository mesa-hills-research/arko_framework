# Changelog

## 0.4.0 (2026-11-25)

### Models

- Gemini 4 Neon, with a 1,048,576-token window and 65,536 output tokens per synchronous response.
  It costs $3 per million input tokens and $15 per million output tokens, and $0.15 for cached
  input. It thinks at low, medium, high or xhigh, and at high when a request names no level.
  `Model::supports_gemini_xhigh_thinking` is true for it, as for Hydrogen.

### Antigravity

- `Provider::Antigravity` models run in the installed Antigravity CLI (`agy`, 1.5.14 or later) on
  the user's Google plan. `Model::antigravity(model)` runs a model there with its limits and list
  prices. `Model::list(Provider::Antigravity)` gives Gemini 4 Neon, Argon and Hydrogen, Gemini 3.8
  Flash, Claude Opus 5.5, Claude Sonnet 5.5 and GPT-OSS 120B (medium), and `antigravity_family`
  names the model to select in the CLI. `LLMClient` takes no requests for them, since arko_agents
  runs them.
- The `antigravity` module drives the CLI. `AcpClient` speaks ACP with `agy acp`, starts a session
  or resumes one an earlier process saved in the same profile (`new_session`, `resume_session`),
  and turns off the CLI's auto-update on every launch. `Profile` writes the private profile that
  gives the CLI the host's system prompt and tools in place of its own, and `prompt_digest` tells
  whether a saved session started with the same prompt. `RelayHost` serves the host's tools on
  127.0.0.1 to connections that give the session's token, and `run_relay` is the helper the
  profile starts as the CLI's MCP server. When the CLI falls back to its default agent, the
  profile's hooks (`run_gate`, `run_tripwire`) refuse every tool but the host's and record it
  (`Profile::tripped`).
- `check_sign_in` finds the CLI and checks its version and sign-in. `plan_usage` reads the plan's
  usage windows (`agy -p /usage`), and `plan_usage_report` turns them into a `PlanUsage` of
  `SubscriptionPlan::Antigravity`, with every model group in `PlanUsage::antigravity_groups`.

### Context size

- `UsageInfo::context_tokens` and `ModelCallUsage::context_tokens` hold the conversation's size
  after a call where it differs from the call's input, as for a CLI that makes several requests
  for one prompt. `context_size()` gives it, or the input when it isn't set, and
  `with_context_tokens` sets it. `ModelCallUsage` figures its share of the window from it. A sum
  of records has none, and records saved before it load without it.

### Breaking changes

- `Provider` and `SubscriptionPlan` have an `Antigravity` variant, so an exhaustive `match` on
  either needs an arm for it. `SubscriptionPlan::ALL` holds three plans.
- `UsageInfo`, `ModelCallUsage` and `PlanUsage` have new public fields (`context_tokens`, and
  `antigravity_groups`), so struct literals break.

## 0.3.1 (2026-11-18)

### Prompt caching

- `PromptCacheLifetime::longest_wait` gives the longest an agent may wait before its next request
  and still find its prompt cached. It is the lifetime less a margin of a tenth of it, at least
  `prompt_cache::MIN_CACHE_MARGIN` (30 seconds). That is 4 min 30 s for a 5-minute cache, 27 min
  for 30 minutes and 54 min for an hour. An unknown lifetime, or caching off, gives the 5-minute
  cache's.

## 0.3.0 (2026-11-18)

### Models

- GPT-6.2 Sol and GPT-6.2 Astra, with 922K input of a 1.05M context, 128K output, image input and
  Fast mode at twice the price. They take low, medium, high, xhigh and max effort. Their requests
  send the system prompt as a developer message at the start of the input, as Codex does, since
  GPT-6.2 bills top-level instructions uncached on every request.
- Gemini 4 Carbon, Argon and Hydrogen, with a 1,048,576-token window and 65,536 output tokens per
  synchronous response. Carbon and Argon cost $2 per million input tokens and $10 per million
  output tokens through 2026-11-30, and $4 and $20 from 2026-12-01. Hydrogen costs $1.50 and
  $7.50. They think at low, medium or high, and Hydrogen also at xhigh.
- Over ChatGPT sign-in, the input limit follows each model's window in Codex's catalog. It is
  500,000 tokens for GPT-6.2 Sol, GPT-6.1 Sol and GPT-6.1 Terra, and 272,000 for the others.
  `chatgpt_oauth_max_input_tokens` gives it for a model ID.

### Breaking changes

- `GeminiThinkingLevel` has the new level `XHigh` and is `#[non_exhaustive]`, so a `match` on it
  needs a `_` arm.
