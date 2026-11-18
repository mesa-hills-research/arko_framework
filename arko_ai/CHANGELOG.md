# Changelog

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
