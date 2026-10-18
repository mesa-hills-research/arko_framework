# arko_ai

arko_ai is a Rust client for language models from OpenAI, Anthropic, Google Gemini and
OpenRouter, for Claude models run through an installed Claude Code CLI, and for Voyage text
embeddings. One `LLMClient` sends a `Conversation` to the provider of its model and returns
the same response type for every provider, with text and reasoning, tool calls, stop reasons
and refusals, token usage and cost.

- OpenAI with an API key or a ChatGPT sign-in (OAuth, also from a Codex `auth.json`),
  Anthropic, Google Gemini and OpenRouter
- Streaming, native tool calls, images in user messages and tool results, and reasoning
  settings for each provider
- Prompt caching on by default, with each provider's reasoning replayed in later turns
- A model registry with context limits and prices, and the cost of every call
- Errors sorted into kinds (rate limits, capacity, authentication, context overflow and more)
  with retry hints
- Claude Code as a provider, with in-process tools, hooks and resumable sessions
- Voyage embeddings, with tokens counted locally

## Install

```toml
[dependencies]
arko_ai = "0.1.0"
tokio = { version = "1", features = ["full"] }
futures = "0.3"    # for consuming response streams
```

Building needs Rust 1.89 or later and a C compiler, for ring and for Oniguruma (through
tokenizers).

## Example

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

This example needs `ANTHROPIC_API_KEY`. The guide's
[Providers and credentials](../docs/arko_ai-guide.md#providers-and-credentials) lists the
variables of every provider and the builder methods that take credentials directly.

The `llm_client_cli` example is a command-line chat with any configured provider:

```bash
cargo run -p arko_ai --example llm_client_cli -- models
```

## Documentation

- The [arko_ai guide](../docs/arko_ai-guide.md) covers providers and credentials, models,
  generation settings and tool calls, images, stop reasons and refusals, ChatGPT OAuth, usage
  and cost, errors and retries, Voyage embeddings, OpenRouter and Claude Code.
- The [reference map](../docs/reference-map.md#arko_ai) lists every public type with its
  defaults and the file that defines it.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT), at your option.
