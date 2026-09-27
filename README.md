# Arko Framework

Rust crates for calling language models, running tool-using agents, orchestrating
multi-agent workflows, and indexing source code for retrieval.

## Crates

| Crate | Purpose |
| --- | --- |
| [arko-ai](arko-ai/README.md) | Model clients for OpenAI (API key or ChatGPT OAuth), Anthropic, Google Gemini, and OpenRouter, plus Voyage embeddings. Streaming, tool calls, reasoning controls, and usage and cost reporting. |
| [arko-agents](arko-agents/README.md) | Agent configuration and runtime, built-in and custom tools, skills, MCP servers, and interactive terminal tools. |
| [arko-workflows](arko-workflows/README.md) | Multi-agent workflows with shared memory, messaging, budgets, hooks, completion verification, and persistence. Optional Python bindings behind the `python` feature. |
| [arko-index](arko-index/README.md) | Indexing of source code and documentation with model-selected chunk boundaries, Voyage embeddings, and local SQLite hybrid search. |

arko-agents builds on arko-ai; arko-workflows builds on both; arko-index builds on arko-ai.

## Build and test

Building requires Rust 1.89 or later and a C compiler.

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --lib --locked
cargo test -p arko-index --locked
```

Each crate's README starts with a minimal example, and each crate's `examples/`
directory contains runnable programs.

## Guides

- [Context management](docs/context-efficiency.md): versioned memory reads,
  file-tree change notices, tool-result distillation, and background shell jobs.
- [Interactive terminals](docs/interactive-terminals.md): PTY and pipe terminal
  tools with input requests and completion notifications.
- [Asynchronous model features](docs/async-model-features.md): a provider comparison
  of asynchronous tool calling and mid-turn steering.
- [Arko Workflows API index](arko-workflows/INDEX.md): the complete arko-workflows
  API reference.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
