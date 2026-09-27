# Arko Framework

Rust crates for calling language models, running tool-using agents, orchestrating
multi-agent workflows, and indexing source code for retrieval.

## Crates

| Crate | Purpose |
| --- | --- |
| [arko_ai](arko_ai/README.md) | Model clients for OpenAI (API key or ChatGPT OAuth), Anthropic, Google Gemini, and OpenRouter, plus Voyage embeddings. Streaming, tool calls, reasoning controls, and usage and cost reporting. |
| [arko_agents](arko_agents/README.md) | Agent configuration and runtime, built-in and custom tools, skills, MCP servers, and interactive terminal tools. |
| [arko_workflows](arko_workflows/README.md) | Multi-agent workflows with shared memory, messaging, budgets, hooks, completion verification, and persistence. Optional Python bindings behind the `python` feature. |
| [arko_index](arko_index/README.md) | Indexing of source code and documentation with model-selected chunk boundaries, Voyage embeddings, and local SQLite hybrid search. |

arko_agents builds on arko_ai; arko_workflows builds on both; arko_index builds on arko_ai.

## Build and test

Building requires Rust 1.89 or later and a C compiler.

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --lib --locked
cargo test -p arko_index --locked
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
- [Arko Workflows API index](arko_workflows/INDEX.md): the complete arko_workflows
  API reference.
- [Arko Index reference](docs/arko_index-reference.md): file selection, chunk
  validation, jobs and events, search, usage accounting, and the example CLI.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
