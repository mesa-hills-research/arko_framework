# Arko Framework

Rust crates for calling language models, running tool-using agents, orchestrating
multi-agent workflows, and indexing source code for retrieval.

## Crates

| Crate | Purpose |
| --- | --- |
| [arko_ai](arko_ai/README.md) | Model clients for OpenAI (API key or ChatGPT OAuth), Anthropic, Google Gemini, and OpenRouter, plus Voyage embeddings. Streaming, tool calls, reasoning controls, and usage and cost reporting. |
| [arko_agents](arko_agents/README.md) | Agent configuration and runtime, built-in and custom tools, skills, MCP servers, and interactive terminal tools. |
| [arko_workflows](arko_workflows/README.md) | Multi-agent workflows with shared memory, messaging, budgets, hooks, completion verification, and persistence. Optional Python bindings behind the `python` feature. |
| [arko_index](arko_index/README.md) | Deterministic tree-sitter chunking of source repositories, offline BM25 code search, and durable watchers that report chunk deltas. No network or model calls. |

arko_agents builds on arko_ai; arko_workflows builds on both; arko_index depends on neither.

## Build and test

Building requires Rust 1.89 or later and a C compiler (arko_index alone needs none).

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --lib --locked
cargo test -p arko_index --locked
```

Each crate's README starts with a minimal example, and each crate's `examples/`
directory contains runnable programs.

## Guides

- [Defaults reference](docs/reference-map.md): every setting's default across the
  four crates, the opt-ins, and the behaviours to know before configuring them.
- [Prompting guides](docs/prompting/README.md): model-specific guidance for Opus 5.5,
  Fable 5.1, and Astra 6.
- [General agent design guide](docs/guides/agent-design.md): agent roles, tools,
  prompting, model selection, and coordination.
- [General workflow design guide](docs/guides/workflow-design.md): workflow topology,
  shared state, verification, persistence, and lifecycle design.
- [Context management](docs/context-efficiency.md): versioned memory reads,
  file-tree change notices, tool-result distillation, and background shell jobs.
- [Interactive terminals](docs/interactive-terminals.md): PTY and pipe terminal
  tools with input requests and completion notifications.
- [Asynchronous model features](docs/async-model-features.md): a provider comparison
  of asynchronous tool calling and mid-turn steering.
- [Arko Workflows API index](arko_workflows/INDEX.md): the complete arko_workflows
  API reference.
- [Complete workflow example](arko_workflows/examples/complete_workflow.rs): explicit agent
  configuration, memory, messaging, verification, distillation, persistence, and resume.
- [Arko Index reference](docs/arko_index-reference.md): the chunking, search and
  watcher API, file selection, the example CLI, environment variables and on-disk state.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
