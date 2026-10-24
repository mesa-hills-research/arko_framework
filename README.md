# Arko Framework

Rust crates for calling language models, running tool-using agents, orchestrating
multi-agent workflows, and searching source code.

| Crate | What it does |
| --- | --- |
| [arko_ai](arko_ai/README.md) | Clients for OpenAI (API key or ChatGPT sign-in), Anthropic, Google Gemini and OpenRouter, Claude models through the Claude Code CLI, and Voyage embeddings. Streaming, tool calls, reasoning settings, usage and cost. |
| [arko_agents](arko_agents/README.md) | An agent runtime with built-in and custom tools, skills, MCP servers, interactive terminals, approvals and recovery from failed model calls. |
| [arko_workflows](arko_workflows/README.md) | Multi-agent workflows in plain Rust, with shared memory, messaging, budgets, hooks, completion verification and persistence. |
| [arko_index](arko_index/README.md) | Splits source repositories into chunks along their syntax, searches them lexically, densely or both, and keeps them current with durable watchers. |

arko_agents builds on arko_ai, and arko_workflows builds on both. arko_index stands alone.

## Install

Add the crates you need:

```toml
[dependencies]
arko_ai = "0.2"
arko_agents = "0.2"
```

Building needs Rust 1.89 or later and a C compiler. As a dependency, arko_index needs no C compiler.
Each crate's README has a first example, and each crate's `examples/` directory has
programs you can run.

## Build and test

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --lib --locked
cargo test -p arko_index --locked
```

The tests make no provider calls. The terminal, MCP and Claude Code tests need Python 3.

## Documentation

- The [reference map](docs/reference-map.md) lists every public type and setting of the four
  crates, with its default and the file that defines it.
- The crate guides cover each crate in depth, in [arko_ai](docs/arko_ai-guide.md),
  [arko_agents](docs/arko_agents-guide.md) and [arko_workflows](docs/arko_workflows-guide.md),
  with the [arko_workflows API index](arko_workflows/INDEX.md) beside them.
- The [arko_index reference](docs/arko_index-reference.md) covers chunking, search, the
  watchers, the CLI and the on-disk state.
- The [agent design guide](docs/guides/agent-design.md) and the
  [workflow design guide](docs/guides/workflow-design.md) collect ideas for roles, tools,
  topology, shared state, verification and persistence.
- [Writing prompts](docs/prompting/README.md) covers tasks, roles and configuration.
- [Context management](docs/context-efficiency.md) explains versioned memory reads, file-tree
  change notices, tool-result distillation and background shell jobs.
- [Interactive terminals](docs/interactive-terminals.md) describes the PTY and pipe terminal
  tools, their input requests and their completion notices.
- [`complete_workflow.rs`](arko_workflows/examples/complete_workflow.rs) is one workflow that
  uses agent configurations, memory, messaging, verification, distillation, persistence and
  resume together.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
