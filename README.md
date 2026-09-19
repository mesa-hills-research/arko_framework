# Arko Framework

A Rust workspace for model clients, tool-using agents, and workflow orchestration.

| Package | Purpose |
| --- | --- |
| [arko-ai](arko-ai/README.md) | Provider clients, streaming, tool calls, reasoning, embeddings, and usage metadata. |
| [arko-agents](arko-agents/README.md) | Agent configuration, runtime, tools, skills, and MCP integration. |
| [arko-index](arko-index/README.md) | Validated source chunking, incremental embeddings, and compact local retrieval. |
| [arko-workflows](arko-workflows/README.md) | Workflow execution, shared state, persistence, budgets, and verification. |

## Build and test

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --lib --locked
cargo test -p arko-index --locked
```

Package and directory names use hyphens. Rust imports use underscores, such as
`arko_ai`, `arko_agents`, `arko_workflows`, and `arko_index`. Optional Python bindings are in
`arko-workflows` behind its `python` feature.

See each package README and its `examples/` directory for usage.

See [context management](docs/context-efficiency.md) for memory diffs, file-tree
change notices, and configurable tool-result distillation.
