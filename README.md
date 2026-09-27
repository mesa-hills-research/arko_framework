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

## Native (C) dependencies

Building the workspace compiles C code from three crates. A C compiler is required;
no system libraries need to be installed.

| Crate | Pulled in by | Why | Pure-Rust alternative and its cost |
| --- | --- | --- | --- |
| `ring` | `rustls`, used by `reqwest` in arko-ai and by `hf-hub`/`ureq` for tokenizer downloads | TLS cryptography | A pure-Rust rustls crypto provider exists (e.g. `rustls-rustcrypto`) but is not audited. |
| `onig_sys` | `tokenizers` (`onig` feature) in arko-ai | Regex engine for Hugging Face tokenizer pre-tokenizers | `tokenizers` exposes `fancy-regex` only through its `unstable_wasm` feature, which is not meant for native builds. |
| `libsqlite3-sys` | `rusqlite` (`bundled` feature) in arko-index | SQLite, compiled from source | Dropping `bundled` links a system SQLite instead (still C, and requires it on every machine); a pure-Rust store would mean rewriting arko-index storage. |

`iana-time-zone-haiku` (via `chrono`) also compiles C, but only when targeting Haiku.

TLS deliberately avoids `reqwest`'s default `aws-lc-rs` backend (another large C
build) and OpenSSL: arko-ai uses rustls with the `ring` provider and the operating
system's trust store. CI fails if `aws-lc-sys`, `aws-lc-rs`, or `openssl-sys` enter
the dependency graph.
