# Notes for AI coding agents

- Read [docs/defaults-reference.md](docs/defaults-reference.md) first: every setting's
  default, the opt-ins, and the behaviours to configure explicitly. Then the crate
  READMEs and [arko_workflows/INDEX.md](arko_workflows/INDEX.md) for details and
  examples.
- Crates: arko_agents builds on arko_ai, arko_workflows on both, and arko_index on
  arko_ai.
- Building needs Rust 1.89 or later and a C compiler. Run `cargo fmt --all` before
  committing; CI checks formatting.
- Test with `cargo test --workspace --locked`. The tests make no provider calls; the
  terminal and MCP tests need Python 3 (`ARKO_TEST_PYTHON` selects the interpreter).
- Saved data must stay loadable: agent configurations, tasks, conversations, workflow
  checkpoints and index databases written by earlier versions. Give new fields serde
  defaults and keep reading old ones.
