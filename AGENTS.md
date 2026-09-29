# Notes for AI coding agents

- Read [docs/defaults-reference.md](docs/defaults-reference.md) first: every setting's
  default, the opt-ins, and the behaviours to configure explicitly. Then the crate
  READMEs and [arko_workflows/INDEX.md](arko_workflows/INDEX.md) for details and
  examples.
- When designing an agent or workflow, treat the [design guides](docs/guides/README.md)
  as ideas, not rules. Start with the simplest approach that could work and build up
  only when it falls short. Aim for autonomy: add approval gates, human checkpoints, or
  restrictions only for a concrete risk. Workflow runs are expensive, so iterate on a
  workflow with live test runs or benchmarks only when explicitly asked.
- Crates: arko_agents builds on arko_ai, arko_workflows on both, and arko_index on
  arko_ai.
- Building needs Rust 1.89 or later and a C compiler. Run `cargo fmt --all` before
  committing; CI checks formatting.
- Test with `cargo test --workspace --locked`. The tests make no provider calls; the
  terminal and MCP tests need Python 3 (`ARKO_TEST_PYTHON` selects the interpreter).
- Saved data must stay loadable: agent configurations, tasks, conversations, workflow
  checkpoints and index databases written by earlier versions. Give new fields serde
  defaults and keep reading old ones.
