# Notes for AI coding agents

- Read [docs/defaults-reference.md](docs/defaults-reference.md) first: every setting's
  default, the opt-ins, and the behaviours to configure explicitly. Then the crate
  READMEs and [arko_workflows/INDEX.md](arko_workflows/INDEX.md) for details and
  examples.
- A workflow built with arko_workflows is always a binary, never a wrapper around
  arko_agents (such as a configuration run through `arko_agents_cli`) unless that is
  explicitly requested.
- When designing an agent or workflow, treat the [design guides](docs/guides/README.md)
  as ideas, not rules. Start with the simplest approach that could work and build up
  only when it falls short. Aim for autonomy: set-and-forget workflows keep the default
  `ApprovalPolicy::ApproveAll` and `ErrorRecoveryPolicy::Automatic`; don't trade them
  for `Limited`, `ReturnToCaller`, or approval gates out of caution. Workflow runs are
  expensive, so gauge the user's budget before iterating on a workflow with live test
  runs or benchmarks; extensive testing and refinement is fine once they authorize it.
  Either way, start simple and change only what the results show is necessary.
- Crates: arko_agents builds on arko_ai and arko_workflows on both; arko_index stands
  alone (tree-sitter chunking, offline search, watchers).
- Building needs Rust 1.89 or later and a C compiler. Run `cargo fmt --all` before
  committing; CI checks formatting.
- Test with `cargo test --workspace --locked`. The tests make no provider calls; the
  terminal, MCP and Claude Code tests need Python 3 (`ARKO_TEST_PYTHON` selects the
  interpreter).
- Saved data must stay loadable: agent configurations, tasks, conversations and workflow
  checkpoints written by earlier versions. Give new fields serde defaults and keep
  reading old ones. arko_index's on-disk data (index and watcher state, and the
  databases of the earlier, model-chunked arko_index) has no compatibility guarantee
  while the crate is unpublished.
