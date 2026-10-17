# Notes for AI coding agents

- Read [docs/defaults-reference.md](docs/defaults-reference.md) first. It collects the
  defaults, the opt-ins and the behaviours to configure explicitly. Then read the crate
  READMEs, the crate guides in `docs/` and [arko_workflows/INDEX.md](arko_workflows/INDEX.md)
  for details and examples.
- A workflow built with arko_workflows is always a binary. It isn't a wrapper around
  arko_agents (such as a configuration run through `arko_agents_cli`) unless the user asks
  for one.
- When designing an agent or workflow, treat the [design guides](docs/guides/README.md)
  as ideas, not rules. Start with the simplest approach that could work and build up
  only when it falls short. Aim for autonomy. Set-and-forget workflows keep the default
  `ApprovalPolicy::ApproveAll` and `ErrorRecoveryPolicy::Automatic` rather than trading
  them for `Limited`, `ReturnToCaller` or approval gates out of caution. Workflow runs are
  expensive, so gauge the user's budget before iterating on a workflow with live test
  runs or benchmarks. Extensive testing and refinement is fine once they authorize it.
  Either way, start simple and change only what the results show is necessary.
- arko_agents builds on arko_ai, and arko_workflows on both. arko_index stands alone
  (tree-sitter chunking, lexical and dense search, watchers).
- Building needs Rust 1.89 or later and a C compiler, for arko_ai's dependencies (ring,
  and Oniguruma through tokenizers). arko_index as a dependency needs none. Run
  `cargo fmt --all` before committing, because CI checks formatting.
- Test with `cargo test --workspace --locked`. The tests make no provider calls. The
  terminal, MCP and Claude Code tests need Python 3 (`ARKO_TEST_PYTHON` selects the
  interpreter).
- Saved data must stay loadable. Agent configurations, tasks, conversations and workflow
  checkpoints written by earlier versions have to load in later ones, so give new fields
  serde defaults and keep reading old ones. arko_index's on-disk data (index and watcher
  state) has no compatibility guarantee while the crate is unpublished.
