# arko_workflows

arko_workflows runs multi-agent workflows on [arko_agents](../arko_agents/README.md) and
[arko_ai](../arko_ai/README.md). A workflow is ordinary Rust code. The library runs the agents and gives them shared memory, messaging, budgets,
hooks, context-freshness tracking, completion verification, durable steps and persistence,
and your code decides which agents run and in what order.

- Agents run in sequence, in parallel, or in the background as subagents that a lead agent
  starts
- Shared memory tiers, and messages between agents, which can wait for one
- Budgets in dollars, and limits on how much of a subscription plan's usage a run takes
- Completion verification by verifier agents, and distillation of large tool results
- Operator notes and stops for a live run, from the same process or from another one
- Checkpoints and resume, and durable steps for side effects
- PyO3 bindings behind the `python` feature, which still need packaging as a Python module

## Install

```toml
[dependencies]
arko_workflows = "0.1.0"
arko_agents = "0.1.0"
arko_ai = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

## Example

```rust
use arko_workflows::prelude::*;
use arko_agents::AgentConfig;
use arko_ai::LLMClient;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), WorkflowError> {
    // Configures every provider whose credentials are in the environment.
    let client = Arc::new(LLMClient::from_env()?);

    let workflow = Workflow::builder()
        .name("my-workflow")
        .workspace("./project")
        .llm_client(client)
        .agent("planner", AgentConfig::load_from_path("configs/planner.json".as_ref()).unwrap())
        .agent("coder", AgentConfig::load_from_path("configs/coder.json".as_ref()).unwrap())
        .budget(5.0)
        .build()?;

    let plan = workflow.run_agent("planner", "Create an implementation plan").await?;
    let _code = workflow.run_agent("coder", &format!("Implement:\n{}", plan.summary)).await?;

    println!("Done. Cost ${:.4}", workflow.result().total_cost);
    Ok(())
}
```

[`complete_workflow.rs`](examples/complete_workflow.rs) puts the workflow primitives together
in one program, and `--dry-run` runs it without credentials. The
[forge](examples/forge/README.md) designs and writes a workflow crate from a description of
what you want, and checks that it builds and passes its tests.

## Documentation

- The [arko_workflows guide](../docs/arko_workflows-guide.md) covers running and stopping
  agents, shared memory, hooks, budgets, messaging, operator notes, approvals and error
  recovery, verification, background subagents, persistence and the Python bindings.
- The [API index](INDEX.md) lists every public type, signature and default, and the tools
  agents see.
- The [workflow design guide](../docs/guides/workflow-design.md) and the
  [agent design guide](../docs/guides/agent-design.md) collect ideas for designing them.
- The [reference map](../docs/reference-map.md#arko_workflows) lists every public type with
  its defaults and the file that defines it.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT), at your option.
