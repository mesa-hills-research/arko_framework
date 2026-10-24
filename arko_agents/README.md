# arko_agents

arko_agents runs tool-using agents on [arko_ai](../arko_ai/README.md). An `AgentConfig`
describes an agent with its role and instructions, model, tools, and approval and error
policies. A `Task` holds one run's conversation and state. `AgentRuntime` runs a task, streams
`AgentEvent`s and takes `UserCommand`s for approvals, answers, error recovery, context
updates, cancellation and suspension.

- Built-in tools for files (`read_file`, `write_file`, `edit_file`, `apply_diff`,
  `list_files`), for talking to the user (`ask_user`, `message_user`) and for finishing
  (`complete_task`)
- Custom tools, MCP servers and skills
- Interactive terminals in a PTY or with pipes, with background jobs, input requests,
  monitors that alert the agent, and a guard against destructive commands
- Native tool calls, or tool calls written in the model's output for models without them
- Approvals, workspace path rules, tool time limits and limits on tool result size
- Recovery from failed model calls, refusals and requests over the model's input limit
- Conversation handover, which continues a long task in a fresh conversation that starts
  from the agent's own report
- Snapshots of running tasks, and saved tasks that can be resumed
- Claude models run through an installed Claude Code CLI

## Install

```toml
[dependencies]
arko_agents = "0.2"
arko_ai = "0.2"
tokio = { version = "1", features = ["full"] }
```

## Example

```rust
use arko_agents::{
    AgentConfig, AgentEvent, AgentLLMConfig, AgentRuntime, AgentToolConfig, Task, TaskState,
    ToolCallingMode, ToolRegistry, UserCommand,
};
use arko_ai::{GenerationConfig, LLMClient, Model};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm_config = AgentLLMConfig {
        model: Model::anthropic_claude_opus_5_5(),
        generation_config: GenerationConfig::default(),
    };
    let config = AgentConfig::builder("writer".into(), "Writer".into(), llm_config)
        .role_description("You are a careful software engineer.".into())
        .instructions("Make the requested change, then call complete_task with a summary.".into())
        .tool_config(AgentToolConfig {
            available_tool_names: vec![
                "list_files".into(),
                "read_file".into(),
                "write_file".into(),
                "complete_task".into(),
            ],
            tool_calling_mode: ToolCallingMode::Native,
            ..AgentToolConfig::default()
        })
        .build()?;

    // Configures every provider whose credentials are in the environment.
    let client = Arc::new(LLMClient::from_env()?);
    let workspace = std::fs::canonicalize("./project")?;
    let task = Task::new(config, "Add a README that explains how to build the project.".into(), workspace, None);

    let (mut events, commands, join) = AgentRuntime::new(client).run_task(
        task,
        Arc::new(ToolRegistry::new()),
        None, // event throttle
        true, // simplified tool events
        None, // skills
        None, // tool-result interceptor
        None, // tool execution middleware
    );

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::AssistantResponseChunk(text) => print!("{text}"),
            AgentEvent::ToolCallAwaitingApproval { tool_call_id, tool_name, parameters, .. } => {
                println!("\nApproving {tool_name} with {parameters:?}");
                commands
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await?;
            }
            AgentEvent::ErrorEncountered { error_message, requires_approval_to_retry: true, .. } => {
                eprintln!("\n{error_message}");
                commands.send(UserCommand::RejectErrorRecovery { message: None }).await?;
            }
            AgentEvent::TaskStreamEnd => break,
            _ => {}
        }
    }

    let task = join.await??;
    match &task.current_state {
        TaskState::Completed { summary } => println!("\nCompleted with {summary}"),
        state => println!("\nStopped in {state:?}"),
    }
    Ok(())
}
```

The task's working directory is the agent's workspace. File tools resolve relative paths
against it and reject paths outside it by default. The guide's
[Running tasks](../docs/arko_agents-guide.md#running-tasks) explains the event and command
channels.

The `arko_agents_cli` example runs one agent from a JSON configuration, with interactive
approvals, answers and error recovery:

```bash
cargo run -p arko_agents --example arko_agents_cli -- \
  --agent-config /path/to/agent.json \
  --workspace-dir /path/to/workspace --prompt "Add a README"
```

## Documentation

- The [arko_agents guide](../docs/arko_agents-guide.md) covers agent configuration, tools and
  approvals, running tasks, snapshots and recovery, monitoring, Claude Code models,
  conversation handover and refusals.
- [Interactive terminals](../docs/interactive-terminals.md) describes the terminal tools, their
  background jobs, monitors and the command guard.
- [Context management](../docs/context-efficiency.md) describes file-tree change notices,
  tool-result distillation and background shell jobs.
- The [reference map](../docs/reference-map.md#arko_agents) lists every public type with its
  defaults and the file that defines it.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT), at your option.
