//! Moves an agent that runs in Claude Code to another directory and shows
//! where Claude Code's own tools run afterwards. The session starts in
//! `--from`, then the task moves to `--to`. Its second run asks for the
//! working directory (`Bash`), the text files in the directory (`Glob`), the
//! files that contain "hello" (`Grep`), and the file `note.txt` by its
//! relative name (`Read`). Each should show `--to`.
//!
//! ```text
//! mkdir -p /tmp/move/from /tmp/move/to
//! echo hello from > /tmp/move/from/note.txt
//! echo hello to > /tmp/move/to/note.txt
//! cargo run -p arko_agents --example claude_code_move -- \
//!     --from /tmp/move/from --to /tmp/move/to
//! ```
//!
//! It uses the CLI's login and counts against its plan usage. With
//! `--ask`, every call waits for an approval, which the example gives.

use std::path::PathBuf;
use std::sync::Arc;

use arko_agents::agent_config::{AgentLLMConfig, AgentToolConfig};
use arko_agents::runtime::command::UserCommand;
use arko_agents::runtime::event::AgentEvent;
use arko_agents::{AgentConfig, AgentRuntime, ClaudeCodeSettings, Task, TaskState, ToolRegistry};
use arko_ai::Model;
use clap::Parser;

#[derive(Parser)]
struct Args {
    /// The directory the session starts in.
    #[arg(long)]
    from: PathBuf,
    /// The directory the task moves to.
    #[arg(long)]
    to: PathBuf,
    /// An Anthropic model id, run through Claude Code.
    #[arg(long, default_value = "claude-haiku-5-5")]
    model: String,
    /// Have each call wait for an approval, which the example gives.
    #[arg(long)]
    ask: bool,
}

async fn run(task: Task, show: bool, approve: bool) -> Result<Task, Box<dyn std::error::Error>> {
    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()));
    let (mut events, commands, handle) = runtime.run_task(
        task,
        Arc::new(ToolRegistry::new()),
        None,
        false,
        None,
        None,
        None,
    );
    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } if approve => {
                commands
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await?;
            }
            AgentEvent::ToolCallFormatted {
                tool_name,
                parameters,
                ..
            } if show => println!("→ {tool_name} {}", serde_json::to_string(&parameters)?),
            AgentEvent::ToolResult {
                tool_name, output, ..
            } if show => {
                let shown: String = output.chars().take(300).collect();
                println!("← {tool_name} {}", shown.trim());
            }
            AgentEvent::ToolRejected {
                tool_name, reason, ..
            } => println!("✗ {tool_name} {}", reason.unwrap_or_default()),
            AgentEvent::ErrorEncountered { error_message, .. } => {
                println!("[error] {error_message}")
            }
            _ => {}
        }
    }
    Ok(handle.await??)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let model = Model::claude_code(Model::get(arko_ai::Provider::Anthropic, &args.model)?);
    let config = AgentConfig::builder(
        "move-example".into(),
        "Mover".into(),
        AgentLLMConfig {
            model,
            generation_config: Default::default(),
        },
    )
    .role_description("You report what your tools return.".into())
    .instructions("Make exactly the calls you are asked for, with exactly the input given.".into())
    .tool_config(AgentToolConfig {
        default_auto_approve_tool_use: !args.ask,
        available_tool_names: vec!["Bash".into(), "Glob".into(), "Grep".into(), "Read".into()],
        ..AgentToolConfig::default()
    })
    .claude_code(ClaudeCodeSettings::default())
    .build()?;

    let from = std::fs::canonicalize(&args.from)?;
    let to = std::fs::canonicalize(&args.to)?;
    let task = Task::new(
        config,
        "Say hello, then call complete_task.".into(),
        from.clone(),
        None,
    );
    let mut task = run(task, false, args.ask).await?;
    println!("The session started in {}.", from.display());

    task.move_to(to.clone());
    task.prepare_for_resumption_from_completed(
        "Make these calls in order and then call complete_task. 1. Bash with the command `pwd`. \
         2. Glob with the pattern `*.txt` and no path. 3. Grep with the pattern `hello`, no path \
         and the output mode files_with_matches. 4. Read with the file_path `note.txt` (the \
         relative name)."
            .into(),
    )?;
    println!("The task moved to {}.", to.display());
    let task = run(task, true, args.ask).await?;
    println!(
        "Final state {}",
        match task.current_state {
            TaskState::Completed { .. } => "completed".to_string(),
            other => format!("{other:?}"),
        }
    );
    Ok(())
}
