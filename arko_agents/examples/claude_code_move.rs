//! Moves an agent that runs in Claude Code to another directory and shows
//! where its tools run afterwards. The session starts in `--from`, then the
//! task moves to `--to`. Its second run asks for the working directory
//! (`Bash`), the text files in the directory (`Glob`), the files that contain
//! "hello" (`Grep`), and the file `note.txt` by its relative name (`Read`),
//! and on Windows for `Get-Location` (`PowerShell`). Each should show `--to`.
//!
//! The framework's own shell tool (`terminal_start`) then prints the working
//! directory of the moved task, without a model. On Windows it runs once
//! with cmd.exe and once with PowerShell, and elsewhere with `/bin/sh`.
//!
//! ```text
//! mkdir -p /tmp/move/from /tmp/move/to
//! echo hello from > /tmp/move/from/note.txt
//! echo hello to > /tmp/move/to/note.txt
//! cargo run -p arko_agents --example claude_code_move -- \
//!     --from /tmp/move/from --to /tmp/move/to
//! ```
//!
//! On Windows, run this in PowerShell from the repository. The `Bash` call
//! needs Git for Windows.
//!
//! ```text
//! $move = "$HOME\claude-code-move"
//! New-Item -ItemType Directory -Force "$move\from", "$move\to" | Out-Null
//! Set-Content "$move\from\note.txt" 'hello from'
//! Set-Content "$move\to\note.txt" 'hello to'
//! cargo run -p arko_agents --example claude_code_move -- --from "$move\from" --to "$move\to"
//! ```
//!
//! Every `←` line and both `terminal_start` lines should name the `to`
//! folder. `Read` shows `hello to`, and Git Bash writes the folder as
//! `/c/Users/...`. A line that names the `from` folder is a tool that ran in
//! the old directory.
//!
//! It uses the CLI's login and counts against its plan usage. With
//! `--ask`, every call waits for an approval, which the example gives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arko_agents::agent_config::{AgentLLMConfig, AgentToolConfig};
use arko_agents::runtime::command::UserCommand;
use arko_agents::runtime::event::AgentEvent;
use arko_agents::terminal::{TerminalConfig, TerminalManager, TerminalSnapshot, TerminalTool};
use arko_agents::tool::Tool;
use arko_agents::{
    AgentConfig, AgentRuntime, ClaudeCodeSettings, Task, TaskContext, TaskState, ToolRegistry,
};
use arko_ai::Model;
use clap::Parser;

/// The shells `terminal_start` runs with, each as its name, program,
/// arguments before the command, and a command that prints the directory.
const SHELLS: &[(&str, &str, &[&str], &str)] = if cfg!(windows) {
    &[
        ("cmd.exe", "cmd.exe", &["/D", "/S", "/C"], "cd"),
        (
            "PowerShell",
            "powershell.exe",
            &["-NoLogo", "-NoProfile", "-Command"],
            "(Get-Location).Path",
        ),
    ]
} else {
    &[("sh", "/bin/sh", &["-c"], "pwd")]
};

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

/// The directory's full path. On Windows `canonicalize` gives a `\\?\` path,
/// which cmd.exe doesn't take as its working directory.
fn full_path(path: &Path) -> std::io::Result<PathBuf> {
    if cfg!(windows) {
        std::fs::metadata(path)?;
        std::path::absolute(path)
    } else {
        std::fs::canonicalize(path)
    }
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

/// Calls the framework's `terminal_start` tool for `task`, set up to run
/// jobs with `shell` and `shell_args`, and returns what `command` printed.
async fn terminal_output(
    task: &Task,
    shell: &str,
    shell_args: &[&str],
    command: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let jobs = std::env::temp_dir().join(format!("claude-code-move-{}", uuid::Uuid::new_v4()));
    let mut config = TerminalConfig::new(&jobs);
    config.shell = shell.into();
    config.shell_args = shell_args.iter().map(|arg| arg.to_string()).collect();
    let manager = TerminalManager::new(config)?;
    let start = TerminalTool::all(manager.clone())
        .into_iter()
        .find(|tool| tool.definition().name == TerminalTool::NAMES[0])
        .ok_or("terminal_start is missing")?;
    let context = TaskContext::new(
        task.id,
        &task.agent_config.name,
        &task.initial_task_prompt,
        &task.current_working_directory,
    );
    let parameters = HashMap::from([
        ("command".to_string(), serde_json::json!(command)),
        ("terminal".to_string(), serde_json::json!(false)),
    ]);
    let mut snapshot: TerminalSnapshot =
        serde_json::from_str(&start.execute(&context, parameters).await?)?;
    while !snapshot.state.is_terminal() {
        snapshot = manager
            .wait(
                task.id,
                snapshot.job_id,
                snapshot.revision,
                Duration::from_secs(5),
            )
            .await?;
    }
    manager.shutdown().await;
    drop(manager);
    let _ = std::fs::remove_dir_all(&jobs);
    Ok(snapshot.output.text.trim().to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let model = Model::claude_code(Model::get(arko_ai::Provider::Anthropic, &args.model)?);
    let mut tools: Vec<String> = ["Bash", "Glob", "Grep", "Read"].map(String::from).to_vec();
    if cfg!(windows) {
        tools.push("PowerShell".into());
    }
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
        available_tool_names: tools,
        ..AgentToolConfig::default()
    })
    .claude_code(ClaudeCodeSettings::default())
    .build()?;

    let from = full_path(&args.from)?;
    let to = full_path(&args.to)?;
    let task = Task::new(
        config,
        "Say hello, then call complete_task.".into(),
        from.clone(),
        None,
    );
    let mut task = run(task, false, args.ask).await?;
    println!("The session started in {}.", from.display());

    task.move_to(to.clone());
    let mut calls = "Make these calls in order and then call complete_task. 1. Bash with the \
                     command `pwd`. 2. Glob with the pattern `*.txt` and no path. 3. Grep with \
                     the pattern `hello`, no path and the output mode files_with_matches. 4. \
                     Read with the file_path `note.txt` (the relative name)."
        .to_string();
    if cfg!(windows) {
        calls.push_str(" 5. PowerShell with the command `Get-Location`.");
    }
    task.prepare_for_resumption_from_completed(calls)?;
    println!("The task moved to {}.", to.display());
    let task = run(task, true, args.ask).await?;
    println!(
        "Final state {}",
        match &task.current_state {
            TaskState::Completed { .. } => "completed".to_string(),
            other => format!("{other:?}"),
        }
    );
    for (name, shell, shell_args, command) in SHELLS {
        let output = terminal_output(&task, shell, shell_args, command).await?;
        println!("terminal_start with {name}: {output}");
    }
    Ok(())
}
