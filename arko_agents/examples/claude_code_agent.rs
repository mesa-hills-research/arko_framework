//! Runs an arko agent in the installed Claude Code CLI, under its login, and
//! prints what happens: tool calls through the gate, usage, handovers and
//! compactions.
//!
//! ```text
//! cargo run -p arko_agents --example claude_code_agent -- \
//!     --workspace arko_ai/src/providers/claude_code --handover-at 20000
//! ```
//!
//! With `--compact`, Claude Code compacts the conversation at that size
//! instead of a handover.
//!
//! It uses the CLI's login and counts against its plan usage. For the raw
//! stream-json traffic (every line to and from the CLI), add
//! `RUST_LOG=arko_ai::claude_code::wire=trace`; `--prompt-file` reads the
//! task from a file, and `--refusal-fallback` sets the model arko switches to
//! for one turn when a refusal stands.

use std::path::PathBuf;
use std::sync::Arc;

use arko_agents::agent_config::{AgentLLMConfig, AgentToolConfig};
use arko_agents::handover::HandoverConfig;
use arko_agents::runtime::event::AgentEvent;
use arko_agents::{
    AgentConfig, AgentRuntime, ClaudeCodeSettings, ClaudeCodeTools, Task, ToolRegistry,
};
use arko_ai::Model;
use clap::Parser;

#[derive(Parser)]
struct Args {
    /// The directory the agent works in.
    #[arg(long)]
    workspace: PathBuf,
    /// The task.
    #[arg(
        long,
        default_value = "Read each .rs file in this directory, one at a time, and note its public items. When you have read them all, call complete_task with the list, file by file."
    )]
    prompt: String,
    /// Read the task from this file instead of `--prompt`.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// An Anthropic model id, run through Claude Code.
    #[arg(long, default_value = "claude-sonnet-5-5")]
    model: String,
    /// Hand over once a request's input reaches this many tokens.
    #[arg(long)]
    handover_at: Option<usize>,
    /// Compact the conversation in Claude Code at that size instead.
    #[arg(long, requires = "handover_at")]
    compact: bool,
    /// Replace Claude Code's system prompt with the agent's.
    #[arg(long)]
    replace_system_prompt: bool,
    /// An Anthropic model id to answer one turn when the model's refusal
    /// stands (Claude Code didn't retry it on its own fallback model).
    #[arg(long)]
    refusal_fallback: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let model = Model::claude_code(Model::get(arko_ai::Provider::Anthropic, &args.model)?);
    let mut builder = AgentConfig::builder(
        "claude-code-example".into(),
        "Reader".into(),
        AgentLLMConfig {
            model,
            generation_config: Default::default(),
        },
    )
    .role_description("You read source files and report what they define.".into())
    .instructions("Work through the files one at a time. Keep notes brief.".into())
    .tool_config(AgentToolConfig {
        default_auto_approve_tool_use: true,
        available_tool_names: ClaudeCodeTools::read_only(),
        ..AgentToolConfig::default()
    })
    .claude_code(ClaudeCodeSettings {
        replace_system_prompt: args.replace_system_prompt,
        ..ClaudeCodeSettings::default()
    });
    if let Some(tokens) = args.handover_at {
        builder = builder.handover(HandoverConfig {
            compact_in_claude_code: args.compact,
            ..HandoverConfig::at_input_tokens(tokens)
        });
    }
    if let Some(fallback) = &args.refusal_fallback {
        builder = builder.refusal_fallback(AgentLLMConfig {
            model: Model::claude_code(Model::get(arko_ai::Provider::Anthropic, fallback)?),
            generation_config: Default::default(),
        });
    }
    let config = builder.build()?;
    let prompt = match &args.prompt_file {
        Some(path) => std::fs::read_to_string(path)?,
        None => args.prompt.clone(),
    };
    let workspace = std::fs::canonicalize(&args.workspace)?;
    let task = Task::new(config, prompt, workspace, None);

    let runtime = AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()));
    let (mut events, _commands, handle) = runtime.run_task(
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
            AgentEvent::ToolCallFormatted {
                tool_name,
                parameters,
                ..
            } => println!("→ {tool_name} {}", serde_json::to_string(&parameters)?),
            AgentEvent::ToolRejected {
                tool_name, reason, ..
            } => {
                println!("✗ {tool_name}: {}", reason.unwrap_or_default())
            }
            AgentEvent::ModelCallUsageReported(usage) => {
                println!("  [request: {} input tokens]", usage.input_tokens)
            }
            AgentEvent::PlanUsageReported(usage) => println!(
                "  [plan usage: five-hour {:?}%, seven-day {:?}%]",
                usage.primary.map(|window| window.used_percent),
                usage.secondary.map(|window| window.used_percent)
            ),
            AgentEvent::RefusalReceived(refusal) => println!("[refusal] {refusal:?}"),
            AgentEvent::RefusalFallback { model_id } => println!("[refusal fallback] {model_id}"),
            AgentEvent::ModelRerouted {
                requested_model,
                served_model,
                ..
            } => println!("[rerouted] {requested_model} -> {served_model}"),
            AgentEvent::HandoverRequested { input_tokens } => {
                println!("=== handover requested at {input_tokens} input tokens ===")
            }
            AgentEvent::ConversationHandedOver {
                report,
                input_tokens,
                handovers,
            } => println!(
                "=== handover {handovers} done ({input_tokens} tokens); report:\n{report}\n==="
            ),
            AgentEvent::ConversationCompacted {
                input_tokens,
                compacted_tokens,
            } => println!(
                "=== Claude Code compacted the conversation: {input_tokens} tokens before, {} after ===",
                compacted_tokens.map_or("unknown".to_string(), |tokens| tokens.to_string())
            ),
            AgentEvent::ToolResult {
                tool_name, output, ..
            } => {
                let shown: String = output.chars().take(300).collect();
                println!("← {tool_name}: {shown}")
            }
            AgentEvent::AssistantResponseChunk(text) => println!("  {}", text.trim()),
            AgentEvent::TaskStateChanged(state) => println!("[state] {state:?}"),
            AgentEvent::ErrorEncountered { error_message, .. } => {
                println!("[error] {error_message}")
            }
            _ => {}
        }
    }
    let task = handle.await??;
    let usage = task.total_usage();
    println!(
        "\nFinal state: {:?}\nRequests: {}, input tokens: {}, output tokens: {}, list-price estimate: ${:.4}, billed: ${:.2}\nHandovers: {}",
        task.current_state,
        task.usage_history.len(),
        usage.input_tokens,
        usage.output_tokens,
        usage.total_cost,
        usage.billed(),
        task.handovers.len()
    );
    Ok(())
}
