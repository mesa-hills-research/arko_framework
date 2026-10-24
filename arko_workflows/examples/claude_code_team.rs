//! Two agents running in the installed Claude Code CLI (under its login)
//! work together through workflow messaging: a lead that reads files and
//! hands its conversation over at a low threshold, and a worker that
//! answers the lead's question. It prints each agent's tool calls, its
//! messages, and the lead's handovers.
//!
//! ```text
//! cargo run -p arko_workflows --example claude_code_team -- \
//!     --workspace arko_ai/src/providers/claude_code --handover-at 15000
//! ```
//!
//! It uses the CLI's login and counts against its plan usage.

use std::path::PathBuf;
use std::sync::Arc;

use arko_agents::agent_config::{AgentLLMConfig, AgentToolConfig};
use arko_agents::handover::HandoverConfig;
use arko_agents::runtime::event::AgentEvent;
use arko_agents::{AgentConfig, ClaudeCodeTools};
use arko_ai::{LLMClient, Model};
use arko_workflows::Workflow;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    workspace: PathBuf,
    #[arg(long, default_value = "claude-sonnet-5-5")]
    model: String,
    #[arg(long, default_value_t = 15_000)]
    handover_at: usize,
}

const LEAD: &str = "You coordinate with the agent `worker`. First, send `worker` a message asking it to read session.rs and reply with a short summary of the public methods of ClaudeSession. Then read locate.rs, protocol.rs and options.rs yourself, one at a time, noting each file's public items. When you have read them, wait for the worker's reply with wait_for_message if it hasn't arrived yet. Then call complete_task with your notes and the worker's summary.";

const WORKER: &str = "Wait for a request from `lead` with wait_for_message, do what it asks, send the lead your answer with send_message, then call complete_task.";

fn agent(name: &str, model: &Model, tools: Vec<String>) -> AgentConfig {
    AgentConfig::builder(
        name.to_string(),
        name.to_string(),
        AgentLLMConfig {
            model: model.clone(),
            generation_config: Default::default(),
        },
    )
    .role_description(format!(
        "The {name} of a two-agent team reading Rust sources."
    ))
    .instructions("Keep notes and messages brief.".into())
    .tool_config(AgentToolConfig {
        default_auto_approve_tool_use: true,
        available_tool_names: tools,
        ..AgentToolConfig::default()
    })
    .build()
    .expect("valid configuration")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let model = Model::claude_code(Model::get(arko_ai::Provider::Anthropic, &args.model)?);
    let mut lead = agent("lead", &model, ClaudeCodeTools::read_only());
    lead.handover = Some(HandoverConfig::at_input_tokens(args.handover_at));
    let worker = agent("worker", &model, ClaudeCodeTools::read_only());

    let workspace = std::fs::canonicalize(&args.workspace)?;
    let workflow = Workflow::builder()
        .name("claude-code-team")
        .workspace(&workspace)
        .llm_client(Arc::new(LLMClient::new()))
        .agent("lead", lead)
        .agent("worker", worker)
        .build()?;

    let mut events = workflow.subscribe_events();
    let printer = tokio::spawn(async move {
        while let Ok(event) = events.recv().await {
            let who = event.agent_name;
            match event.event {
                AgentEvent::ToolCallFormatted {
                    tool_name,
                    parameters,
                    ..
                } => {
                    let parameters = serde_json::to_string(&parameters).unwrap_or_default();
                    let cut: String = parameters.chars().take(160).collect();
                    println!("[{who}] → {tool_name} {cut}");
                }
                AgentEvent::ToolRejected {
                    tool_name, reason, ..
                } => {
                    let reason: String = reason.unwrap_or_default().chars().take(100).collect();
                    println!("[{who}] ✗ {tool_name}: {reason}");
                }
                AgentEvent::ToolResult {
                    tool_name, output, ..
                } if tool_name == "send_message" || tool_name == "wait_for_message" => {
                    println!("[{who}] ← {tool_name}: {output}")
                }
                AgentEvent::ModelCallUsageReported(usage) => {
                    println!("[{who}]   request: {} input tokens", usage.input_tokens)
                }
                AgentEvent::HandoverRequested { input_tokens } => {
                    println!("[{who}] === handover requested at {input_tokens} tokens ===")
                }
                AgentEvent::ConversationHandedOver { report, .. } => {
                    println!("[{who}] === handed over\nReport\n{report}\n===")
                }
                AgentEvent::ErrorEncountered { error_message, .. } => {
                    println!("[{who}] error: {error_message}")
                }
                _ => {}
            }
        }
    });

    let results = workflow
        .run_agents_parallel(vec![
            ("lead", LEAD.to_string()),
            ("worker", WORKER.to_string()),
        ])
        .await?;
    printer.abort();
    for result in &results {
        println!(
            "\n## {} — {:?}\n{}\nhandovers: {}, requests: {}",
            result.agent_name,
            result.status,
            result.summary,
            result.task()?.handovers.len(),
            result.info().model_calls
        );
    }
    let lead = results[0].task()?;
    let received = lead.conversation.messages.iter().any(|message| {
        message
            .content
            .iter()
            .filter_map(arko_ai::Content::as_text)
            .any(|text| text.contains("# Workflow context update") && text.contains("worker"))
    });
    println!(
        "\nThe worker's message is in the lead's current conversation: {received}\nMessages still queued for the lead: {}",
        workflow.context().messages.pending("lead").len()
    );
    Ok(())
}
