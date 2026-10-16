//! Runs the user's installed Claude Code for a short conversation, to check
//! the client against the real CLI: an in-process tool, a hook, the
//! permission handler, two turns in one process, and a resume in a new
//! process. It uses the CLI's current login and so counts against its usage.
//!
//! cargo run -p arko_ai --example claude_code_live -- --model sonnet --effort low

use std::sync::Arc;

use arko_ai::claude_code::*;
use async_trait::async_trait;
use clap::Parser;
use serde_json::{Value, json};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "sonnet")]
    model: String,
    #[arg(long, default_value = "low")]
    effort: String,
}

struct Add;

#[async_trait]
impl ClaudeTool for Add {
    fn name(&self) -> String {
        "add".into()
    }
    fn description(&self) -> String {
        "Adds two numbers and returns the sum.".into()
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        })
    }
    async fn call(&self, arguments: Value) -> ToolOutput {
        let sum = arguments["a"].as_f64().unwrap_or(0.0) + arguments["b"].as_f64().unwrap_or(0.0);
        println!("  [tool] add({arguments}) = {sum}");
        ToolOutput::text(sum.to_string())
    }
}

struct LogHook;

#[async_trait]
impl HookCallback for LogHook {
    async fn call(&self, input: Value, _tool_use_id: Option<String>) -> Value {
        println!("  [hook] PreToolUse {}", input["tool_name"]);
        Hook::proceed()
    }
}

struct AllowOnlyAdd;

#[async_trait]
impl PermissionHandler for AllowOnlyAdd {
    async fn decide(&self, request: PermissionRequest) -> PermissionDecision {
        println!("  [permission] {} {}", request.tool_name, request.input);
        if request.tool_name == "mcp__math__add" {
            PermissionDecision::allow()
        } else {
            PermissionDecision::deny("Only the add tool is available here.")
        }
    }
}

fn show(turn: &Turn) {
    for message in &turn.messages {
        match message {
            Message::System(system) => println!("  system/{}", system.subtype),
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => println!("  assistant: {text}"),
                        ContentBlock::ToolUse { name, input, .. } => {
                            println!("  tool_use: {name} {input}")
                        }
                        ContentBlock::Thinking(_) => println!("  (thinking)"),
                        other => println!("  {other:?}"),
                    }
                }
            }
            Message::User(_) => println!("  (tool result)"),
            Message::RateLimit { info, .. } => println!("  rate limit: {info}"),
            Message::Result(_) => {}
            other => println!("  {other:?}"),
        }
    }
    let result = &turn.result;
    println!(
        "  result: {} is_error={} turns={} cost=${:.4} session={}",
        result.subtype,
        result.is_error,
        result.num_turns,
        result.total_cost_usd.unwrap_or_default(),
        result.session_id
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let effort = match args.effort.as_str() {
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        "xhigh" => Effort::XHigh,
        "max" => Effort::Max,
        other => return Err(format!("unknown effort {other}").into()),
    };
    let math = Arc::new(ToolServer::new("math").tool(Arc::new(Add)));
    let options = ClaudeOptions {
        model: Some(args.model.clone()),
        effort: Some(effort),
        built_in_tools: BuiltInTools::None,
        system_prompt: SystemPrompt::Append("Keep answers to one short line.".into()),
        permission_handler: Some(Arc::new(AllowOnlyAdd)),
        hooks: vec![Hook::new(HookEvent::PreToolUse, None, Arc::new(LogHook))],
        tool_servers: vec![math.clone()],
        working_directory: Some(std::env::temp_dir()),
        ..ClaudeOptions::default()
    };

    let mut session = ClaudeSession::start(options.clone()).await?;
    let init = session.initialization();
    println!(
        "initialized: {} commands, account {}",
        init["commands"].as_array().map_or(0, Vec::len),
        init["account"]
            .get("subscriptionType")
            .unwrap_or(&Value::Null)
    );

    println!("turn 1");
    session
        .send("Use the add tool to add 17 and 25. Reply with just the sum.")
        .await?;
    show(&session.receive_turn().await?);

    println!("turn 2 (same process)");
    session
        .send("Now double that number in your head. Reply with just the number.")
        .await?;
    show(&session.receive_turn().await?);

    let session_id = session.session_id().ok_or("no session id")?;
    session.close().await?;

    println!("turn 3 (new process, resumed)");
    let mut resumed = ClaudeSession::start(ClaudeOptions {
        session: SessionStart::Resume {
            session: session_id,
            fork: false,
            at: None,
        },
        ..options
    })
    .await?;
    resumed
        .send("What was the last number you replied with? Reply with just the number.")
        .await?;
    show(&resumed.receive_turn().await?);
    resumed.close().await?;
    Ok(())
}
