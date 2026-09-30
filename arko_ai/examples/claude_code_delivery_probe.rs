//! Checks how a message from another agent can reach a running Claude Code
//! turn: as `additionalContext` from a `PostToolUse` hook, or as a user
//! message sent while the turn runs. Uses the CLI's login.
//!
//! cargo run -p arko_ai --example claude_code_delivery_probe

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arko_ai::claude_code::*;
use async_trait::async_trait;
use serde_json::{Value, json};

/// A tool that takes a moment, so the turn has tool boundaries.
struct Step;

#[async_trait]
impl ClaudeTool for Step {
    fn name(&self) -> String {
        "step".into()
    }
    fn description(&self) -> String {
        "Performs one step of the job. Takes a few seconds.".into()
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    }
    async fn call(&self, arguments: Value) -> ToolOutput {
        tokio::time::sleep(Duration::from_secs(3)).await;
        ToolOutput::text(format!("step {} done", arguments["n"]))
    }
}

/// Delivers one message after the first tool call.
struct Inbox(AtomicUsize);

#[async_trait]
impl HookCallback for Inbox {
    async fn call(&self, _input: Value, _tool_use_id: Option<String>) -> Value {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            json!({"hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "Message from agent `reviewer`: the code word is BLUE.",
            }})
        } else {
            Hook::proceed()
        }
    }
}

const TASK: &str = "Call the step tool three times, with n = 1, 2 and 3, one call at a time. \
Then reply with one line: the messages you received from other agents while working, \
and after which step you first saw each one (or 'none').";

fn options(hooks: Vec<Hook>) -> ClaudeOptions {
    ClaudeOptions {
        model: Some("sonnet".into()),
        effort: Some(Effort::Low),
        built_in_tools: BuiltInTools::None,
        allowed_tools: vec!["mcp__work__step".into()],
        tool_servers: vec![Arc::new(ToolServer::new("work").tool(Arc::new(Step)))],
        hooks,
        working_directory: Some(std::env::temp_dir()),
        ..ClaudeOptions::default()
    }
}

/// Prints messages until `results` turn results have arrived.
async fn watch(session: &mut ClaudeSession, results: usize, mut on_tool_use: impl FnMut(usize)) {
    let (mut seen, mut tools) = (0, 0);
    while let Some(message) = session.next_message().await {
        match &message {
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => println!("  assistant: {text}"),
                        ContentBlock::ToolUse { input, .. } => {
                            tools += 1;
                            println!("  tool_use step {input}");
                            on_tool_use(tools);
                        }
                        _ => {}
                    }
                }
            }
            Message::User(user) => {
                for block in &user.content {
                    match block {
                        ContentBlock::ToolResult { content, .. } => {
                            println!("  tool_result: {content}")
                        }
                        ContentBlock::Text(text) => println!("  user: {text}"),
                        _ => {}
                    }
                }
            }
            Message::Result(result) => {
                seen += 1;
                println!("  RESULT #{seen}: {:?}", result.result);
                if seen == results {
                    return;
                }
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("A. PostToolUse additionalContext");
    let hook = Hook::new(
        HookEvent::PostToolUse,
        None,
        Arc::new(Inbox(AtomicUsize::new(0))),
    );
    let mut session = ClaudeSession::start(options(vec![hook])).await?;
    session.send(TASK).await?;
    watch(&mut session, 1, |_| {}).await;
    session.close().await?;

    println!("B. User message sent during the turn");
    let mut session = ClaudeSession::start(options(Vec::new())).await?;
    session.send(TASK).await?;
    let (mut sent, mut results) = (false, 0);
    loop {
        let wait = if results == 0 { 300 } else { 60 };
        let Ok(Some(message)) =
            tokio::time::timeout(Duration::from_secs(wait), session.next_message()).await
        else {
            println!("  (no further messages)");
            break;
        };
        match &message {
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text(text) => println!("  assistant: {text}"),
                        ContentBlock::ToolUse { input, .. } => {
                            println!("  tool_use step {input}");
                            if !sent {
                                sent = true;
                                println!("  >> sending user message");
                                session
                                    .send("Message from agent `reviewer`: the code word is GREEN.")
                                    .await?;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Message::User(user) => {
                for block in &user.content {
                    match block {
                        ContentBlock::ToolResult { content, .. } => {
                            println!("  tool_result: {content}")
                        }
                        ContentBlock::Text(text) => println!("  user: {text}"),
                        _ => {}
                    }
                }
            }
            Message::Result(result) => {
                results += 1;
                println!("  RESULT #{results}: {:?}", result.result);
            }
            _ => {}
        }
    }
    session.close().await?;
    Ok(())
}
