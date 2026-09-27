use arko_agents::{
    AgentConfig, AgentRuntime, McpConfig, McpManager, SkillManager, Task, ToolCallingMode,
    ToolRegistry,
    runtime::{command::ToolApprovalDecision, command::UserCommand, event::AgentEvent},
};
use arko_ai::{CodexAuthFile, LLMClient, UsageInfo};
use clap::Parser;
use dotenv::dotenv;
use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
};

#[derive(Parser)]
#[command(author, version, about = "CLI example for arko_agents", long_about = None)]
struct Cli {
    #[arg(short = 'c', long)]
    agent_config: PathBuf,

    #[arg(short = 'w', long)]
    workspace_dir: PathBuf,

    #[arg(short = 'p', long)]
    prompt: String,

    #[arg(long, value_parser = parse_tool_calling_mode)]
    tool_calling_mode: Option<ToolCallingMode>,

    #[arg(long)]
    event_throttle_ms: Option<u64>,

    #[arg(long, default_value_t = true)]
    simplified_tool_events: bool,

    #[arg(long)]
    save_task_state: Option<PathBuf>,

    #[arg(long)]
    user_command_timeout: Option<u64>,

    /// Path to an MCP server configuration JSON file.
    #[arg(long)]
    mcp_config: Option<PathBuf>,

    /// Path to an OpenAI Codex auth.json file for ChatGPT OAuth.
    ///
    /// When provided, the CLI uses ChatGPT OAuth authentication instead of
    /// the OPENAI_API_KEY environment variable. The file is created by
    /// running `npx @openai/codex login` (typically at ~/.codex/auth.json).
    ///
    /// Token refresh is handled automatically by the agent runtime when the
    /// access token expires. Every refresh first re-reads this file (using
    /// tokens Codex already refreshed instead of refreshing again), and
    /// rotated tokens are saved back to it immediately.
    #[arg(long)]
    openai_auth_json: Option<PathBuf>,
}

fn parse_tool_calling_mode(s: &str) -> Result<ToolCallingMode, String> {
    match s.to_lowercase().as_str() {
        "output_based" | "outputbased" | "output-based" => Ok(ToolCallingMode::OutputBased),
        "native" => Ok(ToolCallingMode::Native),
        _ => Err(format!(
            "Unknown tool calling mode '{}'. Use 'output_based' or 'native'.",
            s
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    env_logger::init();

    let cli = Cli::parse();

    let mut llm_client_builder = LLMClient::from_env();

    if let Some(ref auth_json_path) = cli.openai_auth_json {
        println!("[SYSTEM] Loading OpenAI OAuth from {:?}", auth_json_path);
        let auth = CodexAuthFile::load(auth_json_path)?;
        let credentials = auth.credentials()?;
        llm_client_builder = llm_client_builder.with_codex_auth_file(&auth)?;
        println!(
            "[SYSTEM] OpenAI ChatGPT OAuth configured (token refresh {})",
            if credentials.refresh_token.is_some() {
                "enabled"
            } else {
                "disabled — no refresh_token"
            }
        );
    }
    let llm_client = Arc::new(llm_client_builder);

    let agent_config_json = fs::read_to_string(&cli.agent_config)?;
    let mut agent_config: AgentConfig = serde_json::from_str(&agent_config_json)?;

    if let Some(mode) = cli.tool_calling_mode {
        agent_config.tool_config.tool_calling_mode = mode;
    }

    if let Some(timeout_seconds) = cli.user_command_timeout {
        agent_config.user_command_timeout_seconds = Some(timeout_seconds);
    }

    if !cli.workspace_dir.exists() {
        fs::create_dir_all(&cli.workspace_dir)?;
        println!(
            "[SYSTEM] Created workspace directory: {:?}",
            cli.workspace_dir
        );
    }
    let absolute_workspace_dir = cli.workspace_dir.canonicalize()?;

    let mut tool_registry = ToolRegistry::new();

    let mcp_manager: Option<McpManager> = if let Some(ref mcp_config_path) = cli.mcp_config {
        println!(
            "[SYSTEM] Loading MCP configuration from {:?}",
            mcp_config_path
        );
        let mcp_config = McpConfig::load_from_path(mcp_config_path)?;

        if !mcp_config.mcp_servers.is_empty()
            && agent_config.tool_config.tool_calling_mode != ToolCallingMode::Native
        {
            println!(
                "[SYSTEM] MCP tools require native tool calling mode. \
                 Switching from {:?} to Native.",
                agent_config.tool_config.tool_calling_mode
            );
            agent_config.tool_config.tool_calling_mode = ToolCallingMode::Native;
        }

        println!(
            "[SYSTEM] Connecting to {} MCP server(s)...",
            mcp_config.mcp_servers.len()
        );
        let manager = McpManager::from_config(&mcp_config).await?;

        let mcp_tool_names = manager.tool_names();
        manager.register_tools(&mut tool_registry);

        for name in &mcp_tool_names {
            if !agent_config.tool_config.available_tool_names.contains(name) {
                agent_config
                    .tool_config
                    .available_tool_names
                    .push(name.clone());
            }
        }

        println!("[SYSTEM] Registered {} MCP tool(s):", mcp_tool_names.len());
        for name in &mcp_tool_names {
            println!("[SYSTEM]   - {}", name);
        }

        Some(manager)
    } else {
        None
    };

    let skill_prompt_context = if let Some(ref skills_config) = agent_config.skills_config {
        match SkillManager::load(skills_config, &absolute_workspace_dir) {
            Ok(skill_manager) => {
                let ctx = skill_manager.setup(&mut tool_registry, &mut agent_config);
                println!(
                    "[SYSTEM] Loaded {} skill(s) ({} on-demand, {} auto-loaded)",
                    skill_manager.total_skill_count(),
                    skill_manager.prompt_context().on_demand_skills.len(),
                    skill_manager.prompt_context().auto_loaded_skills.len()
                );
                Some(ctx)
            }
            Err(e) => {
                eprintln!(
                    "[WARN] Failed to load skills: {}. Continuing without skills.",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    println!("[SYSTEM] Agent: '{}'", agent_config.name);
    println!(
        "[SYSTEM] Model: {} ({:?})",
        agent_config.llm_config.model.id, agent_config.llm_config.model.provider
    );
    println!(
        "[SYSTEM] Tool calling mode: {:?}",
        agent_config.tool_config.tool_calling_mode
    );
    println!("[SYSTEM] Workspace: {:?}", absolute_workspace_dir.display());

    let tool_registry = Arc::new(tool_registry);

    let task = Task::new(
        agent_config.clone(),
        cli.prompt.clone(),
        absolute_workspace_dir,
        None,
    );
    let agent_runtime = AgentRuntime::new(llm_client.clone());

    let (mut event_receiver, command_sender, task_join_handle) = agent_runtime.run_task(
        task,
        tool_registry,
        cli.event_throttle_ms,
        cli.simplified_tool_events,
        skill_prompt_context,
        None,
        None,
    );

    println!("[SYSTEM] Task started. Prompt: '{}'", cli.prompt);
    println!("---");

    let mut accumulated_cost: f64 = 0.0;
    let mut last_turn_input_tokens: Option<usize> = None;
    let mut in_reasoning_block = false;
    let mut in_assistant_response_block = false;

    while let Some(event) = event_receiver.recv().await {
        match event {
            AgentEvent::TaskStateChanged(new_state) => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!("\n[STATE] {:?}", new_state);
                if new_state.is_terminal() {
                    println!("[SYSTEM] Task reached terminal state.");
                }
            }
            AgentEvent::AssistantResponseChunk(chunk) => {
                if in_reasoning_block {
                    print!("\n[END REASONING]\n");
                    io::stdout().flush()?;
                    in_reasoning_block = false;
                }
                if !in_assistant_response_block {
                    print!("\n[RESPONSE] ");
                    io::stdout().flush()?;
                    in_assistant_response_block = true;
                }
                print!("{}", chunk);
                io::stdout().flush()?;
            }
            AgentEvent::ReasoningChunk(chunk) => {
                if in_assistant_response_block {
                    println!();
                    io::stdout().flush()?;
                    in_assistant_response_block = false;
                }
                if !in_reasoning_block {
                    print!("\n[REASONING] ");
                    io::stdout().flush()?;
                    in_reasoning_block = true;
                }
                print!("{}", chunk);
                io::stdout().flush()?;
            }
            AgentEvent::UsageReported(usage) => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                print_usage_info(&usage, last_turn_input_tokens, accumulated_cost);
                accumulated_cost += usage.billed();
                last_turn_input_tokens = Some(usage.input_tokens);
            }
            AgentEvent::ToolCallInitiated {
                tool_call_id: _,
                tool_name,
            } => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!("\n[TOOL] Calling: {}", tool_name);
            }
            AgentEvent::ToolCallFormatted {
                tool_call_id: _,
                tool_name,
                parameters,
            } => {
                println!(
                    "[TOOL] {} params: {}",
                    tool_name,
                    serde_json::to_string(&parameters).unwrap_or_else(|_| "{}".to_string())
                );
            }
            AgentEvent::ToolResult {
                tool_call_id: _,
                tool_name,
                output,
            } => {
                let display_output = if output.len() > 500 {
                    let mut end = 500;
                    while !output.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}... ({} chars total)", &output[..end], output.len())
                } else {
                    output
                };
                println!("[TOOL RESULT] {}: {}", tool_name, display_output);
            }
            AgentEvent::ToolCallAwaitingApproval {
                tool_call_id,
                tool_name,
                parameters,
                tool_description,
            } => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!("\n--- TOOL APPROVAL REQUIRED ---");
                println!("Tool: {}", tool_name);
                println!("Description: {}", tool_description);
                println!("Parameters: {}", serde_json::to_string_pretty(&parameters)?);

                let decision = prompt_for_single_tool_approval(&tool_name)?;
                let cmd = match decision {
                    ToolApprovalDecision::Approve {
                        custom_tool_output,
                        message_to_agent,
                    } => UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output,
                        message_to_agent,
                    },
                    ToolApprovalDecision::Reject { reason } => UserCommand::RejectToolUse {
                        tool_call_id,
                        reason,
                    },
                };
                if command_sender.send(cmd).await.is_err() {
                    eprintln!("[ERROR] Failed to send approval command.");
                    break;
                }
            }
            AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!(
                    "\n--- BATCH TOOL APPROVAL REQUIRED (Batch ID: {}) ---",
                    batch_id
                );
                let mut decisions = HashMap::new();
                for (idx, call_info) in pending_calls.iter().enumerate() {
                    println!("\nCall {}/{}:", idx + 1, pending_calls.len());
                    println!("  Tool: {}", call_info.tool_name);
                    println!(
                        "  Parameters: {}",
                        serde_json::to_string_pretty(&call_info.parameters)?
                    );

                    let decision = prompt_for_single_tool_approval(&call_info.tool_name)?;
                    decisions.insert(call_info.tool_call_id, decision);
                }
                let cmd = UserCommand::SubmitBatchToolApprovals {
                    batch_id,
                    decisions,
                };
                if command_sender.send(cmd).await.is_err() {
                    eprintln!("[ERROR] Failed to send batch approval command.");
                    break;
                }
            }
            AgentEvent::UserQuestionsAsked { questions } => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!("\n--- AGENT QUESTIONS ---");
                println!("{}", questions);
                println!("--- Please provide your response ---");

                print!("> ");
                io::stdout().flush()?;
                let mut response = String::new();
                io::stdin().read_line(&mut response)?;

                let cmd = UserCommand::SubmitUserResponse {
                    response: response.trim().to_string(),
                };
                if command_sender.send(cmd).await.is_err() {
                    eprintln!("[ERROR] Failed to send user response.");
                    break;
                }
            }
            AgentEvent::ErrorEncountered {
                error_message,
                requires_approval_to_retry,
                tool_call_id: _,
            } => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                eprintln!("\n[ERROR] {}", error_message);
                if requires_approval_to_retry {
                    print!("Attempt recovery? (y/n): ");
                    io::stdout().flush()?;
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    let cmd = match input.trim().to_lowercase().as_str() {
                        "y" | "yes" => {
                            print!("Message for recovery (optional): ");
                            io::stdout().flush()?;
                            let mut msg_input = String::new();
                            io::stdin().read_line(&mut msg_input)?;
                            UserCommand::ApproveErrorRecovery {
                                message: Some(msg_input.trim().to_string())
                                    .filter(|s| !s.is_empty()),
                            }
                        }
                        _ => UserCommand::RejectErrorRecovery {
                            message: Some("User rejected recovery.".to_string()),
                        },
                    };
                    if command_sender.send(cmd).await.is_err() {
                        eprintln!("[ERROR] Failed to send recovery command.");
                        break;
                    }
                }
            }
            AgentEvent::RefusalReceived(refusal) => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!("\n[REFUSAL] {}", refusal);
            }
            AgentEvent::TaskStreamEnd => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                println!("--- Agent stream ended ---");
            }
            _ => {
                close_active_blocks(&mut in_reasoning_block, &mut in_assistant_response_block)?;
                log::debug!("[EVENT] {:?}", event);
            }
        }
    }

    let final_task_result = task_join_handle.await;

    match final_task_result {
        Ok(Ok(final_task)) => {
            println!("\n=== TASK COMPLETE ===");
            println!("ID: {}", final_task.id);
            println!("State: {:?}", final_task.current_state);
            if let Some(duration) = final_task.get_run_time() {
                println!(
                    "Duration: {:.2}s",
                    duration.num_milliseconds() as f64 / 1000.0
                );
            }
            println!("Total cost: ${:.6}", accumulated_cost);
            println!("LLM calls: {}", final_task.usage_history.len());

            if let Some(save_path) = &cli.save_task_state {
                match final_task.save_to_path(save_path) {
                    Ok(_) => println!("[SYSTEM] Task state saved to: {:?}", save_path),
                    Err(e) => eprintln!("[ERROR] Failed to save task state: {}", e),
                }
            }
        }
        Ok(Err(e)) => {
            eprintln!("\n=== TASK FAILED (Runtime Error) ===");
            eprintln!("Error: {:?}", e);
        }
        Err(e) => {
            eprintln!("\n=== TASK FAILED (Join Error) ===");
            eprintln!("Error: {:?}", e);
        }
    }

    if let Some(manager) = mcp_manager {
        println!("[SYSTEM] Shutting down MCP servers...");
        manager.shutdown().await;
        println!("[SYSTEM] MCP servers shut down.");
    }

    Ok(())
}

fn close_active_blocks(in_reasoning: &mut bool, in_response: &mut bool) -> Result<(), io::Error> {
    if *in_reasoning {
        print!("\n[END REASONING]\n");
        io::stdout().flush()?;
        *in_reasoning = false;
    }
    if *in_response {
        println!();
        io::stdout().flush()?;
        *in_response = false;
    }
    Ok(())
}

fn prompt_for_single_tool_approval(tool_name: &str) -> Result<ToolApprovalDecision, io::Error> {
    loop {
        print!("Approve '{}'? (y/n/m=yes+message): ", tool_name);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => {
                return Ok(ToolApprovalDecision::Approve {
                    custom_tool_output: None,
                    message_to_agent: None,
                });
            }
            "n" | "no" => {
                print!("Reason (optional): ");
                io::stdout().flush()?;
                let mut reason_input = String::new();
                io::stdin().read_line(&mut reason_input)?;
                return Ok(ToolApprovalDecision::Reject {
                    reason: Some(reason_input.trim().to_string()).filter(|s| !s.is_empty()),
                });
            }
            "m" => {
                print!("Message to agent: ");
                io::stdout().flush()?;
                let mut msg_input = String::new();
                io::stdin().read_line(&mut msg_input)?;
                return Ok(ToolApprovalDecision::Approve {
                    custom_tool_output: None,
                    message_to_agent: Some(msg_input.trim().to_string()).filter(|s| !s.is_empty()),
                });
            }
            _ => println!("Invalid input. Enter 'y', 'n', or 'm'."),
        }
    }
}

fn print_usage_info(usage: &UsageInfo, last_input_tokens: Option<usize>, accumulated_cost: f64) {
    println!("\n--- Usage ---");
    if let Some(last_tokens) = last_input_tokens {
        let delta = usage.input_tokens as i64 - last_tokens as i64;
        println!("  Input:  {} ({:+})", usage.input_tokens, delta);
    } else {
        println!("  Input:  {}", usage.input_tokens);
    }
    println!("  Output: {}", usage.output_tokens);
    println!(
        "  Cost:   ${:.6} billed, ${:.6} at list prices (accumulated billed: ${:.6})",
        usage.billed(),
        usage.total_cost,
        accumulated_cost + usage.billed()
    );
    if let Some(cached) = usage.cached_tokens
        && cached > 0
    {
        println!("  Cached: {}", cached);
    }
    println!("---");
}
