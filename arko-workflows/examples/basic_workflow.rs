//! Basic example: sequential multi-agent workflow.
//!
//! Usage:
//!   cargo run --example basic_workflow -- \
//!     --workspace ./my-project \
//!     --architect configs/architect.json \
//!     --coder configs/coder.json
//!
//! Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY) in env.

use std::sync::Arc;

use arko_agents::AgentConfig;
use arko_ai::LLMClient;
use arko_workflows::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut client = LLMClient::new();
    if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
        client = client.with_anthropic_api_key(&k);
    }
    if let Ok(k) = std::env::var("OPENAI_API_KEY") {
        client = client.with_openai_api_key(&k);
    }
    if let Ok(k) = std::env::var("GOOGLE_API_KEY") {
        client = client.with_google_api_key(&k);
    }
    let llm_client = Arc::new(client);

    let architect_config = AgentConfig::load_from_path("configs/architect.json".as_ref())
        .expect("Failed to load architect config");
    let coder_config = AgentConfig::load_from_path("configs/coder.json".as_ref())
        .expect("Failed to load coder config");

    let workflow = Workflow::builder()
        .name("example-workflow")
        .workspace("./my-project")
        .llm_client(llm_client)
        .agent("architect", architect_config)
        .agent("coder", coder_config)
        .memory(MemoryConfig::new().global_tier("decisions"))
        .budget(5.0)
        .budget_warning(4.0)
        .hook(
            HookPoint::PostAgentCompletion,
            Arc::new(|ctx| {
                println!(
                    "[HOOK] Agent '{}' completed. Budget: ${:.6}",
                    ctx.agent_name.as_deref().unwrap_or("?"),
                    ctx.budget_spent,
                );
                HookResult::Continue
            }),
        )
        .build()?;

    println!("Running architect...");
    let arch_result = workflow
        .run_agent(
            "architect",
            "Analyze this project and create an architecture document.",
        )
        .await?;
    println!("Architect: {}", arch_result.summary);
    println!("Cost so far: ${:.6}", arch_result.total_cost());

    println!("\nRunning coder...");
    let coder_result = workflow
        .run_agent(
            "coder",
            &format!(
                "Implement based on this architecture:\n{}",
                arch_result.summary
            ),
        )
        .await?;
    println!("Coder: {}", coder_result.summary);

    let result = workflow.result();
    println!("\n=== Workflow Complete ===");
    println!("Total cost: ${:.6}", result.total_cost);
    println!("Agents run: {}", result.agent_results.len());

    Ok(())
}
