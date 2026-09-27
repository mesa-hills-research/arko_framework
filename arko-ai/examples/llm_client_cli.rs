use arko_ai::{
    model::ModelType, reasoning::{AnthropicReasoningFragment, OpenAIEffort},
    AnthropicReasoningEffort, Content, Error, GenerationConfig, GeminiThinkingLevel, Image,
    LLMClient, Model, OpenRouterReasoning, Provider, ReasoningArtifact, ReasoningEffort,
    ReasoningConfig, StreamEvent, UsageInfo,
};
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use futures::StreamExt;
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt (optionally streaming)
    Text {
        /// The model to use
        #[arg(short, long)]
        model: String,

        /// The prompt to generate from
        #[arg(short, long)]
        prompt: String,

        /// The system message
        #[arg(short, long)]
        system: Option<String>,

        /// The temperature (0.0-2.0)
        #[arg(short, long)]
        temperature: Option<f64>,

        /// The maximum number of tokens to generate
        #[arg(short = 'n', long)]
        max_tokens: Option<usize>,

        /// Enable reasoning (for supported models)
        #[arg(short, long)]
        reasoning: bool,

        /// The reasoning level (minimal/low/medium/high for OpenAI, integer for Anthropic/Google 2.x, high/low for Google 3.x, an effort or a token budget for OpenRouter)
        #[arg(long)]
        reasoning_level: Option<String>,

        /// The reasoning effort for Claude Opus 4.5 (low/medium/high)
        #[arg(long)]
        reasoning_effort: Option<String>,

        /// Image input as file path
        #[arg(short, long)]
        image: Option<PathBuf>,

        /// Stream the response
        #[arg(long)]
        stream: bool,

        /// Enable 1M token context window (Claude Sonnet 4/4.5 only)
        #[arg(long)]
        long_context: bool,
    },

    /// Start an interactive, streaming conversation
    Chat {
        /// The model to use
        #[arg(short, long)]
        model: String,

        /// The system message
        #[arg(short, long)]
        system: Option<String>,

        /// The temperature (0.0-2.0)
        #[arg(short, long)]
        temperature: Option<f64>,

        /// The maximum number of tokens to generate
        #[arg(short = 'n', long)]
        max_tokens: Option<usize>,

        /// Enable reasoning (for supported models)
        #[arg(short, long)]
        reasoning: bool,

        /// The reasoning level (minimal/low/medium/high for OpenAI, integer for Anthropic/Google 2.x, high/low for Google 3.x, an effort or a token budget for OpenRouter)
        #[arg(long)]
        reasoning_level: Option<String>,

        /// The reasoning effort for Claude Opus 4.5 (low/medium/high)
        #[arg(long)]
        reasoning_effort: Option<String>,

        /// Enable 1M token context window (Claude Sonnet 4/4.5 only)
        #[arg(long)]
        long_context: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<arko_ai::Error>> {
    dotenv().ok();
    env_logger::init();

    let cli = Cli::parse();
    let client = create_client()?;

    match cli.command {
        Commands::Text {
            model,
            prompt,
            system,
            temperature,
            max_tokens,
            reasoning,
            reasoning_level,
            reasoning_effort,
            image,
            stream,
            long_context,
        } => {
            let model = get_model(&model)?;
            let config = build_config(
                model.clone(),
                temperature,
                max_tokens,
                reasoning,
                reasoning_level,
                reasoning_effort,
                long_context,
            )?;

            let mut conversation = client.new_conversation(model);
            if let Some(system_msg) = system {
                conversation = conversation.with_system_message(&system_msg);
            }

            if let Some(image_path) = image {
                let image_data = load_image(&image_path)?;
                conversation.add_user_message_with_contents(vec![
                    Content::Text(prompt),
                    Content::Image(image_data),
                ]);
            } else {
                conversation.add_user_message(&prompt);
            }

            if stream {
                let mut stream = client.generate_stream(&conversation, config).await?;
                let mut reasoning_shown = false;
                let mut response_started = false;

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::ReasoningChunk(r)) => {
                            if !reasoning_shown {
                                print!("\nThinking: ");
                                io::stdout().flush().map_err(Error::IoError)?;
                                reasoning_shown = true;
                            }
                            print!("{}", r);
                            io::stdout().flush().map_err(Error::IoError)?;
                        }
                        Ok(StreamEvent::TextChunk(t)) => {
                            if !response_started {
                                if reasoning_shown {
                                    println!("\n");
                                }
                                print!("\nResponse: ");
                                io::stdout().flush().map_err(Error::IoError)?;
                                response_started = true;
                            }
                            print!("{}", t);
                            io::stdout().flush().map_err(Error::IoError)?;
                        }
                        Ok(StreamEvent::ToolCallStart { .. } | StreamEvent::ToolCallArgumentsDelta { .. }) => {}
                        Ok(StreamEvent::Refusal(refusal)) => {
                            println!("\n\nThe response was {refusal}");
                        }
                        Ok(StreamEvent::End(boxed_resp)) => {
                            let resp = *boxed_resp;
                            println!("\n");
                            print_usage_info(&resp.usage, None, 0.0, 0.0);
                            break;
                        }
                        Err(e) => return Err(Box::new(e)),
                    }
                }
            } else {
                let response = client.generate_text(&conversation, config).await?;

                println!("Response: {}", response.text());
                if let Some(refusal) = &response.refusal {
                    println!("The response was {refusal}");
                }

                print_reasoning_artifact(response.message.reasoning_artifact.as_ref());
                print_usage_info(&response.usage, None, 0.0, 0.0);
            }
        }

        Commands::Chat {
            model,
            system,
            temperature,
            max_tokens,
            reasoning,
            reasoning_level,
            reasoning_effort,
            long_context,
        } => {
            let model = get_model(&model)?;
            let config = build_config(
                model.clone(),
                temperature,
                max_tokens,
                reasoning,
                reasoning_level,
                reasoning_effort,
                long_context,
            )?;

            let mut conversation = client.new_conversation(model);
            if let Some(system_msg) = system {
                conversation = conversation.with_system_message(&system_msg);
            }

            println!(
                "Starting chat with model '{}' (type 'exit' to quit)",
                conversation.model.id
            );
            if let Some(sys) = &conversation.system_message {
                println!("System: {}", sys);
            }
            println!("---");

            // Billed amount (zero for subscription calls) and the estimate
            // at API list prices, accumulated separately.
            let mut total_billed: f64 = 0.0;
            let mut total_estimate: f64 = 0.0;
            let mut last_input_tokens: Option<usize> = None;

            loop {
                print!("You: ");
                io::stdout().flush().map_err(Error::IoError)?;
                let mut input = String::new();
                io::stdin().read_line(&mut input).map_err(Error::IoError)?;
                let input = input.trim();

                if input.is_empty() {
                    continue;
                }
                if input.to_lowercase() == "exit" {
                    break;
                }

                conversation.add_user_message(input);

                print!("Assistant: ");
                io::stdout().flush().map_err(Error::IoError)?;

                let mut stream = client
                    .generate_stream(&conversation, config.clone())
                    .await?;
                let mut reasoning_shown_this_turn = false;
                let mut response_started_this_turn = false;
                let mut final_response_for_turn = None;

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::ReasoningChunk(r)) => {
                            if !reasoning_shown_this_turn {
                                print!("\n  (Thinking: ");
                                io::stdout().flush().map_err(Error::IoError)?;
                                reasoning_shown_this_turn = true;
                            }
                            print!("{}", r);
                            io::stdout().flush().map_err(Error::IoError)?;
                        }
                        Ok(StreamEvent::TextChunk(t)) => {
                            if !response_started_this_turn {
                                if reasoning_shown_this_turn {
                                    println!(")");
                                    print!("  ");
                                }
                                response_started_this_turn = true;
                            }
                            print!("{}", t);
                            io::stdout().flush().map_err(Error::IoError)?;
                        }
                        Ok(StreamEvent::ToolCallStart { .. } | StreamEvent::ToolCallArgumentsDelta { .. }) => {}
                        Ok(StreamEvent::Refusal(refusal)) => {
                            println!("\n  (The response was {refusal})");
                        }
                        Ok(StreamEvent::End(boxed_resp)) => {
                            let resp = *boxed_resp;
                            final_response_for_turn = Some(resp);
                            break;
                        }
                        Err(e) => {
                            eprintln!("\nError during generation: {}", e);
                            final_response_for_turn = None;
                            break;
                        }
                    }
                }
                println!();

                if let Some(resp) = final_response_for_turn {
                    // A refused turn that can't stay in history is removed
                    // together with the prompt that triggered it.
                    if resp.refusal.as_ref().is_some_and(|refusal| !refusal.source.keeps_turn()) {
                        conversation.messages.pop();
                    } else {
                        conversation.add_assistant_message_from_response(resp.clone());
                    }
                    total_billed += resp.usage.billed();
                    total_estimate += resp.usage.total_cost;
                    print_usage_info(&resp.usage, last_input_tokens, total_billed, total_estimate);
                    last_input_tokens = Some(resp.usage.input_tokens);
                } else {
                    conversation.messages.pop();
                    println!("Turn failed. Please try again.");
                }
                println!("---");
            }
            println!(
                "Chat ended. Final accumulated billed cost: ${:.6} (estimate at list prices: ${:.6})",
                total_billed, total_estimate
            );
        }
    }

    Ok(())
}

fn create_client() -> Result<LLMClient, Error> {
    Ok(LLMClient::from_env())
}

fn get_model(name: &str) -> Result<Model, Error> {
    match name {
        "gpt-5" => Ok(Model::openai_gpt_5()),
        "gpt-5-mini" => Ok(Model::openai_gpt_5_mini()),
        "gpt-5-nano" => Ok(Model::openai_gpt_5_nano()),
        "o4-mini" => Ok(Model::openai_o4_mini()),
        "o3" => Ok(Model::openai_o3()),
        "o3-pro" => Ok(Model::openai_o3_pro()),
        "gpt-4.1" => Ok(Model::openai_gpt_4_1()),
        "gpt-4.1-mini" => Ok(Model::openai_gpt_4_1_mini()),
        "gpt-4.1-nano" => Ok(Model::openai_gpt_4_1_nano()),
        "claude-opus-4" => Ok(Model::anthropic_claude_opus_4()),
        "claude-opus-4-5" | "claude-opus-4.5" => Ok(Model::anthropic_claude_opus_4_5()),
        "claude-sonnet-4" => Ok(Model::anthropic_claude_sonnet_4()),
        "claude-sonnet-4-5" | "claude-sonnet-4.5" => Ok(Model::anthropic_claude_sonnet_4_5()),
        "claude-haiku-4-5" | "claude-haiku-4.5" => Ok(Model::anthropic_claude_haiku_4_5()),
        "claude-3-7-sonnet" => Ok(Model::anthropic_claude_3_7_sonnet()),
        "claude-3-5-sonnet-20241022" => Ok(Model::anthropic_claude_3_5_sonnet_20241022()),
        "claude-3-5-sonnet" | "claude-3-5-sonnet-20240620" => {
            Ok(Model::anthropic_claude_3_5_sonnet_20240620())
        }
        "gemini-3-pro" | "gemini-3.0-pro" => Ok(Model::google_gemini_3_0_pro()),
        "gemini-2.5-pro" => Ok(Model::google_gemini_2_5_pro()),
        "gemini-2.5-flash" => Ok(Model::google_gemini_2_5_flash()),
        "gemini-2.5-flash-lite" => Ok(Model::google_gemini_2_5_flash_lite()),
        "gemini-2.0-flash" => Ok(Model::google_gemini_2_0_flash()),
        // OpenRouter slugs: author/model, ~author/alias-latest, @preset/name.
        slug if slug.contains('/') => Ok(Model::openrouter(slug)),
        _ => Model::find(name).ok_or_else(|| {
            Error::ModelNotSupported(format!(
                "Unknown model '{}'. Use a known alias or a registered model id.",
                name
            ))
        }),
    }
}

fn build_config(
    model: Model,
    temperature: Option<f64>,
    max_tokens: Option<usize>,
    reasoning: bool,
    reasoning_level: Option<String>,
    reasoning_effort: Option<String>,
    long_context: bool,
) -> Result<GenerationConfig, Error> {
    let mut config = GenerationConfig::default();

    if let Some(temp) = temperature {
        config = config.with_temperature(temp);
    }
    if let Some(tokens) = max_tokens {
        config = config.with_max_tokens(tokens);
    }
    if long_context {
        config = config.with_long_context(true);
    }

    if reasoning {
        let reasoning_config = match model.provider {
            Provider::OpenAI => {
                if model.model_type != ModelType::Reasoning {
                    return Err(Error::InvalidRequest(format!(
                        "Reasoning is only supported for OpenAI O-models (e.g., o4-mini, o3), not '{}'.",
                        model.id
                    )));
                }
                if config.temperature_specified {
                    println!(
                        "Warning: Temperature is ignored for OpenAI O-models when reasoning is enabled."
                    );
                }
                let effort_str = reasoning_level.unwrap_or_else(|| "medium".to_string());

                let effort = match effort_str.as_str() {
                    "minimal" => {
                        if !model.id.starts_with("gpt-5-") {
                            return Err(Error::InvalidRequest(
                                "Reasoning effort 'minimal' is only supported for GPT-5 models."
                                    .to_string(),
                            ));
                        }
                        OpenAIEffort::Minimal
                    }
                    "low" => OpenAIEffort::Low,
                    "medium" => OpenAIEffort::Medium,
                    "high" => OpenAIEffort::High,
                    _ => {
                        return Err(Error::InvalidRequest(
                            "OpenAI reasoning effort must be 'minimal' (GPT-5 only), 'low', 'medium', or 'high'"
                                .to_string(),
                        ));
                    }
                };

                ReasoningConfig::openai(effort, true)
            }
            Provider::Anthropic => {
                if model.model_type != ModelType::Reasoning {
                    return Err(Error::InvalidRequest(format!(
                        "Reasoning is only supported for Anthropic reasoning models (e.g., claude-3-7-sonnet), not '{}'.",
                        model.id
                    )));
                }

                let max_budget = if model.is_opus_4_5() { 52000 } else { 102400 };

                let budget = reasoning_level
                    .map(|s| s.parse::<usize>())
                    .transpose()
                    .map_err(|_| {
                        Error::InvalidRequest(
                            "Invalid reasoning budget, must be an integer".to_string(),
                        )
                    })?
                    .unwrap_or(1024);

                if !(1024..=max_budget).contains(&budget) {
                    return Err(Error::InvalidRequest(format!(
                        "Anthropic reasoning budget must be between 1024 and {}",
                        max_budget
                    )));
                }

                if config.temperature_specified && config.temperature != 1.0 {
                    return Err(Error::InvalidRequest(
                        "Anthropic models with reasoning enabled require temperature=1.0. Omit --temperature or set it to 1.0."
                            .to_string(),
                    ));
                } else if !config.temperature_specified {
                    println!(
                        "Info: Setting temperature to 1.0 as required by Anthropic reasoning."
                    );
                    config.temperature = 1.0;
                    config.temperature_specified = true;
                }

                if model.is_opus_4_5() {
                    if let Some(effort_str) = reasoning_effort {
                        let effort = match effort_str.to_lowercase().as_str() {
                            "low" => AnthropicReasoningEffort::Low,
                            "medium" => AnthropicReasoningEffort::Medium,
                            "high" => AnthropicReasoningEffort::High,
                            _ => {
                                return Err(Error::InvalidRequest(
                                    "Anthropic reasoning effort must be 'low', 'medium', or 'high'"
                                        .to_string(),
                                ));
                            }
                        };
                        ReasoningConfig::anthropic_with_effort(budget, effort)
                    } else {
                        ReasoningConfig::anthropic(budget)
                    }
                } else {
                    if reasoning_effort.is_some() {
                        return Err(Error::InvalidRequest(
                            "Reasoning effort parameter is only supported for Claude Opus 4.5"
                                .to_string(),
                        ));
                    }
                    ReasoningConfig::anthropic(budget)
                }
            }
            Provider::OpenRouter => {
                // An effort name, or a number as a reasoning token budget.
                let level = reasoning_level.or(reasoning_effort).unwrap_or_else(|| "medium".into());
                let reasoning = match level.as_str() {
                    "none" => OpenRouterReasoning::effort(ReasoningEffort::None),
                    "minimal" => OpenRouterReasoning::effort(ReasoningEffort::Minimal),
                    "low" => OpenRouterReasoning::effort(ReasoningEffort::Low),
                    "medium" => OpenRouterReasoning::effort(ReasoningEffort::Medium),
                    "high" => OpenRouterReasoning::effort(ReasoningEffort::High),
                    "xhigh" => OpenRouterReasoning::effort(ReasoningEffort::XHigh),
                    "max" => OpenRouterReasoning::effort(ReasoningEffort::Max),
                    budget => OpenRouterReasoning::max_tokens(budget.parse().map_err(|_| {
                        Error::InvalidRequest(
                            "OpenRouter reasoning takes an effort (none..max) or a token budget".to_string(),
                        )
                    })?),
                };
                ReasoningConfig::openrouter(reasoning)
            }
            Provider::Google => {
                if model.model_type != ModelType::Reasoning {
                    return Err(Error::InvalidRequest(format!(
                        "Reasoning is only supported for Google reasoning models (e.g., gemini-2.5-pro), not '{}'.",
                        model.id
                    )));
                }

                if model.is_gemini_3_pro() {
                    let level_str = reasoning_level.unwrap_or_else(|| "high".to_string());
                    let level = match level_str.to_uppercase().as_str() {
                        "HIGH" => GeminiThinkingLevel::High,
                        "LOW" => GeminiThinkingLevel::Low,
                        _ => {
                            return Err(Error::InvalidRequest(
                                "Gemini 3.0 Pro thinking level must be 'high' or 'low'".to_string(),
                            ));
                        }
                    };
                    ReasoningConfig::gemini(level, true)
                } else {
                    let budget = reasoning_level
                        .map(|s| s.parse::<isize>())
                        .transpose()
                        .map_err(|_| {
                            Error::InvalidRequest(
                                "Invalid reasoning budget, must be an integer".to_string(),
                            )
                        })?
                        .unwrap_or(-1);

                    let is_pro = model.id.contains("pro");
                    if is_pro {
                        if budget != -1 && !(1024..=32768).contains(&budget) {
                            return Err(Error::InvalidRequest(
                                "Gemini Pro reasoning budget must be between 1024 and 32768, or -1 for auto."
                                    .to_string(),
                            ));
                        }
                    } else if budget != 0 && budget != -1 && !(1024..=24576).contains(&budget) {
                        return Err(Error::InvalidRequest(
                            "Gemini Flash reasoning budget must be between 1024-24576, 0 to disable, or -1 for auto."
                                .to_string(),
                        ));
                    }
                    ReasoningConfig::google(budget, true)
                }
            }
        };
        config = config.with_reasoning(reasoning_config);
    }

    Ok(config)
}

fn load_image(path: &PathBuf) -> Result<Image, Box<Error>> {
    let data = fs::read(path).map_err(|e| Box::new(Error::IoError(e)))?;
    let engine = base64::engine::general_purpose::STANDARD;
    let encoded = base64::engine::Engine::encode(&engine, &data);

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("png")
        .to_lowercase();
    let mime_type = match extension.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => {
            eprintln!(
                "Warning: Unknown image extension '.{}', assuming 'image/png'. Supported: jpg, png, gif, webp",
                extension
            );
            "image/png"
        }
    };

    Ok(Image::Base64 {
        data: encoded,
        mime_type: mime_type.to_string(),
    })
}

fn print_reasoning_artifact(artifact: Option<&ReasoningArtifact>) {
    if let Some(artifact) = artifact {
        match artifact {
            ReasoningArtifact::OpenAIEncryptedReasoning { items } => {
                let summaries: Vec<&String> = items
                    .iter()
                    .filter_map(|item| item.summaries.as_ref())
                    .flatten()
                    .collect();
                if summaries.is_empty() {
                    println!("\nReasoning Artifact Present (Encrypted, No Summaries)");
                } else {
                    println!("\nReasoning Summaries:");
                    for summary in summaries {
                        println!("- {}", summary);
                    }
                }
            }
            ReasoningArtifact::OpenAIOutputItems { items } => {
                let summaries: Vec<&str> = items
                    .iter()
                    .filter(|item| item["type"] == "reasoning")
                    .filter_map(|item| item["summary"].as_array())
                    .flatten()
                    .filter_map(|summary| summary["text"].as_str())
                    .collect();
                if summaries.is_empty() {
                    println!("\nOpenAI Output Items: {} (no reasoning summaries)", items.len());
                } else {
                    println!("\nReasoning Summaries:");
                    for summary in summaries {
                        println!("- {}", summary);
                    }
                }
            }
            ReasoningArtifact::AnthropicReasoning { fragments } => {
                println!("\nAnthropic Reasoning Artifact:");
                for fragment in fragments {
                    match fragment {
                        AnthropicReasoningFragment::Thinking { thinking, signature } => {
                            println!("- Thinking (sig: {}): {} bytes", signature, thinking.len());
                        }
                        AnthropicReasoningFragment::RedactedThinking { data } => {
                            println!("- Redacted Thinking: {} bytes", data.len());
                        }
                    }
                }
            }
            ReasoningArtifact::GoogleReasoning { fragments } => {
                println!("\nGoogle Reasoning Summaries:");
                for fragment in fragments {
                    println!("- {}", fragment.summary);
                }
            }
            ReasoningArtifact::OpenRouter { details } => {
                println!("\nOpenRouter Reasoning Details:");
                for detail in details {
                    let kind = detail.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let format = detail.get("format").and_then(|v| v.as_str()).unwrap_or("unknown");
                    println!("- {kind} ({format})");
                }
            }
        }
    }
}

fn print_usage_info(
    usage: &UsageInfo,
    last_input_tokens: Option<usize>,
    accumulated_billed: f64,
    accumulated_estimate: f64,
) {
    println!("\nUsage Info (This Turn):");

    if let Some(last_tokens) = last_input_tokens {
        let delta = usage.input_tokens as i64 - last_tokens as i64;
        println!(
            "  Input Tokens (Context): {} ({:+})",
            usage.input_tokens, delta
        );
    } else {
        println!("  Input Tokens (Context): {}", usage.input_tokens);
    }

    println!("  Output Tokens: {}", usage.output_tokens);
    println!("  Input Cost:  ${:<10.6}", usage.input_cost);
    println!("  Output Cost: ${:<10.6}", usage.output_cost);

    if let Some(cached_tokens) = usage.cached_tokens {
        if cached_tokens > 0 {
            println!("  Cached Tokens Read: {}", cached_tokens);
            if let Some(cache_read_cost) = usage.cache_read_cost {
                println!("  Cache Read Cost: ${:<10.6}", cache_read_cost);
            }
        }
    }
    if let Some(cache_write_tokens) = usage.cache_write_tokens {
        if cache_write_tokens > 0 {
            println!("  Cache Write Tokens: {}", cache_write_tokens);
            if let Some(cache_write_cost) = usage.cache_write_cost {
                println!("  Cache Write Cost: ${:<10.6}", cache_write_cost);
            }
        }
    }

    println!("-----------------------------");
    println!("  Total Turn Cost (estimate): ${:<10.6}", usage.total_cost);
    if usage.billing != arko_ai::BillingKind::Metered {
        println!("  Billed ({:?}): ${:<10.6}", usage.billing, usage.billed());
    }

    if accumulated_billed > 0.0 || accumulated_estimate > 0.0 {
        println!("  Accumulated Billed: ${:<10.6}", accumulated_billed);
        if accumulated_estimate != accumulated_billed {
            println!("  Accumulated Estimate: ${:<10.6}", accumulated_estimate);
        }
    }
}