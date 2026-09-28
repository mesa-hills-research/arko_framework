//! Command-line client for `arko_ai`: one-shot text generation and an
//! interactive chat against any configured provider.
//!
//! Credentials come from the environment (and a `.env` file) through
//! `LLMClient::from_env`: ChatGPT OAuth or `OPENAI_API_KEY`,
//! `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY`, and
//! `OPENROUTER_API_KEY`. `--codex-auth` (the default Codex `auth.json`) or
//! `--codex-auth-json PATH` takes OpenAI's ChatGPT OAuth credentials from a
//! Codex `auth.json` instead, kept in step with the file: every token
//! refresh reads it again first, and rotated tokens are saved to it at once.
//!
//! ```text
//! cargo run -p arko_ai --example llm_client_cli -- models
//! cargo run -p arko_ai --example llm_client_cli -- text -m gpt-5.5 -p "Hello"
//! cargo run -p arko_ai --example llm_client_cli -- text -m gpt-6-astra -p "Hello" --codex-auth
//! cargo run -p arko_ai --example llm_client_cli -- chat -m anthropic/claude-sonnet-5 --reasoning-level high
//! ```
//!
//! A model is a registered id (see `models`), an alias of one, or an
//! OpenRouter slug (anything with a `/`). The flags are parsed here; which
//! efforts, levels, budgets and parameters a model accepts is checked by the
//! library when the request is built.

use arko_ai::{
    AnthropicReasoningEffort, CodexAuthFile, Content, Error, GeminiThinkingLevel, GenerationConfig,
    GenerationResponse, Image, LLMClient, Model, OpenAIReasoningMode, OpenRouterReasoning,
    Provider, ReasoningArtifact, ReasoningConfig, ReasoningEffort, StreamEvent, UsageInfo,
    reasoning::{AnthropicReasoningFragment, OpenAIEffort},
};
use clap::{Args, Parser, Subcommand};
use dotenv::dotenv;
use futures::StreamExt;
use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Use the ChatGPT OAuth login in the Codex auth.json
    /// ($CODEX_HOME/auth.json, else ~/.codex/auth.json) for OpenAI instead of
    /// the environment; token refreshes keep the file up to date
    #[arg(long, global = true)]
    codex_auth: bool,

    /// --codex-auth with the auth.json at PATH
    #[arg(
        long,
        value_name = "PATH",
        global = true,
        conflicts_with = "codex_auth"
    )]
    codex_auth_json: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt (optionally streaming)
    Text {
        #[command(flatten)]
        generation: GenerationArgs,

        /// The prompt to generate from
        #[arg(short, long)]
        prompt: String,

        /// Image input as file path
        #[arg(short, long)]
        image: Option<PathBuf>,

        /// Stream the response
        #[arg(long)]
        stream: bool,
    },

    /// Start an interactive, streaming conversation
    Chat {
        #[command(flatten)]
        generation: GenerationArgs,
    },

    /// List the registered models and the aliases accepted for them
    Models,
}

/// Model and generation settings shared by `text` and `chat`.
#[derive(Args)]
struct GenerationArgs {
    /// A registered model id or alias (see `models`), or an OpenRouter slug
    /// such as anthropic/claude-sonnet-5, openrouter/auto or @preset/name
    #[arg(short, long)]
    model: String,

    /// The system message
    #[arg(short, long)]
    system: Option<String>,

    /// The temperature (0.0-2.0), for models that accept one
    #[arg(short, long)]
    temperature: Option<f64>,

    /// The maximum number of tokens to generate
    #[arg(short = 'n', long)]
    max_tokens: Option<usize>,

    /// Enable reasoning with the model's defaults (implied by the other
    /// reasoning options)
    #[arg(short, long)]
    reasoning: bool,

    /// An effort (none, minimal, low, medium, high, xhigh, max) for OpenAI,
    /// Claude adaptive thinking and OpenRouter; a thinking level (minimal,
    /// low, medium, high) for Gemini; or a thinking token budget (Claude
    /// manual thinking, Gemini, OpenRouter)
    #[arg(long)]
    reasoning_level: Option<String>,

    /// Claude effort (low, medium, high, xhigh, max), with adaptive thinking
    /// or together with a --reasoning-level token budget (Claude Opus 4.5);
    /// otherwise the same setting as --reasoning-level, which it must match
    /// when both are given
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Reasoning mode (standard or pro) for GPT-5.6 and GPT-6, directly or
    /// through OpenRouter; GPT-6 Pro is gpt-6-* with --reasoning-mode pro
    #[arg(long)]
    reasoning_mode: Option<String>,

    /// Request the 1M-token context window (Claude 4.6 and later)
    #[arg(long)]
    long_context: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<arko_ai::Error>> {
    dotenv().ok();
    env_logger::init();

    let cli = Cli::parse();
    let client = create_client(&cli)?;

    match cli.command {
        Commands::Models => print_models(),

        Commands::Text {
            generation,
            prompt,
            image,
            stream,
        } => {
            let model = get_model(&generation.model)?;
            let config = build_config(&model, &generation)?;

            let mut conversation = client.new_conversation(model);
            if let Some(system_msg) = &generation.system {
                conversation = conversation.with_system_message(system_msg);
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
                        Ok(
                            StreamEvent::ToolCallStart { .. }
                            | StreamEvent::ToolCallArgumentsDelta { .. },
                        ) => {}
                        Ok(StreamEvent::Refusal(refusal)) => {
                            println!("\n\nThe response was {refusal}");
                        }
                        Ok(StreamEvent::End(boxed_resp)) => {
                            let resp = *boxed_resp;
                            println!("\n");
                            print_served_model(&resp);
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
                print_served_model(&response);
                print_usage_info(&response.usage, None, 0.0, 0.0);
            }
        }

        Commands::Chat { generation } => {
            let model = get_model(&generation.model)?;
            let config = build_config(&model, &generation)?;

            let mut conversation = client.new_conversation(model);
            if let Some(system_msg) = &generation.system {
                conversation = conversation.with_system_message(system_msg);
            }

            println!(
                "Starting chat with model '{}' (type 'exit' to quit)",
                conversation.model.id
            );
            if let Some(sys) = &conversation.system_message {
                println!("System: {}", sys);
            }
            println!("---");

            // Billed amount (zero for subscription calls; OpenRouter's
            // reported charge) and the estimate at API list prices,
            // accumulated separately.
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
                        Ok(
                            StreamEvent::ToolCallStart { .. }
                            | StreamEvent::ToolCallArgumentsDelta { .. },
                        ) => {}
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
                    if resp
                        .refusal
                        .as_ref()
                        .is_some_and(|refusal| !refusal.source.keeps_turn())
                    {
                        conversation.messages.pop();
                    } else {
                        conversation.add_assistant_message_from_response(resp.clone());
                    }
                    total_billed += resp.usage.billed();
                    total_estimate += resp.usage.total_cost;
                    print_served_model(&resp);
                    print_usage_info(&resp.usage, last_input_tokens, total_billed, total_estimate);
                    last_input_tokens = Some(resp.usage.input_tokens);
                } else {
                    conversation.messages.pop();
                    println!("Turn failed. Please try again.");
                }
                println!("---");
            }
            print!(
                "Chat ended. Final accumulated billed cost: ${:.6}",
                total_billed
            );
            if total_estimate > 0.0 {
                print!(" (estimate at list prices: ${:.6})", total_estimate);
            }
            println!();
        }
    }

    Ok(())
}

/// `LLMClient::from_env`, with OpenAI from a Codex `auth.json` when asked.
fn create_client(cli: &Cli) -> Result<LLMClient, Error> {
    let client = LLMClient::from_env()?;
    let path = match (&cli.codex_auth_json, cli.codex_auth) {
        (Some(path), _) => path.clone(),
        (None, true) => CodexAuthFile::default_path().ok_or_else(|| {
            Error::AuthenticationFailed(
                "No home directory to find ~/.codex/auth.json in; pass --codex-auth-json".into(),
            )
        })?,
        (None, false) => return Ok(client),
    };
    let file = CodexAuthFile::load(&path).map_err(|error| {
        Error::AuthenticationFailed(format!("Cannot use {}: {error}", path.display()))
    })?;
    client.with_codex_auth_file(&file)
}

/// The id without a trailing release date (`-2025-08-07`, `-20250514`) or
/// `-preview`, the part the dated and preview registry ids add.
fn undated(id: &str) -> &str {
    let bytes = id.as_bytes();
    let dated = |date_len: usize, pattern: &[usize]| {
        id.len() > date_len + 1
            && bytes[id.len() - date_len - 1] == b'-'
            && bytes[id.len() - date_len..]
                .iter()
                .enumerate()
                .all(|(i, b)| {
                    if pattern.contains(&i) {
                        *b == b'-'
                    } else {
                        b.is_ascii_digit()
                    }
                })
    };
    if dated(10, &[4, 7]) {
        &id[..id.len() - 11]
    } else if dated(8, &[]) {
        &id[..id.len() - 9]
    } else {
        id.strip_suffix("-preview").unwrap_or(id)
    }
}

/// Every alias that names exactly one registered model and isn't itself a
/// registered id: the undated id and, for Claude, the version written with
/// dots (`claude-opus-4.5`, `claude-3.7-sonnet`).
fn aliases() -> HashMap<String, Vec<Model>> {
    let models = Model::all();
    let mut aliases: HashMap<String, Vec<Model>> = HashMap::new();
    for model in &models {
        let short = undated(&model.id);
        let mut names = vec![short.to_string()];
        if short.starts_with("claude-") {
            // A '-' between two digits is a version separator.
            let dotted: String = short
                .char_indices()
                .map(|(i, c)| {
                    let digit_at =
                        |j: usize| short.as_bytes().get(j).is_some_and(u8::is_ascii_digit);
                    if c == '-' && i > 0 && digit_at(i - 1) && digit_at(i + 1) {
                        '.'
                    } else {
                        c
                    }
                })
                .collect();
            names.push(dotted);
        }
        names.sort();
        names.dedup();
        for name in names {
            if !models.iter().any(|registered| registered.id == name) {
                aliases.entry(name).or_default().push(model.clone());
            }
        }
    }
    aliases
}

fn get_model(name: &str) -> Result<Model, Error> {
    // OpenRouter slugs: author/model, ~author/alias-latest, @preset/name.
    if name.contains('/') {
        return Ok(Model::openrouter(name));
    }
    if let Some(model) = Model::find(name) {
        return Ok(model);
    }
    match aliases().remove(name).as_deref() {
        Some([model]) => Ok(model.clone()),
        Some(models) => Err(Error::ModelNotSupported(format!(
            "'{name}' names several models; use one of: {}",
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        None => Err(Error::ModelNotSupported(format!(
            "Unknown model '{name}'. Run the `models` command for registered ids and their aliases, or pass an OpenRouter slug (author/model)."
        ))),
    }
}

fn print_models() {
    let mut unique: HashMap<String, Vec<String>> = HashMap::new();
    for (alias, models) in aliases() {
        if let [model] = models.as_slice() {
            unique.entry(model.id.clone()).or_default().push(alias);
        }
    }
    for model in Model::all() {
        let mut names = unique.remove(&model.id).unwrap_or_default();
        names.sort();
        if names.is_empty() {
            println!("{:<10} {}", format!("{:?}", model.provider), model.id);
        } else {
            println!(
                "{:<10} {} (also: {})",
                format!("{:?}", model.provider),
                model.id,
                names.join(", ")
            );
        }
    }
    println!("OpenRouter any author/model slug, ~author/alias-latest or @preset/name");
}

fn invalid(message: String) -> Error {
    Error::InvalidRequest(message)
}

fn build_config(model: &Model, args: &GenerationArgs) -> Result<GenerationConfig, Error> {
    let mut config = GenerationConfig::default();

    if let Some(temp) = args.temperature {
        // OpenAI reasoning models and Gemini reject a temperature with an
        // error; these Claude models get none, so say so.
        if model.provider == Provider::Anthropic && model.omits_anthropic_sampling_parameters() {
            eprintln!(
                "Warning: {} takes no temperature; --temperature is not sent.",
                model.id
            );
        }
        config = config.with_temperature(temp);
    }
    if let Some(tokens) = args.max_tokens {
        config = config.with_max_tokens(tokens);
    }
    if args.long_context {
        config = config.with_long_context(true);
    }

    let mode = match args.reasoning_mode.as_deref() {
        None => None,
        Some("standard") => Some(OpenAIReasoningMode::Standard),
        Some("pro") => Some(OpenAIReasoningMode::Pro),
        Some(other) => {
            return Err(invalid(format!(
                "Reasoning mode must be 'standard' or 'pro', not '{other}'."
            )));
        }
    };
    match (mode, &model.provider) {
        (None, _) | (Some(_), Provider::OpenRouter) => {}
        (Some(mode), Provider::OpenAI) => config = config.with_openai_reasoning_mode(mode),
        (Some(_), _) => {
            return Err(invalid(
                "--reasoning-mode applies to OpenAI GPT-5.6 and GPT-6 models, directly or through OpenRouter.".into(),
            ));
        }
    }

    let reasoning = args.reasoning
        || args.reasoning_level.is_some()
        || args.reasoning_effort.is_some()
        || (mode.is_some() && model.provider == Provider::OpenRouter);
    if reasoning {
        config = config.with_reasoning(reasoning_config(model, args, mode)?);
    }
    Ok(config)
}

/// The reasoning configuration the flags ask for. Unset options keep the
/// model's defaults where the provider has them; OpenAI and OpenRouter
/// default to a medium effort so reasoning summaries are requested.
fn reasoning_config(
    model: &Model,
    args: &GenerationArgs,
    mode: Option<OpenAIReasoningMode>,
) -> Result<ReasoningConfig, Error> {
    let level = args.reasoning_level.as_deref();
    let effort = args.reasoning_effort.as_deref();
    // Outside a Claude token budget, the two flags name the same setting.
    let setting = || match (level, effort) {
        (Some(level), Some(effort)) if level != effort => Err(invalid(format!(
            "--reasoning-level '{level}' and --reasoning-effort '{effort}' disagree; give one of them."
        ))),
        (level, effort) => Ok(level.or(effort)),
    };
    Ok(match model.provider {
        Provider::OpenAI => {
            let effort = match setting()?.unwrap_or("medium") {
                "none" => OpenAIEffort::None,
                "minimal" => OpenAIEffort::Minimal,
                "low" => OpenAIEffort::Low,
                "medium" => OpenAIEffort::Medium,
                "high" => OpenAIEffort::High,
                "xhigh" => OpenAIEffort::XHigh,
                "max" => OpenAIEffort::Max,
                other => {
                    return Err(invalid(format!(
                        "OpenAI reasoning effort must be none, minimal, low, medium, high, xhigh or max, not '{other}'."
                    )));
                }
            };
            ReasoningConfig::openai(effort, true)
        }
        Provider::Anthropic => {
            // A number is a manual thinking budget, which an effort may
            // accompany (Claude Opus 4.5); a name is an effort, for adaptive
            // thinking where the model supports it.
            if let Some(budget) = level.and_then(|level| level.parse::<usize>().ok()) {
                match effort.map(anthropic_effort).transpose()? {
                    Some(effort) => ReasoningConfig::anthropic_with_effort(budget, effort),
                    None => ReasoningConfig::anthropic(budget),
                }
            } else {
                let effort = setting()?.map(anthropic_effort).transpose()?;
                match (model.supports_anthropic_adaptive_thinking(), effort) {
                    (true, Some(effort)) => ReasoningConfig::anthropic_adaptive_with_effort(effort),
                    (true, None) => ReasoningConfig::anthropic_adaptive(),
                    (false, Some(effort)) => ReasoningConfig::anthropic_with_effort(1024, effort),
                    (false, None) => ReasoningConfig::anthropic(1024),
                }
            }
        }
        Provider::Google => match setting()? {
            // -1: the model's default level or dynamic budget.
            None => ReasoningConfig::google(-1, true),
            Some(level) => match level.parse::<isize>() {
                Ok(budget) => ReasoningConfig::google(budget, true),
                Err(_) => {
                    let level = match level {
                        "minimal" => GeminiThinkingLevel::Minimal,
                        "low" => GeminiThinkingLevel::Low,
                        "medium" => GeminiThinkingLevel::Medium,
                        "high" => GeminiThinkingLevel::High,
                        other => {
                            return Err(invalid(format!(
                                "Gemini thinking takes a level (minimal, low, medium, high) or a token budget, not '{other}'."
                            )));
                        }
                    };
                    ReasoningConfig::gemini(level, true)
                }
            },
        },
        Provider::OpenRouter => {
            // An effort name, or a number as a reasoning token budget.
            let mut reasoning = match setting()? {
                None if !args.reasoning => OpenRouterReasoning::default(),
                None | Some("medium") => OpenRouterReasoning::effort(ReasoningEffort::Medium),
                Some("none") => OpenRouterReasoning::effort(ReasoningEffort::None),
                Some("minimal") => OpenRouterReasoning::effort(ReasoningEffort::Minimal),
                Some("low") => OpenRouterReasoning::effort(ReasoningEffort::Low),
                Some("high") => OpenRouterReasoning::effort(ReasoningEffort::High),
                Some("xhigh") => OpenRouterReasoning::effort(ReasoningEffort::XHigh),
                Some("max") => OpenRouterReasoning::effort(ReasoningEffort::Max),
                Some(budget) => OpenRouterReasoning::max_tokens(budget.parse().map_err(|_| {
                    invalid(
                        "OpenRouter reasoning takes an effort (none..max) or a token budget".into(),
                    )
                })?),
            };
            if let Some(mode) = mode {
                reasoning = reasoning.with_mode(mode);
            }
            ReasoningConfig::openrouter(reasoning)
        }
    })
}

fn anthropic_effort(name: &str) -> Result<AnthropicReasoningEffort, Error> {
    Ok(match name {
        "low" => AnthropicReasoningEffort::Low,
        "medium" => AnthropicReasoningEffort::Medium,
        "high" => AnthropicReasoningEffort::High,
        "xhigh" => AnthropicReasoningEffort::XHigh,
        "max" => AnthropicReasoningEffort::Max,
        other => {
            return Err(invalid(format!(
                "Claude effort must be low, medium, high, xhigh or max, not '{other}'."
            )));
        }
    })
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
                    println!(
                        "\nOpenAI Output Items: {} (no reasoning summaries)",
                        items.len()
                    );
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
                        AnthropicReasoningFragment::Thinking {
                            thinking,
                            signature,
                        } => {
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
                    let kind = detail
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let format = detail
                        .get("format")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    println!("- {kind} ({format})");
                }
            }
        }
    }
}
/// Which model served the call when the provider routed it elsewhere (for
/// example `openrouter/auto`, an OpenRouter fallback, or a ChatGPT backend
/// reroute), and the upstream provider of a routed OpenRouter call.
fn print_served_model(response: &GenerationResponse) {
    let usage = &response.model_call_usage;
    if response.rerouted() {
        let served = usage.served_model.as_deref().unwrap_or(&usage.model);
        match &usage.requested_model {
            Some(requested) => println!("Served by {served} (requested {requested})"),
            None => println!("Served by {served}"),
        }
    }
    if let Some(provider) = &usage.upstream_provider {
        println!("Upstream provider: {provider}");
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
    if let Some(reasoning_tokens) = usage.reasoning_tokens.filter(|tokens| *tokens > 0) {
        println!("  Reasoning Tokens: {}", reasoning_tokens);
    }
    if let Some(cached_tokens) = usage.cached_tokens.filter(|tokens| *tokens > 0) {
        println!("  Cached Tokens Read: {}", cached_tokens);
        if let Some(cache_read_cost) = usage.cache_read_cost {
            println!("  Cache Read Cost: ${:<10.6}", cache_read_cost);
        }
    }
    if let Some(cache_write_tokens) = usage.cache_write_tokens.filter(|tokens| *tokens > 0) {
        println!("  Cache Write Tokens: {}", cache_write_tokens);
        if let Some(cache_write_cost) = usage.cache_write_cost {
            println!("  Cache Write Cost: ${:<10.6}", cache_write_cost);
        }
    }

    println!("-----------------------------");
    // A provider-reported charge (OpenRouter) comes without an estimate
    // unless the model's prices are known.
    let reported = usage.billing == arko_ai::BillingKind::ProviderReported;
    if !reported || usage.total_cost > 0.0 {
        println!("  Input Cost:  ${:<10.6}", usage.input_cost);
        println!("  Output Cost: ${:<10.6}", usage.output_cost);
        println!("  Total Turn Cost (estimate): ${:<10.6}", usage.total_cost);
    }
    if let Some(reported_cost) = usage.reported_cost {
        println!("  Reported Cost: ${:<10.6}", reported_cost);
    }
    if let Some(upstream_cost) = usage.upstream_cost {
        println!(
            "  Upstream Cost (your provider key): ${:<10.6}",
            upstream_cost
        );
    }
    if usage.billing != arko_ai::BillingKind::Metered {
        println!("  Billed ({:?}): ${:<10.6}", usage.billing, usage.billed());
    }

    if accumulated_billed > 0.0 || accumulated_estimate > 0.0 {
        println!("  Accumulated Billed: ${:<10.6}", accumulated_billed);
        if accumulated_estimate > 0.0 && accumulated_estimate != accumulated_billed {
            println!("  Accumulated Estimate: ${:<10.6}", accumulated_estimate);
        }
    }
}
