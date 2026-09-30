//! The forge: a workflow that builds workflows.
//!
//! Describe a use case, and an architect agent (Claude Opus 5.5 at xhigh
//! effort by default) designs and writes an arko_workflows binary crate for
//! it, helped by explorer agents (Claude Sonnet 5.5) that look up the
//! framework's primitives. The forge checks the result (it builds, its tests
//! pass, its CLI starts, its docs exist) and sends failures back to the
//! architect. See `README.md` next to this file.
//!
//! ```text
//! cargo run -p arko_workflows --example forge -- \
//!     --output ../my-workflow --use-case "Every morning, summarise ..."
//! ```
//!
//! By default the agents run in the installed Claude Code CLI, under its
//! login, and count against its plan usage.

mod prompts;
mod scaffold;
mod tools;

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

use arko_agents::agent_config::{AgentLLMConfig, AgentToolConfig, PathSafetyPolicy};
use arko_agents::handover::HandoverConfig;
use arko_agents::runtime::event::AgentEvent;
use arko_agents::terminal::CommandGuard;
use arko_agents::{AgentConfig, ClaudeCodeSettings, ClaudeCodeTools, PerToolConfigOverride};
use arko_ai::{AnthropicReasoningEffort, GenerationConfig, LLMClient, Model, ReasoningConfig};
use arko_workflows::prelude::*;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use serde_json::json;

type ForgeResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const ARCHITECT: &str = "architect";
const EXPLORER: &str = "explorer";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
enum Involvement {
    /// Build without asking anything; report at the end.
    None,
    /// Ask a batch of questions at the start, then work alone.
    Clarify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
enum Backend {
    /// The installed Claude Code CLI, under its login (plan usage).
    ClaudeCode,
    /// The Anthropic API (`ANTHROPIC_API_KEY`), billed per token.
    Api,
}

#[derive(Debug, Clone, Copy, ValueEnum, serde::Serialize, serde::Deserialize)]
enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl From<Effort> for AnthropicReasoningEffort {
    fn from(effort: Effort) -> Self {
        match effort {
            Effort::Low => Self::Low,
            Effort::Medium => Self::Medium,
            Effort::High => Self::High,
            Effort::Xhigh => Self::XHigh,
            Effort::Max => Self::Max,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "forge",
    about = "Describe a use case; get an arko workflow that does it"
)]
struct Args {
    /// The use case, in your own words.
    #[arg(long, conflicts_with = "use_case_file")]
    use_case: Option<String>,

    /// Read the use case from this file.
    #[arg(long)]
    use_case_file: Option<PathBuf>,

    /// Where to create the workflow's crate: a new or empty directory.
    #[arg(long, required_unless_present = "resume")]
    output: Option<PathBuf>,

    /// The crate's name; the output directory's name when omitted.
    #[arg(long)]
    name: Option<String>,

    /// The arko checkout; this repository when omitted.
    #[arg(long)]
    arko: Option<PathBuf>,

    /// The Cargo target directory builds share; the checkout's `target`
    /// when omitted.
    #[arg(long)]
    target_dir: Option<PathBuf>,

    /// The run's state directory; one under `~/.arko/forge/runs` when omitted.
    #[arg(long)]
    state_dir: Option<PathBuf>,

    /// Continue the run whose state directory this is, with the settings it
    /// started with.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Print every agent's tool calls and text, not just the architect's.
    #[arg(long)]
    verbose: bool,

    #[command(flatten)]
    settings: Settings,
}

/// The settings a run keeps for its whole life, resumes included.
#[derive(Debug, Clone, clap::Args, serde::Serialize, serde::Deserialize)]
struct Settings {
    /// How much the forge involves you while it builds.
    #[arg(long, value_enum, default_value_t = Involvement::Clarify)]
    involvement: Involvement,

    /// Where the agents run.
    #[arg(long, value_enum, default_value_t = Backend::ClaudeCode)]
    backend: Backend,

    #[arg(long, default_value = "claude-opus-5-5")]
    architect_model: String,

    #[arg(long, value_enum, default_value_t = Effort::Xhigh)]
    architect_effort: Effort,

    #[arg(long, default_value = "claude-sonnet-5-5")]
    explorer_model: String,

    #[arg(long, value_enum, default_value_t = Effort::High)]
    explorer_effort: Effort,

    /// Don't offer explorer agents: the architect looks everything up itself.
    #[arg(long)]
    no_explorers: bool,

    /// How many explorers can run at once.
    #[arg(long, default_value_t = 4)]
    max_active_calls: u32,

    /// Live runs of the new workflow the architect may make (real model
    /// usage). Off by default.
    #[arg(long, default_value_t = 0)]
    trial_runs: u32,

    /// The longest a trial run may take, in minutes.
    #[arg(long, default_value_t = 20)]
    trial_minutes: u64,

    /// The architect hands its conversation over at this many input tokens.
    #[arg(long, default_value_t = 300_000)]
    handover_at: usize,

    /// Times the forge sends failed checks back to the architect.
    #[arg(long, default_value_t = 3)]
    max_repairs: u32,

    /// The lessons file; `~/.arko/forge/lessons.md` when omitted.
    #[arg(long)]
    lessons: Option<PathBuf>,

    /// Don't read or record lessons.
    #[arg(long)]
    no_lessons: bool,

    /// Stop spending at this many dollars (API backend; a plan subscription
    /// isn't billed per token).
    #[arg(long)]
    budget: Option<f64>,
}

/// A run's fixed inputs, saved in its state directory. A resume reuses them
/// as they were, so the agents' configurations match their checkpoint even
/// when lessons were recorded or the forge's prompts changed since.
#[derive(serde::Serialize, serde::Deserialize)]
struct RunManifest {
    name: String,
    output: PathBuf,
    arko: PathBuf,
    target_dir: PathBuf,
    settings: Settings,
    architect_instructions: String,
    explorer_instructions: String,
}

#[tokio::main]
async fn main() -> ForgeResult<()> {
    env_logger::init();
    let args = Args::parse();

    // A new run scaffolds its crate; a resumed one finds it in its manifest.
    let (state_dir, manifest, use_case) = match &args.resume {
        Some(state_dir) => {
            let state_dir = std::fs::canonicalize(state_dir)?;
            let manifest: RunManifest =
                serde_json::from_slice(&std::fs::read(state_dir.join("forge-run.json"))?)?;
            let use_case = std::fs::read_to_string(manifest.output.join("USE_CASE.md"))?;
            (state_dir, manifest, use_case)
        }
        None => {
            let arko = std::fs::canonicalize(
                args.arko
                    .clone()
                    .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..")),
            )?;
            if !arko.join("docs/reference-map.md").is_file() {
                return Err(format!("{} is not an arko checkout", arko.display()).into());
            }
            let use_case = read_use_case(&args)?;
            let output = args.output.clone().expect("required without --resume");
            let name = args
                .name
                .clone()
                .or_else(|| {
                    output
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .and_then(|name| scaffold::crate_name(&name))
                .ok_or("pass --name: the crate needs a name starting with a letter")?;
            let state_dir = args.state_dir.clone().unwrap_or_else(|| {
                scaffold::forge_home().join("runs").join(format!(
                    "{name}-{}",
                    chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
                ))
            });
            scaffold::scaffold(&output, &name, &arko, &use_case)?;
            let output = std::fs::canonicalize(&output)?;
            let target_dir = args
                .target_dir
                .clone()
                .unwrap_or_else(|| arko.join("target"));
            std::fs::create_dir_all(&target_dir)?;
            std::fs::create_dir_all(&state_dir)?;
            let state_dir = std::fs::canonicalize(state_dir)?;
            let mut manifest = RunManifest {
                name,
                output,
                arko,
                target_dir: std::fs::canonicalize(target_dir)?,
                settings: args.settings.clone(),
                architect_instructions: String::new(),
                explorer_instructions: String::new(),
            };
            manifest.architect_instructions = architect_instructions(&manifest);
            manifest.explorer_instructions = fill(prompts::EXPLORER_INSTRUCTIONS, &manifest);
            std::fs::write(
                state_dir.join("forge-run.json"),
                serde_json::to_vec_pretty(&manifest)?,
            )?;
            (state_dir, manifest, use_case)
        }
    };
    let settings = &manifest.settings;
    let workspace = manifest.output.clone();
    let lessons_path = lessons_path(settings);

    let agents = Agents {
        backend: settings.backend,
        target_dir: manifest.target_dir.clone(),
    };
    let mut architect_tools = match settings.backend {
        Backend::ClaudeCode => {
            let mut tools = ClaudeCodeTools::editing();
            tools.extend(["Bash", "WebFetch", "WebSearch"].map(String::from));
            tools
        }
        Backend::Api => ["read_file", "write_file", "apply_diff", "list_files"]
            .map(String::from)
            .to_vec(),
    };
    architect_tools.push("complete_task".into());
    if settings.involvement == Involvement::Clarify {
        architect_tools.push("ask_user".into());
    }
    let mut architect = agents.config(
        ARCHITECT,
        "Workflow architect",
        &settings.architect_model,
        settings.architect_effort,
        "You design and build arko workflows: Rust binaries that run teams of agents.",
        &manifest.architect_instructions,
        architect_tools,
    )?;
    architect.handover = Some(HandoverConfig::at_input_tokens(settings.handover_at));
    architect.complete_task_summary_prompt_override = Some(prompts::SUMMARY_REQUEST.into());

    let shell = Arc::new(tools::RunShell {
        guard: CommandGuard::default(),
        target_dir: manifest.target_dir.clone(),
    });
    let client = Arc::new(match settings.backend {
        Backend::ClaudeCode => LLMClient::new(),
        Backend::Api => LLMClient::from_env()?,
    });
    let mut builder = Workflow::builder()
        .name("forge")
        .workspace(&workspace)
        .llm_client(client)
        .agent(ARCHITECT, architect)
        .persistence(PersistenceConfig::new(&state_dir));
    if !settings.no_explorers {
        let explorer_tools = match settings.backend {
            Backend::ClaudeCode => ClaudeCodeTools::read_only(),
            Backend::Api => ["read_file", "list_files"].map(String::from).to_vec(),
        };
        let explorer = agents.config(
            EXPLORER,
            "Framework explorer",
            &settings.explorer_model,
            settings.explorer_effort,
            "You find things in the arko framework and report them exactly.",
            &manifest.explorer_instructions,
            explorer_tools
                .into_iter()
                .chain(["complete_task".to_string()])
                .collect(),
        )?;
        builder = builder.agent(EXPLORER, explorer).workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller(ARCHITECT)
                .allow_target(EXPLORER)
                .max_active_calls(settings.max_active_calls),
        );
        if settings.backend == Backend::Api {
            builder = builder.agent_tool(EXPLORER, shell.clone());
        }
    }
    if settings.backend == Backend::Api {
        builder = builder.agent_tool(ARCHITECT, shell);
    }
    if let Some(path) = &lessons_path {
        builder = builder.agent_tool(
            ARCHITECT,
            Arc::new(tools::RecordLesson { path: path.clone() }),
        );
    }
    if settings.trial_runs > 0 {
        builder = builder.agent_tool(
            ARCHITECT,
            Arc::new(tools::TrialRun {
                crate_dir: workspace.clone(),
                target_dir: manifest.target_dir.clone(),
                log_dir: state_dir.join("trials"),
                allowed: settings.trial_runs,
                // Runs made before a restart count too.
                used: AtomicU32::new(
                    std::fs::read_dir(state_dir.join("trials"))
                        .map(|entries| entries.count() as u32)
                        .unwrap_or(0),
                ),
                max_minutes: settings.trial_minutes,
            }),
        );
    }
    if settings.involvement == Involvement::Clarify {
        builder = builder.user_input_handler(Arc::new(TerminalInput));
    }
    // Version control and publishing stay with the user, whatever the
    // agents' shells are asked to run.
    builder = builder.hook(
        HookPoint::PreToolExecution,
        Arc::new(|context| {
            let command = match context.tool_name.as_deref() {
                Some("Bash" | "PowerShell" | "run_shell") => context
                    .tool_params
                    .as_ref()
                    .and_then(|params| params.get("command"))
                    .and_then(|command| command.as_str()),
                _ => None,
            };
            match command.and_then(tools::refused_command) {
                Some(reason) => {
                    eprintln!(
                        "forge: refused `{}`",
                        command
                            .unwrap_or_default()
                            .chars()
                            .take(120)
                            .collect::<String>()
                    );
                    HookResult::Reject(reason.to_string())
                }
                None => HookResult::Continue,
            }
        }),
    );
    if let Some(budget) = settings.budget {
        builder = builder.budget(budget);
    }
    // Checkpoints are saved when an agent run finishes. A run stopped before
    // any finished resumes from the files in the crate instead.
    if args.resume.is_some()
        && let Ok(pointer) = std::fs::read_to_string(state_dir.join("latest-checkpoint"))
    {
        builder = builder.resume_from(state_dir.join(pointer.trim()));
    }
    let workflow = builder.build()?;

    eprintln!(
        "forge: building `{}` in {}",
        manifest.name,
        workspace.display()
    );
    eprintln!("forge: state and logs in {}", state_dir.display());
    let plan_usage = Arc::new(Mutex::new(None::<String>));
    let printer = tokio::spawn(print_events(
        workflow.subscribe_events(),
        state_dir.join("events.jsonl"),
        args.verbose,
        plan_usage.clone(),
    ));

    let providers = providers();
    let task = fill(prompts::ARCHITECT_TASK, &manifest)
        .replace("{providers}", &providers)
        .replace("{use_case}", use_case.trim());
    let earlier = workflow
        .result()
        .agent_results
        .into_iter()
        .rfind(|result| result.agent_name == ARCHITECT);
    let mut result = match earlier {
        Some(earlier) if earlier.status == AgentRunStatus::Completed => earlier,
        Some(earlier) => {
            let resume = fill(prompts::ARCHITECT_RESUME, &manifest);
            match workflow.continue_agent(&earlier, &resume).await {
                Ok(result) => result,
                Err(error) => {
                    eprintln!(
                        "forge: couldn't continue the earlier task ({error}); starting again"
                    );
                    workflow.run_agent(ARCHITECT, &resume).await?
                }
            }
        }
        None if args.resume.is_some() => {
            workflow
                .run_agent(ARCHITECT, &fill(prompts::ARCHITECT_RESUME, &manifest))
                .await?
        }
        None => workflow.run_agent(ARCHITECT, &task).await?,
    };

    // The forge's own checks; failures go back to the architect.
    let mut repairs = 0;
    let checks = loop {
        if result.status != AgentRunStatus::Completed {
            break Err(format!(
                "the architect stopped: {:?} {}",
                result.status,
                result.error.clone().unwrap_or_default()
            ));
        }
        eprintln!("forge: checking the crate");
        match scaffold::check(&workspace, &manifest.target_dir).await {
            Ok(passed) => break Ok(passed),
            Err(failure) if repairs < settings.max_repairs => {
                repairs += 1;
                eprintln!(
                    "forge: `{}` failed; back to the architect (repair {repairs} of {})",
                    failure.check, settings.max_repairs
                );
                let prompt = format!(
                    "The forge's check `{}` failed after you finished. Fix the cause, run the checks again, and call complete_task when they all pass.\n\n```text\n{}\n```",
                    failure.check, failure.output
                );
                result = workflow.continue_agent(&result, &prompt).await?;
            }
            Err(failure) => {
                break Err(format!(
                    "`{}` still fails after {} repairs:\n{}",
                    failure.check, settings.max_repairs, failure.output
                ));
            }
        }
    };
    let finished = workflow.finish().await?;
    printer.abort();

    let report = report(
        &manifest,
        &result.summary,
        &checks,
        &finished,
        plan_usage.lock().unwrap().clone(),
    );
    std::fs::write(state_dir.join("report.md"), &report)?;
    println!("\n{report}");
    if checks.is_err() {
        std::process::exit(1);
    }
    Ok(())
}

/// Fills a prompt's `{placeholders}` from the run's manifest.
fn fill(text: &str, manifest: &RunManifest) -> String {
    let settings = &manifest.settings;
    text.replace("{arko}", &manifest.arko.display().to_string())
        .replace("{target_dir}", &manifest.target_dir.display().to_string())
        .replace("{workspace}", &manifest.output.display().to_string())
        .replace("{name}", &manifest.name)
        .replace("{explorer_model}", &settings.explorer_model)
        .replace("{max_calls}", &settings.max_active_calls.to_string())
        .replace("{trial_runs}", &settings.trial_runs.to_string())
        .replace("{trial_minutes}", &settings.trial_minutes.to_string())
}

fn lessons_path(settings: &Settings) -> Option<PathBuf> {
    (!settings.no_lessons).then(|| {
        settings
            .lessons
            .clone()
            .unwrap_or_else(|| scaffold::forge_home().join("lessons.md"))
    })
}

/// The architect's instructions for this run, with the lessons recorded so
/// far.
fn architect_instructions(manifest: &RunManifest) -> String {
    let settings = &manifest.settings;
    let lessons = match lessons_path(settings) {
        Some(path) => {
            let earlier = std::fs::read_to_string(path).unwrap_or_default();
            let earlier = if earlier.trim().is_empty() {
                "(none yet)".to_string()
            } else {
                scaffold::tail(&earlier, 16_000)
            };
            format!("{}\n\n{earlier}", prompts::LESSONS_ENABLED)
        }
        None => prompts::LESSONS_DISABLED.to_string(),
    };
    let text = prompts::ARCHITECT_INSTRUCTIONS
        .replace(
            "{explorers}",
            if settings.no_explorers {
                ""
            } else {
                prompts::EXPLORERS
            },
        )
        .replace(
            "{trials}",
            if settings.trial_runs > 0 {
                prompts::TRIALS
            } else {
                ""
            },
        )
        .replace(
            "{involvement}",
            match settings.involvement {
                Involvement::None => prompts::INVOLVEMENT_NONE,
                Involvement::Clarify => prompts::INVOLVEMENT_CLARIFY,
            },
        );
    fill(&text, manifest).replace("{lessons}", &lessons)
}

/// Builds agent configurations for the chosen backend.
struct Agents {
    backend: Backend,
    target_dir: PathBuf,
}

impl Agents {
    #[allow(clippy::too_many_arguments)]
    fn config(
        &self,
        id: &str,
        name: &str,
        model: &str,
        effort: Effort,
        role: &str,
        instructions: &str,
        tools: Vec<String>,
    ) -> ForgeResult<AgentConfig> {
        let anthropic = Model::get(arko_ai::Provider::Anthropic, model)?;
        let model = match self.backend {
            Backend::ClaudeCode => Model::claude_code(anthropic),
            Backend::Api => anthropic,
        };
        let generation_config = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::anthropic_adaptive_with_effort(
                effort.into(),
            ))
            .with_prompt_caching(true);
        // Reading may reach the arko checkout; writing stays in the crate.
        let reads_anywhere: HashMap<String, PerToolConfigOverride> =
            ["Read", "Glob", "Grep", "read_file", "list_files"]
                .into_iter()
                .map(|tool| {
                    (
                        tool.to_string(),
                        PerToolConfigOverride {
                            out_of_workspace_path_policy: Some(PathSafetyPolicy::AutoApprove),
                            ..PerToolConfigOverride::default()
                        },
                    )
                })
                .filter(|(tool, _)| tools.contains(tool))
                .collect();
        let mut builder = AgentConfig::builder(
            id.to_string(),
            name.to_string(),
            AgentLLMConfig {
                model,
                generation_config,
            },
        )
        .role_description(role.to_string())
        .instructions(instructions.trim().to_string())
        .tool_config(AgentToolConfig {
            available_tool_names: tools,
            default_auto_approve_tool_use: true,
            tool_specific_configs: reads_anywhere,
            ..AgentToolConfig::default()
        });
        if self.backend == Backend::ClaudeCode {
            builder = builder.claude_code(ClaudeCodeSettings {
                env: BTreeMap::from([(
                    "CARGO_TARGET_DIR".to_string(),
                    self.target_dir.display().to_string(),
                )]),
                ..ClaudeCodeSettings::default()
            });
        }
        Ok(builder.build()?)
    }
}

fn read_use_case(args: &Args) -> ForgeResult<String> {
    let use_case = match (&args.use_case, &args.use_case_file) {
        (Some(text), _) => text.clone(),
        (None, Some(path)) => std::fs::read_to_string(path)?,
        (None, None) => {
            if !io::stdin().is_terminal() {
                return Err("pass --use-case or --use-case-file".into());
            }
            eprintln!("Describe the workflow you want (finish with an empty line):");
            read_paragraph()?
        }
    };
    if use_case.trim().is_empty() {
        return Err("the use case is empty".into());
    }
    Ok(use_case)
}

/// Lines from stdin up to the first empty line after some text.
fn read_paragraph() -> io::Result<String> {
    let mut text = String::new();
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() && !text.trim().is_empty() {
            break;
        }
        text.push_str(&line);
        text.push('\n');
    }
    Ok(text.trim().to_string())
}

/// The provider access this machine has, by name only.
fn providers() -> String {
    let mut found = Vec::new();
    let on_path = |program: &str| {
        std::env::var_os("PATH").is_some_and(|path| {
            std::env::split_paths(&path).any(|dir| {
                dir.join(program).is_file() || dir.join(format!("{program}.exe")).is_file()
            })
        })
    };
    if on_path("claude") {
        found.push("the Claude Code CLI (`claude`, a plan login: Claude models through `Model::claude_code`)");
    }
    for (variable, provider) in [
        ("ANTHROPIC_API_KEY", "an Anthropic API key"),
        ("OPENAI_API_KEY", "an OpenAI API key"),
        ("GEMINI_API_KEY", "a Google Gemini API key"),
        ("OPENROUTER_API_KEY", "an OpenRouter API key"),
        ("VOYAGE_API_KEY", "a Voyage API key"),
    ] {
        if std::env::var_os(variable).is_some() {
            found.push(provider);
        }
    }
    if arko_ai::CodexAuthFile::default_path().is_some_and(|path| path.is_file()) {
        found.push("a Codex ChatGPT login (OpenAI models through ChatGPT OAuth)");
    }
    if found.is_empty() {
        "none detected".into()
    } else {
        found.join("; ")
    }
}

/// Answers `ask_user` at the terminal.
#[derive(Debug)]
struct TerminalInput;

#[async_trait]
impl UserInputHandler for TerminalInput {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String> {
        tokio::task::spawn_blocking(move || {
            eprintln!(
                "\n──── {} asks ────\n{}\n",
                request.agent_name, request.questions
            );
            eprintln!("Your answer (finish with an empty line):");
            io::stderr().flush().map_err(|error| error.to_string())?;
            let answer = read_paragraph().map_err(|error| error.to_string())?;
            if answer.is_empty() {
                Err("no answer".to_string())
            } else {
                Ok(answer)
            }
        })
        .await
        .map_err(|error| error.to_string())?
    }
}

/// Prints the run as it goes, and logs every event to `log`.
async fn print_events(
    mut events: tokio::sync::broadcast::Receiver<WorkflowEvent>,
    log: PathBuf,
    verbose: bool,
    plan_usage: Arc<Mutex<Option<String>>>,
) {
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .ok();
    let mut requests: HashMap<String, usize> = HashMap::new();
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => break,
        };
        if let Some(file) = &mut log {
            let line = json!({
                "time": event.timestamp,
                "agent": event.agent_name,
                "run": event.agent_run_id,
                "event": event.event,
            });
            let _ = writeln!(file, "{line}");
        }
        let who = &event.agent_name;
        let shown = verbose || who == ARCHITECT;
        match event.event {
            AgentEvent::ToolCallFormatted {
                tool_name,
                parameters,
                ..
            } if shown => println!("[{who}] → {tool_name} {}", brief(&tool_name, &parameters)),
            AgentEvent::ToolResult {
                tool_name, output, ..
            } if shown && tool_name == "run_workflow_agent" => {
                println!("[{who}] ← explorer: {}", cut(&output, 400))
            }
            AgentEvent::ToolRejected {
                tool_name, reason, ..
            } if shown => println!(
                "[{who}] ✗ {tool_name}: {}",
                cut(&reason.unwrap_or_default(), 300)
            ),
            AgentEvent::AssistantResponseChunk(text) if shown && !text.trim().is_empty() => {
                println!(
                    "[{who}]   {}",
                    cut(text.trim(), if verbose { 4_000 } else { 600 })
                )
            }
            AgentEvent::ModelCallUsageReported(usage) => {
                let count = requests.entry(who.clone()).or_default();
                *count += 1;
                if shown && (*count).is_multiple_of(10) {
                    println!(
                        "[{who}]   ({count} requests; this one {}k input tokens)",
                        usage.input_tokens / 1_000
                    );
                }
            }
            AgentEvent::PlanUsageReported(usage) => {
                let text = format!(
                    "five-hour {}%, seven-day {}%",
                    usage
                        .primary
                        .map(|window| format!("{:.0}", window.used_percent))
                        .unwrap_or("?".into()),
                    usage
                        .secondary
                        .map(|window| format!("{:.0}", window.used_percent))
                        .unwrap_or("?".into())
                );
                let mut last = plan_usage.lock().unwrap();
                if last.as_deref() != Some(text.as_str()) {
                    println!("[plan usage] {text}");
                    *last = Some(text);
                }
            }
            AgentEvent::HandoverRequested { input_tokens } => {
                println!(
                    "[{who}] handing over at {}k input tokens",
                    input_tokens / 1_000
                )
            }
            AgentEvent::ConversationHandedOver { handovers, .. } => {
                println!("[{who}] handover {handovers} done; continuing in a fresh conversation")
            }
            AgentEvent::ErrorEncountered { error_message, .. } => {
                println!("[{who}] error: {}", cut(&error_message, 600))
            }
            AgentEvent::RefusalReceived(refusal) => {
                println!("[{who}] refusal: {:?}", refusal.category)
            }
            AgentEvent::RefusalFallback { model_id } => {
                println!("[{who}] refusal fallback to {model_id}")
            }
            _ => {}
        }
    }
}

/// A tool call's parameters, briefly: the path or command when it has one.
fn brief(tool: &str, parameters: &HashMap<String, serde_json::Value>) -> String {
    for field in ["file_path", "path", "command", "pattern", "url", "agent"] {
        if let Some(value) = parameters.get(field).and_then(|value| value.as_str()) {
            let extra = match (tool, parameters.get("prompt").and_then(|p| p.as_str())) {
                ("run_workflow_agent", Some(prompt)) => format!(": {}", cut(prompt, 240)),
                _ => String::new(),
            };
            return format!("{}{extra}", cut(value, 200));
        }
    }
    cut(&serde_json::to_string(parameters).unwrap_or_default(), 200)
}

fn cut(text: &str, limit: usize) -> String {
    let single: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if single.chars().count() <= limit {
        single
    } else {
        format!("{}…", single.chars().take(limit).collect::<String>())
    }
}

fn report(
    manifest: &RunManifest,
    summary: &str,
    checks: &Result<Vec<String>, String>,
    result: &WorkflowResult,
    plan_usage: Option<String>,
) -> String {
    let mut text = format!("# Forge report: `{}`\n\n", manifest.name);
    match checks {
        Ok(passed) => text.push_str(&format!(
            "The workflow is ready in `{}`.\n\nChecks passed: {}.\n\n",
            manifest.output.display(),
            passed.join(", ")
        )),
        Err(problem) => text.push_str(&format!(
            "The workflow in `{}` is not finished: {problem}\n\n",
            manifest.output.display()
        )),
    }
    text.push_str(&format!(
        "## The architect's summary\n\n{}\n\n## Usage\n\n",
        summary.trim()
    ));
    text.push_str("| Agent | Runs | Requests | Input tokens | Output tokens | List-price estimate |\n|---|---|---|---|---|---|\n");
    let mut by_agent: BTreeMap<&str, (usize, usize, usize, usize, f64)> = BTreeMap::new();
    for run in &result.agent_results {
        let usage = run.task.total_usage();
        let entry = by_agent.entry(&run.agent_name).or_default();
        entry.0 += 1;
        entry.1 += run.task.usage_history.len();
        entry.2 += usage.input_tokens;
        entry.3 += usage.output_tokens;
        entry.4 += usage.total_cost;
    }
    for (agent, (runs, requests, input, output, cost)) in by_agent {
        text.push_str(&format!(
            "| {agent} | {runs} | {requests} | {input} | {output} | ${cost:.2} |\n"
        ));
    }
    text.push_str(&format!("\nBilled: ${:.2}", result.total_cost));
    if let Some(plan_usage) = plan_usage {
        text.push_str(&format!(" (plan usage at the end: {plan_usage})"));
    }
    text.push('\n');
    text
}
