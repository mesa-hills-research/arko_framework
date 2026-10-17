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
//! login, and count against its plan usage. With `--deliberate`, one-way-door
//! decisions go through a deliberation (`deliberation.rs`), with proposers on
//! several models (OpenAI's through the ChatGPT login when there is one).
//! `--update DIR` changes an existing crate instead of building a new one.

mod deliberation;
mod prompts;
mod scaffold;
mod tools;

use std::collections::HashSet;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use arko_agents::agent_config::{AgentLLMConfig, AgentToolConfig, PathSafetyPolicy};
use arko_agents::handover::HandoverConfig;
use arko_agents::runtime::event::AgentEvent;
use arko_agents::{AgentConfig, ClaudeCodeSettings, ClaudeCodeTools, PerToolConfigOverride};
use arko_ai::{
    AnthropicReasoningEffort, GeminiThinkingLevel, GenerationConfig, LLMClient, Model,
    OpenAIEffort, Provider, ReasoningConfig, SubscriptionPlan,
};
use arko_workflows::prelude::*;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
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

impl From<Effort> for OpenAIEffort {
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

impl From<Effort> for GeminiThinkingLevel {
    fn from(effort: Effort) -> Self {
        match effort {
            Effort::Low => Self::Low,
            Effort::Medium => Self::Medium,
            Effort::High | Effort::Xhigh | Effort::Max => Self::High,
        }
    }
}

/// A model and its reasoning effort, written `id` or `id:effort`
/// (`gpt-6.1-sol:xhigh`); the effort is `xhigh` when omitted. Anthropic
/// models run on the `--backend`; OpenAI models through the ChatGPT login
/// (or `OPENAI_API_KEY`); Google models with `GEMINI_API_KEY`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct ModelSpec {
    id: String,
    effort: Effort,
}

impl ModelSpec {
    fn new(id: &str, effort: Effort) -> ForgeResult<Self> {
        Ok(format!("{id}:{}", effort_name(effort)).parse::<Self>()?)
    }
}

fn effort_name(effort: Effort) -> String {
    effort
        .to_possible_value()
        .map(|value| value.get_name().to_string())
        .unwrap_or_default()
}

impl std::str::FromStr for ModelSpec {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let (id, effort) = match text.trim().split_once(':') {
            Some((id, effort)) => (
                id.trim(),
                <Effort as ValueEnum>::from_str(effort.trim(), true).map_err(|_| {
                    format!("`{effort}` isn't an effort. Use low, medium, high, xhigh or max")
                })?,
            ),
            None => (text.trim(), Effort::Xhigh),
        };
        if Model::find(id).is_none() {
            return Err(format!("`{id}` isn't a model arko knows"));
        }
        Ok(Self {
            id: id.to_string(),
            effort,
        })
    }
}

impl std::fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.id, effort_name(self.effort))
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "forge",
    about = "Describe a use case and get an arko workflow that does it"
)]
struct Args {
    /// The use case, in your own words.
    #[arg(long, conflicts_with = "use_case_file")]
    use_case: Option<String>,

    /// Read the use case from this file.
    #[arg(long)]
    use_case_file: Option<PathBuf>,

    /// Where to create the workflow's crate: a new or empty directory.
    #[arg(long, required_unless_present_any = ["resume", "update"])]
    output: Option<PathBuf>,

    /// Change the existing workflow crate in this directory instead: the use
    /// case is then the change to make.
    #[arg(long, conflicts_with_all = ["output", "name"])]
    update: Option<PathBuf>,

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

    /// The most all trial runs together may spend, in dollars at list
    /// prices. Each run reserves its cap in full; without this, trials draw
    /// on `--budget`.
    #[arg(long)]
    #[serde(default)]
    trial_budget: Option<f64>,

    /// The architect hands its conversation over at this many input tokens.
    #[arg(long, default_value_t = 300_000)]
    handover_at: usize,

    /// Deliberation agents hand their conversations over at this many input
    /// tokens (at most `--handover-at`); 200,000 when omitted.
    #[arg(long, value_name = "TOKENS")]
    #[serde(default)]
    deliberation_handover_at: Option<usize>,

    /// Agents on OpenAI models hand their conversations over at this many
    /// input tokens, in place of the thresholds above (at most the ChatGPT
    /// login's input limit); three quarters of that limit, or less when the
    /// thresholds above are lower, when omitted.
    #[arg(long, value_name = "TOKENS")]
    #[serde(default)]
    openai_handover_at: Option<usize>,

    /// Times the forge sends failed checks back to the architect.
    #[arg(long, default_value_t = 3)]
    max_repairs: u32,

    /// The lessons file; `~/.arko/forge/lessons.md` when omitted.
    #[arg(long)]
    lessons: Option<PathBuf>,

    /// Don't read or record lessons.
    #[arg(long)]
    no_lessons: bool,

    /// Stop spending at this many dollars: the API backend, or a Claude
    /// Code CLI billed per token (`--metered`). A plan subscription isn't
    /// billed per token.
    #[arg(long)]
    budget: Option<f64>,

    /// Stop at this many percentage points of a subscription plan's
    /// five-hour usage window (Claude Code, ChatGPT sign-in), counted from
    /// its level when the run started; each plan's window is counted on its
    /// own. Other use of the same plan meanwhile counts too.
    #[arg(long, value_name = "POINTS")]
    #[serde(default)]
    plan_budget_5h: Option<f64>,

    /// The same for each plan's seven-day usage window.
    #[arg(long, value_name = "POINTS")]
    #[serde(default)]
    plan_budget_7d: Option<f64>,

    /// The Claude Code plan's own five-hour limit, in points; replaces
    /// `--plan-budget-5h` for that plan.
    #[arg(long, value_name = "POINTS")]
    #[serde(default)]
    claude_code_budget_5h: Option<f64>,

    /// The Claude Code plan's own seven-day limit, in points.
    #[arg(long, value_name = "POINTS")]
    #[serde(default)]
    claude_code_budget_7d: Option<f64>,

    /// The ChatGPT plan's own five-hour limit, in points.
    #[arg(long, value_name = "POINTS")]
    #[serde(default)]
    chatgpt_budget_5h: Option<f64>,

    /// The ChatGPT plan's own seven-day limit, in points.
    #[arg(long, value_name = "POINTS")]
    #[serde(default)]
    chatgpt_budget_7d: Option<f64>,

    /// The Claude Code CLI is billed per token (an API key, cloud credits, a
    /// host-managed provider), not covered by a plan: count its usage in
    /// costs and budgets.
    #[arg(long)]
    #[serde(default)]
    metered: bool,

    #[command(flatten)]
    #[serde(default)]
    deliberation: deliberation::Flags,
}

impl Settings {
    /// The budgets: dollars, then the plan limits, in the order of
    /// [`budgets_mut`](Self::budgets_mut).
    fn budgets(&self) -> [Option<f64>; 7] {
        [
            self.budget,
            self.plan_budget_5h,
            self.plan_budget_7d,
            self.claude_code_budget_5h,
            self.claude_code_budget_7d,
            self.chatgpt_budget_5h,
            self.chatgpt_budget_7d,
        ]
    }

    fn budgets_mut(&mut self) -> [&mut Option<f64>; 7] {
        [
            &mut self.budget,
            &mut self.plan_budget_5h,
            &mut self.plan_budget_7d,
            &mut self.claude_code_budget_5h,
            &mut self.claude_code_budget_7d,
            &mut self.chatgpt_budget_5h,
            &mut self.chatgpt_budget_7d,
        ]
    }

    /// The plan budget: the general limits apply to each plan's windows, a
    /// plan's own replace them.
    fn plan_budget(&self) -> PlanBudget {
        let limits = |five_hour, seven_day| PlanLimits {
            five_hour,
            seven_day,
        };
        PlanBudget {
            five_hour: self.plan_budget_5h,
            seven_day: self.plan_budget_7d,
            claude_code: limits(self.claude_code_budget_5h, self.claude_code_budget_7d),
            chatgpt: limits(self.chatgpt_budget_5h, self.chatgpt_budget_7d),
        }
    }
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
    /// The crate existed: the run changes it (`--update`).
    #[serde(default)]
    update: bool,
    /// Deliberation, resolved from the flags when the run started.
    #[serde(default)]
    deliberation: Option<deliberation::Config>,
    /// The deliberation agents' instructions, by agent name.
    #[serde(default)]
    agent_instructions: BTreeMap<String, String>,
}

#[tokio::main]
async fn main() -> ForgeResult<()> {
    env_logger::init();
    let args = Args::parse();

    // A new run scaffolds its crate; a resumed one finds it in its manifest.
    let (state_dir, manifest, use_case) = match &args.resume {
        Some(state_dir) => {
            let state_dir = std::fs::canonicalize(state_dir)?;
            let mut manifest: RunManifest =
                serde_json::from_slice(&std::fs::read(state_dir.join("forge-run.json"))?)?;
            check_dependencies(&forge_metadata(
                &manifest.settings,
                manifest.deliberation.as_ref(),
            ));
            // A budget given with --resume (dollars or plan points) replaces
            // the saved one (to raise a budget the run stopped at); the other
            // settings stay.
            let saved = manifest.settings.budgets();
            let settings = &mut manifest.settings;
            let given = args.settings.budgets();
            for (given, kept) in given.into_iter().zip(settings.budgets_mut()) {
                if given.is_some() {
                    *kept = given;
                }
            }
            if saved != manifest.settings.budgets() {
                std::fs::write(
                    state_dir.join("forge-run.json"),
                    serde_json::to_vec_pretty(&manifest)?,
                )?;
            }
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
            let (output, name) = match &args.update {
                Some(dir) => {
                    let dir = std::fs::canonicalize(dir)?;
                    if !dir.join("Cargo.toml").is_file() {
                        return Err(format!(
                            "{} has no Cargo.toml: --update changes an existing crate",
                            dir.display()
                        )
                        .into());
                    }
                    let name = scaffold::package_name(&dir)
                        .and_then(|name| scaffold::crate_name(&name))
                        .ok_or("the crate's package needs a name starting with a letter")?;
                    if !scaffold::depends_on(&dir, &arko) {
                        eprintln!(
                            "forge: {} doesn't depend on the arko crates in {}, which the agents read",
                            dir.display(),
                            arko.display()
                        );
                    }
                    (dir, name)
                }
                None => {
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
                        .ok_or("pass --name. The crate needs a name starting with a letter")?;
                    (output, name)
                }
            };
            let architect = ModelSpec::new(
                &args.settings.architect_model,
                args.settings.architect_effort,
            )?;
            let deliberation = args
                .settings
                .deliberation
                .resolve(&architect, openai_access())?;
            // Before the use case is asked for or the crate is touched.
            check_dependencies(&forge_metadata(&args.settings, deliberation.as_ref()));
            let use_case = read_use_case(&args)?;
            let state_dir = args.state_dir.clone().unwrap_or_else(|| {
                scaffold::forge_home().join("runs").join(format!(
                    "{name}-{}",
                    chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
                ))
            });
            match &args.update {
                Some(_) => scaffold::add_change(&output, &use_case)?,
                None => scaffold::scaffold(&output, &name, &arko, &use_case)?,
            }
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
                update: args.update.is_some(),
                deliberation,
                agent_instructions: BTreeMap::new(),
            };
            manifest.architect_instructions = architect_instructions(&manifest);
            manifest.explorer_instructions = fill(prompts::EXPLORER_INSTRUCTIONS, &manifest);
            manifest.agent_instructions = deliberation_instructions(&manifest);
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
        metered: settings.metered,
        spike_root: state_dir.join("deliberations"),
    };
    let architect_model = ModelSpec::new(&settings.architect_model, settings.architect_effort)?;
    let mut architect = agents.config(
        ARCHITECT,
        "Workflow architect",
        &architect_model,
        "You design and build arko workflows: Rust binaries that run teams of agents.",
        &manifest.architect_instructions,
        Access::Build,
        if settings.involvement == Involvement::Clarify {
            &["ask_user"]
        } else {
            &[]
        },
    )?;
    architect.handover = Some(HandoverConfig::at_input_tokens(handover_at(
        &architect_model,
        settings.handover_at,
        settings.openai_handover_at,
    )));
    if settings.trial_runs > 0 {
        architect.tool_config.tool_specific_configs.insert(
            "trial_run".into(),
            trial_run_time_limit(settings.trial_minutes),
        );
    }
    architect.complete_task_summary_prompt_override = Some(prompts::SUMMARY_REQUEST.into());
    let mut configs = vec![(ARCHITECT.to_string(), architect)];
    if !settings.no_explorers {
        let explorer_model = ModelSpec::new(&settings.explorer_model, settings.explorer_effort)?;
        let mut explorer = agents.config(
            EXPLORER,
            "Framework explorer",
            &explorer_model,
            "You find things in the arko framework and report them exactly.",
            &manifest.explorer_instructions,
            Access::Read,
            &[],
        )?;
        // An explorer on an OpenAI model stays under the ChatGPT login's
        // input limit.
        if Model::find(&explorer_model.id).is_some_and(|model| model.provider == Provider::OpenAI) {
            explorer.handover = Some(HandoverConfig::at_input_tokens(handover_at(
                &explorer_model,
                settings.handover_at,
                settings.openai_handover_at,
            )));
        }
        configs.push((EXPLORER.to_string(), explorer));
    }
    // Deliberation: its agents, the tools they submit their parts with, and
    // the architect's request (and, when it rules, its ruling).
    let ledger = match &manifest.deliberation {
        Some(_) => Some(Arc::new(deliberation::Store::open(&state_dir, &workspace)?)),
        None => None,
    };
    let mut agent_tools: Vec<(String, Arc<dyn arko_agents::Tool>)> = Vec::new();
    let mut spikers = Vec::new();
    if let (Some(config), Some(ledger)) = (&manifest.deliberation, &ledger) {
        for (name, role, slot, model) in config.agents() {
            let (title, description, access) = deliberation_role(role);
            if access == Access::Experiment {
                spikers.push(name.clone());
            }
            let mut agent = agents.config(
                &name,
                title,
                model,
                description,
                manifest
                    .agent_instructions
                    .get(&name)
                    .map(String::as_str)
                    .unwrap_or_default(),
                access,
                &[],
            )?;
            agent.handover = Some(HandoverConfig::at_input_tokens(handover_at(
                model,
                settings
                    .handover_at
                    .min(settings.deliberation_handover_at.unwrap_or(200_000)),
                settings.openai_handover_at,
            )));
            configs.push((name.clone(), agent));
            for &part in deliberation::Config::parts(role) {
                agent_tools.push((
                    name.clone(),
                    Arc::new(deliberation::Submit {
                        part,
                        slot,
                        store: ledger.clone(),
                        config: config.clone(),
                        lead: false,
                    }),
                ));
            }
        }
        agent_tools.push((
            ARCHITECT.to_string(),
            Arc::new(deliberation::RequestDeliberation {
                store: ledger.clone(),
                config: config.clone(),
            }),
        ));
        if config.judges.is_empty() {
            agent_tools.push((
                ARCHITECT.to_string(),
                Arc::new(deliberation::Submit {
                    part: deliberation::Part::Ruling,
                    slot: 0,
                    store: ledger.clone(),
                    config: config.clone(),
                    lead: true,
                }),
            ));
        }
    }

    // Version control and publishing stay with the user, whatever the
    // agents' shells are asked to run: the command rules let git and gh
    // only read (spikes may use git fully in their scratch crates).
    let shell = |spike_root: Option<&Path>| {
        Arc::new(tools::RunShell::new(
            spike_root,
            manifest.target_dir.clone(),
        ))
    };
    let (shell, spike_shell) = (shell(None), shell(Some(&agents.spike_root)));
    let client = Arc::new(client(settings.backend, &configs)?);
    // The forge's agents don't message each other (and proposers must stay
    // independent).
    let mut builder = Workflow::builder()
        .name("forge")
        .metadata(forge_metadata(settings, manifest.deliberation.as_ref()))
        .workspace(&workspace)
        .llm_client(client)
        .persistence(PersistenceConfig::new(&state_dir))
        .disable_messaging();
    for (name, config) in configs {
        // Agents outside Claude Code get the forge's shell instead of `Bash`.
        if !config.runs_in_claude_code() {
            let shell = if spikers.contains(&name) {
                &spike_shell
            } else {
                &shell
            };
            builder = builder.agent_tool(&name, shell.clone());
        }
        builder = builder.agent(&name, config);
    }
    for (name, tool) in agent_tools {
        builder = builder.agent_tool(&name, tool);
    }
    if !settings.no_explorers {
        builder = builder
            .workflow_services(
                WorkflowServicesConfig::default()
                    .allow_caller(ARCHITECT)
                    .allow_target(EXPLORER)
                    .max_active_calls(settings.max_active_calls),
            )
            .hook(HookPoint::PostAgentFailure, {
                let failed_once = Arc::new(AtomicBool::new(false));
                let model = ModelSpec::new(&settings.explorer_model, settings.explorer_effort)?;
                Arc::new(move |context| explorer_failed(&failed_once, &model, context))
            });
    }
    if let Some(path) = &lessons_path {
        builder = builder.agent_tool(
            ARCHITECT,
            Arc::new(tools::RecordLesson { path: path.clone() }),
        );
    }
    let trial_ledger = Arc::new(std::sync::OnceLock::new());
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
                trial_budget: settings.trial_budget,
                workflow: trial_ledger.clone(),
            }),
        );
    }
    if settings.involvement == Involvement::Clarify {
        builder = builder.user_input_handler(Arc::new(TerminalInput));
    }
    // Over the budget the agents pause (the default policy) and the forge
    // stops, saved, until it is resumed with a higher budget.
    if let Some(budget) = settings.budget {
        builder = builder.budget(budget);
    }
    builder = builder.plan_budget(settings.plan_budget());
    // Running agents are saved as they run, so the latest checkpoint holds
    // the architect as it was when the forge stopped. Without one (the forge
    // stopped before the architect's first save), a resume starts from the
    // files in the crate instead.
    if args.resume.is_some()
        && let Ok(pointer) = std::fs::read_to_string(state_dir.join("latest-checkpoint"))
    {
        builder = builder.resume_from(state_dir.join(pointer.trim()));
    }
    let workflow = builder.build()?;
    let _ = trial_ledger.set(workflow.context().clone());
    // The architect's run calls: after a login failure, one fresh architect
    // run per process continues from the crate, then the forge stops.
    let restarted = Arc::new(AtomicBool::new(false));
    let architect = |restarted: &Arc<AtomicBool>| Architect {
        workflow: &workflow,
        state_dir: &state_dir,
        model: architect_model.clone(),
        restart: fill(prompts::ARCHITECT_RESUME, &manifest),
        restarted: restarted.clone(),
    };
    let deliberator = match (&manifest.deliberation, &ledger) {
        (Some(config), Some(ledger)) => Some(deliberation::Deliberator {
            workflow: &workflow,
            state_dir: &state_dir,
            store: ledger.clone(),
            config,
            arko: &manifest.arko,
            target_dir: &manifest.target_dir,
            resumed: Mutex::default(),
            architect: architect(&restarted),
        }),
        _ => None,
    };
    let architect = architect(&restarted);

    eprintln!(
        "forge: building `{}` in {}",
        manifest.name,
        workspace.display()
    );
    eprintln!("forge: state and logs in {}", state_dir.display());
    let plan_usage = Arc::new(Mutex::new(BTreeMap::<&'static str, String>::new()));
    let printer = tokio::spawn(print_events(
        workflow.subscribe_events(),
        args.verbose,
        plan_usage.clone(),
    ));

    let providers = providers();
    let task = if manifest.update {
        prompts::ARCHITECT_UPDATE_TASK
    } else {
        prompts::ARCHITECT_TASK
    };
    let task = fill(task, &manifest)
        .replace("{providers}", &providers)
        .replace("{use_case}", use_case.trim());
    let earlier = workflow
        .result()
        .agent_results
        .into_iter()
        .rfind(|result| result.agent_name == ARCHITECT);
    // An architect that was running when the forge stopped continues where
    // it was saved, its explorers with it; if that fails, it continues from
    // the crate as below.
    let recovered = match workflow
        .interrupted_runs()
        .into_iter()
        .rfind(|run| run.agent_name == ARCHITECT && run.task.parent.is_none())
    {
        Some(interrupted) => {
            eprintln!("forge: recovering the architect where it stopped");
            match architect.call(workflow.recover_agent(&interrupted)).await {
                Ok(result) => Some(result),
                Err(error) => {
                    eprintln!("forge: couldn't recover the architect ({error})");
                    // Its latest recorded state.
                    workflow.agent_result(interrupted.agent_run_id)
                }
            }
        }
        None => None,
    };
    let mut result = match recovered.or(earlier) {
        Some(earlier) if earlier.status == AgentRunStatus::Completed => earlier,
        // Paused by the budget (or an error the forge doesn't wait out):
        // it continues with its held request.
        Some(earlier) if earlier.status == AgentRunStatus::Paused => {
            eprintln!("forge: continuing the paused architect");
            let resume = AgentResume::ContinueAfterError { note: None };
            architect
                .call(workflow.resume_agent(&earlier, resume))
                .await?
        }
        // A run that failed can't be continued: a fresh architect continues
        // from the crate (after a login failure, that is its one fresh
        // start in this process).
        Some(earlier) if earlier.status == AgentRunStatus::Failed => {
            if let Some(error) = deliberation::login_failure(&state_dir, &earlier) {
                eprintln!(
                    "forge: the architect couldn't reach its model ({}). Starting it again from the crate",
                    cut(&error, 200)
                );
                restarted.store(true, Ordering::SeqCst);
            }
            architect
                .call(workflow.run_agent(ARCHITECT, &architect.restart))
                .await?
        }
        Some(earlier) => {
            let resume = fill(prompts::ARCHITECT_RESUME, &manifest);
            match architect
                .call(workflow.continue_agent(&earlier, &resume))
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    eprintln!(
                        "forge: couldn't continue the earlier task ({error}). Starting again"
                    );
                    architect
                        .call(workflow.run_agent(ARCHITECT, &resume))
                        .await?
                }
            }
        }
        None if args.resume.is_some() => {
            architect
                .call(workflow.run_agent(ARCHITECT, &architect.restart))
                .await?
        }
        None => architect.call(workflow.run_agent(ARCHITECT, &task)).await?,
    };
    // Deliberations the architect requested run now, and it continues with
    // their records (also after a restart: finished stages aren't run again).
    result = settle(&deliberator, result).await?;
    // Explorers the architect was running when the forge stopped and that it
    // didn't resume itself finish their work too (their results are in the
    // crate and the event log). A deliberation's runs are moved on by the
    // deliberation alone, while its decision is open.
    for orphan in workflow
        .interrupted_runs()
        .into_iter()
        .filter(|run| !deliberation::is_stage_agent(&run.agent_name))
    {
        eprintln!(
            "forge: recovering {} run {} where it stopped",
            orphan.agent_name, orphan.agent_run_id
        );
        if let Err(error) =
            within_budget(&workflow, &state_dir, workflow.recover_agent(&orphan)).await
        {
            eprintln!(
                "forge: couldn't recover {} run {} ({error})",
                orphan.agent_name, orphan.agent_run_id
            );
        }
    }

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
                    "forge: `{}` failed. Back to the architect (repair {repairs} of {})",
                    failure.check, settings.max_repairs
                );
                let prompt = format!(
                    "The forge's check `{}` failed after you finished. Fix the cause, run the checks again, and call complete_task when they all pass.\n\n```text\n{}\n```",
                    failure.check, failure.output
                );
                result = architect
                    .call(workflow.continue_agent(&result, &prompt))
                    .await?;
                result = settle(&deliberator, result).await?;
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

    let decisions = ledger
        .map(|ledger| {
            ledger
                .get()
                .decisions
                .iter()
                .map(deliberation::summary_line)
                .collect()
        })
        .unwrap_or_default();
    let report = report(
        &manifest,
        &result.summary,
        &checks,
        &finished,
        trial_reserved(&workflow),
        plan_usage
            .lock()
            .unwrap()
            .iter()
            .map(|(plan, text)| format!("{plan} {text}"))
            .collect::<Vec<_>>(),
        decisions,
    );
    std::fs::write(state_dir.join("report.md"), &report)?;
    println!("\n{report}");
    if checks.is_err() {
        std::process::exit(1);
    }
    Ok(())
}

/// Runs the deliberations the architect requested, when deliberation is on,
/// and returns its result after it has their records.
async fn settle(
    deliberator: &Option<deliberation::Deliberator<'_>>,
    result: AgentResult,
) -> Result<AgentResult, WorkflowError> {
    match deliberator {
        Some(deliberator) => deliberator.settle(result).await,
        None => Ok(result),
    }
}

/// Runs a workflow call, unless the budget stops the forge first: when the
/// budget holds the whole run (an explorer waits for a raise while the
/// architect waits for it) or the call returns paused by the budget, the
/// forge saves a checkpoint and exits, telling how to continue.
async fn within_budget<T>(
    workflow: &Workflow,
    state_dir: &Path,
    call: impl std::future::Future<Output = Result<T, WorkflowError>>,
) -> Result<T, WorkflowError> {
    tokio::pin!(call);
    let result = tokio::select! {
        result = &mut call => result,
        // The call is still running here, so the checkpoint records its
        // agents as they are, held.
        () = workflow.budget_blocked() => budget_stop(workflow, state_dir).await,
    };
    match result {
        Err(WorkflowError::AgentPaused {
            pause:
                AgentPause::ErrorRecovery {
                    cause: arko_agents::ErrorRecoveryCause::BudgetExceeded,
                    ..
                },
            ..
        }) => budget_stop(workflow, state_dir).await,
        // An explorer couldn't reach its model a second time
        // ([`explorer_failed`] stopped the run).
        Err(WorkflowError::AbortedByHook { hook_point, reason })
            if hook_point == "PostAgentFailure" =>
        {
            deliberation::login_stop(workflow, state_dir, &reason).await
        }
        result => result,
    }
}

/// The architect's run calls. When the architect's run fails because its
/// model request needs a new credential or a configuration change (an
/// expired login), a fresh architect run continues from the crate, once per
/// process; when that fails the same way, the forge stops, saved, with
/// status 3.
pub(crate) struct Architect<'a> {
    pub workflow: &'a Workflow,
    pub state_dir: &'a Path,
    pub model: ModelSpec,
    /// The fresh run's prompt.
    pub restart: String,
    /// The fresh start of this process is used (shared by every copy).
    pub restarted: Arc<AtomicBool>,
}

impl Architect<'_> {
    /// Runs `call` (one of the architect's run calls) within the budget,
    /// starting the architect afresh after a login failure, and stopping the
    /// forge on a second one.
    pub(crate) async fn call(
        &self,
        call: impl std::future::Future<Output = Result<AgentResult, WorkflowError>>,
    ) -> Result<AgentResult, WorkflowError> {
        match self.attempt(call).await {
            Ok(result) => result,
            Err(reason) => deliberation::login_stop(self.workflow, self.state_dir, &reason).await,
        }
    }

    /// The call's outcome, or why the forge stops: the architect can't reach
    /// its model, also when started afresh.
    pub(crate) async fn attempt(
        &self,
        call: impl std::future::Future<Output = Result<AgentResult, WorkflowError>>,
    ) -> Result<Result<AgentResult, WorkflowError>, String> {
        let mut failed_before = self.failed_runs();
        let mut outcome = within_budget(self.workflow, self.state_dir, call).await;
        loop {
            // The architect run that failed in this call, if one did.
            let error = self
                .workflow
                .context()
                .agent_results
                .read()
                .iter()
                .filter(|run| Self::is_architects(run) && run.status == AgentRunStatus::Failed)
                .find(|run| !failed_before.contains(&run.agent_run_id))
                .and_then(|run| deliberation::login_failure(self.state_dir, run));
            let Some(error) = error else {
                return Ok(outcome);
            };
            if self.restarted.swap(true, Ordering::SeqCst) {
                return Err(deliberation::login_reason(
                    ARCHITECT,
                    Some(&self.model),
                    &error,
                ));
            }
            eprintln!(
                "forge: the architect couldn't reach its model ({}). Starting it again from the crate",
                cut(&error, 200)
            );
            failed_before = self.failed_runs();
            outcome = within_budget(
                self.workflow,
                self.state_dir,
                self.workflow.run_agent(ARCHITECT, &self.restart),
            )
            .await;
        }
    }

    fn is_architects(run: &AgentResult) -> bool {
        run.agent_name == ARCHITECT && run.task.parent.is_none()
    }

    /// The architect's runs that failed.
    fn failed_runs(&self) -> HashSet<uuid::Uuid> {
        self.workflow
            .context()
            .agent_results
            .read()
            .iter()
            .filter(|run| Self::is_architects(run) && run.status == AgentRunStatus::Failed)
            .map(|run| run.agent_run_id)
            .collect()
    }
}

/// An explorer whose model request needs a new credential or a configuration
/// change (an expired login): the architect may ask again once (a fresh
/// explorer run); a second such failure in this process stops the run, which
/// [`within_budget`] turns into the forge's saved stop (status 3).
fn explorer_failed(
    failed_once: &AtomicBool,
    model: &ModelSpec,
    context: &arko_workflows::hooks::context::HookContext,
) -> HookResult {
    let login = context.agent_name.as_deref() == Some(EXPLORER)
        && context
            .metadata
            .get("cause")
            .and_then(|cause| cause.as_str())
            == Some("non_retryable");
    if !login {
        return HookResult::Continue;
    }
    let error = context.error_message.clone().unwrap_or_default();
    if !failed_once.swap(true, Ordering::SeqCst) {
        eprintln!(
            "forge: an explorer couldn't reach its model ({}). The architect may ask again",
            cut(&error, 200)
        );
        return HookResult::Continue;
    }
    HookResult::Abort(deliberation::login_reason(EXPLORER, Some(model), &error))
}

/// The budget is used up: saves the paused agents and exits with status 2.
async fn budget_stop(workflow: &Workflow, state_dir: &Path) -> ! {
    if let Err(error) = workflow.checkpoint().await {
        eprintln!("forge: couldn't save a checkpoint ({error}). The latest one is kept");
    }
    let spent = workflow.cost_ledger().total();
    let mut used_up = Vec::new();
    let mut flags = Vec::new();
    if let Some(limit) = workflow.budget()
        && spent > limit
    {
        used_up.push(format!("the budget of ${limit:.2} (spent ${spent:.2})"));
        flags.push("--budget <dollars>".to_string());
    }
    let plan = workflow.plan_budget();
    for subscription in SubscriptionPlan::ALL {
        let spent = workflow.plan_usage_spent_of(subscription);
        for window in PlanWindow::ALL {
            let Some(limit) = plan.limit_for(subscription, window) else {
                continue;
            };
            if spent.get(window).is_none_or(|used| used <= limit) {
                continue;
            }
            used_up.push(format!(
                "the {} plan budget of {limit} points of its {} window (used {:.1})",
                subscription.label(),
                window.label(),
                spent.get(window).unwrap_or_default()
            ));
            // The flag that set it: the plan's own, or the general one.
            let own = match subscription {
                SubscriptionPlan::ClaudeCode => plan.claude_code.limit(window),
                SubscriptionPlan::ChatGpt => plan.chatgpt.limit(window),
            };
            let span = match window {
                PlanWindow::FiveHour => "5h",
                PlanWindow::SevenDay => "7d",
            };
            let flag = match (own, subscription) {
                (None, _) => format!("--plan-budget-{span} <points>"),
                (Some(_), SubscriptionPlan::ClaudeCode) => {
                    format!("--claude-code-budget-{span} <points>")
                }
                (Some(_), SubscriptionPlan::ChatGpt) => {
                    format!("--chatgpt-budget-{span} <points>")
                }
            };
            if !flags.contains(&flag) {
                flags.push(flag);
            }
        }
    }
    // A run resumed from a checkpoint written before plans were counted
    // apart counts without a plan until one reports: while no plan has
    // counting of its own, the run's counting is that.
    let spent = workflow.plan_usage_spent();
    for (window, flag) in [
        (PlanWindow::FiveHour, "--plan-budget-5h <points>"),
        (PlanWindow::SevenDay, "--plan-budget-7d <points>"),
    ] {
        let unnamed = SubscriptionPlan::ALL
            .iter()
            .all(|&plan| workflow.plan_usage_spent_of(plan).get(window).is_none());
        if unnamed
            && let (Some(limit), Some(used)) = (plan.limit(window), spent.get(window))
            && used > limit
        {
            used_up.push(format!(
                "the plan budget of {limit} points of the {} window (used {used:.1})",
                window.label()
            ));
            if !flags.iter().any(|known| known == flag) {
                flags.push(flag.to_string());
            }
        }
    }
    if used_up.is_empty() {
        // An agent's own budget; the forge sets none, but say something.
        used_up.push("a budget".to_string());
        flags.push("--budget <dollars>".to_string());
    }
    eprintln!(
        "\nforge: {} is used up. The agents are paused where they were and saved.",
        used_up.join(" and ")
    );
    eprintln!(
        "forge: to continue, resume with a higher budget:\n\n    forge --resume {} {}\n",
        state_dir.display(),
        flags.join(" ")
    );
    std::process::exit(2);
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
    let deliberation = manifest
        .deliberation
        .as_ref()
        .map(|config| config.architect_section(lessons_path(settings).is_some()))
        .unwrap_or_default();
    let text = prompts::ARCHITECT_INSTRUCTIONS
        .replace(
            "{workspace_intro}",
            if manifest.update {
                prompts::EXISTING_CRATE
            } else {
                prompts::NEW_CRATE
            },
        )
        .replace("{deliberation}", &deliberation)
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

/// The deliberation agents' instructions, by agent name. Those with a shell
/// are told how to treat the forge's refusals.
fn deliberation_instructions(manifest: &RunManifest) -> BTreeMap<String, String> {
    let Some(config) = &manifest.deliberation else {
        return BTreeMap::new();
    };
    config
        .agents()
        .into_iter()
        .map(|(name, role, _, model)| {
            let text = match role {
                deliberation::PROPOSER => prompts::PROPOSER_INSTRUCTIONS,
                deliberation::CRITIC => prompts::CRITIC_INSTRUCTIONS,
                deliberation::SPIKER => prompts::SPIKER_INSTRUCTIONS,
                _ => prompts::JUDGE_INSTRUCTIONS,
            };
            let in_claude_code = manifest.settings.backend == Backend::ClaudeCode
                && Model::find(&model.id).is_some_and(|m| m.provider == Provider::Anthropic);
            let shell = role == deliberation::SPIKER || !in_claude_code;
            let text = format!(
                "{}\n{}",
                fill(text, manifest).trim_end(),
                if shell { prompts::REFUSALS } else { "" }
            );
            (name, text)
        })
        .collect()
}

/// A deliberation role's title, description and access.
fn deliberation_role(role: &str) -> (&'static str, &'static str, Access) {
    match role {
        deliberation::PROPOSER => (
            "Design proposer",
            "You propose designs for decisions about arko workflows.",
            Access::Read,
        ),
        deliberation::CRITIC => (
            "Design critic",
            "You review proposed designs and report the problems that matter.",
            Access::Read,
        ),
        deliberation::SPIKER => (
            "Spike runner",
            "You settle contested claims about designs with small experiments.",
            Access::Experiment,
        ),
        _ => (
            "Design judge",
            "You rule on design decisions, weighing findings, answers and experiments.",
            Access::Read,
        ),
    }
}

/// The architect's time limit for a `trial_run` call: a trial may take
/// `--trial-minutes`, past the runtime's default tool time limit (20
/// minutes), and the call waits for it, with ten minutes to spare.
fn trial_run_time_limit(trial_minutes: u64) -> PerToolConfigOverride {
    PerToolConfigOverride {
        tool_timeout_seconds: Some(Some(trial_minutes * 60 + 600)),
        ..PerToolConfigOverride::default()
    }
}

/// The handover threshold for an agent on `model`: `wanted`, kept under the
/// ChatGPT login's input limit for OpenAI models, or `openai` in its place
/// (`--openai-handover-at`) for them.
fn handover_at(model: &ModelSpec, wanted: usize, openai: Option<usize>) -> usize {
    match Model::find(&model.id).map(|model| model.provider) {
        Some(Provider::OpenAI) => match openai {
            Some(openai) => openai.min(arko_ai::CHATGPT_OAUTH_MAX_INPUT_TOKENS),
            None => wanted.min(arko_ai::CHATGPT_OAUTH_MAX_INPUT_TOKENS * 3 / 4),
        },
        _ => wanted,
    }
}

/// A non-empty environment variable.
fn var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// Whether OpenAI models can be reached: a ChatGPT login (Codex's
/// `auth.json`, or `OPENAI_OAUTH_ACCESS_TOKEN` with `OPENAI_OAUTH_ACCOUNT_ID`)
/// or `OPENAI_API_KEY`, as [`client`] reaches them.
fn openai_access() -> bool {
    arko_ai::CodexAuthFile::default_path().is_some_and(|path| path.is_file())
        || (var("OPENAI_OAUTH_ACCESS_TOKEN").is_some() && var("OPENAI_OAUTH_ACCOUNT_ID").is_some())
        || var("OPENAI_API_KEY").is_some()
}

/// The client for the agents' providers. Claude Code agents don't use it.
/// OpenAI agents use the ChatGPT login when there is one (Codex's
/// `auth.json`, else the `OPENAI_OAUTH_*` variables), else `OPENAI_API_KEY`.
fn client(backend: Backend, configs: &[(String, AgentConfig)]) -> ForgeResult<LLMClient> {
    let mut client = match backend {
        Backend::ClaudeCode => LLMClient::new(),
        Backend::Api => LLMClient::from_env()?,
    };
    let uses = |provider: Provider| {
        configs
            .iter()
            .any(|(_, config)| config.llm_config.model.provider == provider)
    };
    if uses(Provider::OpenAI) && !client.has_provider(&Provider::OpenAI) {
        client = match arko_ai::CodexAuthFile::load_default() {
            Ok(auth) => client.with_codex_auth_file(&auth)?,
            Err(error) => match (
                var("OPENAI_OAUTH_ACCESS_TOKEN"),
                var("OPENAI_OAUTH_ACCOUNT_ID"),
                var("OPENAI_API_KEY"),
            ) {
                (Some(access_token), Some(account_id), _) => client.with_openai_chatgpt_oauth(
                    &access_token,
                    var("OPENAI_OAUTH_REFRESH_TOKEN").as_deref(),
                    &account_id,
                ),
                (_, _, Some(key)) => client.with_openai_api_key(&key),
                _ => {
                    return Err(format!(
                        "the OpenAI models need a ChatGPT login (`codex login`, or OPENAI_OAUTH_ACCESS_TOKEN and OPENAI_OAUTH_ACCOUNT_ID) or OPENAI_API_KEY ({error})"
                    )
                    .into());
                }
            },
        };
    }
    if uses(Provider::Google) && !client.has_provider(&Provider::Google) {
        let key = var("GEMINI_API_KEY")
            .or_else(|| var("GOOGLE_API_KEY"))
            .ok_or("the Google models need GEMINI_API_KEY")?;
        client = client.with_google_api_key(&key);
    }
    Ok(client)
}

/// The forge's metadata: the programs it needs (`cargo`, and `claude` when
/// an agent runs in Claude Code) and the ones that help (`rg`). Logins and
/// API keys aren't programs; the README lists them.
fn forge_metadata(
    settings: &Settings,
    deliberation: Option<&deliberation::Config>,
) -> WorkflowMetadata {
    let claude =
        |id: &str| Model::find(id).is_some_and(|model| model.provider == Provider::Anthropic);
    let in_claude_code = settings.backend == Backend::ClaudeCode
        && (claude(&settings.architect_model)
            || (!settings.no_explorers && claude(&settings.explorer_model))
            || deliberation.is_some_and(|config| {
                config
                    .agents()
                    .into_iter()
                    .any(|(_, _, _, model)| claude(&model.id))
            }));
    let mut dependencies = vec![
        WorkflowDependency::new(
            "cargo",
            "Rust's build tool, used by the forge and its agents to build, test and run the workflow crate",
        )
        .executable("cargo")
        .install_hint("rustup (https://rustup.rs)"),
    ];
    // On Windows they use `cmd.exe`.
    for (os, hint) in [
        (
            "linux",
            "your distribution's package manager (`apt install bash`)",
        ),
        ("macos", "`brew install bash`"),
    ] {
        dependencies.push(
            WorkflowDependency::new(
                "bash",
                "the shell the forge's `run_shell` and `trial_run` tools run commands in",
            )
            .executable("bash")
            .os(os)
            .install_hint(hint),
        );
    }
    if in_claude_code {
        dependencies.push(
            WorkflowDependency::new(
                "Claude Code",
                "the Claude Code CLI, signed in, where agents on Claude models run (`--backend claude-code`)",
            )
            .executable("claude")
            .install_hint("`npm install -g @anthropic-ai/claude-code` or Anthropic's native installer, then run `claude` once to sign in (or run the agents through the API with `--backend api`)"),
        );
    }
    dependencies.push(
        WorkflowDependency::new(
            "ripgrep",
            "the fast code search the agents run in their shells. Without it they search with slower tools such as grep",
        )
        .executable("rg")
        .install_hint(
            "`apt install ripgrep`, `brew install ripgrep` or `winget install BurntSushi.ripgrep.MSVC`",
        )
        .optional(),
    );
    WorkflowMetadata {
        description: "Designs and builds an arko workflow crate from a use case".into(),
        required_dependencies: dependencies,
        ..WorkflowMetadata::default()
    }
}

/// Checks the programs the forge needs before anything starts: a missing
/// required one stops the forge (status 1), a missing optional one is a
/// warning.
fn check_dependencies(metadata: &WorkflowMetadata) {
    let describe = |dependency: &WorkflowDependency| {
        format!(
            "{} (`{}`): {}.{}",
            dependency.name,
            dependency.executable.as_deref().unwrap_or_default(),
            dependency.description,
            dependency
                .install_hint
                .as_ref()
                .map(|hint| format!(" To install: {hint}."))
                .unwrap_or_default()
        )
    };
    let (optional, required): (Vec<_>, Vec<_>) = metadata
        .missing_dependencies()
        .into_iter()
        .partition(|dependency| dependency.optional);
    for dependency in optional {
        eprintln!("forge: {} was not found", describe(dependency));
    }
    if !required.is_empty() {
        eprintln!(
            "forge: can't start because required dependencies were not found on this machine"
        );
        for dependency in required {
            eprintln!("  - {}", describe(dependency));
        }
        std::process::exit(1);
    }
}

/// What an agent may do with files and commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    /// Read the crate and the arko checkout.
    Read,
    /// Also write files, outside the crate too, and run commands: spikes in
    /// their scratch crates.
    Experiment,
    /// Edit the crate, run commands, and fetch from the web: the architect.
    Build,
}

/// What trial runs reserved in the workflow's ledger.
fn trial_reserved(workflow: &Workflow) -> f64 {
    workflow
        .cost_ledger()
        .entries
        .iter()
        .filter(|entry| {
            matches!(&entry.source, arko_workflows::budget::CostSource::Manual { description }
                if description == tools::TRIAL_RESERVATION)
        })
        .map(|entry| entry.amount)
        .sum()
}

/// Builds agent configurations for the chosen backend.
struct Agents {
    backend: Backend,
    target_dir: PathBuf,
    metered: bool,
    /// The folder of the spikes' scratch crates, where they may use git
    /// fully.
    spike_root: PathBuf,
}

impl Agents {
    /// The configuration of agent `id` on `model`, with the tools `access`
    /// allows on its backend, `complete_task`, and `extra` tools.
    #[allow(clippy::too_many_arguments)]
    fn config(
        &self,
        id: &str,
        name: &str,
        model: &ModelSpec,
        role: &str,
        instructions: &str,
        access: Access,
        extra: &[&str],
    ) -> ForgeResult<AgentConfig> {
        let registered = Model::find(&model.id)
            .ok_or_else(|| format!("`{}` isn't a model arko knows", model.id))?;
        let (llm, reasoning) = match registered.provider {
            Provider::Anthropic => (
                match self.backend {
                    Backend::ClaudeCode => Model::claude_code(registered),
                    Backend::Api => registered,
                },
                ReasoningConfig::anthropic_adaptive_with_effort(model.effort.into()),
            ),
            Provider::OpenAI => (
                registered,
                ReasoningConfig::openai(model.effort.into(), false),
            ),
            Provider::Google => (
                registered,
                ReasoningConfig::gemini(model.effort.into(), false),
            ),
            provider => {
                return Err(format!("{provider:?} models aren't supported by the forge").into());
            }
        };
        let in_claude_code = llm.provider == Provider::ClaudeCode;
        let mut tools: Vec<String> = match (in_claude_code, access) {
            (true, Access::Read) => ClaudeCodeTools::read_only(),
            (true, Access::Experiment) => {
                let mut tools = ClaudeCodeTools::editing();
                tools.push("Bash".into());
                tools
            }
            (true, Access::Build) => {
                let mut tools = ClaudeCodeTools::editing();
                tools.extend(["Bash", "WebFetch", "WebSearch"].map(String::from));
                tools
            }
            (false, Access::Read) => ["read_file", "list_files"].map(String::from).to_vec(),
            (false, _) => ["read_file", "write_file", "apply_diff", "list_files"]
                .map(String::from)
                .to_vec(),
        };
        tools.push("complete_task".into());
        tools.extend(extra.iter().map(|tool| tool.to_string()));
        let generation_config = GenerationConfig::default()
            .with_reasoning(reasoning)
            .with_prompt_caching(true);
        // Reading may reach the arko checkout; writing stays in the crate,
        // except for spikes, which write in their scratch crates.
        let reads = ["Read", "Glob", "Grep", "read_file", "list_files"];
        let writes = ["Edit", "Write", "NotebookEdit", "write_file", "apply_diff"];
        let anywhere: HashMap<String, PerToolConfigOverride> = reads
            .iter()
            .chain(if access == Access::Experiment {
                &writes[..]
            } else {
                &[]
            })
            .filter(|tool| tools.iter().any(|name| name == *tool))
            .map(|tool| {
                (
                    tool.to_string(),
                    PerToolConfigOverride {
                        out_of_workspace_path_policy: Some(PathSafetyPolicy::AutoApprove),
                        ..PerToolConfigOverride::default()
                    },
                )
            })
            .collect();
        let mut builder = AgentConfig::builder(
            id.to_string(),
            name.to_string(),
            AgentLLMConfig {
                model: llm,
                generation_config,
            },
        )
        .role_description(role.to_string())
        .instructions(instructions.trim().to_string())
        .tool_config(AgentToolConfig {
            available_tool_names: tools,
            default_auto_approve_tool_use: true,
            tool_specific_configs: anywhere,
            ..AgentToolConfig::default()
        });
        if in_claude_code {
            let spike_root = (access == Access::Experiment).then_some(self.spike_root.as_path());
            let mut env = BTreeMap::from([(
                "CARGO_TARGET_DIR".to_string(),
                self.target_dir.display().to_string(),
            )]);
            env.extend(spike_root.map(tools::spike_environment).unwrap_or_default());
            builder = builder.claude_code(ClaudeCodeSettings {
                env,
                metered_billing: self.metered,
                command_rules: tools::command_rules(spike_root),
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
        found.push("the Claude Code CLI (`claude`, with a plan login for Claude models through `Model::claude_code`)");
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

/// Prints the run as it goes. The workflow's persistence journals every
/// event in the state directory's `events.jsonl`.
async fn print_events(
    mut events: tokio::sync::broadcast::Receiver<WorkflowEvent>,
    verbose: bool,
    plan_usage: Arc<Mutex<BTreeMap<&'static str, String>>>,
) {
    let mut requests: HashMap<String, usize> = HashMap::new();
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => break,
        };
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
            // The command guard's refusals (they start "Refused:") are shown
            // for every agent. On `run_shell` they are errors, shown below.
            AgentEvent::ToolRejected {
                tool_name, reason, ..
            } if shown
                || reason
                    .as_deref()
                    .is_some_and(|reason| reason.starts_with("Refused:")) =>
            {
                println!(
                    "[{who}] ✗ {tool_name}: {}",
                    cut(&reason.unwrap_or_default(), 300)
                )
            }
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
                        "[{who}]   ({count} requests, this one {}k input tokens)",
                        usage.input_tokens / 1_000
                    );
                }
            }
            AgentEvent::PlanUsageReported(usage) => {
                let text = [(&usage.primary, true), (&usage.secondary, false)]
                    .into_iter()
                    .filter_map(|(window, primary)| {
                        let window = window.as_ref()?;
                        Some(format!(
                            "{} {:.0}%",
                            window_name(window.window_minutes, primary),
                            window.used_percent
                        ))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let plan = usage.plan.map_or("plan", SubscriptionPlan::label);
                let mut last = plan_usage.lock().unwrap();
                if last.get(plan) != Some(&text) {
                    println!("[{plan} usage] {text}");
                    last.insert(plan, text);
                }
            }
            AgentEvent::HandoverRequested { input_tokens } => {
                println!(
                    "[{who}] handing over at {}k input tokens",
                    input_tokens / 1_000
                )
            }
            AgentEvent::ConversationHandedOver { handovers, .. } => {
                println!("[{who}] handover {handovers} done. Continuing in a fresh conversation")
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

/// A reported plan window's name: by its length when reported (plans differ:
/// some have no five-hour window), otherwise by its position.
fn window_name(minutes: Option<u64>, primary: bool) -> String {
    match minutes {
        Some(300) => "five-hour".into(),
        Some(10_080) => "seven-day".into(),
        Some(minutes) => format!("{minutes}-minute"),
        None if primary => "five-hour".into(),
        None => "seven-day".into(),
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
    trials: f64,
    plan_usage: Vec<String>,
    decisions: Vec<String>,
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
        "## The architect's summary\n\n{}\n\n",
        summary.trim()
    ));
    if !decisions.is_empty() {
        text.push_str(&format!(
            "## Decisions deliberated\n\n{}\n\n",
            decisions.join("\n")
        ));
    }
    text.push_str("## Usage\n\n");
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
    // A trial is a separate program: the forge sees only what it reserved
    // for it, not what it spent.
    text.push_str(&format!(
        "\nBilled for the forge's agents: ${:.2}",
        (result.total_cost - trials).max(0.0)
    ));
    if trials > 0.0 {
        text.push_str(&format!(
            ", ${trials:.2} reserved for trial runs (their full caps, with spending reported in each trial's output)"
        ));
    }
    if !plan_usage.is_empty() {
        text.push_str(&format!(
            " (plan usage at the end: {})",
            plan_usage.join("; ")
        ));
    }
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trial_can_run_past_the_default_tool_time_limit() {
        let limit = trial_run_time_limit(75).tool_timeout_seconds;
        assert_eq!(limit, Some(Some(75 * 60 + 600)));
        assert!(arko_agents::agent_config::DEFAULT_TOOL_TIMEOUT_SECONDS < 75 * 60);
    }

    #[test]
    fn handover_thresholds_follow_the_provider() {
        let opus: ModelSpec = "claude-opus-5-5".parse().unwrap();
        let astra: ModelSpec = "gpt-6-astra".parse().unwrap();
        assert_eq!(handover_at(&opus, 450_000, Some(200_000)), 450_000);
        assert_eq!(handover_at(&astra, 450_000, Some(200_000)), 200_000);
        assert_eq!(handover_at(&astra, 450_000, None), 204_000);
        assert_eq!(handover_at(&astra, 150_000, None), 150_000);
        assert_eq!(handover_at(&astra, 150_000, Some(900_000)), 272_000);
        let args = Args::parse_from(["forge", "--output", "crate"]);
        assert_eq!(args.settings.deliberation_handover_at, None);
        assert_eq!(args.settings.openai_handover_at, None);
        // Settings saved before these flags existed still load.
        let mut saved = serde_json::to_value(&args.settings).unwrap();
        let object = saved.as_object_mut().unwrap();
        object.remove("deliberation_handover_at");
        object.remove("openai_handover_at");
        let loaded: Settings = serde_json::from_value(saved).unwrap();
        assert_eq!(loaded.openai_handover_at, None);
    }

    fn metadata(flags: &[&str]) -> WorkflowMetadata {
        let args = Args::parse_from([&["forge", "--output", "crate"], flags].concat());
        let settings = &args.settings;
        let architect =
            ModelSpec::new(&settings.architect_model, settings.architect_effort).unwrap();
        let deliberation = settings.deliberation.resolve(&architect, true).unwrap();
        forge_metadata(settings, deliberation.as_ref())
    }

    /// The programs the forge declares on every system for these flags:
    /// `(executable, optional)`.
    fn programs(flags: &[&str]) -> Vec<(String, bool)> {
        metadata(flags)
            .required_dependencies
            .into_iter()
            .filter(|dependency| dependency.os.is_none())
            .map(|dependency| (dependency.executable.unwrap(), dependency.optional))
            .collect()
    }

    /// A hook context for a run of `agent` that failed with `cause`.
    fn failure(agent: &str, cause: &str) -> arko_workflows::hooks::context::HookContext {
        arko_workflows::hooks::context::HookContext {
            timestamp: chrono::Utc::now(),
            hook_point: HookPoint::PostAgentFailure,
            workflow_run_id: uuid::Uuid::new_v4(),
            workflow_name: "forge".into(),
            agent_run_id: Some(uuid::Uuid::new_v4()),
            agent_name: Some(agent.into()),
            tool_name: None,
            tool_call_id: None,
            tool_params: None,
            tool_result: None,
            original_tool_result: None,
            effective_tool_result: None,
            agent_summary: None,
            approval_reason: None,
            error_message: Some("401 refresh_token_invalidated".into()),
            budget_spent: 0.0,
            budget_limit: None,
            metadata: HashMap::from([("cause".to_string(), serde_json::json!(cause))]),
        }
    }

    #[test]
    fn an_explorers_expired_login_lets_the_architect_ask_once_then_stops_the_forge() {
        let failed_once = AtomicBool::new(false);
        let model = ModelSpec::new("claude-sonnet-5-5", Effort::High).unwrap();
        let hook = |agent: &str, cause: &str| {
            explorer_failed(&failed_once, &model, &failure(agent, cause))
        };
        // Other agents, and other failures, are the forge's usual business.
        assert!(matches!(
            hook(ARCHITECT, "non_retryable"),
            HookResult::Continue
        ));
        assert!(matches!(hook(EXPLORER, "refusal"), HookResult::Continue));
        // The first: the architect may ask again (a fresh explorer run).
        assert!(matches!(
            hook(EXPLORER, "non_retryable"),
            HookResult::Continue
        ));
        // The second stops the run; `within_budget` makes it the saved stop.
        match hook(EXPLORER, "non_retryable") {
            HookResult::Abort(reason) => {
                assert!(
                    reason.starts_with("explorer (claude-sonnet-5-5:high, Anthropic)"),
                    "{reason}"
                );
                assert!(reason.contains("`claude`"), "{reason}");
            }
            _ => panic!("the forge should stop"),
        }
    }

    #[test]
    fn a_spikes_git_looks_for_repositories_no_higher_than_its_folder() {
        let state_dir = std::env::temp_dir().join("forge-state");
        let spike_root = state_dir.join("deliberations");
        // git doesn't look in its ceiling, or above it, from below it.
        let ceiling = || {
            vec![(
                "GIT_CEILING_DIRECTORIES".to_string(),
                state_dir.display().to_string(),
            )]
        };
        // Claude Code's `Bash`.
        let agents = Agents {
            backend: Backend::ClaudeCode,
            target_dir: std::env::temp_dir().join("forge-target"),
            metered: false,
            spike_root: spike_root.clone(),
        };
        let model = ModelSpec::new("claude-opus-5-5", Effort::High).unwrap();
        let env = |access| {
            let config = agents
                .config("agent", "Agent", &model, "Role.", "Do it.", access, &[])
                .unwrap();
            config
                .claude_code
                .unwrap()
                .env
                .into_iter()
                .filter(|(name, _)| name.starts_with("GIT_"))
                .collect::<Vec<_>>()
        };
        assert_eq!(env(Access::Experiment), ceiling());
        assert_eq!(env(Access::Build), []);
        // The forge's `run_shell`.
        let shell = |root: Option<&Path>| tools::RunShell::new(root, PathBuf::new()).env;
        assert_eq!(shell(Some(&spike_root)), ceiling());
        assert_eq!(shell(None), []);
    }

    #[test]
    fn bash_is_required_where_the_shell_tools_use_it() {
        for flags in [&[][..], &["--backend", "api"]] {
            let bash: Vec<(Option<String>, bool)> = metadata(flags)
                .required_dependencies
                .into_iter()
                .filter(|dependency| dependency.executable.as_deref() == Some("bash"))
                .map(|dependency| (dependency.os, dependency.optional))
                .collect();
            assert_eq!(
                bash,
                [(Some("linux".into()), false), (Some("macos".into()), false)]
            );
        }
    }

    #[test]
    fn claude_code_is_required_only_when_an_agent_runs_in_it() {
        let with_claude = [("cargo", false), ("claude", false), ("rg", true)]
            .map(|(name, optional)| (name.to_string(), optional))
            .to_vec();
        let without = [("cargo", false), ("rg", true)]
            .map(|(name, optional)| (name.to_string(), optional))
            .to_vec();
        assert_eq!(programs(&[]), with_claude);
        assert_eq!(programs(&["--backend", "api"]), without);
        let openai = ["--architect-model", "gpt-6.1-sol"];
        // The explorers run on Claude Sonnet.
        assert_eq!(programs(&openai), with_claude);
        assert_eq!(
            programs(&[&openai[..], &["--no-explorers"]].concat()),
            without
        );
        // So does a proposer.
        assert_eq!(
            programs(&[&openai[..], &["--no-explorers", "--deliberate", "light"]].concat()),
            with_claude
        );
    }
}
