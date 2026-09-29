//! DeepCoder as a code-defined `arko_workflows` example.
//!
//! The lead agent decomposes a specification into bounded coder runs. Every
//! coder completion is independently checked for correctness and then for
//! material inefficiency. A separate verifier gates final workflow completion.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arko_agents::agent_config::PathSafetyPolicy;
use arko_agents::terminal::{CommandGuard, ShellDialect};
use arko_agents::{
    AgentConfig, AgentLLMConfig, AgentToolConfig, BuiltInEnvironmentDetail,
    EnvironmentDetailPlacement, EnvironmentDetailsConfig, LLMErrorPolicy, TaskContext, Tool,
    ToolCallingMode, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};
use arko_ai::{
    CodexAuthFile, GenerationConfig, LLMClient, Model, OpenAIEffort, ReasoningConfig, Verbosity,
};
use arko_workflows::prelude::*;
use async_trait::async_trait;
use clap::Parser;
use serde_json::Value;
use sha2::{Digest, Sha256};

type ExampleResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const RUN_SHELL_TOOL: &str = "run_shell";
const REPAIR_CONTEXT_BYTES: usize = 65_536;

#[derive(Debug, Parser)]
#[command(
    name = "deepcoder",
    about = "Generic sequential coder/verifier workflow built with arko_workflows"
)]
struct Args {
    /// Existing project directory the agents should modify.
    #[arg(long)]
    workspace: PathBuf,

    /// Markdown or text specification to implement.
    #[arg(long)]
    spec: PathBuf,

    /// Run directory. A unique sibling directory is created when omitted.
    #[arg(long)]
    state_dir: Option<PathBuf>,

    /// Checkpoint JSON, or run directory containing `latest-checkpoint`.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Disable checkpoints and durable run artifacts.
    #[arg(long)]
    ephemeral: bool,

    #[arg(long, default_value_t = 50.0)]
    budget: f64,

    #[arg(long, default_value_t = 20)]
    max_minor_corrections: u32,

    #[arg(long, default_value_t = 3)]
    max_efficiency_corrections: u32,

    #[arg(long, default_value_t = 4)]
    max_final_corrections: u32,

    #[arg(long, default_value_t = 1_800)]
    shell_timeout_seconds: u64,

    #[arg(long, default_value_t = 65_536)]
    shell_output_bytes: usize,

    /// Let `run_shell` run every command, including ones that delete the
    /// filesystem root or the home directory.
    #[arg(long)]
    dangerously_allow_all_commands: bool,
}

#[derive(Debug)]
struct RunShellTool {
    default_timeout: Duration,
    output_limit: usize,
    guard: CommandGuard,
}

impl RunShellTool {
    fn new(timeout_seconds: u64, output_limit: usize, guard: CommandGuard) -> Self {
        Self {
            default_timeout: Duration::from_secs(timeout_seconds.max(1)),
            output_limit: output_limit.max(1_024),
            guard,
        }
    }

    fn bounded_output(&self, bytes: &[u8]) -> String {
        let text = String::from_utf8_lossy(bytes);
        if text.len() <= self.output_limit {
            return text.into_owned();
        }

        let half = self.output_limit / 2;
        let mut head_end = half.min(text.len());
        while head_end > 0 && !text.is_char_boundary(head_end) {
            head_end -= 1;
        }
        let mut tail_start = text.len().saturating_sub(half);
        while tail_start < text.len() && !text.is_char_boundary(tail_start) {
            tail_start += 1;
        }
        format!(
            "{}\n\n... {} bytes omitted ...\n\n{}",
            &text[..head_end],
            text.len()
                .saturating_sub(head_end + text.len().saturating_sub(tail_start)),
            &text[tail_start..]
        )
    }
}

#[async_trait]
impl Tool for RunShellTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: RUN_SHELL_TOOL.to_string(),
            description: "Run one local shell command in the workflow workspace. Use it for inspection, formatting, linting, builds, and tests. Use write_file/apply_diff for source edits so context-freshness tracking remains precise. Commands that would delete, move or overwrite protected paths or paths outside the workspace are refused. This is not a security sandbox and must never be used to deploy, publish, push, or contact production services.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "command".to_string(),
                    description: "Shell command to execute.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "timeout_seconds".to_string(),
                    description: "Optional positive timeout no larger than the workflow default."
                        .to_string(),
                    param_type: ToolParameterType::String,
                    required: false,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let command = params
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "command".to_string(),
            })?;
        let requested_timeout = params
            .get("timeout_seconds")
            .and_then(Value::as_str)
            .map(str::parse::<u64>)
            .transpose()
            .map_err(|error| ToolError::ParameterValidation {
                param_name: "timeout_seconds".to_string(),
                message: error.to_string(),
            })?
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout);
        let dialect = if cfg!(target_os = "windows") {
            ShellDialect::Cmd
        } else {
            ShellDialect::Posix
        };
        self.guard
            .check(command, dialect, context.current_working_directory)
            .map_err(|refusal| ToolError::ExecutionFailed(refusal.message))?;
        if requested_timeout.is_zero() || requested_timeout > self.default_timeout {
            return Err(ToolError::ParameterValidation {
                param_name: "timeout_seconds".to_string(),
                message: format!(
                    "timeout must be between 1 and {} seconds",
                    self.default_timeout.as_secs()
                ),
            });
        }

        #[cfg(target_os = "windows")]
        let mut process = {
            let mut process = tokio::process::Command::new("cmd.exe");
            process.args(["/D", "/S", "/C", command]);
            process
        };
        #[cfg(not(target_os = "windows"))]
        let mut process = {
            let mut process = tokio::process::Command::new("bash");
            process.args(["-lc", command]);
            process
        };
        process
            .current_dir(context.current_working_directory)
            .kill_on_drop(true);
        for (name, _) in std::env::vars_os() {
            if is_sensitive_environment_name(&name.to_string_lossy()) {
                process.env_remove(name);
            }
        }

        let output = tokio::time::timeout(requested_timeout, process.output())
            .await
            .map_err(|_| {
                ToolError::ExecutionFailed(format!(
                    "command timed out after {} seconds",
                    requested_timeout.as_secs()
                ))
            })?
            .map_err(|source| ToolError::Io {
                tool_name: RUN_SHELL_TOOL.to_string(),
                source,
            })?;
        let stdout = self.bounded_output(&output.stdout);
        let stderr = self.bounded_output(&output.stderr);
        if !output.status.success() {
            return Err(ToolError::ShellCommandFailed {
                tool_name: RUN_SHELL_TOOL.to_string(),
                exit_code: output.status.code(),
                stdout,
                stderr,
            });
        }
        Ok(format!(
            "exit_code: 0\nstdout:\n{}\nstderr:\n{}",
            stdout, stderr
        ))
    }
}

#[derive(Debug)]
struct TerminalInput;

#[async_trait]
impl UserInputHandler for TerminalInput {
    async fn respond(&self, request: UserInputRequest) -> Result<String, String> {
        tokio::task::spawn_blocking(move || {
            eprintln!(
                "\nAgent '{}' needs input:\n{}\n",
                request.agent_name, request.questions
            );
            eprint!("> ");
            io::stderr().flush().map_err(|error| error.to_string())?;
            let mut response = String::new();
            io::stdin()
                .read_line(&mut response)
                .map_err(|error| error.to_string())?;
            let response = response.trim().to_string();
            if response.is_empty() {
                Err("empty user response".to_string())
            } else {
                Ok(response)
            }
        })
        .await
        .map_err(|error| error.to_string())?
    }
}

fn agent_config(
    id: &str,
    name: &str,
    model: Model,
    effort: OpenAIEffort,
    role: &str,
    instructions: &str,
    tools: &[&str],
) -> ExampleResult<AgentConfig> {
    let generation_config = GenerationConfig::default()
        .with_max_tokens(128_000)
        .with_verbosity(Verbosity::Medium)
        .with_reasoning(ReasoningConfig::openai(effort, true))
        .with_prompt_caching(true);
    let llm_config = AgentLLMConfig {
        model,
        generation_config,
    };
    let environment_details_config = EnvironmentDetailsConfig {
        file_tree_mode: arko_agents::FileTreeMode::Diff,
        current_working_directory: BuiltInEnvironmentDetail {
            include: true,
            placement: EnvironmentDetailPlacement::InitialTaskPrompt,
            depth: None,
        },
        file_tree: BuiltInEnvironmentDetail {
            include: true,
            placement: EnvironmentDetailPlacement::InitialTaskPrompt,
            depth: Some(4),
        },
        current_datetime: BuiltInEnvironmentDetail {
            include: true,
            placement: EnvironmentDetailPlacement::InitialTaskPrompt,
            depth: None,
        },
        operating_system: BuiltInEnvironmentDetail {
            include: true,
            placement: EnvironmentDetailPlacement::InitialTaskPrompt,
            depth: None,
        },
    };
    let tool_config = AgentToolConfig {
        available_tool_names: tools.iter().map(|tool| (*tool).to_string()).collect(),
        default_auto_approve_tool_use: true,
        default_out_of_workspace_path_policy: PathSafetyPolicy::AutoReject,
        tool_specific_configs: HashMap::new(),
        tool_calling_mode: ToolCallingMode::Native,
        ..AgentToolConfig::default()
    };
    Ok(AgentConfig::builder(id.to_string(), name.to_string(), llm_config)
        .role_description(role.to_string())
        .instructions(instructions.to_string())
        .tool_config(tool_config)
        .environment_details_config(environment_details_config)
        .llm_error_policy(LLMErrorPolicy::PauseAndAwaitApproval)
        .complete_task_summary_prompt_override(
            "State the outcome, files or artifacts involved, deterministic checks run, and any remaining limitation. Be concrete and concise.".to_string(),
        )
        .build()?)
}

/// OAuth environment variables, then the Codex `auth.json`, then
/// `OPENAI_API_KEY`. A client from `auth.json` keeps the file in step: every
/// token refresh reads it again first, and rotated tokens are saved to it at
/// once.
fn build_client() -> ExampleResult<LLMClient> {
    if let (Ok(access_token), Ok(account_id)) = (
        std::env::var("OPENAI_OAUTH_ACCESS_TOKEN"),
        std::env::var("OPENAI_OAUTH_ACCOUNT_ID"),
    ) {
        let refresh_token = std::env::var("OPENAI_OAUTH_REFRESH_TOKEN").ok();
        return Ok(LLMClient::new().with_openai_chatgpt_oauth(
            &access_token,
            refresh_token.as_deref(),
            &account_id,
        ));
    }
    if let Some(path) = CodexAuthFile::default_path().filter(|path| path.is_file()) {
        match CodexAuthFile::load(&path) {
            Ok(auth) => {
                return Ok(LLMClient::new().with_openai_client(auth.openai_client()?));
            }
            Err(error) => eprintln!("ignoring {}: {error}", path.display()),
        }
    }
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        return Ok(LLMClient::new().with_openai_api_key(&api_key));
    }
    Err(
        io::Error::other("configure Codex OAuth, OAuth environment variables, or OPENAI_API_KEY")
            .into(),
    )
}

fn resolve_checkpoint(path: &Path) -> ExampleResult<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let pointer = std::fs::read_to_string(path.join("latest-checkpoint"))?;
    Ok(path.join(pointer.trim()))
}

fn is_sensitive_environment_name(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.ends_with("_API_KEY")
        || name.ends_with("_ACCESS_TOKEN")
        || name.ends_with("_REFRESH_TOKEN")
        || name.ends_with("_AUTH_TOKEN")
        || name.ends_with("_PASSWORD")
        || name.ends_with("_SECRET")
        || name == "OPENAI_API_KEY"
        || name == "ANTHROPIC_API_KEY"
        || name == "GOOGLE_API_KEY"
        || [
            "OPENAI_",
            "ANTHROPIC_",
            "GOOGLE_",
            "AWS_",
            "AZURE_",
            "CLOUDFLARE_",
            "TELEGRAM_",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

fn specification_hash(specification: &str) -> String {
    Sha256::digest(specification.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn bind_or_validate_run_inputs(
    state_dir: &Path,
    workspace: &Path,
    specification: &str,
    is_resume: bool,
) -> ExampleResult<()> {
    let manifest_path = state_dir.join("deepcoder-input.json");
    let expected_workspace = workspace.to_string_lossy();
    let expected_hash = specification_hash(specification);
    if is_resume {
        let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let saved_workspace = manifest
            .get("workspace")
            .and_then(Value::as_str)
            .ok_or_else(|| io::Error::other("run input manifest has no workspace"))?;
        let saved_hash = manifest
            .get("specification_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| io::Error::other("run input manifest has no specification hash"))?;
        if saved_workspace != expected_workspace || saved_hash != expected_hash {
            return Err(io::Error::other(
                "resume workspace or specification does not match the original run",
            )
            .into());
        }
        return Ok(());
    }

    if state_dir.join("workflow-run-id").exists() || manifest_path.exists() {
        return Err(io::Error::other(
            "state directory already belongs to a run; use --resume or choose another directory",
        )
        .into());
    }
    std::fs::create_dir_all(state_dir)?;
    let manifest = serde_json::json!({
        "workspace": expected_workspace,
        "specification_sha256": expected_hash,
    });
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

fn new_state_dir(workspace: &Path) -> PathBuf {
    let parent = workspace.parent().unwrap_or_else(|| Path::new("."));
    let workspace_name = workspace
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("workspace");
    parent.join(".arko-deepcoder-runs").join(format!(
        "{}-{}-{}",
        workspace_name,
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    ))
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    env_logger::init();
    let args = Args::parse();
    if args.ephemeral && args.resume.is_some() {
        return Err(io::Error::other("--ephemeral and --resume cannot be combined").into());
    }
    let workspace = std::fs::canonicalize(&args.workspace)?;
    if !workspace.is_dir() {
        return Err(io::Error::other("--workspace must be an existing directory").into());
    }
    let specification = std::fs::read_to_string(&args.spec)?;
    if specification.trim().is_empty() {
        return Err(io::Error::other("the specification is empty").into());
    }

    let resume_checkpoint = args.resume.as_deref().map(resolve_checkpoint).transpose()?;
    let state_dir = if let Some(checkpoint) = &resume_checkpoint {
        checkpoint
            .parent()
            .ok_or_else(|| io::Error::other("checkpoint has no parent directory"))?
            .to_path_buf()
    } else {
        args.state_dir
            .clone()
            .unwrap_or_else(|| new_state_dir(&workspace))
    };
    let state_dir = if args.ephemeral {
        state_dir
    } else {
        std::fs::create_dir_all(&state_dir)?;
        let state_dir = std::fs::canonicalize(state_dir)?;
        if state_dir.starts_with(&workspace) {
            return Err(io::Error::other(
                "the persistent state directory must be outside the mutable workspace",
            )
            .into());
        }
        bind_or_validate_run_inputs(
            &state_dir,
            &workspace,
            &specification,
            resume_checkpoint.is_some(),
        )?;
        state_dir
    };

    let orchestrator = agent_config(
        "deepcoder-orchestrator",
        "DeepCoder Orchestrator",
        Model::openai_gpt_5_6_sol(),
        OpenAIEffort::High,
        "You lead a code-defined implementation workflow. You decompose, delegate, and integrate evidence; you never edit source or run project commands yourself.",
        ORCHESTRATOR_INSTRUCTIONS,
        &["read_file", "list_files", "ask_user", "complete_task"],
    )?;
    let coder = agent_config(
        "deepcoder-coder",
        "DeepCoder Coder",
        Model::openai_gpt_5_6_sol(),
        OpenAIEffort::Medium,
        "You implement one bounded, dependency-aware subtask and do not approve your own work.",
        CODER_INSTRUCTIONS,
        &[
            "read_file",
            "write_file",
            "apply_diff",
            "list_files",
            "ask_user",
            "complete_task",
        ],
    )?;
    let mut correctness = agent_config(
        "deepcoder-correctness-verifier",
        "DeepCoder Correctness Verifier",
        Model::openai_gpt_5_6_sol(),
        OpenAIEffort::High,
        "You independently verify one coder subtask for correctness, completeness, and maintainability. You never edit source.",
        CORRECTNESS_INSTRUCTIONS,
        &["read_file", "list_files", "complete_task"],
    )?;
    correctness.complete_task_summary_prompt_override = Some(VERIFIER_OUTPUT_CONTRACT.to_string());
    let mut efficiency = agent_config(
        "deepcoder-efficiency-verifier",
        "DeepCoder Efficiency Verifier",
        Model::openai_gpt_5_6_sol(),
        OpenAIEffort::High,
        "You identify only concrete, material avoidable work. You never edit source and efficiency findings are advisory minor repairs.",
        EFFICIENCY_INSTRUCTIONS,
        &["read_file", "list_files", "complete_task"],
    )?;
    efficiency.complete_task_summary_prompt_override = Some(VERIFIER_OUTPUT_CONTRACT.to_string());
    let mut final_verifier = agent_config(
        "deepcoder-final-verifier",
        "DeepCoder Final Verifier",
        Model::openai_gpt_5_6_sol(),
        OpenAIEffort::High,
        "You independently verify the complete specification against the current workspace. You never edit source.",
        FINAL_VERIFIER_INSTRUCTIONS,
        &["read_file", "list_files", "complete_task"],
    )?;
    final_verifier.complete_task_summary_prompt_override =
        Some(VERIFIER_OUTPUT_CONTRACT.to_string());
    let researcher = agent_config(
        "deepcoder-researcher",
        "DeepCoder Researcher",
        Model::openai_gpt_5_6_luna(),
        OpenAIEffort::Max,
        "You answer one bounded repository research question with paths and direct evidence. You never implement.",
        RESEARCHER_INSTRUCTIONS,
        &["read_file", "list_files", "complete_task"],
    )?;

    let requirement_limit = specification.len().saturating_add(1_024);
    let memory = MemoryConfig::new()
        .tier(MemoryTier {
            name: "requirements".to_string(),
            content: specification,
            readers: AccessScope::All,
            writers: AccessScope::None,
            write_mode: WriteMode::FullEdit,
            entries: Vec::new(),
            max_bytes: Some(requirement_limit),
        })
        .tier(MemoryTier {
            name: "decisions".to_string(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::Only(HashSet::from(["orchestrator".to_string()])),
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: Some(512 * 1_024),
        })
        .tier(MemoryTier {
            name: "implementation_log".to_string(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::Only(HashSet::from(["coder".to_string()])),
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: Some(2 * 1_024 * 1_024),
        })
        .tier(MemoryTier {
            name: "verification_notes".to_string(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::Only(HashSet::from([
                "correctness_verifier".to_string(),
                "efficiency_verifier".to_string(),
                "final_verifier".to_string(),
            ])),
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            max_bytes: Some(1_024 * 1_024),
        });

    let shell = Arc::new(RunShellTool::new(
        args.shell_timeout_seconds,
        args.shell_output_bytes,
        CommandGuard {
            dangerously_allow_all_commands: args.dangerously_allow_all_commands,
            ..CommandGuard::default()
        },
    ));
    let client = Arc::new(build_client()?);
    let mut builder = Workflow::builder()
        .name("deepcoder-example")
        .metadata(WorkflowMetadata {
            description: "Sequential orchestrator/coder/verifier software implementation workflow"
                .to_string(),
            version: "1.0.0".to_string(),
            supported_platforms: vec![
                SupportedPlatform::new("linux"),
                SupportedPlatform::new("macos"),
            ],
            required_dependencies: vec![WorkflowDependency {
                name: "shell".to_string(),
                description: "Local command runner used for deterministic project checks"
                    .to_string(),
                executable: Some("bash".to_string()),
                install_hint: None,
                os: None,
                arch: None,
            }],
            tags: vec![
                "coding".to_string(),
                "verification".to_string(),
                "deepcoder".to_string(),
            ],
        })
        .workspace(workspace.clone())
        .llm_client(client.clone())
        .agent("orchestrator", orchestrator)
        .agent("coder", coder)
        .agent("correctness_verifier", correctness)
        .agent("efficiency_verifier", efficiency)
        .agent("final_verifier", final_verifier)
        .agent("researcher", researcher)
        .memory(memory)
        .budget(args.budget)
        .budget_warning(args.budget * 0.8)
        .context_freshness(ContextFreshnessConfig::default())
        .workflow_services(
            WorkflowServicesConfig::default()
                .allow_caller("orchestrator")
                .allow_target("coder")
                .allow_target("researcher")
                .max_active_calls(1),
        )
        .completion_verifier(
            CompletionVerifierConfig::correctness("subtask-correctness", "correctness_verifier")
                .target("coder")
                .minor_rejection_cap(args.max_minor_corrections)
                .major_rejection_cap(0)
                .reuse_conversation(true)
                .repair_context(REPAIR_CONTEXT_BYTES)
                .instructions(CORRECTNESS_GATE.to_string()),
        )
        .completion_verifier(
            CompletionVerifierConfig::efficiency("subtask-efficiency", "efficiency_verifier")
                .target("coder")
                .minor_rejection_cap(args.max_efficiency_corrections)
                .reuse_conversation(true)
                .repair_context(REPAIR_CONTEXT_BYTES)
                .instructions(EFFICIENCY_GATE.to_string()),
        )
        .completion_verifier(
            CompletionVerifierConfig::correctness("final-release", "final_verifier")
                .target("orchestrator")
                .minor_rejection_cap(args.max_final_corrections)
                .major_rejection_cap(0)
                .reuse_conversation(true)
                .repair_context(REPAIR_CONTEXT_BYTES)
                .instructions(FINAL_GATE.to_string()),
        )
        .agent_tool("coder", shell.clone())
        .agent_tool("correctness_verifier", shell.clone())
        .agent_tool("efficiency_verifier", shell.clone())
        .agent_tool("final_verifier", shell)
        .user_input_handler(Arc::new(TerminalInput))
        .hook(
            HookPoint::PostAgentCompletion,
            Arc::new(|context| {
                eprintln!(
                    "completed {:<24} total budget ${:.4}",
                    context.agent_name.as_deref().unwrap_or("unknown"),
                    context.budget_spent
                );
                HookResult::Continue
            }),
        )
        .hook(
            HookPoint::OnBudgetWarning,
            Arc::new(|context| {
                eprintln!("budget warning: ${:.4} spent", context.budget_spent);
                HookResult::Continue
            }),
        );

    if args.ephemeral {
        builder = builder.ephemeral();
    } else {
        builder = builder.persistence(PersistenceConfig::new(&state_dir));
    }
    if let Some(checkpoint) = resume_checkpoint {
        builder = builder.resume_from(checkpoint);
    }

    let workflow = builder.build()?;
    eprintln!("workflow run: {}", workflow.result().workflow_run_id);
    if !args.ephemeral {
        eprintln!("state directory: {}", state_dir.display());
    }

    let prompt = format!(
        "Implement the specification stored in the read-only `requirements` memory tier in workspace `{}`. Lead the work through small dependency-aware coder subtasks. Use the researcher only for bounded repository questions. Record durable decisions and require deterministic evidence. Do not implement code yourself. Complete only when every requirement is covered and the final verifier accepts the current workspace.",
        workspace.display()
    );
    let run = async {
        let orchestrator_result = workflow.run_agent("orchestrator", &prompt).await?;
        ExampleResult::Ok((orchestrator_result, workflow.finish().await?))
    }
    .await;
    // A token refresh still in flight holds the credentials until its new
    // tokens are saved: wait for it before the runtime shuts down.
    client.openai_chatgpt_oauth_credentials().await;
    let (orchestrator_result, result) = run?;

    println!("{}", orchestrator_result.summary);
    println!("workflow_run_id: {}", result.workflow_run_id);
    println!("total_cost: ${:.6}", result.total_cost);
    println!("recorded_agent_runs: {}", result.agent_results.len());
    Ok(())
}

const ORCHESTRATOR_INSTRUCTIONS: &str = r#"
Read the `requirements` memory before planning. Maintain a dependency-aware checklist and append
stable choices to `decisions`. Delegate one cohesive, independently verifiable slice at a time with
`run_workflow_agent` targeting `coder`. Give each coder the outcome, applicable requirements,
relevant files, constraints, and focused checks; do not prescribe edits you have not verified.
Use `researcher` only for bounded repository questions. Correctness, efficiency, and final verifier
roles are automatic gates and must never be invoked through `run_workflow_agent`.

Treat a successful coder response as correctness- and efficiency-reviewed evidence. If delegation
fails, inspect the error, narrow or reframe the subtask, and start a fresh coder run. Re-read shared
memory between slices. Never edit source or run commands yourself. Ask the user only when a missing
fact or product choice would
materially change the implementation. Call complete_task only after the whole specification is
covered; an independent final verifier will inspect the workspace and may return repairs.
"#;

const CODER_INSTRUCTIONS: &str = r#"
Implement only the delegated slice, but complete it end to end. Read relevant source and the
`requirements` and `decisions` memory tiers before editing. Preserve unrelated work. Use
write_file/apply_diff for edits and use run_shell for inspection and deterministic checks. Diagnose
failures and iterate. Do not deploy, publish, push, or contact production services.
Do not assume the workspace is a Git repository: check for `.git` before using Git. If Git reports
that the workspace is not a repository, record that fact in `implementation_log` and do not retry
Git commands in this run.

Before completion, append a concise entry to `implementation_log` containing the delegated outcome,
changed files, and exact successful checks. Call complete_task only when the slice is ready for
independent review. A verifier rejection is actionable feedback: retain context, fix the cause, and
rerun the relevant checks rather than merely editing the summary.
"#;

const CORRECTNESS_INSTRUCTIONS: &str = r#"
Remain read-only. Trace each delegated requirement to concrete code and deterministic evidence.
Inspect relevant call paths and data flow, check likely edge cases, and test an alternative failure
hypothesis before deciding. Use run_shell for focused checks, not broad speculative release work.
On the first pass, cover the relevant failure classes systematically and return all evidenced related
findings together. On repair passes, reuse prior evidence and inspect only the delta and affected paths.
Batch independent read-only calls; invoke run_shell, writes, messaging, and complete_task alone.
Do not assume the workspace is a Git repository: check for `.git` before using Git. If Git reports
that the workspace is not a repository, record that fact in `verification_notes` and do not retry
Git commands in this run.
Reject minor for localized repairable issues. Reject major when the approach is fundamentally wrong,
unsafe, or does not implement the delegated outcome. Append concise evidence and your decision to
`verification_notes`. Your complete_task summary must follow the required decision-line protocol.
"#;

const EFFICIENCY_INSTRUCTIONS: &str = r#"
Remain read-only. Look for concrete material waste: repeated file/network/database reads that could
be shared, avoidable N+1 work, poor asymptotic behavior on realistic inputs, redundant serialization,
or unnecessary external calls. Do not request speculative optimization or enterprise machinery.
Efficiency cannot issue a hard failure; express every actionable finding as REJECTED_MINOR. Your
evidence and decision should be appended to `verification_notes`. Your complete_task summary must be
only the required decision-line response.
Batch independent read-only calls, but invoke every other tool alone.
Check for `.git` before using Git. If Git reports that the workspace is not a repository, record that
fact in `verification_notes` and do not retry Git commands in this run.
"#;

const FINAL_VERIFIER_INSTRUCTIONS: &str = r#"
Remain read-only. Verify the entire requirements memory against the current workspace, including
cross-subtask integration, security and data boundaries, build/lint/test health, and omitted
requirements. Choose checks from the project's own documented tooling. Do not trust summaries as
proof. Append the release evidence and decision to `verification_notes`, then return only the
required decision-line response through complete_task.
Consolidate all evidenced related findings in the first verdict. For a repair, reuse prior evidence
and focus on the supplied delta and directly affected integration paths. Batch independent reads only.
Check for `.git` before using Git. If Git reports that the workspace is not a repository, record that
fact in `verification_notes` and do not retry Git commands in this run.
"#;

const RESEARCHER_INSTRUCTIONS: &str = r#"
Answer only the bounded question. Inspect the repository and return concise evidence with paths,
symbols, and implications. Do not edit files, propose unrelated architecture, or claim behavior you
did not inspect.
"#;

const CORRECTNESS_GATE: &str = r#"
Verify the coder's exact delegated outcome. Read requirements, decisions, implementation log, and
the actual changed code. Run focused deterministic checks. Use a compact semi-formal audit:
requirement -> implementation path -> observable evidence, followed by one plausible counterexample.
Return ACCEPTED only when the implementation, not merely its summary, is correct and complete.
"#;

const EFFICIENCY_GATE: &str = r#"
Review only material avoidable work in the accepted implementation. In particular, detect repeated
I/O or API calls whose result can safely be passed through a pipeline, N+1 behavior, and needless
whole-resource processing. Do not reject for stylistic micro-optimization. Any rejection is minor.
"#;

const FINAL_GATE: &str = r#"
Perform an independent whole-project release review against every item in the requirements memory.
Inspect source and run the repository's relevant deterministic check matrix. Reject invented or
summary-only evidence. Return specific repair slices for every real finding.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_output_keeps_head_and_tail_when_bounded() {
        let tool = RunShellTool::new(10, 1_024, CommandGuard::default());
        let input = format!("HEAD{}TAIL", "x".repeat(2_000));
        let output = tool.bounded_output(input.as_bytes());
        assert!(output.starts_with("HEAD"));
        assert!(output.ends_with("TAIL"));
        assert!(output.contains("bytes omitted"));
        assert!(output.len() < input.len());
    }

    #[test]
    fn run_directory_resolves_latest_checkpoint() {
        let directory = std::env::temp_dir().join(format!(
            "arko-deepcoder-example-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("latest-checkpoint"), "checkpoint-1.json\n").unwrap();
        let resolved = resolve_checkpoint(&directory).unwrap();
        assert_eq!(resolved, directory.join("checkpoint-1.json"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn sensitive_environment_names_are_removed_from_shell_children() {
        assert!(is_sensitive_environment_name("OPENAI_API_KEY"));
        assert!(is_sensitive_environment_name("NPM_AUTH_TOKEN"));
        assert!(is_sensitive_environment_name("DATABASE_PASSWORD"));
        assert!(is_sensitive_environment_name("AWS_SECRET_ACCESS_KEY"));
        assert!(!is_sensitive_environment_name("HOME"));
        assert!(!is_sensitive_environment_name("PATH"));
    }

    #[test]
    fn persistent_run_inputs_are_bound_and_validated() {
        let directory = std::env::temp_dir().join(format!(
            "arko-deepcoder-input-test-{}",
            uuid::Uuid::new_v4()
        ));
        let workspace = std::env::current_dir().unwrap();
        bind_or_validate_run_inputs(&directory, &workspace, "spec", false).unwrap();
        bind_or_validate_run_inputs(&directory, &workspace, "spec", true).unwrap();
        assert!(bind_or_validate_run_inputs(&directory, &workspace, "changed", true).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
