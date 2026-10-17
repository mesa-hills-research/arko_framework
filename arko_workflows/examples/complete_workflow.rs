//! A complete workflow: three agents write an incident report from a service
//! log. It uses agent configurations, the built-in tools, the workflow
//! builder, shared memory, messaging, completion verification, tool-result
//! distillation, checkpoints, and resuming.
//!
//! - `investigator` reads the log (a large `read_file` result reaches it
//!   distilled), appends its findings to the `findings` memory tier, and
//!   messages the writer.
//! - `writer` waits for that message, reads the findings, asks the user who
//!   owns the follow-up actions, and writes `INCIDENT.md`.
//! - `reviewer` is a completion verifier: it checks the report each time the
//!   writer calls `complete_task`, and a rejection sends the writer back to
//!   work with the feedback.
//!
//! The investigator and the writer run in parallel. The writer's question
//! suspends it, and the host answers with `continue_agent`. Every finalized
//! agent run is checkpointed; `--resume` builds the workflow from a
//! checkpoint and carries on from there.
//!
//! ```text
//! # Build and validate everything and write a checkpoint: no credentials, no network.
//! cargo run -p arko_workflows --example complete_workflow -- --data-dir /tmp/incident --dry-run
//!
//! # Run with ChatGPT OAuth from the Codex login ($CODEX_HOME/auth.json, else ~/.codex/auth.json).
//! cargo run -p arko_workflows --example complete_workflow -- --data-dir /tmp/incident
//!
//! # Resume from a checkpoint file, or from a run directory's latest checkpoint.
//! cargo run -p arko_workflows --example complete_workflow -- --data-dir /tmp/incident \
//!     --resume /tmp/incident/runs/<run>
//! ```
//!
//! `--data-dir` holds the agents' workspace (`workspace/`) and one directory
//! per run (`runs/<run>/`: checkpoints, task files, the event log).

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arko_agents::agent_config::PathSafetyPolicy;
use arko_agents::{
    AgentConfig, AgentLLMConfig, AgentToolConfig, DEFAULT_TOOL_TIMEOUT_SECONDS,
    EnvironmentDetailsConfig, ErrorRecoveryCause, ErrorRecoveryConfig, InterceptionFailurePolicy,
    LLMErrorPolicy, LLMRetryConfig, ToolCallingMode, ToolErrorPolicy, ToolResultLimits,
};
use arko_ai::{CodexAuthFile, GenerationConfig, LLMClient, Model, OpenAIEffort, ReasoningConfig};
use arko_workflows::persistence::{DEFAULT_EVENT_FLUSH_INTERVAL, PersistenceManager};
use arko_workflows::prelude::*;
use arko_workflows::workflow::cancel::DEFAULT_CANCEL_GRACE_PERIOD;
use clap::Parser;

type ExampleResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const WORKFLOW_NAME: &str = "incident-report";
// Registration names: every workflow setting names agents by these.
const INVESTIGATOR: &str = "investigator";
const WRITER: &str = "writer";
const REVIEWER: &str = "reviewer";
const BRIEF_TIER: &str = "brief";
const FINDINGS_TIER: &str = "findings";
const LOG_FILE: &str = "logs/service.log";
const CHANGELOG_FILE: &str = "CHANGELOG.md";
const REPORT_FILE: &str = "INCIDENT.md";
/// Rounds of answering questions and continuing paused runs before the
/// example stops and leaves the rest to `--resume`.
const MAX_FOLLOW_UP_ROUNDS: usize = 5;

#[derive(Debug, Parser)]
#[command(
    name = "complete_workflow",
    about = "Three agents write an incident report using memory, messaging, verification, distillation, checkpoints and resume"
)]
struct Args {
    /// Directory for the workspace (`workspace/`) and the run directories
    /// (`runs/`); created when missing. Keep it outside the repository.
    #[arg(long)]
    data_dir: PathBuf,

    /// Build and validate the workflow and write a checkpoint, without
    /// credentials or network calls.
    #[arg(long)]
    dry_run: bool,

    /// A checkpoint file, or a run directory (its `latest-checkpoint`), to
    /// resume from. It must be under `--data-dir`'s `runs/`.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// The answer the writer gets when it asks who owns the follow-up actions.
    #[arg(long, default_value = "the orders platform team")]
    owner: String,
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.data_dir)?;
    let data_dir = std::fs::canonicalize(&args.data_dir)?;
    let workspace = data_dir.join("workspace");
    seed_workspace(&workspace)?;

    // A new run gets a new run directory. A resumed run keeps its own: the
    // directory holds the run's identity (`workflow-run-id`), which the
    // checkpoint must match.
    let (run_dir, checkpoint) = match &args.resume {
        Some(path) => {
            let checkpoint = checkpoint_file(path)?;
            let run_dir = checkpoint
                .parent()
                .ok_or("the checkpoint has no run directory")?
                .to_path_buf();
            if !run_dir.starts_with(data_dir.join("runs")) {
                return Err("the checkpoint is not under this --data-dir's runs/ directory".into());
            }
            let state = PersistenceManager::load_checkpoint(&checkpoint).await?;
            println!(
                "resuming from {} (saved {}{})",
                checkpoint.display(),
                state.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                if state.cancellation.is_some() {
                    ", after the run was stopped"
                } else {
                    ""
                }
            );
            (run_dir, Some(checkpoint))
        }
        None => (new_run_dir(&data_dir), None),
    };

    let client = if args.dry_run {
        // No provider is configured: building a workflow sends no request.
        Arc::new(LLMClient::new())
    } else {
        Arc::new(chatgpt_oauth_client()?)
    };
    let workflow = build_workflow(&workspace, &run_dir, checkpoint.as_deref(), client.clone())?;
    println!(
        "workflow '{}', run {}, run directory {}",
        WORKFLOW_NAME,
        workflow.result().workflow_run_id,
        run_dir.display()
    );

    if args.dry_run {
        describe(&workflow);
        // Persistence is configured, so this writes `checkpoint-<id>.json`
        // in the run directory and points `latest-checkpoint` at it.
        workflow.checkpoint().await?;
        println!("checkpoint: {}", latest_checkpoint(&run_dir)?.display());
        return Ok(());
    }

    // Ctrl-C stops the run: running agents are cancelled, their results are
    // recorded, and a checkpoint records the stop. `--resume` goes on.
    let cancel = workflow.cancel_handle();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel.cancel();
        }
    });

    let outcome = run(&workflow, &args.owner).await;
    // A token refresh still in flight holds the credentials until its new
    // tokens are saved: wait for it before the runtime shuts down.
    client.openai_chatgpt_oauth_credentials().await;
    report(&workflow, &workspace, &run_dir, &data_dir);
    outcome
}

/// Registers the agents and configures the workflow. Settings that equal
/// their default are set anyway, with the default named, so that every
/// decision is visible.
fn build_workflow(
    workspace: &Path,
    run_dir: &Path,
    checkpoint: Option<&Path>,
    client: Arc<LLMClient>,
) -> ExampleResult<Workflow> {
    let memory = MemoryConfig::new()
        // Read-only context from the host: every agent reads it, none writes.
        .tier(MemoryTier {
            name: BRIEF_TIER.to_string(),
            content: INCIDENT_BRIEF.to_string(),
            readers: AccessScope::All,
            writers: AccessScope::None,
            // The default mode; with no writers it never applies.
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            // Default: no size limit.
            max_bytes: None,
        })
        // The investigator's findings: every agent reads them.
        .tier(MemoryTier {
            name: FINDINGS_TIER.to_string(),
            content: String::new(),
            readers: AccessScope::All,
            writers: AccessScope::Only(HashSet::from([INVESTIGATOR.to_string()])),
            // Default AppendOnly: each write is appended (FullEdit replaces).
            write_mode: WriteMode::AppendOnly,
            entries: Vec::new(),
            // Opt-in limit on the content, in bytes; a write past it fails.
            max_bytes: Some(16 * 1024),
        });

    let mut persistence = PersistenceConfig::new(run_dir)
        // Default true: a checkpoint after every finalized agent run.
        .auto_checkpoint(true)
        // Default Full: every agent event, streamed fragments included.
        .event_log(EventLog::Full)
        // Default one second: buffered events are written at least this often.
        .event_flush_interval(DEFAULT_EVENT_FLUSH_INTERVAL);
    // Default KeepAll: every checkpoint and task version is kept, so any
    // checkpoint can be resumed. `keep_last_checkpoints(n)` keeps the newest n.
    persistence.retain_checkpoints = CheckpointRetention::KeepAll;

    let mut builder = Workflow::builder()
        .name(WORKFLOW_NAME)
        // Must exist already; every agent works in it.
        .workspace(workspace)
        // Shared by the agents, the verifier's runs and the distiller.
        .llm_client(client)
        .agent(
            INVESTIGATOR,
            agent_config(
                "incident-investigator",
                "Incident investigator",
                "Investigates production incidents from logs and change history. Does not edit files.",
                INVESTIGATOR_INSTRUCTIONS,
                &["read_file", "list_files"],
                None,
            )?,
        )
        .agent(
            WRITER,
            agent_config(
                "incident-writer",
                "Incident writer",
                "Writes blameless incident reports in Markdown.",
                WRITER_INSTRUCTIONS,
                &[
                    "read_file",
                    "list_files",
                    "write_file",
                    "apply_diff",
                    "ask_user",
                ],
                None,
            )?,
        )
        .agent(
            REVIEWER,
            agent_config(
                "incident-reviewer",
                "Incident report reviewer",
                "Reviews incident reports against their sources. Never edits files.",
                REVIEWER_INSTRUCTIONS,
                &["read_file", "list_files"],
                // The verdict format, as the description of complete_task's
                // `summary` parameter.
                Some(VERIFIER_OUTPUT_CONTRACT),
            )?,
        )
        // Default: no memory tiers. An agent is offered read_memory,
        // write_memory and list_memories by its access to the tiers.
        .memory(memory)
        // Default MessagingPolicy::all(): every agent may message every other.
        // An agent with a permitted recipient is offered send_message and
        // list_recipients; one that another agent may message is offered
        // wait_for_message.
        .messaging(
            MessagingPolicy::all()
                .can_message(INVESTIGATOR, [WRITER])
                .can_message(WRITER, [INVESTIGATOR])
                .cannot_message(REVIEWER),
        )
        // Default Auto: a wait lasts just inside the agent's prompt-cache
        // lifetime; the agent calls wait_for_message again to wait longer.
        .wait_for_message_timeout(WaitForMessageTimeout::Auto)
        // Default: off. Tracks the files agents read and write; the
        // verifier's repair context below needs it.
        .context_freshness(ContextFreshnessConfig {
            // Default true: a write to a file that another agent run changed
            // after this run read it is rejected until this run reads it again.
            block_on_conflict: true,
            // Default mapping: read_file, list_files, write_file, apply_diff.
            ..ContextFreshnessConfig::default()
        })
        // Default: none. Gates the writer's complete_task on a review by the
        // reviewer agent.
        .completion_verifier(
            CompletionVerifierConfig::correctness("report-review", REVIEWER)
                // Required: the agents whose completions it reviews.
                .target(WRITER)
                .instructions(REVIEW_INSTRUCTIONS)
                // Default 3 (correctness): minor rejections allowed; the next
                // one fails the writer's task.
                .minor_rejection_cap(3)
                // Default 0: the first major rejection fails the task.
                .major_rejection_cap(0)
                // Default false: a re-review continues the reviewer's
                // conversation instead of starting a new one.
                .reuse_conversation(true)
                // Default none: a continued review is also given the tracked
                // file and memory changes since its verdict, up to this many
                // bytes.
                .repair_context(16 * 1024)
                // Default true: a new review (without conversation reuse) is
                // given the previous verdict and findings.
                .include_prior_findings(true)
                // Default 1: an unparseable verdict is asked for again once.
                .format_retries(1)
                // Default 1: after a review that fails (the reviewer's run
                // fails, or its verdict stays unusable), the writer is asked
                // once to call complete_task again; the next failure pauses
                // it (WorkflowError::VerifierFailed).
                .review_retries(1),
        )
        // Default: none. Replaces the investigator's large read_file results
        // with a model-written summary.
        .distiller(
            DistillationConfig::new("log-digest", Model::openai_gpt_6_luna())
                // Required: at least one agent and one tool.
                .target_agent(INVESTIGATOR)
                .target_tool("read_file")
                // Default 8,000: shorter results (in characters) are left alone.
                .min_chars(8_000)
                .instructions(DISTILL_INSTRUCTIONS)
                // Default none: no reasoning setting is sent (over ChatGPT
                // OAuth the request then asks for medium effort).
                .reasoning(ReasoningConfig::openai(OpenAIEffort::Low, false)),
            // max_output_tokens stays unset (default): no limit is sent over
            // ChatGPT OAuth, whose backend rejects one.
        )
        // Default UseRawResult: when the distiller fails, the agent gets the
        // raw tool result.
        .interception_failure_policy(InterceptionFailurePolicy::UseRawResult)
        // No user_input_handler (the default): an ask_user question suspends
        // the agent, and the host answers it with continue_agent.
        //
        // Default ApproveAll: every tool call an agent asks approval for is
        // approved. These agents auto-approve their tools, so none asks.
        .approval_policy(ApprovalPolicy::ApproveAll)
        // Default Automatic: after an agent's own retries run out, outages
        // and plan usage limits are waited out and the request is sent
        // again; errors that retrying can't fix fail the task. Pauses it
        // can't decide return to the caller.
        .error_recovery_policy(ErrorRecoveryPolicy::Automatic)
        // Default off: in run_agents_parallel, one agent's failure does not
        // stop the others.
        .fail_fast(false)
        // Default 30 seconds: how long a stopped run waits for each running
        // agent before aborting it.
        .cancel_grace_period(DEFAULT_CANCEL_GRACE_PERIOD)
        // Hooks are synchronous callbacks at lifecycle points; these log.
        .hook(
            HookPoint::OnAgentMessage,
            Arc::new(|context| {
                println!(
                    "[message] {} -> {}: {}",
                    context.agent_name.as_deref().unwrap_or("?"),
                    context
                        .metadata
                        .get("recipient")
                        .and_then(|value| value.as_str())
                        .unwrap_or("?"),
                    first_line(
                        context
                            .metadata
                            .get("content")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                    )
                );
                HookResult::Continue
            }),
        )
        .hook(
            HookPoint::PostAgentCompletion,
            Arc::new(|context| {
                println!(
                    "[completed] {}: {}",
                    context.agent_name.as_deref().unwrap_or("?"),
                    first_line(context.agent_summary.as_deref().unwrap_or_default())
                );
                HookResult::Continue
            }),
        )
        // Default: nothing is saved, and durable steps are unavailable.
        .persistence(persistence);
    if let Some(checkpoint) = checkpoint {
        // Restores memory, pending messages, recorded agent results (with
        // their tasks), verification and distillation records, the cost
        // ledger and freshness state. Agents, tools, policies and memory
        // access come from this builder.
        builder = builder.resume_from(checkpoint);
    }
    Ok(builder.build()?)
}

/// One agent's configuration. `id` and `name` describe the agent; the
/// workflow refers to it by the name it is registered under.
fn agent_config(
    id: &str,
    name: &str,
    role: &str,
    instructions: &str,
    built_in_tools: &[&str],
    summary_description: Option<&str>,
) -> ExampleResult<AgentConfig> {
    let llm_config = AgentLLMConfig {
        model: Model::openai_gpt_6_luna(),
        // Default: no reasoning setting (over ChatGPT OAuth the request then
        // asks for medium effort).
        generation_config: GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::Low, false)),
    };
    let tool_config = AgentToolConfig {
        // The tools offered besides complete_task, which is always offered.
        // Default: complete_task only. The workflow adds its own tools.
        available_tool_names: built_in_tools.iter().map(|tool| tool.to_string()).collect(),
        // Default Native: the provider's tool calling. OutputBased is legacy.
        tool_calling_mode: ToolCallingMode::Native,
        // Default false: every call except ask_user waits for approval,
        // which the workflow's ApprovalPolicy then decides.
        default_auto_approve_tool_use: true,
        // Default AutoReject: a path outside the workspace is refused.
        default_out_of_workspace_path_policy: PathSafetyPolicy::AutoReject,
        // Default false: a response may hold several calls; parallel-safe
        // ones run at the same time.
        one_tool_call_per_turn: false,
        // Default 20 minutes: the host-side limit of one tool call.
        tool_timeout_seconds: Some(DEFAULT_TOOL_TIMEOUT_SECONDS),
        // Default empty: per-tool overrides of the settings above.
        tool_specific_configs: HashMap::new(),
        // Default: the overflow guard on (a turn's results that would put the
        // next request over the model's input limit are shortened, with a
        // notice), and no caps.
        result_limits: ToolResultLimits::default(),
        // Default false: read_file reads whole files, and a file too large
        // for the model's context is shortened with a notice (pointing to the
        // terminal when one is offered). True adds start_line and end_line.
        read_file_line_ranges: false,
        // Default false: a reply in plain text, without a tool call, gets a
        // note asking for a tool call. True sends it to the user as a
        // message (AgentEvent::MessageToUser) and the agent carries on.
        text_reply_as_message: false,
        // Default false: the task ends with a complete_task call. True makes
        // a reply in plain text the final reply, which ends the task, and
        // complete_task is no longer offered.
        text_reply_ends_task: false,
        // Default false: an agent with tools. True makes it a chat: no tools
        // at all (the workflow adds none either), and its reply in text ends
        // its task.
        chat: false,
    };
    let mut builder = AgentConfig::builder(id.to_string(), name.to_string(), llm_config)
        .role_description(role.to_string())
        .instructions(instructions.to_string())
        .tool_config(tool_config)
        // Default: no environment details in the prompts.
        .environment_details_config(EnvironmentDetailsConfig::default())
        // Default InformAgentAndContinue: a failed tool call's error is its
        // result, and the agent goes on.
        .error_recovery_config(ErrorRecoveryConfig {
            on_tool_execution_error: ToolErrorPolicy::InformAgentAndContinue,
        })
        // Default: a failed model request is sent up to 10 times in all, then
        // the agent pauses, and the workflow's ErrorRecoveryPolicy decides.
        .llm_error_policy(LLMErrorPolicy::RetryThenPauseAndAwaitApproval { max_attempts: 10 })
        // Default: backoff from 2 to 64 seconds within a 5-minute budget, with
        // rate limits waited out after the provider's delay.
        .llm_retry_config(LLMRetryConfig::default())
        // Default None: a wait for a decision is not timed.
        .user_command_timeout_seconds(None);
    if let Some(description) = summary_description {
        // Default: the built-in description of complete_task's summary.
        builder = builder.complete_task_summary_prompt_override(description.to_string());
    }
    // Left at their defaults: no monitoring, no refusal fallback, no
    // reminder, no skills.
    Ok(builder.build()?)
}

/// ChatGPT OAuth from the Codex login. The client reads the file again
/// before each token refresh and writes rotated tokens back to it.
fn chatgpt_oauth_client() -> ExampleResult<LLMClient> {
    let path = CodexAuthFile::default_path().ok_or("no home directory for ~/.codex/auth.json")?;
    if !path.is_file() {
        return Err(format!("no Codex login at {}", path.display()).into());
    }
    let auth = CodexAuthFile::load(&path)?;
    println!("credentials from ChatGPT OAuth in {}", path.display());
    Ok(auth.client()?)
}

/// Starts the agents that have no usable run yet, answers questions and
/// continues paused runs until both agents have completed, and finishes the
/// run. The same code serves a new run and a resumed one.
async fn run(workflow: &Workflow, owner: &str) -> ExampleResult<()> {
    // A new run has no results. A resumed run has the checkpoint's: an agent
    // whose latest run completed or is paused is not started again.
    let starts = [(INVESTIGATOR, INVESTIGATOR_TASK), (WRITER, WRITER_TASK)]
        .into_iter()
        .filter(|(agent, _)| {
            latest_result(workflow, agent).is_none_or(|result| {
                result.status != AgentRunStatus::Completed && !result.status.is_paused()
            })
        })
        .map(|(agent, task)| (agent, task.to_string()))
        .collect::<Vec<_>>();
    if !starts.is_empty() {
        // Both are marked running before either is prompted, so each may
        // message the other from its first turn. A run that pauses is not a
        // failure: it comes back with the others, WaitingForInput or Paused.
        workflow.run_agents_parallel(starts).await?;
    }

    for _ in 0..MAX_FOLLOW_UP_ROUNDS {
        let paused = [INVESTIGATOR, WRITER]
            .into_iter()
            .filter_map(|agent| latest_result(workflow, agent))
            .filter(|result| result.status.is_paused())
            .collect::<Vec<_>>();
        if paused.is_empty() {
            break;
        }
        for result in paused {
            follow_up(workflow, &result, owner).await?;
        }
    }

    let unfinished = [INVESTIGATOR, WRITER]
        .into_iter()
        .filter(|agent| {
            latest_result(workflow, agent)
                .is_none_or(|result| result.status != AgentRunStatus::Completed)
        })
        .collect::<Vec<_>>();
    if !unfinished.is_empty() {
        // The checkpoint holds the paused runs; --resume continues them.
        workflow.checkpoint().await?;
        return Err(format!("not completed yet: {}", unfinished.join(", ")).into());
    }
    // Fires PostWorkflow and saves a final checkpoint.
    workflow.finish().await?;
    Ok(())
}

/// Continues one paused run: answers a question with `continue_agent`, or
/// decides an error-recovery pause with `continue_agent` or `resume_agent`.
async fn follow_up(workflow: &Workflow, paused: &AgentResult, owner: &str) -> ExampleResult<()> {
    let outcome = match paused.status {
        // The agent called ask_user, and no user input handler is set: its
        // result holds the questions, and the answer continues it.
        AgentRunStatus::WaitingForInput => {
            println!(
                "[question] {}: {}",
                paused.agent_name,
                paused.error.as_deref().unwrap_or_default()
            );
            println!("[answer] {owner}");
            let answer = format!("The follow-up actions are owned by {owner}.");
            workflow.continue_agent(paused, &answer).await
        }
        _ => match AgentPause::of(&paused.task.current_state) {
            // The reviewer could not review a completion
            // (WorkflowError::VerifierFailed): the prompt is the writer's
            // next message, and its next complete_task is reviewed again.
            Some(AgentPause::ErrorRecovery {
                cause: ErrorRecoveryCause::Middleware,
                ..
            }) => workflow.continue_agent(paused, REVIEW_RETRY_PROMPT).await,
            // Another error-recovery pause that ErrorRecoveryPolicy::Automatic
            // leaves to the caller: continue without a note, as the policy
            // does when it recovers on its own.
            Some(AgentPause::ErrorRecovery { .. }) => {
                workflow
                    .resume_agent(paused, AgentResume::ContinueAfterError { note: None })
                    .await
            }
            // ApprovalPolicy::ApproveAll approves every call, so no run pauses
            // for a tool approval.
            other => {
                return Err(format!("{}: unexpected pause {other:?}", paused.agent_name).into());
            }
        },
    };
    match outcome {
        // Paused again (a new question, or a pause): the next round goes on.
        Ok(_)
        | Err(
            WorkflowError::UserInputRequired { .. }
            | WorkflowError::AgentPaused { .. }
            | WorkflowError::VerifierFailed { .. },
        ) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// The agent's latest recorded run: a continuation replaces its run's entry,
/// and a new run is added after the earlier ones.
fn latest_result(workflow: &Workflow, agent: &str) -> Option<AgentResult> {
    workflow
        .result()
        .agent_results
        .into_iter()
        .rev()
        .find(|result| result.agent_name == agent)
}

/// Prints what the workflow set up for each agent, from its public state.
fn describe(workflow: &Workflow) {
    let context = workflow.context();
    let mut agents = context.agent_configs.iter().collect::<Vec<_>>();
    agents.sort_by_key(|(name, _)| name.as_str());
    for (name, config) in agents {
        println!(
            "agent '{name}' ({}, {}): {}",
            config.llm_config.model.id,
            workflow
                .agent_status(name)
                .map(|status| status.describe())
                .unwrap_or_default(),
            config.role_description
        );
        println!(
            "  built-in tools: {} (and complete_task)",
            config.tool_config.available_tool_names.join(", ")
        );
        let memory = context
            .memory
            .list_accessible(name)
            .into_iter()
            .map(|(tier, read, write)| match (read, write) {
                (true, true) => format!("{tier} (read/write)"),
                (true, false) => format!("{tier} (read)"),
                _ => format!("{tier} (write)"),
            })
            .collect::<Vec<_>>();
        println!("  memory: {}", list(memory));
        println!(
            "  may message {}, may be messaged by {}",
            list(context.message_recipients(name)),
            list(context.message_senders(name))
        );
    }
    for verifier in &context.completion_verifiers {
        println!(
            "completion verifier '{}': {} reviews {}",
            verifier.name,
            verifier.verifier_agent,
            list(sorted(&verifier.target_agents))
        );
    }
    for distiller in &context.distillation_configs {
        println!(
            "distiller '{}' ({}): {} results of {} from {} characters",
            distiller.name,
            distiller.model.id,
            list(sorted(&distiller.target_tools)),
            list(sorted(&distiller.target_agents)),
            distiller.min_chars
        );
    }
}

/// Prints the outcome: each agent's latest run, the reviews, and the
/// distillations, with the command that resumes the run.
fn report(workflow: &Workflow, workspace: &Path, run_dir: &Path, data_dir: &Path) {
    println!();
    for agent in [INVESTIGATOR, WRITER] {
        match latest_result(workflow, agent) {
            Some(result) => println!(
                "{agent}: {:?}{}",
                result.status,
                match (&result.error, result.summary.is_empty()) {
                    (Some(error), _) => format!(" ({})", first_line(error)),
                    (None, false) => format!(": {}", first_line(&result.summary)),
                    (None, true) => String::new(),
                }
            ),
            None => println!("{agent}: not run"),
        }
    }
    let context = workflow.context();
    for record in context.verification_records.read().iter() {
        println!(
            "review '{}' of {}: {:?}",
            record.verifier_name, record.target_agent, record.decision
        );
    }
    for record in context.distillation_records.read().iter() {
        println!(
            "distilled {} for {}: {} -> {} characters{}",
            record.tool_name,
            record.agent_name,
            record.input_chars,
            record.output_chars,
            record
                .rejection
                .as_deref()
                .map(|reason| format!(" (not used: {reason})"))
                .unwrap_or_default()
        );
    }
    println!("report: {}", workspace.join(REPORT_FILE).display());
    println!(
        "resume: --data-dir {} --resume {}",
        data_dir.display(),
        run_dir.display()
    );
}

/// `path` as a checkpoint file: the file itself, or the checkpoint a run
/// directory's `latest-checkpoint` names.
fn checkpoint_file(path: &Path) -> ExampleResult<PathBuf> {
    let path = std::fs::canonicalize(path)?;
    if path.is_file() {
        return Ok(path);
    }
    latest_checkpoint(&path)
}

/// The checkpoint that `latest-checkpoint` in `run_dir` names.
fn latest_checkpoint(run_dir: &Path) -> ExampleResult<PathBuf> {
    let name = std::fs::read_to_string(run_dir.join("latest-checkpoint"))?;
    Ok(run_dir.join(name.trim()))
}

/// A new run directory, `runs/<UTC time>-<random suffix>`.
fn new_run_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("runs").join(format!(
        "{}-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    ))
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

fn sorted(names: &HashSet<String>) -> Vec<String> {
    let mut names = names.iter().cloned().collect::<Vec<_>>();
    names.sort();
    names
}

fn list(names: Vec<String>) -> String {
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

/// Writes the files the agents work from, unless they exist: a resumed run
/// keeps the workspace as the agents left it.
fn seed_workspace(workspace: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(workspace.join("logs"))?;
    for (file, content) in [
        (LOG_FILE, service_log()),
        (CHANGELOG_FILE, CHANGELOG.to_string()),
    ] {
        let path = workspace.join(file);
        if !path.exists() {
            std::fs::write(path, content)?;
        }
    }
    Ok(())
}

/// Forty-five minutes of `orders-api` logs, the same every time: a deploy
/// lowers the database pool size, requests fail for 28 minutes, and a
/// rollback ends it. Metrics lines cover the minute before them, and a final
/// report totals them.
fn service_log() -> String {
    const fn at(hours: u32, minutes: u32, seconds: u32) -> u32 {
        hours * 3_600 + minutes * 60 + seconds
    }
    let clock = |seconds: u32| {
        format!(
            "2026-09-24T{:02}:{:02}:{:02}Z",
            seconds / 3_600,
            seconds / 60 % 60,
            seconds % 60
        )
    };
    let failing = at(14, 5, 0)..at(14, 33, 0);
    let events = [
        (
            at(14, 2, 10),
            r#"INFO  orders-api deploy started version=2.8.1 previous=2.8.0 change="db.pool_size 20 -> 5""#,
        ),
        (
            at(14, 2, 40),
            "INFO  orders-api deploy finished version=2.8.1 instances=3",
        ),
        (
            at(14, 12, 30),
            "WARN  orders-api alert fired name=orders-5xx-rate threshold=5.0% value=27.4%",
        ),
        (
            at(14, 31, 0),
            r#"INFO  orders-api rollback started version=2.8.0 reason="db pool exhaustion""#,
        ),
        (
            at(14, 31, 40),
            "INFO  orders-api rollback finished version=2.8.0 db.pool_size=20",
        ),
        (
            at(14, 36, 10),
            "INFO  orders-api alert resolved name=orders-5xx-rate value=0.0%",
        ),
    ];
    let mut log = String::new();
    let (mut requests_total, mut errors_total) = (0_u32, 0_u32);
    let mut metrics = |log: &mut String, time: u32| {
        let minute = time / 60;
        // The minute before `time` had failures.
        let (requests, errors, p99) = if failing.contains(&(time - 60)) {
            (
                1_150 + minute * 13 % 40,
                280 + minute * 37 % 90,
                2_004 + minute % 5,
            )
        } else {
            (1_180 + minute * 9 % 60, 0, 170 + minute % 5 * 4)
        };
        requests_total += requests;
        errors_total += errors;
        log.push_str(&format!(
            "{} INFO  orders-api metrics window=60s requests={requests} errors_5xx={errors} p99_latency_ms={p99}\n",
            clock(time)
        ));
    };
    for time in (at(13, 55, 0)..at(14, 40, 0)).step_by(10) {
        if time % 60 == 0 && time > at(13, 55, 0) {
            metrics(&mut log, time);
        }
        let request = time / 10 % 1_000_000;
        let order = 40_000 + time * 7 % 9_000;
        if failing.contains(&time) {
            if time % 30 == 0 {
                log.push_str(&format!(
                    "{} WARN  orders-api db pool exhausted active=5 idle=0 waiting={} wait_ms=2000\n",
                    clock(time),
                    25 + time % 21
                ));
            }
            log.push_str(&format!(
                "{} ERROR orders-api request failed method=GET path=/orders/{order} status=503 latency_ms=2003 reason=\"timeout acquiring db connection after 2000ms\" request_id=req-{request:06}\n",
                clock(time)
            ));
        } else {
            log.push_str(&format!(
                "{} INFO  orders-api request method=GET path=/orders/{order} status=200 latency_ms={} request_id=req-{request:06}\n",
                clock(time),
                40 + time % 37
            ));
        }
        for (_, event) in events.iter().filter(|(event_time, _)| *event_time == time) {
            log.push_str(&format!("{} {event}\n", clock(time)));
        }
    }
    let end = at(14, 40, 0);
    metrics(&mut log, end);
    log.push_str(&format!(
        "{} INFO  orders-api slo-report window=13:55:00-14:40:00 requests={requests_total} errors_5xx={errors_total} availability={:.2}%\n",
        clock(end),
        100.0 * f64::from(requests_total - errors_total) / f64::from(requests_total)
    ));
    log
}

const CHANGELOG: &str = "# orders-api changelog

## 2.8.1 (2026-09-24)

- Lower the default database connection pool size (`db.pool_size`) from 20 to 5, to reduce idle connections.
- Upgrade the HTTP server library.

## 2.8.0 (2026-09-10)

- Paginate `GET /orders`.
";

const INCIDENT_BRIEF: &str = "Customers reported failed order lookups on orders-api on 2026-09-24, in the afternoon (UTC).
The incident report goes in INCIDENT.md, with these sections: Summary, Timeline (UTC), Impact, Root cause, Resolution, Follow-up actions.
The sources are logs/service.log and CHANGELOG.md in the workspace. Every time and number in the report must appear in them.";

const INVESTIGATOR_INSTRUCTIONS: &str = "Work only from the workspace files and the shared memory. \
Quote timestamps and figures exactly as the sources give them. Do not edit files.";

const WRITER_INSTRUCTIONS: &str = "Write concise, blameless incident reports in Markdown. Use only \
facts from the findings and the workspace files; never invent times or numbers.";

const REVIEWER_INSTRUCTIONS: &str = "Check the report against its sources yourself rather than \
trusting the summary you are given. Do not edit files.";

const INVESTIGATOR_TASK: &str = "Investigate the incident described in the `brief` memory tier. \
Read logs/service.log and CHANGELOG.md. Append your findings to the `findings` memory tier in one \
write: the timeline with exact UTC timestamps, the impact figures as the log states them (the \
metrics lines and the final slo-report line), the root cause, and the resolution. Then send the \
writer a short message saying the findings are ready, with the root cause in one sentence. Then \
call complete_task with a two-sentence summary.";

const WRITER_TASK: &str = "Write the incident report described in the `brief` memory tier. First \
call wait_for_message until the investigator's message arrives; if the investigator has already \
finished, go on. Read the `findings` memory tier, and check details in logs/service.log when you \
need to. Then ask the user once, with ask_user, which team owns the follow-up actions. Then write \
INCIDENT.md, naming that team in the Follow-up actions section, and call complete_task with a \
short summary.";

const REVIEW_INSTRUCTIONS: &str = "Check INCIDENT.md against logs/service.log and CHANGELOG.md. It \
must have the sections Summary, Timeline (UTC), Impact, Root cause, Resolution and Follow-up \
actions; every timestamp and figure in it must appear in the sources; and the root cause must \
match the log and the changelog. The Follow-up actions section must name an owning team, which \
comes from the user and can't be checked against the sources. Reject minor for errors the writer \
can fix; reject major only when the report is missing or gives the wrong cause.";

const REVIEW_RETRY_PROMPT: &str =
    "The review could not run. Check INCIDENT.md and call complete_task again.";

const DISTILL_INSTRUCTIONS: &str = "Keep verbatim every line that is not a successful request: \
deploys, rollbacks, alerts, metrics, the slo-report, and the first and last of each run of WARN \
and ERROR lines with the count of lines between them. Replace successful request lines with the \
time ranges they cover. Never invent lines or numbers.";
