//! Running agents: one arko workflow per agent run, with limits and usage
//! accounting, and the benchmark run itself (solvers in every arm, graders
//! as answers come in), resumable and gracefully stoppable.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use arko_agents::runtime::event::AgentEvent;
use arko_agents::{AgentConfig, Tool};
use arko_ai::LLMClient;
use arko_workflows::prelude::*;
use arko_workflows::workflow::cancel::WorkflowCancelHandle;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agents::{ModelSettings, agent_config};
use crate::answer::{self, ParsedAnswer};
use crate::ledger::{Appender, read_jsonl};
use crate::prompts;
use crate::repos::{Layout, PreparedRepo};
use crate::shell::{ShellEnvironment, ShellSettings, ShellTool};
use crate::tasks::{BenchTask, TaskKind};
use crate::{Arm, BenchResult};

// ---------------------------------------------------------------------------
// One agent run.

/// What an agent run may use before it is stopped and counted as failed.
#[derive(Debug, Clone, Copy)]
pub struct JobLimits {
    /// Input tokens over all model calls, cached included.
    pub max_input_tokens: usize,
    /// Working time, waits for usage limits and retries excluded.
    pub timeout: Option<Duration>,
}

/// The agent runs in progress, so a second Ctrl-C can cancel them all.
#[derive(Default)]
pub struct ActiveRuns {
    runs: parking_lot::Mutex<HashMap<Uuid, WorkflowCancelHandle>>,
}

impl ActiveRuns {
    fn insert(&self, id: Uuid, handle: WorkflowCancelHandle) {
        self.runs.lock().insert(id, handle);
    }

    fn remove(&self, id: Uuid) {
        self.runs.lock().remove(&id);
    }

    pub fn cancel_all(&self) {
        for handle in self.runs.lock().values() {
            handle.cancel();
        }
    }

    pub fn len(&self) -> usize {
        self.runs.lock().len()
    }
}

/// Limits and waits of one agent run, shared with its monitor and tools.
#[derive(Default)]
pub struct JobControl {
    handle: std::sync::OnceLock<WorkflowCancelHandle>,
    limit: parking_lot::Mutex<Option<String>>,
    waiting_since: parking_lot::Mutex<Option<Instant>>,
    waited: parking_lot::Mutex<Duration>,
}

impl JobControl {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Stops the run because it went over a limit; the first reason stands.
    pub fn trip(&self, reason: String) {
        {
            let mut limit = self.limit.lock();
            if limit.is_some() {
                return;
            }
            *limit = Some(reason.clone());
        }
        if let Some(handle) = self.handle.get() {
            handle.stop(StopRequest::new(StopKind::Limit, reason));
        }
    }

    /// The callback a tool calls when the agent goes over its limit.
    pub fn limit_callback(self: &Arc<Self>) -> crate::shell::LimitCallback {
        let control = Arc::downgrade(self);
        Arc::new(move |reason| {
            if let Some(control) = control.upgrade() {
                control.trip(reason);
            }
        })
    }

    pub fn limit_reason(&self) -> Option<String> {
        self.limit.lock().clone()
    }

    fn start_wait(&self) {
        let mut since = self.waiting_since.lock();
        if since.is_none() {
            *since = Some(Instant::now());
        }
    }

    fn end_wait(&self) {
        if let Some(since) = self.waiting_since.lock().take() {
            *self.waited.lock() += since.elapsed();
        }
    }

    /// Time spent waiting for usage limits and retries so far.
    pub fn waited(&self) -> Duration {
        let current = self
            .waiting_since
            .lock()
            .map_or(Duration::ZERO, |since| since.elapsed());
        *self.waited.lock() + current
    }
}

/// The latest ChatGPT plan usage any agent reported.
#[derive(Default)]
pub struct PlanUsageLog {
    latest: parking_lot::Mutex<Option<(DateTime<Utc>, arko_ai::usage::PlanUsage)>>,
}

impl PlanUsageLog {
    fn record(&self, usage: arko_ai::usage::PlanUsage) {
        *self.latest.lock() = Some((Utc::now(), usage));
    }

    pub fn latest(&self) -> Option<(DateTime<Utc>, arko_ai::usage::PlanUsage)> {
        self.latest.lock().clone()
    }
}

/// Watches a run's events: counts input tokens against the limit, tracks
/// waits for usage limits and retries, and checks the time limit.
async fn monitor(
    mut events: tokio::sync::broadcast::Receiver<WorkflowEvent>,
    control: Arc<JobControl>,
    limits: JobLimits,
    started: Instant,
    plan_usage: Arc<PlanUsageLog>,
) {
    let mut input_tokens = 0usize;
    let mut tick = tokio::time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            event = events.recv() => {
                let event = match event {
                    Ok(event) => event.event,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                };
                match event {
                    AgentEvent::UsageReported(usage) => {
                        control.end_wait();
                        input_tokens += usage.input_tokens;
                        if input_tokens > limits.max_input_tokens {
                            control.trip(format!(
                                "input-token limit: {input_tokens} input tokens, over the limit of {}",
                                limits.max_input_tokens
                            ));
                        }
                    }
                    AgentEvent::UsageLimitReached { .. } | AgentEvent::RetryScheduled { .. } => {
                        control.start_wait();
                    }
                    AgentEvent::ErrorEncountered { requires_approval_to_retry: true, .. } => {
                        control.start_wait();
                    }
                    AgentEvent::AssistantResponseChunk(_)
                    | AgentEvent::ReasoningChunk(_)
                    | AgentEvent::ToolCallInitiated { .. } => control.end_wait(),
                    AgentEvent::PlanUsageReported(usage) => plan_usage.record(usage),
                    _ => {}
                }
            }
            _ = tick.tick() => {}
        }
        if let Some(timeout) = limits.timeout {
            let active = started.elapsed().saturating_sub(control.waited());
            if active > timeout {
                control.trip(format!(
                    "time limit: worked {} s, over the limit of {} s",
                    active.as_secs(),
                    timeout.as_secs()
                ));
            }
        }
    }
}

/// Everything one agent run needs.
pub struct AgentJob {
    pub client: Arc<LLMClient>,
    pub workspace: PathBuf,
    pub agent_name: &'static str,
    pub config: AgentConfig,
    pub tools: Vec<Arc<dyn Tool>>,
    pub control: Arc<JobControl>,
    pub limits: JobLimits,
    pub active: Arc<ActiveRuns>,
    pub plan_usage: Arc<PlanUsageLog>,
}

/// A running agent: its own workflow (one agent, its tools, no messaging),
/// and the monitor enforcing its limits.
pub struct AgentSession {
    workflow: Workflow,
    agent_name: &'static str,
    control: Arc<JobControl>,
    monitor: tokio::task::JoinHandle<()>,
    active: Arc<ActiveRuns>,
    id: Uuid,
    started: Instant,
}

impl AgentSession {
    pub fn start(job: AgentJob) -> BenchResult<Self> {
        let mut builder = Workflow::builder()
            .name("index-benchmark")
            .workspace(&job.workspace)
            .llm_client(job.client)
            .agent(job.agent_name, job.config)
            .disable_messaging()
            .cancel_grace_period(Duration::from_secs(10));
        for tool in job.tools {
            builder = builder.agent_tool(job.agent_name, tool);
        }
        let workflow = builder.build()?;
        let handle = workflow.cancel_handle();
        let _ = job.control.handle.set(handle.clone());
        let id = Uuid::new_v4();
        job.active.insert(id, handle);
        let started = Instant::now();
        let monitor = tokio::spawn(monitor(
            workflow.subscribe_events(),
            job.control.clone(),
            job.limits,
            started,
            job.plan_usage,
        ));
        Ok(Self {
            workflow,
            agent_name: job.agent_name,
            control: job.control,
            monitor,
            active: job.active,
            id,
            started,
        })
    }

    /// Runs the agent on `prompt`; the result, when the run recorded one.
    pub async fn ask(&self, prompt: &str) -> (Option<AgentResult>, Option<String>) {
        let outcome = self.workflow.run_agent(self.agent_name, prompt).await;
        self.settle(outcome)
    }

    /// Continues a completed run with `prompt`.
    pub async fn ask_again(
        &self,
        previous: &AgentResult,
        prompt: &str,
    ) -> (Option<AgentResult>, Option<String>) {
        let outcome = self.workflow.continue_agent(previous, prompt).await;
        self.settle(outcome)
    }

    fn settle(
        &self,
        outcome: Result<AgentResult, WorkflowError>,
    ) -> (Option<AgentResult>, Option<String>) {
        match outcome {
            Ok(result) => (Some(result), None),
            Err(error) => (
                self.workflow.result().agent_results.into_iter().last(),
                Some(error.to_string()),
            ),
        }
    }

    /// Ends the session: wall time and working time.
    pub fn finish(self) -> (Duration, Duration) {
        self.monitor.abort();
        self.active.remove(self.id);
        let wall = self.started.elapsed();
        let active = wall.saturating_sub(self.control.waited());
        (wall, active)
    }
}

/// Usage of one agent run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageSummary {
    /// Model calls (turns).
    pub llm_calls: usize,
    /// Tool calls of every tool, `complete_task` included.
    pub tool_calls: usize,
    pub shell_calls: usize,
    /// Shell commands that ran `arko_index` (arm B and C).
    pub index_calls: usize,
    /// Shell commands that ran `arko_index_dense` (arm C).
    pub dense_calls: usize,
    /// Input tokens over all calls (cached included), and the cached part.
    pub input_tokens: usize,
    pub cached_input_tokens: usize,
    /// Output tokens (reasoning included), and the reasoning part.
    pub output_tokens: usize,
    pub reasoning_tokens: usize,
    /// The largest input of one call: the context the run reached.
    pub max_context_tokens: usize,
    /// API list price of the tokens (ChatGPT OAuth bills nothing per call).
    pub api_cost: f64,
}

impl UsageSummary {
    pub fn of_task(task: &arko_agents::Task, model: &ModelSettings) -> Self {
        let mut summary = Self {
            llm_calls: task.usage_history.len(),
            ..Self::default()
        };
        for usage in &task.usage_history {
            let cached = usage.cached_tokens.unwrap_or(0);
            summary.input_tokens += usage.input_tokens;
            summary.cached_input_tokens += cached;
            summary.output_tokens += usage.output_tokens;
            summary.reasoning_tokens += usage.reasoning_tokens.unwrap_or(0);
            summary.max_context_tokens = summary.max_context_tokens.max(usage.input_tokens);
            summary.api_cost += model.api_cost(usage.input_tokens, cached, usage.output_tokens);
        }
        for message in &task.conversation.messages {
            for call in &message.tool_calls {
                summary.tool_calls += 1;
                if call.name != crate::shell::SHELL_TOOL {
                    continue;
                }
                summary.shell_calls += 1;
                let command = call
                    .arguments
                    .get("command")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let (index, dense) = crate::shell::index_programs(command);
                if dense {
                    summary.dense_calls += 1;
                } else if index {
                    summary.index_calls += 1;
                }
            }
        }
        summary
    }

    pub fn add(&mut self, other: &UsageSummary) {
        self.llm_calls += other.llm_calls;
        self.tool_calls += other.tool_calls;
        self.shell_calls += other.shell_calls;
        self.index_calls += other.index_calls;
        self.dense_calls += other.dense_calls;
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.max_context_tokens = self.max_context_tokens.max(other.max_context_tokens);
        self.api_cost += other.api_cost;
    }
}

/// How an agent run ended, for the benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The agent answered (`complete_task`).
    Answered,
    /// The run failed for a reason of its own (an error the model's requests
    /// can't get past, a context overflow): counted, scored 0.
    Failed,
    /// The run went over a limit (tokens, time, shell calls): counted,
    /// scored 0.
    Limit,
    /// The run ended for a reason outside the agent (the benchmark was
    /// stopped, the login was lost): not counted; it runs again.
    Infra,
}

impl Outcome {
    /// Whether this attempt settles its task and arm.
    pub fn is_final(self) -> bool {
        !matches!(self, Outcome::Infra)
    }
}

/// The outcome of a finished session, from its result and control.
pub fn classify(
    result: Option<&AgentResult>,
    error: Option<&str>,
    control: &JobControl,
) -> (Outcome, Option<String>, Option<String>) {
    if let Some(reason) = control.limit_reason() {
        return (Outcome::Limit, Some(reason), None);
    }
    let Some(result) = result else {
        return (
            Outcome::Infra,
            Some(error.unwrap_or("the run recorded no result").to_string()),
            None,
        );
    };
    let kind = result.failure_kind().map(|kind| format!("{kind:?}"));
    match result.status {
        AgentRunStatus::Completed => (Outcome::Answered, None, None),
        AgentRunStatus::Failed => {
            let detail = result.error.clone().or_else(|| error.map(str::to_string));
            match result.failure_kind() {
                Some(
                    arko_ai::ErrorKind::Authentication
                    | arko_ai::ErrorKind::Billing
                    | arko_ai::ErrorKind::Permission,
                ) => (Outcome::Infra, detail, kind),
                _ => (Outcome::Failed, detail, kind),
            }
        }
        _ => (
            Outcome::Infra,
            result
                .error
                .clone()
                .or_else(|| error.map(str::to_string))
                .or_else(|| Some(format!("ended {:?}", result.status))),
            kind,
        ),
    }
}

// ---------------------------------------------------------------------------
// The benchmark run.

/// Settings of a run. Those in [`RunManifest`] must not change when it is
/// resumed.
#[derive(Debug, Clone)]
pub struct RunSettings {
    pub name: String,
    pub arms: Vec<Arm>,
    pub samples: u32,
    pub concurrency: usize,
    pub model: ModelSettings,
    pub shell: ShellSettings,
    pub max_input_tokens: usize,
    pub attempt_timeout: Option<Duration>,
    pub max_attempts: u32,
    pub grade: bool,
    pub grade_only: bool,
    pub grader_shell: bool,
    pub seed: u64,
    pub limit: Option<usize>,
    pub mock: bool,
    pub tasks_file: PathBuf,
    pub index_cli: PathBuf,
    pub index_cli_sha256: String,
}

/// What a run was made with, saved in its directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub model: ModelSettings,
    pub mock: bool,
    pub shell_timeout_seconds: u64,
    pub shell_max_output_bytes: usize,
    pub shell_max_calls: usize,
    pub max_input_tokens: usize,
    pub attempt_timeout_seconds: Option<u64>,
    pub grader_shell: bool,
    pub index_cli_sha256: String,
    /// The arko_index CLI arms B and C run (its hash above must not change).
    #[serde(default)]
    pub index_cli: PathBuf,
    /// Settings that may grow on resume.
    pub arms: Vec<Arm>,
    pub samples: u32,
    pub tasks_file: PathBuf,
    /// Each start of the run: when, and with which task count.
    #[serde(default)]
    pub sessions: Vec<RunSession>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSession {
    pub started_at: DateTime<Utc>,
    pub tasks: usize,
    pub concurrency: usize,
}

impl RunManifest {
    fn of(settings: &RunSettings) -> Self {
        Self {
            name: settings.name.clone(),
            created_at: Utc::now(),
            model: settings.model.clone(),
            mock: settings.mock,
            shell_timeout_seconds: settings.shell.timeout.as_secs(),
            shell_max_output_bytes: settings.shell.max_output_bytes,
            shell_max_calls: settings.shell.max_calls,
            max_input_tokens: settings.max_input_tokens,
            attempt_timeout_seconds: settings.attempt_timeout.map(|timeout| timeout.as_secs()),
            grader_shell: settings.grader_shell,
            index_cli_sha256: settings.index_cli_sha256.clone(),
            index_cli: settings.index_cli.clone(),
            arms: settings.arms.clone(),
            samples: settings.samples,
            tasks_file: settings.tasks_file.clone(),
            sessions: Vec::new(),
        }
    }

    pub fn load(run_dir: &Path) -> BenchResult<Self> {
        let path = run_dir.join("manifest.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error} (no such run?)", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    fn save(&self, run_dir: &Path) -> BenchResult<()> {
        crate::write_atomic(
            &run_dir.join("manifest.json"),
            serde_json::to_string_pretty(self)?.as_bytes(),
        )
    }

    /// The differences that make a resumed run's results incomparable.
    fn incompatible_with(&self, other: &RunManifest) -> Vec<String> {
        let mut differences = Vec::new();
        let mut check = |name: &str, same: bool| {
            if !same {
                differences.push(name.to_string());
            }
        };
        check("model", self.model == other.model);
        check("mock", self.mock == other.mock);
        check(
            "shell settings",
            (
                self.shell_timeout_seconds,
                self.shell_max_output_bytes,
                self.shell_max_calls,
            ) == (
                other.shell_timeout_seconds,
                other.shell_max_output_bytes,
                other.shell_max_calls,
            ),
        );
        check(
            "limits",
            (self.max_input_tokens, self.attempt_timeout_seconds)
                == (other.max_input_tokens, other.attempt_timeout_seconds),
        );
        check("grader shell", self.grader_shell == other.grader_shell);
        check(
            "arko_index CLI",
            self.index_cli_sha256 == other.index_cli_sha256,
        );
        differences
    }
}

/// One solver run, as `attempts.jsonl` records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub attempt_id: String,
    pub task_id: String,
    pub repo: String,
    pub kind: TaskKind,
    pub arm: Arm,
    pub sample: u32,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<String>,
    /// The answer (`complete_task`'s summary).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    pub started_at: DateTime<Utc>,
    pub wall_ms: u64,
    /// Wall time less waits for usage limits and retries.
    pub active_ms: u64,
    pub usage: UsageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    /// Commands of the run named the benchmark's own files (task sets,
    /// records, transcripts) outside the checkout: its answer may come from
    /// a key or another arm's answer. Reported, and to be checked by hand.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub looked_outside: bool,
}

impl AttemptRecord {
    pub fn key(&self) -> JobKey {
        JobKey {
            task_id: self.task_id.clone(),
            arm: self.arm,
            sample: self.sample,
        }
    }
}

/// A grader run, as `grades.jsonl` records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradeRecord {
    pub grade_id: String,
    pub attempt_id: String,
    pub task_id: String,
    /// Whether the grading worked (a verdict that parses, for every item).
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<answer::GraderVerdict>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Lines naming search commands were removed from the answer the grader
    /// saw.
    pub scrubbed: bool,
    pub started_at: DateTime<Utc>,
    pub wall_ms: u64,
    pub usage: UsageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    /// The kind of error that failed the grader's run, when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<String>,
}

/// Whether a failure kind is a lost login or account problem: every run on
/// the credential fails until a person fixes it.
pub fn is_login_failure(kind: Option<&str>) -> bool {
    matches!(kind, Some("Authentication" | "Billing" | "Permission"))
}

/// The failure kind of a run's result, as records keep it.
pub fn failure_kind_of(result: Option<&AgentResult>) -> Option<String> {
    result
        .and_then(AgentResult::failure_kind)
        .map(|kind| format!("{kind:?}"))
}

/// Names of the benchmark's own files, which a solver has no business
/// reading.
const BENCHMARK_FILES: [&str; 8] = [
    "tasks.json",
    "candidates.jsonl",
    "verifications.jsonl",
    "writers.jsonl",
    "attempts.jsonl",
    "grades.jsonl",
    "prepare.json",
    "transcripts/",
];

/// Whether a run's commands name the benchmark's files outside its
/// checkout: one of [`BENCHMARK_FILES`], or a path under `out` other than
/// the run's checkout, index and tools.
pub fn looked_outside(task: &arko_agents::Task, out: &Path, repo: &str) -> bool {
    let out = out.display().to_string();
    let allowed = [
        format!("{out}/repos/{repo}"),
        format!("{out}/indexes/{repo}"),
        format!("{out}/bin/"),
        format!("{out}/scratch/"),
    ];
    task.conversation
        .messages
        .iter()
        .flat_map(|message| &message.tool_calls)
        .filter_map(|call| {
            call.arguments
                .get("command")
                .and_then(|value| value.as_str())
        })
        .any(|command| {
            BENCHMARK_FILES.iter().any(|name| command.contains(name))
                || command.match_indices(out.as_str()).any(|(at, _)| {
                    !allowed
                        .iter()
                        .any(|prefix| command[at..].starts_with(prefix.as_str()))
                })
        })
}

/// A task, arm and sample: one cell of the benchmark.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobKey {
    pub task_id: String,
    pub arm: Arm,
    pub sample: u32,
}

impl std::fmt::Display for JobKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.task_id, self.arm, self.sample)
    }
}

/// The work left: solver runs and gradings, in the order to start them.
#[derive(Debug, Default, PartialEq)]
pub struct Plan {
    pub solve: Vec<JobKey>,
    /// Attempt ids to grade.
    pub grade: Vec<String>,
    /// Cells given up after `max_attempts` runs that ended outside the
    /// agent's control.
    pub abandoned: Vec<JobKey>,
    /// Cells already settled.
    pub done: usize,
}

/// Grading attempts per answer before it is left ungraded.
pub const MAX_GRADE_ATTEMPTS: usize = 3;

/// The work left, from the records so far. Tasks are taken in an order
/// shuffled by `seed` (the same on every resume), and each task's arms run
/// together, in an order shuffled per task, so the arms see the same
/// conditions (time of day, backend load).
pub fn plan(
    tasks: &[BenchTask],
    arms: &[Arm],
    samples: u32,
    attempts: &[AttemptRecord],
    grades: &[GradeRecord],
    max_attempts: u32,
    seed: u64,
) -> Plan {
    let mut by_key: HashMap<JobKey, Vec<&AttemptRecord>> = HashMap::new();
    for attempt in attempts {
        by_key.entry(attempt.key()).or_default().push(attempt);
    }
    let mut grades_by_attempt: HashMap<&str, Vec<&GradeRecord>> = HashMap::new();
    for grade in grades {
        grades_by_attempt
            .entry(grade.attempt_id.as_str())
            .or_default()
            .push(grade);
    }
    let mut order: Vec<&BenchTask> = tasks.iter().collect();
    let mut rng = crate::report::Rng::new(seed);
    rng.shuffle(&mut order);
    let mut plan = Plan::default();
    for task in order {
        for sample in 0..samples {
            let mut task_arms = arms.to_vec();
            rng.shuffle(&mut task_arms);
            for arm in task_arms {
                let key = JobKey {
                    task_id: task.id.clone(),
                    arm,
                    sample,
                };
                let records = by_key.get(&key).map(Vec::as_slice).unwrap_or(&[]);
                if let Some(settled) = records
                    .iter()
                    .rev()
                    .find(|record| record.outcome.is_final())
                {
                    plan.done += 1;
                    if settled.outcome == Outcome::Answered {
                        let graded = grades_by_attempt
                            .get(settled.attempt_id.as_str())
                            .map(Vec::as_slice)
                            .unwrap_or(&[]);
                        if !graded.iter().any(|grade| grade.ok) && graded.len() < MAX_GRADE_ATTEMPTS
                        {
                            plan.grade.push(settled.attempt_id.clone());
                        }
                    }
                } else if records.len() >= max_attempts as usize {
                    plan.abandoned.push(key);
                } else {
                    plan.solve.push(key);
                }
            }
        }
    }
    plan
}

/// How far the run has to stop: 0 running, 1 no new jobs, 2 cancel the
/// running ones.
#[derive(Default)]
pub struct StopState {
    level: AtomicU8,
    reason: parking_lot::Mutex<Option<String>>,
}

impl StopState {
    pub fn level(&self) -> u8 {
        self.level.load(Ordering::SeqCst)
    }

    pub fn raise_to(&self, level: u8, reason: &str) {
        let previous = self.level.fetch_max(level, Ordering::SeqCst);
        if previous < level {
            let mut current = self.reason.lock();
            if current.is_none() {
                *current = Some(reason.to_string());
            }
        }
    }

    pub fn reason(&self) -> Option<String> {
        self.reason.lock().clone()
    }
}

/// On Ctrl-C or SIGTERM: first, start no new agent runs and let the
/// running ones finish; second, cancel them (they run again on resume);
/// third, exit at once.
pub fn watch_signals(stop: Arc<StopState>, active: Arc<ActiveRuns>) {
    tokio::spawn(async move {
        loop {
            #[cfg(unix)]
            {
                let mut terminate =
                    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    {
                        Ok(signal) => signal,
                        Err(_) => return,
                    };
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = terminate.recv() => {}
                }
            }
            #[cfg(not(unix))]
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            let level = stop.level() + 1;
            stop.raise_to(level, "signal");
            match level {
                1 => eprintln!(
                    "\nstopping: no new agent runs start; the {} running finish (signal again to cancel them; they run again on resume)",
                    active.len()
                ),
                2 => {
                    eprintln!("\ncancelling the running agents (signal again to exit at once)");
                    active.cancel_all();
                }
                _ => {
                    eprintln!("\nexiting");
                    std::process::exit(130);
                }
            }
        }
    });
}

pub struct RunInputs {
    pub layout: Layout,
    pub settings: RunSettings,
    pub tasks: Vec<BenchTask>,
    pub prepared: BTreeMap<String, PreparedRepo>,
    pub client: Arc<LLMClient>,
}

struct RunContext {
    layout: Layout,
    run_dir: PathBuf,
    settings: RunSettings,
    tasks: HashMap<String, BenchTask>,
    prepared: BTreeMap<String, PreparedRepo>,
    client: Arc<LLMClient>,
    solver_configs: HashMap<Arm, AgentConfig>,
    grader_config: AgentConfig,
    attempts: Appender,
    grades: Appender,
    active: Arc<ActiveRuns>,
    stop: Arc<StopState>,
    plan_usage: Arc<PlanUsageLog>,
    dense_socket: Option<PathBuf>,
}

enum JobDone {
    Solved(Box<AttemptRecord>),
    Graded(Box<GradeRecord>),
}

pub async fn run(inputs: RunInputs) -> BenchResult<String> {
    let RunInputs {
        layout,
        settings,
        tasks,
        prepared,
        client,
    } = inputs;
    let run_dir = layout.run_dir(&settings.name);
    std::fs::create_dir_all(run_dir.join("transcripts"))?;
    let mut manifest = RunManifest::of(&settings);
    if let Ok(existing) = RunManifest::load(&run_dir) {
        let differences = existing.incompatible_with(&manifest);
        if !differences.is_empty() {
            return Err(format!(
                "run {} was started with other settings ({}); use another --name, or the same settings",
                settings.name,
                differences.join(", ")
            )
            .into());
        }
        let mut arms = existing.arms.clone();
        for arm in &settings.arms {
            if !arms.contains(arm) {
                arms.push(*arm);
            }
        }
        arms.sort();
        manifest = RunManifest {
            arms,
            samples: existing.samples.max(settings.samples),
            created_at: existing.created_at,
            sessions: existing.sessions,
            ..manifest
        };
    }
    manifest.sessions.push(RunSession {
        started_at: Utc::now(),
        tasks: tasks.len(),
        concurrency: settings.concurrency,
    });
    manifest.save(&run_dir)?;

    let attempts_path = run_dir.join("attempts.jsonl");
    let grades_path = run_dir.join("grades.jsonl");
    let attempts: Vec<AttemptRecord> = read_jsonl(&attempts_path)?;
    let grades: Vec<GradeRecord> = read_jsonl(&grades_path)?;
    let mut task_list = tasks.clone();
    if let Some(limit) = settings.limit {
        // The pilot subset is the first tasks of the run's own order.
        let mut order: Vec<BenchTask> = task_list.clone();
        crate::report::Rng::new(settings.seed).shuffle(&mut order);
        order.truncate(limit);
        task_list = order;
    }
    let mut plan = plan(
        &task_list,
        &settings.arms,
        settings.samples,
        &attempts,
        &grades,
        settings.max_attempts,
        settings.seed,
    );
    if settings.grade_only {
        plan.solve.clear();
    }
    if !settings.grade && !settings.grade_only {
        plan.grade.clear();
    }

    let mut solver_configs = HashMap::new();
    for arm in Arm::ALL {
        solver_configs.insert(arm, solver_config(&settings, arm)?);
    }
    let grader_config = agent_config(
        "grader",
        "Grader",
        prompts::GRADER_ROLE,
        prompts::GRADER_INSTRUCTIONS,
        "Exactly one JSON object: the items, the wrong claims, the verdict and notes, as your instructions describe.",
        &settings.model,
        settings.shell.timeout.as_secs() + 60,
    )?;
    let dense_service = if settings.arms.contains(&Arm::C) && !settings.grade_only {
        Some(crate::dense::DenseService::start(&run_dir, &prepared, settings.mock).await?)
    } else {
        None
    };
    let context = Arc::new(RunContext {
        layout,
        run_dir: run_dir.clone(),
        tasks: tasks
            .into_iter()
            .map(|task| (task.id.clone(), task))
            .collect(),
        prepared,
        client,
        solver_configs,
        grader_config,
        attempts: Appender::open(&attempts_path)?,
        grades: Appender::open(&grades_path)?,
        active: Arc::new(ActiveRuns::default()),
        stop: Arc::new(StopState::default()),
        plan_usage: Arc::new(PlanUsageLog::default()),
        dense_socket: dense_service
            .as_ref()
            .map(|service| service.socket().to_path_buf()),
        settings,
    });
    let _ = std::fs::remove_file(run_dir.join("STOP"));
    watch_signals(context.stop.clone(), context.active.clone());

    let attempts_by_id: HashMap<String, AttemptRecord> = attempts
        .into_iter()
        .map(|attempt| (attempt.attempt_id.clone(), attempt))
        .collect();
    let mut solve_queue: VecDeque<JobKey> = plan.solve.into_iter().collect();
    let mut grade_queue: VecDeque<AttemptRecord> = plan
        .grade
        .iter()
        .filter_map(|id| attempts_by_id.get(id).cloned())
        .collect();
    let total_solve = solve_queue.len();
    let total_grade_known = grade_queue.len();
    eprintln!(
        "run {}: {} cells settled before, {} solver runs and {} gradings to do, {} given up; {} at once{}",
        context.settings.name,
        plan.done,
        total_solve,
        total_grade_known,
        plan.abandoned.len(),
        context.settings.concurrency,
        if context.settings.mock {
            " (mock model)"
        } else {
            ""
        }
    );
    eprintln!(
        "stop gracefully with Ctrl-C, SIGTERM or `touch {}`",
        run_dir.join("STOP").display()
    );

    let started = Instant::now();
    let mut running: tokio::task::JoinSet<JobDone> = tokio::task::JoinSet::new();
    let (mut solved, mut graded, mut answered) = (0usize, 0usize, 0usize);
    let mut last_status = Instant::now();
    loop {
        if run_dir.join("STOP").exists() {
            context.stop.raise_to(1, "STOP file");
        }
        while running.len() < context.settings.concurrency && context.stop.level() == 0 {
            if let Some(attempt) = grade_queue.pop_front() {
                let context = context.clone();
                running.spawn(
                    async move { JobDone::Graded(Box::new(grade(&context, attempt).await)) },
                );
            } else if let Some(key) = solve_queue.pop_front() {
                let context = context.clone();
                running.spawn(async move { JobDone::Solved(Box::new(solve(&context, key).await)) });
            } else {
                break;
            }
        }
        if running.is_empty() {
            break;
        }
        let done = tokio::select! {
            done = running.join_next() => done,
            _ = tokio::time::sleep(Duration::from_secs(2)) => None,
        };
        match done {
            Some(Ok(JobDone::Solved(record))) => {
                solved += 1;
                let line = format!(
                    "[solve {solved}/{total_solve}] {}/{}/{} {:?}{} {:.0}s, {} calls, {} shell ({} index), {}k in, {}k out",
                    record.task_id,
                    record.arm,
                    record.sample,
                    record.outcome,
                    record
                        .detail
                        .as_ref()
                        .map_or(String::new(), |detail| format!(
                            " ({})",
                            first_line(detail, 100)
                        )),
                    record.wall_ms as f64 / 1000.0,
                    record.usage.llm_calls,
                    record.usage.shell_calls,
                    record.usage.index_calls,
                    record.usage.input_tokens / 1000,
                    record.usage.output_tokens / 1000,
                );
                eprintln!("{line}");
                if record.outcome == Outcome::Answered {
                    answered += 1;
                    if context.settings.grade {
                        grade_queue.push_back(*record.clone());
                    }
                }
                if record.outcome == Outcome::Infra
                    && is_login_failure(record.failure_kind.as_deref())
                {
                    eprintln!(
                        "the ChatGPT login failed ({}); stopping the run. Fix the login, then run again to resume.",
                        record.detail.as_deref().unwrap_or("")
                    );
                    context.stop.raise_to(1, "login failure");
                }
            }
            Some(Ok(JobDone::Graded(record))) => {
                if !record.ok && is_login_failure(record.failure_kind.as_deref()) {
                    eprintln!(
                        "the ChatGPT login failed while grading ({}); stopping the run. Fix the login, then run again to resume.",
                        record.detail.as_deref().unwrap_or("")
                    );
                    context.stop.raise_to(1, "login failure");
                }
                graded += 1;
                eprintln!(
                    "[grade {graded}] {} {}{}",
                    record.task_id,
                    match (record.ok, record.pass) {
                        (true, Some(true)) => "correct".to_string(),
                        (true, _) => format!(
                            "{} (items {:.2})",
                            record
                                .verdict
                                .as_ref()
                                .map_or("?", |verdict| verdict.verdict.as_str()),
                            record.item_score.unwrap_or(0.0)
                        ),
                        (false, _) => "grading failed".to_string(),
                    },
                    record
                        .detail
                        .as_ref()
                        .map_or(String::new(), |detail| format!(
                            " ({})",
                            first_line(detail, 100)
                        ))
                );
            }
            Some(Err(error)) => eprintln!("an agent job panicked: {error}"),
            None => {}
        }
        if last_status.elapsed() > Duration::from_secs(60) {
            last_status = Instant::now();
            let status = serde_json::json!({
                "at": Utc::now(),
                "elapsed_s": started.elapsed().as_secs(),
                "solver_runs_done": solved,
                "solver_runs_left": solve_queue.len(),
                "gradings_done": graded,
                "gradings_queued": grade_queue.len(),
                "running": running.len(),
                "stop_level": context.stop.level(),
                "plan_usage": context.plan_usage.latest(),
            });
            eprintln!(
                "status: {} solver runs left, {} gradings queued, {} running, {:.0} min",
                solve_queue.len(),
                grade_queue.len(),
                running.len(),
                started.elapsed().as_secs_f64() / 60.0
            );
            let _ = crate::write_atomic(
                &run_dir.join("progress.json"),
                serde_json::to_string_pretty(&status)?.as_bytes(),
            );
        }
    }
    if let Some(service) = dense_service {
        service.stop();
    }
    let stopped = context.stop.reason();
    let mut summary = format!(
        "run {}: {solved} solver runs ({answered} answered) and {graded} gradings in {:.1} min",
        context.settings.name,
        started.elapsed().as_secs_f64() / 60.0
    );
    if let Some(reason) = stopped {
        summary.push_str(&format!(
            "; stopped ({reason}) with {} solver runs and {} gradings left: run the same command again to resume",
            solve_queue.len(),
            grade_queue.len()
        ));
    }
    let manifest = RunManifest::load(&run_dir)?;
    let task_list: Vec<BenchTask> = context.tasks.values().cloned().collect();
    match crate::report::write(&run_dir, &manifest, &task_list, &context.prepared, 2000) {
        Ok(_) => summary.push_str(&format!(
            "\nreport: {}",
            run_dir.join("report.md").display()
        )),
        Err(error) => summary.push_str(&format!("\nthe report failed: {error}")),
    }
    let _ = &context.layout;
    Ok(summary)
}

fn first_line(text: &str, max: usize) -> String {
    let line = text.lines().next().unwrap_or("");
    if line.chars().count() > max {
        format!("{}...", line.chars().take(max).collect::<String>())
    } else {
        line.to_string()
    }
}

/// The solver's configuration in `arm`.
fn solver_config(settings: &RunSettings, arm: Arm) -> BenchResult<AgentConfig> {
    agent_config(
        "solver",
        "Solver",
        prompts::SOLVER_ROLE,
        &solver_instructions(arm, which_rg()),
        "Your whole answer, in the format of your instructions: ANSWER: then LOCATIONS:.",
        &settings.model,
        settings.shell.timeout.as_secs() + 60,
    )
}

/// The solver's instructions in `arm`.
pub fn solver_instructions(arm: Arm, has_rg: bool) -> String {
    let rg = if has_rg {
        ", including `rg` (ripgrep)"
    } else {
        ""
    };
    let tools = match arm {
        Arm::A => prompts::TOOLS_TERMINAL.replace("{rg}", rg),
        Arm::B => prompts::TOOLS_INDEX.replace("{rg}", rg),
        Arm::C => format!(
            "{}{}",
            prompts::TOOLS_INDEX.replace("{rg}", rg),
            prompts::TOOLS_DENSE
        ),
    };
    prompts::SOLVER_INSTRUCTIONS.replace("{tools}", &tools)
}

/// Whether ripgrep is installed (every arm's prompt says so the same way).
pub fn which_rg() -> bool {
    std::process::Command::new("rg")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The task prompt a solver gets.
pub fn solver_prompt(task: &BenchTask, repo: &PreparedRepo) -> String {
    format!(
        "Repository: {} ({}), at {} (commit {}), checked out in your working directory.\n\nQuestion ({}): {}\n{}\n{}",
        repo.spec.id,
        repo.spec.description,
        repo.spec.rev,
        &repo.commit[..repo.commit.len().min(12)],
        task.kind,
        task.question.trim(),
        task.scope
            .as_ref()
            .map_or(String::new(), |scope| format!("Scope: {scope}\n")),
        prompts::kind_guidance(task.kind.as_str())
    )
}

/// The shell environment of `arm` for a job in `repo`.
fn shell_environment(
    context: &RunContext,
    arm: Arm,
    repo: &PreparedRepo,
    scratch: &Path,
) -> ShellEnvironment {
    arm_environment(
        &context.layout,
        arm,
        repo,
        scratch,
        context.dense_socket.as_deref(),
    )
}

pub fn arm_environment(
    layout: &Layout,
    arm: Arm,
    repo: &PreparedRepo,
    scratch: &Path,
    dense_socket: Option<&Path>,
) -> ShellEnvironment {
    let mut vars = vec![("TMPDIR".to_string(), scratch.as_os_str().to_owned())];
    if arm != Arm::A {
        vars.push((
            "ARKO_INDEX_STATE".to_string(),
            repo.index.as_os_str().to_owned(),
        ));
    }
    if arm == Arm::C {
        if let Some(socket) = dense_socket {
            vars.push((
                crate::dense::SOCKET_VARIABLE.to_string(),
                socket.as_os_str().to_owned(),
            ));
        }
        vars.push((
            crate::dense::REPO_VARIABLE.to_string(),
            repo.spec.id.clone().into(),
        ));
    }
    ShellEnvironment {
        vars,
        path_prefix: vec![layout.bin_dir(arm)],
    }
}

/// A scratch directory for one job's temporary files.
fn scratch_dir(context: &RunContext, id: &str) -> PathBuf {
    let dir = context.layout.scratch_dir().join(id);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn save_transcript(run_dir: &Path, name: &str, task: Option<&arko_agents::Task>) -> Option<String> {
    let task = task?;
    let relative = format!("transcripts/{name}.json");
    let text = serde_json::to_string(task).ok()?;
    crate::write_atomic(&run_dir.join(&relative), text.as_bytes()).ok()?;
    Some(relative)
}

async fn solve(context: &RunContext, key: JobKey) -> AttemptRecord {
    let attempt_id = Uuid::new_v4().simple().to_string();
    let started_at = Utc::now();
    let task = &context.tasks[&key.task_id];
    let mut record = AttemptRecord {
        attempt_id: attempt_id.clone(),
        task_id: task.id.clone(),
        repo: task.repo.clone(),
        kind: task.kind,
        arm: key.arm,
        sample: key.sample,
        outcome: Outcome::Infra,
        detail: None,
        failure_kind: None,
        answer: None,
        started_at,
        wall_ms: 0,
        active_ms: 0,
        usage: UsageSummary::default(),
        transcript: None,
        looked_outside: false,
    };
    let Some(repo) = context.prepared.get(&task.repo) else {
        record.detail = Some(format!("repository {} is not prepared", task.repo));
        let _ = context.attempts.append(&record);
        return record;
    };
    let scratch = scratch_dir(context, &attempt_id);
    let control = JobControl::new();
    let shell = ShellTool::new(
        context.settings.shell.clone(),
        shell_environment(context, key.arm, repo, &scratch),
        Some(control.limit_callback()),
    );
    let session = AgentSession::start(AgentJob {
        client: context.client.clone(),
        workspace: repo.checkout.clone(),
        agent_name: "solver",
        config: context.solver_configs[&key.arm].clone(),
        tools: vec![Arc::new(shell)],
        control: control.clone(),
        limits: JobLimits {
            max_input_tokens: context.settings.max_input_tokens,
            timeout: context.settings.attempt_timeout,
        },
        active: context.active.clone(),
        plan_usage: context.plan_usage.clone(),
    });
    let session = match session {
        Ok(session) => session,
        Err(error) => {
            record.detail = Some(format!("the agent could not start: {error}"));
            let _ = context.attempts.append(&record);
            return record;
        }
    };
    let (result, error) = session.ask(&solver_prompt(task, repo)).await;
    let (wall, active) = session.finish();
    let (outcome, detail, failure_kind) = classify(result.as_ref(), error.as_deref(), &control);
    record.outcome = outcome;
    record.detail = detail;
    record.failure_kind = failure_kind;
    record.wall_ms = wall.as_millis() as u64;
    record.active_ms = active.as_millis() as u64;
    if let Some(result) = &result {
        record.usage = UsageSummary::of_task(&result.task, &context.settings.model);
        record.looked_outside = looked_outside(&result.task, &context.layout.out, &task.repo);
        if outcome == Outcome::Answered {
            record.answer = Some(result.summary.clone());
        }
    }
    record.transcript = save_transcript(
        &context.run_dir,
        &format!("solve-{attempt_id}"),
        result.as_ref().map(|result| &result.task),
    );
    let _ = std::fs::remove_dir_all(&scratch);
    // A run cancelled by the operator's stop runs again on resume without
    // using up one of its attempts.
    let cancelled = outcome == Outcome::Infra && context.stop.level() >= 2;
    if !cancelled && let Err(error) = context.attempts.append(&record) {
        eprintln!("could not record attempt {attempt_id}: {error}");
    }
    record
}

/// The grader's prompt: the question, the key and the candidate answer,
/// nothing about the arm.
pub fn grader_prompt(task: &BenchTask, repo: &PreparedRepo, candidate: &str) -> String {
    format!(
        "Repository: {} ({}), at {}, checked out in your working directory.\n\nQuestion ({}): {}\n{}\nANSWER KEY\n{}\nCANDIDATE ANSWER\n<<<\n{}\n>>>\n\nJudge every item of the key, list the wrong claims, and give your verdict as the JSON object your instructions describe.",
        repo.spec.id,
        repo.spec.description,
        repo.spec.rev,
        task.kind,
        task.question.trim(),
        task.scope
            .as_ref()
            .map_or(String::new(), |scope| format!("Scope: {scope}\n")),
        task.key.render(),
        candidate.trim()
    )
}

async fn grade(context: &RunContext, attempt: AttemptRecord) -> GradeRecord {
    let grade_id = Uuid::new_v4().simple().to_string();
    let task = &context.tasks[&attempt.task_id];
    let mut record = GradeRecord {
        grade_id: grade_id.clone(),
        attempt_id: attempt.attempt_id.clone(),
        task_id: task.id.clone(),
        ok: false,
        verdict: None,
        item_score: None,
        pass: None,
        detail: None,
        scrubbed: false,
        started_at: Utc::now(),
        wall_ms: 0,
        usage: UsageSummary::default(),
        transcript: None,
        failure_kind: None,
    };
    let Some(repo) = context.prepared.get(&task.repo) else {
        record.detail = Some(format!("repository {} is not prepared", task.repo));
        return record;
    };
    let (candidate, scrubbed) = answer::scrub_for_grader(attempt.answer.as_deref().unwrap_or(""));
    record.scrubbed = scrubbed;
    let scratch = scratch_dir(context, &grade_id);
    let control = JobControl::new();
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    if context.settings.grader_shell {
        tools.push(Arc::new(ShellTool::new(
            context.settings.shell.clone(),
            shell_environment(context, Arm::A, repo, &scratch),
            Some(control.limit_callback()),
        )));
    }
    let session = match AgentSession::start(AgentJob {
        client: context.client.clone(),
        workspace: repo.checkout.clone(),
        agent_name: "grader",
        config: context.grader_config.clone(),
        tools,
        control: control.clone(),
        limits: JobLimits {
            max_input_tokens: context.settings.max_input_tokens,
            timeout: context.settings.attempt_timeout,
        },
        active: context.active.clone(),
        plan_usage: context.plan_usage.clone(),
    }) {
        Ok(session) => session,
        Err(error) => {
            record.detail = Some(format!("the grader could not start: {error}"));
            let _ = context.grades.append(&record);
            return record;
        }
    };
    let (mut result, mut error) = session.ask(&grader_prompt(task, repo, &candidate)).await;
    let mut usage = UsageSummary::default();
    let mut verdict = None;
    for round in 0..2 {
        let Some(current) = &result else { break };
        if current.status != AgentRunStatus::Completed {
            break;
        }
        match answer::parse_grader_verdict(&current.summary, &task.key) {
            Ok(parsed) => {
                verdict = Some(parsed);
                break;
            }
            Err(problem) if round == 0 => {
                let previous = current.clone();
                let (next, next_error) = session
                    .ask_again(
                        &previous,
                        &format!(
                            "Your reply could not be read: {problem}. Call `complete_task` again with only the JSON object, judging every item of the key."
                        ),
                    )
                    .await;
                result = next;
                error = next_error;
            }
            Err(problem) => error = Some(problem),
        }
    }
    let (wall, _) = session.finish();
    record.wall_ms = wall.as_millis() as u64;
    if let Some(result) = &result {
        usage.add(&UsageSummary::of_task(
            &result.task,
            &context.settings.model,
        ));
    }
    record.usage = usage;
    record.transcript = save_transcript(
        &context.run_dir,
        &format!("grade-{grade_id}"),
        result.as_ref().map(|result| &result.task),
    );
    let _ = std::fs::remove_dir_all(&scratch);
    match verdict {
        Some(verdict) => {
            record.ok = true;
            record.item_score = Some(verdict.item_score(&task.key));
            record.pass = Some(verdict.pass());
            record.verdict = Some(verdict);
        }
        None => {
            record.failure_kind = failure_kind_of(result.as_ref());
            record.detail = Some(
                control
                    .limit_reason()
                    .or(error)
                    .or_else(|| result.as_ref().and_then(|result| result.error.clone()))
                    .unwrap_or_else(|| "no verdict".to_string()),
            );
        }
    }
    // A grading cancelled by a stop, or failed by a lost login, is not a
    // failed grading: it runs again on resume.
    let not_the_graders = !record.ok
        && (context.stop.level() >= 2 || is_login_failure(record.failure_kind.as_deref()));
    if !not_the_graders && let Err(error) = context.grades.append(&record) {
        eprintln!("could not record grade {grade_id}: {error}");
    }
    record
}

/// The parsed answer of an attempt.
pub fn parsed_answer(attempt: &AttemptRecord) -> ParsedAnswer {
    answer::parse_answer(attempt.answer.as_deref().unwrap_or(""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{AnswerKey, ItemKind, KeyItem, Location, Origin};

    fn task(id: &str) -> BenchTask {
        BenchTask {
            id: id.into(),
            repo: "demo".into(),
            kind: TaskKind::Definition,
            question: "Where?".into(),
            scope: None,
            key: AnswerKey {
                answer: "There.".into(),
                items: vec![KeyItem {
                    id: "L1".into(),
                    kind: ItemKind::Location,
                    text: "it".into(),
                    location: Some(Location::new("a.rs", 1, 2)),
                }],
                distractors: vec![],
            },
            origin: Origin::default(),
            difficulty: None,
            verification: None,
        }
    }

    fn attempt(task_id: &str, arm: Arm, outcome: Outcome, id: &str) -> AttemptRecord {
        AttemptRecord {
            attempt_id: id.into(),
            task_id: task_id.into(),
            repo: "demo".into(),
            kind: TaskKind::Definition,
            arm,
            sample: 0,
            outcome,
            detail: None,
            failure_kind: None,
            answer: Some("ANSWER: x".into()),
            started_at: Utc::now(),
            wall_ms: 1,
            active_ms: 1,
            usage: UsageSummary::default(),
            transcript: None,
            looked_outside: false,
        }
    }

    fn grade_record(attempt_id: &str, ok: bool) -> GradeRecord {
        GradeRecord {
            grade_id: format!("g-{attempt_id}-{ok}"),
            attempt_id: attempt_id.into(),
            task_id: "t".into(),
            ok,
            verdict: None,
            item_score: ok.then_some(1.0),
            pass: ok.then_some(true),
            detail: None,
            scrubbed: false,
            started_at: Utc::now(),
            wall_ms: 1,
            usage: UsageSummary::default(),
            transcript: None,
            failure_kind: None,
        }
    }

    #[test]
    fn plan_resumes_where_the_records_stop() {
        let tasks = vec![task("t1"), task("t2"), task("t3")];
        let arms = [Arm::A, Arm::B];
        let fresh = plan(&tasks, &arms, 1, &[], &[], 3, 7);
        assert_eq!(fresh.solve.len(), 6);
        assert_eq!(fresh.done, 0);
        // Each task's arms are adjacent in the order.
        for pair in fresh.solve.chunks(2) {
            assert_eq!(pair[0].task_id, pair[1].task_id);
            assert_ne!(pair[0].arm, pair[1].arm);
        }
        // The same seed gives the same order on resume.
        assert_eq!(plan(&tasks, &arms, 1, &[], &[], 3, 7).solve, fresh.solve);

        let attempts = vec![
            // t1/A answered and graded; t1/B answered, its grading failed
            // once; t2/A failed (final); t2/B stopped twice (infra); t3/A
            // stopped three times (given up).
            attempt("t1", Arm::A, Outcome::Answered, "a1"),
            attempt("t1", Arm::B, Outcome::Answered, "b1"),
            attempt("t2", Arm::A, Outcome::Failed, "a2"),
            attempt("t2", Arm::B, Outcome::Infra, "b2"),
            attempt("t2", Arm::B, Outcome::Infra, "b2bis"),
            attempt("t3", Arm::A, Outcome::Infra, "a3"),
            attempt("t3", Arm::A, Outcome::Infra, "a3bis"),
            attempt("t3", Arm::A, Outcome::Infra, "a3ter"),
        ];
        let grades = vec![grade_record("a1", true), grade_record("b1", false)];
        let resumed = plan(&tasks, &arms, 1, &attempts, &grades, 3, 7);
        assert_eq!(resumed.done, 3);
        assert_eq!(resumed.grade, vec!["b1".to_string()]);
        let mut solve: Vec<String> = resumed.solve.iter().map(ToString::to_string).collect();
        solve.sort();
        assert_eq!(solve, vec!["t2/B/0", "t3/B/0"]);
        assert_eq!(resumed.abandoned.len(), 1);
        assert_eq!(resumed.abandoned[0].to_string(), "t3/A/0");

        // A grading that failed three times is left alone.
        let grades = vec![
            grade_record("a1", true),
            grade_record("b1", false),
            GradeRecord {
                grade_id: "x".into(),
                ..grade_record("b1", false)
            },
            GradeRecord {
                grade_id: "y".into(),
                ..grade_record("b1", false)
            },
        ];
        assert!(
            plan(&tasks, &arms, 1, &attempts, &grades, 3, 7)
                .grade
                .is_empty()
        );
    }

    #[test]
    fn plan_adds_samples_and_arms_on_resume() {
        let tasks = vec![task("t1")];
        let attempts = vec![attempt("t1", Arm::A, Outcome::Answered, "a1")];
        let more = plan(&tasks, &[Arm::A, Arm::B, Arm::C], 2, &attempts, &[], 3, 1);
        let mut keys: Vec<String> = more.solve.iter().map(ToString::to_string).collect();
        keys.sort();
        assert_eq!(keys, vec!["t1/A/1", "t1/B/0", "t1/B/1", "t1/C/0", "t1/C/1"]);
        assert_eq!(more.grade, vec!["a1".to_string()]);
    }

    #[test]
    fn records_survive_a_restart_through_the_ledger() {
        let dir = crate::test_dir("runner-ledger");
        let path = dir.join("attempts.jsonl");
        let appender = Appender::open(&path).unwrap();
        let first = attempt("t1", Arm::A, Outcome::Answered, "a1");
        appender.append(&first).unwrap();
        drop(appender);
        let read: Vec<AttemptRecord> = read_jsonl(&path).unwrap();
        assert_eq!(read, vec![first]);
        let tasks = vec![task("t1")];
        let resumed = plan(&tasks, &[Arm::A], 1, &read, &[], 3, 1);
        assert!(resumed.solve.is_empty());
        assert_eq!(resumed.grade, vec!["a1".to_string()]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifests_flag_changes_that_break_comparability() {
        let settings = RunSettings {
            name: "x".into(),
            arms: vec![Arm::A],
            samples: 1,
            concurrency: 2,
            model: ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap(),
            shell: ShellSettings {
                timeout: Duration::from_secs(120),
                max_output_bytes: 16384,
                max_calls: 80,
            },
            max_input_tokens: 1000,
            attempt_timeout: None,
            max_attempts: 3,
            grade: true,
            grade_only: false,
            grader_shell: true,
            seed: 1,
            limit: None,
            mock: false,
            tasks_file: "tasks.json".into(),
            index_cli: "cli".into(),
            index_cli_sha256: "abc".into(),
        };
        let manifest = RunManifest::of(&settings);
        let mut other = manifest.clone();
        other.arms.push(Arm::B);
        other.samples = 3;
        assert!(manifest.incompatible_with(&other).is_empty());
        other.model.effort = "high".into();
        other.shell_max_output_bytes = 1;
        assert_eq!(
            manifest.incompatible_with(&other),
            vec!["model".to_string(), "shell settings".to_string()]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn each_arm_gets_only_its_own_tools() {
        let dir = crate::test_dir("arms");
        let layout = Layout::new(&dir.join("out")).unwrap();
        let cli = dir.join("fake_cli");
        std::fs::write(
            &cli,
            "#!/bin/sh\necho \"index cli: $1 $ARKO_INDEX_STATE\"\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        crate::repos::setup_arm_bins(&layout, Some(&cli)).unwrap();
        let repo = PreparedRepo {
            spec: crate::repos::RepoSpec {
                id: "demo".into(),
                url: String::new(),
                rev: "v1".into(),
                commit: None,
                language: "rust".into(),
                description: String::new(),
            },
            checkout: dir.clone(),
            index: dir.join("index-state"),
            commit: "abc".into(),
            files: 0,
            bytes: 0,
            files_by_language: Default::default(),
            index_build_ms: 0,
            index_summary: String::new(),
            index_cli_sha256: String::new(),
            dense: None,
        };
        async fn run(
            layout: &Layout,
            repo: &PreparedRepo,
            dir: &Path,
            arm: Arm,
            command: &str,
        ) -> String {
            let shell = ShellTool::new(
                ShellSettings {
                    timeout: Duration::from_secs(5),
                    max_output_bytes: 4096,
                    max_calls: 10,
                },
                arm_environment(layout, arm, repo, dir, None),
                None,
            );
            let context = arko_agents::TaskContext::new(Uuid::new_v4(), "solver", "", dir);
            let params = HashMap::from([(
                "command".to_string(),
                serde_json::Value::String(command.to_string()),
            )]);
            shell.execute(&context, params).await.unwrap()
        }
        let a = run(
            &layout,
            &repo,
            &dir,
            Arm::A,
            "arko_index search2 x y; echo \"state=$ARKO_INDEX_STATE\"",
        )
        .await;
        assert!(a.contains("arko_index: command not found"), "{a}");
        assert!(a.contains("state=\n"), "{a}");
        let a_cli = run(&layout, &repo, &dir, Arm::A, "arko_index_cli outline .").await;
        assert!(a_cli.starts_with("exit code: 127"), "{a_cli}");
        let b = run(&layout, &repo, &dir, Arm::B, "arko_index search2").await;
        assert!(b.contains("index cli: search2"), "{b}");
        assert!(b.contains("index-state"), "{b}");
        let b_dense = run(&layout, &repo, &dir, Arm::B, "arko_index_dense query").await;
        assert!(b_dense.starts_with("exit code: 127"), "{b_dense}");
        let c_dense = run(&layout, &repo, &dir, Arm::C, "command -v arko_index_dense").await;
        assert!(c_dense.starts_with("exit code: 0"), "{c_dense}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn task_with_commands(commands: &[&str]) -> arko_agents::Task {
        let model = ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap();
        let config = agent_config("t", "T", "role", "instructions", "summary", &model, 60).unwrap();
        let mut task = arko_agents::Task::new(config, "q".into(), PathBuf::from("/x"), None);
        let mut message = arko_ai::Message::new_assistant("");
        message.tool_calls = commands
            .iter()
            .enumerate()
            .map(|(index, command)| {
                serde_json::from_value(serde_json::json!({
                    "id": format!("call_{index}"),
                    "name": "shell",
                    "arguments": {"command": command},
                }))
                .unwrap()
            })
            .collect();
        task.conversation.messages.push(message);
        task
    }

    #[test]
    fn counts_index_use_and_flags_reads_of_the_benchmarks_files() {
        let out = Path::new("/data/bench");
        let model = ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap();
        let ordinary = task_with_commands(&[
            "rg -n foo src",
            "sed -n 1,9p /data/bench/repos/redis/src/a.c",
            "arko_index search2 \"$ARKO_INDEX_STATE\" 'retry delay'",
            "cat docs/arko_index-reference.md",
        ]);
        let usage = UsageSummary::of_task(&ordinary, &model);
        assert_eq!(
            (usage.tool_calls, usage.shell_calls, usage.index_calls),
            (4, 4, 1)
        );
        assert!(!looked_outside(&ordinary, out, "redis"));
        assert!(!looked_outside(
            &task_with_commands(&["ls /data/bench/indexes/redis"]),
            out,
            "redis"
        ));
        assert!(looked_outside(
            &task_with_commands(&["cat ../../tasks.json"]),
            out,
            "redis"
        ));
        assert!(looked_outside(
            &task_with_commands(&["ls /data/bench/runs/main"]),
            out,
            "redis"
        ));
        assert!(looked_outside(
            &task_with_commands(&["rg -n x /data/bench/repos/caddy"]),
            out,
            "redis"
        ));
    }

    #[tokio::test]
    async fn a_lost_login_while_grading_stops_the_run_and_uses_no_attempt() {
        let dir = crate::test_dir("grade-login");
        let layout = Layout::new(&dir.join("out")).unwrap();
        let repo = crate::mock::demo_repo(&layout.out);
        let mut demo = task("t1");
        demo.repo = "demo".into();
        let run_dir = layout.run_dir("main");
        let attempts = Appender::open(&run_dir.join("attempts.jsonl")).unwrap();
        for id in ["a1", "a2"] {
            let mut answered = attempt("t1", Arm::A, Outcome::Answered, id);
            answered.repo = "demo".into();
            if id == "a2" {
                answered.arm = Arm::B;
            }
            attempts.append(&answered).unwrap();
        }
        drop(attempts);
        let url = crate::mock::unauthorized_server().await;
        let client = crate::agents::build_client(&[], Some(&url)).unwrap();
        let settings = RunSettings {
            name: "main".into(),
            arms: vec![Arm::A, Arm::B],
            samples: 1,
            concurrency: 1,
            model: ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap(),
            shell: ShellSettings {
                timeout: Duration::from_secs(5),
                max_output_bytes: 4096,
                max_calls: 5,
            },
            max_input_tokens: 1_000_000,
            attempt_timeout: None,
            max_attempts: 3,
            grade: true,
            grade_only: true,
            grader_shell: true,
            seed: 1,
            limit: None,
            mock: true,
            tasks_file: "tasks.json".into(),
            index_cli: "cli".into(),
            index_cli_sha256: String::new(),
        };
        let summary = run(RunInputs {
            layout: layout.clone(),
            settings,
            tasks: vec![demo],
            prepared: BTreeMap::from([("demo".to_string(), repo)]),
            client,
        })
        .await
        .unwrap();
        assert!(summary.contains("stopped (login failure)"), "{summary}");
        // The failed grading isn't recorded: both answers are graded on
        // resume, with all their attempts left.
        let grades: Vec<GradeRecord> = read_jsonl(&run_dir.join("grades.jsonl")).unwrap();
        assert!(grades.is_empty(), "{grades:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn arm_prompts_differ_only_in_their_tool_section() {
        let a = solver_instructions(Arm::A, true);
        let b = solver_instructions(Arm::B, true);
        let c = solver_instructions(Arm::C, true);
        assert!(!a.contains("arko_index"));
        assert!(b.contains("arko_index search2") && !b.contains("arko_index_dense"));
        assert!(c.contains("arko_index_dense"));
        let tail = |text: &str| text[text.find("Rules:").unwrap()..].to_string();
        assert_eq!(tail(&a), tail(&b));
        assert_eq!(tail(&b), tail(&c));
    }
}
