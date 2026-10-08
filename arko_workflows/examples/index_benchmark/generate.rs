//! Task generation: a writer agent writes questions with answer keys it has
//! checked in the code (`submit_question` checks every location against the
//! checkout), a separate verifier agent answers each question blind and
//! then judges the key, and the task set keeps the questions both agree on,
//! beside the seed tasks converted from the earlier question sets.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arko_agents::{
    AgentConfig, TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition,
    ToolParameterType, ToolTypeDetails,
};
use arko_ai::LLMClient;
use arko_workflows::prelude::AgentRunStatus;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::agents::{ModelSettings, agent_config};
use crate::answer::{self, DeterministicCheck, VerifierVerdict};
use crate::ledger::{Appender, read_jsonl};
use crate::prompts;
use crate::repos::{Layout, PreparedRepo};
use crate::runner::{
    ActiveRuns, AgentJob, AgentSession, JobControl, JobLimits, PlanUsageLog, UsageSummary,
    arm_environment, failure_kind_of, is_login_failure,
};
use crate::shell::{ShellSettings, ShellTool};
use crate::tasks::{
    AnswerKey, BenchTask, ItemKind, KeyItem, Location, Origin, TaskKind, check_location,
    normalize_path,
};
use crate::{Arm, BenchResult};

/// The seed tasks: the earlier arko question sets (behaviour, depth and
/// breadth), converted to tasks with their keys checked at the pinned
/// commit.
pub const SEED_TASKS: &str = include_str!("seed-tasks.json");

/// Questions a writer run is asked for at most.
const QUESTIONS_PER_WRITER_RUN: usize = 4;

/// A question a writer submitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub repo: String,
    pub kind: TaskKind,
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub answer: String,
    /// Each location with what it is.
    pub locations: Vec<(Location, String)>,
    #[serde(default)]
    pub facts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<String>,
    #[serde(default)]
    pub evidence: String,
    pub writer_run: String,
    pub created_at: DateTime<Utc>,
}

impl Candidate {
    pub fn key(&self) -> AnswerKey {
        let mut items = Vec::new();
        for (index, (location, text)) in self.locations.iter().enumerate() {
            items.push(KeyItem {
                id: format!("L{}", index + 1),
                kind: ItemKind::Location,
                text: text.clone(),
                location: Some(location.clone()),
            });
        }
        for (index, fact) in self.facts.iter().enumerate() {
            items.push(KeyItem {
                id: format!("F{}", index + 1),
                kind: ItemKind::Fact,
                text: fact.clone(),
                location: None,
            });
        }
        AnswerKey {
            answer: self.answer.clone(),
            items,
            distractors: Vec::new(),
        }
    }

    pub fn to_task(&self, verification: Option<&VerificationRecord>) -> BenchTask {
        BenchTask {
            id: self.id.clone(),
            repo: self.repo.clone(),
            kind: self.kind,
            question: self.question.clone(),
            scope: self.scope.clone(),
            key: self.key(),
            origin: Origin {
                source: "generated".into(),
                reference: self.writer_run.clone(),
            },
            difficulty: self.difficulty.clone(),
            verification: verification.map(|record| {
                serde_json::json!({
                    "verdict": record.verdict.as_ref().map(|verdict| verdict.verdict.clone()),
                    "notes": record.verdict.as_ref().map(|verdict| verdict.notes.clone()),
                    "blind_check_pass": record.blind_check.as_ref().map(|check| check.pass),
                })
            }),
        }
    }
}

/// A writer run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriterRecord {
    pub run_id: String,
    pub repo: String,
    pub kind: TaskKind,
    pub requested: usize,
    pub accepted: usize,
    pub outcome: crate::runner::Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<String>,
    pub wall_ms: u64,
    pub usage: UsageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

/// A verifier run on a candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRecord {
    pub candidate_id: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<VerifierVerdict>,
    /// The verifier's own answer, before it saw the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blind_answer: Option<String>,
    /// That answer checked against the key's locations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blind_check: Option<DeterministicCheck>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<String>,
    pub wall_ms: u64,
    pub usage: UsageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

/// Normalized text for duplicate detection.
fn normalized(question: &str) -> String {
    question
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// What every writer run shares: the candidates file, the questions so
/// far, and the id counters.
pub struct GenerationState {
    candidates: Appender,
    known: parking_lot::Mutex<HashSet<String>>,
    by_repo: parking_lot::Mutex<HashMap<String, Vec<String>>>,
    next_id: parking_lot::Mutex<HashMap<(String, TaskKind), usize>>,
    new_candidates: parking_lot::Mutex<Vec<Candidate>>,
}

impl GenerationState {
    fn new(path: &Path, existing: &[Candidate], seeds: &[BenchTask]) -> BenchResult<Self> {
        let mut known = HashSet::new();
        let mut by_repo: HashMap<String, Vec<String>> = HashMap::new();
        let mut next_id: HashMap<(String, TaskKind), usize> = HashMap::new();
        for candidate in existing {
            known.insert(normalized(&candidate.question));
            by_repo
                .entry(candidate.repo.clone())
                .or_default()
                .push(candidate.question.clone());
            let number = candidate
                .id
                .rsplit('g')
                .next()
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(0);
            let slot = next_id
                .entry((candidate.repo.clone(), candidate.kind))
                .or_insert(1);
            *slot = (*slot).max(number + 1);
        }
        for seed in seeds {
            known.insert(normalized(&seed.question));
            by_repo
                .entry(seed.repo.clone())
                .or_default()
                .push(seed.question.clone());
        }
        Ok(Self {
            candidates: Appender::open(path)?,
            known: parking_lot::Mutex::new(known),
            by_repo: parking_lot::Mutex::new(by_repo),
            next_id: parking_lot::Mutex::new(next_id),
            new_candidates: parking_lot::Mutex::new(Vec::new()),
        })
    }

    fn questions_of(&self, repo: &str) -> Vec<String> {
        self.by_repo.lock().get(repo).cloned().unwrap_or_default()
    }

    fn take_new(&self) -> Vec<Candidate> {
        std::mem::take(&mut *self.new_candidates.lock())
    }
}

/// The writer's `submit_question` tool, for one writer run.
pub struct SubmitQuestionTool {
    state: Arc<GenerationState>,
    repo: String,
    kind: TaskKind,
    checkout: PathBuf,
    run_id: String,
    requested: usize,
    accepted: parking_lot::Mutex<usize>,
}

impl std::fmt::Debug for SubmitQuestionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmitQuestionTool")
            .field("repo", &self.repo)
            .field("kind", &self.kind)
            .finish()
    }
}

fn param(name: &str, description: &str, required: bool) -> ToolParameterDefinition {
    ToolParameterDefinition {
        name: name.to_string(),
        description: description.to_string(),
        param_type: ToolParameterType::String,
        required,
        is_path: false,
        flag: None,
        default_value: None,
    }
}

impl SubmitQuestionTool {
    pub fn accepted(&self) -> usize {
        *self.accepted.lock()
    }

    /// Checks a submission and turns it into a candidate (without an id).
    fn check(&self, params: &HashMap<String, Value>) -> Result<Candidate, String> {
        let text = |name: &str| {
            params
                .get(name)
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("")
                .to_string()
        };
        let kind = TaskKind::parse(&text("kind"))
            .ok_or_else(|| "`kind` must be definition, callers, behaviour or impact".to_string())?;
        if kind != self.kind {
            return Err(format!("this run writes `{}` questions", self.kind));
        }
        let question = text("question");
        if question.len() < 15 {
            return Err("the question is missing or too short".into());
        }
        if self.state.known.lock().contains(&normalized(&question)) {
            return Err("this question exists already; write a different one".into());
        }
        let answer = text("answer");
        if answer.is_empty() {
            return Err("`answer` (the reference answer in prose) is required".into());
        }
        let mut locations = Vec::new();
        let mut cache = HashMap::new();
        let mut problems = Vec::new();
        for line in text("locations")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            let line = line.trim_start_matches(['-', '*']).trim();
            let (spec, what) = line.split_once('|').unwrap_or((line, ""));
            let Some(location) = Location::parse(spec.trim()) else {
                problems.push(format!("`{line}` is not `path:start-end | what it is`"));
                continue;
            };
            let location = Location {
                path: normalize_path(&location.path),
                ..location
            };
            if let Err(problem) = check_location(&self.checkout, &location, &mut cache) {
                problems.push(problem);
                continue;
            }
            locations.push((location, what.trim().to_string()));
        }
        if !problems.is_empty() {
            return Err(format!("fix these locations: {}", problems.join("; ")));
        }
        let facts: Vec<String> = text("facts")
            .lines()
            .map(|line| {
                line.trim()
                    .trim_start_matches(['-', '*'])
                    .trim()
                    .to_string()
            })
            .filter(|line| !line.is_empty())
            .collect();
        let (min_locations, max_locations, min_facts, max_facts) = match kind {
            TaskKind::Definition => (1, 2, 0, 3),
            TaskKind::Callers => (1, 12, 0, 3),
            TaskKind::Behaviour => (1, 4, 3, 6),
            TaskKind::Impact => (1, 12, 1, 4),
        };
        if locations.len() < min_locations || locations.len() > max_locations {
            return Err(format!(
                "a `{kind}` key has {min_locations} to {max_locations} locations; this one has {}",
                locations.len()
            ));
        }
        if facts.len() < min_facts || facts.len() > max_facts {
            return Err(format!(
                "a `{kind}` key has {min_facts} to {max_facts} facts; this one has {}",
                facts.len()
            ));
        }
        let lowered = question.to_lowercase();
        for (location, _) in &locations {
            let path = location.path.to_lowercase();
            let file = path.rsplit('/').next().unwrap_or(&path);
            if lowered.contains(&path) || (file.contains('.') && lowered.contains(file)) {
                return Err(format!(
                    "the question gives away the file {}: ask without naming it",
                    location.path
                ));
            }
        }
        let scope = Some(text("scope")).filter(|scope| !scope.is_empty());
        let difficulty = Some(text("difficulty")).filter(|text| !text.is_empty());
        Ok(Candidate {
            id: String::new(),
            repo: self.repo.clone(),
            kind,
            question,
            scope,
            answer,
            locations,
            facts,
            difficulty,
            evidence: text("evidence"),
            writer_run: self.run_id.clone(),
            created_at: Utc::now(),
        })
    }
}

#[async_trait]
impl Tool for SubmitQuestionTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "submit_question".to_string(),
            description: "Submit one benchmark question with its answer key. The key's locations are checked against the checkout; a rejected submission says what to fix.".to_string(),
            output_based_example: None,
            parameters: vec![
                param("kind", "definition, callers, behaviour or impact (this run's kind).", true),
                param("question", "The question, as a developer would ask it. Don't name the file.", true),
                param("scope", "Optional: what counts, e.g. \"non-test code\".", false),
                param("answer", "The reference answer in prose.", true),
                param("locations", "One per line: `path:start-end | what it is` (or `path:line | ...`), relative to the repository root.", true),
                param("facts", "One per line: the facts a correct answer must state (behaviour: 3 to 6; impact: 1 to 4).", false),
                param("difficulty", "Optional: easy, medium or hard.", false),
                param("evidence", "The commands and lines you checked the key with.", false),
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let mut candidate = match self.check(&params) {
            Ok(candidate) => candidate,
            Err(problem) => return Ok(format!("Rejected: {problem}")),
        };
        {
            let mut known = self.state.known.lock();
            if !known.insert(normalized(&candidate.question)) {
                return Ok("Rejected: this question exists already; write a different one".into());
            }
        }
        let number = {
            let mut next = self.state.next_id.lock();
            let slot = next.entry((self.repo.clone(), self.kind)).or_insert(1);
            let number = *slot;
            *slot += 1;
            number
        };
        candidate.id = format!("{}-{}-g{number:03}", self.repo, self.kind);
        if let Err(error) = self.state.candidates.append(&candidate) {
            return Err(ToolError::ExecutionFailed(format!(
                "the question could not be saved: {error}"
            )));
        }
        self.state
            .by_repo
            .lock()
            .entry(self.repo.clone())
            .or_default()
            .push(candidate.question.clone());
        let id = candidate.id.clone();
        self.state.new_candidates.lock().push(candidate);
        let accepted = {
            let mut accepted = self.accepted.lock();
            *accepted += 1;
            *accepted
        };
        Ok(format!(
            "Accepted as {id} ({accepted} of the {} asked for).{}",
            self.requested,
            if accepted >= self.requested {
                " That's all: call `complete_task`."
            } else {
                ""
            }
        ))
    }
}

pub struct GenerateOptions {
    pub layout: Layout,
    pub prepared: BTreeMap<String, PreparedRepo>,
    pub repo_ids: Vec<String>,
    pub kinds: Vec<TaskKind>,
    pub per_kind: usize,
    pub oversample: f64,
    pub concurrency: usize,
    pub model: ModelSettings,
    pub client: Arc<LLMClient>,
    pub shell: ShellSettings,
    pub tasks_path: PathBuf,
    pub mock: bool,
    /// The tools of the writer and the verifier: arm A's (a terminal) or
    /// arm B's (with arko_index).
    pub writer_arm: Arm,
}

struct Paths {
    candidates: PathBuf,
    verifications: PathBuf,
    writers: PathBuf,
    transcripts: PathBuf,
}

fn paths(layout: &Layout) -> Paths {
    let dir = layout.tasks_dir();
    Paths {
        candidates: dir.join("candidates.jsonl"),
        verifications: dir.join("verifications.jsonl"),
        writers: dir.join("writers.jsonl"),
        transcripts: dir.join("transcripts"),
    }
}

/// Directories of a checkout that hold source files of its language, most
/// files first: the focus areas handed to successive writer runs.
fn focus_areas(checkout: &Path, language: &str) -> Vec<String> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut stack = vec![checkout.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                if entry.file_name() != ".git" {
                    stack.push(path);
                }
                continue;
            }
            let matches = crate::repos::language_of(&path)
                .is_some_and(|found| found == language || (language == "cpp" && found == "c"));
            if !matches {
                continue;
            }
            let Ok(relative) = path.strip_prefix(checkout) else {
                continue;
            };
            let parts: Vec<String> = relative
                .parent()
                .map(|parent| {
                    parent
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect()
                })
                .unwrap_or_default();
            let area = if parts.is_empty() {
                ".".to_string()
            } else {
                parts[..parts.len().min(2)].join("/")
            };
            *counts.entry(area).or_insert(0) += 1;
        }
    }
    let mut areas: Vec<(String, usize)> = counts.into_iter().collect();
    areas.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    areas.into_iter().map(|(area, _)| area).collect()
}

/// A writer run's work.
#[derive(Debug, Clone)]
struct WriterJob {
    repo: String,
    kind: TaskKind,
    count: usize,
    batch: usize,
}

/// Seed tasks per (repo, kind).
fn seed_counts(seeds: &[BenchTask]) -> HashMap<(String, TaskKind), usize> {
    let mut counts = HashMap::new();
    for seed in seeds {
        *counts.entry((seed.repo.clone(), seed.kind)).or_insert(0) += 1;
    }
    counts
}

/// Writer runs per repository and kind at most, over every `generate`:
/// enough to write twice the questions wanted.
pub fn max_writer_runs(per_kind: usize, oversample: f64) -> usize {
    ((per_kind as f64 * oversample * 2.0).ceil() as usize).div_ceil(QUESTIONS_PER_WRITER_RUN) + 2
}

/// Writer runs per repository and kind, in one `generate`, that may end for
/// reasons outside the writer before the pair is left alone.
const MAX_INFRA_WRITER_RUNS: usize = 3;

/// Where generation stands for one repository and kind.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairProgress {
    /// Seed tasks of the pair.
    pub seeded: usize,
    /// Questions the verifier agreed with.
    pub agreed: usize,
    /// Questions waiting for (another) verification.
    pub pending: usize,
    /// Writer runs that ended for reasons of their own: they count against
    /// [`max_writer_runs`].
    pub writer_runs: usize,
    /// Writer runs of this `generate` that ended outside the writer's
    /// control (not counted against the limit).
    pub infra_runs: usize,
}

impl PairProgress {
    /// Questions to write now: enough that the agreed questions, and the
    /// pending ones at the expected yield (one in `oversample`), reach
    /// `per_kind` with the seeds. None once the pair has had its writer runs.
    pub fn to_write(&self, per_kind: usize, oversample: f64, max_runs: usize) -> usize {
        if self.writer_runs >= max_runs || self.infra_runs >= MAX_INFRA_WRITER_RUNS {
            return 0;
        }
        let needed = per_kind.saturating_sub(self.seeded + self.agreed);
        let target = (needed as f64 * oversample).ceil() as usize;
        target.saturating_sub(self.pending)
    }
}

/// Verifications of a candidate that may fail (no verdict) before it is
/// dropped.
const MAX_VERIFICATION_FAILURES: usize = 3;

pub async fn generate(options: GenerateOptions) -> BenchResult<Vec<BenchTask>> {
    let paths = paths(&options.layout);
    std::fs::create_dir_all(&paths.transcripts)?;
    let seeds = crate::tasks::parse_tasks(SEED_TASKS)?;
    let existing: Vec<Candidate> = read_jsonl(&paths.candidates)?;
    let verifications: Vec<VerificationRecord> = read_jsonl(&paths.verifications)?;
    let writers: Vec<WriterRecord> = read_jsonl(&paths.writers)?;
    let state = Arc::new(GenerationState::new(&paths.candidates, &existing, &seeds)?);
    let seed_counts = seed_counts(&seeds);
    let max_runs = max_writer_runs(options.per_kind, options.oversample);

    let pairs: Vec<(String, TaskKind)> = options
        .repo_ids
        .iter()
        .flat_map(|repo| options.kinds.iter().map(move |kind| (repo.clone(), *kind)))
        .collect();
    let mut progress: HashMap<(String, TaskKind), PairProgress> = pairs
        .iter()
        .map(|pair| {
            let seeded = seed_counts.get(pair).copied().unwrap_or(0);
            (
                pair.clone(),
                PairProgress {
                    seeded,
                    ..PairProgress::default()
                },
            )
        })
        .collect();
    let mut runs_so_far: HashMap<(String, TaskKind), usize> = HashMap::new();
    for record in &writers {
        let pair = (record.repo.clone(), record.kind);
        *runs_so_far.entry(pair.clone()).or_insert(0) += 1;
        if record.outcome.is_final()
            && let Some(progress) = progress.get_mut(&pair)
        {
            progress.writer_runs += 1;
        }
    }
    let mut failures: HashMap<String, usize> = HashMap::new();
    let mut verdicts: HashMap<&str, &str> = HashMap::new();
    for record in &verifications {
        match (&record.verdict, record.ok) {
            (Some(verdict), true) => {
                verdicts.insert(record.candidate_id.as_str(), verdict.verdict.as_str());
            }
            _ => *failures.entry(record.candidate_id.clone()).or_insert(0) += 1,
        }
    }
    let mut verify_queue: VecDeque<Candidate> = VecDeque::new();
    for candidate in &existing {
        let Some(progress) = progress.get_mut(&(candidate.repo.clone(), candidate.kind)) else {
            continue;
        };
        match verdicts.get(candidate.id.as_str()) {
            Some(&"agree") => progress.agreed += 1,
            Some(_) => {}
            None if failures.get(&candidate.id).copied().unwrap_or(0)
                < MAX_VERIFICATION_FAILURES =>
            {
                progress.pending += 1;
                verify_queue.push_back(candidate.clone());
            }
            None => {}
        }
    }
    let queued_writes: usize = pairs
        .iter()
        .map(|pair| progress[pair].to_write(options.per_kind, options.oversample, max_runs))
        .sum();
    eprintln!(
        "generate: {queued_writes} questions to write over {} repository/kind pairs, {} to verify; writer and verifier tools: {}{}",
        pairs.len(),
        verify_queue.len(),
        options.writer_arm.label(),
        if options.mock { " (mock model)" } else { "" }
    );

    let index_tools = if options.writer_arm == Arm::A {
        String::new()
    } else {
        ", and `arko_index search2 \"$ARKO_INDEX_STATE\" 'words'` or `arko_index outline PATH`"
            .to_string()
    };
    let context = Arc::new(GenerateContext {
        writer_config: agent_config(
            "writer",
            "Question writer",
            prompts::WRITER_ROLE,
            &prompts::WRITER_INSTRUCTIONS.replace("{index}", &index_tools),
            "A one-line summary: how many questions you submitted.",
            &options.model,
            options.shell.timeout.as_secs() + 60,
        )?,
        verifier_config: agent_config(
            "verifier",
            "Question verifier",
            prompts::VERIFIER_ROLE,
            &prompts::VERIFIER_INSTRUCTIONS.replace("{index}", &index_tools),
            "First your answer (ANSWER:/LOCATIONS:); after the key, only the JSON verdict.",
            &options.model,
            options.shell.timeout.as_secs() + 60,
        )?,
        state: state.clone(),
        writers: Appender::open(&paths.writers)?,
        verifications: Appender::open(&paths.verifications)?,
        transcripts: paths.transcripts.clone(),
        options,
        active: Arc::new(ActiveRuns::default()),
        plan_usage: Arc::new(PlanUsageLog::default()),
        stop: Arc::new(crate::runner::StopState::default()),
    });

    let stop = context.stop.clone();
    crate::runner::watch_signals(stop.clone(), context.active.clone());

    enum Done {
        Writer((String, TaskKind), Box<WriterRecord>),
        Verified(Box<Candidate>, Box<VerificationRecord>, bool),
    }
    let mut running: tokio::task::JoinSet<Done> = tokio::task::JoinSet::new();
    let mut busy: HashSet<(String, TaskKind)> = HashSet::new();
    let (mut written, mut checked, mut agreed) = (0usize, 0usize, 0usize);
    let per_kind = context.options.per_kind;
    let oversample = context.options.oversample;
    loop {
        // Questions submitted since: count them, and verify them.
        for candidate in context.state.take_new() {
            if let Some(progress) = progress.get_mut(&(candidate.repo.clone(), candidate.kind)) {
                progress.pending += 1;
            }
            verify_queue.push_back(candidate);
        }
        while running.len() < context.options.concurrency && stop.level() == 0 {
            if let Some(candidate) = verify_queue.pop_front() {
                let context = context.clone();
                running.spawn(async move {
                    let (record, recorded) = verify(&context, &candidate).await;
                    Done::Verified(Box::new(candidate), Box::new(record), recorded)
                });
                continue;
            }
            // One writer run at a time per pair, so runs of a pair don't
            // write about the same code.
            let Some(pair) = pairs.iter().find(|pair| {
                !busy.contains(*pair)
                    && progress[*pair].to_write(per_kind, oversample, max_runs) > 0
            }) else {
                break;
            };
            let count = progress[pair]
                .to_write(per_kind, oversample, max_runs)
                .min(QUESTIONS_PER_WRITER_RUN);
            let batch = runs_so_far.entry(pair.clone()).or_insert(0);
            let job = WriterJob {
                repo: pair.0.clone(),
                kind: pair.1,
                count,
                batch: *batch,
            };
            *batch += 1;
            busy.insert(pair.clone());
            let context = context.clone();
            let pair = pair.clone();
            running.spawn(async move { Done::Writer(pair, Box::new(write(&context, job).await)) });
        }
        let Some(done) = running.join_next().await else {
            break;
        };
        match done {
            Ok(Done::Writer(pair, record)) => {
                busy.remove(&pair);
                written += record.accepted;
                if let Some(progress) = progress.get_mut(&pair) {
                    if record.outcome.is_final() {
                        progress.writer_runs += 1;
                    } else {
                        progress.infra_runs += 1;
                    }
                }
                eprintln!(
                    "[writer] {} {}: {} of {} questions accepted ({:?}{}), {}k tokens in",
                    record.repo,
                    record.kind,
                    record.accepted,
                    record.requested,
                    record.outcome,
                    record
                        .detail
                        .as_ref()
                        .map_or(String::new(), |detail| format!(
                            ": {}",
                            detail.lines().next().unwrap_or("")
                        )),
                    record.usage.input_tokens / 1000
                );
                if is_login_failure(record.failure_kind.as_deref()) {
                    eprintln!(
                        "the ChatGPT login failed; stopping. Fix the login, then run `generate` again to resume."
                    );
                    stop.raise_to(1, "login failure");
                }
            }
            Ok(Done::Verified(candidate, record, recorded)) => {
                let pair = (candidate.repo.clone(), candidate.kind);
                let verdict = record
                    .verdict
                    .as_ref()
                    .map_or("no verdict", |verdict| verdict.verdict.as_str());
                if record.ok {
                    checked += 1;
                    if let Some(progress) = progress.get_mut(&pair) {
                        progress.pending = progress.pending.saturating_sub(1);
                        if verdict == "agree" {
                            progress.agreed += 1;
                        }
                    }
                    if verdict == "agree" {
                        agreed += 1;
                    }
                } else if recorded {
                    let failed = failures.entry(candidate.id.clone()).or_insert(0);
                    *failed += 1;
                    if *failed < MAX_VERIFICATION_FAILURES {
                        verify_queue.push_back(*candidate.clone());
                    } else if let Some(progress) = progress.get_mut(&pair) {
                        progress.pending = progress.pending.saturating_sub(1);
                    }
                }
                eprintln!(
                    "[verify] {}: {verdict}{}",
                    record.candidate_id,
                    record
                        .verdict
                        .as_ref()
                        .and_then(|verdict| verdict.problems.first())
                        .or(record.detail.as_ref())
                        .map_or(String::new(), |problem| format!(
                            " ({})",
                            problem.lines().next().unwrap_or("")
                        ))
                );
                if is_login_failure(record.failure_kind.as_deref()) {
                    eprintln!(
                        "the ChatGPT login failed; stopping. Fix the login, then run `generate` again to resume."
                    );
                    stop.raise_to(1, "login failure");
                }
            }
            Err(error) => eprintln!("a generation job panicked: {error}"),
        }
    }
    // Questions submitted by runs that ended after the loop stopped taking
    // new work are verified on the next `generate`.
    let _ = context.state.take_new();
    let short: Vec<String> = pairs
        .iter()
        .filter(|pair| progress[*pair].seeded + progress[*pair].agreed < per_kind)
        .map(|(repo, kind)| {
            format!(
                "{repo} {kind} ({})",
                progress[&(repo.clone(), *kind)].seeded + progress[&(repo.clone(), *kind)].agreed
            )
        })
        .collect();
    eprintln!(
        "generate: {written} questions written, {checked} verified ({agreed} agreed){}",
        stop.reason().map_or(String::new(), |reason| format!(
            "; stopped ({reason}): run again to resume"
        ))
    );
    if !short.is_empty() && stop.reason().is_none() {
        eprintln!(
            "short of {per_kind} tasks (writer runs used up; raise --oversample to write more): {}",
            short.join(", ")
        );
    }
    assemble(
        &context.options.layout,
        context.options.per_kind,
        &context.options.tasks_path,
    )
}

struct GenerateContext {
    options: GenerateOptions,
    writer_config: AgentConfig,
    verifier_config: AgentConfig,
    state: Arc<GenerationState>,
    writers: Appender,
    verifications: Appender,
    transcripts: PathBuf,
    active: Arc<ActiveRuns>,
    plan_usage: Arc<PlanUsageLog>,
    stop: Arc<crate::runner::StopState>,
}

impl GenerateContext {
    fn shell(
        &self,
        repo: &PreparedRepo,
        scratch: &Path,
        control: &Arc<JobControl>,
        max_calls: usize,
    ) -> ShellTool {
        let mut settings = self.options.shell.clone();
        settings.max_calls = max_calls;
        ShellTool::new(
            settings,
            arm_environment(
                &self.options.layout,
                self.options.writer_arm,
                repo,
                scratch,
                crate::runner::ArmServices::default(),
            ),
            Some(control.limit_callback()),
        )
    }

    fn save_transcript(&self, name: &str, task: Option<&arko_agents::Task>) -> Option<String> {
        let task = task?;
        let path = self.transcripts.join(format!("{name}.json"));
        crate::write_atomic(&path, serde_json::to_string(task).ok()?.as_bytes()).ok()?;
        Some(path.display().to_string())
    }
}

async fn write(context: &GenerateContext, job: WriterJob) -> WriterRecord {
    let run_id = format!(
        "{}-{}-w{}-{}",
        job.repo,
        job.kind,
        job.batch,
        &Uuid::new_v4().simple().to_string()[..8]
    );
    let mut record = WriterRecord {
        run_id: run_id.clone(),
        repo: job.repo.clone(),
        kind: job.kind,
        requested: job.count,
        accepted: 0,
        outcome: crate::runner::Outcome::Infra,
        detail: None,
        failure_kind: None,
        wall_ms: 0,
        usage: UsageSummary::default(),
        transcript: None,
    };
    let Some(repo) = context.options.prepared.get(&job.repo) else {
        record.detail = Some("not prepared".into());
        return record;
    };
    let scratch = context.options.layout.scratch_dir().join(&run_id);
    let _ = std::fs::create_dir_all(&scratch);
    let control = JobControl::new();
    let submit = Arc::new(SubmitQuestionTool {
        state: context.state.clone(),
        repo: job.repo.clone(),
        kind: job.kind,
        checkout: repo.checkout.clone(),
        run_id: run_id.clone(),
        requested: job.count,
        accepted: parking_lot::Mutex::new(0),
    });
    let shell = context.shell(repo, &scratch, &control, 150);
    let session = match AgentSession::start(AgentJob {
        client: context.options.client.clone(),
        workspace: repo.checkout.clone(),
        agent_name: "writer",
        config: context.writer_config.clone(),
        tools: vec![Arc::new(shell), submit.clone()],
        control: control.clone(),
        limits: JobLimits {
            max_input_tokens: 10_000_000,
            timeout: Some(Duration::from_secs(90 * 60)),
        },
        active: context.active.clone(),
        plan_usage: context.plan_usage.clone(),
    }) {
        Ok(session) => session,
        Err(error) => {
            record.detail = Some(error.to_string());
            return record;
        }
    };
    let areas = focus_areas(&repo.checkout, &repo.spec.language);
    let focus = if areas.is_empty() {
        String::new()
    } else {
        let start = (job.batch * 3) % areas.len();
        let picked: Vec<&str> = areas
            .iter()
            .cycle()
            .skip(start)
            .take(areas.len().min(3))
            .map(String::as_str)
            .collect();
        format!(
            "\nFocus this run on code under {} (other runs cover other parts), unless they can't give {} good questions.",
            picked
                .iter()
                .map(|area| format!("`{area}`"))
                .collect::<Vec<_>>()
                .join(", "),
            job.count
        )
    };
    let prompt = format!(
        "{}{focus}",
        prompts::writer_prompt(
            &repo.spec.id,
            &repo.spec.description,
            &repo.spec.language,
            job.kind.as_str(),
            job.count,
            &context.state.questions_of(&job.repo),
        )
    );
    let (result, error) = session.ask(&prompt).await;
    let (wall, _) = session.finish();
    let (outcome, detail, failure_kind) =
        crate::runner::classify(result.as_ref(), error.as_deref(), &control);
    record.outcome = outcome;
    record.detail = detail;
    record.failure_kind = failure_kind;
    record.wall_ms = wall.as_millis() as u64;
    record.accepted = submit.accepted();
    if let Some(result) = &result {
        record.usage = UsageSummary::of_task(&result.task, &context.options.model);
    }
    record.transcript =
        context.save_transcript(&run_id, result.as_ref().map(|result| &result.task));
    let _ = std::fs::remove_dir_all(&scratch);
    // A run cancelled by the operator, or failed by a lost login, isn't the
    // writer's: it isn't recorded. (Questions it submitted are kept.)
    let not_the_writers = record.outcome == crate::runner::Outcome::Infra
        && (context.stop.level() >= 2 || is_login_failure(record.failure_kind.as_deref()));
    if !not_the_writers && let Err(error) = context.writers.append(&record) {
        eprintln!("could not record writer run {run_id}: {error}");
    }
    record
}

/// The key as the verifier reads it.
fn key_text(candidate: &Candidate) -> String {
    format!(
        "{}{}",
        candidate
            .scope
            .as_ref()
            .map_or(String::new(), |scope| format!("Scope: {scope}\n")),
        candidate.key().render()
    )
}

/// Verifies `candidate`; also returns whether the verification was
/// recorded (one cancelled by the operator, or failed by a lost login, is
/// not: it runs again).
async fn verify(context: &GenerateContext, candidate: &Candidate) -> (VerificationRecord, bool) {
    let mut record = VerificationRecord {
        candidate_id: candidate.id.clone(),
        ok: false,
        verdict: None,
        blind_answer: None,
        blind_check: None,
        detail: None,
        failure_kind: None,
        wall_ms: 0,
        usage: UsageSummary::default(),
        transcript: None,
    };
    let Some(repo) = context.options.prepared.get(&candidate.repo) else {
        record.detail = Some("not prepared".into());
        let _ = context.verifications.append(&record);
        return (record, true);
    };
    let name = format!(
        "verify-{}-{}",
        candidate.id,
        &Uuid::new_v4().simple().to_string()[..8]
    );
    let scratch = context.options.layout.scratch_dir().join(&name);
    let _ = std::fs::create_dir_all(&scratch);
    let control = JobControl::new();
    let shell = context.shell(repo, &scratch, &control, 80);
    let session = match AgentSession::start(AgentJob {
        client: context.options.client.clone(),
        workspace: repo.checkout.clone(),
        agent_name: "verifier",
        config: context.verifier_config.clone(),
        tools: vec![Arc::new(shell)],
        control: control.clone(),
        limits: JobLimits {
            max_input_tokens: 6_000_000,
            timeout: Some(Duration::from_secs(60 * 60)),
        },
        active: context.active.clone(),
        plan_usage: context.plan_usage.clone(),
    }) {
        Ok(session) => session,
        Err(error) => {
            record.detail = Some(error.to_string());
            let _ = context.verifications.append(&record);
            return (record, true);
        }
    };
    let question = format!(
        "Repository: {} ({}), checked out in your working directory.\n\n{}",
        repo.spec.id,
        repo.spec.description,
        prompts::verifier_question(
            &repo.spec.id,
            candidate.kind.as_str(),
            &candidate.question,
            candidate.scope.as_deref()
        )
    );
    let (mut result, mut error) = session.ask(&question).await;
    if let Some(first) = result
        .clone()
        .filter(|result| result.status == AgentRunStatus::Completed)
    {
        record.blind_answer = Some(first.summary.clone());
        let task = candidate.to_task(None);
        record.blind_check = Some(answer::deterministic_check(
            &task,
            &answer::parse_answer(&first.summary),
        ));
        (result, error) = session
            .ask_again(&first, &prompts::verifier_key(&key_text(candidate)))
            .await;
        for round in 0..2 {
            let Some(current) = result.clone() else { break };
            if current.status != AgentRunStatus::Completed {
                break;
            }
            match answer::parse_verifier_verdict(&current.summary) {
                Ok(verdict) => {
                    record.verdict = Some(verdict);
                    record.ok = true;
                    break;
                }
                Err(problem) if round == 0 => {
                    (result, error) = session
                        .ask_again(
                            &current,
                            &format!("Your reply could not be read: {problem}. Call `complete_task` again with only the JSON verdict."),
                        )
                        .await;
                }
                Err(problem) => error = Some(problem),
            }
        }
    }
    let (wall, _) = session.finish();
    record.wall_ms = wall.as_millis() as u64;
    if let Some(result) = &result {
        record.usage = UsageSummary::of_task(&result.task, &context.options.model);
    }
    if !record.ok {
        record.failure_kind = failure_kind_of(result.as_ref());
        record.detail = Some(
            control
                .limit_reason()
                .or(error)
                .or_else(|| result.as_ref().and_then(|result| result.error.clone()))
                .unwrap_or_else(|| "no verdict".into()),
        );
    }
    record.transcript = context.save_transcript(&name, result.as_ref().map(|result| &result.task));
    let _ = std::fs::remove_dir_all(&scratch);
    // A verification cancelled by a stop, or failed by a lost login, isn't
    // the verifier's: it isn't recorded, and runs again later.
    let recorded = record.ok
        || !(context.stop.level() >= 2 || is_login_failure(record.failure_kind.as_deref()));
    if recorded && let Err(error) = context.verifications.append(&record) {
        eprintln!("could not record verification of {}: {error}", candidate.id);
    }
    (record, recorded)
}

/// Builds the task set: the seed tasks, then for each repository and kind
/// the verified questions (the verifier agreed with the key; questions with
/// locations no longer in the checkout are dropped) in the order they were
/// written, up to `per_kind` tasks per repository and kind with the seeds.
pub fn assemble(layout: &Layout, per_kind: usize, path: &Path) -> BenchResult<Vec<BenchTask>> {
    let paths = paths(layout);
    let seeds = crate::tasks::parse_tasks(SEED_TASKS)?;
    let candidates: Vec<Candidate> = read_jsonl(&paths.candidates)?;
    let verifications: Vec<VerificationRecord> = read_jsonl(&paths.verifications)?;
    let prepared = crate::repos::load_prepared(layout)?;
    let mut latest: HashMap<&str, &VerificationRecord> = HashMap::new();
    for record in verifications.iter().filter(|record| record.ok) {
        latest.insert(record.candidate_id.as_str(), record);
    }
    let tasks = select_tasks(&seeds, &candidates, &latest, per_kind);
    let roots: HashMap<String, PathBuf> = prepared
        .iter()
        .map(|(id, repo)| (id.clone(), repo.checkout.clone()))
        .collect();
    let issues = crate::tasks::validate_against_checkout(&tasks, &roots);
    let bad: HashSet<&str> = issues.iter().map(|issue| issue.task.as_str()).collect();
    for issue in &issues {
        eprintln!("dropping {}: {}", issue.task, issue.message);
    }
    let tasks: Vec<BenchTask> = tasks
        .into_iter()
        .filter(|task| !bad.contains(task.id.as_str()))
        .collect();
    crate::tasks::save_tasks(path, &tasks)?;
    eprintln!("wrote {} tasks to {}", tasks.len(), path.display());
    Ok(tasks)
}

/// The selection of [`assemble`], without files.
pub fn select_tasks(
    seeds: &[BenchTask],
    candidates: &[Candidate],
    verdicts: &HashMap<&str, &VerificationRecord>,
    per_kind: usize,
) -> Vec<BenchTask> {
    let mut tasks: Vec<BenchTask> = seeds.to_vec();
    let mut counts = seed_counts(seeds);
    let mut ordered: Vec<&Candidate> = candidates.iter().collect();
    ordered.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
    for candidate in ordered {
        let Some(record) = verdicts.get(candidate.id.as_str()) else {
            continue;
        };
        if record
            .verdict
            .as_ref()
            .is_none_or(|verdict| verdict.verdict != "agree")
        {
            continue;
        }
        let count = counts
            .entry((candidate.repo.clone(), candidate.kind))
            .or_insert(0);
        if *count >= per_kind {
            continue;
        }
        *count += 1;
        tasks.push(candidate.to_task(Some(record)));
    }
    tasks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_tasks_load() {
        let seeds = crate::tasks::parse_tasks(SEED_TASKS).unwrap();
        assert!(seeds.len() >= 19, "{}", seeds.len());
        assert!(seeds.iter().all(|task| task.repo == "arko"));
        assert!(seeds.iter().all(|task| task.origin.source == "seed"));
    }

    fn submit_tool(dir: &Path, kind: TaskKind) -> SubmitQuestionTool {
        let state =
            Arc::new(GenerationState::new(&dir.join("candidates.jsonl"), &[], &[]).unwrap());
        SubmitQuestionTool {
            state,
            repo: "demo".into(),
            kind,
            checkout: dir.join("repo"),
            run_id: "run-1".into(),
            requested: 2,
            accepted: parking_lot::Mutex::new(0),
        }
    }

    fn submission(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), Value::String(value.to_string())))
            .collect()
    }

    #[tokio::test]
    async fn submit_question_checks_keys_against_the_checkout() {
        let dir = crate::test_dir("submit");
        std::fs::create_dir_all(dir.join("repo/src")).unwrap();
        std::fs::write(dir.join("repo/src/retry.rs"), "fn a() {}\n".repeat(30)).unwrap();
        let tool = submit_tool(&dir, TaskKind::Definition);
        let root = dir.join("repo");
        let context = TaskContext::new(Uuid::new_v4(), "writer", "", &root);
        let good = submission(&[
            ("kind", "definition"),
            (
                "question",
                "Where is the function that computes the retry delay defined?",
            ),
            ("answer", "In src/retry.rs, `a`."),
            ("locations", "src/retry.rs:3-5 | the function"),
        ]);
        let reply = tool.execute(&context, good.clone()).await.unwrap();
        assert!(
            reply.starts_with("Accepted as demo-definition-g001"),
            "{reply}"
        );
        let again = tool.execute(&context, good).await.unwrap();
        assert!(again.contains("exists already"), "{again}");
        let past_end = submission(&[
            ("kind", "definition"),
            (
                "question",
                "Where is the type that holds the retry policy defined?",
            ),
            ("answer", "x"),
            (
                "locations",
                "src/retry.rs:29-31 | the type\nsrc/nope.rs:1 | nothing",
            ),
        ]);
        let reply = tool.execute(&context, past_end).await.unwrap();
        assert!(reply.contains("the file has 30 lines"), "{reply}");
        assert!(reply.contains("src/nope.rs does not exist"), "{reply}");
        let gives_away = submission(&[
            ("kind", "definition"),
            ("question", "Where in retry.rs is the backoff computed?"),
            ("answer", "x"),
            ("locations", "src/retry.rs:3 | it"),
        ]);
        assert!(
            tool.execute(&context, gives_away)
                .await
                .unwrap()
                .contains("gives away")
        );
        let wrong_kind = submission(&[
            ("kind", "callers"),
            ("question", "Which functions call the retry helper here?"),
            ("answer", "x"),
            ("locations", "src/retry.rs:3 | it"),
        ]);
        assert!(
            tool.execute(&context, wrong_kind)
                .await
                .unwrap()
                .contains("this run writes")
        );
        let saved: Vec<Candidate> = read_jsonl(&dir.join("candidates.jsonl")).unwrap();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].locations[0].0, Location::new("src/retry.rs", 3, 5));
        assert_eq!(tool.accepted(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn behaviour_keys_need_facts() {
        let dir = crate::test_dir("submit-facts");
        std::fs::create_dir_all(dir.join("repo")).unwrap();
        std::fs::write(dir.join("repo/a.py"), "x = 1\n".repeat(10)).unwrap();
        let tool = submit_tool(&dir, TaskKind::Behaviour);
        let one_fact = submission(&[
            ("kind", "behaviour"),
            ("question", "How does the module decide which value to use?"),
            ("answer", "x"),
            ("locations", "a.py:1-2 | it"),
            ("facts", "It uses x."),
        ]);
        assert!(tool.check(&one_fact).unwrap_err().contains("3 to 6 facts"));
        let enough = submission(&[
            ("kind", "behaviour"),
            ("question", "How does the module decide which value to use?"),
            ("answer", "x"),
            ("locations", "- a.py:1-2 | it"),
            ("facts", "- It uses x.\n- x is 1.\n- Nothing else."),
        ]);
        let candidate = tool.check(&enough).unwrap();
        let key = candidate.key();
        assert_eq!(
            key.items
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["L1", "F1", "F2", "F3"]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn candidate(id: &str, repo: &str, kind: TaskKind, minute: u32) -> Candidate {
        Candidate {
            id: id.into(),
            repo: repo.into(),
            kind,
            question: format!("question {id}"),
            scope: None,
            answer: "a".into(),
            locations: vec![(Location::new("a.rs", 1, 1), "it".into())],
            facts: vec![],
            difficulty: None,
            evidence: String::new(),
            writer_run: "w".into(),
            created_at: chrono::DateTime::from_timestamp(i64::from(minute) * 60, 0).unwrap(),
        }
    }

    fn verification(id: &str, verdict: &str) -> VerificationRecord {
        VerificationRecord {
            candidate_id: id.into(),
            ok: true,
            verdict: Some(VerifierVerdict {
                verdict: verdict.into(),
                problems: vec![],
                notes: String::new(),
            }),
            blind_answer: None,
            blind_check: None,
            detail: None,
            failure_kind: None,
            wall_ms: 0,
            usage: UsageSummary::default(),
            transcript: None,
        }
    }

    #[test]
    fn selection_keeps_agreed_questions_up_to_the_quota() {
        let candidates = vec![
            candidate("r-definition-g001", "r", TaskKind::Definition, 1),
            candidate("r-definition-g002", "r", TaskKind::Definition, 2),
            candidate("r-definition-g003", "r", TaskKind::Definition, 3),
            candidate("r-definition-g004", "r", TaskKind::Definition, 4),
            candidate("r-callers-g001", "r", TaskKind::Callers, 5),
        ];
        let records = [
            verification("r-definition-g001", "agree"),
            verification("r-definition-g002", "disagree"),
            verification("r-definition-g003", "agree"),
            verification("r-definition-g004", "agree"),
            verification("r-callers-g001", "ambiguous"),
        ];
        let verdicts: HashMap<&str, &VerificationRecord> = records
            .iter()
            .map(|record| (record.candidate_id.as_str(), record))
            .collect();
        let tasks = select_tasks(&[], &candidates, &verdicts, 2);
        let ids: Vec<&str> = tasks.iter().map(|task| task.id.as_str()).collect();
        assert_eq!(ids, vec!["r-definition-g001", "r-definition-g003"]);
        assert_eq!(tasks[0].origin.source, "generated");
        // Seeds count toward the quota.
        let mut seed = tasks[0].clone();
        seed.id = "seed-1".into();
        let tasks = select_tasks(&[seed], &candidates, &verdicts, 2);
        let ids: Vec<&str> = tasks.iter().map(|task| task.id.as_str()).collect();
        assert_eq!(ids, vec!["seed-1", "r-definition-g001"]);
    }

    #[test]
    fn writes_until_agreed_and_pending_questions_reach_the_quota() {
        let max_runs = max_writer_runs(9, 1.5);
        assert_eq!(max_runs, 9);
        let fresh = PairProgress::default();
        assert_eq!(fresh.to_write(9, 1.5, max_runs), 14);
        // Seeds count; pending questions count at the expected yield.
        let seeded = PairProgress {
            seeded: 3,
            pending: 4,
            ..PairProgress::default()
        };
        assert_eq!(seeded.to_write(9, 1.5, max_runs), 5);
        // The verifier disagreed with some: write more, until the runs are
        // used up.
        let short = PairProgress {
            agreed: 5,
            writer_runs: 4,
            ..PairProgress::default()
        };
        assert_eq!(short.to_write(9, 1.5, max_runs), 6);
        let used_up = PairProgress {
            writer_runs: max_runs,
            ..short.clone()
        };
        assert_eq!(used_up.to_write(9, 1.5, max_runs), 0);
        let failing = PairProgress {
            infra_runs: 3,
            ..short
        };
        assert_eq!(failing.to_write(9, 1.5, max_runs), 0);
        let done = PairProgress {
            agreed: 9,
            ..PairProgress::default()
        };
        assert_eq!(done.to_write(9, 1.5, max_runs), 0);
    }

    #[tokio::test]
    async fn a_lost_login_stops_generation_and_uses_no_writer_run() {
        let dir = crate::test_dir("generate-login");
        let layout = Layout::new(&dir.join("out")).unwrap();
        let repo = crate::mock::demo_repo(&layout.out);
        let url = crate::mock::unauthorized_server().await;
        let client = crate::agents::build_client(&[], Some(&url)).unwrap();
        let options = || GenerateOptions {
            layout: layout.clone(),
            prepared: BTreeMap::from([("demo".to_string(), repo.clone())]),
            repo_ids: vec!["demo".into()],
            kinds: vec![TaskKind::Definition, TaskKind::Callers],
            per_kind: 2,
            oversample: 1.5,
            concurrency: 4,
            model: ModelSettings::parse("gpt-6-astra:xhigh", true).unwrap(),
            client: client.clone(),
            shell: ShellSettings {
                timeout: Duration::from_secs(5),
                max_output_bytes: 4096,
                max_calls: 5,
            },
            tasks_path: layout.out.join("tasks.json"),
            mock: true,
            writer_arm: Arm::A,
        };
        for _ in 0..3 {
            generate(options()).await.unwrap();
        }
        // No writer run was the writer's: none is recorded, so none counts
        // against the pair's runs, however often the login fails.
        let writers: Vec<WriterRecord> =
            read_jsonl(&layout.tasks_dir().join("writers.jsonl")).unwrap();
        assert!(writers.is_empty(), "{writers:?}");
        // The writers did run (and failed on the login) every time.
        let runs = std::fs::read_dir(layout.tasks_dir().join("transcripts"))
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .is_ok_and(|entry| entry.file_name().to_string_lossy().starts_with("demo-"))
            })
            .count();
        assert!(runs >= 3, "{runs} writer transcripts");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn focus_areas_rank_directories_by_source_files() {
        let dir = crate::test_dir("focus");
        for (path, body) in [
            ("src/core/a.go", "package a"),
            ("src/core/b.go", "package a"),
            ("src/util/c.go", "package c"),
            ("docs/readme.md", "# hi"),
            ("main.go", "package main"),
        ] {
            std::fs::create_dir_all(dir.join(path).parent().unwrap()).unwrap();
            std::fs::write(dir.join(path), body).unwrap();
        }
        assert_eq!(focus_areas(&dir, "go"), vec!["src/core", ".", "src/util"]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
