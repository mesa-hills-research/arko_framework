//! The run's registry of agent statuses.
//!
//! One registry per workflow run records, under one lock, which agents are
//! running (and whether a running agent is waiting, and why), how each
//! agent's latest run ended, and a version that every status change
//! increases. Agents read it three ways:
//!
//! - an agent's initial prompt lists its permitted recipients with their
//!   statuses from a registry snapshot, and the run records the snapshot's
//!   version as seen.
//! - at each safe turn boundary a running agent is told about every
//!   permitted recipient that started after the version it has seen, and
//!   then has seen the current version (start notices are derived from the
//!   registry, not queued when an agent starts, so no ordering of starts
//!   loses or repeats one).
//! - `list_recipients` reports current statuses.
//!
//! A watch channel publishes the version, so `wait_for_message` wakes when
//! an agent starts or finishes without polling.

use std::collections::{BTreeMap, HashMap};

use arko_agents::AgentEvent;
use arko_agents::task::TaskState;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use uuid::Uuid;

use crate::workflow::result::{AgentResult, AgentRunStatus};

/// Why a running agent is waiting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitReason {
    /// Waiting out a provider rate limit.
    RateLimited { until: Option<DateTime<Utc>> },
    /// Waiting to retry a failed model request (a transient or capacity
    /// error).
    Retrying { until: Option<DateTime<Utc>> },
    /// Waiting for a plan usage limit to reset.
    UsageLimit { until: Option<DateTime<Utc>> },
    /// Waiting for a tool call or error recovery to be approved.
    AwaitingApproval,
    /// Waiting for an answer to its questions.
    AwaitingInput,
    /// Waiting in `wait_for_message` for a message from another agent.
    AwaitingMessage,
    /// The workflow is paused.
    Paused,
    /// Held before a model request until the budget is raised (see
    /// [`BudgetExceededPolicy::Pause`](crate::budget::BudgetExceededPolicy::Pause)).
    BudgetExceeded,
}

/// An agent's status in the workflow run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// No run of the agent has started.
    NotStarted,
    /// A run of the agent is running.
    Running,
    /// Every running run of the agent is waiting. The reason is one of
    /// theirs.
    Waiting(WaitReason),
    /// No run is running. The latest ended with this outcome.
    Finished(AgentRunStatus),
}

impl AgentStatus {
    /// Running or waiting, so a message is delivered at its next safe turn
    /// boundary.
    pub fn is_reachable(&self) -> bool {
        matches!(self, Self::Running | Self::Waiting(_))
    }

    /// A compact description, as agents are shown it, `not_started`,
    /// `running`, `waiting (rate_limited until 2026-09-27T12:00:00Z)`, or
    /// `finished (completed)`.
    pub fn describe(&self) -> String {
        match self {
            Self::NotStarted => "not_started".to_string(),
            Self::Running => "running".to_string(),
            Self::Waiting(reason) => {
                let (name, until) = match reason {
                    WaitReason::RateLimited { until } => ("rate_limited", *until),
                    WaitReason::Retrying { until } => ("retrying", *until),
                    WaitReason::UsageLimit { until } => ("usage_limit", *until),
                    WaitReason::AwaitingApproval => ("awaiting_approval", None),
                    WaitReason::AwaitingInput => ("awaiting_input", None),
                    WaitReason::AwaitingMessage => ("awaiting_message", None),
                    WaitReason::Paused => ("paused", None),
                    WaitReason::BudgetExceeded => ("budget_exceeded", None),
                };
                match until {
                    Some(until) => format!(
                        "waiting ({name} until {})",
                        until.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
                    ),
                    None => format!("waiting ({name})"),
                }
            }
            Self::Finished(outcome) => format!("finished ({})", outcome_name(outcome)),
        }
    }
}

fn outcome_name(outcome: &AgentRunStatus) -> &'static str {
    match outcome {
        AgentRunStatus::Completed => "completed",
        AgentRunStatus::Failed => "failed",
        AgentRunStatus::Interrupted => "interrupted",
        AgentRunStatus::Cancelled => "cancelled",
        AgentRunStatus::WaitingForInput => "waiting_for_input",
        AgentRunStatus::Paused => "paused",
    }
}

/// What the registry records about one agent across runs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRecord {
    /// The registry version at the agent's latest start (0 means never started).
    #[serde(default)]
    pub last_start: u64,
    /// How the agent's latest run ended, when no run is running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished: Option<AgentRunStatus>,
    /// The registry version at which the agent last finished, when its last
    /// running run ended (0 means never, or not recorded by an older version).
    #[serde(default)]
    pub last_finish: u64,
}

/// A change in an agent's activity a run is told about. See
/// [`AgentRegistry::changes_since_seen`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ActivityChange {
    /// The agent started and is running.
    Started(String),
    /// The agent finished, with its latest run's outcome, and no run of it
    /// is running.
    Finished(String, AgentRunStatus),
}

/// The registry state a checkpoint saves in
/// [`WorkflowState::agent_registry`](crate::persistence::state::WorkflowState::agent_registry).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistrySnapshot {
    /// Increases with every status change.
    #[serde(default)]
    pub version: u64,
    /// Per registered agent name.
    #[serde(default)]
    pub agents: BTreeMap<String, AgentRecord>,
    /// Per agent run (task ID), the version whose starts it has been told
    /// about.
    #[serde(default)]
    pub seen: HashMap<Uuid, u64>,
}

struct Run {
    agent: String,
    waiting: Option<WaitReason>,
    /// In `wait_for_message`, set and cleared by the tool alone, so the
    /// driver's events (observed a little later) can't end the wait early.
    awaiting_message: bool,
    /// The agent's record before this run started, restored if the run is
    /// abandoned before it starts.
    previous: AgentRecord,
    /// The version of the start this run made, if it made one (no other run
    /// of the agent was running).
    started: Option<u64>,
    /// The run inside whose tool call this one runs, if any.
    parent: Option<Uuid>,
}

#[derive(Default)]
struct RegistryState {
    saved: RegistrySnapshot,
    runs: HashMap<Uuid, Run>,
    /// Per run being recovered (see
    /// [`AgentRegistry::claim_recovery`]), a receiver that closes when its
    /// recovery ends. Closed entries are removed as they are found.
    recoveries: HashMap<Uuid, watch::Receiver<()>>,
}

/// A recovery's claim on the interrupted run it recovers (see
/// [`AgentRegistry::claim_recovery`]). Held until the recovery has recorded
/// the run's outcome. Dropping it wakes the recoveries waiting for it.
pub(crate) struct RecoveryClaim(#[allow(dead_code)] watch::Sender<()>);

/// The status registry of one workflow run.
pub(crate) struct AgentRegistry {
    state: Mutex<RegistryState>,
    /// The current version, published under the state lock.
    versions: watch::Sender<u64>,
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::with_state(RegistryState::default())
    }
}

/// Whether an agent runs and when it last started: what a change of tells
/// a waiting agent that the agent started or finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Activity {
    pub last_start: u64,
    pub running: bool,
}

impl AgentRegistry {
    /// Restores the registry of a resumed run. No run is running after a
    /// resume. An agent that was running at the checkpoint counts as
    /// finished, interrupted. A checkpoint from before the registry existed
    /// gives each agent with a recorded result its latest result's outcome.
    pub fn restore(snapshot: Option<RegistrySnapshot>, results: &[AgentResult]) -> Self {
        let mut saved = snapshot.unwrap_or_default();
        let mut latest = HashMap::<&str, &AgentResult>::new();
        for result in results {
            let newer = latest
                .get(result.agent_name.as_str())
                .is_none_or(|current| {
                    (result.info().end_time, result.info().start_time)
                        >= (current.info().end_time, current.info().start_time)
                });
            if newer {
                latest.insert(&result.agent_name, result);
            }
        }
        for record in saved.agents.values_mut() {
            if record.last_start > 0 && record.finished.is_none() {
                record.finished = Some(AgentRunStatus::Interrupted);
            }
        }
        for (agent, result) in &latest {
            let record = saved.agents.entry((*agent).to_string()).or_default();
            if record.finished.is_none() {
                record.finished = Some(result.status.clone());
            }
        }
        Self::with_state(RegistryState {
            saved,
            runs: HashMap::new(),
            recoveries: HashMap::new(),
        })
    }

    fn with_state(state: RegistryState) -> Self {
        Self {
            versions: watch::Sender::new(state.saved.version),
            state: Mutex::new(state),
        }
    }

    /// Publishes `state`'s version. Called with the state lock held, so
    /// versions are published in order.
    fn publish(&self, state: &RegistryState) {
        let version = state.saved.version;
        self.versions.send_if_modified(|published| {
            let changed = *published != version;
            *published = version;
            changed
        });
    }

    /// The current version, changing with every status change.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.versions.subscribe()
    }

    /// The state a checkpoint saves.
    pub fn snapshot(&self) -> RegistrySnapshot {
        self.state.lock().saved.clone()
    }

    /// Marks agent runs as running, together. A batch started at once sees
    /// each other as running. An agent that had no running run starts at a
    /// new version.
    pub fn start(&self, runs: &[(Uuid, &str)]) {
        let mut guard = self.state.lock();
        self.start_locked(&mut guard, runs);
    }

    /// [`start`](Self::start), with the state lock held.
    fn start_locked(&self, state: &mut RegistryState, runs: &[(Uuid, &str)]) {
        for (task_id, agent) in runs {
            let already_running = state.runs.values().any(|run| run.agent == *agent);
            let record = state.saved.agents.entry((*agent).to_string()).or_default();
            let previous = record.clone();
            let started = (!already_running).then(|| {
                state.saved.version += 1;
                record.last_start = state.saved.version;
                record.finished = None;
                state.saved.version
            });
            state.runs.insert(
                *task_id,
                Run {
                    agent: (*agent).to_string(),
                    waiting: None,
                    awaiting_message: false,
                    previous,
                    started,
                    parent: None,
                },
            );
        }
        self.publish(state);
    }

    /// Whether run `task_id` is marked running.
    pub fn is_running(&self, task_id: Uuid) -> bool {
        self.state.lock().runs.contains_key(&task_id)
    }

    /// The runs of `agent` marked running.
    pub fn runs_of(&self, agent: &str) -> Vec<Uuid> {
        self.state
            .lock()
            .runs
            .iter()
            .filter(|(_, run)| run.agent == agent)
            .map(|(task_id, _)| *task_id)
            .collect()
    }

    /// The running runs inside a tool call of run `task_id`.
    pub fn children(&self, task_id: Uuid) -> Vec<Uuid> {
        self.state
            .lock()
            .runs
            .iter()
            .filter(|(_, run)| run.parent == Some(task_id))
            .map(|(child, _)| *child)
            .collect()
    }

    /// Marks run `task_id` of `agent` running, as [`start`](Self::start)
    /// does, unless it is running already. Returns whether it was marked.
    pub fn start_unless_running(&self, task_id: Uuid, agent: &str) -> bool {
        let mut guard = self.state.lock();
        if guard.runs.contains_key(&task_id) {
            return false;
        }
        self.start_locked(&mut guard, &[(task_id, agent)]);
        true
    }

    /// Marks interrupted run `task_id` of `agent` running for its recovery,
    /// as [`start_unless_running`](Self::start_unless_running) does, and
    /// returns the claim to hold until the recovery has recorded the run's
    /// outcome. When the run is running already, returns the receiver of
    /// the recovery running it (see [`recovery`](Self::recovery)), or
    /// `None` when no recovery does.
    pub fn claim_recovery(
        &self,
        task_id: Uuid,
        agent: &str,
    ) -> Result<RecoveryClaim, Option<watch::Receiver<()>>> {
        let mut guard = self.state.lock();
        if let Some(recovering) = Self::recovery_locked(&mut guard, task_id) {
            return Err(Some(recovering));
        }
        if guard.runs.contains_key(&task_id) {
            return Err(None);
        }
        self.start_locked(&mut guard, &[(task_id, agent)]);
        let (claim, recovering) = watch::channel(());
        guard.recoveries.insert(task_id, recovering);
        Ok(RecoveryClaim(claim))
    }

    /// The recovery running run `task_id`, if one does. Its receiver's
    /// `changed` fails once the recovery has ended (and recorded the run's
    /// outcome).
    pub fn recovery(&self, task_id: Uuid) -> Option<watch::Receiver<()>> {
        Self::recovery_locked(&mut self.state.lock(), task_id)
    }

    fn recovery_locked(state: &mut RegistryState, task_id: Uuid) -> Option<watch::Receiver<()>> {
        let recovering = state.recoveries.get(&task_id)?;
        if recovering.has_changed().is_ok() {
            return Some(recovering.clone());
        }
        state.recoveries.remove(&task_id);
        None
    }

    /// Removes a run marked running whose start then failed. When no other
    /// run of its agent is running and this run made the agent's latest
    /// start, the agent's previous record is restored. Otherwise the record
    /// keeps the outcome of the agent's runs that did run.
    pub fn abandon(&self, task_id: Uuid) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let Some(run) = state.runs.remove(&task_id) else {
            return;
        };
        state.saved.version += 1;
        if !state.runs.values().any(|other| other.agent == run.agent) {
            let record = state.saved.agents.entry(run.agent).or_default();
            if run.started == Some(record.last_start) {
                *record = run.previous;
            }
        }
        state.saved.seen.remove(&task_id);
        self.publish(state);
    }

    /// Records that a run ended with `outcome`, the agent's latest outcome.
    /// The agent is finished once no other run of it is running. What a run
    /// that can't be continued has seen is forgotten.
    pub fn finish(&self, task_id: Uuid, outcome: AgentRunStatus) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let continuable = matches!(
            outcome,
            AgentRunStatus::Completed | AgentRunStatus::WaitingForInput | AgentRunStatus::Paused
        );
        if !continuable {
            state.saved.seen.remove(&task_id);
        }
        let Some(run) = state.runs.remove(&task_id) else {
            return;
        };
        state.saved.version += 1;
        let last_run = !state.runs.values().any(|other| other.agent == run.agent);
        let record = state.saved.agents.entry(run.agent).or_default();
        record.finished = Some(outcome);
        if last_run {
            record.last_finish = state.saved.version;
        }
        self.publish(state);
    }

    /// Updates a running run's waiting state from one of its events.
    pub fn observe(&self, task_id: Uuid, event: &AgentEvent) {
        let waiting = match event {
            AgentEvent::RetryScheduled {
                delay_ms,
                rate_limit,
                ..
            } => {
                let until = Some(
                    Utc::now()
                        + chrono::Duration::milliseconds(
                            i64::try_from(*delay_ms).unwrap_or(i64::MAX),
                        ),
                );
                Some(Some(if *rate_limit {
                    WaitReason::RateLimited { until }
                } else {
                    WaitReason::Retrying { until }
                }))
            }
            AgentEvent::UsageLimitReached {
                waiting_until_unix_ms,
                ..
            } => Some(Some(WaitReason::UsageLimit {
                until: waiting_until_unix_ms
                    .and_then(|ms| i64::try_from(ms).ok())
                    .and_then(DateTime::from_timestamp_millis),
            })),
            AgentEvent::TaskStateChanged(
                TaskState::AwaitingToolApproval { .. }
                | TaskState::AwaitingBatchToolApproval { .. }
                | TaskState::AwaitingErrorRecoveryApproval { .. },
            ) => Some(Some(WaitReason::AwaitingApproval)),
            AgentEvent::TaskStateChanged(TaskState::AwaitingUserResponse { .. })
            | AgentEvent::UserQuestionsAsked { .. } => Some(Some(WaitReason::AwaitingInput)),
            // Reports about the wait itself, or about usage, don't end it.
            AgentEvent::ErrorEncountered { .. }
            | AgentEvent::PlanUsageReported(_)
            | AgentEvent::PlanUsageThresholdReached { .. }
            | AgentEvent::CacheEfficiencyReported { .. }
            | AgentEvent::CacheEfficiencyLow { .. }
            | AgentEvent::TaskStreamEnd => None,
            // Anything else is progress.
            _ => Some(None),
        };
        let Some(waiting) = waiting else {
            return;
        };
        let mut guard = self.state.lock();
        let state = &mut *guard;
        if let Some(run) = state.runs.get_mut(&task_id)
            && run.waiting != waiting
        {
            run.waiting = waiting;
            state.saved.version += 1;
            self.publish(state);
        }
    }

    /// Marks a running run as waiting for `reason` until its next event says
    /// otherwise (the workflow's own waits, such as automatic error
    /// recovery's).
    pub fn set_waiting(&self, task_id: Uuid, reason: WaitReason) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        if let Some(run) = state.runs.get_mut(&task_id)
            && run.waiting.as_ref() != Some(&reason)
        {
            run.waiting = Some(reason);
            state.saved.version += 1;
            self.publish(state);
        }
    }

    /// Records that running run `task_id` runs inside a tool call of run
    /// `parent`.
    pub fn set_parent(&self, task_id: Uuid, parent: Option<Uuid>) {
        if let Some(run) = self.state.lock().runs.get_mut(&task_id) {
            run.parent = parent;
        }
    }

    /// Whether the budget holds up the whole run because some run waits for a
    /// raise ([`WaitReason::BudgetExceeded`]), and every other running run
    /// does too, waits for a message, or waits in a tool call for a run it
    /// started (one of them, in turn). Nothing then runs until the budget
    /// is raised.
    pub fn blocked_on_budget(&self) -> bool {
        let state = self.state.lock();
        let mut held = false;
        for (task_id, run) in &state.runs {
            if run.waiting == Some(WaitReason::BudgetExceeded) {
                held = true;
            } else if !run.awaiting_message
                && !state
                    .runs
                    .values()
                    .any(|other| other.parent == Some(*task_id))
            {
                return false;
            }
        }
        held
    }

    /// Marks run `task_id` as waiting in `wait_for_message`, or no longer.
    pub fn set_awaiting_message(&self, task_id: Uuid, awaiting: bool) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        if let Some(run) = state.runs.get_mut(&task_id)
            && run.awaiting_message != awaiting
        {
            run.awaiting_message = awaiting;
            state.saved.version += 1;
            self.publish(state);
        }
    }

    /// Whether each of `agents` runs, and when it last started.
    pub fn activity(&self, agents: &[String]) -> Vec<Activity> {
        let state = self.state.lock();
        agents
            .iter()
            .map(|agent| Activity {
                last_start: state
                    .saved
                    .agents
                    .get(agent)
                    .map_or(0, |record| record.last_start),
                running: state.runs.values().any(|run| &run.agent == agent),
            })
            .collect()
    }

    /// The statuses of `agents` now. `paused` reports running agents as
    /// waiting for the paused workflow.
    pub fn statuses(&self, agents: &[String], paused: bool) -> Vec<(String, AgentStatus)> {
        let state = self.state.lock();
        agents
            .iter()
            .map(|agent| (agent.clone(), status(&state, agent, paused)))
            .collect()
    }

    /// The statuses of `recipients` for run `task_id`'s initial prompt,
    /// recording their version as seen by the run.
    pub fn roster(
        &self,
        task_id: Uuid,
        recipients: &[String],
        paused: bool,
    ) -> Vec<(String, AgentStatus)> {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let roster = recipients
            .iter()
            .map(|agent| (agent.clone(), status(state, agent, paused)))
            .collect();
        state.saved.seen.insert(task_id, state.saved.version);
        roster
    }

    /// The recipients (of `recipients`) that started after the version run
    /// `task_id` has seen and are still running, in order. The run has then
    /// seen the current version. A run without a seen version (a task from
    /// an older checkpoint) is told about every running recipient.
    #[cfg(test)]
    pub fn started_since_seen(&self, task_id: Uuid, recipients: &[String]) -> Vec<String> {
        self.changes_since_seen(task_id, recipients)
            .into_iter()
            .filter_map(|change| match change {
                ActivityChange::Started(agent) => Some(agent),
                ActivityChange::Finished(..) => None,
            })
            .collect()
    }

    /// The recipients (of `recipients`) that started or finished after the
    /// version run `task_id` has seen, in order: those running that started
    /// since, and those not running that finished since (one that did both
    /// is reported finished). The run has then seen the current version. A
    /// run without a seen version (a task from an older checkpoint) is told
    /// about every running recipient, and none that finished.
    pub(crate) fn changes_since_seen(
        &self,
        task_id: Uuid,
        recipients: &[String],
    ) -> Vec<ActivityChange> {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let seen = state.saved.seen.get(&task_id).copied();
        let changes = recipients
            .iter()
            .filter_map(|agent| {
                let record = state.saved.agents.get(agent)?;
                let running = state.runs.values().any(|run| &run.agent == agent);
                if running {
                    (record.last_start > seen.unwrap_or(0))
                        .then(|| ActivityChange::Started(agent.clone()))
                } else {
                    let outcome = record.finished.clone()?;
                    seen.filter(|seen| record.last_finish > *seen)
                        .map(|_| ActivityChange::Finished(agent.clone(), outcome))
                }
            })
            .collect();
        state.saved.seen.insert(task_id, state.saved.version);
        changes
    }
}

fn until_passed(reason: &WaitReason, now: DateTime<Utc>) -> bool {
    match reason {
        WaitReason::RateLimited { until }
        | WaitReason::Retrying { until }
        | WaitReason::UsageLimit { until } => until.is_some_and(|until| until <= now),
        _ => false,
    }
}

fn status(state: &RegistryState, agent: &str, paused: bool) -> AgentStatus {
    let mut runs = state
        .runs
        .values()
        .filter(|run| run.agent == agent)
        .peekable();
    if runs.peek().is_some() {
        if paused {
            return AgentStatus::Waiting(WaitReason::Paused);
        }
        let now = Utc::now();
        let mut first_wait = None;
        for run in runs {
            if run.awaiting_message {
                first_wait.get_or_insert(WaitReason::AwaitingMessage);
                continue;
            }
            match &run.waiting {
                // A wait whose time has passed is over: the retry is under way.
                Some(reason) if !until_passed(reason, now) => {
                    first_wait.get_or_insert_with(|| reason.clone());
                }
                _ => return AgentStatus::Running,
            }
        }
        return AgentStatus::Waiting(first_wait.unwrap_or(WaitReason::Paused));
    }
    match state
        .saved
        .agents
        .get(agent)
        .and_then(|record| record.finished.clone())
    {
        Some(outcome) => AgentStatus::Finished(outcome),
        None => AgentStatus::NotStarted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(agents: &[&str]) -> Vec<String> {
        agents.iter().map(|agent| agent.to_string()).collect()
    }

    fn status_of(registry: &AgentRegistry, agent: &str) -> AgentStatus {
        registry.statuses(&names(&[agent]), false)[0].1.clone()
    }

    #[test]
    fn statuses_follow_a_run_from_start_through_waits_to_its_end() {
        let registry = AgentRegistry::default();
        let run = Uuid::new_v4();
        assert_eq!(status_of(&registry, "worker"), AgentStatus::NotStarted);
        registry.start(&[(run, "worker")]);
        assert_eq!(status_of(&registry, "worker"), AgentStatus::Running);
        registry.observe(
            run,
            &AgentEvent::RetryScheduled {
                attempt: 1,
                delay_ms: 30_000,
                elapsed_ms: 0,
                rate_limit: true,
            },
        );
        // The error report that accompanies the wait doesn't end it.
        registry.observe(
            run,
            &AgentEvent::ErrorEncountered {
                error_message: "429".into(),
                requires_approval_to_retry: false,
                tool_call_id: None,
                error_kind: None,
            },
        );
        let AgentStatus::Waiting(WaitReason::RateLimited { until: Some(until) }) =
            status_of(&registry, "worker")
        else {
            panic!("rate limited");
        };
        assert!(until > Utc::now());
        assert!(
            status_of(&registry, "worker")
                .describe()
                .starts_with("waiting (rate_limited until ")
        );
        assert_eq!(
            registry.statuses(&names(&["worker"]), true)[0].1,
            AgentStatus::Waiting(WaitReason::Paused)
        );
        registry.observe(run, &AgentEvent::AssistantResponseChunk("ok".into()));
        assert_eq!(status_of(&registry, "worker"), AgentStatus::Running);
        registry.observe(
            run,
            &AgentEvent::UsageLimitReached {
                resets_at_unix_ms: None,
                waiting_until_unix_ms: Some(4_102_444_800_000),
                error_message: "limit".into(),
            },
        );
        assert_eq!(
            status_of(&registry, "worker").describe(),
            "waiting (usage_limit until 2100-01-01T00:00:00Z)"
        );
        registry.finish(run, AgentRunStatus::Completed);
        assert_eq!(
            status_of(&registry, "worker").describe(),
            "finished (completed)"
        );
    }

    #[test]
    fn a_run_is_told_about_each_start_after_its_roster_exactly_once() {
        let registry = AgentRegistry::default();
        let (lead, worker) = (Uuid::new_v4(), Uuid::new_v4());
        let recipients = names(&["worker"]);
        registry.start(&[(lead, "lead")]);
        assert_eq!(
            registry.roster(lead, &recipients, false),
            [("worker".to_string(), AgentStatus::NotStarted)]
        );
        assert!(registry.started_since_seen(lead, &recipients).is_empty());
        registry.start(&[(worker, "worker")]);
        assert_eq!(registry.started_since_seen(lead, &recipients), ["worker"]);
        assert!(registry.started_since_seen(lead, &recipients).is_empty());
        // A second run of a running agent is not a new start.
        registry.start(&[(Uuid::new_v4(), "worker")]);
        assert!(registry.started_since_seen(lead, &recipients).is_empty());
    }

    /// Whatever the order of the other agent's start relative to a run's
    /// start and roster, the run learns that it is running exactly once.
    #[test]
    fn every_ordering_of_two_starts_tells_each_run_once() {
        for order in 0..3 {
            let registry = AgentRegistry::default();
            let (lead, worker) = (Uuid::new_v4(), Uuid::new_v4());
            let recipients = names(&["worker"]);
            if order == 0 {
                registry.start(&[(worker, "worker")]);
            }
            registry.start(&[(lead, "lead")]);
            if order == 1 {
                registry.start(&[(worker, "worker")]);
            }
            let roster = registry.roster(lead, &recipients, false);
            if order == 2 {
                registry.start(&[(worker, "worker")]);
            }
            let notices = registry.started_since_seen(lead, &recipients);
            let in_roster = roster[0].1 == AgentStatus::Running;
            // Exactly one of the two tells the lead: the roster or a notice.
            assert_eq!(in_roster, notices.is_empty(), "order {order}");
            assert_eq!(in_roster, order < 2, "order {order}");
            assert!(registry.started_since_seen(lead, &recipients).is_empty());
        }
    }

    /// Three agents that may all message each other start concurrently, many
    /// times over: each learns of each other agent exactly once, from its
    /// roster or a notice.
    #[test]
    fn concurrent_starts_neither_miss_nor_repeat_a_start() {
        let agents = ["a", "b", "c"];
        for _ in 0..200 {
            let registry = std::sync::Arc::new(AgentRegistry::default());
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(agents.len()));
            let threads = agents
                .iter()
                .map(|agent| {
                    let registry = registry.clone();
                    let barrier = barrier.clone();
                    let agent = agent.to_string();
                    std::thread::spawn(move || {
                        let others = ["a", "b", "c"]
                            .iter()
                            .filter(|other| **other != agent)
                            .map(|other| other.to_string())
                            .collect::<Vec<_>>();
                        let run = Uuid::new_v4();
                        barrier.wait();
                        registry.start(&[(run, agent.as_str())]);
                        let roster = registry.roster(run, &others, false);
                        let mut told = roster
                            .into_iter()
                            .filter(|(_, status)| *status == AgentStatus::Running)
                            .map(|(other, _)| other)
                            .collect::<Vec<_>>();
                        // Turn boundaries until everyone has started.
                        barrier.wait();
                        for _ in 0..3 {
                            told.extend(registry.started_since_seen(run, &others));
                        }
                        told.sort();
                        assert_eq!(told, others, "{agent}");
                    })
                })
                .collect::<Vec<_>>();
            for thread in threads {
                thread.join().unwrap();
            }
        }
    }

    #[test]
    fn a_batch_started_together_sees_each_other_as_running() {
        let registry = AgentRegistry::default();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        registry.start(&[(a, "a"), (b, "b")]);
        assert_eq!(
            registry.roster(a, &names(&["b"]), false)[0].1,
            AgentStatus::Running
        );
        assert_eq!(
            registry.roster(b, &names(&["a"]), false)[0].1,
            AgentStatus::Running
        );
        assert!(registry.started_since_seen(a, &names(&["b"])).is_empty());
        assert!(registry.started_since_seen(b, &names(&["a"])).is_empty());
        // A run abandoned before it started restores its agent's status.
        registry.abandon(b);
        assert_eq!(status_of(&registry, "b"), AgentStatus::NotStarted);

        // Two runs of one agent in a batch: the first runs and completes,
        // the second is abandoned; the agent is finished, not unstarted.
        let (first, second) = (Uuid::new_v4(), Uuid::new_v4());
        registry.start(&[(first, "c"), (second, "c")]);
        registry.finish(first, AgentRunStatus::Completed);
        assert_eq!(status_of(&registry, "c"), AgentStatus::Running);
        registry.abandon(second);
        assert_eq!(
            status_of(&registry, "c"),
            AgentStatus::Finished(AgentRunStatus::Completed)
        );
    }

    /// A run in `wait_for_message` is waiting until the tool says otherwise,
    /// whatever events the driver observes meanwhile. A pause still reports
    /// paused. Each change is a new, published version.
    #[test]
    fn a_run_awaiting_a_message_waits_until_the_tool_ends_the_wait() {
        let registry = AgentRegistry::default();
        let mut versions = registry.subscribe();
        let run = Uuid::new_v4();
        registry.start(&[(run, "worker")]);
        assert!(versions.has_changed().unwrap());
        let started = *versions.borrow_and_update();
        registry.observe(
            run,
            &AgentEvent::TaskStateChanged(TaskState::AwaitingToolApproval {
                tool_call_id: Uuid::new_v4(),
                tool_name: "wait_for_message".into(),
                parameters: Default::default(),
                tool_description: String::new(),
                approval_reason: Default::default(),
            }),
        );
        registry.set_awaiting_message(run, true);
        let waiting = AgentStatus::Waiting(WaitReason::AwaitingMessage);
        assert_eq!(status_of(&registry, "worker"), waiting);
        assert_eq!(waiting.describe(), "waiting (awaiting_message)");
        // A progress event observed late doesn't end the wait.
        registry.observe(run, &AgentEvent::AssistantResponseChunk("ok".into()));
        assert_eq!(status_of(&registry, "worker"), waiting);
        assert_eq!(
            registry.statuses(&names(&["worker"]), true)[0].1,
            AgentStatus::Waiting(WaitReason::Paused)
        );
        assert!(*versions.borrow_and_update() > started);
        // Setting it again changes nothing.
        registry.set_awaiting_message(run, true);
        assert!(!versions.has_changed().unwrap());
        registry.set_awaiting_message(run, false);
        assert_eq!(status_of(&registry, "worker"), AgentStatus::Running);
        assert!(versions.has_changed().unwrap());
        // With another run of the agent running, the agent is running.
        let other = Uuid::new_v4();
        registry.start(&[(other, "worker")]);
        registry.set_awaiting_message(run, true);
        assert_eq!(status_of(&registry, "worker"), AgentStatus::Running);
        registry.set_awaiting_message(other, true);
        assert_eq!(status_of(&registry, "worker"), waiting);
        // Unknown runs are ignored.
        registry.set_awaiting_message(Uuid::new_v4(), true);
    }

    /// Activity changes with starts and finishes only.
    #[test]
    fn activity_follows_starts_and_finishes() {
        let registry = AgentRegistry::default();
        let agents = names(&["worker"]);
        let before = registry.activity(&agents);
        assert_eq!(
            before,
            [Activity {
                last_start: 0,
                running: false
            }]
        );
        let run = Uuid::new_v4();
        registry.start(&[(run, "worker")]);
        let running = registry.activity(&agents);
        assert!(running[0].running && running[0].last_start > 0);
        registry.set_awaiting_message(run, true);
        registry.observe(
            run,
            &AgentEvent::RetryScheduled {
                attempt: 1,
                delay_ms: 60_000,
                elapsed_ms: 0,
                rate_limit: true,
            },
        );
        assert_eq!(registry.activity(&agents), running);
        registry.finish(run, AgentRunStatus::Completed);
        let finished = registry.activity(&agents);
        assert!(!finished[0].running);
        assert_eq!(finished[0].last_start, running[0].last_start);
    }

    #[test]
    fn the_message_wait_reason_serializes_in_snake_case() {
        let waiting = AgentStatus::Waiting(WaitReason::AwaitingMessage);
        let json = serde_json::to_value(&waiting).unwrap();
        assert_eq!(json, serde_json::json!({"waiting": "awaiting_message"}));
        assert_eq!(
            serde_json::from_value::<AgentStatus>(json).unwrap(),
            waiting
        );
    }

    #[test]
    fn a_restored_registry_publishes_its_version() {
        let registry = AgentRegistry::default();
        registry.start(&[(Uuid::new_v4(), "worker")]);
        let restored = AgentRegistry::restore(Some(registry.snapshot()), &[]);
        assert_eq!(*restored.subscribe().borrow(), registry.snapshot().version);
    }

    #[test]
    fn a_wait_whose_time_has_passed_is_over() {
        let registry = AgentRegistry::default();
        let run = Uuid::new_v4();
        registry.start(&[(run, "worker")]);
        registry.observe(
            run,
            &AgentEvent::RetryScheduled {
                attempt: 1,
                delay_ms: 0,
                elapsed_ms: 0,
                rate_limit: false,
            },
        );
        assert_eq!(status_of(&registry, "worker"), AgentStatus::Running);
    }

    #[test]
    fn a_resumed_run_neither_repeats_nor_misses_a_start() {
        let registry = AgentRegistry::default();
        let (lead, worker) = (Uuid::new_v4(), Uuid::new_v4());
        let recipients = names(&["worker", "reviewer"]);
        registry.start(&[(worker, "worker")]);
        registry.start(&[(lead, "lead")]);
        registry.roster(lead, &recipients, false);
        registry.finish(lead, AgentRunStatus::Completed);
        let snapshot = registry.snapshot();
        let json = serde_json::to_value(&snapshot).unwrap();
        let restored: RegistrySnapshot = serde_json::from_value(json).unwrap();
        assert_eq!(restored, snapshot);

        // The worker was running at the checkpoint: after the resume it is
        // no longer running.
        let resumed = AgentRegistry::restore(Some(restored), &[]);
        assert_eq!(
            status_of(&resumed, "worker"),
            AgentStatus::Finished(AgentRunStatus::Interrupted)
        );
        // The lead continues; a reviewer starts, then the worker again.
        resumed.start(&[(lead, "lead")]);
        assert!(resumed.started_since_seen(lead, &recipients).is_empty());
        resumed.start(&[(Uuid::new_v4(), "reviewer")]);
        resumed.start(&[(Uuid::new_v4(), "worker")]);
        assert_eq!(
            resumed.started_since_seen(lead, &recipients),
            ["worker", "reviewer"]
        );
        assert!(resumed.started_since_seen(lead, &recipients).is_empty());
        // An older checkpoint has no registry.
        let legacy = AgentRegistry::restore(None, &[]);
        assert_eq!(status_of(&legacy, "worker"), AgentStatus::NotStarted);
    }

    /// A run is told once when a recipient finishes after its roster, only
    /// when the recipient's last running run ends. A recipient that started
    /// and finished since is reported finished, and a finish before the
    /// roster is not reported.
    #[test]
    fn a_run_is_told_about_each_finish_after_its_roster_once() {
        let registry = AgentRegistry::default();
        let recipients = names(&["early", "worker"]);
        let early = Uuid::new_v4();
        registry.start(&[(early, "early")]);
        registry.finish(early, AgentRunStatus::Completed);
        let lead = Uuid::new_v4();
        registry.start(&[(lead, "lead")]);
        registry.roster(lead, &recipients, false);

        let (first, second) = (Uuid::new_v4(), Uuid::new_v4());
        registry.start(&[(first, "worker"), (second, "worker")]);
        registry.finish(first, AgentRunStatus::Completed);
        // Another run of it still runs: it started, not finished.
        assert_eq!(
            registry.changes_since_seen(lead, &recipients),
            [ActivityChange::Started("worker".into())]
        );
        registry.finish(second, AgentRunStatus::Failed);
        assert_eq!(
            registry.changes_since_seen(lead, &recipients),
            [ActivityChange::Finished(
                "worker".into(),
                AgentRunStatus::Failed
            )]
        );
        assert!(registry.changes_since_seen(lead, &recipients).is_empty());

        // Started and finished between two looks: reported finished.
        let again = Uuid::new_v4();
        registry.start(&[(again, "worker")]);
        registry.finish(again, AgentRunStatus::Completed);
        assert_eq!(
            registry.changes_since_seen(lead, &recipients),
            [ActivityChange::Finished(
                "worker".into(),
                AgentRunStatus::Completed
            )]
        );

        // A run with no seen version (an older checkpoint's) hears of no
        // finish.
        assert!(
            registry
                .changes_since_seen(Uuid::new_v4(), &recipients)
                .is_empty()
        );
    }

    #[test]
    fn records_saved_before_finish_versions_load() {
        let record: AgentRecord = serde_json::from_str(r#"{"last_start":3}"#).unwrap();
        assert_eq!(record.last_finish, 0);
    }
}
