//! Cancelling a workflow run.
//!
//! A run stops when a [`WorkflowCancelHandle`] cancels or
//! [stops](WorkflowCancelHandle::stop) it, or a hook returns
//! [`HookResult::Abort`](crate::hooks::HookResult::Abort). Either way every
//! running agent is told to stop, and every run call in progress or made
//! later returns the stop's error: [`WorkflowError::Cancelled`] or
//! [`WorkflowError::AbortedByHook`]. A graceful stop with a wrap-up first
//! lets running agents finish; see [`stop`](crate::workflow::stop).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use arko_agents::runtime::command::UserCommand;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::workflow::stop::{RunStopIntent, StopKind, StopRecord, StopRequest};

/// How long a stopped run waits for each running agent to stop before it
/// aborts the agent; see [`WorkflowBuilder::cancel_grace_period`](crate::WorkflowBuilder::cancel_grace_period).
pub const DEFAULT_CANCEL_GRACE_PERIOD: Duration = Duration::from_secs(30);

/// Why a workflow run was stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelCause {
    /// A [`WorkflowCancelHandle`] cancelled the run.
    Requested,
    /// A hook returned [`HookResult::Abort`](crate::hooks::HookResult::Abort).
    AbortedByHook { hook_point: String, reason: String },
    /// The run was stopped with a [`StopRequest`]
    /// ([`Workflow::stop`](crate::Workflow::stop)), after its wrap-up if it
    /// had one. In a [`Cancellation`] it is saved in a form older versions
    /// read as [`Requested`](Self::Requested); see [`Cancellation`].
    Stopped { kind: StopKind, reason: String },
}

impl CancelCause {
    /// The error run calls return once the run stopped for this cause:
    /// `Cancelled` for a cancellation or a stop.
    pub fn to_error(&self) -> WorkflowError {
        match self {
            Self::Requested | Self::Stopped { .. } => WorkflowError::Cancelled,
            Self::AbortedByHook { hook_point, reason } => WorkflowError::AbortedByHook {
                hook_point: hook_point.clone(),
                reason: reason.clone(),
            },
        }
    }

    /// The reason recorded on the agents the stop cancels.
    pub fn agent_reason(&self) -> String {
        match self {
            Self::Requested => "Workflow cancelled.".to_string(),
            Self::AbortedByHook { hook_point, reason } => {
                format!("Workflow aborted by {hook_point} hook: {reason}")
            }
            Self::Stopped { kind, reason } => {
                let reason = reason.trim().trim_end_matches('.');
                if reason.is_empty() {
                    format!("Workflow stopped ({kind}).")
                } else {
                    format!("Workflow stopped ({kind}): {reason}.")
                }
            }
        }
    }
}

/// A stopped run: why, and when. Saved in checkpoints as
/// [`WorkflowState::cancellation`](crate::persistence::state::WorkflowState::cancellation).
///
/// A [`CancelCause::Stopped`] cause is saved as `"cause": "Requested"` with
/// its kind and reason in a `stop` field beside it, so that versions from
/// before graceful stops read it as a plain cancellation; this version reads
/// it back as `Stopped`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "CancellationRepr", into = "CancellationRepr")]
pub struct Cancellation {
    pub cause: CancelCause,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// The saved form of [`Cancellation`].
#[derive(Clone, Serialize, Deserialize)]
struct CancellationRepr {
    cause: CancelCause,
    timestamp: chrono::DateTime<chrono::Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<SavedStop>,
}

/// A [`CancelCause::Stopped`]'s fields, saved beside a `Requested` cause.
#[derive(Clone, Serialize, Deserialize)]
struct SavedStop {
    kind: StopKind,
    reason: String,
}

impl From<Cancellation> for CancellationRepr {
    fn from(cancellation: Cancellation) -> Self {
        match cancellation.cause {
            CancelCause::Stopped { kind, reason } => Self {
                cause: CancelCause::Requested,
                timestamp: cancellation.timestamp,
                stop: Some(SavedStop { kind, reason }),
            },
            cause => Self {
                cause,
                timestamp: cancellation.timestamp,
                stop: None,
            },
        }
    }
}

impl From<CancellationRepr> for Cancellation {
    fn from(saved: CancellationRepr) -> Self {
        let cause = match (saved.cause, saved.stop) {
            (CancelCause::Requested, Some(SavedStop { kind, reason })) => {
                CancelCause::Stopped { kind, reason }
            }
            (cause, _) => cause,
        };
        Self {
            cause,
            timestamp: saved.timestamp,
        }
    }
}

/// A graceful stop in its wrap-up.
struct WrapUpState {
    /// The stop made when the wrap-up ends.
    cause: CancelCause,
    /// What running agents are told.
    notice: String,
}

/// A stop of one agent run ([`WorkflowCancelHandle::stop_run`]).
struct RunStop {
    /// Tells this stop from a later one of the same run (a continuation):
    /// its wrap-up's timer cancels only the stop it was started for.
    id: uuid::Uuid,
    /// What its result records (`intent` set).
    record: StopRecord,
    /// What the run is told; `None` for a stop without a wrap-up.
    notice: Option<String>,
    /// The run was given the notice: its completion reports progress.
    told: bool,
}

/// A running agent run, from its launch until it ends.
struct RunEntry {
    /// This launch of the run (see [`RunControl::begin_run`]): a run
    /// relaunched with the same task ID (a restart, a continuation) gets a
    /// new entry, which the old launch's late `unregister` leaves alone.
    launch: u64,
    /// The run inside whose tool call this one runs, if any: it takes over
    /// that run's own stop (see [`RunControl::begin_run`]).
    parent: Option<uuid::Uuid>,
    /// Its tools' stop signal
    /// ([`TaskContext::stop`](arko_agents::TaskContext::stop)): fired when
    /// the workflow starts stopping, and by the run's own stop.
    tools: arko_agents::StopSource,
    /// Set, with the reason, once the run's own stop cancels it: its driver
    /// winds it down within the grace period.
    cancel: watch::Sender<Option<String>>,
    /// A completion verifier's review, which is never stopped alone.
    review: bool,
    /// The run's own stop, once the host asked for one.
    stop: Option<RunStop>,
}

/// What [`RunControl::begin_run_stop`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunStopBegun {
    /// Nothing: the run isn't running (or is a review), the workflow is
    /// stopped, or the run is being stopped already and the request has a
    /// wrap-up too.
    Unchanged,
    /// The run is told to wrap up; the caller times it (stop `id`).
    WrapUp(uuid::Uuid),
    /// The run is being cancelled.
    Cancelled,
}

/// The stop state of one run, shared by the engine, tools and handles.
pub(crate) struct RunControl {
    state: watch::Sender<Option<Cancellation>>,
    /// Command channels of the run's agents, to tell them to stop at once.
    agents: parking_lot::Mutex<HashMap<uuid::Uuid, mpsc::WeakSender<UserCommand>>>,
    /// Per agent run, whether the workflow is stopping it: a tool that
    /// waits (`wait_for_message`) watches it, since the runtime reads the
    /// `CancelTask` it is sent only once the tool call returns.
    stopping: parking_lot::Mutex<HashMap<uuid::Uuid, watch::Sender<bool>>>,
    grace_period: Duration,
    /// When the run stopped, for the drivers' wind-down deadlines.
    stopped_at: parking_lot::Mutex<Option<tokio::time::Instant>>,
    stop_saved: AtomicBool,
    /// Fires the agents' tool stop signal
    /// ([`TaskContext::stop`](arko_agents::TaskContext::stop)) when the run
    /// starts stopping: at a wrap-up's start, or at the stop.
    tools_stop: arko_agents::StopSource,
    /// A graceful stop in its wrap-up; taken when the run stops. Locked
    /// before `state` is changed.
    wrap_up: parking_lot::Mutex<Option<WrapUpState>>,
    /// The stop the host asked for, once one began: a stop request's (from
    /// its wrap-up's start) or a cancellation's.
    host_stop: parking_lot::Mutex<Option<StopRecord>>,
    /// The agent runs given the wrap-up notice.
    told: parking_lot::Mutex<HashSet<uuid::Uuid>>,
    /// The running agent runs, with their own stops. Locked before
    /// `agents`, never after.
    runs: parking_lot::Mutex<HashMap<uuid::Uuid, RunEntry>>,
    /// The last launch number given (see [`RunEntry::launch`]).
    launches: std::sync::atomic::AtomicU64,
}

impl RunControl {
    pub fn new(grace_period: Duration) -> Self {
        Self {
            state: watch::Sender::new(None),
            agents: parking_lot::Mutex::new(HashMap::new()),
            stopping: parking_lot::Mutex::new(HashMap::new()),
            grace_period,
            stopped_at: parking_lot::Mutex::new(None),
            stop_saved: AtomicBool::new(false),
            tools_stop: arko_agents::StopSource::new(),
            wrap_up: parking_lot::Mutex::new(None),
            host_stop: parking_lot::Mutex::new(None),
            told: parking_lot::Mutex::new(HashSet::new()),
            runs: parking_lot::Mutex::new(HashMap::new()),
            launches: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Fires the tools' stop signal of the run and of every running agent
    /// run (the first reason stands).
    fn fire_tools_stop(&self, reason: String) {
        let runs = self.runs.lock();
        self.tools_stop.stop(reason.clone());
        for entry in runs.values() {
            entry.tools.stop(reason.clone());
        }
    }

    pub fn grace_period(&self) -> Duration {
        self.grace_period
    }

    /// Stops the run for `cause` and sends every registered agent
    /// `CancelTask`, unless the run is already stopped (the first cause
    /// stands). During a wrap-up the wrap-up's stop is made instead, which
    /// came first. Returns whether this call stopped the run.
    pub fn stop(&self, cause: CancelCause) -> bool {
        let mut wrap_up = self.wrap_up.lock();
        let cause = wrap_up.take().map_or(cause, |wrap_up| wrap_up.cause);
        let timestamp = chrono::Utc::now();
        let mut made = None;
        let stopped = self.state.send_if_modified(|state| {
            if state.is_some() {
                return false;
            }
            made = Some(cause.clone());
            *state = Some(Cancellation { cause, timestamp });
            true
        });
        drop(wrap_up);
        if let Some(cause) = made {
            {
                let mut host_stop = self.host_stop.lock();
                if host_stop.is_none() {
                    *host_stop = StopRecord::of_cause(&cause, timestamp);
                }
            }
            self.fire_tools_stop(cause.agent_reason());
            *self.stopped_at.lock() = Some(tokio::time::Instant::now());
            // Queued before any agent takes its next step; the driver's own
            // (awaited) CancelTask covers a full command channel.
            for sender in self.agents.lock().values() {
                if let Some(sender) = sender.upgrade() {
                    let _ = sender.try_send(UserCommand::CancelTask);
                }
            }
        }
        stopped
    }

    pub fn cancellation(&self) -> Option<Cancellation> {
        self.state.borrow().clone()
    }

    /// When the run stopped, once it has.
    pub fn stopped_at(&self) -> Option<tokio::time::Instant> {
        *self.stopped_at.lock()
    }

    pub fn is_stopped(&self) -> bool {
        self.state.borrow().is_some()
    }

    /// The stop's error, once the run is stopped.
    pub fn error(&self) -> Option<WorkflowError> {
        self.state
            .borrow()
            .as_ref()
            .map(|cancellation| cancellation.cause.to_error())
    }

    /// `Err` with the stop's error once the run is stopped.
    pub fn check(&self) -> Result<(), WorkflowError> {
        self.error().map_or(Ok(()), Err)
    }

    /// Resolves once the run is stopped.
    pub async fn stopped(&self) {
        let mut receiver = self.state.subscribe();
        let _ = receiver.wait_for(Option::is_some).await;
    }

    /// Registers a started agent's command channel; an agent started after
    /// the stop is sent `CancelTask` at once.
    pub fn register(&self, task_id: uuid::Uuid, sender: &mpsc::Sender<UserCommand>) {
        {
            let mut agents = self.agents.lock();
            agents.retain(|_, sender| sender.strong_count() > 0);
            agents.insert(task_id, sender.downgrade());
        }
        let cancelled = self
            .runs
            .lock()
            .get(&task_id)
            .is_some_and(|entry| entry.cancel.borrow().is_some());
        if self.is_stopped() || cancelled {
            let _ = sender.try_send(UserCommand::CancelTask);
        }
    }

    /// Unregisters launch `launch` of agent run `task_id` once it has
    /// ended. When the run was launched again meanwhile (a host may restart
    /// it as soon as its result is recorded), the entries are the new
    /// launch's, and are kept.
    pub fn unregister(&self, task_id: uuid::Uuid, launch: u64) {
        let mut runs = self.runs.lock();
        if runs
            .get(&task_id)
            .is_some_and(|entry| entry.launch != launch)
        {
            return;
        }
        runs.remove(&task_id);
        self.agents.lock().remove(&task_id);
        self.stopping.lock().remove(&task_id);
    }

    /// Records agent run `task_id` as running from its launch (`review`: a
    /// completion verifier's review; `parent`: the run inside whose tool
    /// call it runs), and returns the stop signal its tools see, and the
    /// launch's number for [`unregister`](Self::unregister). The signal
    /// fires when the workflow starts stopping (at once if it has), and when
    /// the run's own stop begins. A run started inside the tool call of a
    /// run being stopped alone (other than a review) is stopped with it: it
    /// takes over that stop, and is cancelled at once if that run is.
    pub fn begin_run(
        &self,
        task_id: uuid::Uuid,
        review: bool,
        parent: Option<uuid::Uuid>,
    ) -> (arko_agents::StopSignal, u64) {
        let launch = self
            .launches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let mut runs = self.runs.lock();
        let tools = arko_agents::StopSource::new();
        if let Some(reason) = self.tools_stop.signal().reason() {
            tools.stop(reason);
        }
        let inherited = parent
            .filter(|_| !review)
            .and_then(|parent| runs.get(&parent))
            .and_then(|entry| {
                let stop = entry.stop.as_ref()?;
                Some((
                    RunStop {
                        id: uuid::Uuid::new_v4(),
                        record: stop.record.clone(),
                        notice: stop.notice.clone(),
                        told: false,
                    },
                    entry.cancel.borrow().clone(),
                ))
            });
        let (stop, cancel) = match inherited {
            Some((stop, cancel)) => {
                tools.stop(stop.record.run_reason());
                (Some(stop), cancel)
            }
            None => (None, None),
        };
        let signal = tools.signal();
        runs.insert(
            task_id,
            RunEntry {
                launch,
                parent,
                tools,
                cancel: watch::Sender::new(cancel),
                review,
                stop,
            },
        );
        (signal, launch)
    }

    /// Begins a stop of agent run `task_id` alone, recorded as `record`:
    /// with a wrap-up (`notice`), the run's tools see their stop signal and
    /// the run is told at its next turn boundary; without one, it is
    /// cancelled at once. A request without a wrap-up during the run's
    /// wrap-up cancels it at once (the first stop's record stands); one with
    /// a wrap-up changes nothing.
    pub fn begin_run_stop(
        &self,
        task_id: uuid::Uuid,
        record: StopRecord,
        notice: Option<String>,
    ) -> RunStopBegun {
        if self.is_stopped() {
            return RunStopBegun::Unchanged;
        }
        let mut runs = self.runs.lock();
        let Some(entry) = runs.get_mut(&task_id).filter(|entry| !entry.review) else {
            return RunStopBegun::Unchanged;
        };
        if entry.stop.is_some() {
            drop(runs);
            return if notice.is_none() && self.cancel_run(task_id, None) {
                RunStopBegun::Cancelled
            } else {
                RunStopBegun::Unchanged
            };
        }
        let id = uuid::Uuid::new_v4();
        let wraps_up = notice.is_some();
        entry.tools.stop(record.run_reason());
        entry.stop = Some(RunStop {
            id,
            record,
            notice,
            told: false,
        });
        drop(runs);
        if wraps_up {
            RunStopBegun::WrapUp(id)
        } else {
            self.cancel_run(task_id, Some(id));
            RunStopBegun::Cancelled
        }
    }

    /// Cancels agent run `task_id` for its own stop (stop `id`, when given:
    /// a wrap-up's timer cancels only the stop it was started for), unless
    /// it is being cancelled already. Its driver winds it down within the
    /// grace period; a run driven through `into_parts` is sent `CancelTask`.
    /// Returns whether this call cancelled it.
    ///
    /// The runs inside its tool calls that are being stopped alone (with it)
    /// are cancelled with it.
    pub fn cancel_run(&self, task_id: uuid::Uuid, id: Option<uuid::Uuid>) -> bool {
        let children = {
            let runs = self.runs.lock();
            let Some(entry) = runs.get(&task_id) else {
                return false;
            };
            let Some(stop) = entry
                .stop
                .as_ref()
                .filter(|stop| id.is_none_or(|id| stop.id == id))
            else {
                return false;
            };
            let reason = stop.record.run_reason();
            let cancelled = entry.cancel.send_if_modified(|cancel| {
                if cancel.is_some() {
                    return false;
                }
                *cancel = Some(reason);
                true
            });
            if !cancelled {
                return false;
            }
            if let Some(sender) = self
                .agents
                .lock()
                .get(&task_id)
                .and_then(mpsc::WeakSender::upgrade)
            {
                let _ = sender.try_send(UserCommand::CancelTask);
            }
            runs.iter()
                .filter(|(_, child)| child.parent == Some(task_id))
                .map(|(child, _)| *child)
                .collect::<Vec<_>>()
        };
        for child in children {
            self.cancel_run(child, None);
        }
        true
    }

    /// Set, with the reason, once agent run `task_id`'s own stop cancels
    /// it; `None` for a run that isn't running.
    pub fn run_cancel(&self, task_id: uuid::Uuid) -> Option<watch::Receiver<Option<String>>> {
        self.runs
            .lock()
            .get(&task_id)
            .map(|entry| entry.cancel.subscribe())
    }

    /// The reason agent run `task_id`'s own stop cancels it, once it does.
    pub fn run_cancel_reason(&self, task_id: uuid::Uuid) -> Option<String> {
        self.runs
            .lock()
            .get(&task_id)
            .and_then(|entry| entry.cancel.borrow().clone())
    }

    /// The record of agent run `task_id`'s own stop, while the run runs.
    pub fn run_stop_record(&self, task_id: uuid::Uuid) -> Option<StopRecord> {
        self.runs
            .lock()
            .get(&task_id)
            .and_then(|entry| entry.stop.as_ref())
            .map(|stop| stop.record.clone())
    }

    /// Whether agent run `task_id` is being stopped alone.
    pub fn is_run_stopping(&self, task_id: uuid::Uuid) -> bool {
        self.runs
            .lock()
            .get(&task_id)
            .is_some_and(|entry| entry.stop.is_some())
    }

    /// The signal that fires when a wrap-up begins, or when the run stops:
    /// each agent run's tools see it through their own
    /// ([`begin_run`](Self::begin_run)).
    #[cfg(test)]
    pub fn tools_signal(&self) -> arko_agents::StopSignal {
        self.tools_stop.signal()
    }

    /// Begins a graceful stop's wrap-up: `cause` is the stop made when it
    /// ends, `notice` what running agents are told. Fires the tools' stop
    /// signal. Returns `false`, changing nothing, when the run is stopped
    /// or wrapping up already.
    pub fn begin_wrap_up(&self, record: StopRecord, cause: CancelCause, notice: String) -> bool {
        let reason = cause.agent_reason();
        {
            let mut wrap_up = self.wrap_up.lock();
            if wrap_up.is_some() || self.is_stopped() {
                return false;
            }
            *wrap_up = Some(WrapUpState { cause, notice });
            *self.host_stop.lock() = Some(record);
        }
        self.fire_tools_stop(reason);
        true
    }

    /// Whether a graceful stop's wrap-up is under way (the run is not
    /// stopped yet).
    pub fn is_wrapping_up(&self) -> bool {
        self.wrap_up.lock().is_some()
    }

    /// Ends a wrap-up under way: the run is stopped with its stop. Returns
    /// whether this call stopped the run.
    pub fn end_wrap_up(&self) -> bool {
        let cause = match &*self.wrap_up.lock() {
            Some(wrap_up) => wrap_up.cause.clone(),
            None => return false,
        };
        self.stop(cause)
    }

    /// Ends a wrap-up under way once no agent of the run is running: there
    /// is nothing left to wrap up. Returns whether this call stopped the
    /// run.
    pub fn end_wrap_up_if_idle(&self) -> bool {
        if !self.is_wrapping_up() {
            return false;
        }
        let running = self
            .agents
            .lock()
            .values()
            .any(|sender| sender.strong_count() > 0);
        !running && self.end_wrap_up()
    }

    /// `Err` with the stop's error once the run is stopped, and with
    /// `Cancelled` during a wrap-up unless the start is a completion
    /// verifier's `review` (which may still start then).
    pub fn check_admission(&self, review: bool) -> Result<(), WorkflowError> {
        self.check()?;
        if !review && self.is_wrapping_up() {
            return Err(WorkflowError::Cancelled);
        }
        Ok(())
    }

    /// The wrap-up notices for agent run `task_id` not given yet, each once:
    /// its own stop's, then the workflow's. `None` when there are none.
    pub fn wrap_up_notice(&self, task_id: uuid::Uuid) -> Option<String> {
        let mut notices = Vec::new();
        if let Some(stop) = self
            .runs
            .lock()
            .get_mut(&task_id)
            .and_then(|entry| entry.stop.as_mut())
            .filter(|stop| !stop.told)
            && let Some(notice) = &stop.notice
        {
            stop.told = true;
            notices.push(notice.clone());
        }
        {
            let wrap_up = self.wrap_up.lock();
            if let Some(wrap_up) = wrap_up.as_ref()
                && self.told.lock().insert(task_id)
            {
                notices.push(wrap_up.notice.clone());
            }
        }
        (!notices.is_empty()).then(|| notices.join("\n"))
    }

    /// Whether agent run `task_id` was given a wrap-up notice, the
    /// workflow's or its own stop's: its completion is a progress report,
    /// which completion verifiers don't review.
    pub fn was_told(&self, task_id: uuid::Uuid) -> bool {
        self.told.lock().contains(&task_id) || self.told_to_stop_alone(task_id)
    }

    fn told_to_stop_alone(&self, task_id: uuid::Uuid) -> bool {
        self.runs
            .lock()
            .get(&task_id)
            .and_then(|entry| entry.stop.as_ref())
            .is_some_and(|stop| stop.told)
    }

    /// The stop that ended agent run `task_id` with `status`, if a stop the
    /// host asked for did: the run's own stop, when it has one and the run
    /// was cancelled, interrupted, or completed after a wrap-up notice;
    /// otherwise the workflow's, for a run cancelled once the run is
    /// stopped (by a stop request or a cancellation), or a run that
    /// completed after it was given the wrap-up notice.
    pub fn stopped_by(
        &self,
        task_id: uuid::Uuid,
        status: &crate::workflow::result::AgentRunStatus,
    ) -> Option<StopRecord> {
        use crate::workflow::result::AgentRunStatus;
        if let Some(record) = self.run_stop_record(task_id) {
            let applies = match status {
                AgentRunStatus::Cancelled | AgentRunStatus::Interrupted => true,
                AgentRunStatus::Completed => self.was_told(task_id),
                _ => false,
            };
            if applies {
                return Some(record);
            }
        }
        let applies = match status {
            AgentRunStatus::Cancelled => self.is_stopped(),
            AgentRunStatus::Completed => self.was_told(task_id),
            _ => false,
        };
        if applies {
            self.host_stop.lock().clone()
        } else {
            None
        }
    }

    /// Marks agent run `task_id` as being stopped by the workflow (the run
    /// may go on): its waits end.
    pub fn stop_agent(&self, task_id: uuid::Uuid) {
        self.stopping
            .lock()
            .entry(task_id)
            .or_insert_with(|| watch::Sender::new(false))
            .send_replace(true);
    }

    /// Whether agent run `task_id` is being stopped, now and as it changes.
    pub fn agent_stopping(&self, task_id: uuid::Uuid) -> watch::Receiver<bool> {
        self.stopping
            .lock()
            .entry(task_id)
            .or_insert_with(|| watch::Sender::new(false))
            .subscribe()
    }

    /// True once, for the first caller after the run stopped: the caller
    /// saves the checkpoint recording the stop.
    pub fn claim_stop_save(&self) -> bool {
        self.is_stopped() && !self.stop_saved.swap(true, Ordering::AcqRel)
    }
}

/// Cancels or stops a workflow run from any task or thread.
///
/// Obtained from [`Workflow::cancel_handle`](crate::Workflow::cancel_handle);
/// clones cancel the same run. [`cancel`](Self::cancel) tells every running
/// agent of the run to stop (`CancelTask`) and makes run calls in progress,
/// and every later one, return [`WorkflowError::Cancelled`]. Each call in
/// progress returns once its agents have stopped, or have been aborted after
/// the run's grace period. [`stop`](Self::stop) does the same for a
/// [`StopRequest`], after its wrap-up if it has one; see
/// [`stop_on_signals`](crate::workflow::stop::stop_on_signals) for Ctrl-C
/// and `SIGTERM`.
#[derive(Clone)]
pub struct WorkflowCancelHandle {
    control: Arc<RunControl>,
    context: Weak<SharedContext>,
    /// The runtime the handle was made in, to time a wrap-up when
    /// [`stop`](Self::stop) is called from outside one.
    runtime: Option<tokio::runtime::Handle>,
}

impl WorkflowCancelHandle {
    pub(crate) fn new(context: &Arc<SharedContext>) -> Self {
        Self {
            control: context.run_control.clone(),
            context: Arc::downgrade(context),
            runtime: tokio::runtime::Handle::try_current().ok(),
        }
    }

    /// Cancels the run and fires
    /// [`HookPoint::OnWorkflowCancelled`](crate::hooks::HookPoint::OnWorkflowCancelled)
    /// (its result is ignored). Returns `false`, changing nothing, when the
    /// run was already cancelled or aborted by a hook. During a graceful
    /// stop's wrap-up it ends the wrap-up at once, and the stop's cause
    /// stands.
    pub fn cancel(&self) -> bool {
        if !self.control.stop(CancelCause::Requested) {
            return false;
        }
        self.fire_stopped_hook();
        true
    }

    /// Stops the run gracefully; see [`Workflow::stop`](crate::Workflow::stop).
    pub fn stop(&self, request: StopRequest) -> bool {
        let StopRequest {
            kind,
            reason,
            wrap_up,
        } = request;
        let cause = CancelCause::Stopped {
            kind,
            reason: reason.clone(),
        };
        let Some(wrap_up) = wrap_up else {
            // Ends a wrap-up under way at once; its stop stands.
            if !self.control.stop(cause) {
                return false;
            }
            self.fire_stopped_hook();
            return true;
        };
        let record = StopRecord {
            kind,
            reason: reason.clone(),
            timestamp: chrono::Utc::now(),
            intent: None,
        };
        let notice = wrap_up
            .message
            .unwrap_or_else(|| crate::workflow::stop::wrap_up_notice(&reason, wrap_up.within));
        if !self.control.begin_wrap_up(record, cause, notice) {
            return false;
        }
        // Nothing runs: nothing to wrap up.
        if self.control.end_wrap_up_if_idle() {
            self.fire_stopped_hook();
            return true;
        }
        let runtime = tokio::runtime::Handle::try_current()
            .ok()
            .or_else(|| self.runtime.clone());
        let Some(runtime) = runtime else {
            log::warn!(
                "Workflow stop: no Tokio runtime to time the wrap-up, so the run is stopped at once."
            );
            if self.control.end_wrap_up() {
                self.fire_stopped_hook();
            }
            return true;
        };
        let start = tokio::time::Instant::now();
        // A wrap-up too long to represent lasts as long as possible.
        let deadline = start
            .checked_add(wrap_up.within)
            .unwrap_or_else(|| start + Duration::from_secs(30 * 365 * 86_400));
        let handle = self.clone();
        runtime.spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {}
                _ = handle.control.stopped() => return,
            }
            if handle.control.end_wrap_up() {
                handle.fire_stopped_hook();
            }
        });
        true
    }

    /// Stops agent run `agent_run_id` alone, with the runs inside its tool
    /// calls; see [`Workflow::stop_run`](crate::Workflow::stop_run).
    pub fn stop_run(
        &self,
        agent_run_id: uuid::Uuid,
        request: StopRequest,
        intent: RunStopIntent,
    ) -> bool {
        let Some(context) = self.context.upgrade() else {
            return false;
        };
        self.stop_one(&context, agent_run_id, &request, intent)
    }

    /// Stops every running run of `agent_name` alone; see
    /// [`Workflow::stop_agent`](crate::Workflow::stop_agent).
    pub fn stop_agent(
        &self,
        agent_name: &str,
        request: StopRequest,
        intent: RunStopIntent,
    ) -> Result<Vec<uuid::Uuid>, WorkflowError> {
        let context = self
            .context
            .upgrade()
            .ok_or_else(|| WorkflowError::BuilderInvalid("the workflow was dropped".to_string()))?;
        if !context.agent_configs.contains_key(agent_name) {
            return Err(WorkflowError::AgentNotFound(agent_name.to_string()));
        }
        let runs = context.registry.runs_of(agent_name);
        Ok(runs
            .into_iter()
            .filter(|task_id| self.stop_one(&context, *task_id, &request, intent))
            .collect())
    }

    /// Stops run `root` alone, then the runs inside its tool calls (a
    /// delegated agent's, deeper ones after): returns whether `root`'s stop
    /// began (or, without a wrap-up, ended its wrap-up).
    fn stop_one(
        &self,
        context: &Arc<SharedContext>,
        root: uuid::Uuid,
        request: &StopRequest,
        intent: RunStopIntent,
    ) -> bool {
        let record = StopRecord {
            kind: request.kind,
            reason: request.reason.clone(),
            timestamp: chrono::Utc::now(),
            intent: Some(intent),
        };
        let notice = request.wrap_up.as_ref().map(|wrap_up| {
            wrap_up.message.clone().unwrap_or_else(|| {
                crate::workflow::stop::run_wrap_up_notice(&request.reason, wrap_up.within)
            })
        });
        let mut runs = vec![root];
        let mut next = 0;
        while next < runs.len() {
            runs.extend(context.registry.children(runs[next]));
            next += 1;
        }
        for (index, task_id) in runs.into_iter().enumerate() {
            match self
                .control
                .begin_run_stop(task_id, record.clone(), notice.clone())
            {
                RunStopBegun::Unchanged if index == 0 => return false,
                RunStopBegun::Unchanged => {}
                RunStopBegun::WrapUp(id) => {
                    let within = request
                        .wrap_up
                        .as_ref()
                        .map_or(Duration::ZERO, |w| w.within);
                    self.time_run_wrap_up(task_id, id, within);
                }
                RunStopBegun::Cancelled => {}
            }
            if index == 0 {
                log::info!(
                    "Workflow '{}': stopping agent run {task_id} alone ({}, {intent}){}: {}",
                    context.workflow_name,
                    request.kind,
                    request
                        .wrap_up
                        .as_ref()
                        .map_or_else(String::new, |wrap_up| format!(
                            " within {:?}",
                            wrap_up.within
                        )),
                    request.reason
                );
            }
        }
        // Saved at once while agents run, so that a resume keeps the
        // host's intent for the run.
        context.save_live_checkpoint_soon("the stop of an agent run");
        true
    }

    /// Cancels run `task_id` for its stop `id` when the stop's wrap-up of
    /// `within` is over, unless the run ends (or is cancelled) first.
    fn time_run_wrap_up(&self, task_id: uuid::Uuid, id: uuid::Uuid, within: Duration) {
        // Ended already: nothing to time.
        let Some(mut cancel) = self.control.run_cancel(task_id) else {
            return;
        };
        let runtime = tokio::runtime::Handle::try_current()
            .ok()
            .or_else(|| self.runtime.clone());
        let Some(runtime) = runtime else {
            log::warn!(
                "Agent run stop: no Tokio runtime to time the wrap-up, so run {task_id} is stopped at once."
            );
            self.control.cancel_run(task_id, Some(id));
            return;
        };
        // Held weakly: once the workflow is dropped, the run's entry and
        // with it the sender are gone, and the timer ends.
        let control = Arc::downgrade(&self.control);
        runtime.spawn(async move {
            let start = tokio::time::Instant::now();
            // A wrap-up too long to represent lasts as long as possible.
            let deadline = start
                .checked_add(within)
                .unwrap_or_else(|| start + Duration::from_secs(30 * 365 * 86_400));
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    if let Some(control) = control.upgrade() {
                        control.cancel_run(task_id, Some(id));
                    }
                }
                // Cancelled sooner, or ended (its entry, and the sender, are gone).
                _ = cancel.changed() => {}
            }
        });
    }

    /// Whether a graceful stop's wrap-up is under way: the run is stopping,
    /// but not stopped yet.
    pub fn is_stopping(&self) -> bool {
        self.control.is_wrapping_up()
    }

    /// Fires `OnWorkflowCancelled` for the run's stop.
    fn fire_stopped_hook(&self) {
        if let Some(context) = self.context.upgrade() {
            crate::workflow::engine::fire_stop_hook(&context);
        }
    }

    /// Whether the run is stopped, by cancellation, a stop or a hook's abort
    /// (`false` during a graceful stop's wrap-up; see [`is_stopping`](Self::is_stopping)).
    pub fn is_cancelled(&self) -> bool {
        self.control.is_stopped()
    }

    /// Why and when the run stopped, once it has.
    pub fn cancellation(&self) -> Option<Cancellation> {
        self.control.cancellation()
    }

    /// Resolves once the run is stopped, for example to stop work of the
    /// application's own running alongside it.
    pub async fn cancelled(&self) {
        self.control.stopped().await
    }
}

impl std::fmt::Debug for WorkflowCancelHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkflowCancelHandle")
            .field("cancellation", &self.control.cancellation())
            .finish()
    }
}

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod tests;
