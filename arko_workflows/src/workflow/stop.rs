//! Stopping a workflow run gracefully, and restarting what the stop ended.
//!
//! [`Workflow::stop`](crate::Workflow::stop) stops the run with a
//! [`StopRequest`]: why ([`StopKind`] and a reason) and, optionally, a
//! [wrap-up](StopRequest::wrap_up) in which running agents are told to save
//! their progress and finish before they are cancelled. The stop is built on
//! the run's cancellation (see [`WorkflowCancelHandle`]): after the wrap-up
//! the run is cancelled with [`CancelCause::Stopped`], its agents are
//! stopped within the grace period, and the stop is checkpointed.
//!
//! Runs a stop ended are marked with [`AgentResult::stopped_by`], so a host
//! doesn't count them as failures. After resuming from the checkpoint,
//! [`Workflow::stopped_runs`](crate::Workflow::stopped_runs) lists them and
//! [`Workflow::restart`](crate::Workflow::restart) relaunches each one.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::workflow::cancel::{CancelCause, WorkflowCancelHandle};
use crate::workflow::result::AgentResult;

/// Why a run is being stopped, for the host's own handling (a stop is never
/// a failure of the agents it ends). Recorded in [`CancelCause::Stopped`]
/// and [`StopRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopKind {
    /// Someone asked for it: Ctrl-C, `SIGTERM`, a command from a person.
    Operator,
    /// The process is going away and the run will be resumed: a redeploy,
    /// a host shutting down, a move to another machine.
    Drain,
    /// A limit of the host's own was reached: a deadline, a budget, a
    /// number of jobs.
    Limit,
    /// The host stops the run because something failed (a lost login, a
    /// broken dependency), not because of the work.
    Failure,
}

impl StopKind {
    /// The kind's name, as saved: `operator`, `drain`, `limit` or
    /// `failure`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Operator => "operator",
            Self::Drain => "drain",
            Self::Limit => "limit",
            Self::Failure => "failure",
        }
    }
}

impl std::fmt::Display for StopKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The opt-in wrap-up of a [`StopRequest`]: how long agents get to save
/// their progress and finish before they are cancelled, and what they are
/// told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrapUp {
    /// The wrap-up's length, counted from the stop.
    pub within: Duration,
    /// The notice agents get, replacing [`wrap_up_notice`]'s.
    pub message: Option<String>,
}

/// A request to stop the run; see [`Workflow::stop`](crate::Workflow::stop).
///
/// ```
/// use arko_workflows::workflow::stop::StopRequest;
/// use std::time::Duration;
///
/// let request = StopRequest::drain("deploying a fix").wrap_up(Duration::from_secs(90));
/// assert_eq!(request.reason, "deploying a fix");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopRequest {
    pub kind: StopKind,
    /// Why, in words: shown to agents in their stop notices and recorded
    /// with the stop and on the results it ends.
    pub reason: String,
    /// `None` (the default): the run is cancelled at once, as
    /// [`cancel`](crate::Workflow::cancel) does. `Some`: a wrap-up first;
    /// see [`wrap_up`](Self::wrap_up).
    pub wrap_up: Option<WrapUp>,
}

impl StopRequest {
    /// A stop of `kind` for `reason`, without a wrap-up.
    pub fn new(kind: StopKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
            wrap_up: None,
        }
    }

    /// [`StopKind::Operator`].
    pub fn operator(reason: impl Into<String>) -> Self {
        Self::new(StopKind::Operator, reason)
    }

    /// [`StopKind::Drain`]: the run is stopped to be resumed (a redeploy).
    pub fn drain(reason: impl Into<String>) -> Self {
        Self::new(StopKind::Drain, reason)
    }

    /// [`StopKind::Limit`].
    pub fn limit(reason: impl Into<String>) -> Self {
        Self::new(StopKind::Limit, reason)
    }

    /// [`StopKind::Failure`].
    pub fn failure(reason: impl Into<String>) -> Self {
        Self::new(StopKind::Failure, reason)
    }

    /// Adds a wrap-up of `within`: no new agent run starts, every running
    /// agent's tools see their stop signal at once, and each agent is told
    /// at its next turn boundary to save its progress and call
    /// `complete_task` ([`wrap_up_notice`]). The wrap-up ends when no agent
    /// runs any more, or after `within`; then the run is cancelled as
    /// without a wrap-up.
    pub fn wrap_up(mut self, within: Duration) -> Self {
        // A message set before is kept.
        let message = self.wrap_up.take().and_then(|wrap_up| wrap_up.message);
        self.wrap_up = Some(WrapUp { within, message });
        self
    }

    /// The wrap-up notice agents get, replacing [`wrap_up_notice`]'s. Call
    /// it after [`wrap_up`](Self::wrap_up); without a wrap-up it changes
    /// nothing.
    pub fn wrap_up_message(mut self, message: impl Into<String>) -> Self {
        if let Some(wrap_up) = &mut self.wrap_up {
            wrap_up.message = Some(message.into());
        }
        self
    }
}

/// The stop that ended an agent run; see [`AgentResult::stopped_by`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopRecord {
    pub kind: StopKind,
    pub reason: String,
    /// When the stop began (the start of its wrap-up, if it had one).
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl StopRecord {
    /// The record of `cause`, a stop the host asked for: a
    /// [`Workflow::stop`](crate::Workflow::stop), or a
    /// [`cancel`](crate::Workflow::cancel) (recorded as an operator's stop
    /// with the reason `Workflow cancelled.`). `None` for a hook's abort.
    pub(crate) fn of_cause(
        cause: &CancelCause,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Option<Self> {
        match cause {
            CancelCause::Stopped { kind, reason } => Some(Self {
                kind: *kind,
                reason: reason.clone(),
                timestamp,
            }),
            CancelCause::Requested => Some(Self {
                kind: StopKind::Operator,
                reason: cause.agent_reason(),
                timestamp,
            }),
            CancelCause::AbortedByHook { .. } => None,
        }
    }
}

/// The default notice an agent gets when a wrap-up begins (at its next turn
/// boundary, as a workflow context update):
///
/// > The workflow is stopping: deploying a fix. You have about 90 seconds
/// > before you are stopped. Save your progress where your instructions say
/// > to keep it, and don't start new long work. Then call complete_task with
/// > a summary of where you stopped and what is left, so the work can be
/// > picked up from there.
pub fn wrap_up_notice(reason: &str, within: Duration) -> String {
    let reason = reason.trim().trim_end_matches('.');
    let why = if reason.is_empty() {
        String::new()
    } else {
        format!(": {reason}")
    };
    format!(
        "The workflow is stopping{why}. You have about {} before you are stopped. Save your progress where your instructions say to keep it, and don't start new long work. Then call complete_task with a summary of where you stopped and what is left, so the work can be picked up from there.",
        describe_seconds(within)
    )
}

fn describe_seconds(duration: Duration) -> String {
    match duration.as_secs() {
        1 => "1 second".to_string(),
        seconds => format!("{seconds} seconds"),
    }
}

/// How [`Workflow::restart`](crate::Workflow::restart) relaunches a
/// [`StoppedRun`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restart {
    /// The run was running when the process ended (its status is
    /// `Interrupted`): [`recover_agent`](crate::Workflow::recover_agent)
    /// continues it where its task was saved, without a message.
    Recover,
    /// The stop cancelled the run:
    /// [`reopen_agent`](crate::Workflow::reopen_agent) continues it, with
    /// the restart's note.
    Reopen,
    /// The run completed in the stop's wrap-up:
    /// [`continue_agent`](crate::Workflow::continue_agent) continues it,
    /// telling it it was restarted, with the restart's note.
    Continue,
}

/// A run to relaunch after resuming a stopped workflow; see
/// [`Workflow::stopped_runs`](crate::Workflow::stopped_runs).
#[derive(Debug, Clone)]
pub struct StoppedRun {
    pub result: AgentResult,
    pub restart: Restart,
}

/// The message a run that completed in a stop's wrap-up gets when it is
/// continued, followed by `note`.
pub(crate) fn continued_message(result: &AgentResult, note: &str) -> String {
    let stop = result
        .stopped_by()
        .map(|stop| {
            let reason = stop.reason.trim().trim_end_matches('.');
            if reason.is_empty() {
                String::new()
            } else {
                format!(" ({reason})")
            }
        })
        .unwrap_or_default();
    let message = format!(
        "The workflow was stopped{stop} while you worked, so you finished early, and it has now restarted. Continue your task from where you stopped, and call complete_task when it is done."
    );
    let note = note.trim();
    if note.is_empty() {
        message
    } else {
        format!("{message}\n\n{note}")
    }
}

/// Stops the run when the process gets Ctrl-C or `SIGTERM`.
///
/// The first signal stops the run with `request`. The second ends a
/// wrap-up in progress at once (the run is cancelled; its agents still get
/// the grace period). A third kills the children of running
/// [`StopAwareCommand`](arko_agents::StopAwareCommand)s (on Unix;
/// [`kill_stop_aware_commands`](arko_agents::kill_stop_aware_commands))
/// and exits the process at once with status `128 + the signal's number`
/// (130 for Ctrl-C), as an unhandled signal would end it. Handles Ctrl-C (`SIGINT`) and `SIGTERM` on Unix, and
/// Ctrl-C and Ctrl-Break on Windows.
///
/// Call it within a Tokio runtime, before the signals can arrive: the
/// handlers are installed before it returns. Once installed they replace
/// the default action for the rest of the process. Abort the returned task
/// to stop listening (the signals are then ignored).
///
/// Under systemd, give the unit `TimeoutStopSec` longer than the wrap-up
/// plus the cancel grace period plus the longest tool stop grace, and
/// `KillMode=mixed`, so the main process gets `SIGTERM` and its children
/// are left to it until the timeout.
pub fn stop_on_signals(
    handle: WorkflowCancelHandle,
    request: StopRequest,
) -> std::io::Result<tokio::task::JoinHandle<()>> {
    let mut signals = Signals::install()?;
    Ok(tokio::spawn(async move {
        let mut received = 0;
        while let Some(number) = signals.next().await {
            received += 1;
            match received {
                1 => {
                    log::warn!(
                        "Signal {number}: stopping the workflow run ({}: {}).",
                        request.kind,
                        request.reason
                    );
                    handle.stop(request.clone());
                }
                2 => {
                    log::warn!("Signal {number} again: stopping the workflow run now.");
                    handle.stop(StopRequest {
                        wrap_up: None,
                        ..request.clone()
                    });
                }
                _ => {
                    log::warn!("Signal {number} a third time: exiting.");
                    // Exiting runs no destructors: the commands' children,
                    // in their own process groups, would outlive it.
                    arko_agents::kill_stop_aware_commands();
                    std::process::exit(128 + number);
                }
            }
        }
    }))
}

/// The signals [`stop_on_signals`] handles; yields each one's number.
struct Signals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
    #[cfg(windows)]
    ctrl_c: tokio::signal::windows::CtrlC,
    #[cfg(windows)]
    ctrl_break: tokio::signal::windows::CtrlBreak,
}

impl Signals {
    fn install() -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            Ok(Self {
                interrupt: signal(SignalKind::interrupt())?,
                terminate: signal(SignalKind::terminate())?,
            })
        }
        #[cfg(windows)]
        {
            Ok(Self {
                ctrl_c: tokio::signal::windows::ctrl_c()?,
                ctrl_break: tokio::signal::windows::ctrl_break()?,
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            Ok(Self {})
        }
    }

    /// The next signal's number; `None` once no more can arrive.
    async fn next(&mut self) -> Option<i32> {
        #[cfg(unix)]
        {
            tokio::select! {
                received = self.interrupt.recv() => received.map(|()| 2),
                received = self.terminate.recv() => received.map(|()| 15),
            }
        }
        #[cfg(windows)]
        {
            tokio::select! {
                received = self.ctrl_c.recv() => received.map(|()| 2),
                received = self.ctrl_break.recv() => received.map(|()| 21),
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            tokio::signal::ctrl_c().await.ok().map(|()| 2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_notices_read_as_documented() {
        let request = StopRequest::drain("deploying a fix.")
            .wrap_up(Duration::from_secs(60))
            .wrap_up_message("Save and finish.")
            .wrap_up(Duration::from_secs(90));
        assert_eq!(request.kind, StopKind::Drain);
        assert_eq!(
            request.wrap_up,
            Some(WrapUp {
                within: Duration::from_secs(90),
                message: Some("Save and finish.".into())
            })
        );
        assert_eq!(
            StopRequest::operator("x")
                .wrap_up_message("ignored")
                .wrap_up,
            None
        );
        assert_eq!(
            wrap_up_notice("deploying a fix.", Duration::from_secs(90)),
            "The workflow is stopping: deploying a fix. You have about 90 seconds before you are stopped. Save your progress where your instructions say to keep it, and don't start new long work. Then call complete_task with a summary of where you stopped and what is left, so the work can be picked up from there."
        );
        assert!(wrap_up_notice("", Duration::from_secs(1)).starts_with(
            "The workflow is stopping. You have about 1 second before you are stopped."
        ));
        assert_eq!(serde_json::to_value(StopKind::Drain).unwrap(), "drain");
    }
}
