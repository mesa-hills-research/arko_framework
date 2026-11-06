//! A wait for the host's command, which both runtimes keep for each of
//! their waits (an approval, an answer, an error-recovery decision).

use std::time::Duration;

use tokio::time::Instant;

use super::HostDecisions;

/// One wait for a command, timed by the opt-in
/// `AgentConfig::user_command_timeout_seconds`. Its deadline is set once,
/// when the wait starts. Context injected during the wait and commands the
/// wait ignores don't move it, and a decision ends the wait (the next one
/// starts its own). A wait that hasn't started has no value, so only a
/// started one has a deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CommandWait {
    /// Waiting until this deadline (`None` means no timeout).
    Until(Option<Instant>),
    /// The host is deciding this wait itself ([`UserCommand::HostDeciding`](super::command::UserCommand::HostDeciding),
    /// or the run's [`HostDecisions`] when it started), so no deadline.
    HostDeciding,
}

impl CommandWait {
    /// A wait that starts now. One that starts while the host decides the
    /// run's waits is never timed. A timeout too long for the clock means
    /// no timeout.
    pub(super) fn start(timeout_seconds: Option<u64>, host: &HostDecisions) -> Self {
        if host.get() {
            return Self::HostDeciding;
        }
        Self::Until(
            timeout_seconds
                .and_then(|seconds| Instant::now().checked_add(Duration::from_secs(seconds))),
        )
    }

    /// The wait's deadline now. `None` while it isn't timed, and while the
    /// host decides the run's waits.
    pub(super) fn deadline(self, host: &HostDecisions) -> Option<Instant> {
        match self {
            Self::Until(deadline) if !host.get() => deadline,
            _ => None,
        }
    }
}
