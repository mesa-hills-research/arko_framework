//! Moving running tool calls to the background, and telling a running call
//! that context arrived for its task.
//!
//! A tool says how its calls move with [`Tool::background_mode`](super::Tool::background_mode).
//! A call that can move is listed while it runs in the foreground, and its
//! host can move it, which the runtime does through a
//! [`TerminalManager`](crate::terminal::TerminalManager) set with
//! [`AgentRuntime::with_background_jobs`](crate::runtime::AgentRuntime::with_background_jobs).
//! A [`Detach`](ToolBackground::Detach) call goes on as one of the manager's
//! jobs, and its result reaches the agent as a note when it ends. An
//! [`Own`](ToolBackground::Own) tool sees [`TaskContext::moved`](super::TaskContext::moved)
//! fire and moves itself.
//!
//! [`TaskContext::context_arrived`](super::TaskContext::context_arrived)
//! tells a call that waits (for other agents, for a message) that a note, a
//! terminal update or other context arrived for its task, so it can return
//! and let the model read it.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::watch;

/// How a running call of a tool can move to the background. See
/// [`Tool::background_mode`](super::Tool::background_mode).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolBackground {
    /// It can't, which is the default. The call is quick, or its result has
    /// to come in the turn.
    #[default]
    Never,
    /// The runtime moves it. The call goes on as a job of the host's
    /// terminal manager, the model gets the job's ID as the call's result,
    /// and the real result comes as a note when the job ends. For a call
    /// that just awaits its result (a remote request, a command), whose
    /// future can go on anywhere.
    Detach,
    /// The tool moves itself. It watches [`TaskContext::moved`](super::TaskContext::moved)
    /// and returns soon after it fires, with a result that says what
    /// happened (the job its command goes on in, a wait that ended).
    Own,
}

/// Whether a running call was moved to the background, for an
/// [`Own`](ToolBackground::Own) tool to watch. See
/// [`TaskContext::moved`](super::TaskContext::moved).
///
/// Clones observe the same move. [`never`](Self::never) (also the
/// `Default`) never fires. Once fired, it stays fired.
#[derive(Clone, Default)]
pub struct MoveSignal {
    source: Option<watch::Receiver<bool>>,
}

impl MoveSignal {
    /// A signal that never fires, for a [`TaskContext`](super::TaskContext)
    /// built outside a runtime, a call that can't move, and tests.
    pub fn never() -> Self {
        Self::default()
    }

    /// Whether the call was moved to the background.
    pub fn is_moved(&self) -> bool {
        self.source.as_ref().is_some_and(|source| *source.borrow())
    }

    /// Resolves once the call is moved (at once if it was). A signal that
    /// can no longer fire, such as [`never`](Self::never), never resolves.
    pub async fn moved(&self) {
        let Some(source) = &self.source else {
            return std::future::pending().await;
        };
        let mut source = source.clone();
        // Dropped without firing: this signal never fires.
        if source.wait_for(|moved| *moved).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

impl fmt::Debug for MoveSignal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoveSignal")
            .field("moved", &self.is_moved())
            .finish()
    }
}

/// Fires a [`MoveSignal`], for a host or a test that moves a call itself.
/// Clones fire the same signal.
#[derive(Clone)]
pub struct MoveSource(Arc<watch::Sender<bool>>);

impl MoveSource {
    /// A source that has not fired.
    pub fn new() -> Self {
        Self(Arc::new(watch::Sender::new(false)))
    }

    /// The signal this source fires.
    pub fn signal(&self) -> MoveSignal {
        MoveSignal {
            source: Some(self.0.subscribe()),
        }
    }

    /// Fires the signal. Returns `false`, changing nothing, when it had
    /// fired already.
    pub fn fire(&self) -> bool {
        self.0
            .send_if_modified(|moved| !std::mem::replace(moved, true))
    }

    /// Whether the signal has fired.
    pub fn is_moved(&self) -> bool {
        *self.0.borrow()
    }
}

impl Default for MoveSource {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MoveSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoveSource")
            .field("moved", &self.is_moved())
            .finish()
    }
}

/// Whether context arrived for a call's task that the model hasn't read.
/// The runtime makes the signal as the call starts. It counts the context
/// that arrives from then on, and any that arrived after the model's
/// request the call came from. See
/// [`TaskContext::context_arrived`](super::TaskContext::context_arrived).
///
/// Clones share the same starting point. [`never`](Self::never) (also the
/// `Default`) never fires.
#[derive(Clone, Default)]
pub struct ContextSignal {
    source: Option<watch::Receiver<u64>>,
    /// The source's count the signal counts from.
    start: u64,
}

impl ContextSignal {
    /// A signal that never fires, for a [`TaskContext`](super::TaskContext)
    /// built outside a runtime, and for tests.
    pub fn never() -> Self {
        Self::default()
    }

    /// Whether context arrived since the signal was made.
    pub fn has_arrived(&self) -> bool {
        self.source
            .as_ref()
            .is_some_and(|source| *source.borrow() != self.start)
    }

    /// Resolves once context arrives (at once if some arrived since the
    /// signal was made). A signal that can no longer fire, such as
    /// [`never`](Self::never), never resolves.
    pub async fn arrived(&self) {
        let Some(source) = &self.source else {
            return std::future::pending().await;
        };
        let mut source = source.clone();
        let start = self.start;
        if source.wait_for(|count| *count != start).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

impl fmt::Debug for ContextSignal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ContextSignal")
            .field("arrived", &self.has_arrived())
            .finish()
    }
}

/// Fires the [`ContextSignal`]s made from it each time context arrives,
/// for a host that runs tools itself, and for tests. Clones notify the same
/// signals.
#[derive(Clone)]
pub struct ContextSource {
    arrivals: Arc<watch::Sender<u64>>,
    /// The count of the context the model has been given, or that waits
    /// for a later request. A call that starts counts from it, so context
    /// taken after it fires the call's signal at once.
    caught_up: Arc<AtomicU64>,
}

impl ContextSource {
    /// A source that has seen no context.
    pub fn new() -> Self {
        Self {
            arrivals: Arc::new(watch::Sender::new(0)),
            caught_up: Arc::default(),
        }
    }

    /// A signal that fires when context arrives from now on.
    pub fn signal(&self) -> ContextSignal {
        let source = self.arrivals.subscribe();
        let start = *source.borrow();
        ContextSignal {
            source: Some(source),
            start,
        }
    }

    /// A signal for a call starting now. Context taken since the model's
    /// request (or Claude Code's latest hand-over) fires it at once, since
    /// the model made the call without reading it.
    pub(crate) fn call_signal(&self) -> ContextSignal {
        ContextSignal {
            source: Some(self.arrivals.subscribe()),
            start: self.caught_up.load(Ordering::Acquire),
        }
    }

    /// Context arrived. Fires every signal made before now.
    pub fn notify(&self) {
        self.arrivals
            .send_modify(|count| *count = count.wrapping_add(1));
    }

    /// The model is given all the context that arrived so far, or it waits
    /// for a later request, so it no longer fires a
    /// [`call_signal`](Self::call_signal).
    pub(crate) fn caught_up(&self) {
        self.caught_up
            .store(*self.arrivals.borrow(), Ordering::Release);
    }
}

impl Default for ContextSource {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ContextSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ContextSource")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn a_move_fires_once_and_stays_fired() {
        let source = MoveSource::new();
        let signal = source.signal();
        assert!(!signal.is_moved());
        let waiting = tokio::spawn({
            let signal = signal.clone();
            async move { signal.moved().await }
        });
        assert!(source.fire());
        assert!(!source.fire(), "a second move changes nothing");
        timeout(Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
        assert!(signal.is_moved() && source.is_moved());
        // A signal made later sees the move at once.
        timeout(Duration::from_secs(5), source.signal().moved())
            .await
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_signal_that_cannot_fire_never_resolves() {
        assert!(!MoveSignal::never().is_moved());
        assert!(
            timeout(Duration::from_secs(60), MoveSignal::never().moved())
                .await
                .is_err()
        );
        let dropped = MoveSource::new().signal();
        assert!(
            timeout(Duration::from_secs(60), dropped.moved())
                .await
                .is_err()
        );
        assert!(!ContextSignal::never().has_arrived());
        assert!(
            timeout(Duration::from_secs(60), ContextSignal::never().arrived())
                .await
                .is_err()
        );
        let dropped = ContextSource::new().signal();
        assert!(
            timeout(Duration::from_secs(60), dropped.arrived())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn context_counts_from_when_the_signal_was_made() {
        let source = ContextSource::new();
        source.notify();
        let signal = source.signal();
        assert!(!signal.has_arrived(), "earlier context doesn't count");
        let waiting = tokio::spawn({
            let signal = signal.clone();
            async move { signal.arrived().await }
        });
        source.notify();
        timeout(Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
        assert!(signal.has_arrived());
        // Once arrived, waiting again returns at once.
        timeout(Duration::from_secs(5), signal.arrived())
            .await
            .unwrap();
        assert!(!source.signal().has_arrived());
    }
}
