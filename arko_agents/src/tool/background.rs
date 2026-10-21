//! Telling a running call that context arrived for its task.
//!
//! [`TaskContext::context_arrived`](super::TaskContext::context_arrived)
//! tells a call that waits (for other agents, for a message) that a note, a
//! terminal update or other context arrived for its task, so it can return
//! and let the model read it.

use std::fmt;
use std::sync::Arc;

use tokio::sync::watch;

/// Whether context arrived for a call's task since the signal was made,
/// which the runtime does as the call starts. See
/// [`TaskContext::context_arrived`](super::TaskContext::context_arrived).
///
/// Clones share the same starting point. [`never`](Self::never) (also the
/// `Default`) never fires.
#[derive(Clone, Default)]
pub struct ContextSignal {
    source: Option<watch::Receiver<u64>>,
    /// The source's count when the signal was made.
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
pub struct ContextSource(Arc<watch::Sender<u64>>);

impl ContextSource {
    /// A source that has seen no context.
    pub fn new() -> Self {
        Self(Arc::new(watch::Sender::new(0)))
    }

    /// A signal that fires when context arrives from now on.
    pub fn signal(&self) -> ContextSignal {
        let source = self.0.subscribe();
        let start = *source.borrow();
        ContextSignal {
            source: Some(source),
            start,
        }
    }

    /// Context arrived. Fires every signal made before now.
    pub fn notify(&self) {
        self.0.send_modify(|count| *count = count.wrapping_add(1));
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

    #[tokio::test(start_paused = true)]
    async fn a_signal_that_cannot_fire_never_resolves() {
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
