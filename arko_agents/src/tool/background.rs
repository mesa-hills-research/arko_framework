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
//! tells a call that waits (for other agents, for a message, for a job)
//! that a note, a terminal update or other context arrived for its task, so
//! it can return and let the model read it.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use tokio::sync::watch;

use crate::runtime::TurnContextSource;

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
        // Dropped without firing, so this signal never fires.
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
/// request the call came from. It also counts, for as long as it is there,
/// the context the task's turn context source holds for the next update
/// that wakes calls ([`TurnContextSource::has_waking_update`]), such as a
/// note posted for the task. See
/// [`TaskContext::context_arrived`](super::TaskContext::context_arrived).
///
/// Clones share the same starting point. [`never`](Self::never) (also the
/// `Default`) never fires.
#[derive(Clone, Default)]
pub struct ContextSignal {
    source: Option<watch::Receiver<u64>>,
    /// The source's count the signal counts from.
    start: u64,
    /// The source's word on whether a note the user typed waits.
    note: Option<Arc<AtomicBool>>,
    /// The turn context source asked for what it holds, and the task. Held
    /// weakly, since a call moved to the background owns its signal, and
    /// the source can hold what owns the call (a workflow's holds the
    /// terminal manager).
    updates: Option<(Weak<dyn TurnContextSource>, uuid::Uuid)>,
}

impl ContextSignal {
    /// A signal that never fires, for a [`TaskContext`](super::TaskContext)
    /// built outside a runtime, and for tests.
    pub fn never() -> Self {
        Self::default()
    }

    /// This signal, also fired while `source` holds context for task
    /// `task_id` that wakes calls ([`TurnContextSource::has_waking_update`]).
    /// The runtime gives its calls such a signal for the task's own source.
    /// For a host that runs tools itself, and for tests. The signal doesn't
    /// keep the source alive. Once it is dropped, only the context the
    /// runtime took fires the signal.
    pub fn with_updates(
        mut self,
        source: &Arc<dyn TurnContextSource>,
        task_id: uuid::Uuid,
    ) -> Self {
        self.updates = Some((Arc::downgrade(source), task_id));
        self
    }

    /// This signal without what the turn context source holds, for a call
    /// that takes that context itself, as `wait_for_agents` takes its mail.
    pub fn without_updates(&self) -> Self {
        Self {
            updates: None,
            ..self.clone()
        }
    }

    /// Whether context arrived since the signal was made, or the turn
    /// context source holds some that wakes calls.
    pub fn has_arrived(&self) -> bool {
        self.taken()
            || self.updates.as_ref().is_some_and(|(source, task_id)| {
                source
                    .upgrade()
                    .is_some_and(|source| source.has_waking_update(*task_id))
            })
    }

    /// Whether context the runtime took arrived since the signal was made.
    fn taken(&self) -> bool {
        self.source
            .as_ref()
            .is_some_and(|source| *source.borrow() != self.start)
    }

    /// Whether a note the user typed waits for the model now, for a call
    /// that says what ended its wait. A Claude Code agent reads such a note
    /// only after every call of the response it is in, so a call that
    /// starts while one waits counts it as arrived.
    pub fn note_waits(&self) -> bool {
        self.note
            .as_ref()
            .is_some_and(|note| note.load(Ordering::Acquire))
    }

    /// Resolves once context arrives (at once if some arrived since the
    /// signal was made, or the turn context source holds some that wakes
    /// calls). A signal that can no longer fire, such as
    /// [`never`](Self::never), never resolves.
    pub async fn arrived(&self) {
        tokio::select! {
            () = self.taken_arrives() => {}
            () = self.update_arrives() => {}
        }
    }

    async fn taken_arrives(&self) {
        let Some(source) = &self.source else {
            return std::future::pending().await;
        };
        let mut source = source.clone();
        let start = self.start;
        if source.wait_for(|count| *count != start).await.is_err() {
            std::future::pending::<()>().await;
        }
    }

    async fn update_arrives(&self) {
        let Some((source, task_id)) = &self.updates else {
            return std::future::pending().await;
        };
        loop {
            let changed = {
                let Some(source) = source.upgrade() else {
                    return std::future::pending().await;
                };
                // Made before the check, so a change after it resolves it.
                let changed = source.update_changed(*task_id);
                if source.has_waking_update(*task_id) {
                    return;
                }
                // The source isn't held while the wait goes on.
                changed
            };
            match changed {
                Some(changed) => changed.await,
                None => return std::future::pending().await,
            }
        }
    }
}

impl fmt::Debug for ContextSignal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Without asking the source, which a formatter shouldn't do.
        formatter
            .debug_struct("ContextSignal")
            .field("arrived", &self.taken())
            .field("reads_updates", &self.updates.is_some())
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
    /// A note the user typed waits for the model.
    note: Arc<AtomicBool>,
}

impl ContextSource {
    /// A source that has seen no context.
    pub fn new() -> Self {
        Self {
            arrivals: Arc::new(watch::Sender::new(0)),
            caught_up: Arc::default(),
            note: Arc::default(),
        }
    }

    /// A signal that fires when context arrives from now on.
    pub fn signal(&self) -> ContextSignal {
        let source = self.arrivals.subscribe();
        let start = *source.borrow();
        ContextSignal {
            source: Some(source),
            start,
            note: Some(self.note.clone()),
            updates: None,
        }
    }

    /// A signal for a call starting now. Context taken since the model's
    /// request (or Claude Code's latest hand-over) fires it at once, since
    /// the model made the call without reading it.
    pub(crate) fn call_signal(&self) -> ContextSignal {
        ContextSignal {
            source: Some(self.arrivals.subscribe()),
            start: self.caught_up.load(Ordering::Acquire),
            note: Some(self.note.clone()),
            updates: None,
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

    /// Whether a note the user typed waits for the model, which the
    /// signals' [`note_waits`](ContextSignal::note_waits) report.
    pub fn set_note_waits(&self, waits: bool) {
        self.note.store(waits, Ordering::Release);
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

    #[test]
    fn signals_report_whether_a_typed_note_waits() {
        let source = ContextSource::new();
        let signal = source.signal();
        assert!(!signal.note_waits() && !ContextSignal::never().note_waits());
        source.set_note_waits(true);
        assert!(signal.note_waits() && source.call_signal().note_waits());
        source.set_note_waits(false);
        assert!(!signal.note_waits());
    }

    /// A turn context source that holds context while `waiting` is set. With
    /// `post_after_check`, its next answer of `false` is followed at once by
    /// a post, as a note posted just after a wait looked.
    #[derive(Default)]
    struct Held {
        waiting: AtomicBool,
        post_after_check: AtomicBool,
        changes: watch::Sender<u64>,
    }

    impl Held {
        fn post(&self) {
            self.waiting.store(true, Ordering::SeqCst);
            self.changes.send_modify(|count| *count += 1);
        }
    }

    impl TurnContextSource for Held {
        fn turn_context(&self, _task_id: uuid::Uuid) -> Option<String> {
            None
        }

        fn has_waking_update(&self, _task_id: uuid::Uuid) -> bool {
            let waiting = self.waiting.load(Ordering::SeqCst);
            if !waiting && self.post_after_check.swap(false, Ordering::SeqCst) {
                self.post();
            }
            waiting
        }

        fn update_changed(
            &self,
            _task_id: uuid::Uuid,
        ) -> Option<std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>> {
            let mut changes = self.changes.subscribe();
            Some(Box::pin(async move {
                let _ = changes.changed().await;
            }))
        }
    }

    /// The signal counts what the source holds for as long as it holds it,
    /// and asking takes nothing.
    #[tokio::test]
    async fn a_signal_counts_what_the_turn_context_source_holds() {
        let held = Arc::new(Held::default());
        let source: Arc<dyn TurnContextSource> = held.clone();
        let task_id = uuid::Uuid::new_v4();
        let signal = ContextSource::new()
            .call_signal()
            .with_updates(&source, task_id);
        assert!(!signal.has_arrived());
        let waiting = tokio::spawn({
            let signal = signal.clone();
            async move { signal.arrived().await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished());
        held.post();
        timeout(Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
        assert!(signal.has_arrived() && signal.has_arrived(), "still held");
        assert!(!signal.without_updates().has_arrived());
        // Once the runtime took it, nothing is left to count.
        held.waiting.store(false, Ordering::SeqCst);
        assert!(!signal.has_arrived());
    }

    /// Context the source gets just after a wait looked still ends the wait,
    /// since the wait subscribed to the source's changes before it looked.
    #[tokio::test]
    async fn context_held_just_after_the_check_ends_the_wait() {
        let held = Arc::new(Held::default());
        held.post_after_check.store(true, Ordering::SeqCst);
        let source: Arc<dyn TurnContextSource> = held.clone();
        let signal = ContextSource::new()
            .signal()
            .with_updates(&source, uuid::Uuid::new_v4());
        timeout(Duration::from_secs(5), signal.arrived())
            .await
            .expect("the change after the check ends it");
    }

    /// The signal holds its source weakly, also while it waits. Once the
    /// source is gone, only the context the runtime took fires it.
    #[tokio::test]
    async fn a_dropped_source_leaves_the_taken_context() {
        let held = Arc::new(Held::default());
        held.post();
        let source: Arc<dyn TurnContextSource> = held.clone();
        let signal = ContextSource::new()
            .signal()
            .with_updates(&source, uuid::Uuid::new_v4());
        assert!(signal.has_arrived());
        drop((held, source));
        assert!(!signal.has_arrived());

        let held = Arc::new(Held::default());
        let source: Arc<dyn TurnContextSource> = held.clone();
        let arrivals = ContextSource::new();
        let signal = arrivals
            .signal()
            .with_updates(&source, uuid::Uuid::new_v4());
        let waiting = tokio::spawn({
            let signal = signal.clone();
            async move { signal.arrived().await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        let gone = Arc::downgrade(&source);
        drop((held, source));
        assert!(gone.upgrade().is_none(), "the waiting signal holds it");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished());
        arrivals.notify();
        timeout(Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
    }
}
