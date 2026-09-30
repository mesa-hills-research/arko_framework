//! Live snapshots of a running task, so that a host can save it while it
//! runs and continue it after a crash; see [`TaskSnapshotSink`].

use std::hash::Hasher;
use std::sync::Arc;

use async_trait::async_trait;

use crate::task::Task;

/// Receives a running task at its safe points, so that the host can save
/// it while it runs
/// ([`AgentRuntime::with_task_snapshots`](crate::runtime::AgentRuntime::with_task_snapshots)).
/// A task saved this way runs again from where it was saved: calls that
/// were running when the host stopped
/// ([`Task::running_tool_calls`]) are settled by their tools'
/// [`recovery_mode`](crate::tool::Tool::recovery_mode).
///
/// The runtime reports the task:
/// - after the model's turn is added to the conversation, before its tool
///   calls run;
/// - before a group of tool calls starts, with the calls recorded as
///   running: the write-ahead point, reported `durable` (unless every
///   call in the group is one recovery simply runs again:
///   [`ToolRecovery::Rerun`](crate::tool::ToolRecovery::Rerun) without the
///   host's own output, or, for a Claude Code agent, a
///   [read-only built-in](crate::claude_code::READ_ONLY_BUILT_IN_TOOLS));
/// - after each group of calls' results are recorded;
/// - at every turn boundary (which covers pauses and waits for a
///   command), after a handover, and when the run ends.
///
/// A safe point where nothing changed since the last report reports
/// nothing, except a durable one.
#[async_trait]
pub trait TaskSnapshotSink: Send + Sync {
    /// The task changed. With `durable` set (before tool calls run), return
    /// only once this snapshot, and whatever the host must save with it, is
    /// on disk: the calls start once it returns. Otherwise the sink may
    /// save it later, and a snapshot that arrives meanwhile may replace it
    /// (only the latest one matters).
    ///
    /// An error is logged as a warning and the run goes on: a task is never
    /// stopped because it could not be saved, not even at a durable point.
    async fn task_changed(&self, task: &Task, durable: bool) -> Result<(), String>;
}

/// Reports a runtime's task to its sink, skipping safe points where the
/// task hasn't changed since the last report.
#[derive(Default)]
pub(crate) struct TaskSnapshots {
    sink: Option<Arc<dyn TaskSnapshotSink>>,
    /// A hash of the task as last reported.
    last: Option<u64>,
}

impl TaskSnapshots {
    pub(crate) fn new(sink: Option<Arc<dyn TaskSnapshotSink>>) -> Self {
        Self { sink, last: None }
    }

    /// Reports `task` if it changed since the last report, or always when
    /// `durable`.
    pub(crate) async fn report(&mut self, task: &Task, durable: bool) {
        let Some(sink) = &self.sink else {
            return;
        };
        let hash = fingerprint(task);
        if !durable && hash.is_some() && hash == self.last {
            return;
        }
        match sink.task_changed(task, durable).await {
            Ok(()) => self.last = hash,
            Err(error) => {
                // The next safe point reports the task again.
                self.last = None;
                log::warn!(
                    "Task {}: the task snapshot could not be saved{}: {error}",
                    task.id,
                    if durable {
                        " before its tool calls"
                    } else {
                        ""
                    }
                );
            }
        }
    }
}

/// A hash of the task's serialized form: equal for a task that did not
/// change. `None` if it can't be serialized (the sink will say why).
fn fingerprint(task: &Task) -> Option<u64> {
    struct HashWriter(std::collections::hash_map::DefaultHasher);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.write(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(Default::default());
    serde_json::to_writer(&mut writer, task).ok()?;
    Some(writer.0.finish())
}
