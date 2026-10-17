//! Live snapshots of a running task, so that a host can save it while it
//! runs and continue it after a crash. See [`TaskSnapshotSink`].

use std::hash::Hasher;
use std::sync::Arc;

use async_trait::async_trait;

use crate::task::Task;

/// Receives a running task at its safe points, so that the host can save
/// it while it runs
/// ([`AgentRuntime::with_task_snapshots`](crate::runtime::AgentRuntime::with_task_snapshots)).
/// A task saved this way runs again from where it was saved. Calls that
/// were running when the host stopped
/// ([`Task::running_tool_calls`]) are settled by their tools'
/// [`recovery_mode`](crate::tool::Tool::recovery_mode).
///
/// The runtime reports the task:
/// - after the model's turn is added to the conversation, before its tool
///   calls run.
/// - before a group of tool calls starts, with the calls recorded as
///   running. This is the write-ahead point, reported `durable` (unless every
///   call in the group is one recovery simply runs again through
///   [`ToolRecovery::Rerun`](crate::tool::ToolRecovery::Rerun) without the
///   host's own output, or, for a Claude Code agent, a
///   [read-only built-in](crate::claude_code::READ_ONLY_BUILT_IN_TOOLS)).
/// - after each group of calls' results are recorded.
/// - at every turn boundary (which covers pauses and waits for a
///   command), after a handover, and when the run ends.
///
/// A safe point where nothing changed since the last report reports
/// nothing, except a durable one.
#[async_trait]
pub trait TaskSnapshotSink: Send + Sync {
    /// The task changed. With `durable` set (before tool calls run), return
    /// only once this snapshot, and whatever the host must save with it, is
    /// on disk. The calls start once it returns. Otherwise the sink may
    /// save it later, and a snapshot that arrives meanwhile may replace it
    /// (only the latest one matters).
    ///
    /// An error is logged as a warning and the run goes on. A task is never
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
/// change. `None` if it can't be serialized (the sink will say why). Each
/// image counts by its identity (where its data is, its length and a sample
/// of it), not by all its data: hashing the screenshots of a conversation
/// at every safe point would cost more than the rest of the task. An image
/// isn't changed in place: one replaced by another is in another place.
fn fingerprint(task: &Task) -> Option<u64> {
    /// Names each image by its identity.
    struct Identities;
    impl arko_ai::image_store::ImageStore for Identities {
        fn save(&self, data: &str, _mime_type: &str) -> Result<String, String> {
            Ok(format!(
                "{:p}-{}-{:x}",
                data.as_ptr(),
                data.len(),
                arko_ai::image_store::sample_hash(data)
            ))
        }
        fn load(&self, _reference: &str, _mime_type: &str) -> Result<String, String> {
            Err("image identities can't be read".to_string())
        }
    }
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
    arko_ai::image_store::with_image_store(Arc::new(Identities), || {
        serde_json::to_writer(&mut writer, task)
    })
    .ok()?;
    Some(writer.0.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentConfig, AgentLLMConfig};
    use arko_ai::{Content, Image, Message};

    fn task() -> Task {
        let config = AgentConfig::builder(
            "worker".into(),
            "Worker".into(),
            AgentLLMConfig {
                model: arko_ai::Model::anthropic_claude_opus_5_5(),
                generation_config: Default::default(),
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .build()
        .unwrap();
        Task::new(config, "Look.".into(), "/w".into(), None)
    }

    /// The fingerprint tells a task that changed from one that didn't, also
    /// by its images, which count by identity, not by all their data.
    #[test]
    fn the_fingerprint_sees_images_change_without_hashing_their_data() {
        let mut task = task();
        let image = |fill: u8| Image::from_bytes(&vec![fill; 1 << 20], "image/png");
        task.conversation
            .messages
            .push(Message::new_user_with_contents(vec![Content::Image(
                image(1),
            )]));
        let first = fingerprint(&task).unwrap();
        assert_eq!(fingerprint(&task), Some(first));
        // Another image of the same size in its place.
        let last = task.conversation.messages.last_mut().unwrap();
        last.content = vec![Content::Image(image(2))];
        let second = fingerprint(&task).unwrap();
        assert_ne!(second, first);
        let last = task.conversation.messages.last_mut().unwrap();
        last.content.push(Content::Image(image(2)));
        assert_ne!(fingerprint(&task).unwrap(), second);
        // A copy of the task, its images elsewhere, counts as changed: at
        // worst it is reported once more.
        let copy = task.clone();
        assert_ne!(fingerprint(&copy), fingerprint(&task));
    }
}
