//! A subscription to a workflow's events that loses none of them
//! ([`Workflow::subscribe_events_lossless`](crate::Workflow::subscribe_events_lossless)).
//!
//! Each subscriber has a queue of its own. While it keeps up, it gets the
//! events as they come. While it lags, a streamed chunk of text or
//! reasoning joins the chunk of the same kind and run waiting at the end of
//! its queue, so the queue grows with the lifecycle events only and the
//! text arrives whole, in fewer pieces.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};

use arko_agents::runtime::event::AgentEvent;
use parking_lot::Mutex;

use super::result::WorkflowEvent;

/// The lossless subscribers of a workflow's events.
#[derive(Default)]
pub(crate) struct EventHub {
    subscribers: Mutex<Vec<Weak<Queue>>>,
}

impl EventHub {
    /// Gives `event` to every subscriber.
    pub(crate) fn publish(&self, event: &WorkflowEvent) {
        let mut subscribers = self.subscribers.lock();
        subscribers.retain(|queue| match queue.upgrade() {
            Some(queue) => {
                queue.push(event.clone());
                true
            }
            None => false,
        });
    }

    pub(crate) fn subscribe(&self) -> EventReceiver {
        let queue = Arc::new(Queue::default());
        self.subscribers.lock().push(Arc::downgrade(&queue));
        EventReceiver { queue }
    }
}

impl Drop for EventHub {
    /// The workflow is gone: its subscribers get what is queued, then `None`.
    fn drop(&mut self) {
        for queue in self.subscribers.lock().iter().filter_map(Weak::upgrade) {
            queue.close();
        }
    }
}

#[derive(Default)]
struct Queue {
    state: Mutex<QueueState>,
    ready: tokio::sync::Notify,
}

#[derive(Default)]
struct QueueState {
    events: VecDeque<WorkflowEvent>,
    closed: bool,
}

impl Queue {
    fn push(&self, event: WorkflowEvent) {
        {
            let mut state = self.state.lock();
            if let Some(last) = state.events.back_mut()
                && let Some(text) = joined(last, &event)
            {
                text.push_str(chunk_text(&event.event).unwrap_or_default());
            } else {
                state.events.push_back(event);
            }
        }
        self.ready.notify_one();
    }

    fn close(&self) {
        self.state.lock().closed = true;
        self.ready.notify_one();
    }
}

/// The text of `last` that `event` joins: both chunks of the same kind
/// from the same run.
fn joined<'a>(last: &'a mut WorkflowEvent, event: &WorkflowEvent) -> Option<&'a mut String> {
    if last.agent_run_id != event.agent_run_id {
        return None;
    }
    match (&mut last.event, &event.event) {
        (AgentEvent::AssistantResponseChunk(text), AgentEvent::AssistantResponseChunk(_))
        | (AgentEvent::ReasoningChunk(text), AgentEvent::ReasoningChunk(_)) => Some(text),
        _ => None,
    }
}

fn chunk_text(event: &AgentEvent) -> Option<&str> {
    match event {
        AgentEvent::AssistantResponseChunk(text) | AgentEvent::ReasoningChunk(text) => Some(text),
        _ => None,
    }
}

/// Receives every event of a workflow, from when it subscribed, in order.
/// While the receiver lags, streamed chunks of text or reasoning of one
/// run wait joined into one event, so nothing is dropped and the queue
/// stays as small as the lifecycle events allow.
pub struct EventReceiver {
    queue: Arc<Queue>,
}

impl EventReceiver {
    /// The next event, waiting for one. `None` once the workflow is dropped
    /// and every event before was received.
    pub async fn recv(&mut self) -> Option<WorkflowEvent> {
        loop {
            let ready = self.queue.ready.notified();
            tokio::pin!(ready);
            ready.as_mut().enable();
            {
                let mut state = self.queue.state.lock();
                if let Some(event) = state.events.pop_front() {
                    return Some(event);
                }
                if state.closed {
                    return None;
                }
            }
            ready.await;
        }
    }

    /// The next event if one is waiting.
    pub fn try_recv(&mut self) -> Option<WorkflowEvent> {
        self.queue.state.lock().events.pop_front()
    }

    /// How many events wait (joined chunks counting once).
    pub fn len(&self) -> usize {
        self.queue.state.lock().events.len()
    }

    /// Whether no event waits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for EventReceiver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EventReceiver")
            .field("waiting", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(run: uuid::Uuid, event: AgentEvent) -> WorkflowEvent {
        WorkflowEvent {
            timestamp: chrono::Utc::now(),
            workflow_run_id: uuid::Uuid::nil(),
            agent_name: "worker".into(),
            agent_run_id: run,
            event,
        }
    }

    /// A lagging subscriber gets each run's chunks joined, in order with the
    /// other events, and nothing is lost.
    #[tokio::test]
    async fn chunks_join_while_the_subscriber_lags() {
        let hub = EventHub::default();
        let mut receiver = hub.subscribe();
        let (one, two) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        for piece in ["a", "b", "c"] {
            hub.publish(&event(
                one,
                AgentEvent::AssistantResponseChunk(piece.into()),
            ));
        }
        hub.publish(&event(two, AgentEvent::AssistantResponseChunk("x".into())));
        hub.publish(&event(one, AgentEvent::AssistantResponseChunk("d".into())));
        hub.publish(&event(one, AgentEvent::ReasoningChunk("r".into())));
        hub.publish(&event(one, AgentEvent::TaskStreamEnd));
        hub.publish(&event(one, AgentEvent::AssistantResponseChunk("e".into())));
        let received: Vec<_> = std::iter::from_fn(|| receiver.try_recv())
            .map(|event| (event.agent_run_id == one, event.event))
            .collect();
        assert_eq!(
            received,
            [
                (true, AgentEvent::AssistantResponseChunk("abc".into())),
                (false, AgentEvent::AssistantResponseChunk("x".into())),
                (true, AgentEvent::AssistantResponseChunk("d".into())),
                (true, AgentEvent::ReasoningChunk("r".into())),
                (true, AgentEvent::TaskStreamEnd),
                (true, AgentEvent::AssistantResponseChunk("e".into())),
            ]
        );
    }

    /// A subscriber waits for the next event, and gets `None` once the
    /// workflow is gone and it has every event.
    #[tokio::test]
    async fn a_subscriber_waits_and_ends_with_the_workflow() {
        let hub = EventHub::default();
        let mut receiver = hub.subscribe();
        let run = uuid::Uuid::new_v4();
        let waiting = tokio::spawn(async move {
            let first = receiver.recv().await.map(|event| event.event);
            let second = receiver.recv().await.map(|event| event.event);
            let end = receiver.recv().await;
            (first, second, end.is_none())
        });
        tokio::task::yield_now().await;
        hub.publish(&event(run, AgentEvent::TaskStreamEnd));
        hub.publish(&event(run, AgentEvent::AssistantResponseChunk("z".into())));
        drop(hub);
        let (first, second, ended) = waiting.await.unwrap();
        assert_eq!(first, Some(AgentEvent::TaskStreamEnd));
        assert_eq!(second, Some(AgentEvent::AssistantResponseChunk("z".into())));
        assert!(ended);
    }
}
