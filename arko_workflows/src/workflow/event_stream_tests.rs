//! The lossless event subscription under a burst: several agents streaming
//! thousands of chunks into a subscriber that keeps falling behind.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arko_agents::runtime::event::AgentEvent;
use serde_json::json;

use super::*;
use crate::test_support::{Reply, ResponsesServer, Workspace, agent_config};

/// The pieces agent `n`'s reply streams in.
const PIECES: usize = 3_000;

fn piece(agent: usize, index: usize) -> String {
    format!("a{agent}p{index} ")
}

/// A reply streamed in [`PIECES`] deltas.
fn streamed_reply(agent: usize) -> String {
    let id = format!("m{agent}");
    let text: String = (0..PIECES).map(|index| piece(agent, index)).collect();
    let mut events: Vec<serde_json::Value> = (0..PIECES)
        .map(|index| {
            json!({"type": "response.output_text.delta", "item_id": format!("msg_{id}"),
                   "output_index": 0, "delta": piece(agent, index)})
        })
        .collect();
    events.push(json!({"type": "response.output_item.done", "output_index": 0, "item": {
        "type": "message", "id": format!("msg_{id}"), "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }}));
    events.push(json!({"type": "response.completed", "response": {
        "id": format!("resp_{id}"), "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-5.6-luna", "output": [], "usage": {"input_tokens": 10, "output_tokens": 5},
        "metadata": {},
    }}));
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

/// Four agents stream 3,000 pieces each at once while a subscriber reads
/// nothing, so it falls as far behind as it can: it receives every agent's
/// whole text, and every lifecycle event. The first response of agent 2 is
/// cut off and sent again, so the text it streamed first is dropped as the
/// host is told to (`StreamedOutputDiscarded`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lagging_subscriber_receives_every_piece_of_text() {
    let cuts = AtomicUsize::new(0);
    let server = ResponsesServer::start_async(move |_, body| {
        let agent = (1..=4)
            .find(|agent| body.contains(&format!("Stream reply {agent}.")))
            .unwrap();
        let stream = streamed_reply(agent);
        let reply = if agent == 2 && cuts.fetch_add(1, Ordering::SeqCst) == 0 {
            let sent = stream.len() * 85 / 100;
            Reply::Cut { stream, sent }
        } else {
            Reply::Stream(stream)
        };
        Box::pin(async move { Some(reply) })
    })
    .await;
    let workspace = Workspace::new();
    let mut builder = Workflow::builder()
        .name("event-stream-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()));
    for agent in 1..=4 {
        let mut chat = agent_config(&format!("Chat {agent}"));
        chat.tool_config.text_reply_ends_task = true;
        builder = builder.agent(format!("chat{agent}"), chat);
    }
    let workflow = builder.build().unwrap();
    let mut events = workflow.subscribe_events_lossless();
    let results = workflow
        .run_agents_parallel(vec![
            ("chat1", "Stream reply 1.".to_string()),
            ("chat2", "Stream reply 2.".to_string()),
            ("chat3", "Stream reply 3.".to_string()),
            ("chat4", "Stream reply 4.".to_string()),
        ])
        .await
        .unwrap();
    let mut texts: HashMap<uuid::Uuid, String> = HashMap::new();
    let mut ended = 0;
    while let Some(event) = events.try_recv() {
        match event.event {
            AgentEvent::AssistantResponseChunk(text) => {
                texts.entry(event.agent_run_id).or_default().push_str(&text)
            }
            AgentEvent::StreamedOutputDiscarded { text_bytes, .. } => {
                let text = texts.entry(event.agent_run_id).or_default();
                text.truncate(text.len() - text_bytes);
            }
            AgentEvent::TaskStateChanged(arko_agents::TaskState::Completed { .. }) => ended += 1,
            _ => {}
        }
    }
    assert_eq!(ended, 4);
    assert_eq!(server.requests(), 5, "agent 2 asked twice");
    for (index, result) in results.iter().enumerate() {
        let expected: String = (0..PIECES)
            .map(|piece_index| piece(index + 1, piece_index))
            .collect();
        assert_eq!(texts[&result.agent_run_id], expected, "agent {}", index + 1);
    }
}
