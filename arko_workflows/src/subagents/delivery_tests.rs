//! What a lead's runtime keeps of a wait's result decides what the turn-end
//! check counts as delivered. These tests shorten results as Claude Code
//! does and check that every report reaches the lead once.

use std::collections::HashMap;
use std::sync::Arc;

use arko_agents::{InterceptedResult, TaskContext, ToolError, ToolResultInterceptor};
use serde_json::{Value, json};

use super::tests::{bodies_of, builder, complete, result_of, serve, start, within};
use super::*;
use crate::test_support::Workspace;

/// Claude Code saves an MCP result over its limit (50,000 characters) and
/// shows the model the first 2,000 characters of it in a `<persisted-output>`
/// block. `over` is the limit here.
struct ClaudeCodePersists {
    over: usize,
}

#[async_trait::async_trait]
impl ToolResultInterceptor for ClaudeCodePersists {
    async fn intercept(
        &self,
        tool_name: &str,
        raw_result: &str,
        _params: &HashMap<String, Value>,
        _task_context: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        if tool_name != "wait_for_agents" || raw_result.chars().count() <= self.over {
            return Ok(InterceptedResult::Unchanged);
        }
        let json =
            serde_json::to_string_pretty(&json!([{ "type": "text", "text": raw_result }])).unwrap();
        let preview: String = json.chars().take(2_000).collect();
        Ok(InterceptedResult::Replaced(format!(
            "<persisted-output>\nOutput too large. Full output saved to: /tmp/x.json\n\nPreview (first 2KB):\n{preview}\n...\n</persisted-output>"
        )))
    }
}

/// A report of about `chars` bytes, one distinct line after another.
fn report(who: &str, chars: usize) -> String {
    let mut text = format!("# Report of {who}\n");
    let mut line = 0;
    while text.len() < chars {
        line += 1;
        text.push_str(&format!("{who} finding {line}: something worth knowing.\n"));
    }
    text
}

struct Outcome {
    /// The results of the lead's waits, as its conversation holds them.
    waits: Vec<String>,
    /// The mail the host is told is waiting once the lead's run ended.
    news: Vec<u64>,
    pending: Vec<MailItem>,
}

/// A lead starts a subagent for each report, then waits `waits` times
/// (`all` or not). Its runtime shortens a result over `over` characters.
async fn delegate(reports: &[String], waits: usize, all: bool, over: usize) -> Outcome {
    let mut lead: Vec<_> = (1..=reports.len())
        .map(|n| start(&format!("Task {n}.")))
        .collect();
    lead.extend((0..waits).map(|_| ("wait_for_agents", json!({ "all": all }))));
    let mut scripts = vec![("lead", lead)];
    for (worker, report) in ["r1", "r2"].into_iter().zip(reports) {
        scripts.push((worker, vec![complete(report)]));
    }
    let server = serve(scripts).await;
    let workspace = Workspace::new();
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .agent_tool_result_interceptor("lead", Arc::new(ClaudeCodePersists { over }))
        .build()
        .unwrap();
    let mut events = workflow.subscribe_subagents();
    let lead = within(workflow.run_agent("lead", "Delegate."))
        .await
        .unwrap();
    assert_eq!(lead.summary, "lead done");
    let bodies = bodies_of(&server, "lead");
    let waits = (0..waits)
        .map(|n| result_of(&bodies[reports.len() + n + 1], "lead", reports.len() + n))
        .collect();
    // The turn-end check runs as the lead's run is finalized.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let mut news = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let SubagentEventKind::NewsWaiting { mail_ids } = event.kind {
            news.extend(mail_ids);
        }
    }
    Outcome {
        waits,
        news,
        pending: workflow.pending_news(lead.agent_run_id),
    }
}

/// A result the runtime cut before the end of a report doesn't count as
/// delivered, though it holds the report's first marker. The report goes out
/// again.
#[tokio::test]
async fn a_report_cut_short_in_the_conversation_is_given_again() {
    let outcome = delegate(&[report("A", 5_000)], 1, false, 3_000).await;
    let waited = &outcome.waits[0];
    assert!(waited.starts_with("<persisted-output>"));
    assert!(waited.contains("[mail 1]") && !waited.contains("[end of mail 1]"));
    assert_eq!(outcome.news, [1]);
    assert_eq!(outcome.pending.len(), 1);
}
