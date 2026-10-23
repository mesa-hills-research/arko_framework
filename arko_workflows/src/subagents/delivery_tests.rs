//! What a lead's runtime keeps of a wait's result decides what the turn-end
//! check counts as delivered. These tests shorten results as Claude Code
//! does and check that every report reaches the lead once.

use std::collections::HashMap;
use std::sync::Arc;

use arko_agents::{InterceptedResult, TaskContext, Tool, ToolError, ToolResultInterceptor};
use serde_json::{Value, json};

use super::tests::{Gate, bodies_of, builder, complete, result_of, serve, start, until, within};
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

/// Starts a subagent for `lead` (a run id that is no run of the workflow).
async fn start_run(
    workflow: &crate::Workflow,
    workspace: &Workspace,
    lead: uuid::Uuid,
    task: &str,
) {
    let none = std::collections::BTreeMap::new();
    let ctx = workflow.context();
    let started = ctx
        .subagents
        .start(
            ctx,
            lead,
            "lead",
            0,
            &workspace.0,
            "worker",
            task,
            None,
            false,
            None,
            none,
        )
        .await;
    started.unwrap();
}

/// A lead (a run id that is no run of the workflow) whose subagents have
/// finished with these reports.
struct Waiter {
    server: crate::test_support::ResponsesServer,
    workspace: Workspace,
    workflow: crate::Workflow,
    lead: uuid::Uuid,
}

impl Waiter {
    async fn new(reports: &[String]) -> Self {
        let scripts = ["r1", "r2"]
            .into_iter()
            .zip(reports)
            .map(|(worker, report)| (worker, vec![complete(report)]))
            .collect();
        let server = serve(scripts).await;
        let workspace = Workspace::new();
        let workflow = builder(&server, &workspace, SubagentsConfig::default())
            .build()
            .unwrap();
        let lead = uuid::Uuid::new_v4();
        for n in 1..=reports.len() {
            start_run(&workflow, &workspace, lead, &format!("Task {n}.")).await;
        }
        until(|| {
            workflow
                .subagents()
                .iter()
                .all(|run| run.state == SubagentState::Finished)
        })
        .await;
        Self {
            server,
            workspace,
            workflow,
            lead,
        }
    }

    /// A call of `wait_for_agents` by the lead, in Claude Code or not.
    async fn wait(&self, claude_code: bool, all: bool) -> String {
        let ctx = self.workflow.context();
        let tool =
            WaitForAgentsTool::new(ctx.clone(), "lead".into(), Arc::new(self.server.client()));
        let mut task_context = TaskContext::new(self.lead, "lead", "", &self.workspace.0);
        task_context.runs_in_claude_code = claude_code;
        let params = HashMap::from([("all".to_string(), json!(all))]);
        within(tool.execute(&task_context, params)).await.unwrap()
    }
}

/// A caller that doesn't run in Claude Code has no limit on the result of a
/// wait, so reports come whole and together, whatever their size.
#[tokio::test]
async fn a_wait_of_a_native_caller_returns_every_report_whole() {
    let (a, b) = (report("A", 30_000), report("B", 30_000));
    let waiter = Waiter::new(&[a.clone(), b.clone()]).await;
    let waited = waiter.wait(false, true).await;
    assert!(waited.len() > 60_000, "{}", waited.len());
    assert!(waited.contains(a.trim()) && waited.contains(b.trim()));
    assert!(waited.contains("[end of mail 1]") && waited.contains("[end of mail 2]"));
    assert!(!waited.contains("follows next"), "{waited}");
    assert!(waiter.workflow.pending_news(waiter.lead).is_empty());
    let big = report("C", 60_000);
    let waiter = Waiter::new(std::slice::from_ref(&big)).await;
    let waited = waiter.wait(false, false).await;
    assert!(waited.contains(big.trim()) && !waited.contains("cut here"));
}

/// Reports that don't fit the result of a wait in Claude Code don't come in
/// one. The mail that didn't fit stays in the mailbox, and the next wait
/// returns it whole.
#[tokio::test]
async fn a_claude_code_wait_leaves_the_mail_that_does_not_fit_for_the_next() {
    let (a, b) = (report("A", 30_000), report("B", 30_000));
    let waiter = Waiter::new(&[a.clone(), b.clone()]).await;
    let first = waiter.wait(true, true).await;
    assert!(first.len() < 40_000, "{}", first.len());
    assert!(first.contains("[mail 1]") && first.contains("[end of mail 1]"));
    assert!(!first.contains("[mail 2]"), "{first}");
    assert!(
        first.ends_with("didn't fit here and follows next."),
        "{first}"
    );
    assert_eq!(waiter.workflow.pending_news(waiter.lead).len(), 1);
    let second = waiter.wait(true, true).await;
    assert!(second.contains("[mail 2]") && second.contains("[end of mail 2]"));
    assert!(!second.contains("follows next"), "{second}");
    let both = format!("{first}{second}");
    assert!(both.contains(a.trim()) && both.contains(b.trim()));
    assert!(waiter.workflow.pending_news(waiter.lead).is_empty());
}

/// The mail a wait in Claude Code leaves comes in the lead's next context
/// update, with both its markers.
#[tokio::test]
async fn the_mail_a_claude_code_wait_leaves_comes_in_the_next_update() {
    let waiter = Waiter::new(&[report("A", 30_000), report("B", 30_000)]).await;
    waiter.wait(true, true).await;
    let (update, ids) = waiter
        .workflow
        .context()
        .subagents
        .update(waiter.lead)
        .unwrap();
    assert_eq!(ids, ["mail:2"]);
    assert!(update.contains("[mail 2]") && update.contains("[end of mail 2]"));
    assert!(waiter.workflow.pending_news(waiter.lead).is_empty());
}

/// One report over the limit of a wait in Claude Code comes as its first
/// part, with its file and the line to read on from.
#[tokio::test]
async fn a_claude_code_wait_cuts_a_report_over_the_limit_with_its_file() {
    let big = report("A", 60_000);
    let waiter = Waiter::new(std::slice::from_ref(&big)).await;
    let waited = waiter.wait(true, false).await;
    assert!(waited.len() < 40_000, "{}", waited.len());
    let notice = waited
        .find("[... cut here. All of it is in ")
        .expect(&waited);
    assert!(waited.find("[end of mail 1]").unwrap() > notice);
    let rest = &waited[notice + "[... cut here. All of it is in ".len()..];
    let (file, rest) = rest.split_once(". Read on from line ").unwrap();
    let line: usize = rest.split_once(' ').unwrap().0.parse().unwrap();
    let saved = std::fs::read_to_string(file).unwrap();
    assert!(saved.contains(big.trim()));
    // The part shown ends where the file's line `line` begins.
    let shown = waited[..notice].trim_end().lines().last().unwrap();
    assert_eq!(saved.lines().nth(line - 2), Some(shown));
    assert!(waiter.workflow.pending_news(waiter.lead).is_empty());
}

/// Two reports that fit still come together, whole.
#[tokio::test]
async fn reports_that_fit_come_together() {
    let (a, b) = (report("A", 10_000), report("B", 10_000));
    let outcome = delegate(&[a.clone(), b.clone()], 1, true, 50_000).await;
    let waited = &outcome.waits[0];
    assert!(waited.starts_with("None of your agent runs is running now."));
    assert!(waited.contains(a.trim()) && waited.contains(b.trim()));
    assert!(waited.contains("[end of mail 1]") && waited.contains("[end of mail 2]"));
    assert!(!waited.contains("follows next"), "{waited}");
    assert!(outcome.news.is_empty(), "{:?}", outcome.news);
    assert!(outcome.pending.is_empty());
}

/// A result the runtime cut before the end of a report doesn't count as
/// delivered, though it holds the report's first marker. The report goes out
/// again, marked.
#[tokio::test]
async fn a_report_cut_short_in_the_conversation_is_given_again() {
    let outcome = delegate(&[report("A", 5_000)], 1, false, 3_000).await;
    let waited = &outcome.waits[0];
    assert!(waited.starts_with("<persisted-output>"));
    assert!(waited.contains("[mail 1]") && !waited.contains("[end of mail 1]"));
    assert_eq!(outcome.news, [1]);
    assert_eq!(outcome.pending.len(), 1);
    assert!(outcome.pending[0].resent);
}

/// In Claude Code, a small report behind one that doesn't fit stays behind
/// it, and a wait for all returns when its result is full though a run still
/// works. The mail left comes in order with the next wait.
#[tokio::test]
async fn a_full_wait_returns_while_runs_go_on_and_keeps_the_order() {
    let (a, b) = (report("A", 30_000), report("B", 30_000));
    let server = serve(vec![
        ("r1", vec![complete(&a)]),
        ("r2", vec![complete(&b)]),
        ("r3", vec![complete("A small report.")]),
        ("r4", vec![("gate", json!({})), complete("Late.")]),
    ])
    .await;
    let workspace = Workspace::new();
    let gate = Arc::new(Gate::default());
    let workflow = builder(&server, &workspace, SubagentsConfig::default())
        .agent_tool("worker", gate.clone())
        .build()
        .unwrap();
    let lead = uuid::Uuid::new_v4();
    // One after the other, so that the mail numbers follow the run numbers.
    for (done, task) in ["Task 1.", "Task 2.", "Task 3.", "Task 4."]
        .into_iter()
        .enumerate()
    {
        start_run(&workflow, &workspace, lead, task).await;
        if done < 3 {
            until(|| {
                let runs = workflow.subagents();
                runs.iter()
                    .filter(|run| run.state == SubagentState::Finished)
                    .count()
                    == done + 1
            })
            .await;
        }
    }
    within(gate.started.notified()).await;
    let ctx = workflow.context();
    let tool = WaitForAgentsTool::new(ctx.clone(), "lead".into(), Arc::new(server.client()));
    let mut task_context = TaskContext::new(lead, "lead", "", &workspace.0);
    task_context.runs_in_claude_code = true;
    let all = HashMap::from([("all".to_string(), json!(true))]);
    let first = within(tool.execute(&task_context, all)).await.unwrap();
    assert!(first.starts_with("Still running: r4."), "{first}");
    assert!(first.contains("[end of mail 1]"));
    assert!(
        !first.contains("[mail 2]") && !first.contains("[mail 3]"),
        "{first}"
    );
    assert!(first.ends_with("Mail from r2 and r3 didn't fit here and follows next."));
    assert_eq!(workflow.pending_news(lead).len(), 2);
    let second = within(tool.execute(&task_context, HashMap::new()))
        .await
        .unwrap();
    let (two, three) = (
        second.find("[mail 2]").unwrap(),
        second.find("[mail 3]").unwrap(),
    );
    assert!(
        two < three && second.contains("[end of mail 3]"),
        "{second}"
    );
    gate.open.notify_one();
}
