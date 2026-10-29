//! The hub, the bridge and the shared servers, against the Python stub in
//! `tests/fixtures` (`ARKO_TEST_PYTHON` selects the interpreter).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::*;
use crate::tool::{TaskContext, Tool, ToolError, ToolOutput, ToolRecovery};

const STUB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mcp_stub_server.py"
);
const TIMEOUT: Duration = Duration::from_secs(30);

fn python() -> String {
    std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into())
}

/// The stub as `name`, with its issue tool and instructions, and `env`.
fn stub(name: &str, env: &[(&str, &str)]) -> McpServerConfig {
    let mut env: HashMap<String, String> = env
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    env.insert("MCP_STUB_ISSUES".into(), "1".into());
    McpServerConfig::new(name, python())
        .with_args([STUB])
        .with_env(env)
}

/// A server whose program doesn't exist.
fn missing(name: &str) -> McpServerConfig {
    McpServerConfig::new(name, "arko-no-such-mcp-server")
}

fn workspace() -> PathBuf {
    std::env::temp_dir()
}

/// A bridge call's text, or its error's.
async fn ask(bridge: &McpBridgeTool, params: Value) -> Result<ToolOutput, String> {
    ask_as(bridge, uuid::Uuid::new_v4(), params).await
}

async fn ask_as(
    bridge: &McpBridgeTool,
    task_id: uuid::Uuid,
    params: Value,
) -> Result<ToolOutput, String> {
    let workspace = workspace();
    let context = TaskContext::new(task_id, "tester", "test", &workspace);
    let params: HashMap<String, Value> = serde_json::from_value(params).unwrap();
    bridge
        .execute_with_output(&context, params)
        .await
        .map_err(|error| match error {
            ToolError::ExecutionFailed(message) => message,
            other => other.to_string(),
        })
}

async fn text(bridge: &McpBridgeTool, params: Value) -> String {
    ask(bridge, params.clone())
        .await
        .unwrap_or_else(|error| panic!("{params} failed: {error}"))
        .text
}

async fn error(bridge: &McpBridgeTool, params: Value) -> String {
    match ask(bridge, params.clone()).await {
        Ok(output) => panic!("{params} succeeded: {}", output.text),
        Err(error) => error,
    }
}

/// A host's adds and removes, recorded, with the hub changed as a host
/// would after the user's approval.
#[derive(Default)]
struct Changes {
    hub: Mutex<Option<Arc<McpHub>>>,
    added: Mutex<Vec<AddServer>>,
    removed: Mutex<Vec<(String, String)>>,
}

#[async_trait]
impl McpChanges for Changes {
    async fn add(&self, _ctx: &TaskContext<'_>, request: AddServer) -> Result<String, String> {
        self.added.lock().unwrap().push(request.clone());
        let hub = self.hub.lock().unwrap().clone().unwrap();
        let mut config = stub(&request.name, &[]);
        config.workers = request.workers;
        config.shared = request.shared;
        config.description = Some(format!("Added by an agent. {}", request.why));
        let facts = hub.add(config).await?;
        Ok(format!("Added {} ({:?}).", facts.name, facts.state))
    }

    async fn remove(
        &self,
        _ctx: &TaskContext<'_>,
        name: &str,
        why: &str,
    ) -> Result<String, String> {
        self.removed
            .lock()
            .unwrap()
            .push((name.to_string(), why.to_string()));
        let hub = self.hub.lock().unwrap().clone().unwrap();
        hub.remove(name).await?;
        Ok(format!("Removed {name}."))
    }
}

/// Gives each task the servers listed for it, and every server to others.
struct ByTask(HashMap<uuid::Uuid, Vec<String>>);

impl McpAccess for ByTask {
    fn servers_for(&self, ctx: &TaskContext<'_>) -> Option<Vec<String>> {
        self.0.get(&ctx.task_id).cloned()
    }
}

fn lead(hub: &Arc<McpHub>) -> (McpBridgeTool, Arc<Changes>) {
    let changes = Arc::new(Changes::default());
    *changes.hub.lock().unwrap() = Some(hub.clone());
    (
        McpBridgeTool::lead(hub.clone(), Arc::new(AllServers), changes.clone()),
        changes,
    )
}

#[tokio::test]
async fn the_hub_starts_stops_adds_and_removes_servers_and_reports_failures() {
    let hub = McpHub::new(vec![stub("issues", &[]), missing("broken")]);
    let facts = hub.servers();
    assert_eq!(facts.len(), 2);
    assert!(
        facts
            .iter()
            .all(|server| server.state == ServerState::Stopped)
    );

    hub.start_all(TIMEOUT).await;
    let facts = hub.servers();
    assert_eq!(facts[0].state, ServerState::Running);
    assert_eq!(facts[0].tools, Some(3));
    assert_eq!(
        facts[0].instructions.as_deref(),
        Some("Tracks issues for tests. Use create_issue to file one.")
    );
    let ServerState::Failed(reason) = &facts[1].state else {
        panic!("{:?}", facts[1].state);
    };
    assert_eq!(reason, "its program wasn't found");
    assert_eq!(
        hub.start("broken").await.unwrap_err(),
        "Couldn't start broken, because its program wasn't found."
    );
    assert_eq!(
        hub.start("nothing").await.unwrap_err(),
        "There's no MCP server named nothing."
    );

    hub.stop("issues").await.unwrap();
    assert_eq!(hub.servers()[0].state, ServerState::Stopped);
    // A call starts a stopped server.
    let workspace = workspace();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let arguments = json!({"text": "hi"}).as_object().unwrap().clone();
    let output = hub
        .call(&context, "issues", "echo", arguments)
        .await
        .unwrap();
    assert_eq!(output.text, "echo: hi");
    assert_eq!(hub.servers()[0].state, ServerState::Running);

    // Adding starts the server; a name taken is replaced.
    let added = hub.add(stub("more", &[])).await.unwrap();
    assert_eq!(added.state, ServerState::Running);
    let mut replacement = stub("more", &[]);
    replacement.description = Some("The second one.".into());
    let replaced = hub.add(replacement).await.unwrap();
    assert_eq!(replaced.description.as_deref(), Some("The second one."));
    let names: Vec<String> = hub.servers().into_iter().map(|facts| facts.name).collect();
    assert_eq!(names, ["issues", "broken", "more"]);
    // A server that doesn't start stays added, failed.
    let failed = hub.add(missing("later")).await.unwrap();
    assert!(matches!(failed.state, ServerState::Failed(_)));
    assert_eq!(
        hub.add(McpServerConfig::new("", "npx")).await.unwrap_err(),
        "An MCP server needs a name."
    );

    hub.remove("more").await.unwrap();
    hub.remove("later").await.unwrap();
    assert_eq!(
        hub.remove("more").await.unwrap_err(),
        "There's no MCP server named more."
    );
    assert_eq!(hub.servers().len(), 2);

    // A server that exits is reported failed, with its exit.
    let exit = json!({"text": "exit"}).as_object().unwrap().clone();
    assert!(hub.call(&context, "issues", "echo", exit).await.is_err());
    let state = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match hub.servers()[0].state.clone() {
                ServerState::Failed(reason) if reason.contains("code 7") => break reason,
                _ => tokio::time::sleep(Duration::from_millis(20)).await,
            }
        }
    })
    .await
    .expect("the exit is reported");
    // What it wrote to stderr goes to the log only.
    assert_eq!(state, "it exited with code 7");
    // The next call starts it again.
    let again = json!({"text": "again"}).as_object().unwrap().clone();
    let output = hub.call(&context, "issues", "echo", again).await.unwrap();
    assert_eq!(output.text, "echo: again");
    hub.shutdown().await;
    assert!(
        hub.servers()
            .iter()
            .all(|server| server.state == ServerState::Stopped || server.name == "broken")
    );
}

const ENV_SECRET: &str = "ghp_fake_env_secret_1234";
const ARG_SECRET: &str = "sk-fake-arg-secret-5678";

/// A server that writes the secrets of its environment and arguments to
/// stderr, then exits with code 3 (or with `stall`, waits a minute).
fn leaky(name: &str, stall: bool) -> McpServerConfig {
    let end = if stall {
        "time.sleep(60)"
    } else {
        "sys.exit(3)"
    };
    let script = format!(
        "import os, sys, time\nsys.stderr.write('token ' + os.environ['LEAKY_TOKEN'] + ' key ' + sys.argv[1] + '\\n')\nsys.stderr.flush()\n{end}"
    );
    McpServerConfig::new(name, python())
        .with_args(["-c".to_string(), script, ARG_SECRET.to_string()])
        .with_env([("LEAKY_TOKEN", ENV_SECRET)])
        .with_expanded([("API_KEY", ARG_SECRET)])
}

/// What an agent reads holds none of a server's secrets, its command or
/// what it wrote.
fn assert_clean(text: &str) {
    for secret in [ENV_SECRET, ARG_SECRET, "python", "LEAKY_TOKEN", "token "] {
        assert!(!text.contains(secret), "{secret} in {text}");
    }
}

/// A failure's reason is the server's name and fixed words, wherever an
/// agent reads it: `servers`, `start` and `call`. What the server wrote, and
/// the program the host expanded, go to the log, with the values of its
/// environment replaced by their names.
#[tokio::test]
async fn failures_show_no_secrets_and_the_log_shows_their_names() {
    crate::test_log::install();
    let mut gone = McpServerConfig::new("gone", format!("/nonexistent/{ARG_SECRET}"));
    gone.expanded = HashMap::from([("API_KEY".into(), ARG_SECRET.into())]);
    let hub = McpHub::new(vec![leaky("leaky", false), gone]);
    hub.start_all(TIMEOUT).await;
    let (bridge, _) = lead(&hub);
    let servers = text(&bridge, json!({"action": "servers"})).await;
    assert_eq!(
        servers,
        [
            "leaky  failed (it exited with code 3). start tries again.",
            "gone   failed (its program wasn't found). start tries again.",
        ]
        .join("\n")
    );
    let started = error(&bridge, json!({"action": "start", "server": "leaky"})).await;
    assert_eq!(
        started,
        "Couldn't start leaky, because it exited with code 3."
    );
    let called = error(
        &bridge,
        json!({"action": "call", "server": "leaky", "tool": "echo"}),
    )
    .await;
    assert_eq!(called, started);
    let workspace = workspace();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let direct = hub
        .call(&context, "gone", "echo", serde_json::Map::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        direct.contains("Couldn't start gone, because its program wasn't found."),
        "{direct}"
    );
    for text in [&servers, &started, &called, &direct] {
        assert_clean(text);
    }
    for facts in hub.servers() {
        assert_clean(&format!("{facts:?}"));
    }

    let logged = crate::test_log::logged(log::Level::Info, "MCP server '");
    let leaked: Vec<&String> = logged
        .iter()
        .filter(|line| line.contains(ENV_SECRET) || line.contains(ARG_SECRET))
        .collect();
    assert!(leaked.is_empty(), "{leaked:?}");
    assert!(
        logged
            .iter()
            .any(|line| line == "MCP server 'leaky' stderr: token ${LEAKY_TOKEN} key ${API_KEY}"),
        "{logged:#?}"
    );
    assert!(
        logged
            .iter()
            .any(|line| line.starts_with("MCP server 'gone': ${API_KEY} couldn't be started: ")),
        "{logged:#?}"
    );
    hub.shutdown().await;
}

/// A server that doesn't answer in time is said to be so, also without
/// what it wrote.
#[tokio::test]
async fn a_server_that_doesnt_answer_in_time_shows_no_secrets() {
    let hub = McpHub::new(vec![leaky("slow", true)]);
    hub.start_all(Duration::from_secs(1)).await;
    let (bridge, _) = lead(&hub);
    let servers = text(&bridge, json!({"action": "servers"})).await;
    assert_eq!(
        servers,
        "slow  failed (it didn't start within 1 s). start tries again."
    );
    let started = error(&bridge, json!({"action": "start", "server": "slow"})).await;
    assert_eq!(
        started,
        "Couldn't start slow, because it didn't start within 1 s."
    );
    assert_clean(&servers);
    assert_clean(&started);
    hub.shutdown().await;
}

#[tokio::test]
async fn tools_asks_the_server_each_time() {
    let hub = McpHub::new(vec![stub("growing", &[("MCP_STUB_GROWS", "1")])]);
    hub.start_all(TIMEOUT).await;
    let first = hub.tools("growing").await.unwrap();
    let second = hub.tools("growing").await.unwrap();
    assert_eq!(second.len(), first.len() + 1);
    assert_eq!(hub.listed_tools("growing").unwrap().len(), second.len());
    hub.shutdown().await;
}

#[tokio::test]
async fn servers_lists_each_server_with_its_state_and_what_it_is_for() {
    let mut quiet = stub("quiet", &[]);
    quiet.env.remove("MCP_STUB_ISSUES");
    let mut kept = stub("kept", &[]);
    kept.workers = false;
    kept.description = Some("Added by an agent to read this project's errors.".into());
    let hub = McpHub::new(vec![stub("issues", &[]), quiet, kept, missing("figma")]);
    hub.start_all(TIMEOUT).await;
    hub.stop("quiet").await.unwrap();
    let (bridge, _) = lead(&hub);
    assert_eq!(
        text(&bridge, json!({"action": "servers"})).await,
        [
            "issues  running, 3 tools  Tracks issues for tests.",
            "quiet   stopped",
            "kept    running, 3 tools  Added by an agent to read this project's errors. Main agent only.",
            "figma   failed (its program wasn't found). start tries again.",
        ]
        .join("\n")
    );
    // A worker doesn't see the server kept to the main agent.
    let worker = McpBridgeTool::worker(hub.clone(), Arc::new(AllServers));
    let lines = text(&worker, json!({"action": "servers"})).await;
    assert!(!lines.contains("kept"), "{lines}");
    assert!(
        lines.starts_with("issues  running, 3 tools  Tracks issues"),
        "{lines}"
    );

    let empty = McpHub::new(Vec::new());
    let (bridge, _) = lead(&empty);
    assert_eq!(
        text(&bridge, json!({"action": "servers"})).await,
        "No MCP servers are set up. add sets one up once the user approves it."
    );
    let worker = McpBridgeTool::worker(empty, Arc::new(AllServers));
    assert_eq!(
        text(&worker, json!({"action": "servers"})).await,
        "You have no MCP servers."
    );
    hub.shutdown().await;
}

#[tokio::test]
async fn tools_lists_a_line_per_tool_with_a_filter_schemas_and_a_cap() {
    let hub = McpHub::new(vec![
        stub("issues", &[]),
        stub("big", &[("MCP_STUB_EXTRA_TOOLS", "70")]),
    ]);
    hub.start_all(TIMEOUT).await;
    let (bridge, _) = lead(&hub);
    assert_eq!(
        text(&bridge, json!({"action": "tools", "server": "issues"})).await,
        [
            "echo(text?)  Echoes text.",
            "write_note(text?)  Writes a note.",
            "create_issue(title, body?, labels?)  Creates an issue in the tracker.",
        ]
        .join("\n")
    );
    // Every word, in any case, in the name or the description.
    assert_eq!(
        text(
            &bridge,
            json!({"action": "tools", "server": "issues", "filter": "ISSUE tracker"})
        )
        .await,
        "create_issue(title, body?, labels?)  Creates an issue in the tracker."
    );
    assert_eq!(
        text(
            &bridge,
            json!({"action": "tools", "server": "issues", "filter": "deploy"})
        )
        .await,
        "None of the 3 tools of issues match “deploy”."
    );
    // The schemas asked for follow the lines.
    let with_schema = text(
        &bridge,
        json!({"action": "tools", "server": "issues", "filter": "note", "schema": ["create_issue", "create_isue"]}),
    )
    .await;
    let lines: Vec<&str> = with_schema.lines().collect();
    assert_eq!(lines[0], "write_note(text?)  Writes a note.");
    assert_eq!(lines[1], "");
    assert_eq!(lines[2], "Input schema of create_issue");
    let schema: Value = serde_json::from_str(lines[3]).unwrap();
    assert_eq!(schema["required"], json!(["title"]));
    assert_eq!(lines[4], "");
    assert_eq!(
        lines[5],
        "issues has no tool named create_isue. The closest are create_issue, write_note and echo."
    );

    // At most 60 lines.
    let many = text(&bridge, json!({"action": "tools", "server": "big"})).await;
    let lines: Vec<&str> = many.lines().collect();
    assert_eq!(lines.len(), 61);
    assert_eq!(lines[60], "13 more. Narrow them with filter.");
    let narrowed = text(
        &bridge,
        json!({"action": "tools", "server": "big", "filter": "extra_06"}),
    )
    .await;
    assert_eq!(narrowed.lines().count(), 10, "{narrowed}");
    hub.shutdown().await;
}

#[tokio::test]
async fn call_passes_text_images_resources_and_errors_through() {
    let hub = McpHub::new(vec![stub("issues", &[])]);
    hub.start_all(TIMEOUT).await;
    let (bridge, _) = lead(&hub);
    let call = |text: &str| json!({"action": "call", "server": "issues", "tool": "echo", "arguments": {"text": text}});
    assert_eq!(text(&bridge, call("hello")).await, "echo: hello");
    let mixed = ask(&bridge, call("mixed-images")).await.unwrap();
    assert_eq!(
        mixed.text,
        "before images\nResource file:///note.txt:\nresource text\nafter images"
    );
    assert_eq!(mixed.images.len(), 4);
    assert_eq!(mixed.images[1].description, "file:///tiny.gif");
    let structured = text(&bridge, call("structured-only")).await;
    assert_eq!(
        serde_json::from_str::<Value>(&structured).unwrap(),
        json!({"answer": 42})
    );
    assert_eq!(
        error(&bridge, call("error-with-image")).await,
        "render failed"
    );
    // Arguments written as JSON text are read too.
    assert_eq!(
        text(&bridge, json!({"action": "call", "server": "issues", "tool": "echo", "arguments": "{\"text\": \"as text\"}"})).await,
        "echo: as text"
    );
    assert_eq!(
        error(
            &bridge,
            json!({"action": "call", "server": "issues", "tool": "echo", "arguments": "[1]"})
        )
        .await,
        "arguments takes a JSON object, as the tool's schema describes."
    );
    hub.shutdown().await;
}

#[tokio::test]
async fn the_bridge_names_what_to_do_next_when_a_call_goes_wrong() {
    let hub = McpHub::new(vec![stub("issues", &[]), missing("figma")]);
    hub.start_all(TIMEOUT).await;
    let (bridge, _) = lead(&hub);
    assert_eq!(
        error(
            &bridge,
            json!({"action": "call", "server": "github", "tool": "x"})
        )
        .await,
        "There's no MCP server named github. Your servers are issues and figma."
    );
    assert_eq!(
        error(&bridge, json!({"action": "tools"})).await,
        "Name the server in server. Your servers are issues and figma."
    );
    assert_eq!(
        error(
            &bridge,
            json!({"action": "call", "server": "issues", "tool": "creat_issue"})
        )
        .await,
        "issues has no tool named creat_issue. The closest are create_issue, write_note and echo."
    );
    assert_eq!(
        error(&bridge, json!({"action": "call", "server": "issues"})).await,
        "Name the tool in tool, as tools lists it."
    );
    assert_eq!(
        error(&bridge, json!({"action": "start", "server": "figma"})).await,
        "Couldn't start figma, because its program wasn't found."
    );
    assert_eq!(
        text(&bridge, json!({"action": "start", "server": "issues"})).await,
        "issues is running, with 3 tools."
    );
    assert_eq!(
        text(&bridge, json!({"action": "stop", "server": "issues"})).await,
        "Stopped issues."
    );
    assert_eq!(
        text(&bridge, json!({"action": "start", "server": "issues"})).await,
        "Started issues with 3 tools."
    );

    // A missing argument, or arguments the server rejects, bring the
    // tool's schema.
    let missing_title = error(
        &bridge,
        json!({"action": "call", "server": "issues", "tool": "create_issue", "arguments": {"body": "b"}}),
    )
    .await;
    let (said, schema) = missing_title.split_once('\n').unwrap();
    assert_eq!(
        said,
        "create_issue needs title in arguments. Its input schema follows."
    );
    assert_eq!(
        serde_json::from_str::<Value>(schema).unwrap()["required"],
        json!(["title"])
    );
    let rejected = error(
        &bridge,
        json!({"action": "call", "server": "issues", "tool": "create_issue", "arguments": {"title": 5}}),
    )
    .await;
    let (said, schema) = rejected.split_once('\n').unwrap();
    assert_eq!(
        said,
        "issues rejected the arguments of create_issue (title must be text). Its input schema follows."
    );
    assert!(serde_json::from_str::<Value>(schema).is_ok());
    assert_eq!(
        text(&bridge, json!({"action": "call", "server": "issues", "tool": "create_issue", "arguments": {"title": "Bug"}})).await,
        "Created issue 1: Bug"
    );

    assert_eq!(
        error(&bridge, json!({"action": "launch"})).await,
        "There's no action launch. Use one of servers, tools, call, start, stop, add and remove."
    );
    assert_eq!(
        error(&bridge, json!({})).await,
        "Give an action, one of servers, tools, call, start, stop, add and remove."
    );
    hub.shutdown().await;
}

#[tokio::test]
async fn access_decides_the_servers_a_run_reaches() {
    let mut kept = stub("kept", &[]);
    kept.workers = false;
    let hub = McpHub::new(vec![stub("issues", &[]), stub("other", &[]), kept]);
    hub.start_all(TIMEOUT).await;
    let (narrow, wide) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
    let access = Arc::new(ByTask(HashMap::from([(
        narrow,
        vec!["issues".to_string()],
    )])));
    let worker = McpBridgeTool::worker(hub.clone(), access);
    let servers = |task| {
        let worker = &worker;
        async move {
            ask_as(worker, task, json!({"action": "servers"}))
                .await
                .unwrap()
                .text
        }
    };
    assert_eq!(servers(narrow).await.lines().count(), 1);
    // `None` is every server, but never one kept to the main agent.
    assert_eq!(servers(wide).await.lines().count(), 2);
    assert_eq!(
        ask_as(
            &worker,
            narrow,
            json!({"action": "call", "server": "other", "tool": "echo"})
        )
        .await
        .unwrap_err(),
        "other isn't one of your MCP servers. Your servers are issues."
    );
    assert_eq!(
        ask_as(&worker, wide, json!({"action": "tools", "server": "kept"}))
            .await
            .unwrap_err(),
        "kept isn't one of your MCP servers. Your servers are issues and other."
    );
    assert_eq!(
        ask_as(
            &worker,
            narrow,
            json!({"action": "stop", "server": "issues"})
        )
        .await
        .unwrap_err(),
        "Only the main agent can stop MCP servers. Your actions are servers, tools, call and start."
    );
    hub.shutdown().await;
}

#[tokio::test]
async fn add_and_remove_go_through_the_hosts_changes() {
    let hub = McpHub::new(Vec::new());
    let (bridge, changes) = lead(&hub);
    let added = text(
        &bridge,
        json!({
            "action": "add", "server": "tracker", "command": "npx",
            "args": ["-y", "tracker"], "env": ["TRACKER_TOKEN"], "scope": "user",
            "workers": false, "shared": true, "why": "To file issues."
        }),
    )
    .await;
    assert_eq!(added, "Added tracker (Running).");
    let mut expected = AddServer::new("tracker", "npx");
    expected.args = vec!["-y".into(), "tracker".into()];
    expected.env = vec!["TRACKER_TOKEN".into()];
    expected.scope = McpScope::User;
    expected.workers = false;
    expected.shared = true;
    expected.why = "To file issues.".into();
    assert_eq!(changes.added.lock().unwrap().clone(), [expected]);
    // The server is usable at once.
    assert!(
        text(&bridge, json!({"action": "servers"}))
            .await
            .starts_with(
                "tracker  running, 3 tools  Added by an agent. To file issues. Main agent only."
            )
    );
    // The defaults, and output-based text for the lists and flags.
    text(
        &bridge,
        json!({"action": "add", "server": "plain", "command": "uvx", "args": "a, b", "workers": "FALSE"}),
    )
    .await;
    let plain = changes.added.lock().unwrap()[1].clone();
    assert_eq!(plain.scope, McpScope::Project);
    assert_eq!(plain.args, ["a", "b"]);
    assert!(!plain.workers && !plain.shared);
    assert_eq!(
        error(
            &bridge,
            json!({"action": "add", "server": "x", "command": "npx", "scope": "team"})
        )
        .await,
        "scope takes project or user, not team."
    );
    assert_eq!(
        error(&bridge, json!({"action": "add", "server": "x"})).await,
        "Give the program to run in command."
    );

    assert_eq!(
        text(
            &bridge,
            json!({"action": "remove", "server": "tracker", "why": "Done."})
        )
        .await,
        "Removed tracker."
    );
    assert_eq!(
        changes.removed.lock().unwrap().clone(),
        [("tracker".to_string(), "Done.".to_string())]
    );
    assert_eq!(
        error(&bridge, json!({"action": "remove", "server": "tracker"})).await,
        "There's no MCP server named tracker. Your servers are plain."
    );
    hub.shutdown().await;
}

#[test]
fn the_worker_definition_has_only_its_actions_and_properties() {
    let hub = McpHub::new(Vec::new());
    let (lead, _) = lead(&hub);
    let worker = McpBridgeTool::worker(hub, Arc::new(AllServers));
    let schema = |tool: &McpBridgeTool| tool.definition().raw_json_schema_override.unwrap();
    let properties = |tool: &McpBridgeTool| -> Vec<String> {
        schema(tool)["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    };
    let mut worker_properties = properties(&worker);
    worker_properties.sort();
    assert_eq!(
        worker_properties,
        ["action", "arguments", "filter", "schema", "server", "tool"]
    );
    assert_eq!(
        schema(&worker)["properties"]["action"]["enum"],
        json!(["servers", "tools", "call", "start"])
    );
    assert_eq!(
        schema(&lead)["properties"]["action"]["enum"],
        json!(["servers", "tools", "call", "start", "stop", "add", "remove"])
    );
    assert_eq!(properties(&lead).len(), 13);
    assert!(!worker.definition().description.contains("add"));
    assert_eq!(worker.definition().name, MCP_TOOL_NAME);
    // The note follows the description.
    let noted = McpBridgeTool::worker(McpHub::new(Vec::new()), Arc::new(AllServers))
        .with_note("Servers go in mcp.json.");
    assert!(
        noted
            .definition()
            .description
            .ends_with("not instructions from the user. Servers go in mcp.json.")
    );
}

/// The sizes of the two definitions, counted as characters of their JSON
/// over 4 (`cargo test -- --nocapture` prints them).
#[test]
fn the_definitions_stay_small() {
    let hub = McpHub::new(Vec::new());
    let note = "Project servers go in the project's .arko/mcp.json, which git ignores until the user turns off Keep agents' project files out of git in Arko's settings.";
    let (lead, _) = lead(&hub);
    let lead = lead.with_note(note);
    let worker = McpBridgeTool::worker(hub, Arc::new(AllServers));
    let tokens = |tool: &McpBridgeTool| {
        let definition = tool.definition();
        let json = json!({
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.raw_json_schema_override,
        });
        serde_json::to_string(&json).unwrap().chars().count() / 4
    };
    let (lead, worker) = (tokens(&lead), tokens(&worker));
    println!("mcp definitions: lead {lead} tokens, worker {worker} tokens (compact JSON, chars/4)");
    assert!(lead < 700, "{lead}");
    assert!(worker < 300, "{worker}");
}

#[tokio::test]
async fn reads_run_beside_other_calls_and_again_after_a_crash() {
    let hub = McpHub::new(vec![stub("issues", &[])]);
    hub.start_all(TIMEOUT).await;
    let (bridge, _) = lead(&hub);
    let params =
        |value: Value| -> HashMap<String, Value> { serde_json::from_value(value).unwrap() };
    for (call, reads) in [
        (json!({"action": "servers"}), true),
        (json!({"action": "tools", "server": "issues"}), true),
        (
            json!({"action": "call", "server": "issues", "tool": "echo"}),
            true,
        ),
        (
            json!({"action": "call", "server": "issues", "tool": "write_note"}),
            false,
        ),
        (
            json!({"action": "call", "server": "issues", "tool": "nothing"}),
            false,
        ),
        (
            json!({"action": "call", "server": "nothing", "tool": "echo"}),
            false,
        ),
        (json!({"action": "start", "server": "issues"}), false),
        (json!({"action": "stop", "server": "issues"}), false),
        (json!({"action": "add", "server": "x"}), false),
        (json!({"action": "remove", "server": "x"}), false),
    ] {
        let params = params(call.clone());
        assert_eq!(bridge.parallel_safe_for(&params), Some(reads), "{call}");
        let mode = if reads {
            ToolRecovery::Rerun
        } else {
            ToolRecovery::InDoubt
        };
        assert_eq!(bridge.recovery_mode_for(&params), mode, "{call}");
    }
    assert!(!bridge.definition().parallel_safe);
    hub.shutdown().await;
}

#[tokio::test]
async fn direct_tools_keep_their_names_and_call_through_the_hub() {
    let hub = McpHub::new(vec![stub("issues", &[]), missing("figma")]);
    hub.start_all(TIMEOUT).await;
    let direct = hub.direct_tools(Arc::new(AllServers));
    let names: Vec<String> = direct.iter().map(|tool| tool.name()).collect();
    assert_eq!(
        names,
        [
            "mcp__issues__echo",
            "mcp__issues__write_note",
            "mcp__issues__create_issue"
        ]
    );
    let echo = &direct[0];
    let definition = echo.definition();
    assert_eq!(definition.description, "[MCP:issues] Echoes text.");
    assert!(definition.parallel_safe);
    assert_eq!(echo.recovery_mode(), ToolRecovery::Rerun);
    assert_eq!(direct[1].recovery_mode(), ToolRecovery::InDoubt);
    assert_eq!(echo.background_mode(), crate::tool::ToolBackground::Detach);
    // A stopped server is started by a call.
    hub.stop("issues").await.unwrap();
    let workspace = workspace();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let output = echo
        .execute(&context, HashMap::from([("text".into(), json!("direct"))]))
        .await
        .unwrap();
    assert_eq!(output, "echo: direct");
    hub.shutdown().await;
}

/// A direct tool reaches only the servers its run may reach, as the bridge
/// does, when it's called: the access's servers, and for a worker never one
/// kept to the main agent. The definitions stay those the conversation was
/// saved with.
#[tokio::test]
async fn direct_tools_check_the_runs_servers_when_called() {
    let mut kept = stub("kept", &[]);
    kept.workers = false;
    let hub = McpHub::new(vec![stub("issues", &[]), stub("other", &[]), kept]);
    hub.start_all(TIMEOUT).await;
    let (narrow, wide, lead) = (
        uuid::Uuid::new_v4(),
        uuid::Uuid::new_v4(),
        uuid::Uuid::new_v4(),
    );
    let access = Arc::new(ByTask(HashMap::from([
        (narrow, vec!["issues".to_string()]),
        (wide, vec!["issues".into(), "other".into(), "kept".into()]),
    ])));
    let main = hub.direct_tools(access.clone());
    let workers = hub.worker_direct_tools(access);
    let definitions = |tools: &[Arc<dyn Tool>]| -> Vec<String> {
        tools
            .iter()
            .map(|tool| serde_json::to_string(&tool.definition()).unwrap())
            .collect()
    };
    assert_eq!(definitions(&main), definitions(&workers));
    assert_eq!(main.len(), 9);
    let echo = |tools: &[Arc<dyn Tool>], server: &str| {
        let name = direct_tool_name(server, "echo");
        tools
            .iter()
            .find(|tool| tool.name() == name)
            .unwrap()
            .clone()
    };
    let call = |tool: Arc<dyn Tool>, task: uuid::Uuid| async move {
        let workspace = workspace();
        let context = TaskContext::new(task, "tester", "test", &workspace);
        tool.execute(&context, HashMap::from([("text".into(), json!("hi"))]))
            .await
            .map_err(|error| error.to_string())
    };

    // The main agent's run has no list, so it reaches every server.
    assert_eq!(call(echo(&main, "kept"), lead).await.unwrap(), "echo: hi");
    assert_eq!(
        call(echo(&main, "issues"), narrow).await.unwrap(),
        "echo: hi"
    );
    let refused = call(echo(&main, "other"), narrow).await.unwrap_err();
    assert!(
        refused.contains("other isn't one of your MCP servers."),
        "{refused}"
    );
    // A worker's never reaches a server kept to the main agent, even when
    // its list names it.
    assert_eq!(
        call(echo(&workers, "other"), wide).await.unwrap(),
        "echo: hi"
    );
    let refused = call(echo(&workers, "kept"), wide).await.unwrap_err();
    assert!(
        refused.contains("kept isn't one of your MCP servers."),
        "{refused}"
    );
    let refused = call(echo(&workers, "kept"), lead).await.unwrap_err();
    assert!(refused.contains("kept isn't one"), "{refused}");
    // The config when the call is made counts.
    let mut now_kept = stub("other", &[]);
    now_kept.workers = false;
    hub.add(now_kept).await.unwrap();
    let refused = call(echo(&workers, "other"), wide).await.unwrap_err();
    assert!(refused.contains("other isn't one"), "{refused}");
    assert_eq!(call(echo(&main, "other"), wide).await.unwrap(), "echo: hi");
    hub.shutdown().await;
}

#[test]
fn direct_tool_names_are_ones_the_providers_take() {
    assert_eq!(
        direct_tool_name("github", "create_issue"),
        "mcp__github__create_issue"
    );
    assert_eq!(
        direct_tool_name("my server", "files.read"),
        "mcp__my_server__files_read"
    );
    let long = direct_tool_name(
        "a-server-with-a-long-name",
        "a_tool_with_an_even_longer_name_than_that",
    );
    assert_eq!(long.len(), 64);
    // The same names as before the hub, which conversations saved.
    assert_eq!(
        long,
        "mcp__a-server-with-a-long-name__a_tool_with_an_even_lon_2f27f3a8"
    );
    assert_ne!(
        long,
        direct_tool_name(
            "a-server-with-a-long-name",
            "a_tool_with_an_even_longer_name_than_this"
        )
    );
}

/// The processes the stub started with `pid_file`, in order.
#[cfg(unix)]
fn pids(pid_file: &std::path::Path) -> Vec<i32> {
    std::fs::read_to_string(pid_file)
        .unwrap_or_default()
        .lines()
        .map(|pid| pid.trim().parse().unwrap())
        .collect()
}

/// How many of them run.
#[cfg(unix)]
fn running(pid_file: &std::path::Path) -> usize {
    pids(pid_file)
        .into_iter()
        .filter(|pid| !super::manager::tests::has_ended(*pid))
        .count()
}

/// Waits until every one of them has ended (and kills any left when it
/// fails).
#[cfg(unix)]
async fn all_end(pid_file: &std::path::Path) {
    for pid in pids(pid_file) {
        super::manager::tests::Program(pid).ends().await;
    }
    let _ = std::fs::remove_file(pid_file);
}

#[cfg(unix)]
fn pid_file() -> PathBuf {
    std::env::temp_dir().join(format!("arko-mcp-shared-{}.pid", uuid::Uuid::new_v4()))
}

#[cfg(unix)]
#[tokio::test]
async fn a_shared_server_runs_once_for_every_hub_and_stops_with_the_last() {
    let pid_file = pid_file();
    let mut shared = stub(
        "files",
        &[("MCP_STUB_PID_FILE", &pid_file.display().to_string())],
    );
    shared.shared = true;
    let workspace = workspace();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let echo = |text: &str| json!({"text": text}).as_object().unwrap().clone();

    // The first hub runs on a runtime of its own, which ends while the
    // second hub still uses the server.
    let starting = shared.clone();
    let (first_runtime, first) = std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let hub = runtime.block_on(async {
            let hub = McpHub::new(vec![starting]);
            hub.start_all(TIMEOUT).await;
            hub
        });
        (runtime, hub)
    })
    .join()
    .unwrap();
    let mut renamed = shared.clone();
    renamed.name = "same-files".into();
    let second = McpHub::new(vec![renamed.clone()]);
    second.start_all(TIMEOUT).await;
    assert_eq!(pids(&pid_file).len(), 1, "one process for both hubs");
    assert_eq!(pool::users(&shared), 2);
    std::thread::spawn(move || drop(first_runtime))
        .join()
        .unwrap();
    let output = second
        .call(&context, "same-files", "echo", echo("still"))
        .await
        .unwrap();
    assert_eq!(output.text, "echo: still");

    // Stopping it in one hub stops only that hub's use.
    let (bridge, _) = lead(&second);
    assert_eq!(
        text(&bridge, json!({"action": "stop", "server": "same-files"})).await,
        "Stopped same-files here. Another session shares it, so it keeps running there."
    );
    drop(bridge);
    assert_eq!(second.servers()[0].state, ServerState::Stopped);
    assert_eq!(pool::users(&shared), 1);
    assert_eq!(running(&pid_file), 1);
    second.start("same-files").await.unwrap();
    assert_eq!(pids(&pid_file).len(), 1);

    // Dropping one hub keeps it; dropping both stops it.
    drop(first);
    let output = second
        .call(&context, "same-files", "echo", echo("kept"))
        .await
        .unwrap();
    assert_eq!(output.text, "echo: kept");
    assert_eq!(running(&pid_file), 1);
    drop(second);
    all_end(&pid_file).await;
    assert_eq!(pool::users(&shared), 0);
}

#[cfg(unix)]
#[tokio::test]
async fn a_server_not_shared_runs_once_per_hub() {
    let pid_file = pid_file();
    let own = stub(
        "files",
        &[("MCP_STUB_PID_FILE", &pid_file.display().to_string())],
    );
    let first = McpHub::new(vec![own.clone()]);
    let second = McpHub::new(vec![own]);
    first.start_all(TIMEOUT).await;
    second.start_all(TIMEOUT).await;
    assert_eq!(running(&pid_file), 2);
    first.shutdown().await;
    assert_eq!(running(&pid_file), 1);
    drop(second);
    all_end(&pid_file).await;
}

/// A host whose own environment or working directory differs gets a shared
/// server of its own, and the process starts with exactly the environment
/// it was taken with. The same environment gets the same process.
#[cfg(unix)]
#[tokio::test]
async fn a_shared_server_is_one_process_per_host_environment() {
    use std::ffi::OsString;
    let (first_file, second_file) = (pid_file(), pid_file());
    let mut shared = stub("files", &[]);
    shared.shared = true;
    // The pid file is named by the host's environment only, as an inherited
    // DATABASE_URL would be, so a process that writes to it got that
    // environment.
    let host = |file: &std::path::Path| -> Vec<(OsString, OsString)> {
        std::env::vars_os()
            .chain([("MCP_STUB_PID_FILE".into(), file.into())])
            .collect()
    };
    let dir = std::env::current_dir().ok();
    let first = pool::acquire_in(&shared, host(&first_file), dir.clone());
    let same = pool::acquire_in(&shared, host(&first_file), dir.clone());
    let other = pool::acquire_in(&shared, host(&second_file), dir.clone());
    assert!(Arc::ptr_eq(&first, &same));
    assert!(!Arc::ptr_eq(&first, &other));
    let elsewhere = pool::acquire_in(&shared, host(&first_file), Some(workspace().join("..")));
    assert!(!Arc::ptr_eq(&first, &elsewhere));
    drop(elsewhere);

    first.start(TIMEOUT).await.unwrap();
    same.start(TIMEOUT).await.unwrap();
    other.start(TIMEOUT).await.unwrap();
    assert_eq!(running(&first_file), 1);
    assert_eq!(running(&second_file), 1);
    // A config's env goes over the host's, and the environment the process
    // gets is what counts, wherever its values came from.
    let mut over = shared.clone();
    over.env.insert(
        "MCP_STUB_PID_FILE".into(),
        second_file.display().to_string(),
    );
    let overridden = pool::acquire_in(&over, host(&first_file), dir);
    assert!(Arc::ptr_eq(&overridden, &other));
    let mut fresh = over.clone();
    fresh.env.insert("MCP_STUB_FRESH".into(), "1".into());
    let fresh = pool::acquire_in(&fresh, host(&first_file), std::env::current_dir().ok());
    fresh.start(TIMEOUT).await.unwrap();
    assert_eq!(running(&first_file), 1);
    assert_eq!(running(&second_file), 2);
    drop((first, same, other, overridden, fresh));
    all_end(&first_file).await;
    all_end(&second_file).await;
}

/// A shared server's start that is cancelled while the server is silent
/// takes its process with it, long before the start would time out.
#[cfg(unix)]
#[tokio::test]
async fn a_cancelled_shared_start_ends_its_process() {
    let pid_file = pid_file();
    let mut silent = McpServerConfig::new("silent", "sh").with_args([
        "-c".to_string(),
        "echo $$ >> \"$0\"; exec sleep 300".to_string(),
        pid_file.display().to_string(),
    ]);
    silent.shared = true;
    let hub = McpHub::new(vec![silent]);
    let starting = tokio::spawn({
        let hub = hub.clone();
        async move { hub.start("silent").await }
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while pids(&pid_file).is_empty() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the server starts");
    starting.abort();
    assert!(starting.await.unwrap_err().is_cancelled());
    drop(hub);
    // The start would wait 60 s for the handshake.
    all_end(&pid_file).await;
}

/// A shared server's process that ends is started again by the next call,
/// for every hub.
#[cfg(unix)]
#[tokio::test]
async fn a_shared_server_that_exits_is_started_again() {
    let pid_file = pid_file();
    let mut shared = stub(
        "files",
        &[("MCP_STUB_PID_FILE", &pid_file.display().to_string())],
    );
    shared.shared = true;
    let workspace = workspace();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let echo = |text: &str| json!({"text": text}).as_object().unwrap().clone();
    let first = McpHub::new(vec![shared.clone()]);
    let second = McpHub::new(vec![shared.clone()]);
    first.start_all(TIMEOUT).await;
    second.start_all(TIMEOUT).await;
    assert!(
        first
            .call(&context, "files", "echo", echo("exit"))
            .await
            .is_err()
    );
    let exited = pids(&pid_file)[0];
    super::manager::tests::Program(exited).ends().await;
    let output = second
        .call(&context, "files", "echo", echo("back"))
        .await
        .unwrap();
    assert_eq!(output.text, "echo: back");
    let output = first
        .call(&context, "files", "echo", echo("here"))
        .await
        .unwrap();
    assert_eq!(output.text, "echo: here");
    assert_eq!(pids(&pid_file).len(), 2, "one more process, shared");
    assert_eq!(running(&pid_file), 1);
    drop((first, second));
    all_end(&pid_file).await;
}
