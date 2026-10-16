use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::*;

/// Options that run the fake CLI in `tests/fixtures/fake_claude.py`.
fn fake() -> ClaudeOptions {
    let python = std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_claude.py");
    ClaudeOptions {
        cli_path: Some(python.into()),
        cli_prefix_args: vec![script.display().to_string()],
        ..ClaudeOptions::default()
    }
}

/// The final text of a turn, parsed as JSON.
fn result_json(turn: &Turn) -> Value {
    serde_json::from_str(turn.result.result.as_deref().unwrap()).unwrap()
}

struct Decide(PermissionDecision);

#[async_trait]
impl PermissionHandler for Decide {
    async fn decide(&self, request: PermissionRequest) -> PermissionDecision {
        assert_eq!(request.tool_name, "Bash");
        assert_eq!(request.tool_use_id.as_deref(), Some("t1"));
        self.0.clone()
    }
}

struct DenyDangerous;

#[async_trait]
impl HookCallback for DenyDangerous {
    async fn call(&self, input: Value, tool_use_id: Option<String>) -> Value {
        assert_eq!(tool_use_id.as_deref(), Some("t2"));
        if input["tool_input"]["command"] == "rm -rf ~" {
            Hook::deny_tool("that would delete the home directory")
        } else {
            Hook::proceed()
        }
    }
}

struct Add;

#[async_trait]
impl ClaudeTool for Add {
    fn name(&self) -> String {
        "add".into()
    }
    fn description(&self) -> String {
        "Adds two numbers.".into()
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}})
    }
    async fn call(&self, arguments: Value) -> ToolOutput {
        let sum = arguments["a"].as_f64().unwrap_or(0.0) + arguments["b"].as_f64().unwrap_or(0.0);
        ToolOutput::text(sum.to_string())
    }
}

#[test]
fn options_become_cli_arguments() {
    let options = ClaudeOptions {
        model: Some("opus".into()),
        effort: Some(Effort::XHigh),
        system_prompt: SystemPrompt::Append("Be brief.".into()),
        built_in_tools: BuiltInTools::None,
        allowed_tools: vec!["Read".into(), "mcp__math__add".into()],
        permission_mode: Some(PermissionMode::AcceptEdits),
        session: SessionStart::Resume {
            session: "abc".into(),
            fork: true,
            at: Some("m1".into()),
        },
        permission_handler: Some(Arc::new(Decide(PermissionDecision::allow()))),
        tool_servers: vec![Arc::new(ToolServer::new("math").tool(Arc::new(Add)))],
        ..ClaudeOptions::default()
    };
    let args = options.arguments();
    let joined = args.join(" ");
    for expected in [
        "--output-format stream-json --verbose --input-format stream-json",
        "--model opus",
        "--effort xhigh",
        "--append-system-prompt Be brief.",
        "--allowedTools Read,mcp__math__add",
        "--permission-prompt-tool stdio",
        "--permission-mode acceptEdits",
        "--resume=abc --fork-session --resume-session-at=m1",
    ] {
        assert!(
            joined.contains(expected),
            "{expected:?} missing from {joined}"
        );
    }
    let tools = args.iter().position(|arg| arg == "--tools").unwrap();
    assert_eq!(args[tools + 1], "");
    let mcp = args.iter().position(|arg| arg == "--mcp-config").unwrap();
    let config: Value = serde_json::from_str(&args[mcp + 1]).unwrap();
    assert_eq!(
        config["mcpServers"]["math"],
        json!({"type": "sdk", "name": "math"})
    );
    assert_eq!(
        ToolServer::new("math").qualified_name("add"),
        "mcp__math__add"
    );
    assert!(!joined.contains("--system-prompt-snapshot"), "{joined}");
    let fork = ClaudeOptions {
        session: SessionStart::Fork {
            session: "abc".into(),
            at: Some("m1".into()),
            new_session: "def".into(),
        },
        ..ClaudeOptions::default()
    };
    assert!(
        fork.arguments()
            .join(" ")
            .contains("--resume=abc --fork-session --session-id=def --resume-session-at=m1")
    );
    let unrecorded = ClaudeOptions {
        system_prompt_snapshot: Some(false),
        ..ClaudeOptions::default()
    };
    assert!(
        unrecorded
            .arguments()
            .join(" ")
            .contains("--system-prompt-snapshot off")
    );
}

#[tokio::test]
async fn a_session_initializes_and_runs_turns() {
    let hook = Hook::new(HookEvent::PreToolUse, Some("Bash"), Arc::new(DenyDangerous));
    let mut session = ClaudeSession::start(ClaudeOptions {
        model: Some("sonnet".into()),
        hooks: vec![hook],
        ..fake()
    })
    .await
    .unwrap();
    let initialization = session.initialization();
    assert_eq!(
        initialization["hooks"]["PreToolUse"][0],
        json!({"matcher": "Bash", "hookCallbackIds": ["hook_0"]})
    );
    assert!(
        initialization["argv"]
            .to_string()
            .contains("\"--model\",\"sonnet\"")
    );

    session.send("hello").await.unwrap();
    let turn = session.receive_turn().await.unwrap();
    assert_eq!(turn.result.result.as_deref(), Some("echo:hello:model=None"));
    assert!(!turn.result.is_error);
    assert_eq!(session.session_id().as_deref(), Some("sess-1"));
    assert!(matches!(&turn.messages[0], Message::System(system) if system.subtype == "init"));
    let Message::Assistant(assistant) = &turn.messages[1] else {
        panic!("expected an assistant message: {:?}", turn.messages);
    };
    assert_eq!(
        assistant.content,
        vec![ContentBlock::Text("echo:hello:model=None".into())]
    );

    // A second turn in the same process, after switching the model.
    session.set_model(Some("opus")).await.unwrap();
    session
        .set_permission_mode(PermissionMode::Plan)
        .await
        .unwrap();
    session.send("again").await.unwrap();
    let turn = session.receive_turn().await.unwrap();
    assert_eq!(turn.result.result.as_deref(), Some("echo:again:model=opus"));

    // The hook answers a PreToolUse callback from the CLI.
    session.send("hook").await.unwrap();
    let response = result_json(&session.receive_turn().await.unwrap());
    assert_eq!(response["subtype"], "success");
    assert_eq!(
        response["response"]["hookSpecificOutput"]["permissionDecision"],
        "deny"
    );

    assert!(session.close().await.unwrap().success());
}

#[tokio::test]
async fn permission_requests_reach_the_handler() {
    let allow = PermissionDecision::Allow {
        updated_input: Some(json!({"command": "ls -la"})),
        updated_permissions: vec![],
    };
    let mut session = ClaudeSession::start(ClaudeOptions {
        permission_handler: Some(Arc::new(Decide(allow))),
        ..fake()
    })
    .await
    .unwrap();
    session.send("permission").await.unwrap();
    let response = result_json(&session.receive_turn().await.unwrap());
    assert_eq!(
        response["response"],
        json!({"behavior": "allow", "updatedInput": {"command": "ls -la"}})
    );

    let mut session = ClaudeSession::start(ClaudeOptions {
        permission_handler: Some(Arc::new(Decide(PermissionDecision::deny("not now")))),
        ..fake()
    })
    .await
    .unwrap();
    session.send("permission").await.unwrap();
    let response = result_json(&session.receive_turn().await.unwrap());
    assert_eq!(
        response["response"],
        json!({"behavior": "deny", "message": "not now"})
    );

    // Without a handler the CLI gets an error answer rather than silence.
    let mut session = ClaudeSession::start(fake()).await.unwrap();
    session.send("permission").await.unwrap();
    let response = result_json(&session.receive_turn().await.unwrap());
    assert_eq!(response["subtype"], "error");
}

#[tokio::test]
async fn in_process_tools_answer_mcp_requests() {
    let mut session = ClaudeSession::start(ClaudeOptions {
        tool_servers: vec![Arc::new(ToolServer::new("math").tool(Arc::new(Add)))],
        ..fake()
    })
    .await
    .unwrap();
    session.send("tools").await.unwrap();
    let replies = result_json(&session.receive_turn().await.unwrap());
    let reply = |index: usize| replies[index]["response"]["mcp_response"].clone();
    assert_eq!(reply(0)["result"]["serverInfo"]["name"], "math");
    assert_eq!(reply(0)["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(reply(1), json!({"jsonrpc": "2.0", "result": {}}));
    assert_eq!(reply(2)["result"]["tools"][0]["name"], "add");
    assert_eq!(
        reply(3)["result"],
        json!({"content": [{"type": "text", "text": "5"}], "isError": false})
    );
}

#[tokio::test]
async fn interrupt_ends_the_running_turn() {
    let mut session = ClaudeSession::start(fake()).await.unwrap();
    session.send("slow").await.unwrap();
    // The turn has started once the init message arrives.
    assert!(matches!(
        session.next_message().await,
        Some(Message::System(_))
    ));
    session.interrupt().await.unwrap();
    let turn = session.receive_turn().await.unwrap();
    assert_eq!(turn.result.subtype, "error_during_execution");
    assert!(turn.result.is_error);
}

#[tokio::test]
async fn a_crash_reports_the_cli_error() {
    let mut session = ClaudeSession::start(fake()).await.unwrap();
    session.send("exit").await.unwrap();
    let error = session.receive_turn().await.unwrap_err();
    // Stderr is read concurrently; give the tail a moment to fill.
    let ClaudeError::Exited { .. } = error else {
        panic!("expected an exit error, got {error}");
    };
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert!(matches!(
        session.interrupt().await,
        Err(ClaudeError::Exited { stderr }) if stderr.contains("boom")
    ));
}

#[tokio::test]
async fn a_missing_cli_is_a_spawn_error() {
    let error = ClaudeSession::start(ClaudeOptions {
        cli_path: Some("/definitely/not/claude".into()),
        ..ClaudeOptions::default()
    })
    .await
    .err()
    .unwrap();
    assert!(matches!(error, ClaudeError::Spawn { .. }), "{error}");
}

#[test]
fn claude_code_models_mirror_the_anthropic_registry() {
    use crate::{Model, Provider};
    let model = Model::get(Provider::ClaudeCode, "claude-opus-5-5").unwrap();
    assert_eq!(model.provider, Provider::ClaudeCode);
    assert_eq!(model.max_input_tokens, 1_000_000);
    assert_eq!(
        model.id,
        Model::claude_code(Model::anthropic_claude_opus_5_5()).id
    );
    // Ids stay unique across `all`, which leaves the mirror out.
    assert!(
        Model::all()
            .iter()
            .all(|model| model.provider != Provider::ClaudeCode)
    );
    assert_eq!(
        Model::find("claude-opus-5-5").unwrap().provider,
        Provider::Anthropic
    );
    // Saved configurations name the provider.
    let saved = serde_json::to_string(&Provider::ClaudeCode).unwrap();
    assert_eq!(saved, "\"ClaudeCode\"");
}

#[test]
fn setting_sources_map_to_the_flag() {
    let none = ClaudeOptions {
        setting_sources: Some(Vec::new()),
        ..ClaudeOptions::default()
    };
    assert!(none.arguments().contains(&"--setting-sources=".to_string()));
    let some = ClaudeOptions {
        setting_sources: Some(vec!["user".into(), "project".into()]),
        ..ClaudeOptions::default()
    };
    assert!(
        some.arguments()
            .contains(&"--setting-sources=user,project".to_string())
    );
    assert!(
        !ClaudeOptions::default()
            .arguments()
            .iter()
            .any(|argument| argument.starts_with("--setting-sources"))
    );
}
