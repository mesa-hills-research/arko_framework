//! MCP servers started by `McpManager`: handshake, tool discovery, calls,
//! and a server that exits. Uses the Python stub in `tests/fixtures`
//! (`ARKO_TEST_PYTHON` selects the interpreter).

use arko_agents::{McpConfig, McpManager, McpServerConfig, TaskContext, Tool};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

fn stub_config() -> McpConfig {
    let python = std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp_stub_server.py");
    McpConfig {
        mcp_servers: vec![McpServerConfig {
            name: "stub".into(),
            command: python,
            args: vec![script.display().to_string()],
            env: HashMap::new(),
        }],
    }
}

#[tokio::test]
async fn a_server_is_started_its_tools_called_and_its_exit_survived() {
    let manager = McpManager::from_config(&stub_config()).await.unwrap();
    let mut names = manager.tool_names();
    names.sort();
    assert_eq!(names, ["mcp__stub__echo", "mcp__stub__write_note"]);

    let echo = manager
        .tools()
        .iter()
        .find(|tool| tool.remote_tool_name() == "echo")
        .unwrap()
        .clone();
    let workspace = std::env::temp_dir();
    let context = TaskContext {
        task_id: uuid::Uuid::new_v4(),
        agent_name: "tester",
        initial_task_prompt: "test",
        current_working_directory: &workspace,
    };
    let call = |text: &str| HashMap::from([("text".to_string(), json!(text))]);
    assert_eq!(
        echo.execute(&context, call("hello")).await.unwrap(),
        "echo: hello"
    );

    // The server exits during a call: that call and later ones fail cleanly.
    assert!(echo.execute(&context, call("exit")).await.is_err());
    let error = echo.execute(&context, call("again")).await.unwrap_err();
    assert!(error.to_string().contains("stub"), "{error}");
    manager.shutdown().await;
}
