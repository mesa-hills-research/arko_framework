//! MCP servers started by `McpManager`: handshake, tool discovery, calls,
//! and a server that exits. Uses the Python stub in `tests/fixtures`
//! (`ARKO_TEST_PYTHON` selects the interpreter).

use arko_agents::{McpConfig, McpManager, McpServerConfig, TaskContext, Tool, ToolError};
use arko_ai::Image;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

fn stub_config() -> McpConfig {
    let python = std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp_stub_server.py");
    McpConfig {
        mcp_servers: vec![
            McpServerConfig::new("stub", python).with_args([script.display().to_string()]),
        ],
    }
}

#[tokio::test]
async fn a_server_is_started_its_tools_called_and_its_exit_survived() {
    let manager = McpManager::from_config(&stub_config()).await.unwrap();
    let mut names = manager.tool_names();
    names.sort();
    assert_eq!(names, ["mcp__stub__echo", "mcp__stub__write_note"]);

    // Only the tool the server marks read-only is parallel-safe. Every call
    // can move to the background, its job awaiting the response.
    for tool in manager.tools() {
        assert_eq!(
            tool.definition().parallel_safe,
            tool.remote_tool_name() == "echo",
            "{}",
            tool.remote_tool_name()
        );
        assert_eq!(tool.background_mode(), arko_agents::ToolBackground::Detach);
    }
    let echo = manager
        .tools()
        .iter()
        .find(|tool| tool.remote_tool_name() == "echo")
        .unwrap()
        .clone();
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
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

#[tokio::test]
async fn images_and_embedded_image_resources_survive_mixed_mcp_results() {
    let manager = McpManager::from_config(&stub_config()).await.unwrap();
    let echo = manager
        .tools()
        .iter()
        .find(|tool| tool.remote_tool_name() == "echo")
        .unwrap()
        .clone();
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let params = HashMap::from([("text".to_string(), json!("mixed-images"))]);
    let output = echo
        .execute_with_output(&context, params.clone())
        .await
        .unwrap();
    assert_eq!(
        output.text,
        "before images\nResource file:///note.txt:\nresource text\nafter images"
    );
    assert_eq!(output.images.len(), 4);
    for (image, (mime_type, data)) in output.images.iter().zip([
        ("image/png", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJb8AAAAASUVORK5CYII="),
        ("image/gif", "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"),
        ("image/jpeg", "/9j/"),
        ("image/webp", "UklGRgAAAABXRUJQ"),
    ]) {
        assert_eq!(
            image.image,
            Image::Base64 {
                data: data.into(),
                mime_type: mime_type.into(),
            }
        );
        assert!(!output.text.contains(data), "image bytes leaked into text");
    }
    assert_eq!(output.images[1].description, "file:///tiny.gif");
    assert_eq!(output.images[3].description, "file:///tiny.webp");
    assert_eq!(echo.execute(&context, params).await.unwrap(), output.text);

    let image_only = echo
        .execute_with_output(
            &context,
            HashMap::from([("text".into(), json!("image-only"))]),
        )
        .await
        .unwrap();
    assert!(image_only.text.is_empty());
    assert_eq!(image_only.images.len(), 1);
    assert_eq!(image_only.images[0].image, output.images[0].image);
    let structured_image = echo
        .execute_with_output(
            &context,
            HashMap::from([("text".into(), json!("image-structured"))]),
        )
        .await
        .unwrap();
    assert_eq!(structured_image.images.len(), 1);
    assert_eq!(structured_image.images[0].image, image_only.images[0].image);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&structured_image.text).unwrap(),
        json!({"width": 1, "height": 1})
    );
    manager.shutdown().await;
}

#[tokio::test]
async fn unsupported_mcp_media_is_described_without_exposing_binary_data() {
    let manager = McpManager::from_config(&stub_config()).await.unwrap();
    let echo = manager
        .tools()
        .iter()
        .find(|tool| tool.remote_tool_name() == "echo")
        .unwrap()
        .clone();
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let output = echo
        .execute_with_output(
            &context,
            HashMap::from([("text".into(), json!("unsupported-content"))]),
        )
        .await
        .unwrap();
    assert!(output.images.is_empty());
    for description in [
        "image/svg+xml",
        "audio/wav",
        "file:///report.pdf",
        "application/pdf",
        "file:///unknown.bin",
        "unknown type",
        "Resource link: file:///linked.png (linked image)",
    ] {
        assert!(output.text.contains(description), "{}", output.text);
    }
    for encoded in ["c3Zn", "d2F2", "cGRm", "Ymlu"] {
        assert!(!output.text.contains(encoded), "{}", output.text);
    }
    let structured = echo
        .execute_with_output(
            &context,
            HashMap::from([("text".into(), json!("unsupported-structured"))]),
        )
        .await
        .unwrap();
    let (descriptions, metadata) = structured.text.rsplit_once("\n\n").unwrap();
    assert_eq!(descriptions, output.text);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(metadata).unwrap(),
        json!({"answer": 42})
    );
    manager.shutdown().await;
}

#[tokio::test]
async fn mcp_images_preserve_error_results_and_structured_text_fallback() {
    let manager = McpManager::from_config(&stub_config()).await.unwrap();
    let echo = manager
        .tools()
        .iter()
        .find(|tool| tool.remote_tool_name() == "echo")
        .unwrap()
        .clone();
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    for (text, expected) in [
        ("error-with-image", "render failed"),
        (
            "error-image-only",
            "MCP tool 'echo' returned an error with no text content",
        ),
    ] {
        let error = echo
            .execute_with_output(&context, HashMap::from([("text".into(), json!(text))]))
            .await
            .unwrap_err();
        assert!(matches!(error, ToolError::ExecutionFailed(message) if message == expected));
    }
    // A tool error must not take down the server or prevent ordinary results.
    let output = echo
        .execute_with_output(
            &context,
            HashMap::from([("text".into(), json!("structured-only"))]),
        )
        .await
        .unwrap();
    assert!(output.images.is_empty());
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&output.text).unwrap(),
        json!({"answer": 42})
    );
    manager.shutdown().await;
}

/// A call dropped before its response (stopped, or past its time limit)
/// cancels its request on the server, which can then take the next one.
#[tokio::test]
async fn a_dropped_call_cancels_its_request_on_the_server() {
    let manager = McpManager::from_config(&stub_config()).await.unwrap();
    let echo = manager
        .tools()
        .iter()
        .find(|tool| tool.remote_tool_name() == "echo")
        .unwrap()
        .clone();
    let workspace = std::env::temp_dir();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "test", &workspace);
    let call = |text: &str| HashMap::from([("text".to_string(), json!(text))]);
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(300),
            echo.execute(&context, call("hang"))
        )
        .await
        .is_err(),
        "the server doesn't answer it"
    );
    let cancelled = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let cancelled = echo.execute(&context, call("cancelled")).await.unwrap();
            if cancelled != "[]" {
                return cancelled;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the server is told");
    let ids: Vec<serde_json::Value> = serde_json::from_str(&cancelled).unwrap();
    assert_eq!(ids.len(), 1, "{cancelled}");
    // A call that's answered cancels nothing.
    assert_eq!(
        echo.execute(&context, call("hello")).await.unwrap(),
        "echo: hello"
    );
    assert_eq!(
        echo.execute(&context, call("cancelled")).await.unwrap(),
        cancelled
    );
    manager.shutdown().await;
}
