//! Built-in tools provided by the arko_agents library.

pub mod apply_diff;
pub mod ask_user;
pub mod complete_task;
pub mod list_files;
pub mod read_file;
pub mod write_file;

pub use complete_task::COMPLETE_TASK_TOOL_NAME;
pub use complete_task::CompleteTaskTool;

pub use read_file::READ_FILE_TOOL_NAME;
pub use read_file::ReadFileTool;

pub use write_file::WRITE_FILE_TOOL_NAME;
pub use write_file::WriteFileTool;

pub use apply_diff::APPLY_DIFF_TOOL_NAME;
pub use apply_diff::ApplyDiffTool;

pub use list_files::LIST_FILES_TOOL_NAME;
pub use list_files::ListFilesTool;

pub use ask_user::ASK_USER_TOOL_NAME;
pub use ask_user::AskUserTool;

/// The path a file tool opens for its `param_name` parameter: resolved with
/// [`crate::utils::resolve_path`], which the runtime's workspace check also
/// uses, so the check judges exactly this path.
fn resolve_path_parameter(
    param_name: &str,
    path: &str,
    working_directory: &std::path::Path,
) -> Result<std::path::PathBuf, crate::tool::ToolError> {
    crate::utils::resolve_path(path, working_directory).map_err(|error| {
        crate::tool::ToolError::ParameterValidation {
            param_name: param_name.to_string(),
            message: format!("Invalid path '{path}': {error}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{TaskContext, Tool};
    use crate::utils::{is_path_in_workspace, resolve_path};
    use serde_json::{Value, json};
    use std::collections::HashMap;

    fn params(value: Value) -> HashMap<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    /// A new directory under the canonical temporary directory (on macOS,
    /// `/private/var/...` rather than the `/var/...` link).
    fn temp_root() -> std::path::PathBuf {
        let root = std::fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("arko-file-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[tokio::test]
    async fn file_tools_open_the_path_the_workspace_check_judges() {
        let root = temp_root();
        let workspace = root.join("ws");
        for directory in ["~", "$HOME", "sub"] {
            std::fs::create_dir_all(workspace.join(directory)).unwrap();
        }
        let context = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "tester",
            initial_task_prompt: "",
            current_working_directory: &workspace,
        };
        for (path, inside) in [
            ("~/a.txt", true),
            ("$HOME/b.txt", true),
            (" sub/../c.txt\n", true),
            ("~/../../outside.txt", false),
        ] {
            let target = resolve_path(path, &workspace).unwrap();
            assert_eq!(target.starts_with(&workspace), inside, "{path}");
            assert_eq!(
                is_path_in_workspace(path, &workspace).unwrap(),
                inside,
                "{path}"
            );
            WriteFileTool
                .execute(&context, params(json!({"path": path, "content": "one"})))
                .await
                .unwrap();
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "one", "{path}");
            let diff = "<<<<<<< SEARCH\n:start_line:1\n-------\none\n=======\ntwo\n>>>>>>> REPLACE";
            ApplyDiffTool
                .execute(&context, params(json!({"path": path, "diff": diff})))
                .await
                .unwrap();
            let read = ReadFileTool
                .execute(&context, params(json!({"path": path})))
                .await
                .unwrap();
            assert_eq!(read, "1 | two", "{path}");
        }
        let listed = ListFilesTool
            .execute(&context, params(json!({"directory": " ~/ "})))
            .await
            .unwrap();
        assert!(
            listed.contains("a.txt") && !listed.contains("b.txt"),
            "{listed}"
        );
        assert!(root.join("outside.txt").exists());
        let error = ReadFileTool
            .execute(&context, params(json!({"path": "a\u{0}b"})))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("NUL"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_file_tools_take_either_separator_and_any_case() {
        let root = temp_root();
        let workspace = root.join("ws");
        std::fs::create_dir_all(workspace.join(r"sub\inner")).unwrap();
        std::fs::write(workspace.join(r"sub\inner\deep.txt"), "deep").unwrap();
        let context = TaskContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "tester",
            initial_task_prompt: "",
            current_working_directory: &workspace,
        };
        // `\` separates components on Windows, as `/` does.
        for directory in [r"sub\inner", "sub/inner", r"sub/inner\", r"sub\x\..\inner"] {
            assert!(
                is_path_in_workspace(directory, &workspace).unwrap(),
                "{directory}"
            );
            let listed = ListFilesTool
                .execute(&context, params(json!({"directory": directory})))
                .await
                .unwrap();
            assert!(listed.contains("deep.txt"), "{directory}: {listed}");
        }
        // The workspace written in another case is the same directory.
        let upper = format!(
            r"{}\SUB\INNER\DEEP.TXT",
            workspace.to_string_lossy().to_uppercase()
        );
        assert!(is_path_in_workspace(&upper, &workspace).unwrap(), "{upper}");
        let read = ReadFileTool
            .execute(&context, params(json!({"path": upper})))
            .await
            .unwrap();
        assert_eq!(read, "1 | deep");
        std::fs::remove_dir_all(root).unwrap();
    }
}
