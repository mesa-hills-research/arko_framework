//! Agents whose model runs in Claude Code.
//!
//! An agent whose model is a [`Provider::ClaudeCode`](arko_ai::Provider::ClaudeCode)
//! model (for example `Model::claude_code(Model::anthropic_claude_opus_5_5())`)
//! runs in a session of the user's installed Claude Code CLI, under its login,
//! instead of through [`LLMClient`](arko_ai::LLMClient) requests. Claude Code
//! runs the agent loop; the arko runtime drives the session and keeps the task,
//! its events, commands, approvals and handovers as for any other agent:
//!
//! - The agent's arko tools (`complete_task`, `ask_user`, a workflow's
//!   `send_message`, custom tools) are served to Claude Code in-process, as
//!   the MCP server `arko`: the model sees them as `mcp__arko__<name>`. They
//!   run as they do for any agent, with the execution middleware, result
//!   interception and time limits.
//! - Claude Code's own tools ([`BUILT_IN_TOOLS`]: `Read`, `Edit`, `Bash`, …)
//!   are offered when `available_tool_names` lists them, and only then. Every
//!   call, built-in or arko, first passes the runtime's gate: auto-approval or
//!   an approval wait, the workspace path policy, and (for built-ins) the
//!   execution middleware's `before`. Claude Code runs a built-in itself, so
//!   its output can't be intercepted and arko's per-tool time limit doesn't
//!   apply; [`AgentConfig::validate`](crate::AgentConfig::validate) rejects
//!   those settings for built-ins.
//! - Environment details, reminders, the system prompt (appended to Claude
//!   Code's, or replacing it), workflow context updates and handovers work as
//!   for other agents.
//!
//! [`ClaudeCodeSettings`] sets how the CLI runs.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Claude Code built-in tools this crate knows, as of Claude Code 2.1.
/// `available_tool_names` may name others the installed CLI has; they are
/// offered as built-ins too. `AskUserQuestion` is left out: arko's `ask_user`
/// asks through the host.
pub const BUILT_IN_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "Bash",
    "PowerShell",
    "Monitor",
    "TaskOutput",
    "TaskStop",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "Agent",
    "Task",
    "Skill",
    "LSP",
    "ToolSearch",
];

/// Built-in tools arko doesn't offer: they would bypass the host (a question
/// to the user, or leaving plan mode).
pub const UNSUPPORTED_BUILT_IN_TOOLS: &[&str] =
    &["AskUserQuestion", "ExitPlanMode", "EnterPlanMode"];

/// Whether `name` is a Claude Code built-in tool this crate knows.
pub fn is_built_in_tool(name: &str) -> bool {
    BUILT_IN_TOOLS.contains(&name) || UNSUPPORTED_BUILT_IN_TOOLS.contains(&name)
}

/// The input fields of a built-in tool that name a path, which the
/// workspace path policy checks.
pub fn built_in_path_parameters(name: &str) -> &'static [&'static str] {
    match name {
        "Read" | "Write" | "Edit" | "MultiEdit" => &["file_path"],
        "NotebookEdit" => &["notebook_path"],
        "Glob" | "Grep" => &["path"],
        _ => &[],
    }
}

/// Common sets of built-in tools, to add to `available_tool_names`.
pub struct ClaudeCodeTools;

impl ClaudeCodeTools {
    /// Reading and searching files: `Read`, `Glob`, `Grep`.
    pub fn read_only() -> Vec<String> {
        names(&["Read", "Glob", "Grep"])
    }

    /// [`read_only`](Self::read_only), and editing files: `Edit`, `Write`,
    /// `NotebookEdit`.
    pub fn editing() -> Vec<String> {
        let mut tools = Self::read_only();
        tools.extend(names(&["Edit", "Write", "NotebookEdit"]));
        tools
    }

    /// Running shell commands: `Bash` (and `PowerShell` on Windows), with
    /// the tools for background commands.
    pub fn shell() -> Vec<String> {
        names(&["Bash", "PowerShell", "Monitor", "TaskOutput", "TaskStop"])
    }

    /// Fetching and searching the web: `WebFetch`, `WebSearch`.
    pub fn web() -> Vec<String> {
        names(&["WebFetch", "WebSearch"])
    }

    /// Every built-in tool this crate knows ([`BUILT_IN_TOOLS`]).
    pub fn all() -> Vec<String> {
        names(BUILT_IN_TOOLS)
    }
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| name.to_string()).collect()
}

/// How an agent's Claude Code session runs; see the [module docs](self).
/// Only for agents whose model is a Claude Code model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ClaudeCodeSettings {
    /// The CLI to run; `claude` on the `PATH` when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cli_path: Option<PathBuf>,
    /// Arguments before the CLI's own, for a CLI run through an interpreter
    /// (`cli_path` `node`, `cli_args` `["cli.js"]`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cli_args: Vec<String>,
    /// The user's settings files the session loads (`user`, `project`,
    /// `local`), with their `CLAUDE.md`, hooks, MCP servers and skills. Empty
    /// (the default): none, so the agent behaves the same on any machine.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub setting_sources: Vec<String>,
    /// Replace Claude Code's system prompt with the agent's, instead of
    /// appending the agent's to it (the default). Claude Code's prompt
    /// explains its built-in tools; an agent without them can do without it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub replace_system_prompt: bool,
    /// Environment variables for the CLI.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_sets_are_known_built_ins() {
        for tool in ClaudeCodeTools::all()
            .into_iter()
            .chain(ClaudeCodeTools::editing())
            .chain(ClaudeCodeTools::shell())
            .chain(ClaudeCodeTools::web())
        {
            assert!(is_built_in_tool(&tool), "{tool}");
        }
        assert!(!is_built_in_tool("read_file"));
        assert_eq!(built_in_path_parameters("Edit"), ["file_path"]);
        assert!(built_in_path_parameters("Bash").is_empty());
    }

    #[test]
    fn settings_load_from_empty_json() {
        let settings: ClaudeCodeSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(settings, ClaudeCodeSettings::default());
        assert_eq!(serde_json::to_string(&settings).unwrap(), "{}");
    }
}
