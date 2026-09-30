use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::protocol::PermissionHandler;
use super::tools::ToolServer;

/// How a session is started.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum SessionStart {
    /// A new conversation with an ID the CLI chooses.
    #[default]
    New,
    /// A new conversation with this ID (a UUID).
    WithId(String),
    /// An earlier conversation, by session ID or transcript path. With
    /// `fork`, the resumed conversation gets a new session ID and the
    /// original is left unchanged.
    Resume { session: String, fork: bool },
    /// The most recent conversation in the working directory.
    Continue,
}

/// The system prompt Claude Code runs with.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum SystemPrompt {
    /// Claude Code's own system prompt.
    #[default]
    ClaudeCode,
    /// Claude Code's system prompt followed by this text.
    Append(String),
    /// This text instead of Claude Code's system prompt.
    Replace(String),
}

/// Which of Claude Code's built-in tools (Bash, Read, Edit, …) the model
/// can see.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum BuiltInTools {
    #[default]
    Default,
    None,
    Only(Vec<String>),
}

/// Reasoning effort, `--effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

/// `--permission-mode`, also settable mid-session with
/// [`ClaudeSession::set_permission_mode`](super::ClaudeSession::set_permission_mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    AcceptEdits,
    Plan,
    Auto,
    DontAsk,
    BypassPermissions,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PermissionMode::Default => "default",
            PermissionMode::AcceptEdits => "acceptEdits",
            PermissionMode::Plan => "plan",
            PermissionMode::Auto => "auto",
            PermissionMode::DontAsk => "dontAsk",
            PermissionMode::BypassPermissions => "bypassPermissions",
        }
    }
}

/// A Claude Code hook event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    Stop,
    SubagentStop,
    PreCompact,
    Notification,
    Other(String),
}

impl HookEvent {
    pub fn as_str(&self) -> &str {
        match self {
            HookEvent::PreToolUse => "PreToolUse",
            HookEvent::PostToolUse => "PostToolUse",
            HookEvent::UserPromptSubmit => "UserPromptSubmit",
            HookEvent::Stop => "Stop",
            HookEvent::SubagentStop => "SubagentStop",
            HookEvent::PreCompact => "PreCompact",
            HookEvent::Notification => "Notification",
            HookEvent::Other(name) => name,
        }
    }
}

/// A host function Claude Code calls at a hook event. `input` is the hook
/// input Claude Code documents for the event (for `PreToolUse`:
/// `tool_name`, `tool_input`, `session_id`, `cwd`, …); the returned JSON is
/// the hook output, for example [`Hook::deny_tool`].
#[async_trait]
pub trait HookCallback: Send + Sync {
    async fn call(&self, input: Value, tool_use_id: Option<String>) -> Value;
}

/// A hook callback registered for an event.
#[derive(Clone)]
pub struct Hook {
    pub event: HookEvent,
    /// A tool-name pattern (`"Bash"`, `"Edit|Write"`); `None` matches all.
    pub matcher: Option<String>,
    pub callback: Arc<dyn HookCallback>,
    pub timeout_seconds: Option<f64>,
}

impl fmt::Debug for Hook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hook")
            .field("event", &self.event)
            .field("matcher", &self.matcher)
            .finish_non_exhaustive()
    }
}

impl Hook {
    pub fn new(event: HookEvent, matcher: Option<&str>, callback: Arc<dyn HookCallback>) -> Self {
        Self {
            event,
            matcher: matcher.map(str::to_string),
            callback,
            timeout_seconds: None,
        }
    }

    /// `PreToolUse` output that blocks the call and tells the model why.
    pub fn deny_tool(reason: impl Into<String>) -> Value {
        json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason.into(),
            }
        })
    }

    /// Output that lets Claude Code continue as it would without the hook.
    pub fn proceed() -> Value {
        json!({})
    }
}

/// How to run Claude Code. Everything maps to a documented CLI flag, except
/// hooks, permission handling, and in-process tools, which travel over the
/// control protocol.
#[derive(Clone, Default)]
pub struct ClaudeOptions {
    /// The CLI to run: `claude` on the `PATH` by default. On Windows an npm
    /// launcher script (`claude.cmd`) is followed to the package's
    /// `claude.exe`, or its `cli.js` run with `node`.
    pub cli_path: Option<PathBuf>,
    /// Arguments before the ones this client adds, for a CLI run through an
    /// interpreter (`node cli.js`).
    pub cli_prefix_args: Vec<String>,
    pub working_directory: Option<PathBuf>,
    /// `--model`: an alias (`opus`, `sonnet`, `fable`) or a model ID.
    pub model: Option<String>,
    pub fallback_model: Option<String>,
    pub effort: Option<Effort>,
    pub system_prompt: SystemPrompt,
    pub built_in_tools: BuiltInTools,
    /// Tools that run without asking (`Read`, `Bash(git diff *)`,
    /// `mcp__server__tool`).
    pub allowed_tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    pub permission_mode: Option<PermissionMode>,
    pub max_turns: Option<u32>,
    pub session: SessionStart,
    /// `--settings`: Claude Code settings for this session.
    pub settings: Option<Value>,
    /// `--setting-sources`: which of the user's settings files (`user`,
    /// `project`, `local`), with their `CLAUDE.md`, hooks and skills, the
    /// session loads. `None` leaves the CLI's default; `Some(vec![])` loads
    /// none, so the session behaves the same on any machine.
    pub setting_sources: Option<Vec<String>>,
    pub add_directories: Vec<PathBuf>,
    /// External MCP servers, as in `.mcp.json` (`{"command": …}`,
    /// `{"type": "http", "url": …}`).
    pub mcp_servers: Map<String, Value>,
    /// Stream token deltas as [`Message::StreamEvent`](super::Message::StreamEvent).
    pub include_partial_messages: bool,
    pub env: HashMap<String, String>,
    /// Further CLI arguments, added last.
    pub extra_args: Vec<String>,
    /// Decides tool calls Claude Code would otherwise ask about.
    pub permission_handler: Option<Arc<dyn PermissionHandler>>,
    pub hooks: Vec<Hook>,
    /// Tools the host serves in-process, as MCP servers named here.
    pub tool_servers: Vec<Arc<ToolServer>>,
}

impl fmt::Debug for ClaudeOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaudeOptions")
            .field("cli_path", &self.cli_path)
            .field("model", &self.model)
            .field("effort", &self.effort)
            .field("session", &self.session)
            .field("permission_mode", &self.permission_mode)
            .field("hooks", &self.hooks)
            .field("permission_handler", &self.permission_handler.is_some())
            .finish_non_exhaustive()
    }
}

impl ClaudeOptions {
    pub fn program(&self) -> PathBuf {
        self.cli_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("claude"))
    }

    /// The CLI arguments for these options.
    pub fn arguments(&self) -> Vec<String> {
        let mut args: Vec<String> = self.cli_prefix_args.clone();
        let mut push = |values: &[&str]| args.extend(values.iter().map(|v| v.to_string()));
        push(&[
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "stream-json",
        ]);
        match &self.system_prompt {
            SystemPrompt::ClaudeCode => {}
            SystemPrompt::Append(text) => push(&["--append-system-prompt", text]),
            SystemPrompt::Replace(text) => push(&["--system-prompt", text]),
        }
        match &self.built_in_tools {
            BuiltInTools::Default => {}
            BuiltInTools::None => push(&["--tools", ""]),
            BuiltInTools::Only(tools) => push(&["--tools", &tools.join(",")]),
        }
        if !self.allowed_tools.is_empty() {
            push(&["--allowedTools", &self.allowed_tools.join(",")]);
        }
        if !self.disallowed_tools.is_empty() {
            push(&["--disallowedTools", &self.disallowed_tools.join(",")]);
        }
        if let Some(turns) = self.max_turns {
            push(&["--max-turns", &turns.to_string()]);
        }
        if let Some(model) = &self.model {
            push(&["--model", model]);
        }
        if let Some(model) = &self.fallback_model {
            push(&["--fallback-model", model]);
        }
        if let Some(effort) = self.effort {
            push(&["--effort", effort.as_str()]);
        }
        if self.permission_handler.is_some() {
            // The CLI sends `can_use_tool` requests over stdout instead of
            // prompting.
            push(&["--permission-prompt-tool", "stdio"]);
        }
        if let Some(mode) = self.permission_mode {
            push(&["--permission-mode", mode.as_str()]);
        }
        match &self.session {
            SessionStart::New => {}
            SessionStart::WithId(id) => push(&[&format!("--session-id={id}")]),
            SessionStart::Resume { session, fork } => {
                push(&[&format!("--resume={session}")]);
                if *fork {
                    push(&["--fork-session"]);
                }
            }
            SessionStart::Continue => push(&["--continue"]),
        }
        if let Some(settings) = &self.settings {
            push(&["--settings", &settings.to_string()]);
        }
        if let Some(sources) = &self.setting_sources {
            push(&[&format!("--setting-sources={}", sources.join(","))]);
        }
        for directory in &self.add_directories {
            push(&["--add-dir", &directory.to_string_lossy()]);
        }
        let mut servers = self.mcp_servers.clone();
        for server in &self.tool_servers {
            servers.insert(
                server.name().to_string(),
                json!({"type": "sdk", "name": server.name()}),
            );
        }
        if !servers.is_empty() {
            push(&[
                "--mcp-config",
                &json!({ "mcpServers": servers }).to_string(),
            ]);
        }
        if self.include_partial_messages {
            push(&["--include-partial-messages"]);
        }
        args.extend(self.extra_args.iter().cloned());
        args
    }
}
