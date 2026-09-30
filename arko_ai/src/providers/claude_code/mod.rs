//! A client for the user's installed Claude Code CLI, the transport of
//! [`Provider::ClaudeCode`](crate::Provider::ClaudeCode) models.
//!
//! [`ClaudeSession::start`] runs `claude` with `--input-format stream-json
//! --output-format stream-json` and keeps it running for a whole
//! conversation: each [`ClaudeSession::send`] starts a turn, and
//! [`ClaudeSession::receive_turn`] collects its messages up to the turn's
//! [`ResultMessage`]. Over the same pipes the CLI asks the host to approve
//! tool calls ([`PermissionHandler`]), runs host hook callbacks
//! ([`Hook`]), and calls tools the host serves in-process
//! ([`ToolServer`]); the host can interrupt the turn or switch the model or
//! permission mode mid-session.
//!
//! The CLI runs as published, under the user's own login (or an API key in
//! its environment). This client does not touch credentials. Model and
//! effort are fixed per process: to change the effort, start a new session
//! with [`SessionStart::Resume`].
//!
//! The protocol follows Anthropic's Python Agent SDK, which bundles the same
//! CLI version (2.1.285) this module was written against.

mod locate;
mod options;
mod protocol;
mod session;
mod tools;

#[cfg(test)]
mod tests;

pub use options::{
    BuiltInTools, ClaudeOptions, Effort, Hook, HookCallback, HookEvent, PermissionMode,
    SessionStart, SystemPrompt,
};
pub use protocol::{
    AssistantMessage, ContentBlock, Message, PermissionDecision, PermissionHandler,
    PermissionRequest, ResultMessage, SystemMessage, UserMessage,
};
pub use session::{ClaudeError, ClaudeSession, Turn};
pub use tools::{ClaudeTool, ToolOutput, ToolServer};
