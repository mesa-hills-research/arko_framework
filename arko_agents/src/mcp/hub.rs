//! [`McpHub`]: a session's MCP servers, started, stopped, added and removed
//! while it runs, and asked for their tools each time.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rmcp::service::{Peer, RoleClient};
use serde_json::{Map, Value};

use super::bridge::McpAccess;
use super::call::{CallFailure, call_tool};
use super::config::McpServerConfig;
use super::connection::{Connection, Redactions};
use super::pool::{self, SharedServer};
use super::tool::mcp_definition;
use crate::tool::{TaskContext, Tool, ToolBackground, ToolDefinition, ToolError, ToolOutput};

/// How long a server gets to start and list its tools, unless
/// [`McpHub::start_all`] says otherwise.
pub const DEFAULT_START_TIMEOUT: Duration = Duration::from_secs(60);

/// The longest tool name the model providers take (`^[a-zA-Z0-9_-]{1,64}$`).
const MAX_TOOL_NAME: usize = 64;

/// A server's state, as its hub sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerState {
    /// It is starting.
    Starting,
    /// It runs, and the hub is connected to it.
    Running,
    /// It was never started, or was stopped. A call starts it.
    Stopped,
    /// It didn't start, or it ended on its own, for this reason in fixed
    /// words (`it exited with code 1`, `its program wasn't found`, `it
    /// didn't start within 60 s`). The reason never holds the server's
    /// command or what it wrote to stderr, which go to the log.
    Failed(String),
}

/// What a hub knows of one of its servers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ServerFacts {
    pub name: String,
    pub state: ServerState,
    /// How many tools it had when they were last listed, or `None` before
    /// they were.
    pub tools: Option<usize>,
    /// The server's own instructions, from its handshake.
    pub instructions: Option<String>,
    /// The config's [`description`](McpServerConfig::description).
    pub description: Option<String>,
    /// The config's [`workers`](McpServerConfig::workers).
    pub workers: bool,
    /// The config's [`shared`](McpServerConfig::shared).
    pub shared: bool,
}

/// One tool of a server, as the server lists it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct McpToolInfo {
    pub name: String,
    /// Its description, empty when it has none.
    pub description: String,
    /// Its JSON input schema, as the server gives it.
    pub input_schema: Value,
    /// The server marks it read-only (`readOnlyHint`).
    pub read_only: bool,
}

impl McpToolInfo {
    pub(crate) fn from_listed(tool: &rmcp::model::Tool) -> Self {
        Self {
            name: tool.name.to_string(),
            description: tool.description.as_deref().unwrap_or("").to_string(),
            input_schema: Value::Object(tool.input_schema.as_ref().clone()),
            read_only: tool
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.read_only_hint)
                == Some(true),
        }
    }

    /// The names its schema says are required.
    pub fn required(&self) -> Vec<&str> {
        self.input_schema
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    }
}

/// The tool name of `server`'s tool `tool` as a tool of its own:
/// `mcp__<server>__<tool>`, with the characters the providers don't take
/// replaced by `_`, and cut to 64 characters ending in a hash of the full
/// name, to stay unique. The same names every time, as conversations keep
/// tool names.
pub fn direct_tool_name(server: &str, tool: &str) -> String {
    let clean = |text: &str| -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    let name = format!("mcp__{}__{}", clean(server), clean(tool));
    if name.len() <= MAX_TOOL_NAME {
        return name;
    }
    // FNV-1a: stable across Rust versions, as names saved in a session's
    // checkpoints must be.
    let hash = format!("{server}\n{tool}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let hash = format!("{:08x}", hash as u32);
    format!("{}_{hash}", &name[..MAX_TOOL_NAME - hash.len() - 1])
}

/// A session's MCP servers, shared by its agents and locked inside.
///
/// The hub starts, stops, adds and removes servers while the session runs,
/// and reports their state. [`tools`](Self::tools) asks the server each
/// time, so a server that changes its tools needs no notification, and
/// [`call`](Self::call) starts a server that isn't running first. Agents
/// reach the hub through [`McpBridgeTool`](super::McpBridgeTool), and a
/// conversation saved with a tool per MCP tool through
/// [`direct_tools`](Self::direct_tools).
///
/// A server runs with the programs it starts as one process tree, as with
/// [`McpManager`](super::McpManager). Its process runs on the runtime that
/// started it, and ends with it. A [shared](McpServerConfig::shared) server
/// is one process for every hub of the process that names the same
/// command, arguments and environment, on a runtime of its own. The host's
/// environment and working directory count too. The process gets exactly
/// those it was first started with, and a hub that starts the server in
/// another environment gets a process of its own. It stops
/// when no hub uses it any more, and [`stop`](Self::stop) and
/// [`start`](Self::start) change only this hub's use of it.
///
/// [`shutdown`](Self::shutdown) stops the servers, each given three seconds
/// to exit. Dropping the hub kills its own servers at once.
pub struct McpHub {
    entries: Mutex<Vec<Arc<Entry>>>,
    start_timeout: Mutex<Duration>,
}

impl std::fmt::Debug for McpHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<String> = self
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|entry| entry.config.name.clone())
            .collect();
        f.debug_struct("McpHub")
            .field("servers", &names)
            .finish_non_exhaustive()
    }
}

struct Entry {
    config: McpServerConfig,
    /// Held while the server starts or stops.
    lifecycle: tokio::sync::Mutex<()>,
    live: Mutex<Live>,
}

struct Live {
    state: ServerState,
    link: Option<Link>,
    /// Its tools when they were last listed. Kept while it is stopped, for
    /// the calls' parallel safety and the direct tools.
    tools: Option<Arc<Vec<McpToolInfo>>>,
    instructions: Option<String>,
}

/// How the hub reaches a running server.
enum Link {
    Own(Connection),
    Shared(Arc<SharedServer>),
}

impl Link {
    fn peer(&self) -> Option<Peer<RoleClient>> {
        match self {
            Self::Own(connection) => connection.exited().is_none().then(|| connection.peer()),
            Self::Shared(server) => server.peer(),
        }
    }

    fn exited(&self) -> Option<String> {
        match self {
            Self::Own(connection) => connection.exited(),
            Self::Shared(server) => server.exited(),
        }
    }
}

impl Entry {
    fn facts(&self) -> ServerFacts {
        let live = self.live.lock().unwrap();
        let state = match (&live.state, live.link.as_ref().and_then(Link::exited)) {
            (ServerState::Running, Some(exit)) => ServerState::Failed(exit),
            (state, _) => state.clone(),
        };
        ServerFacts {
            name: self.config.name.clone(),
            state,
            tools: live.tools.as_ref().map(|tools| tools.len()),
            instructions: live.instructions.clone(),
            description: self.config.description.clone(),
            workers: self.config.workers,
            shared: self.config.shared,
        }
    }

    /// The running server's connection, or `None` when it isn't running.
    fn peer(&self) -> Option<Peer<RoleClient>> {
        let live = self.live.lock().unwrap();
        if live.state != ServerState::Running {
            return None;
        }
        live.link.as_ref()?.peer()
    }
}

/// "Couldn't start github, because it exited with code 1."
fn couldnt_start(server: &str, reason: &str) -> String {
    format!(
        "Couldn't start {server}, because {}.",
        reason.trim_end_matches('.')
    )
}

/// Sets a server whose start was dropped before it ended back to stopped.
struct Starting<'a>(Option<&'a Entry>);

impl Drop for Starting<'_> {
    fn drop(&mut self) {
        if let Some(entry) = self.0 {
            let mut live = entry.live.lock().unwrap();
            if live.state == ServerState::Starting {
                live.state = ServerState::Stopped;
            }
        }
    }
}

/// Stops a server the hub reached: its own is stopped with what it started,
/// and a shared one is let go.
async fn let_go(link: Link) {
    match link {
        Link::Own(mut connection) => connection.stop().await,
        Link::Shared(server) => drop(server),
    }
}

impl McpHub {
    /// A hub of `servers`, none started yet. A name given twice keeps the
    /// first.
    pub fn new(servers: Vec<McpServerConfig>) -> Arc<Self> {
        let mut names = HashSet::new();
        let entries = servers
            .into_iter()
            .filter(|config| {
                let first = names.insert(config.name.clone());
                if !first {
                    log::warn!(
                        "MCP server '{}' is set up twice. The first is kept.",
                        config.name
                    );
                }
                first
            })
            .map(Self::entry_of)
            .collect();
        Arc::new(Self {
            entries: Mutex::new(entries),
            start_timeout: Mutex::new(DEFAULT_START_TIMEOUT),
        })
    }

    fn entry_of(config: McpServerConfig) -> Arc<Entry> {
        Arc::new(Entry {
            config,
            lifecycle: tokio::sync::Mutex::new(()),
            live: Mutex::new(Live {
                state: ServerState::Stopped,
                link: None,
                tools: None,
                instructions: None,
            }),
        })
    }

    fn entry(&self, name: &str) -> Result<Arc<Entry>, String> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .find(|entry| entry.config.name == name)
            .cloned()
            .ok_or_else(|| format!("There's no MCP server named {name}."))
    }

    fn timeout(&self) -> Duration {
        *self.start_timeout.lock().unwrap()
    }

    /// Starts every server at the same time, each given `timeout` to start
    /// and list its tools. A server that fails is reported by
    /// [`servers`](Self::servers). Later starts take the same time.
    pub async fn start_all(&self, timeout: Duration) {
        *self.start_timeout.lock().unwrap() = timeout;
        let entries: Vec<Arc<Entry>> = self.entries.lock().unwrap().clone();
        futures::future::join_all(entries.iter().map(|entry| self.start_entry(entry))).await;
    }

    /// Starts the server `name` unless it runs, and returns its facts, or
    /// a sentence saying why it didn't start (`Couldn't start github,
    /// because it exited with code 1.`). Its state then holds the reason.
    pub async fn start(&self, name: &str) -> Result<ServerFacts, String> {
        let entry = self.entry(name)?;
        self.start_entry(&entry)
            .await
            .map_err(|reason| couldnt_start(name, &reason))?;
        Ok(entry.facts())
    }

    /// Starts `entry` unless it runs, within the hub's start timeout, and
    /// lists its tools. Returns its connection.
    async fn start_entry(&self, entry: &Arc<Entry>) -> Result<Peer<RoleClient>, String> {
        let _changing = entry.lifecycle.lock().await;
        if let Some(peer) = entry.peer() {
            return Ok(peer);
        }
        let ended = {
            let mut live = entry.live.lock().unwrap();
            live.state = ServerState::Starting;
            live.link.take()
        };
        // A start dropped before it ends leaves the server stopped.
        let mut starting = Starting(Some(entry));
        if let Some(ended) = ended {
            let_go(ended).await;
        }
        let timeout = self.timeout();
        let config = &entry.config;
        let started = tokio::time::timeout(timeout, async {
            let (link, peer) = if config.shared {
                let server = pool::acquire(config);
                let peer = server.start(timeout).await?;
                (Link::Shared(server), peer)
            } else {
                let connection = Connection::open(config)
                    .await
                    .map_err(|error| error.reason())?;
                let peer = connection.peer();
                (Link::Own(connection), peer)
            };
            let tools = peer.list_all_tools().await.map_err(|error| {
                log::warn!(
                    "MCP server '{}': its tools couldn't be listed: {}",
                    config.name,
                    Redactions::of(config).apply(&error.to_string())
                );
                "its tools couldn't be listed".to_string()
            })?;
            Ok::<_, String>((link, peer, tools))
        })
        .await
        .unwrap_or_else(|_| Err(format!("it didn't start within {} s", timeout.as_secs())));
        starting.0 = None;
        let mut live = entry.live.lock().unwrap();
        match started {
            Ok((link, peer, tools)) => {
                log::info!(
                    "Connected to MCP server '{}': {} tool(s) available",
                    config.name,
                    tools.len()
                );
                live.instructions = match &link {
                    Link::Own(connection) => connection.instructions(),
                    Link::Shared(server) => server.instructions(),
                };
                live.tools = Some(Arc::new(
                    tools.iter().map(McpToolInfo::from_listed).collect(),
                ));
                live.link = Some(link);
                live.state = ServerState::Running;
                Ok(peer)
            }
            Err(reason) => {
                log::warn!("MCP server '{}' didn't start: {reason}", config.name);
                live.state = ServerState::Failed(reason.clone());
                Err(reason)
            }
        }
    }

    /// Stops the server `name`, with what it started. A shared server is
    /// only let go, and stops once no hub uses it.
    pub async fn stop(&self, name: &str) -> Result<(), String> {
        let entry = self.entry(name)?;
        Self::stop_entry(&entry).await;
        Ok(())
    }

    async fn stop_entry(entry: &Entry) {
        let _changing = entry.lifecycle.lock().await;
        let link = {
            let mut live = entry.live.lock().unwrap();
            live.state = ServerState::Stopped;
            live.link.take()
        };
        if let Some(link) = link {
            let_go(link).await;
        }
    }

    /// How many other hubs use the shared server `name` with this one, or 0.
    pub(crate) fn others_sharing(&self, name: &str) -> usize {
        let Ok(entry) = self.entry(name) else {
            return 0;
        };
        let live = entry.live.lock().unwrap();
        match &live.link {
            Some(Link::Shared(server)) => server.users().saturating_sub(1),
            _ => 0,
        }
    }

    /// Adds the server `config` describes and starts it, and returns its
    /// facts: [`Running`](ServerState::Running), or
    /// [`Failed`](ServerState::Failed) with the reason when it didn't start
    /// (it stays added, and a later start tries again). A server of the
    /// same name is replaced. An error is a config without a name or a
    /// command.
    pub async fn add(&self, config: McpServerConfig) -> Result<ServerFacts, String> {
        if config.name.trim().is_empty() {
            return Err("An MCP server needs a name.".into());
        }
        if config.command.trim().is_empty() {
            return Err(format!("The MCP server {} needs a command.", config.name));
        }
        let entry = Self::entry_of(config);
        let replaced = {
            let mut entries = self.entries.lock().unwrap();
            match entries
                .iter()
                .position(|known| known.config.name == entry.config.name)
            {
                Some(index) => Some(std::mem::replace(&mut entries[index], entry.clone())),
                None => {
                    entries.push(entry.clone());
                    None
                }
            }
        };
        if let Some(replaced) = replaced {
            Self::stop_entry(&replaced).await;
        }
        let _ = self.start_entry(&entry).await;
        Ok(entry.facts())
    }

    /// Stops the server `name` and takes it out of the hub.
    pub async fn remove(&self, name: &str) -> Result<(), String> {
        let entry = {
            let mut entries = self.entries.lock().unwrap();
            let index = entries
                .iter()
                .position(|entry| entry.config.name == name)
                .ok_or_else(|| format!("There's no MCP server named {name}."))?;
            entries.remove(index)
        };
        Self::stop_entry(&entry).await;
        Ok(())
    }

    /// Every server's facts, in the order they were set up and added.
    pub fn servers(&self) -> Vec<ServerFacts> {
        let entries: Vec<Arc<Entry>> = self.entries.lock().unwrap().clone();
        entries.iter().map(|entry| entry.facts()).collect()
    }

    /// The server `name`'s config.
    pub fn config(&self, name: &str) -> Option<McpServerConfig> {
        self.entry(name).ok().map(|entry| entry.config.clone())
    }

    /// The tools of the server `name`, asked of the server now. A server
    /// that isn't running is started first. An error is a sentence saying
    /// what went wrong.
    pub async fn tools(&self, name: &str) -> Result<Vec<McpToolInfo>, String> {
        let entry = self.entry(name)?;
        let peer = match entry.peer() {
            Some(peer) => peer,
            None => self
                .start_entry(&entry)
                .await
                .map_err(|reason| couldnt_start(name, &reason))?,
        };
        let tools: Vec<McpToolInfo> = peer
            .list_all_tools()
            .await
            .map_err(|error| format!("Couldn't list the tools of {name} ({error})."))?
            .iter()
            .map(McpToolInfo::from_listed)
            .collect();
        entry.live.lock().unwrap().tools = Some(Arc::new(tools.clone()));
        Ok(tools)
    }

    /// The tools of the server `name` when they were last listed, without
    /// asking it.
    pub fn listed_tools(&self, name: &str) -> Option<Arc<Vec<McpToolInfo>>> {
        self.entry(name).ok()?.live.lock().unwrap().tools.clone()
    }

    /// Calls `tool` on the server `server` with `arguments`, and returns its
    /// result as [`McpTool`](super::McpTool) gives it: text, images,
    /// resources and structured content, a result marked `isError` as a
    /// tool error, and cancellation sent to the server when the call is
    /// dropped. A server that isn't running is started first.
    pub async fn call(
        &self,
        ctx: &TaskContext<'_>,
        server: &str,
        tool: &str,
        arguments: Map<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        self.call_reporting(ctx, server, tool, arguments)
            .await
            .map_err(|failure| failure.error)
    }

    /// [`call`](Self::call), saying whether the server rejected the
    /// arguments.
    pub(crate) async fn call_reporting(
        &self,
        _ctx: &TaskContext<'_>,
        server: &str,
        tool: &str,
        arguments: Map<String, Value>,
    ) -> Result<ToolOutput, CallFailure> {
        let entry = self.entry(server).map_err(ToolError::ExecutionFailed)?;
        let peer = match entry.peer() {
            Some(peer) => peer,
            None => self
                .start_entry(&entry)
                .await
                .map_err(|reason| ToolError::ExecutionFailed(couldnt_start(server, &reason)))?,
        };
        call_tool(&peer, server, tool, arguments).await
    }

    /// A tool for each tool of each server, as a conversation saved with
    /// them knows them: named by [`direct_tool_name`], with the server's
    /// description and schema, and its calls made through the hub. Servers
    /// whose tools were never listed (they didn't start) give none. A name
    /// taken by an earlier tool is left out.
    ///
    /// These are the main agent's. Each call first asks `access` whether
    /// its run may reach the server, as [`McpBridgeTool`](super::McpBridgeTool)
    /// does, and fails if it may not. Give workers
    /// [`worker_direct_tools`](Self::worker_direct_tools).
    pub fn direct_tools(self: &Arc<Self>, access: Arc<dyn McpAccess>) -> Vec<Arc<dyn Tool>> {
        self.direct_tools_for(access, false)
    }

    /// The [`direct_tools`](Self::direct_tools) for workers, with the same
    /// names and definitions, so a conversation saved with them keeps its
    /// tool list. A call also fails when the server's config sets
    /// [`workers`](McpServerConfig::workers) off, as the config is when the
    /// call is made.
    pub fn worker_direct_tools(self: &Arc<Self>, access: Arc<dyn McpAccess>) -> Vec<Arc<dyn Tool>> {
        self.direct_tools_for(access, true)
    }

    fn direct_tools_for(
        self: &Arc<Self>,
        access: Arc<dyn McpAccess>,
        worker: bool,
    ) -> Vec<Arc<dyn Tool>> {
        let entries: Vec<Arc<Entry>> = self.entries.lock().unwrap().clone();
        let mut taken = HashSet::new();
        let mut direct: Vec<Arc<dyn Tool>> = Vec::new();
        for entry in entries {
            let server = entry.config.name.clone();
            let Some(tools) = entry.live.lock().unwrap().tools.clone() else {
                continue;
            };
            for tool in tools.iter() {
                let name = direct_tool_name(&server, &tool.name);
                if !taken.insert(name.clone()) {
                    log::warn!(
                        "MCP server '{server}': tool '{}' is left out, as its name {name} is taken",
                        tool.name
                    );
                    continue;
                }
                direct.push(Arc::new(DirectTool {
                    hub: self.clone(),
                    access: access.clone(),
                    worker,
                    server: server.clone(),
                    tool: tool.name.clone(),
                    definition: mcp_definition(name, &server, tool),
                }));
            }
        }
        direct
    }

    /// Stops every server, each given three seconds to exit once its
    /// connection closes before it is killed with what it started. Shared
    /// servers are let go.
    pub async fn shutdown(&self) {
        let entries: Vec<Arc<Entry>> = self.entries.lock().unwrap().clone();
        futures::future::join_all(entries.iter().map(|entry| Self::stop_entry(entry))).await;
    }
}

/// A tool of a server as a tool of its own, called through the hub, for the
/// runs `access` lets reach the server.
struct DirectTool {
    hub: Arc<McpHub>,
    access: Arc<dyn McpAccess>,
    /// A worker's: a server kept to the main agent is refused.
    worker: bool,
    server: String,
    tool: String,
    definition: ToolDefinition,
}

impl DirectTool {
    /// Whether `ctx`'s run may reach the server now.
    fn reaches(&self, ctx: &TaskContext<'_>) -> bool {
        if self.worker
            && self
                .hub
                .config(&self.server)
                .is_some_and(|config| !config.workers)
        {
            return false;
        }
        self.access
            .servers_for(ctx)
            .is_none_or(|allowed| allowed.contains(&self.server))
    }
}

impl std::fmt::Debug for DirectTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectTool")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for DirectTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// A tool the server marks read-only runs again when its call was
    /// running as the task stopped. Any other is in doubt.
    fn recovery_mode(&self) -> crate::tool::ToolRecovery {
        if self.definition.parallel_safe {
            crate::tool::ToolRecovery::Rerun
        } else {
            crate::tool::ToolRecovery::InDoubt
        }
    }

    /// A request to the server, whose response a job can go on awaiting
    /// when the call moves to the background.
    fn background_mode(&self) -> ToolBackground {
        ToolBackground::Detach
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.execute_with_output(context, params)
            .await
            .map(|output| output.text)
    }

    async fn execute_with_output(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        if !self.reaches(context) {
            return Err(ToolError::ExecutionFailed(format!(
                "{} isn't one of your MCP servers.",
                self.server
            )));
        }
        self.hub
            .call(
                context,
                &self.server,
                &self.tool,
                params.into_iter().collect(),
            )
            .await
    }
}
