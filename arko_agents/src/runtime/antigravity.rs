//! Antigravity's ACP session and MCP bridge for the shared CLI runtime.

use super::*;
use arko_ai::antigravity as agy;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, thiserror::Error)]
pub(super) enum CliError {
    #[error(transparent)]
    Claude(#[from] cc::ClaudeError),
    #[error(transparent)]
    Antigravity(#[from] agy::AcpError),
    #[error("{0}")]
    Setup(String),
    #[error("{0}")]
    BrokenSetup(String),
    #[error("The Antigravity profile is missing.")]
    MissingProfile,
}

pub(super) const UNTRUSTED_AGENT: &str = "Antigravity didn't load Arko's agent.";

pub(super) enum CliSession {
    Claude(Box<cc::ClaudeSession>),
    Antigravity(Box<AntigravityRun>),
}

pub(super) enum CliEvent {
    Claude(cc::Message),
    Antigravity(agy::AcpEvent),
}

impl CliSession {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Claude(_) => "Claude Code",
            Self::Antigravity(_) => "Antigravity",
        }
    }

    pub(super) async fn send(&self, text: &str) -> Result<String, CliError> {
        self.send_content(Value::String(text.to_string())).await
    }

    pub(super) async fn send_content(&self, content: Value) -> Result<String, CliError> {
        match self {
            Self::Claude(session) => Ok(session.send_content(content).await?),
            Self::Antigravity(session) => {
                let blocks = match content {
                    Value::String(text) => vec![agy::text_block(&text)],
                    Value::Array(blocks) => antigravity_blocks(&blocks),
                    _ => unreachable!("A CLI prompt has text or content blocks"),
                };
                session.prompt(blocks).await
            }
        }
    }

    pub(super) async fn send_human(&self, text: &str) -> Result<String, CliError> {
        match self {
            Self::Claude(session) => Ok(session.send_human(text).await?),
            Self::Antigravity(session) => session.prompt(vec![agy::text_block(text)]).await,
        }
    }

    pub(super) async fn next_message(&mut self) -> Option<CliEvent> {
        match self {
            Self::Claude(session) => session.next_message().await.map(CliEvent::Claude),
            Self::Antigravity(session) => {
                session.client.next_event().await.map(CliEvent::Antigravity)
            }
        }
    }

    pub(super) async fn interrupt(&self) -> Result<(), CliError> {
        match self {
            Self::Claude(session) => Ok(session.interrupt().await?),
            Self::Antigravity(session) => Ok(session.client.cancel(&session.id).await?),
        }
    }

    pub(super) async fn set_model(&self, model: Option<&str>) -> Result<(), CliError> {
        match self {
            Self::Claude(session) => Ok(session.set_model(model).await?),
            Self::Antigravity(session) => {
                if let Some(model) = model {
                    session
                        .client
                        .set_config_option(&session.id, "model", model)
                        .await?;
                }
                Ok(())
            }
        }
    }

    pub(super) async fn close(self) -> Result<(), CliError> {
        match self {
            Self::Claude(session) => Ok(session.close().await.map(drop)?),
            Self::Antigravity(session) => Ok(session.client.close().await.map(drop)?),
        }
    }
}

/// Joins text before handing it to ACP. ACP does not set the separator
/// between adjacent text blocks.
fn antigravity_blocks(blocks: &[Value]) -> Vec<Value> {
    let texts: Vec<_> = blocks
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect();
    let mut prompt = Vec::new();
    if !texts.is_empty() {
        prompt.push(agy::text_block(&texts.join("\n\n")));
    }
    prompt.extend(blocks.iter().filter_map(|block| {
        Some(agy::image_block(
            block.pointer("/source/media_type")?.as_str()?,
            block.pointer("/source/data")?.as_str()?,
        ))
    }));
    prompt
}

/// A private profile. One a task keeps holds the CLI's copy of the
/// conversation and outlives the run. Any other is removed when the run ends,
/// after the CLI is dropped.
struct ProfileCleanup {
    profile: agy::Profile,
    kept: bool,
}

impl Drop for ProfileCleanup {
    fn drop(&mut self) {
        if self.kept {
            return;
        }
        let dir = self.profile.dir().to_path_buf();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn_blocking(move || {
                if let Err(error) = std::fs::remove_dir_all(&dir) {
                    log::warn!(
                        "Could not remove Antigravity profile {}. {error}",
                        dir.display()
                    );
                }
            });
        }
    }
}

/// What every launch of a run's CLI shares.
struct Launch {
    cli: PathBuf,
    cwd: PathBuf,
    env: Vec<(String, String)>,
    family: String,
    thought_level: Option<String>,
    system_prompt: String,
    tools: Vec<String>,
    helper: agy::HelperCommand,
    relay: agy::RelayHost,
}

/// A started CLI with its session set to the task's model.
struct Opened {
    client: agy::AcpClient,
    profile: ProfileCleanup,
    id: String,
    tier: Option<String>,
    display_name: Option<String>,
}

/// The CLI, profile and relay of a single run. The host's gate and tool
/// runner are shared with Claude Code, not part of the relay.
pub(super) struct AntigravityRun {
    client: agy::AcpClient,
    _relay: agy::RelayHost,
    profile: ProfileCleanup,
    id: String,
    latest_prompt: AtomicU64,
    cli: PathBuf,
    env: Vec<(String, String)>,
    tier: Option<String>,
    display_name: Option<String>,
    resumed: bool,
}

impl AntigravityRun {
    /// Starts the CLI. With a profile root, a task keeps its profile in
    /// `ROOT/<task id>` and its session ([`Task::antigravity_session`]), and
    /// the run resumes the session the task saved when the system prompt is
    /// the same. Otherwise, or when agy can't resume it, the profile is
    /// written afresh and a new session starts.
    pub(super) async fn start(
        run: &mut ExternalCliRun,
        requests: mpsc::UnboundedSender<HostRequest>,
    ) -> Result<Self, CliError> {
        let config = &run.task.agent_config;
        let settings = config.antigravity.clone().unwrap_or_default();
        let model = config.llm_config.model.clone();
        let family = model.antigravity_family().ok_or_else(|| {
            CliError::Setup(format!("Antigravity does not offer model '{}'.", model.id))
        })?;
        let cwd = run.task.current_working_directory.clone();
        let system_prompt = run
            .task
            .conversation
            .system_message
            .clone()
            .unwrap_or_default();
        let helper = settings.helper_command.ok_or_else(|| {
            CliError::Setup(
                "Set AntigravitySettings.helper_command to run an Antigravity agent.".into(),
            )
        })?;
        let program = settings.cli_path;
        let cli = tokio::task::spawn_blocking(move || {
            let cli = program.or_else(agy::find_agy).ok_or_else(|| {
                CliError::Setup(
                    "Install the Antigravity CLI and sign in by running `agy` once.".into(),
                )
            })?;
            std::fs::canonicalize(cli).map_err(io_error)
        })
        .await
        .map_err(|error| CliError::Setup(error.to_string()))??;
        let (server, tools) = run.antigravity_tool_server(requests);
        let relay = agy::RelayHost::bind(Arc::new(server))
            .await
            .map_err(io_error)?;
        let launch = Launch {
            cli,
            cwd,
            env: settings.env.into_iter().collect(),
            family: family.to_string(),
            thought_level: settings.thought_level,
            system_prompt,
            tools,
            helper,
            relay,
        };
        let digest = agy::prompt_digest(&launch.system_prompt);
        let task = run.task.id;
        let kept = settings
            .profile_root
            .map(std::path::absolute)
            .transpose()
            .map_err(io_error)?
            .map(|root| root.join(task.to_string()));
        if run
            .task
            .antigravity_session
            .as_ref()
            .is_some_and(|saved| saved.pending_from.is_some())
        {
            run.discard_antigravity_session();
        }
        let saved = run.task.antigravity_session.take().filter(|saved| {
            let same = saved.prompt == digest;
            if !same {
                log::info!(
                    "Task {task}: The system prompt changed since the Antigravity session started, so a new one starts from the conversation's record."
                );
            }
            same && kept.is_some()
        });
        if let (Some(dir), Some(saved)) = (&kept, saved) {
            match launch.open(dir, Some(&saved.id), true).await {
                Ok(opened) => {
                    run.task.antigravity_session = Some(saved);
                    return Ok(Self::new(launch, opened, true));
                }
                Err(error)
                    if matches!(&error, CliError::MissingProfile)
                        || matches!(&error, CliError::Antigravity(error) if error.cannot_read_session(&saved.id)) =>
                {
                    log::info!(
                        "Task {task}: Antigravity couldn't read the saved session. Starting from the conversation record. {error}"
                    );
                }
                Err(error) => {
                    run.task.antigravity_session = Some(saved);
                    return Err(error);
                }
            }
        }
        let opened = match &kept {
            Some(dir) => {
                remove_profile(dir.clone()).await?;
                launch.open(dir, None, true).await?
            }
            None => {
                let dir = std::path::absolute(std::env::temp_dir())
                    .map_err(io_error)?
                    .join(format!("arko-agy-{}", Uuid::new_v4()));
                launch.open(&dir, None, false).await?
            }
        };
        if kept.is_some() {
            run.task.antigravity_session = Some(crate::AntigravitySession {
                id: opened.id.clone(),
                prompt: digest,
                pending_from: None,
            });
        }
        Ok(Self::new(launch, opened, false))
    }

    fn new(launch: Launch, opened: Opened, resumed: bool) -> Self {
        Self {
            client: opened.client,
            _relay: launch.relay,
            profile: opened.profile,
            id: opened.id,
            latest_prompt: AtomicU64::new(0),
            cli: launch.cli,
            env: launch.env,
            tier: opened.tier,
            display_name: opened.display_name,
            resumed,
        }
    }

    /// Whether the run resumed the session the task saved, which already
    /// holds the conversation.
    pub(super) fn resumed(&self) -> bool {
        self.resumed
    }

    /// The profile the task keeps after the run, if it keeps one.
    pub(super) fn kept_profile(&self) -> Option<PathBuf> {
        self.profile
            .kept
            .then(|| self.profile.profile.dir().to_path_buf())
    }

    pub(super) fn plan_usage(&self) -> oneshot::Receiver<Result<Value, String>> {
        let (reply, answer) = oneshot::channel();
        let (cli, env, tier, model) = (
            self.cli.clone(),
            self.env.clone(),
            self.tier.clone(),
            self.display_name.clone(),
        );
        tokio::spawn(async move {
            let result = shared_usage(cli, env).await.and_then(|groups| {
                let mut usage = agy::plan_usage_report(groups, model.as_deref());
                usage.plan_type = tier;
                serde_json::to_value(usage).map_err(|error| error.to_string())
            });
            let _ = reply.send(result);
        });
        answer
    }

    pub(super) async fn tripped(&self) -> bool {
        profile_tripped(&self.profile.profile).await
    }

    async fn prompt(&self, blocks: Vec<Value>) -> Result<String, CliError> {
        if self.latest_prompt.load(Ordering::Relaxed) == 0 {
            check_profile(&self.profile.profile).await?;
        }
        let request = self.client.prompt(&self.id, blocks).await?;
        self.latest_prompt.store(request, Ordering::Relaxed);
        Ok(Uuid::new_v4().to_string())
    }
}

impl Launch {
    /// Starts the CLI on the profile `dir`. With `resume`, the profile there
    /// is written again and its session `resume` resumed. Without, the profile
    /// is written in `dir`, which must not exist, and a new session starts.
    /// The session is then set to the task's model and thought level. A
    /// `kept` profile stays when the run ends. On an error the CLI has exited.
    async fn open(&self, dir: &Path, resume: Option<&str>, kept: bool) -> Result<Opened, CliError> {
        let (dir, address, resuming) = (dir.to_path_buf(), self.relay.address(), resume.is_some());
        let (system_prompt, tools, helper) = (
            self.system_prompt.clone(),
            self.tools.clone(),
            self.helper.clone(),
        );
        let profile = tokio::task::spawn_blocking(move || {
            let spec = agy::ProfileSpec {
                system_prompt: &system_prompt,
                tools: &tools,
                helper: &helper,
            };
            let profile = if resuming {
                agy::Profile::update(&dir, &spec).map_err(|error| {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        CliError::MissingProfile
                    } else {
                        io_error(error)
                    }
                })?
            } else {
                agy::Profile::write(&dir, &spec).map_err(io_error)?
            };
            agy::write_address_file(profile.relay_file(), address).map_err(io_error)?;
            Ok::<_, CliError>(profile)
        })
        .await
        .map_err(|error| CliError::Setup(error.to_string()))??;
        let profile = ProfileCleanup { profile, kept };
        let mut options = agy::AcpOptions::new(
            self.cli.clone(),
            profile.profile.dir().to_path_buf(),
            self.cwd.clone(),
        );
        options.agent = Some(agy::AGENT_NAME.into());
        options.client_info = agy::ClientInfo {
            name: "arko_agents".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        };
        options.log_file = Some(profile.profile.log_file());
        options.env.clone_from(&self.env);
        let (key, token) = self.relay.env();
        options.env.push((key.into(), token.into()));
        let client = agy::AcpClient::start(options, Arc::new(agy::ArkoToolsOnly)).await?;
        match self.prepare(&client, &profile.profile, resume).await {
            Ok((id, tier, display_name)) => Ok(Opened {
                client,
                profile,
                id,
                tier,
                display_name,
            }),
            Err(error) => {
                // So that it writes nothing more in the profile.
                let _ = client.close().await;
                Err(error)
            }
        }
    }

    /// Opens the session and sets its model and thought level. Returns its
    /// ID, its plan and the model's name.
    async fn prepare(
        &self,
        client: &agy::AcpClient,
        profile: &agy::Profile,
        resume: Option<&str>,
    ) -> Result<(String, Option<String>, Option<String>), CliError> {
        let version = client.version().unwrap_or("unknown");
        if !agy::version_at_least(version, agy::MINIMUM_VERSION) {
            return Err(CliError::Setup(format!(
                "Antigravity {version} is not supported. Install {} or later.",
                agy::MINIMUM_VERSION
            )));
        }
        let session = match resume {
            Some(id) => client.resume_session(id, &self.cwd).await?,
            None => client.new_session(&self.cwd).await?,
        };
        check_profile(profile).await?;
        let display_name = session
            .config_options
            .iter()
            .find(|o| o.id == "model")
            .and_then(|o| {
                o.options
                    .iter()
                    .find(|option| option.value == self.family)
                    .map(|option| option.name.clone())
            });
        if display_name.is_none() {
            return Err(CliError::Setup(format!(
                "Antigravity does not offer '{}' under this sign-in.",
                self.family
            )));
        }
        let config = client
            .set_config_option(&session.session_id, "model", &self.family)
            .await?;
        let levels = agy::config_values(&config, "thought_level");
        if let Some(level) = self
            .thought_level
            .as_deref()
            .and_then(|wanted| thought_level(wanted, &levels))
        {
            client
                .set_config_option(&session.session_id, "thought_level", level)
                .await?;
        }
        Ok((session.session_id, session.tier, display_name))
    }
}

/// Removes the profile `dir` and what agy kept in it, if it's there.
pub(super) async fn remove_profile(dir: PathBuf) -> Result<(), CliError> {
    tokio::task::spawn_blocking(move || match std::fs::remove_dir_all(&dir) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(io_error(error)),
        _ => Ok(()),
    })
    .await
    .map_err(|error| CliError::Setup(error.to_string()))?
}

type UsageResult = Result<Vec<agy::UsageGroup>, String>;
type UsageReading = tokio::sync::watch::Receiver<Option<(tokio::time::Instant, UsageResult)>>;
type UsageCache = tokio::sync::Mutex<HashMap<(PathBuf, u64), UsageReading>>;

async fn shared_usage(cli: PathBuf, env: Vec<(String, String)>) -> UsageResult {
    static CACHE: OnceLock<UsageCache> = OnceLock::new();
    let mut hash = std::hash::DefaultHasher::new();
    env.hash(&mut hash);
    let key = (cli.clone(), hash.finish());
    let fresh = |reading: &UsageReading| {
        reading
            .borrow()
            .as_ref()
            .is_none_or(|(at, _)| at.elapsed() < Duration::from_secs(60))
    };
    let mut receiver = {
        let mut cache = CACHE.get_or_init(Default::default).lock().await;
        cache.retain(|_, reading| fresh(reading));
        if let Some(reading) = cache.get(&key) {
            reading.clone()
        } else {
            let (sender, receiver) = tokio::sync::watch::channel(None);
            cache.insert(key, receiver.clone());
            // The command belongs to the cache, so a run that stops does not
            // stop the usage read for every other run waiting on it.
            tokio::spawn(async move {
                let result = agy::plan_usage(&cli, &env).await;
                sender.send_replace(Some((tokio::time::Instant::now(), result)));
            });
            receiver
        }
    };
    let result = receiver
        .wait_for(Option::is_some)
        .await
        .map_err(|error| error.to_string())?;
    result.as_ref().expect("usage arrived").1.clone()
}

fn io_error(error: std::io::Error) -> CliError {
    CliError::Antigravity(agy::AcpError::Io(error))
}

/// A log refusal comes before the first prompt. The tripwire is checked
/// again as the turn runs, in case a fallback got as far as its hooks.
async fn check_profile(profile: &agy::Profile) -> Result<(), CliError> {
    let fallback = agy::log_shows_fallback(&profile.log_file(), agy::AGENT_NAME)
        .await
        .map_err(|error| {
            CliError::BrokenSetup(format!(
                "Could not check that Antigravity loaded Arko's agent. {error}"
            ))
        })?;
    if fallback || profile_tripped(profile).await {
        return Err(CliError::BrokenSetup(UNTRUSTED_AGENT.into()));
    }
    Ok(())
}

async fn profile_tripped(profile: &agy::Profile) -> bool {
    tokio::fs::metadata(profile.tripwire_file()).await.is_ok()
}

fn thought_level<'a>(wanted: &str, offered: &[&'a str]) -> Option<&'a str> {
    const LEVELS: &[&str] = &["low", "medium", "high", "xhigh"];
    if let Some(&exact) = offered.iter().find(|level| **level == wanted) {
        return Some(exact);
    }
    let rank = LEVELS.iter().position(|&level| level == wanted)?;
    let mut ranked = offered
        .iter()
        .filter_map(|&level| {
            LEVELS
                .iter()
                .position(|&known| level == known)
                .map(|rank| (rank, level))
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|(rank, _)| *rank);
    ranked
        .iter()
        .rev()
        .find(|(offered, _)| *offered < rank)
        .or_else(|| ranked.first())
        .map(|(_, level)| *level)
}

struct AgTool {
    tool: ServedTool,
    requests: mpsc::UnboundedSender<HostRequest>,
}

#[async_trait]
impl cc::ClaudeTool for AgTool {
    fn name(&self) -> String {
        self.tool.name.clone()
    }
    fn description(&self) -> String {
        self.tool.description.clone()
    }
    fn input_schema(&self) -> Value {
        self.tool.schema.clone()
    }
    fn read_only(&self) -> bool {
        self.tool.read_only
    }

    async fn call(&self, input: Value) -> cc::ToolOutput {
        let call_id = Uuid::new_v4().to_string();
        // The CLI's permission checks only the tool's name. Arko's approval
        // and execution both use the arguments that arrive over MCP.
        let served = format!("mcp__{ARKO_TOOL_SERVER}__{}", self.tool.name);
        let (reply, answer) = oneshot::channel();
        let queued = self.requests.send(HostRequest::PreToolUse {
            tool_use_id: Some(call_id.clone()),
            tool_name: served.clone(),
            input: input.clone(),
            reply,
        });
        if queued.is_err() {
            return cc::ToolOutput::error(STOPPING);
        }
        let Ok(answer) = answer.await else {
            return cc::ToolOutput::error(STOPPING);
        };
        let answer = &answer["hookSpecificOutput"];
        let refusal = match answer["permissionDecision"].as_str() {
            Some("allow") => None,
            Some("ask") => {
                let (reply, decision) = oneshot::channel();
                let request = cc::PermissionRequest {
                    tool_name: served,
                    input: input.clone(),
                    tool_use_id: Some(call_id.clone()),
                    agent_id: None,
                    blocked_path: None,
                    decision_reason: None,
                    title: None,
                    description: None,
                    suggestions: Vec::new(),
                    raw: Value::Null,
                };
                if self
                    .requests
                    .send(HostRequest::Permission { request, reply })
                    .is_err()
                {
                    return cc::ToolOutput::error(STOPPING);
                }
                match decision.await {
                    Ok(cc::PermissionDecision::Allow { .. }) => None,
                    Ok(cc::PermissionDecision::Deny { message, .. }) => Some(message),
                    Err(_) => return cc::ToolOutput::error(STOPPING),
                }
            }
            _ => Some(
                answer["permissionDecisionReason"]
                    .as_str()
                    .unwrap_or(STOPPING)
                    .to_string(),
            ),
        };
        let (reply, output) = oneshot::channel();
        let request = match refusal {
            Some(reason) => HostRequest::AntigravityRefusal {
                call_id,
                reply,
                reason,
            },
            None => HostRequest::AntigravityExecute { call_id, reply },
        };
        if self.requests.send(request).is_err() {
            return cc::ToolOutput::error(STOPPING);
        }
        output
            .await
            .unwrap_or_else(|_| cc::ToolOutput::error(STOPPING))
    }
}

/// The host owns a running call. The relay owns only the reply receiver.
pub(super) struct AntigravityCallResult {
    pub(super) call_id: String,
    pub(super) output: cc::ToolOutput,
    pub(super) reply: oneshot::Sender<cc::ToolOutput>,
}

/// The text and usage of the current ACP turn. Notes change the ACP request
/// that ends it without starting a new model turn.
#[derive(Default)]
pub(super) struct AntigravityTurn {
    text: String,
    warning: Option<String>,
    context_limit: Option<usize>,
    folded_usage: Option<agy::PromptUsage>,
}

impl AntigravityTurn {
    fn count_usage(&mut self, usage: agy::PromptUsage) {
        let total = self.folded_usage.get_or_insert_default();
        total.input_tokens = total.input_tokens.saturating_add(usage.input_tokens);
        total.output_tokens = total.output_tokens.saturating_add(usage.output_tokens);
        total.cached_read_tokens = total
            .cached_read_tokens
            .saturating_add(usage.cached_read_tokens);
        total.thought_tokens = total.thought_tokens.saturating_add(usage.thought_tokens);
        total.total_tokens = total.total_tokens.saturating_add(usage.total_tokens);
    }
}

impl ExternalCliRun {
    /// A saved turn without a result may or may not be in the CLI's copy.
    /// Keep the host's input and outcomes, and start the next run from them.
    fn discard_antigravity_session(&mut self) {
        if let Some(from) = self
            .task
            .antigravity_session
            .take()
            .and_then(|saved| saved.pending_from)
        {
            self.task.claude_code_synced = from;
        }
    }

    pub(super) fn mark_antigravity_turn(&mut self, from: usize) {
        if let Some(saved) = &mut self.task.antigravity_session {
            saved.pending_from.get_or_insert(from);
        }
    }

    fn antigravity_tool_server(
        &mut self,
        requests: mpsc::UnboundedSender<HostRequest>,
    ) -> (cc::ToolServer, Vec<String>) {
        let server = cc::ToolServer::new(ARKO_TOOL_SERVER);
        let mut names = self.registry.list_tool_names();
        names.sort();
        let tools = self.served_definitions();
        let host = self.tool_host();
        let server = tools
            .iter()
            .fold(server, |server, (definition, read_only)| {
                server.tool(Arc::new(AgTool {
                    tool: ServedTool {
                        name: definition.name.clone(),
                        description: definition.description.clone(),
                        schema: definition.parameters.clone(),
                        read_only: *read_only,
                        host: host.clone(),
                    },
                    requests: requests.clone(),
                }))
            });
        (server, names)
    }

    /// The first prompt of a new session for a conversation that went on
    /// before it, from the messages the last session had (`synced`). `None`
    /// when nothing came before the new input, which then goes as it is.
    pub(super) fn antigravity_seed(&self, synced: usize) -> Option<Value> {
        seed(&self.task.conversation.messages, synced).map(|content| input_blocks(content.iter()))
    }

    pub(super) async fn start_antigravity_call(
        &mut self,
        call_id: String,
        reply: oneshot::Sender<cc::ToolOutput>,
        calls: &mut JoinSet<AntigravityCallResult>,
    ) {
        let host = self.tool_host();
        let Some(call) = self.calls.get(&call_id) else {
            return;
        };
        let Some(tool) = host.registry.get(&call.name) else {
            return;
        };
        let Some(admitted) = host.take_admitted(&call.name, &call.input, Some(&call_id)) else {
            return;
        };
        if self.ending() || reply.is_closed() {
            return;
        }
        let (name, input) = (call.name.clone(), call.input.clone());
        self.record_running(Some(&call_id), admitted.custom_output.clone())
            .await;
        calls.spawn(async move {
            let output = host.run_admitted(&name, tool, &input, admitted).await;
            AntigravityCallResult {
                call_id,
                output,
                reply,
            }
        });
    }

    pub(super) async fn finish_antigravity_call(
        &mut self,
        completed: AntigravityCallResult,
        session: &CliSession,
        in_turn: bool,
    ) -> Result<(), ArkoAgentsError> {
        let AntigravityCallResult {
            call_id,
            output,
            reply,
        } = completed;
        let Some(call) = self.calls.remove(&call_id) else {
            return Ok(());
        };
        let result = if output.is_error {
            ai::ToolResult::error_from_tool_call(&call.record, &output.text)
        } else {
            ai::ToolResult::from_tool_call(&call.record, &output.text)
        }
        .with_images(
            output
                .images
                .iter()
                .cloned()
                .map(|image| ai::ToolResultImage::new("", image))
                .collect(),
        );
        self.commit(ai::Message::new_user_with_contents(vec![
            ai::Content::ToolResult(result),
        ]));
        self.task
            .running_tool_calls
            .retain(|running| running.call_id.as_deref() != Some(call_id.as_str()));
        self.report(true).await;
        self.call_finished(
            call.id,
            &call.name,
            &call.input,
            &output.text,
            output.is_error,
            session,
        )
        .await?;
        self.report(true).await;
        if in_turn {
            self.send_antigravity_context(session).await?;
        }
        let _ = reply.send(output);
        Ok(())
    }

    pub(super) async fn send_antigravity_context(
        &mut self,
        session: &CliSession,
    ) -> Result<(), ArkoAgentsError> {
        if self.ending()
            || self.pause_after_turn.cause().is_some()
            || self.add_all_context().is_none()
        {
            return Ok(());
        }
        // The turn is already pending. Save this note with it before the
        // model can act on it or the host receives its receipt.
        self.report(true).await;
        let content = &self
            .task
            .conversation
            .messages
            .last()
            .expect("context was added")
            .content;
        let input = input_blocks(content.iter());
        match session.send_content(input).await {
            Ok(_) => {
                self.task.claude_code_synced = self.task.conversation.messages.len();
                self.emit_context_receipts().await?;
            }
            Err(error) => {
                // The recorded message is in the seed if the connection ends.
                log::warn!(
                    "Task {}: Could not send Antigravity a note. {error}",
                    self.task.id
                );
            }
        }
        Ok(())
    }

    pub(super) async fn antigravity_setup_failed(
        &mut self,
        message: String,
    ) -> Result<Option<SessionEnd>, ArkoAgentsError> {
        // A profile agy didn't run as set up isn't resumed.
        self.task.antigravity_session = None;
        self.emit(AgentEvent::ErrorEncountered {
            error_message: message.clone(),
            requires_approval_to_retry: false,
            tool_call_id: None,
            error_kind: None,
        })
        .await?;
        self.set_state(TaskState::Failed {
            reason: message,
            cause: Some(ErrorRecoveryCause::NonRetryable),
            error_kind: None,
        })
        .await?;
        Ok(Some(SessionEnd::Stop))
    }

    pub(super) async fn antigravity_event(
        &mut self,
        event: agy::AcpEvent,
        turn: &mut AntigravityTurn,
        session: &mut CliSession,
    ) -> Result<Option<SessionEnd>, ArkoAgentsError> {
        let CliSession::Antigravity(agy_session) = session else {
            unreachable!()
        };
        if matches!(
            &event,
            agy::AcpEvent::PromptFinished { .. }
                | agy::AcpEvent::Permission { .. }
                | agy::AcpEvent::Update {
                    update: agy::SessionUpdate::ToolCall(_) | agy::SessionUpdate::ToolCallUpdate(_),
                    ..
                }
        ) && agy_session.tripped().await
        {
            let _ = session.interrupt().await;
            return self.antigravity_setup_failed(UNTRUSTED_AGENT.into()).await;
        }
        let immediately_failed = matches!(&event, agy::AcpEvent::PromptFinished { request, outcome: Ok(result) }
            if *request == agy_session.latest_prompt.load(Ordering::Relaxed)
                && turn.warning.as_deref().and_then(|text| agy::turn_failure(text, result)).is_some());
        if !immediately_failed {
            self.flush_antigravity_warning(turn).await?;
        }
        match event {
            agy::AcpEvent::Update { session_id, update } if session_id == agy_session.id => {
                match update {
                    agy::SessionUpdate::AgentMessage { text }
                        if text.trim_start().starts_with('⚠') =>
                    {
                        turn.warning = Some(text)
                    }
                    agy::SessionUpdate::AgentMessage { text } => {
                        self.streamed_bytes.0 += text.len();
                        turn.text.push_str(&text);
                        self.emit(AgentEvent::AssistantResponseChunk(text)).await?;
                    }
                    agy::SessionUpdate::AgentThought { text } => {
                        self.streamed_bytes.1 += text.len();
                        self.emit(AgentEvent::ReasoningChunk(text)).await?;
                    }
                    agy::SessionUpdate::Usage { used, size } => {
                        turn.context_limit = Some(size as usize);
                        if used > 0 {
                            self.context_size = Some(used as usize);
                            self.first_context_size.get_or_insert(used as usize);
                            self.check_handover_threshold(turn.context_limit).await?;
                        }
                    }
                    agy::SessionUpdate::ToolCall(call)
                    | agy::SessionUpdate::ToolCallUpdate(call) => {
                        // ACP names tell us which tools agy tried to run.
                        // Only the MCP requests and replies describe Arko calls.
                        if let Some(name) = call.name.as_deref()
                            && !name.starts_with(agy::ARKO_TOOL_PREFIX)
                            && !matches!(name, "list_resources" | "read_resource" | "manage_task")
                        {
                            let _ = session.interrupt().await;
                            return self
                                .antigravity_setup_failed(format!(
                                    "Antigravity tried to run '{name}', which is not an Arko tool."
                                ))
                                .await;
                        }
                    }
                    _ => {}
                }
            }
            agy::AcpEvent::Permission {
                request,
                decision: agy::PermissionDecision::Cancel,
            } => {
                return self.antigravity_setup_failed(format!("Antigravity asked for permission for '{}'. Only Arko tools can be allowed.", request.tool_name.as_deref().unwrap_or("an unknown tool"))).await;
            }
            agy::AcpEvent::PromptFinished { request, outcome }
                if request != agy_session.latest_prompt.load(Ordering::Relaxed) =>
            {
                // A note ended this ACP request while the logical turn goes on.
                // agy reports distinct counts for the two replies.
                if let Ok(result) = outcome
                    && let Some(usage) = result.usage
                {
                    turn.count_usage(usage);
                }
            }
            agy::AcpEvent::PromptFinished { outcome, .. } => {
                let result = match outcome {
                    Ok(result) => result,
                    Err(error) => {
                        return self.session_failed(error.to_string()).await.map(Some);
                    }
                };
                if let Some(message) = turn
                    .warning
                    .take()
                    .and_then(|text| agy::turn_failure(&text, &result))
                {
                    self.discard_unfinished_stream().await?;
                    if let Some(usage) = turn.folded_usage.take() {
                        self.antigravity_usage(usage, &result.stop_reason, turn.context_limit)
                            .await?;
                    }
                    return self.antigravity_failed(message).await.map(Some);
                }
                self.agy_retried = false;
                if let Some(saved) = &mut self.task.antigravity_session {
                    saved.pending_from = None;
                }
                let text = turn.text.clone();
                self.commit_antigravity_text(turn).await?;
                self.streamed_bytes = (0, 0);
                self.latest_response.stop_reason = Some(result.stop_reason.clone());
                if let Some(usage) = result.usage {
                    turn.count_usage(usage);
                }
                if let Some(usage) = turn.folded_usage.take() {
                    self.antigravity_usage(usage, &result.stop_reason, turn.context_limit)
                        .await?;
                }
                let result = cc::ResultMessage {
                    subtype: "success".into(),
                    is_error: false,
                    num_turns: 1,
                    session_id: agy_session.id.clone(),
                    duration_ms: 0,
                    result: (!text.is_empty()).then_some(text),
                    total_cost_usd: None,
                    usage: None,
                    stop_reason: Some(result.stop_reason),
                    raw: Value::Null,
                };
                let limit = turn.context_limit;
                *turn = AntigravityTurn {
                    context_limit: limit,
                    ..AntigravityTurn::default()
                };
                return self.turn_ended(&result, session).await;
            }
            _ => {}
        }
        Ok(None)
    }

    pub(super) async fn flush_antigravity_warning(
        &mut self,
        turn: &mut AntigravityTurn,
    ) -> Result<(), ArkoAgentsError> {
        if let Some(text) = turn.warning.take() {
            self.streamed_bytes.0 += text.len();
            turn.text.push_str(&text);
            self.emit(AgentEvent::AssistantResponseChunk(text)).await?;
        }
        Ok(())
    }

    pub(super) async fn commit_antigravity_text(
        &mut self,
        turn: &mut AntigravityTurn,
    ) -> Result<(), ArkoAgentsError> {
        if !turn.text.is_empty() {
            let text = std::mem::take(&mut turn.text);
            self.commit(ai::Message::new_assistant(&text));
            self.report(false).await;
        }
        Ok(())
    }

    async fn antigravity_failed(&mut self, reason: String) -> Result<SessionEnd, ArkoAgentsError> {
        if !self.ending() && !std::mem::replace(&mut self.agy_retried, true) {
            self.emit(AgentEvent::ServiceWarning {
                text: format!(
                    "Antigravity could not finish the turn. Retrying in a new session. {reason}"
                ),
            })
            .await?;
            self.add_user_message(
                "The previous response failed. Continue the task from the conversation record.",
            )?;
            // The failure may stay with the session, so a new one starts.
            self.discard_antigravity_session();
            return Ok(SessionEnd::Restart);
        }
        self.session_failed(reason).await
    }

    async fn antigravity_usage(
        &mut self,
        usage: agy::PromptUsage,
        stop: &str,
        limit: Option<usize>,
    ) -> Result<(), ArkoAgentsError> {
        // ACP reports uncached input separately from cached input. Thought
        // tokens are part of the output count. The counts add up every model
        // request of the turn, and the conversation's size is the context
        // agy reported last.
        let input = usage.input_tokens.saturating_add(usage.cached_read_tokens) as usize;
        let output = usage.output_tokens as usize;
        let mut info = ai::UsageInfo::new(input, output)
            .with_cached_tokens(usage.cached_read_tokens as usize)
            .with_context_tokens(self.context_size);
        info.reasoning_tokens = Some(usage.thought_tokens as usize);
        let config = &self.task.agent_config.llm_config;
        let model = &config.model;
        info.calculate_cost(model, Some(&config.generation_config));
        let info = info.with_billing(ai::BillingKind::Subscription);
        let mut call = ai::ModelCallUsage::new(
            Uuid::new_v4().to_string(),
            model.provider.as_telemetry_str().into(),
            model.id.clone(),
            input,
            output,
            limit.unwrap_or(model.max_input_tokens),
        )
        .with_plan_usage(self.plan_usage.clone())
        .with_context_tokens(self.context_size);
        call.stop_reason = Some(stop.to_string());
        let anchor = self.hand_offs.freeze().unwrap_or_else(Stamp::now);
        self.task.latest_request = crate::task::RequestRecord::Latest(
            crate::task::LatestRequest::new(anchor.at, &info).with_sent_instant(anchor.instant),
        );
        self.publish_prompt_cache(Some(anchor.instant));
        self.task.record_usage(info.clone());
        let call = self.task.record_model_call_usage(call);
        self.emit(AgentEvent::UsageReported(info)).await?;
        self.emit(AgentEvent::ModelCallUsageReported(call)).await?;
        self.monitor_cache().await
    }
}

/// How a new session's first prompt starts, before the record.
pub(super) const SEED_INTRO: &str = "This conversation goes on in a new session. The record below says what happened so far. Its tool calls already ran, and each result follows its call, so don't run them again to check. Work on the user's new message at the end.";

/// The heading of the record of the conversation so far.
pub(super) const SEED_RECORD: &str = "# The conversation so far";

/// The heading of what the new session is to work on.
pub(super) const SEED_NEW: &str = "# The user's new message";

/// The most of a tool's output the record keeps, in characters. The rest
/// of a longer one is left out from its middle.
pub(super) const SEED_RESULT_CHARS: usize = 2_000;

/// A new session's first prompt for `messages`, where the last session had
/// the first `synced`. The record holds the messages up to the later of
/// `synced` and the model's last message, with each call's result under it
/// and long outputs cut. The new input (the user's messages after them,
/// with their images) comes last under a heading of its own. `None` when
/// the record would be empty.
pub(super) fn seed(messages: &[ai::Message], synced: usize) -> Option<Vec<ai::Content>> {
    let after_reply = messages
        .iter()
        .rposition(|message| message.role == ai::Role::Assistant)
        .map_or(0, |last| last + 1);
    let split = synced.min(messages.len()).max(after_reply);
    if split == 0 {
        return None;
    }
    let (record, new) = messages.split_at(split);
    let results: HashMap<&str, &ai::ToolResult> = record
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(ai::Content::as_tool_result)
        .filter_map(|result| Some((result.call_id.as_deref()?, result)))
        .collect();
    let mut text = vec![SEED_INTRO.to_string(), SEED_RECORD.to_string()];
    for message in record {
        match message.role {
            ai::Role::Assistant => {
                let mut said = vec!["## You".to_string()];
                said.extend(texts(message));
                for call in &message.tool_calls {
                    let result = call.id.as_deref().and_then(|id| results.get(id));
                    said.push(called(call, result.copied()));
                }
                text.push(said.join("\n\n"));
            }
            _ => {
                let said = texts(message);
                let images = message
                    .content
                    .iter()
                    .filter(|content| content.as_image().is_some())
                    .count();
                if said.is_empty() && images == 0 {
                    // Only tool results, which follow their calls.
                    continue;
                }
                let mut block = vec![if said
                    .first()
                    .is_some_and(|first| first.starts_with(CONTEXT_UPDATE))
                {
                    "## Context update".to_string()
                } else {
                    "## The user".to_string()
                }];
                block.extend(said.iter().map(|said| {
                    said.strip_prefix(CONTEXT_UPDATE)
                        .map_or(said.as_str(), str::trim_start)
                        .to_string()
                }));
                if images > 0 {
                    block.push(left_out_images(images));
                }
                text.push(block.join("\n\n"));
            }
        }
    }
    let mut content = vec![ai::Content::Text(text.join("\n\n"))];
    content.push(ai::Content::Text(SEED_NEW.to_string()));
    let mut new_input = false;
    for message in new.iter().filter(|message| message.role == ai::Role::User) {
        for item in &message.content {
            match item {
                ai::Content::Text(said) if !said.trim().is_empty() => {
                    content.push(item.clone());
                    new_input |= !said.trim().is_empty();
                }
                ai::Content::Image(_) => content.push(item.clone()),
                _ => {}
            }
        }
    }
    if !new_input {
        content.push(ai::Content::Text(CONTINUE.to_string()));
    }
    Some(content)
}

/// The text a message holds, each part with words in it.
fn texts(message: &ai::Message) -> Vec<String> {
    message
        .content
        .iter()
        .filter_map(ai::Content::as_text)
        .map(str::trim)
        .filter(|said| !said.is_empty())
        .map(str::to_string)
        .collect()
}

/// A call as the record has it, with its result under it.
fn called(call: &ai::ToolCall, result: Option<&ai::ToolResult>) -> String {
    let mut said = format!("Called `{}` with {}", call.name, call.arguments);
    match result {
        Some(result) => {
            said.push_str(if result.is_error {
                "\nIt failed.\n"
            } else {
                "\nIt returned:\n"
            });
            said.push_str(&cut_middle(&result.output, SEED_RESULT_CHARS));
            if !result.images.is_empty() {
                said.push('\n');
                said.push_str(&left_out_images(result.images.len()));
            }
        }
        None => said.push_str("\nIts result isn't in the record."),
    }
    said
}

fn left_out_images(count: usize) -> String {
    match count {
        1 => "(An image here is left out of the record.)".to_string(),
        count => format!("({count} images here are left out of the record.)"),
    }
}

/// `text` with no more than `most` characters, its middle left out and said
/// so.
fn cut_middle(text: &str, most: usize) -> String {
    let count = text.chars().count();
    if count <= most {
        return text.to_string();
    }
    let head: String = text.chars().take(most * 3 / 4).collect();
    let tail: String = text.chars().skip(count - most / 4).collect();
    let left_out = count - head.chars().count() - tail.chars().count();
    format!("{head}\n[… {left_out} characters left out …]\n{tail}")
}
