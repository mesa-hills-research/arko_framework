use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use self::command::UserCommand;
use self::event::AgentEvent;
use self::r#loop::AgentLoop;
pub use self::result_budget::{DEFAULT_BYTES_PER_TOKEN, calibrated_bytes_per_token};
pub use self::turn_context::{ContextImage, TurnContext, TurnContextSource};
use crate::error::ArkoAgentsError;
use crate::handover::{HandoverHook, HandoverThreshold};
use crate::skill::SkillPromptContext;
use crate::snapshot::{TaskSnapshotSink, TaskSnapshots};
use crate::stop::StopSignal;
use crate::task::Task;
use crate::tool::{InterceptorConfig, ToolExecutionMiddleware, ToolRegistry};
use arko_ai::LLMClient;

mod claude_code;
pub mod command;
pub mod event;
mod r#loop;
mod monitor;
mod result_budget;
mod retry;
mod tool_run;
pub mod turn_context;

#[cfg(test)]
mod keep_warm_live_tests;

const DEFAULT_EVENT_CHANNEL_SIZE: usize = 100;

/// What a prompt-cache refresh asks after the conversation: a one-word
/// reply (see [`AgentRuntime::refresh_prompt_cache`]).
pub(crate) const CACHE_PING: &str =
    "Reply with the single word ok. This message only checks the connection.";

/// What a prompt-cache refresh sent and cost. See
/// [`AgentRuntime::refresh_prompt_cache`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CacheRefresh {
    /// The refresh request's usage. Its cache reads show what it kept.
    pub usage: arko_ai::UsageInfo,
    /// The output limit the request was sent with, or `None` where the
    /// provider takes none (ChatGPT OAuth) or Claude Code chooses it.
    pub output_cap: Option<usize>,
    /// The task's last request went again unchanged, instead of a ping
    /// after its conversation, because the task waits with calls of the model's
    /// that have no results yet (paused for an approval).
    pub resent_last_request: bool,
    /// Why the provider refused the smaller output limit tried first, when
    /// the request went again with `output_cap`.
    pub refused_cap: Option<String>,
}

impl CacheRefresh {
    /// A refresh that cost `usage`.
    pub fn new(usage: arko_ai::UsageInfo) -> Self {
        Self {
            usage,
            output_cap: None,
            resent_last_request: false,
            refused_cap: None,
        }
    }
}
const DEFAULT_COMMAND_CHANNEL_SIZE: usize = 10;

/// Whether the host decides every wait of a run itself. Every tool
/// approval, answer and error-recovery decision the task waits for is
/// decided by the host (or the host stops the run), however long it takes.
/// While it is set, the task's
/// [`user_command_timeout_seconds`](crate::agent_config::AgentConfig::user_command_timeout_seconds)
/// doesn't apply, so no wait of the run is timed. A workflow sets it for the
/// runs it drives.
///
/// This run-level setting is not part of the task and is not saved, so a
/// task run later without it is timed as configured. Clones share the
/// setting, and the host may change it while the run runs. Turned on during
/// a wait, that wait is no longer timed. Turned off, the waits that start
/// afterwards are.
#[derive(Clone, Debug, Default)]
pub struct HostDecisions(Arc<AtomicBool>);

impl HostDecisions {
    /// With `true`, the host decides every wait of the run.
    pub fn new(host_decides: bool) -> Self {
        Self(Arc::new(AtomicBool::new(host_decides)))
    }

    /// Changes the setting for every clone.
    pub fn set(&self, host_decides: bool) {
        self.0.store(host_decides, Ordering::Release);
    }

    /// Whether the host decides every wait of the run.
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The main runtime for executing agent tasks.
pub struct AgentRuntime {
    llm_client: Arc<LLMClient>,
    turn_context: Option<Arc<dyn TurnContextSource>>,
    host_decisions: HostDecisions,
    handover_hook: Option<Arc<dyn HandoverHook>>,
    handover_threshold: HandoverThreshold,
    task_snapshots: Option<Arc<dyn TaskSnapshotSink>>,
    stop_signal: StopSignal,
    image_sink: Option<Arc<dyn crate::tool::ToolImageSink>>,
    #[cfg(any(unix, windows))]
    background_jobs: Option<Arc<crate::terminal::TerminalManager>>,
}

impl AgentRuntime {
    /// Creates a new AgentRuntime with the given LLM client.
    pub fn new(llm_client: Arc<LLMClient>) -> Self {
        Self {
            llm_client,
            turn_context: None,
            host_decisions: HostDecisions::default(),
            handover_hook: None,
            handover_threshold: HandoverThreshold::default(),
            task_snapshots: None,
            stop_signal: StopSignal::never(),
            image_sink: None,
            #[cfg(any(unix, windows))]
            background_jobs: None,
        }
    }

    /// Gives `sink` the images the tool calls of the tasks this runtime runs
    /// return, keyed by each call's ID. See
    /// [`ToolImageSink`](crate::tool::ToolImageSink).
    pub fn with_tool_image_sink(mut self, sink: Arc<dyn crate::tool::ToolImageSink>) -> Self {
        self.image_sink = Some(sink);
        self
    }

    /// Lets the tool calls of the tasks this runtime runs move to the
    /// background as jobs of `jobs`. While a call of a tool whose
    /// [`background_mode`](crate::tool::Tool::background_mode) isn't `Never`
    /// runs, `jobs` lists it
    /// ([`foreground_calls`](crate::terminal::TerminalManager::foreground_calls)),
    /// and the host moves it with
    /// [`move_to_background`](crate::terminal::TerminalManager::move_to_background).
    /// A [`Detach`](crate::tool::ToolBackground::Detach) call then goes on as
    /// a job, the model gets the job's ID as the call's result, and the
    /// result reaches the task as a note when it ends. An
    /// [`Own`](crate::tool::ToolBackground::Own) tool sees
    /// [`TaskContext::moved`](crate::tool::TaskContext::moved) fire. The
    /// tasks are bound to `jobs` for the notes. A Claude Code agent's arko
    /// tool calls move the same way.
    #[cfg(any(unix, windows))]
    pub fn with_background_jobs(mut self, jobs: Arc<crate::terminal::TerminalManager>) -> Self {
        self.background_jobs = Some(jobs);
        self
    }

    /// Runs tasks with `decisions`. While it is set, the host decides every
    /// wait of the runs itself, and their command timeout doesn't apply.
    /// See [`HostDecisions`].
    pub fn with_host_decisions(mut self, decisions: HostDecisions) -> Self {
        self.host_decisions = decisions;
        self
    }

    /// Asks `source` for context before each model request of the tasks this
    /// runtime runs. See [`TurnContextSource`].
    pub fn with_turn_context(mut self, source: Arc<dyn TurnContextSource>) -> Self {
        self.turn_context = Some(source);
        self
    }

    /// Calls `hook` when a task this runtime runs hands its conversation
    /// over, or Claude Code compacts it. See [`HandoverHook`].
    pub fn with_handover_hook(mut self, hook: Arc<dyn HandoverHook>) -> Self {
        self.handover_hook = Some(hook);
        self
    }

    /// Hands the tasks this runtime runs over at `threshold`, which the
    /// host may change while they run, in place of their configuration's
    /// threshold. See [`HandoverThreshold`].
    pub fn with_handover_threshold(mut self, threshold: HandoverThreshold) -> Self {
        self.handover_threshold = threshold;
        self
    }

    /// Reports the tasks this runtime runs to `sink` at their safe points,
    /// so that the host can save a task while it runs and continue it after
    /// a crash. See [`TaskSnapshotSink`]. Before tool calls start, the run
    /// waits until the sink has saved the task with the calls recorded as
    /// running ([`Task::running_tool_calls`]).
    pub fn with_task_snapshots(mut self, sink: Arc<dyn TaskSnapshotSink>) -> Self {
        self.task_snapshots = Some(sink);
        self
    }

    /// Tells the tool calls of the tasks this runtime runs when the host is
    /// stopping them. The `signal` reaches each call as part of
    /// [`TaskContext::stop`](crate::tool::TaskContext::stop), which also
    /// fires when the runtime receives `CancelTask` or `SuspendTask` while
    /// the call runs. Firing it changes nothing else. The task goes on until
    /// the host cancels or suspends it (a workflow's graceful stop fires it
    /// at the start of its wrap-up, and cancels its agents afterwards).
    pub fn with_stop_signal(mut self, signal: StopSignal) -> Self {
        self.stop_signal = signal;
        self
    }

    /// Keeps `task`'s prompt cache warm while it waits between turns. Sends its
    /// next request (system prompt, tools, conversation) with a short ping
    /// after it and caps the output where the provider allows (one token on
    /// Anthropic, OpenAI's minimum of 16, the ChatGPT backend takes no cap).
    /// Discards the reply, so the provider reads the cached prefix and
    /// its lifetime starts over. A task waiting with calls of the model's
    /// that have no results yet (paused for an approval) sends its last
    /// request again unchanged instead, which ends where the cache entry the
    /// next request reads ends. A provider that refuses the cap gets the
    /// request once more with 1,024 (see [`CacheRefresh::refused_cap`]). A
    /// Claude Code task pings a fork of its session that isn't saved.
    /// Nothing is added to the task. Unsupported
    /// (`arko_ai::Error::UnsupportedFeature`) for Gemini, OpenRouter models
    /// other than Anthropic's and Anthropic's with reasoning on (OpenRouter
    /// derives the thinking budget from the cap, so a capped request would
    /// miss the cached conversation), a manual Anthropic thinking budget,
    /// and a task that has made no request with its current prompt.
    /// `tool_registry` and `skill_prompt_context` are the task's own, as for
    /// [`run_task`](Self::run_task), so the request offers the same tools.
    /// The host refreshes a task its runtime isn't running.
    pub async fn refresh_prompt_cache(
        &self,
        task: Task,
        tool_registry: Arc<ToolRegistry>,
        skill_prompt_context: Option<SkillPromptContext>,
    ) -> Result<CacheRefresh, ArkoAgentsError> {
        let (event_sender, _events) = mpsc::channel(DEFAULT_EVENT_CHANNEL_SIZE);
        let (_commands, command_receiver) = mpsc::channel(DEFAULT_COMMAND_CHANNEL_SIZE);
        let refresh = if task.agent_config.runs_in_claude_code() {
            let usage = claude_code::ClaudeCodeRun::new(
                task,
                tool_registry,
                event_sender,
                command_receiver,
                skill_prompt_context,
                None,
                None,
            )
            .refresh_prompt_cache()
            .await?;
            CacheRefresh::new(usage)
        } else {
            AgentLoop::new(
                task,
                Arc::clone(&self.llm_client),
                tool_registry,
                event_sender,
                command_receiver,
                None,
                true,
                skill_prompt_context,
                None,
                None,
            )
            .refresh_prompt_cache()
            .await?
        };
        Ok(refresh)
    }

    /// Runs a task with the given agent configuration and tool registry.
    ///
    /// The task runs only the registry's tools its configuration offers
    /// ([`AgentToolConfig::offered_tool_names`](crate::agent_config::AgentToolConfig::offered_tool_names)).
    /// Those tools are bound to the task's command channel
    /// ([`Tool::bind_runtime`](crate::tool::Tool::bind_runtime)).
    ///
    /// An optional `InterceptorConfig` enables tool result interception,
    /// allowing results to be transformed (e.g., distilled) before they
    /// enter the agent's conversation. `complete_task` and `ask_user`
    /// results are never intercepted.
    ///
    /// A task that stopped while tool calls ran (saved through a
    /// [`TaskSnapshotSink`] by a host that crashed) settles those calls
    /// first, each by its tool's
    /// [`recovery_mode`](crate::tool::Tool::recovery_mode).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn run_task(
        &self,
        task: Task,
        tool_registry: Arc<ToolRegistry>,
        event_throttle_ms: Option<u64>,
        simplified_tool_events: bool,
        skill_prompt_context: Option<SkillPromptContext>,
        interceptor_config: Option<InterceptorConfig>,
        execution_middleware: Option<Arc<dyn ToolExecutionMiddleware>>,
    ) -> (
        mpsc::Receiver<AgentEvent>,
        mpsc::Sender<UserCommand>,
        JoinHandle<Result<Task, ArkoAgentsError>>,
    ) {
        let (event_sender, event_receiver) = mpsc::channel(DEFAULT_EVENT_CHANNEL_SIZE);
        let (command_sender, command_receiver) = mpsc::channel(DEFAULT_COMMAND_CHANNEL_SIZE);

        if task.agent_config.runs_in_claude_code() {
            let mut run = claude_code::ClaudeCodeRun::new(
                task,
                tool_registry,
                event_sender,
                command_receiver,
                skill_prompt_context,
                interceptor_config,
                execution_middleware,
            );
            run.turn_context = self.turn_context.clone();
            run.host_decisions = self.host_decisions.clone();
            run.handover_hook = self.handover_hook.clone();
            run.handover_threshold = self.handover_threshold.clone();
            run.snapshots = TaskSnapshots::new(self.task_snapshots.clone());
            run.host_stop = self.stop_signal.clone();
            run.image_sink = self.image_sink.clone();
            #[cfg(any(unix, windows))]
            {
                run.background_jobs = self.background_jobs.clone();
            }
            run.bind_tools(&command_sender);
            let join_handle = tokio::spawn(async move { run.run().await });
            return (event_receiver, command_sender, join_handle);
        }
        let mut agent_loop = AgentLoop::new(
            task,
            Arc::clone(&self.llm_client),
            tool_registry,
            event_sender,
            command_receiver,
            event_throttle_ms,
            simplified_tool_events,
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
        );
        agent_loop.turn_context = self.turn_context.clone();
        agent_loop.host_decisions = self.host_decisions.clone();
        agent_loop.handover_hook = self.handover_hook.clone();
        agent_loop.handover_threshold = self.handover_threshold.clone();
        agent_loop.snapshots = TaskSnapshots::new(self.task_snapshots.clone());
        agent_loop.host_stop = self.stop_signal.clone();
        agent_loop.image_sink = self.image_sink.clone();
        #[cfg(any(unix, windows))]
        {
            agent_loop.background_jobs = self.background_jobs.clone();
        }
        agent_loop.bind_tools(&command_sender);

        let task_join_handle = tokio::spawn(async move { agent_loop.run().await });

        (event_receiver, command_sender, task_join_handle)
    }
}
