use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

pub mod background;
pub mod builtins;
pub mod command_builder;
pub mod definition;
pub mod error;
pub mod interceptor;
pub mod middleware;
pub mod registry;
pub mod types;

pub use background::{ContextSignal, ContextSource, MoveSignal, MoveSource, ToolBackground};
pub use definition::{ToolDefinition, ToolParameterDefinition, ToolParameterType};
pub use error::ToolError;
pub use interceptor::{
    InterceptedResult, InterceptionFailurePolicy, InterceptorConfig, ToolResultInterceptor,
};
pub use middleware::{
    ToolCallContext, ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware,
    ToolExecutionOutcome,
};
pub use registry::ToolRegistry;
pub use types::ToolTypeDetails;

/// A real image input returned by a tool, with a human-readable source label.
/// Image bytes belong in saved task state, never in text tool-result events.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolImage {
    pub description: String,
    pub image: arko_ai::Image,
}

impl Debug for ToolImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.image {
            arko_ai::Image::Base64 { mime_type, .. } => mime_type.as_str(),
            arko_ai::Image::Url(_) => "URL",
        };
        f.debug_struct("ToolImage")
            .field("description", &self.description)
            .field("kind", &kind)
            .finish_non_exhaustive()
    }
}

/// The largest image file `read_file` shows, equal to 5 MiB. (The images a tool
/// returns are fitted to the model's limits when the runtime admits them,
/// [`ToolOutput::admit_images`].)
pub const MAX_TOOL_IMAGE_BYTES: usize = 5 * 1024 * 1024;

impl ToolImage {
    /// A base64 image with its label.
    pub fn base64(description: impl Into<String>, data: String, mime_type: &str) -> Self {
        Self {
            description: description.into(),
            image: arko_ai::Image::Base64 {
                data: data.into(),
                mime_type: mime_type.to_string(),
            },
        }
    }

    /// Its size in bytes once decoded, or `None` for an image given by URL.
    pub fn decoded_len(&self) -> Option<usize> {
        match &self.image {
            arko_ai::Image::Base64 { data, .. } => {
                let data = data.trim_end_matches('=');
                Some(data.len() / 4 * 3 + (data.len() % 4).saturating_sub(1))
            }
            arko_ai::Image::Url(_) => None,
        }
    }
}

/// Receives the images tool calls return, for a host that shows them
/// (thumbnails under a call, an image viewer). Set with
/// [`AgentRuntime::with_tool_image_sink`](crate::runtime::AgentRuntime::with_tool_image_sink).
///
/// Called once for each call that returned images, as the call ends, with
/// the runtime's ID of the call (the `tool_call_id` of its events), the
/// tool's name and the images as the tool returned them, before those the
/// model can't take are left out of its result. Covers every arko tool
/// (`read_file`, MCP tools, a host's own), and for a Claude Code agent
/// also Claude Code's built-in tools that return images (`Read`). A call
/// whose result is recovered after its task stopped is not reported again.
/// It runs on the call's task, so keep it quick (hand the images to another
/// thread to scale or store them).
///
/// The runtime calls
/// [`tool_images_with_call_id`](Self::tool_images_with_call_id), which also
/// gives the ID the model's provider gave the call. A sink that needs only
/// the runtime's ID implements [`tool_images`](Self::tool_images).
///
/// Events carry no image bytes because every subscriber of a workflow's
/// event broadcast, and its event log, would get a copy.
pub trait ToolImageSink: Send + Sync {
    fn tool_images(
        &self,
        task_id: uuid::Uuid,
        tool_call_id: uuid::Uuid,
        tool_name: &str,
        images: &[ToolImage],
    );

    /// [`tool_images`](Self::tool_images) with `provider_call_id`, the ID
    /// the model's provider gave the call (the `id` of the conversation's
    /// [`ToolCall`](arko_ai::ToolCall), a Claude Code `tool_use_id`). It is
    /// `None` for a call the model wrote as text, which has no provider ID.
    /// A host that matches a call with the conversation's messages overrides
    /// this method. The default drops the ID and calls `tool_images`.
    fn tool_images_with_call_id(
        &self,
        task_id: uuid::Uuid,
        tool_call_id: uuid::Uuid,
        provider_call_id: Option<&str>,
        tool_name: &str,
        images: &[ToolImage],
    ) {
        let _ = provider_call_id;
        self.tool_images(task_id, tool_call_id, tool_name, images);
    }
}

/// Structured output is opt-in. Existing string-returning tools need no changes.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub text: String,
    pub images: Vec<ToolImage>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    /// Fits the images to what `model` can be sent
    /// ([`arko_ai::Model::sendable_image`]). One over the model's limits is
    /// shrunk, and one the model can't be sent (every image for a model
    /// without image input, a URL where images go as data, GIF on Google,
    /// one that can't be read or shrunk enough) is left out, with a line
    /// saying why after the text.
    pub fn admit_images(&mut self, model: &arko_ai::Model) {
        self.admit_images_within(model, &ImageRoom::unlimited());
    }

    /// [`admit_images`](Self::admit_images), admitting only the images
    /// `room` has room for (each one admitted takes its place and its bytes
    /// from it). A request holds at most
    /// [`ImageLimits::max_images`](arko_ai::image_fit::ImageLimits::max_images)
    /// images and
    /// [`max_request_image_bytes`](arko_ai::image_fit::ImageLimits::max_request_image_bytes)
    /// of them, so the images past it are left out too, with a line saying
    /// so.
    pub fn admit_images_within(&mut self, model: &arko_ai::Model, room: &ImageRoom) {
        let mut notes = Vec::new();
        let images = std::mem::take(&mut self.images);
        for mut image in images {
            let label = if image.description.is_empty() {
                "an image".to_string()
            } else {
                format!("image {}", image.description)
            };
            match model.sendable_image(&image.image) {
                Ok(fitted) => {
                    if let Some(fitted) = fitted {
                        image.image = fitted;
                    }
                    let bytes = match &image.image {
                        arko_ai::Image::Base64 { data, .. } => data.len(),
                        arko_ai::Image::Url(_) => 0,
                    };
                    if let Err(full) = room.take(bytes) {
                        let limits = model.image_limits();
                        notes.push(match full {
                            Full::Images => format!(
                                "[The tool returned {label}, which was left out because the conversation holds as many images as a request to this model takes ({}). A handover or a cut back makes room.]",
                                limits.max_images
                            ),
                            Full::Bytes => format!(
                                "[The tool returned {label}, which was left out because the conversation holds as much image data as a request to this model takes ({} MB). A handover or a cut back makes room.]",
                                limits.max_request_image_bytes / (1024 * 1024)
                            ),
                        });
                        continue;
                    }
                    self.images.push(image);
                }
                Err(reason) => {
                    notes.push(format!("[The tool returned {label}, left out: {reason}.]"))
                }
            }
        }
        for note in notes {
            if !self.text.is_empty() {
                self.text.push_str("\n\n");
            }
            self.text.push_str(&note);
        }
    }
}

/// How many more images a conversation can take in a request,
/// and how many more bytes of base64 data (see
/// [`ToolOutput::admit_images_within`]). Shared by the calls of a turn, so
/// their images together stay within it.
#[derive(Debug)]
pub struct ImageRoom {
    images: std::sync::atomic::AtomicUsize,
    bytes: std::sync::atomic::AtomicUsize,
}

/// What an [`ImageRoom`] has no more room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Full {
    Images,
    Bytes,
}

impl ImageRoom {
    /// Room for `images` more images, of `bytes` more bytes in all.
    pub fn new(images: usize, bytes: usize) -> Self {
        Self {
            images: images.into(),
            bytes: bytes.into(),
        }
    }

    /// Room for any number of images.
    pub fn unlimited() -> Self {
        Self::new(usize::MAX, usize::MAX)
    }

    /// The room left for `model` in a conversation holding `messages`.
    pub fn left(model: &arko_ai::Model, messages: &[arko_ai::Message]) -> Self {
        let limits = model.image_limits();
        let (mut images, mut bytes) = (0, 0);
        for message in messages {
            for content in &message.content {
                let held: Vec<&arko_ai::Image> = match content {
                    arko_ai::Content::Image(image) => vec![image],
                    arko_ai::Content::ToolResult(result) => {
                        result.images.iter().map(|image| &image.image).collect()
                    }
                    arko_ai::Content::Text(_) => Vec::new(),
                };
                for image in held {
                    images += 1;
                    if let arko_ai::Image::Base64 { data, .. } = image {
                        bytes += data.len();
                    }
                }
            }
        }
        Self::new(
            limits.max_images.saturating_sub(images),
            limits.max_request_image_bytes.saturating_sub(bytes),
        )
    }

    /// Takes room for one image of `bytes` bytes, or says what is full.
    // Newer Rust renames `fetch_update` to `try_update`, which Rust 1.89 doesn't have.
    #[allow(deprecated)]
    pub fn take(&self, bytes: usize) -> Result<(), Full> {
        use std::sync::atomic::Ordering::SeqCst;
        self.images
            .fetch_update(SeqCst, SeqCst, |left| left.checked_sub(1))
            .map_err(|_| Full::Images)?;
        if self
            .bytes
            .fetch_update(SeqCst, SeqCst, |left| left.checked_sub(bytes))
            .is_err()
        {
            self.images.fetch_add(1, SeqCst);
            return Err(Full::Bytes);
        }
        Ok(())
    }
}

/// Context provided to a tool during its execution.
///
/// This allows tools to access relevant information from the current task's environment,
/// such as the current working directory.
///
/// A host that builds one itself (to call a tool outside a runtime, or in a
/// test) uses [`TaskContext::new`], which leaves the call fields unset and
/// the [stop](Self::stop), [move](Self::moved) and
/// [context](Self::context_arrived) signals ones that never fire. The
/// fields stay public to set afterwards. The struct is `#[non_exhaustive]`,
/// so fields can be added without breaking callers. Outside this crate it
/// can't be built with a struct literal.
#[derive(Debug)]
#[non_exhaustive]
pub struct TaskContext<'a> {
    pub task_id: uuid::Uuid,
    pub agent_name: &'a str,
    pub initial_task_prompt: &'a str,
    pub current_working_directory: &'a Path,
    /// The runtime's id of the call being executed, as its events and the
    /// execution middleware see it (`None` outside a call, such as for a
    /// result interceptor run by a host). A call recovered after its host
    /// stopped keeps its id. See [`Tool::recovery_mode`].
    pub tool_call_id: Option<uuid::Uuid>,
    /// The model's id of the call, as the task's conversation records it
    /// (the provider's call id, or Claude Code's `tool_use_id`), when it has
    /// one. A host that keeps something about a call (an answer it asked
    /// the user for) finds the call in the conversation by it later.
    pub call_id: Option<String>,
    /// The call is being run again because the task stopped while it ran
    /// (its host crashed, say). Its tool declared
    /// [`ToolRecovery::Rerun`] or [`ToolRecovery::Resume`]. A
    /// [`Resume`](ToolRecovery::Resume) tool finds the work the first run
    /// started, by [`tool_call_id`](Self::tool_call_id), and continues it.
    pub recovering: bool,
    /// Fires when the task is being stopped while the call runs because the
    /// runtime received `CancelTask` or `SuspendTask`, or the host fired the
    /// signal it gave the runtime (a workflow's graceful stop). A long call
    /// can watch it to save its work and return early. See
    /// [`StopSignal`](crate::stop::StopSignal) and
    /// [`StopAwareCommand`](crate::stop::StopAwareCommand) for a child
    /// process. An [interruptible](Tool::interruptible) call is dropped as
    /// soon as `CancelTask` or `SuspendTask` arrives whatever it does, so
    /// it sees the signal only for a host's graceful stop.
    pub stop: crate::stop::StopSignal,
    /// Fires when the host moves this call to the background. Only the calls
    /// of a tool whose [`background_mode`](Tool::background_mode) is
    /// [`Own`](ToolBackground::Own) see it fire. Such a tool returns soon
    /// after, with a result that says what happened (the job its work goes
    /// on in, a wait that ended).
    pub moved: MoveSignal,
    /// Fires when context (a note, a terminal update) arrives for the task
    /// while the call runs. A tool that waits can return early, so the model
    /// reads it. Context that arrived after the request the call came from
    /// (while an earlier call of the turn ran, say) fires it at once.
    pub context_arrived: ContextSignal,
    /// The task's model runs in Claude Code
    /// ([`Provider::ClaudeCode`](arko_ai::Provider::ClaudeCode)), which
    /// shows the model only a short preview of a very long tool result. A
    /// tool with long results can keep them short for it.
    pub runs_in_claude_code: bool,
}

impl<'a> TaskContext<'a> {
    /// The context of a task outside a call, with no call ID, not recovering, and
    /// stop, move and context signals that never fire. Set the public fields
    /// to change them, or [`with_stop`](Self::with_stop).
    pub fn new(
        task_id: uuid::Uuid,
        agent_name: &'a str,
        initial_task_prompt: &'a str,
        current_working_directory: &'a Path,
    ) -> Self {
        Self {
            task_id,
            agent_name,
            initial_task_prompt,
            current_working_directory,
            tool_call_id: None,
            call_id: None,
            recovering: false,
            stop: crate::stop::StopSignal::never(),
            moved: MoveSignal::never(),
            context_arrived: ContextSignal::never(),
            runs_in_claude_code: false,
        }
    }

    /// This context with `stop` as its [stop signal](Self::stop).
    pub fn with_stop(mut self, stop: crate::stop::StopSignal) -> Self {
        self.stop = stop;
        self
    }
}

/// How a tool call that was running when its task stopped (its host
/// crashed, say) is settled when the task runs again. See
/// [`Tool::recovery_mode`] and
/// [`Task::running_tool_calls`](crate::task::Task::running_tool_calls).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRecovery {
    /// The call is not run again. Its result tells the model the task
    /// restarted while the call ran, that it may have run partly or fully,
    /// and to check its effects before calling it again. For a tool with
    /// effects outside the task (writing files, running commands).
    #[default]
    InDoubt,
    /// The call runs again with its original arguments. For a tool without
    /// side effects, which returns a fresh result (reading a file, listing a
    /// directory, loading a skill).
    Rerun,
    /// The call runs again with
    /// [`TaskContext::recovering`] set and its original
    /// [`TaskContext::tool_call_id`]. The tool reattaches to the work its
    /// first run started (a workflow's `run_workflow_agent` recovers the
    /// agent run it started) and returns its result, or else runs the call.
    Resume,
}

/// A trait for tools that an agent can use.
///
/// Tools must be `Send + Sync` as they can be executed in concurrent environments.
#[async_trait]
pub trait Tool: Debug + Send + Sync {
    /// Returns the definition of the tool.
    fn definition(&self) -> ToolDefinition;

    /// Stateful, exact-result protocols can opt out of result transformation.
    fn allow_result_interception(&self) -> bool {
        true
    }

    /// Whether the agent's host-side time limit applies to this tool's calls
    /// ([`AgentToolConfig::tool_timeout_seconds`](crate::agent_config::AgentToolConfig::tool_timeout_seconds)).
    /// Tools with their own wait and job handling return `false`, as the
    /// terminal tools do. A per-tool setting applies either way.
    fn uses_host_timeout(&self) -> bool {
        true
    }

    /// Whether `CancelTask` and `SuspendTask` stop a running call of this
    /// tool. When true, the runtime drops the call (killing a command it
    /// started), and its result says it was stopped. Tools that wind their own work down
    /// when the task stops return `false`, as a workflow's
    /// `run_workflow_agent` does (it stops the agent it runs within the
    /// workflow's grace period). The stop then takes effect once the call
    /// returns.
    fn interruptible(&self) -> bool {
        true
    }

    /// How a call of this tool that was running when its task stopped (its
    /// host crashed while the call ran) is settled when the task runs
    /// again. Defaults to [`ToolRecovery::InDoubt`], answered without
    /// running it again. Read-only tools return [`ToolRecovery::Rerun`].
    fn recovery_mode(&self) -> ToolRecovery {
        ToolRecovery::InDoubt
    }

    /// [`recovery_mode`](Self::recovery_mode) for one call, given its
    /// arguments, for a tool whose calls differ (one that reads or writes by
    /// its arguments). The runtime asks this, and the default gives
    /// `recovery_mode`.
    fn recovery_mode_for(&self, params: &HashMap<String, Value>) -> ToolRecovery {
        let _ = params;
        self.recovery_mode()
    }

    /// Whether this call may run beside other parallel-safe calls of a
    /// response, given its arguments, for a tool whose calls differ. `None`
    /// leaves it to the definition's
    /// [`parallel_safe`](ToolDefinition::parallel_safe), as the default
    /// does. An agent's per-tool
    /// [`parallel_safe`](crate::agent_config::PerToolConfigOverride::parallel_safe)
    /// still overrides both. Native calls go by it, and a batch of
    /// output-based calls by the definition. A Claude Code agent's calls
    /// run as Claude Code schedules them.
    fn parallel_safe_for(&self, params: &HashMap<String, Value>) -> Option<bool> {
        let _ = params;
        None
    }

    /// How a running call of this tool can move to the background, when its
    /// runtime has a terminal manager for background jobs
    /// ([`AgentRuntime::with_background_jobs`](crate::runtime::AgentRuntime::with_background_jobs)).
    /// Defaults to [`ToolBackground::Never`].
    /// [`Detach`](ToolBackground::Detach) suits a call that just awaits its
    /// result, such as an MCP request or a command. The runtime moves its
    /// future to a job, which goes on after the turn and sends the result as
    /// a note. Its host time limit still counts from the call's start, and
    /// its [`TaskContext::stop`] then fires when the job is stopped instead
    /// of when the turn is. [`Own`](ToolBackground::Own) suits a tool that
    /// moves its work itself, or a wait that can simply end. It watches
    /// [`TaskContext::moved`].
    fn background_mode(&self) -> ToolBackground {
        ToolBackground::Never
    }

    /// Discard delivery baselines when a task starts or resumes in a runtime.
    fn reset_context(&self, _task_id: uuid::Uuid) {}

    /// Observe messages committed to the conversation before its next model call.
    /// This is synchronous bookkeeping, not a place for I/O or model calls.
    fn observe_context(&self, _task_id: uuid::Uuid, _message: &arko_ai::Message) {}

    /// Bind background tools to this task's context channel. A weak sender must
    /// not keep a stopped runtime alive. Hosts can also bind custom runners.
    fn bind_runtime(
        &self,
        _task_id: uuid::Uuid,
        _sender: tokio::sync::mpsc::WeakSender<crate::runtime::command::UserCommand>,
    ) {
    }

    /// Executes the tool with the given parameters and task context.
    ///
    /// # Arguments
    /// * `context` gives information about the current task, like CWD.
    /// * `params` maps parameter names to their `serde_json::Value` representations.
    ///
    /// # Returns
    /// A `Result` containing a `String` output of the tool on success, or a `ToolError` on failure.
    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError>;

    /// Execute with optional image attachments. The runtime preserves images
    /// independently of text interception and sends them as image content blocks.
    async fn execute_with_output(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<ToolOutput, ToolError> {
        self.execute(context, params).await.map(ToolOutput::text)
    }

    /// Helper method to get the tool's name from its definition.
    fn name(&self) -> String {
        self.definition().name
    }
}

/// A tool defined by an end user via a command template.
#[derive(Debug, Clone)]
pub(crate) struct EndUserTool {
    definition: ToolDefinition,
}

impl EndUserTool {
    /// Creates a new EndUserTool from a tool definition.
    pub fn new(definition: ToolDefinition) -> Result<Self, ToolError> {
        if !matches!(
            definition.tool_type_details,
            ToolTypeDetails::EndUserDefined { .. }
        ) {
            return Err(ToolError::Configuration {
                tool_name: definition.name.clone(),
                message: "ToolDefinition is not for an EndUserDefined tool.".to_string(),
            });
        }
        Ok(Self { definition })
    }
}

#[async_trait]
impl Tool for EndUserTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// A command, which goes on as a background job when the call moves.
    fn background_mode(&self) -> ToolBackground {
        ToolBackground::Detach
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let command_template_str = match &self.definition.tool_type_details {
            ToolTypeDetails::EndUserDefined { command_template } => command_template,
            _ => return Err(ToolError::ExecutionFailed(
                "EndUserTool executed with non-EndUserDefined details. This is an internal error."
                    .to_string(),
            )),
        };

        command_builder::execute_command_from_template(
            &self.definition.name,
            command_template_str,
            &self.definition.parameters,
            &params,
            context.current_working_directory,
        )
        .await
    }
}
