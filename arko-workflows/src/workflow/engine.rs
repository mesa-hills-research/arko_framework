use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use arko_agents::skill::{SkillManager, SkillPromptContext};
use arko_agents::{
    AgentConfig, AgentRuntime, InterceptedResult, InterceptorConfig, Task, ToolCallContext,
    ToolError, ToolExecutionContext, ToolExecutionDecision, ToolExecutionMiddleware,
    ToolExecutionOutcome, ToolRegistry, ToolResultInterceptor,
    runtime::{command::UserCommand, event::AgentEvent},
    task::TaskState,
    tool::TaskContext,
};
use arko_ai::LLMClient;

use crate::context::SharedContext;
use crate::distillation::DistillationInterceptor;
use crate::error::WorkflowError;
use crate::freshness::ActorId;
use crate::hooks::context::HookContext;
use crate::hooks::{HookPoint, HookResult};
use crate::memory::tools::*;
use crate::messaging::tools::*;
use crate::persistence::PersistenceManager;
use crate::services::{RUN_WORKFLOW_AGENT_TOOL_NAME, RunWorkflowAgentTool};
use crate::verification::{
    CompletionVerifierConfig, VerificationDecision, VerificationKind, VerificationRecord,
};
use crate::voting::{CAST_VOTE_TOOL_NAME, CastVoteTool};
use crate::workflow::result::AgentResult;

use super::AgentHandle;

/// Chains multiple `ToolResultInterceptor` implementations, running them
/// in sequence. Each interceptor receives the output of the previous one.
struct ChainedInterceptor {
    interceptors: Vec<Arc<dyn ToolResultInterceptor>>,
    failure_policy: arko_agents::InterceptionFailurePolicy,
}

impl std::fmt::Debug for ChainedInterceptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainedInterceptor")
            .field("count", &self.interceptors.len())
            .finish()
    }
}

#[async_trait]
impl ToolResultInterceptor for ChainedInterceptor {
    async fn intercept(
        &self,
        tool_name: &str,
        raw_result: &str,
        params: &HashMap<String, Value>,
        task_context: &TaskContext<'_>,
    ) -> Result<InterceptedResult, ToolError> {
        let mut current = raw_result.to_string();
        let mut was_modified = false;

        for interceptor in &self.interceptors {
            let attempts = match self.failure_policy {
                arko_agents::InterceptionFailurePolicy::UseRawResult => 1,
                arko_agents::InterceptionFailurePolicy::RetryThenUseRawResult { max_attempts } => {
                    max_attempts.max(1)
                }
            };
            for attempt in 1..=attempts {
                match interceptor
                    .intercept(tool_name, &current, params, task_context)
                    .await
                {
                    Ok(InterceptedResult::Unchanged) => break,
                    Ok(InterceptedResult::Replaced(new_value)) => {
                        current = new_value;
                        was_modified = true;
                        break;
                    }
                    Err(error) if attempt < attempts => {
                        log::warn!(
                            "Tool result interceptor failed on attempt {attempt}/{attempts}: {error}"
                        );
                    }
                    Err(error) => {
                        log::warn!(
                            "Tool result interceptor failed; preserving earlier transformations: {error}"
                        );
                        break;
                    }
                }
            }
        }

        if was_modified {
            Ok(InterceptedResult::Replaced(current))
        } else {
            Ok(InterceptedResult::Unchanged)
        }
    }
}

/// Fire lifecycle hooks using only the shared context.
///
/// Iterates all hooks registered at the given point. If any hook returns
/// `Abort`, the workflow is stopped with a `WorkflowError::AbortedByHook`.
/// `Skip` is ignored where there is no operation to skip.
pub(crate) fn fire_hooks_on_context(
    ctx: &Arc<SharedContext>,
    point: HookPoint,
    configure: impl FnOnce(&mut HookContext),
) -> Result<(), WorkflowError> {
    let point_name = format!("{point:?}");
    match fire_hooks_returning_impl(ctx, point, configure)? {
        HookResult::Abort(reason) => Err(WorkflowError::AbortedByHook {
            hook_point: point_name,
            reason,
        }),
        _ => Ok(()),
    }
}

fn fire_hooks_returning_impl(
    ctx: &Arc<SharedContext>,
    point: HookPoint,
    configure: impl FnOnce(&mut HookContext),
) -> Result<HookResult, WorkflowError> {
    let Some(hooks) = ctx.hooks.get(&point) else {
        return Ok(HookResult::Continue);
    };

    let mut hctx = HookContext::base(
        ctx.workflow_run_id,
        &ctx.workflow_name,
        point,
        ctx.budget.total_spent(),
        ctx.budget.limit(),
    );
    configure(&mut hctx);

    for hook in hooks {
        let result = hook(&hctx);
        match result {
            HookResult::Continue => continue,
            other => return Ok(other),
        }
    }

    Ok(HookResult::Continue)
}

/// Drive the agent event loop with default workflow behavior.
///
/// Handles tool approval, budget tracking, and error recovery automatically.
/// Tool hooks and freshness policy are enforced by execution middleware, so
/// auto-approved and batched calls follow the same path.
///
/// This is the shared implementation used by both `WorkflowEngine::run_agent`
/// and `AgentHandle::wait`.
pub(crate) async fn drive_event_loop(
    agent_name: &str,
    agent_run_id: uuid::Uuid,
    event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>,
    cmd_tx: &tokio::sync::mpsc::Sender<UserCommand>,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<(), WorkflowError> {
    let mut approved_error_recoveries = 0_u32;
    while let Some(event) = event_rx.recv().await {
        loop {
            let resumed = ctx.resume_notify.notified();
            if !ctx.paused.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            resumed.await;
        }
        let _ = ctx.event_tx.send(crate::workflow::result::WorkflowEvent {
            timestamp: chrono::Utc::now(),
            workflow_run_id: ctx.workflow_run_id,
            agent_name: agent_name.to_string(),
            agent_run_id,
            event: event.clone(),
        });
        if let Some(persistence) = persistence {
            persistence
                .append_event(agent_name, agent_run_id, &event)
                .await?;
        }
        if ctx.budget.is_exceeded() && !matches!(&event, AgentEvent::UsageReported(_)) {
            let _ = cmd_tx.send(UserCommand::CancelTask).await;
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        match event {
            AgentEvent::ToolCallAwaitingApproval { tool_call_id, .. } => {
                let _ = cmd_tx
                    .send(UserCommand::ApproveToolUse {
                        tool_call_id,
                        custom_tool_output: None,
                        message_to_agent: None,
                    })
                    .await;
            }

            AgentEvent::BatchToolCallAwaitingApproval {
                batch_id,
                pending_calls,
            } => {
                let mut decisions = HashMap::new();
                for call in &pending_calls {
                    decisions.insert(
                        call.tool_call_id,
                        arko_agents::runtime::command::ToolApprovalDecision::Approve {
                            custom_tool_output: None,
                            message_to_agent: None,
                        },
                    );
                }
                let _ = cmd_tx
                    .send(UserCommand::SubmitBatchToolApprovals {
                        batch_id,
                        decisions,
                    })
                    .await;
            }

            AgentEvent::UsageReported(ref usage) => {
                // Subscription (ChatGPT OAuth) calls are recorded with a
                // billed amount of zero.
                let cost_result = ctx.budget.add_agent_cost(agent_name, usage);

                if cost_result.warning_triggered {
                    let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetWarning, |hctx| {
                        hctx.budget_spent = cost_result.new_total;
                    });
                }

                if cost_result.exceeded {
                    let _ = fire_hooks_on_context(ctx, HookPoint::OnBudgetExceeded, |hctx| {
                        hctx.budget_spent = cost_result.new_total;
                    });
                    let _ = cmd_tx.send(UserCommand::CancelTask).await;
                    return Err(WorkflowError::BudgetExceeded {
                        spent: cost_result.new_total,
                        limit: ctx.budget.limit().unwrap_or(0.0),
                    });
                }
                if ctx.budget.is_agent_exceeded(agent_name) {
                    let _ = cmd_tx.send(UserCommand::CancelTask).await;
                    return Err(WorkflowError::AgentBudgetExceeded {
                        agent: agent_name.to_string(),
                        spent: ctx.budget.agent_spent(agent_name),
                        limit: ctx.budget.agent_limit(agent_name).unwrap_or(0.0),
                    });
                }

                if ctx.budget.is_exceeded() {
                    let _ = cmd_tx.send(UserCommand::CancelTask).await;
                    return Err(WorkflowError::BudgetExceeded {
                        spent: ctx.budget.total_spent(),
                        limit: ctx.budget.limit().unwrap_or(0.0),
                    });
                }
            }

            AgentEvent::ErrorEncountered {
                requires_approval_to_retry,
                ..
            } => {
                if requires_approval_to_retry {
                    if approved_error_recoveries == 0 {
                        approved_error_recoveries += 1;
                        let _ = cmd_tx
                            .send(UserCommand::ApproveErrorRecovery { message: None })
                            .await;
                    } else {
                        let _ = cmd_tx
                            .send(UserCommand::RejectErrorRecovery {
                                message: Some(
                                    "Workflow-managed error recovery limit reached".to_string(),
                                ),
                            })
                            .await;
                    }
                }
            }

            AgentEvent::UserQuestionsAsked { questions } => {
                if let Some(handler) = &ctx.user_input_handler {
                    let response = handler
                        .respond(crate::user_input::UserInputRequest {
                            workflow_run_id: ctx.workflow_run_id,
                            agent_run_id,
                            agent_name: agent_name.to_string(),
                            questions: questions.clone(),
                        })
                        .await
                        .map_err(WorkflowError::Channel)?;
                    cmd_tx
                        .send(UserCommand::SubmitUserResponse { response })
                        .await
                        .map_err(|error| WorkflowError::Channel(error.to_string()))?;
                } else {
                    let _ = cmd_tx.send(UserCommand::SuspendTask).await;
                    return Err(WorkflowError::UserInputRequired {
                        agent: agent_name.to_string(),
                        task_id: agent_run_id,
                        questions,
                    });
                }
            }

            AgentEvent::ToolResultIntercepted {
                tool_call_id,
                ref tool_name,
                ref original_output,
                ref intercepted_output,
            } => {
                fire_hooks_on_context(ctx, HookPoint::PostToolResultInterception, |hook| {
                    hook.agent_name = Some(agent_name.to_string());
                    hook.agent_run_id = Some(agent_run_id);
                    hook.tool_call_id = Some(tool_call_id);
                    hook.tool_name = Some(tool_name.clone());
                    hook.original_tool_result = Some(original_output.clone());
                    hook.effective_tool_result = Some(intercepted_output.clone());
                    hook.tool_result = Some(intercepted_output.clone());
                })?;
            }

            AgentEvent::TaskStateChanged(TaskState::Interrupted { reason }) => {
                // The agent runtime can wait for commands after a persistent
                // transport failure. A workflow invocation must return its error
                // instead of waiting forever for TaskStreamEnd.
                let _ = cmd_tx.send(UserCommand::CancelTask).await;
                return Err(WorkflowError::AgentInterrupted {
                    agent: agent_name.to_string(),
                    reason,
                });
            }

            AgentEvent::TaskStateChanged(ref state) if state.is_terminal() => {
                break;
            }

            AgentEvent::TaskStreamEnd => break,

            _ => {}
        }
    }

    Ok(())
}

pub(crate) async fn finalize_agent_task(
    agent_name: String,
    task: Task,
    ctx: &Arc<SharedContext>,
    persistence: Option<&Arc<PersistenceManager>>,
) -> Result<AgentResult, WorkflowError> {
    let agent_run_id = task.id;
    if matches!(task.current_state, TaskState::Completed { .. }) {
        // Channel acceptance is not consumption: complete_task may finish with
        // unread InjectContext commands. Confirm only finalized user context.
        ctx.messages
            .confirm_active_context(&agent_name, agent_run_id, &task.conversation);
    }
    let (status, summary, error, hook_point) = match &task.current_state {
        TaskState::Completed { summary } => (
            crate::workflow::result::AgentRunStatus::Completed,
            summary.clone(),
            None,
            HookPoint::PostAgentCompletion,
        ),
        TaskState::Failed { reason } if reason.to_ascii_lowercase().contains("cancel") => (
            crate::workflow::result::AgentRunStatus::Cancelled,
            String::new(),
            Some(reason.clone()),
            HookPoint::PostAgentFailure,
        ),
        TaskState::Failed { reason } => (
            crate::workflow::result::AgentRunStatus::Failed,
            String::new(),
            Some(reason.clone()),
            HookPoint::PostAgentFailure,
        ),
        TaskState::Interrupted { reason } => (
            crate::workflow::result::AgentRunStatus::Interrupted,
            String::new(),
            Some(reason.clone()),
            HookPoint::PostAgentInterruption,
        ),
        TaskState::AwaitingUserResponse { questions } => (
            crate::workflow::result::AgentRunStatus::WaitingForInput,
            String::new(),
            Some(questions.clone()),
            HookPoint::OnUserInputRequired,
        ),
        state => (
            crate::workflow::result::AgentRunStatus::Interrupted,
            String::new(),
            Some(format!("Agent ended in non-terminal state: {state:?}")),
            HookPoint::PostAgentInterruption,
        ),
    };

    let result = AgentResult {
        agent_run_id,
        agent_name: agent_name.clone(),
        status,
        summary: summary.clone(),
        error: error.clone(),
        task,
    };
    {
        let mut results = ctx.agent_results.write();
        if let Some(existing) = results
            .iter_mut()
            .find(|existing| existing.agent_run_id == agent_run_id)
        {
            *existing = result.clone();
        } else {
            results.push(result.clone());
        }
    }

    let hook_result = fire_hooks_on_context(ctx, hook_point, |hook| {
        hook.agent_name = Some(agent_name);
        hook.agent_run_id = Some(agent_run_id);
        hook.agent_summary = (!summary.is_empty()).then_some(summary);
        hook.error_message = error;
    });

    if let Some(persistence) = persistence {
        persistence.save_agent_task(&result.task).await?;
        persistence.auto_checkpoint(ctx).await?;
    }

    hook_result?;

    Ok(result)
}

pub(crate) struct WorkflowEngine {
    llm_client: Arc<LLMClient>,
    persistence: Option<Arc<PersistenceManager>>,
}

impl WorkflowEngine {
    fn setup_skills(
        config: &mut AgentConfig,
        registry: &mut ToolRegistry,
        workspace: &std::path::Path,
    ) -> Result<Option<SkillPromptContext>, WorkflowError> {
        let Some(source) = config.skills_config.as_ref() else {
            return Ok(None);
        };
        let manager = SkillManager::load(source, workspace)?;
        Ok(Some(manager.setup(registry, config)))
    }

    pub fn new(llm_client: Arc<LLMClient>, persistence: Option<PersistenceManager>) -> Self {
        Self {
            llm_client,
            persistence: persistence.map(Arc::new),
        }
    }

    pub(crate) fn from_shared(
        llm_client: Arc<LLMClient>,
        persistence: Option<Arc<PersistenceManager>>,
    ) -> Self {
        Self {
            llm_client,
            persistence,
        }
    }

    fn build_tool_registry(&self, agent_name: &str, ctx: &Arc<SharedContext>) -> ToolRegistry {
        let mut registry = ToolRegistry::new();

        if !ctx.memory.is_empty() {
            registry.register(Arc::new(ReadMemoryTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
            registry.register(Arc::new(WriteMemoryTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
            registry.register(Arc::new(ListMemoriesTool::new(
                ctx.clone(),
                agent_name.to_string(),
            )));
        }

        registry.register(Arc::new(SendMessageTool::new(
            ctx.clone(),
            agent_name.to_string(),
        )));
        if let Some(services) = ctx.workflow_services.get()
            && services.is_allowed(agent_name)
        {
            registry.register(Arc::new(RunWorkflowAgentTool::new(
                services.clone(),
                agent_name.to_string(),
            )));
        }
        if let Some(ballots) = &ctx.ballots {
            registry.register(Arc::new(CastVoteTool::new(
                ballots.clone(),
                agent_name.to_string(),
            )));
        }

        for tool in &ctx.global_tools {
            registry.register(tool.clone());
        }
        if let Some(agent_tools) = ctx.agent_tools.get(agent_name) {
            for tool in agent_tools {
                registry.register(tool.clone());
            }
        }

        registry
    }

    /// Composes global and agent-specific interceptors into a single
    /// `InterceptorConfig`, or `None` if no interceptors are configured
    /// for this agent.
    fn build_interceptor_config(
        &self,
        agent_name: &str,
        ctx: &Arc<SharedContext>,
    ) -> Option<InterceptorConfig> {
        let mut all: Vec<Arc<dyn ToolResultInterceptor>> = Vec::new();
        all.extend(ctx.global_interceptors.iter().cloned());
        if let Some(agent_specific) = ctx.agent_interceptors.get(agent_name) {
            all.extend(agent_specific.iter().cloned());
        }
        for config in &ctx.distillation_configs {
            all.push(Arc::new(DistillationInterceptor {
                config: config.clone(),
                client: self.llm_client.clone(),
                context: ctx.clone(),
            }));
        }

        if all.is_empty() {
            return None;
        }

        let interceptor: Arc<dyn ToolResultInterceptor> = if all.len() == 1 {
            all.remove(0)
        } else {
            Arc::new(ChainedInterceptor {
                interceptors: all,
                failure_policy: ctx.interception_failure_policy.clone(),
            })
        };

        Some(InterceptorConfig {
            interceptor,
            failure_policy: ctx.interception_failure_policy.clone(),
        })
    }

    fn build_execution_middleware(
        &self,
        agent_name: &str,
        ctx: &Arc<SharedContext>,
    ) -> Option<Arc<dyn ToolExecutionMiddleware>> {
        if ctx.resource_tracker.is_none()
            && ctx.completion_verifiers.is_empty()
            && !ctx.hooks.has_hooks(&HookPoint::PreToolExecution)
            && !ctx.hooks.has_hooks(&HookPoint::PostToolExecution)
        {
            return None;
        }

        Some(Arc::new(WorkflowExecutionMiddleware {
            agent_name: agent_name.to_string(),
            context: ctx.clone(),
            llm_client: self.llm_client.clone(),
            persistence: self.persistence.clone(),
        }))
    }

    fn prepare_agent_config(
        &self,
        agent_name: &str,
        base_config: &AgentConfig,
        ctx: &Arc<SharedContext>,
    ) -> AgentConfig {
        let mut config = base_config.clone();

        if !ctx.memory.is_empty() {
            ensure_tool_name(&mut config, READ_MEMORY_TOOL_NAME);
            ensure_tool_name(&mut config, WRITE_MEMORY_TOOL_NAME);
            ensure_tool_name(&mut config, LIST_MEMORIES_TOOL_NAME);
        }
        ensure_tool_name(&mut config, SEND_MESSAGE_TOOL_NAME);
        if let Some(services) = ctx.workflow_services.get()
            && services.is_allowed(agent_name)
        {
            ensure_tool_name(&mut config, RUN_WORKFLOW_AGENT_TOOL_NAME);
        }
        if ctx.ballots.is_some() {
            ensure_tool_name(&mut config, CAST_VOTE_TOOL_NAME);
        }

        for tool in &ctx.global_tools {
            ensure_tool_name(&mut config, &tool.name());
        }
        if let Some(agent_tools) = ctx.agent_tools.get(agent_name) {
            for tool in agent_tools {
                ensure_tool_name(&mut config, &tool.name());
            }
        }

        config
    }

    fn augment_prompt_with_messages(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> (String, Vec<uuid::Uuid>) {
        let messages = ctx.messages.pending(agent_name);
        let message_ids = messages.iter().map(|message| message.id).collect();
        let mut augmented = prompt.to_string();
        if ctx.agent_configs.len() > 1 {
            augmented.push_str("\n\n# Workflow agent roster\n");
            let mut roster = ctx.agent_configs.iter().collect::<Vec<_>>();
            roster.sort_by_key(|(name, _)| *name);
            for (name, config) in roster {
                augmented.push_str(&format!("\n- {name}: {}", config.role_description));
            }
        }
        if !messages.is_empty() {
            augmented.push_str("\n\n# Messages from other agents:\n");
            for msg in messages {
                augmented.push_str(&format!(
                    "\n--- Message {} from '{}' ---\n{}{}\n--- End message ---\n",
                    msg.id,
                    msg.from,
                    msg.content,
                    if msg.attachments.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "\nContext references: {}",
                            serde_json::to_string(&msg.attachments).unwrap_or_default()
                        )
                    }
                ));
            }
        }
        (augmented, message_ids)
    }

    /// Run a single agent to completion, blocking until it finishes.
    pub async fn run_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        if ctx.budget.is_exceeded() {
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        fire_hooks_on_context(ctx, HookPoint::PreAgentSpawn, |hctx| {
            hctx.agent_name = Some(agent_name.to_string());
        })?;

        let base_config = ctx
            .agent_configs
            .get(agent_name)
            .ok_or_else(|| WorkflowError::AgentNotFound(agent_name.to_string()))?
            .clone();

        let mut config = self.prepare_agent_config(agent_name, &base_config, ctx);
        let mut registry = self.build_tool_registry(agent_name, ctx);
        let skill_prompt_context = Self::setup_skills(&mut config, &mut registry, &ctx.workspace)?;
        let tool_registry = Arc::new(registry);
        let interceptor_config = self.build_interceptor_config(agent_name, ctx);
        let execution_middleware = self.build_execution_middleware(agent_name, ctx);
        let (augmented_prompt, pending_message_ids) =
            self.augment_prompt_with_messages(agent_name, prompt, ctx);

        let task = Task::new(config, augmented_prompt, ctx.workspace.clone(), None);
        let agent_run_id = task.id;
        let runtime = AgentRuntime::new(self.llm_client.clone());
        let (mut event_rx, cmd_tx, join_handle) = runtime.run_task(
            task,
            tool_registry,
            None,
            false,
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
        );
        ctx.messages
            .register_active(agent_name, agent_run_id, cmd_tx.clone());

        let drive_result = drive_event_loop(
            agent_name,
            agent_run_id,
            &mut event_rx,
            &cmd_tx,
            ctx,
            self.persistence.as_ref(),
        )
        .await;

        if drive_result
            .as_ref()
            .is_err_and(|error| !matches!(error, WorkflowError::UserInputRequired { .. }))
        {
            let _ = cmd_tx.send(UserCommand::CancelTask).await;
        }

        let join_result = join_handle.await;
        let final_task = join_result
            .map_err(|e| WorkflowError::Channel(format!("Agent join error: {}", e)))?
            .map_err(WorkflowError::Agent)?;

        let agent_result = finalize_agent_task(
            agent_name.to_string(),
            final_task,
            ctx,
            self.persistence.as_ref(),
        )
        .await?;
        if agent_result.status == crate::workflow::result::AgentRunStatus::Completed {
            ctx.messages.acknowledge_active(agent_name, agent_run_id);
            ctx.messages.acknowledge(agent_name, &pending_message_ids);
        }
        ctx.messages.unregister_active(agent_name, agent_run_id);

        drive_result?;
        match agent_result.status {
            crate::workflow::result::AgentRunStatus::Completed => Ok(agent_result),
            crate::workflow::result::AgentRunStatus::WaitingForInput => {
                Err(WorkflowError::UserInputRequired {
                    agent: agent_name.to_string(),
                    task_id: agent_result.agent_run_id,
                    questions: agent_result.error.clone().unwrap_or_default(),
                })
            }
            crate::workflow::result::AgentRunStatus::Interrupted => {
                Err(WorkflowError::AgentInterrupted {
                    agent: agent_name.to_string(),
                    reason: agent_result.error.clone().unwrap_or_default(),
                })
            }
            _ => Err(WorkflowError::Agent(
                arko_agents::ArkoAgentsError::TaskExecution(
                    agent_result
                        .error
                        .clone()
                        .unwrap_or_else(|| "Agent failed".to_string()),
                ),
            )),
        }
    }

    /// Run multiple agents in parallel.
    pub async fn run_agents_parallel(
        &self,
        agents: Vec<(&str, String)>,
        ctx: &Arc<SharedContext>,
    ) -> Result<Vec<AgentResult>, WorkflowError> {
        let mut join_handles = Vec::new();

        for (name, prompt) in agents {
            let engine_client = self.llm_client.clone();
            let persistence = self.persistence.clone();
            let ctx_clone = ctx.clone();
            let name_owned = name.to_string();

            let handle = tokio::spawn({
                let engine = WorkflowEngine {
                    llm_client: engine_client,
                    persistence,
                };
                async move { engine.run_agent(&name_owned, &prompt, &ctx_clone).await }
            });
            join_handles.push(handle);
        }

        let mut results = Vec::new();
        let mut first_error = None;
        while !join_handles.is_empty() {
            let handle = join_handles.remove(0);
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(WorkflowError::Channel(format!(
                            "Parallel agent join error: {error}"
                        )));
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(results),
        }
    }

    /// Spawn an agent without blocking, returning a handle for driving it.
    ///
    /// Performs all agent setup (hook firing, config preparation, tool registry
    /// construction, message augmentation, task creation) then starts the agent
    /// runtime and returns the raw channels wrapped in an [`AgentHandle`].
    ///
    /// The caller drives execution either by calling [`AgentHandle::wait`]
    /// (default workflow behavior) or [`AgentHandle::into_parts`] (custom
    /// event processing).
    pub async fn spawn_agent(
        &self,
        agent_name: &str,
        prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentHandle, WorkflowError> {
        if ctx.budget.is_exceeded() {
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        fire_hooks_on_context(ctx, HookPoint::PreAgentSpawn, |hctx| {
            hctx.agent_name = Some(agent_name.to_string());
        })?;

        let base_config = ctx
            .agent_configs
            .get(agent_name)
            .ok_or_else(|| WorkflowError::AgentNotFound(agent_name.to_string()))?
            .clone();

        let mut config = self.prepare_agent_config(agent_name, &base_config, ctx);
        let mut registry = self.build_tool_registry(agent_name, ctx);
        let skill_prompt_context = Self::setup_skills(&mut config, &mut registry, &ctx.workspace)?;
        let tool_registry = Arc::new(registry);
        let interceptor_config = self.build_interceptor_config(agent_name, ctx);
        let execution_middleware = self.build_execution_middleware(agent_name, ctx);
        let (augmented_prompt, pending_message_ids) =
            self.augment_prompt_with_messages(agent_name, prompt, ctx);

        let task = Task::new(config, augmented_prompt, ctx.workspace.clone(), None);
        let agent_run_id = task.id;
        let runtime = AgentRuntime::new(self.llm_client.clone());
        let (event_rx, cmd_tx, join_handle) = runtime.run_task(
            task,
            tool_registry,
            None,
            false,
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
        );
        ctx.messages
            .register_active(agent_name, agent_run_id, cmd_tx.clone());

        Ok(AgentHandle {
            event_rx,
            cmd_tx,
            join_handle,
            agent_name: agent_name.to_string(),
            agent_run_id,
            context: ctx.clone(),
            persistence: self.persistence.clone(),
            pending_message_ids,
        })
    }

    pub async fn continue_agent(
        &self,
        previous: &AgentResult,
        continuation_prompt: &str,
        ctx: &Arc<SharedContext>,
    ) -> Result<AgentResult, WorkflowError> {
        if !matches!(
            previous.status,
            crate::workflow::result::AgentRunStatus::Completed
                | crate::workflow::result::AgentRunStatus::WaitingForInput
        ) {
            return Err(WorkflowError::BuilderInvalid(
                "only a completed or user-input-suspended agent run can be continued".to_string(),
            ));
        }
        if ctx.budget.is_exceeded() {
            return Err(WorkflowError::BudgetExceeded {
                spent: ctx.budget.total_spent(),
                limit: ctx.budget.limit().unwrap_or(0.0),
            });
        }
        let agent_name = &previous.agent_name;
        fire_hooks_on_context(ctx, HookPoint::PreAgentSpawn, |hook| {
            hook.agent_name = Some(agent_name.clone());
            hook.agent_run_id = Some(previous.agent_run_id);
        })?;
        let mut task = previous.task.clone();
        let (prompt, pending_message_ids) =
            self.augment_prompt_with_messages(agent_name, continuation_prompt, ctx);
        match previous.status {
            crate::workflow::result::AgentRunStatus::Completed => {
                task.prepare_for_resumption_from_completed(prompt)?;
            }
            crate::workflow::result::AgentRunStatus::WaitingForInput => {
                task.prepare_for_resumption_with_user_response(prompt)?;
            }
            _ => unreachable!(),
        }
        let mut registry = self.build_tool_registry(agent_name, ctx);
        let skill_prompt_context =
            Self::setup_skills(&mut task.agent_config, &mut registry, &ctx.workspace)?;
        let tool_registry = Arc::new(registry);
        let interceptor_config = self.build_interceptor_config(agent_name, ctx);
        let execution_middleware = self.build_execution_middleware(agent_name, ctx);
        let agent_run_id = task.id;
        let runtime = AgentRuntime::new(self.llm_client.clone());
        let (mut event_rx, cmd_tx, join_handle) = runtime.run_task(
            task,
            tool_registry,
            None,
            false,
            skill_prompt_context,
            interceptor_config,
            execution_middleware,
        );
        ctx.messages
            .register_active(agent_name, agent_run_id, cmd_tx.clone());
        let drive_result = drive_event_loop(
            agent_name,
            agent_run_id,
            &mut event_rx,
            &cmd_tx,
            ctx,
            self.persistence.as_ref(),
        )
        .await;
        if drive_result
            .as_ref()
            .is_err_and(|error| !matches!(error, WorkflowError::UserInputRequired { .. }))
        {
            let _ = cmd_tx.send(UserCommand::CancelTask).await;
        }
        let join_result = join_handle.await;
        let task = join_result
            .map_err(|error| WorkflowError::Channel(format!("Agent join error: {error}")))?
            .map_err(WorkflowError::Agent)?;
        let result =
            finalize_agent_task(agent_name.clone(), task, ctx, self.persistence.as_ref()).await?;
        if result.status == crate::workflow::result::AgentRunStatus::Completed {
            ctx.messages.acknowledge_active(agent_name, agent_run_id);
            ctx.messages.acknowledge(agent_name, &pending_message_ids);
        }
        ctx.messages.unregister_active(agent_name, agent_run_id);
        drive_result?;
        if result.status == crate::workflow::result::AgentRunStatus::Completed {
            Ok(result)
        } else {
            Err(WorkflowError::Agent(
                arko_agents::ArkoAgentsError::TaskExecution(
                    result
                        .error
                        .clone()
                        .unwrap_or_else(|| "Agent did not complete".to_string()),
                ),
            ))
        }
    }

    /// Save a checkpoint of the current workflow state.
    pub async fn checkpoint(&self, ctx: &Arc<SharedContext>) -> Result<(), WorkflowError> {
        if let Some(ref persistence) = self.persistence {
            persistence.save_checkpoint(ctx).await?;
        }
        Ok(())
    }
}

fn ensure_tool_name(config: &mut AgentConfig, tool_name: &str) {
    if !config
        .tool_config
        .available_tool_names
        .iter()
        .any(|n| n == tool_name)
    {
        config
            .tool_config
            .available_tool_names
            .push(tool_name.to_string());
    }
}

/// Enforces workflow policy at the actual tool execution boundary, independent
/// of whether a call was auto-approved, manually approved, or part of a batch.
struct WorkflowExecutionMiddleware {
    /// Stable key used to register this agent in the workflow. AgentConfig::name
    /// is a human-readable display name and must not be used for policy matching.
    agent_name: String,
    context: Arc<SharedContext>,
    llm_client: Arc<LLMClient>,
    persistence: Option<Arc<PersistenceManager>>,
}

impl WorkflowExecutionMiddleware {
    fn applicable_completion_verifiers(&self) -> impl Iterator<Item = &CompletionVerifierConfig> {
        self.context
            .completion_verifiers
            .iter()
            .filter(|config| config.applies_to(&self.agent_name))
    }

    async fn verify_completion(
        &self,
        execution: &ToolExecutionContext<'_>,
        summary: &str,
    ) -> Option<ToolExecutionDecision> {
        const MAX_FORMAT_RETRIES: usize = 1;

        for verifier in self.applicable_completion_verifiers() {
            if verifier.kind == VerificationKind::Efficiency {
                let repair_count = self
                    .context
                    .verification_attempts
                    .lock()
                    .get(&(execution.task_id, verifier.name.clone()))
                    .map_or(0, |counts| counts.0);
                if verifier.efficiency_repair_budget_exhausted(repair_count) {
                    continue;
                }
            }
            let engine = WorkflowEngine {
                llm_client: self.llm_client.clone(),
                persistence: self.persistence.clone(),
            };
            let session_key = (execution.task_id, verifier.name.clone());
            let prior_record = verifier
                .reuse_conversation
                .then(|| {
                    self.context
                        .verification_records
                        .read()
                        .iter()
                        .rev()
                        .find(|record| {
                            record.target_task_id == execution.task_id
                                && record.verifier_name == verifier.name
                        })
                        .cloned()
                })
                .flatten();
            let prior_result = prior_record.as_ref().and_then(|record| {
                self.context
                    .agent_results
                    .read()
                    .iter()
                    .find(|result| result.agent_run_id == record.verifier_task_id)
                    .cloned()
            });
            let prior_baseline = self
                .context
                .verification_baselines
                .lock()
                .get(&session_key)
                .cloned();
            let target_actor = ActorId::new(execution.task_id, &self.agent_name);

            let mut result = if let (Some(record), Some(previous)) =
                (prior_record.as_ref(), prior_result.as_ref())
            {
                let prior_decision = match record.decision {
                    VerificationDecision::Approved => "ACCEPTED",
                    VerificationDecision::RejectedMinor => "REJECTED_MINOR",
                    VerificationDecision::RejectedMajor => "REJECTED_MAJOR",
                };
                let repair_delta = verifier
                    .repair_context
                    .as_ref()
                    .zip(prior_baseline.as_ref())
                    .map(|(config, baseline)| {
                        crate::verification::render_repair_delta(
                            &self.context,
                            &verifier.verifier_agent,
                            baseline,
                            config.max_bytes,
                        )
                    })
                    .unwrap_or_else(|| {
                        "No bounded repair delta is available; perform only the focused reads needed to validate the new completion.".to_string()
                    });
                let prompt = format!(
                    "Continue your review of the same '{}' task.\n\nYour prior decision:\n{}\n{}\n\nNew candidate completion summary:\n{}\n\n{}\n\nRe-review instructions:\n- Verify every prior finding and any correctness or efficiency regression materially exposed by the delta.\n- Reuse unchanged passing evidence from your conversation; do not repeat the complete audit or rerun unchanged broad checks.\n- Run focused reproductions and directly affected checks.\n- Report a new issue only when it is material and tied to an explicit requirement or directly exposed by the repair.\n- Batch independent read-only tool calls when useful, but call non-batch-safe tools alone.\n\nVerification instructions remain:\n{}\n\nFinish with complete_task using this exact contract:\n\n{}",
                    self.agent_name,
                    prior_decision,
                    record.feedback,
                    summary,
                    repair_delta,
                    verifier.instructions,
                    crate::verification::VERIFIER_OUTPUT_CONTRACT,
                );
                match engine
                    .continue_agent(previous, &prompt, &self.context)
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' could not continue: {error}",
                                verifier.name
                            ),
                        });
                    }
                }
            } else {
                let prompt = format!(
                    "You are a completion verifier for agent '{}'.\n\nOriginal task:\n{}\n\nCandidate completion summary:\n{}\n\nVerification instructions:\n{}\n\nThis is the first review pass. Inspect the actual workspace and relevant artifacts, systematically cover the applicable requirement and failure classes, and consolidate every evidenced related finding into this verdict rather than drip-feeding one counterexample per repair cycle. Do not approve merely because the summary claims success. Batch independent read-only tool calls when useful, but call non-batch-safe tools alone. Finish by calling complete_task using this exact contract:\n\n{}",
                    self.agent_name,
                    execution.initial_task_prompt,
                    summary,
                    verifier.instructions,
                    crate::verification::VERIFIER_OUTPUT_CONTRACT,
                );
                match engine
                    .run_agent(&verifier.verifier_agent, &prompt, &self.context)
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' could not complete: {error}",
                                verifier.name
                            ),
                        });
                    }
                }
            };

            let mut format_attempt = 0;
            let response = loop {
                match crate::verification::parse_response(&result.summary) {
                    Ok(response) => break response,
                    Err(error) if format_attempt < MAX_FORMAT_RETRIES => {
                        format_attempt += 1;
                        let prompt = format!(
                            "Your verification conclusion could not be parsed ({error}). Do not repeat the review or change your conclusion. Re-emit it through complete_task using only this contract:\n\n{}",
                            crate::verification::VERIFIER_OUTPUT_CONTRACT,
                        );
                        result = match engine.continue_agent(&result, &prompt, &self.context).await
                        {
                            Ok(result) => result,
                            Err(retry_error) => {
                                return Some(ToolExecutionDecision::FailTask {
                                    reason: format!(
                                        "completion verifier '{}' could not repair its decision format: {retry_error}",
                                        verifier.name
                                    ),
                                });
                            }
                        };
                    }
                    Err(error) => {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' produced an unusable decision after {} attempts: {error}",
                                verifier.name,
                                format_attempt + 1,
                            ),
                        });
                    }
                }
            };
            let mut decision = response.decision;
            if verifier.kind == VerificationKind::Efficiency
                && decision == VerificationDecision::RejectedMajor
            {
                decision = VerificationDecision::RejectedMinor;
            }
            if let Some(config) = &verifier.repair_context {
                let verifier_actor = ActorId::new(result.agent_run_id, &verifier.verifier_agent);
                let baseline = crate::verification::capture_repair_baseline(
                    &self.context,
                    &target_actor,
                    &verifier_actor,
                    config.max_bytes,
                );
                self.context
                    .verification_baselines
                    .lock()
                    .insert(session_key, baseline);
            }
            self.context
                .verification_records
                .write()
                .push(VerificationRecord {
                    timestamp: chrono::Utc::now(),
                    target_task_id: execution.task_id,
                    target_agent: self.agent_name.clone(),
                    verifier_name: verifier.name.clone(),
                    verifier_agent: verifier.verifier_agent.clone(),
                    verifier_task_id: result.agent_run_id,
                    decision: decision.clone(),
                    feedback: response.feedback.clone(),
                });

            match decision {
                VerificationDecision::Approved => {}
                VerificationDecision::RejectedMinor => {
                    let count = {
                        let mut attempts = self.context.verification_attempts.lock();
                        let counts = attempts
                            .entry((execution.task_id, verifier.name.clone()))
                            .or_default();
                        counts.0 = counts.0.saturating_add(1);
                        counts.0
                    };
                    if count > verifier.max_minor_rejections {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' exhausted its minor-correction cap ({}): {}",
                                verifier.name, verifier.max_minor_rejections, response.feedback
                            ),
                        });
                    }
                    return Some(ToolExecutionDecision::Reject {
                        message_to_agent: format!(
                            "Completion was rejected with minor corrections by verifier '{}'. Fix the issues and try again ({count}/{}): {}",
                            verifier.name, verifier.max_minor_rejections, response.feedback
                        ),
                    });
                }
                VerificationDecision::RejectedMajor => {
                    let count = {
                        let mut attempts = self.context.verification_attempts.lock();
                        let counts = attempts
                            .entry((execution.task_id, verifier.name.clone()))
                            .or_default();
                        counts.1 = counts.1.saturating_add(1);
                        counts.1
                    };
                    if count > verifier.max_major_rejections {
                        return Some(ToolExecutionDecision::FailTask {
                            reason: format!(
                                "completion verifier '{}' issued a major rejection: {}",
                                verifier.name, response.feedback
                            ),
                        });
                    }
                    return Some(ToolExecutionDecision::Reject {
                        message_to_agent: format!(
                            "Completion was rejected for a major issue by verifier '{}'. Rework it before trying again ({count}/{}): {}",
                            verifier.name, verifier.max_major_rejections, response.feedback
                        ),
                    });
                }
            }
        }
        None
    }
}

#[async_trait]
impl ToolExecutionMiddleware for WorkflowExecutionMiddleware {
    async fn before(
        &self,
        execution: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
    ) -> Result<ToolExecutionDecision, ToolError> {
        match fire_hooks_returning_impl(&self.context, HookPoint::PreToolExecution, |hook| {
            hook.agent_name = Some(self.agent_name.clone());
            hook.agent_run_id = Some(execution.task_id);
            hook.tool_name = Some(call.tool_name.to_string());
            hook.tool_call_id = Some(call.tool_call_id);
            hook.tool_params = Some(call.parameters.clone());
        })
        .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?
        {
            HookResult::Continue => {}
            HookResult::Skip => {
                return Ok(ToolExecutionDecision::Reject {
                    message_to_agent: "Skipped by PreToolExecution hook".to_string(),
                });
            }
            HookResult::Abort(reason) => {
                return Err(ToolError::ExecutionFailed(format!(
                    "workflow aborted by PreToolExecution hook: {reason}"
                )));
            }
        }

        if let Some(tracker) = &self.context.resource_tracker
            && let Some((resource, true)) =
                tracker.resource_for_tool(call.tool_name, call.parameters)
        {
            let actor = ActorId::new(execution.task_id, &self.agent_name);
            let conflicts = tracker.check_write(&actor, &resource);
            if tracker.config().block_on_conflict && !conflicts.is_empty() {
                let writers = conflicts
                    .iter()
                    .filter_map(|conflict| conflict.last_writer.as_ref())
                    .map(|writer| writer.agent_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Ok(ToolExecutionDecision::Reject {
                    message_to_agent: format!(
                        "Context freshness conflict for {resource:?}; your context was changed by {writers}. Re-read the affected resource before writing."
                    ),
                });
            }
        }

        if call.tool_name == arko_agents::tool::builtins::COMPLETE_TASK_TOOL_NAME {
            let summary = call
                .parameters
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(decision) = self.verify_completion(execution, summary).await {
                return Ok(decision);
            }
        }

        Ok(ToolExecutionDecision::Continue)
    }

    async fn after(
        &self,
        execution: &ToolExecutionContext<'_>,
        call: &ToolCallContext<'_>,
        outcome: ToolExecutionOutcome<'_>,
    ) -> Result<(), ToolError> {
        let successful_output = match outcome {
            ToolExecutionOutcome::Success(output)
            | ToolExecutionOutcome::CallerProvided(output) => Some(output),
            ToolExecutionOutcome::Error(_) => None,
        };

        if successful_output.is_some()
            && call.tool_name == arko_agents::tool::builtins::COMPLETE_TASK_TOOL_NAME
        {
            self.context
                .verification_baselines
                .lock()
                .retain(|(task_id, _), _| *task_id != execution.task_id);
            self.context
                .verification_attempts
                .lock()
                .retain(|(task_id, _), _| *task_id != execution.task_id);
        }

        if successful_output.is_some()
            && let Some(tracker) = &self.context.resource_tracker
            && let Some((resource, is_write)) =
                tracker.resource_for_tool(call.tool_name, call.parameters)
        {
            let actor = ActorId::new(execution.task_id, &self.agent_name);
            if is_write {
                let stale_readers = tracker.record_write(&actor, resource.clone());
                for reader in stale_readers {
                    let notice = format!(
                        "Context freshness notice: {resource:?} changed after you read it. Re-read it before relying on or modifying that context."
                    );
                    self.context
                        .messages
                        .inject_task(reader.task_id, notice)
                        .await;
                    fire_hooks_on_context(&self.context, HookPoint::OnMesiConflict, |hook| {
                        hook.agent_name = Some(reader.agent_name.clone());
                        hook.agent_run_id = Some(reader.task_id);
                        hook.metadata.insert(
                            "resource".to_string(),
                            serde_json::json!(format!("{resource:?}")),
                        );
                        hook.metadata
                            .insert("writer".to_string(), serde_json::json!(self.agent_name));
                    })
                    .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
                }
            } else {
                tracker.record_read(&actor, resource);
            }
        }

        let result = match outcome {
            ToolExecutionOutcome::Success(output)
            | ToolExecutionOutcome::CallerProvided(output)
            | ToolExecutionOutcome::Error(output) => output,
        };

        match fire_hooks_returning_impl(&self.context, HookPoint::PostToolExecution, |hook| {
            hook.agent_name = Some(self.agent_name.clone());
            hook.agent_run_id = Some(execution.task_id);
            hook.tool_name = Some(call.tool_name.to_string());
            hook.tool_call_id = Some(call.tool_call_id);
            hook.tool_params = Some(call.parameters.clone());
            hook.tool_result = Some(result.to_string());
            hook.original_tool_result = Some(result.to_string());
        })
        .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?
        {
            HookResult::Abort(reason) => Err(ToolError::ExecutionFailed(format!(
                "workflow aborted by PostToolExecution hook: {reason}"
            ))),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use arko_agents::AgentLLMConfig;
    use arko_ai::{GenerationConfig, Model};

    use crate::workflow::Workflow;

    #[tokio::test]
    async fn interrupted_runtime_is_cancelled_instead_of_waiting_for_stream_end() {
        let config = AgentConfig::builder(
            "designer".into(),
            "Designer".into(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_sol(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("Designer".into())
        .instructions("Design".into())
        .build()
        .unwrap();
        let workflow = Workflow::builder()
            .name("interruption-test")
            .workspace(std::env::current_dir().unwrap())
            .llm_client(Arc::new(LLMClient::new()))
            .agent("designer", config)
            .build()
            .unwrap();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(4);
        let (command_tx, mut command_rx) = tokio::sync::mpsc::channel(4);
        event_tx
            .send(AgentEvent::TaskStateChanged(TaskState::Interrupted {
                reason: "Persistent network failure".into(),
            }))
            .await
            .unwrap();
        // Keep event_tx alive: an interrupted runtime has not closed its stream.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            drive_event_loop(
                "designer",
                uuid::Uuid::new_v4(),
                &mut event_rx,
                &command_tx,
                workflow.context(),
                None,
            ),
        )
        .await
        .expect("interrupted workflow must not hang");
        assert!(
            matches!(result, Err(WorkflowError::AgentInterrupted { reason, .. }) if reason == "Persistent network failure")
        );
        assert!(matches!(
            command_rx.try_recv().unwrap(),
            UserCommand::CancelTask
        ));
    }

    #[test]
    fn workflow_skill_setup_injects_auto_load_and_registers_on_demand() {
        use arko_agents::skill::{InvalidSkillPolicy, SkillSource, SkillsConfig};
        let workspace =
            std::env::temp_dir().join(format!("arko-workflow-skills-{}", uuid::Uuid::new_v4()));
        for (name, auto_load) in [("principles", true), ("reference", false)] {
            let dir = workspace.join(".skills").join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("SKILL.md"), format!("---\nname: {name}\ndescription: Test skill\nauto_load: {auto_load}\n---\n\n{name} content")).unwrap();
        }
        let mut config = AgentConfig::builder(
            "designer".into(),
            "Designer".into(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_sol(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("Designer".into())
        .instructions("Design".into())
        .build()
        .unwrap();
        config.skills_config = Some(SkillsConfig {
            workspace_source: Some(SkillSource {
                path: ".skills".into(),
                include: None,
                exclude: None,
            }),
            global_sources: vec![],
            on_invalid_skill: InvalidSkillPolicy::Error,
        });
        let mut registry = ToolRegistry::new();
        let context = WorkflowEngine::setup_skills(&mut config, &mut registry, &workspace)
            .unwrap()
            .unwrap();
        assert_eq!(
            context.auto_loaded_skills[0].content.trim(),
            "principles content"
        );
        assert_eq!(context.on_demand_skills[0].name, "reference");
        assert!(registry.get("load_skill").is_some());
        assert!(
            config
                .tool_config
                .available_tool_names
                .contains(&"load_skill".into())
        );
        std::fs::write(
            workspace.join(".skills/principles/SKILL.md"),
            "invalid frontmatter",
        )
        .unwrap();
        assert!(WorkflowEngine::setup_skills(&mut config, &mut registry, &workspace).is_err());
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[tokio::test]
    async fn middleware_uses_registered_agent_name_instead_of_display_name() {
        let workspace = std::env::temp_dir().join(format!(
            "arko-workflow-agent-identity-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&workspace).unwrap();
        let observed_agent = Arc::new(Mutex::new(None));
        let hook_observation = observed_agent.clone();
        let config = AgentConfig::builder(
            "coder-config".to_string(),
            "Friendly Coder Display Name".to_string(),
            AgentLLMConfig {
                model: Model::openai_gpt_5_6_luna(),
                generation_config: GenerationConfig::default(),
            },
        )
        .role_description("test coder".to_string())
        .instructions("test instructions".to_string())
        .build()
        .unwrap();
        let mut reviewer_config = config.clone();
        reviewer_config.id = "reviewer-config".to_string();
        reviewer_config.name = "Friendly Reviewer Display Name".to_string();
        let workflow = Workflow::builder()
            .name("agent-identity-test")
            .workspace(&workspace)
            .llm_client(Arc::new(LLMClient::new()))
            .agent("coder", config)
            .agent("reviewer", reviewer_config)
            .completion_verifier(
                CompletionVerifierConfig::correctness("correctness", "reviewer").target("coder"),
            )
            .hook(
                HookPoint::PreToolExecution,
                Arc::new(move |context| {
                    *hook_observation.lock().unwrap() = context.agent_name.clone();
                    HookResult::Continue
                }),
            )
            .build()
            .unwrap();
        let concrete_middleware = WorkflowExecutionMiddleware {
            agent_name: "coder".to_string(),
            context: workflow.context().clone(),
            llm_client: Arc::new(LLMClient::new()),
            persistence: None,
        };
        assert_eq!(
            concrete_middleware
                .applicable_completion_verifiers()
                .map(|verifier| verifier.name.as_str())
                .collect::<Vec<_>>(),
            vec!["correctness"]
        );
        let middleware = workflow
            .engine
            .build_execution_middleware("coder", workflow.context())
            .unwrap();
        let parameters = HashMap::new();
        let execution = ToolExecutionContext {
            task_id: uuid::Uuid::new_v4(),
            agent_name: "Friendly Coder Display Name",
            initial_task_prompt: "test",
            current_working_directory: &workspace,
        };
        let call = ToolCallContext {
            tool_call_id: uuid::Uuid::new_v4(),
            tool_name: "test_tool",
            parameters: &parameters,
        };

        let decision = middleware.before(&execution, &call).await.unwrap();

        assert_eq!(decision, ToolExecutionDecision::Continue);
        assert_eq!(observed_agent.lock().unwrap().as_deref(), Some("coder"));
        std::fs::remove_dir(&workspace).unwrap();
    }
}
