/// Checkpoint and resume support for workflows.
use std::path::PathBuf;

use crate::context::SharedContext;
use crate::error::WorkflowError;

pub mod state;

/// Configuration for workflow persistence.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Directory where checkpoint files are stored.
    pub checkpoint_dir: PathBuf,
    /// Automatically checkpoint after each agent completion.
    pub auto_checkpoint: bool,
}

/// Manages checkpointing and restoration of workflow state.
#[derive(Clone)]
pub struct PersistenceManager {
    config: PersistenceConfig,
    durable_store: std::sync::Arc<crate::durable::DurableStore>,
    write_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
}

impl PersistenceConfig {
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            auto_checkpoint: true,
        }
    }

    /// Construct an OS-appropriate per-user run directory without requiring
    /// the workflow developer to hard-code a platform path.
    pub fn user_data(workflow_name: &str) -> Result<Self, WorkflowError> {
        let base = if cfg!(target_os = "windows") {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join("Library").join("Application Support"))
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .map(|path| path.join(".local/share"))
                })
        }
        .ok_or_else(|| {
            WorkflowError::Persistence("could not determine the user data directory".to_string())
        })?;
        Ok(Self::new(
            base.join("arko")
                .join("workflows")
                .join(workflow_name)
                .join(uuid::Uuid::new_v4().to_string()),
        ))
    }
}

impl PersistenceManager {
    /// Open an existing run identity or create one for a new run directory.
    pub fn resolve_run_id(config: &PersistenceConfig) -> Result<uuid::Uuid, WorkflowError> {
        std::fs::create_dir_all(&config.checkpoint_dir)?;
        let path = config.checkpoint_dir.join("workflow-run-id");
        match std::fs::read_to_string(&path) {
            Ok(value) => uuid::Uuid::parse_str(value.trim())
                .map_err(|error| WorkflowError::Persistence(error.to_string())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let run_id = uuid::Uuid::new_v4();
                let mut options = std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                use std::io::Write;
                let mut file = options.open(&path)?;
                file.write_all(run_id.to_string().as_bytes())?;
                file.sync_data()?;
                Ok(run_id)
            }
            Err(error) => Err(error.into()),
        }
    }

    pub fn new(config: PersistenceConfig, workflow_run_id: uuid::Uuid) -> Self {
        let durable_store = std::sync::Arc::new(crate::durable::DurableStore::new(
            workflow_run_id,
            config.checkpoint_dir.clone(),
        ));
        Self {
            config,
            durable_store,
            write_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn durable_store(&self) -> std::sync::Arc<crate::durable::DurableStore> {
        self.durable_store.clone()
    }

    /// Save a checkpoint of the current workflow state.
    pub async fn save_checkpoint(&self, ctx: &SharedContext) -> Result<PathBuf, WorkflowError> {
        let _guard = self.write_lock.lock().await;
        let ws = state::WorkflowState {
            workflow_run_id: ctx.workflow_run_id,
            workflow_name: ctx.workflow_name.clone(),
            workflow_metadata: ctx.workflow_metadata.clone(),
            checkpoint_id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            memory_snapshot: ctx.memory.snapshot(),
            message_queues: ctx.messages.snapshot(),
            budget_spent: ctx.budget.total_spent(),
            cost_ledger: ctx.budget.snapshot_ledger(),
            freshness_snapshot: ctx
                .resource_tracker
                .as_ref()
                .map(|tracker| tracker.snapshot()),
            agent_results: ctx.agent_results.read().clone(),
            verification_records: ctx.verification_records.read().clone(),
            distillation_records: ctx.distillation_records.read().clone(),
            completed_agents: ctx
                .agent_results
                .read()
                .iter()
                .map(|r| state::SerializableAgentResult {
                    agent_name: r.agent_name.clone(),
                    summary: r.summary.clone(),
                })
                .collect(),
            cancellation: ctx.run_control.cancellation(),
            agent_registry: Some(ctx.registry.snapshot()),
        };

        std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        let path = self
            .config
            .checkpoint_dir
            .join(format!("checkpoint-{}.json", ws.checkpoint_id));
        let json = serde_json::to_string_pretty(&ws)?;
        let temporary = path.with_extension("json.tmp");
        tokio::fs::write(&temporary, json).await?;
        tokio::fs::File::open(&temporary).await?.sync_all().await?;
        tokio::fs::rename(&temporary, &path).await?;
        let latest = self.config.checkpoint_dir.join("latest-checkpoint");
        let latest_temporary = self.config.checkpoint_dir.join("latest-checkpoint.tmp");
        tokio::fs::write(
            &latest_temporary,
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default(),
        )
        .await?;
        tokio::fs::rename(&latest_temporary, &latest).await?;
        log::info!("Checkpoint saved to {}", path.display());
        Ok(path)
    }

    pub async fn append_event(
        &self,
        agent_name: &str,
        agent_run_id: uuid::Uuid,
        event: &arko_agents::runtime::event::AgentEvent,
    ) -> Result<(), WorkflowError> {
        #[derive(serde::Serialize)]
        struct PersistedEvent<'a> {
            timestamp: chrono::DateTime<chrono::Utc>,
            workflow_run_id: uuid::Uuid,
            agent_name: &'a str,
            agent_run_id: uuid::Uuid,
            event: &'a arko_agents::runtime::event::AgentEvent,
        }

        let _guard = self.write_lock.lock().await;
        tokio::fs::create_dir_all(&self.config.checkpoint_dir).await?;
        let path = self.config.checkpoint_dir.join("events.jsonl");
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        let mut line = serde_json::to_vec(&PersistedEvent {
            timestamp: chrono::Utc::now(),
            workflow_run_id: self.durable_store.workflow_run_id(),
            agent_name,
            agent_run_id,
            event,
        })?;
        line.push(b'\n');
        use tokio::io::AsyncWriteExt;
        file.write_all(&line).await?;
        Ok(())
    }

    pub async fn save_agent_task(
        &self,
        task: &arko_agents::Task,
    ) -> Result<PathBuf, WorkflowError> {
        let _guard = self.write_lock.lock().await;
        let directory = self.config.checkpoint_dir.join("agents");
        tokio::fs::create_dir_all(&directory).await?;
        let path = directory.join(format!("{}.json", task.id));
        let temporary = directory.join(format!("{}.json.tmp", task.id));
        tokio::fs::write(&temporary, serde_json::to_vec_pretty(task)?).await?;
        tokio::fs::File::open(&temporary).await?.sync_all().await?;
        tokio::fs::rename(&temporary, &path).await?;
        Ok(path)
    }

    /// Auto-checkpoint if configured.
    pub async fn auto_checkpoint(&self, ctx: &SharedContext) -> Result<(), WorkflowError> {
        if self.config.auto_checkpoint {
            self.save_checkpoint(ctx).await?;
        }
        Ok(())
    }

    /// Load a checkpoint from disk.
    pub async fn load_checkpoint(
        path: &std::path::Path,
    ) -> Result<state::WorkflowState, WorkflowError> {
        let content = tokio::fs::read_to_string(path).await?;
        let ws: state::WorkflowState = serde_json::from_str(&content)?;
        Ok(ws)
    }
}
