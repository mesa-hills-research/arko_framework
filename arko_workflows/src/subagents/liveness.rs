//! Whether an agent's workers run, for the terminal manager's monitors that
//! last while they do.

use std::sync::{Arc, Weak};

use tokio::sync::watch;
use uuid::Uuid;

use crate::context::SharedContext;

/// Says from the workflow's board whether the runs an agent started are
/// queued or running. `WorkflowBuilder::build` sets it on the manager given
/// to `background_jobs` when the workflow has subagents. It holds the
/// workflow weakly, so once the workflow is dropped no worker runs.
pub(crate) struct WorkerRuns(Weak<SharedContext>);

impl WorkerRuns {
    pub(crate) fn of(context: &Arc<SharedContext>) -> Self {
        Self(Arc::downgrade(context))
    }
}

impl arko_agents::terminal::WorkerLiveness for WorkerRuns {
    fn has_running_workers(&self, owner: Uuid) -> bool {
        self.0
            .upgrade()
            .is_some_and(|context| context.subagents.has_active_workers(owner))
    }

    fn changes(&self) -> watch::Receiver<u64> {
        match self.0.upgrade() {
            Some(context) => context.subagents.changes(),
            // Closed, as no worker runs again.
            None => watch::channel(0).1,
        }
    }
}
