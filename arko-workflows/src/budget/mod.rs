/// Budget tracking and cost ledger for workflows.
///
/// Tracks accumulated cost across all agents and manual additions,
/// maintaining both a lock-free running total (for fast threshold checks)
/// and a detailed serializable ledger (for persistence and analysis).
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

/// Scale factor: store cost as integer micro-dollars for lock-free atomics.
const COST_SCALE: f64 = 1_000_000.0;

/// Identifies the origin of a cost entry in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostSource {
    /// Cost incurred by an agent's LLM call.
    Agent {
        agent_name: String,
        usage: arko_ai::UsageInfo,
    },
    /// Cost added manually by the workflow operator.
    Manual { description: String },
}

/// A single cost record in the workflow's cost ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub amount: f64,
    pub source: CostSource,
    pub timestamp: DateTime<Utc>,
    /// Free-form metadata for accounting purposes.
    ///
    /// Intended primarily for manual cost entries where the caller may
    /// want to record the originating service, URL, API call body or
    /// response, or any other context relevant for cost analysis. Not
    /// constrained to a specific schema so different services can use
    /// whatever format suits their needs.
    pub metadata: Option<String>,
}

/// The complete cost history for a workflow run.
///
/// Contains all cost entries — both automatic agent costs and manual
/// additions — in chronological order. Serializable for checkpoint
/// persistence and post-run analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostLedger {
    pub entries: Vec<CostEntry>,
}

impl CostLedger {
    /// Returns the sum of all entry amounts in dollars.
    pub fn total(&self) -> f64 {
        self.entries.iter().map(|e| e.amount).sum()
    }
}

/// Returned after recording a cost, indicating whether budget
/// thresholds were crossed by this addition.
#[derive(Debug, Clone)]
pub struct AddCostResult {
    /// The new running total in dollars after this addition.
    pub new_total: f64,
    /// True if this addition caused the warning threshold to be
    /// crossed for the first time.
    pub warning_triggered: bool,
    /// True if the budget limit has been exceeded.
    pub exceeded: bool,
}

/// Tracks accumulated cost across all agents in a workflow.
///
/// Uses a dual strategy: an atomic counter for lock-free budget threshold
/// checks on the hot path, and a `Mutex`-protected [`CostLedger`] for
/// detailed per-entry recording.
pub struct BudgetManager {
    spent: AtomicU64,
    limit: Option<u64>,
    warning: Option<u64>,
    warned: AtomicBool,
    oauth_exempt: RwLock<HashSet<String>>,
    ledger: Mutex<CostLedger>,
    agent_limits: RwLock<HashMap<String, u64>>,
    agent_spent: Mutex<HashMap<String, u64>>,
}

impl BudgetManager {
    /// Creates a new budget manager with optional limit and warning threshold (in dollars).
    pub fn new(limit: Option<f64>, warning: Option<f64>) -> Self {
        Self {
            spent: AtomicU64::new(0),
            limit: limit.map(|l| (l * COST_SCALE) as u64),
            warning: warning.map(|w| (w * COST_SCALE) as u64),
            warned: AtomicBool::new(false),
            oauth_exempt: RwLock::new(HashSet::new()),
            ledger: Mutex::new(CostLedger::default()),
            agent_limits: RwLock::new(HashMap::new()),
            agent_spent: Mutex::new(HashMap::new()),
        }
    }

    pub fn from_ledger(
        limit: Option<f64>,
        warning: Option<f64>,
        ledger: CostLedger,
        oauth_exempt: HashSet<String>,
    ) -> Self {
        let total = ledger.total().max(0.0);
        let warning_scaled = warning.map(|value| (value * COST_SCALE) as u64);
        let spent = (total * COST_SCALE) as u64;
        let mut agent_spent = HashMap::new();
        for entry in &ledger.entries {
            if let CostSource::Agent { agent_name, .. } = &entry.source {
                let spent = agent_spent.entry(agent_name.clone()).or_insert(0_u64);
                *spent = spent.saturating_add((entry.amount * COST_SCALE) as u64);
            }
        }
        Self {
            spent: AtomicU64::new(spent),
            limit: limit.map(|value| (value * COST_SCALE) as u64),
            warning: warning_scaled,
            warned: AtomicBool::new(warning_scaled.is_some_and(|value| spent >= value)),
            oauth_exempt: RwLock::new(oauth_exempt),
            ledger: Mutex::new(ledger),
            agent_limits: RwLock::new(HashMap::new()),
            agent_spent: Mutex::new(agent_spent),
        }
    }

    pub fn oauth_exempt_snapshot(&self) -> HashSet<String> {
        self.oauth_exempt.read().clone()
    }

    /// Mark an agent as OAuth-exempt (not billed per token).
    pub fn mark_oauth_exempt(&self, agent_name: &str) {
        self.oauth_exempt.write().insert(agent_name.to_string());
    }

    /// Returns true if the agent is OAuth-exempt.
    pub fn is_oauth_exempt(&self, agent_name: &str) -> bool {
        self.oauth_exempt.read().contains(agent_name)
    }

    /// Record an agent's LLM usage cost in the ledger and update the running total.
    ///
    /// The full [`UsageInfo`](arko_ai::UsageInfo) is stored in the ledger entry
    /// for detailed cost breakdown analysis (input/output tokens, cache costs, etc.).
    pub fn add_agent_cost(&self, agent_name: &str, usage: &arko_ai::UsageInfo) -> AddCostResult {
        let entry = CostEntry {
            amount: usage.total_cost,
            source: CostSource::Agent {
                agent_name: agent_name.to_string(),
                usage: usage.clone(),
            },
            timestamp: Utc::now(),
            metadata: None,
        };
        let result = self.record_entry(entry);
        let scaled = (usage.total_cost * COST_SCALE) as u64;
        let mut spent = self.agent_spent.lock();
        let current = spent.entry(agent_name.to_string()).or_default();
        *current = current.saturating_add(scaled);
        result
    }

    pub fn set_agent_limit(&self, agent_name: impl Into<String>, limit: f64) {
        self.agent_limits
            .write()
            .insert(agent_name.into(), (limit * COST_SCALE) as u64);
    }

    pub fn agent_limit(&self, agent_name: &str) -> Option<f64> {
        self.agent_limits
            .read()
            .get(agent_name)
            .map(|value| *value as f64 / COST_SCALE)
    }

    pub fn agent_spent(&self, agent_name: &str) -> f64 {
        self.agent_spent
            .lock()
            .get(agent_name)
            .copied()
            .unwrap_or(0) as f64
            / COST_SCALE
    }

    pub fn is_agent_exceeded(&self, agent_name: &str) -> bool {
        self.agent_limits
            .read()
            .get(agent_name)
            .is_some_and(|limit| {
                self.agent_spent
                    .lock()
                    .get(agent_name)
                    .copied()
                    .unwrap_or(0)
                    > *limit
            })
    }

    /// Record a manual cost in the ledger and update the running total.
    ///
    /// Manual costs participate in budget threshold checks identically
    /// to agent costs. The optional `metadata` field accepts a free-form
    /// string for recording source details, API call context, or any
    /// other accounting information without imposing a fixed schema.
    pub fn add_manual_cost(
        &self,
        amount: f64,
        description: &str,
        metadata: Option<String>,
    ) -> AddCostResult {
        let entry = CostEntry {
            amount,
            source: CostSource::Manual {
                description: description.to_string(),
            },
            timestamp: Utc::now(),
            metadata,
        };
        self.record_entry(entry)
    }

    fn record_entry(&self, entry: CostEntry) -> AddCostResult {
        let amount = entry.amount;
        self.ledger.lock().entries.push(entry);

        let scaled = (amount * COST_SCALE) as u64;
        let new_total_scaled = self.spent.fetch_add(scaled, Ordering::Relaxed) + scaled;
        let new_total = new_total_scaled as f64 / COST_SCALE;

        AddCostResult {
            new_total,
            warning_triggered: self.check_warning(),
            exceeded: self.is_exceeded(),
        }
    }

    /// Returns total spent so far in dollars.
    pub fn total_spent(&self) -> f64 {
        self.spent.load(Ordering::Relaxed) as f64 / COST_SCALE
    }

    /// Returns true if the budget limit has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.limit
            .is_some_and(|limit| self.spent.load(Ordering::Relaxed) > limit)
    }

    /// Returns true the first time the warning threshold is crossed.
    pub fn check_warning(&self) -> bool {
        if let Some(warning) = self.warning
            && self.spent.load(Ordering::Relaxed) >= warning
            && !self.warned.swap(true, Ordering::Relaxed)
        {
            return true;
        }
        false
    }

    /// Returns the remaining budget in dollars, if a limit is set.
    pub fn remaining(&self) -> Option<f64> {
        self.limit.map(|limit| {
            let spent = self.spent.load(Ordering::Relaxed);
            (limit.saturating_sub(spent)) as f64 / COST_SCALE
        })
    }

    /// Returns the budget limit in dollars, if set.
    pub fn limit(&self) -> Option<f64> {
        self.limit.map(|l| l as f64 / COST_SCALE)
    }

    /// Returns a snapshot of the cost ledger for persistence or analysis.
    pub fn snapshot_ledger(&self) -> CostLedger {
        self.ledger.lock().clone()
    }
}
