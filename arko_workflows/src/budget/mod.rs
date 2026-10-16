/// Budget tracking and cost ledger for workflows.
///
/// Tracks accumulated cost across all agents and manual additions,
/// maintaining both a lock-free running total (for fast threshold checks)
/// and a detailed serializable ledger (for persistence and analysis).
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

mod plan;

pub use plan::{
    PlanBudget, PlanBudgetCrossing, PlanBudgetState, PlanLimits, PlanUsageSpent, PlanUsageTracking,
    PlanWindow, PlanWindowUsage,
};

/// Scale factor: store cost as integer micro-dollars for lock-free atomics.
const COST_SCALE: f64 = 1_000_000.0;

/// The stored limit that stands for "no limit".
const NO_LIMIT: u64 = u64::MAX;

/// A limit in dollars, scaled; `None` is [`NO_LIMIT`].
fn scaled_limit(limit: Option<f64>) -> u64 {
    match limit {
        Some(limit) => ((limit.max(0.0) * COST_SCALE) as u64).min(NO_LIMIT - 1),
        None => NO_LIMIT,
    }
}

/// What a workflow does when its budget, or an agent's
/// ([`WorkflowBuilder::agent_budget`](crate::WorkflowBuilder::agent_budget)),
/// is exceeded, or a [`PlanBudget`] window's limit. Set with
/// [`WorkflowBuilder::budget_exceeded_policy`](crate::WorkflowBuilder::budget_exceeded_policy).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetExceededPolicy {
    /// The default: nothing is cancelled. Each agent over the budget is
    /// held before its next model request, paused for error recovery with
    /// [`ErrorRecoveryCause::BudgetExceeded`](arko_agents::ErrorRecoveryCause::BudgetExceeded)
    /// (tool calls it is running finish first). A run inside another
    /// agent's tool call waits until the budget is raised
    /// ([`Workflow::set_budget`](crate::Workflow::set_budget),
    /// [`Workflow::set_agent_budget`](crate::Workflow::set_agent_budget));
    /// a run the host started returns
    /// [`WorkflowError::AgentPaused`](crate::WorkflowError::AgentPaused), to
    /// be continued with [`Workflow::resume_agent`](crate::Workflow::resume_agent)
    /// once the budget is raised, or waits too with
    /// [`WorkflowBuilder::wait_for_budget_raise`](crate::WorkflowBuilder::wait_for_budget_raise).
    /// New runs start and are held at their first request. Spending can go
    /// past the limit by the requests in flight when it is reached.
    #[default]
    Pause,
    /// Each agent over the budget is cancelled (its run is recorded
    /// `Cancelled`; [`Workflow::reopen_agent`](crate::Workflow::reopen_agent)
    /// continues it), and later runs are refused while the budget stays
    /// exceeded.
    Cancel,
}

/// Identifies the origin of a cost entry in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostSource {
    /// Cost incurred by an agent's LLM call.
    Agent {
        agent_name: String,
        usage: Box<arko_ai::UsageInfo>,
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
    /// True if this addition exceeded the limit: the first addition over it
    /// since the budget was last within it (a raise can bring it back).
    pub exceeded_crossed: bool,
}

/// One agent's running cost totals; see [`BudgetManager::totals_by_agent`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CostTotals {
    /// What the account was billed, in dollars (calls a plan covered count
    /// zero).
    pub billed: f64,
    /// What the calls would have cost at list prices: the billed amount of
    /// metered calls, and the list-price estimate of those a plan covered
    /// (Claude Code, the ChatGPT login).
    pub list_price: f64,
    /// Input tokens, cached input included.
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// The model calls counted.
    pub calls: usize,
}

impl CostTotals {
    fn add(&mut self, billed: f64, usage: &arko_ai::UsageInfo) {
        self.billed += billed;
        self.list_price += if usage.billing == arko_ai::BillingKind::Subscription {
            usage.total_cost
        } else {
            billed
        };
        self.input_tokens += usage.input_tokens;
        self.output_tokens += usage.output_tokens;
        self.calls += 1;
    }
}

/// Tracks accumulated cost across all agents in a workflow.
///
/// Uses a dual strategy: an atomic counter for lock-free budget threshold
/// checks on the hot path, and a `Mutex`-protected [`CostLedger`] for
/// detailed per-entry recording.
pub struct BudgetManager {
    spent: AtomicU64,
    /// Each agent's running totals ([`totals_by_agent`](Self::totals_by_agent)).
    agent_totals: Mutex<BTreeMap<String, CostTotals>>,
    /// The limit, scaled ([`NO_LIMIT`]: none); it can change while agents
    /// run ([`set_limit`](Self::set_limit)).
    limit: AtomicU64,
    warning: Option<u64>,
    warned: AtomicBool,
    /// The spend has been reported over the limit since it was last within.
    exceeded_reported: AtomicBool,
    ledger: Mutex<CostLedger>,
    agent_limits: RwLock<HashMap<String, u64>>,
    agent_spent: Mutex<HashMap<String, u64>>,
    policy: BudgetExceededPolicy,
    wait_for_raise: bool,
    /// The plan budget: its limits and the plan usage the run has seen.
    plan: Mutex<plan::PlanBudgetTracker>,
    /// Counts the changes of the limits.
    changes: tokio::sync::watch::Sender<u64>,
}

impl BudgetManager {
    /// Creates a new budget manager with optional limit and warning threshold (in dollars).
    pub fn new(limit: Option<f64>, warning: Option<f64>) -> Self {
        Self {
            spent: AtomicU64::new(0),
            agent_totals: Mutex::default(),
            limit: AtomicU64::new(scaled_limit(limit)),
            warning: warning.map(|w| (w * COST_SCALE) as u64),
            warned: AtomicBool::new(false),
            exceeded_reported: AtomicBool::new(false),
            ledger: Mutex::new(CostLedger::default()),
            agent_limits: RwLock::new(HashMap::new()),
            agent_spent: Mutex::new(HashMap::new()),
            policy: BudgetExceededPolicy::default(),
            wait_for_raise: false,
            plan: Mutex::default(),
            changes: tokio::sync::watch::Sender::new(0),
        }
    }

    /// Sets what happens when the budget is exceeded (see
    /// [`BudgetExceededPolicy`]); with `wait_for_raise`, runs the host
    /// started wait for a raise under `Pause` too.
    pub fn with_exceeded_policy(
        mut self,
        policy: BudgetExceededPolicy,
        wait_for_raise: bool,
    ) -> Self {
        self.policy = policy;
        self.wait_for_raise = wait_for_raise;
        self
    }

    /// What happens when the budget is exceeded.
    pub fn exceeded_policy(&self) -> BudgetExceededPolicy {
        self.policy
    }

    /// Whether runs the host started wait for a raise when held under
    /// [`BudgetExceededPolicy::Pause`], instead of returning `AgentPaused`.
    pub fn waits_for_raise(&self) -> bool {
        self.wait_for_raise
    }

    pub fn from_ledger(limit: Option<f64>, warning: Option<f64>, ledger: CostLedger) -> Self {
        let total = ledger.total().max(0.0);
        let warning_scaled = warning.map(|value| (value * COST_SCALE) as u64);
        let spent = (total * COST_SCALE) as u64;
        let mut agent_spent = HashMap::new();
        let mut agent_totals = BTreeMap::new();
        for entry in &ledger.entries {
            if let CostSource::Agent { agent_name, usage } = &entry.source {
                let spent = agent_spent.entry(agent_name.clone()).or_insert(0_u64);
                *spent = spent.saturating_add((entry.amount * COST_SCALE) as u64);
                agent_totals
                    .entry(agent_name.clone())
                    .or_insert_with(CostTotals::default)
                    .add(entry.amount, usage);
            }
        }
        let limit = scaled_limit(limit);
        Self {
            spent: AtomicU64::new(spent),
            agent_totals: Mutex::new(agent_totals),
            limit: AtomicU64::new(limit),
            warning: warning_scaled,
            warned: AtomicBool::new(warning_scaled.is_some_and(|value| spent >= value)),
            exceeded_reported: AtomicBool::new(spent > limit),
            ledger: Mutex::new(ledger),
            agent_limits: RwLock::new(HashMap::new()),
            agent_spent: Mutex::new(agent_spent),
            policy: BudgetExceededPolicy::default(),
            wait_for_raise: false,
            plan: Mutex::default(),
            changes: tokio::sync::watch::Sender::new(0),
        }
    }

    /// Record an agent's LLM usage cost in the ledger and update the running total.
    ///
    /// The amount is what the account is billed ([`UsageInfo::billed`]), so
    /// subscription calls (ChatGPT OAuth) count as zero while their estimate
    /// stays in the stored [`UsageInfo`] for cost breakdown analysis
    /// (input/output tokens, cache costs, etc.).
    ///
    /// [`UsageInfo::billed`]: arko_ai::UsageInfo::billed
    /// [`UsageInfo`]: arko_ai::UsageInfo
    pub fn add_agent_cost(&self, agent_name: &str, usage: &arko_ai::UsageInfo) -> AddCostResult {
        let amount = usage.billed();
        self.agent_totals
            .lock()
            .entry(agent_name.to_string())
            .or_default()
            .add(amount, usage);
        let entry = CostEntry {
            amount,
            source: CostSource::Agent {
                agent_name: agent_name.to_string(),
                usage: Box::new(usage.clone()),
            },
            timestamp: Utc::now(),
            metadata: None,
        };
        let result = self.record_entry(entry);
        let scaled = (amount * COST_SCALE) as u64;
        let mut spent = self.agent_spent.lock();
        let current = spent.entry(agent_name.to_string()).or_default();
        *current = current.saturating_add(scaled);
        result
    }

    /// Sets `agent_name`'s own limit, in dollars; it applies from the
    /// agent's next model request. Raising it releases the agent's runs
    /// held on it.
    pub fn set_agent_limit(&self, agent_name: impl Into<String>, limit: f64) {
        self.agent_limits
            .write()
            .insert(agent_name.into(), scaled_limit(Some(limit)));
        self.changed();
    }

    /// Removes `agent_name`'s own limit, releasing its runs held on it.
    pub fn clear_agent_limit(&self, agent_name: &str) {
        self.agent_limits.write().remove(agent_name);
        self.changed();
    }

    /// The agents' own limits, in dollars.
    pub fn agent_limits(&self) -> HashMap<String, f64> {
        self.agent_limits
            .read()
            .iter()
            .map(|(agent, limit)| (agent.clone(), *limit as f64 / COST_SCALE))
            .collect()
    }

    /// Sets the workflow's limit, in dollars (`None`: no limit), while
    /// agents run: it applies from their next model requests, and raising
    /// it releases the runs held on it. `OnBudgetExceeded` fires again when
    /// a later charge exceeds a raised limit.
    pub fn set_limit(&self, limit: Option<f64>) {
        self.limit.store(scaled_limit(limit), Ordering::Relaxed);
        if !self.is_exceeded() {
            self.exceeded_reported.store(false, Ordering::Relaxed);
        }
        self.changed();
    }

    /// Wakes the runs waiting for a raise.
    fn changed(&self) {
        self.changes
            .send_modify(|count| *count = count.wrapping_add(1));
    }

    /// A receiver that sees every change of the limits.
    pub(crate) fn subscribe(&self) -> tokio::sync::watch::Receiver<u64> {
        self.changes.subscribe()
    }

    /// Why a run of `agent_name` is held before its next model request
    /// under [`BudgetExceededPolicy::Pause`]: the workflow's budget or the
    /// agent's is exceeded. `None` under `Cancel`, or within budget.
    pub fn hold_reason(&self, agent_name: &str) -> Option<String> {
        if self.policy != BudgetExceededPolicy::Pause {
            return None;
        }
        if self.is_exceeded() {
            return Some(format!(
                "Budget exceeded: spent ${:.4} of the workflow's ${:.4} limit. The request waits until the budget is raised.",
                self.total_spent(),
                self.limit().unwrap_or(0.0)
            ));
        }
        if self.is_agent_exceeded(agent_name) {
            return Some(format!(
                "Agent budget exceeded: agent '{agent_name}' spent ${:.4} of its ${:.4} limit. The request waits until the budget is raised.",
                self.agent_spent(agent_name),
                self.agent_limit(agent_name).unwrap_or(0.0)
            ));
        }
        if let Some(crossing) = self.plan_exceeded() {
            return Some(format!(
                "Plan budget exceeded: {} used since the run started. The request waits until the plan budget is raised.",
                crossing.describe()
            ));
        }
        None
    }

    /// Takes a plan usage report (from any agent: plan usage is
    /// account-wide), counted with the windows of the plan it names;
    /// returns the windows it took over their plan budget limits, each once
    /// per crossing.
    pub fn record_plan_usage(&self, usage: &arko_ai::PlanUsage) -> Vec<PlanBudgetCrossing> {
        let now = Utc::now().timestamp().max(0) as u64;
        self.plan.lock().observe(usage, now)
    }

    /// Sets the plan budget's limits, while agents run too: raising them
    /// releases the runs held on them.
    pub fn set_plan_limits(&self, limits: PlanBudget) {
        self.plan.lock().set_limits(limits);
        self.changed();
    }

    /// The plan budget's limits.
    pub fn plan_limits(&self) -> PlanBudget {
        self.plan.lock().limits
    }

    /// The points of each plan window the run has used: the most any plan
    /// has used.
    pub fn plan_spent(&self) -> PlanUsageSpent {
        self.plan.lock().spent()
    }

    /// The points of each of `plan`'s windows the run has used.
    pub fn plan_spent_of(&self, plan: arko_ai::SubscriptionPlan) -> PlanUsageSpent {
        self.plan.lock().spent_of(plan)
    }

    /// A window over its plan budget limit, if any.
    pub fn plan_exceeded(&self) -> Option<PlanBudgetCrossing> {
        self.plan.lock().exceeded()
    }

    /// The plan budget's limits and tracking, for a checkpoint.
    pub fn plan_state(&self) -> PlanBudgetState {
        self.plan.lock().state()
    }

    /// Restores the plan usage tracking a checkpoint recorded (the limits
    /// are set with [`set_plan_limits`](Self::set_plan_limits)).
    pub(crate) fn restore_plan_usage(&self, state: PlanBudgetState) {
        self.plan.lock().restore(state.usage, state.plans);
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

        let exceeded = self.is_exceeded();
        AddCostResult {
            new_total,
            warning_triggered: self.check_warning(),
            exceeded,
            exceeded_crossed: exceeded && !self.exceeded_reported.swap(true, Ordering::Relaxed),
        }
    }

    /// Returns total spent so far in dollars.
    pub fn total_spent(&self) -> f64 {
        self.spent.load(Ordering::Relaxed) as f64 / COST_SCALE
    }

    /// Returns true if the budget limit has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        let limit = self.limit.load(Ordering::Relaxed);
        limit != NO_LIMIT && self.spent.load(Ordering::Relaxed) > limit
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
        let limit = self.limit.load(Ordering::Relaxed);
        (limit != NO_LIMIT).then(|| {
            let spent = self.spent.load(Ordering::Relaxed);
            limit.saturating_sub(spent) as f64 / COST_SCALE
        })
    }

    /// Returns the budget limit in dollars, if set.
    pub fn limit(&self) -> Option<f64> {
        let limit = self.limit.load(Ordering::Relaxed);
        (limit != NO_LIMIT).then(|| limit as f64 / COST_SCALE)
    }

    /// Each agent's running totals: what its model calls were billed, what
    /// they would have cost at list prices, their tokens and their count.
    /// Kept as costs are added (and rebuilt from the ledger on resume), so a
    /// host reading them often doesn't copy the whole ledger. Manual costs
    /// belong to no agent and aren't in them.
    pub fn totals_by_agent(&self) -> BTreeMap<String, CostTotals> {
        self.agent_totals.lock().clone()
    }

    /// Returns a snapshot of the cost ledger for persistence or analysis.
    pub fn snapshot_ledger(&self) -> CostLedger {
        self.ledger.lock().clone()
    }

    /// How many entries the ledger holds.
    pub(crate) fn ledger_len(&self) -> usize {
        self.ledger.lock().entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arko_ai::{BillingKind, Model, UsageInfo};

    #[test]
    fn the_limit_changes_live_and_each_crossing_is_reported_once() {
        let budget = BudgetManager::new(Some(1.0), None);
        let cost = |amount: f64| budget.add_manual_cost(amount, "test", None);
        let first = cost(1.5);
        assert!(first.exceeded && first.exceeded_crossed);
        let second = cost(0.1);
        assert!(second.exceeded && !second.exceeded_crossed);
        assert!(budget.hold_reason("coder").is_some());

        let changes = budget.subscribe();
        budget.set_limit(Some(10.0));
        assert!(changes.has_changed().unwrap());
        assert!(!budget.is_exceeded());
        assert!(budget.hold_reason("coder").is_none());
        assert_eq!(budget.remaining(), Some(8.4));
        budget.set_limit(Some(2.0));
        assert!(cost(0.5).exceeded_crossed, "a new crossing");
        budget.set_limit(None);
        assert_eq!(budget.limit(), None);
        assert!(!cost(100.0).exceeded);

        budget.set_agent_limit("coder", 0.0);
        let agent = || arko_ai::UsageInfo {
            billed_cost: Some(0.5),
            ..arko_ai::UsageInfo::new(1, 1)
        };
        budget.add_agent_cost("coder", &agent());
        assert!(
            budget
                .hold_reason("coder")
                .unwrap()
                .contains("agent 'coder'")
        );
        assert!(budget.hold_reason("reviewer").is_none());
        budget.clear_agent_limit("coder");
        assert!(budget.hold_reason("coder").is_none());

        let cancel = BudgetManager::new(Some(0.0), None)
            .with_exceeded_policy(BudgetExceededPolicy::Cancel, false);
        cancel.add_manual_cost(1.0, "test", None);
        assert!(cancel.is_exceeded());
        assert!(cancel.hold_reason("coder").is_none(), "cancelled, not held");
    }

    /// Each agent's totals add its calls up as they come: billed amounts, list
    /// prices (a plan-covered call's estimate), tokens and calls; manual
    /// costs aren't an agent's. A resumed budget rebuilds them from its
    /// ledger.
    #[test]
    fn each_agent_s_totals_are_kept_as_costs_come() {
        let budget = BudgetManager::new(None, None);
        let metered = UsageInfo {
            billed_cost: Some(0.5),
            ..UsageInfo::new(1000, 100)
        };
        let covered = UsageInfo {
            total_cost: 0.75,
            billed_cost: Some(0.0),
            billing: BillingKind::Subscription,
            ..UsageInfo::new(2000, 200)
        };
        budget.add_agent_cost("main", &metered);
        budget.add_agent_cost("main", &covered);
        budget.add_agent_cost("worker", &metered);
        budget.add_manual_cost(3.0, "embeddings", None);
        let totals = budget.totals_by_agent();
        assert_eq!(
            totals["main"],
            CostTotals {
                billed: 0.5,
                list_price: 1.25,
                input_tokens: 3000,
                output_tokens: 300,
                calls: 2,
            }
        );
        assert_eq!(totals["worker"].calls, 1);
        assert_eq!(totals.len(), 2);
        let resumed = BudgetManager::from_ledger(None, None, budget.snapshot_ledger());
        assert_eq!(resumed.totals_by_agent(), totals);
    }

    #[test]
    fn agent_costs_count_the_billed_amount_and_keep_the_estimate() {
        let budget = BudgetManager::new(Some(1.0), None);
        let model = Model::openai_gpt_6_luna();
        let mut subscription =
            UsageInfo::new(1_000_000, 1_000_000).with_billing(BillingKind::Subscription);
        subscription.calculate_cost(&model, None);
        let mut metered = UsageInfo::new(1_000_000, 0);
        metered.calculate_cost(&model, None);

        assert_eq!(budget.add_agent_cost("oauth", &subscription).new_total, 0.0);
        assert_eq!(budget.agent_spent("oauth"), 0.0);
        let result = budget.add_agent_cost("api", &metered);
        assert!((result.new_total - metered.total_cost).abs() < 1e-6);

        let ledger = budget.snapshot_ledger();
        let CostSource::Agent { usage, .. } = &ledger.entries[0].source else {
            panic!("expected an agent entry");
        };
        assert!(usage.total_cost > 0.0, "the subscription estimate is kept");
        assert_eq!(ledger.entries[0].amount, 0.0);
    }

    /// Subscription calls (ChatGPT OAuth, Claude Code unless billed per
    /// token) never count toward a dollar budget, the workflow's or an
    /// agent's, whatever their estimate.
    #[test]
    fn a_subscription_call_never_counts_toward_the_dollar_budget() {
        let budget = BudgetManager::new(Some(0.0), None);
        budget.set_agent_limit("coder", 0.0);
        let mut subscription =
            UsageInfo::new(10_000_000, 1_000_000).with_billing(BillingKind::Subscription);
        subscription.calculate_cost(&Model::openai_gpt_6_luna(), None);
        assert!(subscription.total_cost > 0.0);
        let result = budget.add_agent_cost("coder", &subscription);
        assert_eq!(result.new_total, 0.0);
        assert!(!result.exceeded);
        assert!(!budget.is_exceeded() && !budget.is_agent_exceeded("coder"));
        assert!(budget.hold_reason("coder").is_none());
        assert_eq!(budget.total_spent(), 0.0);
        assert_eq!(budget.snapshot_ledger().total(), 0.0);
    }
}
