//! Per-job provider usage and cost.
use arko_ai::UsageInfo;
use serde::{Deserialize, Serialize};

/// Token usage and cost of one job's provider calls: chunk-model replies
/// (initial requests and repairs) and embedding requests, summed across
/// retries, pauses, resumes and restarts.
/// [`JobStatus::usage`](crate::JobStatus::usage) reports it. Each usage is
/// saved in the same transaction as the one event that carries it, so the
/// events add up to these totals: `usage` of `request_finished` (accepted
/// chunk replies), `event.detail.usage` of failed `request_attempt` events
/// (rejected chunk responses, saved before any retry wait), and `usage` and
/// `rejected_usage` of `embedded`, `embedding_rejected` and
/// `embedding_failed` events.
///
/// A call counts when the provider returned a response: accepted replies,
/// embedding replies whose vectors the index then rejects, and responses the
/// backend rejected and reported with [`Error::with_usage`](crate::Error::with_usage)
/// (for [`ArkoBackend`](crate::backend::ArkoBackend): refusals, token limits,
/// a missing, extra or malformed tool call), including every retried attempt.
/// Failures without a response (connection errors, provider error statuses,
/// streams cut off mid-response) and calls cancelled mid-flight carry no
/// usage and are not included, although a provider may bill some of them.
/// Search query embeddings belong to no job.
///
/// Non-finite costs a custom backend reports are counted as zero. Totals
/// saved with a cost that isn't a number (a non-finite cost was once saved
/// as `null`) load with that cost as zero and every count kept; totals that
/// still can't be read are logged and start again from zero.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JobUsage {
    /// Chunk-model replies the backend accepted.
    pub chunk_calls: usize,
    /// Summed usage of the chunk-model responses that reported it, accepted
    /// and rejected.
    pub chunking: Option<UsageInfo>,
    /// Embedding replies the backend accepted.
    pub embedding_calls: usize,
    /// Summed usage of the embedding responses that reported it, accepted and
    /// rejected.
    pub embedding: Option<UsageInfo>,
    /// Accepted replies whose backend reported no usage; their tokens and
    /// cost are unknown and not included.
    pub unreported_calls: usize,
    /// Responses the backend rejected, with their usage, whether the request
    /// was then retried or failed. Their usage is in `chunking` or
    /// `embedding`; they are not in `chunk_calls` or `embedding_calls`.
    pub rejected_calls: usize,
}

impl JobUsage {
    /// Summed usage of every call that reported it; `None` when none did.
    pub fn total(&self) -> Option<UsageInfo> {
        if self.chunking.is_none() && self.embedding.is_none() {
            return None;
        }
        Some(self.chunking.iter().chain(&self.embedding).sum())
    }

    /// What the account was charged for the calls that reported usage (see
    /// [`UsageInfo::billed`]).
    pub fn billed(&self) -> f64 {
        [&self.chunking, &self.embedding]
            .into_iter()
            .flatten()
            .map(UsageInfo::billed)
            .sum()
    }

    pub(crate) fn record_chunk(&mut self, usage: Option<&UsageInfo>) {
        self.chunk_calls = self.chunk_calls.saturating_add(1);
        match usage {
            Some(usage) => add(&mut self.chunking, usage),
            None => self.unreported_calls = self.unreported_calls.saturating_add(1),
        }
    }

    pub(crate) fn record_embedding(&mut self, usage: Option<&UsageInfo>) {
        self.embedding_calls = self.embedding_calls.saturating_add(1);
        match usage {
            Some(usage) => add(&mut self.embedding, usage),
            None => self.unreported_calls = self.unreported_calls.saturating_add(1),
        }
    }

    pub(crate) fn record_rejected_chunk(&mut self, usage: &UsageInfo) {
        self.rejected_calls = self.rejected_calls.saturating_add(1);
        add(&mut self.chunking, usage);
    }

    pub(crate) fn record_rejected_embedding(&mut self, usage: &UsageInfo) {
        self.rejected_calls = self.rejected_calls.saturating_add(1);
        add(&mut self.embedding, usage);
    }
}

fn add(total: &mut Option<UsageInfo>, usage: &UsageInfo) {
    let usage = finite(usage);
    match total {
        Some(total) => {
            *total += &usage;
            *total = finite(total);
        }
        // Summing keeps the first call's billing kind (a default total would
        // turn it into Mixed) and its billed cost.
        None => *total = Some(std::iter::once(&usage).sum()),
    }
}

/// `usage` with every non-finite cost set to zero. JSON has no NaN or
/// infinity: serde_json writes them as `null`, which then fails to load.
pub(crate) fn finite(usage: &UsageInfo) -> UsageInfo {
    let zero_if_not_finite = |cost: &mut f64| {
        if !cost.is_finite() {
            *cost = 0.0;
        }
    };
    let mut usage = usage.clone();
    for cost in [
        &mut usage.input_cost,
        &mut usage.output_cost,
        &mut usage.total_cost,
    ] {
        zero_if_not_finite(cost);
    }
    for cost in [
        &mut usage.cache_read_cost,
        &mut usage.cache_write_cost,
        &mut usage.reported_cost,
        &mut usage.upstream_cost,
        &mut usage.billed_cost,
    ]
    .into_iter()
    .flatten()
    {
        zero_if_not_finite(cost);
    }
    usage
}

#[cfg(test)]
mod tests {
    use super::*;
    use arko_ai::BillingKind;
    use serde_json::json;

    #[test]
    fn totals_sum_reported_calls_and_count_the_rest() {
        let mut usage = JobUsage::default();
        assert_eq!(usage.total(), None);
        assert_eq!(usage.billed(), 0.0);
        let subscription = UsageInfo {
            total_cost: 0.5,
            ..UsageInfo::new(1_000, 200)
        }
        .with_billing(BillingKind::Subscription);
        usage.record_chunk(Some(&subscription));
        usage.record_chunk(Some(&subscription));
        usage.record_chunk(None);
        let chunking = usage.chunking.as_ref().unwrap();
        assert_eq!(
            (chunking.input_tokens, chunking.output_tokens),
            (2_000, 400)
        );
        assert_eq!(chunking.billing, BillingKind::Subscription);
        assert_eq!(chunking.total_cost, 1.0);
        assert_eq!(chunking.billed(), 0.0);
        let embedding = UsageInfo::for_embedding(1_000_000, 0.06);
        usage.record_embedding(Some(&embedding));
        assert_eq!(
            (
                usage.chunk_calls,
                usage.embedding_calls,
                usage.unreported_calls
            ),
            (3, 1, 1)
        );
        assert!((usage.billed() - 0.06).abs() < 1e-12);
        let total = usage.total().unwrap();
        assert_eq!(total.input_tokens, 1_002_000);
        assert_eq!(total.billing, BillingKind::Mixed);
        assert!((total.billed() - 0.06).abs() < 1e-12);
    }

    #[test]
    fn saved_totals_load_with_missing_fields() {
        let usage: JobUsage = serde_json::from_value(json!({})).unwrap();
        assert_eq!(usage, JobUsage::default());
        let usage: JobUsage = serde_json::from_value(json!({
            "chunk_calls": 2,
            "chunking": {"input_tokens": 10, "output_tokens": 5, "input_cost": 0.0,
                         "output_cost": 0.0, "total_cost": 0.25},
            "added_later": true
        }))
        .unwrap();
        assert_eq!(usage.chunk_calls, 2);
        assert_eq!(usage.rejected_calls, 0);
        // A record from before billing kinds existed bills its estimate.
        assert_eq!(usage.billed(), 0.25);
    }

    #[test]
    fn rejected_responses_add_to_the_totals_but_not_the_accepted_calls() {
        let mut usage = JobUsage::default();
        let call = UsageInfo::new(100, 10).with_reported_cost(0.5);
        usage.record_rejected_chunk(&call);
        usage.record_rejected_chunk(&call);
        usage.record_chunk(Some(&call));
        usage.record_rejected_embedding(&UsageInfo::for_embedding(1_000, 1.0));
        assert_eq!(
            (
                usage.chunk_calls,
                usage.embedding_calls,
                usage.rejected_calls
            ),
            (1, 0, 3)
        );
        assert_eq!(usage.chunking.as_ref().unwrap().input_tokens, 300);
        assert_eq!(usage.embedding.as_ref().unwrap().input_tokens, 1_000);
        assert!((usage.billed() - 1.501).abs() < 1e-12);
    }

    #[test]
    fn non_finite_costs_count_as_zero_and_totals_still_save_and_load() {
        let broken = UsageInfo {
            input_cost: f64::NAN,
            output_cost: f64::INFINITY,
            cache_read_cost: Some(f64::NEG_INFINITY),
            total_cost: f64::NAN,
            reported_cost: Some(f64::NAN),
            ..UsageInfo::new(100, 10)
        }
        .with_billing(BillingKind::ProviderReported);
        let mut usage = JobUsage::default();
        usage.record_chunk(Some(&broken));
        usage.record_rejected_chunk(&broken);
        usage.record_rejected_chunk(&UsageInfo {
            total_cost: f64::MAX,
            billed_cost: Some(f64::MAX),
            ..UsageInfo::new(1, 1)
        });
        usage.record_rejected_chunk(&UsageInfo {
            total_cost: f64::MAX,
            billed_cost: Some(f64::MAX),
            ..UsageInfo::new(1, 1)
        });
        let saved = serde_json::to_string(&usage).unwrap();
        let loaded: JobUsage = serde_json::from_str(&saved).unwrap();
        let chunking = loaded.chunking.as_ref().unwrap();
        assert_eq!(chunking.input_tokens, 202);
        assert_eq!(chunking.input_cost, 0.0);
        assert_eq!(chunking.cache_read_cost, Some(0.0));
        // Two finite costs that overflow when summed are zeroed too.
        assert!(chunking.total_cost.is_finite() && loaded.billed().is_finite());
    }
}
