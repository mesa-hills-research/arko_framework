//! Per-job provider usage and cost.
use arko_ai::UsageInfo;
use serde::{Deserialize, Serialize};

/// Token usage and cost of one job's completed provider calls: chunk-model
/// replies (initial requests and repairs) and embedding requests, summed
/// across pauses and resumes. [`JobStatus::usage`](crate::JobStatus::usage)
/// reports it; `request_finished`, `embedded` and `embedding_rejected` events
/// carry each call's share.
///
/// Calls report usage when the backend returns a reply, including embedding
/// replies whose vectors the index then rejects. Failed attempts, including
/// chunk responses the backend rejected (refusals, token limits, a wrong tool
/// call), and calls cancelled mid-flight are not included, although the
/// provider may have billed them. Search query embeddings belong to no job.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JobUsage {
    /// Completed chunk-model calls.
    pub chunk_calls: usize,
    /// Summed usage of the chunk-model calls that reported it.
    pub chunking: Option<UsageInfo>,
    /// Completed embedding requests.
    pub embedding_calls: usize,
    /// Summed usage of the embedding requests that reported it.
    pub embedding: Option<UsageInfo>,
    /// Completed calls whose backend reported no usage; their tokens and
    /// cost are unknown and not included.
    pub unreported_calls: usize,
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
        add(&mut self.chunking, usage, &mut self.unreported_calls);
    }

    pub(crate) fn record_embedding(&mut self, usage: Option<&UsageInfo>) {
        self.embedding_calls = self.embedding_calls.saturating_add(1);
        add(&mut self.embedding, usage, &mut self.unreported_calls);
    }
}

fn add(total: &mut Option<UsageInfo>, usage: Option<&UsageInfo>, unreported: &mut usize) {
    match (total.as_mut(), usage) {
        (_, None) => *unreported = unreported.saturating_add(1),
        (Some(total), Some(usage)) => *total += usage,
        // Summing keeps the first call's billing kind (a default total would
        // turn it into Mixed) and its billed cost.
        (None, Some(usage)) => *total = Some(std::iter::once(usage).sum()),
    }
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
        assert_eq!((chunking.input_tokens, chunking.output_tokens), (2_000, 400));
        assert_eq!(chunking.billing, BillingKind::Subscription);
        assert_eq!(chunking.total_cost, 1.0);
        assert_eq!(chunking.billed(), 0.0);
        let embedding = UsageInfo::for_embedding(1_000_000, 0.06);
        usage.record_embedding(Some(&embedding));
        assert_eq!(
            (usage.chunk_calls, usage.embedding_calls, usage.unreported_calls),
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
        // A record from before billing kinds existed bills its estimate.
        assert_eq!(usage.billed(), 0.25);
    }
}
