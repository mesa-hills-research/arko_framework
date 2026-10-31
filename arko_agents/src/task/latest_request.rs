//! What a task knows of its current conversation's latest completed model
//! request, which a host times and prices the conversation's prompt cache
//! by. See [`Task::latest_request`](super::Task::latest_request).

use std::time::Duration;

use arko_ai::UsageInfo;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A completed model request of a task's current conversation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatestRequest {
    /// When it was sent. The provider uses the prompt cache as it reads a
    /// request, so the cache lasts from about then. In arko's own runtime
    /// this is just before the provider call, a little early. Claude Code
    /// sends a Claude Code model's requests itself, so there it is the
    /// response's first streamed token, a few seconds late (or the response
    /// itself, for one that wasn't streamed).
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub sent_at: DateTime<Utc>,
    /// Its whole input in tokens, cache reads and writes included. Continuing
    /// the conversation sends it again.
    pub input_tokens: usize,
    /// How much of the input was read from the prompt cache.
    #[serde(default)]
    pub cached_tokens: usize,
    /// How much of the input was written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: usize,
    /// How long the conversation's cache lasts, in seconds, where the
    /// provider chooses it and its cache writes show it. That's a Claude
    /// Code model's, from its latest write's lifetime. `None` where the run's
    /// configuration sets the lifetime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_lifetime_secs: Option<u64>,
}

impl LatestRequest {
    /// The request sent at `sent_at` that had `usage`.
    pub fn new(sent_at: DateTime<Utc>, usage: &UsageInfo) -> Self {
        Self {
            sent_at,
            input_tokens: usage.input_tokens,
            cached_tokens: usage.cached_tokens.unwrap_or(0),
            cache_write_tokens: usage.cache_write_tokens.unwrap_or(0),
            cache_lifetime_secs: None,
        }
    }

    /// The request with the cache lifetime its writes showed.
    pub fn with_cache_lifetime(mut self, lifetime: Option<Duration>) -> Self {
        self.cache_lifetime_secs = lifetime.map(|lifetime| lifetime.as_secs());
        self
    }

    /// How long the conversation's cache lasts, when the provider showed it
    /// ([`cache_lifetime_secs`](Self::cache_lifetime_secs)).
    pub fn cache_lifetime(&self) -> Option<Duration> {
        self.cache_lifetime_secs.map(Duration::from_secs)
    }
}

/// What a task knows of its current conversation's latest completed model
/// request.
///
/// It is saved in a field of its own, and the key's absence is told apart
/// from `null`. A task saved before the record existed has no key and reads
/// as [`Unrecorded`](Self::Unrecorded). One with no request has `null` and
/// reads as [`NoRequest`](Self::NoRequest). An `Unrecorded` task is saved
/// again without the key, so it stays so until its next request completes or
/// its conversation is replaced.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RequestRecord {
    /// Saved by a version before the record existed, so not known. A host
    /// can't tell how long such a conversation's cache lasts.
    #[default]
    Unrecorded,
    /// No request of the current conversation has completed, as in a new
    /// task, or one handed over, compacted, cut back or branched since its
    /// last request. Nothing of it is cached.
    NoRequest,
    /// The current conversation's latest completed request.
    Latest(LatestRequest),
}

impl RequestRecord {
    /// The latest request, when there is one.
    pub fn latest(&self) -> Option<&LatestRequest> {
        match self {
            Self::Latest(request) => Some(request),
            Self::Unrecorded | Self::NoRequest => None,
        }
    }

    /// Whether the task was saved before the record existed. The field
    /// isn't saved then.
    pub fn is_unrecorded(&self) -> bool {
        matches!(self, Self::Unrecorded)
    }
}

impl Serialize for RequestRecord {
    /// `null` without a request, else the request. (A field skips
    /// `Unrecorded`, which would otherwise read back as `NoRequest`.)
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.latest().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RequestRecord {
    /// Read only for a key that is present, where `null` is `NoRequest`. A
    /// field takes `#[serde(default)]` for an absent key to read as
    /// `Unrecorded`. Without it, serde gives a missing key's `None` to this
    /// deserializer, which reads it as `NoRequest`.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Option::<LatestRequest>::deserialize(deserializer)?
            .map_or(Self::NoRequest, Self::Latest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Saved {
        #[serde(default, skip_serializing_if = "RequestRecord::is_unrecorded")]
        latest_request: RequestRecord,
    }

    fn request() -> LatestRequest {
        let mut usage = UsageInfo::new(52_000, 300)
            .with_cached_tokens(50_000)
            .with_cache_write_tokens(2_000);
        usage.cache_write_1h_tokens = Some(2_000);
        LatestRequest::new(
            DateTime::from_timestamp_millis(1_790_000_000_123).unwrap(),
            &usage,
        )
        .with_cache_lifetime(Some(Duration::from_secs(3_600)))
    }

    /// An absent key, `null` and a request each read back as they were
    /// saved, the absent key included.
    #[test]
    fn the_three_states_survive_a_save() {
        let read = |json: &str| serde_json::from_str::<Saved>(json).unwrap();
        let unrecorded = read("{}");
        assert_eq!(unrecorded.latest_request, RequestRecord::Unrecorded);
        assert_eq!(serde_json::to_string(&unrecorded).unwrap(), "{}");
        let none = read(r#"{"latest_request":null}"#);
        assert_eq!(none.latest_request, RequestRecord::NoRequest);
        assert_eq!(
            serde_json::to_string(&none).unwrap(),
            r#"{"latest_request":null}"#
        );
        let saved = Saved {
            latest_request: RequestRecord::Latest(request()),
        };
        let json = serde_json::to_string(&saved).unwrap();
        assert!(json.contains(r#""sent_at":1790000000123"#), "{json}");
        assert_eq!(read(&json), saved);
        let request = saved.latest_request.latest().unwrap();
        assert_eq!(
            (
                request.input_tokens,
                request.cached_tokens,
                request.cache_write_tokens
            ),
            (52_000, 50_000, 2_000)
        );
        assert_eq!(request.cache_lifetime(), Some(Duration::from_secs(3_600)));
    }

    /// A new task knows it has no request yet. A task saved before the
    /// record existed reads as unrecorded, and is saved again without it.
    #[test]
    fn a_task_saved_before_the_record_reads_as_unrecorded() {
        let config = crate::agent_config::AgentConfig::builder(
            "chat".into(),
            "Chat".into(),
            crate::agent_config::AgentLLMConfig {
                model: arko_ai::Model::anthropic_claude_opus_5_5(),
                generation_config: Default::default(),
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .build()
        .unwrap();
        let task = super::super::Task::new(config, "Plan a trip.".into(), "/w".into(), None);
        assert_eq!(task.latest_request, RequestRecord::NoRequest);
        let mut saved = serde_json::to_value(&task).unwrap();
        assert_eq!(saved["latest_request"], serde_json::Value::Null);
        saved.as_object_mut().unwrap().remove("latest_request");
        let old: super::super::Task = serde_json::from_value(saved).unwrap();
        assert_eq!(old.latest_request, RequestRecord::Unrecorded);
        let again = serde_json::to_value(&old).unwrap();
        assert!(again.get("latest_request").is_none(), "{again}");
    }
}
