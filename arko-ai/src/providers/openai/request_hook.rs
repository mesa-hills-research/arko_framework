use reqwest::header::HeaderMap;
use serde_json::Value;

/// Adjusts OpenAI Responses requests just before they are sent, and observes
/// response headers. Intended for diagnostics and A/B experiments (for
/// example, trying extra routing headers); nothing changes when no hook is
/// installed.
///
/// `conversation_key` is the conversation's `prompt_cache_key`. The body is
/// re-serialized only if [`prepare`](Self::prepare) changes it, so a hook
/// that only edits headers sends exactly the provider's bytes.
pub trait OpenAIRequestHook: Send + Sync {
    fn prepare(&self, conversation_key: &str, headers: &mut HeaderMap, body: &mut Value) {
        let _ = (conversation_key, headers, body);
    }

    fn observe_response(&self, conversation_key: &str, headers: &HeaderMap) {
        let _ = (conversation_key, headers);
    }
}
