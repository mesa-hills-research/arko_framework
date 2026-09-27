//! Refresh tokens this process knows are spent.
//!
//! A ChatGPT OAuth refresh token stops working once a refresh with it
//! returns a new one, and the OAuth server takes a second use for a reused
//! token (Codex then asks the user to sign in again). So every refresh that
//! returns a new refresh token records the one it replaced here, and so does
//! every refresh the server refuses because the token was reused, expired or
//! revoked (`refresh_token_reused`, `refresh_token_expired`,
//! `refresh_token_invalidated`, which Codex treats as final too). This
//! process then never sends such a token again, and never adopts or saves
//! credentials holding one: the client checks this record immediately before
//! sending any refresh request and before adopting stored credentials, and
//! [`CodexAuthStore`](crate::codex_auth::CodexAuthStore) checks it before
//! offering or saving over the file's credentials.
//!
//! Refresh tokens are unique, so the record is one set for the whole
//! process. Only SHA-256 digests are kept, never the tokens, and nothing is
//! removed (32 bytes per refresh).
//!
//! What this can't know:
//!
//! - A refresh request whose answer is lost (the connection drops or the
//!   request times out after it was sent) may still have rotated the token,
//!   but no new token arrived, so the old one is not recorded and the next
//!   refresh sends it again; the server then refuses it (and it is recorded).
//! - Clients that share neither their credentials nor a store's refresh lock
//!   can send the same token at the same time: a token is known spent only
//!   once a refresh with it has been answered.
//! - Tokens spent by another process, or by an earlier run of this one: an
//!   old `auth.json` restored from a backup looks live until its refresh is
//!   refused.

use super::ChatGptOAuthCredentials;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::{Mutex, MutexGuard};

type TokenDigest = [u8; 32];

static SPENT: Mutex<BTreeSet<TokenDigest>> = Mutex::new(BTreeSet::new());

fn spent() -> MutexGuard<'static, BTreeSet<TokenDigest>> {
    // Only whole digests are ever inserted: valid after any panic.
    SPENT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
thread_local! {
    /// Tests run in parallel threads of one process, many with the same fake
    /// tokens: each test thread (with its current-thread runtime) gets its
    /// own record, as if it were a process of its own.
    static TEST_SCOPE: u64 = {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    };
}

fn digest(refresh_token: &str) -> TokenDigest {
    let mut hasher = Sha256::new();
    #[cfg(test)]
    hasher.update(TEST_SCOPE.with(|scope| scope.to_le_bytes()));
    hasher.update(refresh_token.as_bytes());
    hasher.finalize().into()
}

/// Records `refresh_token` as spent: a refresh with it returned a new one,
/// or the server refused it for good.
pub(crate) fn record(refresh_token: &str) {
    let digest = digest(refresh_token);
    spent().insert(digest);
}

/// Whether this process knows `refresh_token` is spent.
pub(crate) fn is_spent(refresh_token: &str) -> bool {
    let digest = digest(refresh_token);
    spent().contains(&digest)
}

/// Whether `credentials` hold a refresh token this process knows is spent.
pub(crate) fn credentials_spent(credentials: &ChatGptOAuthCredentials) -> bool {
    credentials.refresh_token.as_deref().is_some_and(is_spent)
}

/// Error codes of a refused refresh after which the refresh token can never
/// be used again (as Codex classifies them: exhausted, expired, revoked).
const FINAL_REFRESH_ERRORS: [&str; 3] = [
    "refresh_token_reused",
    "refresh_token_expired",
    "refresh_token_invalidated",
];

/// Whether the error `body` of a refused refresh says the refresh token can
/// never be used again. The code is read as Codex reads it: `error` when it
/// is a string, else `error.code`, else a top-level `code`.
pub(crate) fn refused_for_good(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) else {
        return false;
    };
    fn text(value: Option<&serde_json::Value>) -> Option<&str> {
        value
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.is_empty())
    }
    let code = text(value.get("error"))
        .or_else(|| text(value.get("error").and_then(|error| error.get("code"))))
        .or_else(|| text(value.get("code")));
    code.is_some_and(|code| {
        FINAL_REFRESH_ERRORS
            .iter()
            .any(|final_code| final_code.eq_ignore_ascii_case(code))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_recorded_by_digest() {
        assert!(!is_spent("spent-tokens-test-refresh"));
        record("spent-tokens-test-refresh");
        assert!(is_spent("spent-tokens-test-refresh"));
        assert!(!is_spent("spent-tokens-test-other"));
        let credentials = ChatGptOAuthCredentials {
            access_token: "access".into(),
            refresh_token: Some("spent-tokens-test-refresh".into()),
            account_id: "account".into(),
        };
        assert!(credentials_spent(&credentials));
        assert!(!credentials_spent(&ChatGptOAuthCredentials {
            refresh_token: None,
            ..credentials
        }));
        // Only digests are kept.
        assert!(spent().contains(&digest("spent-tokens-test-refresh")));

        // Another test thread has a record of its own.
        std::thread::spawn(|| assert!(!is_spent("spent-tokens-test-refresh")))
            .join()
            .unwrap();
    }

    #[test]
    fn refusals_for_good_are_read_as_codex_reads_them() {
        for body in [
            r#"{"error":"refresh_token_reused"}"#,
            r#"{"error":{"code":"refresh_token_expired","message":"expired"}}"#,
            r#"{"code":"REFRESH_TOKEN_INVALIDATED"}"#,
        ] {
            assert!(refused_for_good(body), "{body}");
        }
        for body in [
            r#"{"error":"invalid_grant"}"#,
            r#"{"error":{"code":"rate_limit_exceeded"}}"#,
            "not json",
            "",
        ] {
            assert!(!refused_for_good(body), "{body}");
        }
    }
}
