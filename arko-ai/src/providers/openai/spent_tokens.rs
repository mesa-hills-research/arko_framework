//! Refresh tokens this process has spent.
//!
//! A ChatGPT OAuth refresh token stops working once a refresh with it
//! returns a new one, and the OAuth server takes a second use for a reused
//! token (Codex then asks the user to sign in again). Every refresh that
//! returns a new refresh token records the one it replaced here, so that
//! this process never sends it again, nor adopts or saves credentials
//! holding it. Only SHA-256 digests are kept, never the tokens, and nothing
//! is removed (32 bytes per refresh).
//!
//! Tokens are recorded twice:
//!
//! - By the token endpoint they were spent at (in practice the default one,
//!   so this covers every refresh the process makes). The client checks this
//!   immediately before sending any refresh request, and before adopting
//!   credentials from a credential store.
//! - By the Codex `auth.json` (canonical path) whose
//!   [`CodexAuthStore`](crate::codex_auth::CodexAuthStore) saw the refresh.
//!   The store checks this before adopting the file's credentials or saving
//!   over them.
//!
//! What this can't know: a refresh request whose answer is lost (the
//! connection drops or the request times out after it was sent) may still
//! have rotated the token, but no new token arrived, so the old one is not
//! recorded and the next refresh sends it again; the server then refuses it,
//! as it would without this record. And clients that share neither their
//! credentials nor a store's refresh lock can send the same token at the
//! same time: a token is known spent only once a refresh with it has been
//! answered.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

type TokenDigest = [u8; 32];

struct Spent {
    by_endpoint: BTreeMap<String, BTreeSet<TokenDigest>>,
    by_file: BTreeMap<PathBuf, BTreeSet<TokenDigest>>,
}

static SPENT: Mutex<Spent> = Mutex::new(Spent {
    by_endpoint: BTreeMap::new(),
    by_file: BTreeMap::new(),
});

fn spent() -> MutexGuard<'static, Spent> {
    // Only whole digests are ever inserted: valid after any panic.
    SPENT.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn digest(refresh_token: &str) -> TokenDigest {
    Sha256::digest(refresh_token.as_bytes()).into()
}

/// Records `refresh_token` as spent at `token_url`: a refresh with it
/// returned a new one.
pub(crate) fn record_at_endpoint(token_url: &str, refresh_token: &str) {
    let digest = digest(refresh_token);
    spent()
        .by_endpoint
        .entry(token_url.to_string())
        .or_default()
        .insert(digest);
}

/// Whether `refresh_token` was spent at `token_url` by this process.
pub(crate) fn spent_at_endpoint(token_url: &str, refresh_token: &str) -> bool {
    let digest = digest(refresh_token);
    spent()
        .by_endpoint
        .get(token_url)
        .is_some_and(|tokens| tokens.contains(&digest))
}

/// Records `refresh_token`, read from the `auth.json` at the canonical path
/// `file`, as spent.
pub(crate) fn record_for_file(file: &Path, refresh_token: &str) {
    let digest = digest(refresh_token);
    spent()
        .by_file
        .entry(file.to_path_buf())
        .or_default()
        .insert(digest);
}

/// Whether `refresh_token` was spent by a refresh of the credentials in the
/// `auth.json` at the canonical path `file`.
pub(crate) fn spent_for_file(file: &Path, refresh_token: &str) -> bool {
    let digest = digest(refresh_token);
    spent()
        .by_file
        .get(file)
        .is_some_and(|tokens| tokens.contains(&digest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_recorded_by_digest_per_endpoint_and_per_file() {
        let (endpoint, other_endpoint) = ("http://spent-tokens-test/token", "http://other/token");
        assert!(!spent_at_endpoint(endpoint, "spent-tokens-test-refresh"));
        record_at_endpoint(endpoint, "spent-tokens-test-refresh");
        assert!(spent_at_endpoint(endpoint, "spent-tokens-test-refresh"));
        assert!(!spent_at_endpoint(other_endpoint, "spent-tokens-test-refresh"));
        assert!(!spent_at_endpoint(endpoint, "spent-tokens-test-other"));

        let (file, other_file) = (Path::new("/spent-tokens-test/auth.json"), Path::new("/other/auth.json"));
        assert!(!spent_for_file(file, "spent-tokens-test-refresh"), "scopes are separate");
        record_for_file(file, "spent-tokens-test-refresh");
        assert!(spent_for_file(file, "spent-tokens-test-refresh"));
        assert!(!spent_for_file(other_file, "spent-tokens-test-refresh"));

        // Only digests are kept.
        let spent = spent();
        let tokens = &spent.by_endpoint[endpoint];
        assert!(tokens.contains(&digest("spent-tokens-test-refresh")));
        assert_eq!(digest("spent-tokens-test-refresh").len(), 32);
    }
}
