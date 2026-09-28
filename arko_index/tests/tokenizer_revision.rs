//! Which tokenizer `ArkoBackend::new` loads for an embedding profile, with
//! no network: the Hugging Face cache (`HF_HOME`) is a temporary directory
//! holding a tokenizer, and the Hub (`HF_ENDPOINT`) is a local server that
//! answers 404. Both are process-wide environment variables, so this file
//! is its own test binary with one test.

use arko_ai::{EmbeddingClient, LLMClient};
use arko_index::{EmbeddingProfile, backend::ArkoBackend};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A word-level tokenizer: enough for `ArkoBackend::new` to load.
const TOKENIZER: &str = r#"{
    "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
    "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
    "decoder": null,
    "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "[UNK]": 2}, "unk_token": "[UNK]"}
}"#;

/// Caches `tokenizer.json` of `repo` at `revision` (a branch such as
/// `main`, or a commit), as a download from the Hub leaves it.
fn cache(home: &std::path::Path, repo: &str, revision: &str) {
    let folder = home
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    let commit = if revision.len() == 40 {
        revision
    } else {
        "0123456789abcdef0123456789abcdef01234567"
    };
    std::fs::create_dir_all(folder.join("refs")).unwrap();
    std::fs::write(folder.join("refs").join(revision), commit).unwrap();
    let snapshot = folder.join("snapshots").join(commit);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("tokenizer.json"), TOKENIZER).unwrap();
}

/// A Hub that answers every request with 404 (as huggingface.co does for a
/// revision a repository doesn't have), recording the request paths.
fn hub() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            seen.lock().unwrap().push(
                line.split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string(),
            );
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
                    break;
                }
            }
            let body = "Invalid rev id";
            let _ = write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nX-Error-Code: RevisionNotFound\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (format!("http://{address}"), requests)
}

async fn backend(profile: EmbeddingProfile) -> Result<ArkoBackend, arko_index::Error> {
    ArkoBackend::new(LLMClient::new(), EmbeddingClient::new(), profile).await
}

#[test]
fn the_tokenizer_revision_is_the_profiles_or_the_repositorys_current_version() {
    let home = tempfile::tempdir().unwrap();
    let (endpoint, requests) = hub();
    // SAFETY: this test is the only one in its binary, and no other thread
    // reads the environment yet.
    unsafe {
        std::env::set_var("HF_HOME", home.path());
        std::env::set_var("HF_ENDPOINT", &endpoint);
    }
    let pin = "fedcba9876543210fedcba9876543210fedcba98";
    cache(home.path(), "voyageai/voyage-4-lite", "main");
    cache(home.path(), "voyageai/voyage-3.5", pin);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        // A profile that names only another model loads that model's
        // tokenizer at the repository's current version (it once inherited
        // voyage-4's pinned commit, which the Hub doesn't have for another
        // repository, and failed with advice about network access).
        let lite = EmbeddingProfile {
            model: "voyage-4-lite".into(),
            ..Default::default()
        };
        assert_eq!(lite.tokenizer_revision, None);
        let loaded = backend(lite).await.unwrap();
        assert_eq!(
            arko_index::backend::Backend::output_tokenizer(&loaded),
            "voyageai/voyage-4-lite@main"
        );
        assert!(requests.lock().unwrap().is_empty(), "read from the cache");

        // A pinned revision is honoured: voyage-3.5 is cached only there.
        let pinned = EmbeddingProfile {
            model: "voyage-3.5".into(),
            tokenizer_revision: Some(pin.into()),
            ..Default::default()
        };
        let loaded = backend(pinned).await.unwrap();
        assert_eq!(
            arko_index::backend::Backend::output_tokenizer(&loaded),
            format!("voyageai/voyage-3.5@{pin}")
        );
        assert!(requests.lock().unwrap().is_empty(), "read from the cache");

        // A revision the repository doesn't have is a clear error naming
        // both, without advice about network access.
        let missing = "0000000000000000000000000000000000000000";
        let absent = EmbeddingProfile {
            model: "voyage-4-lite".into(),
            tokenizer_revision: Some(missing.into()),
            ..Default::default()
        };
        let error = match backend(absent).await {
            Ok(_) => panic!("loaded a revision the Hub doesn't have"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&format!("voyageai/voyage-4-lite@{missing}"))
                && error.contains("no tokenizer at this revision"),
            "{error}"
        );
        assert!(!error.contains("HF_ENDPOINT"), "{error}");
        assert_eq!(
            requests.lock().unwrap().as_slice(),
            [format!(
                "/voyageai/voyage-4-lite/resolve/{missing}/tokenizer.json"
            )]
        );
    });
}
