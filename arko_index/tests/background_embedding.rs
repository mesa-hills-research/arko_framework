//! Embedding while the index is in use: a build publishes its generation for lexical
//! search before anything is embedded, and an embedding run adds the vectors request by
//! request. Searches during a run, edits and rebuilds while a run is going, a crash and a
//! stop mid-run, the token cap, and watcher-driven rebuilds. A slow embedder (the hash
//! embedder behind a gate the test opens one request at a time) stands in for a model
//! that takes minutes or hours.
use arko_index::{
    bench_v1::{Chunk, Error, Options, chunk_dir},
    dense::{
        DenseBuild, EmbedLimits, EmbedState, Embedder, Embedding, HashEmbedder, InputKind,
        ModelDescriptor, StopSignal, embed_query, render,
    },
    hybrid::{self, Mode, SearchRequest, SearchResults},
    index2::{self, BuildOptions, BuildStats},
    vector::{CaprockStore, VectorStore},
};
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    sync::{
        Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const DIMS: usize = 32;

/// The hash embedder behind a gate: each document request waits until the test lets
/// one through ([`SlowEmbedder::allow`]), as a request to a slow model would take its
/// time. Queries pass at once. It records the texts it embeds, and can panic on a
/// given request, as a process that dies mid-run.
struct SlowEmbedder {
    inner: HashEmbedder,
    permits: Mutex<u64>,
    gate: Condvar,
    started: AtomicU64,
    texts: Mutex<Vec<String>>,
    panic_on_request: Option<u64>,
}
impl SlowEmbedder {
    /// Closed: no request passes until [`allow`](Self::allow).
    fn new() -> Self {
        Self {
            inner: HashEmbedder::new(DIMS),
            permits: Mutex::new(0),
            gate: Condvar::new(),
            started: AtomicU64::new(0),
            texts: Mutex::new(Vec::new()),
            panic_on_request: None,
        }
    }
    /// Every request passes at once.
    fn open() -> Self {
        let e = Self::new();
        e.allow(u64::MAX / 2);
        e
    }
    fn allow(&self, requests: u64) {
        *self.permits.lock().unwrap() += requests;
        self.gate.notify_all();
    }
    /// Waits until `n` document requests have started: the run is at the gate of the
    /// `n`th, its earlier requests done.
    fn wait_for_request(&self, n: u64) {
        wait_until(|| self.started.load(Ordering::SeqCst) >= n);
    }
    fn texts(&self) -> Vec<String> {
        self.texts.lock().unwrap().clone()
    }
}
impl Embedder for SlowEmbedder {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor::new("test", "slow", DIMS).with_input_types(true)
    }
    fn limits(&self) -> EmbedLimits {
        EmbedLimits::new(4, 400, 300)
    }
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        self.inner.count_tokens(texts)
    }
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        if kind == InputKind::Document {
            let n = self.started.fetch_add(1, Ordering::SeqCst) + 1;
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut permits = self.permits.lock().unwrap();
            while *permits == 0 {
                assert!(
                    Instant::now() < deadline,
                    "request {n} was never let through"
                );
                permits = self
                    .gate
                    .wait_timeout(permits, Duration::from_millis(50))
                    .unwrap()
                    .0;
            }
            *permits -= 1;
            drop(permits);
            if self.panic_on_request == Some(n) {
                panic!("the process died during request {n}");
            }
            self.texts
                .lock()
                .unwrap()
                .extend(texts.iter().map(|t| t.to_string()));
        }
        self.inner.embed(texts, kind)
    }
}

/// Polls `done` for up to 30 seconds.
fn wait_until(done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn write(root: &Path, path: &str, text: &str) {
    let p = root.join(path);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, text).unwrap();
}

/// A small tree: Rust, Python and Markdown.
fn corpus(root: &Path) {
    write(
        root,
        "src/net/retry.rs",
        "pub struct RetryPolicy { base: u64 }\n\n\
         impl RetryPolicy {\n    \
             /// Exponential backoff between attempts.\n    \
             pub fn backoff(&self, attempt: u32) -> u64 {\n        \
                 self.base.saturating_mul(1 << attempt.min(16))\n    \
             }\n\n    \
             pub fn jitter(&self, sample: u64) -> u64 {\n        \
                 sample % self.base.max(1)\n    \
             }\n\
         }\n\n\
         pub fn connect_with_retries(host: &str) -> bool {\n    \
             !host.is_empty()\n\
         }\n",
    );
    write(
        root,
        "src/store/journal.rs",
        "pub fn append_journal_entry(log: &mut Vec<String>, entry: &str) {\n    \
             log.push(entry.to_owned());\n\
         }\n\n\
         pub fn replay_journal(log: &[String]) -> usize {\n    \
             log.iter().filter(|e| !e.is_empty()).count()\n\
         }\n",
    );
    write(
        root,
        "tools/report.py",
        "class Report:\n    \
             def summary(self, rows):\n        \
                 return sum(len(r) for r in rows)\n\n\
         def render_table(rows):\n    \
             return '\\n'.join(rows)\n",
    );
    write(
        root,
        "docs/guide.md",
        "# Guide\n\n## Install\n\nRun the installer with cargo.\n\n## Retry\n\nThe client retries with exponential backoff.\n",
    );
    write(
        root,
        "src/store/index.rs",
        "pub fn rebuild_index(entries: &[String]) -> Vec<String> {\n    \
             let mut sorted = entries.to_vec();\n    \
             sorted.sort();\n    \
             sorted\n\
         }\n\n\
         pub fn lookup_entry(index: &[String], key: &str) -> Option<usize> {\n    \
             index.binary_search_by(|e| e.as_str().cmp(key)).ok()\n\
         }\n",
    );
}

/// The text a chunk is embedded with, read from the tree now.
fn current_text(root: &Path, c: &Chunk) -> String {
    let source = fs::read_to_string(root.join(&c.path)).unwrap();
    render(c, &source[c.start_byte..c.end_byte])
}

/// The rendered texts of a tree's chunks.
fn rendered(root: &Path) -> BTreeSet<String> {
    chunk_dir(root, &Options::default())
        .unwrap()
        .iter()
        .map(|c| current_text(root, c))
        .collect()
}

fn build(root: &Path, state: &Path, dense: Option<DenseBuild<'_>>) -> BuildStats {
    let mut settings = BuildOptions::default().with_threads(1);
    if let Some(dense) = dense {
        settings = settings.with_dense(dense);
    }
    index2::build(root, state, &Options::default(), &settings).unwrap()
}

/// A build that publishes its generation and leaves the embedding to a run.
fn build_lexical_first(root: &Path, state: &Path, e: &dyn Embedder) -> BuildStats {
    build(root, state, Some(DenseBuild::new(e).with_embed_later(true)))
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / norm(a) / norm(b)
}

/// A dense search with the query's vector, every result: each hit is a chunk with a
/// vector, scored against the embedding of its text as the tree holds it now (never a
/// vector of an older text).
fn dense_exact(root: &Path, state: &Path, query: &str) -> SearchResults {
    let hash = HashEmbedder::new(DIMS);
    let (vector, _) = embed_query(&hash, query).unwrap();
    let r = hybrid::search(
        state,
        &SearchRequest::new(query)
            .with_mode(Mode::Dense)
            .with_limit(200)
            .with_query_vector(vector.clone()),
        None,
    )
    .unwrap();
    for hit in &r.hits {
        let text = current_text(root, &hit.chunk);
        let own = hash
            .embed(&[&text], InputKind::Document)
            .unwrap()
            .vectors
            .remove(0);
        let want = cosine(&vector, &own);
        assert!(
            (hit.score - want).abs() < 1e-4,
            "{}: scored {} but its text scores {want}",
            hit.header,
            hit.score
        );
    }
    r
}

fn ids(r: &SearchResults) -> Vec<String> {
    r.hits.iter().map(|h| h.chunk.id.clone()).collect()
}

/// The vectors in the index's store.
fn stored(state: &Path) -> usize {
    let info = index2::dense_info(state).unwrap().unwrap();
    CaprockStore::open(&state.join(&info.store)).unwrap().len()
}

#[test]
fn lexical_search_is_ready_while_dense_coverage_fills_in() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let lexical = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    let e = SlowEmbedder::new();

    let s = build_lexical_first(root.path(), state.path(), &e);
    let d = s.dense.unwrap();
    assert_eq!((d.embedded_docs, d.cached, d.new, d.requests), (0, 0, 0, 0));
    assert_eq!((d.docs, d.texts), (s.docs, all.len() as u64));
    assert!(e.texts().is_empty());

    // Lexical search is complete: the same as an index built without embeddings.
    build(root.path(), lexical.path(), None);
    let queries = [
        "retry backoff",
        "journal entry",
        "render table rows",
        "install cargo",
    ];
    for q in queries {
        assert_eq!(
            index2::search(state.path(), q).unwrap(),
            index2::search(lexical.path(), q).unwrap(),
            "{q}"
        );
    }
    // Nothing embedded yet: dense finds nothing, hybrid is the lexical ranking, and the
    // query is not embedded.
    let hybrid_now = |q: &str| {
        hybrid::search(
            state.path(),
            &SearchRequest::new(q).with_mode(Mode::Hybrid),
            Some(&e),
        )
        .unwrap()
    };
    let r = hybrid_now("retry backoff");
    let info = r.dense.clone().unwrap();
    assert_eq!((info.embedded, info.state), (0, EmbedState::Embedding));
    assert_eq!(r.query_tokens, 0);
    assert_eq!(
        r.hits.iter().map(|h| &h.chunk).collect::<Vec<_>>(),
        index2::search(lexical.path(), "retry backoff")
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    );
    assert!(
        dense_exact(root.path(), state.path(), "retry backoff")
            .hits
            .is_empty()
    );
    let status = index2::status(state.path()).unwrap();
    assert!(!status.embedding && status.progress.is_none());

    std::thread::scope(|scope| {
        let run = scope.spawn(|| index2::embed(state.path(), &DenseBuild::new(&e)));
        // One request through; the run waits at its second.
        e.allow(1);
        e.wait_for_request(2);
        wait_until(|| index2::dense_info(state.path()).unwrap().unwrap().embedded > 0);
        let info = index2::dense_info(state.path()).unwrap().unwrap();
        assert_eq!(info.state, EmbedState::Embedding);
        assert!(info.embedded >= 4 && info.embedded < info.docs, "{info:?}");
        let status = index2::status(state.path()).unwrap();
        assert!(status.embedding);
        let progress = status.progress.unwrap();
        assert_eq!((progress.requests, progress.texts), (1, 4));
        assert_eq!(progress.total, all.len() as u64);
        assert_eq!(progress.embedded_docs, info.embedded);

        // Dense search finds exactly the chunks with a vector, scored exactly; hybrid
        // fuses them with the lexical ranking and says how far the coverage goes.
        let r = dense_exact(root.path(), state.path(), "exponential backoff retry");
        assert_eq!(r.hits.len() as u64, info.embedded);
        let h = hybrid_now("exponential backoff retry");
        assert!(h.query_tokens > 0);
        assert!(h.hits.iter().any(|hit| hit.dense_rank.is_some()));
        assert_eq!(h.dense.unwrap().embedded, info.embedded);
        for q in queries {
            assert_eq!(
                index2::search(state.path(), q).unwrap(),
                index2::search(lexical.path(), q).unwrap()
            );
        }

        e.allow(u64::MAX / 2);
        let stats = run.join().unwrap().unwrap();
        assert_eq!(stats.new, all.len() as u64);
        assert_eq!(stats.embedded_docs, stats.docs);
        assert!(!stats.superseded && !stats.stopped && !stats.cap_reached);
    });
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    assert_eq!(
        (info.state, info.embedded),
        (EmbedState::Complete, info.docs)
    );
    assert!(!index2::status(state.path()).unwrap().embedding);
    assert_eq!(e.texts().len(), all.len(), "no text embedded twice");
    // The same results as an index embedded before it was published.
    let embedded_first = tempfile::tempdir().unwrap();
    let open = SlowEmbedder::open();
    build(
        root.path(),
        embedded_first.path(),
        Some(DenseBuild::new(&open)),
    );
    for q in queries {
        let a = dense_exact(root.path(), state.path(), q);
        let b = dense_exact(root.path(), embedded_first.path(), q);
        assert_eq!(ids(&a), ids(&b), "{q}");
    }
}

#[test]
fn edits_during_a_run_never_return_stale_vectors() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = SlowEmbedder::new();
    build_lexical_first(root.path(), state.path(), &e);

    std::thread::scope(|scope| {
        let first = scope.spawn(|| index2::embed(state.path(), &DenseBuild::new(&e)));
        e.allow(2);
        e.wait_for_request(3);
        wait_until(|| {
            index2::status(state.path())
                .unwrap()
                .progress
                .is_some_and(|p| p.requests == 2)
        });
        // Edit a chunk that has a vector: its old text's vector is in the store.
        let r = dense_exact(root.path(), state.path(), "journal entry backoff");
        let edited = r
            .hits
            .iter()
            .map(|h| h.chunk.clone())
            .find(|c| c.path.ends_with(".rs") && current_text(root.path(), c).contains("fn "))
            .expect("a Rust function has a vector");
        let old_text = current_text(root.path(), &edited);
        let source = fs::read_to_string(root.path().join(&edited.path)).unwrap();
        let body = &source[edited.start_byte..edited.end_byte];
        let changed = body.replacen("(", "(_edited: (), ", 1);
        assert_ne!(changed, body);
        write(
            root.path(),
            &edited.path,
            &format!(
                "{}{changed}{}",
                &source[..edited.start_byte],
                &source[edited.end_byte..]
            ),
        );
        // A rebuild while the run waits: the new generation is searchable at once.
        let s = build_lexical_first(root.path(), state.path(), &e);
        let d = s.dense.unwrap();
        assert!(d.embedded_docs > 0 && d.embedded_docs < d.docs);
        // The old text's vector is still in the store, but no chunk is found by it:
        // every hit is scored against its current text.
        let r = dense_exact(root.path(), state.path(), &old_text);
        assert!(
            r.hits.iter().all(|h| h.score < 0.9999),
            "the edited chunk's old vector was returned"
        );
        assert_eq!(r.hits.len() as u64, d.embedded_docs);
        assert!(stored(state.path()) as u64 > d.embedded_docs);

        // The first run sends its next request, stores it, and yields to the new
        // generation's run.
        e.allow(1);
        let s1 = first.join().unwrap().unwrap();
        assert!(s1.superseded && !s1.stopped);
        assert_eq!(s1.requests, 3);
        let before = e.texts();
        e.allow(u64::MAX / 2);
        let s2 = index2::embed(state.path(), &DenseBuild::new(&e)).unwrap();
        assert!(!s2.superseded);
        assert_eq!(s2.embedded_docs, s2.docs);
        // What the first run embedded came from the cache: no text was embedded twice.
        let now = rendered(root.path());
        let reused = before.iter().filter(|t| now.contains(*t)).count() as u64;
        assert_eq!(s2.cached, reused);
        assert_eq!(s2.new, now.len() as u64 - reused);
        let texts = e.texts();
        assert_eq!(
            texts.iter().collect::<BTreeSet<_>>().len(),
            texts.len(),
            "a text was embedded twice"
        );
        assert!(s2.vectors_removed > 0, "the old text's vector is removed");
    });
    // The store holds exactly the current generation's vectors, all exact.
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    assert_eq!(
        (info.state, info.embedded),
        (EmbedState::Complete, info.docs)
    );
    assert_eq!(stored(state.path()) as u64, info.docs);
    for q in ["journal entry", "retry backoff", "lookup entry index"] {
        assert_eq!(
            dense_exact(root.path(), state.path(), q).hits.len() as u64,
            info.docs
        );
    }
}

#[test]
fn a_crash_mid_run_resumes_from_the_cache_without_paying_again() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    let first = SlowEmbedder::new();
    build_lexical_first(root.path(), state.path(), &first);

    // The process dies during the third request: two were paid for and stored.
    let crashing = SlowEmbedder {
        panic_on_request: Some(3),
        ..SlowEmbedder::open()
    };
    let crashed = std::thread::scope(|scope| {
        scope
            .spawn(|| index2::embed(state.path(), &DenseBuild::new(&crashing)))
            .join()
    });
    assert!(crashed.is_err());
    let paid: BTreeSet<String> = crashing.texts().into_iter().collect();
    assert_eq!(paid.len(), 8);
    let status = index2::status(state.path()).unwrap();
    assert!(!status.embedding, "a dead run holds no lock");
    let info = status.dense.unwrap();
    assert_eq!(info.state, EmbedState::Embedding);
    assert!(info.embedded >= 8 && info.embedded < info.docs);
    assert_eq!(
        dense_exact(root.path(), state.path(), "retry").hits.len() as u64,
        info.embedded
    );

    // The next run takes the paid texts from the cache and embeds the rest only.
    let e = SlowEmbedder::open();
    let s = index2::embed(state.path(), &DenseBuild::new(&e)).unwrap();
    assert_eq!(s.cached, 8);
    assert_eq!(s.new, all.len() as u64 - 8);
    let again: BTreeSet<String> = e.texts().into_iter().collect();
    assert!(again.is_disjoint(&paid));
    let rest: Vec<&str> = all
        .iter()
        .filter(|t| !paid.contains(*t))
        .map(String::as_str)
        .collect();
    assert_eq!(s.tokens, e.count_tokens(&rest).unwrap().iter().sum::<u64>());
    assert_eq!(s.embedded_docs, s.docs);
    assert_eq!(
        index2::dense_info(state.path()).unwrap().unwrap().state,
        EmbedState::Complete
    );
    dense_exact(root.path(), state.path(), "journal");
}

#[test]
fn the_cap_stops_a_run_and_a_later_run_finishes() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    let e = SlowEmbedder::open();
    let texts: Vec<&str> = all.iter().map(String::as_str).collect();
    let total: u64 = e.count_tokens(&texts).unwrap().iter().sum();
    build_lexical_first(root.path(), state.path(), &e);

    let cap = total / 2;
    let s = index2::embed(
        state.path(),
        &DenseBuild::new(&e).with_max_tokens(Some(cap)),
    )
    .unwrap();
    assert!(s.cap_reached && !s.stopped && !s.superseded);
    assert!(s.tokens > 0 && s.tokens <= cap);
    assert_eq!(s.new + s.not_embedded, all.len() as u64);
    assert_eq!(s.tokens + s.not_embedded_tokens, total);
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    assert_eq!(info.state, EmbedState::CapReached);
    assert_eq!(info.embedded, s.embedded_docs);
    assert!(info.embedded > 0 && info.embedded < info.docs);
    let r = hybrid::search(
        state.path(),
        &SearchRequest::new("retry backoff").with_mode(Mode::Hybrid),
        Some(&e),
    )
    .unwrap();
    assert_eq!(r.dense.unwrap().state, EmbedState::CapReached);
    assert_eq!(
        dense_exact(root.path(), state.path(), "retry").hits.len() as u64,
        info.embedded
    );

    // A run without a cap embeds what the capped one left, and only that.
    let rest = SlowEmbedder::open();
    let s2 = index2::embed(state.path(), &DenseBuild::new(&rest)).unwrap();
    assert_eq!((s2.cached, s2.new), (s.new, s.not_embedded));
    assert_eq!(s2.tokens, s.not_embedded_tokens);
    assert_eq!(s2.embedded_docs, s2.docs);
    let mut both = e.texts();
    both.extend(rest.texts());
    assert_eq!(both.into_iter().collect::<BTreeSet<_>>(), all);
}

#[test]
fn a_stopped_run_continues_and_a_lexical_build_waits_for_a_run_to_yield() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    let e = SlowEmbedder::new();
    build_lexical_first(root.path(), state.path(), &e);

    // A stop request ends the run after its current request.
    let stop = StopSignal::new();
    std::thread::scope(|scope| {
        let run =
            scope.spawn(|| index2::embed(state.path(), &DenseBuild::new(&e).with_stop(&stop)));
        e.wait_for_request(1);
        assert!(stop.stop("asked to stop"));
        assert!(!stop.stop("asked again"));
        e.allow(1);
        let s = run.join().unwrap().unwrap();
        assert!(s.stopped && !s.superseded);
        assert_eq!(s.requests, 1);
    });
    let status = index2::status(state.path()).unwrap();
    let info = status.dense.unwrap();
    assert_eq!(info.state, EmbedState::Embedding);
    assert!(info.embedded > 0 && info.embedded < info.docs);
    assert_eq!(status.stopped.as_deref(), Some("asked to stop"));
    assert!(!status.embedding);

    // A lexical build during a run publishes at once and removes the vectors once the
    // run, superseded, has stopped writing them.
    std::thread::scope(|scope| {
        let run = scope.spawn(|| index2::embed(state.path(), &DenseBuild::new(&e)));
        e.allow(1);
        e.wait_for_request(3);
        let lexical = scope.spawn(|| build(root.path(), state.path(), None));
        wait_until(|| index2::dense_info(state.path()).unwrap().is_none());
        assert!(!index2::search(state.path(), "journal").unwrap().is_empty());
        e.allow(1);
        let s = run.join().unwrap().unwrap();
        assert!(s.superseded);
        assert!(lexical.join().unwrap().dense.is_none());
    });
    let stores: Vec<_> = fs::read_dir(state.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("vectors-"))
        .collect();
    assert!(stores.is_empty(), "{stores:?}");
    // Embedding again (a build with embeddings) takes every paid text from the cache.
    let paid: BTreeSet<String> = e.texts().into_iter().collect();
    let open = SlowEmbedder::open();
    let d = build(root.path(), state.path(), Some(DenseBuild::new(&open)))
        .dense
        .unwrap();
    assert_eq!(d.cached, paid.len() as u64);
    assert_eq!(d.new, (all.len() - paid.len()) as u64);
    assert_eq!(d.embedded_docs, d.docs);
}

/// A host keeps the index current from a watcher's deltas, rebuilding on each, while
/// one thread embeds: each run goes until it finishes or a newer generation is
/// published, and then the next run starts.
#[cfg(unix)]
#[test]
fn watcher_driven_rebuilds_while_a_run_is_embedding() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let watch_state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = SlowEmbedder::new();
    build_lexical_first(root.path(), state.path(), &e);
    let mut watcher =
        arko_index::bench_v1::open_watch(root.path(), watch_state.path(), &Options::default())
            .unwrap();
    let initial = watcher.next_change(Duration::ZERO).unwrap().unwrap();
    watcher.ack(initial.sequence).unwrap();

    let runs = std::thread::scope(|scope| {
        let embedding = scope.spawn(|| {
            let mut runs = Vec::new();
            loop {
                let s = index2::embed(state.path(), &DenseBuild::new(&e)).unwrap();
                let again = s.superseded;
                runs.push(s);
                if !again {
                    return runs;
                }
            }
        });
        e.allow(1);
        e.wait_for_request(2);
        // Two edits while the run waits at its second request.
        let path = root.path().join("src/net/retry.rs");
        let text = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            text.replace("sample % self.base.max(1)", "sample % self.base.max(2)"),
        )
        .unwrap();
        write(
            root.path(),
            "src/net/timeout.rs",
            "pub fn connect_timeout_millis(attempt: u32) -> u64 {\n    250 * u64::from(attempt + 1)\n}\n",
        );
        let delta = loop {
            if let Some(d) = watcher.next_change(Duration::from_millis(500)).unwrap() {
                break d;
            }
        };
        assert!(!delta.added.is_empty() && !delta.removed.is_empty());
        let s = build_lexical_first(root.path(), state.path(), &e);
        watcher.ack(delta.sequence).unwrap();
        // The new chunk is found lexically at once.
        let hits = index2::search(state.path(), "connect_timeout_millis").unwrap();
        assert_eq!(hits[0].path, "src/net/timeout.rs");
        assert!(s.dense.unwrap().embedded_docs > 0);
        e.allow(u64::MAX / 2);
        embedding.join().unwrap()
    });
    assert_eq!(runs.len(), 2);
    assert!(runs[0].superseded && !runs[1].superseded);
    assert_eq!(runs[1].embedded_docs, runs[1].docs);
    let texts = e.texts();
    assert_eq!(
        texts.iter().collect::<BTreeSet<_>>().len(),
        texts.len(),
        "a text was embedded twice"
    );
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    assert_eq!(
        (info.state, info.embedded),
        (EmbedState::Complete, info.docs)
    );
    assert_eq!(stored(state.path()) as u64, info.docs);
    // The added chunk, searched by its own text, is first with similarity 1.
    let chunk = chunk_dir(root.path(), &Options::default())
        .unwrap()
        .into_iter()
        .find(|c| c.path == "src/net/timeout.rs")
        .unwrap();
    let r = dense_exact(
        root.path(),
        state.path(),
        &current_text(root.path(), &chunk),
    );
    assert_eq!(r.hits[0].chunk, chunk);
    assert!((r.hits[0].score - 1.0).abs() < 1e-5);
}

/// The `arko_index_cli` example, which `cargo test` builds before the tests.
#[cfg(unix)]
fn cli() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().and_then(Path::parent).unwrap();
    let cli = profile_dir.join("examples").join("arko_index_cli");
    assert!(
        cli.is_file(),
        "{} is missing: run the whole suite (`cargo test -p arko_index`) or build it first",
        cli.display()
    );
    cli
}

/// The CLI's `index2` returns once lexical search is ready, a background process
/// embeds, `status` follows it, and `--wait-embeddings` and `embed` run in the
/// foreground.
#[cfg(unix)]
#[test]
fn the_cli_embeds_in_the_background_and_reports_progress() {
    use std::process::Command;
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let run = |args: &[&str]| {
        let out = Command::new(cli()).args(args).output().unwrap();
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(out.status.success(), "{args:?}: {stderr}");
        (String::from_utf8(out.stdout).unwrap(), stderr)
    };
    let (root_arg, state_arg) = (
        root.path().to_str().unwrap(),
        state.path().to_str().unwrap(),
    );
    let status =
        || -> serde_json::Value { serde_json::from_str(&run(&["status", state_arg]).0).unwrap() };

    let (_, stderr) = run(&[
        "index2",
        "--embed",
        "hash",
        "--dimensions",
        "64",
        root_arg,
        state_arg,
    ]);
    assert!(stderr.contains("lexical search is ready"), "{stderr}");
    assert!(stderr.contains("embedding in the background"), "{stderr}");
    let (hits, _) = run(&[
        "search2", "--mode", "lexical", state_arg, "journal", "entry",
    ]);
    assert!(hits.contains("src/store/journal.rs"), "{hits}");
    wait_until(|| {
        let s = status();
        s["dense"]["state"] == "complete" && s["embedding"] == false
    });
    let s = status();
    assert_eq!(s["dense"]["embedded"], s["docs"]);
    assert_eq!(s["progress"]["embedded_docs"], s["docs"]);
    let log = fs::read_to_string(state.path().join("embed.log")).unwrap();
    assert!(
        log.contains("embeddings: hash/hash-v1 (64 dimensions)"),
        "{log}"
    );
    let (hits, stderr) = run(&["search2", state_arg, "retry", "backoff"]);
    assert!(!hits.is_empty());
    assert!(
        !stderr.contains("dense:"),
        "complete coverage: no notice\n{stderr}"
    );

    // An edit, rebuilt with --wait-embeddings: one new text, embedded before it returns.
    let path = root.path().join("src/store/journal.rs");
    let text = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        text.replace("entry.to_owned()", "entry.trim().to_owned()"),
    )
    .unwrap();
    let (_, stderr) = run(&["index2", "--wait-embeddings", root_arg, state_arg]);
    assert!(stderr.contains("1 embedded now"), "{stderr}");
    assert!(stderr.contains("1 vectors added, 1 removed"), "{stderr}");
    let s = status();
    assert!(s["dense"]["state"] == "complete" && s["embedding"] == false);
    // Nothing left for a foreground run.
    let (_, stderr) = run(&["embed", state_arg]);
    assert!(stderr.contains("0 embedded now in 0 requests"), "{stderr}");
}
