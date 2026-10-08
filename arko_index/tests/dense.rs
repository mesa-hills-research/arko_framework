//! Dense and hybrid search with a deterministic mock embedder (no network): the
//! embedding cache, the token cap, persistence and the atomic switch, the vector
//! store's incremental updates and recovery, watcher-driven rebuilds, filters, headers,
//! and chunk identities.
use arko_index::{
    bench_v1::{Chunk, Error, Options, chunk_dir, chunk_str},
    dense::{
        DenseBuild, DenseConfig, EmbedLimits, EmbedText, Embedder, Embedding, HashEmbedder,
        InputKind, ModelDescriptor, RENDER_VERSION, render, render_with,
    },
    hybrid::{self, Fusion, Mode, SearchRequest},
    index2::{self, BuildOptions, BuildStats},
    vector::{Filter, MhrStore, Quantization, VectorMeta, VectorRecord, VectorStore},
};
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// The hash embedder under a chosen model name, with small request limits, counting
/// what it embeds, and optionally failing from a given request on.
struct TestEmbedder {
    inner: HashEmbedder,
    model: String,
    dims: usize,
    limits: EmbedLimits,
    requests: AtomicU64,
    texts: Mutex<Vec<String>>,
    fail_from_request: Option<u64>,
}
impl TestEmbedder {
    fn new(model: &str, dims: usize) -> Self {
        Self {
            inner: HashEmbedder::new(dims),
            model: model.into(),
            dims,
            limits: EmbedLimits::new(4, 400, 300),
            requests: AtomicU64::new(0),
            texts: Mutex::new(Vec::new()),
            fail_from_request: None,
        }
    }
    fn embedded(&self) -> Vec<String> {
        self.texts.lock().unwrap().clone()
    }
    fn reset(&self) {
        self.texts.lock().unwrap().clear();
        self.requests.store(0, Ordering::SeqCst);
    }
}
impl Embedder for TestEmbedder {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor::new("test", self.model.clone(), self.dims).with_input_types(true)
    }
    fn limits(&self) -> EmbedLimits {
        self.limits
    }
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        self.inner.count_tokens(texts)
    }
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        let n = self.requests.fetch_add(1, Ordering::SeqCst);
        if self.fail_from_request.is_some_and(|f| n >= f) {
            return Err("simulated provider outage".into());
        }
        assert!(texts.len() <= self.limits.max_items);
        if kind == InputKind::Document {
            self.texts
                .lock()
                .unwrap()
                .extend(texts.iter().map(|t| t.to_string()));
        }
        self.inner.embed(texts, kind)
    }
}

fn write(root: &Path, path: &str, text: &str) {
    let p = root.join(path);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, text).unwrap();
}

/// A small tree: Rust, Python and Markdown, with distinctive vocabulary per file.
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
}

fn build_with(
    root: &Path,
    state: &Path,
    embedder: Option<&dyn Embedder>,
    cap: Option<u64>,
    threads: usize,
) -> Result<BuildStats, Error> {
    let mut settings = BuildOptions::default().with_threads(threads);
    if let Some(e) = embedder {
        settings = settings.with_dense(DenseBuild::new(e).with_max_tokens(cap));
    }
    index2::build(root, state, &Options::default(), &settings)
}

fn generation(state: &Path) -> std::path::PathBuf {
    state.join(fs::read_to_string(state.join("CURRENT2")).unwrap().trim())
}

/// The bytes of a generation's dense files.
fn dense_files(state: &Path) -> Vec<(String, Vec<u8>)> {
    let g = generation(state);
    ["dense.json", "vkeys.dat"]
        .iter()
        .map(|f| (f.to_string(), fs::read(g.join(f)).unwrap()))
        .collect()
}

/// The vector store directories in a state directory.
fn stores(state: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(state)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("vectors-"))
        .collect();
    names.sort();
    names
}

/// Each hit's chunk ID and score.
fn ranked(r: &hybrid::SearchResults) -> Vec<(String, f64)> {
    r.hits
        .iter()
        .map(|h| (h.chunk.id.clone(), h.score))
        .collect()
}

/// The rendered texts of a tree's chunks, as the index embeds them.
fn rendered(root: &Path) -> BTreeSet<String> {
    chunk_dir(root, &Options::default())
        .unwrap()
        .iter()
        .map(|c| {
            let text = fs::read_to_string(root.join(&c.path)).unwrap();
            render(c, &text[c.start_byte..c.end_byte])
        })
        .collect()
}

#[test]
fn cache_reuses_unchanged_texts_and_reembeds_on_model_change() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    let a = TestEmbedder::new("model-a", 32);

    // First build: every distinct text once, in requests within the limits.
    let s = build_with(root.path(), state.path(), Some(&a), None, 1).unwrap();
    let d = s.dense.unwrap();
    assert_eq!(d.docs, s.docs);
    assert_eq!(d.embedded_docs, s.docs);
    assert_eq!(
        (d.texts, d.cached, d.new),
        (all.len() as u64, 0, all.len() as u64)
    );
    assert_eq!(a.embedded().into_iter().collect::<BTreeSet<_>>(), all);
    assert_eq!(a.embedded().len(), all.len(), "no text embedded twice");
    assert!(d.requests >= (all.len() as u64).div_ceil(4));
    assert!(d.tokens > 0 && !d.cap_reached);
    assert!(d.store_created);
    assert_eq!((d.vectors_added, d.vectors_removed), (s.docs, 0));
    let first = dense_files(state.path());
    let store_a = stores(state.path());
    assert_eq!(store_a.len(), 1);

    // The config is saved with the index.
    let config = DenseConfig::load(state.path()).unwrap().unwrap();
    assert_eq!(config.embedder, a.descriptor());
    assert_eq!(config.render_version, RENDER_VERSION);
    assert!(config.embedder.input_types);
    assert_eq!(config.quantization, Quantization::Int8);
    assert_eq!(config.max_tokens_per_run, None);

    // Unchanged tree, other thread count: nothing embedded, the store unchanged,
    // identical vector ids.
    a.reset();
    let d = build_with(root.path(), state.path(), Some(&a), None, 4)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!((d.new, d.requests, d.tokens), (0, 0, 0));
    assert_eq!(d.cached, all.len() as u64);
    assert!(a.embedded().is_empty());
    assert_eq!((d.vectors_added, d.vectors_removed), (0, 0));
    assert!(!d.store_created);
    assert_eq!(dense_files(state.path()), first);
    assert_eq!(stores(state.path()), store_a);

    // Lines shift above every chunk of a file (blank lines on top): the rendered texts
    // (path, scope, trimmed source) are unchanged, so nothing is embedded.
    let retry = fs::read_to_string(root.path().join("src/net/retry.rs")).unwrap();
    write(root.path(), "src/net/retry.rs", &format!("\n\n\n{retry}"));
    a.reset();
    let d = build_with(root.path(), state.path(), Some(&a), None, 2)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(d.new, 0, "{:?}", a.embedded());

    // One function edited: exactly the texts that changed are embedded.
    write(
        root.path(),
        "src/store/journal.rs",
        "pub fn append_journal_entry(log: &mut Vec<String>, entry: &str) {\n    \
             log.push(entry.trim().to_owned());\n\
         }\n\n\
         pub fn replay_journal(log: &[String]) -> usize {\n    \
             log.iter().filter(|e| !e.is_empty()).count()\n\
         }\n",
    );
    let now = rendered(root.path());
    let changed: BTreeSet<String> = now.difference(&all).cloned().collect();
    assert_eq!(changed.len(), 1);
    a.reset();
    let d = build_with(root.path(), state.path(), Some(&a), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(a.embedded().into_iter().collect::<BTreeSet<_>>(), changed);
    assert_eq!((d.new, d.cached), (1, now.len() as u64 - 1));
    // The store is updated in place: the edited chunk's vector replaced.
    assert_eq!((d.vectors_added, d.vectors_removed), (1, 1));
    assert_eq!(stores(state.path()), store_a);

    // Another model: everything again, with that model; its config replaces the old.
    let b = TestEmbedder::new("model-b", 32);
    let d = build_with(root.path(), state.path(), Some(&b), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!((d.new, d.cached), (now.len() as u64, 0));
    assert_eq!(
        DenseConfig::load(state.path()).unwrap().unwrap().embedder,
        b.descriptor()
    );
    // Its own vector store; the other model's is removed.
    assert!(d.store_created);
    let store_b = stores(state.path());
    assert!(store_b.len() == 1 && store_b != store_a);
    // Other dimensions are another model too.
    let a64 = TestEmbedder::new("model-a", 64);
    let d = build_with(root.path(), state.path(), Some(&a64), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(d.new, now.len() as u64);
    // Back to the first model: its cache is still there.
    a.reset();
    let d = build_with(root.path(), state.path(), Some(&a), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(d.new, 0);
    assert!(a.embedded().is_empty());
    assert_eq!(stores(state.path()), store_a);

    // A lexical-only build leaves the saved config alone and drops the vectors.
    let s = build_with(root.path(), state.path(), None, None, 1).unwrap();
    assert!(s.dense.is_none());
    assert!(DenseConfig::load(state.path()).unwrap().is_some());
    assert!(index2::dense_info(state.path()).unwrap().is_none());
    assert!(stores(state.path()).is_empty());
    assert!(DenseConfig::remove(state.path()).unwrap());
    assert!(DenseConfig::load(state.path()).unwrap().is_none());
}

#[test]
fn token_cap_stops_cleanly_and_a_rerun_finishes() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    let e = TestEmbedder::new("model-a", 32);
    let counts = e
        .count_tokens(&all.iter().map(String::as_str).collect::<Vec<_>>())
        .unwrap();
    let total: u64 = counts.iter().sum();

    // A cap of 0: nothing sent; the report says what embedding would take.
    let d = build_with(root.path(), state.path(), Some(&e), Some(0), 1)
        .unwrap()
        .dense
        .unwrap();
    assert!(d.cap_reached);
    assert_eq!((d.new, d.requests, d.tokens, d.embedded_docs), (0, 0, 0, 0));
    assert_eq!(
        (d.not_embedded, d.not_embedded_tokens),
        (all.len() as u64, total)
    );
    assert!(e.embedded().is_empty());
    // Lexical search still works on that generation; dense finds nothing.
    assert!(!index2::search(state.path(), "backoff").unwrap().is_empty());
    let r = hybrid::search(
        state.path(),
        &SearchRequest::new("backoff").with_mode(Mode::Dense),
        Some(&e),
    )
    .unwrap();
    assert!(r.hits.is_empty());

    // A cap of about half: stops before the request that would pass it.
    let cap = total / 2;
    let d = build_with(root.path(), state.path(), Some(&e), Some(cap), 1)
        .unwrap()
        .dense
        .unwrap();
    assert!(d.cap_reached);
    assert!(d.tokens <= cap && d.tokens > 0, "{} > {cap}", d.tokens);
    assert!(d.new > 0 && d.new < all.len() as u64);
    assert_eq!(d.new + d.not_embedded, all.len() as u64);
    assert_eq!(d.tokens + d.not_embedded_tokens, total);
    assert!(d.embedded_docs < d.docs);
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    assert!(!info.complete());
    assert_eq!(info.embedded, d.embedded_docs);
    assert_eq!(
        DenseConfig::load(state.path())
            .unwrap()
            .unwrap()
            .max_tokens_per_run,
        Some(cap)
    );
    let first_part = d.new;

    // Without a cap, the rest: what the capped run embedded comes from the cache.
    e.reset();
    let d = build_with(root.path(), state.path(), Some(&e), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert!(!d.cap_reached);
    assert_eq!(d.cached, first_part);
    assert_eq!(d.new, all.len() as u64 - first_part);
    assert_eq!(d.embedded_docs, d.docs);
    assert_eq!(e.embedded().len() as u64, d.new);
}

/// The generation directories in a state directory.
fn generations(state: &Path) -> Vec<String> {
    fs::read_dir(state)
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().to_string_lossy().into_owned();
            name.starts_with("index2-").then_some(name)
        })
        .collect()
}

#[test]
fn failed_embedding_keeps_the_published_generation_and_the_paid_vectors() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let all = rendered(root.path());
    // A lexical generation first.
    build_with(root.path(), state.path(), None, None, 1).unwrap();
    let before = fs::read_to_string(state.path().join("CURRENT2")).unwrap();
    let lexical = index2::search(state.path(), "journal").unwrap();

    // The provider fails on the third request: the build fails, nothing is published.
    let mut e = TestEmbedder::new("model-a", 32);
    e.fail_from_request = Some(2);
    assert!(build_with(root.path(), state.path(), Some(&e), None, 1).is_err());
    assert_eq!(
        fs::read_to_string(state.path().join("CURRENT2")).unwrap(),
        before
    );
    // The failed build removed its unpublished generation.
    assert_eq!(generations(state.path()), [before.trim().to_owned()]);
    assert_eq!(index2::search(state.path(), "journal").unwrap(), lexical);
    assert!(index2::dense_info(state.path()).unwrap().is_none());
    assert!(DenseConfig::load(state.path()).unwrap().is_none());
    let paid = e.embedded().len();
    assert_eq!(paid, 8, "two requests of four texts succeeded");

    // The next build embeds only what the failed one did not.
    e.fail_from_request = None;
    e.reset();
    let d = build_with(root.path(), state.path(), Some(&e), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(d.cached, paid as u64);
    assert_eq!(d.new, (all.len() - paid) as u64);
    // The switch published the new generation and removed the old ones.
    let after = fs::read_to_string(state.path().join("CURRENT2")).unwrap();
    assert_ne!(after, before);
    assert_eq!(generations(state.path()), [after.trim().to_owned()]);
    // No build leftovers in the generation.
    let g = generation(state.path());
    assert!(!g.join("vhash.tmp").exists() && !g.join("embed-spool.tmp").exists());
}

#[test]
fn searches_by_mode_with_filters_headers_and_token_counts() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 64);

    // A lexical-only index: dense and hybrid modes explain what is missing.
    build_with(root.path(), state.path(), None, None, 1).unwrap();
    let err = hybrid::search(
        state.path(),
        &SearchRequest::new("retry").with_mode(Mode::Hybrid),
        Some(&e),
    )
    .unwrap_err();
    assert!(err.to_string().contains("no embeddings"), "{err}");

    build_with(root.path(), state.path(), Some(&e), None, 1).unwrap();
    // Lexical: the same ten results as index2::search, with headers and ranks.
    let lexical = hybrid::search(state.path(), &SearchRequest::new("backoff retry"), None).unwrap();
    assert_eq!(lexical.query_tokens, 0);
    assert_eq!(
        lexical
            .hits
            .iter()
            .map(|h| h.chunk.clone())
            .collect::<Vec<_>>(),
        index2::search(state.path(), "backoff retry").unwrap()
    );
    let top = &lexical.hits[0];
    assert_eq!(top.lexical_rank, Some(1));
    assert_eq!(
        top.header,
        format!(
            "{}:{}-{}{}",
            top.chunk.path,
            top.chunk.start_line,
            top.chunk.end_line,
            if top.chunk.scope.is_empty() {
                String::new()
            } else {
                format!(" · {}", top.chunk.scope)
            }
        )
    );

    // Dense: a query that is exactly a chunk's rendered text finds that chunk first,
    // with similarity 1, and the query's tokens are reported.
    let chunks: Vec<Chunk> = chunk_dir(root.path(), &Options::default()).unwrap();
    let target = chunks
        .iter()
        .find(|c| c.scope == "impl RetryPolicy › fn jitter")
        .expect("the jitter method has its own chunk");
    let source = fs::read_to_string(root.path().join(&target.path)).unwrap();
    let text = render(target, &source[target.start_byte..target.end_byte]);
    let dense = hybrid::search(
        state.path(),
        &SearchRequest::new(text.clone()).with_mode(Mode::Dense),
        Some(&e),
    )
    .unwrap();
    assert_eq!(dense.hits[0].chunk, *target);
    assert!((dense.hits[0].score - 1.0).abs() < 1e-5);
    assert_eq!(dense.hits[0].dense_rank, Some(1));
    assert!(dense.query_tokens > 0);
    assert_eq!(dense.dense.as_ref().unwrap().embedder, e.descriptor());
    // A supplied query vector costs nothing and gives the same results.
    let (vector, tokens) = arko_index::dense::embed_query(&e, &text).unwrap();
    assert_eq!(tokens, dense.query_tokens);
    let supplied = hybrid::search(
        state.path(),
        &SearchRequest::new(text.clone())
            .with_mode(Mode::Dense)
            .with_query_vector(vector),
        None,
    )
    .unwrap();
    assert_eq!(supplied.query_tokens, 0);
    assert_eq!(
        supplied.hits.iter().map(|h| &h.chunk).collect::<Vec<_>>(),
        dense.hits.iter().map(|h| &h.chunk).collect::<Vec<_>>()
    );

    // Hybrid: RRF of both lists; a chunk both rank first is first, with both ranks.
    let hybrid_results = hybrid::search(
        state.path(),
        &SearchRequest::new(text.clone()).with_mode(Mode::Hybrid),
        Some(&e),
    )
    .unwrap();
    let first = &hybrid_results.hits[0];
    assert_eq!(first.chunk, *target);
    assert!(first.dense_rank == Some(1) && first.lexical_rank.is_some());
    let linear = hybrid::search(
        state.path(),
        &SearchRequest::new(text)
            .with_mode(Mode::Hybrid)
            .with_fusion(Fusion::Linear {
                lexical_weight: 0.5,
            })
            .with_limit(3),
        Some(&e),
    )
    .unwrap();
    assert_eq!(linear.hits.len(), 3);

    // Filters apply to every mode.
    for mode in [Mode::Lexical, Mode::Dense, Mode::Hybrid] {
        let only_python = SearchRequest::new("summary rows retry journal backoff")
            .with_mode(mode)
            .with_filter(Filter::default().with_language("python"));
        let r = hybrid::search(state.path(), &only_python, Some(&e)).unwrap();
        assert!(!r.hits.is_empty(), "{mode:?}");
        assert!(
            r.hits.iter().all(|h| h.chunk.language == "python"),
            "{mode:?}"
        );
        let under_src_store = SearchRequest::new("summary rows retry journal backoff")
            .with_mode(mode)
            .with_filter(Filter::default().with_path_prefix("src/store/"));
        let r = hybrid::search(state.path(), &under_src_store, Some(&e)).unwrap();
        assert!(!r.hits.is_empty(), "{mode:?}");
        assert!(
            r.hits
                .iter()
                .all(|h| h.chunk.path.starts_with("src/store/")),
            "{mode:?}"
        );
    }

    // The query must be embedded with the index's model.
    let other = TestEmbedder::new("model-b", 64);
    let err = hybrid::search(
        state.path(),
        &SearchRequest::new("retry").with_mode(Mode::Dense),
        Some(&other),
    )
    .unwrap_err();
    assert!(err.to_string().contains("embedded with"), "{err}");
}

/// Runs `hook` once, the first time it embeds a query.
struct RebuildingEmbedder<'a> {
    inner: TestEmbedder,
    queries: AtomicU64,
    hook: Mutex<Option<Box<dyn FnOnce() + Send + 'a>>>,
}
impl Embedder for RebuildingEmbedder<'_> {
    fn descriptor(&self) -> ModelDescriptor {
        self.inner.descriptor()
    }
    fn limits(&self) -> EmbedLimits {
        self.inner.limits()
    }
    fn count_tokens(&self, texts: &[&str]) -> Result<Vec<u64>, Error> {
        self.inner.count_tokens(texts)
    }
    fn embed(&self, texts: &[&str], kind: InputKind) -> Result<Embedding, Error> {
        if kind == InputKind::Query {
            self.queries.fetch_add(1, Ordering::SeqCst);
            if let Some(hook) = self.hook.lock().unwrap().take() {
                hook();
            }
        }
        self.inner.embed(texts, kind)
    }
}

/// A search that was reading a generation when a build published a new one and
/// removed the old one starts over on the new one, without embedding the query again.
#[test]
fn a_search_survives_a_generation_switch_and_embeds_the_query_once() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 32);
    build_with(root.path(), state.path(), Some(&e), None, 1).unwrap();
    let old = generation(state.path());
    let (r, s) = (root.path().to_path_buf(), state.path().to_path_buf());
    let searcher = RebuildingEmbedder {
        inner: TestEmbedder::new("model-a", 32),
        queries: AtomicU64::new(0),
        hook: Mutex::new(Some(Box::new(move || {
            write(&r, "src/new.rs", "pub fn freshly_added() {}\n");
            let rebuild = TestEmbedder::new("model-a", 32);
            build_with(&r, &s, Some(&rebuild), None, 1).unwrap();
        }))),
    };
    let results = hybrid::search(
        state.path(),
        &SearchRequest::new("retry backoff").with_mode(Mode::Hybrid),
        Some(&searcher),
    )
    .unwrap();
    assert!(!old.exists(), "the generation being read was removed");
    assert_eq!(searcher.queries.load(Ordering::SeqCst), 1);
    assert_eq!(
        results.query_tokens,
        searcher.inner.count_tokens(&["retry backoff"]).unwrap()[0]
    );
    assert_eq!(
        results.dense.unwrap().docs,
        index2::dense_info(state.path()).unwrap().unwrap().docs
    );
    assert!(!results.hits.is_empty());
}

#[test]
fn int8_vectors_rescore_to_the_exact_ranking() {
    let root = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 64);
    let mut results = Vec::new();
    for quantization in [Quantization::F32, Quantization::Int8] {
        let state = tempfile::tempdir().unwrap();
        let settings =
            BuildOptions::default().with_dense(DenseBuild::new(&e).with_quantization(quantization));
        index2::build(root.path(), state.path(), &Options::default(), &settings).unwrap();
        assert_eq!(
            index2::dense_info(state.path())
                .unwrap()
                .unwrap()
                .quantization,
            quantization
        );
        let r = hybrid::search(
            state.path(),
            &SearchRequest::new("exponential backoff between attempts").with_mode(Mode::Dense),
            Some(&e),
        )
        .unwrap();
        results.push(
            r.hits
                .iter()
                .map(|h| (h.chunk.id.clone(), h.score))
                .collect::<Vec<_>>(),
        );
    }
    // Few vectors: every one is a rescoring candidate, so the results are exact.
    assert_eq!(results[0], results[1]);
}

#[cfg(unix)]
#[test]
fn a_watcher_delta_reembeds_only_its_new_chunks() {
    use std::time::Duration;
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let watch_state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 32);
    build_with(root.path(), state.path(), Some(&e), None, 1).unwrap();
    let mut watcher =
        arko_index::bench_v1::open_watch(root.path(), watch_state.path(), &Options::default())
            .unwrap();
    let initial = watcher.next_change(Duration::ZERO).unwrap().unwrap();
    watcher.ack(initial.sequence).unwrap();

    // Edit one method's body; the methods after it move.
    let path = root.path().join("src/net/retry.rs");
    let text = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        text.replace(
            "self.base.saturating_mul(1 << attempt.min(16))",
            "// Doubles per attempt, at most 2^16 times the base.\n        self.base.saturating_mul(1 << attempt.min(16))",
        ),
    )
    .unwrap();
    let delta = loop {
        if let Some(d) = watcher.next_change(Duration::from_millis(500)).unwrap() {
            break d;
        }
    };
    assert_eq!((delta.added.len(), delta.removed.len()), (1, 1));
    assert!(
        !delta.changed.is_empty(),
        "moved chunks are reported as changed"
    );

    // The rebuild the delta triggers embeds the added chunk only: moved chunks keep
    // their rendered text.
    e.reset();
    let d = build_with(root.path(), state.path(), Some(&e), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(d.new, 1);
    let added = &delta.added[0];
    let source = fs::read_to_string(&path).unwrap();
    assert_eq!(
        e.embedded(),
        [render(added, &source[added.start_byte..added.end_byte])]
    );
}

/// A build adds the vectors of new texts before it publishes its generation and
/// removes those of texts that are gone after, so a build interrupted between the two
/// leaves vectors the current generation lacks: searches skip them, and the next build
/// removes them.
#[test]
fn an_interrupted_build_leaves_searches_exact_and_the_next_build_cleans_up() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 32);
    build_with(root.path(), state.path(), Some(&e), None, 1).unwrap();
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    let search = |mode: Mode| {
        let request = SearchRequest::new("exponential backoff journal entry")
            .with_mode(mode)
            .with_limit(5)
            .with_candidates(5);
        ranked(&hybrid::search(state.path(), &request, Some(&e)).unwrap())
    };
    let before = (search(Mode::Dense), search(Mode::Hybrid));
    assert_eq!(before.0.len(), 5);

    // What an interrupted build's upserts leave: vectors under ids this generation
    // lacks, at least as similar to the query as its own.
    let dir = state.path().join(&info.store);
    let mut store = MhrStore::open_for(&dir, &info.store_config()).unwrap();
    let (query, _) =
        arko_index::dense::embed_query(&e, "exponential backoff journal entry").unwrap();
    let vectors: Vec<Vec<f32>> = (0..40)
        .map(|i| query.iter().map(|x| x + 0.001 * i as f32).collect())
        .collect();
    let records: Vec<VectorRecord<'_>> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| VectorRecord {
            id: u128::MAX - i as u128,
            vector: v,
            meta: VectorMeta {
                path: "src/net/retry.rs",
                language: "rust",
            },
        })
        .collect();
    store.upsert(&records).unwrap();
    store.persist().unwrap();
    drop(store);
    assert_eq!((search(Mode::Dense), search(Mode::Hybrid)), before);

    // The next build removes them and adds nothing.
    let d = build_with(root.path(), state.path(), Some(&e), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!((d.vectors_added, d.vectors_removed), (0, 40));
    assert_eq!((search(Mode::Dense), search(Mode::Hybrid)), before);
    assert_eq!(
        MhrStore::open(&dir).unwrap().len() as u64,
        index2::dense_info(state.path()).unwrap().unwrap().embedded
    );
}

/// Vectors stored by an earlier version (a generation's `dense.json` of format 1), or
/// a damaged store, fail dense search with a clear error; lexical search still works,
/// and the next build stores the vectors again from the cache.
#[test]
fn old_or_damaged_vectors_fail_clearly_and_a_build_recovers() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 32);
    build_with(root.path(), state.path(), Some(&e), None, 1).unwrap();
    let dense = SearchRequest::new("journal entry").with_mode(Mode::Dense);
    let good = ranked(&hybrid::search(state.path(), &dense, Some(&e)).unwrap());

    // A generation written by the earlier format: its own vectors in `dense/`.
    let g = generation(state.path());
    let mut old: serde_json::Value =
        serde_json::from_slice(&fs::read(g.join("dense.json")).unwrap()).unwrap();
    old["version"] = 1.into();
    let old = old.as_object_mut().unwrap();
    old.remove("store");
    old.remove("vector_index");
    fs::write(g.join("dense.json"), serde_json::to_vec(&old).unwrap()).unwrap();
    let err = hybrid::search(state.path(), &dense, Some(&e)).unwrap_err();
    assert!(err.to_string().contains("earlier version"), "{err}");
    assert!(!index2::search(state.path(), "journal").unwrap().is_empty());
    e.reset();
    let d = build_with(root.path(), state.path(), Some(&e), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert_eq!(d.new, 0, "vectors come from the cache");
    assert_eq!(
        ranked(&hybrid::search(state.path(), &dense, Some(&e)).unwrap()),
        good
    );

    // A damaged store.
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    fs::write(state.path().join(&info.store).join("CURRENT"), b"damaged").unwrap();
    assert!(hybrid::search(state.path(), &dense, Some(&e)).is_err());
    assert!(!index2::search(state.path(), "journal").unwrap().is_empty());
    let d = build_with(root.path(), state.path(), Some(&e), None, 1)
        .unwrap()
        .dense
        .unwrap();
    assert!(d.store_created && d.new == 0);
    assert_eq!(d.vectors_added, d.embedded_docs);
    assert_eq!(
        ranked(&hybrid::search(state.path(), &dense, Some(&e)).unwrap()),
        good
    );
}

/// Context headers (`Chunk::scope`) for each language family, and chunk boundaries and
/// IDs identical to chunk format 1 (recorded from the build before scopes existed).
/// Markdown, plain text in format 1, is chunked by its grammar since.
#[test]
fn scopes_are_metadata_and_identities_are_unchanged() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let chunks = chunk_dir(&dir.join("scopes"), &Options::default()).unwrap();
    let golden: Vec<Chunk> = fs::read_to_string(dir.join("scopes-format1.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Chunk>(l).unwrap())
        .filter(|c| c.path != "guide.md")
        .collect();
    let code: Vec<&Chunk> = chunks.iter().filter(|c| c.path != "guide.md").collect();
    assert_eq!(code.len(), golden.len());
    for (new, old) in code.into_iter().zip(&golden) {
        assert!(old.scope.is_empty(), "format 1 has no scope");
        let without_scope = Chunk {
            scope: String::new(),
            ..new.clone()
        };
        assert_eq!(&without_scope, old);
    }
    let scope_of = |path: &str, line: usize| -> &str {
        &chunks
            .iter()
            .find(|c| c.path == path && c.start_line <= line && line <= c.end_line)
            .unwrap()
            .scope
    };
    // Rust: modules, inherent and trait impls, traits, macros.
    assert_eq!(
        scope_of("sample.rs", 17),
        "mod geometry › mod shapes › impl Point › fn distance"
    );
    assert_eq!(
        scope_of("sample.rs", 30),
        "mod geometry › mod shapes › impl fmt::Display for Point › fn fmt"
    );
    assert_eq!(
        scope_of("sample.rs", 10),
        "mod geometry › mod shapes › struct Point"
    );
    assert_eq!(scope_of("sample.rs", 38), "trait Area › fn describe");
    assert_eq!(scope_of("sample.rs", 44), "impl Area for Vec<T> › fn area");
    assert_eq!(scope_of("sample.rs", 50), "macro_rules! square");
    assert_eq!(scope_of("sample.rs", 55), "fn exponential_backoff");
    assert_eq!(scope_of("sample.rs", 2), "");
    // Python: methods of a class, async methods, decorated functions.
    assert_eq!(scope_of("sample.py", 16), "class RetryPolicy › def delay");
    assert_eq!(
        scope_of("sample.py", 25),
        "class RetryPolicy › async def run"
    );
    assert_eq!(scope_of("sample.py", 42), "def fibonacci");
    // TypeScript: namespaces, exported classes and functions, bound arrow functions.
    assert_eq!(scope_of("sample.ts", 10), "namespace Storage");
    assert_eq!(scope_of("sample.ts", 30), "class ConnectionPool");
    assert_eq!(scope_of("sample.ts", 56), "function retryDelay");
    assert_eq!(scope_of("sample.ts", 62), "function formatBytes");
    // Go and C++.
    assert_eq!(scope_of("sample.go", 10), "func (s *Server) Start");
    assert_eq!(scope_of("sample.go", 5), "type Server");
    assert_eq!(
        scope_of("sample.cpp", 20),
        "namespace net › namespace http › class Router › function find"
    );
    // Markdown: the whole small file (without its front matter) is one chunk, under the
    // front matter's title.
    assert_eq!(
        scope_of("guide.md", 10),
        "A guide with front matter › # Guide"
    );
    assert!(
        chunks
            .iter()
            .all(|c| c.path != "guide.md" || c.start_line > 4)
    );
}

#[test]
fn markdown_heading_chains_in_small_chunks() {
    let text = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/scopes/guide.md"),
    )
    .unwrap();
    let chunks = chunk_str(
        Path::new("guide.md"),
        &text,
        &Options {
            max_bytes: 64,
            ..Options::default()
        },
    )
    .unwrap();
    let scope_at = |needle: &str| -> String {
        let at = text.find(needle).unwrap();
        chunks
            .iter()
            .find(|c| c.start_byte <= at && at < c.end_byte)
            .unwrap()
            .scope
            .clone()
    };
    assert_eq!(
        scope_at("This guide explains"),
        "A guide with front matter › # Guide"
    );
    assert_eq!(
        scope_at("sudo apt install"),
        "A guide with front matter › # Guide › ## Install › ### Linux"
    );
    assert_eq!(
        scope_at("brew install"),
        "A guide with front matter › # Guide › ## Install › ### macOS"
    );
    assert_eq!(
        scope_at("Run `the-tool --help`"),
        "A guide with front matter › # Guide › ## Use"
    );
    assert_eq!(
        scope_at("The configuration file"),
        "A guide with front matter › # Guide › ## Configuration"
    );
    // Front matter and fenced comments are not headings.
    assert!(chunks.iter().all(|c| !c.scope.contains("not a heading")));
    // The header shown with a result.
    let c = chunks
        .iter()
        .find(|c| c.scope.ends_with("### macOS"))
        .unwrap();
    assert_eq!(
        c.header(),
        format!(
            "guide.md:{}-{} · A guide with front matter › # Guide › ## Install › ### macOS",
            c.start_line, c.end_line
        )
    );
}

#[test]
fn render_format_version_one() {
    let chunks = chunk_str(
        Path::new("src/lib.rs"),
        "impl Foo for Bar {\n    fn baz(&self) {}\n}\n",
        &Options::default(),
    )
    .unwrap();
    let c = &chunks[0];
    let text = "impl Foo for Bar {\n    fn baz(&self) {}\n}\n";
    assert_eq!(
        render(c, &text[c.start_byte..c.end_byte]),
        format!(
            "src/lib.rs\n{}{}",
            if c.scope.is_empty() {
                String::new()
            } else {
                format!("{}\n", c.scope)
            },
            text[c.start_byte..c.end_byte].trim()
        )
    );
    let top = chunk_str(Path::new("a.txt"), "  plain text  \n", &Options::default()).unwrap();
    assert_eq!(render(&top[0], "  plain text  \n"), "a.txt\nplain text");
}

#[test]
fn embedded_text_choices_render_and_are_saved() {
    let source = "impl Foo for Bar {\n    fn baz(&self) {}\n}\n";
    let chunks = chunk_str(Path::new("src/lib.rs"), source, &Options::default()).unwrap();
    let c = chunks.iter().find(|c| !c.scope.is_empty()).unwrap();
    let text = &source[c.start_byte..c.end_byte];
    let body = text.trim();
    assert_eq!(render_with(c, text, EmbedText::Source), body);
    assert_eq!(
        render_with(c, text, EmbedText::PathSource),
        format!("src/lib.rs\n{body}")
    );
    assert_eq!(
        render_with(c, text, EmbedText::PathScopeSource),
        render(c, text)
    );
    assert_eq!(EmbedText::default(), EmbedText::PathScopeSource);
    for choice in [
        EmbedText::Source,
        EmbedText::PathSource,
        EmbedText::PathScopeSource,
    ] {
        assert_eq!(choice.as_str().parse::<EmbedText>().unwrap(), choice);
    }
    assert!("path".parse::<EmbedText>().is_err());

    // A build embeds the chosen text, and the index's config and generation say so.
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    corpus(root.path());
    let e = TestEmbedder::new("model-a", 32);
    let settings = BuildOptions::default()
        .with_threads(1)
        .with_dense(DenseBuild::new(&e).with_embed_text(EmbedText::Source));
    let s = index2::build(root.path(), state.path(), &Options::default(), &settings).unwrap();
    assert_eq!(s.dense.unwrap().embed_text, EmbedText::Source);
    let sources: BTreeSet<String> = chunk_dir(root.path(), &Options::default())
        .unwrap()
        .iter()
        .map(|c| {
            let text = fs::read_to_string(root.path().join(&c.path)).unwrap();
            text[c.start_byte..c.end_byte].trim().to_string()
        })
        .collect();
    assert_eq!(e.embedded().into_iter().collect::<BTreeSet<_>>(), sources);
    let config = DenseConfig::load(state.path()).unwrap().unwrap();
    assert_eq!(config.embed_text, EmbedText::Source);
    let info = index2::dense_info(state.path()).unwrap().unwrap();
    assert_eq!(info.embed_text, EmbedText::Source);
    // A config saved without the field (before it existed) means the default.
    let file = state.path().join(DenseConfig::FILE);
    let mut json: serde_json::Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("embed_text");
    fs::write(&file, serde_json::to_vec(&json).unwrap()).unwrap();
    let config = DenseConfig::load(state.path()).unwrap().unwrap();
    assert_eq!(config.embed_text, EmbedText::PathScopeSource);

    // The default text is another text for every chunk with a path: all embedded anew.
    e.reset();
    let s = build_with(root.path(), state.path(), Some(&e), None, 1).unwrap();
    assert_eq!(s.dense.unwrap().embed_text, EmbedText::PathScopeSource);
    assert_eq!(
        e.embedded().into_iter().collect::<BTreeSet<_>>(),
        rendered(root.path())
    );
}

#[test]
fn lexical_scope_indexes_the_scope_words() {
    let root = tempfile::tempdir().unwrap();
    corpus(root.path());
    write(
        root.path(),
        "docs/handbook.md",
        "---\ntitle: Zanzibar handbook\ndescription: Quokka onboarding notes\n---\n\n## Setup\n\nInstall the tools first.\n",
    );
    let paths = |state: &Path, query: &str| -> Vec<String> {
        index2::search(state, query)
            .unwrap()
            .into_iter()
            .map(|c| c.path)
            .collect()
    };
    let build = |lexical_scope: bool, options: &Options| {
        let state = tempfile::tempdir().unwrap();
        let settings = BuildOptions::default()
            .with_threads(1)
            .with_lexical_scope(lexical_scope);
        index2::build(root.path(), state.path(), options, &settings).unwrap();
        state
    };
    // The title is the front matter's, outside every chunk: only the scope holds it.
    let plain = build(false, &Options::default());
    assert!(paths(plain.path(), "zanzibar").is_empty());
    let scoped = build(true, &Options::default());
    assert_eq!(paths(scoped.path(), "zanzibar"), ["docs/handbook.md"]);
    assert!(paths(scoped.path(), "quokka").is_empty());
    // The description joins the scope with `front_matter_description`.
    let described = build(
        true,
        &Options {
            front_matter_description: true,
            ..Options::default()
        },
    );
    assert_eq!(paths(described.path(), "quokka"), ["docs/handbook.md"]);
}
