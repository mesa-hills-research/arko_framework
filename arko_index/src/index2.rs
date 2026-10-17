//! Scaling prototype: a disk-resident inverted index that scores exactly like
//! `search::search` (same tokens, same BM25 expression, same tie-break) but reads
//! only the postings of the query terms instead of deserializing every document.
//! [`BuildOptions::lexical_scope`] (off by default) also indexes each chunk's scope,
//! which `search::index` does not.
//!
//! Layout of one generation directory `index2-<stamp>` (published atomically by
//! rewriting `CURRENT2`, so readers keep the previous generation until the switch):
//!   meta.json  N, total token count, exact f64 bits of the average length
//!   lens.dat   u32 LE token length per doc (doc ids follow (path, start_byte) order)
//!   docs.off   u64 LE byte offset of each doc's Chunk JSON in docs.dat (N + 1)
//!   docs.dat   concatenated Chunk JSON
//!   terms.dat  sorted entries: u32 len, term, u32 df, u64 postings offset, u32 postings len
//!   terms.idx  every BLOCK-th term: u32 len, term, u64 offset into terms.dat
//!   post.dat   per term: (varint doc delta, varint tf)*
//!   files.dat  per file with chunks: u32 LE first doc, u32 LE path length, path
//!              (for path and language filters)
//! With embeddings ([`BuildOptions::dense`]) also:
//!   dense.json  the generation's embedding model, settings and vector store
//!               ([`DenseIndexInfo`]), with the coverage it had when published
//!   vkeys.dat   per doc, sorted by vector id: u128 BE id, u32 LE doc
//!   vhash.dat   per doc, in doc order: the 32-byte hash of its embedded text
//!   embed.spool the texts the cache lacked when the generation was built, until a
//!               run has embedded them all
//! The vectors themselves are in a store beside the generations,
//! `STATE_DIR/vectors-<digest of model and settings>/` ([`CaprockStore`]): a doc's vector
//! id is a digest of its path and its embedded text, so an unchanged chunk keeps its
//! vector, whatever its doc id, and a stored vector always embeds its doc's current text.
//! A build publishes its generation as soon as the lexical files are written. An
//! embedding run ([`embed`], or the build itself unless [`DenseBuild::embed_later`])
//! then brings the store up to date for the current generation: it removes the vectors
//! the generation lacks, adds the ones the cache holds, embeds the rest request by
//! request, and records the coverage in `STATE_DIR/embed-progress.json` after each
//! commit of the store. A run stops when a newer generation is published.
//! Build memory is bounded: postings accumulate up to `mem_budget` bytes, then spill
//! to a sorted run file. Runs cover increasing doc ranges and are merged by term.
//! Texts to embed are spooled to disk.
use crate::{
    bench_v1::{Chunk, Error, Options},
    dense::{
        DenseBuild, DenseConfig, DenseIndexInfo, EmbedProgress, EmbedState, EmbedStats, EmbedText,
        RENDER_VERSION, StopSignal,
        cache::{Cache, cached_hashes},
        render_with,
        run::{self, Flow},
        text_format, text_hash,
    },
    files,
    files::{FileRules, Report},
    fsio::{atomic_write, read_exact_at, read_u32, read_u64},
    par,
    search::tokens,
    tokenize::{self, Counts, FxHasher, FxMap},
    vector::{
        self, CaprockStore, Filter, StoreConfig, VectorId, VectorMeta, VectorRecord, VectorStore,
    },
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet},
    fs::{File, OpenOptions},
    hash::Hasher,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const BLOCK: usize = 64;

#[derive(Serialize, Deserialize)]
struct Meta {
    version: u32,
    docs: u64,
    total: u64,
    average_bits: u64,
    terms: u64,
    /// The generation has embeddings (`dense.json` and `vkeys.dat`). Absent before this
    /// was recorded.
    #[serde(default)]
    dense: bool,
}

/// Removes a generation directory unless `CURRENT2` names it: a build that fails before
/// publishing leaves nothing behind (the embedding cache keeps every vector paid for).
struct Unpublished {
    state_dir: PathBuf,
    generation: Option<String>,
}
impl Drop for Unpublished {
    fn drop(&mut self) {
        if let Some(generation) = &self.generation {
            let current = std::fs::read_to_string(self.state_dir.join("CURRENT2"));
            if current.is_ok_and(|c| c.trim() == generation) {
                return;
            }
            let _ = std::fs::remove_dir_all(self.state_dir.join(generation));
        }
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}
fn get_varint(buf: &[u8], pos: &mut usize) -> Result<u64, Error> {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = *buf.get(*pos).ok_or("truncated postings")?;
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return Ok(v);
        }
        shift += 7;
        if shift > 63 {
            return Err("bad varint".into());
        }
    }
}

struct Acc {
    first: u32,
    last: u32,
    df: u32,
    // varint(tf of first), then (varint delta, varint tf)*
    bytes: Vec<u8>,
}

/// One partition of the term space (terms are assigned by hash). Each shard sees
/// its terms' postings in increasing doc order, accumulates them up to its share of
/// the memory budget, then spills a sorted run. Runs of one shard cover increasing
/// doc ranges, and no term is in two shards, so the final merge is unchanged.
struct Shard {
    dir: PathBuf,
    id: usize,
    terms: FxMap<Box<[u8]>, Acc>,
    mem: usize,
    budget: usize,
    runs: Vec<PathBuf>,
}

impl Shard {
    fn add(&mut self, term: &[u8], doc: u32, tf: u32) {
        match self.terms.get_mut(term) {
            Some(acc) => {
                let before = acc.bytes.len();
                put_varint(&mut acc.bytes, u64::from(doc - acc.last));
                put_varint(&mut acc.bytes, u64::from(tf));
                acc.last = doc;
                acc.df += 1;
                self.mem += acc.bytes.len() - before;
            }
            None => {
                let mut bytes = Vec::with_capacity(4);
                put_varint(&mut bytes, u64::from(tf));
                self.mem += term.len() + bytes.capacity() + 96;
                self.terms.insert(
                    term.into(),
                    Acc {
                        first: doc,
                        last: doc,
                        df: 1,
                        bytes,
                    },
                );
            }
        }
    }
    /// Adds one file's postings for this shard: (varint local doc, varint term
    /// length, term, varint tf)*, with doc ids relative to `base`.
    fn ingest(&mut self, base: u32, buf: &[u8]) -> Result<(), Error> {
        let mut pos = 0;
        while pos < buf.len() {
            let doc = base + get_varint(buf, &mut pos)? as u32;
            let n = get_varint(buf, &mut pos)? as usize;
            let term = buf.get(pos..pos + n).ok_or("truncated term")?;
            pos += n;
            let tf = get_varint(buf, &mut pos)? as u32;
            self.add(term, doc, tf);
            if self.mem >= self.budget {
                self.spill()?;
            }
        }
        Ok(())
    }
    fn spill(&mut self) -> Result<(), Error> {
        if self.terms.is_empty() {
            return Ok(());
        }
        let path = self
            .dir
            .join(format!("run-{:03}-{:05}", self.id, self.runs.len()));
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        let mut entries: Vec<_> = std::mem::take(&mut self.terms).into_iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (term, acc) in entries {
            out.write_all(&(term.len() as u32).to_le_bytes())?;
            out.write_all(&term)?;
            for v in [acc.first, acc.last, acc.df, acc.bytes.len() as u32] {
                out.write_all(&v.to_le_bytes())?;
            }
            out.write_all(&acc.bytes)?;
        }
        out.flush()?;
        self.runs.push(path);
        self.mem = 0;
        Ok(())
    }
}

/// What a worker produces for one file: the chunks' JSON (doc store bytes), their
/// token lengths, and their postings split by shard.
struct FileDocs {
    /// Not indexed: the report entry (`None` for a file that vanished).
    skip: Option<Option<files::Skipped>>,
    path: String,
    json: Vec<u8>,
    ends: Vec<u64>,
    lens: Vec<u32>,
    shards: Vec<Vec<u8>>,
    /// With embeddings: each chunk's rendered-text hash...
    hashes: Vec<[u8; 32]>,
    /// ...and the rendered text of each chunk the cache does not hold.
    missing: Vec<([u8; 32], String)>,
    /// The report's warning when the file's parser failed and it was split as text.
    warning: Option<String>,
}

fn shard_of(term: &[u8], shards: usize) -> usize {
    if shards == 1 {
        return 0;
    }
    let mut h = FxHasher::default();
    h.write(term);
    (h.finish() % shards as u64) as usize
}

/// What a document holds besides its source: the scope's terms for lexical search,
/// and the parts of the embedded text.
#[derive(Clone, Copy)]
struct DocText {
    lexical_scope: bool,
    embed_text: EmbedText,
}

/// Reads, chunks and tokenizes one file (runs on a worker thread).
fn file_docs(
    item: &files::Listed,
    options: &Options,
    shards: usize,
    cached: Option<&HashSet<[u8; 32]>>,
    doc_text: DocText,
) -> Result<FileDocs, Error> {
    let mut out = FileDocs {
        skip: None,
        path: String::new(),
        json: Vec::new(),
        ends: Vec::new(),
        lens: Vec::new(),
        shards: vec![Vec::new(); shards],
        hashes: Vec::new(),
        missing: Vec::new(),
        warning: None,
    };
    let (path, file) = match files::read_listed(item, false) {
        files::Read::Text(path, file) => (path, file),
        files::Read::Skip(s) => {
            out.skip = Some(Some(s));
            return Ok(out);
        }
        files::Read::Gone => {
            out.skip = Some(None);
            return Ok(out);
        }
    };
    let (chunks, warning) = crate::syntax::chunk_noted(Path::new(&path), &file.text, options)?;
    let chunks = chunks.chunks;
    out.warning = warning;
    if chunks.is_empty() {
        return Ok(out);
    }
    // A document's text is "path\npath-relative source"; '\n' separates words, so its
    // terms are the path's terms plus the source's terms (plus, with `lexical_scope`,
    // the scope's terms).
    let path_lower = path.to_ascii_lowercase();
    let mut path_counts = Counts::default();
    let path_total = tokenize::count(&path, &path_lower, &mut path_counts);
    let lowered = file.text.to_ascii_lowercase();
    let mut counts = Counts::default();
    for (local, chunk) in chunks.iter().enumerate() {
        let (s, e) = (chunk.start_byte, chunk.end_byte);
        counts.clear();
        let mut len = tokenize::count(&file.text[s..e], &lowered[s..e], &mut counts);
        for (t, n) in &path_counts {
            *counts.entry(Cow::Borrowed(t.as_ref())).or_insert(0) += n;
        }
        len += path_total;
        if doc_text.lexical_scope {
            for t in tokens(&chunk.scope) {
                *counts.entry(Cow::Owned(t)).or_insert(0) += 1;
                len += 1;
            }
        }
        for (t, &tf) in &counts {
            let t = t.as_bytes();
            let buf = &mut out.shards[shard_of(t, shards)];
            put_varint(buf, local as u64);
            put_varint(buf, t.len() as u64);
            buf.extend_from_slice(t);
            put_varint(buf, u64::from(tf));
        }
        out.lens.push(len);
        serde_json::to_writer(&mut out.json, chunk)?;
        out.ends.push(out.json.len() as u64);
        if let Some(cached) = cached {
            let text = render_with(chunk, &file.text[s..e], doc_text.embed_text);
            let hash = text_hash(&text);
            out.hashes.push(hash);
            if !cached.contains(&hash) {
                out.missing.push((hash, text));
            }
        }
    }
    out.path = path;
    Ok(out)
}

struct RunReader {
    r: BufReader<File>,
    /// (first doc, last doc, df, postings bytes) of the current term. The term
    /// itself lives in the merge heap.
    head: Option<(u32, u32, u32, Vec<u8>)>,
}
impl RunReader {
    /// Moves to the next term of the run and returns it (None at the end).
    fn advance(&mut self) -> Result<Option<String>, Error> {
        let mut n = [0u8; 4];
        match self.r.read_exact(&mut n) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.head = None;
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
        let mut term = vec![0u8; u32::from_le_bytes(n) as usize];
        self.r.read_exact(&mut term)?;
        let mut f = [0u32; 4];
        for v in &mut f {
            self.r.read_exact(&mut n)?;
            *v = u32::from_le_bytes(n);
        }
        let mut bytes = vec![0u8; f[3] as usize];
        self.r.read_exact(&mut bytes)?;
        self.head = Some((f[0], f[1], f[2], bytes));
        Ok(Some(String::from_utf8(term)?))
    }
}

pub struct BuildStats {
    pub docs: u64,
    pub terms: u64,
    pub runs: usize,
    /// What the walk indexed and skipped (also written to `STATE_DIR/report.json`).
    pub report: Report,
    /// What embedding did, when the build had [`BuildOptions::dense`].
    pub dense: Option<EmbedStats>,
}

/// Settings of [`build`]. `Default`: [`threads()`](crate::threads), a 256 MB postings
/// budget, the default [`FileRules`], no scope terms, no embeddings.
#[non_exhaustive]
pub struct BuildOptions<'a> {
    /// Worker threads for chunking and accumulating postings.
    pub threads: usize,
    /// Bytes of postings held in memory before spilling a sorted run.
    pub mem_budget: usize,
    pub rules: FileRules,
    /// Also index the words of each chunk's scope (`impl RetryPolicy › fn backoff`,
    /// a Markdown heading chain) for lexical search, beside its path and source.
    /// Off by default.
    pub lexical_scope: bool,
    /// Embed the chunks too (dense search). `None`: lexical only. Opt-in: embedding
    /// calls the embedder's provider, which may bill for it.
    pub dense: Option<DenseBuild<'a>>,
}
impl Default for BuildOptions<'_> {
    fn default() -> Self {
        Self {
            threads: crate::threads(),
            mem_budget: 256 << 20,
            rules: FileRules::default(),
            lexical_scope: false,
            dense: None,
        }
    }
}
impl<'a> BuildOptions<'a> {
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
    pub fn with_mem_budget(mut self, bytes: usize) -> Self {
        self.mem_budget = bytes;
        self
    }
    pub fn with_rules(mut self, rules: FileRules) -> Self {
        self.rules = rules;
        self
    }
    pub fn with_lexical_scope(mut self, lexical_scope: bool) -> Self {
        self.lexical_scope = lexical_scope;
        self
    }
    pub fn with_dense(mut self, dense: DenseBuild<'a>) -> Self {
        self.dense = Some(dense);
        self
    }
}

pub fn index(
    root: &Path,
    state_dir: &Path,
    options: &Options,
    threads: usize,
    mem_budget: usize,
) -> Result<BuildStats, Error> {
    index_with(
        root,
        state_dir,
        options,
        threads,
        mem_budget,
        &FileRules::default(),
    )
}

/// [`index`] with explicit file selection rules.
pub fn index_with(
    root: &Path,
    state_dir: &Path,
    options: &Options,
    threads: usize,
    mem_budget: usize,
    rules: &FileRules,
) -> Result<BuildStats, Error> {
    build(
        root,
        state_dir,
        options,
        &BuildOptions {
            threads,
            mem_budget,
            rules: rules.clone(),
            lexical_scope: false,
            dense: None,
        },
    )
}

/// Builds a new generation and publishes it. With [`BuildOptions::dense`] the chunks
/// are also embedded: texts the state's embedding cache holds for the embedder's
/// model are reused, the others are embedded (stopping cleanly at the token cap), the
/// vector store of the model and settings is brought up to date, and the settings are
/// saved as the index's [`DenseConfig`]. Without it, a saved config is left as it is.
pub fn build(
    root: &Path,
    state_dir: &Path,
    options: &Options,
    settings: &BuildOptions<'_>,
) -> Result<BuildStats, Error> {
    let (threads, mem_budget, rules) = (settings.threads, settings.mem_budget, &settings.rules);
    let dense = settings.dense.as_ref();
    let doc_text = DocText {
        lexical_scope: settings.lexical_scope,
        embed_text: dense.map_or(EmbedText::default(), |d| d.embed_text),
    };
    rules.validate()?;
    let root = root.canonicalize()?;
    if !root.is_dir() {
        return Err("index root must be a directory".into());
    }
    std::fs::create_dir_all(state_dir)?;
    let state_dir = state_dir.canonicalize()?;
    if root.starts_with(&state_dir) {
        return Err("index state directory may not contain the source root".into());
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(state_dir.join(".index.lock"))?;
    lock.try_lock_exclusive()
        .map_err(|e| format!("index state already in use: {e}"))?;
    File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(state_dir.join(".arko-index-state"))?
        .sync_all()?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let generation = format!("index2-{stamp}-{}", std::process::id());
    let dir = state_dir.join(&generation);
    std::fs::create_dir(&dir)?;
    let mut unpublished = Unpublished {
        state_dir: state_dir.clone(),
        generation: Some(generation.clone()),
    };
    let w = |name: &str| -> Result<BufWriter<File>, Error> {
        Ok(BufWriter::with_capacity(
            1 << 20,
            File::create(dir.join(name))?,
        ))
    };
    let mut docs_dat = w("docs.dat")?;
    let mut docs_off = w("docs.off")?;
    let mut lens = w("lens.dat")?;
    let mut files_dat = w("files.dat")?;
    // (first doc, path) of every file with chunks, for the vector keys.
    let mut file_table: Vec<(u32, String)> = Vec::new();
    // Embeddings: the texts the model's cache holds (read only: embedding runs write
    // it), each doc's text hash (in doc order), the texts to embed, and the distinct
    // hashes seen.
    let cached = match dense {
        Some(d) => Some(cached_hashes(&state_dir, &d.embedder.descriptor())?),
        None => None,
    };
    let mut vhash = match dense {
        Some(_) => Some(w(VHASH)?),
        None => None,
    };
    let mut spool = match dense {
        Some(_) => Some(run::Spool::create(&dir.join(SPOOL))?),
        None => None,
    };
    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut pending: HashSet<[u8; 32]> = HashSet::new();
    let (mut offset, mut next_doc, mut total) = (0u64, 0u32, 0u64);
    let clock = std::time::Instant::now();
    let mut phases: Vec<(&str, f64)> = Vec::new();
    let files::Walked {
        files: listed,
        mut report,
    } = files::list_files(&root, Some(&state_dir), rules)?;
    let weights = par::source_weights(&listed, threads);
    phases.push(("walk_s", clock.elapsed().as_secs_f64()));
    // Postings are accumulated by `shards` threads, each owning the terms that hash to
    // it, so no single thread touches every posting. One thread: everything inline.
    let shards = if threads > 1 { threads } else { 1 };
    let new_shard = |id: usize| Shard {
        dir: dir.clone(),
        id,
        terms: FxMap::default(),
        mem: 0,
        budget: (mem_budget / shards).max(1),
        runs: Vec::new(),
    };
    let mut inline = new_shard(0);
    let mut runs: Vec<PathBuf> = Vec::new();
    std::thread::scope(|scope| -> Result<(), Error> {
        let mut senders = Vec::new();
        let mut handles = Vec::new();
        if shards > 1 {
            for id in 0..shards {
                // Bounded: a slow shard applies back-pressure instead of buffering postings.
                let (tx, rx) = std::sync::mpsc::sync_channel::<(u32, Vec<u8>)>(64);
                senders.push(tx);
                let mut shard = new_shard(id);
                handles.push(scope.spawn(move || -> Result<Vec<PathBuf>, Error> {
                    for (base, buf) in rx {
                        shard.ingest(base, &buf)?;
                    }
                    shard.spill()?;
                    Ok(shard.runs)
                }));
            }
        }
        let fed = par::ordered_map_weighted(
            &listed,
            threads,
            par::window_bytes(threads),
            |i| weights.get(i).copied().unwrap_or(1),
            |item| file_docs(item, options, shards, cached.as_ref(), doc_text),
            |f| {
                match f.skip {
                    Some(skip) => {
                        report.skipped.extend(skip);
                        return Ok(());
                    }
                    None => report.files += 1,
                }
                report.warnings.extend(f.warning);
                let base = next_doc;
                let mut start = 0u64;
                for (&end, &len) in f.ends.iter().zip(&f.lens) {
                    docs_off.write_all(&offset.to_le_bytes())?;
                    offset += end - start;
                    start = end;
                    lens.write_all(&len.to_le_bytes())?;
                    total += u64::from(len);
                }
                docs_dat.write_all(&f.json)?;
                if !f.ends.is_empty() {
                    files_dat.write_all(&base.to_le_bytes())?;
                    files_dat.write_all(&(f.path.len() as u32).to_le_bytes())?;
                    files_dat.write_all(f.path.as_bytes())?;
                    if dense.is_some() {
                        file_table.push((base, f.path.clone()));
                    }
                }
                if let (Some(vhash), Some(spool)) = (&mut vhash, &mut spool) {
                    for h in &f.hashes {
                        vhash.write_all(h)?;
                        seen.insert(*h);
                    }
                    for (h, text) in &f.missing {
                        if pending.insert(*h) {
                            spool.push(h, text)?;
                        }
                    }
                }
                next_doc = u32::try_from(f.ends.len())
                    .ok()
                    .and_then(|n| next_doc.checked_add(n))
                    .ok_or("too many documents")?;
                for (id, buf) in f.shards.into_iter().enumerate() {
                    if buf.is_empty() {
                        continue;
                    }
                    if shards == 1 {
                        inline.ingest(base, &buf)?;
                    } else if senders[id].send((base, buf)).is_err() {
                        // The shard stopped on an error; its join below reports it.
                        return Err("index shard stopped".into());
                    }
                }
                Ok(())
            },
        );
        drop(senders);
        // A shard's own error explains a failed send better than the send does.
        let mut first_err = None;
        for h in handles {
            match h
                .join()
                .map_err(|_| -> Error { "index shard panicked".into() })
            {
                Ok(Ok(r)) => runs.extend(r),
                Ok(Err(e)) | Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        first_err.or(fed.err()).map_or(Ok(()), Err)
    })?;
    if shards == 1 {
        inline.spill()?;
        runs = std::mem::take(&mut inline.runs);
    }
    report.finish();
    phases.push(("chunk_tokenize_accumulate_s", clock.elapsed().as_secs_f64()));
    let n = u64::from(next_doc);
    docs_off.write_all(&offset.to_le_bytes())?;
    // Written now, made durable after the merge (before publishing), so their
    // writeback overlaps the merge instead of blocking it.
    for f in [&mut docs_dat, &mut docs_off, &mut lens, &mut files_dat] {
        f.flush()?;
    }
    // Merge runs by term; runs hold increasing doc ranges, so a term's postings are
    // the concatenation of its run segments, re-basing each segment's first delta.
    // `runs` lists each shard's runs in creation order, and a term lives in one shard.
    let mut readers = Vec::new();
    let mut heap: BinaryHeap<std::cmp::Reverse<(String, usize)>> = BinaryHeap::new();
    for (i, p) in runs.iter().enumerate() {
        let mut r = RunReader {
            r: BufReader::with_capacity(256 << 10, File::open(p)?),
            head: None,
        };
        if let Some(t) = r.advance()? {
            heap.push(std::cmp::Reverse((t, i)));
        }
        readers.push(r);
    }
    let mut terms_dat = w("terms.dat")?;
    let mut terms_idx = w("terms.idx")?;
    let mut post = w("post.dat")?;
    let (mut terms_off, mut post_off, mut nterms) = (0u64, 0u64, 0u64);
    let mut buf = Vec::new();
    let mut entry = Vec::new();
    let mut same = Vec::new();
    while let Some(std::cmp::Reverse((term, first_run))) = heap.pop() {
        buf.clear();
        let mut df = 0u32;
        let mut prev_last: Option<u32> = None;
        // Same term in several runs: take them in run (= doc) order.
        same.clear();
        same.push(first_run);
        while heap
            .peek()
            .is_some_and(|std::cmp::Reverse((t, _))| *t == term)
        {
            let std::cmp::Reverse((_, i)) = heap.pop().unwrap();
            same.push(i);
        }
        same.sort_unstable();
        for &i in &same {
            let (first, last, sdf, bytes) = readers[i].head.take().unwrap();
            put_varint(&mut buf, u64::from(first - prev_last.unwrap_or(0)));
            buf.extend_from_slice(&bytes);
            prev_last = Some(last);
            df += sdf;
            if let Some(t) = readers[i].advance()? {
                heap.push(std::cmp::Reverse((t, i)));
            }
        }
        if (nterms as usize).is_multiple_of(BLOCK) {
            terms_idx.write_all(&(term.len() as u32).to_le_bytes())?;
            terms_idx.write_all(term.as_bytes())?;
            terms_idx.write_all(&terms_off.to_le_bytes())?;
        }
        entry.clear();
        entry.extend_from_slice(&(term.len() as u32).to_le_bytes());
        entry.extend_from_slice(term.as_bytes());
        entry.extend_from_slice(&df.to_le_bytes());
        entry.extend_from_slice(&post_off.to_le_bytes());
        entry.extend_from_slice(&(buf.len() as u32).to_le_bytes());
        terms_dat.write_all(&entry)?;
        terms_off += entry.len() as u64;
        post.write_all(&buf)?;
        post_off += buf.len() as u64;
        nterms += 1;
    }
    phases.push(("merged_s", clock.elapsed().as_secs_f64()));
    for f in [&mut terms_dat, &mut terms_idx, &mut post] {
        f.flush()?;
    }
    for f in [
        &docs_dat, &docs_off, &lens, &files_dat, &terms_dat, &terms_idx, &post,
    ] {
        f.get_ref().sync_all()?;
    }
    for p in &runs {
        std::fs::remove_file(p)?;
    }
    // Embeddings: every doc's vector key, and the coverage the store already gives
    // this generation. The store is only read here: embedding runs write it.
    let mut dense_stats = None;
    if let (Some(d), Some(cached), Some(vhash), Some(spool)) =
        (dense, cached.as_ref(), vhash.take(), spool.take())
    {
        vhash.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        spool.finish()?;
        let descriptor = d.embedder.descriptor();
        let keys = vector_keys(&File::open(dir.join(VHASH))?, next_doc, &file_table)?;
        write_vector_keys(&dir, &keys)?;
        let config = StoreConfig::new(
            descriptor.clone(),
            text_format(RENDER_VERSION, d.embed_text),
        )
        .with_quantization(d.quantization)
        .with_index(d.vector_index);
        let store = config.dir_name()?;
        let embedded = stored_count(&state_dir.join(&store), &config, &keys);
        let info = DenseIndexInfo {
            version: DenseIndexInfo::VERSION,
            embedder: descriptor.clone(),
            render_version: RENDER_VERSION,
            embed_text: d.embed_text,
            quantization: d.quantization,
            vector_index: d.vector_index,
            store,
            docs: n,
            embedded,
            state: if embedded == n {
                EmbedState::Complete
            } else {
                EmbedState::Embedding
            },
        };
        atomic_write(&dir, "dense.json", &serde_json::to_vec_pretty(&info)?)?;
        let mut stats = EmbedStats::new(descriptor, d.embed_text, d.max_tokens);
        stats.docs = n;
        stats.embedded_docs = embedded;
        stats.texts = seen.len() as u64;
        stats.cached = (seen.len() - pending.len()) as u64;
        stats.cache_records = cached.len() as u64;
        dense_stats = Some(stats);
        phases.push(("vector_keys_s", clock.elapsed().as_secs_f64()));
    }
    let average = if n > 0 { total as f64 / n as f64 } else { 1.0 };
    let meta = Meta {
        version: 2,
        docs: n,
        total,
        average_bits: average.to_bits(),
        terms: nterms,
        dense: dense_stats.is_some(),
    };
    atomic_write(&dir, "meta.json", &serde_json::to_vec(&meta)?)?;
    crate::files::sync_dir(&state_dir)?;
    atomic_write(&state_dir, "CURRENT2", generation.as_bytes())?;
    unpublished.generation = None;
    phases.push(("merge_publish_s", clock.elapsed().as_secs_f64()));
    if std::env::var_os("ARKO_INDEX_STATS").is_some() {
        // Cumulative seconds since the start of the build, per phase end.
        eprintln!(
            "{}",
            serde_json::json!({ "index2_phases": phases, "shards": shards })
        );
    }
    // Older generations: readers that already opened their files keep reading them. One
    // that cannot be removed now (an embedding run still reading it, on a platform that
    // keeps open files) goes with a later build.
    for e in std::fs::read_dir(&state_dir)? {
        let e = e?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("index2-") && name != generation.as_str() {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
    if let Some(d) = dense {
        DenseConfig::new(d.embedder.descriptor())
            .with_quantization(d.quantization)
            .with_vector_index(d.vector_index)
            .with_max_tokens_per_run(d.max_tokens)
            .with_embed_text(d.embed_text)
            .save(&state_dir)?;
    }
    files::write_report(&state_dir, &report)?;
    // The embedding cache is compacted only here, with no other build reading it (this
    // one holds the index lock) and no run writing it: a run in progress keeps it as
    // it is, and a later build compacts it.
    if let (Some(d), Some(stats)) = (dense, dense_stats.as_mut()) {
        let embed_lock = lock_file(&state_dir, EMBED_LOCK)?;
        if embed_lock.try_lock_exclusive().is_ok() {
            let mut cache = Cache::open(&state_dir, &d.embedder.descriptor())?;
            cache.compact_if_sparse(&seen)?;
            stats.cache_records = cache.records();
        }
    }
    // Another build may start while this one's embedding runs.
    drop(lock);
    let dense_stats = match (dense, dense_stats) {
        (Some(d), Some(published)) if !d.embed_later => {
            let run = embed_run(&state_dir, d, Some(&generation)).map_err(|e| {
                format!(
                    "{e} (the build published its generation, so lexical search works, and a later embedding run continues where this one stopped)"
                )
            })?;
            // Superseded before it began: what the build published.
            Some(if run.superseded && run.docs == 0 {
                EmbedStats {
                    superseded: true,
                    ..published
                }
            } else {
                run
            })
        }
        (None, _) => {
            // A lexical generation: the vector stores go once no run writes them, unless
            // a newer generation was published meanwhile (its run keeps its own store).
            let embed_lock = lock_file(&state_dir, EMBED_LOCK)?;
            embed_lock.lock_exclusive()?;
            if current_generation(&state_dir)? == generation {
                remove_other_stores(&state_dir, None)?;
            }
            None
        }
        (_, published) => published,
    };
    Ok(BuildStats {
        docs: n,
        terms: nterms,
        runs: runs.len(),
        report,
        dense: dense_stats,
    })
}

/// The embedding lock of a state directory, held by an embedding run (and by a
/// lexical build while it removes the vector stores). Index builds take `.index.lock`.
const EMBED_LOCK: &str = ".embed.lock";
/// The coverage of the generation an embedding run is embedding, written after each
/// commit of the vector store.
const PROGRESS: &str = "embed-progress.json";
/// A generation's text hash per doc, and the texts its build found missing from the
/// cache.
const VHASH: &str = "vhash.dat";
const SPOOL: &str = "embed.spool";
/// Vectors a run adds from the cache between commits of the store.
const REFILL_COMMIT: usize = 16 * 1024;

fn lock_file(state_dir: &Path, name: &str) -> Result<File, Error> {
    Ok(OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(state_dir.join(name))?)
}

/// The name of the current generation (`CURRENT2`).
fn current_generation(state_dir: &Path) -> Result<String, Error> {
    Ok(std::fs::read_to_string(state_dir.join("CURRENT2"))?
        .trim()
        .to_owned())
}

/// `embed-progress.json`: the latest run's generation, its state and figures, and why
/// it was stopped, when its stop signal ended it.
#[derive(Serialize, Deserialize)]
struct RunRecord {
    version: u32,
    generation: String,
    state: EmbedState,
    progress: EmbedProgress,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stopped: Option<String>,
}

fn read_progress(state_dir: &Path) -> Option<RunRecord> {
    let bytes = std::fs::read(state_dir.join(PROGRESS)).ok()?;
    serde_json::from_slice::<RunRecord>(&bytes)
        .ok()
        .filter(|r| r.version == 1)
}

fn write_progress(
    state_dir: &Path,
    generation: &str,
    state: EmbedState,
    progress: &EmbedProgress,
    stopped: Option<String>,
) -> Result<(), Error> {
    let record = RunRecord {
        version: 1,
        generation: generation.to_owned(),
        state,
        progress: progress.clone(),
        stopped,
    };
    atomic_write(state_dir, PROGRESS, &serde_json::to_vec(&record)?)
}

/// Whether an embedding run holds the state directory's embedding lock.
fn embedding_active(state_dir: &Path) -> bool {
    let Ok(file) = File::open(state_dir.join(EMBED_LOCK)) else {
        return false;
    };
    match FileExt::try_lock_shared(&file) {
        Ok(()) => {
            let _ = FileExt::unlock(&file);
            false
        }
        Err(_) => true,
    }
}

/// How many of `keys` the store in `dir` holds, read only: 0 when it is missing,
/// unreadable or of other settings (an embedding run then replaces it).
fn stored_count(dir: &Path, config: &StoreConfig, keys: &[(VectorId, u32)]) -> u64 {
    let Ok(store) = CaprockStore::open(dir) else {
        return 0;
    };
    if !store.matches(config) {
        return 0;
    }
    let Ok(ids) = store.ids() else {
        return 0;
    };
    let (missing, _) = diff_ids(keys, &ids);
    (keys.len() - missing.len()) as u64
}

/// Embeds what the current generation of `state_dir` lacks and brings the vector store
/// up to date for it, with the generation's model and settings. Waits for a run already
/// in progress to finish, or to stop because this call's generation is newer. Then the
/// store's vectors of chunks the generation lacks are removed, the vectors the
/// embedding cache holds are added, and the remaining texts are embedded request by
/// request: each request's vectors are stored and committed at once, so dense and
/// hybrid search use them while the run goes on.
///
/// The run returns when every chunk has a vector (except texts over the model's input
/// limit), at its token cap ([`DenseBuild::max_tokens`]), when [`DenseBuild::stop`] is
/// set ([`EmbedStats::stopped`]), or when a newer generation is published
/// ([`EmbedStats::superseded`]). What it embedded stays in the cache, so a later run
/// continues without paying for it again. Only the embedder, the cap, the progress
/// callback and the stop flag of `dense` apply: the quantization, vector index and
/// embedded text are the generation's. An error when the generation has no embeddings
/// or embeds with another model.
pub fn embed(state_dir: &Path, dense: &DenseBuild<'_>) -> Result<EmbedStats, Error> {
    embed_run(state_dir, dense, None)
}

/// [`embed`], or with `expect` only that generation: superseded when another is current.
fn embed_run(
    state_dir: &Path,
    d: &DenseBuild<'_>,
    expect: Option<&str>,
) -> Result<EmbedStats, Error> {
    let descriptor = d.embedder.descriptor();
    let mut stats = EmbedStats::new(descriptor.clone(), d.embed_text, d.max_tokens);
    let lock = lock_file(state_dir, EMBED_LOCK)?;
    lock.lock_exclusive()?;
    let name = current_generation(state_dir)?;
    if expect.is_some_and(|g| g != name) {
        stats.superseded = true;
        return Ok(stats);
    }
    let superseded = || current_generation(state_dir).ok().as_deref() != Some(name.as_str());
    // A build may publish a newer generation and remove this one before the run has
    // read it: the run is superseded, not failed.
    let read = || -> Result<_, Error> {
        let generation = Generation::open_at(state_dir, &name)?;
        let Some(info) = generation.stored_dense_info()? else {
            return Ok((generation, None));
        };
        let dir = &generation.dir;
        let files = (read_hashes(dir)?, read_vector_keys(dir)?, read_files(dir)?);
        Ok((generation, Some((info, files))))
    };
    let (generation, read) = match read() {
        Err(_) if superseded() => {
            stats.superseded = true;
            return Ok(stats);
        }
        read => read?,
    };
    let (info, (hashes, keys, files)) = read.ok_or(
        "the index's current generation has no embeddings. Build it with them (index2 --embed)",
    )?;
    check_dense_format(&info)?;
    if info.embedder != descriptor {
        return Err(format!(
            "the index embeds with {}, not {}",
            info.embedder.display(),
            descriptor.display()
        )
        .into());
    }
    stats.embed_text = info.embed_text;
    stats.docs = info.docs;
    let dir = &generation.dir;
    if hashes.len() as u64 != info.docs || keys.len() as u64 != info.docs {
        return Err(
            "the generation's vector keys do not match its chunks. Rebuild the index".into(),
        );
    }
    let live: HashSet<[u8; 32]> = hashes.iter().copied().collect();
    stats.texts = live.len() as u64;
    let mut cache = Cache::open(state_dir, &descriptor)?;
    stats.cached = live.iter().filter(|h| cache.contains(h)).count() as u64;

    // The vectors of chunks the generation lacks go first (the generation is current,
    // so no reader needs them), and stores of other models or settings. A newer
    // generation may still need them: then its run decides.
    let (mut store, created) =
        vector::open_or_create(&state_dir.join(&info.store), &info.store_config())?;
    stats.store_created = created;
    let (missing, stale) = diff_ids(&keys, &store.ids()?);
    if superseded() {
        stats.embedded_docs = (keys.len() - missing.len()) as u64;
        stats.superseded = true;
        return Ok(stats);
    }
    for batch in stale.chunks(UPSERT_BATCH) {
        store.delete(batch)?;
    }
    store.persist()?;
    stats.vectors_removed = stale.len() as u64;
    remove_other_stores(state_dir, Some(&info.store))?;

    // The docs without a vector: those whose text the cache holds get it now, the
    // others when their text is embedded.
    let mut refill = Vec::new();
    let mut by_hash: HashMap<[u8; 32], Vec<(VectorId, u32)>> = HashMap::new();
    for &(id, doc) in &missing {
        let hash = hashes[doc as usize];
        if cache.contains(&hash) {
            refill.push((id, doc));
        } else {
            by_hash.entry(hash).or_default().push((id, doc));
        }
    }
    let mut embedded = (keys.len() - missing.len()) as u64;
    let mut progress = EmbedProgress {
        total: by_hash.len() as u64,
        embedded_docs: embedded,
        docs: info.docs,
        ..EmbedProgress::default()
    };
    let working = if missing.is_empty() {
        EmbedState::Complete
    } else {
        EmbedState::Embedding
    };
    write_progress(state_dir, &name, working, &progress, None)?;
    let (mut was_superseded, mut stopped) = (false, false);
    let stop_requested = || d.stop.is_some_and(|s| s.is_stopped());
    let mut vector = Vec::new();
    for part in refill.chunks(REFILL_COMMIT) {
        if stop_requested() {
            stopped = true;
            break;
        }
        if superseded() {
            was_superseded = true;
            break;
        }
        for batch in part.chunks(UPSERT_BATCH) {
            let mut vectors = Vec::with_capacity(batch.len());
            for &(_, doc) in batch {
                if !cache.get(&hashes[doc as usize], &mut vector)? {
                    return Err("an embedding left the cache during the run".into());
                }
                vectors.push(std::mem::take(&mut vector));
            }
            let records: Vec<VectorRecord<'_>> = batch
                .iter()
                .zip(&vectors)
                .map(|(&(id, doc), v)| record(&files, id, doc, v))
                .collect();
            store.upsert(&records)?;
        }
        store.persist()?;
        embedded += part.len() as u64;
        stats.vectors_added += part.len() as u64;
        progress.embedded_docs = embedded;
        write_progress(state_dir, &name, working, &progress, None)?;
    }

    // The texts no cache holds, embedded request by request from the spool.
    let mut totals = run::RunTotals::default();
    if !by_hash.is_empty() && !was_superseded && !stopped {
        let spool = match File::open(dir.join(SPOOL)) {
            Ok(spool) => Some(spool),
            // Gone with the generation: superseded. Gone otherwise: its texts are
            // counted as left below.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        if stop_requested() {
            stopped = true;
        } else if superseded() {
            was_superseded = true;
        } else if let Some(spool) = spool {
            let wanted: HashSet<[u8; 32]> = by_hash.keys().copied().collect();
            {
                totals = run::embed_spool(
                    spool,
                    &|h| wanted.contains(h),
                    &stop_requested,
                    &mut cache,
                    d.embedder,
                    d.max_tokens,
                    &mut |batch, totals| {
                        let mut docs = Vec::new();
                        for (i, (hash, _)) in batch.iter().enumerate() {
                            for (id, doc) in by_hash.remove(hash).unwrap_or_default() {
                                docs.push((id, doc, i));
                            }
                        }
                        let records: Vec<VectorRecord<'_>> = docs
                            .iter()
                            .map(|&(id, doc, i)| record(&files, id, doc, &batch[i].1))
                            .collect();
                        for part in records.chunks(UPSERT_BATCH) {
                            store.upsert(part)?;
                        }
                        store.persist()?;
                        embedded += docs.len() as u64;
                        stats.vectors_added += docs.len() as u64;
                        progress.texts = totals.new;
                        progress.tokens = totals.tokens;
                        progress.requests = totals.requests;
                        progress.embedded_docs = embedded;
                        write_progress(state_dir, &name, EmbedState::Embedding, &progress, None)?;
                        if let Some(callback) = d.progress {
                            callback(&progress);
                        }
                        if stop_requested() {
                            stopped = true;
                            return Ok(Flow::Stop);
                        }
                        if superseded() {
                            was_superseded = true;
                            return Ok(Flow::Stop);
                        }
                        Ok(Flow::Continue)
                    },
                )?;
            }
        }
    }
    stats.new = totals.new;
    stats.requests = totals.requests;
    stats.tokens = totals.tokens;
    stats.cap_reached = totals.cap_reached;
    stats.not_embedded = totals.not_embedded;
    stats.not_embedded_tokens = totals.not_embedded_tokens;
    stats.too_long = totals.too_long;
    stats.superseded = was_superseded;
    stats.stopped = stopped;
    let state = if was_superseded || stopped {
        EmbedState::Embedding
    } else if totals.cap_reached {
        EmbedState::CapReached
    } else {
        // Left now: texts over the model's limit, and any the spool lacks (its build
        // found them in a cache that has since lost them; a rebuild embeds them).
        stats.not_embedded += (by_hash.len() as u64).saturating_sub(totals.too_long);
        EmbedState::Complete
    };
    if created && state != EmbedState::Embedding {
        store.compact()?;
        store.persist()?;
    }
    drop(store);
    stats.embedded_docs = embedded;
    let reason = stopped.then(|| {
        d.stop
            .and_then(StopSignal::reason)
            .unwrap_or_else(|| "stopped".into())
    });
    write_progress(state_dir, &name, state, &progress, reason)?;
    if state == EmbedState::Complete && !superseded() {
        let _ = std::fs::remove_file(dir.join(SPOOL));
    }
    stats.cache_records = cache.records();
    Ok(stats)
}

/// The store record of `doc`'s vector.
fn record<'a>(
    files: &'a [(u32, String)],
    id: VectorId,
    doc: u32,
    vector: &'a [f32],
) -> VectorRecord<'a> {
    let path = file_path(files, doc);
    VectorRecord {
        id,
        vector,
        meta: VectorMeta {
            path,
            language: crate::syntax::language(Path::new(path)),
        },
    }
}

/// Each doc's embedded-text hash (`vhash.dat`).
fn read_hashes(dir: &Path) -> Result<Vec<[u8; 32]>, Error> {
    let raw = std::fs::read(dir.join(VHASH)).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("{e}. This generation was built by an earlier version. Rebuild the index (index2) to embed it"),
        )
    })?;
    if raw.len() % 32 != 0 {
        return Err("truncated text hashes".into());
    }
    Ok(raw
        .chunks_exact(32)
        .map(|h| h.try_into().unwrap())
        .collect())
}

/// (vector id, doc) of every doc, by id (`vkeys.dat`).
fn read_vector_keys(dir: &Path) -> Result<Vec<(VectorId, u32)>, Error> {
    let raw = std::fs::read(dir.join("vkeys.dat"))?;
    if raw.len() % VKEY_BYTES != 0 {
        return Err("truncated vector keys".into());
    }
    Ok(raw
        .chunks_exact(VKEY_BYTES)
        .map(|e| {
            (
                u128::from_be_bytes(e[..16].try_into().unwrap()),
                read_u32(e, 16),
            )
        })
        .collect())
}

/// Vectors per store upsert or delete call.
const UPSERT_BATCH: usize = 1024;

/// Removes every vector store directory but `keep`. Each is first renamed to a name no
/// settings produce (`vectors-removed-...`), so a removal that stops halfway never
/// leaves part of a store under a name a build would open. The next build removes it.
fn remove_other_stores(state_dir: &Path, keep: Option<&str>) -> Result<(), Error> {
    let mut names = Vec::new();
    for e in std::fs::read_dir(state_dir)? {
        let name = e?.file_name().to_string_lossy().into_owned();
        if name.starts_with("vectors-") && keep != Some(name.as_str()) {
            names.push(name);
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    for (i, name) in names.into_iter().enumerate() {
        let mut dir = state_dir.join(&name);
        if !name.starts_with("vectors-removed-") {
            let removed = state_dir.join(format!(
                "vectors-removed-{stamp}-{}-{i}",
                std::process::id()
            ));
            std::fs::rename(&dir, &removed)?;
            dir = removed;
        }
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}
/// Bytes per `vkeys.dat` entry: a u128 vector id (big-endian, so byte order is id
/// order) and a u32 doc.
const VKEY_BYTES: usize = 20;

/// A doc's vector id: a digest of its path, its embedded text's hash, and how many
/// docs of the file have the same text before it. A chunk keeps its id, so its
/// vector, while its path and text stay the same, whatever its doc id.
fn vector_id(path: &str, hash: &[u8; 32], occurrence: u32) -> VectorId {
    let mut h = Sha256::new();
    h.update((path.len() as u64).to_le_bytes());
    h.update(path.as_bytes());
    h.update(hash);
    h.update(occurrence.to_le_bytes());
    u128::from_be_bytes(h.finalize()[..16].try_into().unwrap())
}

/// The path of the file holding `doc`, from the (first doc, path) table.
fn file_path(files: &[(u32, String)], doc: u32) -> &str {
    let i = files.partition_point(|(first, _)| *first <= doc);
    &files[i - 1].1
}

/// (vector id, doc) of every doc, by id. `hashes` holds each doc's text hash, in doc
/// order.
fn vector_keys(
    hashes: &File,
    docs: u32,
    files: &[(u32, String)],
) -> Result<Vec<(VectorId, u32)>, Error> {
    let mut r = BufReader::with_capacity(1 << 20, hashes);
    let mut keys = Vec::new();
    let mut seen: HashMap<[u8; 32], u32> = HashMap::new();
    let (mut file, mut hash) = (0, [0u8; 32]);
    for doc in 0..docs {
        r.read_exact(&mut hash)?;
        while file + 1 < files.len() && files[file + 1].0 <= doc {
            file += 1;
            seen.clear();
        }
        let n = seen.entry(hash).or_insert(0);
        keys.push((vector_id(&files[file].1, &hash, *n), doc));
        *n += 1;
    }
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err("two chunks have the same vector id".into());
    }
    Ok(keys)
}

fn write_vector_keys(dir: &Path, keys: &[(VectorId, u32)]) -> Result<(), Error> {
    let mut out = BufWriter::with_capacity(1 << 20, File::create(dir.join("vkeys.dat"))?);
    for (id, doc) in keys {
        out.write_all(&id.to_be_bytes())?;
        out.write_all(&doc.to_le_bytes())?;
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    Ok(())
}

/// What a store holding `stored` (ascending) needs to hold `keys` (by id): the keys
/// to add, and the ids to remove.
fn diff_ids(
    keys: &[(VectorId, u32)],
    stored: &[VectorId],
) -> (Vec<(VectorId, u32)>, Vec<VectorId>) {
    let (mut added, mut removed) = (Vec::new(), Vec::new());
    let (mut i, mut j) = (0, 0);
    while i < keys.len() || j < stored.len() {
        match (keys.get(i), stored.get(j)) {
            (Some(k), Some(s)) if k.0 == *s => {
                i += 1;
                j += 1;
            }
            (Some(k), Some(s)) if k.0 < *s => {
                added.push(*k);
                i += 1;
            }
            (Some(k), None) => {
                added.push(*k);
                i += 1;
            }
            (_, Some(s)) => {
                removed.push(*s);
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    (added, removed)
}

/// An error for a generation whose vectors an earlier version stored.
pub(crate) fn check_dense_format(info: &DenseIndexInfo) -> Result<(), Error> {
    if info.version == DenseIndexInfo::VERSION {
        return Ok(());
    }
    Err(format!(
        "this index's vectors were stored by an earlier version of arko_index (format {}). Rebuild the index (index2) to search it densely, or search lexically",
        info.version
    )
    .into())
}

/// The sparse term index (`terms.idx`) without a `String` per entry: the raw bytes
/// plus the start of each entry, searched by byte comparison (String order).
struct TermIndex {
    raw: Vec<u8>,
    starts: Vec<u32>,
}
impl TermIndex {
    fn read(path: &Path) -> Result<Self, Error> {
        let raw = std::fs::read(path)?;
        let mut starts = Vec::new();
        let mut at = 0;
        while at < raw.len() {
            starts.push(u32::try_from(at).map_err(|_| "term index too large")?);
            at += 4 + read_u32(&raw, at) as usize + 8;
        }
        if at != raw.len() {
            return Err("truncated term index".into());
        }
        Ok(Self { raw, starts })
    }
    fn term(&self, i: usize) -> &[u8] {
        let at = self.starts[i] as usize;
        &self.raw[at + 4..at + 4 + read_u32(&self.raw, at) as usize]
    }
    fn offset(&self, i: usize) -> u64 {
        let at = self.starts[i] as usize;
        read_u64(&self.raw, at + 4 + read_u32(&self.raw, at) as usize)
    }
}

/// Looks up `(df, postings offset, postings len)` for one term.
fn lookup(
    idx: &TermIndex,
    terms: &File,
    terms_len: u64,
    term: &str,
) -> Result<Option<(u32, u64, u32)>, Error> {
    let n = idx.starts.len();
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if idx.term(mid) <= term.as_bytes() {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let pos = lo;
    if pos == 0 {
        return Ok(None);
    }
    let start = idx.offset(pos - 1);
    let end = if pos < n { idx.offset(pos) } else { terms_len };
    let mut block = vec![0u8; (end - start) as usize];
    read_exact_at(terms, &mut block, start)?;
    let mut at = 0;
    while at < block.len() {
        let n = read_u32(&block, at) as usize;
        let t = &block[at + 4..at + 4 + n];
        at += 4 + n;
        if t == term.as_bytes() {
            return Ok(Some((
                read_u32(&block, at),
                read_u64(&block, at + 4),
                read_u32(&block, at + 12),
            )));
        }
        at += 16;
    }
    Ok(None)
}

/// Per-document score sums: a dense array when the query's postings cover a large
/// share of the documents, else a map of the touched documents only. Either way
/// each document's sum is accumulated in the same order, so scores are identical.
enum Scores {
    Dense(Vec<f64>, Vec<u32>),
    Sparse(FxMap<u32, f64>),
}
impl Scores {
    #[inline]
    fn add(&mut self, d: u32, v: f64) {
        match self {
            Scores::Dense(s, touched) => {
                let x = &mut s[d as usize];
                if *x == 0.0 {
                    touched.push(d);
                }
                *x += v;
            }
            Scores::Sparse(m) => *m.entry(d).or_insert(0.0) += v,
        }
    }
    /// The `k` best (doc, score) among the docs `allow` accepts, by score desc, doc
    /// asc, without collecting all touched documents first.
    fn top(self, k: usize, allow: &dyn Fn(u32) -> bool) -> Vec<(u32, f64)> {
        // Doc ids are assigned in (path, start_byte) order, so the original tie-break
        // (score desc, path asc, start_byte asc) is score desc, doc id asc.
        let cmp = |a: &(u32, f64), b: &(u32, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        let mut best: Vec<(u32, f64)> = Vec::with_capacity(k + 1);
        if k == 0 {
            return best;
        }
        let mut offer = |p: (u32, f64)| {
            if (best.len() < k || cmp(&p, &best[k - 1]).is_lt()) && allow(p.0) {
                let at = best.partition_point(|b| cmp(b, &p).is_lt());
                best.insert(at, p);
                best.truncate(k);
            }
        };
        match self {
            Scores::Dense(s, touched) => {
                touched.into_iter().for_each(|d| offer((d, s[d as usize])))
            }
            Scores::Sparse(m) => m.into_iter().for_each(offer),
        }
        best
    }
}

/// The first doc and the path of every file with chunks (`files.dat`).
fn read_files(dir: &Path) -> Result<Vec<(u32, String)>, Error> {
    // The kind stays: a search that finds the file gone (a build removed the
    // generation) starts over.
    let raw = std::fs::read(dir.join("files.dat")).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!(
                "{e}. This index has no file table (built by an earlier version). Rebuild it to filter"
            ),
        )
    })?;
    let mut c = crate::fsio::Cursor::new(&raw);
    let mut out = Vec::new();
    while c.at < raw.len() {
        let first = c.u32()?;
        out.push((first, c.string()?));
    }
    Ok(out)
}

/// The current generation of an index2 state directory, opened for reading: every
/// read goes to the generation `CURRENT2` named when it was opened, so lexical and
/// dense results of one search come from the same build.
pub(crate) struct Generation {
    state_dir: PathBuf,
    name: String,
    dir: PathBuf,
    meta: Meta,
}
impl Generation {
    pub(crate) fn open(state_dir: &Path) -> Result<Self, Error> {
        Self::open_at(state_dir, &current_generation(state_dir)?)
    }

    /// The generation named `name`.
    fn open_at(state_dir: &Path, name: &str) -> Result<Self, Error> {
        let dir = state_dir.join(name);
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        if meta.version != 2 {
            return Err("unsupported index version".into());
        }
        Ok(Self {
            state_dir: state_dir.to_path_buf(),
            name: name.to_owned(),
            dir,
            meta,
        })
    }

    /// The embeddings of this generation, if it has them, with their coverage now: the
    /// latest embedding run's, when it ran on this generation. Read before the store
    /// is opened, since a run records the coverage after it commits the store.
    pub(crate) fn dense_info(&self) -> Result<Option<DenseIndexInfo>, Error> {
        let mut info = self.stored_dense_info()?;
        if let Some(info) = &mut info
            && let Some(run) = read_progress(&self.state_dir)
            && run.generation == self.name
        {
            info.embedded = run.progress.embedded_docs;
            info.state = run.state;
        }
        Ok(info)
    }

    /// The generation's `dense.json`, as its build wrote it.
    fn stored_dense_info(&self) -> Result<Option<DenseIndexInfo>, Error> {
        match std::fs::read(self.dir.join("dense.json")) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            // Missing from a generation built with embeddings: a build removed the
            // generation, and the error makes a search start over on the new one.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && !self.meta.dense => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// The vector store of this generation's embeddings, read-only. It holds the
    /// vectors the generation's coverage counts, and can hold others: those of chunks
    /// an older generation had, until the run for this one removes them.
    pub(crate) fn open_store(&self, info: &DenseIndexInfo) -> Result<CaprockStore, Error> {
        check_dense_format(info)?;
        if !info.store.starts_with("vectors-") || info.store.contains(['/', '\\']) {
            return Err(format!("bad vector store name {:?}", info.store).into());
        }
        let dir = self.state_dir.join(&info.store);
        let store = CaprockStore::open(&dir)
            .map_err(|e| format!("{e} (rebuild the index to restore its vector store)"))?;
        if !store.matches(&info.store_config()) {
            return Err(format!(
                "the vector store {} does not hold this index's model and settings ({}). Rebuild the index",
                dir.display(),
                info.embedder.display()
            )
            .into());
        }
        // A replacement store (a run repairing an unusable one) is empty until the run
        // fills it and records its coverage: refuse it rather than find nothing. A run
        // that removed vectors of this generation works for a newer one, and the
        // search starts over on that.
        if (store.len() as u64) < info.embedded {
            return Err(format!(
                "the vector store {} holds {} of this index's {} vectors (an embedding run is replacing it, or one stopped while it did). Rebuild the index",
                dir.display(),
                store.len(),
                info.embedded
            )
            .into());
        }
        Ok(store)
    }

    /// The doc of each vector id (`None` for an id this generation lacks).
    pub(crate) fn docs_of(&self, ids: &[VectorId]) -> Result<Vec<Option<u32>>, Error> {
        let file = File::open(self.dir.join("vkeys.dat"))?;
        let n = file.metadata()?.len() / VKEY_BYTES as u64;
        let mut entry = [0u8; VKEY_BYTES];
        ids.iter()
            .map(|&id| {
                let (mut lo, mut hi) = (0, n);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    read_exact_at(&file, &mut entry, mid * VKEY_BYTES as u64)?;
                    let at = u128::from_be_bytes(entry[..16].try_into().unwrap());
                    match at.cmp(&id) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => return Ok(Some(read_u32(&entry, 16))),
                    }
                }
                Ok(None)
            })
            .collect()
    }

    /// The best `k` docs for `query` by BM25 among those whose file passes `filter`.
    pub(crate) fn lexical(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<(u32, f64)>, Error> {
        let meta = &self.meta;
        let dir = &self.dir;
        let average = f64::from_bits(meta.average_bits);
        let n_docs = meta.docs as f64;
        let idx = TermIndex::read(&dir.join("terms.idx"))?;
        let terms = File::open(dir.join("terms.dat"))?;
        let terms_len = terms.metadata()?.len();
        let query: BTreeSet<_> = tokens(query).into_iter().collect();
        // Same accumulation order per document as search::search: query terms ascending.
        let mut found = Vec::new();
        for t in &query {
            if let Some(hit) = lookup(&idx, &terms, terms_len, t)? {
                found.push(hit);
            }
        }
        if found.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        // A filter is applied per file: (first doc, passes) of each file, by doc.
        let allowed: Option<Vec<(u32, bool)>> = if filter.is_empty() {
            None
        } else {
            Some(
                read_files(dir)?
                    .into_iter()
                    .map(|(first, path)| {
                        let language = crate::syntax::language(Path::new(&path));
                        (first, filter.matches(&path, language))
                    })
                    .collect(),
            )
        };
        let allow = |doc: u32| -> bool {
            allowed.as_ref().is_none_or(|files| {
                let i = files.partition_point(|(first, _)| *first <= doc);
                i > 0 && files[i - 1].1
            })
        };
        let post = File::open(dir.join("post.dat"))?;
        let lens_raw = std::fs::read(dir.join("lens.dat"))?;
        let postings: u64 = found.iter().map(|&(df, _, _)| u64::from(df)).sum();
        let mut scores = if postings.saturating_mul(8) >= meta.docs {
            Scores::Dense(vec![0.0; meta.docs as usize], Vec::new())
        } else {
            Scores::Sparse(FxMap::default())
        };
        let mut buf = Vec::new();
        for (df, off, len) in found {
            buf.resize(len as usize, 0);
            read_exact_at(&post, &mut buf, off)?;
            let df = df as f64;
            let idf = (1.0 + (n_docs - df + 0.5) / (df + 0.5)).ln();
            let mut pos = 0;
            let mut doc = 0u64;
            while pos < buf.len() {
                doc += get_varint(&buf, &mut pos)?;
                let tf = get_varint(&buf, &mut pos)?;
                let d = doc as usize;
                let dlen = read_u32(&lens_raw, d * 4);
                let norm = 1.2 * (0.25 + 0.75 * dlen as f64 / average.max(1.0));
                scores.add(d as u32, idf * tf as f64 * 2.2 / (tf as f64 + norm));
            }
        }
        Ok(scores.top(k, &allow))
    }

    /// The chunks of `docs`, in the given order.
    pub(crate) fn chunks(&self, docs: &[u32]) -> Result<Vec<Chunk>, Error> {
        let offs = File::open(self.dir.join("docs.off"))?;
        let store = File::open(self.dir.join("docs.dat"))?;
        let mut out = Vec::with_capacity(docs.len());
        for &d in docs {
            if u64::from(d) >= self.meta.docs {
                return Err(format!("doc {d} is not in the index").into());
            }
            let mut o = [0u8; 16];
            read_exact_at(&offs, &mut o, u64::from(d) * 8)?;
            let (s, e) = (read_u64(&o, 0), read_u64(&o, 8));
            let mut j = vec![0u8; (e - s) as usize];
            read_exact_at(&store, &mut j, s)?;
            out.push(serde_json::from_slice(&j)?);
        }
        Ok(out)
    }
}

/// The embeddings of the current generation (model, render version, coverage), or
/// `None` when it was built without them. The coverage grows while the generation is
/// embedded.
pub fn dense_info(state_dir: &Path) -> Result<Option<DenseIndexInfo>, Error> {
    read_current(state_dir, |g| g.dense_info())
}

/// Reads the current generation with `read`, starting over when a build publishes a
/// new generation and removes the one being read.
fn read_current<T>(
    state_dir: &Path,
    read: impl Fn(&Generation) -> Result<T, Error>,
) -> Result<T, Error> {
    let current = || std::fs::read_to_string(state_dir.join("CURRENT2")).ok();
    for attempt in 0.. {
        let before = current();
        match Generation::open(state_dir).and_then(|g| read(&g)) {
            Err(e)
                if attempt < 2
                    && e.downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                    && current() != before => {}
            result => return result,
        }
    }
    unreachable!()
}

/// The current generation of an index and how far its embedding has come
/// ([`status`]).
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct IndexStatus {
    /// The generation's directory name in the state directory.
    pub generation: String,
    /// Chunks in the generation.
    pub docs: u64,
    /// The generation's embeddings, with their coverage and state, when it has them.
    pub dense: Option<DenseIndexInfo>,
    /// Whether an embedding run is working in the state directory now.
    pub embedding: bool,
    /// The figures of the latest embedding run on this generation (texts embedded and
    /// to embed, tokens billed, requests), when one has run.
    pub progress: Option<EmbedProgress>,
    /// Why the latest run on this generation was stopped before it finished, when its
    /// stop signal ended it (the reason given to [`StopSignal::stop`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
}

/// The current generation, its embedding coverage, and whether a run is embedding it.
pub fn status(state_dir: &Path) -> Result<IndexStatus, Error> {
    read_current(state_dir, |g| {
        let run = read_progress(state_dir).filter(|r| r.generation == g.name);
        Ok(IndexStatus {
            generation: g.name.clone(),
            docs: g.meta.docs,
            dense: g.dense_info()?,
            embedding: embedding_active(state_dir),
            stopped: run.as_ref().and_then(|r| r.stopped.clone()),
            progress: run.map(|r| r.progress),
        })
    })
}

/// The ten best chunks for `query` by BM25 (lexical). [`hybrid::search`](crate::hybrid::search)
/// also searches embeddings, filters, and returns more or fewer results.
pub fn search(state_dir: &Path, query: &str) -> Result<Vec<Chunk>, Error> {
    let generation = Generation::open(state_dir)?;
    let best: Vec<u32> = generation
        .lexical(query, 10, &Filter::default())?
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    generation.chunks(&best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn not_found(e: Error) -> bool {
        e.downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
    }

    /// Reads of a generation that a build has since removed fail with "not found", so a
    /// search starts over on the new generation instead of reporting that the index
    /// has no embeddings or no file table.
    #[test]
    fn reads_of_a_removed_generation_are_not_found() {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.rs"), "fn journal() {}\n").unwrap();
        let embedder = crate::dense::HashEmbedder::new(16);
        let dense = BuildOptions::default().with_dense(DenseBuild::new(&embedder));
        let options = Options::default();
        build(root.path(), state.path(), &options, &dense).unwrap();
        let old = Generation::open(state.path()).unwrap();
        assert!(old.meta.dense && old.dense_info().unwrap().is_some());
        build(root.path(), state.path(), &options, &dense).unwrap();
        assert!(!old.dir.exists());
        assert!(not_found(old.dense_info().unwrap_err()));
        assert!(not_found(read_files(&old.dir).unwrap_err()));
        // A generation without embeddings has none.
        build(
            root.path(),
            state.path(),
            &options,
            &BuildOptions::default(),
        )
        .unwrap();
        let lexical = Generation::open(state.path()).unwrap();
        assert!(!lexical.meta.dense && lexical.dense_info().unwrap().is_none());
        assert!(dense_info(state.path()).unwrap().is_none());
    }
}
