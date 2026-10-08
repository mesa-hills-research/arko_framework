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
//!   dense.json the generation's embedding model, coverage and vector store
//!              ([`DenseIndexInfo`])
//!   vkeys.dat  per doc with a vector, sorted by vector id: u128 BE id, u32 LE doc
//! The vectors themselves are in a store beside the generations,
//! `STATE_DIR/vectors-<digest of model and settings>/` ([`MhrStore`]), which every
//! build updates in place: a doc's vector id is a digest of its path and its embedded
//! text, so an unchanged chunk keeps its vector, whatever its doc id.
//! Build memory is bounded: postings accumulate up to `mem_budget` bytes, then spill
//! to a sorted run file; runs cover increasing doc ranges and are merged by term.
//! Texts to embed are spooled to disk.
use crate::{
    bench_v1::{Chunk, Error, Options},
    dense::{
        DenseBuild, DenseConfig, DenseIndexInfo, EmbedStats, EmbedText, RENDER_VERSION,
        cache::Cache, render_with, run, text_format, text_hash,
    },
    files,
    files::{FileRules, Report},
    fsio::{atomic_write, read_exact_at, read_u32, read_u64},
    par,
    search::tokens,
    tokenize::{self, Counts, FxHasher, FxMap},
    vector::{
        self, Filter, MhrStore, StoreConfig, VectorId, VectorMeta, VectorRecord, VectorStore,
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
    cache: Option<&Cache>,
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
        if let Some(cache) = cache {
            let text = render_with(chunk, &file.text[s..e], doc_text.embed_text);
            let hash = text_hash(&text);
            out.hashes.push(hash);
            if !cache.contains(&hash) {
                out.missing.push((hash, text));
            }
        }
    }
    out.path = path;
    Ok(out)
}

struct RunReader {
    r: BufReader<File>,
    /// (first doc, last doc, df, postings bytes) of the current term; the term
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
/// vectors are stored with the generation, and the settings are saved as the index's
/// [`DenseConfig`]. Without it, a saved config is left as it is.
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
    // (first doc, path) of every file with chunks, for the vector store's metadata.
    let mut file_table: Vec<(u32, String)> = Vec::new();
    // Embeddings: the cache of the embedder's model, each doc's text hash (in doc
    // order), the texts to embed, and the distinct hashes seen.
    let mut cache = match dense {
        Some(d) => Some(Cache::open(&state_dir, &d.embedder.descriptor())?),
        None => None,
    };
    let mut vhash = match dense {
        Some(_) => Some(w("vhash.tmp")?),
        None => None,
    };
    let mut spool = match dense {
        Some(_) => Some(run::Spool::create(&dir.join("embed-spool.tmp"))?),
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
            |item| file_docs(item, options, shards, cache.as_ref(), doc_text),
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
    // Embeddings: embed what the cache lacks, then bring the vector store up to date.
    let mut dense_stats = None;
    // The store's directory name, the store, and the vectors to remove after the switch.
    let mut store_update: Option<(String, MhrStore, Vec<VectorId>)> = None;
    if let (Some(d), Some(cache), Some(mut vhash), Some(spool)) =
        (dense, cache.as_mut(), vhash.take(), spool.take())
    {
        vhash.flush()?;
        drop(vhash);
        let (spool_path, to_embed) = spool.finish()?;
        let descriptor = d.embedder.descriptor();
        let totals = run::embed_spool(
            &spool_path,
            to_embed,
            cache,
            d.embedder,
            d.max_tokens,
            d.progress,
        )?;
        phases.push(("embedded_s", clock.elapsed().as_secs_f64()));
        let hashes = File::open(dir.join("vhash.tmp"))?;
        let keys = vector_keys(&hashes, next_doc, &file_table, cache)?;
        write_vector_keys(&dir, &keys)?;
        let config = StoreConfig::new(
            descriptor.clone(),
            text_format(RENDER_VERSION, d.embed_text),
        )
        .with_quantization(d.quantization)
        .with_index(d.vector_index);
        let name = config.dir_name()?;
        let (mut store, created) = vector::open_or_create(&state_dir.join(&name), &config)?;
        let (added, removed) = diff_ids(&keys, &store.ids()?);
        // The new vectors go in now, the vectors of chunks that are gone after the
        // switch: until then readers may still open the current generation, so the
        // store holds the vectors of both. A build that fails before the switch (its
        // generation then removed) leaves only additions, which searches skip and the
        // next build removes.
        let (mut vector, mut hash) = (Vec::new(), [0u8; 32]);
        for batch in added.chunks(UPSERT_BATCH) {
            let mut vectors = Vec::with_capacity(batch.len());
            for &(_, doc) in batch {
                read_exact_at(&hashes, &mut hash, u64::from(doc) * 32)?;
                if !cache.get(&hash, &mut vector)? {
                    return Err("an embedding left the cache during the build".into());
                }
                vectors.push(std::mem::take(&mut vector));
            }
            let records: Vec<VectorRecord<'_>> = batch
                .iter()
                .zip(&vectors)
                .map(|(&(id, doc), v)| {
                    let path = file_path(&file_table, doc);
                    VectorRecord {
                        id,
                        vector: v,
                        meta: VectorMeta {
                            path,
                            language: crate::syntax::language(Path::new(path)),
                        },
                    }
                })
                .collect();
            store.upsert(&records)?;
        }
        if created {
            store.compact()?;
        }
        store.persist()?;
        drop(hashes);
        let embedded = keys.len() as u64;
        let info = DenseIndexInfo {
            version: DenseIndexInfo::VERSION,
            embedder: descriptor.clone(),
            render_version: RENDER_VERSION,
            embed_text: d.embed_text,
            quantization: d.quantization,
            vector_index: d.vector_index,
            store: name.clone(),
            docs: n,
            embedded,
        };
        atomic_write(&dir, "dense.json", &serde_json::to_vec_pretty(&info)?)?;
        std::fs::remove_file(dir.join("vhash.tmp"))?;
        std::fs::remove_file(&spool_path)?;
        phases.push(("vectors_stored_s", clock.elapsed().as_secs_f64()));
        let vectors_added = added.len() as u64;
        let vectors_removed = removed.len() as u64;
        store_update = Some((name, store, removed));
        dense_stats = Some(EmbedStats {
            embedder: descriptor,
            render_version: RENDER_VERSION,
            embed_text: d.embed_text,
            docs: n,
            embedded_docs: embedded,
            texts: seen.len() as u64,
            cached: (seen.len() - pending.len()) as u64,
            new: totals.new,
            requests: totals.requests,
            tokens: totals.tokens,
            max_tokens: d.max_tokens,
            cap_reached: totals.cap_reached,
            not_embedded: totals.not_embedded,
            not_embedded_tokens: totals.not_embedded_tokens,
            too_long: totals.too_long,
            cache_records: 0,
            vectors_added,
            vectors_removed,
            store_created: created,
        });
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
    // Older generations: readers that already opened their files keep reading them.
    for e in std::fs::read_dir(&state_dir)? {
        let e = e?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("index2-") && name != generation.as_str() {
            std::fs::remove_dir_all(e.path())?;
        }
    }
    // The vectors of chunks that are gone, and stores of other models or settings
    // (or of no generation, after a lexical build).
    let store_name = match store_update {
        Some((name, mut store, removed)) => {
            for batch in removed.chunks(UPSERT_BATCH) {
                store.delete(batch)?;
            }
            store.persist()?;
            Some(name)
        }
        None => None,
    };
    for e in std::fs::read_dir(&state_dir)? {
        let e = e?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("vectors-") && store_name.as_deref() != Some(&*name) {
            std::fs::remove_dir_all(e.path())?;
        }
    }
    files::write_report(&state_dir, &report)?;
    if let (Some(d), Some(cache)) = (dense, cache.as_mut()) {
        DenseConfig::new(d.embedder.descriptor())
            .with_quantization(d.quantization)
            .with_vector_index(d.vector_index)
            .with_max_tokens_per_run(d.max_tokens)
            .with_embed_text(d.embed_text)
            .save(&state_dir)?;
        cache.compact_if_sparse(&seen)?;
        if let Some(stats) = &mut dense_stats {
            stats.cache_records = cache.records();
        }
    }
    Ok(BuildStats {
        docs: n,
        terms: nterms,
        runs: runs.len(),
        report,
        dense: dense_stats,
    })
}

/// Vectors per store upsert or delete call.
const UPSERT_BATCH: usize = 1024;
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

/// (vector id, doc) of every doc whose text has a vector in the cache, by id.
/// `hashes` holds each doc's text hash, in doc order.
fn vector_keys(
    hashes: &File,
    docs: u32,
    files: &[(u32, String)],
    cache: &Cache,
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
        if cache.contains(&hash) {
            let n = seen.entry(hash).or_insert(0);
            keys.push((vector_id(&files[file].1, &hash, *n), doc));
            *n += 1;
        }
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
                "{e}: this index has no file table (built by an earlier version); rebuild it to filter"
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
    dir: PathBuf,
    meta: Meta,
}
impl Generation {
    pub(crate) fn open(state_dir: &Path) -> Result<Self, Error> {
        let generation = std::fs::read_to_string(state_dir.join("CURRENT2"))?;
        let dir = state_dir.join(generation.trim());
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        if meta.version != 2 {
            return Err("unsupported index version".into());
        }
        Ok(Self {
            state_dir: state_dir.to_path_buf(),
            dir,
            meta,
        })
    }

    /// The embeddings of this generation, if it has them.
    pub(crate) fn dense_info(&self) -> Result<Option<DenseIndexInfo>, Error> {
        match std::fs::read(self.dir.join("dense.json")) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            // Missing from a generation built with embeddings: a build removed the
            // generation, and the error makes a search start over on the new one.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && !self.meta.dense => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// The vector store of this generation's embeddings, read-only. It holds every
    /// vector of the generation, and can hold others: those a build in progress (or an
    /// interrupted one) added for its own generation.
    pub(crate) fn open_store(&self, info: &DenseIndexInfo) -> Result<MhrStore, Error> {
        if info.version != DenseIndexInfo::VERSION {
            return Err(format!(
                "this index's vectors were stored by an earlier version of arko_index (format {}): rebuild the index (index2) to search it densely",
                info.version
            )
            .into());
        }
        if !info.store.starts_with("vectors-") || info.store.contains(['/', '\\']) {
            return Err(format!("bad vector store name {:?}", info.store).into());
        }
        let dir = self.state_dir.join(&info.store);
        let store = MhrStore::open(&dir).map_err(|e| {
            format!("{e} (the vector store of this index: rebuild the index to restore it)")
        })?;
        if !store.matches(&info.store_config()) {
            return Err(format!(
                "the vector store {} does not hold this index's model and settings ({}): rebuild the index",
                dir.display(),
                info.embedder.display()
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
/// `None` when it was built without them.
pub fn dense_info(state_dir: &Path) -> Result<Option<DenseIndexInfo>, Error> {
    // A build may publish a new generation and remove the one being read: start over.
    let current = || std::fs::read_to_string(state_dir.join("CURRENT2")).ok();
    for attempt in 0.. {
        let before = current();
        match Generation::open(state_dir).and_then(|g| g.dense_info()) {
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
