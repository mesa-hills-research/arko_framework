//! Vector storage for dense search: the [`VectorStore`] trait and
//! [`BruteForceStore`], an exact store that scans every vector.
//!
//! A store maps caller-chosen `u64` ids to vectors, each with the path and language
//! of what it embeds, and returns the `k` most similar ids for a query, optionally
//! restricted by a [`Filter`]. Similarity is cosine: stores normalize vectors on
//! insert and queries on search, so scores are in `[-1, 1]`. Results are ordered by
//! score, best first, then by id ascending, so equal scores always come out in the
//! same order.
use crate::{
    bench_v1::Error,
    fsio::{Cursor, atomic_write, put_string, read_exact_at, write_all_at},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

/// Restricts a search to vectors whose path starts with `path_prefix` and whose
/// language (the chunk's `language`, such as `rust` or `text`) equals `language`.
/// The default restricts nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Filter {
    /// A plain string prefix of the relative, slash-separated path (`src/net/`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}
impl Filter {
    pub fn with_path_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.path_prefix = Some(prefix.into());
        self
    }
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
    /// Whether the filter restricts nothing.
    pub fn is_empty(&self) -> bool {
        self.path_prefix.is_none() && self.language.is_none()
    }
    pub fn matches(&self, path: &str, language: &str) -> bool {
        self.path_prefix
            .as_deref()
            .is_none_or(|p| path.starts_with(p))
            && self.language.as_deref().is_none_or(|l| l == language)
    }
}

/// What a stored vector embeds, for filtering.
#[derive(Clone, Copy, Debug)]
pub struct VectorMeta<'a> {
    pub path: &'a str,
    pub language: &'a str,
}

/// One search result: a stored id and its cosine similarity to the query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorHit {
    pub id: u64,
    pub score: f32,
}

/// A searchable set of vectors. Implementations: [`BruteForceStore`] (exact).
///
/// `insert` and `delete` change the store in memory and in its files; `persist`
/// makes every change since the last `persist` durable at once. `open` loads the last
/// persisted state. How a store is created (its directory, dimensions and
/// parameters) is specific to each implementation.
pub trait VectorStore: Send + Sync {
    /// The length of every vector.
    fn dimensions(&self) -> usize;
    /// The number of live (not deleted) vectors.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Adds a vector under `id`, replacing the vector already stored under it.
    fn insert(&mut self, id: u64, vector: &[f32], meta: VectorMeta<'_>) -> Result<(), Error>;
    /// Removes the vector stored under `id`; returns whether there was one.
    fn delete(&mut self, id: u64) -> Result<bool, Error>;
    /// The `k` most similar vectors that pass `filter`: by score, best first, then
    /// by id ascending.
    fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<VectorHit>, Error>;
    /// Makes all changes durable.
    fn persist(&mut self) -> Result<(), Error>;
    /// Opens a persisted store.
    fn open(dir: &Path) -> Result<Self, Error>
    where
        Self: Sized;
}

/// How [`BruteForceStore`] keeps vectors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// 32-bit floats: exact scores; the whole file is read per search.
    #[default]
    F32,
    /// Also an 8-bit copy (one scale per vector): a search scans the 8-bit copy, a
    /// quarter of the bytes, then rescores the best `4 × k` (at least 40) candidates
    /// exactly from the 32-bit vectors. Returned scores are exact; a vector the
    /// 8-bit scan ranks below the candidates can be missed.
    Int8,
}

/// The factor and floor of int8 rescoring candidates.
const RESCORE_FACTOR: usize = 4;
const RESCORE_MIN: usize = 40;
const MAGIC: &[u8; 8] = b"ARKOVEC1";
const SLOTS_FILE: &str = "slots.dat";
const F32_FILE: &str = "vectors.f32";
const I8_FILE: &str = "vectors.i8";
const DELETED: u64 = u64::MAX;
/// Bytes read per block while scanning.
const SCAN_BLOCK: usize = 1 << 20;

/// An exact vector store: every search computes the similarity to every live
/// vector, reading the vector file in 1 MB blocks, so memory stays small (about 12
/// bytes per vector for ids and paths, plus the path table) whatever the store's
/// size.
///
/// Files in its directory: `vectors.f32` (each vector, normalized, as little-endian
/// f32 in insertion order), `vectors.i8` with [`Quantization::Int8`], and
/// `slots.dat` (ids, paths, languages and int8 scales, with a SHA-256 checksum),
/// replaced atomically by [`persist`](VectorStore::persist). The vector files are
/// only appended to (slots are never reused), so a reader that opened the store
/// keeps a consistent view while a writer adds to it.
pub struct BruteForceStore {
    dir: PathBuf,
    dims: usize,
    quantization: Quantization,
    /// Per slot: the id, or `DELETED`.
    ids: Vec<u64>,
    /// Per slot: index into `paths`.
    path_of: Vec<u32>,
    /// Per slot, with int8: the scale of its 8-bit vector.
    scales: Vec<f32>,
    /// (path, language) of the stored vectors.
    paths: Vec<(String, String)>,
    path_index: HashMap<String, u32>,
    /// id -> slot of the live vectors; built on the first change after `open`.
    slot_of: Option<HashMap<u64, u32>>,
    live: usize,
    f32_file: File,
    i8_file: Option<File>,
    /// Whether the files were opened for writing.
    writable: bool,
}

impl BruteForceStore {
    /// Creates an empty store in `dir` (created if missing; it must not already hold
    /// a store).
    pub fn create(dir: &Path, dims: usize, quantization: Quantization) -> Result<Self, Error> {
        if dims == 0 {
            return Err("vector dimensions must be at least 1".into());
        }
        std::fs::create_dir_all(dir)?;
        if dir.join(SLOTS_FILE).exists() {
            return Err(format!("{} already holds a vector store", dir.display()).into());
        }
        let create = |name: &str| -> Result<File, Error> {
            Ok(OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(dir.join(name))?)
        };
        let f32_file = create(F32_FILE)?;
        let i8_file = match quantization {
            Quantization::F32 => None,
            Quantization::Int8 => Some(create(I8_FILE)?),
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            dims,
            quantization,
            ids: Vec::new(),
            path_of: Vec::new(),
            scales: Vec::new(),
            paths: Vec::new(),
            path_index: HashMap::new(),
            slot_of: Some(HashMap::new()),
            live: 0,
            f32_file,
            i8_file,
            writable: true,
        })
    }

    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    fn row_bytes(&self) -> usize {
        self.dims * 4
    }

    /// Reopens the vector files for writing (a store opened with `open` reads only).
    fn make_writable(&mut self) -> Result<(), Error> {
        if self.writable {
            return Ok(());
        }
        let open = |name: &str| -> Result<File, Error> {
            Ok(OpenOptions::new()
                .read(true)
                .write(true)
                .open(self.dir.join(name))?)
        };
        self.f32_file = open(F32_FILE)?;
        if self.i8_file.is_some() {
            self.i8_file = Some(open(I8_FILE)?);
        }
        self.writable = true;
        Ok(())
    }

    fn slots(&mut self) -> &mut HashMap<u64, u32> {
        let ids = &self.ids;
        self.slot_of.get_or_insert_with(|| {
            ids.iter()
                .enumerate()
                .filter(|(_, id)| **id != DELETED)
                .map(|(slot, id)| (*id, slot as u32))
                .collect()
        })
    }

    fn encode_slots(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.ids.len() * 16);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(self.dims as u32).to_le_bytes());
        out.push(match self.quantization {
            Quantization::F32 => 0,
            Quantization::Int8 => 1,
        });
        out.extend_from_slice(&[0; 7]);
        out.extend_from_slice(&(self.ids.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.paths.len() as u64).to_le_bytes());
        for (path, language) in &self.paths {
            put_string(&mut out, path);
            put_string(&mut out, language);
        }
        for (slot, id) in self.ids.iter().enumerate() {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&self.path_of[slot].to_le_bytes());
            if self.quantization == Quantization::Int8 {
                out.extend_from_slice(&self.scales[slot].to_le_bytes());
            }
        }
        let digest = Sha256::digest(&out);
        out.extend_from_slice(&digest);
        out
    }

    /// The live slots whose path passes the filter, as a per-path table.
    fn allowed_paths(&self, filter: &Filter) -> Vec<bool> {
        self.paths
            .iter()
            .map(|(p, l)| filter.matches(p, l))
            .collect()
    }

    /// Calls `visit(slot, row)` for every slot in order, reading `file` in blocks of
    /// rows of `row_bytes` bytes.
    fn scan(
        &self,
        file: &File,
        row_bytes: usize,
        mut visit: impl FnMut(usize, &[u8]),
    ) -> Result<(), Error> {
        let rows_per_block = (SCAN_BLOCK / row_bytes).max(1);
        let mut buf = vec![0u8; rows_per_block * row_bytes];
        let mut slot = 0;
        while slot < self.ids.len() {
            let n = rows_per_block.min(self.ids.len() - slot);
            let block = &mut buf[..n * row_bytes];
            read_exact_at(file, block, (slot * row_bytes) as u64)?;
            for (i, row) in block.chunks_exact(row_bytes).enumerate() {
                visit(slot + i, row);
            }
            slot += n;
        }
        Ok(())
    }

    fn exact_row(&self, slot: usize, out: &mut Vec<f32>) -> Result<(), Error> {
        let mut bytes = vec![0u8; self.row_bytes()];
        read_exact_at(&self.f32_file, &mut bytes, (slot * self.row_bytes()) as u64)?;
        decode_f32(&bytes, out);
        Ok(())
    }
}

/// Little-endian f32s into `out` (replacing its contents).
fn decode_f32(bytes: &[u8], out: &mut Vec<f32>) {
    out.clear();
    out.extend(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
    );
}

/// Dot product with a fixed summation order (eight lanes, then a fixed reduction),
/// so a score is the same bits on every run.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// The vector scaled to length 1; an error for a zero, non-finite or wrongly sized
/// vector.
pub(crate) fn normalized(vector: &[f32], dims: usize) -> Result<Vec<f32>, Error> {
    if vector.len() != dims {
        return Err(format!("vector has {} dimensions, the store {dims}", vector.len()).into());
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err("vector has a non-finite component".into());
    }
    let norm = vector
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err("cannot store or search a zero vector".into());
    }
    Ok(vector
        .iter()
        .map(|v| (f64::from(*v) / norm) as f32)
        .collect())
}

/// Ordering of hits: higher score first, then lower id.
fn better(a: &VectorHit, b: &VectorHit) -> Ordering {
    b.score.total_cmp(&a.score).then(a.id.cmp(&b.id))
}

/// Keeps the best `k` hits seen.
pub(crate) struct TopK {
    k: usize,
    /// Max-heap by "worse": the top is the worst hit kept.
    heap: BinaryHeap<Worst>,
}
struct Worst(VectorHit);
impl PartialEq for Worst {
    fn eq(&self, other: &Self) -> bool {
        better(&self.0, &other.0) == Ordering::Equal
    }
}
impl Eq for Worst {}
impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        better(&self.0, &other.0)
    }
}
impl TopK {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }
    pub(crate) fn offer(&mut self, hit: VectorHit) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(Worst(hit));
        } else if let Some(worst) = self.heap.peek()
            && better(&hit, &worst.0) == Ordering::Less
        {
            self.heap.pop();
            self.heap.push(Worst(hit));
        }
    }
    pub(crate) fn into_sorted(self) -> Vec<VectorHit> {
        let mut hits: Vec<_> = self.heap.into_iter().map(|w| w.0).collect();
        hits.sort_by(better);
        hits
    }
}

impl VectorStore for BruteForceStore {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn len(&self) -> usize {
        self.live
    }

    fn insert(&mut self, id: u64, vector: &[f32], meta: VectorMeta<'_>) -> Result<(), Error> {
        if id == DELETED {
            return Err("vector id u64::MAX is reserved".into());
        }
        let v = normalized(vector, self.dims)?;
        self.make_writable()?;
        let slot = self.ids.len();
        let slot32 = u32::try_from(slot).map_err(|_| "too many vectors")?;
        let mut bytes = Vec::with_capacity(self.row_bytes());
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        // Positional writes: a failed write leaves nothing in memory, and the next
        // insert overwrites the partial row.
        write_all_at(&self.f32_file, &bytes, (slot * self.row_bytes()) as u64)?;
        let mut scale = 0.0;
        if let Some(file) = &self.i8_file {
            let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            scale = max / 127.0;
            let codes: Vec<u8> = v
                .iter()
                .map(|x| ((x / scale).round().clamp(-127.0, 127.0) as i8) as u8)
                .collect();
            write_all_at(file, &codes, (slot * self.dims) as u64)?;
        }
        let path_index = match self.path_index.get(meta.path) {
            Some(&i) => i,
            None => {
                let i = u32::try_from(self.paths.len()).map_err(|_| "too many paths")?;
                self.paths
                    .push((meta.path.to_owned(), meta.language.to_owned()));
                self.path_index.insert(meta.path.to_owned(), i);
                i
            }
        };
        let replaced = self.slots().insert(id, slot32);
        if let Some(old) = replaced {
            self.ids[old as usize] = DELETED;
        } else {
            self.live += 1;
        }
        self.ids.push(id);
        self.path_of.push(path_index);
        if self.quantization == Quantization::Int8 {
            self.scales.push(scale);
        }
        Ok(())
    }

    fn delete(&mut self, id: u64) -> Result<bool, Error> {
        match self.slots().remove(&id) {
            Some(slot) => {
                self.ids[slot as usize] = DELETED;
                self.live -= 1;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<VectorHit>, Error> {
        let q = normalized(query, self.dims)?;
        if k == 0 || self.live == 0 {
            return Ok(Vec::new());
        }
        let allowed = self.allowed_paths(filter);
        let pass = |slot: usize| self.ids[slot] != DELETED && allowed[self.path_of[slot] as usize];
        match &self.i8_file {
            None => {
                let mut top = TopK::new(k);
                let mut row = Vec::with_capacity(self.dims);
                self.scan(&self.f32_file, self.row_bytes(), |slot, bytes| {
                    if pass(slot) {
                        decode_f32(bytes, &mut row);
                        top.offer(VectorHit {
                            id: self.ids[slot],
                            score: dot(&q, &row),
                        });
                    }
                })?;
                Ok(top.into_sorted())
            }
            Some(file) => {
                // Candidates by the 8-bit approximation (ids here are slots), then
                // exact scores from the f32 rows.
                let mut candidates = TopK::new((k * RESCORE_FACTOR).max(RESCORE_MIN));
                let mut row = vec![0f32; self.dims];
                self.scan(file, self.dims, |slot, codes| {
                    if pass(slot) {
                        for (r, c) in row.iter_mut().zip(codes) {
                            *r = f32::from(*c as i8);
                        }
                        candidates.offer(VectorHit {
                            id: slot as u64,
                            score: dot(&q, &row) * self.scales[slot],
                        });
                    }
                })?;
                let mut top = TopK::new(k);
                let mut exact = Vec::with_capacity(self.dims);
                for c in candidates.into_sorted() {
                    let slot = c.id as usize;
                    self.exact_row(slot, &mut exact)?;
                    top.offer(VectorHit {
                        id: self.ids[slot],
                        score: dot(&q, &exact),
                    });
                }
                Ok(top.into_sorted())
            }
        }
    }

    fn persist(&mut self) -> Result<(), Error> {
        if self.writable {
            self.f32_file.sync_all()?;
            if let Some(f) = &self.i8_file {
                f.sync_all()?;
            }
        }
        atomic_write(&self.dir, SLOTS_FILE, &self.encode_slots())
    }

    fn open(dir: &Path) -> Result<Self, Error> {
        let raw = std::fs::read(dir.join(SLOTS_FILE))?;
        if raw.len() < 32 + MAGIC.len() {
            return Err("vector store slots file is truncated".into());
        }
        let (body, digest) = raw.split_at(raw.len() - 32);
        if Sha256::digest(body).as_slice() != digest {
            return Err("vector store slots file is corrupt (checksum mismatch)".into());
        }
        let mut c = Cursor::new(body);
        if c.bytes(8)? != MAGIC {
            return Err("not a vector store slots file".into());
        }
        let version = c.u32()?;
        if version != 1 {
            return Err(format!("unsupported vector store version {version}").into());
        }
        let dims = c.u32()? as usize;
        let quantization = match c.u8()? {
            0 => Quantization::F32,
            1 => Quantization::Int8,
            q => return Err(format!("unknown vector quantization {q}").into()),
        };
        c.bytes(7)?;
        let slots = usize::try_from(c.u64()?)?;
        let npaths = usize::try_from(c.u64()?)?;
        let mut paths = Vec::with_capacity(npaths.min(1 << 20));
        let mut path_index = HashMap::with_capacity(npaths.min(1 << 20));
        for i in 0..npaths {
            let path = c.string()?;
            let language = c.string()?;
            path_index.insert(path.clone(), i as u32);
            paths.push((path, language));
        }
        let mut ids = Vec::with_capacity(slots.min(1 << 24));
        let mut path_of = Vec::with_capacity(slots.min(1 << 24));
        let mut scales = Vec::new();
        let mut live = 0;
        for _ in 0..slots {
            let id = c.u64()?;
            let p = c.u32()?;
            if p as usize >= paths.len() {
                return Err("vector store path index out of range".into());
            }
            if quantization == Quantization::Int8 {
                scales.push(c.f32()?);
            }
            live += usize::from(id != DELETED);
            ids.push(id);
            path_of.push(p);
        }
        if c.at != body.len() {
            return Err("vector store slots file has trailing bytes".into());
        }
        let f32_file = File::open(dir.join(F32_FILE))?;
        let need = (slots * dims * 4) as u64;
        if f32_file.metadata()?.len() < need {
            return Err("vector file is shorter than its slots".into());
        }
        let i8_file = match quantization {
            Quantization::F32 => None,
            Quantization::Int8 => {
                let f = File::open(dir.join(I8_FILE))?;
                if f.metadata()?.len() < (slots * dims) as u64 {
                    return Err("int8 vector file is shorter than its slots".into());
                }
                Some(f)
            }
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            dims,
            quantization,
            ids,
            path_of,
            scales,
            paths,
            path_index,
            slot_of: None,
            live,
            f32_file,
            i8_file,
            writable: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(path: &str) -> VectorMeta<'_> {
        VectorMeta {
            path,
            language: crate::syntax::language(Path::new(path)),
        }
    }

    /// A deterministic pseudo-random vector.
    fn vector(seed: u64, dims: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
        (0..dims)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Exact reference ranking by brute force in memory.
    fn reference(vectors: &[(u64, Vec<f32>, &str)], q: &[f32], k: usize, f: &Filter) -> Vec<u64> {
        let q = normalized(q, q.len()).unwrap();
        let mut scored: Vec<VectorHit> = vectors
            .iter()
            .filter(|(_, _, p)| f.matches(p, crate::syntax::language(Path::new(p))))
            .map(|(id, v, _)| VectorHit {
                id: *id,
                score: dot(&q, &normalized(v, v.len()).unwrap()),
            })
            .collect();
        scored.sort_by(better);
        scored.into_iter().take(k).map(|h| h.id).collect()
    }

    #[test]
    fn exact_search_filters_ties_persistence_and_deletes() {
        let dims = 19; // not a multiple of the 8 lanes
        for quantization in [Quantization::F32, Quantization::Int8] {
            let dir = tempfile::tempdir().unwrap();
            let mut store = BruteForceStore::create(dir.path(), dims, quantization).unwrap();
            let paths = ["src/a.rs", "src/b.py", "docs/c.md", "src/net/d.rs"];
            let mut vectors = Vec::new();
            for id in 0..300u64 {
                let v = vector(id, dims);
                let p = paths[id as usize % paths.len()];
                store.insert(id, &v, meta(p)).unwrap();
                vectors.push((id, v, p));
            }
            // Two identical vectors tie: the lower id comes first.
            let twin = vector(7, dims);
            store.insert(1000, &twin, meta("src/a.rs")).unwrap();
            vectors.push((1000, twin.clone(), "src/a.rs"));
            for filter in [
                Filter::default(),
                Filter::default().with_path_prefix("src/"),
                Filter::default().with_language("rust"),
                Filter::default()
                    .with_path_prefix("src/n")
                    .with_language("rust"),
                Filter::default().with_language("go"),
            ] {
                for seed in [7u64, 11, 500] {
                    let q = vector(seed, dims);
                    let got: Vec<u64> = store
                        .search(&q, 10, &filter)
                        .unwrap()
                        .iter()
                        .map(|h| h.id)
                        .collect();
                    assert_eq!(
                        got,
                        reference(&vectors, &q, 10, &filter),
                        "{quantization:?} {filter:?}"
                    );
                }
            }
            let hits = store.search(&twin, 2, &Filter::default()).unwrap();
            assert_eq!((hits[0].id, hits[1].id), (7, 1000));
            assert!((hits[0].score - 1.0).abs() < 1e-5);
            store.persist().unwrap();

            // Reopened: same answers; deletes and replacements persist.
            let mut store = BruteForceStore::open(dir.path()).unwrap();
            assert_eq!(store.len(), 301);
            assert_eq!(store.quantization(), quantization);
            assert!(store.delete(7).unwrap());
            assert!(!store.delete(7).unwrap());
            store
                .insert(1000, &vector(42, dims), meta("src/a.rs"))
                .unwrap();
            store.persist().unwrap();
            let store = BruteForceStore::open(dir.path()).unwrap();
            assert_eq!(store.len(), 300);
            vectors.retain(|(id, _, _)| *id != 7 && *id != 1000);
            vectors.push((1000, vector(42, dims), "src/a.rs"));
            let q = vector(7, dims);
            let got: Vec<u64> = store
                .search(&q, 10, &Filter::default())
                .unwrap()
                .iter()
                .map(|h| h.id)
                .collect();
            assert_eq!(got, reference(&vectors, &q, 10, &Filter::default()));
        }
    }

    #[test]
    fn rejects_bad_vectors_and_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = BruteForceStore::create(dir.path(), 4, Quantization::F32).unwrap();
        assert!(store.insert(1, &[0.0; 4], meta("a.rs")).is_err());
        assert!(store.insert(1, &[1.0; 3], meta("a.rs")).is_err());
        assert!(
            store
                .insert(1, &[f32::NAN, 1.0, 1.0, 1.0], meta("a.rs"))
                .is_err()
        );
        assert!(store.insert(u64::MAX, &[1.0; 4], meta("a.rs")).is_err());
        store.insert(1, &[1.0; 4], meta("a.rs")).unwrap();
        assert!(store.search(&[0.0; 4], 1, &Filter::default()).is_err());
        store.persist().unwrap();
        assert!(BruteForceStore::create(dir.path(), 4, Quantization::F32).is_err());
        let path = dir.path().join(SLOTS_FILE);
        let mut raw = std::fs::read(&path).unwrap();
        raw[20] ^= 1;
        std::fs::write(&path, raw).unwrap();
        assert!(BruteForceStore::open(dir.path()).is_err());
    }
}
