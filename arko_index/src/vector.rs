//! Vector storage for dense search: the [`VectorStore`] trait and [`MhrStore`], its
//! implementation on [mhr_vector](https://github.com/mesa-hills-research/mhr_vector).
//!
//! A store maps 128-bit ids to vectors, each with the path and language of what it
//! embeds, and returns the `k` most similar ids for a query, optionally restricted by a
//! [`Filter`]. Similarity is cosine: stores normalize vectors and queries, so scores are
//! in `[-1, 1]`. Results are ordered by score, best first, then by id ascending, so
//! equal scores always come out in the same order.
use crate::{bench_v1::Error, dense::ModelDescriptor};
use mhr_vector::{
    AnnKind, CodeKind, EmbeddingModel, IndexConfig, Op, Record, SearchParams, StoreOptions,
    VamanaParams,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The id of a stored vector.
pub type VectorId = u128;

/// Restricts a search to vectors under `path_prefix` whose language (the chunk's
/// `language`, such as `rust` or `text`) equals `language`. The default restricts
/// nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Filter {
    /// A relative, slash-separated path prefix matched by whole components: `src/net`
    /// matches `src/net/retry.rs` (and a file named `src/net`), but not
    /// `src/network.rs`. A trailing `/` is ignored.
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
            .is_none_or(|p| mhr_vector::filter::path_has_prefix(path, p))
            && self.language.as_deref().is_none_or(|l| l == language)
    }
    fn to_mhr(&self) -> Option<mhr_vector::Filter> {
        (!self.is_empty()).then(|| mhr_vector::Filter {
            path_prefixes: self.path_prefix.iter().cloned().collect(),
            languages: self.language.iter().cloned().collect(),
        })
    }
}

/// What a stored vector embeds, for filtering.
#[derive(Clone, Copy, Debug)]
pub struct VectorMeta<'a> {
    pub path: &'a str,
    pub language: &'a str,
}

/// A vector to store under an id.
#[derive(Clone, Copy, Debug)]
pub struct VectorRecord<'a> {
    pub id: VectorId,
    pub vector: &'a [f32],
    pub meta: VectorMeta<'a>,
}

/// One search result: a stored id and its cosine similarity to the query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorHit {
    pub id: VectorId,
    pub score: f32,
}

/// A searchable set of vectors. Implementation: [`MhrStore`].
///
/// `upsert` and `delete` change what the store's own searches see at once; `persist`
/// makes every change since the last `persist` durable at once, and visible to stores
/// opened afterwards. How a store is created and opened (its directory, model and
/// settings) is specific to each implementation.
pub trait VectorStore: Send + Sync {
    /// The length of every vector.
    fn dimensions(&self) -> usize;
    /// The number of stored vectors.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Stores each record's vector under its id, replacing the vector stored under it.
    /// A batch is applied in order, all or nothing.
    fn upsert(&mut self, records: &[VectorRecord<'_>]) -> Result<(), Error>;
    /// Removes the vectors stored under `ids`; returns how many there were.
    fn delete(&mut self, ids: &[VectorId]) -> Result<usize, Error>;
    /// Every stored id, ascending.
    fn ids(&self) -> Result<Vec<VectorId>, Error>;
    /// The `k` most similar vectors that pass `filter`: by score, best first, then
    /// by id ascending.
    fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<VectorHit>, Error>;
    /// Makes all changes durable, atomically.
    fn persist(&mut self) -> Result<(), Error>;
}

/// The codes a store keeps in memory to find candidates. Full-precision vectors are
/// kept on disk, and every search rescores its candidates with them, so returned
/// scores are exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// The vectors themselves, 4 bytes per dimension: exact results.
    F32,
    /// One byte per dimension, scaled to each dimension's range: a search scans these
    /// codes, then rescores the best `max(4 × k, 40)` candidates of each segment.
    /// A vector the codes rank below the candidates can be missed.
    #[default]
    Int8,
}

/// How a search finds its candidates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorIndex {
    /// A scan of every vector's codes.
    #[default]
    Flat,
    /// A Vamana graph (DiskANN's) on each segment of 20,000 vectors or more, searched
    /// approximately: for very large indexes or high query rates. It costs build
    /// time, and a little recall.
    Graph,
}

/// The int8 rescoring depth: `max(RESCORE_FACTOR × k, RESCORE_MIN)` candidates per
/// segment.
const RESCORE_FACTOR: u32 = 4;
const RESCORE_MIN: u32 = 40;
/// Part of every store directory's name: a change to how ids or settings map onto
/// mhr_vector gets a new store.
const STORE_FORMAT: u32 = 1;

/// What a store holds and how it searches, fixed when it is created.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoreConfig {
    /// The model the vectors come from.
    pub model: ModelDescriptor,
    /// How the embedded texts were made (arko_index's render version and embedded
    /// text). Vectors of other texts are not comparable, so it is part of the model.
    pub text_format: String,
    pub quantization: Quantization,
    pub index: VectorIndex,
}
impl StoreConfig {
    pub fn new(model: ModelDescriptor, text_format: impl Into<String>) -> Self {
        Self {
            model,
            text_format: text_format.into(),
            quantization: Quantization::default(),
            index: VectorIndex::default(),
        }
    }
    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }
    pub fn with_index(mut self, index: VectorIndex) -> Self {
        self.index = index;
        self
    }

    fn embedding_model(&self) -> Result<EmbeddingModel, Error> {
        let m = &self.model;
        let input = |kind: &str| m.input_types.then(|| kind.to_owned());
        Ok(EmbeddingModel {
            provider: m.provider.clone(),
            model: m.model.clone(),
            dimensions: u32::try_from(m.dimensions).map_err(|_| "too many dimensions")?,
            document_input: input("document"),
            query_input: input("query"),
            output_dtype: "float".into(),
            normalized: false,
            text_format: self.text_format.clone(),
            revision: m.revision.clone(),
        })
    }

    fn index_config(&self) -> Result<IndexConfig, Error> {
        let mut config =
            IndexConfig::for_model(self.embedding_model()?).with_codes(match self.quantization {
                Quantization::F32 => CodeKind::F32,
                Quantization::Int8 => CodeKind::Int8,
            });
        if self.quantization == Quantization::Int8 {
            config.rescore_factor = RESCORE_FACTOR;
            config.rescore_min = RESCORE_MIN;
        }
        config.ann.kind = match self.index {
            VectorIndex::Flat => AnnKind::Flat,
            VectorIndex::Graph => AnnKind::Vamana(VamanaParams::default()),
        };
        config.validate()?;
        Ok(config)
    }

    /// The name of this config's store directory in an index's state directory:
    /// `vectors-` and 16 hex digits of a digest of the model and the settings, so
    /// another model or setting gets its own store.
    pub fn dir_name(&self) -> Result<String, Error> {
        let json = serde_json::to_vec(&(STORE_FORMAT, self.index_config()?))?;
        let digest = Sha256::digest(&json);
        let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
        Ok(format!("vectors-{hex}"))
    }
}

/// A vector store in a directory, kept by mhr_vector: segments of codes held in memory
/// (1 byte per dimension with [`Quantization::Int8`]) beside full-precision vectors read
/// from disk for rescoring, an in-memory delta of recent upserts, and deletions as
/// tombstones. [`persist`](VectorStore::persist) merges segments by its policy and
/// commits a new generation atomically: a crash keeps the last persisted state.
///
/// One handle writes a directory at a time (an exclusive lock); any number of
/// read-only handles ([`MhrStore::open`]) search its last persisted state meanwhile.
pub struct MhrStore {
    handle: Handle,
    dims: usize,
    /// Changes since the last persist.
    dirty: bool,
}

enum Handle {
    Writer(Box<mhr_vector::Writer>),
    Reader(mhr_vector::Snapshot),
}

fn store_options() -> StoreOptions {
    StoreOptions {
        threads: crate::threads(),
        ..StoreOptions::default()
    }
}

/// The mhr_vector key of an id: 32 hex digits, so key order is id order.
fn key(id: VectorId) -> String {
    format!("{id:032x}")
}
fn parse_key(key: &str) -> Result<VectorId, Error> {
    match u128::from_str_radix(key, 16) {
        Ok(id) if key.len() == 32 => Ok(id),
        _ => Err(format!("vector store key {key:?} is not an arko_index id").into()),
    }
}

impl MhrStore {
    /// Creates an empty store in `dir` (created if missing; it must not already hold a
    /// store).
    pub fn create(dir: &Path, config: &StoreConfig) -> Result<Self, Error> {
        Self::create_with(dir, config, store_options())
    }

    pub(crate) fn create_with(
        dir: &Path,
        config: &StoreConfig,
        options: StoreOptions,
    ) -> Result<Self, Error> {
        let writer = mhr_vector::Writer::create(dir, config.index_config()?, options)?;
        Ok(Self::from_writer(writer))
    }

    /// Opens the store in `dir` for writing. An error when it holds another model's
    /// vectors or was created with other settings.
    pub fn open_for(dir: &Path, config: &StoreConfig) -> Result<Self, Error> {
        let expected = config.index_config()?;
        let writer = mhr_vector::Writer::open(dir, store_options())?;
        check_config(writer.config(), &expected, dir)?;
        Ok(Self::from_writer(writer))
    }

    /// Opens the last persisted state of the store in `dir`, read-only.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let options = StoreOptions {
            threads: 1,
            ..StoreOptions::default()
        };
        let snapshot = mhr_vector::Reader::open(dir, options)?.snapshot();
        Ok(Self {
            dims: snapshot.config().dim as usize,
            handle: Handle::Reader(snapshot),
            dirty: false,
        })
    }

    /// Whether the store was created with `config`.
    pub fn matches(&self, config: &StoreConfig) -> bool {
        config
            .index_config()
            .is_ok_and(|expected| *self.snapshot().config() == expected)
    }

    /// Merges segments into as few as memory allows, with no deletions left (and
    /// graphs, with [`VectorIndex::Graph`]); for after a bulk load. Durable at the next
    /// [`persist`](VectorStore::persist).
    pub fn compact(&mut self) -> Result<(), Error> {
        self.writer()?.compact()?;
        self.dirty = true;
        Ok(())
    }

    fn from_writer(writer: mhr_vector::Writer) -> Self {
        Self {
            dims: writer.config().dim as usize,
            handle: Handle::Writer(Box::new(writer)),
            dirty: false,
        }
    }

    fn snapshot(&self) -> mhr_vector::Snapshot {
        match &self.handle {
            Handle::Writer(w) => w.snapshot(),
            Handle::Reader(s) => s.clone(),
        }
    }

    fn writer(&mut self) -> Result<&mut mhr_vector::Writer, Error> {
        match &mut self.handle {
            Handle::Writer(w) => Ok(w),
            Handle::Reader(_) => Err("this vector store was opened read-only".into()),
        }
    }
}

fn check_config(stored: &IndexConfig, expected: &IndexConfig, dir: &Path) -> Result<(), Error> {
    if stored == expected {
        return Ok(());
    }
    let model = |c: &IndexConfig| {
        c.embedding
            .as_ref()
            .map_or_else(|| "unknown vectors".to_owned(), ToString::to_string)
    };
    Err(if stored.embedding != expected.embedding {
        format!(
            "the vector store {} holds {}, not {}",
            dir.display(),
            model(stored),
            model(expected)
        )
    } else {
        format!(
            "the vector store {} was created with other settings ({} codes, {:?})",
            dir.display(),
            stored.codes.name(),
            stored.ann.kind
        )
    }
    .into())
}

/// Opens the store in `dir` for writing with `config`, creating it when there is
/// none. A store that cannot be used (another model or settings, a format this
/// version cannot read, damage) is replaced by an empty one. Returns whether the store
/// is new.
pub(crate) fn open_or_create(dir: &Path, config: &StoreConfig) -> Result<(MhrStore, bool), Error> {
    let expected = config.index_config()?;
    match mhr_vector::Writer::open(dir, store_options()) {
        Ok(writer) if *writer.config() == expected => {
            return Ok((MhrStore::from_writer(writer), false));
        }
        Ok(_)
        | Err(
            mhr_vector::Error::NotFound(_)
            | mhr_vector::Error::Corrupt { .. }
            | mhr_vector::Error::UnsupportedFormat { .. }
            | mhr_vector::Error::InvalidConfig(_),
        ) => {}
        Err(e) => return Err(e.into()),
    }
    match std::fs::remove_dir_all(dir) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    Ok((MhrStore::create(dir, config)?, true))
}

impl VectorStore for MhrStore {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn len(&self) -> usize {
        self.snapshot().len() as usize
    }

    fn upsert(&mut self, records: &[VectorRecord<'_>]) -> Result<(), Error> {
        let keys: Vec<String> = records.iter().map(|r| key(r.id)).collect();
        let ops: Vec<Op<'_>> = records
            .iter()
            .zip(&keys)
            .map(|(r, key)| {
                Op::Upsert(Record {
                    key,
                    vector: r.vector,
                    path: r.meta.path,
                    language: r.meta.language,
                })
            })
            .collect();
        self.writer()?.apply(&ops)?;
        self.dirty |= !ops.is_empty();
        Ok(())
    }

    fn delete(&mut self, ids: &[VectorId]) -> Result<usize, Error> {
        let keys: Vec<String> = ids.iter().map(|id| key(*id)).collect();
        let ops: Vec<Op<'_>> = keys.iter().map(|k| Op::Delete(k)).collect();
        let deleted = self.writer()?.apply(&ops)?.deleted;
        self.dirty |= deleted > 0;
        Ok(deleted)
    }

    fn ids(&self) -> Result<Vec<VectorId>, Error> {
        self.snapshot()
            .keys()
            .iter()
            .map(|k| parse_key(k))
            .collect()
    }

    fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<VectorHit>, Error> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let hits = self.snapshot().search(
            query,
            k.min(mhr_vector::MAX_K),
            filter.to_mhr().as_ref(),
            &SearchParams::default(),
        )?;
        hits.into_iter()
            .map(|h| {
                Ok(VectorHit {
                    id: parse_key(&h.key)?,
                    score: h.score,
                })
            })
            .collect()
    }

    fn persist(&mut self) -> Result<(), Error> {
        if !self.dirty {
            return Ok(());
        }
        let writer = self.writer()?;
        writer.maintain()?;
        writer.commit(None)?;
        self.dirty = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model: &str, dims: usize, quantization: Quantization) -> StoreConfig {
        StoreConfig::new(ModelDescriptor::new("test", model, dims), "test-format")
            .with_quantization(quantization)
    }

    /// Small deltas, so a few hundred vectors make several segments.
    fn small_segments() -> StoreOptions {
        StoreOptions {
            threads: 2,
            delta_max_vectors: 64,
            ..StoreOptions::default()
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

    const PATHS: [(&str, &str); 5] = [
        ("src/a.rs", "rust"),
        ("src/b.py", "python"),
        ("docs/c.md", "text"),
        ("src/net/d.rs", "rust"),
        ("src/network.rs", "rust"),
    ];

    type Vectors = [(VectorId, Vec<f32>, usize)];

    /// Exact reference ranking in memory: f64 cosine, ties by id.
    fn reference(vectors: &Vectors, q: &[f32], k: usize, f: &Filter) -> Vec<VectorId> {
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let mut scored: Vec<(f64, VectorId)> = vectors
            .iter()
            .filter(|(_, _, p)| f.matches(PATHS[*p].0, PATHS[*p].1))
            .map(|(id, v, _)| {
                let dot: f64 = v
                    .iter()
                    .zip(q)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                (dot / norm(v) / norm(q), *id)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn records(vectors: &Vectors) -> Vec<VectorRecord<'_>> {
        vectors
            .iter()
            .map(|(id, v, p)| VectorRecord {
                id: *id,
                vector: v,
                meta: VectorMeta {
                    path: PATHS[*p].0,
                    language: PATHS[*p].1,
                },
            })
            .collect()
    }

    fn filters() -> Vec<Filter> {
        vec![
            Filter::default(),
            Filter::default().with_path_prefix("src"),
            Filter::default().with_path_prefix("src/net/"),
            Filter::default().with_path_prefix("src/n"),
            Filter::default().with_language("rust"),
            Filter::default()
                .with_path_prefix("src/net")
                .with_language("rust"),
            Filter::default().with_language("go"),
        ]
    }

    fn assert_exact(store: &MhrStore, vectors: &Vectors, dims: usize) {
        for filter in filters() {
            for seed in [7u64, 11, 500, 9001] {
                let q = vector(seed, dims);
                let got: Vec<VectorId> = store
                    .search(&q, 10, &filter)
                    .unwrap()
                    .iter()
                    .map(|h| h.id)
                    .collect();
                assert_eq!(got, reference(vectors, &q, 10, &filter), "{filter:?}");
            }
        }
    }

    #[test]
    fn filters_match_whole_path_components() {
        let f = Filter::default().with_path_prefix("src/net");
        assert!(f.matches("src/net/retry.rs", "rust"));
        assert!(f.matches("src/net", "rust"));
        assert!(!f.matches("src/network.rs", "rust"));
        assert!(
            Filter::default()
                .with_path_prefix("src/")
                .matches("src/a.rs", "rust")
        );
        assert!(
            Filter::default()
                .with_path_prefix("")
                .matches("a.rs", "rust")
        );
        assert!(!f.with_language("go").matches("src/net/x.rs", "rust"));
    }

    #[test]
    fn search_equals_exact_search_with_filters_and_ties() {
        let dims = 19; // not a multiple of any SIMD width
        for quantization in [Quantization::F32, Quantization::Int8] {
            let dir = tempfile::tempdir().unwrap();
            let config = config("m", dims, quantization);
            let mut store = MhrStore::create_with(dir.path(), &config, small_segments()).unwrap();
            let mut vectors: Vec<(VectorId, Vec<f32>, usize)> = (0..600u64)
                .map(|i| {
                    let id = u128::from(i) * 0x1_0000_0001 + 3;
                    (id, vector(i, dims), i as usize % PATHS.len())
                })
                .collect();
            // Two identical vectors tie: the lower id comes first.
            vectors.push((1, vector(7, dims), 0));
            for batch in vectors.chunks(100) {
                store.upsert(&records(batch)).unwrap();
            }
            // In the delta and in segments alike.
            assert_exact(&store, &vectors, dims);
            let hits = store
                .search(&vector(7, dims), 2, &Filter::default())
                .unwrap();
            assert_eq!((hits[0].id, hits[1].id), (1, 7 * 0x1_0000_0001 + 3));
            assert!((hits[0].score - 1.0).abs() < 1e-5);
            store.persist().unwrap();
            assert_exact(&store, &vectors, dims);
            drop(store);
            let store = MhrStore::open(dir.path()).unwrap();
            assert_eq!(store.len(), vectors.len());
            assert_eq!(store.dimensions(), dims);
            assert_exact(&store, &vectors, dims);
            let none = store.search(&vector(1, dims), 0, &Filter::default());
            assert!(none.unwrap().is_empty());
        }
    }

    #[test]
    fn incremental_upserts_deletes_and_reopening() {
        let dims = 16;
        let dir = tempfile::tempdir().unwrap();
        let config = config("m", dims, Quantization::Int8);
        let mut store = MhrStore::create_with(dir.path(), &config, small_segments()).unwrap();
        let mut vectors: Vec<(VectorId, Vec<f32>, usize)> = (0..300u64)
            .map(|i| (u128::from(i), vector(i, dims), i as usize % PATHS.len()))
            .collect();
        store.upsert(&records(&vectors)).unwrap();
        store.persist().unwrap();
        drop(store);

        // Reopened for writing: delete some, replace some (another vector and path),
        // add some.
        let mut store = MhrStore::open_for(dir.path(), &config).unwrap();
        assert_eq!(store.len(), 300);
        let gone: Vec<VectorId> = (0..300).step_by(3).collect();
        assert_eq!(store.delete(&gone).unwrap(), 100);
        assert_eq!(store.delete(&gone).unwrap(), 0);
        vectors.retain(|(id, _, _)| id % 3 != 0);
        for (id, v, p) in vectors.iter_mut().filter(|(id, _, _)| id % 5 == 1) {
            *v = vector(u64::try_from(*id).unwrap() + 10_000, dims);
            *p = (*p + 1) % PATHS.len();
        }
        vectors.extend((1000..1050u64).map(|i| (u128::from(i), vector(i, dims), 3)));
        let changed: Vec<_> = vectors
            .iter()
            .filter(|(id, _, _)| id % 5 == 1 || *id >= 1000)
            .cloned()
            .collect();
        store.upsert(&records(&changed)).unwrap();
        assert_exact(&store, &vectors, dims);
        store.persist().unwrap();
        drop(store);

        let store = MhrStore::open(dir.path()).unwrap();
        let mut ids: Vec<VectorId> = vectors.iter().map(|(id, _, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(store.ids().unwrap(), ids);
        assert_exact(&store, &vectors, dims);
    }

    #[test]
    fn a_crash_between_apply_and_commit_keeps_the_last_commit() {
        let dims = 8;
        let dir = tempfile::tempdir().unwrap();
        let config = config("m", dims, Quantization::Int8);
        let mut store = MhrStore::create_with(dir.path(), &config, small_segments()).unwrap();
        let vectors: Vec<(VectorId, Vec<f32>, usize)> = (0..200u64)
            .map(|i| (u128::from(i), vector(i, dims), i as usize % PATHS.len()))
            .collect();
        store.upsert(&records(&vectors)).unwrap();
        store.persist().unwrap();
        // Applied, sealed into segment files, never committed: as if the process died.
        let more: Vec<_> = (500..800u64)
            .map(|i| (u128::from(i), vector(i, dims), 1))
            .collect();
        store.upsert(&records(&more)).unwrap();
        assert_eq!(store.delete(&[1, 2, 3]).unwrap(), 3);
        assert_eq!(store.len(), 497);
        drop(store);

        let store = MhrStore::open(dir.path()).unwrap();
        assert_eq!(store.len(), 200);
        assert_exact(&store, &vectors, dims);
        let mut store = MhrStore::open_for(dir.path(), &config).unwrap();
        assert_eq!(store.ids().unwrap(), (0..200).collect::<Vec<_>>());
        store.persist().unwrap();
    }

    #[test]
    fn another_model_or_setting_is_refused_or_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let a = config("model-a", 8, Quantization::Int8);
        let mut store = MhrStore::create(dir.path(), &a).unwrap();
        store.upsert(&records(&[(5, vector(5, 8), 0)])).unwrap();
        store.persist().unwrap();
        assert!(store.matches(&a));
        drop(store);
        assert!(MhrStore::create(dir.path(), &a).is_err());

        let b = config("model-b", 8, Quantization::Int8);
        assert!(!MhrStore::open(dir.path()).unwrap().matches(&b));
        let err = MhrStore::open_for(dir.path(), &b).err().unwrap();
        assert!(err.to_string().contains("holds test/model-a"), "{err}");
        let f32 = config("model-a", 8, Quantization::F32);
        let err = MhrStore::open_for(dir.path(), &f32).err().unwrap();
        assert!(err.to_string().contains("other settings"), "{err}");
        assert_ne!(a.dir_name().unwrap(), b.dir_name().unwrap());
        assert_ne!(a.dir_name().unwrap(), f32.dir_name().unwrap());
        assert_eq!(a.dir_name().unwrap(), a.clone().dir_name().unwrap());

        // The same model and settings: kept. Others: replaced by an empty store.
        let (store, created) = open_or_create(dir.path(), &a).unwrap();
        assert!(!created && store.len() == 1);
        drop(store);
        let (store, created) = open_or_create(dir.path(), &b).unwrap();
        assert!(created && store.is_empty() && store.matches(&b));
    }

    #[test]
    fn damage_fails_clearly_and_is_replaced_by_a_writer() {
        let dir = tempfile::tempdir().unwrap();
        let a = config("model-a", 8, Quantization::Int8);
        let mut store = MhrStore::create(dir.path(), &a).unwrap();
        store.upsert(&records(&[(5, vector(5, 8), 0)])).unwrap();
        store.persist().unwrap();
        drop(store);
        std::fs::write(dir.path().join("CURRENT"), b"garbage").unwrap();
        assert!(MhrStore::open(dir.path()).is_err());
        let (store, created) = open_or_create(dir.path(), &a).unwrap();
        assert!(created && store.is_empty());
    }

    #[test]
    fn rejects_bad_vectors_and_writes_to_a_reader() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = MhrStore::create(dir.path(), &config("m", 4, Quantization::F32)).unwrap();
        let meta = VectorMeta {
            path: "a.rs",
            language: "rust",
        };
        for bad in [&[0.0; 4][..], &[1.0; 3], &[f32::NAN, 1.0, 1.0, 1.0]] {
            let record = VectorRecord {
                id: 1,
                vector: bad,
                meta,
            };
            assert!(store.upsert(&[record]).is_err());
        }
        let good = VectorRecord {
            id: 1,
            vector: &[1.0; 4],
            meta,
        };
        store.upsert(&[good]).unwrap();
        assert!(store.search(&[0.0; 4], 1, &Filter::default()).is_err());
        store.persist().unwrap();
        drop(store);
        let mut reader = MhrStore::open(dir.path()).unwrap();
        assert!(reader.upsert(&[good]).is_err());
        assert!(reader.delete(&[1]).is_err());
        let hits = reader.search(&[1.0; 4], 5, &Filter::default()).unwrap();
        assert_eq!(hits.len(), 1);
    }
}
