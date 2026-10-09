//! The embedding cache of a state directory: vectors by (model descriptor, SHA-256 of
//! the rendered text), kept across index generations so a rebuild embeds only new or
//! changed texts, and a model change embeds everything with the new model (each
//! descriptor has its own directory, so switching back is free).
//!
//! `STATE_DIR/embed-cache/<descriptor key>/`:
//!
//! ```text
//! cache.json   version, the full descriptor, and the current epoch E
//! keys.E       32-byte text hashes, one per record
//! vectors.E    the records' vectors, little-endian f32 (as the provider returned
//!              them, not normalized), in the same order
//! ```
//!
//! Records are appended (vectors first, both fsynced) after every request, so tokens
//! spent are never lost when a run stops. On open, a torn tail (a crash between
//! the two appends) is cut to the records both files hold. Compaction writes the live
//! records to the next epoch's files and switches `cache.json` atomically.
//! An embedding run writes the cache under the state's embedding lock. A build reads
//! which texts it holds ([`cached_hashes`]) without writing anything, and compacts it
//! after publishing, holding the index writer lock (no other build reads it) and the
//! embedding lock (no run writes it).
use super::ModelDescriptor;
use crate::{
    bench_v1::Error,
    fsio::{atomic_write, read_exact_at, write_all_at},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

pub(crate) const CACHE_DIR: &str = "embed-cache";

#[derive(Serialize, Deserialize)]
struct CacheMeta {
    version: u32,
    descriptor: ModelDescriptor,
    epoch: u64,
}

pub(crate) struct Cache {
    dir: PathBuf,
    descriptor: ModelDescriptor,
    dims: usize,
    epoch: u64,
    index: HashMap<[u8; 32], u64>,
    records: u64,
    keys: File,
    vectors: File,
}

/// The text hashes the cache of `descriptor` holds, read without writing anything (a
/// run may be appending meanwhile: the records both files hold so far). Empty when
/// there is no cache for the model.
pub(crate) fn cached_hashes(
    state_dir: &Path,
    descriptor: &ModelDescriptor,
) -> Result<HashSet<[u8; 32]>, Error> {
    let dir = state_dir.join(CACHE_DIR).join(descriptor.key());
    let meta: CacheMeta = match std::fs::read(dir.join("cache.json")) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashSet::new()),
        Err(e) => return Err(e.into()),
    };
    if meta.version != 1 || &meta.descriptor != descriptor {
        // Cache::open explains it when a run opens the cache.
        return Ok(HashSet::new());
    }
    let open = |name: String| match File::open(dir.join(name)) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    };
    let (Some(keys), Some(vectors)) = (
        open(format!("keys.{}", meta.epoch))?,
        open(format!("vectors.{}", meta.epoch))?,
    ) else {
        return Ok(HashSet::new());
    };
    let row = (descriptor.dimensions.max(1) * 4) as u64;
    let records = (keys.metadata()?.len() / 32).min(vectors.metadata()?.len() / row);
    let mut raw = vec![0u8; (records * 32) as usize];
    read_exact_at(&keys, &mut raw, 0)?;
    Ok(raw
        .chunks_exact(32)
        .map(|k| k.try_into().unwrap())
        .collect())
}

fn open_rw(path: &Path) -> Result<File, Error> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?)
}

impl Cache {
    pub(crate) fn open(state_dir: &Path, descriptor: &ModelDescriptor) -> Result<Self, Error> {
        if descriptor.dimensions == 0 {
            return Err("embedding dimensions must be at least 1".into());
        }
        let dir = state_dir.join(CACHE_DIR).join(descriptor.key());
        std::fs::create_dir_all(&dir)?;
        let epoch = match std::fs::read(dir.join("cache.json")) {
            Ok(bytes) => {
                let meta: CacheMeta = serde_json::from_slice(&bytes)?;
                if meta.version != 1 {
                    return Err(
                        format!("unsupported embedding cache version {}", meta.version).into(),
                    );
                }
                if &meta.descriptor != descriptor {
                    return Err(format!(
                        "embedding cache {} belongs to another model ({})",
                        dir.display(),
                        meta.descriptor.display()
                    )
                    .into());
                }
                meta.epoch
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let meta = CacheMeta {
                    version: 1,
                    descriptor: descriptor.clone(),
                    epoch: 0,
                };
                atomic_write(&dir, "cache.json", &serde_json::to_vec_pretty(&meta)?)?;
                0
            }
            Err(e) => return Err(e.into()),
        };
        // Files of other epochs: an interrupted compaction's output, or the input of
        // a finished one.
        for entry in std::fs::read_dir(&dir)? {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            if let Some((stem, e)) = name.split_once('.')
                && matches!(stem, "keys" | "vectors")
                && e.parse::<u64>().is_ok_and(|e| e != epoch)
            {
                std::fs::remove_file(dir.join(name.as_ref()))?;
            }
        }
        let keys = open_rw(&dir.join(format!("keys.{epoch}")))?;
        let vectors = open_rw(&dir.join(format!("vectors.{epoch}")))?;
        let row = (descriptor.dimensions * 4) as u64;
        let records = (keys.metadata()?.len() / 32).min(vectors.metadata()?.len() / row);
        keys.set_len(records * 32)?;
        vectors.set_len(records * row)?;
        let mut raw = vec![0u8; (records * 32) as usize];
        read_exact_at(&keys, &mut raw, 0)?;
        let mut index = HashMap::with_capacity(records as usize);
        for (i, k) in raw.chunks_exact(32).enumerate() {
            index.entry(k.try_into().unwrap()).or_insert(i as u64);
        }
        Ok(Self {
            dir,
            descriptor: descriptor.clone(),
            dims: descriptor.dimensions,
            epoch,
            index,
            records,
            keys,
            vectors,
        })
    }

    pub(crate) fn contains(&self, hash: &[u8; 32]) -> bool {
        self.index.contains_key(hash)
    }

    /// Reads the vector of `hash` into `out`; false when it is not cached.
    pub(crate) fn get(&self, hash: &[u8; 32], out: &mut Vec<f32>) -> Result<bool, Error> {
        let Some(&rec) = self.index.get(hash) else {
            return Ok(false);
        };
        let row = self.dims * 4;
        let mut bytes = vec![0u8; row];
        read_exact_at(&self.vectors, &mut bytes, rec * row as u64)?;
        out.clear();
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        Ok(true)
    }

    /// Appends records durably: vectors, fsync, keys, fsync.
    pub(crate) fn append(&mut self, items: &[([u8; 32], Vec<f32>)]) -> Result<(), Error> {
        let row = self.dims * 4;
        let mut vbytes = Vec::with_capacity(items.len() * row);
        let mut kbytes = Vec::with_capacity(items.len() * 32);
        for (hash, v) in items {
            if v.len() != self.dims {
                return Err(format!(
                    "{} returned a vector of {} dimensions, not {}",
                    self.descriptor.display(),
                    v.len(),
                    self.dims
                )
                .into());
            }
            for x in v {
                vbytes.extend_from_slice(&x.to_le_bytes());
            }
            kbytes.extend_from_slice(hash);
        }
        write_all_at(&self.vectors, &vbytes, self.records * row as u64)?;
        self.vectors.sync_data()?;
        write_all_at(&self.keys, &kbytes, self.records * 32)?;
        self.keys.sync_data()?;
        for (hash, _) in items {
            self.index.entry(*hash).or_insert(self.records);
            self.records += 1;
        }
        Ok(())
    }

    /// Rewrites the cache with only the `live` texts when other records outnumber
    /// them (more than twice as many records as live texts, plus 1,024). Returns
    /// whether it compacted.
    pub(crate) fn compact_if_sparse(&mut self, live: &HashSet<[u8; 32]>) -> Result<bool, Error> {
        if self.records <= 2 * live.len() as u64 + 1024 {
            return Ok(false);
        }
        let next = self.epoch + 1;
        let keys_path = self.dir.join(format!("keys.{next}"));
        let vectors_path = self.dir.join(format!("vectors.{next}"));
        let keys = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&keys_path)?;
        let vectors = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&vectors_path)?;
        // Keep the old record order (by record number) so a compaction is deterministic.
        let mut kept: Vec<(u64, [u8; 32])> = self
            .index
            .iter()
            .filter(|(h, _)| live.contains(*h))
            .map(|(h, r)| (*r, *h))
            .collect();
        kept.sort_unstable();
        let row = self.dims * 4;
        let mut bytes = vec![0u8; row];
        let mut index = HashMap::with_capacity(kept.len());
        for (i, (rec, hash)) in kept.iter().enumerate() {
            read_exact_at(&self.vectors, &mut bytes, rec * row as u64)?;
            write_all_at(&vectors, &bytes, (i * row) as u64)?;
            write_all_at(&keys, hash, (i * 32) as u64)?;
            index.insert(*hash, i as u64);
        }
        vectors.sync_all()?;
        keys.sync_all()?;
        let meta = CacheMeta {
            version: 1,
            descriptor: self.descriptor.clone(),
            epoch: next,
        };
        atomic_write(&self.dir, "cache.json", &serde_json::to_vec_pretty(&meta)?)?;
        let old = self.epoch;
        self.epoch = next;
        self.index = index;
        self.records = kept.len() as u64;
        self.keys = keys;
        self.vectors = vectors;
        for name in [format!("keys.{old}"), format!("vectors.{old}")] {
            std::fs::remove_file(self.dir.join(name))?;
        }
        Ok(true)
    }

    pub(crate) fn records(&self) -> u64 {
        self.records
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u8) -> [u8; 32] {
        [i; 32]
    }

    #[test]
    fn appends_survive_reopen_torn_tails_and_compaction() {
        let state = tempfile::tempdir().unwrap();
        let d = ModelDescriptor::new("hash", "hash-v1", 3);
        let mut c = Cache::open(state.path(), &d).unwrap();
        c.append(&[(h(1), vec![1.0, 2.0, 3.0]), (h(2), vec![4.0, 5.0, 6.0])])
            .unwrap();
        drop(c);
        // A torn tail: half a vector record and no key.
        let dir = state.path().join(CACHE_DIR).join(d.key());
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.join("vectors.0"))
            .unwrap();
        std::io::Write::write_all(&mut f, &[0u8; 6]).unwrap();
        let mut c = Cache::open(state.path(), &d).unwrap();
        assert_eq!(c.records(), 2);
        let mut v = Vec::new();
        assert!(c.get(&h(2), &mut v).unwrap());
        assert_eq!(v, [4.0, 5.0, 6.0]);
        assert!(!c.get(&h(3), &mut v).unwrap());
        assert!(c.append(&[(h(3), vec![1.0])]).is_err());
        // Another model has its own directory; a foreign cache.json is refused.
        let other = ModelDescriptor::new("hash", "hash-v1", 4);
        assert_ne!(other.key(), d.key());
        drop(Cache::open(state.path(), &other).unwrap());
        let other_dir = state.path().join(CACHE_DIR).join(other.key());
        std::fs::copy(dir.join("cache.json"), other_dir.join("cache.json")).unwrap();
        assert!(Cache::open(state.path(), &other).is_err());
        // Compaction keeps the live records only.
        let many: Vec<_> = (10..=255u8)
            .map(|i| (h(i), vec![f32::from(i); 3]))
            .collect();
        for part in many.chunks(50) {
            c.append(part).unwrap();
        }
        c.append(
            &(0..1024u32)
                .map(|i| {
                    let mut k = [0u8; 32];
                    k[..4].copy_from_slice(&i.to_le_bytes());
                    k[31] = 9;
                    (k, vec![1.0; 3])
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let live: HashSet<_> = [h(2), h(200)].into();
        assert!(c.compact_if_sparse(&live).unwrap());
        assert_eq!(c.records(), 2);
        drop(c);
        let c = Cache::open(state.path(), &d).unwrap();
        assert_eq!(c.records(), 2);
        assert!(c.get(&h(200), &mut v).unwrap());
        assert_eq!(v, [200.0; 3]);
        assert!(!c.contains(&h(1)));
        assert!(!dir.join("keys.0").exists());
    }
}
