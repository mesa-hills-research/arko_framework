//! The embedding cache of a state directory, holding vectors by (model descriptor, SHA-256 of
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
//! embedding lock (no run writes it). A build or run may also borrow vectors from other
//! state directories' caches ([`Cache::borrow_from`]), reading them without locks and
//! copying valid hits into its own cache so each index stays self-contained.
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
/// Vectors appended to the target cache per flush when borrowing, keeping memory bounded.
const BORROW_BATCH: usize = 1024;
/// Bytes of a donor's `keys.<epoch>` scanned per read (a multiple of 32).
const KEY_SCAN_BYTES: usize = 1 << 20;

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
/// run may be appending meanwhile, and the records both files hold so far are read). Empty when
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

/// A donor state directory's cache opened read-only without taking any lock.
struct DonorCache {
    keys: File,
    vectors: File,
    records: u64,
    dims: usize,
}

fn open_donor(state_dir: &Path, descriptor: &ModelDescriptor) -> Option<DonorCache> {
    if descriptor.dimensions == 0 {
        return None;
    }
    let row = u64::try_from(descriptor.dimensions).ok()?.checked_mul(4)?;
    let dir = state_dir.join(CACHE_DIR).join(descriptor.key());
    let bytes = std::fs::read(dir.join("cache.json")).ok()?;
    let meta: CacheMeta = serde_json::from_slice(&bytes).ok()?;
    if meta.version != 1 || &meta.descriptor != descriptor {
        return None;
    }
    let keys = File::open(dir.join(format!("keys.{}", meta.epoch))).ok()?;
    let vectors = File::open(dir.join(format!("vectors.{}", meta.epoch))).ok()?;
    let keys_len = keys.metadata().ok()?.len();
    let vectors_len = vectors.metadata().ok()?.len();
    let records = (keys_len / 32).min(vectors_len / row);
    if records == 0 {
        return None;
    }
    Some(DonorCache {
        keys,
        vectors,
        records,
        dims: descriptor.dimensions,
    })
}

/// Whether `v` is finite, has `dims` components and can be L2-normalized in `f32`
/// (matching the vector store's requirements).
fn is_valid_vector(v: &[f32], dims: usize) -> bool {
    if v.len() != dims || v.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let squared: f32 = v.iter().map(|x| x * x).sum();
    if squared == 0.0 || !squared.is_finite() {
        return false;
    }
    let norm = squared.sqrt();
    v.iter().all(|x| (x / norm).is_finite())
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
        // Files of other epochs, such as an interrupted compaction's output, or the input of
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

    /// Reads the vector of `hash` into `out`. False when it is not cached.
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

    /// Appends records durably (vectors, fsync, keys, fsync).
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

    /// Copies vectors for any hashes in `wanted` that this cache lacks from the given
    /// donor state directories, appending them durably into this cache. Donor caches are
    /// only read without taking their locks, and any donor that is missing, uses another
    /// model, has corrupt or invalid vectors, or fails to read counts as a miss. Returns
    /// the hashes copied from donors.
    pub(crate) fn borrow_from(
        &mut self,
        own_state_dir: &Path,
        donors: &[PathBuf],
        wanted: &HashSet<[u8; 32]>,
    ) -> Result<HashSet<[u8; 32]>, Error> {
        if donors.is_empty() || wanted.is_empty() {
            return Ok(HashSet::new());
        }
        let mut remaining: HashSet<[u8; 32]> = wanted
            .iter()
            .copied()
            .filter(|h| !self.contains(h))
            .collect();
        if remaining.is_empty() {
            return Ok(HashSet::new());
        }
        let own_canon = std::fs::canonicalize(own_state_dir).ok();
        let mut visited: HashSet<PathBuf> = HashSet::new();
        let mut borrowed: HashSet<[u8; 32]> = HashSet::new();
        for donor in donors {
            if remaining.is_empty() {
                break;
            }
            if donor == own_state_dir {
                continue;
            }
            let donor_canon = std::fs::canonicalize(donor).ok();
            if let (Some(oc), Some(dc)) = (&own_canon, &donor_canon)
                && oc == dc
            {
                continue;
            }
            let dedup_key = donor_canon.unwrap_or_else(|| donor.clone());
            if !visited.insert(dedup_key) {
                continue;
            }
            self.borrow_from_one(donor, &mut remaining, &mut borrowed)?;
        }
        Ok(borrowed)
    }

    fn borrow_from_one(
        &mut self,
        donor_state_dir: &Path,
        remaining: &mut HashSet<[u8; 32]>,
        borrowed: &mut HashSet<[u8; 32]>,
    ) -> Result<(), Error> {
        let Some(donor) = open_donor(donor_state_dir, &self.descriptor) else {
            return Ok(());
        };
        let row = donor.dims * 4;
        let row_u64 = row as u64;
        let keys_per_chunk = (KEY_SCAN_BYTES / 32) as u64;
        let mut key_buf = Vec::new();
        let mut vec_buf = vec![0u8; row];
        let mut batch: Vec<([u8; 32], Vec<f32>)> = Vec::new();
        let mut rec_base = 0u64;
        'scan: while rec_base < donor.records && !remaining.is_empty() {
            let count = (donor.records - rec_base).min(keys_per_chunk) as usize;
            let Some(key_off) = rec_base.checked_mul(32) else {
                break;
            };
            key_buf.resize(count * 32, 0);
            if read_exact_at(&donor.keys, &mut key_buf, key_off).is_err() {
                break;
            }
            for (i, chunk) in key_buf.chunks_exact(32).enumerate() {
                let hash: [u8; 32] = chunk.try_into().unwrap();
                if !remaining.contains(&hash) {
                    continue;
                }
                let Some(vec_off) = (rec_base + i as u64).checked_mul(row_u64) else {
                    continue;
                };
                if read_exact_at(&donor.vectors, &mut vec_buf, vec_off).is_err() {
                    break 'scan;
                }
                let vector: Vec<f32> = vec_buf
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                if !is_valid_vector(&vector, donor.dims) {
                    continue;
                }
                remaining.remove(&hash);
                batch.push((hash, vector));
                if batch.len() >= BORROW_BATCH {
                    self.append(&batch)?;
                    borrowed.extend(batch.drain(..).map(|(h, _)| h));
                }
                if remaining.is_empty() {
                    break 'scan;
                }
            }
            rec_base += count as u64;
        }
        if !batch.is_empty() {
            self.append(&batch)?;
            borrowed.extend(batch.drain(..).map(|(h, _)| h));
        }
        Ok(())
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
        // A torn tail, half a vector record and no key.
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
        // Another model has its own directory, and a foreign cache.json is refused.
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

    #[test]
    fn borrow_from_copies_valid_vectors_and_treats_damage_as_misses() {
        let d = ModelDescriptor::new("hash", "hash-v1", 3);
        let donor1 = tempfile::tempdir().unwrap();
        let donor2 = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();

        // Donor 1 holds h(1), h(2) and h(3), with h(2) corrupted to NaN and h(3) zeroed.
        let mut c1 = Cache::open(donor1.path(), &d).unwrap();
        c1.append(&[
            (h(1), vec![1.0, 2.0, 3.0]),
            (h(2), vec![4.0, 5.0, 6.0]),
            (h(3), vec![7.0, 8.0, 9.0]),
        ])
        .unwrap();
        drop(c1);
        let d1_vecs = donor1
            .path()
            .join(CACHE_DIR)
            .join(d.key())
            .join("vectors.0");
        let mut raw = std::fs::read(&d1_vecs).unwrap();
        // Record 1 (bytes 12..24) gets a NaN in its first float.
        raw[12..16].copy_from_slice(&f32::NAN.to_le_bytes());
        // Record 2 (bytes 24..36) is all zeros.
        raw[24..36].fill(0);
        std::fs::write(&d1_vecs, raw).unwrap();

        // Donor 2 holds valid h(2), subnormal-zero-norm h(3), and valid h(4).
        let mut c2 = Cache::open(donor2.path(), &d).unwrap();
        c2.append(&[
            (h(2), vec![40.0, 50.0, 60.0]),
            (h(3), vec![f32::MIN_POSITIVE, 0.0, 0.0]),
            (h(4), vec![9.0, 8.0, 7.0]),
        ])
        .unwrap();
        drop(c2);

        // Damaged donors (missing path, empty dir, corrupt cache.json, wrong version,
        // mismatched descriptor, and truncated vectors file).
        let missing = tempfile::tempdir().unwrap().path().join("does-not-exist");
        let empty = tempfile::tempdir().unwrap();
        let corrupt_json = tempfile::tempdir().unwrap();
        let cj_dir = corrupt_json.path().join(CACHE_DIR).join(d.key());
        std::fs::create_dir_all(&cj_dir).unwrap();
        std::fs::write(cj_dir.join("cache.json"), b"{not json").unwrap();

        let bad_version = tempfile::tempdir().unwrap();
        let bv_dir = bad_version.path().join(CACHE_DIR).join(d.key());
        std::fs::create_dir_all(&bv_dir).unwrap();
        std::fs::write(
            bv_dir.join("cache.json"),
            serde_json::to_vec(&CacheMeta {
                version: 99,
                descriptor: d.clone(),
                epoch: 0,
            })
            .unwrap(),
        )
        .unwrap();

        let wrong_model = tempfile::tempdir().unwrap();
        let wm_dir = wrong_model.path().join(CACHE_DIR).join(d.key());
        std::fs::create_dir_all(&wm_dir).unwrap();
        std::fs::write(
            wm_dir.join("cache.json"),
            serde_json::to_vec(&CacheMeta {
                version: 1,
                descriptor: ModelDescriptor::new("hash", "other", 3),
                epoch: 0,
            })
            .unwrap(),
        )
        .unwrap();

        let truncated = tempfile::tempdir().unwrap();
        let mut ct = Cache::open(truncated.path(), &d).unwrap();
        ct.append(&[(h(5), vec![1.0, 1.0, 1.0]), (h(6), vec![2.0, 2.0, 2.0])])
            .unwrap();
        drop(ct);
        // Cut vectors.0 to 1.5 records so only h(5) is complete and h(6) is a miss.
        let t_vecs = truncated
            .path()
            .join(CACHE_DIR)
            .join(d.key())
            .join("vectors.0");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&t_vecs)
            .unwrap()
            .set_len(18)
            .unwrap();

        let mut target_cache = Cache::open(target.path(), &d).unwrap();
        target_cache.append(&[(h(9), vec![9.0, 9.0, 9.0])]).unwrap();

        let donors = vec![
            target.path().to_path_buf(),
            missing,
            empty.path().to_path_buf(),
            corrupt_json.path().to_path_buf(),
            bad_version.path().to_path_buf(),
            wrong_model.path().to_path_buf(),
            donor1.path().to_path_buf(),
            donor1.path().to_path_buf(),
            donor2.path().to_path_buf(),
            truncated.path().to_path_buf(),
        ];
        let wanted: HashSet<_> = [h(1), h(2), h(3), h(4), h(5), h(6), h(9)].into();
        let borrowed = target_cache
            .borrow_from(target.path(), &donors, &wanted)
            .unwrap();
        assert_eq!(borrowed, [h(1), h(2), h(4), h(5)].into());

        // Deleting the donors leaves the borrowed vectors intact in target's cache.
        drop((donor1, donor2, truncated, target_cache));
        let reopened = Cache::open(target.path(), &d).unwrap();
        assert_eq!(reopened.records(), 5);
        let mut v = Vec::new();
        assert!(reopened.get(&h(1), &mut v).unwrap());
        assert_eq!(v, [1.0, 2.0, 3.0]);
        assert!(reopened.get(&h(2), &mut v).unwrap());
        assert_eq!(v, [40.0, 50.0, 60.0]);
        assert!(reopened.get(&h(4), &mut v).unwrap());
        assert_eq!(v, [9.0, 8.0, 7.0]);
        assert!(reopened.get(&h(5), &mut v).unwrap());
        assert_eq!(v, [1.0, 1.0, 1.0]);
        assert!(!reopened.contains(&h(3)));
        assert!(!reopened.contains(&h(6)));
    }

    #[test]
    fn borrow_from_flushes_multiple_batches() {
        let d = ModelDescriptor::new("hash", "hash-v1", 2);
        let donor = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let total = BORROW_BATCH + 150;
        let items: Vec<_> = (0..total)
            .map(|i| {
                let mut k = [0u8; 32];
                k[..8].copy_from_slice(&(i as u64).to_le_bytes());
                (k, vec![(i + 1) as f32, 1.0])
            })
            .collect();
        let mut dc = Cache::open(donor.path(), &d).unwrap();
        dc.append(&items).unwrap();
        drop(dc);

        let wanted: HashSet<_> = items.iter().map(|(k, _)| *k).collect();
        let mut tc = Cache::open(target.path(), &d).unwrap();
        let borrowed = tc
            .borrow_from(target.path(), &[donor.path().to_path_buf()], &wanted)
            .unwrap();
        assert_eq!(borrowed.len(), total);
        assert_eq!(tc.records(), total as u64);
        let mut v = Vec::new();
        assert!(tc.get(&items[total - 1].0, &mut v).unwrap());
        assert_eq!(v, items[total - 1].1);
    }
}
