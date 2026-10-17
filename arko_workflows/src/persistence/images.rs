//! The images of a persisted run, saved once each in the run directory's
//! `images` folder. The checkpoints, task files and event log refer to them
//! by name (see [`arko_ai::image_store`]): a conversation saved at every
//! step doesn't rewrite its images.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use arko_ai::image_store::{ImageStore, sample_hash as sample, with_image_store};
use base64::Engine as _;
use parking_lot::Mutex;

/// The folder of a run directory holding its images.
pub const IMAGES_DIR: &str = "images";

/// The most image data (base64) an [`ImageFiles`] keeps in memory to know
/// the images it saved without hashing them again.
const CACHE_BYTES: usize = 256 << 20;

/// A run directory's image files: `images/<SHA-256 of the image>.<type>`,
/// each written once (atomically, flushed to disk before a file that refers
/// to it is written). Files are never removed while the run directory
/// exists: older checkpoints, task versions and the event log may refer to
/// them.
///
/// Wrap code that reads a run directory's task files or checkpoints
/// yourself in [`ImageFiles::with`]; [`WorkflowState::agent_results_in`]
/// (crate::persistence::state::WorkflowState::agent_results_in) does so
/// already.
pub struct ImageFiles {
    dir: PathBuf,
    cache: Mutex<Cache>,
    /// The most data the cache holds ([`CACHE_BYTES`]).
    budget: usize,
}

/// The images saved or read lately, to recognize them without hashing:
/// by length and a sample of their data, then compared in full.
#[derive(Default)]
struct Cache {
    entries: HashMap<(usize, u64), Vec<Entry>>,
    bytes: usize,
    clock: u64,
}

struct Entry {
    data: Arc<str>,
    mime_type: String,
    name: String,
    used: u64,
}

impl ImageFiles {
    /// The image files of run directory `run_dir` (its `images` folder).
    /// Every caller in the process shares one per folder, and with it what
    /// it knows of the images saved.
    pub fn in_run_dir(run_dir: &Path) -> Arc<Self> {
        static OPEN: LazyLock<Mutex<HashMap<PathBuf, Weak<ImageFiles>>>> =
            LazyLock::new(Default::default);
        let dir = run_dir.join(IMAGES_DIR);
        let mut open = OPEN.lock();
        open.retain(|_, files| files.strong_count() > 0);
        if let Some(files) = open.get(&dir).and_then(Weak::upgrade) {
            return files;
        }
        let files = Arc::new(Self::new(dir.clone(), CACHE_BYTES));
        open.insert(dir, Arc::downgrade(&files));
        files
    }

    fn new(dir: PathBuf, budget: usize) -> Self {
        Self {
            dir,
            cache: Mutex::default(),
            budget,
        }
    }

    /// Runs `f` with these files as the thread's image store: images
    /// serialized in `f` are saved here and written by name, and names read
    /// in `f` are loaded from here.
    pub fn with<R>(self: &Arc<Self>, f: impl FnOnce() -> R) -> R {
        with_image_store(self.clone(), f)
    }

    /// The folder holding the files.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The name of the image, if this process saved or read it lately.
    fn known(&self, data: &str, mime_type: &str) -> Option<String> {
        let mut cache = self.cache.lock();
        cache.clock += 1;
        let clock = cache.clock;
        let entry = cache
            .entries
            .get_mut(&(data.len(), sample(data)))?
            .iter_mut()
            .find(|entry| entry.mime_type == mime_type && *entry.data == *data)?;
        entry.used = clock;
        Some(entry.name.clone())
    }

    /// The data of image `name`, if this process saved or read it lately.
    fn known_data(&self, name: &str) -> Option<String> {
        let mut cache = self.cache.lock();
        cache.clock += 1;
        let clock = cache.clock;
        let entry = cache
            .entries
            .values_mut()
            .flatten()
            .find(|entry| entry.name == name)?;
        entry.used = clock;
        Some(entry.data.to_string())
    }

    fn remember(&self, data: &str, mime_type: &str, name: String) {
        let mut cache = self.cache.lock();
        cache.clock += 1;
        let used = cache.clock;
        let bucket = cache.entries.entry((data.len(), sample(data))).or_default();
        if bucket
            .iter()
            .any(|entry| entry.mime_type == mime_type && *entry.data == *data)
        {
            return;
        }
        bucket.push(Entry {
            data: Arc::from(data),
            mime_type: mime_type.to_string(),
            name,
            used,
        });
        cache.bytes += data.len();
        // The least recently used go first; one image larger than the whole
        // budget stays until the next.
        while cache.bytes > self.budget {
            let Some((key, index, len)) = cache
                .entries
                .iter()
                .flat_map(|(key, bucket)| {
                    bucket
                        .iter()
                        .enumerate()
                        .map(move |(index, entry)| (*key, index, entry.data.len(), entry.used))
                })
                .filter(|(.., entry_used)| *entry_used != used)
                .min_by_key(|(.., entry_used)| *entry_used)
                .map(|(key, index, len, _)| (key, index, len))
            else {
                break;
            };
            if let Some(bucket) = cache.entries.get_mut(&key) {
                bucket.swap_remove(index);
                if bucket.is_empty() {
                    cache.entries.remove(&key);
                }
            }
            cache.bytes -= len;
        }
    }

    /// Writes the file of image `name` unless it is there.
    fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        static TEMPORARIES: AtomicU64 = AtomicU64::new(0);
        let path = self.dir.join(name);
        if path.is_file() {
            return Ok(());
        }
        let created = !self.dir.is_dir();
        std::fs::create_dir_all(&self.dir)?;
        if created {
            sync_directory(self.dir.parent().unwrap_or(Path::new(".")));
        }
        let temporary = self.dir.join(format!(
            "{name}.{}-{}{}",
            std::process::id(),
            TEMPORARIES.fetch_add(1, Ordering::Relaxed),
            super::TEMPORARY_SUFFIX
        ));
        let written = (|| {
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temporary, &path)
        })();
        if let Err(error) = written {
            let _ = std::fs::remove_file(&temporary);
            // Another writer may have saved the same image meanwhile.
            if path.is_file() {
                return Ok(());
            }
            return Err(error);
        }
        sync_directory(&self.dir);
        Ok(())
    }
}

impl ImageStore for ImageFiles {
    fn save(&self, data: &str, mime_type: &str) -> Result<String, String> {
        if let Some(name) = self.known(data, mime_type) {
            return Ok(name);
        }
        let (name, bytes) = image_file(data, mime_type);
        self.write(&name, bytes.as_deref().unwrap_or(data.as_bytes()))
            .map_err(|error| {
                format!(
                    "could not save an image to {}: {error}",
                    self.dir.join(&name).display()
                )
            })?;
        self.remember(data, mime_type, name.clone());
        Ok(name)
    }

    fn load(&self, name: &str, mime_type: &str) -> Result<String, String> {
        if !valid_name(name) {
            return Err(format!("'{name}' isn't the name of an image file"));
        }
        if let Some(data) = self.known_data(name) {
            return Ok(data);
        }
        let path = self.dir.join(name);
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("could not read image {}: {error}", path.display()))?;
        // The name is the content's hash: a file cut short or changed is
        // damaged.
        if !name.starts_with(&crate::digest::sha256_hex(&bytes)) {
            return Err(format!(
                "image {} is damaged: its content doesn't match its name",
                path.display()
            ));
        }
        let data = if name.ends_with(TEXT_EXTENSION) {
            String::from_utf8(bytes)
                .map_err(|error| format!("image {} isn't text: {error}", path.display()))?
        } else {
            base64::engine::general_purpose::STANDARD.encode(bytes)
        };
        self.remember(&data, mime_type, name.to_string());
        Ok(data)
    }
}

/// The extension of an image saved as its base64 text (data that doesn't
/// decode as standard base64, which is then kept exactly as given).
const TEXT_EXTENSION: &str = ".b64";

/// The file name of an image (`<SHA-256 of its bytes>.<type>`) and its
/// bytes; `None` for data that isn't standard base64, saved as text.
pub(crate) fn image_file(data: &str, mime_type: &str) -> (String, Option<Vec<u8>>) {
    match base64::engine::general_purpose::STANDARD.decode(data) {
        Ok(bytes) => {
            let hash = crate::digest::sha256_hex(&bytes);
            (format!("{hash}.{}", extension(mime_type)), Some(bytes))
        }
        Err(_) => (
            format!(
                "{}{TEXT_EXTENSION}",
                crate::digest::sha256_hex(data.as_bytes())
            ),
            None,
        ),
    }
}

/// A file extension for `mime_type`: its subtype when that is short and
/// plain (`jpg` for JPEG), else `img`.
fn extension(mime_type: &str) -> String {
    let subtype = mime_type
        .trim()
        .strip_prefix("image/")
        .unwrap_or_default()
        .to_ascii_lowercase();
    match subtype.as_str() {
        "jpeg" => "jpg".to_string(),
        "b64" => "img".to_string(),
        subtype
            if (1..=8).contains(&subtype.len())
                && subtype.bytes().all(|b| b.is_ascii_alphanumeric()) =>
        {
            subtype.to_string()
        }
        _ => "img".to_string(),
    }
}

/// Whether `name` is an image file's name as [`image_file`] makes them, so
/// a name read from a file can't point outside the folder.
fn valid_name(name: &str) -> bool {
    let Some((hash, extension)) = name.split_once('.') else {
        return false;
    };
    hash.len() == 64
        && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        && (1..=8).contains(&extension.len())
        && extension.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// Flushes `directory`'s entries to disk (Unix; Windows can't open a
/// directory for that). A failure is ignored: the file is written either
/// way, and only a crash of the machine could lose its name.
fn sync_directory(directory: &Path) {
    #[cfg(unix)]
    if let Ok(directory) = std::fs::File::open(directory) {
        let _ = directory.sync_all();
    }
    #[cfg(not(unix))]
    let _ = directory;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("arko-images-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An image is saved once, as its bytes, under its hash; the JSON
    /// holds the name, and reads back through the files, also in a fresh
    /// process (an empty cache).
    #[test]
    fn an_image_is_saved_once_and_read_back_by_name() {
        let dir = run_dir();
        let files = ImageFiles::in_run_dir(&dir);
        let image = arko_ai::Image::from_bytes(b"\x89PNG\r\n\x1a\n pixels", "image/png");
        let json = files.with(|| serde_json::to_value(&image).unwrap());
        let name = json["Base64"]["stored"].as_str().unwrap().to_string();
        assert!(name.ends_with(".png"), "{name}");
        assert!(json["Base64"].get("data").is_none(), "{json}");
        let path = dir.join(IMAGES_DIR).join(&name);
        assert_eq!(std::fs::read(&path).unwrap(), b"\x89PNG\r\n\x1a\n pixels");
        let written = std::fs::metadata(&path).unwrap().modified().unwrap();

        // Saved again: the same name, the file untouched.
        let again = files.with(|| serde_json::to_value(image.clone()).unwrap());
        assert_eq!(again, json);
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            written
        );
        assert_eq!(std::fs::read_dir(dir.join(IMAGES_DIR)).unwrap().count(), 1);

        let fresh = Arc::new(ImageFiles::new(dir.join(IMAGES_DIR), CACHE_BYTES));
        let read: arko_ai::Image = fresh.with(|| serde_json::from_value(json.clone()).unwrap());
        assert_eq!(read, image);
        // What was read is known: saving it again doesn't hash it.
        assert_eq!(
            fresh.known(
                match &image {
                    arko_ai::Image::Base64 { data, .. } => data,
                    arko_ai::Image::Url(_) => unreachable!(),
                },
                "image/png"
            ),
            Some(name)
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Data that isn't standard base64 is kept exactly as given.
    #[test]
    fn data_that_is_not_base64_is_kept_as_text() {
        let dir = run_dir();
        let files = ImageFiles::in_run_dir(&dir);
        let image = arko_ai::Image::new_base64("not base64!", "image/png");
        let json = files.with(|| serde_json::to_value(&image).unwrap());
        let name = json["Base64"]["stored"].as_str().unwrap();
        assert!(name.ends_with(".b64"), "{name}");
        let fresh = Arc::new(ImageFiles::new(dir.join(IMAGES_DIR), CACHE_BYTES));
        let read: arko_ai::Image = fresh.with(|| serde_json::from_value(json.clone()).unwrap());
        assert_eq!(read, image);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A name read from a file can't reach outside the folder.
    #[test]
    fn names_outside_the_folder_are_refused() {
        let dir = run_dir();
        let files = ImageFiles::in_run_dir(&dir);
        for name in ["../secret.png", "/etc/passwd", "abc.png", ""] {
            let error = files.load(name, "image/png").unwrap_err();
            assert!(error.contains("isn't the name"), "{error}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Images differing only between the sampled slices are told apart.
    #[test]
    fn images_alike_in_the_samples_are_compared_in_full() {
        let dir = run_dir();
        let files = ImageFiles::in_run_dir(&dir);
        let mut bytes = vec![7u8; 30_000];
        let first = arko_ai::Image::from_bytes(&bytes, "image/png");
        bytes[15_001] = 8;
        let second = arko_ai::Image::from_bytes(&bytes, "image/png");
        let names = files.with(|| {
            [&first, &second]
                .map(|image| serde_json::to_value(image).unwrap()["Base64"]["stored"].clone())
        });
        assert_ne!(names[0], names[1]);
        assert_eq!(std::fs::read_dir(dir.join(IMAGES_DIR)).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The cache keeps the most recently used images within its budget.
    #[test]
    fn the_cache_forgets_the_least_recently_used() {
        let dir = run_dir();
        let files = ImageFiles::new(dir.join(IMAGES_DIR), 2_000);
        let big = |fill: u8| "A".repeat(996) + &format!("{fill:04}");
        files.remember(&big(1), "image/png", "one".into());
        files.remember(&big(2), "image/png", "two".into());
        assert!(files.known(&big(1), "image/png").is_some());
        files.remember(&big(3), "image/png", "three".into());
        assert!(files.known(&big(1), "image/png").is_some());
        assert!(files.known(&big(2), "image/png").is_none());
        assert!(files.known(&big(3), "image/png").is_some());
        let _ = std::fs::remove_dir_all(dir);
    }
}
