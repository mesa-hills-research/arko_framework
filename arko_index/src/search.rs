//! Durable offline BM25, with identifier and camel-case components in addition to whole words.
use crate::{
    bench_v1::{Chunk, Error, Options, chunk_str},
    files::{FileRules, Report, read_dir, write_report},
    watch::atomic_json,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{File, OpenOptions},
    io::BufReader,
    path::Path,
};

#[derive(Serialize, Deserialize)]
struct Document {
    chunk: Chunk,
    terms: HashMap<String, usize>,
    len: usize,
}
#[derive(Serialize, Deserialize)]
struct Index {
    version: u32,
    docs: Vec<Document>,
    df: HashMap<String, usize>,
    average: f64,
}
pub(crate) fn tokens(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    for word in text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
    {
        let lower = word.to_lowercase();
        result.push(lower.clone());
        let mut starts = vec![0];
        let chars: Vec<_> = word.char_indices().collect();
        for i in 1..chars.len() {
            let c = chars[i].1;
            let prev = chars[i - 1].1;
            if c == '_'
                || prev == '_'
                || (c.is_uppercase()
                    && (prev.is_lowercase()
                        || (prev.is_uppercase()
                            && chars.get(i + 1).is_some_and(|(_, c)| c.is_lowercase()))))
            {
                starts.push(chars[i].0);
            }
        }
        starts.push(word.len());
        for pair in starts.windows(2) {
            let part = word[pair[0]..pair[1]].trim_matches('_').to_lowercase();
            if !part.is_empty() && part != lower {
                result.push(part);
            }
        }
    }
    result
}
pub fn index(root: &Path, state_dir: &Path, options: &Options) -> Result<usize, Error> {
    index_with(root, state_dir, options, &FileRules::default()).map(|(n, _)| n)
}
/// [`index`] with explicit file selection rules. Returns the chunk count and the
/// report, which is also written to `STATE_DIR/report.json` after the index commits.
pub fn index_with(
    root: &Path,
    state_dir: &Path,
    options: &Options,
    rules: &FileRules,
) -> Result<(usize, Report), Error> {
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
    // Independent index lock as in S-21956424. Readers keep seeing the last
    // atomic snapshot, while competing writers fail instead of sharing a temp.
    let _lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(state_dir.join(".index.lock"))?;
    _lock
        .try_lock_exclusive()
        .map_err(|e| format!("index state already in use: {e}"))?;
    // Directory marker is also recognized by chunk_dir and independent watcher instances.
    File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(state_dir.join(".arko-index-state"))?
        .sync_all()?;
    let (files, report) = read_dir(&root, Some(&state_dir), rules)?;
    let mut docs = Vec::new();
    let mut df = HashMap::new();
    let mut total = 0;
    for (path, file) in files {
        for chunk in chunk_str(Path::new(&path), &file.text, options)? {
            let text = format!("{}\n{}", path, &file.text[chunk.start_byte..chunk.end_byte]);
            let mut terms = HashMap::new();
            for t in tokens(&text) {
                *terms.entry(t).or_insert(0) += 1;
            }
            let len: usize = terms.values().sum();
            total += len;
            for term in terms.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            docs.push(Document { chunk, terms, len });
        }
    }
    let count = docs.len();
    let idx = Index {
        version: 1,
        docs,
        df,
        average: if count > 0 {
            total as f64 / count as f64
        } else {
            1.0
        },
    };
    atomic_json(&state_dir, "index.json", &idx)?;
    write_report(&state_dir, &report)?;
    Ok((count, report))
}
pub fn search(state_dir: &Path, query: &str) -> Result<Vec<Chunk>, Error> {
    let idx: Index =
        serde_json::from_reader(BufReader::new(File::open(state_dir.join("index.json"))?))?;
    if idx.version != 1 {
        return Err("unsupported index version".into());
    }
    let query: BTreeSet<_> = tokens(query).into_iter().collect();
    let mut scored: Vec<_> = idx
        .docs
        .iter()
        .filter_map(|d| {
            let mut score = 0.0;
            for t in &query {
                if let Some(&tf) = d.terms.get(t) {
                    let df = *idx.df.get(t).unwrap_or(&0) as f64;
                    let idf = (1.0 + (idx.docs.len() as f64 - df + 0.5) / (df + 0.5)).ln();
                    let norm = 1.2 * (0.25 + 0.75 * d.len as f64 / idx.average.max(1.0));
                    score += idf * tf as f64 * 2.2 / (tf as f64 + norm);
                }
            }
            if score > 0.0 {
                Some((score, &d.chunk))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by(|(a, ca), (b, cb)| {
        b.total_cmp(a)
            .then_with(|| ca.path.cmp(&cb.path))
            .then(ca.start_byte.cmp(&cb.start_byte))
    });
    Ok(scored
        .into_iter()
        .take(10)
        .map(|(_, c)| c.clone())
        .collect())
}
/// Convenience human-readable symbol outline, with no additional source copying.
pub fn outline(chunks: &[Chunk]) -> BTreeMap<String, Vec<&Chunk>> {
    let mut result: BTreeMap<String, Vec<&Chunk>> = BTreeMap::new();
    for c in chunks {
        result.entry(c.path.clone()).or_default().push(c);
    }
    result
}
