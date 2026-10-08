//! One build's embedding run. While the tree is chunked, texts missing from the cache
//! are spooled to a file in the new generation (so memory stays bounded); then they
//! are embedded in requests within the embedder's limits, stopping before a request
//! would take the run over its token cap, and every request's vectors are appended to
//! the cache at once.
use super::{EmbedProgress, Embedder, InputKind, cache::Cache};
use crate::bench_v1::Error;
use std::{
    collections::VecDeque,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

/// Texts to embed, in first-seen order: (32-byte hash, u32 LE length, UTF-8 text)*.
pub(crate) struct Spool {
    path: PathBuf,
    out: BufWriter<File>,
    count: u64,
}
impl Spool {
    pub(crate) fn create(path: &Path) -> Result<Self, Error> {
        Ok(Self {
            path: path.to_path_buf(),
            out: BufWriter::with_capacity(1 << 20, File::create(path)?),
            count: 0,
        })
    }
    pub(crate) fn push(&mut self, hash: &[u8; 32], text: &str) -> Result<(), Error> {
        let len = u32::try_from(text.len()).map_err(|_| "text too long to embed")?;
        self.out.write_all(hash)?;
        self.out.write_all(&len.to_le_bytes())?;
        self.out.write_all(text.as_bytes())?;
        self.count += 1;
        Ok(())
    }
    pub(crate) fn finish(mut self) -> Result<(PathBuf, u64), Error> {
        self.out.flush()?;
        Ok((self.path, self.count))
    }
}

struct SpoolReader(BufReader<File>);
impl SpoolReader {
    fn next(&mut self) -> Result<Option<([u8; 32], String)>, Error> {
        let mut hash = [0u8; 32];
        match self.0.read_exact(&mut hash) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let mut len = [0u8; 4];
        self.0.read_exact(&mut len)?;
        let mut text = vec![0u8; u32::from_le_bytes(len) as usize];
        self.0.read_exact(&mut text)?;
        Ok(Some((hash, String::from_utf8(text)?)))
    }
}

#[derive(Default)]
pub(crate) struct RunTotals {
    pub new: u64,
    pub requests: u64,
    pub tokens: u64,
    pub cap_reached: bool,
    pub not_embedded: u64,
    pub not_embedded_tokens: u64,
    pub too_long: u64,
}

/// Embeds the spooled texts into the cache.
pub(crate) fn embed_spool(
    path: &Path,
    total: u64,
    cache: &mut Cache,
    embedder: &dyn Embedder,
    cap: Option<u64>,
    progress: Option<&dyn Fn(&EmbedProgress)>,
) -> Result<RunTotals, Error> {
    let limits = embedder.limits();
    let mut reader = SpoolReader(BufReader::with_capacity(1 << 20, File::open(path)?));
    // (hash, text, counted tokens), in spool order.
    let mut queue: VecDeque<([u8; 32], String, u64)> = VecDeque::new();
    let mut exhausted = false;
    let mut totals = RunTotals::default();
    let count = |texts: &[([u8; 32], String)]| -> Result<Vec<u64>, Error> {
        let refs: Vec<&str> = texts.iter().map(|(_, t)| sendable(t)).collect();
        let counts = embedder.count_tokens(&refs)?;
        if counts.len() != refs.len() {
            return Err(format!(
                "the embedder counted {} of {} texts",
                counts.len(),
                refs.len()
            )
            .into());
        }
        Ok(counts)
    };
    loop {
        if !exhausted && queue.len() < limits.max_items {
            let mut fresh = Vec::new();
            while queue.len() + fresh.len() < limits.max_items {
                match reader.next()? {
                    Some(item) => fresh.push(item),
                    None => {
                        exhausted = true;
                        break;
                    }
                }
            }
            if !fresh.is_empty() {
                let counts = count(&fresh)?;
                for ((h, t), n) in fresh.into_iter().zip(counts) {
                    queue.push_back((h, t, n));
                }
            }
        }
        // A text over a per-input or per-request limit can never be sent.
        if let Some(&(_, _, n)) = queue.front()
            && (n > limits.max_input_tokens || n > limits.max_request_tokens)
        {
            queue.pop_front();
            totals.too_long += 1;
            totals.not_embedded += 1;
            totals.not_embedded_tokens += n;
            continue;
        }
        if queue.is_empty() {
            if exhausted {
                break;
            }
            continue;
        }
        // The longest prefix within the request limits...
        let (mut n, mut sum) = (0usize, 0u64);
        for (_, _, t) in &queue {
            if n == limits.max_items
                || *t > limits.max_input_tokens
                || sum + t > limits.max_request_tokens
            {
                break;
            }
            n += 1;
            sum += t;
        }
        // ...and within the cap: tokens billed so far plus this request's count.
        if let Some(cap) = cap {
            let room = cap.saturating_sub(totals.tokens);
            while n > 0 && sum > room {
                n -= 1;
                sum -= queue[n].2;
            }
            if n == 0 {
                totals.cap_reached = true;
                break;
            }
        }
        let batch: Vec<_> = queue.drain(..n).collect();
        let texts: Vec<&str> = batch.iter().map(|(_, t, _)| sendable(t)).collect();
        let embedded = embedder.embed(&texts, InputKind::Document)?;
        if embedded.vectors.len() != batch.len() {
            return Err(format!(
                "the embedder returned {} vectors for {} texts",
                embedded.vectors.len(),
                batch.len()
            )
            .into());
        }
        if embedded
            .vectors
            .iter()
            .any(|v| v.iter().any(|x| !x.is_finite()) || v.iter().all(|x| *x == 0.0))
        {
            return Err("the embedder returned a zero or non-finite vector".into());
        }
        let records: Vec<([u8; 32], Vec<f32>)> = batch
            .iter()
            .map(|(h, _, _)| *h)
            .zip(embedded.vectors)
            .collect();
        cache.append(&records)?;
        totals.tokens += if embedded.tokens > 0 {
            embedded.tokens
        } else {
            sum
        };
        totals.requests += 1;
        totals.new += batch.len() as u64;
        if let Some(progress) = progress {
            progress(&EmbedProgress {
                texts: totals.new,
                total,
                tokens: totals.tokens,
                requests: totals.requests,
            });
        }
    }
    if totals.cap_reached {
        // What is left, counted (no request is sent), so the report says what
        // finishing would take.
        for (_, _, n) in queue.drain(..) {
            totals.not_embedded += 1;
            totals.not_embedded_tokens += n;
        }
        loop {
            let mut rest = Vec::new();
            while rest.len() < limits.max_items {
                match reader.next()? {
                    Some(item) => rest.push(item),
                    None => break,
                }
            }
            if rest.is_empty() {
                break;
            }
            for n in count(&rest)? {
                totals.not_embedded += 1;
                totals.not_embedded_tokens += n;
            }
        }
    }
    Ok(totals)
}

/// What is sent for a text: providers such as Voyage reject empty inputs, and a
/// chunk's text can be empty or blank (an empty file with `--embed-text source`),
/// so a blank text is embedded as a fixed placeholder. Its cache key stays the
/// text's own hash.
fn sendable(text: &str) -> &str {
    if text.trim().is_empty() {
        EMPTY_TEXT
    } else {
        text
    }
}

/// The placeholder [`sendable`] sends for a blank text.
const EMPTY_TEXT: &str = "(empty)";

#[cfg(test)]
mod sendable_tests {
    use super::*;

    #[test]
    fn blank_texts_are_sent_as_a_placeholder() {
        assert_eq!(sendable(""), EMPTY_TEXT);
        assert_eq!(sendable(" \n\t"), EMPTY_TEXT);
        assert_eq!(sendable("fn main() {}"), "fn main() {}");
    }
}
