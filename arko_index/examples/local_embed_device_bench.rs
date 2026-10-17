//! Throughput of a local embedding model on a device and precision, over a fixed
//! corpus of rendered chunks (`--features local-embed`, with `local-embed-gpu` or
//! `local-embed-cuda` for GPUs).
//!
//! `corpus` chunks source trees, renders each chunk's embedded text the way an index
//! build does, and writes an even sample of each tree's distinct texts as JSON lines:
//!
//! ```sh
//! cargo run --release -p arko_index --features local-embed --example local_embed_device_bench -- \
//!     corpus --out corpus.jsonl [--per-root N] ROOT...
//! ```
//!
//! `run` embeds the corpus on one device in one precision, at each `--batch-tokens`
//! budget: a first pass (kernel compilation and tuning included, reported apart), then
//! timed passes in requests packed as a build packs them. It prints and appends to
//! `--out` one JSON line per budget: texts and tokens per second (median, lowest and
//! highest of the timed passes), the forward passes and their padding, the process's
//! peak memory and, with `nvidia-smi`, the GPU memory in use.
//!
//! ```sh
//! cargo run --release -p arko_index --features local-embed-cuda --example local_embed_device_bench -- \
//!     run --corpus corpus.jsonl --device cuda --precision f16 --batch-tokens 2048,4096,8192 \
//!     [--model NAME] [--repeats 3] [--max-seconds 120] [--limit N] [--sample N] [--long N] \
//!     [--only-sample] [--save-sample FILE] [--reference FILE] [--out results.jsonl] [--label TEXT]
//! ```
//!
//! The sample (`--sample N`, every k-th text, and `--long N` texts of corpus chunks
//! joined to near the model's token limit) is what precisions are compared on:
//! `--save-sample FILE` writes its vectors, `--reference FILE` compares them with a
//! saved run's (cosine similarity per text). `--only-sample` embeds just the sample, for
//! a slow reference such as the CPU in `f32`. `--limit N` runs an even subset of the
//! corpus.
mod support;

use arko_index::{
    bench_v1::{Error, Options, chunk_dir},
    dense::{
        Embedder, InputKind,
        local::{self, Device, LocalEmbedder, Precision},
        render,
    },
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    io::{BufRead, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Instant,
};

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("corpus") => corpus(args.collect()),
        Some("run") => run(args.collect()),
        _ => Err("usage: local_embed_device_bench corpus|run ... (see the file's docs)".into()),
    }
}

/// `--flag value` pairs and the other arguments, in order.
fn flags(args: Vec<String>) -> Result<(HashMap<String, String>, Vec<String>), Error> {
    let (mut map, mut rest) = (HashMap::new(), Vec::new());
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == "--only-sample" {
            map.insert(a, String::new());
        } else if a.starts_with("--") {
            let v = it.next().ok_or(format!("{a} needs a value"))?;
            map.insert(a, v);
        } else {
            rest.push(a);
        }
    }
    Ok((map, rest))
}

fn corpus(args: Vec<String>) -> Result<(), Error> {
    let (f, roots) = flags(args)?;
    let out = f.get("--out").ok_or("corpus needs --out FILE")?;
    let per_root: usize = f.get("--per-root").map_or(Ok(1000), |v| v.parse())?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    for root in &roots {
        let root = Path::new(root);
        let name = root
            .file_name()
            .map_or("root".into(), |n| n.to_string_lossy().into_owned());
        let chunks = chunk_dir(root, &Options::default())?;
        let mut files: HashMap<String, String> = HashMap::new();
        let mut seen = BTreeSet::new();
        let mut texts = Vec::new();
        for c in &chunks {
            if !files.contains_key(&c.path) {
                let bytes = std::fs::read(root.join(&c.path))?;
                files.insert(c.path.clone(), String::from_utf8_lossy(&bytes).into_owned());
            }
            let text = render(
                c,
                files[&c.path].get(c.start_byte..c.end_byte).unwrap_or(""),
            );
            if seen.insert(text.clone()) {
                texts.push(text);
            }
        }
        let picked = even(&texts, per_root);
        eprintln!(
            "{name}: {} chunks, {} distinct texts, {} written",
            chunks.len(),
            texts.len(),
            picked.len()
        );
        for t in picked {
            serde_json::to_writer(&mut file, &json!({"root": name, "text": t}))?;
            file.write_all(b"\n")?;
        }
    }
    file.flush()?;
    Ok(())
}

/// An even sample of `n` items (every k-th), or all of them.
fn even<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if n >= items.len() {
        return items.to_vec();
    }
    let step = items.len() as f64 / n as f64;
    (0..n)
        .map(|i| items[(i as f64 * step) as usize].clone())
        .collect()
}

/// Indices of an even sample of `n` of `len` items.
fn even_indices(len: usize, n: usize) -> Vec<usize> {
    even(&(0..len).collect::<Vec<_>>(), n)
}

fn list<T: std::str::FromStr>(v: &str) -> Result<Vec<T>, Error>
where
    T::Err: std::fmt::Display,
{
    v.split(',')
        .map(|s| {
            s.trim()
                .parse::<T>()
                .map_err(|e| format!("{s}: {e}").into())
        })
        .collect()
}

fn run(args: Vec<String>) -> Result<(), Error> {
    let (f, rest) = flags(args)?;
    if let Some(a) = rest.first() {
        return Err(format!("unexpected argument {a}").into());
    }
    let corpus_path = f.get("--corpus").ok_or("run needs --corpus FILE")?;
    let model = match f.get("--model") {
        Some(name) => local::find(name).ok_or(format!("unknown local model {name}"))?,
        None => local::DEFAULT_MODEL,
    };
    let device: Device = f.get("--device").map_or(Ok(Device::Cpu), |v| v.parse())?;
    let precision: Precision = match f.get("--precision") {
        Some(p) => p.parse()?,
        None => device.default_precision(),
    };
    let budgets: Vec<usize> = match f.get("--batch-tokens") {
        Some(v) => list(v)?,
        None => vec![0],
    };
    let repeats: usize = f.get("--repeats").map_or(Ok(3), |v| v.parse())?;
    let max_seconds: f64 = f.get("--max-seconds").map_or(Ok(120.0), |v| v.parse())?;
    let sample_n: usize = f.get("--sample").map_or(Ok(256), |v| v.parse())?;
    let long_n: usize = f.get("--long").map_or(Ok(0), |v| v.parse())?;
    let only_sample = f.contains_key("--only-sample");
    let label = f.get("--label").cloned().unwrap_or_default();

    let mut texts = Vec::new();
    let mut roots = Vec::new();
    for line in std::fs::read_to_string(corpus_path)?.lines() {
        let v: Value = serde_json::from_str(line)?;
        texts.push(
            v["text"]
                .as_str()
                .ok_or("a corpus line has no text")?
                .to_owned(),
        );
        roots.push(v["root"].as_str().unwrap_or("").to_owned());
    }
    if let Some(limit) = f.get("--limit") {
        texts = even(&texts, limit.parse()?);
    }
    let sample_ids = even_indices(texts.len(), sample_n);

    let started = Instant::now();
    let dir = support::local::model_dir(model)?;
    let found = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let embedder = LocalEmbedder::load_on(model, &dir, precision, device)?;
    let load_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "{} in {precision} on {} loaded in {load_seconds:.1} s (model found in {found:.1} s), {}",
        model.repo,
        embedder.device(),
        memory()
    );

    // The long texts: corpus texts joined until the next would pass the token limit.
    let mut long = Vec::new();
    let limit = model.max_tokens as u64 - 16;
    let mut k = 0;
    while long.len() < long_n && k < texts.len() {
        let mut joined = String::new();
        while k < texts.len() {
            let next = if joined.is_empty() {
                texts[k].clone()
            } else {
                format!("{joined}\n\n{}", texts[k])
            };
            if embedder.count_tokens(&[&next])?[0] >= limit && !joined.is_empty() {
                break;
            }
            joined = next;
            k += 1;
        }
        long.push(joined);
    }
    let sample: Vec<String> = sample_ids
        .iter()
        .map(|&i| texts[i].clone())
        .chain(long.iter().cloned())
        .collect();
    let sample_refs: Vec<&str> = sample.iter().map(String::as_str).collect();
    let long_tokens =
        embedder.count_tokens(&long.iter().map(String::as_str).collect::<Vec<_>>())?;
    if !long.is_empty() {
        eprintln!("long texts: {long_tokens:?} tokens");
    }

    let corpus: Vec<&str> = if only_sample {
        sample_refs.clone()
    } else {
        texts.iter().map(String::as_str).collect()
    };
    let started = Instant::now();
    let counts = embedder.count_tokens(&corpus)?;
    let tokenize_seconds = started.elapsed().as_secs_f64();
    let total_tokens: u64 = counts.iter().sum();
    let mut lens: Vec<u64> = counts.clone();
    lens.sort_unstable();
    let pct = |q: f64| lens[((lens.len() - 1) as f64 * q).round() as usize];
    eprintln!(
        "{} texts, {total_tokens} tokens (median {}, 90th percentile {}, longest {}), tokenizing {tokenize_seconds:.2} s",
        corpus.len(),
        pct(0.5),
        pct(0.9),
        pct(1.0)
    );

    let reference: Option<Vec<Vec<f32>>> = match f.get("--reference") {
        Some(path) => Some(serde_json::from_str(&std::fs::read_to_string(path)?)?),
        None => None,
    };
    let device_name = embedder.device().to_owned();
    let mut embedder = embedder;
    let mem_before = gpu_memory_now();
    for budget in budgets {
        if budget > 0 {
            embedder = embedder.with_batch_tokens(budget);
        }
        let watch = GpuWatch::start();
        let before = embedder.batch_stats();
        let started = Instant::now();
        let first = embed_all(&embedder, &corpus, &counts)?;
        let first_seconds = started.elapsed().as_secs_f64();
        let after = embedder.batch_stats();
        let mut seconds = Vec::new();
        let mut vectors = first;
        let timing = Instant::now();
        while seconds.len() < repeats
            && (seconds.is_empty() || timing.elapsed().as_secs_f64() < max_seconds)
        {
            let started = Instant::now();
            vectors = embed_all(&embedder, &corpus, &counts)?;
            seconds.push(started.elapsed().as_secs_f64());
        }
        // `--repeats 0`: the first pass is the measurement (the CPU compiles nothing).
        if seconds.is_empty() {
            seconds.push(first_seconds);
        }
        let (gpu_process_mib, gpu_device_mib) = watch.map(GpuWatch::stop).unwrap_or_default();
        // The sample's vectors: from the corpus pass, or one more call for the long
        // texts (not part of the corpus).
        let sample_vectors: Vec<Vec<f32>> = if only_sample {
            vectors.clone()
        } else {
            let mut v: Vec<Vec<f32>> = sample_ids.iter().map(|&i| vectors[i].clone()).collect();
            for t in &long {
                v.push(embedder.embed(&[t], InputKind::Document)?.vectors.remove(0));
            }
            v
        };
        let passes = after.passes - before.passes;
        let pass_texts = after.texts - before.texts;
        let padded = after.padded_tokens - before.padded_tokens;
        let tokens = after.tokens - before.tokens;
        let mut rates: Vec<f64> = seconds.iter().map(|s| corpus.len() as f64 / s).collect();
        rates.sort_by(f64::total_cmp);
        let median = |v: &[f64]| v[v.len() / 2];
        let tok_rates: Vec<f64> = rates
            .iter()
            .map(|r| r * total_tokens as f64 / corpus.len() as f64)
            .collect();
        let cos = reference
            .as_ref()
            .map(|r| cosine_stats(r, &sample_vectors, sample_ids.len()));
        let line = json!({
            "label": label,
            "model": model.repo,
            "device": device.as_str(),
            "device_name": device_name,
            "precision": precision.as_str(),
            "batch_tokens": embedder.batch_tokens(),
            "texts": corpus.len(),
            "tokens": total_tokens,
            "only_sample": only_sample,
            "load_seconds": load_seconds,
            "tokenize_seconds": tokenize_seconds,
            "first_pass_seconds": first_seconds,
            "pass_seconds": seconds,
            "texts_per_second": {"median": median(&rates), "min": rates[0], "max": rates[rates.len() - 1]},
            "tokens_per_second": {"median": median(&tok_rates), "min": tok_rates[0], "max": tok_rates[tok_rates.len() - 1]},
            "forward_passes": passes,
            "texts_per_pass": pass_texts as f64 / passes.max(1) as f64,
            "padded_tokens_per_pass": padded as f64 / passes.max(1) as f64,
            "padding_overhead": padded as f64 / tokens.max(1) as f64 - 1.0,
            "peak_rss_mib": peak_rss_mib(),
            "gpu_memory_before_mib": mem_before,
            "gpu_process_peak_mib": gpu_process_mib,
            "gpu_device_peak_mib": gpu_device_mib,
            "cosine_vs_reference": cos,
            "long_text_tokens": long_tokens,
        });
        println!("{line}");
        eprintln!(
            "{precision} on {device_name}, {} batch tokens, {:.1} texts/s, {:.0} tokens/s (median of {}, {:.0} to {:.0}), first pass {first_seconds:.1} s, {:.1} texts and {:.0} padded tokens per pass ({:.1}% padding), GPU peak {} MiB, {}",
            embedder.batch_tokens(),
            median(&rates),
            median(&tok_rates),
            seconds.len(),
            tok_rates[0],
            tok_rates[tok_rates.len() - 1],
            pass_texts as f64 / passes.max(1) as f64,
            padded as f64 / passes.max(1) as f64,
            100.0 * (padded as f64 / tokens.max(1) as f64 - 1.0),
            gpu_process_mib
                .or(gpu_device_mib)
                .map_or("?".into(), |m| m.to_string()),
            memory()
        );
        if let Some(c) = &cos {
            eprintln!("cosine against the reference: {c}");
        }
        if let Some(out) = f.get("--out") {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out)?;
            writeln!(file, "{line}")?;
        }
        if let Some(path) = f.get("--save-sample") {
            std::fs::write(path, serde_json::to_vec(&sample_vectors)?)?;
        }
    }
    Ok(())
}

/// Every request of a build over `texts`, packed within the embedder's limits.
fn embed_all(e: &LocalEmbedder, texts: &[&str], counts: &[u64]) -> Result<Vec<Vec<f32>>, Error> {
    let limits = e.limits();
    let (mut i, mut vectors) = (0, Vec::with_capacity(texts.len()));
    while i < texts.len() {
        let (mut j, mut sum) = (i, 0);
        while j < texts.len()
            && j - i < limits.max_items
            && (j == i || sum + counts[j] <= limits.max_request_tokens)
        {
            sum += counts[j];
            j += 1;
        }
        vectors.extend(e.embed(&texts[i..j], InputKind::Document)?.vectors);
        i = j;
    }
    Ok(vectors)
}

/// Cosine similarity of each sample text's vectors (unit length): the corpus sample
/// (`real` texts) and the long texts after it, each lowest, 1st percentile, median, mean.
fn cosine_stats(reference: &[Vec<f32>], got: &[Vec<f32>], real: usize) -> Value {
    let sims: Vec<f64> = reference
        .iter()
        .zip(got)
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x * y) as f64).sum())
        .collect();
    let stats = |s: &[f64]| -> Value {
        if s.is_empty() {
            return Value::Null;
        }
        let mut s = s.to_vec();
        s.sort_by(f64::total_cmp);
        json!({
            "n": s.len(),
            "min": s[0],
            "p1": s[((s.len() - 1) as f64 * 0.01).round() as usize],
            "median": s[s.len() / 2],
            "mean": s.iter().sum::<f64>() / s.len() as f64,
        })
    };
    let real = real.min(sims.len());
    json!({
        "compared": sims.len().min(reference.len()),
        "sample": stats(&sims[..real]),
        "long": stats(&sims[real..]),
        "same_count": reference.len() == got.len(),
    })
}

fn memory() -> String {
    format!("peak resident memory {} MiB", peak_rss_mib().unwrap_or(0))
}

/// The process's peak resident memory (Linux).
fn peak_rss_mib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().strip_suffix(" kB"))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
}

/// The memory in use on the first NVIDIA GPU, in MiB (none without `nvidia-smi`).
fn gpu_memory_now() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// `nvidia-smi` polled every 100 ms while passes run: the most memory this process held
/// on the GPU, and the most in use on the first GPU.
struct GpuWatch {
    children: Vec<Child>,
    peaks: Arc<Mutex<(Option<u64>, Option<u64>)>>,
    readers: Vec<std::thread::JoinHandle<()>>,
}

impl GpuWatch {
    fn start() -> Option<GpuWatch> {
        let pid = std::process::id().to_string();
        let peaks = Arc::new(Mutex::new((None, None)));
        let mut children = Vec::new();
        let mut readers = Vec::new();
        for (query, process) in [
            ("--query-compute-apps=pid,used_memory", true),
            ("--query-gpu=memory.used", false),
        ] {
            let mut child = Command::new("nvidia-smi")
                .args([query, "--format=csv,noheader,nounits", "-lms", "100"])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let stdout = child.stdout.take()?;
            let peaks = peaks.clone();
            let pid = pid.clone();
            readers.push(std::thread::spawn(move || {
                for line in std::io::BufReader::new(stdout)
                    .lines()
                    .map_while(Result::ok)
                {
                    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
                    let value = match (process, fields.as_slice()) {
                        (true, [p, mib]) if *p == pid => mib.parse::<u64>().ok(),
                        (false, [mib, ..]) => mib.parse::<u64>().ok(),
                        _ => None,
                    };
                    if let Some(v) = value {
                        let mut p = peaks.lock().unwrap();
                        let slot = if process { &mut p.0 } else { &mut p.1 };
                        *slot = Some(slot.map_or(v, |old: u64| old.max(v)));
                    }
                }
            }));
            children.push(child);
        }
        Some(GpuWatch {
            children,
            peaks,
            readers,
        })
    }

    fn stop(mut self) -> (Option<u64>, Option<u64>) {
        for c in &mut self.children {
            let _ = c.kill();
            let _ = c.wait();
        }
        for r in self.readers.drain(..) {
            let _ = r.join();
        }
        *self.peaks.lock().unwrap()
    }
}
