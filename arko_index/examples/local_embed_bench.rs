//! Throughput of a local embedding model on a source tree (`--features local-embed`).
//!
//! Chunks the tree, renders each chunk's embedded text the way an index build does,
//! and embeds the distinct texts at each thread count, in requests packed as a build
//! packs them, printing texts and tokens per second and the process's memory. With
//! two precisions it also compares their vectors: the cosine similarity of each
//! text's two vectors. With `--index STATE` it then builds an index2 with the model
//! (in the first precision, at the last thread count) and prints the size of the
//! state: the lexical index, the vectors and the embedding cache.
//!
//! ```sh
//! cargo run --release -p arko_index --features local-embed --example local_embed_bench -- \
//!     ROOT [--model NAME] [--precision int8,f32] [--threads 1,6] [--limit N] \
//!     [--batch-tokens N] [--index STATE]
//! ```
//!
//! `--limit N` embeds an even sample of N texts (every k-th) instead of all of them.
mod support;

use arko_index::{
    bench_v1::{Error, Options, chunk_dir},
    dense::{
        DenseBuild, Embedder, InputKind,
        local::{self, Precision},
        render,
    },
    index2::{BuildOptions, build},
};
use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    time::Instant,
};

struct Args {
    root: PathBuf,
    model: &'static local::LocalModel,
    precisions: Vec<Precision>,
    threads: Vec<usize>,
    limit: Option<usize>,
    batch_tokens: Option<usize>,
    index: Option<PathBuf>,
}

fn parse_args() -> Result<Args, Error> {
    let mut root = None;
    let mut a = Args {
        root: PathBuf::new(),
        model: local::DEFAULT_MODEL,
        precisions: vec![Precision::default()],
        threads: vec![arko_index::threads()],
        limit: None,
        batch_tokens: None,
        index: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => {
                let name = value()?;
                a.model = local::find(&name).ok_or(format!("unknown local model {name}"))?;
            }
            "--precision" => {
                a.precisions = value()?
                    .split(',')
                    .map(|p| p.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--threads" => {
                a.threads = value()?
                    .split(',')
                    .map(|n| n.trim().parse())
                    .collect::<Result<_, _>>()?
            }
            "--limit" => a.limit = Some(value()?.parse()?),
            "--batch-tokens" => a.batch_tokens = Some(value()?.parse()?),
            "--index" => a.index = Some(value()?.into()),
            _ if !arg.starts_with("--") && root.is_none() => root = Some(PathBuf::from(arg)),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    a.root = root.ok_or("usage: local_embed_bench ROOT [--model NAME] [--precision int8,f32] [--threads 1,6] [--limit N] [--batch-tokens N] [--index STATE]")?;
    Ok(a)
}

/// The distinct rendered texts of the tree's chunks, in chunk order.
fn texts(root: &Path) -> Result<(usize, Vec<String>), Error> {
    let chunks = chunk_dir(root, &Options::default())?;
    let mut files: HashMap<String, String> = HashMap::new();
    let mut seen = BTreeSet::new();
    let mut texts = Vec::new();
    for c in &chunks {
        if !files.contains_key(&c.path) {
            let bytes = std::fs::read(root.join(&c.path))?;
            files.insert(c.path.clone(), String::from_utf8_lossy(&bytes).into_owned());
        }
        let file = &files[&c.path];
        let text = render(c, file.get(c.start_byte..c.end_byte).unwrap_or(""));
        if seen.insert(text.clone()) {
            texts.push(text);
        }
    }
    Ok((chunks.len(), texts))
}

/// Total bytes of the files under `dir`.
fn size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| match e.file_type() {
                    Ok(t) if t.is_dir() => size(&e.path()),
                    Ok(_) => e.metadata().map_or(0, |m| m.len()),
                    Err(_) => 0,
                })
                .sum()
        })
        .unwrap_or(0)
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

/// The process's resident memory now and at its peak (Linux), for messages.
fn memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().strip_suffix(" kB"))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or("unknown".to_owned(), |kb| mb(kb * 1024))
    };
    format!("resident {}, peak {}", field("VmRSS:"), field("VmHWM:"))
}

/// Every text's vector in requests packed as a build packs them; the tokens and
/// requests.
fn embed_all(
    embedder: &local::LocalEmbedder,
    refs: &[&str],
    counts: &[u64],
) -> Result<(Vec<Vec<f32>>, usize), Error> {
    let limits = embedder.limits();
    let (mut i, mut requests, mut vectors) = (0, 0, Vec::with_capacity(refs.len()));
    while i < refs.len() {
        let (mut j, mut sum) = (i, 0);
        while j < refs.len()
            && j - i < limits.max_items
            && (j == i || sum + counts[j] <= limits.max_request_tokens)
        {
            sum += counts[j];
            j += 1;
        }
        vectors.extend(embedder.embed(&refs[i..j], InputKind::Document)?.vectors);
        requests += 1;
        i = j;
    }
    Ok((vectors, requests))
}

fn main() -> Result<(), Error> {
    let args = parse_args()?;
    let cpu = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    println!(
        "cpu {}, {} CPUs available, load average {}",
        cpu.lines()
            .find_map(|l| l.strip_prefix("model name"))
            .map_or("unknown", |l| l.trim_start_matches([' ', '\t', ':'])),
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .split(' ')
            .take(3)
            .collect::<Vec<_>>()
            .join(" ")
    );
    let started = Instant::now();
    let (chunks, mut texts) = texts(&args.root)?;
    println!(
        "{}: {chunks} chunks, {} distinct texts (chunked and rendered in {:.1} s)",
        args.root.display(),
        texts.len(),
        started.elapsed().as_secs_f64()
    );
    if let Some(limit) = args.limit.filter(|&l| l < texts.len()) {
        let step = texts.len() as f64 / limit as f64;
        texts = (0..limit)
            .map(|i| texts[(i as f64 * step) as usize].clone())
            .collect();
        println!("sample: {} texts", texts.len());
    }

    let started = Instant::now();
    let dir = support::local::model_dir(args.model)?;
    println!(
        "model {} at {} found in {:.1} s",
        args.model.repo,
        &args.model.revision[..12],
        started.elapsed().as_secs_f64()
    );

    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let mut results: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut first = None;
    for &precision in &args.precisions {
        let started = Instant::now();
        let mut embedder = local::LocalEmbedder::load_with_precision(args.model, &dir, precision)?;
        println!(
            "{precision} verified and loaded in {:.2} s, {}",
            started.elapsed().as_secs_f64(),
            memory()
        );
        if let Some(n) = args.batch_tokens {
            embedder = embedder.with_batch_tokens(n);
        }
        let mut vectors = Vec::new();
        for &threads in &args.threads {
            embedder = embedder.with_threads(threads)?;
            let started = Instant::now();
            let counts = embedder.count_tokens(&refs)?;
            let counted = started.elapsed().as_secs_f64();
            let total: u64 = counts.iter().sum();
            let started = Instant::now();
            let (got, requests) = embed_all(&embedder, &refs, &counts)?;
            let secs = started.elapsed().as_secs_f64();
            vectors = got;
            println!(
                "{precision} {threads:>3} threads, {} texts, {total} tokens ({:.0} per text) in {requests} requests, {secs:.1} s, {:.2} texts/s, {:.0} tokens/s (tokenizing alone {counted:.2} s), {}",
                refs.len(),
                total as f64 / refs.len() as f64,
                refs.len() as f64 / secs,
                total as f64 / secs,
                memory()
            );
        }
        let started = Instant::now();
        let q = embedder.embed(&["exponential backoff retry"], InputKind::Query)?;
        println!(
            "{precision} one query: {} tokens in {:.1} ms",
            q.tokens,
            started.elapsed().as_secs_f64() * 1e3
        );
        results.push(vectors);
        // Kept for the index build. Otherwise dropped, so the next precision's memory
        // is its own.
        if first.is_none() && args.index.is_some() {
            first = Some(embedder);
        }
    }
    if let [a, b, ..] = &results[..] {
        let mut sims: Vec<f64> = a
            .iter()
            .zip(b)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x * y) as f64).sum())
            .collect();
        sims.sort_by(f64::total_cmp);
        let at = |q: f64| sims[((sims.len() - 1) as f64 * q).round() as usize];
        println!(
            "cosine similarity of {} and {} vectors over {} texts, min {:.5}, 1st percentile {:.5}, median {:.5}, mean {:.5}",
            args.precisions[0],
            args.precisions[1],
            sims.len(),
            sims[0],
            at(0.01),
            at(0.5),
            sims.iter().sum::<f64>() / sims.len() as f64
        );
    }
    if let (Some(state), Some(embedder)) = (&args.index, first) {
        let _ = std::fs::remove_dir_all(state);
        let started = Instant::now();
        let s = build(
            &args.root,
            state,
            &Options::default(),
            &BuildOptions::default()
                .with_threads(arko_index::threads())
                .with_dense(DenseBuild::new(&embedder)),
        )?;
        let d = s.dense.expect("built with embeddings");
        println!(
            "index2 build: {} chunks, {} texts embedded, {} tokens, {:.1} s",
            s.docs,
            d.new,
            d.tokens,
            started.elapsed().as_secs_f64()
        );
        let cache = size(&state.join("embed-cache"));
        // The vector store: `vectors-<digest>/` beside the generations.
        let dense: u64 = std::fs::read_dir(state)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("vectors-"))
            .map(|e| size(&e.path()))
            .sum();
        let total = size(state);
        println!(
            "state {} has {} in all, vectors {}, embedding cache {}, the rest (lexical index, metadata) {}",
            state.display(),
            mb(total),
            mb(dense),
            mb(cache),
            mb(total - dense - cache)
        );
    }
    Ok(())
}
