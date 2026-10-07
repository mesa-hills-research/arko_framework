//! Command-line interface to arko_index: `chunk`, `outline`, `index`/`search`,
//! `index2`/`search2` (lexical, dense or hybrid), `watch`, `watch2` (Linux) and
//! `bench-scan` (Unix). Run
//! `cargo run --release -p arko_index --example arko_index_cli -- --help`; the
//! crate README describes the commands, the file selection flags and the protocol.
//! Embedding with Voyage goes through arko_ai (`support/voyage.rs`) and needs
//! `VOYAGE_API_KEY`.
mod support;

use arko_index::{
    bench_v1::*,
    dense::{DenseBuild, DenseConfig, EmbedProgress, EmbedStats},
    hybrid::{Fusion, Mode, SearchRequest},
    search,
    vector::{Filter, Quantization},
};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
fn write_json<T: serde::Serialize>(value: &T) -> Result<(), Error> {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
/// File selection flags, accepted by every command that walks a tree.
struct Walk {
    rules: FileRules,
    report: Option<PathBuf>,
}
const RULE_FLAGS: &str =
    "[--respect-ignore] [--exclude PATTERN]... [--include PATTERN]... [--report FILE]";
const EMBED_FLAGS: &str = "[--embed MODEL] [--dimensions N] [--embed-config FILE] [--quantize f32|int8] [--max-embed-tokens N|none] [--no-embed]";
const SEARCH_FLAGS: &str = "[--mode lexical|dense|hybrid] [--limit N] [--candidates N] [--fusion rrf|linear] [--rrf-k K] [--lexical-weight W] [--dense-weight W] [--path-prefix P] [--language L] [--text]";

/// Every command's flags; a command accepts the groups it passes to `parse_flags`.
struct Flags {
    walk: Walk,
    embed: Option<String>,
    dimensions: Option<usize>,
    embed_config: Option<PathBuf>,
    quantize: Option<Quantization>,
    /// `Some(None)`: `--max-embed-tokens none`.
    max_embed_tokens: Option<Option<u64>>,
    no_embed: bool,
    mode: Option<Mode>,
    limit: Option<usize>,
    candidates: Option<usize>,
    fusion: Option<String>,
    rrf_k: Option<f64>,
    lexical_weight: Option<f64>,
    dense_weight: Option<f64>,
    filter: Filter,
    text: bool,
}
#[derive(Clone, Copy, Default)]
struct Accepts {
    walk: bool,
    embed: bool,
    search: bool,
}
fn number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, Error> {
    value
        .parse()
        .map_err(|_| format!("{flag} needs a number, not {value:?}").into())
}
/// Splits the flags from the positional arguments. `--` ends the flags.
fn parse_flags(args: &[String], accepts: Accepts) -> Result<(Flags, Vec<String>), Error> {
    let mut f = Flags {
        walk: Walk {
            rules: FileRules::default(),
            report: None,
        },
        embed: None,
        dimensions: None,
        embed_config: None,
        quantize: None,
        max_embed_tokens: None,
        no_embed: false,
        mode: None,
        limit: None,
        candidates: None,
        fusion: None,
        rrf_k: None,
        lexical_weight: None,
        dense_weight: None,
        filter: Filter::default(),
        text: false,
    };
    let options: Vec<&str> = [
        (accepts.walk, RULE_FLAGS),
        (accepts.embed, EMBED_FLAGS),
        (accepts.search, SEARCH_FLAGS),
    ]
    .into_iter()
    .filter_map(|(on, list)| on.then_some(list))
    .collect();
    let options = options.join(" ");
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--" {
            positional.extend(it.cloned());
            break;
        }
        if !a.starts_with("--") {
            positional.push(a.clone());
            continue;
        }
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) => (f, Some(v.to_owned())),
            None => (a.as_str(), None),
        };
        let unknown = || -> Error { format!("unknown option {a}; options: {options}").into() };
        let accepted = match flag {
            "--respect-ignore" | "--exclude" | "--include" | "--report" => accepts.walk,
            "--embed" | "--dimensions" | "--embed-config" | "--quantize" | "--max-embed-tokens"
            | "--no-embed" => accepts.embed,
            "--mode" | "--limit" | "--candidates" | "--fusion" | "--rrf-k" | "--lexical-weight"
            | "--dense-weight" | "--path-prefix" | "--language" | "--text" => accepts.search,
            _ => false,
        };
        if !accepted {
            return Err(unknown());
        }
        // Switches take no value.
        if inline.is_none() {
            match flag {
                "--respect-ignore" => {
                    f.walk.rules.respect_ignore = true;
                    continue;
                }
                "--no-embed" => {
                    f.no_embed = true;
                    continue;
                }
                "--text" => {
                    f.text = true;
                    continue;
                }
                _ => {}
            }
        }
        let value = match inline {
            Some(v) => v,
            None => it.next().cloned().ok_or(format!("{flag} needs a value"))?,
        };
        match flag {
            "--exclude" => f.walk.rules.exclude.push(value),
            "--include" => f.walk.rules.include.push(value),
            "--report" => f.walk.report = Some(PathBuf::from(value)),
            "--embed" => f.embed = Some(value),
            "--dimensions" => f.dimensions = Some(number(flag, &value)?),
            "--embed-config" => f.embed_config = Some(PathBuf::from(value)),
            "--quantize" => {
                f.quantize = Some(match value.as_str() {
                    "f32" => Quantization::F32,
                    "int8" => Quantization::Int8,
                    _ => return Err(format!("--quantize takes f32 or int8, not {value:?}").into()),
                })
            }
            "--max-embed-tokens" => {
                f.max_embed_tokens = Some(if value == "none" {
                    None
                } else {
                    Some(number(flag, &value)?)
                })
            }
            "--mode" => f.mode = Some(value.parse()?),
            "--limit" => f.limit = Some(number(flag, &value)?),
            "--candidates" => f.candidates = Some(number(flag, &value)?),
            "--fusion" => f.fusion = Some(value),
            "--rrf-k" => f.rrf_k = Some(number(flag, &value)?),
            "--lexical-weight" => f.lexical_weight = Some(number(flag, &value)?),
            "--dense-weight" => f.dense_weight = Some(number(flag, &value)?),
            "--path-prefix" => f.filter = f.filter.clone().with_path_prefix(value),
            "--language" => f.filter = f.filter.clone().with_language(value),
            _ => return Err(unknown()),
        }
    }
    f.walk.rules.validate()?;
    Ok((f, positional))
}
/// Prints the report summary and the entries worth listing to stderr, and writes the
/// full report to `--report FILE` if given.
fn report(walk: &Walk, report: &Report) -> Result<(), Error> {
    eprintln!("arko_index: {}", report.summary());
    for line in report.notable(20) {
        eprintln!("arko_index: {line}");
    }
    write_report_file(walk, report)
}
/// Writes the full report to `--report FILE`, if given.
fn write_report_file(walk: &Walk, report: &Report) -> Result<(), Error> {
    if let Some(path) = &walk.report {
        let mut bytes = serde_json::to_vec_pretty(report)?;
        bytes.push(b'\n');
        std::fs::write(path, bytes)?;
    }
    Ok(())
}
fn rules_changed_note() {
    eprintln!(
        "arko_index: the file selection rules differ from the previous run's (or the state predates recorded rules); the first delta applies the new rules"
    );
}

/// The embedding settings of an `index2` run: `--no-embed` (lexical, and forget the
/// saved settings), `--embed-config FILE`, `--embed MODEL`, else the index's saved
/// `dense.json`; then `--dimensions`, `--quantize` and `--max-embed-tokens`.
fn dense_config(state: &Path, f: &Flags) -> Result<Option<DenseConfig>, Error> {
    if f.no_embed {
        if f.embed.is_some() || f.embed_config.is_some() {
            return Err("--no-embed contradicts --embed and --embed-config".into());
        }
        if DenseConfig::remove(state)? {
            eprintln!("arko_index: embeddings turned off for this index (removed dense.json)");
        }
        return Ok(None);
    }
    let saved = DenseConfig::load(state)?;
    let mut config = if let Some(path) = &f.embed_config {
        DenseConfig::read(path)?
    } else if let Some(spec) = &f.embed {
        let embedder = support::descriptor_for(spec, f.dimensions)?;
        // A new model keeps the saved quantization and cap unless they are given.
        match saved {
            Some(saved) => DenseConfig::new(embedder)
                .with_quantization(saved.quantization)
                .with_max_tokens_per_run(saved.max_tokens_per_run),
            None => DenseConfig::new(embedder),
        }
    } else if let Some(mut saved) = saved {
        if let Some(dims) = f.dimensions {
            let spec = format!("{}:{}", saved.embedder.provider, saved.embedder.model);
            saved.embedder = support::descriptor_for(&spec, Some(dims))?;
        }
        saved
    } else {
        if f.dimensions.is_some() || f.quantize.is_some() || f.max_embed_tokens.is_some() {
            return Err("this index has no saved embedding model: choose one with --embed".into());
        }
        return Ok(None);
    };
    if let Some(q) = f.quantize {
        config = config.with_quantization(q);
    }
    if let Some(cap) = f.max_embed_tokens {
        config = config.with_max_tokens_per_run(cap);
    }
    Ok(Some(config))
}

/// The stderr lines of an embedding run.
fn print_embed_stats(s: &EmbedStats) {
    eprintln!(
        "arko_index: embeddings: {}; {} chunks, {} distinct texts: {} from the cache, {} embedded now in {} requests, {} tokens",
        s.embedder.display(),
        s.docs,
        s.texts,
        s.cached,
        s.new,
        s.requests,
        s.tokens
    );
    if s.cap_reached {
        eprintln!(
            "arko_index: stopped at the token cap ({}): {} texts ({} counted tokens) not embedded; their chunks are found by lexical search only. Rerun with a higher --max-embed-tokens (or none) to finish; embedded texts come from the cache",
            s.max_tokens.unwrap_or(0),
            s.not_embedded - s.too_long,
            s.not_embedded_tokens
        );
    }
    if s.too_long > 0 {
        eprintln!(
            "arko_index: {} texts are over the model's per-input token limit and were not embedded",
            s.too_long
        );
    }
    eprintln!(
        "arko_index: {} of {} chunks have vectors; the cache holds {} vectors for this model",
        s.embedded_docs, s.docs, s.cache_records
    );
}

/// `search2`: the request from the flags; the default mode is hybrid when the index
/// has embeddings, else lexical.
fn search2(state: &Path, query: &str, f: &Flags) -> Result<(), Error> {
    let info = arko_index::index2::dense_info(state)?;
    let mode = f.mode.unwrap_or(if info.is_some() {
        Mode::Hybrid
    } else {
        Mode::Lexical
    });
    let fusion = match f.fusion.as_deref().unwrap_or("rrf") {
        "rrf" => {
            let Fusion::Rrf {
                k,
                lexical_weight,
                dense_weight,
            } = Fusion::default()
            else {
                unreachable!()
            };
            Fusion::Rrf {
                k: f.rrf_k.unwrap_or(k),
                lexical_weight: f.lexical_weight.unwrap_or(lexical_weight),
                dense_weight: f.dense_weight.unwrap_or(dense_weight),
            }
        }
        "linear" => {
            if f.rrf_k.is_some() || f.dense_weight.is_some() {
                return Err("--fusion linear takes --lexical-weight only".into());
            }
            Fusion::Linear {
                lexical_weight: f.lexical_weight.unwrap_or(0.5),
            }
        }
        other => return Err(format!("--fusion takes rrf or linear, not {other:?}").into()),
    };
    let mut request = SearchRequest::new(query)
        .with_mode(mode)
        .with_fusion(fusion)
        .with_filter(f.filter.clone());
    if let Some(limit) = f.limit {
        request = request.with_limit(limit);
    }
    if let Some(candidates) = f.candidates {
        request = request.with_candidates(candidates);
    }
    let embedder = match (&info, mode) {
        (Some(info), Mode::Dense | Mode::Hybrid) => Some(
            support::embedder_for(&info.embedder)
                .map_err(|e| format!("{e} (or search with --mode lexical)"))?,
        ),
        _ => None,
    };
    let results = arko_index::hybrid::search(state, &request, embedder.as_deref())?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    for (i, hit) in results.hits.iter().enumerate() {
        if f.text {
            let mut ranks = Vec::new();
            if let Some(r) = hit.lexical_rank {
                ranks.push(format!("lexical #{r}"));
            }
            if let Some(r) = hit.dense_rank {
                ranks.push(format!("dense #{r}"));
            }
            writeln!(
                out,
                "{:>2}. {}  [{} {:.4}; {}]",
                i + 1,
                hit.header,
                results.mode.as_str(),
                hit.score,
                ranks.join(", ")
            )?;
        } else {
            serde_json::to_writer(&mut out, hit)?;
            writeln!(out)?;
        }
    }
    out.flush()?;
    if mode != Mode::Lexical
        && let Some(info) = &results.dense
    {
        eprintln!(
            "arko_index: query embedded with {}: {} tokens",
            info.embedder.display(),
            results.query_tokens
        );
        if !info.complete() {
            eprintln!(
                "arko_index: {} of {} chunks have vectors (the build stopped at its token cap); the others are found by lexical search only",
                info.embedded, info.docs
            );
        }
    }
    Ok(())
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = Options::default();
    let usage = format!(
        "usage: arko_index_cli chunk ROOT | index ROOT STATE_DIR | search STATE_DIR QUERY | watch ROOT STATE_DIR | outline ROOT\n\
         \x20      (also index2, search2, watch2, bench-scan)\n\
         commands that walk a tree accept: {RULE_FLAGS}\n\
         index2 ROOT STATE_DIR also accepts: {EMBED_FLAGS}\n\
         search2 STATE_DIR QUERY... accepts: {SEARCH_FLAGS}"
    );
    let command = args.first().map(String::as_str);
    let walks = matches!(
        command,
        Some("chunk" | "outline" | "index" | "index2" | "watch" | "watch2" | "bench-scan")
    );
    let accepts = Accepts {
        walk: walks,
        embed: command == Some("index2"),
        search: command == Some("search2"),
    };
    let (flags, args) = if accepts.walk || accepts.search {
        let (flags, mut rest) = parse_flags(&args[1..], accepts)?;
        rest.insert(0, args[0].clone());
        (flags, rest)
    } else {
        (parse_flags(&[], accepts)?.0, args)
    };
    let walk = &flags.walk;
    match args.first().map(String::as_str) {
        Some("chunk") | Some("outline") if args.len() == 2 => {
            let root = Path::new(&args[1]);
            // Scaling prototype: workers chunk and render each file; the main thread
            // writes the rendered bytes in sequential file order through one buffer.
            let outline = args[0] == "outline";
            let mut out = io::BufWriter::with_capacity(1 << 20, io::stdout().lock());
            let render = |chunks: Vec<Chunk>| -> Result<Vec<u8>, Error> {
                let mut buf = Vec::new();
                for c in chunks {
                    if outline {
                        writeln!(
                            buf,
                            "{}:{}-{}\t{}\t{}",
                            c.path, c.start_line, c.end_line, c.kind, c.symbol_path
                        )?;
                    } else {
                        serde_json::to_writer(&mut buf, &c)?;
                        buf.push(b'\n');
                    }
                }
                Ok(buf)
            };
            if root.is_dir() {
                let r = chunk_dir_map_with(
                    root,
                    &options,
                    &walk.rules,
                    arko_index::threads(),
                    render,
                    |b| {
                        out.write_all(&b)?;
                        Ok(())
                    },
                )?;
                out.flush()?;
                // Quiet when only the built-in exclusions (.git, CACHEDIR.TAG, state
                // directories) or the user's own rules removed anything; the full
                // report still goes to --report FILE.
                if r.worth_printing() {
                    report(walk, &r)?;
                } else {
                    write_report_file(walk, &r)?;
                }
            } else {
                let chunks = chunk_file(root, &options)?;
                if chunks.is_empty() {
                    // Say why a single file produced nothing (empty, binary or not UTF-8).
                    let bytes = std::fs::read(root)?;
                    if bytes.contains(&0) {
                        eprintln!("arko_index: skipped (binary): {}", root.display());
                    } else if std::str::from_utf8(&bytes).is_err() {
                        eprintln!("arko_index: skipped (not-utf8): {}", root.display());
                    }
                }
                out.write_all(&render(chunks)?)?;
            }
            out.flush()?;
        }
        Some("index") if args.len() == 3 => {
            let (n, r) = search::index_with(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &options,
                &walk.rules,
            )?;
            eprintln!("indexed {n} chunks");
            report(walk, &r)?;
        }
        Some("index2") if args.len() == 3 => {
            let budget_mb = std::env::var("ARKO_INDEX_INDEX_MEM_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256);
            let state = Path::new(&args[2]);
            std::fs::create_dir_all(state)?;
            let config = dense_config(state, &flags)?;
            let embedder = config
                .as_ref()
                .map(|c| {
                    support::embedder_for(&c.embedder).map_err(|e| {
                        format!(
                            "{e} (this index embeds with {}; --no-embed builds it lexically and forgets the model)",
                            c.embedder.display()
                        )
                    })
                })
                .transpose()?;
            let progress = |p: &EmbedProgress| {
                eprintln!(
                    "arko_index: embedded {}/{} texts, {} tokens, {} requests",
                    p.texts, p.total, p.tokens, p.requests
                );
            };
            let mut settings = arko_index::index2::BuildOptions::default()
                .with_threads(arko_index::threads())
                .with_mem_budget(budget_mb << 20)
                .with_rules(walk.rules.clone());
            if let (Some(config), Some(embedder)) = (&config, &embedder) {
                eprintln!(
                    "arko_index: embedding with {}{}",
                    config.embedder.display(),
                    match config.max_tokens_per_run {
                        Some(cap) => format!(", at most {cap} tokens"),
                        None => String::new(),
                    }
                );
                settings = settings.with_dense(
                    DenseBuild::from_config(embedder.as_ref(), config).with_progress(&progress),
                );
            }
            let s = arko_index::index2::build(Path::new(&args[1]), state, &options, &settings)?;
            eprintln!(
                "indexed {} chunks, {} terms, {} runs",
                s.docs, s.terms, s.runs
            );
            if let Some(dense) = &s.dense {
                print_embed_stats(dense);
            }
            report(walk, &s.report)?;
        }
        Some("search2") if args.len() >= 3 => {
            search2(Path::new(&args[1]), &args[2..].join(" "), &flags)?;
        }
        #[cfg(target_os = "linux")]
        Some("watch2") if args.len() == 3 => {
            let running = Arc::new(AtomicBool::new(true));
            let flag = running.clone();
            ctrlc::set_handler(move || flag.store(false, Ordering::SeqCst))?;
            let mut watcher = arko_index::watch2::Watcher2::open_with(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &options,
                &walk.rules,
            )?;
            if watcher.rules_changed() {
                rules_changed_note();
            }
            eprintln!(
                "{}",
                serde_json::json!({"files": watcher.file_count(), "stats": watcher.stats})
            );
            report(walk, watcher.report())?;
            let mut shown = watcher.report().clone();
            // Each delta is streamed from the record store as one complete, flushed
            // line before it is acknowledged (the same bytes as `write_json`).
            let mut out = io::BufWriter::with_capacity(1 << 20, io::stdout().lock());
            while running.load(Ordering::SeqCst) {
                if let Some(seq) = watcher.next_change_to(Duration::from_millis(250), &mut out)? {
                    out.flush()?;
                    watcher.ack(seq)?;
                    eprintln!(
                        "{}",
                        serde_json::json!({"seq": seq, "stats": watcher.stats})
                    );
                }
                // Skipped entries changed: one summary line (the list is in report.json).
                if watcher.report().counts != shown.counts {
                    shown = watcher.report().clone();
                    eprintln!("arko_index: {}", shown.summary());
                }
            }
        }
        #[cfg(unix)]
        Some("bench-scan") if args.len() == 2 => {
            arko_index::bench_scan_with(Path::new(&args[1]), arko_index::threads(), &walk.rules)?;
        }
        Some("search") if args.len() >= 3 => {
            for c in search::search(Path::new(&args[1]), &args[2..].join(" "))? {
                write_json(&c)?;
            }
        }
        Some("watch") if args.len() == 3 => {
            let running = Arc::new(AtomicBool::new(true));
            let flag = running.clone();
            ctrlc::set_handler(move || flag.store(false, Ordering::SeqCst))?;
            let root = Path::new(&args[1]);
            let state = Path::new(&args[2]);
            let mut watcher = open_watch_with(root, state, &options, &walk.rules)?;
            if watcher.rules_changed() {
                rules_changed_note();
            }
            report(walk, watcher.report())?;
            let mut shown = watcher.report().clone();
            while running.load(Ordering::SeqCst) {
                if let Some(d) = watcher.next_change(Duration::from_millis(250))? {
                    write_json(&d)?;
                    // The CLI acknowledges after a complete, flushed line. Library consumers can
                    // instead ack after processing and deduplicate by durable sequence.
                    watcher.ack(d.sequence)?;
                }
                // Skipped entries changed: one summary line (the list is in report.json).
                if watcher.report().counts != shown.counts {
                    shown = watcher.report().clone();
                    eprintln!("arko_index: {}", shown.summary());
                }
            }
            watcher.flush()?;
        }
        #[cfg(not(target_os = "linux"))]
        Some("watch2") => return Err("watch2 needs Linux (inotify); use watch".into()),
        #[cfg(not(unix))]
        Some("bench-scan") => return Err("bench-scan needs a Unix system".into()),
        Some("--help") | Some("-h") => eprintln!("{usage}"),
        _ => return Err(usage.into()),
    }
    Ok(())
}
/// glibc raises its mmap threshold (up to 32 MB) after large blocks are freed, so the
/// per-thread arenas then keep up to ~64 MB each of freed file text and parse trees
/// resident. A fixed 1 MB threshold returns large blocks to the OS on free: measured
/// on Linux, peak RSS of chunk at 12 threads 659 -> 512 MB and index2 1293 -> 869 MB
/// for about 1 percent more wall time. Allocation results are unaffected.
fn tune_allocator() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_MMAP_THRESHOLD_").is_none() {
        unsafe {
            libc::mallopt(libc::M_MMAP_THRESHOLD, 1 << 20);
        }
    }
}
fn main() {
    tune_allocator();
    if let Err(e) = run() {
        eprintln!("arko_index: {e}");
        std::process::exit(1);
    }
}
