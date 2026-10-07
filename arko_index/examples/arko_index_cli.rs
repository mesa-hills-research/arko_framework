//! Command-line interface to arko_index: `chunk`, `outline`, `index`/`search`,
//! `index2`/`search2`, `watch`, `watch2` (Linux) and `bench-scan` (Unix). Run
//! `cargo run --release -p arko_index --example arko_index_cli -- --help`; the
//! crate README describes the commands, the file selection flags and the protocol.
use arko_index::{bench_v1::*, search};
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
/// Splits the file selection flags from the positional arguments. `--` ends the flags.
fn walk_args(args: &[String]) -> Result<(Walk, Vec<String>), Error> {
    let mut walk = Walk {
        rules: FileRules::default(),
        report: None,
    };
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
        if flag == "--respect-ignore" && inline.is_none() {
            walk.rules.respect_ignore = true;
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => it.next().cloned().ok_or(format!("{flag} needs a value"))?,
        };
        match flag {
            "--exclude" => walk.rules.exclude.push(value),
            "--include" => walk.rules.include.push(value),
            "--report" => walk.report = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {a}; options: {RULE_FLAGS}").into()),
        }
    }
    walk.rules.validate()?;
    Ok((walk, positional))
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
fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = Options::default();
    let usage = format!(
        "usage: arko_index_cli chunk ROOT | index ROOT STATE_DIR | search STATE_DIR QUERY | watch ROOT STATE_DIR | outline ROOT\n\
         \x20      (also index2, search2, watch2, bench-scan)\n\
         commands that walk a tree accept: {RULE_FLAGS}"
    );
    let walks = matches!(
        args.first().map(String::as_str),
        Some("chunk" | "outline" | "index" | "index2" | "watch" | "watch2" | "bench-scan")
    );
    let (walk, args) = if walks {
        let (walk, mut rest) = walk_args(&args[1..])?;
        rest.insert(0, args[0].clone());
        (walk, rest)
    } else {
        let walk = Walk {
            rules: FileRules::default(),
            report: None,
        };
        (walk, args)
    };
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
                    report(&walk, &r)?;
                } else {
                    write_report_file(&walk, &r)?;
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
            report(&walk, &r)?;
        }
        Some("index2") if args.len() == 3 => {
            let budget_mb = std::env::var("ARKO_INDEX_INDEX_MEM_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256);
            let s = arko_index::index2::index_with(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &options,
                arko_index::threads(),
                budget_mb << 20,
                &walk.rules,
            )?;
            eprintln!(
                "indexed {} chunks, {} terms, {} runs",
                s.docs, s.terms, s.runs
            );
            report(&walk, &s.report)?;
        }
        Some("search2") if args.len() >= 3 => {
            let mut out = io::BufWriter::new(io::stdout().lock());
            for c in arko_index::index2::search(Path::new(&args[1]), &args[2..].join(" "))? {
                serde_json::to_writer(&mut out, &c)?;
                writeln!(out)?;
            }
            out.flush()?;
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
            report(&walk, watcher.report())?;
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
            report(&walk, watcher.report())?;
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
