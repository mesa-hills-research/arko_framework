//! Command-line interface to arko_index: `chunk`, `outline`, `index`/`search`,
//! `index2`/`search2` (lexical, dense or hybrid), `embed` and `status`, `watch`,
//! `watch2` (Linux) and `bench-scan` (Unix). Run
//! `cargo run --release -p arko_index --example arko_index_cli -- --help`; the
//! `docs/arko_index-reference.md` describes the commands, the file selection flags and
//! the protocol.
//! Embedding with Voyage goes through arko_ai (`support/voyage.rs`) and needs
//! `VOYAGE_API_KEY`.
mod support;

use arko_index::{
    bench_v1::*,
    dense::{
        DenseBuild, DenseConfig, EmbedProgress, EmbedState, EmbedStats, EmbedText, StopSignal,
    },
    hybrid::{Fusion, Mode, SearchRequest},
    index2::IndexStatus,
    search,
    vector::{Filter, Quantization, VectorIndex},
};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use support::process::{self, Owner};
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
    /// `--front-matter-description`: Markdown scopes start with title and description.
    front_matter_description: bool,
}
const RULE_FLAGS: &str = "[--respect-ignore] [--exclude PATTERN]... [--include PATTERN]... [--report FILE] [--front-matter-description]";
const EMBED_FLAGS: &str = "[--lexical-scope] [--embed MODEL] [--dimensions N] [--embed-text source|path-source|path-scope-source] [--embed-config FILE] [--quantize f32|int8] [--vector-index flat|graph] [--max-embed-tokens N|none] [--no-embed] [--wait-embeddings] [--owner-pid PID]";
const RUN_FLAGS: &str = "[--max-embed-tokens N|none] [--owner-pid PID]";
const WATCH_FLAGS: &str = "[--index INDEX_STATE_DIR] [--lexical-scope]";
const SEARCH_FLAGS: &str = "[--mode lexical|dense|hybrid] [--limit N] [--candidates N] [--fusion rrf|linear] [--rrf-k K] [--lexical-weight W] [--dense-weight W] [--path-prefix P] [--language L] [--text]";

/// Every command's flags; a command accepts the groups it passes to `parse_flags`.
struct Flags {
    walk: Walk,
    /// `--lexical-scope`: index2 also indexes each chunk's scope for lexical search.
    lexical_scope: bool,
    embed: Option<String>,
    embed_text: Option<EmbedText>,
    dimensions: Option<usize>,
    embed_config: Option<PathBuf>,
    quantize: Option<Quantization>,
    vector_index: Option<VectorIndex>,
    /// `Some(None)`: `--max-embed-tokens none`.
    max_embed_tokens: Option<Option<u64>>,
    no_embed: bool,
    /// `index2 --wait-embeddings`: embed in this process before returning.
    wait_embeddings: bool,
    /// `watch`/`watch2 --index DIR`: keep that index2 state current.
    index: Option<PathBuf>,
    /// `--owner-pid PID`: the process an embedding run belongs to (0: none).
    owner_pid: Option<u32>,
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
    /// `embed`: the run's cap.
    run: bool,
    /// `watch`, `watch2`: `--index`.
    watch: bool,
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
            front_matter_description: false,
        },
        lexical_scope: false,
        embed: None,
        embed_text: None,
        dimensions: None,
        embed_config: None,
        quantize: None,
        vector_index: None,
        max_embed_tokens: None,
        no_embed: false,
        wait_embeddings: false,
        index: None,
        owner_pid: None,
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
        (accepts.run, RUN_FLAGS),
        (accepts.watch, WATCH_FLAGS),
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
            "--respect-ignore"
            | "--exclude"
            | "--include"
            | "--report"
            | "--front-matter-description" => accepts.walk,
            "--embed" | "--dimensions" | "--embed-text" | "--embed-config" | "--quantize"
            | "--vector-index" | "--no-embed" | "--wait-embeddings" => accepts.embed,
            "--max-embed-tokens" | "--owner-pid" => accepts.embed || accepts.run,
            "--lexical-scope" => accepts.embed || accepts.watch,
            "--index" => accepts.watch,
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
                "--front-matter-description" => {
                    f.walk.front_matter_description = true;
                    continue;
                }
                "--no-embed" => {
                    f.no_embed = true;
                    continue;
                }
                "--wait-embeddings" => {
                    f.wait_embeddings = true;
                    continue;
                }
                "--lexical-scope" => {
                    f.lexical_scope = true;
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
            "--embed-text" => f.embed_text = Some(value.parse()?),
            "--embed-config" => f.embed_config = Some(PathBuf::from(value)),
            "--index" => f.index = Some(PathBuf::from(value)),
            "--owner-pid" => f.owner_pid = Some(number(flag, &value)?),
            "--quantize" => {
                f.quantize = Some(match value.as_str() {
                    "f32" => Quantization::F32,
                    "int8" => Quantization::Int8,
                    _ => return Err(format!("--quantize takes f32 or int8, not {value:?}").into()),
                })
            }
            "--vector-index" => {
                f.vector_index = Some(match value.as_str() {
                    "flat" => VectorIndex::Flat,
                    "graph" => VectorIndex::Graph,
                    _ => {
                        return Err(
                            format!("--vector-index takes flat or graph, not {value:?}").into()
                        );
                    }
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
/// SIGINT and SIGTERM: the first stops the returned signal (a watcher stops after its
/// current batch, an embedding run after its current request), a second exits at once
/// (status 130).
fn stop_on_signal() -> Result<Arc<StopSignal>, Error> {
    let stop = Arc::new(StopSignal::new());
    let signal = stop.clone();
    ctrlc::set_handler(move || {
        if !signal.stop("it was stopped by a signal") {
            eprintln!("arko_index: stopped by a second signal");
            std::process::exit(130);
        }
    })?;
    Ok(stop)
}

/// The owner an embedding run in this process gets from `--owner-pid` or
/// `ARKO_INDEX_OWNER_PID`, when one is given (0: none).
fn explicit_owner(f: &Flags) -> Result<Option<u32>, Error> {
    if let Some(pid) = f.owner_pid {
        return Ok(Some(pid));
    }
    match std::env::var("ARKO_INDEX_OWNER_PID") {
        Ok(v) if !v.trim().is_empty() => Ok(Some(number("ARKO_INDEX_OWNER_PID", v.trim())?)),
        _ => Ok(None),
    }
}

/// How long a run whose owner ended may finish its current request before the process
/// leaves it. Everything committed before it is kept.
const OWNER_GRACE: Duration = Duration::from_secs(30);

/// Stops `stop` when the owner process `pid` ends (0: no owner), checking every second,
/// and leaves the process if the run hasn't stopped after `OWNER_GRACE`.
fn stop_with_owner(pid: u32, stop: &Arc<StopSignal>) {
    if pid == 0 {
        return;
    }
    let Some(owner) = Owner::watch(pid) else {
        let reason = format!("the program that started it (process {pid}) was no longer running");
        eprintln!("arko_index: {reason}: not embedding");
        stop.stop(reason);
        return;
    };
    let stop = stop.clone();
    std::thread::spawn(move || {
        while owner.alive() {
            if stop.is_stopped() {
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        let reason = format!(
            "the program that started it (process {}) ended",
            owner.pid()
        );
        eprintln!("arko_index: {reason}: stopping after the current request");
        stop.stop(reason);
        std::thread::sleep(OWNER_GRACE);
        eprintln!(
            "arko_index: the current request is still running after {} s: leaving it; the next run continues from what was saved",
            OWNER_GRACE.as_secs()
        );
        std::process::exit(0);
    });
}
fn rules_changed_note() {
    eprintln!(
        "arko_index: the file selection rules differ from the previous run's (or the state predates recorded rules); the first delta applies the new rules"
    );
}

/// The embedding settings of an `index2` run: `--no-embed` (lexical, and forget the
/// saved settings), or one of `--embed-config FILE` and `--embed MODEL`, else the
/// index's saved `dense.json`; then `--dimensions`, `--embed-text`, `--quantize`,
/// `--vector-index` and `--max-embed-tokens`.
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
    if f.embed.is_some() && f.embed_config.is_some() {
        return Err("--embed and --embed-config both choose the model: give one".into());
    }
    let saved = DenseConfig::load(state)?;
    let mut config = if let Some(path) = &f.embed_config {
        let mut config = DenseConfig::read(path)?;
        if let Some(dims) = f.dimensions {
            // Checked against the model; the file's other model settings stay.
            let spec = format!("{}:{}", config.embedder.provider, config.embedder.model);
            support::descriptor_for(&spec, Some(dims))?;
            config.embedder.dimensions = dims;
        }
        config
    } else if let Some(spec) = &f.embed {
        let embedder = support::descriptor_for(spec, f.dimensions)?;
        // A new model keeps the saved quantization, vector index, cap and embedded text
        // unless they are given.
        match saved {
            Some(saved) => DenseConfig::new(embedder)
                .with_quantization(saved.quantization)
                .with_vector_index(saved.vector_index)
                .with_max_tokens_per_run(saved.max_tokens_per_run)
                .with_embed_text(saved.embed_text),
            None => DenseConfig::new(embedder),
        }
    } else if let Some(mut saved) = saved {
        if let Some(dims) = f.dimensions {
            let spec = format!("{}:{}", saved.embedder.provider, saved.embedder.model);
            saved.embedder = support::descriptor_for(&spec, Some(dims))?;
        }
        saved
    } else {
        if f.dimensions.is_some()
            || f.embed_text.is_some()
            || f.quantize.is_some()
            || f.vector_index.is_some()
            || f.max_embed_tokens.is_some()
        {
            return Err("this index has no saved embedding model: choose one with --embed".into());
        }
        return Ok(None);
    };
    if let Some(q) = f.quantize {
        config = config.with_quantization(q);
    }
    if let Some(index) = f.vector_index {
        config = config.with_vector_index(index);
    }
    if let Some(embed_text) = f.embed_text {
        config = config.with_embed_text(embed_text);
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
    if s.superseded {
        eprintln!(
            "arko_index: a newer build was published, and its embedding run continues from the cache"
        );
    }
    if s.stopped {
        eprintln!(
            "arko_index: stopped; `embed STATE_DIR` (or the next index2) continues, with the texts embedded so far from the cache"
        );
    }
    if s.cap_reached {
        eprintln!(
            "arko_index: stopped at the token cap ({}): {} texts ({} counted tokens) not embedded; their chunks are found by lexical search only. Run `embed STATE_DIR` with a higher --max-embed-tokens (or none) to finish; embedded texts come from the cache",
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
    eprintln!(
        "arko_index: vector store{}: {} vectors added, {} removed",
        if s.store_created { " (new)" } else { "" },
        s.vectors_added,
        s.vectors_removed
    );
}

/// One stderr line on the embedding progress of a run: after each request, and in
/// `status`.
fn progress_line(p: &EmbedProgress) -> String {
    format!(
        "embedded {}/{} texts, {} tokens, {} requests; {} of {} chunks have vectors",
        p.texts, p.total, p.tokens, p.requests, p.embedded_docs, p.docs
    )
}

/// Why the dense coverage is partial, for the notices of `search2` and `status`.
fn coverage_note(state: EmbedState, status: &IndexStatus) -> String {
    match (state, status.embedding) {
        (EmbedState::Embedding, true) => "embedding continues in the background".into(),
        (EmbedState::Embedding, false) => format!(
            "the embedding stopped before it finished{}. `embed STATE_DIR` or the next index2 continues it",
            status
                .stopped
                .as_ref()
                .map(|why| format!(" because {why}"))
                .unwrap_or_default()
        ),
        (EmbedState::CapReached, _) => {
            "the last run stopped at its token cap: `embed --max-embed-tokens N|none STATE_DIR` continues".into()
        }
        (EmbedState::Complete, _) => "the others are over the model's input limit".into(),
    }
}

/// Embeds what the current generation lacks, in this process, printing progress. The
/// first SIGINT or SIGTERM stops it after its current request, and so does the end of
/// the owner process `owner` (0: none).
fn embed_now(state: &Path, max_tokens: Option<Option<u64>>, owner: u32) -> Result<(), Error> {
    let info = arko_index::index2::dense_info(state)?
        .ok_or("this index has no embeddings: build it with them (index2 --embed MODEL)")?;
    let embedder = support::embedder_for(&info.embedder)?;
    let cap = match max_tokens {
        Some(cap) => cap,
        None => DenseConfig::load(state)?.and_then(|c| c.max_tokens_per_run),
    };
    if arko_index::index2::status(state)?.embedding {
        eprintln!("arko_index: waiting for the embedding run in progress to finish or stop");
    }
    let stop = stop_on_signal()?;
    stop_with_owner(owner, &stop);
    let progress = |p: &EmbedProgress| eprintln!("arko_index: {}", progress_line(p));
    let stats = arko_index::index2::embed(
        state,
        &DenseBuild::new(embedder.as_ref())
            .with_max_tokens(cap)
            .with_progress(&progress)
            .with_stop(&stop),
    )?;
    print_embed_stats(&stats);
    Ok(())
}

/// Starts `embed STATE_DIR` as a detached process of this executable, its output in
/// `STATE_DIR/embed.log`, belonging to the owner process `owner` (0: none). Returns its
/// process id.
fn embed_in_background(state: &Path, owner: u32) -> Result<u32, Error> {
    let state = state.canonicalize()?;
    // A new file: a run still writing the previous one keeps its own (where the file
    // can't be removed, it is truncated).
    let log = state.join(EMBED_LOG);
    let _ = std::fs::remove_file(&log);
    let log = std::fs::File::create(log)?;
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command
        .arg("embed")
        .arg("--owner-pid")
        .arg(owner.to_string())
        .arg("--")
        .arg(&state)
        .current_dir(&state)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log);
    process::detach(&mut command);
    Ok(command.spawn()?.id())
}

/// The log of the background embedding run that `index2` starts.
const EMBED_LOG: &str = "embed.log";

/// `status`: one JSON line on stdout, a summary on stderr.
fn print_status(status: &IndexStatus) -> Result<(), Error> {
    write_json(status)?;
    match &status.dense {
        None => eprintln!(
            "arko_index: {}: {} chunks, lexical search only",
            status.generation, status.docs
        ),
        Some(info) => {
            eprintln!(
                "arko_index: {}: {} chunks; dense ({}): {} of {} chunks have vectors{}",
                status.generation,
                status.docs,
                info.embedder.display(),
                info.embedded,
                info.docs,
                if info.complete() {
                    String::new()
                } else {
                    format!(" ({})", coverage_note(info.state, status))
                }
            );
            if let Some(p) = &status.progress {
                eprintln!(
                    "arko_index: {} run: {}",
                    if status.embedding { "this" } else { "the last" },
                    progress_line(p)
                );
            }
        }
    }
    Ok(())
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
        if info.embedded > 0 {
            eprintln!(
                "arko_index: query embedded with {}: {} tokens",
                info.embedder.display(),
                results.query_tokens
            );
        }
        if !info.complete() {
            let status = arko_index::index2::status(state)?;
            eprintln!(
                "arko_index: dense: {} of {} chunks embedded ({}); the others are found by lexical search only",
                info.embedded,
                info.docs,
                coverage_note(info.state, &status)
            );
        }
    }
    Ok(())
}

/// `index2`'s postings memory budget: `ARKO_INDEX_INDEX_MEM_MB`, else 256 MB.
fn index_mem_budget() -> usize {
    std::env::var("ARKO_INDEX_INDEX_MEM_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256)
        << 20
}

/// The watchers' `--index INDEX_STATE_DIR`: that index2 state is rebuilt when the tree
/// is quiet after deltas (with the embedding model saved there, if any), and a thread
/// of this process embeds what each generation lacks.
struct Keeper<'a> {
    root: &'a Path,
    index: Option<&'a Path>,
    flags: &'a Flags,
    options: &'a Options,
    dirty: bool,
    wake: Option<std::sync::mpsc::Sender<()>>,
}
impl Keeper<'_> {
    /// The watcher delivered a delta.
    fn changed(&mut self) {
        self.dirty = self.index.is_some();
    }
    /// A poll found the tree quiet: rebuild the index if it changed. A failed rebuild is
    /// reported, and the next delta tries again.
    fn quiet(&mut self) {
        if let Err(e) = self.rebuild() {
            eprintln!("arko_index: index rebuild: {e}");
        }
    }
    fn rebuild(&mut self) -> Result<(), Error> {
        let Some(index) = self.index else {
            return Ok(());
        };
        if !std::mem::take(&mut self.dirty) {
            return Ok(());
        }
        let config = DenseConfig::load(index)?;
        let embedder = config
            .as_ref()
            .map(|c| support::embedder_for(&c.embedder))
            .transpose()?;
        let mut settings = arko_index::index2::BuildOptions::default()
            .with_mem_budget(index_mem_budget())
            .with_rules(self.flags.walk.rules.clone())
            .with_lexical_scope(self.flags.lexical_scope);
        if let (Some(config), Some(embedder)) = (&config, &embedder) {
            settings = settings.with_dense(
                DenseBuild::from_config(embedder.as_ref(), config).with_embed_later(true),
            );
        }
        let s = arko_index::index2::build(self.root, index, self.options, &settings)?;
        match &s.dense {
            Some(d) => eprintln!(
                "arko_index: index rebuilt: {} chunks; dense: {} have vectors, {} distinct texts to embed",
                s.docs,
                d.embedded_docs,
                d.texts - d.cached
            ),
            None => eprintln!("arko_index: index rebuilt: {} chunks", s.docs),
        }
        if let Some(wake) = &self.wake {
            let _ = wake.send(());
        }
        Ok(())
    }
}

/// Runs a watcher loop with `--index`, when given: the embedding thread runs beside it,
/// and stops after its current request when the watcher stops.
fn keep_index(
    root: &Path,
    flags: &Flags,
    options: &Options,
    stop: &StopSignal,
    watch: impl FnOnce(&mut Keeper<'_>) -> Result<(), Error>,
) -> Result<(), Error> {
    let index = flags.index.as_deref();
    std::thread::scope(|scope| {
        let mut keeper = Keeper {
            root,
            index,
            flags,
            options,
            dirty: false,
            wake: None,
        };
        if let Some(index) = index {
            let (wake, woken) = std::sync::mpsc::channel::<()>();
            // An index that exists resumes its embedding now; a new one is built when
            // the tree is first quiet.
            if arko_index::index2::status(index).is_ok() {
                let _ = wake.send(());
            } else {
                keeper.dirty = true;
            }
            keeper.wake = Some(wake);
            scope.spawn(move || {
                while woken.recv().is_ok() {
                    while woken.try_recv().is_ok() {}
                    while !stop.is_stopped() {
                        match embed_for_watcher(index, stop) {
                            // A newer generation: embed it too.
                            Ok(true) => {}
                            Ok(false) => break,
                            Err(e) => {
                                eprintln!("arko_index: embedding: {e}");
                                break;
                            }
                        }
                    }
                }
            });
        }
        let result = watch(&mut keeper);
        if let Err(e) = &result {
            // The run in progress stops after its current request.
            stop.stop(format!("the watcher stopped on an error: {e}"));
        }
        // Closes the channel: the embedding thread ends once its run stops or ends.
        drop(keeper);
        result
    })
}

/// One embedding run of `--index`, with progress lines on stderr. Whether a newer
/// generation superseded it.
fn embed_for_watcher(index: &Path, stop: &StopSignal) -> Result<bool, Error> {
    let Some(info) = arko_index::index2::dense_info(index)? else {
        return Ok(false);
    };
    let embedder = support::embedder_for(&info.embedder)?;
    let cap = DenseConfig::load(index)?.and_then(|c| c.max_tokens_per_run);
    let progress = |p: &EmbedProgress| eprintln!("arko_index: {}", progress_line(p));
    let s = arko_index::index2::embed(
        index,
        &DenseBuild::new(embedder.as_ref())
            .with_max_tokens(cap)
            .with_progress(&progress)
            .with_stop(stop),
    )?;
    if s.requests > 0 || s.vectors_added > 0 || s.vectors_removed > 0 || s.cap_reached {
        print_embed_stats(&s);
    }
    Ok(s.superseded)
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = format!(
        "usage: arko_index_cli chunk ROOT | index ROOT STATE_DIR | search STATE_DIR QUERY | watch ROOT STATE_DIR | outline ROOT\n\
         \x20      (also index2, search2, embed, status, watch2, bench-scan)\n\
         commands that walk a tree accept: {RULE_FLAGS}\n\
         index2 ROOT STATE_DIR also accepts: {EMBED_FLAGS}\n\
         search2 STATE_DIR QUERY... accepts: {SEARCH_FLAGS}\n\
         embed STATE_DIR accepts: {RUN_FLAGS}; status STATE_DIR\n\
         watch and watch2 also accept: {WATCH_FLAGS}"
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
        run: command == Some("embed"),
        watch: matches!(command, Some("watch" | "watch2")),
    };
    let (flags, args) = if accepts.walk || accepts.search || accepts.run {
        let (flags, mut rest) = parse_flags(&args[1..], accepts)?;
        rest.insert(0, args[0].clone());
        (flags, rest)
    } else {
        (parse_flags(&[], accepts)?.0, args)
    };
    let walk = &flags.walk;
    let options = Options {
        front_matter_description: walk.front_matter_description,
        ..Options::default()
    };
    match args.first().map(String::as_str) {
        Some("chunk") | Some("outline") if args.len() == 2 => {
            let root = Path::new(&args[1]);
            // Scaling prototype: workers chunk and render each file; the main thread
            // writes the rendered bytes in sequential file order through one buffer.
            let outline = args[0] == "outline";
            let mut out = io::BufWriter::with_capacity(1 << 20, io::stdout().lock());
            // A file's metadata region (Markdown front matter) comes before its chunks:
            // in the outline as `path:start-end\tmetadata\ttitle`, in JSON as a line
            // with a `file` key.
            let render = |file: FileChunks| -> Result<Vec<u8>, Error> {
                let mut buf = Vec::new();
                if let Some(m) = &file.metadata {
                    if outline {
                        let span = &m.metadata_range;
                        writeln!(
                            buf,
                            "{}:{}-{}\tmetadata\t{}",
                            m.file,
                            span.start_line,
                            span.end_line,
                            m.title.as_deref().unwrap_or("")
                        )?;
                    } else {
                        serde_json::to_writer(&mut buf, m)?;
                        buf.push(b'\n');
                    }
                }
                for c in file.chunks {
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
                let file = chunk_file_with_metadata(root, &options)?;
                if file.chunks.is_empty() && file.metadata.is_none() {
                    // Say why a single file produced nothing (empty, binary or not UTF-8).
                    let bytes = std::fs::read(root)?;
                    if bytes.contains(&0) {
                        eprintln!("arko_index: skipped (binary): {}", root.display());
                    } else if std::str::from_utf8(&bytes).is_err() {
                        eprintln!("arko_index: skipped (not-utf8): {}", root.display());
                    }
                }
                out.write_all(&render(file)?)?;
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
            let mut settings = arko_index::index2::BuildOptions::default()
                .with_threads(arko_index::threads())
                .with_mem_budget(index_mem_budget())
                .with_rules(walk.rules.clone())
                .with_lexical_scope(flags.lexical_scope);
            if let (Some(config), Some(embedder)) = (&config, &embedder) {
                eprintln!(
                    "arko_index: embedding with {}, text {}{}",
                    config.embedder.display(),
                    config.embed_text.as_str(),
                    match config.max_tokens_per_run {
                        Some(cap) => format!(", at most {cap} tokens per run"),
                        None => String::new(),
                    }
                );
                // The generation is published for lexical search first; the embedding
                // follows, in this process or in the background.
                settings = settings.with_dense(
                    DenseBuild::from_config(embedder.as_ref(), config).with_embed_later(true),
                );
            } else if flags.wait_embeddings {
                return Err("--wait-embeddings: this index has no embedding model (choose one with --embed)".into());
            }
            let s = arko_index::index2::build(Path::new(&args[1]), state, &options, &settings)?;
            eprintln!(
                "indexed {} chunks, {} terms, {} runs",
                s.docs, s.terms, s.runs
            );
            report(walk, &s.report)?;
            if let Some(d) = &s.dense {
                eprintln!(
                    "arko_index: lexical search is ready; dense: {} of {} chunks have vectors, {} of {} distinct texts to embed ({} in the cache)",
                    d.embedded_docs,
                    d.docs,
                    d.texts - d.cached,
                    d.texts,
                    d.cached
                );
                let owner = explicit_owner(&flags)?;
                if flags.wait_embeddings {
                    embed_now(state, None, owner.unwrap_or(0))?;
                } else {
                    // The run belongs to the program that ran this command.
                    let owner = owner.unwrap_or_else(|| process::default_owner().unwrap_or(0));
                    let pid = embed_in_background(state, owner)?;
                    eprintln!(
                        "arko_index: embedding in the background (process {pid}{}); `status {}` shows the progress, {} the run's output",
                        if owner == 0 {
                            String::new()
                        } else {
                            format!(", until process {owner} ends")
                        },
                        state.display(),
                        state.join(EMBED_LOG).display()
                    );
                }
            }
        }
        Some("embed") if args.len() == 2 => embed_now(
            Path::new(&args[1]),
            flags.max_embed_tokens,
            explicit_owner(&flags)?.unwrap_or(0),
        )?,
        Some("status") if args.len() == 2 => {
            print_status(&arko_index::index2::status(Path::new(&args[1]))?)?
        }
        Some("search2") if args.len() >= 3 => {
            search2(Path::new(&args[1]), &args[2..].join(" "), &flags)?;
        }
        #[cfg(target_os = "linux")]
        Some("watch2") if args.len() == 3 => {
            let stop = stop_on_signal()?;
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
            let root = Path::new(&args[1]);
            keep_index(root, &flags, &options, &stop, |keeper| {
                while !stop.is_stopped() {
                    if let Some(seq) =
                        watcher.next_change_to(Duration::from_millis(250), &mut out)?
                    {
                        out.flush()?;
                        watcher.ack(seq)?;
                        eprintln!(
                            "{}",
                            serde_json::json!({"seq": seq, "stats": watcher.stats})
                        );
                        keeper.changed();
                    } else {
                        keeper.quiet();
                    }
                    // Skipped entries changed: one summary line (the list is in report.json).
                    if watcher.report().counts != shown.counts {
                        shown = watcher.report().clone();
                        eprintln!("arko_index: {}", shown.summary());
                    }
                }
                Ok(())
            })?;
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
            let stop = stop_on_signal()?;
            let root = Path::new(&args[1]);
            let state = Path::new(&args[2]);
            let mut watcher = open_watch_with(root, state, &options, &walk.rules)?;
            if watcher.rules_changed() {
                rules_changed_note();
            }
            report(walk, watcher.report())?;
            let mut shown = watcher.report().clone();
            keep_index(root, &flags, &options, &stop, |keeper| {
                while !stop.is_stopped() {
                    if let Some(d) = watcher.next_change(Duration::from_millis(250))? {
                        write_json(&d)?;
                        // The CLI acknowledges after a complete, flushed line. Library
                        // consumers can instead ack after processing and deduplicate by
                        // durable sequence.
                        watcher.ack(d.sequence)?;
                        keeper.changed();
                    } else {
                        keeper.quiet();
                    }
                    // Skipped entries changed: one summary line (the list is in report.json).
                    if watcher.report().counts != shown.counts {
                        shown = watcher.report().clone();
                        eprintln!("arko_index: {}", shown.summary());
                    }
                }
                Ok(())
            })?;
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
