//! Live check of dense and hybrid retrieval with Voyage. **Spends Voyage tokens** and
//! needs `VOYAGE_API_KEY`.
//!
//! Builds an index2 with embeddings of a small corpus (by default the arko snapshot
//! the retrieval evaluation was written against), then runs the evaluation questions
//! (`breadth-depth-arko.json`, `behaviour-arko.json`) in lexical, dense and hybrid
//! modes and prints each question's rank, the scores, and the tokens used.
//!
//! ```sh
//! cargo run --release -p arko_index --example dense_live_check -- --dry-run   # counts tokens only
//! VOYAGE_API_KEY=... cargo run --release -p arko_index --example dense_live_check
//! ```
//!
//! Flags: `--corpus DIR`, `--eval-dir DIR` (holding the two JSON files), `--state DIR`
//! (kept between runs: a rerun embeds nothing new and spends only query tokens),
//! `--model MODEL` (voyage-code-4; `hash` runs the whole check offline with the
//! non-semantic hash embedder, to test the harness: its lexical column is the real
//! lexical result), `--dimensions N` (the model's default), `--max-embed-tokens N`
//! (2,500,000), `--fusion rrf|linear`, `--rrf-k K`, `--lexical-weight W`,
//! `--dense-weight W`, `--candidates N` (hybrid tuning: a rerun on the same state costs
//! only the queries' tokens), `--f32` (f32 vector codes instead of int8), `--dry-run`
//! (build with a cap of 0 tokens in a scratch state and print what embedding would
//! take; needs no key, only the tokenizer download).
//!
//! The default corpus (7.7 MB, 265 files, 7,018 chunks) took 1,735,222 billed tokens
//! without headers (path and source only, 2026-10-04). With the headers, `--dry-run`
//! counted 1,797,537 tokens (2026-10-07; Voyage bills about one token per text less
//! than its tokenizer counts): about $0.22 at voyage-code-4's list price of $0.12 per
//! million tokens, plus about 200 tokens for the 19 queries.
mod support;

use arko_ai::EmbeddingClient;
use arko_index::{
    bench_v1::{Error, Options},
    dense::{DenseBuild, EmbedProgress, Embedder, InputKind},
    hybrid::{Fusion, Mode, SearchHit, SearchRequest},
    index2::{BuildOptions, build},
    vector::Quantization,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use support::voyage::VoyageEmbedder;

/// The embedder of `--model`: Voyage (needs the key), or the offline hash embedder.
fn embedder(args: &Args) -> Result<Box<dyn Embedder>, Error> {
    if args.model == "hash" {
        return support::embedder_for(&support::descriptor_for("hash", args.dimensions)?);
    }
    Ok(Box::new(VoyageEmbedder::from_env(
        &args.model,
        args.dimensions,
        true,
    )?))
}

#[derive(Deserialize, Clone)]
struct Target {
    path: String,
    line: usize,
}
#[derive(Deserialize)]
struct Depth {
    id: String,
    query: String,
    expect: Vec<Target>,
    #[serde(default)]
    distractors: Vec<Target>,
}
#[derive(Deserialize)]
struct Breadth {
    id: String,
    query: String,
    definition: Target,
    callers: Vec<Target>,
}
#[derive(Deserialize)]
struct BreadthDepth {
    depth: Vec<Depth>,
    breadth: Vec<Breadth>,
}
#[derive(Deserialize)]
struct Behaviour {
    id: String,
    query: String,
    expect: Vec<Target>,
}

/// One question: what counts as found, and for breadth the callers.
struct Case {
    kind: &'static str,
    id: String,
    query: String,
    expect: Vec<Target>,
    callers: Vec<Target>,
    distractors: Vec<Target>,
}

/// The 1-based rank of the first hit containing one of the targets' lines.
fn rank(hits: &[SearchHit], targets: &[Target]) -> Option<usize> {
    hits.iter()
        .position(|h| {
            targets.iter().any(|t| {
                h.chunk.path == t.path && h.chunk.start_line <= t.line && t.line <= h.chunk.end_line
            })
        })
        .map(|i| i + 1)
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

struct Args {
    corpus: PathBuf,
    eval_dir: PathBuf,
    state: Option<PathBuf>,
    model: String,
    dimensions: Option<usize>,
    cap: u64,
    fusion: Fusion,
    candidates: usize,
    quantization: Quantization,
    dry_run: bool,
}

fn parse_args() -> Result<Args, Error> {
    let mut a = Args {
        corpus: home().join("Desktop/auto-research-forge/port-check/eval/arko-snapshot"),
        eval_dir: home().join("Desktop/auto-research-forge/eval"),
        state: None,
        model: "voyage-code-4".into(),
        dimensions: None,
        cap: 2_500_000,
        fusion: Fusion::default(),
        candidates: 50,
        quantization: Quantization::default(),
        dry_run: false,
    };
    let (mut method, mut rrf_k, mut lexical_weight, mut dense_weight) =
        ("rrf".to_owned(), None::<f64>, None::<f64>, None::<f64>);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--corpus" => a.corpus = value()?.into(),
            "--eval-dir" => a.eval_dir = value()?.into(),
            "--state" => a.state = Some(value()?.into()),
            "--model" => a.model = value()?,
            "--dimensions" => a.dimensions = Some(value()?.parse()?),
            "--max-embed-tokens" => a.cap = value()?.parse()?,
            "--fusion" => method = value()?,
            "--rrf-k" => rrf_k = Some(value()?.parse()?),
            "--lexical-weight" => lexical_weight = Some(value()?.parse()?),
            "--dense-weight" => dense_weight = Some(value()?.parse()?),
            "--candidates" => a.candidates = value()?.parse()?,
            "--f32" => a.quantization = Quantization::F32,
            "--dry-run" => a.dry_run = true,
            "--help" | "-h" => {
                eprintln!(
                    "usage: dense_live_check [--corpus DIR] [--eval-dir DIR] [--state DIR] [--model MODEL] [--dimensions N] [--max-embed-tokens N] [--fusion rrf|linear] [--rrf-k K] [--lexical-weight W] [--dense-weight W] [--candidates N] [--f32] [--dry-run]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    a.fusion = match method.as_str() {
        "rrf" => Fusion::Rrf {
            k: rrf_k.unwrap_or(60.0),
            lexical_weight: lexical_weight.unwrap_or(1.0),
            dense_weight: dense_weight.unwrap_or(1.0),
        },
        "linear" => Fusion::Linear {
            lexical_weight: lexical_weight.unwrap_or(0.5),
        },
        other => return Err(format!("unknown fusion {other}").into()),
    };
    Ok(a)
}

fn load_cases(dir: &Path) -> Result<Vec<Case>, Error> {
    let bd: BreadthDepth =
        serde_json::from_slice(&std::fs::read(dir.join("breadth-depth-arko.json"))?)?;
    let behaviour: Vec<Behaviour> =
        serde_json::from_slice(&std::fs::read(dir.join("behaviour-arko.json"))?)?;
    let mut cases = Vec::new();
    for d in bd.depth {
        cases.push(Case {
            kind: "depth",
            id: d.id,
            query: d.query,
            expect: d.expect,
            callers: Vec::new(),
            distractors: d.distractors,
        });
    }
    for b in bd.breadth {
        cases.push(Case {
            kind: "breadth",
            id: b.id,
            query: b.query,
            expect: vec![b.definition],
            callers: b.callers,
            distractors: Vec::new(),
        });
    }
    for h in behaviour {
        cases.push(Case {
            kind: "behaviour",
            id: h.id,
            query: h.query,
            expect: h.expect,
            callers: Vec::new(),
            distractors: Vec::new(),
        });
    }
    Ok(cases)
}

fn show(rank: Option<usize>) -> String {
    rank.map_or("-".into(), |r| r.to_string())
}

fn main() -> Result<(), Error> {
    let args = parse_args()?;
    let cases = load_cases(&args.eval_dir)?;
    let dims_name = args.dimensions.map_or("default".into(), |d| d.to_string());
    let state = args.state.clone().unwrap_or_else(|| {
        home()
            .join("Desktop/arko-framework-work/scratch/dense-live-check")
            .join(format!("{}-{dims_name}", args.model))
    });
    let progress = |p: &EmbedProgress| {
        eprintln!(
            "embedded {}/{} texts, {} tokens, {} requests",
            p.texts, p.total, p.tokens, p.requests
        );
    };

    if args.dry_run {
        // No key needed: the cap of 0 sends nothing; the tokenizer counts.
        let embedder =
            VoyageEmbedder::new(EmbeddingClient::new(), &args.model, args.dimensions, true)?;
        let name = state.file_name().unwrap_or_default().to_string_lossy();
        let scratch = state.with_file_name(format!("{name}-dry-run"));
        let _ = std::fs::remove_dir_all(&scratch);
        let settings =
            BuildOptions::default().with_dense(DenseBuild::new(&embedder).with_max_tokens(Some(0)));
        let s = build(&args.corpus, &scratch, &Options::default(), &settings)?;
        let d = s.dense.expect("built with embeddings");
        println!(
            "dry run: {} chunks, {} distinct texts; {} would be embedded: {} tokens counted ({} cached)",
            d.docs, d.texts, d.not_embedded, d.not_embedded_tokens, d.cached
        );
        std::fs::remove_dir_all(&scratch)?;
        return Ok(());
    }

    let embedder = embedder(&args)?;
    let embedder = embedder.as_ref();
    println!(
        "corpus {}\nstate {}\nmodel {}, cap {} tokens\nfusion {:?}, {} candidates per side",
        args.corpus.display(),
        state.display(),
        embedder.descriptor().display(),
        args.cap,
        args.fusion,
        args.candidates
    );
    let settings = BuildOptions::default().with_dense(
        DenseBuild::new(embedder)
            .with_quantization(args.quantization)
            .with_max_tokens(Some(args.cap))
            .with_progress(&progress),
    );
    let s = build(&args.corpus, &state, &Options::default(), &settings)?;
    let d = s.dense.expect("built with embeddings");
    println!(
        "index: {} chunks, {} distinct texts, {} from the cache, {} embedded now in {} requests: {} tokens{}",
        d.docs,
        d.texts,
        d.cached,
        d.new,
        d.requests,
        d.tokens,
        if d.cap_reached {
            format!(
                "; STOPPED AT THE CAP: {} texts ({} tokens) not embedded",
                d.not_embedded, d.not_embedded_tokens
            )
        } else {
            String::new()
        }
    );

    // Every query embedded once, in one request.
    let queries: Vec<&str> = cases.iter().map(|c| c.query.as_str()).collect();
    let embedded = embedder.embed(&queries, InputKind::Query)?;
    let query_tokens = embedded.tokens;
    println!(
        "queries: {} embedded in one request: {query_tokens} tokens",
        queries.len()
    );

    let modes = [Mode::Lexical, Mode::Dense, Mode::Hybrid];
    // (depth found, definitions found, callers found, behaviour found) per mode.
    let mut totals = [[0usize; 4]; 3];
    let (mut n_depth, mut n_def, mut n_callers, mut n_behaviour) = (0, 0, 0, 0);
    println!(
        "\ncase  kind       lexical  dense  hybrid  (rank of the expected chunk in the top 10; breadth: definition, then callers found)"
    );
    for (case, vector) in cases.iter().zip(&embedded.vectors) {
        let mut cells = Vec::new();
        for (m, mode) in modes.iter().enumerate() {
            let request = SearchRequest::new(&case.query)
                .with_mode(*mode)
                .with_limit(10)
                .with_fusion(args.fusion.clone())
                .with_candidates(args.candidates)
                .with_query_vector(vector.clone());
            let hits = arko_index::hybrid::search(&state, &request, None)?.hits;
            let r = rank(&hits, &case.expect);
            match case.kind {
                "depth" => {
                    totals[m][0] += usize::from(r.is_some());
                    let distractor = rank(&hits, &case.distractors);
                    cells.push(format!("{}/{}", show(r), show(distractor)));
                }
                "breadth" => {
                    totals[m][1] += usize::from(r.is_some());
                    let found = case
                        .callers
                        .iter()
                        .filter(|c| rank(&hits, std::slice::from_ref(*c)).is_some())
                        .count();
                    totals[m][2] += found;
                    cells.push(format!("{} +{}/{}", show(r), found, case.callers.len()));
                }
                _ => {
                    totals[m][3] += usize::from(r.is_some());
                    cells.push(show(r));
                }
            }
        }
        match case.kind {
            "depth" => n_depth += 1,
            "breadth" => {
                n_def += 1;
                n_callers += case.callers.len();
            }
            _ => n_behaviour += 1,
        }
        println!(
            "{:<5} {:<10} {:>8} {:>6} {:>7}  {}",
            case.id, case.kind, cells[0], cells[1], cells[2], case.query
        );
    }
    println!("(depth cells: expected rank / best distractor rank)\n");
    println!("mode     depth  definitions  callers  behaviour");
    for (m, mode) in modes.iter().enumerate() {
        let t = totals[m];
        println!(
            "{:<8} {:>2}/{:<3} {:>6}/{:<5} {:>4}/{:<4} {:>4}/{}",
            mode.as_str(),
            t[0],
            n_depth,
            t[1],
            n_def,
            t[2],
            n_callers,
            t[3],
            n_behaviour
        );
    }
    println!(
        "\nearlier evaluation (2026-10-04, no headers): lexical 4/9, 4/4, 5/7, 4/6; dense (path + source) 9/9, 4/4, 2/7, 5/6"
    );
    println!(
        "tokens: index build {} (this run), queries {query_tokens}; total {}",
        d.tokens,
        d.tokens + query_tokens
    );
    Ok(())
}
