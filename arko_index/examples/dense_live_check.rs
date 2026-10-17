//! Live check of dense and hybrid retrieval with Voyage. **Spends Voyage tokens** and
//! needs `VOYAGE_API_KEY`.
//!
//! Builds an index2 with embeddings of the corpus given by `--corpus DIR`, then
//! runs the evaluation questions (`breadth-depth-arko.json`, `behaviour-arko.json`)
//! from the required `--eval-dir DIR` in lexical, dense and hybrid modes. Prints
//! each question's rank, the scores, and the tokens used.
//!
//! ```sh
//! cargo run --release -p arko_index --example dense_live_check -- --corpus DIR --eval-dir DIR --dry-run
//! VOYAGE_API_KEY=... cargo run --release -p arko_index --example dense_live_check -- --corpus DIR --eval-dir DIR
//! ```
//!
//! `--state DIR` selects persistent state, which defaults to
//! `EVAL_DIR/.dense-live-check/MODEL-DIMENSIONS`. A rerun embeds nothing new and
//! spends only query tokens.
//!
//! Other flags include `--model MODEL` (voyage-code-4 by default), `--dimensions N`
//! (the model's default), `--max-embed-tokens N` (2,500,000), `--fusion rrf|linear`,
//! `--rrf-k K`, `--lexical-weight W`, `--dense-weight W`, `--candidates N`, and
//! `--f32` (f32 vector codes instead of int8).
//!
//! `--dry-run` builds with a cap of 0 tokens in scratch state and prints what
//! embedding would take. It needs only the tokenizer download. `--top FILE` writes
//! every question's top 10 per mode as JSON lines (`-` prints them last).
//! `--voyage-key-stdin` reads the key from standard input instead of the environment.
//!
//! `--model hash` runs offline with the non-semantic hash embedder to exercise the
//! harness. Its lexical column is the real lexical result. `--model local`,
//! `local:MODEL`, `local:PRECISION` or `local:MODEL:PRECISION` runs a local model
//! with `--features local-embed`, in int8 by default or f32.
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

/// The embedder of `--model`: Voyage (needs the key), a local model, or the offline
/// hash embedder.
fn embedder(args: &Args) -> Result<Box<dyn Embedder>, Error> {
    if args.model == "hash" || args.model.starts_with("local") {
        return support::embedder_for(&support::descriptor_for(&args.model, args.dimensions)?);
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
    top: Option<PathBuf>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, Error> {
    let (mut corpus, mut eval_dir) = (None, None);
    let mut a = Args {
        corpus: PathBuf::new(),
        eval_dir: PathBuf::new(),
        state: None,
        model: "voyage-code-4".into(),
        dimensions: None,
        cap: 2_500_000,
        fusion: Fusion::default(),
        candidates: 50,
        quantization: Quantization::default(),
        dry_run: false,
        top: None,
    };
    let (mut method, mut rrf_k, mut lexical_weight, mut dense_weight) =
        ("rrf".to_owned(), None::<f64>, None::<f64>, None::<f64>);
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--corpus" => corpus = Some(value()?.into()),
            "--eval-dir" => eval_dir = Some(value()?.into()),
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
            "--top" => a.top = Some(value()?.into()),
            "--voyage-key-stdin" => {
                let mut key = String::new();
                std::io::stdin().read_line(&mut key)?;
                // SAFETY: no other thread runs yet.
                unsafe { std::env::set_var("VOYAGE_API_KEY", key.trim()) };
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: dense_live_check --corpus DIR --eval-dir DIR [--state DIR] [--model MODEL] [--dimensions N] [--max-embed-tokens N] [--fusion rrf|linear] [--rrf-k K] [--lexical-weight W] [--dense-weight W] [--candidates N] [--f32] [--dry-run] [--top FILE] [--voyage-key-stdin]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    a.corpus = corpus.ok_or("--corpus DIR is required. Choose the source tree to index.")?;
    a.eval_dir = eval_dir.ok_or(
        "--eval-dir DIR is required. Choose the folder containing breadth-depth-arko.json and behaviour-arko.json.",
    )?;
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

fn state_dir(args: &Args) -> PathBuf {
    args.state.clone().unwrap_or_else(|| {
        let dims = args.dimensions.map_or("default".into(), |d| d.to_string());
        args.eval_dir
            .join(".dense-live-check")
            .join(format!("{}-{dims}", args.model.replace(['/', ':'], "-")))
    })
}

fn main() -> Result<(), Error> {
    let args = parse_args(std::env::args().skip(1))?;
    let cases = load_cases(&args.eval_dir)?;
    let state = state_dir(&args);
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
            "dry run found {} chunks, {} distinct texts, {} to embed, {} tokens counted ({} cached)",
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
    let started = std::time::Instant::now();
    let s = build(&args.corpus, &state, &Options::default(), &settings)?;
    let build_secs = started.elapsed().as_secs_f64();
    let d = s.dense.expect("built with embeddings");
    println!(
        "index: {} chunks, {} distinct texts, {} from the cache, {} embedded now in {} requests: {} tokens, {build_secs:.1} s{}",
        d.docs,
        d.texts,
        d.cached,
        d.new,
        d.requests,
        d.tokens,
        if d.cap_reached {
            format!(
                ". Stopped at the cap with {} texts ({} tokens) not embedded",
                d.not_embedded, d.not_embedded_tokens
            )
        } else {
            String::new()
        }
    );

    // Every query embedded once, in one request.
    let queries: Vec<&str> = cases.iter().map(|c| c.query.as_str()).collect();
    let started = std::time::Instant::now();
    let embedded = embedder.embed(&queries, InputKind::Query)?;
    let query_tokens = embedded.tokens;
    println!(
        "queries: {} embedded in one request: {query_tokens} tokens, {:.2} s",
        queries.len(),
        started.elapsed().as_secs_f64()
    );
    // `--top`: one JSON line per question and mode, written at the end.
    let mut top = Vec::new();

    let modes = [Mode::Lexical, Mode::Dense, Mode::Hybrid];
    // (depth found, definitions found, callers found, behaviour found) per mode.
    let mut totals = [[0usize; 4]; 3];
    let (mut n_depth, mut n_def, mut n_callers, mut n_behaviour) = (0, 0, 0, 0);
    println!(
        "\ncase  kind       lexical  dense  hybrid  (rank of the expected chunk in the top 10, breadth shows the definition and then callers found)"
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
            if args.top.is_some() {
                let spans: Vec<String> = hits
                    .iter()
                    .map(|h| {
                        format!(
                            "{}:{}-{}",
                            h.chunk.path, h.chunk.start_line, h.chunk.end_line
                        )
                    })
                    .collect();
                top.push(
                    serde_json::json!({"id": case.id, "mode": mode.as_str(), "query": case.query, "hits": spans})
                        .to_string(),
                );
            }
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
    println!("(depth cells show expected rank / best distractor rank)\n");
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
        "\nearlier evaluation (2026-10-04, no headers) had lexical 4/9, 4/4, 5/7, 4/6 and dense (path + source) 9/9, 4/4, 2/7, 5/6"
    );
    println!(
        "tokens used for index build {} (this run), queries {query_tokens}, total {}",
        d.tokens,
        d.tokens + query_tokens
    );
    if let Some(path) = &args.top {
        let mut lines = top.join("\n");
        lines.push('\n');
        if path.as_os_str() == "-" {
            print!("{lines}");
        } else {
            std::fs::write(path, lines)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(flags: &[&str]) -> Result<Args, Error> {
        parse_args(flags.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn requires_corpus_and_evaluation_paths() {
        assert_eq!(
            args(&[]).err().unwrap().to_string(),
            "--corpus DIR is required. Choose the source tree to index."
        );
        assert_eq!(
            args(&["--corpus", "source"]).err().unwrap().to_string(),
            "--eval-dir DIR is required. Choose the folder containing breadth-depth-arko.json and behaviour-arko.json."
        );
        assert!(args(&["--eval-dir", "evaluation"]).is_err());
    }

    #[test]
    fn state_defaults_to_evaluation_folder_and_accepts_override() {
        let parsed = args(&["--corpus", "source", "--eval-dir", "evaluation"]).unwrap();
        assert_eq!(parsed.corpus, PathBuf::from("source"));
        assert_eq!(parsed.eval_dir, PathBuf::from("evaluation"));
        assert_eq!(
            state_dir(&parsed),
            PathBuf::from("evaluation/.dense-live-check/voyage-code-4-default")
        );
        let parsed = args(&[
            "--corpus",
            "source",
            "--eval-dir",
            "evaluation",
            "--state",
            "saved",
        ])
        .unwrap();
        assert_eq!(state_dir(&parsed), PathBuf::from("saved"));
    }

    #[test]
    fn model_names_stay_in_one_state_subdirectory() {
        let parsed = args(&[
            "--corpus",
            "source",
            "--eval-dir",
            "evaluation",
            "--model",
            "local:org/model",
        ])
        .unwrap();
        assert_eq!(
            state_dir(&parsed),
            PathBuf::from("evaluation/.dense-live-check/local-org-model-default")
        );
    }
}
