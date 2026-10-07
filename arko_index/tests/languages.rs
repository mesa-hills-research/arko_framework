//! Every language's chunks: the invariants (exact coverage, byte bound, UTF-8
//! boundaries, lines, identities), determinism, stable IDs, bounded work on adversarial
//! input, mutated inputs, the structure the rules give each sample, Markdown's
//! metadata region, and the original languages' output against a golden file.
use arko_index::bench_v1::*;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    time::{Duration, Instant},
};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// One sample per grammar (and per file name rule), with the `language` its chunks get.
const SAMPLES: &[(&str, &str)] = &[
    ("sample.sh", "bash"),
    ("sample.cs", "c_sharp"),
    ("CMakeLists.txt", "cmake"),
    ("sample.css", "css"),
    ("sample.cu", "cuda"),
    ("sample.dart", "dart"),
    ("sample.dtd", "dtd"),
    ("sample.ex", "elixir"),
    ("sample.elm", "elm"),
    ("sample.erb", "embedded_template"),
    ("sample.erl", "erlang"),
    ("sample.f90", "fortran"),
    ("sample.fs", "fsharp"),
    ("sample.frag", "glsl"),
    ("sample.hs", "haskell"),
    ("sample.tf", "hcl"),
    ("sample.html", "html"),
    ("sample.jl", "julia"),
    ("sample.json", "json"),
    ("sample.kt", "kotlin"),
    ("sample.lua", "lua"),
    ("Makefile", "make"),
    ("sample.md", "markdown"),
    ("sample.mdx", "markdown"),
    ("sample.m", "objc"),
    ("sample.ml", "ocaml"),
    ("sample.mli", "ocaml"),
    ("sample.odin", "odin"),
    ("sample.pl", "perl"),
    ("sample.php", "php"),
    ("sample.ps1", "powershell"),
    ("sample.proto", "proto"),
    ("sample.r", "r"),
    ("sample.rb", "ruby"),
    ("sample.scala", "scala"),
    ("sample.sol", "solidity"),
    ("sample.sql", "sql"),
    ("sample.svelte", "svelte"),
    ("sample.swift", "swift"),
    ("sample.toml", "toml"),
    ("sample.v", "verilog"),
    ("sample.xml", "xml"),
    ("sample.yaml", "yaml"),
    ("sample.zig", "zig"),
];

/// The classic languages' samples (their output is frozen).
const CLASSIC: &[(&str, &str)] = &[
    ("../classic/Orders.java", "java"),
    ("../classic/emitter.js", "javascript"),
    ("../classic/Orders.tsx", "typescript"),
    ("../classic/ring.c", "c"),
    ("../classic/ring.h", "c"),
    ("../scopes/sample.rs", "rust"),
    ("../scopes/sample.py", "python"),
    ("../scopes/sample.ts", "typescript"),
    ("../scopes/sample.go", "go"),
    ("../scopes/sample.cpp", "cpp"),
];

fn all_samples() -> impl Iterator<Item = (&'static str, &'static str)> {
    SAMPLES.iter().chain(CLASSIC).copied()
}

fn read(sample: &str) -> String {
    fs::read_to_string(Path::new(DATA).join("languages").join(sample)).unwrap()
}

/// The file name a sample is chunked under (its own name, without directories).
fn name(sample: &str) -> &str {
    sample.rsplit('/').next().unwrap()
}

fn options(max_bytes: usize) -> Options {
    Options {
        max_bytes,
        ..Options::default()
    }
}

/// Chunks `text` and checks every invariant: the metadata region starts at byte 0, the
/// chunks cover the rest exactly and in order, each within the bound and on UTF-8
/// boundaries, with correct lines, IDs (path + trimmed content + occurrence) and token
/// estimates, all in one language.
fn verify(path: &str, text: &str, max: usize) -> FileChunks {
    let file = chunk_str_with_metadata(Path::new(path), text, &options(max)).unwrap();
    let mut at = 0;
    if let Some(m) = &file.metadata {
        assert_eq!(m.file, path);
        assert_eq!(m.metadata_range.start_byte, 0, "{path}");
        assert!(m.metadata_range.end_byte <= text.len());
        assert!(text.is_char_boundary(m.metadata_range.end_byte));
        at = m.metadata_range.end_byte;
    }
    let mut seen: HashMap<String, u64> = HashMap::new();
    let mut ids = HashSet::new();
    for c in &file.chunks {
        assert_eq!(
            c.start_byte, at,
            "{path} (max {max}): gap or overlap at {at}"
        );
        assert!(c.end_byte > c.start_byte && c.end_byte <= text.len());
        assert!(text.is_char_boundary(c.start_byte) && text.is_char_boundary(c.end_byte));
        assert!(
            c.end_byte - c.start_byte <= max,
            "{path} (max {max}): {c:?}"
        );
        assert!(path.len() + 1 + c.end_byte - c.start_byte <= 24_000);
        assert_eq!(
            c.start_line,
            text[..c.start_byte].matches('\n').count() + 1,
            "{path}"
        );
        assert_eq!(
            c.end_line,
            text.as_bytes()[..c.end_byte - 1]
                .iter()
                .filter(|&&b| b == b'\n')
                .count()
                + 1,
            "{path}"
        );
        let mut digest = Sha256::new();
        digest.update(path.as_bytes());
        digest.update([0]);
        digest.update(text[c.start_byte..c.end_byte].trim().as_bytes());
        let content = format!("{:x}", digest.finalize());
        let occurrence = seen.entry(content.clone()).or_default();
        assert_eq!(c.id, format!("{content}-{occurrence}"), "{path}");
        *occurrence += 1;
        assert!(ids.insert(c.id.clone()));
        assert_eq!(c.token_estimate, c.end_byte - c.start_byte + path.len() + 1);
        assert_eq!(c.language, file.chunks[0].language);
        assert_eq!(c.path, path);
        at = c.end_byte;
    }
    assert_eq!(at, text.len(), "{path} (max {max}): covered to {at}");
    file
}

#[test]
fn every_language_holds_the_invariants() {
    for (sample, language) in all_samples() {
        let text = read(sample);
        for max in [4, 7, 23, 64, 128, 256, 512, 1024, 4096, 100_000] {
            let file = verify(name(sample), &text, max);
            assert!(
                file.chunks.iter().all(|c| c.language == language),
                "{sample}: {:?}",
                file.chunks.first().map(|c| &c.language)
            );
        }
    }
}

#[test]
fn chunking_is_deterministic() {
    for (sample, _) in all_samples() {
        let text = read(sample);
        for max in [64, 4096] {
            let a = verify(name(sample), &text, max);
            let b = verify(name(sample), &text, max);
            assert_eq!(a, b, "{sample}");
        }
    }
}

/// An edit inside one chunk (a space before one of its line breaks) changes that chunk
/// and at most its neighbours; every other chunk keeps its ID.
#[test]
fn ids_survive_unrelated_edits() {
    for (sample, _) in all_samples() {
        let text = read(sample);
        for max in [256, 4096] {
            let before = verify(name(sample), &text, max).chunks;
            if before.len() < 3 {
                continue;
            }
            let middle = &before[before.len() / 2];
            let inner = &text[middle.start_byte..middle.end_byte];
            let Some(newline) = inner.trim_end().find('\n') else {
                continue;
            };
            let at = middle.start_byte + newline;
            let edited = format!("{} {}", &text[..at], &text[at..]);
            let after = verify(name(sample), &edited, max).chunks;
            let kept: HashSet<&str> = after.iter().map(|c| c.id.as_str()).collect();
            let lost: Vec<usize> = (0..before.len())
                .filter(|&i| !kept.contains(before[i].id.as_str()))
                .collect();
            let i = before.len() / 2;
            assert!(
                lost.iter().all(|&l| l + 1 >= i && l <= i + 1),
                "{sample} (max {max}): an edit in chunk {i} changed chunks {lost:?}"
            );
        }
    }
}

/// Giant lines, deep nesting and long runs of brackets or quotes stay bounded: exact
/// coverage, within the bound, in bounded time (the parser's work budget and the
/// fallback).
#[test]
fn adversarial_inputs_stay_bounded() {
    let inputs = [
        "x".repeat(70_000),
        "(".repeat(2_000),
        "{".repeat(1_000) + &"}".repeat(1_000),
        "[".repeat(1_000) + "\n" + &"\"".repeat(1_000),
        "<a>".repeat(1_000),
        "- ".repeat(500) + "\n" + &"  - x\n".repeat(300),
        "#".repeat(10) + " " + &"`".repeat(2_000),
        "\n".repeat(2_000),
        "é🦀 ".repeat(1_000),
    ];
    for (sample, _) in all_samples() {
        for (i, input) in inputs.iter().enumerate() {
            let started = Instant::now();
            verify(name(sample), input, [4096, 64][i % 2]);
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "{sample}: {} bytes took {:?}",
                input.len(),
                started.elapsed()
            );
        }
    }
}

/// A small deterministic generator (xorshift).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// A char boundary in `text` near a random position.
fn boundary(text: &str, rng: &mut Rng) -> usize {
    let mut at = rng.below(text.len() + 1);
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

const SNIPPETS: &[&str] = &[
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "\"",
    "'",
    "`",
    "<",
    ">",
    "</",
    "/*",
    "*/",
    "<!--",
    "-->",
    "\n",
    "\n\n",
    "    ",
    "\t",
    "#",
    "---\n",
    "+++\n",
    "@",
    "=",
    ":",
    ";",
    ",",
    "end",
    "def ",
    "fn ",
    "function ",
    "class ",
    "module ",
    "é",
    "🦀",
    "\u{feff}",
    "\r\n",
    "\\",
    "$",
    "%",
    "```",
    "import x from 'y'\n",
    "<script>",
    "</script>",
    "{{",
    "}}",
    "<%",
    "%>",
];

/// Random edits: deletions, duplications, insertions of syntax-heavy snippets, swapped
/// lines and truncations, on every grammar, at several bounds.
#[test]
fn mutated_inputs_hold_the_invariants() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (sample, _) in all_samples() {
        let original = read(sample);
        for round in 0..24 {
            let mut text = original.clone();
            for _ in 0..1 + rng.below(4) {
                let a = boundary(&text, &mut rng);
                let b = boundary(&text, &mut rng);
                let (a, b) = (a.min(b), a.max(b));
                text = match rng.below(5) {
                    0 => format!("{}{}", &text[..a], &text[b..]),
                    1 => format!("{}{}", &text[..b], &text[a..]),
                    2 => {
                        let snippet = SNIPPETS[rng.below(SNIPPETS.len())];
                        format!(
                            "{}{}{}",
                            &text[..a],
                            snippet.repeat(1 + rng.below(3)),
                            &text[a..]
                        )
                    }
                    3 => {
                        let mut lines: Vec<&str> = text.split_inclusive('\n').collect();
                        if lines.len() > 1 {
                            let i = rng.below(lines.len());
                            let j = rng.below(lines.len());
                            lines.swap(i, j);
                        }
                        lines.concat()
                    }
                    _ => text[..b].to_owned(),
                };
            }
            let max = [4, 16, 64, 256, 1024, 4096][round % 6];
            verify(name(sample), &text, max);
        }
    }
}

fn scope_at(file: &FileChunks, text: &str, needle: &str) -> String {
    let at = text.find(needle).unwrap_or_else(|| panic!("no {needle:?}"));
    file.chunks
        .iter()
        .find(|c| c.start_byte <= at && at < c.end_byte)
        .unwrap_or_else(|| panic!("{needle:?} is in no chunk"))
        .scope
        .clone()
}

/// The structure each language's rules give its sample: the scope of the chunk holding
/// a piece of text, at the default bound.
#[test]
fn rules_give_each_sample_its_structure() {
    let expected: &[(&str, &str, &str)] = &[
        ("sample.sh", "local host=$1", "function upload"),
        (
            "sample.cs",
            "foreach (var line",
            "namespace Shop.Orders › class OrderTotals › method Total",
        ),
        (
            "sample.cs",
            "_cache.Clear()",
            "namespace Shop.Orders › class OrderTotals › method Dispose",
        ),
        (
            "sample.cu",
            "out[i] = a[i] + b[i]",
            "namespace kernels › function add",
        ),
        ("sample.dart", "enum Shape", "enum Shape"),
        ("sample.dart", "sx += p.x", "extension PointList"),
        ("sample.ex", "Enum.sum()", "defmodule Shop.Cart › def total"),
        ("sample.ex", "@spec add", "defmodule Shop.Cart › def add"),
        ("sample.elm", "SetStep step ->", "update"),
        ("sample.erl", "maps:find", "handle_call"),
        (
            "sample.f90",
            "v = v / n",
            "module vectors › subroutine normalize",
        ),
        (
            "sample.fs",
            "match prices.TryGetValue",
            "namespace Shop › type Pricing",
        ),
        ("sample.frag", "reflect(-l, n)", "function specular"),
        ("sample.hs", "step (a : s) Dup", "step"),
        ("sample.hs", "pretty Mul", "instance Pretty Instr"),
        ("sample.tf", "required_providers", "terraform"),
        ("sample.jl", "s += a.x", "module Geometry › function area"),
        ("sample.kt", "fun main()", "fun main"),
        ("sample.lua", "table.insert", "function Emitter:on"),
        (
            "sample.m",
            "self.width * self.height",
            "@implementation Rectangle",
        ),
        ("sample.ml", "List.rev q.back", "module Queue"),
        ("sample.mli", "val add : elt", "module Make"),
        ("sample.odin", "math.PI", "proc area"),
        ("sample.pl", "bless $self", "sub new"),
        ("sample.php", "$this->cache[$sku]", "class OrderTotals"),
        ("sample.php", "number_format", "function format_money"),
        ("sample.ps1", "Test-Connection", "class Deployment"),
        ("sample.proto", "oneof payment", "message Order"),
        ("sample.proto", "rpc Get", "service Orders"),
        ("sample.r", "sd(x, na.rm", "function cv"),
        ("sample.rb", "JSON.parse", "module Shop › class Cart"),
        ("sample.scala", "getOrElseUpdate", "class OrderTotals"),
        (
            "sample.sol",
            "block.timestamp",
            "contract Vault › function deposit",
        ),
        (
            "sample.swift",
            "cache.removeAll()",
            "final class OrderTotals",
        ),
        ("sample.v", "count + 1'b1", "module counter"),
        ("sample.zig", "@sqrt", "const Point"),
        ("sample.zig", "std.testing", "test \"distance\""),
        (
            "sample.md",
            "cargo add geometry",
            "Geometry guide › # Geometry",
        ),
    ];
    for (sample, needle, scope) in expected {
        let text = read(sample);
        let file = verify(sample, &text, 4096);
        assert_eq!(
            &scope_at(&file, &text, needle),
            scope,
            "{sample}: {needle:?}"
        );
    }
    // Data and markup files that fit stay whole.
    for sample in [
        "sample.json",
        "sample.yaml",
        "sample.toml",
        "sample.xml",
        "sample.html",
    ] {
        let text = read(sample);
        assert_eq!(verify(sample, &text, 4096).chunks.len(), 1, "{sample}");
    }
}

/// Small bounds split prose, markup and data along their structure, with scopes.
#[test]
fn small_bounds_follow_the_structure() {
    let cases: &[(&str, usize, &str, &str)] = &[
        (
            "sample.json",
            80,
            "\"vitest\": \"^2.1.0\"",
            "devDependencies › vitest",
        ),
        ("sample.yaml", 200, "cargo clippy", "jobs › lint"),
        ("sample.toml", 120, "memory = \"512M\"", "workers.limits"),
        (
            "sample.xml",
            300,
            "junit-jupiter",
            "<project xmlns=\"http://maven.apache.org/POM/4.0.0\"> › <dependencies> › <dependency>",
        ),
        (
            "sample.html",
            300,
            "data.count",
            "<html lang=\"en\"> › <body> › <script>",
        ),
        (
            "sample.css",
            120,
            "grid-template-columns",
            "@media (min-width: 800px) › .grid",
        ),
        (
            "sample.sql",
            200,
            "GROUP BY c.id",
            "CREATE VIEW customer_totals AS",
        ),
        ("Makefile", 200, "pandoc README.md", "docs"),
        (
            "sample.md",
            200,
            "Points are",
            "Geometry guide › # Geometry › ## Usage",
        ),
    ];
    for (sample, max, needle, scope) in cases {
        let text = read(sample);
        let file = verify(sample, &text, *max);
        assert!(file.chunks.len() > 1, "{sample}");
        assert_eq!(
            &scope_at(&file, &text, needle),
            scope,
            "{sample}: {needle:?}"
        );
    }
    // A code fence that fits stays whole.
    let text = read("sample.md");
    let file = verify("sample.md", &text, 256);
    let fence = text.find("```rust").unwrap();
    let end = text[fence + 3..].find("```").unwrap() + fence + 6;
    assert!(
        file.chunks
            .iter()
            .any(|c| c.start_byte <= fence && end <= c.end_byte)
    );
}

#[test]
fn markdown_metadata_region_title_and_description() {
    let text = "---\ntitle: SQLite storage\ndescription: How chunks are stored.\nsidebar_position: 3\n---\n\nIntro.\n\n## Storage API\n\n### Transactions\n\nBEGIN.\n";
    let file = verify("storage.md", text, 4096);
    let m = file.metadata.as_ref().unwrap();
    assert_eq!(m.metadata_range.start_line, 1);
    assert_eq!(m.metadata_range.end_line, 5);
    assert_eq!(m.title.as_deref(), Some("SQLite storage"));
    assert_eq!(m.description.as_deref(), Some("How chunks are stored."));
    assert!(m.front_matter.contains("sidebar_position"));
    // Bookkeeping fields are in no chunk and no scope.
    assert!(file.chunks.iter().all(
        |c| !text[c.start_byte..c.end_byte].contains("sidebar") && !c.scope.contains("sidebar")
    ));
    let small = verify("storage.md", text, 24);
    assert_eq!(
        scope_at(&small, text, "BEGIN"),
        "SQLite storage › ## Storage API › ### Transactions"
    );
    assert_eq!(scope_at(&small, text, "Intro"), "SQLite storage");
    // Title and description, on request.
    let described = chunk_str_with_metadata(
        Path::new("storage.md"),
        text,
        &Options {
            max_bytes: 24,
            front_matter_description: true,
        },
    )
    .unwrap();
    assert_eq!(
        scope_at(&described, text, "BEGIN"),
        "SQLite storage — How chunks are stored. › ## Storage API › ### Transactions"
    );
    // Without front matter the first `# H1` is the root, and is not repeated.
    let text = "Preface.\n\n# Guide\n\nText.\n\n## Install\n\nSteps.\n";
    let file = verify("guide.md", text, 16);
    assert!(file.metadata.is_none());
    assert_eq!(scope_at(&file, text, "Preface"), "Guide");
    assert_eq!(scope_at(&file, text, "Steps"), "Guide › ## Install");
    // A title that repeats as the first heading is shown once.
    let text = "+++\ntitle = \"Guide\"\n+++\n# Guide\n\n## Use\n\nRun it.\n";
    let file = verify("guide.markdown", text, 16);
    assert_eq!(scope_at(&file, text, "Run it"), "Guide › ## Use");
    // MDX import/export lines after the front matter are metadata too.
    let text = read("sample.mdx");
    let file = verify("sample.mdx", &text, 4096);
    let end = file.metadata.as_ref().unwrap().metadata_range.end_byte;
    assert!(text[..end].contains("import { Story }"));
    assert!(
        text[end..].contains("export const meta"),
        "only the leading lines"
    );
    // A file of front matter alone has no chunks.
    let file = verify("empty.md", "---\ntitle: x\n---\n", 4096);
    assert!(file.chunks.is_empty() && file.metadata.is_some());
}

/// The original nine languages chunk exactly as before the other grammars: the golden
/// file is the `chunk` output of the build before them (branch arko-index-dense).
#[test]
fn original_languages_match_the_golden_output() {
    let dir = Path::new(DATA).join("classic");
    let chunks = chunk_dir(&dir, &Options::default()).unwrap();
    let golden: Vec<Chunk> = fs::read_to_string(Path::new(DATA).join("classic-golden.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(chunks, golden);
    for c in &golden {
        let line = serde_json::to_string(c).unwrap();
        assert!(!line.contains("\"file\""));
    }
}

/// Prints samples' chunks: `ARKO_SAMPLES=sample.ex:256,sample.md cargo test -p
/// arko_index --test languages print_samples -- --ignored --nocapture` (each sample
/// with an optional bound, 4096 by default).
#[test]
#[ignore]
fn print_samples() {
    let list = std::env::var("ARKO_SAMPLES").unwrap_or_else(|_| "sample.md".into());
    for spec in list.split(',') {
        let (sample, max) = match spec.split_once(':') {
            Some((s, m)) => (s, m.parse().unwrap()),
            None => (spec, 4096),
        };
        let text = read(sample);
        let file = verify(name(sample), &text, max);
        println!("== {sample} (max {max})");
        if let Some(m) = &file.metadata {
            println!("metadata {:?}", m.metadata_range);
        }
        for c in &file.chunks {
            let first = text[c.start_byte..c.end_byte]
                .trim()
                .lines()
                .next()
                .unwrap_or("");
            println!(
                "{:>4}-{:<4} {:>5}B {:<26} {:<50} {}",
                c.start_line,
                c.end_line,
                c.end_byte - c.start_byte,
                c.kind,
                c.scope,
                first
            );
        }
    }
}
