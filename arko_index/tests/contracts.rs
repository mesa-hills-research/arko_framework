use arko_index::bench_v1::*;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    time::Duration,
};

fn verify(path: &str, text: &str, max: usize) -> Vec<Chunk> {
    let chunks = chunk_str(Path::new(path), text, &Options { max_bytes: max }).unwrap();
    let mut prev = 0;
    let mut ids = HashSet::new();
    for c in &chunks {
        assert_eq!(c.start_byte, prev, "gap/overlap {c:?}");
        assert!(c.end_byte > c.start_byte && c.end_byte <= text.len());
        assert!(text.is_char_boundary(c.start_byte) && text.is_char_boundary(c.end_byte));
        assert!(c.end_byte - c.start_byte <= max);
        assert!(path.len() + 1 + c.end_byte - c.start_byte <= 24_000);
        assert_eq!(
            c.start_line,
            text[..c.start_byte].bytes().filter(|&b| b == b'\n').count() + 1
        );
        assert_eq!(
            c.end_line,
            text.as_bytes()[..c.end_byte - 1]
                .iter()
                .copied()
                .filter(|&b| b == b'\n')
                .count()
                + 1
        );
        assert!(ids.insert(c.id.clone()), "duplicate ID");
        prev = c.end_byte;
    }
    assert_eq!(prev, text.len());
    chunks
}
#[test]
fn utf8_fallback_single_line() {
    let text = "中文🦀🙂wordé".repeat(25000);
    let chunks = verify("single-line-utf8.txt", &text, 4096);
    assert!(chunks.len() > 100);
    assert!(
        chunks
            .iter()
            .all(|c| c.start_line == 1 && c.end_line == 1 && c.language == "text")
    );
    for max in [4, 5, 7, 29, 127, 1_000_000] {
        verify("unicode.xml", &text[..2100], max);
    }
}
#[test]
fn eight_languages_and_coverage() {
    let cases = [
        (
            "x.rs",
            "rust",
            "// a\nfn café() { println!(\"🦀\"); }\n\nfn b() {}\n",
        ),
        (
            "x.py",
            "python",
            "# a\nclass Things:\n    def get(self):\n        return '中'\n\n# end\n",
        ),
        (
            "x.ts",
            "typescript",
            "export type P<T> = T extends object ? keyof T : never;\n",
        ),
        (
            "x.tsx",
            "typescript",
            "export const Title = () => <b>中</b>;\n",
        ),
        ("x.js", "javascript", "function foo() { return 'é'; }\n"),
        (
            "x.go",
            "go",
            "package p\nfunc Foo() string { return \"中\" }\n",
        ),
        (
            "x.java",
            "java",
            "class Foo { public int get() { return 42; } }\n",
        ),
        ("x.c", "c", "/* hi */\nint f(void) { return 1; }\n"),
        (
            "x.cpp",
            "cpp",
            "namespace A { template<class T> T f(T x) { return x; } }\n",
        ),
        (
            "x.h",
            "c",
            "namespace A { template<class T> T f(T x) { return x; } }\n",
        ),
    ];
    for (path, lang, text) in cases {
        for max in [4, 23, 4096] {
            let chunks = verify(path, text, max);
            assert!(chunks.iter().all(|c| c.language == lang));
        }
    }
}
#[test]
fn malformed_and_giant_atoms() {
    for path in [
        "x.rs", "x.py", "x.js", "x.c", "x.cpp", "x.java", "x.go", "x.ts", "x.xml",
    ] {
        let text = format!(
            " \n\n/* {} */\n\nfn ☂ {{'{}' }}\n ",
            "🦀".repeat(10000),
            "x".repeat(10000)
        );
        verify(path, &text, 4096);
    }
    verify("x.md", "\n\r\n  \nend\n\n", 4);
    assert!(verify("x.rs", "", 4096).is_empty());
}
#[test]
fn stable_ids_after_unrelated_edit_and_duplicates() {
    let a = "fn first() { one(); }\n\nfn second() { two(); }\nfn same() {}\nfn same() {}\n";
    let b =
        "fn first() { one(); extra(); }\n\nfn second() { two(); }\nfn same() {}\nfn same() {}\n";
    let ca = verify("x.rs", a, 4096);
    let cb = verify("x.rs", b, 4096);
    assert_eq!(ca.len(), 4);
    assert_eq!(cb.len(), 4);
    for i in 1..ca.len() {
        assert_eq!(ca[i].id, cb[i].id);
    }
    assert_ne!(ca[2].id, ca[3].id);
}
#[test]
fn ignore_and_binary_rules() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join(".gitignore"), "omit/\n*.bin\n").unwrap();
    for d in ["omit", "target", ".git", "include"] {
        std::fs::create_dir(root.path().join(d)).unwrap();
        std::fs::write(root.path().join(d).join("x.py"), "pass\n").unwrap();
    }
    std::fs::write(root.path().join("nul"), b"a\0b").unwrap();
    std::fs::write(root.path().join("invalid"), [255u8, 254]).unwrap();
    std::fs::write(root.path().join("unicode.txt"), "🦀\n").unwrap();
    std::fs::write(root.path().join(".visible"), "yes\n").unwrap();
    // Defaults: ignore files are not read and `target` is an ordinary name; only `.git`
    // is excluded. Binary and invalid UTF-8 files are skipped and reported.
    let chunks = chunk_dir(root.path(), &Options::default()).unwrap();
    let paths: HashSet<_> = chunks.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(
        paths,
        HashSet::from([
            ".gitignore",
            ".visible",
            "include/x.py",
            "omit/x.py",
            "target/x.py",
            "unicode.txt"
        ])
    );
    let (_, report) =
        chunk_dir_with(root.path(), &Options::default(), &FileRules::default()).unwrap();
    let skipped: Vec<_> = report
        .skipped
        .iter()
        .map(|s| (s.path.as_str(), s.reason))
        .collect();
    assert_eq!(
        skipped,
        [
            (".git/", SkipReason::GitDir),
            ("invalid", SkipReason::NotUtf8),
            ("nul", SkipReason::Binary)
        ]
    );
    assert_eq!(report.files, 6);
    // Opt-in: the root .gitignore applies.
    let mut rules = FileRules::default();
    rules.respect_ignore = true;
    let (chunks, _) = chunk_dir_with(root.path(), &Options::default(), &rules).unwrap();
    let paths: HashSet<_> = chunks.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(
        paths,
        HashSet::from([
            ".gitignore",
            ".visible",
            "include/x.py",
            "target/x.py",
            "unicode.txt"
        ])
    );
}
fn apply(snapshot: &mut HashMap<String, Chunk>, d: &Delta) {
    for id in &d.removed {
        assert!(snapshot.remove(id).is_some(), "unknown removal");
    }
    for c in &d.added {
        assert!(
            snapshot.insert(c.id.clone(), c.clone()).is_none(),
            "duplicate add"
        );
    }
    for c in &d.changed {
        assert!(
            snapshot.insert(c.id.clone(), c.clone()).is_some(),
            "unknown change"
        );
    }
}
#[test]
fn watch_replay_ack_debounce_and_restart() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let a = root.path().join("a.rs");
    std::fs::write(&a, "fn first() {}\nfn second() {}\n").unwrap();
    let mut w = open_watch(root.path(), &state, &Options::default()).unwrap();
    let d1 = w.next_change(Duration::ZERO).unwrap().unwrap();
    let mut snapshot = HashMap::new();
    apply(&mut snapshot, &d1);
    assert_eq!(snapshot.len(), 2);
    assert!(w.next_change(Duration::from_millis(10)).unwrap().is_none());
    w.flush().unwrap();
    drop(w);
    // A modification while down is a new event, after the durable unacked event.
    std::fs::write(&a, "fn first() { value(); }\nfn second() {}\n").unwrap();
    let mut w = open_watch(root.path(), &state, &Options::default()).unwrap();
    let replay = w.next_change(Duration::ZERO).unwrap().unwrap();
    assert_eq!(replay.sequence, d1.sequence);
    let d2 = w.next_change(Duration::ZERO).unwrap().unwrap();
    assert!(d2.sequence > d1.sequence);
    apply(&mut snapshot, &d2);
    assert_eq!(d2.changed.len(), 1); // shifted second, stable content ID
    assert_eq!(d2.added.len(), 1);
    assert_eq!(d2.removed.len(), 1);
    w.ack(d2.sequence).unwrap();
    drop(w);
    let mut w = open_watch(root.path(), &state, &Options::default()).unwrap();
    assert!(w.next_change(Duration::from_millis(10)).unwrap().is_none());
    std::fs::rename(&a, root.path().join("b.rs")).unwrap();
    let d3 = w.next_change(Duration::from_secs(2)).unwrap().unwrap();
    apply(&mut snapshot, &d3);
    assert!(snapshot.values().all(|c| c.path == "b.rs"));
    w.ack(d3.sequence).unwrap();
    std::fs::remove_file(root.path().join("b.rs")).unwrap();
    let d4 = w.next_change(Duration::from_secs(2)).unwrap().unwrap();
    apply(&mut snapshot, &d4);
    assert!(snapshot.is_empty());
    assert!(w.snapshot().unwrap().is_empty());
}
#[test]
fn lexical_cli_library() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("x.rs"),
        "fn retriesExponentialBackoff() { retry(); }\nfn other() { banana(); }\n",
    )
    .unwrap();
    assert_eq!(
        arko_index::search::index(root.path(), state.path(), &Options::default()).unwrap(),
        2
    );
    let hits = arko_index::search::search(state.path(), "exponential backoff").unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].symbol_path.contains("retriesExponentialBackoff"));
}
#[test]
fn facade_exact_shapes() {
    use std::path::PathBuf;
    let _x: fn(&Path, &str, &Options) -> Result<Vec<Chunk>, Error> = chunk_str;
    let _x: fn(&Path, &Options) -> Result<Vec<Chunk>, Error> = chunk_file;
    let _x: fn(&Path, &Options) -> Result<Vec<Chunk>, Error> = chunk_dir;
    let _x: fn(&Path, &Path, &Options) -> Result<Watcher, Error> = open_watch;
    let _x: fn(&mut Watcher, Duration) -> Result<Option<Delta>, Error> = Watcher::next_change;
    let _x: fn(&Watcher) -> Result<Vec<Chunk>, Error> = Watcher::snapshot;
    let _x: fn(&mut Watcher, u64) -> Result<(), Error> = Watcher::ack;
    let _x: fn(&mut Watcher) -> Result<(), Error> = Watcher::flush;
    let _ = PathBuf::new();
    let span = SourceSpan {
        start_byte: 0,
        end_byte: 1,
        start_line: 1,
        end_line: 1,
    };
    assert!(serde_json::to_string(&span).is_ok());
}

#[test]
fn debounce_survives_short_poll_timeouts() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    assert!(w.next_change(Duration::ZERO).unwrap().is_none());
    std::fs::write(root.path().join("new.py"), "print('🦀')\n").unwrap();
    let began = std::time::Instant::now();
    loop {
        if let Some(d) = w.next_change(Duration::from_millis(15)).unwrap() {
            assert_eq!(d.added.len(), 1);
            w.ack(d.sequence).unwrap();
            break;
        }
        assert!(
            began.elapsed() < Duration::from_secs(2),
            "short polls starved debounce"
        );
    }
}
#[test]
fn parser_work_and_render_bounds() {
    let giant = format!("{}{}", "?é☂/*\"".repeat(15000), "(".repeat(50000));
    for ext in [
        "rs", "py", "ts", "tsx", "js", "java", "go", "c", "h", "cpp", "txt", "xml",
    ] {
        let path = format!("x.{ext}");
        let cs = verify(&path, &giant, 4096);
        assert!(cs.len() > 10);
    }
    // Nondefault huge options cannot bypass the rendered-byte guard.
    verify("long.md", &"x".repeat(100000), usize::MAX);
    let path = format!("{}.txt", "x".repeat(23000));
    verify(&path, &"🦀".repeat(100), 4096);
    assert!(chunk_str(Path::new(&"x".repeat(24000)), "abc", &Options::default()).is_err());
    assert!(chunk_str(Path::new("x.rs"), "abc", &Options { max_bytes: 0 }).is_err());
}
#[test]
fn macro_recovery_and_nonspecific_wrappers() {
    let cpp = "namespace example {\nLIBRARY_API inline int alpha(int x) { return x + 1; }\nLIBRARY_INLINE int beta(int x) { return x - 1; }\n}\n";
    let cs = verify("export.h", cpp, 4096);
    assert!(cs.iter().any(|c| c.symbol_path.contains("beta")), "{cs:?}");
    let js = "object.extend({alpha: function () { return 1; }, beta: function() { return 2; }});\n";
    let cs = verify("methods.js", js, 4096);
    assert!(cs.iter().any(|c| c.symbol_path.contains("alpha")), "{cs:?}");
    assert!(cs.iter().any(|c| c.symbol_path.contains("beta")), "{cs:?}");
}

#[test]
fn same_size_whitespace_edit_detected_durably() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let path = root.path().join("code.rs");
    std::fs::write(&path, "fn untouched() {} \n").unwrap();
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    let d = w.next_change(Duration::ZERO).unwrap().unwrap();
    let old = d.added[0].id.clone();
    w.ack(d.sequence).unwrap();
    drop(w);
    std::fs::write(&path, "fn untouched() {}\t\n").unwrap();
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    let d = w.next_change(Duration::ZERO).unwrap().unwrap();
    assert!(d.added.is_empty() && d.removed.is_empty());
    assert_eq!(d.changed.len(), 1);
    assert_eq!(d.changed[0].id, old);
    w.ack(d.sequence).unwrap();
    drop(w);
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    assert!(w.next_change(Duration::from_millis(10)).unwrap().is_none());
}
#[test]
fn generated_tiny_siblings_merge_and_resynchronize() {
    let source = (0..150)
        .map(|i| format!("function repeat_{i}() {{ return {i}; }}\n"))
        .collect::<String>();
    let chunks = verify("helpers.js", &source, 4096);
    assert!(chunks.len() < 100, "tiny run was not merged");
    let prefix = "function new_one() { return 9; }\n";
    let edited = format!("{prefix}{source}");
    let cs = verify("helpers.js", &edited, 4096);
    let old: HashSet<_> = chunks.iter().map(|c| c.id.clone()).collect();
    let stable = cs.iter().filter(|c| old.contains(&c.id)).count();
    assert!(
        stable >= chunks.len() - 3,
        "content-defined cuts did not resynchronize"
    );
}

#[test]
fn deeply_nested_ast_descent_is_stack_bounded() {
    // Long-line detection cannot handle this: every parenthesis is on its own line.
    let text = format!(
        "fn main() {{ let value = {}1{}; }}\n",
        "(\n".repeat(100000),
        ")\n".repeat(100000)
    );
    let cs = chunk_str(Path::new("deep.rs"), &text, &Options::default()).unwrap();
    let mut p = 0;
    for c in cs {
        assert_eq!(c.start_byte, p);
        assert!(c.end_byte - c.start_byte <= 4096);
        assert!(text.is_char_boundary(c.end_byte));
        p = c.end_byte;
    }
    assert_eq!(p, text.len());
}
