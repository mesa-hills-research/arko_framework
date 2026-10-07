use arko_index::bench_v1::{Chunk, Options, chunk_str};
use std::{collections::HashSet, path::Path};

fn chunks(path: &str, text: &str, max_bytes: usize) -> Vec<Chunk> {
    let cs = chunk_str(Path::new(path), text, &Options { max_bytes }).unwrap();
    let mut end = 0;
    for c in &cs {
        assert_eq!(c.start_byte, end);
        assert!(c.end_byte - c.start_byte <= max_bytes);
        assert!(text.is_char_boundary(c.start_byte) && text.is_char_boundary(c.end_byte));
        end = c.end_byte;
    }
    assert_eq!(end, text.len());
    cs
}

#[test]
fn compound_function_families_across_grammars() {
    for (path, source) in [
        (
            "x.rs",
            "fn render_plain() -> i32 { 10 }\nfn render_debug() -> i32 { 20 }\n",
        ),
        (
            "x.py",
            "def render_plain():\n    return 10\ndef render_debug():\n    return 20\n",
        ),
        (
            "x.ts",
            "function renderPlain(): number { return 10; }\nfunction renderDebug(): number { return 20; }\n",
        ),
        (
            "x.js",
            "function renderPlain() { return 10; }\nfunction renderDebug() { return 20; }\n",
        ),
        (
            "x.go",
            "package p\nfunc renderPlain() int { return 10 }\nfunc renderDebug() int { return 20 }\n",
        ),
        (
            "x.c",
            "int render_plain() { return 10; }\nint render_debug() { return 20; }\n",
        ),
        (
            "x.cpp",
            "int render_plain() { return 10; }\nint render_debug() { return 20; }\n",
        ),
    ] {
        let cs = chunks(path, source, 4096);
        assert!(
            cs.iter().any(|c| c.kind == "function_family"),
            "{path}: {cs:?}"
        );
    }
    // Tiny Java classes deliberately remain whole. Expose members in a larger
    // container; a preceding declaration must not become part of the family.
    let java = format!(
        "class Renderer {{\nint unrelated() {{ return 1; }}{}\nint renderPlain() {{ return 10; }}\nint renderDebug() {{ return 20; }}\n}}\n",
        " ".repeat(800)
    );
    assert!(
        chunks("x.java", &java, 4096)
            .iter()
            .any(|c| c.kind == "function_family")
    );
}

#[test]
fn families_keep_individual_definition_concentration() {
    let a = "fn render_plain() -> i32 { 10 }\n";
    let b = format!(
        "fn render_debug() -> i32 {{ {} 20 }}\n",
        "call(); ".repeat(70)
    );
    let source = format!("{a}{b}");
    let cs = chunks("x.rs", &source, 4096);
    assert!(!cs.iter().any(|c| c.kind == "function_family"));
    let short = "fn render_plain() -> i32 { 10 }\nfn render_debug() -> i32 { 20 }\n";
    assert!(
        !chunks("x.rs", short, 45)
            .iter()
            .any(|c| c.kind == "function_family")
    );
}

#[test]
fn families_respect_scope_and_declaration_barriers() {
    for text in [
        "mod a { fn render_plain() { 10; } }\nmod b { fn render_debug() { 20; } }\n",
        "fn render_plain() { 10; }\nconst OTHER: usize = 0;\nfn render_debug() { 20; }\n",
        "fn render() { 10; }\nfn render_debug() { 20; }\n",
        "fn render_plain() { 10; }\nfn compute_debug() { 20; }\n",
    ] {
        assert!(
            !chunks("x.rs", text, 4096)
                .iter()
                .any(|c| c.kind == "function_family"),
            "{text}"
        );
    }
}

#[test]
fn family_insertion_does_not_rephase_distant_groups() {
    let source = (0..60)
        .map(|i| {
            format!(
                "fn family{i}_plain() {{ work({i}); }}\nfn family{i}_debug() {{ work({i}); }}\n"
            )
        })
        .collect::<String>();
    let before = chunks("x.rs", &source, 4096);
    let after = chunks("x.rs", &format!("fn unrelated() {{}}\n{source}"), 4096);
    let ids: HashSet<_> = after.iter().map(|c| &c.id).collect();
    assert!(before.iter().all(|c| ids.contains(&c.id)));
    // An extremely long same-stem run is not globally packed into family windows.
    let run = (0..1000)
        .map(|i| format!("fn render_{i}() {{ work({i}); }}\n"))
        .collect::<String>();
    let before = chunks("x.rs", &run, 4096);
    let after = chunks("x.rs", &format!("fn unrelated() {{}}\n{run}"), 4096);
    let ids: HashSet<_> = after.iter().map(|c| &c.id).collect();
    assert!(before.iter().filter(|c| ids.contains(&c.id)).count() >= before.len() - 3);
}

#[test]
fn overloads_are_coherent_but_duplicate_definitions_stay_distinct() {
    let overloads = "int parse(int x) { return x; }\nint parse(double x) { return (int)x; }\n";
    assert_eq!(chunks("x.cpp", overloads, 4096)[0].kind, "function_family");
    let duplicate = "int parse(int x) { return x; }\nint parse(int x) { return x; }\n";
    let cs = chunks("x.cpp", duplicate, 4096);
    assert_eq!(cs.len(), 2);
    assert_ne!(cs[0].id, cs[1].id);
    let conditional = "#if FEATURE_A\nint render_plain() { return 1; }\n#else\nint render_debug() { return 2; }\n#endif\n";
    assert!(
        !chunks("x.cpp", conditional, 4096)
            .iter()
            .any(|c| c.kind == "function_family")
    );
}

#[test]
fn family_watch_delta_preserves_other_groups_and_replays() {
    use arko_index::bench_v1::open_watch;
    use std::{collections::HashMap, time::Duration};
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let source = (0..30)
        .map(|i| {
            format!(
                "fn family{i}_plain() {{ work({i}); }}\nfn family{i}_debug() {{ work({i}); }}\n"
            )
        })
        .collect::<String>();
    std::fs::write(root.path().join("x.rs"), &source).unwrap();
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    let mut snapshot: HashMap<_, _> = w
        .snapshot()
        .unwrap()
        .into_iter()
        .map(|c| (c.id.clone(), c))
        .collect();
    assert_eq!(snapshot.len(), 30);
    let initial = w.next_change(Duration::from_secs(1)).unwrap().unwrap();
    w.ack(initial.sequence).unwrap();
    let edited = source.replacen("work(0)", "more(0)", 1);
    std::fs::write(root.path().join("x.rs"), edited).unwrap();
    let d = w.next_change(Duration::from_secs(2)).unwrap().unwrap();
    assert_eq!(d.removed.len(), 1);
    assert_eq!(d.added.len(), 1);
    assert!(d.changed.is_empty());
    for id in &d.removed {
        assert!(snapshot.remove(id).is_some());
    }
    for c in &d.added {
        assert!(snapshot.insert(c.id.clone(), c.clone()).is_none());
    }
    let got: HashMap<_, _> = w
        .snapshot()
        .unwrap()
        .into_iter()
        .map(|c| (c.id.clone(), c))
        .collect();
    assert_eq!(
        serde_json::to_value(&snapshot).unwrap(),
        serde_json::to_value(&got).unwrap()
    );
    w.flush().unwrap();
    drop(w);
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    let replay = w.next_change(Duration::from_secs(1)).unwrap().unwrap();
    assert_eq!(
        serde_json::to_value(&d).unwrap(),
        serde_json::to_value(&replay).unwrap()
    );
    w.ack(d.sequence).unwrap();
    assert!(w.next_change(Duration::from_millis(5)).unwrap().is_none());
}
