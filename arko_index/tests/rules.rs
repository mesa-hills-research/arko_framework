//! Which files are indexed: defaults, opt-in ignore files, exclude/include lists,
//! skipped-file reports, and the watchers following rule changes.
#![cfg(unix)]
use arko_index::bench_v1::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    time::Duration,
};

const TAG: &str = "Signature: 8a477f597d28d172789f06886806bc55\n# a cache directory\n";

/// The `arko_index_cli` example. `cargo test` builds the package's examples before
/// it runs the tests; integration tests live in `<target>/<profile>/deps`, examples in
/// `<target>/<profile>/examples`.
fn cli() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().and_then(Path::parent).unwrap();
    let cli = profile_dir
        .join("examples")
        .join(format!("arko_index_cli{}", std::env::consts::EXE_SUFFIX));
    assert!(
        cli.is_file(),
        "{} is missing: run the whole suite (`cargo test -p arko_index`) or build it first \
         (`cargo build -p arko_index --example arko_index_cli`)",
        cli.display()
    );
    cli
}

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, text).unwrap();
}
fn paths(root: &Path, rules: &FileRules) -> BTreeSet<String> {
    let (chunks, _) = chunk_dir_with(root, &Options::default(), rules).unwrap();
    chunks.into_iter().map(|c| c.path).collect()
}
fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}
fn skipped(report: &Report) -> Vec<(String, SkipReason)> {
    report
        .skipped
        .iter()
        .map(|s| (s.path.clone(), s.reason))
        .collect()
}
fn respect() -> FileRules {
    let mut r = FileRules::default();
    r.respect_ignore = true;
    r
}

#[test]
fn ignore_files_above_the_root_never_apply() {
    let parent = tempfile::tempdir().unwrap();
    // A repository above the root that ignores everything the root contains.
    fs::create_dir(parent.path().join(".git")).unwrap();
    write(parent.path(), ".gitignore", "*.rs\nproj/sub/\n");
    write(parent.path(), ".ignore", "*\n");
    let root = parent.path().join("proj");
    write(&root, "a.rs", "fn a() {}\n");
    write(&root, "sub/b.txt", "b\n");
    let all = set(&["a.rs", "sub/b.txt"]);
    assert_eq!(paths(&root, &FileRules::default()), all);
    assert_eq!(paths(&root, &respect()), all);
}

#[test]
fn root_and_nested_ignore_files_apply_only_when_opted_in() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".gitignore", "*.log\ngen/\n");
    write(r, ".ignore", "skip.txt\n");
    write(r, "sub/.gitignore", "secret.rs\n!keep.log\n");
    // Applied with the opt-in, as Git does.
    write(r, ".git/info/exclude", "a.rs\n");
    for f in [
        "a.rs",
        "x.log",
        "gen/g.rs",
        "skip.txt",
        "sub/secret.rs",
        "sub/keep.log",
        "sub/o.rs",
    ] {
        write(r, f, "fn f() {}\n");
    }
    let everything = set(&[
        ".gitignore",
        ".ignore",
        "a.rs",
        "gen/g.rs",
        "skip.txt",
        "sub/.gitignore",
        "sub/keep.log",
        "sub/o.rs",
        "sub/secret.rs",
        "x.log",
    ]);
    assert_eq!(paths(r, &FileRules::default()), everything);
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &respect()).unwrap();
    let got: BTreeSet<String> = chunks.into_iter().map(|c| c.path).collect();
    assert_eq!(
        got,
        set(&[
            ".gitignore",
            ".ignore",
            "sub/.gitignore",
            "sub/keep.log",
            "sub/o.rs"
        ])
    );
    // Removed entries are counted and listed per source (a directory counts once).
    assert_eq!(report.ignored_by_gitignore, 3);
    assert_eq!(report.ignored_by_ignore_file, 1);
    assert_eq!(report.ignored_by_git_info_exclude, 1);
    let ignored: Vec<_> = report
        .ignored
        .iter()
        .map(|s| (s.path.as_str(), s.reason))
        .collect();
    assert_eq!(
        ignored,
        [
            ("a.rs", SkipReason::GitInfoExclude),
            ("gen/", SkipReason::Gitignore),
            ("skip.txt", SkipReason::IgnoreFile),
            ("sub/secret.rs", SkipReason::Gitignore),
            ("x.log", SkipReason::Gitignore),
        ]
    );
    // Without the default, nothing is counted.
    let (_, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    assert_eq!(report.ignored_total(), 0);
    // A scan without a Git checkout applies the ignore files the same way.
    fs::remove_dir_all(r.join(".git")).unwrap();
    let got = paths(r, &respect());
    assert!(!got.contains("x.log") && got.contains("a.rs"));
}

#[test]
fn ignore_file_precedence_follows_git_and_the_ignore_crate() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".gitignore", "*.log\n!b.tmp\n!sub/y.txt\n");
    write(r, ".ignore", "!keep.log\n");
    write(r, ".git/info/exclude", "*.tmp\n");
    write(r, "sub/.ignore", "y.txt\n");
    write(r, "sub/.git/info/exclude", "z.rs\n");
    for f in [
        "keep.log",
        "x.log",
        "b.tmp",
        "c.tmp",
        "sub/y.txt",
        "sub/z.rs",
        "sub/w.tmp",
    ] {
        write(r, f, "x\n");
    }
    let (_, report) = chunk_dir_with(r, &Options::default(), &respect()).unwrap();
    let ignored: Vec<_> = report
        .ignored
        .iter()
        .map(|s| (s.path.as_str(), s.reason))
        .collect();
    assert_eq!(
        ignored,
        [
            // .git/info/exclude ranks below .gitignore, which re-includes b.tmp.
            ("c.tmp", SkipReason::GitInfoExclude),
            // The outer repository's exclude file also applies inside a nested one.
            ("sub/w.tmp", SkipReason::GitInfoExclude),
            // .ignore outranks .gitignore at any depth.
            ("sub/y.txt", SkipReason::IgnoreFile),
            ("sub/z.rs", SkipReason::GitInfoExclude),
            ("x.log", SkipReason::Gitignore),
        ]
    );
    assert!(paths(r, &respect()).contains("keep.log"));
}

#[test]
fn ignored_list_is_capped_but_counts_are_complete() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".gitignore", "*.gen\n");
    for i in 0..1005 {
        fs::write(r.join(format!("f{i:04}.gen")), "x\n").unwrap();
    }
    let (_, report) = chunk_dir_with(r, &Options::default(), &respect()).unwrap();
    assert_eq!(report.ignored_by_gitignore, 1005);
    assert_eq!(report.ignored.len(), 1000);
    assert_eq!(report.ignored[0].path, "f0000.gen");
    assert!(report.summary().contains("1005 ignored (1005 .gitignore"));
}

#[test]
fn git_files_are_excluded_and_default_exclusions_alone_are_quiet() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".git/HEAD", "ref: refs/heads/main\n");
    write(r, "vendor/lib/.git", "gitdir: ../../.git/modules/lib\n");
    write(r, "vendor/lib/lib.rs", "fn lib() {}\n");
    write(r, "target/CACHEDIR.TAG", TAG);
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    assert!(chunks.iter().all(|c| c.path == "vendor/lib/lib.rs"));
    assert_eq!(
        skipped(&report),
        [
            (".git/".to_string(), SkipReason::GitDir),
            ("target/".to_string(), SkipReason::CachedirTag),
            ("vendor/lib/.git".to_string(), SkipReason::GitFile),
        ]
    );
    assert!(!report.worth_printing());
    fs::write(r.join("blob.bin"), b"\0").unwrap();
    let (_, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    assert!(report.worth_printing());
}

#[test]
fn an_include_pattern_naming_a_tagged_directory_brings_it_back() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, "src/a.rs", "fn a() {}\n");
    write(r, "target/CACHEDIR.TAG", TAG);
    write(r, "target/debug/gen.rs", "fn gen() {}\n");
    let with = |include: &[&str], exclude: &[&str]| {
        let mut rules = FileRules::default();
        rules.include = include.iter().map(|s| s.to_string()).collect();
        rules.exclude = exclude.iter().map(|s| s.to_string()).collect();
        paths(r, &rules)
    };
    // A pattern that names the directory brings it back.
    assert_eq!(
        with(&["src/", "target/"], &[]),
        set(&["src/a.rs", "target/CACHEDIR.TAG", "target/debug/gen.rs"])
    );
    assert!(with(&["/target/"], &[]).contains("target/debug/gen.rs"));
    // A pattern that only matches files inside it does not.
    assert_eq!(with(&["*.rs"], &[]), set(&["src/a.rs"]));
    // An explicit exclusion still wins.
    assert_eq!(with(&["src/", "target/"], &["target/"]), set(&["src/a.rs"]));
    assert_eq!(
        with(&["src/", "target/"], &["*.rs"]),
        set(&["target/CACHEDIR.TAG"])
    );
}

#[test]
fn cachedir_tag_excludes_only_valid_tags_and_target_is_an_ordinary_name() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, "target/CACHEDIR.TAG", TAG);
    write(r, "target/debug/build.rs", "fn cached() {}\n");
    write(r, "drivers/target/core.c", "int core(void) { return 0; }\n");
    write(r, "bad/CACHEDIR.TAG", "Signature: 0000\n");
    write(r, "bad/x.rs", "fn x() {}\n");
    write(
        r,
        "short/CACHEDIR.TAG",
        "Signature: 8a477f597d28d172789f06886806bc5",
    );
    write(r, "short/y.rs", "fn y() {}\n");
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    let got: BTreeSet<String> = chunks.into_iter().map(|c| c.path).collect();
    assert_eq!(
        got,
        set(&[
            "bad/CACHEDIR.TAG",
            "bad/x.rs",
            "drivers/target/core.c",
            "short/CACHEDIR.TAG",
            "short/y.rs"
        ])
    );
    assert_eq!(
        skipped(&report),
        [("target/".to_string(), SkipReason::CachedirTag)]
    );
    // A tagged root is still indexed when it is asked for explicitly.
    assert_eq!(
        paths(&r.join("target"), &FileRules::default()),
        set(&["CACHEDIR.TAG", "debug/build.rs"])
    );
}

#[test]
fn exclude_and_include_patterns() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    for f in [
        "src/a.rs",
        "src/gen/b.rs",
        "docs/x.md",
        "vendor/v.rs",
        "README.md",
        "notes.txt",
        "sub/notes.txt",
    ] {
        write(r, f, "text\n");
    }
    let mut rules = FileRules::default();
    rules.exclude = vec![
        "vendor/".into(),
        "*.md".into(),
        "!README.md".into(),
        "/notes.txt".into(),
    ];
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &rules).unwrap();
    let got: BTreeSet<String> = chunks.into_iter().map(|c| c.path).collect();
    assert_eq!(
        got,
        set(&["README.md", "src/a.rs", "src/gen/b.rs", "sub/notes.txt"])
    );
    assert_eq!(report.excluded_by_pattern, 3); // vendor/, docs/x.md, notes.txt
    assert!(report.skipped.is_empty());

    let mut rules = FileRules::default();
    rules.include = vec!["src/".into(), "*.md".into()];
    rules.exclude = vec!["src/gen/".into(), "docs/".into()];
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &rules).unwrap();
    let got: BTreeSet<String> = chunks.into_iter().map(|c| c.path).collect();
    // Exclusions win over inclusions.
    assert_eq!(got, set(&["README.md", "src/a.rs"]));
    assert_eq!(report.not_included, 3); // notes.txt, sub/notes.txt, vendor/v.rs
    assert_eq!(report.excluded_by_pattern, 2);

    let mut bad = FileRules::default();
    bad.exclude = vec!["src/{a".into()];
    assert!(bad.validate().is_err());
    assert!(chunk_dir_with(r, &Options::default(), &bad).is_err());
}

#[test]
fn invalid_ignore_file_lines_are_warnings_not_errors() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".gitignore", "src/{a\n*.log\n");
    write(r, "a.rs", "fn a() {}\n");
    write(r, "x.log", "x\n");
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &respect()).unwrap();
    let got: BTreeSet<String> = chunks.into_iter().map(|c| c.path).collect();
    // The valid line still applies; the invalid one is reported, and the run goes on.
    assert_eq!(got, set(&[".gitignore", "a.rs"]));
    assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    assert!(report.skipped.is_empty());
}

#[test]
fn latin1_binary_symlink_and_special_files_are_skipped_and_reported() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, "ok.rs", "fn ok() {}\n");
    fs::write(r.join("latin1.map"), b"caf\xe9 au lait\n").unwrap();
    fs::write(r.join("blob.bin"), b"\x7fELF\0\0\x01").unwrap();
    std::os::unix::fs::symlink("ok.rs", r.join("link.rs")).unwrap();
    let _socket = std::os::unix::net::UnixListener::bind(r.join("sock")).unwrap();
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    assert!(chunks.iter().all(|c| c.path == "ok.rs"));
    assert_eq!(report.files, 1);
    assert_eq!(
        skipped(&report),
        [
            ("blob.bin".to_string(), SkipReason::Binary),
            ("latin1.map".to_string(), SkipReason::NotUtf8),
            ("link.rs".to_string(), SkipReason::Symlink),
            ("sock".to_string(), SkipReason::Special),
        ]
    );
    assert_eq!(report.counts.get(&SkipReason::NotUtf8), Some(&1));
    // The JSON form uses the documented names.
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["skipped"][1]["reason"], "not-utf8");
    assert_eq!(json["counts"]["binary"], 1);
}

#[test]
fn unreadable_files_and_directories_are_reported_not_fatal() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, "ok.rs", "fn ok() {}\n");
    write(r, "secret.rs", "fn secret() {}\n");
    write(r, "locked/inner.rs", "fn inner() {}\n");
    fs::set_permissions(r.join("secret.rs"), fs::Permissions::from_mode(0o000)).unwrap();
    fs::set_permissions(r.join("locked"), fs::Permissions::from_mode(0o000)).unwrap();
    let readable = fs::read(r.join("secret.rs")).is_ok(); // e.g. running as root
    let result = chunk_dir_with(r, &Options::default(), &FileRules::default());
    fs::set_permissions(r.join("locked"), fs::Permissions::from_mode(0o755)).unwrap();
    let (chunks, report) = result.unwrap();
    if readable {
        return;
    }
    assert!(chunks.iter().all(|c| c.path == "ok.rs"));
    let got = skipped(&report);
    assert_eq!(
        got,
        [
            ("locked/".to_string(), SkipReason::Unreadable),
            ("secret.rs".to_string(), SkipReason::Unreadable)
        ]
    );
    assert!(report.skipped.iter().all(|s| s.detail.is_some()));
}

#[test]
fn non_utf8_names_are_skipped_and_reported() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, "good.rs", "fn good() {}\n");
    // APFS (macOS) refuses names that aren't UTF-8, so the case can't arise there.
    if let Err(error) = fs::write(
        r.join(OsString::from_vec(b"bad\xff.rs".to_vec())),
        "fn bad() {}\n",
    ) {
        assert_eq!(error.raw_os_error(), Some(libc::EILSEQ), "{error}");
        eprintln!("skipped: this file system refuses non-UTF-8 names ({error})");
        return;
    }
    let dir = r.join(OsString::from_vec(b"dir\xfe".to_vec()));
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("inner.rs"), "fn inner() {}\n").unwrap();
    let (chunks, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    assert!(chunks.iter().all(|c| c.path == "good.rs"));
    assert_eq!(
        skipped(&report),
        [
            ("bad\\xFF.rs".to_string(), SkipReason::NameNotUtf8),
            ("dir\\xFE/".to_string(), SkipReason::NameNotUtf8),
        ]
    );
}

#[test]
fn indexes_exclude_their_state_and_write_the_report() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, "a.rs", "fn alpha() {}\n");
    fs::write(r.join("b.bin"), b"\0").unwrap();
    let state = r.join("state");
    let (n, report) =
        arko_index::search::index_with(r, &state, &Options::default(), &FileRules::default())
            .unwrap();
    assert!(n > 0);
    assert_eq!(
        skipped(&report),
        [
            ("b.bin".to_string(), SkipReason::Binary),
            ("state/".to_string(), SkipReason::StateDir)
        ]
    );
    let saved: Report =
        serde_json::from_slice(&fs::read(state.join("report.json")).unwrap()).unwrap();
    assert_eq!(saved, report);
    // index2 applies the same rules (here: an exclude pattern).
    let mut rules = FileRules::default();
    rules.exclude = vec!["*.rs".into()];
    let state2 = r.join("state2");
    let stats = arko_index::index2::index_with(r, &state2, &Options::default(), 1, 1 << 20, &rules)
        .unwrap();
    assert_eq!(stats.docs, 0);
    assert_eq!(stats.report.excluded_by_pattern, 1);
    let saved: Report =
        serde_json::from_slice(&fs::read(state2.join("report.json")).unwrap()).unwrap();
    assert_eq!(saved, stats.report);
    // Independent chunk_dir scans skip both state directories through their marker.
    let (_, report) = chunk_dir_with(r, &Options::default(), &FileRules::default()).unwrap();
    let states: Vec<_> = report
        .skipped
        .iter()
        .filter(|s| s.reason == SkipReason::StateDir)
        .map(|s| s.path.as_str())
        .collect();
    assert_eq!(states, ["state/", "state2/"]);
}

// ---- Watchers ----

type Model = BTreeMap<String, Chunk>;
fn apply(model: &mut Model, d: &Delta) {
    for id in &d.removed {
        assert!(model.remove(id).is_some(), "removed unknown id {id}");
    }
    for c in d.added.iter().chain(&d.changed) {
        model.insert(c.id.clone(), c.clone());
    }
}
fn fresh(root: &Path, rules: &FileRules) -> Model {
    let (chunks, _) = chunk_dir_with(root, &Options::default(), rules).unwrap();
    chunks.into_iter().map(|c| (c.id.clone(), c)).collect()
}
fn model_paths(model: &Model) -> BTreeSet<String> {
    model.values().map(|c| c.path.clone()).collect()
}

/// The two watchers behind one interface for the shared scenarios.
trait Watch {
    fn next(&mut self, timeout: Duration) -> Option<Delta>;
    fn report(&self) -> Report;
    fn rules_changed(&self) -> bool;
}
impl Watch for Watcher {
    fn next(&mut self, timeout: Duration) -> Option<Delta> {
        let d = self.next_change(timeout).unwrap()?;
        self.ack(d.sequence).unwrap();
        Some(d)
    }
    fn report(&self) -> Report {
        Watcher::report(self).clone()
    }
    fn rules_changed(&self) -> bool {
        Watcher::rules_changed(self)
    }
}
#[cfg(target_os = "linux")]
impl Watch for arko_index::watch2::Watcher2 {
    fn next(&mut self, timeout: Duration) -> Option<Delta> {
        let d = self.next_change(timeout).unwrap()?;
        self.ack(d.sequence).unwrap();
        Some(d)
    }
    fn report(&self) -> Report {
        arko_index::watch2::Watcher2::report(self).clone()
    }
    fn rules_changed(&self) -> bool {
        arko_index::watch2::Watcher2::rules_changed(self)
    }
}

/// Applies deltas until the watcher is quiet, then checks the model against a fresh scan.
fn settle(w: &mut dyn Watch, model: &mut Model, root: &Path, rules: &FileRules) {
    while let Some(d) = w.next(Duration::from_millis(800)) {
        apply(model, &d);
    }
    assert_eq!(model_paths(model), model_paths(&fresh(root, rules)));
    assert_eq!(*model, fresh(root, rules));
}

fn rule_changes_scenario(open: &dyn Fn(&Path, &Path, &FileRules) -> Box<dyn Watch>) {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".gitignore", "*.log\n");
    write(r, "a.rs", "fn a() {}\n");
    write(r, "b.rs", "fn b() {}\n");
    write(r, "x.log", "log line\n");
    write(r, "sub/c.rs", "fn c() {}\n");
    write(r, "cache/CACHEDIR.TAG", TAG);
    write(r, "cache/d.rs", "fn cached() {}\n");
    write(r, "target/e.rs", "fn e() {}\n");
    write(r, "target/deep/g.rs", "fn g() {}\n");
    write(r, ".git/info/exclude", "# git's template\n");
    let rules = respect();
    let mut w = open(r, state.path(), &rules);
    let mut model = Model::new();
    settle(&mut *w, &mut model, r, &rules);
    assert_eq!(
        model_paths(&model),
        set(&[
            ".gitignore",
            "a.rs",
            "b.rs",
            "sub/c.rs",
            "target/deep/g.rs",
            "target/e.rs"
        ])
    );
    assert!(skipped(&w.report()).contains(&("cache/".to_string(), SkipReason::CachedirTag)));

    // Editing ignore files adds and removes files (root and nested).
    write(r, ".gitignore", "*.rs\n");
    settle(&mut *w, &mut model, r, &rules);
    assert_eq!(model_paths(&model), set(&[".gitignore", "x.log"]));
    write(r, ".gitignore", "");
    settle(&mut *w, &mut model, r, &rules);
    assert!(model_paths(&model).contains("b.rs"));
    write(r, "sub/.gitignore", "c.rs\n");
    settle(&mut *w, &mut model, r, &rules);
    assert!(!model_paths(&model).contains("sub/c.rs"));
    fs::remove_file(r.join("sub/.gitignore")).unwrap();
    settle(&mut *w, &mut model, r, &rules);
    assert!(model_paths(&model).contains("sub/c.rs"));
    // So does editing .git/info/exclude.
    write(r, ".git/info/exclude", "b.rs\n");
    settle(&mut *w, &mut model, r, &rules);
    assert!(!model_paths(&model).contains("b.rs"));
    assert_eq!(w.report().ignored_by_git_info_exclude, 1);
    write(r, ".git/info/exclude", "");
    settle(&mut *w, &mut model, r, &rules);
    assert!(model_paths(&model).contains("b.rs"));

    // Removing the tag of a directory excluded since startup brings its files back;
    // tagging a directory removes them.
    fs::remove_file(r.join("cache/CACHEDIR.TAG")).unwrap();
    settle(&mut *w, &mut model, r, &rules);
    assert!(model_paths(&model).contains("cache/d.rs"));
    write(r, "target/CACHEDIR.TAG", TAG);
    settle(&mut *w, &mut model, r, &rules);
    assert!(!model_paths(&model).contains("target/e.rs"));
    assert!(!model_paths(&model).contains("target/deep/g.rs"));
    assert!(skipped(&w.report()).contains(&("target/".to_string(), SkipReason::CachedirTag)));
    // Files written inside the tagged directory stay out.
    write(r, "target/f.rs", "fn f() {}\n");
    settle(&mut *w, &mut model, r, &rules);
    fs::remove_file(r.join("target/CACHEDIR.TAG")).unwrap();
    settle(&mut *w, &mut model, r, &rules);
    assert!(model_paths(&model).contains("target/f.rs"));

    // Skipped files appear in and leave the report as they change.
    fs::write(r.join("blob.bin"), b"\0\x01").unwrap();
    settle(&mut *w, &mut model, r, &rules);
    assert!(skipped(&w.report()).contains(&("blob.bin".to_string(), SkipReason::Binary)));
    fs::write(r.join("a.rs"), b"caf\xe9\n").unwrap();
    settle(&mut *w, &mut model, r, &rules);
    assert!(skipped(&w.report()).contains(&("a.rs".to_string(), SkipReason::NotUtf8)));
    assert!(!model_paths(&model).contains("a.rs"));
    fs::write(r.join("a.rs"), "fn a_again() {}\n").unwrap();
    fs::remove_file(r.join("blob.bin")).unwrap();
    settle(&mut *w, &mut model, r, &rules);
    assert!(
        w.report()
            .skipped
            .iter()
            .all(|s| s.reason.is_default_exclusion())
    );
    let saved: Report =
        serde_json::from_slice(&fs::read(state.path().join("report.json")).unwrap()).unwrap();
    assert_eq!(saved, w.report());
    write(r, ".gitignore", "*.log\n");
    settle(&mut *w, &mut model, r, &rules);
    assert!(!model_paths(&model).contains("x.log"));
    assert!(!w.rules_changed());
    drop(w);

    // Restart with different rules: the first delta reconciles to them.
    let mut other = FileRules::default();
    other.exclude = vec!["sub/".into()];
    let mut w = open(r, state.path(), &other);
    assert!(w.rules_changed());
    settle(&mut *w, &mut model, r, &other);
    assert!(model_paths(&model).contains("x.log")); // ignore files no longer honoured
    assert!(!model_paths(&model).contains("sub/c.rs"));
    drop(w);
    let mut w = open(r, state.path(), &other);
    assert!(!w.rules_changed());
    settle(&mut *w, &mut model, r, &other);
    drop(w);

    // Without the opt-in, an ignore file is just a file.
    let defaults = FileRules::default();
    let mut w = open(r, state.path(), &defaults);
    settle(&mut *w, &mut model, r, &defaults);
    write(r, ".gitignore", "*\n");
    settle(&mut *w, &mut model, r, &defaults);
    assert!(model_paths(&model).contains("b.rs"));
}

#[test]
fn polling_watcher_follows_rule_changes() {
    rule_changes_scenario(&|r, s, rules| {
        Box::new(open_watch_with(r, s, &Options::default(), rules).unwrap())
    });
}

#[cfg(target_os = "linux")]
#[test]
fn inotify_watcher_follows_rule_changes() {
    rule_changes_scenario(&|r, s, rules| {
        Box::new(arko_index::watch2::Watcher2::open_with(r, s, &Options::default(), rules).unwrap())
    });
}

#[test]
fn cli_flags_and_report() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    write(r, ".gitignore", "*.log\n");
    write(r, "a.rs", "fn a() {}\n");
    write(r, "x.log", "x\n");
    write(r, "vendor/v.rs", "fn v() {}\n");
    fs::write(r.join("latin1.txt"), b"\xe9\n").unwrap();
    let out_dir = tempfile::tempdir().unwrap();
    let report_path = out_dir.path().join("report.json");
    let out = std::process::Command::new(cli())
        .arg("chunk")
        .arg("--respect-ignore")
        .arg("--exclude=vendor/")
        .arg("--report")
        .arg(&report_path)
        .arg(r)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let got: BTreeSet<String> = stdout
        .lines()
        .map(|l| serde_json::from_str::<Chunk>(l).unwrap().path)
        .collect();
    assert_eq!(got, set(&[".gitignore", "a.rs"]));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("2 files indexed, 1 skipped (1 not-utf8), 1 excluded by pattern"));
    assert!(stderr.contains("skipped (not-utf8): latin1.txt"));
    let saved: Report = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(saved.files, 2);
    assert!(saved.rules.respect_ignore);
    // Only built-in exclusions (.git): stderr stays quiet; --report still has them.
    let quiet = tempfile::tempdir().unwrap();
    write(quiet.path(), ".git/HEAD", "ref: refs/heads/main\n");
    write(quiet.path(), "a.rs", "fn a() {}\n");
    let out = std::process::Command::new(cli())
        .arg("chunk")
        .arg("--report")
        .arg(&report_path)
        .arg(quiet.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stderr), "");
    let saved: Report = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(saved.counts.get(&SkipReason::GitDir), Some(&1));
    let bad = std::process::Command::new(cli())
        .args(["chunk", "--bogus"])
        .arg(r)
        .output()
        .unwrap();
    assert!(!bad.status.success());
}
