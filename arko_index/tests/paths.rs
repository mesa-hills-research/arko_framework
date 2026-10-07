//! Source path identity must be injective: filenames are data, not escape sequences.
#![cfg(unix)]
use arko_index::{bench_v1::*, search};
use std::{fs, path::Path, time::Duration};

#[cfg(unix)]
#[test]
fn literal_backslash_does_not_alias_a_directory_separator() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("a")).unwrap();
    fs::write(root.path().join("a/b.rs"), "fn nested() {}\n").unwrap();
    fs::write(root.path().join(r"a\b.rs"), "fn literal() {}\n").unwrap();
    let chunks = chunk_dir(root.path(), &Options::default()).unwrap();
    assert_eq!(
        chunks.len(),
        2,
        "distinct source files must not overwrite in traversal"
    );
    assert!(chunks.iter().any(|c| c.path == "a/b.rs"));
    assert!(chunks.iter().any(|c| c.path == r"a\b.rs"));
    assert_ne!(chunks[0].id, chunks[1].id);
    for c in &chunks {
        let text = fs::read_to_string(root.path().join(&c.path)).unwrap();
        assert_eq!(c.end_byte, text.len());
        assert_eq!(
            chunk_str(Path::new(&c.path), &text, &Options::default()).unwrap(),
            vec![c.clone()]
        );
    }
    let file_chunks = chunk_file(&root.path().join(r"a\b.rs"), &Options::default()).unwrap();
    assert_eq!(file_chunks[0].path, r"a\b.rs");
    assert_eq!(
        search::index(root.path(), state.path(), &Options::default()).unwrap(),
        2
    );
    assert_eq!(
        search::search(state.path(), "literal").unwrap()[0].path,
        r"a\b.rs"
    );
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    let initial = w.next_change(Duration::ZERO).unwrap().unwrap();
    w.ack(initial.sequence).unwrap();
    fs::remove_file(root.path().join(r"a\b.rs")).unwrap();
    let d = w.next_change(Duration::from_secs(2)).unwrap().unwrap();
    assert_eq!(d.removed.len(), 1);
    assert!(d.added.is_empty() && d.changed.is_empty());
    assert_eq!(w.snapshot().unwrap()[0].path, "a/b.rs");
    drop(w);
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    assert_eq!(
        w.next_change(Duration::ZERO).unwrap().unwrap().sequence,
        d.sequence
    );
    assert_eq!(w.snapshot().unwrap().len(), 1);
}

#[cfg(unix)]
#[test]
fn undecodable_paths_are_skipped_or_rejected_instead_of_lossy_collisions() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let root = tempfile::tempdir().unwrap();
    let invalid = root.path().join(OsString::from_vec(b"a\xff.rs".to_vec()));
    // APFS (macOS) refuses names that aren't UTF-8, so the case can't arise there.
    if let Err(error) = fs::write(&invalid, "fn invalid_name() {}\n") {
        assert_eq!(error.raw_os_error(), Some(libc::EILSEQ), "{error}");
        eprintln!("skipped: this file system refuses non-UTF-8 names ({error})");
        return;
    }
    fs::write(
        root.path().join("a�.rs"),
        "fn real_replacement_character() {}\n",
    )
    .unwrap();
    // A directory walk skips the undecodable name and reports it; the real U+FFFD
    // name is indexed under its own path, so nothing aliases.
    let (chunks, report) =
        chunk_dir_with(root.path(), &Options::default(), &FileRules::default()).unwrap();
    let paths: Vec<_> = chunks.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(paths, ["a\u{FFFD}.rs"]);
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].path, "a\\xFF.rs");
    assert_eq!(report.skipped[0].reason, SkipReason::NameNotUtf8);
    // An explicitly named file or path is still an error.
    assert!(chunk_file(&invalid, &Options::default()).is_err());
    assert!(chunk_str(&invalid, "fn a() {}", &Options::default()).is_err());
}
