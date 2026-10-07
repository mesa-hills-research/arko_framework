use arko_index::{bench_v1::*, search};
use fs2::FileExt;
use std::{
    fs::{self, OpenOptions},
    time::Duration,
};

#[test]
fn index_rejects_competing_writer_without_replacing_snapshot() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    fs::write(root.path().join("source.rs"), "fn old_snapshot() {}\n").unwrap();
    search::index(root.path(), state.path(), &Options::default()).unwrap();
    let previous = fs::read(state.path().join("index.json")).unwrap();
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(state.path().join(".index.lock"))
        .unwrap();
    lock.lock_exclusive().unwrap();
    fs::write(root.path().join("source.rs"), "fn new_snapshot() {}\n").unwrap();
    assert!(search::index(root.path(), state.path(), &Options::default()).is_err());
    assert_eq!(previous, fs::read(state.path().join("index.json")).unwrap());
    assert!(
        !search::search(state.path(), "old_snapshot")
            .unwrap()
            .is_empty()
    );
    drop(lock);
    search::index(root.path(), state.path(), &Options::default()).unwrap();
    assert!(
        !search::search(state.path(), "new_snapshot")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn index_rejects_ancestor_state_instead_of_silently_dropping_all_sources() {
    let state = tempfile::tempdir().unwrap();
    let root = state.path().join("repo");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.py"), "print('included')\n").unwrap();
    assert!(search::index(&root, state.path(), &Options::default()).is_err());
    assert!(!state.path().join(".arko-index-state").exists());
    assert!(!state.path().join("index.json").exists());
}

#[test]
fn failed_ack_and_reconciliation_leave_memory_and_journal_unchanged() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let source = root.path().join("a.rs");
    fs::write(&source, "fn first() {}\nfn stable() {}\n").unwrap();
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    let first = w.next_change(Duration::ZERO).unwrap().unwrap();
    let snapshot = w.snapshot().unwrap();
    let journal = fs::read(state.path().join("watch.json")).unwrap();
    // Deterministic filesystem I/O failure, including on privileged test runners.
    let blocked = state
        .path()
        .join(format!(".watch.json.{}.tmp", std::process::id()));
    fs::create_dir(&blocked).unwrap();
    assert!(w.ack(first.sequence).is_err());
    fs::write(&source, "fn first() { changed(); }\nfn stable() {}\n").unwrap();
    assert!(w.next_change(Duration::from_secs(1)).is_err());
    assert_eq!(snapshot, w.snapshot().unwrap());
    assert_eq!(journal, fs::read(state.path().join("watch.json")).unwrap());
    fs::remove_dir(&blocked).unwrap();
    // Failed observation wasn't consumed; it produces exactly the next sequence.
    let next = w.next_change(Duration::from_secs(1)).unwrap().unwrap();
    assert_eq!(next.sequence, first.sequence + 1);
    assert_eq!(next.added.len(), 1);
    assert_eq!(next.removed.len(), 1);
    assert_eq!(next.changed.len(), 1);
    assert_eq!(snapshot[1].id, next.changed[0].id);
    let latest = w.snapshot().unwrap();
    drop(w);
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    assert_eq!(
        w.next_change(Duration::ZERO).unwrap().unwrap().sequence,
        first.sequence
    );
    assert_eq!(
        w.next_change(Duration::ZERO).unwrap().unwrap().sequence,
        next.sequence
    );
    assert_eq!(w.snapshot().unwrap(), latest);
    w.ack(next.sequence).unwrap();
    drop(w);
    let mut w = open_watch(root.path(), state.path(), &Options::default()).unwrap();
    assert!(w.next_change(Duration::ZERO).unwrap().is_none());
}
