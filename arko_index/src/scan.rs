//! Scaling prototype: measures the change-detection primitives a watcher pass needs.
use crate::{bench_v1::Error, files, files::FileRules, par};
use std::{os::unix::fs::MetadataExt, path::Path, time::Instant};

/// Prints one JSON line of timings for: the current full `read_dir` pass (read, UTF-8
/// check, copy, SHA-256 of every file, what each watcher poll does), a walk that only
/// lists files, a metadata-only pass (lstat: size, mtime, ctime, inode), and the same
/// read+hash and metadata passes with `threads` workers.
pub fn bench_scan(root: &Path, threads: usize) -> Result<(), Error> {
    bench_scan_with(root, threads, &FileRules::default())
}
/// [`bench_scan`] with explicit file selection rules.
pub fn bench_scan_with(root: &Path, threads: usize, rules: &FileRules) -> Result<(), Error> {
    let t = Instant::now();
    let (full, report) = files::read_dir(root, None, rules)?;
    let read_dir_s = t.elapsed().as_secs_f64();
    let text_bytes: usize = full.values().map(|f| f.text.len()).sum();
    let text_files = full.len();
    drop(full);

    let t = Instant::now();
    let listed = files::list_files(root, None, rules)?.files;
    let walk_s = t.elapsed().as_secs_f64();

    let stat = |item: &files::Listed| -> Result<(u64, i64, i64, i64, i64, u64), Error> {
        let m = std::fs::symlink_metadata(&item.path)?;
        Ok((
            m.size(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
            m.ino(),
        ))
    };
    let t = Instant::now();
    let mut n = 0usize;
    par::ordered_map(&listed, 1, 1, stat, |_| {
        n += 1;
        Ok(())
    })?;
    let stat_1_s = t.elapsed().as_secs_f64();
    // Tiny work items are claimed in batches of 256: per-item claiming contends on the
    // claim lock and is slower than one thread (measured).
    let batches: Vec<&[files::Listed]> = listed.chunks(256).collect();
    let t = Instant::now();
    par::ordered_map(
        &batches,
        threads,
        threads * 4,
        |b| b.iter().map(stat).collect::<Result<Vec<_>, _>>(),
        |_| Ok(()),
    )?;
    let stat_n_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut hashed = 0usize;
    let small: Vec<&[files::Listed]> = listed.chunks(16).collect();
    par::ordered_map(
        &small,
        threads,
        threads * 8,
        |b| {
            Ok(b.iter()
                .map(|item| match files::read_listed(item, true) {
                    files::Read::Text(_, d) => Some(d.hash),
                    _ => None,
                })
                .collect::<Vec<_>>())
        },
        |r| {
            hashed += r.iter().filter(|h| h.is_some()).count();
            Ok(())
        },
    )?;
    let read_hash_n_s = t.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({
            "files_listed": listed.len(), "text_files": text_files, "text_bytes": text_bytes,
            "threads": threads,
            "read_dir_1thread_s": read_dir_s, "walk_only_s": walk_s,
            "lstat_1thread_s": stat_1_s, "lstat_parallel_s": stat_n_s,
            "read_hash_parallel_s": read_hash_n_s, "hashed_files": hashed, "stat_count": n,
            "skipped": report.counts
        })
    );
    Ok(())
}
