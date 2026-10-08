//! Scaling prototype: an order-preserving, bounded-window parallel map.
//!
//! Workers claim item indices from a shared counter, but never more than `window`
//! items ahead of the consumer, so at most `window` results wait in the reorder
//! buffer (bounded memory even when one huge file holds up the head). The sink
//! sees results strictly in input order, so output is identical to a sequential
//! loop, including which error is reported first.
use crate::bench_v1::Error;
use std::{
    collections::BTreeMap,
    sync::{Condvar, Mutex, mpsc},
};

/// Worker count: `ARKO_INDEX_THREADS` if set (>= 1), else the CPUs this process may
/// use (`available_parallelism` honours affinity masks and cgroup CPU quotas).
pub fn threads() -> usize {
    std::env::var("ARKO_INDEX_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
}

/// Bytes of source that may be read but not yet handed to the sink (being processed,
/// or finished and waiting for an earlier item): `ARKO_INDEX_WINDOW_MB` if set, else
/// 32 MB per thread. A byte budget instead of an item count lets workers run far
/// ahead past one huge file at the head while still bounding memory.
pub(crate) fn window_bytes(threads: usize) -> usize {
    std::env::var("ARKO_INDEX_WINDOW_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .map_or(threads.max(1).saturating_mul(32 << 20), |mb| mb << 20)
}

/// Window weights for listed files: size from a parallel `lstat` pass (batches of
/// 256: per-item claiming of such tiny work is slower than one thread) plus 4 KB, so
/// many empty files still count. Only a memory heuristic: a file that changes or
/// vanishes before it is read just has a stale weight.
pub(crate) fn source_weights(listed: &[crate::files::Listed], threads: usize) -> Vec<usize> {
    let mut weights = Vec::with_capacity(listed.len());
    if threads <= 1 {
        return weights; // the sequential path never consults weights
    }
    let batches: Vec<&[crate::files::Listed]> = listed.chunks(256).collect();
    let _ = ordered_map(
        &batches,
        threads,
        threads * 4,
        |b| {
            Ok(b.iter()
                .map(|i| std::fs::symlink_metadata(&i.path).map_or(0, |m| m.len() as usize) + 4096)
                .collect::<Vec<_>>())
        },
        |w| {
            weights.extend(w);
            Ok(())
        },
    );
    weights
}

struct Claim {
    next: usize,
    /// Total weight of claimed items not yet emitted.
    outstanding: usize,
    abort: bool,
}

/// `ordered_map` with a window of `window` items.
pub fn ordered_map<T, R, F, S>(
    items: &[T],
    threads: usize,
    window: usize,
    f: F,
    sink: S,
) -> Result<(), Error>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R, Error> + Sync,
    S: FnMut(R) -> Result<(), Error>,
{
    let window = window.max(threads);
    ordered_map_weighted(items, threads, window, |_| 1, f, sink)
}

/// Maps `items` with `threads` workers and hands the results to `sink` strictly in
/// input order. A worker claims the next item only while the claimed-but-unemitted
/// items weigh at most `budget` (or nothing is outstanding, so an item heavier than
/// the budget still runs, alone). The head item is always claimable eventually, so
/// the pipeline cannot deadlock. A panic in `f` or `sink` becomes that item's error.
pub fn ordered_map_weighted<T, R, W, F, S>(
    items: &[T],
    threads: usize,
    budget: usize,
    weight: W,
    f: F,
    mut sink: S,
) -> Result<(), Error>
where
    T: Sync,
    R: Send,
    W: Fn(usize) -> usize + Sync,
    F: Fn(&T) -> Result<R, Error> + Sync,
    S: FnMut(R) -> Result<(), Error>,
{
    let threads = threads.max(1).min(items.len().max(1));
    if threads == 1 {
        for item in items {
            let r = unwound(|| f(item))?;
            unwound(|| sink(r))?;
        }
        return Ok(());
    }
    let claim = Mutex::new(Claim {
        next: 0,
        outstanding: 0,
        abort: false,
    });
    let cv = Condvar::new();
    let (tx, rx) = mpsc::channel::<(usize, Result<R, Error>)>();
    let mut outcome = Ok(());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (claim, cv, f, weight) = (&claim, &cv, &f, &weight);
            scope.spawn(move || {
                loop {
                    let idx = {
                        let mut c = claim.lock().unwrap();
                        while !c.abort
                            && c.next < items.len()
                            && c.outstanding > 0
                            && c.outstanding.saturating_add(weight(c.next)) > budget
                        {
                            c = cv.wait(c).unwrap();
                        }
                        if c.abort || c.next >= items.len() {
                            return;
                        }
                        c.outstanding += weight(c.next);
                        c.next += 1;
                        c.next - 1
                    };
                    if tx.send((idx, unwound(|| f(&items[idx])))).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, Result<R, Error>> = BTreeMap::new();
        let mut head = 0usize;
        'recv: for (idx, r) in rx.iter() {
            pending.insert(idx, r);
            while let Some(r) = pending.remove(&head) {
                let step = r.and_then(|r| unwound(|| sink(r)));
                {
                    let mut c = claim.lock().unwrap();
                    c.outstanding -= weight(head);
                    if step.is_err() {
                        c.abort = true;
                    }
                }
                head += 1;
                cv.notify_all();
                if let Err(e) = step {
                    outcome = Err(e);
                    break 'recv;
                }
            }
        }
        // Dropping the receiver makes any still-running worker stop at its next send.
    });
    outcome
}

/// Runs `f`, turning a panic into an error, so a lost item can never stall the
/// in-order pipeline (its successors would wait for it forever).
fn unwound<R>(f: impl FnOnce() -> Result<R, Error>) -> Result<R, Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|payload| Err(format!("panicked: {}", panic_message(&*payload)).into()))
}

/// The message of a panic's payload (`panic!` with a literal or a format string).
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_ends_the_walk_with_an_error() {
        let items: Vec<usize> = (0..200).collect();
        for threads in [1, 4] {
            // A budget of a few items: the workers wait for the panicked head item.
            let result = ordered_map_weighted(
                &items,
                threads,
                4,
                |_| 1,
                |&i| {
                    assert!(i != 3, "item {i}");
                    Ok(i)
                },
                |_| Ok(()),
            );
            assert_eq!(result.unwrap_err().to_string(), "panicked: item 3");
            let mut emitted = 0;
            let result = ordered_map_weighted(
                &items,
                threads,
                4,
                |_| 1,
                |&i| Ok(i),
                |i| {
                    assert!(i != 5, "sink {i}");
                    emitted += 1;
                    Ok(())
                },
            );
            assert_eq!(result.unwrap_err().to_string(), "panicked: sink 5");
            assert_eq!(emitted, 5);
        }
    }
}
