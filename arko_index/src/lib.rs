//! Syntax-aware, exact source partitioning, offline BM25 search and durable watchers.
//! Chunking and the polling watcher are in [`bench_v1`]; the indexes in [`search`] and
//! [`index2`]; the inotify watcher in `watch2` (Linux). See the README for the rules
//! and trade-offs.
pub mod bench_v1;
mod files;
pub mod search;
mod syntax;
mod watch;

pub mod index2;
mod normalize;
mod par;
pub use par::threads;
#[cfg(unix)]
mod scan;
mod tokenize;
#[cfg(target_os = "linux")]
pub mod watch2;
pub use files::{FileRules, Report, SkipReason, Skipped};
#[cfg(unix)]
pub use scan::{bench_scan, bench_scan_with};
