//! Syntax-aware, exact source partitioning, offline BM25 search, optional dense
//! (embedding) and hybrid search, and durable watchers. Chunking and the polling
//! watcher are in [`bench_v1`], the indexes in [`search`] and [`index2`], embeddings in
//! [`dense`], vector storage in [`vector`], lexical, dense and hybrid search in
//! [`hybrid`], and the inotify watcher in `watch2` (Linux). See
//! `docs/arko_index-reference.md` for the rules and trade-offs.
pub mod bench_v1;
pub mod dense;
mod files;
mod front_matter;
mod fsio;
mod headers;
pub mod hybrid;
mod languages;
pub mod search;
mod syntax;
pub mod vector;
mod watch;

pub mod index2;
mod normalize;
mod par;
mod rules;
pub use par::threads;
#[cfg(unix)]
mod scan;
mod tokenize;
#[cfg(target_os = "linux")]
pub mod watch2;
pub use files::{FileRules, Report, SkipReason, Skipped};
#[cfg(unix)]
pub use scan::{bench_scan, bench_scan_with};
