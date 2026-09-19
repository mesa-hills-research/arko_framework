//! Durable source snapshots, validated boundaries, and local hybrid retrieval.
pub mod backend;
pub mod chunker;
pub mod config;
pub mod error;
mod index;
pub mod packing;
mod scheduler;
mod search;
pub mod source;
mod storage;
mod watcher;
pub use config::*;
pub use error::{Error, Result};
pub use index::*;
pub use search::*;
pub use watcher::{FileChange, WatchHandle, coalesce};
