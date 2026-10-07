//! Records the locked tree-sitter crate versions for the watcher's engine id
//! (`src/watch.rs`), so a grammar upgrade invalidates cached partitions.
use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    // The workspace's lock file, or the package's own when built from a packaged crate.
    let lock = manifest_dir
        .ancestors()
        .map(|dir| dir.join("Cargo.lock"))
        .find(|path| path.is_file());
    let mut entries = String::new();
    if let Some(path) = &lock {
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(path).unwrap_or_default();
        for entry in text.split("[[package]]").skip(1) {
            if entry.trim_start().starts_with("name = \"tree-sitter") {
                entries.push_str("[[package]]");
                entries.push_str(entry.trim_end());
                entries.push('\n');
            }
        }
    }
    fs::write(out_dir.join("engine_lock.txt"), entries).unwrap();
}
