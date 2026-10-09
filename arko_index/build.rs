//! Build script: records the locked tree-sitter crates (`mesquite` and its
//! grammar crates) for the watcher's engine id (`src/watch.rs`), so a grammar upgrade
//! invalidates cached partitions.
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    engine_lock(&manifest_dir, &out_dir);
}

/// Writes the `[[package]]` entries of the tree-sitter crates from the nearest
/// `Cargo.lock` (the workspace's, or the package's own when built from a packaged crate).
/// A git or registry dependency's entry names its exact source; a path dependency's
/// entry has only its name, version and dependencies.
fn engine_lock(manifest_dir: &Path, out_dir: &Path) {
    let lock = manifest_dir
        .ancestors()
        .map(|dir| dir.join("Cargo.lock"))
        .find(|path| path.is_file());
    let mut entries = String::new();
    if let Some(path) = &lock {
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(path).unwrap_or_default();
        for entry in text.split("[[package]]").skip(1) {
            if entry.trim_start().starts_with("name = \"mesquite") {
                entries.push_str("[[package]]");
                entries.push_str(entry.trim_end());
                entries.push('\n');
            }
        }
    }
    fs::write(out_dir.join("engine_lock.txt"), entries).unwrap();
}
