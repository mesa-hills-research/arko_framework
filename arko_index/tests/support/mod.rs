//! Helpers the integration tests share.

use std::{path::PathBuf, process::Command, sync::OnceLock};

/// The `arko_index_cli` example. Cargo builds it, once per test binary, and reports
/// where it put the executable, since the layout of its build directory changes between
/// releases.
pub fn cli() -> PathBuf {
    static CLI: OnceLock<PathBuf> = OnceLock::new();
    CLI.get_or_init(build_cli).clone()
}

/// The variables Cargo sets for a test, besides `CARGO_PKG_*` and `CARGO_BIN_EXE_*`.
const SET_FOR_TESTS: [&str; 9] = [
    "CARGO",
    "CARGO_MANIFEST_DIR",
    "CARGO_MANIFEST_PATH",
    "CARGO_CRATE_NAME",
    "CARGO_BIN_NAME",
    "CARGO_PRIMARY_PACKAGE",
    "CARGO_TARGET_TMPDIR",
    "CARGO_RUSTC_CURRENT_DIR",
    "OUT_DIR",
];

/// Builds the example as `cargo test` did, so the build is usually done already. It
/// takes this test's features, the release profile when debug assertions are off, and
/// the environment without the variables Cargo set for this test, since build scripts
/// that watch them would run again.
fn build_cli() -> PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let mut cargo = Command::new(env!("CARGO"));
    cargo
        .args(["build", "--locked", "--example", "arko_index_cli"])
        .args(["--message-format", "json-render-diagnostics"])
        .arg("--manifest-path")
        .arg(manifest);
    if !cfg!(debug_assertions) {
        cargo.arg("--release");
    }
    let features = [
        ("local-embed", cfg!(feature = "local-embed")),
        ("local-embed-gpu", cfg!(feature = "local-embed-gpu")),
        ("local-embed-cuda", cfg!(feature = "local-embed-cuda")),
    ];
    for (feature, _) in features.iter().filter(|(_, on)| *on) {
        cargo.args(["--features", feature]);
    }
    for (name, _) in std::env::vars_os() {
        let text = name.to_string_lossy();
        if text.starts_with("CARGO_PKG_")
            || text.starts_with("CARGO_BIN_EXE_")
            || SET_FOR_TESTS.contains(&&*text)
        {
            cargo.env_remove(&name);
        }
    }
    let out = cargo.output().expect("cargo didn't start");
    let log = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "building arko_index_cli failed:\n{log}"
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact"
                && message["target"]["name"] == "arko_index_cli"
        })
        .find_map(|message| message["executable"].as_str().map(PathBuf::from))
        .unwrap_or_else(|| panic!("cargo reported no arko_index_cli executable:\n{log}"))
}
