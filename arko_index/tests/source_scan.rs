use arko_index::{
    BuildSpec, DEFAULT_SKIPPED_DIRECTORIES,
    source::{TOO_LARGE_TO_READ, scan},
};

#[test]
fn authored_docs_in_build_directories_are_indexed_unless_explicitly_ignored() {
    let root = tempfile::tempdir().unwrap();
    let docs = root.path().join("workflows/build");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("index.mdx"),
        "# Build a workflow\n\nInstructions.\n",
    )
    .unwrap();

    let spec = BuildSpec::new("docs", root.path());
    let manifest = scan(&spec).unwrap();
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(manifest.files[0].path, "workflows/build/index.mdx");

    std::fs::write(root.path().join(".gitignore"), "workflows/build/\n").unwrap();
    assert!(scan(&spec).unwrap().files.is_empty());
}

#[test]
fn skipped_directory_names_are_configurable() {
    let root = tempfile::tempdir().unwrap();
    for dir in ["docs/dist", "node_modules/pkg", "src"] {
        std::fs::create_dir_all(root.path().join(dir)).unwrap();
        std::fs::write(root.path().join(dir).join("notes.md"), "# Notes\n").unwrap();
    }
    let paths = |spec: &BuildSpec| -> Vec<String> {
        scan(spec)
            .unwrap()
            .files
            .into_iter()
            .map(|f| f.path)
            .collect()
    };
    // The default names are skipped at any depth without an ignore file.
    let spec = BuildSpec::new("docs", root.path());
    assert_eq!(paths(&spec), ["src/notes.md"]);
    let spec = spec.with_skip_directories(["node_modules"]);
    assert_eq!(paths(&spec), ["docs/dist/notes.md", "src/notes.md"]);
    let none = BuildSpec::new("docs", root.path()).with_skip_directories(Vec::<String>::new());
    assert_eq!(paths(&none).len(), 3);
    // Specs saved before the setting existed keep the default list.
    let saved: BuildSpec = serde_json::from_value(serde_json::json!({
        "name": "docs", "root": root.path(), "include": [], "exclude": []
    }))
    .unwrap();
    assert_eq!(saved.skip_directories, DEFAULT_SKIPPED_DIRECTORIES);
    assert_eq!(paths(&saved), ["src/notes.md"]);
}

#[test]
fn files_over_the_size_ceiling_are_excluded_unread() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("big.json"), "[1]\n".repeat(1_000)).unwrap();
    std::fs::write(root.path().join("small.json"), "[1]\n").unwrap();
    let spec = BuildSpec::new("data", root.path()).with_max_file_bytes(1_000);
    let manifest = scan(&spec).unwrap();
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(manifest.files[0].path, "small.json");
    assert_eq!(manifest.excluded.len(), 1);
    assert_eq!(manifest.excluded[0].path, "big.json");
    assert_eq!(manifest.excluded[0].reason, TOO_LARGE_TO_READ);
    // The default ceiling admits it.
    assert_eq!(
        scan(&BuildSpec::new("data", root.path()))
            .unwrap()
            .files
            .len(),
        2
    );
}

#[cfg(unix)]
#[test]
fn unreadable_entries_are_excluded_and_the_scan_continues() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("ok.md"), "# Ok\n").unwrap();
    std::fs::create_dir(root.path().join("locked")).unwrap();
    std::fs::write(root.path().join("locked/inner.md"), "# Inner\n").unwrap();
    std::fs::write(root.path().join("secret.md"), "# Secret\n").unwrap();
    let mode = |path: &str, mode: u32| {
        std::fs::set_permissions(
            root.path().join(path),
            std::fs::Permissions::from_mode(mode),
        )
        .unwrap()
    };
    mode("locked", 0o000);
    mode("secret.md", 0o000);
    // Permissions don't bind a privileged user; nothing to test then.
    if std::fs::read(root.path().join("secret.md")).is_ok() {
        mode("locked", 0o755);
        return;
    }
    let manifest = scan(&BuildSpec::new("docs", root.path()));
    mode("locked", 0o755);
    mode("secret.md", 0o644);
    let manifest = manifest.unwrap();
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(manifest.files[0].path, "ok.md");
    let excluded: Vec<_> = manifest
        .excluded
        .iter()
        .map(|e| (e.path.as_str(), e.reason.as_str()))
        .collect();
    assert_eq!(
        excluded,
        [
            ("locked", "unreadable: permission denied"),
            ("secret.md", "unreadable: permission denied")
        ]
    );
}

#[cfg(target_os = "linux")]
#[test]
fn names_that_are_not_utf8_are_excluded_with_a_reason() {
    use std::os::unix::ffi::OsStrExt;
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("good.md"), "# Good\n").unwrap();
    std::fs::write(
        root.path().join(std::ffi::OsStr::from_bytes(b"bad\xff.md")),
        "# Bad\n",
    )
    .unwrap();
    // Names that would be skipped anyway are not reported for their encoding.
    std::fs::write(
        root.path()
            .join(std::ffi::OsStr::from_bytes(b"other\xff.bin")),
        "data",
    )
    .unwrap();
    let manifest = scan(&BuildSpec::new("docs", root.path())).unwrap();
    assert_eq!(manifest.files.len(), 1);
    let excluded: Vec<_> = manifest
        .excluded
        .iter()
        .map(|e| (e.path.as_str(), e.reason.as_str()))
        .collect();
    assert_eq!(
        excluded,
        [
            ("bad\u{fffd}.md", "path is not valid UTF-8"),
            ("other\u{fffd}.bin", "unsupported extension")
        ]
    );
}
