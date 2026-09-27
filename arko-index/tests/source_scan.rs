use arko_index::{BuildSpec, source::scan};

#[test]
fn authored_docs_in_build_directories_are_indexed_unless_explicitly_ignored() {
    let root = tempfile::tempdir().unwrap();
    let docs = root.path().join("workflows/build");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("index.mdx"), "# Build a workflow\n\nInstructions.\n").unwrap();

    let spec = BuildSpec::new("docs", root.path());
    let manifest = scan(&spec).unwrap();
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(manifest.files[0].path, "workflows/build/index.mdx");

    std::fs::write(root.path().join(".gitignore"), "workflows/build/\n").unwrap();
    assert!(scan(&spec).unwrap().files.is_empty());
}
