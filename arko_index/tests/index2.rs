use arko_index::{bench_v1::Options, index2, search};
use std::{fs, path::Path};

fn generation_files(state: &Path) -> Vec<(String, Vec<u8>)> {
    let generation = fs::read_to_string(state.join("CURRENT2")).unwrap();
    let dir = state.join(generation.trim());
    [
        "meta.json",
        "lens.dat",
        "docs.off",
        "docs.dat",
        "terms.dat",
        "terms.idx",
        "post.dat",
    ]
    .iter()
    .map(|f| (f.to_string(), fs::read(dir.join(f)).unwrap()))
    .collect()
}

#[test]
fn sharded_build_is_identical_and_ranks_like_v1() {
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    fs::create_dir_all(r.join("src/net")).unwrap();
    fs::write(
        r.join("src/lib.rs"),
        "fn retriesExponentialBackoff() { retry(); }\nfn other_thing() { banana(); }\n\
         struct HTTPServer { port: u16 }\nimpl HTTPServer { fn start(&self) { listen(); } }\n",
    )
    .unwrap();
    fs::write(
        r.join("src/net/tcp.c"),
        "static int tcp_cubic_init(struct sock *sk) { return 0; }\n\
         int kmalloc_wrapper(void) { return GFP_KERNEL; }\n",
    )
    .unwrap();
    fs::write(
        r.join("src/über_ΣΑΣ.py"),
        "def straße_ΣΑΣ(x):\n    return x  # café ΌΣΟΣ\n\nclass XMLHttpRequest:\n    pass\n",
    )
    .unwrap();
    for i in 0..40 {
        fs::write(
            r.join(format!("src/gen_{i}.rs")),
            format!(
                "fn handler_{i}() {{ retry_backoff({i}); }}\nfn the_int_{i}() -> i32 {{ {i} }}\n"
            ),
        )
        .unwrap();
    }
    let options = Options::default();
    let v1 = tempfile::tempdir().unwrap();
    search::index(r, v1.path(), &options).unwrap();
    let mut builds = Vec::new();
    // One inline shard with a large budget, and several shards that spill many runs.
    for (threads, budget) in [(1, 256 << 20), (3, 1 << 10), (5, 64)] {
        let state = tempfile::tempdir().unwrap();
        index2::index(r, state.path(), &options, threads, budget).unwrap();
        builds.push(state);
    }
    let reference = generation_files(builds[0].path());
    for b in &builds[1..] {
        assert!(generation_files(b.path()) == reference);
    }
    for q in [
        "exponential backoff retry",
        "HTTP server start",
        "tcp cubic",
        "straße σας",
        "the int",
        "handler",
        "zzzz_no_such_term",
    ] {
        let a = search::search(v1.path(), q).unwrap();
        for b in &builds {
            assert_eq!(index2::search(b.path(), q).unwrap(), a, "{q}");
        }
    }
}
