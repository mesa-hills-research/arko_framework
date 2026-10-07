//! Build script.
//!
//! 1. Records the locked tree-sitter crate versions for the watcher's engine id
//!    (`src/watch.rs`), so a grammar upgrade invalidates cached partitions.
//! 2. Feature `pgo`: compiles the tree-sitter grammars a second time with GCC
//!    profile-guided optimization, from the profiles in `pgo/` (written by
//!    `pgo/train.sh`), under renamed symbols (`arko_index_pgo_tree_sitter_*`), and sets
//!    `cfg(arko_index_pgo)` so `src/pgo.rs` parses with them. Only for optimized builds
//!    with GCC targeting x86-64 Linux on the same host, when the GCC major version, the
//!    grammar versions and the grammar sources match the profiles. Any other build is a
//!    normal one: it uses the grammar crates' own objects, silently when the compiler or
//!    target is simply not covered, with a cargo warning when GCC was expected to work.
//!    A profile that does not fit the sources makes GCC fail (coverage mismatch), which
//!    is caught and also falls back. `ARKO_INDEX_PGO_PROFILES=DIR` reads the profiles
//!    (`profile.txt`, `gcda/`) from DIR instead of `pgo/`. `ARKO_INDEX_PGO_TRAIN=DIR` builds instrumented
//!    grammars instead and writes the profile manifest and the object directory to DIR
//!    (used by `pgo/train.sh`).
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(arko_index_pgo)");
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    engine_lock(&manifest_dir, &out_dir);
    if env::var_os("CARGO_FEATURE_PGO").is_some() {
        println!("cargo:rerun-if-env-changed=ARKO_INDEX_PGO_TRAIN");
        println!("cargo:rerun-if-env-changed=ARKO_INDEX_PGO_PROFILES");
        match pgo(&manifest_dir, &out_dir) {
            Ok(()) => println!("cargo:rustc-cfg=arko_index_pgo"),
            Err(Skip::Silent) => {}
            Err(Skip::Warn(why)) => {
                println!("cargo:warning=arko_index: building without the PGO grammars: {why}")
            }
        }
    }
}

/// Writes the `[[package]]` entries of the tree-sitter crates from the nearest
/// `Cargo.lock` (the workspace's, or the package's own when built from a packaged crate).
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
            if entry.trim_start().starts_with("name = \"tree-sitter") {
                entries.push_str("[[package]]");
                entries.push_str(entry.trim_end());
                entries.push('\n');
            }
        }
    }
    fs::write(out_dir.join("engine_lock.txt"), entries).unwrap();
}

/// One grammar: its crate, the directory of its sources inside the crate, and the C
/// files the crate compiles. The crate defines `tree_sitter_<name>` and, with a
/// scanner, `tree_sitter_<name>_external_scanner_{create,destroy,scan,serialize,
/// deserialize}`; nothing else is global.
struct Grammar {
    name: &'static str,
    krate: &'static str,
    dir: &'static str,
    files: &'static [&'static str],
}

const BOTH: &[&str] = &["parser.c", "scanner.c"];
const PARSER: &[&str] = &["parser.c"];
const GRAMMARS: &[Grammar] = &[
    Grammar {
        name: "c",
        krate: "tree-sitter-c",
        dir: "src",
        files: PARSER,
    },
    Grammar {
        name: "cpp",
        krate: "tree-sitter-cpp",
        dir: "src",
        files: BOTH,
    },
    Grammar {
        name: "go",
        krate: "tree-sitter-go",
        dir: "src",
        files: PARSER,
    },
    Grammar {
        name: "java",
        krate: "tree-sitter-java",
        dir: "src",
        files: PARSER,
    },
    Grammar {
        name: "javascript",
        krate: "tree-sitter-javascript",
        dir: "src",
        files: BOTH,
    },
    Grammar {
        name: "python",
        krate: "tree-sitter-python",
        dir: "src",
        files: BOTH,
    },
    Grammar {
        name: "rust",
        krate: "tree-sitter-rust",
        dir: "src",
        files: BOTH,
    },
    Grammar {
        name: "typescript",
        krate: "tree-sitter-typescript",
        dir: "typescript/src",
        files: BOTH,
    },
    Grammar {
        name: "tsx",
        krate: "tree-sitter-typescript",
        dir: "tsx/src",
        files: BOTH,
    },
];
const SCANNER_FUNCTIONS: [&str; 5] = ["create", "destroy", "scan", "serialize", "deserialize"];
const TARGET: &str = "x86_64-unknown-linux-gnu";

enum Skip {
    /// The build is not one the profiles cover (not GCC, another target, unoptimized).
    Silent,
    /// GCC on the covered target, but the profiles could not be used.
    Warn(String),
}

impl From<String> for Skip {
    fn from(why: String) -> Self {
        Skip::Warn(why)
    }
}

/// One compiled translation unit: `<grammar>_<file stem>`, e.g. `rust_scanner`.
struct Unit<'a> {
    name: String,
    grammar: &'a Grammar,
    source: PathBuf,
}

fn pgo(manifest_dir: &Path, out_dir: &Path) -> Result<(), Skip> {
    let train = env::var_os("ARKO_INDEX_PGO_TRAIN").map(PathBuf::from);
    let target = env::var("TARGET").unwrap_or_default();
    if target != TARGET || env::var("HOST").ok().as_deref() != Some(TARGET) {
        return Err(Skip::Silent);
    }
    if !matches!(env::var("OPT_LEVEL").as_deref(), Ok("2" | "3")) && train.is_none() {
        return Err(Skip::Silent);
    }
    let Ok(compiler) = cc::Build::new().try_get_compiler() else {
        return Err(Skip::Silent);
    };
    if !compiler.is_like_gnu() {
        return Err(Skip::Silent);
    }
    let gcc_version = command_output(compiler.path(), &["-dumpfullversion"])
        .ok_or_else(|| "cannot run the C compiler for its version".to_string())?;
    let gcc_major = gcc_version.split('.').next().unwrap_or("").to_string();

    let profiles = env::var_os("ARKO_INDEX_PGO_PROFILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("pgo"));
    println!(
        "cargo:rerun-if-changed={}",
        profiles.join("profile.txt").display()
    );
    println!("cargo:rerun-if-changed={}", profiles.join("gcda").display());
    let pinned = pinned_versions(&manifest_dir.join("Cargo.toml"))?;
    let profile_text = fs::read_to_string(profiles.join("profile.txt")).unwrap_or_default();
    let profile = Profile::parse(&profile_text);
    if train.is_none() {
        if profile.gcc_major.is_empty() {
            return Err(format!(
                "{} is missing or empty",
                profiles.join("profile.txt").display()
            )
            .into());
        }
        if profile.gcc_major != gcc_major {
            return Err(format!(
                "the profiles are for GCC {}, the compiler is GCC {gcc_version}",
                profile.gcc_major
            )
            .into());
        }
        for (krate, version) in &pinned {
            if profile.crates.get(krate) != Some(version) {
                return Err(format!(
                    "the profiles were made for {krate} {}, the crate is {version} (rerun pgo/train.sh)",
                    profile.crates.get(krate).map_or("(none)", String::as_str)
                )
                .into());
            }
        }
    }

    let crate_dirs = locate_crates(&pinned)?;
    let mut units = Vec::new();
    for grammar in GRAMMARS {
        let dir = crate_dirs[grammar.krate].join(grammar.dir);
        for file in grammar.files {
            let stem = file.trim_end_matches(".c");
            units.push(Unit {
                name: format!("{}_{stem}", grammar.name),
                grammar,
                source: dir.join(file),
            });
        }
    }
    let mut hashes = Vec::new();
    for unit in &units {
        let bytes = fs::read(&unit.source)
            .map_err(|e| format!("cannot read {}: {e}", unit.source.display()))?;
        let hash = format!("{:016x}", fnv1a(&bytes));
        if train.is_none() && profile.units.get(&unit.name) != Some(&hash) {
            return Err(format!(
                "{} differs from the source the profiles were made with",
                unit.source.display()
            )
            .into());
        }
        hashes.push(hash);
    }

    let obj_dir = out_dir.join("pgo");
    let _ = fs::remove_dir_all(&obj_dir);
    fs::create_dir_all(&obj_dir)
        .map_err(|e| format!("cannot create {}: {e}", obj_dir.display()))?;
    let mode_flags: &[&str] = if train.is_some() {
        &["-fprofile-generate", "-fprofile-update=atomic"]
    } else {
        &[
            "-fprofile-use",
            "-fprofile-partial-training",
            "-Wno-missing-profile",
            "-Werror=coverage-mismatch",
        ]
    };
    if train.is_none() {
        for unit in &units {
            let from = profiles.join("gcda").join(format!("{}.gcda", unit.name));
            fs::copy(&from, obj_dir.join(format!("{}.gcda", unit.name)))
                .map_err(|e| format!("cannot copy {}: {e}", from.display()))?;
        }
    }
    let objects = compile_units(&compiler, &units, &obj_dir, mode_flags)?;
    // The guard in src/pgo.rs compares each grammar's node types with the crate's.
    for grammar in GRAMMARS {
        let from = crate_dirs[grammar.krate]
            .join(grammar.dir)
            .join("node-types.json");
        fs::copy(
            &from,
            obj_dir.join(format!("{}.node-types.json", grammar.name)),
        )
        .map_err(|e| format!("cannot copy {}: {e}", from.display()))?;
    }
    let mut archive = cc::Build::new();
    archive.cargo_warnings(false);
    for object in &objects {
        archive.object(object);
    }
    archive
        .try_compile("arko_index_pgo")
        .map_err(|e| format!("cannot archive the PGO objects: {e}"))?;

    if let Some(train) = &train {
        println!("cargo:rustc-link-lib=gcov");
        let mut manifest = format!(
            "# GCC profiles of the tree-sitter grammars, written by pgo/train.sh.\n\
             gcc {gcc_version}\ntarget {TARGET}\n"
        );
        for (krate, version) in &pinned {
            manifest.push_str(&format!("crate {krate} {version}\n"));
        }
        for (unit, hash) in units.iter().zip(&hashes) {
            manifest.push_str(&format!("unit {} {hash}\n", unit.name));
        }
        fs::create_dir_all(train).map_err(|e| format!("cannot create {}: {e}", train.display()))?;
        fs::write(train.join("profile.txt"), manifest).map_err(|e| e.to_string())?;
        fs::write(train.join("objdir"), obj_dir.display().to_string())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Compiles every unit with the flags `cc` uses for this target and profile, a few at a
/// time (`NUM_JOBS`), and returns the object paths in unit order.
fn compile_units(
    compiler: &cc::Tool,
    units: &[Unit],
    obj_dir: &Path,
    mode_flags: &[&str],
) -> Result<Vec<PathBuf>, String> {
    let jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, units.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs)
            .map(|_| {
                scope.spawn(|| -> Result<(), String> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let Some(unit) = units.get(i) else {
                            return Ok(());
                        };
                        compile_unit(compiler, unit, obj_dir, mode_flags)?;
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    results.into_iter().collect::<Result<(), String>>()?;
    Ok(units
        .iter()
        .map(|unit| obj_dir.join(format!("{}.o", unit.name)))
        .collect())
}

fn compile_unit(
    compiler: &cc::Tool,
    unit: &Unit,
    obj_dir: &Path,
    mode_flags: &[&str],
) -> Result<(), String> {
    let mut command = compiler.to_command();
    let symbol = format!("tree_sitter_{}", unit.grammar.name);
    let mut renames = vec![symbol.clone()];
    renames.extend(
        SCANNER_FUNCTIONS
            .iter()
            .map(|f| format!("{symbol}_external_scanner_{f}")),
    );
    for name in &renames {
        command.arg(format!("-D{name}=arko_index_pgo_{name}"));
    }
    command
        .arg("-std=c11")
        .arg("-I")
        .arg(unit.source.parent().unwrap())
        .args(mode_flags)
        .arg("-c")
        .arg(&unit.source)
        .arg("-o")
        .arg(obj_dir.join(format!("{}.o", unit.name)));
    let output = command
        .output()
        .map_err(|e| format!("cannot run the C compiler: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let first = stderr
        .lines()
        .find(|l| l.contains("error"))
        .unwrap_or("")
        .trim();
    Err(format!("compiling {} failed: {first}", unit.name))
}

/// The exact versions `Cargo.toml` pins for the grammar crates (`tree-sitter-x = "=1.2.3"`).
fn pinned_versions(manifest: &Path) -> Result<std::collections::BTreeMap<String, String>, String> {
    let text = fs::read_to_string(manifest).map_err(|e| format!("cannot read Cargo.toml: {e}"))?;
    let mut pinned = std::collections::BTreeMap::new();
    for grammar in GRAMMARS {
        let version = text.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            if key.trim() != grammar.krate {
                return None;
            }
            let value = value.trim().trim_matches('"');
            value.strip_prefix('=').map(|v| v.trim().to_string())
        });
        let version = version
            .ok_or_else(|| format!("{} is not pinned to an exact version", grammar.krate))?;
        pinned.insert(grammar.krate.to_string(), version);
    }
    Ok(pinned)
}

/// Finds each grammar crate's sources: next to the tree-sitter crate (the registry's
/// source directory, or a `cargo vendor` directory), else in `$CARGO_HOME/registry/src`.
/// A candidate must be the pinned version.
fn locate_crates(
    pinned: &std::collections::BTreeMap<String, String>,
) -> Result<std::collections::BTreeMap<&'static str, PathBuf>, String> {
    let mut roots = Vec::new();
    // tree-sitter's build script exports its `include` directory (links = "tree-sitter").
    if let Some(include) = env::var_os("DEP_TREE_SITTER_INCLUDE")
        && let Some(container) = Path::new(&include).parent().and_then(Path::parent)
    {
        roots.push(container.to_path_buf());
    }
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")));
    if let Some(registry) = cargo_home.map(|h| h.join("registry/src"))
        && let Ok(entries) = fs::read_dir(registry)
    {
        roots.extend(entries.flatten().map(|e| e.path()));
    }
    let mut found = std::collections::BTreeMap::new();
    for grammar in GRAMMARS {
        if found.contains_key(grammar.krate) {
            continue;
        }
        let version = &pinned[grammar.krate];
        let dir = roots
            .iter()
            .flat_map(|root| {
                [
                    root.join(format!("{}-{version}", grammar.krate)),
                    root.join(grammar.krate),
                ]
            })
            .find(|dir| crate_version(dir, grammar.krate).as_ref() == Some(version))
            .ok_or_else(|| format!("cannot find the sources of {} {version}", grammar.krate))?;
        found.insert(grammar.krate, dir);
    }
    Ok(found)
}

/// The `version` of the package `name` whose manifest is in `dir`.
fn crate_version(dir: &Path, name: &str) -> Option<String> {
    let text = fs::read_to_string(dir.join("Cargo.toml")).ok()?;
    let package = text
        .split("\n[")
        .find(|s| s.trim_start().starts_with("package]"))?;
    let field = |key: &str| {
        package.lines().find_map(|line| {
            let (k, v) = line.split_once('=')?;
            (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
        })
    };
    (field("name")? == name).then(|| field("version"))?
}

/// `pgo/profile.txt`: `gcc VERSION`, `target TRIPLE`, `crate NAME VERSION`, `unit NAME HASH`.
struct Profile {
    gcc_major: String,
    crates: std::collections::BTreeMap<String, String>,
    units: std::collections::BTreeMap<String, String>,
}

impl Profile {
    fn parse(text: &str) -> Profile {
        let mut profile = Profile {
            gcc_major: String::new(),
            crates: Default::default(),
            units: Default::default(),
        };
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            match fields.as_slice() {
                ["gcc", version] => {
                    profile.gcc_major = version.split('.').next().unwrap_or("").to_string()
                }
                ["crate", name, version] => {
                    profile.crates.insert(name.to_string(), version.to_string());
                }
                ["unit", name, hash] => {
                    profile.units.insert(name.to_string(), hash.to_string());
                }
                _ => {}
            }
        }
        profile
    }
}

fn command_output(program: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// 64-bit FNV-1a: detects a grammar source that differs from the profiled one.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
