#!/usr/bin/env bash
# Regenerates the GCC profiles of arko_index's `pgo` feature: pgo/profile.txt and
# pgo/gcda/*.gcda.
#
# Run it on x86-64 Linux with GCC:
# - after upgrading a tree-sitter grammar crate (Cargo.toml pins them): until then,
#   builds fall back to the grammar crates' own objects, with a cargo warning;
# - after moving to a new GCC major version: profiles only apply with the major
#   version that made them;
# - after changing the training corpus (pgo/corpus).
#
# It builds the arko_index_cli example with instrumented grammars, chunks the training
# corpus with it, copies the profiles into pgo/, then checks that an ordinary release
# build uses them. The corpus: pgo/corpus (samples of all eight language families),
# arko_index's own Rust sources, and from the pinned tree-sitter crates the core's C
# sources and each grammar's grammar.js and scanners.
#
# Usage: arko_index/pgo/train.sh [WORK_DIR]   (default: a new temporary directory)
set -euo pipefail
crate=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=${1:-$(mktemp -d)}
mkdir -p "$work"
work=$(cd "$work" && pwd)
export CARGO_TARGET_DIR="$work/target"
train="$work/train"
rm -rf "$train"

echo "train.sh: building instrumented grammars in $CARGO_TARGET_DIR" >&2
ARKO_INDEX_PGO_TRAIN="$train" cargo build --release --locked \
  --manifest-path "$crate/Cargo.toml" -p arko_index --example arko_index_cli
objdir=$(cat "$train/objdir")
rm -f "$objdir"/*.gcda

corpus="$work/corpus"
rm -rf "$corpus"
mkdir -p "$corpus/arko_index"
cp -r "$crate/pgo/corpus" "$corpus/samples"
cp -r "$crate/src" "$crate/tests" "$crate/examples" "$corpus/arko_index/"
cargo metadata --format-version 1 --locked --manifest-path "$crate/Cargo.toml" \
  > "$work/metadata.json"
python3 - "$work/metadata.json" "$corpus" <<'EOF'
import json, os, shutil, sys
metadata, corpus = sys.argv[1], sys.argv[2]
for package in json.load(open(metadata))["packages"]:
    name = package["name"]
    if not name.startswith("tree-sitter"):
        continue
    root = os.path.dirname(package["manifest_path"])
    out = os.path.join(corpus, "crates", f"{name}-{package['version']}")
    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            keep = (
                (name == "tree-sitter" and rel.startswith("src") and f.endswith((".c", ".h")))
                or f == "grammar.js"
                or f in ("scanner.c", "scanner.h")
            )
            if keep and "wasm" not in rel:
                os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
                shutil.copy(os.path.join(dirpath, f), os.path.join(out, rel))
EOF

echo "train.sh: chunking the training corpus" >&2
"$CARGO_TARGET_DIR/release/examples/arko_index_cli" chunk "$corpus" > "$work/chunks.jsonl"
python3 - "$work/chunks.jsonl" <<'EOF'
import collections, json, sys
languages = collections.Counter(json.loads(line)["language"] for line in open(sys.argv[1]))
print("train.sh: chunks per language:", dict(sorted(languages.items())), file=sys.stderr)
EOF

units=$(grep -c '^unit ' "$train/profile.txt")
profiles=$(find "$objdir" -name '*.gcda' | wc -l)
if [ "$units" != "$profiles" ]; then
  echo "train.sh: expected $units profiles, found $profiles in $objdir" >&2
  exit 1
fi
rm -rf "$crate/pgo/gcda"
mkdir -p "$crate/pgo/gcda"
cp "$objdir"/*.gcda "$crate/pgo/gcda/"
cp "$train/profile.txt" "$crate/pgo/profile.txt"
echo "train.sh: wrote $profiles profiles ($(du -sh "$crate/pgo/gcda" | cut -f1)) and pgo/profile.txt" >&2

echo "train.sh: checking that a release build uses them" >&2
cargo test --release --locked --manifest-path "$crate/Cargo.toml" -p arko_index --lib \
  every_pgo_grammar_matches_its_crate
