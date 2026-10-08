#!/usr/bin/env python3
"""Checks arko_index's chunks of whole corpora, language by language.

Usage:
  language_check.py --cli NEW_CLI [--baseline BASE_CLI] [--twice] [--max-bytes N]
                    [--json OUT.json] [--] CORPUS [CORPUS ...]

CORPUS is a directory, optionally followed by `:FLAGS` with flags for the CLI's walk
(for example `~/src/repo:--exclude=runs/,--exclude=.runtime/`). For each corpus the
script runs `NEW_CLI chunk CORPUS` and checks every file's output against the file on
disk, without using the crate's code:

- coverage: the file's chunks, in order, cover it byte for byte, after its metadata
  region (a `file` entry: Markdown front matter and MDX import/export lines), which
  starts at byte 0;
- bounds: each chunk is at most --max-bytes (default 4096) bytes, and the path, a
  newline and the chunk fit in 24,000 bytes;
- UTF-8: every range starts and ends on a character boundary;
- lines: start_line and end_line match the bytes;
- identity: each ID is the SHA-256 of the path, a zero byte and the trimmed chunk
  text, then `-` and the occurrence number of that hash in the file;
- the token estimate is the chunk's bytes plus the path's plus one.

With --twice the corpus is chunked a second time and the outputs must be identical
(determinism). With --baseline, BASE_CLI (a build of the branch before the new
languages, for example arko-index-dense) chunks the corpus too, and every file it
chunked as one of the nine original languages (rust, python, typescript, javascript,
go, java, c, cpp) must have byte-identical output lines from NEW_CLI.

It prints a table per language (files, chunks, bytes, mean and largest chunk, the share
of chunks with a scope, files left unparsed, files with a metadata region, failed files)
and writes it to --json. A file is unparsed when every chunk has the kind `text`: the
bounded fallback chunked it (a line over 64 KiB, or the parser's work budget ran out).
The exit status is 1 when any check failed. Files of the original languages' grammars
that only the new build parses (new extensions such as `.mjs` or `.pyi`) are listed
in the JSON, not counted as failures.

Run it with the release CLI of each build, e.g.
  cargo build --release -p arko_index --example arko_index_cli
"""

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import threading

CLASSIC = {"rust", "python", "typescript", "javascript", "go", "java", "c", "cpp"}
RENDER_LIMIT = 24_000


def groups(proc):
    """(path, lines) per file, in output order, from a `chunk` process's stdout. A file
    entry (a line with a `file` key) opens its file's group."""
    current, lines = None, []
    for raw in proc.stdout:
        line = raw.rstrip(b"\n")
        if not line:
            continue
        record = json.loads(line)
        path = record.get("file") if "file" in record else record["path"]
        if path != current and lines:
            yield current, lines
            lines = []
        current = path
        lines.append((line, record))
    if lines:
        yield current, lines


def start(cli, root, flags):
    return subprocess.Popen(
        [cli, "chunk", root, *flags],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1 << 20,
    )


def drain_stderr(proc, sink):
    def run():
        sink.append(proc.stderr.read().decode("utf-8", "replace"))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def line_starts(data):
    starts = [0]
    at = data.find(b"\n")
    while at >= 0:
        starts.append(at + 1)
        at = data.find(b"\n", at + 1)
    return starts


def line_of(starts, byte, inclusive):
    """1-based line of `byte`: the number of line starts <= byte (inclusive) or < byte."""
    import bisect

    return bisect.bisect_right(starts, byte) if inclusive else bisect.bisect_left(starts, byte)


def check_file(root, path, lines, max_bytes):
    """Problems with one file's output, and its (language, chunk records, fallback count)."""
    problems = []
    try:
        data = open(os.path.join(root, path), "rb").read()
    except OSError as e:
        return [f"{path}: unreadable: {e}"], None
    meta_end = 0
    chunks = []
    for _, record in lines:
        if "file" in record:
            span = record["metadata_range"]
            if chunks or span["start_byte"] != 0:
                problems.append(f"{path}: metadata region not at the start")
            meta_end = span["end_byte"]
        else:
            chunks.append(record)
    starts = line_starts(data)
    expected = meta_end
    seen = collections.Counter()
    language = chunks[0]["language"] if chunks else "?"
    for c in chunks:
        s, e = c["start_byte"], c["end_byte"]
        if s != expected:
            problems.append(f"{path}: gap or overlap at byte {expected} (chunk starts at {s})")
        if not (s < e <= len(data)):
            problems.append(f"{path}: bad range {s}..{e} of {len(data)}")
            expected = max(expected, e)
            continue
        expected = e
        if e - s > max_bytes:
            problems.append(f"{path}: chunk {s}..{e} has {e - s} bytes > {max_bytes}")
        if len(path.encode()) + 1 + (e - s) > RENDER_LIMIT:
            problems.append(f"{path}: chunk {s}..{e} exceeds the rendered limit")
        piece = data[s:e]
        try:
            text = piece.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{path}: chunk {s}..{e} is not on UTF-8 boundaries")
            continue
        if c["start_line"] != line_of(starts, s, True) or c["end_line"] != line_of(starts, e, False):
            problems.append(f"{path}: wrong lines {c['start_line']}-{c['end_line']} for {s}..{e}")
        digest = hashlib.sha256(path.encode() + b"\0" + text.strip().encode()).hexdigest()
        occurrence = seen[digest]
        seen[digest] += 1
        if c["id"] != f"{digest}-{occurrence}":
            problems.append(f"{path}: id of {s}..{e} is not path + trimmed content")
        if c["token_estimate"] != (e - s) + len(path.encode()) + 1:
            problems.append(f"{path}: token estimate of {s}..{e}")
        if c["language"] != language:
            problems.append(f"{path}: mixed languages")
    if chunks or meta_end:
        if expected != len(data):
            problems.append(f"{path}: covered to {expected} of {len(data)} bytes")
    unparsed = language != "text" and bool(chunks) and all(c["kind"] == "text" for c in chunks)
    return problems, (language, chunks, unparsed, meta_end)


class Stats:
    def __init__(self):
        self.files = 0
        self.chunks = 0
        self.bytes = 0
        self.largest = 0
        self.scoped = 0
        self.unparsed = 0
        self.metadata = 0
        self.failures = 0

    def row(self):
        mean = self.bytes / self.chunks if self.chunks else 0
        return {
            "files": self.files,
            "chunks": self.chunks,
            "bytes": self.bytes,
            "mean_chunk": round(mean),
            "largest_chunk": self.largest,
            "scoped_pct": round(100 * self.scoped / self.chunks, 1) if self.chunks else 0,
            "unparsed_files": self.unparsed,
            "metadata_files": self.metadata,
            "failed_files": self.failures,
        }


def run_corpus(args, spec, stats, report):
    root, _, flag_text = spec.partition(":")
    root = os.path.expanduser(root)
    flags = [f for f in flag_text.split(",") if f]
    stderr = []
    proc = start(args.cli, root, flags)
    drain = drain_stderr(proc, stderr)
    digest = hashlib.sha256()
    new_out = {}
    problems_total = 0
    for path, lines in groups(proc):
        for raw, _ in lines:
            digest.update(raw + b"\n")
        problems, info = check_file(root, path, lines, args.max_bytes)
        if info is None:
            continue
        language, chunks, unparsed, meta_end = info
        s = stats[language]
        s.files += 1
        s.chunks += len(chunks)
        for c in chunks:
            size = c["end_byte"] - c["start_byte"]
            s.bytes += size
            s.largest = max(s.largest, size)
            s.scoped += bool(c.get("scope"))
        s.unparsed += unparsed
        s.metadata += bool(meta_end)
        if problems:
            s.failures += 1
            problems_total += len(problems)
            for p in problems[:5]:
                report["problems"].append(f"{root}: {p}")
        if args.baseline and language in CLASSIC:
            new_out[path] = hashlib.sha256(b"\n".join(raw for raw, _ in lines)).digest()
    proc.wait()
    drain.join()
    if proc.returncode != 0:
        report["problems"].append(f"{root}: {args.cli} exited {proc.returncode}: {stderr[0][-2000:] if stderr else ''}")
        problems_total += 1
    corpus = {"root": root, "flags": flags, "sha256": digest.hexdigest(), "problems": problems_total}
    if args.twice:
        again = start(args.cli, root, flags)
        drain2 = drain_stderr(again, [])
        second = hashlib.sha256()
        for block in iter(lambda: again.stdout.read(1 << 20), b""):
            second.update(block)
        again.wait()
        drain2.join()
        corpus["deterministic"] = second.hexdigest() == corpus["sha256"]
        if not corpus["deterministic"]:
            report["problems"].append(f"{root}: a second run gave different output")
    if args.baseline:
        base = start(args.baseline, root, flags)
        drain3 = drain_stderr(base, [])
        compared = differing = missing = 0
        for path, lines in groups(base):
            language = next((r["language"] for _, r in lines if "language" in r), None)
            if language not in CLASSIC:
                continue
            compared += 1
            old = hashlib.sha256(b"\n".join(raw for raw, _ in lines)).digest()
            new = new_out.pop(path, None)
            if new is None:
                missing += 1
                report["problems"].append(f"{root}: {path}: {language} in the baseline, not in the new output")
            elif new != old:
                differing += 1
                report["problems"].append(f"{root}: {path}: {language} output differs from the baseline")
        base.wait()
        drain3.join()
        # Files the baseline did not parse (`.mjs`, `.pyi`, ...): new, not a change.
        extra = len(new_out)
        report["new_classic_files"].extend(f"{root}: {path}" for path in list(new_out)[:20])
        corpus["baseline"] = {
            "classic_files": compared,
            "identical": compared - differing - missing,
            "differing": differing,
            "missing": missing,
            "new_only": extra,
        }
    report["corpora"].append(corpus)
    print(f"{root}: {json.dumps(corpus)}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--twice", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--json")
    parser.add_argument("corpora", nargs="+")
    args = parser.parse_args()
    stats = collections.defaultdict(Stats)
    report = {"corpora": [], "problems": [], "languages": {}, "new_classic_files": []}
    for spec in args.corpora:
        run_corpus(args, spec, stats, report)
    report["languages"] = {k: v.row() for k, v in sorted(stats.items())}
    header = f"{'language':<18}{'files':>9}{'chunks':>10}{'MB':>9}{'mean':>7}{'max':>7}{'scope%':>8}{'unparsed':>9}{'meta':>6}{'fail':>6}"
    print(header)
    for language, row in report["languages"].items():
        print(
            f"{language:<18}{row['files']:>9}{row['chunks']:>10}{row['bytes'] / 1e6:>9.1f}"
            f"{row['mean_chunk']:>7}{row['largest_chunk']:>7}{row['scoped_pct']:>8}"
            f"{row['unparsed_files']:>9}{row['metadata_files']:>6}{row['failed_files']:>6}"
        )
    for corpus in report["corpora"]:
        if "baseline" in corpus:
            b = corpus["baseline"]
            print(
                f"{corpus['root']}: original languages: {b['identical']}/{b['classic_files']} files identical to the baseline"
                f" ({b['differing']} differ, {b['missing']} missing, {b['new_only']} only in the new output)"
            )
        if "deterministic" in corpus:
            print(f"{corpus['root']}: deterministic: {corpus['deterministic']}")
    for p in report["problems"][:50]:
        print("PROBLEM", p)
    if len(report["problems"]) > 50:
        print(f"... {len(report['problems']) - 50} more problems")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    sys.exit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
