#!/usr/bin/env python3
"""Writes arko_index/src/rules/tags.rs: what each grammar's `queries/tags.scm` marks as a
definition, for the chunking rules in arko_index/src/rules/mod.rs.

Usage: gen_language_rules.py GRAMMAR_META [OUT]

GRAMMAR_META holds one folder per grammar key, with the grammar's `node-types.json` and
its `queries/` folder (a copy of the grammar repository's files at the version
mhr_tree_sitter ports). OUT defaults to arko_index/src/rules/tags.rs next to this script.

For every `@definition.<category>` capture in tags.scm, the node pattern it captures
gives a (kind, category) pair; alternations (`[ (a) (b) ] @definition.x`) give one pair
per alternative. Kinds that node-types.json does not list as named are dropped. The
output is sorted, so a run on the same inputs writes the same file.

The classic languages (their rules are frozen) and php_only (no file type) are left out.
"""

import json
import os
import re
import sys

SKIP = {"c", "cpp", "go", "java", "javascript", "python", "rust", "tsx", "typescript", "php_only"}

TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'  # strings
    r"|;[^\n]*"  # comments
    r"|@[\w.\-]+"  # captures
    r"|[()\[\]]"
    r"|[^\s()\[\]\"]+"
)


def tokens(text):
    for m in TOKEN.finditer(text):
        t = m.group(0)
        if not t.startswith(";"):
            yield t


def parse(text):
    """Nested lists: `(` and `[` open a list (its first element is "(" or "["), and a
    capture following an element is attached to it as ("capture", element, name)."""
    stack = [[]]
    for t in tokens(text):
        if t in "([":
            stack.append([t])
        elif t in ")]":
            done = stack.pop()
            stack[-1].append(done)
        elif t.startswith("@"):
            if stack[-1]:
                prev = stack[-1].pop()
                stack[-1].append(("capture", prev, t[1:]))
        else:
            stack[-1].append(t)
    return stack[0]


def kinds_of(pattern):
    """The node kinds a pattern element matches: `(kind ...)` gives kind, `[a b]` the
    kinds of its alternatives, a captured element the kinds of the element."""
    if isinstance(pattern, tuple):
        return kinds_of(pattern[1])
    if isinstance(pattern, list) and pattern:
        if pattern[0] == "(":
            head = pattern[1] if len(pattern) > 1 else None
            if isinstance(head, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", head):
                return [head]
            return []
        if pattern[0] == "[":
            out = []
            for alt in pattern[1:]:
                out += kinds_of(alt)
            return out
    return []


def definitions(tree, out):
    for element in tree:
        if isinstance(element, tuple):
            _, inner, name = element
            if name.startswith("definition."):
                for kind in kinds_of(inner):
                    out.add((kind, name.split(".", 1)[1]))
            definitions([inner], out)
        elif isinstance(element, list):
            definitions(element[1:], out)


def named_kinds(node_types):
    named = set()
    for entry in node_types:
        if entry.get("named"):
            named.add(entry["type"])
        for sub in entry.get("subtypes", []):
            if sub.get("named"):
                named.add(sub["type"])
    return named


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    meta = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "src", "rules", "tags.rs")
    lines = [
        "//! What each grammar's `queries/tags.scm` marks as a definition: `(kind, category)` for",
        "//! every `@definition.<category>` capture. Written by `tools/gen_language_rules.py`",
        "//! from the grammars' `tags.scm` and `node-types.json`: do not edit by hand. The",
        "//! curated rules in `mod.rs` start from these lists.",
        "",
    ]
    for key in sorted(os.listdir(meta)):
        tags = os.path.join(meta, key, "queries", "tags.scm")
        if key in SKIP or not os.path.isfile(tags):
            continue
        with open(os.path.join(meta, key, "node-types.json"), encoding="utf-8") as f:
            named = named_kinds(json.load(f))
        with open(tags, encoding="utf-8") as f:
            found = set()
            definitions(parse(f.read()), found)
        pairs = sorted((k, c) for k, c in found if k in named)
        lines.append(f"/// `{key}`: tags.scm definitions.")
        lines.append(f"pub(super) const {key.upper()}: &[(&str, &str)] = &[")
        for kind, category in pairs:
            lines.append(f'    ("{kind}", "{category}"),')
        lines.append("];")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
