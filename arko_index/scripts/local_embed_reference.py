#!/usr/bin/env python3
"""Reference vectors for arko_index's local embedding models, from each model's own
PyTorch code (`trust_remote_code`).

Usage:
  local_embed_reference.py coderank|jina OUT.json

Writes, for eight texts (the Jina model card's example, code, Unicode, a placeholder
and a 2,400-token text), the token ids, the first 16 components of each unit vector,
a projection over all components (sum of x_k * sin(k + 1)) and every pairwise cosine
similarity. `tests/local_embed.rs` compares the Rust implementation with these
(`tests/data/local-embed/`). Needs torch, transformers 4.56, einops and
huggingface_hub; downloads the pinned revisions.
"""
import json
import math
import sys

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModel

# name: (repository, revision, pooling, query prefix, document prefix, max tokens)
MODELS = {
    "jina": ("jinaai/jina-embeddings-v2-base-code", "516f4baf13dec4ddddda8631e019b5737c8bc250",
             "mean", "", "", 8192),
    "coderank": ("nomic-ai/CodeRankEmbed", "3c4b60807d71f79b43f3c4363786d9493691f8b1",
                 "cls", "Represent this query for searching relevant code: ", "", 8192),
}

LONG = "\n".join(
    f"fn step_{i}(state: &mut State, input: u32) -> Result<u32, Error> {{ let x = state.value.checked_add(input * {i}).ok_or(Error::Overflow)?; state.value = x; Ok(x) }}"
    for i in range(40)
)
TEXTS = [
    ("query", "How do I access the index while iterating over a sequence with a for loop?"),
    ("document", "# Use the built-in enumerator\nfor idx, x in enumerate(xs):\n    print(idx, x)"),
    ("query", "exponential backoff retry"),
    ("document", "src/net/retry.rs\nimpl RetryPolicy › fn backoff\npub fn backoff(&self, attempt: u32) -> Duration {\n    let base = self.base.as_millis() as u64;\n    Duration::from_millis(base.saturating_mul(1u64 << attempt.min(16)))\n}"),
    ("document", "func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {\n\ts.mux.ServeHTTP(w, r)\n}"),
    ("document", "Ünïcödé — ñ 日本語のテキスト 🚀 emoji"),
    ("document", "(empty)"),
    ("document", LONG),
]


def main(name, path):
    repo, rev, pooling, qprefix, dprefix, max_len = MODELS[name]
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
    tok.no_padding()
    tok.enable_truncation(max_len)
    # CodeRankEmbed's loader takes sentence-transformers' model_args.
    extra = {"safe_serialization": True, "add_pooling_layer": False} if name == "coderank" else {}
    model = AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True,
                                      torch_dtype=torch.float32, **extra)
    model.eval()
    out = {"repo": repo, "revision": rev, "texts": [], "ids": [], "vectors_head": [], "projection": []}
    vecs = []
    for kind, text in TEXTS:
        ids = tok.encode((qprefix if kind == "query" else dprefix) + text).ids
        with torch.no_grad():
            t = torch.tensor([ids])
            h = model(input_ids=t, attention_mask=torch.ones_like(t))[0][0]
        v = h.mean(0) if pooling == "mean" else h[0]
        v = v / v.norm()
        vecs.append(v)
        out["texts"].append([kind, text])
        out["ids"].append(ids)
        out["vectors_head"].append([round(float(x), 6) for x in v[:16]])
        out["projection"].append(round(sum(float(x) * math.sin(i + 1) for i, x in enumerate(v)), 6))
    out["similarity"] = [[round(float(a @ b), 6) for b in vecs] for a in vecs]
    with open(path, "w") as f:
        f.write("{\n")
        for i, (k, v) in enumerate(out.items()):
            sep = ",\n" if i < len(out) - 1 else "\n"
            f.write(json.dumps(k) + ": " + json.dumps(v, separators=(",", ":"), ensure_ascii=False) + sep)
        f.write("}\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main(sys.argv[1], sys.argv[2])
