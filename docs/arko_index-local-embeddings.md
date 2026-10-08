# arko_index local embeddings

With the `local-embed` feature, `dense::local::LocalEmbedder` runs a code-embedding
model on the CPU, in pure Rust, behind the same `dense::Embedder` trait as Voyage.
Dense and hybrid search then need no API key and cost nothing per token, and the code
stays on the machine. The [crate README](../arko_index/README.md#local-models) has the
overview, this page the details.

## Using it

```sh
cargo build --release -p arko_index --features local-embed --example arko_index_cli
CLI=target/release/examples/arko_index_cli
$CLI index2 --embed local ./repo ./index-state        # CodeRankEmbed, the default
$CLI index2 --embed local:jina-embeddings-v2-base-code ./repo ./index-state
$CLI search2 --text ./index-state 'where is the input limit applied'
```

The first build downloads the model's files into the Hugging Face cache (`$HF_HOME/hub`,
by default `~/.cache/huggingface/hub`), at the pinned commit. Later builds and
searches load them from there. The choice is saved in the index's `dense.json` like a
Voyage model, with the commit as the revision:

```json
"embedder": {
  "provider": "local",
  "model": "nomic-ai/CodeRankEmbed",
  "dimensions": 768,
  "input_types": true,
  "revision": "3c4b60807d71f79b43f3c4363786d9493691f8b1"
}
```

The descriptor is the embedding cache key, so vectors of another model or revision
never mix. A search embeds the query with the model the index names.

In a program: `LocalEmbedder::load(&local::CODE_RANK_EMBED, dir)` reads
`config.json`, `tokenizer.json` and `model.safetensors` from `dir` after checking each
file's size and SHA-256 against `LocalModel::files`. The library leaves fetching to the
caller. `examples/support/local.rs` does it with hf-hub in about 50 lines and can move
unchanged into a host crate.

## Models

| Model | Parameters | Licence | Prefixes | Pooling | Download |
|---|---|---|---|---|---|
| `nomic-ai/CodeRankEmbed` (default) | 137M | MIT | queries: `Represent this query for searching relevant code: ` | `[CLS]` | 548 MB |
| `jinaai/jina-embeddings-v2-base-code` | 161M | Apache-2.0 | none | mean | 324 MB |

Both have 768 dimensions and take up to 8,192 tokens. CodeRankEmbed is a code
retriever fine-tuned from Arctic-Embed-M-Long on 21M code pairs (CoRNStack). Its
model card reports CodeSearchNet MRR 77.9 and CoIR NDCG@10 60.1, against 67.2 and 58.4
for Jina's code model and 68.5 and 56.3 for voyage-code-002. On the arko evaluation
below it matched voyage-code-4, while the Jina model fell well behind, so CodeRankEmbed
is the default. The Jina model reads code with fewer tokens (about 14% fewer on the
arko snapshot) and so runs a little faster.

Qwen3-Embedding-0.6B scores higher on MTEB code tasks but has four times the
parameters, so it would embed about four times slower on the CPU. Adding it means a
decoder implementation (grouped-query attention, RMSNorm, last-token pooling).

## How it runs

- **Tokens.** The model's own `tokenizer.json`, through the `tokenizers` crate with the
  pure-Rust fancy-regex engine. Texts longer than 8,192 tokens are truncated, so every
  chunk gets a vector (arko_index's chunks stay far below that).
- **Batches.** A build sends up to 256 texts and 8,192 tokens per request. The texts of
  a request run through the encoder packed back to back, without padding: the linear
  layers run once over all tokens, attention runs per text, head and block of 64 query
  rows.
- **Arithmetic.** f32 matrices multiplied by `gemm`, which picks AVX-512, AVX2 or NEON
  kernels at run time. Weights stored in f16 are widened on load.
- **Threads.** A dedicated rayon pool of `threads()` workers (`ARKO_INDEX_THREADS`),
  or `LocalEmbedder::with_threads`.
- **Output.** Mean or `[CLS]` pooling as the model was trained, then unit length.
  `tokens` in the build's statistics counts the tokens processed. Nothing is billed, and
  a `--max-embed-tokens` cap still applies as a work limit.
- **Memory.** About 1 GB while embedding: the f32 weights (550 MB for CodeRankEmbed) and
  about 12 KB per token of the batch in flight.

The feature adds gemm, rayon, tokenizers and libm, all pure Rust:
`cargo tree -p arko_index --features local-embed -e normal,build --target all -i cc`
prints nothing.

## Verification

`tests/local_embed.rs` (ignored, it downloads both models) embeds eight texts with
each model, among them the Jina model card's example and a 2,400-token text, and
compares them with vectors from the models' own PyTorch code
(`scripts/local_embed_reference.py`, transformers 4.56 with `trust_remote_code`). Every
component and every pairwise similarity agrees within 2·10⁻⁴, token counts are
identical, and the Jina model card's similarity of 0.7282 comes out the same.

The offline unit tests build tiny random models of both architectures and check the
packed, threaded forward pass against a direct transcription of the PyTorch code, and
that batching, thread count and order leave every vector unchanged.

## Measurements

On 2026-10-08, on an AMD EPYC 9354P (Zen 4, AVX-512) where the build account may use 40
of the 64 hardware threads, on 28 physical cores, with other jobs running (load
average 13 to 29). CodeRankEmbed, on the arko repository (7,438 chunks, 303 tokens per
chunk on average):

| Threads | Chunks/s | Tokens/s |
|---|---|---|
| 1 | 1.3 | 368 |
| 4 | 4.8 | 1,400 |
| 8 | 9.2 | 2,691 |
| 16 | 16.7 | 5,042 |
| 40 | 22.3 | 6,753 |

A full `index2` build with embeddings took 330 s at 40 threads. The index state is
50.3 MB: 23.0 MB of vectors (768 × 4 bytes per chunk), 23.1 MB of embedding cache and
4.2 MB of lexical index. Loading the model with its checksums takes 0.95 s, and a
query embeds in about 30 ms. AVX-512 kernels made the 16-thread run 54% faster than
AVX2.

Retrieval on the arko evaluation snapshot (7,521 chunks, 19 questions with known
answers, `dense_live_check`), all three models on the same chunks:

| Model | Mode | Depth | Definitions | Callers | Behaviour | Embedding time |
|---|---|---|---|---|---|---|
| none | lexical | 4/9 | 4/4 | 4/7 | 3/6 | |
| voyage-code-4 | dense | 9/9 | 4/4 | 2/7 | 4/6 | 34 s (1.8M tokens billed) |
| voyage-code-4 | hybrid | 8/9 | 4/4 | 2/7 | 4/6 | |
| CodeRankEmbed | dense | 9/9 | 4/4 | 2/7 | 5/6 | 409 s |
| CodeRankEmbed | hybrid | 9/9 | 4/4 | 2/7 | 4/6 | |
| jina-embeddings-v2-base-code | dense | 5/9 | 4/4 | 2/7 | 2/6 | 333 s |
| jina-embeddings-v2-base-code | hybrid | 7/9 | 4/4 | 2/7 | 3/6 | |

The top ten results differ more than the found answers do. Against voyage-code-4,
CodeRankEmbed's dense top 10 shared 4.3 chunks on average (2 to 7) and the same first
result for 9 of 19 questions. Hybrid shared 6.5 (4 to 8) and the same first result for
12. Jina shared 3.7 dense and 6.1 hybrid, and the same first result for 3 and 4.

## Making dense search the default

Dense search can turn on by default once these hold:

1. **The host builds with `local-embed`** and a first use downloads 548 MB, which needs
   a notice and an opt-out (`--no-embed` exists).
2. **Embedding runs behind the lexical index.** A build now embeds before it
   publishes, so a first build of a large repository waits on the CPU (Kubernetes,
   259k chunks, would take hours). The lexical generation should publish first and
   vectors follow in the background. The cache already keeps every finished request,
   and hybrid search already treats chunks without vectors as lexical only.
3. **A thread budget.** Embedding takes every CPU by default. A background embedder
   should use a share of them, at low priority, so the machine stays usable.
4. **Measurements on laptops**, Apple Silicon (NEON kernels) included, since the numbers
   above come from a server.
5. **Memory.** Keeping the weights in f16 would halve the 550 MB, at some speed cost.
