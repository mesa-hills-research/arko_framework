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
$CLI index2 --embed local ./repo ./index-state        # CodeRankEmbed in int8, the default
$CLI status ./index-state                             # the vectors fill in the background
$CLI search2 --text ./index-state 'where is the input limit applied'
$CLI index2 --embed local:f32 ./repo ./index-state    # the same model in f32
$CLI index2 --embed local:jina-embeddings-v2-base-code ./repo ./index-state
```

`--embed local[:NAME][:PRECISION]` names the model and the arithmetic it runs in:
`int8` (the default) or `f32`, described under [Precision](#precision). As with Voyage,
`index2` returns once lexical search is ready, and a background `embed` process loads
the model and fills in the vectors. The first build downloads the model's files into
the Hugging Face cache (`$HF_HOME/hub`, by default `~/.cache/huggingface/hub`), at the
pinned commit. Later builds and searches load them from there. The choice is saved in
the index's `dense.json` like a Voyage model, with the commit as the revision, followed
by `+int8` in int8:

```json
"embedder": {
  "provider": "local",
  "model": "nomic-ai/CodeRankEmbed",
  "dimensions": 768,
  "input_types": true,
  "revision": "3c4b60807d71f79b43f3c4363786d9493691f8b1+int8"
}
```

The descriptor is the embedding cache key, so vectors of another model, revision or
precision never mix, and the background run and every search load the model in the
precision the index names. The `quantization` setting beside it is separate: it is how
the vector store keeps the vectors.

In a program: `LocalEmbedder::load(&local::CODE_RANK_EMBED, dir)` loads the model in
int8, and `LocalEmbedder::load_with_precision(&local::CODE_RANK_EMBED, dir,
Precision::F32)` in f32. Both read `config.json`, `tokenizer.json` and
`model.safetensors` from `dir` after checking each file's size and SHA-256 against
`LocalModel::files`. The library leaves fetching to the caller.
`examples/support/local.rs` does it with hf-hub in about 50 lines and can move unchanged
into a host crate.

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

## Precision

The linear layers after attention, the output projection and the two feed-forward
layers, hold most of a model's weights and do most of its work. In int8, the default,
they run on integers:

- Both sides of each product are first rotated by a Hadamard matrix, in blocks of 256
  values. The rotation is orthogonal, so the products stay the same, and it spreads the
  few very large activations these models produce over their block, so they no longer
  set the quantization step of every other value.
- Weights are quantized per output channel at load, to 7 bits, and activations per
  token at run time, to 8 bits. The products are summed exactly in 32-bit integers, then
  scaled back to f32.
- The query, key and value projection stays in f32, as do attention, the embeddings,
  layer norms, pooling and normalization. The softmax magnifies errors in queries and
  keys: in int8, that projection alone moves the vectors more than the other three
  layers together.

On x86 CPUs with AVX2 the kernel multiplies bytes with `vpmaddubsw`, chosen at run time.
Elsewhere it is plain Rust that the compiler vectorizes for the target, and on ARM CPUs
with the dot product extension, Apple Silicon among them, its inner loop becomes `sdot`
instructions. Both compute the same integer sums. Weights take 7 bits because
`vpmaddubsw` adds pairs of byte products into 16 bits with saturation, and pairs of
8-bit activations times 7-bit weights stay within the 16-bit range.

Int8 embeds about 1.4 times as fast as f32 and takes 0.2 GB less memory. Its
vectors stay within a cosine similarity of 0.998 of the f32 ones, and on the arko
evaluation it finds what f32 finds ([measurements](#measurements)). `local:f32` runs
the model as it was trained.

## How it runs

- **Tokens.** The model's own `tokenizer.json`, through the `tokenizers` crate with the
  pure-Rust fancy-regex engine. Every text is embedded whole. One longer than the
  model's 8,192 tokens, which arko_index's default chunks never reach, runs in windows
  of up to 8,192 tokens, each with the model's special tokens and the query prefix.
  Its vector is the mean of the windows' vectors, weighted by the text tokens each
  holds, at unit length.
- **Batches.** A build sends up to 256 texts and 8,192 tokens per request. The texts of
  a request run through the encoder packed back to back, without padding: the linear
  layers run once over all tokens, attention runs per text, head and block of 128
  query rows.
- **Arithmetic.** The int8 layers as above. f32 matrices are multiplied by `gemm`,
  which picks AVX-512, AVX2 or NEON kernels at run time. Weights stored in f16 are
  widened on load. The softmax and SiLU take a polynomial exponential that vectorizes.
- **Threads.** A dedicated rayon pool of `threads()` workers (`ARKO_INDEX_THREADS`),
  or `LocalEmbedder::with_threads`. Loading quantizes the weights on the same pool, in
  about 0.7 s for CodeRankEmbed, checksums included.
- **Output.** Mean or `[CLS]` pooling as the model was trained, then unit length.
  `tokens` in the build's statistics counts the tokens processed. Nothing is billed, and
  a `--max-embed-tokens` cap still applies as a work limit.
- **Memory.** CodeRankEmbed holds 0.34 GB in int8 and 0.57 GB in f32 once loaded. A
  forward pass adds about 58 KB per token in flight, so a full pass of 8,192 tokens
  peaks at 0.8 GB in int8 and 1.0 GB in f32. A smaller `with_batch_tokens` lowers
  that. Loading peaks at 1.0 GB in int8 and 1.2 GB in f32, with the model file in
  memory.

The feature adds gemm, pulp, rayon, tokenizers and libm, all pure Rust:
`cargo tree -p arko_index --features local-embed -e normal,build --target all -i cc`
prints nothing.

## Verification

```sh
cargo test -p arko_index --features local-embed
cargo test -p arko_index --features local-embed --test local_embed -- --ignored   # downloads both models
```

The ignored tests compare both models' f32 vectors with their PyTorch code, within
2·10⁻⁴, and the int8 vectors with the f32 ones.

## Measurements

### Speed

On 6 cores of an Intel Core i9-13980HX, a laptop chip, in a virtual machine that gives
it AVX2. CodeRankEmbed embedded 150 chunks sampled evenly from the arko evaluation
snapshot, 327 tokens per chunk on average, each chunk whole. The better of two runs:

| Threads | f32 chunks/s | f32 tokens/s | int8 chunks/s | int8 tokens/s | int8 / f32 |
|---|---|---|---|---|---|
| 1 | 1.31 | 428 | 1.83 | 596 | 1.39 |
| 2 | 2.38 | 777 | 3.37 | 1,100 | 1.42 |
| 4 | 4.07 | 1,329 | 5.66 | 1,849 | 1.39 |
| 6 | 5.55 | 1,815 | 7.64 | 2,496 | 1.38 |

The whole snapshot, 7,521 chunks and 2.35 million tokens, took 982 s in int8 and
1,384 s in f32 at 6 threads. The Jina model ran 7.8 chunks/s in int8 and 6.0 in f32. A
query embeds in about 20 ms. Memory is under [How it runs](#how-it-runs).

### Int8 against f32

Over 3,000 chunks of the snapshot, the cosine similarity of each chunk's int8 and f32
vectors:

| Model | Lowest | 1st percentile | Median | Mean |
|---|---|---|---|---|
| CodeRankEmbed | 0.9982 | 0.9984 | 0.9989 | 0.9989 |
| jina-embeddings-v2-base-code (600 chunks) | 0.9989 | 0.9990 | 0.9993 | 0.9993 |

Retrieval on the arko evaluation snapshot (7,521 chunks, 19 questions with known
answers, `dense_live_check`), in both precisions:

| Model | Mode | Depth | Definitions | Callers | Behaviour |
|---|---|---|---|---|---|
| none | lexical | 4/9 | 4/4 | 4/7 | 3/6 |
| CodeRankEmbed, f32 | dense | 9/9 | 4/4 | 2/7 | 5/6 |
| CodeRankEmbed, f32 | hybrid | 9/9 | 4/4 | 2/7 | 4/6 |
| CodeRankEmbed, int8 | dense | 9/9 | 4/4 | 2/7 | 5/6 |
| CodeRankEmbed, int8 | hybrid | 9/9 | 4/4 | 2/7 | 4/6 |

The two precisions found the same answers. Their top ten results shared 9.7 chunks on
average in dense mode and 9.8 in hybrid mode (9 or 10 for every question), with the
same first result for 18 of the 19 questions in both.

### Against voyage-code-4

The same evaluation with voyage-code-4 and the Jina model, the local models in f32:

| Model | Mode | Depth | Definitions | Callers | Behaviour |
|---|---|---|---|---|---|
| voyage-code-4 | dense | 9/9 | 4/4 | 2/7 | 4/6 |
| voyage-code-4 | hybrid | 8/9 | 4/4 | 2/7 | 4/6 |
| CodeRankEmbed | dense | 9/9 | 4/4 | 2/7 | 5/6 |
| CodeRankEmbed | hybrid | 9/9 | 4/4 | 2/7 | 4/6 |
| jina-embeddings-v2-base-code | dense | 5/9 | 4/4 | 2/7 | 2/6 |
| jina-embeddings-v2-base-code | hybrid | 7/9 | 4/4 | 2/7 | 3/6 |

The top ten results differ more than the found answers do. Against voyage-code-4,
CodeRankEmbed's dense top 10 shared 4.3 chunks on average (2 to 7) and the same first
result for 9 of 19 questions. Hybrid shared 6.5 (4 to 8) and the same first result for
12. Jina shared 3.7 dense and 6.1 hybrid, and the same first result for 3 and 4.

### On a server

On 40 threads of an AMD EPYC 9354P (Zen 4, AVX-512) shared with other jobs, CodeRankEmbed
in f32 embedded the 7,438 chunks of the arko repository in 330 s, 22 chunks/s. The
index state took 50.3 MB: 23.0 MB of vectors, 23.1 MB of embedding cache and 4.2 MB of
lexical index. AVX-512 made the f32 matrix products 54% faster than AVX2 there.
