# arko_index local embeddings

With the `local-embed` feature, `dense::local::LocalEmbedder` runs a code-embedding
model on the CPU, in pure Rust, behind the same `dense::Embedder` trait as Voyage, and
with `local-embed-cuda` or `local-embed-gpu` on a [GPU](#gpus).
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
precision the index names. The `quantization` setting beside it is separate, and says how
the vector store keeps the vectors.

In a program, `LocalEmbedder::load(&local::CODE_RANK_EMBED, dir)` loads the model in
int8, and `LocalEmbedder::load_with_precision(&local::CODE_RANK_EMBED, dir,
Precision::F32)` in f32. Both read `config.json`, `tokenizer.json` and
`model.safetensors` from `dir` after checking each file's size and SHA-256 against
`LocalModel::files`. The library leaves fetching to the caller.
`examples/support/local.rs` does it with hf-hub in about 50 lines and can move unchanged
into a host crate.

## Models

| Model | Parameters | Licence | Prefixes | Pooling | Download |
|---|---|---|---|---|---|
| `nomic-ai/CodeRankEmbed` (default) | 137M | MIT | `Represent this query for searching relevant code: ` before queries | `[CLS]` | 548 MB |
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
  keys, and in int8 that projection alone moves the vectors more than the other three
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

- Tokens come from the model's own `tokenizer.json`, through the `tokenizers` crate with the
  pure-Rust fancy-regex engine. Every text is embedded whole. One longer than the
  model's 8,192 tokens, which arko_index's default chunks never reach, runs in windows
  of up to 8,192 tokens, each with the model's special tokens and the query prefix.
  Its vector is the mean of the windows' vectors, weighted by the text tokens each
  holds, at unit length.
- A build sends up to 256 texts and 8,192 tokens per request. The texts of
  a request run through the encoder packed back to back, without padding. The linear
  layers run once over all tokens, and attention runs per text, head and block of 128
  query rows.
- The int8 layers run as above. f32 matrices are multiplied by `gemm`,
  which picks AVX-512, AVX2 or NEON kernels at run time. Weights stored in f16 are
  widened on load. The softmax and SiLU take a polynomial exponential that vectorizes.
- The work runs on a dedicated rayon pool of `threads()` workers (`ARKO_INDEX_THREADS`),
  or `LocalEmbedder::with_threads`. Loading quantizes the weights on the same pool, in
  about 0.7 s for CodeRankEmbed, checksums included.
- The output is mean or `[CLS]` pooling as the model was trained, then unit length.
  `tokens` in the build's statistics counts the tokens processed. Nothing is billed, and
  a `--max-embed-tokens` cap still applies as a work limit.
- CodeRankEmbed holds 0.34 GB in int8 and 0.57 GB in f32 once loaded. A
  forward pass adds about 58 KB per token in flight, so a full pass of 8,192 tokens
  peaks at 0.8 GB in int8 and 1.0 GB in f32. A smaller `with_batch_tokens` lowers
  that. Loading peaks at 1.0 GB in int8 and 1.2 GB in f32, with the model file in
  memory.

The feature adds gemm, pulp, rayon, tokenizers and libm, all pure Rust:
`cargo tree -p arko_index --features local-embed -e normal,build --target all -i cc`
prints nothing.

## GPUs

With `local-embed-cuda` or `local-embed-gpu` the same models run on a GPU, on burn's
tensors in pure Rust. `local-embed-cuda` runs NVIDIA GPUs through CUDA. cubecl loads the
driver and NVRTC when it starts and compiles its kernels at run time, so the build needs
no CUDA toolkit. The machine needs the driver, NVRTC and the toolkit's headers (cudart,
crt and CCCL) of a CUDA version no newer than the driver's. They are found through
`CUDA_PATH`, else `/usr/local/cuda`, `/opt/cuda` or `/usr` (with `nvcc`), and `libnvrtc`
through the library path. NVIDIA's redistributable archives (`cuda_nvrtc`, `cuda_cudart`,
`cuda_cccl` and the `include/crt` folder of `cuda_nvcc`) unpacked into one folder named
by `CUDA_PATH` serve as well. `local-embed-gpu` runs any GPU through wgpu, with Vulkan, Metal
or DirectX 12.

```sh
cargo build --release -p arko_index --features local-embed-cuda --example arko_index_cli
$CLI index2 --embed local:gpu ./repo ./index-state          # CodeRankEmbed in f32 on the GPU
$CLI index2 --embed local:f16:cuda ./repo ./index-state     # in f16 through CUDA
```

| Device | Runs |
|---|---|
| `auto` (the default) | int8 on the CPU. f32 on CUDA when the build has it and the machine an NVIDIA GPU with NVRTC, else on a discrete GPU through wgpu, else on the CPU. f16 and bf16 as `gpu` |
| `cpu` | int8 and f32 |
| `gpu` | CUDA when available, else the first GPU wgpu finds, discrete ones first |
| `cuda`, `wgpu` | that backend |

`--embed local[:NAME][:PRECISION][:DEVICE]` names the device for that command and its
background embedding, and `ARKO_INDEX_EMBED_DEVICE` sets it for any command. A library
caller passes a `Device` to `LocalEmbedder::load_on`, or finds the device first with
`Placement::find`, which reads no model, and loads with `LocalEmbedder::load_placed` when
the model is first needed. The CLI works that way. It finds the device when a command
starts, so a missing GPU is reported at once, and loads the model on its first embedding.

- A GPU runs `f32` by default, and `f16` and `bf16` (CUDA only) on request. In
  `f16` and `bf16` every matrix product takes 16-bit inputs, the weights stored in
  them. Layer norms, softmax, residual sums and pooling stay in f32. The precision is
  part of the vectors' identity and the device isn't, so f32 vectors from a GPU and from
  the CPU differ by rounding and share one index. Searches embed their queries on the
  CPU when it runs the index's precision.
- Long texts are split into the same windows as on the CPU and combined the same way.
- A request takes up to 1,024 texts and eight forward passes' worth of
  tokens. Its texts are sorted by token count and split into batches of at most 4,096
  tokens with padding (`with_batch_tokens`), so texts of similar length share a batch.
  Padded keys are masked out of attention, and the vectors come back in input order.
  Smaller batches ran faster on the GPUs measured ([GPU speed](#gpu-speed)).
- Every tensor stays within the device's largest buffer, which is the adapter's
  largest storage buffer on wgpu (128 MiB on Mesa's llvmpipe) and a quarter of the GPU's
  memory on CUDA. A batch holds at most one buffer of hidden states. Attention splits a
  batch's query rows into blocks when its scores would pass that or 64M values (256 MB in
  f32), and the query, key and value projection and the feed-forward layer run on blocks
  of tokens when their outputs would.
- The background embedding holds the GPU only while it runs. It
  ends after its current request when its owner ends, and the driver frees what the
  process held when it exits.
- cubecl compiles a kernel the first time it needs it and times kernel
  variants the first time it sees a shape, about two minutes on a new machine. The
  timings stay in the user's cache directory (`~/.cache/cubecl` on Linux), and on CUDA
  the compiled kernels too (`~/.cache/cubecl/kernels/`, about 120 MB per GPU
  architecture and NVRTC version), so a later run starts in seconds.
- A vector with NaN or infinite values fails the request. In f16 the
  queries are scaled before their product with the keys, which keeps the scores far
  from f16's largest value.

The GPU features add burn 0.21 (its tensor, ndarray, wgpu and CUDA crates), cubecl, wgpu
and cudarc, also Rust only, so `cargo tree -p arko_index --features
local-embed-cuda,local-embed-gpu -e normal,build --target all -i cc` prints nothing as
well. burn's operation fusion is off. Its fused kernels returned wrong vectors for
whole batches of some shapes (on an L4 through CUDA and Vulkan, on an RTX 5090 through
CUDA), and the same operations unfused did not. `local_embed_device_bench` measures a
device and precision over a fixed corpus and compares the vectors with a reference
run's.

## Tests

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

### GPU speed

On an NVIDIA GeForce RTX 5090 through CUDA 13.2, CodeRankEmbed embedded 1,000 code
chunks (225,638 tokens, a median of 132 per chunk) in requests as a build sends them,
with the kernels already compiled and tuned. Texts per second, by batch size:

| Batch tokens | f32 texts/s | f32 tokens/s | f16 texts/s | f16 tokens/s | Padding |
|---|---|---|---|---|---|
| 4,096 | 548 | 124k | 706 | 159k | 17% |
| 8,192 | 466 | 105k | 566 | 128k | 21% |
| 16,384 | 402 | 91k | 470 | 106k | 25% |

Every vector of a sample of 130 chunks, two of them near the 8,192-token limit, stayed
within a cosine similarity of 0.999997 of the CPU's f32 vector in both precisions. The
first run compiled and tuned the kernels in 130 to 145 s, and a later run's first pass
took 2.2 s. An NVIDIA L4 ran f32 at 95 texts/s through CUDA and 55 through Vulkan, at
16,384 batch tokens and with burn's operation fusion, which made the RTX 5090 1.6 times
as fast in f16 at 8,192.

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
index state took 50.3 MB, with 23.0 MB of vectors, 23.1 MB of embedding cache and 4.2 MB of
lexical index. AVX-512 made the f32 matrix products 54% faster than AVX2 there.
