//! The transformer encoders of the local models, on `f32` matrices in row-major order.
//!
//! A batch is packed: every text's tokens follow the previous text's, with no padding,
//! so the linear layers run once over all tokens of the batch and attention runs per
//! text, head and block of query rows. Matrix products use `gemm`; every other step is
//! spread over the thread pool the caller runs this in.
use super::safetensors::Tensors;
use crate::bench_v1::Error;
use gemm::{Parallelism, gemm};
use rayon::prelude::*;

/// The model families the local embedder runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Architecture {
    /// Jina BERT v2 with layer norms on queries and keys (`jina-bert-v2-qk-post-norm`):
    /// ALiBi positions, GEGLU feed-forward, post-norm layers.
    JinaBert,
    /// Nomic BERT: rotary positions, SwiGLU feed-forward, post-norm layers.
    NomicBert,
}

/// How token states become one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pooling {
    /// The mean of every token's state.
    Mean,
    /// The state of the first token (`[CLS]`).
    Cls,
}

/// The shape of an encoder, from the model's `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Config {
    pub arch: Architecture,
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub rotary_base: f32,
    pub qkv_bias: bool,
    pub fc1_bias: bool,
    pub fc2_bias: bool,
}

impl Config {
    /// Reads `config.json`, refusing variants this implementation does not run.
    pub(crate) fn parse(arch: Architecture, json: &[u8]) -> Result<Self, Error> {
        let v: serde_json::Value =
            serde_json::from_slice(json).map_err(|e| format!("config.json: {e}"))?;
        let num = |k: &str| -> Result<usize, Error> {
            v[k].as_u64()
                .map(|n| n as usize)
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("config.json: no {k}").into())
        };
        let float = |k: &str| -> Result<f32, Error> {
            v[k].as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| format!("config.json: no {k}").into())
        };
        let flag = |k: &str| v[k].as_bool().unwrap_or(false);
        let require = |ok: bool, what: &str| -> Result<(), Error> {
            if ok {
                Ok(())
            } else {
                Err(format!("config.json: unsupported model variant ({what})").into())
            }
        };
        let cfg = match arch {
            Architecture::JinaBert => {
                require(
                    v["position_embedding_type"] == "alibi",
                    "position_embedding_type",
                )?;
                require(v["feed_forward_type"] == "geglu", "feed_forward_type")?;
                Config {
                    arch,
                    hidden: num("hidden_size")?,
                    heads: num("num_attention_heads")?,
                    layers: num("num_hidden_layers")?,
                    intermediate: num("intermediate_size")?,
                    eps: float("layer_norm_eps")?,
                    rotary_base: 0.0,
                    qkv_bias: true,
                    fc1_bias: false,
                    fc2_bias: true,
                }
            }
            Architecture::NomicBert => {
                require(v["activation_function"] == "swiglu", "activation_function")?;
                require(!flag("prenorm"), "prenorm")?;
                require(!flag("causal"), "causal")?;
                require(!flag("parallel_block"), "parallel_block")?;
                require(!flag("rotary_emb_interleaved"), "rotary_emb_interleaved")?;
                require(v["rotary_emb_fraction"] == 1.0, "rotary_emb_fraction")?;
                require(
                    v["rotary_scaling_factor"].is_null(),
                    "rotary_scaling_factor",
                )?;
                let inner = num("n_inner")?;
                Config {
                    arch,
                    hidden: num("n_embd")?,
                    heads: num("n_head")?,
                    layers: num("n_layer")?,
                    // The gated MLP rounds its width up to a multiple of 256.
                    intermediate: inner.div_ceil(256) * 256,
                    eps: float("layer_norm_epsilon")?,
                    rotary_base: float("rotary_emb_base")?,
                    qkv_bias: flag("qkv_proj_bias"),
                    fc1_bias: flag("mlp_fc1_bias"),
                    fc2_bias: flag("mlp_fc2_bias"),
                }
            }
        };
        if cfg.hidden % cfg.heads != 0 || (cfg.hidden / cfg.heads) % 2 != 0 {
            return Err("config.json: the hidden size must split into even-sized heads".into());
        }
        Ok(cfg)
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(t: &Tensors, name: &str, n: usize) -> Result<Self, Error> {
        Ok(Self {
            w: t.get(&format!("{name}.weight"), &[n])?,
            b: t.get(&format!("{name}.bias"), &[n])?,
        })
    }

    /// Layer normalization of one row in place.
    fn apply(&self, row: &mut [f32], eps: f32) {
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for ((v, w), b) in row.iter_mut().zip(&self.w).zip(&self.b) {
            *v = (*v - mean) * inv * w + b;
        }
    }
}

/// `y = x Wᵀ + b` with `W` stored `[out, in]`, as PyTorch stores it.
struct Linear {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    out: usize,
    inp: usize,
}

impl Linear {
    /// Loads and stacks the weights (and biases) of `names` along the output axis.
    fn load(
        t: &Tensors,
        names: &[String],
        out: usize,
        inp: usize,
        bias: bool,
    ) -> Result<Self, Error> {
        let mut w = Vec::with_capacity(names.len() * out * inp);
        let mut b = Vec::with_capacity(names.len() * out);
        for name in names {
            w.extend(t.get(&format!("{name}.weight"), &[out, inp])?);
            if bias {
                b.extend(t.get(&format!("{name}.bias"), &[out])?);
            }
        }
        Ok(Self {
            w,
            b: bias.then_some(b),
            out: names.len() * out,
            inp,
        })
    }

    /// `y[rows, out] = x[rows, in] Wᵀ + b`.
    fn forward(&self, x: &[f32], y: &mut [f32]) {
        let rows = x.len() / self.inp;
        debug_assert_eq!(y.len(), rows * self.out);
        // SAFETY: the slices hold `rows × inp`, `out × inp` and `rows × out` values,
        // and the strides below stay inside them.
        unsafe {
            gemm(
                rows,
                self.out,
                self.inp,
                y.as_mut_ptr(),
                1,
                self.out as isize,
                false,
                x.as_ptr(),
                1,
                self.inp as isize,
                // Wᵀ: element (k, j) is w[j × inp + k].
                self.w.as_ptr(),
                self.inp as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                Parallelism::Rayon(0),
            );
        }
        if let Some(b) = &self.b {
            y.par_chunks_mut(self.out).for_each(|row| {
                for (v, b) in row.iter_mut().zip(b) {
                    *v += b;
                }
            });
        }
    }
}

struct Layer {
    /// Queries, keys and values, stacked: `[3 × hidden, hidden]`.
    qkv: Linear,
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    out: Linear,
    /// Jina: the attention block's own `LayerNorm(out + input)`.
    attn_norm: Option<Norm>,
    norm1: Norm,
    /// The value half then the gate half: `[2 × intermediate, hidden]`.
    up: Linear,
    down: Linear,
    norm2: Norm,
}

/// A loaded encoder.
pub(crate) struct Encoder {
    cfg: Config,
    vocab: usize,
    word: Vec<f32>,
    token_type: Vec<f32>,
    emb_norm: Norm,
    layers: Vec<Layer>,
    /// ALiBi: each head's slope.
    slopes: Vec<f32>,
    /// Rotary: the inverse frequency of each pair of a head's dimensions.
    inv_freq: Vec<f32>,
}

/// A raw pointer that parallel tasks share to write disjoint parts of one buffer.
#[derive(Clone, Copy)]
struct SharedMut(*mut f32);
// SAFETY: every task writes only its own rows and columns.
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}

/// Query rows per attention task.
const QUERY_BLOCK: usize = 64;

impl Encoder {
    pub(crate) fn load(cfg: Config, t: &Tensors) -> Result<Self, Error> {
        let (h, i) = (cfg.hidden, cfg.intermediate);
        // Some checkpoints keep the `bert.` prefix of the pretraining model.
        let pre = if t.contains("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let word_name = format!("{pre}embeddings.word_embeddings.weight");
        let vocab = t.rows(&word_name)?;
        let word = t.get(&word_name, &[vocab, h])?;
        let tt_name = format!("{pre}embeddings.token_type_embeddings.weight");
        let types = t.rows(&tt_name)?;
        let token_type = t.get(&tt_name, &[types, h])?[..h].to_vec();
        let names = |p: &str, list: &[&str]| -> Vec<String> {
            list.iter().map(|n| format!("{p}{n}")).collect()
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        let emb_norm;
        match cfg.arch {
            Architecture::JinaBert => {
                emb_norm = Norm::load(t, &format!("{pre}embeddings.LayerNorm"), h)?;
                for l in 0..cfg.layers {
                    let p = format!("{pre}encoder.layer.{l}.");
                    layers.push(Layer {
                        qkv: Linear::load(
                            t,
                            &names(
                                &p,
                                &[
                                    "attention.self.query",
                                    "attention.self.key",
                                    "attention.self.value",
                                ],
                            ),
                            h,
                            h,
                            true,
                        )?,
                        q_norm: Some(Norm::load(
                            t,
                            &format!("{p}attention.self.layer_norm_q"),
                            h,
                        )?),
                        k_norm: Some(Norm::load(
                            t,
                            &format!("{p}attention.self.layer_norm_k"),
                            h,
                        )?),
                        out: Linear::load(t, &names(&p, &["attention.output.dense"]), h, h, true)?,
                        attn_norm: Some(Norm::load(
                            t,
                            &format!("{p}attention.output.LayerNorm"),
                            h,
                        )?),
                        norm1: Norm::load(t, &format!("{p}layer_norm_1"), h)?,
                        up: Linear::load(t, &names(&p, &["mlp.up_gated_layer"]), 2 * i, h, false)?,
                        down: Linear::load(t, &names(&p, &["mlp.down_layer"]), h, i, true)?,
                        norm2: Norm::load(t, &format!("{p}layer_norm_2"), h)?,
                    });
                }
            }
            Architecture::NomicBert => {
                emb_norm = Norm::load(t, &format!("{pre}emb_ln"), h)?;
                for l in 0..cfg.layers {
                    let p = format!("{pre}encoder.layers.{l}.");
                    layers.push(Layer {
                        qkv: Linear::load(t, &names(&p, &["attn.Wqkv"]), 3 * h, h, cfg.qkv_bias)?,
                        q_norm: None,
                        k_norm: None,
                        out: Linear::load(t, &names(&p, &["attn.out_proj"]), h, h, cfg.qkv_bias)?,
                        attn_norm: None,
                        norm1: Norm::load(t, &format!("{p}norm1"), h)?,
                        // fc11 is the value, fc12 the gate: fc11(x) · silu(fc12(x)).
                        up: Linear::load(
                            t,
                            &names(&p, &["mlp.fc11", "mlp.fc12"]),
                            i,
                            h,
                            cfg.fc1_bias,
                        )?,
                        down: Linear::load(t, &names(&p, &["mlp.fc2"]), h, i, cfg.fc2_bias)?,
                        norm2: Norm::load(t, &format!("{p}norm2"), h)?,
                    });
                }
            }
        }
        let dh = h / cfg.heads;
        let (slopes, inv_freq) = match cfg.arch {
            Architecture::JinaBert => (alibi_slopes(cfg.heads), Vec::new()),
            Architecture::NomicBert => (
                Vec::new(),
                (0..dh / 2)
                    .map(|k| 1.0 / cfg.rotary_base.powf((2 * k) as f32 / dh as f32))
                    .collect(),
            ),
        };
        Ok(Self {
            cfg,
            vocab,
            word,
            token_type,
            emb_norm,
            layers,
            slopes,
            inv_freq,
        })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.cfg.hidden
    }

    /// The final hidden states, `[tokens, hidden]`, of the packed texts: `ids` holds
    /// every text's tokens back to back and `lens` their counts. Runs on the current
    /// rayon pool.
    pub(crate) fn forward(&self, ids: &[u32], lens: &[usize]) -> Result<Vec<f32>, Error> {
        let (h, i, eps) = (self.cfg.hidden, self.cfg.intermediate, self.cfg.eps);
        let t = ids.len();
        if lens.iter().sum::<usize>() != t || lens.contains(&0) {
            return Err("token counts do not match the tokens".into());
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.vocab) {
            return Err(format!("token id {id} is outside the model's vocabulary").into());
        }
        let mut starts = Vec::with_capacity(lens.len());
        let mut pos = Vec::with_capacity(t);
        for &len in lens {
            starts.push(pos.len());
            pos.extend(0..len as u32);
        }

        let mut x = vec![0f32; t * h];
        x.par_chunks_mut(h)
            .zip(ids.par_iter())
            .for_each(|(row, &id)| {
                let w = &self.word[id as usize * h..(id as usize + 1) * h];
                for ((v, w), tt) in row.iter_mut().zip(w).zip(&self.token_type) {
                    *v = w + tt;
                }
                self.emb_norm.apply(row, eps);
            });

        let mut qkv = vec![0f32; t * 3 * h];
        let mut ctx = vec![0f32; t * h];
        let mut tmp = vec![0f32; t * h];
        let mut up = vec![0f32; t * 2 * i];
        let mut act = vec![0f32; t * i];
        let rotary = self.rotary_table(lens.iter().copied().max().unwrap_or(0));
        for layer in &self.layers {
            layer.qkv.forward(&x, &mut qkv);
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                qkv.par_chunks_mut(3 * h).for_each(|row| {
                    qn.apply(&mut row[..h], eps);
                    kn.apply(&mut row[h..2 * h], eps);
                });
            }
            if let Some((cos, sin)) = &rotary {
                self.rotate(&mut qkv, &pos, cos, sin);
            }
            self.attention(&qkv, &starts, lens, &mut ctx);
            layer.out.forward(&ctx, &mut tmp);
            if let Some(an) = &layer.attn_norm {
                tmp.par_chunks_mut(h)
                    .zip(x.par_chunks(h))
                    .for_each(|(a, x)| {
                        for (a, x) in a.iter_mut().zip(x) {
                            *a += x;
                        }
                        an.apply(a, eps);
                    });
            }
            add_norm(&mut x, &tmp, &layer.norm1, h, eps);
            layer.up.forward(&x, &mut up);
            let gelu = self.cfg.arch == Architecture::JinaBert;
            act.par_chunks_mut(i)
                .zip(up.par_chunks(2 * i))
                .for_each(|(o, u)| {
                    let (value, gate) = u.split_at(i);
                    for ((o, v), g) in o.iter_mut().zip(value).zip(gate) {
                        *o = v * if gelu { gelu_erf(*g) } else { silu(*g) };
                    }
                });
            layer.down.forward(&act, &mut tmp);
            add_norm(&mut x, &tmp, &layer.norm2, h, eps);
        }
        Ok(x)
    }

    /// Rotary cos and sin for positions `0..len`, `[len, head_dim / 2]` each.
    fn rotary_table(&self, len: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.inv_freq.is_empty() {
            return None;
        }
        let half = self.inv_freq.len();
        let (mut cos, mut sin) = (
            Vec::with_capacity(len * half),
            Vec::with_capacity(len * half),
        );
        for p in 0..len {
            for f in &self.inv_freq {
                let angle = p as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Some((cos, sin))
    }

    /// Rotates the query and key of every head by the token's position (the two halves
    /// of a head form the pairs, GPT-NeoX style).
    fn rotate(&self, qkv: &mut [f32], pos: &[u32], cos: &[f32], sin: &[f32]) {
        let h = self.cfg.hidden;
        let dh = h / self.cfg.heads;
        let half = dh / 2;
        qkv.par_chunks_mut(3 * h)
            .zip(pos.par_iter())
            .for_each(|(row, &p)| {
                let c = &cos[p as usize * half..(p as usize + 1) * half];
                let s = &sin[p as usize * half..(p as usize + 1) * half];
                for head in row[..2 * h].chunks_exact_mut(dh) {
                    let (a, b) = head.split_at_mut(half);
                    for k in 0..half {
                        let (x1, x2) = (a[k], b[k]);
                        a[k] = x1 * c[k] - x2 * s[k];
                        b[k] = x2 * c[k] + x1 * s[k];
                    }
                }
            });
    }

    /// Scaled dot-product attention within each text: `ctx[tokens, hidden]` from the
    /// packed `qkv[tokens, 3 × hidden]`. One task per text, head and block of query rows.
    fn attention(&self, qkv: &[f32], starts: &[usize], lens: &[usize], ctx: &mut [f32]) {
        let h = self.cfg.hidden;
        let heads = self.cfg.heads;
        let dh = h / heads;
        let stride = 3 * h;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut tasks = Vec::new();
        for (s, &len) in lens.iter().enumerate() {
            for head in 0..heads {
                for r0 in (0..len).step_by(QUERY_BLOCK) {
                    tasks.push((s, head, r0));
                }
            }
        }
        let out = SharedMut(ctx.as_mut_ptr());
        tasks
            .par_iter()
            .for_each_init(Vec::<f32>::new, |scores, &(s, head, r0)| {
                // Captures the whole wrapper: a closure using only `out.0` would
                // capture the bare pointer, which is not Sync.
                #[allow(clippy::redundant_locals)]
                let out = out;
                let (start, len) = (starts[s], lens[s]);
                let rows = QUERY_BLOCK.min(len - r0);
                scores.clear();
                scores.resize(rows * len, 0.0);
                // SAFETY: the query, key and value columns of this head and text lie
                // inside `qkv`; this task alone writes rows `start + r0..+rows`,
                // columns `head × dh..+dh` of `ctx`.
                unsafe {
                    // scores = scale · Q_block Kᵀ
                    gemm(
                        rows,
                        len,
                        dh,
                        scores.as_mut_ptr(),
                        1,
                        len as isize,
                        false,
                        qkv.as_ptr().add((start + r0) * stride + head * dh),
                        1,
                        stride as isize,
                        // Kᵀ: element (k, j) is key j's component k.
                        qkv.as_ptr().add(start * stride + h + head * dh),
                        stride as isize,
                        1,
                        0.0,
                        scale,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
                for (r, row) in scores.chunks_exact_mut(len).enumerate() {
                    if let Some(&slope) = self.slopes.get(head) {
                        let q = (r0 + r) as f32;
                        for (j, v) in row.iter_mut().enumerate() {
                            *v -= slope * (q - j as f32).abs();
                        }
                    }
                    softmax(row);
                }
                unsafe {
                    // ctx_block = scores V
                    gemm(
                        rows,
                        dh,
                        len,
                        out.0.add((start + r0) * h + head * dh),
                        1,
                        h as isize,
                        false,
                        scores.as_ptr(),
                        1,
                        len as isize,
                        qkv.as_ptr().add(start * stride + 2 * h + head * dh),
                        1,
                        stride as isize,
                        0.0,
                        1.0,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
            });
    }
}

/// `x = norm(x + y)`, row by row.
fn add_norm(x: &mut [f32], y: &[f32], norm: &Norm, h: usize, eps: f32) {
    x.par_chunks_mut(h).zip(y.par_chunks(h)).for_each(|(x, y)| {
        for (x, y) in x.iter_mut().zip(y) {
            *x += y;
        }
        norm.apply(x, eps);
    });
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// GELU with the exact error function (PyTorch's default).
fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ALiBi head slopes (Press et al.), as Jina computes them: a geometric sequence for
/// the largest power of two at most `heads`, then every other slope of the next power
/// of two.
fn alibi_slopes(heads: usize) -> Vec<f32> {
    fn power_of_two(n: usize) -> Vec<f64> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| start * start.powi(i as i32)).collect()
    }
    let closest = 1usize << heads.ilog2();
    let mut slopes = power_of_two(closest);
    if closest < heads {
        slopes.extend(
            power_of_two(2 * closest)
                .into_iter()
                .step_by(2)
                .take(heads - closest),
        );
    }
    slopes.into_iter().map(|s| s as f32).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn slopes(heads: usize) -> Vec<f32> {
        alibi_slopes(heads)
    }

    #[test]
    fn alibi_slopes_match_jina() {
        let s = alibi_slopes(12);
        let want = [
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
            0.70710677,
            0.35355338,
            0.17677669,
            0.088388346,
        ];
        assert_eq!(s.len(), 12);
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-7, "{s:?}");
        }
        assert_eq!(alibi_slopes(8)[0], 0.5);
    }

    #[test]
    fn activations() {
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_erf(-3.0) - -0.004_049_7).abs() < 1e-6);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
        let mut row = [1.0, 2.0, 3.0];
        softmax(&mut row);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((row[2] - 0.665_240_9).abs() < 1e-6);
    }
}
