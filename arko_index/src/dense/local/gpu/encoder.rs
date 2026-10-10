//! The encoders of the local models on burn tensors, generic over the backend: the
//! same layers as the CPU encoder ([`model`](super::super::model)), on padded batches.
//!
//! A batch is `rows × len` tokens, each text from position 0 and padded at its end.
//! Padded keys get a large negative score before the softmax, so a text's vector
//! depends only on its own tokens. Matrix products take the precision's float type
//! (`f32`, `f16` or `bf16`), with the weights stored in it. Layer norms, softmax,
//! residual sums and pooling run in `f32`.
//!
//! No tensor holds more than the device's largest buffer (`max_elements` f32 values).
//! Attention splits its query rows into blocks when one batch's scores would take more
//! than that or [`SCORE_ELEMENTS`], and the query, key and value projection and the
//! feed-forward layer, whose outputs are wider than a hidden state, run on blocks of
//! tokens when a batch's outputs would take more. The caller keeps a batch's hidden
//! states within the limit.
use super::super::model::{
    Architecture, Config, LayerWeights, Norm as NormWeights, Pooling, Weights, alibi_slopes,
    inv_freq, read_weights, rotary_table,
};
use super::super::safetensors::Tensors;
use crate::bench_v1::Error;
use burn_tensor::{
    FloatDType, Int, Tensor, TensorData,
    activation::{gelu, silu, softmax},
    backend::Backend,
};

/// The most attention scores computed at once (one block of query rows of a whole
/// batch, every head), 256 MB in `f32`.
pub(crate) const SCORE_ELEMENTS: usize = 1 << 26;

/// Tests lower the limit to run attention in blocks (0: [`SCORE_ELEMENTS`]).
#[cfg(test)]
pub(crate) static SCORE_LIMIT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn score_limit() -> usize {
    #[cfg(test)]
    {
        let limit = SCORE_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
        if limit > 0 {
            return limit;
        }
    }
    SCORE_ELEMENTS
}

/// Checks that a weight of `elements` f32 values fits in one buffer of the device.
fn fits(what: &str, elements: usize, max_elements: usize) -> Result<(), Error> {
    if elements <= max_elements {
        return Ok(());
    }
    Err(format!(
        "the model's {what} take {} MB, more than this device's largest buffer ({} MB)",
        (elements * 4).div_ceil(1 << 20),
        (max_elements * 4) >> 20
    )
    .into())
}

/// `f` applied to blocks of at most `rows` rows of `x` (`[n, _]`), joined again.
fn by_rows<B: Backend>(
    x: Tensor<B, 2>,
    rows: usize,
    f: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
) -> Tensor<B, 2> {
    let n = x.dims()[0];
    if n <= rows {
        return f(x);
    }
    let parts = (0..n)
        .step_by(rows)
        .map(|r0| f(x.clone().narrow(0, r0, rows.min(n - r0))))
        .collect();
    Tensor::cat(parts, 0)
}

/// The score of a padded key: `e^(x - max)` is zero for it, and adding ALiBi biases to
/// it stays finite.
const MASKED: f32 = -1.0e30;

struct Norm<B: Backend> {
    /// `[1, n]`.
    w: Tensor<B, 2>,
    b: Tensor<B, 2>,
}

impl<B: Backend> Norm<B> {
    fn new(n: NormWeights, device: &B::Device) -> Self {
        let len = n.w.len();
        Self {
            w: Tensor::from_data(TensorData::new(n.w, [1, len]), device),
            b: Tensor::from_data(TensorData::new(n.b, [1, len]), device),
        }
    }

    /// Layer normalization of each row of `x` (`f32`).
    fn apply(&self, x: Tensor<B, 2>, eps: f32) -> Tensor<B, 2> {
        let (var, mean) = x.clone().var_mean_bias(1);
        (x - mean) / (var + eps).sqrt() * self.w.clone() + self.b.clone()
    }
}

/// `y = x Wᵀ + b`, with `Wᵀ` stored `[in, out]` in the matrix product's float type and
/// the bias in `f32`.
struct Linear<B: Backend> {
    wt: Tensor<B, 2>,
    b: Option<Tensor<B, 2>>,
}

impl<B: Backend> Linear<B> {
    fn new(w: Weights, dtype: FloatDType, device: &B::Device) -> Self {
        let Weights { w, b, out, inp } = w;
        let mut wt = vec![0f32; w.len()];
        for (j, row) in w.chunks_exact(inp).enumerate() {
            for (k, v) in row.iter().enumerate() {
                wt[k * out + j] = *v;
            }
        }
        drop(w);
        Self {
            wt: Tensor::<B, 2>::from_data(TensorData::new(wt, [inp, out]), device).cast(dtype),
            b: b.map(|b| Tensor::from_data(TensorData::new(b, [1, out]), device)),
        }
    }

    /// `x[rows, in]` (`f32`) to `y[rows, out]` (`f32`).
    fn forward(&self, x: Tensor<B, 2>, dtype: FloatDType) -> Tensor<B, 2> {
        let y = x.cast(dtype).matmul(self.wt.clone()).cast(FloatDType::F32);
        match &self.b {
            Some(b) => y + b.clone(),
            None => y,
        }
    }
}

struct Layer<B: Backend> {
    qkv: Linear<B>,
    q_norm: Option<Norm<B>>,
    k_norm: Option<Norm<B>>,
    out: Linear<B>,
    attn_norm: Option<Norm<B>>,
    norm1: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
    norm2: Norm<B>,
}

impl<B: Backend> Layer<B> {
    fn new(
        w: LayerWeights,
        dtype: FloatDType,
        device: &B::Device,
        max_elements: usize,
    ) -> Result<Self, Error> {
        for weights in [&w.qkv, &w.out, &w.up, &w.down] {
            fits("linear weights", weights.w.len(), max_elements)?;
        }
        let norm = |n: Option<NormWeights>| n.map(|n| Norm::new(n, device));
        Ok(Self {
            qkv: Linear::new(w.qkv, dtype, device),
            q_norm: norm(w.q_norm),
            k_norm: norm(w.k_norm),
            out: Linear::new(w.out, dtype, device),
            attn_norm: norm(w.attn_norm),
            norm1: Norm::new(w.norm1, device),
            up: Linear::new(w.up, dtype, device),
            down: Linear::new(w.down, dtype, device),
            norm2: Norm::new(w.norm2, device),
        })
    }
}

/// A loaded encoder on one device.
pub(crate) struct Encoder<B: Backend> {
    cfg: Config,
    pooling: Pooling,
    dtype: FloatDType,
    device: B::Device,
    /// The most f32 values one tensor may hold: the device's largest buffer.
    max_elements: usize,
    vocab: usize,
    /// `[vocab, hidden]`, `f32`.
    word: Tensor<B, 2>,
    /// `[1, hidden]`.
    token_type: Tensor<B, 2>,
    emb_norm: Norm<B>,
    layers: Vec<Layer<B>>,
    /// ALiBi: minus each head's slope, `[1, heads, 1, 1]`.
    neg_slopes: Option<Tensor<B, 4>>,
    /// Rotary: the inverse frequency of each pair of a head's dimensions.
    inv_freq: Vec<f32>,
}

impl<B: Backend> Encoder<B> {
    /// Loads the weights onto `device`, the matrices in `dtype`, one layer at a time.
    /// `max_elements` is the most f32 values one tensor may hold on the device.
    pub(crate) fn load(
        cfg: Config,
        pooling: Pooling,
        t: &Tensors,
        dtype: FloatDType,
        device: &B::Device,
        max_elements: usize,
    ) -> Result<Self, Error> {
        let h = cfg.hidden;
        // The output of the widest projection of one token must fit.
        fits(
            "activations of one token",
            (3 * h).max(2 * cfg.intermediate),
            max_elements,
        )?;
        let (emb, layers) = read_weights(&cfg, t, |w| Layer::new(w, dtype, device, max_elements))?;
        fits("word embeddings", emb.word.len(), max_elements)?;
        let neg_slopes = (cfg.arch == Architecture::JinaBert).then(|| {
            let s: Vec<f32> = alibi_slopes(cfg.heads).iter().map(|s| -s).collect();
            Tensor::from_data(TensorData::new(s, [1, cfg.heads, 1, 1]), device)
        });
        let inv_freq = inv_freq(&cfg);
        Ok(Self {
            pooling,
            dtype,
            device: device.clone(),
            max_elements,
            vocab: emb.vocab,
            word: Tensor::from_data(TensorData::new(emb.word, [emb.vocab, h]), device),
            token_type: Tensor::from_data(TensorData::new(emb.token_type, [1, h]), device),
            emb_norm: Norm::new(emb.norm, device),
            layers,
            neg_slopes,
            inv_freq,
            cfg,
        })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.cfg.hidden
    }

    /// The pooled final states of `batch` (each text's token ids), `[rows, hidden]`
    /// in `f32`, not yet normalized. The texts are padded to `len` tokens, at least the
    /// longest one's. The batch's hidden states, `rows × len × hidden` f32 values, must
    /// fit in one buffer of the device.
    pub(crate) fn forward(&self, batch: &[&[u32]], len: usize) -> Result<Tensor<B, 2>, Error> {
        let (h, i, eps) = (self.cfg.hidden, self.cfg.intermediate, self.cfg.eps);
        let (heads, rows) = (self.cfg.heads, batch.len());
        let dh = h / heads;
        if batch.is_empty() || batch.iter().any(|t| t.is_empty() || t.len() > len) {
            return Err("a text is empty or longer than its batch".into());
        }
        let mut ids = vec![0i32; rows * len];
        let mut mask = vec![MASKED; rows * len];
        for (r, text) in batch.iter().enumerate() {
            for (k, &id) in text.iter().enumerate() {
                if id as usize >= self.vocab {
                    return Err(format!("token id {id} is outside the model's vocabulary").into());
                }
                ids[r * len + k] = id as i32;
                mask[r * len + k] = 0.0;
            }
        }
        let dev = &self.device;
        let ids = Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [rows * len]), dev);
        // Added to every score: `[rows, 1, 1, len]`.
        let mask = Tensor::<B, 4>::from_data(TensorData::new(mask, [rows, 1, 1, len]), dev);
        let rotary = (!self.inv_freq.is_empty()).then(|| {
            let half = self.inv_freq.len();
            let (cos, sin) = rotary_table(&self.inv_freq, len);
            (
                Tensor::<B, 4>::from_data(TensorData::new(cos, [1, len, 1, half]), dev),
                Tensor::<B, 4>::from_data(TensorData::new(sin, [1, len, 1, half]), dev),
            )
        });

        // Tokens per block of the projections wider than a hidden state.
        let qkv_rows = (self.max_elements / (3 * h)).max(1);
        let ffn_rows = (self.max_elements / (2 * i)).max(1);
        let x = self.word.clone().select(0, ids) + self.token_type.clone();
        let mut x = self.emb_norm.apply(x, eps);
        for layer in &self.layers {
            let (mut q, mut k, v) = if rows * len <= qkv_rows {
                let qkv = layer.qkv.forward(x.clone(), self.dtype);
                (
                    qkv.clone().narrow(1, 0, h),
                    qkv.clone().narrow(1, h, h),
                    qkv.narrow(1, 2 * h, h),
                )
            } else {
                // Queries, keys and values of each block of tokens, each joined over
                // the blocks.
                let (mut qs, mut ks, mut vs) = (Vec::new(), Vec::new(), Vec::new());
                for r0 in (0..rows * len).step_by(qkv_rows) {
                    let n = qkv_rows.min(rows * len - r0);
                    let qkv = layer.qkv.forward(x.clone().narrow(0, r0, n), self.dtype);
                    qs.push(qkv.clone().narrow(1, 0, h));
                    ks.push(qkv.clone().narrow(1, h, h));
                    vs.push(qkv.narrow(1, 2 * h, h));
                }
                (Tensor::cat(qs, 0), Tensor::cat(ks, 0), Tensor::cat(vs, 0))
            };
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                q = qn.apply(q, eps);
                k = kn.apply(k, eps);
            }
            let shape = [rows, len, heads, dh];
            let (mut q, mut k, v) = (q.reshape(shape), k.reshape(shape), v.reshape(shape));
            if let Some((cos, sin)) = &rotary {
                q = rotate(q, cos, sin);
                k = rotate(k, cos, sin);
            }
            let ctx = self.attention(
                q.swap_dims(1, 2),
                k.swap_dims(1, 2),
                v.swap_dims(1, 2),
                &mask,
            );
            let ctx = ctx.swap_dims(1, 2).reshape([rows * len, h]);
            let mut tmp = layer.out.forward(ctx, self.dtype);
            if let Some(an) = &layer.attn_norm {
                tmp = an.apply(tmp + x.clone(), eps);
            }
            x = layer.norm1.apply(x + tmp, eps);
            let ffn = |x: Tensor<B, 2>| {
                let up = layer.up.forward(x, self.dtype);
                let value = up.clone().narrow(1, 0, i);
                let gate = up.narrow(1, i, i);
                let act = match self.cfg.arch {
                    Architecture::JinaBert => value * gelu(gate),
                    Architecture::NomicBert => value * silu(gate),
                };
                layer.down.forward(act, self.dtype)
            };
            let tmp = by_rows(x.clone(), ffn_rows, ffn);
            x = layer.norm2.apply(x + tmp, eps);
        }
        let x = x.reshape([rows, len, h]);
        Ok(match self.pooling {
            Pooling::Cls => x.narrow(1, 0, 1).reshape([rows, h]),
            Pooling::Mean => {
                let lens: Vec<f32> = batch.iter().map(|t| t.len() as f32).collect();
                let lens = Tensor::<B, 2>::from_data(TensorData::new(lens, [rows, 1]), dev);
                // 1 for a text's tokens, 0 for padding: `[rows, len, 1]`.
                let keep = mask.reshape([rows, len, 1]).equal_elem(0.0).float();
                (x * keep).sum_dim(1).reshape([rows, h]) / lens
            }
        })
    }

    /// Scaled dot-product attention of `q`, `k` and `v` (`[rows, heads, len, head_dim]`,
    /// `f32`) with the padding `mask`, and ALiBi biases for Jina, in blocks of query rows
    /// of at most [`SCORE_ELEMENTS`] scores and what one buffer holds. Returns `f32`.
    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [rows, heads, len, dh] = q.dims();
        let scale = 1.0 / (dh as f32).sqrt();
        let block = query_block(rows * heads, len, score_limit().min(self.max_elements));
        let q = q.cast(self.dtype);
        let kt = k.cast(self.dtype).swap_dims(2, 3);
        let v = v.cast(self.dtype);
        let mut parts = Vec::with_capacity(len.div_ceil(block));
        for r0 in (0..len).step_by(block) {
            let n = block.min(len - r0);
            let qb = if n == len {
                q.clone()
            } else {
                q.clone().narrow(2, r0, n)
            };
            let mut scores = qb.matmul(kt.clone()).cast(FloatDType::F32) * scale + mask.clone();
            if let Some(neg_slopes) = &self.neg_slopes {
                let dev = &self.device;
                let r = Tensor::<B, 1, Int>::arange(r0 as i64..(r0 + n) as i64, dev)
                    .float()
                    .reshape([1, 1, n, 1]);
                let c = Tensor::<B, 1, Int>::arange(0..len as i64, dev)
                    .float()
                    .reshape([1, 1, 1, len]);
                scores = scores + (r - c).abs() * neg_slopes.clone();
            }
            let probs = softmax(scores, 3).cast(self.dtype);
            parts.push(probs.matmul(v.clone()));
        }
        let ctx = if parts.len() == 1 {
            parts.pop().expect("one block")
        } else {
            Tensor::cat(parts, 2)
        };
        ctx.cast(FloatDType::F32)
    }
}

/// Query rows per attention block: all of them when `streams × len × len` scores fit in
/// `limit`, else the most that fit, a multiple of 16 when at least 16 fit, and at least
/// one.
pub(crate) fn query_block(streams: usize, len: usize, limit: usize) -> usize {
    let fit = limit / (streams * len).max(1);
    if fit >= len {
        len
    } else if fit >= 16 {
        fit / 16 * 16
    } else {
        fit.max(1)
    }
}

/// Rotates queries or keys (`[rows, len, heads, head_dim]`) by their positions: the two
/// halves of a head form the pairs, GPT-NeoX style. `cos` and `sin` are
/// `[1, len, 1, head_dim / 2]`.
fn rotate<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let half = x.dims()[3] / 2;
    let a = x.clone().narrow(3, 0, half);
    let b = x.narrow(3, half, half);
    Tensor::cat(
        vec![
            a.clone() * cos.clone() - b.clone() * sin.clone(),
            b * cos.clone() + a * sin.clone(),
        ],
        3,
    )
}
