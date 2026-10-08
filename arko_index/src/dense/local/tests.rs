//! Offline tests on tiny random models of both architectures: the packed, threaded
//! forward pass against a direct transcription of the PyTorch models, batching and
//! thread invariance, prefixes, truncation and file checks. The real models are
//! checked against PyTorch by the ignored test in `tests/local_embed.rs`.
use super::safetensors::tests::write as write_safetensors;
use super::*;
use std::collections::HashMap;

type Weights = HashMap<String, Vec<f32>>;
/// A tensor to write: name, shape, values.
type Tensor = (String, Vec<usize>, Vec<f32>);

struct Tiny {
    arch: Architecture,
    hidden: usize,
    heads: usize,
    layers: usize,
    inter: usize,
    vocab: usize,
}

const JINA: Tiny = Tiny {
    arch: Architecture::JinaBert,
    hidden: 16,
    heads: 4,
    layers: 2,
    inter: 24,
    vocab: 64,
};
const NOMIC: Tiny = Tiny {
    arch: Architecture::NomicBert,
    hidden: 16,
    heads: 2,
    layers: 2,
    inter: 256,
    vocab: 64,
};

/// Deterministic values that half precision stores exactly: k / 1024, |k| ≤ 512.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 1025) as f32 / 1024.0 - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
    /// Layer-norm scales near one, multiples of 1 / 1024 (exact in half precision).
    fn scale(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| 1.0 + (self.next() * 256.0).round() / 1024.0)
            .collect()
    }
}

fn weights(t: &Tiny, seed: u64) -> (Weights, Vec<Tensor>) {
    let mut r = Rng(seed);
    let (h, i) = (t.hidden, t.inter);
    let mut list: Vec<Tensor> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, values: Vec<f32>| {
        list.push((name, shape, values));
    };
    add(
        "embeddings.word_embeddings.weight".into(),
        vec![t.vocab, h],
        r.vec(t.vocab * h),
    );
    add(
        "embeddings.token_type_embeddings.weight".into(),
        vec![2, h],
        r.vec(2 * h),
    );
    let norm = |add: &mut dyn FnMut(String, Vec<usize>, Vec<f32>), name: &str, r: &mut Rng| {
        add(format!("{name}.weight"), vec![h], r.scale(h));
        add(format!("{name}.bias"), vec![h], r.vec(h));
    };
    match t.arch {
        Architecture::JinaBert => {
            norm(&mut add, "embeddings.LayerNorm", &mut r);
            for l in 0..t.layers {
                let p = format!("encoder.layer.{l}.");
                for n in ["query", "key", "value"] {
                    add(
                        format!("{p}attention.self.{n}.weight"),
                        vec![h, h],
                        r.vec(h * h),
                    );
                    add(format!("{p}attention.self.{n}.bias"), vec![h], r.vec(h));
                }
                norm(&mut add, &format!("{p}attention.self.layer_norm_q"), &mut r);
                norm(&mut add, &format!("{p}attention.self.layer_norm_k"), &mut r);
                add(
                    format!("{p}attention.output.dense.weight"),
                    vec![h, h],
                    r.vec(h * h),
                );
                add(format!("{p}attention.output.dense.bias"), vec![h], r.vec(h));
                norm(&mut add, &format!("{p}attention.output.LayerNorm"), &mut r);
                norm(&mut add, &format!("{p}layer_norm_1"), &mut r);
                norm(&mut add, &format!("{p}layer_norm_2"), &mut r);
                add(
                    format!("{p}mlp.up_gated_layer.weight"),
                    vec![2 * i, h],
                    r.vec(2 * i * h),
                );
                add(
                    format!("{p}mlp.down_layer.weight"),
                    vec![h, i],
                    r.vec(h * i),
                );
                add(format!("{p}mlp.down_layer.bias"), vec![h], r.vec(h));
            }
        }
        Architecture::NomicBert => {
            norm(&mut add, "emb_ln", &mut r);
            for l in 0..t.layers {
                let p = format!("encoder.layers.{l}.");
                add(
                    format!("{p}attn.Wqkv.weight"),
                    vec![3 * h, h],
                    r.vec(3 * h * h),
                );
                add(format!("{p}attn.out_proj.weight"), vec![h, h], r.vec(h * h));
                add(format!("{p}mlp.fc11.weight"), vec![i, h], r.vec(i * h));
                add(format!("{p}mlp.fc12.weight"), vec![i, h], r.vec(i * h));
                add(format!("{p}mlp.fc2.weight"), vec![h, i], r.vec(h * i));
                norm(&mut add, &format!("{p}norm1"), &mut r);
                norm(&mut add, &format!("{p}norm2"), &mut r);
            }
        }
    }
    let map = list
        .iter()
        .map(|(n, _, v)| (n.clone(), v.clone()))
        .collect();
    (map, list)
}

fn config(t: &Tiny) -> String {
    match t.arch {
        Architecture::JinaBert => serde_json::json!({
            "hidden_size": t.hidden, "num_attention_heads": t.heads,
            "num_hidden_layers": t.layers, "intermediate_size": t.inter,
            "vocab_size": t.vocab, "layer_norm_eps": 1e-12,
            "position_embedding_type": "alibi", "feed_forward_type": "geglu",
        }),
        Architecture::NomicBert => serde_json::json!({
            "n_embd": t.hidden, "n_head": t.heads, "n_layer": t.layers, "n_inner": t.inter,
            "vocab_size": t.vocab, "layer_norm_epsilon": 1e-12, "activation_function": "swiglu",
            "rotary_emb_base": 1000, "rotary_emb_fraction": 1.0, "rotary_emb_interleaved": false,
            "rotary_scaling_factor": null, "prenorm": false, "causal": false,
            "qkv_proj_bias": false, "mlp_fc1_bias": false, "mlp_fc2_bias": false,
        }),
    }
    .to_string()
}

/// A word-level tokenizer: `[CLS] words [SEP]`, unknown words as `[UNK]`.
fn tokenizer(vocab: usize) -> String {
    let mut words: Vec<String> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "search", "for", ":"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    words.extend((words.len()..vocab).map(|n| format!("w{n}")));
    let vocab: serde_json::Map<String, serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(id, w)| (w.clone(), id.into()))
        .collect();
    let special = |id: usize, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true})
    };
    let tok = |id: &str| serde_json::json!({"SpecialToken": {"id": id, "type_id": 0}});
    let seq = |id: &str| serde_json::json!({"Sequence": {"id": id, "type_id": 0}});
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(2, "[CLS]"), special(3, "[SEP]")],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [tok("[CLS]"), seq("A"), tok("[SEP]")],
            "pair": [tok("[CLS]"), seq("A"), tok("[SEP]"), seq("B"), tok("[SEP]")],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]},
            },
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    })
    .to_string()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Writes a tiny model's files and returns its (leaked) catalog entry and weights.
fn tiny_model(
    t: &Tiny,
    dir: &Path,
    query_prefix: &'static str,
    max_tokens: usize,
) -> (&'static LocalModel, Weights) {
    let (map, list) = weights(t, 7 + t.hidden as u64 + t.heads as u64);
    let dtype = if t.arch == Architecture::JinaBert {
        "F16"
    } else {
        "F32"
    };
    let files = [
        ("config.json", config(t).into_bytes()),
        ("tokenizer.json", tokenizer(t.vocab).into_bytes()),
        ("model.safetensors", write_safetensors(&list, dtype)),
    ];
    let mut entries = Vec::new();
    for (name, bytes) in files {
        std::fs::write(dir.join(name), &bytes).unwrap();
        entries.push(ModelFile {
            name,
            size: bytes.len() as u64,
            sha256: Box::leak(hex(&bytes).into_boxed_str()),
        });
    }
    let model = LocalModel {
        name: "tiny",
        repo: "test/tiny",
        revision: "0000",
        dimensions: t.hidden,
        max_tokens,
        query_prefix,
        document_prefix: "",
        licence: "MIT",
        architecture: t.arch,
        pooling: if t.arch == Architecture::JinaBert {
            Pooling::Mean
        } else {
            Pooling::Cls
        },
        files: Box::leak(entries.into_boxed_slice()),
    };
    (Box::leak(Box::new(model)), map)
}

// --- A direct transcription of the PyTorch models, one text at a time ---

fn linear(w: &Weights, name: &str, x: &[Vec<f32>], out: usize, bias: bool) -> Vec<Vec<f32>> {
    let wt = &w[&format!("{name}.weight")];
    let inp = x[0].len();
    x.iter()
        .map(|row| {
            (0..out)
                .map(|j| {
                    let mut s: f32 = (0..inp).map(|k| row[k] * wt[j * inp + k]).sum();
                    if bias {
                        s += w[&format!("{name}.bias")][j];
                    }
                    s
                })
                .collect()
        })
        .collect()
}

fn layer_norm(w: &Weights, name: &str, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (g, b) = (&w[&format!("{name}.weight")], &w[&format!("{name}.bias")]);
    x.iter()
        .map(|row| {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            row.iter()
                .enumerate()
                .map(|(k, v)| (v - mean) / (var + 1e-12).sqrt() * g[k] + b[k])
                .collect()
        })
        .collect()
}

fn add(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(a, b)| a.iter().zip(b).map(|(a, b)| a + b).collect())
        .collect()
}

fn reference(t: &Tiny, w: &Weights, ids: &[u32]) -> Vec<f32> {
    let (h, i, nh) = (t.hidden, t.inter, t.heads);
    let dh = h / nh;
    let n = ids.len();
    let emb: Vec<Vec<f32>> = ids
        .iter()
        .map(|&id| {
            (0..h)
                .map(|k| {
                    w["embeddings.word_embeddings.weight"][id as usize * h + k]
                        + w["embeddings.token_type_embeddings.weight"][k]
                })
                .collect()
        })
        .collect();
    let jina = t.arch == Architecture::JinaBert;
    let mut x = layer_norm(
        w,
        if jina {
            "embeddings.LayerNorm"
        } else {
            "emb_ln"
        },
        &emb,
    );
    let slopes = super::model::tests::slopes(nh);
    for l in 0..t.layers {
        let (q, k, v) = if jina {
            let p = format!("encoder.layer.{l}.attention.self.");
            (
                layer_norm(
                    w,
                    &format!("{p}layer_norm_q"),
                    &linear(w, &format!("{p}query"), &x, h, true),
                ),
                layer_norm(
                    w,
                    &format!("{p}layer_norm_k"),
                    &linear(w, &format!("{p}key"), &x, h, true),
                ),
                linear(w, &format!("{p}value"), &x, h, true),
            )
        } else {
            let qkv = linear(
                w,
                &format!("encoder.layers.{l}.attn.Wqkv"),
                &x,
                3 * h,
                false,
            );
            let rot = |part: usize| -> Vec<Vec<f32>> {
                qkv.iter()
                    .enumerate()
                    .map(|(p, row)| {
                        let mut out = row[part * h..(part + 1) * h].to_vec();
                        for head in 0..nh {
                            for m in 0..dh / 2 {
                                let f = 1.0 / 1000f32.powf((2 * m) as f32 / dh as f32);
                                let (c, s) = ((p as f32 * f).cos(), (p as f32 * f).sin());
                                let (a, b) = (out[head * dh + m], out[head * dh + m + dh / 2]);
                                out[head * dh + m] = a * c - b * s;
                                out[head * dh + m + dh / 2] = b * c + a * s;
                            }
                        }
                        out
                    })
                    .collect()
            };
            let v = qkv.iter().map(|r| r[2 * h..].to_vec()).collect();
            (rot(0), rot(1), v)
        };
        let mut ctx = vec![vec![0f32; h]; n];
        for head in 0..nh {
            for a in 0..n {
                let mut s: Vec<f32> = (0..n)
                    .map(|b| {
                        let dot: f32 = (0..dh)
                            .map(|d| q[a][head * dh + d] * k[b][head * dh + d])
                            .sum();
                        let bias = if jina {
                            -slopes[head] * (a as f32 - b as f32).abs()
                        } else {
                            0.0
                        };
                        dot / (dh as f32).sqrt() + bias
                    })
                    .collect();
                let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                s.iter_mut().for_each(|v| *v = (*v - max).exp());
                let sum: f32 = s.iter().sum();
                for d in 0..dh {
                    ctx[a][head * dh + d] = (0..n).map(|b| s[b] / sum * v[b][head * dh + d]).sum();
                }
            }
        }
        let (gate_act, up_name, down_name, down_bias): (fn(f32) -> f32, String, String, bool);
        if jina {
            let p = format!("encoder.layer.{l}.");
            let dense = linear(w, &format!("{p}attention.output.dense"), &ctx, h, true);
            let attn = layer_norm(
                w,
                &format!("{p}attention.output.LayerNorm"),
                &add(&dense, &x),
            );
            x = layer_norm(w, &format!("{p}layer_norm_1"), &add(&x, &attn));
            gate_act = |g| 0.5 * g * (1.0 + libm::erff(g / 2f32.sqrt()));
            up_name = format!("{p}mlp.up_gated_layer");
            down_name = format!("{p}mlp.down_layer");
            down_bias = true;
        } else {
            let p = format!("encoder.layers.{l}.");
            let out = linear(w, &format!("{p}attn.out_proj"), &ctx, h, false);
            x = layer_norm(w, &format!("{p}norm1"), &add(&out, &x));
            gate_act = |g| g / (1.0 + (-g).exp());
            up_name = String::new();
            down_name = format!("{p}mlp.fc2");
            down_bias = false;
        }
        let act: Vec<Vec<f32>> = if jina {
            linear(w, &up_name, &x, 2 * i, false)
                .iter()
                .map(|u| (0..i).map(|m| u[m] * gate_act(u[i + m])).collect())
                .collect()
        } else {
            let p = format!("encoder.layers.{l}.");
            let y = linear(w, &format!("{p}mlp.fc11"), &x, i, false);
            let g = linear(w, &format!("{p}mlp.fc12"), &x, i, false);
            y.iter()
                .zip(&g)
                .map(|(y, g)| y.iter().zip(g).map(|(y, g)| y * gate_act(*g)).collect())
                .collect()
        };
        let down = linear(w, &down_name, &act, h, down_bias);
        let norm2 = if jina {
            format!("encoder.layer.{l}.layer_norm_2")
        } else {
            format!("encoder.layers.{l}.norm2")
        };
        x = layer_norm(w, &norm2, &add(&x, &down));
    }
    let mut v = if jina {
        (0..h)
            .map(|k| x.iter().map(|r| r[k]).sum::<f32>() / n as f32)
            .collect()
    } else {
        x[0].clone()
    };
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| (a - b).abs() <= tol)
}

/// Texts of 1 to 150 words: longer than one attention block of 64 query rows.
fn texts() -> Vec<String> {
    let mut texts = vec![
        "w10".to_string(),
        "w11 w12 w13".into(),
        "search for : unknown words w40".into(),
    ];
    texts.push(
        (0..150)
            .map(|k| format!("w{}", 7 + k % 57))
            .collect::<Vec<_>>()
            .join(" "),
    );
    texts.push(
        (0..70)
            .map(|k| format!("w{}", 63 - k % 50))
            .collect::<Vec<_>>()
            .join(" "),
    );
    texts
}

#[test]
fn matches_a_direct_transcription_of_both_architectures() {
    for t in [&JINA, &NOMIC] {
        let dir = tempfile::tempdir().unwrap();
        let (model, w) = tiny_model(t, dir.path(), "", 4096);
        let e = LocalEmbedder::load(model, dir.path())
            .unwrap()
            .with_threads(3)
            .unwrap();
        let texts = texts();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let got = e.embed(&refs, InputKind::Document).unwrap();
        let ids = e.encode(&refs, InputKind::Document).unwrap();
        assert_eq!(got.vectors.len(), texts.len());
        assert_eq!(got.tokens, ids.iter().map(|i| i.len() as u64).sum::<u64>());
        for (v, ids) in got.vectors.iter().zip(&ids) {
            let want = reference(t, &w, ids);
            assert!(close(v, &want, 2e-5), "{:?}\n{v:?}\n{want:?}", t.arch);
            assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn batching_threads_and_order_leave_vectors_unchanged() {
    for t in [&JINA, &NOMIC] {
        let dir = tempfile::tempdir().unwrap();
        let (model, _) = tiny_model(t, dir.path(), "", 4096);
        let texts = texts();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let one = LocalEmbedder::load(model, dir.path())
            .unwrap()
            .with_threads(1)
            .unwrap();
        let all = one.embed(&refs, InputKind::Document).unwrap().vectors;
        for threads in [2, 5] {
            // Small batches: forward passes of at most 40 tokens.
            let e = LocalEmbedder::load(model, dir.path())
                .unwrap()
                .with_threads(threads)
                .unwrap()
                .with_batch_tokens(40);
            let mut reversed = refs.clone();
            reversed.reverse();
            let mut got = e.embed(&reversed, InputKind::Document).unwrap().vectors;
            got.reverse();
            for (a, b) in all.iter().zip(&got) {
                assert!(close(a, b, 1e-5));
            }
            for (text, want) in refs.iter().zip(&all) {
                let alone = e.embed(&[text], InputKind::Document).unwrap().vectors;
                assert!(close(&alone[0], want, 1e-5));
            }
        }
    }
}

#[test]
fn queries_take_the_prefix_and_long_texts_are_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let (model, w) = tiny_model(&NOMIC, dir.path(), "search for : ", 32);
    let e = LocalEmbedder::load(model, dir.path()).unwrap();
    assert!(model.descriptor().input_types);
    assert_eq!(e.descriptor(), model.descriptor());
    let q = e.embed(&["w20 w21"], InputKind::Query).unwrap();
    let d = e.embed(&["w20 w21"], InputKind::Document).unwrap();
    assert_eq!((q.tokens, d.tokens), (7, 4));
    assert!(!close(&q.vectors[0], &d.vectors[0], 1e-3));
    assert!(close(
        &q.vectors[0],
        &reference(&NOMIC, &w, &[2, 4, 5, 6, 20, 21, 3]),
        2e-5
    ));
    // 100 words become 30 words and the two special tokens.
    let long = (0..100)
        .map(|k| format!("w{}", 10 + k % 50))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(e.count_tokens(&[&long, "w9"]).unwrap(), vec![32, 3]);
    let v = e.embed(&[&long], InputKind::Document).unwrap();
    assert_eq!(v.tokens, 32);
    let mut ids = vec![2u32];
    ids.extend((0..30).map(|k| 10 + k % 50));
    ids.push(3);
    assert!(close(&v.vectors[0], &reference(&NOMIC, &w, &ids), 2e-5));
    let limits = e.limits();
    assert_eq!(limits.max_input_tokens, 32);
    assert!(limits.max_request_tokens >= 32);
}

#[test]
fn damaged_or_missing_files_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (model, _) = tiny_model(&JINA, dir.path(), "", 64);
    let path = dir.path().join("model.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let err = LocalEmbedder::load(model, dir.path())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("SHA-256"), "{err}");
    std::fs::remove_file(dir.path().join("tokenizer.json")).unwrap();
    assert!(LocalEmbedder::load(model, dir.path()).is_err());
}

#[test]
fn the_catalog_names_pinned_models() {
    assert_eq!(DEFAULT_MODEL, &MODELS[0]);
    for m in MODELS {
        assert_eq!(find(m.name), Some(m));
        assert_eq!(find(&m.repo.to_uppercase()), Some(m));
        assert_eq!(LocalModel::for_descriptor(&m.descriptor()).unwrap(), m);
        assert_eq!(m.revision.len(), 40);
        let names: Vec<&str> = m.files.iter().map(|f| f.name).collect();
        assert_eq!(
            names,
            ["config.json", "tokenizer.json", "model.safetensors"]
        );
        assert!(m.files.iter().all(|f| f.sha256.len() == 64));
    }
    let d = CODE_RANK_EMBED.descriptor();
    assert_eq!(
        (d.provider.as_str(), d.model.as_str()),
        ("local", "nomic-ai/CodeRankEmbed")
    );
    assert!(d.input_types);
    assert!(!JINA_CODE_V2.descriptor().input_types);
    assert!(LocalModel::for_descriptor(&d.clone().with_revision("other")).is_err());
    assert!(find("voyage-code-4").is_none());
}
