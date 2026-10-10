//! The GPU encoder against the CPU encoder on tiny random models of both
//! architectures: on burn's ndarray backend (no GPU needed) and, when wgpu finds an
//! adapter, on wgpu. Also the batch plan and the device choice.
use super::super::tests::{JINA, NOMIC, close, cosine, texts, tiny_model};
use super::encoder::{SCORE_LIMIT, query_block};
use super::*;
use crate::dense::{Embedder, InputKind, local::LocalEmbedder};
use std::sync::atomic::Ordering;

/// The tiny models' texts and a few longer ones (up to 400 tokens), so batches pad.
fn more_texts() -> Vec<String> {
    let mut t = texts();
    for n in [9, 33, 120, 251, 398] {
        t.push(
            (0..n)
                .map(|k| format!("w{}", 4 + (k * 7 + n) % 60))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    t
}

fn embed(e: &LocalEmbedder, texts: &[String]) -> Vec<Vec<f32>> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    e.embed(&refs, InputKind::Document).unwrap().vectors
}

/// The GPU encoder on `choice` against the CPU's `f32` vectors, within `tol` per
/// component and `min_cos` cosine similarity, at several batch sizes.
fn matches_cpu(choice: impl Fn() -> Choice, precision: Precision, tol: f32, min_cos: f32) {
    for t in [&JINA, &NOMIC] {
        let dir = tempfile::tempdir().unwrap();
        let (model, _) = tiny_model(t, dir.path(), "search for : ", 4096);
        let cpu = LocalEmbedder::load_with_precision(model, dir.path(), Precision::F32).unwrap();
        let texts = more_texts();
        let want = embed(&cpu, &texts);
        for batch_tokens in [16384, 400, 64] {
            let gpu = LocalEmbedder::load_choice(model, dir.path(), precision, Some(choice()))
                .unwrap()
                .with_batch_tokens(batch_tokens);
            assert!(gpu.on_gpu());
            let got = embed(&gpu, &texts);
            assert_eq!(got.len(), want.len());
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                let c = cosine(a, b);
                assert!(
                    close(a, b, tol) && c >= min_cos,
                    "{:?} {precision} text {k} ({batch_tokens} batch tokens): cosine {c}\n{a:?}\n{b:?}",
                    t.arch
                );
                assert!((cosine(a, a) - 1.0).abs() < 1e-4);
            }
            let stats = gpu.batch_stats();
            assert_eq!(stats.texts, texts.len() as u64);
            assert!(stats.padded_tokens >= stats.tokens);
            if batch_tokens == 64 {
                // A pass holds at most 64 padded tokens, or one longer text alone.
                assert!(stats.passes as usize >= texts.len() - 3);
            }
            // Queries take the prefix on the GPU as on the CPU.
            let q = gpu.embed(&["w20 w21"], InputKind::Query).unwrap();
            let want_q = cpu.embed(&["w20 w21"], InputKind::Query).unwrap();
            assert_eq!(q.tokens, want_q.tokens);
            assert!(cosine(&q.vectors[0], &want_q.vectors[0]) >= min_cos);
        }
    }
}

#[test]
fn ndarray_matches_the_cpu_encoder() {
    matches_cpu(|| Choice::NdArray, Precision::F32, 2e-5, 0.99999);
}

/// Texts over the model's input run in the same windows on the GPU as on the CPU,
/// combined the same way, queries with their prefix in every window.
#[test]
fn long_texts_run_in_the_cpus_windows() {
    let dir = tempfile::tempdir().unwrap();
    let (model, _) = tiny_model(&NOMIC, dir.path(), "search for : ", 32);
    let cpu = LocalEmbedder::load_with_precision(model, dir.path(), Precision::F32).unwrap();
    let gpu = LocalEmbedder::load_choice(model, dir.path(), Precision::F32, Some(Choice::NdArray))
        .unwrap();
    let long = (0..100)
        .map(|k| format!("w{}", 10 + k % 50))
        .collect::<Vec<_>>()
        .join(" ");
    let texts = [long.as_str(), "w9", long.as_str()];
    for kind in [InputKind::Document, InputKind::Query] {
        let want = cpu.embed(&texts, kind).unwrap();
        let got = gpu.embed(&texts, kind).unwrap();
        assert_eq!(got.tokens, want.tokens);
        assert_eq!(got.vectors.len(), 3);
        for (a, b) in got.vectors.iter().zip(&want.vectors) {
            assert!(close(a, b, 2e-5), "{kind:?}\n{a:?}\n{b:?}");
        }
    }
    assert_eq!(
        gpu.count_tokens(&texts).unwrap(),
        cpu.count_tokens(&texts).unwrap()
    );
    // Four windows of the long text (twice) and the short text: nine sequences.
    assert_eq!(gpu.batch_stats().texts, 2 * 9);
}

#[test]
fn attention_in_blocks_matches_one_block() {
    // 16 query rows per block: the 400-token texts take 25 blocks.
    SCORE_LIMIT.store(16 * 400 * 4, Ordering::Relaxed);
    let result =
        std::panic::catch_unwind(|| matches_cpu(|| Choice::NdArray, Precision::F32, 2e-5, 0.99999));
    SCORE_LIMIT.store(0, Ordering::Relaxed);
    result.unwrap();
}

#[test]
fn ndarray_refuses_half_precision() {
    let dir = tempfile::tempdir().unwrap();
    let (model, _) = tiny_model(&NOMIC, dir.path(), "", 64);
    for p in [Precision::F16, Precision::Bf16] {
        let err = LocalEmbedder::load_choice(model, dir.path(), p, Some(Choice::NdArray))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains(&format!("no {p} arithmetic")), "{err}");
    }
    let err = LocalEmbedder::load_choice(model, dir.path(), Precision::Int8, Some(Choice::NdArray))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("int8 runs on the CPU"), "{err}");
}

#[test]
fn plans_sort_by_length_within_the_budget() {
    let lens = [5usize, 300, 17, 300, 1, 64, 120, 9];
    let plan = Plan::new(&lens, 640);
    let order: Vec<usize> = plan.order().copied().collect();
    // Longest first, ties in input order.
    assert_eq!(order, vec![1, 3, 6, 5, 2, 7, 0, 4]);
    for (texts, len) in &plan.batches {
        assert_eq!(len % PAD, 0);
        assert!(texts.iter().all(|&i| lens[i] <= *len));
        assert!(
            texts.len() == 1 || texts.len() * len <= 640,
            "{texts:?} {len}"
        );
    }
    assert_eq!(plan.batches[0], (vec![1, 3], 304));
    // A text longer than the budget runs alone.
    let plan = Plan::new(&lens, 100);
    assert_eq!(plan.batches[0], (vec![1], 304));
    assert_eq!(plan.batches[1], (vec![3], 304));
    assert_eq!(
        plan.padded_tokens(),
        plan.batches
            .iter()
            .map(|(t, l)| (t.len() * l) as u64)
            .sum::<u64>()
    );
}

#[test]
fn query_blocks_bound_the_scores() {
    assert_eq!(query_block(12, 512, 1 << 26), 512);
    // 8,192 tokens and 12 heads: 682 rows fit, 672 in multiples of 16.
    assert_eq!(query_block(12, 8192, 1 << 26), 672);
    assert_eq!(query_block(1000, 8192, 1 << 20), 16);
    assert_eq!(query_block(1, 8, 1), 8);
}

#[test]
fn devices_parse_and_take_their_precisions() {
    use super::super::choose_gpu;
    for d in Device::ALL {
        assert_eq!(d.as_str().parse::<Device>().unwrap(), d);
    }
    assert_eq!("GPU".parse::<Device>().unwrap(), Device::Gpu);
    assert!("tpu".parse::<Device>().is_err());
    assert_eq!(Device::default(), Device::Auto);
    assert_eq!(Device::Auto.default_precision(), Precision::Int8);
    assert_eq!(Device::Gpu.default_precision(), Precision::F32);
    // Int8 stays on the CPU with Auto. The CPU refuses f16, a GPU refuses int8.
    assert!(choose_gpu(Device::Auto, Precision::Int8).unwrap().is_none());
    assert!(choose_gpu(Device::Cpu, Precision::F32).unwrap().is_none());
    let err = choose_gpu(Device::Cpu, Precision::F16)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("runs on a GPU"), "{err}");
    let err = choose_gpu(Device::Gpu, Precision::Int8)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("int8 runs on the CPU"), "{err}");
}

/// The first wgpu adapter (a GPU, or a CPU adapter such as Mesa's lavapipe), or `None`
/// with a note when there is none.
#[cfg(feature = "local-embed-gpu")]
fn wgpu_adapter() -> Option<Choice> {
    match wgpu(Adapters::Any) {
        Ok(choice) => Some(choice),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

#[cfg(feature = "local-embed-gpu")]
#[test]
fn wgpu_matches_the_cpu_encoder() {
    let Some(Choice::Wgpu {
        device,
        name,
        max_buffer,
    }) = wgpu_adapter()
    else {
        return;
    };
    eprintln!("wgpu adapter: {name}");
    let choice = || Choice::Wgpu {
        device: device.clone(),
        name: name.clone(),
        max_buffer,
    };
    matches_cpu(choice, Precision::F32, 1e-4, 0.99999);
    // Half precision where the adapter has it.
    let dir = tempfile::tempdir().unwrap();
    let (model, _) = tiny_model(&NOMIC, dir.path(), "", 4096);
    match LocalEmbedder::load_choice(model, dir.path(), Precision::F16, Some(choice())) {
        Ok(_) => matches_cpu(choice, Precision::F16, 2e-2, 0.999),
        Err(e) => eprintln!("f16 skipped: {e}"),
    }
}
