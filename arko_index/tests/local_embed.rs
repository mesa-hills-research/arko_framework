//! The local models against their own PyTorch code (`--features local-embed`).
//!
//! Ignored by default: it downloads both models (about 870 MB) into the Hugging Face
//! cache. Run it with
//! `cargo test -p arko_index --features local-embed --test local_embed -- --ignored`.
//!
//! `tests/data/local-embed/*.json` hold, for eight texts (a model-card example, code,
//! Unicode, a 2,400-token text), the token ids and reference vectors computed with each
//! model's `trust_remote_code` PyTorch implementation by `scripts/local_embed_reference.py`:
//! the first 16 components, a projection over all 768 and every pairwise similarity.
#[path = "../examples/support/local.rs"]
mod local;

use arko_index::dense::{
    Embedder, InputKind,
    local::{CODE_RANK_EMBED, JINA_CODE_V2, LocalModel},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    repo: String,
    revision: String,
    texts: Vec<(String, String)>,
    ids: Vec<Vec<u32>>,
    vectors_head: Vec<Vec<f32>>,
    projection: Vec<f32>,
    similarity: Vec<Vec<f32>>,
}

fn check(model: &'static LocalModel, reference: &str) {
    let r: Reference = serde_json::from_str(reference).unwrap();
    assert_eq!(
        (r.repo.as_str(), r.revision.as_str()),
        (model.repo, model.revision)
    );
    let e = local::embedder(model).unwrap();
    // Every text in one batch: the packed forward pass with the real weights.
    let (mut texts, mut kinds) = (Vec::new(), Vec::new());
    for (kind, text) in &r.texts {
        texts.push(text.as_str());
        kinds.push(if kind == "query" {
            InputKind::Query
        } else {
            InputKind::Document
        });
    }
    let mut vectors = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let mut same_kind: Vec<&str> = vec![text];
        same_kind.extend(
            texts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && kinds[*j] == kinds[i])
                .map(|(_, t)| *t),
        );
        let got = e.embed(&same_kind, kinds[i]).unwrap();
        assert_eq!(got.vectors.len(), same_kind.len());
        vectors.push(got.vectors[0].clone());
    }
    for (i, v) in vectors.iter().enumerate() {
        let tokens = e.embed(&[texts[i]], kinds[i]).unwrap().tokens;
        assert_eq!(
            tokens as usize,
            r.ids[i].len(),
            "{}: token count of text {i}",
            model.name
        );
        for (k, (a, b)) in v.iter().zip(&r.vectors_head[i]).enumerate() {
            assert!(
                (a - b).abs() < 2e-4,
                "{}: text {i} component {k}: {a} vs {b}",
                model.name
            );
        }
        let projection: f32 = v
            .iter()
            .enumerate()
            .map(|(k, x)| x * ((k + 1) as f32).sin())
            .sum();
        assert!(
            (projection - r.projection[i]).abs() < 1e-3,
            "{}: text {i} projection {projection} vs {}",
            model.name,
            r.projection[i]
        );
        for (j, w) in vectors.iter().enumerate() {
            let sim: f32 = v.iter().zip(w).map(|(a, b)| a * b).sum();
            assert!(
                (sim - r.similarity[i][j]).abs() < 2e-4,
                "{}: similarity {i},{j}: {sim} vs {}",
                model.name,
                r.similarity[i][j]
            );
        }
    }
    // The tokenizer: ids identical to the Python tokenizers library's.
    let documents: Vec<&str> = r
        .texts
        .iter()
        .filter(|(k, _)| k == "document")
        .map(|(_, t)| t.as_str())
        .collect();
    let counts = e.count_tokens(&documents).unwrap();
    let want: Vec<u64> = r
        .texts
        .iter()
        .zip(&r.ids)
        .filter(|((k, _), _)| k == "document")
        .map(|(_, ids)| ids.len() as u64)
        .collect();
    assert_eq!(counts, want);
}

#[test]
#[ignore = "downloads both models (about 870 MB) from Hugging Face"]
fn code_rank_embed_matches_pytorch() {
    check(
        &CODE_RANK_EMBED,
        include_str!("data/local-embed/coderank.json"),
    );
}

#[test]
#[ignore = "downloads both models (about 870 MB) from Hugging Face"]
fn jina_code_matches_pytorch_and_its_model_card() {
    check(&JINA_CODE_V2, include_str!("data/local-embed/jina.json"));
    // The model card: cos_sim of its query and code example is 0.7282.
    let e = local::embedder(&JINA_CODE_V2).unwrap();
    let v = e
        .embed(
            &[
                "How do I access the index while iterating over a sequence with a for loop?",
                "# Use the built-in enumerator\nfor idx, x in enumerate(xs):\n    print(idx, x)",
            ],
            InputKind::Document,
        )
        .unwrap()
        .vectors;
    let sim: f32 = v[0].iter().zip(&v[1]).map(|(a, b)| a * b).sum();
    assert!((sim - 0.7282).abs() < 5e-4, "{sim}");
}
