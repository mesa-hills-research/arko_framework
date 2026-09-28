use crate::{
    Result,
    backend::{Backend, Limits},
    error::invalid,
    source::hash,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Payload {
    pub id: String,
    pub text: String,
    pub tokens: usize,
}
pub fn payload(backend: &dyn Backend, text: String) -> Result<Payload> {
    // The profile's key, not the profile: an unset tokenizer revision keys
    // vectors as the revision earlier versions pinned by default.
    let id = hash(serde_json::to_vec(&(
        backend.profile().key(),
        "float32",
        "document",
        "source-v1",
        &text,
    ))?);
    let tokens = backend.input_tokens(&text, false)?;
    Ok(Payload { id, text, tokens })
}
/// Global, deterministic first-fit decreasing packing. Input must be deduplicated.
pub fn pack(mut items: Vec<Payload>, limits: Limits) -> Result<Vec<Vec<Payload>>> {
    if limits.batch_items == 0 || limits.batch_tokens == 0 {
        return Err(invalid("zero embedding capacity"));
    }
    items.sort_by(|a, b| b.tokens.cmp(&a.tokens).then(a.id.cmp(&b.id)));
    let mut bins: Vec<(usize, Vec<Payload>)> = vec![];
    for item in items {
        if item.tokens > limits.item_tokens || item.tokens > limits.batch_tokens {
            return Err(invalid("payload exceeds embedding capacity"));
        }
        if let Some(bin) = bins.iter_mut().find(|(sum, items)| {
            items.len() < limits.batch_items && *sum <= limits.batch_tokens - item.tokens
        }) {
            bin.0 += item.tokens;
            bin.1.push(item);
        } else {
            bins.push((item.tokens, vec![item]));
        }
    }
    Ok(bins.into_iter().map(|(_, items)| items).collect())
}
pub fn normalize(mut vector: Vec<f32>, dimensions: usize) -> Result<Vec<f32>> {
    if vector.len() != dimensions || vector.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid embedding dimension/value"));
    }
    let norm = vector
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(invalid("invalid zero/nonfinite embedding norm"));
    }
    for v in &mut vector {
        *v = (*v as f64 / norm) as f32;
    }
    Ok(vector)
}
