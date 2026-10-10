//! Texts longer than a model's input: split into windows the model takes, embedded
//! one by one, and combined into one vector per text. Independent of how the encoder
//! runs.
use crate::bench_v1::Error;

/// One window of a text: the token ids the encoder runs, and the weight of its vector
/// in the text's vector (the text tokens it holds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    pub ids: Vec<u32>,
    pub weight: usize,
}

/// Splits a tokenized text into windows of at most `max_tokens` ids. `ids` holds the
/// text's tokens with `lead` special tokens before them and `tail` after (`[CLS]` and
/// `[SEP]`, say). When the text starts with `prefix` (a query instruction), every
/// window repeats it. A text within `max_tokens` is one window, unchanged. Otherwise
/// each window holds the special tokens, the prefix and the next run of the text's
/// other tokens, as many as fit.
pub(crate) fn split(
    ids: &[u32],
    lead: usize,
    tail: usize,
    prefix: &[u32],
    max_tokens: usize,
) -> Result<Vec<Window>, Error> {
    if ids.len() <= max_tokens {
        return Ok(vec![Window {
            ids: ids.to_vec(),
            weight: ids.len(),
        }]);
    }
    let (head, rest) = ids.split_at(lead.min(ids.len()));
    let (body, end) = rest.split_at(rest.len().saturating_sub(tail));
    let (prefix, body) = match body.strip_prefix(prefix) {
        Some(body) => (prefix, body),
        None => (&[][..], body),
    };
    let room = max_tokens
        .checked_sub(head.len() + prefix.len() + end.len())
        .filter(|&r| r > 0)
        .ok_or("the special tokens and the prefix leave no room for text in a window")?;
    Ok(body
        .chunks(room)
        .map(|piece| Window {
            ids: [head, prefix, piece, end].concat(),
            weight: piece.len(),
        })
        .collect())
}

/// The vector of a text from the unit vectors of its windows: their mean weighted by
/// `weights`, at unit length.
pub(crate) fn combine(vectors: &[Vec<f32>], weights: &[usize]) -> Vec<f32> {
    if let [one] = vectors {
        return one.clone();
    }
    let mut v = vec![0f32; vectors.first().map_or(0, Vec::len)];
    for (w, &weight) in vectors.iter().zip(weights) {
        for (s, x) in v.iter_mut().zip(w) {
            *s += weight as f32 * x;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_texts_are_one_window() {
        let w = split(&[1, 10, 11, 2], 1, 1, &[], 4).unwrap();
        assert_eq!(
            w,
            [Window {
                ids: vec![1, 10, 11, 2],
                weight: 4
            }]
        );
    }

    #[test]
    fn long_texts_split_with_their_special_tokens() {
        // [CLS] 10..=16 [SEP] in windows of five: three text tokens each.
        let ids = [1, 10, 11, 12, 13, 14, 15, 16, 2];
        let w = split(&ids, 1, 1, &[], 5).unwrap();
        let got: Vec<(Vec<u32>, usize)> = w.into_iter().map(|w| (w.ids, w.weight)).collect();
        assert_eq!(
            got,
            [
                (vec![1, 10, 11, 12, 2], 3),
                (vec![1, 13, 14, 15, 2], 3),
                (vec![1, 16, 2], 1),
            ]
        );
    }

    #[test]
    fn every_window_repeats_the_prefix() {
        let ids = [1, 7, 8, 10, 11, 12, 13, 2];
        let w = split(&ids, 1, 1, &[7, 8], 6).unwrap();
        let ids: Vec<Vec<u32>> = w.iter().map(|w| w.ids.clone()).collect();
        assert_eq!(ids, [vec![1, 7, 8, 10, 11, 2], vec![1, 7, 8, 12, 13, 2]]);
        // A text without the prefix is split without it.
        let w = split(&[1, 10, 11, 12, 13, 2], 1, 1, &[7, 8], 4).unwrap();
        assert_eq!(w[0].ids, [1, 10, 11, 2]);
        assert!(split(&ids[0], 1, 1, &[7, 8], 4).is_err());
    }

    #[test]
    fn vectors_combine_by_weight() {
        assert_eq!(combine(&[vec![0.6, 0.8]], &[5]), [0.6, 0.8]);
        let v = combine(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[3, 1]);
        assert!((v[0] - 0.9486833).abs() < 1e-6 && (v[1] - 0.31622776).abs() < 1e-6);
    }
}
