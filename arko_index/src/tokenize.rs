//! Allocation-light term counting for the index builders: exactly the multiset of
//! terms that `search::tokens` returns, without a `String` per token.
//!
//! ASCII words (almost all source code) are handled on a lower-cased copy of the
//! text, so every term is a slice of that copy. A word with any non-ASCII character
//! goes through the original per-word code, because Unicode lower-casing is
//! context-sensitive (final sigma) and can change byte lengths.
use std::{
    borrow::Cow,
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

/// FxHash (rustc's hasher) with a final rotation that moves the best-mixed bits of
/// the last multiplication into the low bits hashbrown uses for bucket selection.
/// Not DoS-resistant. Keys are terms from the indexed tree.
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher {
    hash: u64,
}
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}
impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut b = bytes;
        while b.len() >= 8 {
            self.add(u64::from_le_bytes(b[..8].try_into().unwrap()));
            b = &b[8..];
        }
        if b.len() >= 4 {
            self.add(u64::from(u32::from_le_bytes(b[..4].try_into().unwrap())));
            b = &b[4..];
        }
        for &x in b {
            self.add(u64::from(x));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}
pub(crate) type FxBuild = BuildHasherDefault<FxHasher>;
pub(crate) type FxMap<K, V> = HashMap<K, V, FxBuild>;

/// Per-document term counts. Keys borrow the lower-cased text where possible.
pub(crate) type Counts<'a> = FxMap<Cow<'a, str>, u32>;

#[inline]
fn bump<'a>(counts: &mut Counts<'a>, term: Cow<'a, str>) {
    match counts.get_mut(term.as_ref()) {
        Some(n) => *n += 1,
        None => {
            counts.insert(term, 1);
        }
    }
}

/// Adds the terms of `text` to `counts`. `lowered` must be `text.to_ascii_lowercase()`
/// (same byte positions). The result equals counting `search::tokens(text)`.
pub(crate) fn count<'a>(text: &str, lowered: &'a str, counts: &mut Counts<'a>) -> u32 {
    debug_assert_eq!(text.len(), lowered.len());
    let bytes = text.as_bytes();
    let mut total = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        // Skip separators: anything that is neither alphanumeric nor '_'.
        let b = bytes[i];
        if b < 0x80 {
            if !(b.is_ascii_alphanumeric() || b == b'_') {
                i += 1;
                continue;
            }
        } else {
            let c = text[i..].chars().next().unwrap();
            if !c.is_alphanumeric() {
                i += c.len_utf8();
                continue;
            }
        }
        // A word: maximal run of alphanumeric characters and '_'.
        let start = i;
        let mut ascii = true;
        while i < bytes.len() {
            let b = bytes[i];
            if b < 0x80 {
                if b.is_ascii_alphanumeric() || b == b'_' {
                    i += 1;
                } else {
                    break;
                }
            } else {
                let c = text[i..].chars().next().unwrap();
                if c.is_alphanumeric() {
                    ascii = false;
                    i += c.len_utf8();
                } else {
                    break;
                }
            }
        }
        if ascii {
            total += ascii_word(&bytes[start..i], &lowered[start..i], counts);
        } else {
            for t in crate::search::tokens(&text[start..i]) {
                bump(counts, Cow::Owned(t));
                total += 1;
            }
        }
    }
    total
}

/// The `search::tokens` rules for one ASCII word: the whole word lower-cased, then
/// its `_` / camel-case parts (trimmed of `_`) that differ from the whole word.
#[inline]
fn ascii_word<'a>(w: &[u8], lower: &'a str, counts: &mut Counts<'a>) -> u32 {
    bump(counts, Cow::Borrowed(lower));
    let mut total = 1;
    let n = w.len();
    let mut prev_start = 0;
    let mut part = |a: usize, b: usize, counts: &mut Counts<'a>| {
        let p = lower[a..b].trim_matches('_');
        if !p.is_empty() && p != lower {
            bump(counts, Cow::Borrowed(p));
            total += 1;
        }
    };
    for i in 1..n {
        let (c, prev) = (w[i], w[i - 1]);
        if c == b'_'
            || prev == b'_'
            || (c.is_ascii_uppercase()
                && (prev.is_ascii_lowercase()
                    || (prev.is_ascii_uppercase() && i + 1 < n && w[i + 1].is_ascii_lowercase())))
        {
            part(prev_start, i, counts);
            prev_start = i;
        }
    }
    part(prev_start, n, counts);
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(text: &str) -> std::collections::BTreeMap<String, u32> {
        let mut m = std::collections::BTreeMap::new();
        for t in crate::search::tokens(text) {
            *m.entry(t).or_insert(0) += 1;
        }
        m
    }
    fn fast(text: &str) -> (std::collections::BTreeMap<String, u32>, u32) {
        let lowered = text.to_ascii_lowercase();
        let mut c = Counts::default();
        let total = count(text, &lowered, &mut c);
        (
            c.into_iter().map(|(k, v)| (k.into_owned(), v)).collect(),
            total,
        )
    }

    #[test]
    fn same_terms_as_search_tokens() {
        let cases = [
            "",
            "a",
            "_",
            "__init__ __x_ _a_b_",
            "retriesExponentialBackoff HTTPServer getHTTPResponse XMLHttpRequest",
            "ABC aBC AbC abC A1b2C3 x86_64 __NR_read sys_call_table",
            "kmalloc(GFP_KERNEL); struct file_operations *fops;",
            "café Ünïcode ΣΑΣ ΌΣΟΣ straße İstanbul ǅemal ﬁle",
            "mixed_ÄBC_def abcΣ ΣA_b Ωmega_Test",
            "日本語のテキスト 変数_name foo日本Bar",
            "tab\tnew\nline\r\nend 123 4five six7",
            "a__b ___ _A _a A_ a_ AB_CD ab_CD Ab_Cd",
        ];
        for t in cases {
            let (m, total) = fast(t);
            assert_eq!(m, reference(t), "{t:?}");
            assert_eq!(total as usize, crate::search::tokens(t).len(), "{t:?}");
        }
        // Deterministic pseudo-random strings over an alphabet with tricky characters.
        let alphabet: Vec<char> = "aZ_9 \n.Σσςİıß日Ab_xY0(é)Ǆǅ\u{301}".chars().collect();
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..20_000 {
            let len = (x % 24) as usize;
            let mut s = String::new();
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                s.push(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            let (m, total) = fast(&s);
            assert_eq!(m, reference(&s), "{s:?}");
            assert_eq!(total as usize, crate::search::tokens(&s).len(), "{s:?}");
        }
    }
}
