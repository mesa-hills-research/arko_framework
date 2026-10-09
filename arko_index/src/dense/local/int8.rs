//! Int8 linear layers.
//!
//! Both sides of a product are first rotated by a block Hadamard matrix ([`hadamard`]),
//! which leaves the product unchanged and spreads a few large activations or weights
//! over their block, so they no longer set the quantization step of everything else.
//! Each weight matrix is then quantized at load, per output channel and symmetrically:
//! the channel's largest magnitude maps to [`W_MAX`]. At run time each input row (one
//! token) is quantized the same way, with its own scale, to [`A_MAX`] levels. The
//! kernels sum the products of the levels exactly in `i32`, so every kernel computes
//! the same sums and the same outputs. The epilogue scales the sums back to `f32` and
//! adds the bias.
//!
//! Weights take 7 bits so the AVX2 kernel can use `vpmaddubsw`, which multiplies
//! unsigned by signed bytes and adds pairs of products into 16 bits with saturation:
//! activations stored as unsigned bytes (the level plus 128, at most 255) times weights
//! within ±63 give pairs within 32,130, which never saturate.
use rayon::prelude::*;

/// Output channels per kernel call.
pub(crate) const NR: usize = 16;
/// Input rows per kernel call.
pub(crate) const MR: usize = 4;
/// Inputs per packed group: four bytes, one 32-bit lane.
const GROUP: usize = 4;
/// Largest weight level: 7 bits.
pub(crate) const W_MAX: i32 = 63;
/// Largest activation level: 8 bits.
pub(crate) const A_MAX: i32 = 127;
/// The largest rotation block. Larger blocks spread outliers over more values but
/// cost more to rotate, and 256 already gives the full gain on the local models.
const MAX_ROTATION: usize = 256;
/// Rows per parallel task at most: a task's activations stay in the core's cache.
const MAX_BLOCK_ROWS: usize = 128;

/// `y = x Wᵀ + b` with `W` quantized per output channel.
pub(crate) struct Int8Linear {
    out: usize,
    inp: usize,
    /// The kernel the weights are packed for.
    kernel: Kernel,
    /// Inputs per row, padded with zeros to a multiple of [`GROUP`].
    k: usize,
    /// The weight levels in the kernel's layout ([`Kernel::pack`]), in panels of
    /// [`NR`] output channels (the last one padded with zeros).
    weights: Vec<i8>,
    /// Per output channel (padded to a multiple of `NR`): the weight scale.
    scales: Vec<f32>,
    /// Per output channel: the share of the activations' zero point in a sum, the
    /// zero point times the sum of the channel's levels.
    offsets: Vec<i32>,
    /// Per output channel: the bias (zero when the layer has none).
    bias: Vec<f32>,
    /// The size of the Hadamard blocks both sides are rotated by (1: none).
    rotation: usize,
}

/// The rotation block for `inp` inputs: the largest power of two dividing it, at most
/// [`MAX_ROTATION`].
fn rotation_block(inp: usize) -> usize {
    let mut b = 1;
    while b * 2 <= MAX_ROTATION && inp.is_multiple_of(b * 2) {
        b *= 2;
    }
    b
}

/// Rotates each block of `block` values of `x` by the normalized Hadamard matrix (a
/// fast Walsh-Hadamard transform). The matrix is orthogonal and symmetric, so
/// `x · w = Hx · Hw` for every pair of rows.
pub(crate) fn hadamard(x: &mut [f32], block: usize) {
    if block <= 1 {
        return;
    }
    let norm = 1.0 / (block as f32).sqrt();
    for b in x.chunks_exact_mut(block) {
        let mut h = 1;
        if block >= 8 {
            // The first three stages within each run of eight, written out: the same
            // sums as the loop below, about twice as fast.
            for c in b.as_chunks_mut::<8>().0 {
                let a = [
                    c[0] + c[1],
                    c[0] - c[1],
                    c[2] + c[3],
                    c[2] - c[3],
                    c[4] + c[5],
                    c[4] - c[5],
                    c[6] + c[7],
                    c[6] - c[7],
                ];
                let b = [
                    a[0] + a[2],
                    a[1] + a[3],
                    a[0] - a[2],
                    a[1] - a[3],
                    a[4] + a[6],
                    a[5] + a[7],
                    a[4] - a[6],
                    a[5] - a[7],
                ];
                *c = [
                    b[0] + b[4],
                    b[1] + b[5],
                    b[2] + b[6],
                    b[3] + b[7],
                    b[0] - b[4],
                    b[1] - b[5],
                    b[2] - b[6],
                    b[3] - b[7],
                ];
            }
            h = 8;
        }
        while h < block {
            for pair in b.chunks_exact_mut(2 * h) {
                let (lo, hi) = pair.split_at_mut(h);
                for (a, c) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, v) = (*a, *c);
                    *a = u + v;
                    *c = u - v;
                }
            }
            h *= 2;
        }
        b.iter_mut().for_each(|v| *v *= norm);
    }
}

/// The largest magnitude in `x` (NaN is ignored), in lanes the compiler vectorizes.
fn abs_max(x: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for c in chunks {
        for (m, v) in lanes.iter_mut().zip(c) {
            let a = v.abs();
            *m = if a > *m { a } else { *m };
        }
    }
    for v in rest {
        let a = v.abs();
        lanes[0] = if a > lanes[0] { a } else { lanes[0] };
    }
    lanes
        .into_iter()
        .fold(0f32, |m, a| if a > m { a } else { m })
}

/// Quantizes `x` symmetrically to `-max..=max`: returns the scale and writes each
/// value's level, rounded to nearest, ties to even. A row of zeros has scale zero.
pub(crate) fn quantize(x: &[f32], max: i32, levels: &mut [i32]) -> f32 {
    let amax = abs_max(x);
    if amax == 0.0 || !amax.is_finite() {
        levels.fill(0);
        return if amax == 0.0 { 0.0 } else { f32::NAN };
    }
    let (inv, bound) = (max as f32 / amax, max as f32);
    // Adding 1.5 × 2^23 to a float of magnitude below 2^22 rounds it to an integer,
    // ties to even, held in the low bits of the sum: plain arithmetic that vectorizes
    // on every target (`round_ties_even` and saturating casts do not without SSE4.1).
    const ROUND: f32 = 12_582_912.0;
    let base = ROUND.to_bits() as i32;
    for (l, v) in levels.iter_mut().zip(x) {
        let t = v * inv;
        // Within ±max, and NaN to -max.
        let t = if t >= -bound { t } else { -bound };
        let t = if t <= bound { t } else { bound };
        *l = (t + ROUND).to_bits() as i32 - base;
    }
    amax / bound
}

impl Int8Linear {
    /// Rotates and quantizes `w` (`[out, inp]`, row-major, as PyTorch stores it) for
    /// the fastest kernel, and keeps `bias`.
    pub(crate) fn new(w: &[f32], bias: Option<&[f32]>, out: usize, inp: usize) -> Self {
        Self::with_kernel(Kernel::best(), w, bias, out, inp)
    }

    pub(crate) fn with_kernel(
        kernel: Kernel,
        w: &[f32],
        bias: Option<&[f32]>,
        out: usize,
        inp: usize,
    ) -> Self {
        assert_eq!(w.len(), out * inp);
        // Sums of `inp` products of at most 255 × 63 must fit an i32.
        assert!(inp <= 1 << 17, "too many inputs for int8 sums");
        let k = inp.next_multiple_of(GROUP);
        let padded = out.next_multiple_of(NR);
        let rotation = rotation_block(inp);
        let mut levels = vec![0i8; padded * k];
        let mut scales = vec![0f32; padded];
        let mut offsets = vec![0i32; padded];
        let mut bias_all = vec![0f32; padded];
        if let Some(b) = bias {
            bias_all[..out].copy_from_slice(b);
        }
        // Row by row on the current rayon pool.
        levels
            .par_chunks_mut(k)
            .zip(w.par_chunks(inp))
            .zip(scales.par_iter_mut().zip(offsets.par_iter_mut()))
            .for_each_init(
                || (vec![0f32; inp], vec![0i32; inp]),
                |(row, row_levels), ((levels, w), (scale, offset))| {
                    row.copy_from_slice(w);
                    hadamard(row, rotation);
                    *scale = quantize(row, W_MAX, row_levels);
                    *offset = kernel.zero() * row_levels.iter().sum::<i32>();
                    for (l, &v) in levels.iter_mut().zip(row_levels.iter()) {
                        *l = v as i8;
                    }
                },
            );
        Self {
            out,
            inp,
            kernel,
            k,
            weights: kernel.pack(levels, k),
            scales,
            offsets,
            bias: bias_all,
            rotation,
        }
    }

    /// `y[rows, out] = x[rows, inp] Wᵀ + b`, on the current rayon pool.
    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32]) {
        let (inp, k, out) = (self.inp, self.k, self.out);
        let rows = x.len() / inp;
        debug_assert_eq!(y.len(), rows * out);
        if rows == 0 {
            return;
        }
        let zero = self.kernel.zero();
        // The activations' bytes, rows padded to a multiple of MR.
        let mut bytes = vec![zero as u8; rows.next_multiple_of(MR) * k];
        let mut row_scales = vec![0f32; rows];
        bytes
            .par_chunks_mut(k)
            .zip(x.par_chunks(inp))
            .zip(row_scales.par_iter_mut())
            .for_each_init(
                || (vec![0f32; inp], vec![0i32; inp]),
                |(row, levels), ((bytes, x), scale)| {
                    row.copy_from_slice(x);
                    hadamard(row, self.rotation);
                    *scale = quantize(row, A_MAX, levels);
                    for (b, &l) in bytes.iter_mut().zip(levels.iter()) {
                        *b = (l + zero) as u8;
                    }
                },
            );
        let threads = rayon::current_num_threads();
        let block = rows
            .div_ceil(4 * threads)
            .next_multiple_of(MR)
            .clamp(MR, MAX_BLOCK_ROWS);
        let (bytes, row_scales) = (&bytes, &row_scales);
        y.par_chunks_mut(block * out)
            .enumerate()
            .for_each(|(b, y)| {
                let first = b * block;
                let n_rows = y.len() / out;
                let mut tile = [[0i32; NR]; MR];
                let mut values = [0f32; NR];
                for p in 0..out.div_ceil(NR) {
                    let panel = &self.weights[p * NR * k..(p + 1) * NR * k];
                    let cols = NR.min(out - p * NR);
                    let channels = p * NR..(p + 1) * NR;
                    let offsets: &[i32; NR] = self.offsets[channels.clone()].try_into().unwrap();
                    let scales: &[f32; NR] = self.scales[channels.clone()].try_into().unwrap();
                    let bias: &[f32; NR] = self.bias[channels].try_into().unwrap();
                    for r in (0..n_rows).step_by(MR) {
                        let a = &bytes[(first + r) * k..(first + r + MR) * k];
                        self.kernel.tile(a, k, panel, &mut tile);
                        for (i, sums) in tile.iter().enumerate().take(n_rows - r) {
                            let s = row_scales[first + r + i];
                            for c in 0..NR {
                                values[c] =
                                    (sums[c] - offsets[c]) as f32 * (s * scales[c]) + bias[c];
                            }
                            y[(r + i) * out + p * NR..][..cols].copy_from_slice(&values[..cols]);
                        }
                    }
                }
            });
    }
}

/// A kernel computing one tile of sums: [`MR`] activation rows against the [`NR`]
/// output channels of a panel.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Kernel {
    /// Plain Rust that the compiler vectorizes for the target: dot products of signed
    /// bytes, which become `sdot` instructions on ARM CPUs with the dot product
    /// extension (Apple Silicon among them) and widening multiply-adds elsewhere.
    Portable,
    /// AVX2: `vpmaddubsw`, `vpmaddwd` and `vpaddd` on 256-bit vectors.
    #[cfg(target_arch = "x86_64")]
    Avx2(pulp::x86::V3),
}

impl Kernel {
    /// The fastest kernel this CPU runs, detected once.
    pub(crate) fn best() -> Self {
        static BEST: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();
        *BEST.get_or_init(|| Self::available().pop().expect("the portable kernel"))
    }

    /// Every kernel this CPU runs, slowest first.
    pub(crate) fn available() -> Vec<Self> {
        #[allow(unused_mut)]
        let mut kernels = vec![Kernel::Portable];
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = pulp::x86::V3::try_new() {
            kernels.push(Kernel::Avx2(simd));
        }
        kernels
    }

    /// The activations' zero point: a stored byte is the level plus this. AVX2
    /// multiplies unsigned activation bytes, the portable kernel signed ones.
    fn zero(self) -> i32 {
        match self {
            Kernel::Portable => 0,
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2(_) => 128,
        }
    }

    /// Packs row-major levels (`[channels, k]`, channels a multiple of `NR`) into the
    /// kernel's layout. Portable: unchanged, so a panel is `NR` rows of `k` levels.
    /// AVX2: panel `p`, group `g` holds 64 bytes, channel `c`'s four levels at
    /// `4c..4c + 4`.
    fn pack(self, levels: Vec<i8>, k: usize) -> Vec<i8> {
        match self {
            Kernel::Portable => {
                debug_assert!(levels.len().is_multiple_of(k));
                levels
            }
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2(_) => {
                let mut packed = vec![0i8; levels.len()];
                for (n, row) in levels.chunks_exact(k).enumerate() {
                    let (p, c) = (n / NR, n % NR);
                    for (g, four) in row.chunks_exact(GROUP).enumerate() {
                        let at = ((p * (k / GROUP) + g) * NR + c) * GROUP;
                        packed[at..at + GROUP].copy_from_slice(four);
                    }
                }
                packed
            }
        }
    }

    /// `out[r][c]` = the sum over the `k` inputs of activation byte × weight level, for
    /// the `MR` rows of `a` and the `NR` channels of `panel`.
    #[inline]
    pub(crate) fn tile(self, a: &[u8], k: usize, panel: &[i8], out: &mut [[i32; NR]; MR]) {
        match self {
            Kernel::Portable => portable(a, k, panel, out),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2(simd) => pulp::Simd::vectorize(
                simd,
                avx2::Tile {
                    simd,
                    a,
                    k,
                    panel,
                    out,
                },
            ),
        }
    }
}

fn portable(a: &[u8], k: usize, panel: &[i8], out: &mut [[i32; NR]; MR]) {
    for (c, w) in panel.chunks_exact(k).enumerate() {
        for (out, a) in out.iter_mut().zip(a.chunks_exact(k)) {
            out[c] = a
                .iter()
                .zip(w)
                .map(|(&a, &w)| a as i8 as i32 * w as i32)
                .sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GROUP, MR, NR};
    use core::arch::x86_64::__m256i;
    use pulp::x86::V3;

    pub(super) struct Tile<'a> {
        pub simd: V3,
        pub a: &'a [u8],
        pub k: usize,
        pub panel: &'a [i8],
        pub out: &'a mut [[i32; NR]; MR],
    }

    impl pulp::WithSimd for Tile<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) {
            let (avx, avx2) = (self.simd.avx, self.simd.avx2);
            let groups = self.k / GROUP;
            let ones = avx._mm256_set1_epi16(1);
            let mut acc = [[avx._mm256_setzero_si256(); 2]; MR];
            let panel = &self.panel.as_chunks::<{ NR * GROUP }>().0[..groups];
            let rows: [&[[u8; GROUP]]; MR] = std::array::from_fn(|r| {
                &self.a[r * self.k..(r + 1) * self.k].as_chunks::<GROUP>().0[..groups]
            });
            for (g, w) in panel.iter().enumerate() {
                let (low, high) = w.split_at(NR * GROUP / 2);
                let low: __m256i = pulp::cast(<[i8; 32]>::try_from(low).unwrap());
                let high: __m256i = pulp::cast(<[i8; 32]>::try_from(high).unwrap());
                for (acc, row) in acc.iter_mut().zip(rows) {
                    // The row's four bytes in every lane, against four weights per
                    // channel: pairs of products in 16 bits, then lanes of 32 bits.
                    let x = avx._mm256_set1_epi32(i32::from_le_bytes(row[g]));
                    let sum = |w| avx2._mm256_madd_epi16(avx2._mm256_maddubs_epi16(x, w), ones);
                    acc[0] = avx2._mm256_add_epi32(acc[0], sum(low));
                    acc[1] = avx2._mm256_add_epi32(acc[1], sum(high));
                }
            }
            for (out, acc) in self.out.iter_mut().zip(acc) {
                let low: [i32; 8] = pulp::cast(acc[0]);
                let high: [i32; 8] = pulp::cast(acc[1]);
                out[..8].copy_from_slice(&low);
                out[8..].copy_from_slice(&high);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }
        fn float(&mut self) -> f32 {
            (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
        }
    }

    /// Every kernel's tiles against a plain triple loop, over the full range of
    /// activation and weight levels, the extremes included.
    #[test]
    fn kernels_match_a_scalar_reference() {
        let mut rng = Rng(1);
        type Fill = fn(&mut Rng) -> (i32, i32);
        let fills: [Fill; 4] = [
            |r| (r.int(-A_MAX, A_MAX), r.int(-W_MAX, W_MAX)),
            |_| (A_MAX, W_MAX),
            |_| (A_MAX, -W_MAX),
            |r| (-A_MAX, if r.next() % 2 == 0 { W_MAX } else { -W_MAX }),
        ];
        for inp in [1usize, 3, 4, 17, 64, 769] {
            let k = inp.next_multiple_of(GROUP);
            for fill in fills {
                let mut acts = vec![vec![0i32; k]; MR];
                let mut levels = vec![0i8; NR * k];
                for i in 0..inp {
                    for row in acts.iter_mut() {
                        row[i] = fill(&mut rng).0;
                    }
                    for c in 0..NR {
                        levels[c * k + i] = fill(&mut rng).1 as i8;
                    }
                }
                let mut want = [[0i32; NR]; MR];
                for (r, row) in acts.iter().enumerate() {
                    for c in 0..NR {
                        want[r][c] = (0..k).map(|i| row[i] * levels[c * k + i] as i32).sum();
                    }
                }
                for kernel in Kernel::available() {
                    let zero = kernel.zero();
                    let bytes: Vec<u8> = acts.iter().flatten().map(|&l| (l + zero) as u8).collect();
                    let mut got = [[0i32; NR]; MR];
                    kernel.tile(&bytes, k, &kernel.pack(levels.clone(), k), &mut got);
                    // The sums hold the zero point's share: zero × the levels' sum.
                    for (c, column) in levels.chunks_exact(k).enumerate() {
                        let offset = zero * column.iter().map(|&l| l as i32).sum::<i32>();
                        for row in got.iter_mut() {
                            row[c] -= offset;
                        }
                    }
                    assert_eq!(got, want, "{kernel:?}, {inp} inputs");
                }
            }
        }
    }

    #[test]
    fn quantization_round_trips() {
        let mut rng = Rng(2);
        for max in [W_MAX, A_MAX] {
            for n in [1, 5, 768] {
                let x: Vec<f32> = (0..n).map(|_| rng.float() * 30.0).collect();
                let mut levels = vec![0; n];
                let scale = quantize(&x, max, &mut levels);
                let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!((scale * max as f32 - amax).abs() <= amax * 1e-6);
                for (&l, &v) in levels.iter().zip(&x) {
                    assert!(l.abs() <= max);
                    // Within half a step, and the largest value lands on ±max.
                    assert!((l as f32 * scale - v).abs() <= scale * 0.5001, "{v} {l}");
                    if v.abs() == amax {
                        assert_eq!(l.abs(), max);
                    }
                }
            }
            // Halves round to even.
            let mut l = [0; 4];
            quantize(&[max as f32, 2.5, -3.5, 0.5], max, &mut l);
            assert_eq!(l, [max, 2, -4, 0]);
            let mut levels = vec![7; 4];
            assert_eq!(quantize(&[0.0; 4], max, &mut levels), 0.0);
            assert_eq!(levels, [0; 4]);
        }
    }

    #[test]
    fn the_rotation_is_orthogonal() {
        let mut rng = Rng(4);
        for block in [1, 2, 8, 256] {
            let x: Vec<f32> = (0..3 * block).map(|_| rng.float()).collect();
            let w: Vec<f32> = (0..3 * block).map(|_| rng.float()).collect();
            let (mut hx, mut hw) = (x.clone(), w.clone());
            hadamard(&mut hx, block);
            hadamard(&mut hw, block);
            let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
            assert!((dot(&x, &w) - dot(&hx, &hw)).abs() < 1e-4);
            // Its own inverse.
            hadamard(&mut hx, block);
            assert!(x.iter().zip(&hx).all(|(a, b)| (a - b).abs() < 1e-5));
        }
        // One large value spreads evenly over its block.
        let mut x = vec![0f32; 256];
        x[3] = 16.0;
        hadamard(&mut x, 256);
        assert!(x.iter().all(|v| v.abs() == 1.0));
        assert_eq!(rotation_block(768), 256);
        assert_eq!(rotation_block(24), 8);
        assert_eq!(rotation_block(13), 1);
    }

    /// A layer's outputs against f64 arithmetic, within the error the quantization
    /// allows, and identical from every kernel at every thread count.
    #[test]
    fn layers_are_close_to_f32_and_identical_across_kernels() {
        let mut rng = Rng(3);
        for (rows, out, inp) in [(1, 1, 1), (5, 17, 13), (9, 48, 64), (70, 33, 130)] {
            let x: Vec<f32> = (0..rows * inp).map(|_| rng.float() * 4.0).collect();
            let w: Vec<f32> = (0..out * inp).map(|_| rng.float() * 0.1).collect();
            let b: Vec<f32> = (0..out).map(|_| rng.float()).collect();
            let mut first: Option<Vec<f32>> = None;
            for kernel in Kernel::available() {
                let layer = Int8Linear::with_kernel(kernel, &w, Some(&b), out, inp);
                for threads in [1, 3] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let mut y = vec![f32::NAN; rows * out];
                    pool.install(|| layer.forward(&x, &mut y));
                    match &first {
                        None => first = Some(y),
                        Some(f) => assert_eq!(f, &y, "{kernel:?}, {threads} threads"),
                    }
                }
            }
            let y = first.unwrap();
            let block = rotation_block(inp);
            for m in 0..rows {
                let mut xr = x[m * inp..(m + 1) * inp].to_vec();
                hadamard(&mut xr, block);
                let sa = abs_max(&xr) / A_MAX as f32;
                for n in 0..out {
                    let mut wr = w[n * inp..(n + 1) * inp].to_vec();
                    hadamard(&mut wr, block);
                    let sw = abs_max(&wr) / W_MAX as f32;
                    let exact = (0..inp)
                        .map(|i| x[m * inp + i] as f64 * w[n * inp + i] as f64)
                        .sum::<f64>()
                        + b[n] as f64;
                    // |x w − x̂ ŵ| ≤ Σ |x| |Δw| + |ŵ| |Δx| in the rotated basis, with each
                    // |Δ| at most half a step.
                    let bound = xr
                        .iter()
                        .zip(&wr)
                        .map(|(a, w)| {
                            a.abs() as f64 * sw as f64 / 2.0
                                + (w.abs() + sw / 2.0) as f64 * sa as f64 / 2.0
                        })
                        .sum::<f64>()
                        * 1.001
                        + 1e-4;
                    let got = y[m * out + n] as f64;
                    assert!(
                        (got - exact).abs() <= bound,
                        "{got} vs {exact} (bound {bound})"
                    );
                }
            }
        }
    }
}
