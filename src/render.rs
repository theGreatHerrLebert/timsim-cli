//! The streaming frame render — sweep-line core, plus an **independent** reference render used only
//! to prove the sweep correct.
//!
//! The production path is [`stream_render`]: a 1-D temporal sweep that, for each frame, holds an
//! active set (min-heap on `frame_end`), accumulates every active ion's contribution into a sparse
//! per-frame `(scan, tof)` buffer, hands that buffer to a callback, and drops it. Its working set is
//! bounded by the elution window, not the run length (see `docs/v2-design/TIMSIM_V2_RENDER.md` §7 in
//! the [timsim-necro](https://github.com/theGreatHerrLebert/timsim-necro) repo — the v2 render design
//! docs live there, not here).
//!
//! # Why a second, independent render lives here
//!
//! The load-bearing correctness claim is "the sweep emits each ion's mass **exactly once** — no
//! double-count across the frames it is active in, no leak, no off-by-one at a window edge." A
//! conservation check that reconstructs the expected mass from the *same* frame/scan partitioning and
//! index math the sweep uses is only a *consistency* check: a fault duplicated in both paths passes.
//!
//! So [`reference_render`] is written to share **nothing** with the sweep except the pure Gaussian
//! weight (which is physics, not indexing, and is unit-tested on its own): it is **ion-major** (the
//! sweep is frame-major), it discovers each ion's frame window by a direct `fs..=fe` loop (the sweep
//! discovers it through heap enter/leave against a moving frame cursor), it uses no active set, no
//! per-frame buffer, and no input sort. If the sweep's heap lifetime logic drops or duplicates a
//! frame, the two renders disagree at that bin — and the tests below compare **every bin**, not just
//! totals. The metamorphic tests (duplicate → exactly 2×, permute-order invariance, chunk-union
//! linearity) catch the bugs a single reference render can't.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// The acquisition geometry a render needs: the frame/scan grid and the peak widths (as Gaussian
/// sigmas in frame/scan units) with a truncation radius. This is the render-time image of the
/// portable `[0,1]` elution/mobility shapes — the gradient/ramp mapping happens upstream.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub n_frames: u32,
    pub n_scans: u32,
    pub sigma_frames: f64,
    pub sigma_scans: f64,
    /// Truncate each peak at this many sigma (the `target_p` analog).
    pub n_sigma: f64,
    /// The CHROMATOGRAPHIC peak shape. The mobility (scan) axis is always Gaussian — only the
    /// elution axis has a tail. See [`PeakShape`].
    pub shape: PeakShape,
}

// ---------------------------------------------------------------------------------------------
// Chromatographic peak shape
// ---------------------------------------------------------------------------------------------

/// Which shape the **elution** term uses.
///
/// # !!! CHANGING THE DEFAULT SILENTLY STALES EVERY CACHED RENDER !!!
///
/// The default is [`PeakShape::Emg`] (v1 parity). necroflow fingerprints a node on its **command
/// string**, and `--peak-shape` defaults to `emg` without appearing in that string — so a render
/// cached before this change was produced with the OLD symmetric Gaussian and necroflow **cannot
/// tell**. Cached artifacts under `work/nodes/render_a2/` are stale-but-undetected. They must be
/// invalidated by hand (or the arm re-rendered with an explicit `--peak-shape` in the command, which
/// changes the fingerprint and forces a rebuild). See `PEAK_SHAPE.md`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PeakShape {
    /// Symmetric Gaussian — v2's historical shape, and what every cached render used.
    Gaussian,
    /// v1's exponentially modified Gaussian (Gaussian convolved with a one-sided exponential tail).
    Emg(Emg),
}

/// v1's EMG, reduced to the one dimensionless parameter the shape actually depends on.
///
/// v1 (`mscore::algorithm::utility::emg_function`, driven by
/// `imspy_simulation/timsim/jobs/simulate_frame_distributions_emg.py`) parameterises the peak as
/// `(mu, sigma, lambda)` and samples `sigma` and a tailing factor `k` from scaled Beta
/// distributions, then sets `lambda = 1 / (k * sigma)` (`simulate_frame_distributions_emg.py:287`).
///
/// Because `k = 1 / (sigma * lambda)` is **dimensionless**, it carries across v1's seconds axis and
/// v2's frame axis unchanged — which is why the render can adopt v1's tailing without knowing the
/// cycle time. Substituting `z = (x - mu) / sigma` collapses the EMG CDF to a function of `z` and
/// `k` alone (see [`emg_cdf_std`]), so `sigma` stays exactly the width knob it already was
/// (`--sigma-frames`) and `k` is the only new number.
///
/// `k -> 0` is the Gaussian limit (infinitely fast tail decay); larger `k` means a longer tail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emg {
    /// v1's tailing factor `k = 1 / (sigma * lambda)` = (tail time constant) / sigma.
    pub k: f64,
    /// Mode of `EMG(mu = 0, sigma = 1, lambda = 1/k)`, in units of sigma.
    ///
    /// v1 anchors the peak at its **mode** (the RT predictor's output is treated as the apex) and
    /// solves backwards for `mu` via `estimate_mu_from_mode_emg`
    /// (`simulate_frame_distributions_emg.py:146`). The render is handed an `apex_frame`, so it must
    /// do the same or every peptide elutes systematically late. Precomputed once — the mode search
    /// must never run in the render's inner loop.
    mode_offset: f64,
    /// How far past `n_sigma * sigma` the right-hand truncation must reach to hold as much tail mass
    /// as the Gaussian leaves beyond `n_sigma`. In units of sigma.
    tail_reach: f64,
    /// The standardised PDF's value AT the mode. Divides out in [`elution_ordinate`] so an EMG peak,
    /// like the Gaussian one, has height exactly 1.0 at its apex.
    peak_pdf: f64,
}

/// v1's default tailing factor: the **mean** of the distribution v1 draws `k` from.
///
/// v1 samples `k = k_lower + Beta(k_alpha, k_beta) * (k_upper - k_lower)` with
/// `k_lower_rt = 0`, `k_upper_rt = 10`, `k_alpha_rt = 1`, `k_beta_rt = 20`
/// (`imspy_simulation/timsim/simulator.py:410-424`, and the shipped `configs/config.toml`), so
/// `E[k] = 10 * 1/(1+20) = 10/21`.
///
/// Cross-check against a measured v1 run on a 1861.3 s gradient: v1's auto-derived width is
/// `sigma = gradient/3600 * 0.75 + 1.125 = 1.5128 s` and the observed tail constant was 0.72 s,
/// giving `k = 0.72 / 1.5128 = 0.476` — i.e. exactly 10/21.
pub const V1_DEFAULT_EMG_K: f64 = 10.0 / 21.0;

/// Numerical-Recipes `erfc`, scaled by `exp(x^2)` — i.e. `erfcx(x) = exp(x^2) * erfc(x)`.
///
/// This is **v1's own erf** (`mscore::algorithm::utility::erf`, the same rational/Chebyshev form),
/// not the Abramowitz-Stegun 7.1.26 used by [`erf`] above. Reusing v1's approximation keeps the EMG
/// numerics as close to v1's as they can be; the Gaussian path keeps its A&S `erf` untouched so its
/// output stays bit-for-bit what it always was.
///
/// The `exp(x^2)` cancels analytically against the `exp(-x*x - 1.26551223 + ...)` inside the NR
/// form, so this **cannot overflow** — which is the whole reason the EMG below is written in terms
/// of `erfcx` rather than `exp(...) * erfc(...)` (that product overflows to `inf * 0 = NaN` in the
/// left tail, and v1 dodges it only by never evaluating there).
fn erfcx_nr_nonneg(x: f64) -> f64 {
    debug_assert!(x >= 0.0);
    let t = 1.0 / (1.0 + 0.5 * x);
    t * (-1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp()
}

/// The EMG's tail term, `exp(-z^2/2) * erfcx((1/k - z)/sqrt(2))`, evaluated **as a product** so the
/// two halves' exponents cancel analytically instead of overflowing.
///
/// Writing it as `erfcx(w)` alone is not enough. `erfcx` is only bounded for `w >= 0`; on the far
/// right of the peak (`z > 1/k`) `w` goes negative and the reflection `erfcx(w) = 2*exp(w^2) -
/// erfcx(-w)` overflows to `inf` — which the `exp(-z^2/2)` factor then multiplies by zero, giving
/// `NaN` weights out in the tail. Folding the factor in first turns `exp(w^2 - z^2/2)` into
/// `exp((1/(2k) - z)/k)`, which for `z > 1/k` is a small number, not an overflow.
#[inline]
fn emg_tail_term(z: f64, k: f64) -> f64 {
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    let w = (1.0 / k - z) / SQRT2;
    if w >= 0.0 {
        (-0.5 * z * z).exp() * erfcx_nr_nonneg(w)
    } else {
        2.0 * ((1.0 / (2.0 * k) - z) / k).exp() - (-0.5 * z * z).exp() * erfcx_nr_nonneg(-w)
    }
}

/// `erfc` via the same Numerical-Recipes form (used for the Gaussian half of the EMG CDF).
fn erfc_nr(x: f64) -> f64 {
    let tau = erfcx_nr_nonneg(x.abs()) * (-x * x).exp();
    if x >= 0.0 { tau } else { 2.0 - tau }
}

/// The **standardised** EMG CDF: `P(Z <= z)` for `Z = (X - mu) / sigma` with `lambda = 1/(k*sigma)`.
///
/// Derived from v1's PDF (`emg_function`) in closed form rather than v1's 1000-step Riemann sum —
/// the render evaluates this once per ion per frame, so a quadrature is not affordable. The identity
/// used is that the EMG satisfies `f' + lambda*f = lambda * phi_sigma(x - mu)`, hence
///
/// ```text
///   F(x) = Phi((x-mu)/sigma) - f(x)/lambda
/// ```
///
/// and, substituting `z` and folding the `exp(x^2)` into `erfcx` (see [`erfcx_nr_nonneg`]),
///
/// ```text
///   F(z) = 0.5 * [ erfc(-z/sqrt2) - exp(-z^2/2) * erfcx((1/k - z)/sqrt2) ]
/// ```
///
/// which is scale-free, overflow-free, and reduces to the Gaussian CDF as `k -> 0`.
#[inline]
pub fn emg_cdf_std(z: f64, k: f64) -> f64 {
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    let _ = SQRT2;
    0.5 * (erfc_nr(-z / SQRT2) - emg_tail_term(z, k))
}

/// Standardised EMG PDF (in `z`), up to the `1/sigma` Jacobian. Only used to locate the mode.
#[inline]
fn emg_pdf_std(z: f64, k: f64) -> f64 {
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    // lambda*sigma = 1/k, and the prefactor lambda/2 is a constant in z — dropped, since only the
    // ARGMAX is wanted.
    let _ = SQRT2;
    emg_tail_term(z, k)
}

impl Emg {
    /// Build the shape for tailing factor `k`, truncated at `n_sigma` on the Gaussian side.
    ///
    /// Both derived constants are computed here, once, because both are far too expensive for the
    /// render's inner loop.
    pub fn new(k: f64, n_sigma: f64) -> Emg {
        let k = k.max(1e-12); // k = 0 is the Gaussian limit; keep 1/k finite.

        // The mode, by golden-section search on the (unimodal) standardised PDF. The mode of an EMG
        // lies between mu and the mean mu + 1/lambda (z = k), so [-1, k+1] brackets it with room.
        let (mut lo, mut hi) = (-1.0f64, k + 1.0);
        const INV_PHI: f64 = 0.618_033_988_749_894_9;
        let (mut c, mut d) = (hi - (hi - lo) * INV_PHI, lo + (hi - lo) * INV_PHI);
        let (mut fc, mut fd) = (emg_pdf_std(c, k), emg_pdf_std(d, k));
        for _ in 0..200 {
            if fc > fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - (hi - lo) * INV_PHI;
                fc = emg_pdf_std(c, k);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + (hi - lo) * INV_PHI;
                fd = emg_pdf_std(d, k);
            }
        }
        let mode_offset = 0.5 * (lo + hi);

        // Right-tail reach. The Gaussian truncation at n_sigma leaves p = 0.5*erfc(n_sigma/sqrt2) of
        // the mass outside; the exponential tail (time constant k, in units of sigma) falls to that
        // same fraction after k*ln(1/p). Matching the two keeps `--n-sigma` meaning what it meant.
        let p = 0.5 * erfc_nr(n_sigma / std::f64::consts::SQRT_2);
        let tail_reach = if p > 0.0 && p < 1.0 { k * (1.0 / p).ln() } else { 0.0 };

        Emg { k, mode_offset, tail_reach, peak_pdf: emg_pdf_std(mode_offset, k) }
    }

    /// Where this shape's peak sits relative to `mu`, in units of sigma — v1's `mode - mu`.
    /// Exposed so the v1-parity test can check it against v1's own `erfcxinv` inversion.
    pub fn mode_offset(&self) -> f64 {
        self.mode_offset
    }

    /// Mass between `a` and `b`, for a peak whose **mode** (not `mu`) sits at `apex`.
    #[inline]
    fn frac(&self, a: f64, b: f64, apex: f64, sigma: f64) -> f64 {
        // v1 anchors on the mode and solves for mu; here mu = apex - sigma*mode_offset, so the
        // standardised coordinate is z = (x - mu)/sigma = (x - apex)/sigma + mode_offset.
        let z = |x: f64| (x - apex) / sigma + self.mode_offset;
        emg_cdf_std(z(b), self.k) - emg_cdf_std(z(a), self.k)
    }
}

/// Mass of the elution peak between `a` and `b` for a peak apexing at `apex`. Dispatches on shape;
/// the [`PeakShape::Gaussian`] arm is character-for-character the original [`gauss_frac`] call, so
/// `--peak-shape gaussian` output is bit-identical to the pre-EMG binary.
#[inline]
pub fn elution_frac(a: f64, b: f64, apex: f64, sigma: f64, shape: &PeakShape) -> f64 {
    match shape {
        PeakShape::Gaussian => gauss_frac(a, b, apex, sigma),
        PeakShape::Emg(e) => e.frac(a, b, apex, sigma),
    }
}

/// Peak **ordinate** (height), normalised to 1.0 at the apex.
///
/// The Bruker writer integrates the peak over each frame ([`elution_frac`]); the Thermo and SCIEX
/// writers instead sample the curve's height at the scan's timestamp. Same shape, different
/// convention — so the shape switch has to exist in both forms or `--peak-shape emg` would silently
/// mean nothing on two of the three writers. `t`, `apex` and `sigma` are in whatever unit the caller
/// uses (seconds for Thermo/SCIEX); the EMG's `k` is dimensionless, so it needs no conversion.
#[inline]
pub fn elution_ordinate(t: f64, apex: f64, sigma: f64, shape: &PeakShape) -> f64 {
    match shape {
        // Bit-for-bit the expression the Thermo/SCIEX writers used before the switch existed.
        PeakShape::Gaussian => (-((t - apex).powi(2)) / (2.0 * sigma * sigma)).exp(),
        PeakShape::Emg(e) => emg_pdf_std((t - apex) / sigma + e.mode_offset, e.k) / e.peak_pdf,
    }
}

/// Truncation half-widths `(left, right)` around the apex, in frames. Symmetric for a Gaussian;
/// right-extended for an EMG, whose tail would otherwise be clipped by the symmetric window.
#[inline]
pub fn elution_half_widths(sigma: f64, n_sigma: f64, shape: &PeakShape) -> (f64, f64) {
    match shape {
        PeakShape::Gaussian => {
            let h = n_sigma * sigma;
            (h, h)
        }
        PeakShape::Emg(e) => (
            (n_sigma + e.mode_offset) * sigma,
            (n_sigma + e.tail_reach - e.mode_offset) * sigma,
        ),
    }
}

/// One ion to render: an elution apex (in frames), a mobility apex (in scans), a total abundance,
/// and the `(tof, relative-intensity)` peaks it deposits at that locus. MS1 isotope envelopes and
/// MS2 fragment lists both reduce to this shape — the render does not care which.
#[derive(Clone, Debug)]
pub struct Ion {
    pub apex_frame: f64,
    pub scan_center: f64,
    pub abundance: f64,
    pub peaks: Vec<(u32, f32)>,
}

/// erf via Abramowitz-Stegun 7.1.26 (max error ~1.5e-7). The two renders share this, so its error
/// cancels in their bin-for-bin comparison; its absolute accuracy is pinned by [`tests`] separately.
fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

/// Mass of a Gaussian(mean, sigma) between `a` and `b` — an exact CDF difference over the bin.
/// Pure physics; both renders call it identically.
#[inline]
pub fn gauss_frac(a: f64, b: f64, mean: f64, sigma: f64) -> f64 {
    let z = |x: f64| 0.5 * (1.0 + erf((x - mean) / (sigma * std::f64::consts::SQRT_2)));
    z(b) - z(a)
}

/// The frame window `[frame_start, frame_end]` an ion is active in — the **production** derivation
/// (truncate then clamp). [`reference_render`] deliberately recomputes this a different way so a bug
/// here cannot hide.
#[inline]
/// The bin range whose *centres* fall within ±`n_sigma·sigma` of the apex. NOTE: bins are selected by
/// centre, not by interval overlap, so for a very NARROW peak (`sigma_frames` ≲ 1 bin) this can select a
/// single bin that misses the bin actually holding most of the mass — the emitted fraction then depends
/// on `sigma`/`n_sigma` rather than being a fixed truncation. Harmless at the widths we run
/// (`sigma_frames`≈12, `sigma_scans`≈6, many bins across the peak) and it does NOT distort the
/// precursor↔fragment ratio (MS1 and MS2 share these weights), but if sub-bin peaks are ever needed,
/// select bins whose *intervals* overlap the support and renormalise. Same caveat in [`scan_window`].
fn active_window(apex_frame: f64, g: &Geometry) -> (u32, u32) {
    let (left, right) = elution_half_widths(g.sigma_frames, g.n_sigma, &g.shape);
    let start = (apex_frame - left).max(0.0) as u32;
    let end = ((apex_frame + right) as u32).min(g.n_frames - 1);
    (start, end)
}

/// Per-frame emission the callback sees: the frame index, the active-set size at that frame (for the
/// memory bound), and the sparse `(scan, tof) -> intensity` buffer. The buffer is borrowed and
/// dropped right after — the callback must not retain it if the streaming memory property is to hold.
pub struct FrameEmission<'a> {
    pub frame: u32,
    pub active: usize,
    pub buffer: &'a HashMap<(u32, u32), f64>,
}

/// The streaming sweep-line render. Calls `emit` once per frame that has any active ion, with that
/// frame's sparse buffer, then clears it. Working set stays bounded by the elution window.
///
/// This is the single production render path — the benchmark and [`sweep_render`] both drive it, so
/// the code the tests exercise is the code that runs.
pub fn stream_render<F: FnMut(FrameEmission)>(ions: &[Ion], g: &Geometry, emit: F) {
    stream_render_range(ions, g, 0, g.n_frames, emit)
}

/// [`stream_render`] restricted to the frame sub-range `[frame_lo, frame_hi)`. This is the unit of
/// **parallel render-by-chunk**: contiguous frame ranges are disjoint in their *output*, so K chunks
/// render on K cores and their emissions concatenate — no summing, no double-emit. A boundary ion
/// simply appears in the active set of both chunks it straddles (a read-only input), but each chunk
/// emits only its own frames. Correctness is pinned by `frame_range_partition_equals_whole`.
///
/// Starting the sweep at `frame_lo` with a fresh cursor is what rebuilds the active set correctly: the
/// first iteration pushes every ion with `frame_start <= frame_lo` and then pops those already expired
/// (`frame_end < frame_lo`), leaving exactly the ions alive at `frame_lo`. Give each chunk only the
/// ions overlapping its range (bucketed by the caller) so that initial push is cheap.
pub fn stream_render_range<F: FnMut(FrameEmission)>(
    ions: &[Ion],
    g: &Geometry,
    frame_lo: u32,
    frame_hi: u32,
    mut emit: F,
) {
    // A run must have at least one frame and one scan; otherwise `n_frames - 1` / `n_scans - 1`
    // underflow (panic in debug, a `u32::MAX` loop in release). The CLI also rejects zero, but the
    // library API guards itself.
    if g.n_frames == 0 || g.n_scans == 0 {
        return;
    }
    let windows: Vec<(u32, u32)> = ions.iter().map(|io| active_window(io.apex_frame, g)).collect();

    // Enter in frame_start order; sort indices, not the data (§3.6).
    let mut order: Vec<usize> = (0..ions.len()).collect();
    order.sort_unstable_by_key(|&i| windows[i].0);

    let mut active: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut cursor = 0usize;
    let mut buf: HashMap<(u32, u32), f64> = HashMap::new();

    for frame in frame_lo..frame_hi {
        while cursor < order.len() && windows[order[cursor]].0 <= frame {
            let idx = order[cursor];
            active.push(Reverse((windows[idx].1, idx)));
            cursor += 1;
        }
        while let Some(&Reverse((fe, _))) = active.peek() {
            if fe < frame {
                active.pop();
            } else {
                break;
            }
        }
        if active.is_empty() {
            continue;
        }

        let f = frame as f64;
        for &Reverse((_, idx)) in active.iter() {
            let io = &ions[idx];
            let ew = elution_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames, &g.shape);
            if ew <= 0.0 {
                continue;
            }
            let s_lo = (io.scan_center - g.n_sigma * g.sigma_scans).max(0.0) as u32;
            let s_hi = ((io.scan_center + g.n_sigma * g.sigma_scans) as u32).min(g.n_scans - 1);
            for scan in s_lo..=s_hi {
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = io.abundance * ew * mw;
                for &(tof, iv) in &io.peaks {
                    let val = base * iv as f64;
                    if val <= 0.0 {
                        continue;
                    }
                    *buf.entry((scan, tof)).or_insert(0.0) += val;
                }
            }
        }

        emit(FrameEmission { frame, active: active.len(), buffer: &buf });
        buf.clear();
    }
}

/// Per-frame emission for the **flat** accumulator: a `(scan, tof, value)` list that may contain
/// duplicate `(scan, tof)` keys (co-eluting ions are appended, not summed on the fly). The consumer
/// dedups — which the real TDF block encoder does anyway — so this trades accumulate-time hashing for
/// a single dedup at encode time.
pub struct FlatEmission<'a> {
    pub frame: u32,
    pub active: usize,
    pub triples: &'a [(u32, u32, f64)],
}

/// Like [`stream_render`], but accumulates into a flat `Vec<(scan, tof, value)>` (append, no hashing)
/// instead of a per-frame `HashMap`. Bin-identical after dedup (proved by [`tests`]). Whether this or
/// the HashMap path is faster end-to-end is exactly what the throughput benchmark measures.
pub fn stream_render_flat<F: FnMut(FlatEmission)>(ions: &[Ion], g: &Geometry, emit: F) {
    stream_render_flat_range(ions, g, 0, g.n_frames, emit)
}

/// [`stream_render_flat`] restricted to `[frame_lo, frame_hi)` — the flat-accumulator unit of
/// parallel render-by-chunk. See [`stream_render_range`] for the partition correctness argument.
pub fn stream_render_flat_range<F: FnMut(FlatEmission)>(
    ions: &[Ion],
    g: &Geometry,
    frame_lo: u32,
    frame_hi: u32,
    mut emit: F,
) {
    if g.n_frames == 0 || g.n_scans == 0 {
        return;
    }
    let windows: Vec<(u32, u32)> = ions.iter().map(|io| active_window(io.apex_frame, g)).collect();
    let mut order: Vec<usize> = (0..ions.len()).collect();
    order.sort_unstable_by_key(|&i| windows[i].0);

    let mut active: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut cursor = 0usize;
    let mut buf: Vec<(u32, u32, f64)> = Vec::new();

    for frame in frame_lo..frame_hi {
        while cursor < order.len() && windows[order[cursor]].0 <= frame {
            active.push(Reverse((windows[order[cursor]].1, order[cursor])));
            cursor += 1;
        }
        while let Some(&Reverse((fe, _))) = active.peek() {
            if fe < frame {
                active.pop();
            } else {
                break;
            }
        }
        if active.is_empty() {
            continue;
        }

        let f = frame as f64;
        for &Reverse((_, idx)) in active.iter() {
            let io = &ions[idx];
            let ew = elution_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames, &g.shape);
            if ew <= 0.0 {
                continue;
            }
            let s_lo = (io.scan_center - g.n_sigma * g.sigma_scans).max(0.0) as u32;
            let s_hi = ((io.scan_center + g.n_sigma * g.sigma_scans) as u32).min(g.n_scans - 1);
            for scan in s_lo..=s_hi {
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                let base = io.abundance * ew * mw;
                for &(tof, iv) in &io.peaks {
                    let val = base * iv as f64;
                    if val <= 0.0 {
                        continue;
                    }
                    buf.push((scan, tof, val));
                }
            }
        }

        emit(FlatEmission { frame, active: active.len(), triples: &buf });
        buf.clear();
    }
}

/// Drive [`stream_render`] and materialise the whole `(frame, scan, tof) -> intensity` cube. For
/// **tests only** — this defeats the streaming memory property on purpose, so the output can be
/// compared bin-for-bin against [`reference_render`].
pub fn sweep_render(ions: &[Ion], g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    let mut out = BTreeMap::new();
    stream_render(ions, g, |e| {
        for (&(scan, tof), &v) in e.buffer {
            // A (scan, tof) key is unique within a frame, so no cross-frame collision here.
            out.insert((e.frame, scan, tof), v);
        }
    });
    out
}

/// The **independent** reference render (see module docs). Ion-major, no heap, no per-frame buffer,
/// no input sort; the frame window is a direct `fs..=fe` loop with its bounds recomputed via a
/// different expression than [`active_window`]. Shares only [`gauss_frac`]. Used solely to prove the
/// sweep — never on a real run.
pub fn reference_render(ions: &[Ion], g: &Geometry) -> BTreeMap<(u32, u32, u32), f64> {
    let mut out: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
    let (fleft, fright) = elution_half_widths(g.sigma_frames, g.n_sigma, &g.shape);
    let shalf = g.n_sigma * g.sigma_scans;
    let last_frame = (g.n_frames - 1) as f64;
    let last_scan = (g.n_scans - 1) as f64;

    for io in ions {
        // Independent window derivation: clamp on the reals, then floor — a different code path from
        // active_window's truncate-then-clamp. For non-negative values the two agree, so any
        // divergence signals a real off-by-one in one of them, which is exactly what we want to see.
        let fs = (io.apex_frame - fleft).max(0.0).floor() as u32;
        let fe = (io.apex_frame + fright).min(last_frame).floor() as u32;
        let ss = (io.scan_center - shalf).max(0.0).floor() as u32;
        let se = (io.scan_center + shalf).min(last_scan).floor() as u32;

        for frame in fs..=fe {
            let ew = elution_frac(frame as f64 - 0.5, frame as f64 + 0.5, io.apex_frame, g.sigma_frames, &g.shape);
            if ew <= 0.0 {
                continue;
            }
            for scan in ss..=se {
                let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan_center, g.sigma_scans);
                if mw <= 0.0 {
                    continue;
                }
                for &(tof, iv) in &io.peaks {
                    let val = io.abundance * ew * mw * iv as f64;
                    if val <= 0.0 {
                        continue;
                    }
                    *out.entry((frame, scan, tof)).or_insert(0.0) += val;
                }
            }
        }
    }
    out
}

/// Partition `[0, n_frames)` into `k` contiguous, near-equal frame ranges — the chunks of a parallel
/// render-by-chunk. The remainder is spread across the first ranges so sizes differ by at most one.
pub fn frame_chunks(n_frames: u32, k: usize) -> Vec<(u32, u32)> {
    let k = (k.max(1) as u32).min(n_frames.max(1));
    let base = n_frames / k;
    let rem = n_frames % k;
    let mut out = Vec::with_capacity(k as usize);
    let mut lo = 0u32;
    for i in 0..k {
        let hi = lo + base + if i < rem { 1 } else { 0 };
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// For each chunk, the indices of ions whose active window overlaps that chunk's frame range. An ion
/// straddling a boundary lands in every chunk it touches (correct: each chunk emits only its own
/// frames, so the ion contributes to each without being double-emitted). Bucketing keeps each chunk's
/// sweep O(its ions), not O(all ions).
pub fn bucket_ions(ions: &[Ion], g: &Geometry, chunks: &[(u32, u32)]) -> Vec<Vec<usize>> {
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); chunks.len()];
    for (i, io) in ions.iter().enumerate() {
        let (ws, we) = active_window(io.apex_frame, g);
        for (c, &(lo, hi)) in chunks.iter().enumerate() {
            if ws < hi && we >= lo {
                buckets[c].push(i);
            }
        }
    }
    buckets
}

/// Worst relative difference between two render cubes, and the count of bins present in only one.
/// `(worst_rel, only_in_a, only_in_b)`. Rel diff uses the larger magnitude as denominator so a bin
/// that exists in one render but is ~0 in the other still registers.
pub fn cube_diff(
    a: &BTreeMap<(u32, u32, u32), f64>,
    b: &BTreeMap<(u32, u32, u32), f64>,
) -> (f64, usize, usize) {
    let mut worst = 0.0f64;
    let mut only_a = 0usize;
    for (k, &va) in a {
        match b.get(k) {
            Some(&vb) => {
                let d = (va - vb).abs();
                let denom = va.abs().max(vb.abs());
                if denom > 0.0 {
                    worst = worst.max(d / denom);
                }
            }
            None => only_a += 1,
        }
    }
    let only_b = b.keys().filter(|k| !a.contains_key(*k)).count();
    (worst, only_a, only_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> Geometry {
        Geometry { n_frames: 40, n_scans: 30, sigma_frames: 2.5, sigma_scans: 1.5, n_sigma: 3.0, shape: PeakShape::Gaussian }
    }

    /// A fixture that deliberately exercises the bug-prone cases Codex called out:
    ///  - an ion living across MANY frames (lifetime / enter-leave off-by-one),
    ///  - two co-eluting ions colliding into IDENTICAL (scan, tof) bins (accumulation),
    ///  - an ion pinned against the scan-0 boundary (window clamping),
    ///  - an ion pinned against the last frame (upper clamp),
    ///  - multi-peak envelopes (per-peak tof stepping).
    fn fixture() -> Vec<Ion> {
        vec![
            // long-lived, mid grid, two isotope peaks
            Ion { apex_frame: 20.0, scan_center: 15.0, abundance: 100.0, peaks: vec![(500, 1.0), (504, 0.4)] },
            // co-elutes with the next one at the SAME locus + SAME tof -> bins must add
            Ion { apex_frame: 20.3, scan_center: 15.0, abundance: 50.0, peaks: vec![(500, 1.0)] },
            Ion { apex_frame: 20.1, scan_center: 15.0, abundance: 30.0, peaks: vec![(500, 1.0)] },
            // scan-0 boundary (half the mobility peak is clamped away)
            Ion { apex_frame: 8.0, scan_center: 0.4, abundance: 70.0, peaks: vec![(300, 1.0)] },
            // last-frame boundary
            Ion { apex_frame: 39.2, scan_center: 22.0, abundance: 40.0, peaks: vec![(900, 1.0), (905, 0.2)] },
        ]
    }

    /// The load-bearing test: the streaming sweep and the independent ion-major reference must agree
    /// at EVERY (frame, scan, tof) bin, with no bin present in only one. Bit-for-bit up to float
    /// summation order (~1e-12).
    #[test]
    fn sweep_matches_independent_reference_every_bin() {
        let g = geom();
        let ions = fixture();
        let sweep = sweep_render(&ions, &g);
        let reference = reference_render(&ions, &g);

        assert!(!sweep.is_empty(), "fixture rendered nothing — the test would be vacuous");
        let (worst, only_sweep, only_ref) = cube_diff(&sweep, &reference);
        assert_eq!(only_sweep, 0, "{only_sweep} bins the sweep emitted are absent from the reference");
        assert_eq!(only_ref, 0, "{only_ref} bins the reference emitted are absent from the sweep");
        assert!(worst < 1e-12, "worst per-bin relative diff {worst:.3e} exceeds 1e-12");
    }

    /// Duplicating one ion must exactly double THAT ion's bins and change nothing else. Catches any
    /// cross-ion interference or double-counting in the accumulation.
    #[test]
    fn duplicating_one_ion_doubles_exactly_its_bins() {
        let g = geom();
        let ions = fixture();
        let base = sweep_render(&ions, &g);

        let mut dup = ions.clone();
        dup.push(ions[0].clone()); // duplicate the long-lived ion
        let with_dup = sweep_render(&dup, &g);

        // (with_dup - base) must equal exactly the render of ion 0 alone, every bin.
        let solo = sweep_render(&ions[0..1], &g);
        let mut delta: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        for (k, &v) in &with_dup {
            delta.insert(*k, v - base.get(k).copied().unwrap_or(0.0));
        }
        // drop numerical zeros that fall out of subtraction
        delta.retain(|_, v| v.abs() > 1e-9);
        let (worst, only_delta, only_solo) = cube_diff(&delta, &solo);
        assert_eq!(only_delta, 0, "duplicate delta has {only_delta} bins the solo render lacks");
        assert_eq!(only_solo, 0, "solo render has {only_solo} bins the duplicate delta lacks");
        assert!(worst < 1e-9, "duplicate delta != solo render (worst {worst:.3e})");
    }

    /// The sweep sorts internally, so input order must not change a single bin. Catches any
    /// order-dependence in enter/leave.
    #[test]
    fn permuting_input_order_is_invariant() {
        let g = geom();
        let ions = fixture();
        let canonical = sweep_render(&ions, &g);

        let mut permuted = ions.clone();
        permuted.reverse();
        permuted.swap(0, 2);
        let out = sweep_render(&permuted, &g);

        let (worst, a, b) = cube_diff(&canonical, &out);
        assert_eq!((a, b), (0, 0), "input order changed the bin set ({a}, {b})");
        assert!(worst < 1e-12, "input order changed values (worst {worst:.3e})");
    }

    /// Rendering two arbitrary subsets and summing must equal rendering the whole set — the render is
    /// linear in the ion set, so overlapping chunks compose. Catches leakage between ions sharing a
    /// buffer bin.
    #[test]
    fn chunk_union_equals_whole() {
        let g = geom();
        let ions = fixture();
        let whole = sweep_render(&ions, &g);

        let a = sweep_render(&ions[0..2], &g);
        let b = sweep_render(&ions[2..], &g);
        let mut union: BTreeMap<(u32, u32, u32), f64> = a.clone();
        for (k, &v) in &b {
            *union.entry(*k).or_insert(0.0) += v;
        }

        let (worst, only_whole, only_union) = cube_diff(&whole, &union);
        assert_eq!((only_whole, only_union), (0, 0), "chunk union bin set differs");
        assert!(worst < 1e-12, "chunk union != whole render (worst {worst:.3e})");
    }

    /// Total emitted mass of a lone, interior ion equals abundance × (frame window mass) ×
    /// (scan window mass) × (Σ peak intensities), computed here from first principles with NO render
    /// code. Independent conservation, complementary to the bin-for-bin equality.
    #[test]
    fn lone_interior_ion_conserves_analytic_mass() {
        let g = geom();
        let ion = Ion { apex_frame: 20.0, scan_center: 15.0, abundance: 100.0, peaks: vec![(500, 1.0), (504, 0.4)] };
        let cube = sweep_render(std::slice::from_ref(&ion), &g);
        let emitted: f64 = cube.values().sum();

        let fhalf = g.n_sigma * g.sigma_frames;
        let shalf = g.n_sigma * g.sigma_scans;
        let (fs, fe) = ((ion.apex_frame - fhalf).floor() as i64, (ion.apex_frame + fhalf).floor() as i64);
        let (ss, se) = ((ion.scan_center - shalf).floor() as i64, (ion.scan_center + shalf).floor() as i64);
        let frame_mass: f64 = (fs..=fe).map(|f| gauss_frac(f as f64 - 0.5, f as f64 + 0.5, ion.apex_frame, g.sigma_frames)).sum();
        let scan_mass: f64 = (ss..=se).map(|s| gauss_frac(s as f64 - 0.5, s as f64 + 0.5, ion.scan_center, g.sigma_scans)).sum();
        let peak_sum: f64 = ion.peaks.iter().map(|&(_, iv)| iv as f64).sum();
        let expected = ion.abundance * frame_mass * scan_mass * peak_sum;

        let rel = (emitted - expected).abs() / expected;
        assert!(rel < 1e-9, "emitted {emitted:.6} vs analytic {expected:.6} (rel {rel:.3e})");
    }

    /// Parallel render-by-chunk correctness: rendering contiguous frame ranges (each given only its
    /// bucketed ions) and concatenating the emissions must reproduce the whole render exactly. This is
    /// the invariant the parallel sweep relies on — distinct from the ion-partition invariant
    /// (`chunk_union_equals_whole`), because here the partition is over OUTPUT FRAMES, not ions, so the
    /// pieces concatenate rather than sum. Boundary ions (bucketed into two chunks) must not
    /// double-emit.
    #[test]
    fn frame_range_partition_equals_whole() {
        let g = geom();
        let ions = fixture();
        let whole = sweep_render(&ions, &g);

        // Deliberately uneven chunk count so boundaries fall mid-peak.
        let chunks = frame_chunks(g.n_frames, 4);
        assert!(chunks.len() >= 2, "need multiple chunks to exercise boundaries");
        let buckets = bucket_ions(&ions, &g, &chunks);

        let mut parts: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        for (&(lo, hi), bucket) in chunks.iter().zip(buckets.iter()) {
            let sub: Vec<Ion> = bucket.iter().map(|&i| ions[i].clone()).collect();
            stream_render_range(&sub, &g, lo, hi, |e| {
                for (&(scan, tof), &v) in e.buffer {
                    // Each frame is owned by exactly one chunk, so no cross-chunk key collision.
                    assert!(parts.insert((e.frame, scan, tof), v).is_none(), "frame emitted twice");
                }
            });
        }

        let (worst, only_whole, only_parts) = cube_diff(&whole, &parts);
        assert_eq!((only_whole, only_parts), (0, 0), "frame-partition bin set differs from whole");
        assert!(worst < 1e-12, "frame-partition render diverges from whole (worst {worst:.3e})");
    }

    /// The flat accumulator, once its duplicate `(scan, tof)` keys are summed, must reproduce the
    /// HashMap sweep's cube exactly — otherwise the two accumulators disagree and the throughput
    /// comparison would be between two different renders.
    #[test]
    fn flat_accumulator_dedups_to_the_same_cube() {
        let g = geom();
        let ions = fixture();
        let hashmap_cube = sweep_render(&ions, &g);

        let mut flat_cube: BTreeMap<(u32, u32, u32), f64> = BTreeMap::new();
        stream_render_flat(&ions, &g, |e| {
            for &(scan, tof, v) in e.triples {
                *flat_cube.entry((e.frame, scan, tof)).or_insert(0.0) += v;
            }
        });

        let (worst, only_hm, only_flat) = cube_diff(&hashmap_cube, &flat_cube);
        assert_eq!((only_hm, only_flat), (0, 0), "flat vs hashmap bin set differs");
        assert!(worst < 1e-12, "flat accumulator diverges from hashmap (worst {worst:.3e})");
    }

    /// Pins the shared physics: a Gaussian's ±1σ mass ≈ 0.6827 and its mass over a wide window ≈ 1.
    /// This is the one thing both renders share, so it is verified on its own.
    #[test]
    fn gauss_frac_matches_known_values() {
        // ±1σ around 0 with sigma 1
        let one_sigma = gauss_frac(-1.0, 1.0, 0.0, 1.0);
        assert!((one_sigma - 0.6827).abs() < 1e-3, "±1σ mass {one_sigma}");
        // effectively the whole distribution
        let whole = gauss_frac(-20.0, 20.0, 0.0, 1.0);
        assert!((whole - 1.0).abs() < 1e-6, "full mass {whole}");
    }

    // -----------------------------------------------------------------------------------------
    // EMG peak shape
    // -----------------------------------------------------------------------------------------

    /// `--peak-shape gaussian` must be the OLD code path, not merely an equal one: the guarantee we
    /// sell is bit-for-bit reproduction of every render made before the EMG existed.
    #[test]
    fn gaussian_shape_is_bit_identical_to_gauss_frac() {
        let s = PeakShape::Gaussian;
        for &sigma in &[0.5, 2.5, 30.0] {
            for i in -60..60 {
                let f = i as f64 * 0.37;
                let a = elution_frac(f - 0.5, f + 0.5, 4.25, sigma, &s);
                let b = gauss_frac(f - 0.5, f + 0.5, 4.25, sigma);
                assert_eq!(a.to_bits(), b.to_bits(), "sigma={sigma} f={f}: {a} vs {b}");
            }
        }
        // ...and the truncation window must be the untouched symmetric one.
        assert_eq!(elution_half_widths(30.0, 3.0, &s), (90.0, 90.0));
    }

    /// The EMG must actually be a distribution: monotone, spanning [0,1], integrating to 1.
    #[test]
    fn emg_cdf_is_a_proper_cdf() {
        for &k in &[0.05, V1_DEFAULT_EMG_K, 2.0, 9.5] {
            assert!(emg_cdf_std(-40.0, k) < 1e-9, "k={k} left limit");
            assert!((emg_cdf_std(60.0 + 400.0 * k, k) - 1.0).abs() < 1e-6, "k={k} right limit");
            let mut prev = 0.0;
            for i in 0..2000 {
                let z = -10.0 + i as f64 * 0.02;
                let c = emg_cdf_std(z, k);
                assert!(c >= prev - 1e-12, "k={k} not monotone at z={z}");
                assert!((-1e-9..=1.0 + 1e-9).contains(&c), "k={k} out of [0,1] at z={z}: {c}");
                prev = c;
            }
        }
    }

    /// `k -> 0` is the Gaussian limit. This is what makes `k` a *shape* knob orthogonal to `sigma`.
    #[test]
    fn emg_degenerates_to_a_gaussian_as_k_goes_to_zero() {
        // The convergence is first order: F_emg(z) = Phi(z) - k*phi(z) + O(k^2), so the deviation
        // must fall PROPORTIONALLY with k. Asserting that (rather than one fixed epsilon) is what
        // actually pins the limit.
        for &k in &[1e-3, 1e-5, 1e-7] {
            for i in -30..30 {
                let z = i as f64 * 0.25;
                let emg = emg_cdf_std(z, k);
                let gauss = 0.5 * erfc_nr(-z / std::f64::consts::SQRT_2);
                // |phi| <= 0.3990, plus room for the erfc approximation's own ~1e-9 noise.
                let tol = 0.4 * k + 1e-8;
                assert!((emg - gauss).abs() < tol, "k={k} z={z}: emg {emg} vs gauss {gauss}");
            }
        }
    }

    /// The whole point: an EMG peak is TAILED. Mass to the right of the apex must exceed mass to the
    /// left, the excess must grow with `k`, and a Gaussian must show none of it.
    #[test]
    fn emg_peaks_are_right_tailed_and_gaussians_are_not() {
        let sigma = 30.0;
        let apex = 500.0;
        let mass = |shape: &PeakShape, lo: f64, hi: f64| elution_frac(lo, hi, apex, sigma, shape);

        let g = PeakShape::Gaussian;
        let (gl, gr) = (mass(&g, apex - 400.0, apex), mass(&g, apex, apex + 400.0));
        // 1e-7 not 1e-9: the A&S erf backing `gauss_frac` is itself only good to ~1.5e-7.
        assert!((gl - gr).abs() < 1e-7, "Gaussian must be symmetric: {gl} vs {gr}");

        let mut last_ratio = 1.0;
        for &k in &[0.25, V1_DEFAULT_EMG_K, 1.5] {
            let e = PeakShape::Emg(Emg::new(k, 3.0));
            let (l, r) = (mass(&e, apex - 400.0, apex), mass(&e, apex, apex + 400.0));
            assert!(r > l, "k={k}: EMG must carry more mass right of the apex ({r} vs {l})");
            let ratio = r / l;
            assert!(ratio > last_ratio, "k={k}: tailing must increase with k ({ratio} <= {last_ratio})");
            last_ratio = ratio;
        }
    }

    /// The apex must be where the peak actually PEAKS. v1 anchors on the mode and solves back for
    /// `mu`; if the render skipped that inversion every peptide would elute systematically late.
    #[test]
    fn emg_apex_frame_is_the_mode() {
        let sigma = 30.0;
        let apex = 500.0;
        for &k in &[0.1, V1_DEFAULT_EMG_K, 3.0] {
            let e = PeakShape::Emg(Emg::new(k, 3.0));
            let at_apex = elution_frac(apex - 0.5, apex + 0.5, apex, sigma, &e);
            for d in [-90.0, -30.0, -5.0, -1.0, 1.0, 5.0, 30.0, 90.0] {
                let off = elution_frac(apex + d - 0.5, apex + d + 0.5, apex, sigma, &e);
                assert!(off <= at_apex, "k={k}: bin at +{d} ({off}) beats the apex bin ({at_apex})");
            }
        }
    }

    /// The truncation window must actually hold the peak. A symmetric window would clip the tail.
    #[test]
    fn emg_window_is_asymmetric_and_captures_the_mass() {
        let (sigma, n_sigma) = (30.0, 3.0);
        let e = PeakShape::Emg(Emg::new(V1_DEFAULT_EMG_K, n_sigma));
        let (left, right) = elution_half_widths(sigma, n_sigma, &e);
        assert!(right > left, "EMG window must reach further right: {left} / {right}");
        let apex = 10_000.0;
        let captured = elution_frac(apex - left, apex + right, apex, sigma, &e);
        // The Gaussian at 3 sigma keeps 99.73%; the EMG window is built to hold at least as much.
        assert!(captured > 0.997, "window keeps only {captured}");
    }

    /// The ordinate form (Thermo/SCIEX) must agree with the Gaussian bit-for-bit and must peak at 1.
    #[test]
    fn elution_ordinate_matches_both_conventions() {
        let (sigma, apex) = (3.0, 120.0);
        let two_sig2 = 2.0 * sigma * sigma;
        for i in -50..50 {
            let t = apex + i as f64 * 0.3;
            let got = elution_ordinate(t, apex, sigma, &PeakShape::Gaussian);
            let want = (-((t - apex).powi(2)) / two_sig2).exp();
            assert_eq!(got.to_bits(), want.to_bits(), "t={t}");
        }
        for &k in &[0.2, V1_DEFAULT_EMG_K, 2.0] {
            let e = PeakShape::Emg(Emg::new(k, 3.0));
            let top = elution_ordinate(apex, apex, sigma, &e);
            assert!((top - 1.0).abs() < 1e-9, "k={k}: apex height {top} != 1");
            for d in [-20.0, -3.0, 3.0, 20.0] {
                assert!(elution_ordinate(apex + d, apex, sigma, &e) <= top, "k={k} d={d}");
            }
        }
    }
}
