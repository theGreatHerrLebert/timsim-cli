# Chromatographic peak shape — EMG by default

## TL;DR

`timsim-render` (and `-bench`, `-thermo`, `-sciex`) grew `--peak-shape {gaussian|emg}`, **defaulting
to `emg`**: v1's exponentially modified Gaussian instead of the symmetric Gaussian v2 shipped with.
`--peak-shape gaussian` reproduces the old binary **byte for byte**.

> ## ⚠️ READ THIS BEFORE TRUSTING ANY CACHED RENDER
>
> **Changing the default changed render output while leaving the command string untouched, so
> necroflow cannot tell that cached renders are stale.** Every `.d` under
> `/media/hd01/timsim-cohort/work/nodes/render_a2/` was produced with the OLD Gaussian and is
> reported by necroflow as `up_to_date`. Nothing in this change fixes that — see
> [Cache staleness](#cache-staleness) for the measurement and the options.

## Why

v1 models the LC peak as an **exponentially modified Gaussian**; v2 used a symmetric Gaussian. Real
chromatographic peaks tail, so on this axis v2 was the regression, and a head-to-head found that v2's
broader, untailed peak plausibly inflates its recall. Peak shape has to be reconciled before any
v1-vs-v2 fidelity claim means anything.

## Where v1's EMG lives, and what was reused

| piece | v1 location |
| --- | --- |
| PDF | `mscore::algorithm::utility::emg_function` / `emg` (`mscore/src/algorithm/utility.rs`) |
| per-frame weight | `calculate_frame_abundance_emg` — `emg_cdf_range(t - cycle, t, mu, sigma, lambda)`, a 1000-step **left Riemann sum** |
| support window | `calculate_bounds_emg` — binary search on `[mu - 20σ - 2, mu + 60σ]` for `target_p` |
| parameter draw | `imspy_simulation/timsim/jobs/simulate_frame_distributions_emg.py` — `sigma ~ Beta(4,4)` scaled into a gradient band, `k ~ Beta(1,20)*10`, then `lambda = 1/(k*sigma)` |
| mode → mu | `estimate_mu_from_mode_emg` (`erfcxinv`, 10 Newton steps) — v1 treats the predicted RT as the **mode** and solves backwards for `mu` |
| defaults | `imspy_simulation/timsim/simulator.py:410-424` (`k_lower_rt=0, k_upper_rt=10, k_alpha_rt=1, k_beta_rt=20`) |

The reused piece is **v1's parameterisation**, not its code:

* v1's tailing factor `k = 1 / (sigma * lambda)` is **dimensionless**, so it transfers unchanged from
  v1's seconds axis to v2's frame axis. That is why `--emg-k` needs no cycle-time conversion and why
  `sigma` stays exactly the width knob it already was (`--sigma-frames`).
* `--emg-k` defaults to `E[k] = 10/21 = 0.47619`, the mean of v1's own draw. Independently confirmed
  against a real 1861.3 s-gradient v1 run: v1's auto-derived `sigma = gradient/3600*0.75 + 1.125 =
  1.5128 s` and its measured tail constant 0.72 s give `k = 0.476`.
* v2 evaluates the EMG CDF in **closed form** rather than by v1's quadrature (the render evaluates it
  once per ion per frame; a 1000-step Riemann sum is not affordable). Substituting `z = (x-mu)/sigma`
  collapses it to `F(z) = ½[erfc(-z/√2) - exp(-z²/2)·erfcx((1/k - z)/√2)]`, using v1's own
  Numerical-Recipes `erf` coefficients.

## Where it applies

**The render stage owns the elution weight, entirely.** `peptide_rt` carries only an `rt_index`
(a portable apex position); the render maps index → apex frame → per-frame weight. So the switch
lives in the render and nowhere else.

* `src/render.rs` — `PeakShape`, `Emg`, `elution_frac` (bin mass, Bruker), `elution_ordinate`
  (peak height, Thermo/SCIEX), `elution_half_widths` (asymmetric truncation window).
* `src/ms2.rs`, `src/bin/render.rs` — DIA/DDA deposition and the active-set windows.
* `src/dda.rs` — the selection scheme. Its private duplicate Gaussian (with its own `erf`) was
  **removed**: the scheduler ranked precursors with one implementation while the writer deposited
  with another, so a shape change applied to only one would have made DDA select on an intensity it
  never emits.
* `src/bin/render_thermo.rs`, `src/bin/render_sciex.rs` — same shape, ordinate convention.

### Not done

* **Width is untouched.** `--sigma-frames` still defaults to 30 (≈3.16 s at a 0.105 s cycle) against
  v1's ≈1.51 s. This change fixes the *shape* axis only; matching v1's width would need
  `--sigma-frames ≈ 14.3`, which is a separate decision with its own staleness consequences.
* **`k` is global, not per peptide.** v1 draws `k` and `sigma` per peptide. The `peptide_rt` artifact
  **already carries** `rt_sigma_hat` and `rt_k_hat` (unit Beta draws, produced by `rt.py` to match
  v1's defaults) and no consumer reads them. Wiring those through `Ion` is the natural next step to
  full per-peptide parity.

## Verification

| check | result |
| --- | --- |
| `--peak-shape gaussian` vs the pre-EMG binary, real cohort render replayed from its own necroflow command | **byte-identical**; `analysis.tdf_bin` sha256 `aa0bea037e3b8b7383c2a00995bc4a5d0339e17a3d76e8bb92e6b7e3562a8550` (pre-EMG binary verified self-reproducible first) |
| EMG vs **v1's realised per-frame profile** (598 frames, 5 peptides, `k` 0.20–1.91) | max 5.06e-5, median 2.63e-5 — i.e. at v1's own 4-decimal storage granularity |
| mode anchoring: v2's golden-section mode vs v1's `erfcxinv` inversion | agree to 3e-8 … 3e-7 sigma across `k` 0.097–1.91 |
| default (emg) vs pre-EMG binary | differs, as it must |

`tests/emg_v1_parity.rs` pins all three. Its fixture (`tests/data/v1_emg_profiles.json`) is a verbatim
dump of a real v1 run's `synthetic_data.db` — the comparison is against v1's **output**, never against
a re-evaluation of the same formula.

### Shape metrics (measured off the code, `tests::emg_shape_metrics`)

| shape | FWHM | As(10%) |
| --- | --- | --- |
| gaussian | 2.3548 σ | 1.0000 |
| emg, k=0.25 | 2.4194 σ | 1.0176 |
| **emg, k=0.4762 (v1 default)** | **2.5441 σ** | **1.0835** |
| emg, k=1.0 | 2.8909 σ | 1.3622 |
| emg, k=2.0 | 3.5865 σ | 2.0556 |

At v2's default `--sigma-frames 30` / 0.10545 s cycle (σ = 3.163 s) the EMG FWHM is **8.05 s**, against
v1's **3.85 s** (σ = 1.513 s). The 2.09× width gap is unchanged by this work, by design.

### A v1 defect found on the way

**13.9% of peptides (782 / 5645) in the reference v1 run have a truncated elution profile** — their
stored `frame_abundance` sums to well under `target_p = 0.999` (as low as 1e-4) because
`calculate_bounds_emg`'s binary search misses the apex. It is concentrated at small `k` (large
`lambda`, i.e. narrow near-Gaussian peaks) where the peak is tiny next to the `[mu-20σ-2, mu+60σ]`
seed span. Those peptides are effectively rendered with a fraction of their intensity in v1. Not
fixed here; flagged because it affects any v1-as-ground-truth comparison.

## Cache staleness

necroflow's `v2` fingerprint is computed from the **config + command template + input hashes**. It
does **not** hash the binary — the command references
`/scratch/timsim-demo/timsim-cli/target/release/timsim-render` by *path only*. And no cached command
mentions `--peak-shape` or `--sigma-frames`, because neither existed. Therefore:

**rebuilding `timsim-render` with a new default silently changes render output while every
fingerprint stays the same.** All 70 node directories report `state = up_to_date`.

Measured on `/media/hd01/timsim-cohort/work/nodes/render_a2/` (2026-08-09):

| group | nodes | recorded wall | output |
| --- | --- | --- | --- |
| live cohort arms (`max_peptides = 0`, 30 samples `mild_R1..15` / `severe_R1..15`) | 30 | **124.2 h** | **151.9 GB** |
| superseded smaller configs (`max_peptides` 3000 / 400000) | 40 | 1.0 h | 17.5 GB |
| **total** | **70** | **125.2 h** | **169.4 GB** (158 GB on disk) |

* Commands containing `--peak-shape`: **0 / 70**. Containing `--sigma-frames`: **0 / 70**.
* Nodes reporting `up_to_date`: **70 / 70**.
* Re-render cost for the live arms: **124.2 h serial** (~5.2 days) and 151.9 GB rewritten.
  `/media/hd01` has 485 GB free, so a side-by-side re-render fits.
* `truth.parquet` is **identical** between the Gaussian and EMG renders (sha256
  `f0857aa7c2e7f3c705dfa2e1f1f8b5f4d5644ea361b57767131d7d5f48e4f7f4`), so the answer key does not
  record which shape produced the `.d` either. A stale `.d` cannot be detected from its own outputs.

### Options (none applied here)

1. **Do nothing and pass `--peak-shape gaussian` explicitly** in the flow's render command. The 30
   cached arms stay valid; the flag now appears in the command string, so any later change is
   fingerprint-visible. Costs one edit to `timsim_flow.py` and re-fingerprints all 30 arms (they
   would rebuild once).
2. **Adopt EMG and re-render**, again by putting `--peak-shape emg` in the command so the change is
   visible. ~124 h.
3. **Delete the 70 node directories** to force a rebuild. Loses the Gaussian arms irrecoverably and
   still leaves the default invisible to future fingerprints.

Whichever is chosen, **the flag belongs in the command string.** A default that changes output but
not the fingerprint is exactly the failure mode this file exists to document.
