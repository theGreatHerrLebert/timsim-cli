# Render intensity calibration — target, model, and what NOT to touch

The v2 render must produce data that *looks* like real timsTOF. The naive reading of that — "reshape
the rendered intensities to match the real per-peak distribution" — is a **category error** that would
damage the simulator. This document states the corrected framing, the observation model to build, and
the acceptance criteria, after a domain review (Codex, 2026-07-19) caught the error below.

> **The error to avoid:** the real per-peak intensity distribution is *narrow* (~55× dynamic range),
> but that is **not** because real peptide abundances are narrow. It is a property of *measurement* —
> one peptide's ions spread across RT / mobility / isotope / fragment bins, and a hard count floor
> censors the low end. Narrowing the render's **biological abundance** distribution to hit ~55× would
> erase the ~6-order ground-truth abundance axis the whole eval harness exists to test
> (recall-vs-abundance). **Never compress the abundance axis to match a per-peak statistic.**

## Two axes — keep them separate

| axis | what it is | for the render |
|---|---|---|
| **Biological abundance** | `amount × ionization × modform × charge` — the ground truth, ~6 orders | **Held FIXED. Wide. Never calibrated to a per-peak target.** |
| **Measurement / observation** | how an ion's abundance becomes stored per-peak counts: spreading, floor/censoring, count noise, background | **This is what we calibrate.** The narrow per-peak shape is an *emergent output* of this layer, not an input target. |

So "signal calibration" and "noise model" are **not two sequential features** — they are **one instrument
observation model**, built together, because their effects are statistically coupled (adding the missing
near-floor population lowers the pooled median and changes the apparent dynamic range — see below).

## What real data shows — as *combined observables* (not decompositions)

Measured on real `K240723` DIA-PASEF (24-window, 2640 s, m/z 400–1000; 3 replicates, ±4% stable; raw
peaks validated exact vs stored `NumPeaks/MaxIntensity/SummedIntensities`). Reproduce:
`python -m imspy_simulation.timsim.validate.peak_distribution <.d> <n_frames>`.

| per-peak (real) | floor | p50 | p99 | p99.9 | max | dyn (p99.9/p1) | peaks/scan |
|---|---|---|---|---|---|---|---|
| **MS1 precursor** | **21** | 53 | 246 | 1,375 | ~60,000 | ~55× | **~335** |
| **MS2 fragment**  | **21** | 70 | 266 | 1,161 | ~8,600  | ~46× | **~24**  |

**These are combined signal + isotopes + co-elution + background + thresholded noise, pooled over all
retained bins.** They are therefore **not** estimates of peptide abundance and **not** a detector
transfer function. Read them only as: "the observation model, run on a HeLa-like load at this method,
must produce roughly this combined distribution per frame type." Sample- and method-specific.

Current render (same probe, `out/250k/v2_250k.d`): MS1 floor 3 / median 10 / ~11 peaks/scan;
MS2 floor 3 / median 6 / ~1.9 peaks/scan — i.e. too dim in the bulk, floored too low, and **~30× (MS1)
/ ~13× (MS2) too sparse**. The sparsity is the missing measurement layer, not missing abundance.

## MEASURED STATE, 2026-08-11 — supersedes the stale figures above

The `~30x too sparse` numbers above were measured 2026-07-19 on `out/250k/v2_250k.d`, against real
data acquired with a DIFFERENT method. Re-measured properly: 80,000-peptide design (485,029
precursors) rendered against **K240723_002 itself**, so acquisition schema, gradient, window layout
and mobility range are identical and the only difference is real vs simulated.

**Real-vs-real first, so "close" has a definition.** K240723_002 vs its replicate _012:
MS1 333.6 vs 321.6 peaks/scan (3.7% apart), mean intensity/peak 68.4 vs 69.1 (1% apart). That is the
measurement's own noise floor.

| | sim | real K240723_002 | ratio |
|---|---|---|---|
| **MS1 total ion current** | 63.4e9 | 59.3e9 | **1.07x** |
| MS1 peaks | 59.0 M | 866.8 M | 0.068x |
| MS1 intensity/peak | 1073.3 | 68.4 | 15.7x |
| MS2 total ion current | 21.3e9 | 41.0e9 | 0.52x |
| MS2 peaks | 216.9 M | 500.5 M | 0.43x |
| MS2 intensity/peak | 98.4 | 81.9 | 1.2x |

**The headline: the absolute scale is roughly right; the DISTRIBUTION is not.** MS1 total signal is
within 7% of real. The gap is that the (correct) total is packed into 15x too few peaks at 16x the
intensity each — precisely the missing observation model, and precisely NOT an abundance problem.
This is the strongest evidence yet for the "never compress the abundance axis" rule above.

### The 1.07x is an OVERSHOOT once background is accounted for

The sim is noiseless; the real run contains background. Measured on a properly matched pair —
`G241217_011` (blank) vs `O240206_015` (loaded), both 36 windows / 300-1165 Th / 1/K0 0.6-1.6,
differing only in gradient:

| | background share of peaks | of intensity |
|---|---|---|
| MS1 | **17.6%** | **11.4%** |
| MS2 | 8.3% | 4.7% |

So the analyte target is ~89% of MS1 TIC, and the sim's analyte is **~1.20x** real analyte. Both
noise modes ADD (`--spike-into` is `real + synthetic`; `--noise-real-data` deposits additively), so
overlaying background takes the total to ~1.18x. **Calibrate `--intensity-scale` against
blank-subtracted TIC, not against a loaded run's total.**

### Background is not a shortcut to density

A real blank has **41.3 MS1 peaks/scan** — more than the sim's entire analyte output (24.3). Overlay
would give ~65.6 against real's 333.6, still 5x short. Background and the observation model must be
fitted together, as one layer, exactly as this document already argues.

### Evidence for a saturation ceiling

Both K240723 replicates report an MS1 max of **exactly 94,166** — identical to the digit across
separate acquisitions. That is the signature of a hard ceiling, not a coincidence. MS2 maxima differ
(41,498 vs 40,006), so it looks MS1-specific. The sim reaches 6.5 M, 69x above it, with no saturation
model. This is data arguing against the "default to linear response" position below; a dilution
series would settle it.

### Why MS2 is under-produced

`sum(fragments) == sum(precursor isotopes) == abundance` exactly, verified on all 485,029 ions (both
normalised to unit total, then scaled by the same per-ion abundance). But MS1 spreads that total over
**3.0** peaks and MS2 over **157.6**, so an MS2 bin is ~50x dimmer. The floor therefore censors MS2
~50x harder in abundance terms, which is why sim MS2/MS1 TIC is **0.34** against real's **0.69**. The
fragments are not missing from the model; they are being censored before reaching the file.

### What A3 (counting statistics) already recovered

`--ion-count-noise` moves both metrics the right way while preserving TIC to +0.5%:

| | peaks/scan | intensity/peak |
|---|---|---|
| deterministic | 9.96 / 5.34 | 1078 / 80.4 |
| + shot noise | 11.96 / 6.92 | 898 / 57.6 |

Denser and dimmer, because bins expecting 0.4 counts now sometimes draw 1 — the near-floor
population that did not exist before. Roughly 20% of the density gap; the rest is spreading and
background.

### End-to-end: what a search actually finds

DIA-NN 2.5.0, q<0.01, on the 80k render: **19,761 peptides / 28,426 precursors / 2,089 protein
groups**, i.e. **25.3% peptide recall** against the 76,870 peptides with non-zero abundance. Today's
render changes gained +7.8% peptides while emitting 21.8% FEWER MS1 peaks.

FDP is **1.47%** against a nominal 1%: 1.22% are library peptides never simulated (the FASTA yields
~278k peptides, the design simulated 80k), 0.20% RT-mismatched, 0.04% right sequence at a charge
never rendered, and exactly **1** of 2,965 zero-abundance control peptides. RT agreement on true
positives: median 0.43 s, p95 0.9 s.

**Note for any future FDP claim:** searching a 278k library against an 80k simulation builds a ~1.2%
unavoidable FP floor into every result. Restrict the library to simulated peptides, or subtract the
floor, or the number measures the benchmark's subset ratio rather than the simulator.

## The observation model to build (abundance held fixed)

Applied *after* ion generation, per frame type, ideally conditioned on m/z / mobility / gradient position:

1. **Signal spreading** — an ion's current over RT (frames) and mobility (scans) Gaussians, and its
   isotope/fragment structure. Partly present already; it is a large part of why per-peak intensity is
   far below total ion abundance.
2. **Count floor / censoring** at the detector threshold (~21). Implement as a *real* floor/censor, not
   only the current post-quantisation drop cutoff (`--min-peak-intensity`).
3. **Ion-count noise** — shot/counting statistics on the (small) per-bin counts.
4. **Background process** — the dense low-level population that fills the ~30× density gap, conditioned
   on frame type / m/z / mobility / gradient region. **This needs a method-matched blank to measure**
   (see the sample request); do not assume it is "just noise" — it may include real low-level analyte.
5. **Signal→response transfer** (nonlinearity / saturation) — **only if a dilution series shows it.**
   The current evidence does *not* demonstrate saturation (real maxima ~60k/8.6k are not obviously
   clipped; the `u32` ceiling is irrelevant to instrument saturation). Default to linear response until
   data says otherwise.

## Acceptance criteria (corrected)

**Primary — truth preservation (must hold):**
- Recall-vs-**unchanged** abundance still spans the full range (the abundance axis was not compressed).
- Response curve for identified / spiked precursors is monotonic and linear where the real data is.
- Feature-level isotope/envelope intensities keep their true ratios.

**Hard compatibility check:**
- Per-peak floor is exactly **21** on MS1 and MS2 (an instrument/method threshold, verified in blanks).

**Secondary — emergent-shape diagnostics (regression checks, NOT primary objectives):**
- After the full observation model, the pooled per-peak median / density / dynamic range land within
  tolerance of real, **stratified by frame type and gradient region**, with **analyte and background
  peaks compared separately** (using the blank).
- These are explicitly *joint post-model* diagnostics. They can "pass for the wrong reasons" (a high
  cutoff + injected floor noise matches floor/median/density while destroying abundance-response
  fidelity), so they gate nothing on their own — truth preservation does.

## What we can and cannot do before the new calibration samples

Without a **method-matched blank** we cannot separate background from signal — only match the *combined*
distribution, which risks looking-right-for-wrong-reasons. Without a **dilution series** we cannot fit
or verify the response curve. Until those exist, the honest scope is: keep abundance fixed; add
spreading + a real floor/censor + count noise + a *provisional* background fit to the combined real
distribution; label the response linear; and treat every shape diagnostic as provisional. The clean fit
comes from the samples specified in `CALIBRATION_SAMPLE_REQUEST.md`.
