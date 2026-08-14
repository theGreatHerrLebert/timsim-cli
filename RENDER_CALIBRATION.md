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
2. **Count floor / censoring** at the detector threshold. **DONE (2026-08-13), by inheritance rather
   than by constant** — see "The floor is not a constant" below. `--min-peak-intensity 0` measures the
   floor from the reference `.d`'s own recorded minimum.
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
- Per-peak floor equals **the reference `.d`'s own floor** on MS1 and MS2. NOT a constant — see below.

## The floor is not a constant (2026-08-13)

An earlier version of this document asserted the floor "is exactly 21". A survey of nine real `.d`
files says otherwise: it is **10** (`G211202` ×2, `F164xx` ×3, `G241217` blank) or **21** (`K240723`
×3) — stable within an acquisition batch, different between batches. Hardcoding 21 would have been
the fourth instance of the same mistake this render has already made three times (clock, mobility
window, intensity floor), so the floor is now **inherited from the reference `.d`**.

Measured effect at realistic complexity (490k peptides, L050). **Comparator matters and has been got
wrong repeatedly**: the sim is rendered against blank `G241217_011`, whose method-matched loaded run
is `O240206_015` (both 36w / 300–1165 / 1/K0 0.6–1.6). `K240723` is a *different method* and is
denser with a tighter distribution, so quoting the gap against it overstates it by roughly 2×.

| | peaks/scan | floor | p50 | p99 | p99.9 | max | dyn |
|---|---|---|---|---|---|---|---|
| sim, floor = 1 (old default) | 121.9 | 1 | 23 | 3,874 | 37,794 | 648,327 | 37,794× |
| sim, floor inherited | 76.4 | 10 | **55** | 6,735 | 54,801 | 603,201 | 5,480× |
| sim, spiked into the blank | 76.3 | 10 | **55** | 6,737 | 54,809 | 603,201 | 5,481× |
| **real `O240206_015` (method-matched)** | 240.9 | 21 | **56** | 605 | 3,283 | 26,943 | 131× |
| real blank `G241217_011` | 39.5 | 10 | 57 | 137 | 370 | 3,099 | 37× |
| real `K240723` (different method) | 333.6 | 21 | 53 | 245 | 1,366 | 63,965 | 55× |

Against the correct comparator the residual is **3.2× on peak density** and **11× on `p99`**, not the
4.4× / 27× that the mismatched comparison gave. Netting out background (39.5 peaks/scan, present in
both), the **analyte** deficit is 201 vs 36.8 peaks/scan ≈ **5.5×**.

Note the floors differ *within* the matched pair — blank 10, loaded 21. Method matching fixes peak
density and m/z range; it does **not** fix the floor, which tracks acquisition batch. So "inherit the
floor from the reference" reproduces the reference's censoring, which is the right guarantee, but it
does not automatically equal the floor of whichever loaded run is being imitated.

Two things follow, and the second corrects a framing error made earlier in this document's history.

**The median is now right** (55 vs 53). The old default of 1 let counting noise manufacture 1–2 count
peaks no instrument would have recorded, which inflated density and dragged the median to 23.

**The dynamic-range gap is an upper-tail problem, not a whole-distribution problem.** The blank's own
`p50` is 57 against the matched loaded run's 56 — i.e. *the median peak in real data is a background
peak*, and the analyte load adds ~200 peaks/scan and a bright tail while barely moving the median.
So the residual error is concentrated in `p99`/`p99.9` and in peak density, both symptoms of the same
defect: one peptide's signal occupies too few bins, each too bright. Item 1 (signal spreading) remains
the blocker, and it still needs the measurement.

Do **not** close the gap by narrowing abundance or lowering `--intensity-scale`. Both would reproduce
real data's per-peak shape while destroying the recall-vs-abundance harness this benchmark rests on.
A provisional knob belongs on **spreading** — intensity-conserving, on the observation model — never
on the truth axis.

### Identification effect, measured (DIA-NN 2.5.0, q<0.01, 490k peptides, `hela5k.speclib`)

Two variables were separated: the floor, and whether background is *modelled* (`--noise-real-data`)
or *literally superimposed* (`--spike-into` the blank).

**Match the realism level before comparing rows.** An earlier version of this table did not, and drew
a wrong conclusion. The hand-run arms passed `--ion-count-noise true --instrument-cv 0.15`, which is
**R3**, not R2. All rows are L050.

| render | realism | floor | background | IDs | FDP | recall (detectable) |
|---|---|---|---|---|---|---|
| ramp-004 `L050_R2` | R2 | 1 | modelled | 66,561 | 0.33% | — |
| **ramp-005 `L050_R2`** | R2 | 10 (inherited) | modelled | **32,410** | 0.27% | **14.7%** |
| ramp-004 `L050_R3` | R3 | 1 | modelled | 140,230 | 0.30% | — |
| hand-run A2 arm | R3 | 10 (inherited) | modelled | 39,537 | 0.34% | 17.9% |
| hand-run spike arm | R3 | 10 (inherited) | **real blank** | 39,388 | **0.26%** | 17.9% |

**The floor costs 51% of identifications at R2** (66,561 → 32,410) and **72% at R3** (140,230 →
39,537). It bites harder the more counting noise is on, which is the point: without a floor,
`--ion-count-noise` is unconstrained and manufactures recordable peaks from bins expecting a
fraction of a count. Both are corrections, not regressions.

### A2 matches the histogram, NOT the searchable content (2026-08-13)

The agreement below (0.4% on peak counts, distributions equal to 3 s.f.) was initially read as
validating the A2 background model. **That reading is wrong, and the error is instructive.**

Searched with the SAME library (`hela5k.speclib`, 4,995 proteins), same DiaNN settings:

| | precursors @1% FDR | proteins |
|---|---|---|
| real blank `G241217_011` | **1,389** | **341** |
| A2 modelled background (noise-only control) | 201 | **0** |

A real blank yields 341 proteins of ordinary human carryover — no contaminant panel required. The
modelled background yields none. A2 reproduces *where peaks are and how bright they are*; it does not
reproduce the coherent peptide-like structure across RT × mobility × isotope that a search engine
assembles into an identification.

**Matching the marginal peak distribution is not matching the data.** Any future background model
must be validated by SEARCHING it, not only by comparing histograms — the two diagnostics disagree
here by a factor of ∞ on proteins while agreeing to 0.4% on peak count.

Consequences: (1) the FDP background control is effectively inert for A2 runs — it subtracts 2–3 IDs
where real background offers ~1,389 identifiable precursors; it works as designed only for
`--spike-into`, where the control IS the real blank. (2) Every A2 arm is an easier search than
reality.

**CLOSED — the answer key is not inflated by blank-derived IDs.** Intersecting the three searches:

| | |
|---|---|
| real blank IDs | 1,353 |
| blank ∩ spike-arm | 183 |
| blank ∩ A2-arm | **215** |
| (blank ∩ spike) \ A2 — blank-driven candidates | 4 |
| genuinely blank-only, credited as recall | **1** of 39,388 |

The decisive check is that **`blank ∩ A2` (215) EXCEEDS `blank ∩ spike` (183)**. The A2 arm contains
no real-blank signal, so those 215 can only be simulated peptides the blank happens to share (both
are HeLa off the same 5k proteome). Contamination would make spike overlap MORE than A2; it overlaps
less. The blank's identifications are simply outcompeted: simulated signal is orders of magnitude
brighter, and with ~39k strong IDs setting the FDR threshold the weak blank peptides fall below it —
which also explains why spike mode gains no recall over A2.

**Spike-mode recall is safe to quote.** The A2 background being too clean (above) is still real, and
still makes every A2 arm an easier search than reality.

**Superimposing real background costs 0.5% of identifications** (39,388 vs 39,537). That comparison
is sound where the floor one was not: both hand-run arms carried identical flags and differ only in
background source.

Two consequences for how the benchmark is used:

- **Spike-into-real is safe.** FDR control is unaffected (FDP 0.26% against a 1% nominal, marginally
  *better* than the modelled background), and the recall-vs-abundance curve is unchanged. So real
  background can be used to get real chemical noise and real interferences without paying for it in
  search quality — the noise-only control already exists to subtract background IDs from FDP.
- **Any recall figure predating 2026-08-13 is inflated** by the floor-of-1 artefact and is not
  comparable to figures measured after it. The ramp-004 arms are affected.
- **The R2 baseline is 14.7% (ramp-005 L050), not the 17.9% quoted earlier**, which came from an
  R3-configured hand-run. Realism level is part of the comparison, not a detail.

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
