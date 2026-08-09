# Chromatographic peak shape — EMG by default

## TL;DR

`timsim-render` (and `-bench`, `-thermo`, `-sciex`) grew `--peak-shape {gaussian|emg}`, **defaulting
to `emg`**: v1's exponentially modified Gaussian instead of the symmetric Gaussian v2 shipped with.
`--peak-shape gaussian` reproduces the pre-EMG binary **byte for byte** — proved on real `.d`,
`.raw` and mzML output for all four writer/mode combinations by
[the Gaussian golden](#the-gaussian-golden), not just on kernel values.

Every render now also **stamps its resolved kernel into its own output**
([self-describing artifacts](#self-describing-artifacts-the-actual-fix)), `--emg-k` is
[validated instead of clamped](#input-validation), and the truncation window
[inverts the survival function](#the-truncation-window-inverted-not-approximated) instead of
approximating it.

> ## ⚠️ READ THIS BEFORE TRUSTING ANY CACHED RENDER
>
> **Changing the default changed render output while leaving the command string untouched, so
> necroflow's fingerprint cannot tell that cached renders are stale.** Every `.d` under
> `/media/hd01/timsim-cohort/work/nodes/render_a2/` was produced with the OLD Gaussian and is
> reported by necroflow as `up_to_date`.
>
> What is fixed: **the artifact now says which kernel made it.** Every render stamps
> `(peak_shape, emg_k, n_sigma)` into the `.d`'s `GlobalMetadata` and into the answer key's parquet
> metadata, so a stale artifact is identifiable *from its own contents* — see
> [Self-describing artifacts](#self-describing-artifacts-the-actual-fix). A `.d` with no
> `SimPeakShape` row predates this change and is Gaussian.
>
> What is **not** fixed: the fingerprint itself. That needs one edit to the live flow, which
> re-fingerprints 30 arms. The exact edit and its measured cost are in
> [The flow edit](#the-flow-edit-not-applied-here).

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
| `--peak-shape gaussian` vs parent commit `52e6c91`, **all four writers**, whole real artifacts | **byte-identical** on every signal carrier — see [the Gaussian golden](#the-gaussian-golden) |
| the recorded `(shape, k, n_sigma)` reconstructing the shape it describes | `PeakShape` equality including derived constants, over a 0 … 1e3 `k` grid × four `n_sigma` (`provenance_round_trips`) |
| fractions/ordinates finite and non-negative | 20-point `k` grid (`0`, subnormal, 1e-300 … 1e300) × 21-point `z` grid (`±1e300`) |
| captured mass `>= 1 - 2p` | every `k` in that grid × `n_sigma` 2/3/4/6 (`truncation_window_captures_the_promised_mass_for_every_k`) |

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
* `truth.parquet` **used to be identical** between the Gaussian and EMG renders (sha256
  `f0857aa7c2e7f3c705dfa2e1f1f8b5f4d5644ea361b57767131d7d5f48e4f7f4` either way), so the answer key
  did not record which shape produced the `.d`. **A stale `.d` could not be detected from its own
  outputs.** That is now fixed — see below.

## Self-describing artifacts (the actual fix)

Documentation and manual invalidation are not reproducibility mechanisms. A fingerprint is a
mechanism, but it only protects artifacts *inside* the flow: a `.d` that is copied, re-pathed,
restored from backup or handed to a collaborator arrives with no fingerprint attached. So the record
goes **into the artifact**.

Every render now stamps its resolved kernel (`timsim_cli::provenance`):

| artifact | where | keys |
| --- | --- | --- |
| Bruker `.d` (DIA and DDA) | `analysis.tdf` → `GlobalMetadata` | `SimPeakShape`, `SimEmgK`, `SimNSigma` |
| every answer key parquet (Bruker `--truth` / `--dda-truth`, Thermo `--thermo-truth`, SCIEX `--truth`) | Arrow schema metadata in the parquet footer | `peak_shape`, `emg_k`, `n_sigma` |

* **Three keys, not one.** They are exactly the arguments to `PeakShape::emg`, so the record
  *reconstructs the kernel* rather than labelling it: `provenance::parse_shape` returns a `PeakShape`
  that compares `==` to the one the render used, derived constants (mode offset, tail reach, peak
  ordinate) included. `n_sigma` is in there because every one of those constants depends on it.
* **The Gaussian records `emg_k = 0`.** Not a placeholder: the Gaussian *is* the `k = 0` member of
  the family (see [Input validation](#input-validation-and-the-k--0-limit)), so `(name, k)` is a
  complete description, and `k = 0` parses back to `PeakShape::Gaussian`.
* **The parquet stamp is the one that covers all four writers.** The Thermo `.raw` and SCIEX mzML
  writers are upstream crates with no provenance seam; their answer key is where the shape can be
  recorded without forking them. Since the answer key is what every downstream scorer reads, a scored
  number can always be traced back to the kernel behind it.

Measured on a real render (12,228 precursors, `.d` + `truth.parquet`):

```
$ sqlite3 gaussian.d/analysis.tdf "SELECT Key,Value FROM GlobalMetadata WHERE Key LIKE 'Sim%'"
SimPeakShape|gaussian     SimEmgK|0.0                     SimNSigma|3.0
$ sqlite3 emg.d/analysis.tdf      "SELECT Key,Value FROM GlobalMetadata WHERE Key LIKE 'Sim%'"
SimPeakShape|emg          SimEmgK|0.47619047619047616     SimNSigma|3.0

truth.parquet   gaussian -> 794ca2c6131f89266d8ca9c7fdcd20bb037ed5fd62d177996072c208a76f0146
                emg      -> 2a34906a5c5eba26d0c125bd4f970690973c19b1e3d983bd132869b795fe00f7
```

The two answer keys are no longer the same file. **A `.d` with no `SimPeakShape` row predates this
change and is therefore Gaussian** — which is exactly what the 30 cached cohort arms are.

## The flow edit (NOT applied here)

Making the *fingerprint* see the shape needs one edit to the live
`/scratch/timsim-demo/timsim-necro-repo/flow/timsim_flow.py`. It is written up rather than made,
because it is a cache decision with a five-day price tag and it belongs to the person who owns the
cohort.

### The edit

necroflow fingerprints a node on its **command template + config + input hashes**, and shell-quotes
each `{placeholder}` individually, so flag NAMES must be literal in the template and only VALUES may
be placeholders. Four templates carry a render, and each needs two literal tokens plus one parameter:

| template | line (at 1777-line revision) |
| --- | --- |
| `_RENDER_HEAD` (Bruker DIA; feeds `render` and `render_a2`) | 680 |
| `render_dda` | 769 |
| `render_thermo` | 637 |
| `render_sciex` | 807 |

For each: append `--peak-shape {peak_shape}` to the command string, add `peak_shape: str = "emg"`
(or `"gaussian"`) to the decorated function's signature, and thread it through the factory that
builds the node (`_render(sid)` at 1369 and its Thermo/SCIEX/DDA siblings) plus a
`--peak-shape` CLI argument next to `--intensity-scale` at 1540. Roughly 12 lines across 4 command
sites. `--emg-k` and `--sigma-frames` deserve the same treatment for the same reason; they are
currently invisible too.

### The cost, measured

Re-measured on `/media/hd01/timsim-cohort/work/nodes/` on the day of this change:

| node kind | dirs | live (`max_peptides = 0`) | commands mentioning `--peak-shape` | live wall | live output |
| --- | --- | --- | --- | --- | --- |
| `render_a2` | 70 | **30** | 0 | **124.2 h** | **151.9 GB** |
| `render_thermo` | 2 | 0 | 0 | — | — |

Adding the flag changes the command template, hence the fingerprint, hence **all 30 live arms
rebuild once: ~124 h serial (~5.2 days) and 151.9 GB rewritten.** That cost is identical whichever
value is passed — it is the price of making the parameter visible, not of changing it. `/media/hd01`
had 485 GB free, so a side-by-side re-render fits.

### The three choices, and what each costs

1. **Pin `--peak-shape gaussian` in the template.** The 30 cached arms are *scientifically* still
   valid (they are Gaussian, and the flag now says so), but they still rebuild once because the
   fingerprint moved. ~124 h, and the cohort keeps the shape it has always had.
2. **Pin `--peak-shape emg`.** Same ~124 h, and the cohort gains v1 shape parity. This is the reason
   the EMG work exists, so it is the recommendation — but it is a science decision, not a code one.
3. **Do neither.** The arms stay cached and stay Gaussian. This is now *safe but silent*: the
   artifacts identify themselves correctly, so nothing is misread, but the next default change is
   again invisible to the fingerprint.

**Whichever is chosen, the flag belongs in the command string.** A default that changes output but
not the fingerprint is exactly the failure mode this file exists to document — the artifact stamp
makes it detectable, not impossible.

## Input validation

`--emg-k` is user input and was not treated as such. `Emg::new` ran every value through
`k.max(1e-12)`:

| input | old behaviour | now |
| --- | --- | --- |
| `k < 0` | silently became `1e-12` — a shape the user never asked for | `Err`: `--emg-k must be >= 0` |
| `NaN` | `f64::max` returns the *other* operand, so `NaN` also became `1e-12` | `Err`: `--emg-k must be finite` |
| `+inf` | survived the clamp; bracket `[-1, inf]` → `NaN` mode → `NaN` half-widths → `NaN` weights **written to disk** | `Err`: `--emg-k must be finite` |
| `k == 0` | `1e-12`, i.e. an EMG that is merely *close* to the documented Gaussian limit | **exactly `PeakShape::Gaussian`** |
| subnormal `k` (`< ~5.6e-309`) | `1/k` overflows, the tail term underflows, the peak-height normaliser becomes `0/0` → `NaN` ordinates | `PeakShape::Gaussian`, documented and tested |

`--sigma-frames`, `--sigma-scans`, `--sigma-seconds` and `--n-sigma` are now validated by one shared
`render::validate_elution_widths`, called by all four renderers. Previously only
`timsim-render-thermo` checked anything; `timsim-render`, `-bench` and `-sciex` accepted a `NaN`
`--n-sigma` and rendered an empty active set.

### The `k = 0` limit

`k = 0` returns the `Gaussian` **variant**, not an EMG with a tiny `k`. That makes the advertised
limit exact rather than approximate: it is bit-identical to `--peak-shape gaussian`, which the tests
assert directly (`k_zero_is_exactly_the_gaussian_variant` compares `to_bits()` against
`gauss_frac`). It also means `(shape_name, k)` is a total encoding of the kernel, which is what lets
the provenance record round-trip.

## The truncation window: inverted, not approximated

The right-hand truncation used `tail_reach = k·ln(1/p)` — the exponential tail's *asymptote*, exact
as `k → ∞` and progressively loose below it. It provably keeps the apex (so it avoids v1's
truncation bug), but ">99.7 % captured" could only be checked at whichever `k` someone tested.

It now **numerically inverts the actual EMG survival function**: bisect `S(n_sigma + tail_reach) = p`
where `p = ½·erfc(n_sigma/√2)` is the mass a Gaussian leaves outside `n_sigma`. Two consequences:

* `emg_sf_std` computes `S` directly as `½·[erfc(z/√2) + exp(−z²/2)·erfcx((1/k − z)/√2)]` — a sum of
  non-negative terms. `1 − emg_cdf_std(z)` is useless here: the CDF is built from `erfc(−z/√2)`,
  which saturates at 2 for `z ≳ 6`, so the complement is pure cancellation exactly where the window
  is solved.
* The guarantee becomes **structural**: the left edge is at `z = −n_sigma`, where the EMG's CDF is
  bounded above by the Gaussian's, and the right edge is where the EMG's own `S` equals `p`. So
  captured mass `≥ 1 − 2p` for **every** `k`, not for one. `truncation_window_captures_the_promised_mass_for_every_k`
  asserts it over a 20-point `k` grid (`0`, subnormals, `1e-300` … `1e300`) × four `n_sigma`.

The alternative — declaring and enforcing a supported `k` range — was rejected: the inversion costs
one ~160-iteration loop **per render** (not per ion, not per frame), and an enforced range would have
had to be justified by the same numerics anyway.

### This CHANGES the default (emg) render — measured

The old asymptote was *over*-generous, so the inversion NARROWS the window. At the default
`k = 10/21`, `n_sigma = 3`: `tail_reach` 3.1465 σ → 1.1927 σ, captured mass 99.947 % → 99.814 %
(floor 99.730 %). Same direction at every `k` (0.25: 1.652 → 0.400; 9.5: 62.77 → 59.83).

Measured end to end on a 12,228-precursor DIA render (`--sigma-frames 30`), commit `04645db` vs this
tree, both at the default EMG:

| | `04645db` | now | delta |
| --- | --- | --- | --- |
| `analysis.tdf_bin` | 14,120,245 B | 13,923,141 B | −1.40 % |
| summed ion current | 1.181317e9 | 1.180168e9 | **−0.097 %** |
| peaks written | 8,052,421 | 7,941,772 | −1.37 % |

The ion current barely moves — what disappears is far-tail bins that were below the emission floor
anyway — but **`--peak-shape emg` output is not byte-identical to `3336bfe`/`04645db`.** Nothing was
cached at those commits (the flag is 0/70 in every node command), so nothing is invalidated; it is
recorded here because it is a deliberate change to the default kernel, not a refactor.

The mode search picked up a related fix. Golden section shrinks its bracket by `0.618²⁰⁰ ≈ 1.8e-42`,
so on `[-1, k+1]` with `k = 1e100` it resolved the mode only to `~1e58`. A second bound —
`u ≤ √(2 ln k)` at the mode, from `Φ(u) ≥ ½` — caps the bracket at ~25 for every representable `k`.
It is inert for `k ≤ 1`, so v1's default `k = 10/21` keeps the exact mode offset it always had.

## The Gaussian golden

`tests/emg_v1_parity.rs` and the unit tests prove **kernel** equivalence: `elution_frac(Gaussian, ..)`
is bit-for-bit `gauss_frac(..)`. That is a statement about one function. What 124 h of cached renders
depend on is a statement about the whole pipeline — placement, sweep, quantisation, zstd framing,
SQLite tables, vendor container, answer key.

So `tests/golden_gaussian.rs` renders **real artifacts** and hashes them against artifacts produced
by the pre-EMG parent commit `52e6c91`, which has no `--peak-shape` flag at all and therefore renders
the Gaussian by construction. Four writer/mode combinations, ~9 s total on a 12,228-precursor
fixture:

| case | binary | signal artifact | parent-equal |
| --- | --- | --- | --- |
| Bruker DIA | `timsim-render --dia` | `analysis.tdf_bin` `562f5ec8…` | ✅ |
| Bruker DDA | `timsim-render --dda` | `analysis.tdf_bin` `97193228…` | ✅ |
| SCIEX SWATH | `timsim-render-sciex` | `sciex.mzML` `4fcb4c0b…` | ✅ |
| Thermo Astral | `timsim-render-thermo` | `data.raw` `d8c07cf5…` | ✅ |

plus each case's answer key, hashed over the parquet **data region** (file minus thrift footer, so
the added footer metadata is excluded and nothing else is) — all four parent-equal. `analysis.tdf` is
the one artifact that legitimately differs, by exactly the three `Sim*` rows; that was verified by
diffing full `sqlite3 .dump` output, and its hash is pinned too so the stamp cannot drift either.

`tests/golden/regenerate.sh` re-derives all of it, including building the parent commit in its own
worktree and target dir.

**Running it: set `TIMSIM_GOLDEN=1` on a machine that holds the fixtures.**

```sh
cargo build --release --features tdf,thermo,sciex     # the golden shells out to the release binaries
TIMSIM_GOLDEN=1 cargo test --features tdf,thermo,sciex --test golden_gaussian
```

The fixtures are machine-local paths — a 12,228-precursor feature space, a 5,692-frame reference `.d`,
a real Astral template — so the test has two modes, and the split matters:

* **A drift always fails.** If a case runs, its hashes must match. That is not environment-dependent.
* **Having nothing to run** is a failure only under `TIMSIM_GOLDEN=1`, where absent fixtures mean a
  provisioning bug. Without it the test prints `GOLDEN: 0/4 combinations checked` and returns green.

Neither hardcoding works. Always-failing makes `cargo test` red on every machine but this one, and a
permanently red suite is one nobody reads; always-passing turns a vanished fixture into silent loss of
coverage. The env var is what carries "these fixtures are supposed to be here", and it is the thing to
set in CI once the fixtures are provisioned there.

**Not exercised, and why:**

* **SCIEX native `.wiff`** — no `.wiff` writer exists in this repo or is reachable from it
  (`sciexwiff` is legal-held and lives in the rustims satellite), and no `.wiff` template exists on
  this machine. The open-mzML SCIEX path *is* covered.
* **`timsim-render-bench`** — not a writer. It renders into memory to measure throughput and emits no
  artifact to hash. It shares `stream_render_flat` and the shared validators with `timsim-render`, so
  its shape handling is covered by the unit tests.
