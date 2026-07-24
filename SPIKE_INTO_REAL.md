# Spike-into-real (mode B) — synthetic ground-truth overlaid on a REAL run

The last P0 realism piece and the strongest benchmark we can produce: instead of rendering onto a blank
template, deposit the synthetic precursors **additively onto a real experimental `.d`**. The real run is the
background — real co-elution, real chemical noise, real dynamic range — and **only the synthetic spikes are
labeled**, so a tool's spike recovery + quant accuracy is measured in a genuine matrix with perfect ground
truth for the spikes. This is the PlasmaBENCH PYE pattern.

## What v1 does (match it)

`superimpose_reference_frames` (imspy `add_noise_from_real_data.py`): per output frame,
`output[f] = ref_frame(f) + synthetic_frame(f)` for `f <= num_ref_frames` (TimsFrame `+` merges peaks by
`(scan, tof)`, summing intensities), else synthetic alone. Bruker/tof: a direct same-grid merge (the real
`.d`'s calibration IS the output's). Thermo has a `superimpose_merge_ppm` (m/z merge) — not in scope (Bruker
DIA only for now).

## v2 design — reuse A2's machinery

The A2 path already deposits real reference peaks onto rendered frames via `dedup_and_quantise`'s `noise`
argument (real counts added post-scale). Spike-into-real is **the same deposit with a different source**: the
"background" for output frame `f` is the **entire real frame `f`** (all peaks, no sampling, no `[1,cap]`
filter, no downsample), read with the `TimsDataset::get_frame` reader A2 already uses.

- **Flag:** `--spike-into <real.d>`. Implies `--reference-d <real.d>` (schedule + calibration + frame count
  all come from the spike target) and a **1:1 output↔real frame** mapping — the whole point is to reproduce
  that exact run. Mutually exclusive with `--noise-real-data` (A2 samples a blank; spike copies a real run).
- **Populate `frame_noise[f]` = all grid-safe `(scan, tof, intensity)` of real frame `f`.** Then the
  synthetic signal renders on top exactly as today (`dedup_and_quantise(synthetic, scale, floor,
  frame_noise[f])`). Output = real + synthetic, on the real grid.
- **Truth: unchanged** — the render already co-emits the synthetic answer key. Only the spikes are labeled,
  which is precisely what spike-recovery wants.
- **`--noise-only` generalises:** `--spike-into X --noise-only` deposits the real frames with **no synthetic**
  → a re-encoded copy of X through our writer. That is the **background control** for the eval (searched, its
  IDs subtracted), and it's the correct control because it shares the spike render's exact frame
  representation (same decode→merge→re-encode round-trip), unlike searching raw X. Relax the current guard to
  `--noise-only requires (--noise-real-data OR --spike-into)`.

## Eval — already built

Spike-recovery = the `--background-report` subtraction from A2, verbatim: recall over the injected spikes
(the synthetic truth), real-run IDs excluded from FDP. Control = the `--spike-into X --noise-only` render,
searched. No new scorer needed.

## Flow wiring — mirror the A2 branch

When `cfg.spike_into` set on the Bruker DIA pipeline: the `render` node gets `--spike-into` (via a small flag
suffix like `noise_flags`); a control render (`--spike-into X --noise-only`) + its search feed
`score_bruker_bg`'s `--background-report`. Same shape as the A2 control branch.

## Review resolutions (codex)

- **Floor exemption (bug):** `dedup_and_quantise` applies `min_peak_intensity` to every emitted bin. Real
  detector counts are NOT synthetic quantisation haze, so a floor >1 would drop real bins → the control is
  no longer a faithful copy of X. FIX: exempt the **background-only** emit loop from the floor (keep every
  `q >= 1`). Default floor is 1 so no behaviour change in practice; off-path unaffected (loop only runs with
  a background). Collision bins keep the floor (they carry synthetic signal).
- **1:1 alignment must be ENFORCED, not assumed.** Equal frame count ≠ phase alignment. GUARDS for
  `--spike-into`: (a) reject a different `--reference-d` (force the same path); (b) require
  `n_frames == real_frame_count` (don't silently default); (c) require real frame IDs contiguous `1..=N`
  and frame 1 = MS1 (a regular DIA run) so `get_frame(f)` is the f-th frame and the schedule's cycle replay
  matches the real MS1/MS2 sequence. Error clearly otherwise — irregular runs are out of scope for a first
  cut (copying the exact per-frame sequence is a bigger change, deferred).
- **Memory: resident-first WITH a gate.** Log the footprint (unfiltered real peaks can dwarf A2's sampled
  ~3GB). Bounded streaming (sequential producer builds a frame-chunk's background, parallel workers render
  it, release) is the production design — flagged as the follow-up, same as A2.
- **Control caveat (report, don't fix):** `--spike-into X --noise-only` (re-encoded copy) is the correct
  primary control (shares the spike run's decode→merge→re-encode representation). But subtraction can't be
  complete: a spike can make a pre-existing real peptide newly detectable (search interaction), leaving real
  IDs the control missed → some residual FDP. Inherent; document it, optionally measure raw-X vs control drift.
- **Shared `--intensity-scale` is correct** (scales only synthetic; real counts untouched). A separate
  `--spike-intensity-scale` is optional clarity, not required — skip for now.
- **Truth = emitted detectable signal:** a spike that quantises below the floor (or saturates) is handled by
  the existing detectable-recall hierarchy (present & in-window & has-frags & abundance>floor); no change.

## Load-bearing decisions (resolved above)

1. **Memory.** Spike holds the FULL real run's peaks resident in `frame_noise` (A2 filtered+downsampled to
   ~3GB; spike is unfiltered, so potentially much larger — a real run's every peak). Parity-first: build
   resident, **measure**, and flag streaming (read frame `f` just before deposit) as a follow-up if it
   blows memory. Is resident acceptable for a first cut, or stream from the start?
2. **Geometry alignment.** `--spike-into X` sets reference=X so `DiaSchedule`/n_frames/n_scans/tof_max all
   come from X → 1:1 by construction. Grid-safe guards (`scan<n_scans`, `tof<tof_max`) stay as a backstop.
   Anything that can still desync output frame `f` from real frame `f` here (unlike A2, we DO want 1:1)?
3. **Intensity scale.** Synthetic spikes are `synthetic*intensity_scale`; real peaks are absolute counts.
   The spike must sit at a realistic level within the real dynamic range — `--intensity-scale` is the knob.
   Same reconciliation as A2 (real counts added post-scale). Right, or does spike need its own scale so the
   spike:background ratio is set independently of the render's internal scaling?
4. **Control correctness.** Is `--spike-into X --noise-only` (re-encoded copy of X) the right background
   control, vs. searching raw X? (It shares the render's exact peak representation, so subtraction is
   apples-to-apples.)
5. **Byte-identity.** `--spike-into` unset ⇒ `frame_noise` empty ⇒ byte-identical noiseless render (as A2).
   Confirm no path change when the flag is absent.

## Validation

- `--spike-into` unset ⇒ tdf_bin byte-identical to baseline.
- set ⇒ output = real frames + synthetic; a synthetic spike is recoverable; the real `.d`'s peaks are
  preserved (spot-check a frame's peaks ⊇ real frame's peaks). Deterministic (no RNG in the spike itself).
- `--spike-into X --noise-only` reproduces X's peaks through the writer (control).
- End-to-end: search the spiked `.d`, subtract the control's IDs, recover the spikes at a believable FDP.
