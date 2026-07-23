# A2 — real-data-noise injection (background noise from the reference `.d`)

The second half of true v1 DIA parity. v1's real DIA recipe (`IT-DIA-HYE-B.toml`) runs **A1 signal-m/z
noise AND A2 real-data background together**. A1 is done (`timsim-cli` 7105d73). This is A2: sample **actual
background peaks from the reference `.d`** and deposit them onto the rendered frames, so the search engine
sees real chemical/electronic background (the thing that moves measured FDP toward v1's 3–5%).

Bruker DIA only (needs the reference's real peak data). Thermo/SCIEX get A1 (ppm) only.

## What v1 does (the impl to match)

`imspy_simulation/timsim/jobs/add_noise_from_real_data.py::add_real_data_noise_to_frames` (DIA branch) +
`rustdf/src/data/dia.rs::sample_precursor_signal / sample_fragment_signal`. Per **output** frame:

1. Classify the frame: **MS1 (precursor)** vs **MS2 (fragment)**; MS2 also carries a **DIA `window_group`**
   (from `dia_ms_ms_info`).
2. Build a candidate **pool** of reference frame_ids: MS1 → all `ms_ms_type==0` frames; MS2 → reference
   frames whose `window_group` matches this frame's group.
3. Sample `num_frames` (default **5**) reference frames from the pool **with replacement**. For each:
   - `filter_ranged(mz 0..2000, scan 0..1000, 1/K0 0..5, intensity **1.0..max_intensity**, tof 0..i32::MAX)`
     — keep only **low-intensity background** peaks (cap = `reference_noise_intensity_max`, config 150000;
     the function default is 30). This is the key: real peaks *below* the cap are background, not signal.
   - `generate_random_sample(take_probability)` — keep each surviving peak with prob **0.2** (default).
4. Sum the sampled+filtered frames into one noise frame; **add it to the output frame** (`frame + noise`).

v1 params (config keys → function args):
`num_precursor_noise_frames`/`num_fragment_noise_frames` (5/5), `reference_noise_intensity_max` (150000, one
value feeds both precursor+fragment max in the config path), `precursor_sample_fraction`/
`fragment_sample_fraction` (0.2/0.2). v1 draws with `rand::thread_rng()` — **non-deterministic**.

**Note:** `superimpose_reference_frames` in the same file is a *different* mode — it adds the synthetic
signal onto the **full** real frame (all peaks, not filtered). That's mode **B (spike-into-real)**, not A2.

## v2 design

The building blocks are already ours: `ms_io::data::dataset::TimsDataset::new(lib, path, in_memory=false,
use_bruker_sdk=false)` opens the reference with the **native** Rust reader (no Bruker SDK — same as the
schedule read today) and `get_frame(fid) -> TimsFrame` decodes real peaks to `(scan: i32, tof: i32,
intensity: f64)` (via `ims_frame.intensity`). `TimsFrame::filter_ranged(...)` is **v1's exact filter** and
is deterministic — reuse it verbatim. Only the two RNG steps (frame choice, peak downsample) get a seeded
replacement.

### Determinism (deliberate divergence, as in A1)

v1 uses `thread_rng` → a different noise realisation every run. v2 makes it **seeded + reproducible**: the
frame picks and the per-peak keep/drop are drawn from `noise_state(output_frame_id, sample_index,
peak_index, seed)` (the same splitmix64 avalanche keying A1 uses). Same *distribution and logic* as v1;
reproducible under a seed and stable for the golden gate. Off ⇒ byte-identical to the noiseless render.

### Where the peaks come from (cache, don't re-read)

v1 re-loads `num_frames` reference frames **per output frame** (5 × ~17646 ≈ 88k frame decodes). Instead:
**pre-load + `filter_ranged` each *unique* reference frame once**, sequentially, into a cache
`HashMap<frame_id, Vec<(scan, tof, intensity)>>` (only background peaks survive the filter, so it's small).
Pools (MS1 ids; per-window-group MS2 ids) come from the schedule we already read (`dia_ms_ms_info` +
frame meta). One pass over the reference's real data; bounded memory (background peaks only).

### Injection point (count space — the unit reconciliation)

Synthetic triples are **relative** intensities that `dedup_and_quantise` multiplies by `--intensity-scale`.
Real noise peaks are **absolute counts** and must NOT be scaled. So inject **after** the scale: extend
`dedup_and_quantise(triples, scale, floor, noise)` to take a per-frame `noise: &[(u32 scan, u32 tof, f64
count)]`; build the synthetic `summed` map, then when emitting compute `q = (v*scale + noise_at_key)`, and
also emit noise-only keys absent from the synthetic map. This is exactly v1's `frame + noise` (both in count
space) and keeps the noiseless path (`noise = &[]`) byte-identical.

### Pipeline placement

Pre-sample sequentially (single reader, before the rayon render): build `frame_noise:
HashMap<frame_id, Vec<(scan,tof,count)>>` for every output frame. The parallel render closure then just
passes `frame_noise.get(&frame)` into `dedup_and_quantise` — no reader crosses the rayon boundary. Peak
memory: background peaks per frame × n_frames (small; background is sparse after the cap filter).

### Flags (mirror v1, all opt-in; reuse `--noise-seed`)

- `--noise-real-data` (bool, enable A2; off ⇒ byte-identical)
- `--noise-precursor-frames <usize=5>` / `--noise-fragment-frames <usize=5>`
- `--noise-intensity-max <f64=150000>` (the `reference_noise_intensity_max` cap; one value → both, per v1
  config path — optionally split later)
- `--noise-precursor-fraction <f64=0.2>` / `--noise-fragment-fraction <f64=0.2>`

## Review resolutions (claudex, codex)

- **Frame classification (Q2 — was a latent bug).** Do **not** assume output frame f ⇔ reference frame f;
  `DiaSchedule` replays a cycle modulo `cycle_len`, so under an explicit `--n-frames` they desync. Classify
  each output frame by `sched.window_group(frame)` / `sched.ms_level(frame)`; build pools from the reference
  metadata (`read_dia_ms_ms_info` → MS2 frame_ids per window_group; frame meta `ms_ms_type==0` → MS1). Fail
  clearly if a needed pool is empty.
- **Duplicate picks need independent draws.** Key the per-peak keep/drop on `(output_frame, sample_index,
  peak_index, seed)` — `sample_index` (0..num_frames) ensures the same reference frame picked twice among
  the 5 gets two independent downsamples, matching v1's with-replacement draw. Cache stores the deterministic
  `filter_ranged` result with a **stable peak order**; never dedupe across sampled occurrences.
- **Unbiased frame selection:** reduce the hash to `[0,pool_len)` via multiply-high
  (`(h as u128 * len as u128 >> 64)`), not `% len` (modulo bias).
- **Injection order (must match `frame + noise`):** accumulate all synthetic *relative* contributions →
  scale once → add real counts at the bin → then floor + `u32` saturation. Do it inside
  `dedup_and_quantise` (one map), never per-contribution. Off (`noise=&[]`) ⇒ byte-identical.
- **Grid safety:** the reference's `(scan, tof)` may exceed the output grid if geometry is overridden —
  skip any peak with `scan >= n_scans || tof >= tof_max` at injection (don't let out-of-grid data reach the
  writer). Real tof shares the output calibration (same `.d`) so no re-projection when geometry is inherited.
- **Determinism:** claim distributional/logic parity with v1, NOT byte parity (v1 is `thread_rng`).
  Byte-reproducibility across our own runs holds because the FxHashMap has a fixed hasher and identical
  inserts → deterministic iteration (same reason the noiseless render is already reproducible); the
  seed + keying are part of the render command, so golden bytes are seed-specific. Validate by re-running.
- **Cap is absolute counts**, not scaled by `--intensity-scale` (v1 parity); changing `intensity-scale`
  intentionally changes signal:background. Measure the realized distributions, don't eyeball.
- **Memory/IO:** parity-first = decode each unique reference frame once (sequential), cache filtered
  background peaks. Measure peak-count/bytes; subsampling the pool would change the distribution, so it's an
  explicit non-parity flag if ever needed, not a silent safeguard.

## Original open questions

1. **Cache vs v1's re-sample-per-frame.** Caching each unique reference frame once then sampling *frame
   ids* (with replacement) per output frame reproduces v1's *distribution* (same pool, same 5-with-
   replacement, same per-peak 0.2) while decoding each reference frame ≤1×. Is there a statistical
   difference that matters (e.g. v1 could pick the same real frame's peaks differently across two output
   frames — we preserve that via the seeded per-peak downsample keyed on output_frame_id)? Believed
   equivalent; confirm.
2. **Reference frame count vs output frame count.** Output frames mirror the reference schedule 1:1 (same
   count, same types/window-groups). So an output MS2 frame's `window_group` is just its reference frame's
   group. Correct, or can `--n-frames` desync them?
3. **Intensity realism.** Real counts added on top of `v*scale` synthetic counts: with `--intensity-scale
   5e5`, a synthetic apex is ~1e5–1e8 and background peaks are ≤150000 — same order, so background is
   visible but sub-signal. Right, or should the cap scale with `--intensity-scale`?
4. **`in_memory=false` (lazy) reader** decoding ~17646 real frames sequentially — acceptable one-time cost
   (seconds)? Or sample a *subset* of reference frames into the pool to bound IO?
5. **tof range.** Real reference tof indices share the output's calibration (same `.d`), so real `tof` can
   be deposited directly (no re-projection). Any risk the reference's `tof_max` exceeds the output grid?
6. **Determinism divergence** from v1 (seeded vs thread_rng) — same call as A1. Acceptable?

## Validation

- `--noise-real-data` **off** ⇒ tdf_bin byte-identical to the frozen baseline.
- **on**: deterministic (same seed → same bytes; different seed → different); frame `NumPeaks` rises;
  the added peaks are real reference `(scan, tof)` background (spot-check against `get_frame` on the ref);
  measured FDP moves toward v1's 3–5% on a searched run.
- Unit test: the seeded downsample keeps ≈`take_probability` of peaks; pool classification (MS1 vs
  window-group MS2) matches the schedule.
