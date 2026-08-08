# timsim-cli

The **timsim v2 protocol tools**: one small Rust binary per stage of a synthetic proteomics
experiment, from FASTA to a real vendor raw file. Every stage reads and writes **Parquet** whose
columns come from `timsim-schema`, so the pipeline is inspectable at every seam and each step can be
cached, re-run, or replaced on its own.

This is a **single crate** (`timsim-cli` 0.1.0, MIT) with **13 `[[bin]]` targets and no
`src/main.rs`** — there is no `timsim <verb>` dispatcher. One binary per tool is deliberate:
necroflow's cache identity can hash the binary, so changing the renderer does not invalidate your
digest.

It is also where **all three instrument writers** live — Bruker timsTOF `.d`, Thermo Astral DIA
`.raw`, and SCIEX ZenoTOF SWATH open mzML — driven off one instrument-independent feature space.

## Build: read this before you `cargo build`

A plain release build gives you **nine lightweight tools and none of the renderers**. The renderers
are behind cargo features, so the default build never pulls ms-io (polars + bundled SQLite),
`timsim-core`, or the Thermo/mzML writer stacks.

```bash
cargo build --release                      # 9 binaries — no renderer
cargo build --release --features tdf       # + timsim-render, timsim-spectra
cargo build --release --features thermo    # + timsim-render-thermo
cargo build --release --features sciex     # + timsim-render-sciex
```

| binary | feature | what it does |
|---|---|---|
| `timsim-proteome` | *default* | FASTA (or a multi-source spec with organism tags + contaminants — this is how HYE works) → the protein universe. Structure only, no amounts. |
| `timsim-digest` | *default* | Proteome → peptides + occurrences + cleavage sites. **Analytic** (an exact expectation, no seed); opt-in `--max-peptides` draws a seeded sample instead. |
| `timsim-modify` | *default* | Peptides → **modforms**, driven by per-site **occupancy** (the chemist's measurable number) rather than a search engine's variable-mod combinatorics. Also emits the modification spec that `timsim-yield` reads, so the two stages cannot disagree about which mods block trypsin. |
| `timsim-design` | *default* | Proteome + `design.toml` → samples, runs, sample↔run map, protein quantities. You specify a **mixture**; fold changes are derived, not typed in. |
| `timsim-yield` | *default* | Shared structure + one sample's protein amounts → peptide amounts. `--digestion-efficiency` and `--cleavage-p` are the same parameter under two vocabularies. |
| `timsim-precursors` | *default* | Peptides/modforms → the ion layer: m/z, isotope envelope, charge distribution, flyability. Each multiplier of `ion_amol = peptide_amol × modform_fraction × ionization_propensity × charge_fraction` stays its own column, so the chain back to the protein is invertible. |
| `timsim-localization` | *default* | Modforms → the site-localization answer key: true position, all candidate positions, and the b/y fragments that actually discriminate between them. |
| `timsim-frag-input` | *default* | Freezes the fragment-prediction input `(precursor_id, [UNIMOD]-annotated sequence, charge)` as an explicit artifact, using the *same* `annotate()` the spectrum builder uses — so predicted intensities and fragment m/z agree on what the molecule is. |
| `timsim-frag-ce` | `tdf` | Per-precursor **mobility-derived collision energy**: CCS → `1/K0` → mobility scan → CE, using the run's own Bruker calibration and the same `ActivationPolicy` v1's dda-PASEF selection drives. An *optional* input to `timsim-fragments` (`--collision-energies`) — without it that stage keeps predicting every precursor at one CE, which is right for Astral and wrong for the timsTOF. |
| `timsim-render-bench` | *default* | The streaming-render prototype + memory benchmark: proves the sweep's working set is bounded by the **elution window, not the run length**. |
| `timsim-spectra` | `tdf` | Peptide ions → the instrument-independent MS1 + MS2 spectra (pure `(m/z, intensity)`, via mscore). The seam that lets one spectrum computation drive any instrument. |
| `timsim-render` | `tdf` | **Bruker timsTOF `.d`** — see below. |
| `timsim-render-thermo` | `thermo` | **Thermo Astral DIA `.raw`**, by authoring into a real template's scan slots (the template *is* the schedule): MS1 isotope centroids + DIA MS2 fragment centroids in each slot's inherited isolation window. `--fragment-spectra` can point at a device-specific predictor (e.g. Orbitrap-HCD) while everything else is held fixed. |
| `timsim-render-sciex` | `sciex` | **SCIEX ZenoTOF SWATH → open mzML** (no IMS). The schedule is *synthesised* from a SWATH window table rather than copied from a template, and the output is vendor-neutral mzML. No `sciexwiff` dependency — the lean path stays legally clean. |

## `timsim-render` — the Bruker path

Feature space → sweep-line render → a real timsTOF `.d` via `ms-io`'s `TdfWriter`.

- **MS1** and **DIA-PASEF** (`--dia`): the reference `.d`'s cycle is replayed and fragments are gated
  by mscore's diagonal quadrupole transmission. `--truth` writes a per-precursor DIA answer key.
- **DDA-PASEF** (`--dda`): MS1 surveys every `--precursors-every` frames, top-N selection with dynamic
  exclusion, band-limited MS2, plus a sidecar per-selection-event answer key (`--dda-truth`).
- **`--reference-d PATH`**: copy a real Bruker `MzCalibration`/`TimsCalibration`/`GlobalMetadata`
  verbatim and *place* ions with that same calibration, so a vendor reader (openTIMS/DiaNN) derives
  the correct m/z and 1/K0. Without it, a self-consistent reference-free converter is used.
- **Noise A1** (`--noise-mz-ppm`, `--noise-frag-ppm`): Gaussian (or `--noise-mz-uniform`) m/z scatter
  as a **ppm envelope**, matching v1's convention where ppm = 3σ.
- **Noise A2** (`--noise-real-data`): real background peaks sampled from the reference `.d` and added
  on top of the synthetic signal.
- **`--noise-only`**: the background-only control run — search it, and subtract its IDs from your FDP
  (`timsim-eval`'s `--background-report`).
- **`--spike-into REAL.d`**: additive overlay onto a **real** `.d`, so the real run supplies the
  background, co-elution and dynamic range while only the synthetic spikes are labeled.
- Chunked (`--render-chunks`) and parallel by default; `--no-parallel` forces the bounded streaming
  path. Any chunk count must produce byte-identical output.

## The CLI contract

Every tool obeys the shared contract in `src/lib.rs`, because as far as necroflow is concerned the
CLI *is* the schema contract:

```text
  --out / --out-*   explicit output paths (necroflow derives them; you never choose)
  --schema          print the output schema and exit
  --explain         print derived physical parameters and exit
  --report FILE     measured accounting as TOML — an error bound is data, not a log line
  --threads N       MUST NOT change the output
```

Each flag appears where it has meaning: `--schema` on the seven feature-space tools (`proteome`,
`digest`, `modify`, `design`, `yield`, `precursors`, `localization`), `--explain` where a stage
derives physical parameters (`modify`, `yield`, `precursors`), `--report` where there is measured
accounting to emit (`yield`), and `--threads` on `modify` (the renderers expose their own
`--render-chunks` / `--no-parallel` / `--encode-threads` knobs, under the same must-not-change-the-
output rule). Every artifact is Parquet, written against a `timsim_schema::tables` schema and stamped
with a producer string, so an output knows what made it.

## Determinism

Analytic stages have **no seed at all** — a digest is an exact expectation, not a sample. Sampled
stages are **identity-keyed**, never sequence-keyed: `timsim-digest --max-peptides` orders peptides
by `splitmix64(peptide_id ^ seed)`, and the renderer's noise draws are seeded per
`(precursor_id, peak_index)`. So adding an ion never reshuffles the others, and neither the thread
count nor the size of the set changes any individual result.

## Driven by necroflow

These binaries are designed to be run as content-addressed DAG nodes by
[timsim-necro](https://github.com/theGreatHerrLebert/timsim-necro), which also holds the v2 design
documents (`REALISM_PLAN.md`, `docs/v2-design/TIMSIM_V2_RENDER.md`, …). The design docs *in this repo*
(`THERMO_PLAN.md`, `DDA_PLAN.md`, `SCIEX_CONSOLIDATION.md`, `NECROFLOW_WIRING.md`,
`A2_REAL_DATA_NOISE.md`, `SPIKE_INTO_REAL.md`, `PARALLEL_RUN_DIA.md`, …) are the design record for
work that has landed here.

**Not in this repo:** `timsim-fragments`, `timsim-ccs` and `timsim-rt` are Python CLIs and live in
[timsim-predict](https://github.com/theGreatHerrLebert/timsim-predict). Scoring lives in
[timsim-eval](https://github.com/theGreatHerrLebert/timsim-eval).

## License

MIT.
