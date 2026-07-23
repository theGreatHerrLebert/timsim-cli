//! `timsim-render` — the measurement stage: render the instrument-independent feature space into a
//! real Bruker timsTOF `.d` (MS1-only for this milestone).
//!
//! Pipeline: read `precursors` + `peptide_rt` → place each precursor on the acquisition grid (m/z→TOF
//! and 1/K0→scan, rt_index→frame) → stream-render isotope envelopes with the proven sweep
//! ([`timsim_cli::render::stream_render_flat`]) → write frames through the Rust-native
//! [`ms_io::data::tdf_writer::TdfWriter`].
//!
//! Two calibration modes:
//!   - **`--reference-d <PATH>`** (recommended): copy the Bruker `MzCalibration`/`TimsCalibration`/
//!     `GlobalMetadata`/`Segments` verbatim from a real `.d`, derive the acquisition geometry
//!     (num_scans, TOF/mobility ranges) from it, and PLACE ions with that same calibration
//!     ([`MzCalibrator`]/[`MobilityCalibrator`], ModelType 2). A vendor reader (openTIMS/DiaNN via the
//!     Bruker SDK) then derives correct m/z and 1/K0 because placement and coefficients agree.
//!   - **fallback** (no reference): the reference-free `SimpleIndexConverter` (sqrt TOF↔m/z, linear
//!     scan↔1/K0) from CLI ranges — a valid, self-consistent file for our own tooling.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, Float64Array, ListArray, StringArray, UInt64Array, UInt8Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility;
use timsim_schema::tables::ion_spectra as SP;
use timsim_schema::tables::precursor_ccs as CCS;

use ms_io::data::calibration::{MobilityCalibrator, MzCalibrator};
use ms_io::data::handle::{
    IndexConverter, SimpleIndexConverter, TimsData, TimsIndexConverter, TimsLazyLoder,
    TimsRawDataLayout,
};
use ms_io::data::meta::{read_global_meta_sql, read_meta_data_sql, read_mz_calibration, read_tims_calibration};
use ms_io::data::tdf_writer::{RenderedFrame, TdfWriter, TdfWriterConfig};
use ms_io::data::utility::flatten_scan_values;
use timsim_cli::render::{stream_render_flat, Geometry, Ion};
use timsim_schema::tables::{peptide_rt as RT, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-render", about = "feature space -> streaming render -> MS1 Bruker .d")]
struct Args {
    #[arg(long)]
    precursors: PathBuf,
    #[arg(long)]
    peptide_rt: PathBuf,
    /// The instrument-independent spectra (`timsim-spectra` output). The render is a pure PROJECTOR:
    /// it reads each ion's materialised MS1 spectrum and places its peaks onto the acquisition grid.
    #[arg(long)]
    ion_spectra: PathBuf,
    /// `precursor_ccs` artifact. When given, each precursor's mobility is CCS→1/K0 (Mason-Schamp,
    /// per-run gas/temperature) — physical mobility, required for a search engine. Without it, mobility
    /// falls back to a non-physical m/z trend (fine for format checks, not for DiaNN).
    #[arg(long)]
    precursor_ccs: Option<PathBuf>,
    /// `peptide_quantities` artifact — the per-sample biological abundance axis. When given, each ion's
    /// intensity is scaled by `amount_amol × ionization_propensity × modform_fraction` (v1's `events`),
    /// restoring the ~6-order dynamic range real DIA data has. WITHOUT it only `charge_fraction` varies,
    /// so every peptide renders at ~the same level (~1 order) — no intense anchors, which a search
    /// engine needs to calibrate against.
    #[arg(long)]
    peptide_quantities: Option<PathBuf>,
    /// Which sample of the design to render (default: first sorted). One render is one sample.
    #[arg(long)]
    sample: Option<String>,
    /// Output `.d` directory (overwritten if it exists).
    #[arg(long)]
    out: PathBuf,
    /// Reference `.d` to copy Bruker calibration/metadata from and place ions with. Omit for the
    /// reference-free Simple calibration.
    #[arg(long)]
    reference_d: Option<PathBuf>,
    /// Number of frames = the run's LENGTH. `0` (default) INHERITS the reference `.d`'s own frame count
    /// (so the render matches the reference gradient, not a fixed 5-min stub) — or 3000 with no reference.
    /// Set a positive value to force a length (e.g. a shorter debug run). NOTE the schedule replays the
    /// reference cycle over this many frames, so a value far from the reference's count desyncs the gradient
    /// from the window timing.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..))]
    n_frames: u32,
    /// Mobility scans per frame (ignored in --reference-d mode: taken from the reference).
    #[arg(long, default_value_t = 709, value_parser = clap::value_parser!(u32).range(1..))]
    n_scans: u32,
    #[arg(long, default_value_t = 30.0)]
    sigma_frames: f64,
    #[arg(long, default_value_t = 4.0)]
    sigma_scans: f64,
    #[arg(long, default_value_t = 3.0)]
    n_sigma: f64,
    /// Simple-mode m/z and 1/K0 ranges + digitizer size (ignored in --reference-d mode).
    #[arg(long, default_value_t = 100.0)]
    mz_min: f64,
    #[arg(long, default_value_t = 1700.0)]
    mz_max: f64,
    #[arg(long, default_value_t = 0.6)]
    im_min: f64,
    #[arg(long, default_value_t = 1.6)]
    im_max: f64,
    #[arg(long, default_value_t = 400_000)]
    digitizer_num_samples: u32,
    #[arg(long, default_value_t = 0.1)]
    cycle_seconds: f64,
    #[arg(long, default_value_t = 100_000.0)]
    intensity_scale: f64,
    /// Drop quantised (scan, tof) bins whose intensity is below this floor. Emulates a peak-picking
    /// cutoff: without it, the 2-D Gaussian spread emits a haze of intensity-1 bins that dominate the
    /// peak count and drown the chromatographic shape in quantisation noise. `1` keeps every non-zero
    /// bin (legacy behaviour).
    #[arg(long, default_value_t = 1)]
    min_peak_intensity: u32,
    /// Incomplete fragmentation (DIA only): the fraction of each precursor that survives the quad
    /// INTACT and bleeds into the MS2 windows, drawn per-ion (identity-keyed on precursor id) in
    /// `[min, max]`. Default `0..0` = full fragmentation. Mirrors v1's `precursor_survival_min/max`;
    /// set e.g. `--precursor-survival-max 0.3` for v1's PFRAG-MED regime.
    #[arg(long, default_value_t = 0.0)]
    precursor_survival_min: f64,
    #[arg(long, default_value_t = 0.0)]
    precursor_survival_max: f64,
    /// Debug cap: render only the first N precursors in file order (0 = all). NOTE this caps INPUT
    /// precursors, not surviving ions — a precursor later dropped for lacking spectra still consumes a
    /// slot, so under `--limit` the ion set can differ from an unlimited run. Fine for quick smoke tests;
    /// don't use it when byte-for-byte comparing against another render.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// DIA render: number of apex-ordered frame chunks to stream the ion spectra in (0 = auto by a
    /// memory budget). Peak memory is one chunk's active ions, not the whole dataset — the render stays
    /// bounded by the elution set regardless of how many precursors there are. Force a value to test the
    /// chunk-stitching (any N ≥ 1 must produce byte-identical output to N = 1).
    #[arg(long, default_value_t = 0)]
    render_chunks: u32,
    /// DIA render: force the STREAMING single-threaded path (bounded to one chunk's ions), disabling the
    /// resident multi-threaded fast path. The fast path is used by default when the ion set is estimated to
    /// fit a memory budget; this forces streaming for huge uncapped datasets or byte-identity testing.
    #[arg(long, default_value_t = false)]
    no_parallel: bool,
    /// Noise A1 — Gaussian m/z scatter on **precursor (MS1)** peaks, as a ppm standard deviation, applied
    /// before m/z→tof so a search engine sees a realistic non-degenerate mass-error distribution to
    /// calibrate against. `0` = off (byte-identical to the noiseless render). Seeded per
    /// `(precursor_id, peak_index)` so adding an ion never reshuffles the others. See `REALISM_PLAN.md`.
    #[arg(long, default_value_t = 0.0)]
    noise_mz_ppm: f64,
    /// Noise A1 — Gaussian m/z scatter on **fragment (MS2)** peaks (ppm standard deviation). `0` = off.
    #[arg(long, default_value_t = 0.0)]
    noise_frag_ppm: f64,
    /// Seed for the noise draws. Same seed + same inputs → identical render; change it to draw a different
    /// (still deterministic) mass-error realisation. Ignored when both noise ppm values are `0`.
    #[arg(long, default_value_t = 0)]
    noise_seed: u64,
    /// Render a DIA run: interleave MS1 + MS2 frames on the reference `.d`'s cycle, gate fragments by
    /// the diagonal quadrupole transmission. Requires `--reference-d` (a DIA `.d` for the schedule).
    #[arg(long, default_value_t = false)]
    dia: bool,
    /// Render a DDA-PASEF run: MS1 surveys every `--precursors-every` frames, top-N precursor selection
    /// with dynamic exclusion, band-limited MS2. Writes a sidecar answer key (`--dda-truth`) tying each
    /// selection event to the true precursor. Requires `--reference-d`. (Oracle-isolation baseline: clean
    /// single-precursor MS2; in-window co-isolation contaminants are a follow-up.)
    #[arg(long, default_value_t = false)]
    dda: bool,
    /// DDA: MS1 survey cadence — every Nth frame is a precursor (MS1) frame; the N-1 between are MS2.
    #[arg(long, default_value_t = 10)]
    precursors_every: u32,
    /// DDA: max precursors packed into one MS2 (PASEF) frame.
    #[arg(long, default_value_t = 25)]
    max_precursors: usize,
    /// DDA: minimum MS1 intensity (abundance × elution) for a precursor to be selectable. Note this is
    /// the currently-uncalibrated abundance scale (see RENDER_CALIBRATION.md); default 0 = top-N only.
    #[arg(long, default_value_t = 0.0)]
    intensity_threshold: f64,
    /// DDA: dynamic-exclusion window in frames — an ion isn't re-selected until this many frames after
    /// its last selection.
    #[arg(long, default_value_t = 25)]
    exclusion_width: u32,
    /// DDA: path for the sidecar answer key (Parquet). Default: `<out>.dda_selected.parquet`.
    #[arg(long)]
    dda_truth: Option<PathBuf>,
    /// DIA: path for the per-precursor answer key (Parquet) — the same 8-column schema as the Thermo
    /// render's `--thermo-truth` (precursor_id, peptide_id, charge, mz, rt_seconds, abundance, has_ms2,
    /// in_any_window). Lets a DiaNN search of the `.d` be scored against the render's ground truth, the
    /// way the Thermo path already closes search→score. Omit to skip the answer key.
    #[arg(long)]
    truth: Option<PathBuf>,
    /// After writing, reopen the `.d` through the rustims reader and report what round-trips.
    #[arg(long, default_value_t = false)]
    verify: bool,
}

/// The acquisition geometry + calibration used to place ions. Closures hide whether the calibration
/// is the reference's Bruker model or the Simple fallback.
struct Placement {
    n_scans: u32,
    tof_max: u32,
    mz_min: f64,
    mz_max: f64,
    im_min: f64,
    im_max: f64,
    reference_d: Option<String>,
    /// The reference `.d`'s own frame count — the run length to inherit when `--n-frames 0`.
    ref_n_frames: Option<u32>,
    to_tof: Box<dyn Fn(f64) -> u32>,
    to_scan: Box<dyn Fn(f64) -> u32>,
    to_mz: Box<dyn Fn(u32) -> f64>,
}

fn build_placement(a: &Args) -> Result<Placement> {
    match &a.reference_d {
        Some(ref_d) => {
            let ref_s = ref_d.to_str().unwrap().to_string();
            let gm = read_global_meta_sql(&ref_s).map_err(|e| anyhow!("read reference GlobalMetadata: {e}"))?;
            let frames = read_meta_data_sql(&ref_s).map_err(|e| anyhow!("read reference Frames: {e}"))?;
            let f0 = frames.first().ok_or_else(|| anyhow!("reference .d has no frames"))?;
            let n_scans = frames.iter().map(|f| f.num_scans).max().unwrap_or(0) as u32;

            // Build the pure-Rust ModelType-2 calibrators from the reference coefficients + frame temps.
            // Select the SAME calibration rows the copied Frames reference (f0's ids) — a reference with
            // several calibrations would otherwise place peaks with coefficients that disagree with what
            // the output stores, so a vendor reader would derive wrong m/z / mobility.
            let mzc_row = read_mz_calibration(&ref_s).map_err(|e| anyhow!("{e}"))?
                .into_iter().find(|c| c.id == f0.mz_calibration)
                .ok_or_else(|| anyhow!("no MzCalibration with id {} in reference", f0.mz_calibration))?;
            let mz = MzCalibrator::new(
                mzc_row.model_type, mzc_row.digitizer_timebase, mzc_row.digitizer_delay,
                mzc_row.t1, mzc_row.t2, mzc_row.dc1, mzc_row.dc2,
                mzc_row.c0, mzc_row.c1, mzc_row.c2, mzc_row.c3, mzc_row.c4, f0.t_1, f0.t_2,
            );
            let mz_for_tof = MzCalibrator::new(
                mzc_row.model_type, mzc_row.digitizer_timebase, mzc_row.digitizer_delay,
                mzc_row.t1, mzc_row.t2, mzc_row.dc1, mzc_row.dc2,
                mzc_row.c0, mzc_row.c1, mzc_row.c2, mzc_row.c3, mzc_row.c4, f0.t_1, f0.t_2,
            );
            let tc_row = read_tims_calibration(&ref_s).map_err(|e| anyhow!("{e}"))?
                .into_iter().find(|c| c.id == f0.tims_calibration)
                .ok_or_else(|| anyhow!("no TimsCalibration with id {} in reference", f0.tims_calibration))?;
            let mob = MobilityCalibrator::new(
                tc_row.c0, tc_row.c1, tc_row.c2, tc_row.c3, tc_row.c4,
                tc_row.c5, tc_row.c6, tc_row.c7, tc_row.c8, tc_row.c9,
            );

            eprintln!(
                "  reference .d: {}  (num_scans {}, tof_max {}, m/z {:.0}-{:.0}, 1/K0 {:.2}-{:.2})",
                ref_s, n_scans, gm.tof_max_index, gm.mz_acquisition_range_lower,
                gm.mz_acquisition_range_upper, gm.one_over_k0_range_lower, gm.one_over_k0_range_upper,
            );
            Ok(Placement {
                n_scans,
                tof_max: gm.tof_max_index,
                mz_min: gm.mz_acquisition_range_lower,
                mz_max: gm.mz_acquisition_range_upper,
                im_min: gm.one_over_k0_range_lower,
                im_max: gm.one_over_k0_range_upper,
                reference_d: Some(ref_s),
                ref_n_frames: Some(frames.len() as u32),
                to_tof: Box::new(move |m| mz_for_tof.mz_to_tof(m)),
                to_scan: Box::new(move |k0| mob.one_over_k0_to_scan(k0)),
                to_mz: Box::new(move |tof| mz.tof_to_mz(tof)),
            })
        }
        None => {
            let conv = SimpleIndexConverter::from_boundaries(
                a.mz_min, a.mz_max, a.digitizer_num_samples, a.im_min, a.im_max, a.n_scans - 1,
            );
            let (tof_intercept, tof_slope) = (conv.tof_intercept, conv.tof_slope);
            let (scan_intercept, scan_slope) = (conv.scan_intercept, conv.scan_slope);
            Ok(Placement {
                n_scans: a.n_scans,
                tof_max: a.digitizer_num_samples,
                mz_min: a.mz_min,
                mz_max: a.mz_max,
                im_min: a.im_min,
                im_max: a.im_max,
                reference_d: None,
                ref_n_frames: None,
                // tof = (sqrt(mz) - tof_intercept) / tof_slope ; scan = (1/K0 - scan_intercept) / scan_slope
                to_tof: Box::new(move |m| ((m.sqrt() - tof_intercept) / tof_slope).max(0.0) as u32),
                to_scan: Box::new(move |k0| ((k0 - scan_intercept) / scan_slope).max(0.0) as u32),
                to_mz: Box::new(move |tof| {
                    let c = SimpleIndexConverter {
                        tof_intercept,
                        tof_slope,
                        scan_intercept,
                        scan_slope,
                    };
                    c.tof_to_mz(0, &vec![tof])[0]
                }),
            })
        }
    }
}

fn main() -> Result<()> {
    let mut a = Args::parse();
    if !(a.intensity_scale.is_finite() && a.intensity_scale > 0.0) {
        return Err(anyhow!("--intensity-scale must be finite and > 0, got {}", a.intensity_scale));
    }
    let p = build_placement(&a)?;
    // Resolve the run length: `--n-frames 0` inherits the reference `.d`'s own frame count (so the render
    // matches the reference GRADIENT, the fix for the 5-min stub that crushed recall via co-elution) — or
    // 3000 without a reference. A positive `--n-frames` is an explicit override.
    if a.n_frames == 0 {
        a.n_frames = p.ref_n_frames.unwrap_or(3000);
        eprintln!(
            "  n_frames = {} ({})",
            a.n_frames,
            if p.ref_n_frames.is_some() { "inherited from reference .d" } else { "default (no reference)" }
        );
    }
    let g = Geometry {
        n_frames: a.n_frames,
        n_scans: p.n_scans,
        sigma_frames: a.sigma_frames,
        sigma_scans: a.sigma_scans,
        n_sigma: a.n_sigma,
    };

    // peptide_id -> rt_index.
    let mut rt: HashMap<u64, f64> = HashMap::new();
    for b in timsim_schema::read(&a.peptide_rt, RT::TABLE)? {
        let id: &UInt64Array = b.column_by_name(RT::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let idx: &Float64Array = b.column_by_name(RT::RT_INDEX).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if Array::is_valid(idx, i) {
                rt.insert(id.value(i), idx.value(i));
            }
        }
    }
    // The index -> frame mapping MUST use the artifact's fixed reference range (stamped over the whole
    // peptide space), NOT the min/max of whatever subset is loaded — otherwise the same peptide lands
    // at a different frame depending on `--limit` or which sample is rendered, defeating the whole
    // point of a portable RT index (rt.py stamps these precisely so no consumer re-derives them).
    let md = timsim_schema::metadata(&a.peptide_rt)?;
    let parse = |key: &str| -> Result<f64> {
        md.get(key)
            .ok_or_else(|| anyhow!("peptide_rt missing {key} — cannot map rt index to frames"))?
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow!("bad {key}: {e}"))
    };
    let lo = parse("timsim.rt.index_min")?;
    let hi = parse("timsim.rt.index_max")?;
    let span = (hi - lo).max(1e-9);

    if a.dda {
        return run_dda(&a, &p, &g, &rt, lo, span);
    }
    if a.dia {
        return run_dia(&a, &p, &g, &rt, lo, span);
    }

    // ── project: place each ion's materialised MS1 spectrum onto the grid ──────
    // The instrument-independent MS1 spectrum lives in ion_spectra; the render only PROJECTS it. Load
    // precursor_id -> MS1 peaks (ms_level 1). (MS2 rows are the projector's business once DIA lands.)
    let mut ms1: HashMap<u64, Vec<(f64, f32)>> = HashMap::new();
    for b in timsim_schema::read(&a.ion_spectra, SP::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
        let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
        let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if level.value(i) != 1 {
                continue;
            }
            let mzv = mz.value(i);
            let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
            let iv = inten.value(i);
            let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
            let peaks: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
            ms1.insert(pcid.value(i), peaks);
        }
    }

    // Precursors give each ion its placement coordinates: CCS -> mobility scan, peptide_id -> elution
    // frame. The peaks themselves come from the materialised spectrum, projected via m/z -> TOF.
    let ccs = load_ccs(&a.precursor_ccs)?;
    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;
    let mut ions: Vec<Ion> = Vec::new();
    let mut skipped = 0usize;
    'outer: for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let frac: &Float32Array = b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        let ionz: &Float32Array = b.column_by_name(PRE::IONIZATION_PROPENSITY).unwrap().as_any().downcast_ref().unwrap();
        let mff: &Float32Array = b.column_by_name(PRE::MODFORM_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(spec) = ms1.get(&pcid.value(i)) else { continue }; // no materialised spectrum
            let Some(&rt_index) = rt.get(&pid.value(i)) else { continue };
            // Map the index range onto the LAST valid 0-based frame (n_frames - 1); scaling by n_frames
            // puts index_max one frame past the end (and disagrees with the DIA path).
            let apex_frame = (rt_index - lo) / span * (a.n_frames as f64 - 1.0);
            let scan = place_scan(pcid.value(i), mz.value(i), chg.value(i).max(1) as u32, &ccs, &p);

            let peaks: Vec<(u32, f32)> = spec
                .iter()
                .filter_map(|&(m, inten)| {
                    if m < p.mz_min || m > p.mz_max {
                        None // out of the acquisition range — the instrument wouldn't record it
                    } else {
                        // Highest valid TOF index is DigitizerNumSamples = tof_max - 1; tof_max itself
                        // is one past the declared range and some readers reject it.
                        Some(((p.to_tof)(m).min(p.tof_max.saturating_sub(1)), inten))
                    }
                })
                .collect();
            if peaks.is_empty() {
                skipped += 1;
                continue;
            }
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            ions.push(Ion { apex_frame, scan_center: scan as f64, abundance, peaks });
            if a.limit > 0 && ions.len() >= a.limit {
                break 'outer;
            }
        }
    }
    eprintln!(
        "  projected {} ions ({} skipped: no in-range peaks) — rendering to {}",
        ions.len(), skipped, a.out.display()
    );

    // ── render -> write ──────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&a.out);
    let cfg = TdfWriterConfig {
        num_scans: p.n_scans,
        digitizer_num_samples: p.tof_max.saturating_sub(1),
        mz_range: (p.mz_min, p.mz_max),
        one_over_k0_range: (p.im_min, p.im_max),
        compression_level: 1,
        scan_mode: 9,
        reference_d: p.reference_d.clone(),
    };
    let mut writer = TdfWriter::create(&a.out, cfg).map_err(|e| anyhow!("{e}"))?;

    let mut next_fid: u32 = 1;
    let mut total_peaks: u64 = 0;
    let mut err: Result<()> = Ok(());
    stream_render_flat(&ions, &g, |e| {
        if err.is_err() {
            return;
        }
        let target = e.frame + 1;
        while next_fid < target {
            if let Err(x) = write_frame(&mut writer, next_fid, 0, a.cycle_seconds, Vec::new(), Vec::new(), Vec::new()) {
                err = Err(x);
                return;
            }
            next_fid += 1;
        }
        let (scans, tofs, ints) = dedup_and_quantise(e.triples, a.intensity_scale, a.min_peak_intensity);
        total_peaks += scans.len() as u64;
        if let Err(x) = write_frame(&mut writer, target, 0, a.cycle_seconds, scans, tofs, ints) {
            err = Err(x);
            return;
        }
        next_fid = target + 1;
    });
    err?;
    while next_fid <= a.n_frames {
        write_frame(&mut writer, next_fid, 0, a.cycle_seconds, Vec::new(), Vec::new(), Vec::new())?;
        next_fid += 1;
    }
    writer.finalize().map_err(|e| anyhow!("{e}"))?;
    println!(
        "  wrote {} frames, {} MS1 peaks ({} calibration) -> {}",
        a.n_frames, total_peaks,
        if p.reference_d.is_some() { "reference Bruker" } else { "Simple" },
        a.out.display()
    );

    if a.verify {
        verify(&a.out, &p)?;
    }
    Ok(())
}

/// One-shot latch so a saturating `intensity_scale` warns once, not once per frame.
static SATURATION_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sum a frame's (possibly duplicated) triples in f64 per (scan, tof), then quantise. Summing before
/// quantising keeps co-eluting sub-quantum signal (the invariant the throughput bench enforces).
///
/// The quantum is `intensity_scale`. Bins below `floor` counts are dropped (a peak-picking cutoff);
/// bins whose scaled value exceeds `u32::MAX` would be silently clipped by the `as u32` saturating cast,
/// so we detect that and warn ONCE — a saturated frame means `intensity_scale` is too hot for the most
/// abundant ion and the dynamic range is being crushed at the top (calibrate the scale down).
/// Dep-free FxHash-style hasher. The DIA sweep spends ~40% of the render in `dedup_and_quantise`, almost
/// all of it SipHashing `(u32,u32)` keys (profiled). SipHash is DoS-resistant, which this hot inner loop
/// does not need; an integer multiply-rotate is far cheaper. Byte-identity is unaffected: the per-key f64
/// sum (accumulated in input-triple order) is unchanged, and `encode_frame_block` re-sorts to canonical
/// block bytes — only the (already-non-canonical) HashMap drain order differs.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}
impl std::hash::Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(K);
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
}
type FxBuild = std::hash::BuildHasherDefault<FxHasher>;

fn dedup_and_quantise(triples: &[(u32, u32, f64)], scale: f64, floor: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    debug_assert!(scale.is_finite() && scale > 0.0, "intensity_scale must be finite and > 0");
    const CEIL: f64 = u32::MAX as f64;
    // Pack (scan, tof) into a single u64 key so one cheap u64 hash replaces the tuple's two-field SipHash.
    let mut summed: HashMap<u64, f64, FxBuild> =
        HashMap::with_capacity_and_hasher(triples.len(), Default::default());
    for &(scan, tof, v) in triples {
        *summed.entry(((scan as u64) << 32) | tof as u64).or_insert(0.0) += v;
    }
    let (mut scans, mut tofs, mut ints) = (Vec::new(), Vec::new(), Vec::new());
    for (key, v) in summed {
        let (scan, tof) = ((key >> 32) as u32, key as u32);
        let scaled = v * scale;
        if scaled >= CEIL
            && !SATURATION_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "  WARNING: intensity saturated a u32 bin (scaled {:.3e} >= {:.3e}) — --intensity-scale \
                 is too high for the most abundant ion; top of the dynamic range is being clipped",
                scaled, CEIL
            );
        }
        let q = scaled.min(CEIL) as u32;
        if q < floor.max(1) {
            continue;
        }
        scans.push(scan);
        tofs.push(tof);
        ints.push(q);
    }
    (scans, tofs, ints)
}

/// Standard TIMS gas / temperature for Mason-Schamp (N2 at ~305 K — the imspy defaults the CCS model
/// was trained against). These are the "per-run" settings the CCS→1/K0 conversion needs.
const MASS_GAS: f64 = 28.013;
const TEMP: f64 = 31.85;
const T_DIFF: f64 = 273.15;

/// `precursor_id -> CCS` (Å²), or an empty map if no artifact is given.
fn load_ccs(path: &Option<PathBuf>) -> Result<HashMap<u64, f64>> {
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };
    for b in timsim_schema::read(path, CCS::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(CCS::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let ccs: &Float64Array = b.column_by_name(CCS::CCS).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            out.insert(pcid.value(i), ccs.value(i));
        }
    }
    Ok(out)
}

/// `peptide_id -> amount_amol` for one sample of the design (the first sorted `sample_id` if `sample`
/// is None). Empty map when no `peptide_quantities` path is given — the caller then falls back to a
/// unit amount, i.e. abundance driven by charge/ionisation propensities only.
fn load_amounts(path: &Option<PathBuf>, sample: &Option<String>) -> Result<HashMap<u64, f64>> {
    use timsim_schema::tables::peptide_quantities as PQ;
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };

    // Resolve the sample: the caller's choice, or the first sorted id present.
    let chosen = match sample {
        Some(s) => s.clone(),
        None => {
            let mut samples: Vec<String> = Vec::new();
            for b in timsim_schema::read(path, PQ::TABLE)? {
                let s: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
                for i in 0..b.num_rows() {
                    samples.push(s.value(i).to_string());
                }
            }
            samples.sort();
            samples.dedup();
            samples.into_iter().next().ok_or_else(|| anyhow!("{} has no samples", path.display()))?
        }
    };

    for b in timsim_schema::read(path, PQ::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PQ::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let sid: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
        let amt: &Float64Array = b.column_by_name(PQ::AMOUNT_AMOL).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if sid.value(i) == chosen {
                out.insert(pid.value(i), amt.value(i));
            }
        }
    }
    Ok(out)
}

/// splitmix64 finaliser: a full-avalanche `u64 -> u64` bijection. Reused as both a 1-word hash and a
/// mixing step when chaining several key fields together.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic `u64 -> [0, 1)`. Identity-keyed randomness: the same id always maps to the same value, so
/// per-ion draws (e.g. survival) don't reshuffle when the ion set changes.
fn hash01(x: u64) -> f64 {
    (splitmix64(x) >> 11) as f64 / (1u64 << 53) as f64
}

/// Deterministic standard-normal draw (mean 0, sd 1) for peak `k` of precursor `pc`, via Box–Muller over
/// two decorrelated uniform draws. Identity-keyed on `(pc, is_frag, k, seed)` so the noise realisation is
/// reproducible and stable under `--limit` — adding or removing ions leaves every other draw unchanged.
/// The fields are folded in through successive `splitmix64` avalanches (NOT a single linear sum), so no
/// field's contribution can be algebraically cancelled by another: the MS1 vs MS2 peak streams (`is_frag`)
/// and distinct `seed`s stay genuinely independent, not merely offset. Used by A1 m/z-ppm scatter.
fn gauss_unit(pc: u64, is_frag: bool, k: usize, seed: u64) -> f64 {
    let mut z = splitmix64(seed ^ if is_frag { 0x9E37_79B9_7F4A_7C15 } else { 0 });
    z = splitmix64(z ^ pc);
    z = splitmix64(z ^ k as u64);
    let u1 = ((splitmix64(z) >> 11) as f64 / (1u64 << 53) as f64).max(1e-12); // clamp away from ln(0)
    let u2 = (splitmix64(z ^ 0xD1B5_4A32_D192_ED03) >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// The mobility scan for a precursor: physical CCS→1/K0 (Mason-Schamp) when its CCS is known, else a
/// non-physical m/z trend. The 1/K0 is clamped to the acquisition band, then mapped to a scan by the
/// run's mobility calibration.
fn place_scan(pcid: u64, mz: f64, charge: u32, ccs: &HashMap<u64, f64>, p: &Placement) -> f64 {
    let one_over_k0 = match ccs.get(&pcid) {
        Some(&c) => ccs_to_one_over_reduced_mobility(c, mz, charge, MASS_GAS, TEMP, T_DIFF),
        None => {
            let f = ((mz - p.mz_min) / (p.mz_max - p.mz_min)).clamp(0.0, 1.0);
            p.im_min + (p.im_max - p.im_min) * f
        }
    };
    (p.to_scan)(one_over_k0.clamp(p.im_min, p.im_max)).min(p.n_scans - 1) as f64
}

/// v1's precursor isolation-m/z metadata from the raw MS1 isotope envelope: keep isotopes above 5% of the
/// max, then mono = first, the envelope span's far end = last, `IsolationMz` = the most-intense isotope.
fn iso_metadata(ms1: &[(f64, f32)], precursor_mz: f64) -> (f64, f64, f64) {
    if ms1.is_empty() {
        return (precursor_mz, precursor_mz, precursor_mz);
    }
    let max_i = ms1.iter().map(|&(_, i)| i).fold(0.0f32, f32::max);
    let kept: Vec<(f64, f32)> = ms1.iter().copied().filter(|&(_, i)| i > 0.05 * max_i).collect();
    let kept = if kept.is_empty() { ms1.to_vec() } else { kept };
    let mono = kept.first().map(|&(m, _)| m).unwrap_or(precursor_mz);
    let last = kept.last().map(|&(m, _)| m).unwrap_or(precursor_mz);
    let largest = kept.iter().fold((mono, 0.0f32), |acc, &(m, i)| if i > acc.1 { (m, i) } else { acc }).0;
    (mono, largest, (mono + last) / 2.0)
}

/// Mobility (scan) window `[lo, hi]` an ion deposits across (n_sigma × sigma_scans around its apex scan).
fn scan_window_dda(scan: i64, g: &Geometry, n_scans: u32) -> (u32, u32) {
    let h = g.n_sigma * g.sigma_scans;
    let lo = (scan as f64 - h).max(0.0) as u32;
    let hi = ((scan as f64 + h) as u32).min(n_scans.saturating_sub(1));
    (lo, hi)
}

/// DDA-PASEF render — MS1 surveys + top-N selection (`timsim_cli::dda`) + band-limited MS2, plus a sidecar
/// answer key tying each selection event to the true precursor. Oracle-isolation baseline: clean single
/// target per band; in-window co-isolation contaminants (and DDA memory streaming) are follow-ups.
fn run_dda(a: &Args, p: &Placement, g: &Geometry, rt: &HashMap<u64, f64>, lo: f64, span: f64) -> Result<()> {
    use ms_io::data::tdf_writer::{DdaPasefWindow, DdaPrecursor, TdfWriter, TdfWriterConfig};
    use timsim_cli::dda::{schedule, Candidate, SelectionParams};
    use timsim_cli::render::gauss_frac;

    let ref_d = a.reference_d.as_ref().ok_or_else(|| anyhow!("--dda requires --reference-d"))?;
    let _ = ref_d;
    let project = |peaks: &[(f64, f32)]| -> Vec<(u32, f32)> {
        peaks.iter().filter_map(|&(m, iv)| {
            if m < p.mz_min || m > p.mz_max { None } else { Some(((p.to_tof)(m).min(p.tof_max.saturating_sub(1)), iv)) }
        }).collect()
    };
    let ccs = load_ccs(&a.precursor_ccs)?;
    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;

    // Load ALL ion_spectra (raw m/z peaks). DDA memory streaming is a follow-up (the DIA path is chunked).
    let (mut ms1_raw, mut ms2_raw): (HashMap<u64, Vec<(f64, f32)>>, HashMap<u64, Vec<(f64, f32)>>) = (HashMap::new(), HashMap::new());
    for b in timsim_schema::read_stream(&a.ion_spectra, SP::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
        let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
        let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let mzv = mz.value(i); let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
            let iv = inten.value(i); let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
            let peaks: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
            match level.value(i) { 1 => { ms1_raw.insert(pcid.value(i), peaks); } 2 => { ms2_raw.insert(pcid.value(i), peaks); } _ => {} }
        }
    }

    struct DdaIon { peptide_id: u64, apex_frame: f64, scan: i64, abundance: f64, ms1: Vec<(u32, f32)>, ms2: Vec<(u32, f32)> }
    let mut ions: HashMap<u64, DdaIon> = HashMap::new();
    let mut cands: Vec<Candidate> = Vec::new();
    let mut order: u32 = 0;
    'outer: for b in timsim_schema::read_stream(&a.precursors, PRE::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let frac: &Float32Array = b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        let ionz: &Float32Array = b.column_by_name(PRE::IONIZATION_PROPENSITY).unwrap().as_any().downcast_ref().unwrap();
        let mff: &Float32Array = b.column_by_name(PRE::MODFORM_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(&rt_index) = rt.get(&pid.value(i)) else { continue };
            let Some(ms1raw) = ms1_raw.remove(&pcid.value(i)) else { continue };
            let apex_frame = 1.0 + (rt_index - lo) / span * (a.n_frames as f64 - 1.0);
            let scan = place_scan(pcid.value(i), mz.value(i), chg.value(i).max(1) as u32, &ccs, p) as i64;
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            let (mono_mz, largest_mz, average_mz) = iso_metadata(&ms1raw, mz.value(i));
            let ms1 = project(&ms1raw);
            let ms2 = ms2_raw.remove(&pcid.value(i)).map(|s| project(&s)).unwrap_or_default();
            if ms1.is_empty() && ms2.is_empty() { continue; }
            // Only a precursor that HAS fragments is selectable — a DDA MS2 with no fragments can't be
            // identified and would be a phantom "eligible" event. It still appears in the MS1 survey.
            if !ms2.is_empty() {
                cands.push(Candidate {
                    precursor_id: pcid.value(i), order, apex_frame, scan_apex: scan,
                    mono_mz, largest_mz, average_mz, charge: chg.value(i).max(1) as i64, abundance,
                    sigma_frames: g.sigma_frames, n_sigma: g.n_sigma,
                });
            }
            ions.insert(pcid.value(i), DdaIon { peptide_id: pid.value(i), apex_frame, scan, abundance, ms1, ms2 });
            order += 1;
            if a.limit > 0 && ions.len() >= a.limit { break 'outer; }
        }
    }

    let params = SelectionParams {
        precursors_every: a.precursors_every.max(1), max_precursors: a.max_precursors,
        intensity_threshold: a.intensity_threshold, exclusion_frames: a.exclusion_width,
        band_half_width: 11, n_scans: p.n_scans, ce_bias: 54.1984, ce_slope: -0.0345,
    };
    let sched = schedule(&cands, &params, a.n_frames);
    eprintln!("  DDA: {} of {} precursors selected, {} MS2 events", sched.precursors.len(), cands.len(), sched.events.len());

    // Sequential TDF precursor ids (vendor requires 1..N; our u64 hash overflows i64). our_id -> tdf_id.
    let tdf_id: HashMap<u64, i64> = sched.precursors.iter().enumerate().map(|(i, c)| (c.precursor_id, i as i64 + 1)).collect();
    let mut events_by_frame: HashMap<i64, Vec<&timsim_cli::dda::SelectionEvent>> = HashMap::new();
    for e in &sched.events { events_by_frame.entry(e.ms2_frame).or_default().push(e); }

    let _ = std::fs::remove_dir_all(&a.out);
    // One source of truth for the zstd level: the writer config and the parallel encoder MUST agree, else
    // the parallel-encoded blocks would differ from what a serial write_frame would emit.
    let compression_level = 1i32;
    let cfg = TdfWriterConfig {
        num_scans: p.n_scans, digitizer_num_samples: p.tof_max.saturating_sub(1),
        mz_range: (p.mz_min, p.mz_max), one_over_k0_range: (p.im_min, p.im_max),
        compression_level, scan_mode: 8, reference_d: p.reference_d.clone(),
    };
    let mut writer = TdfWriter::create(&a.out, cfg).map_err(|e| anyhow!("{e}"))?;

    // Active-set sweep over ions by apex frame, for the MS1 survey deposition.
    let win: Vec<(u32, u32, u64)> = ions.iter().map(|(&id, io)| {
        let h = g.n_sigma * g.sigma_frames;
        ((io.apex_frame - h).max(1.0) as u32, ((io.apex_frame + h) as u32).min(a.n_frames), id)
    }).collect();
    let mut order_start: Vec<usize> = (0..win.len()).collect();
    // Sort by (frame_start, precursor_id) — the precursor_id tiebreak makes the active-set insertion, and
    // therefore the per-frame MS1 deposit order, deterministic (win was built from a HashMap iteration).
    order_start.sort_unstable_by_key(|&i| (win[i].0, win[i].2));
    let per = a.precursors_every.max(1);

    // Parallel render-by-chunk. Split the frame axis into contiguous ranges, render+encode each on a rayon
    // pool, then append the blocks in frame order. Byte-identical to a serial loop: each chunk reproduces
    // the exact per-frame active set (same `order_start` sweep, pre-filled to its first frame) and both
    // `dedup_and_quantise` and `encode_frame_block` are pure, so blocks are position-independent and are
    // appended in the same order. Ions/events are shared read-only; only compressed blocks cross the
    // boundary. `compression_level` is the single local bound with the `TdfWriterConfig` above.
    use rayon::prelude::*;
    let n_frames = a.n_frames;
    // Bind the only `Placement` field the parallel closure needs — `p` itself holds boxed `to_tof`/`from_tof`
    // closures that are not `Sync`, so it must not be captured across the rayon boundary.
    let n_scans = p.n_scans;
    // More chunks than threads so rayon work-steals across the uneven elution density (mid-gradient frames
    // carry far more active ions than the edges); the per-chunk pre-fill is cheap next to deposition.
    let n_chunks = (rayon::current_num_threads() * 8).max(1).min(n_frames.max(1) as usize);
    let bounds: Vec<(u32, u32)> = (0..n_chunks)
        .map(|c| {
            let f0 = 1 + (n_frames as u64 * c as u64 / n_chunks as u64) as u32;
            let f1 = (n_frames as u64 * (c as u64 + 1) / n_chunks as u64) as u32;
            (f0, f1)
        })
        .filter(|&(f0, f1)| f0 <= f1)
        .collect();

    type ChunkOut = (Vec<(u32, u8, ms_io::data::tdf_writer::EncodedBlock)>, u64, u64);
    let chunks: Vec<Result<ChunkOut, String>> = bounds
        .par_iter()
        .map(|&(f0, f1)| {
            // Pre-fill the active set to frame f0 (ions started by f0 and not yet ended), in order_start
            // order — identical to what the serial sweep holds on entering frame f0.
            let mut cursor = 0usize;
            let mut active: Vec<usize> = Vec::new();
            while cursor < win.len() && win[order_start[cursor]].0 <= f0 { active.push(order_start[cursor]); cursor += 1; }
            active.retain(|&i| win[i].1 >= f0);
            let (mut c_ms1, mut c_ms2) = (0u64, 0u64);
            let mut out: Vec<(u32, u8, ms_io::data::tdf_writer::EncodedBlock)> = Vec::with_capacity((f1 - f0 + 1) as usize);
            for frame in f0..=f1 {
                while cursor < win.len() && win[order_start[cursor]].0 <= frame { active.push(order_start[cursor]); cursor += 1; }
                active.retain(|&i| win[i].1 >= frame);
                let is_ms1 = (frame - 1) % per == 0;
                let f = frame as f64;
                let mut tri: Vec<(u32, u32, f64)> = Vec::new();
                if is_ms1 {
                    for &i in &active {
                        let io = &ions[&win[i].2];
                        let ew = gauss_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames);
                        if ew <= 0.0 { continue; }
                        let (slo, shi) = scan_window_dda(io.scan, g, n_scans);
                        for scan in slo..=shi {
                            let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan as f64, g.sigma_scans);
                            if mw <= 0.0 { continue; }
                            let base = io.abundance * ew * mw;
                            for &(tof, iv) in &io.ms1 { tri.push((scan, tof, base * iv as f64)); }
                        }
                    }
                } else if let Some(evs) = events_by_frame.get(&(frame as i64)) {
                    for e in evs {
                        let io = &ions[&e.precursor_id];
                        let ew = gauss_frac(f - 0.5, f + 0.5, io.apex_frame, g.sigma_frames);
                        if ew <= 0.0 { continue; }
                        let s0 = e.scan_begin.max(0) as u32;
                        let s1 = (e.scan_end.min(n_scans as i64 - 1)).max(0) as u32;
                        for scan in s0..=s1 {
                            let mw = gauss_frac(scan as f64 - 0.5, scan as f64 + 0.5, io.scan as f64, g.sigma_scans);
                            if mw <= 0.0 { continue; }
                            let base = io.abundance * ew * mw;
                            for &(tof, iv) in &io.ms2 { tri.push((scan, tof, base * iv as f64)); }
                        }
                    }
                }
                let (scans, tofs, ints) = dedup_and_quantise(&tri, a.intensity_scale, a.min_peak_intensity);
                if is_ms1 { c_ms1 += scans.len() as u64 } else { c_ms2 += scans.len() as u64 }
                let blk = ms_io::data::tdf_writer::encode_frame_block(&scans, &tofs, &ints, n_scans, compression_level)
                    .map_err(|e| e.to_string())?;
                out.push((frame, if is_ms1 { 0u8 } else { 8u8 }, blk));
            }
            Ok((out, c_ms1, c_ms2))
        })
        .collect();

    let (mut ms1_n, mut ms2_n) = (0u64, 0u64);
    for chunk in chunks {
        let (frames_blk, c1, c2) = chunk.map_err(|e| anyhow!("{e}"))?;
        ms1_n += c1;
        ms2_n += c2;
        for (fid, mstype, blk) in frames_blk {
            writer.append_encoded_frame(fid, fid as f64 * a.cycle_seconds, mstype, blk).map_err(|e| anyhow!("{e}"))?;
        }
    }

    let precursors: Vec<DdaPrecursor> = sched.precursors.iter().map(|c| DdaPrecursor {
        id: tdf_id[&c.precursor_id], largest_peak_mz: c.largest_mz, average_mz: c.average_mz,
        monoisotopic_mz: c.mono_mz, charge: c.charge, scan_number: c.scan_apex as f64,
        intensity: c.abundance, parent: c.parent_ms1_frame,
    }).collect();
    let pasef: Vec<DdaPasefWindow> = sched.events.iter().map(|e| DdaPasefWindow {
        frame: e.ms2_frame, scan_num_begin: e.scan_begin, scan_num_end: e.scan_end,
        isolation_mz: e.isolation_mz, isolation_width: e.isolation_width, collision_energy: e.collision_energy,
        precursor: tdf_id[&e.precursor_id],
    }).collect();
    writer.set_dda_schedule(precursors, pasef);
    writer.finalize().map_err(|e| anyhow!("{e}"))?;

    // Sidecar answer key: one row per selection EVENT, keyed on (ms2_frame, scan_begin).
    {
        use arrow::array::{Float64Array, Int64Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        let (mut fr, mut sb, mut se, mut td, mut pc, mut pe, mut ch, mut iso, mut mo, mut pa, mut it, mut rtc):
            (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<u64>, Vec<u64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>, Vec<f64>, Vec<f64>) = Default::default();
        for e in &sched.events {
            let io = &ions[&e.precursor_id];
            fr.push(e.ms2_frame); sb.push(e.scan_begin); se.push(e.scan_end);
            td.push(tdf_id[&e.precursor_id]); pc.push(e.precursor_id); pe.push(io.peptide_id);
            ch.push(e.charge); iso.push(e.isolation_mz); mo.push(e.mono_mz);
            pa.push(e.parent_ms1_frame); it.push(e.event_intensity); rtc.push(io.apex_frame * a.cycle_seconds);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("ms2_frame", DataType::Int64, false),
            Field::new("scan_begin", DataType::Int64, false),
            Field::new("scan_end", DataType::Int64, false),
            Field::new("tdf_precursor_id", DataType::Int64, false),
            Field::new("precursor_id", DataType::UInt64, false),
            Field::new("peptide_id", DataType::UInt64, false),
            Field::new("charge", DataType::Int64, false),
            Field::new("isolation_mz", DataType::Float64, false),
            Field::new("mono_mz", DataType::Float64, false),
            Field::new("parent_ms1_frame", DataType::Int64, false),
            Field::new("event_intensity", DataType::Float64, false),
            Field::new("rt_seconds", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int64Array::from(fr)), Arc::new(Int64Array::from(sb)), Arc::new(Int64Array::from(se)),
            Arc::new(Int64Array::from(td)), Arc::new(UInt64Array::from(pc)), Arc::new(UInt64Array::from(pe)),
            Arc::new(Int64Array::from(ch)), Arc::new(Float64Array::from(iso)), Arc::new(Float64Array::from(mo)),
            Arc::new(Int64Array::from(pa)), Arc::new(Float64Array::from(it)), Arc::new(Float64Array::from(rtc)),
        ])?;
        let truth_path = a.dda_truth.clone().unwrap_or_else(|| a.out.with_extension("dda_selected.parquet"));
        let file = std::fs::File::create(&truth_path)?;
        let mut w = ArrowWriter::try_new(file, schema, None)?;
        w.write(&batch)?;
        w.close()?;
        println!("  wrote DDA .d ({} MS1 + {} MS2 peaks) + {} answer-key events -> {}", ms1_n, ms2_n, sched.events.len(), truth_path.display());
    }
    Ok(())
}

/// DIA render: MS1+MS2 frames on the reference's cycle, fragments gated by the diagonal transmission.
fn run_dia(a: &Args, p: &Placement, g: &Geometry, rt: &HashMap<u64, f64>, lo: f64, span: f64) -> Result<()> {
    use timsim_cli::dia::DiaSchedule;
    use timsim_cli::ms2::{active_frames, dia_render_range, DiaIon};

    let ref_d = a.reference_d.as_ref().ok_or_else(|| anyhow!("--dia requires --reference-d (a DIA .d for the window schedule)"))?;
    let sched = DiaSchedule::from_reference(ref_d.to_str().unwrap(), a.n_frames, p.n_scans)?;
    eprintln!("  DIA schedule: cycle_len {}, {} windows", sched.cycle_len, sched.windows.len());

    // ion_spectra is NOT loaded up front — it is streamed once per apex-chunk below, so peak memory is
    // one chunk's active ions rather than every precursor's spectra at once.

    // Project each precursor's spectra to tof and build DIA ions.
    let project = |pc: u64, is_frag: bool, peaks: &[(f64, f32)]| -> Vec<(u32, f32)> {
        // A1 noise: Gaussian m/z scatter (ppm) applied per peak before m/z→tof. `ppm == 0` keeps `m`
        // untouched, so the noiseless render stays byte-identical.
        let ppm = if is_frag { a.noise_frag_ppm } else { a.noise_mz_ppm };
        peaks
            .iter()
            .enumerate()
            .filter_map(|(k, &(m, inten))| {
                let m = if ppm > 0.0 {
                    m * (1.0 + gauss_unit(pc, is_frag, k, a.noise_seed) * ppm * 1e-6)
                } else {
                    m
                };
                if m < p.mz_min || m > p.mz_max {
                    None
                } else {
                    Some(((p.to_tof)(m).min(p.tof_max.saturating_sub(1)), inten))
                }
            })
            .collect()
    };
    let ccs = load_ccs(&a.precursor_ccs)?;
    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;
    if amounts.is_empty() {
        eprintln!("  WARNING: no peptide_quantities — abundance is charge/ionisation only (~1 order of dynamic range, no anchor precursors)");
    }
    // Incomplete fragmentation is on only when max > 0; clamp to [0,1] and keep min <= max.
    let survival_span: Option<(f64, f64)> = if a.precursor_survival_max > 0.0 {
        if a.precursor_survival_min > a.precursor_survival_max {
            eprintln!(
                "  WARNING: --precursor-survival-min ({}) > --precursor-survival-max ({}); clamping min to max",
                a.precursor_survival_min, a.precursor_survival_max
            );
        }
        let mx = a.precursor_survival_max.clamp(0.0, 1.0);
        let mn = a.precursor_survival_min.clamp(0.0, mx);
        eprintln!("  incomplete fragmentation: precursor survival drawn per-ion in [{mn:.3}, {mx:.3}]");
        Some((mn, mx))
    } else {
        None
    };
    // Metadata pass: one lightweight record per precursor — apex frame, scan, abundance, survival, and its
    // file-order rank. NO peaks. This is the only O(n_precursors) structure that stays resident, and it is
    // tiny (~tens of bytes/ion); the heavy spectra are streamed per chunk below. `order` preserves the
    // precursor-file order so each chunk can visit its ions in the same order the single-pass render did —
    // keeping the per-frame deposit sequence, and thus the output, byte-identical regardless of chunking.
    struct IonMeta {
        apex_frame: f64,
        scan: f64,
        abundance: f64,
        precursor_mz: f64,
        survival: f64,
        order: u32,
        // Identity, carried through for the answer key (`--truth`) — the render itself doesn't need them.
        peptide_id: u64,
        charge: i64,
    }
    let mut meta: HashMap<u64, IonMeta> = HashMap::new();
    let mut order: u32 = 0;
    'outer: for b in timsim_schema::read_stream(&a.precursors, PRE::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let frac: &Float32Array = b.column_by_name(PRE::CHARGE_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        let ionz: &Float32Array = b.column_by_name(PRE::IONIZATION_PROPENSITY).unwrap().as_any().downcast_ref().unwrap();
        let mff: &Float32Array = b.column_by_name(PRE::MODFORM_FRACTION).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let Some(&rt_index) = rt.get(&pid.value(i)) else { continue };
            // 1-based apex frame (the DIA schedule + Frames.Id are 1-based).
            let apex_frame = 1.0 + (rt_index - lo) / span * (a.n_frames as f64 - 1.0);
            let scan = place_scan(pcid.value(i), mz.value(i), chg.value(i).max(1) as u32, &ccs, p);
            // v1's `events`: amount_amol (per sample) × ionisation propensity × modform fraction ×
            // charge fraction. amount 1.0 when no quantities given (propensities-only fallback).
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            // Incomplete-fragmentation survival, drawn per-ion in [min, max] — IDENTITY-KEYED on the
            // precursor id, not a bulk RNG, so adding an ion doesn't reshuffle everyone else's survival.
            let survival = survival_span
                .map(|(mn, mx)| mn + (mx - mn) * hash01(pcid.value(i)))
                .unwrap_or(0.0);
            meta.insert(
                pcid.value(i),
                IonMeta {
                    apex_frame, scan, abundance, precursor_mz: mz.value(i), survival, order,
                    peptide_id: pid.value(i), charge: chg.value(i).max(1) as i64,
                },
            );
            order += 1;
            if a.limit > 0 && meta.len() >= a.limit {
                break 'outer;
            }
        }
    }

    // Chunk the run into apex-ordered frame ranges; each chunk streams ion_spectra once and holds only the
    // ions active in its range. Auto-size to a memory budget (a chunk ≈ budget worth of ions), capped so
    // FD/scan counts stay sane; `--render-chunks` overrides. Peak memory is thus one chunk, not the dataset.
    const BYTES_PER_ION_EST: usize = 3072;
    const BUDGET_BYTES: usize = 512 * 1024 * 1024;
    let n_chunks = if a.render_chunks > 0 {
        a.render_chunks
    } else {
        (meta.len() * BYTES_PER_ION_EST / BUDGET_BYTES + 1).clamp(1, 128) as u32
    }
    .clamp(1, a.n_frames.max(1));
    // Split the frame axis at equal-ION-COUNT quantiles of the elution windows (frame_start), NOT into
    // equal-width slices: elution clusters in time, so equal-width slices would put most ions in the busy
    // middle chunk. Quantile boundaries give every chunk ~n/n_chunks ions with a WELL-DISTRIBUTED elution;
    // peak memory is then one chunk's active set. (It cannot go below the active set itself: a pathological
    // elution window as wide as the whole run makes every ion active everywhere, i.e. O(total) — inherent,
    // not a chunking failure.) `starts` is a transient O(n) u32 vector, dropped once `bounds` is built.
    // bounds[0]=1 .. bounds[n_chunks]=n_frames+1, NON-decreasing (the clamp can repeat a value, which just
    // yields an empty range, handled below); chunk c renders [bounds[c], bounds[c+1]-1].
    let bounds: Vec<u32> = {
        let mut starts: Vec<u32> = meta.values().map(|m| active_frames(m.apex_frame, g).0).collect();
        starts.sort_unstable();
        let mut bounds: Vec<u32> = Vec::with_capacity(n_chunks as usize + 1);
        bounds.push(1);
        for c in 1..n_chunks {
            let mut f = if starts.is_empty() {
                1 + a.n_frames * c / n_chunks
            } else {
                starts[(starts.len() * c as usize / n_chunks as usize).min(starts.len() - 1)]
            };
            let prev = *bounds.last().unwrap();
            f = f.max(prev + 1).min(a.n_frames); // keep non-decreasing; leave room for the final chunk
            bounds.push(f);
        }
        bounds.push(a.n_frames + 1);
        bounds
    };
    eprintln!(
        "  streaming render: {} precursors in {} apex-chunk(s) (equal-ion) -> {}",
        meta.len(), n_chunks, a.out.display()
    );

    // Render + write, per-frame MsMsType.
    let _ = std::fs::remove_dir_all(&a.out);
    let cfg = TdfWriterConfig {
        num_scans: p.n_scans,
        digitizer_num_samples: p.tof_max.saturating_sub(1),
        mz_range: (p.mz_min, p.mz_max),
        one_over_k0_range: (p.im_min, p.im_max),
        compression_level: 1,
        scan_mode: 9,
        reference_d: p.reference_d.clone(),
    };
    let mut writer = TdfWriter::create(&a.out, cfg).map_err(|e| anyhow!("{e}"))?;
    // Persist OUR replayed frame -> window group (DiaFrameMsMsInfo) + the reference's windows.
    let frame_to_group: Vec<(u32, u32)> = (1..=a.n_frames)
        .filter_map(|f| sched.window_group(f).map(|g| (f, g)))
        .collect();
    writer.set_dia_schedule(frame_to_group);

    let mut next_fid: u32 = 1;
    let (mut ms1_peaks, mut ms2_peaks) = (0u64, 0u64);
    // Precursors that projected at least one in-range MS2 fragment — the `has_ms2` truth flag. Accumulated
    // across chunks (a precursor active in several chunks projects the same fragments each time; a set
    // dedups). Only meaningful when `--truth` is requested, but cheap to keep either way.
    let mut with_ms2: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let gap_ms = |f: u32| if sched.ms_level(f) == 1 { 0u8 } else { 9u8 };

    // Fast path (default when the ion set is estimated to fit): project all ions resident ONCE, then
    // render fine frame-chunks in PARALLEL — each encodes its blocks with the pure `encode_frame_block`
    // and they are appended in frame order, which ms-io guarantees is byte-identical to the serial
    // `write_frame` loop. Gated on a memory estimate; `--no-parallel`, an explicit `--render-chunks`, or a
    // dataset too big for the budget fall back to the streaming single-threaded path below (which keeps
    // peak memory bounded to one chunk regardless of dataset size).
    const RESIDENT_BUDGET: u64 = 6 * 1024 * 1024 * 1024; // ions + encoded blocks + overhead
    let use_parallel = !a.no_parallel
        && a.render_chunks == 0
        && (meta.len() as u64) * (BYTES_PER_ION_EST as u64) < RESIDENT_BUDGET;

    if use_parallel {
        use rayon::prelude::*;
        // 1) Build ALL DiaIons resident (one streaming pass). Project m/z->tof HERE, sequentially — the
        //    projection closure captures the non-Sync `Placement`, so it must not cross the rayon boundary.
        let mut builders: HashMap<u64, (bool, Vec<(u32, f32)>, Vec<(u32, f32)>)> =
            HashMap::with_capacity(meta.len());
        for b in timsim_schema::read_stream(&a.ion_spectra, SP::TABLE)? {
            let b = b?;
            let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
            let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
            let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
            let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
            for i in 0..b.num_rows() {
                let pc = pcid.value(i);
                if !meta.contains_key(&pc) { continue; }
                let mzv = mz.value(i); let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
                let iv = inten.value(i); let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
                let raw: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
                let proj = project(pc, level.value(i) == 2, &raw);
                let e = builders.entry(pc).or_insert((false, Vec::new(), Vec::new()));
                match level.value(i) {
                    1 => { e.0 = true; e.1 = proj; }
                    2 => { if !proj.is_empty() { with_ms2.insert(pc); } e.2 = proj; }
                    _ => {}
                }
            }
        }
        // Same keep rule + global-order sort as the serial path (so per-frame deposit order is identical).
        let mut ions: Vec<(u32, DiaIon)> = builders.into_iter().filter_map(|(pc, (had, m1, m2))| {
            if !had || (m1.is_empty() && m2.is_empty()) { return None; }
            let m = meta.get(&pc)?;
            Some((m.order, DiaIon {
                apex_frame: m.apex_frame, scan_center: m.scan, abundance: m.abundance,
                precursor_mz: m.precursor_mz, ms1_peaks: m1, ms2_peaks: m2, survival: m.survival,
            }))
        }).collect();
        ions.sort_unstable_by_key(|x| x.0);
        let ions: Vec<DiaIon> = ions.into_iter().map(|x| x.1).collect();

        // 2) Fine frame-chunks (threads*8) at equal-ion quantiles, for load-balance + work-stealing.
        let pn = ((rayon::current_num_threads() * 8) as u32).clamp(1, a.n_frames.max(1));
        let pbounds: Vec<u32> = {
            let mut starts: Vec<u32> = ions.iter().map(|io| active_frames(io.apex_frame, g).0).collect();
            starts.sort_unstable();
            let mut bnds = Vec::with_capacity(pn as usize + 1);
            bnds.push(1u32);
            for c in 1..pn {
                let mut f = if starts.is_empty() { 1 + a.n_frames * c / pn }
                    else { starts[(starts.len() * c as usize / pn as usize).min(starts.len() - 1)] };
                let prev = *bnds.last().unwrap();
                f = f.max(prev + 1).min(a.n_frames);
                bnds.push(f);
            }
            bnds.push(a.n_frames + 1);
            bnds
        };
        eprintln!("  resident parallel render: {} ions, {} frame-chunks on {} threads -> {}",
            ions.len(), pn, rayon::current_num_threads(), a.out.display());

        // 3) Render + encode each chunk in PARALLEL. The closure is pure: reads `ions`/`sched`/`g`, calls
        //    the Sync `apply_transmission`, and emits owned EncodedBlocks. Gated memory bounds the collect.
        type ChunkOut = (Vec<(u32, u8, ms_io::data::tdf_writer::EncodedBlock)>, u64, u64);
        let (n_scans, iscale, floor) = (p.n_scans, a.intensity_scale, a.min_peak_intensity);
        let chunks: Vec<Result<ChunkOut, String>> = (0..pn).into_par_iter().map(|c| {
            let (fc0, fc1) = (pbounds[c as usize], pbounds[c as usize + 1] - 1);
            if fc1 < fc0 { return Ok((Vec::new(), 0, 0)); }
            // Ions active anywhere in [fc0, fc1] (inclusive overlap), preserving global order.
            let subset: Vec<DiaIon> = ions.iter()
                .filter(|io| { let (fs, fe) = active_frames(io.apex_frame, g); fe >= fc0 && fs <= fc1 })
                .cloned().collect();
            let (mut c1, mut c2) = (0u64, 0u64);
            let mut out: Vec<(u32, u8, ms_io::data::tdf_writer::EncodedBlock)> = Vec::new();
            let mut err: Option<String> = None;
            dia_render_range(&subset, &sched, g, fc0, fc1, |frame, ms_type, tri| {
                if err.is_some() { return; }
                let (scans, tofs, ints) = dedup_and_quantise(tri, iscale, floor);
                if ms_type == 0 { c1 += scans.len() as u64 } else { c2 += scans.len() as u64 }
                match ms_io::data::tdf_writer::encode_frame_block(&scans, &tofs, &ints, n_scans, 1) {
                    Ok(blk) => out.push((frame, ms_type, blk)),
                    Err(e) => err = Some(e.to_string()),
                }
            });
            match err { Some(e) => Err(e), None => Ok((out, c1, c2)) }
        }).collect();

        // 4) Append in frame order (chunks are frame-ordered), filling empty gap frames via `write_frame`
        //    (identical to the serial empty-frame write). Sequential; append is ~0.02% of runtime.
        for chunk in chunks {
            let (blocks, c1, c2) = chunk.map_err(|e| anyhow!("{e}"))?;
            ms1_peaks += c1; ms2_peaks += c2;
            for (frame, ms_type, blk) in blocks {
                while next_fid < frame {
                    write_frame(&mut writer, next_fid, gap_ms(next_fid), a.cycle_seconds, Vec::new(), Vec::new(), Vec::new())?;
                    next_fid += 1;
                }
                writer.append_encoded_frame(frame, frame as f64 * a.cycle_seconds, ms_type, blk)
                    .map_err(|e| anyhow!("{e}"))?;
                next_fid = frame + 1;
            }
        }
    } else {
    for chunk in 0..n_chunks {
        let fc0 = bounds[chunk as usize];
        let fc1 = bounds[chunk as usize + 1] - 1;
        if fc1 < fc0 {
            continue; // empty range (more chunks than distinct apex starts)
        }

        // Build this chunk's active ions by streaming ion_spectra once and keeping only the precursors
        // whose elution window overlaps [fc0, fc1]. (had_ms1, ms1_peaks, ms2_peaks) per precursor.
        let mut builders: HashMap<u64, (bool, Vec<(u32, f32)>, Vec<(u32, f32)>)> = HashMap::new();
        for b in timsim_schema::read_stream(&a.ion_spectra, SP::TABLE)? {
            let b = b?;
            let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
            let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
            let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
            let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
            for i in 0..b.num_rows() {
                let pc = pcid.value(i);
                let Some(m) = meta.get(&pc) else { continue };
                let (fs, fe) = active_frames(m.apex_frame, g);
                if fe < fc0 || fs > fc1 {
                    continue; // not active anywhere in this chunk's frame range
                }
                let mzv = mz.value(i);
                let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
                let iv = inten.value(i);
                let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
                let raw: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
                let proj = project(pc, level.value(i) == 2, &raw);
                let e = builders.entry(pc).or_insert((false, Vec::new(), Vec::new()));
                match level.value(i) {
                    1 => { e.0 = true; e.1 = proj; }
                    2 => { if !proj.is_empty() { with_ms2.insert(pc); } e.2 = proj; }
                    _ => {}
                }
            }
        }

        // Assemble DiaIons — same keep rule as the single-pass render (an MS1 spectrum must exist and at
        // least one projected spectrum is non-empty) — sorted back into precursor-file order so the sweep
        // deposits in the same sequence as the unchunked render (byte-identical output).
        let mut ions: Vec<(u32, DiaIon)> = builders
            .into_iter()
            .filter_map(|(pc, (had_ms1, ms1p, ms2p))| {
                if !had_ms1 || (ms1p.is_empty() && ms2p.is_empty()) {
                    return None;
                }
                let m = meta.get(&pc)?;
                Some((
                    m.order,
                    DiaIon {
                        apex_frame: m.apex_frame,
                        scan_center: m.scan,
                        abundance: m.abundance,
                        precursor_mz: m.precursor_mz,
                        ms1_peaks: ms1p,
                        ms2_peaks: ms2p,
                        survival: m.survival,
                    },
                ))
            })
            .collect();
        ions.sort_unstable_by_key(|x| x.0);
        let ions: Vec<DiaIon> = ions.into_iter().map(|x| x.1).collect();

        let mut err: Result<()> = Ok(());
        dia_render_range(&ions, &sched, g, fc0, fc1, |frame, ms_type, tri| {
            if err.is_err() {
                return;
            }
            while next_fid < frame {
                if let Err(x) = write_frame(&mut writer, next_fid, gap_ms(next_fid), a.cycle_seconds, Vec::new(), Vec::new(), Vec::new()) {
                    err = Err(x);
                    return;
                }
                next_fid += 1;
            }
            let (scans, tofs, ints) = dedup_and_quantise(tri, a.intensity_scale, a.min_peak_intensity);
            if ms_type == 0 { ms1_peaks += scans.len() as u64 } else { ms2_peaks += scans.len() as u64 }
            if let Err(x) = write_frame(&mut writer, frame, ms_type, a.cycle_seconds, scans, tofs, ints) {
                err = Err(x);
                return;
            }
            next_fid = frame + 1;
        });
        err?;
    }
    }
    while next_fid <= a.n_frames {
        write_frame(&mut writer, next_fid, gap_ms(next_fid), a.cycle_seconds, Vec::new(), Vec::new(), Vec::new())?;
        next_fid += 1;
    }
    writer.finalize().map_err(|e| anyhow!("{e}"))?;
    println!("  wrote {} frames ({} MS1 + {} MS2 peaks) -> {}", a.n_frames, ms1_peaks, ms2_peaks, a.out.display());

    // Answer key: per-precursor DIA truth, the SAME 8-column schema render_thermo writes — so the eval
    // harness scores a DiaNN search of this `.d` exactly as it does the Thermo `.raw`. rt_seconds mirrors
    // run_dda (apex_frame × cycle_seconds, the render's own time axis).
    //
    // in_any_window is MOBILITY-AWARE here (the ion-mobility-correct denominator): in dia-PASEF the
    // quadrupole samples a DIAGONAL — window W isolates m/z [center±width/2] only for the mobility scans
    // [scan_begin, scan_end]. A precursor is truly isolatable (co-isolatable with its fragments) iff its
    // m/z AND its placed mobility scan both fall in SOME window. The m/z-only test (used by the no-mobility
    // Thermo/SCIEX writers) OVERCOUNTS on timsTOF: it flags precursors whose m/z is in a window but whose
    // mobility is off the diagonal — never co-isolated, so uncountable in the denominator. Using the full
    // (m/z, scan) window support makes Bruker recall comparable to the no-IMS instruments.
    if let Some(truth) = &a.truth {
        use arrow::array::{BooleanArray, Float64Array as F64, Int64Array, UInt64Array as U64};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        // File-order rows (by `order`) for a reproducible answer key, like render_thermo's `precs`.
        let mut rows: Vec<(&u64, &IonMeta)> = meta.iter().collect();
        rows.sort_unstable_by_key(|(_, m)| m.order);
        let (mut pc, mut pe, mut ch, mut mo, mut rtc, mut ab, mut hm, mut iw):
            (Vec<u64>, Vec<u64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>, Vec<bool>) = Default::default();
        for (&pcid, m) in &rows {
            pc.push(pcid); pe.push(m.peptide_id); ch.push(m.charge);
            mo.push(m.precursor_mz); rtc.push(m.apex_frame * a.cycle_seconds); ab.push(m.abundance);
            hm.push(with_ms2.contains(&pcid));
            // Mobility-aware isolatability: some window covers this precursor in BOTH m/z and mobility scan.
            iw.push(sched.windows.iter().any(|w|
                (m.precursor_mz - w.isolation_mz).abs() <= w.isolation_width * 0.5
                && m.scan >= w.scan_num_begin as f64
                && m.scan <= w.scan_num_end as f64));
        }
        let n = pc.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("precursor_id", DataType::UInt64, false),
            Field::new("peptide_id", DataType::UInt64, false),
            Field::new("charge", DataType::Int64, false),
            Field::new("mz", DataType::Float64, false),
            Field::new("rt_seconds", DataType::Float64, false),
            Field::new("abundance", DataType::Float64, false),
            Field::new("has_ms2", DataType::Boolean, false),
            Field::new("in_any_window", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(U64::from(pc)), Arc::new(U64::from(pe)), Arc::new(Int64Array::from(ch)),
            Arc::new(F64::from(mo)), Arc::new(F64::from(rtc)), Arc::new(F64::from(ab)),
            Arc::new(BooleanArray::from(hm)), Arc::new(BooleanArray::from(iw)),
        ])?;
        let file = std::fs::File::create(truth)?;
        let mut w = ArrowWriter::try_new(file, schema, None)?;
        w.write(&batch)?; w.close()?;
        println!("  answer key ({n} precursors) -> {}", truth.display());
    }

    if a.verify {
        verify(&a.out, p)?;
    }
    Ok(())
}

fn write_frame(
    writer: &mut TdfWriter,
    frame_id: u32,
    ms_ms_type: u8,
    cycle_seconds: f64,
    scans: Vec<u32>,
    tofs: Vec<u32>,
    intensities: Vec<u32>,
) -> Result<()> {
    writer
        .write_frame(&RenderedFrame {
            frame_id,
            retention_time: frame_id as f64 * cycle_seconds,
            ms_ms_type,
            scans,
            tofs,
            intensities,
        })
        .map_err(|e| anyhow!("{e}"))
}

/// Reopen the `.d` through the rustims reader and report what round-trips.
fn verify(dir: &std::path::Path, p: &Placement) -> Result<()> {
    let layout = TimsRawDataLayout::new(dir.to_str().unwrap());
    let n = layout.frame_meta_data.len();
    // Any converter suffices for raw reads; m/z below is computed via the placement's own calibration.
    let sic = SimpleIndexConverter::from_boundaries(p.mz_min, p.mz_max, p.tof_max, p.im_min, p.im_max, p.n_scans - 1);
    let reader = TimsLazyLoder { raw_data_layout: layout, index_converter: TimsIndexConverter::Simple(sic) };

    let mut total = 0u64;
    let mut non_empty = 0u64;
    let mut best: (u32, usize) = (0, 0);
    for fid in 1..=n as u32 {
        let raw = reader.get_raw_frame(fid);
        let peaks = raw.tof.len();
        total += peaks as u64;
        if peaks > 0 {
            non_empty += 1;
            if peaks > best.1 {
                best = (fid, peaks);
            }
        }
    }
    println!();
    println!("  ── verify (reopened through the rustims reader) ────────────");
    println!("  frames read           : {n}");
    println!("  non-empty frames      : {non_empty}");
    println!("  total MS1 peaks        : {total}");
    if best.1 > 0 {
        let raw = reader.get_raw_frame(best.0);
        let scans = flatten_scan_values(&raw.scan, true);
        let mut idx: Vec<usize> = (0..raw.intensity.len()).collect();
        idx.sort_by(|&x, &y| raw.intensity[y].partial_cmp(&raw.intensity[x]).unwrap());
        println!("  busiest frame          : id {} with {} peaks", best.0, best.1);
        for &j in idx.iter().take(3) {
            let mz = (p.to_mz)(raw.tof[j]);
            println!(
                "      scan {:>4}  tof {:>7}  m/z {:>9.4}  intensity {:.0}",
                scans[j], raw.tof[j], mz, raw.intensity[j]
            );
        }
    }
    Ok(())
}
