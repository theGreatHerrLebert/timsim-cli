//! `timsim-render-thermo` (M2) — render the instrument-independent feature space into a real Thermo
//! Orbitrap **Astral DIA `.raw`** by authoring into a template's scan slots (no IMS dimension).
//!
//! The template IS the acquisition schedule: we walk its slots in order and, for each, deposit the
//! eluting biology at that slot's own retention time —
//!   - **MS1 (FTMS profile):** every active precursor's isotope CENTROIDS (peak-shape de-risk settled:
//!     author centroids, not shapes), scaled by `abundance · elution(rt)`.
//!   - **MS2 (ASTMS centroid, DIA):** the fragment centroids of active precursors whose m/z falls in
//!     that slot's inherited isolation window.
//! Out-of-range peaks drop-and-account in the writer; the run-level lost ion current is reported.
//!
//! Multi-device hook (David's idea): MS1 isotopes are instrument-agnostic, so they come from
//! `--ion-spectra`; the MS2 fragment intensities are instrument-DEPENDENT, so `--fragment-spectra` can
//! point at a different predictor's output (e.g. Orbitrap-HCD for Astral) while everything else is held
//! fixed — the same sample "acquired" on another device.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, Float64Array, ListArray, StringArray, UInt64Array, UInt8Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use mscore::timstof::quadrupole::WindowTransmission;
use timsim_core::acquisition::{AcquisitionWriter, ScanDescriptor, ThermoRawWriter};
use timsim_schema::tables::ion_spectra as SP;
use timsim_schema::tables::{peptide_rt as RT, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-render-thermo", about = "feature space -> Thermo Astral DIA .raw (template-based)")]
struct Args {
    #[arg(long)] precursors: PathBuf,
    #[arg(long)] peptide_rt: PathBuf,
    /// MS1 isotope spectra (instrument-agnostic). Also the MS2 source unless --fragment-spectra is set.
    #[arg(long)] ion_spectra: PathBuf,
    /// Optional MS2 fragment spectra from a device-specific predictor (e.g. Orbitrap-HCD). Overrides the
    /// level-2 rows of --ion-spectra — the multi-device hook.
    #[arg(long)] fragment_spectra: Option<PathBuf>,
    #[arg(long)] peptide_quantities: Option<PathBuf>,
    #[arg(long)] sample: Option<String>,
    /// A real Astral DIA `.raw` — supplies the scan schedule + isolation windows we author into.
    #[arg(long)] template: PathBuf,
    #[arg(long)] out: PathBuf,
    /// Chromatographic peak width (Gaussian sigma) in SECONDS of the template gradient.
    #[arg(long, default_value_t = 3.0)] sigma_seconds: f64,
    #[arg(long, default_value_t = 3.0)] n_sigma: f64,
    /// Chromatographic peak shape for the ELUTION axis. `emg` (the DEFAULT) is v1's exponentially
    /// modified Gaussian — a Gaussian of width `--sigma-seconds` convolved with a one-sided
    /// exponential tail of time constant `--emg-k * sigma`. `gaussian` restores the pre-EMG
    /// behaviour BIT-FOR-BIT.
    ///
    /// !!! This default is INVISIBLE to necroflow's command-string fingerprint. See PEAK_SHAPE.md.
    #[arg(long, value_enum, default_value_t = PeakShapeArg::Emg)] peak_shape: PeakShapeArg,
    /// EMG tailing factor `k = 1/(sigma*lambda)`, i.e. the tail time constant in units of sigma.
    /// Default = v1's mean draw, `E[k] = 10/21`. Ignored unless `--peak-shape emg`.
    #[arg(long, default_value_t = timsim_cli::render::V1_DEFAULT_EMG_K)] emg_k: f64,
    /// Quadrupole edge steepness `k` (sigmoid) for the isolation-window transmission — same as the
    /// timsTOF TimsTransmissionDIA default.
    #[arg(long, default_value_t = 15.0)] transmission_k: f64,
    /// Fraction of the template RT span trimmed at each end (avoid loading/wash regions).
    #[arg(long, default_value_t = 0.05)] gradient_trim: f64,
    /// Multiplies every authored peak. `0` (the DEFAULT) CALIBRATES against the template: the
    /// renderer measures the template's own MS1 intensity median, then solves for the scale at which
    /// the median of its OWN authored peaks — counting only those that clear the inherited floor,
    /// which is the same population the template's median is taken over — matches it. Comparing
    /// against every authored peak instead, including the dim ones the floor deletes, biases the
    /// scale up. Orbitrap intensity is an arbitrary
    /// detector unit, not the timsTOF's ion count, so the Bruker path's constant is meaningless
    /// here — carrying `5e5` over from `timsim-render` put the rendered MS1 median four orders of
    /// magnitude BELOW the template's own reporting floor. This is a unit conversion, not a tuning
    /// knob: it sets the axis the floor is then applied on. Pass a positive value to override.
    #[arg(long, default_value_t = 0.0)] intensity_scale: f64,
    /// Drop authored peaks below this floor. `0` (the DEFAULT) inherits the TEMPLATE acquisition's
    /// own reporting floor — the smallest non-zero intensity it ever recorded — for the same reason
    /// `timsim-render` inherits the reference `.d`'s: it is a property of the acquisition being
    /// replayed, not of this simulator. Without a readable template it falls back to 1.
    #[arg(long, default_value_t = 0.0)] min_peak_intensity: f64,
    /// The template's own MS1 intensity MEDIAN, i.e. the target `--intensity-scale 0` calibrates
    /// onto. `0` (the DEFAULT) measures it off the template. Supply it — together with an explicit
    /// `--min-peak-intensity` — when the template is PURE PROFILE: the two numbers must come from
    /// the SAME domain, and a profile template exposes no stored centroids, so the renderer can only
    /// read the baseline. Mixing a centroid-domain floor with a profile-domain median silently
    /// calibrates onto the wrong axis.
    #[arg(long, default_value_t = 0.0)] template_ms1_median: f64,
    /// Resolve the floor + scale, print the calibration record as JSON, and EXIT before authoring.
    ///
    /// This is how a cohort gets ONE calibration constant instead of N. Re-estimating the scale per
    /// render lets a difference in sample composition be absorbed into a compensating rescale —
    /// which, in a cohort whose whole point is a planted per-protein differential, partially cancels
    /// the signal being measured. Calibrate once against the template, freeze the number into the
    /// job config, and every arm renders on the same axis.
    #[arg(long)] calibrate_only: bool,
    /// The MS2 reporting floor. `0` (the DEFAULT) inherits the template's own MS2 floor, which is a
    /// DIFFERENT number from the MS1 one — measured on a stock Exploris DIA run, MS1 censors at
    /// 25,760 and MS2 at 575.5. Reusing the MS1 floor for MS2 censors fragments ~45x too hard, and a
    /// DIA search scores on fragments.
    #[arg(long, default_value_t = 0.0)] min_peak_intensity_ms2: f64,
    /// Sidecar answer key (per-precursor DIA truth).
    #[arg(long)] thermo_truth: Option<PathBuf>,
    /// Durable run manifest (JSON): renderer identity, template digest, method, counts, truth schema.
    #[arg(long)] manifest: Option<PathBuf>,
    /// Fragment model that produced --ion-spectra (recorded in the manifest for reproducibility).
    #[arg(long, default_value = "")] frag_model: String,
    /// Acquisition method label recorded in the manifest (the windows come from the template).
    #[arg(long, default_value = "DIA")] method: String,
    /// The collision energy (NCE) the fragments were predicted at — validated against the template's
    /// actual NCE (a mismatch means the library was built for a different CE than the template was run at).
    #[arg(long)] expected_ce: Option<f64>,
}

/// `p10/p50/p90/p99` of a SORTED slice — the acceptance-test summary. The median alone is a
/// location anchor and is insensitive to exactly the defect that matters here: a large excess at the
/// top can coexist with a near-perfect median whenever fewer than half the peaks are affected.
fn quantiles(sorted: &[f32]) -> [f32; 4] {
    let at = |f: f64| -> f32 {
        if sorted.is_empty() { return f32::NAN; }
        let i = ((sorted.len() - 1) as f64 * f).round() as usize;
        sorted[i.min(sorted.len() - 1)]
    };
    [at(0.10), at(0.50), at(0.90), at(0.99)]
}

/// The template's own MS1 intensity regime: `(floor, median)`, sampling `sample_scans` MS1 scans
/// spread across the acquisition.
///
/// The Bruker sibling of this is `render.rs::reference_intensity_floor`, and the reasoning is the
/// same: the reporting floor is a property of the acquisition being replayed. The difference is
/// that Orbitrap intensity is an ARBITRARY DETECTOR UNIT, so the floor alone is not enough — the
/// median comes back too, to set the axis the floor is applied on.
///
/// Deliberately UNBOUNDED in m/z: any bound can only bias the floor upward by hiding the peak that
/// actually sets it. FTMS MS1 is stored as a profile, so the signal is read out of the profile
/// chunks; a centroided template falls back to `centroid_peaks`.
fn template_level_stats(template: &std::path::Path, level: u8, sample_scans: usize) -> Option<(f32, [f32; 4], usize)> {
    let rf = thermorawfile::RawFile::open(template).ok()?;
    let n = rf.scan_count() as u32;
    if n == 0 {
        return None;
    }
    // Scan numbers for THIS level only. The two levels have genuinely different reporting floors —
    // measured on a stock Exploris DIA run, MS1 censors at 25,760 and MS2 at 575.5, a factor of 45.
    // Applying one floor to both censors MS2 ~45x too hard and guts the fragment signal a DIA search
    // scores on (measured: 0.03x the template's MS2 peaks/scan).
    let ms1: Vec<u32> = (1..=n)
        .filter(|&k| rf.scan_event(k).map(|e| e.ms_order == level as u8).unwrap_or(false))
        .collect();
    if ms1.is_empty() {
        return None;
    }
    // CENTROIDS FIRST, and the domain is the whole point. A profile scan's stored signal includes
    // the BASELINE BETWEEN peaks, so its smallest non-zero value is the detector's noise floor —
    // ~4 orders of magnitude below the smallest peak the instrument actually REPORTS. Inheriting
    // that baseline as the floor makes the cut ~100x too permissive relative to the median, the dim
    // tail survives, and the rendered dynamic range blows past the real one by orders of magnitude
    // (measured: 1.8e13 against the template's 8.8e4). What a search consumes is centroids, and the
    // render deposits discrete authored peaks rather than a true profile, so the peak-reporting
    // floor is the one that governs. Profile is the fallback, and it says so.
    let step = (ms1.len() / sample_scans.max(1)).max(1);
    let sampled: Vec<u32> = ms1.iter().step_by(step).take(sample_scans).copied().collect();

    // ONE DOMAIN FOR THE WHOLE LEVEL. Deciding per scan and concatenating would mix a profile
    // BASELINE (the detector noise between peaks) with CENTROID peak heights in a single pool —
    // they differ by ~4 orders of magnitude, so a handful of fallback scans could set the floor
    // while centroids set the median, and the resulting "floor" and "median" would not describe the
    // same quantity. That is the exact defect this function exists to avoid.
    let n_centroid = sampled.iter().filter(|&&s| !rf.centroid_peaks(s).is_empty()).count();
    let use_centroids = n_centroid * 2 >= sampled.len();     // majority decides, all scans follow
    let mut vals: Vec<f32> = Vec::new();
    for &scan in &sampled {
        if use_centroids {
            vals.extend(rf.centroid_peaks(scan).iter().map(|p| p.intensity).filter(|v| *v > 0.0));
        } else if let Some(prof) = rf.profile(scan) {
            for ch in &prof.chunks {
                vals.extend(ch.signal.iter().copied().filter(|v| *v > 0.0));
            }
        }
    }
    if !use_centroids {
        eprintln!(
            "  WARNING: template MS{level} exposes centroids on only {n_centroid}/{} sampled scans, so \
             the whole level falls back to the PROFILE baseline — well below the instrument's \
             peak-reporting threshold, and it will under-censor the rendered dim tail. Supply \
             --min-peak-intensity (and --template-ms1-median for MS1) from a centroid-domain \
             measurement instead.", sampled.len());
    } else if n_centroid < sampled.len() {
        eprintln!(
            "  note: template MS{level} has centroids on {n_centroid}/{} sampled scans; the \
             centroid-only scans set the statistics and the rest are skipped, so both numbers stay \
             in one domain.", sampled.len());
    }
    if vals.is_empty() {
        return None;
    }
    vals.sort_unstable_by(|a, b| a.total_cmp(b));
    Some((vals[0], quantiles(&vals), vals.len()))
}

fn load_list_spectra(path: &PathBuf, want_level: u8) -> Result<HashMap<u64, Vec<(f64, f32)>>> {
    let mut out: HashMap<u64, Vec<(f64, f32)>> = HashMap::new();
    for b in timsim_schema::read_stream(path, SP::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(SP::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let level: &UInt8Array = b.column_by_name(SP::MS_LEVEL).unwrap().as_any().downcast_ref().unwrap();
        let mz: &ListArray = b.column_by_name(SP::MZ).unwrap().as_any().downcast_ref().unwrap();
        let inten: &ListArray = b.column_by_name(SP::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if level.value(i) != want_level { continue; }
            let mzv = mz.value(i); let mzv: &Float64Array = mzv.as_any().downcast_ref().unwrap();
            let iv = inten.value(i); let iv: &Float32Array = iv.as_any().downcast_ref().unwrap();
            let peaks: Vec<(f64, f32)> = (0..mzv.len()).map(|k| (mzv.value(k), iv.value(k))).collect();
            out.insert(pcid.value(i), peaks);
        }
    }
    Ok(out)
}

fn load_amounts(path: &Option<PathBuf>, sample: &Option<String>) -> Result<HashMap<u64, f64>> {
    use timsim_schema::tables::peptide_quantities as PQ;
    let mut out = HashMap::new();
    let Some(path) = path else { return Ok(out) };
    let chosen = match sample {
        Some(s) => s.clone(),
        None => {
            let mut samples: Vec<String> = Vec::new();
            for b in timsim_schema::read(path, PQ::TABLE)? {
                let s: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
                for i in 0..b.num_rows() { samples.push(s.value(i).to_string()); }
            }
            samples.sort(); samples.dedup();
            samples.into_iter().next().ok_or_else(|| anyhow!("{} has no samples", path.display()))?
        }
    };
    for b in timsim_schema::read(path, PQ::TABLE)? {
        let pid: &UInt64Array = b.column_by_name(PQ::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let sid: &StringArray = b.column_by_name(PQ::SAMPLE_ID).unwrap().as_any().downcast_ref().unwrap();
        let amt: &Float64Array = b.column_by_name(PQ::AMOUNT_AMOL).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if sid.value(i) == chosen { out.insert(pid.value(i), amt.value(i)); }
        }
    }
    Ok(out)
}

struct Prec {
    precursor_id: u64,
    peptide_id: u64,
    mz: f64,
    charge: i64,
    abundance: f64,
    apex_rt: f64,
    ms1: Vec<(f64, f32)>,
    ms2: Vec<(f64, f32)>,
}


/// CLI spelling of [`timsim_cli::render::PeakShape`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum PeakShapeArg { Gaussian, Emg }

impl PeakShapeArg {
    /// Fallible: `--emg-k` is user input, and a `NaN`/negative/infinite `k` used to be absorbed into
    /// a silently different shape rather than refused. See `timsim_cli::render::PeakShape::emg`.
    fn resolve(self, emg_k: f64, n_sigma: f64) -> Result<timsim_cli::render::PeakShape> {
        Ok(match self {
            PeakShapeArg::Gaussian => timsim_cli::render::PeakShape::Gaussian,
            PeakShapeArg::Emg => timsim_cli::render::PeakShape::emg(emg_k, n_sigma)?,
        })
    }
}

fn main() -> Result<()> {
    let mut a = Args::parse();
    // This binary's hand-rolled width checks are what the other three renderers were missing. They
    // now live in one shared validator that all four call, so the rules cannot drift apart again.
    timsim_cli::render::validate_elution_widths("sigma-seconds", a.sigma_seconds, a.n_sigma)?;
    if !(a.gradient_trim.is_finite() && (0.0..0.5).contains(&a.gradient_trim)) {
        return Err(anyhow!("--gradient-trim must be in [0, 0.5)"));
    }
    // 0 is the CALIBRATE sentinel (resolved against the template below), so the guard rejects only
    // non-finite and negative values.
    if !(a.intensity_scale.is_finite() && a.intensity_scale >= 0.0) {
        return Err(anyhow!("--intensity-scale must be finite and >= 0 (0 = calibrate from the template)"));
    }
    if !(a.min_peak_intensity.is_finite() && a.min_peak_intensity >= 0.0) {
        return Err(anyhow!("--min-peak-intensity must be finite and >= 0 (0 = inherit the template's floor)"));
    }
    if !(a.min_peak_intensity_ms2.is_finite() && a.min_peak_intensity_ms2 >= 0.0) {
        return Err(anyhow!("--min-peak-intensity-ms2 must be finite and >= 0 (0 = inherit the template's MS2 floor)"));
    }
    if !(a.template_ms1_median.is_finite() && a.template_ms1_median >= 0.0) {
        return Err(anyhow!("--template-ms1-median must be finite and >= 0 (0 = measure it from the template)"));
    }
    // A supplied target BELOW the floor has no solution: no scale makes the median of the surviving
    // peaks equal a value the floor itself censors. Catch it here rather than letting the solver
    // converge onto an empty survivor set and report a one-element "distribution".
    if a.template_ms1_median > 0.0 && a.min_peak_intensity > 0.0
        && a.template_ms1_median < a.min_peak_intensity {
        return Err(anyhow!(
            "--template-ms1-median {:.4e} is below --min-peak-intensity {:.4e}: the floor censors the \
             target, so no scale satisfies it. The two must come from the SAME domain.",
            a.template_ms1_median, a.min_peak_intensity));
    }
    // EARLY, before the template parse and the input loads: asking to calibrate while pinning the
    // scale is a contradiction, and discovering it after minutes of work helps nobody.
    if a.calibrate_only && a.intensity_scale > 0.0 {
        return Err(anyhow!(
            "--calibrate-only was passed together with an explicit --intensity-scale {:.4e}, so there \
             is nothing to calibrate. Drop --intensity-scale (or set it to 0).", a.intensity_scale));
    }

    // peptide_id -> rt_index, and the artifact's fixed reference range (stamped over the whole space).
    let mut rt: HashMap<u64, f64> = HashMap::new();
    for b in timsim_schema::read(&a.peptide_rt, RT::TABLE)? {
        let id: &UInt64Array = b.column_by_name(RT::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let idx: &Float64Array = b.column_by_name(RT::RT_INDEX).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            if Array::is_valid(idx, i) { rt.insert(id.value(i), idx.value(i)); }
        }
    }
    let md = timsim_schema::metadata(&a.peptide_rt)?;
    let parse = |k: &str| -> Result<f64> {
        md.get(k).ok_or_else(|| anyhow!("peptide_rt missing {k}"))?.trim().parse::<f64>().map_err(|e| anyhow!("bad {k}: {e}"))
    };
    let (lo, hi) = (parse("timsim.rt.index_min")?, parse("timsim.rt.index_max")?);
    let span = (hi - lo).max(1e-9);

    let amounts = load_amounts(&a.peptide_quantities, &a.sample)?;
    let mut ms1_raw = load_list_spectra(&a.ion_spectra, 1)?;
    let mut ms2_raw = load_list_spectra(a.fragment_spectra.as_ref().unwrap_or(&a.ion_spectra), 2)?;

    // Open the template: schedule (rt + isolation per slot) drives the whole render.
    let _ = std::fs::remove_dir_all(&a.out);
    let mut writer = ThermoRawWriter::from_template(&a.template, &a.out).map_err(|e| anyhow!("{e}"))?;
    let manifest = writer.manifest().to_vec();
    // Thermo stores scan retention time in MINUTES; convert to seconds so --sigma-seconds is literal and
    // the answer-key rt_seconds is in seconds.
    let schedule: Vec<(f64, Option<timsim_core::acquisition::IsolationWindow>)> =
        writer.schedule().into_iter().map(|(t, iso)| (t * 60.0, iso)).collect();
    let (ms1_cap, ms2_cap) = writer.capacity();
    // The active-set sweep requires slot RTs finite and nondecreasing in manifest (acquisition) order.
    let mut prev = f64::NEG_INFINITY;
    for (i, (t, iso)) in schedule.iter().enumerate() {
        if !t.is_finite() {
            return Err(anyhow!("template slot {i} has non-finite retention time"));
        }
        if *t + 1e-6 < prev {
            return Err(anyhow!(
                "template slot RTs not monotonic at slot {i} ({t}s < {prev}s) — the sweep needs acquisition order"
            ));
        }
        prev = *t;
        if let Some(w) = iso {
            if !(w.center_mz.is_finite() && w.width_mz.is_finite() && w.width_mz > 0.0) {
                return Err(anyhow!("template slot {i} has a degenerate isolation window"));
            }
        }
    }
    // Gradient window from the (validated monotonic) schedule ends; trim the loading/wash edges.
    let (t0, t1) = (schedule.first().unwrap().0, schedule.last().unwrap().0);
    let trim = (t1 - t0) * a.gradient_trim;
    let (g0, g1) = (t0 + trim, t1 - trim);
    let gspan = (g1 - g0).max(1e-9);
    eprintln!("template: {} slots (MS1={ms1_cap}, MS2={ms2_cap}), gradient {:.1}..{:.1}s", manifest.len(), g0, g1);

    // Build precursors: rt_index -> apex_rt on the analytical gradient (quantile-lite: linear on the
    // trimmed span). abundance = amount (peptide-level); m/z-native isotopes/fragments already predicted.
    let mut precs: Vec<Prec> = Vec::new();
    for b in timsim_schema::read_stream(&a.precursors, PRE::TABLE)? {
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
            let apex_rt = g0 + (rt_index - lo) / span * gspan;
            let amount = amounts.get(&pid.value(i)).copied().unwrap_or(1.0);
            let abundance = amount * ionz.value(i) as f64 * mff.value(i) as f64 * frac.value(i) as f64;
            // Skip non-finite apex/abundance (a NaN rt_index or quantity would poison the sweep sort and
            // the deposition); m/z must be finite for the window test.
            if !(apex_rt.is_finite() && abundance.is_finite() && mz.value(i).is_finite()) { continue; }
            let ms1 = ms1_raw.remove(&pcid.value(i)).unwrap_or_default();
            let ms2 = ms2_raw.remove(&pcid.value(i)).unwrap_or_default();
            if ms1.is_empty() && ms2.is_empty() { continue; }
            precs.push(Prec {
                precursor_id: pcid.value(i), peptide_id: pid.value(i), mz: mz.value(i),
                charge: chg.value(i).max(1) as i64, abundance, apex_rt, ms1, ms2,
            });
        }
    }
    eprintln!("precursors: {} eligible", precs.len());

    // Fail fast: validate the template is compatible with the requested acquisition BEFORE the expensive
    // authoring sweep — a mismatched template should be rejected in milliseconds, not after a multi-minute
    // render that produces a subtly-wrong .raw.
    let n_ms1_slots = manifest.iter().filter(|(_, l, _)| *l == 1).count();
    let n_ms2_slots = manifest.len() - n_ms1_slots;
    let n_window_slots = schedule.iter().filter(|(_, iso)| iso.is_some()).count();
    if precs.is_empty() {
        return Err(anyhow!(
            "no eligible precursors to render — the feature space is empty after MS1/MS2 spectrum \
             filtering; check --ion-spectra / --peptide-quantities for sample {:?}", a.sample));
    }
    if n_ms1_slots == 0 {
        return Err(anyhow!("template {} has no MS1 scans — cannot author precursor signal", a.template.display()));
    }
    if a.method.eq_ignore_ascii_case("DIA") {
        if n_ms2_slots == 0 {
            return Err(anyhow!(
                "template {} has no MS2 scans — it is not a DIA acquisition (method=DIA)", a.template.display()));
        }
        if n_window_slots == 0 {
            return Err(anyhow!(
                "template {} has {} MS2 scans but no parseable isolation windows — cannot author DIA fragments",
                a.template.display(), n_ms2_slots));
        }
    }
    // Soft m/z-coverage check: a template whose isolation windows don't span the sample's precursor m/z
    // range will leave most precursors unfragmented. Warn (don't fail) — it is usually a template mismatch.
    let (mut wlo, mut whi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (_, iso) in schedule.iter() {
        if let Some(w) = iso { wlo = wlo.min(w.center_mz - w.width_mz / 2.0); whi = whi.max(w.center_mz + w.width_mz / 2.0); }
    }
    if a.method.eq_ignore_ascii_case("DIA") && wlo.is_finite() {
        let outside = precs.iter().filter(|p| p.mz < wlo || p.mz > whi).count();
        let frac = outside as f64 / precs.len() as f64;
        if frac > 0.5 {
            eprintln!(
                "  WARNING: {:.0}% of precursors ({}/{}) fall outside the template isolation range \
                 [{:.1}, {:.1}] Th — most will not be fragmented; is this the right template for the sample?",
                frac * 100.0, outside, precs.len(), wlo, whi);
        }
    }
    eprintln!(
        "template check OK: {} MS1 + {} MS2 slots ({} with windows), method={}",
        n_ms1_slots, n_ms2_slots, n_window_slots, a.method);

    // CE validation (#8): the template's own MS2 scans carry the NCE it was acquired at. Compare it to the
    // CE the fragment library was predicted at (--expected-ce); a mismatch means the library and the
    // template disagree on collision energy, so the fragment intensities are for the wrong regime.
    let mut ces: Vec<f64> = schedule.iter()
        .filter_map(|(_, iso)| iso.map(|w| w.collision_energy))
        .filter(|c| c.is_finite() && *c > 0.0)
        .collect();
    let (mut template_nce, mut template_nce_min, mut template_nce_max) = (None, None, None);
    if ces.is_empty() {
        eprintln!("  note: template exposes no per-scan NCE — cannot validate collision energy");
    } else {
        ces.sort_by(f64::total_cmp);
        let (cmin, cmax, median) = (ces[0], ces[ces.len() - 1], ces[ces.len() / 2]);
        template_nce = Some(median);
        template_nce_min = Some(cmin);
        template_nce_max = Some(cmax);
        // A single fragment CE cannot represent a stepped/mixed-NCE acquisition — check the SPREAD across
        // windows, not just the median (which can pass while half the windows are off).
        let stepped = (cmax - cmin) / median.max(1.0) > 0.15;
        if let Some(ece) = a.expected_ce {
            if stepped {
                eprintln!(
                    "  WARNING: template uses stepped/mixed NCE [{:.1}..{:.1}] (median {:.1}) — a single \
                     fragment CE {:.1} cannot match every window", cmin, cmax, median, ece);
            } else if (median - ece).abs() / median.max(1.0) > 0.15 {
                eprintln!(
                    "  WARNING: fragment CE {:.1} differs from template NCE {:.1} by {:.0}% — the library \
                     was predicted at a collision energy the template was not acquired at",
                    ece, median, (median - ece).abs() / median.max(1.0) * 100.0);
            } else {
                eprintln!("  CE check OK: fragment CE {:.1} ≈ template NCE {:.1} [{:.1}..{:.1}]", ece, median, cmin, cmax);
            }
        }
    }

    // Active-set sweep over slots (schedule RT is monotonic). A precursor is active in [apex ± nσ·σ].
    let shape = a.peak_shape.resolve(a.emg_k, a.n_sigma)?;
    let (hleft, hright) = timsim_cli::render::elution_half_widths(a.sigma_seconds, a.n_sigma, &shape);
    let mut order: Vec<usize> = (0..precs.len()).collect();
    order.sort_by(|&x, &y| precs[x].apex_rt.total_cmp(&precs[y].apex_rt)); // total_cmp: NaN-safe (guarded finite above)

    // Inherit the template's intensity regime, for the same reason `timsim-render` inherits the
    // reference `.d`'s reporting floor: both are properties of the acquisition being replayed.
    // Orbitrap intensity is an arbitrary detector unit, so BOTH numbers are needed — a floor without
    // the matching scale would censor the whole render (the Bruker constant put the authored median
    // four orders of magnitude below the template's own floor).
    let need_measure = (a.intensity_scale <= 0.0 && a.template_ms1_median <= 0.0)
        || a.min_peak_intensity <= 0.0;
    let measured = if need_measure { template_level_stats(&a.template, 1, 24) } else { None };
    let tquants = measured.map(|(_, q, _)| q);
    // An explicitly supplied median overrides the measured one, and both are reported so a run's log
    // always states which domain the calibration was done in.
    let tstats: Option<(f32, f32)> = match (measured, a.template_ms1_median > 0.0) {
        (m, true) => Some((m.map(|(f, _, _)| f).unwrap_or(0.0), a.template_ms1_median as f32)),
        (Some((f, q, _)), false) => Some((f, q[1])),
        (None, false) => None,
    };
    if a.template_ms1_median > 0.0 {
        eprintln!("  template MS1 median = {:.4e} (supplied, not measured)", a.template_ms1_median);
    }
    // FLOOR FIRST: the scale is calibrated against the population that SURVIVES the floor, so the
    // floor has to be known before the scale can be.
    if a.min_peak_intensity <= 0.0 {
        a.min_peak_intensity = match tstats {
            Some((tfloor, _)) => {
                eprintln!("  min_peak_intensity = {tfloor:.4e} (inherited from the template's own floor)");
                tfloor as f64
            }
            None => {
                eprintln!("  min_peak_intensity = 1 (template floor unreadable; keeps every non-zero peak)");
                1.0
            }
        };
    }
    if a.min_peak_intensity_ms2 <= 0.0 {
        a.min_peak_intensity_ms2 = match template_level_stats(&a.template, 2, 24) {
            Some((f2, _, _)) => {
                eprintln!("  min_peak_intensity (MS2) = {f2:.4e} (inherited from the template's own MS2 floor)");
                f2 as f64
            }
            None => {
                // Falling back to the MS1 floor is the WRONG default (it over-censors), so fall back
                // to keeping everything and say why.
                eprintln!("  min_peak_intensity (MS2) = 1 (template MS2 floor unreadable; NOT reusing \
                           the MS1 floor, which would over-censor fragments)");
                1.0
            }
        };
    }
    if a.intensity_scale <= 0.0 {
        match tstats {
            Some((_, tmedian)) => {
                // What the sweep is about to author, WITHOUT any scale: the same active-set walk as
                // the real pass, recording only on every `CAL_STRIDE`-th MS1 slot.
                const CAL_STRIDE: usize = 32;
                let (mut cur, mut act, mut vals) = (0usize, Vec::<usize>::new(), Vec::<f32>::new());
                for (si, (&(_s, lvl, _p), &(t, _iso))) in manifest.iter().zip(schedule.iter()).enumerate() {
                    while cur < order.len() && precs[order[cur]].apex_rt - hleft <= t { act.push(order[cur]); cur += 1; }
                    act.retain(|&i| precs[i].apex_rt + hright >= t);
                    if lvl != 1 || si % CAL_STRIDE != 0 { continue; }
                    for &i in &act {
                        let p = &precs[i];
                        let w = timsim_cli::render::elution_ordinate(t, p.apex_rt, a.sigma_seconds, &shape);
                        if w <= 1e-6 { continue; }
                        let base = p.abundance * w;
                        for &(_m, iv) in &p.ms1 { vals.push((base * iv as f64) as f32); }
                    }
                }
                vals.retain(|v| *v > 0.0);
                if vals.is_empty() {
                    return Err(anyhow!(
                        "--intensity-scale 0 (calibrate) but the sweep authored no MS1 signal to calibrate \
                         against; pass an explicit positive --intensity-scale"));
                }
                vals.sort_unstable_by(|x, y| x.total_cmp(y));

                // STATE THE DENOMINATOR. `tmedian` is the median of peaks the template actually
                // WROTE — already above its own reporting floor. The naive ratio
                // `tmedian / median(vals)` compares that against a median taken over EVERY authored
                // peak, including the dim ones this floor is about to delete, which drags the
                // authored median down and biases the scale UP. The two medians have to be over the
                // same population: peaks that survive.
                //
                // Surviving under scale `s` is `v * s >= floor`, i.e. `v >= floor / s` — and since
                // `vals` is sorted that set is a SUFFIX, so each iteration is a binary search plus
                // an index. Solve `s * median(vals[i(s)..]) == tmedian` by fixed point; it converges
                // in a handful of steps because raising `s` only ever admits more (dimmer) peaks,
                // which lowers the surviving median monotonically.
                let floor_v = a.min_peak_intensity as f32;
                let median_from = |i: usize| vals[i + (vals.len() - i) / 2];
                let surviving_lo = |s: f64| -> usize {
                    let cut = (floor_v as f64 / s) as f32;
                    vals.partition_point(|v| *v < cut)
                };
                let mut s = (tmedian / median_from(0)) as f64;   // seed: the naive, biased ratio
                let (mut lo, mut iters, mut converged) = (surviving_lo(s), 0u32, false);
                for _ in 0..64 {
                    iters += 1;
                    if lo >= vals.len() {
                        return Err(anyhow!(
                            "--intensity-scale 0 (calibrate): the floor {:.4e} censors every authored \
                             MS1 peak, so no scale reproduces the target median. Either the template is \
                             not the acquisition this sample belongs to, or the floor and the median \
                             came from different domains.", floor_v));
                    }
                    let next = (tmedian / median_from(lo)) as f64;
                    converged = ((next - s) / s).abs() < 1e-4;
                    s = next;
                    lo = surviving_lo(s);
                    if converged { break; }
                }
                // The loop tests `lo` at the TOP, so the value produced by the LAST assignment has
                // never been checked. If it censors everything, clamping it (as this used to) would
                // report a one-element "survivor distribution" and a meaningless residual while the
                // authoring pass went on to emit nothing.
                if lo >= vals.len() {
                    return Err(anyhow!(
                        "--intensity-scale 0 (calibrate): the solver settled on scale {s:.4e}, at which \
                         the floor {floor_v:.4e} censors every authored MS1 peak. No usable axis exists \
                         for this (template, sample) pair."));
                }
                a.intensity_scale = s;
                let kept = vals.len() - lo.min(vals.len());
                let frac = kept as f64 / vals.len() as f64;
                let surv_median = median_from(lo.min(vals.len() - 1));
                // Post-solve residual. The equation is PIECEWISE — the survivor set jumps
                // discontinuously at `s = floor / v_i` — so an exact root need not exist and the
                // iteration can stop on tolerance rather than on a solution. Reporting
                // |log(achieved/target)| makes that visible instead of letting the solver always
                // return "a" scalar and conceal a model mismatch.
                let residual = ((s * surv_median as f64) / tmedian as f64).ln().abs();
                let surv_q = quantiles(&vals[lo.min(vals.len().saturating_sub(1))..]);
                eprintln!(
                    "  intensity_scale = {:.4e} ({} in {iters} steps; survivors {kept}/{} = {:.1}%; \
                     log-residual {:.3e}; floor {:.4e})",
                    a.intensity_scale, if converged { "converged" } else { "STOPPED AT THE ITERATION CAP" },
                    vals.len(), frac * 100.0, residual, floor_v);
                if !converged {
                    eprintln!(
                        "  WARNING: the fixed point did not converge in {iters} iterations. The map is \
                         piecewise — the survivor set jumps at each `floor/s` — so it can cycle across a \
                         threshold instead of settling. The scale below is the last iterate, not a root.");
                }
                if residual > 0.05 {
                    eprintln!(
                        "  WARNING: calibration did not land on its target (log-residual {:.3e}); the \
                         survivor set is discontinuous in the scale, so no exact solution may exist \
                         here. Treat the axis as approximate.", residual);
                }
                if frac < 0.02 || frac > 0.98 {
                    eprintln!(
                        "  WARNING: the floor censors {:.1}% of authored peaks — a degenerate \
                         censoring regime. Either the floor and the median came from DIFFERENT \
                         domains (a profile baseline paired with a centroid median, or the reverse), \
                         or this template is not the acquisition this sample belongs to.",
                        (1.0 - frac) * 100.0);
                }
                // ACCEPTANCE TEST, reported and not enforced. Matching a median proves the location
                // is right and nothing else; the upper quantiles are where the signal-spreading
                // defect lives. These WARN rather than fail because that defect is known and open —
                // a fail-closed gate here would block every render and produce nothing.
                if tquants.is_none() {
                    eprintln!(
                        "  acceptance: SKIPPED. The template median was supplied rather than measured, \
                         so this renderer does not know which domain it is in and cannot build a \
                         like-for-like quantile table. Run the external centroid-domain probe against \
                         the rendered output and the template — that comparison is the authority for \
                         a pure-profile template.");
                }
                if let Some(tq) = tquants {
                    let names = ["p10", "p50", "p90", "p99"];
                    eprintln!("  acceptance (authored*scale vs template, same domain):");
                    for i in 0..4 {
                        let got = surv_q[i] as f64 * s;
                        let want = tq[i] as f64;
                        let ratio = if want > 0.0 { got / want } else { f64::NAN };
                        let flag = if ratio.is_finite() && (0.2..5.0).contains(&ratio) { "ok" } else { "OFF" };
                        eprintln!("    {:>3}  {:.4e} vs {:.4e}  = {:.3}x  [{}]", names[i], got, want, ratio, flag);
                    }
                }
                if a.calibrate_only {
                    println!(
                        "{{\"template\":{:?},\"intensity_scale\":{:.6e},\"min_peak_intensity\":{:.6e},\
                         \"survivor_fraction\":{:.6},\"iterations\":{},\"log_residual\":{:.6e},\
                         \"template_median\":{:.6e},\"sampled_peaks\":{}}}",
                        a.template.display().to_string(), a.intensity_scale, a.min_peak_intensity,
                        frac, iters, residual, tmedian, vals.len());
                    eprintln!("  --calibrate-only: exiting before authoring. Freeze intensity_scale \
                               and min_peak_intensity into the job config so every arm shares one axis.");
                    return Ok(());
                }
            }
            None => {
                return Err(anyhow!(
                    "--intensity-scale 0 (calibrate) but the template {} could not be read for its \
                     MS1 intensity regime; pass an explicit positive --intensity-scale",
                    a.template.display()));
            }
        }
    }
    let floor = a.min_peak_intensity as f32;
    let floor_ms2 = a.min_peak_intensity_ms2 as f32;

    // The .raw peak count is a u32 on disk, but the thermorawfile author_centroids/author_profile
    // functions currently guard at u16::MAX (65_535) and also must fit the template scan's existing
    // packet budget — so authoring more than this errors. We respect that here. (FOLLOW-UP: relaxing the
    // writer's u16 guard to u32 + a repack path would let very dense scans keep all peaks; the format
    // supports it. Until then this keeps the most intense peaks — realistic centroiding — and accounts
    // for the rest.)
    const MAX_PEAKS: usize = 65_535;
    let (mut cursor, mut ms1_n, mut ms2_n) = (0usize, 0u64, 0u64);
    let (mut capped_slots, mut capped_peaks) = (0u64, 0u64);
    let mut active: Vec<usize> = Vec::new();
    for (slot, (&(_scan, ms_level, _is_profile), &(t, iso))) in manifest.iter().zip(schedule.iter()).enumerate() {
        // Advance/retract the active set to slot time t.
        while cursor < order.len() && precs[order[cursor]].apex_rt - hleft <= t { active.push(order[cursor]); cursor += 1; }
        active.retain(|&i| precs[i].apex_rt + hright >= t);

        let mut peaks: Vec<(f64, f32)> = Vec::new();
        if ms_level == 1 {
            for &i in &active {
                let p = &precs[i];
                let w = timsim_cli::render::elution_ordinate(t, p.apex_rt, a.sigma_seconds, &shape);
                if w <= 1e-6 { continue; }
                let base = p.abundance * w * a.intensity_scale;
                for &(m, iv) in &p.ms1 {
                    let v = (base * iv as f64) as f32;
                    if v >= floor { peaks.push((m, v)); }
                }
            }
        } else if let Some(w) = iso {
            // Quadrupole isolation is a flat-top passband with sigmoid soft edges (mscore's
            // WindowTransmission — the no-IMS sibling of the TimsTransmissionDIA curve the timsTOF path
            // uses), NOT a hard rectangle: an edge precursor is only partially transmitted, so its
            // fragments contribute proportionally.
            let wt = WindowTransmission::new(w.center_mz, w.width_mz, a.transmission_k);
            for &i in &active {
                let p = &precs[i];
                let tprob = wt.probabilities(&[p.mz])[0];
                if tprob <= 1e-3 { continue; }
                let ew = timsim_cli::render::elution_ordinate(t, p.apex_rt, a.sigma_seconds, &shape);
                if ew <= 1e-6 { continue; }
                let base = p.abundance * ew * tprob * a.intensity_scale;
                for &(m, iv) in &p.ms2 {
                    let v = (base * iv as f64) as f32;
                    if v >= floor_ms2 { peaks.push((m, v)); }
                }
            }
        }
        // Respect the format's per-spectrum peak ceiling: at very high co-elution density a slot can
        // exceed 65_535 peaks. Keep the most intense (what a real instrument's centroiding does) rather
        // than aborting the whole render, and account for what was dropped so the cap is never silent.
        if peaks.len() > MAX_PEAKS {
            peaks.sort_unstable_by(|x, y| y.1.total_cmp(&x.1)); // intensity desc
            capped_peaks += (peaks.len() - MAX_PEAKS) as u64;
            capped_slots += 1;
            peaks.truncate(MAX_PEAKS);
            peaks.sort_unstable_by(|x, y| x.0.total_cmp(&y.0)); // restore m/z order for the writer
        }
        if ms_level == 1 { ms1_n += peaks.len() as u64; } else { ms2_n += peaks.len() as u64; }
        // isolation:None preserves the template's inherited DIA window (we don't re-window).
        let desc = ScanDescriptor { ms_level, retention_time: t, isolation: None, peaks };
        writer.write_scan(&desc).map_err(|e| anyhow!("slot {slot}: {e}"))?;
    }
    writer.finalize().map_err(|e| anyhow!("{e}"))?;
    let ps = writer.profile_summary();
    eprintln!(
        "wrote Astral DIA .raw ({ms1_n} MS1 + {ms2_n} MS2 authored peaks) -> {}\n  MS1 drop tally: {} bins written, {} peaks dropped (ion current {:.3e})\n  per-slot peak cap: {} slots capped at {} peaks, {} peaks dropped",
        a.out.display(), ps.written_bins, ps.dropped_total(), ps.dropped_intensity,
        capped_slots, MAX_PEAKS, capped_peaks
    );

    // Answer key: per-precursor DIA truth (join in the harness by peptide_id -> sequence + charge + mz).
    if let Some(truth) = &a.thermo_truth {
        use arrow::array::{BooleanArray, Float64Array as F64, Int64Array, UInt64Array as U64};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        // Distinct MS2 isolation windows (the DIA scheme repeats), as quad-transmission profiles for the
        // in-window eligibility flag (transmitted > 0.5 by any window — consistent with the render).
        let mut wpairs: Vec<(f64, f64)> = schedule.iter()
            .filter_map(|(_, iso)| iso.map(|w| (w.center_mz, w.width_mz)))
            .collect();
        wpairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
        wpairs.dedup_by(|x, y| (x.0 - y.0).abs() < 1e-6 && (x.1 - y.1).abs() < 1e-6);
        let windows: Vec<WindowTransmission> = wpairs.iter()
            .map(|&(c, w)| WindowTransmission::new(c, w, a.transmission_k)).collect();
        let (mut pc, mut pe, mut ch, mut mo, mut rtc, mut ab, mut hm, mut iw):
            (Vec<u64>, Vec<u64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>, Vec<bool>) = Default::default();
        for p in &precs {
            pc.push(p.precursor_id); pe.push(p.peptide_id); ch.push(p.charge);
            mo.push(p.mz); rtc.push(p.apex_rt); ab.push(p.abundance);
            // Eligibility for DIA: a precursor can only be identified if it has fragments AND its m/z
            // falls in some inherited isolation window. The harness uses these to define the denominator.
            hm.push(!p.ms2.is_empty());
            iw.push(windows.iter().any(|wt| wt.probabilities(&[p.mz])[0] > 0.5));
        }
        // The answer key self-identifies its elution kernel: `truth.parquet` used to be BYTE-IDENTICAL
        // between a Gaussian and an EMG render, so a scored result could not be traced back to the
        // shape that produced it. Arrow schema metadata, so `timsim_schema::metadata()` — the reader
        // the pipeline already uses for `peptide_rt` — picks it up. Footer only: the data pages are
        // unchanged. See `timsim_cli::provenance`.
        let schema = Arc::new(Schema::new_with_metadata(vec![
            Field::new("precursor_id", DataType::UInt64, false),
            Field::new("peptide_id", DataType::UInt64, false),
            Field::new("charge", DataType::Int64, false),
            Field::new("mz", DataType::Float64, false),
            Field::new("rt_seconds", DataType::Float64, false),
            Field::new("abundance", DataType::Float64, false),
            Field::new("has_ms2", DataType::Boolean, false),
            Field::new("in_any_window", DataType::Boolean, false),
        ],
            timsim_cli::provenance::schema_metadata(&shape, a.n_sigma),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(U64::from(pc)), Arc::new(U64::from(pe)), Arc::new(Int64Array::from(ch)),
            Arc::new(F64::from(mo)), Arc::new(F64::from(rtc)), Arc::new(F64::from(ab)),
            Arc::new(BooleanArray::from(hm)), Arc::new(BooleanArray::from(iw)),
        ])?;
        let file = std::fs::File::create(truth)?;
        let mut w = ArrowWriter::try_new(file, schema, None)?;
        w.write(&batch)?; w.close()?;
        eprintln!("  answer key ({} precursors) -> {}", precs.len(), truth.display());
    }

    // Durable run manifest: the auditable boundary for a render. Records renderer identity, template
    // identity (path + size + mtime — the robust file identity the flow also hashes for invalidation),
    // the fragment model / method, the content-addressed input paths (their hashes ARE the artifact ids),
    // and the render's own counts. This is what makes a `.raw` reproducible after the fact.
    if let Some(mpath) = &a.manifest {
        let tmeta = std::fs::metadata(&a.template).ok();
        let tbytes = tmeta.as_ref().map(|m| m.len());
        let tmtime = tmeta.as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        let manifest_json = serde_json::json!({
            "renderer": { "name": "timsim-render-thermo", "version": env!("CARGO_PKG_VERSION") },
            "acquisition": {
                "method": a.method,
                "windows_from_template": true,
                "template_nce": template_nce,
                "template_nce_min": template_nce_min,
                "template_nce_max": template_nce_max,
                "fragment_ce": a.expected_ce,
            },
            "template": { "path": a.template.display().to_string(), "bytes": tbytes, "mtime_unix": tmtime },
            "fragment_model": a.frag_model,
            "sample": a.sample,
            "intensity_scale": a.intensity_scale,
            "inputs": {
                "precursors": a.precursors.display().to_string(),
                "ion_spectra": a.ion_spectra.display().to_string(),
                "peptide_rt": a.peptide_rt.display().to_string(),
                "peptide_quantities": a.peptide_quantities.as_ref().map(|p| p.display().to_string()),
            },
            "counts": {
                "precursors_eligible": precs.len(),
                "template_slots": manifest.len(),
                "ms1_slots": n_ms1_slots,
                "ms2_slots": n_ms2_slots,
                "ms1_peaks_authored": ms1_n,
                "ms2_peaks_authored": ms2_n,
            },
            "peak_cap": { "max_peaks_per_slot": MAX_PEAKS, "slots_capped": capped_slots, "peaks_dropped": capped_peaks },
            "ms1_profile_drop": { "bins_written": ps.written_bins, "peaks_dropped": ps.dropped_total(), "ion_current_dropped": ps.dropped_intensity },
            "truth": {
                "path": a.thermo_truth.as_ref().map(|p| p.display().to_string()),
                "rows": precs.len(),
                "columns": ["precursor_id","peptide_id","charge","mz","rt_seconds","abundance","has_ms2","in_any_window"],
            },
        });
        std::fs::write(mpath, serde_json::to_string_pretty(&manifest_json)?)?;
        eprintln!("  run manifest -> {}", mpath.display());
    }
    Ok(())
}
