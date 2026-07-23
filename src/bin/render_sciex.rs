//! `timsim-render-sciex` — render the instrument-independent feature space into an open **mzML** SWATH
//! acquisition (SCIEX ZenoTOF, no IMS), the lean v2 counterpart of the v1 `timsim` build-from-`.wiff`.
//!
//! The Thermo path authors into a real `.raw` TEMPLATE (the template IS the schedule). SCIEX has no
//! open template, and the native `.wiff` writer (`sciexwiff`) is legal-held — so here the schedule is
//! SYNTHESISED from a SWATH window table via the public `timsim-core` acquisition scheme, and the output
//! is vendor-neutral mzML (DiaNN/alphaDIA-readable). Same render kernels as `timsim-render-thermo`:
//!   - **MS1:** every active precursor's isotope centroids, scaled by `abundance · elution(rt)`.
//!   - **MS2:** the fragment centroids of active precursors whose m/z falls in that window (soft-edge
//!     quadrupole transmission), scaled by `abundance · elution · transmission`.
//! No `.wiff`/`sciexwiff` dependency — the lean mzML render stays legally clean (native `.wiff` is a
//! separate rustims-local satellite reusing the validated `sciexwiff` writer).

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float32Array, Float64Array, ListArray, StringArray, UInt64Array, UInt8Array};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use mscore::timstof::quadrupole::WindowTransmission;
use timsim_core::mzml::{write_scans_mzml, MzMlIsolation, MzMlScan};
use timsim_core::scheme::{
    Analyzer, AcquisitionScheme, CollisionEnergyPolicy, DataMode, DiaGeometry, DiaWindow,
    InstrumentKind, IsolationWindow, Ms1Event, RepeatPolicy,
};
use timsim_schema::tables::ion_spectra as SP;
use timsim_schema::tables::{peptide_rt as RT, precursors as PRE};

#[derive(Parser)]
#[command(name = "timsim-render-sciex", about = "feature space -> SCIEX ZenoTOF SWATH mzML (synthesised schedule)")]
struct Args {
    #[arg(long)] precursors: PathBuf,
    #[arg(long)] peptide_rt: PathBuf,
    /// MS1 isotope spectra (instrument-agnostic). Also the MS2 source unless --fragment-spectra is set.
    #[arg(long)] ion_spectra: PathBuf,
    /// Optional MS2 fragment spectra from a device-specific predictor. Overrides level-2 of --ion-spectra.
    #[arg(long)] fragment_spectra: Option<PathBuf>,
    #[arg(long)] peptide_quantities: Option<PathBuf>,
    #[arg(long)] sample: Option<String>,
    /// Output mzML file.
    #[arg(long)] out: PathBuf,

    // ── SWATH schedule (synthesised — no .wiff) ──
    /// Optional window table: lines `center_mz,width_mz[,ce]`. Overrides the fixed-width generator below.
    #[arg(long)] windows: Option<PathBuf>,
    /// Fixed-width SWATH generator (used when --windows is absent): window coverage + width.
    #[arg(long, default_value_t = 400.0)] mz_min: f64,
    #[arg(long, default_value_t = 1200.0)] mz_max: f64,
    #[arg(long, default_value_t = 25.0)] window_width: f64,
    /// Cycle time (one MS1 + all SWATH windows) in seconds.
    #[arg(long, default_value_t = 3.0)] cycle_time_s: f64,
    /// Gradient length in seconds (the SWATH runs from 0 to this).
    #[arg(long, default_value_t = 1800.0)] gradient_length_s: f64,

    // ── render kernel (shared with render_thermo) ──
    /// Chromatographic peak width (Gaussian sigma) in seconds.
    #[arg(long, default_value_t = 3.0)] sigma_seconds: f64,
    #[arg(long, default_value_t = 3.0)] n_sigma: f64,
    /// Quadrupole edge steepness `k` for the isolation-window transmission.
    #[arg(long, default_value_t = 15.0)] transmission_k: f64,
    /// Fraction of the RT span trimmed at each end (avoid loading/wash regions).
    #[arg(long, default_value_t = 0.05)] gradient_trim: f64,
    #[arg(long, default_value_t = 1.0e5)] intensity_scale: f64,
    #[arg(long, default_value_t = 1.0)] min_peak_intensity: f64,
    /// Collision energy recorded on each MS2 window (the CE the fragments were predicted at).
    #[arg(long, default_value_t = 30.0)] collision_energy: f64,

    /// Sidecar answer key (per-precursor DIA truth) — same 8-column schema as the Thermo/Bruker paths.
    #[arg(long)] truth: Option<PathBuf>,
    /// Fragment model that produced --ion-spectra (recorded for provenance).
    #[arg(long, default_value = "")] frag_model: String,
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

/// Build the SWATH window table: either read `center,width[,ce]` rows, or generate contiguous
/// fixed-width windows over [mz_min, mz_max]. Returns (center, width) pairs; CE is applied uniformly.
fn build_windows(a: &Args) -> Result<Vec<(f64, f64)>> {
    if let Some(path) = &a.windows {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow!("read {}: {e}", path.display()))?;
        let mut w = Vec::new();
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 2 { return Err(anyhow!("{}:{}: need center,width[,ce]", path.display(), ln + 1)); }
            let c: f64 = f[0].trim().parse().map_err(|e| anyhow!("{}:{} center: {e}", path.display(), ln + 1))?;
            let wd: f64 = f[1].trim().parse().map_err(|e| anyhow!("{}:{} width: {e}", path.display(), ln + 1))?;
            if !(c.is_finite() && wd.is_finite() && wd > 0.0) { return Err(anyhow!("{}:{}: degenerate window", path.display(), ln + 1)); }
            w.push((c, wd));
        }
        if w.is_empty() { return Err(anyhow!("{} has no windows", path.display())); }
        Ok(w)
    } else {
        if !(a.mz_max > a.mz_min && a.window_width > 0.0) {
            return Err(anyhow!("need mz_max > mz_min and window_width > 0"));
        }
        let n = ((a.mz_max - a.mz_min) / a.window_width).ceil() as usize;
        Ok((0..n).map(|k| (a.mz_min + a.window_width * (k as f64 + 0.5), a.window_width)).collect())
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    for (name, v, pos) in [("sigma-seconds", a.sigma_seconds, true), ("intensity-scale", a.intensity_scale, true)] {
        if !(v.is_finite() && (!pos || v > 0.0)) { return Err(anyhow!("--{name} must be finite and > 0")); }
    }
    if !(a.gradient_trim.is_finite() && (0.0..0.5).contains(&a.gradient_trim)) {
        return Err(anyhow!("--gradient-trim must be in [0, 0.5)"));
    }

    // peptide_id -> rt_index + the artifact's fixed reference range.
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

    // ── Synthesise the SWATH schedule (no .wiff) ──
    let win_pairs = build_windows(&a)?;
    let (cover_lo, cover_hi) = win_pairs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &(c, w)| {
        (l.min(c - w / 2.0), h.max(c + w / 2.0))
    });
    let windows: Vec<DiaWindow> = win_pairs.iter().map(|&(c, w)| DiaWindow {
        isolation: IsolationWindow { center_mz: c, width_mz: w },
        collision_energy: CollisionEnergyPolicy::Value(a.collision_energy),
        geometry: DiaGeometry::MzOnly,
    }).collect();
    let ms1 = Ms1Event { analyzer: Analyzer::Tof, data_mode: DataMode::Centroid, mz_range: Some((cover_lo, cover_hi)), duration_s: None };
    let scheme = AcquisitionScheme::from_window_table(
        InstrumentKind::SciexZenoTof, ms1, windows,
        RepeatPolicy::FixedCycleTime { cycle_time_s: a.cycle_time_s, gradient_length_s: a.gradient_length_s, start_time_s: 0.0 },
        (cover_lo, cover_hi),
    );
    scheme.validate().map_err(|e| anyhow!("SWATH scheme invalid: {e}"))?;
    // (scan, rt_s, ms_level, center, width, ce) — RT already in seconds.
    let sched = scheme.dia_frame_schedule();
    if sched.is_empty() { return Err(anyhow!("empty SWATH schedule (cycle_time {} > gradient {}?)", a.cycle_time_s, a.gradient_length_s)); }

    // Gradient window from the schedule ends; trim the edges (same as the Thermo path).
    let (t0, t1) = (sched.first().unwrap().1, sched.last().unwrap().1);
    let trim = (t1 - t0) * a.gradient_trim;
    let (g0, g1) = (t0 + trim, t1 - trim);
    let gspan = (g1 - g0).max(1e-9);
    let n_ms1 = sched.iter().filter(|r| r.2 == 1).count();
    let n_ms2 = sched.len() - n_ms1;
    eprintln!(
        "SWATH schedule: {} scans ({} MS1 + {} MS2), {} windows [{:.1}, {:.1}] Th, gradient {:.1}..{:.1}s",
        sched.len(), n_ms1, n_ms2, win_pairs.len(), cover_lo, cover_hi, g0, g1);

    // Build precursors: rt_index -> apex_rt on the analytical gradient; abundance = amount·ionz·mff·frac.
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
    if precs.is_empty() {
        return Err(anyhow!("no eligible precursors — feature space empty after spectrum filtering (sample {:?})", a.sample));
    }
    let outside = precs.iter().filter(|p| p.mz < cover_lo || p.mz > cover_hi).count();
    if outside as f64 / precs.len() as f64 > 0.5 {
        eprintln!("  WARNING: {:.0}% of precursors fall outside the SWATH coverage [{:.1}, {:.1}] Th — most won't be fragmented",
            outside as f64 / precs.len() as f64 * 100.0, cover_lo, cover_hi);
    }
    eprintln!("precursors: {} eligible ({} outside window coverage)", precs.len(), outside);

    // Active-set sweep over scans (schedule RT monotonic). A precursor is active in [apex ± nσ·σ].
    let half = a.n_sigma * a.sigma_seconds;
    let mut order: Vec<usize> = (0..precs.len()).collect();
    order.sort_by(|&x, &y| precs[x].apex_rt.total_cmp(&precs[y].apex_rt));
    let two_sig2 = 2.0 * a.sigma_seconds * a.sigma_seconds;
    let floor = a.min_peak_intensity as f32;

    let (mut cursor, mut ms1_n, mut ms2_n) = (0usize, 0u64, 0u64);
    let mut active: Vec<usize> = Vec::new();
    let mut with_ms2: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut scans: Vec<MzMlScan> = Vec::with_capacity(sched.len());
    for &(_scan, t, ms_level, center, width, ce) in &sched {
        while cursor < order.len() && precs[order[cursor]].apex_rt - half <= t { active.push(order[cursor]); cursor += 1; }
        active.retain(|&i| precs[i].apex_rt + half >= t);

        let mut peaks: Vec<(f64, f32)> = Vec::new();
        let mut isolation = None;
        if ms_level == 1 {
            for &i in &active {
                let p = &precs[i];
                let w = (-((t - p.apex_rt).powi(2)) / two_sig2).exp();
                if w <= 1e-6 { continue; }
                let base = p.abundance * w * a.intensity_scale;
                for &(m, iv) in &p.ms1 {
                    let v = (base * iv as f64) as f32;
                    if v >= floor { peaks.push((m, v)); }
                }
            }
        } else if let (Some(c), Some(wd)) = (center, width) {
            let wt = WindowTransmission::new(c, wd, a.transmission_k);
            for &i in &active {
                let p = &precs[i];
                let tprob = wt.probabilities(&[p.mz])[0];
                if tprob <= 1e-3 { continue; }
                let ew = (-((t - p.apex_rt).powi(2)) / two_sig2).exp();
                if ew <= 1e-6 { continue; }
                let base = p.abundance * ew * tprob * a.intensity_scale;
                for &(m, iv) in &p.ms2 {
                    let v = (base * iv as f64) as f32;
                    if v >= floor { peaks.push((m, v)); with_ms2.insert(p.precursor_id); }
                }
            }
            isolation = Some(MzMlIsolation {
                center_mz: c, lower_mz: c - wd / 2.0, upper_mz: c + wd / 2.0,
                collision_energy: ce.unwrap_or(a.collision_energy),
            });
        }
        // mzML has no per-spectrum peak ceiling (unlike the .raw u16 cap) — keep every peak.
        peaks.sort_unstable_by(|x, y| x.0.total_cmp(&y.0)); // m/z ascending (mzML convention)
        if ms_level == 1 { ms1_n += peaks.len() as u64; } else { ms2_n += peaks.len() as u64; }
        scans.push(MzMlScan {
            ms_level,
            retention_time_s: t,
            mz: peaks.iter().map(|&(m, _)| m).collect(),
            intensity: peaks.iter().map(|&(_, v)| v).collect(),
            isolation,
        });
    }

    if let Some(parent) = a.out.parent() { std::fs::create_dir_all(parent).ok(); }
    let n = write_scans_mzml(&scans, &a.out)?;
    eprintln!("wrote SWATH mzML ({n} scans, {ms1_n} MS1 + {ms2_n} MS2 peaks) -> {}", a.out.display());

    // Answer key: per-precursor DIA truth — the SAME 8-column schema as render_thermo / run_dia, so the
    // instrument-agnostic timsim_eval scorer consumes it unchanged.
    if let Some(truth) = &a.truth {
        use arrow::array::{BooleanArray, Float64Array as F64, Int64Array, UInt64Array as U64};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        // Distinct isolation windows (center, width) → transmission profiles for the in-window flag.
        let mut wpairs = win_pairs.clone();
        wpairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
        wpairs.dedup_by(|x, y| (x.0 - y.0).abs() < 1e-6 && (x.1 - y.1).abs() < 1e-6);
        let wts: Vec<WindowTransmission> = wpairs.iter().map(|&(c, w)| WindowTransmission::new(c, w, a.transmission_k)).collect();
        let (mut pc, mut pe, mut ch, mut mo, mut rtc, mut ab, mut hm, mut iw):
            (Vec<u64>, Vec<u64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>, Vec<bool>) = Default::default();
        for p in &precs {
            pc.push(p.precursor_id); pe.push(p.peptide_id); ch.push(p.charge);
            mo.push(p.mz); rtc.push(p.apex_rt); ab.push(p.abundance);
            hm.push(with_ms2.contains(&p.precursor_id));
            iw.push(wts.iter().any(|wt| wt.probabilities(&[p.mz])[0] > 0.5));
        }
        let nrows = pc.len();
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
        eprintln!("  answer key ({nrows} precursors, frag_model={:?}) -> {}", a.frag_model, truth.display());
    }
    Ok(())
}
