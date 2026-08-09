//! `timsim-frag-ce` — per-precursor collision energy. MEASUREMENT.
//!
//! The timsTOF does not fragment every precursor at the run's nominal collision energy, and the two
//! acquisition modes miss it in different ways — so this tool has two modes, and you must pick the one
//! that matches the run you are simulating.
//!
//! **dda-PASEF (default).** CE is a linear ramp in the mobility scan: an ion is fragmented at whatever
//! the ramp is at when it drifts out of the tunnel.
//!
//! ```text
//!   CCS --Mason-Schamp--> 1/K0 --run mobility calibration--> scan --activation policy--> CE (eV)
//! ```
//!
//! **dia-PASEF (`--dia`).** There is no ramp. The method ships a `DiaFrameMsMsWindows` table in which
//! every `(window group, scan range)` carries its OWN collision energy, and the ion is fragmented at
//! the CE of whichever window isolates it:
//!
//! ```text
//!   CCS --Mason-Schamp--> 1/K0 --run mobility calibration--> scan
//!        + precursor m/z --reference .d window table--> CE (eV)
//! ```
//!
//! This is what v1 does (`handle.get_collision_energy_dia` -> `TimsTofCollisionEnergyDIA`, read per
//! `(frame, scan)` in `ion_map_fn_dia`), and it matters: on the benchmark reference the window table
//! spans 20.00-58.12 eV, while the flat 25.0 eV the flow passes today and the dda ramp (24.94-43.28 eV
//! on the same precursors) are both wrong for a DIA run.
//!
//! Every arrow above is the component that already owns it — see [`timsim_cli::mobility_ce`] and
//! [`timsim_cli::dia_ce`]; nothing is re-derived from a formula, and the DIA CE lookup is literally
//! v1's own mscore struct.
//!
//! The output is an OPTIONAL input to `timsim-fragments` (`--collision-energies`). Without it that
//! tool keeps using its single `--collision-energy` for every precursor, byte-for-byte as before —
//! this node adds a capability, it does not change a default. `--dia` is likewise opt-in: without it
//! this tool emits exactly the dda ramp it always did.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float64Array, UInt64Array, UInt8Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use timsim_cli::dia_ce::DiaCollisionEnergy;
use timsim_cli::mobility_ce::{
    collision_energy_at, pasef_policy, MobilityGeometry, CE_BIAS, CE_SLOPE,
};
use timsim_schema::tables::precursor_ccs as CCS;
use timsim_schema::tables::precursors as PRE;

#[derive(Parser)]
#[command(
    name = "timsim-frag-ce",
    about = "precursors + CCS + acquisition geometry -> per-precursor collision energy"
)]
struct Args {
    /// `precursors` artifact — supplies `mz` and `charge` for Mason-Schamp.
    #[arg(long)]
    precursors: PathBuf,
    /// `precursor_ccs` artifact — the instrument-independent structure axis.
    #[arg(long)]
    precursor_ccs: PathBuf,
    #[arg(long)]
    out: PathBuf,

    /// Reference `.d` to take the acquisition geometry from (num_scans, 1/K0 range) AND the Bruker
    /// ModelType-2 mobility calibration. Strongly preferred: this is what makes the scan here the
    /// same scan `timsim-render --reference-d` places the ion in.
    #[arg(long)]
    reference_d: Option<PathBuf>,

    /// Reference-free geometry (used only when `--reference-d` is absent): mobility ramp length.
    #[arg(long, default_value_t = 918)]
    n_scans: u32,
    #[arg(long, default_value_t = 0.6)]
    im_min: f64,
    #[arg(long, default_value_t = 1.6)]
    im_max: f64,

    /// dda-PASEF activation ramp, `CE = ce_bias + ce_slope * scan` (eV). Defaults are v1's
    /// `dda_selection_scheme` values; change them to model a different method. Ignored under `--dia`.
    #[arg(long, default_value_t = CE_BIAS)]
    ce_bias: f64,
    #[arg(long, default_value_t = CE_SLOPE)]
    ce_slope: f64,

    /// dia-PASEF mode: read the CE from the reference `.d`'s `DiaFrameMsMsWindows` table (the window
    /// that isolates the precursor) instead of applying the dda ramp. Requires `--reference-d`, which
    /// must be a DIA acquisition, and `--dia-fallback-ce`. This is v1's DIA behaviour.
    #[arg(long)]
    dia: bool,

    /// `--dia` only: the CE to write for precursors NO window isolates. In dia-PASEF the quadrupole
    /// samples a diagonal, so roughly half the `(m/z, mobility)` plane is never isolated; those
    /// precursors deposit no MS2 peaks in the render, so their fragment intensities are never used —
    /// but the table must stay hole-free (`timsim-fragments` refuses one with holes). Pass the run's
    /// nominal `--collision-energy` so the fallback is the value the flat path would have used anyway.
    /// Required under `--dia`: there is no defensible default, and guessing one would silently put a
    /// fabricated CE on half the precursors.
    #[arg(long)]
    dia_fallback_ce: Option<f64>,

    /// `--dia` only: the ion's mobility spread, in scans, and how many sigma it deposits over. MUST
    /// match `timsim-render`'s `--sigma-scans` / `--n-sigma`, or the CE would be read off a mobility
    /// support the render does not actually use. Defaults are the renderer's.
    #[arg(long, default_value_t = 4.0)]
    sigma_scans: f64,
    #[arg(long, default_value_t = 3.0)]
    n_sigma: f64,

    /// Row-group size.
    #[arg(long, default_value_t = 2_000_000)]
    chunk: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();

    let geometry = match &a.reference_d {
        Some(d) => {
            let p = d.to_str().ok_or_else(|| anyhow!("--reference-d is not valid UTF-8"))?;
            let g = MobilityGeometry::from_reference_d(p)?;
            eprintln!(
                "  reference .d: {p}  (num_scans {}, 1/K0 {:.3}-{:.3}, Bruker ModelType-2 calibration)",
                g.n_scans, g.im_min, g.im_max
            );
            g
        }
        None => {
            let g = MobilityGeometry::linear(a.n_scans, a.im_min, a.im_max)?;
            eprintln!(
                "  no --reference-d: reference-free linear geometry (num_scans {}, 1/K0 {:.3}-{:.3})",
                g.n_scans, g.im_min, g.im_max
            );
            g
        }
    };
    let policy = pasef_policy(a.ce_bias, a.ce_slope);

    // dia-PASEF: the reference's own window table replaces the ramp. Built here so a bad reference or a
    // missing fallback fails before a single precursor is read.
    let dia = if a.dia {
        let d = a.reference_d.as_ref().ok_or_else(|| {
            anyhow!(
                "--dia needs --reference-d: the dia-PASEF collision energies live in that .d's \
                 DiaFrameMsMsWindows table and cannot be derived without it"
            )
        })?;
        if a.dia_fallback_ce.is_none() {
            return Err(anyhow!(
                "--dia needs --dia-fallback-ce: in dia-PASEF the quadrupole samples a diagonal, so \
                 some precursors are isolated by no window at all and have no collision energy. Pass \
                 the run's nominal --collision-energy for those rather than having one invented."
            ));
        }
        let p = d.to_str().ok_or_else(|| anyhow!("--reference-d is not valid UTF-8"))?;
        let ce = DiaCollisionEnergy::from_reference(p, geometry.n_scans, a.sigma_scans, a.n_sigma)?;
        let (lo, hi) = ce.table_range();
        eprintln!(
            "  activation   : dia-PASEF window table  ({} windows, CE {lo:.3}-{hi:.3} eV, \
             sigma_scans {} x {} sigma)",
            ce.n_windows(), a.sigma_scans, a.n_sigma
        );
        Some(ce)
    } else {
        eprintln!(
            "  activation   : bruker_pasef  CE = {} + {} * scan  (eV, hcd)",
            a.ce_bias, a.ce_slope
        );
        None
    };

    // precursor_id -> CCS. Held resident: one f64 per precursor (~24 MB at 1M precursors), which is
    // the same map `timsim-render` holds.
    let mut ccs: HashMap<u64, f64> = HashMap::new();
    for b in timsim_schema::read(&a.precursor_ccs, CCS::TABLE)? {
        let pcid: &UInt64Array =
            b.column_by_name(CCS::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let v: &Float64Array = b.column_by_name(CCS::CCS).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            ccs.insert(pcid.value(i), v.value(i));
        }
    }
    if ccs.is_empty() {
        return Err(anyhow!("precursor_ccs is empty — there is no mobility to derive a CE from"));
    }

    // Metadata on the ARROW schema, not on WriterProperties: that is where `timsim_schema::write`
    // stamps it, so this artifact is introspected the same way every other timsim table is
    // (`ParquetFile(...).schema_arrow.metadata`).
    //
    // The dda branch is left EXACTLY as it was, keys and all, so a default run's artifact stays
    // byte-identical; `--dia` swaps the ramp coefficients for the window-table provenance.
    let mut meta: HashMap<String, String> = [
        ("timsim.table", "precursor_collision_energy".to_string()),
        ("timsim.schema_version", "2.0".to_string()),
        ("timsim.axis", "measurement".to_string()),
        ("timsim.producer", timsim_cli::producer("timsim-frag-ce")),
        ("timsim.ce.model", "bruker_pasef".to_string()),
        ("timsim.ce.unit", "ev".to_string()),
        ("timsim.ce.bias", a.ce_bias.to_string()),
        ("timsim.ce.slope", a.ce_slope.to_string()),
        ("timsim.ce.n_scans", geometry.n_scans.to_string()),
        (
            "timsim.ce.reference_d",
            a.reference_d.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    if let Some(d) = &dia {
        let (lo, hi) = d.table_range();
        meta.remove("timsim.ce.bias");
        meta.remove("timsim.ce.slope");
        meta.insert("timsim.ce.model".into(), "dia_window_table".into());
        meta.insert("timsim.ce.windows".into(), d.n_windows().to_string());
        meta.insert("timsim.ce.table_min".into(), lo.to_string());
        meta.insert("timsim.ce.table_max".into(), hi.to_string());
        meta.insert("timsim.ce.fallback".into(), a.dia_fallback_ce.unwrap().to_string());
        meta.insert("timsim.ce.sigma_scans".into(), a.sigma_scans.to_string());
        meta.insert("timsim.ce.n_sigma".into(), a.n_sigma.to_string());
    }
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("precursor_id", DataType::UInt64, false),
            Field::new("scan", DataType::UInt32, false),
            Field::new("collision_energy", DataType::Float64, false),
        ])
        .with_metadata(meta),
    );
    let file = std::fs::File::create(&a.out)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;

    // Stream the precursors: only the current row-group is resident.
    let (mut pc, mut sc, mut ce): (Vec<u64>, Vec<u32>, Vec<f64>) = Default::default();
    let (mut written, mut missing) = (0u64, 0u64);
    let mut missing_examples: Vec<u64> = Vec::new();
    let mut all_ce: Vec<f64> = Vec::new();
    // `--dia` accounting: how many precursors a window actually isolates, how many straddle two window
    // groups (so a representative CE had to be picked), and the worst CE disagreement that cost.
    let (mut in_window, mut unisolated, mut straddling) = (0u64, 0u64, 0u64);
    let mut straddle_span = 0.0f64;

    let flush = |pc: &mut Vec<u64>,
                     sc: &mut Vec<u32>,
                     ce: &mut Vec<f64>,
                     w: &mut ArrowWriter<std::fs::File>|
     -> Result<()> {
        if pc.is_empty() {
            return Ok(());
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(std::mem::take(pc))),
                Arc::new(UInt32Array::from(std::mem::take(sc))),
                Arc::new(Float64Array::from(std::mem::take(ce))),
            ],
        )?;
        w.write(&batch)?;
        Ok(())
    };

    for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pcid: &UInt64Array =
            b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let mz: &Float64Array = b.column_by_name(PRE::MZ).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let id = pcid.value(i);
            match ccs.get(&id) {
                Some(&c) => {
                    let scan = geometry.scan_for_ccs(c, mz.value(i), chg.value(i).max(1) as u32);
                    let e = match &dia {
                        // dia-PASEF: the CE of the window that isolates this precursor.
                        Some(d) => match d.resolve(mz.value(i), scan) {
                            Some(r) => {
                                if r.distinct.len() > 1 {
                                    straddling += 1;
                                    let s = r.distinct[r.distinct.len() - 1] - r.distinct[0];
                                    straddle_span = straddle_span.max(s);
                                }
                                in_window += 1;
                                r.collision_energy
                            }
                            None => {
                                // Isolated by no window: off the quadrupole's diagonal. See
                                // --dia-fallback-ce — the render deposits no MS2 for this precursor,
                                // so this value is never actually fragmented at.
                                unisolated += 1;
                                a.dia_fallback_ce.unwrap()
                            }
                        },
                        // dda-PASEF: the mobility ramp.
                        None => collision_energy_at(&policy, scan)?,
                    };
                    pc.push(id);
                    sc.push(scan);
                    ce.push(e);
                    all_ce.push(e);
                    written += 1;
                }
                None => {
                    // A precursor with no CCS has no mobility, so it has no mobility-derived CE.
                    // Emitting a row anyway would fabricate one; skipping silently would hand
                    // `timsim-fragments` a table with holes. Count it and fail below.
                    missing += 1;
                    if missing_examples.len() < 5 {
                        missing_examples.push(id);
                    }
                }
            }
            if pc.len() >= a.chunk {
                flush(&mut pc, &mut sc, &mut ce, &mut writer)?;
            }
        }
    }
    flush(&mut pc, &mut sc, &mut ce, &mut writer)?;
    writer.close()?;

    if missing > 0 {
        return Err(anyhow!(
            "{missing} precursors have no CCS (e.g. {missing_examples:?}) — the precursors and \
             precursor_ccs artifacts are out of sync. Refusing to emit a collision-energy table \
             with holes, which would silently fall back to the flat CE for those precursors."
        ));
    }
    if written == 0 {
        return Err(anyhow!("no precursors — nothing to assign a collision energy to"));
    }

    all_ce.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (lo, hi) = (all_ce[0], all_ce[all_ce.len() - 1]);
    let med = all_ce[all_ce.len() / 2];
    eprintln!("  precursors   : {written}");
    if dia.is_some() {
        let pct = |n: u64| 100.0 * n as f64 / written as f64;
        eprintln!(
            "  isolated by a window : {in_window} ({:.2}%)   isolated by NO window: {unisolated} \
             ({:.2}%, written at the --dia-fallback-ce {})",
            pct(in_window), pct(unisolated), a.dia_fallback_ce.unwrap()
        );
        eprintln!(
            "  straddling 2 window groups: {straddling} ({:.2}% of all precursors) — dominant \
             window's CE taken, worst within-precursor CE disagreement {straddle_span:.4} eV",
            pct(straddling)
        );
    }
    eprintln!("  collision energy (eV): min {lo:.3}  median {med:.3}  max {hi:.3}  spread {:.3}", hi - lo);
    eprintln!("wrote {written} per-precursor collision energies -> {}", a.out.display());
    Ok(())
}
