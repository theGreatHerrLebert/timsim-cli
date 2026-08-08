//! `timsim-frag-ce` — per-precursor, mobility-derived collision energy. MEASUREMENT.
//!
//! The timsTOF fragments an ion at whatever collision energy the dda-PASEF ramp happens to be at
//! when that ion drifts out of the tunnel, so CE is a property of the ion's mobility, not of the
//! run. This tool turns the instrument-independent structure axis (`precursor_ccs`) plus a run's
//! acquisition geometry into the CE each precursor would actually see:
//!
//! ```text
//!   CCS --Mason-Schamp--> 1/K0 --run mobility calibration--> scan --activation policy--> CE (eV)
//! ```
//!
//! Every one of those arrows is the component that already owns it (see [`timsim_cli::mobility_ce`]);
//! nothing is re-derived from a formula. The scan is the one `timsim-render` places the ion in, so
//! the CE and the rendered mobility agree.
//!
//! The output is an OPTIONAL input to `timsim-fragments` (`--collision-energies`). Without it that
//! tool keeps using its single `--collision-energy` for every precursor, byte-for-byte as before —
//! this node adds a capability, it does not change a default.

use anyhow::{anyhow, Result};
use arrow::array::{Array, Float64Array, UInt64Array, UInt8Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

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
    /// `dda_selection_scheme` values; change them to model a different method.
    #[arg(long, default_value_t = CE_BIAS)]
    ce_bias: f64,
    #[arg(long, default_value_t = CE_SLOPE)]
    ce_slope: f64,

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
    eprintln!(
        "  activation   : bruker_pasef  CE = {} + {} * scan  (eV, hcd)",
        a.ce_bias, a.ce_slope
    );

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
    let meta: HashMap<String, String> = [
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
                    let e = collision_energy_at(&policy, scan)?;
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
    eprintln!("  collision energy (eV): min {lo:.3}  median {med:.3}  max {hi:.3}  spread {:.3}", hi - lo);
    eprintln!("wrote {written} per-precursor collision energies -> {}", a.out.display());
    Ok(())
}
