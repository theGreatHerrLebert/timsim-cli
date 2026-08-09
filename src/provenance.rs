//! **Which elution kernel produced this artifact** — stamped into the artifact itself.
//!
//! # Why this module exists
//!
//! `--peak-shape` changed the render's output while leaving the command string untouched, so
//! necroflow's command-hash fingerprint could not see it. That is a cache bug, and it is fixable in
//! the flow. But it exposed a *deeper* one, which is not:
//!
//! > `truth.parquet` was **byte-identical** between a Gaussian render and an EMG render
//! > (sha256 `f0857aa7…` either way), and the `.d` recorded nothing about the shape either.
//! > **A stale artifact could not be identified as stale from its own contents.**
//!
//! No fingerprint scheme fixes that. A `.d` that is copied, re-pathed, restored from backup, or
//! handed to a collaborator arrives with no fingerprint attached — and, before this module, with no
//! way to answer "which kernel made you?" short of re-running the render and diffing. Documentation
//! and manual invalidation are not reproducibility mechanisms; a self-describing artifact is.
//!
//! # What is recorded, and why these three
//!
//! | key | why |
//! | --- | --- |
//! | `peak_shape` | `gaussian` or `emg` — the kernel |
//! | `emg_k` | the tailing factor; `0` for the Gaussian, which is the exact truth (the Gaussian *is* the `k = 0` member of the family), not a filler |
//! | `n_sigma` | the truncation radius, which the EMG's derived constants (mode offset, tail reach, peak ordinate) all depend on |
//!
//! Those three are exactly the inputs to [`PeakShape::emg`], which is what makes the round trip
//! *total*: [`parse_shape`] reconstructs a [`PeakShape`] that compares `==` to the one the render
//! used, derived constants and all. Recording only the shape NAME would be a label; recording the
//! constructor arguments is a reproduction.
//!
//! # Where it lands
//!
//! * **Bruker `.d`** — rows in `analysis.tdf`'s `GlobalMetadata` (`SimPeakShape`, `SimEmgK`,
//!   `SimNSigma`). That table is the format's own key/value provenance store, it survives every
//!   vendor reader, and `.d` copies carry it.
//! * **Every answer key parquet** (Bruker DIA `--truth`, Bruker DDA `--dda-truth`, Thermo
//!   `--thermo-truth`, SCIEX `--truth`) — Arrow schema metadata, readable through the
//!   `timsim_schema::metadata` the pipeline already uses for `peptide_rt`.
//!
//! The parquet stamp is the one that covers all four writers: the Thermo `.raw` and SCIEX mzML
//! writers are upstream crates with no provenance seam, so their answer key is where the shape can
//! be recorded without forking them. Since the answer key is the artifact every downstream scorer
//! reads, a scored result can always be traced back to the kernel that produced it.
//!
//! # Byte-identity note
//!
//! Stamping deliberately changes `analysis.tdf` and the truth parquet **footers**. It cannot change
//! `analysis.tdf_bin`, the parquet data pages, the `.raw`, or the mzML: the rendered signal is
//! untouched. `tests/golden/` pins exactly that split — signal byte-identical to the pre-EMG
//! binary, metadata differing by precisely these keys and nothing else.

use crate::render::{PeakShape, PeakShapeError};
use std::collections::HashMap;

/// Parquet / Arrow schema metadata key: the kernel name.
pub const KEY_PEAK_SHAPE: &str = "peak_shape";
/// Parquet / Arrow schema metadata key: the EMG tailing factor (`0` for the Gaussian).
pub const KEY_EMG_K: &str = "emg_k";
/// Parquet / Arrow schema metadata key: the truncation radius the shape was built against.
pub const KEY_N_SIGMA: &str = "n_sigma";

/// `GlobalMetadata` key in `analysis.tdf`: the kernel name.
pub const TDF_KEY_PEAK_SHAPE: &str = "SimPeakShape";
/// `GlobalMetadata` key in `analysis.tdf`: the EMG tailing factor.
pub const TDF_KEY_EMG_K: &str = "SimEmgK";
/// `GlobalMetadata` key in `analysis.tdf`: the truncation radius.
pub const TDF_KEY_N_SIGMA: &str = "SimNSigma";

/// Format an `f64` so `str::parse` returns the **same bits**.
///
/// `{}` is lossy for a value like `V1_DEFAULT_EMG_K = 10/21`; `{:?}` is Rust's shortest
/// round-trippable form, which is the whole requirement here — a recorded `k` that reconstructs a
/// *slightly different* kernel would be worse than recording nothing, because it would read as
/// proof.
fn f64_str(v: f64) -> String {
    format!("{v:?}")
}

/// The three `(key, value)` pairs describing a resolved shape, in Arrow/parquet spelling.
pub fn shape_metadata(shape: &PeakShape, n_sigma: f64) -> Vec<(String, String)> {
    vec![
        (KEY_PEAK_SHAPE.to_string(), shape.name().to_string()),
        (KEY_EMG_K.to_string(), f64_str(shape.emg_k())),
        (KEY_N_SIGMA.to_string(), f64_str(n_sigma)),
    ]
}

/// The same three pairs in Bruker `GlobalMetadata` spelling.
pub fn tdf_shape_metadata(shape: &PeakShape, n_sigma: f64) -> Vec<(&'static str, String)> {
    vec![
        (TDF_KEY_PEAK_SHAPE, shape.name().to_string()),
        (TDF_KEY_EMG_K, f64_str(shape.emg_k())),
        (TDF_KEY_N_SIGMA, f64_str(n_sigma)),
    ]
}

/// Merge [`shape_metadata`] into an Arrow schema's metadata map, ready for `Schema::new_with_metadata`.
pub fn schema_metadata(shape: &PeakShape, n_sigma: f64) -> HashMap<String, String> {
    shape_metadata(shape, n_sigma).into_iter().collect()
}

/// Why a recorded shape could not be read back.
#[derive(Clone, Debug, PartialEq)]
pub enum ProvenanceError {
    /// The artifact carries no shape stamp — it predates this record, or was not written by us.
    Missing(&'static str),
    /// The stamp is present but unreadable. Never silently defaulted: a corrupt provenance record
    /// is a stronger reason to stop than a missing one.
    Malformed { key: &'static str, value: String },
    /// The stamp parsed but does not describe a shape this build can construct.
    Invalid(PeakShapeError),
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProvenanceError::Missing(k) => write!(f, "no peak-shape provenance: {k} absent"),
            ProvenanceError::Malformed { key, value } => write!(f, "peak-shape provenance {key}={value:?} is malformed"),
            ProvenanceError::Invalid(e) => write!(f, "recorded peak shape is not constructible: {e}"),
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// Rebuild the [`PeakShape`] a stamp describes.
///
/// This is the round trip the record exists for: `parse_shape(shape_metadata(s, n))` must equal `s`
/// for every constructible `s`, which is what `provenance_round_trips` asserts. `gaussian` ignores
/// `k` (there is nothing else it could be); `emg` goes back through [`PeakShape::emg`], so the
/// reconstruction runs the same mode search and survival-function inversion the render ran.
/// `n-sigma` must be a real, non-negative number on EVERY branch of [`parse_shape`], including the
/// Gaussian one where it is not otherwise consumed.
fn require_finite_nonnegative_n_sigma(n_sigma: f64) -> Result<(), ProvenanceError> {
    if !n_sigma.is_finite() {
        return Err(ProvenanceError::Invalid(PeakShapeError::NotFinite {
            name: "n-sigma",
            value: n_sigma,
        }));
    }
    if n_sigma < 0.0 {
        return Err(ProvenanceError::Invalid(PeakShapeError::Negative {
            name: "n-sigma",
            value: n_sigma,
        }));
    }
    Ok(())
}

pub fn parse_shape(name: &str, k: &str, n_sigma: &str) -> Result<PeakShape, ProvenanceError> {
    let n_sigma: f64 = n_sigma
        .trim()
        .parse()
        .map_err(|_| ProvenanceError::Malformed { key: KEY_N_SIGMA, value: n_sigma.to_string() })?;
    match name.trim() {
        "gaussian" => {
            // `n_sigma` is part of the truncation record even for a Gaussian, so it must be a usable
            // number here too. Parsing it and then ignoring it let `("gaussian", <junk>, "nan")` read
            // back clean, which defeats the point of stamping: a malformed provenance record would
            // certify itself as valid.
            require_finite_nonnegative_n_sigma(n_sigma)?;
            Ok(PeakShape::Gaussian)
        }
        "emg" => {
            let k: f64 = k
                .trim()
                .parse()
                .map_err(|_| ProvenanceError::Malformed { key: KEY_EMG_K, value: k.to_string() })?;
            PeakShape::emg(k, n_sigma).map_err(ProvenanceError::Invalid)
        }
        _ => Err(ProvenanceError::Malformed { key: KEY_PEAK_SHAPE, value: name.to_string() }),
    }
}

/// Read the shape stamp back out of a map of recorded key/values (either spelling).
fn shape_from_map(
    md: &HashMap<String, String>,
    (kshape, kk, kn): (&'static str, &'static str, &'static str),
) -> Result<PeakShape, ProvenanceError> {
    let name = md.get(kshape).ok_or(ProvenanceError::Missing(kshape))?;
    let k = md.get(kk).ok_or(ProvenanceError::Missing(kk))?;
    let n = md.get(kn).ok_or(ProvenanceError::Missing(kn))?;
    parse_shape(name, k, n)
}

/// Read the shape a **parquet answer key** was rendered with.
pub fn read_parquet_shape(path: impl AsRef<std::path::Path>) -> Result<PeakShape, Box<dyn std::error::Error>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(path.as_ref())?;
    let md = ParquetRecordBatchReaderBuilder::try_new(file)?.schema().metadata().clone();
    Ok(shape_from_map(&md, (KEY_PEAK_SHAPE, KEY_EMG_K, KEY_N_SIGMA))?)
}

/// Write the shape stamp into a finished `.d`'s `GlobalMetadata`.
///
/// Called AFTER `TdfWriter::finalize`, on purpose: the writer owns that table until it closes it
/// (it rewrites `ClosedProperly` last), and ms-io exposes no seam for extra rows. `INSERT OR
/// REPLACE` so re-stamping is idempotent.
#[cfg(feature = "tdf")]
pub fn stamp_tdf(d: impl AsRef<std::path::Path>, shape: &PeakShape, n_sigma: f64) -> Result<(), Box<dyn std::error::Error>> {
    let tdf = d.as_ref().join("analysis.tdf");
    let conn = rusqlite::Connection::open(&tdf)?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare("INSERT OR REPLACE INTO GlobalMetadata (Key, Value) VALUES (?1, ?2)")?;
        for (k, v) in tdf_shape_metadata(shape, n_sigma) {
            stmt.execute(rusqlite::params![k, v])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Read the shape a **Bruker `.d`** was rendered with.
#[cfg(feature = "tdf")]
pub fn read_tdf_shape(d: impl AsRef<std::path::Path>) -> Result<PeakShape, Box<dyn std::error::Error>> {
    let tdf = d.as_ref().join("analysis.tdf");
    let conn = rusqlite::Connection::open(&tdf)?;
    let mut stmt = conn.prepare("SELECT Key, Value FROM GlobalMetadata")?;
    let mut md: HashMap<String, String> = HashMap::new();
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (k, v) = row?;
        md.insert(k, v);
    }
    Ok(shape_from_map(&md, (TDF_KEY_PEAK_SHAPE, TDF_KEY_EMG_K, TDF_KEY_N_SIGMA))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::V1_DEFAULT_EMG_K;

    /// Every shape the CLI can resolve must survive record → parse as the SAME shape, derived
    /// constants included. `PeakShape: PartialEq` compares `Emg`'s mode offset, tail reach and peak
    /// ordinate too, so this is equality of the kernel, not of its label.
    #[test]
    fn provenance_round_trips() {
        let mut shapes = vec![PeakShape::Gaussian];
        for &n_sigma in &[0.0, 1.0, 3.0, 7.5] {
            for &k in &[1e-9, 1e-3, 0.25, V1_DEFAULT_EMG_K, 1.0, 9.5, 1e3] {
                shapes.push(PeakShape::emg(k, n_sigma).unwrap());
            }
            for s in &shapes {
                let md = schema_metadata(s, n_sigma);
                let back = parse_shape(&md[KEY_PEAK_SHAPE], &md[KEY_EMG_K], &md[KEY_N_SIGMA]).unwrap();
                assert_eq!(&back, s, "round trip lost the shape at n_sigma={n_sigma}: {md:?}");
            }
            shapes.truncate(1);
        }
    }

    /// The Gaussian records `k = 0` — and `k = 0` parses back to the Gaussian. That closes the loop
    /// on the `k == 0` semantics: the same number means the same shape on both sides of the record.
    #[test]
    fn gaussian_records_k_zero_and_zero_parses_gaussian() {
        let md = schema_metadata(&PeakShape::Gaussian, 3.0);
        assert_eq!(md[KEY_PEAK_SHAPE], "gaussian");
        assert_eq!(md[KEY_EMG_K], "0.0");
        assert_eq!(parse_shape("emg", "0", "3").unwrap(), PeakShape::Gaussian);
    }

    /// A shape stamp that cannot be trusted must not be silently replaced by a default.
    #[test]
    fn malformed_or_missing_provenance_is_an_error() {
        assert!(matches!(parse_shape("lorentzian", "1", "3"), Err(ProvenanceError::Malformed { .. })));
        assert!(matches!(parse_shape("emg", "not-a-number", "3"), Err(ProvenanceError::Malformed { .. })));
        assert!(matches!(parse_shape("emg", "1", "nan"), Err(ProvenanceError::Invalid(_))));
        assert!(matches!(parse_shape("emg", "-1", "3"), Err(ProvenanceError::Invalid(_))));

        // The GAUSSIAN branch must validate `n-sigma` too. It ignores `k` (there is nothing else a
        // Gaussian could do with it), but `n-sigma` is part of the truncation record on both
        // branches — accepting a NaN there let a malformed stamp read back as a valid one.
        assert!(matches!(parse_shape("gaussian", "0", "nan"), Err(ProvenanceError::Invalid(_))));
        assert!(matches!(parse_shape("gaussian", "0", "inf"), Err(ProvenanceError::Invalid(_))));
        assert!(matches!(parse_shape("gaussian", "0", "-1"), Err(ProvenanceError::Invalid(_))));
        // ... while a well-formed Gaussian stamp still round-trips.
        assert_eq!(parse_shape("gaussian", "0", "3").unwrap(), PeakShape::Gaussian);

        let empty: HashMap<String, String> = HashMap::new();
        assert!(matches!(
            shape_from_map(&empty, (KEY_PEAK_SHAPE, KEY_EMG_K, KEY_N_SIGMA)),
            Err(ProvenanceError::Missing(KEY_PEAK_SHAPE))
        ));
    }

    /// `{:?}` must be exact for the values actually recorded — a `k` that reads back one ulp off
    /// would reconstruct a different kernel while looking like proof that it did not.
    #[test]
    fn recorded_k_is_bit_exact() {
        for &k in &[V1_DEFAULT_EMG_K, 1.0 / 3.0, 9.5, 1e-9, f64::MIN_POSITIVE, 1.7976931348623157e308] {
            assert_eq!(f64_str(k).parse::<f64>().unwrap().to_bits(), k.to_bits(), "k={k}");
        }
    }
}
