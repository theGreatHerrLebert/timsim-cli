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

/// Bumped when the MEANING of any key changes. Readers must fail closed on an unknown major.
pub const PROVENANCE_SCHEMA_VERSION: &str = "2";

/// The elution IMPLEMENTATION, distinct from the law it implements.
///
/// The zero-`k` collapse threshold, the frame rounding and the EMG's derived constants live here.
/// Any of them can move the rendered shapes while every other recorded field is unchanged — so a
/// descriptor without this cannot recompute what the render did. That was the central finding of the
/// review of `PEAK_SHAPE_PROVENANCE_V2.md`, against a first draft that called the descriptor "total".
pub const ELUTION_MODEL_VERSION: &str = "timsim-elution/2";

/// How `peptide_rt`'s draws are keyed, salts included. A different normalisation silently repoints
/// every peptide at a different draw, so the key is part of the record.
pub const IDENTITY_KEY: &str = "blake2b-64/sequence#rt_sigma|rt_k";

/// Additional keys for the per-peptide record (parquet/Arrow spelling).
pub const KEY_SCHEMA_VERSION: &str = "provenance_schema_version";
pub const KEY_MODEL_VERSION: &str = "elution_model_version";
pub const KEY_IDENTITY_KEY: &str = "identity_key";
pub const KEY_SIGMA_LAW: &str = "sigma_law";
pub const KEY_SIGMA_HAT_DIST: &str = "sigma_hat_dist";
pub const KEY_K_HAT_DIST: &str = "k_hat_dist";
pub const KEY_GRADIENT_SECONDS: &str = "gradient_seconds";
pub const KEY_CYCLE_SECONDS: &str = "cycle_seconds";
pub const KEY_SIGMA_BAND: &str = "sigma_band_seconds";
pub const KEY_K_UPPER: &str = "k_upper";
pub const KEY_REALIZED_DIGEST: &str = "shape_population_digest";
pub const KEY_N_SHAPED: &str = "n_peptides_shaped";
pub const KEY_N_COLLAPSED: &str = "n_gaussian_collapsed";
pub const KEY_SIGMA_STATS: &str = "sigma_frames_min_mean_max";
pub const KEY_K_STATS: &str = "emg_k_min_mean_max";

/// What actually came out of a per-peptide run.
///
/// The aggregates are a human-readable diagnostic. **The digest is the commitment**: six aggregates
/// can be satisfied by a writer that computed them over the wrong population; a digest over the
/// realized `(peptide, sigma, k)` set cannot.
#[derive(Clone, Debug, PartialEq)]
pub struct Realized {
    pub n_shaped: usize,
    pub n_gaussian_collapsed: usize,
    pub sigma_frames_min: f64,
    pub sigma_frames_mean: f64,
    pub sigma_frames_max: f64,
    pub emg_k_min: f64,
    pub emg_k_mean: f64,
    pub emg_k_max: f64,
    /// `sha256` over the realized set in peptide-id order — see [`RealizedBuilder`].
    pub digest: String,
}

/// Accumulates the realized set and commits to it.
#[derive(Default)]
pub struct RealizedBuilder {
    rows: Vec<(u64, f64, f64)>,
    n_gaussian_collapsed: usize,
}

impl RealizedBuilder {
    pub fn push(&mut self, peptide_id: u64, sigma_frames: f64, shape: &PeakShape) {
        if matches!(shape, PeakShape::Gaussian) {
            self.n_gaussian_collapsed += 1;
        }
        self.rows.push((peptide_id, sigma_frames, shape.emg_k()));
    }

    /// Sorting by `peptide_id` rather than hashing in insertion order is load-bearing: the shapes
    /// are accumulated from a `HashMap` whose iteration order is not stable between runs, so an
    /// insertion-order digest would differ between two byte-identical renders — worse than useless,
    /// because it would read as a detected difference.
    pub fn finish(mut self) -> Realized {
        use sha2::{Digest, Sha256};
        self.rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.total_cmp(&b.2)));
        let mut h = Sha256::new();
        // Fixed-width big-endian BIT PATTERNS, not text: a formatted float is a precision-dependent
        // encoding, and this digest has to be reproducible across producers.
        h.update(PROVENANCE_SCHEMA_VERSION.as_bytes());
        h.update([0u8]);
        h.update(ELUTION_MODEL_VERSION.as_bytes());
        h.update([0u8]);
        for (pid, sigma, k) in &self.rows {
            h.update(pid.to_be_bytes());
            h.update(sigma.to_bits().to_be_bytes());
            h.update(k.to_bits().to_be_bytes());
        }
        let digest = format!("{:x}", h.finalize());

        let n = self.rows.len();
        let (mut smin, mut smax, mut ssum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
        let (mut kmin, mut kmax, mut ksum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
        for (_, s, k) in &self.rows {
            smin = smin.min(*s); smax = smax.max(*s); ssum += *s;
            kmin = kmin.min(*k); kmax = kmax.max(*k); ksum += *k;
        }
        let d = n.max(1) as f64;
        Realized {
            n_shaped: n,
            n_gaussian_collapsed: self.n_gaussian_collapsed,
            sigma_frames_min: if n == 0 { 0.0 } else { smin },
            sigma_frames_mean: ssum / d,
            sigma_frames_max: if n == 0 { 0.0 } else { smax },
            emg_k_min: if n == 0 { 0.0 } else { kmin },
            emg_k_mean: ksum / d,
            emg_k_max: if n == 0 { 0.0 } else { kmax },
            digest,
        }
    }
}

/// How a run's elution shapes were produced — the ONE typed record every writer stamps.
///
/// Single type, single serialisation ([`Self::pairs`]), consumed by the Bruker `GlobalMetadata`
/// writer and all four answer-key writers. None of them may recompute any of it independently, or
/// four writers can disagree about one run.
#[derive(Clone, Debug, PartialEq)]
pub enum ElutionProvenance {
    /// One shape for the whole run. Keeps the historical three keys and their exact round trip.
    Global { shape: PeakShape, n_sigma: f64 },
    /// v1's model: width and tail drawn per peptide.
    ///
    /// Deliberately carries NO single `emg_k`. Writing the population mean into that key would be
    /// worse than omitting it: a reader cannot tell a mean from a value, and would reconstruct a
    /// kernel that no peak in the run actually had.
    PerPeptide {
        n_sigma: f64,
        gradient_seconds: f64,
        cycle_seconds: f64,
        sigma_band_seconds: (f64, f64),
        k_upper: f64,
        realized: Realized,
    },
}

impl ElutionProvenance {
    /// The value written to `peak_shape` / `SimPeakShape`.
    pub fn name(&self) -> &'static str {
        match self {
            ElutionProvenance::Global { shape, .. } => shape.name(),
            ElutionProvenance::PerPeptide { .. } => "per-peptide",
        }
    }

    /// THE canonical serialisation. Both spellings derive from this, so they cannot drift.
    pub fn pairs(&self) -> Vec<(String, String)> {
        let mut v = vec![(KEY_PEAK_SHAPE.to_string(), self.name().to_string())];
        match self {
            ElutionProvenance::Global { shape, n_sigma } => {
                // Unchanged from schema v1, so an old reader still round-trips a global run exactly.
                v.push((KEY_EMG_K.to_string(), f64_str(shape.emg_k())));
                v.push((KEY_N_SIGMA.to_string(), f64_str(*n_sigma)));
            }
            ElutionProvenance::PerPeptide {
                n_sigma, gradient_seconds, cycle_seconds, sigma_band_seconds, k_upper, realized,
            } => {
                v.push((KEY_N_SIGMA.to_string(), f64_str(*n_sigma)));
                v.push((KEY_SCHEMA_VERSION.to_string(), PROVENANCE_SCHEMA_VERSION.to_string()));
                v.push((KEY_MODEL_VERSION.to_string(), ELUTION_MODEL_VERSION.to_string()));
                v.push((KEY_IDENTITY_KEY.to_string(), IDENTITY_KEY.to_string()));
                v.push((KEY_SIGMA_LAW.to_string(), "gradient-affine/v1".to_string()));
                v.push((KEY_SIGMA_HAT_DIST.to_string(), "beta(4,4)".to_string()));
                v.push((KEY_K_HAT_DIST.to_string(), "beta(1,20)".to_string()));
                v.push((KEY_GRADIENT_SECONDS.to_string(), f64_str(*gradient_seconds)));
                v.push((KEY_CYCLE_SECONDS.to_string(), f64_str(*cycle_seconds)));
                v.push((KEY_SIGMA_BAND.to_string(), format!(
                    "[{},{}]", f64_str(sigma_band_seconds.0), f64_str(sigma_band_seconds.1))));
                v.push((KEY_K_UPPER.to_string(), f64_str(*k_upper)));
                v.push((KEY_REALIZED_DIGEST.to_string(), realized.digest.clone()));
                v.push((KEY_N_SHAPED.to_string(), realized.n_shaped.to_string()));
                v.push((KEY_N_COLLAPSED.to_string(), realized.n_gaussian_collapsed.to_string()));
                v.push((KEY_SIGMA_STATS.to_string(), format!("{}/{}/{}",
                    f64_str(realized.sigma_frames_min), f64_str(realized.sigma_frames_mean), f64_str(realized.sigma_frames_max))));
                v.push((KEY_K_STATS.to_string(), format!("{}/{}/{}",
                    f64_str(realized.emg_k_min), f64_str(realized.emg_k_mean), f64_str(realized.emg_k_max))));
            }
        }
        v
    }

    /// The same record in Bruker `GlobalMetadata` spelling (`Sim`-prefixed, CamelCase).
    pub fn tdf_pairs(&self) -> Vec<(String, String)> {
        let out: Vec<(String, String)> = self.pairs().into_iter().map(|(k, v)| (tdf_key(&k), v)).collect();
        // `tdf_key` is not injective (see its docs) and the stamp is an INSERT OR REPLACE, so a
        // collision would silently drop a field rather than fail. Cheap to check, and the failure it
        // prevents is invisible.
        let mut seen = std::collections::HashSet::with_capacity(out.len());
        for (k, _) in &out {
            assert!(seen.insert(k.clone()), "tdf provenance key collision on {k:?} — rename the offending key");
        }
        out
    }

    pub fn schema_metadata(&self) -> HashMap<String, String> {
        self.pairs().into_iter().collect()
    }
}

/// `peak_shape` -> `SimPeakShape`. ONE mapping, so the two spellings cannot drift.
///
/// NOT injective in general: it drops underscores, so `a_b`, `a__b` and `_a_b` all collapse to
/// `SimAB`. The declared keys are collision-free today, and [`ElutionProvenance::tdf_pairs`] asserts
/// that rather than trusting it — a future key added with a doubled or leading underscore would
/// otherwise silently overwrite another key's value in `GlobalMetadata`, where the write is an
/// `INSERT OR REPLACE`.
fn tdf_key(k: &str) -> String {
    let mut out = String::from("Sim");
    let mut upper = true;
    for c in k.chars() {
        if c == '_' {
            upper = true;
        } else if upper {
            out.extend(c.to_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
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
    /// The artifact was rendered with PER-PEPTIDE shapes, so "the run's shape" does not exist.
    ///
    /// Distinct from [`Self::Malformed`] on purpose: the record is well-formed and truthful, and the
    /// caller's QUESTION is what has no answer. A reader that treats this as corruption will reach
    /// for a fallback; one that reads it correctly knows to go to `peptide_rt` and the descriptor
    /// keys instead.
    NotASingleShape,
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProvenanceError::Missing(k) => write!(f, "no peak-shape provenance: {k} absent"),
            ProvenanceError::Malformed { key, value } => write!(f, "peak-shape provenance {key}={value:?} is malformed"),
            ProvenanceError::Invalid(e) => write!(f, "recorded peak shape is not constructible: {e}"),
            ProvenanceError::NotASingleShape => write!(
                f,
                "this artifact was rendered with per-peptide peak shapes, so it has no single \
                 kernel; read `{KEY_SIGMA_BAND}`/`{KEY_K_UPPER}` and the peptide_rt draws instead"
            ),
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
        // A per-peptide run HAS no single shape, so there is nothing honest to return. Failing here
        // is the point: a caller that wants "the run's kernel" is asking a question with no answer,
        // and any value handed back — the mean, the mode, the first peptide's — would be a kernel no
        // peak in the run actually had, presented as if it were the run's. That is the exact defect
        // this module exists to prevent, and it would be introduced BY the module.
        "per-peptide" => Err(ProvenanceError::NotASingleShape),
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
pub fn stamp_tdf(d: impl AsRef<std::path::Path>, prov: &ElutionProvenance) -> Result<(), Box<dyn std::error::Error>> {
    let tdf = d.as_ref().join("analysis.tdf");
    let conn = rusqlite::Connection::open(&tdf)?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare("INSERT OR REPLACE INTO GlobalMetadata (Key, Value) VALUES (?1, ?2)")?;
        for (k, v) in prov.tdf_pairs() {
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

    fn realized(rows: &[(u64, f64, f64)]) -> Realized {
        let mut b = RealizedBuilder::default();
        for &(pid, s, k) in rows {
            let shape = if k > 0.0 { PeakShape::emg(k, 3.0).unwrap() } else { PeakShape::Gaussian };
            b.push(pid, s, &shape);
        }
        b.finish()
    }

    fn per_peptide(rows: &[(u64, f64, f64)]) -> ElutionProvenance {
        ElutionProvenance::PerPeptide {
            n_sigma: 3.0,
            gradient_seconds: 1861.3,
            cycle_seconds: 0.105445749226364,
            sigma_band_seconds: (1.134375, 1.890625),
            k_upper: 10.0,
            realized: realized(rows),
        }
    }

    /// **A per-peptide run must NOT answer "what was the run's kernel?"** — it has no such thing.
    ///
    /// The tempting failure is to hand back the population mean. It parses, it constructs, it looks
    /// exactly like a valid stamp, and it describes a kernel that no peak in the run actually had.
    /// That is the same self-certifying-wrong-kernel defect this module was written to prevent, and
    /// it would be introduced BY the module — so `parse_shape` refuses, with a distinct error a
    /// reader can tell apart from corruption.
    #[test]
    fn a_per_peptide_run_has_no_single_shape_to_parse() {
        let p = per_peptide(&[(1, 12.0, 0.4), (2, 14.0, 0.6)]);
        let m: HashMap<String, String> = p.pairs().into_iter().collect();
        assert_eq!(m[KEY_PEAK_SHAPE], "per-peptide");

        // The mean k is 0.5 here — exactly the value a naive implementation would stamp. It must
        // NOT appear as `emg_k`, because a reader cannot tell a mean from a value.
        assert!(!m.contains_key(KEY_EMG_K), "per-peptide must not write a single emg_k");

        // And reading it back as "the shape" fails LOUDLY, distinctly from a malformed record: a
        // reader that sees Malformed reaches for a fallback; one that sees this knows where to look.
        let got = parse_shape("per-peptide", "0.5", "3.0");
        assert_eq!(got, Err(ProvenanceError::NotASingleShape));
        assert!(format!("{}", got.unwrap_err()).contains("per-peptide"));
    }

    /// The digest must depend on the SET, not on the order it was accumulated in.
    ///
    /// The shapes come out of a `HashMap`, whose iteration order is not stable between runs. An
    /// insertion-order digest would therefore differ between two byte-identical renders — and would
    /// read as a detected difference, which is worse than no digest at all.
    #[test]
    fn the_digest_commits_to_the_set_not_the_insertion_order() {
        let a = realized(&[(1, 12.0, 0.4), (2, 14.0, 0.6), (3, 9.0, 0.1)]);
        let b = realized(&[(3, 9.0, 0.1), (1, 12.0, 0.4), (2, 14.0, 0.6)]);
        assert_eq!(a.digest, b.digest, "digest must be insertion-order independent");

        // But it must still SEE every component: change any one and it moves.
        assert_ne!(a.digest, realized(&[(1, 12.0, 0.4), (2, 14.0, 0.6), (4, 9.0, 0.1)]).digest, "peptide id");
        assert_ne!(a.digest, realized(&[(1, 12.5, 0.4), (2, 14.0, 0.6), (3, 9.0, 0.1)]).digest, "sigma");
        assert_ne!(a.digest, realized(&[(1, 12.0, 0.5), (2, 14.0, 0.6), (3, 9.0, 0.1)]).digest, "k");
        // ... including a peptide simply being absent, which the aggregates alone can hide.
        assert_ne!(a.digest, realized(&[(1, 12.0, 0.4), (2, 14.0, 0.6)]).digest, "missing peptide");
    }

    /// The aggregates are a diagnostic; they must at least be self-consistent.
    #[test]
    fn realized_aggregates_are_ordered_and_finite() {
        let r = realized(&[(1, 12.0, 0.4), (2, 14.0, 0.6), (3, 9.0, 0.0)]);
        assert_eq!(r.n_shaped, 3);
        assert_eq!(r.n_gaussian_collapsed, 1, "k=0 is the Gaussian limit and must be counted");
        assert!(r.sigma_frames_min <= r.sigma_frames_mean && r.sigma_frames_mean <= r.sigma_frames_max);
        assert!(r.emg_k_min <= r.emg_k_mean && r.emg_k_mean <= r.emg_k_max);
        for v in [r.sigma_frames_min, r.sigma_frames_mean, r.sigma_frames_max, r.emg_k_min, r.emg_k_mean, r.emg_k_max] {
            assert!(v.is_finite(), "aggregate {v} is not finite");
        }
        // An empty run must not produce inf/NaN aggregates.
        let e = realized(&[]);
        assert_eq!(e.n_shaped, 0);
        for v in [e.sigma_frames_min, e.sigma_frames_mean, e.sigma_frames_max] {
            assert!(v.is_finite(), "empty-run aggregate {v} is not finite");
        }
    }

    /// The two spellings must stay in lockstep — one record, one serialisation.
    #[test]
    fn tdf_and_parquet_spellings_carry_the_same_record() {
        let p = per_peptide(&[(1, 12.0, 0.4)]);
        let pq = p.pairs();
        let tdf = p.tdf_pairs();
        assert_eq!(pq.len(), tdf.len(), "the two spellings must carry the same key count");
        for ((_, v_pq), (k_tdf, v_tdf)) in pq.iter().zip(tdf.iter()) {
            assert_eq!(v_pq, v_tdf, "value drift between spellings");
            assert!(k_tdf.starts_with("Sim"), "tdf key {k_tdf} must be Sim-prefixed");
        }
        assert_eq!(tdf[0].0, TDF_KEY_PEAK_SHAPE, "the historical key name must not move");

        // The global record keeps schema v1's three keys EXACTLY, so an old reader still works.
        let g = ElutionProvenance::Global { shape: PeakShape::emg(0.5, 3.0).unwrap(), n_sigma: 3.0 };
        let gp = g.pairs();
        let keys: Vec<&str> = gp.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec![KEY_PEAK_SHAPE, KEY_EMG_K, KEY_N_SIGMA]);
        let gt: Vec<String> = g.tdf_pairs().into_iter().map(|(k, _)| k).collect();
        assert_eq!(gt, vec![TDF_KEY_PEAK_SHAPE, TDF_KEY_EMG_K, TDF_KEY_N_SIGMA]);
    }

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
