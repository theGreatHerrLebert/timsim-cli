//! The TOML config surface.
//!
//! **Two vocabularies, one physics.** The chemist's parameters are the primary interface;
//! they are transfer functions over a physical parameterisation the informatician may
//! override directly. Both are recorded in the report, so neither audience has to learn the
//! other's language and neither is lied to.

use anyhow::{bail, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use timsim_chem::design;

// ─────────────────────────────────────────────────────────────────────────────
// proteome.toml
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-source, because that is how HYE actually works.
///
/// v1 recovers the organism by substring-matching `"HUMAN"` / `"YEAST"` / `"ECOLI"` in the
/// FASTA header (`table_labeling.py:12`), and peptides shared between organisms silently
/// become `"Unknown"` and get **dropped**. Here the organism is a declared column from the
/// moment the protein enters the model.
#[derive(Debug, Deserialize)]
pub struct ProteomeSpec {
    #[serde(default, rename = "source")]
    pub sources: Vec<Source>,
}

#[derive(Debug, Deserialize)]
pub struct Source {
    pub path: String,
    #[serde(default)]
    pub organism: Option<String>,
    #[serde(default)]
    pub is_contaminant: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// design.toml
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DesignFile {
    pub design: DesignHeader,
    pub abundance: BTreeMap<String, AbundanceSpec>,
    #[serde(rename = "condition")]
    pub conditions: Vec<ConditionSpec>,
    #[serde(default)]
    pub variance: VarianceSpec,
}

#[derive(Debug, Deserialize)]
pub struct DesignHeader {
    /// Condition against which `true_log2fc` is computed.
    pub reference: String,
    /// Total peptide mass on column, per run. The physically meaningful replacement for
    /// `num_sample_peptides`, which had no physical meaning at all.
    pub load_ng: f64,
    /// How many proteins are actually in the sample. Excluded proteins keep their place in the
    /// structure and receive amount 0 — which is what keeps protein-level FDR answerable. There is
    /// deliberately no peptide-count knob: peptides vanish by falling below the detection limit,
    /// not at random.
    #[serde(default)]
    pub n_proteins: Option<usize>,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_seed() -> u64 {
    42
}

#[derive(Debug, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum AbundanceSpec {
    /// σ ≈ 2 gives roughly the ~7 orders of dynamic range a real proteome spans.
    Lognormal { sigma: f64 },
    /// The rank-abundance curve v1 uses (`get_tenzer_hokey`) — a better-shaped marginal than a
    /// log-normal, and here the rank is keyed by identity rather than by draw order.
    Hockeystick {
        #[serde(default = "hs_decay")]
        decay: f64,
        #[serde(default = "hs_tail")]
        tail: f64,
    },
    /// From real data — PaxDb, or a real quantification. Two columns: id, abundance.
    /// **The only source that gets protein IDENTITY right**, not merely the shape.
    Table { path: String },
}

fn hs_decay() -> f64 {
    0.06
}
fn hs_tail() -> f64 {
    1e-4
}

#[derive(Debug, Deserialize)]
pub struct ConditionSpec {
    pub name: String,
    /// organism → mass fraction, or the string `"rest"`.
    pub mix: BTreeMap<String, toml::Value>,
    #[serde(default = "one")]
    pub replicates: u32,
    #[serde(default = "one")]
    pub technical_replicates: u32,
    /// One regulation block, or a **list** of them. Hand-parsed (§`parse_regulate`) rather than
    /// deserialised into an untagged enum: an untagged enum discards the inner error and reports
    /// only *"data did not match any variant"*, which for a 14-protein map is unusable.
    #[serde(default)]
    pub regulate: Option<toml::Value>,
}

fn one() -> u32 {
    1
}

/// Parse `[[condition]].regulate` into regulation blocks.
///
/// # The canonical form — a LIST of TAGGED blocks
///
/// ```toml
/// regulate = [
///   { kind = "explicit", proteins = { P0DJI8 = 1.6, P02741 = 1.4, P17540 = 0.7 } },
///   { kind = "generative", fraction = 0.05, log2fc_sd = 1.0 },
/// ]
/// ```
///
/// Each named protein carries **its own** log2 fold change, because a real signature does not
/// move every member by the same amount — and a benchmark that collapses them makes a volcano
/// rank the planted set by noise instead of by effect size.
///
/// A single block may be written without the surrounding list. And because a TOML **inline table
/// may not span lines**, a set of any size wants the array-of-tables spelling instead — same data
/// model, one accession per line, so each can carry a comment:
///
/// ```toml
/// [[condition.regulate]]
/// kind = "explicit"
/// [condition.regulate.proteins]
/// P0DJI8 = 1.6   # SAA1
/// P17540 = 0.7   # CKMT2
/// ```
///
/// # The deprecated scalar form
///
/// ```toml
/// regulate = { proteins = ["P0DJI8", "P02741"], log2fc = 1.0 }
/// ```
///
/// Still accepted, because published configs are written this way — but it warns, and only the
/// tagged form is documented and emitted.
fn parse_regulate(cond: &str, v: &toml::Value) -> Result<Vec<design::Regulation>> {
    let blocks: Vec<&toml::Value> = match v {
        toml::Value::Array(a) => a.iter().collect(),
        toml::Value::Table(_) => vec![v],
        other => bail!(
            "condition {cond:?}: `regulate` must be a block or a list of blocks, got {other}"
        ),
    };

    let mut out = Vec::new();
    for b in blocks {
        let t = b.as_table().ok_or_else(|| {
            anyhow::anyhow!("condition {cond:?}: each `regulate` entry must be a table, got {b}")
        })?;

        // A number that TOML happened to write as an integer is still a fold change.
        let num = |k: &str| -> Result<f64> {
            match t.get(k) {
                Some(toml::Value::Float(x)) => Ok(*x),
                Some(toml::Value::Integer(x)) => Ok(*x as f64),
                Some(other) => bail!("condition {cond:?}: `regulate.{k}` must be a number, got {other}"),
                None => bail!("condition {cond:?}: `regulate` block is missing `{k}`"),
            }
        };

        match t.get("kind").and_then(|k| k.as_str()) {
            Some("explicit") => {
                let ps = t.get("proteins").ok_or_else(|| {
                    anyhow::anyhow!("condition {cond:?}: an explicit block needs `proteins`")
                })?;
                let map = ps.as_table().ok_or_else(|| {
                    anyhow::anyhow!(
                        "condition {cond:?}: `proteins` in an explicit block is a MAP of \
                         accession -> log2fc, e.g. {{ P0DJI8 = 1.6, P17540 = 0.7 }} — a bare list \
                         has no per-protein magnitude, which is the point of this form"
                    )
                })?;
                let mut proteins = BTreeMap::new();
                for (id, fc) in map {
                    let fc = match fc {
                        toml::Value::Float(x) => *x,
                        toml::Value::Integer(x) => *x as f64,
                        other => bail!(
                            "condition {cond:?}: log2fc for {id:?} must be a number, got {other}"
                        ),
                    };
                    proteins.insert(id.clone(), fc);
                }
                if proteins.is_empty() {
                    bail!("condition {cond:?}: an explicit `regulate` block names no proteins");
                }
                out.push(design::Regulation::Explicit { proteins });
            }
            Some("generative") => out.push(design::Regulation::Generative {
                fraction: num("fraction")?,
                log2fc_sd: num("log2fc_sd")?,
            }),
            Some(other) => bail!(
                "condition {cond:?}: unknown regulate kind {other:?} — expected \"explicit\" or \
                 \"generative\""
            ),
            // ── DEPRECATED compatibility paths, kept so published configs keep working ──
            None if t.contains_key("proteins") => {
                let list = t["proteins"].as_array().ok_or_else(|| {
                    anyhow::anyhow!(
                        "condition {cond:?}: the deprecated scalar `regulate` form takes \
                         `proteins = [...]` and one `log2fc`; for per-protein magnitudes use \
                         `{{ kind = \"explicit\", proteins = {{ ACC = 1.6, ... }} }}`"
                    )
                })?;
                let log2fc = num("log2fc")?;
                eprintln!(
                    "  warning: condition {cond:?}: `regulate = {{ proteins = [...], log2fc = ... }}` \
                     is deprecated — it gives every protein the SAME magnitude. Use \
                     `regulate = [{{ kind = \"explicit\", proteins = {{ ACC = <log2fc>, ... }} }}]`"
                );
                let mut proteins = BTreeMap::new();
                for id in list {
                    let id = id.as_str().ok_or_else(|| {
                        anyhow::anyhow!("condition {cond:?}: `proteins` must be accession strings")
                    })?;
                    if proteins.insert(id.to_string(), log2fc).is_some() {
                        bail!("condition {cond:?}: protein {id:?} is listed twice in `regulate`");
                    }
                }
                if proteins.is_empty() {
                    bail!("condition {cond:?}: `regulate` names no proteins");
                }
                out.push(design::Regulation::Explicit { proteins });
            }
            None if t.contains_key("fraction") || t.contains_key("log2fc_sd") => {
                out.push(design::Regulation::Generative {
                    fraction: num("fraction")?,
                    log2fc_sd: num("log2fc_sd")?,
                });
            }
            None => bail!(
                "condition {cond:?}: a `regulate` block needs `kind = \"explicit\"` (with a \
                 `proteins` map) or `kind = \"generative\"` (with `fraction` and `log2fc_sd`)"
            ),
        }
    }
    Ok(out)
}

#[derive(Debug, Default, Deserialize)]
pub struct VarianceSpec {
    /// CV between biological replicates. Distinct material, so the amounts really differ.
    #[serde(default)]
    pub biological: f64,
    /// Spread of per-protein CVs (natural-log units). 0 ⇒ every protein shares the mean CV.
    #[serde(default)]
    pub biological_heterogeneity: f64,
    /// CV between injections. Declared here, applied on the measurement axis — a technical
    /// replicate is the *same tube*, so its amounts are identical and all the variation is
    /// in the measurement.
    #[serde(default)]
    pub technical: f64,
}

// ─────────────────────────────────────────────────────────────────────────────

pub fn load_proteome_spec(path: &Path) -> Result<ProteomeSpec> {
    let s: ProteomeSpec = toml::from_str(&std::fs::read_to_string(path)?)?;
    if s.sources.is_empty() {
        bail!("{}: no [[source]] entries", path.display());
    }
    Ok(s)
}

pub fn load_design(path: &Path, abundance_dir: &Path) -> Result<design::DesignSpec> {
    let f: DesignFile = toml::from_str(&std::fs::read_to_string(path)?)?;

    let mut abundance = BTreeMap::new();
    for (org, a) in &f.abundance {
        let profile = match a {
            AbundanceSpec::Lognormal { sigma } => {
                if !sigma.is_finite() || *sigma < 0.0 {
                    bail!("abundance.{org}: sigma must be finite and non-negative, got {sigma}");
                }
                design::AbundanceProfile::LogNormal { sigma: *sigma }
            }
            AbundanceSpec::Hockeystick { decay, tail } => {
                if !decay.is_finite() || *decay <= 0.0 || !tail.is_finite() || *tail < 0.0 {
                    bail!("abundance.{org}: hockeystick decay must be > 0 and tail >= 0");
                }
                design::AbundanceProfile::HockeyStick { decay: *decay, tail: *tail }
            }
            AbundanceSpec::Table { path: p } => {
                let full = abundance_dir.join(p);
                let text = std::fs::read_to_string(&full)
                    .map_err(|e| anyhow::anyhow!("abundance.{org}: {}: {e}", full.display()))?;
                let mut t = std::collections::HashMap::new();
                for (lineno, line) in text.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let mut it = line.split_whitespace();
                    match (it.next(), it.next()) {
                        (Some(id), Some(v)) => {
                            let v: f64 = v.parse().map_err(|_| {
                                anyhow::anyhow!("{}:{}: {v:?} is not a number", full.display(), lineno + 1)
                            })?;
                            t.insert(id.to_string(), v);
                        }
                        _ => bail!("{}:{}: expected `id abundance`", full.display(), lineno + 1),
                    }
                }
                design::AbundanceProfile::Table(t)
            }
        };
        abundance.insert(org.clone(), profile);
    }

    let mut conditions = Vec::new();
    for c in &f.conditions {
        let mut mix = BTreeMap::new();
        for (org, v) in &c.mix {
            let share = match v {
                toml::Value::Float(x) => design::Share::Fraction(*x),
                toml::Value::Integer(x) => design::Share::Fraction(*x as f64),
                toml::Value::String(s) if s.eq_ignore_ascii_case("rest") => design::Share::Rest,
                other => bail!(
                    "condition {:?}: mix.{org} must be a number or \"rest\", got {other}",
                    c.name
                ),
            };
            mix.insert(org.clone(), share);
        }
        conditions.push(design::Condition {
            name: c.name.clone(),
            mix,
            replicates: c.replicates,
            technical_replicates: c.technical_replicates,
            regulate: match &c.regulate {
                Some(v) => parse_regulate(&c.name, v)?,
                None => Vec::new(),
            },
        });
    }

    Ok(design::DesignSpec {
        reference: f.design.reference,
        load_ng: f.design.load_ng,
        complexity: design::Complexity { n_proteins: f.design.n_proteins },
        abundance,
        conditions,
        variance: design::Variance {
            biological: f.variance.biological,
            biological_heterogeneity: f.variance.biological_heterogeneity,
            technical: f.variance.technical,
        },
        seed: f.design.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(toml_src: &str) -> Result<Vec<design::Regulation>> {
        let v: toml::Value = toml::from_str(toml_src).unwrap();
        parse_regulate("severe", &v["regulate"])
    }

    fn explicit(r: &design::Regulation) -> &BTreeMap<String, f64> {
        match r {
            design::Regulation::Explicit { proteins } => proteins,
            other => panic!("expected an explicit block, got {other:?}"),
        }
    }

    /// THE canonical form: a list of tagged blocks, the explicit one carrying a per-protein map.
    /// This is what makes a benchmark's authored 0.5–1.8 spread survive into the answer key
    /// instead of collapsing to one number.
    #[test]
    fn a_list_of_tagged_blocks_keeps_per_protein_magnitudes() {
        let rs = reg(
            r#"
            regulate = [
              { kind = "explicit", proteins = { P0DJI8 = 1.6, P02741 = 1.4, P17540 = 0.7 } },
              { kind = "generative", fraction = 0.05, log2fc_sd = 1.0 },
            ]
            "#,
        )
        .unwrap();
        assert_eq!(rs.len(), 2);
        let m = explicit(&rs[0]);
        assert_eq!(m["P0DJI8"], 1.6);
        assert_eq!(m["P02741"], 1.4);
        assert_eq!(m["P17540"], 0.7);
        match &rs[1] {
            design::Regulation::Generative { fraction, log2fc_sd } => {
                assert_eq!((*fraction, *log2fc_sd), (0.05, 1.0));
            }
            other => panic!("{other:?}"),
        }
    }

    /// A single block needs no surrounding list.
    #[test]
    fn one_tagged_block_without_a_list_is_accepted() {
        let rs = reg(r#"regulate = { kind = "explicit", proteins = { P0DJI8 = 2 } }"#).unwrap();
        assert_eq!(explicit(&rs[0])["P0DJI8"], 2.0); // an integer is still a fold change
    }

    /// BACKWARDS COMPATIBILITY. The deprecated scalar form still parses, and still means
    /// "one magnitude for the whole set".
    #[test]
    fn the_deprecated_scalar_form_still_works() {
        let rs = reg(r#"regulate = { proteins = ["P0DJI8", "P02741"], log2fc = 1.0 }"#).unwrap();
        assert_eq!(rs.len(), 1);
        let m = explicit(&rs[0]);
        assert_eq!(m.len(), 2);
        assert_eq!(m["P0DJI8"], 1.0);
        assert_eq!(m["P02741"], 1.0);

        // …and so does the untagged generative form.
        let rs = reg(r#"regulate = { fraction = 0.05, log2fc_sd = 1.0 }"#).unwrap();
        assert!(matches!(rs[0], design::Regulation::Generative { .. }));
    }

    /// The old untagged enum reported only *"data did not match any variant of untagged enum
    /// RegulateSpec"*, which for a 14-protein map is unusable. Every rejection must name what
    /// was wrong.
    #[test]
    fn malformed_blocks_are_rejected_with_a_usable_message() {
        // `kind = "explicit"` with a bare list — the form that has no per-protein magnitude.
        let e = reg(r#"regulate = { kind = "explicit", proteins = ["P0DJI8"] }"#).unwrap_err();
        assert!(format!("{e}").contains("MAP"), "{e}");

        let e = reg(r#"regulate = [{ kind = "sideways", proteins = { A = 1 } }]"#).unwrap_err();
        assert!(format!("{e}").contains("sideways"), "{e}");

        let e = reg(r#"regulate = { log2fc = 1.0 }"#).unwrap_err();
        assert!(format!("{e}").contains("severe"), "{e}");

        let e = reg(r#"regulate = { kind = "generative", fraction = 0.05 }"#).unwrap_err();
        assert!(format!("{e}").contains("log2fc_sd"), "{e}");

        assert!(reg(r#"regulate = "everything""#).is_err());
        assert!(reg(r#"regulate = { kind = "explicit", proteins = {} }"#).is_err());
    }
}
