//! `timsim-digest` — which peptides exist. STRUCTURE.
//!
//! There is **no `--digestion-efficiency` here, and no seed**. Efficiency is a *quantity*: it
//! says how much, not what exists. And cleavage-blocking modifications (acetyl-K, ubiquitin-GG)
//! are regulated, so cleavage probabilities differ between conditions — putting a yield here
//! would make the structure condition-dependent and silently destroy sharing across samples.
//! Yields are emitted per-sample by `timsim-yield`.
//!
//! No seed because the digest is **analytic**: an exact expectation, not a sample.

use anyhow::Result;
use arrow::array::{ArrayRef, Float64Array, StringArray, UInt16Array, UInt32Array, UInt64Array};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use timsim_chem::{ids, mass, Enumerator, Protocol};
use timsim_cli::{batch, print_schema, producer};
use timsim_schema::tables::{cleavage_sites as CS, peptide_occurrences as PO, peptides as PEP, proteome as PROT};

#[derive(Parser)]
#[command(name = "timsim-digest", about = "proteome.parquet -> peptides + occurrences + cleavage sites (structure)")]
struct Args {
    #[arg(long)]
    proteome: Option<PathBuf>,
    #[arg(long)]
    enzyme: Option<String>,
    /// Numerical truncation, NOT a model parameter. Its cost is measured and reported.
    #[arg(long, default_value_t = 2)]
    max_missed_cleavages: u32,
    #[arg(long, default_value_t = 7)]
    min_length: u32,
    #[arg(long, default_value_t = 50)]
    max_length: u32,
    #[arg(long)]
    out_peptides: Option<PathBuf>,
    #[arg(long)]
    out_occurrences: Option<PathBuf>,
    #[arg(long)]
    out_cleavage_sites: Option<PathBuf>,
    /// OPT-IN cap on the number of unique peptides. `0` (default) keeps the full ANALYTIC digest (every
    /// peptide — the exact expectation). A positive value draws a DETERMINISTIC, seed-keyed RANDOM sample
    /// of that many peptides (and filters occurrences to match) — for a tractable run on a large proteome
    /// while keeping the full FASTA as the search space (v1's `num_sample_peptides`). Cleavage sites are
    /// protein-level and kept in full.
    #[arg(long, default_value_t = 0)]
    max_peptides: u64,
    /// Seed for `--max-peptides` sampling (ignored without a cap; the analytic digest has no seed).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    schema: bool,
}

/// splitmix64 — a stable, dep-free hash so the peptide sample is reproducible across runs (unlike the
/// std hasher, whose output is not guaranteed stable).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.schema {
        for t in [PEP::TABLE, PO::TABLE, CS::TABLE] {
            print_schema(t)?;
            println!();
        }
        return Ok(());
    }

    let proteome = a.proteome.ok_or_else(|| anyhow::anyhow!("--proteome is required"))?;
    let enzyme = a.enzyme.unwrap_or_else(|| "trypsin".into());
    let (op, oo, oc) = (
        a.out_peptides.ok_or_else(|| anyhow::anyhow!("--out-peptides is required"))?,
        a.out_occurrences.ok_or_else(|| anyhow::anyhow!("--out-occurrences is required"))?,
        a.out_cleavage_sites.ok_or_else(|| anyhow::anyhow!("--out-cleavage-sites is required"))?,
    );

    // Validate-on-read: a wrong or renamed column fails HERE, not three stages downstream.
    let batches = timsim_schema::read(&proteome, PROT::TABLE)?;
    let mut input: Vec<(String, String)> = Vec::new();
    for b in &batches {
        let id = b.column_by_name(PROT::PROTEIN_ID).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        let sq = b.column_by_name(PROT::SEQUENCE).unwrap().as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            input.push((id.value(i).to_string(), sq.value(i).to_string()));
        }
    }

    let e = Enumerator::new(
        Protocol::parse(&enzyme).map_err(anyhow::Error::msg)?,
        a.max_missed_cleavages,
        a.min_length,
        a.max_length,
    )
    .map_err(anyhow::Error::msg)?;

    let t = std::time::Instant::now();
    let digests = e.enumerate_all(&input);
    let elapsed = t.elapsed();

    // ── peptides (unique) + occurrences ──
    let seqs: std::collections::HashMap<&str, &str> =
        input.iter().map(|(i, s)| (i.as_str(), s.as_str())).collect();

    let mut coll = ids::CollisionCheck::new();
    let mut pep_seen: std::collections::HashMap<u64, (String, u16, f64)> = Default::default();
    let (mut o_pid, mut o_prot, mut o_start, mut o_end, mut o_mc) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut c_prot, mut c_pos) = (Vec::new(), Vec::new());
    let mut skipped_mass = 0usize;

    for d in &digests {
        let pseq = seqs[&*d.protein_id];
        for &pos in &d.cleavage_sites {
            c_prot.push(d.protein_id.to_string());
            c_pos.push(pos);
        }
        for o in &d.occurrences {
            let s = o.sequence(pseq);
            let id = ids::peptide_id(s);
            coll.insert(id, s).map_err(anyhow::Error::msg)?;
            if !pep_seen.contains_key(&id) {
                match mass::monoisotopic(s) {
                    Ok(m) => {
                        pep_seen.insert(id, (s.to_string(), s.len() as u16, m));
                    }
                    Err(_) => {
                        // Non-standard residue (X/B/Z). We refuse to guess a mass: a quietly
                        // wrong mass corrupts the mass balance without failing anything.
                        skipped_mass += 1;
                        continue;
                    }
                }
            }
            o_pid.push(id);
            o_prot.push(d.protein_id.to_string());
            o_start.push(o.start);
            o_end.push(o.end);
            o_mc.push(o.n_missed_cleavages);
        }
    }

    let mut peps: Vec<(u64, String, u16, f64)> =
        pep_seen.into_iter().map(|(k, (s, l, m))| (k, s, l, m)).collect();
    peps.sort_unstable_by_key(|p| p.0); // deterministic row order

    // Opt-in peptide cap (v1's num_sample_peptides): a deterministic seed-keyed random sample for a
    // tractable run on a large proteome, keeping the full FASTA as the search space. Occurrences are
    // filtered to the kept peptides; cleavage sites are protein-level and kept in full.
    let (mut o_pid, mut o_prot, mut o_start, mut o_end, mut o_mc) = (o_pid, o_prot, o_start, o_end, o_mc);
    let full_peptides = peps.len();
    if a.max_peptides > 0 && (peps.len() as u64) > a.max_peptides {
        peps.sort_by_key(|p| splitmix64(p.0 ^ a.seed)); // random order keyed by (id, seed)
        peps.truncate(a.max_peptides as usize);
        peps.sort_unstable_by_key(|p| p.0); // restore id order
        let kept: std::collections::HashSet<u64> = peps.iter().map(|p| p.0).collect();
        let keep: Vec<usize> = (0..o_pid.len()).filter(|&i| kept.contains(&o_pid[i])).collect();
        let (np, npr): (Vec<u64>, Vec<String>) = keep.iter().map(|&i| (o_pid[i], o_prot[i].clone())).unzip();
        let ns: Vec<u32> = keep.iter().map(|&i| o_start[i]).collect();
        let ne: Vec<u32> = keep.iter().map(|&i| o_end[i]).collect();
        let nm: Vec<u16> = keep.iter().map(|&i| o_mc[i]).collect();
        o_pid = np; o_prot = npr; o_start = ns; o_end = ne; o_mc = nm;
        eprintln!("  --max-peptides       : sampled {} of {full_peptides} unique peptides (seed {})", a.max_peptides, a.seed);
    }

    timsim_schema::write(&op, PEP::TABLE, &batch(PEP::TABLE, vec![
        Arc::new(UInt64Array::from(peps.iter().map(|p| p.0).collect::<Vec<_>>())) as ArrayRef,
        Arc::new(StringArray::from(peps.iter().map(|p| p.1.clone()).collect::<Vec<_>>())),
        Arc::new(UInt16Array::from(peps.iter().map(|p| p.2).collect::<Vec<_>>())),
        Arc::new(Float64Array::from(peps.iter().map(|p| p.3).collect::<Vec<_>>())),
    ])?, &producer("timsim-digest"), None)?;

    // The bounds travel WITH the structure. A consumer must never re-enter them: yielding at
    // a smaller bound than the digest used silently zeroes every occurrence beyond it.
    let bounds_meta: Vec<(&str, String)> = vec![
        (timsim_schema::meta::MAX_MISSED_CLEAVAGES, a.max_missed_cleavages.to_string()),
        (timsim_schema::meta::MIN_LENGTH, a.min_length.to_string()),
        (timsim_schema::meta::MAX_LENGTH, a.max_length.to_string()),
    ];

    let n_occ = o_pid.len();
    timsim_schema::write_with(&oo, PO::TABLE, &batch(PO::TABLE, vec![
        Arc::new(UInt64Array::from(o_pid)) as ArrayRef,
        Arc::new(StringArray::from(o_prot)),
        Arc::new(UInt32Array::from(o_start)),
        Arc::new(UInt32Array::from(o_end)),
        Arc::new(UInt16Array::from(o_mc)),
    ])?, &producer("timsim-digest"), None, &bounds_meta)?;

    let n_sites = c_pos.len();
    timsim_schema::write(&oc, CS::TABLE, &batch(CS::TABLE, vec![
        Arc::new(StringArray::from(c_prot)) as ArrayRef,
        Arc::new(UInt32Array::from(c_pos)),
    ])?, &producer("timsim-digest"), None)?;

    println!("  enzyme               : {enzyme}");
    println!("  max missed cleavages : {}  (numerical truncation; cost measured by timsim-yield)", a.max_missed_cleavages);
    println!("  length filter        : {}-{}", a.min_length, a.max_length);
    println!("  proteins             : {}", input.len());
    println!("  cleavage sites       : {n_sites}");
    println!("  unique peptides      : {}", peps.len());
    println!("  occurrences          : {n_occ}   (+{} — repeats and shared peptides, which a protein_ids list column would destroy)",
             n_occ.saturating_sub(peps.len()));
    if skipped_mass > 0 {
        eprintln!("  warning: {skipped_mass} peptides skipped (non-standard residues; we refuse to guess a mass)");
    }
    println!("  wall time            : {elapsed:.2?}");
    Ok(())
}
