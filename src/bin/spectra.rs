//! `timsim-spectra` — materialise the **instrument-independent** two-spectra-per-ion artifact.
//!
//! For each peptide ion (precursor) it writes two rows: an **MS1** spectrum (precursor isotopes) and
//! an **MS2** spectrum (fragment isotopes, with Prosit intensities), both as pure `(m/z, intensity)`
//! via mscore's peptide-ion path. No instrument geometry: a downstream *projector* (`timsim-render`
//! for timsTOF, others for Thermo/Sciex) maps these peaks onto `(frame, scan, tof)` / `(scan, m/z)`.
//!
//! This is the seam that lets one spectrum computation drive any instrument — the chemistry is done
//! once here; only the projection is instrument-specific.
//!
//! # Memory: chunked by ROW INDEX, streamed out
//!
//! At full-proteome scale this stage used to hold three complete copies of the run — every fragment
//! row grouped into a nested map (462M rows for 9M precursors), every generated spectrum, and then
//! an Arrow copy of all of it — and peaked past 20 GB, which is what stopped a 20-sample cohort.
//!
//! It now works a **chunk of precursor rows at a time** and appends each chunk to a streaming
//! writer, so peak memory is one chunk plus one parquet row group rather than the whole run.
//!
//! The chunking is by **row index in the precursor file**, deliberately, and not by `precursor_id`:
//!
//! - `precursor_id` is a hash, so the `fragment_intensities` file is unsorted on it and every row
//!   group spans nearly the whole `u64` range — there is no merge-join to exploit, and any id-keyed
//!   partition (`id % K`) would emit the output in a **different order**.
//! - Output order is load-bearing. The render consumes `ion_spectra` in precursor-file order and its
//!   byte-identity depends on it, so chunk *c* must contain precursor rows `c*S .. (c+1)*S` and the
//!   chunks must be processed in ascending order. That is exactly what happens below.
//!
//! Fragments are matched to their chunk by a single streaming **partition pass**: each fragment row
//! is routed to a fixed-width temp file for its precursor's chunk, which turns one 20 GB in-memory
//! grouping into one sequential read and K sequential writes. The temp directory is removed on the
//! way out, including on error.

use anyhow::{anyhow, bail, Result};
use arrow::array::{Array, Float32Array, UInt8Array, UInt16Array, UInt64Array};
use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use timsim_cli::sequences::{load_annotated, load_bare, load_mod_info};
use timsim_cli::spectrum::{fragment_peaks, normalize_total, precursor_peaks, FragKey, Peaks, SpectrumOpts};
use timsim_schema::tables::{
    fragment_intensities as FI, ion_spectra as SP, precursors as PRE,
};

#[derive(Parser)]
#[command(name = "timsim-spectra", about = "peptide ions -> instrument-independent MS1+MS2 spectra")]
struct Args {
    #[arg(long)]
    precursors: PathBuf,
    #[arg(long)]
    peptides: PathBuf,
    #[arg(long)]
    modforms: PathBuf,
    #[arg(long)]
    modifications: PathBuf,
    #[arg(long)]
    fragment_intensities: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Cap on precursors (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Precursor rows per chunk (0 = auto, sized so a chunk holds roughly [`TARGET_FRAG_ROWS`]
    /// fragment rows). Peak memory is one chunk, not the whole run. Any value must produce
    /// **byte-identical** output — the chunk boundary moves where the work happens, never what is
    /// written — so force a small one to test the stitching, exactly as `--render-chunks` does.
    #[arg(long, default_value_t = 0)]
    chunk_size: usize,
}

/// Fragment rows a chunk aims to hold. The nested per-precursor map costs ~40 B/row, so ~16M rows is
/// well under a GB — small enough for a cohort machine, large enough that the partition pass writes
/// few files and the rayon pool has real work per chunk.
const TARGET_FRAG_ROWS: u64 = 16_000_000;

/// Temp files held open at once during the partition pass. A chunk count above this is handled in
/// waves (one extra streaming read of the fragment file each) rather than by opening 10k file
/// descriptors — which only happens if a caller forces a tiny `--chunk-size`.
const MAX_OPEN_TEMP_FILES: usize = 128;

/// Fixed-width partition record: `precursor_id | intensity | ion_type | ordinal | frag_charge | pad`.
/// Fixed width so a chunk file is read back with no framing and no allocation per row.
const REC: usize = 20;

struct PrecRow {
    precursor_id: u64,
    modform_id: u64,
    charge: i32,
}

fn main() -> Result<()> {
    let a = Args::parse();

    // Pass 1: precursor rows + the modform/peptide ids they touch.
    let mut rows: Vec<PrecRow> = Vec::new();
    let (mut need_pep, mut need_mf): (HashSet<u64>, HashSet<u64>) = (HashSet::new(), HashSet::new());
    'outer: for b in timsim_schema::read(&a.precursors, PRE::TABLE)? {
        let pcid: &UInt64Array = b.column_by_name(PRE::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let pid: &UInt64Array = b.column_by_name(PRE::PEPTIDE_ID).unwrap().as_any().downcast_ref().unwrap();
        let mfid: &UInt64Array = b.column_by_name(PRE::MODFORM_ID).unwrap().as_any().downcast_ref().unwrap();
        let chg: &UInt8Array = b.column_by_name(PRE::CHARGE).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            need_pep.insert(pid.value(i));
            need_mf.insert(mfid.value(i));
            rows.push(PrecRow {
                precursor_id: pcid.value(i),
                modform_id: mfid.value(i),
                charge: chg.value(i).max(1) as i32,
            });
            if a.limit > 0 && rows.len() >= a.limit {
                break 'outer;
            }
        }
    }

    // Sequences.
    let mod_info = load_mod_info(&a.modifications)?;
    let bare = load_bare(&a.peptides, &need_pep)?;
    let annotated = load_annotated(&a.modforms, &need_mf, &bare, &mod_info)?;
    drop(bare);
    drop(need_pep);
    drop(need_mf);

    // How to slice the precursor rows. Auto-size from the REAL fragments-per-precursor of this input
    // (read from the parquet footer, no data pages touched) rather than a guessed constant.
    let total_frag_rows = fragment_row_count(&a.fragment_intensities)?;
    let n_prec = rows.len();
    let chunk_size = if a.chunk_size > 0 {
        a.chunk_size
    } else if n_prec == 0 {
        1
    } else {
        let per_prec = (total_frag_rows as f64 / n_prec as f64).max(1.0);
        ((TARGET_FRAG_ROWS as f64 / per_prec) as usize).clamp(1, n_prec)
    };
    let n_chunks = if n_prec == 0 { 0 } else { (n_prec - 1) / chunk_size + 1 };

    // `precursor_id -> chunk`. This map replaces the old `keep` set (membership is "has a chunk"),
    // so the id side costs one map, not two.
    let mut chunk_of: HashMap<u64, u32> = HashMap::with_capacity(n_prec);
    for (i, r) in rows.iter().enumerate() {
        let c = (i / chunk_size) as u32;
        if let Some(prev) = chunk_of.insert(r.precursor_id, c) {
            if prev != c {
                // precursor_id is the primary key of the precursors table. If it repeats across a
                // chunk boundary its fragments can only be routed to one of the two chunks, and the
                // other row would silently lose its MS2 — so say so instead of writing it.
                bail!(
                    "precursor_id {} appears twice in {} (rows in chunks {prev} and {c}); \
                     precursor_id must be unique",
                    r.precursor_id,
                    a.precursors.display()
                );
            }
        }
    }

    println!(
        "  timsim-spectra: {} precursors, {} fragment rows -> {} chunk(s) of {} precursors",
        n_prec, total_frag_rows, n_chunks, chunk_size
    );

    let scratch = Scratch::new(&a.out)?;
    let mut out = SpectraOut::new(&a.out)?;
    let opts = SpectrumOpts::default();
    let (mut n_ms1, mut n_ms2) = (0u64, 0u64);

    // Waves exist only so a forced tiny `--chunk-size` cannot blow the fd limit; with the auto size
    // there is exactly one wave, hence exactly one pass over the fragment file.
    let mut wave_lo = 0u32;
    while (wave_lo as usize) < n_chunks {
        let wave_hi = ((wave_lo as usize + MAX_OPEN_TEMP_FILES).min(n_chunks)) as u32;
        partition(&a.fragment_intensities, &chunk_of, wave_lo, wave_hi, &scratch)?;

        for c in wave_lo..wave_hi {
            // Only this chunk's fragments are resident, in the same shape the whole-file map had.
            let frags = load_chunk(&scratch.chunk_path(c))?;
            let lo = c as usize * chunk_size;
            let hi = (lo + chunk_size).min(n_prec);

            // Generate the two spectra per ion — IN PARALLEL. Each precursor is independent (the
            // annotated/frags maps are read-only), and mscore's isotope/fragment maths are pure, so
            // this is embarrassingly parallel. `flat_map_iter` runs the per-precursor closure across
            // the rayon pool; the order-preserving `collect` keeps the output deterministic (same
            // bytes as the serial version). This is the pole at 250K+ scale — serial it is ~100 min,
            // parallel ~7 min on 16 cores.
            use rayon::prelude::*;
            let generated: Vec<(u64, u8, Peaks)> = rows[lo..hi]
                .par_iter()
                .flat_map_iter(|r| {
                    let mut out: Vec<(u64, u8, Peaks)> = Vec::new();
                    if let Some(ann) = annotated.get(&r.modform_id) {
                        // MS1 — always present. Unit-total so precursor and fragments share the
                        // current scale; the render supplies the absolute level (per-ion abundance).
                        let mut ms1 = precursor_peaks(ann, r.charge, opts);
                        if !ms1.is_empty() {
                            normalize_total(&mut ms1);
                            out.push((r.precursor_id, 1, ms1));
                        }
                        // MS2 — only if the ion has predicted fragments. Same unit total as MS1
                        // (ion-current conservation): raw Prosit fragments sum to ~7, which would
                        // render ~7× too hot.
                        if let Some(per_ion) = frags.get(&r.precursor_id) {
                            let mut ms2 = fragment_peaks(ann, r.charge, per_ion, opts);
                            if !ms2.is_empty() {
                                normalize_total(&mut ms2);
                                out.push((r.precursor_id, 2, ms2));
                            }
                        }
                    }
                    out
                })
                .collect();
            drop(frags);
            let _ = std::fs::remove_file(scratch.chunk_path(c));

            // Consumed, not borrowed: each spectrum is dropped as soon as it has been appended, so
            // the chunk's peaks and the pending row group are never both fully resident.
            for (pc, lv, pk) in generated {
                if lv == 1 {
                    n_ms1 += 1;
                } else {
                    n_ms2 += 1;
                }
                out.push(pc, lv, &pk)?;
            }
        }
        wave_lo = wave_hi;
    }

    out.close()?;
    drop(scratch);
    println!("  timsim-spectra: {} MS1 + {} MS2 spectra for {} precursors -> {}",
             n_ms1, n_ms2, rows.len(), a.out.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Output: spectra in, row-group-aligned batches out.

/// Accumulates generated spectra and appends them to the artifact a **row group at a time**.
///
/// The flush size is not a tuning knob: it is `timsim_schema::ROW_GROUP_ROWS`, because that is the
/// boundary at which a streamed file is byte-identical to the single-batch file this stage used to
/// write (parquet page boundaries otherwise move with the producer's batch sizes — see
/// `timsim_schema::Writer`). Handing the writer exactly that unit also takes its zero-copy path, so
/// the peak here is one row group of peaks (~1 GB at 77 peaks/spectrum) and nothing more.
struct SpectraOut {
    writer: timsim_schema::Writer,
    pcid: Vec<u64>,
    level: Vec<u8>,
    mz: arrow::array::ListBuilder<arrow::array::Float64Builder>,
    inten: arrow::array::ListBuilder<arrow::array::Float32Builder>,
}

impl SpectraOut {
    fn new(out: &Path) -> Result<Self> {
        use arrow::array::{Float32Builder, Float64Builder, ListBuilder};
        Ok(SpectraOut {
            writer: timsim_schema::Writer::new(out, SP::TABLE, "timsim-spectra", None)?,
            pcid: Vec::with_capacity(timsim_schema::ROW_GROUP_ROWS),
            level: Vec::with_capacity(timsim_schema::ROW_GROUP_ROWS),
            mz: ListBuilder::new(Float64Builder::new()),
            inten: ListBuilder::new(Float32Builder::new()),
        })
    }

    fn push(&mut self, pc: u64, level: u8, peaks: &Peaks) -> Result<()> {
        self.pcid.push(pc);
        self.level.push(level);
        for &(mz, i) in peaks {
            self.mz.values().append_value(mz);
            self.inten.values().append_value(i as f32);
        }
        self.mz.append(true);
        self.inten.append(true);
        if self.pcid.len() >= timsim_schema::ROW_GROUP_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        use arrow::array::{UInt64Array as U64, UInt8Array as U8};
        use arrow::record_batch::RecordBatch;
        if self.pcid.is_empty() {
            return Ok(());
        }
        let spec = timsim_schema::tables::by_name(SP::TABLE)
            .ok_or_else(|| anyhow!("no ion_spectra spec"))?;
        let cols: Vec<arrow::array::ArrayRef> = vec![
            std::sync::Arc::new(U64::from(std::mem::take(&mut self.pcid))),
            std::sync::Arc::new(U8::from(std::mem::take(&mut self.level))),
            std::sync::Arc::new(self.mz.finish()),
            std::sync::Arc::new(self.inten.finish()),
        ];
        self.writer.write(&RecordBatch::try_new(spec.schema.clone(), cols)?)?;
        Ok(())
    }

    fn close(mut self) -> Result<()> {
        self.flush()?;
        self.writer.close()?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragment partitioning.

/// The temp directory the partition files live in. Beside the output, so it lands on the same
/// filesystem (the artifact's volume is the one sized for this run), and removed on the way out —
/// including on error and on panic, which is what the `Drop` is for.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(out: &Path) -> Result<Self> {
        let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let dir = parent.join(format!(".timsim-spectra-part.{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        Ok(Scratch { dir })
    }

    fn chunk_path(&self, chunk: u32) -> PathBuf {
        self.dir.join(format!("c{chunk:06}.frag"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Total rows in the fragment artifact, from the parquet footer — no data page is read. Used to size
/// the chunks from this input's real fragments-per-precursor.
fn fragment_row_count(path: &Path) -> Result<u64> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    Ok(b.metadata().file_metadata().num_rows().max(0) as u64)
}

/// Stream the fragment artifact once and route every row belonging to chunks `[lo, hi)` into that
/// chunk's temp file. Rows of precursors this run does not keep (`--limit`) are dropped here, exactly
/// as the old `keep` filter dropped them.
///
/// Memory is `hi - lo` write buffers. **Order within a chunk is the file's order**, so the
/// last-write-wins behaviour of the old `insert` on a duplicate `(ion_type, ordinal, charge)` key is
/// preserved.
fn partition(
    frags: &Path,
    chunk_of: &HashMap<u64, u32>,
    lo: u32,
    hi: u32,
    scratch: &Scratch,
) -> Result<()> {
    let mut sinks: Vec<BufWriter<File>> = Vec::with_capacity((hi - lo) as usize);
    for c in lo..hi {
        sinks.push(BufWriter::with_capacity(1 << 16, File::create(scratch.chunk_path(c))?));
    }

    let mut rec = [0u8; REC];
    for b in timsim_schema::read_stream(frags, FI::TABLE)? {
        let b = b?;
        let pcid: &UInt64Array = b.column_by_name(FI::PRECURSOR_ID).unwrap().as_any().downcast_ref().unwrap();
        let it: &arrow::array::StringArray = b.column_by_name(FI::ION_TYPE).unwrap().as_any().downcast_ref().unwrap();
        let ord: &UInt16Array = b.column_by_name(FI::ORDINAL).unwrap().as_any().downcast_ref().unwrap();
        let fc: &UInt8Array = b.column_by_name(FI::FRAG_CHARGE).unwrap().as_any().downcast_ref().unwrap();
        let inten: &Float32Array = b.column_by_name(FI::INTENSITY).unwrap().as_any().downcast_ref().unwrap();
        for i in 0..b.num_rows() {
            let pc = pcid.value(i);
            let c = match chunk_of.get(&pc) {
                Some(&c) if c >= lo && c < hi => c,
                _ => continue,
            };
            let ion = it.value(i).chars().next().unwrap_or('?');
            rec[0..8].copy_from_slice(&pc.to_le_bytes());
            rec[8..12].copy_from_slice(&inten.value(i).to_le_bytes());
            rec[12..16].copy_from_slice(&(ion as u32).to_le_bytes());
            rec[16..18].copy_from_slice(&ord.value(i).to_le_bytes());
            rec[18] = fc.value(i);
            rec[19] = 0;
            sinks[(c - lo) as usize].write_all(&rec)?;
        }
    }

    for s in sinks.iter_mut() {
        s.flush()?;
    }
    Ok(())
}

/// Read one chunk's partition file back into the map shape the generator wants:
/// `precursor_id -> (ion_type, ordinal, frag_charge) -> intensity`.
fn load_chunk(path: &Path) -> Result<HashMap<u64, HashMap<FragKey, f64>>> {
    let file = File::open(path)?;
    let n = (file.metadata()?.len() / REC as u64) as usize;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut out: HashMap<u64, HashMap<FragKey, f64>> = HashMap::new();
    const BLOCK: usize = 8192;
    let mut buf = vec![0u8; REC * BLOCK];
    let mut left = n;
    while left > 0 {
        let take = left.min(BLOCK);
        r.read_exact(&mut buf[..take * REC])?;
        for j in 0..take {
            let b = &buf[j * REC..(j + 1) * REC];
            let pc = u64::from_le_bytes(b[0..8].try_into().unwrap());
            let inten = f32::from_le_bytes(b[8..12].try_into().unwrap());
            let ion = char::from_u32(u32::from_le_bytes(b[12..16].try_into().unwrap())).unwrap_or('?');
            let ord = u16::from_le_bytes(b[16..18].try_into().unwrap());
            out.entry(pc).or_default().insert((ion, ord, b[18]), inten as f64);
        }
        left -= take;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The partition record is the one place a fragment leaves Arrow and comes back. Every field
    /// must survive — an `ordinal` or `frag_charge` mangled here would attach a Prosit intensity to
    /// the wrong ion, which is a silently wrong spectrum rather than a crash.
    #[test]
    fn a_fragment_survives_the_round_trip_through_a_partition_file() {
        let dir = std::env::temp_dir().join(format!("timsim_spectra_part_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.frag");

        let want: Vec<(u64, char, u16, u8, f32)> = vec![
            (1, 'b', 1, 1, 0.5),
            (1, 'y', 12, 2, 0.125),
            (u64::MAX, 'y', u16::MAX, 3, 1.0),
            (7, '?', 0, 1, 0.0),
        ];
        {
            let mut w = BufWriter::new(File::create(&path).unwrap());
            let mut rec = [0u8; REC];
            for (pc, ion, ord, fc, i) in &want {
                rec[0..8].copy_from_slice(&pc.to_le_bytes());
                rec[8..12].copy_from_slice(&i.to_le_bytes());
                rec[12..16].copy_from_slice(&(*ion as u32).to_le_bytes());
                rec[16..18].copy_from_slice(&ord.to_le_bytes());
                rec[18] = *fc;
                rec[19] = 0;
                w.write_all(&rec).unwrap();
            }
            w.flush().unwrap();
        }

        let got = load_chunk(&path).unwrap();
        assert_eq!(got.len(), 3, "one entry per precursor: {got:?}");
        for (pc, ion, ord, fc, i) in &want {
            assert_eq!(got[pc][&(*ion, *ord, *fc)], *i as f64);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Chunk *c* must be precursor rows `c*S .. (c+1)*S` of the file, in file order — the render
    /// reads `ion_spectra` positionally, so a partition that reordered rows (an `id % K` scheme, say)
    /// would change the output even though every spectrum in it was right.
    #[test]
    fn chunking_partitions_the_row_axis_contiguously_and_in_order() {
        let n_prec = 1000usize;
        for chunk_size in [1usize, 7, 999, 1000, 4096] {
            let n_chunks = (n_prec - 1) / chunk_size + 1;
            let mut seen: Vec<usize> = Vec::with_capacity(n_prec);
            for c in 0..n_chunks {
                let lo = c * chunk_size;
                let hi = (lo + chunk_size).min(n_prec);
                assert!(lo < hi, "chunk {c} of {chunk_size} is empty");
                seen.extend(lo..hi);
            }
            assert_eq!(seen, (0..n_prec).collect::<Vec<_>>(), "chunk_size {chunk_size}");
        }
    }
}
