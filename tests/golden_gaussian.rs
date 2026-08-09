//! **`--peak-shape gaussian` must stay byte-identical to the pre-EMG binary — on real output, not
//! on kernel values.**
//!
//! # Why the unit tests were not enough
//!
//! `src/render.rs` already proves `elution_frac(Gaussian, ..)` is bit-for-bit `gauss_frac(..)`. That
//! is a statement about one function. What the 30 cached cohort arms (124 h, 152 GB) actually depend
//! on is a statement about the whole pipeline: placement, the frame sweep, intensity quantisation,
//! zstd framing, the SQLite tables, the vendor container, and the answer key — end to end, per
//! writer, per acquisition mode. A refactor can leave the kernel bit-exact and still move a bin
//! boundary, a rounding step, or a compression level.
//!
//! So this test renders **real artifacts** and hashes them against artifacts produced by the parent
//! commit `52e6c91`, which predates `--peak-shape` entirely and therefore renders the Gaussian by
//! construction. Every hash in `tests/golden/gaussian_golden.json` was produced by both binaries and
//! agreed.
//!
//! # The signal / metadata split
//!
//! This work deliberately ADDS a provenance stamp (peak shape, `emg_k`, `n_sigma`) to the output, so
//! "byte-identical" cannot be claimed for the whole file — that is the point of the stamp. The
//! manifest therefore marks each artifact:
//!
//! * `parent_equal: true` — byte-identical to the parent commit. This is every carrier of rendered
//!   signal: `analysis.tdf_bin`, the Thermo `.raw`, the SCIEX mzML, and every answer key's parquet
//!   DATA region (hashed as the file minus its thrift footer, so the added footer metadata is
//!   excluded and nothing else is).
//! * `parent_equal: false` — `analysis.tdf` only, where the stamp lives. Parent-vs-head equality was
//!   verified by diffing full `sqlite3 .dump` output and confirming the ONLY difference is the three
//!   `Sim*` rows the manifest lists. Its hash is still pinned here, so the stamp cannot drift either.
//!
//! # Fixtures
//!
//! The inputs are machine-local paths (a 12,228-precursor feature space, a 5,692-frame reference
//! `.d`, a real Astral template). A case whose inputs are absent is skipped with a LOUD, specific
//! message — and if that leaves nothing to check at all, the test FAILS rather than passing vacuously.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MANIFEST: &str = include_str!("golden/gaussian_golden.json");

/// sha256 of a whole file.
fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// sha256 of a parquet file's DATA region — everything before the thrift footer.
///
/// Layout is `PAR1 | data pages | footer thrift | u32 footer_len | PAR1`. Hashing up to
/// `len - 8 - footer_len` isolates the column data from the footer, which is where the peak-shape
/// stamp now lives. That makes "the rendered numbers are unchanged" a checkable claim even though
/// the file as a whole is intentionally not.
fn parquet_data_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let len = f.seek(SeekFrom::End(0))?;
    assert!(len > 12, "{} is too short to be parquet", path.display());
    f.seek(SeekFrom::End(-8))?;
    let mut tail = [0u8; 8];
    f.read_exact(&mut tail)?;
    assert_eq!(&tail[4..], b"PAR1", "{} has no PAR1 trailer", path.display());
    let footer_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as u64;
    let data_end = len - 8 - footer_len;

    f.seek(SeekFrom::Start(0))?;
    let mut h = Sha256::new();
    let mut left = data_end;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let want = std::cmp::min(buf.len() as u64, left) as usize;
        f.read_exact(&mut buf[..want])?;
        h.update(&buf[..want]);
        left -= want as u64;
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Where the release binaries live. `CARGO_TARGET_DIR` wins if set, as it does in CI.
fn bin_dir() -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(d) => PathBuf::from(d).join("release"),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("release"),
    }
}

#[test]
fn gaussian_output_matches_the_pre_emg_binary() {
    let m: Value = serde_json::from_str(MANIFEST).expect("golden manifest parses");
    let fixtures = &m["fixtures"];
    let parent = m["parent_commit"].as_str().unwrap();

    // `cargo test` runs concurrently with other targets; keep the work under target/ so a stale run
    // never lands in the repo, and use a per-process directory so parallel invocations don't collide.
    let work = std::env::temp_dir().join(format!("timsim-gaussian-golden-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).expect("create work dir");

    let mut ran = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in m["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bin = bin_dir().join(case["bin"].as_str().unwrap());

        if !bin.exists() {
            skipped.push(format!(
                "{name}: binary {} not built (needs `cargo build --release --features tdf,thermo,sciex`)",
                bin.display()
            ));
            continue;
        }
        let missing: Vec<String> = case["needs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap())
            .filter(|k| !Path::new(fixtures[k].as_str().unwrap()).exists())
            .map(|k| format!("{k}={}", fixtures[k].as_str().unwrap()))
            .collect();
        if !missing.is_empty() {
            skipped.push(format!("{name}: fixture(s) absent on this machine: {}", missing.join(", ")));
            continue;
        }

        let out = work.join(name);
        std::fs::create_dir_all(&out).unwrap();
        let args: Vec<String> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                let s = a.as_str().unwrap();
                if let Some(key) = s.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                    if key == "out" {
                        return out.display().to_string();
                    }
                    return fixtures[key].as_str().unwrap().to_string();
                }
                // `{out}/data.d` and friends.
                s.replace("{out}", &out.display().to_string())
            })
            .collect();

        let status = std::process::Command::new(&bin)
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .status()
            .unwrap_or_else(|e| panic!("{name}: spawning {} failed: {e}", bin.display()));
        assert!(status.success(), "{name}: render exited {status}");
        ran += 1;

        for art in case["artifacts"].as_array().unwrap() {
            let rel = art["path"].as_str().unwrap();
            let path = out.join(rel);
            let want = art["sha256"].as_str().unwrap();
            let got = match art["hash"].as_str().unwrap() {
                "file_sha256" => file_sha256(&path),
                "parquet_data_sha256" => parquet_data_sha256(&path),
                other => panic!("{name}: unknown hash kind {other}"),
            }
            .unwrap_or_else(|e| panic!("{name}: hashing {} failed: {e}", path.display()));

            if got != want {
                let equal = art["parent_equal"].as_bool().unwrap();
                failures.push(format!(
                    "{name}/{rel} ({}): got {got}, golden {want}\n      {}",
                    art["role"].as_str().unwrap(),
                    if equal {
                        format!("This artifact is byte-identical to parent commit {parent}. It just changed, so `--peak-shape gaussian` no longer reproduces the pre-EMG render — every cached Gaussian artifact is now unreproducible from this tree.")
                    } else {
                        "This artifact carries the peak-shape provenance stamp; the stamp or the tables around it changed.".to_string()
                    }
                ));
            }
        }

        // The stamp must be READABLE BACK from the artifact, not merely present in some byte range —
        // that is the whole point of recording it. Both readers reconstruct a full `PeakShape`.
        for art in case["artifacts"].as_array().unwrap() {
            let rel = art["path"].as_str().unwrap();
            if rel.ends_with(".parquet") {
                let shape = timsim_cli::provenance::read_parquet_shape(out.join(rel))
                    .unwrap_or_else(|e| panic!("{name}/{rel}: answer key does not self-identify its shape: {e}"));
                assert_eq!(
                    shape,
                    timsim_cli::render::PeakShape::Gaussian,
                    "{name}/{rel}: answer key claims the wrong kernel"
                );
            }
            #[cfg(feature = "tdf")]
            if rel.ends_with("analysis.tdf") {
                let d = out.join(rel).parent().unwrap().to_path_buf();
                let shape = timsim_cli::provenance::read_tdf_shape(&d)
                    .unwrap_or_else(|e| panic!("{name}: .d does not self-identify its shape: {e}"));
                assert_eq!(shape, timsim_cli::render::PeakShape::Gaussian, "{name}: .d claims the wrong kernel");
            }
        }
    }

    let _ = std::fs::remove_dir_all(&work);

    for s in &skipped {
        eprintln!("GOLDEN SKIPPED — {s}");
    }
    for w in m["not_exercised"].as_array().unwrap() {
        eprintln!(
            "GOLDEN NOT EXERCISED — {}: {}",
            w["writer"].as_str().unwrap(),
            w["reason"].as_str().unwrap()
        );
    }
    assert!(failures.is_empty(), "Gaussian output drifted:\n  - {}", failures.join("\n  - "));
    assert!(
        ran > 0,
        "the Gaussian golden checked NOTHING — every case was skipped, which is not a pass:\n  - {}",
        skipped.join("\n  - ")
    );
    eprintln!("GOLDEN: {ran}/{} writer/mode combinations verified against parent {parent}", m["cases"].as_array().unwrap().len());
}
