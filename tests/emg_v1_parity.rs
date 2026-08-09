//! Cross-tool check: v2's EMG must reproduce **v1's realised elution profile**.
//!
//! # What makes this a cross-tool test and not a tautology
//!
//! The right-hand side of every assertion below is a number that a real timsim **v1** run wrote to
//! disk — `frame_abundance` in `synthetic_data.db`, produced by v1's own Rust kernel
//! (`mscore::algorithm::utility::calculate_frame_abundance_emg`, a 1000-step left Riemann sum over
//! `emg_function`) driven by v1's own Python job. Nothing in this file re-derives them. The fixture
//! is a verbatim dump; see `tests/data/v1_emg_profiles.json` for provenance.
//!
//! The left-hand side is v2's closed-form `emg_cdf_std`. The two implementations share no code and
//! not even the same quadrature — v1 integrates numerically, v2 evaluates a closed-form CDF — so
//! agreement is evidence about the *shape*, which is the claim being made.
//!
//! # The two things checked
//!
//! 1. **Profile.** For each frame v1 emitted, v2's mass over the same interval must match v1's
//!    weight. v1 integrates `[t - cycle, t]` in seconds with v1's own `(mu, sigma, lambda)`; v2 is
//!    handed those same parameters, so this isolates the peak shape from every unit/indexing
//!    convention. (v2's *render* uses a different bin convention — frame index `f` is the CENTRE of
//!    `[f-0.5, f+0.5]`, where v1's frame time is the END of `[t-cycle, t]`. That half-bin offset is a
//!    pre-existing v1/v2 convention difference on the Gaussian path too, and is deliberately NOT
//!    folded in here: this test is about the curve, not the sampling grid.)
//!
//! 2. **Mode anchoring.** v1 treats the predicted RT as the peak's MODE and solves backwards for
//!    `mu` through `erfcxinv` (`estimate_mu_from_mode_emg`). v2 instead locates the mode by a
//!    golden-section search on the standardised PDF. Two completely different inversions; they must
//!    land on the same `mu`, or every v2 peptide would elute systematically off.

use serde_json::Value;
use timsim_cli::render::{elution_ordinate, emg_cdf_std, Emg, PeakShape, V1_DEFAULT_EMG_K};

fn fixture() -> Value {
    let raw = include_str!("data/v1_emg_profiles.json");
    serde_json::from_str(raw).expect("fixture parses")
}

/// v2's mass over `[a, b]` for v1's `(mu, sigma, lambda)`, via the standardised CDF.
fn v2_mass(a: f64, b: f64, mu: f64, sigma: f64, lambda: f64) -> f64 {
    let k = 1.0 / (sigma * lambda); // v1's own reparameterisation
    emg_cdf_std((b - mu) / sigma, k) - emg_cdf_std((a - mu) / sigma, k)
}

#[test]
fn emg_reproduces_v1_realised_elution_profile() {
    let f = fixture();
    let cycle = f["cycle_seconds"].as_f64().unwrap();
    let mut all_abs: Vec<f64> = Vec::new();
    let mut worst = (0.0f64, String::new());

    let mut degenerate = 0usize;

    for p in f["peptides"].as_array().unwrap() {
        let (mu, sigma, lambda) =
            (p["mu"].as_f64().unwrap(), p["sigma"].as_f64().unwrap(), p["lam"].as_f64().unwrap());
        let times: Vec<f64> = p["times"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let w1: Vec<f64> = p["weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(times.len(), w1.len());

        // A well-formed v1 profile holds ~target_p = 0.999 of the peak. Anything far below that means
        // v1's OWN support search (`calculate_bounds_emg`, which seeds a binary search on
        // `[mu - 20*sigma - 2, mu + 60*sigma]`) failed and stored a window that misses the apex
        // entirely — measured at 13.9% of peptides in the source run, concentrated at small `k`
        // (large lambda), where the peak is narrow next to that seed span. Those rows say nothing
        // about the SHAPE, so they are reported but not used to bound the comparison.
        let total: f64 = w1.iter().sum();
        let well_formed = total > 0.99;
        if !well_formed {
            degenerate += 1;
        }

        let mut pmax = 0.0f64;
        for (&t, &want) in times.iter().zip(&w1) {
            let got = v2_mass(t - cycle, t, mu, sigma, lambda);
            let d = (got - want).abs();
            pmax = pmax.max(d);
            if well_formed {
                all_abs.push(d);
                if d > worst.0 {
                    worst = (d, format!("peptide {} t={t} v1={want} v2={got}", p["peptide_id"]));
                }
            }
        }
        println!(
            "peptide {:>5} k={:.4} frames={:>3} v1_mass={:.4} {} max_dev={:.2e}",
            p["peptide_id"],
            1.0 / (sigma * lambda),
            times.len(),
            total,
            if well_formed { "OK      " } else { "TRUNCATED" },
            pmax
        );
    }

    all_abs.sort_by(f64::total_cmp);
    let median = all_abs[all_abs.len() / 2];
    let max = *all_abs.last().unwrap();
    println!(
        "\nv1-vs-v2 elution weight, {} frames across well-formed profiles: max {max:.3e}, median {median:.3e}",
        all_abs.len()
    );
    println!("worst: {}", worst.1);
    println!("({degenerate} peptide(s) skipped: v1's own support search truncated them)");

    // v1's weights are stored ROUNDED TO 4 DECIMALS in the database (`0.0174, 0.0173, ...`), so the
    // floor on any comparison is 5e-5 — an artifact of v1's serialisation, not of either shape.
    // Landing AT that floor is the strongest agreement the stored data can express.
    assert!(max < 1e-4, "max deviation {max:.3e} exceeds v1's own 4-decimal storage granularity");
    assert!(median < 5e-5, "median deviation {median:.3e}");
    assert!(all_abs.len() > 500, "too few comparable frames ({})", all_abs.len());
}

/// Report (and pin) the peak's SHAPE metrics: width and asymmetry, measured off the real code.
///
/// Units are sigma, so they convert to seconds by multiplying by `sigma_frames * cycle_seconds`.
#[test]
fn emg_shape_metrics() {
    // Peak height, in sigma units, for a peak apexing at 0.
    let h = |shape: &PeakShape, z: f64| elution_ordinate(z, 0.0, 1.0, shape);
    // Solve h(z) = frac on one side of the apex by bisection.
    let edge = |shape: &PeakShape, frac: f64, dir: f64| {
        let (mut near, mut far) = (0.0f64, dir * 200.0);
        for _ in 0..200 {
            let mid = 0.5 * (near + far);
            if h(shape, mid) > frac { near = mid } else { far = mid }
        }
        0.5 * (near + far)
    };
    let metrics = |shape: &PeakShape| {
        let (l50, r50) = (edge(shape, 0.5, -1.0), edge(shape, 0.5, 1.0));
        let (l10, r10) = (edge(shape, 0.1, -1.0), edge(shape, 0.1, 1.0));
        // FWHM, and the chromatographic asymmetry factor As = B/A at 10% height.
        (r50 - l50, (r50 - l50) / (-l50) / 2.0, r10 / -l10)
    };

    let (g_fwhm, g_sym, g_as) = metrics(&PeakShape::Gaussian);
    println!("gaussian : FWHM {g_fwhm:.4} sigma, half-width ratio {g_sym:.4}, As(10%) {g_as:.4}");
    assert!((g_fwhm - 2.354_820).abs() < 1e-4, "Gaussian FWHM must be 2*sqrt(2 ln2) sigma");
    assert!((g_as - 1.0).abs() < 1e-6, "a Gaussian must be symmetric");

    for &k in &[0.25, V1_DEFAULT_EMG_K, 1.0, 2.0] {
        let e = PeakShape::Emg(Emg::new(k, 3.0));
        let (fwhm, _, asf) = metrics(&e);
        println!("emg k={k:.4}: FWHM {fwhm:.4} sigma, As(10%) {asf:.4}");
        assert!(fwhm > g_fwhm, "k={k}: tailing must widen the peak ({fwhm} <= {g_fwhm})");
        assert!(asf > 1.0, "k={k}: As(10%) must exceed 1 for a right-tailed peak");
    }
}

/// v2's golden-section mode search must agree with v1's `erfcxinv` inversion.
#[test]
fn emg_mode_offset_matches_v1_mu_inversion() {
    let f = fixture();
    let mut worst: f64 = 0.0;
    for p in f["peptides"].as_array().unwrap() {
        let (apex, mu, sigma, lambda) = (
            p["apex_seconds"].as_f64().unwrap(),
            p["mu"].as_f64().unwrap(),
            p["sigma"].as_f64().unwrap(),
            p["lam"].as_f64().unwrap(),
        );
        let k = 1.0 / (sigma * lambda);
        // v1: mu = apex - (mode offset). v2 must recover the same offset from k alone.
        let v1_offset_sigmas = (apex - mu) / sigma;
        let v2_offset_sigmas = Emg::new(k, 3.0).mode_offset();
        let d = (v1_offset_sigmas - v2_offset_sigmas).abs();
        println!(
            "peptide {} k={k:.4}: v1 mode offset {v1_offset_sigmas:.6} sigma, v2 {v2_offset_sigmas:.6} sigma, diff {d:.2e}",
            p["peptide_id"]
        );
        worst = worst.max(d);
    }
    // v1's mu is stored at full f64 precision, but its own inversion is a 10-step Newton iteration
    // on `erfcx`; 1e-4 sigma is far tighter than a frame (a frame is ~0.07 sigma here).
    assert!(worst < 1e-4, "worst mode-offset disagreement {worst:.2e} sigma");
}
