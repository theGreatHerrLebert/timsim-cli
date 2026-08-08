//! Mobility-derived collision energy — the CE a timsTOF DDA-PASEF run *actually* applies to a
//! precursor.
//!
//! # Why this exists
//!
//! `timsim-fragments` predicts fragment intensities at ONE collision energy for the whole run. That
//! is right for a no-IMS instrument (an Astral/Orbitrap DIA method sets one NCE), but it is wrong
//! for the timsTOF: on a dda-PASEF method the collision energy is **scan-driven** — the quadrupole
//! ramps CE with the mobility scan, so an early (compact, low `1/K0`) ion is fragmented harder than
//! a late one. v1 has always done this; see `dda_selection_scheme.schedule_precursors`, which writes
//! `activation_policy.collision_energy_for_scan(ion.scan_apex)` into `pasef_meta.CollisionEnergy`
//! and thereby into every transmitted ion's CE.
//!
//! # Nothing here is re-derived
//!
//! Every step reuses the component that already owns it, because a second implementation of a
//! calibration is a second place for it to be wrong:
//!
//! | step | owner |
//! |------|-------|
//! | CCS → `1/K0` (Mason-Schamp) | [`mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility`] |
//! | `1/K0` → scan | [`ms_io::data::calibration::MobilityCalibrator`] (Bruker ModelType-2), or [`SimpleIndexConverter`] |
//! | scan → CE | [`timsim_types::ActivationPolicy::bruker_pasef`] — the same policy object v1 calls |
//!
//! The placement half is deliberately identical to `timsim-render`'s `place_scan`: CCS→`1/K0`,
//! clamp into the acquisition band, map with the run's mobility calibration, clamp to the last
//! scan. An ion's CE is therefore the CE at the scan the renderer will actually put it in.
//!
//! # The dedup invariant
//!
//! CCS is predicted from `(sequence, charge, mz)` and `mz` from `(composition, charge)`, so CCS —
//! and therefore the scan, and therefore the CE — is a **deterministic function of the precursor's
//! (annotated sequence, charge)**. Positional isomers, which share both, share their CE. That is
//! what lets `timsim-fragments` keep deduplicating on `(sequence, charge)`: turning this capability
//! on adds a per-precursor CE column without adding a single model call. `timsim-fragments` — the
//! stage that holds the keys — measures the largest within-key CE spread and refuses the run if it
//! exceeds a tolerance, so the invariant is checked rather than assumed.

use anyhow::{anyhow, Result};

use ms_io::data::calibration::MobilityCalibrator;
use ms_io::data::handle::SimpleIndexConverter;
use ms_io::data::meta::{read_global_meta_sql, read_meta_data_sql, read_tims_calibration};
use mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility;
use timsim_types::ActivationPolicy;

/// Standard TIMS gas / temperature for Mason-Schamp (N2 at ~305 K — the imspy defaults the CCS
/// model was trained against). Same constants `timsim-render` places ions with; they must agree or
/// the CE would be read off a different scan than the ion is rendered into.
pub const MASS_GAS: f64 = 28.013;
pub const TEMP: f64 = 31.85;
pub const T_DIFF: f64 = 273.15;

/// v1's dda-PASEF activation defaults (`dda_selection_scheme.schedule_precursors`): CE in eV,
/// linear in the mobility scan. 54.1984 eV at scan 0 falling to ~22.6 eV at scan 917.
pub const CE_BIAS: f64 = 54.1984;
pub const CE_SLOPE: f64 = -0.0345;

/// How `1/K0` becomes a scan index.
pub enum ScanMap {
    /// The reference `.d`'s own Bruker ModelType-2 mobility calibration — the calibrator
    /// `timsim-render --reference-d` places ions with, so scan numbers agree exactly.
    Calibrated(MobilityCalibrator),
    /// Reference-free linear map, from [`SimpleIndexConverter::from_boundaries`] — the same
    /// fallback `timsim-render` uses when no reference `.d` is given.
    Linear { intercept: f64, slope: f64 },
}

/// The mobility half of an acquisition geometry: how many scans the ramp has, the `1/K0` band it
/// covers, and the map between them.
pub struct MobilityGeometry {
    pub n_scans: u32,
    pub im_min: f64,
    pub im_max: f64,
    pub map: ScanMap,
}

impl MobilityGeometry {
    /// Derive the geometry from a real `.d`: `NumScans` (max over frames) and the `GlobalMetadata`
    /// `1/K0` acquisition range, with the `TimsCalibration` row the first frame references — the
    /// exact selection `timsim-render::build_placement` makes.
    pub fn from_reference_d(path: &str) -> Result<Self> {
        let gm = read_global_meta_sql(path).map_err(|e| anyhow!("read reference GlobalMetadata: {e}"))?;
        let frames = read_meta_data_sql(path).map_err(|e| anyhow!("read reference Frames: {e}"))?;
        let f0 = frames.first().ok_or_else(|| anyhow!("reference .d has no frames"))?;
        let n_scans = frames.iter().map(|f| f.num_scans).max().unwrap_or(0) as u32;
        if n_scans == 0 {
            return Err(anyhow!("reference .d reports zero mobility scans"));
        }

        // Select the SAME calibration row the frames reference: a `.d` with several calibrations
        // would otherwise be read with coefficients that disagree with the frames' own.
        let tc = read_tims_calibration(path)
            .map_err(|e| anyhow!("{e}"))?
            .into_iter()
            .find(|c| c.id == f0.tims_calibration)
            .ok_or_else(|| anyhow!("no TimsCalibration with id {} in reference", f0.tims_calibration))?;
        if tc.model_type != 2 {
            return Err(anyhow!(
                "reference .d TimsCalibration ModelType is {} — MobilityCalibrator implements \
                 ModelType 2 only. Use the reference-free --n-scans/--im-min/--im-max geometry \
                 instead of silently mis-mapping mobility to scan.",
                tc.model_type
            ));
        }
        let mob = MobilityCalibrator::new(
            tc.c0, tc.c1, tc.c2, tc.c3, tc.c4, tc.c5, tc.c6, tc.c7, tc.c8, tc.c9,
        );
        Ok(MobilityGeometry {
            n_scans,
            im_min: gm.one_over_k0_range_lower,
            im_max: gm.one_over_k0_range_upper,
            map: ScanMap::Calibrated(mob),
        })
    }

    /// The reference-free geometry: a linear `1/K0`↔scan map over `[im_min, im_max]`, built by
    /// [`SimpleIndexConverter::from_boundaries`] so the coefficients are the renderer's, not ours.
    pub fn linear(n_scans: u32, im_min: f64, im_max: f64) -> Result<Self> {
        if n_scans < 2 {
            return Err(anyhow!("--n-scans must be at least 2 (got {n_scans})"));
        }
        if !(im_max > im_min) {
            return Err(anyhow!("--im-max ({im_max}) must exceed --im-min ({im_min})"));
        }
        // The m/z arguments are irrelevant to the mobility half; pass the renderer's own defaults so
        // the constructor is used exactly as `timsim-render` uses it.
        let conv = SimpleIndexConverter::from_boundaries(100.0, 1700.0, 400_000, im_min, im_max, n_scans - 1);
        Ok(MobilityGeometry {
            n_scans,
            im_min,
            im_max,
            map: ScanMap::Linear { intercept: conv.scan_intercept, slope: conv.scan_slope },
        })
    }

    /// `1/K0` → mobility scan, clamped into the acquisition band and onto the ramp. Mirrors
    /// `timsim-render`'s `place_scan` tail exactly.
    pub fn scan_for_one_over_k0(&self, one_over_k0: f64) -> u32 {
        let k0 = one_over_k0.clamp(self.im_min, self.im_max);
        let scan = match &self.map {
            ScanMap::Calibrated(mob) => mob.one_over_k0_to_scan(k0),
            ScanMap::Linear { intercept, slope } => ((k0 - intercept) / slope).max(0.0) as u32,
        };
        scan.min(self.n_scans - 1)
    }

    /// CCS (Å²) → the mobility scan the renderer will place this ion in.
    pub fn scan_for_ccs(&self, ccs: f64, mz: f64, charge: u32) -> u32 {
        self.scan_for_one_over_k0(ccs_to_one_over_reduced_mobility(
            ccs, mz, charge, MASS_GAS, TEMP, T_DIFF,
        ))
    }
}

/// The dda-PASEF activation policy, straight from `timsim-types` — the same constructor
/// `dda_selection_scheme` reaches through `PyActivationPolicy.bruker_pasef`.
pub fn pasef_policy(ce_bias: f64, ce_slope: f64) -> ActivationPolicy {
    ActivationPolicy::bruker_pasef(ce_bias, ce_slope)
}

/// Collision energy (eV) at a mobility scan, via the policy. Errors rather than guessing if a
/// non-scan-parameterised policy is ever passed in.
pub fn collision_energy_at(policy: &ActivationPolicy, scan: u32) -> Result<f64> {
    policy.collision_energy_for_scan(scan).ok_or_else(|| {
        anyhow!("activation policy is not scan-parameterised (no IMS) — CE is not a function of scan")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CE we emit is the policy's, over the whole plausible scan range — not a re-derivation of
    /// `ce_bias + ce_slope*scan`. Pinning it against the closed form here means a change in
    /// `timsim-types` that silently altered the ramp would fail this test rather than quietly shift
    /// every predicted spectrum.
    #[test]
    fn collision_energy_matches_the_v1_linear_ramp_over_the_full_scan_range() {
        let p = pasef_policy(CE_BIAS, CE_SLOPE);
        let mut worst = 0.0f64;
        for scan in 0u32..=2000 {
            let got = collision_energy_at(&p, scan).unwrap();
            let want = CE_BIAS + CE_SLOPE * scan as f64;
            worst = worst.max((got - want).abs());
        }
        assert_eq!(worst, 0.0, "CE must be the v1 ramp exactly, worst |diff| = {worst}");
    }

    /// A non-scan policy must not be read as if the scan number were an m/z.
    #[test]
    fn a_per_window_policy_is_refused_rather_than_misread() {
        use timsim_types::CollisionEnergyPolicy;
        let p = ActivationPolicy::thermo_nce(CollisionEnergyPolicy::Value(27.0));
        assert!(collision_energy_at(&p, 100).is_err());
    }

    /// The reference-free geometry is monotone (higher `1/K0` = earlier scan, as on the instrument),
    /// spans the ramp, and never leaves it.
    #[test]
    fn linear_geometry_is_monotone_and_bounded() {
        let g = MobilityGeometry::linear(918, 0.6, 1.6).unwrap();
        assert_eq!(g.scan_for_one_over_k0(1.6), 0);
        assert_eq!(g.scan_for_one_over_k0(0.6), 917);
        // Out-of-band mobilities clamp instead of running off the ramp.
        assert_eq!(g.scan_for_one_over_k0(9.9), 0);
        assert_eq!(g.scan_for_one_over_k0(0.01), 917);
        let mut last = 0u32;
        for i in 0..=100 {
            let k0 = 1.6 - (1.0 * i as f64 / 100.0);
            let s = g.scan_for_one_over_k0(k0);
            assert!(s >= last, "scan must increase as 1/K0 falls");
            assert!(s < g.n_scans);
            last = s;
        }
    }

    /// CE is a deterministic function of `(ccs, mz, charge)` — the property the `(sequence, charge)`
    /// dedup in `timsim-fragments` rests on.
    #[test]
    fn ce_is_deterministic_in_ccs_mz_charge() {
        let g = MobilityGeometry::linear(918, 0.6, 1.6).unwrap();
        let p = pasef_policy(CE_BIAS, CE_SLOPE);
        let a = collision_energy_at(&p, g.scan_for_ccs(420.0, 780.5, 2)).unwrap();
        let b = collision_energy_at(&p, g.scan_for_ccs(420.0, 780.5, 2)).unwrap();
        assert_eq!(a, b);
        // A bigger ion (higher CCS) drifts later -> lower scan -> HIGHER collision energy is NOT
        // implied; what is implied is that a different CCS can only change CE through the scan.
        let c = collision_energy_at(&p, g.scan_for_ccs(520.0, 780.5, 2)).unwrap();
        assert!(c != a || g.scan_for_ccs(520.0, 780.5, 2) == g.scan_for_ccs(420.0, 780.5, 2));
    }
}
