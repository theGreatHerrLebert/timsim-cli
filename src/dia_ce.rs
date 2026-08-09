//! dia-PASEF collision energy — the CE the *instrument's own window table* applies to a precursor.
//!
//! # Why this exists next to [`crate::mobility_ce`]
//!
//! [`crate::mobility_ce`] models **dda**-PASEF, where CE is a linear ramp in the mobility scan.
//! dia-PASEF does not work that way. A dia-PASEF method ships a `DiaFrameMsMsWindows` table: each
//! `(window group, scan range)` row carries its OWN `CollisionEnergy`, and the acquisition schedule
//! decides which group fires in which frame. So the CE an ion sees is a property of the **window
//! that transmits it**, read out of the reference `.d` — not of a formula.
//!
//! This is exactly what v1 does. `TimsTofSyntheticsDataHandle::get_collision_energy_dia` builds
//! [`TimsTofCollisionEnergyDIA`] from `DiaFrameMsMsWindows` + `DiaFrameMsMsInfo`, and
//! `ion_map_fn_dia` reads `get_collision_energy(frame, scan)` for every `(frame, scan)` the ion is
//! transmitted in. On the benchmark reference the table spans 20.00-58.12 eV; a flat 25.0 eV (what
//! the flow passes today) or the dda ramp (24.94-43.28 eV on the same ions) are both wrong for DIA.
//!
//! # Nothing here is re-derived
//!
//! | step | owner |
//! |------|-------|
//! | which frames are MS2, and of which window group | [`DiaSchedule`] — the renderer's own replayed schedule |
//! | does `(frame, scan)` transmit this m/z | [`mscore::timstof::quadrupole::TimsTransmissionDIA`] — the renderer's own diagonal gate |
//! | what CE does `(frame, scan)` carry | [`mscore::timstof::collision::TimsTofCollisionEnergyDIA`] — **v1's own lookup, same struct** |
//! | mobility spread of the ion | [`crate::render::gauss_frac`] — the renderer's own Gaussian |
//!
//! There is no second copy of the window table, the logistic, or the CE map anywhere in this file.
//!
//! # The one place a choice had to be made
//!
//! v1 emits one `(ion, CE)` row per transmitting `(frame, scan)`, so an ion straddling the 1 Th
//! overlap between two adjacent window groups gets TWO CEs. v2's fragment predictor holds ONE CE per
//! precursor. Measured on the benchmark reference (`G241217_011_Slot2-2_1_16312.d`) over 485 029
//! precursors: 95.6 % of transmitted precursors see exactly one CE and 4.4 % see two, differing by at
//! most 1.653 eV. So a representative is taken: the CE of the window that carries the LARGEST share
//! of the ion's transmitted signal (mobility Gaussian x transmission probability, summed over the
//! ion's rendered scan support and one acquisition cycle). That is a real value from the instrument's
//! table, and it is the CE most of the ion's fragment signal is actually produced at — not an average
//! that would land between two table entries.

use anyhow::{anyhow, Result};
use mscore::timstof::collision::{TimsTofCollisionEnergy, TimsTofCollisionEnergyDIA};
use mscore::timstof::quadrupole::IonTransmission;

use crate::dia::DiaSchedule;
use crate::render::gauss_frac;

/// v1's inclusion rule: `ion_map_fn_dia` calls `any_transmitted(frame, scan, mz, Some(0.5))`, i.e. a
/// peak counts as transmitted when the quad's soft-edge probability reaches 0.5. Kept identical so the
/// set of `(frame, scan)` pairs considered here is the set v1 considers.
pub const MIN_TRANSMISSION: f64 = 0.5;

/// The dia-PASEF collision-energy source: the reference `.d`'s window table, queried through the
/// renderer's own schedule and mscore's own lookup.
pub struct DiaCollisionEnergy {
    schedule: DiaSchedule,
    energy: TimsTofCollisionEnergyDIA,
    /// The MS2 frames of ONE acquisition cycle. The schedule is periodic, so every distinct
    /// `(window group, scan)` an ion can ever be transmitted in already appears here — iterating a
    /// whole run's frames would just repeat them.
    cycle_ms2_frames: Vec<u32>,
    /// The ion's mobility spread, in scans, and how many sigma the renderer deposits over. Must match
    /// `timsim-render`'s `--sigma-scans` / `--n-sigma` or the CE would be read off a mobility support
    /// the render does not use.
    sigma_scans: f64,
    n_sigma: f64,
    n_scans: u32,
}

/// What a precursor's CE lookup found.
#[derive(Debug, Clone, PartialEq)]
pub struct CeResolution {
    /// The representative CE (eV) — the dominant window's.
    pub collision_energy: f64,
    /// Every distinct CE the ion is transmitted at, ascending. Length > 1 is the straddling case.
    pub distinct: Vec<f64>,
}

impl DiaCollisionEnergy {
    /// Build from a reference DIA `.d`, replaying exactly one acquisition cycle.
    ///
    /// `n_scans` must be the reference's own scan count (the renderer's `Placement::n_scans`), because
    /// the window table's scan ranges are indices into that grid.
    pub fn from_reference(ref_d: &str, n_scans: u32, sigma_scans: f64, n_sigma: f64) -> Result<Self> {
        let schedule = DiaSchedule::one_cycle_from_reference(ref_d, n_scans)?;
        Self::from_schedule(schedule, n_scans, sigma_scans, n_sigma)
    }

    /// Build from an already-replayed schedule (the unit-testable core).
    pub fn from_schedule(
        schedule: DiaSchedule,
        n_scans: u32,
        sigma_scans: f64,
        n_sigma: f64,
    ) -> Result<Self> {
        if n_scans == 0 {
            return Err(anyhow!("n_scans is 0 — there is no mobility grid to read a window off"));
        }
        // `is_finite()` first, so a NaN is rejected rather than silently passing a `>` test.
        if !sigma_scans.is_finite() || sigma_scans <= 0.0 || !n_sigma.is_finite() || n_sigma <= 0.0 {
            return Err(anyhow!(
                "--sigma-scans ({sigma_scans}) and --n-sigma ({n_sigma}) must be positive — an ion \
                 with no mobility spread has no window support"
            ));
        }

        // Our replayed frame -> window group, for exactly the cycle we hold. This is the same pair of
        // vectors `handle.get_collision_energy_dia` passes in v1, just sourced from our schedule
        // instead of the reference's frame list — the CE map itself is mscore's.
        let cycle_ms2_frames: Vec<u32> =
            (1..=schedule.cycle_len).filter(|f| schedule.window_group(*f).is_some()).collect();
        if cycle_ms2_frames.is_empty() {
            return Err(anyhow!("the reference cycle has no MS2 frames — nothing fragments"));
        }
        let frames: Vec<i32> = cycle_ms2_frames.iter().map(|f| *f as i32).collect();
        let frame_groups: Vec<i32> = cycle_ms2_frames
            .iter()
            .map(|f| schedule.window_group(*f).unwrap() as i32)
            .collect();

        let energy = TimsTofCollisionEnergyDIA::new(
            frames,
            frame_groups,
            schedule.windows.iter().map(|w| w.window_group as i32).collect(),
            schedule.windows.iter().map(|w| w.scan_num_begin as i32).collect(),
            schedule.windows.iter().map(|w| w.scan_num_end as i32).collect(),
            schedule.windows.iter().map(|w| w.collision_energy).collect(),
        );

        Ok(DiaCollisionEnergy { schedule, energy, cycle_ms2_frames, sigma_scans, n_sigma, n_scans })
    }

    /// The min/max CE in the reference's window table — the range v2 can possibly emit.
    pub fn table_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for w in &self.schedule.windows {
            lo = lo.min(w.collision_energy);
            hi = hi.max(w.collision_energy);
        }
        (lo, hi)
    }

    /// Number of window-table rows (for the run banner).
    pub fn n_windows(&self) -> usize {
        self.schedule.windows.len()
    }

    /// The mobility scan support the renderer deposits this ion over: `+/- n_sigma * sigma_scans`
    /// around its placed scan, clamped to the grid. Mirrors `timsim-render`'s `scan_window`.
    fn scan_support(&self, scan: u32) -> (u32, u32) {
        let c = scan as f64;
        let h = self.n_sigma * self.sigma_scans;
        let lo = (c - h).max(0.0) as u32;
        let hi = ((c + h) as u32).min(self.n_scans - 1);
        (lo, hi)
    }

    /// Resolve the CE for a precursor placed at mobility `scan` with precursor m/z `mz`.
    ///
    /// Returns `None` when NO window transmits the ion anywhere in its mobility support — a real and
    /// common state in dia-PASEF, where the quadrupole samples a diagonal and roughly half the
    /// `(m/z, mobility)` plane is never isolated. Such an ion deposits no MS2 peaks in the render at
    /// all, so it has no collision energy; the caller decides what to write for it rather than this
    /// function inventing one.
    pub fn resolve(&self, mz: f64, scan: u32) -> Option<CeResolution> {
        let (lo, hi) = self.scan_support(scan);
        let probe = vec![mz];
        // Distinct CEs and the transmitted weight each carries. A dia-PASEF cycle has O(10) groups and
        // O(3) CEs reachable at any one scan, so a linear scan of a tiny vec beats hashing an f64.
        let mut weights: Vec<(f64, f64)> = Vec::new();
        for s in lo..=hi {
            let mw = gauss_frac(s as f64 - 0.5, s as f64 + 0.5, scan as f64, self.sigma_scans);
            if mw <= 0.0 {
                continue;
            }
            for &f in &self.cycle_ms2_frames {
                let p = self.schedule.transmission.apply_transmission(f as i32, s as i32, &probe)[0];
                if p < MIN_TRANSMISSION {
                    continue;
                }
                let ce = self.energy.get_collision_energy(f as i32, s as i32);
                // mscore returns 0.0 for "no window at this (group, scan)". The transmission gate has
                // already established that a window DOES cover it, so a 0.0 here would mean the two
                // mscore maps disagree; skip rather than emit a fabricated 0 eV.
                if ce == 0.0 {
                    continue;
                }
                let w = mw * p;
                match weights.iter_mut().find(|(e, _)| *e == ce) {
                    Some(slot) => slot.1 += w,
                    None => weights.push((ce, w)),
                }
            }
        }
        if weights.is_empty() {
            return None;
        }
        // Dominant window wins; ties break to the lower CE so the result is a pure function of the
        // inputs regardless of iteration order.
        let mut best = weights[0];
        for &(ce, w) in &weights[1..] {
            if w > best.1 || (w == best.1 && ce < best.0) {
                best = (ce, w);
            }
        }
        let mut distinct: Vec<f64> = weights.iter().map(|(e, _)| *e).collect();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(CeResolution { collision_energy: best.0, distinct })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_io::data::meta::{DiaMsMisInfo, DiaMsMsWindow};

    fn win(group: u32, b: u32, e: u32, mz: f64, ce: f64) -> DiaMsMsWindow {
        DiaMsMsWindow {
            window_group: group,
            scan_num_begin: b,
            scan_num_end: e,
            isolation_mz: mz,
            isolation_width: 25.0,
            collision_energy: ce,
        }
    }

    /// Two groups, each with a high-m/z window early in the mobility ramp and a low-m/z window late —
    /// a miniature of the real diagonal, with the same 1 Th group-to-group overlap (windows are 25 Th
    /// wide and centres are 24 Th apart).
    fn diag() -> DiaCollisionEnergy {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 3, window_group: 2 },
            DiaMsMisInfo { frame_id: 5, window_group: 1 },
            DiaMsMisInfo { frame_id: 6, window_group: 2 },
        ];
        let windows = vec![
            win(1, 0, 49, 1000.0, 50.0),
            win(1, 60, 99, 500.0, 20.0),
            win(2, 0, 49, 1024.0, 51.5),
            win(2, 60, 99, 524.0, 22.0),
        ];
        let sched = DiaSchedule::build(&info, windows, 3, 100).unwrap();
        DiaCollisionEnergy::from_schedule(sched, 100, 4.0, 3.0).unwrap()
    }

    /// The plain case: one window transmits, and its table CE comes back — not a ramp value.
    #[test]
    fn reads_the_windows_own_collision_energy() {
        let d = diag();
        let r = d.resolve(1000.0, 25).unwrap();
        assert_eq!(r.distinct, vec![50.0]);
        assert_eq!(r.collision_energy, 50.0);
        // Same m/z, LATE in the mobility ramp: off the diagonal, nothing isolates it.
        assert!(d.resolve(1000.0, 80).is_none());
        // The low-m/z leg of the same group, late in the ramp.
        let r = d.resolve(500.0, 80).unwrap();
        assert_eq!(r.collision_energy, 20.0);
    }

    /// The straddling case the representative choice exists for: 1 Th of overlap between adjacent
    /// groups gives two CEs, and the dominant one is returned.
    #[test]
    fn straddling_two_groups_reports_both_and_picks_the_dominant() {
        let d = diag();
        // 1012.4 sits inside group 1 [987.5, 1012.5] and group 2 [1011.5, 1036.5].
        let r = d.resolve(1012.4, 25).unwrap();
        assert_eq!(r.distinct.len(), 2, "expected the overlap to expose two window CEs");
        assert_eq!(r.distinct, vec![50.0, 51.5]);
        assert!(r.distinct.contains(&r.collision_energy));
        // Deep inside group 1 only.
        assert_eq!(d.resolve(1000.0, 25).unwrap().distinct, vec![50.0]);
        // Deep inside group 2 only.
        assert_eq!(d.resolve(1030.0, 25).unwrap().distinct, vec![51.5]);
    }

    /// The result must not depend on the order the windows happen to be visited in, or a re-run could
    /// silently pick the other CE.
    #[test]
    fn resolution_is_deterministic() {
        let d = diag();
        let a = d.resolve(1012.4, 25).unwrap();
        for _ in 0..20 {
            assert_eq!(d.resolve(1012.4, 25).unwrap(), a);
        }
    }

    /// An ion near the ramp edge must clamp its support instead of running off the grid.
    #[test]
    fn scan_support_clamps_to_the_grid() {
        let d = diag();
        assert_eq!(d.scan_support(0), (0, 12));
        assert_eq!(d.scan_support(99), (87, 99));
        assert!(d.resolve(1000.0, 0).is_some());
    }

    /// A precursor whose m/z is outside every window is unisolatable — and must NOT be handed a
    /// fabricated 0 eV (which is what mscore's map returns for an unknown key).
    #[test]
    fn an_unisolatable_precursor_returns_none_not_zero() {
        let d = diag();
        assert!(d.resolve(1500.0, 25).is_none());
        assert!(d.resolve(100.0, 25).is_none());
    }

    #[test]
    fn table_range_is_the_windows_range() {
        assert_eq!(diag().table_range(), (20.0, 51.5));
    }

    #[test]
    fn rejects_a_degenerate_mobility_spread() {
        let info = vec![
            DiaMsMisInfo { frame_id: 2, window_group: 1 },
            DiaMsMisInfo { frame_id: 4, window_group: 1 },
        ];
        let sched =
            DiaSchedule::build(&info, vec![win(1, 0, 49, 1000.0, 50.0)], 2, 100).unwrap();
        assert!(DiaCollisionEnergy::from_schedule(sched, 100, 0.0, 3.0).is_err());
    }

    /// Integration against a real dia-PASEF `.d`. `DIA_REF=/path/to.d cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn integration_real_reference() {
        let d = std::env::var("DIA_REF").expect("set DIA_REF to a dia-PASEF .d");
        let ce = DiaCollisionEnergy::from_reference(&d, 927, 4.0, 3.0).unwrap();
        let (lo, hi) = ce.table_range();
        eprintln!("windows {} CE {lo}..{hi}", ce.n_windows());
        // Every window's own centre, at the middle of its own scan range, must resolve to its own CE.
        for w in &ce.schedule.windows {
            let s = (w.scan_num_begin + w.scan_num_end) / 2;
            let r = ce.resolve(w.isolation_mz, s).expect("a window centre must transmit");
            assert!(
                r.distinct.contains(&w.collision_energy),
                "group {} scan {s}: got {:?}, table says {}",
                w.window_group, r.distinct, w.collision_energy
            );
        }
    }
}
