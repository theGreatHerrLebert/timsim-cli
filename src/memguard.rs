//! Pre-flight memory admission for the parallel render.
//!
//! The parallel DIA render's peak RSS is essentially linear in the thread count, and when it lands on
//! top of the machine's total RAM the failure is not an OOM kill — it is a `Parquet argument error:
//! External: Data corruption detected` from somewhere deep in the write path, hours in. That is a
//! terrible way to find out you asked for too many threads, so we ask the question BEFORE starting:
//! how much peak RSS will `N` threads need, and how much does this machine/container actually have?
//!
//! This is an **admission heuristic, not a proof of safety**. `MemAvailable` is a kernel snapshot of
//! what could be handed out right now; another process can take it a second later, and our own
//! estimate is a linear fit to a handful of measurements, not a bound. It exists to turn the common
//! "16 threads on a 31 GB box" mistake into a loud, actionable message instead of a four-hour
//! corrupted render.
//!
//! Swap is deliberately NOT counted as capacity. A render that swaps does not fail cleanly; it
//! thrashes for days. `MemAvailable` already excludes swap, and cgroup `memory.max` is a
//! memory-only limit (`memory.swap.max` is a separate knob we intentionally ignore).

use std::path::{Path, PathBuf};

const GIB: u64 = 1024 * 1024 * 1024;

/// Thread-independent part of the render's peak RSS: one apex-chunk's resident ions and their
/// projected peak vectors, the A2 per-frame background map, the arrow readers, and the `TdfWriter`
/// state. See [`PER_THREAD_BYTES`] for the measurements this was fitted from.
pub const BASELINE_BYTES: u64 = 3 * GIB;

/// Per-thread peak-RSS cost of the parallel DIA render.
///
/// Fitted from measured `/usr/bin/time -v` peaks of the SAME binary on the SAME staged full-proteome
/// `severe_R1` render (17646 frames, 5.1e8 MS1 + 2.1e9 MS2 peaks, `--noise-real-data`):
///
/// ```text
///    4 threads (doxytocin, 31 GB) -> ~10.0 GiB peak, 19/19 runs clean
///   16 threads (doxytocin, 31 GB) -> ~30.8 GiB peak, 2 of 4 runs died mid-write
///   16 threads (monster3, 250 GB) -> 30.84 GiB peak (32339952 / 32255264 / 32459536 kB), 3/3 clean
/// ```
///
/// slope = (30.84 - 10.0) GiB / 12 threads = 1.74 GiB/thread, intercept = 3.05 GiB. Rounded to
/// 1.75 GiB/thread and a 3.0 GiB baseline. Each thread carries one sub-range's triples buffer, the
/// two `dedup_and_quantise` hash maps, and the encoded blocks it has produced but not yet handed
/// back for appending.
pub const PER_THREAD_BYTES: u64 = 1_879_048_192; // 1.75 GiB

/// Encoded blocks retained between the parallel render and the ordered append — the render processes
/// sub-ranges in waves, so at most one wave's compressed output is in flight at a time.
pub const RETAINED_OUTPUT_BYTES: u64 = 805_306_368; // 0.75 GiB

/// Safety margin applied on top of the fit, in percent. Covers allocator overhead/fragmentation and
/// the fact that a linear fit through three points is not an upper bound.
pub const SAFETY_MARGIN_PCT: u64 = 15;

/// Estimated peak RSS, in bytes, for a parallel render at `threads` threads.
pub fn estimate_peak_bytes(threads: usize) -> u64 {
    let raw = BASELINE_BYTES
        .saturating_add(PER_THREAD_BYTES.saturating_mul(threads.max(1) as u64))
        .saturating_add(RETAINED_OUTPUT_BYTES);
    raw.saturating_mul(100 + SAFETY_MARGIN_PCT) / 100
}

/// The largest thread count whose estimate fits in `limit`, or 0 if not even one thread fits.
pub fn fit_threads(requested: usize, limit: u64) -> usize {
    let mut n = requested.max(1);
    while n > 0 && estimate_peak_bytes(n) > limit {
        n -= 1;
    }
    n
}

/// What the guard measured the machine to have, and where the number came from.
#[derive(Debug, Clone)]
pub struct Limit {
    /// Usable bytes, swap excluded.
    pub bytes: u64,
    /// Human-readable provenance, e.g. `cgroup v2 /user.slice (memory.max - memory.current)`.
    pub source: String,
    /// Every source we could read, for the message (so a container limit that beats host RAM is visible).
    pub detail: String,
}

/// Parse `MemAvailable` (kB) out of `/proc/meminfo` contents. Returns bytes.
///
/// `MemAvailable` is the kernel's own estimate of what can be allocated without swapping — it already
/// excludes swap and accounts for reclaimable page cache, which is why we prefer it to `MemFree`.
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|l| {
        let rest = l.strip_prefix("MemAvailable:")?;
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        Some(kb * 1024)
    })
}

/// Parse a cgroup v2 `memory.max` / `memory.current` value. `"max"` means unlimited → `None`.
pub fn parse_cgroup_value(s: &str) -> Option<u64> {
    let t = s.trim();
    if t == "max" {
        return None;
    }
    t.parse().ok()
}

/// The cgroup v2 directory chain for this process, leaf LAST, from `/proc/self/cgroup` contents.
///
/// A v2 line is `0::/some/path`. Ancestors matter: a limit set on a parent slice binds us just as
/// hard as one set on our own leaf, so the caller takes the minimum headroom over the whole chain.
pub fn cgroup_chain(proc_self_cgroup: &str, root: &Path) -> Vec<PathBuf> {
    let Some(rel) = proc_self_cgroup
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .map(str::trim)
    else {
        return Vec::new();
    };
    let mut out = vec![root.to_path_buf()];
    let mut cur = root.to_path_buf();
    for seg in rel.split('/').filter(|s| !s.is_empty()) {
        cur = cur.join(seg);
        out.push(cur.clone());
    }
    out
}

/// Tightest cgroup v2 headroom (`memory.max - memory.current`) over this process's cgroup chain.
///
/// `None` means no cgroup in the chain sets a finite `memory.max` (or cgroup v2 is not mounted), in
/// which case the host-wide `MemAvailable` is all we have to go on.
fn cgroup_headroom() -> Option<(u64, String)> {
    let root = Path::new("/sys/fs/cgroup");
    let sc = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut best: Option<(u64, String)> = None;
    for dir in cgroup_chain(&sc, root) {
        let Some(max) = std::fs::read_to_string(dir.join("memory.max"))
            .ok()
            .and_then(|s| parse_cgroup_value(&s))
        else {
            continue; // "max" (unlimited) or unreadable
        };
        let cur = std::fs::read_to_string(dir.join("memory.current"))
            .ok()
            .and_then(|s| parse_cgroup_value(&s))
            .unwrap_or(0);
        let head = max.saturating_sub(cur);
        let label = format!(
            "cgroup v2 {} (memory.max {:.1} GiB - memory.current {:.1} GiB)",
            dir.display(),
            max as f64 / GIB as f64,
            cur as f64 / GIB as f64
        );
        if best.as_ref().map_or(true, |(b, _)| head < *b) {
            best = Some((head, label));
        }
    }
    best
}

/// What this machine (or container) can actually give the render, swap excluded.
///
/// A cgroup limit WINS over host RAM: inside a container `/proc/meminfo` reports the host's memory,
/// which would happily admit a thread count the container will be killed for. We therefore take the
/// MINIMUM of the cgroup headroom and `MemAvailable` — the cgroup binds when it is tighter (the
/// container case), and `MemAvailable` binds when the host itself is already under pressure.
///
/// `None` (neither source readable, e.g. not Linux) means the guard cannot form an opinion and stands
/// down rather than guessing.
pub fn effective_limit() -> Option<Limit> {
    let avail = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .as_deref()
        .and_then(parse_mem_available);
    let cg = cgroup_headroom();
    let mut detail = String::new();
    if let Some(a) = avail {
        detail.push_str(&format!("MemAvailable {:.1} GiB", a as f64 / GIB as f64));
    }
    if let Some((h, ref l)) = cg {
        if !detail.is_empty() {
            detail.push_str("; ");
        }
        detail.push_str(&format!("{} => {:.1} GiB headroom", l, h as f64 / GIB as f64));
    }
    detail.push_str("; swap NOT counted as capacity");
    match (cg, avail) {
        (Some((h, l)), Some(a)) if h <= a => Some(Limit { bytes: h, source: l, detail }),
        (Some(_), Some(a)) => Some(Limit {
            bytes: a,
            source: "/proc/meminfo MemAvailable".into(),
            detail,
        }),
        (Some((h, l)), None) => Some(Limit { bytes: h, source: l, detail }),
        (None, Some(a)) => Some(Limit {
            bytes: a,
            source: "/proc/meminfo MemAvailable".into(),
            detail,
        }),
        (None, None) => None,
    }
}

/// The guard's verdict.
pub enum Admission {
    /// The requested thread count fits (or the guard could not measure the machine).
    Proceed,
    /// Too many threads for this machine — run with `threads` instead. `message` is ready to print.
    Reduce { threads: usize, message: String },
    /// Not even one thread fits. `message` is ready to return as an error.
    Refuse { message: String },
}

/// Decide whether a parallel render at `requested` threads may start on this machine.
pub fn admit(requested: usize) -> Admission {
    let Some(limit) = effective_limit() else {
        return Admission::Proceed; // no readable limit — do not guess
    };
    admit_against(requested, &limit)
}

/// [`admit`] against an explicit [`Limit`] — the testable half.
pub fn admit_against(requested: usize, limit: &Limit) -> Admission {
    let want = estimate_peak_bytes(requested);
    if want <= limit.bytes {
        return Admission::Proceed;
    }
    let g = |b: u64| b as f64 / GIB as f64;
    let model = format!(
        "model: {:.1} GiB baseline + threads x {:.2} GiB + {:.2} GiB retained output, +{}% safety margin",
        g(BASELINE_BYTES),
        g(PER_THREAD_BYTES),
        g(RETAINED_OUTPUT_BYTES),
        SAFETY_MARGIN_PCT
    );
    let head = format!(
        "MEMORY GUARD: a parallel render at {} threads needs ~{:.1} GiB peak RSS, but only {:.1} GiB \
         is usable here.\n    limit source: {}\n    ({})\n    {}",
        requested,
        g(want),
        g(limit.bytes),
        limit.source,
        limit.detail,
        model
    );
    let fits = fit_threads(requested, limit.bytes);
    if fits == 0 {
        Admission::Refuse {
            message: format!(
                "{head}\n    Not even ONE thread fits ({:.1} GiB needed). Free memory, raise the \
                 container limit, or render with --no-parallel (lowest footprint). To start anyway \
                 and accept the risk of a mid-write corruption failure, pass --no-memory-guard.",
                g(estimate_peak_bytes(1))
            ),
        }
    } else {
        Admission::Reduce {
            threads: fits,
            message: format!(
                "{head}\n    -> AUTO-REDUCING to {} threads (~{:.1} GiB estimated peak). Set \
                 RAYON_NUM_THREADS to choose a different value, or pass --no-memory-guard to run at \
                 {} anyway.\n    This is an admission heuristic, not a proof of safety: MemAvailable \
                 is a snapshot and another process may take memory after this check.",
                fits,
                g(estimate_peak_bytes(fits)),
                requested
            ),
        }
    }
}

/// The thread count the render would use if nothing intervened: `RAYON_NUM_THREADS` if set (that is
/// what rayon itself reads), else the machine's parallelism.
pub fn requested_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_parse() {
        let t = "MemTotal:       32819800 kB\nMemFree:         2000 kB\nMemAvailable:   26959520 kB\n";
        assert_eq!(parse_mem_available(t), Some(26959520 * 1024));
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn cgroup_value_parse() {
        assert_eq!(parse_cgroup_value("max\n"), None);
        assert_eq!(parse_cgroup_value("2147483648\n"), Some(2147483648));
        assert_eq!(parse_cgroup_value("garbage"), None);
    }

    #[test]
    fn chain_includes_root_and_every_ancestor() {
        let c = cgroup_chain("0::/user.slice/user-1000.slice/x.scope\n", Path::new("/sys/fs/cgroup"));
        assert_eq!(
            c,
            vec![
                PathBuf::from("/sys/fs/cgroup"),
                PathBuf::from("/sys/fs/cgroup/user.slice"),
                PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice"),
                PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice/x.scope"),
            ]
        );
        assert!(cgroup_chain("1:name=systemd:/foo\n", Path::new("/sys/fs/cgroup")).is_empty());
    }

    #[test]
    fn estimate_is_monotonic_and_matches_the_fit() {
        // 16 threads on the measured render: fit says ~31.75 GiB raw, 36.5 GiB with the margin —
        // comfortably above the 30.84 GiB actually measured, which is the point of the margin.
        let e16 = estimate_peak_bytes(16) as f64 / GIB as f64;
        assert!((36.0..37.5).contains(&e16), "16-thread estimate {e16}");
        let e4 = estimate_peak_bytes(4) as f64 / GIB as f64;
        assert!((12.0..13.0).contains(&e4), "4-thread estimate {e4}");
        assert!(estimate_peak_bytes(4) < estimate_peak_bytes(5));
    }

    #[test]
    fn fits_and_reduces() {
        // A 31 GB box with ~25.7 GiB available must not start 16 threads.
        let limit = Limit {
            bytes: (25.7 * GIB as f64) as u64,
            source: "test".into(),
            detail: "test".into(),
        };
        match admit_against(16, &limit) {
            Admission::Reduce { threads, .. } => {
                assert!(threads < 16 && threads >= 1);
                assert!(estimate_peak_bytes(threads) <= limit.bytes);
                assert!(estimate_peak_bytes(threads + 1) > limit.bytes);
            }
            _ => panic!("expected a reduction at 16 threads on 25.7 GiB"),
        }
        // A 250 GB box admits 16 threads unchanged.
        let big = Limit { bytes: 244 * GIB, source: "test".into(), detail: "test".into() };
        assert!(matches!(admit_against(16, &big), Admission::Proceed));
        // A 4 GiB container cannot run even one thread.
        let tiny = Limit { bytes: 4 * GIB, source: "test".into(), detail: "test".into() };
        assert!(matches!(admit_against(8, &tiny), Admission::Refuse { .. }));
    }

    #[test]
    fn swap_is_never_capacity() {
        // MemAvailable is the only /proc/meminfo field we read, and it excludes swap by construction.
        let t = "MemAvailable:   1024 kB\nSwapFree:      64000000 kB\n";
        assert_eq!(parse_mem_available(t), Some(1024 * 1024));
    }
}
