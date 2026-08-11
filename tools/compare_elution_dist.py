#!/usr/bin/env python3
"""Compare v1's and v2's per-peptide elution-shape POPULATIONS.

Why distributional and not per-peptide
--------------------------------------
v1 draws sigma and k in one bulk `np.random.beta(size=n)` call over a dataframe, so a peptide's
shape depends on its ROW POSITION; v2 keys each draw on `blake2b(sequence#salt)`, so it depends on
the peptide. The two therefore cannot agree peptide-by-peptide even when both are correct, and a
paired comparison would report a failure that is not one. What must agree is the POPULATION each
draws from -- which is also the thing a search engine actually sees.

This is the only non-circular check available for the v2 elution work. Every unit test on the v2 side
derives its expected value from the same formula the implementation uses, so it can catch a typo and
nothing else. This reads v1's realized output.

What is compared
----------------
sigma, in SECONDS (not frames -- the two tools run different frame clocks, and seconds is the unit
v1's law is written in), and the dimensionless tail ratio k = 1/(sigma*lambda).

Metrics, pre-registered before looking at any output:
  * two-sample KS statistic D, with its p-value
  * Wasserstein-1 distance, normalised by the v1 population's own spread
  * the quantile table, which is what a reviewer should actually read

D and p are reported for completeness but are NOT the acceptance criterion: at n in the thousands,
KS rejects differences far below anything that changes a chromatogram. The effect sizes are the
criterion, and the thresholds are stated in ACCEPT below rather than chosen after the fact.
"""
import argparse, json, math, sqlite3, sys

import numpy as np

# Pre-registered acceptance thresholds. Stated here, before any run, deliberately: with the author of
# both tools also choosing the metric, a threshold picked after seeing the numbers is not evidence.
ACCEPT = {
    "sigma_mean_rel": 0.02,      # population means within 2%
    "sigma_w1_rel": 0.10,        # Wasserstein-1 within 10% of v1's own IQR
    "k_mean_rel": 0.05,          # k is a heavier-tailed draw; allow more
    "k_w1_rel": 0.15,
    "mobility_mean_rel": 0.05,
    "mobility_w1_rel": 0.15,
}


def v1_mobility(db):
    """v1's realized per-ion mobility width, in 1/K0.

    Read from the `ions` table AS STORED, not from the config's declared target. Those differ: on
    V1-TINY the log says "Standard deviation distribution scaled from mean 0.0085 to 0.0090", and the
    stored population averages 0.00701, because ions are removed AFTER the rescale and the survivors
    are enriched in 2+ (the narrowest relative width). Comparing declared parameters would call this
    a match; comparing realized populations does not.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT charge, inv_mobility_gru_predictor, inv_mobility_gru_predictor_std FROM ions "
            "WHERE inv_mobility_gru_predictor_std IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        con.close()
        return None, None
    con.close()
    z = np.array([r[0] for r in rows], float)
    k0 = np.array([r[1] for r in rows], float)
    sd = np.array([r[2] for r in rows], float)
    ok = np.isfinite(k0) & np.isfinite(sd) & (k0 > 0) & (sd > 0)
    return sd[ok], z[ok]


def v2_mobility(precursors, ccs_parquet, target, reference):
    """v2's per-ion width in 1/K0: k0 * (ccs_std/ccs) * (target/reference).

    Mason-Schamp is linear through the origin, so the constant cancels and this needs no gas mass,
    temperature or charge -- they are inside k0. Deliberately compared in 1/K0 rather than SCANS: the
    scan conversion depends on the reference .d's calibration ramp, and folding that in would compare
    two tools AND two instrument geometries at once.
    """
    import pyarrow.parquet as pq

    pc = pq.read_table(precursors).to_pandas()
    cc = pq.read_table(ccs_parquet).to_pandas()
    d = pc.merge(cc, on="precursor_id")
    d = d[np.isfinite(d.ccs) & (d.ccs > 0) & np.isfinite(d.ccs_std) & (d.ccs_std > 0)]
    SC, MG, T = 18509.8632163405, 28.013, 31.85 + 273.15
    rm = (d.mz * d.charge * MG) / (d.mz * d.charge + MG)
    k0 = (np.sqrt(rm * T) * d.ccs) / (SC * d.charge)
    return np.asarray(k0 * (d.ccs_std / d.ccs) * (target / reference)), np.asarray(d.charge, float)


def v1_shapes(db):
    """v1: sigma in seconds, lambda = 1/(k*sigma) -> k = 1/(sigma*lambda), both per peptide."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT rt_sigma, rt_lambda FROM peptides WHERE rt_sigma IS NOT NULL AND rt_lambda IS NOT NULL"
    ).fetchall()
    grad = con.execute("SELECT MAX(time) - MIN(time) FROM frames").fetchone()[0]
    nfr = con.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    con.close()
    s = np.array([r[0] for r in rows], float)
    lam = np.array([r[1] for r in rows], float)
    ok = np.isfinite(s) & np.isfinite(lam) & (s > 0) & (lam > 0)
    return s[ok], 1.0 / (s[ok] * lam[ok]), grad, nfr


def v2_shapes(rt_parquet, band, k_upper):
    """v2: the stored unit draws mapped through the run's own band -- the reconstruction the
    provenance descriptor exists to make possible, so running this also exercises that claim."""
    import pyarrow.parquet as pq

    t = pq.read_table(rt_parquet)
    cols = set(t.column_names)
    for need in ("rt_sigma_hat", "rt_k_hat"):
        if need not in cols:
            sys.exit(f"{rt_parquet} has no `{need}` -- it predates the per-peptide draws; re-run timsim-rt")
    sh = np.asarray(t["rt_sigma_hat"], float)
    kh = np.asarray(t["rt_k_hat"], float)
    lo, hi = band
    return lo + sh * (hi - lo), kh * k_upper


def band_from_gradient(g):
    """v1's `calculate_rt_defaults` -- affine in the gradient, with a dominant intercept."""
    mid = g / 3600.0 * 0.75 + 1.125
    return mid * 0.75, mid * 1.25


def ks(a, b):
    """Two-sample KS statistic and p-value, without scipy."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, "right") / a.size
    cdf_b = np.searchsorted(b, allv, "right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return d, min(1.0, max(0.0, p))


def w1(a, b):
    """Wasserstein-1 = integral_0^1 |Q_a(u) - Q_b(u)| du, by trapezoid on a uniform grid.

    `mean` over an INCLUSIVE grid is not that integral -- it over-weights the endpoints by half a
    cell each. Negligible at 2001 points, but this is an acceptance measure, so it should be the
    quantity it claims to be rather than one that is usually close to it.
    """
    q = np.linspace(0, 1, 2001)
    # `trapezoid` is NumPy >= 2.0; `trapz` is the pre-2.0 spelling and this box has 1.x.
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.abs(np.quantile(a, q) - np.quantile(b, q)), q))


def report(name, v1, v2, mean_tol, w1_tol):
    d, p = ks(v1, v2)
    iqr = float(np.subtract(*np.percentile(v1, [75, 25]))) or 1.0
    w = w1(v1, v2)
    dmean = abs(v2.mean() - v1.mean()) / (abs(v1.mean()) or 1.0)
    ok = dmean <= mean_tol and w / iqr <= w1_tol
    print(f"\n{'=' * 74}\n{name}   [{'PASS' if ok else 'FAIL'}]\n{'=' * 74}")
    print(f"  {'':10} {'v1':>14} {'v2':>14}   {'delta':>12}")
    print(f"  {'n':10} {v1.size:>14d} {v2.size:>14d}")
    for lbl, f in (("mean", np.mean), ("sd", np.std), ("min", np.min), ("max", np.max)):
        print(f"  {lbl:10} {f(v1):>14.5f} {f(v2):>14.5f}   {f(v2) - f(v1):>+12.5f}")
    print(f"  {'-' * 66}")
    for qq in (1, 5, 25, 50, 75, 95, 99):
        a, b = np.percentile(v1, qq), np.percentile(v2, qq)
        print(f"  p{qq:<9d} {a:>14.5f} {b:>14.5f}   {b - a:>+12.5f}")
    print(f"  {'-' * 66}")
    print(f"  mean shift      {dmean * 100:8.3f}%   (accept <= {mean_tol * 100:.1f}%)")
    print(f"  Wasserstein-1   {w:8.5f}  = {w / iqr * 100:6.2f}% of v1 IQR   (accept <= {w1_tol * 100:.0f}%)")
    print(f"  KS D            {d:8.5f}   p = {p:.3g}   [reported, not a criterion at this n]")
    return ok, {"n_v1": int(v1.size), "n_v2": int(v2.size), "mean_v1": float(v1.mean()),
                "mean_v2": float(v2.mean()), "mean_rel": dmean, "w1": w, "w1_rel_iqr": w / iqr,
                "ks_d": d, "ks_p": p, "pass": bool(ok)}


def conforms_to_declared_law(k2, k_upper):
    """Does v2's k match the Beta(1,20)*k_upper it CLAIMS to draw from? One-sample KS.

    This exists because the v1 comparison alone cannot answer it. If v1's population is censored
    (see `diagnose_v1_truncation`), then "v2 differs from v1" is uninformative about v2 -- and a
    reviewer who stops there can be talked into excusing a genuine v2 error, e.g. `k_hat * 8`
    instead of `k_hat * 10`, on the grounds that v1 is censored anyway. So v2 is held to its own
    declaration, with no reference to v1 at all.

    Beta(1,b) has the closed-form CDF 1-(1-x)^b, so this needs no scipy.
    """
    x = np.sort(np.asarray(k2, float)) / k_upper
    if x.size == 0 or x[0] < 0 or x[-1] > 1:
        return {"pass": False, "why": "values outside [0, k_upper]", "ks_d": float("nan")}
    n = x.size
    cdf = 1.0 - (1.0 - x) ** 20
    d = float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(0, n) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    p = min(1.0, max(0.0, p))
    return {"pass": bool(p > 0.01), "ks_d": d, "ks_p": p,
            "mean_observed": float(np.mean(k2)), "mean_declared": k_upper / 21.0}


def diagnose_v1_truncation(k1):
    """Detect v1's small-k peptide loss, and say so rather than blaming v2.

    v1 and v2 draw `k` from the same Beta(1,20)*10. But for small `k`, v1's `lambda = 1/(k*sigma)`
    explodes, and its mode->mu back-solve carries a `-sigma^2*lambda` term that pushes the apex off
    the FRONT of the gradient; the peptide then elutes in no frame and v1 DELETES it
    ("Removing N peptides that do not elute in any frame"). Its surviving `k` population is therefore
    the draw conditioned on `k > floor`, which has a higher mean than the draw itself.

    So a `k` mismatch here is ambiguous between "v2 is wrong" and "v1 lost 15% of its peptides", and
    a comparison that cannot tell those apart is not worth running. Beta(1,20) gives
    `E[X | X > a] = a + (1-a)/21`, so the hypothesis is checkable rather than a story: if v1's
    observed mean matches its own observed floor under that identity, the shift is survivorship.
    """
    a = float(np.min(k1)) / 10.0
    if a <= 1e-6:
        return None
    predicted = (a + (1 - a) / 21.0) * 10.0
    observed = float(np.mean(k1))
    lost = 1 - (1 - a) ** 20
    agrees = abs(predicted - observed) / observed < 0.05
    return {"floor": a * 10, "predicted_conditional_mean": predicted, "observed_mean": observed,
            "implied_fraction_lost": lost, "explains_shift": bool(agrees)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True, help="v1 synthetic_data.db")
    ap.add_argument("--v2-rt", required=True, help="v2 peptide_rt.parquet")
    ap.add_argument("--gradient", type=float, default=None,
                    help="gradient seconds for the v2 band; default = v1's own span, so the two are "
                         "compared under the SAME gradient (the band is gradient-dependent)")
    ap.add_argument("--k-upper", type=float, default=10.0)
    ap.add_argument("--v2-precursors", default=None, help="v2 precursors.parquet (enables the mobility comparison)")
    ap.add_argument("--v2-ccs", default=None, help="v2 precursor_ccs.parquet")
    ap.add_argument("--mobility-target", type=float, default=0.009)
    ap.add_argument("--mobility-reference", type=float, default=0.009197)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    s1, k1, v1_grad, v1_frames = v1_shapes(a.v1)
    grad = a.gradient if a.gradient is not None else v1_grad
    band = band_from_gradient(grad)
    s2, k2 = v2_shapes(a.v2_rt, band, a.k_upper)

    print(f"v1: {a.v1}\n    {v1_frames} frames, span {v1_grad:.1f} s")
    print(f"v2: {a.v2_rt}")
    print(f"comparison gradient {grad:.1f} s -> sigma band [{band[0]:.5f}, {band[1]:.5f}] s")
    if a.gradient is None:
        print("    (v1's own span; NOTE v1 feeds its CONFIG's declared gradient_length to the sigma\n"
              "     formula, not the acquisition span, so its realized band may sit a hair off this)")

    ok_s, js = report("sigma (seconds)", s1, s2, ACCEPT["sigma_mean_rel"], ACCEPT["sigma_w1_rel"])
    ok_k, jk = report("k (tail / sigma, dimensionless)", k1, k2, ACCEPT["k_mean_rel"], ACCEPT["k_w1_rel"])

    trunc = diagnose_v1_truncation(k1)
    if trunc:
        jk["v1_truncation"] = trunc
        print(f"\n  v1 SURVIVORSHIP CHECK")
        print(f"    v1's k floor                       {trunc['floor']:.5f}")
        print(f"    E[k | k > floor] for Beta(1,20)*10 {trunc['predicted_conditional_mean']:.5f}")
        print(f"    v1's observed mean                 {trunc['observed_mean']:.5f}")
        print(f"    implied peptides lost by v1        {trunc['implied_fraction_lost']*100:.1f}%")
        if trunc["explains_shift"]:
            print("    => CONSISTENT with v1 deleting its small-k peptides (they elute in no frame).")
            print("       NOTE this is a hypothesis fitted to v1's own order statistic, not a")
            print("       measurement of its cutoff, and v1's removal may depend on sigma as well as")
            print("       k. It EXPLAINS the divergence; it does not license ignoring it.")
        else:
            print("    => does NOT explain the shift; treat the k mismatch as a real divergence.")

    # ---- mobility, if both sides can supply it ----
    if a.v2_precursors:
        m1, z1 = v1_mobility(a.v1)
        if m1 is None:
            print("\n(v1 db has no ions.inv_mobility_gru_predictor_std -- skipping mobility)")
        else:
            m2, z2 = v2_mobility(a.v2_precursors, a.v2_ccs, a.mobility_target, a.mobility_reference)
            ok_m, jm = report("mobility sigma (1/K0)", m1, m2,
                              ACCEPT["mobility_mean_rel"], ACCEPT["mobility_w1_rel"])
            print("\n  BY CHARGE (v1 realized vs v2) -- one global gain does not equalise these")
            print(f"    {'z':>3} {'n v1':>7} {'v1 mean':>10} {'n v2':>7} {'v2 mean':>10} {'delta':>9}")
            for z in sorted(set(np.unique(z1)) | set(np.unique(z2))):
                a1, a2 = m1[z1 == z], m2[z2 == z]
                if a1.size and a2.size:
                    print(f"    {int(z):>3} {a1.size:>7d} {a1.mean():>10.6f} {a2.size:>7d} "
                          f"{a2.mean():>10.6f} {a2.mean() - a1.mean():>+9.6f}")
            # Direct standardisation: v2's per-charge means reweighted to v1's charge MIX. If the
            # marginal gap is confounding rather than a width error, this collapses it -- and if it
            # does, reading the marginal alone would have condemned a model that is close to right.
            w = np.array([np.mean(z1 == z) for z in sorted(set(np.unique(z1)))])
            mu2 = np.array([m2[z2 == z].mean() if (z2 == z).any() else np.nan
                            for z in sorted(set(np.unique(z1)))])
            if not np.isnan(mu2).any():
                std2 = float((w * mu2).sum())
                raw = abs(m2.mean() - m1.mean()) / m1.mean()
                adj = abs(std2 - m1.mean()) / m1.mean()
                print(f"\n  CHARGE-MIX STANDARDISATION")
                print(f"    v2 mean, v2's own charge mix   {m2.mean():.6f}   ({raw*100:5.1f}% from v1)")
                print(f"    v2 mean, reweighted to v1's mix {std2:.6f}   ({adj*100:5.1f}% from v1)")
                print(f"    v1 realized mean                {m1.mean():.6f}")
                if adj < raw / 2:
                    print("    => most of the marginal gap is CHARGE MIX, not the width model. The two")
                    print("       tools populate charge states differently (v2 site-specific vs v1")
                    print("       binomial + min_charge_contrib), so comparing marginals alone would")
                    print("       have condemned a width model that is close to correct.")
            print(f"\n  v2 targets {a.mobility_target} (v1's DECLARED inverse_mobility_std_mean).")
            print(f"  v1's REALIZED population mean is {m1.mean():.6f} -- v1 rescales to its target and")
            print("  then removes ions, so its declared parameter is not what it renders. Calibrating")
            print("  v2 to the declared value is a CHOICE; it is stated here rather than hidden in a")
            print("  constant, because targeting v1's realized value instead would import that artifact.")
            jm["v1_realized_mean"] = float(m1.mean())
            jm["v2_target"] = a.mobility_target

    # v2 against its OWN declaration -- the check that does not depend on v1 being uncensored.
    conf = conforms_to_declared_law(k2, a.k_upper)
    print(f"\n{'=' * 74}\nv2 vs its DECLARED law: Beta(1,20)*{a.k_upper}   "
          f"[{'PASS' if conf['pass'] else 'FAIL'}]\n{'=' * 74}")
    print(f"  one-sample KS D {conf['ks_d']:8.5f}   p = {conf.get('ks_p', float('nan')):.3g}")
    print(f"  mean observed   {conf.get('mean_observed', float('nan')):8.5f}   "
          f"declared {conf.get('mean_declared', float('nan')):.5f}")
    print("  (independent of v1. A censored v1 makes 'v2 != v1' uninformative about v2, so v2 is")
    print("   held to its own declaration -- otherwise a real v2 error could be excused as v1 noise.)")
    jk["v2_conforms_to_declared_law"] = conf

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"gradient_seconds": grad, "band_seconds": list(band), "sigma": js, "k": jk,
                       "v2_conforms": conf["pass"], "parity": bool(ok_s and ok_k)}, f, indent=2)

    # THREE verdicts, deliberately not collapsed into one. An earlier version flipped the parity
    # result to PASS once the survivorship story fit, which turns an explanation into a rubber stamp:
    # any v2 error would be excused whenever v1 happened to be censored. What v1 does and whether v2
    # is correct are different questions and are reported as such.
    print(f"\n{'=' * 74}")
    print(f"  v2 conforms to its declared law : {'PASS' if conf['pass'] else 'FAIL'}")
    print(f"  v1/v2 population parity         : {'PASS' if ok_s and ok_k else 'DIVERGENT'}")
    if conf["pass"] and not (ok_s and ok_k):
        print("  => v2 is correct BY ITS OWN DECLARATION and the populations still differ, so the")
        print("     divergence is in what v1 renders, not in what v2 draws. The benchmark must")
        print("     declare that the two tools do not render the same peptide set.")
    print(f"{'=' * 74}")
    # Non-zero unless v2 conforms AND parity holds: a divergence stays visible to CI either way.
    return 0 if (conf["pass"] and ok_s and ok_k) else 1


if __name__ == "__main__":
    sys.exit(main())
