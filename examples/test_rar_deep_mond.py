#!/usr/bin/env python3
"""
Radial Acceleration Relation (RAR) analysis from SPARC data.

Tests:
  1. RAR plot: log10(g_obs) vs log10(g_bar) for all 175 SPARC galaxies
  2. Theoretical overlays: Khronon (a0=cH0/2pi), McGaugh MOND (a0=1.2e-10), 1:1 line
  3. Scatter analysis in bins, especially deep MOND regime (g_bar < 1e-11)
  4. Deep MOND asymptotic check: g_obs -> sqrt(g_bar * a0)
  5. Best-fit a0 from RAR scatter minimization
  6. Comparison: Khronon a0 = cH0/(2pi) vs best-fit a0

Key finding: If scatter stays tight in deep MOND regime, this strongly
supports modified gravity over dark matter (DM halos would add scatter).

Usage:
    python examples/test_rar_deep_mond.py
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar

# ── Constants ────────────────────────────────────────────────────────────
G_SI = 6.674e-11        # m^3/(kg*s^2)
c_SI = 2.998e8           # m/s
H0_SI = 73e3 / 3.086e22  # 73 km/s/Mpc -> s^-1
Msun = 1.989e30          # kg
kpc_m = 3.086e19         # m per kpc
km_s = 1e3               # m/s per km/s

# MOND acceleration scales
a0_Khronon = c_SI * H0_SI / (2 * np.pi)  # cH0/(2pi) ~ 1.13e-10 m/s^2
a0_McGaugh = 1.20e-10                     # McGaugh (2016) empirical value

# M/L ratios at 3.6 um (Schombert & McGaugh 2014)
ML_DISK = 0.5
ML_BULGE = 0.7

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sparc")


# ── Data Loading ─────────────────────────────────────────────────────────

def parse_mass_models(filepath):
    """Parse MassModels_Lelli2016c.mrt (fixed-width format)."""
    galaxies = {}
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header lines
            if line.startswith(('Title', 'Authors', 'Table', '=', '-',
                                'Byte', 'Note', '\n')):
                continue
            if line.startswith(' ') and len(line.strip()) > 0:
                parts = line.strip().split()
                if len(parts) >= 9:
                    try:
                        float(parts[1])
                    except ValueError:
                        continue
                else:
                    continue
            elif line.startswith(' '):
                continue

            parts = line.strip().split()
            if len(parts) < 9:
                continue

            try:
                name = parts[0]
                D = float(parts[1])
                R = float(parts[2])      # kpc
                Vobs = float(parts[3])   # km/s
                e_Vobs = float(parts[4]) # km/s
                Vgas = float(parts[5])   # km/s
                Vdisk = float(parts[6])  # km/s
                Vbul = float(parts[7])   # km/s

                if name not in galaxies:
                    galaxies[name] = {
                        'D': D,
                        'R': [], 'Vobs': [], 'e_Vobs': [],
                        'Vgas': [], 'Vdisk': [], 'Vbul': [],
                    }
                g = galaxies[name]
                g['R'].append(R)
                g['Vobs'].append(Vobs)
                g['e_Vobs'].append(e_Vobs)
                g['Vgas'].append(Vgas)
                g['Vdisk'].append(Vdisk)
                g['Vbul'].append(Vbul)
            except (ValueError, IndexError):
                continue

    for name in galaxies:
        for key in ['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']:
            galaxies[name][key] = np.array(galaxies[name][key])

    return galaxies


def compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ML_DISK, ml_bulge=ML_BULGE):
    """V^2_bar = V^2_gas + (M/L_disk)*V^2_disk + (M/L_bulge)*V^2_bulge"""
    V2_gas = np.sign(Vgas) * Vgas**2
    V2_disk = ml_disk * np.sign(Vdisk) * Vdisk**2
    V2_bul = ml_bulge * np.sign(Vbul) * Vbul**2
    V2_bar = V2_gas + V2_disk + V2_bul
    return np.sign(V2_bar) * np.sqrt(np.abs(V2_bar))


# ── RAR theoretical curves ──────────────────────────────────────────────

def rar_interpolating(g_bar, a0):
    """
    RAR interpolating function (McGaugh 2016):
      g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))
    """
    x = np.sqrt(g_bar / a0)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-30)
    return g_bar / denom


def deep_mond_asymptotic(g_bar, a0):
    """Deep MOND limit: g_obs = sqrt(g_bar * a0)"""
    return np.sqrt(g_bar * a0)


# ── Main Analysis ────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("  RADIAL ACCELERATION RELATION (RAR) -- DEEP MOND ANALYSIS")
    print("  SPARC: 175 Disk Galaxies (Lelli, McGaugh & Schombert 2016)")
    print("=" * 75)
    print()

    # ── 1. Load data ─────────────────────────────────────────────────────
    mm_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")
    if not os.path.exists(mm_path):
        print(f"ERROR: SPARC data not found at {mm_path}")
        sys.exit(1)

    galaxies = parse_mass_models(mm_path)
    print(f"  Loaded {len(galaxies)} galaxies from SPARC")

    # ── 2. Compute accelerations for every data point ────────────────────
    all_g_obs = []
    all_g_bar = []
    all_e_g_obs = []
    all_galaxy_names = []
    n_points_total = 0
    n_skipped = 0

    for name, g in sorted(galaxies.items()):
        R_kpc = g['R']
        Vobs = g['Vobs']       # km/s
        e_Vobs = g['e_Vobs']   # km/s
        Vgas = g['Vgas']
        Vdisk = g['Vdisk']
        Vbul = g['Vbul']

        Vbar = compute_Vbar(Vgas, Vdisk, Vbul)

        for i in range(len(R_kpc)):
            R_m = R_kpc[i] * kpc_m
            if R_m <= 0 or Vobs[i] <= 0:
                n_skipped += 1
                continue

            Vobs_ms = Vobs[i] * km_s
            e_Vobs_ms = max(e_Vobs[i], 1.0) * km_s  # floor at 1 km/s
            Vbar_ms = np.abs(Vbar[i]) * km_s

            # g_obs = V_obs^2 / R
            g_obs_val = Vobs_ms**2 / R_m
            # g_bar = V_bar^2 / R  (use absolute value for bar)
            g_bar_val = Vbar_ms**2 / R_m

            # Error propagation: delta(g_obs) = 2 * V_obs * delta(V_obs) / R
            e_g_obs_val = 2.0 * Vobs_ms * e_Vobs_ms / R_m

            # Only keep physical points with positive accelerations
            if g_obs_val > 0 and g_bar_val > 0:
                all_g_obs.append(g_obs_val)
                all_g_bar.append(g_bar_val)
                all_e_g_obs.append(e_g_obs_val)
                all_galaxy_names.append(name)
                n_points_total += 1
            else:
                n_skipped += 1

    all_g_obs = np.array(all_g_obs)
    all_g_bar = np.array(all_g_bar)
    all_e_g_obs = np.array(all_e_g_obs)

    print(f"  Total data points: {n_points_total}")
    print(f"  Skipped (non-physical): {n_skipped}")
    print(f"  g_bar range: [{all_g_bar.min():.2e}, {all_g_bar.max():.2e}] m/s^2")
    print(f"  g_obs range: [{all_g_obs.min():.2e}, {all_g_obs.max():.2e}] m/s^2")
    print()

    # ── 3. RAR residuals and scatter ─────────────────────────────────────
    print("-" * 75)
    print("  SECTION 3: RAR SCATTER ANALYSIS")
    print("-" * 75)

    # Log-space residuals: Delta = log10(g_obs) - log10(g_obs_predicted)
    log_g_obs = np.log10(all_g_obs)
    log_g_bar = np.log10(all_g_bar)

    # Predictions
    g_obs_khronon = rar_interpolating(all_g_bar, a0_Khronon)
    g_obs_mcgaugh = rar_interpolating(all_g_bar, a0_McGaugh)

    log_g_obs_khronon = np.log10(g_obs_khronon)
    log_g_obs_mcgaugh = np.log10(g_obs_mcgaugh)

    residuals_khronon = log_g_obs - log_g_obs_khronon
    residuals_mcgaugh = log_g_obs - log_g_obs_mcgaugh

    # Overall scatter (rms in dex)
    scatter_khronon = np.std(residuals_khronon)
    scatter_mcgaugh = np.std(residuals_mcgaugh)
    mean_res_khronon = np.mean(residuals_khronon)
    mean_res_mcgaugh = np.mean(residuals_mcgaugh)

    print(f"\n  Overall RAR scatter (rms in dex):")
    print(f"    Khronon (a0 = {a0_Khronon:.3e}): {scatter_khronon:.4f} dex  "
          f"(mean residual = {mean_res_khronon:+.4f})")
    print(f"    McGaugh (a0 = {a0_McGaugh:.3e}): {scatter_mcgaugh:.4f} dex  "
          f"(mean residual = {mean_res_mcgaugh:+.4f})")
    print(f"    (McGaugh+16 reported: 0.13 dex)")
    print()

    # ── 3b. Scatter in bins of g_bar ─────────────────────────────────────
    print("  Scatter in bins of log10(g_bar):")
    print(f"  {'bin_center':>12s}  {'N_pts':>6s}  {'scatter_K':>10s}  "
          f"{'scatter_M':>10s}  {'median_err':>11s}  {'regime':>15s}")
    print("  " + "-" * 72)

    bin_edges = np.arange(-13.0, -8.5, 0.5)
    bin_centers = []
    bin_scatters_k = []
    bin_scatters_m = []
    bin_counts = []
    bin_median_errs = []

    for j in range(len(bin_edges) - 1):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        mask = (log_g_bar >= lo) & (log_g_bar < hi)
        n_bin = np.sum(mask)
        if n_bin < 5:
            continue

        center = 0.5 * (lo + hi)
        sc_k = np.std(residuals_khronon[mask])
        sc_m = np.std(residuals_mcgaugh[mask])

        # Measurement noise estimate in log space:
        # delta(log10 g_obs) ~ (1/ln10) * e_g_obs / g_obs = (1/ln10) * 2 * e_V/V
        log_err = (1.0 / np.log(10)) * all_e_g_obs[mask] / all_g_obs[mask]
        med_err = np.median(log_err)

        regime = ""
        if 10**center < 1e-11:
            regime = "DEEP MOND"
        elif 10**center < a0_McGaugh:
            regime = "MOND"
        else:
            regime = "Newtonian"

        bin_centers.append(center)
        bin_scatters_k.append(sc_k)
        bin_scatters_m.append(sc_m)
        bin_counts.append(n_bin)
        bin_median_errs.append(med_err)

        print(f"  {center:12.1f}  {n_bin:6d}  {sc_k:10.4f}  "
              f"{sc_m:10.4f}  {med_err:11.4f}  {regime:>15s}")

    bin_centers = np.array(bin_centers)
    bin_scatters_k = np.array(bin_scatters_k)
    bin_scatters_m = np.array(bin_scatters_m)
    bin_counts = np.array(bin_counts)
    bin_median_errs = np.array(bin_median_errs)

    print()

    # ── 4. Deep MOND regime analysis ─────────────────────────────────────
    print("-" * 75)
    print("  SECTION 4: DEEP MOND REGIME (g_bar < 1e-11 m/s^2)")
    print("-" * 75)

    mask_deep = all_g_bar < 1e-11
    n_deep = np.sum(mask_deep)
    print(f"\n  Data points in deep MOND regime: {n_deep}")

    if n_deep > 10:
        scatter_deep_k = np.std(residuals_khronon[mask_deep])
        scatter_deep_m = np.std(residuals_mcgaugh[mask_deep])
        mean_deep_k = np.mean(residuals_khronon[mask_deep])
        mean_deep_m = np.mean(residuals_mcgaugh[mask_deep])

        print(f"  Scatter (Khronon):  {scatter_deep_k:.4f} dex  "
              f"(mean = {mean_deep_k:+.4f})")
        print(f"  Scatter (McGaugh):  {scatter_deep_m:.4f} dex  "
              f"(mean = {mean_deep_m:+.4f})")

        # Compare to overall
        print(f"\n  Deep MOND vs overall scatter:")
        print(f"    Khronon: deep={scatter_deep_k:.4f} vs overall={scatter_khronon:.4f} dex")
        print(f"    McGaugh: deep={scatter_deep_m:.4f} vs overall={scatter_mcgaugh:.4f} dex")

        if scatter_deep_k < scatter_khronon * 1.3:
            print(f"  --> Scatter stays TIGHT in deep MOND regime!")
            print(f"      This is a strong signature of modified gravity (not DM halos)")
        else:
            print(f"  --> Scatter INCREASES somewhat in deep MOND regime")
            print(f"      (Ratio: {scatter_deep_k / scatter_khronon:.2f}x)")
    else:
        print("  Insufficient points for deep MOND analysis")

    print()

    # ── 5. Deep MOND asymptotic verification ─────────────────────────────
    print("-" * 75)
    print("  SECTION 5: DEEP MOND ASYMPTOTIC: g_obs -> sqrt(g_bar * a0)")
    print("-" * 75)

    if n_deep > 10:
        g_bar_deep = all_g_bar[mask_deep]
        g_obs_deep = all_g_obs[mask_deep]

        # In deep MOND: log10(g_obs) = 0.5*log10(g_bar) + 0.5*log10(a0)
        # So slope in log-log should be 0.5
        log_gbar_deep = np.log10(g_bar_deep)
        log_gobs_deep = np.log10(g_obs_deep)

        # Linear fit: log10(g_obs) = slope * log10(g_bar) + intercept
        coeffs = np.polyfit(log_gbar_deep, log_gobs_deep, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Expected: slope = 0.5, intercept = 0.5*log10(a0)
        a0_from_intercept = 10**(2 * intercept)

        print(f"\n  Linear fit in deep MOND (log-log):")
        print(f"    log10(g_obs) = {slope:.4f} * log10(g_bar) + {intercept:.4f}")
        print(f"    Expected slope = 0.5000 (deep MOND)")
        print(f"    Measured slope  = {slope:.4f}")
        print(f"    Deviation from 0.5: {abs(slope - 0.5):.4f}")
        print()
        print(f"    a0 from intercept: {a0_from_intercept:.3e} m/s^2")
        print(f"    a0 Khronon:        {a0_Khronon:.3e} m/s^2")
        print(f"    a0 McGaugh:        {a0_McGaugh:.3e} m/s^2")

        # Check asymptotic ratio: g_obs / sqrt(g_bar * a0)
        ratio_k = g_obs_deep / deep_mond_asymptotic(g_bar_deep, a0_Khronon)
        ratio_m = g_obs_deep / deep_mond_asymptotic(g_bar_deep, a0_McGaugh)

        print(f"\n  g_obs / sqrt(g_bar*a0) in deep MOND:")
        print(f"    Khronon: median = {np.median(ratio_k):.4f}, "
              f"std = {np.std(ratio_k):.4f}")
        print(f"    McGaugh: median = {np.median(ratio_m):.4f}, "
              f"std = {np.std(ratio_m):.4f}")
        print(f"    (should be ~1 if asymptotic holds)")
    else:
        print("  Insufficient points")

    print()

    # ── 6. Best-fit a0 ───────────────────────────────────────────────────
    print("-" * 75)
    print("  SECTION 6: BEST-FIT a0 FROM RAR")
    print("-" * 75)

    def rar_scatter(log_a0):
        """Compute rms scatter of RAR residuals for given a0."""
        a0_val = 10**log_a0
        g_obs_pred = rar_interpolating(all_g_bar, a0_val)
        residuals = log_g_obs - np.log10(g_obs_pred)
        return np.std(residuals)

    # Search over a0 range
    result = minimize_scalar(rar_scatter, bounds=(-10.5, -9.0), method='bounded',
                             options={'xatol': 1e-5})
    a0_best = 10**result.x
    scatter_best = result.fun

    print(f"\n  Best-fit a0 (minimizing RAR scatter):")
    print(f"    a0_best = {a0_best:.4e} m/s^2")
    print(f"    scatter = {scatter_best:.4f} dex")
    print()
    print(f"  Comparison:")
    print(f"    {'Model':>20s}  {'a0 (m/s^2)':>14s}  {'scatter (dex)':>14s}  "
          f"{'ratio to best':>14s}")
    print(f"    {'':->20s}  {'':->14s}  {'':->14s}  {'':->14s}")
    print(f"    {'Khronon cH0/(2pi)':>20s}  {a0_Khronon:14.4e}  "
          f"{scatter_khronon:14.4f}  {a0_Khronon/a0_best:14.4f}")
    print(f"    {'McGaugh (2016)':>20s}  {a0_McGaugh:14.4e}  "
          f"{scatter_mcgaugh:14.4f}  {a0_McGaugh/a0_best:14.4f}")
    print(f"    {'Best-fit':>20s}  {a0_best:14.4e}  "
          f"{scatter_best:14.4f}  {1.0:14.4f}")

    # Percent deviations
    pct_k = 100 * (a0_Khronon - a0_best) / a0_best
    pct_m = 100 * (a0_McGaugh - a0_best) / a0_best
    print(f"\n    Khronon vs best-fit: {pct_k:+.1f}%")
    print(f"    McGaugh vs best-fit: {pct_m:+.1f}%")
    print()

    # ── 7. Scatter trend in deep MOND ────────────────────────────────────
    print("-" * 75)
    print("  SECTION 7: SCATTER TREND vs g_bar (DEEP MOND DIAGNOSTIC)")
    print("-" * 75)
    print()
    print(f"  If scatter INCREASES in deep MOND -> DM halo diversity matters")
    print(f"  If scatter STAYS CONSTANT or DECREASES -> supports modified gravity")
    print()

    # Scatter in finer bins for deep MOND regime
    deep_edges = np.arange(-13.5, -10.0, 0.5)
    print(f"  {'log10(g_bar)':>14s}  {'N':>5s}  {'scatter_K (dex)':>16s}  "
          f"{'noise_est (dex)':>16s}  {'intrinsic (dex)':>16s}")
    print(f"  {'':->14s}  {'':->5s}  {'':->16s}  {'':->16s}  {'':->16s}")

    for j in range(len(deep_edges) - 1):
        lo, hi = deep_edges[j], deep_edges[j + 1]
        mask = (log_g_bar >= lo) & (log_g_bar < hi)
        n_bin = np.sum(mask)
        if n_bin < 3:
            continue

        sc_k = np.std(residuals_khronon[mask])
        log_err = (1.0 / np.log(10)) * all_e_g_obs[mask] / all_g_obs[mask]
        noise = np.sqrt(np.mean(log_err**2))  # rms noise estimate

        # Intrinsic scatter: sigma_int^2 = sigma_total^2 - sigma_noise^2
        intrinsic2 = sc_k**2 - noise**2
        intrinsic = np.sqrt(max(intrinsic2, 0))

        center = 0.5 * (lo + hi)
        print(f"  {center:14.1f}  {n_bin:5d}  {sc_k:16.4f}  "
              f"{noise:16.4f}  {intrinsic:16.4f}")

    print()

    # ── 8. Newtonian regime cross-check ──────────────────────────────────
    print("-" * 75)
    print("  SECTION 8: NEWTONIAN REGIME CROSS-CHECK (g_bar > a0)")
    print("-" * 75)

    mask_newton = all_g_bar > a0_McGaugh
    n_newton = np.sum(mask_newton)
    if n_newton > 10:
        scatter_newton_k = np.std(residuals_khronon[mask_newton])
        ratio_newton = all_g_obs[mask_newton] / all_g_bar[mask_newton]
        print(f"\n  Points in Newtonian regime: {n_newton}")
        print(f"  g_obs/g_bar: median = {np.median(ratio_newton):.4f}, "
              f"std = {np.std(ratio_newton):.4f}")
        print(f"  Scatter (Khronon): {scatter_newton_k:.4f} dex")
        print(f"  (Should approach 1:1 line; g_obs/g_bar -> 1)")
    print()

    # ── 9. How many galaxies contribute to each regime ───────────────────
    print("-" * 75)
    print("  SECTION 9: GALAXY CONTRIBUTIONS BY REGIME")
    print("-" * 75)

    galaxy_names_arr = np.array(all_galaxy_names)

    deep_galaxies = set(galaxy_names_arr[mask_deep])
    mond_mask = (all_g_bar >= 1e-11) & (all_g_bar < a0_McGaugh)
    mond_galaxies = set(galaxy_names_arr[mond_mask])
    newton_galaxies = set(galaxy_names_arr[mask_newton])

    print(f"\n  Deep MOND (g_bar < 1e-11): {len(deep_galaxies)} galaxies, "
          f"{np.sum(mask_deep)} points")
    print(f"  MOND (1e-11 < g_bar < a0): {len(mond_galaxies)} galaxies, "
          f"{np.sum(mond_mask)} points")
    print(f"  Newtonian (g_bar > a0):    {len(newton_galaxies)} galaxies, "
          f"{np.sum(mask_newton)} points")
    print()

    # Top 10 galaxies contributing most points to deep MOND
    if n_deep > 0:
        deep_names = galaxy_names_arr[mask_deep]
        unique_deep, counts_deep = np.unique(deep_names, return_counts=True)
        sort_idx = np.argsort(-counts_deep)
        print(f"  Top 10 deep MOND galaxies (by data points):")
        for k in range(min(10, len(unique_deep))):
            idx = sort_idx[k]
            print(f"    {unique_deep[idx]:15s}: {counts_deep[idx]:3d} points")
    print()

    # ── 10. Summary ──────────────────────────────────────────────────────
    print("=" * 75)
    print("  SUMMARY OF RESULTS")
    print("=" * 75)
    print()
    print(f"  Total galaxies: {len(galaxies)}")
    print(f"  Total data points: {n_points_total}")
    print()
    print(f"  --- RAR scatter (rms in dex) ---")
    print(f"  Khronon (a0 = cH0/2pi = {a0_Khronon:.3e}): {scatter_khronon:.4f} dex")
    print(f"  McGaugh (a0 = 1.20e-10):                    {scatter_mcgaugh:.4f} dex")
    print(f"  Best-fit a0 = {a0_best:.4e}:               {scatter_best:.4f} dex")
    print(f"  (McGaugh+16 reported: 0.13 dex)")
    print()
    print(f"  --- Best-fit a0 ---")
    print(f"  a0_best    = {a0_best:.4e} m/s^2")
    print(f"  a0_Khronon = {a0_Khronon:.4e} m/s^2  ({pct_k:+.1f}% from best-fit)")
    print(f"  a0_McGaugh = {a0_McGaugh:.4e} m/s^2  ({pct_m:+.1f}% from best-fit)")
    print(f"  Khronon prediction: a0 = cH0/(2pi) = {c_SI:.3e} * {H0_SI:.3e} / (2pi)")
    print()

    if n_deep > 10:
        print(f"  --- Deep MOND (g_bar < 1e-11) ---")
        print(f"  N_points = {n_deep} from {len(deep_galaxies)} galaxies")
        print(f"  Scatter = {scatter_deep_k:.4f} dex (vs {scatter_khronon:.4f} overall)")
        print(f"  Deep MOND slope = {slope:.4f} (expected: 0.5000)")
        print(f"  g_obs/sqrt(g_bar*a0_K) median = {np.median(ratio_k):.4f}")
        print()
        if scatter_deep_k < scatter_khronon * 1.3:
            print(f"  ** CONCLUSION: RAR scatter stays TIGHT in deep MOND **")
            print(f"  ** This strongly supports MODIFIED GRAVITY over CDM **")
            print(f"  ** (DM halos would introduce galaxy-dependent scatter) **")
        else:
            print(f"  Note: Scatter increases moderately in deep MOND")
            print(f"  ({scatter_deep_k/scatter_khronon:.2f}x the overall scatter)")

    print()

    # ── Plot (save to file) ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # ---- Panel (a): Full RAR ----
        ax = axes[0, 0]
        ax.scatter(log_g_bar, log_g_obs, s=0.3, alpha=0.15, c='steelblue',
                   rasterized=True, label='SPARC data')

        g_theory = np.logspace(-13, -8, 500)
        ax.plot(np.log10(g_theory), np.log10(rar_interpolating(g_theory, a0_Khronon)),
                'r-', lw=2.0, label=f'Khronon ($a_0 = cH_0/2\\pi = {a0_Khronon:.2e}$)')
        ax.plot(np.log10(g_theory), np.log10(rar_interpolating(g_theory, a0_McGaugh)),
                'g--', lw=2.0, label=f'McGaugh ($a_0 = {a0_McGaugh:.2e}$)')
        ax.plot(np.log10(g_theory), np.log10(g_theory),
                'k:', lw=1.5, label='1:1 (Newtonian)')
        ax.plot(np.log10(g_theory),
                np.log10(deep_mond_asymptotic(g_theory, a0_Khronon)),
                ':', color='orange', lw=1.5, alpha=0.7,
                label=r'Deep MOND: $\sqrt{g_{\rm bar}\,a_0}$')

        ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}$ [m/s$^2$]', fontsize=12)
        ax.set_ylabel(r'$\log_{10}\,g_{\rm obs}$ [m/s$^2$]', fontsize=12)
        ax.set_title('(a) Radial Acceleration Relation', fontsize=13)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(-13, -8.5)
        ax.set_ylim(-13, -8.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # ---- Panel (b): Residuals vs g_bar ----
        ax = axes[0, 1]
        ax.scatter(log_g_bar, residuals_khronon, s=0.3, alpha=0.15,
                   c='steelblue', rasterized=True)
        ax.axhline(0, color='r', lw=1.5)

        # Binned scatter
        if len(bin_centers) > 0:
            ax.errorbar(bin_centers, np.zeros_like(bin_centers),
                        yerr=bin_scatters_k, fmt='ro-', ms=5, lw=1.5,
                        capsize=3, label=r'$\pm 1\sigma$ (Khronon)')
        ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}$ [m/s$^2$]', fontsize=12)
        ax.set_ylabel(r'$\Delta\log_{10}\,g_{\rm obs}$ [dex]', fontsize=12)
        ax.set_title('(b) RAR Residuals (Khronon)', fontsize=13)
        ax.set_xlim(-13, -8.5)
        ax.set_ylim(-0.8, 0.8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- Panel (c): Scatter vs g_bar (key diagnostic) ----
        ax = axes[1, 0]
        if len(bin_centers) > 0:
            ax.plot(bin_centers, bin_scatters_k, 'ro-', ms=6, lw=2,
                    label='Total scatter (Khronon)')
            ax.plot(bin_centers, bin_scatters_m, 'gs--', ms=5, lw=1.5,
                    label='Total scatter (McGaugh)')
            ax.plot(bin_centers, bin_median_errs, 'b^:', ms=5, lw=1.5,
                    label='Median meas. noise')
            ax.axhline(scatter_khronon, color='r', ls=':', alpha=0.5,
                       label=f'Overall scatter = {scatter_khronon:.3f} dex')
            ax.axvline(np.log10(a0_McGaugh), color='gray', ls='--', alpha=0.5,
                       label=r'$a_0$ (McGaugh)')
            ax.axvline(np.log10(1e-11), color='purple', ls='--', alpha=0.5,
                       label=r'Deep MOND ($10^{-11}$)')

        ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}$ [m/s$^2$]', fontsize=12)
        ax.set_ylabel('Scatter [dex]', fontsize=12)
        ax.set_title('(c) RAR Scatter vs $g_{\\rm bar}$ (Key Diagnostic)', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(-13, -8.5)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)

        # ---- Panel (d): Deep MOND zoom ----
        ax = axes[1, 1]
        if n_deep > 10:
            ax.scatter(log_gbar_deep, log_gobs_deep, s=2, alpha=0.4,
                       c='steelblue', label='SPARC (deep MOND)')

            g_deep_th = np.logspace(-13, -10.5, 200)
            ax.plot(np.log10(g_deep_th),
                    np.log10(rar_interpolating(g_deep_th, a0_Khronon)),
                    'r-', lw=2, label='Khronon RAR')
            ax.plot(np.log10(g_deep_th),
                    np.log10(deep_mond_asymptotic(g_deep_th, a0_Khronon)),
                    ':', color='orange', lw=2,
                    label=r'$\sqrt{g_{\rm bar}\,a_0}$')
            ax.plot(np.log10(g_deep_th), np.log10(g_deep_th),
                    'k:', lw=1.5, label='1:1')

            # Show fit line
            x_fit = np.linspace(-13, -10.5, 50)
            ax.plot(x_fit, slope * x_fit + intercept, 'm--', lw=1.5,
                    label=f'Fit: slope = {slope:.3f}')

        ax.set_xlabel(r'$\log_{10}\,g_{\rm bar}$ [m/s$^2$]', fontsize=12)
        ax.set_ylabel(r'$\log_{10}\,g_{\rm obs}$ [m/s$^2$]', fontsize=12)
        ax.set_title('(d) Deep MOND Regime Zoom', fontsize=13)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(-13, -10.5)
        ax.set_ylim(-12, -9.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "rar_deep_mond_analysis.png")
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"  Plot saved: {plot_path}")
        plt.close()

    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    print()
    print("  Done.")
    print("=" * 75)


if __name__ == '__main__':
    main()
