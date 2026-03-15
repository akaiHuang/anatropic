#!/usr/bin/env python3
"""
Systematic Residual Analysis for Khronon Rotation Curve Fits.

Analyzes whether Khronon (RAR with a₀ = cH₀/2π) residuals are random or systematic
by grouping them across:
  (a) Galaxy morphological type
  (b) Galaxy luminosity (3 bins)
  (c) Surface brightness (HSB vs LSB)
  (d) Radius bin (inner / middle / outer)
  (e) Acceleration regime (Newtonian / transition / deep-MOND)

Key diagnostic: random residuals → theory correct; systematic → theory biased.

Usage:
    python examples/test_residual_systematics.py
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

# ── Constants (same as build_rotation_curves.py) ──────────────────────────
G_SI = 6.674e-11
c_SI = 2.998e8
H0_SI = 73e3 / 3.086e22  # 73 km/s/Mpc → s⁻¹
Msun = 1.989e30
kpc_m = 3.086e19
km_s = 1e3

a0_Khronon = c_SI * H0_SI / (2 * np.pi)  # ≈ 1.13e-10 m/s²

ML_DISK = 0.5
ML_BULGE = 0.7

rho_crit = 3 * H0_SI**2 / (8 * np.pi * G_SI)
DELTA_VIR = 200

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sparc")


# ── Parse SPARC data ──────────────────────────────────────────────────────

def parse_mass_models(filepath):
    """Parse MassModels_Lelli2016c.mrt (fixed-width format)."""
    galaxies = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('Title', 'Authors', 'Table', '=', '-',
                                'Byte', 'Note', '\n')):
                if line.startswith(' ') and len(line.strip()) > 0:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        try:
                            float(parts[1])
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue

            parts = line.strip().split()
            if len(parts) < 9:
                continue

            try:
                name = parts[0]
                D = float(parts[1])
                R = float(parts[2])
                Vobs = float(parts[3])
                e_Vobs = float(parts[4])
                Vgas = float(parts[5])
                Vdisk = float(parts[6])
                Vbul = float(parts[7])
                SBdisk = float(parts[8])
                SBbul = float(parts[9]) if len(parts) > 9 else 0.0

                if name not in galaxies:
                    galaxies[name] = {
                        'D': D,
                        'R': [], 'Vobs': [], 'e_Vobs': [],
                        'Vgas': [], 'Vdisk': [], 'Vbul': [],
                        'SBdisk': [], 'SBbul': [],
                    }
                g = galaxies[name]
                g['R'].append(R)
                g['Vobs'].append(Vobs)
                g['e_Vobs'].append(e_Vobs)
                g['Vgas'].append(Vgas)
                g['Vdisk'].append(Vdisk)
                g['Vbul'].append(Vbul)
                g['SBdisk'].append(SBdisk)
                g['SBbul'].append(SBbul)
            except (ValueError, IndexError):
                continue

    for name in galaxies:
        for key in ['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']:
            galaxies[name][key] = np.array(galaxies[name][key])

    return galaxies


def parse_galaxy_properties(filepath):
    """Parse SPARC_Lelli2016c.mrt for galaxy metadata."""
    props = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('Title', 'Authors', 'Table', '=', '-',
                                'Byte', 'Note', '\n', ' ')):
                if line.startswith(' ') and len(line.strip()) > 0:
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        try:
                            float(parts[1])
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue

            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                name = parts[0]
                T = int(parts[1])          # Hubble type
                D = float(parts[2])        # Mpc
                Inc = float(parts[5])      # deg
                L36 = float(parts[7])      # 10^9 L_sun
                Reff = float(parts[9])     # kpc
                SBeff = float(parts[10])   # solLum/pc^2
                Rdisk = float(parts[11])   # kpc
                SBdisk = float(parts[12])  # solLum/pc^2 (central)
                Vflat = float(parts[15]) if len(parts) > 15 else 0.0
                Q = int(parts[17]) if len(parts) > 17 else 2

                hubble_names = {
                    0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc',
                    5: 'Sc', 6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm',
                    10: 'Im', 11: 'BCD'
                }

                props[name] = {
                    'type': hubble_names.get(T, f'T{T}'),
                    'T': T,
                    'D_Mpc': D,
                    'Inc_deg': Inc,
                    'L36_1e9Lsun': L36,
                    'SBeff': SBeff,
                    'SBdisk_central': SBdisk,
                    'Vflat_kms': Vflat,
                    'quality': Q,
                }
            except (ValueError, IndexError):
                continue

    return props


# ── Physics ───────────────────────────────────────────────────────────────

def compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ML_DISK, ml_bulge=ML_BULGE):
    V2_gas = np.sign(Vgas) * Vgas**2
    V2_disk = ml_disk * np.sign(Vdisk) * Vdisk**2
    V2_bul = ml_bulge * np.sign(Vbul) * Vbul**2
    V2_bar = V2_gas + V2_disk + V2_bul
    return np.sign(V2_bar) * np.sqrt(np.abs(V2_bar))


def compute_V_khronon(R_kpc, Vbar_kms, a0=a0_Khronon):
    R_m = R_kpc * kpc_m
    Vbar_ms = Vbar_kms * km_s
    g_bar = Vbar_ms**2 / R_m
    x = np.sqrt(g_bar / a0)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-30)
    g_obs = g_bar / denom
    V_total_ms = np.sqrt(g_obs * R_m)
    return V_total_ms / km_s, g_bar


def fit_khronon_ml(R_kpc, Vobs_kms, e_Vobs_kms, Vgas, Vdisk, Vbul, a0=a0_Khronon):
    """Fit Khronon RAR with M/L_disk as single free parameter."""
    def chi2(params):
        ml_d = params[0]
        if ml_d < 0.1 or ml_d > 1.5:
            return 1e10
        ml_b = 1.4 * ml_d
        Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
        V_k, _ = compute_V_khronon(R_kpc, np.abs(Vbar), a0=a0)
        e_safe = np.maximum(e_Vobs_kms, 1.0)
        return np.sum(((Vobs_kms - V_k) / e_safe)**2)

    best = None
    best_chi2 = 1e20
    for ml0 in [0.3, 0.5, 0.7, 0.9]:
        try:
            res = minimize(chi2, [ml0], method='Nelder-Mead',
                           options={'maxiter': 500})
            if res.fun < best_chi2:
                best_chi2 = res.fun
                best = res
        except Exception:
            continue

    if best is None:
        ml_d = 0.5
    else:
        ml_d = np.clip(best.x[0], 0.1, 1.5)

    ml_b = 1.4 * ml_d
    Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
    V_k, g_bar = compute_V_khronon(R_kpc, np.abs(Vbar), a0=a0)
    e_safe = np.maximum(e_Vobs_kms, 1.0)
    chi2_val = np.sum(((Vobs_kms - V_k) / e_safe)**2)

    return ml_d, chi2_val, V_k, np.abs(Vbar), g_bar


def nfw_velocity(R_kpc, log10_M200, log10_c):
    M200 = 10**log10_M200 * Msun
    c = 10**log10_c
    r200 = (3 * M200 / (4 * np.pi * DELTA_VIR * rho_crit))**(1./3.)
    R_m = R_kpc * kpc_m
    x = c * R_m / r200
    fx = np.log(1 + x) - x / (1 + x)
    fc = np.log(1 + c) - c / (1 + c)
    V2_nfw = G_SI * M200 * fx / (R_m * fc)
    V2_nfw = np.maximum(V2_nfw, 0)
    return np.sqrt(V2_nfw) / km_s


def fit_nfw(R_kpc, Vobs_kms, e_Vobs_kms, Vgas, Vdisk, Vbul):
    """Fit NFW halo + free M/L (3 free params)."""
    def chi2(params):
        ml_d, log10_M200, log10_c = params
        if ml_d < 0.1 or ml_d > 1.5:
            return 1e10
        if log10_c < -0.5 or log10_c > 2.5:
            return 1e10
        if log10_M200 < 6 or log10_M200 > 16:
            return 1e10
        ml_b = 1.4 * ml_d
        Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
        V_nfw = nfw_velocity(R_kpc, log10_M200, log10_c)
        V2_total = Vbar**2 + V_nfw**2
        V_total = np.sqrt(np.maximum(V2_total, 0))
        residuals = (Vobs_kms - V_total) / np.maximum(e_Vobs_kms, 1.0)
        return np.sum(residuals**2)

    best_result = None
    best_chi2 = 1e20
    for ml0 in [0.3, 0.5, 0.7]:
        for lM in [10, 11, 12, 13]:
            for lc in [0.5, 1.0, 1.5]:
                try:
                    res = minimize(chi2, [ml0, lM, lc], method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 0.01,
                                            'fatol': 0.1})
                    if res.fun < best_chi2:
                        best_chi2 = res.fun
                        best_result = res
                except Exception:
                    continue

    if best_result is None:
        Vbar = compute_Vbar(Vgas, Vdisk, Vbul)
        return 0.5, 11.0, 1.0, 1e10, np.abs(Vbar), np.abs(Vbar)

    ml_d, log10_M200, log10_c = best_result.x
    ml_d = np.clip(ml_d, 0.1, 1.5)
    ml_b = 1.4 * ml_d
    Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
    V_nfw = nfw_velocity(R_kpc, log10_M200, log10_c)
    V_total = np.sqrt(Vbar**2 + V_nfw**2)

    return ml_d, log10_M200, log10_c, best_chi2, V_total, np.abs(Vbar)


# ── Helpers ───────────────────────────────────────────────────────────────

def compute_group_stats(residuals):
    """Compute mean, std, RMS, and count for a group of residuals."""
    if len(residuals) == 0:
        return {'mean': np.nan, 'std': np.nan, 'rms': np.nan, 'median': np.nan,
                'n': 0, 'mean_err': np.nan}
    arr = np.array(residuals)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    rms = np.sqrt(np.mean(arr**2))
    median = np.median(arr)
    mean_err = std / np.sqrt(n) if n > 0 else np.nan
    return {'mean': mean, 'std': std, 'rms': rms, 'median': median,
            'n': n, 'mean_err': mean_err}


def significance_flag(mean, mean_err):
    """Flag if mean residual deviates from zero significantly."""
    if np.isnan(mean) or np.isnan(mean_err) or mean_err == 0:
        return "  "
    sigma = abs(mean / mean_err)
    if sigma > 3:
        return "***"
    elif sigma > 2:
        return "** "
    elif sigma > 1:
        return "*  "
    else:
        return "   "


def print_table(title, groups_khronon, groups_nfw, group_order=None):
    """Print a formatted comparison table."""
    if group_order is None:
        group_order = sorted(groups_khronon.keys(), key=str)

    print()
    print("=" * 110)
    print(f"  {title}")
    print("=" * 110)
    print(f"  {'Group':<16s}  {'N':>5s}  |  {'<res>_K':>8s} {'err_K':>7s} {'RMS_K':>7s} {'sig':>3s}  |"
          f"  {'<res>_N':>8s} {'err_N':>7s} {'RMS_N':>7s} {'sig':>3s}  |  {'K better?':>10s}")
    print("-" * 110)

    for grp in group_order:
        sk = compute_group_stats(groups_khronon.get(grp, []))
        sn = compute_group_stats(groups_nfw.get(grp, []))
        sig_k = significance_flag(sk['mean'], sk['mean_err'])
        sig_n = significance_flag(sn['mean'], sn['mean_err'])

        # Determine which has lower RMS
        if not np.isnan(sk['rms']) and not np.isnan(sn['rms']):
            comparison = "YES" if sk['rms'] <= sn['rms'] else "no"
        else:
            comparison = "---"

        print(f"  {str(grp):<16s}  {sk['n']:5d}  |  {sk['mean']:+8.4f} {sk['mean_err']:7.4f} {sk['rms']:7.4f} {sig_k:>3s}  |"
              f"  {sn['mean']:+8.4f} {sn['mean_err']:7.4f} {sn['rms']:7.4f} {sig_n:>3s}  |  {comparison:>10s}")

    # Overall
    all_k = []
    all_n = []
    for grp in group_order:
        all_k.extend(groups_khronon.get(grp, []))
        all_n.extend(groups_nfw.get(grp, []))
    sk = compute_group_stats(all_k)
    sn = compute_group_stats(all_n)
    sig_k = significance_flag(sk['mean'], sk['mean_err'])
    sig_n = significance_flag(sn['mean'], sn['mean_err'])
    comparison = "YES" if sk['rms'] <= sn['rms'] else "no"
    print("-" * 110)
    print(f"  {'TOTAL':<16s}  {sk['n']:5d}  |  {sk['mean']:+8.4f} {sk['mean_err']:7.4f} {sk['rms']:7.4f} {sig_k:>3s}  |"
          f"  {sn['mean']:+8.4f} {sn['mean_err']:7.4f} {sn['rms']:7.4f} {sig_n:>3s}  |  {comparison:>10s}")
    print()
    print("  Significance: *** = >3σ from zero,  ** = >2σ,  * = >1σ")
    print("  <res> = mean fractional residual (V_obs - V_model)/V_obs")
    print("  Positive <res> → model UNDERPREDICTS; Negative → OVERPREDICTS")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 110)
    print("  SYSTEMATIC RESIDUAL ANALYSIS: Khronon vs NFW Rotation Curve Fits")
    print("  Khronon: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0))), a0 = cH0/(2pi)")
    print(f"  a0(Khronon) = {a0_Khronon:.4e} m/s^2  |  a0(McGaugh) = 1.2000e-10 m/s^2")
    print("=" * 110)

    # Parse data
    mm_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")
    gp_path = os.path.join(DATA_DIR, "SPARC_Lelli2016c.mrt")

    print("\n  Parsing SPARC data...")
    galaxies = parse_mass_models(mm_path)
    props = parse_galaxy_properties(gp_path)
    print(f"  Found {len(galaxies)} galaxies with rotation curves")
    print(f"  Found {len(props)} galaxies with properties")

    # Residual collectors — grouped by various categories
    # (a) Morphological type
    res_by_type_K = defaultdict(list)
    res_by_type_N = defaultdict(list)

    # (b) Luminosity bin
    res_by_lum_K = defaultdict(list)
    res_by_lum_N = defaultdict(list)

    # (c) Surface brightness
    res_by_sb_K = defaultdict(list)
    res_by_sb_N = defaultdict(list)

    # (d) Radius bin
    res_by_radius_K = defaultdict(list)
    res_by_radius_N = defaultdict(list)

    # (e) Acceleration regime
    res_by_accel_K = defaultdict(list)
    res_by_accel_N = defaultdict(list)

    # Per-galaxy summary collectors
    galaxy_mean_res_K = {}
    galaxy_mean_res_N = {}
    galaxy_rms_K = {}
    galaxy_rms_N = {}

    # Also track early-type vs late-type
    res_by_early_late_K = defaultdict(list)
    res_by_early_late_N = defaultdict(list)

    n_gal = 0
    n_total_pts = 0
    all_res_K = []
    all_res_N = []
    all_weighted_res_K = []
    all_weighted_res_N = []

    print("\n  Fitting galaxies (Khronon M/L fit + NFW 3-param fit)...\n")

    sorted_galaxies = sorted(galaxies.items())
    n_total = len(sorted_galaxies)

    for i, (name, g) in enumerate(sorted_galaxies):
        R = g['R']
        Vobs = g['Vobs']
        e_Vobs = g['e_Vobs']
        n_pts = len(R)

        if n_pts < 3:
            continue

        # Get properties
        p = props.get(name, {})
        morph_type = p.get('type', 'Unknown')
        T = p.get('T', -1)
        L36 = p.get('L36_1e9Lsun', np.nan)
        SBdisk_central = p.get('SBdisk_central', np.nan)

        # Khronon fit
        ml_k, chi2_k, V_khronon, Vbar_k, g_bar = fit_khronon_ml(
            R, Vobs, e_Vobs, g['Vgas'], g['Vdisk'], g['Vbul'])

        # NFW fit
        ml_n, log10_M200, log10_c, chi2_n, V_nfw, Vbar_n = fit_nfw(
            R, Vobs, e_Vobs, g['Vgas'], g['Vdisk'], g['Vbul'])

        # Compute fractional residuals: (V_obs - V_model) / V_obs
        # Positive = underprediction, negative = overprediction
        mask = Vobs > 5.0  # Exclude very low velocity points (unreliable)
        if np.sum(mask) < 2:
            mask = np.ones(n_pts, dtype=bool)

        frac_res_K = (Vobs[mask] - V_khronon[mask]) / Vobs[mask]
        frac_res_N = (Vobs[mask] - V_nfw[mask]) / Vobs[mask]
        e_safe = np.maximum(e_Vobs[mask], 1.0)
        weighted_res_K = (Vobs[mask] - V_khronon[mask]) / e_safe
        weighted_res_N = (Vobs[mask] - V_nfw[mask]) / e_safe

        n_gal += 1
        n_total_pts += np.sum(mask)

        # Per-galaxy stats
        galaxy_mean_res_K[name] = np.mean(frac_res_K)
        galaxy_mean_res_N[name] = np.mean(frac_res_N)
        galaxy_rms_K[name] = np.sqrt(np.mean(frac_res_K**2))
        galaxy_rms_N[name] = np.sqrt(np.mean(frac_res_N**2))

        # Classify by luminosity
        if not np.isnan(L36) and L36 > 0:
            log_L = np.log10(L36 * 1e9)  # in L_sun
            if log_L >= 10.5:
                lum_bin = "High (>10.5)"
            elif log_L >= 9.5:
                lum_bin = "Med (9.5-10.5)"
            else:
                lum_bin = "Low (<9.5)"
        else:
            lum_bin = "Unknown"

        # Classify by surface brightness
        # LSB threshold: SBdisk_central < 100 L_sun/pc^2 (Freeman value ~140)
        if not np.isnan(SBdisk_central):
            if SBdisk_central >= 100:
                sb_class = "HSB (>=100)"
            else:
                sb_class = "LSB (<100)"
        else:
            sb_class = "Unknown"

        # Early vs late type
        if T <= 4:
            early_late = "Early (S0-Sbc)"
        else:
            early_late = "Late (Sc-BCD)"

        # Add point-by-point residuals to groups
        for j in range(len(frac_res_K)):
            idx = np.where(mask)[0][j]
            r_val = R[idx]
            g_bar_val = g_bar[idx]

            # (a) Morphological type
            res_by_type_K[morph_type].append(frac_res_K[j])
            res_by_type_N[morph_type].append(frac_res_N[j])

            # (b) Luminosity
            res_by_lum_K[lum_bin].append(frac_res_K[j])
            res_by_lum_N[lum_bin].append(frac_res_N[j])

            # (c) Surface brightness
            res_by_sb_K[sb_class].append(frac_res_K[j])
            res_by_sb_N[sb_class].append(frac_res_N[j])

            # (d) Radius bin
            if r_val < 5.0:
                r_bin = "Inner (<5 kpc)"
            elif r_val < 15.0:
                r_bin = "Middle (5-15)"
            else:
                r_bin = "Outer (>15 kpc)"
            res_by_radius_K[r_bin].append(frac_res_K[j])
            res_by_radius_N[r_bin].append(frac_res_N[j])

            # (e) Acceleration regime
            if g_bar_val > 5 * a0_Khronon:
                a_bin = "Newtonian (>5a0)"
            elif g_bar_val > 0.2 * a0_Khronon:
                a_bin = "Transition"
            else:
                a_bin = "Deep-MOND (<0.2a0)"
            res_by_accel_K[a_bin].append(frac_res_K[j])
            res_by_accel_N[a_bin].append(frac_res_N[j])

            # Early/late
            res_by_early_late_K[early_late].append(frac_res_K[j])
            res_by_early_late_N[early_late].append(frac_res_N[j])

        all_res_K.extend(frac_res_K.tolist())
        all_res_N.extend(frac_res_N.tolist())
        all_weighted_res_K.extend(weighted_res_K.tolist())
        all_weighted_res_N.extend(weighted_res_N.tolist())

        if (i + 1) % 25 == 0 or i == 0:
            print(f"    [{i+1:3d}/{n_total}] {name:14s}  "
                  f"<res>_K={np.mean(frac_res_K):+.3f}  RMS_K={np.sqrt(np.mean(frac_res_K**2)):.3f}  "
                  f"<res>_N={np.mean(frac_res_N):+.3f}  RMS_N={np.sqrt(np.mean(frac_res_N**2)):.3f}")

    print(f"\n  Processed {n_gal} galaxies, {n_total_pts} data points total.\n")

    # ── Print Results ─────────────────────────────────────────────────────

    # (a) By morphological type
    morph_order = ['S0', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'Im', 'BCD']
    morph_present = [m for m in morph_order if m in res_by_type_K]
    for m in res_by_type_K:
        if m not in morph_present:
            morph_present.append(m)
    print_table("(a) RESIDUALS BY MORPHOLOGICAL TYPE", res_by_type_K, res_by_type_N, morph_present)

    # (b) By luminosity
    lum_order = ["Low (<9.5)", "Med (9.5-10.5)", "High (>10.5)"]
    lum_present = [l for l in lum_order if l in res_by_lum_K]
    if "Unknown" in res_by_lum_K:
        lum_present.append("Unknown")
    print_table("(b) RESIDUALS BY LUMINOSITY (log L_[3.6] in L_sun)", res_by_lum_K, res_by_lum_N, lum_present)

    # (c) By surface brightness
    sb_order = ["HSB (>=100)", "LSB (<100)"]
    sb_present = [s for s in sb_order if s in res_by_sb_K]
    if "Unknown" in res_by_sb_K:
        sb_present.append("Unknown")
    print_table("(c) RESIDUALS BY SURFACE BRIGHTNESS (SB_disk central, L_sun/pc^2)", res_by_sb_K, res_by_sb_N, sb_present)

    # (d) By radius
    r_order = ["Inner (<5 kpc)", "Middle (5-15)", "Outer (>15 kpc)"]
    r_present = [r for r in r_order if r in res_by_radius_K]
    print_table("(d) RESIDUALS BY RADIUS BIN", res_by_radius_K, res_by_radius_N, r_present)

    # (e) By acceleration regime
    a_order = ["Newtonian (>5a0)", "Transition", "Deep-MOND (<0.2a0)"]
    a_present = [a for a in a_order if a in res_by_accel_K]
    print_table("(e) RESIDUALS BY ACCELERATION REGIME", res_by_accel_K, res_by_accel_N, a_present)

    # Early vs Late
    el_order = ["Early (S0-Sbc)", "Late (Sc-BCD)"]
    el_present = [e for e in el_order if e in res_by_early_late_K]
    print_table("(f) RESIDUALS BY EARLY vs LATE TYPE", res_by_early_late_K, res_by_early_late_N, el_present)

    # ── Global Summary ────────────────────────────────────────────────────

    all_res_K = np.array(all_res_K)
    all_res_N = np.array(all_res_N)
    all_weighted_K = np.array(all_weighted_res_K)
    all_weighted_N = np.array(all_weighted_res_N)

    print()
    print("=" * 110)
    print("  GLOBAL SUMMARY")
    print("=" * 110)
    print(f"  Galaxies:         {n_gal}")
    print(f"  Data points:      {n_total_pts}")
    print(f"  a0(Khronon):      {a0_Khronon:.4e} m/s^2")
    print()

    print(f"  {'Metric':<40s}  {'Khronon':>12s}  {'NFW':>12s}  {'K better?':>10s}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*12}  {'-'*10}")

    mean_k = np.mean(all_res_K)
    mean_n = np.mean(all_res_N)
    rms_k = np.sqrt(np.mean(all_res_K**2))
    rms_n = np.sqrt(np.mean(all_res_N**2))
    median_k = np.median(all_res_K)
    median_n = np.median(all_res_N)
    std_k = np.std(all_res_K)
    std_n = np.std(all_res_N)
    print(f"  {'Mean fractional residual':<40s}  {mean_k:+12.5f}  {mean_n:+12.5f}  {'YES' if abs(mean_k) <= abs(mean_n) else 'no':>10s}")
    print(f"  {'Median fractional residual':<40s}  {median_k:+12.5f}  {median_n:+12.5f}  {'YES' if abs(median_k) <= abs(median_n) else 'no':>10s}")
    print(f"  {'RMS fractional residual':<40s}  {rms_k:12.5f}  {rms_n:12.5f}  {'YES' if rms_k <= rms_n else 'no':>10s}")
    print(f"  {'Std fractional residual':<40s}  {std_k:12.5f}  {std_n:12.5f}  {'YES' if std_k <= std_n else 'no':>10s}")

    # Weighted (chi-like)
    rms_wk = np.sqrt(np.mean(all_weighted_K**2))
    rms_wn = np.sqrt(np.mean(all_weighted_N**2))
    print(f"  {'RMS weighted residual (chi-like)':<40s}  {rms_wk:12.3f}  {rms_wn:12.3f}  {'YES' if rms_wk <= rms_wn else 'no':>10s}")

    # Per-galaxy: how many galaxies does each model win?
    n_k_wins = sum(1 for name in galaxy_rms_K if galaxy_rms_K[name] <= galaxy_rms_N.get(name, 1e10))
    n_n_wins = n_gal - n_k_wins
    print(f"  {'Galaxies where K has lower RMS':<40s}  {n_k_wins:12d}  {n_n_wins:12d}")

    print()

    # ── Systematic Trend Detection ────────────────────────────────────────

    print("=" * 110)
    print("  SYSTEMATIC TREND DETECTION")
    print("=" * 110)

    # Check for monotonic trends in acceleration
    print("\n  (1) Acceleration-dependent trend (Khronon):")
    for a_bin in a_order:
        if a_bin in res_by_accel_K:
            s = compute_group_stats(res_by_accel_K[a_bin])
            sigma = s['mean'] / s['mean_err'] if s['mean_err'] > 0 else 0
            print(f"      {a_bin:<25s}: <res> = {s['mean']:+.4f} +/- {s['mean_err']:.4f}  ({sigma:+.1f}sigma)")

    # Check for radius-dependent trend
    print("\n  (2) Radius-dependent trend (Khronon):")
    for r_bin in r_order:
        if r_bin in res_by_radius_K:
            s = compute_group_stats(res_by_radius_K[r_bin])
            sigma = s['mean'] / s['mean_err'] if s['mean_err'] > 0 else 0
            print(f"      {r_bin:<25s}: <res> = {s['mean']:+.4f} +/- {s['mean_err']:.4f}  ({sigma:+.1f}sigma)")

    # Check for luminosity-dependent trend
    print("\n  (3) Luminosity-dependent trend (Khronon):")
    for l_bin in lum_order:
        if l_bin in res_by_lum_K:
            s = compute_group_stats(res_by_lum_K[l_bin])
            sigma = s['mean'] / s['mean_err'] if s['mean_err'] > 0 else 0
            print(f"      {l_bin:<25s}: <res> = {s['mean']:+.4f} +/- {s['mean_err']:.4f}  ({sigma:+.1f}sigma)")

    # Check for morphology-dependent trend (early to late)
    print("\n  (4) Morphology-dependent trend (Khronon):")
    for e_bin in el_order:
        if e_bin in res_by_early_late_K:
            s = compute_group_stats(res_by_early_late_K[e_bin])
            sigma = s['mean'] / s['mean_err'] if s['mean_err'] > 0 else 0
            print(f"      {e_bin:<25s}: <res> = {s['mean']:+.4f} +/- {s['mean_err']:.4f}  ({sigma:+.1f}sigma)")

    # ── Worst galaxies ────────────────────────────────────────────────────

    print("\n" + "=" * 110)
    print("  TOP 15 WORST KHRONON FITS (by |mean residual|)")
    print("=" * 110)
    worst_k = sorted(galaxy_mean_res_K.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    print(f"  {'Galaxy':<15s}  {'Type':<6s}  {'<res>_K':>9s}  {'RMS_K':>7s}  {'<res>_N':>9s}  {'RMS_N':>7s}  {'K vs N':>8s}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*8}")
    for name, mean_res in worst_k:
        p = props.get(name, {})
        morph = p.get('type', '?')
        rk = galaxy_rms_K.get(name, np.nan)
        rn = galaxy_rms_N.get(name, np.nan)
        mn = galaxy_mean_res_N.get(name, np.nan)
        better = "K" if rk <= rn else "NFW"
        print(f"  {name:<15s}  {morph:<6s}  {mean_res:+9.4f}  {rk:7.4f}  {mn:+9.4f}  {rn:7.4f}  {better:>8s}")

    # ── Overall Verdict ───────────────────────────────────────────────────

    print("\n" + "=" * 110)
    print("  VERDICT: Are Khronon residuals random or systematic?")
    print("=" * 110)

    # Count how many groupings show >2sigma bias
    n_systematic = 0
    n_checked = 0

    all_groups = [
        ("Type", res_by_type_K),
        ("Luminosity", res_by_lum_K),
        ("Surface Brightness", res_by_sb_K),
        ("Radius", res_by_radius_K),
        ("Acceleration", res_by_accel_K),
        ("Early/Late", res_by_early_late_K),
    ]

    for group_name, group_dict in all_groups:
        for key, vals in group_dict.items():
            if key == "Unknown":
                continue
            s = compute_group_stats(vals)
            if s['n'] >= 10 and s['mean_err'] > 0:
                n_checked += 1
                sigma = abs(s['mean'] / s['mean_err'])
                if sigma > 2:
                    n_systematic += 1

    print(f"\n  Sub-groups checked (N >= 10):  {n_checked}")
    print(f"  Sub-groups with |<res>| > 2sigma:  {n_systematic}  ({100*n_systematic/max(n_checked,1):.0f}%)")

    global_sigma_k = abs(mean_k) / (std_k / np.sqrt(len(all_res_K)))
    global_sigma_n = abs(mean_n) / (std_n / np.sqrt(len(all_res_N)))

    print(f"\n  Global Khronon bias:  <res> = {mean_k:+.5f} ({global_sigma_k:.1f}sigma from zero)")
    print(f"  Global NFW bias:      <res> = {mean_n:+.5f} ({global_sigma_n:.1f}sigma from zero)")

    if n_systematic / max(n_checked, 1) > 0.5:
        print("\n  ** SYSTEMATIC RESIDUALS DETECTED **")
        print("  More than 50% of sub-groups show >2sigma bias.")
        print("  Khronon theory has systematic issues in certain regimes.")
    elif n_systematic / max(n_checked, 1) > 0.2:
        print("\n  ** MILD SYSTEMATIC TRENDS DETECTED **")
        print("  Some sub-groups show significant bias (20-50%).")
        print("  Khronon works well overall but has localized issues.")
    else:
        print("\n  ** RESIDUALS ARE PREDOMINANTLY RANDOM **")
        print("  Fewer than 20% of sub-groups show >2sigma bias.")
        print("  Khronon theory is a good description of rotation curves.")

    print(f"\n  Khronon (1 free param) RMS = {rms_k:.4f}")
    print(f"  NFW     (3 free params) RMS = {rms_n:.4f}")
    if rms_k <= rms_n * 1.1:
        print("  Khronon achieves comparable/better fit with FEWER parameters.")
    else:
        print(f"  NFW achieves {rms_n/rms_k:.2f}x lower RMS but uses 3x more parameters.")

    print("\n  Done.\n")


if __name__ == '__main__':
    main()
