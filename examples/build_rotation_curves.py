#!/usr/bin/env python3
"""
Build rotation curve comparison data from SPARC observations.

Computes:
  - V_bar: baryonic velocity (gas + disk + bulge)
  - V_Khronon: Khronon/τ framework prediction (ZERO free parameters)
  - V_NFW: best-fit NFW dark matter halo (2 free parameters per galaxy)

Exports JSON for interactive Three.js/web visualization.

Usage:
    python examples/build_rotation_curves.py
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

# ── Constants ────────────────────────────────────────────────────────────
G_SI = 6.674e-11        # m³/(kg·s²)
c_SI = 2.998e8           # m/s
H0_SI = 73e3 / 3.086e22  # 73 km/s/Mpc → s⁻¹
Msun = 1.989e30          # kg
kpc_m = 3.086e19         # m per kpc
km_s = 1e3               # m/s per km/s

# MOND acceleration scale from Khronon: a₀ = cH₀/(2π)
a0_Khronon = c_SI * H0_SI / (2 * np.pi)  # ≈ 1.13e-10 m/s²
# McGaugh empirical: a₀ = 1.2e-10 m/s² — remarkably close

# M/L ratios at 3.6 μm (Schombert & McGaugh 2014; SPS models)
ML_DISK = 0.5   # solar masses per solar luminosity
ML_BULGE = 0.7

# NFW cosmology
rho_crit = 3 * H0_SI**2 / (8 * np.pi * G_SI)  # critical density
DELTA_VIR = 200  # virial overdensity

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sparc")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "web", "data")


# ── Parse SPARC data ────────────────────────────────────────────────────

def parse_mass_models(filepath):
    """Parse MassModels_Lelli2016c.mrt (fixed-width format)."""
    galaxies = {}
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header lines
            if line.startswith(('Title', 'Authors', 'Table', '=', '-',
                                ' ', 'Byte', 'Note', '\n')):
                if line.startswith(' ') and len(line.strip()) > 0:
                    # Check if it's a data line (starts with space + galaxy name)
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        try:
                            float(parts[1])  # distance should be a number
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
                D = float(parts[1])      # Mpc
                R = float(parts[2])      # kpc
                Vobs = float(parts[3])   # km/s
                e_Vobs = float(parts[4]) # km/s
                Vgas = float(parts[5])   # km/s
                Vdisk = float(parts[6])  # km/s
                Vbul = float(parts[7])   # km/s
                SBdisk = float(parts[8]) # solLum/pc²
                SBbul = float(parts[9]) if len(parts) > 9 else 0.0

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

    # Convert to numpy arrays
    for name in galaxies:
        for key in ['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']:
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
                T = int(parts[1])       # Hubble type
                D = float(parts[2])     # Mpc
                Inc = float(parts[5])   # deg
                L36 = float(parts[7])   # 10⁹ L☉
                Vflat = float(parts[15]) if len(parts) > 15 else 0.0
                Q = int(parts[17]) if len(parts) > 17 else 2

                hubble_names = {
                    0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc',
                    5: 'Sc', 6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm',
                    10: 'Im', 11: 'BCD'
                }

                props[name] = {
                    'type': hubble_names.get(T, f'T{T}'),
                    'D_Mpc': D,
                    'Inc_deg': Inc,
                    'L36_1e9Lsun': L36,
                    'Vflat_kms': Vflat,
                    'quality': Q,
                }
            except (ValueError, IndexError):
                continue

    return props


# ── Physics: Baryonic velocity ──────────────────────────────────────────

def compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ML_DISK, ml_bulge=ML_BULGE):
    """
    V²_bar = V²_gas + (M/L_disk) × V²_disk + (M/L_bulge) × V²_bulge

    Note: Vdisk, Vbul in SPARC are for M/L=1, so we scale by sqrt(M/L).
    The sign of V components encodes direction; V² uses sign(V)*V².
    """
    V2_gas = np.sign(Vgas) * Vgas**2
    V2_disk = ml_disk * np.sign(Vdisk) * Vdisk**2
    V2_bul = ml_bulge * np.sign(Vbul) * Vbul**2
    V2_bar = V2_gas + V2_disk + V2_bul
    return np.sign(V2_bar) * np.sqrt(np.abs(V2_bar))


# ── Physics: Khronon/τ prediction (RAR, ZERO free parameters) ──────────

def compute_V_khronon(R_kpc, Vbar_kms, a0=a0_Khronon):
    """
    Khronon RAR prediction:
      g_obs = g_bar / (1 - exp(-√(g_bar/a₀)))

    This is the McGaugh (2016) RAR form, but with a₀ = cH₀/(2π) PREDICTED
    by the Khronon framework (Paper 3), not fitted.

    Returns V_total in km/s.
    """
    R_m = R_kpc * kpc_m
    Vbar_ms = Vbar_kms * km_s

    # Baryonic acceleration (handle negative V²_bar)
    g_bar = Vbar_ms**2 / R_m  # always positive since Vbar is |Vbar|

    # RAR interpolation
    x = np.sqrt(g_bar / a0)
    # Avoid division by zero for very small g_bar
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-30)
    g_obs = g_bar / denom

    V_total_ms = np.sqrt(g_obs * R_m)
    return V_total_ms / km_s


def fit_khronon_ml(R_kpc, Vobs_kms, e_Vobs_kms, Vgas, Vdisk, Vbul,
                   a0=a0_Khronon):
    """
    Fit Khronon RAR with M/L_disk as single free parameter.
    a₀ = cH₀/(2π) is FIXED (predicted, not fitted).
    M/L_bulge = 1.4 × M/L_disk (SPS constraint).

    Returns: (ml_disk, chi2, V_khronon)
    """
    def chi2(params):
        ml_d = params[0]
        if ml_d < 0.1 or ml_d > 1.5:
            return 1e10
        ml_b = 1.4 * ml_d
        Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
        V_k = compute_V_khronon(R_kpc, np.abs(Vbar), a0=a0)
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
    V_k = compute_V_khronon(R_kpc, np.abs(Vbar), a0=a0)
    e_safe = np.maximum(e_Vobs_kms, 1.0)
    chi2_val = np.sum(((Vobs_kms - V_k) / e_safe)**2)

    return ml_d, chi2_val, V_k, np.abs(Vbar)


# ── Physics: NFW dark matter halo (2 free parameters) ──────────────────

def nfw_velocity(R_kpc, log10_M200, log10_c):
    """
    NFW halo circular velocity.

    Parameters:
        R_kpc: radii in kpc
        log10_M200: log10(M200 / M☉)
        log10_c: log10(concentration)

    Returns V_NFW in km/s.
    """
    M200 = 10**log10_M200 * Msun  # kg
    c = 10**log10_c

    # Virial radius
    r200 = (3 * M200 / (4 * np.pi * DELTA_VIR * rho_crit))**(1./3.)  # m

    R_m = R_kpc * kpc_m
    x = c * R_m / r200

    # NFW enclosed mass function
    fx = np.log(1 + x) - x / (1 + x)
    fc = np.log(1 + c) - c / (1 + c)

    V2_nfw = G_SI * M200 * fx / (R_m * fc)
    V2_nfw = np.maximum(V2_nfw, 0)

    return np.sqrt(V2_nfw) / km_s


def fit_nfw(R_kpc, Vobs_kms, e_Vobs_kms, Vgas, Vdisk, Vbul):
    """
    Fit NFW halo + free M/L to rotation curve.

    V²_total = V²_bar(M/L) + V²_NFW(M200, c)
    3 free parameters: M/L_disk, log10(M200), log10(c)

    Returns: (ml_disk, log10_M200, log10_c, chi2, V_total_kms, Vbar_kms)
    """
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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SPARC Rotation Curve Comparison")
    print("  Khronon (1 param: M/L) vs NFW (3 params: M/L + M200 + c)")
    print("=" * 70)
    print(f"  a₀(Khronon) = cH₀/(2π) = {a0_Khronon:.3e} m/s²")
    print(f"  a₀(McGaugh) = 1.20e-10 m/s²")
    print(f"  Ratio: {a0_Khronon / 1.2e-10:.3f}")
    print()

    # Parse data
    mm_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")
    gp_path = os.path.join(DATA_DIR, "SPARC_Lelli2016c.mrt")

    print("  Parsing SPARC data...")
    galaxies = parse_mass_models(mm_path)
    props = parse_galaxy_properties(gp_path)
    print(f"  Found {len(galaxies)} galaxies with rotation curves")
    print(f"  Found {len(props)} galaxies with properties")
    print()

    # Process each galaxy
    results = []
    chi2_khronon_list = []
    chi2_nfw_list = []
    n_khronon_wins = 0
    n_nfw_wins = 0
    n_tie = 0

    for i, (name, g) in enumerate(sorted(galaxies.items())):
        R = g['R']
        Vobs = g['Vobs']
        e_Vobs = g['e_Vobs']
        n_pts = len(R)

        if n_pts < 3:
            continue

        # Khronon fit: a₀ fixed (predicted), M/L free → 1 param
        ml_k, chi2_k, V_khronon, Vbar_k = fit_khronon_ml(
            R, Vobs, e_Vobs, g['Vgas'], g['Vdisk'], g['Vbul'])
        chi2_k_red = chi2_k / max(n_pts - 1, 1)  # 1 free param

        # NFW fit: M/L + M200 + c free → 3 params
        ml_n, log10_M200, log10_c, chi2_n, V_nfw_total, Vbar_n = fit_nfw(
            R, Vobs, e_Vobs, g['Vgas'], g['Vdisk'], g['Vbul'])
        chi2_n_red = chi2_n / max(n_pts - 3, 1)  # 3 free params

        # BIC comparison (lower = better)
        BIC_k = chi2_k + 1 * np.log(n_pts)       # 1 param
        BIC_n = chi2_n + 3 * np.log(n_pts)        # 3 params
        delta_BIC = BIC_n - BIC_k  # positive = Khronon preferred

        if delta_BIC > 2:
            n_khronon_wins += 1
            winner = "Khronon"
        elif delta_BIC < -2:
            n_nfw_wins += 1
            winner = "NFW"
        else:
            n_tie += 1
            winner = "tie"

        chi2_khronon_list.append(chi2_k_red)
        chi2_nfw_list.append(chi2_n_red)

        # Get properties
        p = props.get(name, {})

        entry = {
            'name': name,
            'type': p.get('type', ''),
            'D_Mpc': float(g['D']),
            'Vflat': float(p.get('Vflat_kms', 0)),
            'L36': float(p.get('L36_1e9Lsun', 0)),
            'quality': int(p.get('quality', 2)),
            'n_pts': n_pts,
            'R': R.tolist(),
            'Vobs': Vobs.tolist(),
            'e_Vobs': e_Vobs.tolist(),
            'Vbar': Vbar_k.tolist(),
            'V_khronon': V_khronon.tolist(),
            'V_nfw': V_nfw_total.tolist(),
            'Vbar_nfw': Vbar_n.tolist(),
            'chi2_khronon': round(chi2_k_red, 2),
            'chi2_nfw': round(chi2_n_red, 2),
            'ml_khronon': round(ml_k, 2),
            'ml_nfw': round(ml_n, 2),
            'nfw_log10_M200': round(log10_M200, 2),
            'nfw_log10_c': round(log10_c, 2),
            'delta_BIC': round(delta_BIC, 1),
            'winner': winner,
        }
        results.append(entry)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(galaxies)}] {name:12s}  "
                  f"χ²_K={chi2_k_red:6.1f}  χ²_N={chi2_n_red:6.1f}  "
                  f"ΔBIC={delta_BIC:+6.1f}  → {winner}")

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Galaxies processed: {len(results)}")
    print(f"  Khronon preferred (ΔBIC > 2):  {n_khronon_wins}")
    print(f"  NFW preferred (ΔBIC < -2):     {n_nfw_wins}")
    print(f"  Inconclusive (|ΔBIC| ≤ 2):     {n_tie}")
    print()
    print(f"  Median χ²_red (Khronon): {np.median(chi2_khronon_list):.2f}")
    print(f"  Median χ²_red (NFW):     {np.median(chi2_nfw_list):.2f}")
    print(f"  a₀(Khronon) = {a0_Khronon:.4e} m/s²")
    print()

    # Sort by luminosity for nice default ordering
    results.sort(key=lambda x: -x.get('L36', 0))

    # Featured galaxies (well-known, high quality)
    featured = ['NGC2403', 'NGC3198', 'NGC2841', 'NGC7331', 'NGC6503',
                'NGC2903', 'NGC5055', 'NGC3521', 'NGC1003', 'DDO154',
                'NGC2976', 'IC2574', 'NGC3741', 'NGC7793', 'UGC02885']

    # Build output
    output = {
        'metadata': {
            'source': 'SPARC (Lelli, McGaugh & Schombert 2016, AJ 152, 157)',
            'n_galaxies': len(results),
            'a0_khronon_m_s2': a0_Khronon,
            'a0_mcgaugh_m_s2': 1.2e-10,
            'ML_disk': ML_DISK,
            'ML_bulge': ML_BULGE,
            'H0_km_s_Mpc': 73,
            'summary': {
                'khronon_preferred': n_khronon_wins,
                'nfw_preferred': n_nfw_wins,
                'inconclusive': n_tie,
                'median_chi2_khronon': round(np.median(chi2_khronon_list), 2),
                'median_chi2_nfw': round(np.median(chi2_nfw_list), 2),
            },
            'featured': featured,
        },
        'galaxies': results,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "sparc_rotation_curves.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Exported: {out_path} ({size_kb:.0f} KB)")
    print("  Done.")


if __name__ == '__main__':
    main()
