#!/usr/bin/env python3
"""
Test: Global a0 fit across all 175 SPARC galaxies.

Fits ONE global a0 shared across all galaxies, with per-galaxy M/L_disk free.
Compares to NFW (3 params per galaxy) via BIC.

Two-level optimization:
  - Outer: scipy.optimize.minimize_scalar over a0
  - Inner: scipy.optimize.minimize (Nelder-Mead) over M/L_disk per galaxy
"""

import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

# ── Constants ────────────────────────────────────────────────────────────
G_SI = 6.674e-11        # m^3/(kg·s^2)
c_SI = 2.998e8           # m/s
H0_SI = 73e3 / 3.086e22  # 73 km/s/Mpc -> s^-1
Msun = 1.989e30          # kg
kpc_m = 3.086e19         # m per kpc
km_s = 1e3               # m/s per km/s

# Predicted a0
a0_predicted = c_SI * H0_SI / (2 * np.pi)  # cH0/(2pi) ~ 1.13e-10
a0_mcgaugh = 1.20e-10  # McGaugh 2016 empirical

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sparc")
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "data", "sparc_rotation_curves.json")


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


# ── Physics ──────────────────────────────────────────────────────────────

def compute_Vbar(Vgas, Vdisk, Vbul, ml_disk, ml_bulge):
    """V^2_bar = V^2_gas + ML_disk * V^2_disk + ML_bulge * V^2_bulge"""
    V2_gas = np.sign(Vgas) * Vgas**2
    V2_disk = ml_disk * np.sign(Vdisk) * Vdisk**2
    V2_bul = ml_bulge * np.sign(Vbul) * Vbul**2
    V2_bar = V2_gas + V2_disk + V2_bul
    return np.sign(V2_bar) * np.sqrt(np.abs(V2_bar))


def compute_V_khronon(R_kpc, Vbar_kms, a0):
    """
    Khronon RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))
    Returns V_total in km/s.
    """
    R_m = R_kpc * kpc_m
    Vbar_ms = Vbar_kms * km_s

    g_bar = Vbar_ms**2 / R_m

    x = np.sqrt(g_bar / a0)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-30)
    g_obs = g_bar / denom

    V_total_ms = np.sqrt(g_obs * R_m)
    return V_total_ms / km_s


def fit_ml_for_galaxy(R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, a0):
    """
    Fit M/L_disk for a single galaxy at fixed a0.
    M/L_bulge = 1.4 * M/L_disk.
    Returns (ml_disk, chi2).
    """
    e_safe = np.maximum(e_Vobs, 1.0)

    def chi2_func(params):
        ml_d = params[0]
        if ml_d < 0.1 or ml_d > 1.5:
            return 1e10
        ml_b = 1.4 * ml_d
        Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
        V_k = compute_V_khronon(R, np.abs(Vbar), a0)
        return np.sum(((Vobs - V_k) / e_safe)**2)

    best = None
    best_chi2 = 1e20
    for ml0 in [0.3, 0.5, 0.7, 0.9]:
        try:
            res = minimize(chi2_func, [ml0], method='Nelder-Mead',
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

    # Recompute final chi2
    ml_b = 1.4 * ml_d
    Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
    V_k = compute_V_khronon(R, np.abs(Vbar), a0)
    chi2_val = np.sum(((Vobs - V_k) / e_safe)**2)

    return ml_d, chi2_val


def global_chi2_for_a0(a0, galaxy_list):
    """
    For a given global a0, fit M/L per galaxy and return total chi2.
    galaxy_list: list of (R, Vobs, e_Vobs, Vgas, Vdisk, Vbul) tuples.
    """
    total_chi2 = 0.0
    for (R, Vobs, e_Vobs, Vgas, Vdisk, Vbul) in galaxy_list:
        ml_d, chi2 = fit_ml_for_galaxy(R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, a0)
        total_chi2 += chi2
    return total_chi2


# ── Main ────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print("=" * 74)
    print("  GLOBAL a0 FIT: One a0 across all 175 SPARC galaxies")
    print("  Khronon RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))")
    print("=" * 74)
    print()

    # Parse data
    mm_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")
    print("  Parsing SPARC data...")
    galaxies = parse_mass_models(mm_path)
    print(f"  Found {len(galaxies)} galaxies")
    print()

    # Prepare galaxy list (skip galaxies with < 3 points)
    galaxy_names = []
    galaxy_data = []
    total_data_points = 0

    for name in sorted(galaxies.keys()):
        g = galaxies[name]
        R = g['R']
        if len(R) < 3:
            continue
        galaxy_names.append(name)
        galaxy_data.append((g['R'], g['Vobs'], g['e_Vobs'],
                            g['Vgas'], g['Vdisk'], g['Vbul']))
        total_data_points += len(R)

    n_galaxies = len(galaxy_names)
    print(f"  Galaxies with >= 3 data points: {n_galaxies}")
    print(f"  Total data points: {total_data_points}")
    print()

    # ── Step 1: Coarse scan of a0 ────────────────────────────────────────
    print("  Step 1: Coarse scan over a0...")
    a0_values = np.linspace(0.5e-10, 3.0e-10, 26)
    chi2_scan = []

    for i, a0_test in enumerate(a0_values):
        chi2_val = global_chi2_for_a0(a0_test, galaxy_data)
        chi2_scan.append(chi2_val)
        print(f"    a0 = {a0_test:.3e} m/s^2  ->  chi2_total = {chi2_val:.1f}")
        sys.stdout.flush()

    chi2_scan = np.array(chi2_scan)
    idx_min = np.argmin(chi2_scan)
    a0_coarse = a0_values[idx_min]
    print(f"\n  Coarse minimum: a0 = {a0_coarse:.3e}, chi2 = {chi2_scan[idx_min]:.1f}")
    print()

    # ── Step 2: Fine optimization with minimize_scalar ──────────────────
    print("  Step 2: Fine optimization with minimize_scalar...")
    # Search around the coarse minimum
    a0_lo = max(0.5e-10, a0_coarse - 0.3e-10)
    a0_hi = min(3.0e-10, a0_coarse + 0.3e-10)

    result = minimize_scalar(
        lambda a0: global_chi2_for_a0(a0, galaxy_data),
        bounds=(a0_lo, a0_hi),
        method='bounded',
        options={'xatol': 1e-13, 'maxiter': 30}
    )

    a0_best = result.x
    chi2_best = result.fun
    print(f"  Best-fit a0 = {a0_best:.4e} m/s^2")
    print(f"  Total chi2  = {chi2_best:.1f}")
    print()

    # ── Step 3: Collect per-galaxy results at best a0 ───────────────────
    print("  Step 3: Computing per-galaxy fits at best a0...")
    ml_results = {}
    chi2_per_galaxy = {}
    total_chi2_khronon = 0.0

    for i, (name, gdata) in enumerate(zip(galaxy_names, galaxy_data)):
        R, Vobs, e_Vobs, Vgas, Vdisk, Vbul = gdata
        ml_d, chi2 = fit_ml_for_galaxy(R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, a0_best)
        ml_results[name] = ml_d
        chi2_per_galaxy[name] = chi2
        total_chi2_khronon += chi2

    ml_vals = np.array(list(ml_results.values()))
    print(f"  M/L_disk: median = {np.median(ml_vals):.3f}, "
          f"mean = {np.mean(ml_vals):.3f}, "
          f"std = {np.std(ml_vals):.3f}")
    print(f"  M/L_disk range: [{np.min(ml_vals):.3f}, {np.max(ml_vals):.3f}]")
    print()

    # ── Step 4: Load NFW results from JSON ──────────────────────────────
    print("  Step 4: Loading NFW results from JSON...")
    total_chi2_nfw = 0.0
    n_nfw_galaxies = 0
    nfw_n_pts_total = 0

    try:
        with open(JSON_PATH, 'r') as f:
            json_data = json.load(f)

        for gal in json_data['galaxies']:
            name = gal['name']
            n_pts = gal['n_pts']
            if n_pts < 3:
                continue
            # chi2_nfw in JSON is reduced chi2 (per dof = n_pts - 3)
            chi2_red_nfw = gal['chi2_nfw']
            dof_nfw = max(n_pts - 3, 1)
            chi2_abs_nfw = chi2_red_nfw * dof_nfw
            total_chi2_nfw += chi2_abs_nfw
            nfw_n_pts_total += n_pts
            n_nfw_galaxies += 1

        print(f"  Loaded NFW fits for {n_nfw_galaxies} galaxies")
        print(f"  Total chi2 (NFW) = {total_chi2_nfw:.1f}")
    except Exception as e:
        print(f"  WARNING: Could not load NFW JSON: {e}")
        print("  Skipping NFW comparison.")
        total_chi2_nfw = None

    print()

    # ── Step 5: BIC comparison ──────────────────────────────────────────
    print("=" * 74)
    print("  RESULTS")
    print("=" * 74)
    print()

    # Khronon parameters: 1 global a0 + 175 per-galaxy M/L = 176 total
    n_params_khronon = 1 + n_galaxies  # = 176
    # NFW parameters: 3 per galaxy (M/L + M200 + c) = 525 total
    n_params_nfw = 3 * n_galaxies  # = 525

    N = total_data_points

    BIC_khronon = chi2_best + n_params_khronon * np.log(N)
    print(f"  Khronon global fit:")
    print(f"    Best-fit a0       = {a0_best:.4e} m/s^2")
    print(f"    Predicted cH0/2pi = {a0_predicted:.4e} m/s^2")
    print(f"    McGaugh empirical = {a0_mcgaugh:.4e} m/s^2")
    print(f"    Ratio (fit/predicted) = {a0_best / a0_predicted:.4f}")
    print(f"    Ratio (fit/McGaugh)   = {a0_best / a0_mcgaugh:.4f}")
    print()
    print(f"    Total chi2     = {chi2_best:.1f}")
    print(f"    N_data         = {N}")
    print(f"    N_params       = {n_params_khronon} (1 global a0 + {n_galaxies} M/L)")
    print(f"    chi2/dof       = {chi2_best / (N - n_params_khronon):.4f}")
    print(f"    chi2/param     = {chi2_best / n_params_khronon:.2f}")
    print(f"    BIC            = {BIC_khronon:.1f}")
    print()

    if total_chi2_nfw is not None:
        BIC_nfw = total_chi2_nfw + n_params_nfw * np.log(N)
        delta_BIC = BIC_nfw - BIC_khronon

        print(f"  NFW (from pre-computed JSON):")
        print(f"    Total chi2     = {total_chi2_nfw:.1f}")
        print(f"    N_params       = {n_params_nfw} (3 per galaxy x {n_galaxies})")
        print(f"    chi2/dof       = {total_chi2_nfw / (N - n_params_nfw):.4f}")
        print(f"    chi2/param     = {total_chi2_nfw / n_params_nfw:.2f}")
        print(f"    BIC            = {BIC_nfw:.1f}")
        print()
        print(f"  BIC comparison:")
        print(f"    delta_BIC = BIC_NFW - BIC_Khronon = {delta_BIC:.1f}")
        if delta_BIC > 10:
            print(f"    -> VERY STRONG evidence for Khronon over NFW")
        elif delta_BIC > 6:
            print(f"    -> STRONG evidence for Khronon over NFW")
        elif delta_BIC > 2:
            print(f"    -> POSITIVE evidence for Khronon over NFW")
        elif delta_BIC > -2:
            print(f"    -> INCONCLUSIVE")
        elif delta_BIC > -6:
            print(f"    -> POSITIVE evidence for NFW over Khronon")
        else:
            print(f"    -> STRONG evidence for NFW over Khronon")
        print()
        print(f"  Parameter efficiency:")
        print(f"    Khronon: {chi2_best/n_params_khronon:.2f} chi2/param  ({n_params_khronon} params)")
        print(f"    NFW:     {total_chi2_nfw/n_params_nfw:.2f} chi2/param  ({n_params_nfw} params)")
        print(f"    Khronon uses {n_params_nfw - n_params_khronon} FEWER parameters ({n_params_nfw/n_params_khronon:.1f}x more efficient)")
    print()

    # ── Step 6: Also compute chi2 at predicted a0 for comparison ────────
    print("  Bonus: chi2 at specific a0 values (all with per-galaxy M/L free):")
    for label, a0_val in [("cH0/2pi (predicted)", a0_predicted),
                           ("McGaugh 2016", a0_mcgaugh),
                           ("Best fit", a0_best)]:
        chi2_val = global_chi2_for_a0(a0_val, galaxy_data)
        print(f"    a0 = {a0_val:.4e}  ({label:20s})  chi2 = {chi2_val:.1f}")

    print()
    elapsed = time.time() - t_start
    print(f"  Total runtime: {elapsed:.1f} s")
    print("=" * 74)


if __name__ == '__main__':
    main()
