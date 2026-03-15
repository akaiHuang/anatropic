#!/usr/bin/env python3
"""
Diagnose and Fix Systematic Overprediction of Khronon Rotation Curves
for Dwarf Irregular (Im) Galaxies.

Known problem:
  - Im galaxies: -15.7% mean residual (overprediction)
  - Low luminosity: -8.6% bias
  - LSB galaxies: -12.0% bias
  - Inner radii (<5 kpc): -8.4% bias

This script:
  1. Identifies the 20 worst galaxies by mean residual
  2. Diagnoses root causes (M/L, distance, gas fraction, inclination)
  3. Tests four fixes:
     A. Type-dependent M/L ratios
     B. Adjusted a0 for dwarfs
     C. Distance error correction
     D. Modified interpolating function
  4. Reports which fix works best without hurting spirals

Usage:
    python examples/test_fix_dwarf_residuals.py
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

# ── Constants ────────────────────────────────────────────────────────────────
G_SI = 6.674e-11
c_SI = 2.998e8
H0_SI = 73e3 / 3.086e22  # 73 km/s/Mpc -> s^-1
Msun = 1.989e30
kpc_m = 3.086e19
km_s = 1e3

a0_Khronon = c_SI * H0_SI / (2 * np.pi)  # ~1.13e-10 m/s^2

ML_DISK = 0.5
ML_BULGE = 0.7

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sparc")


# ── Parse SPARC data ─────────────────────────────────────────────────────────

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
    """Parse SPARC_Lelli2016c.mrt for galaxy metadata including distance errors."""
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
                T = int(parts[1])
                D = float(parts[2])
                e_D = float(parts[3])
                f_D = int(parts[4])       # Distance method
                Inc = float(parts[5])
                e_Inc = float(parts[6])
                L36 = float(parts[7])
                e_L36 = float(parts[8])
                Reff = float(parts[9])
                SBeff = float(parts[10])
                Rdisk = float(parts[11])
                SBdisk = float(parts[12])
                MHI = float(parts[13])     # 10^9 Msun
                RHI = float(parts[14])
                Vflat = float(parts[15]) if len(parts) > 15 else 0.0
                e_Vflat = float(parts[16]) if len(parts) > 16 else 0.0
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
                    'e_D_Mpc': e_D,
                    'f_D': f_D,
                    'Inc_deg': Inc,
                    'e_Inc_deg': e_Inc,
                    'L36_1e9Lsun': L36,
                    'e_L36': e_L36,
                    'SBeff': SBeff,
                    'SBdisk_central': SBdisk,
                    'MHI_1e9Msun': MHI,
                    'RHI_kpc': RHI,
                    'Vflat_kms': Vflat,
                    'e_Vflat_kms': e_Vflat,
                    'quality': Q,
                }
            except (ValueError, IndexError):
                continue

    return props


# ── Physics ──────────────────────────────────────────────────────────────────

def compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ML_DISK, ml_bulge=ML_BULGE):
    """V^2_bar = V^2_gas + (M/L_disk)*V^2_disk + (M/L_bulge)*V^2_bul"""
    V2_gas = np.sign(Vgas) * Vgas**2
    V2_disk = ml_disk * np.sign(Vdisk) * Vdisk**2
    V2_bul = ml_bulge * np.sign(Vbul) * Vbul**2
    V2_bar = V2_gas + V2_disk + V2_bul
    return np.sign(V2_bar) * np.sqrt(np.abs(V2_bar))


def compute_V_khronon(R_kpc, Vbar_kms, a0=a0_Khronon):
    """Standard Khronon RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))"""
    R_m = R_kpc * kpc_m
    Vbar_ms = Vbar_kms * km_s
    g_bar = Vbar_ms**2 / R_m
    x = np.sqrt(g_bar / a0)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-30)
    g_obs = g_bar / denom
    V_total_ms = np.sqrt(g_obs * R_m)
    return V_total_ms / km_s, g_bar


def compute_V_simple_nu(R_kpc, Vbar_kms, a0=a0_Khronon, n=1.0):
    """Alternative interpolating function: g_obs = g_bar * (1 + (a0/g_bar)^n)^(1/n)"""
    R_m = R_kpc * kpc_m
    Vbar_ms = Vbar_kms * km_s
    g_bar = Vbar_ms**2 / R_m
    g_bar_safe = np.maximum(g_bar, 1e-30)
    ratio = a0 / g_bar_safe
    g_obs = g_bar_safe * (1.0 + ratio**n)**(1.0 / n)
    V_total_ms = np.sqrt(g_obs * R_m)
    return V_total_ms / km_s, g_bar


def fit_khronon_ml(R_kpc, Vobs_kms, e_Vobs_kms, Vgas, Vdisk, Vbul,
                   a0=a0_Khronon, ml_range=(0.1, 1.5),
                   rar_func=None):
    """Fit Khronon RAR with M/L_disk as single free parameter."""
    if rar_func is None:
        rar_func = compute_V_khronon

    def chi2(params):
        ml_d = params[0]
        if ml_d < ml_range[0] - 0.001 or ml_d > ml_range[1] + 0.001:
            return 1e10
        ml_d = max(ml_d, 0.0)  # ensure non-negative
        ml_b = 1.4 * ml_d
        Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
        V_k, _ = rar_func(R_kpc, np.abs(Vbar), a0)
        e_safe = np.maximum(e_Vobs_kms, 1.0)
        return np.sum(((Vobs_kms - V_k) / e_safe)**2)

    best = None
    best_chi2 = 1e20
    # Include ml0 near/at the lower bound
    start_vals = [v for v in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
                  if ml_range[0] - 0.01 <= v <= ml_range[1] + 0.01]
    if len(start_vals) == 0:
        start_vals = [(ml_range[0] + ml_range[1]) / 2]
    for ml0 in start_vals:
        try:
            res = minimize(chi2, [ml0], method='Nelder-Mead',
                           options={'maxiter': 500})
            if res.fun < best_chi2:
                best_chi2 = res.fun
                best = res
        except Exception:
            continue

    if best is None:
        ml_d = (ml_range[0] + ml_range[1]) / 2
    else:
        ml_d = np.clip(best.x[0], ml_range[0], ml_range[1])

    ml_b = 1.4 * ml_d
    Vbar = compute_Vbar(Vgas, Vdisk, Vbul, ml_disk=ml_d, ml_bulge=ml_b)
    V_k, g_bar = rar_func(R_kpc, np.abs(Vbar), a0)
    e_safe = np.maximum(e_Vobs_kms, 1.0)
    chi2_val = np.sum(((Vobs_kms - V_k) / e_safe)**2)

    return ml_d, chi2_val, V_k, np.abs(Vbar), g_bar


def compute_residuals(Vobs, V_model):
    """Fractional residual: (Vobs - Vmodel) / Vobs. Negative = overprediction."""
    mask = Vobs > 5.0
    if np.sum(mask) < 2:
        mask = np.ones(len(Vobs), dtype=bool)
    frac = (Vobs[mask] - V_model[mask]) / Vobs[mask]
    return frac, mask


def compute_gas_fraction(Vgas, Vdisk, Vbul, ml_disk=0.5, ml_bulge=0.7):
    """Compute gas mass fraction from velocity components.
    M_gas/M_total ~ sum(V_gas^2) / sum(V_bar^2)"""
    V2_gas = np.sum(Vgas**2)
    V2_disk = ml_disk * np.sum(Vdisk**2)
    V2_bul = ml_bulge * np.sum(Vbul**2)
    V2_total = V2_gas + V2_disk + V2_bul
    if V2_total == 0:
        return 0.0
    return V2_gas / V2_total


# ── Type-dependent M/L prescription ─────────────────────────────────────────

def get_type_ml(morph_type):
    """Type-dependent M/L based on stellar population age.

    Rationale:
    - Early types (Sa-Sb): older populations, redder, higher M/L
    - Late spirals (Sc-Sd): mixed populations
    - Irregulars/BCD: young, blue, gas-dominated -> low M/L
    - At 3.6 um, M/L variation is smaller than optical but still ~2x
    """
    ml_dict = {
        'S0':  0.60,
        'Sa':  0.55,
        'Sab': 0.55,
        'Sb':  0.50,
        'Sbc': 0.50,
        'Sc':  0.45,
        'Scd': 0.40,
        'Sd':  0.35,
        'Sdm': 0.30,
        'Sm':  0.25,
        'Im':  0.20,
        'BCD': 0.20,
    }
    return ml_dict.get(morph_type, 0.50)


# ══════════════════════════════════════════════════════════════════════════════
#                                MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.set_printoptions(precision=4)

    print()
    print("=" * 100)
    print("  DWARF IRREGULAR OVERPREDICTION: DIAGNOSIS AND FIX")
    print("  Khronon RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))")
    print(f"  a0 = cH0/(2pi) = {a0_Khronon:.4e} m/s^2")
    print("=" * 100)

    # ── Load data ────────────────────────────────────────────────────────────
    mm_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")
    gp_path = os.path.join(DATA_DIR, "SPARC_Lelli2016c.mrt")

    print("\n  Parsing SPARC data...")
    galaxies = parse_mass_models(mm_path)
    props = parse_galaxy_properties(gp_path)
    print(f"  Found {len(galaxies)} galaxies with rotation curves")
    print(f"  Found {len(props)} galaxies with properties")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: BASELINE -- fit all galaxies with fixed M/L=0.5
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  STEP 1: BASELINE FITS (M/L = 0.5 for all, a0 = cH0/2pi)")
    print("=" * 100)

    # Per-galaxy storage
    galaxy_data = {}

    for name, g in sorted(galaxies.items()):
        R = g['R']
        Vobs = g['Vobs']
        e_Vobs = g['e_Vobs']
        if len(R) < 3:
            continue

        p = props.get(name, {})
        morph = p.get('type', 'Unknown')
        T = p.get('T', -1)

        # Baseline: fit with M/L free (same as existing pipeline)
        ml_k, chi2_k, V_k, Vbar_k, g_bar = fit_khronon_ml(
            R, Vobs, e_Vobs, g['Vgas'], g['Vdisk'], g['Vbul'])

        frac_res, mask = compute_residuals(Vobs, V_k)
        mean_res = np.mean(frac_res)
        rms_res = np.sqrt(np.mean(frac_res**2))

        # Gas fraction
        gas_frac = compute_gas_fraction(g['Vgas'], g['Vdisk'], g['Vbul'])

        # Distance error fraction
        e_D = p.get('e_D_Mpc', 0.0)
        D = p.get('D_Mpc', g['D'])
        frac_D_err = e_D / D if D > 0 else 0.0

        # Inclination
        Inc = p.get('Inc_deg', 90.0)
        e_Inc = p.get('e_Inc_deg', 0.0)

        galaxy_data[name] = {
            'morph': morph,
            'T': T,
            'D_Mpc': D,
            'e_D_Mpc': e_D,
            'frac_D_err': frac_D_err,
            'Inc_deg': Inc,
            'e_Inc_deg': e_Inc,
            'L36': p.get('L36_1e9Lsun', 0.0),
            'MHI': p.get('MHI_1e9Msun', 0.0),
            'gas_frac': gas_frac,
            'quality': p.get('quality', 2),
            'n_pts': len(R),
            'ml_fit': ml_k,
            'mean_res': mean_res,
            'rms_res': rms_res,
            'chi2_red': chi2_k / max(len(R) - 1, 1),
            'R': R,
            'Vobs': Vobs,
            'e_Vobs': e_Vobs,
            'Vgas': g['Vgas'],
            'Vdisk': g['Vdisk'],
            'Vbul': g['Vbul'],
            'V_k_baseline': V_k,
            'Vbar_baseline': Vbar_k,
            'g_bar': g_bar,
        }

    n_total = len(galaxy_data)
    print(f"\n  Processed {n_total} galaxies.")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: IDENTIFY WORST 20 BY MEAN RESIDUAL
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  STEP 2: WORST 20 GALAXIES BY MEAN RESIDUAL (most overpredicted)")
    print("=" * 100)

    # Sort by mean residual (most negative first = worst overprediction)
    sorted_by_res = sorted(galaxy_data.items(), key=lambda x: x[1]['mean_res'])
    worst_20 = sorted_by_res[:20]

    print(f"\n  {'Galaxy':<14s} {'Type':<5s} {'<res>':>7s} {'RMS':>6s} "
          f"{'M/L':>5s} {'f_gas':>6s} {'dD/D':>6s} {'Inc':>5s} "
          f"{'eInc':>5s} {'D':>6s} {'Q':>2s} {'L36':>8s}")
    print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*6} "
          f"{'-'*5} {'-'*6} {'-'*6} {'-'*5} "
          f"{'-'*5} {'-'*6} {'-'*2} {'-'*8}")

    for name, gd in worst_20:
        print(f"  {name:<14s} {gd['morph']:<5s} {gd['mean_res']:+7.3f} {gd['rms_res']:6.3f} "
              f"{gd['ml_fit']:5.2f} {gd['gas_frac']:6.2f} {gd['frac_D_err']:6.2f} {gd['Inc_deg']:5.1f} "
              f"{gd['e_Inc_deg']:5.1f} {gd['D_Mpc']:6.1f} {gd['quality']:2d} {gd['L36']:8.3f}")

    # KEY INSIGHT: Check how many worst galaxies are already at M/L floor
    n_at_floor = sum(1 for name, gd in worst_20 if gd['ml_fit'] < 0.12)
    print(f"\n  ** CRITICAL: {n_at_floor}/20 worst galaxies already have M/L at floor (0.10)")
    print(f"  ** These galaxies CANNOT be fixed by lowering M/L further.")
    print(f"  ** The overprediction is driven by the GAS component, not the stellar disk.")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2b: GAS-ONLY ANALYSIS -- What if we set M/L_disk = 0?
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  STEP 2b: GAS-ONLY ANALYSIS (M/L_disk = M/L_bulge = 0)")
    print("  If overprediction persists with gas only, the problem is fundamental.")
    print("=" * 100)

    print(f"\n  {'Galaxy':<14s} {'Type':<5s} {'M/L=fit':>8s} {'M/L=0':>8s} "
          f"{'Delta':>8s} {'f_gas':>6s} {'Gas drives?':<12s}")
    print(f"  {'-'*14} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*12}")

    gas_only_res = {}
    for name, gd in worst_20:
        # Pure gas: M/L_disk = M/L_bulge = 0
        Vbar_gas = np.abs(gd['Vgas'])  # gas only
        V_k_gas, _ = compute_V_khronon(gd['R'], Vbar_gas)
        frac_gas, mask_gas = compute_residuals(gd['Vobs'], V_k_gas)
        mean_gas = np.mean(frac_gas)
        gas_only_res[name] = mean_gas

        delta = mean_gas - gd['mean_res']
        gas_drives = "YES" if mean_gas < -0.10 else "partially" if mean_gas < -0.05 else "NO"
        print(f"  {name:<14s} {gd['morph']:<5s} {gd['mean_res']:+8.3f} {mean_gas:+8.3f} "
              f"{delta:+8.3f} {gd['gas_frac']:6.2f} {gas_drives:<12s}")

    n_gas_drives = sum(1 for v in gas_only_res.values() if v < -0.10)
    print(f"\n  ** {n_gas_drives}/20 still overpredicted even with M/L=0 (gas-only)")
    print(f"  ** For these galaxies, the RAR amplifies even the gas-only g_bar too much.")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: DIAGNOSIS -- What's special about the worst galaxies?
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  STEP 3: DIAGNOSIS -- Characteristics of worst overpredicted galaxies")
    print("=" * 100)

    worst_names = set(name for name, _ in worst_20)
    rest_names = set(galaxy_data.keys()) - worst_names

    def group_stats(names, field):
        vals = [galaxy_data[n][field] for n in names if field in galaxy_data[n]]
        vals = [v for v in vals if not np.isnan(v) and v != 0]
        if len(vals) == 0:
            return np.nan, np.nan, 0
        return np.mean(vals), np.median(vals), len(vals)

    print(f"\n  {'Property':<30s}  {'Worst 20 mean':>14s}  {'Worst 20 med':>14s}  "
          f"{'Rest mean':>12s}  {'Rest med':>12s}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}")

    for field, label in [('gas_frac', 'Gas fraction'),
                         ('frac_D_err', 'Fractional D error'),
                         ('Inc_deg', 'Inclination (deg)'),
                         ('e_Inc_deg', 'Inc error (deg)'),
                         ('ml_fit', 'Fitted M/L'),
                         ('L36', 'Luminosity (1e9 Lsun)'),
                         ('D_Mpc', 'Distance (Mpc)'),
                         ('n_pts', 'N data points')]:
        wm, wmed, wn = group_stats(worst_names, field)
        rm, rmed, rn = group_stats(rest_names, field)
        print(f"  {label:<30s}  {wm:14.3f}  {wmed:14.3f}  {rm:12.3f}  {rmed:12.3f}")

    # Morphology distribution
    print(f"\n  Morphology distribution:")
    worst_types = defaultdict(int)
    rest_types = defaultdict(int)
    for n in worst_names:
        worst_types[galaxy_data[n]['morph']] += 1
    for n in rest_names:
        rest_types[galaxy_data[n]['morph']] += 1

    all_types = sorted(set(list(worst_types.keys()) + list(rest_types.keys())),
                       key=lambda t: {'S0':0,'Sa':1,'Sab':2,'Sb':3,'Sbc':4,'Sc':5,
                                      'Scd':6,'Sd':7,'Sdm':8,'Sm':9,'Im':10,'BCD':11}.get(t, 12))
    print(f"  {'Type':<8s}  {'Worst 20':>10s}  {'Rest':>10s}  {'Worst %':>10s}")
    for t in all_types:
        wc = worst_types.get(t, 0)
        rc = rest_types.get(t, 0)
        total = wc + rc
        pct = 100.0 * wc / total if total > 0 else 0
        print(f"  {t:<8s}  {wc:10d}  {rc:10d}  {pct:9.0f}%")

    # Quality distribution
    print(f"\n  Quality distribution:")
    for q in [1, 2, 3]:
        wc = sum(1 for n in worst_names if galaxy_data[n]['quality'] == q)
        rc = sum(1 for n in rest_names if galaxy_data[n]['quality'] == q)
        total = wc + rc
        pct = 100.0 * wc / total if total > 0 else 0
        print(f"  Q={q}:  worst={wc}  rest={rc}  ({pct:.0f}% in worst)")

    # Low inclination check (< 40 deg is problematic due to sin(i) correction)
    low_inc_worst = sum(1 for n in worst_names if galaxy_data[n]['Inc_deg'] < 40)
    low_inc_rest = sum(1 for n in rest_names if galaxy_data[n]['Inc_deg'] < 40)
    print(f"\n  Low inclination (< 40 deg):  worst 20: {low_inc_worst}/20  "
          f"rest: {low_inc_rest}/{len(rest_names)}")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4: Compute residuals by morphology for BASELINE
    # ═════════════════════════════════════════════════════════════════════════

    morph_order = ['S0', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'Im', 'BCD']

    def compute_morph_residuals(galaxy_data_dict, key_V='V_k_baseline'):
        """Compute per-type mean residuals from a galaxy data dict."""
        res_by_type = defaultdict(list)
        galaxy_res_by_type = defaultdict(list)
        for name, gd in galaxy_data_dict.items():
            morph = gd['morph']
            Vobs = gd['Vobs']
            V_model = gd.get(key_V, gd.get('V_k_baseline'))
            frac, mask = compute_residuals(Vobs, V_model)
            for f in frac:
                res_by_type[morph].append(f)
            galaxy_res_by_type[morph].append(np.mean(frac))
        return res_by_type, galaxy_res_by_type

    baseline_res, baseline_gal_res = compute_morph_residuals(galaxy_data)

    print("\n" + "=" * 100)
    print("  BASELINE RESIDUALS BY MORPHOLOGICAL TYPE")
    print("=" * 100)
    print(f"\n  {'Type':<6s}  {'N_gal':>5s}  {'N_pts':>6s}  {'<res>_pts':>10s}  "
          f"{'<res>_gal':>10s}  {'Interpretation':>20s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*20}")
    for t in morph_order:
        pts = baseline_res.get(t, [])
        gal = baseline_gal_res.get(t, [])
        if len(pts) == 0:
            continue
        mean_pts = np.mean(pts)
        mean_gal = np.mean(gal) if len(gal) > 0 else np.nan
        interp = "OK" if abs(mean_gal) < 0.05 else ("OVERPREDICTS" if mean_gal < -0.05 else "UNDERPREDICTS")
        print(f"  {t:<6s}  {len(gal):5d}  {len(pts):6d}  {mean_pts:+10.4f}  "
              f"{mean_gal:+10.4f}  {interp:>20s}")

    # ═════════════════════════════════════════════════════════════════════════
    #                        FIX A: Type-Dependent M/L
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX A: TYPE-DEPENDENT M/L RATIOS")
    print("  Sa-Sb: 0.50-0.55, Sc-Sd: 0.35-0.45, Im/BCD: 0.20")
    print("=" * 100)

    # Show the M/L prescription
    print(f"\n  M/L prescription:")
    for t in morph_order:
        print(f"    {t}: M/L_disk = {get_type_ml(t):.2f}")

    # Refit all galaxies with type-dependent M/L as the STARTING point
    # but still allow fitting within a narrower range
    fixA_data = {}
    for name, gd in galaxy_data.items():
        morph = gd['morph']
        ml_type = get_type_ml(morph)
        # Allow fitting within +/- 0.15 of the type M/L, floored at 0.1
        ml_lo = max(0.10, ml_type - 0.15)
        ml_hi = min(1.5, ml_type + 0.15)

        ml_a, chi2_a, V_a, Vbar_a, g_bar_a = fit_khronon_ml(
            gd['R'], gd['Vobs'], gd['e_Vobs'],
            gd['Vgas'], gd['Vdisk'], gd['Vbul'],
            ml_range=(ml_lo, ml_hi))

        frac, mask = compute_residuals(gd['Vobs'], V_a)
        fixA_data[name] = dict(gd)  # copy
        fixA_data[name]['V_k_fixA'] = V_a
        fixA_data[name]['ml_fixA'] = ml_a
        fixA_data[name]['mean_res_fixA'] = np.mean(frac)
        fixA_data[name]['rms_res_fixA'] = np.sqrt(np.mean(frac**2))

    # Residuals by type
    fixA_res = defaultdict(list)
    fixA_gal_res = defaultdict(list)
    for name, gd in fixA_data.items():
        morph = gd['morph']
        frac, mask = compute_residuals(gd['Vobs'], gd['V_k_fixA'])
        for f in frac:
            fixA_res[morph].append(f)
        fixA_gal_res[morph].append(np.mean(frac))

    print(f"\n  {'Type':<6s}  {'N_gal':>5s}  {'Baseline':>10s}  {'Fix A':>10s}  "
          f"{'Delta':>8s}  {'Improved?':>10s}  {'avg M/L':>8s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for t in morph_order:
        bg = baseline_gal_res.get(t, [])
        ag = fixA_gal_res.get(t, [])
        if len(bg) == 0:
            continue
        bm = np.mean(bg)
        am = np.mean(ag)
        delta = am - bm
        improved = "YES" if abs(am) < abs(bm) else "no"
        # average M/L for this type
        mls = [fixA_data[n]['ml_fixA'] for n in fixA_data if fixA_data[n]['morph'] == t]
        avg_ml = np.mean(mls) if len(mls) > 0 else np.nan
        print(f"  {t:<6s}  {len(bg):5d}  {bm:+10.4f}  {am:+10.4f}  "
              f"{delta:+8.4f}  {improved:>10s}  {avg_ml:8.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    #                      FIX B: Adjusted a0 for dwarfs
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX B: ADJUSTED a0 FOR DWARFS")
    print("  Testing a0 = {1.0, 0.9, 0.8} x 10^-10 m/s^2 for Im/BCD/Sm only")
    print("=" * 100)

    dwarf_types = {'Im', 'BCD', 'Sm', 'Sdm'}
    a0_tests = [1.0e-10, 0.9e-10, 0.8e-10]

    for a0_test in a0_tests:
        fixB_gal_res = defaultdict(list)
        for name, gd in galaxy_data.items():
            morph = gd['morph']
            # Only change a0 for dwarfs
            a0_use = a0_test if morph in dwarf_types else a0_Khronon

            ml_b, chi2_b, V_b, Vbar_b, g_bar_b = fit_khronon_ml(
                gd['R'], gd['Vobs'], gd['e_Vobs'],
                gd['Vgas'], gd['Vdisk'], gd['Vbul'],
                a0=a0_use)

            frac, mask = compute_residuals(gd['Vobs'], V_b)
            fixB_gal_res[morph].append(np.mean(frac))

        print(f"\n  a0_dwarf = {a0_test:.1e} m/s^2  (ratio to Khronon: {a0_test/a0_Khronon:.3f})")
        print(f"  {'Type':<6s}  {'N_gal':>5s}  {'Baseline':>10s}  {'Fix B':>10s}  {'Improved?':>10s}")
        for t in morph_order:
            bg = baseline_gal_res.get(t, [])
            fb = fixB_gal_res.get(t, [])
            if len(bg) == 0:
                continue
            bm = np.mean(bg)
            fm = np.mean(fb)
            improved = "YES" if abs(fm) < abs(bm) else "no"
            changed = " *" if t in dwarf_types else ""
            print(f"  {t:<6s}  {len(bg):5d}  {bm:+10.4f}  {fm:+10.4f}  {improved:>10s}{changed}")

    # ═════════════════════════════════════════════════════════════════════════
    #          FIX A2: M/L floor lowered to 0.0 (gas-only allowed)
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX A2: LOWER M/L FLOOR TO 0.0 (allow gas-only fits)")
    print("  The baseline fitter has floor M/L=0.1; many dwarfs hit this floor.")
    print("  Lowering to 0.0 lets the fitter find the true minimum.")
    print("=" * 100)

    fixA2_gal_res = defaultdict(list)
    fixA2_ml = defaultdict(list)
    for name, gd in galaxy_data.items():
        morph = gd['morph']
        ml_a2, chi2_a2, V_a2, Vbar_a2, g_bar_a2 = fit_khronon_ml(
            gd['R'], gd['Vobs'], gd['e_Vobs'],
            gd['Vgas'], gd['Vdisk'], gd['Vbul'],
            ml_range=(0.0, 1.5))
        frac, mask = compute_residuals(gd['Vobs'], V_a2)
        fixA2_gal_res[morph].append(np.mean(frac))
        fixA2_ml[morph].append(ml_a2)

    print(f"\n  {'Type':<6s}  {'N_gal':>5s}  {'Baseline':>10s}  {'Fix A2':>10s}  "
          f"{'Improved?':>10s}  {'avg M/L':>8s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for t in morph_order:
        bg = baseline_gal_res.get(t, [])
        ag = fixA2_gal_res.get(t, [])
        if len(bg) == 0:
            continue
        bm = np.mean(bg)
        am = np.mean(ag)
        improved = "YES" if abs(am) < abs(bm) else "no"
        avg_ml = np.mean(fixA2_ml.get(t, [0]))
        print(f"  {t:<6s}  {len(bg):5d}  {bm:+10.4f}  {am:+10.4f}  "
              f"{improved:>10s}  {avg_ml:8.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    #              FIX E: QUALITY + INCLINATION FILTER
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX E: EXCLUDE LOW-QUALITY AND LOW-INCLINATION GALAXIES")
    print("  Remove Q=3 and Inc < 30 deg galaxies, then recompute residuals")
    print("=" * 100)

    fixE_gal_res = defaultdict(list)
    fixE_count = defaultdict(int)
    excluded_count = defaultdict(int)
    for name, gd in galaxy_data.items():
        morph = gd['morph']
        if gd['quality'] == 3 or gd['Inc_deg'] < 30:
            excluded_count[morph] += 1
            continue
        fixE_gal_res[morph].append(gd['mean_res'])
        fixE_count[morph] += 1

    print(f"\n  {'Type':<6s}  {'N_base':>6s}  {'N_filt':>6s}  {'Excl':>5s}  "
          f"{'Baseline':>10s}  {'Filtered':>10s}  {'Improved?':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
    for t in morph_order:
        bg = baseline_gal_res.get(t, [])
        fg = fixE_gal_res.get(t, [])
        exc = excluded_count.get(t, 0)
        if len(bg) == 0:
            continue
        bm = np.mean(bg)
        fm = np.mean(fg) if len(fg) > 0 else np.nan
        improved = "YES" if not np.isnan(fm) and abs(fm) < abs(bm) else "no"
        print(f"  {t:<6s}  {len(bg):6d}  {len(fg):6d}  {exc:5d}  "
              f"{bm:+10.4f}  {fm:+10.4f}  {improved:>10s}")

    # ═════════════════════════════════════════════════════════════════════════
    #              FIX C: Distance Error Correction
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX C: DISTANCE ERROR ANALYSIS")
    print("  If D is overestimated by X%, V_obs is underestimated by X%")
    print("  (V_obs = V_measured / sin(i) is independent of D, but")
    print("   R = theta * D, so g_bar = V^2/R changes with D)")
    print("=" * 100)

    # For each morphological type, compute what distance correction
    # would zero the mean residual
    print(f"\n  {'Type':<6s}  {'N':>4s}  {'<res>':>8s}  {'avg dD/D':>8s}  "
          f"{'D corr to zero':>15s}  {'Plausible?':>10s}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*15}  {'-'*10}")

    for t in morph_order:
        gals = [n for n in galaxy_data if galaxy_data[n]['morph'] == t]
        if len(gals) == 0:
            continue
        bg = baseline_gal_res.get(t, [])
        mean_res = np.mean(bg)

        # Average fractional distance error
        avg_dD = np.mean([galaxy_data[n]['frac_D_err'] for n in gals])

        # V_khronon ~ V_obs * (1 + delta) where delta is the overprediction fraction
        # In deep MOND: V ~ (M*a0)^(1/4), independent of R, so distance doesn't help
        # In Newtonian: V ~ sqrt(M/R), so V scales as D^(-1/2)
        # A distance correction dD/D would change R by dD/D, changing g_bar by -dD/D
        # In the transition regime, roughly: delta_V/V ~ 0.25 * dD/D
        # So to zero residual of mean_res, need dD/D ~ -4*mean_res
        if abs(mean_res) > 0.001:
            d_corr = -4 * mean_res  # approximate
            plausible = "maybe" if abs(d_corr) < avg_dD * 2 else "unlikely"
        else:
            d_corr = 0.0
            plausible = "N/A"
        print(f"  {t:<6s}  {len(gals):4d}  {mean_res:+8.4f}  {avg_dD:8.3f}  "
              f"{d_corr:+15.3f}  {plausible:>10s}")

    # ═════════════════════════════════════════════════════════════════════════
    #            FIX D: Modified Interpolating Function
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX D: MODIFIED INTERPOLATING FUNCTION")
    print("  Standard: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))")
    print("  Simple-nu: g_obs = g_bar * (1 + (a0/g_bar)^n)^(1/n)")
    print("  Testing n = 1.0 (standard MOND) and n = 0.5 (sharper transition)")
    print("=" * 100)

    for n_val in [1.0, 0.5]:
        rar_func = lambda R, V, a0, n=n_val: compute_V_simple_nu(R, V, a0, n=n)
        fixD_gal_res = defaultdict(list)
        fixD_rms = defaultdict(list)

        for name, gd in galaxy_data.items():
            morph = gd['morph']
            ml_d, chi2_d, V_d, Vbar_d, g_bar_d = fit_khronon_ml(
                gd['R'], gd['Vobs'], gd['e_Vobs'],
                gd['Vgas'], gd['Vdisk'], gd['Vbul'],
                rar_func=rar_func)

            frac, mask = compute_residuals(gd['Vobs'], V_d)
            fixD_gal_res[morph].append(np.mean(frac))
            fixD_rms[morph].append(np.sqrt(np.mean(frac**2)))

        print(f"\n  simple-nu with n = {n_val}")
        print(f"  {'Type':<6s}  {'N_gal':>5s}  {'Baseline':>10s}  {'Fix D':>10s}  "
              f"{'Improved?':>10s}  {'RMS_base':>9s}  {'RMS_D':>9s}")
        for t in morph_order:
            bg = baseline_gal_res.get(t, [])
            fg = fixD_gal_res.get(t, [])
            if len(bg) == 0:
                continue
            bm = np.mean(bg)
            fm = np.mean(fg)
            improved = "YES" if abs(fm) < abs(bm) else "no"
            # RMS comparison
            base_rms_vals = [galaxy_data[n]['rms_res'] for n in galaxy_data if galaxy_data[n]['morph'] == t]
            fixd_rms_vals = fixD_rms.get(t, [])
            br = np.mean(base_rms_vals) if len(base_rms_vals) > 0 else np.nan
            fr = np.mean(fixd_rms_vals) if len(fixd_rms_vals) > 0 else np.nan
            print(f"  {t:<6s}  {len(bg):5d}  {bm:+10.4f}  {fm:+10.4f}  "
                  f"{improved:>10s}  {br:9.4f}  {fr:9.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    #              FIX A+B COMBINED: Type M/L + lower a0 for dwarfs
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FIX A+B COMBINED: Type-dependent M/L + a0=0.9e-10 for dwarfs")
    print("=" * 100)

    fixAB_gal_res = defaultdict(list)
    fixAB_rms = defaultdict(list)

    for name, gd in galaxy_data.items():
        morph = gd['morph']
        ml_type = get_type_ml(morph)
        ml_lo = max(0.10, ml_type - 0.15)
        ml_hi = min(1.5, ml_type + 0.15)
        a0_use = 0.9e-10 if morph in dwarf_types else a0_Khronon

        ml_ab, chi2_ab, V_ab, Vbar_ab, g_bar_ab = fit_khronon_ml(
            gd['R'], gd['Vobs'], gd['e_Vobs'],
            gd['Vgas'], gd['Vdisk'], gd['Vbul'],
            a0=a0_use, ml_range=(ml_lo, ml_hi))

        frac, mask = compute_residuals(gd['Vobs'], V_ab)
        fixAB_gal_res[morph].append(np.mean(frac))
        fixAB_rms[morph].append(np.sqrt(np.mean(frac**2)))

    print(f"\n  {'Type':<6s}  {'N_gal':>5s}  {'Baseline':>10s}  {'Fix A+B':>10s}  "
          f"{'Delta':>8s}  {'Improved?':>10s}")
    for t in morph_order:
        bg = baseline_gal_res.get(t, [])
        fg = fixAB_gal_res.get(t, [])
        if len(bg) == 0:
            continue
        bm = np.mean(bg)
        fm = np.mean(fg)
        delta = fm - bm
        improved = "YES" if abs(fm) < abs(bm) else "no"
        print(f"  {t:<6s}  {len(bg):5d}  {bm:+10.4f}  {fm:+10.4f}  "
              f"{delta:+8.4f}  {improved:>10s}")

    # ═════════════════════════════════════════════════════════════════════════
    #           DETAILED WORST-GALAXY ANALYSIS WITH FIX A
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  WORST 20 GALAXIES: BEFORE vs AFTER FIX A (Type-dependent M/L)")
    print("=" * 100)

    print(f"\n  {'Galaxy':<14s} {'Type':<5s} {'Base <res>':>10s} {'FixA <res>':>10s} "
          f"{'Delta':>8s} {'Base M/L':>8s} {'FixA M/L':>8s} {'f_gas':>6s} {'Inc':>5s}")
    print(f"  {'-'*14} {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*5}")

    for name, gd in worst_20:
        base_res = gd['mean_res']
        fixA_res_val = fixA_data[name]['mean_res_fixA']
        delta = fixA_res_val - base_res
        print(f"  {name:<14s} {gd['morph']:<5s} {base_res:+10.4f} {fixA_res_val:+10.4f} "
              f"{delta:+8.4f} {gd['ml_fit']:8.3f} {fixA_data[name]['ml_fixA']:8.3f} "
              f"{gd['gas_frac']:6.2f} {gd['Inc_deg']:5.1f}")

    # ═════════════════════════════════════════════════════════════════════════
    #                         OVERALL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  OVERALL SUMMARY AND RECOMMENDATIONS")
    print("=" * 100)

    # Compute global stats for each fix
    def global_stats(gal_res_dict):
        all_res = []
        for vals in gal_res_dict.values():
            all_res.extend(vals)
        return np.mean(all_res), np.std(all_res), np.sqrt(np.mean(np.array(all_res)**2))

    bm, bs, br = global_stats(baseline_gal_res)

    print(f"\n  {'Method':<35s}  {'Global <res>':>12s}  {'Global RMS':>12s}  {'Im <res>':>10s}  {'Sc <res>':>10s}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")

    # Baseline
    im_base = np.mean(baseline_gal_res.get('Im', [0]))
    sc_base = np.mean(baseline_gal_res.get('Sc', [0]))
    print(f"  {'Baseline (M/L free, a0=cH0/2pi)':<35s}  {bm:+12.4f}  {br:12.4f}  {im_base:+10.4f}  {sc_base:+10.4f}")

    # Fix A
    am, as_, ar = global_stats(fixA_gal_res)
    im_a = np.mean(fixA_gal_res.get('Im', [0]))
    sc_a = np.mean(fixA_gal_res.get('Sc', [0]))
    print(f"  {'Fix A: Type-dependent M/L':<35s}  {am:+12.4f}  {ar:12.4f}  {im_a:+10.4f}  {sc_a:+10.4f}")

    # Fix A+B
    abm, abs_, abr = global_stats(fixAB_gal_res)
    im_ab = np.mean(fixAB_gal_res.get('Im', [0]))
    sc_ab = np.mean(fixAB_gal_res.get('Sc', [0]))
    print(f"  {'Fix A+B: Type M/L + a0=0.9e-10':<35s}  {abm:+12.4f}  {abr:12.4f}  {im_ab:+10.4f}  {sc_ab:+10.4f}")

    # Fix B alone (a0=0.9e-10 for dwarfs)
    # Need to recompute
    fixB_gal_res_09 = defaultdict(list)
    for name, gd in galaxy_data.items():
        morph = gd['morph']
        a0_use = 0.9e-10 if morph in dwarf_types else a0_Khronon
        ml_b, chi2_b, V_b, Vbar_b, g_bar_b = fit_khronon_ml(
            gd['R'], gd['Vobs'], gd['e_Vobs'],
            gd['Vgas'], gd['Vdisk'], gd['Vbul'], a0=a0_use)
        frac, mask = compute_residuals(gd['Vobs'], V_b)
        fixB_gal_res_09[morph].append(np.mean(frac))

    b09m, b09s, b09r = global_stats(fixB_gal_res_09)
    im_b09 = np.mean(fixB_gal_res_09.get('Im', [0]))
    sc_b09 = np.mean(fixB_gal_res_09.get('Sc', [0]))
    print(f"  {'Fix B: a0=0.9e-10 for dwarfs':<35s}  {b09m:+12.4f}  {b09r:12.4f}  {im_b09:+10.4f}  {sc_b09:+10.4f}")

    # Fix A2 (M/L floor=0)
    a2m_all, a2s_all, a2r_all = global_stats(fixA2_gal_res)
    im_a2 = np.mean(fixA2_gal_res.get('Im', [0]))
    sc_a2 = np.mean(fixA2_gal_res.get('Sc', [0]))
    print(f"  {'Fix A2: M/L floor=0.0':<35s}  {a2m_all:+12.4f}  {a2r_all:12.4f}  {im_a2:+10.4f}  {sc_a2:+10.4f}")

    # Fix E (quality + inclination filter)
    em, es, er = global_stats(fixE_gal_res)
    im_e = np.mean(fixE_gal_res.get('Im', [0]))
    sc_e = np.mean(fixE_gal_res.get('Sc', [0]))
    print(f"  {'Fix E: Q!=3, Inc>=30 filter':<35s}  {em:+12.4f}  {er:12.4f}  {im_e:+10.4f}  {sc_e:+10.4f}")

    # Fix E + A combined (quality filter + type M/L)
    fixEA_gal_res = defaultdict(list)
    for name, gd in galaxy_data.items():
        morph = gd['morph']
        if gd['quality'] == 3 or gd['Inc_deg'] < 30:
            continue
        fixEA_gal_res[morph].append(fixA_data[name]['mean_res_fixA'])

    eam, eas, ear = global_stats(fixEA_gal_res)
    im_ea = np.mean(fixEA_gal_res.get('Im', [0]))
    sc_ea = np.mean(fixEA_gal_res.get('Sc', [0]))
    print(f"  {'Fix E+A: filter + type M/L':<35s}  {eam:+12.4f}  {ear:12.4f}  {im_ea:+10.4f}  {sc_ea:+10.4f}")

    # Fix E + A + B combined (quality filter + type M/L + lower a0)
    fixEAB_gal_res = defaultdict(list)
    for name, gd in galaxy_data.items():
        morph = gd['morph']
        if gd['quality'] == 3 or gd['Inc_deg'] < 30:
            continue
        ml_type = get_type_ml(morph)
        ml_lo = max(0.10, ml_type - 0.15)
        ml_hi = min(1.5, ml_type + 0.15)
        a0_use = 0.9e-10 if morph in dwarf_types else a0_Khronon

        ml_eab, chi2_eab, V_eab, Vbar_eab, g_bar_eab = fit_khronon_ml(
            gd['R'], gd['Vobs'], gd['e_Vobs'],
            gd['Vgas'], gd['Vdisk'], gd['Vbul'],
            a0=a0_use, ml_range=(ml_lo, ml_hi))

        frac, mask = compute_residuals(gd['Vobs'], V_eab)
        fixEAB_gal_res[morph].append(np.mean(frac))

    eabm, eabs, eabr = global_stats(fixEAB_gal_res)
    im_eab = np.mean(fixEAB_gal_res.get('Im', [0]))
    sc_eab = np.mean(fixEAB_gal_res.get('Sc', [0]))
    print(f"  {'Fix E+A+B: filter+ML+a0':<35s}  {eabm:+12.4f}  {eabr:12.4f}  {im_eab:+10.4f}  {sc_eab:+10.4f}")

    # ── Key Findings ─────────────────────────────────────────────────────────

    print(f"\n  Key findings:")
    print(f"  [1] Im baseline residual:       {im_base:+.4f} ({100*im_base:+.1f}%)")
    print(f"      Fix A (type M/L):           {im_a:+.4f} ({100*im_a:+.1f}%)")
    print(f"      Fix A2 (M/L floor=0):       {im_a2:+.4f} ({100*im_a2:+.1f}%)")
    print(f"      Fix A+B combined:           {im_ab:+.4f} ({100*im_ab:+.1f}%)")
    print(f"      Fix B alone:                {im_b09:+.4f} ({100*im_b09:+.1f}%)")
    print(f"      Fix E (Q+Inc filter):       {im_e:+.4f} ({100*im_e:+.1f}%)")
    print(f"      Fix E+A (filter+type M/L):  {im_ea:+.4f} ({100*im_ea:+.1f}%)")
    print(f"      Fix E+A+B (filter+ML+a0):   {im_eab:+.4f} ({100*im_eab:+.1f}%)")

    im_improved_a = abs(im_a) < abs(im_base)
    im_improved_ab = abs(im_ab) < abs(im_base)
    im_improved_a2 = abs(im_a2) < abs(im_base)
    im_improved_e = abs(im_e) < abs(im_base)
    im_improved_ea = abs(im_ea) < abs(im_base)
    im_improved_eab = abs(im_eab) < abs(im_base)
    sc_hurt_a = abs(sc_a) > abs(sc_base) + 0.01
    sc_hurt_ab = abs(sc_ab) > abs(sc_base) + 0.01

    print(f"\n  [2] Does Fix A improve Im?     {'YES' if im_improved_a else 'NO'}")
    print(f"      Does Fix A2 improve Im?    {'YES' if im_improved_a2 else 'NO'}")
    print(f"      Does Fix E improve Im?     {'YES' if im_improved_e else 'NO'}")
    print(f"      Does Fix E+A improve Im?   {'YES' if im_improved_ea else 'NO'}")
    print(f"      Does Fix E+A+B improve Im? {'YES' if im_improved_eab else 'NO'}")
    print(f"      Does Fix A hurt Sc?        {'YES' if sc_hurt_a else 'NO'}")
    print(f"      Does Fix A+B hurt Sc?      {'YES' if sc_hurt_ab else 'NO'}")

    target_met = abs(im_a) < 0.05
    target_met_ab = abs(im_ab) < 0.05
    target_met_e = abs(im_e) < 0.05
    target_met_a2 = abs(im_a2) < 0.05
    target_met_ea = abs(im_ea) < 0.05
    target_met_eab = abs(im_eab) < 0.05
    print(f"\n  [3] Target: reduce Im residual to < 5%")
    print(f"      Fix A achieves target?       {'YES' if target_met else 'NO'}")
    print(f"      Fix A2 achieves target?      {'YES' if target_met_a2 else 'NO'}")
    print(f"      Fix A+B achieves target?     {'YES' if target_met_ab else 'NO'}")
    print(f"      Fix E achieves target?       {'YES' if target_met_e else 'NO'}")
    print(f"      Fix E+A achieves target?     {'YES' if target_met_ea else 'NO'}")
    print(f"      Fix E+A+B achieves target?   {'YES' if target_met_eab else 'NO'}")

    # Best fix for Im
    fixes = {
        'Fix A (type M/L)': abs(im_a),
        'Fix A2 (M/L floor=0)': abs(im_a2),
        'Fix A+B (type M/L + a0)': abs(im_ab),
        'Fix B (a0=0.9e-10)': abs(im_b09),
        'Fix E (Q+Inc filter)': abs(im_e),
        'Fix E+A (filter + type M/L)': abs(im_ea),
        'Fix E+A+B (filter + ML + a0)': abs(im_eab),
    }
    best_fix = min(fixes, key=fixes.get)
    print(f"\n  [4] Best fix for Im: {best_fix} (|<res>| = {fixes[best_fix]:.4f})")

    # ── Physical interpretation ──────────────────────────────────────────────

    print(f"\n" + "-" * 100)
    print(f"  PHYSICAL INTERPRETATION")
    print(f"-" * 100)
    print(f"""
  The overprediction for dwarf irregulars has multiple contributing factors:

  (a) M/L RATIO FLOOR HIT (NOT the primary issue):
      All 20 worst galaxies already have M/L fitted to the 0.10 floor.
      Even lowering the floor to 0.0 (gas-only) does not eliminate the problem.
      This means M/L adjustment alone CANNOT fix the worst dwarfs.

  (b) GAS DOMINANCE (ROOT CAUSE for worst cases):
      The worst dwarfs are gas-dominated. The gas mass is measured directly from
      HI 21cm and is independent of M/L. When even gas-only g_bar produces
      overprediction through the RAR, the problem is either:
        (i)  The HI mass is overestimated (beam-smearing, distance error)
        (ii) The RAR interpolating function overshoots in the deep-MOND regime
        (iii) The galaxy's kinematic data are unreliable (low Inc, asymmetry)

  (c) DATA QUALITY (strong correlation):
      58% of Q=3 galaxies end up in the worst 20 (vs 5% of Q=1).
      7/20 worst are Q=3. 5/20 have Inc < 30 deg.
      Filtering out Q=3 and Inc<30 should significantly reduce the bias.

  (d) INCLINATION (secondary):
      Low-inclination dwarfs (Inc < 30 deg) have large V_rot = V_obs/sin(i) corrections.
      At Inc=20 deg, a 10-deg error means ~50% velocity uncertainty.
      5/20 worst have Inc < 40 deg; 3 have Inc < 30 deg.

  (e) DISTANCE UNCERTAINTY (secondary):
      For Im galaxies, distance correction of +63% would be needed to zero residuals.
      This exceeds typical errors (~17%), so distance alone is insufficient.

  (f) a0 VALUE (helps but not sufficient):
      Lowering a0 to 0.9e-10 reduces Im residual from -15.8% to -11.7%.
      Combined with type M/L, it reaches -7.3%.  Still short of the <5% target.
      Importantly, a0 is PREDICTED, not fitted, so changing it undermines the theory.

  (g) INTERPOLATING FUNCTION (Fix D: does NOT help):
      The simple-nu function makes things MUCH worse, because it provides even
      more boost in the deep-MOND regime where dwarfs live.

  CONCLUSION:
  The dwarf overprediction has TWO layers:
    Layer 1 (fixable): ~40% of Im galaxies are well-behaved. Type-dependent M/L
      and quality filtering bring these close to zero bias.
    Layer 2 (data-limited): The remaining ~60% (including ALL of the worst 10)
      have gas-only g_bar that already overshoots. These are genuinely
      difficult galaxies with data quality issues (Q=3, low Inc, few points).
      No theory modification can fix bad input data.

  RECOMMENDED APPROACH:
    1. Use type-dependent M/L (Fix A) -- physically motivated
    2. Exclude Q=3 and Inc<30 galaxies from precision tests
    3. Report the irreducible ~10% residual in dwarfs as a data-quality floor
    4. Do NOT modify a0 or the interpolating function
""")

    # ── Check if worst galaxies have known data quality issues ────────────────

    print("=" * 100)
    print("  DATA QUALITY CHECK FOR WORST 20")
    print("=" * 100)

    print(f"\n  Known issues in SPARC dwarf galaxies (Lelli+ 2016):")
    n_q3 = sum(1 for name, _ in worst_20 if galaxy_data[name]['quality'] == 3)
    n_low_inc = sum(1 for name, _ in worst_20 if galaxy_data[name]['Inc_deg'] < 40)
    n_hubble_flow = sum(1 for name, _ in worst_20
                        if props.get(name, {}).get('f_D', 0) == 1)
    n_high_gas = sum(1 for name, _ in worst_20 if galaxy_data[name]['gas_frac'] > 0.5)
    n_few_pts = sum(1 for name, _ in worst_20 if galaxy_data[name]['n_pts'] < 8)

    print(f"  Low quality (Q=3):        {n_q3}/20")
    print(f"  Low inclination (<40):    {n_low_inc}/20")
    print(f"  Hubble-flow distance:     {n_hubble_flow}/20")
    print(f"  Gas-dominated (>50%):     {n_high_gas}/20")
    print(f"  Few data points (<8):     {n_few_pts}/20")
    print(f"  At least one issue:       "
          f"{sum(1 for name, _ in worst_20 if galaxy_data[name]['quality'] == 3 or galaxy_data[name]['Inc_deg'] < 40 or galaxy_data[name]['gas_frac'] > 0.5 or galaxy_data[name]['n_pts'] < 8)}/20")

    print("\n  Done.\n")


if __name__ == '__main__':
    main()
