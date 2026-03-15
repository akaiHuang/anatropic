#!/usr/bin/env python3
"""
Baryonic Tully-Fisher Relation (BTFR) from SPARC galaxy data.

Computes M_bar vs V_flat for SPARC galaxies, fits a power law via OLS and ODR,
and compares with the Khronon prediction: M_bar = V^4 / (G * a_0)
where a_0 = cH_0/(2*pi) ~ 1.13e-10 m/s^2.

References:
  - Lelli, McGaugh & Schombert 2016, ApJ, 816, L14  (BTFR)
  - McGaugh et al. 2000, ApJ, 533, L99               (original BTFR)
  - SPARC: Lelli, McGaugh & Schombert 2016, AJ, 152, 157
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.odr import ODR, Model, RealData

# ── Constants ────────────────────────────────────────────────────────────
G_SI   = 6.674e-11       # m^3 / (kg * s^2)
c_SI   = 2.998e8         # m/s
H0_SI  = 73e3 / 3.086e22 # 73 km/s/Mpc -> s^-1
Msun   = 1.989e30        # kg

# MOND acceleration scale from Khronon: a_0 = cH_0/(2*pi)
a0_Khronon = c_SI * H0_SI / (2 * np.pi)  # ~ 1.13e-10 m/s^2
a0_McGaugh = 1.2e-10     # m/s^2 (empirical)

# M/L ratios at 3.6 um
ML_DISK  = 0.5   # M_sun / L_sun (Schombert & McGaugh 2014)
ML_BULGE = 0.7

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "sparc")


# ── Parse SPARC galaxy properties ────────────────────────────────────────
def parse_galaxy_properties(filepath):
    """
    Parse SPARC_Lelli2016c.mrt using whitespace splitting.

    Column mapping (from split):
      [0] Name, [1] T, [2] D, [3] e_D, [4] f_D, [5] Inc, [6] e_Inc,
      [7] L[3.6] (1e9 L_sun), [8] e_L[3.6], [9] Reff, [10] SBeff,
      [11] Rdisk, [12] SBdisk, [13] MHI (1e9 M_sun), [14] RHI,
      [15] Vflat (km/s), [16] e_Vflat, [17] Q, [18] Ref
    """
    props = {}
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(('Title', 'Authors', 'Table',
                '=', '-', 'Byte', 'Note')):
                continue
            parts = stripped.split()
            if len(parts) < 18:
                continue
            try:
                name    = parts[0]
                T       = int(parts[1])
                D       = float(parts[2])
                Inc     = float(parts[5])
                L36     = float(parts[7])    # 1e9 L_sun
                e_L36   = float(parts[8])    # 1e9 L_sun
                MHI     = float(parts[13])   # 1e9 M_sun
                Vflat   = float(parts[15])   # km/s
                e_Vflat = float(parts[16])   # km/s
                Q       = int(parts[17])
            except (ValueError, IndexError):
                continue

            props[name] = {
                'T': T, 'D_Mpc': D, 'Inc': Inc,
                'L36_1e9Lsun': L36, 'e_L36_1e9Lsun': e_L36,
                'MHI_1e9Msun': MHI,
                'Vflat_kms': Vflat, 'e_Vflat_kms': e_Vflat,
                'Q': Q,
            }
    return props


# ── Parse Mass Models (rotation curves) ─────────────────────────────────
def parse_mass_models(filepath):
    """Parse MassModels_Lelli2016c.mrt."""
    galaxies = {}
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.rstrip('\n')
            if len(stripped) < 60:
                continue
            try:
                name   = stripped[0:11].strip()
                Vobs   = float(stripped[26:32].strip())
                e_Vobs = float(stripped[33:38].strip())
            except (ValueError, IndexError):
                continue
            if name not in galaxies:
                galaxies[name] = {'Vobs': [], 'e_Vobs': []}
            galaxies[name]['Vobs'].append(Vobs)
            galaxies[name]['e_Vobs'].append(e_Vobs)

    for name in galaxies:
        galaxies[name]['Vobs'] = np.array(galaxies[name]['Vobs'])
        galaxies[name]['e_Vobs'] = np.array(galaxies[name]['e_Vobs'])
    return galaxies


# ── Compute baryonic mass ────────────────────────────────────────────────
def compute_Mbar(L36_1e9Lsun, MHI_1e9Msun, ml_disk=ML_DISK):
    """
    M_bar = M_star + M_gas
    M_star = Upsilon_disk * L_[3.6]     (L36 in 1e9 L_sun)
    M_gas  = 1.33 * M_HI               (MHI in 1e9 M_sun; factor 1.33 for He)
    Returns M_bar in M_sun.
    """
    M_star = ml_disk * L36_1e9Lsun * 1e9
    M_gas  = 1.33 * MHI_1e9Msun * 1e9
    return M_star + M_gas


# ── Get V_flat from rotation curve ───────────────────────────────────────
def get_Vflat_from_curve(Vobs, e_Vobs, n_avg=3):
    """Average the last n_avg points of the rotation curve."""
    n = len(Vobs)
    if n == 0:
        return 0.0, 0.0
    k = min(n_avg, n)
    vflat = np.mean(Vobs[-k:])
    e_vflat = np.sqrt(np.mean(e_Vobs[-k:]**2)) / np.sqrt(k)
    return vflat, e_vflat


# ── ODR fit ──────────────────────────────────────────────────────────────
def power_law_log(B, x):
    """log10(Mbar) = slope * log10(V) + intercept"""
    return B[0] * x + B[1]


def fit_btfr_odr(log_V, log_M, e_log_V, e_log_M):
    """Orthogonal Distance Regression in log-log space."""
    data = RealData(log_V, log_M, sx=e_log_V, sy=e_log_M)
    model = Model(power_law_log)
    odr = ODR(data, model, beta0=[4.0, 2.0])
    return odr.run()


def fit_btfr_ols(log_V, log_M):
    """Simple OLS fit in log-log space."""
    coeffs = np.polyfit(log_V, log_M, 1)
    slope, intercept = coeffs
    residuals = log_M - (slope * log_V + intercept)
    scatter = np.std(residuals)
    return slope, intercept, scatter, residuals


# ── Analysis helper ──────────────────────────────────────────────────────
def analyze_sample(label, log_V, log_M, e_log_V, e_log_M, log_prefactor_khronon):
    """Run OLS + ODR fits and Khronon comparison on a subsample."""
    N = len(log_V)
    if N < 5:
        print(f"    {label}: only {N} galaxies, skipping")
        return None

    slope_ols, intercept_ols, scatter_ols, _ = fit_btfr_ols(log_V, log_M)

    odr_out = fit_btfr_odr(log_V, log_M, e_log_V, e_log_M)
    slope_odr   = odr_out.beta[0]
    inter_odr   = odr_out.beta[1]
    e_slope     = odr_out.sd_beta[0]
    e_inter     = odr_out.sd_beta[1]
    residuals   = log_M - (slope_odr * log_V + inter_odr)
    scatter_odr = np.std(residuals)

    # Residuals vs Khronon (slope=4)
    res_khronon = log_M - (4.0 * log_V + log_prefactor_khronon)
    scatter_k   = np.std(res_khronon)
    offset_k    = np.mean(res_khronon)

    result = {
        'N': N, 'slope_ols': slope_ols, 'intercept_ols': intercept_ols,
        'scatter_ols': scatter_ols,
        'slope_odr': slope_odr, 'e_slope_odr': e_slope,
        'intercept_odr': inter_odr, 'e_intercept_odr': e_inter,
        'scatter_odr': scatter_odr,
        'khronon_scatter': scatter_k, 'khronon_offset': offset_k,
    }

    delta = slope_odr - 4.0
    sigma = abs(delta) / e_slope if e_slope > 0 else float('inf')

    print(f"    {label} (N={N}):")
    print(f"      OLS slope  = {slope_ols:.3f},  scatter = {scatter_ols:.3f} dex")
    print(f"      ODR slope  = {slope_odr:.3f} +/- {e_slope:.3f}")
    print(f"      ODR inter  = {inter_odr:.3f} +/- {e_inter:.3f}")
    print(f"      ODR scatter= {scatter_odr:.3f} dex")
    print(f"      |slope-4|  = {abs(delta):.3f}  ({sigma:.1f} sigma from 4.0)")
    print(f"      vs Khronon: offset={offset_k:+.3f} dex, scatter={scatter_k:.3f} dex")
    print()

    return result


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 74)
    print("  BARYONIC TULLY-FISHER RELATION (BTFR) from SPARC")
    print("=" * 74)
    print()

    # ── Parse data ──
    gp_path = os.path.join(DATA_DIR, "SPARC_Lelli2016c.mrt")
    mm_path = os.path.join(DATA_DIR, "MassModels_Lelli2016c.mrt")

    props = parse_galaxy_properties(gp_path)
    galaxies = parse_mass_models(mm_path)

    print(f"  Galaxies with properties:       {len(props)}")
    print(f"  Galaxies with rotation curves:   {len(galaxies)}")
    print()

    # ── Constants ──
    print("  Physical constants:")
    print(f"    G              = {G_SI:.4e} m^3/(kg s^2)")
    print(f"    H_0            = 73 km/s/Mpc = {H0_SI:.4e} s^-1")
    print(f"    a_0(Khronon)   = c*H_0/(2*pi) = {a0_Khronon:.4e} m/s^2")
    print(f"    a_0(McGaugh)   = {a0_McGaugh:.2e} m/s^2")
    print(f"    Ratio a0_K/a0_M= {a0_Khronon / a0_McGaugh:.3f}")
    print(f"    Upsilon_disk   = {ML_DISK} M_sun/L_sun")
    print(f"    Upsilon_bulge  = {ML_BULGE} M_sun/L_sun")
    print()

    # ── Khronon prefactors ──
    # V^4 = G * M_bar * a_0  =>  M_bar = V^4 / (G * a_0)
    # In solar units:  M_bar/M_sun = (V_kms)^4 * (1e3)^4 / (G * a_0 * M_sun)
    prefactor_K  = (1e3)**4 / (G_SI * a0_Khronon * Msun)
    prefactor_M  = (1e3)**4 / (G_SI * a0_McGaugh * Msun)
    log_pref_K   = np.log10(prefactor_K)
    log_pref_M   = np.log10(prefactor_M)

    print("  Khronon BTFR prediction:")
    print(f"    M_bar = V^4 / (G * a_0)")
    print(f"    1/(G*a_0)     = {1/(G_SI*a0_Khronon):.4e} kg/(m/s)^4")
    print(f"    Prefactor     = {prefactor_K:.2f} M_sun/(km/s)^4")
    print(f"    log10(pref)   = {log_pref_K:.4f}")
    print(f"    => log10(M_bar/M_sun) = 4.000 * log10(V_flat/km s^-1) + {log_pref_K:.3f}")
    print()
    print(f"  McGaugh empirical:")
    print(f"    Prefactor     = {prefactor_M:.2f} M_sun/(km/s)^4")
    print(f"    log10(pref)   = {log_pref_M:.4f}")
    print()
    print(f"  Note: The user-stated value 47.3 corresponds to a_0 ~ 1.59e-10 m/s^2.")
    print(f"  Correct values: {prefactor_K:.1f} (Khronon), {prefactor_M:.1f} (McGaugh)")
    print()

    # ── Build BTFR data for ALL 175 galaxies ──
    # Collect data with multiple flags
    all_names  = []
    all_logV   = []
    all_logM   = []
    all_elogV  = []
    all_elogM  = []
    all_Vflat  = []
    all_Mbar   = []
    all_Q      = []
    all_Inc    = []
    all_has_catalog_Vf = []

    n_from_catalog = 0
    n_from_curve   = 0
    n_skipped      = 0

    for name, p in sorted(props.items()):
        L36 = p['L36_1e9Lsun']
        MHI = p['MHI_1e9Msun']
        Q   = p['Q']
        Inc = p['Inc']

        Mbar = compute_Mbar(L36, MHI, ml_disk=ML_DISK)
        if Mbar <= 0:
            n_skipped += 1
            continue

        # Get V_flat: prefer catalog, fallback to rotation curve
        Vf    = p['Vflat_kms']
        e_Vf  = p['e_Vflat_kms']
        has_cat = False

        if Vf > 0 and e_Vf > 0:
            n_from_catalog += 1
            has_cat = True
        elif name in galaxies:
            g = galaxies[name]
            Vf, e_Vf = get_Vflat_from_curve(g['Vobs'], g['e_Vobs'])
            if Vf <= 0:
                n_skipped += 1
                continue
            if e_Vf <= 0:
                e_Vf = 5.0
            n_from_curve += 1
        else:
            n_skipped += 1
            continue

        if Vf <= 0:
            n_skipped += 1
            continue

        # Errors in log space
        e_Mbar = Mbar * 0.15   # ~15% from M/L uncertainty
        e_lM   = max(e_Mbar / (Mbar * np.log(10)), 0.01)
        e_lV   = max(e_Vf / (Vf * np.log(10)), 0.01)

        all_names.append(name)
        all_Vflat.append(Vf)
        all_Mbar.append(Mbar)
        all_logV.append(np.log10(Vf))
        all_logM.append(np.log10(Mbar))
        all_elogV.append(e_lV)
        all_elogM.append(e_lM)
        all_Q.append(Q)
        all_Inc.append(Inc)
        all_has_catalog_Vf.append(has_cat)

    all_logV   = np.array(all_logV)
    all_logM   = np.array(all_logM)
    all_elogV  = np.array(all_elogV)
    all_elogM  = np.array(all_elogM)
    all_Vflat  = np.array(all_Vflat)
    all_Mbar   = np.array(all_Mbar)
    all_Q      = np.array(all_Q)
    all_Inc    = np.array(all_Inc)
    all_has_cat = np.array(all_has_catalog_Vf)

    N = len(all_names)
    print("-" * 74)
    print("  DATA SUMMARY")
    print("-" * 74)
    print(f"  Total galaxies in sample:  {N}")
    print(f"    V_flat from SPARC catalog: {n_from_catalog}")
    print(f"    V_flat from curve (last 3): {n_from_curve}")
    print(f"    Skipped (no data):          {n_skipped}")
    print(f"  Quality: Q=1: {np.sum(all_Q==1)}, Q=2: {np.sum(all_Q==2)}, "
          f"Q=3: {np.sum(all_Q==3)}")
    print(f"  Inclination range: {np.min(all_Inc):.0f} - {np.max(all_Inc):.0f} deg")
    print(f"  V_flat range:  {np.min(all_Vflat):.1f} - {np.max(all_Vflat):.1f} km/s")
    print(f"  M_bar range:   {np.min(all_Mbar):.2e} - {np.max(all_Mbar):.2e} M_sun")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # FIT RESULTS FOR MULTIPLE SAMPLES
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 74)
    print("  FIT RESULTS")
    print("=" * 74)
    print()

    # Sample A: ALL 175 galaxies
    res_all = analyze_sample(
        "A: ALL galaxies",
        all_logV, all_logM, all_elogV, all_elogM, log_pref_K)

    # Sample B: Only galaxies with catalog Vflat (135)
    mask_cat = all_has_cat
    res_cat = analyze_sample(
        "B: Catalog V_flat only",
        all_logV[mask_cat], all_logM[mask_cat],
        all_elogV[mask_cat], all_elogM[mask_cat], log_pref_K)

    # Sample C: Catalog Vflat + Q<=2 (standard quality cut)
    mask_cq2 = mask_cat & (all_Q <= 2)
    res_cq2 = analyze_sample(
        "C: Catalog V_flat + Q<=2",
        all_logV[mask_cq2], all_logM[mask_cq2],
        all_elogV[mask_cq2], all_elogM[mask_cq2], log_pref_K)

    # Sample D: Catalog Vflat + Q<=2 + Inc>=30 (standard BTFR sample)
    mask_std = mask_cat & (all_Q <= 2) & (all_Inc >= 30)
    res_std = analyze_sample(
        "D: STANDARD (catalog V_flat + Q<=2 + Inc>=30)",
        all_logV[mask_std], all_logM[mask_std],
        all_elogV[mask_std], all_elogM[mask_std], log_pref_K)

    # Sample E: Q=1 only (highest quality)
    mask_q1 = mask_cat & (all_Q == 1)
    res_q1 = analyze_sample(
        "E: Q=1 high-quality only (catalog V_flat)",
        all_logV[mask_q1], all_logM[mask_q1],
        all_elogV[mask_q1], all_elogM[mask_q1], log_pref_K)

    # Sample F: Q=1 + Inc>=30
    mask_q1i = mask_cat & (all_Q == 1) & (all_Inc >= 30)
    res_q1i = analyze_sample(
        "F: Q=1 + Inc>=30 (cleanest sample)",
        all_logV[mask_q1i], all_logM[mask_q1i],
        all_elogV[mask_q1i], all_elogM[mask_q1i], log_pref_K)

    # ── McGaugh comparison for standard sample ──
    if res_std:
        m = mask_std
        res_mc = all_logM[m] - (4.0 * all_logV[m] + log_pref_M)
        print(f"    McGaugh (a_0=1.2e-10) on standard sample:")
        print(f"      offset={np.mean(res_mc):+.3f} dex, scatter={np.std(res_mc):.3f} dex")
        print()

    # ── Normalization comparison at V_flat = 100 km/s ──
    print("-" * 74)
    print("  NORMALIZATION AT V_flat = 100 km/s")
    print("-" * 74)
    V_ref = 100.0
    if res_std:
        M_fit  = 10**(res_std['slope_odr'] * np.log10(V_ref) + res_std['intercept_odr'])
        print(f"    ODR fit (std):  M_bar = {M_fit:.3e} M_sun")
    M_khr = prefactor_K * V_ref**4
    M_mcg = prefactor_M * V_ref**4
    print(f"    Khronon:        M_bar = {M_khr:.3e} M_sun")
    print(f"    McGaugh:        M_bar = {M_mcg:.3e} M_sun")
    print()

    # ── Sample galaxies ──
    # Use standard sample for display
    print("-" * 74)
    print("  TOP 20 GALAXIES (standard sample, by V_flat)")
    print("-" * 74)
    print(f"  {'Name':12s} {'V_flat':>8s} {'logV':>6s} {'logM_bar':>9s} "
          f"{'logM_K':>8s} {'Delta':>7s} {'Q':>3s} {'Inc':>5s}")
    print(f"  {'':12s} {'(km/s)':>8s} {'':>6s} {'':>9s} "
          f"{'(Khronon)':>8s} {'(dex)':>7s} {'':>3s} {'(deg)':>5s}")

    idx_std = np.where(mask_std)[0]
    Vflat_std = all_Vflat[idx_std]
    order = np.argsort(-Vflat_std)
    for rank in range(min(20, len(order))):
        j = idx_std[order[rank]]
        log_pred = 4.0 * all_logV[j] + log_pref_K
        delta = all_logM[j] - log_pred
        print(f"  {all_names[j]:12s} {all_Vflat[j]:8.1f} {all_logV[j]:6.2f} "
              f"{all_logM[j]:9.3f} {log_pred:8.3f} {delta:+7.3f} "
              f"{all_Q[j]:3d} {all_Inc[j]:5.0f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 74)
    print("  COMPARISON TABLE")
    print("=" * 74)
    print()
    print(f"  {'Sample':<40s} {'N':>4s} {'slope':>8s} {'e_slope':>8s} "
          f"{'scatter':>8s} {'|s-4|/e':>8s}")
    print(f"  {'-'*40} {'---':>4s} {'-------':>8s} {'-------':>8s} "
          f"{'-------':>8s} {'-------':>8s}")

    for label, res in [
        ("A: All galaxies", res_all),
        ("B: Catalog V_flat", res_cat),
        ("C: Catalog + Q<=2", res_cq2),
        ("D: STANDARD (cat+Q<=2+Inc>=30)", res_std),
        ("E: Q=1 (catalog V_flat)", res_q1),
        ("F: Q=1 + Inc>=30", res_q1i),
    ]:
        if res is None:
            continue
        sig = abs(res['slope_odr'] - 4.0) / res['e_slope_odr']
        print(f"  {label:<40s} {res['N']:4d} {res['slope_odr']:8.3f} "
              f"{res['e_slope_odr']:8.3f} {res['scatter_odr']:8.3f} "
              f"{sig:8.1f}")

    print()
    print(f"  {'Lelli+ 2016 (literature)':<40s} {'~153':>4s} {'3.850':>8s} "
          f"{'0.090':>8s} {'~0.11':>8s} {'1.7':>8s}")
    print(f"  {'Khronon prediction':<40s} {'---':>4s} {'4.000':>8s} "
          f"{'(exact)':>8s} {'---':>8s} {'---':>8s}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 74)
    print("  FINAL SUMMARY")
    print("=" * 74)

    best = res_std if res_std else res_cat
    best_label = "D (standard)" if res_std else "B (catalog)"

    print(f"  Best sample: {best_label}")
    print(f"    N                = {best['N']}")
    print(f"    ODR slope        = {best['slope_odr']:.3f} +/- {best['e_slope_odr']:.3f}")
    print(f"    ODR intercept    = {best['intercept_odr']:.3f} +/- {best['e_intercept_odr']:.3f}")
    print(f"    ODR scatter      = {best['scatter_odr']:.3f} dex")
    print(f"    OLS slope        = {best['slope_ols']:.3f}")
    print()
    sig_best = abs(best['slope_odr'] - 4.0) / best['e_slope_odr']
    print(f"    Deviation from 4.0: {best['slope_odr'] - 4.0:+.3f} ({sig_best:.1f} sigma)")
    print()
    print(f"  Khronon prediction:")
    print(f"    a_0              = cH_0/(2*pi) = {a0_Khronon:.4e} m/s^2")
    print(f"    Predicted slope  = 4.000")
    print(f"    Prefactor        = {prefactor_K:.2f} M_sun/(km/s)^4")
    print(f"    Mean offset      = {best['khronon_offset']:+.3f} dex")
    print(f"    Scatter          = {best['khronon_scatter']:.3f} dex")
    print()
    print(f"  Literature (Lelli+ 2016): slope = 3.85 +/- 0.09")
    print(f"  McGaugh empirical a_0    = {a0_McGaugh:.2e} m/s^2")
    print(f"  McGaugh prefactor        = {prefactor_M:.2f} M_sun/(km/s)^4")
    print("=" * 74)

    # ── Save CSV ──
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "btfr_data.csv")
    with open(csv_path, 'w') as f:
        f.write("# BTFR data from SPARC (175 galaxies)\n")
        if best:
            f.write(f"# ODR fit (std): slope={best['slope_odr']:.3f}"
                    f"+/-{best['e_slope_odr']:.3f}, "
                    f"intercept={best['intercept_odr']:.3f}\n")
        f.write(f"# Khronon: slope=4.000, log10(pref)={log_pref_K:.4f}\n")
        f.write("# Name,Vflat_kms,logV,Mbar_Msun,logM,logM_Khronon,"
                "residual_dex,Q,Inc,has_catalog_Vf\n")
        for i in range(N):
            log_pred = 4.0 * all_logV[i] + log_pref_K
            resid = all_logM[i] - log_pred
            f.write(f"{all_names[i]},{all_Vflat[i]:.2f},{all_logV[i]:.4f},"
                    f"{all_Mbar[i]:.4e},{all_logM[i]:.4f},"
                    f"{log_pred:.4f},{resid:.4f},"
                    f"{all_Q[i]},{all_Inc[i]:.0f},"
                    f"{int(all_has_cat[i])}\n")
    print(f"\n  CSV saved: {csv_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # -- Left: BTFR --
        ax = axes[0]
        cmap = {1: 'C0', 2: 'C1', 3: 'C2'}
        for q in [3, 2, 1]:  # plot Q=1 on top
            mask = (all_Q == q) & mask_cat
            ax.scatter(all_logV[mask], all_logM[mask],
                       c=cmap[q], s=12, alpha=0.6,
                       label=f'Q={q} (N={np.sum(mask)})', zorder=q)
        # Galaxies without catalog Vflat
        mask_nc = ~mask_cat
        if np.any(mask_nc):
            ax.scatter(all_logV[mask_nc], all_logM[mask_nc],
                       c='gray', s=8, alpha=0.3, marker='x',
                       label=f'No catalog V_flat (N={np.sum(mask_nc)})', zorder=0)

        xfit = np.linspace(np.min(all_logV)-0.05, np.max(all_logV)+0.05, 100)
        if best:
            ax.plot(xfit, best['slope_odr'] * xfit + best['intercept_odr'],
                    'k-', lw=2.0,
                    label=f'ODR fit: slope={best["slope_odr"]:.2f}')
        ax.plot(xfit, 4.0 * xfit + log_pref_K, 'r--', lw=2.0,
                label=f'Khronon: slope=4, a$_0$={a0_Khronon:.2e}')
        ax.plot(xfit, 4.0 * xfit + log_pref_M, 'b:', lw=1.5,
                label=f'McGaugh: slope=4, a$_0$=1.2e-10')

        ax.set_xlabel(r'$\log_{10}(V_{\rm flat}$ / km s$^{-1})$', fontsize=12)
        ax.set_ylabel(r'$\log_{10}(M_{\rm bar}$ / M$_\odot)$', fontsize=12)
        ax.set_title('Baryonic Tully-Fisher Relation (SPARC)', fontsize=13)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

        # -- Right: Residuals vs Khronon --
        ax2 = axes[1]
        log_pred_all = 4.0 * all_logV + log_pref_K
        res_all_k = all_logM - log_pred_all
        for q in [3, 2, 1]:
            mask = (all_Q == q) & mask_cat
            ax2.scatter(all_logV[mask], res_all_k[mask],
                        c=cmap[q], s=12, alpha=0.6, label=f'Q={q}', zorder=q)
        if np.any(mask_nc):
            ax2.scatter(all_logV[mask_nc], res_all_k[mask_nc],
                        c='gray', s=8, alpha=0.3, marker='x',
                        label='No catalog V_flat', zorder=0)
        ax2.axhline(0, color='r', ls='--', lw=1.5, label='Khronon (slope=4)')
        if best:
            ax2.axhline(best['khronon_offset'], color='k', ls=':', lw=1,
                        label=f'Mean offset: {best["khronon_offset"]:+.2f} dex')
        ax2.set_xlabel(r'$\log_{10}(V_{\rm flat}$ / km s$^{-1})$', fontsize=12)
        ax2.set_ylabel(r'$\Delta \log_{10}(M_{\rm bar})$ vs Khronon (dex)',
                       fontsize=12)
        ax2.set_title('Residuals vs Khronon prediction', fontsize=13)
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "btfr_sparc.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {plot_path}")
        plt.close()

    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    print("\n  Done.")


if __name__ == '__main__':
    main()
