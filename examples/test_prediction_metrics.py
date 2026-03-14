#!/usr/bin/env python3
"""
test_prediction_metrics.py
==========================
Three quantitative tests on SPARC rotation curve predictions:
  1. Prediction Accuracy (sigma-band fractions)
  2. RAR Scatter (dex)
  3. Parameter Efficiency (chi2 per parameter, reduced chi2)

Uses pre-computed data from sparc_rotation_curves.json.
"""

import json
import numpy as np
import os

# ── Load data ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "web", "data", "sparc_rotation_curves.json"
)
with open(DATA_PATH) as f:
    data = json.load(f)

metadata = data["metadata"]
galaxies = data["galaxies"]
a0 = metadata["a0_khronon_m_s2"]  # 1.13e-10 m/s^2
n_gal = len(galaxies)

print("=" * 72)
print("  SPARC Rotation Curve — Prediction Metrics")
print(f"  {n_gal} galaxies | a0 = {a0:.4e} m/s^2")
print("=" * 72)

# =====================================================================
# TEST 1: Prediction Accuracy  (sigma-band fractions)
# =====================================================================
print("\n" + "─" * 72)
print("  TEST 1: Prediction Accuracy (fraction within Nσ)")
print("─" * 72)

pulls_khr = []  # |V_pred - V_obs| / max(σ, 1)
pulls_nfw = []

for g in galaxies:
    Vobs = np.array(g["Vobs"])
    eVobs = np.array(g["e_Vobs"])
    Vkhr = np.array(g["V_khronon"])
    Vnfw = np.array(g["V_nfw"])

    sigma = np.maximum(eVobs, 1.0)  # floor at 1 km/s

    pulls_khr.extend(np.abs(Vkhr - Vobs) / sigma)
    pulls_nfw.extend(np.abs(Vnfw - Vobs) / sigma)

pulls_khr = np.array(pulls_khr)
pulls_nfw = np.array(pulls_nfw)
n_total = len(pulls_khr)

print(f"\n  Total data points: {n_total}")
print(f"\n  {'Band':<8} {'Khronon':>10} {'NFW':>10} {'Ideal (Gaussian)':>18}")
print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*18}")
for nsig, ideal in [(1, 68.27), (2, 95.45), (3, 99.73)]:
    frac_k = 100.0 * np.mean(pulls_khr <= nsig)
    frac_n = 100.0 * np.mean(pulls_nfw <= nsig)
    print(f"  {nsig}σ{'':<5} {frac_k:9.1f}% {frac_n:9.1f}% {ideal:>14.1f}%")

# Median and mean pull
print(f"\n  Median pull:  Khronon = {np.median(pulls_khr):.2f}σ,  NFW = {np.median(pulls_nfw):.2f}σ")
print(f"  Mean pull:    Khronon = {np.mean(pulls_khr):.2f}σ,  NFW = {np.mean(pulls_nfw):.2f}σ")

# =====================================================================
# TEST 2: Radial Acceleration Relation (RAR) Scatter
# =====================================================================
print("\n" + "─" * 72)
print("  TEST 2: RAR Scatter (dex)")
print("─" * 72)

kpc_to_m = 3.086e19   # 1 kpc in metres
kms_to_ms = 1e3        # 1 km/s in m/s

log_gobs_all = []
log_gbar_all = []
log_gpred_khr_all = []
log_gpred_nfw_all = []
delta_log_gobs_all = []  # measurement uncertainty in log10(g_obs)

for g in galaxies:
    R_kpc = np.array(g["R"])
    Vobs = np.array(g["Vobs"])
    eVobs = np.array(g["e_Vobs"])
    Vbar = np.array(g["Vbar"])
    Vnfw = np.array(g["V_nfw"])

    # Skip points with R = 0 or V = 0
    mask = (R_kpc > 0) & (Vobs > 0) & (Vbar > 0)
    R_kpc = R_kpc[mask]
    Vobs = Vobs[mask]
    eVobs = eVobs[mask]
    Vbar = Vbar[mask]
    Vnfw = Vnfw[mask]

    R_m = R_kpc * kpc_to_m
    v_obs_ms = Vobs * kms_to_ms
    v_bar_ms = Vbar * kms_to_ms
    v_nfw_ms = Vnfw * kms_to_ms
    e_v_ms = eVobs * kms_to_ms

    g_obs = v_obs_ms**2 / R_m
    g_bar = v_bar_ms**2 / R_m
    g_nfw = v_nfw_ms**2 / R_m

    # Khronon RAR prediction: g_pred = g_bar / (1 - exp(-sqrt(g_bar / a0)))
    x = np.sqrt(g_bar / a0)
    denom = 1.0 - np.exp(-x)
    # Avoid division by zero for very small g_bar
    safe = denom > 1e-30
    g_pred_khr = np.where(safe, g_bar / denom, g_bar)

    # Measurement uncertainty propagation: g_obs = v^2/R  =>  δ(log10 g_obs) = 2 * δv/v / ln(10)
    delta_log = 2.0 * (e_v_ms / v_obs_ms) / np.log(10.0)

    # Filter valid (positive g values)
    valid = (g_obs > 0) & (g_bar > 0) & (g_pred_khr > 0) & (g_nfw > 0)

    log_gobs_all.extend(np.log10(g_obs[valid]))
    log_gbar_all.extend(np.log10(g_bar[valid]))
    log_gpred_khr_all.extend(np.log10(g_pred_khr[valid]))
    log_gpred_nfw_all.extend(np.log10(g_nfw[valid]))
    delta_log_gobs_all.extend(delta_log[valid])

log_gobs = np.array(log_gobs_all)
log_gbar = np.array(log_gbar_all)
log_gpred_khr = np.array(log_gpred_khr_all)
log_gpred_nfw = np.array(log_gpred_nfw_all)
delta_log = np.array(delta_log_gobs_all)

residual_khr = log_gobs - log_gpred_khr
residual_nfw = log_gobs - log_gpred_nfw

scatter_khr = np.std(residual_khr)
scatter_nfw = np.std(residual_nfw)
scatter_meas = np.sqrt(np.mean(delta_log**2))  # RMS measurement scatter

print(f"\n  Valid RAR data points: {len(log_gobs)}")
print(f"\n  {'Metric':<35} {'Value':>10}")
print(f"  {'─'*35} {'─'*10}")
print(f"  {'RAR scatter (Khronon prediction)':<35} {scatter_khr:>8.4f} dex")
print(f"  {'RAR scatter (NFW fit)':<35} {scatter_nfw:>8.4f} dex")
print(f"  {'Measurement scatter (RMS)':<35} {scatter_meas:>8.4f} dex")
print(f"  {'McGaugh+2016 observed scatter':<35} {'~0.13':>10} dex")

ratio = scatter_khr / scatter_meas if scatter_meas > 0 else float("inf")
print(f"\n  Khronon scatter / measurement scatter = {ratio:.2f}")
if ratio < 1.5:
    print("  => Khronon scatter is close to measurement floor — theory is as good as data allows.")
elif ratio < 2.0:
    print("  => Khronon scatter is within ~2x of measurement floor — reasonable.")
else:
    print(f"  => Khronon scatter exceeds measurement floor by {ratio:.1f}x — room for improvement.")

# Mean bias
print(f"\n  Mean residual (bias):  Khronon = {np.mean(residual_khr):+.4f} dex,  NFW = {np.mean(residual_nfw):+.4f} dex")

# =====================================================================
# TEST 3: Parameter Efficiency
# =====================================================================
print("\n" + "─" * 72)
print("  TEST 3: Parameter Efficiency")
print("─" * 72)

total_chi2_khr = 0.0
total_chi2_nfw = 0.0
total_pts = 0
n_gal_used = 0

for g in galaxies:
    npts = g["n_pts"]
    chi2r_k = g["chi2_khronon"]
    chi2r_n = g["chi2_nfw"]

    # Khronon: 1 free param per galaxy (M/L) => dof = n_pts - 1
    dof_k = max(npts - 1, 1)
    # NFW: 3 free params per galaxy (M/L, M200, c) => dof = n_pts - 3
    dof_n = max(npts - 3, 1)

    total_chi2_khr += chi2r_k * dof_k
    total_chi2_nfw += chi2r_n * dof_n
    total_pts += npts
    n_gal_used += 1

# Total parameters
params_khr = n_gal_used * 1   # 1 M/L per galaxy
params_nfw = n_gal_used * 3   # M/L + log10(M200) + log10(c) per galaxy

# Total DOF
dof_total_khr = total_pts - params_khr
dof_total_nfw = total_pts - params_nfw

# Reduced chi2
redchi2_khr = total_chi2_khr / dof_total_khr if dof_total_khr > 0 else float("inf")
redchi2_nfw = total_chi2_nfw / dof_total_nfw if dof_total_nfw > 0 else float("inf")

# Efficiency = total chi2 / n_params
eff_khr = total_chi2_khr / params_khr
eff_nfw = total_chi2_nfw / params_nfw

print(f"\n  {'Quantity':<40} {'Khronon':>12} {'NFW':>12}")
print(f"  {'─'*40} {'─'*12} {'─'*12}")
print(f"  {'Galaxies':<40} {n_gal_used:>12d} {n_gal_used:>12d}")
print(f"  {'Total data points':<40} {total_pts:>12d} {total_pts:>12d}")
print(f"  {'Parameters per galaxy':<40} {'1':>12} {'3':>12}")
print(f"  {'Total parameters':<40} {params_khr:>12d} {params_nfw:>12d}")
print(f"  {'Total DOF':<40} {dof_total_khr:>12d} {dof_total_nfw:>12d}")
print(f"  {'Total chi2':<40} {total_chi2_khr:>12.1f} {total_chi2_nfw:>12.1f}")
print(f"  {'Reduced chi2 (total chi2 / DOF)':<40} {redchi2_khr:>12.3f} {redchi2_nfw:>12.3f}")
print(f"  {'chi2 per parameter':<40} {eff_khr:>12.2f} {eff_nfw:>12.2f}")

print(f"\n  Parameter ratio:  NFW uses {params_nfw / params_khr:.0f}x more parameters than Khronon")
print(f"  Total chi2 ratio: Khronon/NFW = {total_chi2_khr / total_chi2_nfw:.2f}")

# BIC-like comparison (rough)
# BIC = chi2 + k * ln(n)
bic_khr = total_chi2_khr + params_khr * np.log(total_pts)
bic_nfw = total_chi2_nfw + params_nfw * np.log(total_pts)
delta_bic = bic_khr - bic_nfw

print(f"\n  Approximate BIC:  Khronon = {bic_khr:.1f},  NFW = {bic_nfw:.1f}")
print(f"  ΔBIC (Khronon − NFW) = {delta_bic:+.1f}")
if delta_bic < -10:
    print("  => Strong evidence favoring Khronon")
elif delta_bic < 0:
    print("  => Mild evidence favoring Khronon")
elif delta_bic < 10:
    print("  => Inconclusive / comparable")
else:
    print("  => NFW fits better even accounting for its extra parameters")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"""
  1. Prediction Accuracy
     - Khronon: {100*np.mean(pulls_khr<=1):.1f}% within 1σ, {100*np.mean(pulls_khr<=2):.1f}% within 2σ
     - NFW:     {100*np.mean(pulls_nfw<=1):.1f}% within 1σ, {100*np.mean(pulls_nfw<=2):.1f}% within 2σ
     - Khronon uses 1 parameter/galaxy vs NFW's 3

  2. RAR Scatter
     - Khronon scatter:     {scatter_khr:.4f} dex
     - NFW scatter:         {scatter_nfw:.4f} dex
     - Measurement floor:   {scatter_meas:.4f} dex
     - Khronon/measurement: {ratio:.2f}x

  3. Parameter Efficiency
     - Khronon: {params_khr} params, reduced chi2 = {redchi2_khr:.3f}
     - NFW:     {params_nfw} params, reduced chi2 = {redchi2_nfw:.3f}
     - ΔBIC (Khronon − NFW) = {delta_bic:+.1f}
""")
print("=" * 72)
