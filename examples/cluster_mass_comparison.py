#!/usr/bin/env python3
"""
cluster_mass_comparison.py

Galaxy cluster mass problem: comparing Newton, MOND, and Khronon predictions
against observed dynamical mass for the Coma cluster.

Physics:
  Galaxy clusters have baryonic mass (mostly hot X-ray gas) ~5-7x less than
  observed dynamical mass. MOND reduces this gap to ~2x but cannot fully
  explain it. We test whether the Khronon framework does better.

Reference data: Coma cluster
  M_gas ~ 1.4e14 Msun, M_stars ~ 0.3e14 Msun, M_bar ~ 1.7e14 Msun
  r_c ~ 290 kpc, beta ~ 0.75, T_X ~ 8.2 keV
  M_dyn_observed ~ 1.2e15 Msun within r200 ~ 2.0 Mpc
"""

import json
import os
import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G       = 6.674e-11       # m^3 kg^-1 s^-2
c       = 2.998e8         # m/s
H0      = 73e3 / 3.086e22 # s^-1  (73 km/s/Mpc)
Msun    = 1.989e30        # kg
kpc     = 3.086e19        # m
Mpc     = 3.086e22        # m
k_B     = 1.381e-23       # J/K
m_p     = 1.673e-27       # kg
keV_to_J = 1.602e-16      # J per keV

# MOND and Khronon acceleration scales
a0_MOND = 1.2e-10         # m/s^2
a0_K    = c * H0 / (2 * np.pi)  # ~ 1.13e-10 m/s^2

# ---------------------------------------------------------------------------
# Coma cluster parameters
# ---------------------------------------------------------------------------
M_gas_total   = 1.4e14 * Msun   # kg
M_stars_total = 0.3e14 * Msun   # kg
M_bar_total   = M_gas_total + M_stars_total
r_c           = 290 * kpc        # core radius (m)
beta_param    = 0.75             # beta-model parameter
T_X           = 8.2              # keV
r200          = 2.0 * Mpc        # m
M_dyn_obs     = 1.2e15 * Msun   # observed dynamical mass within r200

# Hernquist scale length for stellar component
a_hernquist = 300 * kpc  # m

# ---------------------------------------------------------------------------
# Gas density: beta-model
# ---------------------------------------------------------------------------
# rho_gas(r) = rho0 * (1 + r^2/r_c^2)^(-3*beta/2)
# Normalise so that M_gas(<r200) = M_gas_total.
# M_gas(<R) = int_0^R 4*pi*r^2 * rho0 * (1 + r^2/r_c^2)^(-3*beta/2) dr

def _beta_model_unnorm(r):
    """Un-normalised beta-model density at radius r (m)."""
    return (1.0 + (r / r_c)**2) ** (-1.5 * beta_param)

def _mass_integral_unnorm(R):
    """Integral of 4*pi*r^2 * (1 + r^2/r_c^2)^(-3*beta/2) from 0 to R."""
    integrand = lambda r: 4.0 * np.pi * r**2 * _beta_model_unnorm(r)
    val, _ = quad(integrand, 0, R, limit=200)
    return val

# Compute normalisation constant rho0
_norm_integral = _mass_integral_unnorm(r200)
rho0_gas = M_gas_total / _norm_integral  # kg/m^3

def rho_gas(r):
    return rho0_gas * _beta_model_unnorm(r)

# ---------------------------------------------------------------------------
# Stellar density: Hernquist profile
# ---------------------------------------------------------------------------
# rho_stars(r) = M_stars / (2*pi) * a / (r * (r + a)^3)

def rho_stars(r):
    a = a_hernquist
    if r < 1e-10:  # avoid division by zero
        return 0.0
    return (M_stars_total / (2.0 * np.pi)) * a / (r * (r + a)**3)

# ---------------------------------------------------------------------------
# Total baryonic density
# ---------------------------------------------------------------------------
def rho_bar(r):
    return rho_gas(r) + rho_stars(r)

# ---------------------------------------------------------------------------
# Enclosed baryonic mass M_bar(<r)
# ---------------------------------------------------------------------------
def M_bar_enclosed(R):
    integrand = lambda r: 4.0 * np.pi * r**2 * rho_bar(r)
    val, _ = quad(integrand, 0, R, limit=200)
    return val

# Enclosed gas mass (for thermal Sigma)
def M_gas_enclosed(R):
    integrand = lambda r: 4.0 * np.pi * r**2 * rho_gas(r)
    val, _ = quad(integrand, 0, R, limit=200)
    return val

# ---------------------------------------------------------------------------
# Interpolating function nu(y) -- RAR (simple) form
# ---------------------------------------------------------------------------
def nu_rar(y):
    """nu(y) = 1 / (1 - exp(-sqrt(y))),  y = g_bar / a0."""
    sqrt_y = np.sqrt(np.clip(y, 1e-30, None))
    return 1.0 / (1.0 - np.exp(-sqrt_y))

# ---------------------------------------------------------------------------
# Compute profiles at 100 log-spaced radii
# ---------------------------------------------------------------------------
r_min = 10 * kpc
r_max = 3.0 * Mpc
N_pts = 100
radii = np.geomspace(r_min, r_max, N_pts)

# Storage arrays
M_bar_arr          = np.zeros(N_pts)
M_gas_arr          = np.zeros(N_pts)
g_bar_arr          = np.zeros(N_pts)
g_newton_arr       = np.zeros(N_pts)
g_mond_arr         = np.zeros(N_pts)
g_khronon_rar_arr  = np.zeros(N_pts)
g_khronon_exp_arr  = np.zeros(N_pts)
g_khronon_full_arr = np.zeros(N_pts)

print("Computing mass and acceleration profiles (100 radii)...")
for i, r in enumerate(radii):
    # Enclosed masses
    Mb = M_bar_enclosed(r)
    Mg = M_gas_enclosed(r)
    M_bar_arr[i] = Mb
    M_gas_arr[i] = Mg

    # Baryonic gravitational acceleration
    g_b = G * Mb / r**2
    g_bar_arr[i] = g_b

    # 1) Newton (baryons only, no dark matter)
    g_newton_arr[i] = g_b

    # 2) MOND: g = g_bar / (1 - exp(-sqrt(g_bar/a0_MOND)))
    y_mond = g_b / a0_MOND
    g_mond_arr[i] = g_b * nu_rar(y_mond)

    # 3) Khronon RAR: same functional form, a0_K instead of a0_MOND
    y_K = g_b / a0_K
    g_khronon_rar_arr[i] = g_b * nu_rar(y_K)

    # 4) Khronon exp: RAR * exp(Sigma_metric)
    Sigma_metric = 2.0 * G * Mb / (c**2 * r)
    g_khronon_exp_arr[i] = g_b * nu_rar(y_K) * np.exp(Sigma_metric)

    # 5) Khronon full: RAR * exp(Sigma_metric) * (1 + Sigma_thermal)
    #    Sigma_thermal = kT/(m_p c^2) * M_gas(<r)/M_bar(<r)
    T_X_joules = T_X * keV_to_J
    Sigma_thermal = (T_X_joules / (m_p * c**2)) * (Mg / Mb) if Mb > 0 else 0.0
    g_khronon_full_arr[i] = g_b * nu_rar(y_K) * np.exp(Sigma_metric) * (1.0 + Sigma_thermal)

    if (i + 1) % 25 == 0:
        print(f"  ... {i + 1}/{N_pts} done")

# Dynamical masses M_dyn = g * r^2 / G
M_newton_arr       = g_newton_arr       * radii**2 / G
M_mond_arr         = g_mond_arr         * radii**2 / G
M_khronon_rar_arr  = g_khronon_rar_arr  * radii**2 / G
M_khronon_exp_arr  = g_khronon_exp_arr  * radii**2 / G
M_khronon_full_arr = g_khronon_full_arr * radii**2 / G

# Observed dynamical mass profile (flat at M_dyn_obs; for display we show
# it as a constant since we only have the value at r200).
M_obs_arr = np.full(N_pts, M_dyn_obs)

# ---------------------------------------------------------------------------
# Find index closest to r200
# ---------------------------------------------------------------------------
i200 = np.argmin(np.abs(radii - r200))
r_at_200 = radii[i200]

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
M_unit = 1e14 * Msun  # display unit: 10^14 Msun

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("GALAXY CLUSTER MASS COMPARISON -- Coma Cluster")
print("=" * 72)
print(f"\nReference radius r200 = {r200 / Mpc:.2f} Mpc  "
      f"(grid point {i200}: {r_at_200 / Mpc:.3f} Mpc)")
print(f"Observed M_bar_total  = {M_bar_total / M_unit:.2f} x 10^14 Msun")
print(f"Observed M_dyn        = {M_dyn_obs / M_unit:.2f} x 10^14 Msun")
print(f"\na0_MOND = {a0_MOND:.3e} m/s^2")
print(f"a0_K    = {a0_K:.3e} m/s^2  (cH0/2pi)")

print(f"\n{'Model':<25} {'M(<r200) [1e14 Msun]':>22} {'M/M_bar':>10} {'M_obs/M_model':>15}")
print("-" * 72)

models = [
    ("Newton (baryons only)", M_newton_arr[i200]),
    ("MOND",                  M_mond_arr[i200]),
    ("Khronon RAR",           M_khronon_rar_arr[i200]),
    ("Khronon exp (+ Sigma)", M_khronon_exp_arr[i200]),
    ("Khronon full (+ therm)",M_khronon_full_arr[i200]),
    ("Observed",              M_dyn_obs),
]

Mb200 = M_bar_arr[i200]
for name, M in models:
    ratio = M / Mb200
    missing = M_dyn_obs / M if M > 0 else float('inf')
    print(f"  {name:<23} {M / M_unit:>20.3f}   {ratio:>8.2f}   {missing:>13.2f}")

print("\n" + "-" * 72)
print("INTERPRETATION:")
print("-" * 72)

# Compute Sigma_metric and Sigma_thermal at r200
Sigma_met_200 = 2.0 * G * Mb200 / (c**2 * r_at_200)
Mg200 = M_gas_arr[i200]
T_X_J = T_X * keV_to_J
Sig_therm_200 = (T_X_J / (m_p * c**2)) * (Mg200 / Mb200) if Mb200 > 0 else 0

print(f"\nAt r200:")
print(f"  g_bar          = {g_bar_arr[i200]:.3e} m/s^2")
print(f"  g_bar/a0_MOND  = {g_bar_arr[i200] / a0_MOND:.4f}")
print(f"  g_bar/a0_K     = {g_bar_arr[i200] / a0_K:.4f}")
print(f"  Sigma_metric   = {Sigma_met_200:.6e}  (exp(Sigma) = {np.exp(Sigma_met_200):.8f})")
print(f"  Sigma_thermal  = {Sig_therm_200:.6e}")
print(f"  exp(Sigma_metric)*(1+Sigma_thermal) = {np.exp(Sigma_met_200) * (1 + Sig_therm_200):.8f}")

newton_frac  = M_newton_arr[i200] / M_dyn_obs
mond_frac    = M_mond_arr[i200] / M_dyn_obs
krar_frac    = M_khronon_rar_arr[i200] / M_dyn_obs
kexp_frac    = M_khronon_exp_arr[i200] / M_dyn_obs
kfull_frac   = M_khronon_full_arr[i200] / M_dyn_obs

print(f"\nFraction of observed mass explained:")
print(f"  Newton:           {newton_frac * 100:6.1f}%")
print(f"  MOND:             {mond_frac * 100:6.1f}%")
print(f"  Khronon RAR:      {krar_frac * 100:6.1f}%")
print(f"  Khronon exp:      {kexp_frac * 100:6.1f}%")
print(f"  Khronon full:     {kfull_frac * 100:6.1f}%")

print("\n" + "-" * 72)
print("HONEST ASSESSMENT:")
print("-" * 72)
if kfull_frac < 0.90:
    print(f"\n  Khronon (full, with thermal) explains {kfull_frac*100:.1f}% of observed mass.")
    print("  This does NOT fully close the cluster mass gap.")
    if kfull_frac > mond_frac:
        improvement = (kfull_frac - mond_frac) / mond_frac * 100
        print(f"  However, Khronon full is {improvement:.1f}% better than MOND.")
    else:
        print("  Khronon full does not significantly improve over MOND for clusters.")
    print("\n  The cluster mass problem remains a challenge for modified gravity/entropy")
    print("  theories. Additional physics (e.g., massive neutrinos, stronger non-linear")
    print("  entropy coupling, or cluster-specific baryon processes) may be needed.")
elif kfull_frac < 1.10:
    print(f"\n  Khronon (full, with thermal) explains {kfull_frac*100:.1f}% of observed mass.")
    print("  This is consistent with observations -- the cluster gap is closed!")
else:
    print(f"\n  Khronon (full, with thermal) OVERPREDICTS: {kfull_frac*100:.1f}% of observed mass.")
    print("  This indicates the thermal Sigma term may be too strong.")

# ---------------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------------
output_dir = "/Users/akaihuangm1/Desktop/github/anatropic/web/data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cluster_mass_profiles.json")

# Build summary statistics at r200
summary = {
    "cluster": "Coma",
    "r200_Mpc": round(r200 / Mpc, 3),
    "M_bar_total_1e14Msun": round(M_bar_total / M_unit, 2),
    "M_dyn_observed_1e14Msun": round(M_dyn_obs / M_unit, 2),
    "a0_MOND_m_s2": float(f"{a0_MOND:.3e}"),
    "a0_K_m_s2": float(f"{a0_K:.3e}"),
    "at_r200": {
        "M_bar_1e14Msun":          round(M_bar_arr[i200] / M_unit, 4),
        "M_newton_1e14Msun":       round(M_newton_arr[i200] / M_unit, 4),
        "M_mond_1e14Msun":         round(M_mond_arr[i200] / M_unit, 4),
        "M_khronon_rar_1e14Msun":  round(M_khronon_rar_arr[i200] / M_unit, 4),
        "M_khronon_exp_1e14Msun":  round(M_khronon_exp_arr[i200] / M_unit, 4),
        "M_khronon_full_1e14Msun": round(M_khronon_full_arr[i200] / M_unit, 4),
        "M_observed_1e14Msun":     round(M_dyn_obs / M_unit, 2),
        "ratio_newton":            round(M_newton_arr[i200] / Mb200, 4),
        "ratio_mond":              round(M_mond_arr[i200] / Mb200, 4),
        "ratio_khronon_rar":       round(M_khronon_rar_arr[i200] / Mb200, 4),
        "ratio_khronon_exp":       round(M_khronon_exp_arr[i200] / Mb200, 4),
        "ratio_khronon_full":      round(M_khronon_full_arr[i200] / Mb200, 4),
        "ratio_observed":          round(M_dyn_obs / Mb200, 4),
        "missing_factor_newton":   round(M_dyn_obs / M_newton_arr[i200], 4),
        "missing_factor_mond":     round(M_dyn_obs / M_mond_arr[i200], 4),
        "missing_factor_khronon_rar":  round(M_dyn_obs / M_khronon_rar_arr[i200], 4),
        "missing_factor_khronon_exp":  round(M_dyn_obs / M_khronon_exp_arr[i200], 4),
        "missing_factor_khronon_full": round(M_dyn_obs / M_khronon_full_arr[i200], 4),
        "Sigma_metric":            float(f"{Sigma_met_200:.6e}"),
        "Sigma_thermal":           float(f"{Sig_therm_200:.6e}"),
        "frac_explained_newton":   round(newton_frac, 4),
        "frac_explained_mond":     round(mond_frac, 4),
        "frac_explained_khronon_rar":  round(krar_frac, 4),
        "frac_explained_khronon_exp":  round(kexp_frac, 4),
        "frac_explained_khronon_full": round(kfull_frac, 4),
    },
}

data = {
    "description": "Galaxy cluster mass profiles: Coma cluster, Newton vs MOND vs Khronon",
    "units": {
        "r": "kpc",
        "M": "1e14 Msun",
        "g": "m/s^2",
    },
    "r_kpc": (radii / kpc).tolist(),
    "M_bar":          (M_bar_arr / M_unit).tolist(),
    "M_newton":       (M_newton_arr / M_unit).tolist(),
    "M_mond":         (M_mond_arr / M_unit).tolist(),
    "M_khronon_rar":  (M_khronon_rar_arr / M_unit).tolist(),
    "M_khronon_exp":  (M_khronon_exp_arr / M_unit).tolist(),
    "M_khronon_full": (M_khronon_full_arr / M_unit).tolist(),
    "M_observed":     (M_obs_arr / M_unit).tolist(),
    "g_bar":          g_bar_arr.tolist(),
    "g_newton":       g_newton_arr.tolist(),
    "g_mond":         g_mond_arr.tolist(),
    "g_khronon_rar":  g_khronon_rar_arr.tolist(),
    "g_khronon_exp":  g_khronon_exp_arr.tolist(),
    "g_khronon_full": g_khronon_full_arr.tolist(),
    "summary": summary,
}

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nJSON exported to: {output_path}")
print("Done.")
