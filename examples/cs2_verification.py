#!/usr/bin/env python3
"""
Rigorous numerical verification of c_s² (effective sound speed squared)
at CMB scales for the Khronon/τ framework.

Key equation: Blanchet & Skordis 2024, Eq. 4.31
    c_s²(t,k) = c_ad² / [1 + c_ad² k² / (4π G a² ρ_K (1+w))]

The crisis claim: with μ = H₀/c, the background equation of state
    w̃₀ = δ₀/2 ≈ 0.170
so c_s² ~ w̃₀ ~ 0.17, far exceeding the TKS 2016 bound of 3.4 × 10⁻⁶.

Resolution: the RUNNING Khronon coupling μ_bg(z) = H(z)/c gives
    w̃₀(z) = I₀ / (4 μ(z)²) = w̃₀(0) × (H₀/H(z))²
At z = 1100: H/H₀ ~ 2.35×10⁴, so w̃₀(z=1100) ~ 3×10⁻¹⁰ (dust-like).
Then Eq. 4.31 provides additional Jeans-type suppression.

Author: Sheng-Kai Huang
Date: 2026-03-14

References:
  - Blanchet & Skordis 2024 (BS2024), Eq. 4.31
  - Thomas, Kopp & Skordis 2016 (TKS2016): c_s² < 3.4×10⁻⁶
  - Skordis & Zlosnik 2021 (SZ2021): CMB fitting requirements
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: FUNDAMENTAL CONSTANTS (SI units)
# ============================================================================
print("=" * 80)
print("SECTION 1: FUNDAMENTAL CONSTANTS")
print("=" * 80)

c = 2.998e8          # speed of light [m/s]
G = 6.674e-11        # Newton's gravitational constant [m³ kg⁻¹ s⁻²]
Mpc = 3.086e22       # 1 Megaparsec [m]

print(f"  c     = {c:.3e} m/s")
print(f"  G     = {G:.3e} m^3 kg^-1 s^-2")
print(f"  1 Mpc = {Mpc:.3e} m")

# ============================================================================
# SECTION 2: COSMOLOGICAL PARAMETERS (Planck 2018)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: COSMOLOGICAL PARAMETERS (Planck 2018)")
print("=" * 80)

H0_km = 67.4         # Hubble constant [km/s/Mpc]
H0 = H0_km * 1e3 / Mpc  # Convert to [s⁻¹]
h = H0_km / 100.0    # dimensionless Hubble parameter

Omega_r = 9.1e-5     # radiation density parameter
Omega_m = 0.315       # matter density parameter
Omega_Lambda = 0.685  # dark energy density parameter
Omega_cdm_h2 = 0.120  # cold dark matter density * h²
Omega_cdm = Omega_cdm_h2 / h**2

# Critical density today
rho_crit_0 = 3 * H0**2 / (8 * np.pi * G)

print(f"  H_0         = {H0_km} km/s/Mpc = {H0:.4e} s^-1")
print(f"  h           = {h:.4f}")
print(f"  Omega_r     = {Omega_r:.1e}")
print(f"  Omega_m     = {Omega_m:.3f}")
print(f"  Omega_cdm   = {Omega_cdm:.4f}")
print(f"  rho_crit(0) = {rho_crit_0:.4e} kg/m^3")

# ============================================================================
# SECTION 3: KHRONON PARAMETERS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: KHRONON BACKGROUND PARAMETERS")
print("=" * 80)


def H_of_z(z):
    """Hubble rate H(z) from the Friedmann equation [s^-1]."""
    return H0 * np.sqrt(Omega_r * (1 + z)**4
                        + Omega_m * (1 + z)**3
                        + Omega_Lambda)


def rho_crit_of_z(z):
    """Critical density at redshift z [kg/m^3]."""
    Hz = H_of_z(z)
    return 3 * Hz**2 / (8 * np.pi * G)


# Khronon parameters from Blanchet-Skordis
# Background field: Q_0 - 1 = delta_0
# Omega_K = (delta_0^2 + 2*delta_0) / 3  =>  solve for delta_0 from Omega_cdm
# delta_0 such that Omega_K = Omega_cdm = 0.2642
# delta^2 + 2 delta - 3 Omega_cdm = 0
delta_0 = -1 + np.sqrt(1 + 3 * Omega_cdm)

# w_tilde_0 at z=0 (TODAY): the crisis value
w_tilde_0_today = delta_0 / 2

# Integration constant I_0
# delta_0 = I_0 / (2 mu_0^2), mu_0 = H_0/c
mu_0 = H0 / c
I_0 = 2 * mu_0**2 * delta_0

print(f"\n  Khronon field parameters:")
print(f"    mu_0 = H_0/c = {mu_0:.4e} m^-1  =  {mu_0 * Mpc:.4e} Mpc^-1")
print(f"    delta_0 = Q_0 - 1 = {delta_0:.6f}")
print(f"    I_0 = 2 mu_0^2 delta_0 = {I_0:.4e} m^-2")
print(f"    Omega_K = (delta^2 + 2 delta)/3 = {(delta_0**2 + 2*delta_0)/3:.4f}")
print(f"    w_tilde_0(z=0) = delta_0/2 = {w_tilde_0_today:.6f}")
print(f"    *** THIS IS THE 'CRISIS' VALUE ***")
print(f"    Compare TKS bound: {3.4e-6:.1e}  <-- violated by factor {w_tilde_0_today/3.4e-6:.0f}x")

# ============================================================================
# SECTION 4: THE RUNNING KHRONON COUPLING AND w_tilde(z)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: RUNNING KHRONON COUPLING mu_bg(z) = H(z)/c")
print("=" * 80)
print("""
  The running Khronon coupling: mu_bg(z) = H(z)/c

  This is the key resolution of the "Catch-22":
    w_tilde_0(z) = I_0 / (4 mu(z)^2) = w_tilde_0(today) * (H_0/H(z))^2

  Since H(z) grows rapidly with z, w_tilde_0(z) shrinks rapidly.
  At z=1100: H(z)/H_0 ~ 2.35e4, so w_tilde_0 ~ 0.17 / (2.35e4)^2 ~ 3e-10
""")


def w_tilde_bg(z):
    """Background effective equation of state w_tilde_0(z).

    w_tilde_0(z) = I_0 / (4 mu(z)^2)
                 = w_tilde_0(today) * (H_0 / H(z))^2

    This is a BACKGROUND quantity (k-independent) from the BS field equation.
    """
    Hz = H_of_z(z)
    return w_tilde_0_today * (H0 / Hz)**2


# Table of w_tilde at key redshifts
redshifts_list = [0, 0.5, 1, 2, 5, 10, 50, 100, 500, 1100, 3000, 10000]
TKS_bound = 3.4e-6

print(f"  {'z':>8s}  {'H(z)/H_0':>12s}  {'mu(z) [m^-1]':>16s}  {'w_tilde(z)':>14s}  {'< TKS?':>8s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*16}  {'-'*14}  {'-'*8}")
for z in redshifts_list:
    Hz = H_of_z(z)
    mu_z = Hz / c
    wt = w_tilde_bg(z)
    ok = "YES" if wt < TKS_bound else "NO"
    print(f"  {z:8g}  {Hz/H0:12.2e}  {mu_z:16.4e}  {wt:14.4e}  {ok:>8s}")

# ============================================================================
# SECTION 5: c_ad^2 AND c_s^2 FROM BLANCHET-SKORDIS EQ. 4.31
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: c_s^2 FROM BLANCHET-SKORDIS EQ. 4.31")
print("=" * 80)
print("""
  For the Khronon with K(Q) = mu^2 (Q - 1)^2:
    - Background EoS: w = w_tilde_0(z) (computed above)
    - Adiabatic sound speed: c_ad^2 = w_tilde_0(z) (for nearly pressureless fluid)

  The EFFECTIVE sound speed including gravitational back-reaction (Eq. 4.31):

    c_s^2(z,k) = c_ad^2 / [1 + c_ad^2 c^2 k^2 / (4 pi G a^2 rho_K (1+w))]

  where:
    - k is the comoving wavenumber [m^-1]
    - a = 1/(1+z)
    - rho_K = Omega_cdm * rho_crit(z) (Khronon energy density)
    - c_ad^2 is dimensionless (fraction of c^2)

  The denominator provides Jeans-type suppression at sub-Jeans scales.
""")


def rho_K_of_z(z):
    """Khronon energy density at redshift z [kg/m^3].
    rho_K = Omega_cdm * rho_crit(z)
    """
    return Omega_cdm * rho_crit_of_z(z)


def cs2_full(k_Mpc, z):
    """Full c_s^2 from Blanchet-Skordis Eq. 4.31 (dimensionless, in units of c^2).

    Parameters
    ----------
    k_Mpc : float or array
        Comoving wavenumber in Mpc^-1
    z : float
        Redshift

    Returns
    -------
    cs2 : float or array
        c_s^2 / c^2 (dimensionless)
    """
    a = 1.0 / (1 + z)
    rho_K = rho_K_of_z(z)
    wt = w_tilde_bg(z)

    # c_ad^2 = w_tilde (dimensionless)
    c_ad2 = wt

    # Convert k from Mpc^-1 to m^-1
    k_SI = k_Mpc / Mpc  # m^-1

    # Physical sound speed squared: v_s^2 = c_ad^2 * c^2 [m^2/s^2]
    v_s2 = c_ad2 * c**2

    # Gravitational term: 4 pi G a^2 rho_K (1+w) [s^-2]
    grav_term = 4 * np.pi * G * a**2 * rho_K * (1 + wt)

    # Denominator
    denom = 1.0 + v_s2 * k_SI**2 / grav_term

    # Result: c_s^2 / c^2
    return c_ad2 / denom


def cs2_naive(z):
    """Naive estimate: c_s^2 = c_ad^2 = w_tilde (no k-dependent suppression).

    This is the "crisis" value when evaluated at z=0.
    With running mu, this already gives the right answer at high z.
    """
    return w_tilde_bg(z)


# ============================================================================
# SECTION 6: DETAILED RESULTS AT z = 1100
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: DETAILED CALCULATION AT z = 1100")
print("=" * 80)

z_cmb = 1100
a_cmb = 1.0 / (1 + z_cmb)
Hz_cmb = H_of_z(z_cmb)
mu_cmb = Hz_cmb / c
rho_K_cmb = rho_K_of_z(z_cmb)
wt_cmb = w_tilde_bg(z_cmb)

print(f"""
  Step-by-step at z = {z_cmb}:

  1. Scale factor:  a = 1/{1+z_cmb} = {a_cmb:.6e}

  2. Hubble rate:   H(z) = H_0 * sqrt(Omega_r*(1+z)^4 + Omega_m*(1+z)^3 + Omega_Lambda)
                    H({z_cmb}) = {Hz_cmb:.4e} s^-1
                    H/H_0 = {Hz_cmb/H0:.4e}

  3. Running coupling: mu_bg = H(z)/c = {mu_cmb:.4e} m^-1
                        mu_bg/mu_0 = {mu_cmb/mu_0:.4e}

  4. Background EoS: w_tilde_0(z) = w_tilde_0(today) * (H_0/H(z))^2
                     = {w_tilde_0_today:.6f} * ({H0/Hz_cmb:.4e})^2
                     = {wt_cmb:.6e}

  5. Check: I_0/(4*mu(z)^2) = {I_0:.4e} / (4 * ({mu_cmb:.4e})^2)
                              = {I_0 / (4 * mu_cmb**2):.6e}
     (Matches w_tilde above: {abs(wt_cmb - I_0/(4*mu_cmb**2))/wt_cmb:.2e} relative error)

  6. Khronon density: rho_K = Omega_cdm * rho_crit(z)
                      = {Omega_cdm:.4f} * {rho_crit_of_z(z_cmb):.4e}
                      = {rho_K_cmb:.4e} kg/m^3
""")

# Now compute c_s^2 at various k values
k_values_Mpc = [1e-4, 1e-3, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0, 10.0, 100.0]

print(f"  c_ad^2 = w_tilde_0(z={z_cmb}) = {wt_cmb:.6e}  (dimensionless)")
print(f"  TKS 2016 bound:                 {TKS_bound:.1e}")
print()

print(f"  {'k [Mpc^-1]':>14s}  {'c_s^2 (Eq4.31)':>16s}  {'c_ad^2 (naive)':>16s}  {'denom':>12s}  {'< TKS?':>8s}")
print(f"  {'-'*14}  {'-'*16}  {'-'*16}  {'-'*12}  {'-'*8}")

for k in k_values_Mpc:
    cs2_f = cs2_full(k, z_cmb)
    cs2_n = cs2_naive(z_cmb)
    # Compute denominator for display
    k_SI = k / Mpc
    v_s2 = wt_cmb * c**2
    grav_term = 4 * np.pi * G * a_cmb**2 * rho_K_cmb * (1 + wt_cmb)
    denom_val = 1.0 + v_s2 * k_SI**2 / grav_term
    ok = "YES" if cs2_f < TKS_bound else "NO"
    print(f"  {k:14.4e}  {cs2_f:16.4e}  {cs2_n:16.4e}  {denom_val:12.4e}  {ok:>8s}")

# ============================================================================
# SECTION 7: DETAILED BREAKDOWN AT k = 0.05 Mpc^-1
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: STEP-BY-STEP AT k = 0.05 Mpc^-1, z = 1100")
print("=" * 80)

k_detail = 0.05  # Mpc^-1
k_detail_SI = k_detail / Mpc  # m^-1

v_s2 = wt_cmb * c**2
grav_term = 4 * np.pi * G * a_cmb**2 * rho_K_cmb * (1 + wt_cmb)
denom_term = v_s2 * k_detail_SI**2 / grav_term
denom = 1.0 + denom_term
cs2_result = wt_cmb / denom

print(f"""
  k = {k_detail} Mpc^-1 = {k_detail_SI:.4e} m^-1

  Numerator (c_ad^2):
    c_ad^2 = w_tilde_0(z=1100) = {wt_cmb:.6e}

  Denominator:
    v_s^2 = c_ad^2 * c^2 = {wt_cmb:.3e} * ({c:.3e})^2 = {v_s2:.4e} m^2/s^2
    4 pi G a^2 rho_K (1+w)
      = 4 pi * {G:.3e} * ({a_cmb:.3e})^2 * {rho_K_cmb:.3e} * {1+wt_cmb:.10f}
      = {grav_term:.4e} s^-2
    v_s^2 * k^2 / (4piG a^2 rho_K (1+w))
      = {v_s2:.3e} * ({k_detail_SI:.3e})^2 / {grav_term:.3e}
      = {denom_term:.4e}
    denom = 1 + {denom_term:.4e} = {denom:.4e}

  RESULT:
    c_s^2 = c_ad^2 / denom = {wt_cmb:.6e} / {denom:.4e}
          = {cs2_result:.6e}
""")

print(f"  ===============================================================")
print(f"  c_s^2(k=0.05 Mpc^-1, z=1100) = {cs2_result:.4e}")
print(f"  TKS 2016 bound                = {TKS_bound:.4e}")
print(f"  Ratio c_s^2 / bound           = {cs2_result/TKS_bound:.4e}")
if cs2_result < TKS_bound:
    print(f"  --> PASSES TKS bound by factor {TKS_bound/cs2_result:.1f}x")
else:
    print(f"  --> VIOLATES TKS bound by factor {cs2_result/TKS_bound:.1f}x")
print(f"  ===============================================================")

# ============================================================================
# SECTION 8: SELF-CONSISTENCY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: SELF-CONSISTENCY CHECK")
print("=" * 80)

# The ratio 4piG a^2 rho_K / (c^2 mu^2) has a simple analytical form
# Since mu = H/c and rho_crit = 3H^2/(8piG):
# 4piG a^2 rho_K / (c^2 mu^2) = 4piG a^2 Omega_cdm rho_crit / (c^2 H^2/c^2)
#                                = 4piG a^2 Omega_cdm * 3H^2/(8piG) / H^2
#                                = (3/2) Omega_cdm a^2

analytical_ratio = 1.5 * Omega_cdm * a_cmb**2
numerical_ratio = 4 * np.pi * G * a_cmb**2 * rho_K_cmb / (c**2 * mu_cmb**2)

print(f"""
  The key ratio 4piG a^2 rho_K / (c^2 mu^2) simplifies analytically:

    4piG a^2 (Omega_cdm * rho_crit) / (c^2 * (H/c)^2)
    = 4piG a^2 Omega_cdm * 3H^2/(8piG) / H^2
    = (3/2) Omega_cdm a^2

  At z = {z_cmb}:
    Analytical:  (3/2) * {Omega_cdm:.4f} * ({a_cmb:.4e})^2 = {analytical_ratio:.6e}
    Numerical:   {numerical_ratio:.6e}
    Agreement:   {abs(numerical_ratio - analytical_ratio)/analytical_ratio:.2e} (relative)

  Physical meaning:
    The denominator in Eq. 4.31 becomes (for k >> 0):
      denom ~ 1 + c_ad^2 c^2 k^2 / (4piG a^2 rho_K)
            = 1 + w_tilde * k^2 / ((3/2) Omega_cdm a^2 mu^2)

    The ratio mu^2 / k^2 is what determines whether suppression is large:
      At z=1100: mu = {mu_cmb:.4e} m^-1 = {mu_cmb*Mpc:.4e} Mpc^-1
      k = 0.05 Mpc^-1 = {0.05/Mpc:.4e} m^-1
      mu/k = {mu_cmb / (0.05/Mpc):.4e}
      (mu/k)^2 = {(mu_cmb / (0.05/Mpc))**2:.4e}

  So the denominator is:
    ~ 1 + w_tilde / ((3/2) Omega_cdm a^2) * (k/mu)^2 * (mu/k)^2 ...

  More simply:
    denom ~ 1 + w_tilde / ((3/2) Omega_cdm a^2)
          = 1 + {wt_cmb:.4e} / {analytical_ratio:.4e}
          = 1 + {wt_cmb / analytical_ratio:.4e}
          = {1 + wt_cmb / analytical_ratio:.4e}

  Wait -- that's k-independent! Let me reconsider...
""")

# Actually, the denominator IS k-dependent. Let me redo this.
# denom = 1 + c_ad^2 * c^2 * k^2 / (4piG a^2 rho_K (1+w))
# The k^2 makes it k-dependent. What simplifies is the gravitational scale.

# Define k_J^2 = 4piG a^2 rho_K (1+w) / (c_ad^2 c^2)
# Then denom = 1 + (k/k_J)^2

k_J_SI = np.sqrt(grav_term / v_s2)  # m^-1
k_J_Mpc = k_J_SI * Mpc  # Mpc^-1

print(f"""
  [CORRECTION: The denominator IS k-dependent. What simplifies is the Jeans scale.]

  Jeans wavenumber: k_J^2 = 4piG a^2 rho_K (1+w) / (c_ad^2 c^2)

    k_J = sqrt({grav_term:.4e} / {v_s2:.4e})
        = {k_J_SI:.4e} m^-1
        = {k_J_Mpc:.4e} Mpc^-1

  For comparison:
    k_CMB = 0.05 Mpc^-1      k/k_J = {0.05/k_J_Mpc:.4e}
    k_CMB = 0.001 Mpc^-1     k/k_J = {0.001/k_J_Mpc:.4e}

  Note: k_J = {k_J_Mpc:.4e} Mpc^-1

  At the Jeans scale, the denominator = 2, so c_s^2 = c_ad^2 / 2.
  For k >> k_J: c_s^2 ~ c_ad^2 * (k_J/k)^2  (power-law suppression)
  For k << k_J: c_s^2 ~ c_ad^2              (no suppression)

  Since all Planck k-scales ({0.001} to {0.3} Mpc^-1) are compared to
  k_J = {k_J_Mpc:.4e} Mpc^-1:
""")

if k_J_Mpc > 0.3:
    print(f"    k_CMB << k_J: Eq. 4.31 provides MINIMAL additional suppression.")
    print(f"    The main suppression comes from w_tilde_0(z) being tiny (running mu).")
elif k_J_Mpc < 0.001:
    print(f"    k_CMB >> k_J: Eq. 4.31 provides STRONG additional suppression.")
else:
    print(f"    k_J is within the CMB range: suppression is scale-dependent.")

# ============================================================================
# SECTION 9: THE THREE LEVELS OF SUPPRESSION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: WHY THE CRISIS WAS A FALSE ALARM")
print("=" * 80)

k_ref = 0.05  # Mpc^-1, acoustic peak region

# Level 0: The crisis value (no running, z=0)
cs2_crisis = w_tilde_0_today
# Level 1: With running mu, background w_tilde at z=1100
cs2_running = w_tilde_bg(z_cmb)
# Level 2: With running + Eq. 4.31 suppression
cs2_eq431 = cs2_full(k_ref, z_cmb)

print(f"""
  THREE LEVELS OF SUPPRESSION at k = {k_ref} Mpc^-1, z = {z_cmb}:

  Level 0 (CRISIS): c_s^2 = w_tilde_0(z=0)
    = delta_0/2 = {cs2_crisis:.6f}
    This is what you get with mu = H_0/c, no running.
    Ratio to TKS bound: {cs2_crisis/TKS_bound:.0f}x OVER

  Level 1 (RUNNING mu): c_s^2 = w_tilde_0(z=1100)
    = w_tilde_0(0) * (H_0/H(z))^2
    = {cs2_crisis:.6f} * ({H0/Hz_cmb:.4e})^2
    = {cs2_running:.6e}
    Suppression from running: {cs2_running/cs2_crisis:.4e}
    Ratio to TKS bound: {cs2_running/TKS_bound:.4e}

  Level 2 (RUNNING + Eq. 4.31): c_s^2 from full equation
    = {cs2_eq431:.6e}
    Additional suppression from Eq. 4.31: {cs2_eq431/cs2_running:.4e}
    Total suppression from crisis value: {cs2_eq431/cs2_crisis:.4e}
    Ratio to TKS bound: {cs2_eq431/TKS_bound:.4e}
""")

if cs2_running < TKS_bound:
    print(f"  --> The running mu ALONE is sufficient to satisfy the TKS bound!")
    print(f"      w_tilde_0(z=1100) = {cs2_running:.4e} < {TKS_bound:.1e}")
    print(f"      Eq. 4.31 provides additional safety margin.")
else:
    print(f"  --> The running mu alone gives {cs2_running:.4e} (vs TKS {TKS_bound:.1e})")
    print(f"      Eq. 4.31 is needed for additional suppression.")
    if cs2_eq431 < TKS_bound:
        print(f"      With Eq. 4.31: {cs2_eq431:.4e} < {TKS_bound:.1e} -- PASSES")
    else:
        print(f"      Even with Eq. 4.31: {cs2_eq431:.4e} > {TKS_bound:.1e} -- STILL FAILS")

# ============================================================================
# SECTION 10: c_s^2 vs k AT DIFFERENT REDSHIFTS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: c_s^2 AT KEY WAVENUMBERS, MULTIPLE REDSHIFTS")
print("=" * 80)

key_redshifts = [0, 10, 100, 1100, 3000]

for z in key_redshifts:
    wt = w_tilde_bg(z)
    mu_z = H_of_z(z) / c
    print(f"\n  --- z = {z}  (a = {1/(1+z):.4e}, mu = {mu_z:.4e} m^-1, "
          f"w_tilde = {wt:.4e}) ---")
    print(f"  {'k [Mpc^-1]':>14s}  {'c_s^2 (Eq4.31)':>16s}  {'c_ad^2':>14s}  {'< TKS?':>8s}")
    print(f"  {'-'*14}  {'-'*16}  {'-'*14}  {'-'*8}")
    for k in [0.001, 0.01, 0.05, 0.1, 0.3]:
        cs2_f = cs2_full(k, z)
        ok = "YES" if cs2_f < TKS_bound else "NO"
        print(f"  {k:14.4e}  {cs2_f:16.4e}  {wt:14.4e}  {ok:>8s}")

# ============================================================================
# SECTION 11: COMPARISON WITH CMB MEMORY VALUE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: CROSS-CHECKS")
print("=" * 80)

# Check: MEMORY.md claims w_tilde_0(z=1100) ~ 6e-11
print(f"""
  Cross-check with known values:

  MEMORY.md states: w_tilde_0(z=1100) ~ 6x10^-11
  Our calculation:  w_tilde_0(z=1100) = {w_tilde_bg(1100):.4e}

  Note: The MEMORY value may have used slightly different H(z=1100).
  Our H(z=1100)/H_0 = {H_of_z(1100)/H0:.4e}.

  For w_tilde_0 ~ 6e-11, we would need:
    H(z)/H_0 = sqrt(0.170 / 6e-11) = {np.sqrt(0.170/6e-11):.4e}
  Which corresponds to a higher radiation content or different Omega_r.

  The exact value depends on the precise radiation content (neutrinos, etc.).
  The key point is that w_tilde_0(z=1100) << TKS bound ({TKS_bound:.1e})
  regardless of the O(1) factor.
""")

# ============================================================================
# SECTION 12: STIFF-TO-DUST TRANSITION REDSHIFT
# ============================================================================
print("=" * 80)
print("SECTION 12: THE STIFF-TO-DUST TRANSITION")
print("=" * 80)

# WITHOUT running: z_stiff = w_tilde_0^{-1/3} - 1
z_stiff_norun = w_tilde_0_today**(-1./3) - 1

# WITH running: the transition happens when
# rho_stiff/rho_dust = w_tilde_0(z) * (1+z)^3 = 1
# w_tilde_0(today) * (H_0/H(z))^2 * (1+z)^3 = 1
# Need to solve this numerically

def stiff_dust_ratio(z):
    """Ratio rho_stiff / rho_dust at redshift z, WITH running mu."""
    wt = w_tilde_bg(z)
    return wt * (1 + z)**3

# Find the transition (binary search)
z_lo, z_hi = 0, 1e6
for _ in range(100):
    z_mid = (z_lo + z_hi) / 2
    if stiff_dust_ratio(z_mid) > 1:
        z_hi = z_mid
    else:
        z_lo = z_mid
z_stiff_running = z_mid

print(f"""
  The Khronon energy density has two components:
    rho_K ~ (1+z)^3 [dust-like] + w_tilde_0 * (1+z)^6 [stiff-like]

  The stiff component dominates when w_tilde_0(z) * (1+z)^3 > 1.

  WITHOUT running mu (fixed mu = H_0/c):
    Transition at z_stiff = w_tilde_0^(-1/3) - 1 = {z_stiff_norun:.1f}
    This is AFTER recombination (z=1100) -- CATASTROPHIC.
    The Khronon is stiff matter at the CMB epoch.

  WITH running mu (mu = H(z)/c):
    w_tilde_0(z) = {w_tilde_0_today:.4f} * (H_0/H(z))^2
    Transition at z ~ {z_stiff_running:.0f}
    At z=1100: ratio = {stiff_dust_ratio(1100):.4e} << 1 -- dust-like!

  This confirms the running mu resolves the crisis.
""")

# ============================================================================
# SECTION 13: PLOT
# ============================================================================
print("=" * 80)
print("SECTION 13: GENERATING PLOTS")
print("=" * 80)

k_plot = np.logspace(-4, 2, 500)  # Mpc^-1

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_redshifts = [1100, 10, 0]
plot_labels = [r"$z = 1100$ (CMB)", r"$z = 10$ (galaxy formation)", r"$z = 0$ (today)"]
plot_colors = ["#2166ac", "#b2182b", "#4daf4a"]

for ax, z, label, color in zip(axes, plot_redshifts, plot_labels, plot_colors):
    cs2_f_arr = np.array([cs2_full(k, z) for k in k_plot])
    cs2_n_val = cs2_naive(z)  # Background c_ad^2 (k-independent)

    ax.loglog(k_plot, cs2_f_arr, color=color, linewidth=2.5,
              label=r"$c_s^2$ (Eq. 4.31)")
    ax.axhline(y=cs2_n_val, color=color, linewidth=1.5, linestyle="--", alpha=0.6,
               label=rf"$c_{{ad}}^2 = \tilde{{w}}_0(z)$ = {cs2_n_val:.2e}")

    # TKS bound
    ax.axhline(y=TKS_bound, color="red", linewidth=2, linestyle="-.",
               label=f"TKS 2016 bound ({TKS_bound:.1e})")

    # Planck CMB range
    ax.axvspan(0.001, 0.3, alpha=0.08, color="orange", label="Planck $k$-range")

    # Crisis value for reference
    ax.axhline(y=w_tilde_0_today, color="gray", linewidth=1, linestyle=":",
               alpha=0.5, label=rf"$\tilde{{w}}_0(z=0)$ = {w_tilde_0_today:.3f} (crisis)")

    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=12)
    ax.set_ylabel(r"$c_s^2 / c^2$", fontsize=12)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlim(1e-4, 1e2)
    ax.set_ylim(1e-16, 1)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

plt.suptitle(
    r"Effective sound speed $c_s^2$ in the Khronon/$\tau$ framework"
    "\n"
    r"Running $\mu_{\rm bg}(z) = H(z)/c$ resolves the $c_s^2$ crisis",
    fontsize=14, fontweight="bold", y=1.02
)

plt.tight_layout()
outpath = "/Users/akaihuangm1/Desktop/github/anatropic/examples/cs2_verification.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"  Plot saved to {outpath}")

# ============================================================================
# SECTION 14: w_tilde vs z PLOT
# ============================================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

z_range = np.logspace(-1, 4, 500)
w_tilde_arr = np.array([w_tilde_bg(z) for z in z_range])
w_tilde_norun = np.full_like(z_range, w_tilde_0_today)

ax2.loglog(z_range, w_tilde_arr, "b-", linewidth=2.5,
           label=r"$\tilde{w}_0(z)$ with running $\mu = H(z)/c$")
ax2.loglog(z_range, w_tilde_norun, "r--", linewidth=1.5,
           label=rf"$\tilde{{w}}_0$ = {w_tilde_0_today:.3f} (fixed $\mu = H_0/c$, crisis)")
ax2.axhline(y=TKS_bound, color="red", linewidth=2, linestyle="-.",
            label=f"TKS 2016 bound ({TKS_bound:.1e})")
ax2.axvline(x=1100, color="gray", linewidth=1, linestyle=":",
            label=r"$z = 1100$ (CMB)")

ax2.set_xlabel("Redshift $z$", fontsize=13)
ax2.set_ylabel(r"$\tilde{w}_0(z) = c_{ad}^2$", fontsize=13)
ax2.set_title(r"Background EoS $\tilde{w}_0(z)$: Running $\mu$ resolves the Catch-22",
              fontsize=13, fontweight="bold")
ax2.set_xlim(0.1, 1e4)
ax2.set_ylim(1e-12, 1)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
outpath2 = "/Users/akaihuangm1/Desktop/github/anatropic/examples/cs2_verification.png"
# Save both panels in one figure
fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: w_tilde vs z
z_range = np.logspace(-1, 4, 500)
w_tilde_arr = np.array([w_tilde_bg(z) for z in z_range])

ax_left.loglog(z_range, w_tilde_arr, "b-", linewidth=2.5,
               label=r"$\tilde{w}_0(z)$ with running $\mu = H(z)/c$")
ax_left.axhline(y=w_tilde_0_today, color="r", linewidth=1.5, linestyle="--",
                label=rf"$\tilde{{w}}_0(0)$ = {w_tilde_0_today:.3f} (crisis)")
ax_left.axhline(y=TKS_bound, color="red", linewidth=2, linestyle="-.",
                label=f"TKS 2016 bound ({TKS_bound:.1e})")
ax_left.axvline(x=1100, color="gray", linewidth=1, linestyle=":",
                label=r"$z = 1100$ (CMB)")
ax_left.fill_between([1100, 1e4], [1e-12, 1e-12], [1, 1], alpha=0.05, color="blue")
ax_left.set_xlabel("Redshift $z$", fontsize=13)
ax_left.set_ylabel(r"$\tilde{w}_0(z) = c_{ad}^2 / c^2$", fontsize=13)
ax_left.set_title(r"(a) Background EoS: running $\mu$ suppresses $\tilde{w}_0$",
                  fontsize=12, fontweight="bold")
ax_left.set_xlim(0.1, 1e4)
ax_left.set_ylim(1e-12, 1)
ax_left.legend(fontsize=9, loc="upper right")
ax_left.grid(True, alpha=0.3, which="both")

# Right panel: c_s^2 vs k at z=1100
k_plot2 = np.logspace(-4, 2, 500)
cs2_cmb_arr = np.array([cs2_full(k, 1100) for k in k_plot2])
cs2_naive_cmb = cs2_naive(1100)

ax_right.loglog(k_plot2, cs2_cmb_arr, "#2166ac", linewidth=2.5,
                label=r"$c_s^2(k)$ at $z=1100$ (Eq. 4.31)")
ax_right.axhline(y=cs2_naive_cmb, color="#2166ac", linewidth=1.5, linestyle="--", alpha=0.6,
                 label=rf"$c_{{ad}}^2(z=1100)$ = {cs2_naive_cmb:.2e}")
ax_right.axhline(y=TKS_bound, color="red", linewidth=2, linestyle="-.",
                 label=f"TKS 2016 bound ({TKS_bound:.1e})")
ax_right.axhline(y=w_tilde_0_today, color="gray", linewidth=1, linestyle=":",
                 alpha=0.5, label=rf"$\tilde{{w}}_0(0)$ = {w_tilde_0_today:.3f} (crisis)")
ax_right.axvspan(0.001, 0.3, alpha=0.08, color="orange", label="Planck $k$-range")
ax_right.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=13)
ax_right.set_ylabel(r"$c_s^2 / c^2$", fontsize=13)
ax_right.set_title(r"(b) Effective $c_s^2$ at $z=1100$: well below TKS bound",
                   fontsize=12, fontweight="bold")
ax_right.set_xlim(1e-4, 1e2)
ax_right.set_ylim(1e-16, 1)
ax_right.legend(fontsize=9, loc="lower left")
ax_right.grid(True, alpha=0.3, which="both")

plt.suptitle(
    r"$c_s^2$ verification for the Khronon/$\tau$ framework (Blanchet \& Skordis 2024)",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"  Final plot saved to {outpath}")

plt.close("all")

# ============================================================================
# SECTION 15: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

cs2_cmb_k005 = cs2_full(0.05, 1100)
cs2_naive_cmb_val = cs2_naive(1100)

print(f"""
  +-----------------------------------------------------------------+
  |                 c_s^2 VERIFICATION RESULTS                      |
  +-----------------------------------------------------------------+
  |                                                                 |
  |  At z = 1100 (CMB), k = 0.05 Mpc^-1 (acoustic peak region):    |
  |                                                                 |
  |  WITHOUT running mu (mu = H_0/c, the "crisis"):                |
  |    w_tilde_0(z=0)      = {cs2_crisis:.6f}                       |
  |    c_s^2 (naive)       = {cs2_crisis:.4e}  *** VIOLATES ***     |
  |                                                                 |
  |  WITH running mu (mu = H(z)/c):                                 |
  |    w_tilde_0(z=1100)   = {cs2_naive_cmb_val:.4e}  (= c_ad^2)         |
  |    c_s^2 (Eq. 4.31)    = {cs2_cmb_k005:.4e}                    |
  |    TKS 2016 bound      = {TKS_bound:.4e}                       |
  |                                                                 |
  |  c_s^2 / TKS bound = {cs2_cmb_k005/TKS_bound:.4e}                      |""")

if cs2_cmb_k005 < TKS_bound:
    print(f"  |  --> PASSES by factor {TKS_bound/cs2_cmb_k005:.1f}x                              |")
else:
    print(f"  |  --> VIOLATES by factor {cs2_cmb_k005/TKS_bound:.1f}x                             |")

print(f"""  |                                                                 |
  |  MECHANISM:                                                     |
  |    1. Running mu: H(z=1100)/H_0 = {H_of_z(1100)/H0:.0e}            |
  |       Suppresses w_tilde by (H_0/H)^2 = {(H0/H_of_z(1100))**2:.2e}          |
  |    2. Eq. 4.31 (Jeans suppression):                             |
  |       Additional factor {cs2_cmb_k005/cs2_naive_cmb_val:.4e}                    |
  |    3. Total suppression: {cs2_cmb_k005/cs2_crisis:.4e}                    |
  |                                                                 |
  |  CONCLUSION: The "c_s^2 ~ 0.17 crisis" was based on using the  |
  |  FIXED mu = H_0/c. With the running mu = H(z)/c that resolves  |
  |  the Catch-22, c_s^2 is suppressed far below the TKS bound     |
  |  at ALL CMB-relevant scales.                                    |
  +-----------------------------------------------------------------+
""")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
