"""
Sod shock tube test for the Anatropic Euler solver.

Standard Sod problem (Sod 1978):
  Left state:  rho=1.0, P=1.0, v=0.0
  Right state: rho=0.125, P=0.1, v=0.0
  Ideal gas gamma=1.4
  Domain [0, 1], diaphragm at x=0.5
  Run to t=0.2, N=200 cells

Compares the numerical solution against the exact Riemann solution.
Saves a plot to tests/sod_result.png.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anatropic.eos import IdealGasEOS
from anatropic.euler import evolve, compute_dt


# ========================================================================
# Exact Riemann solver for the Sod problem
# ========================================================================

def exact_sod(x, t, x0=0.5, gamma=1.4,
              rho_L=1.0, v_L=0.0, P_L=1.0,
              rho_R=0.125, v_R=0.0, P_R=0.1):
    """
    Exact solution to the Sod shock tube problem.

    Solves the Riemann problem for the 1D Euler equations with the given
    left/right initial states. Returns density, velocity, and pressure
    at positions x and time t.

    The solution consists of (from left to right):
    1. Left state
    2. Rarefaction fan (left-going)
    3. Contact discontinuity region (star-left)
    4. Contact discontinuity
    5. Shock region (star-right)
    6. Shock front (right-going)
    7. Right state
    """
    # Sound speeds
    cs_L = np.sqrt(gamma * P_L / rho_L)
    cs_R = np.sqrt(gamma * P_R / rho_R)

    # Solve for the star-region pressure P_star using Newton-Raphson
    # The pressure function f(P) = f_L(P) + f_R(P) + v_R - v_L = 0
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    def f_and_df(P_star):
        """Pressure function and its derivative for the exact Riemann solver."""
        # Left wave (rarefaction for Sod)
        if P_star <= P_L:
            # Rarefaction
            ratio = P_star / P_L
            power = gm1 / (2.0 * gamma)
            f_L = (2.0 * cs_L / gm1) * (ratio**power - 1.0)
            df_L = (cs_L / (gamma * P_L)) * ratio**(power - 1.0)
        else:
            # Shock
            A_L = 2.0 / (gp1 * rho_L)
            B_L = gm1 / gp1 * P_L
            sq = np.sqrt(A_L / (P_star + B_L))
            f_L = (P_star - P_L) * sq
            df_L = sq * (1.0 - (P_star - P_L) / (2.0 * (P_star + B_L)))

        # Right wave (shock for Sod)
        if P_star <= P_R:
            # Rarefaction
            ratio = P_star / P_R
            power = gm1 / (2.0 * gamma)
            f_R = (2.0 * cs_R / gm1) * (ratio**power - 1.0)
            df_R = (cs_R / (gamma * P_R)) * ratio**(power - 1.0)
        else:
            # Shock
            A_R = 2.0 / (gp1 * rho_R)
            B_R = gm1 / gp1 * P_R
            sq = np.sqrt(A_R / (P_star + B_R))
            f_R = (P_star - P_R) * sq
            df_R = sq * (1.0 - (P_star - P_R) / (2.0 * (P_star + B_R)))

        f = f_L + f_R + (v_R - v_L)
        df = df_L + df_R
        return f, df

    # Newton-Raphson iteration for P_star
    # Initial guess: arithmetic mean
    P_star = 0.5 * (P_L + P_R)
    for _ in range(100):
        f, df = f_and_df(P_star)
        if abs(df) < 1e-30:
            break
        dP = -f / df
        P_star = max(P_star + dP, 1e-10)
        if abs(dP) < 1e-12 * P_star:
            break

    # Star-region velocity
    # From left wave contribution
    if P_star <= P_L:
        ratio = P_star / P_L
        power = gm1 / (2.0 * gamma)
        f_L = (2.0 * cs_L / gm1) * (ratio**power - 1.0)
    else:
        A_L = 2.0 / (gp1 * rho_L)
        B_L = gm1 / gp1 * P_L
        f_L = (P_star - P_L) * np.sqrt(A_L / (P_star + B_L))

    v_star = 0.5 * (v_L + v_R) + 0.5 * (f_L - (- f_L - (v_R - v_L)))
    # More directly:
    if P_star <= P_R:
        ratio = P_star / P_R
        power = gm1 / (2.0 * gamma)
        f_R = (2.0 * cs_R / gm1) * (ratio**power - 1.0)
    else:
        A_R = 2.0 / (gp1 * rho_R)
        B_R = gm1 / gp1 * P_R
        f_R = (P_star - P_R) * np.sqrt(A_R / (P_star + B_R))
    v_star = 0.5 * (v_L + v_R) + 0.5 * (f_R - f_L)

    # Star-region densities
    # Left of contact (rarefaction side for standard Sod)
    if P_star <= P_L:
        rho_star_L = rho_L * (P_star / P_L) ** (1.0 / gamma)
    else:
        rho_star_L = rho_L * ((P_star / P_L + gm1 / gp1) /
                               (gm1 / gp1 * P_star / P_L + 1.0))

    # Right of contact (shock side for standard Sod)
    if P_star <= P_R:
        rho_star_R = rho_R * (P_star / P_R) ** (1.0 / gamma)
    else:
        rho_star_R = rho_R * ((P_star / P_R + gm1 / gp1) /
                               (gm1 / gp1 * P_star / P_R + 1.0))

    # Sound speed in star region (left side)
    cs_star_L = np.sqrt(gamma * P_star / rho_star_L)

    # Now sample the solution at each x position
    rho_out = np.zeros_like(x)
    v_out = np.zeros_like(x)
    P_out = np.zeros_like(x)

    for i, xi in enumerate(x):
        S = (xi - x0) / t  # characteristic speed at this position

        if S <= v_star:
            # Left of contact discontinuity
            if P_star <= P_L:
                # Left rarefaction
                S_head = v_L - cs_L  # head of rarefaction
                S_tail = v_star - cs_star_L  # tail of rarefaction

                if S <= S_head:
                    # Undisturbed left state
                    rho_out[i] = rho_L
                    v_out[i] = v_L
                    P_out[i] = P_L
                elif S >= S_tail:
                    # Star region (left of contact)
                    rho_out[i] = rho_star_L
                    v_out[i] = v_star
                    P_out[i] = P_star
                else:
                    # Inside rarefaction fan
                    # Self-similar solution
                    cs_fan = (2.0 / gp1) * (cs_L + 0.5 * gm1 * (v_L - S))
                    v_fan = (2.0 / gp1) * (cs_L + 0.5 * gm1 * v_L + S)
                    rho_fan = rho_L * (cs_fan / cs_L) ** (2.0 / gm1)
                    P_fan = P_L * (cs_fan / cs_L) ** (2.0 * gamma / gm1)

                    rho_out[i] = rho_fan
                    v_out[i] = v_fan
                    P_out[i] = P_fan
            else:
                # Left shock
                S_shock = v_L - cs_L * np.sqrt(gp1 / (2.0 * gamma) *
                          P_star / P_L + gm1 / (2.0 * gamma))
                if S <= S_shock:
                    rho_out[i] = rho_L
                    v_out[i] = v_L
                    P_out[i] = P_L
                else:
                    rho_out[i] = rho_star_L
                    v_out[i] = v_star
                    P_out[i] = P_star
        else:
            # Right of contact discontinuity
            if P_star <= P_R:
                # Right rarefaction
                cs_star_R = np.sqrt(gamma * P_star / rho_star_R)
                S_head = v_R + cs_R
                S_tail = v_star + cs_star_R

                if S >= S_head:
                    rho_out[i] = rho_R
                    v_out[i] = v_R
                    P_out[i] = P_R
                elif S <= S_tail:
                    rho_out[i] = rho_star_R
                    v_out[i] = v_star
                    P_out[i] = P_star
                else:
                    cs_fan = (2.0 / gp1) * (-cs_R + 0.5 * gm1 * (v_R - S))
                    cs_fan = abs(cs_fan)
                    v_fan = (2.0 / gp1) * (-cs_R + 0.5 * gm1 * v_R + S)
                    rho_fan = rho_R * (cs_fan / cs_R) ** (2.0 / gm1)
                    P_fan = P_R * (cs_fan / cs_R) ** (2.0 * gamma / gm1)
                    rho_out[i] = rho_fan
                    v_out[i] = v_fan
                    P_out[i] = P_fan
            else:
                # Right shock
                S_shock = v_R + cs_R * np.sqrt(gp1 / (2.0 * gamma) *
                          P_star / P_R + gm1 / (2.0 * gamma))
                if S >= S_shock:
                    rho_out[i] = rho_R
                    v_out[i] = v_R
                    P_out[i] = P_R
                else:
                    rho_out[i] = rho_star_R
                    v_out[i] = v_star
                    P_out[i] = P_star

    return rho_out, v_out, P_out


# ========================================================================
# Test runner
# ========================================================================

def run_sod_test():
    """
    Run the Sod shock tube test.

    Returns
    -------
    passed : bool
        True if the maximum density error is below the tolerance.
    max_error : float
        Maximum absolute error in density.
    """
    # --- Parameters ---
    # We use an extended domain [-0.5, 1.5] so that waves from the Sod
    # diaphragm at x0=0.5 do not reach the periodic boundaries by t=0.2.
    # (The fastest wave is the left rarefaction head at v-cs ~ -1.18,
    #  traveling ~ 0.24 by t=0.2, well within the padding.)
    # The comparison against the exact solution is done on [0.1, 0.9].
    N = 400  # 200 cells per unit length on [-0.5, 1.5]
    x_min, x_max = -0.5, 1.5
    gamma = 1.4
    t_final = 0.2
    cfl = 0.5
    x0 = 0.5  # diaphragm location

    dx = (x_max - x_min) / N
    x = x_min + (np.arange(N) + 0.5) * dx

    # --- EOS ---
    eos = IdealGasEOS(gamma=gamma)

    # --- Initial condition ---
    rho = np.where(x < x0, 1.0, 0.125)
    v = np.zeros(N)
    P = np.where(x < x0, 1.0, 0.1)

    # Convert to conserved variables: U has shape (3, N)
    eint = eos.internal_energy(rho, P)
    U = np.zeros((3, N))
    U[0] = rho
    U[1] = rho * v
    U[2] = rho * (eint + 0.5 * v**2)

    # --- Time integration ---
    t = 0.0
    step = 0
    while t < t_final:
        dt = compute_dt(U, dx, eos, cfl=cfl)
        if t + dt > t_final:
            dt = t_final - t
        U = evolve(U, dx, dt, eos)
        t += dt
        step += 1

    print(f"  Sod test: evolved to t={t:.6f} in {step} steps")

    # --- Extract numerical solution ---
    rho_num = U[0]
    v_num = U[1] / rho_num
    P_num = (gamma - 1.0) * (U[2] - 0.5 * rho_num * v_num**2)

    # --- Exact solution ---
    rho_exact, v_exact, P_exact = exact_sod(x, t_final, x0=x0, gamma=gamma)

    # --- Compute errors ---
    # Compare only in the well-resolved interior [0.1, 0.9]
    interior = (x > 0.1) & (x < 0.9)
    error_rho = np.abs(rho_num[interior] - rho_exact[interior])
    max_error = np.max(error_rho)
    l1_error = np.mean(error_rho)

    print(f"  Max density error (interior): {max_error:.6f}")
    print(f"  L1 density error (interior):  {l1_error:.6f}")

    # --- Plot ---
    # Show only the physical region [0, 1] in the plots
    plot_path = os.path.join(os.path.dirname(__file__), 'sod_result.png')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    plot_mask = (x >= 0.0) & (x <= 1.0)
    xp = x[plot_mask]

    # Density
    axes[0].plot(xp, rho_exact[plot_mask], 'k-', linewidth=1.5, label='Exact')
    axes[0].plot(xp, rho_num[plot_mask], 'ro', markersize=2, label='Numerical')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Sod Shock Tube: Density')
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # Velocity
    axes[1].plot(xp, v_exact[plot_mask], 'k-', linewidth=1.5, label='Exact')
    axes[1].plot(xp, v_num[plot_mask], 'bo', markersize=2, label='Numerical')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Sod Shock Tube: Velocity')
    axes[1].legend()
    axes[1].set_xlim(0, 1)

    # Pressure
    axes[2].plot(xp, P_exact[plot_mask], 'k-', linewidth=1.5, label='Exact')
    axes[2].plot(xp, P_num[plot_mask], 'gs', markersize=2, label='Numerical')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Pressure')
    axes[2].set_title('Sod Shock Tube: Pressure')
    axes[2].legend()
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")

    # --- Pass/fail ---
    # For first-order Godunov + HLLE at this resolution, max error is dominated
    # by numerical smearing of discontinuities (contact, shock). Typical max
    # error is ~0.05-0.15. L1 error is a better measure of overall accuracy.
    tolerance = 0.20
    passed = max_error < tolerance
    return passed, max_error


if __name__ == '__main__':
    passed, error = run_sod_test()
    status = "PASS" if passed else "FAIL"
    print(f"\nSod shock tube test: {status} (max density error = {error:.6f})")
    sys.exit(0 if passed else 1)
