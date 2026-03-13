"""
Linear Jeans instability test for the Anatropic Euler + gravity solver.

Tests two regimes:
  a) k < k_J (wavelength > Jeans length): perturbation GROWS exponentially
     with rate omega = sqrt(4*pi*G*rho0 - k^2*cs^2)
  b) k > k_J (wavelength < Jeans length): perturbation OSCILLATES
     with frequency omega = sqrt(k^2*cs^2 - 4*pi*G*rho0)

Uses isothermal EOS: P = cs^2 * rho, with small perturbation
delta_rho = A * sin(2*pi*x/lambda).

Saves a plot to tests/jeans_result.png.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anatropic.eos import IsothermalEOS
from anatropic.euler import evolve, compute_dt, _add_ghost_cells
from anatropic.gravity import solve_gravity


def run_jeans_unstable():
    """
    Test case (a): k < k_J -- unstable, exponential growth.

    Returns
    -------
    omega_numerical : float
        Measured growth rate.
    omega_analytical : float
        Expected growth rate.
    rel_error : float
        Relative error |omega_num - omega_ana| / omega_ana.
    times : ndarray
        Array of measurement times.
    amplitudes : ndarray
        Measured density perturbation amplitudes at each time.
    """
    # --- Parameters ---
    # Choose units so that the Jeans scale is easily resolved.
    G = 1.0
    rho0 = 1.0
    cs = 0.1     # low sound speed -> small Jeans length

    # Jeans wavenumber: k_J = sqrt(4*pi*G*rho0) / cs
    k_J = np.sqrt(4.0 * np.pi * G * rho0) / cs
    lambda_J = 2.0 * np.pi / k_J

    # Choose wavelength > lambda_J (unstable): use lambda = 4 * lambda_J
    lam = 4.0 * lambda_J
    k = 2.0 * np.pi / lam

    # Domain = one wavelength (periodic)
    L = lam
    N = 256
    dx = L / N
    x = (np.arange(N) + 0.5) * dx

    # Analytical growth rate
    omega_sq = 4.0 * np.pi * G * rho0 - k**2 * cs**2
    assert omega_sq > 0, "This should be the unstable regime"
    omega_analytical = np.sqrt(omega_sq)

    # Growth timescale
    t_grow = 1.0 / omega_analytical

    # Run for a few growth times (enough to measure growth, but stay in linear regime)
    A = 1e-6 * rho0  # very small perturbation amplitude
    t_final = 3.0 * t_grow

    # --- EOS ---
    eos = IsothermalEOS(cs)

    # --- Initial condition ---
    rho = rho0 + A * np.sin(k * x)
    v = np.zeros(N)

    # Conserved variables with isothermal EOS
    # We use the full 3-component Euler solver with IsothermalEOS
    eint = eos.internal_energy(rho, None)  # returns cs^2 everywhere
    U = np.zeros((3, N))
    U[0] = rho
    U[1] = rho * v
    U[2] = rho * (eint + 0.5 * v**2)

    # --- Time evolution with gravity ---
    t = 0.0
    step = 0
    cfl = 0.3  # conservative CFL for stability with gravity

    # Record amplitude vs time
    times = [0.0]
    amplitudes = [A]

    n_samples = 50
    dt_sample = t_final / n_samples
    next_sample = dt_sample

    while t < t_final:
        dt = compute_dt(U, dx, eos, cfl=cfl)

        # Gravity: solve for acceleration
        rho_current = np.maximum(U[0], 1e-30)
        g = solve_gravity(rho_current, dx, G=G)

        if t + dt > t_final:
            dt = t_final - t

        # Evolve with gravity source
        U = evolve(U, dx, dt, eos, gravity_source=g)
        t += dt
        step += 1

        # Sample amplitude
        if t >= next_sample or t >= t_final:
            rho_now = U[0]
            delta_rho = rho_now - rho0
            # Extract amplitude of the sin(kx) mode via projection
            amp = 2.0 * np.mean(delta_rho * np.sin(k * x))
            times.append(t)
            amplitudes.append(abs(amp))
            next_sample += dt_sample

    times = np.array(times)
    amplitudes = np.array(amplitudes)

    # --- Measure growth rate by fitting log(amplitude) vs time ---
    # Use points where amplitude is still in linear regime (< 100 * A)
    mask = (amplitudes > 0) & (amplitudes < 100 * A) & (times > 0.5 * t_grow)
    if np.sum(mask) < 3:
        # Fallback: use all positive-amplitude points
        mask = amplitudes > 0

    log_amp = np.log(amplitudes[mask])
    t_fit = times[mask]

    # Linear fit: log(A(t)) = log(A0) + omega * t
    if len(t_fit) >= 2:
        coeffs = np.polyfit(t_fit, log_amp, 1)
        omega_numerical = coeffs[0]
    else:
        omega_numerical = 0.0

    rel_error = abs(omega_numerical - omega_analytical) / omega_analytical

    print(f"  Jeans UNSTABLE test (k/k_J = {k/k_J:.3f}):")
    print(f"    omega_analytical = {omega_analytical:.6f}")
    print(f"    omega_numerical  = {omega_numerical:.6f}")
    print(f"    relative error   = {rel_error:.4f} ({rel_error*100:.2f}%)")

    return omega_numerical, omega_analytical, rel_error, times, amplitudes


def run_jeans_stable():
    """
    Test case (b): k > k_J -- stable, oscillating perturbation.

    Returns
    -------
    omega_numerical : float
        Measured oscillation frequency.
    omega_analytical : float
        Expected oscillation frequency.
    rel_error : float
        Relative error.
    times : ndarray
        Measurement times.
    amplitudes : ndarray
        Measured density perturbation amplitudes.
    """
    # --- Parameters ---
    G = 1.0
    rho0 = 1.0
    cs = 1.0  # higher sound speed -> larger Jeans length

    k_J = np.sqrt(4.0 * np.pi * G * rho0) / cs
    lambda_J = 2.0 * np.pi / k_J

    # Choose wavelength < lambda_J (stable): use lambda = lambda_J / 4
    lam = lambda_J / 4.0
    k = 2.0 * np.pi / lam

    L = lam
    N = 128
    dx = L / N
    x = (np.arange(N) + 0.5) * dx

    # Analytical oscillation frequency
    omega_sq = k**2 * cs**2 - 4.0 * np.pi * G * rho0
    assert omega_sq > 0, "This should be the stable regime"
    omega_analytical = np.sqrt(omega_sq)

    T_osc = 2.0 * np.pi / omega_analytical
    t_final = 3.0 * T_osc  # run for 3 oscillation periods

    A = 1e-6 * rho0
    eos = IsothermalEOS(cs)

    # Initial condition
    rho = rho0 + A * np.sin(k * x)
    v = np.zeros(N)
    eint = eos.internal_energy(rho, None)
    U = np.zeros((3, N))
    U[0] = rho
    U[1] = rho * v
    U[2] = rho * (eint + 0.5 * v**2)

    t = 0.0
    step = 0
    cfl = 0.3

    times = [0.0]
    amplitudes = [A]

    n_samples = 200
    dt_sample = t_final / n_samples
    next_sample = dt_sample

    while t < t_final:
        dt = compute_dt(U, dx, eos, cfl=cfl)
        rho_current = np.maximum(U[0], 1e-30)
        g = solve_gravity(rho_current, dx, G=G)

        if t + dt > t_final:
            dt = t_final - t

        U = evolve(U, dx, dt, eos, gravity_source=g)
        t += dt
        step += 1

        if t >= next_sample or t >= t_final:
            rho_now = U[0]
            delta_rho = rho_now - rho0
            amp = 2.0 * np.mean(delta_rho * np.sin(k * x))
            times.append(t)
            amplitudes.append(amp)  # keep sign for oscillation
            next_sample += dt_sample

    times = np.array(times)
    amplitudes = np.array(amplitudes)

    # Measure oscillation frequency from zero crossings
    # Find where amplitude changes sign
    sign_changes = np.where(np.diff(np.sign(amplitudes)))[0]
    if len(sign_changes) >= 2:
        # Half-period between consecutive zero crossings
        half_periods = np.diff(times[sign_changes])
        T_measured = 2.0 * np.mean(half_periods)
        omega_numerical = 2.0 * np.pi / T_measured
    else:
        # Fallback: FFT
        dt_avg = np.mean(np.diff(times))
        fft_amp = np.abs(np.fft.rfft(amplitudes))
        freqs = np.fft.rfftfreq(len(amplitudes), d=dt_avg)
        # Skip DC component
        idx_peak = np.argmax(fft_amp[1:]) + 1
        omega_numerical = 2.0 * np.pi * freqs[idx_peak]

    rel_error = abs(omega_numerical - omega_analytical) / omega_analytical

    print(f"  Jeans STABLE test (k/k_J = {k/k_J:.3f}):")
    print(f"    omega_analytical = {omega_analytical:.6f}")
    print(f"    omega_numerical  = {omega_numerical:.6f}")
    print(f"    relative error   = {rel_error:.4f} ({rel_error*100:.2f}%)")

    return omega_numerical, omega_analytical, rel_error, times, amplitudes


def run_jeans_test():
    """
    Run both Jeans instability tests and produce a combined plot.

    Returns
    -------
    passed : bool
        True if both tests pass their error tolerances.
    """
    print("Running Jeans instability tests...")

    # Run both cases
    (omega_u_num, omega_u_ana, err_u,
     times_u, amps_u) = run_jeans_unstable()
    (omega_s_num, omega_s_ana, err_s,
     times_s, amps_s) = run_jeans_stable()

    # --- Plot ---
    plot_path = os.path.join(os.path.dirname(__file__), 'jeans_result.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Unstable case: log-scale amplitude vs time
    axes[0].semilogy(times_u, amps_u, 'b.-', markersize=4, label='Numerical')
    # Analytical prediction
    t_ana = np.linspace(0, times_u[-1], 200)
    A0 = amps_u[0]
    axes[0].semilogy(t_ana, A0 * np.exp(omega_u_ana * t_ana), 'r--',
                     linewidth=2, label=f'Analytical ($\\omega$={omega_u_ana:.4f})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude $|\\delta\\rho|$')
    axes[0].set_title(f'Jeans Unstable ($k < k_J$)\n'
                      f'$\\omega_{{num}}$={omega_u_num:.4f}, '
                      f'error={err_u*100:.1f}%')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Stable case: amplitude vs time (should oscillate)
    axes[1].plot(times_s, amps_s, 'b.-', markersize=3, label='Numerical')
    # Analytical prediction
    t_ana_s = np.linspace(0, times_s[-1], 500)
    A0_s = amps_s[0]
    axes[1].plot(t_ana_s, A0_s * np.cos(omega_s_ana * t_ana_s), 'r--',
                 linewidth=1.5, label=f'Analytical ($\\omega$={omega_s_ana:.4f})')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude $\\delta\\rho$')
    axes[1].set_title(f'Jeans Stable ($k > k_J$)\n'
                      f'$\\omega_{{num}}$={omega_s_num:.4f}, '
                      f'error={err_s*100:.1f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")

    # --- Pass/fail ---
    # Tolerance: 15% for growth rate, 15% for oscillation frequency
    # (first-order Godunov has significant numerical dissipation)
    tol_unstable = 0.15
    tol_stable = 0.15

    pass_unstable = err_u < tol_unstable
    pass_stable = err_s < tol_stable

    if not pass_unstable:
        print(f"  WARNING: Unstable test failed (error {err_u*100:.1f}% > {tol_unstable*100:.0f}%)")
    if not pass_stable:
        print(f"  WARNING: Stable test failed (error {err_s*100:.1f}% > {tol_stable*100:.0f}%)")

    return pass_unstable and pass_stable


if __name__ == '__main__':
    passed = run_jeans_test()
    status = "PASS" if passed else "FAIL"
    print(f"\nJeans instability test: {status}")
    sys.exit(0 if passed else 1)
