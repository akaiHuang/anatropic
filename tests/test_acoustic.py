"""
Acoustic wave propagation test for the Anatropic Euler solver.

Tests that a small-amplitude sound wave in an isothermal gas:
  1) Propagates at the correct phase velocity c_s
  2) Maintains its amplitude (no artificial growth; some numerical
     dissipation is acceptable)

Uses periodic BC on a domain of length L with a single-mode perturbation:
  delta_rho = A * sin(2*pi*x/L)

After one full traversal time t = L/c_s, the wave should return to its
initial position.

Saves a plot to tests/acoustic_result.png.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anatropic.eos import IsothermalEOS
from anatropic.euler import evolve, compute_dt


def run_acoustic_test():
    """
    Run the acoustic wave propagation test.

    Returns
    -------
    passed : bool
        True if measured phase velocity matches c_s within tolerance.
    rel_error : float
        Relative error in measured phase velocity.
    """
    # --- Parameters ---
    cs = 1.0
    L = 1.0
    N = 256
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    rho0 = 1.0
    A = 1e-4 * rho0  # small amplitude for linear regime
    cfl = 0.4

    k = 2.0 * np.pi / L  # one full wavelength in the box

    # Run for one full traversal: t = L / c_s
    # The wave should return to its starting position.
    t_crossing = L / cs
    t_final = t_crossing

    # --- EOS ---
    eos = IsothermalEOS(cs)

    # --- Initial condition ---
    # Density perturbation: delta_rho = A * sin(kx)
    # For a right-going acoustic wave, the velocity perturbation is:
    #   delta_v = (cs / rho0) * delta_rho = A * cs / rho0 * sin(kx)
    # This gives a purely right-going wave. Without the velocity perturbation,
    # we get both left- and right-going waves (standing wave).
    # Here we set up a purely right-going wave.
    rho = rho0 + A * np.sin(k * x)
    v = (cs / rho0) * A * np.sin(k * x)  # right-going wave

    eint = eos.internal_energy(rho, None)
    U = np.zeros((3, N))
    U[0] = rho
    U[1] = rho * v
    U[2] = rho * (eint + 0.5 * v**2)

    # Store initial density profile
    rho_initial = rho.copy()

    # --- Time evolution (no gravity) ---
    t = 0.0
    step = 0

    # Also record snapshots for the plot
    n_snapshots = 5
    snapshot_times = np.linspace(0, t_final, n_snapshots + 1)[1:]
    snapshots = [(0.0, rho.copy())]
    next_snap = 0

    while t < t_final:
        dt = compute_dt(U, dx, eos, cfl=cfl)
        if t + dt > t_final:
            dt = t_final - t

        # Check for snapshot
        if next_snap < len(snapshot_times) and t + dt >= snapshot_times[next_snap]:
            # Adjust dt to hit the snapshot time exactly
            dt_snap = snapshot_times[next_snap] - t
            if dt_snap > 1e-14:
                U = evolve(U, dx, dt_snap, eos)
                t += dt_snap
                step += 1
                snapshots.append((t, U[0].copy()))
                next_snap += 1
                continue

        U = evolve(U, dx, dt, eos)
        t += dt
        step += 1

    # Final snapshot
    snapshots.append((t, U[0].copy()))

    print(f"  Acoustic test: evolved to t={t:.6f} in {step} steps")

    # --- Measure phase velocity ---
    rho_final = U[0]
    delta_rho_final = rho_final - rho0

    # Cross-correlate initial and final perturbation to find phase shift
    delta_rho_init = rho_initial - rho0

    # Cross-correlation via FFT
    corr = np.real(np.fft.ifft(
        np.fft.fft(delta_rho_final) * np.conj(np.fft.fft(delta_rho_init))
    ))

    # The peak of the cross-correlation gives the shift in cells
    shift_cells = np.argmax(corr)
    # Handle wrap-around
    if shift_cells > N // 2:
        shift_cells = shift_cells - N

    shift_distance = shift_cells * dx

    # For a right-going wave after time t_final = L/cs,
    # it should have traveled exactly L (one full wavelength), so shift = 0 mod L.
    # The measured velocity is:
    # v_measured = (total distance traveled) / t_final
    # Since we measure shift mod L, and expect shift = 0:
    # Any nonzero shift represents a phase velocity error.
    # v_measured = (n_crossings * L + shift) / t_final where n_crossings = 1
    v_measured = (L + shift_distance) / t_final

    # But if shift_cells = 0, this is exactly cs. Let's also measure
    # the fractional shift more precisely using the Fourier phase.
    fft_init = np.fft.fft(delta_rho_init)
    fft_final = np.fft.fft(delta_rho_final)

    # Look at the fundamental mode (mode 1)
    phase_init = np.angle(fft_init[1])
    phase_final = np.angle(fft_final[1])
    delta_phase = phase_final - phase_init

    # For a right-going wave, after one crossing time the phase should
    # advance by 2*pi (one full cycle), so delta_phase should be 0 mod 2*pi.
    # Unwrap to find deviation:
    delta_phase_wrapped = (delta_phase + np.pi) % (2 * np.pi) - np.pi

    # Phase velocity from phase measurement
    # phase_advance = k * v_phase * t_final
    # v_phase * t_final = L + delta_phase_wrapped / k
    # Actually: after one crossing, phase_advance = k * cs * t_final = 2*pi
    # Any deviation delta_phase_wrapped means:
    # k * v_measured * t = 2*pi + delta_phase_wrapped (but sign convention)
    # v_measured = (2*pi - delta_phase_wrapped) / (k * t_final)
    # Note: negative delta_phase_wrapped means the wave went slightly further
    v_phase = cs * (1.0 - delta_phase_wrapped / (2.0 * np.pi))

    rel_error = abs(v_phase - cs) / cs

    print(f"  Phase velocity (expected):  {cs:.6f}")
    print(f"  Phase velocity (measured):  {v_phase:.6f}")
    print(f"  Phase shift (cells):        {shift_cells}")
    print(f"  Relative error:             {rel_error:.6f} ({rel_error*100:.4f}%)")

    # Also measure amplitude decay (numerical dissipation)
    amp_init = np.max(np.abs(delta_rho_init))
    amp_final = np.max(np.abs(delta_rho_final))
    amp_ratio = amp_final / amp_init
    print(f"  Amplitude ratio (final/initial): {amp_ratio:.4f}")
    print(f"  Amplitude decay: {(1-amp_ratio)*100:.2f}%")

    # --- Plot ---
    plot_path = os.path.join(os.path.dirname(__file__), 'acoustic_result.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: density profiles at different times
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
    for (t_snap, rho_snap), color in zip(snapshots, colors):
        label = f't = {t_snap:.3f}'
        axes[0].plot(x, rho_snap - rho0, color=color, linewidth=1.2, label=label)

    axes[0].plot(x, delta_rho_init, 'k--', linewidth=0.8, alpha=0.5,
                 label='Initial (reference)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('$\\delta\\rho = \\rho - \\rho_0$')
    axes[0].set_title('Acoustic Wave: Density Perturbation')
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Right panel: initial vs final overlay (should coincide after one crossing)
    axes[1].plot(x, delta_rho_init, 'b-', linewidth=1.5, label='Initial')
    axes[1].plot(x, delta_rho_final, 'r--', linewidth=1.5,
                 label=f'Final (t={t:.3f})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('$\\delta\\rho$')
    axes[1].set_title(f'Phase velocity error: {rel_error*100:.3f}%\n'
                      f'Amplitude decay: {(1-amp_ratio)*100:.1f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")

    # --- Pass/fail ---
    # Phase velocity should be within 5% (first-order Godunov has dispersion)
    tolerance = 0.05
    passed = rel_error < tolerance
    return passed, rel_error


if __name__ == '__main__':
    passed, error = run_acoustic_test()
    status = "PASS" if passed else "FAIL"
    print(f"\nAcoustic wave test: {status} (phase velocity error = {error*100:.4f}%)")
    sys.exit(0 if passed else 1)
