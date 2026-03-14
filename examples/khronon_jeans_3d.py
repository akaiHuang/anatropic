#!/usr/bin/env python3
"""
3D Khronon Jeans Instability Test
==================================

First 3D self-gravitating simulation using the Anatropic solver.

Sets up an isothermal gas with self-gravity in a periodic 64^3 box, seeds
random density perturbations, and evolves for several free-fall times.
All modes with wavelength > lambda_J are gravitationally unstable; with
cs = 0.05, lambda_J ~ 0.3 so nearly every mode in the L=1 box is unstable.

Outputs:
    - Midplane density slices at selected times
    - 3D spherically-averaged power spectrum P(k)
    - Saved npz snapshot of the final state

Author: Anatropic project
"""

import os
import sys
import time as walltime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

# Ensure anatropic is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from anatropic.eos import IsothermalEOS
from anatropic.simulation3d import Simulation3D


# =============================================================================
# Physical parameters
# =============================================================================

G = 1.0
RHO0 = 1.0
L = 1.0           # Box size
N3D = 64           # Grid resolution per side (64^3)
CS = 0.05          # Sound speed (lambda_J ~ cs * sqrt(pi / (G*rho0)) ~ 0.3)
A_PERT = 1e-2      # Perturbation amplitude (1% for visible growth)
CFL = 0.05         # Low CFL for accurate gravity integration (~20 steps/t_ff)
SEED = 42
USE_MODE_PERT = True  # Also add coherent mode for clear instability signal

# Free-fall time
T_FF = 1.0 / np.sqrt(4.0 * np.pi * G * RHO0)
T_END = 5.0 * T_FF   # Run to 5 t_ff (well into nonlinear for mode-1)

# Jeans length
LAMBDA_J = CS * np.sqrt(np.pi / (G * RHO0))

OUTDIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  3D KHRONON JEANS INSTABILITY TEST")
    print("  First 3D Anatropic Simulation")
    print("=" * 70)
    print()

    print(f"  Grid:            {N3D}^3 = {N3D**3:,} cells")
    print(f"  Box size:        L = {L}")
    print(f"  Sound speed:     c_s = {CS}")
    print(f"  Jeans length:    lambda_J = {LAMBDA_J:.4f}")
    print(f"  Jeans time:      t_ff = {T_FF:.4f}")
    print(f"  End time:        t_end = {T_END:.4f} ({T_END/T_FF:.1f} t_ff)")
    print(f"  Perturbation:    delta_rho/rho0 = {A_PERT:.0e} (random)")
    print(f"  CFL:             {CFL}")
    print(f"  Modes per box:   L / lambda_J = {L/LAMBDA_J:.1f}")
    print()

    # -----------------------------------------------------------------
    # Set up simulation
    # -----------------------------------------------------------------
    print("  Setting up simulation...")
    t0 = walltime.time()

    eos = IsothermalEOS(CS)
    sim = Simulation3D()
    sim.setup(Nx=N3D, Ny=N3D, Nz=N3D,
              Lx=L, Ly=L, Lz=L,
              rho0=RHO0, eos=eos,
              use_gravity=True, G=G)
    # Coherent mode-1 perturbation (well-resolved, strongly unstable)
    if USE_MODE_PERT:
        sim.add_perturbation_mode(amplitude=A_PERT, mode_x=1, mode_y=1, mode_z=1)
    # Additional random perturbation to seed all modes
    sim.add_random_perturbation(amplitude=A_PERT * 0.1, seed=SEED)

    print(f"  Setup complete in {walltime.time()-t0:.1f}s")
    print(f"  Initial rho range: [{np.min(sim.state.rho):.6f}, "
          f"{np.max(sim.state.rho):.6f}]")
    print()

    # -----------------------------------------------------------------
    # Collect midplane slices at intervals
    # -----------------------------------------------------------------
    # Save snapshots at key times
    snap_times = [0.0, T_FF, 2.0 * T_FF, 3.0 * T_FF, T_END]
    snap_rho_mid = []  # z-midplane density slices
    snap_labels = []

    # Save initial slice
    snap_rho_mid.append(sim.get_midplane_slice('z'))
    snap_labels.append(f't = 0 $t_{{ff}}$')

    # -----------------------------------------------------------------
    # Run in segments to capture intermediate snapshots
    # -----------------------------------------------------------------
    print("  Running simulation...")
    print("-" * 70)

    for i_seg in range(1, len(snap_times)):
        t_target = snap_times[i_seg]
        print(f"\n  >> Segment {i_seg}: running to t = {t_target:.4f} "
              f"({t_target/T_FF:.1f} t_ff)")

        sim.run(t_end=t_target, cfl=CFL, max_steps=1000000,
                print_every=200, max_walltime_s=600)

        # Save midplane slice
        rho_mid = sim.get_midplane_slice('z')
        snap_rho_mid.append(rho_mid)
        snap_labels.append(f't = {t_target/T_FF:.0f} $t_{{ff}}$')

        print(f"  Snapshot {i_seg}: rho_mid range = "
              f"[{np.min(rho_mid):.4e}, {np.max(rho_mid):.4e}]")

    print()
    print("-" * 70)

    # -----------------------------------------------------------------
    # Save final state
    # -----------------------------------------------------------------
    npz_path = os.path.join(OUTDIR, 'khronon_jeans_3d_final.npz')
    sim.save_snapshot(npz_path)
    print(f"  Saved final state: {npz_path}")

    # -----------------------------------------------------------------
    # Plot 1: Midplane density slices at different times
    # -----------------------------------------------------------------
    print("\n  Generating midplane slice figure...")

    n_snaps = len(snap_rho_mid)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps + 1, 4.5))
    if n_snaps == 1:
        axes = [axes]

    for i, (ax, rho_mid, label) in enumerate(
            zip(axes, snap_rho_mid, snap_labels)):
        vmin = max(np.percentile(rho_mid, 1), 1e-3 * RHO0)
        vmax = np.percentile(rho_mid, 99.5)
        if vmax <= vmin:
            vmax = vmin * 1.1

        im = ax.imshow(rho_mid, origin='lower', extent=[0, L, 0, L],
                       cmap='inferno', aspect='equal',
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('x', fontsize=11)
        if i == 0:
            ax.set_ylabel('y', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.85, label=r'$\rho$')

    fig.suptitle(
        f'3D Khronon Jeans Instability -- z-Midplane Slices\n'
        f'{N3D}$^3$ grid, $c_s$ = {CS}, $\\lambda_J$ = {LAMBDA_J:.3f}, '
        f'L/$\\lambda_J$ = {L/LAMBDA_J:.1f}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    path_slices = os.path.join(OUTDIR, 'khronon_jeans_3d_slices.png')
    plt.savefig(path_slices, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_slices}")

    # -----------------------------------------------------------------
    # Plot 2: 3D power spectrum
    # -----------------------------------------------------------------
    print("  Computing 3D power spectrum...")
    k_bins, P_k = sim.get_power_spectrum_3d()
    print(f"  P(k): {len(k_bins)} bins, "
          f"k range [{k_bins[0]:.1f}, {k_bins[-1]:.1f}]")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k_bins, P_k, 'b-', linewidth=2, label='P(k) [3D]')

    # Mark Jeans wavenumber
    k_J = 2.0 * np.pi / LAMBDA_J
    ax.axvline(k_J, color='red', linestyle='--', alpha=0.7,
               label=f'$k_J$ = {k_J:.1f} ($\\lambda_J$ = {LAMBDA_J:.3f})')

    # Reference slopes
    if len(k_bins) > 5:
        k_ref = np.logspace(np.log10(k_bins[1]), np.log10(k_bins[-2]), 50)
        P_ref = P_k[len(P_k)//4]
        k_ref_val = k_bins[len(k_bins)//4]
        ax.loglog(k_ref, P_ref * (k_ref / k_ref_val) ** (-2),
                  'k--', alpha=0.3, linewidth=1, label=r'$k^{-2}$ ref')

    ax.set_xlabel(r'Wavenumber $k$', fontsize=13)
    ax.set_ylabel(r'$P(k)$ [spherical average]', fontsize=13)
    ax.set_title(
        f'3D Power Spectrum at t = {sim.t/T_FF:.1f} $t_{{ff}}$\n'
        f'{N3D}$^3$, $c_s$ = {CS}, G = {G}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, which='both')

    path_pk = os.path.join(OUTDIR, 'khronon_jeans_3d_power_spectrum.png')
    plt.savefig(path_pk, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_pk}")

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("  3D JEANS INSTABILITY -- SUMMARY")
    print("=" * 70)
    rho_final = sim.state.rho
    rho_mean = np.mean(rho_final)
    delta = (rho_final - rho_mean) / rho_mean
    print(f"  Final time:          {sim.t:.4e} ({sim.t/T_FF:.2f} t_ff)")
    print(f"  Total steps:         {sim.step}")
    print(f"  Mean density:        {rho_mean:.6f}")
    print(f"  Density range:       [{np.min(rho_final):.4e}, "
          f"{np.max(rho_final):.4e}]")
    print(f"  Density contrast:    "
          f"{np.max(rho_final)/max(np.min(rho_final), 1e-30):.2e}")
    print(f"  delta_rho/rho std:   {np.std(delta):.4e}")
    print(f"  delta_rho/rho max:   {np.max(np.abs(delta)):.4e}")

    # Growth factor
    initial_pert = A_PERT
    final_pert = np.std(delta)
    if final_pert > initial_pert:
        growth = final_pert / initial_pert
        print(f"  Perturbation growth: {growth:.2f}x "
              f"(from {initial_pert:.0e} to {final_pert:.4e})")
    else:
        print(f"  No significant growth detected.")

    print()
    print("  Output files:")
    print(f"    {path_slices}")
    print(f"    {path_pk}")
    print(f"    {npz_path}")
    print()
    print("  Done.")


if __name__ == '__main__':
    main()
