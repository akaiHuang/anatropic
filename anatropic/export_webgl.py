"""
Export 3D simulation data to binary format for Three.js WebGL visualization.

Outputs Uint8 binary files (density + τ field) with a JSON manifest,
ready to be loaded by Three.js Data3DTexture.

Data layout matches NumPy (Nz, Ny, Nx) C-order = Three.js (width=Nx,
height=Ny, depth=Nz) with index iz*Ny*Nx + iy*Nx + ix.
"""

import json
import os
import numpy as np

from .gravity3d import solve_potential_3d


def compute_tau_field(rho, dx, dy, dz, G=1.0):
    """
    Compute the temporal asymmetry field τ(x,y,z) from the density field.

    τ = 1 - exp(-Σ/2) where Σ = |Φ|/Φ_scale is the normalized
    gravitational entropy production. Dense regions (deep potential
    wells) have τ → 1 (strong time arrow); voids have τ → 0 (nearly
    reversible).

    Parameters
    ----------
    rho : ndarray, shape (Nz, Ny, Nx)
        Density field.
    dx, dy, dz : float
        Grid spacings.
    G : float
        Gravitational constant.

    Returns
    -------
    tau : ndarray, shape (Nz, Ny, Nx)
        Temporal asymmetry field in [0, 1).
    phi : ndarray, shape (Nz, Ny, Nx)
        Gravitational potential (for diagnostics).
    """
    phi = solve_potential_3d(rho, dx, dy, dz, G)
    phi_abs = np.abs(phi)
    phi_max = np.max(phi_abs)

    if phi_max > 0:
        # Σ_grav ∝ |Φ|, scaled so max Σ ~ 2 (gives τ_max ~ 0.63)
        sigma = 2.0 * phi_abs / phi_max
    else:
        sigma = np.zeros_like(phi)

    tau = 1.0 - np.exp(-sigma / 2.0)
    return tau, phi


def _normalize_to_uint8(data, log_scale=False):
    """Normalize 3D array to Uint8 [0, 255]."""
    if log_scale:
        data = np.log10(np.maximum(data, 1e-30))

    mn, mx = float(data.min()), float(data.max())
    if mx > mn:
        norm = (data - mn) / (mx - mn)
    else:
        norm = np.zeros_like(data)

    return (norm * 255).clip(0, 255).astype(np.uint8), mn, mx


def export_simulation_webgl(sim, output_dir, log_density=True,
                            n_snapshots_max=12):
    """
    Export simulation snapshots as binary files + JSON manifest.

    Creates:
        output_dir/manifest.json
        output_dir/snap_NNN_density.bin  (Uint8, N³ bytes each)
        output_dir/snap_NNN_tau.bin      (Uint8, N³ bytes each)

    Parameters
    ----------
    sim : Simulation3D
        Completed simulation with snapshots.
    output_dir : str
        Output directory (created if needed).
    log_density : bool
        If True, use log10(ρ) for better dynamic range.
    n_snapshots_max : int
        Maximum snapshots to export (subsample if more).
    """
    os.makedirs(output_dir, exist_ok=True)

    snaps = sim.snapshots
    if not snaps:
        raise ValueError("No snapshots to export. Run simulation with "
                         "snapshot_interval set.")

    # Subsample if too many
    if len(snaps) > n_snapshots_max:
        indices = np.linspace(0, len(snaps) - 1, n_snapshots_max,
                              dtype=int)
        snaps = [snaps[i] for i in indices]

    # Find global density range for consistent normalization
    all_rho = np.concatenate([s['rho'].ravel() for s in snaps])
    if log_density:
        rho_for_range = np.log10(np.maximum(all_rho, 1e-30))
    else:
        rho_for_range = all_rho
    global_rho_min = float(rho_for_range.min())
    global_rho_max = float(rho_for_range.max())

    t_ff = sim.compute_jeans_time()

    manifest = {
        'grid': [sim.Nx, sim.Ny, sim.Nz],
        'box_size': [sim.Lx, sim.Ly, sim.Lz],
        'G': sim.G,
        't_ff': t_ff,
        'log_density': log_density,
        'density_range': [global_rho_min, global_rho_max],
        'snapshots': [],
    }

    for i, snap in enumerate(snaps):
        rho = snap['rho']
        t = snap['t']

        # Density → Uint8
        if log_density:
            rho_proc = np.log10(np.maximum(rho, 1e-30))
        else:
            rho_proc = rho

        # Use global range for consistent normalization
        rng = global_rho_max - global_rho_min
        if rng > 0:
            rho_norm = (rho_proc - global_rho_min) / rng
        else:
            rho_norm = np.zeros_like(rho_proc)
        rho_u8 = (rho_norm * 255).clip(0, 255).astype(np.uint8)

        # τ field
        tau, phi = compute_tau_field(rho, sim.dx, sim.dy, sim.dz, sim.G)
        tau_u8 = (tau * 255).clip(0, 255).astype(np.uint8)

        # Write binary files
        density_file = f'snap_{i:03d}_density.bin'
        tau_file = f'snap_{i:03d}_tau.bin'

        rho_u8.ravel().tofile(os.path.join(output_dir, density_file))
        tau_u8.ravel().tofile(os.path.join(output_dir, tau_file))

        manifest['snapshots'].append({
            'index': i,
            'time': float(t),
            'time_ff': float(t / t_ff) if t_ff > 0 else 0,
            'density_file': density_file,
            'tau_file': tau_file,
            'density_stats': {
                'min': float(rho.min()),
                'max': float(rho.max()),
                'mean': float(rho.mean()),
                'contrast': float(rho.max() / max(rho.min(), 1e-30)),
            },
            'tau_stats': {
                'min': float(tau.min()),
                'max': float(tau.max()),
                'mean': float(tau.mean()),
            },
        })

        print(f"  Exported snapshot {i}: t = {t:.4f} "
              f"({t/t_ff:.2f} t_ff), "
              f"ρ ∈ [{rho.min():.4f}, {rho.max():.4f}], "
              f"τ ∈ [{tau.min():.4f}, {tau.max():.4f}]")

    # Write manifest
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, s['density_file']))
        + os.path.getsize(os.path.join(output_dir, s['tau_file']))
        for s in manifest['snapshots']
    )
    print(f"\n  Exported {len(manifest['snapshots'])} snapshots to "
          f"{output_dir}")
    print(f"  Total data: {total_bytes / 1024:.0f} KB "
          f"({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Manifest: {manifest_path}")

    return manifest
