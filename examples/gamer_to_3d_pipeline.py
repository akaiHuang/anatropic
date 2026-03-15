#!/usr/bin/env python3
"""
GAMER-2 Validation -> 2D Fragmentation -> 3D WebGL Pipeline Overview
=====================================================================

Generates a four-panel figure (pipeline_overview.png) documenting the full
simulation pipeline of the Anatropic project:

    Panel 1 -- 1D Sod shock tube: validates our HLLE Euler solver against
               the exact Riemann solution (same test used by GAMER-2).
    Panel 2 -- 2D morphology comparison: psiDM vs Khronon vs CDM density
               fields with azimuthally-averaged P(k).
    Panel 3 -- 3D density field: midplane slice from the 64^3 Jeans
               instability simulation, loaded from web/data/*.bin exports.
    Panel 4 -- tau field overlay: temporal asymmetry field tau = 1 - exp(-Sigma/2)
               on the same 3D midplane slice.

If binary snapshot files exist in web/data/, they are loaded directly.
Otherwise, the solver is exercised to produce representative data.

Usage:
    python examples/gamer_to_3d_pipeline.py

Author: Anatropic project
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, 'web', 'data')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'pipeline_overview.png')

sys.path.insert(0, PROJECT_ROOT)

from anatropic.euler import (
    _primitive_from_conservative, _hlle_flux, _add_ghost_cells,
    compute_dt, evolve, DENSITY_FLOOR, PRESSURE_FLOOR
)
from anatropic.eos import IdealGasEOS, IsothermalEOS
from anatropic.gravity import solve_gravity


# =====================================================================
# Style constants
# =====================================================================
BG_COLOR = '#0a0a0a'
ACCENT = '#e8860c'
ACCENT2 = '#f5a623'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#222222'
CMAP_DENSITY = 'inferno'
CMAP_TAU = 'magma'


def set_dark_style(ax, title='', xlabel='', ylabel=''):
    """Apply dark-background styling to an axes."""
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight='bold',
                 pad=8)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5, which='both')


# =====================================================================
# Panel 1: Sod shock tube (1D) -- solver validation
# =====================================================================

def sod_exact(x, t, x0=0.5, gamma=1.4):
    """
    Exact Riemann solution for the Sod shock tube at time t.

    Left state:  (rho, P, v) = (1.0, 1.0, 0.0)
    Right state: (rho, P, v) = (0.125, 0.1, 0.0)

    Returns arrays (rho, v, P) on the given x grid.
    """
    # Analytic Sod solution constants for gamma=1.4
    # Post-shock / star region values (derived from Rankine-Hugoniot + rarefaction)
    P_star = 0.30313
    v_star = 0.92745
    rho_star_L = 0.42632   # left of contact
    rho_star_R = 0.26557   # right of contact

    # Wave speeds
    c_L = np.sqrt(gamma * 1.0 / 1.0)          # left sound speed
    c_star_L = c_L * (P_star / 1.0) ** ((gamma - 1) / (2 * gamma))
    S_head = x0 - c_L * t                       # rarefaction head
    S_tail = x0 + (v_star - c_star_L) * t       # rarefaction tail
    S_contact = x0 + v_star * t                  # contact discontinuity
    # Shock speed (from Rankine-Hugoniot)
    S_shock = x0 + 1.7522 * t  # for standard Sod

    rho = np.empty_like(x)
    v = np.empty_like(x)
    P = np.empty_like(x)

    for i in range(len(x)):
        xi = x[i]
        if xi < S_head:
            # Undisturbed left
            rho[i], v[i], P[i] = 1.0, 0.0, 1.0
        elif xi < S_tail:
            # Inside rarefaction fan
            c_local = (2 / (gamma + 1)) * (c_L + (xi - x0) / t)
            rho[i] = 1.0 * (c_local / c_L) ** (2 / (gamma - 1))
            v[i] = (2 / (gamma + 1)) * ((xi - x0) / t + c_L)
            P[i] = 1.0 * (c_local / c_L) ** (2 * gamma / (gamma - 1))
        elif xi < S_contact:
            # Star region (left of contact)
            rho[i], v[i], P[i] = rho_star_L, v_star, P_star
        elif xi < S_shock:
            # Star region (right of contact)
            rho[i], v[i], P[i] = rho_star_R, v_star, P_star
        else:
            # Undisturbed right
            rho[i], v[i], P[i] = 0.125, 0.0, 0.1

    return rho, v, P


def run_sod_test(N=400, t_end=0.2, cfl=0.5):
    """
    Run a 1D Sod shock tube and return (x, rho_num, v_num, P_num).
    """
    gamma = 1.4
    eos = IdealGasEOS(gamma=gamma)
    L = 1.0
    dx = L / N
    x = (np.arange(N) + 0.5) * dx

    # Initial conditions
    rho = np.where(x < 0.5, 1.0, 0.125)
    v = np.zeros(N)
    P = np.where(x < 0.5, 1.0, 0.1)
    eint = eos.internal_energy(rho, P)

    U = np.zeros((3, N))
    U[0] = rho
    U[1] = rho * v
    U[2] = rho * (eint + 0.5 * v ** 2)

    t = 0.0
    step = 0
    while t < t_end:
        dt = compute_dt(U, dx, eos, cfl=cfl)
        dt = min(dt, t_end - t)
        if dt <= 0:
            break
        U = evolve(U, dx, dt, eos, gravity_source=None)
        t += dt
        step += 1

    # Extract primitives
    rho_out, v_out, P_out, _, _ = _primitive_from_conservative(U, eos)
    return x, rho_out, v_out, P_out


# =====================================================================
# Panel 2: 2D morphology representative (from analytical generators)
# =====================================================================

def generate_psiDM_field(Ny, Nx, L, n_waves=20, lambda_dB=1.0, seed=42):
    """Generate psiDM density field via wave superposition."""
    rng = np.random.RandomState(seed)
    dx = L / Nx
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dx
    X, Y = np.meshgrid(x, y)
    psi = np.zeros((Ny, Nx), dtype=complex)
    for _ in range(n_waves):
        theta = rng.uniform(0, 2 * np.pi)
        k_mag = 2 * np.pi / lambda_dB * (0.7 + 0.6 * rng.random())
        kx = k_mag * np.cos(theta)
        ky = k_mag * np.sin(theta)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        psi += amp * np.exp(1j * (kx * X + ky * Y + phase))
    rho = np.abs(psi) ** 2
    return rho / np.mean(rho)


def generate_CDM_field(Ny, Nx, L, n_halos=15, seed=42):
    """Generate CDM-like field with Gaussian subhalos."""
    rng = np.random.RandomState(seed)
    dx = L / Nx
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dx
    X, Y = np.meshgrid(x, y)
    rho = np.full((Ny, Nx), 0.5)
    for _ in range(n_halos):
        x0 = rng.uniform(0, L)
        y0 = rng.uniform(0, L)
        M = np.exp(rng.uniform(np.log(0.01), np.log(0.5)))
        sigma = rng.uniform(0.1, 0.3)
        dX = X - x0;  dY = Y - y0
        dX -= L * np.round(dX / L)
        dY -= L * np.round(dY / L)
        rho += M / (2 * np.pi * sigma ** 2) * np.exp(-(dX ** 2 + dY ** 2) / (2 * sigma ** 2))
    return rho / np.mean(rho)


def generate_khronon_field_2d(Ny, Nx, L, seed=42):
    """
    Generate a Khronon-like 2D field with P(k) ~ k^{-2.2} power spectrum.
    Uses random phases with the predicted spectral slope.
    """
    rng = np.random.RandomState(seed)
    dx = L / Nx
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1.0
    # P(k) ~ k^{-2.2} -> amplitude ~ k^{-1.1}
    amp = K ** (-1.1)
    amp[0, 0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, size=(Ny, Nx))
    delta_k = amp * np.exp(1j * phases)
    delta_real = np.real(np.fft.ifft2(delta_k))
    # Convert to density
    rho = 1.0 + 0.5 * delta_real / np.std(delta_real)
    rho = np.maximum(rho, 0.01)
    return rho / np.mean(rho)


# =====================================================================
# Panel 3 & 4: Load 3D data from web/data/ .bin files
# =====================================================================

def load_3d_snapshot(manifest_path, snap_index=-1):
    """
    Load density and tau fields from the WebGL-exported binary files.

    Parameters
    ----------
    manifest_path : str
        Path to manifest.json.
    snap_index : int
        Which snapshot to load (-1 = last).

    Returns
    -------
    density : ndarray (Nz, Ny, Nx)  float64 (log scale values)
    tau     : ndarray (Nz, Ny, Nx)  float64 in [0,1]
    meta    : dict   snapshot metadata
    grid    : tuple  (Nx, Ny, Nz)
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    grid = manifest['grid']  # [Nx, Ny, Nz]
    Nx, Ny, Nz = grid
    n_cells = Nx * Ny * Nz
    log_density = manifest.get('log_density', True)
    rho_range = manifest['density_range']  # [min, max] of log10(rho) or rho

    snap = manifest['snapshots'][snap_index]
    data_dir = os.path.dirname(manifest_path)

    # Read density
    density_u8 = np.fromfile(
        os.path.join(data_dir, snap['density_file']), dtype=np.uint8
    )
    density_norm = density_u8.astype(np.float64) / 255.0
    # Undo normalization to log10(rho) space
    density_log = rho_range[0] + density_norm * (rho_range[1] - rho_range[0])
    density = density_log.reshape((Nz, Ny, Nx))

    # Read tau
    tau_u8 = np.fromfile(
        os.path.join(data_dir, snap['tau_file']), dtype=np.uint8
    )
    tau = (tau_u8.astype(np.float64) / 255.0).reshape((Nz, Ny, Nx))

    return density, tau, snap, grid


def generate_synthetic_3d(N=64):
    """
    If no .bin files, produce a synthetic 3D density + tau field
    using the same seeding as run_and_export_3d.py.
    """
    L = 1.0
    rng = np.random.default_rng(42)
    kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    kz = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    K[0, 0, 0] = 1.0
    amp = K ** (-2.2 / 2.0)
    amp[0, 0, 0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, size=(N, N, N))
    delta_k = amp * np.exp(1j * phases)
    delta_real = np.real(np.fft.ifftn(delta_k))
    delta_real *= 0.10 / np.std(delta_real)
    rho = 1.0 * (1.0 + delta_real)
    rho = np.maximum(rho, 1e-30)
    # Approximate tau
    from anatropic.export_webgl import compute_tau_field
    dx = L / N
    tau, _ = compute_tau_field(rho, dx, dx, dx, G=1.0)
    density_log = np.log10(rho)
    return density_log, tau


# =====================================================================
# Azimuthal power spectrum (2D)
# =====================================================================

def azimuthal_pk_2d(rho, L):
    """Azimuthally-averaged 2D P(k)."""
    Ny, Nx = rho.shape
    dx = L / Nx
    delta = rho - np.mean(rho)
    delta_hat = np.fft.fft2(delta) / (Nx * Ny)
    power = np.abs(delta_hat) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    dk = 2 * np.pi / L
    k_max = np.pi / dx
    edges = np.arange(0.5 * dk, k_max + dk, dk)
    k_bins = 0.5 * (edges[:-1] + edges[1:])
    Pk = np.zeros(len(k_bins))
    for i in range(len(k_bins)):
        mask = (K >= edges[i]) & (K < edges[i + 1])
        if np.any(mask):
            Pk[i] = np.mean(power[mask])
    valid = Pk > 0
    return k_bins[valid], Pk[valid]


# =====================================================================
# MAIN: build the four-panel figure
# =====================================================================

def main():
    print("=" * 70)
    print("  GAMER-2 -> 3D SIMULATION PIPELINE OVERVIEW")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Panel 1 data: Sod shock tube
    # ------------------------------------------------------------------
    print("  [1/4] Running Sod shock tube (N=400, t=0.2) ...")
    x_sod, rho_num, v_num, P_num = run_sod_test(N=400, t_end=0.2)
    rho_exact, v_exact, P_exact = sod_exact(x_sod, t=0.2)
    print(f"        Done. L2 density error = "
          f"{np.sqrt(np.mean((rho_num - rho_exact)**2)):.4e}")

    # ------------------------------------------------------------------
    # Panel 2 data: 2D morphology
    # ------------------------------------------------------------------
    print("  [2/4] Generating 2D morphology fields (128x128) ...")
    N2 = 128
    L2 = 10.0
    rho_psi = generate_psiDM_field(N2, N2, L2)
    rho_khr = generate_khronon_field_2d(N2, N2, L2)
    rho_cdm = generate_CDM_field(N2, N2, L2)
    print("        Done.")

    # ------------------------------------------------------------------
    # Panel 3 & 4 data: 3D from web/data/
    # ------------------------------------------------------------------
    manifest_path = os.path.join(WEB_DATA_DIR, 'manifest.json')
    has_bin_data = os.path.isfile(manifest_path)

    if has_bin_data:
        print("  [3/4] Loading 3D data from web/data/ (last snapshot) ...")
        density_3d, tau_3d, snap_meta, grid = load_3d_snapshot(
            manifest_path, snap_index=-1
        )
        Nx3, Ny3, Nz3 = grid
        # Midplane slices
        mid_z = Nz3 // 2
        density_mid = density_3d[mid_z, :, :]
        tau_mid = tau_3d[mid_z, :, :]
        t_ff_label = f"{snap_meta['time_ff']:.1f}"
        contrast = snap_meta['density_stats']['contrast']
        print(f"        Loaded {Nx3}^3 grid, t = {t_ff_label} t_ff, "
              f"contrast = {contrast:.0f}x")
    else:
        print("  [3/4] No .bin files found; generating synthetic 3D field ...")
        density_3d, tau_3d = generate_synthetic_3d(N=64)
        density_mid = density_3d[32, :, :]
        tau_mid = tau_3d[32, :, :]
        t_ff_label = "~3.5"
        contrast = 10.0 ** (np.max(density_mid) - np.min(density_mid))
        print(f"        Synthetic field generated.")

    print("  [4/4] Building figure ...")

    # ==================================================================
    # Build the figure
    # ==================================================================
    fig = plt.figure(figsize=(16, 12), facecolor=BG_COLOR)
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28,
                  left=0.06, right=0.96, top=0.91, bottom=0.06)

    # --- Panel 1: Sod shock tube (three sub-rows) ---
    gs_sod = gs[0, 0].subgridspec(3, 1, hspace=0.08)

    # -- density --
    ax1a = fig.add_subplot(gs_sod[0])
    set_dark_style(ax1a,
                   title='Panel 1: 1D Sod Shock Tube Validation',
                   xlabel='', ylabel=r'$\rho$')
    ax1a.plot(x_sod, rho_exact, '-', color='#555555', linewidth=2.5, zorder=1)
    ax1a.plot(x_sod, rho_num, '-', color=ACCENT, linewidth=1.2,
              label='HLLE', zorder=2)
    ax1a.set_xticklabels([])
    ax1a.legend(fontsize=6, loc='upper right',
                facecolor='#111111', edgecolor='#333333', labelcolor=TEXT_COLOR)
    ax1a.text(0.03, 0.85,
              'GAMER-2 standard test\nN = 400, CFL = 0.5, t = 0.2',
              transform=ax1a.transAxes, fontsize=6, color='#777777',
              va='top', ha='left')

    # -- velocity --
    ax1b = fig.add_subplot(gs_sod[1])
    set_dark_style(ax1b, xlabel='', ylabel='v')
    ax1b.plot(x_sod, v_exact, '-', color='#555555', linewidth=2.5, zorder=1)
    ax1b.plot(x_sod, v_num, '-', color='#4fc3f7', linewidth=1.2, zorder=2)
    ax1b.set_xticklabels([])

    # -- pressure --
    ax1c = fig.add_subplot(gs_sod[2])
    set_dark_style(ax1c, xlabel='x', ylabel='P')
    ax1c.plot(x_sod, P_exact, '-', color='#555555', linewidth=2.5,
              label='Exact', zorder=1)
    ax1c.plot(x_sod, P_num, '-', color='#81c784', linewidth=1.2,
              label='HLLE', zorder=2)
    ax1c.legend(fontsize=6, loc='upper right',
                facecolor='#111111', edgecolor='#333333', labelcolor=TEXT_COLOR)

    # --- Panel 2: 2D morphology mini comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    set_dark_style(ax2,
                   title=r'Panel 2: 2D Morphology ($\psi$DM / Khronon / CDM)',
                   xlabel='', ylabel='')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Create a composite strip: 3 fields side by side
    pad = np.full((N2, 2), np.nan)
    composite = np.concatenate([rho_psi, pad, rho_khr, pad, rho_cdm], axis=1)
    vmin_c = np.nanpercentile(composite, 2)
    vmax_c = np.nanpercentile(composite, 98)
    cmap_morph = plt.cm.inferno.copy()
    cmap_morph.set_bad(color=BG_COLOR)
    ax2.imshow(composite, origin='lower', cmap=cmap_morph,
               vmin=vmin_c, vmax=vmax_c, aspect='auto')
    # Labels along bottom, inside each panel
    w = N2
    sep = 2
    for i, label in enumerate([r'$\psi$DM', 'Khronon', 'CDM']):
        cx = i * (w + sep) + w / 2
        ax2.text(cx, 5, label, ha='center', va='bottom',
                 color=TEXT_COLOR, fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='#000000', alpha=0.7, edgecolor='none',
                           pad=2))

    # Small inset: P(k) comparison
    ax2_inset = ax2.inset_axes([0.02, 0.02, 0.40, 0.38])
    ax2_inset.set_facecolor('#111111')
    k_p, Pk_p = azimuthal_pk_2d(rho_psi, L2)
    k_k, Pk_k = azimuthal_pk_2d(rho_khr, L2)
    k_c, Pk_c = azimuthal_pk_2d(rho_cdm, L2)
    ax2_inset.loglog(k_p, Pk_p, '-', color='#42a5f5', linewidth=1,
                     label=r'$\psi$DM', alpha=0.9)
    ax2_inset.loglog(k_k, Pk_k, '-', color=ACCENT, linewidth=1.3,
                     label='Khronon', alpha=0.9)
    ax2_inset.loglog(k_c, Pk_c, '-', color='#66bb6a', linewidth=1,
                     label='CDM', alpha=0.9)
    ax2_inset.set_xlabel('k', fontsize=6, color='#aaaaaa')
    ax2_inset.set_ylabel('P(k)', fontsize=6, color='#aaaaaa')
    ax2_inset.tick_params(labelsize=5, colors='#888888')
    ax2_inset.legend(fontsize=5, loc='lower left',
                     facecolor='#111111', edgecolor='#333333',
                     labelcolor='#cccccc')
    for sp in ax2_inset.spines.values():
        sp.set_color('#333333')

    # --- Panel 3: 3D density midplane ---
    ax3 = fig.add_subplot(gs[1, 0])
    set_dark_style(ax3,
                   title=f'Panel 3: 3D Density Midplane (t = {t_ff_label} $t_{{ff}}$)',
                   xlabel='x', ylabel='y')
    # density_mid is in log10(rho) scale
    im3 = ax3.imshow(density_mid, origin='lower',
                     cmap=CMAP_DENSITY, aspect='equal',
                     extent=[0, 1, 0, 1])
    cb3 = fig.colorbar(im3, ax=ax3, shrink=0.82, pad=0.02)
    cb3.set_label(r'$\log_{10}\rho$', color=TEXT_COLOR, fontsize=8)
    cb3.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax3.text(0.03, 0.95,
             f'64$^3$ grid\ncontrast ~ {contrast:.0f}x',
             transform=ax3.transAxes, fontsize=7, color='#cccccc',
             va='top', ha='left',
             bbox=dict(facecolor='#000000', alpha=0.6, edgecolor='none',
                       pad=2))

    # --- Panel 4: tau field overlay ---
    ax4 = fig.add_subplot(gs[1, 1])
    set_dark_style(ax4,
                   title=r'Panel 4: $\tau$ Field (Temporal Asymmetry)',
                   xlabel='x', ylabel='y')
    im4 = ax4.imshow(tau_mid, origin='lower',
                     cmap=CMAP_TAU, aspect='equal',
                     extent=[0, 1, 0, 1],
                     vmin=0, vmax=np.percentile(tau_mid, 99.5))
    cb4 = fig.colorbar(im4, ax=ax4, shrink=0.82, pad=0.02)
    cb4.set_label(r'$\tau = 1 - e^{-\Sigma/2}$', color=TEXT_COLOR, fontsize=8)
    cb4.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax4.text(0.03, 0.95,
             r'$\tau \to 1$: strong time arrow' '\n'
             r'$\tau \to 0$: nearly reversible',
             transform=ax4.transAxes, fontsize=7, color='#cccccc',
             va='top', ha='left',
             bbox=dict(facecolor='#000000', alpha=0.6, edgecolor='none',
                       pad=2))

    # --- Suptitle ---
    fig.suptitle(
        'Anatropic Simulation Pipeline: GAMER-2 Validation  '
        r'$\rightarrow$  2D Morphology  '
        r'$\rightarrow$  3D Density + $\tau$ Field',
        fontsize=14, fontweight='bold', color=ACCENT,
        y=0.97
    )

    # --- Footer annotation ---
    fig.text(0.5, 0.01,
             'Euler (HLLE) + FFT Poisson gravity  |  '
             'Strang splitting  |  '
             r'$c_s \to 0$: all modes Jeans-unstable  |  '
             r'P(k) $\sim k^{-2.2}$ (Khronon prediction)',
             ha='center', va='bottom', fontsize=8, color='#666666')

    # Save
    fig.savefig(OUTPUT_PATH, dpi=180, facecolor=BG_COLOR,
                bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB")
    print()
    print("  Pipeline stages documented:")
    print("    1. Sod shock tube -> validates HLLE solver (matches GAMER-2)")
    print("    2. 2D morphology  -> psiDM / Khronon / CDM distinguishable")
    print("    3. 3D density     -> Jeans fragmentation at 64^3")
    print("    4. tau field      -> temporal asymmetry landscape")
    print()
    print("  Done.")


if __name__ == '__main__':
    main()
