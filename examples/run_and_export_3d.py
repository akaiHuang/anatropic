#!/usr/bin/env python3
"""
Run 3D Khronon Jeans instability simulation and export for Three.js WebGL.

Outputs binary density + τ (temporal asymmetry) fields at multiple time
snapshots, plus a JSON manifest, to web/data/ for the interactive viewer.

Usage:
    python examples/run_and_export_3d.py
"""

import os
import sys

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from anatropic.eos import IsothermalEOS
from anatropic.simulation3d import Simulation3D
from anatropic.export_webgl import export_simulation_webgl

# ── Configuration ─────────────────────────────────────────────────────────
N = 64           # Grid resolution (N³)
L = 1.0          # Box size
RHO0 = 1.0       # Background density
CS = 0.001       # Sound speed → 0 (Khronon prediction: all modes Jeans-unstable)
G = 1.0          # Gravitational constant
AMP = 0.10       # 10% amplitude for red-spectrum perturbation
SEED = 42        # Random seed
CFL = 0.05       # Low CFL for accurate gravity integration
N_TFF = 3.5      # Run into nonlinear regime for filamentary structure
SNAP_INTERVAL_TFF = 0.35  # Save snapshot every 0.35 t_ff (~10 snapshots)
PK_SLOPE = -2.2  # P(k) ∝ k^slope — Khronon prediction (broad continuum)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "web", "data")

# ── Run simulation ────────────────────────────────────────────────────────
print("=" * 70)
print("Khronon Jeans 3D → WebGL Export")
print("=" * 70)
print(f"  Grid: {N}³ = {N**3:,} cells")
print(f"  Box: L = {L}")
print(f"  cs = {CS}, G = {G}, ρ₀ = {RHO0}")
print(f"  Perturbation: Gaussian noise, amp = {AMP}, seed = {SEED}")

eos = IsothermalEOS(cs=CS)
sim = Simulation3D()
sim.setup(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L,
          rho0=RHO0, eos=eos, use_gravity=True, G=G)
# Seed with P(k) ∝ k^{-2.2} power spectrum (Khronon prediction).
# Red spectrum = more large-scale power → sheets → filaments → nodes
# (Zel'dovich collapse physics, same as cosmic web formation).
rng = np.random.default_rng(SEED)
kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
ky = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
kz = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
K = np.sqrt(KX**2 + KY**2 + KZ**2)
K[0, 0, 0] = 1.0  # avoid division by zero

# Amplitude ∝ k^{slope/2} (so P(k) = |δ̂|² ∝ k^slope)
amplitude_k = K ** (PK_SLOPE / 2.0)
amplitude_k[0, 0, 0] = 0.0  # no DC offset

# Random phases
phases = rng.uniform(0, 2 * np.pi, size=(N, N, N))
delta_k = amplitude_k * np.exp(1j * phases)

# Transform to real space, normalize to desired amplitude
delta_real = np.real(np.fft.ifftn(delta_k))
delta_real *= AMP / np.std(delta_real)

# Apply to density (transpose to match (Nz, Ny, Nx) layout)
sim.state.rho *= (1.0 + delta_real.transpose(2, 1, 0))
sim.state.rho = np.maximum(sim.state.rho, 1e-30)
P = sim.eos.pressure(sim.state.rho, np.zeros_like(sim.state.rho))
sim.state.eint = sim.eos.internal_energy(sim.state.rho, P)
print(f"  Initial ρ range: [{sim.state.rho.min():.4f}, {sim.state.rho.max():.4f}]")
print(f"  P(k) slope: {PK_SLOPE} (Khronon prediction)")

t_ff = sim.compute_jeans_time()
lambda_J = sim.compute_jeans_length()
t_end = N_TFF * t_ff
snap_interval = SNAP_INTERVAL_TFF * t_ff

print(f"  t_ff = {t_ff:.4f}, λ_J = {lambda_J:.4f}")
print(f"  Running to t = {t_end:.4f} ({N_TFF} t_ff)")
print(f"  Snapshot interval: {snap_interval:.4f} ({SNAP_INTERVAL_TFF} t_ff)")
print()

sim.run(t_end=t_end, cfl=CFL, snapshot_interval=snap_interval,
        print_every=50)

# ── Export for WebGL ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Exporting for Three.js WebGL...")
print("=" * 70)

manifest = export_simulation_webgl(sim, OUTPUT_DIR, log_density=True)

print("\n" + "=" * 70)
print("Done! Files ready for web visualization.")
print(f"  Serve with: cd web && python -m http.server 8000")
print(f"  Then open: http://localhost:8000/sim.html")
print("=" * 70)
