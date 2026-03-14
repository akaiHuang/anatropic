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

from anatropic.eos import IsothermalEOS
from anatropic.simulation3d import Simulation3D
from anatropic.export_webgl import export_simulation_webgl

# ── Configuration ─────────────────────────────────────────────────────────
N = 64           # Grid resolution (N³)
L = 1.0          # Box size
RHO0 = 1.0       # Background density
CS = 0.05        # Sound speed (λ_J ~ 0.3 for these params)
G = 1.0          # Gravitational constant
AMP = 1e-2       # Initial perturbation amplitude (1% — matches khronon_jeans_3d.py)
SEED = 42        # Random seed
CFL = 0.05       # Low CFL for accurate gravity integration
N_TFF = 5.0      # Run for this many free-fall times
SNAP_INTERVAL_TFF = 0.5  # Save snapshot every 0.5 t_ff
USE_MODE_PERT = True  # Add coherent mode-1 for strong Jeans growth

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
if USE_MODE_PERT:
    sim.add_perturbation_mode(amplitude=AMP, mode_x=1, mode_y=1, mode_z=1)
sim.add_random_perturbation(amplitude=AMP * 0.1, seed=SEED)

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
