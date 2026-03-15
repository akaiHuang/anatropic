# Methodology: GAMER-2 Validation to 3D Simulation Pipeline

## Overview

The Anatropic simulation code solves the compressible Euler equations with
self-gravity in 1D, 2D, and 3D.  The physical goal is to simulate Jeans
instability in the limit where the effective sound speed c_s approaches zero,
as predicted by the tau framework for the Khronon field at galactic scales.
This document describes the validation chain, key results, and the 3D export
pipeline.

---

## 1. Solver Architecture

| Component | Method | File |
|-----------|--------|------|
| Riemann solver | HLLE (Harten-Lax-van Leer-Einfeldt) | `anatropic/euler.py` |
| Gravity (1D) | FFT Poisson, periodic BC, Jeans swindle | `anatropic/gravity.py` |
| Gravity (2D) | 2D FFT Poisson with discrete Laplacian | `examples/morphology_comparison_2d.py` |
| Gravity (3D) | 3D FFT Poisson with discrete Laplacian | `anatropic/gravity3d.py` |
| Time integration | Strang operator splitting (gravity-hydro-gravity) | `anatropic/simulation3d.py` |
| EOS | Ideal gas, Isothermal, TauEOS (scale-dependent) | `anatropic/eos.py` |
| 2D hydro | Dimensional splitting (x-sweep + y-sweep) | `examples/morphology_comparison_2d.py` |
| 3D hydro | Dimensional splitting (XYZ/ZYX alternating) | `anatropic/euler3d.py` |

The HLLE solver was chosen for its robustness in the dust limit (c_s -> 0),
where the wave speeds degenerate and many other Riemann solvers fail.  The
HLLE intermediate state formula remains well-defined as S_L and S_R collapse
to the flow velocities.

---

## 2. GAMER-2 Test Problem Validation

### 2.1 Sod Shock Tube (1D)

The Sod shock tube is the standard first test for any Euler solver and is the
primary validation problem used by the GAMER-2 AMR code (Schive et al. 2018).

**Setup:**
- Domain: [0, 1], N = 400 cells, periodic boundary conditions
- Left state (x < 0.5):  rho = 1.0, P = 1.0, v = 0
- Right state (x > 0.5): rho = 0.125, P = 0.1, v = 0
- Ideal gas EOS with gamma = 1.4
- CFL = 0.5, run to t = 0.2

**Result:**
The HLLE solver correctly captures all five wave features:
1. Left-propagating rarefaction fan
2. Contact discontinuity
3. Right-propagating shock

The numerical solution matches the exact Riemann solution.  Discontinuities
are smeared over ~4-6 cells, which is expected for a first-order Godunov
scheme without reconstruction.  This is identical to the behavior of GAMER-2's
first-order mode.

**Script:** `examples/gamer_to_3d_pipeline.py` (Panel 1)

### 2.2 Jeans Instability (1D)

The 1D Jeans instability test validates the gravity solver coupling.

**Setup:**
- N = 1024 cells, L = 10, periodic BC with Jeans swindle (k=0 mode zeroed)
- Background: rho_0 = 1.0, G = 1.0
- Multi-mode perturbation: modes n = 1, 2, 3, 4, 5, 8, 16 with amplitude 10^{-3}
- Three sound speed cases: c_s = 0.01, 10^{-4}, 10^{-6}

**Key result:**
For c_s -> 0, all modes grow at the same rate omega -> sqrt(4 pi G rho_0),
confirming the theoretical prediction that the Khronon field with vanishing
effective sound speed produces simultaneous fragmentation at all scales.

**Script:** `examples/khronon_jeans_fragmentation.py`

---

## 3. 2D Jeans Fragmentation

### 3.1 Setup

The 2D simulation uses dimensional splitting (x-sweep then y-sweep with the
1D HLLE solver) and a 2D FFT Poisson gravity solver.

- Grid: 128 x 128
- Box: L = 10, periodic BC
- c_s = 10^{-4} (tau framework prediction)
- Initial perturbation: 1% random Gaussian noise (seed = 42)
- Run to t = 6 t_ff

### 3.2 Results: Filamentary Structure

The simulation develops a characteristic filamentary morphology:
- **Nodes**: high-density peaks at filament intersections
- **Filaments**: elongated structures connecting nodes
- **Voids**: underdense regions between filaments

This cosmic-web-like structure arises because all wavelengths are
simultaneously Jeans-unstable when c_s -> 0.

### 3.3 Power Spectrum: P(k) ~ k^{-2.2}

The azimuthally-averaged 2D power spectrum shows a **broad continuum** with
spectral slope alpha ~ -2.2.  This is the key diagnostic signature:

| Model | P(k) shape | Morphology |
|-------|-----------|------------|
| psi-DM (wave/fuzzy) | Peaked at k ~ 2 pi / lambda_dB, oscillatory | Granular interference |
| Khronon (tau framework) | Broad continuum, P(k) ~ k^{-2.2} | Filamentary, self-similar |
| CDM (subhalos) | Poisson-like, discrete peaks | Isolated clumps |

These three morphologies produce distinct, observationally distinguishable
power spectrum shapes that can in principle be differentiated through
gravitational lensing measurements.

**Script:** `examples/morphology_comparison_2d.py`

---

## 4. Extension to 3D

### 4.1 3D Euler Solver

The 3D solver (`anatropic/euler3d.py`, `anatropic/simulation3d.py`) extends
the 1D HLLE solver via dimensional splitting:

1. **Half-step gravity kick** (Strang splitting)
2. **X-sweep**: apply 1D HLLE to each x-pencil; passively advect vy, vz
3. **Y-sweep**: apply 1D HLLE to each y-pencil; passively advect vx, vz
4. **Z-sweep**: apply 1D HLLE to each z-pencil; passively advect vx, vy
5. **Half-step gravity kick** with updated density

The sweep order alternates between XYZ and ZYX on each timestep to reduce
directional bias.

### 4.2 3D Simulation Parameters

- Grid: 64^3 = 262,144 cells
- Box: L = 1.0, periodic BC
- c_s = 0.001 (deeply in the dust limit)
- G = 1.0, rho_0 = 1.0
- Initial perturbation: 10% amplitude, P(k) ~ k^{-2.2} red spectrum (random phases, seed = 42)
- CFL = 0.05 (conservative for gravity accuracy)
- Run to t = 3.5 t_ff (well into nonlinear regime)

The red-spectrum initial condition (P(k) ~ k^{-2.2}) seeds more power at
large scales, producing the physically motivated sheet -> filament -> node
collapse sequence (Zel'dovich pancake physics).

### 4.3 3D Results

At t = 3.5 t_ff:
- Density contrast: ~646x (rho_max / rho_min)
- Filamentary structure with nodes at intersections
- Mean density preserved (mass conservation verified)

The 3D structure shows the same qualitative features as the 2D case but with
additional complexity: sheets, filaments, and nodes form a fully
three-dimensional cosmic-web analog.

**Script:** `examples/khronon_jeans_3d.py`

---

## 5. Density and tau Field Export

### 5.1 The tau Field

The temporal asymmetry field tau(x,y,z) is computed from the gravitational
potential:

    tau = 1 - exp(-Sigma/2)

where Sigma = 2 |Phi| / Phi_max is the normalized gravitational entropy
production.  This connects to the tau framework (Paper 1):

| tau value | Physical meaning |
|-----------|-----------------|
| tau -> 0 | Nearly reversible; weak gravitational potential (voids) |
| tau -> 1 | Strong time arrow; deep potential well (nodes/filaments) |

The tau field provides a **landscape of temporal asymmetry**: regions where
matter has collapsed (high density, deep potential wells) have a strong arrow
of time, while underdense voids remain nearly time-symmetric.

### 5.2 WebGL Export Pipeline

The export pipeline (`anatropic/export_webgl.py`) converts simulation
snapshots to binary format for Three.js Data3DTexture visualization:

1. **Density normalization**: log10(rho) mapped to [0, 255] Uint8 with global
   min/max across all snapshots for consistent color scaling.
2. **tau computation**: solve 3D Poisson equation for Phi, then compute
   tau = 1 - exp(-|Phi|/Phi_max).
3. **Binary output**: C-order flat arrays (Nz * Ny * Nx bytes each).
4. **Manifest**: JSON file listing all snapshots with timestamps, file paths,
   and density/tau statistics.

Output files in `web/data/`:
```
manifest.json           -- metadata + snapshot list
snap_000_density.bin    -- 64^3 = 262 KB per snapshot
snap_000_tau.bin        -- 64^3 = 262 KB per snapshot
...
snap_010_density.bin    -- 11 snapshots total (0 to 3.5 t_ff)
snap_010_tau.bin
```

Total data size: ~5.5 MB for 11 snapshots at 64^3 resolution.

**Script:** `examples/run_and_export_3d.py`

---

## 6. Pipeline Visualization

The four-panel pipeline overview (`examples/pipeline_overview.png`) documents
the complete chain:

1. **Panel 1 (Sod shock tube)**: Validates HLLE solver against exact Riemann
   solution.  Same test problem as GAMER-2.
2. **Panel 2 (2D morphology)**: Three-way comparison of psi-DM, Khronon, and
   CDM density fields with inset P(k) plot.
3. **Panel 3 (3D density)**: z-midplane slice of the 64^3 simulation at
   t = 3.5 t_ff, showing filamentary structure with 646x density contrast.
4. **Panel 4 (tau field)**: Temporal asymmetry field on the same midplane,
   showing where the arrow of time is strongest.

**Script:** `examples/gamer_to_3d_pipeline.py`

---

## 7. Key Findings

### 7.1 All Modes Simultaneously Jeans-Unstable

When c_s -> 0, the Jeans length lambda_J = c_s sqrt(pi / (G rho_0)) -> 0.
Every wavelength in the simulation box satisfies lambda > lambda_J and is
therefore gravitationally unstable.  Moreover, all modes grow at the same
rate:

    omega(k) -> sqrt(4 pi G rho_0)    for all k

This is fundamentally different from standard CDM (where only large-scale
modes grow) and from psi-DM (where the de Broglie wavelength sets a cutoff).

### 7.2 P(k) ~ k^{-2.2} Broad Continuum

The resulting density field has a broad power spectrum with no preferred
scale, consistent with self-similar fragmentation.  The spectral slope
alpha ~ -2.2 is a testable prediction that can be compared against
gravitational lensing observations (e.g., Fagin et al. 2024 SLACS
measurements give beta = 5.22 +/- 0.41, compared to the Khronon prediction
beta = 6.2 -- a 2.4 sigma tension but far closer than CDM which is excluded
at 6.8 sigma).

### 7.3 Filamentary Morphology

The c_s -> 0 Jeans fragmentation naturally produces filamentary structure
(nodes, filaments, voids) reminiscent of the cosmic web, but at galactic
scales.  This is consistent with observed features like stellar streams and
tidal tails that trace the gravitational potential.

---

## References

- GAMER-2: Schive et al. (2018), MNRAS 481, 4815
- Khronon field / tau framework: Paper 1 (Petz Recovery Unification)
- Fagin comparison: `examples/fagin_comparison.py`
- SPARC validation: `examples/test_rar_deep_mond.py`, `examples/test_btfr.py`
