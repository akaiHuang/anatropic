# Anatropic

**Metal-accelerated cosmological hydrodynamics for the τ framework**

Anatropic is a GPU-accelerated adaptive mesh refinement (AMR) code designed to simulate structure formation in the τ framework — a modified gravity theory where the Khronon field's DBI kinetic term produces scale-dependent sound speed c_s²(k) → 0 at sub-galactic scales, enabling Jeans fragmentation at sub-kpc scales.

## Physics

The τ framework (Huang 2026) connects:
- **Running G**: G(r) = G_N[1 + 2k_*r/π] (Kumar 2025, Gubitosi+ 2024)
- **Khronon = GDM equivalence** (Blanchet & Skordis 2024)
- **Scale-dependent sound speed**: c_s²(k) = (μ₀/k)² where μ₀ = H₀/c

When c_s → 0 at galactic scales, the Jeans length λ_J = √(πc_s²c²/(Gρ)) shrinks to sub-kpc, potentially producing density fluctuations observable in gravitational lensing.

## Results (Phase 1)

Our 1D and 2D simulations of the Khronon Jeans instability reveal:
- When c_s → 0, **all modes become simultaneously Jeans-unstable** (dust limit)
- Growth rates converge to the free-fall rate ω_J = √(4πGρ₀) independent of wavelength
- 2D simulations produce **filamentary structure** (nodes, filaments, voids) — a self-similar multi-scale fragmentation pattern
- Power spectrum: P(k) ~ k⁻²·², a broad continuum with no preferred scale

## Architecture

- **Phase 1** (current): Python prototype — 1D/2D Euler + Poisson solver
- **Phase 2** (planned): Apple Metal compute shaders for 3D GPU acceleration
- **Phase 3** (planned): Full AMR with τ-EOS

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `anatropic.euler` | Godunov hydro solver (HLLE Riemann) | Phase 1 |
| `anatropic.gravity` | FFT Poisson solver (self-gravity) | Phase 1 |
| `anatropic.eos` | τ-EOS: c_s²(k) = (μ₀/k)² | Phase 1 |
| `anatropic.metal` | Metal compute shader backend | Phase 2 |
| `anatropic.amr` | Adaptive mesh refinement | Phase 3 |

## Validation

Before running new physics, Anatropic is validated against:
1. Sod shock tube (exact solution)
2. Linear Jeans instability growth rate (analytical)
3. Acoustic wave propagation
4. GAMER test problem results (cross-code comparison)

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib
- (Phase 2) macOS 13+ with Apple Silicon for Metal acceleration

## References

- Kumar 2025 (arXiv:2509.05246): Running G from QFT
- Gubitosi et al. 2024 (arXiv:2403.00531): SPARC validation
- Blanchet & Skordis 2024 (arXiv:2404.06584): Khronon = GDM
- Thomas, Kopp & Skordis 2016 (arXiv:1601.05097): GDM constraints
- Skordis & Złośnik 2021 (PRL 127, 161302): AeST theory
- Schive et al. 2018 (arXiv:1712.07070): GAMER-2

## License

BSD 3-Clause

## Author

Sheng-Kai Huang (akai@fawstudio.com)
