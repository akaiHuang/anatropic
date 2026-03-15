# Anatropic

> **This is a computational physics project, not an AI tool.** The name "Anatropic" refers to anisotropic entropy production in cosmological structure formation. It has no relation to Anthropic (the AI company).

**One equation. 175 galaxies. 69 testable predictions.**

Anatropic is the computational companion to the τ framework (Huang 2026) — a modified gravity theory where gravitational effects emerge from quantum relative entropy between spacetime and matter states:

**Σ = D(ρ_spacetime ‖ ρ_matter)**

This single equation, under different boundary conditions, reproduces galaxy rotation curves, cluster dynamics, gravitational lensing, and large-scale structure — without dark matter particles.

## Live Demo

**[anatropic.pages.dev](https://anatropic.pages.dev/)**

| Page | What it shows |
|------|---------------|
| [Galaxy Predictions](https://anatropic.pages.dev/sim) | 3D point-cloud galaxies from SPARC data, τ field coloring, CDM halo comparison, real Legacy Survey photos, per-galaxy predictions from residuals |
| [Rotation Curves](https://anatropic.pages.dev/rotation) | Interactive viewer for 175 SPARC galaxies: Khronon (1 param) vs NFW (3 params), with residual analysis |
| [12 Tests](https://anatropic.pages.dev/generality) | BTFR, RAR, deep MOND, lensing, TDGs, escape velocity, bar speeds, disk stability, EFE, GCs, dwarfs, wide binaries |
| [3D Cluster](https://anatropic.pages.dev/cluster3d) | Three.js galaxy cluster: ΛCDM (NFW halos + ICM) vs Khronon (Σ field), three scales |
| [Universe](https://anatropic.pages.dev/universe) | Six scales from cosmic web to solar system: ΛCDM vs τ theory interpretations |
| [Predictions](https://anatropic.pages.dev/predictions) | 25 predictions: 7 verified, 8 testable now, 4 mathematical, 6 future frontiers |
| [Bullet Cluster](https://anatropic.pages.dev/bullet) | τ field relaxation: how Khronon explains the lensing-gas offset without dark matter |
| [Scorecard](https://anatropic.pages.dev/scorecard) | Modified Gravity 7 — ΛCDM 3 (4 ties) across 14 observational tests |

## The Core Equation

```
g_obs = g_bar / (1 − exp(−√(g_bar / a₀)))

where a₀ = cH₀/(2π) = 1.13 × 10⁻¹⁰ m/s²  (predicted, not fitted)
```

- **1 free parameter per galaxy** (mass-to-light ratio M/L)
- **a₀ derived from first principles** (de Sitter horizon temperature)
- **175 galaxies, 175 total parameters** (vs NFW: 525 parameters)

## Key Results

| Test | Result |
|------|--------|
| RAR universality | Scatter 0.144 dex (measurement limit: 0.119) |
| BTFR slope | Predicted 4.0, observed 3.68±0.12 |
| Fagin lensing | Khronon β=6.2 (2.4σ), CDM β=8.0 (6.8σ) |
| MW escape velocity | Predicted 558 km/s, observed 550±30 |
| Fast bars | R=1.0–1.3 (Khronon), 1.7 (CDM predicts slow) |
| Globular clusters | Σ-hierarchy: RMS 186% → 33% |
| Galaxy clusters | Three-layer solution: IGIMF × hydrostatic bias × thermal QRE = 1.50 |

## Residuals as Predictions

Unlike NFW (3 free parameters absorb residuals), Khronon's 1-parameter fit exposes every residual. Each one is a testable prediction:

- **Distance**: Khronon predicts "true" distances — testable with Cepheids/TRGB
- **M/L gradients**: Predicted radial stellar population variations — testable with SED fitting
- **Non-circular motions**: Predicted HI velocity field asymmetries — testable with radio interferometry
- **69 total predictions** for specific SPARC galaxies

## Project Structure

```
anatropic/
├── anatropic/          # Python simulation code
│   ├── euler.py        # Godunov hydro solver (HLLE Riemann)
│   ├── gravity.py      # FFT Poisson solver
│   └── eos.py          # τ-EOS: c_s²(k) = (μ₀/k)²
├── examples/           # Analysis scripts
│   ├── build_rotation_curves.py    # Generate SPARC rotation curve data
│   ├── test_btfr.py               # Baryonic Tully-Fisher test
│   ├── test_rar_deep_mond.py      # RAR deep MOND verification
│   ├── test_lensing_predictions.py # Fagin comparison
│   └── ...
├── web/                # Interactive website (anatropic.pages.dev)
│   ├── sim.html        # Galaxy predictions + 3D viewer
│   ├── rotation.html   # Rotation curve explorer
│   ├── generality.html # 12 independent tests
│   ├── data/
│   │   └── sparc_rotation_curves.json  # 175 galaxies
│   └── ...
└── papers/             # LaTeX manuscripts (separate repos)
```

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib
- (Website) Any modern browser with WebGL

## References

- Huang 2026: τ framework ([Paper 1: Zenodo DOI 10.5281/zenodo.18897853](https://doi.org/10.5281/zenodo.18897853))
- Buscemi et al. 2024 (arXiv:2412.12489): Independent verification
- Kumar 2025 (arXiv:2509.05246): Running G from QFT
- Gubitosi et al. 2024 (arXiv:2403.00531): SPARC validation
- Dorau-Much 2025 (PRL): Einstein equations from QRE
- Blanchet & Skordis 2024 (arXiv:2404.06584): Khronon = GDM
- Thomas, Kopp & Skordis 2016 (arXiv:1601.05097): GDM constraints
- Skordis & Złośnik 2021 (PRL 127, 161302): AeST theory
- Lelli, McGaugh & Schombert 2016: SPARC database

## License

BSD 3-Clause

## Author

Sheng-Kai Huang (akai@fawstudio.com)
