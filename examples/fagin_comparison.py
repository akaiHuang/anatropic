#!/usr/bin/env python3
"""
Fagin et al. 2024 (arXiv:2403.13881) Comparison
=================================================

Formal comparison of our Khronon P(k) ~ k^{-2.2} prediction with the
SLACS strong lensing substructure measurement by Fagin et al. (2024).

Key data:
  - Fagin 2024 measured the POTENTIAL power spectrum slope:
      beta_psi = 5.22 +/- 0.41  (P_psi ~ k^{-beta_psi})
  - Conversion between potential and convergence:
      P_kappa(k) = k^4 * P_psi(k)
      => beta_kappa = beta_psi - 4
      => beta_kappa = 1.22 +/- 0.41 (Fagin in convergence space)

  - Our Khronon prediction:
      P_kappa ~ k^{-2.2}  =>  beta_kappa = 2.2
      P_psi   ~ k^{-6.2}  =>  beta_psi   = 6.2

  - CDM prediction (Diaz Rivero+ 2018):
      Discrete subhalos => P_kappa ~ k^{-4} at high k  =>  beta_kappa = 4
      P_psi ~ k^{-8}  =>  beta_psi = 8

Statistical tension:
  |beta_predicted - beta_measured| / sigma

Output:
  - fagin_comparison.png : Publication-quality two-panel figure
  - Console: statistical tension analysis

Author: Sheng-Kai Huang, 2026
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# =============================================================================
# Measurement and model parameters
# =============================================================================

# Fagin et al. 2024 measurement (potential power spectrum slope)
BETA_PSI_FAGIN = 5.22
SIGMA_FAGIN = 0.41

# Convert to convergence: beta_kappa = beta_psi - 4
BETA_KAPPA_FAGIN = BETA_PSI_FAGIN - 4.0  # = 1.22
SIGMA_KAPPA_FAGIN = SIGMA_FAGIN           # same sigma (shift preserves uncertainty)

# Model predictions
MODELS = {
    'Khronon': {
        'beta_kappa': 2.2,
        'beta_psi': 6.2,
        'color': '#00E5FF',       # cyan
        'color_light': '#00E5FF',
        'linestyle': '-',
        'linewidth': 2.5,
        'label': r'Khronon ($\beta_\kappa = 2.2$)',
        'description': r'Continuous filaments, $P_\kappa \propto k^{-2.2}$',
    },
    'CDM': {
        'beta_kappa': 4.0,
        'beta_psi': 8.0,
        'color': '#FF6E40',       # orange-red
        'color_light': '#FF6E40',
        'linestyle': '--',
        'linewidth': 2.0,
        'label': r'CDM subhalos ($\beta_\kappa = 4.0$)',
        'description': r'Discrete subhalos, $P_\kappa \propto k^{-4}$',
    },
    'psiDM': {
        'beta_kappa': 0.0,   # oscillatory -- effectively flat or bump
        'beta_psi': 4.0,
        'color': '#B388FF',       # purple
        'color_light': '#B388FF',
        'linestyle': ':',
        'linewidth': 2.0,
        'label': r'$\psi$DM interference ($\beta_\kappa \approx 0$)',
        'description': r'Oscillatory, characteristic bump at $k \sim m/\hbar$',
    },
}

# k range of Fagin et al. sensitivity (kpc^-1)
K_MIN = 0.05
K_MAX = 20.0

# =============================================================================
# Style: dark background
# =============================================================================

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#444466',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#aaaacc',
    'ytick.color': '#aaaacc',
    'grid.color': '#333355',
    'grid.alpha': 0.4,
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444466',
    'legend.labelcolor': '#e0e0e0',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
})


def power_law(k, beta, A=1.0):
    """Power-law power spectrum P(k) = A * k^{-beta}."""
    return A * k ** (-beta)


def psiDM_convergence_spectrum(k, m22=1.0):
    """
    Approximate psiDM convergence power spectrum.
    Characteristic de Broglie scale produces an oscillatory feature.
    k_dB ~ 2*pi*m*v_halo/hbar ~ few kpc^-1 for m22=1.
    Model: P_kappa ~ k^0 * [1 + sin(k/k_dB)^2 * exp(-k/k_cut)]
    This is schematic -- psiDM produces interference fringes, not a clean power law.
    """
    k_dB = 2.0  # kpc^-1 for m22 = 1, v ~ 200 km/s
    k_cut = 8.0
    # Broad feature around k_dB with oscillatory modulation
    envelope = np.exp(-0.5 * ((np.log10(k) - np.log10(k_dB)) / 0.5)**2)
    oscillation = 1.0 + 0.6 * np.sin(2 * np.pi * k / k_dB)**2
    # Normalize to roughly match other spectra at k ~ 1
    A_psi = 1e2
    return A_psi * oscillation * (1.0 + 5.0 * envelope) * k**(-0.5)


# =============================================================================
# Statistical tension analysis
# =============================================================================

print("=" * 70)
print("  FAGIN et al. 2024 vs KHRONON PREDICTION")
print("  Formal Statistical Comparison")
print("=" * 70)
print()

print("MEASUREMENT (Fagin et al. 2024, arXiv:2403.13881):")
print(f"  Potential PS slope:    beta_psi   = {BETA_PSI_FAGIN:.2f} +/- {SIGMA_FAGIN:.2f}")
print(f"  Convergence PS slope:  beta_kappa = {BETA_KAPPA_FAGIN:.2f} +/- {SIGMA_KAPPA_FAGIN:.2f}")
print(f"  (using P_kappa = k^4 * P_psi => beta_kappa = beta_psi - 4)")
print()

print("MODEL PREDICTIONS:")
print(f"  {'Model':<25s} {'beta_kappa':<12s} {'beta_psi':<12s} {'Tension (sigma)'}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*16}")

tensions = {}
for name, m in MODELS.items():
    # Tension in convergence space
    tension_kappa = abs(m['beta_kappa'] - BETA_KAPPA_FAGIN) / SIGMA_KAPPA_FAGIN
    # Tension in potential space
    tension_psi = abs(m['beta_psi'] - BETA_PSI_FAGIN) / SIGMA_FAGIN
    tensions[name] = {
        'kappa': tension_kappa,
        'psi': tension_psi,
    }
    print(f"  {name:<25s} {m['beta_kappa']:<12.1f} {m['beta_psi']:<12.1f} "
          f"{tension_psi:.2f}sigma (potential) / {tension_kappa:.2f}sigma (convergence)")

print()
print("INTERPRETATION:")
for name in MODELS:
    t_psi = tensions[name]['psi']
    t_kappa = tensions[name]['kappa']
    if t_psi < 1.0:
        compat = "CONSISTENT (< 1sigma)"
    elif t_psi < 2.0:
        compat = "mild tension (1-2sigma)"
    elif t_psi < 3.0:
        compat = "moderate tension (2-3sigma)"
    else:
        compat = "EXCLUDED (> 3sigma)"
    print(f"  {name:<25s} {compat}")

print()
print("NOTE: The conversion P_kappa = k^4 * P_psi assumes the Poisson equation")
print("      nabla^2 psi = 2*kappa in Fourier space => psi_hat = -2*kappa_hat/k^2.")
print("      Thus P_psi(k) = 4/k^4 * P_kappa(k), so beta_psi = beta_kappa + 4.")
print()


# =============================================================================
# Figure: Two-panel comparison
# =============================================================================

fig, (ax_psi, ax_kappa) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    'Khronon Prediction vs Fagin et al. 2024 SLACS Lensing Measurement',
    fontsize=15, fontweight='bold', color='white', y=0.97
)

k = np.logspace(np.log10(K_MIN), np.log10(K_MAX), 500)

# --- Normalization ---
# Normalize all spectra to match at k = 1 kpc^-1 in potential space
# (arbitrary normalization -- only slopes matter for this comparison)
k_norm = 1.0
A_norm = 1e3  # arbitrary amplitude at k = 1 kpc^-1


# ------ LEFT PANEL: Potential power spectrum P_psi(k) ------

ax = ax_psi

# Fagin measurement: central + 1sigma + 2sigma bands
P_fagin_central = A_norm * k**(-BETA_PSI_FAGIN)
P_fagin_1sig_up = A_norm * k**(-(BETA_PSI_FAGIN - SIGMA_FAGIN))
P_fagin_1sig_dn = A_norm * k**(-(BETA_PSI_FAGIN + SIGMA_FAGIN))
P_fagin_2sig_up = A_norm * k**(-(BETA_PSI_FAGIN - 2*SIGMA_FAGIN))
P_fagin_2sig_dn = A_norm * k**(-(BETA_PSI_FAGIN + 2*SIGMA_FAGIN))

# 2sigma band
ax.fill_between(k, P_fagin_2sig_dn, P_fagin_2sig_up,
                alpha=0.12, color='#FFD700', label=r'Fagin 2024 $2\sigma$')
# 1sigma band
ax.fill_between(k, P_fagin_1sig_dn, P_fagin_1sig_up,
                alpha=0.25, color='#FFD700', label=r'Fagin 2024 $1\sigma$')
# Central value
ax.loglog(k, P_fagin_central, color='#FFD700', linewidth=2.0, linestyle='-',
          label=r'Fagin 2024: $\beta_\psi = 5.22 \pm 0.41$', zorder=5)

# Model predictions (potential space)
for name, m in MODELS.items():
    if name == 'psiDM':
        # psiDM in potential space: divide convergence spectrum by k^4
        P_psi_model = psiDM_convergence_spectrum(k) / k**4
        # Renormalize to match at k=1
        P_psi_model *= A_norm / P_psi_model[np.argmin(np.abs(k - k_norm))]
        ax.loglog(k, P_psi_model, color=m['color'], linestyle=m['linestyle'],
                  linewidth=m['linewidth'], label=m['label'], alpha=0.9)
    else:
        P_model = A_norm * k**(-m['beta_psi'])
        ax.loglog(k, P_model, color=m['color'], linestyle=m['linestyle'],
                  linewidth=m['linewidth'], label=m['label'], alpha=0.9)

ax.set_xlabel(r'$k$ (kpc$^{-1}$)')
ax.set_ylabel(r'$P_\psi(k)$ (arbitrary normalization)')
ax.set_title(r'Potential Power Spectrum $P_\psi(k) \propto k^{-\beta_\psi}$',
             fontsize=12, pad=10)
ax.set_xlim(K_MIN, K_MAX)
ax.set_ylim(1e-8, 1e12)
ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
ax.grid(True, which='both', alpha=0.15)

# Annotation: "MEASURED" region
ax.annotate('MEASURED\n(23 SLACS lenses)',
            xy=(1.0, A_norm), xytext=(0.15, 1e8),
            fontsize=9, color='#FFD700', alpha=0.8,
            arrowprops=dict(arrowstyle='->', color='#FFD700', alpha=0.5),
            ha='center')


# ------ RIGHT PANEL: Convergence power spectrum P_kappa(k) ------

ax = ax_kappa

# Fagin measurement converted to convergence
P_fagin_conv_central = A_norm * k**(-BETA_KAPPA_FAGIN)
P_fagin_conv_1sig_up = A_norm * k**(-(BETA_KAPPA_FAGIN - SIGMA_KAPPA_FAGIN))
P_fagin_conv_1sig_dn = A_norm * k**(-(BETA_KAPPA_FAGIN + SIGMA_KAPPA_FAGIN))
P_fagin_conv_2sig_up = A_norm * k**(-(BETA_KAPPA_FAGIN - 2*SIGMA_KAPPA_FAGIN))
P_fagin_conv_2sig_dn = A_norm * k**(-(BETA_KAPPA_FAGIN + 2*SIGMA_KAPPA_FAGIN))

# 2sigma band
ax.fill_between(k, P_fagin_conv_2sig_dn, P_fagin_conv_2sig_up,
                alpha=0.12, color='#FFD700', label=r'Fagin 2024 $2\sigma$')
# 1sigma band
ax.fill_between(k, P_fagin_conv_1sig_dn, P_fagin_conv_1sig_up,
                alpha=0.25, color='#FFD700', label=r'Fagin 2024 $1\sigma$')
# Central value
ax.loglog(k, P_fagin_conv_central, color='#FFD700', linewidth=2.0, linestyle='-',
          label=r'Fagin 2024: $\beta_\kappa = 1.22 \pm 0.41$', zorder=5)

# Model predictions (convergence space)
for name, m in MODELS.items():
    if name == 'psiDM':
        P_kappa_model = psiDM_convergence_spectrum(k)
        # Renormalize
        P_kappa_model *= A_norm / P_kappa_model[np.argmin(np.abs(k - k_norm))]
        ax.loglog(k, P_kappa_model, color=m['color'], linestyle=m['linestyle'],
                  linewidth=m['linewidth'], label=m['label'], alpha=0.9)
    else:
        P_model = A_norm * k**(-m['beta_kappa'])
        ax.loglog(k, P_model, color=m['color'], linestyle=m['linestyle'],
                  linewidth=m['linewidth'], label=m['label'], alpha=0.9)

ax.set_xlabel(r'$k$ (kpc$^{-1}$)')
ax.set_ylabel(r'$P_\kappa(k)$ (arbitrary normalization)')
ax.set_title(r'Convergence Power Spectrum $P_\kappa(k) = k^4 P_\psi(k)$',
             fontsize=12, pad=10)
ax.set_xlim(K_MIN, K_MAX)
ax.set_ylim(1e-4, 1e8)
ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
ax.grid(True, which='both', alpha=0.15)

# Annotation: conversion reminder
ax.annotate(r'$P_\kappa = k^4 \, P_\psi$' + '\n' + r'$\beta_\kappa = \beta_\psi - 4$',
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=9, color='#aaaacc', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                      edgecolor='#444466', alpha=0.8))


# ------ BOTTOM TEXT BOX: Statistical tension summary ------

tension_text_lines = []
tension_text_lines.append(r"$\mathbf{Statistical\ tension\ (potential\ space):}$")
for name, m in MODELS.items():
    t = tensions[name]['psi']
    symbol = {True: r'\checkmark', False: r'\times'}[t < 2.0]
    tension_text_lines.append(
        f"    {m['label']}: "
        + r"$|\Delta\beta_\psi|/\sigma = "
        + f"{abs(m['beta_psi'] - BETA_PSI_FAGIN):.2f}/{SIGMA_FAGIN:.2f} = "
        + f"{t:.1f}\\sigma$"
    )

tension_str = '\n'.join(tension_text_lines)

fig.text(0.5, 0.01, tension_str, ha='center', va='bottom',
         fontsize=10, color='#e0e0e0',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f0f23',
                   edgecolor='#444466', alpha=0.9),
         family='monospace', linespacing=1.5)

plt.tight_layout(rect=[0, 0.12, 1, 0.93])

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fagin_comparison.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved: {outpath}")


# =============================================================================
# Additional figure: Beta comparison (1D summary)
# =============================================================================

fig2, (ax_beta_psi, ax_beta_kappa) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    r'Power Spectrum Slope Comparison: $\beta$ values',
    fontsize=14, fontweight='bold', color='white', y=0.97
)

for ax_b, space, beta_meas, sigma_meas, beta_key in [
    (ax_beta_psi, 'potential', BETA_PSI_FAGIN, SIGMA_FAGIN, 'beta_psi'),
    (ax_beta_kappa, 'convergence', BETA_KAPPA_FAGIN, SIGMA_KAPPA_FAGIN, 'beta_kappa'),
]:
    # Measurement band
    ax_b.axhspan(beta_meas - 2*sigma_meas, beta_meas + 2*sigma_meas,
                 alpha=0.12, color='#FFD700', label=r'Fagin 2024 $2\sigma$')
    ax_b.axhspan(beta_meas - sigma_meas, beta_meas + sigma_meas,
                 alpha=0.3, color='#FFD700', label=r'Fagin 2024 $1\sigma$')
    ax_b.axhline(beta_meas, color='#FFD700', linewidth=2, linestyle='-',
                 label=f'Fagin 2024 central', zorder=3)

    # Model predictions as points with labels
    x_positions = {'Khronon': 1, 'CDM': 2, 'psiDM': 3}
    x_labels = []

    for name, m in MODELS.items():
        xp = x_positions[name]
        beta_val = m[beta_key]

        if name == 'psiDM' and space == 'convergence':
            # psiDM is oscillatory, mark as approximate
            ax_b.plot(xp, beta_val, marker='o', markersize=14,
                      color=m['color'], markeredgecolor='white',
                      markeredgewidth=1.5, zorder=5)
            ax_b.annotate(f'~{beta_val:.1f}\n(oscillatory)',
                          xy=(xp, beta_val), xytext=(xp + 0.3, beta_val + 0.5),
                          fontsize=9, color=m['color'], ha='left',
                          arrowprops=dict(arrowstyle='->', color=m['color'], alpha=0.5))
        else:
            ax_b.plot(xp, beta_val, marker='o', markersize=14,
                      color=m['color'], markeredgecolor='white',
                      markeredgewidth=1.5, zorder=5)
            # Tension annotation
            t = abs(beta_val - beta_meas) / sigma_meas
            ax_b.annotate(f'{beta_val:.1f}\n({t:.1f}$\\sigma$)',
                          xy=(xp, beta_val), xytext=(xp + 0.3, beta_val + 0.3),
                          fontsize=9, color=m['color'], ha='left',
                          arrowprops=dict(arrowstyle='->', color=m['color'], alpha=0.5))
        x_labels.append(name)

    ax_b.set_xticks([1, 2, 3])
    ax_b.set_xticklabels(x_labels, fontsize=11)
    ax_b.set_xlim(0.3, 3.7)

    if space == 'potential':
        ax_b.set_ylabel(r'$\beta_\psi$  (slope of $P_\psi \propto k^{-\beta_\psi}$)',
                        fontsize=11)
        ax_b.set_title(r'Potential space: $P_\psi(k)$', fontsize=12, pad=8)
        ax_b.set_ylim(2, 10)
    else:
        ax_b.set_ylabel(r'$\beta_\kappa$  (slope of $P_\kappa \propto k^{-\beta_\kappa}$)',
                        fontsize=11)
        ax_b.set_title(r'Convergence space: $P_\kappa(k) = k^4 P_\psi(k)$',
                        fontsize=12, pad=8)
        ax_b.set_ylim(-2, 6)

    ax_b.grid(True, axis='y', alpha=0.2)
    ax_b.legend(loc='upper left', fontsize=8, framealpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.93])

outpath2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fagin_comparison_beta.png')
plt.savefig(outpath2, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {outpath2}")


# =============================================================================
# Final summary
# =============================================================================

print()
print("=" * 70)
print("  CONCLUSION")
print("=" * 70)
print()
print(f"  Fagin et al. 2024 measured beta_psi = {BETA_PSI_FAGIN} +/- {SIGMA_FAGIN}")
print(f"  => beta_kappa = {BETA_KAPPA_FAGIN} +/- {SIGMA_KAPPA_FAGIN}")
print()
print(f"  Khronon prediction:  beta_psi = 6.2  (tension = {tensions['Khronon']['psi']:.2f} sigma)")
print(f"  CDM prediction:      beta_psi = 8.0  (tension = {tensions['CDM']['psi']:.2f} sigma)")
print(f"  psiDM prediction:    beta_psi ~ 4.0  (tension = {tensions['psiDM']['psi']:.2f} sigma)")
print()

khronon_t = tensions['Khronon']['psi']
cdm_t = tensions['CDM']['psi']
psiDM_t = tensions['psiDM']['psi']

# Rank models by tension
ranked = sorted(tensions.items(), key=lambda x: x[1]['psi'])
print(f"  RANKING (lowest tension = best fit):")
for rank, (name, t) in enumerate(ranked, 1):
    marker = " <-- BEST" if rank == 1 else ""
    print(f"    {rank}. {name:<15s}  {t['psi']:.2f} sigma{marker}")
print()

if khronon_t < cdm_t:
    print(f"  Khronon is CLOSER to Fagin measurement than CDM")
    print(f"  by {cdm_t - khronon_t:.1f} sigma.")
else:
    print(f"  CDM is closer to Fagin measurement than Khronon")
    print(f"  by {khronon_t - cdm_t:.1f} sigma.")

print()
print("  CAVEAT: Fagin 2024 assumes a power-law model. The actual Khronon")
print("  spectrum may deviate from a pure power law, especially near the")
print("  Jeans scale. A proper comparison requires forward-modeling the")
print("  Khronon density field through the lensing pipeline.")
print()
print("=" * 70)
