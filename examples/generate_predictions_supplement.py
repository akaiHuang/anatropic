#!/usr/bin/env python3
"""
Generate mathematical supplement for 69 galaxy predictions.

Reads SPARC rotation curve data, identifies galaxies with significant residuals
(chi2_khronon > threshold), and computes three categories of testable predictions:

1. Distance corrections:  D_predicted = D_SPARC * (V_obs / V_khronon)^2
                          (weighted average over middle 50% of data points)

2. M/L gradients:         local_ML(r) = global_ML * (V_obs(r) / V_khronon(r))^2
                          Reported for inner / middle / outer thirds

3. Non-circular motions:  V_nc(r) = V_obs(r) - V_khronon(r)
                          Report RMS and mean

The Khronon RAR formula:
    g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))
    where a0 = c * H0 / (2*pi) = 1.13e-10 m/s^2

Outputs:
    web/data/predictions_supplement.json
    examples/predictions_summary.png

Usage:
    python examples/generate_predictions_supplement.py
"""

import json
import os
import sys
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_PATH = os.path.join(ROOT_DIR, "web", "data", "sparc_rotation_curves.json")
OUT_JSON = os.path.join(ROOT_DIR, "web", "data", "predictions_supplement.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "predictions_summary.png")

# ── Constants ─────────────────────────────────────────────────────────────
c_SI = 2.998e8           # m/s
H0_SI = 73e3 / 3.086e22  # 73 km/s/Mpc -> s^-1
a0_Khronon = c_SI * H0_SI / (2 * np.pi)  # ~1.13e-10 m/s^2

# Galaxies we want to select: those with significant Khronon residuals
# (where testable predictions are most meaningful)
TARGET_N = 69
CHI2_THRESHOLD_INIT = 2.0  # start with this, adjust if needed

# Galaxy type -> testability methods
TESTABILITY_MAP = {
    "distance": ["Cepheids", "TRGB", "SNII", "TF relation"],
    "ml_gradient": ["SED fitting", "NIR photometry", "spectral decomposition"],
    "nc_motion": ["HI velocity field", "IFU spectroscopy", "beam smearing correction"],
}

# Common distance methods by type
DISTANCE_METHODS = {
    "S0": ["SBF", "TF relation"],
    "Sa": ["TF relation", "SNII"],
    "Sab": ["TF relation", "SNII"],
    "Sb": ["TF relation", "Cepheids", "SNII"],
    "Sbc": ["TF relation", "Cepheids"],
    "Sc": ["TF relation", "Cepheids", "TRGB"],
    "Scd": ["TF relation", "TRGB"],
    "Sd": ["TF relation", "TRGB"],
    "Sdm": ["TRGB", "TF relation"],
    "Sm": ["TRGB", "TF relation"],
    "Im": ["TRGB", "CMD fitting"],
    "BCD": ["TRGB", "CMD fitting"],
}


def compute_distance_correction(R, Vobs, V_khronon, e_Vobs):
    """
    Compute predicted distance correction.

    D_predicted = D_SPARC * <(V_obs / V_khronon)^2>_weighted

    Uses only the middle 50% of radial data points (25th-75th percentile)
    to avoid noisy inner points and sparse outer points.
    Weighted by 1/e_Vobs^2.

    Returns: correction_factor (multiply D_SPARC by this)
    """
    n = len(R)
    if n < 4:
        return 1.0

    # Middle 50% indices
    i_lo = n // 4
    i_hi = 3 * n // 4
    if i_hi <= i_lo:
        i_lo = 0
        i_hi = n

    V_obs_mid = np.array(Vobs[i_lo:i_hi])
    V_khr_mid = np.array(V_khronon[i_lo:i_hi])
    e_mid = np.array(e_Vobs[i_lo:i_hi])

    # Avoid division by zero
    mask = (V_khr_mid > 5.0) & (V_obs_mid > 5.0) & (e_mid > 0)
    if np.sum(mask) < 2:
        return 1.0

    V_obs_mid = V_obs_mid[mask]
    V_khr_mid = V_khr_mid[mask]
    e_mid = e_mid[mask]

    # Weights: inverse variance
    w = 1.0 / np.maximum(e_mid, 1.0) ** 2

    # Correction factor: weighted mean of (V_obs / V_khronon)^2
    ratio2 = (V_obs_mid / V_khr_mid) ** 2
    factor = np.average(ratio2, weights=w)

    # Clamp to reasonable range (0.5 to 2.0)
    factor = np.clip(factor, 0.5, 2.0)

    return float(factor)


def compute_ml_gradient(R, Vobs, V_khronon, ml_global):
    """
    Compute local M/L gradient.

    local_ML(r) = global_ML * (V_obs(r) / V_khronon(r))^2

    Returns: (ml_inner, ml_middle, ml_outer) averaged over inner/middle/outer thirds.
    """
    n = len(R)
    if n < 6:
        return ml_global, ml_global, ml_global

    V_obs = np.array(Vobs)
    V_khr = np.array(V_khronon)

    # Avoid division issues
    mask = (V_khr > 5.0) & (V_obs > 5.0)
    if np.sum(mask) < 6:
        return ml_global, ml_global, ml_global

    local_ml = ml_global * (V_obs / V_khr) ** 2

    # Split into thirds
    n_third = n // 3
    ml_inner = float(np.median(local_ml[:n_third]))
    ml_middle = float(np.median(local_ml[n_third:2*n_third]))
    ml_outer = float(np.median(local_ml[2*n_third:]))

    # Clamp to physical range
    ml_inner = np.clip(ml_inner, 0.05, 5.0)
    ml_middle = np.clip(ml_middle, 0.05, 5.0)
    ml_outer = np.clip(ml_outer, 0.05, 5.0)

    return ml_inner, ml_middle, ml_outer


def compute_noncircular(Vobs, V_khronon):
    """
    Compute non-circular motion residuals.

    V_nc(r) = V_obs(r) - V_khronon(r)

    Returns: (rms_kms, mean_kms)
    """
    V_nc = np.array(Vobs) - np.array(V_khronon)
    rms = float(np.sqrt(np.mean(V_nc ** 2)))
    mean = float(np.mean(V_nc))
    return rms, mean


def select_galaxies(galaxies, target_n=TARGET_N):
    """
    Select galaxies with significant residuals for predictions.

    Strategy: rank by chi2_khronon (descending), take top target_n,
    but ensure we cover a range of galaxy types.
    """
    # Filter: need enough data points and valid chi2
    valid = [g for g in galaxies if g['n_pts'] >= 5 and g['chi2_khronon'] > 0]

    # Sort by chi2_khronon descending (biggest residuals = most predictive)
    valid.sort(key=lambda g: -g['chi2_khronon'])

    # Take top target_n
    if len(valid) >= target_n:
        selected = valid[:target_n]
    else:
        selected = valid

    return selected


def determine_testability(galaxy):
    """Determine what methods can test each prediction for this galaxy."""
    gtype = galaxy.get('type', '')
    D = galaxy.get('D_Mpc', 0)
    methods = []

    # Distance methods depend on galaxy type and distance
    dist_methods = DISTANCE_METHODS.get(gtype, ["TF relation"])

    # Cepheids only feasible within ~25 Mpc (JWST extends this)
    if D < 25:
        if "Cepheids" in dist_methods:
            methods.append("Cepheids")
    if D < 15:
        methods.append("TRGB")
    if D < 50:
        if "SNII" in dist_methods:
            methods.append("SNII")

    # TF relation always applicable
    methods.append("TF relation")

    # SED fitting for M/L
    methods.append("SED fitting")

    # HI for non-circular motions
    methods.append("HI velocity field")

    return list(set(methods))


def make_predictions(data):
    """Generate all predictions for selected galaxies."""
    galaxies = data['galaxies']
    selected = select_galaxies(galaxies)

    print(f"  Selected {len(selected)} galaxies for predictions")
    print(f"  chi2_khronon range: {selected[-1]['chi2_khronon']:.2f} - {selected[0]['chi2_khronon']:.2f}")
    print()

    predictions = []

    for g in selected:
        R = np.array(g['R'])
        Vobs = np.array(g['Vobs'])
        e_Vobs = np.array(g['e_Vobs'])
        V_khronon = np.array(g['V_khronon'])
        V_nfw = np.array(g['V_nfw'])
        ml_global = g['ml_khronon']
        D_sparc = g['D_Mpc']

        # 1. Distance correction
        d_factor = compute_distance_correction(R, Vobs, V_khronon, e_Vobs)
        D_predicted = D_sparc * d_factor
        d_correction_pct = (d_factor - 1.0) * 100.0

        # 2. M/L gradient
        ml_inner, ml_middle, ml_outer = compute_ml_gradient(
            R, Vobs, V_khronon, ml_global)

        # 3. Non-circular motions
        nc_rms, nc_mean = compute_noncircular(Vobs, V_khronon)

        # Chi2 comparison
        e_safe = np.maximum(e_Vobs, 1.0)
        chi2_k = float(np.mean(((Vobs - V_khronon) / e_safe) ** 2))
        chi2_n = float(np.mean(((Vobs - V_nfw) / e_safe) ** 2))

        # Testability
        testable = determine_testability(g)

        pred = {
            "name": g['name'],
            "type": g.get('type', ''),
            "D_sparc": round(D_sparc, 2),
            "D_predicted": round(D_predicted, 2),
            "d_correction_pct": round(d_correction_pct, 1),
            "ml_inner": round(ml_inner, 3),
            "ml_middle": round(ml_middle, 3),
            "ml_outer": round(ml_outer, 3),
            "ml_global": round(ml_global, 3),
            "nc_rms_kms": round(nc_rms, 1),
            "nc_mean_kms": round(nc_mean, 1),
            "chi2_khronon": round(g['chi2_khronon'], 2),
            "chi2_nfw": round(g['chi2_nfw'], 2),
            "delta_BIC": round(g.get('delta_BIC', 0), 1),
            "n_pts": g['n_pts'],
            "Vflat": round(g.get('Vflat', 0), 1),
            "testable_with": testable,
        }
        predictions.append(pred)

    # Sort by |d_correction_pct| descending for presentation
    predictions.sort(key=lambda p: -abs(p['d_correction_pct']))

    return predictions


def build_output(predictions):
    """Build the full JSON output structure."""
    # Compute summary statistics
    d_corr = [p['d_correction_pct'] for p in predictions]
    nc_rms_vals = [p['nc_rms_kms'] for p in predictions]
    chi2_k_vals = [p['chi2_khronon'] for p in predictions]

    # Count by type
    type_counts = {}
    for p in predictions:
        t = p['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    output = {
        "metadata": {
            "n_predictions": len(predictions),
            "method": "Khronon RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))",
            "a0_m_s2": a0_Khronon,
            "a0_formula": "c * H0 / (2*pi)",
            "H0_km_s_Mpc": 73,
            "source_data": "SPARC (Lelli, McGaugh & Schombert 2016, AJ 152, 157)",
            "selection": "Top galaxies ranked by chi2_khronon (significant residuals)",
            "equations": {
                "distance": "D_predicted = D_SPARC * <(V_obs / V_khronon)^2>_weighted, middle 50% of data",
                "ml_gradient": "local_ML(r) = global_ML * (V_obs(r) / V_khronon(r))^2, inner/middle/outer thirds",
                "noncircular": "V_nc(r) = V_obs(r) - V_khronon(r), report RMS and mean",
                "khronon_rar": "g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0))), a0 = cH0/(2pi)",
            },
            "summary_statistics": {
                "d_correction_pct_median": round(float(np.median(d_corr)), 1),
                "d_correction_pct_mean": round(float(np.mean(d_corr)), 1),
                "d_correction_pct_std": round(float(np.std(d_corr)), 1),
                "d_correction_pct_range": [
                    round(float(np.min(d_corr)), 1),
                    round(float(np.max(d_corr)), 1),
                ],
                "nc_rms_kms_median": round(float(np.median(nc_rms_vals)), 1),
                "nc_rms_kms_mean": round(float(np.mean(nc_rms_vals)), 1),
                "chi2_khronon_median": round(float(np.median(chi2_k_vals)), 2),
                "chi2_khronon_mean": round(float(np.mean(chi2_k_vals)), 2),
            },
            "type_distribution": type_counts,
        },
        "predictions": predictions,
    }

    return output


def make_plot(predictions):
    """Generate publication-quality 4-panel summary plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # ── Style ──────────────────────────────────────────────────────────
    BG = '#0a0a0a'
    FG = '#cccccc'
    ACCENT = '#e8860c'
    ACCENT2 = '#f0a030'
    GRID = '#222222'
    ACCENT_LIGHT = '#f5c882'

    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': FG,
        'text.color': FG,
        'xtick.color': FG,
        'ytick.color': FG,
        'grid.color': GRID,
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Khronon Framework: 69-Galaxy Prediction Supplement',
                 fontsize=16, fontweight='bold', color=ACCENT, y=0.97)

    # ── Panel 1: Predicted vs SPARC distances ──────────────────────────
    ax = axes[0, 0]
    D_sparc = [p['D_sparc'] for p in predictions]
    D_pred = [p['D_predicted'] for p in predictions]
    d_corr = [p['d_correction_pct'] for p in predictions]

    # Color by correction magnitude
    scatter = ax.scatter(D_sparc, D_pred, c=d_corr, cmap='RdYlGn_r',
                         s=30, alpha=0.85, edgecolors='none',
                         vmin=-30, vmax=30, zorder=3)
    cb = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Distance correction [%]', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # 1:1 line
    dmax = max(max(D_sparc), max(D_pred)) * 1.1
    ax.plot([0, dmax], [0, dmax], '--', color=ACCENT, alpha=0.7, lw=1.5,
            label='1:1 line')

    # +/-20% envelope
    ax.fill_between([0, dmax], [0, 0.8*dmax], [0, 1.2*dmax],
                    color=ACCENT, alpha=0.07, label=r'$\pm 20\%$')

    ax.set_xlabel('D$_{\\rm SPARC}$ [Mpc]')
    ax.set_ylabel('D$_{\\rm predicted}$ [Mpc]')
    ax.set_title('Panel A: Predicted vs SPARC Distances', fontsize=12,
                 color=ACCENT_LIGHT)
    ax.set_xlim(0, dmax)
    ax.set_ylim(0, dmax)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.3,
              edgecolor='none', facecolor=BG)
    ax.grid(True, alpha=0.3)

    # Annotation: median correction
    med_corr = np.median(d_corr)
    ax.text(0.97, 0.05,
            f'Median correction: {med_corr:+.1f}%\n'
            f'N = {len(predictions)} galaxies',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=FG, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                      edgecolor='#333333', alpha=0.9))

    # ── Panel 2: M/L gradient (inner vs outer) ────────────────────────
    ax = axes[0, 1]
    ml_inner = [p['ml_inner'] for p in predictions]
    ml_outer = [p['ml_outer'] for p in predictions]
    ml_global = [p['ml_global'] for p in predictions]

    ax.scatter(ml_inner, ml_outer, c=ml_global, cmap='viridis',
               s=30, alpha=0.85, edgecolors='none', zorder=3)
    cb2 = fig.colorbar(
        ax.collections[0], ax=ax, shrink=0.8, pad=0.02)
    cb2.set_label(r'Global $\Upsilon_\star$ [3.6$\mu$m]', fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    # 1:1 line (no gradient)
    ml_lim = max(max(ml_inner), max(ml_outer)) * 1.1
    ml_lim = min(ml_lim, 3.0)
    ax.plot([0, ml_lim], [0, ml_lim], '--', color=ACCENT, alpha=0.7, lw=1.5,
            label='No gradient')
    ax.set_xlabel(r'$\Upsilon_\star^{\rm inner}$ [M$_\odot$/L$_\odot$]')
    ax.set_ylabel(r'$\Upsilon_\star^{\rm outer}$ [M$_\odot$/L$_\odot$]')
    ax.set_title('Panel B: M/L Gradient (Inner vs Outer)', fontsize=12,
                 color=ACCENT_LIGHT)
    ax.set_xlim(0, ml_lim)
    ax.set_ylim(0, ml_lim)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.3,
              edgecolor='none', facecolor=BG)
    ax.grid(True, alpha=0.3)

    # Count galaxies above/below diagonal
    n_above = sum(1 for i, o in zip(ml_inner, ml_outer) if o > i)
    n_below = len(ml_inner) - n_above
    ax.text(0.97, 0.05,
            f'Outer > Inner: {n_above}\n'
            f'Inner > Outer: {n_below}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=FG, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                      edgecolor='#333333', alpha=0.9))

    # ── Panel 3: Non-circular motion RMS by galaxy type ───────────────
    ax = axes[1, 0]

    # Group by type
    type_nc = {}
    for p in predictions:
        t = p['type'] if p['type'] else 'Unknown'
        if t not in type_nc:
            type_nc[t] = []
        type_nc[t].append(p['nc_rms_kms'])

    # Sort types by Hubble sequence
    hubble_order = ['S0', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd',
                    'Sdm', 'Sm', 'Im', 'BCD']
    types_sorted = [t for t in hubble_order if t in type_nc]
    # Add any types not in the standard list
    for t in type_nc:
        if t not in types_sorted:
            types_sorted.append(t)

    positions = np.arange(len(types_sorted))
    medians = [np.median(type_nc[t]) for t in types_sorted]
    means = [np.mean(type_nc[t]) for t in types_sorted]
    counts = [len(type_nc[t]) for t in types_sorted]

    # Box plot data
    bp_data = [type_nc[t] for t in types_sorted]

    bp = ax.boxplot(bp_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor=ACCENT, alpha=0.5),
                    medianprops=dict(color=ACCENT, linewidth=2),
                    boxprops=dict(facecolor='#1a1a1a', edgecolor='#444444'),
                    whiskerprops=dict(color='#555555'),
                    capprops=dict(color='#555555'))

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{t}\n(n={c})' for t, c in
                        zip(types_sorted, counts)],
                       fontsize=8, rotation=0)
    ax.set_ylabel(r'$V_{\rm nc}$ RMS [km/s]')
    ax.set_title('Panel C: Non-Circular Motion RMS by Galaxy Type',
                 fontsize=12, color=ACCENT_LIGHT)
    ax.grid(True, axis='y', alpha=0.3)

    # Overall stats
    all_nc = [p['nc_rms_kms'] for p in predictions]
    ax.axhline(np.median(all_nc), color=ACCENT, ls=':', alpha=0.5,
               label=f'Median = {np.median(all_nc):.1f} km/s')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3,
              edgecolor='none', facecolor=BG)

    # ── Panel 4: Histogram of distance correction percentages ─────────
    ax = axes[1, 1]
    d_corr_arr = np.array(d_corr)

    bins = np.linspace(-50, 50, 31)
    n_vals, bin_edges, patches = ax.hist(d_corr_arr, bins=bins,
                                          color=ACCENT, alpha=0.75,
                                          edgecolor='#333333', linewidth=0.5)

    # Color negative corrections differently
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#4488cc')
            patch.set_alpha(0.65)

    # Vertical lines for median and zero
    ax.axvline(0, color=FG, ls='-', alpha=0.3, lw=1)
    ax.axvline(np.median(d_corr_arr), color=ACCENT, ls='--', lw=2,
               label=f'Median = {np.median(d_corr_arr):+.1f}%')
    ax.axvline(np.mean(d_corr_arr), color=ACCENT2, ls=':', lw=1.5,
               label=f'Mean = {np.mean(d_corr_arr):+.1f}%')

    ax.set_xlabel('Distance correction [%]')
    ax.set_ylabel('Number of galaxies')
    ax.set_title('Panel D: Distribution of Predicted Distance Corrections',
                 fontsize=12, color=ACCENT_LIGHT)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3,
              edgecolor='none', facecolor=BG)
    ax.grid(True, axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Stats annotation
    n_pos = np.sum(d_corr_arr > 0)
    n_neg = np.sum(d_corr_arr < 0)
    n_zero = np.sum(d_corr_arr == 0)
    ax.text(0.03, 0.95,
            f'Farther: {n_pos} ({100*n_pos/len(d_corr_arr):.0f}%)\n'
            f'Closer: {n_neg} ({100*n_neg/len(d_corr_arr):.0f}%)\n'
            f'Unchanged: {n_zero}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color=FG, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                      edgecolor='#333333', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    # Footer
    fig.text(0.5, 0.005,
             r'Khronon RAR: $g_{\rm obs} = g_{\rm bar} / (1 - e^{-\sqrt{g_{\rm bar}/a_0}})$'
             r',  $a_0 = cH_0/(2\pi)$  |  '
             r'Data: SPARC (Lelli+ 2016)  |  anatropic.io',
             ha='center', va='bottom', fontsize=8, color='#666666')

    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f"  Saved plot: {OUT_PNG}")
    plt.close()


def main():
    print("=" * 70)
    print("  Khronon Predictions Supplement Generator")
    print("  69-Galaxy Mathematical Predictions")
    print("=" * 70)
    print(f"  a0 = cH0/(2pi) = {a0_Khronon:.4e} m/s^2")
    print()

    # Load data
    print(f"  Loading: {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    n_gal = len(data['galaxies'])
    print(f"  Loaded {n_gal} galaxies from SPARC")
    print()

    # Generate predictions
    print("  Computing predictions...")
    predictions = make_predictions(data)

    # Build output
    output = build_output(predictions)

    # Save JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    size_kb = os.path.getsize(OUT_JSON) / 1024
    print(f"  Saved JSON: {OUT_JSON} ({size_kb:.1f} KB)")

    # Summary
    meta = output['metadata']
    stats = meta['summary_statistics']
    print()
    print("  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)
    print(f"  Predictions generated:       {meta['n_predictions']}")
    print(f"  Distance correction median:  {stats['d_correction_pct_median']:+.1f}%")
    print(f"  Distance correction std:     {stats['d_correction_pct_std']:.1f}%")
    print(f"  Distance correction range:   [{stats['d_correction_pct_range'][0]:+.1f}%, "
          f"{stats['d_correction_pct_range'][1]:+.1f}%]")
    print(f"  Non-circular RMS median:     {stats['nc_rms_kms_median']:.1f} km/s")
    print(f"  chi2_khronon median:         {stats['chi2_khronon_median']:.2f}")
    print(f"  Type distribution:           {meta['type_distribution']}")
    print()

    # Top 10 predictions by correction magnitude
    print("  Top 10 distance corrections:")
    print("  " + "-" * 60)
    for p in predictions[:10]:
        print(f"    {p['name']:12s}  D_SPARC={p['D_sparc']:5.1f}  "
              f"D_pred={p['D_predicted']:5.1f}  "
              f"corr={p['d_correction_pct']:+5.1f}%  "
              f"chi2_K={p['chi2_khronon']:5.1f}")
    print()

    # Generate plot
    print("  Generating publication-quality plot...")
    make_plot(predictions)

    print()
    print("  Done. Outputs:")
    print(f"    JSON: {OUT_JSON}")
    print(f"    Plot: {OUT_PNG}")


if __name__ == '__main__':
    main()
