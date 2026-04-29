"""
regenerate_kfold_figures.py — Rebuild the K-fold paper figures from cached
results, without retraining or rerunning the per-fold model evaluations.

What this does
--------------
Reads the artifacts that train_kfold_full_coverage_profile_only.py and
compute_spectral_feature_importance.py have already written into a K-fold
results directory, and regenerates every figure from those artifacts.

This lets you iterate on figure styling (colors, fonts, layout, axis
labels, captions, panel sizes) in seconds, instead of rerunning the
21-fold training (hours) or the per-channel perturbation sweep (minutes).

Required artifacts in --results-dir
-----------------------------------
    per_profile_summary.csv          (one row per unique profile)
    per_profile_correlations.json    (Pearson r + Spearman rho per predictor)
    overall_summary.json             (run-level metadata, n_folds, etc.)
    spectral_feature_importance.csv  (per-wavelength mean / std / per-fold)

Figures regenerated (written to --figures-dir)
----------------------------------------------
    per_profile_rmse_distribution.png
    rmse_vs_predictors.png
    per_level_uncertainty.png
    per_profile_rmse_heatmap.png
    spectral_feature_importance.png
    spectral_feature_importance_absolute.png

Usage
-----
    python regenerate_kfold_figures.py \\
        --results-dir standalone_results_profile_only_kfold/run110_K21_20260427_1810
        
    python regenerate_kfold_figures.py \\
        --results-dir standalone_results_profile_only_kfold/run110_K21_20260427_1810 \\
        --h5-path //Users/anbu8374/Documents/VS_CODE/Python-Research/lasp-CU-paper-3/training_data/combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5

Editing tips
------------
Each figure is built by a single dedicated function (plot_*).  To change
one figure, edit only its function.  Common tweaks live near the top of
each function so you do not have to hunt for them.

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Global style — paper-quality serif fonts, mathtext for LaTeX-style labels,
# and a colorblind-safe palette (Okabe-Ito) used throughout the figures.
# -----------------------------------------------------------------------------
# Okabe-Ito palette (CB-safe; the de-facto standard for accessible plots).
CB = {
    'black':         '#000000',
    'orange':        '#E69F00',
    'sky_blue':      '#56B4E9',
    'bluish_green':  '#009E73',
    'yellow':        '#F0E442',
    'blue':          '#0072B2',
    'vermillion':    '#D55E00',
    'reddish_purple':'#CC79A7',
}
# Sequential colormap: viridis is perceptually uniform AND colorblind-safe.
CB_CMAP = 'viridis'


def setup_style():
    """
    Configure matplotlib for paper-quality output: serif font, mathtext
    rendered in Computer Modern (LaTeX-style) without requiring a TeX
    install.  To switch to a real LaTeX backend instead (slower, requires
    MacTeX/TeXLive), set rcParams['text.usetex'] = True.
    """
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
        'axes.labelsize':    12,
        'axes.titlesize':    13,
        'figure.titlesize':  14,
        'legend.fontsize':   10,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'axes.linewidth':    0.8,
    })


# -----------------------------------------------------------------------------
# Tiny stats helpers — recompute Pearson r and Spearman rho on the fly when
# we need correlations against a transformed predictor (e.g. switching from
# log10(wv) to absolute wv).  No scipy dependency.
# -----------------------------------------------------------------------------
def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    if x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x[mask]))
    ry = np.argsort(np.argsort(y[mask]))
    return float(np.corrcoef(rx, ry)[0, 1])


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_per_profile_csv(csv_path: Path) -> dict:
    """
    Read per_profile_summary.csv.  Returns a dict of numpy arrays keyed by
    column name, plus the derived per-level RMSE matrix.
    """
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f'{csv_path} is empty')

    # Identify the per-level RMSE columns in level order.
    level_cols = sorted(
        [c for c in rows[0].keys() if c.startswith('rmse_L') and c.endswith('_um')],
        key=lambda c: int(c[len('rmse_L'):-len('_um')]),
    )
    n_levels = len(level_cols)
    n_unique = len(rows)

    def col(name, dtype=float):
        return np.array([dtype(r[name]) for r in rows])

    rmse_matrix = np.zeros((n_unique, n_levels), dtype=np.float64)
    for j, r in enumerate(rows):
        for L, lc in enumerate(level_cols):
            rmse_matrix[j, L] = float(r[lc])

    return {
        'pid':                  col('pid', int),
        'n_test_samples':       col('n_test_samples', int),
        'mean_rmse_um':         col('mean_rmse_um'),
        'mean_pred_sigma_um':   col('mean_pred_sigma_um'),
        'rmse_sigma_ratio':     col('rmse_sigma_ratio'),
        'tau_c':                col('tau_c'),
        'wv_above_cloud':       col('wv_above_cloud'),
        'wv_in_cloud':          col('wv_in_cloud'),
        'adiabaticity_score':   col('adiabaticity_score'),
        're_top_um':            col('re_top_um'),
        're_base_um':           col('re_base_um'),
        'drizzle_proxy_re_max_lower30pct_um':
                                col('drizzle_proxy_re_max_lower30pct_um'),
        're_max_overall_um':    col('re_max_overall_um'),
        'n_raw_levels':         col('n_raw_levels', int),
        'rmse_per_level':       rmse_matrix,
        'n_levels':             n_levels,
        'n_unique':             n_unique,
    }


def load_correlations(json_path: Path) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def load_overall_summary(json_path: Path) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def load_spectral_csv(csv_path: Path) -> dict:
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f'{csv_path} is empty')
    fold_cols = sorted(
        [c for c in rows[0].keys() if c.startswith('importance_fold_')],
        key=lambda c: int(c.split('_')[2]),
    )
    n_folds = len(fold_cols)
    n_wl    = len(rows)

    def col(name, dtype=float):
        return np.array([dtype(r[name]) for r in rows])

    importance_per_fold = np.zeros((n_folds, n_wl), dtype=np.float64)
    for c, lc in enumerate(fold_cols):
        for w, r in enumerate(rows):
            importance_per_fold[c, w] = float(r[lc])

    return {
        'wavelength_idx':           col('wavelength_idx', int),
        'wavelength_nm':            col('wavelength_nm'),
        'importance_mean_um':       col('importance_mean_um'),
        'importance_std_um':        col('importance_std_um'),
        'importance_relative_mean': col('importance_relative_mean'),
        'importance_per_fold':      importance_per_fold,
        'n_folds':                  n_folds,
    }


# -----------------------------------------------------------------------------
# Figure 1: per-profile mean RMSE distribution
# -----------------------------------------------------------------------------
def plot_per_profile_rmse_distribution(per_profile: dict,
                                       n_folds: int,
                                       fig_path: Path,
                                       dpi: int = 500):
    """
    Two-panel figure (rows = 2):
        top    — histogram of per-profile mean RMSE (matches old plot)
        bottom — empirical cumulative count of profiles vs mean RMSE,
                 showing how quickly the cumulative count saturates near
                 the total (i.e. most profiles have low RMSE).
    """
    # Tweak these for styling -------------------------------------------------
    bins         = 30
    bar_color    = CB['blue']
    median_color = CB['vermillion']
    mean_color   = CB['orange']
    cdf_color    = CB['blue']
    figsize      = (8, 9)
    # ------------------------------------------------------------------------
    mean_rmse = per_profile['mean_rmse_um']
    n_unique  = per_profile['n_unique']
    med = float(np.median(mean_rmse))
    mu  = float(mean_rmse.mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # ── Top: histogram ───────────────────────────────────────────────────────
    ax1.hist(mean_rmse, bins=bins, color=bar_color, edgecolor='white')
    ax1.axvline(med, color=median_color, linewidth=1.5,
                label=fr'median $= {med:.2f}\ \mu\mathrm{{m}}$')
    ax1.axvline(mu, color=mean_color, linewidth=1.5, linestyle='--',
                label=fr'mean $= {mu:.2f}\ \mu\mathrm{{m}}$')
    ax1.set_xlabel(r'Per-profile mean RMSE ($\mu$m)')
    ax1.set_ylabel(r'Number of profiles')
    ax1.set_title(fr'Per-Profile RMSE Distribution '
                  fr'($K={n_folds}$, $n_{{\mathrm{{unique}}}}={n_unique}$ '
                  fr'profiles)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # ── Bottom: empirical CDF (fraction, 0 to 1) ─────────────────────────────
    sorted_rmse = np.sort(mean_rmse)
    cdf         = np.arange(1, len(sorted_rmse) + 1) / len(sorted_rmse)
    ax2.step(sorted_rmse, cdf, where='post',
             color=cdf_color, linewidth=1.6)
    ax2.axhline(1.0, color='k', linewidth=0.5, linestyle=':', alpha=0.5)
    ax2.axvline(med, color=median_color, linewidth=1.5,
                label=fr'median $= {med:.2f}\ \mu\mathrm{{m}}$')
    ax2.set_xlabel(r'Per-profile mean RMSE ($\mu$m)')
    ax2.set_ylabel(r'Cumulative fraction of profiles')
    ax2.set_title(r'Cumulative Distribution')
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
# Figure 2: scatter of per-profile mean RMSE vs each physical predictor
# -----------------------------------------------------------------------------
def plot_rmse_vs_predictors(per_profile: dict,
                            correlations: dict,
                            n_folds: int,
                            fig_path: Path,
                            dpi: int = 500):
    """
    3x2 panel of per-profile mean RMSE vs physical predictors.

    Switched from log10(wv_*) to absolute integrated water vapor on the
    x-axis, so Pearson r is recomputed on the fly (Spearman rho is
    unchanged by a monotonic transform).  The cached
    per_profile_correlations.json is therefore used only for predictors
    we did NOT transform.
    """
    # Tweak these for styling -------------------------------------------------
    point_color = CB['blue']
    figsize     = (13, 12)
    panel_grid  = (3, 2)
    # ------------------------------------------------------------------------

    # Convert column water vapor from molec/cm^2 to mm of precipitable water
    # (equivalently kg/m^2).  The conversion factor is
    #     M_H2O / N_A  *  10                (g/cm^2 to kg/m^2 = mm PW)
    #     = 18.01528 / 6.02214076e23 * 10
    #     ≈ 2.99151e-22  mm-PW per (molec/cm^2)
    # so the typical above-cloud value of ~6e21 molec/cm^2 ≈ 1.8 mm PW.
    MOLEC_PER_CM2_TO_MM_PW = 18.01528 / 6.02214076e23 * 10.0

    # Build the predictors dict — wv_* now converted to mm of precipitable water.
    predictors = {
        'tau_c':                              per_profile['tau_c'],
        'wv_above_cloud':                     (per_profile['wv_above_cloud']
                                               * MOLEC_PER_CM2_TO_MM_PW),
        'wv_in_cloud':                        (per_profile['wv_in_cloud']
                                               * MOLEC_PER_CM2_TO_MM_PW),
        'adiabaticity_score':                 per_profile['adiabaticity_score'],
        'drizzle_proxy_re_max_lower30pct_um': per_profile['drizzle_proxy_re_max_lower30pct_um'],
        're_max_overall_um':                  per_profile['re_max_overall_um'],
    }

    # (key, x-axis label as raw-string mathtext, x-axis scale)
    panel = [
        ('tau_c',
         r'$\tau_c$', 'linear'),
        ('wv_above_cloud',
         r'Above-cloud column water vapor (mm $\equiv$ kg m$^{-2}$)',
         'linear'),
        ('wv_in_cloud',
         r'In-cloud column water vapor (mm $\equiv$ kg m$^{-2}$)',
         'linear'),
        ('adiabaticity_score',
         r'Adiabaticity (Pearson $r$ of $r_e^{\,3}$ vs $z$ above base)',
         'linear'),
        ('drizzle_proxy_re_max_lower30pct_um',
         r'Drizzle proxy: max $r_e$ in base $30\%$ ($\mu$m)', 'linear'),
        ('re_max_overall_um',
         r'Max $r_e$ in profile ($\mu$m)', 'linear'),
    ]

    mean_rmse = per_profile['mean_rmse_um']
    n_unique  = per_profile['n_unique']

    fig, axes = plt.subplots(*panel_grid, figsize=figsize)
    for ax, (key, lbl, scale) in zip(axes.flat, panel):
        x = predictors[key]
        ax.scatter(x, mean_rmse, alpha=0.55, s=24,
                   c=point_color, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(lbl)
        ax.set_ylabel(r'Per-profile mean RMSE ($\mu$m)')
        if scale == 'log':
            ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Re-compute correlations against the predictor values actually
        # plotted, so the r/rho in the title always match the panel.
        pr = _pearson_r(x, mean_rmse)
        sp = _spearman_r(x, mean_rmse)
        ax.set_title(fr'$r = {pr:+.3f},\ \ \rho = {sp:+.3f}$', fontsize=11)

        # Force scientific notation when the data span many orders of
        # magnitude (the wv_* columns are O(1e21)).  Matplotlib will
        # decide whether to apply it via scilimits.
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 4),
                            useMathText=True)

    fig.suptitle(fr'Per-Profile RMSE vs Physical Predictors '
                 fr'($K={n_folds}$, $n_{{\mathrm{{unique}}}}={n_unique}$)',
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Figure 3: per-level RMSE mean +/- std across profiles
# -----------------------------------------------------------------------------
def plot_per_level_uncertainty(per_profile: dict,
                               fig_path: Path,
                               statistic: str = 'mean',
                               dpi: int = 500):
    """
    Per-level RMSE summary across all unique profiles.

    statistic = 'mean'   → centre line = mean,   band = +/- 1 std
    statistic = 'median' → centre line = median, band = IQR (25th-75th
                           percentile).  IQR is the natural companion to
                           the median (the median has no native +/- sigma).

    Y-axis is normalized altitude with respect to the cloud, ranging from
    0 (cloud top) to 1 (cloud base).  Level k (1-indexed) maps to
    (k - 1) / (n_levels - 1).
    """
    # Tweak these for styling -------------------------------------------------
    line_color = CB['bluish_green']
    band_alpha = 0.20
    figsize    = (7, 9)
    # ------------------------------------------------------------------------
    if statistic not in ('mean', 'median'):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")

    rmse = per_profile['rmse_per_level']         # (n_unique, n_levels)
    n_levels = per_profile['n_levels']
    n_unique = per_profile['n_unique']

    # Normalized altitude: 0 = cloud top (level 1), 1 = cloud base (level n).
    z_norm = np.arange(n_levels) / (n_levels - 1)

    if statistic == 'mean':
        centre = rmse.mean(axis=0)
        spread_lo = centre - rmse.std(axis=0)
        spread_hi = centre + rmse.std(axis=0)
        centre_label = r'mean RMSE across profiles'
        band_label   = r'$\pm 1\,\sigma$ across profiles'
        stat_word    = r'Mean'
    else:  # median
        centre    = np.median(rmse, axis=0)
        spread_lo = np.percentile(rmse, 25, axis=0)
        spread_hi = np.percentile(rmse, 75, axis=0)
        centre_label = r'median RMSE across profiles'
        band_label   = r'IQR (25th – 75th percentile)'
        stat_word    = r'Median'

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(centre, z_norm, '-', color=line_color, linewidth=1.6,
            label=centre_label)
    ax.fill_betweenx(z_norm, spread_lo, spread_hi,
                     color=line_color, alpha=band_alpha, label=band_label)
    ax.set_ylabel(r'Normalized Altitude with Cloud ($0 =$ cloud top)')
    ax.set_xlabel(r'RMSE ($\mu$m)')
    ax.set_title(fr'Per-Level RMSE Uncertainty ({stat_word}) '
                 fr'across ${n_unique}$ unique profiles')
    ax.set_ylim(1, 0)        # 0 at top, 1 at bottom
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
# Figure 4: per-profile x per-level RMSE heatmap (rows sorted by mean RMSE)
# -----------------------------------------------------------------------------
def plot_per_profile_rmse_heatmap(per_profile: dict,
                                  fig_path: Path,
                                  dpi: int = 500):
    """
    Per-profile x per-level RMSE heatmap.

    Axes are flipped vs the original: levels are now on the y-axis (level 1
    at the top of the plot = cloud top, level n at the bottom = cloud
    base), and the unique profiles are on the x-axis, sorted left-to-right
    by mean RMSE.
    """
    # Tweak these for styling -------------------------------------------------
    cmap        = CB_CMAP        # viridis: perceptually uniform + CB-safe
    vmax_pct    = 99             # clip color scale at this percentile
    figsize     = (10, 8)
    # ------------------------------------------------------------------------
    rmse      = per_profile['rmse_per_level']     # (n_unique, n_levels)
    mean_rmse = per_profile['mean_rmse_um']
    n_levels  = per_profile['n_levels']

    sort_idx = np.argsort(mean_rmse)
    # Transpose so rows = levels (y-axis), columns = profiles (x-axis).
    # imshow plots row 0 at the top by default → level 1 lands at cloud top.
    img = rmse[sort_idx].T                        # (n_levels, n_unique)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, aspect='auto', origin='upper',
                   cmap=cmap, vmin=0, vmax=np.percentile(rmse, vmax_pct))
    ax.set_xlabel(r'Unique profile (sorted by mean RMSE)')
    ax.set_ylabel(fr'Vertical level ($1 =$ cloud top, ${n_levels} =$ cloud base)')
    ax.set_title(r'Per-Profile $\times$ Per-Level RMSE')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'RMSE ($\mu$m)')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
# Figure 5: spectral feature importance (relative, peak = 1)
# -----------------------------------------------------------------------------
def plot_spectral_importance_relative(spectral: dict,
                                      n_eval_per_fold_label: str,
                                      fig_path: Path,
                                      dpi: int = 500):
    # Tweak these for styling -------------------------------------------------
    line_color    = CB['blue']
    line_alpha    = 0.55
    line_width    = 0.6
    marker_size   = 8
    figsize       = (13, 5)
    # ------------------------------------------------------------------------
    wl     = spectral['wavelength_nm']
    rel    = spectral['importance_relative_mean']
    n_folds = spectral['n_folds']

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wl, rel, '-', linewidth=line_width, color=line_color,
            alpha=line_alpha, zorder=1)
    ax.scatter(wl, rel, s=marker_size, color=line_color,
               edgecolors='none', zorder=2)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.3)
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel(r'Relative feature importance (peak $\Delta$-RMSE $= 1$)')
    ax.set_title(fr'Spectral Feature Importance — mean across ${n_folds}$ '
                 fr'folds (mean-substitution $\Delta$-RMSE'
                 fr'{n_eval_per_fold_label})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
# Figure 6: spectral feature importance (absolute Delta-RMSE in micrometres)
# -----------------------------------------------------------------------------
def plot_spectral_importance_absolute(spectral: dict,
                                      fig_path: Path,
                                      dpi: int = 500):
    # Tweak these for styling -------------------------------------------------
    color       = CB['blue']
    band_alpha  = 0.18
    line_alpha  = 0.85
    line_width  = 0.6
    marker_size = 8
    figsize     = (13, 5)
    # ------------------------------------------------------------------------
    wl  = spectral['wavelength_nm']
    mu  = spectral['importance_mean_um']
    sd  = spectral['importance_std_um']

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(wl, mu - sd, mu + sd,
                    color=color, alpha=band_alpha,
                    label=r'$\pm 1\,\sigma$ across folds')
    ax.plot(wl, mu, '-', linewidth=line_width, color=color,
            alpha=line_alpha, zorder=1,
            label=r'mean $\Delta$-RMSE')
    ax.scatter(wl, mu, s=marker_size, color=color,
               edgecolors='none', zorder=2)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.3)
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel(r'Mean-substitution $\Delta$-RMSE ($\mu$m)')
    ax.set_title(r'Spectral Feature Importance — absolute $\Delta$-RMSE '
                 r'(higher $=$ scrambling this wavelength hurt more)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
# Paper version of the predictors plot — 2x2 panel with the four predictors
# the manuscript focuses on (tau_c, above-cloud column water vapor,
# adiabaticity, drizzle proxy).  The cached 3x2 version is still produced
# by plot_rmse_vs_predictors() above.
# -----------------------------------------------------------------------------
def plot_rmse_vs_predictors_paper(per_profile: dict,
                                  n_folds: int,
                                  fig_path: Path,
                                  dpi: int = 500):
    """
    2x2 panel of per-profile mean RMSE vs the four physical predictors
    chosen for the paper:

        top row    | tau_c                     | above-cloud column water vapor
        bottom row | adiabaticity score        | drizzle proxy

    Subplot titles use the larger fontsize requested for the paper version.
    """
    # Tweak these for styling -------------------------------------------------
    point_color = CB['blue']
    figsize     = (12, 10)
    panel_grid  = (2, 2)
    title_fs    = 14    # subplot-title font size (was 11 in the diagnostic plot)
    # ------------------------------------------------------------------------

    MOLEC_PER_CM2_TO_MM_PW = 18.01528 / 6.02214076e23 * 10.0

    predictors = {
        'tau_c':                              per_profile['tau_c'],
        'wv_above_cloud':                     (per_profile['wv_above_cloud']
                                               * MOLEC_PER_CM2_TO_MM_PW),
        'adiabaticity_score':                 per_profile['adiabaticity_score'],
        'drizzle_proxy_re_max_lower30pct_um': per_profile['drizzle_proxy_re_max_lower30pct_um'],
    }

    # Order: top-left, top-right, bottom-left, bottom-right.
    panel = [
        ('tau_c',
         r'$\tau_c$'),
        ('wv_above_cloud',
         r'Above-cloud column water vapor (mm $\equiv$ kg m$^{-2}$)'),
        ('adiabaticity_score',
         r'Adiabaticity (Pearson $r$ of $r_e^{\,3}$ vs $z$ above base)'),
        ('drizzle_proxy_re_max_lower30pct_um',
         r'Drizzle proxy: max $r_e$ in base $30\%$ ($\mu$m)'),
    ]

    mean_rmse = per_profile['mean_rmse_um']
    n_unique  = per_profile['n_unique']

    fig, axes = plt.subplots(*panel_grid, figsize=figsize)
    for ax, (key, lbl) in zip(axes.flat, panel):
        x = predictors[key]
        ax.scatter(x, mean_rmse, alpha=0.55, s=24,
                   c=point_color, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(lbl)
        ax.set_ylabel(r'Per-profile mean RMSE ($\mu$m)')
        ax.grid(True, alpha=0.3)

        pr = _pearson_r(x, mean_rmse)
        sp = _spearman_r(x, mean_rmse)
        ax.set_title(fr'$r = {pr:+.3f},\ \ \rho = {sp:+.3f}$',
                     fontsize=title_fs)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 4),
                            useMathText=True)

    fig.suptitle(fr'Per-Profile RMSE vs Physical Predictors '
                 fr'($K={n_folds}$, $n_{{\mathrm{{unique}}}}={n_unique}$)',
                 fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Predicted vs True droplet profiles at six RMSE percentiles — 2x3 panel.
# Uses HDF5 (raw r_e and z) + fold_NN/test_predictions.npz (50-level true
# and NN prediction).  Cached to example_profiles_data.npz for fast
# iteration on styling.  See build_example_profiles_data() for details.
# -----------------------------------------------------------------------------
EXAMPLE_PROFILES_CACHE_NAME = 'example_profiles_data.npz'
DEFAULT_EXAMPLE_PERCENTILES = (10, 25, 40, 55, 70, 85)


def select_profiles_at_percentiles(per_profile: dict,
                                   percentiles=DEFAULT_EXAMPLE_PERCENTILES) -> list:
    """
    For each requested mean-RMSE percentile, return the unique-profile
    record (pid, mean_rmse_um, percentile, rank) closest to that
    percentile of the per-profile mean-RMSE distribution.
    """
    mean_rmse = per_profile['mean_rmse_um']
    pids      = per_profile['pid']
    n         = len(mean_rmse)
    sort_idx  = np.argsort(mean_rmse)
    selected = []
    for p in percentiles:
        rank = int(round(p / 100.0 * (n - 1)))
        i = sort_idx[rank]
        selected.append({'pid':        int(pids[i]),
                         'rmse_um':    float(mean_rmse[i]),
                         'percentile': float(p),
                         'rank':       int(rank)})
    return selected


def build_example_profiles_data(results_dir: Path,
                                h5_path: Path,
                                per_profile: dict,
                                percentiles=DEFAULT_EXAMPLE_PERCENTILES) -> dict:
    """
    For each profile picked by select_profiles_at_percentiles(), gather:
        - raw in-situ r_e profile and altitude (variable length, NaN-padded)
        - 50-level "true" profile (the network's training target, fixed grid)
        - 50-level NN prediction (mean across all test samples for that pid)
        - 50-level NN per-level sigma (mean across samples)

    Caches everything to <results_dir>/example_profiles_data.npz so future
    runs can skip the HDF5 entirely.
    """
    import h5py    # lazy: only this code path needs HDF5.

    selected = select_profiles_at_percentiles(per_profile, percentiles)
    target_pids = {s['pid']: s for s in selected}

    fold_dirs = sorted(results_dir.glob('fold_*'))
    if not fold_dirs:
        raise FileNotFoundError(f'No fold_*/ subdirs in {results_dir}')

    # Pass 1 — collect ALL samples for each target pid (predictions + the
    # hdf5 index of every sample).  Each unique pid lives in exactly one
    # fold's test set, so once we find it we can break out of subsequent
    # fold scans for that pid.
    samples = {}      # pid -> dict of (pred, pred_std, true, hdf5_idx) arrays
    for fd in fold_dirs:
        npz_path = fd / 'test_predictions.npz'
        if not npz_path.exists():
            continue
        npz       = np.load(npz_path)
        pred      = npz['pred']
        true      = npz['true']
        pred_std  = npz['pred_std']
        pids_arr  = npz['profile_ids']
        h5idx_arr = npz['hdf5_indices']
        for pid in target_pids:
            if pid in samples:
                continue
            sel = np.where(pids_arr == pid)[0]
            if sel.size == 0:
                continue
            samples[pid] = {
                'pred':     pred[sel].copy(),
                'pred_std': pred_std[sel].copy(),
                'true':     true[sel].copy(),
                'hdf5_idx': h5idx_arr[sel].copy(),
            }
    missing = [pid for pid in target_pids if pid not in samples]
    if missing:
        raise RuntimeError(f'Could not locate predictions for pids {missing} '
                           f'across folds in {results_dir}')

    ordered_pids = [s['pid'] for s in selected]
    n_levels = samples[ordered_pids[0]]['pred'].shape[1]

    # Means across all geometries — used by the mean-version plot.
    pred_means = {pid: s['pred'].mean(axis=0)     for pid, s in samples.items()}
    pred_stds  = {pid: s['pred_std'].mean(axis=0) for pid, s in samples.items()}
    true_means = {pid: s['true'].mean(axis=0)     for pid, s in samples.items()}

    # Read the four geometry arrays + raw profiles in a single HDF5 pass.
    with h5py.File(h5_path, 'r') as f:
        sza_all = f['sza'][:]
        vza_all = f['vza'][:]
        saz_all = f['saz'][:]
        vaz_all = f['vaz'][:]

        max_raw_lev = int(f['profiles_raw'].shape[1])
        raw_re   = np.full((len(target_pids), max_raw_lev), np.nan, dtype=np.float64)
        raw_z_km = np.full((len(target_pids), max_raw_lev), np.nan, dtype=np.float64)
        n_raw    = np.zeros(len(target_pids), dtype=np.int32)
        for j, pid in enumerate(ordered_pids):
            # Any sample of this pid exposes the same raw profile/z grid
            # (raw fields are per-profile, repeated across geometries).
            row = int(samples[pid]['hdf5_idx'][0])
            nL  = int(f['profile_n_levels'][row])
            raw_re[j, :nL]   = f['profiles_raw'][row, :nL]
            raw_z_km[j, :nL] = f['profiles_raw_z'][row, :nL]
            n_raw[j] = nL

    # ── Find a viewing geometry common to all selected profiles ────────────
    # Geometries are gridded at integer-ish degree values; round to 1 dp to
    # absorb any float drift introduced when the HDF5 was written.
    def _round_geom(i):
        return (round(float(sza_all[i]), 1),
                round(float(vza_all[i]), 1),
                round(float(saz_all[i]), 1),
                round(float(vaz_all[i]), 1))

    pid_geoms = {pid: set(_round_geom(int(i)) for i in s['hdf5_idx'])
                 for pid, s in samples.items()}
    common_geoms = set.intersection(*pid_geoms.values())
    if not common_geoms:
        raise RuntimeError(
            'No (SZA, VZA, SAZ, VAZ) combination is common to all '
            f'{len(samples)} selected profiles, so no single shared '
            'geometry can be chosen.  Either pick different percentiles '
            'or relax the selection rule (e.g. nearest-neighbor matching).'
        )
    # Lex-sorted minimum — fully deterministic.  Change this line if you
    # want a different shared-geometry selection rule (e.g. closest to
    # nadir, fixed (SZA, VZA), median across the intersection set).
    shared_geom = min(common_geoms)

    # For each pid, find the sample whose geometry matches shared_geom.
    pred_single, sigma_single, true_single, single_h5idx = {}, {}, {}, {}
    for pid in ordered_pids:
        s = samples[pid]
        match = None
        for j, h5i in enumerate(s['hdf5_idx']):
            if _round_geom(int(h5i)) == shared_geom:
                match = j
                break
        if match is None:
            raise RuntimeError(
                f'pid {pid} has no sample matching the shared geometry '
                f'{shared_geom} — this should not happen since the geometry '
                f'came from set intersection.'
            )
        pred_single[pid]  = s['pred'][match]
        sigma_single[pid] = s['pred_std'][match]
        true_single[pid]  = s['true'][match]
        single_h5idx[pid] = int(s['hdf5_idx'][match])

    # Pack arrays in selection order.
    pred_50  = np.stack([pred_means[pid] for pid in ordered_pids])
    sigma_50 = np.stack([pred_stds[pid]  for pid in ordered_pids])
    true_50  = np.stack([true_means[pid] for pid in ordered_pids])

    pred_single_50  = np.stack([pred_single[pid]  for pid in ordered_pids])
    sigma_single_50 = np.stack([sigma_single[pid] for pid in ordered_pids])
    true_single_50  = np.stack([true_single[pid]  for pid in ordered_pids])
    single_h5idx_arr = np.array([single_h5idx[pid] for pid in ordered_pids],
                                 dtype=np.int64)
    # Per-sample RMSE for the shared-geometry sample of each profile
    # (kept in the cache for diagnostic use; not currently shown on plot).
    single_rmse = np.sqrt(((pred_single_50 - true_single_50) ** 2).mean(axis=1))

    # Shared geometry as a (4,) array.  Note the schema change vs the
    # earlier per-panel-geometry version: single_geom is now 1-D.
    single_geom = np.array(shared_geom, dtype=np.float32)

    cache_path = results_dir / EXAMPLE_PROFILES_CACHE_NAME
    np.savez(cache_path,
             percentiles      = np.array([s['percentile'] for s in selected]),
             pids             = np.array([s['pid'] for s in selected], dtype=np.int32),
             rmse_um          = np.array([s['rmse_um'] for s in selected]),
             pred_50          = pred_50.astype(np.float32),
             sigma_50         = sigma_50.astype(np.float32),
             true_50          = true_50.astype(np.float32),
             raw_re           = raw_re.astype(np.float32),
             raw_z_km         = raw_z_km.astype(np.float32),
             n_raw_levels     = n_raw,
             n_levels         = np.int32(n_levels),
             # Single-geometry fields (shared across all panels).
             pred_single_50   = pred_single_50.astype(np.float32),
             sigma_single_50  = sigma_single_50.astype(np.float32),
             true_single_50   = true_single_50.astype(np.float32),
             single_geom      = single_geom,         # shape (4,) — SHARED
             single_rmse_um   = single_rmse.astype(np.float32),
             single_hdf5_idx  = single_h5idx_arr)
    print(f'  cached {cache_path.name} '
          f'({len(selected)} profiles, percentiles {list(percentiles)})')

    return load_example_profiles_cache(cache_path)


def load_example_profiles_cache(cache_path: Path) -> dict:
    """
    Load the cached example-profiles NPZ as a plain dict.

    Cache schema v2 added single-geometry fields (pred_single_50, sigma_single_50,
    single_geom, single_rmse_um, single_hdf5_idx).  These are surfaced when
    present and silently omitted when absent — older v1 caches still load
    cleanly for the mean-version plot.
    """
    npz  = np.load(cache_path)
    keys = set(npz.files)
    out = {
        'percentiles':  npz['percentiles'],
        'pids':         npz['pids'],
        'rmse_um':      npz['rmse_um'],
        'pred_50':      npz['pred_50'],
        'sigma_50':     npz['sigma_50'],
        'true_50':      npz['true_50'],
        'raw_re':       npz['raw_re'],
        'raw_z_km':     npz['raw_z_km'],
        'n_raw_levels': npz['n_raw_levels'],
        'n_levels':     int(npz['n_levels']),
    }
    for k in ('pred_single_50', 'sigma_single_50', 'true_single_50',
              'single_geom', 'single_rmse_um', 'single_hdf5_idx'):
        if k in keys:
            out[k] = npz[k]
    return out


def plot_example_profiles(data: dict,
                          median_rmse_per_level: np.ndarray,
                          fig_path: Path,
                          dpi: int = 500):
    """
    2x3 panel: predicted vs true droplet profiles at six per-profile mean
    RMSE percentiles (10, 25, 40 in the top row; 55, 70, 85 in the bottom
    row).  Each panel shows:
        - the 50-level interpolated true profile (black markers + thin
          black line),
        - the NN's predicted profile (markers + dashed line) with a
          shaded band of half-width = median per-level RMSE.

    The shaded band is the SAME vector at every panel — it represents the
    typical (median) RMSE the model makes at each vertical level across
    the full set of 290 unique profiles, i.e. the centerline of the
    per-level uncertainty median plot.
    """
    # Tweak these for styling -------------------------------------------------
    true_color   = CB['black']
    pred_color   = CB['vermillion']
    band_alpha   = 0.20
    figsize      = (15, 9)
    title_fs     = 14
    true_lw      = 0.8     # thin connector line for the 50-level markers
    # ------------------------------------------------------------------------

    n_panels   = data['pred_50'].shape[0]
    n_levels   = data['n_levels']
    if median_rmse_per_level.shape[0] != n_levels:
        raise ValueError(
            f'median_rmse_per_level has {median_rmse_per_level.shape[0]} '
            f'levels but the cached predictions have {n_levels}'
        )
    z_norm_50  = np.linspace(0, 1, n_levels)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for k, ax in enumerate(axes.flat):
        if k >= n_panels:
            ax.axis('off')
            continue

        nL          = int(data['n_raw_levels'][k])
        raw_z_km_k  = data['raw_z_km'][k, :nL]
        z_top_km    = float(raw_z_km_k[0])
        z_base_km   = float(raw_z_km_k[-1])
        thick_m     = (z_top_km - z_base_km) * 1000.0
        z_50_m      = z_norm_50 * thick_m

        pred_k = data['pred_50'][k]
        true_k = data['true_50'][k]

        # 50-level interpolated true profile — black markers + thin line
        ax.plot(true_k, z_50_m, 'o-', color=true_color, markersize=3,
                linewidth=true_lw, label=fr'True ({n_levels}-level)')

        # NN prediction — shaded band of half-width median RMSE per level
        lo = pred_k - median_rmse_per_level
        hi = pred_k + median_rmse_per_level
        ax.fill_betweenx(z_50_m, lo, hi, color=pred_color, alpha=band_alpha,
                         linewidth=0,
                         label=r'NN estimate $\pm$ median RMSE per level')
        ax.plot(pred_k, z_50_m, 's--', color=pred_color, markersize=3,
                linewidth=1.2, label=r'NN estimate (mean)')

        ax.invert_yaxis()    # 0 m at top, cloud base at bottom
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r'$r_e$ ($\mu$m)')
        if k % 3 == 0:
            ax.set_ylabel(r'Depth from cloud top (m)')

        pct = float(data['percentiles'][k])
        rmse_k = float(data['rmse_um'][k])
        ax.set_title(fr'{int(pct)}th percentile  '
                     fr'(mean RMSE $= {rmse_k:.2f}\ \mu\mathrm{{m}}$)',
                     fontsize=title_fs)
        if k == 0:
            ax.legend(loc='best', fontsize=9)

    fig.suptitle(r'Predicted vs True Droplet Profiles — six examples '
                 r'spanning the per-profile RMSE distribution',
                 fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_example_profiles_single_geom(data: dict,
                                      median_rmse_per_level: np.ndarray,
                                      fig_path: Path,
                                      dpi: int = 500):
    """
    Single-geometry counterpart to plot_example_profiles().

    Same six profiles, same percentile labels, same shaded median-RMSE
    band — but every panel now shows the network's retrieval for the
    SAME SHARED viewing geometry (chosen by build_example_profiles_data
    as a (SZA, VZA, SAZ, VAZ) tuple common to all six profiles).

    Because the geometry is identical across panels, the geometry
    annotation appears only on the first panel (just below the legend).
    """
    if 'pred_single_50' not in data:
        raise RuntimeError(
            'Cached example-profiles NPZ does not contain single-geometry '
            'fields.  Delete example_profiles_data.npz and rerun with '
            '--h5-path to rebuild the cache.'
        )
    geom = np.asarray(data['single_geom'])
    if geom.ndim != 1 or geom.shape[0] != 4:
        raise RuntimeError(
            'Cached single_geom has unexpected shape '
            f'{tuple(geom.shape)}.  This script now requires a single '
            'shared geometry (shape (4,)) across all panels.  Delete '
            'example_profiles_data.npz and rerun with --h5-path to rebuild.'
        )

    # Tweak these for styling -------------------------------------------------
    true_color   = CB['black']
    pred_color   = CB['vermillion']
    band_alpha   = 0.20
    figsize      = (15, 9)
    title_fs     = 14
    true_lw      = 0.8
    legend_fs    = 9
    annot_fs     = 9
    # y-coord of the top of the geometry textbox (axis fraction).  Tuned
    # to sit just below a 3-line legend at fontsize=9; nudge if you change
    # the legend or font size.
    geom_box_y   = 0.66
    # ------------------------------------------------------------------------

    n_panels  = data['pred_single_50'].shape[0]
    n_levels  = data['n_levels']
    if median_rmse_per_level.shape[0] != n_levels:
        raise ValueError(
            f'median_rmse_per_level has {median_rmse_per_level.shape[0]} '
            f'levels but the cached predictions have {n_levels}'
        )
    z_norm_50 = np.linspace(0, 1, n_levels)
    sza, vza, saz, vaz = (float(v) for v in geom)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for k, ax in enumerate(axes.flat):
        if k >= n_panels:
            ax.axis('off')
            continue

        nL          = int(data['n_raw_levels'][k])
        raw_z_km_k  = data['raw_z_km'][k, :nL]
        z_top_km    = float(raw_z_km_k[0])
        z_base_km   = float(raw_z_km_k[-1])
        thick_m     = (z_top_km - z_base_km) * 1000.0
        z_50_m      = z_norm_50 * thick_m

        pred_k = data['pred_single_50'][k]
        true_k = data['true_50'][k]    # 50-level true is per-profile, not per-sample

        # 50-level true profile
        ax.plot(true_k, z_50_m, 'o-', color=true_color, markersize=3,
                linewidth=true_lw, label=fr'True ({n_levels}-level)')

        # NN single-geometry estimate with median per-level RMSE band
        lo = pred_k - median_rmse_per_level
        hi = pred_k + median_rmse_per_level
        ax.fill_betweenx(z_50_m, lo, hi, color=pred_color, alpha=band_alpha,
                         linewidth=0,
                         label=r'NN estimate $\pm$ median RMSE per level')
        ax.plot(pred_k, z_50_m, 's--', color=pred_color, markersize=3,
                linewidth=1.2, label=r'NN estimate (single geometry)')

        ax.invert_yaxis()
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r'$r_e$ ($\mu$m)')
        if k % 3 == 0:
            ax.set_ylabel(r'Depth from cloud top (m)')

        pct       = float(data['percentiles'][k])
        prof_rmse = float(data['rmse_um'][k])      # per-PROFILE mean RMSE
        ax.set_title(fr'{int(pct)}th percentile  '
                     fr'(profile mean RMSE $= {prof_rmse:.2f}\ \mu\mathrm{{m}}$)',
                     fontsize=title_fs)

        if k == 0:
            ax.legend(loc='upper left', fontsize=legend_fs)
            # Geometry textbox sits just below the legend.  Same geometry
            # applies to every panel, so we annotate once.
            ax.text(0.03, geom_box_y,
                    f'SZA = {sza:.0f}°,  VZA = {vza:.0f}°\n'
                    f'SAZ = {saz:.0f}°,  VAZ = {vaz:.0f}°',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=annot_fs,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='white', alpha=0.85,
                              edgecolor='lightgray', linewidth=0.6))

    fig.suptitle(r'Predicted vs True Droplet Profiles (single geometry) — '
                 r'six examples spanning the per-profile RMSE distribution',
                 fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Per-sample RMSE vs viewing geometry — needs HDF5 + the cached
# fold_NN/test_predictions.npz files (the per-profile CSV does not carry
# geometry).  Cached to per_sample_geometry_rmse.csv on first run so future
# iterations on the plot do not re-touch the HDF5.
# -----------------------------------------------------------------------------
GEOMETRY_CACHE_NAME = 'per_sample_geometry_rmse.csv'


def build_per_sample_geometry_rmse(results_dir: Path,
                                   h5_path: Path) -> dict:
    """
    Walk results_dir/fold_NN/test_predictions.npz, compute per-sample RMSE
    (averaged over the n_levels output dimension), and join geometry from
    the HDF5 by hdf5_indices.

    Writes per_sample_geometry_rmse.csv next to the existing artifacts so
    later runs can skip the HDF5 entirely.
    """
    import h5py    # imported lazily — only this code path needs it.

    fold_dirs = sorted(results_dir.glob('fold_*'))
    if not fold_dirs:
        raise FileNotFoundError(f'No fold_*/ subdirs in {results_dir}')

    # Pull the four geometry arrays once from the HDF5.
    with h5py.File(h5_path, 'r') as f:
        sza_all = f['sza'][:].astype(np.float32)
        vza_all = f['vza'][:].astype(np.float32)
        saz_all = f['saz'][:].astype(np.float32)
        vaz_all = f['vaz'][:].astype(np.float32)

    idx_list, pid_list, fold_list = [], [], []
    rmse_list = []
    for fd in fold_dirs:
        npz_path = fd / 'test_predictions.npz'
        if not npz_path.exists():
            print(f'  [skip] {npz_path.name} missing under {fd.name}')
            continue
        npz   = np.load(npz_path)
        pred  = npz['pred']
        true  = npz['true']
        idx   = npz['hdf5_indices'].astype(np.int64)
        pids  = npz['profile_ids'].astype(np.int32)
        # Per-sample RMSE: sqrt(mean over the n_levels axis of squared error).
        rmse  = np.sqrt(((pred - true) ** 2).mean(axis=1)).astype(np.float64)
        f_idx = int(fd.name.split('_')[-1])

        idx_list.append(idx)
        pid_list.append(pids)
        fold_list.append(np.full(idx.size, f_idx, dtype=np.int32))
        rmse_list.append(rmse)

    hdf5_idx = np.concatenate(idx_list)
    pids     = np.concatenate(pid_list)
    folds    = np.concatenate(fold_list)
    rmse     = np.concatenate(rmse_list)
    sza      = sza_all[hdf5_idx]
    vza      = vza_all[hdf5_idx]
    saz      = saz_all[hdf5_idx]
    vaz      = vaz_all[hdf5_idx]

    # Cache to CSV so subsequent runs can skip the HDF5 entirely.
    cache_path = results_dir / GEOMETRY_CACHE_NAME
    fieldnames = ['hdf5_index', 'profile_id', 'fold_idx',
                  'sza_deg', 'vza_deg', 'saz_deg', 'vaz_deg', 'rmse_um']
    with open(cache_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k in range(hdf5_idx.size):
            w.writerow({
                'hdf5_index': int(hdf5_idx[k]),
                'profile_id': int(pids[k]),
                'fold_idx':   int(folds[k]),
                'sza_deg':    float(sza[k]),
                'vza_deg':    float(vza[k]),
                'saz_deg':    float(saz[k]),
                'vaz_deg':    float(vaz[k]),
                'rmse_um':    float(rmse[k]),
            })
    print(f'  cached {cache_path.name} ({hdf5_idx.size:,} samples)')

    return {'hdf5_index': hdf5_idx, 'profile_id': pids, 'fold_idx': folds,
            'sza': sza, 'vza': vza, 'saz': saz, 'vaz': vaz, 'rmse': rmse}


def load_per_sample_geometry_cache(cache_path: Path) -> dict:
    """Read the cached per-sample geometry/RMSE CSV."""
    with open(cache_path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    def col(name, dtype=float):
        return np.array([dtype(r[name]) for r in rows])

    return {
        'hdf5_index': col('hdf5_index', int),
        'profile_id': col('profile_id', int),
        'fold_idx':   col('fold_idx',   int),
        'sza':        col('sza_deg'),
        'vza':        col('vza_deg'),
        'saz':        col('saz_deg'),
        'vaz':        col('vaz_deg'),
        'rmse':       col('rmse_um'),
    }


def plot_rmse_vs_geometry(geom: dict,
                          fig_path: Path,
                          dpi: int = 500):
    """
    2x2 panel of per-sample RMSE vs the four viewing geometry angles
    (SZA, VZA, SAZ, VAZ).  Each scatter point is a single test sample.

    Pearson r and Spearman rho are shown per panel.  These are computed on
    a per-sample basis (NOT per-profile), since geometry varies sample-to-
    sample within a profile.
    """
    # Tweak these for styling -------------------------------------------------
    point_color = CB['blue']
    figsize     = (12, 10)
    panel_grid  = (2, 2)
    point_size  = 4
    point_alpha = 0.20
    # ------------------------------------------------------------------------

    rmse = geom['rmse']
    n    = rmse.size

    panel = [
        ('sza', r'Solar zenith angle (deg)'),
        ('vza', r'Viewing zenith angle (deg)'),
        ('saz', r'Solar azimuth angle (deg)'),
        ('vaz', r'Viewing azimuth angle (deg)'),
    ]

    fig, axes = plt.subplots(*panel_grid, figsize=figsize)
    for ax, (key, lbl) in zip(axes.flat, panel):
        x = geom[key]
        ax.scatter(x, rmse, s=point_size, c=point_color,
                   alpha=point_alpha, edgecolors='none')
        ax.set_xlabel(lbl)
        ax.set_ylabel(r'Per-sample RMSE ($\mu$m)')
        ax.grid(True, alpha=0.3)
        pr = _pearson_r(x, rmse)
        sp = _spearman_r(x, rmse)
        ax.set_title(fr'$r = {pr:+.3f},\ \ \rho = {sp:+.3f}$', fontsize=11)

    fig.suptitle(fr'Per-Sample RMSE vs Viewing Geometry '
                 fr'($n = {n:,}$ test samples across all folds)',
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# CLI / driver
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--results-dir', type=str, required=True,
                   help='Directory holding per_profile_summary.csv, '
                        'overall_summary.json, per_profile_correlations.json, '
                        'and spectral_feature_importance.csv (typically a '
                        'run<NN>_K<K>_<timestamp>/ subdir).')
    p.add_argument('--figures-dir', type=str, default=None,
                   help='Where to write PNGs.  Default: <results-dir>/figures/')
    p.add_argument('--dpi', type=int, default=500,
                   help='Output DPI (default 500)')
    p.add_argument('--h5-path', type=str, default=None,
                   help='Path to the training HDF5.  Required only the FIRST '
                        'time you build the rmse-vs-geometry plot — after '
                        'that, per_sample_geometry_rmse.csv is cached in '
                        '--results-dir and used automatically.')
    p.add_argument('--skip', type=str, nargs='*', default=[],
                   choices=['distribution', 'predictors', 'predictors_paper',
                            'per_level_mean', 'per_level_median',
                            'heatmap', 'spectral_rel', 'spectral_abs',
                            'geometry', 'example_profiles',
                            'example_profiles_single'],
                   help='Skip one or more figures by name')
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f'--results-dir not found: {results_dir}')

    fig_dir = (Path(args.figures_dir).resolve() if args.figures_dir
               else results_dir / 'figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Required input files.
    pp_csv   = results_dir / 'per_profile_summary.csv'
    corr_js  = results_dir / 'per_profile_correlations.json'
    overall  = results_dir / 'overall_summary.json'
    spec_csv = results_dir / 'spectral_feature_importance.csv'

    print(f'Results dir : {results_dir}')
    print(f'Figures dir : {fig_dir}')
    print(f'DPI         : {args.dpi}')

    setup_style()

    per_profile  = load_per_profile_csv(pp_csv)
    correlations = load_correlations(corr_js)
    summary      = load_overall_summary(overall)
    n_folds      = int(summary.get('n_folds',
                                   per_profile['n_unique']))   # fallback

    print(f'\nLoaded per-profile table: {per_profile["n_unique"]} profiles, '
          f'{per_profile["n_levels"]} levels')

    # Per-profile / per-level figures
    if 'distribution' not in args.skip:
        out = fig_dir / 'per_profile_rmse_distribution.png'
        plot_per_profile_rmse_distribution(per_profile, n_folds, out, args.dpi)
        print(f'  wrote {out.name}')
    if 'predictors' not in args.skip:
        out = fig_dir / 'rmse_vs_predictors.png'
        plot_rmse_vs_predictors(per_profile, correlations, n_folds, out, args.dpi)
        print(f'  wrote {out.name}')
    if 'predictors_paper' not in args.skip:
        out = fig_dir / 'rmse_vs_predictors_paper.png'
        plot_rmse_vs_predictors_paper(per_profile, n_folds, out, args.dpi)
        print(f'  wrote {out.name}')
    if 'per_level_mean' not in args.skip:
        out = fig_dir / 'per_level_uncertainty.png'
        plot_per_level_uncertainty(per_profile, out,
                                   statistic='mean', dpi=args.dpi)
        print(f'  wrote {out.name}')
    if 'per_level_median' not in args.skip:
        out = fig_dir / 'per_level_uncertainty_median.png'
        plot_per_level_uncertainty(per_profile, out,
                                   statistic='median', dpi=args.dpi)
        print(f'  wrote {out.name}')
    if 'heatmap' not in args.skip:
        out = fig_dir / 'per_profile_rmse_heatmap.png'
        plot_per_profile_rmse_heatmap(per_profile, out, args.dpi)
        print(f'  wrote {out.name}')

    # Spectral figures (skip cleanly if the CSV does not exist)
    if spec_csv.exists():
        spectral = load_spectral_csv(spec_csv)
        # The original spectral-importance script bakes "n_eval = N samples/fold"
        # into the title; we cannot recover N from the CSV alone, so omit it
        # unless it is recorded in overall_summary.json.
        n_eval_label = ''
        if 'n_eval_samples_per_fold' in summary:
            n_eval_label = f"; n_eval = {summary['n_eval_samples_per_fold']} samples/fold"
        if 'spectral_rel' not in args.skip:
            out = fig_dir / 'spectral_feature_importance.png'
            plot_spectral_importance_relative(spectral, n_eval_label, out, args.dpi)
            print(f'  wrote {out.name}')
        if 'spectral_abs' not in args.skip:
            out = fig_dir / 'spectral_feature_importance_absolute.png'
            plot_spectral_importance_absolute(spectral, out, args.dpi)
            print(f'  wrote {out.name}')
    else:
        print(f'\n[skip] {spec_csv.name} not found; spectral importance figures '
              f'not regenerated.  Run compute_spectral_feature_importance.py first.')

    # Per-sample RMSE vs geometry (uses cached CSV when present, builds it
    # from the HDF5 + fold_NN/test_predictions.npz files when --h5-path is
    # supplied, and skips cleanly otherwise).
    if 'geometry' not in args.skip:
        cache_path = results_dir / GEOMETRY_CACHE_NAME
        geom = None
        if cache_path.exists():
            geom = load_per_sample_geometry_cache(cache_path)
            print(f'\nLoaded geometry cache: {cache_path.name} '
                  f'({geom["rmse"].size:,} samples)')
        elif args.h5_path is not None:
            h5 = Path(args.h5_path).resolve()
            if not h5.exists():
                print(f'\n[skip] geometry plot: --h5-path not found: {h5}')
            else:
                print(f'\nBuilding per-sample geometry/RMSE table from HDF5 '
                      f'+ fold_NN/test_predictions.npz files ...')
                geom = build_per_sample_geometry_rmse(results_dir, h5)
        else:
            print(f'\n[skip] geometry plot: no {cache_path.name} cache and '
                  f'no --h5-path supplied.  Pass --h5-path once to build the '
                  f'cache, then iterate freely.')
        if geom is not None:
            out = fig_dir / 'rmse_vs_geometry.png'
            plot_rmse_vs_geometry(geom, out, args.dpi)
            print(f'  wrote {out.name}')

    # Example-profile comparison (raw / 50-level / NN) at six RMSE
    # percentiles.  Cache-first, just like the geometry plot.
    if 'example_profiles' not in args.skip:
        cache_path = results_dir / EXAMPLE_PROFILES_CACHE_NAME
        ex = None
        if cache_path.exists():
            ex = load_example_profiles_cache(cache_path)
            cached_pcts = tuple(int(p) for p in ex['percentiles'])
            if cached_pcts != tuple(DEFAULT_EXAMPLE_PERCENTILES):
                print(f'\n[note] example-profiles cache uses percentiles '
                      f'{cached_pcts}, not {tuple(DEFAULT_EXAMPLE_PERCENTILES)}; '
                      f'delete {cache_path.name} and pass --h5-path to rebuild.')
            else:
                print(f'\nLoaded example-profiles cache: {cache_path.name} '
                      f'({ex["pred_50"].shape[0]} profiles)')
        elif args.h5_path is not None:
            h5 = Path(args.h5_path).resolve()
            if not h5.exists():
                print(f'\n[skip] example-profiles plot: --h5-path not found: {h5}')
            else:
                print(f'\nBuilding example-profiles data (raw + 50-level + NN '
                      f'predictions) from HDF5 + folds ...')
                ex = build_example_profiles_data(results_dir, h5, per_profile,
                                                 DEFAULT_EXAMPLE_PERCENTILES)
        else:
            print(f'\n[skip] example-profiles plot: no {cache_path.name} '
                  f'cache and no --h5-path supplied.  Pass --h5-path once '
                  f'to build the cache, then iterate freely.')
        if ex is not None:
            # Median per-level RMSE (across all unique profiles) — the
            # same vector that drives the centerline of
            # per_level_uncertainty_median.png.  Used as the half-width of
            # the shaded uncertainty band on each example panel.
            median_rmse_per_level = np.median(per_profile['rmse_per_level'],
                                              axis=0)
            out = fig_dir / 'example_profiles.png'
            plot_example_profiles(ex, median_rmse_per_level, out, args.dpi)
            print(f'  wrote {out.name}')

            # Single-geometry counterpart.  Needs cache schema v2 fields;
            # if the cache pre-dates this feature we tell the user how to
            # rebuild it instead of failing.
            if 'example_profiles_single' not in args.skip:
                if 'pred_single_50' in ex:
                    out = fig_dir / 'example_profiles_single_geom.png'
                    plot_example_profiles_single_geom(
                        ex, median_rmse_per_level, out, args.dpi)
                    print(f'  wrote {out.name}')
                else:
                    print(f'\n[skip] example-profiles single-geometry plot: '
                          f'cached {EXAMPLE_PROFILES_CACHE_NAME} pre-dates '
                          f'this feature.  Delete it and rerun with '
                          f'--h5-path to rebuild.')

    print('\nDone.')


if __name__ == '__main__':
    main()
