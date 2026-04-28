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
                   choices=['distribution', 'predictors',
                            'per_level_mean', 'per_level_median',
                            'heatmap', 'spectral_rel', 'spectral_abs',
                            'geometry'],
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

    print('\nDone.')


if __name__ == '__main__':
    main()
