"""
plot_feature_importance_run004.py — Feature-importance figures for the
published paper-3 network (sweep run_004, M0).  Reads the NPZ written by
compute_feature_importance_run004.py.

Figures
-------
fig1_feature_importance_ranked_all640.png
    The "standard" feature-importance view, all 640 informative inputs.
    (a) top-30 horizontal bars (labeled, spectral vs geometry, +/-1 sigma).
    (b) full 640 ranked descending curve; the 4 geometry inputs marked.

fig2_spectral_importance_vs_wavelength.png
    636 spectral inputs only, x = wavelength (um), in spectral order — built
    to sit beside King & Vaughan (2012) Fig. 5 (Shannon information content
    per channel for a hyperspectral droplet-profile retrieval).  Permutation
    importance with a +/-1 sigma band; major water-vapor bands shaded.

fig2b_spectral_perm_vs_saliency.png
    Same spectral axis, two stacked rows: permutation importance (top) and
    gradient saliency (bottom).  Saliency is the redundancy-robust measure and
    the closer analogue to a per-channel Jacobian / information-content curve.

Run:
    /opt/anaconda3/bin/python3 plot_feature_importance_run004.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / 'feature_importance_run004'
DPI = 400

C_SPECTRAL = '#1F5FC2'   # blue   — reflectance channels
C_GEOMETRY = '#D55E00'   # vermillion — geometry angles
C_SALIENCY = '#117733'   # green  — gradient saliency
C_BAND = '#E6C98C'       # muted warm tan — water-vapor shading (contrasts w/ blue line)

# Major atmospheric water-vapor absorption bands (um) to shade as visual
# anchors for the King & Vaughan comparison.  Centers are well known; widths
# are approximate windows, not retrieval channel definitions.
WV_BANDS_UM = [(0.91, 0.98), (1.10, 1.18), (1.34, 1.44), (1.78, 1.98)]


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.7,
    })


def load():
    d = np.load(OUT_DIR / 'feature_importance_run004.npz', allow_pickle=True)
    return d


def shade_wv_bands(ax, label_first=True):
    for k, (lo, hi) in enumerate(WV_BANDS_UM):
        ax.axvspan(lo, hi, color=C_BAND, alpha=0.18, lw=0,
                   label='H$_2$O vapor band' if (label_first and k == 0) else None)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ranked importance of all 640 inputs
# ─────────────────────────────────────────────────────────────────────────────
def fig1_ranked(d):
    perm = d['perm_mean_um']
    perm_std = d['perm_std_um']
    labels = d['feat_label']
    kind = d['kind']
    n_spec = int(d['n_spectral'])

    order = np.argsort(-perm)              # most -> least important
    is_geom = kind == 'geometry'

    fig, (ax_bar, ax_curve) = plt.subplots(
        1, 2, figsize=(13.5, 7.2),
        gridspec_kw={'width_ratios': [1.0, 1.25], 'wspace': 0.28})

    # (a) Top-30 horizontal bars ------------------------------------------------
    k = 30
    top = order[:k]
    colors = [C_GEOMETRY if is_geom[c] else C_SPECTRAL for c in top]
    y = np.arange(k)[::-1]                  # rank 1 at top
    ax_bar.barh(y, perm[top], xerr=perm_std[top], color=colors,
                edgecolor='none', height=0.78,
                error_kw=dict(ecolor='0.35', elinewidth=0.7, capsize=1.5))
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([labels[c] for c in top], fontsize=7.2)
    ax_bar.set_xlabel('Permutation importance:  $\\Delta$RMSE (μm)')
    ax_bar.set_title('(a)  Top 30 inputs')
    ax_bar.grid(axis='x', alpha=0.3)
    ax_bar.margins(y=0.01)
    handles = [Line2D([0], [0], color=C_SPECTRAL, lw=6, label='reflectance channel'),
               Line2D([0], [0], color=C_GEOMETRY, lw=6, label='geometry angle')]
    ax_bar.legend(handles=handles, loc='lower right', frameon=True)

    # (b) Full ranked curve, all 640 -------------------------------------------
    ranked = perm[order]
    x = np.arange(1, len(order) + 1)
    ax_curve.fill_between(x, 0, ranked, color=C_SPECTRAL, alpha=0.18, lw=0)
    ax_curve.plot(x, ranked, color=C_SPECTRAL, lw=1.0,
                  label='all 640 inputs (ranked)')
    # mark geometry inputs at their ranks; stagger label offsets so the two
    # near-zero azimuths (high rank, overlapping) stay legible.
    rank_pos = {c: r + 1 for r, c in enumerate(order)}
    geom_idx = np.where(is_geom)[0]
    geom_sorted = sorted(geom_idx, key=lambda c: rank_pos[c])  # by rank
    label_offsets = {0: (8, 8), 1: (8, -4), 2: (-46, 14), 3: (-46, -16)}
    for rank_order, gi in enumerate(geom_sorted):
        xp = rank_pos[gi]
        ax_curve.scatter([xp], [perm[gi]], color=C_GEOMETRY, s=45, zorder=5,
                         edgecolors='k', linewidths=0.4)
        dx, dy = label_offsets[rank_order]
        ax_curve.annotate(f'{labels[gi]} (#{xp})', (xp, perm[gi]),
                          textcoords='offset points', xytext=(dx, dy),
                          fontsize=8.5, color=C_GEOMETRY)
    ax_curve.set_xlabel('Importance rank  (1 = most important of 640)')
    ax_curve.set_ylabel('Permutation importance:  $\\Delta$RMSE (μm)')
    ax_curve.set_title('(b)  All 640 inputs, ranked most → least important')
    ax_curve.grid(alpha=0.3)
    ax_curve.set_xlim(-5, len(order) + 5)
    ax_curve.legend(loc='upper right', frameon=True)

    fig.suptitle('Feature importance — paper-3 network (sweep run_004, M0; '
                 'baseline test RMSE = 0.964 μm)', fontsize=12.5, y=0.98)
    fig.savefig(OUT_DIR / 'fig1_feature_importance_ranked_all640.png',
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig1_feature_importance_ranked_all640.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — spectral importance vs wavelength (King & Vaughan comparison)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_spectral(d, ppt: bool = False):
    """Spectral permutation importance vs wavelength.

    ppt=True renders a slide version: larger axis labels, title, tick labels,
    legend, thicker line and bigger markers, 16:9-friendly canvas.
    """
    perm = d['perm_mean_um'][:int(d['n_spectral'])]
    wl_um = d['wavelengths_nm'] / 1000.0

    if ppt:
        fig, ax = plt.subplots(figsize=(13.33, 6.7))
        lw, ms = 1.4, 16
        lbl_fs, tick_fs, leg_fs, title_fs = 22, 18, 16, 23
        out_name, dpi = 'fig2_spectral_importance_vs_wavelength_PPT.png', 220
        title = 'Per-wavelength feature importance — droplet-profile retrieval'
        # place the (small) legend in the empty zone past the UV spike so it
        # sits fully inside and clears the y-axis ticks
        leg_kw = dict(loc='upper left', bbox_to_anchor=(0.27, 0.99), ncol=1)
    else:
        fig, ax = plt.subplots(figsize=(13, 5.0))
        lw, ms = 0.9, 7
        lbl_fs, tick_fs, leg_fs, title_fs = 11, 10, 9, 11.5
        out_name, dpi = 'fig2_spectral_importance_vs_wavelength.png', DPI
        title = ('Per-wavelength feature importance for the droplet-profile '
                 'retrieval (run_004)\n— compare with King & Vaughan (2012) '
                 'Fig. 5 Shannon information content per channel')
        leg_kw = dict(loc='upper center', ncol=2)

    shade_wv_bands(ax)
    # NB: ±1σ band omitted — at 100 shuffles σ < 0.02 μm, thinner than the line.
    ax.plot(wl_um, perm, color=C_SPECTRAL, lw=lw, zorder=3,
            label='permutation importance')
    ax.scatter(wl_um, perm, s=ms, color=C_SPECTRAL, edgecolors='none', zorder=4)
    ax.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax.set_xlabel('Wavelength (μm)', fontsize=lbl_fs)
    ax.set_ylabel('Permutation importance:  $\\Delta$RMSE (μm)', fontsize=lbl_fs)
    ax.set_title(title, fontsize=title_fs, pad=(12 if ppt else 6))
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlim(wl_um.min() - 0.02, wl_um.max() + 0.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=leg_fs, framealpha=0.92, **leg_kw)
    fig.savefig(OUT_DIR / out_name, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_name}')


def fig2b_perm_vs_saliency(d):
    n_spec = int(d['n_spectral'])
    perm = d['perm_mean_um'][:n_spec]
    perm_std = d['perm_std_um'][:n_spec]
    sal = d['saliency'][:n_spec]
    wl_um = d['wavelengths_nm'] / 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)

    shade_wv_bands(ax1)
    ax1.fill_between(wl_um, perm - perm_std, perm + perm_std,
                     color=C_SPECTRAL, alpha=0.20, lw=0)
    ax1.plot(wl_um, perm, color=C_SPECTRAL, lw=0.9)
    ax1.scatter(wl_um, perm, s=6, color=C_SPECTRAL, edgecolors='none')
    ax1.set_ylabel('Permutation\n$\\Delta$RMSE (μm)')
    ax1.set_title('(a)  Permutation importance — single-channel, '
                  'redundancy-suppressed', fontsize=11)
    ax1.grid(alpha=0.25)

    shade_wv_bands(ax2, label_first=True)
    ax2.plot(wl_um, sal, color=C_SALIENCY, lw=0.9)
    ax2.scatter(wl_um, sal, s=6, color=C_SALIENCY, edgecolors='none')
    ax2.set_ylabel('Gradient saliency\n|∂$\\bar r_e$/∂x|')
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_title('(b)  Gradient saliency — local Jacobian magnitude, '
                  'redundancy-robust (closest analogue to information content)',
                  fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.legend(loc='upper center', frameon=True)
    ax2.set_xlim(wl_um.min() - 0.02, wl_um.max() + 0.02)

    fig.suptitle('Spectral feature importance vs. wavelength — two measures '
                 '(run_004)', fontsize=12.5, y=0.97)
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(OUT_DIR / 'fig2b_spectral_perm_vs_saliency.png',
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig2b_spectral_perm_vs_saliency.png')


def main():
    setup_style()
    d = load()
    fig1_ranked(d)
    fig2_spectral(d, ppt=False)
    fig2_spectral(d, ppt=True)
    fig2b_perm_vs_saliency(d)
    print(f'\nAll figures in {OUT_DIR}')


if __name__ == '__main__':
    main()
