"""
plot_feature_importance_profile_tau.py — feature-importance figures for a
PROFILE + τ_c sweep run.  Reads the NPZ written by
compute_feature_importance_profile_tau.py.

Figures (written next to the NPZ)
---------------------------------
fig1_feature_importance_ranked_profile.png
    Same layout as the published run_004 fig1: (a) top-30 horizontal bars,
    (b) full-640 ranked curve with geometry inputs marked — for the PROFILE
    permutation importance.

fig1b_feature_importance_ranked_tau.png
    Identical layout for the τ_c permutation importance.

fig2_spectral_importance_profile_vs_tau.png
    Two stacked rows sharing the wavelength axis: profile ΔRMSE (μm) on top,
    τ_c ΔRMSE below, water-vapor bands shaded — the direct successor of the
    published fig2 (King & Vaughan-style per-channel view), now showing
    whether the τ head draws on different spectral regions than the profile
    head.

fig2b_saliency_profile_vs_tau.png
    Same two-row layout for gradient saliency (redundancy-robust measure).

Run:
    python plot_feature_importance_profile_tau.py \\
        --run-dir hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DPI = 400

# Same palette as plot_feature_importance_run004.py, extended for τ
C_PROFILE  = '#1F5FC2'   # blue        — profile importance
C_TAU      = '#D55E00'   # vermillion  — τ importance
C_GEOMETRY = '#009E73'   # bluish green — geometry angles in bar charts
C_SALIENCY = '#117733'   # green       — saliency accent
C_BAND     = '#E6C98C'   # warm tan    — water-vapor band shading

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


def shade_wv_bands(ax, label_first=True):
    for k, (lo, hi) in enumerate(WV_BANDS_UM):
        ax.axvspan(lo, hi, color=C_BAND, alpha=0.18, lw=0,
                   label='H$_2$O vapor band' if (label_first and k == 0) else None)


# ─────────────────────────────────────────────────────────────────────────────
# Ranked figure — one target at a time (run_004 fig1 layout)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ranked(d, target: str, out_dir: Path):
    """target ∈ {'profile', 'tau'}: pick arrays, units, filename."""
    if target == 'profile':
        perm     = d['perm_profile_mean_um']
        perm_std = d['perm_profile_std_um']
        unit     = 'μm'
        base     = float(d['baseline_profile_rmse_um'])
        head     = (f'Profile head — baseline test RMSE = {base:.3f} μm')
        out_name = 'fig1_feature_importance_ranked_profile.png'
        c_main   = C_PROFILE
    else:
        perm     = d['perm_tau_mean']
        perm_std = d['perm_tau_std']
        unit     = r'$\tau$ units'
        base     = float(d['baseline_tau_rmse'])
        head     = (fr'$\tau_c$ head — baseline test RMSE = {base:.3f}')
        out_name = 'fig1b_feature_importance_ranked_tau.png'
        c_main   = C_TAU

    labels = d['feat_label']
    kind   = d['kind']
    order  = np.argsort(-perm)
    is_geom = kind == 'geometry'

    fig, (ax_bar, ax_curve) = plt.subplots(
        1, 2, figsize=(13.5, 7.2),
        gridspec_kw={'width_ratios': [1.0, 1.25], 'wspace': 0.28})

    # (a) Top-30 horizontal bars
    k = 30
    top = order[:k]
    colors = [C_GEOMETRY if is_geom[c] else c_main for c in top]
    y = np.arange(k)[::-1]
    ax_bar.barh(y, perm[top], xerr=perm_std[top], color=colors,
                edgecolor='none', height=0.78,
                error_kw=dict(ecolor='0.35', elinewidth=0.7, capsize=1.5))
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([labels[c] for c in top], fontsize=7.2)
    ax_bar.set_xlabel(f'Permutation importance:  $\\Delta$RMSE ({unit})')
    ax_bar.set_title('(a)  Top 30 inputs')
    ax_bar.grid(axis='x', alpha=0.3)
    ax_bar.margins(y=0.01)
    handles = [Line2D([0], [0], color=c_main, lw=6, label='reflectance channel'),
               Line2D([0], [0], color=C_GEOMETRY, lw=6, label='geometry angle')]
    ax_bar.legend(handles=handles, loc='lower right', frameon=True)

    # (b) Full ranked curve, all 640
    ranked = perm[order]
    x = np.arange(1, len(order) + 1)
    ax_curve.fill_between(x, 0, ranked, color=c_main, alpha=0.18, lw=0)
    ax_curve.plot(x, ranked, color=c_main, lw=1.0,
                  label='all 640 inputs (ranked)')
    rank_pos = {c: r + 1 for r, c in enumerate(order)}
    geom_idx = np.where(is_geom)[0]
    geom_sorted = sorted(geom_idx, key=lambda c: rank_pos[c])
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
    ax_curve.set_ylabel(f'Permutation importance:  $\\Delta$RMSE ({unit})')
    ax_curve.set_title('(b)  All 640 inputs, ranked most → least important')
    ax_curve.grid(alpha=0.3)
    ax_curve.set_xlim(-5, len(order) + 5)
    ax_curve.legend(loc='upper right', frameon=True)

    fig.suptitle(f'Feature importance — profile+τ network '
                 f'(run_{int(d["run_id"]):03d}).  {head}',
                 fontsize=12.5, y=0.98)
    fig.savefig(out_dir / out_name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_name}')


# ─────────────────────────────────────────────────────────────────────────────
# Spectral importance vs wavelength — profile row over τ row
# ─────────────────────────────────────────────────────────────────────────────
def fig2_spectral_two_targets(d, out_dir: Path):
    n_spec = int(d['n_spectral'])
    wl_um  = d['wavelengths_nm'] / 1000.0
    perm_p = d['perm_profile_mean_um'][:n_spec]
    perm_t = d['perm_tau_mean'][:n_spec]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)

    shade_wv_bands(ax1, label_first=False)
    ax1.plot(wl_um, perm_p, color=C_PROFILE, lw=0.9)
    ax1.scatter(wl_um, perm_p, s=6, color=C_PROFILE, edgecolors='none')
    ax1.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax1.set_ylabel('Profile permutation\n$\\Delta$RMSE (μm)')
    ax1.set_title('(a)  Profile head — which channels the 7-level $r_e$ '
                  'retrieval relies on', fontsize=11)
    ax1.grid(alpha=0.25)

    shade_wv_bands(ax2, label_first=True)
    ax2.plot(wl_um, perm_t, color=C_TAU, lw=0.9)
    ax2.scatter(wl_um, perm_t, s=6, color=C_TAU, edgecolors='none')
    ax2.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax2.set_ylabel('$\\tau_c$ permutation\n$\\Delta$RMSE ($\\tau$ units)')
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_title(r'(b)  $\tau_c$ head — which channels the optical-thickness '
                  'retrieval relies on', fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.legend(loc='upper center', frameon=True)
    ax2.set_xlim(wl_um.min() - 0.02, wl_um.max() + 0.02)

    rho = (np.corrcoef(perm_p, perm_t)[0, 1]
           if perm_p.std() > 0 and perm_t.std() > 0 else float('nan'))
    fig.suptitle(f'Spectral permutation importance by retrieval target — '
                 f'run_{int(d["run_id"]):03d}   '
                 f'(corr between rows: r = {rho:+.3f})',
                 fontsize=12.5, y=0.97)
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(out_dir / 'fig2_spectral_importance_profile_vs_tau.png',
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig2_spectral_importance_profile_vs_tau.png')


def fig2b_saliency_two_targets(d, out_dir: Path):
    n_spec = int(d['n_spectral'])
    wl_um  = d['wavelengths_nm'] / 1000.0
    sal_p  = d['saliency_profile'][:n_spec]
    sal_t  = d['saliency_tau'][:n_spec]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)

    shade_wv_bands(ax1, label_first=False)
    ax1.plot(wl_um, sal_p, color=C_PROFILE, lw=0.9)
    ax1.scatter(wl_um, sal_p, s=6, color=C_PROFILE, edgecolors='none')
    ax1.set_ylabel('Profile saliency\n|∂$\\bar r_e$/∂x|  (μm)')
    ax1.set_title('(a)  Profile head — local Jacobian magnitude, '
                  'redundancy-robust', fontsize=11)
    ax1.grid(alpha=0.25)

    shade_wv_bands(ax2, label_first=True)
    ax2.plot(wl_um, sal_t, color=C_TAU, lw=0.9)
    ax2.scatter(wl_um, sal_t, s=6, color=C_TAU, edgecolors='none')
    ax2.set_ylabel('$\\tau_c$ saliency\n|∂$\\tau_c$/∂x|')
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_title(r'(b)  $\tau_c$ head — local Jacobian magnitude', fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.legend(loc='upper center', frameon=True)
    ax2.set_xlim(wl_um.min() - 0.02, wl_um.max() + 0.02)

    fig.suptitle(f'Gradient saliency by retrieval target — '
                 f'run_{int(d["run_id"]):03d}', fontsize=12.5, y=0.97)
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(out_dir / 'fig2b_saliency_profile_vs_tau.png',
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig2b_saliency_profile_vs_tau.png')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-dir', default=None,
                   help='PT-sweep run directory; reads '
                        '<run-dir>/feature_importance/feature_importance_profile_tau.npz')
    p.add_argument('--npz', default=None,
                   help='Explicit path to the NPZ (overrides --run-dir).')
    return p.parse_args()


def main():
    args = parse_args()
    if args.npz:
        npz_path = Path(args.npz).resolve()
    elif args.run_dir:
        npz_path = (Path(args.run_dir).resolve() / 'feature_importance'
                    / 'feature_importance_profile_tau.npz')
    else:
        raise SystemExit('Pass --run-dir or --npz.')
    if not npz_path.exists():
        raise FileNotFoundError(
            f'{npz_path} not found — run compute_feature_importance_profile_tau.py first.')

    out_dir = npz_path.parent
    setup_style()
    d = np.load(npz_path, allow_pickle=True)
    fig_ranked(d, 'profile', out_dir)
    fig_ranked(d, 'tau', out_dir)
    fig2_spectral_two_targets(d, out_dir)
    fig2b_saliency_two_targets(d, out_dir)
    print(f'\nAll figures in {out_dir}')


if __name__ == '__main__':
    main()
