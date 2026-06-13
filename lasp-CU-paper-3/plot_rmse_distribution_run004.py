"""
plot_rmse_distribution_run004.py — Distribution of per-profile retrieval RMSE
over the published run_004 test set (889 profiles).

Per-profile RMSE is the same scalar the published scatter figures use
(plot_run004_figures.py: per_sample_rmse = sqrt(mean_over_levels (pred-true)^2)),
read straight from run_004/pred_cache.npz so it matches the paper exactly.

Note on the two "RMSE" numbers
------------------------------
The feature-importance y-axis baseline (0.9641 um) and the *mean* of this
histogram are different aggregations of the same errors:

    feature-importance baseline = mean_levels( sqrt( mean_profiles (err^2) ) )
                                = level-RMSE first, then average over levels
    histogram per-profile RMSE  = sqrt( mean_levels (err^2) )   for each profile
    histogram mean              = mean_profiles( per-profile RMSE )

Because sqrt is concave and the 7 levels have very different error scales,
these do not have to be equal — they are reported side by side below.

Run:
    /opt/anaconda3/bin/python3 plot_rmse_distribution_run004.py
    /opt/anaconda3/bin/python3 plot_rmse_distribution_run004.py --ppt

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

REPO = Path(__file__).resolve().parent
RUN_DIR = (REPO / 'hyper_parameter_sweep'
           / 'sweep_results_profile_only_synthetic_M0_rev2' / 'run_004')
OUT_DIR = REPO / 'feature_importance_run004'
FI_BASELINE_UM = 0.9641   # mean-per-level RMSE (feature-importance baseline)

C_HIST = '#1F5FC2'
C_MEAN = '#CC2A1E'   # red
C_MEDIAN = '#117733'  # green


def style(ppt: bool):
    if ppt:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'axes.labelsize': 22, 'axes.titlesize': 24,
            'xtick.labelsize': 18, 'ytick.labelsize': 18,
            'legend.fontsize': 18, 'axes.linewidth': 1.1,
        })
    else:
        plt.rcParams.update({
            'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
            'mathtext.fontset': 'cm',
            'axes.labelsize': 12, 'axes.titlesize': 13,
            'xtick.labelsize': 10, 'ytick.labelsize': 10,
            'legend.fontsize': 10, 'axes.linewidth': 0.7,
        })


def per_profile_rmse_um() -> np.ndarray:
    c = np.load(RUN_DIR / 'pred_cache.npz')
    pred, true = c['pred'], c['true']                 # (889, 7) um
    return np.sqrt(((pred - true) ** 2).mean(axis=1))  # (889,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppt', action='store_true',
                    help='Large-font sans-serif version for slides.')
    ap.add_argument('--bins', type=int, default=45)
    args = ap.parse_args()

    rmse = per_profile_rmse_um()
    n = rmse.size
    mean_v, median_v = float(rmse.mean()), float(np.median(rmse))
    p05, p95 = np.percentile(rmse, [5, 95])
    sk = float(skew(rmse))

    print(f'Per-profile RMSE over the run_004 test set (N = {n}):')
    print(f'  mean    = {mean_v:.4f} um')
    print(f'  median  = {median_v:.4f} um   (mean - median = {mean_v - median_v:+.4f})')
    print(f'  std     = {rmse.std():.4f} um')
    print(f'  5th/95th pct = {p05:.4f} / {p95:.4f} um')
    print(f'  min/max = {rmse.min():.4f} / {rmse.max():.4f} um')
    print(f'  skewness = {sk:+.3f}  ({"right" if sk > 0 else "left"}-skewed)')
    print(f'  (feature-importance baseline, different aggregation = '
          f'{FI_BASELINE_UM:.4f} um)')

    style(args.ppt)
    figsize = (12.5, 6.0) if args.ppt else (9.5, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(rmse, bins=args.bins, color=C_HIST, alpha=0.8, edgecolor='white',
            linewidth=0.5)
    lw = 2.6 if args.ppt else 1.8
    ax.axvline(mean_v, color=C_MEAN, lw=lw, ls='--',
               label=f'mean = {mean_v:.3f} μm')
    ax.axvline(median_v, color=C_MEDIAN, lw=lw, ls='-.',
               label=f'median = {median_v:.3f} μm')
    ax.set_xlabel('Per-profile RMSE (μm)')
    ax.set_ylabel('Number of test profiles')
    title = ('Distribution of per-profile retrieval RMSE'
             if args.ppt else
             'Distribution of per-profile retrieval RMSE — run_004 test set')
    ax.set_title(title)
    ax.text(0.975, 0.74,
            f'N = {n} profiles\nskewness = {sk:+.2f} (right-skewed)\n'
            f'5–95%: {p05:.2f}–{p95:.2f} μm',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=(16 if args.ppt else 9.5),
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
    ax.legend(loc='upper right', frameon=True,
              bbox_to_anchor=(1.0, 0.97))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    suffix = '_PPT' if args.ppt else ''
    out = OUT_DIR / f'fig3_per_profile_rmse_histogram{suffix}.png'
    fig.savefig(out, dpi=(220 if args.ppt else 400), bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}')


if __name__ == '__main__':
    main()
