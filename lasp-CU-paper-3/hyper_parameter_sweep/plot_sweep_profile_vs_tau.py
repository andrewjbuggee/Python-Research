"""
plot_sweep_profile_vs_tau.py — sweep-level trade-off figure for the
PROFILE + τ_c sweep (sweep_results_profile_tau_synthetic).

Main figure: profile_vs_tau_tradeoff.png (2×2)
    (a) τ RMSE vs profile RMSE, color = log10(tau_weight), marker = transform.
        The joint accuracy picture: is there a Pareto frontier, or do good
        configs win both tasks at once?
    (b) profile RMSE vs tau_weight (log x), split by transform, with binned
        medians.  Where does upweighting τ start to bite the profile task?
    (c) τ RMSE vs tau_weight (log x), same split.  How cheap is τ accuracy?
    (d) The Pareto front of (profile RMSE, τ RMSE), with the top runs labeled.

Optional (--m0-dir): paired_delta_vs_M0.png — per-run Δ profile RMSE
    (PT − M0 profile-only, same run_id = identical shared hyperparameters)
    vs tau_weight, split by transform.
    CAVEAT: the PT sweep uses a true 80/10/10 split (33,593 train / 4,199
    test) while M0_rev2 used 95.8/2.1/2.1 (40,213 train / 889 test), so the
    deltas mix the τ-head effect with a train-size/test-set effect.  The
    figure annotates this; treat the tau_weight TREND as meaningful and the
    absolute offset as confounded.

Usage:
    python plot_sweep_profile_vs_tau.py
    python plot_sweep_profile_vs_tau.py \\
        --sweep-dir sweep_results_profile_tau_synthetic \\
        --m0-dir sweep_results_profile_only_synthetic_M0_rev2

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

DPI = 400

# CVD-safe (Okabe-Ito): one color per τ transform
C_LINEAR = '#0072B2'    # blue
C_LOG10  = '#D55E00'    # vermillion
C_DELTA0 = '0.25'
MARKER   = {'linear': 'o', 'log10': '^'}
COLOR    = {'linear': C_LINEAR, 'log10': C_LOG10}


def setup_style():
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'axes.labelsize':   10,
        'axes.titlesize':   10.5,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'legend.fontsize':  8.5,
        'axes.linewidth':   0.7,
    })


def load_summaries(sweep_dir: Path) -> List[Dict]:
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f'sweep dir not found: {sweep_dir}')
    out = []
    for sub in sorted(sweep_dir.glob('run_*')):
        sj = sub / 'summary.json'
        if sj.is_file():
            with sj.open() as f:
                out.append(json.load(f))
    if not out:
        raise RuntimeError(f'No run_*/summary.json found in {sweep_dir}')
    return out


def extract_arrays(summaries: List[Dict]) -> Dict[str, np.ndarray]:
    """Flat arrays over runs; tau axes come from hyperparams (with the
    summary-level copies as fallback)."""
    def get(s, key, default=np.nan):
        return s.get(key, s['hyperparams'].get(key, default))
    return {
        'run_id':     np.array([s['run_id'] for s in summaries]),
        'prof_rmse':  np.array([s['mean_test_rmse_um'] for s in summaries]),
        'tau_rmse':   np.array([s['tau_test_rmse'] for s in summaries]),
        'tau_weight': np.array([float(get(s, 'tau_weight', 1.0))
                                for s in summaries]),
        'transform':  np.array([str(get(s, 'tau_transform', 'linear'))
                                for s in summaries]),
    }


def binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 7):
    """Median of y in log-spaced x bins (≥3 points per bin). Returns
    (bin_center, median) arrays."""
    lo, hi = np.log10(x.min()), np.log10(x.max())
    edges = np.logspace(lo, hi, n_bins + 1)
    cx, cy = [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1
                               else x <= edges[i + 1])
        if m.sum() >= 3:
            cx.append(np.sqrt(edges[i] * edges[i + 1]))   # geometric center
            cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


def pareto_front(prof: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated (minimize both) runs, sorted by prof."""
    order = np.argsort(prof)
    front, best_tau = [], np.inf
    for i in order:
        if tau[i] < best_tau:
            front.append(i)
            best_tau = tau[i]
    return np.array(front)


# ─────────────────────────────────────────────────────────────────────────────
# Main 2×2 figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_tradeoff(a: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    transforms = ['linear', 'log10']

    # (a) τ RMSE vs profile RMSE, color = log10(tau_weight)
    ax = axes[0, 0]
    vmin, vmax = np.log10(a['tau_weight'].min()), np.log10(a['tau_weight'].max())
    sc = None
    for tf in transforms:
        m = a['transform'] == tf
        sc = ax.scatter(a['prof_rmse'][m], a['tau_rmse'][m],
                        c=np.log10(a['tau_weight'][m]), vmin=vmin, vmax=vmax,
                        cmap='viridis', marker=MARKER[tf], s=42,
                        edgecolors='k', linewidths=0.4,
                        label=f'{tf} ({m.sum()})')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10}$(tau_weight)')
    ax.set_xlabel('mean profile RMSE (μm)')
    ax.set_ylabel(r'$\tau_c$ RMSE')
    ax.set_title('(a)  joint accuracy across the sweep')
    ax.legend(frameon=True, title='τ transform (marker)')
    ax.grid(alpha=0.3)

    # (b) profile RMSE vs tau_weight
    ax = axes[0, 1]
    for tf in transforms:
        m = a['transform'] == tf
        ax.scatter(a['tau_weight'][m], a['prof_rmse'][m], s=26, alpha=0.65,
                   color=COLOR[tf], marker=MARKER[tf], edgecolors='none',
                   label=tf)
        bx, by = binned_median(a['tau_weight'][m], a['prof_rmse'][m])
        if len(bx):
            ax.plot(bx, by, color=COLOR[tf], lw=1.8, ls='-',
                    label=f'{tf} binned median')
    ax.set_xscale('log')
    ax.set_xlabel('tau_weight  (τ NLL weight vs profile NLL)')
    ax.set_ylabel('mean profile RMSE (μm)')
    ax.set_title('(b)  does the τ task tax the profile task?')
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, which='both')

    # (c) τ RMSE vs tau_weight
    ax = axes[1, 0]
    for tf in transforms:
        m = a['transform'] == tf
        ax.scatter(a['tau_weight'][m], a['tau_rmse'][m], s=26, alpha=0.65,
                   color=COLOR[tf], marker=MARKER[tf], edgecolors='none',
                   label=tf)
        bx, by = binned_median(a['tau_weight'][m], a['tau_rmse'][m])
        if len(bx):
            ax.plot(bx, by, color=COLOR[tf], lw=1.8, ls='-',
                    label=f'{tf} binned median')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('tau_weight  (τ NLL weight vs profile NLL)')
    ax.set_ylabel(r'$\tau_c$ RMSE')
    ax.set_title(r'(c)  how much weight does $\tau_c$ accuracy need?')
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, which='both')

    # (d) Pareto front
    ax = axes[1, 1]
    for tf in transforms:
        m = a['transform'] == tf
        ax.scatter(a['prof_rmse'][m], a['tau_rmse'][m], s=22, alpha=0.4,
                   color=COLOR[tf], marker=MARKER[tf], edgecolors='none',
                   label=tf)
    front = pareto_front(a['prof_rmse'], a['tau_rmse'])
    fs = front[np.argsort(a['prof_rmse'][front])]
    ax.plot(a['prof_rmse'][fs], a['tau_rmse'][fs], color=C_DELTA0, lw=1.2,
            ls='--', zorder=4, label='Pareto front')
    for i in fs:
        ax.scatter(a['prof_rmse'][i], a['tau_rmse'][i], s=60, facecolors='none',
                   edgecolors='k', linewidths=0.9, zorder=5)
        ax.annotate(f"{a['run_id'][i]}"
                    f" (w={a['tau_weight'][i]:.2f})",
                    (a['prof_rmse'][i], a['tau_rmse'][i]),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel('mean profile RMSE (μm)')
    ax.set_ylabel(r'$\tau_c$ RMSE')
    ax.set_title('(d)  non-dominated runs  (label: run_id, tau_weight)')
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Profile ↔ τ$_c$ trade-off — profile+τ sweep '
                 f'({len(a["run_id"])} runs)', fontsize=12.5, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'trade-off figure -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Optional: paired Δ vs the profile-only M0 sweep
# ─────────────────────────────────────────────────────────────────────────────
def plot_paired_delta(a: Dict[str, np.ndarray], m0_summaries: List[Dict],
                      out_path: Path) -> None:
    m0_by_id = {s['run_id']: s['mean_test_rmse_um'] for s in m0_summaries}
    shared = np.array([i for i, rid in enumerate(a['run_id'])
                       if rid in m0_by_id])
    if len(shared) == 0:
        print('No shared run_ids with the M0 dir — skipping paired-delta figure.')
        return
    delta = np.array([a['prof_rmse'][i] - m0_by_id[a['run_id'][i]]
                      for i in shared])
    tw    = a['tau_weight'][shared]
    tf_ar = a['transform'][shared]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for tf in ('linear', 'log10'):
        m = tf_ar == tf
        if not m.any():
            continue
        ax.scatter(tw[m], delta[m], s=30, alpha=0.7, color=COLOR[tf],
                   marker=MARKER[tf], edgecolors='none', label=tf)
        bx, by = binned_median(tw[m], delta[m])
        if len(bx):
            ax.plot(bx, by, color=COLOR[tf], lw=1.8,
                    label=f'{tf} binned median')
    ax.axhline(0, color=C_DELTA0, lw=1.0, ls='--',
               label='no change vs profile-only')
    ax.set_xscale('log')
    ax.set_xlabel('tau_weight  (τ NLL weight vs profile NLL)')
    ax.set_ylabel(r'$\Delta$ mean profile RMSE (μm)'
                  '\n(profile+τ  −  profile-only M0, same hyperparameters)')
    ax.set_title(f'Cost of the τ head on profile retrieval '
                 f'({len(shared)} paired runs)')
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, which='both')
    ax.text(0.02, 0.02,
            'Caveat: PT trained on 33,593 samples / tested on 4,199 profiles;\n'
            'M0_rev2 trained on 40,213 / tested on 889.  Deltas mix the τ-head\n'
            'effect with this split difference — read the trend, not the offset.',
            transform=ax.transAxes, fontsize=7.5, va='bottom', color='0.35')
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'paired-delta figure -> {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--sweep-dir', default='sweep_results_profile_tau_synthetic',
                   help='PT sweep results directory (default: '
                        'sweep_results_profile_tau_synthetic).')
    p.add_argument('--m0-dir', default=None,
                   help='Optional profile-only sweep dir (e.g. '
                        'sweep_results_profile_only_synthetic_M0_rev2) for the '
                        'paired-delta figure.')
    args = p.parse_args()
    setup_style()

    sweep_dir = Path(args.sweep_dir).resolve()
    summaries = load_summaries(sweep_dir)
    a = extract_arrays(summaries)
    print(f'Loaded {len(summaries)} runs from {sweep_dir}')
    print(f"  profile RMSE: {a['prof_rmse'].min():.3f} – "
          f"{a['prof_rmse'].max():.3f} μm "
          f"(median {np.median(a['prof_rmse']):.3f})")
    print(f"  τ RMSE      : {a['tau_rmse'].min():.3f} – "
          f"{a['tau_rmse'].max():.3f} "
          f"(median {np.median(a['tau_rmse']):.3f})")

    fig_dir = sweep_dir / 'Figures'
    fig_dir.mkdir(exist_ok=True)
    plot_tradeoff(a, fig_dir / 'profile_vs_tau_tradeoff.png')

    if args.m0_dir:
        m0 = load_summaries(Path(args.m0_dir).resolve())
        plot_paired_delta(a, m0, fig_dir / 'paired_delta_vs_M0.png')


if __name__ == '__main__':
    main()
