"""
plot_permutation_convergence_run004.py — How many shuffles does permutation
importance need?  Shows the running (cumulative-mean) importance estimate and
its standard error vs the number of shuffles, for a few representative inputs.

Why so few shuffles suffice
---------------------------
Each shuffle already evaluates the metric over all 889 test profiles, so a
single shuffle is itself a low-variance estimate.  Extra shuffles only average
out the residual Monte-Carlo noise of *which* random permutation was drawn, and
that noise falls like 1/sqrt(n_shuffles).  The plotted estimates are flat well
before 20 shuffles; the standard error at 20 is already << the signal.

Run:
    /opt/anaconda3/bin/python3 plot_permutation_convergence_run004.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_feature_importance_run004 import (
    pick_device, load_model, build_test_inputs, mean_per_level_rmse, OUT_DIR)

N_MAX = 150
SEED = 42
DPI = 400

# representative channels: idx -> (label, color)
CHANNELS = {
    0:   ('352.0 nm  (UV edge, rank 1)',   '#882255'),
    532: ('1981.6 nm  (2.0 μm peak)',      '#CC2A1E'),
    374: ('1497.6 nm  (1.5 μm peak)',      '#E69F00'),
    637: ('VZA  (geometry, rank ~112)',    '#0072B2'),
    639: ('VAZ  (azimuth, ≈ 0)',           '#117733'),
}


def main():
    device = pick_device(None)
    model = load_model(device)
    X, true_phys, _ = build_test_inputs()
    baseline = mean_per_level_rmse(model, X, true_phys, device)
    print(f'device {device}, baseline {baseline:.4f} μm, seed {SEED}, '
          f'N_MAX {N_MAX}')

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    Xw = X.copy()
    # per-repeat ΔRMSE for each tracked channel
    deltas = {c: np.zeros(N_MAX) for c in CHANNELS}
    for rep in range(N_MAX):
        for c in CHANNELS:
            col = Xw[:, c].copy()
            Xw[:, c] = col[rng.permutation(n)]
            deltas[c][rep] = mean_per_level_rmse(model, Xw, true_phys, device) - baseline
            Xw[:, c] = col

    reps = np.arange(1, N_MAX + 1)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'cm'})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    for c, (label, color) in CHANNELS.items():
        d = deltas[c]
        cum_mean = np.cumsum(d) / reps
        # cumulative standard error of the mean
        cum_sq = np.cumsum(d ** 2) / reps
        cum_std = np.sqrt(np.maximum(cum_sq - cum_mean ** 2, 0.0))
        sem = cum_std / np.sqrt(reps)
        axL.plot(reps, cum_mean, color=color, lw=1.6, label=label)
        axL.fill_between(reps, cum_mean - sem, cum_mean + sem,
                         color=color, alpha=0.18, lw=0)
        axR.plot(reps, sem, color=color, lw=1.6, label=label)

    for ax in (axL, axR):
        ax.axvline(20, color='0.4', ls=':', lw=1.2)
        ax.annotate('20 shuffles\n(original)', (20, ax.get_ylim()[1]),
                    xytext=(24, 0.86), textcoords=('data', 'axes fraction'),
                    fontsize=8.5, color='0.35')
        ax.grid(alpha=0.3)
        ax.set_xlabel('Number of shuffles')
        ax.set_xlim(1, N_MAX)

    axL.set_ylabel('Running importance estimate:  $\\Delta$RMSE (μm)')
    axL.set_title('(a)  Cumulative-mean importance ±1 SEM', fontsize=11)
    axL.legend(loc='center right', fontsize=8.2, frameon=True)

    axR.set_ylabel('Standard error of the mean (μm)')
    axR.set_title('(b)  Monte-Carlo noise falls like 1/√(n_shuffles)',
                  fontsize=11)
    axR.set_yscale('log')

    fig.suptitle('Permutation-importance convergence vs number of shuffles '
                 '(run_004)', fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / 'fig4_permutation_convergence.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out.name}')

    # numeric summary
    print('\nFinal estimate and SEM at 20 vs 150 shuffles:')
    print(f'{"channel":>30} {"mean@150":>10} {"SEM@20":>9} {"SEM@150":>9}')
    for c, (label, _) in CHANNELS.items():
        d = deltas[c]
        m150 = d.mean()
        sem20 = d[:20].std() / np.sqrt(20)
        sem150 = d.std() / np.sqrt(150)
        print(f'{label:>30} {m150:>10.4f} {sem20:>9.4f} {sem150:>9.4f}')


if __name__ == '__main__':
    main()
