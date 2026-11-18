"""
13b_compare_log_std_clamp.py — STEP 13b of the paper-3 pipeline
===============================================================
Compare the arms of a log-σ clamp test (12c_log_std_clamp_configs.py) on the
VALIDATION set: the legacy clamp (log_std_min = −4) against a lowered one.

Reads training_runs/<tag>/results/run_NNN/{summary.json, val_pred_cache.npz}
(the trainer writes val_pred_cache.npz since 17 Nov 2026). Runs are grouped
by hyperparams['log_std_min']; within an arm, seeds differ only in init seed.

Per level and per arm (mean over seeds, with the min–max range over seeds):
  1. r_e RMSE (μm)                — did the mean fit change? Watch levels 1 and
                                    7: with a lower floor, mid-cloud points can
                                    take more of the inverse-variance weight of
                                    the NLL (Seitzer et al. 2022, ICLR, Sec. 3.2,
                                    https://arxiv.org/abs/2203.09168).
  2. share of σ on the floor      — σ_norm == exp(log_std_min) (1e-4 rel. tol.);
                                    ≈ 0 means the clamp no longer decides σ.
  3. std(z), z = (pred − true)/σ  — 1 for calibrated σ; < 1 → σ too large,
                                    > 1 → σ too small (overconfident).
  4. coverage of |z| ≤ 1.96       — 0.95 for calibrated Gaussian σ.
  5. mean predicted σ (μm)        — next to the RMSE it should approach.
and the same for τ_c (RMSE, share on floor, std(z), 95 % coverage).

Decision rule (my choice, not from a source): treat an arm difference as real
only if it exceeds the seed-to-seed range of both arms.

Outputs (in training_runs/<tag>/)
  log_std_clamp_comparison.png      2 x 3 panels of the per-level metrics
  log_std_clamp_comparison.json     every number in the table

Usage
    python 13b_compare_log_std_clamp.py --tag LSC_2026-11-17

Nothing here is stochastic.
Author: Andrew J. Buggee, 17 Nov 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from plot_per_level_error_percentiles import setup_style  # noqa: E402

LEGACY_LOG_STD_MIN = -4.0
Z95 = 1.96                     # standard-normal 97.5 % quantile
# Okabe-Ito (Wong 2011, Nat. Methods 8, 441): control grey-black, test blue.
ARM_COLORS = ['#000000', '#0072B2', '#D55E00', '#009E73']


def run_metrics(npz: dict, log_std_min: float) -> dict:
    """Validation metrics of one run from its val_pred_cache.npz arrays.

    The same quantities the trainer's uncertainty_diagnostics() writes into
    summary.json; recomputed here so older caches and new metrics work alike.
    """
    floor = float(np.exp(log_std_min))
    err = npz['pred'] - npz['true']                         # (n, L) μm
    z = err / npz['pred_std']                               # (n, L)
    tau_err = npz['tau_pred'] - npz['tau_true']             # (n,)
    tau_z = tau_err / npz['tau_pred_std']                   # (n,)
    return {
        'rmse_um':        np.sqrt((err ** 2).mean(axis=0)),                         # (L,)
        'on_floor':       np.isclose(npz['pred_std_norm'], floor, rtol=1e-4, atol=0).mean(axis=0),
        'z_std':          z.std(axis=0),
        'cov95':          (np.abs(z) <= Z95).mean(axis=0),
        'sigma_mean_um':  npz['pred_std'].mean(axis=0),
        'tau_rmse':       np.sqrt((tau_err ** 2).mean()),
        'tau_on_floor':   np.isclose(npz['tau_pred_std_norm'], floor, rtol=1e-4, atol=0).mean(),
        'tau_z_std':      tau_z.std(),
        'tau_cov95':      (np.abs(tau_z) <= Z95).mean(),
    }


def load_arms(tag: str) -> dict:
    """{log_std_min: [ (run_id, init_seed, metrics), ... ]} for every finished run."""
    arms: dict = {}
    split_idx = None
    for rd in sorted((REPO / 'training_runs' / tag / 'results').glob('run_*')):
        sp, cp = rd / 'summary.json', rd / 'val_pred_cache.npz'
        if not (sp.exists() and cp.exists()):
            print(f'  skip {rd.name}: summary.json or val_pred_cache.npz missing')
            continue
        s = json.load(open(sp))
        lsm = float(s['hyperparams'].get('log_std_min', LEGACY_LOG_STD_MIN))
        c = np.load(cp)
        # All runs must be scored on the same validation clouds.
        if split_idx is None:
            split_idx = np.sort(c['h5_idx'])
        elif not np.array_equal(split_idx, np.sort(c['h5_idx'])):
            raise SystemExit(f'{rd.name}: different validation set from the other runs')
        arms.setdefault(lsm, []).append(
            (int(s['run_id']), int(s.get('init_seed', s.get('seed', -1))),
             run_metrics({k: c[k] for k in c.files}, lsm)))
    if len(arms) < 2:
        raise SystemExit(f'need runs from at least two log_std_min values, found {sorted(arms)}')
    return arms


def summarize(runs: list) -> dict:
    """Mean and min/max over the seed replicates of one arm, for every metric."""
    keys = runs[0][2].keys()
    out = {}
    for k in keys:
        a = np.array([r[2][k] for r in runs], dtype=float)   # (n_seeds, ...)
        out[k] = {'mean': a.mean(axis=0), 'min': a.min(axis=0), 'max': a.max(axis=0)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='Compare log-σ clamp arms on validation')
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    out_dir = REPO / 'training_runs' / args.tag

    arms = load_arms(args.tag)
    lsms = sorted(arms, reverse=True)          # −4 (control) first
    stats = {lsm: summarize(arms[lsm]) for lsm in lsms}
    n_lev = len(stats[lsms[0]]['rmse_um']['mean'])
    levels = np.arange(1, n_lev + 1)

    # ── table ────────────────────────────────────────────────────────────
    def fmt(st, k, i=None, f='{:.3f}'):
        m, lo, hi = (st[k][q] if i is None else st[k][q][i] for q in ('mean', 'min', 'max'))
        return f.format(m) + (f' [{f.format(lo)}–{f.format(hi)}]' if lo != hi else '')

    for lsm in lsms:
        seeds = ', '.join(f'run {r[0]} (seed {r[1]})' for r in arms[lsm])
        print(f'\nlog_std_min = {lsm:g}  ({len(arms[lsm])} seeds: {seeds})')
        print(f'  {"lev":>3}  {"val RMSE μm":>22}  {"on floor":>20}  {"std(z)":>20}  '
              f'{"cov |z|≤1.96":>20}  {"mean σ μm":>20}')
        st = stats[lsm]
        for i, L in enumerate(levels):
            print(f'  {L:>3}  {fmt(st, "rmse_um", i):>22}  {fmt(st, "on_floor", i, "{:.2f}"):>20}  '
                  f'{fmt(st, "z_std", i, "{:.2f}"):>20}  {fmt(st, "cov95", i, "{:.2f}"):>20}  '
                  f'{fmt(st, "sigma_mean_um", i):>20}')
        print(f'  τ_c: RMSE {fmt(st, "tau_rmse")}, on floor {fmt(st, "tau_on_floor", None, "{:.2f}")}, '
              f'std(z) {fmt(st, "tau_z_std", None, "{:.2f}")}, '
              f'cov |z|≤1.96 {fmt(st, "tau_cov95", None, "{:.2f}")}')

    # ── figure: 2 x 3 per-level panels ───────────────────────────────────
    setup_style()
    panels = [('rmse_um', r'Validation $r_e$ RMSE ($\mu$m)', None),
              ('sigma_mean_um', r'Mean predicted $\sigma$ ($\mu$m)', None),
              ('on_floor', r'Share of $\sigma$ on the clamp floor', None),
              ('z_std', r'std($z$), $z$ = error / $\sigma$', 1.0),
              ('cov95', r'Share with $|z| \leq 1.96$', 0.95)]
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 5.0), sharey=True)
    axs = axs.ravel()
    for ax, (k, lab, ref) in zip(axs, panels):
        for j, lsm in enumerate(lsms):
            st = stats[lsm][k]
            ax.fill_betweenx(levels, st['min'], st['max'], color=ARM_COLORS[j], alpha=0.2, lw=0)
            ax.plot(st['mean'], levels, 'o-', color=ARM_COLORS[j], ms=3.5, lw=1.2,
                    label=f'log-σ min = {lsm:g}' + (' (control)' if lsm == LEGACY_LOG_STD_MIN else ''))
        if ref is not None:
            ax.axvline(ref, color='0.5', ls=':', lw=0.8)
        if k == 'on_floor':
            ax.set_xlim(-0.02, 1.02)          # a share: always the full 0–1 range
        ax.set_xlabel(lab)
        ax.grid(True, ls='--', lw=0.4, alpha=0.5)
    for ax in axs[::3]:
        ax.set_ylabel('Vertical level (1 = top)')
    axs[0].set_yticks(levels)
    axs[0].invert_yaxis()
    axs[0].legend(loc='best', fontsize=6.5)
    # last panel: τ_c summary as text
    ax = axs[5]
    ax.axis('off')
    lines = ['τ_c (validation)']
    for lsm in lsms:
        st = stats[lsm]
        lines.append(f'log-σ min {lsm:g}:\n  RMSE {st["tau_rmse"]["mean"]:.3f}, '
                     f'on floor {st["tau_on_floor"]["mean"]:.2f}\n'
                     f'  std(z) {st["tau_z_std"]["mean"]:.2f}, '
                     f'cov95 {st["tau_cov95"]["mean"]:.2f}')
    ax.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', fontsize=7, transform=ax.transAxes)
    fig.tight_layout()
    fig_path = out_dir / 'log_std_clamp_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    def jsonable(d):
        return {k: {q: np.asarray(v).tolist() for q, v in st.items()} for k, st in d.items()}
    out = {'tag': args.tag, 'split': 'validation',
           'arms': {f'{lsm:g}': {'runs': [{'run_id': r[0], 'init_seed': r[1]} for r in arms[lsm]],
                                 'metrics': jsonable(stats[lsm])} for lsm in lsms}}
    js = out_dir / 'log_std_clamp_comparison.json'
    json.dump(out, open(js, 'w'), indent=2)
    print(f'\nFigure  -> {fig_path}\nNumbers -> {js}')


if __name__ == '__main__':
    main()
