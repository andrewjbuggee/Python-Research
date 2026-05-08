"""
compare_sweep_profile_only_synthetic.py — rank + visualize a synthetic-cloud
profile-only sweep run by sweep_train_profile_only_synthetic.py.

Sibling of compare_sweep_profile_only.py. The existing K-fold comparator
expects per-fold aggregates (`test_mean_rmse_mean ± test_mean_rmse_std`); the
synthetic sweep is single-split so each `summary.json` carries one value
(`mean_test_rmse_um`) plus a list (`per_level_rmse_um`). This script knows
that schema.

Usage:
    python compare_sweep_profile_only_synthetic.py \\
        --sweep-dir sweep_results_profile_only_synthetic_M0

    # multi-variant comparison
    python compare_sweep_profile_only_synthetic.py \\
        --sweep-dir sweep_results_profile_only_synthetic_M0 \\
                    sweep_results_profile_only_synthetic_M1 \\
                    sweep_results_profile_only_synthetic_M2

Outputs (per sweep dir, written under <sweep_dir>/):
    comparison.csv       — flat table, one row per run, sorted by RMSE
    comparison.json      — same data as JSON for downstream tools
    Figures/             — plots (rank curve, per-level RMSE, hp scatters)

When invoked with multiple sweep dirs an extra figure
`Figures/cross_variant_comparison.png` (in the FIRST dir) compares them.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_summaries(sweep_dir: Path) -> List[Dict]:
    """Read every run_*/summary.json in sweep_dir."""
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f'sweep_dir does not exist: {sweep_dir}')
    out = []
    for sub in sorted(sweep_dir.glob('run_*')):
        sj = sub / 'summary.json'
        if not sj.is_file():
            continue
        with sj.open() as f:
            d = json.load(f)
        # Stamp the source dir for cross-variant labeling
        d['_sweep_dir'] = sweep_dir.name
        out.append(d)
    return out


def _arch_str(hidden_dims) -> str:
    dims = list(hidden_dims)
    if all(h == dims[0] for h in dims):
        return f'{len(dims)}x{dims[0]}'
    return '×'.join(str(h) for h in dims)


def _estimate_n_params(hidden_dims, n_input=643, n_output=7) -> int:
    """Crude param count for the encoder MLP + dual heads (μ and σ).
    Mirrors compare_sweep_profile_only._estimate_n_params, just with the
    synthetic-dataset defaults (643 inputs, 7 outputs)."""
    dims = [n_input] + list(hidden_dims)
    n = sum(dims[i] * dims[i + 1] + dims[i + 1]
            for i in range(len(dims) - 1))
    n += 2 * (dims[-1] * n_output + n_output)   # μ-head + σ-head
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Ranking + table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(summaries: List[Dict], top_n: int = 20):
    """Print a compact ranked table (top_n by RMSE)."""
    sorted_runs = sorted(summaries, key=lambda s: s['mean_test_rmse_um'])

    print(f"\n{'=' * 92}")
    print(f"  Synthetic sweep ranking — {len(summaries)} runs, "
          f"showing top {min(top_n, len(summaries))}")
    print(f"{'=' * 92}")
    print(f"{'rk':>3}  {'run':>5}  {'rmse μm':>7}  {'σ μm':>6}  {'r/σ':>5}  "
          f"{'arch':>14} {'act':>5} {'lr':>9} {'drop':>5} {'lw':>8} "
          f"{'epoch★':>6} {'sec':>5}")
    print('-' * 92)
    for rank, s in enumerate(sorted_runs[:top_n], start=1):
        hp = s['hyperparams']
        print(f"{rank:>3}  {s['run_id']:>5}  "
              f"{s['mean_test_rmse_um']:>7.3f}  "
              f"{s['mean_test_sigma_um']:>6.3f}  "
              f"{s['rmse_sigma_ratio']:>5.2f}  "
              f"{_arch_str(hp['hidden_dims']):>14} "
              f"{hp['activation']:>5} "
              f"{hp['learning_rate']:>9.1e} "
              f"{hp['dropout']:>5.3f} "
              f"{hp['level_weights_name']:>8} "
              f"{s['best_epoch']:>6} "
              f"{s['train_seconds']:>5.0f}")
    print('-' * 92)
    rmses = np.array([s['mean_test_rmse_um'] for s in sorted_runs])
    print(f"  spread        : {rmses.min():.3f} → {rmses.max():.3f} μm "
          f"(median {np.median(rmses):.3f})")
    print(f"  best run      : {sorted_runs[0]['run_id']} "
          f"({sorted_runs[0]['mean_test_rmse_um']:.3f} μm)")
    pl = sorted_runs[0]['per_level_rmse_um']
    print(f"  best per-lev  : "
          + '  '.join(f'L{i+1:02d}={r:.2f}' for i, r in enumerate(pl)))


def save_csv(summaries: List[Dict], path: Path):
    sorted_runs = sorted(summaries, key=lambda s: s['mean_test_rmse_um'])
    n_lev = len(sorted_runs[0]['per_level_rmse_um'])

    cols = (['rank', 'run_id', 'tag', 'variant',
             'mean_test_rmse_um', 'mean_test_sigma_um', 'rmse_sigma_ratio',
             'best_epoch', 'epochs_trained', 'train_seconds',
             'hidden_dims', 'activation', 'learning_rate', 'dropout',
             'lambda_physics', 'lambda_monotonicity',
             'lambda_adiabatic', 'lambda_smoothness',
             'level_weights_name', 'batch_size', 'augment_noise_std']
            + [f'rmse_L{i+1:02d}_um' for i in range(n_lev)])

    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rank, s in enumerate(sorted_runs, start=1):
            hp = s['hyperparams']
            row = [rank, s['run_id'], s.get('tag', ''), s.get('variant', ''),
                   s['mean_test_rmse_um'], s['mean_test_sigma_um'],
                   s['rmse_sigma_ratio'],
                   s['best_epoch'], s['epochs_trained'], s['train_seconds'],
                   '|'.join(str(h) for h in hp['hidden_dims']),
                   hp['activation'], hp['learning_rate'], hp['dropout'],
                   hp.get('lambda_physics', 0), hp.get('lambda_monotonicity', 0),
                   hp.get('lambda_adiabatic', 0), hp.get('lambda_smoothness', 0),
                   hp['level_weights_name'], hp['batch_size'],
                   hp.get('augment_noise_std', 0)]
            row += list(s['per_level_rmse_um'])
            w.writerow(row)


def save_json(summaries: List[Dict], path: Path):
    sorted_runs = sorted(summaries, key=lambda s: s['mean_test_rmse_um'])
    aggregate = []
    for rank, s in enumerate(sorted_runs, start=1):
        aggregate.append({
            'rank':              rank,
            'run_id':            s['run_id'],
            'tag':               s.get('tag', ''),
            'variant':           s.get('variant', ''),
            'mean_test_rmse_um': s['mean_test_rmse_um'],
            'mean_test_sigma_um': s['mean_test_sigma_um'],
            'rmse_sigma_ratio':  s['rmse_sigma_ratio'],
            'per_level_rmse_um': s['per_level_rmse_um'],
            'best_epoch':        s['best_epoch'],
            'best_val_loss':     s['best_val_loss'],
            'train_seconds':     s['train_seconds'],
            'hyperparams':       s['hyperparams'],
        })
    with path.open('w') as f:
        json.dump(aggregate, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset':  'cm',
        'axes.labelsize':    11, 'axes.titlesize': 12,
        'figure.titlesize':  13, 'legend.fontsize':  9,
        'xtick.labelsize':    9, 'ytick.labelsize':  9,
        'axes.linewidth':    0.8,
    })


def make_plots(summaries: List[Dict], fig_dir: Path, n_top: int = 10):
    """Per-variant plots: rank curve, top-N per-level RMSE, hp scatters."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sorted_runs = sorted(summaries, key=lambda s: s['mean_test_rmse_um'])
    rmse = np.array([s['mean_test_rmse_um'] for s in sorted_runs])

    # 1. Rank curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(rmse) + 1), rmse, 'o-', color='steelblue', ms=3)
    ax.axhline(rmse[0], color='firebrick', linestyle='--', lw=1,
               label=f'best = {rmse[0]:.3f} μm (run {sorted_runs[0]["run_id"]})')
    ax.set_xlabel('Rank (best → worst)')
    ax.set_ylabel('Mean test RMSE (μm)')
    ax.set_title(f'Sweep ranking ({len(rmse)} configs)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'sweep_ranking.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # 2. Per-level RMSE for top-N configs (rotated layout: levels on y-axis,
    #    level 1 at the TOP via invert_yaxis — matches compare_sweep_profile_only
    #    visual convention).
    top = sorted_runs[:n_top]
    n_lev = len(top[0]['per_level_rmse_um'])
    levels = np.arange(1, n_lev + 1)
    fig, ax = plt.subplots(figsize=(8, 9))
    cmap = plt.get_cmap('viridis')
    for i, s in enumerate(top):
        c = cmap(i / max(n_top - 1, 1))
        ax.plot(s['per_level_rmse_um'], levels, '-', color=c, linewidth=1.3,
                label=f"Run {s['run_id']:03d} ({s['mean_test_rmse_um']:.2f} μm)")
    ax.set_ylabel(f'Vertical level (1 = cloud top, {n_lev} = cloud base)',
                  fontsize=11)
    ax.set_xlabel('Per-level test RMSE (μm)', fontsize=11)
    ax.set_title(f'Per-Level RMSE — Top {n_top} Runs', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_yticks(levels)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'sweep_per_level_top10.png',
                dpi=400, bbox_inches='tight')
    plt.close(fig)

    # 3. Continuous-axis sensitivity scatter (3x3, mirrors K-fold comparator's
    #    sweep_sensitivity_continuous.png). Cross-fold-std panel is replaced
    #    here with σ-calibration ratio since we have no folds.
    all_rmse = np.array([s['mean_test_rmse_um'] for s in summaries])

    def _scatter(ax, xs, color, xlabel, title, log=False):
        ax.scatter(xs, all_rmse, alpha=0.6, c=color,
                   edgecolors='white', s=35)
        if log:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean RMSE (μm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    _scatter(axes[0, 0],
             [s['hyperparams']['learning_rate']     for s in summaries],
             '#EF4444', 'Learning rate (log)', 'Learning rate', log=True)
    _scatter(axes[0, 1],
             [s['hyperparams']['dropout']           for s in summaries],
             '#3B82F6', 'Dropout',               'Dropout')
    _scatter(axes[0, 2],
             [s['hyperparams']['augment_noise_std'] for s in summaries],
             '#F59E0B', 'augment_noise_std',     'Spectral noise augmentation')
    _scatter(axes[1, 0],
             [s['hyperparams']['lambda_physics']      for s in summaries],
             '#8B5CF6', 'λ_physics',           'λ_physics (outer physics weight)')
    _scatter(axes[1, 1],
             [s['hyperparams']['lambda_monotonicity'] for s in summaries],
             '#10B981', 'λ_monotonicity',       'λ_monotonicity')
    _scatter(axes[1, 2],
             [s['hyperparams']['lambda_adiabatic']    for s in summaries],
             '#06B6D4', 'λ_adiabatic',          'λ_adiabatic')
    _scatter(axes[2, 0],
             [s['hyperparams']['lambda_smoothness']   for s in summaries],
             '#EC4899', 'λ_smoothness',         'λ_smoothness')
    # No cross-fold std for single-split sweep — show σ-calibration instead
    _scatter(axes[2, 1],
             [s['rmse_sigma_ratio'] for s in summaries],
             '#6B7280', 'RMSE / σ',             'σ-calibration vs mean RMSE')
    axes[2, 1].axvline(1.0, color='0.5', linestyle='--', lw=1,
                       label='calibrated')
    axes[2, 1].legend(fontsize=8)
    _scatter(axes[2, 2],
             [_estimate_n_params(s['hyperparams']['hidden_dims'],
                                 n_input=643,
                                 n_output=len(s['per_level_rmse_um']))
              for s in summaries],
             '#D97706', 'n_params (est.)',      'Model capacity')

    fig.suptitle(
        'Continuous-Axis Sensitivity (each dot = 1 config, single split)',
        fontsize=14, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / 'sweep_sensitivity_continuous.png',
                dpi=400, bbox_inches='tight')
    plt.close(fig)

    # 4. Categorical-axis box plots (2x2, mirrors K-fold comparator).
    def _box(ax, key, title, color):
        groups = {}
        for s in summaries:
            k = s['hyperparams'][key]
            groups.setdefault(k, []).append(s['mean_test_rmse_um'])
        names = sorted(groups.keys(), key=lambda x: str(x))
        data  = [groups[n] for n in names]
        bp = ax.boxplot(data, tick_labels=[str(n) for n in names],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
        ax.set_ylabel('Mean RMSE (μm)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

    def _box_arch(ax):
        groups = {}
        for s in summaries:
            arch = _arch_str(s['hyperparams']['hidden_dims'])
            groups.setdefault(arch, []).append(s['mean_test_rmse_um'])
        names = sorted(groups.keys())
        data  = [groups[n] for n in names]
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#DBEAFE')
        ax.set_ylabel('Mean RMSE (μm)')
        ax.set_title('Architecture')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _box_arch(axes[0, 0])
    _box(axes[0, 1], 'level_weights_name',
         'Level-weight scheme', '#D1FAE5')
    _box(axes[1, 0], 'activation',
         'Activation function', '#FEE2E2')
    _box(axes[1, 1], 'batch_size',
         'Batch size', '#FEF3C7')
    fig.suptitle('Categorical-Axis Comparison', fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(fig_dir / 'sweep_sensitivity_categorical.png',
                dpi=400, bbox_inches='tight')
    plt.close(fig)

    # 5. σ-calibration: RMSE/σ ratio across runs
    ratios = np.array([s['rmse_sigma_ratio'] for s in sorted_runs])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(ratios) + 1), ratios, 'o-',
            color='darkorange', ms=3)
    ax.axhline(1.0, color='0.4', linestyle='--', lw=1, label='calibrated (=1)')
    ax.set_xlabel('Rank by RMSE (best → worst)')
    ax.set_ylabel('RMSE / mean σ')
    ax.set_title('Uncertainty calibration across runs')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'sweep_calibration.png',
                dpi=400, bbox_inches='tight')
    plt.close(fig)


def make_cross_variant_plot(per_variant: Dict[str, List[Dict]],
                            out_path: Path, n_top: int = 10):
    """For multi-sweep comparison: best-rank curves overlaid + paired
    same-config comparison if the variants share run_ids."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = {'M0': 'steelblue', 'M1': 'darkorange', 'M2': 'seagreen'}

    # (a) Rank curves overlaid
    ax = axes[0]
    for label, sums in per_variant.items():
        sorted_s = sorted(sums, key=lambda s: s['mean_test_rmse_um'])
        rmse = np.array([s['mean_test_rmse_um'] for s in sorted_s])
        c = cmap.get(label, None)
        ax.plot(np.arange(1, len(rmse) + 1), rmse, 'o-', ms=3, lw=1,
                label=f"{label} (best {rmse[0]:.3f} μm)", color=c)
    ax.set_xlabel('Rank (best → worst)')
    ax.set_ylabel('Mean test RMSE (μm)')
    ax.set_title('Cross-variant ranking')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Paired same-run comparison if variants share run_ids
    ax = axes[1]
    variants = list(per_variant.keys())
    if len(variants) >= 2:
        run_ids = sorted(set.intersection(
            *[{s['run_id'] for s in v} for v in per_variant.values()]))
        for label in variants:
            sums_by_id = {s['run_id']: s for s in per_variant[label]}
            rmse = np.array([sums_by_id[i]['mean_test_rmse_um']
                             for i in run_ids])
            c = cmap.get(label, None)
            ax.plot(run_ids, rmse, '.-', ms=4, lw=0.8, alpha=0.7,
                    label=label, color=c)
        ax.set_xlabel('Run ID (shared hyperparameter draw)')
        ax.set_ylabel('Mean test RMSE (μm)')
        ax.set_title('Same-config comparison across variants')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Cross-variant figure → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--sweep-dir', nargs='+', required=True,
                   help='one or more sweep_results_profile_only_synthetic_* dirs')
    p.add_argument('--top-n', type=int, default=20,
                   help='show top-N rows in the printed table (default 20)')
    args = p.parse_args()

    setup_style()

    # Per-variant single-dir reports
    per_variant: Dict[str, List[Dict]] = {}
    for sd in args.sweep_dir:
        sd_path = Path(sd).resolve()
        summaries = load_summaries(sd_path)
        if not summaries:
            print(f'[skip] {sd_path}: no run_*/summary.json found')
            continue

        label = summaries[0].get('variant') or sd_path.name.split('_')[-1]
        per_variant[label] = summaries

        print(f'\n{"#" * 80}\n# {sd_path.name}  (variant {label})\n{"#" * 80}')
        print_table(summaries, top_n=args.top_n)
        save_csv (summaries, sd_path / 'comparison.csv')
        save_json(summaries, sd_path / 'comparison.json')
        make_plots(summaries, sd_path / 'Figures')
        print(f'  CSV  → {sd_path / "comparison.csv"}')
        print(f'  JSON → {sd_path / "comparison.json"}')
        print(f'  Figs → {sd_path / "Figures"}/')

    # Cross-variant figure if more than one dir
    if len(per_variant) >= 2:
        first_dir = Path(args.sweep_dir[0]).resolve()
        out = first_dir / 'Figures' / 'cross_variant_comparison.png'
        make_cross_variant_plot(per_variant, out)


if __name__ == '__main__':
    main()
