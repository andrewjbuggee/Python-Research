"""
compare_sweep_profile_only.py — Aggregate the profile-only K-fold sweep.

Reads sweep_results_profile_only/run_*/summary.json (one per config; each
contains 5 folds aggregated into mean ± std headline metrics) and produces:

  1. Sorted stdout table (best -> worst by mean RMSE across folds)
  2. sweep_results_profile_only/comparison.csv     (per-config summary; flat)
  3. sweep_results_profile_only/comparison_folds.csv  (per-fold long-form)
  4. sweep_results_profile_only/comparison.json    (sorted machine-readable aggregate)
  5. sweep_results_profile_only/Figures/sweep_*.png plots @ 400 DPI:
        - sweep_ranking.png             — bar chart with cross-fold error bars
        - sweep_per_level_top10.png     — 50-level RMSE for top 10 with ±std band
        - sweep_sensitivity_continuous.png  — LR, dropout, noise, 4 physics weights
        - sweep_sensitivity_categorical.png — arch, weights, activation, batch box plots
        - sweep_overfit_diagnostic.png  — val NLL vs test RMSE (with error bars)
        - sweep_fold_reproducibility.png — across-fold std distribution and
                                           low-RMSE-vs-low-std relationship

Differences vs. compare_sweep.py
--------------------------------
- Drops τ-related columns (the profile-only model has no τ head).
- Reads K-fold summaries: every metric is mean ± std over folds.
- Adds the new sweep axes:  activation, level_weights ∈ {uniform, top, bottom,
  u_shape}, augment_noise_std, batch_size, lambda_physics, lambda_monotonicity,
  lambda_adiabatic, lambda_smoothness.
- Per-level table is replaced with top/middle/base summary cells (50 levels is
  too wide to print) and saved to CSV.

Run after the SLURM array completes (or post-rsync from Alpine):
    rsync -av anbu8374@login.rc.colorado.edu:/projects/anbu8374/Python-Research/lasp-CU-paper-3/hyper_parameter_sweep/sweep_results_profile_only/ \
        sweep_results_profile_only/
    python compare_sweep_profile_only.py
    
   
Exmaple run command (adjust path as needed):
-------------------------------- 
python compare_sweep_profile_only.py --sweep-dir <path_to_sweep_results>

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


SWEEP_DIR_DEFAULT = 'sweep_results_profile_only'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _arch_str(hidden_dims):
    """Compact architecture descriptor matching the generator's run_name style."""
    n = len(hidden_dims)
    w = hidden_dims[0]
    if all(h == w for h in hidden_dims):
        return f"{n}x{w}"
    if hidden_dims[0] > hidden_dims[-1]:
        return f"{hidden_dims[0]}->{hidden_dims[-1]} ({n}L)"
    return f"custom-{n}L"


def _estimate_n_params(hidden_dims, n_input=640, n_output=50):
    """
    Approximate parameter count for ProfileOnlyNetwork.

    Encoder: Linear (in*h + h) + LayerNorm (2*h) per layer.
    Two heads: profile_head and profile_std_head, each Linear(h_last, n_output).
    """
    n = 0
    in_dim = n_input
    for h in hidden_dims:
        n += in_dim * h + h
        n += 2 * h
        in_dim = h
    n += 2 * (in_dim * n_output + n_output)   # two output heads
    return n


def _per_fold_best_epochs(s):
    return [fr['best_epoch'] for fr in s['fold_results']]


def _per_fold_train_time_s(s):
    return [fr['train_time_seconds'] for fr in s['fold_results']]


# ─────────────────────────────────────────────────────────────────────────────
# Load summaries
# ─────────────────────────────────────────────────────────────────────────────
def load_summaries(sweep_dir: str = SWEEP_DIR_DEFAULT):
    """Load all summary.json files from sweep_dir/run_*/."""
    summaries = []
    sweep_path = Path(sweep_dir)
    for run_dir in sorted(sweep_path.glob('run_*')):
        f = run_dir / 'summary.json'
        if f.exists():
            with open(f) as fh:
                summaries.append(json.load(fh))
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Stdout table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(summaries):
    if not summaries:
        return

    sorted_runs = sorted(
        summaries,
        key=lambda s: (s['test_mean_rmse_mean'], s['test_mean_rmse_std']),
    )

    print(f"\n{'=' * 145}")
    print(f"  PROFILE-ONLY K-FOLD SWEEP — {len(summaries)} configs completed "
          f"(K = {summaries[0]['n_folds']})")
    print(f"{'=' * 145}")

    print(
        f"\n{'Rank':>4} {'ID':>3}  {'Architecture':<14} {'Act':<5} "
        f"{'Wts':<8} {'BS':>4} {'Drop':>5} {'LR':>10} "
        f"{'Test RMSE (μm)':>18} {'Val NLL':>16} {'Best Ep':>9} {'Time/fold':>10}"
    )
    print('-' * 145)

    for rank, s in enumerate(sorted_runs, 1):
        hp = s['hyperparams']
        arch = _arch_str(hp['hidden_dims'])
        rmse_str = f"{s['test_mean_rmse_mean']:6.3f} ± {s['test_mean_rmse_std']:5.3f}"
        nll_str  = f"{s['val_loss_mean']:+7.3f} ± {s['val_loss_std']:5.3f}"
        best_eps = _per_fold_best_epochs(s)
        ep_str   = f"{int(np.mean(best_eps)):4d} ± {int(np.std(best_eps)):3d}"
        t_str    = f"{np.mean(_per_fold_train_time_s(s)) / 60:5.1f} min"
        marker   = ' ***' if rank <= 5 else ''
        print(
            f"{rank:4d} {s['run_id']:3d}  {arch:<14} {hp['activation']:<5} "
            f"{hp['level_weights_name']:<8} {hp['batch_size']:4d} "
            f"{hp['dropout']:5.2f} {hp['learning_rate']:10.6f} "
            f"{rmse_str:>18} {nll_str:>16} {ep_str:>9} {t_str:>10}{marker}"
        )

    # Per-level summary for top 10 — top/middle/base means (50 levels is too wide)
    print(f"\n\n{'=' * 100}")
    print(f"  PER-LEVEL RMSE (μm) — TOP 10  (mean across {summaries[0]['n_folds']} folds)")
    print(f"{'=' * 100}")
    print(f"{'Rank':>4} {'ID':>3}  {'Top (1-10)':>12} {'Mid (21-30)':>13} "
          f"{'Base (41-50)':>13} {'Best lvl':>10} {'Worst lvl':>11} {'Mean':>7}")
    print('-' * 100)
    for rank, s in enumerate(sorted_runs[:10], 1):
        per_lvl = np.array(s['rmse_per_level_mean'])
        n_levels = per_lvl.size
        top_end  = max(1, n_levels // 5)            # first 20 % of levels
        base_lo  = n_levels - top_end
        mid_lo   = n_levels // 2 - top_end // 2
        mid_hi   = mid_lo + top_end
        top_mean = per_lvl[:top_end].mean()
        mid_mean = per_lvl[mid_lo:mid_hi].mean()
        base_mean = per_lvl[base_lo:].mean()
        best_idx  = int(per_lvl.argmin())
        worst_idx = int(per_lvl.argmax())
        print(
            f"{rank:4d} {s['run_id']:3d}  {top_mean:12.3f} {mid_mean:13.3f} "
            f"{base_mean:13.3f} "
            f"L{best_idx + 1:02d}={per_lvl[best_idx]:5.2f} "
            f"L{worst_idx + 1:02d}={per_lvl[worst_idx]:5.2f} "
            f"{s['test_mean_rmse_mean']:7.3f}"
        )

    # Summary statistics
    all_rmse = np.array([s['test_mean_rmse_mean'] for s in sorted_runs])
    all_std  = np.array([s['test_mean_rmse_std']  for s in sorted_runs])
    best     = sorted_runs[0]
    print(f"\n\n{'=' * 70}")
    print(f"  HEADLINE STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Best mean RMSE:  {all_rmse.min():.3f} μm  "
          f"(run {best['run_id']:03d}, ±{best['test_mean_rmse_std']:.3f})")
    print(f"  Worst mean RMSE: {all_rmse.max():.3f} μm")
    print(f"  Median:          {np.median(all_rmse):.3f} μm")
    print(f"  Mean cross-fold std (over all configs): {all_std.mean():.3f} μm  "
          f"(low = reproducible)")
    print(f"  Configs completed: {len(summaries)}")

    print(f"\n  Best configuration (run {best['run_id']:03d}):")
    bhp = best['hyperparams']
    print(f"    test mean RMSE:       {best['test_mean_rmse_mean']:.3f} ± "
          f"{best['test_mean_rmse_std']:.3f} μm")
    print(f"    test NLL:             {best['test_nll_mean']:+.4f} ± "
          f"{best['test_nll_std']:.4f}")
    print(f"    hidden_dims:          {bhp['hidden_dims']}")
    print(f"    activation:           {bhp['activation']}")
    print(f"    dropout:              {bhp['dropout']}")
    print(f"    learning_rate:        {bhp['learning_rate']}")
    print(f"    batch_size:           {bhp['batch_size']}")
    print(f"    level_weights:        {bhp['level_weights_name']}")
    print(f"    augment_noise_std:    {bhp['augment_noise_std']}")
    print(f"    lambda_physics:       {bhp['lambda_physics']}")
    print(f"    lambda_monotonicity:  {bhp['lambda_monotonicity']}")
    print(f"    lambda_adiabatic:     {bhp['lambda_adiabatic']}")
    print(f"    lambda_smoothness:    {bhp['lambda_smoothness']}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(summaries, path: str):
    if not summaries:
        return

    n_levels = len(summaries[0]['rmse_per_level_mean'])

    base_fields = [
        'run_id', 'run_name', 'architecture', 'n_layers', 'width_first',
        'width_last', 'activation', 'level_weights_name', 'batch_size',
        'augment_noise_std', 'dropout', 'learning_rate', 'weight_decay',
        'sigma_floor',
        'lambda_physics', 'lambda_monotonicity', 'lambda_adiabatic',
        'lambda_smoothness',
        'n_params_estimate', 'n_folds',
        'test_mean_rmse_mean', 'test_mean_rmse_std',
        'test_nll_mean', 'test_nll_std',
        'val_loss_mean', 'val_loss_std',
        'best_epoch_mean', 'best_epoch_std',
        'train_time_per_fold_min_mean',
    ]
    fieldnames = (base_fields
                  + [f'rmse_L{i+1}_mean' for i in range(n_levels)]
                  + [f'rmse_L{i+1}_std'  for i in range(n_levels)])

    rows = []
    for s in summaries:
        hp = s['hyperparams']
        dims = hp['hidden_dims']
        best_eps = _per_fold_best_epochs(s)
        train_times = _per_fold_train_time_s(s)
        row = {
            'run_id': s['run_id'],
            'run_name': s['run_name'],
            'architecture': _arch_str(dims),
            'n_layers': len(dims),
            'width_first': dims[0],
            'width_last':  dims[-1],
            'activation': hp['activation'],
            'level_weights_name': hp['level_weights_name'],
            'batch_size': hp['batch_size'],
            'augment_noise_std': hp['augment_noise_std'],
            'dropout': hp['dropout'],
            'learning_rate': hp['learning_rate'],
            'weight_decay': hp.get('weight_decay'),
            'sigma_floor':  hp.get('sigma_floor'),
            'lambda_physics':      hp['lambda_physics'],
            'lambda_monotonicity': hp['lambda_monotonicity'],
            'lambda_adiabatic':    hp['lambda_adiabatic'],
            'lambda_smoothness':   hp['lambda_smoothness'],
            'n_params_estimate':   _estimate_n_params(dims, n_output=n_levels),
            'n_folds': s['n_folds'],
            'test_mean_rmse_mean': s['test_mean_rmse_mean'],
            'test_mean_rmse_std':  s['test_mean_rmse_std'],
            'test_nll_mean':       s['test_nll_mean'],
            'test_nll_std':        s['test_nll_std'],
            'val_loss_mean':       s['val_loss_mean'],
            'val_loss_std':        s['val_loss_std'],
            'best_epoch_mean':     float(np.mean(best_eps)),
            'best_epoch_std':      float(np.std(best_eps)),
            'train_time_per_fold_min_mean': float(np.mean(train_times) / 60),
        }
        for i, v in enumerate(s['rmse_per_level_mean']):
            row[f'rmse_L{i+1}_mean'] = v
        for i, v in enumerate(s['rmse_per_level_std']):
            row[f'rmse_L{i+1}_std'] = v
        rows.append(row)

    rows.sort(key=lambda x: x['test_mean_rmse_mean'])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-config CSV saved to {path}")


def save_csv_folds(summaries, path: str):
    """Long-form CSV: one row per (config, fold).  Useful for detecting fold
    outliers, e.g. one fold with much worse RMSE than the others."""
    if not summaries:
        return

    fieldnames = [
        'run_id', 'run_name', 'fold_idx',
        'best_val_loss', 'best_epoch', 'final_epoch',
        'train_time_seconds', 'test_nll', 'mean_rmse', 'mean_std_overall',
    ]
    rows = []
    for s in summaries:
        for fr in s['fold_results']:
            rows.append({
                'run_id': s['run_id'],
                'run_name': s['run_name'],
                'fold_idx': fr['fold_idx'],
                'best_val_loss': fr['best_val_loss'],
                'best_epoch': fr['best_epoch'],
                'final_epoch': fr['final_epoch'],
                'train_time_seconds': fr['train_time_seconds'],
                'test_nll': fr['test_nll'],
                'mean_rmse': fr['test_metrics']['mean_rmse'],
                'mean_std_overall': fr['test_metrics']['mean_std_overall'],
            })
    rows.sort(key=lambda x: (x['run_id'], x['fold_idx']))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-fold CSV saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(summaries, fig_dir: str):
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return
    if not summaries:
        return

    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    sorted_runs = sorted(
        summaries,
        key=lambda s: (s['test_mean_rmse_mean'], s['test_mean_rmse_std']),
    )

    rmse_mean = np.array([s['test_mean_rmse_mean'] for s in sorted_runs])
    rmse_std  = np.array([s['test_mean_rmse_std']  for s in sorted_runs])
    ids       = [s['run_id'] for s in sorted_runs]
    n_runs    = len(sorted_runs)

    # ── 1. Ranked bar chart with cross-fold error bars ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#10B981' if i < 5 else '#6B7280' if i < 15 else '#D1D5DB'
              for i in range(n_runs)]
    ax.bar(range(n_runs), rmse_mean, yerr=rmse_std,
           color=colors, edgecolor='white', linewidth=0.5,
           ecolor='black', capsize=2, error_kw={'linewidth': 0.6, 'alpha': 0.6})
    ax.set_xlabel('Run (sorted by mean RMSE)', fontsize=11)
    ax.set_ylabel('Test mean RMSE (μm)  —  mean ± std across folds', fontsize=11)
    ax.set_title(f'Profile-Only Sweep — Ranked by Cross-Fold Mean RMSE '
                 f'(K = {sorted_runs[0]["n_folds"]})', fontsize=13)
    step = max(1, n_runs // 30)
    ax.set_xticks(range(0, n_runs, step))
    ax.set_xticklabels([str(ids[i]) for i in range(0, n_runs, step)], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_ranking.png', dpi=400)
    plt.close()

    # ── 2. Per-level RMSE for top 10 with ±std band ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 9))
    n_levels = len(sorted_runs[0]['rmse_per_level_mean'])
    levels = np.arange(1, n_levels + 1)
    cmap = plt.cm.viridis
    for i, s in enumerate(sorted_runs[:10]):
        m = np.array(s['rmse_per_level_mean'])
        sd = np.array(s['rmse_per_level_std'])
        c = cmap(i / 10)
        ax.plot(m, levels, '-', color=c, linewidth=1.3,
                label=f"Run {s['run_id']:03d} ({s['test_mean_rmse_mean']:.2f} μm)")
        ax.fill_betweenx(levels, m - sd, m + sd, color=c, alpha=0.10)
    ax.set_ylabel(f'Vertical level (1 = top, {n_levels} = base)', fontsize=11)
    ax.set_xlabel('RMSE (μm)  —  shaded = ±std across folds', fontsize=11)
    ax.set_title('Per-Level RMSE — Top 10 Runs', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_per_level_top10.png', dpi=400)
    plt.close()

    # ── 3. Continuous-axis sensitivity scatter plots ───────────────────────
    all_rmse = np.array([s['test_mean_rmse_mean'] for s in summaries])

    def _scatter(ax, xs, color, xlabel, title, log=False):
        ax.scatter(xs, all_rmse, alpha=0.6, c=color, edgecolors='white', s=35)
        if log:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean RMSE (μm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    _scatter(axes[0, 0],
             [s['hyperparams']['learning_rate']   for s in summaries],
             '#EF4444', 'Learning rate (log)', 'Learning rate', log=True)
    _scatter(axes[0, 1],
             [s['hyperparams']['dropout']         for s in summaries],
             '#3B82F6', 'Dropout', 'Dropout')
    _scatter(axes[0, 2],
             [s['hyperparams']['augment_noise_std'] for s in summaries],
             '#F59E0B', 'augment_noise_std', 'Spectral noise augmentation')
    _scatter(axes[1, 0],
             [s['hyperparams']['lambda_physics']      for s in summaries],
             '#8B5CF6', 'λ_physics', 'λ_physics (outer physics weight)')
    _scatter(axes[1, 1],
             [s['hyperparams']['lambda_monotonicity'] for s in summaries],
             '#10B981', 'λ_monotonicity', 'λ_monotonicity')
    _scatter(axes[1, 2],
             [s['hyperparams']['lambda_adiabatic']    for s in summaries],
             '#06B6D4', 'λ_adiabatic', 'λ_adiabatic')
    _scatter(axes[2, 0],
             [s['hyperparams']['lambda_smoothness']   for s in summaries],
             '#EC4899', 'λ_smoothness', 'λ_smoothness')
    # Fold-to-fold std vs mean — does best mean also = most reproducible?
    _scatter(axes[2, 1],
             [s['test_mean_rmse_std'] for s in summaries],
             '#6B7280', 'Cross-fold std (μm)',
             'Cross-fold std vs mean RMSE')
    # n_params (estimated)
    _scatter(axes[2, 2],
             [_estimate_n_params(s['hyperparams']['hidden_dims'],
                                 n_output=len(s['rmse_per_level_mean']))
              for s in summaries],
             '#D97706', 'n_params (est.)', 'Model capacity')

    plt.suptitle(
        'Continuous-Axis Sensitivity (each dot = 1 config, K-fold mean)',
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_sensitivity_continuous.png', dpi=400,
                bbox_inches='tight')
    plt.close()

    # ── 4. Categorical-axis box plots ──────────────────────────────────────
    def _box(ax, key, title, color):
        groups = {}
        for s in summaries:
            k = s['hyperparams'][key]
            groups.setdefault(k, []).append(s['test_mean_rmse_mean'])
        names = sorted(groups.keys(), key=lambda x: str(x))
        data  = [groups[n] for n in names]
        bp = ax.boxplot(data, tick_labels=[str(n) for n in names], patch_artist=True)
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
            groups.setdefault(arch, []).append(s['test_mean_rmse_mean'])
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
    plt.suptitle('Categorical-Axis Comparison', fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_sensitivity_categorical.png', dpi=400,
                bbox_inches='tight')
    plt.close()

    # ── 5. Overfitting diagnostic ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    val_mean = [s['val_loss_mean'] for s in summaries]
    val_std  = [s['val_loss_std']  for s in summaries]
    rmse_m   = [s['test_mean_rmse_mean'] for s in summaries]
    rmse_s   = [s['test_mean_rmse_std']  for s in summaries]
    ax.errorbar(val_mean, rmse_m, xerr=val_std, yerr=rmse_s,
                fmt='o', alpha=0.55, mfc='#3B82F6', mec='white',
                ecolor='#3B82F6', capsize=2, elinewidth=0.7)
    ax.set_xlabel('Validation NLL (mean ± std across folds)', fontsize=11)
    ax.set_ylabel('Test mean RMSE (μm)  —  mean ± std across folds',
                  fontsize=11)
    ax.set_title('Validation NLL vs Test RMSE  —  '
                 'Overfitting & K-Fold Reproducibility', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_overfit_diagnostic.png', dpi=400)
    plt.close()

    # ── 6. K-fold reproducibility: distribution of cross-fold std ──────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(rmse_std, bins=20, color='#6B7280', edgecolor='white')
    axes[0].axvline(np.median(rmse_std), color='#EF4444', linewidth=1.5,
                    label=f'median = {np.median(rmse_std):.3f} μm')
    axes[0].set_xlabel('Cross-fold std of test mean RMSE (μm)')
    axes[0].set_ylabel('Number of configs')
    axes[0].set_title('Reproducibility — across-fold variability per config')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].scatter(rmse_mean, rmse_std, alpha=0.6, c='#3B82F6',
                    edgecolors='white', s=35)
    # Highlight top 5
    axes[1].scatter(rmse_mean[:5], rmse_std[:5], color='#10B981',
                    s=60, edgecolors='black', linewidth=0.7,
                    label='Top 5')
    axes[1].set_xlabel('Test mean RMSE (μm)')
    axes[1].set_ylabel('Cross-fold std (μm)')
    axes[1].set_title('Are the best configs also the most reproducible?')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_fold_reproducibility.png', dpi=400)
    plt.close()

    print(f"Plots saved to {fig_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--sweep-dir', type=str, default=SWEEP_DIR_DEFAULT,
                   help=f'directory holding run_*/summary.json (default: '
                        f'{SWEEP_DIR_DEFAULT})')
    args = p.parse_args()

    summaries = load_summaries(args.sweep_dir)
    if not summaries:
        print(f"No summary.json files found in {args.sweep_dir}/. "
              f"Run the sweep first, or rsync results from Alpine.")
        return

    print_table(summaries)

    base = Path(args.sweep_dir)
    save_csv(summaries,        str(base / 'comparison.csv'))
    save_csv_folds(summaries,  str(base / 'comparison_folds.csv'))

    aggregate = sorted(
        [{
            'run_id':  s['run_id'],
            'run_name': s['run_name'],
            'test_mean_rmse_mean': s['test_mean_rmse_mean'],
            'test_mean_rmse_std':  s['test_mean_rmse_std'],
            'test_mean_rmse_per_fold': s['test_mean_rmse_per_fold'],
            'test_nll_mean': s['test_nll_mean'],
            'test_nll_std':  s['test_nll_std'],
            'val_loss_mean': s['val_loss_mean'],
            'val_loss_std':  s['val_loss_std'],
            'best_epoch_mean': float(np.mean(_per_fold_best_epochs(s))),
            'rmse_per_level_mean': s['rmse_per_level_mean'],
            'rmse_per_level_std':  s['rmse_per_level_std'],
            'hyperparams': s['hyperparams'],
        } for s in summaries],
        key=lambda x: (x['test_mean_rmse_mean'], x['test_mean_rmse_std']),
    )
    with open(base / 'comparison.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"Aggregate JSON saved to {base / 'comparison.json'}")

    make_plots(summaries, str(base / 'Figures'))


if __name__ == '__main__':
    main()
