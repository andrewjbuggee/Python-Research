"""
compare_sweep.py — Aggregate and compare results from the hyperparameter sweep.

Reads all sweep_results_2/run_*/results.json files and produces:
  1. A sorted table (best → worst by mean RMSE) printed to stdout
  2. sweep_results_2/comparison.csv   — full results for spreadsheet analysis
  3. sweep_results_2/comparison.json  — machine-readable aggregate
  4. sweep_results_2/Figures/sweep_*.png — visualization plots (saved at 400 DPI)

Run after all Alpine jobs complete:
    python compare_sweep.py

Or run locally if you rsync sweep_results_2/ back from Alpine:
    rsync -av anbu8374@login.rc.colorado.edu:/projects/anbu8374/paper3/sweep_results_2/ sweep_results_2/
    python compare_sweep.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import json
import csv
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_results(sweep_dir='sweep_results_2'):
    """Load all results.json files from the sweep directory."""
    results = []
    sweep_path = Path(sweep_dir)

    for run_dir in sorted(sweep_path.glob('run_*')):
        results_file = run_dir / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                results.append(json.load(f))

    return results


def print_table(results):
    """Print a human-readable sorted table of results."""

    # Sort by mean RMSE (primary), then val loss (secondary)
    results_sorted = sorted(results, key=lambda r: r['test_metrics']['mean_rmse'])

    print(f"\n{'='*120}")
    print(f"  HYPERPARAMETER SWEEP RESULTS — {len(results)} runs completed")
    print(f"{'='*120}")

    # Header
    print(f"\n{'Rank':>4} {'ID':>3} {'Architecture':<16} {'Drop':>5} "
          f"{'LR':>10} {'sigma_fl':>8} {'Weights':<12} "
          f"{'Mean RMSE':>9} {'Tau RMSE':>8} {'Val NLL':>9} "
          f"{'Best Ep':>7} {'Time':>6}")
    print("-" * 120)

    for rank, r in enumerate(results_sorted, 1):
        hp = r['hyperparams']
        dims = hp['hidden_dims']
        n = len(dims)
        w = dims[0]
        arch = f"{n}x{w}" if all(h == w for h in dims) else "tapered"

        tm = r['test_metrics']
        minutes = r['train_time_seconds'] / 60

        marker = " ***" if rank <= 5 else ""

        print(f"{rank:4d} {r['run_id']:3d} {arch:<16} {hp['dropout']:5.2f} "
              f"{hp['learning_rate']:10.6f} {hp['sigma_floor']:8.4f} "
              f"{hp['level_weights_name']:<12} "
              f"{tm['mean_rmse']:9.3f} {tm['tau_rmse']:8.3f} "
              f"{r['best_val_loss']:+9.3f} "
              f"{r['best_epoch']:7d} {minutes:5.1f}m{marker}")

    # Per-level breakdown for top 10
    print(f"\n\n{'='*120}")
    print(f"  PER-LEVEL RMSE (um) — TOP 10 RUNS")
    print(f"{'='*120}")
    n_levels = len(results_sorted[0]['test_metrics']['rmse_per_level'])

    header = f"{'Rank':>4} {'ID':>3} "
    for lvl in range(1, n_levels + 1):
        header += f"{'L'+str(lvl):>7}"
    header += f"  {'Mean':>7}"
    print(header)
    print("-" * (20 + 7 * n_levels + 9))

    for rank, r in enumerate(results_sorted[:10], 1):
        line = f"{rank:4d} {r['run_id']:3d} "
        for rmse in r['test_metrics']['rmse_per_level']:
            line += f"{rmse:7.3f}"
        line += f"  {r['test_metrics']['mean_rmse']:7.3f}"
        print(line)

    # Summary statistics
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    all_mean_rmse = [r['test_metrics']['mean_rmse'] for r in results_sorted]
    print(f"  Best mean RMSE:  {min(all_mean_rmse):.3f} um  (run {results_sorted[0]['run_id']:03d})")
    print(f"  Worst mean RMSE: {max(all_mean_rmse):.3f} um")
    print(f"  Median:          {np.median(all_mean_rmse):.3f} um")
    print(f"  Runs completed:  {len(results)}/50")

    # Best hyperparams
    best = results_sorted[0]
    print(f"\n  Best configuration (run {best['run_id']:03d}):")
    print(f"    hidden_dims:    {best['hyperparams']['hidden_dims']}")
    print(f"    dropout:        {best['hyperparams']['dropout']}")
    print(f"    learning_rate:  {best['hyperparams']['learning_rate']}")
    print(f"    sigma_floor:    {best['hyperparams']['sigma_floor']}")
    print(f"    level_weights:  {best['hyperparams']['level_weights_name']}")
    print(f"    best_epoch:     {best['best_epoch']}")
    print(f"    test NLL:       {best['test_nll']:+.4f}")


def save_csv(results, path='sweep_results_2/comparison.csv'):
    """Save results as CSV for spreadsheet analysis."""
    if not results:
        return

    n_levels = len(results[0]['test_metrics']['rmse_per_level'])

    fieldnames = [
        'run_id', 'run_name', 'architecture', 'n_layers', 'width',
        'dropout', 'learning_rate', 'sigma_floor', 'level_weights_name',
        'n_params', 'best_epoch', 'final_epoch', 'train_time_min',
        'best_val_loss', 'test_nll', 'mean_rmse', 'tau_rmse',
        'mean_std_overall', 'mean_tau_std',
    ] + [f'rmse_L{i+1}' for i in range(n_levels)]  \
      + [f'std_L{i+1}' for i in range(n_levels)]

    rows = []
    for r in results:
        hp = r['hyperparams']
        dims = hp['hidden_dims']
        n = len(dims)
        w = dims[0]
        arch = f"{n}x{w}" if all(h == w for h in dims) else "tapered"

        row = {
            'run_id': r['run_id'],
            'run_name': r['run_name'],
            'architecture': arch,
            'n_layers': n,
            'width': w if all(h == w for h in dims) else dims[-1],
            'dropout': hp['dropout'],
            'learning_rate': hp['learning_rate'],
            'sigma_floor': hp['sigma_floor'],
            'level_weights_name': hp['level_weights_name'],
            'n_params': r['n_params'],
            'best_epoch': r['best_epoch'],
            'final_epoch': r['final_epoch'],
            'train_time_min': r['train_time_seconds'] / 60,
            'best_val_loss': r['best_val_loss'],
            'test_nll': r['test_nll'],
            'mean_rmse': r['test_metrics']['mean_rmse'],
            'tau_rmse': r['test_metrics']['tau_rmse'],
            'mean_std_overall': r['test_metrics']['mean_std_overall'],
            'mean_tau_std': r['test_metrics']['mean_tau_std'],
        }
        for i, rmse in enumerate(r['test_metrics']['rmse_per_level']):
            row[f'rmse_L{i+1}'] = rmse
        for i, std in enumerate(r['test_metrics']['mean_std_per_level']):
            row[f'std_L{i+1}'] = std
        rows.append(row)

    # Sort by mean_rmse
    rows.sort(key=lambda x: x['mean_rmse'])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to {path}")


def make_plots(results, fig_dir='sweep_results_2/Figures'):
    """Generate comparison plots."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return

    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    # Sort by mean RMSE
    results_sorted = sorted(results, key=lambda r: r['test_metrics']['mean_rmse'])

    # ── Plot 1: Mean RMSE bar chart (all runs, sorted) ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ids = [r['run_id'] for r in results_sorted]
    rmses = [r['test_metrics']['mean_rmse'] for r in results_sorted]
    colors = ['#10B981' if i < 5 else '#6B7280' if i < 15 else '#D1D5DB'
              for i in range(len(rmses))]
    ax.bar(range(len(rmses)), rmses, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Run (sorted by mean RMSE)', fontsize=11)
    ax.set_ylabel('Mean RMSE (um)', fontsize=11)
    ax.set_title('Hyperparameter Sweep — All Runs Ranked by Mean RMSE', fontsize=13)
    ax.set_xticks(range(0, len(rmses), 5))
    ax.set_xticklabels([str(ids[i]) for i in range(0, len(ids), 5)], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_ranking.png', dpi=400)
    plt.close()

    # ── Plot 2: Per-level RMSE for top 10 runs ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    n_levels = len(results_sorted[0]['test_metrics']['rmse_per_level'])
    levels = np.arange(1, n_levels + 1)

    cmap = plt.cm.viridis
    for i, r in enumerate(results_sorted[:10]):
        rmse_levels = r['test_metrics']['rmse_per_level']
        color = cmap(i / 10)
        ax.plot(rmse_levels, levels, 'o-', color=color, markersize=5,
                linewidth=1.5, label=f"Run {r['run_id']:03d} (mean rmse (μm) ={r['test_metrics']['mean_rmse']:.2f})")

    ax.set_ylabel('Vertical Level (1=top, 10=base)', fontsize=11)
    ax.set_xlabel('RMSE (um)', fontsize=11)
    ax.set_title('Per-Level RMSE — Top 10 Runs', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_yticks(levels)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_per_level_top10.png', dpi=400)
    plt.close()

    # ── Plot 3: Hyperparameter sensitivity (scatter plots) ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    mean_rmses = np.array([r['test_metrics']['mean_rmse'] for r in results])

    # Dropout vs RMSE
    ax = axes[0, 0]
    dropouts = [r['hyperparams']['dropout'] for r in results]
    ax.scatter(dropouts, mean_rmses, alpha=0.6, c='#3B82F6', edgecolors='white')
    ax.set_xlabel('Dropout')
    ax.set_ylabel('Mean RMSE (um)')
    ax.set_title('Dropout Sensitivity')
    ax.grid(True, alpha=0.3)

    # LR vs RMSE
    ax = axes[0, 1]
    lrs = [r['hyperparams']['learning_rate'] for r in results]
    ax.scatter(lrs, mean_rmses, alpha=0.6, c='#EF4444', edgecolors='white')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Mean RMSE (um)')
    ax.set_title('Learning Rate Sensitivity')
    ax.grid(True, alpha=0.3)

    # Sigma floor vs RMSE
    ax = axes[0, 2]
    sigma_floors = [r['hyperparams']['sigma_floor'] for r in results]
    ax.scatter(sigma_floors, mean_rmses, alpha=0.6, c='#8B5CF6', edgecolors='white')
    ax.set_xlabel('Sigma Floor (normalized)')
    ax.set_ylabel('Mean RMSE (um)')
    ax.set_title('Sigma Floor Sensitivity')
    ax.grid(True, alpha=0.3)

    # Architecture vs RMSE (box plot)
    ax = axes[1, 0]
    arch_groups = {}
    for r in results:
        dims = r['hyperparams']['hidden_dims']
        n = len(dims)
        w = dims[0]
        arch = f"{n}x{w}" if all(h == w for h in dims) else "tapered"
        arch_groups.setdefault(arch, []).append(r['test_metrics']['mean_rmse'])
    arch_names = sorted(arch_groups.keys())
    arch_data = [arch_groups[a] for a in arch_names]
    bp = ax.boxplot(arch_data, labels=arch_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#DBEAFE')
    ax.set_ylabel('Mean RMSE (um)')
    ax.set_title('Architecture Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # Level weights vs RMSE (box plot)
    ax = axes[1, 1]
    weight_groups = {}
    for r in results:
        wn = r['hyperparams']['level_weights_name']
        weight_groups.setdefault(wn, []).append(r['test_metrics']['mean_rmse'])
    weight_names = sorted(weight_groups.keys())
    weight_data = [weight_groups[w] for w in weight_names]
    bp = ax.boxplot(weight_data, labels=weight_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#D1FAE5')
    ax.set_ylabel('Mean RMSE (um)')
    ax.set_title('Level Weights Comparison')
    ax.grid(axis='y', alpha=0.3)

    # N params vs RMSE
    ax = axes[1, 2]
    n_params = [r['n_params'] for r in results]
    ax.scatter(n_params, mean_rmses, alpha=0.6, c='#F59E0B', edgecolors='white')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Mean RMSE (um)')
    ax.set_title('Model Capacity vs Performance')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_sensitivity.png', dpi=400, bbox_inches='tight')
    plt.close()

    # ── Plot 4: Overfitting diagnostic (val NLL vs mean RMSE) ──────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    val_losses = [r['best_val_loss'] for r in results]
    ax.scatter(val_losses, mean_rmses, alpha=0.6, c='#3B82F6', edgecolors='white', s=40)
    ax.set_xlabel('Best Validation NLL', fontsize=11)
    ax.set_ylabel('Test Mean RMSE (um)', fontsize=11)
    ax.set_title('Val NLL vs Test RMSE — Overfitting Diagnostic', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / 'sweep_overfit_diagnostic.png', dpi=400)
    plt.close()

    print(f"Plots saved to {fig_path}/")


def main():
    results = load_results()

    if not results:
        print("No results found in sweep_results_2/. "
              "Run the sweep first, or rsync results from Alpine.")
        return

    print_table(results)
    save_csv(results)

    # Save machine-readable aggregate
    aggregate = sorted(
        [{'run_id': r['run_id'],
          'run_name': r['run_name'],
          'mean_rmse': r['test_metrics']['mean_rmse'],
          'tau_rmse': r['test_metrics']['tau_rmse'],
          'best_val_loss': r['best_val_loss'],
          'test_nll': r['test_nll'],
          'best_epoch': r['best_epoch'],
          'hyperparams': r['hyperparams'],
          } for r in results],
        key=lambda x: x['mean_rmse']
    )
    with open('sweep_results_2/comparison.json', 'w') as f:
        json.dump(aggregate, f, indent=2)

    make_plots(results)


if __name__ == '__main__':
    main()
