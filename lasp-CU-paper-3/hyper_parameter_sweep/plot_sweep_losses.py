"""
plot_sweep_losses.py — NLL training/validation loss curves for top-N runs.

Reads sweep_results_3/comparison.csv, picks the top N by mean test RMSE,
and plots each run's train + val NLL vs epoch using data from that run's
history.json.  Runs locally — only needs the JSON history files, no model
checkpoints or HDF5 data.

Outputs:
    sweep_results_3/Figures/loss_top01_<run_name>.png  ... through top10
    sweep_results_3/Figures/loss_top10_grid.png        (all 10 in one fig)

Usage:
    python plot_sweep_losses.py
    python plot_sweep_losses.py --results-dir sweep_results_3 --top-n 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_history(run_dir):
    with open(run_dir / 'history.json') as f:
        h = json.load(f)
    epochs = [e['epoch'] for e in h]
    train  = [e['train_loss'] for e in h]
    val    = [e['val_loss']   for e in h]
    return epochs, train, val


def plot_one(run_dir, run_info, out_path):
    epochs, train, val = load_history(run_dir)
    best_epoch = int(run_info['best_epoch'])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, train, label='Train', linewidth=1.5)
    ax.plot(epochs, val,   label='Validation', linewidth=1.5)
    ax.axvline(best_epoch, color='k', linestyle=':', linewidth=1,
               alpha=0.6, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total NLL Loss', fontsize=12)
    ax.set_title(f"run_{int(run_info['run_id']):03d}: {run_info['run_name']}\n"
                 f"mean RMSE = {run_info['mean_rmse']:.3f} μm, "
                 f"tau RMSE = {run_info['tau_rmse']:.3f}",
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_grid(run_dirs, run_infos, out_path):
    """One figure with all top-N runs in a subplot grid (2 cols)."""
    n = len(run_dirs)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, run_dir, info in zip(axes, run_dirs, run_infos):
        epochs, train, val = load_history(run_dir)
        best_epoch = int(info['best_epoch'])
        ax.plot(epochs, train, label='Train', linewidth=1.3)
        ax.plot(epochs, val,   label='Validation', linewidth=1.3)
        ax.axvline(best_epoch, color='k', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('NLL', fontsize=9)
        ax.set_title(f"run_{int(info['run_id']):03d} — "
                     f"RMSE {info['mean_rmse']:.3f} μm "
                     f"(ep {best_epoch})",
                     fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)

    # Hide any leftover axes
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='sweep_results_3')
    p.add_argument('--top-n', type=int, default=10)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / 'Figures'
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(results_dir / 'comparison.csv')
    top = df.nsmallest(args.top_n, 'mean_rmse').reset_index(drop=True)
    print(f"Plotting loss curves for top {len(top)} runs (by mean RMSE):\n")

    run_dirs, run_infos = [], []
    for rank, row in top.iterrows():
        rid = int(row['run_id'])
        run_dir = results_dir / f'run_{rid:03d}'
        if not (run_dir / 'history.json').exists():
            print(f"  [skip] rank {rank+1}: no history.json in {run_dir}")
            continue
        out = fig_dir / f"loss_top{rank+1:02d}_run_{rid:03d}.png"
        plot_one(run_dir, row, out)
        print(f"  rank {rank+1:2d}  run_{rid:03d}  "
              f"RMSE={row['mean_rmse']:.3f} μm   -> {out.name}")
        run_dirs.append(run_dir)
        run_infos.append(row)

    # Combined grid figure
    grid_out = fig_dir / f"loss_top{args.top_n:02d}_grid.png"
    plot_grid(run_dirs, run_infos, grid_out)
    print(f"\nGrid figure: {grid_out}")
    print(f"All figures written to: {fig_dir}/")


if __name__ == '__main__':
    main()
