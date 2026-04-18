"""
plot_sweep_profiles.py — predicted vs true r_e profiles for top-N runs.

For each of the top-N runs (ranked by mean test RMSE):
  - load best_model.pt + config.json
  - build the test dataloader (profile-held-out split, same seed as training)
  - run inference on the first batch to get r_e predictions + per-level sigma
  - pick 6 test profiles and plot retrieved r_e (±1σ) vs true r_e on
    level-index (cloud top=1 at top, cloud base=10 at bottom)

Outputs:
    sweep_results_2/Figures/profiles_top01_run_XXX.png  ...  top10
    sweep_results_2/Figures/profiles_top10_grid.png  (1 profile per run)

Requires:
    - HDF5 training file accessible (use --h5-path to override if local path
      differs from Alpine's /scratch/alpine/anbu8374/...).
    - torch, numpy, matplotlib, pandas, h5py.

Usage:
    python plot_sweep_profiles.py
    python plot_sweep_profiles.py --h5-path /local/path/training.h5
    python plot_sweep_profiles.py --top-n 10 --n-examples 6 --device cpu

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import DropletProfileNetwork, RetrievalConfig
from data import create_dataloaders


GREEN = '#10B981'


def build_model_from_config(cfg_dict, device):
    hp = cfg_dict['hyperparams']
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=10,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation='gelu',
    )
    model = DropletProfileNetwork(model_config).to(device)
    return model, model_config


@torch.no_grad()
def infer_test_batch(model, test_loader, device, n_examples):
    """Run inference on the first batch and return n_examples profiles."""
    model.eval()
    x, prof_true, tau_true = next(iter(test_loader))
    x = x.to(device)
    out = model(x)
    pred      = out['profile'].cpu().numpy()       # already in μm
    pred_std  = out['profile_std'].cpu().numpy()   # already in μm
    # prof_true is normalized in [0,1]; denormalize
    # (re_min/re_max are fixed by RetrievalConfig defaults)
    # but we'll pull them from the model_config instead to stay safe
    return x.cpu().numpy(), pred, pred_std, prof_true.numpy(), tau_true.numpy()


def plot_profiles(pred, pred_std, true_um, tau_true, run_info, out_path,
                  n_examples=6, seed=0):
    """
    6-panel figure: 6 randomly-selected test profiles, retrieved ±1σ vs true.
    Y-axis is level index 1-10 with 1 at top (cloud top).
    """
    rng = np.random.default_rng(seed)
    n_avail = pred.shape[0]
    idxs = rng.choice(n_avail, size=min(n_examples, n_avail), replace=False)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9), squeeze=False)
    axes = axes.ravel()

    levels = np.arange(1, 11)

    for ax, i in zip(axes, idxs):
        ax.plot(true_um[i], levels, 'ko-', markersize=5, linewidth=1.6,
                label='True (in-situ, 10 levels)')
        ax.errorbar(pred[i], levels,
                    xerr=pred_std[i],
                    fmt='s--', color=GREEN, markersize=4, linewidth=1.5,
                    elinewidth=1.0, capsize=3,
                    label='PINN retrieval ±1σ')
        ax.invert_yaxis()   # level 1 (cloud top) at top
        ax.set_xlabel(r'$r_e$ (μm)', fontsize=10)
        ax.set_ylabel('Level (1=top, 10=base)', fontsize=10)
        ax.set_title(f'test idx {i}  |  '
                     rf'$\tau$ true = {float(tau_true[i]):.1f}',
                     fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle(
        f"run_{int(run_info['run_id']):03d}: {run_info['run_name']}   "
        f"(mean RMSE {run_info['mean_rmse']:.3f} μm)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_grid_single(per_run_examples, out_path):
    """One subplot per top-N run, each showing one example profile."""
    n = len(per_run_examples)
    ncols = 5 if n >= 5 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    levels = np.arange(1, 11)

    for ax, ex in zip(axes, per_run_examples):
        ax.plot(ex['true'], levels, 'ko-', markersize=4, linewidth=1.3,
                label='True')
        ax.errorbar(ex['pred'], levels, xerr=ex['pred_std'],
                    fmt='s--', color=GREEN, markersize=3, linewidth=1.3,
                    elinewidth=0.9, capsize=2, label='PINN ±1σ')
        ax.invert_yaxis()
        ax.set_xlabel(r'$r_e$ (μm)', fontsize=9)
        ax.set_ylabel('Level', fontsize=9)
        ax.set_title(f"run_{ex['run_id']:03d} — "
                     f"RMSE {ex['mean_rmse']:.3f} μm",
                     fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='best')

    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='sweep_results_2')
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--n-examples', type=int, default=6,
                   help='Profiles per run in the per-run figure')
    p.add_argument('--device', default=None,
                   help='"cuda", "cpu", or leave unset for auto-detect')
    p.add_argument('--h5-path', default=None,
                   help='Override HDF5 path from config.json '
                        '(useful when config points at Alpine but running locally)')
    p.add_argument('--seed', type=int, default=0,
                   help='Seed for selecting which test profiles to plot')
    args = p.parse_args()

    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / 'Figures'
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(results_dir / 'comparison.csv')
    top = df.nsmallest(args.top_n, 'mean_rmse').reset_index(drop=True)
    print(f"Top {len(top)} runs by mean RMSE:\n")

    per_run_examples = []

    for rank, row in top.iterrows():
        rid = int(row['run_id'])
        run_dir = results_dir / f'run_{rid:03d}'
        cfg_path = run_dir / 'config.json'
        ckpt_path = run_dir / 'best_model.pt'

        if not cfg_path.exists() or not ckpt_path.exists():
            print(f"  [skip] rank {rank+1}: missing config or checkpoint "
                  f"in {run_dir}")
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        hp = cfg['hyperparams']

        h5_path = args.h5_path if args.h5_path else cfg['data']['h5_path']
        if not Path(h5_path).exists():
            print(f"  [skip] rank {rank+1}: HDF5 not found at {h5_path}")
            print(f"         pass --h5-path /your/local/path to override")
            continue

        # Rebuild loaders with the same split as training (seed=42, same
        # profile_holdout config as sweep_train.py)
        _, _, test_loader = create_dataloaders(
            h5_path=h5_path,
            instrument=cfg['data'].get('instrument', 'hysics'),
            batch_size=hp.get('batch_size', 256),
            num_workers=0,              # single-process for plotting
            seed=42,
            profile_holdout=True,
            n_val_profiles=14,
            n_test_profiles=14,
        )

        model, model_config = build_model_from_config(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        x, pred, pred_std, prof_norm, tau_norm = infer_test_batch(
            model, test_loader, device, args.n_examples)

        # Denormalize truths (predictions from the model head are already in μm)
        re_min, re_max = model_config.re_min, model_config.re_max
        tau_min, tau_max = model_config.tau_min, model_config.tau_max
        true_um = prof_norm * (re_max - re_min) + re_min
        tau_true = tau_norm * (tau_max - tau_min) + tau_min

        out_path = fig_dir / f"profiles_top{rank+1:02d}_run_{rid:03d}.png"
        plot_profiles(pred, pred_std, true_um, tau_true,
                      row, out_path,
                      n_examples=args.n_examples, seed=args.seed)
        print(f"  rank {rank+1:2d}  run_{rid:03d}  "
              f"RMSE={row['mean_rmse']:.3f} μm   -> {out_path.name}")

        # Keep one example for the summary grid
        rng = np.random.default_rng(args.seed + rank)
        k = int(rng.integers(0, pred.shape[0]))
        per_run_examples.append({
            'run_id': rid,
            'mean_rmse': row['mean_rmse'],
            'true': true_um[k],
            'pred': pred[k],
            'pred_std': pred_std[k],
        })

    if per_run_examples:
        grid_out = fig_dir / f"profiles_top{args.top_n:02d}_grid.png"
        plot_grid_single(per_run_examples, grid_out)
        print(f"\nGrid figure: {grid_out}")
    print(f"All figures written to: {fig_dir}/")


if __name__ == '__main__':
    main()
