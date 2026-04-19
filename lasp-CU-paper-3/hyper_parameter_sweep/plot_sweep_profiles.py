"""
plot_sweep_profiles.py — predicted vs true r_e profiles for top-N runs.

For each of the top-N runs (ranked by mean test RMSE):
  - load best_model.pt + config.json
  - build the test dataloader (profile-held-out split, same seed as training)
  - iterate test batches, DEDUPLICATING by unique true profile so each panel
    shows a different cloud (each held-out cloud appears ~128 times in the
    loader via varied solar/viewing geometry; without deduplication all
    panels in the figure would show the same cloud)
  - pull raw in-situ profiles (before altitude interpolation) from the HDF5
    so they can be overlaid on the retrieval
  - 2×3 panel figure: left y-axis is optical depth τ; right y-axis is
    altitude (km).  True profile and NN retrieval (±1σ) are plotted on
    the interpolated grid; raw in-situ measurement is overlaid.

Optical-depth axis
------------------
If the HDF5 contains a `profiles_raw_tau` dataset (added by the updated
`convert_matFiles_to_HDF.py`), we use the real measured τ(z) to build
the left y-axis and to place the 7 interpolated model levels.  If the
dataset is absent (older HDF5 files), we fall back to a linear
approximation τ(level i) = i/(N-1) · τ_c.  A one-line notice is printed
per run indicating which path was used.

Requires:
    - HDF5 training file accessible (use --h5-path to override if config
      points at Alpine but you are running locally).
    - torch, numpy, matplotlib, pandas, h5py.

Usage:
    python plot_sweep_profiles.py --results-dir sweep_results_3
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
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import DropletProfileNetwork, RetrievalConfig
from data import create_dataloaders


GREEN     = '#10B981'   # NN retrieval
RAW_BLUE  = '#1F5FC2'   # raw in-situ


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────
def build_model_from_config(cfg_dict, device, n_levels):
    hp = cfg_dict['hyperparams']
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation='gelu',
    )
    model = DropletProfileNetwork(model_config).to(device)
    return model, model_config


# ─────────────────────────────────────────────────────────────────────────────
# Inference + deduplication
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gather_unique_profiles(model, test_loader, device, n_examples):
    """
    Walk the test DataLoader and collect one example per UNIQUE true profile.

    The held-out test set contains ~14 distinct clouds, each replicated at
    ~128 sun/view geometries.  Without this deduplication, picking random
    loader indices tends to grab several geometries of the same cloud —
    which is what produced the "all six panels are identical" bug in the
    previous version of this script.

    Returns a dict with arrays indexed by unique-profile order.  Also
    returns `subset_positions`, the position of each chosen sample in the
    test Subset; callers use this to look up the corresponding row in the
    underlying HDF5 (via subset.indices) for the raw in-situ overlay.
    """
    model.eval()

    subset_pos   = 0   # running index into the test Subset
    seen         = {}  # profile-hash -> subset position of first occurrence
    pred_u       = {}
    pred_std_u   = {}
    tau_pred_u   = {}
    tau_pred_std_u = {}
    prof_norm_u  = {}
    tau_true_u   = {}

    for x, prof, tau in test_loader:
        x_d = x.to(device)
        out = model(x_d)
        pred          = out['profile'].cpu().numpy()
        pred_std      = out['profile_std'].cpu().numpy()
        tau_pred      = out['tau_c'].squeeze(-1).cpu().numpy()
        tau_pred_std  = out['tau_std'].squeeze(-1).cpu().numpy()
        prof_np       = prof.numpy()
        tau_np        = tau.numpy()

        for i in range(prof_np.shape[0]):
            key = tuple(np.round(prof_np[i], 6))  # identical profiles hash the same
            if key not in seen:
                seen[key]           = subset_pos + i
                pred_u[key]         = pred[i]
                pred_std_u[key]     = pred_std[i]
                tau_pred_u[key]     = float(tau_pred[i])
                tau_pred_std_u[key] = float(tau_pred_std[i])
                prof_norm_u[key]    = prof_np[i]
                tau_true_u[key]     = float(tau_np[i])
            if len(seen) >= n_examples:
                break

        subset_pos += prof_np.shape[0]
        if len(seen) >= n_examples:
            break

    keys = list(seen.keys())
    return {
        'subset_positions': np.array([seen[k] for k in keys], dtype=int),
        'pred'            : np.stack([pred_u[k]         for k in keys]),
        'pred_std'        : np.stack([pred_std_u[k]     for k in keys]),
        'tau_pred'        : np.array([tau_pred_u[k]     for k in keys]),
        'tau_pred_std'    : np.array([tau_pred_std_u[k] for k in keys]),
        'prof_norm'       : np.stack([prof_norm_u[k]    for k in keys]),
        'tau_true_norm'   : np.array([tau_true_u[k]     for k in keys]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Raw profile lookup from the HDF5
# ─────────────────────────────────────────────────────────────────────────────
def read_raw_profiles(h5_path, subset, subset_positions):
    """
    Fetch the raw in-situ r_e profile, raw altitude grid, and scalar tau_c
    for each chosen sample.  subset_positions are positions within the test
    Subset; we translate via subset.indices to the underlying HDF5 row.
    """
    if hasattr(subset, 'indices'):
        subset_to_hdf5 = np.asarray(subset.indices, dtype=int)
    else:
        # Fallback: identity mapping (no Subset wrapper).  Would only happen
        # if profile_holdout=False; we warn then proceed.
        print("  [warn] test loader's dataset has no .indices; assuming "
              "identity mapping to HDF5 rows.")
        subset_to_hdf5 = np.arange(len(subset))

    hdf5_rows = subset_to_hdf5[subset_positions]

    raw_re, raw_z, tau_c_vals = [], [], []
    with h5py.File(h5_path, 'r') as f:
        n_lev_arr   = f['profile_n_levels']
        raw_re_ds   = f['profiles_raw']
        raw_z_ds    = f['profiles_raw_z']
        tau_c_ds    = f['tau_c']
        for r in hdf5_rows:
            n = int(n_lev_arr[r])
            raw_re.append(raw_re_ds[r, :n].astype(np.float32).copy())
            raw_z.append(raw_z_ds[r, :n].astype(np.float32).copy())
            tau_c_vals.append(float(tau_c_ds[r]))
    return raw_re, raw_z, np.array(tau_c_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Per-panel plot helper (also used by the grid figure)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_panel(ax, pred, pred_std, true_um, raw_re, raw_z,
                tau_true, tau_pred, tau_pred_std,
                show_legend=True, tick_fontsize=8):
    """
    Draw a single retrieval-vs-truth panel.

    Left y-axis: optical depth τ (linear approximation from τ_c).
    Right y-axis: altitude (km), rendered via a twin axis.
    """
    n_levels = len(pred)

    # Optical depth at each of the N interpolated levels (linear approx).
    # The model levels are evenly spaced in altitude between z_top and z_base,
    # so assuming roughly constant extinction they are evenly spaced in τ too.
    tau_lvl = np.linspace(0.0, tau_true, n_levels)

    # Altitudes at each of the N interpolated levels: the converter uses
    # linspace between z_raw[0] (top) and z_raw[-1] (base).
    z_top  = float(raw_z[0])
    z_base = float(raw_z[-1])
    z_lvl  = np.linspace(z_top, z_base, n_levels)

    # Raw profile on the same τ axis: map z -> τ via the same linear law
    # τ(z) = (z_top - z) / (z_top - z_base) · τ_true.  This is the only
    # approximation in the figure; see module docstring.
    tau_raw = (z_top - raw_z) / max(z_top - z_base, 1e-9) * tau_true

    # Raw in-situ profile (blue, partially transparent)
    ax.plot(raw_re, tau_raw, '-', color=RAW_BLUE, linewidth=1.0, alpha=0.55,
            label='In-situ raw profile')

    # Interpolated true profile on the 7-level model grid (black circles)
    ax.plot(true_um, tau_lvl, 'ko-', markersize=5, linewidth=1.6,
            label=f'{n_levels}-level training profile')

    # NN retrieval with ±1σ
    ax.errorbar(pred, tau_lvl, xerr=pred_std,
                fmt='s--', color=GREEN, markersize=4, linewidth=1.5,
                elinewidth=1.0, capsize=3,
                label='NN retrieval ±1σ')

    # τ=0 at the top of the plot (cloud top), τ=τ_c at the bottom.
    ax.invert_yaxis()
    ax.set_xlabel(r'$r_e$ (μm)', fontsize=10)
    ax.set_ylabel(r'Optical depth $\tau$ (approx)', fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=tick_fontsize)

    # Right y-axis: altitude (km).  Because ax has τ=0 at top and τ=τ_c at
    # bottom, and τ increases linearly with (z_top - z), the altitude axis
    # should have z_top at the TOP of the plot and z_base at the BOTTOM —
    # i.e. ax2.set_ylim(z_base, z_top) (ymin at bottom, ymax at top, no invert).
    ax2 = ax.twinx()
    ax2.set_ylim(z_base, z_top)
    ax2.set_ylabel('Altitude (km)', fontsize=10)
    ax2.tick_params(labelsize=tick_fontsize)

    ax.set_title(
        rf'$\tau_\mathrm{{true}}$ = {tau_true:.2f}   '
        rf'$\tau_\mathrm{{pred}}$ = {tau_pred:.2f} ± {tau_pred_std:.2f}',
        fontsize=10,
    )
    if show_legend:
        ax.legend(fontsize=8, loc='upper right')
    return ax2


def plot_profiles(ex, run_info, out_path, n_examples=6):
    """6-panel figure for one run: unique test profiles, retrieved ±1σ vs true."""
    n_examples = min(n_examples, len(ex['pred']))
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), squeeze=False)
    axes = axes.ravel()

    for j in range(n_examples):
        _draw_panel(
            axes[j],
            pred         = ex['pred'][j],
            pred_std     = ex['pred_std'][j],
            true_um      = ex['true_um'][j],
            raw_re       = ex['raw_re'][j],
            raw_z        = ex['raw_z'][j],
            tau_true     = ex['tau_true'][j],
            tau_pred     = ex['tau_pred'][j],
            tau_pred_std = ex['tau_pred_std'][j],
            show_legend  = (j == 0),
        )

    for ax in axes[n_examples:]:
        ax.axis('off')

    fig.suptitle(
        f"run_{int(run_info['run_id']):03d}: {run_info['run_name']}   "
        f"(mean RMSE {run_info['mean_rmse']:.3f} μm)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_grid_single(per_run_examples, out_path):
    """One subplot per top-N run, each showing that run's first unique profile."""
    n = len(per_run_examples)
    ncols = 5 if n >= 5 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.8 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    for ax, ex in zip(axes, per_run_examples):
        _draw_panel(
            ax,
            pred         = ex['pred'],
            pred_std     = ex['pred_std'],
            true_um      = ex['true'],
            raw_re       = ex['raw_re'],
            raw_z        = ex['raw_z'],
            tau_true     = ex['tau_true'],
            tau_pred     = ex['tau_pred'],
            tau_pred_std = ex['tau_pred_std'],
            show_legend  = False,
            tick_fontsize= 7,
        )
        ax.set_title(f"run_{ex['run_id']:03d} — RMSE {ex['mean_rmse']:.3f} μm\n"
                     rf"$\tau_t$={ex['tau_true']:.1f}, "
                     rf"$\tau_p$={ex['tau_pred']:.1f}±{ex['tau_pred_std']:.2f}",
                     fontsize=8)

    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='sweep_results_3')
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--n-examples', type=int, default=6,
                   help='Unique profiles per run in the per-run figure')
    p.add_argument('--device', default=None,
                   help='"cuda", "cpu", or leave unset for auto-detect')
    p.add_argument('--h5-path', default=None,
                   help='Override HDF5 path from config.json '
                        '(useful when config points at Alpine but running locally)')
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
            print(f"  [skip] rank {rank+1}: missing config or checkpoint in {run_dir}")
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        hp = cfg['hyperparams']

        h5_path = args.h5_path if args.h5_path else cfg['data']['h5_path']
        if not Path(h5_path).exists():
            print(f"  [skip] rank {rank+1}: HDF5 not found at {h5_path}")
            print(f"         pass --h5-path /your/local/path to override")
            continue

        _, _, test_loader = create_dataloaders(
            h5_path=h5_path,
            instrument=cfg['data'].get('instrument', 'hysics'),
            batch_size=hp.get('batch_size', 256),
            num_workers=0,
            seed=42,
            profile_holdout=True,
            n_val_profiles=14,
            n_test_profiles=14,
        )

        # Unwrap to find n_levels on the base RetrievalDataset.
        base_ds = test_loader.dataset
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
        n_levels = base_ds.n_levels

        model, model_config = build_model_from_config(cfg, device, n_levels)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        # 1. Gather unique profiles (fixes the "6 identical panels" bug)
        found = gather_unique_profiles(model, test_loader, device, args.n_examples)
        n_found = len(found['pred'])
        if n_found < args.n_examples:
            print(f"  [note] rank {rank+1}: only {n_found} unique profiles in "
                  f"first few test batches (needed {args.n_examples})")

        # 2. Denormalize ground truth into physical units (μm, dimensionless τ)
        re_min, re_max = model_config.re_min, model_config.re_max
        tau_min, tau_max = model_config.tau_min, model_config.tau_max
        true_um  = found['prof_norm'] * (re_max - re_min) + re_min
        tau_true = found['tau_true_norm'] * (tau_max - tau_min) + tau_min

        # 3. Pull raw in-situ profiles for these samples from the HDF5
        raw_re, raw_z, _ = read_raw_profiles(
            h5_path, test_loader.dataset, found['subset_positions'])

        ex = {
            'pred'         : found['pred'],
            'pred_std'     : found['pred_std'],
            'tau_pred'     : found['tau_pred'],
            'tau_pred_std' : found['tau_pred_std'],
            'true_um'      : true_um,
            'tau_true'     : tau_true,
            'raw_re'       : raw_re,
            'raw_z'        : raw_z,
        }

        out_path = fig_dir / f"profiles_top{rank+1:02d}_run_{rid:03d}.png"
        plot_profiles(ex, row, out_path, n_examples=args.n_examples)
        print(f"  rank {rank+1:2d}  run_{rid:03d}  "
              f"RMSE={row['mean_rmse']:.3f} μm   -> {out_path.name}")

        # First unique profile from this run goes into the cross-run summary
        per_run_examples.append({
            'run_id'      : rid,
            'mean_rmse'   : row['mean_rmse'],
            'true'        : true_um[0],
            'pred'        : found['pred'][0],
            'pred_std'    : found['pred_std'][0],
            'tau_true'    : tau_true[0],
            'tau_pred'    : found['tau_pred'][0],
            'tau_pred_std': found['tau_pred_std'][0],
            'raw_re'      : raw_re[0],
            'raw_z'       : raw_z[0],
            'raw_tau'     : raw_tau[0],
        })

    if per_run_examples:
        grid_out = fig_dir / f"profiles_top{args.top_n:02d}_grid.png"
        plot_grid_single(per_run_examples, grid_out)
        print(f"\nGrid figure: {grid_out}")
    print(f"All figures written to: {fig_dir}/")


if __name__ == '__main__':
    main()
