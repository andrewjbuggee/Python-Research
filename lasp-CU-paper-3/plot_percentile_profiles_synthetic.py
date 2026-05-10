"""
plot_percentile_profiles_synthetic.py — generate a 2x3 panel of true-vs-predicted
droplet profiles spanning a chosen set of per-sample RMSE percentiles
(default: 10th, 25th, 40th, 55th, 70th, 85th).

Operates on the output directory of train_standalone_profile_only_synthetic.py,
which contains:
    config.json     — h5_path, variant, extras flags, hyperparameters
    best_model.pt   — model weights from the best validation epoch

The script reloads the model + dataloader (same seed and split sizes as
training), runs predict_test on the test set, ranks samples by per-sample
mean RMSE, picks one sample at each percentile, and plots them.

Usage:
    python plot_percentile_profiles_synthetic.py \\
        --results-dir ./standalone_results_profile_only_synthetic/M0_run098_...
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from models                     import RetrievalConfig
from models_profile_only_extras import ProfileOnlyNetworkExtras
from data                       import create_dataloaders_extras

from train_standalone_profile_only_extras import predict_test


DEFAULT_PERCENTILES = (10, 25, 40, 55, 70, 85)
N_EXTRAS = 3


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--results-dir', type=str, required=True,
                   help='Standalone trainer output dir containing config.json '
                        'and best_model.pt.')
    p.add_argument('--output-name', type=str,
                   default='profiles_true_vs_pred_percentiles.png',
                   help='Filename to write under results-dir. (Default: '
                        'profiles_true_vs_pred_percentiles.png)')
    p.add_argument('--percentiles', type=int, nargs='+',
                   default=list(DEFAULT_PERCENTILES),
                   help='Six per-sample RMSE percentiles to plot. '
                        'Default: 10 25 40 55 70 85.')
    p.add_argument('--device', type=str, default=None,
                   choices=['cuda', 'mps', 'cpu'])
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the seed used at training time so the '
                        'profile-aware split reproduces (default 42).')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Profile selection
# ─────────────────────────────────────────────────────────────────────────────
def select_at_percentiles(per_sample_rmse: np.ndarray,
                          percentiles=DEFAULT_PERCENTILES) -> list[dict]:
    """For each requested percentile return the sample index closest to that
    percentile of the empirical per-sample RMSE distribution. Mirrors the
    shape of select_profiles_at_percentiles in regenerate_kfold_figures.py.
    """
    sort_idx = np.argsort(per_sample_rmse)
    n = len(per_sample_rmse)
    out = []
    for p in percentiles:
        rank = int(round(p / 100.0 * (n - 1)))
        i    = int(sort_idx[rank])
        out.append({
            'sample_idx': i,
            'rmse_um':    float(per_sample_rmse[i]),
            'percentile': float(p),
            'rank':       rank,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_percentile_panel(pred, pred_std, true_um, z_test_km,
                          selected: list[dict],
                          median_rmse_per_level: np.ndarray,
                          out_path: Path,
                          dpi: int = 500):
    """2x3 panel mirroring regenerate_kfold_figures.plot_example_profiles
    style: depth-from-cloud-top on the inverted y-axis, r_e on the x-axis,
    pred ± per-level median-RMSE band."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    true_color = 'black'
    pred_color = '#D55E00'   # vermillion (color-blind safe)
    band_alpha = 0.20

    n_levels = pred.shape[1]
    for k, (ax, sel) in enumerate(zip(axes.flat, selected)):
        i = sel['sample_idx']

        z_top_km, z_base_km = float(z_test_km[i, 0]), float(z_test_km[i, -1])
        thick_m = (z_top_km - z_base_km) * 1000.0
        z_norm  = np.linspace(0.0, 1.0, n_levels)
        z_m     = z_norm * thick_m   # depth from cloud top, in meters

        # True (7-level) — black markers + thin connector line
        ax.plot(true_um[i], z_m, 'o-', color=true_color, markersize=4,
                linewidth=0.9, label=fr'True ({n_levels}-level)')

        # Predicted ± median per-level RMSE band
        lo = pred[i] - median_rmse_per_level
        hi = pred[i] + median_rmse_per_level
        ax.fill_betweenx(z_m, lo, hi, color=pred_color, alpha=band_alpha,
                         linewidth=0,
                         label=r'NN $\pm$ median RMSE per level')
        ax.plot(pred[i], z_m, 's--', color=pred_color, markersize=4,
                linewidth=1.3, label='NN estimate')

        # Per-sample ±1σ from the network (thin error bars), shown for context.
        ax.errorbar(pred[i], z_m, xerr=pred_std[i],
                    fmt='none', ecolor=pred_color, elinewidth=0.7, capsize=2,
                    alpha=0.6)

        ax.invert_yaxis()
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r'$r_e$ ($\mu$m)')
        if k % 3 == 0:
            ax.set_ylabel('Depth from cloud top (m)')

        ax.set_title(
            fr'{int(sel["percentile"])}th percentile  '
            fr'(mean RMSE $= {sel["rmse_um"]:.2f}\ \mu$m)',
            fontsize=12,
        )
        if k == 0:
            ax.legend(loc='best', fontsize=8)

    fig.suptitle(
        'Predicted vs True Droplet Profiles — six examples '
        'spanning the per-sample RMSE distribution',
        fontsize=13, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if len(args.percentiles) != 6:
        raise ValueError(f'--percentiles must have exactly 6 values; got '
                         f'{len(args.percentiles)}: {args.percentiles}')

    results_dir = Path(args.results_dir).resolve()
    cfg_path    = results_dir / 'config.json'
    ckpt_path   = results_dir / 'best_model.pt'
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f'Expected config.json and best_model.pt under {results_dir}. '
            f'Confirm --results-dir points at a standalone trainer output dir.')

    with cfg_path.open() as f:
        cfg = json.load(f)
    hp = cfg['hyperparams']
    extras_active = cfg['extras_active']
    h5_path = Path(cfg['h5_path']).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 from saved config not found: {h5_path}')

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Banner
    print('=' * 70)
    print(f'  PERCENTILE PROFILE PANEL — {results_dir.name}')
    print('=' * 70)
    print(f'  config       : {cfg_path}')
    print(f'  HDF5         : {h5_path}')
    print(f'  device       : {device}')
    print(f'  variant      : {cfg.get("source_variant", "?")}, '
          f'run_id = {cfg.get("source_run_id", "?")}')
    print(f'  extras active: '
          f'tau_c={extras_active["tau_c"]} | '
          f'wv_above={extras_active["wv_above_cloud"]} | '
          f'wv_in={extras_active["wv_in_cloud"]}')
    print(f'  percentiles  : {args.percentiles}')
    print()

    # Build dataloaders matching training (same seed + same split sizes)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_loader, val_loader, test_loader = create_dataloaders_extras(
        h5_path=str(h5_path),
        instrument='hysics',
        batch_size=hp['batch_size'],
        num_workers=4,
        seed=args.seed,
        n_val_profiles=cfg['n_val_profiles'],
        n_test_profiles=cfg['n_test_profiles'],
        zero_tau_c=not extras_active['tau_c'],
        zero_wv_above=not extras_active['wv_above_cloud'],
        zero_wv_in=not extras_active['wv_in_cloud'],
    )

    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels

    # Build model and load checkpoint
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    model = ProfileOnlyNetworkExtras(model_config, n_extras=N_EXTRAS).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {int(ckpt["epoch"])}.')

    # Run inference
    results = predict_test(model, test_loader, device, model_config)
    pred     = results['pred']        # (n_test, n_levels) μm
    pred_std = results['pred_std']    # (n_test, n_levels) μm
    true     = results['true']        # (n_test, n_levels) μm
    print(f'Test set : {pred.shape[0]} samples × {pred.shape[1]} levels')

    # Per-sample mean RMSE for percentile selection
    err               = pred - true
    per_sample_rmse   = np.sqrt(np.mean(err ** 2, axis=1))
    per_level_median  = np.median(np.abs(err), axis=0)
    # (median absolute error per level — used as the shaded band; the
    # per-level RMSE could also be used. Both are global, not per-sample.)
    median_rmse_per_level = np.sqrt(np.median(err ** 2, axis=0))

    print(f'RMSE distribution: '
          f'min={per_sample_rmse.min():.3f}, '
          f'p10={np.percentile(per_sample_rmse, 10):.3f}, '
          f'median={np.median(per_sample_rmse):.3f}, '
          f'p85={np.percentile(per_sample_rmse, 85):.3f}, '
          f'max={per_sample_rmse.max():.3f} μm')

    # Profile selection at the requested percentiles
    selected = select_at_percentiles(per_sample_rmse,
                                     tuple(args.percentiles))
    print('\nSelected samples:')
    for s in selected:
        print(f'  pct {int(s["percentile"]):>2}  rank {s["rank"]:>5}  '
              f'sample {s["sample_idx"]:>5}  '
              f'mean RMSE = {s["rmse_um"]:.3f} μm')

    # Need per-sample altitude grids for the depth-from-top y-axis. The test
    # loader's underlying Subset preserves global HDF5 indices, so we can
    # read profiles_raw_z directly.
    subset = test_loader.dataset
    if not hasattr(subset, 'indices'):
        raise RuntimeError("Expected test_loader.dataset to be a torch Subset.")
    test_idx_global = np.asarray(subset.indices, dtype=int)
    sort_order   = np.argsort(test_idx_global)
    unsort_order = np.argsort(sort_order)
    sorted_idx   = test_idx_global[sort_order]

    import h5py
    with h5py.File(h5_path, 'r') as f:
        z_sorted = f['profiles_raw_z'][sorted_idx]
        n_lev_sorted = f['profile_n_levels'][sorted_idx]
    z_test_km    = z_sorted[unsort_order]               # (n_test, max_raw_lev)
    n_lev_test   = n_lev_sorted[unsort_order]
    # For synthetic clouds the per-cloud z grid is exactly N_FIXED_LEVELS=7 long;
    # use the first and last entries of the valid stretch to anchor cloud top
    # and base in plot_percentile_panel.
    z_test_km_packed = np.full((z_test_km.shape[0], n_levels), np.nan)
    for i in range(z_test_km.shape[0]):
        nL = int(n_lev_test[i])
        if nL == n_levels:
            z_test_km_packed[i] = z_test_km[i, :nL]
        else:
            # interpolation fallback if the stored raw grid differs from
            # the trained-on n_levels (shouldn't happen for synthetic data)
            z_top_km, z_base_km = float(z_test_km[i, 0]), float(z_test_km[i, nL - 1])
            z_test_km_packed[i] = np.linspace(z_top_km, z_base_km, n_levels)

    # Plot
    out_path = results_dir / args.output_name
    plot_percentile_panel(pred, pred_std, true, z_test_km_packed,
                          selected, median_rmse_per_level, out_path)
    print(f'\nFigure saved → {out_path}')


if __name__ == '__main__':
    main()
