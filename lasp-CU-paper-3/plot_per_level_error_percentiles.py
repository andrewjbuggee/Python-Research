"""
Replot per-level error metrics from a saved standalone synthetic run as an
IEEE single-column figure. Shows:

    - Median |error| per level                       (line)
    - 5th–95th percentile band of |error| per level  (shaded fill)
    - Mean predicted σ per level (aleatoric)         (dashed line)

All three on a single 3.25"-wide axes with Okabe-Ito CVD-safe colors
(vermillion #D55E00 for the error metric, Wong blue #0072B2 for the
predicted σ). DPI 500. Y-axis is the vertical level with 1 = cloud top
inverted to the top of the plot, n = cloud base at the bottom.

The first run loads `config.json` and `best_model.pt` from the results
directory, reproduces the same test split that training used (same seed
and sizes from config.json), runs the model once, and saves the resulting
(pred, pred_std, true) arrays to pred_cache.npz inside the results
directory. Subsequent runs use the cache and skip the HDF5 + model load,
so this script works on a laptop that does not have the training HDF5
mounted.

Usage:
    python plot_per_level_error_percentiles.py \\
        --results-dir ./standalone_results_profile_only_synthetic/\\
M0_run098_synthetic_training_data_7-levels_8_May_2026_atest_rev2

    # Force re-inference (ignore cache):
    python plot_per_level_error_percentiles.py --results-dir <...> --refresh
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))


# ─── IEEE single-column figure config ─────────────────────────────────────
FIG_WIDTH_IN  = 3.25     # IEEE single column
FIG_HEIGHT_IN = 3.8
DPI           = 500

# Okabe-Ito CVD-safe palette (Wong 2011, Nat. Methods).
COLOR_ERROR  = '#D55E00'  # vermillion — median |error| + 5–95% band
COLOR_SIGMA  = '#0072B2'  # Wong blue — mean predicted σ (aleatoric)
BAND_ALPHA   = 0.25

PERCENTILES = (5, 50, 95)
N_EXTRAS    = 3            # matches train_standalone_profile_only_synthetic


def setup_style():
    """Compact serif style for IEEE single column."""
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
        'axes.labelsize':    9,
        'axes.titlesize':    9.5,
        'figure.titlesize':  10,
        'legend.fontsize':   7,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'axes.linewidth':    0.6,
        'lines.linewidth':   1.3,
    })


def plot_per_level_error_percentiles(pred: np.ndarray, true: np.ndarray,
                                     pred_std: np.ndarray,
                                     out_path: Path) -> dict:
    """Single-axes per-level error figure. Returns the plotted summary stats."""
    abs_err   = np.abs(pred - true)                        # (n_samp, n_lev)
    n_levels  = pred.shape[1]
    levels    = np.arange(1, n_levels + 1)

    p05, p50, p95 = np.percentile(abs_err, PERCENTILES, axis=0)
    sigma_mean    = pred_std.mean(axis=0)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # 5-95% shaded band of |error|
    ax.fill_betweenx(levels, p05, p95,
                     color=COLOR_ERROR, alpha=BAND_ALPHA, linewidth=0,
                     label='5–95% of |error|')
    # Median |error|
    ax.plot(p50, levels, 'o-', color=COLOR_ERROR,
            linewidth=1.5, markersize=4.5,
            label='Median |error|')
    # Mean predicted σ (aleatoric)
    ax.plot(sigma_mean, levels, 'd--', color=COLOR_SIGMA,
            linewidth=1.2, markersize=4, alpha=0.95,
            label=r'Mean predicted $\sigma$ (aleatoric)')

    ax.set_xlabel(r'Per-level error ($\mu$m)')
    ax.set_ylabel(f'Vertical level (1 = cloud top, {n_levels} = cloud base)')
    ax.set_yticks(levels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.legend(loc='center right', frameon=True, framealpha=0.9,
              handlelength=1.6, labelspacing=0.35,
              handletextpad=0.5, borderaxespad=0.3)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    return {
        'levels':                 levels.tolist(),
        'abs_err_p05':            p05.tolist(),
        'abs_err_median':         p50.tolist(),
        'abs_err_p95':            p95.tolist(),
        'mean_predicted_sigma':   sigma_mean.tolist(),
        'n_test_samples':         int(pred.shape[0]),
    }


def _run_inference(results_dir: Path, h5_override: str | None,
                   device_arg: str | None, seed: int) -> dict:
    """Load best_model.pt + reproduce the test split + predict.

    Heavy imports are local so the script can still load a cached .npz on a
    laptop without torch / data dependencies installed correctly.
    """
    import torch

    from plot_percentile_profiles_synthetic import _resolve_h5_path
    from models                              import RetrievalConfig
    from models_profile_only_extras          import ProfileOnlyNetworkExtras
    from data                                import create_dataloaders_extras
    from train_standalone_profile_only_extras import predict_test

    cfg_path  = results_dir / 'config.json'
    ckpt_path = results_dir / 'best_model.pt'
    with cfg_path.open() as f:
        cfg = json.load(f)
    hp            = cfg['hyperparams']
    extras_active = cfg['extras_active']
    h5_path       = _resolve_h5_path(cfg['h5_path'], h5_override)

    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'  HDF5   : {h5_path}')
    print(f'  device : {device}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _, test_loader = create_dataloaders_extras(
        h5_path=str(h5_path),
        instrument='hysics',
        batch_size=hp['batch_size'],
        num_workers=4,
        seed=seed,
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

    model_cfg = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'  Loaded checkpoint from epoch {int(ckpt["epoch"])}.')

    results = predict_test(model, test_loader, device, model_cfg)
    return {
        'pred':     results['pred'],
        'pred_std': results['pred_std'],
        'true':     results['true'],
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--results-dir', required=True,
                   help='Standalone trainer output dir containing '
                        'config.json + best_model.pt.')
    p.add_argument('--output-name', default='per_level_error_percentiles.png',
                   help='Filename for the saved figure inside the results dir.')
    p.add_argument('--refresh', action='store_true',
                   help='Force re-inference even if pred_cache.npz exists.')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None,
                   help='Override device for inference.')
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the training seed (default 42).')
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in config.json '
                        '(useful when replotting on a different machine).')
    return p.parse_args()


def main():
    args = parse_args()
    setup_style()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f'results-dir does not exist: {results_dir}')

    cache_path = results_dir / 'pred_cache.npz'
    if cache_path.exists() and not args.refresh:
        print(f'Loading cached predictions from {cache_path}')
        cached = np.load(cache_path)
        pred, pred_std, true = cached['pred'], cached['pred_std'], cached['true']
    else:
        cfg_path  = results_dir / 'config.json'
        ckpt_path = results_dir / 'best_model.pt'
        if not (cfg_path.exists() and ckpt_path.exists()):
            raise FileNotFoundError(
                f'Need both config.json and best_model.pt under {results_dir} '
                f'to run inference. (Or pre-stage pred_cache.npz.)'
            )
        print(f'Running inference on the saved test split...')
        out = _run_inference(results_dir, args.h5_path, args.device, args.seed)
        pred, pred_std, true = out['pred'], out['pred_std'], out['true']
        np.savez_compressed(cache_path,
                            pred=pred, pred_std=pred_std, true=true)
        print(f'Cached predictions -> {cache_path}')

    print(f'Test set : {pred.shape[0]} samples x {pred.shape[1]} levels')

    out_fig = results_dir / args.output_name
    stats = plot_per_level_error_percentiles(pred, true, pred_std, out_fig)
    print(f'Figure   -> {out_fig}  (dpi={DPI})')

    # Save the numeric summary alongside the figure for traceability.
    stats_path = out_fig.with_suffix('.json')
    with stats_path.open('w') as f:
        json.dump(stats, f, indent=2)
    print(f'Summary  -> {stats_path}')


if __name__ == '__main__':
    main()
