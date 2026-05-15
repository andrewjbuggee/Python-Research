"""
plot_run004_figures.py — produces three diagnostic figures for a single-
split sweep run (e.g. sweep_results_profile_only_synthetic_M0_rev2/run_004):

  (1) per_cloud_rmse_paper_4panel.png       — IEEE double-column 2×2 with
        per-profile RMSE vs τ_c, above-cloud column water vapor (mm),
        Wood (2005) adiabaticity, and max r_e anywhere in the profile.
  (2) per_level_error_percentiles.png       — IEEE single-column figure
        with median |error|, 5–95 % percentile band, mean predicted σ.
  (3) profiles_true_vs_pred_percentiles.png — 2×3 of true-vs-pred profiles
        sampled at the 10/25/40/55/70/85 percentiles of per-profile RMSE.

The sweep summary.json (no top-level config.json) is parsed directly. The
test split is reproduced with seed=42 (sweep default). The first invocation
runs inference once, caches the result to pred_cache.npz inside the run
directory, and writes all three figures; subsequent invocations reuse the
cache.

Usage:
    python plot_run004_figures.py \\
        --run-dir ./hyper_parameter_sweep/\\
sweep_results_profile_only_synthetic_M0_rev2/run_004
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from plot_percentile_profiles_synthetic import (
    _resolve_h5_path, _pearson_r,
    select_at_percentiles, plot_percentile_panel,
)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with NaN/Inf masking. Returns NaN on too few
    valid pairs or zero-variance inputs.

    Spearman is the Pearson correlation of the rank-transformed data, so it
    captures *monotonic* (not just linear) association and is invariant to
    any order-preserving rescaling of either axis. Comparing |ρ| with |r|
    isolates how much of a Pearson correlation is driven by the linear
    structure versus the absolute scale.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    if x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float('nan')
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)
from plot_per_level_error_percentiles import (
    plot_per_level_error_percentiles, setup_style as setup_style_ieee_1col,
)
from plot_per_cloud_rmse_kfold import compute_predictors


N_EXTRAS = 3
DPI = 500
DEFAULT_PERCENTILES = (10, 25, 40, 55, 70, 85)

# Paper 4-panel figure: IEEE double-column 2×2, sized to fit the journal's
# 2-column page width. Each panel uses a distinct Okabe-Ito / Paul Tol
# CVD-safe color so a colorblind reader (deuteranopia / protanopia /
# tritanopia) can still tell them apart.
FIG_W_PAPER = 6.9     # IEEE 2-column width (just under the 7.0" max)
FIG_H_PAPER = 4.25    # compact vertical → ~2.1" per panel height

COLOR_TAUC         = '#0072B2'  # blue            (Okabe-Ito)
COLOR_WV_ABOVE     = '#CC79A7'  # reddish purple  (Okabe-Ito)
COLOR_WOOD         = '#117733'  # forest green    (Paul Tol "muted")
COLOR_RE_MAX       = '#882255'  # deep magenta    (Paul Tol "muted")
COLOR_RE_STD       = '#332288'  # indigo          (Paul Tol "muted")
COLOR_DRIZZLE_TOP  = '#56B4E9'  # sky blue        (Okabe-Ito)
COLOR_DRIZZLE_BOT  = '#D55E00'  # vermillion      (Okabe-Ito)

# 3x2 paper figure: same per-panel size as the 2x2, just stacked taller.
FIG_W_PAPER_6 = 6.9
FIG_H_PAPER_6 = 6.5


def setup_style_paper():
    """Serif style for IEEE double-column 2x2 paper figure (6.9" × 4.25")."""
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
        'axes.labelsize':    8.5,
        'axes.titlesize':    9,
        'figure.titlesize':  9.5,
        'legend.fontsize':   7.0,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'axes.linewidth':    0.6,
        'lines.linewidth':   1.0,
    })


# ─── Single-split inference for sweep run dirs ────────────────────────────
def _run_inference_sweep(run_dir: Path, h5_override: str | None,
                          device_arg: str | None, seed: int = 42) -> dict:
    """Reload sweep model, reproduce test split, predict. Returns arrays."""
    import torch
    from torch.utils.data import DataLoader
    from models                              import RetrievalConfig
    from models_profile_only_extras          import ProfileOnlyNetworkExtras
    from data                                import create_dataloaders_extras
    from train_standalone_profile_only_extras import predict_test

    with (run_dir / 'summary.json').open() as f:
        s = json.load(f)
    hp     = s['hyperparams']
    extras = s['extras']                                # sweep key
    h5_path = _resolve_h5_path(s['data']['h5_path'], h5_override)

    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif (getattr(torch.backends, 'mps', None) is not None
          and torch.backends.mps.is_available()):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'  HDF5   : {h5_path}')
    print(f'  device : {device}')
    print(f'  n_val = {s["n_val_profiles"]}, n_test = {s["n_test_profiles"]}, '
          f'seed = {seed}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _, test_loader = create_dataloaders_extras(
        h5_path=str(h5_path),
        instrument=s['data'].get('instrument', 'hysics'),
        batch_size=hp['batch_size'],
        num_workers=s['data'].get('num_workers', 4),
        seed=seed,
        n_val_profiles=s['n_val_profiles'],
        n_test_profiles=s['n_test_profiles'],
        zero_tau_c=extras['zero_tau_c'],
        zero_wv_above=extras['zero_wv_above'],
        zero_wv_in=extras['zero_wv_in'],
    )
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels

    model_cfg = RetrievalConfig(
        n_wavelengths=636, n_geometry_inputs=4, n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)
    ckpt = torch.load(run_dir / 'best_model.pt',
                       map_location=device, weights_only=False)
    # Sweep saves the state_dict directly; standalone wraps it in
    # {'model_state_dict': ..., 'epoch': ...}. Handle both.
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    print(f'  Loaded checkpoint  (sweep best_epoch = {int(s["best_epoch"])})')

    # Recover global HDF5 indices of the test set so we can look up predictors.
    subset = test_loader.dataset
    if not hasattr(subset, 'indices'):
        raise RuntimeError('Expected test_loader.dataset to be a torch Subset.')
    test_idx_global = np.asarray(subset.indices, dtype=np.int64)

    res = predict_test(model, test_loader, device, model_cfg)
    return {
        'pred':     res['pred'],
        'pred_std': res['pred_std'],
        'true':     res['true'],
        'h5_idx':   test_idx_global,
        'h5_path':  str(h5_path),
    }


# ─── (1) Paper-ready 4-panel figure (IEEE 2-column, 2x2) ──────────────────
def plot_paper_4panel(per_sample_rmse: np.ndarray, predictors: dict,
                       n_test: int, out_path: Path) -> None:
    """IEEE 2-column 2x2:  τ_c | WV_above (mm)  /  adiabaticity | max r_e.

    Each panel uses a distinct CVD-safe color so colorblind readers can still
    tell the predictors apart.
    """
    setup_style_paper()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_PAPER, FIG_H_PAPER))

    def _scatter(ax, x, name, color, label, xlog=False):
        valid = np.isfinite(x) & np.isfinite(per_sample_rmse)
        ax.scatter(x[valid], per_sample_rmse[valid], s=6, alpha=0.55,
                   color=color, edgecolors='none')
        if xlog:
            ax.set_xscale('log')
        r   = _pearson_r(x, per_sample_rmse)
        rho = _spearman_rho(x, per_sample_rmse)
        ax.set_xlabel(name)
        ax.set_ylabel('Per-profile RMSE (μm)')
        # Two-line title so both correlations fit at 8.5 pt within ~3.4" panels.
        ax.set_title(fr'({label})  Pearson $r$ = {r:+.3f}'  '\n'
                     fr'      Spearman $\rho$ = {rho:+.3f}',
                     fontsize=8.5)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    _scatter(axes[0, 0], predictors['tau_c'],
             r'$\tau_c$', COLOR_TAUC, 'a', xlog=True)
    _scatter(axes[0, 1], predictors['wv_above_mm'],
             'above-cloud column water vapor (mm)', COLOR_WV_ABOVE, 'b')
    _scatter(axes[1, 0], predictors['wood_adiab'],
             r'adiabaticity:  '
             r'$\mathrm{LWP}_{\mathrm{obs}}/\mathrm{LWP}_{\mathrm{ad}}$',
             COLOR_WOOD, 'c')
    _scatter(axes[1, 1], predictors['re_max'],
             r'max $r_e$ anywhere in profile ($\mu$m)',
             COLOR_RE_MAX, 'd')

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ─── (1b) Paper-ready 6-panel figure (IEEE 2-column, 3x2) ────────────────
def plot_paper_6panel(per_sample_rmse: np.ndarray, predictors: dict,
                       n_test: int, out_path: Path) -> None:
    """IEEE 2-column 3x2 paper figure with Pearson r and Spearman ρ in titles.

    Layout:
        (a) τ_c                       (b) above-cloud column water vapor (mm)
        (c) adiabaticity              (d) std of r_e profile (μm)
        (e) max r_e in upper 30 %     (f) max r_e in lower 30 %

    Same formatting conventions as plot_paper_4panel (CVD-safe colors, two-
    line titles, same scatter density).
    """
    setup_style_paper()
    fig, axes = plt.subplots(3, 2, figsize=(FIG_W_PAPER_6, FIG_H_PAPER_6))

    def _scatter(ax, x, name, color, label, xlog=False):
        valid = np.isfinite(x) & np.isfinite(per_sample_rmse)
        ax.scatter(x[valid], per_sample_rmse[valid], s=6, alpha=0.55,
                   color=color, edgecolors='none')
        if xlog:
            ax.set_xscale('log')
        r   = _pearson_r(x, per_sample_rmse)
        rho = _spearman_rho(x, per_sample_rmse)
        ax.set_xlabel(name)
        ax.set_ylabel('Per-profile RMSE (μm)')
        ax.set_title(fr'({label})  Pearson $r$ = {r:+.3f}'  '\n'
                     fr'      Spearman $\rho$ = {rho:+.3f}',
                     fontsize=8.5)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    _scatter(axes[0, 0], predictors['tau_c'],
             r'$\tau_c$', COLOR_TAUC, 'a', xlog=True)
    _scatter(axes[0, 1], predictors['wv_above_mm'],
             'above-cloud column water vapor (mm)',
             COLOR_WV_ABOVE, 'b')
    _scatter(axes[1, 0], predictors['wood_adiab'],
             r'adiabaticity:  '
             r'$\mathrm{LWP}_{\mathrm{obs}}/\mathrm{LWP}_{\mathrm{ad}}$',
             COLOR_WOOD, 'c')
    _scatter(axes[1, 1], predictors['re_std'],
             r'std of $r_e$ profile ($\mu$m)',
             COLOR_RE_STD, 'd')
    _scatter(axes[2, 0], predictors['drizzle_proxy_top'],
             r'max $r_e$ in upper 30 % ($\mu$m)',
             COLOR_DRIZZLE_TOP, 'e')
    _scatter(axes[2, 1], predictors['drizzle_proxy'],
             r'max $r_e$ in lower 30 % ($\mu$m)',
             COLOR_DRIZZLE_BOT, 'f')

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def print_droplet_size_correlation_table(predictors: dict) -> None:
    """Print a Pearson-r matrix for the four droplet-size summary scalars.

    Helps the reader decide which panels carry redundant information
    versus which are genuinely distinct. Computed on the run_004 test set.
    """
    keys   = ['re_max',          'drizzle_proxy_top', 'drizzle_proxy', 're_std']
    labels = ['max r_e anywhere','max r_e top 30 %',  'max r_e bot 30 %', 'std r_e']
    width  = 18
    print()
    print('Pairwise Pearson r between droplet-size predictors (test set):')
    print('-' * (width + width * len(labels)))
    header = ' ' * width + ''.join(f'{lab:>{width}s}' for lab in labels)
    print(header)
    for i, ki in enumerate(keys):
        row = f'{labels[i]:<{width}s}'
        for kj in keys:
            r = _pearson_r(predictors[ki], predictors[kj])
            row += f'{r:>+{width}.4f}'
        print(row)
    print('-' * (width + width * len(labels)))


# ─── (3) Percentile profile panel ─────────────────────────────────────────
def plot_profiles_at_percentiles(pred, pred_std, true, h5_idx, h5_path,
                                  per_sample_rmse, out_path,
                                  percentiles=DEFAULT_PERCENTILES,
                                  pick_seed=None, pick_jitter_pct=1.0):
    """Build z-grid per sample and reuse plot_percentile_panel.

    pick_seed=None gives the deterministic closest-rank pick at each target
    percentile. Pass an integer to add ±pick_jitter_pct (in percentile points)
    of uniform jitter to each target, so re-running with a different seed
    shows different example profiles for the same percentile bin.
    """
    n_levels = pred.shape[1]
    sort_order = np.argsort(h5_idx)
    unsort     = np.argsort(sort_order)
    sorted_idx = h5_idx[sort_order]

    with h5py.File(h5_path, 'r') as f:
        n_lev_s = f['profile_n_levels'][sorted_idx]
        z_s     = f['profiles_raw_z'][sorted_idx]

    n_lev_test = n_lev_s[unsort]
    z_test_km  = z_s[unsort]

    z_packed = np.full((len(h5_idx), n_levels), np.nan)
    for i in range(len(h5_idx)):
        nL = int(n_lev_test[i])
        if nL == n_levels:
            z_packed[i] = z_test_km[i, :nL]
        else:
            z_top, z_base = float(z_test_km[i, 0]), float(z_test_km[i, nL - 1])
            z_packed[i] = np.linspace(z_top, z_base, n_levels)

    err = pred - true
    median_rmse_per_level = np.sqrt(np.median(err ** 2, axis=0))

    selected = select_at_percentiles(per_sample_rmse, tuple(percentiles),
                                     seed=pick_seed,
                                     jitter_pct=pick_jitter_pct)
    plot_percentile_panel(pred, pred_std, true, z_packed, selected,
                           median_rmse_per_level, out_path, dpi=DPI)
    return selected


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-dir', required=True,
                   help='Sweep run directory (e.g. .../run_004).')
    p.add_argument('--refresh', action='store_true',
                   help='Force re-inference even if pred_cache.npz exists.')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the seed used by the sweep (default 42).')
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in summary.json.')
    p.add_argument('--pick-seed', type=int, default=None,
                   help='Seed for jittering the percentile-profile picks. '
                        'Default (None) uses the deterministic closest-rank '
                        'sample at each percentile. Pass an integer to draw '
                        'a different set of example profiles for the same '
                        'percentile bins.')
    p.add_argument('--pick-jitter-pct', type=float, default=1.0,
                   help='Half-width of the percentile jitter, in percentile '
                        'points (default 1.0 → ±1 pp). Only used when '
                        '--pick-seed is set.')
    p.add_argument('--profiles-output', default='profiles_true_vs_pred_percentiles.png',
                   help='Filename for the percentile-profile figure inside '
                        'the run directory.')
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f'run-dir does not exist: {run_dir}')
    cache_path = run_dir / 'pred_cache.npz'

    if cache_path.exists() and not args.refresh:
        print(f'Loading cached predictions from {cache_path}')
        c = np.load(cache_path)
        pred, pred_std, true, h5_idx = c['pred'], c['pred_std'], c['true'], c['h5_idx']
        with (run_dir / 'summary.json').open() as f:
            s = json.load(f)
        h5_path = _resolve_h5_path(s['data']['h5_path'], args.h5_path)
    else:
        print('Running inference on the saved test split...')
        out = _run_inference_sweep(run_dir, args.h5_path, args.device, args.seed)
        pred, pred_std, true = out['pred'], out['pred_std'], out['true']
        h5_idx  = out['h5_idx']
        h5_path = Path(out['h5_path'])
        np.savez_compressed(cache_path,
                            pred=pred, pred_std=pred_std, true=true,
                            h5_idx=h5_idx)
        print(f'Cached predictions -> {cache_path}')

    print(f'Test set : {pred.shape[0]:,} samples x {pred.shape[1]} levels')

    err = pred - true
    per_sample_rmse = np.sqrt(np.mean(err ** 2, axis=1))

    print('Computing predictors (τ_c, WV, Wood adiab, max r_e, ...)...')
    predictors = compute_predictors(h5_idx, h5_path)

    # (1) Paper 4-panel
    out_paper = run_dir / 'per_cloud_rmse_paper_4panel.png'
    plot_paper_4panel(per_sample_rmse, predictors, pred.shape[0], out_paper)
    print(f'(1) paper 4-panel       -> {out_paper}')

    # (1b) Paper 6-panel (alternative 3x2 with std r_e and the two
    # vertically-localized drizzle proxies)
    out_paper_6 = run_dir / 'per_cloud_rmse_paper_6panel.png'
    plot_paper_6panel(per_sample_rmse, predictors, pred.shape[0], out_paper_6)
    print(f'(1b) paper 6-panel      -> {out_paper_6}')

    # Pairwise correlation table between droplet-size predictors
    print_droplet_size_correlation_table(predictors)

    # (2) Per-level error percentiles (IEEE single column)
    setup_style_ieee_1col()
    out_per_level = run_dir / 'per_level_error_percentiles.png'
    stats = plot_per_level_error_percentiles(pred, true, pred_std, out_per_level)
    print(f'(2) per-level error     -> {out_per_level}')

    # Per-level absolute-error summary (printed for paper reporting).
    print()
    print('Per-level absolute error (test set, μm):')
    print('  ' + '-' * 60)
    print(f'  {"level":>6s}  {"median |err|":>14s}  '
          f'{"5th pct":>10s}  {"95th pct":>10s}  {"mean pred σ":>14s}')
    print('  ' + '-' * 60)
    for L, med, p05, p95, sig in zip(stats['levels'],
                                      stats['abs_err_median'],
                                      stats['abs_err_p05'],
                                      stats['abs_err_p95'],
                                      stats['mean_predicted_sigma']):
        print(f'  {L:>6d}  {med:>14.4f}  {p05:>10.4f}  {p95:>10.4f}  '
              f'{sig:>14.4f}')
    print('  ' + '-' * 60)

    # (3) True-vs-pred profile percentiles
    out_profiles = run_dir / args.profiles_output
    selected = plot_profiles_at_percentiles(
        pred, pred_std, true, h5_idx, h5_path,
        per_sample_rmse, out_profiles,
        pick_seed=args.pick_seed,
        pick_jitter_pct=args.pick_jitter_pct,
    )
    if args.pick_seed is None:
        pick_note = '(deterministic closest-rank picks)'
    else:
        pick_note = (f'(jittered ±{args.pick_jitter_pct} pp, '
                     f'seed = {args.pick_seed})')
    print(f'(3) percentile profiles -> {out_profiles}  {pick_note}')
    print('    Picked sample indices and RMSEs:')
    for s in selected:
        print(f'      pct {int(s["percentile"]):>2d}  '
              f'sample {s["sample_idx"]:>5}  '
              f'RMSE = {s["rmse_um"]:.3f} μm')


if __name__ == '__main__':
    main()
