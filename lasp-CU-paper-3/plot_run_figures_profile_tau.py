"""
plot_run_figures_profile_tau.py — diagnostic figures for a single run of the
PROFILE + τ_c sweep (sweep_results_profile_tau_synthetic/run_NNN).

Produces the same figure set as plot_run004_figures.py did for the published
profile-only run_004 — byte-compatible layouts, reusing those plotting
functions directly — plus a τ-retrieval diagnostic figure that has no
profile-only counterpart:

  (1)  per_cloud_rmse_paper_4panel.png       — IEEE 2-column 2×2
  (1b) per_cloud_rmse_paper_6panel.png       — IEEE 2-column 3×2, the layout of
         published_figures/per_cloud_rmse_paper_6panel_run004_12_May_2026.png
  (2)  per_level_error_percentiles.png       — IEEE single-column
  (3)  profiles_true_vs_pred_percentiles.png — 2×3 percentile profiles
  (4)  tau_retrieval_diagnostics.png         — 2×2: pred-vs-true τ, τ error vs
         τ, relative-error histogram, and per-profile r_e RMSE vs |τ error|
         (does the model fail on the same clouds for both tasks?)

The first invocation runs inference once, caches to pred_cache.npz inside the
run directory (now including the τ arrays), and writes all figures;
subsequent invocations reuse the cache.

Usage:
    python plot_run_figures_profile_tau.py \\
        --run-dir ./hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004

Author: Andrew J. Buggee, LASP / CU Boulder
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
sys.path.insert(0, str(repo_root / 'hyper_parameter_sweep'))

from plot_percentile_profiles_synthetic import _resolve_h5_path, _pearson_r
from plot_per_level_error_percentiles import (
    plot_per_level_error_percentiles, setup_style as setup_style_ieee_1col,
)
from plot_per_cloud_rmse_kfold import compute_predictors
from plot_run004_figures import (
    plot_paper_4panel, plot_paper_6panel, plot_profiles_at_percentiles,
    print_droplet_size_correlation_table, setup_style_paper, _spearman_rho,
)

DPI = 500
DEFAULT_PERCENTILES = (10, 25, 40, 55, 70, 85)

# CVD-safe (Okabe-Ito / Paul Tol) — consistent with the run_004 figures
COLOR_TAU_SCATTER = '#0072B2'   # blue
COLOR_TAU_ERR     = '#D55E00'   # vermillion
COLOR_TAU_HIST    = '#009E73'   # bluish green
COLOR_CROSS       = '#882255'   # deep magenta


# ─── Inference on the run's saved test split ─────────────────────────────
def _run_inference(run_dir: Path, h5_override: str | None,
                   device_arg: str | None, seed: int = 42) -> dict:
    """Reload the PT-sweep model, reproduce the test split, predict."""
    import torch
    from models import DropletProfileNetwork, RetrievalConfig
    from data import create_dataloaders, TAU_MIN, TAU_MAX
    from sweep_train_profile_tau_synthetic import predict_test

    with (run_dir / 'summary.json').open() as f:
        s = json.load(f)
    hp            = s['hyperparams']
    tau_transform = s.get('tau_transform', hp.get('tau_transform', 'linear'))
    h5_path       = _resolve_h5_path(s['data']['h5_path'], h5_override)

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
          f'seed = {seed}, tau_transform = {tau_transform}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _, test_loader = create_dataloaders(
        h5_path=str(h5_path),
        instrument=s['data'].get('instrument', 'hysics'),
        batch_size=hp['batch_size'],
        num_workers=s['data'].get('num_workers', 4),
        seed=seed,
        profile_holdout=True,
        n_val_profiles=s['n_val_profiles'],
        n_test_profiles=s['n_test_profiles'],
    )
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels

    # Same config the trainer built — τ bounds pinned to the dataset constants
    # (see the τ SCALING note in sweep_train_profile_tau_synthetic.py).
    model_cfg = RetrievalConfig(
        n_wavelengths=636, n_geometry_inputs=4, n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
        tau_min=float(TAU_MIN), tau_max=float(TAU_MAX),
    )
    model = DropletProfileNetwork(model_cfg).to(device)
    ckpt = torch.load(run_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict)
                                         and 'model_state_dict' in ckpt) else ckpt
    model.load_state_dict(state)
    print(f'  Loaded checkpoint  (sweep best_epoch = {int(s["best_epoch"])})')

    subset = test_loader.dataset
    if not hasattr(subset, 'indices'):
        raise RuntimeError('Expected test_loader.dataset to be a torch Subset.')
    test_idx_global = np.asarray(subset.indices, dtype=np.int64)

    res = predict_test(model, test_loader, device, model_cfg, tau_transform)
    res['h5_idx']  = test_idx_global
    res['h5_path'] = str(h5_path)
    return res


# ─── (4) τ retrieval diagnostics ─────────────────────────────────────────
def plot_tau_diagnostics(tau_pred: np.ndarray, tau_true: np.ndarray,
                         tau_pred_std: np.ndarray,
                         per_sample_rmse: np.ndarray,
                         tau_transform: str, out_path: Path) -> None:
    """IEEE 2-column 2×2 τ diagnostics.

    (a) predicted vs true τ (log-log) with the 1:1 line
    (b) τ error vs true τ (log x)
    (c) histogram of relative τ error (%)
    (d) per-profile r_e RMSE vs |τ error| — cross-task failure correlation
    """
    setup_style_paper()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.0))

    err     = tau_pred - tau_true
    rel_pct = 100.0 * err / np.maximum(tau_true, 1e-9)
    rmse    = float(np.sqrt(np.mean(err ** 2)))
    bias    = float(np.mean(err))
    mape    = float(np.mean(np.abs(rel_pct)))
    var     = float(np.var(tau_true))
    r2      = 1.0 - float(np.mean(err ** 2)) / var if var > 0 else float('nan')

    # (a) pred vs true, log-log
    ax = axes[0, 0]
    ax.scatter(tau_true, tau_pred, s=6, alpha=0.35, color=COLOR_TAU_SCATTER,
               edgecolors='none')
    lims = (min(tau_true.min(), tau_pred.min()) * 0.9,
            max(tau_true.max(), tau_pred.max()) * 1.1)
    ax.plot(lims, lims, color='k', lw=0.8, ls='--', label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims);    ax.set_ylim(lims)
    ax.set_xlabel(r'true $\tau_c$')
    ax.set_ylabel(r'retrieved $\tau_c$')
    ax.set_title(fr'(a)  RMSE = {rmse:.2f},  bias = {bias:+.2f}'  '\n'
                 fr'      MAPE = {mape:.1f} %,  $R^2$ = {r2:.3f}',
                 fontsize=8.5)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(alpha=0.3, which='both')

    # (b) error vs true τ
    ax = axes[0, 1]
    ax.scatter(tau_true, err, s=6, alpha=0.35, color=COLOR_TAU_ERR,
               edgecolors='none')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'true $\tau_c$')
    ax.set_ylabel(r'$\tau_c$ error (retrieved $-$ true)')
    r   = _pearson_r(tau_true, err)
    rho = _spearman_rho(tau_true, err)
    ax.set_title(fr'(b)  Pearson $r$ = {r:+.3f}'  '\n'
                 fr'      Spearman $\rho$ = {rho:+.3f}', fontsize=8.5)
    ax.grid(alpha=0.3, which='both')

    # (c) relative error histogram
    ax = axes[1, 0]
    clip = np.clip(rel_pct, *np.percentile(rel_pct, [0.5, 99.5]))
    ax.hist(clip, bins=60, color=COLOR_TAU_HIST, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    med = float(np.median(rel_pct))
    p05, p95 = np.percentile(rel_pct, [5, 95])
    ax.set_xlabel(r'relative $\tau_c$ error (%)')
    ax.set_ylabel('count')
    ax.set_title(fr'(c)  median = {med:+.1f} %'  '\n'
                 fr'      5–95 % : [{p05:+.1f}, {p95:+.1f}] %', fontsize=8.5)
    ax.grid(alpha=0.3)

    # (d) cross-task: per-profile r_e RMSE vs |τ error|
    ax = axes[1, 1]
    abs_err = np.abs(err)
    ax.scatter(abs_err, per_sample_rmse, s=6, alpha=0.35, color=COLOR_CROSS,
               edgecolors='none')
    r   = _pearson_r(abs_err, per_sample_rmse)
    rho = _spearman_rho(abs_err, per_sample_rmse)
    ax.set_xlabel(r'$|\tau_c$ error$|$')
    ax.set_ylabel(r'per-profile $r_e$ RMSE (μm)')
    ax.set_title(fr'(d)  Pearson $r$ = {r:+.3f}'  '\n'
                 fr'      Spearman $\rho$ = {rho:+.3f}', fontsize=8.5)
    ax.grid(alpha=0.3)

    fig.suptitle(fr'$\tau_c$ retrieval diagnostics — test set '
                 fr'({len(tau_true):,} profiles, {tau_transform} loss space)',
                 fontsize=9.5, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-dir', required=True,
                   help='PT-sweep run directory (e.g. .../run_004).')
    p.add_argument('--refresh', action='store_true',
                   help='Force re-inference even if pred_cache.npz exists.')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the seed used by the sweep (default 42).')
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in summary.json.')
    p.add_argument('--pick-seed', type=int, default=None,
                   help='Seed for jittering the percentile-profile picks '
                        '(None = deterministic closest-rank picks).')
    p.add_argument('--pick-jitter-pct', type=float, default=1.0)
    p.add_argument('--profiles-output',
                   default='profiles_true_vs_pred_percentiles.png')
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f'run-dir does not exist: {run_dir}')
    cache_path = run_dir / 'pred_cache.npz'

    with (run_dir / 'summary.json').open() as f:
        s = json.load(f)
    tau_transform = s.get('tau_transform',
                          s['hyperparams'].get('tau_transform', 'linear'))

    need_refresh = args.refresh
    if cache_path.exists() and not need_refresh:
        c = np.load(cache_path)
        if 'tau_pred' not in c.files:      # cache from before the τ arrays
            need_refresh = True
        else:
            print(f'Loading cached predictions from {cache_path}')
            pred, pred_std, true = c['pred'], c['pred_std'], c['true']
            tau_pred, tau_pred_std, tau_true = (c['tau_pred'],
                                                c['tau_pred_std'],
                                                c['tau_true'])
            h5_idx  = c['h5_idx']
            h5_path = _resolve_h5_path(s['data']['h5_path'], args.h5_path)

    if not cache_path.exists() or need_refresh:
        print('Running inference on the saved test split...')
        out = _run_inference(run_dir, args.h5_path, args.device, args.seed)
        pred, pred_std, true = out['pred'], out['pred_std'], out['true']
        tau_pred, tau_pred_std, tau_true = (out['tau_pred'],
                                            out['tau_pred_std'],
                                            out['tau_true'])
        h5_idx  = out['h5_idx']
        h5_path = Path(out['h5_path'])
        np.savez_compressed(cache_path,
                            pred=pred, pred_std=pred_std, true=true,
                            tau_pred=tau_pred, tau_pred_std=tau_pred_std,
                            tau_true=tau_true, h5_idx=h5_idx)
        print(f'Cached predictions -> {cache_path}')

    print(f'Test set : {pred.shape[0]:,} profiles x {pred.shape[1]} levels')

    err = pred - true
    per_sample_rmse = np.sqrt(np.mean(err ** 2, axis=1))

    print('Computing predictors (τ_c, WV, Wood adiab, max r_e, ...)...')
    predictors = compute_predictors(h5_idx, h5_path)

    # (1) Paper 4-panel — identical layout to the published run_004 figure
    out_paper = run_dir / 'per_cloud_rmse_paper_4panel.png'
    plot_paper_4panel(per_sample_rmse, predictors, pred.shape[0], out_paper)
    print(f'(1)  paper 4-panel      -> {out_paper}')

    # (1b) Paper 6-panel — the published_figures/..._6panel layout
    out_paper_6 = run_dir / 'per_cloud_rmse_paper_6panel.png'
    plot_paper_6panel(per_sample_rmse, predictors, pred.shape[0], out_paper_6)
    print(f'(1b) paper 6-panel      -> {out_paper_6}')

    print_droplet_size_correlation_table(predictors)

    # (2) Per-level error percentiles (IEEE single column)
    setup_style_ieee_1col()
    out_per_level = run_dir / 'per_level_error_percentiles.png'
    stats = plot_per_level_error_percentiles(pred, true, pred_std, out_per_level)
    print(f'(2)  per-level error    -> {out_per_level}')

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
        percentiles=DEFAULT_PERCENTILES,
        pick_seed=args.pick_seed,
        pick_jitter_pct=args.pick_jitter_pct,
    )
    print(f'(3)  percentile profiles -> {out_profiles}')
    for sel in selected:
        print(f'      pct {int(sel["percentile"]):>2d}  '
              f'sample {sel["sample_idx"]:>5}  '
              f'RMSE = {sel["rmse_um"]:.3f} μm')

    # (4) τ diagnostics — new for the profile+τ model
    out_tau = run_dir / 'tau_retrieval_diagnostics.png'
    plot_tau_diagnostics(tau_pred, tau_true, tau_pred_std,
                         per_sample_rmse, tau_transform, out_tau)
    print(f'(4)  tau diagnostics    -> {out_tau}')


if __name__ == '__main__':
    main()
