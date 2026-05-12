"""
Replot the per-cloud RMSE diagnostics for a k-fold run with full coverage.

Aggregates the test-set predictions of every fold (so all N_total samples
get a prediction from a model that never saw them in training), then plots
per-sample mean RMSE against six predictors:

    Row 1: τ_c (log)                       | adiabaticity (Pearson r of r_e^3 vs z-z_base)
    Row 2: drizzle proxy (max r_e bot 30%) | CDF of per-sample RMSE
    Row 3: above-cloud PWV (mm)            | in-cloud PWV (mm)

The water-vapor columns are stored in the HDF5 as molec/cm² and converted
here to precipitable-water millimeters (multiply by 2.991e-22).

First run: loops fold_01..fold_K, reloads each best_model.pt, rebuilds the
fold's test split with `make_kfold_splits(seed=42)` to match training, runs
the model on MPS/CUDA/CPU, and writes agg_pred_cache.npz inside the kfold
directory. Subsequent runs reuse the cache.

Usage:
    python plot_per_cloud_rmse_kfold.py \\
        --kfold-dir ./standalone_results_profile_only_synthetic/\\
M0_run098_synthetic_training_data_7-levels_8_May_2026_20kfold_rev3
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from plot_percentile_profiles_synthetic import (
    adiabaticity_pearson_r,
    _pearson_r,
    _resolve_h5_path,
)


N_EXTRAS = 3
DPI      = 500

# Conversion: column number density (molec/cm²) → precipitable water (mm).
# 1 H2O molecule = 18.015/N_A g; 1 mm PWV = 0.1 g/cm² of liquid-equivalent.
WV_MM_PER_MOLEC_CM2 = 2.991e-22

# CVD-safe palette (Okabe-Ito + a few from Paul Tol's "muted" set for
# the extended figure, all distinguishable to deuteranopia/protanopia)
COLOR_TAUC         = '#0072B2'   # blue
COLOR_ADIAB        = '#009E73'   # bluish-green
COLOR_DRIZZLE      = '#D55E00'   # vermillion          (bottom-30 % proxy)
COLOR_WV_ABOVE     = '#CC79A7'   # reddish purple
COLOR_WV_IN        = '#E69F00'   # orange
COLOR_DRIZZLE_TOP  = '#56B4E9'   # sky blue            (top-30 % proxy)
COLOR_RE_MAX       = '#882255'   # deep magenta (Tol)  (max anywhere)
COLOR_RE_RANGE     = '#117733'   # forest green (Tol)  (dynamic range)
COLOR_ALPHA        = '#AA3377'   # purple-pink (Tol)   (α shape parameter)


def setup_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
    })


# ─── Inference: rebuild test split per fold and run the saved model ──────
def _run_kfold_inference(kfold_dir: Path,
                         h5_override: str | None,
                         device_arg: str | None,
                         seed: int) -> dict:
    """Loop folds, predict on each fold's held-out test set, concatenate."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from models                              import RetrievalConfig
    from models_profile_only_extras          import ProfileOnlyNetworkExtras
    from data                                import LibRadtranDatasetExtras
    from train_standalone_profile_only_synthetic import make_kfold_splits
    from train_standalone_profile_only_extras    import predict_test

    with (kfold_dir / 'summary.json').open() as f:
        s = json.load(f)
    hp            = s['hyperparams']
    extras_active = s['extras_active']
    K             = int(s['n_folds'])
    n_val         = int(s['folds'][0]['n_val'])
    h5_path       = _resolve_h5_path(s['h5_path'], h5_override)

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
    print(f'  K = {K}, n_val per fold = {n_val}, seed = {seed}')

    dataset = LibRadtranDatasetExtras(
        str(h5_path), normalize=True, instrument='hysics',
        zero_tau_c     = not extras_active['tau_c'],
        zero_wv_above  = not extras_active['wv_above_cloud'],
        zero_wv_in     = not extras_active['wv_in_cloud'],
    )
    n_total  = len(dataset)
    n_levels = dataset.n_levels
    folds = make_kfold_splits(n_total, K, n_val=n_val, seed=seed)

    model_cfg = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    pin = (device.type == 'cuda')

    all_pred, all_pred_std, all_true, all_idx = [], [], [], []
    for k, (_, _, test_idx) in enumerate(folds, start=1):
        ckpt_path = kfold_dir / f'fold_{k:02d}' / 'best_model.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=hp['batch_size'], shuffle=False,
            num_workers=4, pin_memory=pin,
        )
        out = predict_test(model, test_loader, device, model_cfg)
        all_pred.append(out['pred'])
        all_pred_std.append(out['pred_std'])
        all_true.append(out['true'])
        all_idx.append(np.asarray(test_idx, dtype=np.int64))
        print(f'  fold {k:02d}: n_test = {len(test_idx):5d}  done '
              f'(best epoch {int(ckpt["epoch"])})')

    return {
        'pred':     np.concatenate(all_pred,     axis=0),
        'pred_std': np.concatenate(all_pred_std, axis=0),
        'true':     np.concatenate(all_true,     axis=0),
        'h5_idx':   np.concatenate(all_idx,      axis=0),
    }


# ─── Predictor extraction (incl. water-vapor mm conversion) ──────────────
def compute_predictors(h5_idx: np.ndarray, h5_path: Path) -> dict:
    """Per-sample predictors for the aggregate, recovered from the HDF5."""
    sort_order = np.argsort(h5_idx)
    unsort     = np.argsort(sort_order)
    sorted_idx = h5_idx[sort_order]

    with h5py.File(h5_path, 'r') as f:
        tau_c    = f['tau_c'][sorted_idx]
        profs    = f['profiles_raw'][sorted_idx]
        z_arr    = f['profiles_raw_z'][sorted_idx]
        nlev_arr = f['profile_n_levels'][sorted_idx]
        wv_above = f['wv_above_cloud'][sorted_idx]
        wv_in    = f['wv_in_cloud'][sorted_idx]
        alpha    = f['alpha'][sorted_idx]

    tau_c    = tau_c[unsort]
    profs    = profs[unsort]
    z_arr    = z_arr[unsort]
    nlev_arr = nlev_arr[unsort]
    wv_above = wv_above[unsort]
    wv_in    = wv_in[unsort]
    alpha    = alpha[unsort]

    n = len(h5_idx)
    adiab            = np.full(n, np.nan, dtype=np.float32)
    drizzle_proxy    = np.full(n, np.nan, dtype=np.float32)
    drizzle_proxy_top = np.full(n, np.nan, dtype=np.float32)
    re_max_anywhere  = np.full(n, np.nan, dtype=np.float32)
    re_range         = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        nL = int(nlev_arr[i])
        re = profs[i, :nL].astype(np.float64)
        z  = z_arr[i, :nL].astype(np.float64)
        adiab[i]         = adiabaticity_pearson_r(re, z)
        # Existing bottom-30 % drizzle proxy.
        drizzle_proxy[i] = float(re[int(0.7 * nL):].max())
        # Mirror of the bottom-30 % cut for the cloud top. Using
        # nL - int(0.7*nL) keeps the bin size symmetric with the bottom
        # version (3 levels for nL=7, vs the 2 you get from int(0.3*nL)).
        n_top = max(1, nL - int(0.7 * nL))
        drizzle_proxy_top[i] = float(re[:n_top].max())
        # Max anywhere in the profile (no vertical restriction).
        re_max_anywhere[i]   = float(re.max())
        # Dynamic range (max minus min).
        re_range[i]          = float(re.max() - re.min())

    return {
        'tau_c':             np.asarray(tau_c, dtype=np.float32),
        'adiab_score':       adiab,
        'drizzle_proxy':     drizzle_proxy,
        'drizzle_proxy_top': drizzle_proxy_top,
        're_max':            re_max_anywhere,
        're_range':          re_range,
        'alpha':             np.asarray(alpha, dtype=np.float32),
        'wv_above_mm':       (wv_above * WV_MM_PER_MOLEC_CM2).astype(np.float32),
        'wv_in_mm':          (wv_in    * WV_MM_PER_MOLEC_CM2).astype(np.float32),
    }


# ─── Figure ──────────────────────────────────────────────────────────────
def plot_diagnostics_3x2(per_sample_rmse: np.ndarray,
                         predictors: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    def _scatter(ax, x, label, name, color, xlog=False):
        valid = np.isfinite(x) & np.isfinite(per_sample_rmse)
        ax.scatter(x[valid], per_sample_rmse[valid], s=6, alpha=0.32,
                   color=color, edgecolors='none')
        if xlog:
            ax.set_xscale('log')
        r = _pearson_r(x, per_sample_rmse)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Per-sample mean RMSE (μm)', fontsize=11)
        ax.set_title(fr'({label})  RMSE vs {name}    '
                     fr'(Pearson $r$ = {r:+.3f})',
                     fontsize=11)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    _scatter(axes[0, 0], predictors['tau_c'],
             'a', r'$\tau_c$',
             color=COLOR_TAUC, xlog=True)
    _scatter(axes[0, 1], predictors['adiab_score'],
             'b', r'adiabaticity (Pearson $r$ of $r_e^3$ vs $z-z_{base}$)',
             color=COLOR_ADIAB)
    _scatter(axes[1, 0], predictors['drizzle_proxy'],
             'c', r'drizzle proxy:  max $r_e$ in lower 30 % ($\mu$m)',
             color=COLOR_DRIZZLE)

    # (d) CDF
    ax = axes[1, 1]
    x_sorted = np.sort(per_sample_rmse[np.isfinite(per_sample_rmse)])
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax.plot(x_sorted, cdf, color='black', linewidth=1.6)
    pcts     = (10, 25, 50, 75, 90)
    pct_vals = np.percentile(x_sorted, pcts)
    for p, v in zip(pcts, pct_vals):
        ax.axvline(v, color='0.55', linestyle=':', linewidth=0.8)
        ax.text(v, 1.02, f'p{p}', fontsize=8, ha='center', va='bottom',
                color='0.4')
    ax.set_xlabel('Per-sample mean RMSE (μm)', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title('(d)  CDF of per-sample RMSE on the 20-fold aggregate',
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    _scatter(axes[2, 0], predictors['wv_above_mm'],
             'e', 'above-cloud column water vapor (mm)',
             color=COLOR_WV_ABOVE)
    _scatter(axes[2, 1], predictors['wv_in_mm'],
             'f', 'in-cloud column water vapor (mm)',
             color=COLOR_WV_IN)

    fig.suptitle(f'Per-cloud RMSE diagnostics — 20-fold aggregate '
                 f'({len(per_sample_rmse):,} samples)',
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_diagnostics_5x2(per_sample_rmse: np.ndarray,
                         predictors: dict, out_path: Path) -> None:
    """Extended 5x2 layout: original 3x2 + drizzle-top/max + range/alpha."""
    fig, axes = plt.subplots(5, 2, figsize=(13, 22))

    def _scatter(ax, x, label, name, color, xlog=False):
        valid = np.isfinite(x) & np.isfinite(per_sample_rmse)
        ax.scatter(x[valid], per_sample_rmse[valid], s=6, alpha=0.32,
                   color=color, edgecolors='none')
        if xlog:
            ax.set_xscale('log')
        r = _pearson_r(x, per_sample_rmse)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Per-sample mean RMSE (μm)', fontsize=11)
        ax.set_title(fr'({label})  RMSE vs {name}    '
                     fr'(Pearson $r$ = {r:+.3f})',
                     fontsize=11)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    # ── Rows 1-3: same as the 3x2 figure ────────────────────────────────
    _scatter(axes[0, 0], predictors['tau_c'],
             'a', r'$\tau_c$',
             color=COLOR_TAUC, xlog=True)
    _scatter(axes[0, 1], predictors['adiab_score'],
             'b', r'adiabaticity (Pearson $r$ of $r_e^3$ vs $z-z_{base}$)',
             color=COLOR_ADIAB)
    _scatter(axes[1, 0], predictors['drizzle_proxy'],
             'c', r'drizzle proxy:  max $r_e$ in lower 30 % ($\mu$m)',
             color=COLOR_DRIZZLE)

    # (d) CDF
    ax = axes[1, 1]
    x_sorted = np.sort(per_sample_rmse[np.isfinite(per_sample_rmse)])
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax.plot(x_sorted, cdf, color='black', linewidth=1.6)
    pcts     = (10, 25, 50, 75, 90)
    pct_vals = np.percentile(x_sorted, pcts)
    for p, v in zip(pcts, pct_vals):
        ax.axvline(v, color='0.55', linestyle=':', linewidth=0.8)
        ax.text(v, 1.02, f'p{p}', fontsize=8, ha='center', va='bottom',
                color='0.4')
    ax.set_xlabel('Per-sample mean RMSE (μm)', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title('(d)  CDF of per-sample RMSE on the 20-fold aggregate',
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    _scatter(axes[2, 0], predictors['wv_above_mm'],
             'e', 'above-cloud column water vapor (mm)',
             color=COLOR_WV_ABOVE)
    _scatter(axes[2, 1], predictors['wv_in_mm'],
             'f', 'in-cloud column water vapor (mm)',
             color=COLOR_WV_IN)

    # ── Row 4: top-30 % drizzle proxy + max anywhere ────────────────────
    _scatter(axes[3, 0], predictors['drizzle_proxy_top'],
             'g', r'cloud-top drizzle proxy:  max $r_e$ in upper 30 % ($\mu$m)',
             color=COLOR_DRIZZLE_TOP)
    _scatter(axes[3, 1], predictors['re_max'],
             'h', r'max $r_e$ anywhere in profile ($\mu$m)',
             color=COLOR_RE_MAX)

    # ── Row 5: dynamic range + alpha ────────────────────────────────────
    _scatter(axes[4, 0], predictors['re_range'],
             'i', r'profile dynamic range:  $\max r_e - \min r_e$ ($\mu$m)',
             color=COLOR_RE_RANGE)
    _scatter(axes[4, 1], predictors['alpha'],
             'j', r'$\alpha$ shape parameter '
                  r'($\nu_{\mathrm{eff}} = 1/(\alpha + 3)$)',
             color=COLOR_ALPHA)

    fig.suptitle(f'Per-cloud RMSE diagnostics — extended (20-fold aggregate, '
                 f'{len(per_sample_rmse):,} samples)',
                 fontsize=13, y=1.003)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--kfold-dir', required=True,
                   help='Directory containing fold_01..fold_K + summary.json.')
    p.add_argument('--output-name',
                   default='per_cloud_rmse_vs_predictors_with_wv.png',
                   help='Filename for the 3x2 figure inside --kfold-dir.')
    p.add_argument('--output-name-extended',
                   default='per_cloud_rmse_vs_predictors_extended.png',
                   help='Filename for the 5x2 (extended) figure.')
    p.add_argument('--refresh', action='store_true',
                   help='Force re-inference even if agg_pred_cache.npz exists.')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the seed used for k-fold split at training.')
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in summary.json.')
    return p.parse_args()


def main():
    args = parse_args()
    setup_style()

    kfold_dir = Path(args.kfold_dir).resolve()
    if not kfold_dir.exists():
        raise FileNotFoundError(f'kfold-dir not found: {kfold_dir}')
    cache_path = kfold_dir / 'agg_pred_cache.npz'

    if cache_path.exists() and not args.refresh:
        print(f'Loading cached aggregate from {cache_path}')
        c = np.load(cache_path)
        pred, pred_std, true = c['pred'], c['pred_std'], c['true']
        h5_idx               = c['h5_idx']
    else:
        print('Running 20-fold inference (may take several minutes)...')
        out = _run_kfold_inference(kfold_dir, args.h5_path, args.device, args.seed)
        pred, pred_std, true = out['pred'], out['pred_std'], out['true']
        h5_idx               = out['h5_idx']
        np.savez_compressed(cache_path,
                            pred=pred, pred_std=pred_std, true=true,
                            h5_idx=h5_idx)
        print(f'Cached aggregate -> {cache_path}')

    print(f'Aggregate : {pred.shape[0]:,} samples x {pred.shape[1]} levels')

    err = pred - true
    per_sample_rmse = np.sqrt(np.mean(err**2, axis=1))

    with (kfold_dir / 'summary.json').open() as f:
        s = json.load(f)
    h5_path = _resolve_h5_path(s['h5_path'], args.h5_path)
    predictors = compute_predictors(h5_idx, h5_path)

    out_fig = kfold_dir / args.output_name
    plot_diagnostics_3x2(per_sample_rmse, predictors, out_fig)
    print(f'3x2 figure -> {out_fig}  (dpi={DPI})')

    out_fig_ext = kfold_dir / args.output_name_extended
    plot_diagnostics_5x2(per_sample_rmse, predictors, out_fig_ext)
    print(f'5x2 figure -> {out_fig_ext}  (dpi={DPI})')


if __name__ == '__main__':
    main()
