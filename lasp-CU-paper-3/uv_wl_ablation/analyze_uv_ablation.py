"""
analyze_uv_ablation.py — Feature importance + RMSE comparison for the UV
retrain-ablation (train_uv_ablation_run004.py).

For the chosen ablation/control checkpoint it:
  1. reproduces the same 889-profile test split,
  2. recomputes permutation importance over the surviving inputs,
  3. compares mean test RMSE against run_004 (full spectrum), and
  4. reports whether the 352 nm importance peak MIGRATED to a surviving
     visible channel (the causal signature of "352 nm was not unique").

Outputs: prints a verdict table and writes
    feature_importance_run004/fig6_uv_ablation_comparison.png

Run (after training finishes):
    /opt/anaconda3/bin/python3 analyze_uv_ablation.py \
        --model-dir standalone_results_uv_ablation/ablation_cutoff500nm_keep587

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
HERE = Path(__file__).resolve().parent
REPO = HERE
while not (REPO / 'models.py').exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
from models import RetrievalConfig                                   # noqa: E402
from models_profile_only_extras import ProfileOnlyNetworkExtras      # noqa: E402
from compute_feature_importance_run004 import (                      # noqa: E402
    pick_device, mean_per_level_rmse, build_test_inputs, N_SPECTRAL)

OUT_DIR = REPO / 'feature_importance_run004'
RUN_004_RMSE = 0.9641
C_ORIG = '#1F5FC2'
C_ABL = '#CC2A1E'
C_BAND = '#E6C98C'
WV_BANDS_UM = [(0.91, 0.98), (1.10, 1.18), (1.34, 1.44), (1.78, 1.98)]


def permutation_importance(model, X, true_phys, device, baseline, n_inform,
                           n_repeats, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    imp = np.zeros((n_repeats, n_inform))
    Xw = X.copy()
    for rep in range(n_repeats):
        for c in range(n_inform):
            col = Xw[:, c].copy()
            Xw[:, c] = col[rng.permutation(n)]
            imp[rep, c] = mean_per_level_rmse(model, Xw, true_phys, device) - baseline
            Xw[:, c] = col
    return imp.mean(0), imp.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--n-repeats', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    if not mdir.is_absolute():
        mdir = (HERE / args.model_dir).resolve()   # outputs live under this script's dir
    device = pick_device(args.device)
    summ = json.loads((mdir / 'summary.json').read_text())
    ckpt = torch.load(mdir / 'best_model.pt', map_location=device, weights_only=False)
    keep_idx = np.array(ckpt['keep_idx'], dtype=np.int64)
    kept_wl = np.array(ckpt['kept_wavelengths_nm'], dtype=np.float64)
    mc = ckpt['model_config']
    n_kept = mc['n_wavelengths']
    n_inform = n_kept + 4                         # spectral + geometry (extras zeroed)

    print(f'Model dir : {mdir.name}')
    print(f'cutoff    : {summ["cutoff_nm"]} nm   kept {n_kept} spectral '
          f'({kept_wl.min():.1f}–{kept_wl.max():.1f} nm)')
    print(f'test RMSE : {summ["mean_test_rmse_um"]:.4f} μm   '
          f'(run_004 full spectrum: {RUN_004_RMSE:.4f} μm; '
          f'Δ = {summ["mean_test_rmse_um"] - RUN_004_RMSE:+.4f})')
    print(f'best epoch: {summ["best_epoch"]} / {summ["epochs_trained"]}')

    # build the same 889-profile test inputs, then drop to surviving channels
    X_full, true_phys, wl_full = build_test_inputs()      # (889,643)
    X = X_full[:, keep_idx].astype(np.float32)            # (889, n_kept+4+3)

    model = ProfileOnlyNetworkExtras(
        RetrievalConfig(n_wavelengths=n_kept, n_geometry_inputs=4,
                        n_levels=mc['n_levels'], hidden_dims=tuple(mc['hidden_dims']),
                        dropout=mc['dropout'], activation=mc['activation']),
        n_extras=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    baseline = mean_per_level_rmse(model, X, true_phys, device)
    print(f'recomputed baseline RMSE: {baseline:.4f} μm')

    imp_mean, imp_std = permutation_importance(
        model, X, true_phys, device, baseline, n_inform,
        args.n_repeats, args.seed)
    spec = imp_mean[:n_kept]
    geom = imp_mean[n_kept:n_kept + 4]
    geom_names = ['SZA', 'VZA', 'SAZ', 'VAZ']

    order = np.argsort(-imp_mean)
    labels = [f'{kept_wl[c]:.1f} nm' if c < n_kept else geom_names[c - n_kept]
              for c in range(n_inform)]
    print('\nTOP 10 inputs (ablated model):')
    for r, c in enumerate(order[:10], 1):
        print(f'  {r:>2}  {labels[c]:>10}  Δ-RMSE {imp_mean[c]:+.4f} ± {imp_std[c]:.4f}')

    # migration test: where is the new top spectral channel?
    top_spec = int(np.argmax(spec))
    print(f'\nMIGRATION TEST:')
    print(f'  original peak  : 351.95 nm (removed)')
    print(f'  new top channel: {kept_wl[top_spec]:.1f} nm  '
          f'(Δ-RMSE {spec[top_spec]:+.4f} μm)')
    print(f'  importance of new blue edge {kept_wl[0]:.1f} nm: '
          f'{spec[0]:+.4f} μm (rank {list(np.argsort(-spec)).index(0)+1}/{n_kept})')

    # ── comparison figure ────────────────────────────────────────────────────
    orig = np.load(OUT_DIR / 'feature_importance_run004.npz', allow_pickle=True)
    o_perm = orig['perm_mean_um'][:N_SPECTRAL]
    o_wl_um = orig['wavelengths_nm'] / 1000.0
    a_wl_um = kept_wl / 1000.0

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'cm'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True)
    for lo, hi in WV_BANDS_UM:
        ax1.axvspan(lo, hi, color=C_BAND, alpha=0.5, lw=0)
        ax2.axvspan(lo, hi, color=C_BAND, alpha=0.5, lw=0)

    ax1.axvspan(o_wl_um.min() - 0.02, 0.5, color='0.85', alpha=0.7, lw=0,
                label='removed in ablation (< 500 nm)')
    ax1.plot(o_wl_um, o_perm, color=C_ORIG, lw=0.9)
    ax1.scatter(o_wl_um, o_perm, s=6, color=C_ORIG, edgecolors='none')
    ax1.set_ylabel('Permutation\nΔRMSE (μm)')
    ax1.set_title(f'(a)  Original run_004 — full 636 channels '
                  f'(RMSE {RUN_004_RMSE:.3f} μm); 352 nm is the peak', fontsize=11)
    ax1.legend(loc='upper center', fontsize=9); ax1.grid(alpha=0.25)

    ax2.plot(a_wl_um, spec, color=C_ABL, lw=0.9)
    ax2.scatter(a_wl_um, spec, s=6, color=C_ABL, edgecolors='none')
    ax2.scatter([a_wl_um[top_spec]], [spec[top_spec]], s=70, facecolors='none',
                edgecolors='k', linewidths=1.3, zorder=6,
                label=f'new peak: {kept_wl[top_spec]:.0f} nm')
    ax2.set_ylabel('Permutation\nΔRMSE (μm)')
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_title(f'(b)  Ablated — {n_kept} channels ≥ 500 nm '
                  f'(RMSE {summ["mean_test_rmse_um"]:.3f} μm, '
                  f'Δ {summ["mean_test_rmse_um"]-RUN_004_RMSE:+.3f}); '
                  f'did the peak migrate?', fontsize=11)
    ax2.legend(loc='upper center', fontsize=9); ax2.grid(alpha=0.25)
    ax2.set_xlim(o_wl_um.min() - 0.03, o_wl_um.max() + 0.03)

    fig.suptitle('UV retrain-ablation: remove < 500 nm, retrain run_004 recipe, '
                 'recompute importance', y=0.99, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / 'fig6_uv_ablation_comparison.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out.name}')


if __name__ == '__main__':
    main()
