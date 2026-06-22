"""
plot_ablation_spectral_importance.py — PPT-style per-wavelength permutation
importance for the UV-ablated model (channels < 500 nm removed, retrained).

Renders the same single-panel slide figure as
    feature_importance_run004/fig2_spectral_importance_vs_wavelength_PPT.png
but for the ablated model, so the two can sit side by side. Saved as the
'_withoutUV' version next to the original.

Run (from uv_wl_ablation/ or anywhere):
    /opt/anaconda3/bin/python3 plot_ablation_spectral_importance.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE
while not (REPO / 'models.py').exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
from models import RetrievalConfig                                   # noqa: E402
from models_profile_only_extras import ProfileOnlyNetworkExtras      # noqa: E402
from compute_feature_importance_run004 import (                      # noqa: E402
    pick_device, mean_per_level_rmse, build_test_inputs)

OUT_DIR = REPO / 'feature_importance_run004'        # save beside the original fig2
C_SPECTRAL = '#1F5FC2'
C_BAND = '#E6C98C'
WV_BANDS_UM = [(0.91, 0.98), (1.10, 1.18), (1.34, 1.44), (1.78, 1.98)]


def permutation_importance_spectral(model, X, true_phys, device, baseline,
                                    n_spec, n_repeats, seed):
    """Δ-RMSE (μm) from shuffling each of the first n_spec (spectral) columns."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    imp = np.zeros((n_repeats, n_spec))
    Xw = X.copy()
    for rep in range(n_repeats):
        for c in range(n_spec):
            col = Xw[:, c].copy()
            Xw[:, c] = col[rng.permutation(n)]
            imp[rep, c] = mean_per_level_rmse(model, Xw, true_phys, device) - baseline
            Xw[:, c] = col
    return imp.mean(0), imp.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=str(
        HERE / 'standalone_results_uv_ablation' / 'ablation_cutoff500nm_keep587'))
    ap.add_argument('--n-repeats', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    if not mdir.is_absolute():
        mdir = (HERE / args.model_dir).resolve()
    device = pick_device(args.device)
    summ = json.loads((mdir / 'summary.json').read_text())
    ckpt = torch.load(mdir / 'best_model.pt', map_location=device, weights_only=False)
    keep_idx = np.array(ckpt['keep_idx'], dtype=np.int64)
    kept_wl = np.array(ckpt['kept_wavelengths_nm'], dtype=np.float64)
    mc = ckpt['model_config']
    n_kept = mc['n_wavelengths']

    model = ProfileOnlyNetworkExtras(
        RetrievalConfig(n_wavelengths=n_kept, n_geometry_inputs=4,
                        n_levels=mc['n_levels'], hidden_dims=tuple(mc['hidden_dims']),
                        dropout=mc['dropout'], activation=mc['activation']),
        n_extras=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    X_full, true_phys, _ = build_test_inputs()
    X = X_full[:, keep_idx].astype(np.float32)
    baseline = mean_per_level_rmse(model, X, true_phys, device)
    print(f'ablated baseline RMSE: {baseline:.4f} μm  '
          f'(run_004 full-spectrum 0.9641 μm)')

    perm, perm_std = permutation_importance_spectral(
        model, X, true_phys, device, baseline, n_kept, args.n_repeats, args.seed)
    np.savez_compressed(
        OUT_DIR / 'feature_importance_ablation_withoutUV.npz',
        perm_mean_um=perm, perm_std_um=perm_std, wavelengths_nm=kept_wl,
        n_spectral=n_kept, n_repeats=args.n_repeats, baseline_rmse_um=baseline)

    # ── PPT figure (identical styling to fig2_..._PPT.png) ───────────────────
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'cm'})
    wl_um = kept_wl / 1000.0
    fig, ax = plt.subplots(figsize=(13.33, 6.7))
    for k, (lo, hi) in enumerate(WV_BANDS_UM):
        ax.axvspan(lo, hi, color=C_BAND, alpha=0.5, lw=0,
                   label='H$_2$O vapor band' if k == 0 else None)
    ax.plot(wl_um, perm, color=C_SPECTRAL, lw=1.4, zorder=3,
            label='permutation importance')
    ax.scatter(wl_um, perm, s=16, color=C_SPECTRAL, edgecolors='none', zorder=4)
    ax.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax.set_xlabel('Wavelength (μm)', fontsize=22)
    ax.set_ylabel('Permutation importance:  $\\Delta$RMSE (μm)', fontsize=22)
    ax.set_title('Per-wavelength feature importance — droplet-profile retrieval '
                 '(< 500 nm removed)', fontsize=15, pad=10)
    ax.tick_params(labelsize=18)
    ax.set_xlim(wl_um.min() - 0.02, wl_um.max() + 0.02)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.99), ncol=1,
              frameon=True, fontsize=16, framealpha=0.92)
    out = OUT_DIR / 'fig2_spectral_importance_vs_wavelength_PPT_withoutUV.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
