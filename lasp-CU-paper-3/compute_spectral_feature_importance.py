"""
compute_spectral_feature_importance.py — Per-wavelength feature importance for
the trained ProfileOnlyNetwork models from a full-coverage K-fold run.

Method: mean-substitution importance
------------------------------------
For each fold's best model, evaluate test-set mean RMSE under two conditions:

    baseline_rmse     : all 640 inputs intact
    perturbed_rmse[i] : replace channel `i` of x with its mean across the
                        test set, leaving every other channel intact

    importance[i]     = perturbed_rmse[i] - baseline_rmse        (μm)

Higher Δ-RMSE ⇒ scrambling that channel hurt the model more ⇒ that
wavelength carried more of the information the model was using.  This is
deterministic permutation importance: instead of shuffling channel `i`
across samples, we replace it with the population mean — same expected
effect, no Monte-Carlo noise.

Why not just read first-layer weights?
--------------------------------------
First-layer ‖W‖ is fast but unreliable: a small weight can still pass
strong information forward through downstream nonlinearities, and a large
weight that gets killed by LayerNorm + dropout can look important without
contributing.  Mean-substitution measures the model's *behavioural*
reliance on the channel — what the user actually cares about for
"how important is this wavelength?".

Only the 636 spectral channels are perturbed; the 4 geometry channels
(sza, vza, saz, vaz) are left alone per the user's request.

Inputs
------
    --output-dir : directory containing fold_00/, fold_01/, … from
                   train_kfold_full_coverage_profile_only.py
    --h5-path    : the same HDF5 used for the K-fold runs
    --device     : 'cuda' / 'mps' / 'cpu'
    --n-samples  : subsample size per fold for the perturbation sweep
                   (default 1000; full test set = ~7300 per fold)

Outputs (under --output-dir)
----------------------------
    spectral_feature_importance.csv
        wavelength_idx, wavelength_nm,
        importance_mean_um, importance_std_um,
        importance_relative_mean,
        baseline_rmse_per_fold, importance_per_fold (one column each)

    figures/spectral_feature_importance.png
        scatter + thin line, mean across 21 folds with ±1σ band,
        x-axis = wavelength (nm), y-axis = relative importance (max = 1)

    figures/spectral_feature_importance_absolute.png
        same but y-axis = mean Δ-RMSE in μm

Usage
-----
    python compute_spectral_feature_importance.py \\
        --output-dir standalone_results_profile_only_kfold/run110_..._K21 \\
        --h5-path training_data/combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5 \\
        --device cuda

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_profile_only import ProfileOnlyNetwork
from models              import RetrievalConfig
from data                import (LibRadtranDataset,
                                 create_rotating_kfold_splits,
                                 resolve_h5_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--output-dir', type=str, required=True,
                   help='Parent dir containing fold_NN/ subdirs from the '
                        'full-coverage K-fold trainer.')
    p.add_argument('--h5-path', type=str, required=True)
    p.add_argument('--training-data-dir', type=str, default=None)
    p.add_argument('--n-folds', type=int, default=None,
                   help='Number of folds.  Default: auto-detect from fold_NN/ '
                        'subdirs in --output-dir.')
    p.add_argument('--n-samples', type=int, default=1000,
                   help='Subsample size per fold for perturbation '
                        '(default 1000; full test ~ 7300/fold).  Larger '
                        '= more accurate but slower.')
    p.add_argument('--seed', type=int, default=42,
                   help='Same seed used by the K-fold trainer (so this script '
                        'reconstructs the same splits).')
    p.add_argument('--n-val-profiles', type=int, default=14,
                   help='Same value used by the K-fold trainer.')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=2048,
                   help='Inference batch size (memory-bound, not accuracy)')
    p.add_argument('--n-spectral-channels', type=int, default=636,
                   help='How many channels to perturb (default 636 = '
                        'spectral only; geometry inputs are skipped)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Inference + perturbation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_rmse(model, x_eval, prof_true_phys, re_min, re_max,
                  device, batch_size: int) -> float:
    """
    Run forward pass on x_eval and return mean RMSE in μm against the *true
    physical* profile.  prof_true_phys is in μm.
    """
    sse_per_level = None
    n_total = 0
    for i0 in range(0, x_eval.shape[0], batch_size):
        xb = torch.from_numpy(x_eval[i0:i0 + batch_size]).to(device)
        out = model(xb)['profile'].cpu().numpy()
        sq  = (out - prof_true_phys[i0:i0 + batch_size]) ** 2
        if sse_per_level is None:
            sse_per_level = sq.sum(axis=0)
        else:
            sse_per_level += sq.sum(axis=0)
        n_total += sq.shape[0]
    rmse_per_level = np.sqrt(sse_per_level / n_total)
    return float(rmse_per_level.mean())


def importance_for_one_fold(fold_dir: Path,
                            h5_path: Path,
                            n_folds: int,
                            n_val_profiles: int,
                            seed: int,
                            n_samples: int,
                            n_channels: int,
                            device: torch.device,
                            batch_size: int) -> dict:
    """
    Returns dict with:
        baseline_rmse, importance (n_channels,), n_eval_samples
    """
    fold_idx = int(fold_dir.name.split('_')[-1])

    # 1. Reconstruct the same test split that produced this fold's checkpoint
    train_idx, val_idx, test_idx = create_rotating_kfold_splits(
        str(h5_path), fold_idx=fold_idx, n_folds=n_folds,
        n_val_profiles=n_val_profiles, seed=seed,
    )

    # Subsample for tractability (importance is a population-level
    # statistic; 1000 samples gives stable Δ-RMSE estimates to ~0.005 μm).
    rng = np.random.default_rng(seed + 7919 + fold_idx)
    if len(test_idx) > n_samples:
        sel = rng.choice(len(test_idx), size=n_samples, replace=False)
        sel.sort()
        test_idx_eval = test_idx[sel]
    else:
        test_idx_eval = test_idx

    # 2. Build the input matrix x and physical-units true profile y for the
    #    selected test indices.  Bypass the DataLoader to avoid copying
    #    overhead — we'll repeatedly perturb x in place.
    ds = LibRadtranDataset(str(h5_path), normalize=True, instrument='hysics')
    x_list, y_list = [], []
    for i in test_idx_eval:
        xi, profi, _ = ds[int(i)]
        x_list.append(xi.numpy())
        y_list.append(profi.numpy())
    x_full = np.stack(x_list).astype(np.float32)             # (n, 640)
    prof_norm = np.stack(y_list).astype(np.float32)          # (n, 50)
    re_min, re_max = float(ds.re_min), float(ds.re_max)
    prof_true_phys = (prof_norm * (re_max - re_min) + re_min).astype(np.float32)

    # 3. Load the model.  The checkpoint stores model_config so we can
    #    rebuild the network identically.
    ckpt = torch.load(fold_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    mc_dict = ckpt['model_config']
    model_config = RetrievalConfig(
        n_wavelengths=mc_dict['n_wavelengths'],
        n_geometry_inputs=mc_dict['n_geometry_inputs'],
        n_levels=mc_dict['n_levels'],
        hidden_dims=tuple(mc_dict['hidden_dims']),
        dropout=mc_dict['dropout'],
        activation=mc_dict['activation'],
    )
    model = ProfileOnlyNetwork(model_config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 4. Baseline RMSE
    baseline = _predict_rmse(model, x_full, prof_true_phys,
                             re_min, re_max, device, batch_size)

    # 5. Per-channel mean-substitution
    channel_means = x_full[:, :n_channels].mean(axis=0)      # (n_channels,)
    importance = np.zeros(n_channels, dtype=np.float64)

    # In-place perturbation: copy original column once, restore after each pass.
    for c in range(n_channels):
        orig = x_full[:, c].copy()
        x_full[:, c] = channel_means[c]
        rmse_c = _predict_rmse(model, x_full, prof_true_phys,
                               re_min, re_max, device, batch_size)
        x_full[:, c] = orig
        importance[c] = rmse_c - baseline

    return {
        'fold_idx':       fold_idx,
        'baseline_rmse':  baseline,
        'importance':     importance,
        'n_eval_samples': int(x_full.shape[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    h5_path = resolve_h5_path(args.h5_path, args.training_data_dir).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 not found: {h5_path}')

    fold_dirs = sorted(output_dir.glob('fold_*'))
    if args.n_folds is not None:
        fold_dirs = [d for d in fold_dirs if int(d.name.split('_')[-1]) < args.n_folds]
    n_folds = max((int(d.name.split('_')[-1]) for d in fold_dirs), default=-1) + 1
    if n_folds == 0:
        raise FileNotFoundError(f'No fold_*/ subdirs found in {output_dir}')
    print(f"Detected {len(fold_dirs)} fold dirs in {output_dir} "
          f"(n_folds = {n_folds})")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("  [warn] importance sweep on CPU may take 20+ min for 21 folds")

    # Wavelength axis from the HDF5
    with h5py.File(h5_path, 'r') as f:
        wavelengths_nm = f['wavelengths'][:].astype(np.float64)
    n_wl = wavelengths_nm.size
    if n_wl != args.n_spectral_channels:
        raise RuntimeError(
            f"HDF5 has {n_wl} wavelengths but --n-spectral-channels = "
            f"{args.n_spectral_channels}.  Pass the right value."
        )

    # ── Per-fold importance sweep ────────────────────────────────────────────
    importance_per_fold = np.zeros((len(fold_dirs), args.n_spectral_channels),
                                   dtype=np.float64)
    baselines = []
    n_evals   = []
    t_start = time.time()
    for j, fd in enumerate(fold_dirs):
        t0 = time.time()
        out = importance_for_one_fold(
            fd, h5_path, n_folds=n_folds,
            n_val_profiles=args.n_val_profiles, seed=args.seed,
            n_samples=args.n_samples, n_channels=args.n_spectral_channels,
            device=device, batch_size=args.batch_size,
        )
        importance_per_fold[j] = out['importance']
        baselines.append(out['baseline_rmse'])
        n_evals.append(out['n_eval_samples'])
        dt = time.time() - t0
        print(f"  fold {out['fold_idx']:02d}: baseline RMSE = {out['baseline_rmse']:.3f} μm, "
              f"n_eval = {out['n_eval_samples']:,}, "
              f"max Δ-RMSE = {out['importance'].max():+.3f} μm "
              f"(channel {int(out['importance'].argmax())}, "
              f"{wavelengths_nm[out['importance'].argmax()]:.1f} nm), "
              f"{dt:.0f}s")

    print(f"\nTotal sweep time: {(time.time() - t_start) / 60:.1f} min")

    # ── Aggregate across folds ───────────────────────────────────────────────
    imp_mean = importance_per_fold.mean(axis=0)
    imp_std  = importance_per_fold.std(axis=0)
    # Relative importance: normalize so peak channel = 1.  Use abs() in case
    # any channel happens to *help* the model when scrambled (negative Δ);
    # we want magnitude of effect.
    denom = max(np.abs(imp_mean).max(), 1e-12)
    imp_rel = imp_mean / denom

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / 'spectral_feature_importance.csv'
    fieldnames = (['wavelength_idx', 'wavelength_nm',
                   'importance_mean_um', 'importance_std_um',
                   'importance_relative_mean']
                  + [f'importance_fold_{int(d.name.split("_")[-1]):02d}_um'
                     for d in fold_dirs])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in range(args.n_spectral_channels):
            row = {
                'wavelength_idx':           c,
                'wavelength_nm':            float(wavelengths_nm[c]),
                'importance_mean_um':       float(imp_mean[c]),
                'importance_std_um':        float(imp_std[c]),
                'importance_relative_mean': float(imp_rel[c]),
            }
            for j, d in enumerate(fold_dirs):
                row[f'importance_fold_{int(d.name.split("_")[-1]):02d}_um'] = \
                    float(importance_per_fold[j, c])
            writer.writerow(row)
    print(f"\nCSV written: {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 1. Relative importance (peak = 1)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wavelengths_nm, imp_rel, '-', linewidth=0.6, color='#1F5FC2',
            alpha=0.55, zorder=1)
    ax.scatter(wavelengths_nm, imp_rel, s=8, color='#1F5FC2',
               edgecolors='none', zorder=2)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Relative feature importance  '
                  '(peak Δ-RMSE = 1)', fontsize=11)
    ax.set_title(f'Spectral Feature Importance — mean across {len(fold_dirs)} folds  '
                 f'(mean-substitution Δ-RMSE; n_eval = {n_evals[0]} samples/fold)',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'spectral_feature_importance.png', dpi=400)
    plt.close()

    # 2. Absolute importance (μm) with ±1σ band — useful for the paper
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(wavelengths_nm,
                    imp_mean - imp_std, imp_mean + imp_std,
                    color='#1F5FC2', alpha=0.18, label='±1 std across folds')
    ax.plot(wavelengths_nm, imp_mean, '-', linewidth=0.6, color='#1F5FC2',
            alpha=0.85, zorder=1, label='mean Δ-RMSE')
    ax.scatter(wavelengths_nm, imp_mean, s=8, color='#1F5FC2',
               edgecolors='none', zorder=2)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Mean-substitution Δ-RMSE (μm)', fontsize=11)
    ax.set_title(f'Spectral Feature Importance — absolute Δ-RMSE  '
                 f'(higher = scrambling this wavelength hurt more)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'spectral_feature_importance_absolute.png', dpi=400)
    plt.close()

    # ── Stdout summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TOP-10 MOST IMPORTANT WAVELENGTHS")
    print(f"{'=' * 70}")
    top = np.argsort(-imp_mean)[:10]
    print(f"  {'rank':>4}  {'idx':>4}  {'λ (nm)':>9}  "
          f"{'Δ-RMSE mean (μm)':>17}  {'± std':>8}  rel")
    for rank, c in enumerate(top, 1):
        print(f"  {rank:>4}  {c:>4}  {wavelengths_nm[c]:>9.2f}  "
              f"{imp_mean[c]:>+17.4f}  {imp_std[c]:>8.4f}  {imp_rel[c]:+.3f}")

    print(f"\n  Baseline mean RMSE per fold: "
          f"min {min(baselines):.3f}, max {max(baselines):.3f}, "
          f"mean {np.mean(baselines):.3f} μm")
    print(f"\n  Figures: {fig_dir}/spectral_feature_importance.png  "
          f"(relative)\n           "
          f"{fig_dir}/spectral_feature_importance_absolute.png  (absolute)")


if __name__ == '__main__':
    main()
