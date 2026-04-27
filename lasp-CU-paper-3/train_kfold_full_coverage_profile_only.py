"""
train_kfold_full_coverage_profile_only.py — Full-coverage K-fold standalone
trainer for the profile-only model.

What this does
--------------
Given a chosen sweep config (--run-id from sweep_results_profile_only/), this
script trains the ProfileOnlyNetwork K times.  The K folds rotate the test
set so that every one of the 290 unique droplet profiles in the training
catalog ends up in the held-out test set EXACTLY ONCE.  The aggregated
output is a per-profile mean RMSE table that can be correlated against
profile-level physical quantities (τ_c, above-cloud / in-cloud water vapor,
adiabaticity, drizzle proxies).

Why use it
----------
The K=5 sweep tells you the cross-fold std of the headline mean RMSE — but
that's a single number summarizing model behaviour across 5 splits of 14
test profiles each, with the same 14 test profiles every time.  This script
instead samples *every profile* exactly once, producing a per-profile RMSE
distribution.  That lets you ask:

    Is RMSE worse on optically thick clouds?         (vs τ_c)
    Is RMSE worse when above-cloud humidity is low?  (vs wv_above_cloud)
    Is RMSE worse on non-adiabatic profiles?         (vs adiabaticity score)
    Is RMSE worse when drizzle is present?           (vs r_e at base)

Usage modes
-----------
Default — sequential, one Python process trains all K folds, then aggregates.
Useful for short K (e.g. K=5) on a local machine:
    python train_kfold_full_coverage_profile_only.py \\
        --run-id 110 --h5-path <path> --n-folds 21 --device cuda

SLURM-array friendly — each task trains exactly one fold:
    python train_kfold_full_coverage_profile_only.py \\
        --run-id 110 --h5-path <path> --n-folds 21 --fold-idx ${SLURM_ARRAY_TASK_ID}
After all per-fold tasks have finished, run the aggregator separately:
    python train_kfold_full_coverage_profile_only.py \\
        --run-id 110 --h5-path <path> --n-folds 21 --aggregate-only \\
        --output-dir <same_dir_as_per_fold_runs>

Outputs (per fold)
------------------
    <output-dir>/fold_NN/
        best_model.pt
        config.json
        history.json
        results.json
        test_predictions.npz   <-- pred, true, pred_std, hdf5_indices, profile_ids

Outputs (aggregated)
--------------------
    <output-dir>/
        per_profile_summary.csv          one row per unique profile
        per_profile_correlations.json    Pearson r + Spearman ρ vs each predictor
        figures/per_profile_rmse_distribution.png
        figures/rmse_vs_predictors.png
        figures/per_level_uncertainty.png
        figures/per_profile_rmse_heatmap.png

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_profile_only          import ProfileOnlyNetwork, ProfileOnlyLoss
from models                       import RetrievalConfig
from data                         import (compute_profile_ids,
                                          create_rotating_kfold_dataloaders,
                                          resolve_h5_path)
from train_standalone_profile_only import build_level_weights


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-id', type=int, required=True,
                   help='Sweep run ID to borrow hyperparameters from')
    p.add_argument('--h5-path', type=str, required=True)
    p.add_argument('--training-data-dir', type=str, default=None)
    p.add_argument('--sweep-dir', type=str,
                   default='hyper_parameter_sweep/sweep_results_profile_only')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where fold_NN/ subdirs (and final aggregate) live. '
                        'Default: ./standalone_results_profile_only_kfold/'
                        'run<ID>_<h5stem>_K<n>')
    p.add_argument('--n-folds', type=int, default=21,
                   help='K folds; pick so K x typical_test_fold ≥ n_unique_profiles. '
                        'For 290 profiles, K=21 gives 13–14 profiles/fold (default).')
    p.add_argument('--n-val-profiles', type=int, default=14)

    # Mode-of-operation controls.
    p.add_argument('--fold-idx', type=int, default=None,
                   help='If given, train only this one fold (for SLURM array '
                        'parallelisation).  Otherwise all K folds run sequentially.')
    p.add_argument('--aggregate-only', action='store_true',
                   help='Skip training entirely; aggregate existing fold_NN/ '
                        'results and emit the per-profile CSV/correlations/figures.')

    p.add_argument('--device', type=str, default=None)
    p.add_argument('--n-epochs', type=int, default=None)
    p.add_argument('--level-weights', type=str, default=None,
                   choices=['uniform', 'deep', 'top',
                            'top_50style', 'bottom_50style', 'u_shape'])
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation / inference  (parity with sweep_train_profile_only.py)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    augment_noise_std=0.0,
                    warmup_steps=0, target_lr=None, global_step_start=0):
    model.train()
    loss_sum, n = 0.0, 0
    step = global_step_start
    for batch in loader:
        x, prof = batch[0].to(device), batch[1].to(device)
        if warmup_steps > 0 and step < warmup_steps and target_lr is not None:
            lr_now = target_lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now
        if augment_noise_std > 0.0:
            x = x + augment_noise_std * x * torch.randn_like(x)
        optimizer.zero_grad()
        output = model(x)
        losses = criterion(output, prof)
        losses['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += losses['total'].item()
        n += 1
        step += 1
    return loss_sum / n, step


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    for batch in loader:
        x, prof = batch[0].to(device), batch[1].to(device)
        output = model(x)
        losses = criterion(output, prof)
        loss_sum += losses['total'].item()
        n += 1
    return loss_sum / n


@torch.no_grad()
def predict_test(model, loader, device, config):
    """Return predictions in HDF5-row order (Subset preserves index order)."""
    model.eval()
    pred_list, std_list, true_list = [], [], []
    re_min, re_max = config.re_min, config.re_max
    for batch in loader:
        x, prof = batch[0].to(device), batch[1]
        out = model(x)
        pred_list.append(out['profile'].cpu().numpy())
        std_list.append(out['profile_std'].cpu().numpy())
        true_list.append(prof.numpy() * (re_max - re_min) + re_min)
    return {
        'pred':     np.concatenate(pred_list),
        'pred_std': np.concatenate(std_list),
        'true':     np.concatenate(true_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-profile predictor calculations
# ─────────────────────────────────────────────────────────────────────────────
def adiabaticity_pearson_r(re_raw: np.ndarray, z_raw: np.ndarray) -> float:
    """
    Pearson correlation between r_e^3 and (z - z_base) using the raw in-situ
    profile.  The adiabatic prediction is r_e^3 ∝ (z - z_base) (constant N,
    LWC linearly increasing with height above base), so this metric is +1
    for a perfectly adiabatic profile, 0 for uncorrelated, negative for
    inverted profiles (drizzle / heavy entrainment).
    """
    z_base = z_raw[-1]
    z_above = np.maximum(z_raw - z_base, 0.0)
    mask = z_above > 1e-9
    if mask.sum() < 3:
        return float('nan')
    r3 = re_raw[mask] ** 3
    za = z_above[mask]
    if r3.std() < 1e-9 or za.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(r3, za)[0, 1])


def per_profile_predictors(h5_path: str) -> dict:
    """
    Compute per-unique-profile physical predictors that we'll later regress
    the per-profile RMSE against.  Returns dict keyed by 'pid' (numpy int
    array of length n_unique) plus one numpy array per predictor.
    """
    pids = compute_profile_ids(h5_path)
    n_unique = int(pids.max()) + 1

    with h5py.File(h5_path, 'r') as f:
        profs_raw = f['profiles_raw'][:]
        z_raw     = f['profiles_raw_z'][:]
        n_levels  = f['profile_n_levels'][:]
        tau_c     = f['tau_c'][:]
        wv_above  = f['wv_above_cloud'][:]
        wv_in     = f['wv_in_cloud'][:]

    out = {
        'pid':                  np.arange(n_unique, dtype=np.int32),
        'n_raw_levels':         np.zeros(n_unique, dtype=np.int32),
        'tau_c':                np.full(n_unique, np.nan, dtype=np.float32),
        'wv_above_cloud':       np.full(n_unique, np.nan, dtype=np.float64),
        'wv_in_cloud':          np.full(n_unique, np.nan, dtype=np.float64),
        'adiab_score':          np.full(n_unique, np.nan, dtype=np.float32),
        're_top':               np.full(n_unique, np.nan, dtype=np.float32),
        're_base':              np.full(n_unique, np.nan, dtype=np.float32),
        're_max_lower30pct':    np.full(n_unique, np.nan, dtype=np.float32),
        're_max_overall':       np.full(n_unique, np.nan, dtype=np.float32),
        'z_top_km':             np.full(n_unique, np.nan, dtype=np.float32),
        'z_base_km':            np.full(n_unique, np.nan, dtype=np.float32),
    }

    for pid in range(n_unique):
        i  = int(np.where(pids == pid)[0][0])   # any sample of this profile
        nL = int(n_levels[i])
        re = profs_raw[i, :nL].astype(np.float64)
        z  = z_raw[i, :nL].astype(np.float64)

        out['n_raw_levels'][pid]      = nL
        out['tau_c'][pid]             = float(tau_c[i])
        out['wv_above_cloud'][pid]    = float(wv_above[i])
        out['wv_in_cloud'][pid]       = float(wv_in[i])
        out['adiab_score'][pid]       = adiabaticity_pearson_r(re, z)
        out['re_top'][pid]            = float(re[0])
        out['re_base'][pid]           = float(re[-1])
        out['re_max_lower30pct'][pid] = float(re[int(0.7 * nL):].max())
        out['re_max_overall'][pid]    = float(re.max())
        out['z_top_km'][pid]          = float(z[0])
        out['z_base_km'][pid]         = float(z[-1])

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────────
def train_one_fold(fold_idx: int, n_folds: int,
                   cfg: dict, hp: dict, h5_path: Path,
                   output_dir: Path, device: torch.device,
                   args) -> Path:
    """
    Train ProfileOnlyNetwork on one rotating-K-fold split.  Saves all
    artifacts under output_dir / fold_<idx> / .
    """
    fold_dir = output_dir / f'fold_{fold_idx:02d}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 70}")
    print(f"  FOLD {fold_idx} / {n_folds}")
    print(f"{'─' * 70}")

    # Dataloaders
    train_loader, val_loader, test_loader = create_rotating_kfold_dataloaders(
        h5_path=str(h5_path),
        fold_idx=fold_idx,
        n_folds=n_folds,
        batch_size=hp.get('batch_size', 256),
        num_workers=cfg['data'].get('num_workers', 4),
        seed=args.seed,
        instrument=cfg['data'].get('instrument', 'hysics'),
        n_val_profiles=args.n_val_profiles,
    )

    # n_levels from underlying dataset (Subset doesn't forward attributes)
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels

    # Model + loss + optimizer
    model_config = RetrievalConfig(
        n_wavelengths=636, n_geometry_inputs=4, n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    model = ProfileOnlyNetwork(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} parameters")

    level_weights = torch.tensor(hp['level_weights'], dtype=torch.float32)
    criterion = ProfileOnlyLoss(
        config=model_config,
        lambda_physics=hp.get('lambda_physics', 0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic', 0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        level_weights=level_weights,
        sigma_floor=hp.get('sigma_floor', 0.01),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=hp.get('scheduler_patience', 30),
                                  min_lr=1e-6)

    # Training loop
    n_epochs            = hp.get('n_epochs', 1000)
    early_stop_patience = hp.get('early_stop_patience', 150)
    warmup_steps        = hp.get('warmup_steps', 500)
    augment_noise_std   = hp.get('augment_noise_std', 0.0)
    target_lr           = hp['learning_rate']

    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0
    history = {'train': [], 'val': []}

    t0 = time.time()
    final_epoch = -1
    for epoch in range(n_epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=augment_noise_std,
            warmup_steps=warmup_steps, target_lr=target_lr,
            global_step_start=global_step,
        )
        val_loss = validate(model, val_loader, criterion, device)
        if global_step >= warmup_steps:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'fold_idx': fold_idx,
                'n_folds': n_folds,
                'model_config': {
                    'n_wavelengths':     model_config.n_wavelengths,
                    'n_geometry_inputs': model_config.n_geometry_inputs,
                    'n_levels':          model_config.n_levels,
                    'hidden_dims':       list(model_config.hidden_dims),
                    'dropout':           model_config.dropout,
                    'activation':        model_config.activation,
                },
            }, fold_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  Early stop @ epoch {epoch} "
                      f"(no improve in {early_stop_patience})")
                final_epoch = epoch
                break
        final_epoch = epoch
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:4d} | Train {train_loss:+.4f} | "
                  f"Val {val_loss:+.4f} | LR {lr:.1e} | "
                  f"NoImpr {epochs_no_improve}")

    train_time = time.time() - t0
    print(f"  Fold {fold_idx} done in {train_time:.0f}s ({final_epoch + 1} epochs)")

    # Best-checkpoint test evaluation + per-sample save
    ckpt = torch.load(fold_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    results = predict_test(model, test_loader, device, model_config)

    # Map test samples back to global HDF5 row indices and pids
    subset = test_loader.dataset
    if not hasattr(subset, 'indices'):
        raise RuntimeError("Expected test_loader.dataset to be a torch Subset.")
    test_idx_global = np.asarray(subset.indices, dtype=np.int64)
    pids_all = compute_profile_ids(str(h5_path))
    test_pids = pids_all[test_idx_global].astype(np.int32)

    np.savez(fold_dir / 'test_predictions.npz',
             hdf5_indices=test_idx_global,
             profile_ids=test_pids,
             pred=results['pred'].astype(np.float32),
             pred_std=results['pred_std'].astype(np.float32),
             true=results['true'].astype(np.float32))

    rmse_per_level = np.sqrt(np.mean((results['pred'] - results['true']) ** 2, axis=0))
    fold_summary = {
        'fold_idx':           int(fold_idx),
        'n_folds':            int(n_folds),
        'best_epoch':         int(ckpt['epoch']),
        'final_epoch':        int(final_epoch),
        'train_time_seconds': float(train_time),
        'best_val_loss':      float(best_val_loss),
        'mean_rmse':          float(rmse_per_level.mean()),
        'rmse_per_level':     rmse_per_level.tolist(),
        'test_pids_in_fold':  sorted(set(test_pids.tolist())),
        'n_test_samples':     int(len(test_idx_global)),
    }
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(fold_summary, f, indent=2)
    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(fold_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"  Fold {fold_idx} test mean RMSE: {fold_summary['mean_rmse']:.3f} μm  "
          f"({len(set(test_pids.tolist()))} unique profiles in test)")
    return fold_dir


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate K folds → per-profile summary, correlations, plots
# ─────────────────────────────────────────────────────────────────────────────
def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ via ranks; no scipy dependency."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x[mask]))
    ry = np.argsort(np.argsort(y[mask]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    if x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def aggregate_folds(output_dir: Path, h5_path: Path, n_folds: int):
    """
    Walk fold_NN/test_predictions.npz, build a per-profile RMSE table, attach
    physical predictors, compute correlations, and write CSV + JSON + figures.
    """
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATING {n_folds} FOLDS")
    print(f"{'=' * 70}")

    # 1.  Stitch test predictions across folds
    pred_all, true_all, std_all, pid_all = [], [], [], []
    fold_summaries = []
    for k in range(n_folds):
        fold_dir = output_dir / f'fold_{k:02d}'
        npz_path = fold_dir / 'test_predictions.npz'
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing {npz_path} — run --fold-idx {k} first, or rerun "
                f"the script without --aggregate-only to train any missing folds."
            )
        npz = np.load(npz_path)
        pred_all.append(npz['pred']);    true_all.append(npz['true'])
        std_all.append(npz['pred_std']); pid_all.append(npz['profile_ids'])
        with open(fold_dir / 'results.json') as f:
            fold_summaries.append(json.load(f))

    pred_all = np.concatenate(pred_all)         # (n_test_total, n_levels)
    true_all = np.concatenate(true_all)
    std_all  = np.concatenate(std_all)
    pid_all  = np.concatenate(pid_all)
    n_levels = pred_all.shape[1]
    print(f"  Stitched {pred_all.shape[0]:,} test samples across {n_folds} folds")
    print(f"  Profile coverage: {len(set(pid_all.tolist()))} unique profiles in test "
          f"(should equal n_unique = {int(compute_profile_ids(str(h5_path)).max())+1})")

    # 2.  Per-profile RMSE
    pids_unique = np.array(sorted(set(pid_all.tolist())), dtype=np.int32)
    n_unique = len(pids_unique)
    per_profile_rmse        = np.full((n_unique, n_levels), np.nan, dtype=np.float64)
    per_profile_pred_sigma  = np.full((n_unique, n_levels), np.nan, dtype=np.float64)
    per_profile_n_samples   = np.zeros(n_unique, dtype=np.int64)

    for j, pid in enumerate(pids_unique):
        sel = (pid_all == pid)
        per_profile_n_samples[j]   = int(sel.sum())
        sq_err = (pred_all[sel] - true_all[sel]) ** 2     # (n_samples_j, n_levels)
        per_profile_rmse[j]        = np.sqrt(sq_err.mean(axis=0))
        per_profile_pred_sigma[j]  = std_all[sel].mean(axis=0)

    mean_rmse_per_profile = per_profile_rmse.mean(axis=1)   # (n_unique,)
    mean_pred_sigma_per_profile = per_profile_pred_sigma.mean(axis=1)
    print(f"  Per-profile mean RMSE: "
          f"min {mean_rmse_per_profile.min():.3f}  "
          f"p25 {np.percentile(mean_rmse_per_profile, 25):.3f}  "
          f"p50 {np.percentile(mean_rmse_per_profile, 50):.3f}  "
          f"p75 {np.percentile(mean_rmse_per_profile, 75):.3f}  "
          f"max {mean_rmse_per_profile.max():.3f}  μm")

    # 3.  Per-profile predictors (read once from the HDF5)
    preds = per_profile_predictors(str(h5_path))
    # Order the predictor columns to match pids_unique row ordering.
    ord_idx = preds['pid'][pids_unique]   # since preds['pid'] is just np.arange
    def col(name): return preds[name][pids_unique]
    tau   = col('tau_c')
    wv_a  = col('wv_above_cloud').astype(np.float64)
    wv_i  = col('wv_in_cloud').astype(np.float64)
    adi   = col('adiab_score')
    re_t  = col('re_top')
    re_b  = col('re_base')
    re_d  = col('re_max_lower30pct')
    re_M  = col('re_max_overall')
    nrl   = col('n_raw_levels')

    # 4.  Correlations
    predictors = {
        'tau_c':                  tau,
        'log10_wv_above_cloud':   np.log10(wv_a),
        'log10_wv_in_cloud':      np.log10(wv_i),
        'adiabaticity_score':     adi,
        're_top_um':              re_t,
        're_base_um':             re_b,
        'drizzle_proxy_re_max_lower30pct_um': re_d,
        're_max_overall_um':      re_M,
        'n_raw_levels':           nrl.astype(np.float64),
    }
    correlations = {}
    print(f"\n  Per-profile RMSE correlations  (mean RMSE μm  vs  predictor):")
    print(f"  {'predictor':<38s}  {'pearson r':>10s}  {'spearman ρ':>11s}")
    print(f"  {'-' * 38}  {'-' * 10}  {'-' * 11}")
    for name, x in predictors.items():
        pr = _pearson_r(x, mean_rmse_per_profile)
        sp = _spearman_r(x, mean_rmse_per_profile)
        correlations[name] = {'pearson_r': pr, 'spearman_rho': sp}
        print(f"  {name:<38s}  {pr:+10.3f}  {sp:+11.3f}")

    overall_summary = {
        'n_folds':              int(n_folds),
        'n_unique_profiles':    int(n_unique),
        'n_test_samples_total': int(pred_all.shape[0]),
        'pooled_mean_rmse':     float(np.sqrt(((pred_all - true_all) ** 2).mean())),
        'mean_of_per_profile_rmse':   float(mean_rmse_per_profile.mean()),
        'std_of_per_profile_rmse':    float(mean_rmse_per_profile.std()),
        'median_of_per_profile_rmse': float(np.median(mean_rmse_per_profile)),
        'p5_per_profile_rmse':        float(np.percentile(mean_rmse_per_profile, 5)),
        'p95_per_profile_rmse':       float(np.percentile(mean_rmse_per_profile, 95)),
        'fold_mean_rmse':       [fs['mean_rmse'] for fs in fold_summaries],
        'fold_best_epoch':      [fs['best_epoch'] for fs in fold_summaries],
        'fold_train_time_s':    [fs['train_time_seconds'] for fs in fold_summaries],
        'predictor_correlations': correlations,
    }
    with open(output_dir / 'overall_summary.json', 'w') as f:
        json.dump(overall_summary, f, indent=2)

    # 5.  Per-profile CSV
    csv_path = output_dir / 'per_profile_summary.csv'
    fieldnames = (['pid', 'n_test_samples', 'mean_rmse_um',
                   'mean_pred_sigma_um', 'rmse_sigma_ratio',
                   'tau_c', 'wv_above_cloud', 'wv_in_cloud',
                   'adiabaticity_score',
                   're_top_um', 're_base_um',
                   'drizzle_proxy_re_max_lower30pct_um', 're_max_overall_um',
                   'n_raw_levels']
                   + [f'rmse_L{L+1:02d}_um' for L in range(n_levels)])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j, pid in enumerate(pids_unique):
            row = {
                'pid': int(pid),
                'n_test_samples':   int(per_profile_n_samples[j]),
                'mean_rmse_um':     float(mean_rmse_per_profile[j]),
                'mean_pred_sigma_um': float(mean_pred_sigma_per_profile[j]),
                'rmse_sigma_ratio': float(mean_rmse_per_profile[j]
                                          / max(mean_pred_sigma_per_profile[j], 1e-12)),
                'tau_c':            float(tau[j]),
                'wv_above_cloud':   float(wv_a[j]),
                'wv_in_cloud':      float(wv_i[j]),
                'adiabaticity_score': float(adi[j]),
                're_top_um':        float(re_t[j]),
                're_base_um':       float(re_b[j]),
                'drizzle_proxy_re_max_lower30pct_um': float(re_d[j]),
                're_max_overall_um': float(re_M[j]),
                'n_raw_levels':     int(nrl[j]),
            }
            for L in range(n_levels):
                row[f'rmse_L{L+1:02d}_um'] = float(per_profile_rmse[j, L])
            writer.writerow(row)
    print(f"\nPer-profile CSV    : {csv_path}")
    with open(output_dir / 'per_profile_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=2)
    print(f"Correlations JSON  : {output_dir / 'per_profile_correlations.json'}")

    # 6.  Figures
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 6a. Histogram of per-profile mean RMSE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_rmse_per_profile, bins=30, color='#3B82F6', edgecolor='white')
    ax.axvline(np.median(mean_rmse_per_profile), color='#EF4444',
               linewidth=1.5, label=f"median = {np.median(mean_rmse_per_profile):.2f} μm")
    ax.axvline(mean_rmse_per_profile.mean(), color='#10B981',
               linewidth=1.5, linestyle='--',
               label=f"mean = {mean_rmse_per_profile.mean():.2f} μm")
    ax.set_xlabel('Per-profile mean RMSE (μm)')
    ax.set_ylabel('Number of profiles')
    ax.set_title(f'Per-Profile RMSE Distribution  '
                 f'(K={n_folds}, n_unique={n_unique} profiles, '
                 f'each tested exactly once)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'per_profile_rmse_distribution.png', dpi=400)
    plt.close()

    # 6b. Scatter: mean RMSE vs each predictor (3x2 panel, 6 predictors)
    panel = [
        ('tau_c',                    'τ_c',                       'linear'),
        ('log10_wv_above_cloud',     'log10(wv_above_cloud)',     'linear'),
        ('log10_wv_in_cloud',        'log10(wv_in_cloud)',        'linear'),
        ('adiabaticity_score',       'adiabaticity (Pearson r of r_e³ vs z above base)', 'linear'),
        ('drizzle_proxy_re_max_lower30pct_um', 'drizzle proxy: max r_e in base 30 % (μm)', 'linear'),
        ('re_max_overall_um',        'max r_e in profile (μm)',   'linear'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for ax, (key, lbl, scale) in zip(axes.flat, panel):
        x = predictors[key]
        ax.scatter(x, mean_rmse_per_profile, alpha=0.55, s=24,
                   c='#3B82F6', edgecolors='white', linewidth=0.5)
        ax.set_xlabel(lbl)
        ax.set_ylabel('Per-profile mean RMSE (μm)')
        if scale == 'log':
            ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        pr = correlations[key]['pearson_r']
        sp = correlations[key]['spearman_rho']
        ax.set_title(f'r = {pr:+.3f},  ρ = {sp:+.3f}', fontsize=11)
    fig.suptitle(f'Per-Profile RMSE vs Physical Predictors  '
                 f'(K={n_folds}, n_unique={n_unique})', fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(fig_dir / 'rmse_vs_predictors.png', dpi=400, bbox_inches='tight')
    plt.close()

    # 6c. Per-level uncertainty: mean ± std across profiles
    fig, ax = plt.subplots(figsize=(7, 9))
    levels = np.arange(1, n_levels + 1)
    mu  = per_profile_rmse.mean(axis=0)
    sd  = per_profile_rmse.std(axis=0)
    ax.plot(mu, levels, '-', color='#10B981', linewidth=1.6,
            label='mean RMSE across profiles')
    ax.fill_betweenx(levels, mu - sd, mu + sd, color='#10B981', alpha=0.20,
                     label='±1 std across profiles')
    ax.set_ylabel(f'Vertical level (1 = top, {n_levels} = base)')
    ax.set_xlabel('RMSE (μm)')
    ax.set_title(f'Per-Level RMSE Uncertainty  '
                 f'(across {n_unique} unique profiles, each tested once)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'per_level_uncertainty.png', dpi=400)
    plt.close()

    # 6d. Heatmap: per-profile per-level RMSE (rows = profiles sorted by mean RMSE)
    sort_idx = np.argsort(mean_rmse_per_profile)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(per_profile_rmse[sort_idx], aspect='auto',
                   cmap='magma', vmin=0, vmax=np.percentile(per_profile_rmse, 99))
    ax.set_xlabel('Vertical level (1 = top, 50 = base)')
    ax.set_ylabel('Unique profile (sorted by mean RMSE)')
    ax.set_title('Per-Profile × Per-Level RMSE')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE (μm)')
    plt.tight_layout()
    plt.savefig(fig_dir / 'per_profile_rmse_heatmap.png', dpi=400)
    plt.close()
    print(f"Figures            : {fig_dir}/")

    # 7.  Stdout summary
    print(f"\n{'=' * 70}")
    print(f"  HEADLINE")
    print(f"{'=' * 70}")
    print(f"  pooled mean RMSE         : {overall_summary['pooled_mean_rmse']:.3f} μm")
    print(f"  mean of per-profile RMSE : {overall_summary['mean_of_per_profile_rmse']:.3f} μm")
    print(f"  std  of per-profile RMSE : {overall_summary['std_of_per_profile_rmse']:.3f} μm "
          f"(profile-to-profile variability)")
    print(f"  p5  per-profile RMSE     : {overall_summary['p5_per_profile_rmse']:.3f} μm")
    print(f"  p95 per-profile RMSE     : {overall_summary['p95_per_profile_rmse']:.3f} μm")
    print(f"\n  fold-level mean RMSEs    : "
          f"min {min(overall_summary['fold_mean_rmse']):.3f}  "
          f"max {max(overall_summary['fold_mean_rmse']):.3f}  "
          f"mean {np.mean(overall_summary['fold_mean_rmse']):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    # Locate config
    sweep_dir = (repo_root / args.sweep_dir).resolve()
    run_dir   = sweep_dir / f'run_{args.run_id:03d}'
    cfg_path  = run_dir / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'No config found at {cfg_path}')
    with open(cfg_path) as f:
        cfg = json.load(f)
    hp = cfg['hyperparams']

    # HDF5 path
    h5_path = resolve_h5_path(args.h5_path, args.training_data_dir).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 file not found: {h5_path}')
    cfg.setdefault('data', {})
    cfg['data']['h5_path'] = str(h5_path)

    if args.n_epochs is not None:
        hp['n_epochs'] = args.n_epochs

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (repo_root / 'standalone_results_profile_only_kfold'
                      / f'run{args.run_id:03d}_{h5_path.stem}_K{args.n_folds}'
                      ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate-only short-circuit
    if args.aggregate_only:
        aggregate_folds(output_dir, h5_path, args.n_folds)
        return

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'cpu':
        print("  [warn] running on CPU — expect minutes per epoch.")

    # Banner
    print(f"\n{'=' * 70}")
    print(f"  FULL-COVERAGE K-FOLD STANDALONE TRAIN  "
          f"(run_{args.run_id:03d}, K={args.n_folds})")
    print(f"{'=' * 70}")
    print(f"  sweep run dir : {run_dir}")
    print(f"  HDF5 file     : {h5_path}")
    print(f"  output dir    : {output_dir}")
    print(f"  device        : {device}")
    if args.fold_idx is not None:
        print(f"  mode          : single fold (--fold-idx {args.fold_idx})")
    else:
        print(f"  mode          : sequential — all {args.n_folds} folds, then aggregate")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Level-weights handling: read n_levels from a quick HDF5 peek
    with h5py.File(h5_path, 'r') as f:
        n_levels = int(f['profiles'].shape[1])
    if args.level_weights is not None:
        hp['level_weights']      = build_level_weights(args.level_weights, n_levels)
        hp['level_weights_name'] = args.level_weights
    elif len(hp.get('level_weights', [])) != n_levels:
        raise ValueError(
            f"level_weights in config has {len(hp.get('level_weights', []))} entries "
            f"but the HDF5 grid has {n_levels} levels.  Pass --level-weights uniform "
            f"(or another scheme) to rebuild for this n_levels."
        )

    # Train
    if args.fold_idx is not None:
        if not (0 <= args.fold_idx < args.n_folds):
            raise ValueError(f"--fold-idx {args.fold_idx} out of [0, {args.n_folds})")
        train_one_fold(args.fold_idx, args.n_folds,
                       cfg, hp, h5_path, output_dir, device, args)
        print(f"\nDone with fold {args.fold_idx}.  When all folds complete, "
              f"run with --aggregate-only to build the per-profile summary.")
    else:
        for k in range(args.n_folds):
            train_one_fold(k, args.n_folds,
                           cfg, hp, h5_path, output_dir, device, args)
        # Aggregate
        aggregate_folds(output_dir, h5_path, args.n_folds)


if __name__ == '__main__':
    main()
