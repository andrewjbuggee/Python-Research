"""
sweep_train_profile_only.py — Single-config training with profile-aware K-fold CV.

Differences vs. sweep_train.py
------------------------------
1. PROFILE-ONLY MODEL.  Uses ProfileOnlyNetwork + ProfileOnlyLoss from
   models_profile_only.py.  No τ head, no τ NLL term, no τ test metric.

2. K-FOLD CV PER CONFIG.  For each hyperparameter configuration, this script
   runs `n_folds` independent trainings (default 5) on profile-aware splits.
   Each fold uses 4/5 of training profiles for training and 1/5 for
   validation; the same final-test set (n_test_profiles unique profiles,
   never used in training/validation) is evaluated by every fold.

   Per-fold metrics (val NLL, test mean RMSE, test per-level RMSE, σ
   calibration, training history) are saved.  An aggregate `summary` block
   reports mean ± std across folds for the headline numbers.  This gives
   a real uncertainty quantification on each config's test RMSE rather than
   the single number a one-shot train/val/test split produces.

3. SAME CONFIG SCHEMA as sweep_train.py.  Drop-in replacement for the
   existing sweep pipeline; the only required new field is
   `hyperparams.n_folds` (defaults to 5 if absent).

Usage
-----
    python sweep_train_profile_only.py --config-json sweep_configs_profile/run_007.json
    [--training-data-dir <path>]

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models_profile_only import ProfileOnlyNetwork, ProfileOnlyLoss
from models import RetrievalConfig
from data import create_kfold_dataloaders, resolve_h5_path


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Profile-only single-config trainer with K-fold CV"
    )
    p.add_argument("--config-json", type=str, required=True,
                   help="Path to JSON config for this run")
    p.add_argument("--training-data-dir", type=str, default=None,
                   help="Override directory portion of h5_path (filename "
                        "preserved). Use to switch between Alpine and a "
                        "local copy without regenerating sweep configs.")
    p.add_argument("--instrument", type=str, default=None,
                   choices=['hysics', 'emit'],
                   help="Override the instrument stored in the config "
                        "(cfg['data']['instrument']).  Use to retrain a "
                        "HySICS-tuned config on the EMIT spectra (different "
                        "noise level baked into the HDF5) without "
                        "regenerating the sweep configs.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override the output directory stored in the config "
                        "(cfg['output_dir']).  Use to keep EMIT-trained "
                        "results separate from the HySICS-trained ones, e.g. "
                        "--output-dir sweep_results_profile_only_emit .")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers (profile-only)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    augment_noise_std=0.0,
                    warmup_steps=0, target_lr=None, global_step_start=0):
    model.train()
    loss_sum, n = 0.0, 0
    step = global_step_start
    for batch in loader:
        # The dataset still yields (x, profile, tau) — we ignore tau.
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
    detail = {}
    for batch in loader:
        x, prof = batch[0].to(device), batch[1].to(device)
        output = model(x)
        losses = criterion(output, prof)
        loss_sum += losses['total'].item()
        for k, v in losses.items():
            detail[k] = detail.get(k, 0.0) + v.item()
        n += 1
    return loss_sum / n, {k: v / n for k, v in detail.items()}


@torch.no_grad()
def compute_test_metrics(model, loader, device, model_config):
    """Per-level RMSE in physical units (μm).  Profile-only — no τ."""
    model.eval()
    all_pred, all_true, all_std = [], [], []
    re_min, re_max = model_config.re_min, model_config.re_max

    for batch in loader:
        x, prof_true = batch[0].to(device), batch[1]
        output = model(x)
        all_pred.append(output['profile'].cpu().numpy())
        all_true.append(prof_true.numpy() * (re_max - re_min) + re_min)
        all_std.append(output['profile_std'].cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    std  = np.concatenate(all_std)

    rmse_per_level = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    return {
        'rmse_per_level':       rmse_per_level.tolist(),
        'mean_rmse':            float(rmse_per_level.mean()),
        'mean_std_per_level':   std.mean(axis=0).tolist(),
        'mean_std_overall':     float(std.mean()),
        'n_test_samples':       len(pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# One fold = one full training run
# ─────────────────────────────────────────────────────────────────────────────
def train_one_fold(fold_idx: int, n_folds: int,
                   cfg: dict, hp: dict, h5_path: str,
                   device: torch.device,
                   fold_output_dir: Path) -> dict:
    """
    Train one fold from scratch.  Returns a dict with this fold's history,
    best val loss, best epoch, and test metrics.
    """
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'─' * 70}")
    print(f"  FOLD {fold_idx} / {n_folds}")
    print(f"{'─' * 70}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_kfold_dataloaders(
        h5_path=h5_path,
        fold_idx=fold_idx,
        n_folds=n_folds,
        batch_size=hp.get('batch_size', 256),
        num_workers=cfg['data'].get('num_workers', 4),
        seed=cfg.get('seed', 42),
        instrument=cfg['data'].get('instrument', 'hysics'),
        n_test_profiles=cfg.get('n_test_profiles', 14),
    )

    # n_levels read from the underlying dataset (handles Subset wrapper).
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels

    # ── Model ────────────────────────────────────────────────────────────
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    model = ProfileOnlyNetwork(model_config).to(device)

    # ── Loss ─────────────────────────────────────────────────────────────
    level_weights = torch.tensor(hp['level_weights'], dtype=torch.float32)
    if level_weights.numel() != n_levels:
        raise ValueError(
            f"level_weights has {level_weights.numel()} entries but the HDF5 "
            f"contains {n_levels} profile levels."
        )
    sigma_floor = hp.get('sigma_floor', 0.01)
    criterion = ProfileOnlyLoss(
        config=model_config,
        lambda_physics=hp.get('lambda_physics', 0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic', 0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        level_weights=level_weights,
        sigma_floor=sigma_floor,
    ).to(device)

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=hp.get('scheduler_patience', 30),
                                  min_lr=1e-6)

    # ── Training loop ────────────────────────────────────────────────────
    n_epochs            = hp.get('n_epochs', 400)
    early_stop_patience = hp.get('early_stop_patience', 80)
    warmup_steps        = hp.get('warmup_steps', 0)
    augment_noise_std   = hp.get('augment_noise_std', 0.0)
    target_lr           = hp['learning_rate']

    best_val_loss     = float('inf')
    epochs_no_improve = 0
    history = []
    global_step = 0
    t_start = time.time()

    for epoch in range(n_epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=augment_noise_std,
            warmup_steps=warmup_steps,
            target_lr=target_lr,
            global_step_start=global_step,
        )
        val_loss, val_detail = validate(model, val_loader, criterion, device)

        if global_step >= warmup_steps:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         lr,
            'val_detail': val_detail,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'model_config': {
                    'n_wavelengths':     model_config.n_wavelengths,
                    'n_geometry_inputs': model_config.n_geometry_inputs,
                    'n_levels':          model_config.n_levels,
                    'hidden_dims':       list(model_config.hidden_dims),
                    'dropout':           model_config.dropout,
                    'activation':        model_config.activation,
                },
            }, fold_output_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"  Early stop @ epoch {epoch} "
                      f"(no improve in {early_stop_patience})")
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | Train {train_loss:+.4f} | "
                  f"Val {val_loss:+.4f} | LR {lr:.1e} | "
                  f"NoImpr {epochs_no_improve}")

    train_time = time.time() - t_start
    final_epoch = epoch
    print(f"  Fold {fold_idx} done in {train_time:.0f}s ({final_epoch + 1} epochs)")

    # ── Test evaluation ──────────────────────────────────────────────────
    best_ckpt = torch.load(fold_output_dir / 'best_model.pt',
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = compute_test_metrics(model, test_loader, device, model_config)
    test_val_loss, _ = validate(model, test_loader, criterion, device)

    print(f"  Fold {fold_idx} test mean RMSE: "
          f"{test_metrics['mean_rmse']:.3f} μm "
          f"(NLL {test_val_loss:+.4f})")

    fold_results = {
        'fold_idx':     fold_idx,
        'n_folds':      n_folds,
        'best_val_loss': float(best_val_loss),
        'best_epoch':   int(best_ckpt['epoch']),
        'final_epoch':  int(final_epoch),
        'train_time_seconds': float(train_time),
        'test_nll':       float(test_val_loss),
        'test_metrics':   test_metrics,
    }

    with open(fold_output_dir / 'results.json', 'w') as f:
        json.dump(fold_results, f, indent=2)
    with open(fold_output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# Main: loop over folds, aggregate
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    with open(args.config_json) as f:
        cfg = json.load(f)

    run_id   = cfg['run_id']
    run_name = cfg.get('run_name', f'run_{run_id:03d}')

    print(f"\n{'=' * 70}")
    print(f"  PROFILE-ONLY SWEEP RUN {run_id:03d} (K-fold CV): {run_name}")
    print(f"{'=' * 70}")
    for k, v in cfg['hyperparams'].items():
        print(f"  {k}: {v}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    hp = cfg['hyperparams']
    n_folds = int(hp.get('n_folds', 5))

    # CLI overrides take precedence over the config-stored values so the
    # same JSON can drive both HySICS and EMIT runs without being edited.
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.instrument is not None:
        cfg.setdefault('data', {})['instrument'] = args.instrument
    print(f"Instrument:  {cfg['data'].get('instrument', 'hysics')}  "
          f"{'(CLI override)' if args.instrument else '(from config)'}")
    print(f"Output dir:  {cfg['output_dir']}  "
          f"{'(CLI override)' if args.output_dir else '(from config)'}")

    output_dir = Path(cfg['output_dir']) / f'run_{run_id:03d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = str(resolve_h5_path(cfg['data']['h5_path'], args.training_data_dir))
    cfg['data']['h5_path'] = h5_path
    print(f"HDF5 file: {h5_path}\n")

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # ── Run K folds sequentially ─────────────────────────────────────────
    fold_results = []
    for fold_idx in range(n_folds):
        fold_output_dir = output_dir / f'fold_{fold_idx}'
        fr = train_one_fold(
            fold_idx=fold_idx,
            n_folds=n_folds,
            cfg=cfg, hp=hp,
            h5_path=h5_path,
            device=device,
            fold_output_dir=fold_output_dir,
        )
        fold_results.append(fr)

    # ── Aggregate ────────────────────────────────────────────────────────
    test_mean_rmse_per_fold = np.array([
        fr['test_metrics']['mean_rmse'] for fr in fold_results])
    test_nll_per_fold = np.array([fr['test_nll'] for fr in fold_results])
    val_loss_per_fold = np.array([fr['best_val_loss'] for fr in fold_results])

    # Per-level RMSE: mean across folds.
    per_level_arr = np.stack([
        np.array(fr['test_metrics']['rmse_per_level']) for fr in fold_results])
    per_level_mean = per_level_arr.mean(axis=0)
    per_level_std  = per_level_arr.std(axis=0)

    summary = {
        'run_id':   run_id,
        'run_name': run_name,
        'n_folds':  n_folds,
        'hyperparams': hp,
        # Headline metric: test mean RMSE, mean ± std across folds.
        'test_mean_rmse_mean':  float(test_mean_rmse_per_fold.mean()),
        'test_mean_rmse_std':   float(test_mean_rmse_per_fold.std()),
        'test_mean_rmse_per_fold': test_mean_rmse_per_fold.tolist(),
        'test_nll_mean':        float(test_nll_per_fold.mean()),
        'test_nll_std':         float(test_nll_per_fold.std()),
        'val_loss_mean':        float(val_loss_per_fold.mean()),
        'val_loss_std':         float(val_loss_per_fold.std()),
        'rmse_per_level_mean':  per_level_mean.tolist(),
        'rmse_per_level_std':   per_level_std.tolist(),
        'fold_results':         fold_results,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  K-FOLD SUMMARY (run {run_id:03d})")
    print(f"{'=' * 70}")
    print(f"  Test mean RMSE:  {summary['test_mean_rmse_mean']:.3f}"
          f" ± {summary['test_mean_rmse_std']:.3f} μm  "
          f"(per-fold: "
          f"{', '.join(f'{v:.3f}' for v in test_mean_rmse_per_fold)})")
    print(f"  Test NLL:        {summary['test_nll_mean']:+.4f}"
          f" ± {summary['test_nll_std']:.4f}")
    print(f"  Val loss:        {summary['val_loss_mean']:+.4f}"
          f" ± {summary['val_loss_std']:.4f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
