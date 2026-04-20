"""
sweep_train.py — Single-run training script for hyperparameter sweeps on Alpine.

Called by sweep_alpine.sh with a JSON config via --config-json.
Reads one hyperparameter combination, trains the retrieval network,
saves the best checkpoint + results summary to a numbered output directory.

Usage:
    python sweep_train.py --config-json sweep_configs/run_007.json

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
# This script lives in <repo>/hyper_parameter_sweep/, but models.py and data.py
# live at the repo root one level up.  Add the parent directory to sys.path so
# those modules resolve regardless of where `python sweep_train.py` is invoked
# from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models import DropletProfileNetwork, CombinedLoss, RetrievalConfig
from data import create_dataloaders, resolve_h5_path


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Single sweep run")
    p.add_argument("--config-json", type=str, required=True,
                   help="Path to JSON config for this run")
    p.add_argument("--training-data-dir", type=str, default=None,
                   help="Directory hosting the HDF5 file on this machine. If "
                        "given, replaces the directory portion of the h5_path "
                        "stored in the config (filename is preserved). Use this "
                        "to switch between Alpine "
                        "(/scratch/alpine/anbu8374/neural_network_training_data/) "
                        "and a local copy without regenerating sweep configs.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers (same logic as the notebook, extracted into functions)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    augment_noise_std=0.0,
                    warmup_steps=0, target_lr=None, global_step_start=0):
    """
    One training pass.

    Extras vs. a vanilla loop:
      - augment_noise_std: fractional Gaussian noise added to input reflectance
          each batch.  x -> x + sigma * x * N(0, 1).  Applied ON TOP OF any
          noise already baked into the data.  Zero disables augmentation.
          Only the spectral portion (first n_wavelengths channels) is perturbed;
          geometry inputs are left untouched.  The model currently takes a
          concatenated [spectrum, geometry] vector — we perturb the whole
          vector here because the geometry channels carry much smaller noise
          sensitivity in practice, but if that becomes an issue change this
          to perturb only x[:, :n_wavelengths].
      - warmup_steps / target_lr: if global_step < warmup_steps, override the
          optimizer LR with a linear ramp from 0 to target_lr.  After warmup,
          the LR is left to whatever the scheduler (in the outer loop) set.

    Returns (mean_loss, global_step_end).
    """
    model.train()
    loss_sum, n = 0.0, 0
    step = global_step_start
    for x, prof, tau in loader:
        x, prof, tau = x.to(device), prof.to(device), tau.to(device)

        # Linear warmup
        if warmup_steps > 0 and step < warmup_steps and target_lr is not None:
            lr_now = target_lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

        # Input-noise augmentation (per-batch)
        if augment_noise_std > 0.0:
            x = x + augment_noise_std * x * torch.randn_like(x)

        optimizer.zero_grad()
        output = model(x)
        losses = criterion(output, prof, tau)
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
    for x, prof, tau in loader:
        x, prof, tau = x.to(device), prof.to(device), tau.to(device)
        output = model(x)
        losses = criterion(output, prof, tau)
        loss_sum += losses['total'].item()
        for k, v in losses.items():
            detail[k] = detail.get(k, 0.0) + v.item()
        n += 1
    return loss_sum / n, {k: v / n for k, v in detail.items()}


@torch.no_grad()
def compute_test_metrics(model, loader, device, config):
    """Compute per-level RMSE on test set in physical units (um)."""
    model.eval()
    all_pred, all_true, all_std = [], [], []
    all_tau_pred, all_tau_true, all_tau_std = [], [], []

    re_min, re_max = config.re_min, config.re_max
    tau_min, tau_max = config.tau_min, config.tau_max

    for x, prof_true, tau_true in loader:
        output = model(x.to(device))

        # Profile: output is already in physical units
        all_pred.append(output['profile'].cpu().numpy())
        all_true.append(prof_true.numpy() * (re_max - re_min) + re_min)
        all_std.append(output['profile_std'].cpu().numpy())

        # Tau
        all_tau_pred.append(output['tau_c'].squeeze(-1).cpu().numpy())
        all_tau_true.append(tau_true.numpy() * (tau_max - tau_min) + tau_min)
        all_tau_std.append(output['tau_std'].squeeze(-1).cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    std  = np.concatenate(all_std)
    tau_pred = np.concatenate(all_tau_pred)
    tau_true = np.concatenate(all_tau_true)
    tau_std  = np.concatenate(all_tau_std)

    # Per-level RMSE
    rmse_per_level = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    mean_rmse = rmse_per_level.mean()
    tau_rmse = np.sqrt(np.mean((tau_pred - tau_true) ** 2))

    # Mean predicted uncertainty (calibration diagnostic)
    mean_std_per_level = std.mean(axis=0)

    return {
        'rmse_per_level': rmse_per_level.tolist(),
        'mean_rmse': float(mean_rmse),
        'tau_rmse': float(tau_rmse),
        'mean_std_per_level': mean_std_per_level.tolist(),
        'mean_std_overall': float(std.mean()),
        'mean_tau_std': float(tau_std.mean()),
        'n_test_samples': len(pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load this run's config
    with open(args.config_json) as f:
        cfg = json.load(f)

    run_id   = cfg['run_id']
    run_name = cfg.get('run_name', f'run_{run_id:03d}')

    print(f"\n{'='*70}")
    print(f"  SWEEP RUN {run_id:03d}: {run_name}")
    print(f"{'='*70}")
    for k, v in cfg['hyperparams'].items():
        print(f"  {k}: {v}")
    print()

    # ── Setup ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    hp = cfg['hyperparams']

    # Output directory for this run
    output_dir = Path(cfg['output_dir']) / f'run_{run_id:03d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the actual on-disk path.  If --training-data-dir was provided,
    # only the filename of cfg['data']['h5_path'] is preserved; the directory
    # is overridden.  Update the saved config to reflect what was actually used.
    h5_path = str(resolve_h5_path(cfg['data']['h5_path'], args.training_data_dir))
    cfg['data']['h5_path'] = h5_path
    print(f"HDF5 file: {h5_path}")

    # Save the config into the run directory for reproducibility (with the
    # path that was actually used, post-override).
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=h5_path,
        instrument=cfg['data'].get('instrument', 'hysics'),
        batch_size=hp.get('batch_size', 256),
        num_workers=cfg['data'].get('num_workers', 4),
        seed=42,
        profile_holdout=True,
        n_val_profiles=14,
        n_test_profiles=14,
    )
    print(f"Data: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # n_levels is data-driven: read from the HDF5 via the dataset.  This lets
    # you change N_LEVELS in convert_matFiles_to_HDF.py and regenerate data
    # without having to update the training code.
    #
    # Note: profile_holdout=True wraps the RetrievalDataset in torch's Subset
    # for the train/val/test split.  Subset does not forward attribute access
    # to the underlying dataset, so we have to unwrap it.  Do this
    # defensively (walk `.dataset` until we hit the real thing) to also
    # handle potential future wrappers like ConcatDataset.
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels
    print(f"Profile grid: {n_levels} levels (read from HDF5)")

    # ── Model ──────────────────────────────────────────────────────────────
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation='gelu',
    )
    model = DropletProfileNetwork(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters, "
          f"hidden_dims={hp['hidden_dims']}, dropout={hp['dropout']}")

    # ── Loss ───────────────────────────────────────────────────────────────
    # Sanity check: level_weights must match n_levels from the HDF5.  If the
    # config was generated for a different profile grid, fail fast with a
    # clear message rather than surfacing a shape mismatch deep in the loss.
    level_weights = torch.tensor(hp['level_weights'], dtype=torch.float32)
    if level_weights.numel() != n_levels:
        raise ValueError(
            f"level_weights has {level_weights.numel()} entries but the HDF5 "
            f"contains {n_levels} profile levels.  Regenerate this sweep's "
            f"configs with the correct N_LEVELS in generate_sweep_2.py."
        )
    sigma_floor = hp.get('sigma_floor', 0.01)
    criterion = CombinedLoss(
        config=model_config,
        lambda_physics=hp.get('lambda_physics', 0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic', 0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        lambda_emulator_data=0.0,
        level_weights=level_weights,
        sigma_floor=sigma_floor,
    ).to(device)
    print(f"Sigma floor: {sigma_floor} (normalized) = "
          f"{sigma_floor * (model_config.re_max - model_config.re_min):.2f} um (physical)")

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=hp.get('scheduler_patience', 30),
                                  min_lr=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────
    n_epochs = hp.get('n_epochs', 400)
    early_stop_patience = hp.get('early_stop_patience', 80)
    warmup_steps = hp.get('warmup_steps', 0)
    augment_noise_std = hp.get('augment_noise_std', 0.0)
    target_lr = hp['learning_rate']
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []
    global_step = 0

    print(f"\nTraining for up to {n_epochs} epochs "
          f"(early stop patience={early_stop_patience})")
    print(f"  warmup_steps      = {warmup_steps}")
    print(f"  augment_noise_std = {augment_noise_std} "
          f"(fractional, added on top of data-level noise)")
    print("-" * 70)

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

        # Only let the plateau scheduler act once warmup is fully done.
        # During warmup the LR is being driven manually by train_one_epoch,
        # so letting ReduceLROnPlateau also step would double-count "no
        # improvement" epochs from the unstable ramp-up phase.
        if global_step >= warmup_steps:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
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
                    'n_wavelengths': model_config.n_wavelengths,
                    'n_geometry_inputs': model_config.n_geometry_inputs,
                    'n_levels': model_config.n_levels,
                    'hidden_dims': model_config.hidden_dims,
                    'dropout': model_config.dropout,
                    'activation': model_config.activation,
                },
            }, output_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement in {early_stop_patience} epochs)")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Train: {train_loss:+.4f} | "
                  f"Val: {val_loss:+.4f} | LR: {lr:.1e} | "
                  f"No-improve: {epochs_no_improve}")

    train_time = time.time() - t_start
    final_epoch = epoch
    print(f"\nTraining complete in {train_time:.0f}s ({final_epoch+1} epochs)")

    # ── Test evaluation ────────────────────────────────────────────────────
    # Reload best checkpoint for test evaluation
    best_ckpt = torch.load(output_dir / 'best_model.pt',
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = compute_test_metrics(model, test_loader, device, model_config)
    test_val_loss, _ = validate(model, test_loader, criterion, device)

    print(f"\nTest results (best checkpoint from epoch {best_ckpt['epoch']}):")
    print(f"  Test NLL:  {test_val_loss:+.4f}")
    print(f"  Mean RMSE: {test_metrics['mean_rmse']:.3f} um")
    print(f"  Tau RMSE:  {test_metrics['tau_rmse']:.3f}")
    for lvl, rmse in enumerate(test_metrics['rmse_per_level'], 1):
        print(f"    Level {lvl:2d}: {rmse:.3f} um")

    # ── Save results summary ──────────────────────────────────────────────
    results = {
        'run_id': run_id,
        'run_name': run_name,
        'hyperparams': hp,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'best_epoch': int(best_ckpt['epoch']),
        'final_epoch': final_epoch,
        'train_time_seconds': train_time,
        'test_nll': test_val_loss,
        'test_metrics': test_metrics,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Also save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nResults saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
