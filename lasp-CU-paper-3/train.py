"""
train.py — SLURM-submittable training script for Alpine HPC.

Usage:
    python train.py --config configs/stage1_modis.yaml

Or via SLURM:
    sbatch scripts/train_alpine.sh

This script:
  1. Loads training data from HDF5
  2. Creates the retrieval network and loss function
  3. Trains with early stopping on validation loss
  4. Saves checkpoints that can be loaded in Jupyter notebooks

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import yaml
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import DropletProfileNetwork, CombinedLoss, RetrievalConfig
from src.data import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train droplet profile retrieval network")
    parser.add_argument("--config", type=str, default="configs/stage1_modis.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss dict."""
    model.train()
    running_losses = {}
    n_batches = 0

    for x, profile_true, tau_true in loader:
        x = x.to(device)
        profile_true = profile_true.to(device)
        tau_true = tau_true.to(device)

        optimizer.zero_grad()
        output = model(x)
        losses = criterion(output, profile_true, tau_true)
        losses['total'].backward()

        # Gradient clipping (helps with stability)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key, val in losses.items():
            running_losses[key] = running_losses.get(key, 0.0) + val.item()
        n_batches += 1

    return {k: v / n_batches for k, v in running_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate on held-out data. Returns average loss dict."""
    model.eval()
    running_losses = {}
    n_batches = 0

    for x, profile_true, tau_true in loader:
        x = x.to(device)
        profile_true = profile_true.to(device)
        tau_true = tau_true.to(device)

        output = model(x)
        losses = criterion(output, profile_true, tau_true)

        for key, val in losses.items():
            running_losses[key] = running_losses.get(key, 0.0) + val.item()
        n_batches += 1

    return {k: v / n_batches for k, v in running_losses.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'config': config,
        'model_config': {
            'n_wavelengths': model.config.n_wavelengths,
            'n_geometry_inputs': model.config.n_geometry_inputs,
            'n_levels': model.config.n_levels,
            'hidden_dims': model.config.hidden_dims,
            'dropout': model.config.dropout,
            'activation': model.config.activation,
        },
    }, path)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Data
    print(f"\nLoading data from {config['data']['h5_path']}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=config['data']['h5_path'],
        batch_size=config['training']['batch_size'],
        train_frac=config['data'].get('train_frac', 0.8),
        val_frac=config['data'].get('val_frac', 0.1),
        num_workers=config['data'].get('num_workers', 4),
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Model
    model_config = RetrievalConfig(
        n_wavelengths=config['model']['n_wavelengths'],
        n_geometry_inputs=config['model'].get('n_geometry_inputs', 4),
        n_levels=config['model']['n_levels'],
        hidden_dims=tuple(config['model']['hidden_dims']),
        dropout=config['model'].get('dropout', 0.1),
        activation=config['model'].get('activation', 'gelu'),
    )
    model = DropletProfileNetwork(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    # Loss
    criterion = CombinedLoss(
        config=model_config,
        lambda_physics=config['loss']['lambda_physics'],
        lambda_monotonicity=config['loss'].get('lambda_monotonicity', 1.0),
        lambda_adiabatic=config['loss'].get('lambda_adiabatic', 0.5),
        lambda_smoothness=config['loss'].get('lambda_smoothness', 0.1),
    ).to(device)

    # Optimizer
    lr = config['training']['learning_rate']
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=config['training'].get('weight_decay', 1e-4))

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=config['training'].get('scheduler_patience', 10),
                                   verbose=True)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"\nResuming from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # Training loop
    n_epochs = config['training']['n_epochs']
    patience = config['training'].get('early_stopping_patience', 20)
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {n_epochs} epochs (early stopping patience={patience})")
    print("=" * 80)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = validate(model, val_loader, criterion, device)

        scheduler.step(val_losses['total'])

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'lr': current_lr,
            'time': elapsed,
        })

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d} | "
                  f"Train: {train_losses['total']:.6f} | "
                  f"Val: {val_losses['total']:.6f} | "
                  f"Phys: {val_losses['physics_total']:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s")

        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                          best_val_loss, config, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

        # Periodic checkpoint
        if epoch % 50 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                          val_losses['total'], config, output_dir / f"checkpoint_epoch{epoch:04d}.pt")

    # Save final model and training history
    save_checkpoint(model, optimizer, scheduler, epoch,
                   val_losses['total'], config, output_dir / "final_model.pt")

    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {output_dir}")

    # Quick test evaluation
    print("\nEvaluating on test set...")
    test_losses = validate(model, test_loader, criterion, device)
    print(f"  Test loss: {test_losses['total']:.6f}")
    print(f"  Profile loss: {test_losses['supervised_profile']:.6f}")
    print(f"  Tau loss: {test_losses['supervised_tau']:.6f}")
    print(f"  Physics loss: {test_losses['physics_total']:.6f}")


if __name__ == "__main__":
    main()
