"""
train_emulator.py — Standalone training script for the ForwardModelEmulator.

The ForwardModelEmulator approximates libRadtran radiative transfer:

    (r_e profile, τ_c, geometry) → R̂(λ)

It must be trained and validated INDEPENDENTLY of the retrieval network.
Once its mean relative error (MRE) drops below the target (default 1%),
the script writes an 'emulator_ready.pt' checkpoint.  Only use the emulator
in Stage 2 PINN training after this checkpoint exists.

Profile-held-out evaluation
---------------------------
All 73 K spectra in the dataset come from only 73 unique in-situ droplet
profiles (varied over many solar/viewing geometries).  A random sample-level
split is therefore invalid for assessing generalisation — every profile leaks
into every split.  This script uses a profile-held-out split by default:
the test set contains profiles that were *never* seen during training.

Usage
-----
    python train_emulator.py --config emulator.yaml
    python train_emulator.py --config emulator.yaml --checkpoint checkpoints/emulator/best_model.pt

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import ForwardModelEmulator, EmulatorConfig, DWMSELoss
from data import create_emulator_dataloaders


# =============================================================================
# Argument parsing + config
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train the ForwardModelEmulator")
    parser.add_argument("--config", type=str, default="emulator.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints/emulator",
                        help="Directory to save checkpoints and diagnostics")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Training and evaluation helpers
# =============================================================================

def _split_input(x: torch.Tensor, n_levels: int):
    """Split concatenated input tensor into (profile, tau_c, geometry)."""
    profile  = x[:, :n_levels]
    tau_c    = x[:, n_levels:n_levels + 1]
    geometry = x[:, n_levels + 1:]
    return profile, tau_c, geometry


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    """Train for one epoch. Returns dict of average loss values."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        profile, tau_c, geometry = _split_input(x, model.config.n_levels)
        pred = model(profile, tau_c, geometry)
        loss = criterion(pred, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return {'loss': total_loss / n_batches}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    """Compute validation loss (DWMSE) and plain MSE for monitoring. Returns loss dict."""
    model.eval()
    total_dwmse = 0.0
    total_mse   = 0.0
    mse_fn      = nn.MSELoss()
    n_batches   = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        profile, tau_c, geometry = _split_input(x, model.config.n_levels)
        pred           = model(profile, tau_c, geometry)
        total_dwmse   += criterion(pred, y).item()
        total_mse     += mse_fn(pred, y).item()
        n_batches     += 1

    return {'loss': total_dwmse / n_batches,
            'mse':  total_mse   / n_batches}


@torch.no_grad()
def compute_mre(model, loader, device) -> float:
    """
    Compute mean relative error (%) across all samples and wavelengths.

        MRE = 100 × mean( |R̂(λ) - R(λ)| / (R(λ) + ε) )

    A small epsilon (1e-4) avoids division by near-zero values in deep
    absorption bands where R(λ) ≈ 0.

    Returns
    -------
    mre : float — overall mean relative error in percent
    """
    model.eval()
    total_rel_err = 0.0
    n_elements    = 0
    eps           = 1e-4

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        profile, tau_c, geometry = _split_input(x, model.config.n_levels)
        pred      = model(profile, tau_c, geometry)
        rel_err   = (pred - y).abs() / (y.abs() + eps)
        total_rel_err += rel_err.sum().item()
        n_elements    += y.numel()

    return 100.0 * total_rel_err / n_elements


@torch.no_grad()
def compute_spectral_residuals(model, loader, device,
                                wavelengths=None) -> dict:
    """
    Compute per-channel residual statistics over the full data loader.

    Returns a dict with keys:
        'mean_rel_error_pct'  : (n_wavelengths,) — mean relative error per channel (%)
        'mean_abs_error'      : (n_wavelengths,) — mean |R̂ - R| per channel
        'rmse'                : (n_wavelengths,) — RMSE per channel
        'bias'                : (n_wavelengths,) — mean signed error per channel
        'worst_channels_idx'  : indices of the 10 channels with highest MRE
        'best_channels_idx'   : indices of the 10 channels with lowest MRE
        'wavelengths'         : wavelength grid (nm) if provided, else None
        'n_samples'           : total sample count
    """
    model.eval()
    eps = 1e-4

    sum_abs = sum_rel = sum_sq = sum_sgn = None
    n_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        profile, tau_c, geometry = _split_input(x, model.config.n_levels)
        pred     = model(profile, tau_c, geometry)
        residual = pred - y

        b_abs = residual.abs().sum(dim=0).cpu().numpy()
        b_rel = (residual.abs() / (y.abs() + eps)).sum(dim=0).cpu().numpy()
        b_sq  = residual.pow(2).sum(dim=0).cpu().numpy()
        b_sgn = residual.sum(dim=0).cpu().numpy()

        if sum_abs is None:
            sum_abs, sum_rel, sum_sq, sum_sgn = b_abs, b_rel, b_sq, b_sgn
        else:
            sum_abs += b_abs
            sum_rel += b_rel
            sum_sq  += b_sq
            sum_sgn += b_sgn

        n_samples += y.shape[0]

    mean_abs = sum_abs / n_samples
    mean_rel = 100.0 * sum_rel / n_samples
    rmse     = np.sqrt(sum_sq / n_samples)
    bias     = sum_sgn / n_samples

    worst_idx = np.argsort(mean_rel)[-10:][::-1]
    best_idx  = np.argsort(mean_rel)[:10]

    return {
        'mean_rel_error_pct': mean_rel,
        'mean_abs_error':     mean_abs,
        'rmse':               rmse,
        'bias':               bias,
        'worst_channels_idx': worst_idx,
        'best_channels_idx':  best_idx,
        'wavelengths':        wavelengths,
        'n_samples':          n_samples,
    }


def print_spectral_summary(residuals: dict):
    """Print a human-readable summary of per-channel spectral residuals."""
    mre  = residuals['mean_rel_error_pct']
    rmse = residuals['rmse']
    bias = residuals['bias']
    wl   = residuals.get('wavelengths')

    print(f"  Overall mean MRE : {mre.mean():.3f}%  (std = {mre.std():.3f}%)")
    print(f"  Overall mean RMSE: {rmse.mean():.5f}  (std = {rmse.std():.5f})")
    print(f"  Overall mean bias: {bias.mean():+.5f}")

    print(f"\n  10 worst channels by MRE:")
    for idx in residuals['worst_channels_idx']:
        wl_str = f"{wl[idx]:.1f} nm" if wl is not None else f"ch {idx:3d}"
        print(f"    {wl_str:>10s}  MRE = {mre[idx]:6.2f}%  "
              f"RMSE = {rmse[idx]:.5f}  bias = {bias[idx]:+.5f}")

    print(f"\n  10 best channels by MRE:")
    for idx in residuals['best_channels_idx']:
        wl_str = f"{wl[idx]:.1f} nm" if wl is not None else f"ch {idx:3d}"
        print(f"    {wl_str:>10s}  MRE = {mre[idx]:6.2f}%  "
              f"RMSE = {rmse[idx]:.5f}  bias = {bias[idx]:+.5f}")


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch,
                    val_mse, val_mre, config, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_mse':              val_mse,
        'val_mre_pct':          val_mre,
        'config':               config,
        'emulator_config': {
            'n_levels':          model.config.n_levels,
            'n_geometry_inputs': model.config.n_geometry_inputs,
            'n_wavelengths_out': model.config.n_wavelengths_out,
            'hidden_dims':       list(model.config.hidden_dims),
            'dropout':           model.config.dropout,
            'activation':        model.config.activation,
        },
    }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    config = load_config(args.config)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print(f"  Apple Silicon GPU (MPS)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    h5_path = config['data']['h5_path']
    print(f"\nLoading data from {h5_path} ...")

    profile_holdout = config['data'].get('profile_holdout', True)
    split_mode      = "profile-held-out" if profile_holdout else "random sample-level"
    print(f"  Split mode: {split_mode}")

    train_loader, val_loader, test_loader = create_emulator_dataloaders(
        h5_path         = h5_path,
        batch_size      = config['training']['batch_size'],
        num_workers     = config['data'].get('num_workers', 4),
        seed            = config['data'].get('seed', 42),
        instrument      = config['data'].get('instrument', 'hysics'),
        profile_holdout = profile_holdout,
        n_val_profiles  = config['data'].get('n_val_profiles', 7),
        n_test_profiles = config['data'].get('n_test_profiles', 8),
        train_frac      = config['data'].get('train_frac', 0.8),
        val_frac        = config['data'].get('val_frac', 0.1),
        lhc_h5_path     = config['data'].get('lhc_h5_path', None),
    )

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"  Train: {n_train:,} samples")
    print(f"  Val:   {n_val:,} samples")
    print(f"  Test:  {n_test:,} samples")

    if profile_holdout:
        n_val_prof  = config['data'].get('n_val_profiles', 7)
        n_test_prof = config['data'].get('n_test_profiles', 8)
        print(f"  Val profiles (never in train):  {n_val_prof}")
        print(f"  Test profiles (never in train): {n_test_prof}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    emulator_config = EmulatorConfig(
        n_levels          = config['model'].get('n_levels', 10),
        n_geometry_inputs = config['model'].get('n_geometry_inputs', 4),
        n_wavelengths_out = config['model'].get('n_wavelengths_out', 636),
        hidden_dims       = tuple(config['model']['hidden_dims']),
        dropout           = config['model'].get('dropout', 0.05),
        activation        = config['model'].get('activation', 'gelu'),
    )
    model    = ForwardModelEmulator(emulator_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nForwardModelEmulator: {n_params:,} trainable parameters")
    print(f"  Architecture: {emulator_config.input_dim} → "
          f"{list(emulator_config.hidden_dims)} → {emulator_config.n_wavelengths_out}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = DWMSELoss(eps=config['training'].get('dwmse_eps', 1e-4)).to(device)
    print(f"\nLoss: DWMSELoss  (eps={criterion.eps})")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config['training']['learning_rate'],
        weight_decay = config['training'].get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0     = config['training'].get('scheduler_T0', 50),
        T_mult  = config['training'].get('scheduler_T_mult', 2),
        eta_min = config['training'].get('scheduler_eta_min', 1e-6),
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.checkpoint:
        print(f"\nResuming from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs         = config['training']['n_epochs']
    patience         = config['training'].get('early_stopping_patience', 30)
    mre_target       = config['training'].get('mre_target_pct', 1.0)
    mre_log_interval = config['training'].get('mre_log_interval', 10)

    best_val_mse     = float('inf')
    best_val_mre     = float('inf')
    patience_counter = 0
    emulator_ready   = False
    history          = []

    print(f"\nTraining ForwardModelEmulator for up to {n_epochs} epochs")
    print(f"  Accuracy target:  MRE < {mre_target:.1f}%")
    print(f"  MRE check every:  {mre_log_interval} epochs")
    print(f"  Early stopping:   patience = {patience}")
    print("=" * 80)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_losses   = validate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        elapsed      = time.time() - t0
        val_loss     = val_losses['loss']   # DWMSE — used for checkpointing / early stopping
        val_mse_plain= val_losses['mse']    # plain MSE — logged for interpretability
        current_lr   = optimizer.param_groups[0]['lr']

        # ---- Compute MRE at intervals (more expensive than MSE alone) ----
        val_mre = None
        if epoch % mre_log_interval == 0 or epoch == n_epochs - 1:
            val_mre = compute_mre(model, val_loader, device)

            print(f"Epoch {epoch:4d} | "
                  f"Train DWMSE: {train_losses['loss']:.6f} | "
                  f"Val DWMSE: {val_loss:.6f} | "
                  f"Val MSE: {val_mse_plain:.6f} | "
                  f"Val MRE: {val_mre:.3f}% | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s")

            # ---- Accuracy gate ----
            if val_mre < mre_target and not emulator_ready:
                emulator_ready = True
                ready_path = output_dir / "emulator_ready.pt"
                save_checkpoint(model, optimizer, scheduler, epoch,
                                val_mse_plain, val_mre, config, ready_path)
                print()
                print(f"  ╔══════════════════════════════════════════════════╗")
                print(f"  ║  EMULATOR READY                                  ║")
                print(f"  ║  Val MRE {val_mre:.3f}% < {mre_target:.1f}% target             ║")
                print(f"  ║  Checkpoint: {ready_path.name:<34s}  ║")
                print(f"  ╚══════════════════════════════════════════════════╝")
                print()
        else:
            print(f"Epoch {epoch:4d} | "
                  f"Train DWMSE: {train_losses['loss']:.6f} | "
                  f"Val DWMSE: {val_loss:.6f} | "
                  f"Val MSE: {val_mse_plain:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s")

        history.append({
            'epoch':       epoch,
            'train_dwmse': train_losses['loss'],
            'val_dwmse':   val_loss,
            'val_mse':     val_mse_plain,
            'val_mre':     val_mre,
            'lr':          current_lr,
            'time_s':      elapsed,
        })

        # ---- Checkpointing: track best val DWMSE ----
        if val_loss < best_val_mse:
            best_val_mse     = val_loss
            best_val_mre     = val_mre if val_mre is not None else best_val_mre
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_mse_plain, best_val_mre, config,
                            output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(patience = {patience} epochs without improvement)")
                break

        if epoch % 50 == 0 and epoch > 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_mse_plain, best_val_mre, config,
                            output_dir / f"checkpoint_epoch{epoch:04d}.pt")

    # ------------------------------------------------------------------
    # Save final model and training history
    # ------------------------------------------------------------------
    save_checkpoint(model, optimizer, scheduler, epoch,
                    val_mse_plain, best_val_mre, config,
                    output_dir / "final_model.pt")

    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nTraining complete.")
    print(f"  Best val MSE : {best_val_mse:.6f}")
    print(f"  Best val MRE : {best_val_mre:.3f}%" if best_val_mre < float('inf')
          else "  Best val MRE : not computed (increase mre_log_interval)")

    if not emulator_ready:
        print(f"\n  WARNING: Emulator did NOT reach the {mre_target:.1f}% MRE target.")
        print(f"  DO NOT use this emulator in Stage 2 PINN training.")
        print(f"  Suggestions:")
        print(f"    - Increase hidden_dims, e.g. [512, 512, 512, 512, 512]")
        print(f"    - Train longer  (n_epochs: 600)")
        print(f"    - Lower learning rate  (learning_rate: 5e-4)")
        print(f"    - Add LHC augmentation  (lhc_h5_path: path/to/lhc.h5)")

    # ------------------------------------------------------------------
    # Test evaluation — load best checkpoint for final diagnostics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print(f"Split mode: {split_mode}")
    print("=" * 80)

    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Load wavelength grid for display
    wavelengths = None
    with h5py.File(h5_path, 'r') as f:
        if 'wavelengths' in f:
            wavelengths = f['wavelengths'][:].astype(np.float32)

    test_mse = validate(model, test_loader, criterion, device)['mse']
    test_mre = compute_mre(model, test_loader, device)
    print(f"\n  Test MSE: {test_mse:.6f}")
    print(f"  Test MRE: {test_mre:.3f}%")

    print("\n  Per-channel spectral residuals:")
    test_residuals = compute_spectral_residuals(
        model, test_loader, device, wavelengths
    )
    print_spectral_summary(test_residuals)

    # Check accuracy gate on test set
    if test_mre < mre_target:
        print(f"\n  Test MRE {test_mre:.3f}% < {mre_target:.1f}% — emulator meets accuracy target on held-out profiles.")
    else:
        print(f"\n  WARNING: Test MRE {test_mre:.3f}% >= {mre_target:.1f}% target.")
        print(f"  The emulator generalises poorly to unseen profiles.")
        print(f"  Consider LHC augmentation to improve edge-of-space coverage.")

    # Save full test diagnostics
    diag = {
        'split_mode':            split_mode,
        'n_test_samples':        int(n_test),
        'test_mse':              float(test_mse),
        'test_mre_pct':          float(test_mre),
        'mre_target_pct':        float(mre_target),
        'emulator_ready':        bool(test_mre < mre_target),
        'per_channel_mre_pct':   test_residuals['mean_rel_error_pct'].tolist(),
        'per_channel_rmse':      test_residuals['rmse'].tolist(),
        'per_channel_bias':      test_residuals['bias'].tolist(),
        'worst_10_channels_idx': test_residuals['worst_channels_idx'].tolist(),
        'best_10_channels_idx':  test_residuals['best_channels_idx'].tolist(),
        'wavelengths_nm':        wavelengths.tolist() if wavelengths is not None else None,
    }
    diag_path = output_dir / "test_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)

    print(f"\nDiagnostics saved to: {diag_path}")
    print(f"Checkpoints saved to: {output_dir}/")

    if emulator_ready:
        print(f"\nTo use in Stage 2, set in your retrieval config:")
        print(f"  emulator_checkpoint: \"{output_dir}/emulator_ready.pt\"")
        print(f"  lambda_emulator_data: 1.0")


if __name__ == "__main__":
    main()
