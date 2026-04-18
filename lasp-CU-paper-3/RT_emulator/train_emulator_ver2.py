"""
train_emulator_ver2.py — Per-Band Subnetwork Emulator with MRE Loss.

Key changes from train_emulator.py (v1):
  1. Per-band subnetworks — the 636-channel spectrum is partitioned into
     spectral bands, each with its own small residual MLP.  This follows
     Bue et al. (2019), who showed that per-channel/per-band subnetworks
     achieve <0.1% MAE on radiative transfer emulation.
  2. MRE loss — directly minimizes mean relative error in linear
     reflectance space, matching the evaluation metric.
  3. No PCA — unnecessary since each subnetwork only predicts ~30–120
     channels.

The full emulator is a BandEnsembleEmulator that wraps N small
ForwardModelEmulator instances.  At inference time it concatenates
their outputs to produce a full 636-channel spectrum.

Usage
-----
    python train_emulator_ver2.py --config emulator_ver2.yaml

Author: Andrew J. Buggee, LASP / CU Boulder
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import ForwardModelEmulator, EmulatorConfig, _ResidualBlock
from data import create_emulator_dataloaders


# =============================================================================
# Argument parsing + config
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-band ForwardModelEmulator (v2)")
    parser.add_argument("--config", type=str, default="emulator_ver2.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints/emulator_v2",
                        help="Directory to save checkpoints and diagnostics")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Band assignment utilities
# =============================================================================

def assign_channels_to_bands(wavelengths, band_defs):
    """
    Assign each wavelength channel to a spectral band.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (n_wavelengths,)
        Band-center wavelengths in nm.
    band_defs : list of dict
        Each dict has keys: name, wl_min, wl_max, hidden_dims, dropout.

    Returns
    -------
    band_info : list of dict
        Each dict has keys: name, channel_indices (np.ndarray of int),
        n_channels, hidden_dims, dropout.
    """
    assigned = np.zeros(len(wavelengths), dtype=bool)
    band_info = []

    for bdef in band_defs:
        mask = ((wavelengths >= bdef['wl_min']) &
                (wavelengths < bdef['wl_max']) &
                (~assigned))
        indices = np.where(mask)[0]
        if len(indices) == 0:
            print(f"  WARNING: band '{bdef['name']}' has no channels "
                  f"in [{bdef['wl_min']}, {bdef['wl_max']}) nm — skipping")
            continue
        assigned[indices] = True
        band_info.append({
            'name':            bdef['name'],
            'channel_indices': indices,
            'n_channels':      len(indices),
            'hidden_dims':     bdef['hidden_dims'],
            'dropout':         bdef['dropout'],
            'wl_min':          wavelengths[indices[0]],
            'wl_max':          wavelengths[indices[-1]],
        })

    # Catch any unassigned channels
    unassigned = np.where(~assigned)[0]
    if len(unassigned) > 0:
        print(f"  WARNING: {len(unassigned)} channels not assigned to any band "
              f"(λ = {wavelengths[unassigned[0]]:.1f}–{wavelengths[unassigned[-1]]:.1f} nm)")
        print(f"  Adding them to a catch-all 'other' band.")
        default_hidden = band_defs[0].get('hidden_dims', [256, 256])
        default_drop = band_defs[0].get('dropout', 0.05)
        band_info.append({
            'name':            'other',
            'channel_indices': unassigned,
            'n_channels':      len(unassigned),
            'hidden_dims':     default_hidden,
            'dropout':         default_drop,
            'wl_min':          wavelengths[unassigned[0]],
            'wl_max':          wavelengths[unassigned[-1]],
        })

    return band_info


# =============================================================================
# Per-band subnetwork model
# =============================================================================

class BandSubnetwork(nn.Module):
    """
    Small residual MLP that predicts a subset of spectral channels.

    Same architecture as ForwardModelEmulator but smaller (fewer hidden
    units, fewer layers) since it only covers one spectral band.
    """

    def __init__(self, input_dim, n_channels_out, hidden_dims, dropout, activation):
        super().__init__()
        act_fn = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU()}[activation]

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        blocks = []
        h_in = hidden_dims[0]
        for h_out in hidden_dims:
            blocks.append(_ResidualBlock(h_in, h_out, dropout, act_fn))
            h_in = h_out
        self.blocks = nn.ModuleList(blocks)

        self.output_norm = nn.LayerNorm(h_in)
        self.output_head = nn.Linear(h_in, n_channels_out)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, input_dim) — concatenated normalized inputs

        Returns
        -------
        (batch, n_channels_out) — predicted log10-reflectances for this band
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(self.output_norm(x))


class BandEnsembleEmulator(nn.Module):
    """
    Ensemble of per-band subnetworks that together predict a full spectrum.

    Each BandSubnetwork predicts a contiguous subset of channels.
    At inference time, outputs are stitched into the full wavelength grid.
    """

    def __init__(self, input_dim, band_info, activation, n_wavelengths):
        super().__init__()
        self.input_dim = input_dim
        self.n_wavelengths = n_wavelengths
        self.band_names = [b['name'] for b in band_info]

        # Store channel indices as buffers (move to device with model)
        self.band_indices = []
        subnets = []
        for b in band_info:
            idx = torch.tensor(b['channel_indices'], dtype=torch.long)
            self.register_buffer(f"idx_{b['name']}", idx)
            self.band_indices.append(idx)
            subnets.append(BandSubnetwork(
                input_dim=input_dim,
                n_channels_out=b['n_channels'],
                hidden_dims=b['hidden_dims'],
                dropout=b['dropout'],
                activation=activation,
            ))
        self.subnets = nn.ModuleList(subnets)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        (batch, n_wavelengths) — full-spectrum log10-reflectance prediction
        """
        out = torch.zeros(x.shape[0], self.n_wavelengths,
                          device=x.device, dtype=x.dtype)
        for i, subnet in enumerate(self.subnets):
            idx = getattr(self, f"idx_{self.band_names[i]}")
            out[:, idx] = subnet(x)
        return out

    def forward_band(self, x, band_idx):
        """Predict only one band (used during per-band training)."""
        return self.subnets[band_idx](x)


# =============================================================================
# MRE Loss
# =============================================================================

class MRELoss(nn.Module):
    """
    Mean Relative Error loss in linear reflectance space.

        L = mean( |R_pred - R_true| / (|R_true| + eps) )

    Training targets are in log10-reflectance space, so this loss
    exponentiates both pred and target before computing the relative error.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, pred_log, target_log):
        """
        Parameters
        ----------
        pred_log   : (batch, n_channels) — predicted log10(R)
        target_log : (batch, n_channels) — true log10(R)
        """
        R_pred = torch.pow(10.0, pred_log)
        R_true = torch.pow(10.0, target_log)
        rel_err = (R_pred - R_true).abs() / (R_true.abs() + self.eps)
        return rel_err.mean()


# =============================================================================
# Training and evaluation helpers
# =============================================================================

def _build_input(x, n_levels, n_atm):
    """Return the full concatenated input (same as x, already concatenated by dataset)."""
    return x


@torch.no_grad()
def compute_mre(model, loader, device, log_reflectance=True):
    """Compute overall MRE (%) in linear reflectance space."""
    model.eval()
    total_rel_err = 0.0
    n_elements = 0
    eps = 1e-4

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        if log_reflectance:
            R_pred = torch.pow(10.0, pred)
            R_true = torch.pow(10.0, y)
        else:
            R_pred, R_true = pred, y

        rel_err = (R_pred - R_true).abs() / (R_true.abs() + eps)
        total_rel_err += rel_err.sum().item()
        n_elements += R_true.numel()

    return 100.0 * total_rel_err / n_elements


@torch.no_grad()
def compute_per_band_mre(model, loader, device, band_info, log_reflectance=True):
    """Compute MRE per spectral band."""
    model.eval()
    eps = 1e-4
    n_bands = len(band_info)
    band_rel_err = [0.0] * n_bands
    band_elements = [0] * n_bands

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        if log_reflectance:
            R_pred = torch.pow(10.0, pred)
            R_true = torch.pow(10.0, y)
        else:
            R_pred, R_true = pred, y

        for i, b in enumerate(band_info):
            idx = b['channel_indices']
            rel = (R_pred[:, idx] - R_true[:, idx]).abs() / (R_true[:, idx].abs() + eps)
            band_rel_err[i] += rel.sum().item()
            band_elements[i] += R_true[:, idx].numel()

    return {b['name']: 100.0 * band_rel_err[i] / band_elements[i]
            for i, b in enumerate(band_info)}


@torch.no_grad()
def compute_spectral_residuals(model, loader, device, wavelengths=None,
                                log_reflectance=True):
    """Compute per-channel residual statistics over the full data loader."""
    model.eval()
    eps = 1e-4
    sum_abs = sum_rel = sum_sq = sum_sgn = None
    n_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        if log_reflectance:
            R_pred = torch.pow(10.0, pred)
            R_true = torch.pow(10.0, y)
        else:
            R_pred, R_true = pred, y

        residual = R_pred - R_true
        b_abs = residual.abs().sum(dim=0).cpu().numpy()
        b_rel = (residual.abs() / (R_true.abs() + eps)).sum(dim=0).cpu().numpy()
        b_sq = residual.pow(2).sum(dim=0).cpu().numpy()
        b_sgn = residual.sum(dim=0).cpu().numpy()

        if sum_abs is None:
            sum_abs, sum_rel, sum_sq, sum_sgn = b_abs, b_rel, b_sq, b_sgn
        else:
            sum_abs += b_abs
            sum_rel += b_rel
            sum_sq += b_sq
            sum_sgn += b_sgn

        n_samples += y.shape[0]

    mean_abs = sum_abs / n_samples
    mean_rel = 100.0 * sum_rel / n_samples
    rmse = np.sqrt(sum_sq / n_samples)
    bias = sum_sgn / n_samples

    worst_idx = np.argsort(mean_rel)[-10:][::-1]
    best_idx = np.argsort(mean_rel)[:10]

    return {
        'mean_rel_error_pct': mean_rel,
        'mean_abs_error': mean_abs,
        'rmse': rmse,
        'bias': bias,
        'worst_channels_idx': worst_idx,
        'best_channels_idx': best_idx,
        'wavelengths': wavelengths,
        'n_samples': n_samples,
    }


def print_spectral_summary(residuals):
    """Print a human-readable summary of per-channel spectral residuals."""
    mre = residuals['mean_rel_error_pct']
    rmse = residuals['rmse']
    bias = residuals['bias']
    wl = residuals.get('wavelengths')

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
                    val_mse, val_mre, config, band_info, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_mse': val_mse,
        'val_mre_pct': val_mre,
        'config': config,
        'band_info': [{
            'name': b['name'],
            'channel_indices': b['channel_indices'].tolist(),
            'n_channels': b['n_channels'],
            'hidden_dims': b['hidden_dims'],
            'dropout': b['dropout'],
        } for b in band_info],
    }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  Apple Silicon GPU (MPS)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    h5_path = config['data']['h5_path']
    print(f"\nLoading data from {h5_path} ...")

    profile_holdout = config['data'].get('profile_holdout', True)
    split_mode = "profile-held-out" if profile_holdout else "random sample-level"
    print(f"  Split mode: {split_mode}")

    log_reflectance = config['data'].get('log_reflectance', True)
    log_eps = float(config['data'].get('log_eps', 1e-6))
    use_era5_profile = config['data'].get('use_era5_profile', False)

    # No PCA for per-band approach
    n_pca_components = 0

    train_loader, val_loader, test_loader = create_emulator_dataloaders(
        h5_path=h5_path,
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 0),
        seed=config['data'].get('seed', 42),
        instrument=config['data'].get('instrument', 'hysics'),
        profile_holdout=profile_holdout,
        n_val_profiles=config['data'].get('n_val_profiles', 10),
        n_test_profiles=config['data'].get('n_test_profiles', 10),
        train_frac=config['data'].get('train_frac', 0.8),
        val_frac=config['data'].get('val_frac', 0.1),
        lhc_h5_path=config['data'].get('lhc_h5_path', None),
        log_reflectance=log_reflectance,
        log_eps=log_eps,
        use_era5_profile=use_era5_profile,
        n_pca_components=n_pca_components,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"  Train: {n_train:,} samples")
    print(f"  Val:   {n_val:,} samples")
    print(f"  Test:  {n_test:,} samples")

    if profile_holdout:
        n_val_prof = config['data'].get('n_val_profiles', 10)
        n_test_prof = config['data'].get('n_test_profiles', 10)
        print(f"  Val profiles (never in train):  {n_val_prof}")
        print(f"  Test profiles (never in train): {n_test_prof}")

    # ------------------------------------------------------------------
    # Load wavelength grid and assign channels to bands
    # ------------------------------------------------------------------
    with h5py.File(h5_path, 'r') as f:
        wavelengths = f['wavelengths'][:].astype(np.float32)

    band_defs = config['bands']
    band_info = assign_channels_to_bands(wavelengths, band_defs)

    print(f"\n  Spectral band decomposition ({len(band_info)} bands):")
    total_channels = 0
    for b in band_info:
        print(f"    {b['name']:>12s}: {b['n_channels']:3d} channels "
              f"({b['wl_min']:.1f}–{b['wl_max']:.1f} nm)  "
              f"hidden={b['hidden_dims']}")
        total_channels += b['n_channels']
    print(f"    {'TOTAL':>12s}: {total_channels:3d} channels")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    n_levels = config['model']['n_levels']
    n_geom = config['model']['n_geometry_inputs']
    n_atm = config['model'].get('n_atm_inputs', 0)
    n_wavelengths = config['model']['n_wavelengths_out']
    activation = config['model'].get('activation', 'gelu')
    input_dim = n_levels + 1 + n_geom + n_atm

    model = BandEnsembleEmulator(
        input_dim=input_dim,
        band_info=band_info,
        activation=activation,
        n_wavelengths=n_wavelengths,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBandEnsembleEmulator: {n_params:,} trainable parameters")
    print(f"  Input dim: {input_dim}")
    for i, b in enumerate(band_info):
        sub_params = sum(p.numel() for p in model.subnets[i].parameters()
                         if p.requires_grad)
        print(f"  Band '{b['name']}': {sub_params:,} params → {b['n_channels']} channels")

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    loss_type = config['training'].get('loss', 'mre')
    if loss_type == 'mre':
        mre_eps = float(config['training'].get('mre_eps', 1e-4))
        criterion = MRELoss(eps=mre_eps)
        print(f"\nLoss: MRELoss (eps={mre_eps})")
    else:
        criterion = nn.MSELoss()
        print(f"\nLoss: MSELoss (log-reflectance mode)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4),
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config['training'].get('scheduler_factor', 0.5),
        patience=config['training'].get('scheduler_patience', 30),
        min_lr=float(config['training'].get('scheduler_eta_min', 1e-6)),
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_mre = float('inf')

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_mse', float('inf'))
        best_val_mre = ckpt.get('val_mre_pct', float('inf'))
        print(f"\nResumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs = config['training']['n_epochs']
    patience = config['training'].get('early_stopping_patience', 100)
    mre_target = config['training'].get('mre_target_pct', 1.0)
    mre_log_interval = config['training'].get('mre_log_interval', 10)

    print(f"\nTraining BandEnsembleEmulator for up to {n_epochs} epochs")
    print(f"  Accuracy target:  MRE < {mre_target}%")
    print(f"  MRE check every:  {mre_log_interval} epochs")
    print(f"  Early stopping:   patience = {patience}")
    print("=" * 80)

    patience_counter = 0
    emulator_ready = False
    history = []
    mse_fn = nn.MSELoss()

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1
        train_loss = total_train_loss / n_batches

        # ---- Validate ----
        model.eval()
        total_val_loss = 0.0
        total_val_mse = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                total_val_loss += criterion(pred, y).item()
                total_val_mse += mse_fn(pred, y).item()
                n_val_batches += 1
        val_loss = total_val_loss / n_val_batches
        val_mse = total_val_mse / n_val_batches

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_loss)

        # ---- MRE evaluation at intervals ----
        val_mre = None
        loss_label = loss_type.upper()
        if epoch % mre_log_interval == 0 or epoch == n_epochs - 1:
            val_mre = compute_mre(model, val_loader, device,
                                  log_reflectance=log_reflectance)

            # Per-band MRE
            band_mre = compute_per_band_mre(model, val_loader, device,
                                            band_info, log_reflectance)

            print(f"Epoch {epoch:4d} | "
                  f"Train {loss_label}: {train_loss:.6f} | "
                  f"Val {loss_label}: {val_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"Val MRE: {val_mre:.3f}% | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s")
            # Print per-band MRE
            band_strs = [f"{name}: {mre:.2f}%" for name, mre in band_mre.items()]
            print(f"         Band MRE: {' | '.join(band_strs)}")

            # ---- Accuracy gate ----
            if val_mre < mre_target and not emulator_ready:
                emulator_ready = True
                ready_path = output_dir / "emulator_ready.pt"
                save_checkpoint(model, optimizer, scheduler, epoch,
                                val_mse, val_mre, config, band_info, ready_path)
                print()
                print(f"  ╔══════════════════════════════════════════════════╗")
                print(f"  ║  EMULATOR READY                                  ║")
                print(f"  ║  Val MRE {val_mre:.3f}% < {mre_target:.1f}% target             ║")
                print(f"  ║  Checkpoint: {ready_path.name:<34s}  ║")
                print(f"  ╚══════════════════════════════════════════════════╝")
                print()
        else:
            print(f"Epoch {epoch:4d} | "
                  f"Train {loss_label}: {train_loss:.6f} | "
                  f"Val {loss_label}: {val_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'val_mre': val_mre,
            'lr': current_lr,
            'time_s': elapsed,
        })

        # ---- Checkpointing ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mre = val_mre if val_mre is not None else best_val_mre
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_mse, best_val_mre, config, band_info,
                            output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(patience = {patience} epochs without improvement)")
                break

        if epoch % 50 == 0 and epoch > 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_mse, best_val_mre, config, band_info,
                            output_dir / f"checkpoint_epoch{epoch:04d}.pt")

    # ------------------------------------------------------------------
    # Save final model and history
    # ------------------------------------------------------------------
    save_checkpoint(model, optimizer, scheduler, epoch,
                    val_mse, best_val_mre, config, band_info,
                    output_dir / "final_model.pt")

    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nTraining complete.")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best val MRE : {best_val_mre:.3f}%" if best_val_mre < float('inf')
          else "  Best val MRE : not computed")

    if not emulator_ready:
        print(f"\n  WARNING: Emulator did NOT reach the {mre_target:.1f}% MRE target.")
        print(f"  DO NOT use this emulator in Stage 2 PINN training.")

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print(f"Split mode: {split_mode}")
    print("=" * 80)

    best_ckpt = torch.load(output_dir / "best_model.pt",
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_mse_val = 0.0
    n_test_batches = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_mse_val += mse_fn(pred, y).item()
            n_test_batches += 1
    test_mse = test_mse_val / n_test_batches

    test_mre = compute_mre(model, test_loader, device,
                           log_reflectance=log_reflectance)
    print(f"\n  Test MSE: {test_mse:.6f}")
    print(f"  Test MRE: {test_mre:.3f}%")

    # Per-band MRE on test set
    test_band_mre = compute_per_band_mre(model, test_loader, device,
                                          band_info, log_reflectance)
    print(f"\n  Per-band test MRE:")
    for name, mre in test_band_mre.items():
        print(f"    {name:>12s}: {mre:.3f}%")

    # Per-channel residuals
    print(f"\n  Per-channel spectral residuals:")
    test_residuals = compute_spectral_residuals(
        model, test_loader, device, wavelengths,
        log_reflectance=log_reflectance,
    )
    print_spectral_summary(test_residuals)

    if test_mre < mre_target:
        print(f"\n  Test MRE {test_mre:.3f}% < {mre_target:.1f}% — emulator meets target.")
    else:
        print(f"\n  WARNING: Test MRE {test_mre:.3f}% >= {mre_target:.1f}% target.")

    # Save diagnostics
    diag = {
        'split_mode': split_mode,
        'n_test_samples': n_test,
        'test_mse': float(test_mse),
        'test_mre_pct': float(test_mre),
        'mre_target_pct': float(mre_target),
        'emulator_ready': bool(test_mre < mre_target),
        'per_band_test_mre': {k: float(v) for k, v in test_band_mre.items()},
        'per_channel_mre_pct': test_residuals['mean_rel_error_pct'].tolist(),
        'per_channel_rmse': test_residuals['rmse'].tolist(),
        'per_channel_bias': test_residuals['bias'].tolist(),
        'worst_10_channels_idx': test_residuals['worst_channels_idx'].tolist(),
        'best_10_channels_idx': test_residuals['best_channels_idx'].tolist(),
        'wavelengths_nm': wavelengths.tolist(),
        'band_definitions': [{
            'name': b['name'],
            'n_channels': b['n_channels'],
            'wl_min': float(b['wl_min']),
            'wl_max': float(b['wl_max']),
        } for b in band_info],
    }
    diag_path = output_dir / "test_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)

    print(f"\nDiagnostics saved to: {diag_path}")
    print(f"Checkpoints saved to: {output_dir}/")

    if emulator_ready:
        print(f"\nTo use in Stage 2, set in your retrieval config:")
        print(f"  emulator_checkpoint: \"{output_dir}/emulator_ready.pt\"")


if __name__ == "__main__":
    main()
