"""
train_standalone_profile_only.py — Train a single ProfileOnlyNetwork using
the saved hyperparameters of one config from sweep_results_profile_only/.

This is the profile-only counterpart to train_standalone.py.  Differences:

  • Model:    ProfileOnlyNetwork  (no τ head)
  • Loss:     ProfileOnlyLoss     (Gaussian NLL on the profile + 3 physics regs)
  • Sweep:    hyper_parameter_sweep/sweep_results_profile_only/run_NNN/config.json
  • Splits:   single profile-aware train/val/test split (no K-fold) — for
              quick inspection of one chosen config; if you want the K=5 mean
              ± std headline, look at the original sweep summary.json.

Inputs to the network
---------------------
The 640-dim input vector handed to the model is already
    [ log10(reflectance) (636 channels) ]   ⊕   [ sza, vza, saz, vaz (4 angles) ]

constructed inside data.LibRadtranDataset._normalize_input — the four
geometry angles are present in the HDF5 (/sza, /vza, /saz, /vaz, all
populated, no NaNs) and concatenated onto the spectrum at every sample.
This script does NOT add new inputs; it just exposes the existing 640-dim
pipeline through a one-config standalone trainer.

Outputs
-------
    <output-dir>/
        best_model.pt
        config.json                       (effective config + h5 override)
        history.json                      (per-epoch train+val loss)
        results.json                      (final test metrics)
        loss_curves.png                   (500 DPI)
        profiles_true_vs_pred.png         (500 DPI; no τ overlay)

Usage
-----
    python train_standalone_profile_only.py \
        --run-id 110 \
        --h5-path /path/to/combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5 \
        --output-dir ./standalone_results_profile_only/run110_50levels

    # Smoke test on CPU:
    python train_standalone_profile_only.py --run-id 110 \
        --h5-path <path> --device cpu --n-epochs 5

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
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
from models_profile_only import ProfileOnlyNetwork, ProfileOnlyLoss
from models import RetrievalConfig
from data import create_dataloaders, resolve_h5_path


GREEN    = '#10B981'   # NN retrieval
RAW_BLUE = '#1F5FC2'   # raw in-situ


# Reference 7-level shapes from generate_sweep_2.py + the 50-level shapes
# from generate_sweep_profile_only.py.  When --level-weights is passed,
# regenerate weights to match the HDF5 n_levels.
LEVEL_WEIGHT_REFS_7LEVEL = {
    'deep': np.array([1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    'top':  np.array([6.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
}


def _build_50level_scheme(scheme: str, n_levels: int) -> list:
    """Replicate generate_sweep_profile_only.build_level_weights for any n_levels."""
    levels = np.arange(n_levels, dtype=float)
    end = n_levels - 1
    decay = max(1.0, 8.0 * (n_levels / 50.0))   # rescale decay length to grid
    if scheme == 'uniform':
        w = np.ones(n_levels)
    elif scheme == 'top_50style':
        w = 1.0 + 4.0 * np.exp(-levels / decay)
    elif scheme == 'bottom_50style':
        w = 1.0 + 4.0 * np.exp(-(end - levels) / decay)
    elif scheme == 'u_shape':
        w = 1.0 + 4.0 * (np.exp(-levels / decay)
                         + np.exp(-(end - levels) / decay))
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")
    return [float(x) for x in w]


def build_level_weights(scheme: str, n_levels: int) -> list:
    """
    Generate a length-n_levels level_weights vector for the requested scheme.

    Recognised schemes:
      'uniform'        — all 1.0
      'deep' / 'top'   — interpolated from the 7-level sweep_2 schemes
      'top_50style'    — top-emphasis exponential, matches the 50-level sweep
      'bottom_50style' — bottom-emphasis exponential, matches the 50-level sweep
      'u_shape'        — both-ends emphasis, matches the 50-level sweep

    The two naming conventions co-exist because the 7-level sweeps and the
    50-level profile-only sweep used different shape families.  Pick whichever
    matches the run config you're reusing.
    """
    if scheme == 'uniform':
        return [1.0] * n_levels
    if scheme in LEVEL_WEIGHT_REFS_7LEVEL:
        ref = LEVEL_WEIGHT_REFS_7LEVEL[scheme]
        xs  = np.linspace(0.0, len(ref) - 1, n_levels)
        return np.interp(xs, np.arange(len(ref)), ref).tolist()
    if scheme in ('top_50style', 'bottom_50style', 'u_shape'):
        return _build_50level_scheme(scheme, n_levels)
    raise ValueError(
        f"level_weights scheme must be one of "
        f"['uniform','deep','top','top_50style','bottom_50style','u_shape'], "
        f"got {scheme!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-id', type=int, required=True,
                   help='Sweep run ID to borrow hyperparameters from, e.g. 110 -> run_110')
    p.add_argument('--h5-path', type=str, required=True,
                   help='HDF5 training file (overrides the path saved in the run config). '
                        'May be a full path, or just a filename when --training-data-dir is given.')
    p.add_argument('--training-data-dir', type=str, default=None,
                   help='Directory hosting the HDF5 file on this machine. If given, '
                        'the directory portion of --h5-path is replaced with this; only the '
                        'filename is preserved.')
    p.add_argument('--sweep-dir', type=str,
                   default='hyper_parameter_sweep/sweep_results_profile_only',
                   help='Directory holding the run_NNN/ subfolders (relative to repo root)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to save outputs.  Default: ./standalone_results_profile_only/run<ID>_<h5stem>')
    p.add_argument('--device', type=str, default=None,
                   help='"cuda", "mps", "cpu". Default: cuda > mps > cpu.')
    p.add_argument('--n-epochs', type=int, default=None,
                   help='Override n_epochs from the run config (default: use what was saved)')
    p.add_argument('--level-weights', type=str, default=None,
                   choices=['uniform', 'deep', 'top',
                            'top_50style', 'bottom_50style', 'u_shape'],
                   help='Override the saved level_weights with a rebuilt one matching '
                        'the HDF5 n_levels.  Default: keep saved weights and enforce length match.')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for the train/val/test split and weight init')
    p.add_argument('--n-val-profiles',  type=int, default=14)
    p.add_argument('--n-test-profiles', type=int, default=14)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation / inference loops
# (parity with sweep_train_profile_only.py — ignores the τ tensor returned
#  by the existing dataloader since this network has no τ head.)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    augment_noise_std=0.0,
                    warmup_steps=0, target_lr=None, global_step_start=0):
    model.train()
    loss_sum, n = 0.0, 0
    step = global_step_start
    for batch in loader:
        # Dataset still yields (x, profile, tau) — drop tau for profile-only.
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
    """Run inference on the whole test loader.  Returns arrays in physical units."""
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
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(history, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(history['train'], label='Train', linewidth=1.5)
    ax.plot(history['val'],   label='Validation', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total NLL Loss', fontsize=12)
    ax.set_title('Training Progress (profile-only)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def plot_profiles_true_vs_pred(results, test_loader, h5_path, n_levels,
                               out_path, n_show=6, seed=0):
    """
    2x3 grid of true-vs-predicted droplet profiles, plotted on a τ axis.

    No τ-c line/value in the title — this is a profile-only model.
    """
    pred     = results['pred']
    pred_std = results['pred_std']
    true_um  = results['true']

    # Recover global HDF5 indices from the Subset wrapper to fetch raw profiles
    subset = test_loader.dataset
    if not hasattr(subset, 'indices'):
        raise RuntimeError("Expected test_loader.dataset to be a torch Subset.")
    test_idx_global = np.asarray(subset.indices, dtype=int)
    sort_order   = np.argsort(test_idx_global)
    unsort_order = np.argsort(sort_order)
    sorted_idx   = test_idx_global[sort_order]

    with h5py.File(h5_path, 'r') as f:
        raw_sorted     = f['profiles_raw'][sorted_idx]
        z_sorted       = f['profiles_raw_z'][sorted_idx]
        nlev_sorted    = f['profile_n_levels'][sorted_idx]
        has_raw_tau    = 'profiles_raw_tau' in f
        tau_raw_sorted = f['profiles_raw_tau'][sorted_idx] if has_raw_tau else None
        tau_c_sorted   = f['tau_c'][sorted_idx] if 'tau_c' in f else None
    raw_all     = raw_sorted[unsort_order]
    z_all       = z_sorted[unsort_order]
    nlev_all    = nlev_sorted[unsort_order]
    tau_rawall  = tau_raw_sorted[unsort_order] if has_raw_tau else None
    tau_c_all   = tau_c_sorted[unsort_order] if tau_c_sorted is not None else None

    # Pick n_show *unique* clouds (each cloud appears ~128× via varied geometry)
    rng = np.random.default_rng(seed)
    seen, unique_idx = set(), []
    for i in range(pred.shape[0]):
        key = tuple(np.round(true_um[i], 6))
        if key not in seen:
            seen.add(key)
            unique_idx.append(i)
        if len(unique_idx) >= n_show * 3:
            break
    pick = (rng.choice(unique_idx, size=n_show, replace=False)
            if len(unique_idx) >= n_show else np.array(unique_idx))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for k, (ax, i) in enumerate(zip(axes.flat, pick)):
        n_lev_i  = int(nlev_all[i])
        re_raw   = raw_all[i, :n_lev_i]
        z_raw_km = z_all[i, :n_lev_i]
        tau_raw_i = tau_rawall[i, :n_lev_i] if has_raw_tau else None

        z_top_km, z_base_km = float(z_raw_km[0]), float(z_raw_km[-1])
        z_lvl = np.linspace(z_top_km, z_base_km, n_levels)

        if tau_raw_i is not None:
            tau_lvl = np.interp(z_lvl, z_raw_km[::-1], tau_raw_i[::-1])
            tau_for_raw  = tau_raw_i
            tau_axis_lbl = r'Optical depth $\tau$'
        else:
            tau_c_i = float(tau_c_all[i]) if tau_c_all is not None else 1.0
            tau_lvl = np.linspace(0.0, tau_c_i, n_levels)
            tau_for_raw = ((z_top_km - z_raw_km)
                           / max(z_top_km - z_base_km, 1e-9)) * tau_c_i
            tau_axis_lbl = r'Optical depth $\tau$ (linear approx.)'

        ax.plot(re_raw, tau_for_raw, '-', color=RAW_BLUE,
                linewidth=1.0, alpha=0.60, label='Original in-situ profile')
        ax.plot(true_um[i], tau_lvl, 'ko-', markersize=4, linewidth=1.5,
                label=f'True ({n_levels} levels)')
        ax.errorbar(pred[i], tau_lvl, xerr=pred_std[i],
                    fmt='s--', color=GREEN, markersize=4, linewidth=1.5,
                    elinewidth=1.0, capsize=3, label='NN retrieval ±1σ')

        ax.invert_yaxis()
        ax.set_xlabel(r'$r_e$ (μm)', fontsize=10)
        if k % 3 == 0:
            ax.set_ylabel(tau_axis_lbl, fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

        ax2 = ax.twinx()
        ax2.set_ylim(z_base_km, z_top_km)
        if k % 3 == 2:
            ax2.set_ylabel('Altitude (km)', fontsize=10)

        # Per-panel mean RMSE (μm) — most useful number for a profile-only model
        rmse_i = float(np.sqrt(np.mean((pred[i] - true_um[i]) ** 2)))
        ax.set_title(rf'mean RMSE = {rmse_i:.2f} μm', fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Predicted vs True Droplet Profiles (Test Set, profile-only)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    # ── Locate and load the run's saved config ────────────────────────────────
    sweep_dir = (repo_root / args.sweep_dir).resolve()
    run_dir   = sweep_dir / f'run_{args.run_id:03d}'
    cfg_path  = run_dir / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'No config found at {cfg_path}')

    with open(cfg_path) as f:
        cfg = json.load(f)
    hp = cfg['hyperparams']

    # ── Override h5_path with the user's file ─────────────────────────────────
    h5_path = resolve_h5_path(args.h5_path, args.training_data_dir).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(
            f'HDF5 file not found: {h5_path}\n'
            f'  --h5-path           = {args.h5_path}\n'
            f'  --training-data-dir = {args.training_data_dir}'
        )
    cfg.setdefault('data', {})
    cfg['data']['h5_path'] = str(h5_path)

    # ── Verify the HDF5 has the four geometry datasets the dataloader needs ──
    with h5py.File(h5_path, 'r') as f:
        missing_geom = [k for k in ('sza', 'vza', 'saz', 'vaz') if k not in f]
    if missing_geom:
        raise RuntimeError(
            f'HDF5 {h5_path.name} is missing expected geometry datasets: '
            f'{missing_geom}.  The 640-dim input vector requires all four '
            f'(sza, vza, saz, vaz) to be present.'
        )

    # ── Optional overrides from the CLI ───────────────────────────────────────
    if args.n_epochs is not None:
        hp['n_epochs'] = args.n_epochs

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (repo_root / 'standalone_results_profile_only'
                      / f'run{args.run_id:03d}_{h5_path.stem}').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'cpu':
        print("  [warn] running on CPU — expect several minutes per epoch.")
        print("         Pass --device mps (Apple Silicon) or --device cuda (NVIDIA)")
        print("         if either is available for a ~100x speedup.")

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  STANDALONE TRAIN — profile-only, hyperparameters from run_{args.run_id:03d}")
    print(f"{'=' * 70}")
    print(f"  sweep run dir : {run_dir}")
    print(f"  HDF5 file     : {h5_path}")
    print(f"  output dir    : {output_dir}")
    print(f"  device        : {device}")
    print(f"\n  Network input layout:  636 reflectance ⊕ 4 geometry (sza, vza, saz, vaz) = 640")
    print(f"\n  Hyperparameters being reused:")
    for k, v in hp.items():
        if k == 'level_weights' and isinstance(v, list) and len(v) > 8:
            print(f"    {k:22s} = [{v[0]:.3f}, {v[1]:.3f}, ..., "
                  f"{v[-2]:.3f}, {v[-1]:.3f}]  (len={len(v)})")
        else:
            print(f"    {k:22s} = {v}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataloaders (profile-aware single split) ──────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=str(h5_path),
        instrument=cfg['data'].get('instrument', 'hysics'),
        batch_size=hp.get('batch_size', 256),
        num_workers=cfg['data'].get('num_workers', 4),
        seed=args.seed,
        profile_holdout=True,
        n_val_profiles=args.n_val_profiles,
        n_test_profiles=args.n_test_profiles,
    )
    print(f"Data: {len(train_loader.dataset):,} train, "
          f"{len(val_loader.dataset):,} val, {len(test_loader.dataset):,} test "
          f"(held-out profiles: {args.n_val_profiles} val, {args.n_test_profiles} test)")

    # n_levels read from the underlying dataset (Subset doesn't forward attributes)
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels
    print(f"Profile grid: {n_levels} levels (read from HDF5)")

    if args.level_weights is not None:
        new_weights = build_level_weights(args.level_weights, n_levels)
        print(f"Overriding level_weights "
              f"(was '{hp.get('level_weights_name', '?')}', "
              f"len={len(hp.get('level_weights', []))}) "
              f"with '{args.level_weights}' (len={n_levels})")
        hp['level_weights']      = new_weights
        hp['level_weights_name'] = args.level_weights
    elif len(hp.get('level_weights', [])) != n_levels:
        raise ValueError(
            f"level_weights in config has {len(hp.get('level_weights', []))} entries "
            f"but the HDF5 profile grid has {n_levels} levels.  Re-run with "
            f"--level-weights uniform (or top_50style/bottom_50style/u_shape) "
            f"to rebuild the weights for this profile resolution."
        )

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model_config = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    model = ProfileOnlyNetwork(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters  (input_dim={model_config.input_dim})")

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

    # ── Training loop ─────────────────────────────────────────────────────────
    n_epochs            = hp.get('n_epochs', 1000)
    early_stop_patience = hp.get('early_stop_patience', 150)
    warmup_steps        = hp.get('warmup_steps', 500)
    augment_noise_std   = hp.get('augment_noise_std', 0.0)
    target_lr           = hp['learning_rate']

    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0
    history = {'train': [], 'val': []}

    print(f"\nTraining up to {n_epochs} epochs "
          f"(early-stop patience={early_stop_patience}, "
          f"warmup_steps={warmup_steps}, "
          f"augment_noise_std={augment_noise_std})")
    print("-" * 70)

    t0 = time.time()
    final_epoch = -1
    for epoch in range(n_epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=augment_noise_std,
            warmup_steps=warmup_steps,
            target_lr=target_lr,
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
                'model_config': {
                    'n_wavelengths':     model_config.n_wavelengths,
                    'n_geometry_inputs': model_config.n_geometry_inputs,
                    'n_levels':          model_config.n_levels,
                    'hidden_dims':       list(model_config.hidden_dims),
                    'dropout':           model_config.dropout,
                    'activation':        model_config.activation,
                },
            }, output_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement in {early_stop_patience} epochs)")
                final_epoch = epoch
                break

        final_epoch = epoch
        print(f"Epoch {epoch:4d} | Train: {train_loss:+.4f} | "
              f"Val: {val_loss:+.4f} | LR: {lr:.1e} | "
              f"No-improve: {epochs_no_improve}")

    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.0f}s ({final_epoch + 1} epochs)")

    # ── Reload best checkpoint and evaluate on test set ──────────────────────
    ckpt = torch.load(output_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    results = predict_test(model, test_loader, device, model_config)

    rmse_per_level = np.sqrt(np.mean((results['pred'] - results['true']) ** 2, axis=0))
    mean_rmse = float(rmse_per_level.mean())
    sigma_overall = float(results['pred_std'].mean())

    print(f"\nTest metrics (best checkpoint from epoch {int(ckpt['epoch'])}):")
    print(f"  Mean RMSE:           {mean_rmse:.3f} μm  (across {n_levels} levels)")
    print(f"  Mean predicted σ:    {sigma_overall:.3f} μm")
    print(f"  RMSE/σ ratio:        {mean_rmse / max(sigma_overall, 1e-9):.2f}  "
          f"(1.0 = calibrated)")
    print(f"  Per-level RMSE:")
    for L, rmse in enumerate(rmse_per_level, 1):
        print(f"    L{L:02d}: {rmse:.3f} μm")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'source_run_id':      args.run_id,
            'source_sweep_dir':   str(sweep_dir),
            'h5_path':            str(h5_path),
            'n_levels':           int(n_levels),
            'n_params':           n_params,
            'best_epoch':         int(ckpt['epoch']),
            'final_epoch':        int(final_epoch),
            'train_time_seconds': float(train_time),
            'best_val_loss':      float(best_val_loss),
            'mean_rmse':          mean_rmse,
            'mean_predicted_sigma': sigma_overall,
            'rmse_per_level':     rmse_per_level.tolist(),
        }, f, indent=2)

    plot_loss_curves(history, output_dir / 'loss_curves.png')
    plot_profiles_true_vs_pred(results, test_loader, h5_path, n_levels,
                               output_dir / 'profiles_true_vs_pred.png')

    print(f"\nAll artifacts written to: {output_dir}")
    print(f"  best_model.pt")
    print(f"  config.json, history.json, results.json")
    print(f"  loss_curves.png (500 DPI)")
    print(f"  profiles_true_vs_pred.png (500 DPI)")


if __name__ == '__main__':
    main()
