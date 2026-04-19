"""
train_standalone.py — Train a single model using the saved hyperparameters of
one sweep run, optionally on a different HDF5 dataset.

Why this exists
---------------
sweep_results_3/run_NNN/config.json stores the full hyperparameter set used
for that sweep run (architecture, LR, dropout, sigma_floor, warmup, noise
augmentation, all physics-loss lambdas, level_weights, batch_size, epoch
budget, etc.).  This script loads one such config, overrides only the
`h5_path` with whatever file you point at on the command line, and retrains
from scratch.  Use it to test a previously-selected champion configuration
on new training data (e.g. the 8-level evenly-sampled or tau-weighted files
you just generated) without editing any source.

Outputs
-------
    <output-dir>/
        best_model.pt
        config.json                       (effective config, including the h5 override)
        history.json                      (per-epoch train+val loss)
        results.json                      (final test metrics)
        loss_curves.png                   (plot 1 from notebook, 500 DPI)
        profiles_true_vs_pred.png         (plot 2 from notebook, 500 DPI)

Usage
-----
    python train_standalone.py \
        --run-id 50 \
        --h5-path /path/to/combined_vocals_oracles_training_data_8-levels_17_April_2026.h5 \
        --output-dir ./standalone_results/run050_8levels_even

    # Smoke test on CPU, no GPU needed:
    python train_standalone.py --run-id 50 --h5-path ... --device cpu

Author: Andrew J. Buggee, LASP / CU Boulder
"""

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
from models import DropletProfileNetwork, CombinedLoss, RetrievalConfig
from data import create_dataloaders


GREEN    = '#10B981'   # NN retrieval
RAW_BLUE = '#1F5FC2'   # raw in-situ


# Reference 'deep' scheme from generate_sweep_2.py at the original 7-level grid
# (emphasises cloud base).  For other n_levels we linearly interpolate these
# anchor values onto the target grid — keeps the shape of the scheme intact
# while adapting to the profile resolution.
DEEP_SCHEME_REF_7LEVEL = np.array([1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])


def build_level_weights(scheme: str, n_levels: int) -> list:
    """
    Generate a length-n_levels level_weights vector for the requested scheme.

    Parameters
    ----------
    scheme   : 'uniform' or 'deep'
    n_levels : number of profile levels in the HDF5 you are training on

    Returns
    -------
    list of float, length n_levels
    """
    if scheme == 'uniform':
        return [1.0] * n_levels
    if scheme == 'deep':
        ref = DEEP_SCHEME_REF_7LEVEL
        xs  = np.linspace(0.0, len(ref) - 1, n_levels)
        return np.interp(xs, np.arange(len(ref)), ref).tolist()
    raise ValueError(f"level_weights scheme must be 'uniform' or 'deep', "
                     f"got {scheme!r}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-id', type=int, required=True,
                   help='Sweep run ID to borrow hyperparameters from, e.g. 50 -> run_050')
    p.add_argument('--h5-path', type=str, required=True,
                   help='HDF5 training file to train on (overrides the one saved in the run config)')
    p.add_argument('--sweep-dir', type=str,
                   default='hyper_parameter_sweep/sweep_results_3',
                   help='Directory holding the run_NNN/ subfolders (relative to repo root)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to save outputs. Default: ./standalone_results/run<ID>_<h5stem>')
    p.add_argument('--device', type=str, default=None,
                   help='"cuda", "mps", "cpu". Default: cuda if available, else cpu.')
    p.add_argument('--n-epochs', type=int, default=None,
                   help='Override n_epochs from the run config (default: use what was saved)')
    p.add_argument('--level-weights', choices=['uniform', 'deep'], default=None,
                   help='Override the saved level_weights with a rebuilt one matching '
                        'the HDF5 n_levels.  "uniform" = [1]*n, "deep" = interpolated '
                        'gradient toward base.  Default: keep the saved weights and '
                        'enforce length-matches-n_levels.')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for the train/val/test split and weight init')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (copy of sweep_train.py's, so retrained model matches the sweep)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    augment_noise_std=0.0,
                    warmup_steps=0, target_lr=None, global_step_start=0):
    model.train()
    loss_sum, n = 0.0, 0
    step = global_step_start
    for x, prof, tau in loader:
        x, prof, tau = x.to(device), prof.to(device), tau.to(device)

        if warmup_steps > 0 and step < warmup_steps and target_lr is not None:
            lr_now = target_lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

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
    for x, prof, tau in loader:
        x, prof, tau = x.to(device), prof.to(device), tau.to(device)
        output = model(x)
        losses = criterion(output, prof, tau)
        loss_sum += losses['total'].item()
        n += 1
    return loss_sum / n


@torch.no_grad()
def predict_test(model, loader, device, config):
    """Run inference on the whole test loader.  Returns arrays in physical units."""
    model.eval()
    pred_list, std_list = [], []
    true_list = []
    tau_pred_list, tau_std_list, tau_true_list = [], [], []

    re_min, re_max = config.re_min, config.re_max
    tau_min, tau_max = config.tau_min, config.tau_max

    for x, prof, tau in loader:
        out = model(x.to(device))
        pred_list.append(out['profile'].cpu().numpy())
        std_list.append(out['profile_std'].cpu().numpy())
        true_list.append(prof.numpy() * (re_max - re_min) + re_min)
        tau_pred_list.append(out['tau_c'].squeeze(-1).cpu().numpy())
        tau_std_list.append(out['tau_std'].squeeze(-1).cpu().numpy())
        tau_true_list.append(tau.numpy() * (tau_max - tau_min) + tau_min)

    return {
        'pred':     np.concatenate(pred_list),
        'pred_std': np.concatenate(std_list),
        'true':     np.concatenate(true_list),
        'tau_pred':     np.concatenate(tau_pred_list),
        'tau_pred_std': np.concatenate(tau_std_list),
        'tau_true':     np.concatenate(tau_true_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots — replicated from 02_training_experimentation.ipynb
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(history, out_path):
    """Plot 1: train + val NLL vs epoch.  Same style as notebook cell 13."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(history['train'], label='Train', linewidth=1.5)
    ax.plot(history['val'],   label='Validation', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total NLL Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def plot_profiles_true_vs_pred(results, test_loader, h5_path, n_levels,
                               out_path, n_show=6, seed=0):
    """
    Plot 2: 2×3 panel figure of predicted vs true droplet profiles.

    Left y-axis : cloud optical depth τ (measured from HDF5's profiles_raw_tau
                  if present, otherwise a linear approximation from tau_c).
    Right y-axis: altitude in km (measured from profiles_raw_z).
    Overlays    : raw in-situ r_e profile (blue), interpolated true profile on
                  the model grid (black circles), NN retrieval with ±1σ (green
                  squares).  No adiabatic reference.
    Title       : τ_true and τ_pred ± 1σ.

    Test-set deduplication: each true cloud profile appears ~128 times in the
    test set via varied sun/view geometry.  We pick n_show *unique* clouds so
    every panel shows a different cloud.
    """
    pred       = results['pred']
    pred_std   = results['pred_std']
    true_um    = results['true']
    tau_true   = results['tau_true']
    tau_pred   = results['tau_pred']
    tau_std    = results['tau_pred_std']

    # Unwrap Subset → Dataset to get the global HDF5 indices for this test set.
    subset = test_loader.dataset
    if not hasattr(subset, 'indices'):
        raise RuntimeError(
            "Expected test_loader.dataset to be a torch Subset (from "
            "profile_holdout splitting).  Cannot locate raw HDF5 rows."
        )
    test_idx_global = np.asarray(subset.indices, dtype=int)
    # h5py fancy-indexing requires strictly increasing order:
    sort_order   = np.argsort(test_idx_global)
    unsort_order = np.argsort(sort_order)
    sorted_idx   = test_idx_global[sort_order]

    with h5py.File(h5_path, 'r') as f:
        raw_sorted  = f['profiles_raw'][sorted_idx]       # (n_test, max_lev), NaN-padded
        z_sorted    = f['profiles_raw_z'][sorted_idx]
        nlev_sorted = f['profile_n_levels'][sorted_idx]
        has_raw_tau = 'profiles_raw_tau' in f
        tau_raw_sorted = f['profiles_raw_tau'][sorted_idx] if has_raw_tau else None
    raw_all    = raw_sorted[unsort_order]
    z_all      = z_sorted[unsort_order]
    nlev_all   = nlev_sorted[unsort_order]
    tau_rawall = tau_raw_sorted[unsort_order] if has_raw_tau else None
    print(f"  tau axis source: "
          f"{'measured profiles_raw_tau' if has_raw_tau else 'linear approx from tau_c'}")

    # Dedup: each unique cloud appears ~128 times (varied sun/view geometry).
    # Build a pool of unique profiles, then sample n_show from it.
    rng = np.random.default_rng(seed)
    seen, unique_idx = set(), []
    for i in range(pred.shape[0]):
        key = tuple(np.round(true_um[i], 6))
        if key not in seen:
            seen.add(key)
            unique_idx.append(i)
        if len(unique_idx) >= n_show * 3:  # plenty to sample from
            break
    if len(unique_idx) < n_show:
        print(f"  [warn] only {len(unique_idx)} unique test profiles "
              f"(asked for {n_show}); showing what we have")
        pick = np.array(unique_idx)
    else:
        pick = rng.choice(unique_idx, size=n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for k, (ax, i) in enumerate(zip(axes.flat, pick)):
        n_lev_i  = int(nlev_all[i])
        re_raw   = raw_all[i, :n_lev_i]
        z_raw_km = z_all[i, :n_lev_i]
        tau_raw_i = tau_rawall[i, :n_lev_i] if has_raw_tau else None

        # Model grid altitudes: linspace from cloud top to base (same as the
        # converter's interpolate_profile).
        z_top_km  = float(z_raw_km[0])
        z_base_km = float(z_raw_km[-1])
        z_lvl     = np.linspace(z_top_km, z_base_km, n_levels)

        # τ at each of the N model levels.  If the HDF5 has the measured
        # profiles_raw_tau, interpolate exactly onto the altitude grid;
        # otherwise fall back to a uniform linear stretch from 0 to τ_c.
        if tau_raw_i is not None:
            z_asc   = z_raw_km[::-1]
            tau_asc = tau_raw_i[::-1]
            tau_lvl = np.interp(z_lvl, z_asc, tau_asc)
            tau_for_raw  = tau_raw_i
            tau_axis_lbl = r'Optical depth $\tau$'
        else:
            tau_lvl = np.linspace(0.0, float(tau_true[i]), n_levels)
            tau_for_raw = ((z_top_km - z_raw_km)
                           / max(z_top_km - z_base_km, 1e-9)) * float(tau_true[i])
            tau_axis_lbl = r'Optical depth $\tau$ (linear approx.)'

        # Raw in-situ r_e profile (blue, full resolution)
        ax.plot(re_raw, tau_for_raw, '-', color=RAW_BLUE,
                linewidth=1.0, alpha=0.60,
                label='Original in-situ profile')

        # Interpolated true profile on the model grid (black circles)
        ax.plot(true_um[i], tau_lvl, 'ko-', markersize=4, linewidth=1.5,
                label=f'True ({n_levels} levels)')

        # NN retrieval with ±1σ
        ax.errorbar(pred[i], tau_lvl, xerr=pred_std[i],
                    fmt='s--', color=GREEN, markersize=4, linewidth=1.5,
                    elinewidth=1.0, capsize=3,
                    label='PINN retrieval ±1σ')

        # τ=0 (cloud top) at the top of the plot; τ=τ_c at the bottom.
        ax.invert_yaxis()
        ax.set_xlabel(r'$r_e$ (μm)', fontsize=10)
        if k % 3 == 0:
            ax.set_ylabel(tau_axis_lbl, fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

        # Right-hand y-axis: altitude (km).  Because τ=0 (top of plot) <-> z_top
        # and τ=τ_c (bottom) <-> z_base, set ylim=(z_base, z_top) with no invert.
        ax2 = ax.twinx()
        ax2.set_ylim(z_base_km, z_top_km)
        if k % 3 == 2:
            ax2.set_ylabel('Altitude (km)', fontsize=10)

        # Title: true and predicted column optical depth
        ax.set_title(
            rf'$\tau_\mathrm{{true}}$ = {float(tau_true[i]):.2f}   '
            rf'$\tau_\mathrm{{pred}}$ = {float(tau_pred[i]):.2f} ± '
            rf'{float(tau_std[i]):.2f}',
            fontsize=10,
        )
        if k == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Predicted vs True Droplet Profiles (Test Set)',
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
    h5_path = Path(args.h5_path).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 file not found: {h5_path}')
    cfg.setdefault('data', {})
    cfg['data']['h5_path'] = str(h5_path)

    # ── Optional overrides from the CLI ───────────────────────────────────────
    if args.n_epochs is not None:
        hp['n_epochs'] = args.n_epochs

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (repo_root / 'standalone_results'
                      / f'run{args.run_id:03d}_{h5_path.stem}').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STANDALONE TRAIN — borrowing hyperparameters from run_{args.run_id:03d}")
    print(f"{'='*70}")
    print(f"  sweep run dir : {run_dir}")
    print(f"  HDF5 file     : {h5_path}")
    print(f"  output dir    : {output_dir}")
    print(f"  device        : {device}")
    print(f"\n  Hyperparameters being reused:")
    for k, v in hp.items():
        print(f"    {k:22s} = {v}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=str(h5_path),
        instrument=cfg['data'].get('instrument', 'hysics'),
        batch_size=hp.get('batch_size', 256),
        num_workers=cfg['data'].get('num_workers', 4),
        seed=args.seed,
        profile_holdout=True,
        n_val_profiles=14,
        n_test_profiles=14,
    )
    print(f"Data: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # Data-driven n_levels (Subset doesn't forward attributes, so unwrap).
    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels
    print(f"Profile grid: {n_levels} levels (read from HDF5)")

    if args.level_weights is not None:
        # CLI-requested override: rebuild weights to match this HDF5's n_levels,
        # so length mismatches between saved run_config and new dataset are OK.
        new_weights = build_level_weights(args.level_weights, n_levels)
        print(f"Overriding level_weights from run_{args.run_id:03d}'s "
              f"'{hp.get('level_weights_name', '?')}' "
              f"(len={len(hp['level_weights'])}) with '{args.level_weights}' "
              f"(len={n_levels}):")
        print(f"  {np.round(new_weights, 3).tolist()}")
        hp['level_weights']      = new_weights
        hp['level_weights_name'] = args.level_weights
    elif len(hp['level_weights']) != n_levels:
        raise ValueError(
            f"level_weights in config has {len(hp['level_weights'])} entries but "
            f"the HDF5 profile grid has {n_levels} levels.  Re-run with "
            f"--level-weights uniform  or  --level-weights deep  to rebuild the "
            f"weights for this profile resolution."
        )

    # ── Model, loss, optimizer ────────────────────────────────────────────────
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
    print(f"Model: {n_params:,} parameters")

    level_weights = torch.tensor(hp['level_weights'], dtype=torch.float32)
    criterion = CombinedLoss(
        config=model_config,
        lambda_physics=hp.get('lambda_physics', 0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic', 0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        lambda_emulator_data=0.0,
        level_weights=level_weights,
        sigma_floor=hp.get('sigma_floor', 0.01),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=hp.get('scheduler_patience', 30),
                                  min_lr=1e-6)

    # ── Training loop (sweep_train.py parity) ─────────────────────────────────
    n_epochs            = hp.get('n_epochs', 400)
    early_stop_patience = hp.get('early_stop_patience', 80)
    warmup_steps        = hp.get('warmup_steps', 0)
    augment_noise_std   = hp.get('augment_noise_std', 0.0)
    target_lr           = hp['learning_rate']

    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0
    history = {'train': [], 'val': []}

    print(f"\nTraining for up to {n_epochs} epochs "
          f"(early-stop patience={early_stop_patience}, warmup_steps={warmup_steps}, "
          f"augment_noise_std={augment_noise_std})")
    print("-" * 70)

    t0 = time.time()
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

    train_time = time.time() - t0
    final_epoch = epoch
    print(f"\nTraining complete in {train_time:.0f}s ({final_epoch+1} epochs)")

    # ── Reload best checkpoint and evaluate on test set ──────────────────────
    ckpt = torch.load(output_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    results = predict_test(model, test_loader, device, model_config)

    # Per-level RMSE and summary metrics
    rmse_per_level = np.sqrt(np.mean((results['pred'] - results['true'])**2, axis=0))
    mean_rmse = float(rmse_per_level.mean())
    tau_rmse  = float(np.sqrt(np.mean((results['tau_pred'] - results['tau_true'])**2)))

    print(f"\nTest metrics (best checkpoint from epoch {int(ckpt['epoch'])}):")
    print(f"  Mean RMSE: {mean_rmse:.3f} μm")
    print(f"  Tau RMSE:  {tau_rmse:.3f}")
    for lvl, rmse in enumerate(rmse_per_level, 1):
        print(f"    Level {lvl:2d}: {rmse:.3f} μm")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'source_run_id':      args.run_id,
            'h5_path':            str(h5_path),
            'n_levels':           int(n_levels),
            'n_params':           n_params,
            'best_epoch':         int(ckpt['epoch']),
            'final_epoch':        int(final_epoch),
            'train_time_seconds': float(train_time),
            'best_val_loss':      float(best_val_loss),
            'mean_rmse':          mean_rmse,
            'tau_rmse':           tau_rmse,
            'rmse_per_level':     rmse_per_level.tolist(),
        }, f, indent=2)

    # ── Plots (500 DPI) ───────────────────────────────────────────────────────
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
