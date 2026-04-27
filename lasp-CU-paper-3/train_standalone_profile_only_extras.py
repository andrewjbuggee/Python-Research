"""
train_standalone_profile_only_extras.py — Train a single ProfileOnlyNetworkExtras
model that augments the existing 640-dim spectrum+geometry input with three
scalar physical priors:

    idx 640 : log10(tau_c)              z-scored
    idx 641 : log10(wv_above_cloud)     z-scored
    idx 642 : log10(wv_in_cloud)        z-scored

Hyperparameters are reused from a config in
    hyper_parameter_sweep/sweep_results_profile_only/run_NNN/config.json
so any of the 150 configs from the K=5 profile-only sweep can be retrained
in this 643-input variant for direct comparison.

Ablation flags
--------------
Use the three independent --zero-* flags to measure how much information
each scalar contributes vs. what the network already extracts from the
640-dim spectrum+geometry alone.  Zero = the channel is set to 0.0 (the
z-score mean — i.e. "the dataset mean, no per-sample information").

    --zero-tau-c        zero the cloud optical-thickness channel
    --zero-wv-above     zero the above-cloud water-vapor channel
    --zero-wv-in        zero the within-cloud water-vapor channel

Pass any combination.  All three together = "input vector is back to
640 effective channels" — a parity check against train_standalone_profile_only.py.

Outputs
-------
    <output-dir>/
        best_model.pt                 (state dict + model_config + n_extras)
        config.json                   (effective config + h5 + ablation flags)
        history.json                  (per-epoch train+val loss)
        results.json                  (final test metrics)
        loss_curves.png               (500 DPI)
        profiles_true_vs_pred.png     (500 DPI; no τ overlay)

Usage
-----
    # Full 643-input model:
    python train_standalone_profile_only_extras.py \
        --run-id 110 \
        --h5-path training_data/combined_vocals_oracles_training_data_50-evenZ-levels_23_April_2026.h5 \
        --output-dir ./standalone_results_profile_only_extras/run110_643input

    # Ablation: only spectrum+geometry+tau_c (no water vapor):
    python train_standalone_profile_only_extras.py \
        --run-id 110 --h5-path <path> \
        --zero-wv-above --zero-wv-in \
        --output-dir ./standalone_results_profile_only_extras/run110_no_wv

    # Parity check against the 640-input model:
    python train_standalone_profile_only_extras.py \
        --run-id 110 --h5-path <path> \
        --zero-tau-c --zero-wv-above --zero-wv-in

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
from models_profile_only          import ProfileOnlyLoss
from models_profile_only_extras   import ProfileOnlyNetworkExtras
from models                       import RetrievalConfig
from data                         import create_dataloaders_extras, resolve_h5_path
# Reuse plotting + level-weight helpers (DRY) from the 640-input trainer.
from train_standalone_profile_only import (
    build_level_weights,
    plot_loss_curves,
    plot_profiles_true_vs_pred,
)


N_EXTRAS = 3   # (tau_c, wv_above_cloud, wv_in_cloud)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-id', type=int, required=True,
                   help='Sweep run ID to borrow hyperparameters from, e.g. 110 -> run_110')
    p.add_argument('--h5-path', type=str, required=True,
                   help='HDF5 training file (overrides the path saved in the run config).')
    p.add_argument('--training-data-dir', type=str, default=None,
                   help='Directory hosting the HDF5; overrides the dir portion of --h5-path.')
    p.add_argument('--sweep-dir', type=str,
                   default='hyper_parameter_sweep/sweep_results_profile_only',
                   help='Directory holding run_NNN/ subfolders (relative to repo root)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to save outputs.  Default: '
                        './standalone_results_profile_only_extras/run<ID>_<h5stem>[_ablation]')
    p.add_argument('--device', type=str, default=None,
                   help='"cuda", "mps", "cpu". Default: cuda > mps > cpu.')
    p.add_argument('--n-epochs', type=int, default=None,
                   help='Override n_epochs from the run config')
    p.add_argument('--level-weights', type=str, default=None,
                   choices=['uniform', 'deep', 'top',
                            'top_50style', 'bottom_50style', 'u_shape'],
                   help='Override saved level_weights with a rebuilt scheme matching n_levels')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-val-profiles',  type=int, default=14)
    p.add_argument('--n-test-profiles', type=int, default=14)

    # Ablation flags
    p.add_argument('--zero-tau-c',    action='store_true',
                   help='Zero the tau_c channel (idx 640) — set to z-score 0.0 '
                        '(dataset mean), removing per-sample info from this channel.')
    p.add_argument('--zero-wv-above', action='store_true',
                   help='Zero the wv_above_cloud channel (idx 641).')
    p.add_argument('--zero-wv-in',    action='store_true',
                   help='Zero the wv_in_cloud channel (idx 642).')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation / inference
# (parity with train_standalone_profile_only.py — identical except the network
#  ingests 643 inputs and the dataloaders come from create_dataloaders_extras.)
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
            # Gaussian *fractional* noise on x.  Note that channels 640..642
            # are z-scored (can be negative), so multiplicative noise is a
            # standard-deviation-proportional perturbation — the same shape
            # as the geometry channels.  This keeps behaviour identical to
            # the 640-input trainer.
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

    # ── HDF5 path + sanity check on the three new datasets ───────────────────
    h5_path = resolve_h5_path(args.h5_path, args.training_data_dir).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(
            f'HDF5 file not found: {h5_path}\n'
            f'  --h5-path           = {args.h5_path}\n'
            f'  --training-data-dir = {args.training_data_dir}'
        )
    cfg.setdefault('data', {})
    cfg['data']['h5_path'] = str(h5_path)
    with h5py.File(h5_path, 'r') as f:
        required = ['sza', 'vza', 'saz', 'vaz',
                    'tau_c', 'wv_above_cloud', 'wv_in_cloud']
        missing  = [k for k in required if k not in f]
    if missing:
        raise RuntimeError(
            f'HDF5 {h5_path.name} is missing required dataset(s): {missing}.'
        )

    if args.n_epochs is not None:
        hp['n_epochs'] = args.n_epochs

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        ablation_tag = ''
        zeroed = []
        if args.zero_tau_c:    zeroed.append('noTau')
        if args.zero_wv_above: zeroed.append('noWvAbove')
        if args.zero_wv_in:    zeroed.append('noWvIn')
        if zeroed:
            ablation_tag = '_' + '_'.join(zeroed)
        output_dir = (repo_root / 'standalone_results_profile_only_extras'
                      / f'run{args.run_id:03d}_{h5_path.stem}{ablation_tag}'
                      ).resolve()
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

    # ── Banner ────────────────────────────────────────────────────────────────
    active_extras = []
    for name, flag in [('tau_c',          args.zero_tau_c),
                       ('wv_above_cloud', args.zero_wv_above),
                       ('wv_in_cloud',    args.zero_wv_in)]:
        active_extras.append(f"{name}={'ZEROED' if flag else 'active'}")

    print(f"\n{'=' * 70}")
    print(f"  STANDALONE TRAIN — profile-only + 3 extras, "
          f"hyperparameters from run_{args.run_id:03d}")
    print(f"{'=' * 70}")
    print(f"  sweep run dir : {run_dir}")
    print(f"  HDF5 file     : {h5_path}")
    print(f"  output dir    : {output_dir}")
    print(f"  device        : {device}")
    print(f"\n  Network input layout:  636 reflectance ⊕ 4 geometry "
          f"(sza, vza, saz, vaz) ⊕ 3 extras = 643")
    print(f"  Extras status: {' | '.join(active_extras)}")
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

    # ── Dataloaders (extras + profile-aware split) ───────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders_extras(
        h5_path=str(h5_path),
        instrument=cfg['data'].get('instrument', 'hysics'),
        batch_size=hp.get('batch_size', 256),
        num_workers=cfg['data'].get('num_workers', 4),
        seed=args.seed,
        n_val_profiles=args.n_val_profiles,
        n_test_profiles=args.n_test_profiles,
        zero_tau_c=args.zero_tau_c,
        zero_wv_above=args.zero_wv_above,
        zero_wv_in=args.zero_wv_in,
    )
    print(f"Data: {len(train_loader.dataset):,} train, "
          f"{len(val_loader.dataset):,} val, {len(test_loader.dataset):,} test")

    # n_levels
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
            f"but the HDF5 grid has {n_levels} levels.  Pass --level-weights "
            f"uniform (or another scheme) to rebuild for this n_levels."
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
    model = ProfileOnlyNetworkExtras(model_config, n_extras=N_EXTRAS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters  "
          f"(input_dim={model_config.input_dim} + n_extras={N_EXTRAS} = {model.input_dim})")

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
                'n_extras': N_EXTRAS,
                'ablation': {
                    'zero_tau_c':    bool(args.zero_tau_c),
                    'zero_wv_above': bool(args.zero_wv_above),
                    'zero_wv_in':    bool(args.zero_wv_in),
                },
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

    # ── Best-checkpoint test evaluation ──────────────────────────────────────
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
    cfg['ablation'] = {
        'zero_tau_c':    bool(args.zero_tau_c),
        'zero_wv_above': bool(args.zero_wv_above),
        'zero_wv_in':    bool(args.zero_wv_in),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'source_run_id':        args.run_id,
            'source_sweep_dir':     str(sweep_dir),
            'h5_path':              str(h5_path),
            'n_levels':             int(n_levels),
            'n_params':             n_params,
            'n_extras':             N_EXTRAS,
            'ablation': {
                'zero_tau_c':    bool(args.zero_tau_c),
                'zero_wv_above': bool(args.zero_wv_above),
                'zero_wv_in':    bool(args.zero_wv_in),
            },
            'best_epoch':           int(ckpt['epoch']),
            'final_epoch':          int(final_epoch),
            'train_time_seconds':   float(train_time),
            'best_val_loss':        float(best_val_loss),
            'mean_rmse':            mean_rmse,
            'mean_predicted_sigma': sigma_overall,
            'rmse_per_level':       rmse_per_level.tolist(),
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
