"""
train_uv_ablation_run004.py — Retrain the run_004 recipe from scratch with the
short-wavelength spectral channels REMOVED, to causally test whether the
352 nm (idx 0) feature-importance peak reflects information the retrieval needs.

Why
---
Single-channel permutation gave 352 nm the highest importance, but diagnostics
showed (a) it is a generic optical-depth/LWP proxy shared across the whole
visible continuum and (b) the peak tracks the model's large learned first-layer
weight on channel 0, not unique information. The clean causal test is to delete
those channels and retrain: if test RMSE is unchanged and the importance peak
migrates to a surviving visible channel, 352 nm was not uniquely informative.

What this does
--------------
Rebuilds run_004's EXACT training recipe from its summary.json (same
hyperparameters, same M0 extras-zeroed dataset, same seed-42 profile-aware
split — i.e. the same 889 test profiles), then drops every spectral channel
with wavelength < --cutoff-nm before the network sees it. The first Linear
shrinks from 643 to (n_kept_spectral + 4 geometry + 3 extras).

    --cutoff-nm 500   ablation: drop the 49 channels < 500 nm (keep 587)
    --cutoff-nm 0     control : keep all 636 channels (identical-recipe baseline)

Run BOTH for a fair comparison (ablation vs control removes train-to-train
variance as a confound).

Outputs (under standalone_results_uv_ablation/<tag>/):
    best_model.pt   (model_state_dict + model_config + keep_idx + cutoff_nm)
    summary.json    (test metrics in the same schema as the sweep)
    history.json

Run:
    /opt/anaconda3/bin/python3 train_uv_ablation_run004.py --cutoff-nm 500
    /opt/anaconda3/bin/python3 train_uv_ablation_run004.py --cutoff-nm 0

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from models import RetrievalConfig                                   # noqa: E402
from models_profile_only_extras import ProfileOnlyNetworkExtras      # noqa: E402
from models_profile_only import ProfileOnlyLoss                      # noqa: E402
from data import (LibRadtranDatasetExtras, create_profile_aware_splits)  # noqa: E402
# Canonical, bit-identical training helpers used by the sweep that made run_004.
from train_standalone_profile_only_extras import (                   # noqa: E402
    train_one_epoch, validate, predict_test)

RUN_004 = (REPO / 'hyper_parameter_sweep'
           / 'sweep_results_profile_only_synthetic_M0_rev2' / 'run_004')
H5 = REPO / 'training_data' / 'synthetic_training_data_7-levels_8_May_2026.h5'
N_EXTRAS = 3
N_SPECTRAL_FULL = 636


class ChannelDropDataset(Dataset):
    """Wrap LibRadtranDatasetExtras and keep only the input indices in keep_idx.

    keep_idx indexes the full 643-dim vector (636 spectral ⊕ 4 geometry ⊕ 3
    extras); the geometry and extras indices are always retained.
    """

    def __init__(self, base: LibRadtranDatasetExtras, keep_idx: np.ndarray):
        self.base = base
        self.keep_idx = torch.as_tensor(keep_idx, dtype=torch.long)
        self.n_levels = base.n_levels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, prof, tau = self.base[i]
        return x.index_select(0, self.keep_idx), prof, tau


def build_keep_idx(cutoff_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_idx into 643-dim x, kept spectral wavelengths nm)."""
    with h5py.File(H5, 'r') as f:
        wl = f['wavelengths'][:].astype(np.float64)               # (636,)
    spec_keep = np.where(wl >= cutoff_nm)[0]                       # indices 0..635
    geom = np.arange(N_SPECTRAL_FULL, N_SPECTRAL_FULL + 4)        # 636..639
    extras = np.arange(N_SPECTRAL_FULL + 4, N_SPECTRAL_FULL + 4 + N_EXTRAS)  # 640..642
    keep_idx = np.concatenate([spec_keep, geom, extras]).astype(np.int64)
    return keep_idx, wl[spec_keep]


def pick_device(arg):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--cutoff-nm', type=float, default=500.0,
                   help='Drop spectral channels with wavelength < cutoff. '
                        '0 = keep all 636 (control). Default 500.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--n-epochs', type=int, default=None,
                   help='Override n_epochs (default: run_004 value = 1500).')
    p.add_argument('--output-dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

    # ── run_004 recipe from its summary.json ─────────────────────────────────
    summ = json.loads((RUN_004 / 'summary.json').read_text())
    hp = summ['hyperparams']
    extras = summ['extras']                       # all True in M0
    n_val = summ['n_val_profiles']                # 889
    n_test = summ['n_test_profiles']              # 889
    instrument = summ['data'].get('instrument', 'hysics')
    n_epochs = args.n_epochs if args.n_epochs is not None else int(hp['n_epochs'])

    keep_idx, kept_wl = build_keep_idx(args.cutoff_nm)
    n_kept_spectral = int((keep_idx < N_SPECTRAL_FULL).sum())

    tag = (args.output_dir if args.output_dir else
           f'control_all636' if args.cutoff_nm <= 0 else
           f'ablation_cutoff{int(args.cutoff_nm)}nm_keep{n_kept_spectral}')
    out_dir = (REPO / 'standalone_results_uv_ablation' / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'  UV ABLATION RETRAIN — cutoff {args.cutoff_nm:.0f} nm '
          f'(keep {n_kept_spectral}/{N_SPECTRAL_FULL} spectral)')
    print('=' * 70)
    print(f'  device     : {device}   seed {args.seed}   n_epochs {n_epochs}')
    print(f'  kept spectral λ range: '
          f'{kept_wl.min():.1f}–{kept_wl.max():.1f} nm')
    print(f'  input dim  : {n_kept_spectral} spectral + 4 geom + {N_EXTRAS} extras '
          f'= {n_kept_spectral + 4 + N_EXTRAS}')
    print(f'  output     : {out_dir}')

    # ── seed, then build the SAME split as run_004 (seed 42) ─────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_ds = LibRadtranDatasetExtras(
        str(H5), normalize=True, instrument=instrument,
        zero_tau_c=extras['zero_tau_c'], zero_wv_above=extras['zero_wv_above'],
        zero_wv_in=extras['zero_wv_in'])
    ds = ChannelDropDataset(base_ds, keep_idx)
    n_levels = base_ds.n_levels

    # Materialize the (channel-dropped) inputs + targets to in-memory tensors
    # ONCE via the dataset's own __getitem__ — values are byte-identical to the
    # per-sample pipeline, but epochs become GPU-bound on the tiny MLP instead
    # of paying per-sample numpy cost every epoch.
    N = len(ds)
    X_all = torch.empty((N, ds[0][0].shape[0]), dtype=torch.float32)
    P_all = torch.empty((N, n_levels), dtype=torch.float32)
    for i in range(N):
        xi, pi, _ = ds[i]
        X_all[i], P_all[i] = xi, pi
    mem_ds = torch.utils.data.TensorDataset(X_all, P_all)

    train_idx, val_idx, test_idx = create_profile_aware_splits(
        str(H5), n_val_profiles=n_val, n_test_profiles=n_test, seed=args.seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(Subset(mem_ds, train_idx), batch_size=hp['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(Subset(mem_ds, val_idx), batch_size=hp['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(Subset(mem_ds, test_idx), batch_size=hp['batch_size'],
                             shuffle=False, num_workers=0, pin_memory=pin)
    print(f'  data: {len(train_idx):,} train / {len(val_idx):,} val / '
          f'{len(test_idx):,} test  |  {n_levels} levels')

    # ── model / loss / optim (run_004 hyperparameters) ───────────────────────
    model_cfg = RetrievalConfig(
        n_wavelengths=n_kept_spectral, n_geometry_inputs=4, n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']), dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'))
    model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)
    # sanity: dataset feature count == model input
    assert ds[0][0].shape[-1] == model.input_dim, \
        f'{ds[0][0].shape[-1]} != {model.input_dim}'
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  model: {n_params:,} params, input_dim {model.input_dim}')

    criterion = ProfileOnlyLoss(
        config=model_cfg,
        lambda_physics=hp.get('lambda_physics', 0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic', 0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        level_weights=torch.tensor(hp['level_weights'], dtype=torch.float32),
        sigma_floor=hp.get('sigma_floor', 0.01)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=hp.get('scheduler_patience', 30),
                                  factor=0.5)

    early_stop = int(hp.get('early_stop_patience', 150))
    warmup = int(hp.get('warmup_steps', 500))
    aug_noise = float(hp.get('augment_noise_std', 0.0))
    target_lr = float(hp['learning_rate'])

    best_val, best_epoch, best_state, no_improve = float('inf'), -1, None, 0
    history = {'train': [], 'val': []}
    global_step = 0
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=aug_noise, warmup_steps=warmup,
            target_lr=target_lr, global_step_start=global_step)
        val_loss = validate(model, val_loader, criterion, device)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if global_step >= warmup:
            scheduler.step(val_loss)
        if epoch % 25 == 0 or epoch <= 5:
            print(f'  epoch {epoch:4d} | train {train_loss:7.4f} | '
                  f'val {val_loss:7.4f} | lr {optimizer.param_groups[0]["lr"]:.1e} '
                  f'| best {best_val:7.4f}@{best_epoch} | noimp {no_improve}',
                  flush=True)
        if no_improve >= early_stop:
            print(f'  early stop at epoch {epoch}')
            break
    train_seconds = time.time() - t0

    # ── test on best checkpoint ──────────────────────────────────────────────
    model.load_state_dict(best_state)
    pred = predict_test(model, test_loader, device, model_cfg)
    err = pred['pred'] - pred['true']
    per_level = np.sqrt((err ** 2).mean(0)).tolist()
    mean_rmse = float(np.mean(per_level))
    mean_sigma = float(pred['pred_std'].mean())

    summary = {
        'experiment': 'uv_ablation_run004',
        'cutoff_nm': args.cutoff_nm,
        'n_kept_spectral': n_kept_spectral,
        'kept_wl_min_nm': float(kept_wl.min()),
        'kept_wl_max_nm': float(kept_wl.max()),
        'seed': args.seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'epochs_trained': len(history['train']),
        'train_seconds': train_seconds,
        'mean_test_rmse_um': mean_rmse,
        'mean_test_sigma_um': mean_sigma,
        'rmse_sigma_ratio': mean_rmse / max(mean_sigma, 1e-9),
        'per_level_rmse_um': per_level,
        'run_004_reference_rmse_um': summ['mean_test_rmse_um'],
        'hyperparams': hp,
        'data': summ['data'],
        'n_val_profiles': n_val,
        'n_test_profiles': n_test,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out_dir / 'history.json').write_text(json.dumps(history))
    torch.save({
        'model_state_dict': best_state,
        'n_extras': N_EXTRAS,
        'cutoff_nm': args.cutoff_nm,
        'keep_idx': keep_idx.tolist(),
        'kept_wavelengths_nm': kept_wl.tolist(),
        'model_config': {
            'n_wavelengths': n_kept_spectral, 'n_geometry_inputs': 4,
            'n_levels': n_levels, 'hidden_dims': list(hp['hidden_dims']),
            'dropout': hp['dropout'], 'activation': hp.get('activation', 'gelu')},
    }, out_dir / 'best_model.pt')

    print('\n' + '=' * 70)
    print(f'  DONE in {train_seconds:.0f}s ({len(history["train"])} epochs)')
    print(f'  ablated mean test RMSE : {mean_rmse:.4f} μm   '
          f'(run_004 full-spectrum: {summ["mean_test_rmse_um"]:.4f} μm)')
    print(f'  per-level: ' + ' '.join(f'{r:.3f}' for r in per_level))
    print(f'  saved → {out_dir}')


if __name__ == '__main__':
    main()
