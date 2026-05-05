"""
sweep_train_profile_only_synthetic.py — single-config trainer for the
synthetic-cloud profile-only sweep.

Reads one JSON config (produced by generate_sweep_profile_only_synthetic.py),
trains the ProfileOnlyNetworkExtras on a single profile-aware 80/10/10 split
of the synthetic HDF5, and writes summary.json with the test metrics.

Difference vs. sweep_train_profile_only.py
------------------------------------------
1. Single split, not K-fold. With 8,888 samples the val signal is plenty
   stable; running K-fold per config would 5× the wall time for negligible
   precision gain on the cross-config ranking.
2. Always uses ProfileOnlyNetworkExtras (the 643-input network). Which of
   the three extras are zeroed comes from the config's `extras` block,
   which the generator filled in based on the model variant.
3. Reads h5_path / training-data-dir overrides the same way the existing
   sweep does, so the same configs work whether you run on Alpine or local.
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

from models                     import RetrievalConfig
from models_profile_only_extras import ProfileOnlyNetworkExtras
from models_profile_only        import ProfileOnlyLoss
from data                       import create_dataloaders_extras, resolve_h5_path

# Reuse the canonical training/eval/predict helpers from the standalone trainer
# so the sweep stays bit-identical to interactive runs and any API drift in the
# model/loss only has one place to maintain.
from train_standalone_profile_only_extras import (
    train_one_epoch as _train_one_epoch_canonical,
    validate        as _validate_canonical,
    predict_test    as _predict_test_canonical,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Synthetic-cloud profile-only single-config trainer "
                    "(one 80/10/10 split per run).")
    p.add_argument('--config-json',       type=str, required=True,
                   help='Path to JSON config produced by '
                        'generate_sweep_profile_only_synthetic.py')
    p.add_argument('--training-data-dir', type=str, default=None,
                   help='Override directory portion of cfg["data"]["h5_path"].')
    p.add_argument('--h5-path',           type=str, default=None,
                   help='Full override of cfg["data"]["h5_path"].')
    p.add_argument('--output-dir',        type=str, default=None,
                   help='Override cfg["output_dir"]; results are written under '
                        '<output_dir>/run_<id>/.')
    p.add_argument('--device',            type=str, default=None,
                   choices=['cuda', 'mps', 'cpu'])
    p.add_argument('--seed',              type=int, default=42)
    return p.parse_args()


# Canonical helpers are imported above. Keep thin local aliases so the call
# sites below read clean.
train_one_epoch = _train_one_epoch_canonical
validate        = _validate_canonical
predict_test    = _predict_test_canonical


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg_path = Path(args.config_json).resolve()
    cfg      = json.loads(cfg_path.read_text())

    # ── HDF5 path resolution ─────────────────────────────────────────────
    h5_in_cfg = cfg.get('data', {}).get('h5_path', '')
    if args.h5_path:
        cfg.setdefault('data', {})['h5_path'] = args.h5_path
    if args.training_data_dir:
        h5_path = resolve_h5_path(cfg['data']['h5_path'], args.training_data_dir)
    else:
        h5_path = Path(cfg['data']['h5_path'])
    h5_path = h5_path.resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 not found: {h5_path}')
    cfg['data']['h5_path'] = str(h5_path)

    # ── Output dir ───────────────────────────────────────────────────────
    output_root = Path(args.output_dir or cfg.get('output_dir',
                       'sweep_results_profile_only_synthetic'))
    if not output_root.is_absolute():
        output_root = (cfg_path.parent / output_root).resolve()
    run_dir = output_root / f"run_{cfg['run_id']:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    hp     = cfg['hyperparams']
    extras = cfg.get('extras', {})
    zero_tau_c    = bool(extras.get('zero_tau_c',    True))
    zero_wv_above = bool(extras.get('zero_wv_above', True))
    zero_wv_in    = bool(extras.get('zero_wv_in',    True))
    variant       = cfg.get('variant', '?')

    print("=" * 70)
    print(f"  SYNTHETIC SWEEP — variant {variant}, run {cfg['run_id']}")
    print("=" * 70)
    print(f"  config       : {cfg_path}")
    print(f"  HDF5         : {h5_path}")
    print(f"  output dir   : {run_dir}")
    print(f"  device       : {device}")
    print(f"  extras       : "
          f"tau_c={'ZEROED' if zero_tau_c else 'active'} | "
          f"wv_above={'ZEROED' if zero_wv_above else 'active'} | "
          f"wv_in={'ZEROED' if zero_wv_in else 'active'}")
    print(f"  hidden_dims  : {hp['hidden_dims']}")
    print(f"  activation   : {hp['activation']}")
    print(f"  lr / dropout : {hp['learning_rate']:.2e} / {hp['dropout']:.3f}")
    print(f"  batch_size   : {hp['batch_size']}")
    print(f"  level_weights: {hp['level_weights_name']} (len={len(hp['level_weights'])})")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataloaders (single 80/10/10 profile-aware split) ───────────────
    train_loader, val_loader, test_loader = create_dataloaders_extras(
        h5_path=str(h5_path),
        instrument=cfg['data'].get('instrument', 'hysics'),
        batch_size=hp['batch_size'],
        num_workers=cfg['data'].get('num_workers', 4),
        seed=args.seed,
        n_val_profiles=cfg['n_val_profiles'],
        n_test_profiles=cfg['n_test_profiles'],
        zero_tau_c=zero_tau_c,
        zero_wv_above=zero_wv_above,
        zero_wv_in=zero_wv_in,
    )

    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels
    print(f"Profile grid: {n_levels} levels  (read from HDF5)")
    if len(hp['level_weights']) != n_levels:
        raise ValueError(
            f"level_weights has {len(hp['level_weights'])} entries but "
            f"the HDF5 has {n_levels} levels — regenerate configs.")

    level_weights = torch.tensor(hp['level_weights'], dtype=torch.float32)

    # ── Model + optimizer + criterion (mirrors train_standalone_profile_only_extras) ─
    model_cfg = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    N_EXTRAS = 3
    # base_ds[idx] returns (x, profile, tau) — see LibRadtranDatasetExtras.
    # Sanity-check dataset / model dim agreement before building the network.
    sample    = base_ds[0]
    input_dim = int(sample[0].shape[-1])
    expected  = model_cfg.input_dim + N_EXTRAS
    if input_dim != expected:
        raise RuntimeError(
            f"Dataset x has {input_dim} features but the model expects "
            f"{expected} (= {model_cfg.input_dim} base + {N_EXTRAS} extras).")
    model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)

    criterion = ProfileOnlyLoss(
        config=model_cfg,
        lambda_physics=hp.get('lambda_physics',      0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic',  0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        level_weights=level_weights,
        sigma_floor=hp.get('sigma_floor', 0.01),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=hp.get('scheduler_patience', 30),
                                  factor=0.5)

    # ── Training loop (mirrors train_standalone_profile_only_extras) ───
    n_epochs        = int(hp['n_epochs'])
    early_stop      = int(hp.get('early_stop_patience', 150))
    warmup          = int(hp.get('warmup_steps', 500))
    aug_noise       = float(hp.get('augment_noise_std', 0.0))
    target_lr       = float(hp['learning_rate'])

    best_val   = float('inf')
    best_epoch = -1
    best_state = None
    no_improve = 0
    history    = {'train': [], 'val': []}
    global_step = 0

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        # Warmup is handled inside train_one_epoch (per-step LR ramp).
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=aug_noise,
            warmup_steps=warmup,
            target_lr=target_lr,
            global_step_start=global_step,
        )
        val_loss = validate(model, val_loader, criterion, device)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if global_step >= warmup:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch <= 5:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | Train {train_loss:7.4f} | "
                  f"Val {val_loss:7.4f} | LR {cur_lr:.1e} | NoImp {no_improve}")

        if no_improve >= early_stop:
            print(f"  Early stop at epoch {epoch} (no improvement in {early_stop})")
            break

    train_seconds = time.time() - t0

    # ── Test on best checkpoint ─────────────────────────────────────────
    model.load_state_dict(best_state)
    # predict_test returns dict with 'pred', 'pred_std', 'true' — all in μm
    # (denormalization done inside the canonical helper).
    pred = predict_test(model, test_loader, device, model_cfg)
    mu_um    = pred['pred']
    sigma_um = pred['pred_std']
    y_um     = pred['true']

    err          = mu_um - y_um
    per_level_rmse = np.sqrt(np.mean(err ** 2, axis=0)).tolist()
    mean_rmse    = float(np.mean(per_level_rmse))
    mean_sigma   = float(np.mean(sigma_um))
    rmse_sigma   = mean_rmse / max(mean_sigma, 1e-9)

    summary = {
        'run_id':           cfg['run_id'],
        'tag':              cfg.get('tag', ''),
        'variant':          variant,
        'extras':           extras,
        'best_epoch':       best_epoch,
        'best_val_loss':    best_val,
        'epochs_trained':   len(history['train']),
        'train_seconds':    train_seconds,
        'mean_test_rmse_um':  mean_rmse,
        'mean_test_sigma_um': mean_sigma,
        'rmse_sigma_ratio':   rmse_sigma,
        'per_level_rmse_um':  per_level_rmse,
        'hyperparams':        hp,
        'data':               cfg['data'],
        'n_val_profiles':     cfg['n_val_profiles'],
        'n_test_profiles':    cfg['n_test_profiles'],
    }

    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    torch.save(best_state, run_dir / 'best_model.pt')

    print(f"\n  best epoch  : {best_epoch}  (val NLL {best_val:.4f})")
    print(f"  mean RMSE   : {mean_rmse:.3f} μm")
    print(f"  mean σ      : {mean_sigma:.3f} μm  (RMSE/σ = {rmse_sigma:.2f})")
    print(f"  per-level   : "
          + ' '.join(f"L{l+1:02d}={r:.2f}" for l, r in enumerate(per_level_rmse)))
    print(f"  wall time   : {train_seconds:.0f} s")
    print(f"  saved → {run_dir}")


if __name__ == '__main__':
    main()
