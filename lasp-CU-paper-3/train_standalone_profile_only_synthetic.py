"""
train_standalone_profile_only_synthetic.py — train one ProfileOnlyNetworkExtras
model on the synthetic-cloud HDF5, reusing hyperparameters from a winning run
of one of the M0 / M1 / M2 sweeps.

Sibling of train_standalone_profile_only_extras.py. The differences:

  * Source of hyperparameters: the synthetic sweep stores its per-run hyperparameter
    block in `summary.json` (not `config.json`), so the loader points there.
  * Variant-aware extras: the chosen (variant, run_id) determines which of
    tau_c, wv_above_cloud, wv_in_cloud are active by default. Pass
    `--zero-tau-c` etc. to override.
  * Default 80 / 10 / 10 profile-aware split with `--n-val-profiles` /
    `--n-test-profiles` (defaults set for the ~42k-sample dataset).
  * Same outputs and same plot helpers as the existing standalone trainer
    (loss_curves.png + profiles_true_vs_pred.png at 500 DPI).

Usage:
    python train_standalone_profile_only_synthetic.py \\
        --variant M0 --run-id 42 \\
        --h5-path /path/to/synthetic_training_data_42k.h5 \\
        --output-dir ./standalone_results_profile_only_synthetic/M0_run042

    # Override the variant's default extras (e.g. test M0 hyperparameters but
    # WITH τ_c active to compare):
    python train_standalone_profile_only_synthetic.py \\
        --variant M0 --run-id 42 \\
        --h5-path ... \\
        --no-zero-tau-c
        
    # compare the same hyperparameter draw across variants on the new dataset (run the script three times, change --variant):
    for V in M0 M1 M2; do
        python train_standalone_profile_only_synthetic.py \
            --variant $V --run-id 42 \
            --h5-path /scratch/.../synthetic_training_data_42k.h5 \
            --output-dir ./standalone_results_profile_only_synthetic/${V}_run042_42k
    done

Outputs (under --output-dir):
    best_model.pt
    config.json, history.json, results.json
    loss_curves.png (500 DPI)
    profiles_true_vs_pred.png (500 DPI)
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

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from models                     import RetrievalConfig
from models_profile_only        import ProfileOnlyLoss
from models_profile_only_extras import ProfileOnlyNetworkExtras
from data                       import create_dataloaders_extras, resolve_h5_path

# Reuse the canonical training/eval/predict + plot helpers so this script
# stays bit-identical to the existing standalone trainer.
from train_standalone_profile_only_extras import (
    train_one_epoch, validate, predict_test,
)
from train_standalone_profile_only import (
    plot_loss_curves, plot_profiles_true_vs_pred,
)

N_EXTRAS = 3   # tau_c, wv_above_cloud, wv_in_cloud (always present in the dataset)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])

    # Source of hyperparameters
    p.add_argument('--variant', type=str, required=True,
                   choices=['M0', 'M1', 'M2'],
                   help='Which sweep variant to draw hyperparameters from. '
                        'M0 = no extras, M1 = +tau_c, M2 = +tau_c +wv_above.')
    p.add_argument('--run-id', type=int, required=True,
                   help='Run ID inside the chosen variant\'s sweep results. '
                        'E.g. 42 → run_042/summary.json')
    p.add_argument('--sweep-root', type=str,
                   default='hyper_parameter_sweep',
                   help='Repo-relative directory holding the per-variant '
                        'sweep_results_profile_only_synthetic_<variant>/ trees. '
                        '(default: hyper_parameter_sweep)')

    # Data
    p.add_argument('--h5-path', type=str, required=True,
                   help='Synthetic HDF5 to train on (the new ~42k-sample file).')
    p.add_argument('--training-data-dir', type=str, default=None,
                   help='Override the directory portion of --h5-path.')
    p.add_argument('--n-val-profiles', type=int, default=None,
                   help='Profiles held out for validation. Default: 10%% of '
                        'dataset size, profile-aware.')
    p.add_argument('--n-test-profiles', type=int, default=None,
                   help='Profiles held out for test. Default: 10%% of dataset size.')

    # Output / device / overrides
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to save outputs. Default: '
                        './standalone_results_profile_only_synthetic/'
                        '<variant>_run<id>_<h5stem>')
    p.add_argument('--device',   type=str, default=None,
                   choices=['cuda', 'mps', 'cpu'])
    p.add_argument('--n-epochs', type=int, default=None,
                   help='Override n_epochs from the run config.')
    p.add_argument('--seed',     type=int, default=42)

    # Variant-default extras can be overridden
    p.add_argument('--zero-tau-c',    dest='zero_tau_c',    action='store_true',
                   default=None, help='Force tau_c channel to zero.')
    p.add_argument('--no-zero-tau-c', dest='zero_tau_c',    action='store_false',
                   default=None, help='Force tau_c channel to be active.')
    p.add_argument('--zero-wv-above',    dest='zero_wv_above', action='store_true',
                   default=None, help='Force wv_above_cloud to zero.')
    p.add_argument('--no-zero-wv-above', dest='zero_wv_above', action='store_false',
                   default=None, help='Force wv_above_cloud active.')
    p.add_argument('--zero-wv-in',    dest='zero_wv_in',    action='store_true',
                   default=None, help='Force wv_in_cloud to zero.')
    p.add_argument('--no-zero-wv-in', dest='zero_wv_in',    action='store_false',
                   default=None, help='Force wv_in_cloud active.')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Default 10% split helper
# ─────────────────────────────────────────────────────────────────────────────
def _default_split_sizes(h5_path: Path, n_val: int | None, n_test: int | None
                          ) -> tuple[int, int]:
    """If n_val / n_test were not supplied on the CLI, default each to 10% of
    the HDF5 sample count (rounded down). With ~42k samples → 4200/4200."""
    if n_val is not None and n_test is not None:
        return n_val, n_test
    with h5py.File(h5_path, 'r') as f:
        # Any per-sample dataset works; profiles is always present in the schema.
        n_total = f['profiles'].shape[0]
    auto = max(1, n_total // 10)
    return (n_val if n_val is not None else auto,
            n_test if n_test is not None else auto)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Locate the sweep summary for the requested (variant, run_id) ────────
    sweep_dir = (repo_root / args.sweep_root /
                 f'sweep_results_profile_only_synthetic_{args.variant}').resolve()
    run_dir   = sweep_dir / f'run_{args.run_id:03d}'
    sj_path   = run_dir / 'summary.json'
    if not sj_path.exists():
        raise FileNotFoundError(
            f'No sweep summary at {sj_path}. Confirm --variant and --run-id, '
            f'and that the sweep results are present under {sweep_dir}/.')
    with sj_path.open() as f:
        sweep_summary = json.load(f)
    hp = sweep_summary['hyperparams']

    # Variant default for extras (overridden by CLI if given)
    variant_extras = sweep_summary.get('extras', {}) or {}
    zero_tau_c    = variant_extras.get('zero_tau_c',    True)
    zero_wv_above = variant_extras.get('zero_wv_above', True)
    zero_wv_in    = variant_extras.get('zero_wv_in',    True)
    if args.zero_tau_c    is not None: zero_tau_c    = args.zero_tau_c
    if args.zero_wv_above is not None: zero_wv_above = args.zero_wv_above
    if args.zero_wv_in    is not None: zero_wv_in    = args.zero_wv_in

    if args.n_epochs is not None:
        hp['n_epochs'] = args.n_epochs

    # ── HDF5 path resolution ────────────────────────────────────────────────
    h5_path = resolve_h5_path(args.h5_path, args.training_data_dir).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 not found: {h5_path}')

    n_val, n_test = _default_split_sizes(h5_path,
                                         args.n_val_profiles,
                                         args.n_test_profiles)

    # ── Output dir ──────────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (repo_root /
                      'standalone_results_profile_only_synthetic' /
                      f'{args.variant}_run{args.run_id:03d}_{h5_path.stem}'
                      ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
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

    # ── Banner ──────────────────────────────────────────────────────────────
    extras_status = (
        f"tau_c={'ZEROED' if zero_tau_c else 'active'} | "
        f"wv_above={'ZEROED' if zero_wv_above else 'active'} | "
        f"wv_in={'ZEROED' if zero_wv_in else 'active'}"
    )
    print('=' * 70)
    print(f"  STANDALONE TRAIN — synthetic, variant {args.variant}, "
          f"hyperparameters from run_{args.run_id:03d}")
    print('=' * 70)
    print(f"  sweep summary : {sj_path}")
    print(f"  HDF5 file     : {h5_path}")
    print(f"  output dir    : {output_dir}")
    print(f"  device        : {device}")
    print()
    print(f"  Network input layout: 636 reflectance ⊕ 4 geometry "
          f"(sza, vza, saz, vaz) ⊕ 3 extras = 643")
    print(f"  Extras status: {extras_status}")
    print(f"  Split sizes  : n_val={n_val}, n_test={n_test} (profile-aware)")
    print()
    print(f"  Hyperparameters being reused:")
    for k, v in hp.items():
        if k == 'level_weights' and isinstance(v, list) and len(v) > 8:
            print(f"    {k:22s} = [{v[0]:.3f}, {v[1]:.3f}, ..., "
                  f"{v[-2]:.3f}, {v[-1]:.3f}]  (len={len(v)})")
        else:
            print(f"    {k:22s} = {v}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataloaders (single 80/10/10 profile-aware split) ───────────────────
    train_loader, val_loader, test_loader = create_dataloaders_extras(
        h5_path=str(h5_path),
        instrument='hysics',
        batch_size=hp['batch_size'],
        num_workers=4,
        seed=args.seed,
        n_val_profiles=n_val,
        n_test_profiles=n_test,
        zero_tau_c=zero_tau_c,
        zero_wv_above=zero_wv_above,
        zero_wv_in=zero_wv_in,
    )

    base_ds = train_loader.dataset
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset
    n_levels = base_ds.n_levels
    print(f"Profile grid: {n_levels} levels (read from HDF5)")
    if len(hp['level_weights']) != n_levels:
        raise ValueError(
            f"level_weights in run_{args.run_id} has "
            f"{len(hp['level_weights'])} entries but the HDF5 grid has "
            f"{n_levels} levels. Pick a sweep run whose level grid matches "
            f"this HDF5, or rebuild the weights manually.")
    level_weights = torch.tensor(hp['level_weights'], dtype=torch.float32)

    # ── Model + loss + optimizer ────────────────────────────────────────────
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
          f"(input_dim={model_config.input_dim} + n_extras={N_EXTRAS} = "
          f"{model.input_dim})")

    criterion = ProfileOnlyLoss(
        config=model_config,
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=hp.get('scheduler_patience', 30),
                                  min_lr=1e-6)

    # ── Training loop ───────────────────────────────────────────────────────
    n_epochs            = int(hp.get('n_epochs', 1500))
    early_stop_patience = int(hp.get('early_stop_patience', 150))
    warmup_steps        = int(hp.get('warmup_steps', 500))
    aug_noise           = float(hp.get('augment_noise_std', 0.0))
    target_lr           = float(hp['learning_rate'])

    best_val_loss = float('inf')
    best_epoch    = -1
    epochs_no_improve = 0
    global_step   = 0
    history       = {'train': [], 'val': []}

    print(f"\nTraining up to {n_epochs} epochs "
          f"(early-stop patience={early_stop_patience}, "
          f"warmup_steps={warmup_steps}, augment_noise_std={aug_noise})")
    print('-' * 70)

    t0 = time.time()
    final_epoch = 0
    for epoch in range(n_epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_noise_std=aug_noise,
            warmup_steps=warmup_steps,
            target_lr=target_lr,
            global_step_start=global_step,
        )
        val_loss = validate(model, val_loader, criterion, device)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch':            epoch,
                'val_loss':         val_loss,
            }, output_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1

        if global_step >= warmup_steps:
            scheduler.step(val_loss)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:4d} | Train: {train_loss:7.4f} | "
              f"Val: {val_loss:7.4f} | LR: {cur_lr:.1e} | "
              f"No-improve: {epochs_no_improve}")

        final_epoch = epoch
        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(no improvement in {early_stop_patience} epochs)")
            break

    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.0f}s ({final_epoch + 1} epochs)")

    # ── Best-checkpoint test evaluation ─────────────────────────────────────
    ckpt = torch.load(output_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    results = predict_test(model, test_loader, device, model_config)

    rmse_per_level = np.sqrt(np.mean((results['pred'] - results['true']) ** 2,
                                     axis=0))
    mean_rmse     = float(rmse_per_level.mean())
    sigma_overall = float(results['pred_std'].mean())

    print(f"\nTest metrics (best checkpoint from epoch {int(ckpt['epoch'])}):")
    print(f"  Mean RMSE:           {mean_rmse:.3f} μm  (across {n_levels} levels)")
    print(f"  Mean predicted σ:    {sigma_overall:.3f} μm")
    print(f"  RMSE/σ ratio:        {mean_rmse / max(sigma_overall, 1e-9):.2f}  "
          f"(1.0 = calibrated)")
    print(f"  Per-level RMSE:")
    for L, rmse in enumerate(rmse_per_level, 1):
        print(f"    L{L:02d}: {rmse:.3f} μm")

    # ── Save outputs ────────────────────────────────────────────────────────
    cfg_to_save = {
        'source_variant':     args.variant,
        'source_run_id':      args.run_id,
        'source_sweep_dir':   str(sweep_dir),
        'source_summary':     str(sj_path),
        'h5_path':            str(h5_path),
        'n_val_profiles':     n_val,
        'n_test_profiles':    n_test,
        'extras_active': {
            'tau_c':          not zero_tau_c,
            'wv_above_cloud': not zero_wv_above,
            'wv_in_cloud':    not zero_wv_in,
        },
        'hyperparams':        hp,
    }
    with (output_dir / 'config.json').open('w') as f:
        json.dump(cfg_to_save, f, indent=2)
    with (output_dir / 'history.json').open('w') as f:
        json.dump(history, f)
    with (output_dir / 'results.json').open('w') as f:
        json.dump({
            'source_variant':       args.variant,
            'source_run_id':        args.run_id,
            'h5_path':              str(h5_path),
            'n_levels':             int(n_levels),
            'n_params':             n_params,
            'n_extras':             N_EXTRAS,
            'extras_active': {
                'tau_c':          not zero_tau_c,
                'wv_above_cloud': not zero_wv_above,
                'wv_in_cloud':    not zero_wv_in,
            },
            'best_epoch':           int(ckpt['epoch']),
            'final_epoch':          int(final_epoch),
            'train_time_seconds':   float(train_time),
            'best_val_loss':        float(best_val_loss),
            'mean_rmse':            mean_rmse,
            'mean_predicted_sigma': sigma_overall,
            'rmse_sigma_ratio':     mean_rmse / max(sigma_overall, 1e-9),
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
