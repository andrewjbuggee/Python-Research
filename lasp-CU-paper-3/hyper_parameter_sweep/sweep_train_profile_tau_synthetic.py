"""
sweep_train_profile_tau_synthetic.py — single-config trainer for the
synthetic-cloud PROFILE + τ_c sweep.

Reads one JSON config (produced by generate_sweep_profile_tau_synthetic.py),
trains `DropletProfileNetwork` (7-level r_e profile + per-level σ, plus cloud
optical thickness τ_c + σ) on a single profile-aware split of the synthetic
HDF5, and writes summary.json with test metrics for both outputs.

Relationship to sweep_train_profile_only_synthetic.py
-----------------------------------------------------
Same CLI, same split machinery (create_profile_aware_splits, seed 42; the
n_val/n_test counts come from the config — the PT sweep uses a true 80/10/10
split, see the generator's Split note), same optimizer / scheduler / warmup /
early-stop mechanics, same summary.json key names for the profile metrics —
so `compare_sweep_profile_only_synthetic.py` reads this sweep's results
unchanged, including in cross-variant mode against
sweep_results_profile_only_synthetic_M0_rev2.

Three differences:
  1. `DropletProfileNetwork` + `CombinedLoss` (τ head + τ NLL) in place of
     `ProfileOnlyNetworkExtras` + `ProfileOnlyLoss`.
  2. `create_dataloaders(profile_holdout=True)` (640 inputs) in place of
     `create_dataloaders_extras` (643).  The M0 variant zeroed all three extra
     channels, so the two are information-equivalent; dropping them removes
     three dead input weights and the stale VOCALS-era z-score constants.
  3. τ normalization is explicit — see the τ SCALING note below.

τ SCALING
---------
`LibRadtranDataset` normalizes τ with data.TAU_MIN / data.TAU_MAX = 3 / 97,
while the stock `RetrievalConfig` denormalizes with tau_min / tau_max =
3 / 75.  Those disagree, so `output['tau_c']` from a network built on the
default config is mis-scaled by (97-3)/(75-3) = 1.31.  This trainer builds
RetrievalConfig with the dataset's bounds so the two agree.

With `tau_transform='log10'` the τ target is re-mapped to
    tau_norm = (log10(τ) - log10(TAU_MIN)) / (log10(TAU_MAX) - log10(TAU_MIN))
before the NLL, and physical τ is recovered by inverting that here rather than
reading `output['tau_c']` (which the network computes with the linear map).
τ in this dataset is right-skewed — median 8.0 against a 3–95 range — so the
linear map leaves ~95% of samples inside the bottom fifth of the τ head's
sigmoid.  Both options are swept; see the generator docstring.

Usage:
    python sweep_train_profile_tau_synthetic.py \
        --config-json sweep_configs_profile_tau_synthetic/run_007.json \
        --training-data-dir /scratch/alpine/anbu8374/neural_network_training_data/ \
        --output-dir sweep_results_profile_tau_synthetic

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models import DropletProfileNetwork, CombinedLoss, RetrievalConfig
from data import create_dataloaders, resolve_h5_path, TAU_MIN, TAU_MAX


# log10 τ bounds, used only when tau_transform == 'log10'
_LOG_TAU_MIN = float(np.log10(TAU_MIN))
_LOG_TAU_MAX = float(np.log10(TAU_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Synthetic-cloud profile+τ single-config trainer "
                    "(one profile-aware split per run).")
    p.add_argument('--config-json',       type=str, required=True,
                   help='Path to JSON config produced by '
                        'generate_sweep_profile_tau_synthetic.py')
    p.add_argument('--training-data-dir', type=str, default=None,
                   help='Override directory portion of cfg["data"]["h5_path"] '
                        '(filename preserved).')
    p.add_argument('--h5-path',           type=str, default=None,
                   help='Full override of cfg["data"]["h5_path"].')
    p.add_argument('--output-dir',        type=str, default=None,
                   help='Override cfg["output_dir"]; results are written under '
                        '<output_dir>/run_<id>/.')
    p.add_argument('--device',            type=str, default=None,
                   choices=['cuda', 'mps', 'cpu'])
    p.add_argument('--seed',              type=int, default=42,
                   help='Seeds torch/numpy AND the profile-aware split. At the '
                        'default 42 the M0_rev2 889-profile test set is a '
                        'strict subset of this sweep\'s 4,199-profile test set.')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# τ target / inverse transforms
#
# The dataloader always hands back τ linearly normalized as
# (τ_um_free - TAU_MIN) / (TAU_MAX - TAU_MIN).  These two helpers convert that
# to whatever space the loss is being computed in, and back to physical τ.
# ─────────────────────────────────────────────────────────────────────────────
def tau_norm_to_target(tau_norm_linear: torch.Tensor,
                       tau_transform: str) -> torch.Tensor:
    """Map the loader's linear τ normalization into the loss's τ space."""
    if tau_transform == 'linear':
        return tau_norm_linear
    if tau_transform == 'log10':
        tau_phys = tau_norm_linear * (TAU_MAX - TAU_MIN) + TAU_MIN
        tau_phys = tau_phys.clamp(min=TAU_MIN)      # guard float round-trip
        return ((torch.log10(tau_phys) - _LOG_TAU_MIN)
                / (_LOG_TAU_MAX - _LOG_TAU_MIN))
    raise ValueError(f"Unknown tau_transform: {tau_transform!r}")


def tau_target_to_physical(tau_norm: np.ndarray, tau_transform: str) -> np.ndarray:
    """Inverse of tau_norm_to_target, in numpy, returning optical thickness."""
    if tau_transform == 'linear':
        return tau_norm * (TAU_MAX - TAU_MIN) + TAU_MIN
    if tau_transform == 'log10':
        return 10.0 ** (tau_norm * (_LOG_TAU_MAX - _LOG_TAU_MIN) + _LOG_TAU_MIN)
    raise ValueError(f"Unknown tau_transform: {tau_transform!r}")


def tau_sigma_to_physical(tau_norm: np.ndarray, tau_sigma_norm: np.ndarray,
                          tau_transform: str) -> np.ndarray:
    """
    Propagate the τ head's normalized σ into optical-thickness units.

    'linear' is just a scale factor.  'log10' is nonlinear, so use the local
    derivative dτ/d(tau_norm) = τ · ln(10) · (log10 τ_max − log10 τ_min),
    evaluated at the predicted τ.  That is a first-order approximation and is
    only meaningful while σ is small relative to the τ range — which it is
    here (σ floor 0.01 normalized, τ predictions O(10)).
    """
    if tau_transform == 'linear':
        return tau_sigma_norm * (TAU_MAX - TAU_MIN)
    if tau_transform == 'log10':
        tau_phys = tau_target_to_physical(tau_norm, 'log10')
        dtau_dnorm = tau_phys * np.log(10.0) * (_LOG_TAU_MAX - _LOG_TAU_MIN)
        return tau_sigma_norm * dtau_dnorm
    raise ValueError(f"Unknown tau_transform: {tau_transform!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
#
# Mirrors sweep_train.py / train_standalone_profile_only_extras.py: per-step
# linear LR warmup, fractional Gaussian input noise applied to the whole input
# vector (kept identical to the profile-only trainer so the paired comparison
# is not confounded), grad-norm clip at 1.0.
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    tau_transform: str,
                    augment_noise_std: float = 0.0,
                    warmup_steps: int = 0,
                    target_lr: float = None,
                    global_step_start: int = 0) -> Tuple[float, int]:
    model.train()
    loss_sum, n = 0.0, 0
    step = global_step_start
    for x, prof, tau in loader:
        x, prof, tau = x.to(device), prof.to(device), tau.to(device)
        tau = tau_norm_to_target(tau, tau_transform)

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
def validate(model, loader, criterion, device,
             tau_transform: str) -> Tuple[float, Dict[str, float]]:
    """Returns (mean total loss, mean of every individual loss term)."""
    model.eval()
    loss_sum, n = 0.0, 0
    detail: Dict[str, float] = {}
    for x, prof, tau in loader:
        x, prof, tau = x.to(device), prof.to(device), tau.to(device)
        tau = tau_norm_to_target(tau, tau_transform)
        output = model(x)
        losses = criterion(output, prof, tau)
        loss_sum += losses['total'].item()
        for k, v in losses.items():
            detail[k] = detail.get(k, 0.0) + float(v.item())
        n += 1
    return loss_sum / n, {k: v / n for k, v in detail.items()}


@torch.no_grad()
def predict_test(model, loader, device, config: RetrievalConfig,
                 tau_transform: str) -> Dict[str, np.ndarray]:
    """Test-set predictions in physical units (r_e in μm, τ dimensionless)."""
    model.eval()
    prof_pred, prof_std, prof_true = [], [], []
    tau_pred, tau_std, tau_true = [], [], []
    re_min, re_max = config.re_min, config.re_max

    for x, prof, tau in loader:
        out = model(x.to(device))

        prof_pred.append(out['profile'].cpu().numpy())
        prof_std.append(out['profile_std'].cpu().numpy())
        prof_true.append(prof.numpy() * (re_max - re_min) + re_min)

        # Work from the normalized τ head, not out['tau_c'], so the same code
        # path handles both transforms (see the τ SCALING note in the header).
        tn = out['tau_normalized'].squeeze(-1).cpu().numpy()
        sn = out['tau_std_normalized'].squeeze(-1).cpu().numpy()
        tau_pred.append(tau_target_to_physical(tn, tau_transform))
        tau_std.append(tau_sigma_to_physical(tn, sn, tau_transform))
        # `tau` from the loader is always linearly normalized
        tau_true.append(tau.numpy() * (TAU_MAX - TAU_MIN) + TAU_MIN)

    return {
        'pred':     np.concatenate(prof_pred),
        'pred_std': np.concatenate(prof_std),
        'true':     np.concatenate(prof_true),
        'tau_pred':     np.concatenate(tau_pred),
        'tau_pred_std': np.concatenate(tau_std),
        'tau_true':     np.concatenate(tau_true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg_path = Path(args.config_json).resolve()
    cfg      = json.loads(cfg_path.read_text())

    # ── HDF5 path resolution ─────────────────────────────────────────────
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
                       'sweep_results_profile_tau_synthetic'))
    if not output_root.is_absolute():
        output_root = (cfg_path.parent / output_root).resolve()
    run_dir = output_root / f"run_{cfg['run_id']:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    hp            = cfg['hyperparams']
    variant       = cfg.get('variant', 'PT')
    tau_weight    = float(hp.get('tau_weight', 1.0))
    tau_transform = hp.get('tau_transform', 'linear')
    if tau_transform not in ('linear', 'log10'):
        raise ValueError(f"tau_transform must be 'linear' or 'log10', "
                         f"got {tau_transform!r}")
    early_stop_metric = hp.get('early_stop_metric', 'total')
    if early_stop_metric not in ('total', 'profile'):
        raise ValueError(f"early_stop_metric must be 'total' or 'profile', "
                         f"got {early_stop_metric!r}")

    print("=" * 70)
    print(f"  SYNTHETIC PROFILE+τ SWEEP — variant {variant}, run {cfg['run_id']}")
    print("=" * 70)
    print(f"  config        : {cfg_path}")
    print(f"  HDF5          : {h5_path}")
    print(f"  output dir    : {run_dir}")
    print(f"  device        : {device}")
    print(f"  hidden_dims   : {hp['hidden_dims']}")
    print(f"  activation    : {hp['activation']}")
    print(f"  lr / dropout  : {hp['learning_rate']:.2e} / {hp['dropout']:.3f}")
    print(f"  batch_size    : {hp['batch_size']}")
    print(f"  level_weights : {hp['level_weights_name']} (len={len(hp['level_weights'])})")
    print(f"  tau_weight    : {tau_weight:.4f}")
    print(f"  tau_transform : {tau_transform}  (τ bounds {TAU_MIN}–{TAU_MAX})")
    print(f"  early stop on : val {early_stop_metric}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataloaders (profile-aware split, identical to the M0 sweeps) ────
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=str(h5_path),
        instrument=cfg['data'].get('instrument', 'hysics'),
        batch_size=hp['batch_size'],
        num_workers=cfg['data'].get('num_workers', 4),
        seed=args.seed,
        profile_holdout=True,
        n_val_profiles=cfg['n_val_profiles'],
        n_test_profiles=cfg['n_test_profiles'],
    )
    print(f"Data: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

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

    # ── Model ────────────────────────────────────────────────────────────
    # tau_min/tau_max pinned to the dataset's normalization constants so that
    # output['tau_c'] and the target live on the same scale.
    model_cfg = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
        tau_min=float(TAU_MIN),
        tau_max=float(TAU_MAX),
    )
    sample    = base_ds[0]
    input_dim = int(sample[0].shape[-1])
    if input_dim != model_cfg.input_dim:
        raise RuntimeError(
            f"Dataset x has {input_dim} features but the model expects "
            f"{model_cfg.input_dim}.")
    model = DropletProfileNetwork(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    criterion = CombinedLoss(
        config=model_cfg,
        lambda_physics=hp.get('lambda_physics',      0.1),
        lambda_monotonicity=hp.get('lambda_monotonicity', 0.0),
        lambda_adiabatic=hp.get('lambda_adiabatic',  0.1),
        lambda_smoothness=hp.get('lambda_smoothness', 0.1),
        lambda_emulator_data=0.0,
        level_weights=level_weights,
        sigma_floor=hp.get('sigma_floor', 0.01),
        tau_weight=tau_weight,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=hp['learning_rate'],
                            weight_decay=hp.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=hp.get('scheduler_patience', 30),
                                  factor=0.5)

    # ── Training loop ────────────────────────────────────────────────────
    n_epochs   = int(hp['n_epochs'])
    early_stop = int(hp.get('early_stop_patience', 150))
    warmup     = int(hp.get('warmup_steps', 500))
    aug_noise  = float(hp.get('augment_noise_std', 0.0))
    target_lr  = float(hp['learning_rate'])

    best_val    = float('inf')      # value of the early-stopping metric
    best_total  = float('inf')      # val total loss at the selected epoch
    best_detail: Dict[str, float] = {}
    best_epoch  = -1
    best_state  = None
    no_improve  = 0
    history     = {'train': [], 'val': [], 'val_profile': [], 'val_tau': []}
    global_step = 0

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            tau_transform=tau_transform,
            augment_noise_std=aug_noise,
            warmup_steps=warmup,
            target_lr=target_lr,
            global_step_start=global_step,
        )
        val_loss, val_detail = validate(model, val_loader, criterion, device,
                                        tau_transform)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['val_profile'].append(val_detail.get('supervised_profile', float('nan')))
        history['val_tau'].append(val_detail.get('supervised_tau', float('nan')))

        # Checkpoint selection.  'total' matches the previous τ model; 'profile'
        # makes the selected epoch independent of tau_weight, which keeps
        # checkpoint choice comparable across configs.
        monitored = (val_loss if early_stop_metric == 'total'
                     else val_detail['supervised_profile'])

        if monitored < best_val - 1e-6:
            best_val    = monitored
            best_total  = val_loss
            best_detail = dict(val_detail)
            best_epoch  = epoch
            best_state  = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1

        # The plateau scheduler always tracks the total loss; during warmup the
        # LR is driven manually by train_one_epoch, so hold the scheduler off.
        if global_step >= warmup:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch <= 5:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | Train {train_loss:7.4f} | "
                  f"Val {val_loss:7.4f} "
                  f"(prof {val_detail.get('supervised_profile', float('nan')):7.4f} | "
                  f"tau {val_detail.get('supervised_tau', float('nan')):7.4f}) | "
                  f"LR {cur_lr:.1e} | NoImp {no_improve}")

        if no_improve >= early_stop:
            print(f"  Early stop at epoch {epoch} (no improvement in {early_stop})")
            break

    train_seconds = time.time() - t0

    # ── Test on the selected checkpoint ──────────────────────────────────
    model.load_state_dict(best_state)
    pred = predict_test(model, test_loader, device, model_cfg, tau_transform)

    err            = pred['pred'] - pred['true']
    per_level_rmse = np.sqrt(np.mean(err ** 2, axis=0)).tolist()
    mean_rmse      = float(np.mean(per_level_rmse))
    mean_sigma     = float(np.mean(pred['pred_std']))
    rmse_sigma     = mean_rmse / max(mean_sigma, 1e-9)

    tau_err        = pred['tau_pred'] - pred['tau_true']
    tau_rmse       = float(np.sqrt(np.mean(tau_err ** 2)))
    tau_bias       = float(np.mean(tau_err))
    tau_mape       = float(np.mean(np.abs(tau_err) / np.maximum(pred['tau_true'], 1e-9)) * 100.0)
    tau_var        = float(np.var(pred['tau_true']))
    tau_r2         = float(1.0 - np.mean(tau_err ** 2) / tau_var) if tau_var > 0 else float('nan')
    tau_sigma_mean = float(np.mean(pred['tau_pred_std']))
    tau_rmse_sigma = tau_rmse / max(tau_sigma_mean, 1e-9)

    summary = {
        'run_id':           cfg['run_id'],
        'tag':              cfg.get('tag', ''),
        'variant':          variant,
        'best_epoch':       best_epoch,
        'best_val_loss':    best_total,
        'best_val_metric':  best_val,
        'early_stop_metric': early_stop_metric,
        'best_val_profile_nll': best_detail.get('supervised_profile'),
        'best_val_tau_nll':     best_detail.get('supervised_tau'),
        'epochs_trained':   len(history['train']),
        'train_seconds':    train_seconds,
        'n_params':         int(n_params),
        # Profile metrics — same key names as the profile-only sweep so
        # compare_sweep_profile_only_synthetic.py reads this directory as-is.
        'mean_test_rmse_um':  mean_rmse,
        'mean_test_sigma_um': mean_sigma,
        'rmse_sigma_ratio':   rmse_sigma,
        'per_level_rmse_um':  per_level_rmse,
        # τ metrics (dimensionless optical thickness)
        'tau_transform':      tau_transform,
        'tau_weight':         tau_weight,
        'tau_test_rmse':      tau_rmse,
        'tau_test_bias':      tau_bias,
        'tau_test_mape_pct':  tau_mape,
        'tau_test_r2':        tau_r2,
        'tau_test_sigma':     tau_sigma_mean,
        'tau_rmse_sigma_ratio': tau_rmse_sigma,
        'hyperparams':        hp,
        'data':               cfg['data'],
        'n_val_profiles':     cfg['n_val_profiles'],
        'n_test_profiles':    cfg['n_test_profiles'],
        'seed':               args.seed,
    }

    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save(best_state, run_dir / 'best_model.pt')

    print(f"\n  best epoch  : {best_epoch}  (val total {best_total:.4f})")
    print(f"  mean RMSE   : {mean_rmse:.3f} μm")
    print(f"  mean σ      : {mean_sigma:.3f} μm  (RMSE/σ = {rmse_sigma:.2f})")
    print(f"  per-level   : "
          + ' '.join(f"L{l+1:02d}={r:.2f}" for l, r in enumerate(per_level_rmse)))
    print(f"  τ RMSE      : {tau_rmse:.3f}  (bias {tau_bias:+.3f}, "
          f"MAPE {tau_mape:.1f}%, R² {tau_r2:.3f})")
    print(f"  τ σ         : {tau_sigma_mean:.3f}  (RMSE/σ = {tau_rmse_sigma:.2f})")
    print(f"  wall time   : {train_seconds:.0f} s")
    print(f"  saved → {run_dir}")


if __name__ == '__main__':
    main()
