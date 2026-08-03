"""
compute_feature_importance_profile_tau.py — per-input feature importance for a
PROFILE + τ_c sweep run (sweep_results_profile_tau_synthetic/run_NNN).

Generalization of compute_feature_importance_run004.py to the two-head
`DropletProfileNetwork`: everything is read from the run directory's
summary.json (no hardcoded hyperparameters), and BOTH retrieval targets get
their own importance measures from the same passes:

(1) Permutation importance  [primary]
    For each of the 640 inputs (636 spectral + 4 geometry), shuffle that one
    column across the test samples and re-measure BOTH test metrics:

        Δ profile RMSE  [μm]          — mean-over-levels RMSE, as published
        Δ τ_c RMSE      [τ units]     — physical optical-thickness RMSE

    One shuffled forward pass yields both deltas, so the τ importance is free.
    Averaged over --n-repeats seeded shuffles → mean ± std per input.

(2) Gradient saliency  [secondary; redundancy-robust]
    Mean over test samples of |∂y/∂x_i| for two scalars y:
        mean-over-levels r_e (μm)   and   physical τ_c
    (τ is differentiated through the same linear/log10 inverse transform the
    trainer used, so the saliency is in true optical-thickness units.)

The interesting science: which channels carry τ information (expected:
visible/NIR continuum) vs profile information (expected: 1.5/2.0 μm liquid
absorption + O₂-A) — and whether the two-head model reuses or splits them.

Outputs → <run_dir>/feature_importance/ (override with --out-dir):
    feature_importance_profile_tau.npz   — arrays for the plotting script
    feature_importance_profile_tau.csv   — per-feature table

Run:
    python compute_feature_importance_profile_tau.py \\
        --run-dir hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'hyper_parameter_sweep'))

from models import DropletProfileNetwork, RetrievalConfig            # noqa: E402
from data import (LibRadtranDataset, create_profile_aware_splits,    # noqa: E402
                  TAU_MIN, TAU_MAX)
# Full-path override semantics with a training_data/ fallback (same helper the
# other run-level plot scripts use — NOT data.resolve_h5_path, which treats
# the override as a directory).
from plot_percentile_profiles_synthetic import _resolve_h5_path      # noqa: E402

N_SPECTRAL = 636
N_GEOMETRY = 4
N_INFORMATIVE = N_SPECTRAL + N_GEOMETRY   # 640 — the model's full input
GEOM_NAMES = ['SZA', 'VZA', 'SAZ', 'VAZ']

_LOG_TAU_MIN = float(np.log10(TAU_MIN))
_LOG_TAU_MAX = float(np.log10(TAU_MAX))


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def tau_norm_to_phys_torch(tau_norm: torch.Tensor, tau_transform: str) -> torch.Tensor:
    """Differentiable normalized-τ → physical-τ, matching the trainer."""
    if tau_transform == 'linear':
        return tau_norm * (TAU_MAX - TAU_MIN) + TAU_MIN
    if tau_transform == 'log10':
        return 10.0 ** (tau_norm * (_LOG_TAU_MAX - _LOG_TAU_MIN) + _LOG_TAU_MIN)
    raise ValueError(f'Unknown tau_transform: {tau_transform!r}')


def load_run(run_dir: Path, h5_override: str | None, device: torch.device):
    """Model + config + tau_transform from the run directory."""
    with (run_dir / 'summary.json').open() as f:
        s = json.load(f)
    hp = s['hyperparams']
    tau_transform = s.get('tau_transform', hp.get('tau_transform', 'linear'))
    h5_path = _resolve_h5_path(s['data']['h5_path'], h5_override)

    cfg = RetrievalConfig(
        n_wavelengths=N_SPECTRAL, n_geometry_inputs=N_GEOMETRY,
        n_levels=len(s['per_level_rmse_um']),
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'], activation=hp.get('activation', 'gelu'),
        tau_min=float(TAU_MIN), tau_max=float(TAU_MAX),
    )
    model = DropletProfileNetwork(cfg).to(device)
    ckpt = torch.load(run_dir / 'best_model.pt', map_location=device,
                      weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict)
                                         and 'model_state_dict' in ckpt) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, cfg, s, tau_transform, h5_path


def build_test_inputs(s: dict, h5_path: Path, seed: int):
    """(X[N,640], prof_true_um[N,7], tau_true[N], wavelengths_nm[636])."""
    _, _, test_idx = create_profile_aware_splits(
        str(h5_path),
        n_val_profiles=s['n_val_profiles'],
        n_test_profiles=s['n_test_profiles'],
        seed=seed,
    )
    ds = LibRadtranDataset(str(h5_path), normalize=True, instrument='hysics')
    X, P, T = [], [], []
    for gi in test_idx:
        xi, profi, taui = ds[int(gi)]
        X.append(xi.numpy()); P.append(profi.numpy()); T.append(float(taui))
    X = np.stack(X).astype(np.float32)                       # (N, 640)
    prof_norm = np.stack(P).astype(np.float32)
    re_min, re_max = float(ds.re_min), float(ds.re_max)
    prof_true_um = (prof_norm * (re_max - re_min) + re_min).astype(np.float32)
    # loader τ is linearly normalized regardless of the loss-space transform
    tau_true = (np.asarray(T, dtype=np.float32) * (TAU_MAX - TAU_MIN) + TAU_MIN)
    with h5py.File(h5_path, 'r') as f:
        wavelengths_nm = f['wavelengths'][:].astype(np.float64)
    return X, prof_true_um, tau_true, wavelengths_nm


@torch.no_grad()
def eval_metrics(model, X: np.ndarray, prof_true_um: np.ndarray,
                 tau_true: np.ndarray, tau_transform: str,
                 device: torch.device, batch: int = 4096) -> tuple[float, float]:
    """(mean per-level profile RMSE [μm], τ RMSE [τ units]) in one pass."""
    prof_preds, tau_preds = [], []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out = model(xb)
        prof_preds.append(out['profile'].cpu().numpy())
        tau_norm = out['tau_normalized'].squeeze(-1)
        tau_preds.append(tau_norm_to_phys_torch(tau_norm, tau_transform)
                         .cpu().numpy())
    prof_pred = np.concatenate(prof_preds, axis=0)
    tau_pred  = np.concatenate(tau_preds,  axis=0)
    prof_rmse = float(np.sqrt(((prof_pred - prof_true_um) ** 2)
                              .mean(axis=0)).mean())
    tau_rmse  = float(np.sqrt(((tau_pred - tau_true) ** 2).mean()))
    return prof_rmse, tau_rmse


def permutation_importance(model, X, prof_true_um, tau_true, tau_transform,
                           device, base_prof, base_tau,
                           n_repeats: int, seed: int):
    """Shuffle each input column; Δ of both metrics per repeat.

    Returns (prof_mean, prof_std, tau_mean, tau_std), each shape (640,).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    imp_prof = np.zeros((n_repeats, N_INFORMATIVE), dtype=np.float64)
    imp_tau  = np.zeros((n_repeats, N_INFORMATIVE), dtype=np.float64)
    Xw = X.copy()
    t0 = time.time()
    for rep in range(n_repeats):
        for c in range(N_INFORMATIVE):
            col = Xw[:, c].copy()
            Xw[:, c] = col[rng.permutation(n)]
            pr, tr = eval_metrics(model, Xw, prof_true_um, tau_true,
                                  tau_transform, device)
            imp_prof[rep, c] = pr - base_prof
            imp_tau[rep, c]  = tr - base_tau
            Xw[:, c] = col
        print(f'  permutation repeat {rep + 1}/{n_repeats} '
              f'done ({time.time() - t0:.0f}s elapsed)')
    return (imp_prof.mean(axis=0), imp_prof.std(axis=0),
            imp_tau.mean(axis=0),  imp_tau.std(axis=0))


def gradient_saliency(model, X, tau_transform, device, batch: int = 4096):
    """(saliency_profile[640], saliency_tau[640]) at the real inputs.

    saliency_profile: mean |∂(mean-level r_e μm)/∂x_i|
    saliency_tau    : mean |∂(physical τ_c)/∂x_i|
    """
    acc_prof = np.zeros(X.shape[1], dtype=np.float64)
    acc_tau  = np.zeros(X.shape[1], dtype=np.float64)
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device).requires_grad_(True)
        out = model(xb)
        prof_scalar = out['profile'].mean(dim=1).sum()
        g_prof, = torch.autograd.grad(prof_scalar, xb, retain_graph=True)
        acc_prof += g_prof.abs().sum(dim=0).detach().cpu().numpy()

        tau_phys = tau_norm_to_phys_torch(out['tau_normalized'].squeeze(-1),
                                          tau_transform)
        g_tau, = torch.autograd.grad(tau_phys.sum(), xb)
        acc_tau += g_tau.abs().sum(dim=0).detach().cpu().numpy()
    return acc_prof[:N_INFORMATIVE] / n, acc_tau[:N_INFORMATIVE] / n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--run-dir', required=True,
                   help='PT-sweep run directory containing summary.json + '
                        'best_model.pt.')
    p.add_argument('--out-dir', default=None,
                   help='Output directory (default: <run-dir>/feature_importance).')
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in summary.json.')
    p.add_argument('--n-repeats', type=int, default=20,
                   help='Permutation shuffles per channel (default 20).')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed; also reproduces the test split (must match '
                        'the sweep seed).')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / 'feature_importance'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f'Device: {device}   seed: {args.seed}   n_repeats: {args.n_repeats}')

    model, cfg, s, tau_transform, h5_path = load_run(run_dir, args.h5_path, device)
    print(f'Run {s["run_id"]}  variant {s.get("variant", "?")}  '
          f'tau_transform = {tau_transform}')

    X, prof_true_um, tau_true, wavelengths_nm = build_test_inputs(
        s, h5_path, args.seed)
    print(f'Test inputs: X {X.shape}')

    base_prof, base_tau = eval_metrics(model, X, prof_true_um, tau_true,
                                       tau_transform, device)
    print(f'Baseline profile RMSE: {base_prof:.4f} μm  '
          f'(summary: {s["mean_test_rmse_um"]:.4f})')
    print(f'Baseline τ RMSE      : {base_tau:.4f}     '
          f'(summary: {s.get("tau_test_rmse", float("nan")):.4f})')

    print('Computing permutation importance (both targets per shuffle)...')
    perm_prof_mean, perm_prof_std, perm_tau_mean, perm_tau_std = \
        permutation_importance(model, X, prof_true_um, tau_true, tau_transform,
                               device, base_prof, base_tau,
                               n_repeats=args.n_repeats, seed=args.seed)

    print('Computing gradient saliency (both targets)...')
    sal_prof, sal_tau = gradient_saliency(model, X, tau_transform, device)

    kind = np.array(['spectral'] * N_SPECTRAL + ['geometry'] * N_GEOMETRY)
    feat_wl = np.concatenate([wavelengths_nm, np.full(N_GEOMETRY, np.nan)])
    feat_label = ([f'{w:.1f} nm' for w in wavelengths_nm] + GEOM_NAMES)

    np.savez_compressed(
        out_dir / 'feature_importance_profile_tau.npz',
        perm_profile_mean_um=perm_prof_mean, perm_profile_std_um=perm_prof_std,
        perm_tau_mean=perm_tau_mean, perm_tau_std=perm_tau_std,
        saliency_profile=sal_prof, saliency_tau=sal_tau,
        wavelengths_nm=wavelengths_nm, feat_wl=feat_wl, kind=kind,
        feat_label=np.array(feat_label),
        baseline_profile_rmse_um=base_prof, baseline_tau_rmse=base_tau,
        tau_transform=tau_transform, run_id=s['run_id'],
        n_spectral=N_SPECTRAL, n_geometry=N_GEOMETRY,
        n_repeats=args.n_repeats, seed=args.seed)

    with open(out_dir / 'feature_importance_profile_tau.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['feature_idx', 'kind', 'wavelength_nm', 'label',
                    'perm_profile_mean_um', 'perm_profile_std_um',
                    'perm_tau_mean', 'perm_tau_std',
                    'saliency_profile', 'saliency_tau'])
        for i in range(N_INFORMATIVE):
            w.writerow([i, kind[i],
                        '' if np.isnan(feat_wl[i]) else f'{feat_wl[i]:.2f}',
                        feat_label[i],
                        f'{perm_prof_mean[i]:.6f}', f'{perm_prof_std[i]:.6f}',
                        f'{perm_tau_mean[i]:.6f}',  f'{perm_tau_std[i]:.6f}',
                        f'{sal_prof[i]:.6f}',       f'{sal_tau[i]:.6f}'])

    # ---- diagnostics ---------------------------------------------------
    def top(arr, k=15):
        return np.argsort(-arr)[:k]

    print(f'\nTOP 15 by PROFILE permutation importance (ΔRMSE μm):')
    for r, c in enumerate(top(perm_prof_mean), 1):
        print(f'  {r:>2}  idx {c:>3}  {feat_label[c]:>9}  '
              f'{perm_prof_mean[c]:+.4f} ± {perm_prof_std[c]:.4f}')
    print(f'\nTOP 15 by τ permutation importance (ΔRMSE, τ units):')
    for r, c in enumerate(top(perm_tau_mean), 1):
        print(f'  {r:>2}  idx {c:>3}  {feat_label[c]:>9}  '
              f'{perm_tau_mean[c]:+.4f} ± {perm_tau_std[c]:.4f}')

    spec_p = perm_prof_mean[:N_SPECTRAL]
    spec_t = perm_tau_mean[:N_SPECTRAL]
    if spec_p.std() > 0 and spec_t.std() > 0:
        rho = np.corrcoef(spec_p, spec_t)[0, 1]
        print(f'\ncorr(profile perm, τ perm) over 636 spectral channels: '
              f'r = {rho:+.3f}')
        print('  (low r ⇒ the two heads draw on different spectral regions)')

    print(f'\nWrote {out_dir / "feature_importance_profile_tau.npz"}')
    print(f'Wrote {out_dir / "feature_importance_profile_tau.csv"}')


if __name__ == '__main__':
    main()
