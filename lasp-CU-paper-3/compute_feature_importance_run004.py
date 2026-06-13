"""
compute_feature_importance_run004.py — Per-input feature importance for the
published paper-3 network (hyperparameter-sweep run_004, variant M0).

Target model
------------
    hyper_parameter_sweep/sweep_results_profile_only_synthetic_M0_rev2/
        run_004/best_model.pt

This is the rank-1-of-100 sweep run (mean test RMSE = 0.9641 um) used to make
every model-output figure in published_figures/.  It is a
`ProfileOnlyNetworkExtras` (n_extras=3) trained in the M0 ablation, i.e. all
three extra scalar inputs (tau_c, wv_above_cloud, wv_in_cloud) are pinned to
0.0.  The first Linear is therefore 643-wide but only the first 640 inputs
carry information:

    idx   0..635 : min-max-normalized linear HySICS reflectance (636 channels,
                   351.95-2297.05 nm)
    idx 636..639 : min-max-normalized geometry [SZA, VZA, SAZ, VAZ]
    idx 640..642 : extras, pinned to 0.0 in M0  (excluded from importance)

Two importance measures are computed on the published test split (the 889
profiles stored in run_004/pred_cache.npz['h5_idx'], seed 42):

(1) Permutation importance  [primary; the "standard" feature-importance bars]
    For each of the 640 informative inputs, shuffle that one column across the
    test samples (destroying its covariance with the target while preserving
    its marginal distribution) and re-measure the model's mean per-level RMSE.

        importance_perm[i] = mean_over_repeats( RMSE(col i shuffled) ) - RMSE_baseline   [um]

    Averaged over N_REPEATS independent shuffles (seeded) -> mean +/- std.
    Higher Delta-RMSE => the model relied more on that input.

(2) Gradient saliency  [secondary; robust to channel redundancy]
    Mean over test samples of | d(mean-over-levels r_e) / d x_i |, evaluated at
    the real inputs.  This is the local Jacobian magnitude of the retrieval
    w.r.t. each input and is the closest NN analogue to a per-channel
    information-content / Jacobian measure (e.g. King & Vaughan 2012): unlike
    single-channel permutation, it is NOT suppressed when neighbouring spectral
    channels are mutually redundant.

Outputs (written next to this script, in feature_importance_run004/)
    feature_importance_run004.npz   — all arrays for the plotting script
    feature_importance_run004.csv   — human-readable per-feature table

Run:
    /opt/anaconda3/bin/python3 compute_feature_importance_run004.py
    (the repo .venv has no torch; /opt/anaconda3 python has torch 2.11 + MPS)

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from models import RetrievalConfig                              # noqa: E402
from models_profile_only_extras import ProfileOnlyNetworkExtras  # noqa: E402
from data import LibRadtranDatasetExtras                        # noqa: E402

RUN_DIR = (REPO / 'hyper_parameter_sweep'
           / 'sweep_results_profile_only_synthetic_M0_rev2' / 'run_004')
H5_PATH = REPO / 'training_data' / 'synthetic_training_data_7-levels_8_May_2026.h5'
OUT_DIR = REPO / 'feature_importance_run004'

N_SPECTRAL = 636          # reflectance channels (indices 0..635)
N_GEOMETRY = 4            # [SZA, VZA, SAZ, VAZ]  (indices 636..639)
N_INFORMATIVE = N_SPECTRAL + N_GEOMETRY   # 640 (extras 640..642 are zeroed)
GEOM_NAMES = ['SZA', 'VZA', 'SAZ', 'VAZ']

# run_004 hyperparameters (from summary.json — verified to reproduce 0.9641 um)
HIDDEN_DIMS = (512, 512, 256, 256)
DROPOUT = 0.2637
ACTIVATION = 'gelu'
N_LEVELS = 7


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(device: torch.device) -> ProfileOnlyNetworkExtras:
    cfg = RetrievalConfig(n_wavelengths=N_SPECTRAL, n_geometry_inputs=N_GEOMETRY,
                          n_levels=N_LEVELS, hidden_dims=HIDDEN_DIMS,
                          dropout=DROPOUT, activation=ACTIVATION)
    model = ProfileOnlyNetworkExtras(cfg, n_extras=3).to(device)
    ckpt = torch.load(RUN_DIR / 'best_model.pt', map_location=device,
                      weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict)
                                         and 'model_state_dict' in ckpt) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def build_test_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[889,643] float32, true_phys[889,7] um, wavelengths_nm[636]).

    Uses the exact published test indices from pred_cache.npz so the baseline
    matches the published 0.9641 um.
    """
    cache = np.load(RUN_DIR / 'pred_cache.npz')
    h5_idx = cache['h5_idx']
    ds = LibRadtranDatasetExtras(str(H5_PATH), normalize=True, instrument='hysics',
                                 zero_tau_c=True, zero_wv_above=True, zero_wv_in=True)
    X, Y = [], []
    for gi in h5_idx:
        xi, profi, _ = ds[int(gi)]
        X.append(xi.numpy())
        Y.append(profi.numpy())
    X = np.stack(X).astype(np.float32)                       # (889, 643)
    prof_norm = np.stack(Y).astype(np.float32)               # (889, 7) in [0,1]
    re_min, re_max = float(ds.re_min), float(ds.re_max)
    true_phys = (prof_norm * (re_max - re_min) + re_min).astype(np.float32)
    with h5py.File(H5_PATH, 'r') as f:
        wavelengths_nm = f['wavelengths'][:].astype(np.float64)
    return X, true_phys, wavelengths_nm


@torch.no_grad()
def mean_per_level_rmse(model, X: np.ndarray, true_phys: np.ndarray,
                        device: torch.device, batch: int = 4096) -> float:
    """Mean over levels of sqrt(mean_samples (pred-true)^2)  — the published metric."""
    preds = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        preds.append(model(xb)['profile'].cpu().numpy())
    pred = np.concatenate(preds, axis=0)                     # (N, 7)
    per_level = np.sqrt(((pred - true_phys) ** 2).mean(axis=0))
    return float(per_level.mean())


def permutation_importance(model, X, true_phys, device, baseline,
                           n_repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle-based permutation importance for the first N_INFORMATIVE inputs.

    Returns (mean[640], std[640]) of Delta-RMSE (um) across repeats.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    imp = np.zeros((n_repeats, N_INFORMATIVE), dtype=np.float64)
    Xw = X.copy()  # working copy we perturb in place, one column at a time
    t0 = time.time()
    for rep in range(n_repeats):
        for c in range(N_INFORMATIVE):
            col = Xw[:, c].copy()
            Xw[:, c] = col[rng.permutation(n)]
            imp[rep, c] = mean_per_level_rmse(model, Xw, true_phys, device) - baseline
            Xw[:, c] = col
        print(f'  permutation repeat {rep + 1}/{n_repeats} '
              f'done ({time.time() - t0:.0f}s elapsed)')
    return imp.mean(axis=0), imp.std(axis=0)


def gradient_saliency(model, X, device, batch: int = 4096) -> np.ndarray:
    """Mean over samples of |d(mean-level r_e)/d x_i| for the first N_INFORMATIVE inputs.

    Computed at the real inputs.  Returns array[640].
    """
    acc = np.zeros(X.shape[1], dtype=np.float64)
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device).requires_grad_(True)
        out = model(xb)['profile']                # (b, 7) in um
        scalar = out.mean(dim=1).sum()            # d/dx_{n,c} = d(mean_L r_e_n)/dx_{n,c}
        grad, = torch.autograd.grad(scalar, xb)
        acc += grad.abs().sum(dim=0).detach().cpu().numpy()
    saliency = acc / n
    return saliency[:N_INFORMATIVE]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--n-repeats', type=int, default=20,
                   help='Permutation shuffles per channel (default 20).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    device = pick_device(args.device)
    print(f'Device: {device}   seed: {args.seed}   n_repeats: {args.n_repeats}')

    model = load_model(device)
    X, true_phys, wavelengths_nm = build_test_inputs()
    print(f'Test inputs: X {X.shape}, extras col sums (expect 0): '
          f'{X[:, 640:].sum(axis=0)}')

    baseline = mean_per_level_rmse(model, X, true_phys, device)
    print(f'Baseline mean RMSE: {baseline:.4f} um  (published: 0.9641 um)')

    print('Computing permutation importance...')
    imp_mean, imp_std = permutation_importance(
        model, X, true_phys, device, baseline,
        n_repeats=args.n_repeats, seed=args.seed)

    print('Computing gradient saliency...')
    saliency = gradient_saliency(model, X, device)

    # feature bookkeeping: kind + wavelength (NaN for geometry)
    kind = np.array(['spectral'] * N_SPECTRAL + ['geometry'] * N_GEOMETRY)
    feat_wl = np.concatenate([wavelengths_nm, np.full(N_GEOMETRY, np.nan)])
    feat_label = ([f'{w:.1f} nm' for w in wavelengths_nm] + GEOM_NAMES)

    np.savez_compressed(
        OUT_DIR / 'feature_importance_run004.npz',
        perm_mean_um=imp_mean, perm_std_um=imp_std, saliency=saliency,
        wavelengths_nm=wavelengths_nm, feat_wl=feat_wl, kind=kind,
        feat_label=np.array(feat_label), baseline_rmse_um=baseline,
        n_spectral=N_SPECTRAL, n_geometry=N_GEOMETRY,
        n_repeats=args.n_repeats, seed=args.seed)

    # CSV
    with open(OUT_DIR / 'feature_importance_run004.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['feature_idx', 'kind', 'wavelength_nm', 'label',
                    'perm_importance_mean_um', 'perm_importance_std_um',
                    'gradient_saliency'])
        for i in range(N_INFORMATIVE):
            w.writerow([i, kind[i],
                        '' if np.isnan(feat_wl[i]) else f'{feat_wl[i]:.2f}',
                        feat_label[i], f'{imp_mean[i]:.6f}',
                        f'{imp_std[i]:.6f}', f'{saliency[i]:.6f}'])

    # ---- diagnostics -----------------------------------------------------
    def top(arr, k=15):
        return np.argsort(-arr)[:k]

    print(f'\nBaseline RMSE: {baseline:.4f} um')
    print('\nTOP 15 by PERMUTATION importance (Delta-RMSE um):')
    for r, c in enumerate(top(imp_mean), 1):
        print(f'  {r:>2}  idx {c:>3}  {feat_label[c]:>9}  '
              f'{imp_mean[c]:+.4f} +/- {imp_std[c]:.4f}')
    print('\nTOP 15 by GRADIENT saliency:')
    for r, c in enumerate(top(saliency), 1):
        print(f'  {r:>2}  idx {c:>3}  {feat_label[c]:>9}  {saliency[c]:.4f}')

    # geometry ranking within all 640
    order = np.argsort(-imp_mean)
    rank_of = {c: r for r, c in enumerate(order)}
    print('\nGeometry inputs — permutation rank out of 640:')
    for gi, name in enumerate(GEOM_NAMES):
        c = N_SPECTRAL + gi
        print(f'  {name}: rank {rank_of[c] + 1:>3}  '
              f'(Delta-RMSE {imp_mean[c]:+.4f} um)')

    # how concentrated is the spectral importance? (redundancy check)
    spec = imp_mean[:N_SPECTRAL]
    print(f'\nSpectral permutation importance: '
          f'min {spec.min():+.4f}, max {spec.max():+.4f}, '
          f'median {np.median(spec):+.4f} um')
    print(f'  fraction of spectral channels with Delta-RMSE > 0.005 um: '
          f'{(spec > 0.005).mean():.2%}')
    # Pearson corr between the two measures over spectral channels
    sal = saliency[:N_SPECTRAL]
    if spec.std() > 0 and sal.std() > 0:
        rho = np.corrcoef(spec, sal)[0, 1]
        print(f'  corr(perm, saliency) over 636 spectral channels: r = {rho:+.3f}')

    print(f'\nWrote {OUT_DIR / "feature_importance_run004.npz"}')
    print(f'Wrote {OUT_DIR / "feature_importance_run004.csv"}')


if __name__ == '__main__':
    main()
