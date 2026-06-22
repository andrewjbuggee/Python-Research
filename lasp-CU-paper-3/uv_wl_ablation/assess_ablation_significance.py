"""
assess_ablation_significance.py — Is the +0.02 μm RMSE gap between the control
(all 636 channels) and the UV-ablation (<500 nm removed) "in the noise"?

There are three distinct notions of noise; this script quantifies all three on
the SAME 889 test profiles (a paired comparison — both models see identical
inputs, so we compare per-profile differences, not the raw RMSE spread):

  (1) test-set sampling   → paired SE / t-test / Wilcoxon / bootstrap CI on Δ
  (2) practical magnitude → Δ vs the per-profile RMSE spread, and Cohen's d_z
  (3) training/seed noise → NOT estimable here (training is deterministic at a
                             fixed seed; control reproduced run_004 bit-for-bit)
                             — would need retrains at several seeds.

Run:
    /opt/anaconda3/bin/python3 assess_ablation_significance.py

Author: Andrew J. Buggee, LASP / CU Boulder
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE
while not (REPO / 'models.py').exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
from models import RetrievalConfig                                   # noqa: E402
from models_profile_only_extras import ProfileOnlyNetworkExtras      # noqa: E402
from compute_feature_importance_run004 import (                      # noqa: E402
    pick_device, build_test_inputs)

RESULTS = HERE / 'standalone_results_uv_ablation'
RUN_004_CACHE = (REPO / 'hyper_parameter_sweep'
                 / 'sweep_results_profile_only_synthetic_M0_rev2'
                 / 'run_004' / 'pred_cache.npz')


@torch.no_grad()
def infer(model_dir: Path, X_full: np.ndarray, device) -> np.ndarray:
    """Return per-test-profile predictions (889, 7) in μm for a saved model."""
    ckpt = torch.load(model_dir / 'best_model.pt', map_location=device,
                      weights_only=False)
    keep_idx = np.array(ckpt['keep_idx'], dtype=np.int64)
    mc = ckpt['model_config']
    model = ProfileOnlyNetworkExtras(
        RetrievalConfig(n_wavelengths=mc['n_wavelengths'], n_geometry_inputs=4,
                        n_levels=mc['n_levels'], hidden_dims=tuple(mc['hidden_dims']),
                        dropout=mc['dropout'], activation=mc['activation']),
        n_extras=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    X = torch.from_numpy(X_full[:, keep_idx].astype(np.float32)).to(device)
    return model(X)['profile'].cpu().numpy()


def level_agg_rmse(pred, true):
    """summary.json's mean_test_rmse_um: mean over levels of sqrt(mean over profiles)."""
    return float(np.sqrt(((pred - true) ** 2).mean(axis=0)).mean())


def per_profile_rmse(pred, true):
    """One RMSE per profile: sqrt(mean over levels). Returns (n_profiles,)."""
    return np.sqrt(((pred - true) ** 2).mean(axis=1))


def main():
    device = pick_device(None)
    X_full, true, _ = build_test_inputs()                     # (889,643), (889,7)

    pred_ctrl = infer(RESULTS / 'control_all636', X_full, device)
    pred_abl = infer(RESULTS / 'ablation_cutoff500nm_keep587', X_full, device)

    # sanity: control should reproduce run_004's cached predictions
    cache = np.load(RUN_004_CACHE)
    print(f'control vs run_004 pred_cache  max|Δpred| = '
          f'{np.abs(pred_ctrl - cache["pred"]).max():.2e}  '
          f'(≈0 ⇒ control == run_004, training is deterministic)\n')

    # ---- the two aggregations of "mean RMSE" -------------------------------
    L_ctrl, L_abl = level_agg_rmse(pred_ctrl, true), level_agg_rmse(pred_abl, true)
    r_ctrl, r_abl = per_profile_rmse(pred_ctrl, true), per_profile_rmse(pred_abl, true)
    P_ctrl, P_abl = r_ctrl.mean(), r_abl.mean()
    n = len(r_ctrl)

    print('Mean test RMSE (μm):')
    print(f'  level-aggregated (summary.json metric):  control {L_ctrl:.4f}  '
          f'ablation {L_abl:.4f}  Δ = {L_abl - L_ctrl:+.4f}')
    print(f'  per-profile mean:                        control {P_ctrl:.4f}  '
          f'ablation {P_abl:.4f}  Δ = {P_abl - P_ctrl:+.4f}')

    # ---- (1) test-set sampling: PAIRED stats on per-profile differences -----
    d = r_abl - r_ctrl                                         # paired diffs
    sd, se = d.std(ddof=1), d.std(ddof=1) / np.sqrt(n)
    t_stat, t_p = ttest_rel(r_abl, r_ctrl)
    try:
        w_stat, w_p = wilcoxon(d)
    except ValueError:
        w_p = float('nan')
    ci_lo, ci_hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    print(f'\n(1) Test-set sampling — PAIRED over the same {n} profiles:')
    print(f'    mean per-profile Δ = {d.mean():+.4f} μm   SE = {se:.4f}')
    print(f'    95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]  (excludes 0 ⇒ '
          f'distinguishable from zero)')
    print(f'    paired t = {t_stat:+.2f} (p = {t_p:.2e})   '
          f'Wilcoxon p = {w_p:.2e}')
    worse = int((d > 0).sum())
    print(f'    profiles worse under ablation: {worse}/{n} ({worse/n:.1%}), '
          f'better: {n - worse} ({(n-worse)/n:.1%})')

    # bootstrap CI on the level-aggregated Δ (the headline 0.0203)
    rng = np.random.default_rng(42)
    B = 10000
    boot = np.empty(B)
    err_c = (pred_ctrl - true) ** 2
    err_a = (pred_abl - true) ** 2
    for b in range(B):
        idx = rng.integers(0, n, n)
        lc = np.sqrt(err_c[idx].mean(0)).mean()
        la = np.sqrt(err_a[idx].mean(0)).mean()
        boot[b] = la - lc
    print(f'    bootstrap 95% CI on level-aggregated Δ: '
          f'[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]')

    # ---- (2) practical magnitude / effect size ------------------------------
    spread = r_ctrl.std(ddof=1)
    print(f'\n(2) Practical magnitude:')
    print(f'    Δ / mean RMSE          = {(P_abl - P_ctrl) / P_ctrl:.1%}  '
          f'(of the ~0.96 μm error)')
    print(f'    Δ / per-profile spread = {(P_abl - P_ctrl) / spread:.3f}  '
          f'(spread σ = {spread:.3f} μm)   ← your "ratio" idea')
    print(f"    Cohen's d_z = Δ/std(paired Δ) = {d.mean() / sd:.3f}  "
          f'(<0.2 = small effect)')

    print('\n(3) Training/seed noise: control reproduced run_004 to ~1e-6, so '
          'training is deterministic at seed 42 → seed-noise is 0 here and '
          'NOT estimable without retraining at several seeds.')


if __name__ == '__main__':
    main()
