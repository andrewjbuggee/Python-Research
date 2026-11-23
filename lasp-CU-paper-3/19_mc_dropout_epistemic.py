"""
mc_dropout_epistemic.py — epistemic + aleatoric uncertainty by Monte Carlo dropout
==================================================================================
For one trained profile + τ model (DropletProfileNetwork), run the held-out
test set through the network T times with dropout switched ON, and split the
predictive variance of r_e at each level (and of τ_c) into

    epistemic variance  = Var_t[ μ_t ]          (spread of the T mean predictions)
    aleatoric variance  = Mean_t[ σ_t² ]        (mean of the T predicted variances)
    total variance      = epistemic + aleatoric

where μ_t, σ_t are the network's mean and σ outputs on pass t. This is
Eq. 9 of Kendall & Gal (2017), "What uncertainties do we need in Bayesian
deep learning for computer vision?", NeurIPS 30,
https://arxiv.org/abs/1703.04977 (checked against the paper 19 Nov 2026):
    Var(y) ≈ (1/T) Σ_t ŷ_t² − ((1/T) Σ_t ŷ_t)² + (1/T) Σ_t σ̂_t²,
whose first two terms are the 1/T (ddof = 0) variance of the ŷ_t used here.
T = 50 also follows them (Sec. 4: "we use 50 Monte Carlo dropout samples",
following their ref. [22]). Keeping dropout on at test time as an approximate
Bayesian posterior over the weights is Gal & Ghahramani (2016), "Dropout as a
Bayesian approximation", ICML, https://arxiv.org/abs/1506.02142 (that paper I
have not checked against).

Why only dropout is switched on
  The encoder is [Linear → LayerNorm → activation → Dropout] × n_layers
  (models.DropletProfileNetwork). LayerNorm has no running statistics, so it
  acts the same in train and eval mode; putting only the nn.Dropout modules
  in train mode changes nothing else.

Caveats (my notes, not from the sources above)
  * The epistemic estimate scales with the dropout rate, which was tuned for
    accuracy, not for calibrated uncertainty; treat it as approximate.
  * The aleatoric term inherits the log-σ clamp: for run_004 most mid-cloud σ
    sit on the 0.888 μm floor (17 Nov 2026), so aleatoric (and total) σ there
    is at least the floor. The share on the floor is reported.
  * The test set is drawn from the same synthetic distribution as training,
    so epistemic uncertainty here is expected to be small; it should grow for
    spectra unlike the training data (e.g. in-situ VOCALS-REx / ORACLES).

Outputs (in --out-dir, default the run directory)
  mc_dropout_uncertainty.npz   per test sample: deterministic prediction
                               (dropout off), MC mean, epistemic / aleatoric /
                               total σ for r_e (n, L) in μm and τ_c (n,)
  mc_dropout_uncertainty.json  per-level summaries + run settings (seed, T, device)

Usage
    python mc_dropout_epistemic.py --run-dir hyper_parameter_sweep/sweep_results_profile_tau_synthetic/run_004 \
        --h5-path training_data/synthetic_training_data_7-levels_8_May_2026.h5
    python mc_dropout_epistemic.py --run-dir <run> --n-passes 100 --seed 7 --device mps

The dropout masks are random: --seed (default 0) seeds torch and is saved
with the results. On CPU the run is repeatable bit for bit; GPU/MPS kernels
may differ slightly between runs.
Author: Andrew J. Buggee, 19 Nov 2026
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

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'hyper_parameter_sweep'))
from data import create_dataloaders, tau_bounds_for_run                    # noqa: E402
from models import (DropletProfileNetwork, RetrievalConfig,                # noqa: E402
                    log_std_bounds_for_run, sigma_floor_um)
from plot_percentile_profiles_synthetic import _resolve_h5_path            # noqa: E402
from sweep_train_profile_tau_synthetic import (tau_sigma_to_physical,      # noqa: E402
                                               tau_target_to_physical)

Z95 = 1.96   # standard-normal 97.5 % quantile


# ─── Model + test split, rebuilt exactly as the trainer built them ────────
def load_model_and_test_set(run_dir: Path, h5_override: str | None,
                            device: torch.device):
    """Returns (model, cfg, tau_transform, tau_bounds, X_test, re_true_um,
    tau_true, test_idx). X_test is the normalized network input (n, 640)."""
    s = json.load(open(run_dir / 'summary.json'))
    hp = s['hyperparams']
    tau_transform = s.get('tau_transform', hp.get('tau_transform', 'linear'))
    tau_bounds, _ = tau_bounds_for_run(s)
    log_std_min, log_std_max = log_std_bounds_for_run(s)
    h5_path = _resolve_h5_path(s['data']['h5_path'], h5_override)

    # Same split call as the trainer (profile-aware, the run's split seed).
    # num_workers=0: worker processes are not needed for one pass over the
    # test set, and avoid spawn problems on macOS.
    _, _, test_loader = create_dataloaders(
        h5_path=str(h5_path), instrument=s['data'].get('instrument', 'hysics'),
        batch_size=1024, num_workers=0, seed=int(s.get('seed', 42)),
        profile_holdout=True, n_val_profiles=s['n_val_profiles'],
        n_test_profiles=s['n_test_profiles'], tau_bounds=tau_bounds)
    test_idx = np.asarray(test_loader.dataset.indices, dtype=np.int64)

    xs, profs, taus = [], [], []
    for x, prof, tau in test_loader:            # normalized inputs and targets
        xs.append(x)
        profs.append(prof)
        taus.append(tau)
    X = torch.cat(xs)                                                  # (n, 640)
    cfg = RetrievalConfig(
        n_wavelengths=636, n_geometry_inputs=4, n_levels=profs[0].shape[1],
        hidden_dims=tuple(hp['hidden_dims']), dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
        tau_min=float(tau_bounds[0]), tau_max=float(tau_bounds[1]),
        log_std_min=log_std_min, log_std_max=log_std_max)
    re_true_um = (torch.cat(profs).numpy() * (cfg.re_max - cfg.re_min)
                  + cfg.re_min)                                        # (n, L) μm
    tau_true = (torch.cat(taus).numpy() * (tau_bounds[1] - tau_bounds[0])
                + tau_bounds[0])                                       # (n,)

    model = DropletProfileNetwork(cfg).to(device)
    ckpt = torch.load(run_dir / 'best_model.pt', map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict)
                                         and 'model_state_dict' in ckpt) else ckpt
    model.load_state_dict(state)
    return model, cfg, tau_transform, tau_bounds, X, re_true_um, tau_true, test_idx


@torch.no_grad()
def forward_physical(model, X: torch.Tensor, tau_transform: str, tau_bounds,
                     device: torch.device, chunk: int = 4096) -> dict:
    """One pass over X; outputs in physical units (r_e μm, τ dimensionless).

    τ is converted from the normalized head with the trainer's own helpers,
    so the same code serves the 'linear' and 'log10' τ transforms.
    """
    mu, sd, sdn, tm, ts, tsn = [], [], [], [], [], []
    for i in range(0, X.shape[0], chunk):
        out = model(X[i:i + chunk].to(device))
        mu.append(out['profile'].cpu().numpy())                       # (b, L) μm
        sd.append(out['profile_std'].cpu().numpy())                   # (b, L) μm
        sdn.append(out['profile_std_normalized'].cpu().numpy())       # (b, L)
        tn = out['tau_normalized'].squeeze(-1).cpu().numpy()          # (b,)
        sn = out['tau_std_normalized'].squeeze(-1).cpu().numpy()      # (b,)
        tm.append(tau_target_to_physical(tn, tau_transform, tau_bounds))
        ts.append(tau_sigma_to_physical(tn, sn, tau_transform, tau_bounds))
        tsn.append(sn)
    return {'mu': np.concatenate(mu), 'sigma': np.concatenate(sd),
            'sigma_norm': np.concatenate(sdn), 'tau': np.concatenate(tm),
            'tau_sigma': np.concatenate(ts), 'tau_sigma_norm': np.concatenate(tsn)}


def enable_mc_dropout(model: nn.Module) -> int:
    """Eval mode everywhere, then train mode for the Dropout modules only.

    Returns how many Dropout modules were switched on.
    """
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description='MC-dropout epistemic + aleatoric uncertainty')
    p.add_argument('--run-dir', required=True, help='trained run (summary.json + best_model.pt)')
    p.add_argument('--h5-path', default=None,
                   help='local HDF5 if the path in summary.json is not reachable')
    p.add_argument('--n-passes', type=int, default=50, help='T, forward passes (default 50)')
    p.add_argument('--seed', type=int, default=0, help='seed for the dropout masks')
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--out-dir', default=None, help='default: the run directory')
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    (model, cfg, tau_transform, tau_bounds, X, re_true, tau_true,
     test_idx) = load_model_and_test_set(run_dir, args.h5_path, device)
    n, L = re_true.shape
    print(f'Run     : {run_dir}')
    print(f'Test set: {n} samples x {L} levels; dropout p = {cfg.dropout:.3f}; '
          f'log-σ clamp min {cfg.log_std_min:g} (σ floor {sigma_floor_um(cfg):.4f} μm)')
    cache = run_dir / 'pred_cache.npz'
    if cache.exists() and not np.array_equal(np.load(cache)['h5_idx'], test_idx):
        raise SystemExit('test split differs from pred_cache.npz h5_idx — check the HDF5')

    # ── 1. Deterministic prediction (dropout OFF), the one used everywhere else
    model.eval()
    det = forward_physical(model, X, tau_transform, tau_bounds, device)

    # ── 2. T passes with dropout ON — the core of MC dropout ───────────────
    n_drop = enable_mc_dropout(model)
    torch.manual_seed(args.seed)             # dropout masks; also seeds CUDA
    mu_t = np.empty((args.n_passes, n, L), np.float64)    # (T, n, L) μm
    var_t = np.empty((args.n_passes, n, L), np.float64)   # (T, n, L) μm²
    tau_t = np.empty((args.n_passes, n), np.float64)      # (T, n)
    tau_var_t = np.empty((args.n_passes, n), np.float64)  # (T, n)
    t0 = time.time()
    for t in range(args.n_passes):
        o = forward_physical(model, X, tau_transform, tau_bounds, device)
        mu_t[t], var_t[t] = o['mu'], o['sigma'] ** 2
        tau_t[t], tau_var_t[t] = o['tau'], o['tau_sigma'] ** 2
    model.eval()
    print(f'{args.n_passes} passes with {n_drop} dropout layers on: {time.time() - t0:.1f} s')

    # Kendall & Gal (2017) decomposition, per sample and level.
    mc_mean = mu_t.mean(axis=0)                     # (n, L)
    epi_var = mu_t.var(axis=0)                      # (n, L)  ddof=0
    ale_var = var_t.mean(axis=0)                    # (n, L)
    tot_var = epi_var + ale_var
    tau_mc_mean = tau_t.mean(axis=0)
    tau_epi_var = tau_t.var(axis=0)
    tau_ale_var = tau_var_t.mean(axis=0)
    tau_tot_var = tau_epi_var + tau_ale_var

    # ── 3. Summaries ──────────────────────────────────────────────────────
    floor_norm = float(np.exp(cfg.log_std_min))
    z = (mc_mean - re_true) / np.sqrt(tot_var)                         # (n, L)
    tau_z = (tau_mc_mean - tau_true) / np.sqrt(tau_tot_var)
    per_level = {
        'rmse_deterministic_um': np.sqrt(((det['mu'] - re_true) ** 2).mean(0)),
        'rmse_mc_mean_um':       np.sqrt(((mc_mean - re_true) ** 2).mean(0)),
        'median_epistemic_sigma_um': np.median(np.sqrt(epi_var), axis=0),
        'median_aleatoric_sigma_um': np.median(np.sqrt(ale_var), axis=0),
        'median_total_sigma_um':     np.median(np.sqrt(tot_var), axis=0),
        # share of the total variance that is epistemic (median over samples)
        'median_epistemic_fraction': np.median(epi_var / tot_var, axis=0),
        'share_det_sigma_on_floor':  np.isclose(det['sigma_norm'], floor_norm,
                                                rtol=1e-4, atol=0).mean(0),
        'z_std_total':               z.std(axis=0),
        'coverage_1p96_total':       (np.abs(z) <= Z95).mean(axis=0),
    }
    tau_summary = {
        'rmse_deterministic': float(np.sqrt(((det['tau'] - tau_true) ** 2).mean())),
        'rmse_mc_mean':       float(np.sqrt(((tau_mc_mean - tau_true) ** 2).mean())),
        'median_epistemic_sigma': float(np.median(np.sqrt(tau_epi_var))),
        'median_aleatoric_sigma': float(np.median(np.sqrt(tau_ale_var))),
        'median_total_sigma':     float(np.median(np.sqrt(tau_tot_var))),
        'median_epistemic_fraction': float(np.median(tau_epi_var / tau_tot_var)),
        'share_det_sigma_on_floor': float(np.isclose(det['tau_sigma_norm'], floor_norm,
                                                     rtol=1e-4, atol=0).mean()),
        'z_std_total':        float(tau_z.std()),
        'coverage_1p96_total': float((np.abs(tau_z) <= Z95).mean()),
    }

    print('\nr_e per level (test set; σ values are medians over samples, μm)')
    print(f'  {"lev":>3} {"RMSE det":>9} {"RMSE MC":>8} {"σ_epi":>7} {"σ_ale":>7} '
          f'{"σ_tot":>7} {"epi frac":>8} {"σ on floor":>10} {"std(z)":>7} {"cov95":>6}')
    for i in range(L):
        v = {k: a[i] for k, a in per_level.items()}
        print(f'  {i + 1:>3} {v["rmse_deterministic_um"]:9.3f} {v["rmse_mc_mean_um"]:8.3f} '
              f'{v["median_epistemic_sigma_um"]:7.3f} {v["median_aleatoric_sigma_um"]:7.3f} '
              f'{v["median_total_sigma_um"]:7.3f} {v["median_epistemic_fraction"]:8.2f} '
              f'{v["share_det_sigma_on_floor"]:10.2f} {v["z_std_total"]:7.2f} '
              f'{v["coverage_1p96_total"]:6.2f}')
    ts = tau_summary
    print(f'τ_c: RMSE det {ts["rmse_deterministic"]:.3f}, MC {ts["rmse_mc_mean"]:.3f}; '
          f'σ_epi {ts["median_epistemic_sigma"]:.3f}, σ_ale {ts["median_aleatoric_sigma"]:.3f}, '
          f'σ_tot {ts["median_total_sigma"]:.3f}; epi frac {ts["median_epistemic_fraction"]:.2f}; '
          f'σ on floor {ts["share_det_sigma_on_floor"]:.2f}; std(z) {ts["z_std_total"]:.2f}, '
          f'cov95 {ts["coverage_1p96_total"]:.2f}')

    np.savez_compressed(
        out_dir / 'mc_dropout_uncertainty.npz', h5_idx=test_idx,
        re_true_um=re_true, re_det_um=det['mu'], re_det_sigma_um=det['sigma'],
        re_mc_mean_um=mc_mean, re_sigma_epistemic_um=np.sqrt(epi_var),
        re_sigma_aleatoric_um=np.sqrt(ale_var), re_sigma_total_um=np.sqrt(tot_var),
        tau_true=tau_true, tau_det=det['tau'], tau_det_sigma=det['tau_sigma'],
        tau_mc_mean=tau_mc_mean, tau_sigma_epistemic=np.sqrt(tau_epi_var),
        tau_sigma_aleatoric=np.sqrt(tau_ale_var), tau_sigma_total=np.sqrt(tau_tot_var))
    meta = {'run_dir': str(run_dir), 'n_passes': args.n_passes, 'seed': args.seed,
            'device': args.device, 'dropout_p': cfg.dropout, 'n_dropout_layers': n_drop,
            'log_std_min': cfg.log_std_min, 'sigma_floor_um': sigma_floor_um(cfg),
            'n_test_samples': n, 'tau_transform': tau_transform,
            'r_e_per_level': {k: np.asarray(a).tolist() for k, a in per_level.items()},
            'tau_c': tau_summary}
    json.dump(meta, open(out_dir / 'mc_dropout_uncertainty.json', 'w'), indent=2)
    print(f'\nArrays  -> {out_dir / "mc_dropout_uncertainty.npz"}')
    print(f'Summary -> {out_dir / "mc_dropout_uncertainty.json"}')


if __name__ == '__main__':
    main()
