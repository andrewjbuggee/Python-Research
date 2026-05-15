"""
Replot the per-cloud RMSE diagnostics for a k-fold run with full coverage.

Aggregates the test-set predictions of every fold (so all N_total samples
get a prediction from a model that never saw them in training), then plots
per-sample mean RMSE against six predictors:

    Row 1: τ_c (log)                       | adiabaticity (Pearson r of r_e^3 vs z-z_base)
    Row 2: drizzle proxy (max r_e bot 30%) | CDF of per-sample RMSE
    Row 3: above-cloud PWV (mm)            | in-cloud PWV (mm)

The water-vapor columns are stored in the HDF5 as molec/cm² and converted
here to precipitable-water millimeters (multiply by 2.991e-22).

First run: loops fold_01..fold_K, reloads each best_model.pt, rebuilds the
fold's test split with `make_kfold_splits(seed=42)` to match training, runs
the model on MPS/CUDA/CPU, and writes agg_pred_cache.npz inside the kfold
directory. Subsequent runs reuse the cache.

Usage:
    python plot_per_cloud_rmse_kfold.py \\
        --kfold-dir ./standalone_results_profile_only_synthetic/\\
M0_run098_synthetic_training_data_7-levels_8_May_2026_20kfold_rev3
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from plot_percentile_profiles_synthetic import (
    adiabaticity_pearson_r,
    _pearson_r,
    _resolve_h5_path,
)


N_EXTRAS = 3
DPI      = 500

# Conversion: column number density (molec/cm²) → precipitable water (mm).
# 1 H2O molecule = 18.015/N_A g; 1 mm PWV = 0.1 g/cm² of liquid-equivalent.
WV_MM_PER_MOLEC_CM2 = 2.991e-22

# ─── Thermodynamic constants for Wood (2005) adiabaticity ────────────────
G_GRAV  = 9.81       # m/s²
R_D     = 287.04     # J/(kg K)  dry air
R_V     = 461.5      # J/(kg K)  water vapor
EPSILON = R_D / R_V  # ≈ 0.622
C_P     = 1005.0     # J/(kg K)  specific heat of dry air at const. P
L_V     = 2.50e6     # J/kg      latent heat of vaporization (T ~ 285 K)


def saturation_vapor_pressure_Pa(T_K):
    """Bolton (1980): e_s [Pa] as a function of T [K]."""
    T_C = T_K - 273.15
    return 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))


def saturation_specific_humidity(T_K, P_Pa):
    e_s = saturation_vapor_pressure_Pa(T_K)
    return EPSILON * e_s / (P_Pa - (1.0 - EPSILON) * e_s)


def moist_adiabatic_lapse_rate(T_K, P_Pa):
    """Γ_m in K/m."""
    q_s = saturation_specific_humidity(T_K, P_Pa)
    num = 1.0 + L_V * q_s / (R_D * T_K)
    den = C_P + (L_V ** 2) * q_s * EPSILON / (R_D * T_K ** 2)
    return G_GRAV * num / den


def gamma_ad_kg_per_m4(T_K, P_Pa):
    """Adiabatic LWC lapse rate Γ_ad in kg/m³/m  (Wood 2005, eq. 4).

    Γ_ad = ρ_air · (c_p / L_v) · (Γ_d − Γ_m)
    """
    rho_air = P_Pa / (R_D * T_K)
    gamma_d = G_GRAV / C_P
    gamma_m = moist_adiabatic_lapse_rate(T_K, P_Pa)
    return rho_air * C_P / L_V * (gamma_d - gamma_m)


def cloud_base_T_P_from_era5(era5_T_surface_to_toa: np.ndarray,
                              era5_P_hPa_surface_to_toa: np.ndarray,
                              z_base_km: float) -> tuple[float, float]:
    """Hypsometric integration of the ERA5 column to get T and P at cloud base.

    Assumes the ERA5 arrays are ordered surface → TOA (P descending). Sets
    z = 0 at the lowest ERA5 level (closest to surface) and integrates the
    hypsometric equation upward, then interpolates at z = z_base_km · 1000.
    """
    P_Pa = era5_P_hPa_surface_to_toa * 100.0
    T    = era5_T_surface_to_toa
    n    = len(P_Pa)
    z    = np.zeros(n)
    for i in range(1, n):
        T_avg = 0.5 * (T[i - 1] + T[i])
        z[i]  = z[i - 1] + (R_D * T_avg / G_GRAV) * np.log(P_Pa[i - 1] / P_Pa[i])
    z_base_m = z_base_km * 1000.0
    return float(np.interp(z_base_m, z, T)), float(np.interp(z_base_m, z, P_Pa))


def wood_adiabaticity(lwc_top_to_base_g_m3: np.ndarray,
                       z_top_to_base_km: np.ndarray,
                       era5_T: np.ndarray, era5_P_hPa: np.ndarray) -> float:
    """LWP_observed / LWP_adiabatic  per Wood (2005).

    LWP_obs = 1000 · ∫ LWC dz   (LWC g/m³, z km, → g/m²)
    LWP_ad  = ½ · Γ_ad · h²     (h = z_top − z_base in m, Γ_ad in kg/m⁴ → g/m²)

    Γ_ad is evaluated at cloud-base T,P from the ERA5 column. Returns NaN
    when the cloud has degenerate thickness or yields Γ_ad ≤ 0.
    """
    z_top_km  = float(z_top_to_base_km[0])
    z_base_km = float(z_top_to_base_km[-1])
    h_m = (z_top_km - z_base_km) * 1000.0
    if h_m <= 0:
        return float('nan')

    lwp_obs_g = 1000.0 * abs(np.trapezoid(lwc_top_to_base_g_m3,
                                          z_top_to_base_km))   # g/m²

    T_base, P_base = cloud_base_T_P_from_era5(era5_T, era5_P_hPa, z_base_km)
    gamma_ad = gamma_ad_kg_per_m4(T_base, P_base)               # kg/m⁴
    if gamma_ad <= 0:
        return float('nan')
    lwp_ad_g = 0.5 * gamma_ad * (h_m ** 2) * 1000.0             # g/m²
    return lwp_obs_g / lwp_ad_g


def re_linear_slope(re_top_to_base: np.ndarray,
                    z_top_to_base_km: np.ndarray) -> float:
    """Linear regression slope of r_e (μm) on z (km) over the profile.

    Sign convention: positive = r_e grows with altitude (adiabatic shape);
    negative = r_e decreases with altitude (drizzle / inverted).
    """
    if len(re_top_to_base) < 2 or np.std(z_top_to_base_km) < 1e-12:
        return float('nan')
    return float(np.polyfit(z_top_to_base_km, re_top_to_base, 1)[0])

# CVD-safe palette (Okabe-Ito + a few from Paul Tol's "muted" set for
# the extended figure, all distinguishable to deuteranopia/protanopia)
COLOR_TAUC         = '#0072B2'   # blue
COLOR_ADIAB        = '#009E73'   # bluish-green
COLOR_DRIZZLE      = '#D55E00'   # vermillion          (bottom-30 % proxy)
COLOR_WV_ABOVE     = '#CC79A7'   # reddish purple
COLOR_WV_IN        = '#E69F00'   # orange
COLOR_DRIZZLE_TOP  = '#56B4E9'   # sky blue            (top-30 % proxy)
COLOR_RE_MAX       = '#882255'   # deep magenta (Tol)  (max anywhere)
COLOR_RE_RANGE     = '#117733'   # forest green (Tol)  (dynamic range)
COLOR_ALPHA        = '#AA3377'   # purple-pink (Tol)   (α shape parameter)
COLOR_SLOPE        = '#332288'   # indigo      (Tol)   (linear slope)
COLOR_WOOD         = '#88CCEE'   # cyan        (Tol)   (Wood adiabaticity)


def setup_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif', 'Computer Modern Roman',
                              'Times New Roman'],
        'mathtext.fontset':  'cm',
        'mathtext.rm':       'serif',
    })


# ─── Inference: rebuild test split per fold and run the saved model ──────
def _run_kfold_inference(kfold_dir: Path,
                         h5_override: str | None,
                         device_arg: str | None,
                         seed: int) -> dict:
    """Loop folds, predict on each fold's held-out test set, concatenate."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from models                              import RetrievalConfig
    from models_profile_only_extras          import ProfileOnlyNetworkExtras
    from data                                import LibRadtranDatasetExtras
    from train_standalone_profile_only_synthetic import make_kfold_splits
    from train_standalone_profile_only_extras    import predict_test

    with (kfold_dir / 'summary.json').open() as f:
        s = json.load(f)
    hp            = s['hyperparams']
    extras_active = s['extras_active']
    K             = int(s['n_folds'])
    n_val         = int(s['folds'][0]['n_val'])
    h5_path       = _resolve_h5_path(s['h5_path'], h5_override)

    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif (getattr(torch.backends, 'mps', None) is not None
          and torch.backends.mps.is_available()):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'  HDF5   : {h5_path}')
    print(f'  device : {device}')
    print(f'  K = {K}, n_val per fold = {n_val}, seed = {seed}')

    dataset = LibRadtranDatasetExtras(
        str(h5_path), normalize=True, instrument='hysics',
        zero_tau_c     = not extras_active['tau_c'],
        zero_wv_above  = not extras_active['wv_above_cloud'],
        zero_wv_in     = not extras_active['wv_in_cloud'],
    )
    n_total  = len(dataset)
    n_levels = dataset.n_levels
    folds = make_kfold_splits(n_total, K, n_val=n_val, seed=seed)

    model_cfg = RetrievalConfig(
        n_wavelengths=636,
        n_geometry_inputs=4,
        n_levels=n_levels,
        hidden_dims=tuple(hp['hidden_dims']),
        dropout=hp['dropout'],
        activation=hp.get('activation', 'gelu'),
    )
    pin = (device.type == 'cuda')

    all_pred, all_pred_std, all_true, all_idx = [], [], [], []
    for k, (_, _, test_idx) in enumerate(folds, start=1):
        ckpt_path = kfold_dir / f'fold_{k:02d}' / 'best_model.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ProfileOnlyNetworkExtras(model_cfg, n_extras=N_EXTRAS).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=hp['batch_size'], shuffle=False,
            num_workers=4, pin_memory=pin,
        )
        out = predict_test(model, test_loader, device, model_cfg)
        all_pred.append(out['pred'])
        all_pred_std.append(out['pred_std'])
        all_true.append(out['true'])
        all_idx.append(np.asarray(test_idx, dtype=np.int64))
        print(f'  fold {k:02d}: n_test = {len(test_idx):5d}  done '
              f'(best epoch {int(ckpt["epoch"])})')

    return {
        'pred':     np.concatenate(all_pred,     axis=0),
        'pred_std': np.concatenate(all_pred_std, axis=0),
        'true':     np.concatenate(all_true,     axis=0),
        'h5_idx':   np.concatenate(all_idx,      axis=0),
    }


# ─── Predictor extraction (incl. water-vapor mm conversion) ──────────────
def compute_predictors(h5_idx: np.ndarray, h5_path: Path) -> dict:
    """Per-sample predictors for the aggregate, recovered from the HDF5."""
    sort_order = np.argsort(h5_idx)
    unsort     = np.argsort(sort_order)
    sorted_idx = h5_idx[sort_order]

    with h5py.File(h5_path, 'r') as f:
        tau_c     = f['tau_c'][sorted_idx]
        profs     = f['profiles_raw'][sorted_idx]
        z_arr     = f['profiles_raw_z'][sorted_idx]
        nlev_arr  = f['profile_n_levels'][sorted_idx]
        lwc_arr   = f['lwc'][sorted_idx]
        wv_above  = f['wv_above_cloud'][sorted_idx]
        wv_in     = f['wv_in_cloud'][sorted_idx]
        alpha     = f['alpha'][sorted_idx]
        era5_T    = f['era5_temperature'][sorted_idx]
        era5_P    = f['era5_pressure_levels'][:]      # shape (37,)

    tau_c    = tau_c[unsort]
    profs    = profs[unsort]
    z_arr    = z_arr[unsort]
    nlev_arr = nlev_arr[unsort]
    lwc_arr  = lwc_arr[unsort]
    wv_above = wv_above[unsort]
    wv_in    = wv_in[unsort]
    alpha    = alpha[unsort]
    era5_T   = era5_T[unsort]

    n = len(h5_idx)
    adiab             = np.full(n, np.nan, dtype=np.float32)
    drizzle_proxy     = np.full(n, np.nan, dtype=np.float32)
    drizzle_proxy_top = np.full(n, np.nan, dtype=np.float32)
    re_max_anywhere   = np.full(n, np.nan, dtype=np.float32)
    re_range          = np.full(n, np.nan, dtype=np.float32)
    re_std            = np.full(n, np.nan, dtype=np.float32)
    slope_per_prof    = np.full(n, np.nan, dtype=np.float32)
    wood_adiab        = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        nL = int(nlev_arr[i])
        re  = profs[i,   :nL].astype(np.float64)
        z   = z_arr[i,   :nL].astype(np.float64)
        lwc = lwc_arr[i, :nL].astype(np.float64)

        adiab[i]             = adiabaticity_pearson_r(re, z)
        drizzle_proxy[i]     = float(re[int(0.7 * nL):].max())
        n_top                = max(1, nL - int(0.7 * nL))
        drizzle_proxy_top[i] = float(re[:n_top].max())
        re_max_anywhere[i]   = float(re.max())
        re_range[i]          = float(re.max() - re.min())
        # Vertical std of r_e across the cloud profile — a smooth scalar
        # version of "how variable is droplet size with altitude". Differs
        # from the dynamic range (max-min) in that it down-weights tail
        # behavior and reflects fluctuations across all levels.
        re_std[i]            = float(re.std(ddof=0))
        slope_per_prof[i]    = re_linear_slope(re, z)
        wood_adiab[i]        = wood_adiabaticity(lwc, z, era5_T[i], era5_P)

    return {
        'tau_c':             np.asarray(tau_c, dtype=np.float32),
        'adiab_score':       adiab,
        'drizzle_proxy':     drizzle_proxy,
        'drizzle_proxy_top': drizzle_proxy_top,
        're_max':            re_max_anywhere,
        're_range':          re_range,
        're_std':            re_std,
        'alpha':             np.asarray(alpha, dtype=np.float32),
        'wv_above_mm':       (wv_above * WV_MM_PER_MOLEC_CM2).astype(np.float32),
        'wv_in_mm':          (wv_in    * WV_MM_PER_MOLEC_CM2).astype(np.float32),
        'linear_slope':      slope_per_prof,
        'wood_adiab':        wood_adiab,
    }


# ─── Figure ──────────────────────────────────────────────────────────────
def plot_diagnostics_3x2(per_sample_rmse: np.ndarray,
                         predictors: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    def _scatter(ax, x, label, name, color, xlog=False):
        valid = np.isfinite(x) & np.isfinite(per_sample_rmse)
        ax.scatter(x[valid], per_sample_rmse[valid], s=6, alpha=0.32,
                   color=color, edgecolors='none')
        if xlog:
            ax.set_xscale('log')
        r = _pearson_r(x, per_sample_rmse)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Per-sample mean RMSE (μm)', fontsize=11)
        ax.set_title(fr'({label})  RMSE vs {name}    '
                     fr'(Pearson $r$ = {r:+.3f})',
                     fontsize=11)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    _scatter(axes[0, 0], predictors['tau_c'],
             'a', r'$\tau_c$',
             color=COLOR_TAUC, xlog=True)
    _scatter(axes[0, 1], predictors['adiab_score'],
             'b', r'adiabaticity (Pearson $r$ of $r_e^3$ vs $z-z_{base}$)',
             color=COLOR_ADIAB)
    _scatter(axes[1, 0], predictors['drizzle_proxy'],
             'c', r'drizzle proxy:  max $r_e$ in lower 30 % ($\mu$m)',
             color=COLOR_DRIZZLE)

    # (d) CDF
    ax = axes[1, 1]
    x_sorted = np.sort(per_sample_rmse[np.isfinite(per_sample_rmse)])
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax.plot(x_sorted, cdf, color='black', linewidth=1.6)
    pcts     = (10, 25, 50, 75, 90)
    pct_vals = np.percentile(x_sorted, pcts)
    for p, v in zip(pcts, pct_vals):
        ax.axvline(v, color='0.55', linestyle=':', linewidth=0.8)
        ax.text(v, 1.02, f'p{p}', fontsize=8, ha='center', va='bottom',
                color='0.4')
    ax.set_xlabel('Per-sample mean RMSE (μm)', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title('(d)  CDF of per-sample RMSE on the 20-fold aggregate',
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    _scatter(axes[2, 0], predictors['wv_above_mm'],
             'e', 'above-cloud column water vapor (mm)',
             color=COLOR_WV_ABOVE)
    _scatter(axes[2, 1], predictors['wv_in_mm'],
             'f', 'in-cloud column water vapor (mm)',
             color=COLOR_WV_IN)

    fig.suptitle(f'Per-cloud RMSE diagnostics — 20-fold aggregate '
                 f'({len(per_sample_rmse):,} samples)',
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_diagnostics_5x2(per_sample_rmse: np.ndarray,
                         predictors: dict, out_path: Path) -> None:
    """Extended 6x2 layout: 3x2 base + drizzle-top/max + range/alpha + slope/Wood."""
    fig, axes = plt.subplots(6, 2, figsize=(13, 26.4))

    def _scatter(ax, x, label, name, color, xlog=False):
        valid = np.isfinite(x) & np.isfinite(per_sample_rmse)
        ax.scatter(x[valid], per_sample_rmse[valid], s=6, alpha=0.32,
                   color=color, edgecolors='none')
        if xlog:
            ax.set_xscale('log')
        r = _pearson_r(x, per_sample_rmse)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Per-sample mean RMSE (μm)', fontsize=11)
        ax.set_title(fr'({label})  RMSE vs {name}    '
                     fr'(Pearson $r$ = {r:+.3f})',
                     fontsize=11)
        ax.grid(alpha=0.3, which='both' if xlog else 'major')

    # ── Rows 1-3: same as the 3x2 figure ────────────────────────────────
    _scatter(axes[0, 0], predictors['tau_c'],
             'a', r'$\tau_c$',
             color=COLOR_TAUC, xlog=True)
    _scatter(axes[0, 1], predictors['adiab_score'],
             'b', r'adiabaticity (Pearson $r$ of $r_e^3$ vs $z-z_{base}$)',
             color=COLOR_ADIAB)
    _scatter(axes[1, 0], predictors['drizzle_proxy'],
             'c', r'drizzle proxy:  max $r_e$ in lower 30 % ($\mu$m)',
             color=COLOR_DRIZZLE)

    # (d) CDF
    ax = axes[1, 1]
    x_sorted = np.sort(per_sample_rmse[np.isfinite(per_sample_rmse)])
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax.plot(x_sorted, cdf, color='black', linewidth=1.6)
    pcts     = (10, 25, 50, 75, 90)
    pct_vals = np.percentile(x_sorted, pcts)
    for p, v in zip(pcts, pct_vals):
        ax.axvline(v, color='0.55', linestyle=':', linewidth=0.8)
        ax.text(v, 1.02, f'p{p}', fontsize=8, ha='center', va='bottom',
                color='0.4')
    ax.set_xlabel('Per-sample mean RMSE (μm)', fontsize=11)
    ax.set_ylabel('Cumulative fraction', fontsize=11)
    ax.set_title('(d)  CDF of per-sample RMSE on the 20-fold aggregate',
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    _scatter(axes[2, 0], predictors['wv_above_mm'],
             'e', 'above-cloud column water vapor (mm)',
             color=COLOR_WV_ABOVE)
    _scatter(axes[2, 1], predictors['wv_in_mm'],
             'f', 'in-cloud column water vapor (mm)',
             color=COLOR_WV_IN)

    # ── Row 4: top-30 % drizzle proxy + max anywhere ────────────────────
    _scatter(axes[3, 0], predictors['drizzle_proxy_top'],
             'g', r'cloud-top drizzle proxy:  max $r_e$ in upper 30 % ($\mu$m)',
             color=COLOR_DRIZZLE_TOP)
    _scatter(axes[3, 1], predictors['re_max'],
             'h', r'max $r_e$ anywhere in profile ($\mu$m)',
             color=COLOR_RE_MAX)

    # ── Row 5: dynamic range + alpha ────────────────────────────────────
    _scatter(axes[4, 0], predictors['re_range'],
             'i', r'profile dynamic range:  $\max r_e - \min r_e$ ($\mu$m)',
             color=COLOR_RE_RANGE)
    _scatter(axes[4, 1], predictors['alpha'],
             'j', r'$\alpha$ shape parameter '
                  r'($\nu_{\mathrm{eff}} = 1/(\alpha + 3)$)',
             color=COLOR_ALPHA)

    # ── Row 6: linear slope + Wood (2005) adiabaticity ──────────────────
    _scatter(axes[5, 0], predictors['linear_slope'],
             'k', r'linear slope of $r_e(z)$ ($\mu$m/km)',
             color=COLOR_SLOPE)
    _scatter(axes[5, 1], predictors['wood_adiab'],
             'l', r'Wood (2005) adiabaticity:  $\mathrm{LWP}_{\mathrm{obs}}'
                  r'/\mathrm{LWP}_{\mathrm{ad}}$',
             color=COLOR_WOOD)

    fig.suptitle(f'Per-cloud RMSE diagnostics — extended (20-fold aggregate, '
                 f'{len(per_sample_rmse):,} samples)',
                 fontsize=13, y=1.003)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--kfold-dir', required=True,
                   help='Directory containing fold_01..fold_K + summary.json.')
    p.add_argument('--output-name',
                   default='per_cloud_rmse_vs_predictors_with_wv.png',
                   help='Filename for the 3x2 figure inside --kfold-dir.')
    p.add_argument('--output-name-extended',
                   default='per_cloud_rmse_vs_predictors_extended.png',
                   help='Filename for the 5x2 (extended) figure.')
    p.add_argument('--refresh', action='store_true',
                   help='Force re-inference even if agg_pred_cache.npz exists.')
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the seed used for k-fold split at training.')
    p.add_argument('--h5-path', default=None,
                   help='Override the HDF5 path stored in summary.json.')
    return p.parse_args()


def main():
    args = parse_args()
    setup_style()

    kfold_dir = Path(args.kfold_dir).resolve()
    if not kfold_dir.exists():
        raise FileNotFoundError(f'kfold-dir not found: {kfold_dir}')
    cache_path = kfold_dir / 'agg_pred_cache.npz'

    if cache_path.exists() and not args.refresh:
        print(f'Loading cached aggregate from {cache_path}')
        c = np.load(cache_path)
        pred, pred_std, true = c['pred'], c['pred_std'], c['true']
        h5_idx               = c['h5_idx']
    else:
        print('Running 20-fold inference (may take several minutes)...')
        out = _run_kfold_inference(kfold_dir, args.h5_path, args.device, args.seed)
        pred, pred_std, true = out['pred'], out['pred_std'], out['true']
        h5_idx               = out['h5_idx']
        np.savez_compressed(cache_path,
                            pred=pred, pred_std=pred_std, true=true,
                            h5_idx=h5_idx)
        print(f'Cached aggregate -> {cache_path}')

    print(f'Aggregate : {pred.shape[0]:,} samples x {pred.shape[1]} levels')

    err = pred - true
    per_sample_rmse = np.sqrt(np.mean(err**2, axis=1))

    with (kfold_dir / 'summary.json').open() as f:
        s = json.load(f)
    h5_path = _resolve_h5_path(s['h5_path'], args.h5_path)
    predictors = compute_predictors(h5_idx, h5_path)

    out_fig = kfold_dir / args.output_name
    plot_diagnostics_3x2(per_sample_rmse, predictors, out_fig)
    print(f'3x2 figure -> {out_fig}  (dpi={DPI})')

    out_fig_ext = kfold_dir / args.output_name_extended
    plot_diagnostics_5x2(per_sample_rmse, predictors, out_fig_ext)
    print(f'5x2 figure -> {out_fig_ext}  (dpi={DPI})')


if __name__ == '__main__':
    main()
