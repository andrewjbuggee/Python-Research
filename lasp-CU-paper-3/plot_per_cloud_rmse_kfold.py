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

# ─── Cloud-base rule for Wood (2005) adiabaticity ────────────────────────
# Same rule and defaults as compute_adiabaticity_Wood2005.m (MATLAB in-situ
# pipeline); see extrapolated_cloud_base. Keep the two in sync.
LWC_THRESHOLD_G_M3          = 0.05         # g/m³  in-situ finder LWC threshold, used by the fallback
BASE_FIT_FRACTION           = 1.0 / 3.0    # bottom fraction of the sampled depth fit by the line
SLOPE_LIMITS_OVER_GAMMA_AD  = (0.25, 3.0)  # physical range of the fitted slope, in units of Γ_ad


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
    """Adiabatic LWC lapse rate Γ_ad in kg/m³/m at a single local state (T, P).

    Γ_ad = ρ_air · (c_p / L_v) · (Γ_d − Γ_m)

    Follows from conservation of moist static energy along a saturated
    adiabat, c_p dT + g dz + L_v dq* = 0, so the condensate produced per
    metre of ascent is dq_l/dz = (c_p / L_v)(Γ_d − Γ_m). Γ_m is the local
    moist-adiabatic lapse rate of Randall (2009, "The Moist Adiabatic
    Lapse Rate", eq. 36), a state function of (T, P) — not a finite
    difference across a layer. Standard form; see e.g. Albrecht et al.
    (1990, GRL 17, 89) or Brenguier (1991, JAS 48, 264). Wood (2005) uses
    Γ_ad but does not give a formula for it.
    """
    rho_air = P_Pa / (R_D * T_K)
    gamma_d = G_GRAV / C_P
    gamma_m = moist_adiabatic_lapse_rate(T_K, P_Pa)
    return rho_air * C_P / L_V * (gamma_d - gamma_m)


def era5_T_P_at_heights(era5_T_surface_to_toa: np.ndarray,
                        era5_P_hPa_surface_to_toa: np.ndarray,
                        z_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hypsometric integration of the ERA5 column to get T [K] and P [Pa]
    at the requested heights z_km.

    Assumes the ERA5 arrays are ordered surface → TOA (P descending). Sets
    z = 0 at the lowest ERA5 level (closest to surface) and integrates the
    hypsometric equation upward, then interpolates T and P at each z.
    """
    P_Pa = era5_P_hPa_surface_to_toa * 100.0
    T    = era5_T_surface_to_toa
    n    = len(P_Pa)
    z    = np.zeros(n)
    for i in range(1, n):
        T_avg = 0.5 * (T[i - 1] + T[i])
        z[i]  = z[i - 1] + (R_D * T_avg / G_GRAV) * np.log(P_Pa[i - 1] / P_Pa[i])
    z_m = np.asarray(z_km, dtype=np.float64) * 1000.0
    return np.interp(z_m, z, T), np.interp(z_m, z, P_Pa)


def extrapolated_cloud_base(lwc_g_m3: np.ndarray,
                            z_m: np.ndarray,
                            gamma_ad_g_m3_per_m: float,
                            lwc_threshold_g_m3: float = LWC_THRESHOLD_G_M3,
                            fit_fraction: float = BASE_FIT_FRACTION,
                            slope_limits_over_gamma_ad: tuple[float, float]
                            = SLOPE_LIMITS_OVER_GAMMA_AD,
                            ) -> tuple[float, float, bool]:
    """Cloud base [m] from a straight-line fit to the bottom of the LWC profile.

    The profiles keep only levels that passed the in-situ LWC and Nc thresholds,
    so the lowest level sits above the true cloud base, and right at the base
    the measurements are ambiguous (small droplet number density, probe noise).
    The adiabatic model is linear in height above cloud base,
    w_l = Γ_l (z − z_B) (Albrecht et al. 1990, GRL 17, 89–92,
    doi:10.1029/GL017i001p00089), and so is a sub-adiabatic cloud with a
    height-independent f_ad. A line is therefore fit to LWC(z) over the bottom
    `fit_fraction` of the sampled depth and its z-intercept (LWC = 0) is the
    cloud base. Below the lowest level the line is extrapolated, not fit.

    The base is never above the lowest level: z_cb = min(intercept, z_lowest).
    If the line reaches zero inside the profile, the lowest level is the base.

    Fallback, when the slope is not physical: one adiabatic distance below the
    lowest level, taking that level to sit at the finder's LWC threshold,
        z_cb = z_lowest − lwc_threshold / Γ_ad.
    The slope is not physical when it is ≤ lower·Γ_ad (no usable upward trend;
    a tiny positive slope sends the intercept kilometres down) or > upper·Γ_ad
    (steeper than any adiabatic parcel), or when fewer than 3 levels can be fit.

    Identical to the 'extrapolate' base method of compute_adiabaticity_Wood2005.m
    in the MATLAB in-situ pipeline; keep the two in sync.

    Parameters
    ----------
    lwc_g_m3 : LWC at each level [g/m³], any order
    z_m : altitude of each level [m], same order
    gamma_ad_g_m3_per_m : adiabatic LWC lapse rate [g/m³ per m]
    lwc_threshold_g_m3 : the in-situ finder's LWC threshold [g/m³]
    fit_fraction : bottom fraction of the sampled depth used in the fit
    slope_limits_over_gamma_ad : (lower, upper) physical slope range, in Γ_ad

    Returns
    -------
    z_cb_m : cloud base [m], never above the lowest level
    lwc_at_lowest_g_m3 : LWC the line (or the fallback adiabat) has at the
        lowest level [g/m³], for the unsampled wedge below it; 0 when the
        lowest level is the base
    used_fallback : True if the fallback set the base
    """
    lwc = np.asarray(lwc_g_m3, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    z_lowest_m = float(np.min(z))
    h_sampled_m = float(np.max(z)) - z_lowest_m

    # Bottom fraction of the sampled depth. The 7-level grid puts a level
    # exactly at z* = 1/3, but its heights are stored as float32, so the
    # computed z* lands up to ~1e-6 above 1/3. The 1e-4 tolerance (3 cm in a
    # 300-m cloud) keeps that level; without it 41% of synthetic profiles had
    # only 2 levels to fit and fell back.
    z_star = (z - z_lowest_m) / h_sampled_m
    in_fit = np.isfinite(lwc) & (z_star <= fit_fraction + 1e-4)

    slope_g_m3_per_m = float('nan')
    lwc_line_at_lowest_g_m3 = float('nan')
    if in_fit.sum() >= 3 and np.std(z[in_fit]) > 0:
        # Heights measured from the lowest level, so the intercept term of the
        # fit is the line's LWC at the lowest level
        slope_g_m3_per_m, lwc_line_at_lowest_g_m3 = np.polyfit(
            z[in_fit] - z_lowest_m, lwc[in_fit], 1)

    slope_over_gamma_ad = slope_g_m3_per_m / gamma_ad_g_m3_per_m
    lower, upper = slope_limits_over_gamma_ad
    # NaN fails both comparisons, so a profile that could not be fit falls back
    is_physical = bool(lower < slope_over_gamma_ad <= upper)

    if is_physical:
        # distance from the lowest level down to LWC = 0; negative if above it
        dz_base_m = lwc_line_at_lowest_g_m3 / slope_g_m3_per_m
        if dz_base_m >= 0:
            return z_lowest_m - dz_base_m, lwc_line_at_lowest_g_m3, False
        # the line reaches zero inside the profile: never put the base above
        # a measured level, so the lowest level is the base
        return z_lowest_m, 0.0, False

    dz_base_m = lwc_threshold_g_m3 / gamma_ad_g_m3_per_m
    return z_lowest_m - dz_base_m, lwc_threshold_g_m3, True


def wood_adiabaticity(lwc_top_to_base_g_m3: np.ndarray,
                       z_top_to_base_km: np.ndarray,
                       era5_T: np.ndarray, era5_P_hPa: np.ndarray,
                       lwc_threshold_g_m3: float = LWC_THRESHOLD_G_M3,
                       ) -> float:
    """LWP_observed / LWP_adiabatic  per Wood (2005, JAS 62, 3011; sec. 3b).

    LWP_ad  = ½ · Γ_ad · h²     (h = z_top − z_cb in m, Γ_ad in g/m³ per m → g/m²)
    LWP_obs = ∫ LWC dz from z_cb to z_top (LWC g/m³, z m → g/m²)

    Cloud base z_cb is the z-intercept of a line fit to the bottom third of
    LWC(z), with a fallback of one adiabatic distance below the lowest level
    when the fitted slope is not physical (see extrapolated_cloud_base). This
    is the same rule as the MATLAB in-situ pipeline (VOCALS-REx and ORACLES).

    LWP_obs covers the same layer as LWP_ad. z_cb is never above the lowest
    level, so every level counts; when z_cb is below it, the unsampled wedge
    between them is added with the same line, ½ · LWC_line(z_lowest) · (z_lowest − z_cb).

    Following Wood, a single Γ_ad is used per cloud, evaluated at the mean
    in-cloud temperature and pressure: T and P are taken from the ERA5
    column (hypsometric integration) at every level of the cloud profile
    and averaged over the sampled cloud depth. Returns NaN when the cloud has
    degenerate thickness or yields Γ_ad ≤ 0.
    """
    z_m = np.asarray(z_top_to_base_km, dtype=np.float64) * 1000.0   # m, top → base
    lwc = np.asarray(lwc_top_to_base_g_m3, dtype=np.float64)         # g/m³
    z_top_m = float(np.max(z_m))
    h_sampled_m = z_top_m - float(np.min(z_m))
    if h_sampled_m <= 0:
        return float('nan')

    # Mean in-cloud T and P: sample the ERA5 column at every profile level
    # (top → base) and take the trapezoid-weighted mean over the sampled depth.
    T_lev, P_lev = era5_T_P_at_heights(era5_T, era5_P_hPa, z_top_to_base_km)
    T_mean = abs(np.trapezoid(T_lev, z_m)) / h_sampled_m
    P_mean = abs(np.trapezoid(P_lev, z_m)) / h_sampled_m
    gamma_ad = gamma_ad_kg_per_m4(T_mean, P_mean)                    # kg/m⁴
    if gamma_ad <= 0:
        return float('nan')
    gamma_ad_g_m3_per_m = gamma_ad * 1000.0                          # g/m³ per m

    z_cb_m, lwc_at_lowest_g_m3, _ = extrapolated_cloud_base(
        lwc, z_m, gamma_ad_g_m3_per_m, lwc_threshold_g_m3)
    dz_base_m = float(np.min(z_m)) - z_cb_m                          # m, ≥ 0
    h_m = z_top_m - z_cb_m

    # Observed LWP over every level, plus the unsampled wedge below the lowest
    finite = np.isfinite(lwc)
    lwp_measured_g = abs(np.trapezoid(lwc[finite], z_m[finite]))     # g/m²
    lwp_wedge_g = 0.5 * lwc_at_lowest_g_m3 * dz_base_m               # g/m²
    lwp_ad_g = 0.5 * gamma_ad_g_m3_per_m * h_m ** 2                  # g/m²
    return (lwp_measured_g + lwp_wedge_g) / lwp_ad_g


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
