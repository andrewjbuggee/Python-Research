"""
paper3_physics.py
=================
Shared physics for the numbered training/analysis steps (10_ … 16_):

  1. Wood (2005) adiabaticity with the 7-Nov-2026 cloud-base rule
     (copied verbatim from plot_per_cloud_rmse_kfold.py, which matches the
     MATLAB in-situ pipeline compute_adiabaticity_Wood2005.m to 6e-15).
  2. Mie extinction from the libRadtran custom tables, exactly as the τ_c
     training label is computed in 06_synthetic_profile_generator.compute_tau_c.
  3. LWP from a retrieved r_e profile and τ_c that is consistent with that label.

Why LWP changes (the "wc geometry bias")
----------------------------------------
Before Nov 2026 the τ_c label was 750·2.32·∫LWC/r_e dz, where 2.32 bundled the
Mie efficiency (≈ 2.11 at 550 nm) with an error in the libRadtran water-cloud
file geometry (an extra layer above cloud top). Retrieved LWP was then computed
with Paper-2 Eq. 10, which assumes Q_ext = 2:

    LWP_Eq10 = (2/3) ρ_w τ_c ∫ r_e³ dz / ∫ r_e² dz

so it returned ≈ (2.32/2) LWP, i.e. ~16 % high wherever N is constant with
height (compare_lwp_methods_retrieved_profiles.py, item 4). From the
12-Nov-2026 synthetic set on, the label is the Mie optical depth

    τ_c = ∫ β dz,   β = ext(r_e; α) · LWC,   ext = 750 · Q̄_ext(r_e, α) / r_e

(ext from the same custom Mie tables libRadtran uses; km⁻¹ per g m⁻³), and the
RT now sees exactly that τ_c (write_wc_file_mixed_layers.m). The LWP step
therefore has to use the same ext:

    LWP = 1000 · τ_c · ∫ r_e³ dz / ∫ ext(r_e; α) r_e³ dz        [g m⁻²]   (lwp_constant_N_mie)

Derivation (Claude's, combining Paper-2's constant-N assumption with the
label's Mie extinction; not from a paper): with N and α constant with height,
LWC(z) = c·r_e(z)³ (for a gamma distribution of fixed α, ⟨r³⟩ ∝ r_e³). Then
τ_c = c ∫ ext(r_e) r_e³ dz gives c, and LWP = ∫ LWC dz = c ∫ r_e³ dz; the factor
1000 converts z from km to m. With ext = 1500/r_e (Q̄_ext = 2) this is exactly
Paper-2 Eq. 10. The constant-N assumption itself is unchanged and still
carries its own bias (≈ −20 % vs the LWC-integrated truth for the May-2026
data); this removes only the Q mismatch.

The level spacing cancels in the ratio, so retrieved profiles (7 evenly spaced
levels, true heights unknown) can use unit spacing.

All quantities carry units in their names or docstrings.
Author: Claude (assistant) for Andrew J. Buggee, 14 Nov 2026
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 1. Wood (2005) adiabaticity — verbatim from plot_per_cloud_rmse_kfold.py
# ═══════════════════════════════════════════════════════════════════════════

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

    A line is fit to LWC(z) over the bottom `fit_fraction` of the sampled depth
    and its z-intercept (LWC = 0) is the cloud base (adiabatic model linear in
    height above cloud base, w_l = Γ_l (z − z_B); Albrecht et al. 1990, GRL 17,
    89–92). The base is never above the lowest level. Fallback when the slope is
    not physical (≤ 0.25 Γ_ad, > 3 Γ_ad, or < 3 levels): z_cb = z_lowest −
    lwc_threshold / Γ_ad. Identical to compute_adiabaticity_Wood2005.m and to
    plot_per_cloud_rmse_kfold.extrapolated_cloud_base (see that file for the
    full discussion).

    Returns (z_cb_m, lwc_at_lowest_g_m3, used_fallback).
    """
    lwc = np.asarray(lwc_g_m3, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    z_lowest_m = float(np.min(z))
    h_sampled_m = float(np.max(z)) - z_lowest_m

    # bottom third; 1e-4 tolerance keeps the float32 level at z* = 1/3
    z_star = (z - z_lowest_m) / h_sampled_m
    in_fit = np.isfinite(lwc) & (z_star <= fit_fraction + 1e-4)

    slope_g_m3_per_m = float('nan')
    lwc_line_at_lowest_g_m3 = float('nan')
    if in_fit.sum() >= 3 and np.std(z[in_fit]) > 0:
        slope_g_m3_per_m, lwc_line_at_lowest_g_m3 = np.polyfit(
            z[in_fit] - z_lowest_m, lwc[in_fit], 1)

    slope_over_gamma_ad = slope_g_m3_per_m / gamma_ad_g_m3_per_m
    lower, upper = slope_limits_over_gamma_ad
    is_physical = bool(lower < slope_over_gamma_ad <= upper)

    if is_physical:
        dz_base_m = lwc_line_at_lowest_g_m3 / slope_g_m3_per_m
        if dz_base_m >= 0:
            return z_lowest_m - dz_base_m, lwc_line_at_lowest_g_m3, False
        return z_lowest_m, 0.0, False

    dz_base_m = lwc_threshold_g_m3 / gamma_ad_g_m3_per_m
    return z_lowest_m - dz_base_m, lwc_threshold_g_m3, True


def wood_adiabaticity_details(lwc_top_to_base_g_m3: np.ndarray,
                              z_top_to_base_km: np.ndarray,
                              era5_T: np.ndarray, era5_P_hPa: np.ndarray,
                              lwc_threshold_g_m3: float = LWC_THRESHOLD_G_M3,
                              ) -> dict:
    """Wood (2005, JAS 62, 3011; sec. 3b) adiabaticity and the quantities behind it.

    Same arithmetic as plot_per_cloud_rmse_kfold.wood_adiabaticity (whose
    return value is f_ad here); this version also returns the cloud base,
    Γ_ad and whether the fallback set the base, for storage in the HDF5.

        f_ad    = LWP_obs / LWP_ad
        LWP_ad  = ½ · Γ_ad · h²,  h = z_top − z_cb
        LWP_obs = ∫ LWC dz over the levels + the unsampled wedge below the lowest

    One Γ_ad per cloud at the trapezoid-mean in-cloud T and P from the ERA5
    column. Returns NaNs when the cloud has zero thickness or Γ_ad ≤ 0.
    """
    nan = dict(f_ad=np.nan, z_cb_m=np.nan, gamma_ad_g_m3_per_m=np.nan,
               lwp_obs_g_m2=np.nan, lwp_ad_g_m2=np.nan, used_fallback=False)
    z_m = np.asarray(z_top_to_base_km, dtype=np.float64) * 1000.0   # m, top → base
    lwc = np.asarray(lwc_top_to_base_g_m3, dtype=np.float64)         # g/m³
    z_top_m = float(np.max(z_m))
    h_sampled_m = z_top_m - float(np.min(z_m))
    if h_sampled_m <= 0:
        return nan

    T_lev, P_lev = era5_T_P_at_heights(era5_T, era5_P_hPa, z_top_to_base_km)
    T_mean = abs(np.trapezoid(T_lev, z_m)) / h_sampled_m
    P_mean = abs(np.trapezoid(P_lev, z_m)) / h_sampled_m
    gamma_ad = gamma_ad_kg_per_m4(T_mean, P_mean)                    # kg/m⁴
    if gamma_ad <= 0:
        return nan
    gamma_ad_g_m3_per_m = gamma_ad * 1000.0                          # g/m³ per m

    z_cb_m, lwc_at_lowest_g_m3, used_fallback = extrapolated_cloud_base(
        lwc, z_m, gamma_ad_g_m3_per_m, lwc_threshold_g_m3)
    dz_base_m = float(np.min(z_m)) - z_cb_m                          # m, ≥ 0
    h_m = z_top_m - z_cb_m

    finite = np.isfinite(lwc)
    lwp_measured_g = abs(np.trapezoid(lwc[finite], z_m[finite]))     # g/m²
    lwp_wedge_g = 0.5 * lwc_at_lowest_g_m3 * dz_base_m               # g/m²
    lwp_ad_g = 0.5 * gamma_ad_g_m3_per_m * h_m ** 2                  # g/m²
    return dict(f_ad=(lwp_measured_g + lwp_wedge_g) / lwp_ad_g, z_cb_m=z_cb_m,
                gamma_ad_g_m3_per_m=gamma_ad_g_m3_per_m,
                lwp_obs_g_m2=lwp_measured_g + lwp_wedge_g, lwp_ad_g_m2=lwp_ad_g,
                used_fallback=used_fallback)


def wood_adiabaticity(lwc_top_to_base_g_m3, z_top_to_base_km, era5_T, era5_P_hPa,
                      lwc_threshold_g_m3: float = LWC_THRESHOLD_G_M3) -> float:
    """f_ad only; drop-in for plot_per_cloud_rmse_kfold.wood_adiabaticity."""
    return wood_adiabaticity_details(lwc_top_to_base_g_m3, z_top_to_base_km,
                                     era5_T, era5_P_hPa, lwc_threshold_g_m3)['f_ad']


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mie extinction — same tables and rules as the τ_c label (06 generator)
# ═══════════════════════════════════════════════════════════════════════════

# Laptop location of the libRadtran custom Mie tables; on Alpine pass the
# /projects (or staged /scratch) copy explicitly.
DEFAULT_MIE_TABLE_DIR = Path('/Users/andrewbuggee/Documents/libRadtran-2.0.6/data/wc/custom_mieTables')
MIE_TABLE_REFF_MAX_UM = 50      # the 1–50 μm table family used for the synthetic RT
TAU_WAVELENGTH_UM     = 0.55    # τ_c is defined at 550 nm


@lru_cache(maxsize=4)
def load_mie_ext_tables(table_dir: str = str(DEFAULT_MIE_TABLE_DIR),
                        reff_max_um: int = MIE_TABLE_REFF_MAX_UM,
                        wavelength_um: float = TAU_WAVELENGTH_UM
                        ) -> tuple[np.ndarray, dict]:
    """ext(r_e) = β/LWC [km⁻¹ per g m⁻³] at one wavelength, for every α table.

    Same reading as 06_synthetic_profile_generator.load_mie_ext_tables: one
    netCDF-4 file per α, variable 'ext' (nlam, nreff), linearly interpolated
    to `wavelength_um` (0.55 μm is a table grid point). libRadtran's gamma
    distribution, n(r) ∝ r^α exp(−(α+3) r / r_e) (Hansen & Travis 1974 eq.
    2.56 with v_eff = 1/(α+3)). Cached, so the tables are read once.

    Returns (alphas (n_tables,), {α: (reff_um (nreff,), ext (nreff,))}).
    """
    import h5py

    pattern = f'wc_mieTable_gamma_rEff_1-{reff_max_um}microns_gammaDist_alpha_*.cdf'
    files = sorted(f for f in Path(table_dir).glob(pattern) if not f.name.startswith('._'))
    if not files:
        raise FileNotFoundError(f'No Mie tables matching {pattern} in {table_dir}')
    tables = {}
    for f in files:
        alpha = int(f.stem.rsplit('_', 1)[-1])
        with h5py.File(f, 'r') as ds:
            wl_um = ds['wavelen'][:].astype(np.float64)
            reff_um = ds['reff'][:].astype(np.float64)
            ext = ds['ext'][:].astype(np.float64)                      # (nlam, nreff)
        ext_wl = np.array([np.interp(wavelength_um, wl_um, ext[:, j])
                           for j in range(reff_um.size)])
        tables[alpha] = (reff_um, ext_wl)
    alphas = np.array(sorted(tables), dtype=np.float64)
    return alphas, tables


def ext_per_lwc(re_um: np.ndarray, alpha: float,
                table_dir: str = str(DEFAULT_MIE_TABLE_DIR)) -> np.ndarray:
    """ext(r_e; α) = 750 · Q̄_ext / r_e [km⁻¹ per g m⁻³] from the table with the
    nearest α (rule of find_custom_mieTable_closest_to_alpha_profile.m), linear
    in r_e. r_e outside the table is clipped to its range (retrieved profiles
    can stray slightly; the label generator refuses instead)."""
    alphas, tables = load_mie_ext_tables(table_dir)
    a = float(alphas[np.argmin(np.abs(alphas - float(alpha)))])
    reff_um, ext = tables[int(a)]
    r = np.clip(np.asarray(re_um, dtype=np.float64), reff_um[0], reff_um[-1])
    return np.interp(r, reff_um, ext)


# ═══════════════════════════════════════════════════════════════════════════
# 3. LWP
# ═══════════════════════════════════════════════════════════════════════════

def lwp_from_lwc_g_m2(lwc_g_m3: np.ndarray, z_km: np.ndarray) -> np.ndarray:
    """Truth LWP [g m⁻²] = 1000 · |∫ LWC dz| (trapezoid; LWC g m⁻³, z km).

    Same definition as the generator's LWP_g_per_m2. Arrays (n, L) or (L,)."""
    lwc = np.atleast_2d(lwc_g_m3).astype(np.float64)
    z = np.atleast_2d(z_km).astype(np.float64)
    return 1000.0 * np.abs(np.trapezoid(lwc, z, axis=1))


def lwp_eq10_q2_g_m2(re_um: np.ndarray, tau_c: np.ndarray) -> np.ndarray:
    """Paper-2 Eq. 10 with Q_ext = 2 (the method used before Nov 2026), for
    comparison only: (2/3) τ_c ∫ r_e³ dz / ∫ r_e² dz [g m⁻², r_e μm], unit
    level spacing (cancels in the ratio). re_um (n, L), tau_c (n,)."""
    r = np.atleast_2d(re_um).astype(np.float64)
    return (2.0 / 3.0) * np.asarray(tau_c, dtype=np.float64) * \
        np.trapezoid(r ** 3, axis=1) / np.trapezoid(r ** 2, axis=1)


def lwp_constant_N_mie(re_um: np.ndarray, tau_c: np.ndarray, alpha,
                       table_dir: str = str(DEFAULT_MIE_TABLE_DIR)) -> np.ndarray:
    """LWP [g m⁻²] from an r_e profile and τ_c, consistent with the Mie τ label:

        LWP = 1000 · τ_c · ∫ r_e³ dz / ∫ ext(r_e; α) r_e³ dz

    (constant N and α with height; see the module docstring for the derivation,
    which is Claude's). re_um (n, L) on evenly spaced levels (any order; the
    spacing cancels), tau_c (n,), alpha scalar or (n,). ext in km⁻¹ per g m⁻³,
    hence the 1000 (km → m)."""
    r = np.atleast_2d(re_um).astype(np.float64)
    tau = np.atleast_1d(np.asarray(tau_c, dtype=np.float64))
    a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), tau.shape)
    out = np.empty(tau.shape)
    for i in range(r.shape[0]):
        ext = ext_per_lwc(r[i], a[i], table_dir)                   # (L,) km⁻¹ per g m⁻³
        out[i] = 1000.0 * tau[i] * np.trapezoid(r[i] ** 3) / np.trapezoid(ext * r[i] ** 3)
    return out


def tau_from_profile_mie(re_um: np.ndarray, lwc_g_m3: np.ndarray, z_km: np.ndarray,
                         alpha: float, table_dir: str = str(DEFAULT_MIE_TABLE_DIR)) -> float:
    """τ_c = |∫ ext(r_e; α) LWC dz| (trapezoid over levels, z km), i.e. the
    06 generator's label. Used to check that the HDF5 τ_c is this quantity."""
    beta_per_km = ext_per_lwc(re_um, alpha, table_dir) * np.asarray(lwc_g_m3, dtype=np.float64)
    return float(abs(np.trapezoid(beta_per_km, np.asarray(z_km, dtype=np.float64))))
