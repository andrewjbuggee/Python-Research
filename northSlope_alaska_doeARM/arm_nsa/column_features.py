"""Per-profile column features from THERMOCLDPHASE for the DLR-microphysics study.

Reduces each 30-s (time, height) profile of the THERMOCLDPHASE VAP
(nsathermocldphaseC1.c0; Zhang et al. 2025, DOE/SC-ARM-TR-325) to a row of
scalar features describing the cloud column that the downwelling longwave
(DLR) at the surface "sees": phase composition, vertical layering of liquid
and ice, cloud-boundary temperatures, radar-moment and lidar-depolarization
summaries, and the MWR liquid water path. The notebook
dlr_cloud_microphysics_barrow.ipynb regresses DLR on these features.

Everything is computed on the VAP's own 30 m grid by integer indexing, not
interpolation, so a feature is exactly what the pixel classification says.

Phase codes (file attribute flag_meanings of cloud_phase_<lidar>):
    0 clear_sky  1 liquid  2 ice  3 mixed_phase  4 drizzle
    5 liquid_drizzle  6 rain  7 snow  8 unknown
Liquid-containing = {1, 3, 4, 5, 6}; ice-containing = {2, 3, 7}. Rain (6) is
counted as liquid precipitation; in an Arctic cold season it is rare. Unknown
(8) pixels are counted separately and flagged, because they can hide either
phase.

Ice water content from reflectivity uses the SHEBA winter power law used
elsewhere in this package, IWC [g m-3] = a Ze^b with Ze in mm^6 m^-3,
a = 0.1, b = 0.63 (Shupe et al. 2005 prefactor, Matrosov 1999 exponent;
config.IWC_PREFACTOR_A / IWC_EXPONENT_B). Radar reflectivity in liquid-only
pixels is summarised separately because, for non-drizzling droplet clouds,
Ze together with the LWP constrains droplet size (Frisch et al. 1995,
J. Atmos. Sci. 52, 2788-2799); that conversion is done in the notebook.

Sign convention: radar_mdv is POSITIVE UPWARD in the file ('positive: up'),
so falling hydrometeors have negative mean Doppler velocity. Features keep
that convention; 'mdv_ice_mean_m_s' more negative = faster-falling ice.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

from . import config
from .qc import apply_qc

# --- phase code groups (see module docstring) --------------------------------
CODE_CLEAR = 0
CODE_LIQUID = 1
CODE_ICE = 2
CODE_MIXED = 3
CODE_DRIZZLE = 4
CODE_LIQUID_DRIZZLE = 5
CODE_RAIN = 6
CODE_SNOW = 7
CODE_UNKNOWN = 8
LIQUID_CONTAINING = (1, 3, 4, 5, 6)
ICE_CONTAINING = (2, 3, 7)
LIQUID_PRECIP = (4, 5, 6)
ICE_PRECIP = (7,)
HYDROMETEOR = (1, 2, 3, 4, 5, 6, 7)

# Column classes written to `column_class`
COLUMN_CLASS = {
    0: "no_data",
    1: "clear",
    2: "ice_only",
    3: "liquid_only",
    4: "mixed_column",  # both liquid- and ice-containing pixels somewhere in the column
    5: "unknown_contaminated",  # unknown pixels present and no other decision possible
}

GATE_DZ_M = 30.0  # THERMOCLDPHASE / ARSCL gate spacing at NSA [m]
LOWEST_GATE_M = 160.0  # first gate [m AGL]
INVERSION_SEARCH_TOP_M = 2000.0  # look for the low-level temperature maximum below this


def _linear_ze(ze_dbz: np.ndarray) -> np.ndarray:
    """dBZ -> mm^6 m^-3."""
    return 10.0 ** (ze_dbz / 10.0)


def _nanmean_where(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise mean of x over True entries of mask (NaN where none)."""
    xm = np.where(mask & np.isfinite(x), x, np.nan)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(xm).sum(axis=1)
        s = np.nansum(xm, axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _nanmax_where(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xm = np.where(mask & np.isfinite(x), x, -np.inf)
    m = xm.max(axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def _nanmin_where(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xm = np.where(mask & np.isfinite(x), x, np.inf)
    m = xm.min(axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True along axis 1, -1 where none."""
    has = mask.any(axis=1)
    idx = mask.argmax(axis=1)
    return np.where(has, idx, -1)


def _last_true(mask: np.ndarray) -> np.ndarray:
    has = mask.any(axis=1)
    idx = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    return np.where(has, idx, -1)


def _take(profile: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """profile[t, idx[t]] with NaN where idx < 0."""
    rows = np.arange(profile.shape[0])
    safe = np.where(idx >= 0, idx, 0)
    vals = profile[rows, safe]
    return np.where(idx >= 0, vals, np.nan)


def column_features_from_file(
    path: Path,
    lidar: str = "mplgr",
    iwc_a: float = config.IWC_PREFACTOR_A,
    iwc_b: float = config.IWC_EXPONENT_B,
    apply_qc_flags: bool = True,
) -> xr.Dataset:
    """Reduce one THERMOCLDPHASE day to per-profile column features.

    Parameters
    ----------
    path:
        One nsathermocldphaseC1.c0 daily file.
    lidar:
        "mplgr" (MPL gradient method; available for the whole record) or
        "hsrl" (only where the HSRL ran).
    iwc_a, iwc_b:
        IWC = a Ze^b coefficients for the ice-water-path proxy.
    apply_qc_flags:
        NaN-out phase pixels and LWP samples whose qc_ variable has a Bad bit.

    Returns
    -------
    Dataset over `time` (30 s) with scalar features; see the `long_name`
    attributes and the module docstring.
    """
    ds = xr.open_dataset(path)
    phase_var = f"cloud_phase_{lidar}"
    layer_var = f"cloud_phase_layer_{lidar}"

    height_m = ds["height"].values.astype(float) * 1000.0  # km -> m AGL
    nt, nh = ds.sizes["time"], ds.sizes["height"]

    # ---- phase mask -----------------------------------------------------------
    phase_da = apply_qc(ds, phase_var) if apply_qc_flags else ds[phase_var]
    phase = phase_da.values.astype(float)  # NaN where missing
    qc_phase = ds[f"qc_{phase_var}"].values.astype(int) if f"qc_{phase_var}" in ds else np.zeros(nt, int)
    valid = np.isfinite(phase)
    ph = np.where(valid, phase, -1).astype(np.int8)

    is_cloud = np.isin(ph, HYDROMETEOR)
    is_liq = np.isin(ph, LIQUID_CONTAINING)
    is_ice = np.isin(ph, ICE_CONTAINING)
    is_liq_only_px = ph == CODE_LIQUID
    is_ice_only_px = np.isin(ph, (CODE_ICE, CODE_SNOW))
    is_mixed_px = ph == CODE_MIXED
    is_liq_precip = np.isin(ph, LIQUID_PRECIP)
    is_snow = ph == CODE_SNOW
    is_unknown = ph == CODE_UNKNOWN

    n_valid = valid.sum(axis=1)
    n_cloud = is_cloud.sum(axis=1)
    n_liq = is_liq.sum(axis=1)
    n_ice = is_ice.sum(axis=1)
    n_liq_only = is_liq_only_px.sum(axis=1)
    n_ice_only = is_ice_only_px.sum(axis=1)
    n_mixed = is_mixed_px.sum(axis=1)
    n_liq_precip = is_liq_precip.sum(axis=1)
    n_snow = is_snow.sum(axis=1)
    n_unknown = is_unknown.sum(axis=1)

    # ---- column class ----------------------------------------------------------
    cls = np.zeros(nt, dtype=np.int8)
    has_valid = n_valid > 0
    cls[has_valid] = 5  # provisional: unknown_contaminated
    cls[has_valid & (n_cloud == 0) & (n_unknown == 0)] = 1
    cls[(n_cloud > 0) & (n_ice > 0) & (n_liq == 0)] = 2
    cls[(n_cloud > 0) & (n_liq > 0) & (n_ice == 0)] = 3
    cls[(n_liq > 0) & (n_ice > 0)] = 4
    # A column that has hydrometeors plus unknown pixels keeps its class but is
    # flagged, so the notebook can decide how strict to be.
    unknown_flag = (n_unknown > 0).astype(np.int8)

    with np.errstate(invalid="ignore", divide="ignore"):
        frc_ice = np.where(n_cloud > 0, n_ice / np.maximum(n_cloud, 1), np.nan)

    # ---- geometry (m AGL) ------------------------------------------------------
    i_cb = _first_true(is_cloud)
    i_ct = _last_true(is_cloud)
    i_lb = _first_true(is_liq)
    i_lt = _last_true(is_liq)
    i_ib = _first_true(is_ice)
    i_it = _last_true(is_ice)
    h_of = lambda idx: np.where(idx >= 0, height_m[np.maximum(idx, 0)], np.nan)  # noqa: E731
    cloud_base_m, cloud_top_m = h_of(i_cb), h_of(i_ct)
    liq_base_m, liq_top_m = h_of(i_lb), h_of(i_lt)
    ice_base_m, ice_top_m = h_of(i_ib), h_of(i_it)

    # ice below / above the liquid: precipitating (seeded) vs seeder-from-above
    rows = np.arange(nt)
    col_idx = np.arange(nh)[None, :]
    below_liq = col_idx < i_lb[:, None]
    above_liq = col_idx > i_lt[:, None]
    n_ice_below_liq = (is_ice & below_liq & (i_lb[:, None] >= 0)).sum(axis=1)
    n_ice_above_liq = (is_ice & above_liq & (i_lt[:, None] >= 0)).sum(axis=1)
    # ice inside the liquid span (between liq base and liq top): embedded ice
    within_liq = (col_idx >= i_lb[:, None]) & (col_idx <= i_lt[:, None])
    n_ice_within_liq = (is_ice & within_liq & (i_lb[:, None] >= 0)).sum(axis=1)
    # is there a clear gap between the liquid top and ice above (separate layer)?
    # -> handled through the ARSCL layer structure below.

    # ---- ARSCL layer structure --------------------------------------------------
    clh = ds["cloud_layer_heights"].values * 1000.0  # (time, layer, bound) m
    layer_base = clh[:, :, 0]
    layer_top = clh[:, :, 1]
    has_layer = np.isfinite(layer_base)
    n_layers = has_layer.sum(axis=1).astype(np.int8)
    layer_phase = ds[layer_var].values.astype(float)  # 0 clear 1 liq 2 ice 3 mixed
    lp = np.where(np.isfinite(layer_phase), layer_phase, -1).astype(np.int8)
    n_liq_layers = ((lp == 1) & has_layer).sum(axis=1).astype(np.int8)
    n_ice_layers = ((lp == 2) & has_layer).sum(axis=1).astype(np.int8)
    n_mixed_layers = ((lp == 3) & has_layer).sum(axis=1).astype(np.int8)
    layer1_base_m = np.where(has_layer[:, 0], layer_base[:, 0], np.nan)
    layer1_top_m = np.where(has_layer[:, 0], layer_top[:, 0], np.nan)
    layer2_base_m = np.where(has_layer[:, 1], layer_base[:, 1], np.nan) if clh.shape[1] > 1 else np.full(nt, np.nan)
    with np.errstate(invalid="ignore"):
        top_max_m = np.where(n_layers > 0, np.nanmax(np.where(has_layer, layer_top, -np.inf), axis=1), np.nan)
    layer1_phase = lp[:, 0]
    layer2_phase = lp[:, 1] if clh.shape[1] > 1 else np.full(nt, -1, np.int8)
    # gap between the lowest layer top and the next layer base (NaN if < 2 layers)
    gap12_m = layer2_base_m - layer1_top_m
    # which ARSCL layer holds the liquid top?  liquid_topped = liquid top within
    # two gates of that layer's top (the classic Arctic liquid-topped mixed cloud)
    lt_m = liq_top_m
    in_layer = has_layer & (layer_base <= lt_m[:, None] + 1e-6) & (layer_top >= lt_m[:, None] - 1e-6)
    owner_top = np.where(in_layer.any(axis=1), np.nanmax(np.where(in_layer, layer_top, -np.inf), axis=1), np.nan)
    liquid_topped = np.where(np.isfinite(lt_m) & np.isfinite(owner_top), (owner_top - lt_m) <= 2 * GATE_DZ_M + 1e-6, False)

    # ---- temperatures (degC -> K) at the boundaries -----------------------------
    t_prof = ds["sonde_temp"].values.astype(float) + 273.15  # (time, height) K
    rh_prof = ds["sonde_rh"].values.astype(float)
    t_cloud_base_K = _take(t_prof, i_cb)
    t_cloud_top_K = _take(t_prof, i_ct)
    t_liq_base_K = _take(t_prof, i_lb)
    t_liq_top_K = _take(t_prof, i_lt)
    t_ice_top_K = _take(t_prof, i_it)
    # layer-1 boundaries and highest top via nearest gate index
    def _idx_from_height(h_m: np.ndarray) -> np.ndarray:
        i = np.rint((h_m - LOWEST_GATE_M) / GATE_DZ_M)
        i = np.where(np.isfinite(i), np.clip(i, 0, nh - 1), -1).astype(int)
        return i
    t_layer1_base_K = _take(t_prof, _idx_from_height(layer1_base_m))
    t_layer1_top_K = _take(t_prof, _idx_from_height(layer1_top_m))
    t_top_max_K = _take(t_prof, _idx_from_height(top_max_m))
    t_sonde_sfc_K = t_prof[:, 0]  # lowest gate, 160 m AGL
    # low-level inversion: warmest level below 2 km relative to the lowest gate
    k_top = int(np.searchsorted(height_m, INVERSION_SEARCH_TOP_M))
    low = t_prof[:, :k_top]
    with np.errstate(invalid="ignore"):
        t_inv_max_K = np.nanmax(low, axis=1)
        i_inv = np.nanargmax(np.where(np.isfinite(low), low, -np.inf), axis=1)
    z_inv_max_m = height_m[i_inv]
    inversion_K = t_inv_max_K - t_sonde_sfc_K
    rh_liq_mean_pct = _nanmean_where(rh_prof, is_liq)

    # ---- radar moments ----------------------------------------------------------
    ze = apply_qc(ds, "radar_ze").values.astype(float) if apply_qc_flags else ds["radar_ze"].values.astype(float)
    mdv = apply_qc(ds, "radar_mdv").values.astype(float) if apply_qc_flags else ds["radar_mdv"].values.astype(float)
    sw = apply_qc(ds, "radar_w").values.astype(float) if apply_qc_flags else ds["radar_w"].values.astype(float)
    ldr = apply_qc(ds, "radar_ldr").values.astype(float) if apply_qc_flags else ds["radar_ldr"].values.astype(float)
    ze_lin = np.where(np.isfinite(ze), _linear_ze(np.where(np.isfinite(ze), ze, 0.0)), np.nan)
    with np.errstate(invalid="ignore"):
        ze_max_dbz = _nanmax_where(ze, is_cloud)
        ze_cloud_base_dbz = _take(ze, i_cb)
        ze_liq_max_dbz = _nanmax_where(ze, is_liq_only_px)
        ze_liq_lin_mean = _nanmean_where(ze_lin, is_liq_only_px)
        ze_liq_mean_dbz = np.where(ze_liq_lin_mean > 0, 10 * np.log10(ze_liq_lin_mean), np.nan)
        n_liq_only_radar = (is_liq_only_px & np.isfinite(ze)).sum(axis=1)
        mdv_liq_mean = _nanmean_where(mdv, is_liq_only_px)
        mdv_ice_mean = _nanmean_where(mdv, is_ice)
        mdv_ice_min = _nanmin_where(mdv, is_ice)  # most negative = fastest fall
        mdv_below_liq_mean = _nanmean_where(mdv, is_ice & below_liq & (i_lb[:, None] >= 0))
        sw_ice_mean = _nanmean_where(sw, is_ice)
        sw_liq_mean = _nanmean_where(sw, is_liq_only_px)
        ldr_ice_mean_db = _nanmean_where(ldr, is_ice)
        ldr_liq_mean_db = _nanmean_where(ldr, is_liq_only_px)
        # Ze-based IWP over ice-containing pixels (mixed pixels included: their
        # reflectivity is dominated by the ice)
        iwc = iwc_a * np.where(np.isfinite(ze_lin), ze_lin, 0.0) ** iwc_b
        iwp_proxy = np.where(n_ice > 0, (np.where(is_ice & np.isfinite(ze), iwc, 0.0) * GATE_DZ_M).sum(axis=1), np.where(n_valid > 0, 0.0, np.nan))
        iwp_below_liq = np.where(i_lb >= 0, (np.where(is_ice & below_liq & np.isfinite(ze), iwc, 0.0) * GATE_DZ_M).sum(axis=1), np.nan)
        iwp_above_liq = np.where(i_lt >= 0, (np.where(is_ice & above_liq & np.isfinite(ze), iwc, 0.0) * GATE_DZ_M).sum(axis=1), np.nan)
        n_ice_radar = (is_ice & np.isfinite(ze)).sum(axis=1)

    # ---- lidar depolarisation ---------------------------------------------------
    mpl_ldr = apply_qc(ds, "mpl_ldr").values.astype(float) if ("mpl_ldr" in ds and apply_qc_flags) else (ds["mpl_ldr"].values.astype(float) if "mpl_ldr" in ds else np.full((nt, nh), np.nan))
    mpl_ldr_liq_mean = _nanmean_where(mpl_ldr, is_liq)
    mpl_ldr_ice_mean = _nanmean_where(mpl_ldr, is_ice)
    mpl_ldr_cloud_base = _take(mpl_ldr, i_cb)

    # ---- MWR ----------------------------------------------------------------------
    lwp = apply_qc(ds, "mwr_lwp_be").values.astype(float) if apply_qc_flags else ds["mwr_lwp_be"].values.astype(float)
    pwv = apply_qc(ds, "mwr_pwv_be").values.astype(float) if ("mwr_pwv_be" in ds and apply_qc_flags) else ds["mwr_pwv_be"].values.astype(float)

    out = xr.Dataset(coords={"time": ds["time"].values})
    def put(name, arr, units, long_name, dtype=None):
        a = np.asarray(arr)
        if dtype is not None:
            a = a.astype(dtype)
        elif a.dtype.kind == "f":
            a = a.astype(np.float32)
        out[name] = ("time", a)
        out[name].attrs = {"units": units, "long_name": long_name}

    put("column_class", cls, "1", "0 no_data 1 clear 2 ice_only 3 liquid_only 4 mixed_column 5 unknown_contaminated", np.int8)
    put("unknown_flag", unknown_flag, "1", "1 if any unknown-phase pixel in the column", np.int8)
    put("qc_phase", qc_phase.max(axis=1) if qc_phase.ndim == 2 else qc_phase, "1", f"max qc_{phase_var} bit value over the column", np.int16)
    put("n_valid", n_valid, "1", "valid phase pixels", np.int16)
    put("n_cloud", n_cloud, "1", "hydrometeor pixels (codes 1-7)", np.int16)
    put("n_liq", n_liq, "1", "liquid-containing pixels {1,3,4,5,6}", np.int16)
    put("n_ice", n_ice, "1", "ice-containing pixels {2,3,7}", np.int16)
    put("n_liq_only_px", n_liq_only, "1", "pure liquid pixels (code 1)", np.int16)
    put("n_ice_only_px", n_ice_only, "1", "ice or snow pixels (2,7)", np.int16)
    put("n_mixed_px", n_mixed, "1", "mixed-phase pixels (3)", np.int16)
    put("n_liq_precip_px", n_liq_precip, "1", "drizzle / liquid+drizzle / rain pixels (4,5,6)", np.int16)
    put("n_snow_px", n_snow, "1", "snow pixels (7)", np.int16)
    put("n_unknown_px", n_unknown, "1", "unknown pixels (8)", np.int16)
    put("frc_ice", frc_ice, "1", "ice-containing / hydrometeor pixels")
    put("cloud_base_m", cloud_base_m, "m", "lowest hydrometeor pixel [m AGL]")
    put("cloud_top_m", cloud_top_m, "m", "highest hydrometeor pixel [m AGL]")
    put("liq_base_m", liq_base_m, "m", "lowest liquid-containing pixel [m AGL]")
    put("liq_top_m", liq_top_m, "m", "highest liquid-containing pixel [m AGL]")
    put("liq_depth_m", np.where(n_liq > 0, n_liq * GATE_DZ_M, np.nan), "m", "liquid-containing pixels x 30 m")
    put("ice_base_m", ice_base_m, "m", "lowest ice-containing pixel [m AGL]")
    put("ice_top_m", ice_top_m, "m", "highest ice-containing pixel [m AGL]")
    put("n_ice_below_liq", n_ice_below_liq, "1", "ice-containing pixels below the liquid base (precipitating ice)", np.int16)
    put("n_ice_above_liq", n_ice_above_liq, "1", "ice-containing pixels above the liquid top", np.int16)
    put("n_ice_within_liq", n_ice_within_liq, "1", "ice-containing pixels between liquid base and top", np.int16)
    put("liquid_topped", liquid_topped, "1", "1 if the liquid top is within 2 gates of its ARSCL layer top", np.int8)
    put("n_layers", n_layers, "1", "ARSCL hydrometeor layers", np.int8)
    put("n_liq_layers", n_liq_layers, "1", "layers with VAP layer phase = liquid", np.int8)
    put("n_ice_layers", n_ice_layers, "1", "layers with VAP layer phase = ice", np.int8)
    put("n_mixed_layers", n_mixed_layers, "1", "layers with VAP layer phase = mixed", np.int8)
    put("layer1_phase", layer1_phase, "1", "VAP phase of the lowest layer: 0 clear 1 liquid 2 ice 3 mixed, -1 none", np.int8)
    put("layer2_phase", layer2_phase, "1", "VAP phase of the second layer", np.int8)
    put("layer1_base_m", layer1_base_m, "m", "lowest ARSCL layer base [m AGL]")
    put("layer1_top_m", layer1_top_m, "m", "lowest ARSCL layer top [m AGL]")
    put("layer2_base_m", layer2_base_m, "m", "second ARSCL layer base [m AGL]")
    put("gap12_m", gap12_m, "m", "clear gap between layer 1 top and layer 2 base")
    put("top_max_m", top_max_m, "m", "highest ARSCL layer top [m AGL]")
    put("t_cloud_base_K", t_cloud_base_K, "K", "sonde temperature at the lowest hydrometeor pixel")
    put("t_cloud_top_K", t_cloud_top_K, "K", "sonde temperature at the highest hydrometeor pixel")
    put("t_liq_base_K", t_liq_base_K, "K", "sonde temperature at the liquid base")
    put("t_liq_top_K", t_liq_top_K, "K", "sonde temperature at the liquid top")
    put("t_ice_top_K", t_ice_top_K, "K", "sonde temperature at the ice top")
    put("t_layer1_base_K", t_layer1_base_K, "K", "sonde temperature at the lowest ARSCL layer base")
    put("t_layer1_top_K", t_layer1_top_K, "K", "sonde temperature at the lowest ARSCL layer top")
    put("t_top_max_K", t_top_max_K, "K", "sonde temperature at the highest layer top")
    put("t_sonde_160m_K", t_sonde_sfc_K, "K", "interpolated-sonde temperature at the lowest gate (160 m AGL)")
    put("t_inv_max_K", t_inv_max_K, "K", "warmest sonde level below 2 km")
    put("z_inv_max_m", z_inv_max_m, "m", "height of the warmest level below 2 km")
    put("inversion_K", inversion_K, "K", "t_inv_max_K - t_sonde_160m_K (low-level inversion strength)")
    put("rh_liq_mean_pct", rh_liq_mean_pct, "%", "sonde RH (w.r.t. liquid) averaged over liquid pixels")
    put("ze_max_dbz", ze_max_dbz, "dBZ", "max reflectivity over hydrometeor pixels")
    put("ze_cloud_base_dbz", ze_cloud_base_dbz, "dBZ", "reflectivity at the lowest hydrometeor pixel")
    put("ze_liq_max_dbz", ze_liq_max_dbz, "dBZ", "max reflectivity over pure-liquid pixels")
    put("ze_liq_mean_dbz", ze_liq_mean_dbz, "dBZ", "linear-mean reflectivity over pure-liquid pixels")
    put("n_liq_only_radar", n_liq_only_radar, "1", "pure-liquid pixels with a radar echo", np.int16)
    put("mdv_liq_mean_m_s", mdv_liq_mean, "m s-1", "mean Doppler velocity (positive up) over pure-liquid pixels")
    put("mdv_ice_mean_m_s", mdv_ice_mean, "m s-1", "mean Doppler velocity (positive up) over ice-containing pixels")
    put("mdv_ice_min_m_s", mdv_ice_min, "m s-1", "most negative Doppler velocity over ice-containing pixels")
    put("mdv_below_liq_mean_m_s", mdv_below_liq_mean, "m s-1", "mean Doppler velocity of ice below the liquid base")
    put("sw_ice_mean_m_s", sw_ice_mean, "m s-1", "mean spectral width over ice-containing pixels")
    put("sw_liq_mean_m_s", sw_liq_mean, "m s-1", "mean spectral width over pure-liquid pixels")
    put("ldr_ice_mean_db", ldr_ice_mean_db, "dB", "mean radar LDR over ice-containing pixels")
    put("ldr_liq_mean_db", ldr_liq_mean_db, "dB", "mean radar LDR over pure-liquid pixels")
    put("iwp_proxy_g_m2", iwp_proxy, "g m-2", f"IWP from IWC = {iwc_a} Ze^{iwc_b} over ice-containing pixels")
    put("iwp_below_liq_g_m2", iwp_below_liq, "g m-2", "Ze-based IWP of ice below the liquid base")
    put("iwp_above_liq_g_m2", iwp_above_liq, "g m-2", "Ze-based IWP of ice above the liquid top")
    put("n_ice_radar", n_ice_radar, "1", "ice-containing pixels with a radar echo", np.int16)
    put("mpl_ldr_liq_mean", mpl_ldr_liq_mean, "1", "mean MPL linear depolarisation ratio over liquid-containing pixels")
    put("mpl_ldr_ice_mean", mpl_ldr_ice_mean, "1", "mean MPL linear depolarisation ratio over ice-containing pixels")
    put("mpl_ldr_cloud_base", mpl_ldr_cloud_base, "1", "MPL LDR at the lowest hydrometeor pixel")
    put("lwp_g_m2", lwp, "g m-2", "MWR best-estimate LWP (thermocldphase mwr_lwp_be, Bad-QC removed)")
    put("pwv_cm", pwv, "cm", "MWR best-estimate PWV")

    out.attrs = {
        "title": "THERMOCLDPHASE column features (arm_nsa.column_features)",
        "source_file": Path(path).name,
        "lidar": lidar,
        "source_dod_version": str(ds.attrs.get("dod_version", "")),
        "source_process_version": str(ds.attrs.get("process_version", "")),
        "iwc_relation": f"IWC[g m-3] = {iwc_a} * Ze[mm6 m-3]^{iwc_b}",
        "mdv_sign": "positive upward (file convention)",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    ds.close()
    return out


def build_column_files(
    start_date: str,
    end_date: str,
    out_dir: Optional[Path] = None,
    lidar: str = "mplgr",
    verbose: bool = True,
) -> Dict[str, int]:
    """Run column_features_from_file over every local thermocldphase day in range.

    Writes one file per day to data/processed/dlr_columns/<lidar>/ and skips
    days already done, so it can be re-run as the raw download progresses.
    Returns counts {"done", "skipped", "missing"}.
    """
    from .readers import files_in_range

    out_dir = Path(out_dir) if out_dir else config.PROCESSED_DATA_DIR / "dlr_columns" / lidar
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = files_in_range("nsathermocldphaseC1.c0", start_date, end_date)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for p in paths:
        stamp = p.name.split(".")[2]
        target = out_dir / f"dlr_columns.{stamp}.nc"
        if target.exists():
            counts["skipped"] += 1
            continue
        try:
            feats = column_features_from_file(p, lidar=lidar)
            feats.to_netcdf(target)
            counts["done"] += 1
            if verbose:
                print(f"{stamp}: {feats.sizes['time']} profiles -> {target.name}", flush=True)
        except Exception as err:  # noqa: BLE001 -- one bad day must not kill a season
            counts["failed"] += 1
            print(f"{stamp}: FAILED {type(err).__name__}: {err}", flush=True)
    return counts
